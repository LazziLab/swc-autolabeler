"""Train the Stage 3 branch-correction models (one per seed, leak-free).

For each seed S the correction is trained on the pyramidal files NOT in seed S's
hash-split test set -- the same split the seed's pipeline models use -- so the
multi-seed evaluation stays leak-free. Writes
``pipeline/models/seed{S}/branch_correction.joblib``. The seed-123 model is the
one the shipped labeler uses.

The 3-class (axon/basal/apical) XGBoost is trained on the label-free branch
features from ``pipeline.branch_features`` with inverse-frequency class weights
(basal dominates by node count; apical/axon are upweighted). By default it is
trained on, and at inference applied to, only the branches whose midpoint lies
within ``DEFAULT_SCOPE_UM`` of the soma by path distance: the proximal branches
where local geometry separates the classes and where the earlier stages err.
The scope is stored in the bundle (``scope_root_path_max_um``) and honored by
``branch_correction.apply_correction``; ``--scope-um`` overrides it.

Usage:
    python -m pipeline.stage3.train_branch_correction
    python -m pipeline.stage3.train_branch_correction --data-dir other/pyramidal --out-root other_models

``--out-root`` holds one ``seed{S}/`` folder per seed (default ``pipeline/models``).
Pyramidal files other than the bundled ones do not replace the released corrections
there unless ``--overwrite`` is passed.
"""
from __future__ import annotations

import argparse

import sys
import time
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.features import parse_swc                         # noqa: E402
from pipeline.branch_features import extract_branches, BRANCH_FEATURE_NAMES  # noqa: E402
from pipeline.split import in_test
from pipeline.released_models import refuse_replacing_released

DATA = ROOT / "data" / "pyramidal"
MODELS = ROOT / "pipeline" / "models"
SEEDS = (42, 123, 789)
T2C = {2: 0, 3: 1, 4: 2}
# The correction's scope and weighting, shared by every trainer entry point.
# scope: path distance from the soma to a branch's midpoint, in microns, within which
# branches are trained on and overridden (None = every branch). weight mode:
# "branch" gives each class equal total weight over branches, "node" weights each
# branch by its node count before equalizing the classes.
DEFAULT_SCOPE_UM: float | None = 150.0
DEFAULT_WEIGHT_MODE = "branch"


def build_features(pyramidal_dir: Path) -> list[dict]:
    """One record per pyramidal file: its name, its branch features and their
    ground-truth labels.

    Extracted fresh every time: a pre-extracted pickle would save the half hour
    this takes but carries no record of the feature set it was built from, so a
    stale one could train the correction on the wrong columns.
    """
    files = sorted(Path(pyramidal_dir).glob("*.swc"))
    print(f"extracting branches from {len(files)} pyramidal files...", flush=True)
    out, t0 = [], time.perf_counter()
    for i, p in enumerate(files, 1):
        nodes = parse_swc(p)
        if not nodes:
            continue
        mb = extract_branches(nodes, cell_type="pyramidal", file_path=str(p))
        if not mb.branches:
            continue
        out.append({"file": p.name,
                    "feats": np.stack([br.features for br in mb.branches], 0).astype(np.float32),
                    "branch_gt": np.asarray([br.gt_label for br in mb.branches], dtype=np.int8),
                    "n_nodes": np.asarray([len(br.node_indices) for br in mb.branches], dtype=np.int32)})
        if i % 1000 == 0:
            print(f"  {i}/{len(files)} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    return out


def train_correction_for_seed(pyramidal_dir: Path, seed: int, out_path: Path, *,
                              test_size: float = 0.20, n_estimators: int = 250,
                              recs: list[dict] | None = None,
                              scope_um: float | None = DEFAULT_SCOPE_UM,
                              weight_mode: str = DEFAULT_WEIGHT_MODE) -> int:
    """Train one seed's branch-correction head and save it to ``out_path``.

    The head is fit on the pyramidal branches of files NOT in ``seed``'s
    hash-split test set, so it stays leak-free w.r.t. that seed's evaluation.
    With ``scope_um`` set, only branches whose midpoint lies within that path
    distance of the soma are trained on, and the saved bundle tells
    ``apply_correction`` to override only those. ``weight_mode`` is "branch"
    (each class the same total weight over branches) or "node" (each branch
    weighted by its node count before the classes are equalized). Pass a
    prebuilt ``recs`` (a ``build_features`` result) to avoid re-extracting when
    training several seeds from the same corpus; otherwise features are
    extracted from ``pyramidal_dir``. Returns the number of branches trained on.

    The classifier's random state is 0 whatever seed is being built: the seed
    chooses which files are held out, and that is the whole of what separates
    the three released corrections. Supplementary Section S2.1 of the paper reports it so.
    """
    from xgboost import XGBClassifier
    if recs is None:
        recs = build_features(pyramidal_dir)
    feats = np.concatenate([r["feats"] for r in recs], 0)
    gt = np.concatenate([r["branch_gt"] for r in recs], 0).astype(int)
    file_of = np.concatenate([[i] * len(r["branch_gt"]) for i, r in enumerate(recs)])
    files = [r["file"] for r in recs]
    is_neu = np.isin(gt, [2, 3, 4])
    ycls = np.array([T2C.get(int(g), -1) for g in gt])

    train_file = np.array([not in_test(f, seed, test_size) for f in files])
    tr = train_file[file_of] & is_neu
    if scope_um is not None:
        dist_i = list(BRANCH_FEATURE_NAMES).index("root_path_dist")
        tr &= feats[:, dist_i] <= float(scope_um)
    ytr = ycls[tr]
    if weight_mode == "node":
        if any("n_nodes" not in r for r in recs):
            raise ValueError("node weighting needs per-branch node counts; rebuild the branch features")
        n_nodes = np.concatenate([r["n_nodes"] for r in recs], 0).astype(float)[tr]
        mass = np.array([n_nodes[ytr == c].sum() for c in range(3)])
        sw = n_nodes * (mass.sum() / (3 * np.clip(mass, 1.0, None)))[ytr]
    elif weight_mode == "branch":
        counts = np.bincount(ytr, minlength=3).astype(float)
        sw = (counts.sum() / (3 * np.clip(counts, 1, None)))[ytr]
    else:
        raise ValueError(f"unknown weight_mode {weight_mode!r}")
    clf = XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        eval_metric="mlogloss", random_state=0, num_class=3,
                        objective="multi:softprob")
    clf.fit(feats[tr], ytr, sample_weight=sw)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": list(BRANCH_FEATURE_NAMES),
                 "class_map": {0: 2, 1: 3, 2: 4}, "override_tau": 0.65,
                 "class_weighted": True, "weight_mode": weight_mode,
                 "scope_root_path_max_um": scope_um, "held_out_seed": seed},
                out_path, compress=3)
    return int(tr.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA,
                    help=f"directory of pyramidal .swc files (default {DATA})")
    ap.add_argument("--n-estimators", type=int, default=250)
    ap.add_argument("--scope-um", type=float, default=DEFAULT_SCOPE_UM,
                    help="train on and override only branches whose midpoint lies within this path distance of the soma")
    ap.add_argument("--weight-mode", choices=("branch", "node"), default=DEFAULT_WEIGHT_MODE)
    ap.add_argument("--out-root", type=Path, default=MODELS,
                    help=f"folder holding one seed{{S}}/ folder per seed (default {MODELS})")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow pyramidal files other than the bundled ones to replace the released corrections")
    args = ap.parse_args()

    for S in SEEDS:
        refuse_replacing_released(args.out_root / f"seed{S}", args.data_dir, args.overwrite,
                                  out_option="--out-root")
    recs = build_features(args.data_dir)
    for S in SEEDS:
        t0 = time.perf_counter()
        n = train_correction_for_seed(args.data_dir, S,
                                      args.out_root / f"seed{S}" / "branch_correction.joblib",
                                      n_estimators=args.n_estimators, recs=recs,
                                      scope_um=args.scope_um, weight_mode=args.weight_mode)
        print(f"seed {S}: trained on {n} branches "
              f"({(time.perf_counter()-t0)/60:.1f} min) -> seed{S}/branch_correction.joblib", flush=True)
    print("BRANCH CORRECTION TRAINING DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
