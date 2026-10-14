"""Single-seed Stage 4 quality flag: train it, and apply it at inference.

The flag is a per-cell meta-model over the pipeline's own outputs. It predicts a
labeled cell's neurite F1 and ranks cells by that prediction, so a curator can
review the riskiest ones. It never relabels -- it only scores cells for triage.
There is one flag per cell type, pyramidal and interneuron, and a cell is scored
by the flag of the type Stage 1 assigns it.

TRAIN (called by ``pipeline.train_pipeline`` as Stage 4):
  run the trained pipeline over the seed's 20% HOLDOUT, assemble the flag
  features (confidence, predicted geometry, and the geometry/anatomy block)
  and the target (per-cell neurite F1), and fit one flag on the held-out cells
  of each reference cell type.
  Leak-free: the labeler trains on the 80%, the flag learns only from its
  held-out-cell errors. Ships ``flag_model.joblib`` into the model dir.

APPLY (called by ``pipeline.infer_pipeline`` when that file is present):
  assemble the same features from the cell's pipeline result and return a
  suspicion score in [0, 1] from the flag of the cell's Stage 1 type.

It uses the same estimator and target as the multi-seed flag reported in the
paper. Its features are the numeric columns one labeling pass assembles, a
subset of the multi-seed table's: the table adds count and Stage 1 summary
columns and Stage 1's predicted type. evaluation.stage4.eval_single_seed_flag
measures this flag on exactly its columns.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                 # noqa: E402
from pipeline.infer_pipeline import run_pipeline_on_nodes               # noqa: E402
from pipeline.stage4.flag_features import assemble_flag_features        # noqa: E402
from pipeline.stage4.flag_model import flag_regressor                   # noqa: E402
from pipeline.stage4.flag_metrics import per_cell_neurite_f1            # noqa: E402

CELL_TYPES = ("pyramidal", "interneuron")
BAD_THRESHOLD = 0.60          # reporting definition of a bad cell, not the target
REJECT_RATES = (0.05, 0.10, 0.15)
META_PREFIX = "_"            # bookkeeping keys on a feature row, never features

from pipeline.split import in_test as _in_test


def _risk_from_f1(pred_f1):
    """Suspicion in [0,1] from a predicted per-cell F1 -- the same convention as
    ``pipeline.stage4.flag_model.risk_scores`` uses for the multi-seed flag."""
    return np.clip(1.0 - np.asarray(pred_f1, dtype=float), 0.0, 1.0)


# --------------------------------------------------------------------------- train

def _fit_one(rows: list[dict]) -> dict:
    """Fit one cell type's flag on its feature rows; returns that type's bundle."""
    import pandas as pd
    from sklearn.impute import SimpleImputer

    df = pd.DataFrame(rows)
    meta_cols = [c for c in df.columns if str(c).startswith(META_PREFIX)]
    feat_df = df.drop(columns=meta_cols).apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    cols = list(feat_df.columns)
    X = feat_df.to_numpy()
    f1 = df["_f1"].to_numpy(float)
    y = (f1 < BAD_THRESHOLD).astype(int)      # reporting label only, not the target

    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)
    # The one flag model, shared with the multi-seed evaluation and the deployed
    # bundle, so a flag trained here has the same form as the published one.
    flag = flag_regressor(0)
    flag.fit(Xi, f1)
    scores = _risk_from_f1(flag.predict(Xi))
    thresholds = {f"{r:.2f}": float(np.quantile(scores, 1.0 - r)) for r in REJECT_RATES}
    return {"model": flag, "imputer": imp, "feature_cols": cols, "thresholds": thresholds,
            "n_holdout": int(len(df)), "bad_rate": float(y.mean()), "n_bad": int(y.sum())}


def train_flag_for_seed(model_dir: Path, data_dir: Path, seed: int, *,
                        test_size: float = 0.20,
                        limit: int = 0, progress_every: int = 200) -> dict:
    """Train the single-seed flag, one per cell type, and save it into ``model_dir``.

    Requires the seed's labeler already trained in ``model_dir`` (Stages 1-3).
    ``data_dir`` holds pyramidal/ + interneuron/ subfolders. Returns a summary."""
    model_dir = Path(model_dir); data_dir = Path(data_dir)
    import joblib

    # 80/20 split by the same hash rule the labeler used.
    train_files: dict[str, list[Path]] = {ct: [] for ct in CELL_TYPES}
    holdout: list[tuple[str, Path]] = []
    for ct in CELL_TYPES:
        for p in sorted((data_dir / ct).glob("*.swc")):
            (holdout.append((ct, p)) if _in_test(p.name, seed, test_size) else train_files[ct].append(p))
    import random
    random.Random(0).shuffle(holdout)
    if limit:
        holdout = holdout[:limit]
    print(f"    flag: {sum(len(v) for v in train_files.values())} train, {len(holdout)} holdout cells", flush=True)

    s1, s2 = model_dir / "cell_type_classifier.pkl", model_dir / "branch_classifier.pkl"

    rows: dict[str, list[dict]] = {ct: [] for ct in CELL_TYPES}
    t0 = time.perf_counter()
    for i, (ct, p) in enumerate(holdout, 1):
        nodes = parse_swc(p)
        if not nodes:
            continue
        gt = [n.type for n in nodes]
        pr = run_pipeline_on_nodes(nodes, "", s1, s2)
        labels = list(pr.node_labels); confs = [float(c) for c in pr.node_confidences]
        feat = assemble_flag_features(nodes, labels, confs,
                                      pr.stage1.cell_type, float(pr.stage1.confidence))
        feat.update(_f1=per_cell_neurite_f1(gt, labels, ct), _ct=ct, _file=p.name)
        # Each flag learns from the held-out cells of its reference cell type.
        rows[ct].append(feat)
        if i % progress_every == 0:
            print(f"    flag: dump {i}/{len(holdout)} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    by_type = {ct: _fit_one(rows[ct]) for ct in CELL_TYPES if rows[ct]}
    joblib.dump({"by_cell_type": by_type, "target": "regression:per_cell_neurite_f1", "seed": seed},
                model_dir / "flag_model.joblib", compress=3)
    summary = {ct: {k: b[k] for k in ("n_holdout", "bad_rate", "thresholds")} | {"n_features": len(b["feature_cols"])}
               for ct, b in by_type.items()}
    summary["elapsed_min"] = (time.perf_counter() - t0) / 60.0
    for ct, b in by_type.items():
        print(f"    flag ({ct}): trained on {b['n_holdout']} cells ({b['n_bad']} bad), "
              f"{len(b['feature_cols'])} features", flush=True)
    print("    -> flag_model.joblib", flush=True)
    return summary


# --------------------------------------------------------------------------- apply

def load_flag(model_dir: Path):
    """Load the flag bundle from a model dir, or None if no flag was trained there."""
    import joblib
    model_dir = Path(model_dir)
    fpath = model_dir / "flag_model.joblib"
    if not fpath.is_file():
        return None
    bundle = joblib.load(fpath)
    return bundle


def flag_for(bundle: dict, cell_type: str):
    """The flag that scores a cell of this Stage 1 type, or None if the bundle has
    none. A bundle holds one flag per cell type under ``by_cell_type``, or a single
    flag for the one type named by its ``scope``, as the curator loop writes."""
    if "by_cell_type" in bundle:
        return bundle["by_cell_type"].get(cell_type)
    return bundle if bundle.get("scope") == cell_type else None


def score_flag(nodes, pr, flag) -> dict:
    """Suspicion score in [0,1] for one already-labeled cell, plus a flagged flag
    at the default 10%-reject threshold. ``pr`` is the pipeline result for the
    cell and ``flag`` a ``load_flag`` bundle; no further pipeline pass is needed.

    The cell is scored by the flag of the type Stage 1 assigned it. A bundle with
    no flag for that type returns ``suspicion=None`` with ``out_of_scope`` set
    rather than a score from a model never fitted to such cells."""
    sub = flag_for(flag, pr.stage1.cell_type)
    if sub is None:
        return {"suspicion": None, "predicted_f1": None, "flagged": False,
                "threshold": None, "out_of_scope": pr.stage1.cell_type}
    labels = list(pr.node_labels); confs = [float(c) for c in pr.node_confidences]
    feat = assemble_flag_features(nodes, labels, confs,
                                  pr.stage1.cell_type, float(pr.stage1.confidence))
    x = np.array([[float(feat.get(c, np.nan)) for c in sub["feature_cols"]]], dtype=float)
    xi = sub["imputer"].transform(x)
    pred_f1 = float(sub["model"].predict(xi)[0])
    score = float(_risk_from_f1(pred_f1))
    thr = sub["thresholds"].get("0.10", 1.0)
    return {"suspicion": score, "predicted_f1": pred_f1,
            "flagged": bool(score >= thr), "threshold": thr, "cell_type": pr.stage1.cell_type}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the single-seed Stage 4 flag for a model dir.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--model-dir", type=Path, default=None, help="default pipeline/models/seed{S}")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    md = args.model_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")
    train_flag_for_seed(md, args.data_dir, args.seed, test_size=args.test_size,
                        limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
