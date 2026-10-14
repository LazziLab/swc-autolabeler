"""Curator feedback loop: does reviewing flagged cells improve the labeler?

Mimics one deployment cycle, repeated:

    label a batch -> the flag ranks it by suspicion -> the curator opens the
    top-K and corrects them -> those corrections go back into training ->
    the labeler and the ranking flag are retrained.

Ground truth stands in for the curator: revealing a cell's ground truth is that
cell being reviewed and corrected. Three arms run from the same starting point:
flag-guided review, random review (the ordinary learning-curve control), and
flag-guided review by a noisy curator who, on a fraction of reviews, confirms
the model's labels instead of correcting them.

The labeler is measured every round on a test set that is never reviewed:
neurite macro-F1, per-cell F1 mean / P10, node accuracy.

The flag here only chooses which cells to review. Each round it is refit on the
starting and reviewed cells as that round's retrained pipeline labels them, and
the labeler has just been trained on those cells, so they show fewer errors
than a curator meets. The flag scores this driver also records (kept-P10 @10% reject,
average precision, precision / recall) therefore do not measure the flag as
deployed, and the paper does not report them;
``evaluation.active_learning.flag_feedback`` measures the flag with the labeler
held fixed.

Every labeling, per round and at the endpoints, runs the complete pipeline with
the round's retrained Stage 2 slotted into the factory model set, so all numbers
share the deployment scale. Scope is pyramidal, matching the deployed flag.
Each round retrains the Stage 2 subtree classifier and the flag; Stage 1 and the
Stage 3 branch correction stay at their factory weights.

Cells are parsed from disk on demand and labeled in a process pool, so memory
stays flat with corpus size. A small per-corpus cache holds only the admitted
file list and each file's Stage 2 training rows; build it once before running
seeds in parallel:

    python -m evaluation.active_learning.curator_loop --prepare-only --n-max 5000

The flag-guided arm writes a curator checkpoint: a complete, drop-in model
directory holding the updated labeler and flag. The factory checkpoint that
``train_pipeline`` produced is never modified, so factory and curator models can
be compared on the same cells:

    python -m pipeline.infer_pipeline cell.swc -o out.swc \\
        --model-dir evaluation/results/active_learning/curator_seed123

Usage:
    python -m evaluation.active_learning.curator_loop --seed 123 --workers 5
    python -m evaluation.active_learning.curator_loop ... --workers 0   # in-process

Output (evaluation/results/active_learning/):
    curator_loop_seed{S}.{json,csv,txt}
    curator_seed{S}/            updated labeler + flag + provenance manifest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                    # noqa: E402
from pipeline.branch_features import extract_branches                      # noqa: E402
from pipeline.stage2.subtree_features import (                             # noqa: E402
    PRIMARY_SUBTREE_FEATURE_NAMES, extract_primary_subtrees,
)
from pipeline.stage2.train_stage2 import _build_subtree_pipeline, _node_balanced_weights  # noqa: E402
from pipeline.stage4.flag_geom_features import extract_features as geom_features  # noqa: E402
from pipeline.stage4.flag_metrics import per_cell_neurite_f1, _compute_metrics    # noqa: E402
from pipeline.stage4.flag_model import flag_regressor                      # noqa: E402
from evaluation.paths import display_path                                  # noqa: E402

DATA = ROOT / "data" / "pyramidal"
RESULTS = ROOT / "evaluation" / "results" / "active_learning"
CACHE = RESULTS / "_cache"
CELL_TYPE = "pyramidal"          # the cells the loop reviews, and the flag it retrains
VALID_LABELS = (2, 3, 4)         # the pyramidal Stage 2 label set
# Class balancing for the Stage 2 models this loop retrains. The factory model
# was fitted at 1.0, so an endpoint gap between an arm and the factory carries
# this difference as well as the difference in training-set size; the arms are
# balanced against each other, which is what the comparison rests on.
CLASS_BALANCE_POWER = 1.25
BAD_F1 = 0.60                    # reporting definition of a bad cell
REJECT_RATE = 0.10               # flag operating point used for reporting
DEFAULT_TEST_SIZE = 0.20
# The published protocol of Section 4.7: a 5,000-cell subsample of the
# QC-pass pyramidal corpus, 1,200 of them seen at the start, 200 reviewed
# each round. Running this file with no arguments reproduces the paper.
DEFAULT_N_MAX = 5000
DEFAULT_START_CELLS = 1200
DEFAULT_ROUNDS = 8
DEFAULT_K = 200
DEFAULT_NOISE_RATE = 0.12
# Everything a model dir may hold besides Stage 2.
FACTORY_FILES = ("cell_type_classifier.pkl", "branch_correction.joblib")


def _bucket(name: str, salt: str) -> float:
    return int(hashlib.md5(f"{salt}:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# --------------------------------------------------------------------------- workers

def _label_nodes(path: str, mdir: str):
    """Label one cell with the complete pipeline from the given model dir."""
    from pipeline.infer_pipeline import run_pipeline_on_nodes
    nodes = parse_swc(path)
    pr = run_pipeline_on_nodes(
        nodes, str(path), Path(mdir) / "cell_type_classifier.pkl",
        Path(mdir) / "branch_classifier.pkl", override_cell_type=CELL_TYPE)
    return (nodes, np.asarray(pr.node_labels, dtype=np.int64),
            np.asarray(pr.node_confidences, dtype=np.float64))


def _w_score(task):
    """One cell's flag feature row and per-cell F1, plus node-level ground truth
    and prediction when the cell is scored for pooled metrics."""
    path, mdir, keep_nodes = task
    nodes, pred, conf = _label_nodes(path, mdir)
    gt = [n.type for n in nodes]
    row = geom_features(
        pred, conf,
        np.asarray([n.x for n in nodes], np.float32),
        np.asarray([n.y for n in nodes], np.float32),
        np.asarray([n.z for n in nodes], np.float32),
        np.asarray([n.radius for n in nodes], np.float32),
        np.asarray([n.id for n in nodes], np.int64),
        np.asarray([n.parent for n in nodes], np.int64),
        CELL_TYPE, 1.0)
    out = {"row": row, "f1": per_cell_neurite_f1(gt, pred.tolist(), CELL_TYPE)}
    if keep_nodes:
        out["gt"] = np.asarray(gt, dtype=np.int8)
        out["pred"] = pred.astype(np.int8)
    return out


def _w_corrupt(task):
    """A noisy review: write a copy of the cell whose type column is the model's
    current prediction. The copy then stands in for the cell everywhere a
    reviewed cell is used, so both the Stage 2 trainer and the flag's F1 target
    see the curator's labels rather than the truth."""
    path, mdir, out_path = task
    nodes, pred, _ = _label_nodes(path, mdir)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# curator-noise copy: type column is the model's prediction\n")
        for n, t in zip(nodes, pred.tolist()):
            fh.write(f"{n.id} {int(t)} {np.float32(n.x):.6f} {np.float32(n.y):.6f} "
                     f"{np.float32(n.z):.6f} {np.float32(n.radius):.6f} {n.parent}\n")
    return out_path


def _w_records(path):
    """A file's Stage 2 training rows exactly as the trainer builds them: one per
    primary subtree whose majority label is in the label set, weighted by its
    node count."""
    nodes = parse_swc(path)
    recs = []
    for sub in extract_primary_subtrees(nodes):
        if sub.gt_label not in VALID_LABELS:
            continue
        recs.append((sub.features, sub.gt_label, float(sum(sub.gt_label_counts.values()))))
    return recs


def _w_admit(path):
    """The loop's corpus admission test: the file parses and has a branch."""
    nodes = parse_swc(path)
    if not nodes:
        return False
    try:
        return bool(extract_branches(nodes, CELL_TYPE, path).branches)
    except Exception:
        return False


# --------------------------------------------------------------------------- parent

def _make_pool(workers: int):
    """A spawn pool of single-threaded labeling processes; ``workers=0`` labels
    in this process instead (used to check the pool reproduces it exactly)."""
    if workers <= 0:
        return None
    # Workers are spawned once, single-threaded; the parent keeps its own thread
    # settings for the per-round retraining.
    threads = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
    saved = {var: os.environ.get(var) for var in threads}
    os.environ.update({var: "1" for var in threads})
    try:
        return mp.get_context("spawn").Pool(processes=workers)
    finally:
        for var, val in saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val


def _map(pool, fn, items, label, chunksize=2):
    n = len(items)
    step = max(1, n // 10)
    t0 = time.perf_counter()
    it = map(fn, items) if pool is None else pool.imap(fn, items, chunksize=chunksize)
    out = []
    for i, r in enumerate(it, 1):
        out.append(r)
        if i % step == 0 or i == n:
            print(f"    [{label}] {i}/{n} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    return out


def _subtree_stamp() -> str:
    """Identifies the Stage 2 feature layout the cached training rows were built from.

    The cache was keyed by corpus size alone, so a changed subtree feature set
    was reused silently: the rows still loaded and described different
    quantities.
    """
    return hashlib.md5("|".join(PRIMARY_SUBTREE_FEATURE_NAMES).encode()).hexdigest()[:12]


def _prepare(pool, n_max: int, rebuild: bool):
    """Admitted cells and their Stage 2 training rows, cached per corpus size."""
    tag = "all" if n_max >= 10**9 else str(n_max)
    path = CACHE / f"curator_light_{tag}.pkl"
    stamp = _subtree_stamp()
    d = None
    if path.is_file() and not rebuild:
        with path.open("rb") as fh:
            d = pickle.load(fh)
        if d.get("stamp") != stamp:
            print(f"  {path.name} was built from a different subtree feature set; "
                  f"rebuilding", flush=True)
            d = None
    if d is None:
        files = sorted(DATA.glob("*.swc"))
        files.sort(key=lambda p: _bucket(p.name, "cellpick"))
        files = files[:n_max]
        print(f"  admitting {len(files)} candidate pyramidal cells ...", flush=True)
        ok = _map(pool, _w_admit, [str(p) for p in files], "admit", chunksize=4)
        kept = [p for p, good in zip(files, ok) if good]
        print(f"  {len(kept)} admitted; extracting Stage 2 training rows ...", flush=True)
        recs = _map(pool, _w_records, [str(p) for p in kept], "records", chunksize=4)
        d = {"files": [p.name for p in kept], "stamp": stamp,
             "records": {p.name: r for p, r in zip(kept, recs)}}
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(d, fh)
        tmp.replace(path)
    cells = [{"file": f, "path": str(DATA / f)} for f in d["files"]]
    records = {str(DATA / f): r for f, r in d["records"].items()}
    return cells, records


def _ensure_records(pool, cells, records):
    missing = [c["path"] for c in cells if c["path"] not in records]
    if missing:
        for pth, r in zip(missing, _map(pool, _w_records, missing, "records, noisy copies")):
            records[pth] = r


def _fit_labeler(cells, records, seed: int):
    """Retrain the pipeline's own Stage 2 subtree classifier on the labeled cells,
    exactly as ``_train_subtree_owner_for_cell_type`` does (the same rows in the
    same order, class balancing and model), from rows cached per file instead of
    re-parsing every file each round."""
    X, y, w = [], [], []
    for c in cells:
        for feat, lab, wt in records[c["path"]]:
            X.append(feat)
            y.append(lab)
            w.append(wt)
    if not X or len(set(y)) < 2:
        return None
    X, y, w = np.stack(X), np.array(y), np.array(w)
    model = _build_subtree_pipeline(seed)
    model.fit(X, y, clf__sample_weight=_node_balanced_weights(y, w, CLASS_BALANCE_POWER))
    return model


def _make_round_model_dir(factory: Path, clf, workdir: Path, tag: str) -> Path:
    """A model dir whose Stage 2 pyramidal entry is the given model and everything
    else factory. Each call gets a fresh path so no worker can serve a stale
    Stage 2 from its cache."""
    mdir = workdir / f"_round_{tag}"
    mdir.mkdir(parents=True, exist_ok=True)
    for name in FACTORY_FILES:
        src = factory / name
        if src.is_file() and not (mdir / name).is_file():
            shutil.copy2(src, mdir / name)
    with open(factory / "branch_classifier.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    if clf is not None:
        bundle["subtree_owner_models_by_cell_type"][CELL_TYPE] = clf
    with open(mdir / "branch_classifier.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    return mdir


def _score(pool, mdir, cells, label, keep_nodes=False):
    outs = _map(pool, _w_score, [(c["path"], str(mdir), keep_nodes) for c in cells], label)
    rows = [o["row"] for o in outs]
    f1 = np.asarray([o["f1"] for o in outs], dtype=float)
    gts = [o["gt"] for o in outs] if keep_nodes else []
    preds = [o["pred"] for o in outs] if keep_nodes else []
    return rows, f1, gts, preds


# --------------------------------------------------------------------------- flag

def _fit_flag(rows: list[dict], f1: list[float], seed: int):
    """Train the flag on reviewed cells: features now, true per-cell F1 as target."""
    from sklearn.impute import SimpleImputer
    cols = sorted(rows[0].keys())
    X = np.asarray([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)
    m = flag_regressor(seed)
    m.fit(Xi, np.asarray(f1, dtype=float))
    # Rejection thresholds the way the single-seed flag sets them: score quantiles
    # over the cells the flag was fitted on.
    risk = np.clip(1.0 - m.predict(Xi), 0.0, 1.0)
    thresholds = {f"{r:.2f}": float(np.quantile(risk, 1.0 - r)) for r in (0.05, 0.10, 0.15)}
    return {"model": m, "imputer": imp, "feature_cols": cols, "thresholds": thresholds}


def _flag_risk(flag, rows: list[dict]) -> np.ndarray:
    cols = flag["feature_cols"]
    X = np.asarray([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    return np.clip(1.0 - flag["model"].predict(flag["imputer"].transform(X)), 0.0, 1.0)


def _measure(pool, mdir, flag, test_cells, label) -> dict:
    rows, f1, gts, preds = _score(pool, mdir, test_cells, label, keep_nodes=True)
    m = _compute_metrics(np.concatenate(gts).tolist(), np.concatenate(preds).tolist(), {1, 2, 3, 4})
    per = m.get("per_label", {})
    out = {
        "labeler_node_accuracy": round(float(m.get("accuracy", 0.0)), 4),
        "labeler_neurite_macro_f1": round(float(m.get("neurite_macro_f1", 0.0)), 4),
        "labeler_apical_f1": round(float(per.get("apical", {}).get("f1", 0.0)), 4),
        "labeler_pcf1_mean": round(float(f1.mean()), 4),
        "labeler_pcf1_p10": round(float(np.quantile(f1, 0.10)), 4),
    }
    # flag: rank the same cells, reject the top REJECT_RATE, score what is kept
    risk = _flag_risk(flag, rows)
    n = len(f1)
    k = max(1, int(round(REJECT_RATE * n)))
    # Stable, as flag_model._rejection_metrics sorts: cells of equal risk are
    # otherwise rejected in whatever order the sort produced.
    worst = np.argsort(-risk, kind="stable")[:k]
    keep = np.ones(n, bool)
    keep[worst] = False
    bad = f1 < BAD_F1
    from sklearn.metrics import average_precision_score
    out.update({
        "flag_kept_pcf1_p10": round(float(np.quantile(f1[keep], 0.10)), 4),
        "flag_precision": round(float(bad[worst].mean()), 4),
        "flag_recall": round(float(bad[worst].sum() / max(1, bad.sum())), 4),
        # None, not 0.0: with no bad cell or no good one, average precision is
        # undefined, and a literal zero reads as a flag that ranked perfectly badly.
        "flag_ap": (round(float(average_precision_score(bad, risk)), 4)
                    if 0 < bad.sum() < n else None),
        "n_bad_test": int(bad.sum()),
    })
    return out


# --------------------------------------------------------------------------- loop

def _run_arm(arm, seed_cells, pool_cells, test_cells, rounds, k, seed,
             factory, workdir, pool, records, noise_rate=0.0):
    """One arm of the loop. ``arm`` names the review policy (flag-guided when it
    starts with 'flag', else random); ``noise_rate`` > 0 corrupts that fraction of
    reviews, the curator confirming the model instead of correcting it."""
    labeled = list(seed_cells)
    remaining = list(pool_cells)
    curve = []
    noisy_dir = workdir / f"_noisy_swc_{arm}"
    rng = np.random.RandomState(seed)

    clf = _fit_labeler(labeled, records, seed)
    mdir = _make_round_model_dir(factory, clf, workdir, f"{arm}_r0")
    rows, f1, _, _ = _score(pool, mdir, labeled, f"{arm} r0 labeled")
    flag = _fit_flag(rows, list(f1), seed)
    m = _measure(pool, mdir, flag, test_cells, f"{arm} r0 test")
    m.update({"round": 0, "arm": arm, "n_labeled": len(labeled), "n_corrections": 0})
    curve.append(m)
    print(f"  [{arm}] round 0  labeled={len(labeled)}  "
          f"macroF1={m['labeler_neurite_macro_f1']}  keptP10={m['flag_kept_pcf1_p10']}", flush=True)

    for rnd in range(1, rounds + 1):
        if not remaining:
            break
        t_round = time.perf_counter()
        if arm.startswith("flag"):
            prows, _, _, _ = _score(pool, mdir, remaining, f"{arm} r{rnd} pool")
            # Stable: this order decides which cells the curator reviews, so an
            # unstable tie-break changes the whole run, not one reported number.
            order = np.argsort(-_flag_risk(flag, prows), kind="stable")   # riskiest first
            picked = [remaining[i] for i in order[:k]]
        else:
            picked = [remaining[i] for i in rng.permutation(len(remaining))[:k]]
        names = {c["file"] for c in picked}
        remaining = [c for c in remaining if c["file"] not in names]
        if noise_rate > 0:
            flips = [rng.random_sample() < noise_rate for _ in picked]
            tasks = [(c["path"], str(mdir), str(noisy_dir / c["file"]))
                     for c, flip in zip(picked, flips) if flip]
            new_paths = iter(_map(pool, _w_corrupt, tasks, f"{arm} r{rnd} noisy reviews")
                             if tasks else [])
            picked = [({"file": c["file"], "path": next(new_paths)} if flip else c)
                      for c, flip in zip(picked, flips)]
            _ensure_records(pool, picked, records)
        labeled.extend(picked)                                     # curator reviews

        shutil.rmtree(mdir, ignore_errors=True)
        clf = _fit_labeler(labeled, records, seed)                 # retrain labeler
        mdir = _make_round_model_dir(factory, clf, workdir, f"{arm}_r{rnd}")
        rows, f1, _, _ = _score(pool, mdir, labeled, f"{arm} r{rnd} labeled")
        flag = _fit_flag(rows, list(f1), seed)                     # retrain flag

        m = _measure(pool, mdir, flag, test_cells, f"{arm} r{rnd} test")
        m.update({"round": rnd, "arm": arm, "n_labeled": len(labeled),
                  "n_corrections": len(labeled) - len(seed_cells)})
        curve.append(m)
        print(f"  [{arm}] round {rnd}  corrections={m['n_corrections']}  "
              f"macroF1={m['labeler_neurite_macro_f1']}  keptP10={m['flag_kept_pcf1_p10']}  "
              f"({(time.perf_counter() - t_round) / 60:.1f} min)", flush=True)
    shutil.rmtree(mdir, ignore_errors=True)
    shutil.rmtree(noisy_dir, ignore_errors=True)
    return curve, clf, flag, [c["file"] for c in labeled]


def _endpoint_full_pipeline(pool, factory: Path, arms: dict, test_cells, workdir: Path) -> dict:
    """Measure each arm's final Stage 2 through the complete pipeline, next to the
    factory Stage 2, on the same test cells."""
    out = {}
    for arm, clf in arms.items():
        mdir = _make_round_model_dir(factory, clf, workdir, f"endpoint_{arm}")
        _, f1, gts, preds = _score(pool, mdir, test_cells, f"endpoint {arm}", keep_nodes=True)
        m = _compute_metrics(np.concatenate(gts).tolist(), np.concatenate(preds).tolist(),
                             {1, 2, 3, 4})
        out[arm] = {
            "node_accuracy": round(float(m.get("accuracy", 0.0)), 4),
            "neurite_macro_f1": round(float(m.get("neurite_macro_f1", 0.0)), 4),
            "apical_f1": round(float(m.get("per_label", {}).get("apical", {}).get("f1", 0.0)), 4),
            "pcf1_mean": round(float(f1.mean()), 4),
            "pcf1_p10": round(float(np.quantile(f1, 0.10)), 4),
        }
        print(f"  [{arm}] full pipeline: acc={out[arm]['node_accuracy']} "
              f"macroF1={out[arm]['neurite_macro_f1']} pcF1-P10={out[arm]['pcf1_p10']}", flush=True)
        shutil.rmtree(mdir, ignore_errors=True)
    return out


def _write_checkpoint(out_dir: Path, factory_dir: Path, clf, flag, meta: dict) -> None:
    """Write a drop-in model dir: unchanged stages copied, retrained ones replaced.

    The factory checkpoint is never modified; this is a separate, versioned model
    set carrying the curator's corrections.
    """
    import joblib
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FACTORY_FILES:
        src = factory_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)

    # Stage 2 ships as a bundle keyed by cell type, written with plain pickle
    # (infer_pipeline reads it with pickle.load). Retrain replaces the pyramidal
    # entry only; the interneuron model is inherited from the factory, since the
    # loop reviews pyramidal cells.
    with open(factory_dir / "branch_classifier.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    bundle["subtree_owner_models_by_cell_type"][CELL_TYPE] = clf
    with open(out_dir / "branch_classifier.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    # The loop's flag rows are the geometry block under its bare names, while
    # pipeline.stage4.flag_features serves that block with a geom_ prefix. The
    # saved bundle names its columns the way score_flag looks them up, so the
    # checkpoint's flag scores a cell at inference like any other model dir's.
    joblib.dump({"model": flag["model"], "imputer": flag["imputer"],
                 "feature_cols": [f"geom_{c}" for c in flag["feature_cols"]],
                 "thresholds": flag["thresholds"],
                 "target": "regression:per_cell_neurite_f1", "scope": CELL_TYPE},
                out_dir / "flag_model.joblib", compress=3)
    (out_dir / "curator_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-max", type=int, default=DEFAULT_N_MAX,
                    help="cells to draw from (0 = all pyramidal cells)")
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help="held-out fraction; the test cells are the ones the "
                         "factory models never trained on")
    ap.add_argument("--start-cells", type=int, default=DEFAULT_START_CELLS,
                    help="cells the starting labeler is trained on; the rest of the "
                         "factory's training split becomes the not-yet-reviewed pool")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="cells the curator reviews per round")
    ap.add_argument("--factory-dir", type=Path, default=None,
                    help="default pipeline/models/seed{seed}; never modified")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--noise-rate", type=float, default=DEFAULT_NOISE_RATE,
                    help="fraction of reviews in the noisy-curator arm where the "
                         "curator confirms the model instead of correcting it "
                         "(0 disables the arm)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4),
                    help="labeling processes (0 labels in this process)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="build the corpus cache and exit")
    ap.add_argument("--tag", default="", help="extra suffix for output file names")
    args = ap.parse_args()

    # A run whose settings differ from the published ones takes its own file
    # names, whether or not --tag was remembered: --tag was the only thing
    # standing between a trial run and the released results and checkpoint.
    auto = ""
    if args.n_max != DEFAULT_N_MAX:
        auto += f"_n{args.n_max}"
    if args.start_cells != DEFAULT_START_CELLS:
        auto += f"_start{args.start_cells}"
    if args.rounds != DEFAULT_ROUNDS:
        auto += f"_r{args.rounds}"
    if args.k != DEFAULT_K:
        auto += f"_k{args.k}"
    if args.noise_rate != DEFAULT_NOISE_RATE:
        auto += f"_noise{args.noise_rate:g}"
    if args.test_size != DEFAULT_TEST_SIZE:
        auto += f"_test{args.test_size:g}"
    if auto:
        print(f"  this run differs from the published one; writing with tag {auto}",
              flush=True)
    args.tag = auto + args.tag

    factory = args.factory_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")
    pool = _make_pool(args.workers)
    workdir = None
    try:
        cells, records = _prepare(pool, args.n_max or 10**9, args.rebuild_cache)
        if args.prepare_only:
            print(f"  corpus cache ready: {len(cells)} cells", flush=True)
            return 0

        # The pipeline's own hash split: test cells are exactly the ones the
        # factory never trained on, so factory and curator models are directly
        # comparable. The factory's TRAINING cells become the timeline: a slice
        # of them is what the starting labeler has seen, the rest is the backlog.
        from pipeline.split import in_test
        test = [c for c in cells if in_test(c["file"], args.seed, args.test_size)]
        train_all = [c for c in cells if not in_test(c["file"], args.seed, args.test_size)]
        train_all.sort(key=lambda c: _bucket(c["file"], f"backlog{args.seed}"))
        n_start = min(args.start_cells, len(train_all))
        seed_cells, backlog = train_all[:n_start], train_all[n_start:]
        print(f"\ncurator loop  seed={args.seed}  pyramidal cells={len(cells)}  workers={args.workers}")
        print(f"  {len(train_all)} of these cells are in the factory's training split; "
              f"this run starts with {len(seed_cells)}")
        print(f"  curator backlog (pool)={len(backlog)}   test={len(test)} (never trained on, by anyone)")
        print(f"  rounds={args.rounds}  k={args.k} cells reviewed per round\n", flush=True)

        workdir = RESULTS / f"_loop_work_seed{args.seed}{args.tag}"
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        print("=== ARM: flag-guided ===", flush=True)
        flag_curve, clf_flag, flag_final, rev_flag = _run_arm(
            "flag", seed_cells, backlog, test, args.rounds, args.k, args.seed,
            factory, workdir, pool, records)
        print("\n=== ARM: random (control) ===", flush=True)
        rand_curve, clf_rand, _, _ = _run_arm(
            "random", seed_cells, backlog, test, args.rounds, args.k, args.seed,
            factory, workdir, pool, records)
        noisy_curve, clf_noisy = [], None
        if args.noise_rate > 0:
            print(f"\n=== ARM: flag-guided, noisy curator ({args.noise_rate:.0%}) ===", flush=True)
            noisy_curve, clf_noisy, _, _ = _run_arm(
                "flag_noisy", seed_cells, backlog, test, args.rounds, args.k, args.seed,
                factory, workdir, pool, records, noise_rate=args.noise_rate)
        elapsed = (time.perf_counter() - t0) / 60.0

        RESULTS.mkdir(parents=True, exist_ok=True)
        stem = f"curator_loop_seed{args.seed}{args.tag}"
        # n_factory_train counts this run's cells that lie in the factory's
        # training split, not the factory's whole training set, which
        # report.curator_loop_table counts over the full pyramidal corpus.
        out = {"seed": args.seed, "n_cells": len(cells), "n_start": len(seed_cells),
               "n_factory_train": len(train_all), "n_pool": len(backlog), "n_test": len(test),
               "rounds": args.rounds, "k": args.k, "reject_rate": REJECT_RATE,
               "bad_f1_threshold": BAD_F1, "scope": CELL_TYPE,
               "retrained_per_round": ["stage2_subtree_owner", "flag"],
               "elapsed_min": round(elapsed, 2), "noise_rate": args.noise_rate,
               "flag_arm": flag_curve, "random_arm": rand_curve, "flag_noisy_arm": noisy_curve}

        cols = ["arm", "round", "n_corrections", "labeler_neurite_macro_f1", "labeler_apical_f1",
                "labeler_pcf1_mean", "labeler_pcf1_p10", "labeler_node_accuracy",
                "flag_kept_pcf1_p10", "flag_precision", "flag_recall", "flag_ap"]
        with (RESULTS / f"{stem}.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in flag_curve + rand_curve + noisy_curve:
                w.writerow(r)

        arms_txt = [("flag", flag_curve), ("random", rand_curve)] + (
            [("noisy", noisy_curve)] if noisy_curve else [])
        lines = [f"Curator feedback loop (seed {args.seed}, pyramidal, {len(test)} test cells)",
                 "=" * 78, "",
                 f"{'corrections':>11} | {'labeler macroF1':^{13 * len(arms_txt)}} | "
                 f"{'flag kept-P10 @10%':^{13 * len(arms_txt)}}",
                 f"{'':>11} | " + "".join(f"{a:>13}" for a, _ in arms_txt) + " | "
                 + "".join(f"{a:>13}" for a, _ in arms_txt)]
        for i, f in enumerate(flag_curve):
            cells_l = "".join(f"{c[i]['labeler_neurite_macro_f1']:>13.4f}" for _, c in arms_txt)
            cells_f = "".join(f"{c[i]['flag_kept_pcf1_p10']:>13.4f}" for _, c in arms_txt)
            lines.append(f"{f['n_corrections']:>11} | {cells_l} | {cells_f}")
        report = "\n".join(lines)
        print("\n" + report, flush=True)

        print("\n=== endpoint: complete pipeline on the test set ===", flush=True)
        arms = {"factory": None, "flag": clf_flag, "random": clf_rand}
        if clf_noisy is not None:
            arms["flag_noisy"] = clf_noisy
        endpoints = _endpoint_full_pipeline(pool, factory, arms, test, workdir)
        out["endpoint_full_pipeline"] = endpoints
        (RESULTS / f"{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        ep = ["", "Complete-pipeline endpoints (factory Stage 1 and branch correction, retrained Stage 2):",
              f"{'arm':>11} {'node acc':>10} {'macroF1':>9} {'apical':>8} {'pcF1 mean':>10} {'pcF1 P10':>9}"]
        for a, e in endpoints.items():
            ep.append(f"{a:>11} {e['node_accuracy']:>10.4f} {e['neurite_macro_f1']:>9.4f} "
                      f"{e['apical_f1']:>8.4f} {e['pcf1_mean']:>10.4f} {e['pcf1_p10']:>9.4f}")
        report_ep = "\n".join(ep)
        (RESULTS / f"{stem}.txt").write_text(report + "\n" + report_ep + "\n", encoding="utf-8")
        print(report_ep, flush=True)

        if not args.tag:
            ckpt = RESULTS / f"curator_seed{args.seed}"
            _write_checkpoint(ckpt, factory, clf_flag, flag_final, {
                "base_checkpoint": display_path(factory),
                "seed": args.seed, "rounds": args.rounds, "k": args.k,
                "curator_corrections": flag_curve[-1]["n_corrections"],
                "reviewed_files": sorted(set(rev_flag)),
                "retrained": ["branch_classifier.pkl", "flag_model.joblib"],
                "inherited": [n for n in FACTORY_FILES if (factory / n).is_file()],
                "final": {key: flag_curve[-1][key] for key in
                          ("labeler_neurite_macro_f1", "labeler_pcf1_p10", "flag_kept_pcf1_p10")},
            })
            print(f"  wrote checkpoint {ckpt.name}/ (factory checkpoint untouched)")
        print(f"\n  wrote {stem}.{{json,csv,txt}}")
        print(f"  elapsed {elapsed:.1f} min", flush=True)
        return 0
    finally:
        # In finally, so a crash does not leave round model directories behind.
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    sys.exit(main())
