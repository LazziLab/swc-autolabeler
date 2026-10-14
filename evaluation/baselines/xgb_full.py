"""A single tuned XGBoost on the pipeline's complete feature set.

A controlled baseline that uses everything the pipeline measures, fitted as one
tuned model. This driver trains one
gradient-boosted classifier per cell type that labels each branch directly from
every label-free descriptor the pipeline computes for it, with no subtree stage,
no topology rules and no correction:

  branch    the 57 per-branch descriptors the branch correction reads
  subtree   the 24 descriptors of the branch's primary subtree (Stage 2)
  cell      the 55 whole-cell descriptors (Stage 1)

A branch's target is its majority ground-truth neurite label; training rows are
weighted by node count with node-level class balancing, as Stage 2's are.
Hyperparameters are tuned per cell type and seed with a fixed budget: ``--budget``
random configurations, each scored by node-weighted macro-F1 in 3-fold
cross-validation grouped by cell over ``--tune-cells`` training cells; the best
configuration is refitted on every training cell. On a test cell the soma is the
pipeline's label-free soma anchor, each branch takes its predicted label, and a
node outside every branch inherits its nearest labeled ancestor's label
(basal/dendrite if none). Scores use the pipeline's metric pack on the same
hash split as every other method.

Features are label-free and independent of the split, so they are extracted once
per cell and cached.

Usage:
    python -m evaluation.baselines.xgb_full [--seeds 42,123,789] [--budget 20]

Output (evaluation/results/baselines/):
    xgb_full_seed{S}.json   report (same layout as eval_baselines)
    xgb_full_seed{S}.csv    per-cell rows
    xgb_full_tuning.json    every configuration tried, its CV score, and the one chosen
    _xgb_full_cache/        per-cell feature cache
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
OUT = ROOT / "evaluation" / "results" / "baselines"
#: The feature cache is label-free and split-independent, so it stays here even
#: when --out-dir sends the reports elsewhere; entries are keyed by cell name and
#: feature version, so a reduced run adds to it and never corrupts it.
CACHE = OUT / "_xgb_full_cache"
DEFAULT_BUDGET = 20
DEFAULT_TUNE_CELLS = 1500
DEFAULT_TEST_SIZE = 0.20
from pipeline.unstaged import (                                          # noqa: E402
    CELL_TYPES, FEATURE_VERSION, LABELS, branch_rows, parent_indices,
)


def _extract(task):
    """Cache the label-free features of every branch of one cell, plus what scoring needs."""
    ct, path = task
    out = CACHE / ct / (Path(path).stem + ".npz")
    if out.is_file():
        # The file name cannot tell two feature layouts apart, so an entry that
        # does not carry the current one is rebuilt rather than reused.
        try:
            with np.load(out) as cached:
                stamp = str(cached["feature_version"]) if "feature_version" in cached else None
        except Exception:
            stamp = None
        if stamp == FEATURE_VERSION:
            return ct, Path(path).name, str(out), ""
    from pipeline.features import parse_swc

    nodes = parse_swc(path)
    if not nodes:
        return ct, Path(path).name, None, "empty parse"
    # The feature matrix comes from pipeline.unstaged, not a second copy of the
    # same assembly, so the tuned bound and the released fixed-settings model
    # cannot drift apart: FEATURE_VERSION counts descriptors and would not catch
    # a change of concatenation order or of the missing-subtree fill.
    try:
        rows, node_indices, soma = branch_rows(nodes, ct, str(path))
    except Exception as exc:
        # The reason is carried back rather than discarded: a systematic
        # featurization failure and an unparseable file both ended up in one
        # "skipped" count, which reads as bad data rather than as a bug.
        return ct, Path(path).name, None, f"{type(exc).__name__}: {exc}"
    if not len(rows):
        return ct, Path(path).name, None, "no branches"
    # Everything below reads the type column; branch_rows deliberately does not.
    gt = np.asarray([nd.type for nd in nodes], dtype=np.int16)
    X, y, w, flat, starts = list(rows), [], [], [], [0]
    for node_idx in node_indices:
        idx = np.asarray(node_idx, dtype=np.int32)
        lab = gt[idx]
        m = np.isin(lab, LABELS[ct])
        if m.any():
            vals, cnts = np.unique(lab[m], return_counts=True)
            y.append(int(vals[np.argmax(cnts)]))
            w.append(float(m.sum()))
        else:
            y.append(0)
            w.append(0.0)
        flat.append(idx)
        starts.append(starts[-1] + len(idx))
    parent = np.asarray(parent_indices(nodes), dtype=np.int32)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, X=np.stack(X), y=np.asarray(y, np.int8), w=np.asarray(w, np.float32),
                        node_flat=np.concatenate(flat), node_starts=np.asarray(starts, np.int64),
                        gt=gt.astype(np.int8), parent=parent,
                        soma=np.asarray(soma, np.int32),
                        feature_version=np.asarray(FEATURE_VERSION))
    tmp.replace(out)
    return ct, Path(path).name, str(out), ""


def _stack(cells: list[tuple[str, str]]):
    X, y, w, g = [], [], [], []
    for gi, (_, cpath) in enumerate(cells):
        d = np.load(cpath)
        keep = d["y"] > 0
        X.append(d["X"][keep])
        y.append(d["y"][keep].astype(np.int64))
        w.append(d["w"][keep])
        g.append(np.full(int(keep.sum()), gi, dtype=np.int64))
    return np.concatenate(X), np.concatenate(y), np.concatenate(w), np.concatenate(g)


def _weighted_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray, labels) -> float:
    scores = []
    for c in labels:
        if not (y_true == c).any():
            continue
        tp = w[(y_true == c) & (y_pred == c)].sum()
        fp = w[(y_true != c) & (y_pred == c)].sum()
        fn = w[(y_true == c) & (y_pred != c)].sum()
        scores.append(float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _sample_config(rng: np.random.Generator) -> dict:
    return {"n_estimators": int(rng.choice([150, 300, 600])),
            "max_depth": int(rng.choice([4, 6, 8, 10])),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.03), np.log(0.3)))),
            "subsample": float(rng.choice([0.7, 0.85, 1.0])),
            "colsample_bytree": float(rng.choice([0.5, 0.7, 0.9])),
            "min_child_weight": float(rng.choice([1.0, 5.0, 20.0])),
            "reg_lambda": float(rng.choice([0.5, 1.0, 5.0])),
            "balance_power": float(rng.choice([1.0, 1.25]))}


def _device(choice: str = "auto") -> str:
    """The xgboost device to fit on.

    ``auto`` uses a GPU when this build has one. CPU and GPU ``hist`` choose
    different splits, so a bound fitted on one machine is not reproducible on the
    other; ``--device cpu`` is how a reader with a GPU reproduces a CPU run.
    """
    if choice != "auto":
        return choice
    import xgboost as xgb
    try:
        return "cuda" if xgb.build_info().get("USE_CUDA") else "cpu"
    except Exception:
        return "cpu"


def _fit(cfg: dict, X, y, w, labels, seed: int, device: str):
    import xgboost as xgb
    from pipeline.stage2.train_stage2 import _node_balanced_weights
    enc = {lab: i for i, lab in enumerate(labels)}
    params = {k: v for k, v in cfg.items() if k != "balance_power"}
    clf = xgb.XGBClassifier(**params, tree_method="hist", device=device, random_state=seed, verbosity=0)
    clf.fit(X, np.asarray([enc[v] for v in y]), sample_weight=_node_balanced_weights(y, w, cfg["balance_power"]))
    try:
        clf.get_booster().set_param({"device": "cpu"})
    except Exception:
        pass
    return clf


def _predict(clf, X, labels) -> np.ndarray:
    return np.asarray(labels)[np.asarray(clf.predict(X), dtype=int)]


def _node_labels(d, pred_branch: np.ndarray) -> np.ndarray:
    n = len(d["gt"])
    lab = np.zeros(n, dtype=np.int8)
    starts, flat = d["node_starts"], d["node_flat"]
    for bi, p in enumerate(pred_branch):
        lab[flat[starts[bi]:starts[bi + 1]]] = p
    lab[d["soma"]] = 1
    parent = d["parent"]
    for i in np.flatnonzero(lab == 0):
        j, hops = parent[i], 0
        while j >= 0 and lab[j] in (0, 1) and hops < n:
            j, hops = parent[j], hops + 1
        lab[i] = lab[j] if j >= 0 and lab[j] not in (0, 1) else 3
    return lab


def _run_tag(args) -> str:
    """File-name tag for a run that is not the published one.

    Every file a reduced run writes carries it, the tuning log included: that log
    was the one output without a tag, so a smoke test overwrote the real tuning
    results while leaving the per-seed files alone.
    """
    tag = ""
    if args.limit:
        tag += f"_limit{args.limit}"
    if args.tune_cells != DEFAULT_TUNE_CELLS:
        tag += f"_tune{args.tune_cells}"
    if args.budget != DEFAULT_BUDGET:
        tag += f"_budget{args.budget}"
    if args.test_size != DEFAULT_TEST_SIZE:
        tag += f"_test{args.test_size:g}"
    return tag


def run_seed(seed: int, index: dict, args, device: str, tuning_log: dict) -> None:  # noqa: PLR0915
    from sklearn.model_selection import GroupKFold
    from evaluation.baselines.baseline_report import build_full_report
    from evaluation.stage4._flag_metrics import in_test, per_cell_neurite_f1

    # One generator per purpose: the cell subsample and the configuration search
    # shared a stream, and cell types run in sequence, so changing --tune-cells
    # silently changed which configurations the second cell type tried.
    records, rows, n_train = [], [], 0
    t0 = time.perf_counter()
    for ct in CELL_TYPES:
        pick_rng = np.random.default_rng([seed, CELL_TYPES.index(ct), 1])
        cfg_rng = np.random.default_rng([seed, CELL_TYPES.index(ct), 2])
        labels = LABELS[ct]
        train = [c for c in index[ct] if not in_test(c[0], seed, args.test_size)]
        test = [c for c in index[ct] if in_test(c[0], seed, args.test_size)]
        n_train += len(train)
        pick = sorted(pick_rng.choice(len(train), min(args.tune_cells, len(train)), replace=False))
        Xt, yt, wt, gt_ = _stack([train[i] for i in pick])
        folds = list(GroupKFold(n_splits=3).split(Xt, yt, gt_))
        tried = []
        for ci in range(args.budget):
            cfg = _sample_config(cfg_rng)
            scores = []
            for tr, va in folds:
                clf = _fit(cfg, Xt[tr], yt[tr], wt[tr], labels, seed, device)
                scores.append(_weighted_macro_f1(yt[va], _predict(clf, Xt[va], labels), wt[va], labels))
            tried.append({"config": cfg, "cv_macro_f1": float(np.mean(scores))})
            print(f"  seed {seed} {ct}: config {ci + 1}/{args.budget} cv macro-F1 {np.mean(scores):.4f} "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
        best = max(tried, key=lambda t: t["cv_macro_f1"])
        tuning_log[f"seed{seed}/{ct}"] = {"n_tune_cells": len(pick), "tried": tried, "best": best}
        Xa, ya, wa, _ = _stack(train)
        clf = _fit(best["config"], Xa, ya, wa, labels, seed, device)
        del Xa, ya, wa
        for name, cpath in test:
            d = np.load(cpath)
            pred = _node_labels(d, _predict(clf, d["X"], labels))
            gt = d["gt"].astype(int).tolist()
            pl = pred.astype(int).tolist()
            records.append({"file": name, "cell_type": ct, "n_nodes": len(gt), "gt": gt, "pred": pl})
            rows.append({"method": "xgb_full", "file": name, "cell_type": ct, "n_nodes": len(gt),
                         "accuracy": f"{np.mean(np.asarray(gt) == np.asarray(pl)):.4f}",
                         "neurite_macro_f1": f"{per_cell_neurite_f1(gt, pl, ct):.4f}"})
        print(f"  seed {seed} {ct}: best cv macro-F1 {best['cv_macro_f1']:.4f}; scored {len(test)} test cells",
              flush=True)
    report = build_full_report("xgb_full", [g for r in records for g in r["gt"]],
                               [p for r in records for p in r["pred"]], records)
    report["wall_clock_min"] = (time.perf_counter() - t0) / 60.0
    report["n_train_cells"] = n_train
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = _run_tag(args)
    (args.out_dir / f"xgb_full_seed{seed}{tag}.json").write_text(json.dumps({
        "seed": seed, "n_train": n_train, "n_test": len(records),
        "device": device,   # GPU and CPU hist can split differently
        "limit": args.limit,
        "split_source": "hash-bucket split over data/{pyramidal,interneuron}/*.swc",
        "convention": ("per-cell macro F1 averages only the neurite classes "
                       "present in that cell's ground truth; the soma is "
                       "excluded from every reported metric"),
        "tuning_budget": {"configs_per_cell_type": args.budget, "cv_folds": 3, "tune_cells": args.tune_cells},
        "reports": [report]}, indent=2), encoding="utf-8")
    with (args.out_dir / f"xgb_full_seed{seed}{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["method", "file", "cell_type", "n_nodes",
                                             "accuracy", "neurite_macro_f1"])
        wtr.writeheader()
        wtr.writerows(rows)
    c = report["corpus"]
    pc = report["per_cell"]["neurite_macro_f1"]
    print(f"seed {seed}: acc {c['accuracy']:.4f}  neurite macro-F1 {c['neurite_macro_f1']:.4f}  "
          f"per-cell mean {pc['mean']:.4f}  P10 {pc['p10']:.4f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help="held-out fraction; must match the run this is compared against")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="random configurations per cell type and seed")
    ap.add_argument("--tune-cells", type=int, default=DEFAULT_TUNE_CELLS,
                    help="training cells subsampled for tuning")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="xgboost device. GPU and CPU hist split differently, so a "
                         "bound fitted on one is not reproducible on the other; the "
                         "device used is recorded in the report.")
    ap.add_argument("--workers", type=int, default=4, help="processes for feature extraction")
    ap.add_argument("--limit", type=int, default=0, help="cells per cell type (0 = all; smoke tests)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    tasks = []
    for ct in CELL_TYPES:
        files = sorted((DATA / ct).glob("*.swc"))
        tasks += [(ct, str(p)) for p in (files[:args.limit] if args.limit else files)]
    print(f"feature cache: {len(tasks)} cells, {args.workers} workers", flush=True)
    index = {ct: [] for ct in CELL_TYPES}
    skipped = 0
    skip_reasons: dict[str, int] = {}
    first_skips: list[str] = []
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (ct, name, cpath, why) in enumerate(pool.imap_unordered(_extract, tasks, chunksize=8), 1):
            if cpath is None:
                skipped += 1
                skip_reasons[why] = skip_reasons.get(why, 0) + 1
                if len(first_skips) < 5:
                    first_skips.append(f"{name}: {why}")
            else:
                index[ct].append((name, cpath))
            if i % 1000 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    for ct in CELL_TYPES:
        index[ct].sort()
    print(f"cached {sum(len(v) for v in index.values())} cells, skipped {skipped}", flush=True)
    for why, n in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5d}  {why}", flush=True)
    for line in first_skips:
        print(f"    e.g. {line}", flush=True)

    device = _device(args.device)
    print(f"xgboost device: {device}", flush=True)
    tuning_log: dict = {}
    for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
        run_seed(s, index, args, device, tuning_log)
        (args.out_dir / f"xgb_full_tuning{_run_tag(args)}.json").write_text(json.dumps(
            {"budget": args.budget, "cv_folds": 3, "tune_cells": args.tune_cells,
             "test_size": args.test_size, "device": device,
             "skipped_cells": skipped, "skip_reasons": skip_reasons,
             "runs": tuning_log}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
