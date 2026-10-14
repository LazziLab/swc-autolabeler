"""Stage 1 evaluation: cell-type classifier (pyramidal vs interneuron).

Loads a trained Stage 1 model and scores it on a seed's held-out test split,
reproducing the paper's Stage 1 metrics: overall accuracy, the pyramidal vs
interneuron confusion matrix, and per-class precision / recall / F1.

The test split is the deterministic hash-bucket split used throughout the
paper: a file lands in test iff md5("{seed}:{filename}") maps below the test
fraction, so the same file + seed always lands in the same split.

Feature extraction (the slow step) runs in parallel across CPU cores; the model
is loaded once in the main process and used to predict on the extracted vectors.

Usage:
    python -m evaluation.eval_stage1 --seed 123
    python -m evaluation.eval_stage1 --seed 123 --model-dir pipeline/models/seed123

Output (evaluation/results/, created if missing):
    eval_stage1_seed{seed}.json
    eval_stage1_seed{seed}.txt
"""
from __future__ import annotations

import argparse

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.features import FEATURE_NAMES, extract_global_features, parse_swc  # noqa: E402
from pipeline.stage1.cell_type_detector import CellTypeClassifier               # noqa: E402
from evaluation.paths import display_path as _display_path                     # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed                    # noqa: E402
from pipeline.split import in_test as _in_test

CELL_TYPES = ("pyramidal", "interneuron")
DATA = ROOT / "data"
DEFAULT_TEST_SIZE = 0.20


def _extract(item: tuple[str, str]) -> tuple[str, list[float] | None, str]:
    """Worker: parse + extract the Stage 1 feature vector (no model here).

    The file name comes back with the vector so a cell that does not parse can be
    named in the report: it is dropped from the denominator, and a systematic
    failure would otherwise raise the accuracy with nothing to point at.
    """
    path_str, true_ct = item
    name = Path(path_str).name
    nodes = parse_swc(path_str)
    if not nodes:
        return true_ct, None, name
    features = extract_global_features(nodes)
    return true_ct, [float(features[name_]) for name_ in FEATURE_NAMES], name


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, precision_recall_fscore_support,
    )
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CELL_TYPES), zero_division=0,
    )
    per_class = {
        ct: {"precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f1[i]), "support": int(sup[i])}
        for i, ct in enumerate(CELL_TYPES)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(CELL_TYPES)).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="dir holding cell_type_classifier.pkl (default pipeline/models/seed{seed})")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results")
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()

    model_dir = args.model_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")
    check_model_seed(model_dir, args.seed, args.test_size)
    model_path = model_dir / "cell_type_classifier.pkl"
    if not model_path.is_file():
        raise SystemExit(f"MISSING model: {model_path}")

    test_items: list[tuple[str, str]] = []
    for ct in CELL_TYPES:
        for p in sorted((DATA / ct).glob("*.swc")):
            if _in_test(p.name, args.seed, args.test_size):
                test_items.append((str(p), ct))
    print(f"seed {args.seed}: {len(test_items)} test cells "
          f"{dict(Counter(ct for _, ct in test_items))}", flush=True)

    # 1) parallel feature extraction (no model in workers). Each worker is pinned
    # to one BLAS thread: extraction is numpy-heavy, and without this every worker
    # opens its own thread pool and they fight for the same cores.
    t0 = time.perf_counter()
    extracted: list[tuple[str, list[float] | None, str]] = []
    thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    saved = {v: os.environ.get(v) for v in thread_vars}
    os.environ.update({v: "1" for v in thread_vars})
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(_extract, test_items, chunksize=8), 1):
                extracted.append(res)
                if i % 500 == 0:
                    print(f"  ... features {i}/{len(test_items)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val

    # 2) load the model once and predict on every vector in one call
    clf = CellTypeClassifier.load(str(model_path))
    failed = [name for _ct, fv, name in extracted if fv is None]
    kept = [(ct, fv) for ct, fv, _name in extracted if fv is not None]
    n_failed = len(failed)
    y_true = [ct for ct, _fv in kept]
    X = np.asarray([fv for _ct, fv in kept], dtype=np.float64)
    # One predict_proba over the whole matrix. CellTypeClassifier.predict reshapes
    # a single vector and calls the estimator once per cell, which is thousands of
    # calls for one batch; the estimator underneath takes the matrix directly, and
    # argmax over its classes is exactly what predict returns.
    classes = list(clf._classes)
    y_pred = [classes[i] for i in clf.model.predict_proba(X).argmax(1)]

    m = _metrics(y_true, y_pred)
    m.update({
        "seed": args.seed,
        "test_size": args.test_size,
        "model": _display_path(model_path),
        "model_dir": _display_path(model_dir),
        "n_test": len(y_true),
        "n_failed": n_failed,
        "failed_files": failed,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
    })

    lines = [
        f"Stage 1 cell-type evaluation (seed {args.seed})",
        f"model      : {m['model']}",
        f"test cells : {m['n_test']}  (failed to parse: {n_failed}"
        + (f": {', '.join(failed[:5])}{' ...' if n_failed > 5 else ''})" if failed else ")"),
        f"accuracy   : {m['accuracy']:.4f}",
        f"macro F1   : {m['macro_f1']:.4f}",
        "",
        f"{'class':13s} {'precision':>9s} {'recall':>8s} {'f1':>8s} {'support':>8s}",
    ]
    for ct in CELL_TYPES:
        c = m["per_class"][ct]
        lines.append(f"{ct:13s} {c['precision']:9.4f} {c['recall']:8.4f} {c['f1']:8.4f} {c['support']:8d}")
    lines += ["", "confusion (rows=true, cols=pred; order: pyramidal, interneuron):"]
    for ct, row in zip(CELL_TYPES, m["confusion_matrix"]):
        lines.append(f"  {ct:13s} {row}")
    report = "\n".join(lines)
    print("\n" + report)

    # A run that is not the published one never takes its file name.
    tag = ""
    if args.test_size != DEFAULT_TEST_SIZE:
        tag += f"_test{args.test_size:g}"
    default_dir = ROOT / "pipeline" / "models" / f"seed{args.seed}"
    if model_dir.resolve() != default_dir.resolve():
        tag += "_" + model_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"eval_stage1_seed{args.seed}{tag}"
    stem.with_suffix(".json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    stem.with_suffix(".txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nwrote {_display_path(stem)}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
