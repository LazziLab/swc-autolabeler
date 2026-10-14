"""Train the unstaged labeler: one model per cell type over the whole feature set.

The staged counterpart is ``pipeline.train_pipeline``. This fits the single-model
alternative described in ``pipeline.unstaged``: every descriptor the pipeline
computes for a branch, its primary subtree and its cell, concatenated into one row,
with a single gradient-boosted classifier per cell type labeling each branch
directly. Stage 0 and the Stage 3 rules play no part, and no quality flag is fitted.

The split is the same deterministic ``md5("{seed}:{filename}")`` hash bucket used
everywhere else, and only the train side is fitted, so a model built here is
directly comparable with the staged models of the same seed.

Settings are fixed rather than searched, matching the staged pipeline's own
convention; ``evaluation.baselines.xgb_full`` is the driver that searches them and
reports what the search is worth.

    python -m pipeline.train_unstaged --data-dir data --seed 123
    python -m pipeline.train_unstaged --data-dir data --seed 123 --cell-type pyramidal

Writes ``unstaged_model.joblib`` into ``pipeline/models/seed{S}/`` beside the staged
models, because labeling a file still needs that directory's Stage 1 classifier to
decide the cell type. A corpus other than the bundled ``data/`` does not replace the
model in a released ``seed42``, ``seed123`` or ``seed789`` folder unless ``--overwrite``
is passed; give it its own ``--out-dir`` instead.
"""
from __future__ import annotations

import argparse

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                  # noqa: E402
from pipeline.stage2.train_stage2 import _node_balanced_weights           # noqa: E402
from pipeline.unstaged import (                                           # noqa: E402
    CELL_TYPES, FEATURE_VERSION, LABELS, branch_rows, save,
)
from pipeline.split import in_test as _in_test
from pipeline.released_models import refuse_replacing_released

#: Fixed settings, in the style of the staged pipeline's branch correction.
CONFIG = {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.08,
          "subsample": 0.8, "colsample_bytree": 0.8}


def _rows_for_file(task):
    """One cell's branch rows, their majority neurite label and node counts.

    Module level so it is picklable for parallel extraction. The result does not
    depend on the worker, so any number of workers gives the same training set.
    """
    path, cell_type = task
    nodes = parse_swc(path)
    if not nodes:
        return None
    rows, node_indices, _ = branch_rows(nodes, cell_type, str(path))
    if not len(rows):
        return None
    gt = np.asarray([nd.type for nd in nodes], dtype=np.int16)
    X, y, w = [], [], []
    for row, idx in zip(rows, node_indices):
        lab = gt[np.asarray(idx, dtype=np.int64)]
        keep = np.isin(lab, LABELS[cell_type])
        if not keep.any():
            continue                           # no valid neurite label to learn from
        values, counts = np.unique(lab[keep], return_counts=True)
        X.append(row)
        y.append(int(values[np.argmax(counts)]))
        w.append(float(keep.sum()))            # weight a branch by the nodes it covers
    if not X:
        return None
    return np.stack(X), np.asarray(y), np.asarray(w, dtype=float)


def _training_rows(files, cell_type: str, workers: int = 1, progress_every: int = 500):
    """Feature rows and majority labels of every branch of every training cell."""
    tasks = [(str(p), cell_type) for p in files]
    parts, t0 = [], time.perf_counter()

    def note(i):
        if i % progress_every == 0:
            print(f"    {i}/{len(tasks)} cells "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, part in enumerate(pool.map(_rows_for_file, tasks, chunksize=8), 1):
                if part is not None:
                    parts.append(part)
                note(i)
    else:
        for i, task in enumerate(tasks, 1):
            part = _rows_for_file(task)
            if part is not None:
                parts.append(part)
            note(i)
    if not parts:
        raise SystemExit(f"no usable training branches for {cell_type}")
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def train_unstaged(train_files: dict[str, list[Path]], out_path: Path, seed: int = 42,
                   workers: int = 1) -> dict:
    """Fit one classifier per cell type and write the bundle. Returns a summary."""
    from xgboost import XGBClassifier
    models, summary = {}, {}
    for cell_type in CELL_TYPES:
        files = train_files.get(cell_type) or []
        if not files:
            print(f"  {cell_type}: no training cells, skipping", flush=True)
            continue
        print(f"  {cell_type}: {len(files)} training cells", flush=True)
        X, y, w = _training_rows(files, cell_type, workers=workers)
        labels = LABELS[cell_type]
        encode = {lab: i for i, lab in enumerate(labels)}
        # Neither num_class nor the objective is stated. The sklearn wrapper
        # derives both from y: multi:softprob for the pyramidal three, and
        # binary:logistic for the interneuron two. Forcing multi:softprob broke
        # the two-class fit, because the wrapper only sets num_class above two
        # classes and the native side then rejects the zero it was left with.
        # Stating num_class instead would disagree with the data if a cell type's
        # training branches ever lacked one of its labels.
        clf = XGBClassifier(**CONFIG, tree_method="hist", random_state=seed,
                            verbosity=0)
        clf.fit(X, np.asarray([encode[v] for v in y]),
                sample_weight=_node_balanced_weights(y, w, 1.0))
        try:
            clf.get_booster().set_param({"device": "cpu"})   # CPU at inference
        except Exception:
            pass
        models[cell_type] = clf
        summary[cell_type] = {"n_cells": len(files), "n_branches": int(len(y))}
        print(f"    fitted on {len(y)} branches", flush=True)
    if not models:
        raise SystemExit("no cell type had training data")
    save(models, out_path, meta={"seed": seed, "config": CONFIG, "counts": summary})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default pipeline/models/seed{S}")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow a corpus other than data/ to replace a released seed42/123/789 model")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for feature extraction (default: CPU-1, capped 8)")
    ap.add_argument("--cell-type", choices=CELL_TYPES, default=None,
                    help="fit one cell type only (default: both)")
    args = ap.parse_args()

    import os
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    data_dir = args.data_dir.resolve()
    out_dir = (args.out_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")).resolve()
    refuse_replacing_released(out_dir, data_dir, args.overwrite)
    wanted = (args.cell_type,) if args.cell_type else CELL_TYPES
    train_files: dict[str, list[Path]] = {}
    for cell_type in wanted:
        sub = data_dir / cell_type
        if not sub.is_dir():
            raise SystemExit(f"--data-dir must contain a '{cell_type}/' subfolder; missing: {sub}")
        train_files[cell_type] = [p for p in sorted(sub.glob("*.swc"))
                                  if not _in_test(p.name, args.seed, args.test_size)]

    out_path = out_dir / "unstaged_model.joblib"
    print(f"train_unstaged  seed={args.seed}  features={FEATURE_VERSION}")
    print(f"  data: {data_dir}\n  out : {out_path}\n  workers: {workers}", flush=True)

    t0 = time.perf_counter()
    summary = train_unstaged(train_files, out_path, seed=args.seed, workers=workers)
    elapsed = (time.perf_counter() - t0) / 60.0

    (out_dir / "unstaged_manifest.json").write_text(json.dumps(
        {"seed": args.seed, "test_size": args.test_size, "config": CONFIG,
         "feature_version": FEATURE_VERSION, "counts": summary,
         "minutes": round(elapsed, 2),
         "notes": "One model per cell type over the whole feature set; no rules, no "
                  "correction, no quality flag. Labeling needs this directory's Stage 1 "
                  "classifier to decide the cell type."},
        indent=2), encoding="utf-8")
    print(f"\nDONE ({elapsed:.1f} min) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
