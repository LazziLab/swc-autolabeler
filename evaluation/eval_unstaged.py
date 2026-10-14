"""Held-out evaluation of the unstaged labeler, beside ``evaluation.eval_pipeline``.

Scores the model ``pipeline.train_unstaged`` fitted on the same held-out cells the
staged pipeline is scored on, with the same metric and the same configuration: the
ground-truth cell type is supplied, so what is measured is the labeling, not the
routing. Together with ``evaluation.generalization.eval_unstaged_external`` this
produces the two unstaged rows of the paper's staged-against-unstaged table.

The split is the deterministic hash bucket used throughout, and the model carries
the seed it was trained with, which is checked against ``--seed`` before scoring.

    python -m evaluation.eval_unstaged --seed 123

Output (evaluation/results/):
    eval_unstaged_seed{seed}.json / .txt
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

from pipeline.features import parse_swc                                   # noqa: E402
from pipeline.unstaged import CELL_TYPES, label_cell, load                # noqa: E402
from evaluation.eval_pipeline import node_metrics                             # noqa: E402
from evaluation.paths import display_path as _display                     # noqa: E402
from evaluation.stage4._flag_metrics import (                             # noqa: E402
    check_model_seed, per_cell_neurite_f1,
)
from pipeline.split import in_test as _in_test

DATA = ROOT / "data"
DEFAULT_TEST_SIZE = 0.20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="default pipeline/models/seed{seed}")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results")
    ap.add_argument("--limit", type=int, default=0, help="cells per cell type (0 = all)")
    args = ap.parse_args()

    model_dir = (args.model_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")).resolve()
    bundle = load(model_dir)
    if bundle is None:
        raise SystemExit(f"MISSING unstaged_model.joblib in {model_dir}\n"
                         f"  train it: python -m pipeline.train_unstaged --seed {args.seed}")
    # The bundle carries its own seed, and the model directory carries the
    # manifest train_pipeline wrote. Both are checked: the local test compared
    # seeds only, so a model fitted under a different test_size passed, and a
    # bundle that recorded no seed passed in silence.
    trained = bundle.get("seed")
    if trained is not None and int(trained) != args.seed:
        raise SystemExit(f"{model_dir} holds a model trained with seed {trained}, but --seed "
                         f"is {args.seed}: its held-out cells differ.")
    if trained is None:
        print(f"warning: the model in {_display(model_dir)} records no training seed",
              flush=True)
    check_model_seed(model_dir, args.seed, args.test_size)

    test = [(ct, p) for ct in CELL_TYPES
            for p in sorted((DATA / ct).glob("*.swc"))
            if _in_test(p.name, args.seed, args.test_size)]
    if args.limit:
        test = [x for ct in CELL_TYPES
                for x in [t for t in test if t[0] == ct][:args.limit]]
    print(f"unstaged model {_display(model_dir)}: {len(test)} held-out cells", flush=True)

    gt_pool, pred_pool, ct_pool, pcf1, pcf1_ct = [], [], [], [], []
    failed: list[str] = []
    t0 = time.perf_counter()
    for i, (ct, p) in enumerate(test, 1):
        nodes = parse_swc(p)
        if not nodes:
            # Dropped from the denominator, so it is named: n_test_cells counts
            # the cells actually scored, not the cells attempted.
            failed.append(p.name)
            continue
        gt = [nd.type for nd in nodes]
        pred = label_cell(nodes, ct, bundle)
        gt_pool.extend(gt)
        pred_pool.extend(pred)
        ct_pool.extend([ct] * len(gt))
        pcf1.append(per_cell_neurite_f1(gt, pred, ct))
        pcf1_ct.append(ct)
        if i % 200 == 0:
            print(f"  {i}/{len(test)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    gt_arr, pred_arr = np.asarray(gt_pool), np.asarray(pred_pool)
    ct_arr, pc_arr, pc_ct = np.asarray(ct_pool), np.asarray(pcf1), np.asarray(pcf1_ct)

    def pc(mask):
        # None, not 0.0: a group with no cells has no per-cell figure, and a zero
        # reads as a group the labeler failed on completely.
        v = pc_arr[mask]
        return {"pcf1_mean": float(v.mean()) if v.size else None,
                "pcf1_p10": float(np.percentile(v, 10)) if v.size else None}

    result = {"seed": args.seed, "model_dir": _display(model_dir),
              "feature_version": bundle.get("feature_version"),
              "config": bundle.get("config"),
              "n_test_cells": len(test) - len(failed), "n_cells_attempted": len(test),
              "n_failed": len(failed), "failed_files": failed,
              "n_test_nodes": int(gt_arr.size),
              "inference_min": round((time.perf_counter() - t0) / 60, 2),
              "corpus": {**node_metrics(gt_arr, pred_arr, [1, 2, 3, 4]),
                         **pc(np.ones(pc_arr.size, dtype=bool))},
              "by_cell_type": {}}
    for ct in CELL_TYPES:
        mask = ct_arr == ct
        if not mask.any():
            continue
        labels = [1, 2, 3, 4] if ct == "pyramidal" else [1, 2, 3]
        result["by_cell_type"][ct] = {"n_nodes": int(mask.sum()),
                                      **node_metrics(gt_arr[mask], pred_arr[mask], labels),
                                      **pc(pc_ct == ct)}

    lines = [f"Unstaged labeler, held-out evaluation (seed {args.seed}, GT cell type)",
             f"model : {_display(model_dir)}",
             f"test  : {len(test) - len(failed)} cells, {gt_arr.size} nodes"
             + (f"  ({len(failed)} failed to parse: {', '.join(failed[:5])}"
                f"{' ...' if len(failed) > 5 else ''})" if failed else ""), ""]
    for name, block in [("corpus", result["corpus"]), *result["by_cell_type"].items()]:
        lines += [f"[{name}]",
                  f"  accuracy {block['accuracy']:.4f}   macro-F1 {block['macro_f1']:.4f}   "
                  f"neurite-macro-F1 {block['neurite_macro_f1']:.4f}",
                  f"  pcF1 mean {block['pcf1_mean']:.4f}   pcF1 P10 {block['pcf1_p10']:.4f}"
                  if block["pcf1_mean"] is not None else "  pcF1 --   (no cells)", ""]
    report = "\n".join(lines)
    print("\n" + report)

    # A run that is not the published one never takes its file name.
    tag = f"_limit{args.limit}" if args.limit else ""
    if args.test_size != DEFAULT_TEST_SIZE:
        tag += f"_test{args.test_size:g}"
    if model_dir != (ROOT / "pipeline" / "models" / f"seed{args.seed}").resolve():
        tag += "_" + model_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"eval_unstaged_seed{args.seed}{tag}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / f"eval_unstaged_seed{args.seed}{tag}.txt").write_text(
        report + "\n", encoding="utf-8")
    print(f"wrote {_display(args.out_dir)}/eval_unstaged_seed{args.seed}{tag}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
