"""Full-pipeline evaluation: per-node labeling accuracy (the paper's headline).

Runs the complete pipeline on the seed's held-out test cells with Stage 1
bypassed (each cell's ground-truth cell type is passed as ``override_cell_type``,
so Stages 2 and 3 are measured fairly, the same way baselines receive GT cell
type). Reports pooled per-node accuracy, per-class precision/recall/F1, macro-F1
(over GT-present classes), and a per-cell-type breakdown -- the numbers the
paper's main results table reports.

The test split is the deterministic hash-bucket split used throughout: a file
lands in test iff md5("{seed}:{filename}") maps below the test fraction.

``--source-prefix`` narrows the scoring to one corpus source without touching the
split, so the slice is a strict subset of the same held-out cells and its numbers
stay directly comparable to the all-corpus run.

``--deployment`` lets Stage 1 predict the cell type instead, the paper's
deployment configuration, and ``--no-rules`` disables every Stage 3 topology
rule; each adds a tag to the output name so the headline files are never
overwritten.

Usage:
    python -m evaluation.eval_pipeline --seed 123
    python -m evaluation.eval_pipeline --seed 123 --source-prefix hpf_ca1__

Output (evaluation/results/, created if missing), with the source appended when
one is selected:
    eval_pipeline_seed{seed}.json
    eval_pipeline_seed{seed}.txt
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
from pipeline.infer_pipeline import run_pipeline_on_nodes, load_branch_correction  # noqa: E402
from pipeline.stage3.refine import RULES                                 # noqa: E402
from evaluation.paths import display_path as _display_path              # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed, per_cell_neurite_f1  # noqa: E402
from pipeline.split import in_test as _in_test

CELL_TYPES = ("pyramidal", "interneuron")
DATA = ROOT / "data"
LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal", 4: "apical"}
NEURITE = (2, 3, 4)


def node_metrics(gt: np.ndarray, pred: np.ndarray, labels: list[int]) -> dict:
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, precision_recall_fscore_support,
    )
    p, r, f, sup = precision_recall_fscore_support(
        gt, pred, labels=labels, zero_division=0,
    )
    per_class = {
        LABEL_NAMES[l]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(sup[i])}
        for i, l in enumerate(labels)
    }
    # Macro F1 over classes present in the ground truth (paper convention).
    present = [i for i, l in enumerate(labels) if int((gt == l).sum()) > 0]
    macro = float(np.mean([f[i] for i in present])) if present else 0.0
    neu = [i for i, l in enumerate(labels) if l in NEURITE and int((gt == l).sum()) > 0]
    neurite_macro = float(np.mean([f[i] for i in neu])) if neu else 0.0
    return {
        "accuracy": float(accuracy_score(gt, pred)),
        "macro_f1": macro,
        "neurite_macro_f1": neurite_macro,
        "per_class": per_class,
        "confusion": confusion_matrix(gt, pred, labels=labels).tolist(),
        "confusion_labels": [LABEL_NAMES[l] for l in labels],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="default pipeline/models/seed{seed}")
    ap.add_argument("--source-prefix", default="",
                    help="score only test cells whose filename starts with this "
                         "prefix, e.g. 'hpf_ca1__' for the single-source lab corpus; "
                         "the split itself is unchanged, so the slice is a subset of "
                         "the same held-out cells")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results")
    ap.add_argument("--deployment", action="store_true",
                    help="let Stage 1 predict the cell type instead of overriding it "
                         "with the ground truth; output tag _deploy")
    ap.add_argument("--no-rules", action="store_true",
                    help="disable every Stage 3 topology rule; output tag _norules")
    args = ap.parse_args()
    # Output suffix names the slice and the configuration, so a filtered or
    # ablated run never overwrites the full one.
    tag = ((f"_{args.source_prefix.rstrip('_')}" if args.source_prefix else "")
           + ("_deploy" if args.deployment else "") + ("_norules" if args.no_rules else ""))
    disable = frozenset(RULES) if args.no_rules else frozenset()

    model_dir = args.model_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")
    check_model_seed(model_dir, args.seed, args.test_size)
    # Stage 1 is never consulted here: every cell is scored with override_cell_type,
    # so the pipeline short-circuits before loading it. Not requiring the file lets
    # a model trained on a single-class corpus, which cannot have a Stage 1, be
    # scored with the same driver.
    s1_path = model_dir / "cell_type_classifier.pkl"
    s2_path = model_dir / "branch_classifier.pkl"
    for p in ((s1_path, s2_path) if args.deployment else (s2_path,)):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")
    if load_branch_correction(model_dir) is None:           # correction is intrinsic; assert it ships
        raise SystemExit(f"MISSING branch_correction.joblib in {model_dir}")

    test = [(ct, p) for ct in CELL_TYPES
            for p in sorted((DATA / ct).glob("*.swc"))
            if _in_test(p.name, args.seed, args.test_size)
            and p.name.startswith(args.source_prefix)]
    if not test:
        raise SystemExit(f"no test cells matched --source-prefix {args.source_prefix!r}")
    print(f"seed {args.seed}: {len(test)} test cells"
          f"{f' (source-prefix {args.source_prefix})' if args.source_prefix else ''}",
          flush=True)

    gt_pool: list[int] = []
    pred_pool: list[int] = []
    ct_pool: list[str] = []
    pcf1: list[float] = []
    pcf1_ct: list[str] = []
    skipped: list[str] = []
    n_corrected = 0            # cells the branch correction ran on: pyramidal only
    t0 = time.perf_counter()
    for i, (ct, p) in enumerate(test, 1):
        nodes = parse_swc(p)
        if not nodes:
            skipped.append(p.name)
            continue
        gt = [n.type for n in nodes]
        pr = run_pipeline_on_nodes(
            nodes, file_path=str(p), stage1_model=s1_path, stage2_model=s2_path,
            override_cell_type=None if args.deployment else ct,
            disable_stage3=disable,
        )
        if pr.stage1.cell_type == "pyramidal":
            n_corrected += 1
        gt_pool.extend(gt)
        pred_pool.extend(pr.node_labels)
        ct_pool.extend([ct] * len(gt))
        pcf1.append(per_cell_neurite_f1(gt, list(pr.node_labels), ct))
        pcf1_ct.append(ct)
        if i % 200 == 0:
            print(f"  ... {i}/{len(test)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    elapsed = (time.perf_counter() - t0) / 60.0

    gt_arr = np.asarray(gt_pool)
    pred_arr = np.asarray(pred_pool)
    ct_arr = np.asarray(ct_pool)
    pcf1_arr = np.asarray(pcf1)
    pcf1_ct_arr = np.asarray(pcf1_ct)

    def _pcf1(mask) -> dict:
        v = pcf1_arr[mask]
        return {"pcf1_mean": float(v.mean()) if v.size else 0.0,
                "pcf1_p10": float(np.percentile(v, 10)) if v.size else 0.0}

    result = {
        "seed": args.seed,
        "source_prefix": args.source_prefix,
        # Cells actually scored, which is every selected cell unless one failed
        # to parse; those are named rather than silently dropped from the count.
        "n_test_cells": len(pcf1),
        "n_test_cells_selected": len(test),
        "n_test_nodes": int(gt_arr.size),
        "inference_min": round(elapsed, 2),
        # The correction's presence is asserted above, so this is a corrected run
        # by construction. ``report.aggregate_pipeline`` reads this flag and refuses
        # a result file without it.
        "correction_applied": True,
        # It runs on pyramidal cells only, so the count is a property of the
        # scored set rather than of the configuration.
        "correction_applied_cells": n_corrected,
        "deployment": args.deployment,
        "skipped_unparseable": skipped,
        "rules_disabled": sorted(disable),
        "corpus": {**node_metrics(gt_arr, pred_arr, [1, 2, 3, 4]),
                   **_pcf1(np.ones(pcf1_arr.size, dtype=bool))},
        "by_cell_type": {},
    }
    for ct in CELL_TYPES:
        mask = ct_arr == ct
        if not mask.any():          # a source filter can leave a cell type unrepresented
            continue
        labels = [1, 2, 3, 4] if ct == "pyramidal" else [1, 2, 3]
        result["by_cell_type"][ct] = {
            "n_nodes": int(mask.sum()),
            **node_metrics(gt_arr[mask], pred_arr[mask], labels),
            **_pcf1(pcf1_ct_arr == ct),
        }

    def _fmt_block(name: str, m: dict, extra: str = "") -> list[str]:
        out = [f"[{name}] {extra}".rstrip(),
               f"  accuracy {m['accuracy']:.4f}   macro-F1 {m['macro_f1']:.4f}   "
               f"neurite-macro-F1 {m['neurite_macro_f1']:.4f}",
               f"  pcF1 mean {m.get('pcf1_mean', 0):.4f}   pcF1 P10 {m.get('pcf1_p10', 0):.4f}",
               f"  {'class':14s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'support':>9s}"]
        for cname, c in m["per_class"].items():
            out.append(f"  {cname:14s} {c['precision']:7.4f} {c['recall']:7.4f} "
                       f"{c['f1']:7.4f} {c['support']:9d}")
        return out

    lines = [f"Full-pipeline evaluation (seed {args.seed}, "
             f"{'deployment, Stage 1 routing' if args.deployment else 'GT cell type'}, "
             f"correction=intrinsic{', topology rules off' if args.no_rules else ''}"
             f"{f', source-prefix {args.source_prefix}' if args.source_prefix else ''})",
             f"model : {_display_path(model_dir)}",
             f"test  : {len(test)} cells, {gt_arr.size} nodes  ({elapsed:.1f} min)", ""]
    lines += _fmt_block("corpus", result["corpus"])
    for ct, block in result["by_cell_type"].items():
        lines += ["", *_fmt_block(ct, block, f"n_nodes={block['n_nodes']}")]
    report = "\n".join(lines)
    print("\n" + report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"eval_pipeline_seed{args.seed}{tag}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / f"eval_pipeline_seed{args.seed}{tag}.txt").write_text(
        report + "\n", encoding="utf-8")
    print(f"\nwrote {args.out_dir}/eval_pipeline_seed{args.seed}{tag}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
