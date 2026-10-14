"""Frozen-model accuracy on the external validation set.

Runs the shipped pipeline, unchanged, on cells from laboratories that
contributed nothing to training, and reports the same metrics as the in-corpus
evaluation so the two are directly comparable: per-node accuracy and per-class
F1 over the classes present in the ground truth, plus per-cell neurite F1 mean
and 10th percentile. Results are grouped per archive, per species, and for the
subset of cells whose ground truth includes an axon.

Two configurations. The default is deployment: Stage 1 predicts the cell type,
and its accuracy on each archive is reported alongside. ``--gt-ct`` overrides
Stage 1 with the known cell type, isolating how well the labeling stages
themselves transfer.

For archives whose ground truth carries no axon, the per-archive report also
gives the share of nodes the model predicted as axon. A value near zero means
the model abstains on cells whose axon was not traced rather than inventing
one.

``--cell-type`` selects the pyramidal (default) or interneuron external set;
interneuron ground truth is scored over {axon, dendrite}, with stray apical
codes collapsed to dendrite and no apical F1 reported.

``--no-rules`` disables every Stage 3 topology rule (output tag ``_norules``)
and ``--out-dir`` redirects the output, so an ablated run never overwrites the
shipped-configuration files.

Usage:
    python -m evaluation.generalization.eval_external [--seed 123] [--gt-ct]
        [--cell-type interneuron]

Output (evaluation/results/generalization/):
    external_eval_seed{S}[_interneuron][_gtct].json
    external_percell_seed{S}[_interneuron][_gtct].csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                   # noqa: E402
from pipeline.infer_pipeline import (                                     # noqa: E402
    run_pipeline_on_nodes, load_branch_correction,
)
from evaluation.eval_pipeline import node_metrics                             # noqa: E402
from evaluation.stage4._flag_metrics import per_cell_neurite_f1           # noqa: E402
from pipeline.stage3.refine import RULES                                  # noqa: E402

EXTERNAL = ROOT / "data" / "external"
RESULTS = ROOT / "evaluation" / "results" / "generalization"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gt-ct", action="store_true",
                    help="bypass Stage 1 with the known cell type")
    ap.add_argument("--cell-type", default="pyramidal",
                    choices=("pyramidal", "interneuron"))
    ap.add_argument("--no-rules", action="store_true",
                    help="disable every Stage 3 topology rule; output tag _norules")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    ct = args.cell_type
    disable = frozenset(RULES) if args.no_rules else frozenset()

    mdir = ROOT / "pipeline" / "models" / f"seed{args.seed}"
    s1, s2 = mdir / "cell_type_classifier.pkl", mdir / "branch_classifier.pkl"
    # No check_model_seed here, unlike the in-corpus drivers: these cells come
    # from laboratories that contributed nothing to training, so no seed's
    # held-out split has any bearing on them.
    if load_branch_correction(mdir) is None:
        raise SystemExit(f"MISSING branch_correction.joblib in {mdir}")

    meta = list(csv.DictReader((EXTERNAL / f"{ct}.csv").open(encoding="utf-8", newline="")))
    print(f"external set ({ct}): {len(meta)} cells, seed {args.seed}"
          f"{', GT cell type' if args.gt_ct else ''}", flush=True)

    recs, t0 = [], time.perf_counter()
    for i, m in enumerate(meta, 1):
        nodes = parse_swc(EXTERNAL / ct / m["file_name"])
        if not nodes:
            continue
        path = EXTERNAL / ct / m["file_name"]
        pr = run_pipeline_on_nodes(
            nodes, str(path), s1, s2,
            override_cell_type=ct if args.gt_ct else None, disable_stage3=disable)
        gt = np.fromiter((n.type for n in nodes), dtype=np.int8, count=len(nodes))
        pred = np.fromiter(pr.node_labels, dtype=np.int8, count=len(nodes))
        if ct == "interneuron":
            # Interneurons have no apical: collapse it on both sides, so a stray
            # apical prediction counts as a dendrite error rather than falling
            # outside the label set and vanishing from the confusion matrix.
            gt = np.where(gt == 4, 3, gt)
            pred = np.where(pred == 4, 3, pred)
        recs.append({"m": m, "gt": gt, "pred": pred,
                     "pred_ct": pr.stage1.cell_type,
                     "pcf1": per_cell_neurite_f1(gt, list(pred), ct)})
        if i % 250 == 0:
            print(f"  {i}/{len(meta)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    def block(rs: list[dict], name: str) -> dict:
        labels = [1, 2, 3, 4] if ct == "pyramidal" else [1, 2, 3]
        gt = np.concatenate([r["gt"] for r in rs])
        pred = np.concatenate([r["pred"] for r in rs])
        mm = node_metrics(gt, pred, labels)
        pc = np.array([r["pcf1"] for r in rs])
        axon_rs = [r for r in rs if r["m"]["has_axon_gt"] == "1"]
        # Axon F1 over the cells that actually have an axon traced. Pooling in
        # cells whose ground truth has none would dilute it with nodes that can
        # never be a true positive.
        axon_f1 = None
        if axon_rs:
            am = node_metrics(np.concatenate([r["gt"] for r in axon_rs]),
                              np.concatenate([r["pred"] for r in axon_rs]), labels)
            axon_f1 = round(am["per_class"]["axon"]["f1"], 4)
        has_axon_gt = bool(axon_rs)
        row = {"group": name, "n_cells": len(rs),
               "n_cells_axon_gt": len(axon_rs),
               "stage1_acc": round(float(np.mean(
                   [r["pred_ct"] == ct for r in rs])), 4),
               "node_acc": round(mm["accuracy"], 4),
               # Neurite macro-F1, soma excluded, as every other table reports
               # and as Section 4 defines it. Including soma, which is trivially
               # identified at F1 near 1.0, would lift every figure here and is
               # not comparable with the unstaged scorer beside it.
               "neurite_macro_f1": round(mm["neurite_macro_f1"], 4),
               "axon_f1": axon_f1,
               "basal_f1": round(mm["per_class"]["basal"]["f1"], 4),
               "apical_f1": (round(mm["per_class"]["apical"]["f1"], 4)
                             if ct == "pyramidal" else None),
               "pcf1_mean": round(float(pc.mean()), 4),
               "pcf1_p10": round(float(np.percentile(pc, 10)), 4)}
        if not has_axon_gt:
            # No axon in the ground truth: report how much axon the model
            # predicted anyway instead of a vacuous F1.
            row["pred_axon_share"] = round(float(np.mean(
                [(r["pred"] == 2).mean() for r in rs])), 4)
        return row

    out = [block(recs, "ALL")]
    for arch in sorted({r["m"]["archive"] for r in recs}):
        out.append(block([r for r in recs if r["m"]["archive"] == arch], arch))
    for sp in sorted({r["m"]["species"] for r in recs}):
        out.append(block([r for r in recs if r["m"]["species"] == sp], f"sp:{sp}"))
    axon_rs = [r for r in recs if r["m"]["has_axon_gt"] == "1"]
    if axon_rs:
        out.append(block(axon_rs, "axon-GT-only"))

    cols = ["group", "n_cells", "stage1_acc", "node_acc", "neurite_macro_f1",
            "axon_f1", "basal_f1", "apical_f1", "pcf1_mean", "pcf1_p10"]
    print()
    print("  ".join(f"{c:>10s}" if c != "group" else f"{c:28s}" for c in cols))
    for row in out:
        cells = [row["group"] if c == "group" else
                 ("--" if row.get(c) is None else row[c]) for c in cols]
        print("  ".join(f"{v:>10}" if c != "group" else f"{str(v):28s}"
                        for c, v in zip(cols, cells)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = (("" if ct == "pyramidal" else f"_{ct}") + ("_gtct" if args.gt_ct else "")
           + ("_norules" if args.no_rules else ""))
    with (args.out_dir / f"external_eval_seed{args.seed}{tag}.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with (args.out_dir / f"external_percell_seed{args.seed}{tag}.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "archive", "species", "pred_ct", "pcf1"])
        for r in recs:
            w.writerow([r["m"]["file_name"], r["m"]["archive"], r["m"]["species"],
                        r["pred_ct"], f"{r['pcf1']:.6f}"])
    print(f"\nwrote {args.out_dir / f'external_eval_seed{args.seed}{tag}.json'} "
          f"({(time.perf_counter() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
