"""The unstaged labeler on laboratories outside the corpus.

``evaluation.generalization.eval_external`` scores the staged pipeline on cells from
archives that contributed nothing to training. This does the same for the unstaged
model of ``pipeline.unstaged``, reading the same external set, using the same metric
and reporting the same columns, so the two rows of the paper's staged-against-
unstaged comparison come from one protocol.

The model is frozen: it was fitted on the corpus training split of its seed and has
seen none of these cells and none of these laboratories. The cell type is supplied
rather than predicted, matching how the staged pipeline is scored in that table.

    python -m evaluation.generalization.eval_unstaged_external --seed 123

Input:  data/external/{pyramidal,interneuron}.csv and the matching .swc files,
        pipeline/models/seed{S}/unstaged_model.joblib
Output: evaluation/results/generalization/external_unstaged_seed{S}{tag}.{json,txt}
        and external_unstaged_percell_seed{S}{tag}.csv
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
from pipeline.unstaged import label_cell, load                            # noqa: E402
from evaluation.eval_pipeline import node_metrics                             # noqa: E402
from evaluation.paths import display_path as _display                     # noqa: E402
from evaluation.stage4._flag_metrics import per_cell_neurite_f1           # noqa: E402

EXTERNAL = ROOT / "data" / "external"
RESULTS = ROOT / "evaluation" / "results" / "generalization"


def _block(gts, preds, pcs, labels) -> dict:
    m = node_metrics(np.concatenate(gts), np.concatenate(preds), labels)
    pc = np.asarray(pcs, dtype=float)
    # Same key and same quantity as eval_external beside it: the two rows of the
    # staged-against-unstaged comparison have to measure the same thing.
    out = {"n_cells": len(pcs), "node_acc": round(m["accuracy"], 4),
           "neurite_macro_f1": round(m["neurite_macro_f1"], 4)}
    for name in ("axon", "basal", "apical"):
        if name in m["per_class"]:
            out[f"{name}_f1"] = round(m["per_class"][name]["f1"], 4)
    out["pcf1_mean"] = round(float(pc.mean()), 4)
    out["pcf1_p10"] = round(float(np.percentile(pc, 10)), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cell-type", choices=("pyramidal", "interneuron"), default="pyramidal")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="default pipeline/models/seed{seed}")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--limit", type=int, default=0, help="cells to score (0 = all; smoke tests)")
    args = ap.parse_args()

    ct = args.cell_type
    model_dir = (args.model_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")).resolve()
    bundle = load(model_dir)
    if bundle is None:
        raise SystemExit(f"MISSING unstaged_model.joblib in {model_dir}\n"
                         f"  train it: python -m pipeline.train_unstaged --seed {args.seed}")

    meta = list(csv.DictReader((EXTERNAL / f"{ct}.csv").open(encoding="utf-8", newline="")))
    if args.limit:
        meta = meta[:args.limit]
    labels = [1, 2, 3, 4] if ct == "pyramidal" else [1, 2, 3]
    print(f"unstaged model {_display(model_dir)} on {len(meta)} external {ct} cells", flush=True)

    gts, preds, rows = [], [], []
    t0 = time.perf_counter()
    for i, m in enumerate(meta, 1):
        nodes = parse_swc(EXTERNAL / ct / m["file_name"])
        if not nodes:
            continue
        gt = np.fromiter((nd.type for nd in nodes), dtype=np.int8, count=len(nodes))
        pred = np.fromiter(label_cell(nodes, ct, bundle), dtype=np.int8, count=len(nodes))
        if ct == "interneuron":
            # As eval_external: interneurons have no apical, so collapse it on
            # both sides rather than letting a stray apical prediction fall
            # outside the label set and vanish from the metrics.
            gt = np.where(gt == 4, 3, gt)
            pred = np.where(pred == 4, 3, pred)
        gts.append(gt)
        preds.append(pred)
        rows.append({"file": m["file_name"], "archive": m["archive"],
                     "n_nodes": len(gt),
                     "pcf1": per_cell_neurite_f1(gt.tolist(), pred.tolist(), ct)})
        if i % 250 == 0:
            print(f"  {i}/{len(meta)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    if not rows:
        raise SystemExit("no external cells scored")

    pcs = [r["pcf1"] for r in rows]
    out = [{"group": "ALL", **_block(gts, preds, pcs, labels)}]
    for archive in sorted({r["archive"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["archive"] == archive]
        out.append({"group": archive,
                    **_block([gts[i] for i in sel], [preds[i] for i in sel],
                             [pcs[i] for i in sel], labels)})

    tag = ("" if ct == "pyramidal" else f"_{ct}") + (f"_limit{args.limit}" if args.limit else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"external_unstaged_seed{args.seed}{tag}"
    (args.out_dir / f"{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (args.out_dir / f"external_unstaged_percell_seed{args.seed}{tag}.csv").open(
            "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"Unstaged model on external {ct} laboratories (seed {args.seed}, "
             f"{len(rows)} cells, cell type given)", "",
             f"  {'group':32s} {'cells':>6s} {'node acc':>9s} {'macroF1':>8s} "
             f"{'pcF1 mean':>10s} {'pcF1 P10':>9s}"]
    for r in out:
        lines.append(f"  {r['group']:32s} {r['n_cells']:6d} {r['node_acc']:9.4f} "
                     f"{r['neurite_macro_f1']:8.4f} {r['pcf1_mean']:10.4f} {r['pcf1_p10']:9.4f}")
    report = "\n".join(lines)
    (args.out_dir / f"{stem}.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nwrote {_display(args.out_dir / stem)}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
