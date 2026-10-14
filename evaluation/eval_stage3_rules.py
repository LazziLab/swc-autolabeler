"""What each Stage 3 topology rule contributes.

Stage 3 is rule-based, so its components can be removed individually without
retraining anything. This driver relabels the held-out cells once with every
rule active, then once per rule with that rule alone removed, and once with the
whole stage removed. The difference a rule makes is read off directly rather
than inferred from the stage as a whole.

For each rule it reports:

    dAcc, dMean, dP10   change in per-node accuracy and in per-cell neurite F1
                        (mean and 10th percentile) when the rule is removed,
                        signed so a negative number means removing the rule
                        hurts
    touched             cells whose labels change at all without the rule
    helped / harmed     of those cells, how many the rule improved and how many
                        it made worse, which separates a rule that is reliably
                        right from one that trades errors evenly

Rules are scoped: the pyramidal rules do nothing on interneurons and the
interneuron rules do nothing on pyramidal cells, so each is scored only over
the cells it can act on.

Usage:
    python -m evaluation.eval_stage3_rules [--seed 123] [--per-type 300]

Output (evaluation/results/):
    stage3_rules_seed{S}.json
    stage3_rules_seed{S}.txt

``--per-type`` defaults to the 300 cells per cell type the reported table is
built from. Any other subsample writes to ``stage3_rules_seed{S}_pertype{N}``
instead, so a quick check cannot replace the table's source file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                   # noqa: E402
from pipeline.infer_pipeline import (                                     # noqa: E402
    run_pipeline_on_nodes, load_branch_correction,
)
from pipeline.stage3.refine import RULES                                  # noqa: E402
from evaluation.eval_pipeline import node_metrics                         # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed, per_cell_neurite_f1
from pipeline.split import in_test  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "evaluation" / "results"
CELL_TYPES = ("pyramidal", "interneuron")
#: Cells per cell type behind the reported table. A run at this size owns the
#: canonical file name; any other size is tagged.
CANONICAL_PER_TYPE = 300
# Which cell type each rule can affect, from the branches in refine().
SCOPE = {"subtree_voting": "pyramidal", "single_axon": "pyramidal",
         "apical_principal_axis": "pyramidal", "apical_owner": "pyramidal",
         "axon_bias": "interneuron", "soft_majority": "interneuron"}
# A rule refine gained but this table has no scope for would be silently scored
# over both cell types, which would dilute its helped/harmed counts.
_unscoped = set(RULES) - set(SCOPE)
if _unscoped:
    raise SystemExit(f"no cell-type scope recorded for Stage 3 rule(s) {sorted(_unscoped)}; "
                     f"add them to SCOPE in {Path(__file__).name}")


def _score(gts, preds, pcs) -> dict:
    mm = node_metrics(np.concatenate(gts), np.concatenate(preds), [1, 2, 3, 4])
    pc = np.asarray(pcs, dtype=float)
    return {"acc": mm["accuracy"], "mean": float(pc.mean()),
            "p10": float(np.percentile(pc, 10)),
            "apical_f1": mm["per_class"]["apical"]["f1"],
            "axon_f1": mm["per_class"]["axon"]["f1"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--per-type", type=int, default=CANONICAL_PER_TYPE,
                    help=f"evenly spaced subsample of this many cells per cell type; "
                         f"0 uses every held-out cell. The default {CANONICAL_PER_TYPE} is "
                         f"the size the reported table is built from and owns the canonical "
                         f"file name; any other size is tagged into its own file")
    args = ap.parse_args()
    # A smaller check must not overwrite the file the reported table is read from.
    tag = ("" if args.per_type == CANONICAL_PER_TYPE
           else f"_pertype{args.per_type}" if args.per_type else "_pertypeall")

    mdir = ROOT / "pipeline" / "models" / f"seed{args.seed}"
    s1, s2 = mdir / "cell_type_classifier.pkl", mdir / "branch_classifier.pkl"
    # Refuse a model trained on a different split than the one being scored.
    check_model_seed(mdir, args.seed, args.test_size)
    if load_branch_correction(mdir) is None:
        raise SystemExit(f"MISSING branch_correction.joblib in {mdir}")

    cells = []
    for ct in CELL_TYPES:
        held = [p for p in sorted((DATA / ct).glob("*.swc"))
                if in_test(p.name, args.seed, args.test_size)]
        if args.per_type and len(held) > args.per_type:
            # Even stride rather than the first N, so the subsample spans the
            # corpus's sources and sizes instead of one alphabetical block.
            held = held[::max(1, len(held) // args.per_type)][:args.per_type]
        cells += [(ct, p) for p in held]
    print(f"Stage 3 rule ablation: {len(cells)} test cells, seed {args.seed}", flush=True)

    configs = [("full", frozenset())]
    configs += [(r, frozenset({r})) for r in RULES]
    configs.append(("no_stage3", frozenset(RULES)))

    # Cells outer, configurations inner: a cell is parsed once and released with
    # the next iteration. Holding every parsed cell so the configurations could
    # share them cost 406 bytes a node, several gigabytes over the held-out set.
    per_cell: dict[str, list[float]] = {name: [] for name, _ in configs}
    pooled: dict[str, dict[str, list]] = {name: {"gts": [], "preds": []} for name, _ in configs}
    t0 = time.perf_counter()
    for i, (ct, p) in enumerate(cells, 1):
        nodes = parse_swc(p)
        gt = np.fromiter((x.type for x in nodes), dtype=np.int8, count=len(nodes))
        for name, disable in configs:
            pr = run_pipeline_on_nodes(nodes, str(p), s1, s2,
                                       override_cell_type=ct, disable_stage3=disable)
            lab = np.fromiter(pr.node_labels, dtype=np.int8, count=len(nodes))
            pooled[name]["gts"].append(gt)          # one gt array shared by every config
            pooled[name]["preds"].append(lab)
            per_cell[name].append(per_cell_neurite_f1(gt, lab, ct))
        if i % 50 == 0:
            print(f"  ... {i}/{len(cells)} cells ({(time.perf_counter()-t0)/60:.1f} min)",
                  flush=True)

    scores: dict[str, dict] = {}
    for name, _ in configs:
        scores[name] = _score(pooled[name]["gts"], pooled[name]["preds"], per_cell[name])
        print(f"  {name:22s} acc {scores[name]['acc']:.4f}  mean {scores[name]['mean']:.4f}"
              f"  P10 {scores[name]['p10']:.4f}", flush=True)

    cell_types = [ct for ct, _ in cells]
    base = np.asarray(per_cell["full"])
    out = {"seed": args.seed, "n_cells": len(cells), "scores": scores, "rules": {}}

    L = [f"Stage 3 rule ablation, seed {args.seed}, {len(cells)} held-out cells", "",
         "Each row removes one rule and leaves the rest in place. Negative deltas",
         "mean the rule helps. Scoped columns count only the cells the rule can act on.",
         "", f"  {'rule':22s} {'scope':>12s} {'dAcc':>8s} {'dMean':>8s} {'dP10':>8s}"
         f" {'touched':>8s} {'helped':>7s} {'harmed':>7s}"]

    for name in list(RULES) + ["no_stage3"]:
        cur = np.asarray(per_cell[name])
        scope = SCOPE.get(name)
        mask = (np.array([c == scope for c in cell_types]) if scope
                else np.ones(len(cells), bool))
        changed = (cur != base) & mask
        helped = int(((base > cur) & changed).sum())    # removing it lowered F1 -> rule helped
        harmed = int(((base < cur) & changed).sum())    # removing it raised F1 -> rule hurt
        d = {k: scores[name][k] - scores["full"][k] for k in scores["full"]}
        out["rules"][name] = {"scope": scope or "both", "n_in_scope": int(mask.sum()),
                              "delta": d, "touched": int(changed.sum()),
                              "helped": helped, "harmed": harmed}
        L.append(f"  {name:22s} {(scope or 'both'):>12s} {d['acc']:+8.4f} {d['mean']:+8.4f}"
                 f" {d['p10']:+8.4f} {int(changed.sum()):8d} {helped:7d} {harmed:7d}")

    L += ["", "Absolute scores", "",
          f"  {'config':22s} {'acc':>8s} {'mean':>8s} {'P10':>8s} {'axon F1':>8s} {'apical F1':>10s}"]
    for name, _ in configs:
        s = scores[name]
        L.append(f"  {name:22s} {s['acc']:8.4f} {s['mean']:8.4f} {s['p10']:8.4f}"
                 f" {s['axon_f1']:8.4f} {s['apical_f1']:10.4f}")

    report = "\n".join(L)
    print("\n" + report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"stage3_rules_seed{args.seed}{tag}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    (RESULTS / f"stage3_rules_seed{args.seed}{tag}.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nwrote evaluation/results/stage3_rules_seed{args.seed}{tag}.{{json,txt}} "
          f"({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
