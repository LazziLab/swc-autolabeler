"""False-positive compartment assignment on cells that lack a class entirely.

For held-out cells whose ground truth contains no axon node, or no apical node
(pyramidal cells), the correct behavior is abstention: the pipeline should
assign little or none of the absent class. This driver runs the shipped
pipeline (GT cell type) over those subsets of one seed's test split and
reports, per absent class, how much of it the pipeline predicts anyway: the
mean and maximum predicted node share, and the fraction of cells where the
predicted share stays under 1%.

    python -m evaluation.absent_class_fp --seed 123

Output: evaluation/results/absent_class_fp_seed{S}.{json,txt}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                              # noqa: E402
from pipeline.infer_pipeline import run_pipeline_on_nodes                     # noqa: E402
from evaluation.paths import display_path                            # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed         # noqa: E402
from pipeline.split import in_test                  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "evaluation" / "results"
DEFAULT_TEST_SIZE = 0.20
#: A cell predicting at least this share of a class its ground truth does not
#: have is an inversion rather than a small leak, and is reported by name.
DEFAULT_INVERTED = 0.50


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help="held-out fraction of the hash split (default 0.20)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="default pipeline/models/seed{seed}")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--inverted-threshold", type=float, default=DEFAULT_INVERTED,
                    help="a cell whose predicted share of the absent class reaches "
                         "this is reported as inverted, by name")
    args = ap.parse_args()
    default_md = ROOT / "pipeline" / "models" / f"seed{args.seed}"
    md = (args.model_dir or default_md).resolve()
    check_model_seed(md, args.seed, args.test_size)

    # Each cell is parsed once and scored once: the selection pass keeps the
    # parsed nodes of the qualifying cells, a small fraction of the split, for
    # scoring rather than parsing them again.
    cases: dict[str, list[tuple[str, Path, list]]] = {"axon_absent": [], "apical_absent": []}
    for ct in ("pyramidal", "interneuron"):
        for p in sorted((DATA / ct).glob("*.swc")):
            if not in_test(p.name, args.seed, args.test_size):
                continue
            nodes = parse_swc(p)
            if not nodes:
                continue
            gt = {n.type for n in nodes}
            if 2 not in gt:
                cases["axon_absent"].append((ct, p, nodes))
            if ct == "pyramidal" and 4 not in gt:
                cases["apical_absent"].append((ct, p, nodes))

    out, lines = {}, []
    for case, absent_type in (("axon_absent", 2), ("apical_absent", 4)):
        shares, names, no_neurite = [], [], []
        for ct, p, nodes in cases[case]:
            pr = run_pipeline_on_nodes(
                nodes, str(p), stage1_model=md / "cell_type_classifier.pkl",
                stage2_model=md / "branch_classifier.pkl", override_cell_type=ct)
            neurite = [x for x in pr.node_labels if x in (2, 3, 4)]
            if not neurite:
                # No neurite to assign, so neither abstention nor a false
                # positive. Counted, since these cells qualified and are not in
                # the denominator below.
                no_neurite.append(p.name)
                continue
            shares.append(sum(1 for x in neurite if x == absent_type) / len(neurite))
            names.append(p.name)
        a = np.asarray(shares, dtype=float)
        # The mean is carried by a long tail of near-zero shares, so it says
        # nothing about the cells that fail outright. Those are named: this is
        # the only place the claim about a few fully inverted cells can be read.
        inverted = sorted((n for n, v in zip(names, shares) if v >= args.inverted_threshold),
                          key=str)
        out[case] = {"n_cells": len(a),
                     "n_qualified": len(cases[case]),
                     "n_without_neurite": len(no_neurite),
                     "files_without_neurite": no_neurite,
                     "mean_pred_share": float(a.mean()) if len(a) else None,
                     "max_pred_share": float(a.max()) if len(a) else None,
                     "frac_under_1pct": float((a < 0.01).mean()) if len(a) else None,
                     "inverted_threshold": args.inverted_threshold,
                     "n_inverted": len(inverted),
                     "inverted_files": inverted}
        if len(a):
            lines.append(f"{case}: n={len(a)}  mean predicted share "
                         f"{a.mean()*100:.2f}%  max {a.max()*100:.1f}%  "
                         f"under-1% cells {(a < 0.01).mean()*100:.0f}%")
            lines.append(f"  {len(inverted)} cell(s) at or above "
                         f"{args.inverted_threshold:.0%}"
                         + (f": {', '.join(inverted[:5])}"
                            f"{' ...' if len(inverted) > 5 else ''}" if inverted else ""))
            if no_neurite:
                lines.append(f"  {len(no_neurite)} qualifying cell(s) predicted no "
                             f"neurite at all and are outside the figures above")
        else:
            lines.append(f"{case}: no such cells in this test split")

    # Seed in the name: a second seed's run must not silently replace the first.
    # The same is true of a different split or a different model set, which the
    # name did not distinguish.
    tag = ""
    if args.test_size != DEFAULT_TEST_SIZE:
        tag += f"_test{args.test_size:g}"
    if md != default_md.resolve():
        tag += "_" + md.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"absent_class_fp_seed{args.seed}{tag}"
    stem.with_suffix(".json").write_text(
        json.dumps({"seed": args.seed, "test_size": args.test_size,
                    "model_dir": display_path(md), **out}, indent=2),
        encoding="utf-8")
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {display_path(stem)}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
