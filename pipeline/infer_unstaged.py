"""Label an SWC neuron with the unstaged model.

The staged counterpart is ``pipeline.infer_pipeline``. This reads the same file the
same way, decides the cell type with Stage 1 exactly as the staged command does,
and then labels every branch in one pass with the model
``pipeline.train_unstaged`` fitted, with no topology rules and no branch
correction. There is no quality flag: only the staged pipeline carries one.

    python -m pipeline.infer_unstaged input.swc -o output.swc
    python -m pipeline.infer_unstaged input.swc --cell-type pyramidal

As with the staged command, only the type column of the input is rewritten; every
coordinate, radius, parent link and comment line is copied through unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.infer_pipeline import (                                     # noqa: E402
    DEFAULT_MODEL_DIR, parse_swc_for_labeling, write_labeled_swc,
)
from pipeline.stage1.cell_type_detector import detect_cell_type_from_nodes  # noqa: E402
from pipeline.unstaged import CELL_TYPES, label_cell, load                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Auto-label an SWC neuron with the unstaged model "
                    "(one model per cell type over the whole feature set).")
    ap.add_argument("swc", type=Path, help="input SWC file")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output SWC path (default: <input>.labeled.swc)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="model directory (default: the shipped seed123)")
    ap.add_argument("--cell-type", choices=CELL_TYPES, default=None,
                    help="skip Stage 1 and use this cell type")
    ap.add_argument("--force", action="store_true",
                    help="label the file even if it fails the Stage 0 quality checks")
    args = ap.parse_args()

    bundle = load(args.model_dir)
    if bundle is None:
        raise SystemExit(
            f"No unstaged model in {args.model_dir}. Train one with "
            f"`python -m pipeline.train_unstaged --seed <S>`, or use "
            f"`python -m pipeline.infer_pipeline` for the staged labeler.")

    nodes, notes = parse_swc_for_labeling(args.swc)
    if not nodes:
        raise SystemExit(f"No nodes parsed from {args.swc}")
    for note in notes:
        print(f"note: {note}", flush=True)

    from pipeline.qc import qc_violations  # noqa: PLC0415
    failed = qc_violations(nodes)
    if failed:
        msg = f"{args.swc.name} fails the Stage 0 quality checks: " + "; ".join(failed)
        if not args.force:
            raise SystemExit(msg + "\nPass --force to label it anyway.")
        print(f"warning: {msg} (labeling anyway, --force)", flush=True)

    if args.cell_type:
        cell_type, how = args.cell_type, "given"
    else:
        s1 = detect_cell_type_from_nodes(nodes, args.model_dir / "cell_type_classifier.pkl")
        cell_type, how = s1.cell_type, f"Stage 1, confidence {s1.confidence:.2f}"

    labels = label_cell(nodes, cell_type, bundle)
    out = args.out or args.swc.with_suffix(".labeled.swc")
    n_written = write_labeled_swc(args.swc, nodes, labels, out)
    print(f"{args.swc.name}: cell_type={cell_type} ({how}) nodes={n_written} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
