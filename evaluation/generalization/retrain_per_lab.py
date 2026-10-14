"""Train a fresh model on one external laboratory's own cells and compare it
with the frozen pooled model on the same held-out cells.

The pooled model transfers unevenly to unseen laboratories; this measures the
alternative a laboratory actually has: point the training recipe at its own
corpus. Each archive's cells are staged as a corpus of their own, the standard
trainer fits the labeling stages from scratch on a two-thirds split (Stage 1 is
skipped because a single-cell-type corpus needs no routing, and the flag is
skipped), and both models are evaluated on the held-out third with the cell
type given. Nothing is inherited from the pooled model.

An archive without axon branches cannot fit the three-class branch-correction
head; the trainer's failure there is tolerated and the lab model runs without
correction, which the pipeline supports.

Usage:
    python -m evaluation.generalization.retrain_per_lab [--archives A B ...]
    python -m evaluation.generalization.retrain_per_lab --device cpu \
        --out evaluation/results/generalization/lab_retrain_cpu.json

Output (evaluation/results/generalization/):
    lab_retrain.json
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                   # noqa: E402
from pipeline.infer_pipeline import run_pipeline_on_nodes                     # noqa: E402
from evaluation.eval_pipeline import node_metrics                             # noqa: E402
from evaluation.paths import display_path                                 # noqa: E402
from evaluation.stage4._flag_metrics import per_cell_neurite_f1           # noqa: E402

EXTERNAL = ROOT / "data" / "external"
RESULTS = ROOT / "evaluation" / "results" / "generalization"
WORK = RESULTS / "lab_retrain_work"
DEFAULT_ARCHIVES = ("vdHeuvel", "Hamad", "Bianchi")
DEFAULT_SEED = 123
#: A two-thirds training split, as the docstring says.
DEFAULT_TEST_SIZE = 1.0 / 3.0


def evaluate(files: list[str], mdir: Path) -> dict:
    """Score one model set on the given external pyramidal cells.

    ``s1`` is the Stage 1 classifier, which a laboratory model set does not have:
    it is trained with ``--skip stage1``, since a single-cell-type corpus needs no
    routing. The path is passed anyway and never read, because every cell here is
    labeled under an overridden cell type.
    """
    s1, s2 = mdir / "cell_type_classifier.pkl", mdir / "branch_classifier.pkl"
    gts, preds, pc = [], [], []
    for f in files:
        path = EXTERNAL / "pyramidal" / f
        nodes = parse_swc(path)
        pr = run_pipeline_on_nodes(nodes, str(path), s1, s2, override_cell_type="pyramidal")
        gt = np.array([n.type for n in nodes])
        lab = np.array(list(pr.node_labels))
        gts.append(gt)
        preds.append(lab)
        pc.append(per_cell_neurite_f1(gt, list(lab), "pyramidal"))
    gt_all, pred_all = np.concatenate(gts), np.concatenate(preds)
    mm = node_metrics(gt_all, pred_all, [1, 2, 3, 4])
    # Node accuracy over neurite nodes only. node_metrics counts the soma, which
    # the pipeline places by geometry and always gets right, so including it
    # reports a higher accuracy than every other table in the paper, which
    # excludes the soma throughout.
    neurite = gt_all != 1
    pc = np.array(pc)
    return {"node_acc": float((gt_all[neurite] == pred_all[neurite]).mean()),
            "node_acc_with_soma": mm["accuracy"],
            "neurite_macro_f1": mm["neurite_macro_f1"],
            "apical_f1": mm["per_class"]["apical"]["f1"],
            "basal_f1": mm["per_class"]["basal"]["f1"],
            "pcf1_mean": float(pc.mean()),
            "pcf1_p10": float(np.percentile(pc, 10))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archives", nargs="+", default=list(DEFAULT_ARCHIVES))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="split seed, and the frozen model set compared against")
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help="held-out fraction of each laboratory's own cells")
    ap.add_argument("--out", type=Path, default=None,
                    help="default evaluation/results/generalization/lab_retrain.json")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="where the laboratory models' Stage 2 is fitted (see train_pipeline)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the staged corpora and laboratory models under "
                         "lab_retrain_work/ instead of removing them")
    args = ap.parse_args()
    out_path = args.out or (RESULTS / "lab_retrain.json")

    with (EXTERNAL / "pyramidal.csv").open(encoding="utf-8", newline="") as fh:
        meta = list(csv.DictReader(fh))
    known = {(m["archive"] or "").strip() for m in meta}
    unknown = [a for a in args.archives if a not in known]
    if unknown:
        # A misspelled archive stops the run: skipping it would leave a partial
        # result file standing in for the published one.
        raise SystemExit(f"no cells for {', '.join(unknown)} in the external set.\n"
                         f"  archives present: {', '.join(sorted(known))}")
    frozen_dir = ROOT / "pipeline" / "models" / f"seed{args.seed}"
    # Archives this run did not touch keep the results an earlier run recorded, so
    # a run over one laboratory cannot replace the table with a single row.
    results = {}
    if out_path.is_file():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        results.pop("_run", None)
    failed: dict[str, str] = {}
    for arch in args.archives:
        files = [m["file_name"] for m in meta if m["archive"] == arch]
        corpus = WORK / "corpora" / arch
        (corpus / "pyramidal").mkdir(parents=True, exist_ok=True)
        (corpus / "interneuron").mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = corpus / "pyramidal" / f
            if not dst.exists():
                shutil.copy2(EXTERNAL / "pyramidal" / f, dst)
        mdir = WORK / "models" / arch
        # Start from an empty model dir, so a failed retrain can never be scored on
        # the model an earlier run left behind.
        shutil.rmtree(mdir, ignore_errors=True)

        print(f"\n=== {arch}: training on ~{int(len(files) * (1 - args.test_size))} cells ===",
              flush=True)
        t0 = time.perf_counter()
        run = subprocess.run(
            [sys.executable, "-m", "pipeline.train_pipeline",
             "--data-dir", str(corpus), "--seed", str(args.seed),
             "--test-size", str(args.test_size), "--out-dir", str(mdir),
             "--skip", "stage1,flag", "--device", args.device],
            cwd=str(ROOT), capture_output=True, text=True)
        # Wall clock of the whole training command, process start included.
        train_s = time.perf_counter() - t0
        essential = [mdir / "branch_classifier.pkl"]
        # A corpus with no axon branches cannot fit the three-class correction
        # head, and the trainer exits non-zero when that happens. The labeling
        # stages are still complete, so that one case is the tolerated failure
        # this file's docstring describes, not a broken retrain.
        no_axon_head = (run.returncode != 0
                        and all(q.is_file() for q in essential)
                        and not (mdir / "branch_correction.joblib").is_file()
                        and "Invalid classes inferred" in run.stderr)
        if not no_axon_head and (run.returncode != 0
                                 or not all(q.is_file() for q in essential)):
            tail = "\n".join(run.stderr.splitlines()[-10:])
            print(f"TRAIN FAILED:\n{tail}")
            # Recorded, not merely skipped, so an incomplete run is never
            # mistaken for a complete one in the result file.
            failed[arch] = tail
            results.pop(arch, None)
            continue
        note = ""
        if not (mdir / "branch_correction.joblib").is_file():
            note = "correction head untrainable on this corpus; lab model runs without it"
            print(f"  note: {note}")

        split = json.loads((mdir / "eval_split.json").read_text(encoding="utf-8"))
        test_files = [Path(f).name for f in split["test_files"]["pyramidal"]]
        own = evaluate(test_files, mdir)
        frozen = evaluate(test_files, frozen_dir)
        results[arch] = {"n_train": len(files) - len(test_files), "train_s": round(train_s, 2),
                         "n_test": len(test_files), "own": own, "frozen": frozen}
        if note:
            results[arch]["note"] = note
        print(f"{arch}  n_test={len(test_files)}  trained in {train_s:.1f} s on {args.device}")
        for k in ("node_acc", "apical_f1", "pcf1_mean", "pcf1_p10"):
            print(f"  {k:10s} frozen {frozen[k]:.3f}  ->  own {own[k]:.3f}", flush=True)

    results["_run"] = {"archives": list(args.archives), "seed": args.seed,
                       "test_size": args.test_size, "device": args.device,
                       "frozen_models": display_path(frozen_dir),
                       "failed": failed}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not args.keep_work:
        # The staged corpora and laboratory model sets are inputs to this run, not
        # results; they were left inside evaluation/results at about 60 MB. Only
        # this run's archives are removed: new_class stages its own corpora under
        # the same directory, and they are not this script's to delete.
        for arch in args.archives:
            shutil.rmtree(WORK / "corpora" / arch, ignore_errors=True)
            shutil.rmtree(WORK / "models" / arch, ignore_errors=True)
    print(f"\nwrote {display_path(out_path)}"
          + (f"  ({len(failed)} archive(s) failed to train)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
