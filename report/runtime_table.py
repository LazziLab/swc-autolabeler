"""Compile the computational cost of the pipeline into the paper's table.

Three parts:

  A  Training. Per-stage wall-clock of one timed end-to-end training run on
     the full corpus, the ``train_manifest.json`` of that run.
  B  Inference. Per-cell time and memory from ``evaluation.profile_runtime``:
     each component on its size-bucket sample, and the labeling command end
     to end on its random sample. Inference runs on the CPU alone, so there is
     one profile.
  C  Morphology size. Runtime per cell against node count, and labeling
     accuracy against node count, so that "does it scale" is answered for both
     cost and quality.

Part C reads the per-cell F1 tables written by ``evaluation.eval_ablation``,
which carry each cell's node count.

    python -m report.runtime_table
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
REPORT = Path(__file__).resolve().parent
OUT_TXT = REPORT / "runtime_table.txt"
OUT_JSON = REPORT / "runtime_table.json"

SEEDS = (42, 123, 789)
STAGES = ("stage1", "stage2", "correction", "flag")
STAGE_LABEL = {"stage1": "Stage 1 cell type", "stage2": "Stage 2 subtree classifier",
               "correction": "Stage 3 branch correction", "flag": "Stage 4 quality flags"}
# The full-corpus column comes from one dedicated timed training run, the one
# manifest that describes a single end-to-end train.
TIMING_MANIFEST = RESULTS / "train_timing_full.json"
#: The same run with Stages 1 and 2 fitted on the CPU (--device cpu); optional.
TIMING_MANIFEST_CPU = RESULTS / "train_timing_full_cpu.json"
BUCKETS = ((0, 500), (500, 1000), (1000, 2000), (2000, 5000), (10**9, 10**9))


def _bucket_label(n: int) -> str:
    for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 5000)):
        if lo <= n < hi:
            return f"{lo}-{hi}"
    return "5000+"


def main() -> int:
    L: list[str] = []
    out: dict = {}

    # ---- A: training ------------------------------------------------------
    # One timed end-to-end run on the full corpus. Section 5.6 reports what
    # retraining on one laboratory is worth; this table reports what training
    # costs.
    if not TIMING_MANIFEST.is_file():
        raise SystemExit(
            f"missing {TIMING_MANIFEST}, the training half of Supplementary Table S11. It is the "
            f"manifest of one full training run, copied across:\n"
            f"  python -m pipeline.train_pipeline --data-dir data --seed 123 "
            f"--out-dir timing_run\n"
            f"  cp timing_run/train_manifest.json {TIMING_MANIFEST}")
    run = json.loads(TIMING_MANIFEST.read_text(encoding="utf-8"))
    n_full = sum(run["counts"][ct]["train"] for ct in run["counts"])
    missing = [st for st in STAGES if st not in run["timings_min"]]
    if missing:
        raise SystemExit(
            f"{TIMING_MANIFEST} has no timing for {missing}; it must come from a run "
            f"with nothing skipped, and this one skipped {run.get('skipped') or 'nothing'}.")

    L += ["TABLE -- COMPUTATIONAL COST", "",
          f"A. TRAINING (minutes; one timed end-to-end run, seed {run['seed']})", "",
          f"  {'stage':30s} {'full corpus':>12s}",
          f"  {'':30s} {f'{n_full} cells':>12s}"]
    train = {st: run["timings_min"][st] for st in STAGES}
    for st in STAGES:
        L.append(f"  {STAGE_LABEL[st]:30s} {train[st]:12.1f}")
    tot_full = sum(train.values())
    L.append(f"  {'total':30s} {tot_full:12.1f}")
    out["training_min"] = {"per_stage": train, "total_full": tot_full,
                           "n_train_full": n_full, "seed": run["seed"]}
    if TIMING_MANIFEST_CPU.is_file():
        cpu_run = json.loads(TIMING_MANIFEST_CPU.read_text(encoding="utf-8"))
        if cpu_run.get("device") != "cpu" or cpu_run.get("skipped") or cpu_run["seed"] != run["seed"]:
            raise SystemExit(f"{TIMING_MANIFEST_CPU} must be a --device cpu run at seed {run['seed']} "
                             f"with nothing skipped")
        cpu_train = {st: cpu_run["timings_min"][st] for st in STAGES}
        out["training_min_cpu"] = {"per_stage": cpu_train, "total_full": sum(cpu_train.values()),
                                   "seed": cpu_run["seed"]}
        L.append(f"  {'total, every stage on the CPU':30s} {sum(cpu_train.values()):12.1f}")

    # ---- B: inference -----------------------------------------------------
    p = RESULTS / "runtime.json"
    if not p.is_file():
        raise SystemExit("run evaluation.profile_runtime first")
    run_inf = json.loads(p.read_text(encoding="utf-8"))
    out["inference"] = run_inf

    L += ["", "B. INFERENCE (seconds per cell, median over the sampled cells; CPU)", ""]
    for key, lbl in (("parse_s", "parse SWC"), ("label_s", "label, complete pipeline"),
                     ("stage1_s", "  of which Stage 1"),
                     ("flag_s", "quality flag")):
        L.append(f"  {lbl:30s} {run_inf['per_cell_s'][key]['median']:10.3f}")
    e2e = run_inf.get("end_to_end")
    if e2e:
        L += [f"  {'end to end, median':30s} {e2e['median']:10.3f}",
              f"  {'end to end, mean':30s} {e2e['mean']:10.3f}",
              f"  (end to end: read, Stage 0 checks, label, write, flag; "
              f"{e2e['n_cells']} random cells)"]
    mem = run_inf["memory_mb"]
    L += ["", f"  models load in {run_inf['model_load_s']:.1f} s, "
              f"{mem['model_footprint']:.0f} MB resident for weights, "
              f"peak process {mem['peak_rss']:.0f} MB"]

    # ---- C: size ----------------------------------------------------------
    L += ["", "C. MORPHOLOGY SIZE", "",
          f"  {'nodes':>10s} {'cells':>6s} {'median s':>9s} {'ms/1k nodes':>12s}"
          f" | {'cells':>6s} {'pcF1 mean':>10s} {'pcF1 P10':>9s}"]
    acc: dict[str, list[float]] = {}
    for s in SEEDS:
        p = RESULTS / f"per_cell_f1_seed{s}.csv"
        if not p.is_file():
            continue
        for row in csv.DictReader(p.open(encoding="utf-8", newline="")):
            acc.setdefault(_bucket_label(int(row["n_nodes"])), []).append(
                float(row["neurite_macro_f1"]))
    size = {}
    order = ["0-500", "500-1000", "1000-2000", "2000-5000", "5000+"]
    for b in order:
        rt = run_inf["by_bucket"].get(b)
        fs = np.array(acc.get(b, []), dtype=float)
        size[b] = {
            "runtime": rt,
            "accuracy": ({"n": int(fs.size), "mean": float(fs.mean()),
                          "p10": float(np.percentile(fs, 10))} if fs.size else None)}
        left = (f"  {b:>10s} {rt['n']:6d} {rt['median_label_s']:9.3f} "
                f"{rt['ms_per_1k_nodes']:12.1f}" if rt else f"  {b:>10s} {'--':>6s}"
                f" {'--':>9s} {'--':>12s}")
        right = (f" | {fs.size:6d} {fs.mean():10.3f} {np.percentile(fs, 10):9.3f}"
                 if fs.size else f" | {'--':>6s} {'--':>10s} {'--':>9s}")
        L.append(left + right)
    out["by_size"] = size
    L += ["", "  left: runtime from profile_runtime; "
          f"right: accuracy over seeds {', '.join(map(str, SEEDS))}"]

    report = "\n".join(L)
    print(report)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_TXT.relative_to(ROOT)} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
