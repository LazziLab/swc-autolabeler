"""What labeling a neuron costs: time, memory, and how both scale with size.

Inference is timed per cell and broken into the parts a user can choose
between. Stage 1 is timed directly, by calling the cell-type detector on its
own: differencing two whole-pipeline runs would measure its cost plus the
run-to-run noise of everything else. The quality flag is timed separately: it
featurizes the labeled cell and applies one regression.

Cells are sampled across node-count buckets rather than at random, so the
scaling column reflects the whole size range of the corpus instead of its mode.
Timings exclude the one-off cost of loading models, which is reported once.

A second pass times the labeling command end to end, on cells drawn at random
from the corpus: read the file, run the Stage 0 checks, label, write the labeled
file and score the quality flag, with the models loaded once as in one
long-running process. The bucket sample is drawn to cover the size range, so
its averages do not describe a collection; the mean of the random sample is the
cost per cell of labeling one.

Every inference stage is gradient-boosted and runs on the CPU; nothing in the
pipeline touches the GPU at inference, so there is one profile and no GPU figure.

Usage:
    python -m evaluation.profile_runtime [--seed 123] [--per-bucket 25]
                                         [--end-to-end-cells 1000]
                                         [--end-to-end-only]

Output (evaluation/results/):
    runtime.json
    runtime.txt
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                   # noqa: E402
from pipeline.infer_pipeline import (                                     # noqa: E402
    run_pipeline_on_nodes, _load_correction_cached, load_stage2_bundle,
    parse_swc_for_labeling, write_labeled_swc,
)
from pipeline.qc import qc_violations                                     # noqa: E402
from pipeline.stage1.cell_type_detector import (                          # noqa: E402
    detect_cell_type_from_nodes, load_classifier,
)
from pipeline.stage4.single_seed_flag import score_flag, load_flag        # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed              # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "evaluation" / "results"
CELL_TYPES = ("pyramidal", "interneuron")
# Node-count bucket edges; the corpus spans roughly two orders of magnitude.
BUCKETS = ((0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10**9))
#: Mean bytes per node line in the corpus, used only to bucket by size before
#: sampling. The reported node counts come from the parsed cells.
_BYTES_PER_NODE = 48
#: Seed of the random sample timed end to end.
E2E_SAMPLE_SEED = 2026
#: Where runtime.txt's end-to-end block starts, so a rerun can replace it.
_E2E_HEADER = "END TO END"


def _bucket(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n < hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    return "?"


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        return float("nan")


def _by_component(args, s1: Path, s2: Path, flag, load_s: float,
                  base_rss: float, loaded_rss: float) -> tuple[dict, list[str]]:
    """Per-component timings on the size-bucket sample, and their report lines."""
    # Sample across size buckets so the scaling figures cover the whole range.
    # Bucketed by file size rather than by counting every line of all 11,862
    # files: only ~125 are sampled, and node count and byte size track closely
    # enough to place a cell in a decade-wide bucket.
    by_bucket: dict[str, list] = defaultdict(list)
    for ct in CELL_TYPES:
        for p in sorted((DATA / ct).glob("*.swc")):
            by_bucket[_bucket(int(p.stat().st_size / _BYTES_PER_NODE))].append((ct, p))
    sample = []
    for b in sorted(by_bucket, key=lambda k: int(k.split("-")[0].rstrip("+"))):
        cells = by_bucket[b]
        step = max(1, len(cells) // args.per_bucket)
        sample += [(b, ct, p) for ct, p in cells[::step][:args.per_bucket]]
    print(f"profiling {len(sample)} cells on CPU, seed {args.seed}", flush=True)

    # One warm pass so first-call overhead does not land on the first cell.
    _b, _ct, _p = sample[0]
    _n = parse_swc(_p)
    run_pipeline_on_nodes(_n, str(_p), s1, s2)
    detect_cell_type_from_nodes(_n, s1)

    peak_rss = loaded_rss
    rows = []
    for b, ct, p in sample:
        t = time.perf_counter(); nodes = parse_swc(p); t_parse = time.perf_counter() - t

        t = time.perf_counter()
        run_pipeline_on_nodes(nodes, str(p), s1, s2)
        t_full = time.perf_counter() - t

        # Stage 1 timed on its own, not as the difference of two full runs.
        t = time.perf_counter()
        detect_cell_type_from_nodes(nodes, s1)
        t_stage1 = time.perf_counter() - t

        pr_gt = run_pipeline_on_nodes(nodes, str(p), s1, s2, override_cell_type=ct)

        # Every cell is scored, by the flag of its cell type.
        t = time.perf_counter()
        score_flag(nodes, pr_gt, flag)
        t_flag = time.perf_counter() - t

        peak_rss = max(peak_rss, _rss_mb())
        rows.append({"bucket": b, "cell_type": ct, "n_nodes": len(nodes),
                     "parse_s": t_parse, "label_s": t_full,
                     "stage1_s": t_stage1,
                     "flag_s": t_flag})
        gc.collect()

    def agg(rs, key):
        v = np.array([r[key] for r in rs], dtype=float)
        v = v[~np.isnan(v)]
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95))} if v.size else {}

    out = {"seed": args.seed, "n_cells": len(rows),
           "model_load_s": load_s,
           "memory_mb": {"after_load": loaded_rss, "peak_rss": peak_rss,
                         "model_footprint": loaded_rss - base_rss},
           "per_cell_s": {k: agg(rows, k) for k in
                          ("parse_s", "label_s", "stage1_s", "flag_s")},
           "by_bucket": {}}

    L = [f"Runtime, seed {args.seed}, {len(rows)} cells",
         "", f"model load {load_s:.1f} s; resident {loaded_rss:.0f} MB "
         f"(+{loaded_rss - base_rss:.0f} MB for models), peak {peak_rss:.0f} MB",
         "", "PER CELL (seconds)", "",
         f"  {'component':16s} {'mean':>8s} {'median':>8s} {'p95':>8s}"]
    label = {"parse_s": "parse SWC", "label_s": "label (full)",
             "stage1_s": "  of which S1",
             "flag_s": "quality flag"}
    for k in ("parse_s", "label_s", "stage1_s", "flag_s"):
        a = out["per_cell_s"][k]
        L.append(f"  {label[k]:16s} {a['mean']:8.3f} {a['median']:8.3f} {a['p95']:8.3f}")

    L += ["", "SCALING WITH MORPHOLOGY SIZE", "",
          f"  {'nodes':>12s} {'n':>4s} {'median nodes':>13s} {'label s':>9s} "
          f"{'ms / 1k nodes':>14s}"]
    for b in sorted({r["bucket"] for r in rows},
                    key=lambda k: int(k.split("-")[0].rstrip("+"))):
        rs = [r for r in rows if r["bucket"] == b]
        med_n = float(np.median([r["n_nodes"] for r in rs]))
        med_t = float(np.median([r["label_s"] for r in rs]))
        per_k = 1000.0 * med_t / med_n * 1000
        out["by_bucket"][b] = {"n": len(rs), "median_nodes": med_n,
                               "median_label_s": med_t, "ms_per_1k_nodes": per_k}
        L.append(f"  {b:>12s} {len(rs):4d} {med_n:13.0f} {med_t:9.3f} {per_k:14.1f}")
    return out, L


def _end_to_end(n_cells: int, s1: Path, s2: Path, flag) -> dict:
    """Time the labeling command per cell on a random corpus sample.

    Each cell goes through what ``pipeline.infer_pipeline`` does for one file:
    read, Stage 0 checks, label, write the labeled file, score the flag.
    """
    files = [p for ct in CELL_TYPES for p in sorted((DATA / ct).glob("*.swc"))]
    sample = random.Random(E2E_SAMPLE_SEED).sample(files, min(n_cells, len(files)))
    print(f"timing {len(sample)} random cells end to end", flush=True)

    def one(path: Path, out: Path) -> int:
        nodes, _ = parse_swc_for_labeling(path)
        qc_violations(nodes)
        res = run_pipeline_on_nodes(nodes, str(path), s1, s2)
        write_labeled_swc(path, nodes, res.node_labels, out)
        score_flag(nodes, res, flag)
        return len(nodes)

    times, sizes = [], []
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "labeled.swc"
        one(sample[0], out)  # warm pass, as above
        for p in sample:
            t = time.perf_counter()
            sizes.append(one(p, out))
            times.append(time.perf_counter() - t)
    v = np.array(times)
    return {"n_cells": int(v.size), "sample_seed": E2E_SAMPLE_SEED,
            "mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max()),
            "median_nodes": float(np.median(sizes))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--per-bucket", type=int, default=25)
    ap.add_argument("--end-to-end-cells", type=int, default=1000,
                    help="random corpus cells timed end to end; 0 skips this pass")
    ap.add_argument("--end-to-end-only", action="store_true",
                    help="time only the end-to-end pass and keep the per-component "
                         "results already in runtime.json")
    args = ap.parse_args()
    if args.end_to_end_only and args.end_to_end_cells <= 0:
        raise SystemExit("--end-to-end-only needs --end-to-end-cells above 0")

    mdir = ROOT / "pipeline" / "models" / f"seed{args.seed}"
    check_model_seed(mdir, args.seed)
    s1, s2 = mdir / "cell_type_classifier.pkl", mdir / "branch_classifier.pkl"

    # Every model file, through the loaders the pipeline itself uses. They keep
    # what they read, so the per-cell timings below exclude this cost.
    base_rss = _rss_mb()
    t0 = time.perf_counter()
    load_classifier(s1)
    load_stage2_bundle(s2)
    if _load_correction_cached(s2) is None:
        raise SystemExit(f"MISSING branch_correction.joblib in {mdir}")
    flag = load_flag(mdir)
    load_s = time.perf_counter() - t0
    loaded_rss = _rss_mb()

    if args.end_to_end_only:
        prev_json, prev_txt = RESULTS / "runtime.json", RESULTS / "runtime.txt"
        if not (prev_json.is_file() and prev_txt.is_file()):
            raise SystemExit("no runtime.json/.txt to add to; run without --end-to-end-only")
        out = json.loads(prev_json.read_text(encoding="utf-8"))
        if out["seed"] != args.seed:
            raise SystemExit(f"runtime.json is for seed {out['seed']}, not {args.seed}")
        txt = prev_txt.read_text(encoding="utf-8")
        L = txt.split("\n\n" + _E2E_HEADER)[0].rstrip("\n").split("\n")
    else:
        out, L = _by_component(args, s1, s2, flag, load_s, base_rss, loaded_rss)

    if args.end_to_end_cells > 0:
        e = _end_to_end(args.end_to_end_cells, s1, s2, flag)
        out["end_to_end"] = e
        L += ["", f"{_E2E_HEADER} (read, Stage 0 checks, label, write, flag; "
              f"{e['n_cells']} random cells, sample seed {e['sample_seed']})", "",
              f"  mean {e['mean']:.3f} s, median {e['median']:.3f} s, "
              f"p90 {e['p90']:.3f} s, max {e['max']:.2f} s; "
              f"median cell {e['median_nodes']:.0f} nodes"]

    report = "\n".join(L)
    print("\n" + report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "runtime.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RESULTS / "runtime.txt").write_text(report + "\n", encoding="utf-8")
    print("\nwrote evaluation/results/runtime.{json,txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
