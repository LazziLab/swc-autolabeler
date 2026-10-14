"""Stage ablation ladder for the current auto-labeling pipeline.

Isolates each component's contribution by enabling the pipeline's components one
at a time, in execution order, on each seed's hash-split test set, averaged over
seeds:

  1. Stage 2 only (GT cell type)          - subtree-owner labels
  2. + Stage 3 topology rules (GT)        - rule consolidation
  3. + branch correction (GT)             - Stage 3's learned pass  (full model)
  4. Complete pipeline, predicted type   - Stage 1 chooses the cell type

Rows 1-3 receive the ground-truth cell type; row 3 is the complete current model
and row 4 is its deployment configuration. Intermediate labels come from the
pipeline's own stages via its ``trace`` hook, so no stage is re-implemented. The
"dMacroF1" column is each row's neurite-macro-F1 gain over the row above it, that
component's marginal contribution. The "chgN"/"chgC" columns report how often the
added component acts: the percentage of nodes it relabels and of cells it
touches, each relative to the row above (deployment is compared against row 3, so
its columns measure the effect of switching to a predicted cell type).

    python -m evaluation.eval_ablation                   # run seeds 42,123,789
    python -m evaluation.eval_ablation --seeds 123 --limit 40   # smoke test
    python -m evaluation.eval_ablation --from-json       # reprint the saved result
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from array import array                                              # noqa: E402

from pipeline.features import parse_swc                              # noqa: E402
from pipeline.infer_pipeline import run_pipeline_on_nodes, load_branch_correction  # noqa: E402
from pipeline.stage1.cell_type_detector import detect_cell_type_from_nodes  # noqa: E402
from evaluation.baselines.baseline_report import build_full_report   # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed, per_cell_neurite_f1
from pipeline.split import in_test  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "evaluation" / "results"
LADDER_JSON = RESULTS / "ablation_ladder.json"
MODES = ["stage2_only", "stage3_topology", "full_correction", "deployment"]
# The complete GT-cell-type model; the deployment row is scored against it.
FULL = "full_correction"
LABEL = {
    "stage2_only": "Stage 2 only (GT)",
    "stage3_topology": "+ Stage 3 topology rules (GT)",
    "full_correction": "+ branch correction (GT, full)",
    "deployment": "Complete pipeline, predicted type",
}
METRICS = ["accuracy", "neurite_macro_f1", "axon_f1", "basal_f1", "apical_f1", "pcf1_p10"]


def ladder_preds(nodes, file_path, s1, s2, ct_gt) -> dict[str, list[int]]:
    """Per-node labels for every reported configuration, from the pipeline itself.

    Run A (GT cell type) supplies rungs 1-3 through the pipeline's ``trace``:
    labels after Stage 2 (``s2_labels``), after the Stage 3 rules but before the
    branch correction (``final_labels``), and the pipeline output.

    Run B, the deployment configuration, differs from run A only in where the
    cell type comes from, and every stage dispatches on that type alone:
    ``refine`` reads the Stage 1 result's ``cell_type`` and ``label_set`` and
    nothing else, not its confidence, probabilities or features. So when Stage 1
    predicts the ground-truth type, which it does on about 97.6% of cells, run B
    would reproduce run A exactly and is skipped."""
    common = dict(stage1_model=s1, stage2_model=s2)
    trA: dict = {}
    depA = run_pipeline_on_nodes(nodes, file_path, override_cell_type=ct_gt, trace=trA, **common)
    if detect_cell_type_from_nodes(nodes, s1).cell_type == ct_gt:
        deployment = depA.node_labels
    else:
        deployment = run_pipeline_on_nodes(nodes, file_path, **common).node_labels
    # int8 rather than lists of Python ints: the ladder holds four labelings for
    # every test cell at once, and 8 bytes a node adds up over the corpus.
    return {
        "stage2_only": array("b", trA["s2_labels"]),
        "stage3_topology": array("b", trA["final_labels"]),
        "full_correction": array("b", depA.node_labels),
        "deployment": array("b", deployment),
    }


NEURITE_LABELS = (2, 3, 4)          # axon, basal/dendrite, apical
NEURITE_NAMES = ("axon", "basal", "apical")
# The rung whose configuration matches the headline results tables: GT cell type,
# every stage on. Its per-cell scores are written out so they can be paired
# cell-for-cell against the baselines' per-cell CSVs.
PAPER_MODE = "full_correction"


def _write_per_cell(seed: int, records: list[dict], tag: str = "") -> Path:
    """Write per-cell neurite F1 for the paper configuration, one row per test cell."""
    import csv  # noqa: PLC0415

    out = RESULTS / f"per_cell_f1_seed{seed}{tag}.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "cell_type", "n_nodes", "neurite_macro_f1"])
        for r in records:
            w.writerow([r["file"], r["cell_type"], r["n_nodes"],
                        f"{per_cell_neurite_f1(r['gt'], r['pred'], r['cell_type']):.6f}"])
    return out


def _neurite_confusion(gt, pred) -> dict:
    """Row-normalized neurite confusion in percent, keyed by true class."""
    import numpy as np  # noqa: PLC0415
    from sklearn.metrics import confusion_matrix  # noqa: PLC0415

    cm = confusion_matrix(gt, pred, labels=list(NEURITE_LABELS)).astype(float)
    totals = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm * 100.0, totals, out=np.zeros_like(cm), where=totals > 0)
    return {NEURITE_NAMES[i]: {NEURITE_NAMES[j]: round(float(pct[i, j]), 1)
                               for j in range(len(NEURITE_NAMES))}
            for i in range(len(NEURITE_NAMES))}


def eval_seed(seed, limit=0, test_size=0.20):
    md = ROOT / "pipeline" / "models" / f"seed{seed}"
    s1, s2 = md / "cell_type_classifier.pkl", md / "branch_classifier.pkl"
    # Refuse a model trained on a different split: it would be scored on cells it
    # was trained on, and nothing in the numbers would show it.
    check_model_seed(md, seed, test_size)
    if load_branch_correction(md) is None:   # correction is intrinsic; assert the head ships
        raise SystemExit(f"MISSING branch_correction.joblib in {md}")
    files = [(ct, p) for ct in ("pyramidal", "interneuron")
             for p in sorted((DATA / ct).glob("*.swc")) if in_test(p.name, seed, test_size)]
    if limit:
        files = files[:limit]
    print(f"\n=== seed {seed}: {len(files)} test cells ===", flush=True)
    recs = {m: [] for m in MODES}
    t0 = time.perf_counter()
    for i, (ct, p) in enumerate(files, 1):
        nodes = parse_swc(p)
        if not nodes:
            continue
        gt = array("b", (nd.type for nd in nodes))
        preds = ladder_preds(nodes, str(p), s1, s2, ct)
        for m in MODES:
            # one gt array shared by every rung; only the predictions differ
            recs[m].append({"file": p.name, "cell_type": ct, "n_nodes": len(gt),
                            "gt": gt, "pred": preds[m]})
        if i % 300 == 0:
            print(f"  ... {i}/{len(files)} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    rows = {}
    for mi, m in enumerate(MODES):
        gtp, prp = [], []
        for r in recs[m]:
            gtp.extend(r["gt"]); prp.extend(r["pred"])
        rep = build_full_report(m, gtp, prp, recs[m])
        c = rep["corpus"]; pc = rep["per_cell"]["neurite_macro_f1"]
        rows[m] = {"accuracy": c["accuracy"], "neurite_macro_f1": c["neurite_macro_f1"],
                   "axon_f1": c["per_class"]["axon"]["f1"], "basal_f1": c["per_class"]["basal/dendrite"]["f1"],
                   "apical_f1": c["per_class"]["apical"]["f1"], "pcf1_p10": pc["p10"],
                   # Row-normalized neurite confusion (%), for the paper's before/after table:
                   # rows are the true class, columns the predicted one.
                   "confusion_rownorm_pct": _neurite_confusion(gtp, prp)}
        # How often the component added on this rung acts, relative to the rung
        # above; deployment compares against the full GT-type model.
        if m == "deployment":
            ref = FULL
        else:
            ref = MODES[mi - 1] if mi > 0 else None
        if ref is not None:
            changed_nodes = total_nodes = touched_cells = 0
            for ra, rb in zip(recs[ref], recs[m]):
                diff = sum(1 for a, b in zip(ra["pred"], rb["pred"]) if a != b)
                changed_nodes += diff
                total_nodes += len(ra["pred"])
                touched_cells += 1 if diff else 0
            rows[m]["pct_nodes_changed"] = 100.0 * changed_nodes / max(total_nodes, 1)
            rows[m]["pct_cells_touched"] = 100.0 * touched_cells / max(len(recs[m]), 1)
        else:
            rows[m]["pct_nodes_changed"] = None
            rows[m]["pct_cells_touched"] = None
        chg = ("" if rows[m]["pct_nodes_changed"] is None else
               f" chgN={rows[m]['pct_nodes_changed']:.3f}% chgC={rows[m]['pct_cells_touched']:.1f}%")
        print(f"  {LABEL[m]:<34} acc={rows[m]['accuracy']:.4f} F1={rows[m]['neurite_macro_f1']:.4f} "
              f"P10={rows[m]['pcf1_p10']:.4f}{chg}", flush=True)
    print(f"  -> {_write_per_cell(seed, recs[PAPER_MODE], f'_limit{limit}' if limit else '').name}", flush=True)
    return rows


def print_ladder(per_seed, seeds):
    """Print the ladder (per-stage metrics, marginal dMacroF1, chgN/chgC vs. the
    row above)."""
    def mean(m, k):
        vals = [per_seed[s][m].get(k) for s in seeds]
        if any(v is None for v in vals):
            return None
        return statistics.fmean(vals)

    keys = METRICS + ["pct_nodes_changed", "pct_cells_touched"]
    summary = {m: {k: {"mean": mean(m, k),
                       "sd": (statistics.stdev([per_seed[s][m][k] for s in seeds])
                              if len(seeds) > 1 and mean(m, k) is not None else 0.0)}
                   for k in keys}
               for m in MODES}
    print("\n" + "=" * 106)
    print(f"STAGE ABLATION -- execution-order ladder, mean over {len(seeds)} seed(s)")
    print("=" * 106)
    print(f"{'configuration':<34} {'Acc':>7} {'MacroF1':>8} {'Axon':>6} {'Basal':>6} {'Apical':>7} "
          f"{'P10':>7} {'dMacroF1':>9} {'chgN%':>7} {'chgC%':>6}")
    prev = None
    base_f1 = mean(FULL, "neurite_macro_f1")
    for m in MODES:
        vals = [mean(m, k) for k in METRICS]
        if m == "deployment":
            d = f"{vals[1] - base_f1:+.4f}"
        else:
            d = "  --" if prev is None else f"{vals[1] - prev:+.4f}"
            prev = vals[1]
        cn, cc = mean(m, "pct_nodes_changed"), mean(m, "pct_cells_touched")
        cns = "--" if cn is None else f"{cn:.3f}"
        ccs = "--" if cc is None else f"{cc:.1f}"
        print(f"{LABEL[m]:<34} {vals[0]:>7.4f} {vals[1]:>8.4f} {vals[2]:>6.4f} {vals[3]:>6.4f} "
              f"{vals[4]:>7.4f} {vals[5]:>7.4f} {d:>9} {cns:>7} {ccs:>6}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--test-size", type=float, default=0.20,
                    help="held-out fraction of the hash split (default 0.20)")
    ap.add_argument("--limit", type=int, default=0, help="cells per seed (0=all; for smoke tests)")
    ap.add_argument("--from-json", action="store_true", help="reprint the saved ladder without re-running")
    args = ap.parse_args()

    if args.from_json:
        d = json.loads(LADDER_JSON.read_text())
        per_seed = {int(s): v for s, v in d["per_seed"].items()}
        print_ladder(per_seed, d["seeds"])
        return 0

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    per_seed = {seed: eval_seed(seed, args.limit, args.test_size) for seed in seeds}
    summary = print_ladder(per_seed, seeds)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # A limited run gets its own file names, so a smoke test cannot replace the
    # full ladder or the per-cell tables other scripts read.
    ladder_json = (LADDER_JSON.with_name(f"ablation_ladder_limit{args.limit}.json")
                   if args.limit else LADDER_JSON)
    ladder_json.write_text(json.dumps(
        {"seeds": seeds, "modes": MODES, "full": FULL,
         "summary": summary, "per_seed": {str(s): per_seed[s] for s in seeds}}, indent=2))
    print(f"\nwrote {ladder_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
