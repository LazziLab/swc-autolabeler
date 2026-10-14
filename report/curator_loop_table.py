"""Three-seed summary of the curator feedback loop (Sec 5.7, Table 10).

Reads curator_loop_seed{42,123,789}.json and reports, for every round, the mean
over seeds of each arm's labeler and flag metrics, and each arm's
complete-pipeline endpoint against the factory Stage 2. The factory's training
cell count comes from the pipeline's own hash split of the pyramidal corpus.
The flag metrics are those of the loop's ranking flag, which the paper does not
report (see evaluation.active_learning.curator_loop); Figure 3a plots the
labeler's macro-F1.

    python -m report.curator_loop_table

Outputs (evaluation/results/active_learning/):
    curator_loop_summary.{json,txt}
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AL = ROOT / "evaluation" / "results" / "active_learning"
SEEDS = (42, 123, 789)
ARMS = ("flag", "random", "flag_noisy")
ROUND_METRICS = ("labeler_node_accuracy", "labeler_neurite_macro_f1", "labeler_apical_f1",
                 "labeler_pcf1_mean", "labeler_pcf1_p10", "flag_kept_pcf1_p10", "flag_ap",
                 "flag_precision", "flag_recall")
ENDPOINT_METRICS = ("node_accuracy", "neurite_macro_f1", "apical_f1", "pcf1_mean", "pcf1_p10")


def _stats(values: list[float]) -> dict:
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values),
            "per_seed": dict(zip(map(str, SEEDS), values))}


def main() -> int:
    from pipeline.split import in_test

    runs = {s: json.loads((AL / f"curator_loop_seed{s}.json").read_text(encoding="utf-8")) for s in SEEDS}
    pyramidal = sorted((ROOT / "data" / "pyramidal").glob("*.swc"))
    protocol = {}
    for s, d in runs.items():
        protocol[str(s)] = {k: d[k] for k in ("n_cells", "n_start", "n_pool", "n_test", "rounds", "k", "noise_rate")}
        protocol[str(s)].update({
            "factory_train_cells": sum(not in_test(p.name, s, 0.20) for p in pyramidal),
            "final_train_cells": d["n_start"] + d["rounds"] * d["k"],
            "n_bad_test": d["flag_arm"][0]["n_bad_test"],
        })

    lengths = {len(d[f"{a}_arm"]) for d in runs.values() for a in ARMS}
    if len(lengths) != 1:
        raise SystemExit(f"arms differ in length: {lengths}")
    rounds = []
    for r in range(lengths.pop()):
        corrections = {d["flag_arm"][r]["n_corrections"] for d in runs.values()}
        if len(corrections) != 1:
            raise SystemExit(f"round {r}: seeds differ in corrections {corrections}")
        rounds.append({"round": r, "n_corrections": corrections.pop(),
                       "arms": {a: {m: _stats([runs[s][f"{a}_arm"][r][m] for s in SEEDS]) for m in ROUND_METRICS}
                                for a in ARMS}})

    endpoints = {a: {m: _stats([runs[s]["endpoint_full_pipeline"][a][m] for s in SEEDS]) for m in ENDPOINT_METRICS}
                 for a in ("factory",) + ARMS}
    # Each arm's endpoint is its final Stage 2 in the complete pipeline, which is
    # also how its last round was labeled.
    endpoint_vs_last_round = max(
        abs(runs[s][f"{a}_arm"][-1]["labeler_neurite_macro_f1"] - runs[s]["endpoint_full_pipeline"][a]["neurite_macro_f1"])
        for s in SEEDS for a in ARMS)

    def wins(metric: str, a: str, b: str) -> dict:
        later = [x for x in rounds if x["round"] > 0]
        return {"mean_curve": sum(x["arms"][a][metric]["mean"] > x["arms"][b][metric]["mean"] for x in later),
                "per_seed": {str(s): sum(runs[s][f"{a}_arm"][x["round"]][metric] > runs[s][f"{b}_arm"][x["round"]][metric]
                                         for x in later) for s in SEEDS},
                "rounds": len(later)}

    comparisons = {f"{a}_over_{b}/{m}": wins(m, a, b)
                   for a, b in (("flag", "random"), ("flag_noisy", "random"), ("flag", "flag_noisy"))
                   for m in ("labeler_neurite_macro_f1", "labeler_pcf1_p10", "flag_kept_pcf1_p10")}
    endpoint_comparisons = {
        f"{a}_minus_{b}/{m}": dict(zip(map(str, SEEDS), [runs[s]["endpoint_full_pipeline"][a][m] - runs[s]["endpoint_full_pipeline"][b][m]
                                                          for s in SEEDS]))
        for a, b in (("factory", "flag"), ("flag", "random"), ("flag_noisy", "random"), ("flag", "flag_noisy"))
        for m in ("neurite_macro_f1", "pcf1_p10")}

    out = {"seeds": list(SEEDS), "protocol": protocol, "rounds": rounds, "endpoints": endpoints,
           "round_comparisons": comparisons, "endpoint_differences": endpoint_comparisons,
           "endpoint_vs_last_round_max_abs_diff": endpoint_vs_last_round}
    (AL / "curator_loop_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [f"CURATOR FEEDBACK LOOP -- mean over seeds {', '.join(map(str, SEEDS))}", ""]
    for s, p in protocol.items():
        lines.append(f"seed {s}: {p['n_cells']} cells, start {p['n_start']}, pool {p['n_pool']}, test {p['n_test']} "
                     f"({p['n_bad_test']} bad), {p['rounds']} rounds x {p['k']}, noise {p['noise_rate']:.0%}, "
                     f"final {p['final_train_cells']} vs factory {p['factory_train_cells']} training cells")
    lines += ["", "round  corr   labeler macro-F1 (flag/random/noisy)   labeler P10 (flag/random/noisy)   "
                  "flag kept-P10 (flag/random/noisy)   flag AP (flag/random/noisy)"]
    for x in rounds:
        cols = ["  ".join(f"{x['arms'][a][m]['mean']:.4f}" for a in ARMS)
                for m in ("labeler_neurite_macro_f1", "labeler_pcf1_p10", "flag_kept_pcf1_p10", "flag_ap")]
        lines.append(f"{x['round']:5d}  {x['n_corrections']:4d}   " + "    ".join(cols))
    lines += ["", "complete-pipeline endpoints, mean (SD) and per seed"]
    for a, row in endpoints.items():
        lines.append(f"  {a:10s} " + "  ".join(f"{m} {v['mean']:.4f} ({v['sd']:.4f})" for m, v in row.items()))
        lines.append(f"  {'':10s} per seed macro-F1 {row['neurite_macro_f1']['per_seed']}  P10 {row['pcf1_p10']['per_seed']}")
    lines += ["", "rounds 1-8 where the first arm beats the second (mean curve; per seed)"]
    lines += [f"  {k}: {v['mean_curve']}/{v['rounds']}; {v['per_seed']}" for k, v in comparisons.items()]
    lines += ["", "endpoint differences per seed"]
    lines += [f"  {k}: " + ", ".join(f"{s} {d:+.4f}" for s, d in v.items()) for k, v in endpoint_comparisons.items()]
    lines.append(f"\nendpoint vs last round, max |macro-F1 difference|: {endpoint_vs_last_round:.4f}")
    (AL / "curator_loop_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
