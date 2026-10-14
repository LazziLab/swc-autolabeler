"""Compile the Stage 3 rule ablation table.

Reads ``stage3_rules_seed{42,123,789}.json`` from
``evaluation.eval_stage3_rules`` and reports, per rule, the mean over seeds of
the cost of removing it, together with how many cells it changes and whether
those changes help or hurt. Rules are ordered by the size of their effect on
the 10th-percentile per-cell F1, which is the quantity Stage 3 exists to
protect.

    python -m report.stage3_rules_table
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
REPORT = Path(__file__).resolve().parent
OUT_TXT = REPORT / "stage3_rules_table.txt"
OUT_JSON = REPORT / "stage3_rules_table.json"

SEEDS = (42, 123, 789)


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    return (statistics.mean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> int:
    runs = {}
    for s in SEEDS:
        p = RESULTS / f"stage3_rules_seed{s}.json"
        if p.is_file():
            runs[s] = json.loads(p.read_text(encoding="utf-8"))
    if not runs:
        raise SystemExit("run evaluation.eval_stage3_rules first")
    seeds = sorted(runs)
    any_run = runs[seeds[0]]
    rule_names = [r for r in any_run["rules"] if r != "no_stage3"] + ["no_stage3"]

    rows = []
    for name in rule_names:
        d_p10 = [runs[s]["rules"][name]["delta"]["p10"] for s in seeds]
        d_mean = [runs[s]["rules"][name]["delta"]["mean"] for s in seeds]
        d_acc = [runs[s]["rules"][name]["delta"]["acc"] for s in seeds]
        touched = [runs[s]["rules"][name]["touched"] for s in seeds]
        helped = [runs[s]["rules"][name]["helped"] for s in seeds]
        harmed = [runs[s]["rules"][name]["harmed"] for s in seeds]
        in_scope = [runs[s]["rules"][name]["n_in_scope"] for s in seeds]
        rows.append({
            "rule": name, "scope": any_run["rules"][name]["scope"],
            "d_p10": _mean_sd(d_p10), "d_mean": _mean_sd(d_mean),
            "d_acc": _mean_sd(d_acc),
            # Totals over the seeds, not means, so that helped plus harmed adds
            # up to touched. The deltas stay per-seed means.
            "touched": sum(touched), "helped": sum(helped), "harmed": sum(harmed),
            "touched_per_seed": touched, "helped_per_seed": helped,
            "harmed_per_seed": harmed,
            "in_scope": statistics.mean(in_scope),
        })
    active = [r for r in rows if r["rule"] != "no_stage3"]
    active.sort(key=lambda r: r["d_p10"][0])
    # Dormant means no cell changed on any seed. This tested the mean against 1,
    # so a rule that changed a cell on two seeds and none on the third had mean
    # 0.67 and was reported as never firing, contradicting its own row.
    dormant = [r for r in active if r["touched"] == 0]
    rows = active + [r for r in rows if r["rule"] == "no_stage3"]

    n_cells = any_run["n_cells"]
    L = [f"STAGE 3 RULE ABLATION -- mean over seeds {', '.join(map(str, seeds))}, "
         f"{n_cells} held-out cells each", "",
         "Each row removes one rule and keeps the rest. A negative delta means the",
         "rule helps. The three counts are totals over the seeds; the deltas are",
         "per-seed means (SD). 'touched' counts cells whose labels change without it,",
         "within",
         "the cell type the rule applies to.", "",
         f"  {'rule':22s} {'scope':>12s} {'dP10':>16s} {'dMean':>16s} "
         f"{'touched':>8s} {'helped':>7s} {'harmed':>7s}"]
    for r in rows:
        if r["rule"] == "no_stage3":
            L.append("  " + "-" * 90)
        L.append(f"  {r['rule']:22s} {r['scope']:>12s} "
                 f"{r['d_p10'][0]:+9.4f} ({r['d_p10'][1]:.4f}) "
                 f"{r['d_mean'][0]:+9.4f} ({r['d_mean'][1]:.4f}) "
                 f"{r['touched']:8d} {r['helped']:7d} {r['harmed']:7d}")

    full = [runs[s]["scores"]["full"] for s in seeds]
    none = [runs[s]["scores"]["no_stage3"] for s in seeds]
    L += ["", "Stage 3 as a whole", ""]
    for lbl, group in (("with Stage 3", full), ("without Stage 3", none)):
        acc = _mean_sd([g["acc"] for g in group])
        mn = _mean_sd([g["mean"] for g in group])
        p10 = _mean_sd([g["p10"] for g in group])
        L.append(f"  {lbl:18s} acc {acc[0]:.4f} ({acc[1]:.4f})   "
                 f"pcF1 mean {mn[0]:.4f} ({mn[1]:.4f})   P10 {p10[0]:.4f} ({p10[1]:.4f})")
    if dormant:
        L += ["", f"  Dormant on this corpus (no cell changed): "
              f"{', '.join(r['rule'] for r in dormant)}"]

    report = "\n".join(L)
    print(report)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        {"seeds": seeds, "n_cells": n_cells, "rules": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_TXT.relative_to(ROOT)} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
