"""Aggregate the per-seed full-pipeline evaluations into the paper's Tables 2/3/4.

Reads ``eval_pipeline_seed{42,123,789}.json`` (written by ``evaluation.eval_pipeline``,
GT cell type, intrinsic branch correction always applied) and reports the mean
+/- SD over seeds for:

  Table 2  all cells combined
  Table 3  pyramidal cells
  Table 4  interneuron cells (no apical)

with the metric columns the paper's tables carry: accuracy, neurite macro-F1,
per-class axon/basal/apical F1, per-cell F1 mean, and per-cell F1 P10.

    python -m report.aggregate_pipeline
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"        # inputs: per-seed eval_pipeline_seed*.json
REPORT = Path(__file__).resolve().parent         # outputs (the paper tables) land here, in report/
OUT_JSON = REPORT / "pipeline_tables345.json"
OUT_TXT = REPORT / "pipeline_tables345.txt"

SEEDS = (42, 123, 789)
# (metric key, human label); interneuron has no apical and is skipped for it.
COLS = [("acc", "Acc."), ("macro", "Macro-F1"), ("axon", "Axon"),
        ("basal", "Basal"), ("apical", "Apical"), ("pcf1_mean", "pcF1 mean"),
        ("pcf1_p10", "pcF1 P10")]


def _metric(block: dict, key: str):
    """Pull one paper-column metric out of an eval_pipeline metric block."""
    if key == "acc":
        return block["accuracy"]
    if key == "macro":
        return block["neurite_macro_f1"]
    if key in ("axon", "basal", "apical"):
        pc = block["per_class"].get(key)
        return pc["f1"] if pc else None
    if key == "pcf1_mean":
        return block.get("pcf1_mean")
    if key == "pcf1_p10":
        return block.get("pcf1_p10")
    raise KeyError(key)


def _group_block(doc: dict, group: str) -> dict:
    return doc["corpus"] if group == "all" else doc["by_cell_type"][group]


def main() -> int:
    docs = {}
    for s in SEEDS:
        p = RESULTS / f"eval_pipeline_seed{s}.json"
        if not p.is_file():
            raise SystemExit(f"MISSING {p} -- run `python -m evaluation.eval_pipeline --seed {s}` first")
        d = json.loads(p.read_text())
        if not d.get("correction_applied"):
            raise SystemExit(f"{p} does not record the branch correction as applied "
                             f"(correction_applied != True); regenerate it with evaluation.eval_pipeline")
        docs[s] = d

    out = {"seeds": list(SEEDS), "note": "GT cell type; intrinsic branch correction always applied",
           "tables": {}}
    lines = ["Full-pipeline evaluation -- Tables 2/3/4",
             f"mean +/- SD over seeds {', '.join(map(str, SEEDS))}; GT cell type; correction intrinsic",
             ""]

    for group, tnum in (("all", 2), ("pyramidal", 3), ("interneuron", 4)):
        blocks = [_group_block(docs[s], group) for s in SEEDS]
        row = {}
        lines.append(f"Table {tnum} -- {group}")
        lines.append(f"  {'metric':10s} {'mean':>11s} {'SD':>7s}")
        for key, lab in COLS:
            vals = [_metric(b, key) for b in blocks]
            if any(v is None for v in vals):   # interneuron apical
                continue
            mean = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[key] = {"mean": mean, "sd": sd}
            lines.append(f"  {lab:10s} {mean:>11.4f} {sd:>7.4f}")
        out["tables"][group] = row
        lines.append("")

    report = "\n".join(lines)
    print(report)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}\nwrote {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
