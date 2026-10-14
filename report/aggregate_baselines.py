"""Baseline rows of the paper's Tables 2, 3 and 4: mean and SD over the seeds.

``report.aggregate_pipeline`` compiles the pipeline's own row from the
``eval_pipeline`` results. This does the same for every other row: the six
constructed baselines from the per-seed files ``evaluation.baselines.eval_baselines
--method all`` writes, and the retrained external GCN from the per-seed files
``evaluation.baselines.emissah_gcn`` writes. No row of the comparison tables is
assembled by hand. Metrics are read from each report exactly as the driver
computed them; nothing is recomputed here.

The rows are the corpus (pooled node) accuracy and neurite macro-F1, the per-class
F1 scores, and the per-cell neurite F1 mean and 10th percentile, for all cells and
for each cell type, matching Tables 2, 3 and 4 respectively.

    python -m report.aggregate_baselines
    python -m report.aggregate_baselines --seeds 42,123,789

The retrained Emissah GCN is scored by its own driver and written to its own
files, so it is read from those and emitted into the pyramidal table, the only
one it belongs in: it labels dendritic trees of pyramidal cells. Its protocol
copies the soma and every axon-only tree from the ground truth, so only its
basal and apical F1 measure the method. Every other column mixes in the copied
labels and is left blank (null in the JSON, "--" in the text); the driver's own
per-seed files keep those values.

Input:  evaluation/results/baselines/baselines_seed{seed}.json
        evaluation/results/baselines/emissah_gcn_seed{seed}.json
Output: report/baseline_tables345.{json,txt}
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE_DIR = ROOT / "evaluation" / "results" / "baselines"
#: The GCN labels dendritic trees of pyramidal cells, so it has a row in Table 3
#: and none in Tables 2 or 4.
GCN_METHOD = "emissah_gcn"
GCN_SCOPE = "pyramidal"
#: The only columns the GCN's own predictions decide; the rest include copied labels.
GCN_MEASURED = ("Basal", "Apical")
REPORT = Path(__file__).resolve().parent
OUT_JSON = REPORT / "baseline_tables345.json"
OUT_TXT = REPORT / "baseline_tables345.txt"

# (column label, where to read it from one report's scope block)
COLUMNS = [
    ("Acc.", lambda b: b["accuracy"]),
    ("Macro-F1", lambda b: b["neurite_macro_f1"]),
    ("Axon", lambda b: b["per_class"]["axon"]["f1"]),
    ("Basal", lambda b: b["per_class"]["basal/dendrite"]["f1"]),
    ("Apical", lambda b: b["per_class"]["apical"]["f1"]),
]
SCOPES = [("all", "Table 2 -- all"), ("pyramidal", "Table 3 -- pyramidal"),
          ("interneuron", "Table 4 -- interneuron")]


def _scope_block(report: dict, scope: str) -> dict | None:
    if scope == "all":
        return report["corpus"]
    block = report.get("by_cell_type", {}).get(scope)
    return (block or {}).get("corpus")


def _per_cell(report: dict, scope: str) -> dict | None:
    if scope == "all":
        return report["per_cell"]["neurite_macro_f1"]
    block = report.get("by_cell_type", {}).get(scope) or {}
    pc = block.get("per_cell") or {}
    return pc.get("neurite_macro_f1")


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan")
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,123,789")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    per_seed: dict[int, dict[str, dict]] = {}
    gcn: dict[int, dict] = {}
    for seed in seeds:
        path = BASELINE_DIR / f"baselines_seed{seed}.json"
        if not path.is_file():
            raise SystemExit(f"MISSING {path}\n  run: python -m evaluation.baselines."
                             f"eval_baselines --seed {seed} --method all")
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_seed[seed] = {r["method"]: r for r in payload["reports"]}

        # The GCN's driver writes a flatter record than the baseline reports, so
        # it is reshaped into the same block layout rather than special-cased in
        # every reader below.
        gpath = BASELINE_DIR / f"{GCN_METHOD}_seed{seed}.json"
        if not gpath.is_file():
            raise SystemExit(f"MISSING {gpath}\n  run: python -m evaluation.baselines."
                             f"{GCN_METHOD} --seeds {seed}")
        g = json.loads(gpath.read_text(encoding="utf-8"))["corpus"]
        gcn[seed] = {
            "corpus": {"accuracy": g["accuracy"],
                       "neurite_macro_f1": g["neurite_macro_f1"],
                       "per_class": {k: {"f1": v} for k, v in g["per_class"].items()}},
            "per_cell": {"neurite_macro_f1": {"mean": g["pcf1_mean"],
                                              "p10": g["pcf1_p10"]}},
        }

    methods = [m for m in per_seed[seeds[0]]
               if all(m in per_seed[s] for s in seeds)]
    dropped = sorted(set().union(*(set(per_seed[s]) for s in seeds)) - set(methods))

    out: dict = {"seeds": seeds, "methods": methods + [GCN_METHOD],
                 "incomplete_methods": dropped,
                 "gcn_scope": GCN_SCOPE,
                 "source": ["evaluation/results/baselines/baselines_seed{seed}.json",
                            f"evaluation/results/baselines/{GCN_METHOD}_seed{{seed}}.json"],
                 "tables": {}}
    lines = ["Baseline rows -- Tables 2/3/4",
             f"mean +/- SD over seeds {', '.join(map(str, seeds))}; "
             "GT cell type; read from the per-seed baseline reports", ""]
    if dropped:
        lines.append(f"  NOT every seed has: {', '.join(dropped)} (omitted)")
        lines.append("")

    for scope, title in SCOPES:
        rows = {}
        lines += [title,
                  f"  {'method':18s}" + "".join(f"{c:>16s}" for c, _ in COLUMNS)
                  + f"{'pcF1 mean':>16s}{'pcF1 P10':>16s}"]
        for method in methods:
            blocks = [(_scope_block(per_seed[s][method], scope),
                       _per_cell(per_seed[s][method], scope)) for s in seeds]
            if any(b is None for b, _ in blocks):
                continue
            cells = []
            row: dict[str, dict[str, float]] = {}
            for label, get in COLUMNS:
                m, sd = _mean_sd([get(b) for b, _ in blocks])
                row[label] = {"mean": m, "sd": sd}
                cells.append(f"{m:>10.4f}+/-{sd:.4f}")
            for label, key in (("pcF1 mean", "mean"), ("pcF1 P10", "p10")):
                m, sd = _mean_sd([(pc or {}).get(key) for _, pc in blocks])
                row[label] = {"mean": m, "sd": sd}
                cells.append(f"{m:>10.4f}+/-{sd:.4f}")
            rows[method] = row
            lines.append(f"  {method:18s}" + "".join(cells))

        if scope == GCN_SCOPE:
            blocks = [(gcn[s]["corpus"], gcn[s]["per_cell"]["neurite_macro_f1"])
                      for s in seeds]
            row, cells = {}, []
            for label, get in COLUMNS:
                if label not in GCN_MEASURED:
                    row[label] = None
                    cells.append(f"{'--':>16s}")
                    continue
                m, sd = _mean_sd([get(b) for b, _ in blocks])
                row[label] = {"mean": m, "sd": sd}
                cells.append(f"{m:>10.4f}+/-{sd:.4f}")
            for label in ("pcF1 mean", "pcF1 P10"):
                row[label] = None
                cells.append(f"{'--':>16s}")
            rows[GCN_METHOD] = row
            lines.append(f"  {GCN_METHOD:18s}" + "".join(cells))

        out["tables"][scope] = rows
        lines.append("")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    report = "\n".join(lines)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"wrote {OUT_JSON.relative_to(ROOT).as_posix()} and "
          f"{OUT_TXT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
