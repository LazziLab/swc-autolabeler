"""Compile the quality-flag feature-set ablation (the paper's Supplementary Table S6).

Every row uses the identical leave-one-seed-out protocol of
``evaluation.stage4.eval_flag``: the deployed flag (the canonical run in
``evaluation/results/flag/``), the same flag with only the pipeline's own confidences,
and a no-model ranking by one minus mean node confidence
(``evaluation/results/flag_ablation/<set>/``).

Per-fold differences are taken against the full flag, so a gap can be read
against fold-to-fold variation.

    python -m report.flag_ablation_table

Writes report/flag_ablation_table.{json,txt}.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
ABL = RES / "flag_ablation"
REPORT = Path(__file__).resolve().parent
RATES = (0.05, 0.10, 0.15)
BLOCKS = [
    ("current pipeline", "full", [
        ("full", RES / "flag"),
        ("confidence_only", ABL / "confidence_only"),
        ("mean_confidence", ABL / "mean_confidence"),
    ]),
]


def fold_metrics(path: Path) -> pd.DataFrame | None:
    """One row per held-out seed: AP, AUC, and recall / kept-P10 at each rate."""
    csv_path = path / "final_flag_leave_one_seed_out.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    # Supplementary Table S6 compares feature sets on the pyramidal flag.
    df = df[df["scope"].astype(str) == "pyramidal"]
    folds = df.drop_duplicates("test_seed")[["test_seed", "test_ap", "test_auc"]].set_index("test_seed")
    fixed = df[df["operating_point"] == "fixed_reject_rate"]
    for rate in RATES:
        part = fixed[(fixed["target_value"] - rate).abs() < 1e-9].set_index("test_seed")
        folds[f"recall@{rate:.2f}"] = part["recall_lt_target"]
        folds[f"keptP10@{rate:.2f}"] = part["kept_f1_p10"]
    return folds.sort_index()


def main() -> int:
    metrics = ["test_ap", "test_auc", "recall@0.10"] + [f"keptP10@{r:.2f}" for r in RATES]
    head = f"{'feature set':<17}" + "".join(f"{m:>16}" for m in metrics)
    lines = ["QUALITY-FLAG FEATURE-SET ABLATION -- leave-one-seed-out, pyramidal, mean (SD) over 3 folds"]
    out: dict = {"metrics": metrics, "blocks": {}}
    for title, ref, sets in BLOCKS:
        per_set = {name: fold_metrics(path) for name, path in sets}
        missing = [name for name, folds in per_set.items() if folds is None]
        if missing:
            raise SystemExit(f"{title}: no leave-one-seed-out results for {missing}")
        lines += ["", title.upper(), head, "-" * len(head)]
        block: dict = {"reference": ref, "sets": {}, "delta": {}}
        for name, folds in per_set.items():
            cells = "".join(f"{folds[m].mean():>9.3f} ({folds[m].std():.3f})" for m in metrics)
            lines.append(f"{name:<17}{cells}")
            block["sets"][name] = {m: {"mean": float(folds[m].mean()), "sd": float(folds[m].std()),
                                       "per_fold": {str(s): float(v) for s, v in folds[m].items()}}
                                   for m in metrics}
        base = per_set[ref]
        lines.append(f"  per-fold difference vs {ref} (seeds {', '.join(str(s) for s in base.index)})")
        for name, folds in per_set.items():
            if name == ref:
                continue
            d = {m: folds[m] - base[m] for m in metrics}
            block["delta"][name] = {m: [float(x) for x in d[m].tolist()] for m in metrics}
            cells = "   ".join(f"{m} {d[m].mean():+.3f} [{' '.join(f'{x:+.3f}' for x in d[m])}]"
                               for m in ("test_ap", "keptP10@0.10"))
            lines.append(f"    {name:<16} {cells}")
        out["blocks"][title] = block

    text = "\n".join(lines)
    (REPORT / "flag_ablation_table.txt").write_text(text + "\n", encoding="utf-8")
    (REPORT / "flag_ablation_table.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
