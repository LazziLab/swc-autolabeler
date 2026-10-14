"""Factorial ablation of Stage 3's topology rules and branch correction.

The stage ladder adds the correction after the rules, and
the correction can overturn a rule's decision, so the ladder alone cannot
separate the two components' effects from their interaction. This driver runs
the 2x2 design on every held-out cell of each seed under ground-truth cell type:

    neither            Stage 3 with every rule disabled, no correction
    rules only         Stage 3 rules, no correction
    correction only    every rule disabled, branch correction applied
    both               the shipped pipeline

Two pipeline passes per cell give all four: the shipped pass, whose trace holds
the labels before the correction, and a pass with every rule disabled. Reported
per configuration: node accuracy, neurite macro-F1, per-class F1, per-cell F1
mean and P10, over all cells and over pyramidal cells (the correction's scope);
then the main effects and the interaction, per seed and as a three-seed mean.
The rules-only and both rows must reproduce the stage ladder's rows.

Usage:
    python -m evaluation.factorial_ablation [--seeds 42,123,789] [--jobs 3]

Output (evaluation/results/): factorial_ablation.{json,txt}
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
RESULTS = ROOT / "evaluation" / "results"
CONFIGS = ("neither", "rules_only", "correction_only", "both")
METRICS = ("accuracy", "neurite_macro_f1", "axon_f1", "basal_f1", "apical_f1", "pcf1_mean", "pcf1_p10")
LADDER_MODE = {"rules_only": "stage3_topology", "both": "full_correction"}


def _row(rep: dict) -> dict:
    c = rep["corpus"]
    pc = rep["per_cell"]["neurite_macro_f1"]
    return {"accuracy": c["accuracy"], "neurite_macro_f1": c["neurite_macro_f1"],
            "axon_f1": c["per_class"]["axon"]["f1"], "basal_f1": c["per_class"]["basal/dendrite"]["f1"],
            "apical_f1": c["per_class"]["apical"]["f1"], "pcf1_mean": pc["mean"], "pcf1_p10": pc["p10"]}


def _report(name: str, recs: list[dict]) -> dict:
    from evaluation.baselines.baseline_report import build_full_report
    cells = [{"file": r["file"], "cell_type": r["cell_type"], "n_nodes": r["n_nodes"],
              "gt": r["gt"].tolist(), "pred": r["pred"].tolist()} for r in recs]
    gt_pool = [g for r in cells for g in r["gt"]]
    pred_pool = [p for r in cells for p in r["pred"]]
    return build_full_report(name, gt_pool, pred_pool, cells)


def eval_seed(task: tuple[int, int, float]) -> tuple[int, dict]:
    from pipeline.features import parse_swc
    from pipeline.infer_pipeline import run_pipeline_on_nodes
    from pipeline.stage3.refine import RULES
    from pipeline.split import in_test
    from evaluation.stage4._flag_metrics import check_model_seed

    seed, limit, test_size = task
    md = ROOT / "pipeline" / "models" / f"seed{seed}"
    # Refuse a model trained on a different split than the one being scored.
    check_model_seed(md, seed, test_size)
    s1, s2 = md / "cell_type_classifier.pkl", md / "branch_classifier.pkl"
    files = [(ct, p) for ct in ("pyramidal", "interneuron")
             for p in sorted((DATA / ct).glob("*.swc")) if in_test(p.name, seed, test_size)]
    if limit:
        files = ([f for f in files if f[0] == "pyramidal"][:limit]
                 + [f for f in files if f[0] == "interneuron"][:limit])
    print(f"seed {seed}: {len(files)} test cells", flush=True)
    recs: dict[str, list[dict]] = {c: [] for c in CONFIGS}
    t0 = time.perf_counter()
    for i, (ct, p) in enumerate(files, 1):
        nodes = parse_swc(p)
        if not nodes:
            continue
        gt = np.asarray([nd.type for nd in nodes], dtype=np.int8)
        tr_full: dict = {}
        tr_none: dict = {}
        full = run_pipeline_on_nodes(nodes, str(p), s1, s2, override_cell_type=ct, trace=tr_full)
        none = run_pipeline_on_nodes(nodes, str(p), s1, s2, override_cell_type=ct, trace=tr_none,
                                     disable_stage3=frozenset(RULES))
        preds = {"neither": tr_none["final_labels"], "rules_only": tr_full["final_labels"],
                 "correction_only": none.node_labels, "both": full.node_labels}
        for c in CONFIGS:
            recs[c].append({"file": p.name, "cell_type": ct, "n_nodes": len(nodes), "gt": gt,
                            "pred": np.asarray(preds[c], dtype=np.int8)})
        if i % 300 == 0:
            print(f"  seed {seed}: {i}/{len(files)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    out: dict = {"n_cells": len(recs["both"])}
    for c in CONFIGS:
        out[c] = {"all": _row(_report(c, recs[c])),
                  "pyramidal": _row(_report(c, [r for r in recs[c] if r["cell_type"] == "pyramidal"]))}
        recs[c] = []
    print(f"seed {seed}: done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return seed, out


def _effects(rows: dict) -> dict:
    eff = {}
    for m in METRICS:
        n, r, c, b = (rows[k][m] for k in CONFIGS)
        eff[m] = {"rules_without_correction": r - n, "rules_with_correction": b - c,
                  "correction_without_rules": c - n, "correction_with_rules": b - r,
                  "interaction": (b - c) - (r - n)}
    return eff


def _mean_sd(vals: list[float]) -> dict:
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--jobs", type=int, default=3, help="seeds evaluated in parallel")
    ap.add_argument("--limit", type=int, default=0, help="cells per cell type per seed (0 = all; smoke tests)")
    ap.add_argument("--test-size", type=float, default=0.20,
                    help="held-out fraction of the hash split (default 0.20)")
    ap.add_argument("--no-ladder-check", action="store_true",
                    help="run without cross-checking rules_only and both against the "
                         "stage ladder; the result is then unvalidated")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Read the ladder before the work, not after it. Cross-checking rules_only
    # and both against it is this run's only correctness check, and the run takes
    # hours; discovering the file is missing at the end would waste all of it.
    ladder: dict = {}
    lp = RESULTS / "ablation_ladder.json"
    skip_check = bool(args.limit) or args.no_ladder_check
    if not skip_check:
        if not lp.is_file():
            raise SystemExit(
                f"MISSING {lp}\nrules_only and both are cross-checked against the stage "
                f"ladder, which is the only check on this run. Produce it with\n"
                f"  python -m evaluation.eval_ablation --seeds {args.seeds}\n"
                f"or pass --no-ladder-check to run unvalidated.")
        ladder = json.loads(lp.read_text(encoding="utf-8")).get("per_seed", {})

    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(seeds)))) as ex:
        per_seed = dict(ex.map(eval_seed, [(s, args.limit, args.test_size) for s in seeds]))

    checks = {}
    for s in seeds:
        lad = ladder.get(str(s), {})
        for cfg, mode in LADDER_MODE.items():
            if mode in lad:
                checks[f"seed{s}/{cfg}"] = max(abs(per_seed[s][cfg]["all"][m] - lad[mode][k])
                                               for m, k in (("accuracy", "accuracy"),
                                                            ("neurite_macro_f1", "neurite_macro_f1"),
                                                            ("pcf1_p10", "pcf1_p10")))

    summary: dict = {}
    for scope in ("all", "pyramidal"):
        rows = {c: {m: _mean_sd([per_seed[s][c][scope][m] for s in seeds]) for m in METRICS} for c in CONFIGS}
        eff_seed = {s: _effects({c: per_seed[s][c][scope] for c in CONFIGS}) for s in seeds}
        effects = {m: {k: _mean_sd([eff_seed[s][m][k] for s in seeds]) for k in eff_seed[seeds[0]][m]}
                   for m in METRICS}
        summary[scope] = {"configs": rows, "effects": effects,
                          "effects_per_seed": {str(s): eff_seed[s] for s in seeds}}

    L = [f"FACTORIAL ABLATION -- Stage 3 rules x branch correction, GT cell type, mean (SD) over seeds "
         f"{', '.join(map(str, seeds))}"]
    for scope in ("all", "pyramidal"):
        L += ["", f"{scope.upper()} CELLS",
              f"  {'configuration':<16}" + "".join(f"{m:>18}" for m in METRICS)]
        for c in CONFIGS:
            r = summary[scope]["configs"][c]
            L.append(f"  {c:<16}" + "".join(f"{r[m]['mean']:>10.4f} ({r[m]['sd']:.4f})" for m in METRICS))
        L.append("  effects (mean over seeds):")
        for k in ("rules_without_correction", "rules_with_correction", "correction_without_rules",
                  "correction_with_rules", "interaction"):
            e = summary[scope]["effects"]
            L.append(f"    {k:<26}" + "".join(f"{e[m][k]['mean']:>+10.4f} ({e[m][k]['sd']:.4f})" for m in METRICS))
    if checks:
        L += ["", "Reproduction of the stage ladder (max abs difference): "
              + ", ".join(f"{k} {v:.2e}" for k, v in checks.items())]
    else:
        L += ["", "NOT CROSS-CHECKED against the stage ladder: "
              + ("subsampled run" if args.limit else
                 "--no-ladder-check" if args.no_ladder_check else
                 "the ladder holds no entry for these seeds")]
    text = "\n".join(L)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_limit{args.limit}" if args.limit else ""     # a smoke run keeps its own files
    (args.out_dir / f"factorial_ablation{tag}.json").write_text(json.dumps(
        {"seeds": seeds, "configs": list(CONFIGS), "metrics": list(METRICS),
         "per_seed": {str(s): per_seed[s] for s in seeds}, "summary": summary,
         "ladder_reproduction_max_abs_diff": checks}, indent=2), encoding="utf-8")
    (args.out_dir / f"factorial_ablation{tag}.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    if any(v > 1e-9 for v in checks.values()):
        print("\nERROR: rules-only or both does not reproduce the stage ladder", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
