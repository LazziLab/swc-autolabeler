"""Quality flag in a mixed-cell deployment, from inference-time information only.

The canonical evaluation (``eval_flag``) scores held-out cells of one
ground-truth type with that type's flag. A deployment cannot see that type:
Stage 1 routes every cell, and the cell is scored by the flag of the type Stage 1
assigns it, pyramidal or interneuron. This driver replays that workflow on every
held-out cell of each seed with the saved leave-one-seed-out flag models (each
fitted on the other seeds' cells with this seed's test files removed), and
reports:

  routing      true interneurons routed to pyramidal and true pyramidal cells
               routed to interneuron, and the bad cells among them; a misrouted
               cell is scored by the other type's flag
  triage       at each review budget, the same fraction of each routed group:
               precision, recall of bad cells overall, of bad true-pyramidal and
               bad true-interneuron cells, and of the bad cells Stage 1 misroutes,
               and kept-set per-cell F1 P10 over all cells and over true-pyramidal
               cells
  controls     the same budget spent at random (mean over draws) and by an
               oracle that rejects the lowest-F1 cells of each routed group
  coverage     kept-set mean and P10 as the rejected fraction grows
  calibration  mean absolute error of the predicted per-cell F1 and a decile
               reliability table over all cells
  intervals    bootstrap 95% intervals at the 10% budget, cells resampled
               within each seed

A bad cell has per-cell neurite F1 below 0.60, as everywhere in the paper. As a
check, each flag's average precision on its own ground-truth type must reproduce
that type's canonical fold.

Usage:
    python -m evaluation.stage4.flag_deployment [--n-boot 1000]

Input:  evaluation/results/flag/final_flag_multiseed_{labels,features}.csv
        evaluation/results/flag/loso_models/{pyramidal,interneuron}_ours_oof_testseed{S}_regf1.joblib
        evaluation/results/flag/final_flag_leave_one_seed_out.csv
Output: evaluation/results/flag/flag_deployment.{json,txt}
        evaluation/results/flag/flag_deployment_percell.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.stage4.flag_model import F1_COL, _engineer_features, risk_scores  # noqa: E402
from evaluation.stage4.eval_flag import (                                        # noqa: E402
    FLAG_DIR,
    TARGET_THRESHOLD,
    _load_joined,
    _ours_oof_frame,
)

SEEDS = (42, 123, 789)
CELL_TYPES = ("pyramidal", "interneuron")
BUDGETS = (0.05, 0.10, 0.15, 0.20)
REJECT_FRACTIONS = tuple(round(0.02 * i, 2) for i in range(16))     # of each routed group, 0 .. 0.30
CELL_KEYS = ("risk", "f1", "bad", "true_pyr", "true_int", "routed")
CANONICAL = "final_flag_leave_one_seed_out.csv"


def _p10(x: np.ndarray) -> float:
    return float(np.quantile(x, 0.10)) if len(x) else float("nan")


def _share(hit: np.ndarray, pool: np.ndarray) -> float:
    return float(hit.sum() / pool.sum()) if pool.any() else float("nan")


def _load_model(cell_type: str, seed: int):
    """The fold's fitted pipeline (preprocessing and regressor), as eval_flag saved it.

    The bundle records which seed it held out. Scoring a fold on the cells it was
    fitted on is the one leak available here, so the file's own record is checked
    rather than trusting its name.
    """
    path = FLAG_DIR / "loso_models" / f"{cell_type}_ours_oof_testseed{seed}_regf1.joblib"
    obj = joblib.load(path)
    if isinstance(obj, dict):
        held = obj.get("test_seed_held_out")
        if held is not None and int(held) != int(seed):
            raise SystemExit(
                f"{path.name} held out seed {held}, not {seed}: scoring it on seed "
                f"{seed}'s cells would score cells it was fitted on.")
        return obj["rank_model"]
    return obj


def _reject_top(scores: np.ndarray, routed: np.ndarray, frac: float) -> np.ndarray:
    """Reject the same fraction of each routed group, its highest-scoring cells
    (stable sort, so deterministic)."""
    out = np.zeros(len(scores), dtype=bool)
    for group in (routed, ~routed):
        idx = np.flatnonzero(group)
        k = int(round(frac * len(idx)))
        out[idx[np.argsort(-scores[idx], kind="stable")[:k]]] = True
    return out


def _triage(c: dict, rejected: np.ndarray) -> dict:
    f1, bad, pyr, intn, routed = c["f1"], c["bad"], c["true_pyr"], c["true_int"], c["routed"]
    kept = ~rejected
    n_rej = int(rejected.sum())
    misrouted = (pyr & ~routed) | (intn & routed)
    return {
        "n_rejected": n_rej,
        "precision": float(bad[rejected].mean()) if n_rej else float("nan"),
        "recall_bad": _share(rejected & bad, bad),
        "recall_true_pyr_bad": _share(rejected & bad & pyr, bad & pyr),
        "recall_true_int_bad": _share(rejected & bad & intn, bad & intn),
        "recall_misrouted_bad": _share(rejected & bad & misrouted, bad & misrouted),
        "recall_pyr_routed_int_bad": _share(rejected & bad & pyr & ~routed, bad & pyr & ~routed),
        # Counts, so a mean over seeds or bootstrap draws is fractional by design.
        "interneurons_flagged": int((rejected & intn).sum()),
        "kept_mean_all": float(f1[kept].mean()),
        "kept_p10_all": _p10(f1[kept]),
        "kept_p10_true_pyr": _p10(f1[kept & pyr]),
        "kept_bad_share": float(bad[kept].mean()),
    }


def _seed_cells(df_feat: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame, dict]:
    """One seed's held-out cells, each scored by the flag of its routed type.

    The flag table carries the anatomy block for every true pyramidal cell, the
    rows the pyramidal flag is fitted and evaluated on. A deployment computes it
    only for cells Stage 1 routes to pyramidal, so here it is blanked for the
    rest before they are scored, and each flag sees what it would see in use.
    ``by_flag`` keeps the table's own rows, which reproduce the canonical folds.
    """
    part = df_feat[df_feat["model_seed"].astype(int) == seed].reset_index(drop=True)
    routed = part["stage1_pred"].astype(str).to_numpy() == "pyramidal"
    models = {ct: _load_model(ct, seed) for ct in CELL_TYPES}
    by_flag = {ct: np.asarray(risk_scores(models[ct], part), dtype=float) for ct in CELL_TYPES}
    deployed = part.copy()
    geom = [c for c in deployed.columns if c.startswith("geom_")]
    deployed.loc[~routed, geom] = np.nan
    in_use = {ct: np.asarray(risk_scores(models[ct], deployed), dtype=float) for ct in CELL_TYPES}
    f1 = pd.to_numeric(part[F1_COL], errors="coerce").to_numpy(float)
    ct = part["cell_type_gt"].astype(str).to_numpy()
    cells = {"risk": np.where(routed, in_use["pyramidal"], in_use["interneuron"]),
             "f1": f1, "bad": f1 < TARGET_THRESHOLD,
             "true_pyr": ct == "pyramidal", "true_int": ct == "interneuron", "routed": routed}
    return cells, part, by_flag


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if 0 < int(y.sum()) < len(y) else float("nan")


def _evaluate(c: dict, by_flag: dict, rng: np.random.Generator, n_random: int) -> dict:
    risk, f1, bad, pyr, intn, routed = (c[k] for k in CELL_KEYS)
    n = len(f1)
    out = {
        "n_cells": n, "n_true_pyr": int(pyr.sum()), "n_true_int": int(intn.sum()),
        "n_routed_pyr": int(routed.sum()), "n_routed_int": int((~routed).sum()),
        "routing": {
            "interneurons_routed_pyramidal": int((intn & routed).sum()),
            "interneurons_routed_pyramidal_bad": int((intn & routed & bad).sum()),
            "pyramidal_routed_interneuron": int((pyr & ~routed).sum()),
            "pyramidal_routed_interneuron_bad": int((pyr & ~routed & bad).sum()),
            "pyramidal_routed_interneuron_mean_f1": float(f1[pyr & ~routed].mean()) if (pyr & ~routed).any() else float("nan"),
            "true_pyramidal_bad": int((pyr & bad).sum()),
            "true_interneuron_bad": int((intn & bad).sum()),
        },
        "no_flag": {"mean_all": float(f1.mean()), "p10_all": _p10(f1), "p10_true_pyr": _p10(f1[pyr]),
                    "bad_share": float(bad.mean())},
        "ap_all_cells": _ap(bad, risk),
        # Each flag on its own ground-truth type: the canonical folds, reproduced.
        "ap_gt_pyramidal_cells": _ap(bad[pyr], by_flag["pyramidal"][pyr]),
        "ap_gt_interneuron_cells": _ap(bad[intn], by_flag["interneuron"][intn]),
        "budgets": {}, "coverage": {},
    }
    for bi, b in enumerate(BUDGETS):
        flag = _triage(c, _reject_top(risk, routed, b))
        oracle = _triage(c, _reject_top(-f1, routed, b))
        # A generator per budget, so the random control does not depend on the
        # order the budgets happen to be iterated in.
        brng = np.random.default_rng([rng.bit_generator.seed_seq.entropy or 0, bi])
        draws = [_triage(c, _reject_top(brng.random(n), routed, b)) for _ in range(n_random)]
        out["budgets"][f"{b:.2f}"] = {
            "fraction_of_all_cells": flag["n_rejected"] / n, "flag": flag, "oracle": oracle,
            "random": {key: float(np.nanmean([d[key] for d in draws])) for key in flag},
        }
    for fi, fr in enumerate(REJECT_FRACTIONS):
        flag = _triage(c, _reject_top(risk, routed, fr))
        oracle = _triage(c, _reject_top(-f1, routed, fr))
        crng = np.random.default_rng([rng.bit_generator.seed_seq.entropy or 0, 1000 + fi])
        draws = [_triage(c, _reject_top(crng.random(n), routed, fr)) for _ in range(max(1, n_random // 5))]
        out["coverage"][f"{fr:.2f}"] = {
            "flag": {"kept_mean_all": flag["kept_mean_all"], "kept_p10_all": flag["kept_p10_all"]},
            "random": {"kept_mean_all": float(np.mean([d["kept_mean_all"] for d in draws])),
                       "kept_p10_all": float(np.mean([d["kept_p10_all"] for d in draws]))},
            "oracle": {"kept_mean_all": oracle["kept_mean_all"], "kept_p10_all": oracle["kept_p10_all"]},
        }
    # risk_scores clips 1 - predicted_F1 into [0, 1], so a prediction above 1.0 has
    # already been flattened and this MAE is measured after that clip. It therefore
    # understates the error at the ceiling, where most of the ties sit.
    pred, true = 1.0 - risk, f1
    # Ten equal-count bins by rank. Predictions are clipped at 1.0 and many cells
    # tie there, so quantile edges would coincide and leave a bin empty.
    bins = np.empty(len(pred), dtype=int)
    bins[np.argsort(pred, kind="stable")] = np.arange(len(pred)) * 10 // max(1, len(pred))
    out["calibration"] = {
        "mae_predicted_f1": float(np.mean(np.abs(pred - true))),
        "mae_by_routed_type": {"pyramidal": float(np.mean(np.abs(pred[routed] - true[routed]))),
                               "interneuron": float(np.mean(np.abs(pred[~routed] - true[~routed])))},
        "deciles": [{"n": int((bins == i).sum()),
                     "predicted_f1": float(pred[bins == i].mean()) if (bins == i).any() else float("nan"),
                     "true_f1": float(true[bins == i].mean()) if (bins == i).any() else float("nan")}
                    for i in range(10)],
    }
    return out


def _bootstrap(cells: dict, n_boot: int, rng: np.random.Generator, budget: float = 0.10) -> dict:
    keys = ("precision", "recall_bad", "recall_true_pyr_bad", "recall_true_int_bad",
            "kept_p10_all", "kept_p10_true_pyr")
    reps = {k: [] for k in keys}
    for _ in range(n_boot):
        per_seed = {k: [] for k in keys}
        for c in cells.values():
            idx = rng.integers(0, len(c["f1"]), len(c["f1"]))
            cc = {k: c[k][idx] for k in CELL_KEYS}
            m = _triage(cc, _reject_top(cc["risk"], cc["routed"], budget))
            for k in keys:
                per_seed[k].append(m[k])
        for k in keys:
            reps[k].append(float(np.nanmean(per_seed[k])))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in reps.items()}


def _mean_tree(items: list):
    first = items[0]
    if isinstance(first, dict):
        return {k: _mean_tree([it[k] for it in items]) for k in first if all(k in it for it in items)}
    if isinstance(first, list):
        if all(isinstance(it, list) and len(it) == len(first) for it in items):
            return [_mean_tree([it[i] for it in items]) for i in range(len(first))]
        return None
    if isinstance(first, bool):
        return all(items)
    if isinstance(first, (int, float)):
        vals = [float(v) for v in items if v == v]
        return float(np.mean(vals)) if vals else float("nan")
    return first


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=FLAG_DIR / "final_flag_multiseed_labels.csv")
    ap.add_argument("--features", type=Path, default=FLAG_DIR / "final_flag_multiseed_features.csv")
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=FLAG_DIR)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df_feat, _, _ = _engineer_features(_ours_oof_frame(_load_joined(args.labels, args.features)))
    path = FLAG_DIR / CANONICAL
    if not path.is_file():
        raise SystemExit(f"MISSING {path}\n  run: python -m evaluation.stage4.eval_flag")
    loso = pd.read_csv(path)
    canonical = {ct: loso[loso["scope"].astype(str) == ct] for ct in CELL_TYPES}
    if any(v.empty for v in canonical.values()):
        raise SystemExit(f"{path.name} lacks a cell type; rerun eval_flag for both")
    cells, per_seed, percell = {}, {}, []
    for s in SEEDS:
        c, part, by_flag = _seed_cells(df_feat, s)
        cells[s] = c
        m = _evaluate(c, by_flag, rng, args.n_random)
        m["reproduces_canonical_folds"] = {}
        for ct in CELL_TYPES:
            canon = float(canonical[ct].loc[canonical[ct]["test_seed"].astype(int) == s, "test_ap"].iloc[0])
            m[f"canonical_ap_{ct}"] = canon
            m["reproduces_canonical_folds"][ct] = bool(abs(m[f"ap_gt_{ct}_cells"] - canon) < 1e-6)
        per_seed[str(s)] = m
        percell.append(pd.DataFrame({"model_seed": s, "file": part["file"], "cell_type_gt": part["cell_type_gt"],
                                     "stage1_pred": part["stage1_pred"], "held_out_F1": c["f1"],
                                     "risk": c["risk"], "risk_pyramidal_flag": by_flag["pyramidal"],
                                     "risk_interneuron_flag": by_flag["interneuron"]}))
        print(f"seed {s}: {m['n_cells']} cells, {m['n_routed_pyr']} routed pyramidal, "
              f"{m['n_routed_int']} routed interneuron; AP on GT-pyramidal {m['ap_gt_pyramidal_cells']:.4f}, "
              f"on GT-interneuron {m['ap_gt_interneuron_cells']:.4f}", flush=True)
    mean = _mean_tree(list(per_seed.values()))
    ci = _bootstrap(cells, args.n_boot, rng)

    L = ["QUALITY FLAG IN A MIXED-CELL DEPLOYMENT -- Stage 1 routing, one flag per routed type,",
         "leave-one-seed-out models, mean over seeds", "", "Routing (per seed, mean):"]
    for k, v in mean["routing"].items():
        L.append(f"  {k:<40} {v:9.2f}")
    L += ["", f"  no flag: mean F1 {mean['no_flag']['mean_all']:.4f}, P10 all {mean['no_flag']['p10_all']:.4f}, "
              f"P10 true pyramidal {mean['no_flag']['p10_true_pyr']:.4f}, bad share {mean['no_flag']['bad_share']:.4f}",
          f"  AP on all cells {mean['ap_all_cells']:.4f}; pyramidal flag on GT-pyramidal "
          f"{mean['ap_gt_pyramidal_cells']:.4f}; interneuron flag on GT-interneuron {mean['ap_gt_interneuron_cells']:.4f}",
          "", "Review budgets (the same fraction of each routed group):",
          f"  {'budget':>6} {'prec':>6} {'rec all':>7} {'rec pyr':>7} {'rec int':>7} {'rec mis':>7} "
          f"{'keptP10 all':>11} {'keptP10 pyr':>11} {'kept bad':>8} | {'random P10 pyr':>14} {'oracle P10 pyr':>14}"]
    for b, row in mean["budgets"].items():
        f, r, o = row["flag"], row["random"], row["oracle"]
        L.append(f"  {b:>6} {f['precision']:6.3f} {f['recall_bad']:7.3f} {f['recall_true_pyr_bad']:7.3f} "
                 f"{f['recall_true_int_bad']:7.3f} {f['recall_misrouted_bad']:7.3f} {f['kept_p10_all']:11.4f} "
                 f"{f['kept_p10_true_pyr']:11.4f} {f['kept_bad_share']:8.4f} | {r['kept_p10_true_pyr']:14.4f} "
                 f"{o['kept_p10_true_pyr']:14.4f}")
    L += ["", "Bootstrap 95% intervals at the 10% budget (mean over seeds):"]
    for k, (lo, hi) in ci.items():
        L.append(f"  {k:<24} [{lo:.4f}, {hi:.4f}]")
    L += ["", f"Calibration over all cells: MAE of predicted F1 {mean['calibration']['mae_predicted_f1']:.4f} "
              f"(routed pyramidal {mean['calibration']['mae_by_routed_type']['pyramidal']:.4f}, "
              f"routed interneuron {mean['calibration']['mae_by_routed_type']['interneuron']:.4f})",
          "  decile  predicted  true"]
    for i, d in enumerate(mean["calibration"]["deciles"]):
        L.append(f"  {i + 1:>6}  {d['predicted_f1']:.4f}  {d['true_f1']:.4f}")
    ok = {s: all(per_seed[s]["reproduces_canonical_folds"].values()) for s in per_seed}
    L += ["", "Reproduction of the canonical folds (each flag's AP on its own GT type): "
          + ", ".join(f"seed {s} {'OK' if ok[s] else 'MISMATCH'}" for s in per_seed)]
    text = "\n".join(L)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "flag_deployment.json").write_text(json.dumps(
        {"seeds": list(SEEDS), "budgets": list(BUDGETS), "bad_threshold": TARGET_THRESHOLD,
         "n_random": args.n_random, "n_boot": args.n_boot, "per_seed": per_seed, "mean": mean,
         "bootstrap_ci_budget_0.10": ci}, indent=2), encoding="utf-8")
    (args.out_dir / "flag_deployment.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat(percell).to_csv(args.out_dir / "flag_deployment_percell.csv", index=False)
    print("\n" + text)
    if not all(ok.values()):
        print("\nERROR: rescoring does not reproduce the canonical folds", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
