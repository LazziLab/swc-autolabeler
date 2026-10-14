"""Does the quality flag transfer to laboratories outside its training
distribution?

Deployment order on the external set: Stage 1 routes each cell, the flag of the
routed type scores it, pyramidal or interneuron, and triage rejects the most
suspicious fraction. Both external sets are run, pyramidal and interneuron, so
the workflow is also evaluated on a mixed batch. Six questions, one pass:

  1  Triage. Kept per-cell F1 mean and P10 at each rejection rate over the
     pyramidal set's cells routed to pyramidal, pooled and per archive, against
     a random rejection of the same size and an oracle, with bootstrap 95%
     intervals and a risk-coverage curve. In-corpus triage lifts the kept P10
     substantially; the lift here measures how much of that survives the
     distribution shift.
  2  Interneuron triage. The same for the interneuron set's cells routed to
     interneuron, with the recall of bad cells, since their P10 sits near 1.0.
  3  Misrouting. Pyramidal-set cells Stage 1 routes to interneuron are scored by
     the interneuron flag. Their count and per-cell F1 quantify the failure mode,
     and the mixed batch below counts how many of the bad ones reach review.
  4  Mixed batch. With both sets pooled and the same fraction of each routed
     group rejected: recall of bad pyramidal-set and interneuron-set cells
     (per-cell F1 < 0.60), of the bad pyramidal-set cells routed to interneuron,
     interneurons rejected, and kept-set P10.
  5  Calibration. Mean absolute error of the flag's predicted per-cell F1 and a
     decile reliability table over the scored cells.
  6  Routing gate. Stage 1's own confidence, recorded for every cell in the
     same pass, is evaluated as a guard: at each threshold, the share of
     misrouted cells caught against the share of correctly routed cells
     sacrificed.

Usage:
    python -m evaluation.generalization.flag_transfer [--seed 123]

Output (evaluation/results/generalization/):
    flag_transfer.json
    flag_transfer_percell.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                   # noqa: E402
from pipeline.infer_pipeline import (                                     # noqa: E402
    run_pipeline_on_nodes, load_branch_correction,
)
from pipeline.stage4.single_seed_flag import score_flag, load_flag        # noqa: E402
from evaluation.stage4._flag_metrics import per_cell_neurite_f1           # noqa: E402

EXTERNAL = ROOT / "data" / "external"
RESULTS = ROOT / "evaluation" / "results" / "generalization"
REJECT_RATES = (0.10, 0.15, 0.20, 0.30)
GATE_THRESHOLDS = (0.70, 0.80, 0.90, 0.95)
COVERAGE = tuple(round(0.02 * i, 2) for i in range(16))       # rejected fraction, 0 .. 0.30
BAD_F1 = 0.60
N_RANDOM = 1000
N_BOOT = 1000


def _p10(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.percentile(v, 10)) if len(v) else float("nan")


def _mean(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.mean()) if len(v) else float("nan")


def _kept_mask(scores: np.ndarray, k: int) -> np.ndarray:
    mask = np.ones(len(scores), dtype=bool)
    if k:
        mask[np.argsort(-scores, kind="stable")[:k]] = False
    return mask


def _suspicion(rs: list[dict]) -> np.ndarray:
    """Suspicion scores, with a cell no flag scored placed last in line."""
    return np.array([-np.inf if r["suspicion"] is None else r["suspicion"] for r in rs], dtype=float)


def _run_set(ct: str, s1, s2, flag, rows: list[dict], t0: float, limit: int) -> None:
    meta = list(csv.DictReader((EXTERNAL / f"{ct}.csv").open(encoding="utf-8", newline="")))
    if limit:
        meta = meta[:limit]
    for i, m in enumerate(meta, 1):
        nodes = parse_swc(EXTERNAL / ct / m["file_name"])
        if not nodes:
            continue
        path = EXTERNAL / ct / m["file_name"]
        pr = run_pipeline_on_nodes(nodes, str(path), s1, s2)
        sc = score_flag(nodes, pr, flag)
        gt = np.fromiter((n.type for n in nodes), dtype=np.int8, count=len(nodes))
        pred = np.fromiter(pr.node_labels, dtype=np.int8, count=len(nodes))
        if ct == "interneuron":
            # As eval_external: interneurons have no apical, so collapse it on
            # both sides rather than scoring a prediction against a label set
            # without it.
            gt = np.where(gt == 4, 3, gt)
            pred = np.where(pred == 4, 3, pred)
        rows.append({"file": m["file_name"], "archive": m["archive"],
                     "species": m["species"], "routed": pr.stage1.cell_type,
                     "stage1_conf": float(pr.stage1.confidence),
                     "pcf1": per_cell_neurite_f1(gt, pred, ct),
                     "suspicion": sc["suspicion"], "cell_set": ct,
                     "predicted_f1": sc.get("predicted_f1")})
        if i % 250 == 0:
            print(f"  {ct} {i}/{len(meta)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)


def _calibration(rs: list[dict]) -> dict:
    pred = np.array([np.nan if r["predicted_f1"] is None else r["predicted_f1"] for r in rs], dtype=float)
    true = np.array([r["pcf1"] for r in rs], dtype=float)
    ok = np.isfinite(pred)
    pred, true = pred[ok], true[ok]
    if not len(pred):
        return {"n": 0}
    # Deciles by rank, not by quantile edges: predicted F1 piles up near the
    # ceiling, and equal edges there collapse whole bins to empty, which the
    # table would then report as nan rows.
    order = np.argsort(pred, kind="stable")
    bins = np.empty(len(pred), dtype=int)
    bins[order] = np.minimum(9, (np.arange(len(pred)) * 10) // max(1, len(pred)))
    return {"n": int(len(pred)), "mae_predicted_f1": float(np.mean(np.abs(pred - true))),
            "deciles": [{"n": int((bins == i).sum()),
                         "predicted_f1": float(pred[bins == i].mean()) if (bins == i).any() else float("nan"),
                         "true_f1": float(true[bins == i].mean()) if (bins == i).any() else float("nan")}
                        for i in range(10)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0, help="cells per external set (0 = all; smoke tests)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    mdir = ROOT / "pipeline" / "models" / f"seed{args.seed}"
    s1, s2 = mdir / "cell_type_classifier.pkl", mdir / "branch_classifier.pkl"
    # No check_model_seed: these cells come from laboratories outside the corpus,
    # so no seed's held-out split bears on them. The seed selects which trained
    # flag is being transferred, not which cells are legitimate to score.
    if load_branch_correction(mdir) is None:
        raise SystemExit(f"MISSING branch_correction.joblib in {mdir}")
    flag = load_flag(mdir)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for ct in ("pyramidal", "interneuron"):
        if (EXTERNAL / f"{ct}.csv").is_file():
            _run_set(ct, s1, s2, flag, rows, t0, args.limit)

    pyr_rows = [r for r in rows if r["cell_set"] == "pyramidal"]
    scoped = [r for r in pyr_rows if r["routed"] == "pyramidal"]
    misrouted = [r for r in pyr_rows if r["routed"] != "pyramidal"]
    print(f"\npyramidal set: {len(scoped)} cells routed to pyramidal, {len(misrouted)} to interneuron "
          f"(their pcF1 mean {_mean([r['pcf1'] for r in misrouted]):.3f})")

    def kept(rs: list[dict], reject: float) -> tuple[float, float]:
        # Through _kept_mask, so this table and the controls below break ties
        # the same way and reject the same cells at the same rate.
        sus = _suspicion(rs)
        f1 = np.array([r["pcf1"] for r in rs], dtype=float)
        mask = _kept_mask(sus, int(round(reject * len(rs))))
        return float(f1[mask].mean()), float(np.percentile(f1[mask], 10))

    triage = {}
    print(f"\n{'reject':>8s} {'kept mean':>10s} {'kept P10':>9s}")
    for rej in (0.0,) + REJECT_RATES:
        mean_, p10 = kept(scoped, rej)
        triage[f"{rej:.2f}"] = {"mean": mean_, "p10": p10}
        print(f"{rej:8.0%} {mean_:10.3f} {p10:9.3f}")

    per_archive = {}
    print("\nper-archive kept P10, no triage vs 20% rejection:")
    for arch in sorted({r["archive"] for r in scoped}):
        sub = [r for r in scoped if r["archive"] == arch]
        _, p0 = kept(sub, 0.0)
        _, p2 = kept(sub, 0.20)
        per_archive[arch] = {"n": len(sub), "p10": p0, "p10_reject20": p2}
        print(f"  {arch:30s} n={len(sub):3d}  {p0:.3f} -> {p2:.3f}")

    # Controls, coverage and intervals over the pyramidal set's scored cells.
    S = _suspicion(scoped)
    F = np.array([r["pcf1"] for r in scoped], dtype=float)
    controls, coverage, intervals = {}, {}, {}
    print(f"\n{'reject':>8s} {'flag P10':>9s} {'random P10':>11s} {'oracle P10':>11s}")
    for rej in (0.0,) + REJECT_RATES:
        k = int(round(rej * len(F)))
        rand = [_p10(F[_kept_mask(rng.random(len(F)), k)]) for _ in range(N_RANDOM)]
        c = {"flag_p10": _p10(F[_kept_mask(S, k)]), "random_p10": _mean(rand),
             "oracle_p10": _p10(F[_kept_mask(-F, k)])}
        controls[f"{rej:.2f}"] = c
        print(f"{rej:8.0%} {c['flag_p10']:9.3f} {c['random_p10']:11.3f} {c['oracle_p10']:11.3f}")
    for fr in COVERAGE:
        k = int(round(fr * len(F)))
        draws = [_kept_mask(rng.random(len(F)), k) for _ in range(200)]
        fm, om = _kept_mask(S, k), _kept_mask(-F, k)
        coverage[f"{fr:.2f}"] = {
            "flag": {"mean": _mean(F[fm]), "p10": _p10(F[fm])},
            "random": {"mean": _mean([_mean(F[d]) for d in draws]), "p10": _mean([_p10(F[d]) for d in draws])},
            "oracle": {"mean": _mean(F[om]), "p10": _p10(F[om])},
        }
    for rej in (0.10, 0.20):
        reps = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(F), len(F))
            reps.append(_p10(F[idx][_kept_mask(S[idx], int(round(rej * len(F))))]))
        intervals[f"{rej:.2f}"] = [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]
    print("bootstrap 95% interval of kept P10: "
          + ", ".join(f"{k} rejected [{v[0]:.3f}, {v[1]:.3f}]" for k, v in intervals.items()))

    # Interneuron set: its cells routed to interneuron, scored by the interneuron flag.
    int_rows = [r for r in rows if r["cell_set"] == "interneuron"]
    int_scoped = [r for r in int_rows if r["routed"] != "pyramidal"]
    SI = _suspicion(int_scoped)
    FI = np.array([r["pcf1"] for r in int_scoped], dtype=float)
    bad_i = FI < BAD_F1
    triage_interneuron = {}
    print(f"\ninterneuron set: {len(int_scoped)} cells routed to interneuron, {int(bad_i.sum())} bad")
    print(f"{'reject':>8s} {'kept mean':>10s} {'kept bad':>9s} {'recall':>7s} {'random mean':>12s}")
    for rej in (0.0,) + REJECT_RATES:
        k = int(round(rej * len(FI)))
        m = _kept_mask(SI, k)
        rand = [_mean(FI[_kept_mask(rng.random(len(FI)), k)]) for _ in range(N_RANDOM)]
        t = {"mean": _mean(FI[m]), "bad_share": _mean(bad_i[m]),
             "recall_bad": float((bad_i & ~m).sum() / bad_i.sum()) if bad_i.any() else float("nan"),
             "random_mean": _mean(rand), "oracle_mean": _mean(FI[_kept_mask(-FI, k)])}
        triage_interneuron[f"{rej:.2f}"] = t
        print(f"{rej:8.0%} {t['mean']:10.3f} {t['bad_share']:9.3f} {t['recall_bad']:7.3f} {t['random_mean']:12.3f}")

    # Mixed batch: both sets pooled, the same fraction of each routed group rejected.
    groups = ([r for r in rows if r["routed"] == "pyramidal"], [r for r in rows if r["routed"] != "pyramidal"])
    n_bad_pyr = sum(1 for r in pyr_rows if r["pcf1"] < BAD_F1)
    n_bad_int = sum(1 for r in int_rows if r["pcf1"] < BAD_F1)
    n_bad_mis = sum(1 for r in misrouted if r["pcf1"] < BAD_F1)
    mixed = {"n_cells": len(rows), "n_pyramidal_set": len(pyr_rows), "n_interneuron_set": len(int_rows),
             "n_routed_pyramidal": len(groups[0]), "n_routed_interneuron": len(groups[1]),
             "interneurons_routed_pyramidal": sum(1 for r in int_rows if r["routed"] == "pyramidal"),
             "pyramidal_routed_interneuron": len(misrouted),
             "pyramidal_routed_interneuron_bad": n_bad_mis,
             "pyramidal_bad": n_bad_pyr, "interneuron_bad": n_bad_int,
             "no_flag_p10_all": _p10([r["pcf1"] for r in rows]),
             "no_flag_p10_pyramidal_set": _p10([r["pcf1"] for r in pyr_rows]),
             "budgets": {}}
    print(f"\nmixed batch: {len(rows)} cells, {mixed['interneurons_routed_pyramidal']} interneurons routed "
          f"to pyramidal, {len(misrouted)} pyramidal cells routed to interneuron ({n_bad_mis} bad)")
    for rej in REJECT_RATES:
        rejected = []
        for g in groups:
            order = np.argsort(-_suspicion(g), kind="stable")[:int(round(rej * len(g)))]
            rejected += [g[i] for i in order]
        rejected_ids = {id(r) for r in rejected}
        kept_rows = [r for r in rows if id(r) not in rejected_ids]
        bad_rej = [r for r in rejected if r["pcf1"] < BAD_F1]
        b = {"k": len(rejected),
             "precision": _mean([r["pcf1"] < BAD_F1 for r in rejected]),
             "recall_pyramidal_bad": sum(1 for r in bad_rej if r["cell_set"] == "pyramidal") / max(1, n_bad_pyr),
             "recall_interneuron_bad": sum(1 for r in bad_rej if r["cell_set"] == "interneuron") / max(1, n_bad_int),
             "recall_misrouted_pyramidal_bad": sum(1 for r in bad_rej if r["cell_set"] == "pyramidal"
                                                   and r["routed"] != "pyramidal") / max(1, n_bad_mis),
             "interneurons_rejected": sum(1 for r in rejected if r["cell_set"] == "interneuron"),
             "good_interneurons_rejected": sum(1 for r in rejected if r["cell_set"] == "interneuron"
                                               and r["pcf1"] >= BAD_F1),
             "kept_p10_all": _p10([r["pcf1"] for r in kept_rows]),
             "kept_p10_pyramidal_set": _p10([r["pcf1"] for r in kept_rows if r["cell_set"] == "pyramidal"])}
        mixed["budgets"][f"{rej:.2f}"] = b
        print(f"  reject {rej:4.0%} of each routed group: precision {b['precision']:.3f}  "
              f"recall pyramidal {b['recall_pyramidal_bad']:.3f}  interneuron {b['recall_interneuron_bad']:.3f}  "
              f"misrouted {b['recall_misrouted_pyramidal_bad']:.3f}  kept P10 all {b['kept_p10_all']:.3f}")

    calibration = {"pyramidal_set_scored": _calibration(scoped),
                   "interneuron_set_scored": _calibration(int_scoped),
                   "all_scored": _calibration(rows)}
    print(f"\ncalibration: MAE of predicted per-cell F1 "
          f"{calibration['pyramidal_set_scored'].get('mae_predicted_f1', float('nan')):.3f} (pyramidal set), "
          f"{calibration['interneuron_set_scored'].get('mae_predicted_f1', float('nan')):.3f} (interneuron set), "
          f"{calibration['all_scored'].get('mae_predicted_f1', float('nan')):.3f} (all cells)")

    conf_mis = np.array([r["stage1_conf"] for r in misrouted])
    conf_ok = np.array([r["stage1_conf"] for r in scoped])
    gate = {}
    print("\nStage 1 confidence gate:")
    for thr in GATE_THRESHOLDS:
        caught, cost = _mean(conf_mis < thr), _mean(conf_ok < thr)
        gate[f"{thr:.2f}"] = {"caught": caught, "cost": cost}
        print(f"  gate at {thr:.2f}: catches {caught:5.1%} of misrouted, "
              f"costs {cost:5.1%} of correct")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "flag_transfer.json").open("w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "n": len(pyr_rows), "n_scoped": len(scoped),
                   "n_routed_interneuron": len(misrouted),
                   "routed_interneuron_pcf1_mean": _mean([r["pcf1"] for r in misrouted]),
                   "triage": triage, "per_archive": per_archive, "gate": gate,
                   "controls": controls, "coverage": coverage, "bootstrap_ci_kept_p10": intervals,
                   "interneuron_set": {"n_scoped": len(int_scoped), "n_bad": int(bad_i.sum()),
                                       "triage": triage_interneuron},
                   "mixed_batch": mixed, "calibration": calibration,
                   "n_random": N_RANDOM, "n_boot": N_BOOT, "bad_f1_threshold": BAD_F1},
                  fh, indent=2)
    with (args.out_dir / "flag_transfer_percell.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "archive", "species", "routed", "stage1_conf",
                    "pcf1", "suspicion", "cell_set", "predicted_f1"])
        for r in rows:
            w.writerow([r["file"], r["archive"], r["species"], r["routed"],
                        f"{r['stage1_conf']:.4f}", f"{r['pcf1']:.6f}",
                        "" if r["suspicion"] is None else f"{r['suspicion']:.6f}", r["cell_set"],
                        "" if r["predicted_f1"] is None else f"{r['predicted_f1']:.6f}"])
    print(f"\nwrote {args.out_dir / 'flag_transfer.json'} ({(time.perf_counter() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
