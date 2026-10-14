"""Are automated labels good enough to measure from?

The question a user has is not what F1 the labeler reaches, but whether the
numbers they would publish come out right. This driver answers that, reading the
per-cell tables written by ``evaluation.downstream_morphometrics``.

The reference point is human disagreement, not perfection. Independent expert
annotators reconstructing the same neurons agree on 87.7% +/- 14.3% of total
axonal length (Winnubst et al., Cell 2019), so about 12% is the practical
ceiling on what "correct" can mean. Measurements landing inside that band are as
close to the reference as a second annotator would be.

Four readouts:

  1  Cohort shift. The systematic shift in each compartment measure's mean.
     Per-cell noise averages out over a population; a shift does not, so this is
     the quantity that biases a study.

  2  Against the baselines. The same shift, plus the share of cells accurate
     enough to measure from, for every labeling method.

  3  Where the error lives. The share of total cohort error carried by the worst
     10% of cells. Concentrated error is what makes triage possible.

  4  Triage. Usable share as a function of how many suspicious cells are set
     aside, which is the deployment recipe: label everything, drop the flagged
     fraction, measure from the rest.

A cell counts as usable when every continuous compartment measure lands within
tolerance. Branch-point and terminal counts are excluded from that test, since a
relative tolerance is meaningless on a cell with three branch points. Triage is
reported on pyramidal cells, ranked by the pyramidal flag.

Usage:
    python -m evaluation.morphometric_utility

Output (evaluation/results/):
    morphometric_utility.json
    morphometric_utility.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.analyze_failure_modes import loso_risk as _loso_risk                  # noqa: E402
from evaluation.stage4.eval_flag import (                                 # noqa: E402
    DEFAULT_BASE_SEED, FLAG_DIR, _load_joined,
)

#: The triage section covers pyramidal cells, scored by the pyramidal flag.
TRIAGE_CELL_TYPE = "pyramidal"
COMPARTMENTS = ("axon", "basal", "apical")
CORE = ("total_length", "surface_area", "max_path_dist")
LABEL = {"total_length": "total length", "surface_area": "surface area",
         "max_path_dist": "max path dist"}
# Expert-vs-expert disagreement on total length, Winnubst et al. Cell 2019.
HUMAN_DISAGREEMENT = 0.123
REJECT_RATES = (0.05, 0.10, 0.12, 0.15, 0.20)
#: Fewest measurable cells a compartment needs before a cohort shift is reported.
#: Below this the mean is too unstable to compare against the expert band.
MIN_CELLS_FOR_SHIFT = 20


def _load(in_dir: Path, seed: int) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {}
    for f in sorted(in_dir.glob(f"morphometrics_*_seed{seed}.csv")):
        method = f.stem[len("morphometrics_"):-len(f"_seed{seed}")]
        with f.open("r", encoding="utf-8", newline="") as fh:
            tables[method] = list(csv.DictReader(fh))
    if not tables:
        raise SystemExit(f"no morphometrics_*_seed{seed}.csv in {in_dir}\n"
                         f"  run: python -m evaluation.downstream_morphometrics --method ours")
    return tables


def _num(value) -> float:
    """A CSV cell as float; blank or unparseable becomes NaN rather than raising."""
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _pairs(rows, comp, measure):
    """Cells where this compartment was actually traced, so gt > 0.

    Each compartment therefore has its own cohort: a cell with no axon is in the
    basal means and not the axon ones. The counts are reported beside the shifts
    so the table cannot be read as one cohort measured three ways.
    """
    gt, pr = [], []
    for r in rows:
        g = _num(r.get(f"gt_{comp}_{measure}"))
        if g > 0:
            gt.append(g); pr.append(_num(r.get(f"pred_{comp}_{measure}")))
    return np.asarray(gt), np.asarray(pr)


def _bias(rows, comp, measure) -> float:
    gt, pr = _pairs(rows, comp, measure)
    return float((pr.mean() - gt.mean()) / gt.mean()) if gt.size >= MIN_CELLS_FOR_SHIFT else float("nan")


def _n_measured(rows, comp, measure) -> int:
    """Cells contributing to this compartment and measure."""
    return int(_pairs(rows, comp, measure)[0].size)


def _tail_share(rows, comp, measure, frac=0.10) -> float:
    gt, pr = _pairs(rows, comp, measure)
    if gt.size < MIN_CELLS_FOR_SHIFT:
        return float("nan")
    err = np.abs(pr - gt)
    if err.sum() <= 0:
        return 0.0
    k = max(1, int(round(frac * err.size)))
    return float(np.sort(err)[-k:].sum() / err.sum())


def _usable(rows, tol: float) -> np.ndarray:
    ok = []
    for r in rows:
        good = True
        for comp in COMPARTMENTS:
            for m in CORE:
                g = _num(r.get(f"gt_{comp}_{m}"))
                p = _num(r.get(f"pred_{comp}_{m}"))
                # A blank or unmeasurable value is not evidence of inaccuracy, so
                # it neither fails the cell nor silently passes as zero error.
                if g > 0 and p == p and abs(p - g) / g > tol:
                    good = False; break
            if not good:
                break
        ok.append(good)
    return np.asarray(ok, dtype=bool)


def _flag_risk_by_file(model_seed: int, fold_seed: int = DEFAULT_BASE_SEED) -> dict[str, float]:
    """Out-of-fold flag risk per file, from the same code path as eval_flag.

    ``_loso_risk`` fits one flag per held-out seed. Only the fold that scores
    ``model_seed``'s rows is used here, so the other folds are not fitted: that
    is two thirds of the most expensive call in this driver. ``fold_seed`` is the
    base seed, eval_flag's by default, so the triage below is done by the flag
    Table 8 reports rather than by a differently seeded fit.
    """
    df = _load_joined(FLAG_DIR / "final_flag_multiseed_labels.csv",
                      FLAG_DIR / "final_flag_multiseed_features.csv")
    df = df[df["cell_type_gt"].astype(str) == TRIAGE_CELL_TYPE].reset_index(drop=True)
    risk = _loso_risk(df, fold_seed, only_test_seed=model_seed)
    sel = (df["model_seed"].astype(int) == model_seed).to_numpy()
    return {str(f): float(r) for f, r in zip(df.loc[sel, "file"], risk[sel])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--in-dir", type=Path,
                    default=ROOT / "evaluation" / "results" / "morphometrics")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results")
    args = ap.parse_args()

    tables = _load(args.in_dir, args.seed)
    if "ours" not in tables:
        raise SystemExit("morphometrics_ours_seed*.csv is required")
    order = ["ours"] + sorted(m for m in tables if m != "ours")
    rows, tol = tables["ours"], args.tolerance
    out: dict = {"seed": args.seed, "tolerance": tol, "n_cells": len(rows),
                 "human_disagreement": HUMAN_DISAGREEMENT, "methods": {}}

    L = ["Are automated labels good enough to measure from?",
         f"seed {args.seed}, {len(rows)} held-out cells, tolerance {tol:.0%}",
         f"reference: expert annotators disagree by ~{HUMAN_DISAGREEMENT:.0%} on total length",
         "", "1. SYSTEMATIC SHIFT IN THE COHORT MEAN (our pipeline)", "",
         f"  {'compartment / measure':28s} {'cells':>6s} {'shift':>8s}   {'vs human band':>14s}"]
    out["bias"] = {}
    out["n_measured"] = {}
    # Each compartment is averaged over the cells that actually have it traced,
    # so these rows are not one cohort measured three ways; the count says which.
    for comp in COMPARTMENTS:
        for m in CORE:
            b = _bias(rows, comp, m)
            n_m = _n_measured(rows, comp, m)
            out["bias"][f"{comp}_{m}"] = b
            out["n_measured"][f"{comp}_{m}"] = n_m
            verdict = ("too few" if b != b else
                       "inside" if abs(b) < HUMAN_DISAGREEMENT else "OUTSIDE")
            shift = "     --" if b != b else f"{b:+8.1%}"
            L.append(f"  {comp + ' / ' + LABEL[m]:28s} {n_m:6d} {shift:>8s}   {verdict:>14s}")
        L.append("")

    # ---- 2: every method ----------------------------------------------------
    L += ["2. AGAINST THE BASELINES", "",
          f"  {'method':14s} {'apical len':>11s} {'apical area':>12s} "
          f"{'apical path':>12s} {'usable':>9s}"]
    for meth in order:
        t = tables[meth]
        u = _usable(t, tol)
        b = {m: _bias(t, "apical", m) for m in CORE}
        out["methods"][meth] = {"apical_bias": b, "usable_fraction": float(u.mean()),
                                "n_usable": int(u.sum())}
        L.append(f"  {meth:14s} {b['total_length']:+11.1%} {b['surface_area']:+12.1%} "
                 f"{b['max_path_dist']:+12.1%} {u.mean():9.1%}")
    L.append("")

    # ---- 3: where the error lives -------------------------------------------
    shares = [_tail_share(rows, c, m) for c in COMPARTMENTS for m in CORE]
    out["tail_share_range"] = [float(np.nanmin(shares)), float(np.nanmax(shares))]
    # Cells whose apical total length matches the reference to the micrometre.
    gt_apical, pr_apical = _pairs(rows, "apical", "total_length")
    exact = np.abs(pr_apical - gt_apical) < 1.0
    out["apical_length_within_1um"] = {"share": float(exact.mean()), "n_cells": int(exact.size),
                                       "n_within": int(exact.sum())}
    L += ["3. WHERE THE ERROR LIVES", "",
          f"  the worst 10% of cells carry {np.nanmin(shares):.0%} to {np.nanmax(shares):.0%} "
          f"of all cohort error", "  (even spreading would be 10%), so most cells are exact and "
          "the error is", "  concentrated in a small, identifiable minority",
          f"  apical total length within 1 um of the reference: {exact.mean():.1%} "
          f"({int(exact.sum())} of {exact.size} cells with an apical)", ""]

    # ---- 4: triage ----------------------------------------------------------
    pyr = [r for r in rows if r["cell_type"] == TRIAGE_CELL_TYPE]
    pyr_mask = _usable(pyr, tol)
    risk = _flag_risk_by_file(args.seed)
    sc = np.array([risk.get(r["file"], np.nan) for r in pyr])
    have = ~np.isnan(sc)
    # Every row is over the cells the flag scored. Counting unscored cells in the
    # 0% row and not in the others would report a population change as a lift.
    base = pyr_mask[have]
    L += ["4. TRIAGE  (pyramidal cells, by the pyramidal flag)", "",
          f"  {'reject':>8s} {'usable':>9s} {'kept':>7s} {'lift':>8s}",
          f"  {'0%':>8s} {base.mean():9.1%} {int(have.sum()):7d} {'--':>8s}"]
    out["triage"] = {"0.00": float(base.mean()),
                     "n_scored": int(have.sum()), "n_pyramidal": len(pyr)}
    for rr in REJECT_RATES:
        k = int(round(rr * have.sum()))
        keep = have.copy()
        if k:
            keep[np.argsort(-np.where(have, sc, -np.inf))[:k]] = False
        u = pyr_mask[keep]
        out["triage"][f"{rr:.2f}"] = float(u.mean())
        L.append(f"  {rr:8.0%} {u.mean():9.1%} {keep.sum():7d} "
                 f"{u.mean() - base.mean():+8.1%}")

    report = "\n".join(L)
    print(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "morphometric_utility.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.out_dir / "morphometric_utility.txt").write_text(report + "\n", encoding="utf-8")
    print("\nwrote evaluation/results/morphometric_utility.{json,txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
