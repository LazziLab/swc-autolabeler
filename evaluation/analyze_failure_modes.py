"""Failure-mode breakdown of the labeler's worst cells (the paper's Table 9).

Answers two questions about the cells the pipeline labels worst:

  *What* goes wrong -- each bottom-decile cell is matched against a taxonomy of
  failure modes (apical missed, axon/dendrite confusion, Stage 1 propagation,
  ...). A mode's **enrichment** is its share of the bottom decile divided by its
  share of all cells. The bottom decile is 10% of the corpus, so enrichment
  ceilings at 10x; a value near the ceiling marks a mode almost exclusive to the
  worst cells, one near 1.0 marks a property just as common among good cells and
  therefore not diagnostic on its own.

  Whether the flag *catches* it -- each mode's flag recall is the fraction of its
  bottom-decile cells the flag rejects at its operating point, and its false
  flags are the cells it rejects that were not in the bottom decile.

The rules read ground-truth columns (per-class F1, GT class fractions, Stage 1
correctness). That is sound here and only here: this is a post-hoc description of
what went wrong, not a feature. The flag scores it is judged against are fit
leave-one-seed-out by ``evaluation.stage4.eval_flag``'s own fold machinery, under
the same base seed, so no cell contributes to the model that scores it and the
flag described here is the one Table 8 reports.

Scope is pyramidal, matching the deployed flag.

Usage:
    python -m evaluation.analyze_failure_modes

Input (build with evaluation.stage4.combine_flag_multiseed):
    evaluation/results/flag/final_flag_multiseed_{labels,features}.csv

Output (evaluation/results/):
    failure_modes.json
    failure_modes.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.stage4.flag_model import F1_COL, _engineer_features    # noqa: E402
from evaluation.paths import display_path                          # noqa: E402
from evaluation.stage4.eval_flag import (                            # noqa: E402
    DEFAULT_BASE_SEED,
    FLAG_DIR,
    _fit_flagger,
    _load_joined,
    _loso_split,
    _ours_oof_frame,
    fold_seeds,
    quiet_flag_warnings,
)

DEFAULT_REJECT_RATE = 0.10
#: The failure-mode analysis covers pyramidal cells, the cells of Table 9.
SCOPE = "pyramidal"
BOTTOM_DECILE = 0.10


def _num(row: pd.Series, col: str) -> float:
    """Read a column as float; missing/blank/unparseable becomes NaN."""
    val = row.get(col, np.nan)
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _int(row: pd.Series, col: str) -> int:
    val = _num(row, col)
    return 0 if math.isnan(val) else int(val)


def _flag_false(row: pd.Series, col: str) -> bool:
    """True when a boolean-ish column is explicitly false (CSV round-trips as text).

    score_flag_dataset writes stage1_correct with ``str(bool)``, so it arrives as
    "True" or "False"; pandas may also hand back a real bool or a 0/1. All three
    are recognised. A blank or missing value is not false, so a gap in the table
    never matches a rule by default.
    """
    val = row.get(col, "")
    if isinstance(val, (bool, np.bool_)):
        return not bool(val)
    return str(val).strip().lower() in ("false", "0", "0.0")


# (label, rule) -- a cell may match several modes, so the columns do not sum to
# the corpus total.
CATEGORIES: list[tuple[str, str, object]] = [
    ("Whole-cell mislabel",
     "Stage 1 confidently disagrees with the GT cell type",
     lambda r: _flag_false(r, "stage1_correct") and _num(r, "stage1_conf") > 0.85),
    ("Apical missed",
     "GT has apical nodes, the model predicted none",
     lambda r: _num(r, "apical_frac") > 0.01 and _int(r, "pred_apical") == 0),
    ("Apical under-predicted",
     "under half the GT apical node count, with low apical F1",
     lambda r: (_num(r, "apical_frac") > 0.01
                and _int(r, "pred_apical") < 0.5 * _num(r, "apical_frac") * max(1, _int(r, "n_nodes"))
                and _num(r, "apical_F1") < 0.5)),
    ("Apical over-predicted",
     "far more apical nodes than GT, with low apical F1",
     lambda r: (_int(r, "pred_apical") > 1.5 * max(1.0, _num(r, "apical_frac") * max(1, _int(r, "n_nodes")))
                and _num(r, "apical_F1") < 0.5
                and _int(r, "pred_apical") > 100)),
    ("Apical not opposed to basal",
     "apical and basal leave the soma in the same direction",
     lambda r: _num(r, "apical_basal_cos") > 0),
    ("Axon/dendrite confusion",
     "axon F1 below 0.95",
     lambda r: _num(r, "axon_F1") < 0.95),
    ("Basal lost",
     "basal F1 below 0.5 despite GT basal nodes",
     lambda r: _num(r, "basal_F1") < 0.5 and _num(r, "basal_frac") > 0.01),
    ("Disconnected",
     "reconstruction has more than one connected component",
     lambda r: _int(r, "n_components") > 1),
    ("Truncated apical",
     "GT apical reaches under 100 um from the soma",
     lambda r: 0.0 < _num(r, "apical_reach_um") < 100.0 and _num(r, "apical_frac") > 0.01),
    ("Stage 1 propagation",
     "Stage 1 is wrong and the cell's F1 is low",
     lambda r: _flag_false(r, "stage1_correct") and _num(r, "held_out_F1") < 0.7),
]


def _category_mask(df: pd.DataFrame, rule) -> np.ndarray:
    """Apply one rule row-wise. NaN comparisons are False, so gaps never match."""
    return df.apply(lambda r: bool(rule(r)), axis=1).to_numpy(dtype=bool)


def loso_risk(df: pd.DataFrame, seed: int = DEFAULT_BASE_SEED,
              only_test_seed: int | None = None) -> np.ndarray:
    """Out-of-fold risk score for every row, via eval_flag's own fold machinery.

    ``seed`` is the base seed, and each fold's split and model seeds come from
    ``eval_flag.fold_seeds``, so at the default this reproduces the folds
    eval_flag reports rather than a differently seeded fit of the same protocol.

    ``only_test_seed`` fits just the fold that scores that seed's rows, leaving
    the rest NaN. Callers that need one seed's scores pay for one flag instead of
    three; callers that need every row leave it None.
    """
    with quiet_flag_warnings():
        df_feat, numeric, categorical = _engineer_features(_ours_oof_frame(df))
    f1 = pd.to_numeric(df_feat[F1_COL], errors="coerce")
    y = (f1 < 0.6).astype(int)
    score = np.full(len(df_feat), np.nan)
    wanted = sorted(df["model_seed"].astype(int).unique())
    if only_test_seed is not None:
        wanted = [s for s in wanted if int(s) == int(only_test_seed)]
    for test_seed in wanted:
        split_seed, model_seed = fold_seeds(seed, test_seed)
        split, leak = _loso_split(df_feat, y, test_seed, split_seed)
        if leak["train_test_file_overlap"] or leak["val_test_file_overlap"]:
            raise SystemExit(f"file overlap in fold {test_seed}: {leak}")
        fold = _fit_flagger(df_feat, numeric, categorical, y, f1, split, model_seed)
        score[np.asarray(split.test)] = fold["test_score"]
    if only_test_seed is None and np.isnan(score).any():
        raise SystemExit("some rows were never scored out-of-fold")
    return score


#: Kept for the existing import; ``loso_risk`` is the name to use.
_loso_risk = loso_risk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=FLAG_DIR / "final_flag_multiseed_labels.csv")
    ap.add_argument("--features", type=Path, default=FLAG_DIR / "final_flag_multiseed_features.csv")
    ap.add_argument("--reject-rate", type=float, default=DEFAULT_REJECT_RATE,
                    help="flag operating point (default 0.10, the deployed rate)")
    ap.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED,
                    help="base seed for the folds; the default is eval_flag's, so "
                         "the flag judged here is the one Table 8 reports")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results")
    args = ap.parse_args()

    df = _load_joined(args.labels, args.features)
    df = df[df["cell_type_gt"].astype(str) == SCOPE].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no {SCOPE} rows in {display_path(args.labels)}")

    f1 = pd.to_numeric(df[F1_COL], errors="coerce")
    # "Bad" here is the bottom decile of per-cell F1, so enrichment ceilings at 10x.
    cut = float(np.percentile(f1.dropna(), BOTTOM_DECILE * 100))
    bad = (f1 <= cut).to_numpy(dtype=bool)

    # Flag at its operating point, within each fold, mirroring deployment.
    risk = _loso_risk(df, args.seed)
    flagged = np.zeros(len(df), dtype=bool)
    for ms in sorted(df["model_seed"].astype(int).unique()):
        rows = np.flatnonzero(df["model_seed"].astype(int).to_numpy() == ms)
        # Budget and tie-breaking as flag_model._rejection_metrics: at least one
        # rejected, stable sort. With the default base seed the folds are
        # eval_flag's, so these are the cells its operating point rejects.
        k = max(1, int(round(args.reject_rate * rows.size)))
        flagged[rows[np.argsort(-risk[rows], kind="stable")[:k]]] = True

    n_rows, n_bad = len(df), int(bad.sum())
    # Each rule is a Python call per row, so every mask is computed once and
    # reused for both the table and the unmatched count below.
    masks = {name: _category_mask(df, rule) for name, _d, rule in CATEGORIES}
    table = []
    for name, description, rule in CATEGORIES:
        m = masks[name]
        cat_bad = int((m & bad).sum())
        if not m.any():
            continue
        share_bad = cat_bad / n_bad if n_bad else 0.0
        share_all = int(m.sum()) / n_rows
        table.append({
            "category": name,
            "description": description,
            "rows": int(m.sum()),
            "bad_rows": cat_bad,
            "unique_bad_files": int(df.loc[m & bad, "file"].astype(str).nunique()),
            "enrichment": round(share_bad / share_all, 3) if share_all else 0.0,
            "flag_recall": round(float((m & bad & flagged).sum() / cat_bad), 3) if cat_bad else 0.0,
            "false_flags": int((m & flagged & ~bad).sum()),
        })
    table.sort(key=lambda r: (-r["enrichment"], -r["bad_rows"]))

    matched = np.zeros(len(df), dtype=bool)
    for m in masks.values():
        matched |= m
    unmatched = int((bad & ~matched).sum())

    result = {
        "scope": SCOPE,
        "n_rows": n_rows,
        "n_bad_rows": n_bad,
        "bottom_decile_f1_cutoff": round(cut, 4),
        "reject_rate": args.reject_rate,
        "enrichment_ceiling": round(1.0 / BOTTOM_DECILE, 1),
        "unmatched_bad_rows": unmatched,
        "categories": table,
    }

    w = max(len(r["category"]) for r in table)
    lines = [
        f"Failure-mode breakdown ({SCOPE}, {n_rows} cell-seed rows)",
        f"bad = bottom decile of per-cell neurite F1 (F1 <= {cut:.4f}): {n_bad} rows",
        f"flag operating point: reject {args.reject_rate:.0%}, leave-one-seed-out scores",
        f"enrichment ceiling {1.0 / BOTTOM_DECILE:.0f}x; a cell may match several modes",
        "",
        f"  {'category':{w}s} {'bad':>5s} {'enrich':>8s} {'recall':>7s} {'false':>6s}",
    ]
    for r in table:
        lines.append(f"  {r['category']:{w}s} {r['bad_rows']:5d} "
                     f"{r['enrichment']:7.2f}x {r['flag_recall']:7.3f} {r['false_flags']:6d}")
    lines += ["", f"{unmatched} of {n_bad} bad rows matched no category."]
    report = "\n".join(lines)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "failure_modes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / "failure_modes.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nwrote {display_path(args.out_dir)}/failure_modes.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
