"""Score the single-seed flag: the one ``train_pipeline`` fits into a model dir.

``eval_flag`` measures the released multi-seed flag, which learns from three
seeds' holdouts. A user who runs ``train_pipeline`` gets a flag fit on that one
seed's 20% holdout, over the numeric columns one labeling pass assembles
(``pipeline.stage4.flag_features.assemble_flag_features``): a subset of the
multi-seed table's numeric columns, without its count and Stage 1 summary
columns and without Stage 1's predicted type. This driver measures that flag.

Protocol, per seed independently:

  * take that seed's holdout rows of one cell type, for each type in turn
  * keep exactly the columns that type's shipped flag reads, its
    ``feature_cols`` in ``pipeline/models/seed<S>/flag_model.joblib``
  * grouped 5-fold CV over files, so no cell is scored by a model that trained
    on any row of the same file. Each seed's holdout holds one row per file in
    the released table, so the grouping is defensive: it keeps a table with
    several rows per file leak-free
  * pool the out-of-fold risk scores, reject the top 10%, and report the 10th
    percentile of the kept cells' per-cell F1, the recall of bad cells
    (per-cell F1 below 0.60) and the average precision

Same model and target as ``eval_flag``; the fold scheme and the columns differ,
which is the difference between the two flags.

Usage:
    python -m evaluation.stage4.eval_single_seed_flag

Input (build with evaluation.stage4.combine_flag_multiseed):
    evaluation/results/flag/final_flag_multiseed_{labels,features}.csv

Output (evaluation/results/flag/):
    single_seed_flag.json / .txt     both cell types' flags
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.stage4.flag_model import (                       # noqa: E402
    F1_COL, _engineer_features, _make_preprocessor, flag_regressor, risk_scores,
)
from pipeline.stage4.single_seed_flag import BAD_THRESHOLD, flag_for   # noqa: E402
from sklearn.metrics import average_precision_score            # noqa: E402
from evaluation.paths import display_path as _display          # noqa: E402
from evaluation.stage4.eval_flag import (                      # noqa: E402
    FLAG_DIR, _load_joined, _ours_oof_frame,
)

DEFAULT_SEEDS = "42,123,789"
DEFAULT_FOLDS = 5
DEFAULT_REJECT_RATE = 0.10
#: The flag regressor's own seed. Not a data split: which cells are held out is
#: decided by --seeds and by the grouped folds.
DEFAULT_MODEL_SEED = 0
#: One flag per cell type; a default run scores both.
SCOPES = ("pyramidal", "interneuron")


def _kept_p10(f1: np.ndarray, risk: np.ndarray, reject_rate: float) -> float:
    """10th percentile of per-cell F1 after dropping the riskiest cells.

    Budget and tie-breaking follow ``flag_model._rejection_metrics``: at least
    one cell rejected, and a stable sort. The figure this driver publishes is
    quoted directly against eval_flag's, so the two have to spend the same
    budget and resolve ties the same way.
    """
    k = max(1, int(round(reject_rate * f1.size)))
    return float(np.percentile(f1[np.argsort(-risk, kind="stable")[k:]], 10))


def _recall(f1: np.ndarray, risk: np.ndarray, reject_rate: float) -> float:
    """Share of bad cells among the riskiest ones rejected, same budget as above."""
    k = max(1, int(round(reject_rate * f1.size)))
    bad = f1 < BAD_THRESHOLD
    rejected = np.zeros(f1.size, dtype=bool)
    rejected[np.argsort(-risk, kind="stable")[:k]] = True
    return float((rejected & bad).sum() / bad.sum()) if bad.any() else float("nan")


def _score_type(full: pd.DataFrame, scope: str, seeds: list[int], args) -> tuple[dict, list[str]]:
    """One cell type's single-seed flags, each seed scored on its own holdout."""
    full = full[full["cell_type_gt"].astype(str) == scope].copy()
    if full.empty:
        raise SystemExit(f"no {scope} rows in {_display(args.labels)}")
    per_seed = []
    last_features: list[str] = []
    for model_seed in seeds:
        df = full[full["model_seed"].astype(int) == model_seed].reset_index(drop=True)
        if df.empty:
            continue
        df_feat, numeric, categorical = _engineer_features(_ours_oof_frame(df))
        # Score the columns the seed's shipped flag reads, and nothing else.
        bundle = args.model_root / f"seed{model_seed}" / "flag_model.joblib"
        if not bundle.is_file():
            raise SystemExit(f"MISSING {_display(bundle)}: train the seed first")
        sub = flag_for(joblib.load(bundle), scope)
        if sub is None:
            raise SystemExit(f"{_display(bundle)} holds no {scope} flag")
        shipped = list(sub["feature_cols"])
        absent = [c for c in shipped if c not in df_feat.columns]
        if absent:
            raise SystemExit(f"seed {model_seed}: flag columns absent from the table: {absent[:5]}")
        numeric, categorical = shipped, []
        f1 = pd.to_numeric(df_feat[F1_COL], errors="coerce")
        valid = f1.notna()
        df_feat = df_feat.loc[valid].reset_index(drop=True)
        f1 = f1.loc[valid].reset_index(drop=True)

        # One row per file on this table, so the grouping is defensive rather
        # than load-bearing; n_files beside n_cells in the record says which.
        groups = df_feat["file"].astype(str).to_numpy()
        last_features = list(numeric) + list(categorical)
        oof = np.full(len(df_feat), np.nan)
        for train_idx, test_idx in GroupKFold(n_splits=args.folds).split(df_feat, f1, groups):
            pipe = Pipeline([("preprocess", _make_preprocessor(numeric, categorical)),
                             ("model", flag_regressor(args.model_seed))])
            pipe.fit(df_feat.iloc[train_idx], f1.iloc[train_idx])
            oof[test_idx] = risk_scores(pipe, df_feat.iloc[test_idx])
        if np.isnan(oof).any():
            raise SystemExit(f"seed {model_seed}: some rows never scored out-of-fold")

        v = f1.to_numpy(dtype=float)
        per_seed.append({
            "model_seed": model_seed,
            "n_cells": int(v.size),
            "n_files": int(pd.Series(groups).nunique()),
            "n_features": len(numeric) + len(categorical),
            "no_flag_p10": round(float(np.percentile(v, 10)), 4),
            "kept_p10": round(_kept_p10(v, oof, args.reject_rate), 4),
            "n_bad": int((v < BAD_THRESHOLD).sum()),
            "recall": round(_recall(v, oof, args.reject_rate), 4),
            "ap": round(float(average_precision_score(v < BAD_THRESHOLD, oof)), 4),
        })
    if not per_seed:
        raise SystemExit("no seeds matched")
    res = {"per_seed": per_seed}
    for key, name in (("no_flag_p10", "mean_no_flag_p10"), ("kept_p10", "mean_kept_p10"),
                      ("recall", "mean_recall"), ("ap", "mean_ap")):
        res[name] = round(float(np.mean([r[key] for r in per_seed])), 4)
    return res, last_features


def _report_lines(scope: str, res: dict, n_cols: int, args) -> list[str]:
    lines = [f"Single-seed {scope} flag (grouped {args.folds}-fold CV by file, "
             f"the {n_cols} columns the shipped flag reads)",
             f"kept-P10, recall of bad cells and AP after rejecting the riskiest "
             f"{args.reject_rate:.0%}", "",
             f"  {'seed':>6s} {'cells':>7s} {'files':>7s} {'no-flag P10':>12s} {'kept-P10':>10s} "
             f"{'recall':>8s} {'AP':>7s}"]
    for r in res["per_seed"]:
        lines.append(f"  {r['model_seed']:6d} {r['n_cells']:7d} {r['n_files']:7d} "
                     f"{r['no_flag_p10']:12.4f} {r['kept_p10']:10.4f} {r['recall']:8.4f} {r['ap']:7.4f}")
    lines += ["", f"  {'mean':>6s} {'':7s} {'':7s} {res['mean_no_flag_p10']:12.4f} "
                  f"{res['mean_kept_p10']:10.4f} {res['mean_recall']:8.4f} {res['mean_ap']:7.4f}"]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=FLAG_DIR / "final_flag_multiseed_labels.csv")
    ap.add_argument("--features", type=Path, default=FLAG_DIR / "final_flag_multiseed_features.csv")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS,
                    help="the data seeds whose holdouts are scored, one flag each")
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--reject-rate", type=float, default=DEFAULT_REJECT_RATE)
    ap.add_argument("--model-seed", dest="model_seed", type=int,
                    default=DEFAULT_MODEL_SEED,
                    help="seed of the flag regressor itself, not a data split. "
                         "eval_flag's leave-one-seed-out models are fitted under its "
                         "own base seed, so the two flags compared here are seeded "
                         "differently; the protocols differ anyway, 5-fold CV here "
                         "against leave-one-seed-out there.")
    ap.add_argument("--model-root", type=Path, default=ROOT / "pipeline" / "models",
                    help="holds seed<S>/flag_model.joblib, whose columns are scored")
    ap.add_argument("--out-dir", type=Path, default=FLAG_DIR)
    ap.add_argument("--scopes", default=",".join(SCOPES),
                    help="the cell types whose flags are scored (default: both)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    scopes = [x.strip() for x in args.scopes.split(",") if x.strip()]
    bad_scopes = [x for x in scopes if x not in SCOPES]
    if bad_scopes:
        raise SystemExit(f"unknown cell types {bad_scopes}; choose from {SCOPES}")
    loaded = _load_joined(args.labels, args.features)
    by_type: dict[str, dict] = {}
    features_read: list[str] = []
    lines: list[str] = []
    for scope in scopes:
        res, cols = _score_type(loaded, scope, seeds, args)
        if features_read and cols != features_read:
            raise SystemExit("the cell types' shipped flags read different columns")
        features_read = cols
        by_type[scope] = res
        lines += _report_lines(scope, res, len(cols), args) + [""]
    result = {"protocol": f"grouped {args.folds}-fold CV by file, per seed",
              "cross_seed_features": False, "reject_rate": args.reject_rate,
              "folds": args.folds, "model_seed": args.model_seed, "seeds": seeds,
              "feature_source": "feature_cols of pipeline/models/seed<S>/flag_model.joblib",
              "features": features_read, "by_cell_type": by_type}
    report = "\n".join(lines).rstrip("\n")
    print(report)

    # A run that is not the default one never takes its file name.
    tag = "" if scopes == list(SCOPES) else "_" + "-".join(scopes)
    if args.folds != DEFAULT_FOLDS:
        tag += f"_folds{args.folds}"
    if args.reject_rate != DEFAULT_REJECT_RATE:
        tag += f"_rr{args.reject_rate:g}"
    if args.model_seed != DEFAULT_MODEL_SEED:
        tag += f"_m{args.model_seed}"
    if args.seeds != DEFAULT_SEEDS:
        tag += "_seeds" + "-".join(str(x) for x in seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"single_seed_flag{tag}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stem.with_suffix(".txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nwrote {_display(stem)}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
