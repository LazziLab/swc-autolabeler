"""Leave-one-seed-out evaluation of the flag (rejection) model. The Sec. 5.4 numbers.

Trains the flag model on two seeds' scored cells and evaluates on the third,
rotating which seed is held out. Files shared across seeds' test sets are kept
out of the training fold, so no cell is used to both train and test a fold. One
model is fitted per fold, the shared ``flag_regressor``, with no selection among
models or feature sets. Rejection operating points are then reported: fixed
reject rates, and recall and precision targets whose thresholds are set on the
fold's validation cells. The headline is the kept-cell F1 P10 after rejecting the
top-risk cells.

The model regresses the per-cell neurite F1 and ranks by the predicted value;
``F1 < 0.60`` remains the reporting definition of a bad cell, so precision,
recall, AP and AUC keep their published meaning. Regression is what the metric
needs: once the clearly bad cells are rejected, kept-P10 is set by the mediocre
band (F1 roughly 0.6--0.9), and a binary target gives every cell in that band
the same label and so never learns to rank inside it.

One flag is fitted per cell type, pyramidal and interneuron, each on held-out
cells of its own type, and the two are never pooled into one score. Their rows
share the output files, told apart by the ``scope`` column. ``--scopes``
restricts a run to some of the types, and such a run names its files after them.

The feature set is ``ours_oof``: deployment-safe confidence, geometry and
prediction-shape features from the single labeling pass. Ground-truth-derived
columns are blocked by ``GT_OR_LABEL_COLUMNS``.

``--feature-set`` reruns the identical protocol on a variant feature set:
``with_identity`` (adds the source, laboratory and study tokens parsed from the
file name) or ``confidence_only`` (the pipeline's own confidences); or on
``mean_confidence``, a no-model ranking by one minus mean node confidence.
Variants write to ``evaluation/results/flag_ablation/<set>/``.

Usage:
    python -m evaluation.stage4.eval_flag
    python -m evaluation.stage4.eval_flag --feature-set with_identity

Output (evaluation/results/flag/):
    final_flag_leave_one_seed_out.csv / .json
    final_flag_leave_one_seed_out_summary.csv / .txt
    final_flag_before_after.csv
    loso_models/*.joblib
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.stage4.flag_model import (  # noqa: E402
    F1_COL,
    SplitInfo,
    _apply_precision_thresholds,
    _apply_recall_thresholds,
    flag_regressor,
    risk_scores,
    _engineer_features,
    _make_preprocessor,
    _precision_thresholds,
    _rank_thresholds,
    _recall_thresholds,
    _rejection_metrics,
    _safe_ap,
    _safe_auc,
)

from evaluation.paths import display_path                          # noqa: E402

#: Kept for the modules that import ``_display`` from here.
_display = display_path

FLAG_DIR = ROOT / "evaluation" / "results" / "flag"
FEATURE_MODE = "ours_oof"
TARGET_THRESHOLD = 0.6
DEFAULT_RECALL_TARGETS = (0.5, 0.7, 0.8, 0.9)
DEFAULT_REJECT_RATES = (0.05, 0.10, 0.12, 0.15)
DEFAULT_PRECISION_TARGETS = (0.7, 0.8, 0.9)
FEATURE_SETS = ("full", "with_identity", "confidence_only", "mean_confidence")
IDENTITY_FEATURES = ("source", "lab_prefix", "study_prefix")
CONFIDENCE_PREFIXES = ("conf_", "branch_conf_", "branch_frac_conf", "axon_conf_",
                       "basal_conf_", "apical_conf_")
CONFIDENCE_NAMES = ("stage1_conf", "stage1_uncertainty")

#: Base seed of the published run; see ``fold_seeds``.
#: One flag per cell type; a default run fits and scores both.
SCOPES = ("pyramidal", "interneuron")
DEFAULT_BASE_SEED = 2026


@contextmanager
def quiet_flag_warnings():
    """Silence the two warnings the flag fit raises by construction.

    ``_engineer_features`` inserts columns one at a time, which pandas reports as
    a PerformanceWarning, and the imputer reports a column that is missing
    everywhere, which the flag handles by dropping it. Both were filtered at
    import until now, which silenced them for every module that imports this one,
    including for work unrelated to the flag: the second message is exactly what
    would reveal an all-missing feature column somewhere else. Callers outside
    this module wrap their own calls.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        warnings.filterwarnings(
            "ignore", message="Skipping features without any observed values.*")
        yield


def fold_seeds(base_seed: int, test_seed: int) -> tuple[int, int]:
    """The (split, model) seeds of one leave-one-seed-out fold.

    Both were derived at the call site, and the callers disagreed: this driver
    split on ``base_seed`` and fitted on ``base_seed + test_seed``, while
    analyze_failure_modes passed one seed to both. The folds were therefore split
    differently and the models seeded differently, and at a 10% rejection rate
    the two rejected only about two thirds of the same cells. Every caller now
    takes its pair from here, so the same ``base_seed`` means the same fold.
    """
    return int(base_seed), int(base_seed) + int(test_seed)


def _parse_csv_list(raw: str, cast=str) -> list:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _load_joined(labels_path: Path, features_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    required = {"row_id", "model_seed", "file"}
    if not required.issubset(labels.columns) or not required.issubset(features.columns):
        raise SystemExit("labels/features must include row_id, model_seed, file")
    df = labels.merge(features, on=["row_id", "model_seed", "file"], how="left", suffixes=("", "_feature"))
    if len(df) != len(labels):
        raise SystemExit("labels/features merge changed row count")
    df = df.copy()
    if "source" not in df.columns:
        df["source"] = df["file"].astype(str).str.split("__", n=1).str[0]
    return df


def _ours_oof_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the ``ours_oof`` feature frame: coerce cross-seed columns numeric."""
    out = df.copy()
    for col in [c for c in out.columns if c.startswith("xmodel_oof_")]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _group_val_split(df: pd.DataFrame, y: pd.Series, idx: list[int], seed: int) -> tuple[list[int], list[int]]:
    pool = pd.DataFrame({"idx": idx, "file": df.loc[idx, "file"].astype(str), "y": y.loc[idx].astype(int)})
    grouped = pool.groupby("file", as_index=False)["y"].max()
    strat = grouped["y"] if grouped["y"].nunique() == 2 and grouped["y"].value_counts().min() >= 2 else None
    train_files, val_files = train_test_split(
        grouped["file"],
        test_size=0.25,
        random_state=seed,
        stratify=strat,
    )
    train_set = set(train_files.astype(str))
    val_set = set(val_files.astype(str))
    train_idx = [int(i) for i in idx if str(df.loc[i, "file"]) in train_set]
    val_idx = [int(i) for i in idx if str(df.loc[i, "file"]) in val_set]
    return train_idx, val_idx


def _loso_split(df: pd.DataFrame, y: pd.Series, test_seed: int, seed: int) -> tuple[SplitInfo, dict]:
    test_idx = df.index[df["model_seed"].astype(int) == int(test_seed)].tolist()
    test_files = set(df.loc[test_idx, "file"].astype(str))
    pool_idx = df.index[
        (df["model_seed"].astype(int) != int(test_seed))
        & (~df["file"].astype(str).isin(test_files))
    ].tolist()
    train_idx, val_idx = _group_val_split(df, y, pool_idx, seed + int(test_seed))
    leak = {
        "test_seed": int(test_seed),
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
        "train_val_dropped_for_test_file_overlap": int(
            ((df["model_seed"].astype(int) != int(test_seed)) & df["file"].astype(str).isin(test_files)).sum()
        ),
        "train_test_file_overlap": len(set(df.loc[train_idx, "file"].astype(str)) & test_files),
        "val_test_file_overlap": len(set(df.loc[val_idx, "file"].astype(str)) & test_files),
        "train_val_file_overlap": len(set(df.loc[train_idx, "file"].astype(str)) & set(df.loc[val_idx, "file"].astype(str))),
    }
    return SplitInfo(train=train_idx, val=val_idx, test=test_idx), leak


def _fit_flagger(
    df_feat: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    y: pd.Series,
    f1: pd.Series,
    split: SplitInfo,
    seed: int,
) -> dict:
    """Fit the flag model for one fold.

    The training target is the per-cell F1 itself; ``y`` (F1 < 0.60) is only the
    reporting label, so precision, recall, AP and AUC keep their stated meaning.
    """
    pipe = Pipeline(
        [
            ("preprocess", _make_preprocessor(numeric_features, categorical_features)),
            ("model", flag_regressor(seed)),
        ]
    )
    with quiet_flag_warnings():
        pipe.fit(df_feat.loc[split.train], f1.loc[split.train])
        val_score = risk_scores(pipe, df_feat.loc[split.val])
        test_score = risk_scores(pipe, df_feat.loc[split.test])
    val_y = y.loc[split.val].to_numpy(dtype=int)
    test_y = y.loc[split.test].to_numpy(dtype=int)
    return {
        "model_name": "xgb",
        "cat_features": categorical_features,
        "pipe": pipe,
        "val_score": val_score,
        "test_score": test_score,
        "val_ap": _safe_ap(val_y, val_score),
        "val_auc": _safe_auc(val_y, val_score),
        "test_ap": _safe_ap(test_y, test_score),
        "test_auc": _safe_auc(test_y, test_score),
    }


def _select_feature_set(name: str, numeric: list[str], categorical: list[str]) -> tuple[list[str], list[str]]:
    """Restrict the engineered feature lists to one ablation's feature set."""
    if name in ("full", "mean_confidence"):
        return numeric, categorical
    if name == "with_identity":
        return numeric, list(categorical) + [c for c in IDENTITY_FEATURES if c not in categorical]
    if name == "confidence_only":
        return [c for c in numeric
                if c.startswith(CONFIDENCE_PREFIXES) or c in CONFIDENCE_NAMES], []
    raise ValueError(f"unknown feature set {name!r}")


def _mean_confidence_scorer(df_feat: pd.DataFrame, y: pd.Series, split: SplitInfo) -> dict:
    """No-model baseline: rank cells by one minus their mean node confidence."""
    score = (1.0 - pd.to_numeric(df_feat["conf_mean"], errors="coerce").fillna(0.0)).clip(0.0, 1.0)
    val_score = score.loc[split.val].to_numpy(dtype=float)
    test_score = score.loc[split.test].to_numpy(dtype=float)
    val_y = y.loc[split.val].to_numpy(dtype=int)
    test_y = y.loc[split.test].to_numpy(dtype=int)
    return {
        "model_name": "mean_confidence",
        "cat_features": [],
        "pipe": None,
        "val_score": val_score,
        "test_score": test_score,
        "val_ap": _safe_ap(val_y, val_score),
        "val_auc": _safe_auc(val_y, val_score),
        "test_ap": _safe_ap(test_y, test_score),
        "test_auc": _safe_auc(test_y, test_score),
    }


def _metric_row_base(
    scope: str,
    test_seed: int,
    split: SplitInfo,
    f1: pd.Series,
    y: pd.Series,
    cand: dict,
) -> dict:
    test_y = y.loc[split.test].to_numpy(dtype=int)
    test_f1 = f1.loc[split.test].to_numpy(dtype=float)
    return {
        "scope": scope,
        "feature_mode": FEATURE_MODE,
        "test_seed": int(test_seed),
        "model": cand["model_name"],
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "n_bad_test": int(test_y.sum()),
        "bad_rate_test": float(test_y.mean()) if len(test_y) else float("nan"),
        "before_f1_mean": float(test_f1.mean()) if len(test_f1) else float("nan"),
        "before_f1_p10": float(np.percentile(test_f1, 10)) if len(test_f1) else float("nan"),
        "test_ap": _safe_ap(test_y, cand["test_score"]),
        "test_auc": _safe_auc(test_y, cand["test_score"]),
    }


def _append_eval_rows(
    rows: list[dict],
    before_after: list[dict],
    base: dict,
    test_f1: np.ndarray,
    test_score: np.ndarray,
    val_f1: np.ndarray,
    val_score: np.ndarray,
    recall_targets: list[float],
    reject_rates: list[float],
    precision_targets: list[float],
) -> None:
    val_recall_rows = _recall_thresholds(val_f1, val_score, TARGET_THRESHOLD, recall_targets)
    test_recall_rows = _apply_recall_thresholds(test_f1, test_score, TARGET_THRESHOLD, val_recall_rows)
    val_precision_rows = _precision_thresholds(val_f1, val_score, TARGET_THRESHOLD, precision_targets)
    test_precision_rows = _apply_precision_thresholds(test_f1, test_score, TARGET_THRESHOLD, val_precision_rows)
    test_rate_rows = _rejection_metrics(test_f1, test_score, reject_rates, TARGET_THRESHOLD)

    for val, test in zip(val_recall_rows, test_recall_rows):
        row = {
            **base,
            "operating_point": "validation_recall",
            "target_value": float(val["target_recall"]),
            "validation_precision": float(val["validation_precision"]),
            "validation_recall": float(val["validation_recall"]),
            **{k: test.get(k) for k in ("score_threshold", "reject_rate", "n_rejected", "precision_lt_target", "recall_lt_target", "binary_accuracy", "kept_f1_p10", "rejected_f1_mean", "tp", "fp", "fn", "tn")},
        }
        rows.append(row)
        if abs(float(val["target_recall"]) - 0.8) < 1e-9:
            before_after.append(row)

    for val, test in zip(val_precision_rows, test_precision_rows):
        rows.append(
            {
                **base,
                "operating_point": "validation_precision",
                "target_value": float(val["target_precision"]),
                "validation_precision": float(val["validation_precision"]),
                "validation_recall": float(val["validation_recall"]),
                **{k: test.get(k) for k in ("score_threshold", "reject_rate", "n_rejected", "precision_lt_target", "recall_lt_target", "binary_accuracy", "kept_f1_p10", "rejected_f1_mean", "tp", "fp", "fn", "tn")},
            }
        )

    for test in test_rate_rows:
        row = {
            **base,
            "operating_point": "fixed_reject_rate",
            "target_value": float(test["reject_rate"]),
            "validation_precision": float("nan"),
            "validation_recall": float("nan"),
            **{k: test.get(k) for k in ("reject_rate", "n_rejected", "precision_lt_target", "recall_lt_target", "binary_accuracy", "kept_f1_mean", "kept_f1_p10", "kept_f1_p25", "rejected_f1_mean", "tp", "fp", "fn", "tn")},
        }
        rows.append(row)
        if abs(float(test["reject_rate"]) - 0.10) < 1e-9:
            before_after.append(row)


def _summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    keep = rows[rows["operating_point"].isin(["validation_recall", "fixed_reject_rate"])].copy()
    metrics = [
        "precision_lt_target", "recall_lt_target", "binary_accuracy", "reject_rate",
        "n_rejected", "tp", "fp", "fn", "kept_f1_p10", "test_ap", "test_auc",
    ]
    group_cols = ["scope", "feature_mode", "operating_point", "target_value"]
    out_rows: list[dict] = []
    for keys, part in keep.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["n_folds"] = int(len(part))
        for metric in metrics:
            vals = pd.to_numeric(part[metric], errors="coerce")
            rec[f"{metric}_mean"] = float(vals.mean())
            rec[f"{metric}_sd"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _write_txt_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "Flag model leave-one-seed-out summary",
        "=" * 72,
        "Target bad label: per-cell neurite F1 < 0.60",
        "",
    ]
    if summary.empty:
        lines.append("No rows.")
    else:
        view = summary.sort_values(["scope", "feature_mode", "operating_point", "target_value"])
        for _, row in view.iterrows():
            lines.append(
                f"{row['scope']:<11} {row['feature_mode']:<13} "
                f"{row['operating_point']:<19} {float(row['target_value']):>4.2f} | "
                f"P {row['precision_lt_target_mean']:.3f}+/-{row['precision_lt_target_sd']:.3f} "
                f"R {row['recall_lt_target_mean']:.3f}+/-{row['recall_lt_target_sd']:.3f} "
                f"flag {row['n_rejected_mean']:.1f} "
                f"fp {row['fp_mean']:.1f} fn {row['fn_mean']:.1f} "
                f"keptP10 {row['kept_f1_p10_mean']:.4f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=FLAG_DIR / "final_flag_multiseed_labels.csv")
    ap.add_argument("--features", type=Path, default=FLAG_DIR / "final_flag_multiseed_features.csv",
                    help="Output of combine_flag_multiseed. Contains the deployed "
                         "feature set (confidence, predicted geometry, geom_*); "
                         "no separate cross-seed build step is required.")
    ap.add_argument("--out-csv", type=Path, default=FLAG_DIR / "final_flag_leave_one_seed_out.csv")
    ap.add_argument("--out-summary", type=Path, default=FLAG_DIR / "final_flag_leave_one_seed_out_summary.csv")
    ap.add_argument("--out-before-after", type=Path, default=FLAG_DIR / "final_flag_before_after.csv")
    ap.add_argument("--out-json", type=Path, default=FLAG_DIR / "final_flag_leave_one_seed_out.json")
    ap.add_argument("--model-dir", type=Path, default=FLAG_DIR / "loso_models")
    ap.add_argument("--seeds", default="123,42,789")
    ap.add_argument("--scopes", default=",".join(SCOPES),
                    help="cell types to fit and score, one flag each (default: "
                         "both). A run over some of them names its output files "
                         "with those types as a suffix.")
    ap.add_argument("--recall-targets", default="0.5,0.7,0.8,0.9")
    ap.add_argument("--reject-rates", default="0.05,0.10,0.12,0.15",
                    help="Table 8's rates. The published run used all four; a "
                         "shorter list drops rows from that table.")
    ap.add_argument("--precision-targets", default="0.7,0.8,0.9")
    ap.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED,
                    help="base seed; fold_seeds derives each fold's split and "
                         "model seed from it")
    ap.add_argument("--feature-set", choices=FEATURE_SETS, default="full",
                    help="ablation feature set; anything but 'full' writes to "
                         "evaluation/results/flag_ablation/<set>/")
    args = ap.parse_args()

    seeds = _parse_csv_list(args.seeds, int)
    scopes = _parse_csv_list(args.scopes, str)
    recall_targets = _parse_csv_list(args.recall_targets, float)
    reject_rates = _parse_csv_list(args.reject_rates, float)
    precision_targets = _parse_csv_list(args.precision_targets, float)

    global FEATURE_MODE
    if args.feature_set != "full":
        FEATURE_MODE = f"ours_oof-{args.feature_set}"
        adir = ROOT / "evaluation" / "results" / "flag_ablation" / args.feature_set
        args.out_csv = adir / args.out_csv.name
        args.out_summary = adir / args.out_summary.name
        args.out_before_after = adir / args.out_before_after.name
        args.out_json = adir / args.out_json.name
        args.model_dir = adir / "loso_models"
    if scopes != list(SCOPES):
        # A default output name gains the scopes as a suffix; a name given on
        # the command line is kept as given.
        tag = "_" + "-".join(scopes)
        for name in ("out_csv", "out_summary", "out_before_after", "out_json"):
            path = getattr(args, name)
            if path.name == Path(ap.get_default(name)).name:
                setattr(args, name, path.with_name(path.stem + tag + path.suffix))

    df_loaded = _load_joined(args.labels, args.features)
    rows: list[dict] = []
    before_after: list[dict] = []
    leak_checks: list[dict] = []
    saved_models: list[str] = []
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for scope in scopes:
        df_scope = df_loaded.copy()
        if scope != "all":
            if "cell_type_gt" not in df_scope.columns:
                raise SystemExit("scope filtering requires cell_type_gt")
            df_scope = df_scope[df_scope["cell_type_gt"].astype(str) == scope].copy()
        if df_scope.empty:
            continue
        print(f"\n=== scope={scope} feature_mode={FEATURE_MODE} ===", flush=True)
        df_mode = _ours_oof_frame(df_scope)
        with quiet_flag_warnings():
            df_feat, numeric_features, categorical_features = _engineer_features(df_mode)
        numeric_features, categorical_features = _select_feature_set(
            args.feature_set, numeric_features, categorical_features)
        print(f"  features: {len(numeric_features)} numeric, {len(categorical_features)} categorical",
              flush=True)
        f1 = pd.to_numeric(df_feat[F1_COL], errors="coerce")
        valid = f1.notna()
        df_feat = df_feat.loc[valid].reset_index(drop=True)
        f1 = f1.loc[valid].reset_index(drop=True)
        y = (f1 < TARGET_THRESHOLD).astype(int)
        if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
            print(f"  skip: not enough positives/negatives ({int(y.sum())}/{len(y)})", flush=True)
            continue
        for test_seed in seeds:
            if int((df_feat["model_seed"].astype(int) == test_seed).sum()) == 0:
                continue
            split_seed, model_seed = fold_seeds(args.seed, test_seed)
            split, leak = _loso_split(df_feat, y, test_seed, split_seed)
            leak.update({"scope": scope, "feature_mode": FEATURE_MODE,
                         "split_seed": int(split_seed), "model_seed": int(model_seed)})
            leak_checks.append(leak)
            if leak["train_test_file_overlap"] or leak["val_test_file_overlap"] or leak["train_val_file_overlap"]:
                raise SystemExit(f"Flag split leakage detected: {leak}")
            if int(y.loc[split.train].sum()) < 3 or int(y.loc[split.val].sum()) < 1 or int(y.loc[split.test].sum()) < 1:
                print(f"  seed {test_seed}: skip, positives too sparse in train/val/test", flush=True)
                continue
            if args.feature_set == "mean_confidence":
                cand = _mean_confidence_scorer(df_feat, y, split)
            else:
                cand = _fit_flagger(df_feat, numeric_features, categorical_features, y, f1, split,
                                    model_seed)
            val_f1 = f1.loc[split.val].to_numpy(dtype=float)
            test_f1 = f1.loc[split.test].to_numpy(dtype=float)
            test_score = cand["test_score"]
            val_score = cand["val_score"]
            model_path = args.model_dir / f"{scope}_{FEATURE_MODE}_testseed{test_seed}_regf1.joblib"
            # mean_confidence ranks by a column and fits nothing, so there is no
            # model to save; a file holding rank_model None would read as a flag
            # model that silently cannot score.
            if cand["pipe"] is None:
                model_path = None
            else:
                joblib.dump(
                    {
                        "rank_model": cand["pipe"],
                        "target_threshold": TARGET_THRESHOLD,
                        "numeric_features": numeric_features,
                        "categorical_features": cand["cat_features"],
                        "scope": scope,
                        "feature_mode": FEATURE_MODE,
                        "test_seed_held_out": int(test_seed),
                        "base_seed": int(args.seed),
                        "split_seed": int(split_seed),
                        "model_seed": int(model_seed),
                        "validation_rank_thresholds": _rank_thresholds(val_score, reject_rates),
                        "validation_precision_thresholds": _precision_thresholds(val_f1, val_score, TARGET_THRESHOLD, precision_targets),
                        "validation_recall_thresholds": _recall_thresholds(val_f1, val_score, TARGET_THRESHOLD, recall_targets),
                    },
                    model_path,
                )
                saved_models.append(display_path(model_path))
            base = _metric_row_base(scope, test_seed, split, f1, y, cand)
            base["model_path"] = display_path(model_path) if model_path else ""
            _append_eval_rows(
                rows, before_after, base, test_f1, test_score, val_f1, val_score,
                recall_targets, reject_rates, precision_targets,
            )
            print(
                f"  seed {test_seed}: "
                f"test AP={base['test_ap']:.3f} bad={base['n_bad_test']}/{base['n_test']}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    summary = _summary_table(out) if not out.empty else pd.DataFrame()
    before = pd.DataFrame(before_after)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(args.out_summary, index=False, quoting=csv.QUOTE_MINIMAL)
    before.to_csv(args.out_before_after, index=False, quoting=csv.QUOTE_MINIMAL)
    txt_path = args.out_summary.with_suffix(".txt")
    _write_txt_summary(summary, txt_path)
    payload = {
        "labels": _display(args.labels),
        "features": _display(args.features),
        "feature_mode": FEATURE_MODE,
        "target_threshold": TARGET_THRESHOLD,
        "seeds": seeds,
        "scopes": scopes,
        "recall_targets": recall_targets,
        "reject_rates": reject_rates,
        "precision_targets": precision_targets,
        "n_rows_loaded": int(len(df_loaded)),
        "base_seed": int(args.seed),
        "saved_models": saved_models,
        "leak_checks": leak_checks,
        "outputs": {
            "csv": _display(args.out_csv),
            "summary": _display(args.out_summary),
            "summary_txt": _display(txt_path),
            "before_after": _display(args.out_before_after),
        },
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {_display(args.out_csv)}")
    print(f"Wrote {_display(args.out_summary)}")
    print(f"Wrote {_display(txt_path)}")
    print(f"Wrote {_display(args.out_before_after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
