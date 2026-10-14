"""Learned per-cell flag (rejection) model: shared library.

The flag model is a meta-model over the auto-labeler's pipeline outputs. It
predicts a cell's per-cell neurite F1 and ranks cells by that prediction, so
the highest-risk cells can be set aside for manual review. It is not part of
the per-cell labeling path. It is a quality-control layer applied to a batch of
already-labeled cells.

This module holds the reusable pieces every Stage 4 driver shares: feature
engineering over deployment-safe columns, the sklearn preprocessor, the single
flag model (``flag_regressor``), and the rejection and threshold metrics. One
model is defined here and used everywhere -- the in-pipeline single-seed flag,
the leave-one-seed-out evaluation, and the deployed bundle -- so no driver can
train a different model from the published one.

Features are built from deployment-safe columns only: predicted class counts,
Stage 1 prediction and confidence, per-cell confidence statistics and predicted
geometry. Every quantity is listed once. Where the labels table and the
confidence run both carry a column, only one copy is taken, so the feature set
is a property of this module rather than of whichever table it is handed.
Ground-truth columns such as per-class F1 are used only to construct labels,
never as model inputs; ``_engineer_features`` asserts that none of them leak
into the feature matrix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

F1_COL = "held_out_F1"

# Columns that may exist in the source CSV but must never enter X.
GT_OR_LABEL_COLUMNS = {
    "cell_type_gt",
    "held_out_F1",
    "acc",
    "axon_F1",
    "basal_F1",
    "apical_F1",
    "stage1_correct",
    "axon_frac",
    "apical_frac",
    "basal_frac",
    "apical_mean_z",
    "apical_above_soma",
    "apical_z_extent",
    "apical_reach_um",
    "apical_basal_cos",
    "elapsed_s",
}

# Node count, Stage 1's confidence and the predicted class counts are taken
# from the labels table alone. Older feature tables also carry them under
# ``_conf``/``_conf_run`` names; those copies are deliberately not listed, so a
# table that still has them contributes no duplicate column.
CONFIDENCE_NUMERIC_FEATURES = [
    "confidence_features_missing",
    "conf_mean",
    "conf_std",
    "conf_min",
    "conf_p05",
    "conf_p10",
    "conf_p25",
    "conf_median",
    "conf_frac_lt_05",
    "conf_frac_lt_06",
    "conf_frac_lt_07",
    "conf_frac_lt_08",
    "conf_frac_lt_09",
    "axon_conf_mean",
    "axon_conf_p10",
    "axon_conf_frac_lt_07",
    "basal_conf_mean",
    "basal_conf_p10",
    "basal_conf_frac_lt_07",
    "apical_conf_mean",
    "apical_conf_p10",
    "apical_conf_frac_lt_07",
    "n_pred_branches",
    "branch_conf_mean",
    "branch_conf_p10",
    "branch_frac_conf_lt_07",
    "branch_frac_conf_lt_08",
    "branch_n_nodes_p90",
    "pred_soma_z",
    "pred_axon_z_extent",
    "pred_axon_z_mean_rel_soma",
    "pred_axon_z_std",
    "pred_axon_x_extent",
    "pred_axon_y_extent",
    "pred_basal_z_extent",
    "pred_basal_z_mean_rel_soma",
    "pred_basal_z_std",
    "pred_basal_x_extent",
    "pred_basal_y_extent",
    "pred_apical_z_extent",
    "pred_apical_z_mean_rel_soma",
    "pred_apical_z_std",
    "pred_apical_x_extent",
    "pred_apical_y_extent",
]
# Stage 1's predicted type is taken from the labels table. Older feature tables
# repeat it as ``stage1_pred_conf_run``, identical on every row, so no
# categorical column is taken from here.
CONFIDENCE_CATEGORICAL_FEATURES: list[str] = []
# Optional numeric blocks, picked up by prefix when present in the frame:
#   geom_     geometry/anatomy of the PREDICTED labels (flag_geom_features).
#             The one optional block that earns its cost: +0.012 kept-P10, and
#             it is free -- computed from the same pipeline run.
#   xmodel_   cross-seed disagreement. Needs three trained seeds and adds
#             +0.002, so a single-seed user cannot compute it and loses little.
#   baseline_ L-Measure second opinion. Adds 0.0000 once geom_ is present, and
#             is by far the most expensive block to produce.
EXTRA_NUMERIC_PREFIXES = ("geom_", "xmodel_", "baseline_")


@dataclass
class SplitInfo:
    train: list[int]
    val: list[int]
    test: list[int]


def _lab_tokens(filename: str) -> tuple[str, str]:
    rest = filename.split("__", 1)[1] if "__" in filename else filename
    stem = Path(rest).stem
    toks = [t.lower() for t in re.split(r"[-_\s]+", stem) if t]
    if not toks:
        return "", ""
    lab_prefix = toks[0]
    study_prefix = "_".join(toks[:2]) if len(toks) >= 2 else toks[0]
    return lab_prefix, study_prefix


def _engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()

    numeric_base = [
        "n_nodes",
        "stage1_conf",
        "n_components",
        "pred_axon",
        "pred_basal",
        "pred_apical",
    ]
    for col in numeric_base:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "source" not in df.columns:
        df["source"] = df["file"].astype(str).str.split("__", n=1).str[0]
    if "stage1_pred" not in df.columns:
        df["stage1_pred"] = ""

    lab_parts = df["file"].astype(str).map(_lab_tokens)
    df["lab_prefix"] = [x[0] for x in lab_parts]
    df["study_prefix"] = [x[1] for x in lab_parts]

    non_soma = np.maximum(df["n_nodes"].fillna(0).to_numpy(dtype=float) - 1.0, 1.0)
    for cls in ("axon", "basal", "apical"):
        count_col = f"pred_{cls}"
        frac_col = f"pred_{cls}_frac"
        log_col = f"log_pred_{cls}"
        df[frac_col] = df[count_col].fillna(0).to_numpy(dtype=float) / non_soma
        df[log_col] = np.log1p(df[count_col].fillna(0).to_numpy(dtype=float))

    frac_cols = ["pred_axon_frac", "pred_basal_frac", "pred_apical_frac"]
    fracs = df[frac_cols].fillna(0.0).clip(lower=0.0)
    df["pred_entropy"] = (-(fracs * np.log(fracs.replace(0.0, np.nan)))).sum(axis=1).fillna(0.0)
    df["pred_class_count"] = (
        df[["pred_axon", "pred_basal", "pred_apical"]].fillna(0.0) > 0.0
    ).sum(axis=1)
    df["max_pred_frac"] = fracs.max(axis=1)
    df["pred_apical_zero"] = (df["pred_apical"].fillna(0.0) == 0.0).astype(int)
    df["pred_axon_zero"] = (df["pred_axon"].fillna(0.0) == 0.0).astype(int)
    df["single_class_pred"] = (df["pred_class_count"] == 1).astype(int)
    df["log_n_nodes"] = np.log1p(df["n_nodes"].fillna(0.0))
    df["stage1_uncertainty"] = 1.0 - df["stage1_conf"]
    df["stage1_is_pyramidal"] = (df["stage1_pred"] == "pyramidal").astype(int)
    df["pyr_no_apical"] = (
        (df["stage1_pred"] == "pyramidal") & (df["pred_apical"].fillna(0.0) == 0.0)
    ).astype(int)
    df["int_has_apical"] = (
        (df["stage1_pred"] == "interneuron") & (df["pred_apical"].fillna(0.0) > 0.0)
    ).astype(int)
    numeric_features = [
        "log_n_nodes",
        "n_nodes",
        "stage1_conf",
        "stage1_uncertainty",
        "n_components",
        "pred_axon",
        "pred_basal",
        "pred_apical",
        "log_pred_axon",
        "log_pred_basal",
        "log_pred_apical",
        "pred_axon_frac",
        "pred_basal_frac",
        "pred_apical_frac",
        "pred_entropy",
        "pred_class_count",
        "max_pred_frac",
        "pred_apical_zero",
        "pred_axon_zero",
        "single_class_pred",
        "stage1_is_pyramidal",
        "pyr_no_apical",
        "int_has_apical",
    ]
    # The file-name identity tokens (source, lab_prefix, study_prefix) are
    # computed above as metadata but are not features: they would let the flag
    # learn which laboratories label badly, a shortcut that cannot transfer.
    categorical_features = [
        "stage1_pred",
    ]

    for col in CONFIDENCE_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            numeric_features.append(col)
    for col in CONFIDENCE_CATEGORICAL_FEATURES:
        if col in df.columns:
            categorical_features.append(col)
    for col in sorted(c for c in df.columns if c.startswith(EXTRA_NUMERIC_PREFIXES)):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        numeric_features.append(col)

    numeric_features = [
        c for c in numeric_features if c in df.columns and not df[c].isna().all()
    ]

    forbidden = GT_OR_LABEL_COLUMNS & (set(numeric_features) | set(categorical_features))
    if forbidden:
        raise AssertionError(f"GT/label leakage in feature list: {sorted(forbidden)}")
    return df, numeric_features, categorical_features


def _make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )


def flag_regressor(seed: int = 0):
    """The flag model: gradient-boosted regression onto per-cell neurite F1.

    One model is used everywhere -- the in-pipeline single-seed flag, the
    leave-one-seed-out evaluation, and the deployed multi-seed bundle -- so a
    flag trained on your own data has the same form as the published one.
    """
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                        random_state=seed)


def risk_scores(pipe, X) -> np.ndarray:
    """Suspicion score in [0, 1] (higher = riskier), for classifier or regressor.

    Classifiers give P(bad) directly; regressors predict the per-cell F1, whose
    risk is ``1 - F1`` clipped to the unit interval. That is monotone in ``-F1``,
    so ranking and every rank-derived metric are unaffected by the rescaling,
    and both kinds of flag report suspicion on the same scale.
    """
    model = pipe[-1] if hasattr(pipe, "__getitem__") else pipe
    if hasattr(model, "predict_proba"):
        return np.asarray(pipe.predict_proba(X))[:, 1]
    return np.clip(1.0 - np.asarray(pipe.predict(X), dtype=float), 0.0, 1.0)


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def _rejection_metrics(
    f1: np.ndarray,
    score: np.ndarray,
    reject_rates: Iterable[float],
    target_threshold: float,
) -> list[dict]:
    out: list[dict] = []
    n = len(f1)
    order = np.argsort(-score)
    n_bad_target = int((f1 < target_threshold).sum())
    n_bad_05 = int((f1 < 0.5).sum())
    n_bad_06 = int((f1 < 0.6).sum())

    for rate in reject_rates:
        k = max(1, int(round(rate * n)))
        rejected_idx = order[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[rejected_idx] = False
        rejected_f1 = f1[rejected_idx]
        kept_f1 = f1[keep_mask]
        tp = int((rejected_f1 < target_threshold).sum())
        fp = int(k - tp)
        fn = int(n_bad_target - tp)
        tn = int(n - k - fn)
        out.append(
            {
                "reject_rate": float(rate),
                "n_rejected": int(k),
                "precision_lt_target": float((rejected_f1 < target_threshold).mean()),
                "precision_lt_0_5": float((rejected_f1 < 0.5).mean()),
                "precision_lt_0_6": float((rejected_f1 < 0.6).mean()),
                "recall_lt_target": float((rejected_f1 < target_threshold).sum() / max(1, n_bad_target)),
                "recall_lt_0_5": float((rejected_f1 < 0.5).sum() / max(1, n_bad_05)),
                "recall_lt_0_6": float((rejected_f1 < 0.6).sum() / max(1, n_bad_06)),
                "kept_f1_mean": float(kept_f1.mean()) if kept_f1.size else float("nan"),
                "kept_f1_p10": float(np.percentile(kept_f1, 10)) if kept_f1.size else float("nan"),
                "kept_f1_p25": float(np.percentile(kept_f1, 25)) if kept_f1.size else float("nan"),
                "rejected_f1_mean": float(rejected_f1.mean()),
                "rejected_f1_median": float(np.median(rejected_f1)),
                "binary_accuracy": float((tp + tn) / max(1, n)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return out


def _threshold_rejection_metrics(
    f1: np.ndarray,
    score: np.ndarray,
    score_threshold: float,
    target_threshold: float,
) -> dict:
    rejected_mask = score >= score_threshold
    rejected_f1 = f1[rejected_mask]
    kept_f1 = f1[~rejected_mask]
    n = len(f1)
    n_bad = int((f1 < target_threshold).sum())
    tp = int((rejected_f1 < target_threshold).sum()) if rejected_f1.size else 0
    fp = int(rejected_f1.size - tp)
    fn = int(n_bad - tp)
    tn = int(n - rejected_f1.size - fn)
    return {
        "score_threshold": float(score_threshold),
        "reject_rate": float(rejected_f1.size / max(1, n)),
        "n_rejected": int(rejected_f1.size),
        "precision_lt_target": float(tp / max(1, rejected_f1.size)),
        "recall_lt_target": float(tp / max(1, n_bad)),
        "kept_f1_p10": float(np.percentile(kept_f1, 10)) if kept_f1.size else float("nan"),
        "rejected_f1_mean": float(rejected_f1.mean()) if rejected_f1.size else float("nan"),
        "binary_accuracy": float((tp + tn) / max(1, n)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _precision_thresholds(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    precision_targets: Iterable[float],
) -> list[dict]:
    order = np.argsort(-score)
    sorted_score = score[order]
    sorted_bad = (f1[order] < target_threshold).astype(int)
    cum_bad = np.cumsum(sorted_bad)
    out: list[dict] = []
    n = len(f1)
    n_bad = int((f1 < target_threshold).sum())
    for target_precision in precision_targets:
        best: dict | None = None
        for i in range(n):
            k = i + 1
            if i + 1 < n and sorted_score[i + 1] == sorted_score[i]:
                continue
            precision = float(cum_bad[i] / k)
            if precision + 1e-12 < target_precision:
                continue
            threshold = float(sorted_score[i])
            metrics = _threshold_rejection_metrics(
                f1,
                score,
                threshold,
                target_threshold,
            )
            metrics.update(
                {
                    "target_precision": float(target_precision),
                    "validation_precision": precision,
                    "validation_recall": float(cum_bad[i] / max(1, n_bad)),
                }
            )
            best = metrics
        if best is not None:
            out.append(best)
        else:
            out.append(
                {
                    "target_precision": float(target_precision),
                    "score_threshold": float("inf"),
                    "reject_rate": 0.0,
                    "n_rejected": 0,
                    "precision_lt_target": 0.0,
                    "recall_lt_target": 0.0,
                    "kept_f1_p10": float(np.percentile(f1, 10)) if n else float("nan"),
                    "rejected_f1_mean": float("nan"),
                    "binary_accuracy": float((f1 >= target_threshold).mean()) if n else float("nan"),
                    "tp": 0,
                    "fp": 0,
                    "fn": n_bad,
                    "tn": n - n_bad,
                    "validation_precision": 0.0,
                    "validation_recall": 0.0,
                }
            )
    return out


def _apply_precision_thresholds(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    threshold_rows: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for row in threshold_rows:
        score_threshold = float(row["score_threshold"])
        if np.isinf(score_threshold):
            metrics = _threshold_rejection_metrics(f1, score, float("inf"), target_threshold)
        else:
            metrics = _threshold_rejection_metrics(f1, score, score_threshold, target_threshold)
        metrics["target_precision"] = float(row["target_precision"])
        metrics["validation_precision"] = float(row["validation_precision"])
        metrics["validation_recall"] = float(row["validation_recall"])
        out.append(metrics)
    return out


def _recall_thresholds(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    recall_targets: Iterable[float],
) -> list[dict]:
    bad_scores = np.sort(score[f1 < target_threshold])[::-1]
    n_bad = int(bad_scores.size)
    out: list[dict] = []
    for target_recall in recall_targets:
        if n_bad == 0:
            metrics = _threshold_rejection_metrics(
                f1, score, float("inf"), target_threshold
            )
            metrics.update(
                {
                    "target_recall": float(target_recall),
                    "validation_precision": 0.0,
                    "validation_recall": 0.0,
                }
            )
            out.append(metrics)
            continue
        k = max(1, int(np.ceil(float(target_recall) * n_bad)))
        k = min(k, n_bad)
        threshold = float(bad_scores[k - 1])
        metrics = _threshold_rejection_metrics(
            f1, score, threshold, target_threshold
        )
        metrics.update(
            {
                "target_recall": float(target_recall),
                "validation_precision": metrics["precision_lt_target"],
                "validation_recall": metrics["recall_lt_target"],
            }
        )
        out.append(metrics)
    return out


def _apply_recall_thresholds(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    threshold_rows: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for row in threshold_rows:
        metrics = _threshold_rejection_metrics(
            f1,
            score,
            float(row["score_threshold"]),
            target_threshold,
        )
        metrics["target_recall"] = float(row["target_recall"])
        metrics["validation_precision"] = float(row["validation_precision"])
        metrics["validation_recall"] = float(row["validation_recall"])
        out.append(metrics)
    return out


def _rank_thresholds(score: np.ndarray, reject_rates: Iterable[float]) -> list[dict]:
    n = len(score)
    sorted_scores = np.sort(score)[::-1]
    out: list[dict] = []
    for rate in reject_rates:
        k = max(1, int(round(rate * n)))
        threshold = float(sorted_scores[k - 1])
        out.append(
            {
                "reject_rate": float(rate),
                "n_flagged_on_validation": int((score >= threshold).sum()),
                "rank_score_threshold": threshold,
            }
        )
    return out
