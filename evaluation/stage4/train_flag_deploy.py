"""Train the deployable flag model from the multi-seed table.

The leave-one-seed-out models (``eval_flag``) are for evaluation only. This
trains a production-shaped bundle from the full multi-seed table, using a
grouped file split so duplicate seed rows for the same SWC never cross
train/validation/test. The bundle holds the ranking model and the
validation-derived rejection thresholds.

The model regresses the per-cell neurite F1 and cells are ranked by the
negated prediction (``flag_model.risk_scores``); ``F1 < 0.60`` remains the
reporting definition of a bad cell. One bundle is written per cell type,
pyramidal and interneuron, each fitted on the cells of that reference type.

The feature set is ``ours_oof``: deployment-safe confidence, geometry and
prediction-shape features from the single labeling pass.

Usage:
    python -m evaluation.stage4.train_flag_deploy

Output:
    pipeline/models/flag_model/flag_model_pyramidal_ours_oof_regf1.joblib
    pipeline/models/flag_model/flag_model_interneuron_ours_oof_regf1.joblib
    evaluation/results/flag/final_flag_deploy.json / .csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.paths import display_path as _display                     # noqa: E402
from evaluation.stage4.eval_flag import (                                 # noqa: E402
    DEFAULT_BASE_SEED, DEFAULT_REJECT_RATES, FEATURE_MODE, TARGET_THRESHOLD,
    _load_joined, _ours_oof_frame,
)
from pipeline.stage4.flag_model import (  # noqa: E402
    F1_COL,
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

FLAG_DIR = ROOT / "evaluation" / "results" / "flag"
DEFAULT_TEST_SIZE = 0.20
#: Fraction of the train+validation remainder held out for validation, so the
#: defaults give a 60/20/20 split by file.
DEFAULT_VAL_SIZE = 0.25


def _split_by_file(df: pd.DataFrame, y: pd.Series, seed: int,
                   test_size: float = DEFAULT_TEST_SIZE,
                   val_size: float = DEFAULT_VAL_SIZE) -> tuple[list[int], list[int], list[int]]:
    """Group the rows by file into train, validation and test.

    ``val_size`` is a fraction of what is left after the test split, so the
    defaults give 60/20/20. Both were literals, so the split could be neither
    changed nor recorded in the bundle.
    """
    grouped = pd.DataFrame({"file": df["file"].astype(str), "y": y.astype(int)})
    by_file = grouped.groupby("file", as_index=False)["y"].max()
    strat = by_file["y"] if by_file["y"].nunique() == 2 and by_file["y"].value_counts().min() >= 2 else None
    train_val_files, test_files = train_test_split(
        by_file["file"], test_size=test_size, random_state=seed, stratify=strat,
    )
    train_val = by_file[by_file["file"].isin(set(train_val_files))]
    strat_tv = (
        train_val["y"]
        if train_val["y"].nunique() == 2 and train_val["y"].value_counts().min() >= 2
        else None
    )
    train_files, val_files = train_test_split(
        train_val["file"], test_size=val_size, random_state=seed, stratify=strat_tv,
    )
    train_set = set(train_files.astype(str))
    val_set = set(val_files.astype(str))
    test_set = set(test_files.astype(str))
    return (
        [int(i) for i in df.index if str(df.loc[i, "file"]) in train_set],
        [int(i) for i in df.index if str(df.loc[i, "file"]) in val_set],
        [int(i) for i in df.index if str(df.loc[i, "file"]) in test_set],
    )


# The loader is eval_flag's, not a second copy: this one had drifted, losing the
# key-column check and the source fallback, so a features file missing a key
# merged to all-NaN here and raised there.


def _fit_scope(
    df_loaded: pd.DataFrame,
    *,
    scope: str,
    out_dir: Path,
    seed: int,
    reject_rates: list[float],
    precision_targets: list[float],
    recall_targets: list[float],
    selection_reject_rate: float,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    tag: str = "",
) -> dict:
    df_scope = df_loaded.copy()
    if scope != "all":
        df_scope = df_scope[df_scope["cell_type_gt"].astype(str) == scope].copy()
    df_mode = _ours_oof_frame(df_scope)
    df_feat, numeric_features, categorical_features = _engineer_features(df_mode)
    f1 = pd.to_numeric(df_feat[F1_COL], errors="coerce")
    valid = f1.notna()
    df_feat = df_feat.loc[valid].reset_index(drop=True)
    f1 = f1.loc[valid].reset_index(drop=True)
    y = (f1 < TARGET_THRESHOLD).astype(int)
    if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
        raise SystemExit(f"not enough positives/negatives for scope={scope}: {int(y.sum())}/{len(y)}")

    train_idx, val_idx, test_idx = _split_by_file(df_feat, y, seed, test_size, val_size)
    leak = {
        "train_val_file_overlap": len(set(df_feat.loc[train_idx, "file"]) & set(df_feat.loc[val_idx, "file"])),
        "train_test_file_overlap": len(set(df_feat.loc[train_idx, "file"]) & set(df_feat.loc[test_idx, "file"])),
        "val_test_file_overlap": len(set(df_feat.loc[val_idx, "file"]) & set(df_feat.loc[test_idx, "file"])),
    }
    if any(leak.values()):
        raise SystemExit(f"split leakage detected for {scope}: {leak}")

    pipe = Pipeline(
        [
            ("preprocess", _make_preprocessor(numeric_features, categorical_features)),
            ("model", flag_regressor(seed)),
        ]
    )
    pipe.fit(df_feat.loc[train_idx], f1.loc[train_idx])
    val_score = risk_scores(pipe, df_feat.loc[val_idx])
    best = {
        "model_name": "xgb",
        "cat_features": categorical_features,
        "pipe": pipe,
        "val_score": val_score,
        "val_ap": _safe_ap(y.loc[val_idx].to_numpy(dtype=int), val_score),
        "val_auc": _safe_auc(y.loc[val_idx].to_numpy(dtype=int), val_score),
    }

    val_f1 = f1.loc[val_idx].to_numpy(dtype=float)
    test_f1 = f1.loc[test_idx].to_numpy(dtype=float)
    test_y = y.loc[test_idx].to_numpy(dtype=int)
    test_score = risk_scores(best["pipe"], df_feat.loc[test_idx])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"flag_model_{scope}_{FEATURE_MODE}_regf1{tag}.joblib"
    bundle = {
        # Predicts the per-cell neurite F1; rank by the negated prediction
        # (pipeline.stage4.flag_model.risk_scores does this). No probability
        # calibration: the prediction is already on the F1 scale, and rejection
        # is driven by the validation rank thresholds below.
        "rank_model": best["pipe"],
        "target": "regression:per_cell_neurite_f1",
        "target_threshold": float(TARGET_THRESHOLD),
        "numeric_features": numeric_features,
        "categorical_features": best["cat_features"],
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "selection_reject_rate": float(selection_reject_rate),
        "model": best["model_name"],
        "feature_mode": FEATURE_MODE,
        "cell_type_filter": scope,
        "validation_rank_thresholds": _rank_thresholds(best["val_score"], reject_rates),
        "validation_precision_thresholds": _precision_thresholds(val_f1, best["val_score"], TARGET_THRESHOLD, precision_targets),
        "validation_recall_thresholds": _recall_thresholds(val_f1, best["val_score"], TARGET_THRESHOLD, recall_targets),
    }
    joblib.dump(bundle, out_path)
    return {
        "scope": scope,
        "feature_mode": FEATURE_MODE,
        "model_path": _display(out_path),
        "n_rows": int(len(df_feat)),
        "n_bad": int(y.sum()),
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "leak": leak,
        "model": best["model_name"],
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(best["cat_features"]),
        "test_ap_rank": _safe_ap(test_y, test_score),
        "test_auc_rank": _safe_auc(test_y, test_score),
        "test_fixed_reject": _rejection_metrics(test_f1, test_score, reject_rates, TARGET_THRESHOLD),
    }


def _parse_list(raw: str, cast=str) -> list:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=FLAG_DIR / "final_flag_multiseed_labels.csv")
    ap.add_argument("--features", type=Path, default=FLAG_DIR / "final_flag_multiseed_features.csv")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "pipeline" / "models" / "flag_model")
    ap.add_argument("--out-json", type=Path, default=FLAG_DIR / "final_flag_deploy.json")
    ap.add_argument("--scopes", default="pyramidal,interneuron")
    ap.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    ap.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE,
                    help="fraction of the remainder held out for validation")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the released flag model. Without it a run "
                         "whose settings differ from the released ones writes to "
                         "its own file names instead.")
    ap.add_argument("--reject-rates",
                    default=",".join(f"{r:g}" for r in DEFAULT_REJECT_RATES),
                    help="rates whose thresholds go into the released bundle; the "
                         "default matches eval_flag, so the deployed model carries "
                         "a threshold for every rate Table 8 reports")
    ap.add_argument("--precision-targets", default="0.7,0.8,0.9")
    ap.add_argument("--recall-targets", default="0.5,0.7,0.8,0.9")
    ap.add_argument("--selection-reject-rate", type=float, default=0.05)
    args = ap.parse_args()

    # The released model and JSON keep their names only for a run made the way
    # the released one was. Anything else takes a tagged name, so a trial run
    # cannot replace the model this project ships.
    tag = ""
    if args.seed != DEFAULT_BASE_SEED:
        tag += f"_seed{args.seed}"
    if args.test_size != DEFAULT_TEST_SIZE:
        tag += f"_test{args.test_size:g}"
    if args.val_size != DEFAULT_VAL_SIZE:
        tag += f"_val{args.val_size:g}"
    if _parse_list(args.reject_rates, float) != list(DEFAULT_REJECT_RATES):
        tag += "_rates" + "-".join(f"{r:g}" for r in _parse_list(args.reject_rates, float))
    if tag and args.force:
        print(f"  --force: writing over the released names despite {tag}", flush=True)
        tag = ""
    if tag:
        print(f"  this run differs from the released one; writing with tag {tag}", flush=True)
        args.out_json = args.out_json.with_name(args.out_json.stem + tag + args.out_json.suffix)

    df = _load_joined(args.labels, args.features)
    rows = []
    for scope in _parse_list(args.scopes):
        print(f"Training deploy flagger: scope={scope} feature_mode={FEATURE_MODE}", flush=True)
        row = _fit_scope(
            df,
            scope=scope,
            out_dir=args.model_dir,
            seed=args.seed,
            reject_rates=_parse_list(args.reject_rates, float),
            precision_targets=_parse_list(args.precision_targets, float),
            recall_targets=_parse_list(args.recall_targets, float),
            selection_reject_rate=args.selection_reject_rate,
            test_size=args.test_size,
            val_size=args.val_size,
            tag=tag,
        )
        rows.append(row)
        print(
            f"  saved {row['model_path']}  AP={row['test_ap_rank']:.3f}  "
            f"features={row['n_numeric_features']}+{row['n_categorical_features']}",
            flush=True,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "labels": _display(args.labels),
                "features": _display(args.features),
                "feature_mode": FEATURE_MODE,
                "target_threshold": TARGET_THRESHOLD,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Fixed field names, so the header does not depend on which keys the first
    # row happens to carry, and an empty --scopes writes a header rather than
    # raising on rows[0].
    csv_fields = ["scope", "feature_mode", "model_path", "n_rows", "n_bad", "model",
                  "n_numeric_features", "n_categorical_features",
                  "test_ap_rank", "test_auc_rank"]
    with (args.out_json.with_suffix(".csv")).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {_display(args.out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
