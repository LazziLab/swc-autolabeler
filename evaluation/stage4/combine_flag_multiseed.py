"""Combine per-seed flag labels/features into one multi-seed table.

Each seed's scored test cells become rows, tagged with ``model_seed`` and a
``row_id`` of ``seed{seed}:{file}`` so the same SWC scored under different seeds
stays distinct. A leakage precheck confirms every scored file really is in that
seed's hash-split test set (so no seed model saw its own scored cells during
training) and reports how often the seeds' test sets overlap.

Usage:
    python -m evaluation.stage4.combine_flag_multiseed --seeds 123,42,789

Output (evaluation/results/flag/):
    final_flag_multiseed_labels.csv
    final_flag_multiseed_features.csv
    final_flag_multiseed_leakage_check.json / .txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.split import in_test  # noqa: E402

FLAG_DIR = ROOT / "evaluation" / "results" / "flag"


def _read_seed(seed: int, labels_pattern: str, features_pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_path = Path(labels_pattern.format(seed=seed))
    features_path = Path(features_pattern.format(seed=seed))
    if not labels_path.is_file():
        raise SystemExit(f"MISSING: {labels_path}")
    if not features_path.is_file():
        raise SystemExit(f"MISSING: {features_path}")
    labels = pd.read_csv(labels_path).copy()
    features = pd.read_csv(features_path).copy()
    if "file" not in labels.columns or "file" not in features.columns:
        raise SystemExit(f"{labels_path} or {features_path} is missing 'file'")
    labels["model_seed"] = seed
    features["model_seed"] = seed
    labels["row_id"] = [f"seed{seed}:{name}" for name in labels["file"].astype(str)]
    features["row_id"] = [f"seed{seed}:{name}" for name in features["file"].astype(str)]
    return labels, features


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="123,42,789")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--labels-pattern",
                    default=str(FLAG_DIR / "final_flag_seed{seed}_labels.csv"))
    ap.add_argument("--features-pattern",
                    default=str(FLAG_DIR / "final_flag_seed{seed}_features.csv"))
    ap.add_argument("--out-labels", type=Path, default=FLAG_DIR / "final_flag_multiseed_labels.csv")
    ap.add_argument("--out-features", type=Path, default=FLAG_DIR / "final_flag_multiseed_features.csv")
    ap.add_argument("--out-check", type=Path, default=FLAG_DIR / "final_flag_multiseed_leakage_check.json")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    label_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    seed_checks: dict[str, dict] = {}
    feature_cols: dict[int, set[str]] = {}
    mtimes: list[float] = []

    for seed in seeds:
        for pat in (args.labels_pattern, args.features_pattern):
            p = Path(pat.format(seed=seed))
            if p.is_file():
                mtimes.append(p.stat().st_mtime)
        labels, features = _read_seed(seed, args.labels_pattern, args.features_pattern)
        feature_cols[seed] = set(features.columns)
        files = labels["file"].astype(str).tolist()
        not_in_test = sorted(f for f in files if not in_test(f, seed, args.test_size))
        duplicate_rows = int(labels["row_id"].duplicated().sum() + features["row_id"].duplicated().sum())
        missing_features = sorted(set(labels["row_id"]) - set(features["row_id"]))
        extra_features = sorted(set(features["row_id"]) - set(labels["row_id"]))
        seed_checks[str(seed)] = {
            "n_labels": int(len(labels)),
            "n_features": int(len(features)),
            "n_unique_files": int(labels["file"].nunique()),
            "files_not_in_model_test": len(not_in_test),
            "duplicate_row_ids": duplicate_rows,
            "missing_feature_rows": len(missing_features),
            "extra_feature_rows": len(extra_features),
            "examples": {
                "files_not_in_model_test": not_in_test[:5],
                "missing_feature_rows": missing_features[:5],
                "extra_feature_rows": extra_features[:5],
            },
        }
        label_parts.append(labels)
        feature_parts.append(features)

    # A seed scored by an older pipeline brings a different feature set, and a
    # concat would fill the gap with NaN rather than complain. The flag would
    # then train on columns that exist for only part of the corpus.
    common = set.intersection(*feature_cols.values()) if feature_cols else set()
    for seed, cols in feature_cols.items():
        if cols != common:
            raise SystemExit(
                f"seed {seed}'s feature table has a different column set from the "
                f"others: {sorted(cols - common)[:6]} extra, "
                f"{sorted(common - cols)[:6]} missing. Rescore every seed with the "
                f"same pipeline before combining.")
    if mtimes and (max(mtimes) - min(mtimes)) / 3600.0 > 24:
        print(f"warning: the per-seed tables span "
              f"{(max(mtimes) - min(mtimes)) / 3600.0:.0f} hours; they may not all "
              f"come from the same pipeline.", flush=True)

    labels_all = pd.concat(label_parts, ignore_index=True)
    features_all = pd.concat(feature_parts, ignore_index=True)
    if labels_all["row_id"].duplicated().any():
        raise SystemExit("Duplicate row_id in combined labels")
    if features_all["row_id"].duplicated().any():
        raise SystemExit("Duplicate row_id in combined features")

    first = ["row_id", "model_seed", "file"]
    labels_all = labels_all[first + [c for c in labels_all.columns if c not in first]]
    features_all = features_all[first + [c for c in features_all.columns if c not in first]]

    files_by_seed = {
        str(seed): set(labels_all.loc[labels_all["model_seed"] == seed, "file"].astype(str))
        for seed in seeds
    }
    pair_overlaps = {
        f"{a}_x_{b}": len(files_by_seed[str(a)] & files_by_seed[str(b)])
        for i, a in enumerate(seeds)
        for b in seeds[i + 1:]
    }
    triple = len(set.intersection(*(files_by_seed[str(seed)] for seed in seeds))) if seeds else 0
    # ok covers per-seed integrity only: every scored file really in that seed's
    # test split, no duplicate row, labels and features in step. It deliberately
    # ignores the overlap counts below, because the seeds' test sets are meant to
    # overlap and each row is still scored by a model that never saw that cell.
    ok = all(
        check["files_not_in_model_test"] == 0
        and check["duplicate_row_ids"] == 0
        and check["missing_feature_rows"] == 0
        and check["extra_feature_rows"] == 0
        for check in seed_checks.values()
    )
    summary = {
        "ok": ok,
        "seeds": seeds,
        "n_rows": int(len(labels_all)),
        "n_unique_files": int(labels_all["file"].nunique()),
        "duplicate_file_rows_across_seeds": int(len(labels_all) - labels_all["file"].nunique()),
        "pair_test_file_overlaps": pair_overlaps,
        "triple_test_file_overlap": triple,
        "by_seed": seed_checks,
    }
    # The precheck record is written either way, so a failure leaves evidence.
    # The tables themselves are not: this module exists to refuse a leaked table,
    # and writing one before deciding would leave it for the next driver to read.
    args.out_check.parent.mkdir(parents=True, exist_ok=True)
    args.out_check.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.out_check.with_suffix(".txt").write_text(
        "\n".join([
            "Multi-seed flag combine precheck",
            f"ok: {summary['ok']}",
            f"rows: {summary['n_rows']}",
            f"unique files: {summary['n_unique_files']}",
            f"duplicate file rows across seeds: {summary['duplicate_file_rows_across_seeds']}",
            f"pair overlaps: {summary['pair_test_file_overlaps']}",
            f"triple overlap: {summary['triple_test_file_overlap']}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit(
            f"Leakage precheck failed; no table written. See {args.out_check}")

    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    labels_all.to_csv(args.out_labels, index=False)
    features_all.to_csv(args.out_features, index=False)
    print(f"wrote {args.out_labels.name} and {args.out_features.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
