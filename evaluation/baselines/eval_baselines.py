"""Classical morphometric baselines on the corpus, in the pipeline's metric pack.

Trains and evaluates the four learned baselines (NeuroM-RF, L-Measure-RF,
Sholl-RF, Sholl-MLP) together with the two training-free floors (majority, which
labels every neurite node basal, and rules_only, the anatomical priors with no
learning) on one seed's hash-split train/test partition, and emits the SAME
report shape as the pipeline evaluation, so the numbers drop straight into the
"our method vs prior methods" comparison. ``--method all`` runs all six.

``--source-prefix`` narrows the *scoring* to one corpus source while training is
left alone, giving the single-source slice of the same cross-source model.

Each seed trains its own models into a per-seed cache and is scored only on that
seed's test cells, so a model is never evaluated on cells it was trained on.
Run each seed, then compare against the pipeline's per-seed evaluation.

Usage:
    python -m evaluation.baselines.eval_baselines --seed 123
    python -m evaluation.baselines.eval_baselines --seed 123 --method lmeasure_rf

Output (evaluation/results/baselines/, created if missing):
    baselines_seed{seed}.json    full report dicts (one per method)
    baselines_seed{seed}.csv     per-cell rows (with method column)
    baselines_seed{seed}_table.txt
    seed{seed}/{method}.pkl       cached fitted models (per seed)

``--out-dir`` moves the reports; the model cache follows it unless ``--cache-dir``
says otherwise, so a run redirected to a scratch directory refits every baseline
rather than reading the released cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from evaluation.baselines.baselines import predict_with_cache            # noqa: E402
from evaluation.baselines.baseline_report import (                       # noqa: E402
    _aggregate_per_file,
    build_full_report,
    print_per_cell_type,
    print_per_class_per_cell,
    print_summary_table,
)
from evaluation.paths import display_path as _display                    # noqa: E402
from evaluation.stage4._flag_metrics import per_cell_neurite_f1
from pipeline.split import in_test  # noqa: E402

CELL_TYPES = ("pyramidal", "interneuron")
DATA = ROOT / "data"
OUT_DIR = ROOT / "evaluation" / "results" / "baselines"
METHODS = ("neurom_rf", "lmeasure_rf", "sholl_rf", "sholl_mlp", "majority", "rules_only")


def _split(seed: int, test_size: float) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    train: dict[str, list[Path]] = {ct: [] for ct in CELL_TYPES}
    test: dict[str, list[Path]] = {ct: [] for ct in CELL_TYPES}
    for ct in CELL_TYPES:
        for p in sorted((DATA / ct).glob("*.swc")):
            (test if in_test(p.name, seed, test_size) else train)[ct].append(p)
    return train, test




def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--method", default="all", choices=("all", *METHODS))
    ap.add_argument("--force-retrain", action="store_true",
                    help="Ignore the on-disk model cache and re-train.")
    ap.add_argument("--source-prefix", default="",
                    help="score only test cells whose filename starts with this "
                         "prefix, e.g. 'hpf_ca1__' for the single-source lab corpus. "
                         "Training is unaffected: the model still fits this seed's "
                         "full cross-source train split, so the slice stays "
                         "comparable to the pipeline's lab-subset run.")
    ap.add_argument("--train-source-prefix", default="",
                    help="train on only this source, the baseline analogue of a "
                         "single-source retrain. Fitted models go to their own cache "
                         "so they can never be confused with the cross-source ones.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="where the fitted baselines live (default: under "
                         "--out-dir). Point it at the released cache to redirect "
                         "the reports without refitting six models.")
    args = ap.parse_args()
    tag = f"_{args.source_prefix.rstrip('_')}" if args.source_prefix else ""
    train_tag = f"_{args.train_source_prefix.rstrip('_')}_retrain" if args.train_source_prefix else ""
    tag += train_tag
    # A run that scores one method, or uses a different split, writes its own
    # files: the all-method results feed evaluation.paired_stats and must not be
    # replaced by a subset.
    if args.method != "all":
        tag += f"_{args.method}"
    if args.test_size != 0.20:
        tag += f"_test{args.test_size:g}"

    train, test = _split(args.seed, args.test_size)
    if args.source_prefix:
        test = {ct: [p for p in v if p.name.startswith(args.source_prefix)]
                for ct, v in test.items()}
        if not any(test.values()):
            raise SystemExit(f"no test cells matched --source-prefix {args.source_prefix!r}")
    if args.train_source_prefix:
        train = {ct: [p for p in v if p.name.startswith(args.train_source_prefix)]
                 for ct, v in train.items()}
        if not any(train.values()):
            raise SystemExit(f"no train cells matched "
                             f"--train-source-prefix {args.train_source_prefix!r}")
    n_tr = sum(len(v) for v in train.values())
    n_te = sum(len(v) for v in test.values())
    print(f"seed {args.seed}: train {n_tr} ({len(train['pyramidal'])} pyr + {len(train['interneuron'])} int), "
          f"test {n_te} ({len(test['pyramidal'])} pyr + {len(test['interneuron'])} int)")

    # The cache key carries everything that changes the fit: seed, split fraction,
    # train filter and method. A lab-only model, a cross-source model and a model
    # fitted under a different --test-size are different fits and must never share
    # a cache entry. predict_with_cache additionally checks the stored key against
    # the training files this run assembled, so a stale entry is refused, not used.
    split_tag = "" if args.test_size == 0.20 else f"_test{args.test_size:g}"
    cache_root = args.cache_dir or args.out_dir
    cache_dir = cache_root / f"seed{args.seed}{train_tag}{split_tag}"
    if cache_root != OUT_DIR and args.cache_dir is None:
        print(f"  note: fitting into {_display(cache_dir)}; pass --cache-dir "
              f"{_display(OUT_DIR)} to reuse the released models", flush=True)
    methods = list(METHODS) if args.method == "all" else [args.method]
    reports: list[dict] = []
    all_rows: list[dict] = []

    for method in methods:
        print(f"\n=== {method} ===")
        # sklearn's own deprecations are raised while the baselines fit and
        # predict, and are silenced only there: filtering at import hid them for
        # the whole run, including for code that has nothing to do with fitting.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            t_fit = time.perf_counter()
            predict_fn = predict_with_cache(
                method, train, seed=args.seed,
                cache_path=cache_dir / f"{method}.pkl", force_retrain=args.force_retrain,
            )
            fit_min = (time.perf_counter() - t_fit) / 60.0
            print(f"  predicting on {n_te} test cells...")
            t_pred = time.perf_counter()
            cell_records, all_gt, all_pred = _aggregate_per_file(test, predict_fn)
            predict_min = (time.perf_counter() - t_pred) / 60.0
        report = build_full_report(method, all_gt, all_pred, cell_records)
        # Fit and predict are timed apart: a cache hit costs almost no fit time,
        # so one combined number is not comparable between runs and is not a
        # prediction cost. wall_clock_min stays as their sum for older readers.
        report["fit_min"] = fit_min
        report["predict_min"] = predict_min
        report["fit_from_cache"] = fit_min < 0.01
        report["wall_clock_min"] = fit_min + predict_min
        report["n_train_cells"] = n_tr
        reports.append(report)

        for r in cell_records:
            n_correct = sum(1 for g, p in zip(r["gt"], r["pred"]) if g == p)
            acc = n_correct / r["n_nodes"] if r["n_nodes"] else 0.0
            all_rows.append({
                "method": method,
                "file": Path(r["path"]).name,
                "cell_type": r["cell_type"],
                "n_nodes": r["n_nodes"],
                "accuracy": f"{acc:.4f}",
                "neurite_macro_f1": f"{per_cell_neurite_f1(r['gt'], r['pred'], r['cell_type']):.4f}",
            })

        print(f"  acc={report['corpus']['accuracy']:.4f}  "
              f"neurite_F1={report['corpus']['neurite_macro_f1']:.4f}  "
              f"pc_F1_mean={report['per_cell']['neurite_macro_f1']['mean']:.4f}  "
              f"pc_F1_p10={report['per_cell']['neurite_macro_f1']['p10']:.4f}  "
              f"(fit {report['fit_min']:.1f} + predict {report['predict_min']:.1f} min)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"baselines_seed{args.seed}{tag}.json"
    out_csv = args.out_dir / f"baselines_seed{args.seed}{tag}.csv"
    out_table = args.out_dir / f"baselines_seed{args.seed}{tag}_table.txt"

    out_json.write_text(json.dumps({
        "seed": args.seed,
        "test_size": args.test_size,
        "method": args.method,
        "source_prefix": args.source_prefix,
        "train_source_prefix": args.train_source_prefix,
        "n_train": n_tr,
        "n_test": n_te,
        "split_source": "hash-bucket split over data/{pyramidal,interneuron}/*.swc",
        "convention": ("per-cell macro F1 averages only the neurite classes "
                       "present in that cell's ground truth; the soma is "
                       "excluded from every reported metric"),
        "reports": reports,
    }, indent=2), encoding="utf-8")

    # Always written, even empty: paired_stats reads this file, and a run that
    # produced no rows must not leave a previous run's CSV beside a fresh JSON.
    csv_fields = ["method", "file", "cell_type", "n_nodes", "accuracy",
                  "neurite_macro_f1"]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(all_rows)

    print("\n" + "=" * 120)
    print(f"  BASELINES on {args.source_prefix.rstrip('_') or 'corpus'}  "
          f"(seed={args.seed} hash split; macro F1 over GT-present classes, "
          f"soma excluded)")
    print("=" * 120)
    print_summary_table(reports)
    print("\n" + "=" * 120 + "\n  PER-CLASS PER-CELL F1 DISTRIBUTIONS\n" + "=" * 120)
    print_per_class_per_cell(reports)
    print("\n" + "=" * 120 + "\n  PER-CELL-TYPE BREAKDOWN\n" + "=" * 120)
    print_per_cell_type(reports)

    out_table.write_text(
        f"Baseline summary (macro F1 over GT-present classes, soma excluded; "
        f"seed={args.seed} hash split"
        f"{f', source-prefix {args.source_prefix}' if args.source_prefix else ''})\n"
        # One rounding for the whole table: the report dicts carry a mix of
        # rounded and full-precision floats, which read as different accuracies.
        + json.dumps({r["method"]: {
            "accuracy": round(r["corpus"]["accuracy"], 4),
            "neurite_macro_f1": round(r["corpus"]["neurite_macro_f1"], 4),
            "apical_f1": round(r["corpus"]["per_class"]["apical"]["f1"], 4),
            "per_cell_F1_mean": round(r["per_cell"]["neurite_macro_f1"]["mean"], 4),
            "per_cell_F1_p10": round(r["per_cell"]["neurite_macro_f1"]["p10"], 4),
            "per_cell_F1_p25": round(r["per_cell"]["neurite_macro_f1"]["p25"], 4),
        } for r in reports}, indent=2),
        encoding="utf-8",
    )
    print()
    for path in (out_json, out_csv, out_table):
        print(f"Wrote {_display(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
