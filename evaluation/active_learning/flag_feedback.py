"""Curator feedback on the flag: does reviewing flagged cells improve the flag?

Mimics the deployment loop in which the labeler is the fixed factory checkpoint
and only the flag learns:

    the flag ranks a batch by suspicion -> the curator opens the top-K and
    records whether each was actually mislabeled -> those verdicts become flag
    training examples -> the flag is retrained.

Ground truth stands in for the curator: revealing a cell's true per-cell F1 is
that cell being reviewed. The labeler is not retrained, which is what a real
deployment does -- a handful of corrections is not enough to retrain the
labeler, but it is plenty to sharpen the flag.

Because the cells are labeled by the full pipeline, the numbers here are on the
same scale as ``evaluation.stage4.eval_flag``: same features, same model, same
kept-P10 at a 10% rejection rate.

Review is counted in FILES, not cells. One SWC contributes a row per model seed,
and a file is reviewed or held out whole, so no reviewed cell is ever scored; the
table is about 5,000 rows over about 4,100 files.

Three arms share one starting flag and one never-reviewed test set:

    flag      review the K files the current flag ranks most suspicious,
              a file ranked by its riskiest cell
    random    review K files at random
    factory   never retrain; the starting flag is used throughout

It runs on the pyramidal flag. The interneuron flag's kept-cell P10 is already
1.00 at this rejection rate on every held-out seed, so there it has no room to
improve.

Usage:
    python -m evaluation.active_learning.flag_feedback

Output (evaluation/results/active_learning/):
    flag_feedback.{json,csv,txt}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score                        # noqa: E402
from sklearn.pipeline import Pipeline                                      # noqa: E402

from evaluation.paths import display_path                                   # noqa: E402
from evaluation.stage4.eval_flag import _load_joined, _ours_oof_frame, TARGET_THRESHOLD  # noqa: E402
from pipeline.stage4.flag_model import (                                   # noqa: E402
    F1_COL,
    _engineer_features,
    _make_preprocessor,
    flag_regressor,
    risk_scores,
)

FLAG_DIR = ROOT / "evaluation" / "results" / "flag"
RESULTS = ROOT / "evaluation" / "results" / "active_learning"
SCOPE = "pyramidal"
DEFAULT_REJECT_RATE = 0.10
DEFAULT_SEED = 2026
DEFAULT_ROUNDS = 8
#: Review is counted in FILES, not cells: one SWC contributes a row per model
#: seed, and a file is reviewed or held out as a whole so no reviewed cell is
#: ever scored. The corpus is about 5,000 rows over about 4,100 files.
DEFAULT_K = 150
DEFAULT_SEED_FILES = 200
DEFAULT_TEST_FRAC = 0.25


def _fit(df, num, cat, f1, idx, seed):
    pipe = Pipeline([("preprocess", _make_preprocessor(num, cat)),
                     ("model", flag_regressor(seed))])
    pipe.fit(df.loc[idx], f1.loc[idx])
    return pipe


def _measure(pipe, df, f1, test_idx, reject_rate: float = DEFAULT_REJECT_RATE) -> dict:
    """Flag quality on the never-reviewed test cells, exactly as eval_flag scores it.

    Budget and tie-breaking follow ``flag_model._rejection_metrics``, and
    ``reject_rate`` is the operating point: it was a module constant, so the one
    quantity this driver is read against could not be swept.
    """
    risk = risk_scores(pipe, df.loc[test_idx])
    tf1 = f1.loc[test_idx].to_numpy(dtype=float)
    bad = tf1 < TARGET_THRESHOLD
    n = len(tf1)
    k = max(1, int(round(reject_rate * n)))
    # Stable, as flag_model._rejection_metrics sorts: an unstable sort breaks ties
    # by whatever order the sort happens to produce, so two runs could reject
    # different cells at the same risk and this would not be eval_flag's number.
    worst = np.argsort(-risk, kind="stable")[:k]
    keep = np.ones(n, bool)
    keep[worst] = False
    return {
        "kept_pcf1_p10": round(float(np.quantile(tf1[keep], 0.10)), 4),
        "precision": round(float(bad[worst].mean()), 4),
        "recall": round(float(bad[worst].sum() / max(1, bad.sum())), 4),
        # None, not 0.0: with no bad cell or no good one, average precision is
        # undefined, and a literal zero reads as a flag that ranked perfectly badly.
        "ap": (round(float(average_precision_score(bad, risk)), 4)
               if 0 < bad.sum() < n else None),
        "n_rejected": int(k),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="FILES the curator reviews per round. Review is by file, "
                         "since one SWC appears under several model seeds and a "
                         "file must be wholly reviewed or wholly held out.")
    ap.add_argument("--seed-files", dest="seed_files", type=int,
                    default=DEFAULT_SEED_FILES,
                    help="FILES reviewed before round 1")
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    ap.add_argument("--reject-rate", type=float, default=DEFAULT_REJECT_RATE,
                    help="the flag's operating point, as eval_flag's --reject-rates")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    # A reduced run is not the published one, so it never takes its file name.
    tag = ""
    if args.rounds != DEFAULT_ROUNDS:
        tag += f"_r{args.rounds}"
    if args.k != DEFAULT_K:
        tag += f"_k{args.k}"
    if args.seed_files != DEFAULT_SEED_FILES:
        tag += f"_start{args.seed_files}"
    if args.test_frac != DEFAULT_TEST_FRAC:
        tag += f"_test{args.test_frac:g}"
    if args.reject_rate != DEFAULT_REJECT_RATE:
        tag += f"_rr{args.reject_rate:g}"
    if args.seed != DEFAULT_SEED:
        tag += f"_seed{args.seed}"

    df = _load_joined(FLAG_DIR / "final_flag_multiseed_labels.csv",
                      FLAG_DIR / "final_flag_multiseed_features.csv")
    df = df[df["cell_type_gt"].astype(str) == SCOPE].reset_index(drop=True)
    df_feat, num, cat = _engineer_features(_ours_oof_frame(df))
    f1 = pd.to_numeric(df_feat[F1_COL], errors="coerce")
    ok = f1.notna()
    df_feat, f1 = df_feat.loc[ok].reset_index(drop=True), f1.loc[ok].reset_index(drop=True)

    # split by FILE: the same SWC appears under several seeds, and a file must
    # never be both reviewed and tested.
    rng = np.random.RandomState(args.seed)
    files = df_feat["file"].astype(str)
    uniq = np.array(sorted(files.unique()))
    rng.shuffle(uniq)
    n_test = int(round(args.test_frac * len(uniq)))
    # Keep every collection that is iterated an ordered list: sets of strings iterate
    # in an order that varies per process, and that order reaches the training matrix
    # via rows_for(), which makes the fitted model differ run to run. Sets are used
    # only for membership tests.
    test_list = list(uniq[:n_test])
    test_files = set(test_list)
    rest = list(uniq[n_test:])
    seed_files = rest[:args.seed_files]
    pool_files = rest[args.seed_files:]

    idx_of = {f: np.flatnonzero(files.to_numpy() == f) for f in uniq}
    test_idx = np.concatenate([idx_of[f] for f in test_list])
    reviewed = list(seed_files)
    overlap = set(reviewed) & test_files
    if overlap:
        # Not an assert: asserts vanish under python -O, and this is the check
        # that makes every number below meaningful.
        raise SystemExit(f"{len(overlap)} files are both reviewed and tested")

    print(f"\nflag feedback  scope={SCOPE}  cells={len(df_feat)}  files={len(uniq)}")
    print(f"  reviewed at start={len(seed_files)} files  pool={len(pool_files)} files  "
          f"test={len(test_files)} files / {len(test_idx)} cells (never reviewed)")
    print(f"  rounds={args.rounds}  k={args.k} files reviewed per round\n")

    def rows_for(fs):
        return np.concatenate([idx_of[f] for f in fs]) if fs else np.array([], dtype=int)

    t0 = time.perf_counter()
    start_pipe = _fit(df_feat, num, cat, f1, rows_for(reviewed), args.seed)
    base = _measure(start_pipe, df_feat, f1, test_idx, args.reject_rate)
    curves = {a: [{**base, "round": 0, "arm": a, "n_reviewed_files": len(reviewed),
                   "reviews_actually_made": len(reviewed)}]
              for a in ("flag", "random", "factory")}
    print(f"  round 0  reviewed={len(reviewed)}  keptP10={base['kept_pcf1_p10']}  "
          f"AP={base['ap']}", flush=True)

    for arm in ("flag", "random", "factory"):
        rev = list(reviewed)
        pool = list(pool_files)
        pipe = start_pipe
        r = np.random.RandomState(args.seed)
        for rnd in range(1, args.rounds + 1):
            if not pool:
                break
            if arm == "factory":
                # Nothing about this arm changes between rounds, so it is scored
                # once and the row repeated.
                m = dict(base)
                m.update({"round": rnd, "arm": arm,
                          "n_reviewed_files": len(reviewed) + rnd * args.k,
                          "reviews_actually_made": 0})
                curves[arm].append(m)
                continue
            if arm == "flag":
                pool_idx = rows_for(pool)
                risk = risk_scores(pipe, df_feat.loc[pool_idx])
                # rank files by their riskiest cell
                fr = {}
                for i, ridx in enumerate(pool_idx):
                    fname = files.iat[ridx]
                    fr[fname] = max(fr.get(fname, -1.0), float(risk[i]))
                picked = sorted(pool, key=lambda f: -fr.get(f, 0.0))[:args.k]
            else:                                    # random
                picked = [pool[i] for i in r.permutation(len(pool))[:args.k]]
            rev.extend(picked)
            sel = set(picked)
            pool = [f for f in pool if f not in sel]
            pipe = _fit(df_feat, num, cat, f1, rows_for(rev), args.seed)
            m = _measure(pipe, df_feat, f1, test_idx, args.reject_rate)
            m.update({"round": rnd, "arm": arm, "n_reviewed_files": len(rev),
                      "reviews_actually_made": len(rev)})
            curves[arm].append(m)
            # The printed count is the one recorded, so console and record cannot
            # disagree; the factory arm carries the other arms' budget and prints
            # nothing, since nothing about it changes.
            print(f"  [{arm}] round {rnd}  reviewed={m['n_reviewed_files']}  "
                  f"keptP10={m['kept_pcf1_p10']}  AP={m['ap']}", flush=True)

    elapsed = (time.perf_counter() - t0) / 60.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"scope": SCOPE, "n_cells": int(len(df_feat)), "n_files": int(len(uniq)),
           "n_test_cells": int(len(test_idx)), "n_test_files": int(len(test_files)),
           "review_unit": "file",
           "rounds": args.rounds, "k_files_per_round": args.k,
           "reject_rate": args.reject_rate, "bad_f1_threshold": TARGET_THRESHOLD,
           "labeler": "fixed factory pipeline (all stages)",
           "elapsed_min": round(elapsed, 2), **{f"{a}_arm": curves[a] for a in curves}}
    (args.out_dir / f"flag_feedback{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    cols = ["arm", "round", "n_reviewed_files", "reviews_actually_made",
            "kept_pcf1_p10", "precision", "recall", "ap"]
    with (args.out_dir / f"flag_feedback{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for a in curves:
            for row in curves[a]:
                w.writerow(row)

    lines = [f"Flag under curator feedback ({SCOPE}, {len(test_idx)} never-reviewed test cells)",
             "=" * 78, "",
             "Labeler fixed (full factory pipeline); only the flag is retrained.",
             f"kept-P10 = worst-decile per-cell F1 of the cells kept after a "
             f"{args.reject_rate:.0%} rejection.",
             "Review is counted in FILES: one SWC contributes a row per model seed "
             "and is reviewed or held out whole.", "",
             f"{'files rev':>9} | {'flag':>8} {'random':>8} {'factory':>8} | {'flag AP':>8}"]
    for i in range(len(curves["flag"])):
        f, rr, fa = curves["flag"][i], curves["random"][i], curves["factory"][i]
        lines.append(f"{f['n_reviewed_files']:>9} | {f['kept_pcf1_p10']:>8.4f} "
                     f"{rr['kept_pcf1_p10']:>8.4f} {fa['kept_pcf1_p10']:>8.4f} | {f['ap']:>8.4f}")
    report = "\n".join(lines)
    (args.out_dir / f"flag_feedback{tag}.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\n  wrote {display_path(args.out_dir)}/flag_feedback{tag}."
          f"{{json,csv,txt}}  ({elapsed:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
