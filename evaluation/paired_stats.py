"""Paired per-cell comparison of the pipeline against each baseline (the §5.1 claim).

The headline tables report means over seeds, which says nothing about whether a
difference is larger than cell-to-cell noise. This driver pairs the two systems
cell for cell on exactly the same held-out reconstructions and asks whether the
difference survives resampling.

The paired unit is the SWC file, not the (seed, file) row. A file that lands in
more than one seed's test set is scored more than once, and those repeats are not
independent evidence, so its paired differences are averaged into a single value
before any test runs. Tests are then:

  sign-flip permutation   the null is that the sign of each cell's difference is
                          arbitrary, which is the right null for paired data and
                          assumes nothing about the distribution of differences
  bootstrap CI            95% interval for the mean difference, and separately
                          for the difference in 10th percentiles, since the
                          worst decile is the quantity the paper argues about

Inputs, both per-cell and both on the same GT-cell-type configuration:
    evaluation/results/per_cell_f1_seed{seed}.csv        (from evaluation.eval_ablation)
    evaluation/results/baselines/baselines_seed{seed}.csv (from evaluation.baselines.eval_baselines)

Usage:
    python -m evaluation.paired_stats

Output (evaluation/results/):
    paired_stats.json
    paired_stats.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
BASELINE_DIR = RESULTS / "baselines"
METRIC = "neurite_macro_f1"
SEEDS = (42, 123, 789)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"MISSING {path}\n  run evaluation.eval_ablation and "
                         f"evaluation.baselines.eval_baselines first")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_pairs(seeds: list[int]) -> dict[str, list[dict]]:
    """Pair ours against each baseline method, one row per (seed, file, method)."""
    by_method: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        ours = {r["file"]: r for r in _read_csv(RESULTS / f"per_cell_f1_seed{seed}.csv")}
        for b in _read_csv(BASELINE_DIR / f"baselines_seed{seed}.csv"):
            o = ours.get(b["file"])
            if o is None:                       # cell not in this seed's ablation run
                continue
            try:
                ov, bv = float(o[METRIC]), float(b[METRIC])
            except (TypeError, ValueError):
                print(f"  warning: seed {seed} {b['file']} has no {METRIC} in one of "
                      f"the two tables; excluded from the pairing", flush=True)
                continue
            by_method[b["method"]].append(
                {"file": b["file"], "ours": ov, "base": bv, "diff": ov - bv})
    return by_method


def _cluster_by_file(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated (seed, file) scores of the same file into one value each."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)
    d = [float(np.mean([r["diff"] for r in v])) for v in by_file.values()]
    o = [float(np.mean([r["ours"] for r in v])) for v in by_file.values()]
    b = [float(np.mean([r["base"] for r in v])) for v in by_file.values()]
    return np.asarray(d), np.asarray(o), np.asarray(b)


def _sign_flip_p(diffs: np.ndarray, rng, n_perm: int) -> float:
    """Two-sided p for mean(diff) != 0 under arbitrary per-cell sign flips."""
    obs = abs(float(diffs.mean()))
    if diffs.size == 0 or obs == 0.0:
        return 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diffs.size))
    perm = np.abs((signs * diffs).mean(axis=1))
    return float((np.count_nonzero(perm >= obs) + 1) / (n_perm + 1))


def _boot_mean_ci(v: np.ndarray, rng, n_boot: int) -> tuple[float, float]:
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _boot_p10_ci(o: np.ndarray, b: np.ndarray, rng, n_boot: int) -> tuple[float, float, float]:
    """Difference of the two 10th percentiles, with a paired bootstrap interval.

    The statistic itself is not paired: it compares where each system's worst
    decile sits, which is the quantity the paper argues about. The resampling is
    paired, one index array drawn for both, so the interval reflects the
    cell-to-cell variation the two systems share rather than treating them as
    independent samples.
    """
    obs = float(np.percentile(o, 10) - np.percentile(b, 10))
    idx = rng.integers(0, o.size, size=(n_boot, o.size))
    d = np.percentile(o[idx], 10, axis=1) - np.percentile(b[idx], 10, axis=1)
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--n-perm", type=int, default=100_000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the resampling")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pairs = _load_pairs(seeds)
    if not pairs:
        raise SystemExit("no paired cells found")

    out = []
    for method in sorted(pairs, key=lambda m: -float(np.mean([r["diff"] for r in pairs[m]]))):
        diffs, o, b = _cluster_by_file(pairs[method])
        # A generator per method, seeded from its name, so a method's intervals
        # do not depend on how many methods were resampled before it. One shared
        # stream would make every result a function of the row order above,
        # which is itself sorted by the observed difference.
        mrng = np.random.default_rng(
            [args.seed, int.from_bytes(method.encode(), "little") % (2**31)])
        lo, hi = _boot_mean_ci(diffs, mrng, args.n_boot)
        p10, p10lo, p10hi = _boot_p10_ci(o, b, mrng, args.n_boot)
        wins = int(np.count_nonzero(diffs > 0))
        out.append({
            "baseline": method, "metric": METRIC,
            "n_seed_file_pairs": len(pairs[method]), "n_unique_files": int(diffs.size),
            "ours_mean": float(o.mean()), "baseline_mean": float(b.mean()),
            "mean_diff": float(diffs.mean()), "mean_diff_ci95": [lo, hi],
            "sign_flip_p_two_sided": _sign_flip_p(diffs, mrng, args.n_perm),
            "p10_diff": p10, "p10_diff_ci95": [p10lo, p10hi],
            "ours_win_files": wins, "baseline_win_files": int(np.count_nonzero(diffs < 0)),
            "ours_win_fraction": wins / max(1, int(diffs.size)),
        })

    lines = [
        f"Paired per-cell comparison, metric = {METRIC}",
        f"seeds {', '.join(map(str, seeds))}; paired by file, repeats averaged before testing",
        f"{args.n_perm:,} sign-flip permutations; {args.n_boot:,} bootstrap resamples",
        "",
        f"  {'baseline':13s} {'ours':>7s} {'base':>7s} {'diff':>8s} {'95% CI':>18s} "
        f"{'p':>10s} {'P10 diff':>9s} {'win%':>6s}",
    ]
    for r in out:
        ci = f"[{r['mean_diff_ci95'][0]:+.4f},{r['mean_diff_ci95'][1]:+.4f}]"
        p = r["sign_flip_p_two_sided"]
        pstr = f"<{1.0 / (args.n_perm + 1):.1e}" if p <= 1.0 / (args.n_perm + 1) else f"{p:.2e}"
        lines.append(f"  {r['baseline']:13s} {r['ours_mean']:7.4f} {r['baseline_mean']:7.4f} "
                     f"{r['mean_diff']:+8.4f} {ci:>18s} {pstr:>10s} "
                     f"{r['p10_diff']:+9.4f} {r['ours_win_fraction']:6.1%}")
    # Each method joins against its own baseline table, so the paired set can
    # differ between them; reporting one method's count as the run's would hide
    # a baseline that covers fewer cells.
    ns = sorted({r["n_unique_files"] for r in out})
    n_txt = str(ns[0]) if len(ns) == 1 else f"{ns[0]} to {ns[-1]}"
    lines += ["", f"n = {n_txt} unique files per baseline "
                  f"({out[0]['n_seed_file_pairs']} seed-file pairs before clustering "
                  f"for {out[0]['baseline']})"]
    report = "\n".join(lines)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paired_stats.json").write_text(
        json.dumps({"seeds": seeds, "metric": METRIC, "n_perm": args.n_perm,
                    "n_boot": args.n_boot, "results": out}, indent=2), encoding="utf-8")
    (args.out_dir / "paired_stats.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nwrote {args.out_dir.name}/paired_stats.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
