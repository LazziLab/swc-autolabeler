"""Oracle ceiling for the pyramidal flag's leave-one-seed-out sweep (Sec 5.4, Table 8).

For each held-out seed, the oracle rejects exactly the cells with the lowest
true per-cell F1, as many as the flag rejects at the same rate, and reports the
10th percentile of the rest. Rejection counts follow ``_rejection_metrics``
(round(rate * n), at least one) and are checked against the counts eval_flag
recorded, so flag and oracle are compared on identical budgets. Values are means
over the three held-out seeds, as in eval_flag's summary.

    python -m evaluation.stage4.flag_oracle

Inputs (evaluation/results/flag/):
    final_flag_multiseed_labels.csv       per-cell held-out F1 per seed
    final_flag_leave_one_seed_out.csv     the flag's per-fold operating points
Output (evaluation/results/flag/):
    flag_oracle.{json,txt}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.paths import display_path  # noqa: E402

FLAG = ROOT / "evaluation" / "results" / "flag"
#: Table 8 sweeps the pyramidal flag.
SCOPE = "pyramidal"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path,
                    default=FLAG / "final_flag_multiseed_labels.csv",
                    help="per-cell held-out F1 per seed")
    ap.add_argument("--loso", type=Path,
                    default=FLAG / "final_flag_leave_one_seed_out.csv",
                    help="the flag's per-fold operating points")
    ap.add_argument("--out-dir", type=Path, default=FLAG,
                    help="where to write flag_oracle.{json,txt} (default: beside the inputs)")
    args = ap.parse_args()

    labels_path, loso_path = args.labels, args.loso
    # Both inputs come from the flag generation, so a missing one means that
    # generation has not been run rather than that a path is mistyped.
    missing = [q for q in (labels_path, loso_path) if not q.is_file()]
    if missing:
        raise SystemExit(
            "".join(f"MISSING: {q}\n" for q in missing)
            + "  run: python -m evaluation.stage4.combine_flag_multiseed\n"
              "  then: python -m evaluation.stage4.eval_flag")
    # The two must come from the same generation: the cell counts below catch a
    # gross mismatch, but two files regenerated hours apart can still agree on
    # counts while describing different runs. Modification times are the only
    # signal the CSVs carry, and they survive only in the tree that wrote them:
    # a clone or a copy stamps every file alike, so this warning is silent for a
    # reader and useful only where the files were generated.
    skew_h = abs(labels_path.stat().st_mtime - loso_path.stat().st_mtime) / 3600.0
    if skew_h > 24:
        print(f"warning: {labels_path.name} and {loso_path.name} were written "
              f"{skew_h:.0f} hours apart; they may not describe the same run.", flush=True)
    labels = pd.read_csv(labels_path)
    loso = pd.read_csv(loso_path)
    fixed = loso[(loso["scope"] == SCOPE) & (loso["operating_point"] == "fixed_reject_rate")].copy()
    rates = sorted(float(r) for r in fixed["reject_rate"].unique())
    cells = labels[labels["cell_type_gt"] == SCOPE].copy()
    # The seed identifies a fold in one file and a model in the other. They are
    # matched below, so both are read as integers rather than as whatever dtype
    # each CSV happened to infer; a string against an int would match nothing
    # and surface as an empty fold instead of as the type mismatch it is.
    fixed["test_seed"] = fixed["test_seed"].astype(int)
    cells["model_seed"] = cells["model_seed"].astype(int)

    per_seed = {}
    for seed, part in fixed.groupby("test_seed"):
        # Sorted ascending, so the oracle's rejection below is simply f1[k:].
        f1 = np.sort(cells.loc[cells["model_seed"] == seed, "held_out_F1"].to_numpy(float))
        n_test = int(part["n_test"].iloc[0])
        if len(f1) != n_test:
            raise SystemExit(f"seed {seed}: {len(f1)} labeled cells, eval_flag tested {n_test}")
        rows = {"no_flag_p10": float(np.percentile(f1, 10)), "rates": {}}
        for rate in rates:
            # The budget rule is a copy of the one in
            # pipeline.stage4.flag_model._rejection_metrics, which cannot be
            # called on a CSV. The assertion below is what holds the two
            # together: change either and this run stops.
            k = max(1, int(round(rate * len(f1))))
            hits = part[np.isclose(part["reject_rate"], rate)]
            if len(hits) != 1:
                raise SystemExit(f"seed {seed} rate {rate}: {len(hits)} operating-point "
                                 f"rows in {loso_path.name}, expected exactly one")
            flag_row = hits.iloc[0]
            if k != int(flag_row["n_rejected"]):
                raise SystemExit(f"seed {seed} rate {rate}: oracle rejects {k}, flag rejected {int(flag_row['n_rejected'])}")
            rows["rates"][f"{rate:.2f}"] = {"n_rejected": k, "flag_kept_p10": float(flag_row["kept_f1_p10"]),
                                           "oracle_kept_p10": float(np.percentile(f1[k:], 10))}
        per_seed[str(int(seed))] = rows

    seeds = sorted(per_seed, key=int)
    summary = {"no_flag_p10": float(np.mean([per_seed[s]["no_flag_p10"] for s in seeds])), "rates": {}}
    for rate in rates:
        key = f"{rate:.2f}"
        flag = float(np.mean([per_seed[s]["rates"][key]["flag_kept_p10"] for s in seeds]))
        oracle = float(np.mean([per_seed[s]["rates"][key]["oracle_kept_p10"] for s in seeds]))
        gain = oracle - summary["no_flag_p10"]
        summary["rates"][key] = {"flag_kept_p10": flag, "oracle_kept_p10": oracle, "gap_to_oracle": oracle - flag,
                                 "share_of_oracle_gain": (flag - summary["no_flag_p10"]) / gain if gain > 0 else float("nan")}

    out = {"scope": SCOPE, "seeds": [int(s) for s in seeds],
           "labels": display_path(labels_path), "loso": display_path(loso_path),
           "summary": summary, "per_seed": per_seed}
    # The inputs are options, so a run reading anything but the published pair
    # takes its own file name and never overwrites the published output.
    tag = "" if (labels_path == FLAG / "final_flag_multiseed_labels.csv"
                 and loso_path == FLAG / "final_flag_leave_one_seed_out.csv") else "_custom"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"flag_oracle{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [f"FLAG VS ORACLE -- leave-one-seed-out, {SCOPE}, mean over held-out seeds {', '.join(seeds)}",
             f"no flag: kept-P10 {summary['no_flag_p10']:.4f}", "",
             "  reject   flag P10   oracle P10   gap    share of oracle gain"]
    for key, v in summary["rates"].items():
        share = v["share_of_oracle_gain"]
        # nan when the oracle itself gains nothing at this rate; printing it
        # as a percentage would read as a real figure.
        share_s = "--" if share != share else f"{share:.0%}"
        lines.append(f"  {float(key):5.0%}    {v['flag_kept_p10']:.4f}     {v['oracle_kept_p10']:.4f}    "
                     f"{v['gap_to_oracle']:.3f}   {share_s:>5s}")
    (args.out_dir / f"flag_oracle{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
