"""Leakage-free selection and sensitivity of the pipeline's decision threshold,
the branch-correction override tau (shipped at 0.65).

Protocol. For every held-out cell of every seed, one cached record captures
everything the threshold acts on: Stage 1's predicted type and, for each cell
type, the pre-correction labels and every branch's correction proposal (node
indices, current label, proposed class, posterior). Any tau then composes
offline into exactly the labels the pipeline would emit, replicating
``run_pipeline_on_nodes`` and ``apply_correction`` rule for rule, in the
deployment configuration (Stage 1's type) and under the ground-truth type.

Selection is leave-one-seed-out: seed S's tau is chosen to maximize deployment
neurite macro-F1 over the other two seeds' cells, and seed S is scored at that
choice. The seeds' held-out sets overlap, so a file that is also among seed S's
cells is left out of the pool that selects for S, and no cell contributes to
both selecting and scoring. The report also
gives the full sensitivity curves (grid 0.50-0.90, step 0.05) in both
configurations, and the metric cost of shipping the single value 0.65 instead
of the per-seed LOSO choice.

    python -m evaluation.threshold_sensitivity            # cache + sweep
    python -m evaluation.threshold_sensitivity --from-cache

Output: evaluation/results/threshold_sensitivity.{json,txt}; cache under
evaluation/results/_threshold_cache/ (gitignored size, regenerable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                              # noqa: E402
from pipeline.infer_pipeline import (                                # noqa: E402
    run_pipeline_on_nodes, load_stage2_bundle, _run_stage23,
)
from pipeline.stage1.cell_type_detector import detect_cell_type_from_nodes  # noqa: E402
from pipeline.branch_features import (                               # noqa: E402
    BRANCH_FEATURE_NAMES, extract_branches,
)
from pipeline.stage3.branch_correction import scope_mask, CLASS_TO_TYPE  # noqa: E402
from evaluation.paths import display_path                            # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed, per_cell_neurite_f1
from pipeline.split import in_test  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "evaluation" / "results"
CACHE = RESULTS / "_threshold_cache"
DEFAULT_TEST_SIZE = 0.20
#: Cells self_check compares against the live pipeline before anything is composed.
DEFAULT_CHECK_CELLS = 12
GRID = [round(0.50 + 0.05 * i, 2) for i in range(9)]     # 0.50 .. 0.90
SEEDS = (42, 123, 789)


def load_correction_model(md: Path):
    p = md / "branch_correction.joblib"
    if not p.is_file():
        return None
    import joblib  # noqa: PLC0415
    return joblib.load(p)


def _cache_dir(limit: int, test_size: float = DEFAULT_TEST_SIZE) -> Path:
    """The full cache, or a separate one for a run that is not the published one.

    A truncated run can never be mistaken for the real one, and neither can a run
    over a different split: the split fraction decides which cells are cached, and
    keying on --limit alone meant a --test-size run silently read the 0.20 cache.
    """
    tag = ""
    if limit:
        tag += f"limit{limit}"
    if test_size != DEFAULT_TEST_SIZE:
        tag += f"test{test_size:g}"
    return CACHE / tag if tag else CACHE


def _cache_stamp(md: Path) -> str:
    """What a cached record was built from: the branch feature layout and the
    correction model. A cache that does not carry the current stamp holds
    proposals the current pipeline does not make, and every tau composed from
    it describes a different labeler. self_check catches some of this, but only
    on the cells it samples.
    """
    h = hashlib.md5("|".join(BRANCH_FEATURE_NAMES).encode())
    for name in ("branch_correction.joblib", "branch_classifier.pkl"):
        f = md / name
        h.update(f.read_bytes() if f.is_file() else b"missing")
    return h.hexdigest()[:12]


def cache_seed(seed: int, limit: int = 0, test_size: float = DEFAULT_TEST_SIZE) -> Path:
    md = ROOT / "pipeline" / "models" / f"seed{seed}"
    check_model_seed(md, seed, test_size)
    corr = load_correction_model(md)
    bundle = load_stage2_bundle(md / "branch_classifier.pkl")
    files = [(ct, p) for ct in ("pyramidal", "interneuron")
             for p in sorted((DATA / ct).glob("*.swc")) if in_test(p.name, seed, test_size)]
    if limit:
        files = files[:limit]
    print(f"seed {seed}: caching {len(files)} test cells", flush=True)
    recs, t0 = [], time.perf_counter()
    for i, (ct_gt, p) in enumerate(files, 1):
        nodes = parse_swc(p)
        if not nodes:
            continue
        s1 = detect_cell_type_from_nodes(nodes, md / "cell_type_classifier.pkl")
        rec = {"file": p.name, "ct_gt": ct_gt, "gt": [n.type for n in nodes],
               "s1_type": s1.cell_type, "s1_conf": float(s1.confidence),
               "arms": {}}
        # The correction's branch features are label-free, so one proposal set
        # serves both arms; it only ever applies on the pyramidal arm.
        proposals = []
        if corr is not None:
            morph = extract_branches(nodes, cell_type="pyramidal",
                                     file_path=str(p))
            if morph.branches:
                feats = np.stack([br.features for br in morph.branches], 0)
                proba = corr["model"].predict_proba(feats)
                # Only in-scope branches are ever proposals (see scope_mask).
                in_scope = scope_mask(corr, feats)
                proposals = [(list(br.node_indices), int(proba[bi].argmax()),
                              float(proba[bi].max()))
                             for bi, br in enumerate(morph.branches) if in_scope[bi]]
        for forced in ("pyramidal", "interneuron"):
            trial = _run_stage23(nodes, str(p), forced, bundle, s1)
            rec["arms"][forced] = {
                "labels": list(trial["final_labels"]),
                "mean_conf": float(trial["mean_neurite_conf"]),
                "branches": proposals if forced == "pyramidal" else [],
            }
        recs.append(rec)
        if i % 300 == 0:
            print(f"  {i}/{len(files)} ({(time.perf_counter()-t0)/60:.1f} min)",
                  flush=True)
    cdir = _cache_dir(limit, test_size)
    cdir.mkdir(parents=True, exist_ok=True)
    out = cdir / f"seed{seed}.pkl"
    with out.open("wb") as fh:
        pickle.dump({"stamp": _cache_stamp(md), "seed": seed,
                     "test_size": test_size, "records": recs}, fh)
    print(f"  -> {out.name} ({len(recs)} cells)", flush=True)
    return out


def load_cache(seed: int, limit: int = 0, test_size: float = DEFAULT_TEST_SIZE) -> list:
    """One seed's cached records, refusing a cache built from a different labeler.

    A cache without a stamp, a bare list, cannot be checked and is refused
    rather than composed from, since no tau read out of it can be tied to the
    current labeler.
    """
    path = _cache_dir(limit, test_size) / f"seed{seed}.pkl"
    if not path.is_file():
        raise SystemExit(f"MISSING {display_path(path)}\n"
                         f"  build it: python -m evaluation.threshold_sensitivity")
    with path.open("rb") as fh:
        blob = pickle.load(fh)
    if not isinstance(blob, dict) or "records" not in blob:
        raise SystemExit(f"{display_path(path)} carries no cache stamp, so it cannot "
                         f"be checked against the current correction. Rebuild it by "
                         f"running without --from-cache.")
    want = _cache_stamp(ROOT / "pipeline" / "models" / f"seed{seed}")
    if blob.get("stamp") != want:
        raise SystemExit(f"{display_path(path)} was built from a different branch "
                         f"feature set or correction model. Rebuild it by running "
                         f"without --from-cache.")
    return blob["records"]


def self_check(seed: int, n: int = DEFAULT_CHECK_CELLS, limit: int = 0,
               test_size: float = DEFAULT_TEST_SIZE) -> None:
    """Check composed labels equal the live pipeline at the shipped thresholds.

    Not asserts: python -O removes them, and this is the whole reason composing a
    label offline is allowed at all. It samples ``n`` cells, so it is a check on
    the composition rule rather than on every cached record; the cache stamp is
    what covers the rest.
    """
    md = ROOT / "pipeline" / "models" / f"seed{seed}"
    recs = load_cache(seed, limit, test_size)
    for rec in recs[:n]:
        ct_dir = rec["ct_gt"]
        p = DATA / ct_dir / rec["file"]
        nodes = parse_swc(p)
        live_dep = run_pipeline_on_nodes(
            nodes, str(p), stage1_model=md / "cell_type_classifier.pkl",
            stage2_model=md / "branch_classifier.pkl")
        if list(live_dep.node_labels) != cell_pred(rec, 0.65, True):
            raise SystemExit(
                f"deployment mismatch on {rec['file']}: the offline composition no "
                f"longer reproduces the pipeline. compose() and "
                f"branch_correction.apply_correction have drifted apart.")
        live_gt = run_pipeline_on_nodes(
            nodes, str(p), stage1_model=md / "cell_type_classifier.pkl",
            stage2_model=md / "branch_classifier.pkl", override_cell_type=rec["ct_gt"])
        if list(live_gt.node_labels) != cell_pred(rec, 0.65, False):
            raise SystemExit(
                f"GT-config mismatch on {rec['file']}: the offline composition no "
                f"longer reproduces the pipeline.")
    print(f"self-check OK: {min(n, len(recs))} cells match the live pipeline exactly",
          flush=True)


def compose(rec, forced: str, tau: float):
    """Final labels for one arm at override tau, replicating apply_correction.

    The override rule below is a copy of the one in
    ``pipeline.stage3.branch_correction.apply_correction``, because composing a
    label offline cannot call it. ``self_check`` is what holds the two together:
    change either and it fails. Keep them in step.
    """
    arm = rec["arms"][forced]
    labels = list(arm["labels"])
    for node_idx, cls, conf in arm["branches"]:
        new = CLASS_TO_TYPE[cls]
        cur = labels[node_idx[0]]
        if conf >= tau and new != cur and cur in (2, 3, 4):
            for ni in node_idx:
                labels[ni] = new
    return labels


def cell_pred(rec, tau: float, deployment: bool):
    """The pipeline's final labels for one cell at tau: Stage 1's type in
    deployment, the ground-truth type otherwise."""
    forced = rec["s1_type"] if deployment else rec["ct_gt"]
    return compose(rec, forced, tau)


def macro_f1(gt, pred):
    from sklearn.metrics import f1_score  # noqa: PLC0415
    return float(f1_score(gt, pred, labels=[2, 3, 4], average="macro",
                          zero_division=0))


def sweep(per_seed_recs, deployment: bool):
    """metric[tau] per seed: corpus neurite macro-F1 and pcF1 P10."""
    out = {}
    for seed, recs in per_seed_recs.items():
        out[seed] = {}
        for tau in GRID:
            gtp, prp, pc = [], [], []
            for rec in recs:
                pred = cell_pred(rec, tau, deployment)
                gtp.extend(rec["gt"])
                prp.extend(pred)
                pc.append(per_cell_neurite_f1(rec["gt"], pred, rec["ct_gt"]))
            out[seed][tau] = {
                "macro_f1": macro_f1(gtp, prp),
                "pcf1_p10": float(np.percentile(pc, 10)),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help="held-out fraction of the hash split (default 0.20). The "
                         "cache is keyed on it, so a different split builds its own.")
    ap.add_argument("--check-cells", type=int, default=DEFAULT_CHECK_CELLS,
                    help="cells self_check compares against the live pipeline")
    args = ap.parse_args()

    if not args.from_cache:
        for s in SEEDS:
            cache_seed(s, args.limit, args.test_size)
    # Also on the --from-cache path: a cache built against a different correction
    # composes labels the current pipeline does not emit. The stamp load_cache
    # verifies catches that for every record; this catches a drift in the rule itself.
    self_check(SEEDS[0], n=args.check_cells, limit=args.limit, test_size=args.test_size)

    per_seed = {s: load_cache(s, args.limit, args.test_size) for s in SEEDS}

    dep = sweep(per_seed, deployment=True)
    gt = sweep(per_seed, deployment=False)

    lines = ["THRESHOLD SELECTION AND SENSITIVITY (leave-one-seed-out)", ""]
    result = {"grid": GRID, "shipped": {"tau": 0.65},
              "loso": {}, "deployment_tau_curves": {}, "gt_tau_curves": {}}

    for s in SEEDS:
        others = [o for o in SEEDS if o != s]
        # LOSO: pick tau maximizing mean deployment macro-F1 on the other seeds.
        # The seeds' test sets overlap, so a file that is also among seed s's
        # held-out cells is left out of the pool that selects for s.
        held = {rec["file"] for rec in per_seed[s]}
        pool = {o: [rec for rec in per_seed[o] if rec["file"] not in held] for o in others}
        n_dropped = sum(len(per_seed[o]) - len(pool[o]) for o in others)
        sel = sweep(pool, deployment=True)
        def sel_score(tau):
            return statistics.fmean(sel[o][tau]["macro_f1"] for o in others)
        best = max(GRID, key=sel_score)
        at_best = dep[s][best]
        at_ship = dep[s][0.65]
        result["loso"][s] = {
            "selected_tau": best,
            "n_selection_cells": sum(len(v) for v in pool.values()),
            "n_shared_cells_left_out": n_dropped,
            "macro_f1_at_selected": at_best["macro_f1"],
            "macro_f1_at_shipped": at_ship["macro_f1"],
            "p10_at_selected": at_best["pcf1_p10"],
            "p10_at_shipped": at_ship["pcf1_p10"],
        }
        lines.append(
            f"seed {s}: LOSO selects tau {best:.2f} | "
            f"macro-F1 selected {at_best['macro_f1']:.4f} vs shipped "
            f"{at_ship['macro_f1']:.4f} (delta "
            f"{at_best['macro_f1']-at_ship['macro_f1']:+.4f}) | "
            f"P10 {at_best['pcf1_p10']:.4f} vs {at_ship['pcf1_p10']:.4f}")

    def curves(res):
        return {"macro_f1": [statistics.fmean(res[s][t]["macro_f1"] for s in SEEDS) for t in GRID],
                "pcf1_p10": [statistics.fmean(res[s][t]["pcf1_p10"] for s in SEEDS) for t in GRID]}

    result["deployment_tau_curves"] = curves(dep)
    result["gt_tau_curves"] = curves(gt)
    for title, cv in (("Deployment sensitivity to tau, 3-seed mean:", result["deployment_tau_curves"]),
                      ("GT-cell-type sensitivity to tau, 3-seed mean:", result["gt_tau_curves"])):
        lines += ["", title,
                  "  tau     " + " ".join(f"{t:>7.2f}" for t in GRID),
                  "  macroF1 " + " ".join(f"{v:7.4f}" for v in cv["macro_f1"]),
                  "  P10     " + " ".join(f"{v:7.4f}" for v in cv["pcf1_p10"])]

    RESULTS.mkdir(parents=True, exist_ok=True)
    # A run that is not the published one never takes its file name.
    tag = f"_limit{args.limit}" if args.limit else ""
    if args.test_size != DEFAULT_TEST_SIZE:
        tag += f"_test{args.test_size:g}"
    result["test_size"] = args.test_size
    stem = RESULTS / f"threshold_sensitivity{tag}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {display_path(stem)}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
