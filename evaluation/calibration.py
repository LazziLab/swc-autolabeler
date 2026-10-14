"""Are the pipeline's confidences calibrated where they are thresholded or compared?

Three posteriors drive decisions: Stage 1's cell-type probability, the Stage 2
subtree-owner posterior (every node's confidence, which the topology rules
weigh), and the branch correction's posterior (a branch is overridden at 0.65).

For each posterior this reports a reliability table, the expected calibration
error over ten equal-width confidence bins and the multi-class Brier score, in
distribution (the held-out cells of each seed, scored by that seed's models)
and out of distribution (the external pyramidal and interneuron sets, scored by
the seed-123 models). Stage 2 is read under the ground-truth cell type; a
subtree's or branch's target is its majority ground-truth label, as in training;
node-weighted variants weight each unit by its node count. For the correction
the branches at or above the override threshold are also reported alone.

Usage:
    python -m evaluation.calibration [--jobs 4]

Output (evaluation/results/): calibration.{json,txt}
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
EXTERNAL = DATA / "external"
RESULTS = ROOT / "evaluation" / "results"
SEEDS = (42, 123, 789)
LABELS = {"pyramidal": (2, 3, 4), "interneuron": (2, 3)}
TAU = 0.65
UNITS = ("stage1", "stage2", "correction", "correction_above_tau")


def _new_agg() -> dict:
    return {k: np.zeros(10) for k in ("n", "w", "conf", "wconf", "correct", "wcorrect")} | {"brier": 0.0, "wbrier": 0.0}


def _add(agg: dict, conf, correct, brier, weight) -> None:
    conf, correct, brier, weight = (np.asarray(v, dtype=float) for v in (conf, correct, brier, weight))
    if not len(conf):
        return
    b = np.minimum(9, (conf * 10).astype(int))
    np.add.at(agg["n"], b, 1.0)
    np.add.at(agg["w"], b, weight)
    np.add.at(agg["conf"], b, conf)
    np.add.at(agg["wconf"], b, weight * conf)
    np.add.at(agg["correct"], b, correct)
    np.add.at(agg["wcorrect"], b, weight * correct)
    agg["brier"] += float(brier.sum())
    agg["wbrier"] += float((weight * brier).sum())


def _merge(aggs: list[dict]) -> dict:
    out = _new_agg()
    for a in aggs:
        for k in out:
            out[k] = out[k] + a[k]
    return out


def _summary(agg: dict, weighted: bool) -> dict:
    n = agg["w" if weighted else "n"]
    conf = agg["wconf" if weighted else "conf"]
    corr = agg["wcorrect" if weighted else "correct"]
    total = float(n.sum())
    if total <= 0:
        return {"n": 0}
    return {"units": int(agg["n"].sum()), "weight": total,
            "accuracy": float(corr.sum() / total), "mean_confidence": float(conf.sum() / total),
            "ece": float(np.abs(corr - conf).sum() / total),
            "brier": float(agg["wbrier" if weighted else "brier"] / total),
            "reliability": [{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "share": float(n[i] / total),
                             "confidence": float(conf[i] / n[i]), "accuracy": float(corr[i] / n[i])}
                            for i in range(10) if n[i] > 0]}


def _evaluate_domain(task: tuple) -> tuple[str, dict]:
    from pipeline.branch_features import extract_branches
    from pipeline.features import parse_swc
    from pipeline.infer_pipeline import load_stage2_bundle
    from pipeline.stage1.cell_type_detector import detect_cell_type_from_nodes
    from pipeline.stage2.subtree_features import extract_primary_subtrees
    from pipeline.stage3.branch_correction import scope_mask, CLASS_TO_TYPE
    from evaluation.stage4._flag_metrics import check_model_seed
    import joblib

    name, seed, cells, test_size, in_dist = task
    md = ROOT / "pipeline" / "models" / f"seed{seed}"
    # The external domain deliberately scores the seed-123 models out of
    # distribution, so only the in-distribution tasks pair model with split.
    if in_dist:
        check_model_seed(md, seed, test_size)
    s1_path = md / "cell_type_classifier.pkl"
    s2_models = load_stage2_bundle(md / "branch_classifier.pkl")["subtree_owner_models_by_cell_type"]
    corr = joblib.load(md / "branch_correction.joblib")
    aggs = {u: _new_agg() for u in UNITS}
    t0 = time.perf_counter()
    for i, (ct_gt, path) in enumerate(cells, 1):
        nodes = parse_swc(path)
        if not nodes:
            continue
        gt = np.asarray([nd.type for nd in nodes])

        r = detect_cell_type_from_nodes(nodes, s1_path)
        p_pyr = float(r.probabilities.get("pyramidal", 0.0))
        y_pyr = 1.0 if ct_gt == "pyramidal" else 0.0
        _add(aggs["stage1"], [r.confidence], [float(r.cell_type == ct_gt)], [2.0 * (p_pyr - y_pyr) ** 2], [1.0])

        model = s2_models.get(ct_gt)
        subs = extract_primary_subtrees(nodes)
        if model is not None and subs:
            probs = model.predict_proba(np.stack([s.features for s in subs]))
            classes = [int(c) for c in model.classes_]
            conf, correct, brier, weight = [], [], [], []
            for s, p in zip(subs, probs):
                if s.gt_label not in LABELS[ct_gt]:
                    continue
                k = int(np.argmax(p))
                conf.append(float(p[k]))
                correct.append(float(classes[k] == s.gt_label))
                brier.append(float(np.sum((p - np.array([float(c == s.gt_label) for c in classes])) ** 2)))
                weight.append(float(sum(s.gt_label_counts.values())))
            _add(aggs["stage2"], conf, correct, brier, weight)

        # Only pyramidal cells reach the correction, and branch extraction is the
        # most expensive step in labeling, so it is not run on the rest.
        mp_ = None
        if ct_gt == "pyramidal":
            try:
                mp_ = extract_branches(nodes, "pyramidal", str(path))
            except Exception:
                mp_ = None
        if mp_ is not None and mp_.branches:
            feats = np.stack([br.features for br in mp_.branches])
            probs = corr["model"].predict_proba(feats)
            class_map = {int(k): int(v) for k, v in corr.get("class_map", CLASS_TO_TYPE).items()}
            classes = [class_map[j] for j in range(probs.shape[1])]
            rows = {"conf": [], "correct": [], "brier": [], "weight": []}
            # Only the branches the correction may override are its decisions.
            for br, p, inside in zip(mp_.branches, probs, scope_mask(corr, feats)):
                if not inside:
                    continue
                lab = gt[np.asarray(br.node_indices)]
                m = np.isin(lab, LABELS["pyramidal"])
                if not m.any():
                    continue
                vals, cnts = np.unique(lab[m], return_counts=True)
                target = int(vals[np.argmax(cnts)])
                k = int(np.argmax(p))
                rows["conf"].append(float(p[k]))
                rows["correct"].append(float(classes[k] == target))
                rows["brier"].append(float(np.sum((p - np.array([float(c == target) for c in classes])) ** 2)))
                rows["weight"].append(float(m.sum()))
            _add(aggs["correction"], rows["conf"], rows["correct"], rows["brier"], rows["weight"])
            hi = [j for j, c in enumerate(rows["conf"]) if c >= TAU]
            _add(aggs["correction_above_tau"], [rows["conf"][j] for j in hi], [rows["correct"][j] for j in hi],
                 [rows["brier"][j] for j in hi], [rows["weight"][j] for j in hi])

        if i % 300 == 0:
            print(f"  {name}: {i}/{len(cells)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    print(f"{name}: {len(cells)} cells in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return name, {"aggs": aggs, "n_cells": len(cells)}


def _cells_in_distribution(seed: int, limit: int, test_size: float = 0.20) -> list[tuple[str, Path]]:
    from pipeline.split import in_test
    out = []
    for ct in ("pyramidal", "interneuron"):
        files = [(ct, p) for p in sorted((DATA / ct).glob("*.swc")) if in_test(p.name, seed, test_size)]
        out += files[:limit] if limit else files
    return out


def _cells_external(limit: int) -> list[tuple[str, Path]]:
    out = []
    for ct in ("pyramidal", "interneuron"):
        meta = EXTERNAL / f"{ct}.csv"
        if not meta.is_file():
            continue
        files = [(ct, EXTERNAL / ct / m["file_name"]) for m in csv.DictReader(meta.open(encoding="utf-8", newline=""))]
        out += files[:limit] if limit else files
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4, help="domains evaluated in parallel")
    ap.add_argument("--limit", type=int, default=0, help="cells per cell type per domain (0 = all; smoke tests)")
    ap.add_argument("--test-size", type=float, default=0.20,
                    help="held-out fraction of the hash split (default 0.20)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    tasks = [(f"seed{s}", s, _cells_in_distribution(s, args.limit, args.test_size), args.test_size, True)
             for s in SEEDS]
    tasks.append(("external", 123, _cells_external(args.limit), args.test_size, False))
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(tasks)))) as ex:
        parts = dict(ex.map(_evaluate_domain, tasks))

    domains = {"in_distribution": [parts[f"seed{s}"] for s in SEEDS], "external": [parts["external"]]}
    result: dict = {"tau": TAU, "bins": 10, "units": {}, "per_seed_ece": {}}
    for unit in UNITS:
        result["units"][unit] = {}
        for dom, ps in domains.items():
            agg = _merge([p["aggs"][unit] for p in ps])
            result["units"][unit][dom] = {"unweighted": _summary(agg, False), "node_weighted": _summary(agg, True)}
        eces = [_summary(parts[f"seed{s}"]["aggs"][unit], False).get("ece") for s in SEEDS]
        eces = [e for e in eces if e is not None]
        result["per_seed_ece"][unit] = {"values": eces,
                                        "mean": statistics.fmean(eces) if eces else float("nan"),
                                        "sd": statistics.stdev(eces) if len(eces) > 1 else 0.0}

    L = ["CALIBRATION OF THE PIPELINE'S POSTERIORS -- ECE over 10 equal-width bins, multi-class Brier", ""]
    for unit in UNITS:
        for dom in domains:
            for wmode in ("unweighted", "node_weighted"):
                s = result["units"][unit][dom][wmode]
                if not s.get("units"):
                    continue
                if unit == "stage1" and wmode == "node_weighted":
                    continue
                L.append(f"{unit:<22} {dom:<16} {wmode:<14} units {s['units']:>9}  acc {s['accuracy']:.4f}  "
                         f"conf {s['mean_confidence']:.4f}  ECE {s['ece']:.4f}  Brier {s['brier']:.4f}")
        L.append("")
    L.append("Reliability, in distribution (unweighted): confidence bin -> share, mean confidence, accuracy")
    for unit in UNITS:
        s = result["units"][unit]["in_distribution"]["unweighted"]
        if s.get("units"):
            L.append(f"  {unit}: " + "; ".join(f"{r['bin']} {100 * r['share']:.1f}% {r['confidence']:.3f}/{r['accuracy']:.3f}"
                                               for r in s["reliability"]))
    text = "\n".join(L)

    def _jsonable(o):
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_limit{args.limit}" if args.limit else ""     # a smoke run keeps its own files
    (args.out_dir / f"calibration{tag}.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    (args.out_dir / f"calibration{tag}.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
