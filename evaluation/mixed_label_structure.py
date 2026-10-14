"""How often does one primary subtree or one branch carry more than one label?

Stage 2 assigns one label per primary subtree and the branch
correction one label per branch, so a subtree or branch whose ground truth mixes
compartments cannot be labeled exactly by either. Over every corpus cell, from
ground truth alone, this reports:

  prevalence   mixed-label primary subtrees and branches per source and cell
               type, and label transitions inside an unbranched segment versus
               at a branch point
  ceilings     the best score any one-label-per-subtree or one-label-per-branch
               labeler could reach, each unit taking its majority ground-truth
               label (pooled node-level F1 and per-cell F1)
  categories   axon-carrying dendrites, multiple apical trunks, inverted and
               truncated apicals, and cells missing a compartment, with the
               pipeline's per-cell F1 on the held-out cells of each category
               under ground-truth cell type and in deployment

Subtrees are grouped exactly as Stage 2 labels them, by each branch's primary
root. Nodes outside every branch keep their ground-truth label in the ceilings.
A subtree is substantially mixed when its minority labels cover at least 10
nodes and 5% of its neurite nodes.

Usage:
    python -m evaluation.mixed_label_structure [--workers 4]

Output (evaluation/results/):
    mixed_label_structure.{json,txt}
    mixed_label_percell.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
RESULTS = ROOT / "evaluation" / "results"
SEEDS = (42, 123, 789)
NEURITES = {"pyramidal": (2, 3, 4), "interneuron": (2, 3)}
CLASS_NAME = {2: "axon", 3: "basal/dendrite", 4: "apical"}
SUBSTANTIAL_MIN_NODES = 10
SUBSTANTIAL_MIN_FRAC = 0.05
AXON_IN_DENDRITE_MIN = 10      # axon nodes inside a dendrite-majority subtree
APICAL_TRUNK_MIN = 20          # apical nodes for an apical-majority subtree to count as a trunk
CATEGORIES = ("axon_carrying_dendrite", "multi_apical", "apical_not_opposed",
              "apical_truncated", "no_axon", "no_apical")


def _majority_relabel(gt: np.ndarray, pred: np.ndarray, idxs, neur) -> tuple | None:
    """Set a unit's neurite nodes to their majority ground-truth label.

    Writes into ``pred`` in place and returns what the unit looked like, because
    the two callers need different halves: the ceiling needs the relabelling,
    the prevalence counts need the class count and minority size. None when the
    unit has no neurite node to take a majority over.
    """
    idxs = np.unique(np.asarray(idxs, dtype=np.int64))
    lab = gt[idxs]
    m = np.isin(lab, neur)
    if not m.any():
        return None
    vals, cnts = np.unique(lab[m], return_counts=True)
    maj = int(vals[np.argmax(cnts)])
    pred[idxs[m]] = maj
    n_neur = int(m.sum())
    return maj, n_neur, n_neur - int(cnts.max()), len(vals), lab


def _confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((5, 5), dtype=np.int64)
    ok = (gt >= 1) & (gt <= 4) & (pred >= 1) & (pred <= 4)
    np.add.at(cm, (gt[ok], pred[ok]), 1)
    return cm


def _analyze(task):
    from pipeline.branch_features import extract_branches
    from pipeline.features import parse_swc
    from pipeline.stage4.flag_metrics import per_cell_neurite_f1
    from evaluation.stage4._flag_metrics import _structural_features

    ct, path = task
    path = Path(path)
    nodes = parse_swc(path)
    if not nodes:
        return None
    try:
        morph = extract_branches(nodes, ct, str(path))
    except Exception as exc:
        # Reported rather than dropped: a cell missing here leaves every
        # prevalence denominator below quietly smaller.
        return {"file": path.name, "cell_type": ct, "failed": f"{type(exc).__name__}: {exc}"}
    neur = NEURITES[ct]
    gt = np.asarray([nd.type for nd in nodes], dtype=np.int64)
    n = len(nodes)

    node_branch = np.full(n, -1, dtype=np.int64)
    subtree_nodes: dict[int, list[int]] = defaultdict(list)
    for bi, br in enumerate(morph.branches):
        node_branch[np.asarray(br.node_indices, dtype=np.int64)] = bi
        if br.primary_root_idx is not None:
            subtree_nodes[int(br.primary_root_idx)].extend(br.node_indices)

    # The ceilings start from ground truth, so a node in no branch keeps its true
    # label and costs the ceiling nothing. Branch extraction covers every
    # non-soma node of a connected cell, so this is normally vacuous; the count
    # is carried out so a corpus where it stops being true cannot pass unnoticed.
    uncovered = int(((node_branch < 0) & np.isin(gt, neur)).sum())

    pred_sub = gt.copy()
    sub_any = sub_substantial = apical_trunks = 0
    acd = False
    for idxs in subtree_nodes.values():
        res = _majority_relabel(gt, pred_sub, idxs, neur)
        if res is None:
            continue
        maj, n_neur, minority, n_classes, lab = res
        sub_any += int(n_classes > 1)
        sub_substantial += int(minority >= SUBSTANTIAL_MIN_NODES and minority / n_neur >= SUBSTANTIAL_MIN_FRAC)
        apical_trunks += int(maj == 4 and int((lab == 4).sum()) >= APICAL_TRUNK_MIN)
        acd = acd or (maj in (3, 4) and int((lab == 2).sum()) >= AXON_IN_DENDRITE_MIN)

    pred_br = gt.copy()
    br_mixed = 0
    for br in morph.branches:
        res = _majority_relabel(gt, pred_br, br.node_indices, neur)
        br_mixed += int(res is not None and res[3] > 1)

    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    within = at_point = 0
    for i, nd in enumerate(nodes):
        p = id_to_idx.get(nd.parent)
        if p is None or gt[i] == gt[p] or gt[i] not in neur or gt[p] not in neur:
            continue
        # Both endpoints must actually be in a branch: two nodes outside every
        # branch share node_branch == -1 and would otherwise read as one branch.
        if node_branch[i] >= 0 and node_branch[p] >= 0 and node_branch[i] == node_branch[p]:
            within += 1
        else:
            at_point += 1

    sf = _structural_features(nodes)
    cos_ab = float(sf.get("apical_basal_cos", float("nan")))
    reach = float(sf.get("apical_reach_um", 0.0) or 0.0)
    apical_frac = float(sf.get("apical_frac", 0.0) or 0.0)
    axon_frac = float(sf.get("axon_frac", 0.0) or 0.0)
    pyr = ct == "pyramidal"
    cats = {
        "axon_carrying_dendrite": acd,
        "multi_apical": pyr and apical_trunks >= 2,
        "apical_not_opposed": pyr and not math.isnan(cos_ab) and cos_ab > 0,
        "apical_truncated": pyr and 0.0 < reach < 100.0 and apical_frac > 0.01,
        "no_axon": axon_frac == 0.0,
        "no_apical": pyr and apical_frac == 0.0,
    }
    return {
        "file": path.name, "cell_type": ct,
        "source": path.name.split("__", 1)[0] if "__" in path.name else "",
        "n_nodes": n, "n_neurite_nodes_outside_any_branch": uncovered,
        "n_subtrees": len(subtree_nodes), "subtrees_mixed_any": sub_any,
        "subtrees_mixed_substantial": sub_substantial, "n_branches": len(morph.branches),
        "branches_mixed": br_mixed, "transitions_within_branch": within,
        "transitions_at_branch_point": at_point, "apical_trunks": apical_trunks,
        **{k: bool(v) for k, v in cats.items()},
        "f1_oracle_subtree": per_cell_neurite_f1(gt.tolist(), pred_sub.tolist(), ct),
        "f1_oracle_branch": per_cell_neurite_f1(gt.tolist(), pred_br.tolist(), ct),
        "cm_subtree": _confusion(gt, pred_sub).tolist(),
        "cm_branch": _confusion(gt, pred_br).tolist(),
    }


def _f1_from_cm(cm, classes) -> dict:
    cm = np.asarray(cm)
    out = {}
    for c in classes:
        tp = cm[c, c]
        denom = 2 * tp + (cm[:, c].sum() - tp) + (cm[c, :].sum() - tp)
        out[CLASS_NAME[c]] = float(2 * tp / denom) if denom else float("nan")
    present = [out[CLASS_NAME[c]] for c in classes if cm[c, :].sum() > 0]
    out["neurite_macro_f1"] = float(np.mean(present)) if present else float("nan")
    return out


def _dist(vals) -> dict:
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "p10": float(np.quantile(v, 0.10)),
            "pct_bad": float(100.0 * (v < 0.60).mean())}


def _per_cell_scores() -> tuple[dict, dict]:
    """Held-out per-cell F1 keyed by (seed, file): GT cell type and deployment."""
    gtt, dep = {}, {}
    for s in SEEDS:
        p = RESULTS / f"per_cell_f1_seed{s}.csv"
        if p.is_file():
            for r in csv.DictReader(p.open(encoding="utf-8")):
                gtt[(s, r["file"])] = float(r["neurite_macro_f1"])
        q = RESULTS / "flag" / f"final_flag_seed{s}_labels.csv"
        if q.is_file():
            for r in csv.DictReader(q.open(encoding="utf-8")):
                if r.get("held_out_F1", "") != "":
                    dep[(s, r["file"])] = float(r["held_out_F1"])
    return gtt, dep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cells per cell type (0 = all; smoke tests)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    tasks = []
    for ct in ("pyramidal", "interneuron"):
        files = sorted((DATA / ct).glob("*.swc"))
        tasks += [(ct, str(p)) for p in (files[:args.limit] if args.limit else files)]
    print(f"analyzing {len(tasks)} cells with {args.workers} workers", flush=True)
    rows, failed, t0 = [], [], time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_analyze, tasks, chunksize=8), 1):
            if r is None:
                continue
            (failed if "failed" in r else rows).append(r)
            if i % 500 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    rows.sort(key=lambda r: (r["cell_type"], r["file"]))
    if failed:
        print(f"  {len(failed)} cells could not be analyzed and are excluded from every "
              f"count below, e.g. {failed[0]['file']}: {failed[0]['failed']}", flush=True)
    uncovered_total = sum(r["n_neurite_nodes_outside_any_branch"] for r in rows)
    if uncovered_total:
        print(f"  note: {uncovered_total} neurite nodes lie outside every branch and keep "
              f"their true label in the ceilings, which are optimistic by that much",
              flush=True)

    prevalence = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cell_type"], r["source"])].append(r)
        groups[(r["cell_type"], "all")].append(r)
    for (ct, src), rs in sorted(groups.items()):
        subtrees = sum(r["n_subtrees"] for r in rs)
        branches = sum(r["n_branches"] for r in rs)
        within = sum(r["transitions_within_branch"] for r in rs)
        at_point = sum(r["transitions_at_branch_point"] for r in rs)
        prevalence[f"{ct}/{src}"] = {
            "cells": len(rs), "subtrees": subtrees, "branches": branches,
            "pct_subtrees_mixed_any": 100.0 * sum(r["subtrees_mixed_any"] for r in rs) / max(1, subtrees),
            "pct_subtrees_mixed_substantial": 100.0 * sum(r["subtrees_mixed_substantial"] for r in rs) / max(1, subtrees),
            "pct_cells_with_substantial_mixed_subtree": 100.0 * sum(r["subtrees_mixed_substantial"] > 0 for r in rs) / len(rs),
            "pct_branches_mixed": 100.0 * sum(r["branches_mixed"] for r in rs) / max(1, branches),
            "transitions_within_branch": within, "transitions_at_branch_point": at_point,
            "pct_transitions_within_branch": 100.0 * within / max(1, within + at_point),
            **{f"cells_{k}": sum(bool(r[k]) for r in rs) for k in CATEGORIES},
        }

    ceilings = {}
    for ct in ("pyramidal", "interneuron"):
        rs = [r for r in rows if r["cell_type"] == ct]
        if not rs:
            continue
        ceilings[ct] = {
            "subtree": {**_f1_from_cm(sum(np.asarray(r["cm_subtree"]) for r in rs), NEURITES[ct]),
                        "per_cell": _dist(r["f1_oracle_subtree"] for r in rs)},
            "branch": {**_f1_from_cm(sum(np.asarray(r["cm_branch"]) for r in rs), NEURITES[ct]),
                       "per_cell": _dist(r["f1_oracle_branch"] for r in rs)},
        }

    gtt, dep = _per_cell_scores()
    by_file = {r["file"]: r for r in rows}
    stratified = {}
    for ct in ("pyramidal", "interneuron"):
        for cat in ("all",) + CATEGORIES:
            g_vals, d_vals = [], []
            for s in SEEDS:
                for (seed, f), v in gtt.items():
                    if seed != s or f not in by_file or by_file[f]["cell_type"] != ct:
                        continue
                    if cat == "all" or by_file[f][cat]:
                        g_vals.append(v)
                        if (s, f) in dep:
                            d_vals.append(dep[(s, f)])
            if g_vals:
                stratified[f"{ct}/{cat}"] = {"gt_cell_type": _dist(g_vals), "deployment": _dist(d_vals)}

    L = ["MIXED-LABEL STRUCTURE -- ground truth over the corpus", "",
         f"{'group':<28} {'cells':>6} {'subtrees':>9} {'%mixed':>7} {'%subst':>7} {'%cellsSub':>9} "
         f"{'%brMixed':>8} {'%trans in-branch':>16}"]
    for k, v in prevalence.items():
        L.append(f"{k:<28} {v['cells']:6d} {v['subtrees']:9d} {v['pct_subtrees_mixed_any']:7.2f} "
                 f"{v['pct_subtrees_mixed_substantial']:7.2f} {v['pct_cells_with_substantial_mixed_subtree']:9.2f} "
                 f"{v['pct_branches_mixed']:8.2f} {v['pct_transitions_within_branch']:16.1f}")
    L += ["", "Category counts (cells):"]
    for k, v in prevalence.items():
        if k.endswith("/all"):
            L.append(f"  {k:<18} " + "  ".join(f"{c} {v['cells_' + c]}" for c in CATEGORIES))
    L += ["", "Oracle ceilings (majority ground-truth label per unit):"]
    for ct, v in ceilings.items():
        for unit in ("subtree", "branch"):
            u = v[unit]
            cls = "  ".join(f"{c} {u[c]:.4f}" for c in ("axon", "basal/dendrite", "apical") if c in u)
            L.append(f"  {ct:<12} per {unit:<8} macro-F1 {u['neurite_macro_f1']:.4f}  {cls}  "
                     f"per-cell mean {u['per_cell']['mean']:.4f} P10 {u['per_cell']['p10']:.4f}")
    L += ["", "Held-out per-cell F1 by category (seed-file pairs over seeds 42, 123, 789):",
          f"  {'group':<34} {'n':>6} {'GT mean':>8} {'GT P10':>7} {'GT %bad':>7} | {'dep mean':>8} {'dep P10':>7} {'dep %bad':>8}"]
    for k, v in stratified.items():
        g, d = v["gt_cell_type"], v["deployment"]
        dm = f"{d['mean']:8.4f} {d['p10']:7.4f} {d['pct_bad']:8.2f}" if d.get("n") else f"{'--':>8} {'--':>7} {'--':>8}"
        L.append(f"  {k:<34} {g['n']:6d} {g['mean']:8.4f} {g['p10']:7.4f} {g['pct_bad']:7.2f} | {dm}")
    text = "\n".join(L)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_limit{args.limit}" if args.limit else ""     # a smoke run keeps its own files
    (args.out_dir / f"mixed_label_structure{tag}.json").write_text(json.dumps(
        {"n_cells": len(rows), "substantial_min_nodes": SUBSTANTIAL_MIN_NODES,
         "substantial_min_frac": SUBSTANTIAL_MIN_FRAC, "axon_in_dendrite_min": AXON_IN_DENDRITE_MIN,
         "apical_trunk_min": APICAL_TRUNK_MIN, "prevalence": prevalence, "ceilings": ceilings,
         "stratified": stratified}, indent=2), encoding="utf-8")
    fields = [k for k in rows[0] if not k.startswith("cm_")] if rows else []
    with (args.out_dir / f"mixed_label_percell{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / f"mixed_label_structure{tag}.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
