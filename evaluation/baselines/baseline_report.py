"""Metric reporting for the baseline comparison.

Runs a baseline predictor over the test cells (``_aggregate_per_file``) and
builds the canonical comprehensive metric report (``build_full_report``) in the
exact shape the pipeline evaluation emits, so baseline and pipeline numbers are
directly comparable. The print helpers render the console summary tables.

Every metric here is over ground-truth labels (the report is what we compare);
the shared per-cell F1 definitions come from the evaluation metrics helper.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from pipeline.features import parse_swc
from pipeline.stage1.cell_type_detector import CELL_TYPE_LABEL_SETS
from evaluation.stage4._flag_metrics import _compute_metrics

CLASS_NAMES = ("soma", "axon", "basal/dendrite", "apical")
CLASS_IDS = (1, 2, 3, 4)


def _aggregate_per_file(test_files: dict[str, list], predict_fn) -> tuple[list[dict], list[int], list[int]]:
    """Apply ``predict_fn(nodes, cell_type) -> list[int]`` to every test file.

    Returns (cell_records, all_gt, all_pred), where each cell record holds
    {path, cell_type, n_nodes, gt, pred} in the shape ``build_full_report``
    expects.
    """
    cell_records: list[dict] = []
    all_gt: list[int] = []
    all_pred: list[int] = []
    skipped: list[str] = []
    for ct, files in test_files.items():
        for f in files:
            nodes = parse_swc(f)
            if not nodes:
                skipped.append(f.name)
                continue
            gt = [nd.type for nd in nodes]
            pred = predict_fn(nodes, ct)
            # Raised, not asserted: this is the one check that a predictor
            # returned a usable labelling, and -O would remove an assert.
            if len(pred) != len(gt):
                raise SystemExit(f"{f.name}: predictor returned {len(pred)} labels "
                                 f"for {len(gt)} nodes")
            cell_records.append({
                "path": str(f),
                "cell_type": ct,
                "n_nodes": len(nodes),
                "gt": gt,
                "pred": pred,
            })
            all_gt.extend(gt)
            all_pred.extend(pred)
    if skipped:
        print(f"  warning: {len(skipped)} test cells could not be parsed and are "
              f"excluded from every count, e.g. {skipped[0]}", flush=True)
    return cell_records, all_gt, all_pred


def _dist(values: Iterable[float | None]) -> dict:
    """Distribution summary; Nones are dropped (signal: class absent in cell)."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0,
                "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0, "min": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
    }


def _per_class_dict_from_metrics(metrics: dict) -> dict:
    """Per-class precision/recall/F1/support from ``_compute_metrics`` output."""
    out = {}
    for name in CLASS_NAMES:
        info = metrics.get("per_label", {}).get(name, {})
        out[name] = {
            "precision": float(info.get("precision", 0.0)),
            "recall": float(info.get("recall", 0.0)),
            "f1": float(info.get("f1", 0.0)),
            "support": int(info.get("support", 0)),
        }
    return out


def _confusion_dict_from_metrics(metrics: dict) -> dict:
    """Normalize the confusion matrix to {gt_name: {pred_name: count}}, zeros filled."""
    raw = metrics.get("confusion", {})
    out = {g: {p: 0 for p in CLASS_NAMES} for g in CLASS_NAMES}
    for g_name, row in raw.items():
        for p_name, n in row.items():
            if g_name in out and p_name in out[g_name]:
                out[g_name][p_name] = int(n)
    return out


def build_full_report(
    method: str,
    gt_pool: list[int],
    pred_pool: list[int],
    cell_records: list[dict],
    inference_min: float | None = None,
) -> dict:
    """Build the canonical comprehensive metric report for one method.

    ``gt_pool`` / ``pred_pool`` are all test nodes concatenated (corpus-level
    metrics); ``cell_records`` is one {cell_type, n_nodes, gt, pred} dict per
    cell (per-cell distributions and the per-cell-type breakdown). The output
    layout matches the pipeline evaluation exactly for direct comparison.
    """
    n_cells = len(cell_records)
    n_nodes = len(gt_pool)

    # The pooled block scores against every class the corpus can contain, the
    # union over cell types, while each per-cell-type block below uses that
    # type's own set. Macro averages run over the classes present in the ground
    # truth either way, so a pooled run of interneurons alone still reports no
    # apical; the union only admits apical where pyramidal cells supply it.
    corpus_metrics = _compute_metrics(gt_pool, pred_pool, set(CLASS_IDS))
    corpus_block = {
        "accuracy": float(corpus_metrics.get("accuracy", 0.0)),
        "macro_f1": float(corpus_metrics.get("macro_f1", 0.0)),
        "neurite_macro_f1": float(corpus_metrics.get("neurite_macro_f1", 0.0)),
        "per_class": _per_class_dict_from_metrics(corpus_metrics),
        "confusion": _confusion_dict_from_metrics(corpus_metrics),
    }

    pc_acc_vals: list[float] = []
    pc_neuritef1_vals: list[float] = []
    pc_per_class_vals = {n: [] for n in CLASS_NAMES}
    for rec in cell_records:
        if rec["n_nodes"] > 0:
            n_correct = sum(1 for g, p in zip(rec["gt"], rec["pred"]) if g == p)
            pc_acc_vals.append(n_correct / rec["n_nodes"])
        else:
            pc_acc_vals.append(0.0)
        # One confusion per cell, read twice: see the per-cell-type block below.
        cell_m = _compute_metrics(rec["gt"], rec["pred"],
                                  set(CELL_TYPE_LABEL_SETS.get(rec["cell_type"], {1, 2, 3})))
        pc_neuritef1_vals.append(float(cell_m.get("neurite_macro_f1", 0.0)))
        for cname in CLASS_NAMES:
            info = cell_m.get("per_label", {}).get(cname)
            pc_per_class_vals[cname].append(
                float(info["f1"]) if info and info["support"] > 0 else None)

    per_cell_block = {
        "accuracy": _dist(pc_acc_vals),
        "neurite_macro_f1": _dist(pc_neuritef1_vals),
        "per_class_f1": {n: _dist(pc_per_class_vals[n]) for n in CLASS_NAMES},
    }

    by_ct: dict[str, dict] = {}
    for ct in ("pyramidal", "interneuron"):
        ct_records = [r for r in cell_records if r["cell_type"] == ct]
        if not ct_records:
            continue
        ct_gt_pool: list[int] = []
        ct_pred_pool: list[int] = []
        for r in ct_records:
            ct_gt_pool.extend(r["gt"])
            ct_pred_pool.extend(r["pred"])
        valid = set(CELL_TYPE_LABEL_SETS.get(ct, {1, 2, 3}))
        ct_corpus = _compute_metrics(ct_gt_pool, ct_pred_pool, valid)

        ct_pc_acc, ct_pc_neuritef1 = [], []
        ct_per_class_vals = {n: [] for n in CLASS_NAMES}
        for rec in ct_records:
            if rec["n_nodes"] > 0:
                ct_pc_acc.append(
                    sum(1 for g, p in zip(rec["gt"], rec["pred"]) if g == p) / rec["n_nodes"]
                )
            else:
                ct_pc_acc.append(0.0)
            # One pass over the cell for both: per_cell_neurite_f1 and
            # per_cell_per_class_f1 each build the same confusion internally, so
            # calling both scored every cell twice.
            cell_m = _compute_metrics(rec["gt"], rec["pred"],
                                      set(CELL_TYPE_LABEL_SETS.get(rec["cell_type"], {1, 2, 3})))
            ct_pc_neuritef1.append(float(cell_m.get("neurite_macro_f1", 0.0)))
            for cname in CLASS_NAMES:
                info = cell_m.get("per_label", {}).get(cname)
                ct_per_class_vals[cname].append(
                    float(info["f1"]) if info and info["support"] > 0 else None)

        by_ct[ct] = {
            "n_cells": len(ct_records),
            "n_nodes": len(ct_gt_pool),
            "corpus": {
                "accuracy": float(ct_corpus.get("accuracy", 0.0)),
                "macro_f1": float(ct_corpus.get("macro_f1", 0.0)),
                "neurite_macro_f1": float(ct_corpus.get("neurite_macro_f1", 0.0)),
                "per_class": _per_class_dict_from_metrics(ct_corpus),
                "confusion": _confusion_dict_from_metrics(ct_corpus),
            },
            "per_cell": {
                "accuracy": _dist(ct_pc_acc),
                "neurite_macro_f1": _dist(ct_pc_neuritef1),
                "per_class_f1": {n: _dist(ct_per_class_vals[n]) for n in CLASS_NAMES},
            },
        }

    return {
        "method": method,
        "n_test_cells": n_cells,
        "n_test_nodes": n_nodes,
        "inference_min": inference_min,
        "convention": "v11 (macro F1 over GT-present classes only)",
        "corpus": corpus_block,
        "per_cell": per_cell_block,
        "by_cell_type": by_ct,
    }


def print_summary_table(reports: list[dict]) -> None:
    """One-line-per-method side-by-side summary."""
    cols = [
        ("method", 24, "{:<24}"),
        ("n_test", 8, "{:>8}"),
        ("acc", 8, "{:>8.4f}"),
        ("neurite_F1", 11, "{:>11.4f}"),
        ("axon_F1", 9, "{:>9.4f}"),
        ("basal_F1", 10, "{:>10.4f}"),
        ("apical_F1", 11, "{:>11.4f}"),
        ("pc_acc_mean", 12, "{:>12.4f}"),
        ("pc_acc_p10", 12, "{:>12.4f}"),
        ("pc_F1_mean", 11, "{:>11.4f}"),
        ("pc_F1_p10", 10, "{:>10.4f}"),
        ("pc_F1_p25", 10, "{:>10.4f}"),
    ]
    hdr = "  " + " ".join(f"{n:>{w}}" if i else f"{n:<{w}}"
                          for i, (n, w, _) in enumerate(cols))
    print(hdr)
    for r in reports:
        row = [
            r["method"], r["n_test_cells"],
            r["corpus"]["accuracy"], r["corpus"]["neurite_macro_f1"],
            r["corpus"]["per_class"]["axon"]["f1"],
            r["corpus"]["per_class"]["basal/dendrite"]["f1"],
            r["corpus"]["per_class"]["apical"]["f1"],
            r["per_cell"]["accuracy"]["mean"],
            r["per_cell"]["accuracy"]["p10"],
            r["per_cell"]["neurite_macro_f1"]["mean"],
            r["per_cell"]["neurite_macro_f1"]["p10"],
            r["per_cell"]["neurite_macro_f1"]["p25"],
        ]
        line = "  " + " ".join(c[2].format(v) for c, v in zip(cols, row))
        print(line)


def print_per_class_per_cell(reports: list[dict]) -> None:
    """Per-class per-cell F1 distributions for each method."""
    for r in reports:
        print(f"\n  {r['method']}:")
        print(f"    {'class':<16} {'n_cells':>8} {'mean':>8} {'median':>8} "
              f"{'p10':>8} {'p25':>8} {'p90':>8}")
        for cname in CLASS_NAMES:
            d = r["per_cell"]["per_class_f1"][cname]
            print(f"    {cname:<16} {d['n']:>8} {d['mean']:>8.4f} {d['median']:>8.4f} "
                  f"{d['p10']:>8.4f} {d['p25']:>8.4f} {d['p90']:>8.4f}")


def print_per_cell_type(reports: list[dict]) -> None:
    """Per-cell-type breakdown for each method."""
    for r in reports:
        print(f"\n  ### {r['method']} ###")
        for ct, info in r.get("by_cell_type", {}).items():
            c = info["corpus"]
            pc = info["per_cell"]
            print(f"  {ct}  (n_cells={info['n_cells']}, n_nodes={info['n_nodes']})")
            print(f"    corpus  acc={c['accuracy']:.4f}  neurite_F1={c['neurite_macro_f1']:.4f}  "
                  f"axon={c['per_class']['axon']['f1']:.4f}  "
                  f"basal={c['per_class']['basal/dendrite']['f1']:.4f}  "
                  f"apical={c['per_class']['apical']['f1']:.4f}")
            print(f"    per-cell  acc mean={pc['accuracy']['mean']:.4f} p10={pc['accuracy']['p10']:.4f}  "
                  f"F1 mean={pc['neurite_macro_f1']['mean']:.4f} p10={pc['neurite_macro_f1']['p10']:.4f}")
            cm = c["confusion"]
            print("    confusion (GT\\pred):")
            print(f"      {'':<14} {'soma':>10} {'axon':>10} {'basal':>10} {'apical':>10}")
            for g_name in CLASS_NAMES:
                row = cm.get(g_name, {})
                cells = [f"{row.get(p_name, 0):>10d}" for p_name in CLASS_NAMES]
                print(f"      {g_name:<14} {' '.join(cells)}")
