"""Per-cell ground-truth scoring for the Stage 4 quality flag.

``per_cell_neurite_f1`` is the flag's target quantity -- the per-cell neurite
macro-F1 (soma excluded), averaged over the neurite classes present in the cell's
ground truth, matching the full-pipeline evaluation. It is a *label*, never a
model feature. This lives in ``pipeline/`` so both the flag trainer
(``pipeline.stage4.single_seed_flag``) and the paper's flag eval
(``evaluation.stage4``) share one implementation.
"""
from __future__ import annotations

import numpy as np

from pipeline.stage1.cell_type_detector import CELL_TYPE_LABEL_SETS

LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal/dendrite", 4: "apical"}


def _compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    valid_labels: set[int],
) -> dict:
    """Per-node accuracy, macro-F1, balanced accuracy, and per-class stats.

    Macro-averaged metrics give each class equal weight, and the neurite
    variants exclude soma (label 1). Macro averages are taken over classes
    PRESENT in the ground truth so that absent classes do not force 0.0 into
    the mean.
    """
    pairs = [(gt, pred) for gt, pred in zip(y_true, y_pred) if gt in valid_labels]
    if not pairs:
        return {
            "accuracy": 0.0, "n": 0,
            "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
            "balanced_accuracy": 0.0,
            "neurite_macro_f1": 0.0, "neurite_balanced_accuracy": 0.0,
            "per_label": {}, "confusion": {},
        }

    n = len(pairs)
    correct = sum(1 for g, p in pairs if g == p)
    accuracy = correct / n

    all_labels = sorted(valid_labels)
    per_label: dict[str, dict] = {}
    label_f1: dict[int, float] = {}
    label_prec: dict[int, float] = {}
    label_rec: dict[int, float] = {}
    label_support: dict[int, int] = {}
    for lbl in all_labels:
        tp = sum(1 for g, p in pairs if g == lbl and p == lbl)
        fp = sum(1 for g, p in pairs if g != lbl and p == lbl)
        fn = sum(1 for g, p in pairs if g == lbl and p != lbl)
        support = sum(1 for g, _ in pairs if g == lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        label_f1[lbl] = f1
        label_prec[lbl] = prec
        label_rec[lbl] = rec
        label_support[lbl] = support
        per_label[LABEL_NAMES.get(lbl, str(lbl))] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    present = [lbl for lbl in all_labels if label_support[lbl] > 0]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    macro_f1 = _mean([label_f1[l] for l in present])
    macro_prec = _mean([label_prec[l] for l in present])
    macro_rec = _mean([label_rec[l] for l in present])
    balanced_acc = macro_rec  # balanced accuracy == mean per-class recall

    neurite_present = [l for l in present if l != 1]
    neurite_macro_f1 = _mean([label_f1[l] for l in neurite_present])
    neurite_balanced_acc = _mean([label_rec[l] for l in neurite_present])

    confusion: dict[str, dict[str, int]] = {}
    for gt_l in all_labels:
        gt_name = LABEL_NAMES.get(gt_l, str(gt_l))
        confusion[gt_name] = {}
        for pred_l in all_labels:
            pred_name = LABEL_NAMES.get(pred_l, str(pred_l))
            confusion[gt_name][pred_name] = sum(1 for g, p in pairs if g == gt_l and p == pred_l)

    return {
        "accuracy": round(accuracy, 4),
        "n": n,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "neurite_macro_f1": round(neurite_macro_f1, 4),
        "neurite_balanced_accuracy": round(neurite_balanced_acc, 4),
        "per_label": per_label,
        "confusion": confusion,
    }


def per_cell_neurite_f1(gt: list[int], pred: list[int], cell_type: str) -> float:
    """Per-cell neurite-macro-F1: the flag model's target quantity.

    Macro F1 averaged over the neurite classes (soma excluded) that are present
    in this cell's ground truth. Missing classes are dropped from the average
    rather than counted as F1=0, because "did the model find apical?" is
    undefined when no apical was traced. This is the same per-cell F1 the
    full-pipeline evaluation reports.

    Computed directly rather than through ``_compute_metrics``, which would also
    build per-class tables and a full confusion matrix only to discard them.
    The two agree exactly, including the rounding, on every cell of the corpus.
    """
    valid = CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3})
    g = np.asarray(gt, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    keep = np.isin(g, np.fromiter(valid, dtype=np.int64, count=len(valid)))
    g, p = g[keep], p[keep]
    if g.size == 0:
        return 0.0

    scores: list[float] = []
    for lbl in sorted(valid):
        if lbl == 1:                      # soma excluded from the neurite mean
            continue
        is_gt = g == lbl
        support = int(is_gt.sum())
        if support == 0:                  # absent class: dropped, not scored 0
            continue
        is_pred = p == lbl
        tp = int((is_gt & is_pred).sum())
        fp = int(is_pred.sum()) - tp
        fn = support - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    if not scores:
        return 0.0
    return float(round(sum(scores) / len(scores), 4))
