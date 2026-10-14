"""Subtree-owner map helpers used by the inference pipeline.

These derive per-primary-subtree owner probabilities from the Stage 2
subtree-owner model and locate the most apical-looking primary subtree.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..features import SWCNode
from .subtree_features import extract_primary_subtrees

#: SWC type code for apical dendrite, the class ``best_apical_owner`` looks for.
APICAL = 4


def predict_subtree_owner_map(
    nodes: list[SWCNode],
    subtree_owner_model: Any,
) -> dict[int, dict[str, float | int]]:
    """Per-primary-subtree owner probabilities, keyed by subtree root index.

    Each entry holds ``prob_{label}`` for every owner class the model predicts,
    plus ``pred`` (argmax label) and ``conf``. Empty only when the cell has no
    primary subtree. The model is required: the caller checks that its bundle
    covers this cell type, because only the caller can say which bundle it was.
    """
    subtrees = extract_primary_subtrees(nodes)
    if not subtrees:
        return {}

    X = np.stack([sub.features for sub in subtrees])
    probs = subtree_owner_model.predict_proba(X)
    classes = [int(c) for c in subtree_owner_model.classes_]
    out: dict[int, dict[str, float | int]] = {}
    for sub, row in zip(subtrees, probs):
        info: dict[str, float | int] = {
            f"prob_{cls}": float(prob) for cls, prob in zip(classes, row)
        }
        best = int(np.argmax(row))
        info["pred"] = classes[best]
        info["conf"] = float(row[best])
        out[sub.root_idx] = info
    return out


def best_apical_owner(
    subtree_owner_map: dict[int, dict[str, float | int]],
    threshold: float = 0.45,
) -> int | None:
    """Root index of the primary subtree most likely to own the apical.

    Returns None if the top apical probability is below ``threshold``.
    """
    best_root: int | None = None
    best_prob = 0.0
    for root_idx, info in subtree_owner_map.items():
        prob = float(info.get(f"prob_{APICAL}", 0.0))
        if prob > best_prob:
            best_prob = prob
            best_root = root_idx
    if best_root is None or best_prob < threshold:
        return None
    return best_root
