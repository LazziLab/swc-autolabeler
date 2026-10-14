"""Stage 3 branch correction: a class-weighted gradient-boosted refinement of
apical/basal/axon labels for pyramidal cells.

After the subtree-owner model (Stage 2) and the topology rules have labeled a
pyramidal cell, this head re-decides axon/basal/apical for each branch from the
label-free branch features and overrides the current label only where it is
confident (>= override_tau) and disagrees. It is conservative by design: the
confidence threshold and the proximal scope together leave most branches alone.
The ones it does change are those the earlier stages most often get wrong,
apical trunks swallowed by basal and axon confused with dendrite near the soma,
which is why it lifts per-cell neurite-F1 more than its size suggests.

The correction is proximal by design: a bundle's ``scope_root_path_max_um``
limits both training and overriding to branches whose midpoint lies within that
path distance of the soma (see ``DEFAULT_SCOPE_UM`` in
train_branch_correction.py). Distal branches keep the Stage 2 subtree decision,
which uses whole-subtree context that local branch geometry cannot match.

The correction is a 3-class XGBoost on ``pipeline.branch_features`` features,
trained with inverse-frequency class weights (see train_branch_correction.py).
Pyramidal cells only; interneurons are never touched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib

from ..features import SWCNode
from ..branch_features import MorphologyBranches, extract_branches

# 3-class model index -> SWC neurite type.
CLASS_TO_TYPE = {0: 2, 1: 3, 2: 4}   # axon, basal, apical
DEFAULT_OVERRIDE_TAU = 0.65


def load_correction(path: str | Path) -> dict:
    """Load a saved correction bundle.

    Bundle keys: ``model`` (fitted XGBClassifier), ``feature_names``,
    ``class_map`` ({0:2,1:3,2:4}), ``override_tau``, and optionally
    ``scope_root_path_max_um`` (see :func:`scope_mask`).
    """
    return joblib.load(Path(path))


def scope_mask(correction: dict, feats: np.ndarray) -> np.ndarray:
    """Boolean mask of the branches the correction may override.

    Every branch when the bundle carries no ``scope_root_path_max_um``; otherwise
    the branches whose ``root_path_dist`` feature is at most that many microns.
    """
    scope = correction.get("scope_root_path_max_um")
    if scope is None:
        return np.ones(len(feats), dtype=bool)
    names = list(correction.get("feature_names") or [])
    if "root_path_dist" not in names:
        raise ValueError("a scoped correction bundle must carry feature_names with root_path_dist")
    return feats[:, names.index("root_path_dist")] <= float(scope)


def apply_correction(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    correction: dict,
    morph: MorphologyBranches | None = None,
) -> tuple[list[int], list[float]]:
    """Return (labels, confidences) with the branch correction applied.

    For each branch, if the correction's max class probability clears
    ``override_tau`` and its predicted neurite class differs from the branch's
    current label, the whole branch is relabeled and its confidence raised to the
    correction's. A branch's current label is read from its first node: Stage 2
    labels whole subtrees, so a branch is normally uniform, and where the rules
    have left it mixed the first node stands for it. Non-neurite branches
    (current label not in {axon, basal, apical}) are left untouched. A bundle may
    carry ``scope_root_path_max_um``, in which case only branches whose midpoint
    lies within that path distance of the soma are scored; the rest keep their
    labels. The input lists are copied, not mutated.

    ``morph`` is the caller's already-extracted branches. Branch extraction is
    the most expensive step in labeling a cell, and the pipeline has done it
    before reaching here, so passing it avoids doing the identical work twice.
    Omit it and this extracts its own.
    """
    class_map = {int(k): int(v) for k, v in correction.get("class_map", CLASS_TO_TYPE).items()}
    tau = float(correction.get("override_tau", DEFAULT_OVERRIDE_TAU))
    if morph is None:
        morph = extract_branches(nodes, cell_type="pyramidal")
    if not morph.branches:
        return list(labels), list(confidences)

    feats = np.stack([br.features for br in morph.branches], axis=0)
    in_scope = scope_mask(correction, feats)
    scored = np.flatnonzero(in_scope)
    if not len(scored):
        return list(labels), list(confidences)

    # Only the branches the scope admits are scored; the rest keep their labels
    # whatever the model would have said, so predicting for them is wasted work.
    proba = correction["model"].predict_proba(feats[scored])
    pred_cls = proba.argmax(axis=1)
    pred_conf = proba.max(axis=1)

    out_labels = list(labels)
    out_conf = list(confidences)
    for k, bi in enumerate(scored):
        br = morph.branches[bi]
        current = out_labels[br.node_indices[0]]
        new_label = class_map[int(pred_cls[k])]
        if pred_conf[k] >= tau and new_label != current and current in (2, 3, 4):
            for ni in br.node_indices:
                out_labels[ni] = new_label
                out_conf[ni] = float(pred_conf[k])
    return out_labels, out_conf
