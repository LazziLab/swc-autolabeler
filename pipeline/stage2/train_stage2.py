"""Train the Stage 2 subtree-owner classifier (one model per cell type).

Stage 2 types each primary subtree -- a whole branch-tree hanging off the
soma -- as a unit. One feature vector per subtree (path length, polar angle,
node count, depth, ...) is classified into the cell type's neurite labels, and
at inference that label propagates to every branch in the subtree. Whole-subtree
features carry stronger signal for the hard axon-vs-apical distinction than
per-branch features do.

One model per cell type, trained on that cell type's cells only:
    pyramidal   -> {axon, basal, apical}
    interneuron -> {axon, dendrite}

Call ``train_stage2(train_files, model_path)`` with a seed's train split;
``pipeline.train_pipeline`` is the runnable driver that chains all stages.

Only the subtree descriptors are needed here, so each training file is parsed
once and handed straight to ``extract_primary_subtrees``. Branch extraction,
which Stage 3 needs, plays no part in fitting this stage.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..features import parse_swc
from .subtree_features import extract_primary_subtrees

# Neurite labels each cell type's subtree-owner model predicts (soma excluded).
VALID_LABELS = {
    "pyramidal": {2, 3, 4},
    "interneuron": {2, 3},
}

def _build_subtree_pipeline(seed: int, device: str = "cuda"):
    """Standardize + soft-voting RF/GB ensemble (XGBoost drop-ins, fitted on ``device``)."""
    from sklearn.ensemble import VotingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # XGBoost drop-ins with the sklearn originals' constructor signatures.
    from .._xgb_classifiers import (
        XGBRandomForestClassifier as RandomForestClassifier,
        XGBGradientBoostingClassifier as GradientBoostingClassifier,
    )

    rf = RandomForestClassifier(
        n_estimators=250,
        min_child_weight=2,
        random_state=seed,
        n_jobs=-1,
        device=device,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        min_child_weight=3,
        random_state=seed,
        device=device,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ensemble),
    ])


def _node_balanced_weights(
    y: np.ndarray, w: np.ndarray, power: float = 1.0,
) -> np.ndarray:
    """Combine node-count weights with node-level class balancing.

    effective_w = node_count x class_balance_factor

    With ``power=1.0`` the class factor equalizes total per-class weight: each
    class contributes the same aggregate mass to the loss. With ``power > 1``
    the inverse-frequency weighting is amplified, over-correcting toward
    minority classes -- useful when the model still over-predicts the majority
    class after equal-mass balancing (common for axon-vs-dendrite).
    """
    class_totals: dict[int, float] = {}
    for lbl, weight in zip(y.tolist(), w.tolist()):
        class_totals[lbl] = class_totals.get(lbl, 0.0) + weight
    n_classes = len(class_totals)
    total = sum(class_totals.values())
    if n_classes == 0 or total <= 0:
        return w
    class_factor = {
        c: (total / (n_classes * t)) ** power
        for c, t in class_totals.items()
    }
    return np.array([weight * class_factor[lbl] for lbl, weight in zip(y.tolist(), w.tolist())])


def _train_subtree_owner_for_cell_type(
    train_paths: list[Path],
    cell_type: str,
    valid_labels: set[int],
    seed: int,
    class_balance_power: float = 1.0,
    device: str = "cuda",
) -> Any | None:
    """Train the subtree-owner classifier for one cell type.

    Target = majority node-weighted label of each primary subtree, restricted
    to ``valid_labels``. At inference the predicted owner is propagated to every
    branch in the subtree and broadcast as a subtree-level signal into Stage 3.
    Returns None if the cell type has fewer than two owner classes to learn.
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    w_list: list[float] = []
    unreadable = 0
    for path in train_paths:
        try:
            nodes = parse_swc(path)
        except Exception:
            unreadable += 1
            continue
        if not nodes:
            unreadable += 1
            continue
        for sub in extract_primary_subtrees(nodes):
            if sub.gt_label not in valid_labels:
                continue
            X_list.append(sub.features)
            y_list.append(sub.gt_label)
            w_list.append(float(sum(sub.gt_label_counts.values())))

    if unreadable:
        print(f"    {cell_type}: {unreadable} of {len(train_paths)} cells could not be "
              f"read and were left out", flush=True)
    if not X_list or len(set(y_list)) < 2:
        return None

    X = np.stack(X_list)
    y = np.array(y_list)
    w = np.array(w_list)
    sw = _node_balanced_weights(y, w, class_balance_power)
    model = _build_subtree_pipeline(seed, device)
    model.fit(X, y, clf__sample_weight=sw)
    return model


def train_stage2(
    train_files: dict[str, list[Path]],
    model_path: Path,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Train Stage 2 on a seed's train split and pickle the model bundle.

    Trains one subtree-owner model per cell type and writes them to
    ``model_path`` (the deployed ``branch_classifier.pkl``). ``seed`` fixes the
    classifier ``random_state``, and no caller passes one, so it is 42 for every
    released model: across the seed ensemble only the training split varies.
    Each subtree is weighted by its node count times a class factor that gives
    every class the same total weight (``class_balance_power`` 1.0 in
    ``_train_subtree_owner_for_cell_type``). ``device`` is where the fits run:
    ``cuda``, as for the released models, or ``cpu`` without an NVIDIA GPU.
    """
    # One subtree-owner model per cell type, each trained on its own cells.
    subtree_owner_models_by_cell_type: dict[str, Any] = {}
    for ct, valid_labels in VALID_LABELS.items():
        model = _train_subtree_owner_for_cell_type(
            train_files.get(ct, []), ct, valid_labels, seed, device=device
        )
        if model is not None:
            subtree_owner_models_by_cell_type[ct] = model

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "subtree_owner_models_by_cell_type": subtree_owner_models_by_cell_type,
        }, f)
