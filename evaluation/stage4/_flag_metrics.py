"""Shared helpers for the Stage 4 flag-model evaluation drivers.

Three concerns live here:

- Ground-truth per-cell scoring (``per_cell_neurite_f1``, ``_per_class_f1``,
  ``per_cell_per_class_f1``, ``_structural_features``). These use the
  ground-truth neurite labels and are the *labels* the flag model is trained to
  predict; they are never used as model features. Every column
  ``_structural_features`` produces is listed in
  ``pipeline.stage4.flag_model.GT_OR_LABEL_COLUMNS``, and ``_engineer_features``
  raises if one reaches a feature list, so the rule is enforced rather than
  merely stated.
- ``check_model_seed``, which refuses a model directory trained on a different
  split than the one being scored. This is the guard that stops a driver from
  testing a model on cells it trained on.
- Re-exports, so the evaluation drivers take every scoring helper from this one
  module: ``in_test`` from ``pipeline.split``, and ``_compute_metrics`` and
  ``per_cell_neurite_f1`` from ``pipeline.stage4.flag_metrics``.

Absent values follow one convention: a quantity that does not apply to a cell is
``None`` where the result is a mapping a caller filters, and ``float("nan")``
where it is a fixed-key record a caller averages over.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from pipeline.features import SWCNode
from pipeline.stage1.cell_type_detector import CELL_TYPE_LABEL_SETS
# The per-cell scoring lives in pipeline/ so the flag trainer and this eval share
# one implementation. These names and in_test are re-exported here for the
# evaluation drivers that import them from this module.
from pipeline.stage4.flag_metrics import (  # noqa: F401
    _compute_metrics, per_cell_neurite_f1,
)
from pipeline.split import in_test  # noqa: F401


def check_model_seed(model_dir, seed: int, test_size: float | None = None,
                     strict: bool = False) -> bool:
    """Refuse a model directory trained on a different split than the one scored.

    Every seed holds out a different fifth of the corpus, so a model scored under
    another seed is tested mostly on cells it was trained on, and nothing in the
    numbers would show it. The training seed is read from the manifest written
    next to the models.

    Returns True when a manifest was found and the pairing checked. A directory
    without one, a model set not built by ``pipeline.train_pipeline``, cannot be
    checked: that is reported rather than passed over in silence, since a caller
    that believes it is guarded and is not is the worse failure.

    With ``strict``, an uncheckable directory stops the run instead of warning.
    The warning alone is a printed line that scrolls past in a long run, and no
    caller reads the return value, so a caller that must be sure passes strict.
    """
    for name in ("train_manifest.json", "curator_manifest.json"):
        path = Path(model_dir) / name
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        trained = manifest.get("seed")
        if trained is not None and int(trained) != int(seed):
            raise SystemExit(
                f"{model_dir} was trained with seed {trained} but --seed is {seed}: its "
                f"held-out cells differ, so this would score cells the model trained on.")
        held = manifest.get("test_size")
        if test_size is not None and held is not None and abs(float(held) - float(test_size)) > 1e-9:
            raise SystemExit(
                f"{model_dir} was trained with test_size {held} but --test-size is {test_size}.")
        return True
    message = (f"no training manifest in {model_dir}, so its split cannot be "
               f"checked against --seed {seed}. If it was not trained on this "
               f"seed's split, the cells scored below include cells it trained on.")
    if strict:
        raise SystemExit(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    print(f"warning: {message}", flush=True)
    return False


def per_cell_per_class_f1(gt: list[int], pred: list[int], cell_type: str) -> dict[str, float | None]:
    """Per-class F1 for this single cell, keyed by class name.

    The value is None when that class is absent from the cell's ground truth, so
    callers can exclude Nones from cross-cell distributions (an apical-F1
    distribution should be over only cells that actually have apical traced). The
    class set is the cell type's valid label set.

    ``_per_class_f1`` answers a similar question with fixed keys and marks an
    absent class with NaN instead, because its result is averaged rather than
    filtered. The two are not interchangeable: see its note on one-vs-rest.
    """
    valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
    m = _compute_metrics(gt, pred, valid)
    out: dict[str, float | None] = {}
    for name, info in m.get("per_label", {}).items():
        out[name] = float(info["f1"]) if info["support"] > 0 else None
    return out


def _per_class_f1(gt: list[int], pred: list[int]) -> dict[str, float]:
    """One-vs-rest per-class F1 for axon/basal/apical, for classes present in GT.

    Note this is NOT the per-class F1 in ``_compute_metrics`` above, which is
    multi-class: there, a node predicted apical when the truth is basal is a
    false positive for apical and a false negative for basal. Here each class is
    scored against every other pooled as one negative, which gives a different
    number for the same cell. The two feed different reporting columns and are
    never compared with each other.

    Class IDs: 1=soma, 2=axon, 3=basal, 4=apical. These are reporting columns
    (used for failure-mode breakdowns), never model features.

    An absent class is NaN here, where ``per_cell_per_class_f1`` uses None: this
    result has fixed keys and is averaged, so NaN propagates the absence, while
    that one is a mapping a caller filters.
    """
    out = {"axon_f1": float("nan"), "basal_f1": float("nan"), "apical_f1": float("nan")}
    if not gt:
        return out
    y_t = np.asarray(gt, dtype=int)
    y_p = np.asarray(pred, dtype=int)
    for cls, name in ((2, "axon_f1"), (3, "basal_f1"), (4, "apical_f1")):
        if (y_t == cls).any():
            f1 = f1_score((y_t == cls).astype(int), (y_p == cls).astype(int), zero_division=0)
            out[name] = float(f1)
    return out


def _structural_features(nodes: list[SWCNode]) -> dict[str, float]:
    """Lightweight ground-truth structural fingerprints (for failure analysis).

    Connected components come from the parent-link forest; the z statistics use
    the ground-truth node types. These are reporting columns, never features.
    """
    out: dict[str, float] = {}
    n = len(nodes)
    if n == 0:
        return out

    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    parent_idx = [-1] * n
    children: list[list[int]] = [[] for _ in nodes]
    for i, nd in enumerate(nodes):
        if nd.parent != -1 and nd.parent in id_to_idx:
            p = id_to_idx[nd.parent]
            parent_idx[i] = p
            children[p].append(i)

    visited = [False] * n
    n_comp = 0
    for start in range(n):
        if visited[start]:
            continue
        n_comp += 1
        stack = [start]
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            p = parent_idx[i]
            if p >= 0 and not visited[p]:
                stack.append(p)
            for j in children[i]:
                if not visited[j]:
                    stack.append(j)
    out["n_components"] = float(n_comp)

    # Rotation-invariant apical descriptors. The corpus is not aligned to a common
    # axis -- the apical runs along y in most cells and along z in about a tenth --
    # so an apical is described by how far it reaches from the soma and by whether
    # it lies opposite the basal, not by where it sits in z.
    soma_xyz = np.array([[nd.x, nd.y, nd.z] for nd in nodes if nd.type == 1], dtype=float)
    apical_xyz = np.array([[nd.x, nd.y, nd.z] for nd in nodes if nd.type == 4], dtype=float)
    basal_xyz = np.array([[nd.x, nd.y, nd.z] for nd in nodes if nd.type == 3], dtype=float)
    # Two situations are kept apart: a cell with no apical, and a cell with an
    # apical but no soma-typed node, whose reach cannot be measured at all. Given
    # the same value, the second kind would escape the truncated-apical rule in
    # analyze_failure_modes, which tests 0 < reach < 100.
    if len(soma_xyz) and len(apical_xyz):
        centre = soma_xyz.mean(axis=0)
        out["apical_reach_um"] = float(np.linalg.norm(apical_xyz - centre, axis=1).max())
        if len(basal_xyz):
            va = apical_xyz.mean(axis=0) - centre
            vb = basal_xyz.mean(axis=0) - centre
            na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            # Apical and basal normally leave the soma in opposite directions, so
            # this sits near -1; above 0 they share a direction and the labeler has
            # no geometric separation to work with.
            out["apical_basal_cos"] = float(va @ vb / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")
        else:
            out["apical_basal_cos"] = float("nan")
    elif not len(apical_xyz):
        out["apical_reach_um"] = 0.0          # no apical: the reach really is zero
        out["apical_basal_cos"] = float("nan")
    else:
        out["apical_reach_um"] = float("nan")  # apical but no soma: not measurable
        out["apical_basal_cos"] = float("nan")

    soma_zs = [nd.z for nd in nodes if nd.type == 1]
    axon_zs = [nd.z for nd in nodes if nd.type == 2]
    basal_zs = [nd.z for nd in nodes if nd.type == 3]
    apical_zs = [nd.z for nd in nodes if nd.type == 4]
    out["soma_z"] = float(np.mean(soma_zs)) if soma_zs else float("nan")
    out["axon_frac"] = (len(axon_zs) / n) if n else 0.0
    out["apical_frac"] = (len(apical_zs) / n) if n else 0.0
    out["basal_frac"] = (len(basal_zs) / n) if n else 0.0
    if apical_zs:
        out["apical_mean_z"] = float(np.mean(apical_zs))
        out["apical_z_extent"] = float(max(apical_zs) - min(apical_zs))
        if soma_zs:
            out["apical_above_soma"] = out["apical_mean_z"] - out["soma_z"]
    else:
        out["apical_mean_z"] = float("nan")
        out["apical_z_extent"] = 0.0
        out["apical_above_soma"] = float("nan")
    return out
