"""Deployment-safe per-cell features for the flag model.

The flag model scores each already-labeled cell for rejection. This module
turns one cell's pipeline output (predicted node labels, per-node confidences,
and raw coordinates) into the per-cell feature row the model consumes. Every
feature here is deployment-safe: it uses only predictions, confidences, and raw
SWC geometry, never the ground-truth neurite labels.

Three groups of features are produced:
  - confidence statistics (``_confidence_summary``): per-cell and per-class
    confidence mean/percentiles and low-confidence fractions;
  - branch confidence (``_branch_summary``): segments the reconstruction into
    linear runs between bifurcations and label changes, then summarizes the
    per-branch confidence;
  - predicted geometry (``_predicted_geometry``): spatial extents of each
    predicted neurite class relative to the soma.

The branch segmentation and confidence aggregation (``_segment_branches``,
``summarize_confidence``) turn the pipeline's per-node confidences into
per-branch summaries.

``assemble_flag_features`` adds the pyramidal-anatomy block from
``flag_geom_features`` under a ``geom_`` prefix. The two modules are disjoint by
construction: confidence statistics live here and anatomy lives there, so no
quantity reaches the model as two identical columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..features import SWCNode
from .flag_geom_features import extract_features as _extract_geom

# Per-node confidence below which a node is counted as "uncertain". Every
# low-confidence feature in this module is cut at this value.
NODE_LOW_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Per-branch + per-cell confidence summary
# ---------------------------------------------------------------------------
@dataclass
class BranchConfidence:
    """Confidence summary for one branch.

    A 'branch' here is the contiguous run of nodes between branch points or
    between a branch point and a tip. Branches are segmented by walking parent
    links and breaking at any node that has != 1 child or whose label differs
    from its parent's.
    """

    n_nodes: int
    mean_confidence: float


def _segment_branches(nodes: list[SWCNode], labels: list[int]) -> list[list[int]]:
    """Return a list of branches, each a list of node *indices* into ``nodes``.

    A branch breaks at:
      - any bifurcation (parent has > 1 child), OR
      - any label change (the label of a node differs from its parent's), OR
      - the root.

    Each segment is homogeneous in label and bounded by structural features,
    matching how a human reviewer would scan the reconstruction.
    """
    n = len(nodes)
    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    children: list[list[int]] = [[] for _ in range(n)]
    parent_idx: list[int] = [-1] * n
    for i, nd in enumerate(nodes):
        if nd.parent != -1 and nd.parent in id_to_idx:
            p = id_to_idx[nd.parent]
            parent_idx[i] = p
            children[p].append(i)

    # A node starts a NEW branch if it is a root, its parent has >1 child
    # (a bifurcation), or its label differs from its parent's label.
    starts = [False] * n
    for i in range(n):
        p = parent_idx[i]
        if p < 0:
            starts[i] = True
        elif len(children[p]) > 1:
            starts[i] = True
        elif labels[i] != labels[p]:
            starts[i] = True

    # Walk parent-first (BFS from roots) so each node's branch id is resolved
    # after its parent's.
    branch_id_of: list[int] = [-1] * n
    order: list[int] = []
    seen = [False] * n
    queue = [i for i in range(n) if parent_idx[i] < 0]
    while queue:
        nq = []
        for i in queue:
            if seen[i]:
                continue
            seen[i] = True
            order.append(i)
            nq.extend(children[i])
        queue = nq

    for i in order:
        if starts[i]:
            branch_id_of[i] = i
        else:
            branch_id_of[i] = branch_id_of[parent_idx[i]]

    by_branch: dict[int, list[int]] = {}
    for i in range(n):
        by_branch.setdefault(branch_id_of[i], []).append(i)
    # Sort by smallest node index so the ordering is stable across runs.
    return [by_branch[k] for k in sorted(by_branch)]


def summarize_confidence(
    nodes: list[SWCNode],
    labels: Iterable[int],
    confidences: Iterable[float],
) -> list[BranchConfidence]:
    """Per-branch confidence summaries for one cell.

    Args:
        nodes: list of SWCNode (parsed from the input file)
        labels: per-node predicted labels (same length as nodes)
        confidences: per-node ML confidence (0..1), same length as nodes

    Returns:
        list[BranchConfidence], one entry per segmented branch
    """
    labels = list(labels)
    confidences = [float(c) for c in confidences]
    assert len(labels) == len(nodes), "labels length != nodes length"
    assert len(confidences) == len(nodes), "confidences length != nodes length"

    if not confidences:
        return []

    branches: list[BranchConfidence] = []
    for node_idxs in _segment_branches(nodes, labels):
        if not node_idxs:
            continue
        branch_confs = [confidences[i] for i in node_idxs]
        branches.append(BranchConfidence(
            n_nodes=len(node_idxs),
            mean_confidence=sum(branch_confs) / len(branch_confs),
        ))

    return branches


# ---------------------------------------------------------------------------
# Per-cell feature builders (deployment-safe)
# ---------------------------------------------------------------------------
def _safe_percentile(values: list[float], q: float, default: float = 0.0) -> float:
    return float(np.percentile(values, q)) if values else default


def _confidence_summary(labels: list[int], confs: list[float]) -> dict[str, float]:
    """Per-cell and per-class confidence statistics."""
    arr = np.asarray(confs, dtype=float)
    out: dict[str, float] = {
        "conf_mean": float(arr.mean()) if arr.size else 0.0,
        "conf_std": float(arr.std()) if arr.size else 0.0,
        "conf_min": float(arr.min()) if arr.size else 0.0,
        "conf_p05": float(np.percentile(arr, 5)) if arr.size else 0.0,
        "conf_p10": float(np.percentile(arr, 10)) if arr.size else 0.0,
        "conf_p25": float(np.percentile(arr, 25)) if arr.size else 0.0,
        "conf_median": float(np.median(arr)) if arr.size else 0.0,
    }
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        key = str(thr).replace(".", "")
        out[f"conf_frac_lt_{key}"] = float((arr < thr).mean()) if arr.size else 0.0

    label_names = {2: "axon", 3: "basal", 4: "apical"}
    lab_arr = np.asarray(labels, dtype=int)
    for label, name in label_names.items():
        vals = arr[lab_arr == label]
        out[f"{name}_conf_mean"] = float(vals.mean()) if vals.size else 0.0
        out[f"{name}_conf_p10"] = float(np.percentile(vals, 10)) if vals.size else 0.0
        out[f"{name}_conf_frac_lt_07"] = float((vals < NODE_LOW_THRESHOLD).mean()) if vals.size else 0.0
    return out


def _branch_summary(
    nodes: list[SWCNode],
    labels: list[int],
    confs: list[float],
) -> dict[str, float]:
    """Summarize the per-branch confidence distribution."""
    branches = summarize_confidence(nodes, labels, confs)
    means = [float(b.mean_confidence) for b in branches]
    n_nodes = [int(b.n_nodes) for b in branches]
    # No branch_conf_min here: branches partition every node, so the smallest
    # per-branch minimum is just the cell-wide conf_min already in the row.
    return {
        "n_pred_branches": len(branches),
        "branch_conf_mean": float(np.mean(means)) if means else 0.0,
        "branch_conf_p10": _safe_percentile(means, 10),
        "branch_frac_conf_lt_07": float(np.mean([m < NODE_LOW_THRESHOLD for m in means])) if means else 0.0,
        "branch_frac_conf_lt_08": float(np.mean([m < 0.8 for m in means])) if means else 0.0,
        "branch_n_nodes_p90": _safe_percentile(n_nodes, 90),
    }


def _predicted_geometry(nodes: list[SWCNode], labels: list[int]) -> dict[str, float]:
    """Spatial extents of each predicted neurite class relative to the soma."""
    label_names = {2: "axon", 3: "basal", 4: "apical"}
    labels_arr = np.asarray(labels, dtype=int)
    # Predicted soma only. The file's own type column is ground truth at flag
    # training time and must not be read here; a cell with no predicted soma
    # has none in the file either, so the reference z stays 0.0 as before.
    soma_zs = [nd.z for nd, lab in zip(nodes, labels_arr) if lab == 1]
    soma_z = float(np.mean(soma_zs)) if soma_zs else 0.0

    out: dict[str, float] = {"pred_soma_z": soma_z}
    for label, name in label_names.items():
        xs = [nd.x for nd, lab in zip(nodes, labels_arr) if lab == label]
        ys = [nd.y for nd, lab in zip(nodes, labels_arr) if lab == label]
        zs = [nd.z for nd, lab in zip(nodes, labels_arr) if lab == label]
        if zs:
            out[f"pred_{name}_z_extent"] = float(max(zs) - min(zs))
            out[f"pred_{name}_z_mean_rel_soma"] = float(np.mean(zs) - soma_z)
            out[f"pred_{name}_z_std"] = float(np.std(zs))
            out[f"pred_{name}_x_extent"] = float(max(xs) - min(xs))
            out[f"pred_{name}_y_extent"] = float(max(ys) - min(ys))
        else:
            out[f"pred_{name}_z_extent"] = 0.0
            out[f"pred_{name}_z_mean_rel_soma"] = 0.0
            out[f"pred_{name}_z_std"] = 0.0
            out[f"pred_{name}_x_extent"] = 0.0
            out[f"pred_{name}_y_extent"] = 0.0
    return out


def assemble_flag_features(
    nodes: list[SWCNode],
    labels: list[int],
    confs: list[float],
    stage1_ct: str,
    stage1_conf: float,
) -> dict[str, float]:
    """The full deployment-safe flag feature vector for one labeled cell.

    Combines the confidence summary, branch summary, predicted geometry, and the
    geometry/anatomy block. Called identically at flag-train time and at
    inference so the feature columns line up exactly.

    The ``geom_`` block exists only for cells Stage 1 routes to pyramidal, since
    it encodes pyramidal anatomy; for other cells those columns are missing and
    the flag imputes them.

    There is deliberately no morphometric-baseline disagreement. Measured against
    this feature set it contributes 0.0000 to kept-P10 while being by far the most
    expensive block to produce (it needs a separately trained L-Measure model), so
    it is not part of the flag.
    """
    feats = {
        **_confidence_summary(labels, confs),
        **_branch_summary(nodes, labels, confs),
        **_predicted_geometry(nodes, labels),
    }
    if stage1_ct == "pyramidal":
        geom = _extract_geom(
            labels, confs,
            [n.x for n in nodes], [n.y for n in nodes], [n.z for n in nodes],
            [n.radius for n in nodes], [n.id for n in nodes], [n.parent for n in nodes],
            stage1_ct, stage1_conf,
        )
        feats.update({f"geom_{k}": v for k, v in geom.items()})
    return feats


def _round_row(row: dict) -> dict:
    """Format float values to compact strings for CSV output."""
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            out[k] = f"{v:.6g}"
        else:
            out[k] = v
    return out
