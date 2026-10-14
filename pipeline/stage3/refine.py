"""Stage 3: Topology-aware refinement.

Applies structural constraints and label propagation to clean up
Stage 2 branch classifications. This is purely rule-based and uses
the cell type from Stage 1 to select which constraints apply.

Pyramidal cells get a confidence-weighted majority label per soma-child
subtree, then the anatomical constraints of one axon and one apical trunk.
Interneurons, whose axon often leaves a dendritic subtree rather than the soma,
get softer subtree-level rules that never overwrite a confident prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import SWCNode, euclidean, select_proxy_root, tree_by_index
from ..stage1.cell_type_detector import CellTypeResult


@dataclass
class RefinedLabel:
    """Per-node label assignment after refinement."""
    node_id: int
    label: int
    confidence: float           # ML confidence for this label
    was_refined: bool           # True if changed by Stage 3


@dataclass
class RefinementResult:
    """Output of Stage 3."""
    labels: list[RefinedLabel]  # one per node
    n_refined: int              # how many labels were changed
    subtree_labels: dict[int, int]  # soma_child_id -> assigned label
    cell_type: str


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _subtree_indices(root: int, children: list[list[int]]) -> list[int]:
    """Get all node indices in the subtree rooted at root."""
    result: list[int] = []
    stack = [root]
    while stack:
        idx = stack.pop()
        result.append(idx)
        stack.extend(children[idx])
    return result


def _path_length_from(
    start: int,
    nodes: list[SWCNode],
    children: list[list[int]],
) -> dict[int, float]:
    """BFS path length from start to all descendants."""
    dist: dict[int, float] = {start: 0.0}
    queue = [start]
    while queue:
        idx = queue.pop(0)
        for ci in children[idx]:
            d = dist[idx] + euclidean(nodes[idx], nodes[ci])
            dist[ci] = d
            queue.append(ci)
    return dist


# ---------------------------------------------------------------------------
# Refinement strategies
# ---------------------------------------------------------------------------

def _primary_subtree_voting(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    parent_idx: list[int | None],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
) -> tuple[list[int], dict[int, int]]:
    """Assign each primary subtree a single label via confidence-weighted voting.

    For each soma child, collect all non-soma nodes in its subtree,
    compute confidence-weighted label votes, and assign the winner
    to the entire subtree.
    """
    out = list(labels)
    subtree_labels: dict[int, int] = {}

    # Find primary children of soma
    primary_roots: list[int] = []
    for si in soma_indices:
        for ci in children[si]:
            if ci not in soma_indices:
                primary_roots.append(ci)

    neurite_labels = label_set - {1}
    if not neurite_labels:
        return out, subtree_labels

    for pr in primary_roots:
        subtree = _subtree_indices(pr, children)
        if not subtree:
            continue

        # Confidence-weighted voting (exclude soma nodes)
        votes: dict[int, float] = {lbl: 0.0 for lbl in neurite_labels}
        for idx in subtree:
            lbl = labels[idx]
            if lbl in neurite_labels:
                votes[lbl] += confidences[idx]

        if not any(v > 0 for v in votes.values()):
            continue

        winner = max(neurite_labels, key=lambda lbl: votes.get(lbl, 0.0))
        subtree_labels[pr] = winner

        # Apply to all nodes in subtree
        for idx in subtree:
            if labels[idx] in neurite_labels:
                out[idx] = winner

    return out, subtree_labels


def _pick_apical_by_principal_axis(
    subtree_labels: dict[int, int],
    nodes: list[SWCNode],
    children: list[list[int]],
    soma_indices: set[int],
    labels: list[int],
    apical_label: int = 4,
    fallback_label: int = 3,
) -> tuple[dict[int, int], list[int]]:
    """Enforce ≤1 apical primary subtree, choosing the winner by the cell's
    principal axis (PC1 of all neurite nodes) rather than by confidence sum.

    Rationale: in cells with rotated coordinate frames or sideways apicals,
    the model can confidently mislabel the wrong subtree as apical. PC1
    captures the cell's actual elongation direction, so the subtree whose
    mean position projects highest along PC1 is the true apical trunk.

    Falls back to a no-op if PC1 is degenerate or only one apical subtree
    exists.
    """
    out_labels = list(labels)
    out_subtrees = dict(subtree_labels)

    apical_owners = [pr for pr, lbl in subtree_labels.items() if lbl == apical_label]
    if len(apical_owners) <= 1:
        return out_subtrees, out_labels

    # Soma center (proxy)
    if soma_indices:
        sx = float(np.mean([nodes[i].x for i in soma_indices]))
        sy = float(np.mean([nodes[i].y for i in soma_indices]))
        sz = float(np.mean([nodes[i].z for i in soma_indices]))
    else:
        sx, sy, sz = nodes[0].x, nodes[0].y, nodes[0].z

    # PC1 of neurite cloud, signed away from soma
    n = len(nodes)
    neurite_idx = [i for i in range(n) if i not in soma_indices]
    if len(neurite_idx) < 5:
        return out_subtrees, out_labels
    pts = np.array(
        [[nodes[i].x - sx, nodes[i].y - sy, nodes[i].z - sz] for i in neurite_idx],
        dtype=np.float64,
    )
    try:
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order_eig = np.argsort(eigvals)[::-1]
        pc1 = eigvecs[:, order_eig[0]]
        if float(np.dot(pc1, pts.mean(axis=0))) < 0.0:
            pc1 = -pc1
    except np.linalg.LinAlgError:
        return out_subtrees, out_labels

    # Score each apical-labeled primary by mean PC1 projection of its subtree
    best_pr = None
    best_score = -float("inf")
    for pr in apical_owners:
        sub = _subtree_indices(pr, children)
        if not sub:
            continue
        projs = [
            (nodes[i].x - sx) * pc1[0]
            + (nodes[i].y - sy) * pc1[1]
            + (nodes[i].z - sz) * pc1[2]
            for i in sub
        ]
        score = float(np.mean(projs))
        if score > best_score:
            best_score = score
            best_pr = pr

    if best_pr is None:
        return out_subtrees, out_labels

    # Reassign losers to fallback (basal)
    for pr in apical_owners:
        if pr == best_pr:
            continue
        out_subtrees[pr] = fallback_label
        sub = _subtree_indices(pr, children)
        for idx in sub:
            if out_labels[idx] == apical_label:
                out_labels[idx] = fallback_label

    return out_subtrees, out_labels


def _enforce_single_class(
    subtree_labels: dict[int, int],
    nodes: list[SWCNode],
    children: list[list[int]],
    labels: list[int],
    confidences: list[float],
    target_class: int,
    fallback_class: int,
) -> tuple[dict[int, int], list[int]]:
    """Enforce that at most one primary subtree has target_class.

    The subtree with highest total confidence for target_class wins.
    Others are reassigned to fallback_class.
    """
    out_labels = list(labels)
    out_subtrees = dict(subtree_labels)

    owners = [pr for pr, lbl in subtree_labels.items() if lbl == target_class]
    if len(owners) <= 1:
        return out_subtrees, out_labels

    # Pick the best one by total confidence for this class
    best_pr = None
    best_score = -1.0
    for pr in owners:
        subtree = _subtree_indices(pr, children)
        score = sum(confidences[idx] for idx in subtree if labels[idx] == target_class)
        if score > best_score:
            best_score = score
            best_pr = pr

    # Reassign losers
    for pr in owners:
        if pr == best_pr:
            continue
        out_subtrees[pr] = fallback_class
        subtree = _subtree_indices(pr, children)
        for idx in subtree:
            if out_labels[idx] == target_class:
                out_labels[idx] = fallback_class

    return out_subtrees, out_labels


def _constrain_apical_to_owner_subtree(
    labels: list[int],
    confidences: list[float],
    children: list[list[int]],
    soma_indices: set[int],
    apical_owner_root: int | None,
    apical_label: int = 4,
    fallback_label: int = 3,
    preserve_confidence: float = 0.93,
) -> list[int]:
    """Mostly constrain apical labels to the preferred apical subtree.

    Nodes outside the preferred subtree keep apical only if the branch
    classifier was extremely confident. This keeps the constraint soft
    enough to avoid obvious false negatives while still cleaning up most
    stray apical islands in basal subtrees.
    """
    if apical_owner_root is None:
        return labels

    owner_nodes = set(_subtree_indices(apical_owner_root, children))
    out = list(labels)
    for idx, lbl in enumerate(labels):
        if idx in soma_indices:
            continue
        if lbl == apical_label and idx not in owner_nodes and confidences[idx] < preserve_confidence:
            out[idx] = fallback_label
    return out


def _soft_subtree_majority(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
    high_conf: float = 0.70,
    low_conf: float = 0.55,
    min_margin: float = 0.60,
) -> list[int]:
    """Soft subtree-majority propagation (interneuron-safe).

    Unlike pyramidal "hard" subtree voting (which forces every node in a
    primary subtree to one label), interneurons allow axons to emerge from
    within dendritic subtrees — so we cannot simply majority-vote. Instead:

    1. For each primary subtree, compute confidence-weighted votes using
       ONLY nodes with ml_confidence ≥ high_conf.
    2. If the winning label owns ≥ min_margin of the high-confidence mass,
       propagate it ONLY to low-confidence (< low_conf) nodes in the subtree.
    3. High-confidence nodes are never touched.

    This is a conservative "fill in the unsure parts" operation.

    Thresholds default to high_conf=0.70, low_conf=0.55, min_margin=0.60.
    """
    neurite_labels = sorted(label_set - {1})
    if len(neurite_labels) <= 1:
        return labels

    out = list(labels)
    primary_roots: list[int] = []
    for si in soma_indices:
        for ci in children[si]:
            if ci not in soma_indices:
                primary_roots.append(ci)

    for pr in primary_roots:
        subtree = _subtree_indices(pr, children)
        if len(subtree) < 4:
            continue  # too small to vote

        # Confidence-weighted votes over HIGH-confidence nodes only
        hi_votes: dict[int, float] = {lbl: 0.0 for lbl in neurite_labels}
        hi_total = 0.0
        for idx in subtree:
            if confidences[idx] < high_conf:
                continue
            lbl = out[idx]
            if lbl in hi_votes:
                hi_votes[lbl] += confidences[idx]
                hi_total += confidences[idx]

        if hi_total <= 0:
            continue

        winner = max(neurite_labels, key=lambda lbl: hi_votes.get(lbl, 0.0))
        winner_share = hi_votes[winner] / hi_total

        if winner_share < min_margin:
            continue  # subtree is genuinely mixed — leave it alone

        # Flip low-confidence nodes in this subtree to the winner
        for idx in subtree:
            if confidences[idx] < low_conf and out[idx] != winner and out[idx] in neurite_labels:
                out[idx] = winner

    return out


def _interneuron_axon_bias(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
    low_conf: float = 0.60,
    min_subtree_nodes: int = 20,
    thin_radius_margin: float = 0.25,
) -> list[int]:
    """Bias the thinnest-longest primary subtree toward axon (interneuron).

    Interneuron axons are characterized by a distinctive geometric signature:
    thin (small radius) and long (high path length). This rule identifies
    the primary subtree most axon-like by that signature and gently flips
    low-confidence non-axon nodes in it toward axon (2).

    It does NOT touch high-confidence nodes, so if the ML model is certain
    a thin branch is dendritic, the rule defers.
    """
    if 2 not in label_set:
        return labels

    out = list(labels)
    primary_roots: list[int] = []
    for si in soma_indices:
        for ci in children[si]:
            if ci not in soma_indices:
                primary_roots.append(ci)

    if len(primary_roots) < 2:
        return out  # nothing to compare against

    # Score each primary by an axon-likeness metric: low mean radius + high
    # max path length. Normalize both within this cell so the rule is
    # scale-invariant.
    primary_stats: list[tuple[int, float, float, int]] = []  # (pr, mean_r, max_path, size)
    for pr in primary_roots:
        subtree = _subtree_indices(pr, children)
        if len(subtree) < min_subtree_nodes:
            continue
        radii = [nodes[i].radius for i in subtree]
        if not radii:
            continue
        mean_r = float(np.mean(radii))
        # Path length from pr to the deepest node
        dist = _path_length_from(pr, nodes, children)
        max_path = max(dist.values()) if dist else 0.0
        primary_stats.append((pr, mean_r, max_path, len(subtree)))

    if len(primary_stats) < 2:
        return out

    min_r = min(s[1] for s in primary_stats)
    max_r = max(s[1] for s in primary_stats)
    max_path = max(s[2] for s in primary_stats)
    if max_r - min_r < 1e-9 or max_path < 1e-9:
        return out

    # axon-score = 0.6 * (1 - normalized_radius) + 0.4 * normalized_path
    best_pr = None
    best_score = -1.0
    for pr, mean_r, path, _size in primary_stats:
        r_score = 1.0 - (mean_r - min_r) / (max_r - min_r)
        p_score = path / max_path
        score = 0.6 * r_score + 0.4 * p_score
        if score > best_score:
            best_score = score
            best_pr = pr

    # Only act if the candidate is notably thinner than average — otherwise
    # there isn't a clear axon-like primary to bias toward.
    avg_r = float(np.mean([s[1] for s in primary_stats]))
    best_mean_r = next(s[1] for s in primary_stats if s[0] == best_pr)
    if best_mean_r > avg_r * (1.0 - thin_radius_margin):
        return out  # not distinctly thin → don't force

    # Bias low-confidence non-axon nodes in the winning subtree toward axon
    axon_subtree = _subtree_indices(best_pr, children)
    for idx in axon_subtree:
        if idx in soma_indices:
            continue
        if out[idx] != 2 and confidences[idx] < low_conf and out[idx] != 1:
            out[idx] = 2

    return out


#: The individually skippable Stage 3 rules, in application order. Passing
#: names from this tuple to ``refine(disable=...)`` omits those rules and
#: applies the rest, which is how each rule's contribution is measured.
RULES = (
    "subtree_voting",         # pyramidal: one label per primary subtree
    "single_axon",            # pyramidal: at most one axon subtree
    "apical_principal_axis",  # pyramidal: pick apical by the cell's own axis
    "apical_owner",           # pyramidal: apical confined to its owner subtree
    "axon_bias",              # interneuron: thin, long primary leans axonal
    "soft_majority",          # interneuron: confident subtree majority to unsure nodes
)


def refine(
    nodes: list[SWCNode],
    ml_labels: list[int],
    ml_confidences: list[float],
    cell_type_result: CellTypeResult,
    apical_owner_root: int | None = None,
    disable: frozenset[str] | set[str] = frozenset(),
) -> RefinementResult:
    """Apply Stage 3 topology-aware refinement.

    Args:
        nodes: parsed SWC nodes
        ml_labels: per-node labels from Stage 2 (1=soma, 2=axon, 3=basal, 4=apical)
        ml_confidences: per-node confidence from Stage 2 ML model
        cell_type_result: output from Stage 1
        disable: names from ``RULES`` to skip; the default applies every rule

    Returns:
        RefinementResult with refined labels.
    """
    n = len(nodes)
    if n == 0:
        return RefinementResult([], 0, {}, cell_type_result.cell_type)

    id_to_idx, parent_idx, children, roots = tree_by_index(nodes)
    label_set = cell_type_result.label_set
    cell_type = cell_type_result.cell_type

    # Label-free proxy soma/root anchor
    soma_indices = {select_proxy_root(nodes, children, roots)}

    # Start with ML labels, set soma nodes to 1
    labels = list(ml_labels)
    for si in soma_indices:
        labels[si] = 1

    # Mask out labels not in the label set
    neurite_labels = sorted(label_set - {1})
    for i in range(n):
        if labels[i] != 1 and labels[i] not in neurite_labels:
            # Map to closest valid label
            if neurite_labels:
                labels[i] = neurite_labels[0]  # default to first neurite type

    subtree_labels: dict[int, int] = {}

    # If the label set has one neurite class or fewer there is nothing to
    # flip between, so skip refinement and return early (still enforcing the
    # non-soma-label-1 guard).
    if len(neurite_labels) <= 1:
        refined_labels_early = [
            RefinedLabel(
                node_id=nodes[i].id,
                label=labels[i],
                confidence=ml_confidences[i],
                was_refined=labels[i] != ml_labels[i],
            )
            for i in range(n)
        ]
        return RefinementResult(
            labels=refined_labels_early,
            n_refined=sum(1 for i in range(n) if labels[i] != ml_labels[i]),
            subtree_labels={},
            cell_type=cell_type,
        )

    if cell_type == "interneuron":
        # Interneuron: no HARD subtree voting (axon can emerge from within
        # dendrite subtrees), but soft rules help close the Stage-3 gap.
        #
        # Rule 1 — thin-primary axon bias: identify the primary subtree
        # most likely to be the axon (thinnest mean radius + longest path)
        # and gently flip low-confidence non-axon nodes in it toward axon.
        if "axon_bias" not in disable:
            labels = _interneuron_axon_bias(
                nodes, labels, ml_confidences, children,
                soma_indices, label_set,
                low_conf=0.60,
            )

        # Rule 2 — soft subtree majority: for each primary subtree, if
        # high-confidence nodes strongly agree on one label, propagate that
        # label only to LOW-confidence nodes in the same subtree. High
        # confidence predictions are preserved.
        if "soft_majority" not in disable:
            labels = _soft_subtree_majority(
                nodes, labels, ml_confidences, children,
                soma_indices, label_set,
                high_conf=0.70, low_conf=0.55, min_margin=0.60,
            )

    elif cell_type == "pyramidal":
        # 1. Primary subtree voting. Each subtree's winner is recorded whether or
        # not the rule is on, because the two constraints below act on those
        # winners; switching the rule off only leaves the nodes as Stage 2
        # labeled them, so every rule can be measured on its own.
        voted, subtree_labels = _primary_subtree_voting(
            nodes, labels, ml_confidences, parent_idx, children,
            soma_indices, label_set,
        )
        if "subtree_voting" not in disable:
            labels = voted

        # 2. Single-axon constraint
        if 2 in label_set and "single_axon" not in disable:
            subtree_labels, labels = _enforce_single_class(
                subtree_labels, nodes, children, labels, ml_confidences,
                target_class=2, fallback_class=3,
            )

        # 3. Single-apical constraint: when several primary subtrees are apical,
        # keep the one lying furthest along the cell's own principal axis, which
        # holds in rotated coordinate frames too.
        if 4 in label_set:
            if "apical_principal_axis" not in disable:
                subtree_labels, labels = _pick_apical_by_principal_axis(
                    subtree_labels, nodes, children, soma_indices, labels,
                    apical_label=4, fallback_label=3,
                )

        # 4. Apical confined to the subtree Stage 2 rated most apical.
        if "apical_owner" not in disable:
            labels = _constrain_apical_to_owner_subtree(
                labels, ml_confidences, children, soma_indices, apical_owner_root,
            )


    else:
        # Stage 1 emits only the two known cell types, so this branch is
        # defensive. An unrecognized type gets no whole-cell refinement, since
        # every rule here encodes an assumption about one of the two.
        pass

    n_refined = sum(1 for i in range(n) if labels[i] != ml_labels[i])

    # Build result
    refined_labels = []
    for i in range(n):
        refined_labels.append(RefinedLabel(
            node_id=nodes[i].id,
            label=labels[i],
            confidence=ml_confidences[i],
            was_refined=labels[i] != ml_labels[i],
        ))

    return RefinementResult(
        labels=refined_labels,
        n_refined=n_refined,
        subtree_labels=subtree_labels,
        cell_type=cell_type,
    )
