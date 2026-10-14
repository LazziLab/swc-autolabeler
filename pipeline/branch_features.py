"""Per-branch features: the 57 descriptors the Stage 3 correction reads.

Stage 2 works at the subtree scale and has its own 24 descriptors. These are
finer: one vector per branch, read by the learned correction that closes Stage 3
and by the all-in-one model of ``pipeline.unstaged``.

Nothing here reads the SWC type column except the training target. The soma is
located structurally, through ``select_proxy_root``, so a branch's descriptors
do not depend on the labels the pipeline is about to assign.

A branch is a linear chain of nodes between bifurcations, or between a
bifurcation and a terminal, exactly as the paper defines it. A long unbranched
run is one branch however long it is, and its descriptors are averaged over the
whole of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import SWCNode, euclidean, select_proxy_root, tree_by_index

# ---------------------------------------------------------------------------
# Branch feature names (label-free — no leakage)
# ---------------------------------------------------------------------------

BRANCH_FEATURE_NAMES: list[str] = [
    # Branch geometry
    "path_length",              # total path length of this branch
    "radial_extent",            # max Euclidean distance from anchor
    "n_nodes",                  # number of nodes in this branch
    "mean_radius",              # mean node radius
    "std_radius",               # std of node radii
    "min_radius",               # minimum radius
    "max_radius",               # maximum radius
    "taper_ratio",              # terminal/proximal radius ratio
    "persistence",              # Euclidean/path ratio (straightness)
    "up_alignment",             # z-direction alignment (0=down, 1=up)
    "branchiness",              # fraction of bifurcation nodes
    "symmetry",                 # child radius symmetry at anchor

    # Position relative to soma
    "root_path_dist",           # path distance from soma to branch midpoint
    "root_radial_dist",         # Euclidean distance from soma to branch midpoint
    "z_rel_soma",               # mean z relative to soma center
    "z_span",                   # z range within this branch

    # Subtree context
    "subtree_size",             # total nodes in the subtree rooted at this branch
    "subtree_depth",            # max depth from this branch to leaf
    "subtree_max_path",         # max path length in subtree
    "branch_order",             # bifurcation count from soma to this branch

    # Parent/sibling context
    "parent_radius",            # radius at the anchor (parent) node
    "radius_ratio_to_parent",   # mean_radius / parent_radius
    "n_siblings",               # number of sibling branches at the anchor
    "is_primary",               # 1.0 if direct child of soma, else 0.0

    # Cell-level normalized features (relative to cell stats)
    "path_length_rel",          # path_length / max_path_in_cell
    "radial_extent_rel",        # radial_extent / max_radial_in_cell
    "radius_rel",               # mean_radius / cell_mean_radius
    "subtree_size_rel",         # subtree_size / total_nodes

    # Cell type encoding (from Stage 1)
    "is_pyramidal",             # 1.0 if pyramidal
    "is_interneuron",           # 1.0 if interneuron
    "is_purkinje",              # 1.0 if purkinje

    # Primary-subtree rank features (help apical vs basal: apical is the
    # single dominant z-aligned / longest subtree)
    "is_longest_subtree",       # 1.0 if this branch's primary subtree is the longest in the cell
    "subtree_path_ratio",       # this subtree's max path / max over all primary subtrees
    "subtree_z_ratio",          # this subtree's z_span / max over all primary subtrees
    "subtree_z_rank",           # normalized rank by z_span among primary subtrees (1.0=max)
    "subtree_max_radial_rank",  # normalized rank by max soma radial reach among primary subtrees

    # Axon initial segment / thin-long signatures (help axon vs dendrite
    # for interneurons where axons emerge abruptly and remain thin)
    "starts_at_soma",           # 1.0 if anchor node is a soma node
    "radius_drop_at_anchor",    # (anchor_r - first_nodes_mean_r) / anchor_r
    "thin_fraction",            # fraction of branch nodes with radius < 0.5 µm
    "min_radius_in_subtree",    # min radius across all descendants (thin chain signature)

    # Multi-scale local radius / straightness
    "proximal_mean_radius",     # mean radius over first 1/3 of branch
    "distal_mean_radius",       # mean radius over last 1/3 of branch
    "proximal_persistence",     # straightness of first 1/3
    "distal_persistence",       # straightness of last 1/3

    # --- Apical-vs-basal discrimination features ---
    # Help resolve the main remaining confusion in pyramidal cells
    # (apical points pia-ward as a tall, tight, thick trunk; basal spreads
    # horizontally around the soma).
    "polar_angle_from_up",               # angle (radians) between soma→branch-mid and +z. Apical ≈ 0, basal ≈ π/2
    "vertical_horizontal_span_ratio",    # primary-subtree z_span / max(xy_span, eps). Apical tall/thin → high
    "soma_z_offset_norm",                # (mean_z - soma_z) / cell_z_range. Apical ≈ +1, basal ≈ 0, descending axon < 0
    "primary_subtree_polar_spread",      # std of polar angles within the primary subtree. Apical tight → low

    # --- Cell-intrinsic principal-axis features ---
    # Replace world-z assumptions with the cell's OWN long axis (PC1 of the
    # neurite point cloud). Handles cells with rotated coordinate frames,
    # sideways-projecting apicals, and stunted apicals where the world z-axis
    # is not the apical axis.
    "principal_axis_projection",         # signed branch-midpoint projection onto cell PC1, normalized to [-1, +1]
    "polar_angle_from_principal_axis",   # angle (radians) between soma→branch-mid and the cell's PC1
    "principal_axis_alignment_strength", # PC1 explained variance ratio. High → cell is elongated, axis is meaningful
    "subtree_principal_projection",      # mean PC1 projection of this branch's primary subtree, normalized
    "subtree_principal_rank",            # normalized rank of subtree mean PC1 projection (1.0 = top apical candidate)

    # --- Branching-rate features ---
    # Targets axon-vs-apical confusion when both project upward. Axons have
    # long internodes and few bifurcations per micron; apicals branch
    # frequently (especially in the tuft); basals are moderately bushy.
    "bifurcations_per_micron",           # this branch's bifurcation count / path_length
    "mean_internode_distance",           # mean Euclidean distance between consecutive nodes in this branch
    "subtree_bifurcations_per_micron",   # subtree total bifurcations / subtree total path length (axon: low)

    # --- Trunk-detection features ---
    # Directly encode "apical = one dominant trunk before bifurcating; basal =
    # bushy from the start." A "trunk" here is the longest root-to-leaf path
    # within a primary subtree. Apical subtrees have a long, dominant trunk;
    # basal subtrees branch early and have no clear trunk.
    "path_to_first_bifurcation_norm",    # primary-root to first bifurcation distance / max cell path. Apical: large, basal: small
]



@dataclass
class BranchData:
    """Per-branch data: features + ground truth label."""
    branch_id: int
    node_indices: list[int]     # indices into the node list
    anchor_idx: int             # parent/anchor node index
    features: np.ndarray        # feature vector
    gt_label: int               # ground truth majority label (from SWC type column)
    gt_label_counts: dict[int, int]  # per-class node counts
    n_nodes: int
    primary_root_idx: int | None


@dataclass
class MorphologyBranches:
    """All branches extracted from one morphology."""
    file_path: str
    cell_type: str              # from Stage 1 or ground truth folder
    branches: list[BranchData]
    soma_indices: list[int]     # proxy soma/root anchor indices
    total_nodes: int


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _bfs_order(roots: list[int], children: list[list[int]], n: int) -> list[int]:
    order: list[int] = []
    visited: set[int] = set()
    queue = list(roots)
    while queue:
        idx = queue.pop(0)
        if idx in visited:
            continue
        visited.add(idx)
        order.append(idx)
        queue.extend(sorted(children[idx]))
    return order


# ---------------------------------------------------------------------------
# Branch partitioning
# ---------------------------------------------------------------------------

def _partition_branches(
    nodes: list[SWCNode],
    parent_idx: list[int | None],
    children: list[list[int]],
    soma_indices: set[int],
) -> list[tuple[int, list[int]]]:
    """Partition morphology into (anchor_idx, [node_indices]) branch segments.

    A branch is a linear chain from a bifurcation or root to the next
    bifurcation or terminal, and nothing subdivides it.
    """
    n = len(nodes)
    assigned: list[bool] = [False] * n
    branches: list[tuple[int, list[int]]] = []

    # Find starting points: soma children
    starts: list[tuple[int, int]] = []  # (anchor, start_node)
    for si in sorted(soma_indices):
        assigned[si] = True
        for ci in sorted(children[si]):
            if ci not in soma_indices:
                starts.append((si, ci))

    # If no soma, use roots
    if not starts:
        roots = [i for i, p in enumerate(parent_idx) if p is None]
        for ri in roots:
            for ci in sorted(children[ri]):
                starts.append((ri, ci))
            if not children[ri]:
                starts.append((ri, ri))

    pending = list(starts)
    while pending:
        anchor, start = pending.pop(0)
        if assigned[start]:
            continue

        # Trace linear segment
        segment: list[int] = []
        cur = start
        while not assigned[cur]:
            assigned[cur] = True
            segment.append(cur)

            kids = [c for c in children[cur] if not assigned[c]]
            if len(kids) == 1:
                cur = kids[0]
            else:
                # Terminal or bifurcation — end this segment
                for ci in sorted(kids):
                    pending.append((cur, ci))
                break

        if segment:
            branches.append((anchor, segment))

    return branches


# ---------------------------------------------------------------------------
# Per-branch feature computation
# ---------------------------------------------------------------------------

def _subtree_stats(
    root: int,
    children: list[list[int]],
    nodes: list[SWCNode],
    parent_idx: list[int | None],
) -> tuple[int, int, float]:
    """Compute (subtree_size, subtree_depth, subtree_max_path)."""
    stack: list[tuple[int, int, float]] = [(root, 0, 0.0)]
    size = 0
    max_depth = 0
    max_path = 0.0
    while stack:
        idx, depth, path = stack.pop()
        size += 1
        max_depth = max(max_depth, depth)
        max_path = max(max_path, path)
        for ci in children[idx]:
            seg_len = euclidean(nodes[idx], nodes[ci])
            stack.append((ci, depth + 1, path + seg_len))
    return size, max_depth, max_path


def extract_branches(
    nodes: list[SWCNode],
    cell_type: str,
    file_path: str = "",
) -> MorphologyBranches:
    """Extract all branch segments and their features from a morphology.

    Args:
        nodes: parsed SWC nodes
        cell_type: cell type string (from Stage 1 or ground truth)
        file_path: source file path (for tracking)

    Returns:
        MorphologyBranches with per-branch features and labels.
    """
    if not nodes:
        return MorphologyBranches(file_path, cell_type, [], [], 0)

    id_to_idx, parent_idx, children, roots = tree_by_index(nodes)
    n = len(nodes)
    order = _bfs_order(roots, children, n)

    # Label-free proxy soma/root anchor
    proxy_root = select_proxy_root(nodes, children, roots)
    soma_indices = {proxy_root}

    # Soma center
    soma_x, soma_y, soma_z = nodes[proxy_root].x, nodes[proxy_root].y, nodes[proxy_root].z

    # Path from root for all nodes
    path_from_root = [0.0] * n
    branch_order_arr = [0] * n
    for i in order:
        pidx = parent_idx[i]
        if pidx is not None:
            path_from_root[i] = path_from_root[pidx] + euclidean(nodes[pidx], nodes[i])
            branch_order_arr[i] = branch_order_arr[pidx] + (1 if len(children[pidx]) > 1 else 0)

    # Cell-level stats for normalization
    all_radii = [nd.radius for i, nd in enumerate(nodes) if i not in soma_indices]
    if not all_radii:
        all_radii = [nd.radius for nd in nodes]
    cell_mean_radius = float(np.mean(all_radii)) if all_radii else 1.0
    max_path_in_cell = max(path_from_root) if path_from_root else 1.0
    max_radial_in_cell = max(
        math.sqrt((nd.x - soma_x)**2 + (nd.y - soma_y)**2 + (nd.z - soma_z)**2)
        for nd in nodes
    ) if nodes else 1.0

    # Primary children of soma (for is_primary feature)
    primary_set: set[int] = set()
    for si in soma_indices:
        for ci in children[si]:
            if ci not in soma_indices:
                primary_set.add(ci)

    # Cell type encoding. is_purkinje is always 0: the pipeline routes cells to
    # pyramidal or interneuron and nothing passes a third type, so this feature
    # carries no variation and no model can split on it. It stays because the
    # released models were fitted on a 57-wide row, and dropping a column would
    # shift every feature after it.
    is_pyramidal = 1.0 if cell_type == "pyramidal" else 0.0
    is_interneuron = 1.0 if cell_type == "interneuron" else 0.0
    is_purkinje = 1.0 if cell_type == "purkinje" else 0.0

    # --- Primary-subtree stats (for rank features: apical vs basal) ---
    # For each primary subtree root, compute max path, z_span, min_radius,
    # and a per-node map telling which primary root each node belongs to.
    primary_root_of = [-1] * n
    subtree_max_path_by_root: dict[int, float] = {}
    subtree_z_span_by_root: dict[int, float] = {}
    subtree_min_radius_by_root: dict[int, float] = {}
    subtree_max_radial_by_root: dict[int, float] = {}
    # New (apical-vs-basal): per-primary-subtree xy-span, mean z offset,
    # polar-angle spread (std of soma→node angle from +z).
    subtree_xy_span_by_root: dict[int, float] = {}
    subtree_mean_z_offset_by_root: dict[int, float] = {}
    subtree_polar_std_by_root: dict[int, float] = {}
    for pr in primary_set:
        stack = [pr]
        sub_nodes: list[int] = []
        while stack:
            idx = stack.pop()
            sub_nodes.append(idx)
            primary_root_of[idx] = pr
            for ci in children[idx]:
                stack.append(ci)
        if not sub_nodes:
            continue
        subtree_max_path_by_root[pr] = max(path_from_root[i] for i in sub_nodes)
        z_vals = [nodes[i].z for i in sub_nodes]
        x_vals = [nodes[i].x for i in sub_nodes]
        y_vals = [nodes[i].y for i in sub_nodes]
        subtree_z_span_by_root[pr] = (max(z_vals) - min(z_vals)) if z_vals else 0.0
        xy_x_span = (max(x_vals) - min(x_vals)) if x_vals else 0.0
        xy_y_span = (max(y_vals) - min(y_vals)) if y_vals else 0.0
        subtree_xy_span_by_root[pr] = math.sqrt(xy_x_span ** 2 + xy_y_span ** 2)
        subtree_min_radius_by_root[pr] = min(nodes[i].radius for i in sub_nodes)
        subtree_max_radial_by_root[pr] = max(
            math.sqrt(
                (nodes[i].x - soma_x) ** 2 +
                (nodes[i].y - soma_y) ** 2 +
                (nodes[i].z - soma_z) ** 2
            )
            for i in sub_nodes
        )
        # Mean z offset from soma (apical positive, basal near 0, descending negative)
        subtree_mean_z_offset_by_root[pr] = float(np.mean([nodes[i].z - soma_z for i in sub_nodes]))
        # Polar angle std: angle between (node - soma) and +z axis
        polar_angles: list[float] = []
        for i in sub_nodes:
            dx = nodes[i].x - soma_x
            dy = nodes[i].y - soma_y
            dz = nodes[i].z - soma_z
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r > 1e-9:
                cos_theta = max(-1.0, min(1.0, dz / r))
                polar_angles.append(math.acos(cos_theta))
        subtree_polar_std_by_root[pr] = (
            float(np.std(polar_angles)) if len(polar_angles) > 1 else 0.0
        )

    # Cell-level z-range for normalization (used in soma_z_offset_norm)
    all_z_vals = [nd.z for nd in nodes]
    cell_z_range = (max(all_z_vals) - min(all_z_vals)) if all_z_vals else 1.0
    cell_z_range = max(cell_z_range, 1e-6)

    # --- Cell-intrinsic principal axis (PC1 of neurite point cloud) ---
    # Robust to coordinate-frame rotation: when the cell is not aligned with
    # world z, PC1 still finds the apical/elongation direction. We sign PC1
    # so its dot product with (centroid - soma) is positive — "out from soma
    # along the long axis" → toward the apical end for typical pyramidals.
    neurite_idx = [i for i in range(n) if i not in soma_indices]
    pc1_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # safe default = world z
    pc1_strength = 0.0
    max_abs_proj = 1.0
    if len(neurite_idx) >= 5:
        pts = np.array(
            [[nodes[i].x - soma_x, nodes[i].y - soma_y, nodes[i].z - soma_z]
             for i in neurite_idx],
            dtype=np.float64,
        )
        try:
            # Centered covariance — already centered at soma (close enough)
            cov = np.cov(pts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
            order_eig = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order_eig]
            eigvecs = eigvecs[:, order_eig]
            pc1 = eigvecs[:, 0]
            total_var = float(np.sum(eigvals))
            pc1_strength = (
                float(eigvals[0]) / total_var if total_var > 1e-12 else 0.0
            )
            # Sign convention: PC1 points away from soma along the dominant
            # neurite mass. Use centroid (mean of neurite cloud relative to soma).
            centroid = pts.mean(axis=0)
            if float(np.dot(pc1, centroid)) < 0.0:
                pc1 = -pc1
            pc1_vec = pc1
            # Cell-level max abs projection (for normalization).
            projs = pts @ pc1_vec
            max_abs_proj = float(np.max(np.abs(projs))) if projs.size else 1.0
            max_abs_proj = max(max_abs_proj, 1e-6)
        except np.linalg.LinAlgError:
            pass

    # Per-primary-subtree mean PC1 projection (raw, unnormalized) and rank.
    subtree_principal_proj_by_root: dict[int, float] = {}
    for pr in primary_set:
        # Re-walk subtree to gather projections (we already walked it above
        # but didn't store positions — keep this localized to avoid coupling).
        stack = [pr]
        projs: list[float] = []
        while stack:
            idx = stack.pop()
            dx = nodes[idx].x - soma_x
            dy = nodes[idx].y - soma_y
            dz = nodes[idx].z - soma_z
            projs.append(dx * pc1_vec[0] + dy * pc1_vec[1] + dz * pc1_vec[2])
            for ci in children[idx]:
                stack.append(ci)
        if projs:
            subtree_principal_proj_by_root[pr] = float(np.mean(projs))

    # Normalized rank by mean principal projection (1.0 = top apical candidate)
    principal_rank_of_root: dict[int, float] = {}
    sorted_roots_by_proj = sorted(
        subtree_principal_proj_by_root.items(), key=lambda kv: kv[1], reverse=True
    )
    n_pp = len(sorted_roots_by_proj)
    for rank, (pr, _) in enumerate(sorted_roots_by_proj):
        principal_rank_of_root[pr] = (
            1.0 - (rank / max(1, n_pp - 1)) if n_pp > 1 else 1.0
        )

    # --- Per-primary-subtree bifurcations / micron (axon-discriminator) ---
    # Axons branch sparsely → low bif rate. Apicals + basals branch densely.
    subtree_bif_density_by_root: dict[int, float] = {}
    for pr in primary_set:
        stack = [pr]
        n_bif = 0
        total_path = 0.0
        while stack:
            idx = stack.pop()
            n_kids = len(children[idx])
            if n_kids > 1:
                n_bif += 1
            for ci in children[idx]:
                total_path += euclidean(nodes[idx], nodes[ci])
                stack.append(ci)
        subtree_bif_density_by_root[pr] = (
            n_bif / total_path if total_path > 1e-9 else 0.0
        )

    # --- Per-primary-subtree trunk-detection stats ---
    # Trunk = longest root-to-leaf path inside the subtree. Apical subtrees
    # have one dominant trunk before tufting; basal subtrees bifurcate early
    # and have no clear trunk dominance.
    subtree_first_bif_by_root: dict[int, float] = {}

    for pr in primary_set:
        # Distance from the primary root to its first bifurcation: walk while
        # there is exactly one child. A subtree that reaches a leaf without ever
        # bifurcating gives its whole length.
        first_bif = 0.0
        cur_b = pr
        while len(children[cur_b]) == 1:
            nxt = children[cur_b][0]
            first_bif += euclidean(nodes[cur_b], nodes[nxt])
            cur_b = nxt
        subtree_first_bif_by_root[pr] = first_bif

    max_subtree_path_across = max(subtree_max_path_by_root.values(), default=1.0)
    max_subtree_z_across = max(subtree_z_span_by_root.values(), default=1.0)

    # z-span rank: sort primary roots by z_span, assign normalized rank (1.0 = largest)
    z_rank_of_root: dict[int, float] = {}
    sorted_roots_by_z = sorted(
        subtree_z_span_by_root.items(), key=lambda kv: kv[1], reverse=True
    )
    n_primary = len(sorted_roots_by_z)
    for rank, (pr, _) in enumerate(sorted_roots_by_z):
        # rank 0 is largest → 1.0; last → ~0
        z_rank_of_root[pr] = 1.0 - (rank / max(1, n_primary - 1)) if n_primary > 1 else 1.0

    radial_rank_of_root: dict[int, float] = {}
    sorted_roots_by_radial = sorted(
        subtree_max_radial_by_root.items(), key=lambda kv: kv[1], reverse=True
    )
    n_primary_radial = len(sorted_roots_by_radial)
    for rank, (pr, _) in enumerate(sorted_roots_by_radial):
        radial_rank_of_root[pr] = 1.0 - (rank / max(1, n_primary_radial - 1)) if n_primary_radial > 1 else 1.0

    # Partition into branches
    raw_branches = _partition_branches(nodes, parent_idx, children, soma_indices)

    branches: list[BranchData] = []
    for bid, (anchor, segment) in enumerate(raw_branches):
        if not segment:
            continue

        seg_nodes = [nodes[i] for i in segment]
        anchor_node = nodes[anchor]

        # --- Branch geometry ---
        path_length = 0.0
        for k in range(1, len(segment)):
            path_length += euclidean(nodes[segment[k - 1]], nodes[segment[k]])

        radial_extent = max(
            euclidean(anchor_node, nodes[i]) for i in segment
        ) if segment else 0.0

        radii = [nd.radius for nd in seg_nodes]
        mean_rad = float(np.mean(radii))
        std_rad = float(np.std(radii)) if len(radii) > 1 else 0.0
        min_rad = float(min(radii))
        max_rad = float(max(radii))

        # Taper ratio
        win = min(3, len(radii))
        prox_r = float(np.mean(radii[:win])) if radii else 1.0
        dist_r = float(np.mean(radii[-win:])) if radii else 1.0
        taper = dist_r / prox_r if prox_r > 1e-9 else 1.0

        # Persistence (straightness)
        euclid_dist = euclidean(nodes[segment[0]], nodes[segment[-1]]) if len(segment) > 1 else 0.0
        persistence = euclid_dist / path_length if path_length > 1e-9 else 0.5

        # Up alignment
        if len(segment) > 1:
            dz = nodes[segment[-1]].z - nodes[segment[0]].z
            dist_3d = euclid_dist
            up_align = (dz / dist_3d + 1.0) * 0.5 if dist_3d > 1e-9 else 0.5
            up_align = max(0.0, min(1.0, up_align))
        else:
            up_align = 0.5

        # Branchiness
        bif_count = sum(1 for i in segment if len(children[i]) > 1)
        branchiness = bif_count / max(1, len(segment))

        # Symmetry at anchor
        anchor_kids = children[anchor]
        if len(anchor_kids) > 1:
            kid_radii = [nodes[ci].radius for ci in anchor_kids]
            med = float(np.median(kid_radii))
            if med > 1e-9:
                mad = float(np.mean([abs(r - med) for r in kid_radii]))
                symmetry = 1.0 - min(1.0, mad / med)
            else:
                symmetry = 0.5
        else:
            symmetry = 0.5

        # --- Position relative to soma ---
        mid_idx = segment[len(segment) // 2]
        root_path_dist = path_from_root[mid_idx]
        root_radial_dist = math.sqrt(
            (nodes[mid_idx].x - soma_x)**2 +
            (nodes[mid_idx].y - soma_y)**2 +
            (nodes[mid_idx].z - soma_z)**2
        )
        z_rel = float(np.mean([nd.z - soma_z for nd in seg_nodes]))
        z_sp = max(nd.z for nd in seg_nodes) - min(nd.z for nd in seg_nodes) if seg_nodes else 0.0

        # --- Subtree context ---
        sub_size, sub_depth, sub_max_path = _subtree_stats(
            segment[0], children, nodes, parent_idx
        )
        b_order = branch_order_arr[segment[0]]

        # --- Parent/sibling context ---
        parent_rad = anchor_node.radius
        rad_ratio = mean_rad / parent_rad if parent_rad > 1e-9 else 1.0
        n_siblings = len(anchor_kids) - 1 if anchor in soma_indices else len(anchor_kids)
        is_primary = 1.0 if segment[0] in primary_set else 0.0

        # --- Cell-relative normalization ---
        path_rel = path_length / max_path_in_cell if max_path_in_cell > 1e-9 else 0.0
        radial_rel = radial_extent / max_radial_in_cell if max_radial_in_cell > 1e-9 else 0.0
        radius_rel = mean_rad / cell_mean_radius if cell_mean_radius > 1e-9 else 1.0
        subtree_rel = sub_size / max(1, n)

        # --- Ground truth label (majority vote, excluding soma) ---
        label_counts: dict[int, int] = {}
        for i in segment:
            t = nodes[i].type
            if t != 1:  # ground-truth target extraction only
                label_counts[t] = label_counts.get(t, 0) + 1

        if label_counts:
            gt_label = max(label_counts, key=lambda t: label_counts[t])
        else:
            gt_label = nodes[segment[0]].type  # fallback

        # --- Primary-subtree rank features ---
        br_primary_root = primary_root_of[segment[0]]
        if br_primary_root >= 0 and br_primary_root in subtree_max_path_by_root:
            this_sub_max_p = subtree_max_path_by_root[br_primary_root]
            this_sub_z = subtree_z_span_by_root[br_primary_root]
            this_sub_min_r = subtree_min_radius_by_root[br_primary_root]
            is_longest_sub = 1.0 if this_sub_max_p >= max_subtree_path_across - 1e-9 else 0.0
            sub_path_ratio = (
                this_sub_max_p / max_subtree_path_across
                if max_subtree_path_across > 1e-9 else 1.0
            )
            sub_z_ratio = (
                this_sub_z / max_subtree_z_across
                if max_subtree_z_across > 1e-9 else 0.0
            )
            sub_z_rank = z_rank_of_root.get(br_primary_root, 0.0)
            sub_radial_rank = radial_rank_of_root.get(br_primary_root, 0.0)
        else:
            is_longest_sub = 0.0
            sub_path_ratio = 0.0
            sub_z_ratio = 0.0
            sub_z_rank = 0.0
            sub_radial_rank = 0.0
            this_sub_min_r = mean_rad

        # --- Axon initial segment / thin-long signatures ---
        starts_at_soma = 1.0 if anchor in soma_indices else 0.0
        first_win = min(3, len(segment))
        first_mean_r = float(np.mean([nodes[segment[k]].radius for k in range(first_win)]))
        anchor_r = anchor_node.radius
        radius_drop = (
            (anchor_r - first_mean_r) / anchor_r if anchor_r > 1e-9 else 0.0
        )
        thin_nodes = sum(1 for r in radii if r < 0.5)
        thin_frac = thin_nodes / max(1, len(radii))

        # --- Multi-scale proximal / distal features ---
        seg_len = len(segment)
        third = max(1, seg_len // 3)
        proximal_radii = [nodes[segment[k]].radius for k in range(third)]
        distal_radii = [nodes[segment[k]].radius for k in range(seg_len - third, seg_len)]
        prox_mean_r = float(np.mean(proximal_radii)) if proximal_radii else mean_rad
        dist_mean_r = float(np.mean(distal_radii)) if distal_radii else mean_rad

        if seg_len >= 3:
            # Proximal persistence
            prox_path = 0.0
            for k in range(1, third):
                prox_path += euclidean(nodes[segment[k - 1]], nodes[segment[k]])
            prox_eu = euclidean(nodes[segment[0]], nodes[segment[third - 1]])
            prox_persist = prox_eu / prox_path if prox_path > 1e-9 else persistence

            # Distal persistence
            dist_path = 0.0
            for k in range(seg_len - third + 1, seg_len):
                dist_path += euclidean(nodes[segment[k - 1]], nodes[segment[k]])
            dist_eu = euclidean(nodes[segment[seg_len - third]], nodes[segment[-1]])
            dist_persist = dist_eu / dist_path if dist_path > 1e-9 else persistence
        else:
            prox_persist = persistence
            dist_persist = persistence

        # --- Apical-vs-basal discrimination features ---
        # Polar angle from +z of the soma→branch-midpoint vector.
        # Apical ~ 0 (straight up), basal ~ π/2 (horizontal), axon often > π/2.
        mid_node = nodes[mid_idx]
        dx_mid = mid_node.x - soma_x
        dy_mid = mid_node.y - soma_y
        dz_mid = mid_node.z - soma_z
        r_mid = math.sqrt(dx_mid * dx_mid + dy_mid * dy_mid + dz_mid * dz_mid)
        if r_mid > 1e-9:
            polar_angle_up = math.acos(max(-1.0, min(1.0, dz_mid / r_mid)))
        else:
            polar_angle_up = math.pi / 2.0  # unknown → horizontal default

        # Primary-subtree-based features use the primary root this branch belongs to.
        if br_primary_root >= 0 and br_primary_root in subtree_z_span_by_root:
            sub_z = subtree_z_span_by_root[br_primary_root]
            sub_xy = subtree_xy_span_by_root.get(br_primary_root, 0.0)
            vh_ratio = sub_z / max(sub_xy, 1e-6)
            polar_spread = subtree_polar_std_by_root.get(br_primary_root, 0.0)
        else:
            vh_ratio = 0.0
            polar_spread = 0.0

        # Mean z offset of this branch relative to soma, normalized by cell z-range.
        # Apical branches land near +1, basal near 0, descending axons negative.
        soma_z_offset_norm = z_rel / cell_z_range

        # --- Cell-intrinsic principal-axis features ---
        # Branch midpoint vector from soma, projected onto cell PC1.
        mid_proj_raw = (
            dx_mid * pc1_vec[0] + dy_mid * pc1_vec[1] + dz_mid * pc1_vec[2]
        )
        principal_axis_proj = mid_proj_raw / max_abs_proj  # ~[-1, +1]
        # Polar angle between (mid - soma) and PC1.
        if r_mid > 1e-9:
            cos_pc = max(-1.0, min(1.0, mid_proj_raw / r_mid))
            polar_angle_principal = math.acos(cos_pc)
        else:
            polar_angle_principal = math.pi / 2.0
        # Subtree-level features
        if br_primary_root >= 0 and br_primary_root in subtree_principal_proj_by_root:
            sub_principal_proj = (
                subtree_principal_proj_by_root[br_primary_root] / max_abs_proj
            )
            sub_principal_rank = principal_rank_of_root.get(br_primary_root, 0.0)
        else:
            sub_principal_proj = 0.0
            sub_principal_rank = 0.0

        # --- Branching-rate features ---
        # bifurcations_per_micron in this branch
        bif_per_micron = (
            float(bif_count) / path_length if path_length > 1e-9 else 0.0
        )
        # mean internode distance (path_length already excludes the anchor edge)
        mean_internode = (
            path_length / max(1, len(segment) - 1) if len(segment) > 1 else 0.0
        )
        # subtree-level bif density (axon: low; dendrite: high)
        sub_bif_density = (
            subtree_bif_density_by_root.get(br_primary_root, 0.0)
            if br_primary_root >= 0 else 0.0
        )

        # --- Trunk-detection features ---
        # Direct encoding of "apical = one dominant trunk; basal = bushy".
        if br_primary_root >= 0 and br_primary_root in subtree_first_bif_by_root:
            sub_first_bif = subtree_first_bif_by_root[br_primary_root]
            path_to_first_bif_norm = (
                sub_first_bif / max_path_in_cell if max_path_in_cell > 1e-9 else 0.0
            )
        else:
            path_to_first_bif_norm = 0.0

        # Build feature vector
        fv = np.array([
            path_length,
            radial_extent,
            float(len(segment)),
            mean_rad,
            std_rad,
            min_rad,
            max_rad,
            taper,
            persistence,
            up_align,
            branchiness,
            symmetry,

            root_path_dist,
            root_radial_dist,
            z_rel,
            z_sp,

            float(sub_size),
            float(sub_depth),
            sub_max_path,
            float(b_order),

            parent_rad,
            rad_ratio,
            float(n_siblings),
            is_primary,

            path_rel,
            radial_rel,
            radius_rel,
            subtree_rel,

            is_pyramidal,
            is_interneuron,
            is_purkinje,

            # Primary-subtree rank features
            is_longest_sub,
            sub_path_ratio,
            sub_z_ratio,
            sub_z_rank,
            sub_radial_rank,

            # Axon initial segment / thin-long signatures
            starts_at_soma,
            radius_drop,
            thin_frac,
            this_sub_min_r,

            # Multi-scale proximal / distal
            prox_mean_r,
            dist_mean_r,
            prox_persist,
            dist_persist,

            # Apical-vs-basal discrimination
            polar_angle_up,
            vh_ratio,
            soma_z_offset_norm,
            polar_spread,

            # Cell-intrinsic principal axis (PC1)
            principal_axis_proj,
            polar_angle_principal,
            pc1_strength,
            sub_principal_proj,
            sub_principal_rank,

            # Branching-rate (axon vs dendrite discriminator)
            bif_per_micron,
            mean_internode,
            sub_bif_density,

            # Trunk-detection (apical-vs-basal discriminator)
            path_to_first_bif_norm,
        ], dtype=np.float64)
        if len(fv) != len(BRANCH_FEATURE_NAMES):
            # The correction and the all-in-one model read this row positionally
            # against BRANCH_FEATURE_NAMES. A row that does not match would
            # still fit and predict, with every column read as its neighbour.
            raise AssertionError(
                f"built {len(fv)} branch features but BRANCH_FEATURE_NAMES lists "
                f"{len(BRANCH_FEATURE_NAMES)}; the two must stay in step.")

        branches.append(BranchData(
            branch_id=bid,
            node_indices=segment,
            anchor_idx=anchor,
            features=fv,
            gt_label=gt_label,
            gt_label_counts=label_counts,
            n_nodes=len(segment),
            primary_root_idx=br_primary_root if br_primary_root >= 0 else None,
        ))

    return MorphologyBranches(
        file_path=file_path,
        cell_type=cell_type,
        branches=branches,
        soma_indices=sorted(soma_indices),
        total_nodes=n,
    )
