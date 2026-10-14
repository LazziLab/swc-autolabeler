"""SWC parsing, shared tree helpers, and the 55 whole-cell features of Stage 1.

The features summarize a morphology from its geometry alone. Nothing here reads
the SWC type column except ``parse_swc``, which reads it to normalize it, so a
cell type predicted from these descriptors cannot have been read off the labels
the pipeline is about to assign.

This module also holds the helpers every stage needs: how a tree is built from a
node list, and which node stands in for the soma. They live here because Stages
1, 2 and 3 must agree on both, and a rule copied into three modules agrees only
until someone edits one of them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# SWC parsing (minimal, standalone)
# ---------------------------------------------------------------------------

@dataclass
class SWCNode:
    id: int
    type: int
    x: float
    y: float
    z: float
    radius: float
    parent: int


def parse_swc(path: str | Path, *, normalize_types: bool = True) -> list[SWCNode]:
    """Parse an SWC file into a list of SWCNode.

    If ``normalize_types`` is True (the default), nodes carrying a
    non-standard SWC type value (anything outside {1, 2, 3, 4}) are
    rewritten to the dominant standard type of the branch they belong to.
    This absorbs custom sub-cellular annotations (axon hillock, spines,
    boutons, etc.) into their host neurite so downstream training and
    evaluation see only the four canonical neurite classes. Topology is
    never altered.

    Pass ``normalize_types=False`` if you need the raw on-disk types.
    """
    nodes: list[SWCNode] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            try:
                nodes.append(SWCNode(
                    id=int(float(parts[0])),
                    type=int(float(parts[1])),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    radius=float(parts[5]),
                    parent=int(float(parts[6])),
                ))
            except (ValueError, IndexError):
                continue

    if normalize_types and nodes:
        # Local import to avoid a circular dependency at module-load time.
        # normalize_swc applies BOTH passes in canonical order:
        #   (1) rewrite non-standard SWC types into {1,2,3,4}
        #   (2) consolidate connected multi-point soma components into a
        #       single anchor node (centroid + mega-radius) so the metric
        #       reflects one soma per cell.
        from .swc_normalize import normalize_swc
        nodes, _ = normalize_swc(nodes)
    return nodes


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def tree_by_id(nodes: list[SWCNode]) -> tuple[
    dict[int, SWCNode],
    dict[int, list[int]],
    list[int],
]:
    """The tree keyed by SWC node id: (id->node, id->child ids, root ids).

    ``tree_by_index`` is the same tree keyed by position in ``nodes``. Both are
    kept because the callers are written around different keys; they describe
    the same edges and the same roots.
    """
    by_id = {n.id: n for n in nodes}
    children: dict[int, list[int]] = {n.id: [] for n in nodes}
    roots: list[int] = []
    for n in nodes:
        if n.parent == -1 or n.parent not in by_id:
            roots.append(n.id)
        else:
            children[n.parent].append(n.id)
    return by_id, children, roots


def tree_by_index(nodes: list[SWCNode]) -> tuple[
    dict[int, int],
    list[int | None],
    list[list[int]],
    list[int],
]:
    """The tree keyed by position: (id->index, parent index, children, roots).

    A node whose parent id is absent from the file is treated as a root, so a
    fragmented reconstruction yields several. See ``tree_by_id`` for the other
    keying.
    """
    id_to_idx = {n.id: i for i, n in enumerate(nodes)}
    n = len(nodes)
    parent_idx: list[int | None] = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, nd in enumerate(nodes):
        pidx = id_to_idx.get(nd.parent)
        parent_idx[i] = pidx
        if pidx is not None:
            children[pidx].append(i)
        if nd.parent == -1 or pidx is None:
            roots.append(i)
    return id_to_idx, parent_idx, children, roots


def euclidean(a: SWCNode, b: SWCNode) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def proxy_root_key(radius: float, n_children: int, position: int) -> tuple[float, int, int]:
    """How the soma anchor is chosen, in one place.

    The widest root, ties broken by more children and then by the earlier node
    in the file. Every stage ranks candidates with this, whether its tree is
    keyed by id or by index, so the rule cannot drift between them.
    """
    return (radius, n_children, -position)


def select_proxy_root(
    nodes: list[SWCNode],
    children: list[list[int]],
    roots: list[int],
) -> int:
    """Index of the label-free node that stands in for the soma.

    ``children`` and ``roots`` are keyed by index, as ``tree_by_index`` returns
    them. Callers holding an id-keyed tree rank with ``proxy_root_key`` directly.
    """
    return max(roots or [0],
               key=lambda i: proxy_root_key(nodes[i].radius, len(children[i]), i))


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    # Size / count features
    "n_nodes",
    "n_root_nodes",
    "n_branch_points",
    "n_terminals",
    "n_primary_subtrees",       # soma children count
    "branch_point_ratio",       # branch_points / total_nodes
    "terminal_ratio",           # terminals / total_nodes

    # Path / distance features
    "max_path_length",
    "mean_path_length",
    "std_path_length",
    "max_radial_distance",
    "mean_radial_distance",
    "std_radial_distance",
    "path_radial_ratio_mean",   # mean(path/radial) — tortuosity proxy

    # Radius features
    "mean_radius",
    "std_radius",
    "max_radius",
    "radius_cv",                # coefficient of variation

    # Z-extent features
    "z_span",
    "z_asymmetry",              # (above - below) / span, relative to soma
    "z_max_above_soma",
    "z_max_below_soma",
    "x_span",
    "y_span",
    "xy_aspect_ratio",
    "planarity",
    "linearity",
    "thickness_ratio",

    # Subtree features
    "max_subtree_size",
    "min_subtree_size",
    "subtree_size_std",
    "max_subtree_path",
    "subtree_path_std",
    "max_subtree_size_ratio",
    "max_subtree_path_ratio",

    # Strahler order
    "max_strahler",
    "mean_strahler",

    # Sholl-like features
    "sholl_max_intersections",
    "sholl_peak_distance",
    "sholl_decay_rate",
    "terminal_radial_mean",
    "terminal_radial_std",

    # PCA principal-axis features (rotation- and slice-invariant analogues
    # of the z-axis features above). These let the classifier discriminate
    # pyramidals from interneurons even when the cell is rotated, the
    # tissue was sliced thin (z_span collapsed), or the z-axis is not the
    # cortical depth axis.
    "pc1_span",                    # total extent along PC1
    "pc1_asymmetry",               # (above_soma - below_soma) / span on PC1
    "pc1_max_above_soma",          # extent in signed-positive PC1 direction
    "pc1_max_below_soma",          # extent in signed-negative PC1 direction
    "pc1_radial_max",              # max |projection on PC1| (cell radius along PC1)
    "subtree_pc1_concentration",   # fraction of total |PC1 mass| in top-aligned primary subtree
    "subtree_pc1_top_alignment",   # mean signed PC1 projection of top subtree, normalized to pc1_radial_max

    # Extra discriminative features: soma/neurite geometry and branching
    "soma_radius",                 # radius of the soma anchor node (mega-radius after consolidation)
    "max_neurite_to_soma_ratio",   # max neurite radius / soma radius (interneurons have higher)
    "branching_density",           # n_branch_points / max_path_length (per micron)
    "terminal_density",            # n_terminals / max_path_length (per micron)
    "bif_angle_mean",              # mean bifurcation angle (degrees)
    "bif_angle_std",               # std of bifurcation angles
]

# Set SWCAL_NO_PCA=1 to exclude the PCA features. Filtering FEATURE_NAMES
# keeps training and inference at the same dimensionality: the extraction
# code still populates the PCA keys, but extract_feature_vector reads only
# the filtered FEATURE_NAMES, so the model's input shape matches.
import os as _os  # noqa: E402
if _os.environ.get("SWCAL_NO_PCA") == "1":
    _PCA_KEYS = {
        "pc1_span", "pc1_asymmetry", "pc1_max_above_soma",
        "pc1_max_below_soma", "pc1_radial_max",
        "subtree_pc1_concentration", "subtree_pc1_top_alignment",
    }
    FEATURE_NAMES = [n for n in FEATURE_NAMES if n not in _PCA_KEYS]


def extract_global_features(nodes: list[SWCNode]) -> dict[str, float]:
    """Compute the full global feature vector for a morphology."""
    if not nodes:
        return {name: 0.0 for name in FEATURE_NAMES}

    by_id, children, roots = tree_by_id(nodes)
    n = len(nodes)
    position = {nd.id: i for i, nd in enumerate(nodes)}

    # ---- Soma proxy detection (label-free) ----
    # The same ranking every other stage applies through select_proxy_root,
    # spelled on ids because this function's tree is keyed by id.
    root_ids = roots or [nodes[0].id]
    proxy_root_id = max(root_ids, key=lambda nid: proxy_root_key(
        by_id[nid].radius, len(children.get(nid, [])), position[nid]))
    proxy_root = by_id[proxy_root_id]
    soma_center = SWCNode(
        id=-1, type=1,
        x=proxy_root.x, y=proxy_root.y, z=proxy_root.z,
        radius=0.0, parent=-1,
    )
    non_proxy_nodes = [nd for nd in nodes if nd.id != proxy_root_id]
    if not non_proxy_nodes:
        non_proxy_nodes = list(nodes)

    # ---- BFS to compute path lengths ----
    path_from_root: dict[int, float] = {}
    order: list[int] = []
    queue = list(roots)
    for rid in roots:
        path_from_root[rid] = 0.0
    visited: set[int] = set()
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        order.append(nid)
        nd = by_id[nid]
        for cid in children[nid]:
            child_nd = by_id[cid]
            path_from_root[cid] = path_from_root.get(nid, 0.0) + euclidean(nd, child_nd)
            queue.append(cid)

    # ---- Radial distances from soma ----
    radial_distances = [euclidean(nd, soma_center) for nd in nodes]

    # ---- Basic counts ----
    branch_points = [nid for nid, kids in children.items() if len(kids) >= 2]
    terminals = [nid for nid, kids in children.items() if len(kids) == 0 and nid not in roots]
    n_branch_points = len(branch_points)
    n_terminals = len(terminals)

    # ---- Primary subtrees (soma children) ----
    primary_roots: list[int] = []
    for cid in children.get(proxy_root_id, []):
        if cid != proxy_root_id:
            primary_roots.append(cid)
    if not primary_roots and roots:
        # Fallback: direct root children
        for rid in roots:
            if rid == proxy_root_id:
                continue
            primary_roots.append(rid)
    n_primary = len(primary_roots)

    # ---- Subtree sizes and max paths ----
    subtree_sizes: list[int] = []
    subtree_max_paths: list[float] = []
    for pr in primary_roots:
        stack = [pr]
        size = 0
        max_path = 0.0
        while stack:
            cur = stack.pop()
            size += 1
            max_path = max(max_path, path_from_root.get(cur, 0.0))
            stack.extend(children.get(cur, []))
        subtree_sizes.append(size)
        subtree_max_paths.append(max_path)

    # ---- Path length stats ----
    path_vals = [path_from_root.get(nd.id, 0.0) for nd in non_proxy_nodes]
    if not path_vals:
        path_vals = [0.0]

    # ---- Radius stats (non-soma) ----
    radii = [nd.radius for nd in non_proxy_nodes]
    if not radii:
        radii = [nd.radius for nd in nodes]
    if not radii:
        radii = [0.0]

    # ---- Z stats ----
    soma_z = soma_center.z
    z_vals = [nd.z for nd in nodes]
    x_vals = [nd.x for nd in nodes]
    y_vals = [nd.y for nd in nodes]
    z_above = max(0.0, max(z_vals) - soma_z) if z_vals else 0.0
    z_below = max(0.0, soma_z - min(z_vals)) if z_vals else 0.0
    z_span = z_above + z_below
    z_asymmetry = (z_above - z_below) / z_span if z_span > 1e-9 else 0.0
    x_span = (max(x_vals) - min(x_vals)) if x_vals else 0.0
    y_span = (max(y_vals) - min(y_vals)) if y_vals else 0.0
    xy_aspect_ratio = max(x_span, y_span) / max(1e-9, min(x_span, y_span)) if min(x_span, y_span) > 1e-9 else 1.0

    coords = np.array([[nd.x, nd.y, nd.z] for nd in nodes], dtype=np.float64)
    soma_xyz = np.array([soma_center.x, soma_center.y, soma_center.z], dtype=np.float64)
    pc1_span = 0.0
    pc1_asymmetry = 0.0
    pc1_max_above_soma = 0.0
    pc1_max_below_soma = 0.0
    pc1_radial_max = 0.0
    subtree_pc1_concentration = 0.0
    subtree_pc1_top_alignment = 0.0

    if len(coords) >= 3:
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid
        cov = np.cov(centered.T)
        # eigh returns ascending order; eigvecs columns correspond to eigvals
        eigvals_asc, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals_asc[::-1], 0.0)
        lam1, lam2, lam3 = eigvals.tolist()
        linearity = (lam1 - lam2) / lam1 if lam1 > 1e-9 else 0.0
        planarity = (lam2 - lam3) / lam1 if lam1 > 1e-9 else 0.0
        thickness_ratio = lam3 / lam1 if lam1 > 1e-9 else 0.0

        # ---- PC1 principal axis (rotation-invariant analogue of z-axis) ----
        pc1 = eigvecs[:, -1].astype(np.float64)
        # Sign convention: positive PC1 points from soma toward the cell
        # centroid. For pyramidals this aligns roughly with the apical
        # trunk; for interneurons the signed direction is biology-agnostic.
        if np.dot(centroid - soma_xyz, pc1) < 0:
            pc1 = -pc1
        proj = (coords - soma_xyz) @ pc1                # signed, soma-relative
        pc1_max_above_soma = float(max(0.0, float(proj.max())))
        pc1_max_below_soma = float(max(0.0, float(-proj.min())))
        pc1_span = pc1_max_above_soma + pc1_max_below_soma
        if pc1_span > 1e-9:
            pc1_asymmetry = (pc1_max_above_soma - pc1_max_below_soma) / pc1_span
        pc1_radial_max = max(pc1_max_above_soma, pc1_max_below_soma)

        # ---- Per-primary-subtree PC1 alignment ----
        # For each primary subtree, compute (1) the sum of |projections|
        # contributed by its nodes (a "PC1-mass" of the subtree) and
        # (2) the mean signed projection. The most apical-like subtree is
        # the one with the most positive mean projection.
        if primary_roots and pc1_radial_max > 1e-9:
            sub_abs_mass: list[float] = []
            sub_signed_mean: list[float] = []
            for pr in primary_roots:
                stack = [pr]
                projs: list[float] = []
                while stack:
                    cur = stack.pop()
                    idx = position.get(cur)
                    if idx is not None:
                        projs.append(float(proj[idx]))
                    stack.extend(children.get(cur, []))
                if projs:
                    arr = np.asarray(projs, dtype=np.float64)
                    sub_abs_mass.append(float(np.sum(np.abs(arr))))
                    sub_signed_mean.append(float(np.mean(arr)))
                else:
                    sub_abs_mass.append(0.0)
                    sub_signed_mean.append(0.0)
            if sub_signed_mean:
                top_idx = int(np.argmax(sub_signed_mean))
                subtree_pc1_top_alignment = sub_signed_mean[top_idx] / pc1_radial_max
                total_mass = float(sum(sub_abs_mass))
                if total_mass > 1e-9:
                    subtree_pc1_concentration = sub_abs_mass[top_idx] / total_mass
    else:
        linearity = 0.0
        planarity = 0.0
        thickness_ratio = 0.0

    # ---- Tortuosity ----
    radial_non_soma = [r for r, nd in zip(radial_distances, nodes) if nd.id != proxy_root_id]
    path_non_soma = [path_from_root.get(nd.id, 0.0) for nd in non_proxy_nodes]
    tortuosity_vals = []
    for p, r in zip(path_non_soma, radial_non_soma):
        if r > 1e-6:
            tortuosity_vals.append(p / r)
    if not tortuosity_vals:
        tortuosity_vals = [1.0]

    # ---- Strahler order ----
    strahler = _compute_strahler(by_id, children, roots)
    strahler_vals = list(strahler.values()) if strahler else [0]

    # ---- Sholl analysis (simplified) ----
    sholl_max, sholl_peak_dist, sholl_decay = _sholl_features(radial_distances, nodes)
    terminal_radials = [
        euclidean(by_id[nid], soma_center)
        for nid in terminals if nid in by_id
    ]
    if not terminal_radials:
        terminal_radials = [0.0]

    mean_radius = float(np.mean(radii))
    std_radius = float(np.std(radii))

    feats: dict[str, float] = {
        "n_nodes": float(n),
        "n_root_nodes": float(len(root_ids)),
        "n_branch_points": float(n_branch_points),
        "n_terminals": float(n_terminals),
        "n_primary_subtrees": float(n_primary),
        "branch_point_ratio": n_branch_points / max(1, n),
        "terminal_ratio": n_terminals / max(1, n),

        "max_path_length": float(max(path_vals)),
        "mean_path_length": float(np.mean(path_vals)),
        "std_path_length": float(np.std(path_vals)),
        "max_radial_distance": float(max(radial_distances)),
        "mean_radial_distance": float(np.mean(radial_distances)),
        "std_radial_distance": float(np.std(radial_distances)),
        "path_radial_ratio_mean": float(np.mean(tortuosity_vals)),

        "mean_radius": mean_radius,
        "std_radius": std_radius,
        "max_radius": float(max(radii)),
        "radius_cv": std_radius / mean_radius if mean_radius > 1e-9 else 0.0,

        "z_span": z_span,
        "z_asymmetry": z_asymmetry,
        "z_max_above_soma": z_above,
        "z_max_below_soma": z_below,
        "x_span": x_span,
        "y_span": y_span,
        "xy_aspect_ratio": xy_aspect_ratio,
        "planarity": float(planarity),
        "linearity": float(linearity),
        "thickness_ratio": float(thickness_ratio),

        "max_subtree_size": float(max(subtree_sizes)) if subtree_sizes else 0.0,
        "min_subtree_size": float(min(subtree_sizes)) if subtree_sizes else 0.0,
        "subtree_size_std": float(np.std(subtree_sizes)) if len(subtree_sizes) > 1 else 0.0,
        "max_subtree_path": float(max(subtree_max_paths)) if subtree_max_paths else 0.0,
        "subtree_path_std": float(np.std(subtree_max_paths)) if len(subtree_max_paths) > 1 else 0.0,
        "max_subtree_size_ratio": (float(max(subtree_sizes)) / max(1.0, float(sum(subtree_sizes)))) if subtree_sizes else 0.0,
        "max_subtree_path_ratio": (float(max(subtree_max_paths)) / max(1e-9, float(sum(subtree_max_paths)))) if subtree_max_paths else 0.0,

        "max_strahler": float(max(strahler_vals)),
        "mean_strahler": float(np.mean(strahler_vals)),

        "sholl_max_intersections": float(sholl_max),
        "sholl_peak_distance": float(sholl_peak_dist),
        "sholl_decay_rate": float(sholl_decay),
        "terminal_radial_mean": float(np.mean(terminal_radials)),
        "terminal_radial_std": float(np.std(terminal_radials)),

        # PCA principal-axis features (rotation-invariant)
        "pc1_span": float(pc1_span),
        "pc1_asymmetry": float(pc1_asymmetry),
        "pc1_max_above_soma": float(pc1_max_above_soma),
        "pc1_max_below_soma": float(pc1_max_below_soma),
        "pc1_radial_max": float(pc1_radial_max),
        "subtree_pc1_concentration": float(subtree_pc1_concentration),
        "subtree_pc1_top_alignment": float(subtree_pc1_top_alignment),
    }

    # =========================================================================
    # Extra discriminative features: soma/neurite geometry, branching, angles
    # =========================================================================
    # Soma radius — the proxy_root's radius. After multi-point-soma
    # consolidation this is the "mega-radius" of the whole soma cluster.
    soma_radius_val = float(proxy_root.radius)
    feats["soma_radius"] = soma_radius_val

    # Max non-soma neurite radius / soma_radius. Interneurons tend to have
    # neurites whose thickest segment is a larger fraction of their soma
    # than pyramidals (whose soma is comparatively large).
    non_proxy_radii = [nd.radius for nd in nodes if nd.id != proxy_root_id]
    if non_proxy_radii and soma_radius_val > 1e-6:
        feats["max_neurite_to_soma_ratio"] = float(max(non_proxy_radii)) / soma_radius_val
    else:
        feats["max_neurite_to_soma_ratio"] = 0.0

    # Branching / terminal density — number of branch-points or terminals
    # per micron of longest path. Interneurons are denser.
    max_pl = float(max(path_vals)) if path_vals else 0.0
    feats["branching_density"]  = (n_branch_points / max_pl) if max_pl > 1e-6 else 0.0
    feats["terminal_density"]   = (n_terminals     / max_pl) if max_pl > 1e-6 else 0.0

    # Bifurcation angles. For each branch point with >= 2 children,
    # compute the angle between the unit vectors to the first two children
    # (in 3D). Pyramidals' apical bifurcations tend to be wider (~180°);
    # interneurons' branching is more variable.
    bif_angles_deg: list[float] = []
    for bp_id in branch_points:
        bp_nd = by_id[bp_id]
        kid_ids = list(children[bp_id])[:2]   # take the first two children
        if len(kid_ids) < 2:
            continue
        v1 = np.array([by_id[kid_ids[0]].x - bp_nd.x,
                        by_id[kid_ids[0]].y - bp_nd.y,
                        by_id[kid_ids[0]].z - bp_nd.z], dtype=np.float64)
        v2 = np.array([by_id[kid_ids[1]].x - bp_nd.x,
                        by_id[kid_ids[1]].y - bp_nd.y,
                        by_id[kid_ids[1]].z - bp_nd.z], dtype=np.float64)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        bif_angles_deg.append(math.degrees(math.acos(cos_theta)))
    feats["bif_angle_mean"] = float(np.mean(bif_angles_deg)) if bif_angles_deg else 0.0
    feats["bif_angle_std"]  = float(np.std(bif_angles_deg))  if len(bif_angles_deg) > 1 else 0.0

    if set(feats) != set(FEATURE_NAMES):
        # extract_feature_vector reads FEATURE_NAMES in order, so a name computed
        # here but never listed is silently dropped, and every model fitted after
        # that reads a column it was not trained on. Cheap to check, invisible to
        # get wrong.
        raise AssertionError(
            f"computed {len(feats)} features but FEATURE_NAMES lists "
            f"{len(FEATURE_NAMES)}; extra {sorted(set(feats) - set(FEATURE_NAMES))}, "
            f"missing {sorted(set(FEATURE_NAMES) - set(feats))}")
    return feats


def extract_feature_vector(nodes: list[SWCNode]) -> np.ndarray:
    """Return feature dict values as a numpy array in canonical order."""
    feats = extract_global_features(nodes)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Strahler order
# ---------------------------------------------------------------------------

def _compute_strahler(
    by_id: dict[int, SWCNode],
    children: dict[int, list[int]],
    roots: list[int],
) -> dict[int, int]:
    """Compute Strahler order for each node (bottom-up)."""
    strahler: dict[int, int] = {}

    # Post-order traversal
    post_order: list[int] = []
    stack: list[tuple[int, bool]] = [(r, False) for r in roots]
    while stack:
        nid, processed = stack.pop()
        if processed:
            post_order.append(nid)
            continue
        stack.append((nid, True))
        for cid in children.get(nid, []):
            stack.append((cid, False))

    for nid in post_order:
        kids = children.get(nid, [])
        if not kids:
            strahler[nid] = 1
        else:
            child_orders = sorted([strahler.get(c, 1) for c in kids], reverse=True)
            if len(child_orders) == 1:
                strahler[nid] = child_orders[0]
            elif child_orders[0] == child_orders[1]:
                strahler[nid] = child_orders[0] + 1
            else:
                strahler[nid] = child_orders[0]
    return strahler


# ---------------------------------------------------------------------------
# Sholl analysis (simplified)
# ---------------------------------------------------------------------------

def _sholl_features(
    radial_distances: list[float],
    nodes: list[SWCNode],
    n_bins: int = 20,
) -> tuple[float, float, float]:
    """Compute simplified Sholl features.

    Returns (max_intersections, peak_distance, decay_rate).
    """
    # Sholl excludes every root from the shells, where the rest of this function
    # excludes the one proxy root. On a cell with a single root, which is every
    # one of the 11,862 in the corpus, the two are the same node. They part only
    # on a fragmented reconstruction, where this drops each disconnected piece's
    # first node as well; the three Sholl features are the only ones affected.
    root_ids = {nd.id for nd in nodes if nd.parent == -1}
    if not root_ids and nodes:
        root_ids = {nodes[0].id}
    non_soma_radial = [r for r, nd in zip(radial_distances, nodes) if nd.id not in root_ids]
    if not non_soma_radial:
        return 0.0, 0.0, 0.0

    max_r = max(non_soma_radial)
    if max_r < 1e-6:
        return 0.0, 0.0, 0.0

    bin_edges = np.linspace(0, max_r, n_bins + 1)
    counts = np.zeros(n_bins)
    for r in non_soma_radial:
        idx = int(r / max_r * n_bins)
        idx = min(idx, n_bins - 1)
        counts[idx] += 1

    peak_idx = int(np.argmax(counts))
    peak_dist = float((bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2)
    max_count = float(counts[peak_idx])

    # Decay rate: how fast counts drop after peak
    if peak_idx < n_bins - 1 and max_count > 0:
        post_peak = counts[peak_idx + 1:]
        if len(post_peak) > 0:
            decay = float(max_count - np.mean(post_peak)) / max_count
        else:
            decay = 0.0
    else:
        decay = 0.0

    return max_count, peak_dist, decay
