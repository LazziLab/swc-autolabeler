"""Classical morphometric baselines for the neuron auto-labeling comparison.

Each baseline extracts hand-designed morphometric features from the raw SWC and
trains a shallow classifier, then predicts per-node neurite types. These are the
"features + classifier" rows the comparison table reports against the pipeline,
evaluated on the same hash-split test set.

Six methods are registered in ``_CACHE_REGISTRY``, each reached through
``predict_with_cache`` and each returning a drop-in
``predict_fn(nodes, cell_type) -> list[int]``. Four learn:

    neurom_rf     Per-branch NeuroM-style morphometric features (length,
                  radial extent, radii, branchiness, partition asymmetry,
                  taper) into a RandomForest, per-branch label propagated to
                  nodes.
    lmeasure_rf   Per-subtree L-Measure-style morphometric features into a
                  RandomForest.
    sholl_rf      Per-primary-subtree Sholl features (intersections at multiple
                  radii, peak, total branches and length, principal-axis
                  orientation) into a RandomForest, 3-class prediction
                  propagated to the subtree's nodes.
    sholl_mlp     The same Sholl features with a small sklearn MLP head.

Two are floors with nothing to fit, which bound the metric from below:

    majority      Every neurite node becomes basal dendrite, the labeling a
                  laboratory that draws no compartment distinction produces.
    rules_only    The geometric priors with no learning: thinnest primary
                  subtree is the axon, farthest-reaching of the rest is the
                  apical, everything else basal.

``predict_with_cache`` trains a method (or loads a cached fit) and returns its
predictor. Callers pass a per-seed cache path, so a model trained on one seed's
split is never reused on another seed's test cells (which would leak).

Nothing in ``pipeline/`` imports this module: the baselines exist to be compared
against the labeler, not to run it. They are trained and scored on the same hash
split as the pipeline, so the margins the comparison table reports are measured
on identical held-out cells. The two callers are the paper's baseline comparison
(``evaluation.baselines.eval_baselines``) and the downstream morphometrics study
(``evaluation.downstream_morphometrics``).
"""
from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Quieten routine sklearn chatter, but never the warning that a pickle was
# fitted under a different library version: that one means a cached model may
# not behave as it did when it was written.
try:                                             # sklearn >= 1.3
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("always", category=InconsistentVersionWarning)
except ImportError:                              # older sklearn
    pass
warnings.filterwarnings("ignore", category=FutureWarning)

from pipeline.features import SWCNode, parse_swc                          # noqa: E402
from pipeline.stage1.cell_type_detector import CELL_TYPE_LABEL_SETS       # noqa: E402


# =============================================================================
# Topology helpers (lightweight; replicate just enough of branch_features for
# the baselines to stand on their own without pulling in the full pipeline).
# =============================================================================

@dataclass
class _Topo:
    """Resolved tree topology for one cell."""
    nodes: list
    id_to_idx: dict[int, int]
    parent_idx: list  # list[int | None]
    children: list[list[int]]
    roots: list[int]


def _build_topo(nodes: list[SWCNode]) -> _Topo:
    id_to_idx = {n.id: i for i, n in enumerate(nodes)}
    n = len(nodes)
    parent_idx: list = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, nd in enumerate(nodes):
        pidx = id_to_idx.get(nd.parent)
        parent_idx[i] = pidx
        if pidx is not None:
            children[pidx].append(i)
        if nd.parent == -1 or pidx is None:
            roots.append(i)
    return _Topo(nodes, id_to_idx, parent_idx, children, roots)


def _select_proxy_root(topo: _Topo) -> int:
    """Largest-radius root, falling back to the first root or 0."""
    if not topo.nodes:
        return 0
    candidate_roots = topo.roots or [0]
    return max(
        candidate_roots,
        key=lambda idx: (topo.nodes[idx].radius, len(topo.children[idx]), -idx),
    )


def _subtree_indices(topo: _Topo, root_idx: int) -> list[int]:
    """All node indices reachable from root_idx (inclusive) via children.

    Never empty: the root itself is always the first entry, so callers need no
    empty-subtree case.
    """
    out: list[int] = []
    stack = [root_idx]
    seen: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
        stack.extend(topo.children[idx])
    return out


def _euclid(a: SWCNode, b: SWCNode) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


# =============================================================================
# BASELINE 1 — NeuroM-style per-branch features + RandomForest
# =============================================================================
#
# We split the morphology into linear branch segments anchored at roots /
# bifurcations (same as the pipeline's branch decomposition, but reimplemented
# here so the baseline doesn't depend on the proprietary feature extractor).
# For each branch we compute a feature vector that mirrors the kind of metrics
# NeuroM exposes per-section: length, max-radial, max-Z, mean-radius, taper,
# branchiness (downstream branch count), partition asymmetry, persistence
# (number of bifurcations along the branch), and parent-cell-type one-hot.
# A RandomForest is then trained on (features → ground-truth majority label).
# =============================================================================

def _branch_segments(topo: _Topo, proxy: int) -> list[list[int]]:
    """Return a list of branch segments. Each segment is a list of node
    indices forming a linear path from a bifurcation anchor (or root) to
    the next bifurcation / leaf, in BFS order."""
    segments: list[list[int]] = []
    visited: set[int] = set()
    visited.add(proxy)

    queue: list[int] = list(topo.children[proxy])
    while queue:
        start = queue.pop(0)
        if start in visited:
            continue
        seg: list[int] = []
        cur = start
        while cur is not None and cur not in visited:
            visited.add(cur)
            seg.append(cur)
            kids = topo.children[cur]
            if len(kids) != 1:
                # bifurcation or leaf — branch ends here
                queue.extend(kids)
                break
            cur = kids[0]
        if seg:
            segments.append(seg)
    return segments


def _branch_neurom_features(
    topo: _Topo,
    proxy: int,
    seg: list[int],
    cell_type: str,
) -> np.ndarray:
    """NeuroM-style 17-dim feature vector for a branch segment."""
    nodes = topo.nodes
    soma = nodes[proxy]
    head = nodes[seg[0]]
    tail = nodes[seg[-1]]

    # Path length (sum of segment Euclideans along the branch).
    path_len = 0.0
    for a, b in zip(seg[:-1], seg[1:]):
        path_len += _euclid(nodes[a], nodes[b])
    # Include the anchor → first-node hop so short segments aren't zero.
    parent_idx = topo.parent_idx[seg[0]]
    if parent_idx is not None:
        path_len += _euclid(nodes[parent_idx], nodes[seg[0]])

    # Radial / euclidean distances from soma.
    radial_head = _euclid(head, soma)
    radial_tail = _euclid(tail, soma)
    radial_max = max(radial_head, radial_tail)
    z_above = (tail.z - soma.z)

    # Radius statistics.
    radii = np.array([nodes[i].radius for i in seg], dtype=np.float64)
    mean_r = float(radii.mean())
    min_r = float(radii.min())
    max_r = float(radii.max())
    # Taper = (head_r - tail_r) / max(eps, head_r)
    head_r = float(nodes[seg[0]].radius)
    tail_r = float(nodes[seg[-1]].radius)
    taper = (head_r - tail_r) / max(1e-6, head_r)

    # Downstream branch count (proxy for "branchiness" / partition mass).
    sub = _subtree_indices(topo, seg[-1])
    n_subtree_nodes = len(sub)
    n_subtree_bifs = sum(1 for i in sub if len(topo.children[i]) >= 2)

    # Total length of the arbor below the branch tail: every edge in its
    # subtree, summed.
    sub_path = 0.0
    for i in sub:
        pi = topo.parent_idx[i]
        if pi is None:
            continue
        sub_path += _euclid(nodes[pi], nodes[i])

    # Partition asymmetry: |L - R| / (L + R) on the bigger sub-branches at tail.
    kids = topo.children[seg[-1]]
    if len(kids) >= 2:
        sizes = [len(_subtree_indices(topo, k)) for k in kids]
        sizes.sort(reverse=True)
        a, b = sizes[0], sizes[1]
        partition_asym = abs(a - b) / max(1, (a + b))
    else:
        partition_asym = 0.0

    # Cell type one-hot (2 features).
    is_pyr = 1.0 if cell_type == "pyramidal" else 0.0
    is_int = 1.0 if cell_type == "interneuron" else 0.0

    return np.array([
        path_len, radial_head, radial_tail, radial_max, z_above,
        mean_r, min_r, max_r, head_r, tail_r, taper,
        n_subtree_nodes, n_subtree_bifs, sub_path, partition_asym,
        is_pyr, is_int,
    ], dtype=np.float64)


def _branch_majority_label(topo: _Topo, seg: list[int]) -> int:
    """Majority neurite label among branch nodes (excluding soma)."""
    counts: dict[int, int] = {}
    for i in seg:
        t = int(topo.nodes[i].type)
        if t == 1:
            continue  # ignore soma nodes inside a non-soma branch
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return 3  # default basal
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _extract_branch_dataset(
    files_by_ct: dict[str, list[Path]],
    label: str,
):
    """Extract (X, y, file_owner) over a file collection."""
    X: list[np.ndarray] = []
    y: list[int] = []
    print(f"  [{label}] extracting branch features over {sum(len(v) for v in files_by_ct.values())} files...")
    for ct, files in files_by_ct.items():
        for f in files:
            try:
                nodes = parse_swc(f)
            except Exception:
                continue
            if not nodes:
                continue
            topo = _build_topo(nodes)
            proxy = _select_proxy_root(topo)
            for seg in _branch_segments(topo, proxy):
                X.append(_branch_neurom_features(topo, proxy, seg, ct))
                y.append(_branch_majority_label(topo, seg))
    return np.vstack(X) if X else np.zeros((0, 17)), np.array(y, dtype=np.int64)


def _train_neurom_clf(train_files: dict[str, list[Path]], seed: int = 42):
    """Train and return just the RF classifier — no closure."""
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    Xtr, ytr = _extract_branch_dataset(train_files, "train")
    Xtr = _sanitize_features(Xtr)
    print(f"  fitting RandomForest on {Xtr.shape[0]} branches, {Xtr.shape[1]} features...")
    # n_jobs=4 (not -1) to bound memory: each parallel worker holds its own
    # copy of the 2M-row training matrix; -1 spawns one per core and OOMs
    # on the corpus. Also cap depth so trees don't blow up.
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=30, n_jobs=4, random_state=seed,
        class_weight="balanced",
    )
    clf.fit(Xtr, ytr)
    return clf


def _make_neurom_predictor(clf):
    """Wrap a fitted NeuroM-RF classifier into a `(nodes, cell_type) -> labels` fn."""
    def fn(nodes, cell_type):
        if not nodes:
            return []
        topo = _build_topo(nodes)
        proxy = _select_proxy_root(topo)
        # Unreached nodes default to basal, never to their ground-truth type:
        # a fragmented reconstruction would otherwise score its own labels.
        out = [1 if nd.type == 1 else 3 for nd in nodes]
        out[proxy] = 1
        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))

        segs = [s for s in _branch_segments(topo, proxy) if s]
        if not segs:
            return out
        X = np.vstack([
            _branch_neurom_features(topo, proxy, seg, cell_type) for seg in segs
        ])
        X = _sanitize_features(X)
        preds = clf.predict(X).astype(int).tolist()
        fallback = 3 if 3 in valid else next(iter(valid - {1}), 3)
        for seg, pred in zip(segs, preds):
            if pred not in valid:
                pred = fallback
            for i in seg:
                if topo.nodes[i].type == 1:
                    continue
                out[i] = pred
        return out
    return fn


# =============================================================================
# BASELINE 2 — Sholl + per-subtree features → RandomForest
# (Emissah/Ascoli 2026 style, but with RF instead of GCN for simplicity.)
# =============================================================================

SHOLL_RADII_UM = (10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0)


def _sholl_subtree_features(
    topo: _Topo,
    proxy: int,
    sub_root: int,
    cell_type: str,
) -> np.ndarray:
    """Sholl + structural features for one primary subtree, 21 in all.

    Mirrors the descriptor set used by Emissah/Tecuatl/Ascoli (2026): Sholl
    intersections at each of the seven radii, then max Sholl, the radius at
    which it peaks, node and bifurcation counts, max radial distance, mean
    radius, total arbor length, max root-to-tip path, the three absolute
    components of the subtree's principal axis, the z extent above and below
    the soma, and a cell-type indicator.
    """
    nodes = topo.nodes
    soma = nodes[proxy]
    sub = _subtree_indices(topo, sub_root)

    # Sholl intersections — count parent-child segments crossing each radius.
    # Each node's distance to the soma is computed once and reused across the
    # radii; recomputing both endpoints per radius costs seven times as much.
    dist_to_soma = {i: _euclid(nodes[i], soma) for i in sub}
    parent_dists = np.fromiter(
        (dist_to_soma[i] for i in sub), dtype=np.float64, count=len(sub)
    )
    edges = []
    for i in sub:
        pi = topo.parent_idx[i]
        if pi is None:
            continue
        d_parent = dist_to_soma.get(pi)
        if d_parent is None:                      # parent outside this subtree
            d_parent = _euclid(nodes[pi], soma)
            dist_to_soma[pi] = d_parent
        edges.append((d_parent, dist_to_soma[i]))
    intersections = [
        sum(1 for d_parent, d_child in edges
            if (d_parent < r <= d_child) or (d_child < r <= d_parent))
        for r in SHOLL_RADII_UM
    ]

    max_sholl = float(max(intersections)) if intersections else 0.0
    peak_r_idx = int(np.argmax(intersections)) if intersections else 0
    peak_r = SHOLL_RADII_UM[peak_r_idx]

    n_sub_nodes = len(sub)
    n_bifs = sum(1 for i in sub if len(topo.children[i]) >= 2)
    max_radial = float(parent_dists.max()) if parent_dists.size else 0.0
    mean_radius = float(np.mean([nodes[i].radius for i in sub]))

    # Total arbor length.
    total_len = 0.0
    for i in sub:
        pi = topo.parent_idx[i]
        if pi is not None:
            total_len += _euclid(nodes[pi], nodes[i])

    # Max root-to-tip path (BFS expanding from sub_root).
    dist_along: dict[int, float] = {sub_root: 0.0}
    stack = [sub_root]
    visited_sub: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in visited_sub:
            continue
        visited_sub.add(idx)
        d_here = dist_along.get(idx, 0.0)
        for c in topo.children[idx]:
            dist_along[c] = d_here + _euclid(nodes[idx], nodes[c])
            stack.append(c)
    max_root_to_tip = float(max(dist_along.values())) if dist_along else 0.0

    # Principal axis (PCA z-component) over subtree node coordinates.
    coords = np.array([(nodes[i].x, nodes[i].y, nodes[i].z) for i in sub], dtype=np.float64)
    if coords.shape[0] >= 2:
        coords -= coords.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(coords, full_matrices=False)
            principal_axis = vt[0]  # 3D
        except np.linalg.LinAlgError:
            principal_axis = np.array([0.0, 0.0, 1.0])
    else:
        principal_axis = np.array([0.0, 0.0, 1.0])
    pa_x, pa_y, pa_z = principal_axis.tolist()

    # Z-extent above soma — strong apical signal in the literature.
    z_max = max((nodes[i].z - soma.z) for i in sub)
    z_min = min((nodes[i].z - soma.z) for i in sub)

    # Cell type indicator.
    is_pyr = 1.0 if cell_type == "pyramidal" else 0.0

    return np.array([
        *intersections,
        max_sholl, peak_r, n_sub_nodes, n_bifs, max_radial, mean_radius,
        total_len, max_root_to_tip,
        abs(pa_x), abs(pa_y), abs(pa_z),
        z_max, z_min,
        is_pyr,  # one feature is enough since cell_type is binary
    ], dtype=np.float64)


def _subtree_majority_label(topo: _Topo, sub_root: int) -> int:
    sub = _subtree_indices(topo, sub_root)
    counts: dict[int, int] = {}
    for i in sub:
        t = int(topo.nodes[i].type)
        if t == 1:
            continue
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return 3
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _extract_subtree_dataset(files_by_ct: dict[str, list[Path]], label: str):
    X: list[np.ndarray] = []
    y: list[int] = []
    print(f"  [{label}] extracting per-subtree Sholl features over {sum(len(v) for v in files_by_ct.values())} files...")
    for ct, files in files_by_ct.items():
        for f in files:
            try:
                nodes = parse_swc(f)
            except Exception:
                continue
            if not nodes:
                continue
            topo = _build_topo(nodes)
            proxy = _select_proxy_root(topo)
            for sub_root in topo.children[proxy]:
                X.append(_sholl_subtree_features(topo, proxy, sub_root, ct))
                y.append(_subtree_majority_label(topo, sub_root))
    n_features = len(SHOLL_RADII_UM) + 14
    X_arr = np.vstack(X) if X else np.zeros((0, n_features))
    return X_arr, np.array(y, dtype=np.int64)


# =============================================================================
# BASELINE 3 — L-Measure-style per-subtree morphometrics + RandomForest
#
# L-Measure (Scorcioni et al. 2008, Nature Protocols) is the canonical
# whole-cell morphometric tool. The actual L-Measure
# binary is Java-based and operates per-cell only, so it can't directly
# produce per-node labels. This baseline computes the L-Measure feature
# *content* per primary subtree using NeuroM-equivalent metrics, trains
# an RF on those features, and propagates the subtree label to every
# node. Same task setup as sholl_rf but with a different feature space:
# no Sholl intersections, just classical L-Measure aggregate
# morphometrics (lengths, diameters, branch counts, asymmetries, etc.).
# =============================================================================

def _lmeasure_subtree_features(
    topo: _Topo,
    proxy: int,
    sub_root: int,
    cell_type: str,
) -> np.ndarray:
    """L-Measure-style 23-dim feature vector for one primary subtree.

    Mirrors L-Measure's standard per-cell metric set (Scorcioni et al.
    2008) computed at the subtree level so it can produce a per-node
    label. No Sholl-derived features — those live in
    ``_sholl_subtree_features`` and ``sholl_rf`` already covers that
    ablation row.
    """
    nodes = topo.nodes
    sub = _subtree_indices(topo, sub_root)
    sub_root_node = nodes[sub_root]
    n_sub_nodes = len(sub)

    # Branch / tip / bifurcation counts (L-Measure: N_bifs, N_tips, N_branch).
    n_bifs = sum(1 for i in sub if len(topo.children[i]) >= 2)
    n_tips = sum(1 for i in sub if len(topo.children[i]) == 0)
    n_branches = max(1, n_bifs * 2 + 1)  # planar-tree identity

    # Total length / surface / volume (L-Measure: Length, Surface, Volume).
    total_length = 0.0
    total_surface = 0.0
    total_volume = 0.0
    for i in sub:
        pi = topo.parent_idx[i]
        if pi is None:
            continue
        seg_len = _euclid(nodes[pi], nodes[i])
        seg_radius = 0.5 * (nodes[pi].radius + nodes[i].radius)
        total_length += seg_len
        total_surface += 2.0 * math.pi * seg_radius * seg_len
        total_volume += math.pi * (seg_radius ** 2) * seg_len

    # Diameter statistics (L-Measure: Diameter mean/min/max).
    radii = np.array([nodes[i].radius for i in sub], dtype=np.float64)
    diameter_mean = float(2.0 * radii.mean())
    diameter_min = float(2.0 * radii.min())
    diameter_max = float(2.0 * radii.max())

    # Bounding-box dimensions (L-Measure: Width, Height, Depth).
    coords = np.array([(nodes[i].x, nodes[i].y, nodes[i].z) for i in sub], dtype=np.float64)
    span = coords.max(axis=0) - coords.min(axis=0)
    bbox_x, bbox_y, bbox_z = float(span[0]), float(span[1]), float(span[2])

    # Euclidean / path distance (L-Measure: EucDistance, PathDistance).
    euc_max = float(np.max(np.linalg.norm(coords - coords[0], axis=1))) if coords.shape[0] >= 2 else 0.0
    dist_along: dict[int, float] = {sub_root: 0.0}
    stack = [sub_root]
    seen: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        d = dist_along.get(idx, 0.0)
        for c in topo.children[idx]:
            dist_along[c] = d + _euclid(nodes[idx], nodes[c])
            stack.append(c)
    path_max = float(max(dist_along.values())) if dist_along else 0.0

    # Branch order — max depth from sub_root, treating bifurcations as +1.
    order_at: dict[int, int] = {sub_root: 0}
    stack2 = [sub_root]
    seen2: set[int] = set()
    while stack2:
        idx = stack2.pop()
        if idx in seen2:
            continue
        seen2.add(idx)
        d = order_at.get(idx, 0)
        for c in topo.children[idx]:
            inc = 1 if len(topo.children[idx]) >= 2 else 0
            order_at[c] = d + inc
            stack2.append(c)
    max_branch_order = float(max(order_at.values())) if order_at else 0.0

    # Partition asymmetry across all bifurcations (L-Measure: Partition_asymmetry).
    pa_vals: list[float] = []
    for i in sub:
        kids = topo.children[i]
        if len(kids) < 2:
            continue
        sizes = sorted((len(_subtree_indices(topo, k)) for k in kids), reverse=True)
        a, b = sizes[0], sizes[1]
        pa_vals.append(abs(a - b) / max(1, a + b))
    pa_mean = float(np.mean(pa_vals)) if pa_vals else 0.0
    pa_max = float(np.max(pa_vals)) if pa_vals else 0.0

    # Contraction (mean ratio of euclidean to path-length per branch segment).
    contractions: list[float] = []
    for i in sub:
        kids = topo.children[i]
        if len(kids) != 1:
            continue
        # Walk linear segment until next bifurcation/leaf.
        path = 0.0
        cur = i
        while True:
            nxt_kids = topo.children[cur]
            if len(nxt_kids) != 1:
                break
            nxt = nxt_kids[0]
            path += _euclid(nodes[cur], nodes[nxt])
            cur = nxt
        if path > 0:
            euc = _euclid(nodes[i], nodes[cur])
            contractions.append(euc / path)
    contraction_mean = float(np.mean(contractions)) if contractions else 1.0

    # Taper rate (mean fractional radius drop along linear sub-segments).
    parent_rad = nodes[topo.parent_idx[sub_root]].radius if topo.parent_idx[sub_root] is not None else sub_root_node.radius
    tapers: list[float] = []
    for i in sub:
        pi = topo.parent_idx[i]
        if pi is None or len(topo.children[i]) > 1:
            continue
        rp = nodes[pi].radius
        ri = nodes[i].radius
        if rp > 1e-6:
            tapers.append((rp - ri) / rp)
    taper_mean = float(np.mean(tapers)) if tapers else 0.0

    # Cell type one-hot.
    is_pyr = 1.0 if cell_type == "pyramidal" else 0.0

    return np.array([
        float(n_sub_nodes), float(n_bifs), float(n_tips), float(n_branches),
        total_length, total_surface, total_volume,
        diameter_mean, diameter_min, diameter_max,
        bbox_x, bbox_y, bbox_z,
        euc_max, path_max, max_branch_order,
        pa_mean, pa_max, contraction_mean, taper_mean,
        float(parent_rad), float(sub_root_node.radius),
        is_pyr,
    ], dtype=np.float64)


def _extract_lmeasure_dataset(files_by_ct: dict[str, list[Path]], label: str):
    X: list[np.ndarray] = []
    y: list[int] = []
    print(f"  [{label}] extracting per-subtree L-Measure-style features over {sum(len(v) for v in files_by_ct.values())} files...")
    for ct, files in files_by_ct.items():
        for f in files:
            try:
                nodes = parse_swc(f)
            except Exception:
                continue
            if not nodes:
                continue
            topo = _build_topo(nodes)
            proxy = _select_proxy_root(topo)
            for sub_root in topo.children[proxy]:
                X.append(_lmeasure_subtree_features(topo, proxy, sub_root, ct))
                y.append(_subtree_majority_label(topo, sub_root))
    X_arr = np.vstack(X) if X else np.zeros((0, 23))
    return X_arr, np.array(y, dtype=np.int64)


def _sanitize_features(X: np.ndarray) -> np.ndarray:
    """Replace inf/nan with finite values + clip to float32 range.
    Some corpus cells produce overflowing features (very large
    morphologies). sklearn rejects non-finite inputs; clip to keep
    the row but bound its influence."""
    X = np.asarray(X, dtype=np.float64)
    # Replace nan with 0, +/- inf with large finite values
    X = np.nan_to_num(X, nan=0.0, posinf=1e30, neginf=-1e30)
    # Clip to a safe range (well inside float32 max ~3.4e38)
    return np.clip(X, -1e30, 1e30).astype(np.float32)


def _train_lmeasure_clf(train_files: dict[str, list[Path]], seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    Xtr, ytr = _extract_lmeasure_dataset(train_files, "train")
    Xtr = _sanitize_features(Xtr)
    print(f"  fitting RandomForest on {Xtr.shape[0]} subtrees, {Xtr.shape[1]} features...")
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=seed,
        class_weight="balanced",
    )
    clf.fit(Xtr, ytr)
    return clf


def _make_lmeasure_predictor(clf):
    """Wrap a fitted L-Measure-style classifier into a per-node predictor."""
    def fn(nodes, cell_type):
        if not nodes:
            return []
        topo = _build_topo(nodes)
        proxy = _select_proxy_root(topo)
        # Unreached nodes default to basal, never to their ground-truth type:
        # a fragmented reconstruction would otherwise score its own labels.
        out = [1 if nd.type == 1 else 3 for nd in nodes]
        out[proxy] = 1
        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))

        sub_roots = list(topo.children[proxy])
        if not sub_roots:
            return out
        X = np.vstack([
            _lmeasure_subtree_features(topo, proxy, sub_root, cell_type)
            for sub_root in sub_roots
        ])
        X = _sanitize_features(X)
        preds = clf.predict(X).astype(int).tolist()
        fallback = 3 if 3 in valid else next(iter(valid - {1}), 3)
        for sub_root, pred in zip(sub_roots, preds):
            if pred not in valid:
                pred = fallback
            for i in _subtree_indices(topo, sub_root):
                if topo.nodes[i].type == 1:
                    continue
                out[i] = pred
        return out
    return fn


def _make_sholl_predictor(clf):
    """Wrap a fitted classifier into a `(nodes, cell_type) -> labels` fn.
    Batches all primary-subtree features per cell into one predict() call."""
    def fn(nodes, cell_type):
        if not nodes:
            return []
        topo = _build_topo(nodes)
        proxy = _select_proxy_root(topo)
        # Unreached nodes default to basal, never to their ground-truth type:
        # a fragmented reconstruction would otherwise score its own labels.
        out = [1 if nd.type == 1 else 3 for nd in nodes]
        out[proxy] = 1
        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))

        sub_roots = list(topo.children[proxy])
        if not sub_roots:
            return out
        X = np.vstack([
            _sholl_subtree_features(topo, proxy, sub_root, cell_type)
            for sub_root in sub_roots
        ])
        X = _sanitize_features(X)
        preds = clf.predict(X).astype(int).tolist()
        fallback = 3 if 3 in valid else next(iter(valid - {1}), 3)
        for sub_root, pred in zip(sub_roots, preds):
            if pred not in valid:
                pred = fallback
            for i in _subtree_indices(topo, sub_root):
                if topo.nodes[i].type == 1:
                    continue
                out[i] = pred
        return out
    return fn


def _train_sholl_rf_clf(train_files: dict[str, list[Path]], seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    Xtr, ytr = _extract_subtree_dataset(train_files, "train")
    Xtr = _sanitize_features(Xtr)
    print(f"  fitting RandomForest on {Xtr.shape[0]} subtrees, {Xtr.shape[1]} features...")
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=seed,
        class_weight="balanced",
    )
    clf.fit(Xtr, ytr)
    return clf


def _train_sholl_mlp_pipe(train_files: dict[str, list[Path]], seed: int = 42):
    """Sholl features into a small MLP, mirroring the Emissah/Ascoli architecture
    style more directly than the forest does. It uses sklearn's MLPClassifier so
    this baseline does not pull in torch: their MLP is compact and the feature
    dimension is small enough that sklearn's suffices."""
    from sklearn.neural_network import MLPClassifier  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415

    Xtr, ytr = _extract_subtree_dataset(train_files, "train")
    Xtr = _sanitize_features(Xtr)
    print(f"  fitting MLP on {Xtr.shape[0]} subtrees, {Xtr.shape[1]} features...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # solver, learning rate and validation fraction are sklearn's own
        # defaults, set here so the published specification does not depend on
        # a library default that may change.
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", learning_rate_init=1e-3,
            max_iter=300, early_stopping=True, validation_fraction=0.1,
            random_state=seed,
        )),
    ])
    pipe.fit(Xtr, ytr)
    return pipe


# =============================================================================
# CACHED training: save sklearn classifiers once, load thereafter
# =============================================================================

# (method_name) -> (trainer, wrapper)
# --------------------------------------------------------------------------
# Training-free floor baselines
# --------------------------------------------------------------------------

def _train_majority_stub(train_files, seed: int = 42):
    """Nothing to fit; the cache stores a marker for interface parity."""
    return "majority"


def _make_majority_predictor(_clf):
    """Every neurite node becomes basal dendrite (3).

    This is the labeling a laboratory that draws no compartment distinction
    produces, so it anchors the bottom of the metric scale: any method must
    beat it to be assigning information at all.
    """
    def fn(nodes, cell_type):
        return [1 if nd.type == 1 else 3 for nd in nodes]
    return fn


def _train_rules_stub(train_files, seed: int = 42):
    return "rules_only"


def _make_rules_predictor(_clf):
    """The anatomy heuristics with no learning.

    Per primary subtree: the thinnest (mean radius, among subtrees of at least
    10 nodes) becomes the axon; among the rest, the one reaching farthest from
    the soma becomes the apical (pyramidal only); everything else is basal.
    This isolates what the learned stages add over the geometric priors alone.
    """
    def fn(nodes, cell_type):
        if not nodes:
            return []
        topo = _build_topo(nodes)
        proxy = _select_proxy_root(topo)
        out = [1 if nd.type == 1 else 3 for nd in nodes]
        out[proxy] = 1
        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
        sub_roots = list(topo.children[proxy])
        if not sub_roots:
            return out
        soma = nodes[proxy]
        stats = []
        for r in sub_roots:
            idx = _subtree_indices(topo, r)
            mean_radius = float(np.mean([nodes[i].radius for i in idx]))
            extent = max(_euclid(nodes[i], soma) for i in idx)
            stats.append({"root": r, "idx": idx, "radius": mean_radius,
                          "extent": extent, "n": len(idx)})
        assign = {s["root"]: 3 for s in stats}
        if 2 in valid:
            pool = [s for s in stats if s["n"] >= 10] or stats
            assign[min(pool, key=lambda s: s["radius"])["root"]] = 2
        if 4 in valid:
            pool = [s for s in stats if assign[s["root"]] == 3]
            if pool:
                assign[max(pool, key=lambda s: s["extent"])["root"]] = 4
        for s in stats:
            lab = assign[s["root"]]
            for i in s["idx"]:
                if topo.nodes[i].type == 1:
                    continue
                out[i] = lab
        return out
    return fn


_CACHE_REGISTRY: dict[str, tuple[callable, callable]] = {
    "neurom_rf":   (_train_neurom_clf,    _make_neurom_predictor),
    "lmeasure_rf": (_train_lmeasure_clf,  _make_lmeasure_predictor),
    "sholl_rf":    (_train_sholl_rf_clf,  _make_sholl_predictor),
    "sholl_mlp":   (_train_sholl_mlp_pipe, _make_sholl_predictor),
    "majority":    (_train_majority_stub,  _make_majority_predictor),
    "rules_only":  (_train_rules_stub,     _make_rules_predictor),
}


def predict_with_cache(
    method_name: str,
    train_files: dict[str, list[Path]],
    seed: int = 42,
    cache_path: Path | None = None,
    force_retrain: bool = False,
):
    """Train method (or load from cache), return its (nodes, cell_type) -> labels fn.

    cache_path: where to pickle/load the fitted classifier. None disables caching.
    force_retrain: ignore the cache and re-train.
    """
    import joblib
    trainer, wrapper = _CACHE_REGISTRY[method_name]
    # Each file is keyed by its cell-type bucket as well as its name, so two
    # training sets cannot share a digest by holding the same file names under
    # different cell types. Bucket and name are used rather than the absolute
    # path, which would tie a cache to one machine.
    def _key(names: list[str], version: int) -> dict:
        d = {"method": method_name, "seed": int(seed), "n_train": len(names),
             "train_digest": hashlib.md5("|".join(names).encode()).hexdigest()}
        if version > 1:
            d["key_version"] = version
        return d

    key = _key(sorted(f"{ct}/{Path(f).name}"
                      for ct, files in train_files.items() for f in files), 2)
    # The key names each training file with its cell-type bucket. A key over the
    # bare file names identifies the same training set, so it is accepted too.
    bare_key = _key(sorted(Path(f).name for v in train_files.values() for f in v), 1)
    if cache_path and cache_path.is_file() and not force_retrain:
        blob = joblib.load(cache_path)
        if isinstance(blob, dict) and "model" in blob:
            cached = blob.get("key")
            if cached not in (key, bare_key):
                raise SystemExit(
                    f"{cache_path} was fitted on a different training set than this "
                    f"run needs.\n  cached: {cached}\n  needed: {key}\n"
                    f"Pass --force-retrain, or delete the cache.")
            print(f"  loading cached {method_name} model from {cache_path.name}...")
            return wrapper(blob["model"])
        # A cache without a key cannot be checked, so it is not trusted:
        # re-fitting costs less than scoring the wrong model.
        print(f"  {cache_path.name} carries no cache key; re-training.")
    clf = trainer(train_files, seed=seed)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": clf, "key": key}, cache_path)
        print(f"  cached {method_name} model -> {cache_path.name}")
    return wrapper(clf)
