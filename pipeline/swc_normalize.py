"""SWC normalization — two transforms applied before features/training.

  1. ``normalize_custom_types``  — rewrite non-standard SWC types
     (anything outside {1, 2, 3, 4}) to the dominant standard type of
     the branch the node sits on. Absorbs custom sub-cellular
     annotations (type 10 = AIS, type 11 = spines, etc.) into the
     parent neurite class they belong to.

  2. ``consolidate_multi_point_soma`` — collapse connected groups of
     type-1 soma nodes into a SINGLE anchor node. Multi-point somas
     (3-point, 5-point, or 50-node cloud representations of the cell
     body) confuse the per-node F1 metric: the model labels exactly
     one node as soma per cell, so multi-node GT somas show artificial
     low soma recall. Algorithm:
       - For each connected type-1 component: anchor = root (parent==-1)
         or first in component; centroid = mean(x,y,z); radius = that of
         the sphere whose surface equals the surface the component's own
         points describe.
       - Anchor is updated to (centroid, radius, parent=-1).
       - Non-anchor soma nodes are removed; any non-soma child whose
         parent was a removed soma is rewired to the anchor's id.

     The radius inverts the three-point convention rather than measuring
     the points as a cloud. Those three points exist so that a cylinder
     surface calculation returns 4*pi*r^2, the sphere of radius r, so
     taking sqrt(area / 4pi) gives back exactly r. A single-point soma is
     already a sphere and keeps its radius; a contour or a cloud becomes
     the sphere of the same surface. This is what L-Measure reports as
     Soma_Surface, so the consolidated node agrees with the standard tool.

The combined ``normalize_swc`` wrapper applies (1) then (2). ``parse_swc``
in pipeline.features uses this wrapper by default.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .features import SWCNode

#: The four canonical SWC codes: soma, axon, basal dendrite, apical dendrite.
#: Anything outside this set is a custom annotation and gets rewritten.
STANDARD_TYPES = {1, 2, 3, 4}


def normalize_custom_types(nodes: list[SWCNode]) -> tuple[list[SWCNode], int]:
    """Return (normalized_nodes, n_rewritten).

    Rule:
      For each node whose type is not in {1, 2, 3, 4}:
        1. Find the branch the node belongs to (contiguous run between
           bifurcations).
        2. Replace its type with the most common standard type in that
           branch.
        3. If the branch contains no standard-type nodes at all, walk up
           parent links until a standard-type ancestor is found; use that.
        4. If neither succeeds, set type to 0, the SWC code for undefined.
           That happens only when a custom-typed node has no standard type
           anywhere in its branch and no standard ancestor, so nothing in
           the file says what it is. Such a node is not a neurite to any
           later stage, and a file left with no neurite at all is rejected
           by the Stage 0 gate.

    Standard types (1=soma, 2=axon, 3=basal/dendrite, 4=apical) are NEVER
    touched.
    """
    n = len(nodes)
    if n == 0:
        return nodes, 0

    # Build topology
    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    children: list[list[int]] = [[] for _ in range(n)]
    parent_idx: list[int] = [-1] * n
    for i, nd in enumerate(nodes):
        if nd.parent != -1 and nd.parent in id_to_idx:
            p = id_to_idx[nd.parent]
            children[p].append(i)
            parent_idx[i] = p

    # A node starts a new branch if:
    #   it has no parent (root), OR
    #   its parent has more than one child (bifurcation)
    # Branches are not split on type change: custom-type nodes are
    # absorbed into their branch.
    starts = [False] * n
    for i in range(n):
        p = parent_idx[i]
        if p < 0:
            starts[i] = True
        elif len(children[p]) > 1:
            starts[i] = True

    # Parent-first traversal so branch ids propagate correctly
    order: list[int] = []
    seen = [False] * n
    queue = [i for i in range(n) if parent_idx[i] < 0]
    while queue:
        nq: list[int] = []
        for i in queue:
            if seen[i]:
                continue
            seen[i] = True
            order.append(i)
            nq.extend(children[i])
        queue = nq

    # Assign each node to a branch id (= the start index it descends from)
    branch_id: list[int] = [-1] * n
    for i in order:
        if starts[i]:
            branch_id[i] = i
        else:
            branch_id[i] = branch_id[parent_idx[i]]

    # Group nodes by branch
    branch_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        branch_to_idxs[branch_id[i]].append(i)

    # Determine dominant standard type per branch
    branch_dominant: dict[int, int | None] = {}
    for bid, idxs in branch_to_idxs.items():
        std = [nodes[i].type for i in idxs if nodes[i].type in STANDARD_TYPES]
        branch_dominant[bid] = Counter(std).most_common(1)[0][0] if std else None

    # Rewrite custom types
    new_types = [nd.type for nd in nodes]
    n_rewritten = 0
    for i, nd in enumerate(nodes):
        if nd.type in STANDARD_TYPES:
            continue
        # First try branch dominant
        dom = branch_dominant[branch_id[i]]
        if dom is not None:
            new_types[i] = dom
            n_rewritten += 1
            continue
        # Else walk up the parent chain
        j = parent_idx[i]
        while j >= 0 and nodes[j].type not in STANDARD_TYPES:
            j = parent_idx[j]
        if j >= 0:
            new_types[i] = nodes[j].type
            n_rewritten += 1
        else:
            new_types[i] = 0
            n_rewritten += 1

    out = [
        SWCNode(
            id=nd.id, type=new_types[i],
            x=nd.x, y=nd.y, z=nd.z, radius=nd.radius, parent=nd.parent,
        )
        for i, nd in enumerate(nodes)
    ]
    return out, n_rewritten


def count_custom_types(nodes: list[SWCNode]) -> dict[int, int]:
    """Return a {type_value: count} dict of non-standard type occurrences."""
    out: dict[int, int] = {}
    for nd in nodes:
        if nd.type not in STANDARD_TYPES:
            out[nd.type] = out.get(nd.type, 0) + 1
    return out


def consolidate_multi_point_soma(
    nodes: list[SWCNode],
) -> tuple[list[SWCNode], dict]:
    """Collapse connected groups of type-1 soma nodes into one anchor node.

    Returns ``(new_nodes, info)`` where ``info`` reports:
        soma_count_before / soma_count_after  — number of type-1 nodes
        group_count                            — number of soma components
        complex_group_count                    — components with >1 node
        removed_nodes                          — total nodes deleted
        changed                                — True iff any soma collapsed

    Topology guarantee: surviving node IDs are preserved exactly; only
    non-anchor soma nodes are deleted, and any child pointing at a
    deleted soma is rewired to that group's anchor ID.

    The anchor is made a root: its parent becomes -1 whatever it was. On a
    cell whose soma is already the root, which is every cell in the corpus,
    that changes nothing. On a file whose soma hangs off a neurite it cuts
    that link, leaving what was above the soma as a second root, which the
    Stage 0 gate then rejects as not one connected cell.
    """
    n = len(nodes)
    if n == 0:
        return nodes, {
            "soma_count_before": 0, "soma_count_after": 0,
            "group_count": 0, "complex_group_count": 0,
            "removed_nodes": 0, "changed": False,
        }

    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    parent_idx = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        if nd.parent != -1 and nd.parent in id_to_idx:
            p = id_to_idx[nd.parent]
            parent_idx[i] = p
            children[p].append(i)

    soma_idxs = [i for i, nd in enumerate(nodes) if nd.type == 1]
    if not soma_idxs:
        return nodes, {
            "soma_count_before": 0, "soma_count_after": 0,
            "group_count": 0, "complex_group_count": 0,
            "removed_nodes": 0, "changed": False,
        }
    soma_idx_set = set(soma_idxs)

    # BFS within the type-1 subgraph to find connected components.
    visited: set[int] = set()
    groups: list[list[int]] = []
    for start in soma_idxs:
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        visited.add(start)
        while stack:
            idx = stack.pop()
            component.append(idx)
            p = parent_idx[idx]
            if p in soma_idx_set and p not in visited:
                visited.add(p); stack.append(p)
            for c in children[idx]:
                if c in soma_idx_set and c not in visited:
                    visited.add(c); stack.append(c)
        groups.append(sorted(component))

    # For each group, pick anchor + compute centroid + mega-radius.
    # anchor_map: original_soma_id -> anchor_id (for rewiring children).
    anchor_map: dict[int, int] = {}
    keep_mask = [True] * n
    new_xyz: dict[int, tuple[float, float, float]] = {}
    new_radius: dict[int, float] = {}
    new_parent_for_anchor: dict[int, int] = {}
    complex_count = 0

    for group in groups:
        # Prefer the actual root (parent==-1) as anchor.
        anchor_local = next((i for i in group if parent_idx[i] < 0), group[0])
        anchor_id = nodes[anchor_local].id

        # Centroid of all soma nodes in this component.
        xs = [nodes[i].x for i in group]
        ys = [nodes[i].y for i in group]
        zs = [nodes[i].z for i in group]
        cx = sum(xs) / len(group)
        cy = sum(ys) / len(group)
        cz = sum(zs) / len(group)

        # The radius of the sphere whose surface equals the surface the group's
        # own points describe. The three-point convention exists to make a
        # cylinder-based surface come out at 4*pi*r^2, the sphere of radius r, so
        # inverting that gives back exactly r. A single point is already a
        # sphere. A contour or cloud gets the sphere of the same surface.
        if len(group) == 1:
            mega_r = max(float(nodes[anchor_local].radius), 0.0)
        else:
            in_group = set(group)
            area = 0.0
            for i in group:
                p = parent_idx[i]
                if p < 0 or p not in in_group:
                    continue
                a, b = nodes[i], nodes[p]
                L = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
                ra, rb = max(float(a.radius), 0.0), max(float(b.radius), 0.0)
                area += math.pi * (ra + rb) * ((L * L + (ra - rb) ** 2) ** 0.5)
            mega_r = (area / (4.0 * math.pi)) ** 0.5 if area > 0 else max(
                float(nodes[anchor_local].radius), 0.0)

        new_xyz[anchor_local] = (cx, cy, cz)
        new_radius[anchor_local] = mega_r
        new_parent_for_anchor[anchor_local] = -1

        for i in group:
            anchor_map[nodes[i].id] = anchor_id
            if i != anchor_local:
                keep_mask[i] = False

        if len(group) > 1:
            complex_count += 1

    # Build the output list: anchors updated, non-anchor somas dropped,
    # children of dropped somas rewired to anchor.
    out: list[SWCNode] = []
    for i, nd in enumerate(nodes):
        if not keep_mask[i]:
            continue
        if i in new_xyz:
            cx, cy, cz = new_xyz[i]
            out.append(SWCNode(
                id=nd.id, type=1,
                x=float(cx), y=float(cy), z=float(cz),
                radius=float(new_radius[i]),
                parent=int(new_parent_for_anchor[i]),
            ))
            continue
        # Non-soma surviving node — rewire parent if it pointed at a dropped soma.
        new_parent = nd.parent
        if nd.parent != -1 and nd.parent in anchor_map:
            new_parent = anchor_map[nd.parent]
        out.append(SWCNode(
            id=nd.id, type=nd.type,
            x=nd.x, y=nd.y, z=nd.z, radius=nd.radius,
            parent=new_parent,
        ))

    n_after_soma = sum(1 for nd in out if nd.type == 1)
    info = {
        "soma_count_before":   len(soma_idxs),
        "soma_count_after":    n_after_soma,
        "group_count":         len(groups),
        "complex_group_count": complex_count,
        "removed_nodes":       n - len(out),
        "changed":             complex_count > 0,
    }
    return out, info


def normalize_swc(
    nodes: list[SWCNode],
) -> tuple[list[SWCNode], dict]:
    """Apply both normalization steps in the canonical order.

    1. Rewrite non-standard SWC types into {1, 2, 3, 4}.
    2. Consolidate connected multi-point soma components into a single
       anchor soma node per component.

    Used by ``pipeline.features.parse_swc`` when ``normalize_types=True``.
    Returns ``(nodes, info)`` where info combines the diagnostics from
    both passes.
    """
    n0 = len(nodes)
    nodes, n_type_rewrites = normalize_custom_types(nodes)
    nodes, soma_info = consolidate_multi_point_soma(nodes)
    return nodes, {
        "n_input":            n0,
        "n_output":           len(nodes),
        "n_type_rewrites":    n_type_rewrites,
        "soma":               soma_info,
    }
