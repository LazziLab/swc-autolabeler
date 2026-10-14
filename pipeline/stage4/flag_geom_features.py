"""Geometry + anatomy per-cell features for the pyramidal flag model.

The anatomy half of the flag's feature row, for already-labeled PYRAMIDAL
cells. ``extract_features`` takes one cell's predicted node labels, per-node
confidences, and raw geometry (as arrays) and returns the deployment-safe
anatomical block; the flag then ranks cells by predicted badness. Everything is
computed from PREDICTED labels, confidences, coordinates, and topology -- never
ground-truth labels. The training target (bad = per-cell F1 < a threshold) is
used only as the downstream label.

The design bet: a badly-labeled pyramidal cell violates pyramidal anatomy in
ways visible from predictions + geometry alone -- no/fragmented/incoherent
apical, apical and basal spatially intermixed, or noisy label churn along
branches. Plain confidence statistics are the other half of the row and belong
to ``flag_features``; this module deliberately does not repeat them, since both
blocks reach the model and a repeat would ship as two identical columns.

Label ids: 1 soma, 2 axon, 3 basal, 4 apical.
"""
from __future__ import annotations

import numpy as np

CLASS_IDS = {"axon": 2, "basal": 3, "apical": 4}


def _build_topo(node_id: np.ndarray, parent: np.ndarray):
    n = len(node_id)
    id_to_idx = {int(node_id[i]): i for i in range(n)}
    parent_idx = np.full(n, -1, dtype=np.int64)
    children = [[] for _ in range(n)]
    for i in range(n):
        p = int(parent[i])
        if p != -1 and p in id_to_idx:
            pi = id_to_idx[p]
            parent_idx[i] = pi
            children[pi].append(i)
    return parent_idx, children


def _n_components(mask: np.ndarray, parent_idx: np.ndarray, children) -> int:
    """Connected components among masked nodes, edges = parent links both-masked."""
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return 0
    seen = np.zeros(len(mask), dtype=bool)
    comps = 0
    mset = set(int(i) for i in idxs)
    for start in idxs:
        if seen[start]:
            continue
        comps += 1
        stack = [int(start)]
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            pu = int(parent_idx[u])
            if pu in mset and not seen[pu]:
                stack.append(pu)
            for c in children[u]:
                if c in mset and not seen[c]:
                    stack.append(c)
    return comps


def _largest_component(mask: np.ndarray, parent_idx: np.ndarray, children) -> int:
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return 0
    seen = np.zeros(len(mask), dtype=bool)
    mset = set(int(i) for i in idxs)
    best = 0
    for s in idxs:
        if seen[s]:
            continue
        sz, stack = 0, [int(s)]
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            sz += 1
            pu = int(parent_idx[u])
            if pu in mset and not seen[pu]:
                stack.append(pu)
            for c in children[u]:
                if c in mset and not seen[c]:
                    stack.append(c)
        best = max(best, sz)
    return best


def _geom(coords: np.ndarray, mask: np.ndarray, soma_c: np.ndarray) -> dict:
    """Geometry of a predicted class relative to the soma."""
    pts = coords[mask]
    out = {"n": int(mask.sum())}
    if pts.shape[0] < 2:
        return {**out, "dist": 0.0, "spread": 0.0, "coherence": 0.0,
                "elong": 0.0, "extent": 0.0}
    rel = pts - soma_c
    node_dist = np.linalg.norm(rel, axis=1)
    centroid = rel.mean(axis=0)
    cdist = float(np.linalg.norm(centroid))
    mean_dist = float(node_dist.mean()) + 1e-6
    out["dist"] = cdist                              # centroid distance from soma
    out["spread"] = float(node_dist.std())
    out["coherence"] = cdist / mean_dist             # 1 = all one direction, 0 = around soma
    # principal-axis elongation
    cov = np.cov(rel.T)
    try:
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        ev = np.clip(ev, 1e-9, None)
        out["elong"] = float(np.sqrt(ev[0] / ev[1])) if ev[1] > 0 else 0.0
    except Exception:
        out["elong"] = 0.0
    out["extent"] = float(node_dist.max())
    return out


def extract_features(pred, conf, x, y, z, radius, node_id, parent,
                     stage1_pred, stage1_conf) -> dict:
    pred = np.asarray(pred, dtype=int)
    conf = np.asarray(conf, dtype=float)
    coords = np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)], axis=1)
    n = len(pred)
    parent_idx, children = _build_topo(np.asarray(node_id), np.asarray(parent))

    soma_mask = pred == 1
    soma_c = coords[soma_mask].mean(axis=0) if soma_mask.any() else coords.mean(axis=0)

    f: dict = {}
    non_soma = max(1, int((pred != 1).sum()))
    f["n_nodes"] = n
    f["log_n_nodes"] = float(np.log1p(n))
    f["stage1_is_interneuron"] = 1.0 if str(stage1_pred) == "interneuron" else 0.0

    # ---- predicted composition ----
    for name, cid in CLASS_IDS.items():
        m = pred == cid
        f[f"frac_{name}"] = float(m.sum()) / non_soma
        f[f"has_{name}"] = 1.0 if m.any() else 0.0
    f["pred_class_count"] = float(sum(f[f"has_{k}"] for k in CLASS_IDS))
    f["apical_zero"] = 1.0 - f["has_apical"]          # no apical on a pyramidal -> suspicious
    f["axon_zero"] = 1.0 - f["has_axon"]

    # Per-cell and per-class confidence statistics are NOT computed here:
    # ``flag_features._confidence_summary`` already produces them, and both
    # blocks reach the model, so anything repeated here would ship twice.

    # ---- anatomy / geometry per class ----
    geoms = {name: _geom(coords, pred == cid, soma_c) for name, cid in CLASS_IDS.items()}
    for name, g in geoms.items():
        for k, v in g.items():
            if k == "n":
                continue
            f[f"{name}_{k}"] = v

    # ---- apical structural integrity (the key pyramidal failure mode) ----
    apical_mask = pred == 4
    basal_mask = pred == 3
    # In a tree the components of a node subset and the count of nodes whose
    # parent is outside it are the same number, so the apical trunk count is
    # exactly ``apical_components`` and is not emitted separately.
    f["apical_components"] = float(_n_components(apical_mask, parent_idx, children))
    f["apical_frag"] = f["apical_components"] / max(1.0, geoms["apical"]["n"]) if geoms["apical"]["n"] else 0.0
    f["basal_components"] = float(_n_components(basal_mask, parent_idx, children))

    # apical-basal spatial separation (bad labeling -> they overlap)
    if apical_mask.sum() >= 2 and basal_mask.sum() >= 2:
        ac = coords[apical_mask].mean(axis=0)
        bc = coords[basal_mask].mean(axis=0)
        f["apical_basal_centroid_dist"] = float(np.linalg.norm(ac - bc))
        aspread = float(np.linalg.norm(coords[apical_mask] - ac, axis=1).mean()) + 1e-6
        bspread = float(np.linalg.norm(coords[basal_mask] - bc, axis=1).mean()) + 1e-6
        f["apical_basal_separation"] = f["apical_basal_centroid_dist"] / (aspread + bspread)
        # direction agreement: apical & basal should point OPPOSITE ways from soma
        # (dot ~ -1 normal; ~ +1 same direction -> likely a swap/mislabel)
        a_dir, b_dir = ac - soma_c, bc - soma_c
        na, nb = np.linalg.norm(a_dir), np.linalg.norm(b_dir)
        f["apical_basal_dir_dot"] = float(np.dot(a_dir, b_dir) / (na * nb)) if na > 1e-6 and nb > 1e-6 else 0.0
        # anatomical priors: apical should extend farther and be more elongated than basal
        f["apical_minus_basal_dist"] = geoms["apical"]["dist"] - geoms["basal"]["dist"]
        f["apical_minus_basal_extent"] = geoms["apical"]["extent"] - geoms["basal"]["extent"]
        f["apical_minus_basal_elong"] = geoms["apical"]["elong"] - geoms["basal"]["elong"]
    else:
        f["apical_basal_centroid_dist"] = 0.0
        f["apical_basal_separation"] = 0.0
        f["apical_basal_dir_dot"] = 0.0
        f["apical_minus_basal_dist"] = 0.0
        f["apical_minus_basal_extent"] = 0.0
        f["apical_minus_basal_elong"] = 0.0

    # ---- label churn along branches (messy labeling -> many transitions) ----
    has_parent = parent_idx >= 0
    pi = parent_idx[has_parent]
    child_lab = pred[has_parent]
    par_lab = pred[pi]
    neurite = (child_lab != 1) & (par_lab != 1)
    f["label_transition_frac"] = float((child_lab[neurite] != par_lab[neurite]).mean()) if neurite.any() else 0.0
    # isolated nodes: label differs from parent AND from all children
    iso = 0
    for i in range(n):
        if pred[i] == 1:
            continue
        p = int(parent_idx[i])
        neigh = [pred[p]] if p >= 0 else []
        neigh += [pred[c] for c in children[i]]
        if neigh and all(nl != pred[i] for nl in neigh):
            iso += 1
    f["isolated_node_frac"] = iso / non_soma

    # ---- localized low-confidence structure (a hidden bad region aggregates miss) ----
    for thr in (0.6, 0.7, 0.8):
        lowc = conf < thr
        big = _largest_component(lowc, parent_idx, children)
        f[f"lowconf{int(thr*100)}_cluster_frac"] = big / max(1, n)
    # worst local window: min over nodes of the mean confidence in [self, parent, children]
    local_means = np.empty(n, dtype=np.float32)
    for i in range(n):
        vals = [conf[i]]
        p = int(parent_idx[i])
        if p >= 0:
            vals.append(conf[p])
        for c in children[i]:
            vals.append(conf[c])
        local_means[i] = np.mean(vals)
    neur = pred != 1
    if neur.any():
        f["worst_local_conf"] = float(local_means[neur].min())
        f["worst_local_conf_p05"] = float(np.percentile(local_means[neur], 5))
    else:
        f["worst_local_conf"] = 1.0
        f["worst_local_conf_p05"] = 1.0

    # ---- apical spatial outliers + basal/apical spatial confusion ----
    am, bm = pred == 4, pred == 3
    if am.sum() >= 5:
        ac = coords[am].mean(axis=0)
        ad = np.linalg.norm(coords[am] - ac, axis=1)
        f["apical_outlier_frac"] = float((ad > ad.mean() + 2 * ad.std()).mean())
        f["apical_spread_ratio"] = float(ad.std() / (ad.mean() + 1e-6))
    else:
        f["apical_outlier_frac"] = 0.0
        f["apical_spread_ratio"] = 0.0
    if am.sum() >= 5 and bm.sum() >= 5:
        ac, bc = coords[am].mean(axis=0), coords[bm].mean(axis=0)
        f["basal_closer_to_apical_frac"] = float(
            (np.linalg.norm(coords[bm] - ac, axis=1) < np.linalg.norm(coords[bm] - bc, axis=1)).mean())
        f["apical_closer_to_basal_frac"] = float(
            (np.linalg.norm(coords[am] - bc, axis=1) < np.linalg.norm(coords[am] - ac, axis=1)).mean())
    else:
        f["basal_closer_to_apical_frac"] = 0.0
        f["apical_closer_to_basal_frac"] = 0.0

    return f
