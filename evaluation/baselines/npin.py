"""NPIN (Su et al. 2021) retrained on this corpus's splits: axon versus dendrite.

NPIN, the node-based polarity identifier of Su, Chou, Huang, Li, Charng, Lo and
Wang (Neuroinformatics 19:669-684, 2021), labels each branch point and tip of a
traced neuron as axon or dendrite from nine geometric features with a gradient-
boosted classifier, then relabels uncertain nodes from their neighbours. Its code
(gitlab.com/czsu32/npin, commit 8d632fe) states no licence, so it is not copied
into this repository: clone it and pass ``--npin-dir``.

    git clone https://gitlab.com/czsu32/npin.git && git -C npin checkout 8d632fe
    pip install "seaborn<0.14"      # NPIN's leaf threshold calls seaborn.distplot
    python -m evaluation.baselines.npin --npin-dir npin --seeds 42,123,789

What runs is NPIN's own code: its tree construction and feature extraction
(Data_Cleaner, Create_Axon_Feature), its training table (pyramid), its classifier
call (classification, with XGBClassifier(seed=123) and the nine features of its
published configuration) and its relabeling (post_relabel). Running it on this
corpus needs the following, and nothing else is changed:

- Input. Each cell is parsed as the pipeline parses it. Basal and apical merge
  into dendrite, since NPIN reads code 4 as a node where axon and dendrite meet;
  the soma group collapses into its root, as in NPIN's unipolar fly data; and the
  file is written parents first with the root numbered 1, which NPIN assumes (it
  measures a path in file order and tests ``== 1`` for the soma).
- Multipolar cells. Where a node has more than two children NPIN keeps the two
  longest and discards the rest, which on a mammalian soma would drop most
  primary neurites; its own branch limit is raised so every child is kept.
- Speed, not results. NPIN materialises every (node, ancestor) pair and scans
  that table once per branch and per tip, which takes hours and tens of
  gigabytes on the corpus's largest cells. Four helpers are replaced by versions
  that walk parent links instead and return the same values, computed with
  NPIN's own distance function in the same order; ``--verify`` runs the original
  helpers beside them and compares every feature.
- Training. One model per cell type on the seed's training split, the ground-
  truth cell type every learned baseline receives. NPIN's evaluation chooses the
  relabel threshold on its test cells; here it is chosen from the same grid,
  0.50 to 0.95, by five-fold cross-validation over the training cells, with
  NPIN's own criterion, tip accuracy.
- Scoring. NPIN labels branch points and tips; every node of a segment takes
  the label of the branch point or tip that ends it, and a node where NPIN's
  relabeling leaves axon and dendrite mixed takes its own classifier call. Every
  non-soma node is scored, basal and apical merged into dendrite, and the
  pipeline, with the ground-truth cell type as in Tables 2 to 4, is scored the
  same way on the same nodes.

Outputs (evaluation/results/baselines/):
    npin_seed{S}.json, npin_summary.{json,txt}, npin_verify.json (--verify)
Work files, about 1 GB, in evaluation/results/baselines/_npin_work/.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pickle
import statistics
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc  # noqa: E402
from pipeline.split import in_test  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "evaluation" / "results" / "baselines"
WORK = OUT / "_npin_work"
CELL_TYPES = ("pyramidal", "interneuron")
NPIN_COMMIT = "8d632fe"
#: NPIN's published configuration (nrn3_class_Classify_Axon.py): four soma
#: features, five local ones, the target node only (pyramid layer 1).
FEATURES = sorted(["s", "ns", "ds", "nds", "l", "nl", "c1", "as", "rc"])
FEATURE_COLS = [f + "_1" for f in FEATURES]
#: The relabel thresholds NPIN's evaluate() sweeps.
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.5, 1, 0.05)]
BRANCH_LIMIT = 10 ** 9
CV_FOLDS = 5
MERGE = {1: 1, 2: 2, 3: 3, 4: 3}


# ---------------------------------------------------------------------------
# NPIN import: compatibility and the four fast helpers
# ---------------------------------------------------------------------------
def _install_compat() -> None:
    """NPIN was written against pandas 1.x, NeuroM 1.x and scikit-learn before
    0.24. Only the interfaces it calls are supplied; none changes a value."""
    if not hasattr(pd.DataFrame, "append"):
        def _append(self, other, ignore_index=False, **_kw):
            if isinstance(other, pd.Series):
                other = other.to_frame().T.infer_objects()
            return pd.concat([self, other], ignore_index=ignore_index)
        pd.DataFrame.append = _append
    if not getattr(pd.DataFrame.drop, "_npin_compat", False):
        _drop = pd.DataFrame.drop

        def _drop_compat(self, labels=None, *args, **kw):
            if args:                                   # drop(labels, 1): positional axis
                kw.setdefault("axis", args[0])
            return _drop(self, labels, **kw)
        _drop_compat._npin_compat = True
        pd.DataFrame.drop = _drop_compat
    import sklearn.neighbors
    sys.modules.setdefault("sklearn.neighbors.kde", sklearn.neighbors)

    class _SWC:
        """NeuroM 1.x ``io.swc.read``: the raw rows as x, y, z, r, type, id, parent."""
        def __init__(self, path):
            rows = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    s = line.split()
                    if not s or s[0].startswith("#"):
                        continue
                    i, t, x, y, z, r, p = s[:7]
                    rows.append([float(x), float(y), float(z), float(r), float(t), float(i), float(p)])
            self.data_block = np.asarray(rows, dtype=float)

    nm = types.ModuleType("neurom")
    nm.io = types.SimpleNamespace(swc=types.SimpleNamespace(read=_SWC))
    sys.modules.setdefault("neurom", nm)

    class _ProgressBar:                                # progress display only
        def __call__(self, it):
            return it
    pb = types.ModuleType("progressbar")
    pb.ProgressBar = _ProgressBar
    sys.modules.setdefault("progressbar", pb)
    # util.py imports TensorFlow for NPIN's neural-network option and never uses
    # it; the XGBoost configuration run here does not need it loaded in every worker.
    sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import matplotlib
    matplotlib.use("Agg")


class _Tree:
    """What NPIN reads from its (df_anc, df_path) pair, without building it.

    NPIN's consumers only ever ask for one node's ancestors or its path to the
    root, in the order parent, grandparent, ..., root, which walking parent
    links gives directly."""

    def __init__(self, df, child_col, parent_col):
        self.ids = df[child_col].to_numpy().tolist()
        self.parent = dict(zip(self.ids, df[parent_col].to_numpy().tolist()))
        seen = set()
        for n in self.ids:                            # parents come first: see _write_input
            p = self.parent[n]
            if p != -1 and p not in seen:
                raise ValueError(f"node {n} precedes its parent {p}; NPIN input must be sorted")
            seen.add(n)

    def path(self, d):
        out = [d]
        p = self.parent[d]
        while p != -1:
            out.append(p)
            p = self.parent[p]
        return out


def _fast_ancestors_and_path(df, child_col="ID", parent_col="PARENT_ID"):
    t = _Tree(df, child_col, parent_col)
    return t, t


def _fast_level_branch(df_path, tree_node_dict):
    """Consecutive level points (root, forks, leaves) along every root-leaf path:
    each level point below the root paired with its nearest level ancestor."""
    t = df_path
    level = set(tree_node_dict["leaf"]) | set(tree_node_dict["fork"]) | set(tree_node_dict["root"])
    nla, branches = {}, set()
    for n in t.ids:
        p = t.parent[n]
        if p == -1:
            continue
        a = p if p in level else nla[p]
        nla[n] = a
        if n in level:
            branches.add((n, a))
    return list(branches)


def _make_fast_levelCol(npin_util):
    def _fast_levelCol(df, df_anc, df_path, tree_node_dict, branch_lst, child_col="ID", parent_col="PARENT_ID"):
        t = df_path
        leaf_lst, fork_lst, root_lst = tree_node_dict["leaf"], tree_node_dict["fork"], tree_node_dict["root"]
        first_fork = npin_util.neuron_first_fork(root_lst, fork_lst, branch_lst)
        # NPIN: level 0 from the root to the first fork, then the number of fork
        # ancestors; that number is 0 exactly on the stretch to the first fork.
        forks = set(fork_lst)
        cnt = {}
        for n in t.ids:
            p = t.parent[n]
            cnt[n] = 0 if p == -1 else cnt[p] + (1 if p in forks else 0)
        df = df.reset_index(drop=True).copy()
        df["level"] = df[child_col].map(cnt).astype("int64")
        max_level = max(df["level"])
        # Deepest level of each point: the largest level of any leaf below it.
        lev = dict(zip(df[child_col].tolist(), df["level"].tolist()))
        leaves = set(leaf_lst)
        best = {n: (lev[n] if n in leaves else 0) for n in t.ids}
        for n in reversed(t.ids):
            p = t.parent[n]
            if p != -1 and best[n] > best[p]:
                best[p] = best[n]
        df["dp_level"] = df[child_col].map(best).astype("int64")
        return df, max_level, first_fork
    return _fast_levelCol


def _make_fast_branchCol(npin_util):
    calc = npin_util.calculate_distance

    def _fast_branchCol_QCol_distance(df, df_path, tree_node_dict, branch_lst, first_fork, decimal=0,
                                      child_col="ID", parent_col="PARENT_ID", type_col="T"):
        from collections import OrderedDict
        t = df_path
        soma = tree_node_dict["root"][0]
        ids = df[child_col].to_numpy().tolist()
        xyz = df.loc[:, ["x", "y", "z"]].values
        pos = {n: tuple(xyz[i]) for i, n in enumerate(ids)}

        def step(a, b):                        # NPIN's own step, a earlier in file order
            x1, y1, z1 = pos[a]
            x2, y2, z2 = pos[b]
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

        # Path length to the soma, summed root outward in file order exactly as
        # calculate_distance sums it (Python's sum from 0, one step at a time).
        to_soma = {}
        for n in t.ids:
            p = t.parent[n]
            to_soma[n] = 0 if p == -1 else to_soma[p] + step(p, n)

        def finish(v):
            if decimal is None:
                return v
            if decimal == 0:
                return int(round(v))
            return round(v, decimal)

        length_lst, length_lst_soma, direct_dis_lst_soma = [], [], []
        rows_id, rows_branch, rows_q = [], [], []
        for start, end in branch_lst:
            seg = [start]                      # start, ..., end: the branch in path order
            n = start
            while n != end:
                n = t.parent[n]
                if n == -1:
                    raise ValueError(f"{end} is not an ancestor of {start}")
                seg.append(n)
            keep = seg if (end == 1 and first_fork != 1) else seg[:-1]
            rows_id += keep
            rows_branch += [start] * len(keep)
            rows_q += list(range(len(keep)))
            order = seg[::-1]                  # file order along the branch: end first
            total = 0
            for a, b in zip(order, order[1:]):
                total = total + step(a, b)
            length_lst.append(finish(total))
            length_lst_soma.append(finish(to_soma[start]))
            direct_dis_lst_soma.append(calc([pos[soma], pos[start]], decimal))

        df_temp = pd.DataFrame(OrderedDict({child_col: rows_id, "branch": rows_branch, "Q": rows_q}))
        df = pd.merge(df, df_temp, how="left", on=child_col)
        if first_fork == 1:
            df.loc[df[parent_col] == -1, ["branch", "Q"]] = [1, 0]
        df[["branch", "Q"]] = df[["branch", "Q"]].astype("int")

        df_dis = pd.DataFrame({"branch": branch_lst, "len": length_lst, "len_des_soma": length_lst_soma,
                               "direct_dis_des_soma": direct_dis_lst_soma})
        df_dis["ancestor"] = [tp[1] for tp in branch_lst]
        df_dis["descendant"] = [tp[0] for tp in branch_lst]
        df_dis = df_dis.sort_values(["ancestor", "descendant"]).reset_index(drop=True)
        df_t = df[[child_col, type_col]].copy()
        df_t.columns = ["descendant", "des_T"]
        df_dis = pd.merge(df_dis, df_t, how="left", on="descendant")
        df_dis = df_dis[["branch", "descendant", "ancestor", "len", "len_des_soma", "direct_dis_des_soma", "des_T"]]
        return df, df_dis, length_lst
    return _fast_branchCol_QCol_distance


_ORIGINAL: dict = {}
_HELPERS = ("neuron_ancestors_and_path", "neuron_level_branch", "neuron_levelCol",
            "neuron_branchCol_QCol_distance")


def load_npin(npin_dir: Path, fast: bool = True):
    """Import NPIN from its clone and bind either the fast helpers or NPIN's own
    wherever its modules call them (``from util import *`` copies the names into
    each module, so every copy is set, and set again on every call)."""
    _install_compat()
    d = str(Path(npin_dir).resolve())
    if d not in sys.path:
        sys.path.insert(0, d)
    with contextlib.redirect_stdout(io.StringIO()):
        import util as npin_util
        import nrn3_class_Data_Cleaner as dc
        import nrn3_class_Create_Axon_Feature as caf
    if not _ORIGINAL:
        _ORIGINAL.update({k: getattr(npin_util, k) for k in _HELPERS})
        _ORIGINAL["_fast"] = {"neuron_ancestors_and_path": _fast_ancestors_and_path,
                              "neuron_level_branch": _fast_level_branch,
                              "neuron_levelCol": _make_fast_levelCol(npin_util),
                              "neuron_branchCol_QCol_distance": _make_fast_branchCol(npin_util)}
    use = _ORIGINAL["_fast"] if fast else {k: _ORIGINAL[k] for k in _HELPERS}
    for mod in (npin_util, dc, caf):
        for k, v in use.items():
            if hasattr(mod, k):
                setattr(mod, k, v)
    return npin_util, caf


# ---------------------------------------------------------------------------
# Per-cell features (seed-independent, computed once)
# ---------------------------------------------------------------------------
def _write_input(nodes, out: Path):
    """NPIN input for one cell: parents first, root numbered 1, basal and apical
    merged into dendrite, the root's soma group collapsed into the root.

    Returns, per parsed node, the NPIN id it maps to (the root for soma nodes)
    and, per NPIN id, the id of the branch point or tip that ends its segment."""
    n = len(nodes)
    index = {nd.id: i for i, nd in enumerate(nodes)}
    kids = [[] for _ in range(n)]
    roots = []
    for i, nd in enumerate(nodes):
        j = index.get(nd.parent)
        (kids[j].append(i) if j is not None else roots.append(i))
    if len(roots) != 1:
        raise ValueError(f"{len(roots)} roots")
    root = roots[0]
    soma = {root}
    if nodes[root].type == 1:
        stack = [root]
        while stack:
            u = stack.pop()
            for c in kids[u]:
                if nodes[c].type == 1:
                    soma.add(c)
                    stack.append(c)
    # children of the collapsed soma hang from the root, in parse order
    top = sorted(c for s in soma for c in kids[s] if c not in soma)
    new_id = {}
    order = []
    stack = [root]
    while stack:                               # depth-first, parents before children
        u = stack.pop()
        new_id[u] = len(order) + 1
        order.append(u)
        ch = top if u == root else kids[u]
        stack.extend(reversed(ch))
    npar = {}
    for u in order:
        if u == root:
            npar[u] = -1
        else:
            p = index[nodes[u].parent]
            npar[u] = new_id[root] if p in soma else new_id[p]
    with open(out, "w", encoding="utf-8") as fh:
        for u in order:
            nd = nodes[u]
            t = 1 if u == root else MERGE.get(nd.type, 3)
            fh.write(f"{new_id[u]} {t} {nd.x!r} {nd.y!r} {nd.z!r} {nd.radius!r} {npar[u]}\n")
    # segment end: a level node (root, fork, tip) ends its own segment, a node with
    # one child takes its child's
    nchild = {new_id[u]: 0 for u in order}
    for u in order:
        if npar[u] != -1:
            nchild[npar[u]] += 1
    seg = {}
    for u in reversed(order):
        nid = new_id[u]
        if npar[u] == -1 or nchild[nid] != 1:
            seg[nid] = nid
        else:
            only = next(new_id[c] for c in (top if u == root else kids[u]))
            seg[nid] = seg[only]
    node_seg = np.array([seg[new_id[root]] if i in soma else seg[new_id[i]] for i in range(n)], dtype=np.int64)
    return node_seg


def _cell_features(task):
    """NPIN's features and labels for one cell, and what scoring needs."""
    ct, path, npin_dir, fast = task
    npin_util, caf = load_npin(npin_dir, fast=fast)
    name = f"{ct}__{Path(path).stem}"
    work = WORK / ("cells" if fast else "cells_original")
    (work / "nrn_original").mkdir(parents=True, exist_ok=True)
    (work / "nrn_cleaned").mkdir(parents=True, exist_ok=True)
    nodes = parse_swc(path)
    t0 = time.perf_counter()
    node_seg = _write_input(nodes, work / "nrn_original" / f"{name}.swc")
    with contextlib.redirect_stdout(io.StringIO()):
        ax = caf.Create_Axon_Feature(str(work) + "/", name, None, None)
        ax.branch = BRANCH_LIMIT
        ax = ax.load_data()
        df_axon = ax.df_axon.reset_index(drop=True)
        df_eval = npin_util.update_parent_col(ax.df, ax.df_dis, ax.tree_node_dict, "ID", "PARENT_ID")
        df_eval = df_eval.loc[:, ["nrn", "ID", "PARENT_ID", "NC", "type_pre"]].reset_index(drop=True)
        train_rows = npin_util.pyramid(df_axon, FEATURES, 1, 1, for_train=True, empty_number=-1)
        test_rows = npin_util.pyramid(df_axon, FEATURES, 1, 1, for_train=False, empty_number=-1)
    # Exactly the two files NPIN wrote for this cell: a pattern such as name*.pkl
    # would also match a cell whose name extends this one, in another worker.
    for f in (work / "nrn_cleaned" / f"{name}.pkl", work / "nrn_cleaned" / f"{name}_leaf5.pkl"):
        if f.exists():
            f.unlink()
    (work / "nrn_original" / f"{name}.swc").unlink()
    return {"name": name, "cell_type": ct, "file": Path(path).name, "n_nodes": len(nodes),
            "gt": np.array([MERGE.get(nd.type, 3) for nd in nodes], dtype=np.int8),
            "node_seg": node_seg, "df_axon": df_axon, "df_eval": df_eval,
            "train_rows": train_rows, "test_rows": test_rows,
            "seconds": time.perf_counter() - t0}


def _cache_path(ct: str, file: str) -> Path:
    return WORK / "features" / ct / (Path(file).stem + ".pkl")


def _extract_one(task):
    ct, path, npin_dir = task
    cell = f"{ct}/{Path(path).name}"
    out = _cache_path(ct, Path(path).name)
    if out.is_file():
        return cell, None
    try:
        rec = _cell_features((ct, path, npin_dir, True))
    except (Exception, SystemExit) as exc:       # NPIN stops with sys.exit on some inputs;
        return cell, f"{type(exc).__name__}: {str(exc).strip()}"   # recorded, never dropped
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(rec, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    return cell, None


# ---------------------------------------------------------------------------
# Training, threshold, relabel, scoring
# ---------------------------------------------------------------------------
def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(seed=123)                 # NPIN's model, default settings


def _relabel_one(task):
    """NPIN's post_relabel on one cell at each threshold; tip hits and labels."""
    df_eval, prob, thresholds, npin_dir, want_labels = task
    npin_util, _ = load_npin(npin_dir, fast=True)
    base = pd.merge(df_eval, prob, how="left", on=["nrn", "ID"])
    out = {}
    for th in thresholds:
        with contextlib.redirect_stdout(io.StringIO()):
            df = npin_util.post_relabel(base.copy(), threshold=th)
        tips = df.loc[(df["NC"] == 0) & (df["type_pre"].isin([2, 3]))]
        hits = int((tips["type_post"].to_numpy() == tips["type_pre"].to_numpy()).sum())
        out[th] = {"tips": int(len(tips)), "hits": hits,
                   "labels": (dict(zip(df["ID"].tolist(), df["type_post"].tolist())) if want_labels else None),
                   "prob": (dict(zip(df["ID"].tolist(), df["prob"].tolist())) if want_labels else None)}
    return out


def _fold(file: str, seed: int) -> int:
    h = hashlib.md5(f"npin-cv|{seed}|{file}".encode()).hexdigest()
    return int(h[:8], 16) % CV_FOLDS


def _fit_predict(npin_util, train_cells, pred_cells):
    """NPIN's classification(): XGBClassifier(seed=123) fitted on the training
    rows, unstandardized, and each predicted row's probability of axon."""
    train = pd.concat([c["train_rows"] for c in train_cells], ignore_index=True)
    test = pd.concat([c["test_rows"] for c in pred_cells], ignore_index=True)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, proba, _ = npin_util.classification(train, test, label=["label"], features=FEATURE_COLS,
                                                  model=_xgb())
    out = test[["nrn", "ID"]].copy()
    out["prob"] = proba[:, 1]
    return {n: g for n, g in out.groupby("nrn", sort=False)}


def _choose_threshold(npin_util, cells, seed, npin_dir, pool):
    """Five-fold cross-validation over the training cells: out-of-fold
    probabilities, NPIN's relabel at each threshold, NPIN's tip accuracy."""
    folds = [[c for c in cells if _fold(c["file"], seed) == k] for k in range(CV_FOLDS)]
    oof = {}
    for k in range(CV_FOLDS):
        oof.update(_fit_predict(npin_util, [c for j, f in enumerate(folds) if j != k for c in f], folds[k]))
    tasks = [(c["df_eval"], oof[c["name"]], THRESHOLDS, npin_dir, False) for c in cells if c["name"] in oof]
    tips = {th: 0 for th in THRESHOLDS}
    hits = {th: 0 for th in THRESHOLDS}
    for res in pool.map(_relabel_one, tasks, chunksize=16):
        for th, v in res.items():
            tips[th] += v["tips"]
            hits[th] += v["hits"]
    acc = {th: hits[th] / tips[th] if tips[th] else 0.0 for th in THRESHOLDS}
    best = max(THRESHOLDS, key=lambda th: (acc[th], -th))   # NPIN takes the first maximum
    return best, acc


def _node_labels(cell, labels, prob):
    """Every node of a segment takes the label of the branch point or tip that
    ends it; a mixed or undecided one falls back on its own classifier call."""
    out = np.empty(len(cell["node_seg"]), dtype=np.int8)
    for i, s in enumerate(cell["node_seg"].tolist()):
        lab = labels.get(s, 0)
        if lab not in (2, 3):
            pr = prob.get(s, float("nan"))
            lab = 2 if (pr == pr and pr >= 0.5) else 3
        out[i] = lab
    return out


def _two_class(gt_cells, pred_cells):
    """Axon versus dendrite on non-soma nodes, pooled and per cell."""
    from sklearn.metrics import precision_recall_fscore_support
    g = np.concatenate(gt_cells)
    p = np.concatenate(pred_cells)
    keep = g != 1
    g, p = g[keep], p[keep]
    _, _, f, sup = precision_recall_fscore_support(g, p, labels=[2, 3], zero_division=0)
    present = [i for i in range(2) if sup[i] > 0]
    pcf1 = []
    for gc, pc in zip(gt_cells, pred_cells):
        k = gc != 1
        gc, pc = gc[k], pc[k]
        if gc.size == 0:
            continue
        s = []
        for lbl in (2, 3):
            isg = gc == lbl
            if not isg.any():
                continue
            isp = pc == lbl
            tp = int((isg & isp).sum())
            den = int(isg.sum()) + int(isp.sum())
            s.append(2 * tp / den if den else 0.0)
        pcf1.append(float(np.mean(s)))
    pcf1 = np.asarray(pcf1)
    return {"n_cells": int(pcf1.size), "n_nodes": int(g.size),
            "accuracy": float((g == p).mean()), "axon_f1": float(f[0]), "dendrite_f1": float(f[1]),
            "macro_f1": float(np.mean([f[i] for i in present])),
            "pcf1_mean": float(pcf1.mean()), "pcf1_p10": float(np.percentile(pcf1, 10))}


def _pipeline_labels(task):
    """The pipeline's labels for one test cell, ground-truth cell type."""
    ct, path, model_dir = task
    from pipeline.infer_pipeline import run_pipeline_on_nodes
    nodes = parse_swc(path)
    pr = run_pipeline_on_nodes(nodes, file_path=str(path),
                               stage1_model=Path(model_dir) / "cell_type_classifier.pkl",
                               stage2_model=Path(model_dir) / "branch_classifier.pkl",
                               override_cell_type=ct)
    return np.array([MERGE.get(int(x), 3) for x in pr.node_labels], dtype=np.int8)


def run_seed(seed, cells_by_type, npin_dir, pool):
    result = {"seed": seed, "npin_commit": NPIN_COMMIT, "features": FEATURES, "model": "XGBClassifier(seed=123)",
              "by_cell_type": {}}
    npin_util, _ = load_npin(npin_dir)
    gts, npin_preds, our_preds = {}, {}, {}
    for ct in CELL_TYPES:
        cells = cells_by_type[ct]
        train = [c for c in cells if not in_test(c["file"], seed, 0.20)]
        test = [c for c in cells if in_test(c["file"], seed, 0.20)]
        th, cv_acc = _choose_threshold(npin_util, train, seed, npin_dir, pool)
        probs = _fit_predict(npin_util, train, test)
        tasks = [(c["df_eval"], probs[c["name"]], [th], npin_dir, True) for c in test]
        preds = []
        for c, res in zip(test, pool.map(_relabel_one, tasks, chunksize=8)):
            preds.append(_node_labels(c, res[th]["labels"], res[th]["prob"]))
        model_dir = ROOT / "pipeline" / "models" / f"seed{seed}"
        ours = list(pool.map(_pipeline_labels, [(ct, DATA / ct / c["file"], str(model_dir)) for c in test],
                             chunksize=4))
        gt = [c["gt"] for c in test]
        gts[ct], npin_preds[ct], our_preds[ct] = gt, preds, ours
        no_axon = [i for i, g in enumerate(gt) if not (g == 2).any()]
        result["by_cell_type"][ct] = {
            "n_train_cells": len(train), "n_test_cells": len(test),
            "threshold": th, "threshold_cv_tip_accuracy": {str(k): v for k, v in cv_acc.items()},
            "npin": _two_class(gt, preds), "ours": _two_class(gt, ours),
            "cells_without_axon": {
                "n": len(no_axon),
                "npin_mean_axon_share": float(np.mean([(preds[i][gt[i] != 1] == 2).mean() for i in no_axon])) if no_axon else None,
                "ours_mean_axon_share": float(np.mean([(ours[i][gt[i] != 1] == 2).mean() for i in no_axon])) if no_axon else None}}
        print(f"  seed {seed} {ct}: threshold {th}, npin {result['by_cell_type'][ct]['npin']['macro_f1']:.4f} "
              f"ours {result['by_cell_type'][ct]['ours']['macro_f1']:.4f}", flush=True)
    result["all"] = {"npin": _two_class([g for ct in CELL_TYPES for g in gts[ct]],
                                        [p for ct in CELL_TYPES for p in npin_preds[ct]]),
                     "ours": _two_class([g for ct in CELL_TYPES for g in gts[ct]],
                                        [p for ct in CELL_TYPES for p in our_preds[ct]])}
    return result


def _summary(results):
    def ms(vals):
        return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "per_seed": vals}
    out = {"seeds": [r["seed"] for r in results], "blocks": {}}
    keys = ("accuracy", "axon_f1", "dendrite_f1", "macro_f1", "pcf1_mean", "pcf1_p10")
    for blk in CELL_TYPES + ("all",):
        out["blocks"][blk] = {}
        for who in ("npin", "ours"):
            get = (lambda r: r["all"][who]) if blk == "all" else (lambda r: r["by_cell_type"][blk][who])
            out["blocks"][blk][who] = {k: ms([get(r)[k] for r in results]) for k in keys}
    lines = [f"NPIN (retrained) vs the pipeline, axon vs dendrite, non-soma nodes; mean (SD) over seeds "
             f"{', '.join(str(s) for s in out['seeds'])}", ""]
    for blk, v in out["blocks"].items():
        lines.append(blk)
        for who in ("npin", "ours"):
            lines.append(f"  {who:5s} " + "  ".join(f"{k} {v[who][k]['mean']:.4f} ({v[who][k]['sd']:.4f})" for k in keys))
    return out, "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Verification: the fast helpers against NPIN's own
# ---------------------------------------------------------------------------
def _peak_rss(fn, *args):
    """fn(*args) and the process's peak resident memory while it ran, in GB."""
    import threading
    import psutil
    proc, peak, done = psutil.Process(), [0], threading.Event()

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            done.wait(0.2)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        return fn(*args), peak[0] / 1e9
    finally:
        done.set()
        th.join()


def _verify_one(task):
    ct, path, npin_dir = task
    fast, fast_gb = _peak_rss(_cell_features, (ct, path, npin_dir, True))
    orig, orig_gb = _peak_rss(_cell_features, (ct, path, npin_dir, False))
    diffs = {}
    for key in ("df_axon", "df_eval", "train_rows", "test_rows"):
        a, b = fast[key], orig[key]
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            diffs[key] = "shape or columns differ"
            continue
        a = a.sort_values(["nrn", "ID"]).reset_index(drop=True)
        b = b.sort_values(["nrn", "ID"]).reset_index(drop=True)
        worst = 0.0
        for col in a.columns:
            va, vb = a[col].to_numpy(), b[col].to_numpy()
            if va.dtype.kind in "fc" or vb.dtype.kind in "fc":
                d = np.nanmax(np.abs(va.astype(float) - vb.astype(float))) if va.size else 0.0
                if not (np.isnan(va.astype(float)) == np.isnan(vb.astype(float))).all():
                    d = float("inf")
                worst = max(worst, float(d))
            elif not (va == vb).all():
                worst = float("inf")
        diffs[key] = worst
    return {"file": Path(path).name, "n_nodes": fast["n_nodes"], "fast_s": round(fast["seconds"], 2),
            "original_s": round(orig["seconds"], 2), "fast_peak_gb": round(fast_gb, 2),
            "original_peak_gb": round(orig_gb, 2), "max_abs_diff": diffs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npin-dir", type=Path, required=True, help="clone of gitlab.com/czsu32/npin at 8d632fe")
    ap.add_argument("--seeds", default="42,123,789")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--verify", type=int, default=0,
                    help="instead of the evaluation, compare the fast helpers with NPIN's own on this many "
                         "cells of at most --verify-max-nodes nodes")
    ap.add_argument("--verify-max-nodes", type=int, default=12000)
    ap.add_argument("--verify-files", default="",
                    help="with --verify, also these cells, as cell_type/file.swc separated by commas, whatever "
                         "their size (NPIN's own helpers need minutes and gigabytes on the large ones)")
    args = ap.parse_args()
    if not (args.npin_dir / "nrn3_class_Create_Axon_Feature.py").is_file():
        raise SystemExit(f"not an NPIN clone: {args.npin_dir}")
    npin_dir = str(args.npin_dir.resolve())
    OUT.mkdir(parents=True, exist_ok=True)
    cells = [(ct, p) for ct in CELL_TYPES for p in sorted((DATA / ct).glob("*.swc"))]

    if args.verify or args.verify_files:
        import random
        rng = random.Random(2021)
        small = [(ct, p) for ct, p in cells if sum(1 for _ in open(p, "rb")) <= args.verify_max_nodes]
        pick = rng.sample(small, min(args.verify, len(small)))
        named = [f.strip().split("/", 1) for f in args.verify_files.split(",") if f.strip()]
        pick = [(ct, DATA / ct / name) for ct, name in named] + pick   # the long ones start first
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(_verify_one, [(ct, str(p), npin_dir) for ct, p in pick]))
        worst = max((v for r in rows for v in r["max_abs_diff"].values() if isinstance(v, float)), default=0.0)
        text = [r for r in rows if any(not isinstance(v, float) for v in r["max_abs_diff"].values())]
        out = {"npin_commit": NPIN_COMMIT, "n_cells": len(rows), "max_abs_diff": worst,
               "structural_mismatches": len(text), "cells": rows}
        # Named cells alone (timing the large ones on an otherwise idle machine)
        # keep their own file, so the random sample's is not overwritten.
        name = "npin_verify.json" if args.verify else "npin_verify_named.json"
        (OUT / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"verified {len(rows)} cells: max |fast - original| {worst:.3g}, "
              f"structural mismatches {len(text)}")
        return 0 if (worst < 1e-9 and not text) else 1

    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.perf_counter()
    todo = [(ct, str(p), npin_dir) for ct, p in cells if not _cache_path(ct, p.name).is_file()]
    # largest first, so the long cells do not all land at the end
    todo.sort(key=lambda t: -os.path.getsize(t[1]))
    failures = {}
    if todo:
        print(f"NPIN features for {len(todo)} cells ({args.workers} workers)", flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            for i, (cell, err) in enumerate(pool.map(_extract_one, todo, chunksize=1), 1):
                if err:
                    failures[cell] = err
                if i % 500 == 0:
                    print(f"  ... {i}/{len(todo)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    cells_by_type = {ct: [] for ct in CELL_TYPES}
    for ct, p in cells:
        cp = _cache_path(ct, p.name)
        if cp.is_file():
            with open(cp, "rb") as fh:
                cells_by_type[ct].append(pickle.load(fh))
    # A cell NPIN's code cannot process is left out of both methods' scores, so
    # the two are always compared on the same cells; the count is reported.
    missing = [f"{ct}/{p.name}" for ct, p in cells if not _cache_path(ct, p.name).is_file()]
    if missing:
        print(f"warning: {len(missing)} cells have no NPIN features and are scored for neither method; "
              f"first failures: {dict(list(failures.items())[:5])}", flush=True)
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for s in seeds:
            r = run_seed(s, cells_by_type, npin_dir, pool)
            r["cells_without_npin_features"] = missing
            r["npin_failures"] = failures
            r["feature_seconds_total"] = float(sum(c["seconds"] for ct in CELL_TYPES for c in cells_by_type[ct]))
            (OUT / f"npin_seed{s}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
            results.append(r)
    summary, text = _summary(results)
    summary["npin_commit"] = NPIN_COMMIT
    (OUT / "npin_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "npin_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
