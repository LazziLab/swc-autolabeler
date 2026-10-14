"""The unstaged labeler: one model per cell type over the pipeline's whole feature set.

The staged pipeline decides a cell in steps, a subtree at a time, then repairs the
result with anatomical rules and a proximal branch correction. This model sees the
same descriptors but takes none of those steps: every descriptor the pipeline
computes for a branch, its primary subtree and its whole cell is concatenated into
one row, and a single gradient-boosted model labels each branch in one pass, with
nothing forcing its labels to agree across the cell.

It is a configuration of the same system, not a separate method, and the two have
different strengths. The unstaged variant is the more accurate on cells drawn from
the protocol it was trained on, measured by pooled node accuracy. The staged
pipeline holds up better on the worst cells and on data from laboratories it never
saw, and it is the one that carries the Stage 4 quality flag.

The cell type is still needed: one model is fitted per cell type, exactly as Stage
2 does, so labeling an unseen file uses Stage 1 to route it first.

    train    pipeline.train_unstaged   -> seed{S}/unstaged_model.joblib
    label    pipeline.infer_unstaged   in.swc -o out.swc

The feature layout is shared with ``evaluation.baselines.xgb_full``, which produces
the paper's comparison, so both read exactly the same numbers per branch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .branch_features import BRANCH_FEATURE_NAMES, extract_branches
from .features import SWCNode, extract_global_features
from .stage1.cell_type_detector import FEATURE_NAMES
from .stage2.subtree_features import PRIMARY_SUBTREE_FEATURE_NAMES, extract_primary_subtrees

CELL_TYPES = ("pyramidal", "interneuron")
#: Neurite labels each cell type's model may predict, as Stage 2 uses.
LABELS: dict[str, tuple[int, ...]] = {"pyramidal": (2, 3, 4), "interneuron": (2, 3)}

#: Width of one branch's row: branch, then its primary subtree, then the whole cell.
N_BRANCH, N_SUBTREE, N_CELL = len(BRANCH_FEATURE_NAMES), len(PRIMARY_SUBTREE_FEATURE_NAMES), len(FEATURE_NAMES)
N_FEATURES = N_BRANCH + N_SUBTREE + N_CELL
#: Identifies the feature layout. A cached feature file or a saved model that does
#: not carry this exact string was built from different descriptors and is refused
#: rather than silently reused.
FEATURE_VERSION = f"branch{N_BRANCH}+subtree{N_SUBTREE}+cell{N_CELL}"


class FeatureMismatch(RuntimeError):
    """A cache entry or model was built from a different feature layout."""


def check_feature_version(found: str | None, what: str) -> None:
    if found != FEATURE_VERSION:
        raise FeatureMismatch(
            f"{what} was built with feature layout {found!r}, this build uses "
            f"{FEATURE_VERSION!r}. Rebuild it: the descriptors have changed.")


def branch_rows(nodes: list[SWCNode], cell_type: str, file_path: str = ""):
    """One row per branch: its own descriptors, its primary subtree's, the cell's.

    Returns ``(X, node_indices, soma_indices)`` where ``X`` has one row per branch
    in the order ``extract_branches`` returns them, ``node_indices[i]`` are the node
    positions branch ``i`` covers, and ``soma_indices`` is the label-free soma
    anchor. A branch with no primary subtree gets NaN for that block, which the
    tree model handles natively. Nothing here reads the SWC type column.
    """
    morph = extract_branches(nodes, cell_type, file_path)
    if not morph.branches:
        return np.zeros((0, N_FEATURES), dtype=np.float32), [], sorted(morph.soma_indices)

    glob = extract_global_features(nodes)      # once per cell: it walks every node
    cell = np.asarray([glob[name] for name in FEATURE_NAMES], dtype=np.float32)
    subs = {int(s.root_idx): np.asarray(s.features, dtype=np.float32)
            for s in extract_primary_subtrees(nodes)}
    blank = np.full(N_SUBTREE, np.nan, dtype=np.float32)

    X = np.stack([
        np.concatenate([np.asarray(br.features, dtype=np.float32),
                        subs.get(br.primary_root_idx, blank) if br.primary_root_idx is not None else blank,
                        cell])
        for br in morph.branches
    ])
    return X, [list(br.node_indices) for br in morph.branches], sorted(morph.soma_indices)


def node_labels(n_nodes: int, node_indices, soma_indices, parent_idx, branch_labels) -> list[int]:
    """Spread per-branch labels over nodes, the way the comparison driver does.

    The soma anchor is soma, every branch paints its own nodes, and a node covered
    by no branch inherits its nearest labeled ancestor, falling back to basal.
    """
    lab = np.zeros(n_nodes, dtype=np.int16)
    for idx, value in zip(node_indices, branch_labels):
        lab[np.asarray(idx, dtype=np.int64)] = int(value)
    lab[np.asarray(soma_indices, dtype=np.int64)] = 1
    parent = np.asarray(parent_idx, dtype=np.int64)
    for i in np.flatnonzero(lab == 0):
        j, hops = parent[i], 0
        while j >= 0 and lab[j] in (0, 1) and hops < n_nodes:
            j, hops = parent[j], hops + 1
        lab[i] = lab[j] if j >= 0 and lab[j] not in (0, 1) else 3
    return [int(v) for v in lab]


def parent_indices(nodes: list[SWCNode]) -> list[int]:
    """Parent position of every node, -1 for a root, as ``node_labels`` expects."""
    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    return [id_to_idx.get(nd.parent, -1) for nd in nodes]


def save(models: dict, out_path: str | Path, meta: dict | None = None) -> Path:
    """Write a model bundle: one fitted classifier per cell type, plus provenance."""
    import joblib
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "labels": {ct: list(LABELS[ct]) for ct in models},
                 "feature_version": FEATURE_VERSION, "n_features": N_FEATURES,
                 **(meta or {})}, out_path, compress=3)
    return out_path


def load(model_dir: str | Path, name: str = "unstaged_model.joblib") -> dict | None:
    """Load a bundle from a model dir, or None if that dir has no unstaged model."""
    import joblib
    path = Path(model_dir) / name
    if not path.is_file():
        return None
    bundle = joblib.load(path)
    check_feature_version(bundle.get("feature_version"), str(path))
    return bundle


def label_cell(nodes: list[SWCNode], cell_type: str, bundle: dict) -> list[int]:
    """Label one parsed cell with the unstaged model for its cell type."""
    model = bundle["models"].get(cell_type)
    if model is None:
        raise SystemExit(f"the bundle has no unstaged model for {cell_type!r}; "
                         f"it holds {sorted(bundle['models'])}")
    X, node_indices, soma = branch_rows(nodes, cell_type)
    if not len(X):
        soma_set = set(soma)
        return [1 if i in soma_set else 3 for i in range(len(nodes))]
    # predict_proba then argmax, as the branch correction does: a two-class model
    # fitted with an explicit multi-class objective returns probabilities from
    # predict, not class indices.
    chosen = np.asarray(model.predict_proba(X)).argmax(axis=1)
    # The column order must be the order LABELS was encoded in at training time,
    # or every node would be confidently mislabeled rather than merely wrong.
    expected = list(range(len(LABELS[cell_type])))
    if list(getattr(model, "classes_", expected)) != expected:
        raise FeatureMismatch(
            f"the {cell_type!r} model encodes classes {list(model.classes_)}, not "
            f"{expected}; its columns cannot be read as {list(LABELS[cell_type])}")
    labels = np.asarray(LABELS[cell_type])[chosen]
    return node_labels(len(nodes), node_indices, soma, parent_indices(nodes), labels)
