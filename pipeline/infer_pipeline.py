"""End-to-end auto-labeling pipeline.

Chains Stage 1 (cell-type detection) -> Stage 2 (subtree-owner labeling) ->
Stage 3 (topology refinement and branch correction) into a single call that
returns per-node SWC labels.
"""
from __future__ import annotations

import argparse
import math
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._model_cache import load_cached
from .features import SWCNode, parse_swc
from .stage1.cell_type_detector import (
    CELL_TYPE_LABEL_SETS,
    CellTypeResult,
    detect_cell_type_from_nodes,
)
from .branch_features import extract_branches
from .stage2.owner_features import best_apical_owner, predict_subtree_owner_map
from .stage3.refine import RefinementResult, refine
from .swc_normalize import normalize_swc

# Canonical shipped model directory. Seed 123 is the deployed labeler; seeds 42
# and 789 are trained identically for the multi-seed results.
DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "seed123"
STAGE2_MODEL = DEFAULT_MODEL_DIR / "branch_classifier.pkl"


@dataclass
class PipelineResult:
    """Full pipeline output."""
    stage1: CellTypeResult
    stage3: RefinementResult
    node_labels: list[int]          # final per-node labels
    node_confidences: list[float]   # per-node ML confidence


def _read_pickle(path: Path) -> dict:
    # Unpickling executes code held in the file, as it does for every
    # pickle- or joblib-backed model. Point ``--model-dir`` only at a model set
    # you trust, the shipped one or one you trained yourself.
    with open(path, "rb") as f:
        return pickle.load(f)


def load_stage2_bundle(model_path: str | Path) -> dict:
    """The Stage 2 model bundle (per-cell-type subtree-owner models), read once
    per file state. Callers share the returned dict and must not modify it."""
    return load_cached(model_path, _read_pickle)


def _load_correction_cached(stage2_model: str | Path | None) -> dict | None:
    """The Stage 3 branch correction that ships next to the Stage 2 model, read
    once per file state. The correction is intrinsic to the pipeline -- there is
    no flag to disable it -- so it is loaded unconditionally. Returns None only
    if a model dir genuinely has no ``branch_correction.joblib`` (e.g. a partial
    or third-party model set)."""
    s2 = Path(stage2_model) if stage2_model else STAGE2_MODEL
    path = s2.parent / "branch_correction.joblib"
    if not path.is_file():
        return None
    from .stage3.branch_correction import load_correction  # noqa: PLC0415
    return load_cached(path, load_correction)


# Type given to the neurites of an unlabeled file before normalization. Any
# neurite code would do, since no feature reads it: its only job is to keep
# those nodes from inheriting the soma's type and being merged into the soma.
_UNLABELED_NEURITE_TYPE = 3

# An unlabeled file cannot say which points draw its soma, but the common
# several-point conventions can be read from shape: the extra points hang off the
# root as childless nodes of the root's own radius, no further from it than that
# radius. The tolerances absorb the rounding of values written to file.
_SOMA_POINT_RADIUS_TOL = 0.02      # relative difference from the root's radius
_SOMA_POINT_REACH = 1.05           # distance from the root, in root radii


def _soma_points_by_shape(raw: list[SWCNode], root: SWCNode) -> list[int]:
    """Indices of the nodes that draw the soma together with ``root``.

    A node qualifies when it is a child of the root, has no children of its own,
    has the root's radius and lies within one root radius of it. That is the two
    outer points of the standard three-point soma, and the variants that place
    them closer in. A neurite leaving the soma continues onward and is far
    thinner than the soma, so it does not qualify. A soma drawn as a chain or
    contour of points is not recognised.
    """
    if root.radius <= 0:
        return []
    n_children = Counter(nd.parent for nd in raw)
    here = (root.x, root.y, root.z)
    return [
        i for i, nd in enumerate(raw)
        if nd.parent == root.id and n_children[nd.id] == 0
        and abs(nd.radius - root.radius) <= _SOMA_POINT_RADIUS_TOL * root.radius
        and math.dist((nd.x, nd.y, nd.z), here) <= _SOMA_POINT_REACH * root.radius
    ]


def parse_swc_for_labeling(swc_path: str | Path) -> tuple[list[SWCNode], list[str]]:
    """Parse a file that is to be labeled, whatever labels it already carries.

    Labeling reads the type column for one thing only, the soma, so that a soma
    drawn with several points can be merged into one node. A file without labels
    needs two things settled before the normalization ``parse_swc`` applies:

      * no node is typed soma: the root, the one node whose parent is -1, is
        taken as the soma, together with any further soma points recognised by
        shape (``_soma_points_by_shape``);
      * no node carries a neurite type (2, 3 or 4): every node other than the
        soma is an unlabeled neurite. Left with an undefined code, those nodes
        would inherit the soma's type from their ancestor and be merged into it.

    A file that has a soma and neurite labels parses exactly as ``parse_swc``
    does. Returns the nodes and one note for each of the two cases applied.
    """
    raw = parse_swc(swc_path, normalize_types=False)
    types = [nd.type for nd in raw]
    notes: list[str] = []
    roots = [i for i, nd in enumerate(raw) if nd.parent == -1]
    if raw and 1 not in types and len(roots) == 1:
        root = raw[roots[0]]
        types[roots[0]] = 1
        more = _soma_points_by_shape(raw, root)
        for i in more:
            types[i] = 1
        notes.append(
            f"no node is typed soma; the root (node {root.id}) is taken as the soma"
            + (f", with {len(more)} further soma point{'s' if len(more) != 1 else ''} "
               "recognised by shape" if more else ""))
    if raw and not any(t in (2, 3, 4) for t in types):
        types = [t if t == 1 else _UNLABELED_NEURITE_TYPE for t in types]
        notes.append("no node carries a neurite label; every node other than the soma "
                     "is treated as an unlabeled neurite")
    if notes:
        raw = [SWCNode(nd.id, t, nd.x, nd.y, nd.z, nd.radius, nd.parent)
               for nd, t in zip(raw, types)]
    nodes, _ = normalize_swc(raw)
    return nodes, notes


def run_pipeline(
    swc_path: str | Path,
    stage1_model: str | Path | None = None,
    stage2_model: str | Path | None = None,
) -> PipelineResult:
    """Run the full auto-labeling pipeline on an SWC file, labeled or not.

    The Stage 3 branch correction is an intrinsic part of the pipeline: it is
    loaded from ``branch_correction.joblib`` in the model dir and always applied
    to pyramidal cells. There is no switch to disable it.

    This entry point does NOT apply the Stage 0 structural checks; the command
    line does, and refuses a file that fails them unless given ``--force``.
    Labeling is defined for one connected cell with a soma, so a caller handing
    in something else (several roots, no soma) gets a labeling rather than an
    error. Call ``pipeline.qc.qc_violations`` first if the input is untrusted.
    """
    nodes, _ = parse_swc_for_labeling(swc_path)
    return run_pipeline_on_nodes(nodes, str(swc_path), stage1_model, stage2_model)


def run_pipeline_on_nodes(
    nodes: list[SWCNode],
    file_path: str = "",
    stage1_model: str | Path | None = None,
    stage2_model: str | Path | None = None,
    override_cell_type: str | None = None,
    trace: dict | None = None,
    disable_stage3: frozenset[str] | set[str] = frozenset(),
) -> PipelineResult:
    """Run the full pipeline on pre-parsed nodes.

    Pass ``override_cell_type="pyramidal"`` (or ``"interneuron"``) to bypass
    Stage 1 and dispatch Stage 2 with the given cell type -- used for the
    GT-cell-type evaluation that measures the downstream stages independent of
    Stage 1 errors.

    Pass ``disable_stage3`` with names from ``pipeline.stage3.refine.RULES`` to
    skip those refinement rules; the default applies every rule. This is used to
    measure each rule's contribution and is not a deployment option.

    Pass a ``trace`` dict to capture the chosen run's intermediate per-node
    label/confidence snapshots (Stage 2 owner labels, labels after the Stage 3
    topology rules) alongside the final output. Default ``None`` records nothing
    and leaves behavior unchanged; used only by offline flag-model research.
    """
    # --- Stage 1: cell-type detection (or override) ---
    if override_cell_type is not None:
        s1_result = CellTypeResult(
            cell_type=override_cell_type,
            confidence=1.0,
            probabilities={override_cell_type: 1.0},
            label_set=set(CELL_TYPE_LABEL_SETS.get(override_cell_type, {1, 2, 3})),
            features={},
        )
    else:
        s1_result = detect_cell_type_from_nodes(nodes, stage1_model)

    s2_path = Path(stage2_model) if stage2_model else STAGE2_MODEL
    bundle = load_stage2_bundle(s2_path)

    # --- Stages 2 and 3 under the cell type Stage 1 (or the override) fixed ---
    chosen_ct = s1_result.cell_type
    chosen = _run_stage23(
        nodes, file_path, chosen_ct, bundle, s1_result,
        capture_trace=trace is not None,
        disable_stage3=disable_stage3,
    )

    if trace is not None and chosen.get("trace"):
        trace.update(chosen["trace"])

    # --- Stage 3 branch correction (pyramidal only) ---
    # A class-weighted gradient-boosted head re-decides axon/basal/apical per
    # branch and overrides the current label where confident and disagreeing.
    # This is an intrinsic stage, not an option: the correction head that ships
    # in the model dir is always applied to pyramidal cells.
    final_labels = chosen["final_labels"]
    node_confidences = chosen["node_confidences"]
    if chosen_ct == "pyramidal":
        correction = _load_correction_cached(stage2_model)
        if correction is not None:
            from .stage3.branch_correction import apply_correction  # noqa: PLC0415
            final_labels, node_confidences = apply_correction(
                nodes, final_labels, node_confidences, correction,
                morph=chosen["morph"],
            )

    return PipelineResult(
        stage1=s1_result,
        stage3=chosen["s3_result"],
        node_labels=final_labels,
        node_confidences=node_confidences,
    )


def _default_neurite_label(neurite_labels: list[int]) -> int:
    """The label to fall back on when Stage 2 says nothing usable: basal.

    Every default in this stage goes through here so the sites cannot drift
    apart and every default follows one convention.
    Cell types without basal in their set fall back to their first neurite
    label instead.
    """
    return 3 if 3 in neurite_labels else (neurite_labels[0] if neurite_labels else 3)


#: Confidence attached to any label that came from a default rather than a model.
DEFAULT_LABEL_CONF = 0.5


def _run_stage23(
    nodes: list[SWCNode],
    file_path: str,
    cell_type: str,
    bundle: dict,
    s1_result: CellTypeResult,
    capture_trace: bool = False,
    disable_stage3: frozenset[str] | set[str] = frozenset(),
) -> dict:
    """Run Stage 2 + Stage 3 for a given cell type.

    Returns a dict of: ``final_labels``, ``node_confidences``, ``s3_result``
    (the RefinementResult), ``mean_neurite_conf`` (the mean Stage 2 posterior
    over the cell's branches), ``cell_type``, ``trace`` (the intermediate
    snapshots, or None when ``capture_trace`` is false) and ``morph`` (the
    extracted branches, handed on so the branch correction need not redo them).
    """
    n = len(nodes)
    label_set = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
    neurite_labels = sorted(label_set - {1})

    # --- Stage 2: per-subtree owner labeling ---
    subtree_models_by_ct = bundle.get("subtree_owner_models_by_cell_type") or {}
    subtree_owner_model = subtree_models_by_ct.get(cell_type)
    if subtree_owner_model is None:
        # A bundle trained on one cell type covers only that one. Labeling the
        # other kind anyway would return a whole cell of basal dendrite at a
        # flat 0.5, which reads like a result rather than a missing model.
        raise ValueError(
            f"no Stage 2 model for {cell_type!r} in this bundle, which holds "
            f"{sorted(subtree_models_by_ct) or 'nothing'}. Give the cell type this "
            "bundle was trained on, or use a model set trained on both.")

    morph = extract_branches(nodes, cell_type, file_path)
    subtree_owner_map = predict_subtree_owner_map(nodes, subtree_owner_model)
    apical_owner_root = best_apical_owner(subtree_owner_map) if cell_type == "pyramidal" else None

    proxy_soma = set(morph.soma_indices)
    node_labels = [1 if i in proxy_soma else 0 for i in range(n)]
    node_confidences = [1.0 if i in proxy_soma else 0.0 for i in range(n)]

    # Each primary subtree takes the label its subtree-owner model predicts,
    # propagated to every branch in the subtree. Whole-subtree features carry
    # stronger signal for axon-vs-apical than per-branch features; the branch
    # correction after Stage 3 re-decides individual proximal branches.
    branch_confs: list[float] = []
    if not subtree_owner_map:
        # The cell has no primary subtree for the model to score, so there is
        # nothing to propagate. A missing model cannot reach here: it raises above.
        fallback = _default_neurite_label(neurite_labels)
        for br in morph.branches:
            branch_confs.append(DEFAULT_LABEL_CONF)
            for node_idx in br.node_indices:
                node_labels[node_idx] = fallback
                node_confidences[node_idx] = DEFAULT_LABEL_CONF
    else:
        for br in morph.branches:
            pr = br.primary_root_idx
            info = subtree_owner_map.get(pr) if pr is not None else None
            if info is None:
                best_label = _default_neurite_label(neurite_labels)
                best_conf = DEFAULT_LABEL_CONF
            else:
                pred = int(info.get("pred", 3))
                if pred not in neurite_labels:
                    # Model predicted a class invalid for this cell type
                    # (e.g. apical for an interneuron) -- default to basal.
                    pred = _default_neurite_label(neurite_labels)
                best_label = pred
                best_conf = float(info.get("conf", DEFAULT_LABEL_CONF))
            branch_confs.append(best_conf)
            for node_idx in br.node_indices:
                node_labels[node_idx] = best_label
                node_confidences[node_idx] = best_conf

    # A node that is neither proxy soma nor part of any extracted branch. Branch
    # extraction covers every non-soma node of a connected cell, so this is
    # unreachable on a file that passes the Stage 0 checks, and it fired on no
    # node of the corpus. It takes the same default as every other, rather than
    # a convention of its own.
    for i in range(n):
        if node_labels[i] == 0:
            node_labels[i] = _default_neurite_label(neurite_labels)
            node_confidences[i] = DEFAULT_LABEL_CONF

    # Snapshot the Stage 2 owner labeling before Stage 3, so offline research
    # can measure how much each later stage flipped.
    trace: dict | None = None
    if capture_trace:
        trace = {
            "s2_labels": list(node_labels),
            "s2_confidences": list(node_confidences),
        }

    # Build a Stage-1 result carrying this candidate cell type so Stage 3 uses
    # the matching label set.
    trial_s1 = CellTypeResult(
        cell_type=cell_type,
        confidence=float(s1_result.probabilities.get(cell_type, s1_result.confidence)),
        probabilities=dict(s1_result.probabilities),
        label_set=label_set,
        features=dict(s1_result.features),
    )

    # --- Stage 3: topology refinement ---
    s3_result = refine(
        nodes, node_labels, node_confidences, trial_s1,
        apical_owner_root=apical_owner_root,
        disable=disable_stage3,
    )
    final_labels = [rl.label for rl in s3_result.labels]

    if trace is not None:
        trace["final_labels"] = list(final_labels)
        trace["final_confidences"] = list(node_confidences)

    mean_neurite_conf = float(np.mean(branch_confs)) if branch_confs else 0.0
    return {
        "final_labels": final_labels,
        "node_confidences": node_confidences,
        "s3_result": s3_result,
        "mean_neurite_conf": mean_neurite_conf,
        "cell_type": cell_type,
        "trace": trace,
        # Handed to the branch correction below rather than extracted again:
        # branch extraction is the most expensive step in labeling a cell.
        "morph": morph,
    }


# ---------------------------------------------------------------------------
# Model loading + SWC output + CLI
# ---------------------------------------------------------------------------


def load_branch_correction(model_dir: str | Path = DEFAULT_MODEL_DIR):
    """Introspection helper: return the Stage 3 branch-correction head from a
    model dir, or None if ``branch_correction.joblib`` is absent.

    The pipeline itself loads and applies the correction intrinsically (see
    ``_load_correction_cached``); this is only for callers that want to inspect
    or assert the head's presence, not a way to inject or disable it."""
    from .stage3.branch_correction import load_correction
    path = Path(model_dir) / "branch_correction.joblib"
    return load_correction(path) if path.is_file() else None


_ID_AND_TYPE_FIELDS = re.compile(rb"^(\s*\S+\s+)(\S+)")


def write_labeled_swc(
    swc_path: str | Path,
    nodes: list[SWCNode],
    labels: list[int],
    out_path: str | Path,
) -> int:
    """Write a copy of ``swc_path`` with each node's predicted type in the type column.

    Only the type field of each node line is rewritten. Every other byte of the
    input is copied through: coordinates, radii, parent links, comment lines and
    line endings. The output is therefore the reconstruction the user supplied,
    with its labels filled in.

    ``nodes`` and ``labels`` are the normalized parse the pipeline labeled. That
    parse merges a soma drawn with several points into one node and keeps every
    surviving node's id, so labels are matched back to the file by id, and the
    soma points that were merged away are written as soma (type 1). Returns the
    number of node lines written.
    """
    label_by_id = {nd.id: int(lbl) for nd, lbl in zip(nodes, labels)}
    if len(label_by_id) != len(nodes):
        raise ValueError(f"{swc_path}: node ids are not unique, so labels cannot be matched to lines")

    data = Path(swc_path).read_bytes()
    newline = b"\r\n" if b"\r\n" in data else b"\n"
    out = [b"# Auto-labeled by the SWC auto-labeler pipeline." + newline]
    seen: set[int] = set()
    n_merged_soma = 0
    for raw_line in data.splitlines(keepends=True):
        # The same line filter as pipeline.features.parse_swc, so the node lines
        # found here are exactly the nodes that were parsed and labeled.
        parts = raw_line.decode("utf-8", errors="ignore").split()
        if len(parts) < 7 or parts[0].startswith("#"):
            out.append(raw_line)
            continue
        try:
            node_id = int(float(parts[0]))
            int(float(parts[1])), int(float(parts[6]))
            [float(v) for v in parts[2:6]]
        except ValueError:
            out.append(raw_line)
            continue
        if node_id in seen:
            raise ValueError(f"{swc_path}: node id {node_id} appears twice")
        seen.add(node_id)
        if node_id not in label_by_id:
            n_merged_soma += 1
        label = str(label_by_id.get(node_id, 1)).encode()
        relabeled, n_sub = _ID_AND_TYPE_FIELDS.subn(lambda m: m.group(1) + label, raw_line, count=1)
        if n_sub != 1:
            raise ValueError(f"{swc_path}: cannot locate the type field of node {node_id}")
        out.append(relabeled)

    if not set(label_by_id) <= seen or len(seen) - n_merged_soma != len(nodes):
        raise ValueError(f"{swc_path}: the file's node lines do not match the labeled parse")
    if out[-1] and not out[-1].endswith((b"\n", b"\r")):
        out[-1] += newline
    Path(out_path).write_bytes(b"".join(out))
    return len(seen)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Auto-label an SWC neuron (Stage 1 -> 2 -> 3), and score "
                    "its quality flag if one is trained into the model dir (Stage 4)."
    )
    ap.add_argument("swc", type=Path, help="input SWC file")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output SWC path (default: <input>.labeled.swc)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="model directory (default: the shipped seed123)")
    ap.add_argument("--cell-type", choices=sorted(CELL_TYPE_LABEL_SETS), default=None,
                    help="skip Stage 1 and label under this cell type's label set; the "
                         "quality flag, trained on Stage 1's predictions, is then not scored")
    ap.add_argument("--no-flag", action="store_true",
                    help="skip the Stage 4 quality flag even if a flag model is present")
    ap.add_argument("--force", action="store_true",
                    help="label the file even if it fails the Stage 0 quality checks")
    args = ap.parse_args()

    nodes, notes = parse_swc_for_labeling(args.swc)
    if not nodes:
        raise SystemExit(f"No nodes parsed from {args.swc}")
    for note in notes:
        print(f"note: {note}", flush=True)
    # Stage 0: the same structural gate the corpus passed through. Labeling is
    # defined for one connected cell with a soma and a neurite, so a file that
    # fails is refused rather than labeled badly, unless the user insists.
    from .qc import qc_violations  # noqa: PLC0415
    failed = qc_violations(nodes)
    if failed:
        msg = f"{args.swc.name} fails the Stage 0 quality checks: " + "; ".join(failed)
        if not args.force:
            raise SystemExit(msg + "\nPass --force to label it anyway.")
        print(f"warning: {msg} (labeling anyway, --force)", flush=True)

    # Label the node list the checks above saw, so the labels and the nodes the
    # writer matches them to are the same parse.
    result = run_pipeline_on_nodes(
        nodes, str(args.swc),
        stage1_model=args.model_dir / "cell_type_classifier.pkl",
        stage2_model=args.model_dir / "branch_classifier.pkl",
        override_cell_type=args.cell_type,
    )
    out = args.out or args.swc.with_suffix(".labeled.swc")
    n_written = write_labeled_swc(args.swc, nodes, result.node_labels, out)
    line = (f"{args.swc.name}: cell_type={result.stage1.cell_type} "
            f"nodes={n_written} -> {out}")
    if args.cell_type is not None and not args.no_flag:
        # The flag reads Stage 1's prediction and confidence among its inputs and
        # is trained and measured with Stage 1 active only.
        line += ("\n  quality flag: not scored "
                 "(it reads Stage 1's prediction, which --cell-type replaces)")
    elif not args.no_flag:
        try:
            from pipeline.stage4.single_seed_flag import load_flag, score_flag  # noqa: PLC0415
            flag = load_flag(args.model_dir)
            if flag is not None:
                fr = score_flag(nodes, result, flag)
                if fr.get("out_of_scope"):
                    line += ("\n  quality flag: not scored "
                             f"(this model dir has no flag for {fr['out_of_scope']} cells)")
                else:
                    verdict = "FLAGGED for review" if fr["flagged"] else "ok"
                    line += (f"\n  quality flag: suspicion={fr['suspicion']:.3f} "
                             f"(predicted per-cell F1 {fr['predicted_f1']:.3f})  ({verdict})")
        except Exception as exc:  # the flag is a bonus; never break labeling
            line += f"\n  quality flag: unavailable ({exc})"
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
