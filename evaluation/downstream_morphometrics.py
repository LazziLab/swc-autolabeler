"""Per-compartment morphometrics from predicted labels, against ground truth.

An F1 score counts how many nodes carry the right compartment label. It does not
say whether a morphometric computed from those labels is usable, and the two can
part company: errors spread over short distal twigs barely move total dendritic
length, while one inverted apical/basal call ruins every measurement for that
cell. This driver measures the second thing directly.

For each held-out cell it computes the same morphometrics twice, once from the
ground-truth labels and once from a predictor's labels, and writes both into one
row. The predictor is our pipeline, its single-model (unstaged) variant or one
of the constructed baselines, selected with ``--method``, so the resulting tables
are directly comparable.

Measures, per compartment (axon, basal, apical). Each is dominated by a
different part of the arbor, so together they expose different error shapes:

    total_length     summed edge length; dominated by long segments anywhere
    surface_area     frustum lateral area; sets membrane conductance in a
                     compartmental model, so it is the measure that matters for
                     assigning biophysical properties
    n_branch_points  interior topology; exposes fragmented arbors
    n_terminals      small distal twigs; exposes missed tufts
    max_path_dist    how far the compartment reaches from the soma, measured
                     along the tree; a single extreme node, so it exposes
                     truncation
    mean_radius      caliber
    n_nodes          raw count, for reference

Each edge is attributed to its child's compartment, the usual SWC convention,
and a compartment's branch points and terminals are counted within that
compartment's own subgraph. Geometry is identical between the two passes; only
the labels differ, so every difference is attributable to labeling alone.

Usage:
    python -m evaluation.downstream_morphometrics --method ours       --seed 123
    python -m evaluation.downstream_morphometrics --method unstaged   --seed 123
    python -m evaluation.downstream_morphometrics --method lmeasure_rf --seed 123

Output (evaluation/results/morphometrics/):
    morphometrics_{method}_seed{seed}.csv
"""
from __future__ import annotations

import argparse
import csv

import math
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                   # noqa: E402
from pipeline.infer_pipeline import (                                     # noqa: E402
    run_pipeline_on_nodes, load_branch_correction,
)
from evaluation.paths import display_path                                # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed              # noqa: E402
from pipeline.split import in_test as _in_test

DATA = ROOT / "data"
CELL_TYPES = ("pyramidal", "interneuron")
COMPARTMENTS = {2: "axon", 3: "basal", 4: "apical"}
MEASURES = ("n_nodes", "total_length", "surface_area", "n_branch_points",
            "n_terminals", "max_path_dist", "mean_radius")
BASELINES = ("neurom_rf", "sholl_mlp", "sholl_rf", "lmeasure_rf")


def _split(seed: int, test_size: float):
    train: dict[str, list[Path]] = {ct: [] for ct in CELL_TYPES}
    test: list[tuple[str, Path]] = []
    for ct in CELL_TYPES:
        for p in sorted((DATA / ct).glob("*.swc")):
            if _in_test(p.name, seed, test_size):
                test.append((ct, p))
            else:
                train[ct].append(p)
    return train, test


def _geometry(nodes):
    """Children lists, edge length and lateral area to parent, path distance from soma."""
    idx = {n.id: i for i, n in enumerate(nodes)}
    children: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    roots: list[int] = []
    for i, n in enumerate(nodes):
        p = idx.get(n.parent, -1)
        (children[p].append(i) if p >= 0 else roots.append(i))

    edge = [0.0] * len(nodes)
    area = [0.0] * len(nodes)
    for i, n in enumerate(nodes):
        p = idx.get(n.parent, -1)
        if p < 0:
            continue
        a, b = nodes[i], nodes[p]
        L = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
        edge[i] = L
        # Lateral surface of the truncated cone joining the two nodes.
        dr = a.radius - b.radius
        area[i] = math.pi * (a.radius + b.radius) * math.sqrt(L * L + dr * dr)

    path = [0.0] * len(nodes)
    for r in roots:
        q = deque([r])
        while q:
            i = q.popleft()
            for c in children[i]:
                path[c] = path[i] + edge[c]
                q.append(c)
    return children, edge, area, path


def _morphometrics(labels, children, edge, area, path, nodes) -> dict[str, dict[str, float]]:
    """Per-compartment measures under one labelling, on fixed geometry.

    ``max_path_dist`` is the greatest distance from the soma reached by any node
    of this compartment, measured along the whole tree, so an apical node's
    distance includes whatever lies between it and the soma. It is a reach, not a
    length within the compartment, and it is what exposes a truncated arbor.

    A compartment with no nodes gets zeros rather than being absent. That matters
    asymmetrically downstream: a ground-truth zero means the compartment was
    never traced and morphometric_utility drops the cell from that measure, while
    a predicted zero against a non-zero truth is a real and total error and is
    kept.
    """
    out: dict[str, dict[str, float]] = {}
    for code, name in COMPARTMENTS.items():
        members = [i for i, l in enumerate(labels) if l == code]
        if not members:
            out[name] = {m: 0.0 for m in MEASURES}
            continue
        kids = {i: sum(1 for c in children[i] if labels[c] == code) for i in members}
        out[name] = {
            "n_nodes": float(len(members)),
            "total_length": float(sum(edge[i] for i in members)),
            "surface_area": float(sum(area[i] for i in members)),
            "n_branch_points": float(sum(1 for i in members if kids[i] >= 2)),
            "n_terminals": float(sum(1 for i in members if kids[i] == 0)),
            "max_path_dist": float(max(path[i] for i in members)),
            "mean_radius": float(sum(nodes[i].radius for i in members) / len(members)),
        }
    return out


def _make_predictor(method: str, seed: int, model_dir: Path, train_files,
                    test_size: float = 0.20):
    """Return a (nodes, cell_type, path) -> labels function for our pipeline, its
    single-model variant or a baseline.

    ``path`` is the cell's source file. The baselines read only geometry and
    ignore it; the pipeline records it on the result it builds.
    """
    if method == "ours":
        s1 = model_dir / "cell_type_classifier.pkl"
        s2 = model_dir / "branch_classifier.pkl"
        # Only Stage 2 is required: every cell below is labeled with
        # override_cell_type, so the pipeline never consults Stage 1 and its file
        # need not exist. That lets a single-cell-type model set be scored here.
        if not s2.is_file():
            raise SystemExit(f"MISSING: {s2}")
        if load_branch_correction(model_dir) is None:
            raise SystemExit(f"MISSING branch_correction.joblib in {model_dir}")

        def predict(nodes, cell_type, path=None):
            pr = run_pipeline_on_nodes(
                nodes, file_path=str(path or ""), stage1_model=s1, stage2_model=s2,
                override_cell_type=cell_type)
            return list(pr.node_labels)
        return predict

    if method == "unstaged":
        from pipeline.unstaged import label_cell, load  # noqa: PLC0415
        bundle = load(model_dir)
        if bundle is None:
            raise SystemExit(f"MISSING unstaged_model.joblib in {model_dir}\n"
                             f"  run: python -m pipeline.train_unstaged --seed {seed}")
        # Given the cell type, as the pipeline is below, so the two are scored alike.
        return lambda nodes, cell_type, path=None: label_cell(nodes, cell_type, bundle)

    from evaluation.baselines.baselines import predict_with_cache  # noqa: PLC0415
    # Must match the directory evaluation.baselines.eval_baselines writes: a model
    # fitted under a different --test-size is a different fit.
    split_tag = "" if test_size == 0.20 else f"_test{test_size:g}"
    cache = (ROOT / "evaluation" / "results" / "baselines"
             / f"seed{seed}{split_tag}" / f"{method}.pkl")
    if not cache.is_file():
        raise SystemExit(f"MISSING baseline cache {cache}\n"
                         f"  run: python -m evaluation.baselines.eval_baselines "
                         f"--seed {seed} --method {method}")
    fitted = predict_with_cache(method, train_files, seed=seed, cache_path=cache)
    return lambda nodes, cell_type, path=None: fitted(nodes, cell_type)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="ours", choices=("ours", "unstaged", *BASELINES))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "evaluation" / "results" / "morphometrics")
    args = ap.parse_args()

    model_dir = args.model_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")
    if args.method in ("ours", "unstaged"):
        check_model_seed(model_dir, args.seed, args.test_size)
    train_files, test = _split(args.seed, args.test_size)
    if args.limit:
        test = test[:args.limit]
    print(f"{args.method}  seed {args.seed}: {len(test)} test cells", flush=True)
    predict = _make_predictor(args.method, args.seed, model_dir, train_files,
                              test_size=args.test_size)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, (ct, swc) in enumerate(test, 1):
        nodes = parse_swc(swc)
        if not nodes:
            continue
        gt_labels = [n.type for n in nodes]
        pred_labels = list(predict(nodes, ct, swc))
        children, edge, area, path = _geometry(nodes)
        gt_m = _morphometrics(gt_labels, children, edge, area, path, nodes)
        pr_m = _morphometrics(pred_labels, children, edge, area, path, nodes)

        row = {"file": swc.name, "cell_type": ct, "method": args.method,
               "n_nodes_total": len(nodes)}
        for comp in COMPARTMENTS.values():
            for m in MEASURES:
                row[f"gt_{comp}_{m}"] = round(gt_m[comp][m], 4)
                row[f"pred_{comp}_{m}"] = round(pr_m[comp][m], 4)
        rows.append(row)
        if i % 250 == 0:
            print(f"  ... {i}/{len(test)} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # A reduced run is not the published one, so it never takes its file name.
    tag = f"_limit{args.limit}" if args.limit else ""
    if args.test_size != 0.20:
        tag += f"_test{args.test_size:g}"
    out = args.out_dir / f"morphometrics_{args.method}_seed{args.seed}{tag}.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {display_path(out)}  ({len(rows)} cells, "
          f"{(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
