"""Score one seed's held-out test cells into flag-model labels + features.

Runs the full pipeline (real Stage 1, no ground-truth override) on the seed's
hash-split test cells and writes two CSVs:

  - labels:   per-cell ground-truth F1 (the flag target ``held_out_F1``) plus
              deployment-safe predicted counts and Stage 1 output;
  - features: deployment-safe confidence and geometry features
              (see ``pipeline.stage4.flag_features``).

Each cell is labeled once. Test membership is the deterministic hash split, so
the scored cells were never in the seed model's training set.

Usage:
    python -m evaluation.stage4.score_flag_dataset --seed 123

Output (evaluation/results/flag/, created if missing):
    final_flag_seed{seed}_labels.csv
    final_flag_seed{seed}_features.csv
    final_flag_seed{seed}_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                    # noqa: E402
from pipeline.infer_pipeline import run_pipeline_on_nodes                        # noqa: E402
from pipeline.stage4.flag_features import (                               # noqa: E402
    _branch_summary,
    _confidence_summary,
    _predicted_geometry,
    _round_row,
)
from pipeline.stage4.flag_geom_features import extract_features as _extract_geom  # noqa: E402
from evaluation.paths import display_path as _display                     # noqa: E402
from evaluation.stage4._flag_metrics import (                             # noqa: E402
    _per_class_f1,
    _structural_features,
    check_model_seed,
    in_test,
    per_cell_neurite_f1,
)


CELL_TYPES = ("pyramidal", "interneuron")
DATA = ROOT / "data"


def _geom_features(nodes, labels, confs, pr, cell_type_gt) -> dict[str, float]:
    """Geometry/anatomy features of the predicted labels, prefixed ``geom_``.

    Computed from the pipeline result already in hand, so it costs one extra
    featurization and no extra pipeline run. The featurizer encodes pyramidal
    anatomy (apical trunk coherence, apical-vs-basal separation, fragmentation).
    It is emitted for every true pyramidal cell, the rows the pyramidal flag is
    fitted and evaluated on, and for every cell Stage 1 routes to pyramidal, the cells a
    deployment scores, where ``assemble_flag_features`` computes the same block.
    The remaining interneuron rows carry no ``geom_`` columns.
    """
    if cell_type_gt != "pyramidal" and pr.stage1.cell_type != "pyramidal":
        return {}
    f = _extract_geom(
        labels, confs,
        [n.x for n in nodes], [n.y for n in nodes], [n.z for n in nodes],
        [n.radius for n in nodes], [n.id for n in nodes], [n.parent for n in nodes],
        pr.stage1.cell_type, float(pr.stage1.confidence),
    )
    return {f"geom_{k}": v for k, v in f.items()}


def _load_model_dir(model_dir: Path) -> dict:
    s1 = model_dir / "cell_type_classifier.pkl"
    s2 = model_dir / "branch_classifier.pkl"
    for p in (s1, s2):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")
    return {"stage1_model": s1, "stage2_model": s2}


def _test_files(seed: int, test_size: float) -> list[tuple[str, Path]]:
    return [
        (ct, p)
        for ct in CELL_TYPES
        for p in sorted((DATA / ct).glob("*.swc"))
        if in_test(p.name, seed, test_size)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="default pipeline/models/seed{seed}")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results" / "flag")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    model_dir = (args.model_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")).resolve()
    check_model_seed(model_dir, args.seed, args.test_size)
    model = _load_model_dir(model_dir)
    files = _test_files(args.seed, args.test_size)
    if args.max_cells is not None:
        files = files[: args.max_cells]

    out_labels = args.out_dir / f"final_flag_seed{args.seed}_labels.csv"
    out_features = args.out_dir / f"final_flag_seed{args.seed}_features.csv"
    out_summary = args.out_dir / f"final_flag_seed{args.seed}_summary.json"

    print(f"=== Score seed {args.seed} test split for flag model ===")
    print(f"  model:   {_display(model_dir)}")
    print(f"  cells:   {len(files)}")

    label_rows: list[dict] = []
    feature_rows: list[dict] = []
    n_failed = 0
    t0 = time.perf_counter()

    for i, (ct_gt, path) in enumerate(files, 1):
        try:
            nodes = parse_swc(path)
            if not nodes:
                n_failed += 1
                continue
            gt = [n.type for n in nodes]
            pr = run_pipeline_on_nodes(
                nodes,
                file_path=str(path),
                stage1_model=model["stage1_model"],
                stage2_model=model["stage2_model"],
            )
            labels = list(pr.node_labels)
            confs = [float(c) for c in pr.node_confidences]

            pc_f1 = per_cell_neurite_f1(gt, labels, ct_gt)
            per_class = _per_class_f1(gt, labels)
            n_correct = sum(1 for g, q in zip(gt, labels) if g == q)
            pc_acc = n_correct / max(1, len(gt))
            structural = _structural_features(nodes)
            pred_counter = Counter(labels)

            label_rows.append({
                "file": path.name,
                "cell_type_gt": ct_gt,
                "n_nodes": len(gt),
                "seed_used": f"seed{args.seed}",
                "held_out_F1": f"{pc_f1:.4f}",
                "acc": f"{pc_acc:.4f}",
                "axon_F1": f"{per_class['axon_f1']:.4f}" if not np.isnan(per_class["axon_f1"]) else "",
                "basal_F1": f"{per_class['basal_f1']:.4f}" if not np.isnan(per_class["basal_f1"]) else "",
                "apical_F1": f"{per_class['apical_f1']:.4f}" if not np.isnan(per_class["apical_f1"]) else "",
                "stage1_pred": pr.stage1.cell_type,
                "stage1_conf": f"{float(pr.stage1.confidence):.4f}",
                "stage1_correct": str(pr.stage1.cell_type == ct_gt),
                "n_components": int(structural.get("n_components", 1)),
                "soma_z": f"{structural.get('soma_z', 0):.2f}" if not np.isnan(structural.get("soma_z", float("nan"))) else "",
                "apical_mean_z": f"{structural.get('apical_mean_z', 0):.2f}" if not np.isnan(structural.get("apical_mean_z", float("nan"))) else "",
                "apical_above_soma": f"{structural.get('apical_above_soma', 0):.2f}" if not np.isnan(structural.get("apical_above_soma", float("nan"))) else "",
                "apical_z_extent": f"{structural.get('apical_z_extent', 0):.2f}",
                "apical_reach_um": f"{structural.get('apical_reach_um', 0):.2f}",
                "apical_basal_cos": ("" if np.isnan(structural.get("apical_basal_cos", float("nan")))
                                     else f"{structural['apical_basal_cos']:.4f}"),
                "axon_frac": f"{structural.get('axon_frac', 0):.3f}",
                "apical_frac": f"{structural.get('apical_frac', 0):.3f}",
                "basal_frac": f"{structural.get('basal_frac', 0):.3f}",
                "pred_axon": pred_counter.get(2, 0),
                "pred_basal": pred_counter.get(3, 0),
                "pred_apical": pred_counter.get(4, 0),
                "source": path.name.split("__", 1)[0] if "__" in path.name else "",
            })

            # Node count, Stage 1's call and the predicted class counts are not
            # repeated here: the labels row above carries them, the two tables
            # join on file, and a second copy would reach the model as a
            # duplicate column.
            feature = {
                "file": path.name,
                **_confidence_summary(labels, confs),
                **_branch_summary(nodes, labels, confs),
                **_predicted_geometry(nodes, labels),
                # Geometry/anatomy of the PREDICTED labels, from this same
                # pipeline result. The strongest optional feature block
                # (+0.012 kept-P10) and free: no second model, no extra run.
                # Pyramidal only -- the featurizer encodes pyramidal anatomy.
                **_geom_features(nodes, labels, confs, pr, ct_gt),
            }
            feature_rows.append(_round_row(feature))
        except Exception as exc:
            print(f"  WARN {path.name}: {exc}", flush=True)
            n_failed += 1
            continue

        if i % args.progress_every == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(files) - i) / max(1, i)
            print(f"  ... {i}/{len(files)} ({elapsed:.1f} min, ETA {eta:.1f} min)", flush=True)

    if not label_rows:
        raise SystemExit("No label rows produced")

    def _fieldnames(rows: list[dict]) -> list[str]:
        """Every column any row carries, first-seen order.

        Interneuron rows have no ``geom_`` columns, so taking the header from
        the first row alone makes the file depend on which cell type comes
        first: a later row with extra keys aborts the write outright.
        """
        names: dict[str, None] = {}
        for r in rows:
            names.update(dict.fromkeys(r))
        return list(names)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with out_labels.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames(label_rows))
        writer.writeheader()
        writer.writerows(label_rows)
    with out_features.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames(feature_rows))
        writer.writeheader()
        writer.writerows(feature_rows)

    f1s = np.asarray([float(r["held_out_F1"]) for r in label_rows], dtype=float)
    cts = np.asarray([r["cell_type_gt"] for r in label_rows], dtype=object)
    summary = {
        "seed": args.seed,
        "model_dir": _display(model_dir),
        "n_scored": len(label_rows),
        "n_failed": n_failed,
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "f1_mean": float(f1s.mean()),
        "f1_p10": float(np.percentile(f1s, 10)),
        "f1_p25": float(np.percentile(f1s, 25)),
        "bad_rate_f050": float((f1s < 0.5).mean()),
        "bad_rate_f060": float((f1s < 0.6).mean()),
        "by_cell_type": {
            ct: {
                "n": int((cts == ct).sum()),
                "f1_mean": float(f1s[cts == ct].mean()) if np.any(cts == ct) else None,
                "f1_p10": float(np.percentile(f1s[cts == ct], 10)) if np.any(cts == ct) else None,
                "bad_rate_f060": float((f1s[cts == ct] < 0.6).mean()) if np.any(cts == ct) else None,
            }
            for ct in CELL_TYPES
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone: {len(label_rows)} scored, {n_failed} failed")
    # A handful of unparseable cells is tolerable; a systematic failure would
    # quietly hand the flag a much smaller training table than it asked for.
    if n_failed > max(5, 0.02 * len(files)):
        raise SystemExit(
            f"{n_failed} of {len(files)} cells failed to score, more than 2%: the "
            f"tables above are too incomplete to fit a flag on. Fix the cause "
            f"rather than training on what survived.")
    print(f"Wrote {_display(out_labels)}")
    print(f"Wrote {_display(out_features)}")
    print(
        f"F1 mean={summary['f1_mean']:.4f} P10={summary['f1_p10']:.4f} "
        f"bad<.5={summary['bad_rate_f050']:.3f} bad<.6={summary['bad_rate_f060']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
