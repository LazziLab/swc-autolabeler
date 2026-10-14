"""Stage 2 evaluation: subtree-owner classifier, per cell type.

Loads a trained Stage 2 bundle and scores its subtree-owner models on a seed's
held-out test split. Each primary subtree is typed by its cell type's model and
compared to the ground-truth owner label. Reports per-subtree and node-weighted
accuracy, per-class precision / recall / F1, and the confusion matrix.

The paper reports full-pipeline per-node accuracy rather than an isolated
Stage 2 number, so this is the natural standalone Stage 2 metric. It is also
what confirms a freshly trained model reproduces the deployed one: identical
predictions give an identical confusion matrix and predictions checksum.

The test split is the deterministic hash-bucket split used throughout the
paper: a file lands in test iff md5("{seed}:{filename}") maps below the test
fraction.

Usage:
    python -m evaluation.eval_stage2 --seed 123
    python -m evaluation.eval_stage2 --seed 123 --model-dir pipeline/models/seed123

Output (evaluation/results/, created if missing):
    eval_stage2_seed{seed}.json
    eval_stage2_seed{seed}.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                                  # noqa: E402
from pipeline.infer_pipeline import load_stage2_bundle                  # noqa: E402
from pipeline.stage2.subtree_features import extract_primary_subtrees   # noqa: E402
from evaluation.paths import display_path as _display_path               # noqa: E402
from evaluation.stage4._flag_metrics import check_model_seed             # noqa: E402
from pipeline.split import in_test as _in_test

CELL_TYPES = ("pyramidal", "interneuron")
DATA = ROOT / "data"
OWNER_LABELS = {"pyramidal": [2, 3, 4], "interneuron": [2, 3]}
OWNER_NAMES = {2: "axon", 3: "basal/dendrite", 4: "apical"}


def _predict_owners(model, feats_rows) -> list[int]:
    """Mirror inference: argmax of the class probabilities, a cell at a time.

    ``pipeline.stage2.owner_features`` scores every primary subtree of a cell in
    one ``predict_proba`` call, so this batches the same way. Scoring a subtree
    at a time gives identical answers and spends the call overhead once per
    subtree instead of once per cell.
    """
    X = np.asarray(feats_rows, dtype=np.float64)
    probs = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    return [classes[int(np.argmax(row))] for row in probs]


def _score_cell_type(model, ct: str, seed: int, test_size: float) -> dict:
    from sklearn.metrics import (
        confusion_matrix, precision_recall_fscore_support,
    )
    labels = OWNER_LABELS[ct]
    y_true: list[int] = []
    y_pred: list[int] = []
    weights: list[float] = []
    recs: list[tuple] = []  # (filename, root_idx, gt, pred) -> predictions checksum

    failed: list[str] = []
    for p in sorted((DATA / ct).glob("*.swc")):
        if not _in_test(p.name, seed, test_size):
            continue
        try:
            nodes = parse_swc(str(p))
        except Exception:
            failed.append(p.name)       # counted, not silently dropped
            continue
        if not nodes:
            failed.append(p.name)
            continue
        scored = [sub for sub in extract_primary_subtrees(nodes)
                  if int(sub.gt_label) in labels]
        if not scored:
            continue
        preds = _predict_owners(model, [sub.features for sub in scored])
        for sub, pred in zip(scored, preds):
            y_true.append(int(sub.gt_label))
            y_pred.append(pred)
            weights.append(float(sum(sub.gt_label_counts.values())))
            recs.append((p.name, int(sub.root_idx), int(sub.gt_label), pred))

    n = len(y_true)
    if n == 0:
        return {"n_subtrees": 0, "n_failed": len(failed), "failed": failed}

    w = np.asarray(weights)
    correct = np.asarray(y_true) == np.asarray(y_pred)
    p_, r_, f_, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    per_class = {
        OWNER_NAMES[l]: {"precision": float(p_[i]), "recall": float(r_[i]),
                         "f1": float(f_[i]), "support": int(sup[i])}
        for i, l in enumerate(labels)
    }
    # Deterministic checksum of predictions (sorted, so environment-independent).
    blob = "\n".join(f"{fn}|{rid}|{g}|{pr}" for fn, rid, g, pr in sorted(recs))
    checksum = hashlib.md5(blob.encode()).hexdigest()

    return {
        "n_subtrees": n,
        "n_failed": len(failed),
        "failed": failed,
        "subtree_accuracy": float(correct.mean()),
        "node_weighted_accuracy": float(w[correct].sum() / w.sum()),
        "n_nodes": int(w.sum()),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_labels": [OWNER_NAMES[l] for l in labels],
        "predictions_md5": checksum,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="dir holding branch_classifier.pkl (default pipeline/models/seed{seed})")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results")
    args = ap.parse_args()

    model_dir = args.model_dir or (ROOT / "pipeline" / "models" / f"seed{args.seed}")
    check_model_seed(model_dir, args.seed, args.test_size)
    model_path = model_dir / "branch_classifier.pkl"
    if not model_path.is_file():
        raise SystemExit(f"MISSING model: {model_path}")
    bundle = load_stage2_bundle(model_path)
    owner_models = bundle.get("subtree_owner_models_by_cell_type") or {}

    per_ct: dict[str, dict] = {}
    for ct in CELL_TYPES:
        model = owner_models.get(ct)
        if model is None:
            print(f"seed {args.seed}: no subtree-owner model for {ct}, skipping", flush=True)
            continue
        per_ct[ct] = _score_cell_type(model, ct, args.seed, args.test_size)
        print(f"seed {args.seed}: scored {ct} "
              f"({per_ct[ct].get('n_subtrees', 0)} subtrees)", flush=True)

    result = {
        "seed": args.seed,
        "test_size": args.test_size,
        "model": _display_path(model_path),
        "per_cell_type": per_ct,
    }

    lines = [f"Stage 2 subtree-owner evaluation (seed {args.seed})",
             f"model : {result['model']}", ""]
    for ct in CELL_TYPES:
        m = per_ct.get(ct)
        if not m or m.get("n_subtrees", 0) == 0:
            continue
        lines += [
            f"[{ct}]  {m['n_subtrees']} subtrees / {m['n_nodes']} nodes",
            f"  subtree accuracy       : {m['subtree_accuracy']:.4f}",
            f"  node-weighted accuracy : {m['node_weighted_accuracy']:.4f}",
            f"  predictions md5        : {m['predictions_md5']}",
            f"  {'class':14s} {'precision':>9s} {'recall':>8s} {'f1':>8s} {'support':>8s}",
        ]
        for name, c in m["per_class"].items():
            lines.append(f"  {name:14s} {c['precision']:9.4f} {c['recall']:8.4f} "
                         f"{c['f1']:8.4f} {c['support']:8d}")
        lines.append(f"  confusion (rows=true, cols=pred; {m['confusion_labels']}):")
        for name, row in zip(m["confusion_labels"], m["confusion_matrix"]):
            lines.append(f"    {name:14s} {row}")
        lines.append("")
    report = "\n".join(lines)
    print("\n" + report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"eval_stage2_seed{args.seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / f"eval_stage2_seed{args.seed}.txt").write_text(
        report + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir}/eval_stage2_seed{args.seed}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
