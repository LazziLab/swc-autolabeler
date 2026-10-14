"""Correction scope ladder: how far from the soma the branch correction should reach.

The Stage 3 branch correction re-decides axon/basal/apical per branch from local
geometry alone. This script retrains it under each candidate scope D (a branch
takes part when its midpoint lies within D um of the soma by path distance; "every"
is the unscoped recipe) on the seed's training split, then applies it within the
same scope to the pre-correction labels of the seed's held-out pyramidal cells
(Stage 2 plus the topology rules, ground-truth cell type) and of the external
pyramidal set. For each row it reports pooled node accuracy and neurite macro-F1,
per-cell neurite-F1 mean and 10th percentile, the number of cells below 0.60, and
the share of branches overridden. The shipped scope is
``pipeline.stage3.train_branch_correction.DEFAULT_SCOPE_UM``.

Two caches under ``evaluation/results/_scope_ladder_cache/`` make reruns cheap:
per pyramidal file the branch features and targets (seed independent), and per
seed the pre-correction labels of its held-out cells and of the external set.
The override logic is the offline mirror of ``branch_correction.apply_correction``
(same scope mask, threshold and class map) over the cached branches.

    python -m evaluation.correction_scope_ladder --seed 42
    python -m evaluation.correction_scope_ladder --seed 42,123,789 --dists 100,150,200,300,500

Writes ``evaluation/results/correction_scope_ladder_seed{S}.{json,txt}``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.features import parse_swc                                       # noqa: E402
from pipeline.branch_features import BRANCH_FEATURE_NAMES, extract_branches   # noqa: E402
from pipeline.stage3.branch_correction import load_correction, scope_mask   # noqa: E402
from pipeline.stage3.train_branch_correction import (                       # noqa: E402
    train_correction_for_seed, DEFAULT_SCOPE_UM)
from evaluation.paths import display_path                                    # noqa: E402
from evaluation.stage4._flag_metrics import per_cell_neurite_f1
from pipeline.split import in_test      # noqa: E402

RESULTS = ROOT / "evaluation" / "results"
CACHE = RESULTS / "_scope_ladder_cache"
DATA = ROOT / "data"
DEFAULT_DISTS = "100,150,200,300,500"
DEFAULT_TEST_SIZE = 0.20
_W: dict = {}


def _feature_stamp() -> str:
    """Identifies the branch feature layout a cache entry was built from.

    The file name says which cell a cached entry describes and nothing about how
    it was built, so a changed branch feature set was reused silently: the arrays
    still loaded, still had a plausible shape, and described different
    quantities. Hashing the feature names catches a change of count, of order and
    of meaning alike.
    """
    return hashlib.md5("|".join(BRANCH_FEATURE_NAMES).encode()).hexdigest()[:12]


def _model_stamp(model_dir: Path) -> str:
    """Identifies the Stage 1 and 2 models a cached pre-correction label came from.

    The cached labels are this pipeline's output, so retrained models make them
    stale while leaving them loadable. Hashing the model files is what notices.
    """
    h = hashlib.md5()
    for name in ("cell_type_classifier.pkl", "branch_classifier.pkl"):
        f = model_dir / name
        h.update(f.read_bytes() if f.is_file() else b"missing")
    return h.hexdigest()[:12]


def _stamp_of(path: Path, key: str = "stamp") -> str | None:
    try:
        with np.load(path) as z:
            return str(z[key]) if key in z else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# caches
# ----------------------------------------------------------------------------
def _feature_path(group: str, swc: Path) -> Path:
    return CACHE / "features" / group / (swc.stem + ".npz")


def _label_path(seed: int, group: str, swc: Path) -> Path:
    return CACHE / f"labels_seed{seed}" / group / (swc.stem + ".npz")


def _save_npz(out: Path, **arrays) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out)


def _cache_features(task: tuple[str, str]) -> str | None:
    """Branch features, branch targets and node index lists of one pyramidal file."""
    group, swc = task
    p = Path(swc)
    out = _feature_path(group, p)
    stamp = _feature_stamp()
    if out.is_file() and _stamp_of(out) == stamp:
        return str(out)
    nodes = parse_swc(p)
    if not nodes:
        return None
    morph = extract_branches(nodes, cell_type="pyramidal", file_path=str(p))
    if not morph.branches:
        return None
    _save_npz(out,
              feats=np.stack([br.features for br in morph.branches]).astype(np.float32),
              gt_br=np.asarray([br.gt_label for br in morph.branches], dtype=np.int8),
              n_nodes=np.asarray([len(br.node_indices) for br in morph.branches], dtype=np.int32),
              node_flat=np.concatenate([np.asarray(br.node_indices, dtype=np.int32) for br in morph.branches]),
              node_starts=np.cumsum([0] + [len(br.node_indices) for br in morph.branches]).astype(np.int64),
              gt=np.asarray([nd.type for nd in nodes], dtype=np.int8),
              stamp=np.asarray(stamp))
    return str(out)


def _init_labels(seed: int, stamp: str) -> None:
    """Per-worker setup: the seed's model paths and the stamp those models carry."""
    md = ROOT / "pipeline" / "models" / f"seed{seed}"
    _W["s1"], _W["s2"], _W["seed"] = md / "cell_type_classifier.pkl", md / "branch_classifier.pkl", seed
    _W["stamp"] = stamp


def _cache_labels(task: tuple[str, str]) -> str | None:
    """Pre-correction labels (Stage 2 plus topology rules, pyramidal path) of one cell."""
    group, swc = task
    p = Path(swc)
    out = _label_path(_W["seed"], group, p)
    if out.is_file() and _stamp_of(out) == _W["stamp"]:
        return str(out)
    from pipeline.infer_pipeline import run_pipeline_on_nodes
    nodes = parse_swc(p)
    if not nodes:
        return None
    trace: dict = {}
    run_pipeline_on_nodes(nodes, str(p), _W["s1"], _W["s2"], override_cell_type="pyramidal", trace=trace)
    _save_npz(out, pre=np.asarray(trace["final_labels"], dtype=np.int8),
              stamp=np.asarray(_W["stamp"]))
    return str(out)


def _run_pool(fn, tasks, workers, initializer=None, initargs=(), label="") -> list[str]:
    t0 = time.perf_counter()
    done: list[str] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as ex:
        for i, r in enumerate(ex.map(fn, tasks, chunksize=4), 1):
            if r:
                done.append(r)
            if i % 500 == 0:
                print(f"  {label} {i}/{len(tasks)} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    print(f"{label}: {len(done)} files ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    return done


# ----------------------------------------------------------------------------
# training and scoring
# ----------------------------------------------------------------------------
def _training_records(feature_paths: list[str]) -> list[dict]:
    """The ``build_features`` record layout, read from the corpus feature cache."""
    recs = []
    for fp in feature_paths:
        with np.load(fp) as z:
            recs.append({"file": Path(fp).stem + ".swc", "feats": z["feats"],
                         "branch_gt": z["gt_br"], "n_nodes": z["n_nodes"]})
    return recs


def _score(cells: list[tuple[str, str]], bundle: dict | None) -> dict:
    """Pooled and per-cell metrics of one correction bundle (None = no correction)."""
    cm = np.zeros((4, 4), dtype=np.int64)
    pcs: list[float] = []
    n_over = n_br = 0
    if bundle is not None:
        cmap = {int(k): int(v) for k, v in bundle["class_map"].items()}
        classes = np.array([cmap[i] for i in range(len(cmap))])
        tau = float(bundle["override_tau"])
    for fp, lp in cells:
        with np.load(fp) as z:
            feats, gt = z["feats"], z["gt"].astype(int)
            flat, starts = z["node_flat"], z["node_starts"]
        with np.load(lp) as z:
            lab = z["pre"].astype(int)
        if bundle is not None:
            proba = bundle["model"].predict_proba(feats)
            cls, conf = classes[proba.argmax(1)], proba.max(1)
            for bi in np.flatnonzero(scope_mask(bundle, feats)):
                idx = flat[starts[bi]:starts[bi + 1]]
                cur = int(lab[idx[0]])
                if conf[bi] >= tau and int(cls[bi]) != cur and cur in (2, 3, 4):
                    lab[idx] = int(cls[bi])
                    n_over += 1
            n_br += len(feats)
        m = (gt >= 1) & (gt <= 4) & (lab >= 1) & (lab <= 4)
        np.add.at(cm, (gt[m] - 1, lab[m] - 1), 1)
        pcs.append(per_cell_neurite_f1(gt.tolist(), lab.tolist(), "pyramidal"))
    f1 = {}
    for c in (1, 2, 3):                                   # axon, basal, apical rows of the 4x4
        tp = cm[c, c]
        fp_, fn_ = cm[:, c].sum() - tp, cm[c, :].sum() - tp
        f1[c] = float(2 * tp / max(2 * tp + fp_ + fn_, 1))
    arr = np.asarray(pcs)
    return {"n_cells": int(arr.size), "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            "neurite_macro_f1": float(np.mean(list(f1.values()))),
            "axon_f1": f1[1], "basal_f1": f1[2], "apical_f1": f1[3],
            "pcf1_mean": float(arr.mean()), "pcf1_p10": float(np.percentile(arr, 10)),
            "cells_below_060": int((arr < 0.6).sum()),
            "branches_overridden_pct": 100.0 * n_over / max(n_br, 1)}


def _fmt(name: str, r: dict) -> str:
    return (f"  {name:14s} {r['accuracy']:8.4f} {r['neurite_macro_f1']:8.4f} {r['axon_f1']:7.4f} "
            f"{r['basal_f1']:7.4f} {r['apical_f1']:7.4f} {r['pcf1_mean']:8.4f} {r['pcf1_p10']:7.4f} "
            f"{r['cells_below_060']:6d} {r['branches_overridden_pct']:8.2f}")


HEADER = (f"  {'scope':14s} {'acc':>8s} {'macro-F1':>8s} {'axon':>7s} {'basal':>7s} {'apical':>7s} "
          f"{'pcF1':>8s} {'P10':>7s} {'<0.60':>6s} {'overr.%':>8s}")


def run_seed(seed: int, dists: list[float], workers: int, limit: int, with_external: bool,
             test_size: float = DEFAULT_TEST_SIZE, out_dir: Path = RESULTS) -> dict:
    t_all = time.perf_counter()
    corpus = sorted((DATA / "pyramidal").glob("*.swc"))
    if limit:
        corpus = corpus[:limit]
    feats_all = _run_pool(_cache_features, [("corpus", str(p)) for p in corpus], workers, label="corpus features")
    held = [p for p in corpus if in_test(p.name, seed, test_size)]
    held_feats = {Path(f).stem: f for f in feats_all}
    label_tasks = [("heldout", str(p)) for p in held if p.stem in held_feats]
    ext_meta = []
    if with_external:
        with open(DATA / "external" / "pyramidal.csv", encoding="utf-8", newline="") as fh:
            ext_meta = [DATA / "external" / "pyramidal" / m["file_name"] for m in csv.DictReader(fh)]
        if limit:
            ext_meta = ext_meta[:limit]
        ext_feats = {Path(f).stem: f for f in
                     _run_pool(_cache_features, [("external", str(p)) for p in ext_meta], workers,
                               label="external features")}
        label_tasks += [("external", str(p)) for p in ext_meta if p.stem in ext_feats]
    model_stamp = _model_stamp(ROOT / "pipeline" / "models" / f"seed{seed}")
    labels = _run_pool(_cache_labels, label_tasks, workers, initializer=_init_labels,
                       initargs=(seed, model_stamp),
                       label=f"seed {seed} pre-correction labels")
    label_of = {(Path(l).parent.name, Path(l).stem): l for l in labels}
    sets = {"heldout": [(held_feats[p.stem], label_of[("heldout", p.stem)]) for p in held
                        if ("heldout", p.stem) in label_of]}
    if with_external:
        sets["external"] = [(ext_feats[p.stem], label_of[("external", p.stem)]) for p in ext_meta
                            if ("external", p.stem) in label_of]
    print(f"seed {seed}: {len(sets['heldout'])} held-out pyramidal cells"
          + (f", {len(sets['external'])} external pyramidal cells" if with_external else ""), flush=True)

    recs = _training_records(feats_all)
    rows: dict[str, dict] = {}
    # "none" applies no correction at all; "every" applies the unscoped one. Both
    # had scope_um null in the record, which left the two indistinguishable there.
    variants: list[tuple[str, float | None]] = [("none", None), ("every", None)] + [(f"{int(d)} um", d) for d in dists]
    lines = [f"CORRECTION SCOPE LADDER -- seed {seed}, ground-truth cell type, pyramidal cells only",
             f"training: pyramidal branches of the files outside seed {seed}'s test split, within the scope",
             f"shipped scope: {DEFAULT_SCOPE_UM} um (train_branch_correction.DEFAULT_SCOPE_UM)", ""]
    for name, d in variants:
        t0 = time.perf_counter()
        if name == "none":
            bundle, n_train = None, 0
        else:
            out = CACHE / f"models_seed{seed}" / f"correction_{name.replace(' ', '')}.joblib"
            n_train = train_correction_for_seed(DATA / "pyramidal", seed, out, recs=recs,
                                                scope_um=d, test_size=test_size)
            bundle = load_correction(out)
        rows[name] = {"correction": name != "none", "scope_um": d,
                      "scope": "no correction" if name == "none"
                               else "unscoped" if d is None else f"{d:g} um",
                      "train_branches": int(n_train),
                      **{s: _score(cells, bundle) for s, cells in sets.items()},
                      "minutes": round((time.perf_counter() - t0) / 60, 1)}
        print(f"{name:8s} trained on {n_train} branches ({rows[name]['minutes']} min)", flush=True)
    for s in sets:
        lines += [f"[{s}] {len(sets[s])} cells", HEADER] + [_fmt(n, rows[n][s]) for n in rows] + [""]
    text = "\n".join(lines)
    print(text, flush=True)
    result = {"seed": seed, "dists_um": dists, "shipped_scope_um": DEFAULT_SCOPE_UM,
              "test_size": test_size, "feature_stamp": _feature_stamp(),
              "model_stamp": model_stamp,
              "n_cells": {s: len(c) for s, c in sets.items()}, "rows": rows,
              "minutes": round((time.perf_counter() - t_all) / 60, 1)}
    tag = f"_limit{limit}" if limit else ""
    if test_size != DEFAULT_TEST_SIZE:
        tag += f"_test{test_size:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"correction_scope_ladder_seed{seed}{tag}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stem.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {display_path(stem)}.{{json,txt}} ({result['minutes']} min)", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", default="42", help="comma list of seeds (default 42)")
    ap.add_argument("--dists", default=DEFAULT_DISTS, help=f"comma list of scopes in um (default {DEFAULT_DISTS})")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help="held-out fraction, used for both the scoring split and "
                         "the correction's own training filter")
    ap.add_argument("--no-external", action="store_true", help="skip the external pyramidal set")
    ap.add_argument("--limit", type=int, default=0, help="first N files of each set (smoke test; output tagged)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    dists = [float(x) for x in args.dists.split(",") if x.strip()]
    for s in [int(x) for x in args.seed.split(",") if x.strip()]:
        run_seed(s, dists, args.workers, args.limit, not args.no_external,
                 test_size=args.test_size, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
