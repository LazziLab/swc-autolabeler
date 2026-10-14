"""Cell classes the pipeline never trained on: does the frozen model transfer,
and does retraining on the class's own cells work?

The pipeline's label vocabulary is soma / axon / basal / apical, served by two
schemas: four-class (pyramidal) and three-class without apical (interneuron).
Any cell class whose labels form a subset of that vocabulary maps onto one of
the two. Granule and retinal ganglion cells carry an axon and undifferentiated
dendrites and map onto the interneuron schema. The Urban olfactory bulb
projection cells (tagged mitral, tufted, or both in the repository, and
therefore treated as one mitral/tufted corpus) additionally distinguish the
apical tuft dendrite (code 4) from the lateral dendrites (code 3), the same
three neurite classes as a pyramidal cell; they test the pyramidal schema on
a class that is not pyramidal.

``fetch`` queries NeuroMorpho for each class in archives the corpus never used,
keeps cells whose files actually contain axon and dendrite nodes and pass QC,
and downloads up to the per-class cap. Candidates that failed the same content
check: Purkinje (axon and dendrites stored as separate files, or no soma),
medium spiny (Smith, 2 of 25 probed files carried axon), motoneuron (zebrafish
archives fail QC; Cameron, 12 of 25 usable).

``run``, per class: evaluates the frozen seed-123 model on every cell of the
class, via the class's schema branch and in the deployment configuration
(Stage 1 predicting a type it cannot know), which measures transfer; then
trains a fresh model on two thirds of the cells (Stage 2 only; the flag does
not apply) and scores it, and the frozen model again, on the held-out third,
so the two are compared on the same cells. For interneuron-schema classes, stray apical codes in the
ground truth collapse to dendrite; for the pyramidal-schema mitral/tufted
corpus the apical code is kept and scored.

Usage:
    python -m evaluation.generalization.new_class fetch [--classes granule,ganglion,mitral_tufted]
    python -m evaluation.generalization.new_class run   [--classes granule,ganglion,mitral_tufted]

Output:
    data/external/<class>/*.swc + <class>.csv                 (fetch)
    data/external/manifests/candidates_<class>.csv            (fetch)
    evaluation/results/generalization/new_class.json          (run)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pipeline._scan_corpus_qc as _qc                                    # noqa: E402
from pipeline.features import parse_swc                                   # noqa: E402
from pipeline.infer_pipeline import run_pipeline_on_nodes                     # noqa: E402
from evaluation.eval_pipeline import node_metrics                             # noqa: E402
from evaluation.paths import display_path                                 # noqa: E402
from evaluation.stage4._flag_metrics import per_cell_neurite_f1           # noqa: E402
from evaluation.generalization.fetch_external import (                    # noqa: E402
    download_cng, type_counts, corpus_identity, _fetch_json, API,
)

EXTERNAL = ROOT / "data" / "external"
MANIFESTS = EXTERNAL / "manifests"
RESULTS = ROOT / "evaluation" / "results" / "generalization"
WORK = RESULTS / "lab_retrain_work"

# class -> (NeuroMorpho query, chosen archive, download cap)
CLASSES = {
    "granule": (("cell_type:granule",), "Rodrigues", 150),
    "ganglion": (("cell_type:ganglion",), "Rodger", 150),
    # NeuroMorpho tags most Urban olfactory bulb projection cells with both
    # types (32 of 40 mitral-tagged cells are also tufted-tagged), so mitral
    # and tufted form one corpus at the archive's own granularity: the union
    # of both queries, deduplicated by neuron.
    "mitral_tufted": (("cell_type:mitral", "cell_type:tufted"), "Urban", 120),
}
# Which of the pipeline's two schemas a class maps onto. Granule and retinal
# ganglion carry axon + one dendrite kind (interneuron schema). Every Urban
# mitral/tufted file distinguishes the apical tuft (code 4) from the lateral
# dendrites (code 3), the same three neurite classes as a pyramidal cell, so
# that corpus tests the pyramidal schema.
SCHEMA = {"granule": "interneuron", "ganglion": "interneuron",
          "mitral_tufted": "pyramidal"}
DEFAULT_SEED = 123
#: A two-thirds training split, as the docstring says.
DEFAULT_TEST_SIZE = 1.0 / 3.0


def _query_class(query: str) -> list[dict]:
    rows, page = [], 0
    while True:
        d = _fetch_json(f"{API}/neuron/select?q={query}&size=500&page={page}")
        ns = d.get("_embedded", {}).get("neuronResources", [])
        if not ns:
            break
        rows += [{"neuron_name": n.get("neuron_name", ""),
                  "archive": n.get("archive", ""),
                  "species": n.get("species", ""),
                  "brain_region": ";".join(n.get("brain_region") or []),
                  "domain": n.get("domain", "")} for n in ns]
        total = d.get("page", {}).get("totalPages", 0)
        page += 1
        if page >= total:
            break
        time.sleep(0.3)
    return rows


CANDIDATE_FIELDS = ["neuron_name", "archive", "species", "brain_region", "domain"]
META_FIELDS = ["file_name", "cell_type", "source", "neuron_name", "archive",
               "species", "brain_region"]


def cmd_fetch(class_names: list[str], force: bool = False) -> None:
    used_archives, _ = corpus_identity()
    for cls in class_names:
        queries, archive, cap = CLASSES[cls]
        rows, seen = [], set()
        for query in queries:
            for r in _query_class(query):
                key = (r["archive"], r["neuron_name"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
        cand = [r for r in rows if r["archive"].strip().lower() not in used_archives
                and "axon" in r["domain"].lower()]
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        # Fixed field names: the header came from the first row, so an empty
        # query raised IndexError rather than saying the query found nothing.
        with (MANIFESTS / f"candidates_{cls}.csv").open("w", encoding="utf-8",
                                                        newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(cand)
        if not cand:
            print(f"{cls}: no candidate cells outside the training archives", flush=True)
            continue

        rows = [r for r in cand if r["archive"] == archive]
        random.Random(123).shuffle(rows)
        swc_dir = EXTERNAL / cls
        swc_dir.mkdir(parents=True, exist_ok=True)
        meta = []
        for r in rows:
            if len(meta) >= cap:
                break
            fn = (f"{archive}__"
                  f"{r['neuron_name'].replace(' ', '_').replace('/', '_')}.swc")
            dest = swc_dir / fn
            if not dest.exists() and not download_cng(archive, r["neuron_name"], dest):
                continue
            tc = type_counts(dest)
            need_apical = SCHEMA[cls] == "pyramidal" and not tc.get(4)
            if not tc.get(2) or not tc.get(3) or need_apical or not _qc.qc_one_file(dest, data_root=EXTERNAL).get("qc_pass"):
                dest.unlink(missing_ok=True)
                continue
            meta.append({"file_name": fn, "cell_type": cls, "source": "neuromorpho",
                         "neuron_name": r["neuron_name"], "archive": archive,
                         "species": r["species"], "brain_region": r["brain_region"]})
        # The repository moves, so a re-fetch can return fewer or different cells
        # than the corpus the paper reports on. A smaller result is refused rather
        # than written over the published manifest.
        meta_path = EXTERNAL / f"{cls}.csv"
        if meta_path.is_file() and not force:
            with meta_path.open(encoding="utf-8", newline="") as fh:
                before = list(csv.DictReader(fh))
            if len(meta) < len(before):
                raise SystemExit(
                    f"{cls}: this fetch found {len(meta)} usable cells where "
                    f"{meta_path.name} records {len(before)}. The published corpus "
                    f"would shrink. Pass --force to overwrite it.")
        with meta_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=META_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(meta)
        print(f"{cls}: kept {len(meta)} usable cells from {archive}", flush=True)


def evaluate(cls: str, files: list[str], mdir: Path,
             deployment: bool = False) -> dict:
    """Score one model on the given files of the class under the class's schema
    (see ``SCHEMA``). For three-class classes, stray apical codes in the
    ground truth collapse to dendrite; for the four-class mitral corpus the
    apical code is kept and scored. With ``deployment`` Stage 1 predicts the
    cell type instead of the schema being forced; the routing fraction is
    reported."""
    schema = SCHEMA[cls]
    s1, s2 = mdir / "cell_type_classifier.pkl", mdir / "branch_classifier.pkl"
    gts, preds, pc = [], [], []
    routed_pyramidal = 0
    for f in files:
        path = EXTERNAL / cls / f
        nodes = parse_swc(path)
        pr = run_pipeline_on_nodes(
            nodes, str(path), s1, s2, override_cell_type=None if deployment else schema)
        if deployment and pr.stage1.cell_type == "pyramidal":
            routed_pyramidal += 1
        gt = np.array([n.type for n in nodes])
        if schema == "interneuron":
            gt = np.where(gt == 4, 3, gt)
        lab = np.array(list(pr.node_labels))
        gts.append(gt)
        preds.append(lab)
        pc.append(per_cell_neurite_f1(gt, list(lab), schema))
    labels = [1, 2, 3] if schema == "interneuron" else [1, 2, 3, 4]
    gt_all, pred_all = np.concatenate(gts), np.concatenate(preds)
    mm = node_metrics(gt_all, pred_all, labels)
    pc = np.array(pc)
    # Node accuracy over neurite nodes only. node_metrics counts the soma, which
    # the pipeline places by geometry and always gets right, so including it
    # reports a higher accuracy than every table in the paper, which excludes it.
    neurite = gt_all != 1
    out = {"node_acc": float((gt_all[neurite] == pred_all[neurite]).mean()),
           "node_acc_with_soma": mm["accuracy"],
           "neurite_macro_f1": mm["neurite_macro_f1"],
           "axon_f1": mm["per_class"]["axon"]["f1"],
           "pcf1_mean": float(pc.mean()),
           "pcf1_p10": float(np.percentile(pc, 10))}
    # Class 3 is the only dendrite under the interneuron schema and the basal
    # dendrite under the pyramidal one, so it is named for what it is here
    # rather than called a dendrite on a corpus that also has an apical.
    if schema == "pyramidal":
        out["basal_f1"] = mm["per_class"]["basal"]["f1"]
        out["apical_f1"] = mm["per_class"]["apical"]["f1"]
    else:
        out["dendrite_f1"] = mm["per_class"]["basal"]["f1"]
    if deployment:
        out["stage1_pyramidal_frac"] = routed_pyramidal / max(len(files), 1)
    return out


def cmd_run(class_names: list[str], seed: int = DEFAULT_SEED,
            test_size: float = DEFAULT_TEST_SIZE, out: Path | None = None,
            keep_work: bool = False, device: str = "cuda") -> None:
    result_path = out or (RESULTS / "new_class.json")
    result = (json.loads(result_path.read_text(encoding="utf-8"))
              if result_path.is_file() else {})
    result.pop("_run", None)
    frozen_dir = ROOT / "pipeline" / "models" / f"seed{seed}"
    for cls in class_names:
        _, archive, _ = CLASSES[cls]
        meta = list(csv.DictReader((EXTERNAL / f"{cls}.csv")
                                   .open(encoding="utf-8", newline="")))
        files = [m["file_name"] for m in meta]
        schema = SCHEMA[cls]
        corpus = WORK / "corpora" / f"{cls}_{archive}"
        (corpus / "interneuron").mkdir(parents=True, exist_ok=True)
        (corpus / "pyramidal").mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = corpus / schema / f
            if not dst.exists():
                shutil.copy2(EXTERNAL / cls / f, dst)
        mdir = WORK / "models" / f"{cls}_{archive}"
        print(f"{cls}/{archive}: training on ~{int(len(files) * (1 - test_size))} cells",
              flush=True)
        t0 = time.perf_counter()
        run = subprocess.run(
            [sys.executable, "-m", "pipeline.train_pipeline",
             "--data-dir", str(corpus), "--seed", str(seed),
             "--test-size", str(test_size), "--out-dir", str(mdir),
             "--skip", "stage1,correction,flag", "--device", device],
            cwd=str(ROOT), capture_output=True, text=True)
        # Wall clock of the whole training command, process start included.
        train_s = time.perf_counter() - t0
        if run.returncode != 0 or not (mdir / "branch_classifier.pkl").is_file():
            print("TRAIN FAILED:")
            print("\n".join(run.stderr.splitlines()[-10:]))
            raise SystemExit(1)

        split = json.loads((mdir / "eval_split.json").read_text(encoding="utf-8"))
        test_files = [Path(f).name for f in split["test_files"][schema]]
        own = evaluate(cls, test_files, mdir)
        frozen = evaluate(cls, test_files, frozen_dir)
        frozen_dep = evaluate(cls, test_files, frozen_dir, deployment=True)
        # Transfer needs no training cells, so it is measured on all of them.
        frozen_all = evaluate(cls, files, frozen_dir)
        frozen_dep_all = evaluate(cls, files, frozen_dir, deployment=True)
        result[f"{cls}/{archive}"] = {"n_train": len(files) - len(test_files), "train_s": round(train_s, 2),
                                      "n_test": len(test_files), "n_all": len(files),
                                      "schema": schema,
                                      "own": own, "frozen": frozen,
                                      "frozen_deployment": frozen_dep,
                                      "frozen_all": frozen_all,
                                      "frozen_deployment_all": frozen_dep_all}
        print(f"{cls}/{archive}  train={len(files) - len(test_files)} "
              f"test={len(test_files)}  schema={schema}")
        keys = ["node_acc", "axon_f1", "pcf1_mean", "pcf1_p10"]
        keys[2:2] = ["basal_f1", "apical_f1"] if schema == "pyramidal" else ["dendrite_f1"]
        for k in keys:
            print(f"  {k:12s} frozen {frozen[k]:.3f}  (deploy "
                  f"{frozen_dep[k]:.3f})  ->  own {own[k]:.3f}")
        print(f"  stage1 routes {frozen_dep['stage1_pyramidal_frac']:.0%} pyramidal")
        print(f"  all {len(files)} cells, frozen: pcF1 mean {frozen_all['pcf1_mean']:.3f} "
              f"(deploy {frozen_dep_all['pcf1_mean']:.3f}), axon F1 {frozen_all['axon_f1']:.3f}, "
              f"routed pyramidal {frozen_dep_all['stage1_pyramidal_frac']:.0%}")
    result["_run"] = {"classes": list(class_names), "seed": seed,
                      "test_size": test_size, "device": device,
                      "frozen_models": display_path(frozen_dir)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not keep_work:
        # Staged corpora and per-class model sets are inputs, not results. Only
        # this run's classes are removed: retrain_per_lab stages its own corpora
        # under the same directory.
        for cls in class_names:
            _, archive, _ = CLASSES[cls]
            shutil.rmtree(WORK / "corpora" / f"{cls}_{archive}", ignore_errors=True)
            shutil.rmtree(WORK / "models" / f"{cls}_{archive}", ignore_errors=True)
    print(f"\nwrote {display_path(result_path)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=("fetch", "run"))
    ap.add_argument("--classes", default=",".join(CLASSES),
                    help="comma-separated subset of: " + ", ".join(CLASSES))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="run: split seed, and the frozen model set compared against")
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help="run: held-out fraction of each class's own cells")
    ap.add_argument("--out", type=Path, default=None,
                    help="run: default evaluation/results/generalization/new_class.json")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="run: where the class models' Stage 2 is fitted (see train_pipeline)")
    ap.add_argument("--keep-work", action="store_true",
                    help="run: keep the staged corpora and per-class models")
    ap.add_argument("--force", action="store_true",
                    help="fetch: overwrite a class manifest even when this fetch "
                         "found fewer usable cells than it records")
    args = ap.parse_args()
    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in names if c not in CLASSES]
    if unknown:
        raise SystemExit(f"unknown class(es): {unknown}")
    if args.step == "fetch":
        cmd_fetch(names, force=args.force)
    else:
        cmd_run(names, seed=args.seed, test_size=args.test_size, out=args.out,
                keep_work=args.keep_work, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
