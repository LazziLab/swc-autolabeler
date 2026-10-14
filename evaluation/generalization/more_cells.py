"""More of a corpus's own cells: two of the retrained corpora trained on every usable cell.

Two of the six corpora that ``retrain_per_lab`` and ``new_class`` retrain come from archives
that hold more usable cells than the 150 the external set samples: Bianchi (chimpanzee
pyramidal cells) and Rodger (rat retinal ganglion cells). ``fetch`` downloads the rest of each
archive, keeping only files that pass the same QC gate and carry the same ground truth as the
published selection, and drops any file whose geometry duplicates a published cell, so no
held-out cell can reach training through a second copy. ``run`` trains the labeling stages with
the published corpus model's recipe, on every usable cell outside that corpus's held-out third,
and scores it, and the frozen seed-123 model, on the same held-out third as the published row.

Usage:
    python -m evaluation.generalization.more_cells fetch
    python -m evaluation.generalization.more_cells run [--device cpu]

Output:
    data/external/more_cells/<corpus>/*.swc + <corpus>.csv     (fetch)
    evaluation/results/generalization/more_cells.json          (run)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pipeline._scan_corpus_qc as _qc                                    # noqa: E402
from pipeline.split import in_test                                        # noqa: E402
from evaluation.paths import display_path                                 # noqa: E402
from evaluation.generalization.fetch_external import (                    # noqa: E402
    download_cng, type_counts,
)
from evaluation.generalization import retrain_per_lab, new_class          # noqa: E402

EXTERNAL = ROOT / "data" / "external"
MORE = EXTERNAL / "more_cells"
RESULTS = ROOT / "evaluation" / "results" / "generalization"
WORK = RESULTS / "lab_retrain_work"
SEED = 123
#: The published corpus models' held-out fraction, so the held-out third is theirs.
TEST_SIZE = 1.0 / 3.0

# corpus -> archive, candidate list, published manifest and folder, the cell type the
# corpus is labeled as, the ground-truth codes every file must carry, the stages the
# published corpus model skipped, and that model's key in its result file
CORPORA = {
    "bianchi": {"archive": "Bianchi", "candidates": "candidates_pyramidal.csv",
                "manifest": "pyramidal.csv", "folder": "pyramidal", "schema": "pyramidal",
                "need": (3, 4), "skip": "stage1,flag",
                "published": ("lab_retrain.json", "Bianchi")},
    "rodger": {"archive": "Rodger", "candidates": "candidates_ganglion.csv",
               "manifest": "ganglion.csv", "folder": "ganglion", "schema": "interneuron",
               "need": (2, 3), "skip": "stage1,correction,flag",
               "published": ("new_class.json", "ganglion/Rodger")},
}
META_FIELDS = ["file_name", "neuron_name", "archive", "species", "brain_region",
               "n_nodes", "has_axon_gt", "has_apical_gt"]


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def geometry_key(path: Path) -> str:
    """Node count and coordinates rounded to 0.1 um, hashed: the same for two copies of
    one tracing, whatever their names."""
    pts = []
    for line in path.open(encoding="utf-8", errors="ignore"):
        f = line.split()
        if len(f) >= 7 and line[0] != "#":
            try:
                pts.append("%.1f,%.1f,%.1f" % (float(f[2]), float(f[3]), float(f[4])))
            except ValueError:
                pass
    return hashlib.md5((f"{len(pts)}|" + ";".join(sorted(pts))).encode()).hexdigest()


def published(cfg: dict) -> list[str]:
    return [r["file_name"] for r in _rows(EXTERNAL / cfg["manifest"])
            if r["archive"] == cfg["archive"]]


def cmd_fetch(names: list[str], force: bool = False) -> None:
    for name in names:
        cfg = CORPORA[name]
        cand = [r for r in _rows(EXTERNAL / "manifests" / cfg["candidates"])
                if r["archive"] == cfg["archive"]]
        pub_rows = [r for r in _rows(EXTERNAL / cfg["manifest"]) if r["archive"] == cfg["archive"]]
        pub_names = {r["neuron_name"] for r in pub_rows}
        pub_geometry = {geometry_key(EXTERNAL / cfg["folder"] / r["file_name"]) for r in pub_rows}
        dest_dir = MORE / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        meta, n_failed, n_unusable, n_duplicate = [], 0, 0, 0
        for r in cand:
            if r["neuron_name"] in pub_names:
                continue
            fn = (f"{cfg['archive']}__"
                  f"{r['neuron_name'].replace(' ', '_').replace('/', '_')}.swc")
            dest = dest_dir / fn
            if not dest.exists():
                ok = download_cng(cfg["archive"], r["neuron_name"], dest)
                time.sleep(0.2)
                if not ok:
                    n_failed += 1
                    continue
            tc = type_counts(dest)
            if (not all(tc.get(c) for c in cfg["need"])
                    or not _qc.qc_one_file(dest, data_root=MORE).get("qc_pass")):
                n_unusable += 1
                dest.unlink(missing_ok=True)
                continue
            if geometry_key(dest) in pub_geometry:
                n_duplicate += 1
                dest.unlink(missing_ok=True)
                continue
            meta.append({"file_name": fn, "neuron_name": r["neuron_name"],
                         "archive": cfg["archive"], "species": r.get("species", ""),
                         "brain_region": r.get("brain_region", ""),
                         "n_nodes": sum(tc.values()), "has_axon_gt": int(bool(tc.get(2))),
                         "has_apical_gt": int(bool(tc.get(4)))})
        # The repository moves, so a re-fetch can find fewer usable cells than the list the
        # paper's rows rest on. A smaller result is refused rather than written over it.
        meta_path = MORE / f"{name}.csv"
        if meta_path.is_file() and not force and len(meta) < len(_rows(meta_path)):
            raise SystemExit(f"{name}: this fetch kept {len(meta)} cells where {meta_path.name} "
                             f"records {len(_rows(meta_path))}. Pass --force to overwrite it.")
        with meta_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=META_FIELDS)
            w.writeheader()
            w.writerows(meta)
        print(f"{name}: {len(cand)} candidates, {len(pub_rows)} already published; kept "
              f"{len(meta)}, {n_unusable} failed QC or lacked the required labels, "
              f"{n_duplicate} duplicated a published cell, {n_failed} did not download",
              flush=True)


def cmd_run(names: list[str], device: str = "cuda", keep_work: bool = False) -> None:
    out_path = RESULTS / "more_cells.json"
    result = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    result.pop("_run", None)
    frozen_dir = ROOT / "pipeline" / "models" / f"seed{SEED}"
    for name in names:
        cfg = CORPORA[name]
        pub = published(cfg)
        test = [f for f in pub if in_test(f, SEED, TEST_SIZE)]
        train = [EXTERNAL / cfg["folder"] / f for f in pub if not in_test(f, SEED, TEST_SIZE)]
        extra = [MORE / name / r["file_name"] for r in _rows(MORE / f"{name}.csv")]
        res_file, res_key = cfg["published"]
        ref = json.loads((RESULTS / res_file).read_text(encoding="utf-8"))[res_key]
        if (len(test), len(train)) != (ref["n_test"], ref["n_train"]):
            raise SystemExit(f"{name}: the held-out third no longer matches {res_file}")
        # Stage only the training cells; with --test-size 0 the trainer holds none out,
        # and the held-out third stays exactly the published row's.
        corpus = WORK / "corpora" / f"more_{name}"
        shutil.rmtree(corpus, ignore_errors=True)
        (corpus / "pyramidal").mkdir(parents=True)
        (corpus / "interneuron").mkdir(parents=True)
        for p in train + extra:
            shutil.copy2(p, corpus / cfg["schema"] / p.name)
        mdir = WORK / "models" / f"more_{name}"
        shutil.rmtree(mdir, ignore_errors=True)
        print(f"\n=== {name}: training on {len(train) + len(extra)} cells "
              f"({len(train)} published, {len(extra)} more) ===", flush=True)
        t0 = time.perf_counter()
        run = subprocess.run(
            [sys.executable, "-m", "pipeline.train_pipeline",
             "--data-dir", str(corpus), "--seed", str(SEED), "--test-size", "0",
             "--out-dir", str(mdir), "--skip", cfg["skip"], "--device", device],
            cwd=str(ROOT), capture_output=True, text=True)
        train_s = time.perf_counter() - t0
        if run.returncode != 0 or not (mdir / "branch_classifier.pkl").is_file():
            print("TRAIN FAILED:\n" + "\n".join(run.stderr.splitlines()[-10:]))
            raise SystemExit(1)
        if cfg["schema"] == "pyramidal":
            own = retrain_per_lab.evaluate(test, mdir)
            frozen = retrain_per_lab.evaluate(test, frozen_dir)
        else:
            own = new_class.evaluate("ganglion", test, mdir)
            frozen = new_class.evaluate("ganglion", test, frozen_dir)
        result[name] = {"archive": cfg["archive"], "n_train": len(train) + len(extra),
                        "n_published_train": len(train), "n_more": len(extra),
                        "n_test": len(test), "train_s": round(train_s, 2),
                        "own": own, "frozen": frozen,
                        "published_own": {k: ref["own"][k] for k in ("pcf1_mean", "pcf1_p10")}}
        print(f"{name}  held-out {len(test)}")
        for k in ("pcf1_mean", "pcf1_p10"):
            print(f"  {k:10s} frozen {frozen[k]:.3f}   own: {ref['own'][k]:.3f} from "
                  f"{len(train)} cells -> {own[k]:.3f} from {len(train) + len(extra)}", flush=True)
        if not keep_work:
            shutil.rmtree(corpus, ignore_errors=True)
            shutil.rmtree(mdir, ignore_errors=True)
    result["_run"] = {"corpora": names, "seed": SEED, "test_size": TEST_SIZE, "device": device,
                      "frozen_models": display_path(frozen_dir)}
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {display_path(out_path)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=("fetch", "run"))
    ap.add_argument("--corpora", default=",".join(CORPORA),
                    help="comma-separated subset of: " + ", ".join(CORPORA))
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="run: where Stage 2 is fitted (see train_pipeline)")
    ap.add_argument("--force", action="store_true",
                    help="fetch: overwrite a list even when this fetch kept fewer cells than it records")
    ap.add_argument("--keep-work", action="store_true",
                    help="run: keep the staged corpora and models under lab_retrain_work/")
    args = ap.parse_args()
    names = [c.strip() for c in args.corpora.split(",") if c.strip()]
    unknown = [c for c in names if c not in CORPORA]
    if unknown:
        raise SystemExit(f"unknown corpus: {unknown}")
    if args.step == "fetch":
        cmd_fetch(names, force=args.force)
    else:
        cmd_run(names, device=args.device, keep_work=args.keep_work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
