"""One-command trainer: build a full ``seed{S}/`` labeler model set from a data dir.

Chains every trainable stage of the auto-labeler on a directory of SWC data:

    Stage 1   cell-type classifier        -> cell_type_classifier.pkl
    Stage 2   subtree-owner classifier    -> branch_classifier.pkl
    Stage 3   branch correction           -> branch_correction.joblib
    Stage 4   single-seed quality flag    -> flag_model.joblib

The split this run used is recorded beside them in ``eval_split.json``, which the
generalization drivers read back.

Stage 0 (QC) and the Stage 3 topology rules are rule-based -- there is nothing to
train; the Stage 3 branch correction above is the learned half of that stage.
Stage 4 trains a single-seed quality flag on this seed's held-out cells, one per
cell type, using the same featurizer and target as the paper's multi-seed flag.
Seeing one seed's holdout rather than three costs the pyramidal flag very little:
it reaches kept-P10 0.919 at a 10% rejection rate against 0.922 for the paper's
leave-one-seed-out flag (``evaluation.stage4.eval_single_seed_flag``). Skip it with
``--skip flag`` for a labeler-only train. The multi-seed flag remains a separate
step (``evaluation.stage4``).

The data dir must contain ``pyramidal/`` and ``interneuron/`` subfolders of ``.swc``
files, the same layout as the bundled ``data/``. The train/test split is the
deterministic ``md5("{seed}:{filename}")`` hash bucket used throughout the pipeline;
only the train side is fitted, so models built here stay comparable to the shipped
evaluation. Omit ``--seed`` to draw a random one (it is printed and recorded in the
model dir's ``train_manifest.json``).

    python -m pipeline.train_pipeline --data-dir data --seed 123
    python -m pipeline.train_pipeline --data-dir /path/to/my/corpus        # random seed
    python -m pipeline.train_pipeline --data-dir data --seed 7 --skip stage1,stage2  # resume

Everything lands in ``pipeline/models/seed{S}/`` (override with ``--out-dir``). That
directory is exactly what ``pipeline.infer_pipeline --model-dir`` consumes at inference.
The released ``seed42``, ``seed123`` and ``seed789`` folders are rebuilt only from the
bundled ``data/``: a run on another corpus that would land in one of them stops before
writing anything, unless ``--overwrite`` is passed.
"""
from __future__ import annotations

import argparse

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.stage3.train_branch_correction import DEFAULT_SCOPE_UM  # noqa: E402
from pipeline.split import in_test as _in_test
from pipeline.released_models import refuse_replacing_released

CELL_TYPES = ("pyramidal", "interneuron")
STAGES = ("stage1", "stage2", "correction", "flag")


def _scan_split(data_dir: Path, seed: int, test_size: float):
    """Return (train_paths {ct: [Path]}, split_dict) for the hash-bucket split.

    ``split_dict`` is the ``eval_split.json`` record written next to the models
    (train/test file *names* per cell type), which the generalization drivers
    read back."""
    train_paths: dict[str, list[Path]] = {ct: [] for ct in CELL_TYPES}
    split = {"seed": seed, "test_size": test_size,
             "train_files": {ct: [] for ct in CELL_TYPES},
             "test_files": {ct: [] for ct in CELL_TYPES}}
    for ct in CELL_TYPES:
        for p in sorted((data_dir / ct).glob("*.swc")):
            if _in_test(p.name, seed, test_size):
                split["test_files"][ct].append(p.name)
            else:
                split["train_files"][ct].append(p.name)
                train_paths[ct].append(p)
    return train_paths, split


def _repo_relative(p: Path) -> str:
    """POSIX path relative to the repository when ``p`` lies inside it, else absolute."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data",
                    help="dir with pyramidal/ and interneuron/ subfolders of .swc (default: data/)")
    ap.add_argument("--seed", type=int, default=None,
                    help="split seed; omit for a random one (printed + recorded)")
    ap.add_argument("--test-size", type=float, default=0.20,
                    help="held-out fraction for the hash split (default 0.20)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the model set (default pipeline/models/seed{S})")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow a corpus other than data/ to replace a released seed42/123/789 folder")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for Stage 1 feature extraction (default: CPU-1, capped 8)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="where Stages 1 and 2 are fitted: cuda (default), as the released models were, "
                         "or cpu on a machine without an NVIDIA GPU. Every saved model predicts "
                         "on the CPU either way; a CPU fit is close to a GPU fit but not bit-identical.")
    ap.add_argument("--skip", default="",
                    help=f"comma list of stages to skip for resuming, from: {','.join(STAGES)}")
    ap.add_argument("--flag-limit", type=int, default=0,
                    help="Stage 4: train the flag on only N held-out cells (faster, weaker; default all).")
    ap.add_argument("--correction-scope-um", type=float, default=DEFAULT_SCOPE_UM,
                    help="Stage 3 correction: train on and override only branches whose midpoint lies within this "
                         f"path distance of the soma, in microns (default {DEFAULT_SCOPE_UM}; "
                         "pass a negative value for every branch, the pre-scope recipe).")
    args = ap.parse_args()

    import os
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    bad = skip - set(STAGES)
    if bad:
        raise SystemExit(f"--skip: unknown stage(s) {sorted(bad)}; valid: {','.join(STAGES)}")

    # --- resolve + validate the input corpus ---
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir not found: {data_dir}")
    for ct in CELL_TYPES:
        sub = data_dir / ct
        if not sub.is_dir():
            raise SystemExit(
                f"--data-dir must contain a '{ct}/' subfolder of .swc files; missing: {sub}")
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    out_dir = (args.out_dir or (ROOT / "pipeline" / "models" / f"seed{seed}")).resolve()
    refuse_replacing_released(out_dir, data_dir, args.overwrite)

    # The split is recorded next to the models, and the generalization drivers read
    # it back. Models trained on a different split would leave that record wrong, so
    # a disagreeing record stops the run here, before any stage writes a model.
    split_path = out_dir / "eval_split.json"
    if split_path.is_file():
        try:
            old = json.loads(split_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if old and (old.get("seed") != seed or old.get("test_size") != args.test_size):
            raise SystemExit(
                f"{split_path} records seed={old.get('seed')} "
                f"test_size={old.get('test_size')}, this run is seed={seed} "
                f"test_size={args.test_size}.\nThe models already in {out_dir} were "
                f"trained on the recorded split. Use a different --out-dir, or delete "
                f"that file if the whole directory is being rebuilt.")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths, split = _scan_split(data_dir, seed, args.test_size)
    counts = {ct: {"train": len(split["train_files"][ct]), "test": len(split["test_files"][ct])}
              for ct in CELL_TYPES}
    n_train = sum(c["train"] for c in counts.values())
    if n_train == 0:
        raise SystemExit(f"no training cells found under {data_dir} (both subfolders empty?)")

    print("=" * 78)
    print(f"train_pipeline  seed={seed}"
          f"{'  (random)' if args.seed is None else ''}  test_size={args.test_size}  "
          f"device={args.device}")
    print(f"  data : {data_dir}")
    print(f"  out  : {out_dir}")
    for ct in CELL_TYPES:
        print(f"  {ct:11s} train={counts[ct]['train']:5d}  test={counts[ct]['test']:5d}")
    if skip:
        print(f"  skipping: {', '.join(sorted(skip))}")
    print("=" * 78, flush=True)

    timings: dict[str, float] = {}

    # --- Stage 1: cell-type classifier ---
    if "stage1" not in skip:
        from pipeline.stage1.train import train_stage1  # noqa: PLC0415
        print("\n>>> Stage 1: cell-type classifier", flush=True)
        t0 = time.perf_counter()
        train_stage1(train_paths, out_dir / "cell_type_classifier.pkl", workers=workers,
                     device=args.device)
        timings["stage1"] = (time.perf_counter() - t0) / 60.0
        print(f"    -> cell_type_classifier.pkl ({timings['stage1']:.1f} min)", flush=True)

    # --- Stage 2: subtree-owner classifier ---
    if "stage2" not in skip:
        from pipeline.stage2.train_stage2 import train_stage2  # noqa: PLC0415
        print("\n>>> Stage 2: subtree-owner classifier", flush=True)
        t0 = time.perf_counter()
        train_stage2(train_paths, out_dir / "branch_classifier.pkl", device=args.device)
        timings["stage2"] = (time.perf_counter() - t0) / 60.0
        print(f"    -> branch_classifier.pkl ({timings['stage2']:.1f} min)", flush=True)

    # Record the split next to the models (checked against any earlier record above).
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    # --- Stage 3: branch correction (pyramidal) ---
    if "correction" not in skip:
        from pipeline.stage3.train_branch_correction import train_correction_for_seed  # noqa: PLC0415
        print("\n>>> Stage 3: branch correction", flush=True)
        t0 = time.perf_counter()
        scope_um = (None if args.correction_scope_um is None or args.correction_scope_um < 0
                    else float(args.correction_scope_um))
        n_br = train_correction_for_seed(
            data_dir / "pyramidal", seed, out_dir / "branch_correction.joblib",
            test_size=args.test_size, scope_um=scope_um)
        timings["correction"] = (time.perf_counter() - t0) / 60.0
        print(f"    -> branch_correction.joblib ({n_br} branches, "
              f"{timings['correction']:.1f} min)", flush=True)

    # --- Stage 4: single-seed quality flag (needs the labeler) ---
    if "flag" not in skip:
        # The flag learns from the labeler's mistakes, so it has to be fitted
        # against the labeler that ships. The correction is part of that labeler:
        # trained without it, the flag would be scoring a pipeline nobody runs.
        for req in ("cell_type_classifier.pkl", "branch_classifier.pkl",
                    "branch_correction.joblib"):
            if not (out_dir / req).is_file():
                raise SystemExit(
                    f"Stage 4 (flag) prerequisite missing: {out_dir / req}\n"
                    f"The whole labeler must be trained before the flag, the branch "
                    f"correction included. Use --skip flag as well if you meant to "
                    f"build a labeler without it.")
        from pipeline.stage4.single_seed_flag import train_flag_for_seed  # noqa: PLC0415
        print("\n>>> Stage 4: single-seed quality flag", flush=True)
        t0 = time.perf_counter()
        train_flag_for_seed(out_dir, data_dir, seed, test_size=args.test_size,
                            limit=args.flag_limit)
        timings["flag"] = (time.perf_counter() - t0) / 60.0
        print(f"    -> flag_model.joblib ({timings['flag']:.1f} min)", flush=True)

    # --- manifest + verification ---
    expected = {
        "stage1": "cell_type_classifier.pkl",
        "stage2": "branch_classifier.pkl",
        "correction": "branch_correction.joblib",
        "flag": "flag_model.joblib",
    }
    present = {st: (out_dir / fn).is_file() for st, fn in expected.items()}
    manifest = {
        "seed": seed,
        "seed_was_random": args.seed is None,
        "test_size": args.test_size,
        "device": args.device,
        "data_dir": _repo_relative(data_dir),
        "counts": counts,
        "skipped": sorted(skip),
        "timings_min": {k: round(v, 2) for k, v in timings.items()},
        "artifacts": {expected[st]: ok for st, ok in present.items()},
        "notes": ("Stage 0 (QC) and Stage 3 (topology refinement) are rule-based, "
                  "no training. Stage 4 is the single-seed quality flag, one per cell type "
                  "(skip with --skip flag); the paper's multi-seed flag is separate."),
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(timings.values())
    print("\n" + "=" * 78)
    print(f"DONE  seed={seed}  ({total:.1f} min total)")
    for st, fn in expected.items():
        mark = "ok " if present[st] else ("skip" if st in skip else "MISS")
        tm = f"{timings[st]:6.1f} min" if st in timings else "     --"
        print(f"  [{mark}] {fn:26s} {tm}")
    print(f"  model dir: {out_dir}")
    print(f"  manifest : {out_dir / 'train_manifest.json'}")
    print("  Stage 0/3 = rule-based (no training). Stage 4 = single-seed flag (--skip flag to omit).")
    print("=" * 78, flush=True)

    missing = [expected[st] for st in expected if not present[st] and st not in skip]
    if missing:
        raise SystemExit(f"incomplete: missing {missing}")
    print("TRAIN_PIPELINE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
