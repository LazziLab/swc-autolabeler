"""The released model folders, and the guard that keeps a retrain on another corpus
from replacing them.

``pipeline/models/seed42``, ``seed123`` and ``seed789`` hold the models the paper
reports, trained on the bundled ``data/`` corpus. Retraining one of them from that
corpus rebuilds it; training on any other corpus would replace it without a trace,
so the trainers refuse that unless ``--overwrite`` is passed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLED_DATA = ROOT / "data"
RELEASED_SEEDS = (42, 123, 789)


def released_dirs() -> set[Path]:
    """The released model folders, resolved."""
    return {(ROOT / "pipeline" / "models" / f"seed{s}").resolve() for s in RELEASED_SEEDS}


def refuse_replacing_released(out_dir: Path, data_dir: Path, overwrite: bool = False,
                              out_option: str = "--out-dir") -> None:
    """Exit before models trained on ``data_dir`` are written into a released folder.

    ``data_dir`` is the bundled corpus or one of its cell-type subfolders when a
    released folder is being rebuilt; anything else stops the run with a message
    naming the caller's output option (``out_option``) and ``--overwrite``."""
    if overwrite or Path(out_dir).resolve() not in released_dirs():
        return
    bundled = BUNDLED_DATA.resolve()
    data_dir = Path(data_dir).resolve()
    if data_dir in {bundled, bundled / "pyramidal", bundled / "interneuron"}:
        return
    raise SystemExit(
        f"{Path(out_dir).resolve()} holds released models, trained on {bundled}.\n"
        f"Training on {data_dir} would replace them. Pass {out_option} to write the new "
        f"models elsewhere, or --overwrite to replace the released ones.")
