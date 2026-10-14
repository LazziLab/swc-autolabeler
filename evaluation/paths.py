"""One way for an evaluation driver to name a path in its output.

Result files record which model or corpus a run used. Writing an absolute path
there would tie the record to one machine, so a path inside the repository is
reported relative to it, in POSIX form so the same run reads identically on
Windows and Linux. A path outside the repository has no relative form and is
reported as it is.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display_path(p: str | Path, base: Path = ROOT) -> str:
    """``p`` relative to ``base`` in POSIX form, or absolute if it lies outside."""
    p = Path(p).resolve()
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return p.as_posix()
