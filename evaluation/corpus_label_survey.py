"""How this corpus is labeled before any relabeling.

Section 1 argues that a corpus assembled from many laboratories inherits each
laboratory's labeling choices. That argument is made there from NeuroMorpho.Org's
own repository-wide metadata (see ``evaluation.nm_repository_census``). This
script is the same question asked of the corpus actually used here, released as
supporting evidence; no number it produces is cited in the paper.

Three censuses over the QC-pass pyramidal corpus, each read directly from the
SWC files and the per-cell provenance in ``data/metadata.csv``:

    labels      share of cells carrying an apical label, an axon label, both,
                or neither, over every pyramidal cell
    archives    per NeuroMorpho archive, whether it ever labels apical and
                ever labels axon, cross-tabulated; which compartments a
                laboratory distinguishes is a property of the archive. This
                census covers the NeuroMorpho cells only, so the in-house and
                Allen cells are outside it and the printed denominator says so
    soma        per archive, the median number of soma nodes per cell, which
                separates the three-point convention from traced contours

Usage:
    python -m evaluation.corpus_label_survey

Output (evaluation/results/):
    corpus_label_survey.json
    corpus_label_survey.txt
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.paths import display_path                                # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "evaluation" / "results"


#: Archives smaller than this are reported separately: the archive-level facts
#: below are "ever", so one cell can decide a whole archive's row.
SMALL_ARCHIVE = 5


def type_counts(path: Path) -> tuple[Counter, int]:
    """Type-code histogram for one file, with the count of lines it could not read."""
    c: Counter = Counter()
    skipped = 0
    for line in path.open(encoding="utf-8", errors="ignore"):
        if line.strip() and line[0] != "#":
            f = line.split()
            if len(f) < 7:
                skipped += 1
                continue
            try:
                c[int(f[1])] += 1
            except ValueError:
                skipped += 1
    return c, skipped


def main() -> int:
    meta = {}
    with (DATA / "metadata.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["cell_type"] == "pyramidal":
                meta[r["file_name"]] = (r["source"], (r["archive"] or "").strip())

    cells = []
    skipped_lines = 0
    for p in sorted((DATA / "pyramidal").glob("*.swc")):
        if p.name not in meta:
            raise SystemExit(f"{p.name} is in data/pyramidal but not in "
                             f"data/metadata.csv; the two are out of step")
        src, arch = meta[p.name]
        counts, skipped = type_counts(p)
        skipped_lines += skipped
        cells.append((src, arch, counts))
    n = len(cells)

    has_ap = sum(1 for _, _, c in cells if c.get(4))
    has_ax = sum(1 for _, _, c in cells if c.get(2))
    both = sum(1 for _, _, c in cells if c.get(4) and c.get(2))
    neither = sum(1 for _, _, c in cells if not c.get(4) and not c.get(2))

    by_arch: dict[str, list[Counter]] = defaultdict(list)
    for src, arch, c in cells:
        if src == "neuromorpho" and arch:
            by_arch[arch].append(c)
    n_arch_cells = sum(len(cs) for cs in by_arch.values())
    small = sum(1 for cs in by_arch.values() if len(cs) < SMALL_ARCHIVE)
    # An archive counts as labeling a compartment when any one of its cells does,
    # so an archive of one cell is decided by that cell. The small-archive count
    # above says how much of the cross-tab rests on such archives.
    quad = Counter()
    for arch, cs in by_arch.items():
        quad[(any(c.get(4) for c in cs), any(c.get(2) for c in cs))] += 1

    soma = Counter()
    for arch, cs in by_arch.items():
        med = statistics.median(c.get(1, 0) for c in cs)
        # A median of 0 means the archive's cells carry no soma-coded node at
        # all, which is a different fact from an unusual soma size; it would
        # otherwise be pooled into "other" with the medians of 4 to 9.
        soma["no_soma" if med == 0 else
             "three_point" if med == 3 else
             "contour" if med >= 10 else "other"] += 1

    out = {
        "n_pyramidal": n,
        "labels": {"apical": has_ap / n, "axon": has_ax / n,
                   "both": both / n, "neither": neither / n},
        "archives": {"n": len(by_arch),
                     "n_cells": n_arch_cells,
                     "n_cells_outside": n - n_arch_cells,
                     f"n_under_{SMALL_ARCHIVE}_cells": small,
                     "apical_no_axon": quad[(True, False)],
                     "axon_no_apical": quad[(False, True)],
                     "both": quad[(True, True)],
                     "neither": quad[(False, False)]},
        "soma_per_archive": dict(soma),
        "skipped_lines": skipped_lines,
    }

    L = [f"Corpus label survey: {n} QC-pass pyramidal cells",
         "",
         f"  apical label present   {has_ap / n:6.1%}",
         f"  axon label present     {has_ax / n:6.1%}",
         f"  both                   {both / n:6.1%}",
         f"  neither                {neither / n:6.1%}",
         "",
         f"NeuroMorpho archives: {len(by_arch)}  "
         f"({n_arch_cells} cells; {n - n_arch_cells} in-house and Allen cells "
         f"have no archive and are outside this block)",
         f"  label apical, never axon   {quad[(True, False)]:3d}",
         f"  label axon, never apical   {quad[(False, True)]:3d}",
         f"  label both                 {quad[(True, True)]:3d}",
         f"  label neither              {quad[(False, False)]:3d}",
         f"  (an archive counts as labeling a compartment when any one of its "
         f"cells does; {small} archives hold under {SMALL_ARCHIVE} cells)",
         "",
         "Soma representation (per-archive median type-1 node count)",
         f"  three-point (median 3)     {soma['three_point']:3d}",
         f"  contour (median >= 10)     {soma['contour']:3d}",
         f"  no soma (median 0)         {soma['no_soma']:3d}",
         f"  other (median 1-2, 4-9)    {soma['other']:3d}"]
    if skipped_lines:
        L.append(f"\n lines skipped as unreadable: {skipped_lines:,}")
    report = "\n".join(L)
    print(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "corpus_label_survey.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")
    (RESULTS / "corpus_label_survey.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nwrote {display_path(RESULTS)}/corpus_label_survey.{{json,txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
