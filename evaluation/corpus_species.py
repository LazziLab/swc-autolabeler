"""Species and brain regions of the QC-pass corpus, by source and cell type.

The figures behind the paper's description of the corpus in Section 3. Species
comes from ``data/metadata.csv``. The Allen rows carry no species field, so an
Allen cell's species is read from its structure acronym: the human cells come
from named cortical gyri and lobes, the mouse cells from visual cortex areas.
An acronym outside both lists stops the script rather than being guessed.
NeuroMorpho.Org regions are the first, top-level entry of each cell's structure
list.

Usage:
    python -m evaluation.corpus_species

Writes evaluation/results/corpus_species.json and prints a summary.
"""
from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.paths import display_path                                # noqa: E402

#: Allen structure acronyms of human cortical regions (gyri and lobes).
ALLEN_HUMAN = {"MTG", "MFG", "SFG", "IFG", "ITG", "AnG", "PLP", "FroL", "TemL", "ParL", "OccL"}
#: Allen structure acronyms of mouse cortex start with the area's prefix.
ALLEN_MOUSE_PREFIXES = ("VIS",)
CELL_TYPES = ("pyramidal", "interneuron")


def _species(row: dict) -> str:
    if row["source"] == "allen":
        s = row["structure"].strip()
        if s in ALLEN_HUMAN:
            return "human"
        if s.startswith(ALLEN_MOUSE_PREFIXES):
            return "mouse"
        raise SystemExit(f"Allen structure {s!r} is in neither acronym list; add it to one")
    name = row["species"].strip().lower()
    return {"mus musculus": "mouse", "rattus norvegicus": "rat", "homo sapiens": "human"}.get(name, name)


def _region(row: dict) -> str:
    s = row["structure"].strip()
    if s.startswith("["):
        parts = ast.literal_eval(s)
        return parts[0].lower() if parts else ""
    return s


def _share(counter: Counter) -> dict:
    n = sum(counter.values())
    return {k: {"n": v, "share": v / n} for k, v in counter.most_common()}


def main() -> int:
    rows = list(csv.DictReader((ROOT / "data" / "metadata.csv").open(encoding="utf-8", newline="")))
    out: dict = {"n_cells": len(rows), "species_all": _share(Counter(_species(r) for r in rows)), "by_source": {}}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        block = {"n_cells": len(rs), "species": _share(Counter(_species(r) for r in rs)),
                 "archives": len({r["archive"] for r in rs if r["archive"]}), "by_cell_type": {}}
        if src == "neuromorpho":
            block["regions"] = _share(Counter(_region(r) for r in rs))
        for ct in CELL_TYPES:
            sub = [r for r in rs if r["cell_type"] == ct]
            if not sub:
                continue
            b = {"n_cells": len(sub), "species": _share(Counter(_species(r) for r in sub))}
            if src == "neuromorpho":
                b["regions"] = _share(Counter(_region(r) for r in sub))
            else:
                b["structures"] = _share(Counter(r["structure"] for r in sub))
            block["by_cell_type"][ct] = b
        out["by_source"][src] = block

    res = ROOT / "evaluation" / "results" / "corpus_species.json"
    res.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{out['n_cells']} QC-pass cells, {len(out['species_all'])} species")
    for src, b in out["by_source"].items():
        top = ", ".join(f"{k} {100 * v['share']:.1f}%" for k, v in list(b["species"].items())[:4])
        print(f"  {src:12s} {b['n_cells']:6d} cells, {len(b['species'])} species: {top}")
        for ct, c in b["by_cell_type"].items():
            sp = ", ".join(f"{k} {v['n']}" for k, v in list(c["species"].items())[:3])
            print(f"    {ct:12s} {c['n_cells']:6d}  {sp}")
    print(f"wrote {display_path(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
