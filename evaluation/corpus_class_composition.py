"""Class composition of the QC-pass pyramidal corpus.

Reports the share of each neurite class at two levels: individual nodes and
primary subtrees (each subtree assigned its majority node label). These are
the imbalance figures behind Stage 2's class-balanced sample weighting, cited
in the paper's evaluation-configuration section. Node shares are also given by
source, which is what Section 5.6's single-laboratory corpus is read against.

Usage:
    python evaluation/corpus_class_composition.py

Writes evaluation/results/corpus_class_composition.json and prints a summary.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.paths import display_path                                # noqa: E402

NAMES = {2: "axon", 3: "basal", 4: "apical"}


def main() -> None:
    with (ROOT / "data" / "metadata.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    pyr = [r for r in rows if r["cell_type"] == "pyramidal"]

    node_counts: Counter = Counter()
    by_source: dict[str, Counter] = {}
    subtree_counts: Counter = Counter()
    # Nodes carrying a type outside soma/axon/basal/apical. They are excluded
    # from the shares below, so their count is reported rather than left silent.
    other_nodes = 0
    malformed_lines = 0
    for r in pyr:
        fp = ROOT / "data" / r["rel_path"]
        types: dict[int, int] = {}
        parents: dict[int, int] = {}
        for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split()
            # A standard SWC row is id type x y z radius parent. Anything
            # shorter is not a node record; it is counted and skipped rather
            # than raising, so one bad line cannot lose a whole corpus pass.
            if len(f) < 7:
                malformed_lines += 1
                continue
            types[int(f[0])] = int(f[1])
            parents[int(f[0])] = int(f[6])

        children: dict[int, list[int]] = {}
        for nid, par in parents.items():
            children.setdefault(par, []).append(nid)
        # The soma is read from the type code here, while the labeler anchors it
        # at the largest-radius root and never reads type codes. On this corpus
        # the two agree on every file, so the subtree counts below describe the
        # same processes the pipeline sees.
        soma = {nid for nid, t in types.items() if t == 1}

        src = by_source.setdefault(r["source"], Counter())
        for t in types.values():
            if t in NAMES:
                node_counts[t] += 1
                src[t] += 1
            elif t != 1:
                other_nodes += 1

        for s in soma:
            for c in children.get(s, []):
                if c in soma:
                    continue
                stack, lab, seen = [c], Counter(), set()
                while stack:
                    n = stack.pop()
                    # A parent pointing at its own descendant would otherwise
                    # loop forever; Stage 0 rejects such files, but this script
                    # runs ahead of it, on the raw corpus.
                    if n in seen:
                        continue
                    seen.add(n)
                    if types[n] in NAMES:
                        lab[types[n]] += 1
                    stack.extend(children.get(n, []))
                if lab:
                    subtree_counts[lab.most_common(1)[0][0]] += 1

    tot_n = sum(node_counts.values())
    tot_s = sum(subtree_counts.values())
    if not tot_n or not tot_s:
        raise SystemExit(f"no neurite nodes or primary subtrees found under {ROOT / 'data'}")
    out = {
        "n_pyramidal_files": len(pyr),
        "other_type_nodes": other_nodes,
        "malformed_lines": malformed_lines,
        "neurite_nodes": {
            NAMES[t]: {"count": node_counts[t], "share": node_counts[t] / tot_n}
            for t in (2, 3, 4)
        },
        "primary_subtrees_majority_label": {
            NAMES[t]: {"count": subtree_counts[t], "share": subtree_counts[t] / tot_s}
            for t in (2, 3, 4)
        },
        "neurite_nodes_by_source": {
            name: {NAMES[t]: {"count": c[t], "share": c[t] / sum(c.values())} for t in (2, 3, 4)}
            for name, c in sorted(by_source.items()) if sum(c.values())
        },
    }
    res = ROOT / "evaluation" / "results" / "corpus_class_composition.json"
    res.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"pyramidal files: {len(pyr)}")
    print(f"nodes outside soma/axon/basal/apical (excluded): {other_nodes:,} "
          f"({100 * other_nodes / (tot_n + other_nodes):.3f}% of typed neurite-or-other nodes)")
    if malformed_lines:
        print(f"lines skipped as malformed: {malformed_lines:,}")
    print("neurite nodes:")
    for t in (2, 3, 4):
        print(f"  {NAMES[t]:7s} {node_counts[t]:>10,}  {100 * node_counts[t] / tot_n:5.1f}%")
    print("primary subtrees by majority label:")
    for t in (2, 3, 4):
        print(f"  {NAMES[t]:7s} {subtree_counts[t]:>10,}  {100 * subtree_counts[t] / tot_s:5.1f}%")
    print("neurite nodes by source:")
    for name, c in sorted(by_source.items()):
        n = sum(c.values())
        if n:
            print(f"  {name:12s} " + "  ".join(f"{NAMES[t]} {100 * c[t] / n:5.1f}%" for t in (2, 3, 4)))
    print(f"wrote {display_path(res)}")


if __name__ == "__main__":
    main()
