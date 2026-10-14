#!/usr/bin/env python3
"""Corpus QC scan over the SWC corpus.

Reads every SWC file in a corpus's ``{pyramidal,interneuron}/`` subfolders and
records structural checks per file:

  - parseable          (>= 7 columns, valid numeric fields)
  - has_soma           (>= 1 node with type == 1)
  - single_root        (exactly one node with parent == -1)
  - no_orphan          (every non-root node's parent exists in the file)
  - has_neurites       (>= 1 node with type in {2, 3, 4})
  - radius_positive    (radii are non-NaN and >= 0)
  - coords_finite      (no NaN/Inf in x,y,z)
  - reasonable_size    (``pipeline.qc.MIN_NODES`` <= n_nodes <= ``MAX_NODES``)

Those columns are diagnostics, recorded so a failure can be read at a glance.
The ``qc_pass`` verdict itself comes from ``pipeline.qc.qc_violations``, the
same gate the inference command applies, so the corpus record and the labeling
gate cannot disagree about which cells are labelable.

Output (written to data/qc_report/, created if missing):
  corpus_qc.csv   one row per file (checks + qc_pass + reason + counts)
  corpus_qc.json  summary aggregates

This is a verification and provenance record over the corpus; the full list is
preserved so it is clear which files were dropped and why.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.features import parse_swc                               # noqa: E402
from pipeline.qc import MAX_NODES, MIN_NODES, qc_violations           # noqa: E402
from pipeline.swc_normalize import count_custom_types, normalize_swc  # noqa: E402

DATA = ROOT / "data"
OUT_CSV = ROOT / "data" / "qc_report" / "corpus_qc.csv"
OUT_JSON = OUT_CSV.with_suffix(".json")

# Non-standard types are not a rejection reason: parse_swc rewrites them to
# the branch's dominant standard type. The raw on-disk type values are
# recorded for transparency. The node-count bounds are shared with the
# inference command's gate (pipeline.qc).


def _is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def _rel(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` in POSIX form, or absolute if it lies outside."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def qc_one_file(path: Path, data_root: Path | None = None) -> dict:
    """Return a dict of QC fields for one SWC file.

    ``data_root`` is the corpus the file belongs to, used only to build the
    ``rel_path`` column; it defaults to this module's ``DATA``. Pass it rather
    than reassigning the module global when scanning a corpus elsewhere.
    """
    data_root = Path(data_root) if data_root is not None else DATA
    row = {
        "path": _rel(path, ROOT),
        "rel_path": _rel(path, data_root),
        "cell_type": path.parent.name,
        "source": path.name.split("__", 1)[0] if "__" in path.name else "",
        "n_bytes": path.stat().st_size if path.is_file() else 0,
        # checks (all default False; set True on success)
        "parseable": False,
        "has_soma": False,
        "single_root": False,
        "no_orphan": False,
        "has_neurites": False,
        "radius_positive": False,
        "coords_finite": False,
        "reasonable_size": False,
        # derived counts (AFTER normalization)
        "n_nodes": 0,
        "n_soma": 0,
        "n_axon": 0,
        "n_basal": 0,
        "n_apical": 0,
        "n_other": 0,
        "n_roots": 0,
        "n_orphan": 0,
        # informational: raw on-disk custom types (NOT a rejection reason)
        "n_custom_type_nodes_raw": 0,
        "custom_type_values_raw": "",
        # informational: raw soma node count BEFORE consolidation
        # (diagnoses how prevalent multi-point soma representations are)
        "n_soma_raw":            0,
        "n_soma_groups":         0,
        "n_soma_nodes_removed":  0,
        "n_nodes_raw":           0,
        # PASS flag
        "qc_pass": False,
        "qc_reason": "",
    }

    # Parse RAW so we can record pre-normalization diagnostics, then apply
    # the FULL normalization pipeline (custom types + multi-point soma
    # consolidation) — exactly what `parse_swc` does by default and what
    # training/inference will see. QC checks run on the normalized view.
    try:
        raw_nodes = parse_swc(path, normalize_types=False)
    except Exception as exc:
        row["qc_reason"] = f"parse_error: {exc}"
        return row
    raw_custom = count_custom_types(raw_nodes)
    row["n_nodes_raw"] = len(raw_nodes)
    row["n_soma_raw"]  = sum(1 for n in raw_nodes if n.type == 1)
    row["n_custom_type_nodes_raw"] = int(sum(raw_custom.values()))
    if raw_custom:
        row["custom_type_values_raw"] = ",".join(
            f"{t}:{c}" for t, c in sorted(raw_custom.items())
        )

    try:
        nodes, info = normalize_swc(raw_nodes)
    except Exception as exc:
        row["qc_reason"] = f"normalize_error: {exc}"
        return row
    soma_info = info.get("soma", {})
    row["n_soma_groups"]        = int(soma_info.get("group_count", 0))
    row["n_soma_nodes_removed"] = int(soma_info.get("removed_nodes", 0))
    row["parseable"] = True

    if not nodes:
        row["qc_reason"] = "empty_after_parse"
        return row

    row["n_nodes"] = len(nodes)
    row["reasonable_size"] = MIN_NODES <= len(nodes) <= MAX_NODES

    # Per-type counts (after normalization, types are in {0,1,2,3,4})
    type_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for n in nodes:
        if n.type in type_counts:
            type_counts[n.type] += 1
    row["n_soma"] = type_counts[1]
    row["n_axon"] = type_counts[2]
    row["n_basal"] = type_counts[3]
    row["n_apical"] = type_counts[4]
    row["n_other"] = sum(1 for n in nodes if n.type not in (1, 2, 3, 4))
    row["has_soma"] = row["n_soma"] >= 1
    row["has_neurites"] = (row["n_axon"] + row["n_basal"] + row["n_apical"]) >= 1

    # Roots / orphans
    by_id = {n.id: n for n in nodes}
    n_roots = sum(1 for n in nodes if n.parent == -1)
    n_orphan = sum(1 for n in nodes if n.parent != -1 and n.parent not in by_id)
    row["n_roots"] = n_roots
    row["n_orphan"] = n_orphan
    row["single_root"] = n_roots == 1
    row["no_orphan"] = n_orphan == 0

    # Coords / radii
    row["radius_positive"] = all(_is_finite(n.radius) and n.radius >= 0 for n in nodes)
    row["coords_finite"] = all(
        _is_finite(n.x) and _is_finite(n.y) and _is_finite(n.z) for n in nodes
    )

    # The verdict is the inference command's own gate, not a second opinion:
    # a cell this scan calls passing is exactly a cell the labeler will accept.
    # Non-standard type codes and multi-point somas are resolved by
    # normalization before it runs, so neither is a rejection reason.
    violations = qc_violations(nodes)
    row["qc_pass"] = not violations
    if violations:
        row["qc_reason"] = "fail: " + "; ".join(violations)
    return row


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA,
                    help="corpus with pyramidal/ and interneuron/ subfolders (default: data/)")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV,
                    help="where to write the per-file table (the summary goes beside it)")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    out_csv = args.out_csv.resolve()
    out_json = out_csv.with_suffix(".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for ct in ("pyramidal", "interneuron"):
        d = data_dir / ct
        if not d.is_dir():
            print(f"WARN: missing {d}")
            continue
        files.extend(sorted(d.glob("*.swc")))
    print(f"Found {len(files)} SWCs total in {data_dir}")

    t0 = time.perf_counter()
    rows: list[dict] = []
    by_status_pass = {"pyramidal": 0, "interneuron": 0}
    by_status_fail = {"pyramidal": 0, "interneuron": 0}
    fail_reasons: dict[str, int] = {}

    for i, fp in enumerate(files):
        row = qc_one_file(fp, data_root=data_dir)
        rows.append(row)
        ct = row["cell_type"]
        if row["qc_pass"]:
            by_status_pass[ct] = by_status_pass.get(ct, 0) + 1
        else:
            by_status_fail[ct] = by_status_fail.get(ct, 0) + 1
            fail_reasons[row["qc_reason"]] = fail_reasons.get(row["qc_reason"], 0) + 1
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(files)}  (elapsed {(time.perf_counter()-t0)/60:.1f} min)",
                  flush=True)

    # Write CSV
    if rows:
        keys = list(rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Aggregate summary
    by_source = {}
    for r in rows:
        src = r["source"] or "unknown"
        by_source.setdefault(src, {"pass": 0, "fail": 0})
        by_source[src]["pass" if r["qc_pass"] else "fail"] += 1

    summary = {
        "corpus": _rel(data_dir, ROOT),
        "n_total": len(rows),
        "n_pass": sum(1 for r in rows if r["qc_pass"]),
        "n_fail": sum(1 for r in rows if not r["qc_pass"]),
        "pass_by_cell_type": by_status_pass,
        "fail_by_cell_type": by_status_fail,
        "by_source": by_source,
        "fail_reason_counts": dict(sorted(fail_reasons.items(), key=lambda kv: -kv[1])),
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "csv": _rel(out_csv, ROOT),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"=== QC scan complete ({summary['elapsed_min']:.1f} min) ===")
    print(f"  Total       : {summary['n_total']}")
    print(f"  Passed      : {summary['n_pass']}  ({summary['n_pass'] / max(1, summary['n_total']) * 100:.1f}%)")
    print(f"  Failed      : {summary['n_fail']}")
    print(f"  Pass by cell-type    : {by_status_pass}")
    print(f"  Fail by cell-type    : {by_status_fail}")
    print(f"  By source            : {by_source}")
    print("  Top fail reasons:")
    for reason, n in list(summary["fail_reason_counts"].items())[:8]:
        print(f"    {n:>6}  {reason}")
    print()
    print(f"CSV  -> {out_csv}")
    print(f"JSON -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
