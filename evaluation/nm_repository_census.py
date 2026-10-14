"""Repository-scale label census: how NeuroMorpho.Org describes its own
holdings.

Pages the public API over every neuron and tallies two curated metadata
fields. ``domain`` states which structural domains a reconstruction contains
and distinguishes, so the share recording only undifferentiated processes or
neurites measures label coarseness across the whole repository, not just this
paper's corpus. ``physical_Integrity`` is the curators' completeness grade.

Two files are written, and the paper's Section 1 reads from both: the census
itself, and the per-archive labeling conventions behind the statement that of
the archives contributing at least ten reconstructions, some always type their
neurites, some never do, and some mix.

The query is ``archive:*``, so the totals count neurons that carry an archive
field; a reconstruction without one is outside this census.

The repository grows, so every count depends on the access date. The paper
cites the run of September 2026 (298,339 neurons), and the files shipped in
``evaluation/results/`` are that run: ``nm_repository_census.json`` holding
``total``, ``physical_integrity`` and ``domain``, the counts the paper uses, and
``nm_archive_conventions.json`` holding the per-archive conventions. A new run
also writes the keys derived from those counts. Because a
rerun would silently replace a dated snapshot with a different one, this script
refuses to overwrite either file unless ``--force`` is given.

Requires network access.

Usage:
    python -m evaluation.nm_repository_census
    python -m evaluation.nm_repository_census --max-pages 2 --out-dir /tmp/smoke

Output (evaluation/results/):
    nm_repository_census.json
    nm_repository_census.txt
    nm_archive_conventions.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.paths import display_path                                # noqa: E402

RESULTS = ROOT / "evaluation" / "results"
API = "https://neuromorpho.org/api"
#: Archives with fewer uploads than this are left out of the conventions census.
MIN_UPLOADS = 10


def fetch(url: str, tries: int = 4):
    """One API page as parsed JSON, retrying only what a retry can fix.

    A timeout or a transient server error is worth retrying; a 4xx means the
    request itself is wrong and retrying it four times only delays the error.
    """
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                raise SystemExit(f"NeuroMorpho API returned HTTP {e.code} for {url}")
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            if attempt == tries - 1:
                raise
        time.sleep(2 ** (attempt + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--max-pages", type=int, default=0,
                    help="stop after this many pages (0 = the whole repository); "
                         "a reduced run is tagged in its file name")
    ap.add_argument("--min-uploads", type=int, default=MIN_UPLOADS,
                    help=f"archives smaller than this are left out of the "
                         f"conventions census (default {MIN_UPLOADS})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing snapshot; without it a run that "
                         "would replace the dated files refuses")
    args = ap.parse_args()

    tag = f"_page{args.max_pages}" if args.max_pages else ""
    census_path = args.out_dir / f"nm_repository_census{tag}.json"
    report_path = args.out_dir / f"nm_repository_census{tag}.txt"
    conv_path = args.out_dir / f"nm_archive_conventions{tag}.json"
    existing = [p for p in (census_path, report_path, conv_path) if p.is_file()]
    if existing and not args.force:
        raise SystemExit(
            "this run would replace a dated snapshot:\n"
            + "".join(f"  {display_path(p)}\n" for p in existing)
            + "The repository grows, so the replacement would not hold the same\n"
              "numbers as the run the paper cites. Pass --force to overwrite, or\n"
              "--out-dir to write elsewhere.")

    integrity, domain = Counter(), Counter()
    per_archive: dict[str, Counter] = {}
    # The API reports totalPages afresh on every request, so a repository that
    # grows mid-run shifts the page boundaries. Neurons are keyed by id and
    # counted once, which makes a shifted page a no-op rather than a duplicate.
    seen: set[str] = set()
    duplicates = 0
    page = 0
    while True:
        d = fetch(f"{API}/neuron/select?q=archive:*&size=500&page={page}")
        neurons = d.get("_embedded", {}).get("neuronResources", [])
        if not neurons:
            break
        for n in neurons:
            key = str(n.get("neuron_id") or n.get("neuron_name") or "")
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            integrity[n.get("physical_Integrity") or "MISSING"] += 1
            dom = n.get("domain") or "MISSING"
            domain[dom] += 1
            arch = n.get("archive") or "?"
            bucket = per_archive.setdefault(arch, Counter())
            low = dom.lower()
            if "axon" in low or "dendrite" in low:
                bucket["typed"] += 1
            elif "process" in low or "neurite" in low:
                bucket["untyped"] += 1
        total_pages = d["page"]["totalPages"]
        page += 1
        if page % 100 == 0:
            print(f"  page {page}/{total_pages}", flush=True)
        if page >= total_pages or (args.max_pages and page >= args.max_pages):
            break
        time.sleep(0.15)

    grand = sum(domain.values())
    if not grand:
        raise SystemExit("the API returned no neurons")
    full = sum(n for v, n in domain.items()
               if "dendrites" in v.lower() and "axon" in v.lower()
               and "no axon" not in v.lower() and "no dendrites" not in v.lower())
    undiff = sum(n for v, n in domain.items()
                 if "process" in v.lower() or "neurite" in v.lower())
    big = {a: c for a, c in per_archive.items() if sum(c.values()) >= args.min_uploads}
    always = sum(1 for c in big.values() if c["typed"] and not c["untyped"])
    never = sum(1 for c in big.values() if c["untyped"] and not c["typed"])
    mixed = sum(1 for c in big.values() if c["typed"] and c["untyped"])

    L = [f"NeuroMorpho.Org repository census: {grand} neurons carrying an archive",
         "",
         f"  domain distinguishes dendrites, soma and axon  {full:7d}  {full / grand:6.1%}",
         f"  domain only undifferentiated processes/neurites{undiff:7d}  {undiff / grand:6.1%}",
         "",
         f"  archives with >= {args.min_uploads} uploads: {len(big)}",
         f"    always type neurites {always:4d}   never {never:4d}   mixed {mixed:4d}",
         "",
         "DOMAIN (top 10)"]
    for v, n in domain.most_common(10):
        L.append(f"  {v:48s} {n:7d}  {n / grand:6.1%}")
    L += ["", "PHYSICAL INTEGRITY (top 10)"]
    for v, n in integrity.most_common(10):
        L.append(f"  {v:48s} {n:7d}  {n / grand:6.1%}")
    if duplicates:
        L += ["", f"{duplicates} repeated neuron records were counted once "
                  f"(the repository shifted pages mid-run)"]

    report = "\n".join(L)
    print(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    census_path.write_text(json.dumps(
        {"total": grand, "domain_full_three": full, "domain_undifferentiated": undiff,
         "archives_min10": len(big), "archives_always_typed": always,
         "archives_never_typed": never, "archives_mixed": mixed,
         "duplicate_records": duplicates,
         "physical_integrity": dict(integrity), "domain": dict(domain)}, indent=1),
        encoding="utf-8")
    report_path.write_text(report + "\n", encoding="utf-8")
    # The per-archive detail behind the three archive counts above. Section 1
    # quotes those counts, so the evidence for them ships beside the census.
    conv_path.write_text(json.dumps(
        {"min_uploads": args.min_uploads, "n_archives": len(big),
         "always_diff": always, "never_diff": never, "mixed": mixed,
         "other": len(big) - always - never - mixed,
         "per_archive": {a: {("diff" if k == "typed" else "undiff"): v
                             for k, v in c.items()}
                         for a, c in sorted(big.items())}}, indent=1),
        encoding="utf-8")
    print(f"\nwrote {display_path(census_path)}, {display_path(report_path)} "
          f"and {display_path(conv_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
