"""Assemble the external validation set from NeuroMorpho.Org archives the
training corpus never used.

Selection is two-layer, because NeuroMorpho metadata does not reliably reflect
file content: the ``domain`` field may claim an axon that the CNG file does not
carry. The census filters on what metadata can answer (archive and neuron name
unused by the corpus, axon claimed); the probe and download steps then verify
every file's actual type codes and pass it through the same structural QC gate
as the training corpus (``pipeline._scan_corpus_qc``).

Two cell types are fetched the same way, selected with ``--cell-type``:
pyramidal (ten archives, basal + apical ground truth required, axon scored
where present) and interneuron (five archives, axon + dendrite ground truth
required, since axon versus dendrite is that type's whole task).

Steps:
    census    query the API for all cells of the type, write the candidate list
    probe     download 5 cells from each of the largest candidate archives and
              report per-archive axon/apical presence and QC pass rates
    download  fill the chosen archives up to a per-archive cap, keeping only
              files that pass QC and carry the type's required ground truth

Usage:
    python -m evaluation.generalization.fetch_external census   [--cell-type interneuron]
    python -m evaluation.generalization.fetch_external probe    [--cell-type interneuron]
    python -m evaluation.generalization.fetch_external download [--cell-type interneuron]

Output:
    data/external/manifests/candidates_<type>.csv                  (census)
    data/external/manifests/archive_probe[_interneuron].csv        (probe)
    data/external/<type>/*.swc + <type>.csv                        (download)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pipeline._scan_corpus_qc as _qc                                    # noqa: E402
from evaluation.paths import display_path                                 # noqa: E402

EXTERNAL = ROOT / "data" / "external"
MANIFESTS = EXTERNAL / "manifests"
API = "https://neuromorpho.org/api"

# Archives chosen from the probe: every archive whose sampled cells pass QC and
# carry apical + basal ground truth, capped so no laboratory dominates, spanning
# mouse, rat, human, chimpanzee, and monkey.
CHOSEN = ("vdHeuvel", "Witter", "BICCN-MOp-miniatlas-anatomy", "Zeng", "Lambe",
          "Lewis", "Hamad", "Bianchi", "Kougias_Juraska", "Zhang_Jiang")
# Interneuron archives chosen the same way (probe: axon in files, QC pass),
# spanning mouse, rat and human.
INTERNEURON_CHOSEN = ("Sjostrom", "Scanziani", "Jing_Jiang", "Wang_Ye",
                      "Helmstaedter")
CAP = 150
N_PROBE_ARCHIVES, N_PROBE_CELLS = 40, 5
CANDIDATE_FIELDS = ["neuron_name", "archive", "species", "brain_region",
                    "cell_type", "domain", "integrity"]
PROBE_FIELDS = ["archive", "n_cells", "probed", "axon", "apical", "qc_pass", "species"]
META_FIELDS = ["file_name", "cell_type", "source", "neuron_name", "archive",
               "species", "brain_region", "nm_cell_type", "domain", "integrity",
               "n_nodes", "has_axon_gt", "has_apical_gt"]


def _fetch_json(url: str) -> dict:
    """One API page as parsed JSON, retrying only what a retry can fix.

    A timeout or a transient server error is worth retrying; a 4xx means the
    request itself is wrong, and retrying it four times only delays the error.
    """
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                raise SystemExit(f"NeuroMorpho API returned HTTP {e.code} for {url}")
            if attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            if attempt == 3:
                raise
        time.sleep(2 ** (attempt + 1))


def download_cng(archive: str, neuron_name: str, dest: Path) -> bool:
    url = (f"https://neuromorpho.org/dableFiles/{archive.lower().replace(' ', '%20')}"
           f"/CNG%20version/{neuron_name.replace(' ', '%20')}.CNG.swc")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as resp:
                data = resp.read()
            if b"<html" in data[:200].lower():
                return False
            dest.write_bytes(data)
            return True
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return False


def type_counts(path: Path) -> Counter:
    c: Counter = Counter()
    for line in path.open(encoding="utf-8", errors="ignore"):
        if line.strip() and line[0] != "#":
            f = line.split()
            if len(f) >= 7:
                try:
                    c[int(f[1])] += 1
                except ValueError:
                    pass
    return c


def corpus_identity() -> tuple[set, set]:
    """Archives and neuron names already present in the training corpus."""
    archives, names = set(), set()
    with (ROOT / "data" / "metadata.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["source"] == "neuromorpho":
                archives.add(r["archive"].strip().lower())
            if r["neuron_name"]:
                names.add(r["neuron_name"].strip().lower())
    return archives, names


def cmd_census(cell_type: str) -> None:
    used_archives, used_names = corpus_identity()
    rows, page = [], 0
    while True:
        d = _fetch_json(f"{API}/neuron/select?q=cell_type:{cell_type}&size=500&page={page}")
        ns = d.get("_embedded", {}).get("neuronResources", [])
        if not ns:
            break
        for n in ns:
            rows.append({"neuron_name": n.get("neuron_name", ""),
                         "archive": n.get("archive", ""),
                         "species": n.get("species", ""),
                         "brain_region": ";".join(n.get("brain_region") or []),
                         "cell_type": ";".join(n.get("cell_type") or []),
                         "domain": n.get("domain", ""),
                         "integrity": n.get("physical_Integrity", "")})
        total = d.get("page", {}).get("totalPages", 0)
        page += 1
        if page >= total:
            break
        time.sleep(0.35)
    cand = [r for r in rows
            if r["archive"].strip().lower() not in used_archives
            and r["neuron_name"].strip().lower() not in used_names
            and "axon" in r["domain"].lower()]
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out = MANIFESTS / f"candidates_{cell_type}.csv"
    # Fixed field names: the header came from the first row, so an empty result
    # raised IndexError rather than saying the query found nothing.
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(cand)
    n_arch = len({r["archive"] for r in cand})
    print(f"{len(rows)} {cell_type} cells on NeuroMorpho; "
          f"{len(cand)} candidates in {n_arch} unused archives -> {display_path(out)}")


def cmd_probe(cell_type: str, keep_probe: bool = False) -> None:
    rows = list(csv.DictReader((MANIFESTS / f"candidates_{cell_type}.csv")
                               .open(encoding="utf-8", newline="")))
    by_arch = defaultdict(list)
    for r in rows:
        by_arch[r["archive"]].append(r)
    probe_dir = EXTERNAL / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    out = []
    print(f"{'archive':30s} {'n':>5s} {'dl':>3s} {'axon':>4s} {'apic':>4s} {'qc':>3s}  species")
    for arch, cells in sorted(by_arch.items(), key=lambda kv: -len(kv[1]))[:N_PROBE_ARCHIVES]:
        sample = random.Random(123).sample(cells, min(N_PROBE_CELLS, len(cells)))
        got = ax = ap = qc = 0
        for r in sample:
            dest = probe_dir / (f"{arch.replace(' ', '_')}__"
                                f"{r['neuron_name'].replace(' ', '_')}.swc")
            if not dest.exists() and not download_cng(arch, r["neuron_name"], dest):
                continue
            got += 1
            tc = type_counts(dest)
            ax += bool(tc.get(2))
            ap += bool(tc.get(4))
            qc += bool(_qc.qc_one_file(dest, data_root=EXTERNAL).get("qc_pass"))
        species = Counter(r["species"] for r in cells).most_common(1)[0][0]
        print(f"{arch:30s} {len(cells):5d} {got:3d} {ax:4d} {ap:4d} {qc:3d}  {species}",
              flush=True)
        out.append({"archive": arch, "n_cells": len(cells), "probed": got,
                    "axon": ax, "apical": ap, "qc_pass": qc, "species": species})
    probe_name = ("archive_probe.csv" if cell_type == "pyramidal"
                  else f"archive_probe_{cell_type}.csv")
    with (MANIFESTS / probe_name).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PROBE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    if not keep_probe:
        # The probe downloads are a sample used to choose archives, not data the
        # release needs; they were left behind under data/external.
        shutil.rmtree(probe_dir, ignore_errors=True)
    print(f"wrote {display_path(MANIFESTS / probe_name)}", flush=True)


def cmd_download(cell_type: str, cap: int = CAP, force: bool = False) -> None:
    with (MANIFESTS / f"candidates_{cell_type}.csv").open(encoding="utf-8",
                                                          newline="") as fh:
        rows = list(csv.DictReader(fh))
    swc_dir = EXTERNAL / cell_type
    swc_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    chosen = CHOSEN if cell_type == "pyramidal" else INTERNEURON_CHOSEN
    for arch in chosen:
        pool = [r for r in rows if r["archive"] == arch]
        random.Random(123).shuffle(pool)
        kept = 0
        for r in pool:
            if kept >= cap:
                break
            fn = (f"{arch.replace(' ', '_')}__"
                  f"{r['neuron_name'].replace(' ', '_').replace('/', '_')}.swc")
            dest = swc_dir / fn
            if not dest.exists() and not download_cng(arch, r["neuron_name"], dest):
                continue
            tc = type_counts(dest)
            # Pyramidal cells require basal + apical ground truth, with axon
            # recorded per cell and scored where present; interneurons require
            # axon + dendrite ground truth, the full two-class problem.
            if cell_type == "pyramidal":
                ok = bool(tc.get(3)) and bool(tc.get(4))
            else:
                ok = bool(tc.get(2)) and bool(tc.get(3))
            if not _qc.qc_one_file(dest, data_root=EXTERNAL).get("qc_pass") or not ok:
                dest.unlink(missing_ok=True)
                continue
            kept += 1
            meta.append({"file_name": fn, "cell_type": cell_type,
                         "source": "neuromorpho", "neuron_name": r["neuron_name"],
                         "archive": arch, "species": r["species"],
                         "brain_region": r["brain_region"],
                         "nm_cell_type": r["cell_type"], "domain": r["domain"],
                         "integrity": r["integrity"], "n_nodes": sum(tc.values()),
                         "has_axon_gt": int(bool(tc.get(2))),
                         "has_apical_gt": int(bool(tc.get(4)))})
        print(f"{arch:30s} kept {kept}", flush=True)
    # The repository moves, so a re-run can return fewer or different cells than
    # the corpus every transfer number in the paper rests on. A smaller result is
    # refused rather than written over the published manifest.
    meta_path = EXTERNAL / f"{cell_type}.csv"
    if meta_path.is_file() and not force:
        with meta_path.open(encoding="utf-8", newline="") as fh:
            before = list(csv.DictReader(fh))
        if len(meta) < len(before):
            raise SystemExit(
                f"this download kept {len(meta)} cells where {meta_path.name} "
                f"records {len(before)}. The published external set would shrink, "
                f"and the files already on disk are not removed. Pass --force to "
                f"overwrite the manifest.")
    with meta_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=META_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(meta)
    n_axon = sum(m["has_axon_gt"] for m in meta)
    # The selection is part of the corpus's identity, so it ships beside it: a
    # manifest alone could not say which cap or archive list produced it.
    (MANIFESTS / f"selection_{cell_type}.json").write_text(json.dumps(
        {"cell_type": cell_type, "cap_per_archive": cap, "archives": list(chosen),
         "n_cells": len(meta), "n_with_axon_gt": n_axon,
         "species": dict(Counter(m["species"] for m in meta))}, indent=2),
        encoding="utf-8")
    print(f"\n{len(meta)} cells, {len(chosen)} archives, {n_axon} with axon GT")
    print("species:", dict(Counter(m["species"] for m in meta)))
    print(f"wrote {display_path(meta_path)} and "
          f"{display_path(MANIFESTS / f'selection_{cell_type}.json')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=("census", "probe", "download"))
    ap.add_argument("--cell-type", default="pyramidal",
                    choices=("pyramidal", "interneuron"))
    ap.add_argument("--cap", type=int, default=CAP,
                    help=f"download: cells kept per archive (default {CAP})")
    ap.add_argument("--force", action="store_true",
                    help="download: overwrite the manifest even when this run "
                         "kept fewer cells than it records")
    ap.add_argument("--keep-probe", action="store_true",
                    help="probe: keep the sampled downloads under data/external/probe")
    args = ap.parse_args()
    if args.step == "census":
        cmd_census(args.cell_type)
    elif args.step == "probe":
        cmd_probe(args.cell_type, keep_probe=args.keep_probe)
    else:
        cmd_download(args.cell_type, cap=args.cap, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
