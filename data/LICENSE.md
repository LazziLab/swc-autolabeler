# Terms for the corpus

The software in this repository is MIT-licensed (see `../LICENSE`). The
reconstructions are not part of this repository and are not covered by that
license. They come from three sources with different terms, and they are
obtained in two ways: from the Zenodo deposit of this work, or, for the
Allen Institute's reconstructions, directly from their source.

| Source | Cells | Produced by | Terms | Where to get it |
| --- | --- | --- | --- | --- |
| NeuroMorpho.Org | 10,160 | many contributing laboratories | CC BY 4.0, NeuroMorpho.Org's license | Zenodo deposit |
| NeuroMorpho.Org, "Allen Cell Types" archive | 141 | Allen Institute for Brain Science | Allen Institute terms of use | links in `neuromorpho_allen_archive.csv` |
| Allen Cell Types Database | 203 | Allen Institute for Brain Science | Allen Institute terms of use | links in `allen_specimens.csv` |
| in-house `hpf_ca1` | 1,358 | this project | CC BY 4.0 | Zenodo deposit |

The external test sets under `external/` are further NeuroMorpho.Org
reconstructions, from archives the training corpus does not use. They are in
the same Zenodo deposit under the same CC BY 4.0 terms, and `external/*.csv`
lists them.

Per-cell provenance, including each file's original name and archive, is in
`metadata.csv` and `external/*.csv`, so any individual reconstruction can be
traced back to its source.

## Credit the sources

- **NeuroMorpho.Org files.** NeuroMorpho.Org's terms ask for three citations:
  the original publication of each reconstruction (listed per file in
  `neuromorpho_references.csv`), NeuroMorpho.Org itself (RRID:SCR_002145), and
  Tecuatl, Ljungquist and Ascoli (2024), *FASEB BioAdvances* 6(7):207–221,
  https://doi.org/10.1096/fba.2024-00048.
- **Allen Institute files.** The Allen Institute's terms of use
  (https://alleninstitute.org/terms-of-use/) allow redistribution for
  research and other noncommercial purposes only, which is why these 344 files
  are not in the CC BY 4.0 deposit. Cite them as the Allen Institute's citation
  policy asks.
- **In-house CA1 files.** Cite the paper (see `../CITATION.cff`).

## Note for anyone packaging this repository

The pipeline does not depend on the corpus being present: `pipeline/models/`
ships trained models, and every training entry point takes `--data-dir`, so the
code runs against a corpus you supply yourself.
