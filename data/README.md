# Corpus

Neuron reconstructions (SWC format) that the auto-labeler is trained and
evaluated on. The SWC files are not in this repository; the Data section of
`../README.md` says where to get them, and `LICENSE.md` gives their terms. Loading is path-based: scripts read the `pyramidal/` and
`interneuron/` folders directly and use every `.swc` inside them, so the folder
contents are the single source of truth for the dataset. These 11,862
morphologies are the cells that passed quality control; lower-quality
reconstructions have already been removed, so nothing is filtered out at load
time.

## Contents

| Cell type   | Count      |
|-------------|------------|
| pyramidal   | 8,488      |
| interneuron | 3,374      |
| **total**   | **11,862** |

By source:

| Source           | Pyramidal | Interneuron | Total  |
|------------------|-----------|-------------|--------|
| NeuroMorpho.Org  | 7,053     | 3,248       | 10,301 |
| in-house hpf_ca1 | 1,358     | 0           | 1,358  |
| Allen Cell Types | 77        | 126         | 203    |

## Sources

- **NeuroMorpho.Org** (https://neuromorpho.org): pyramidal and interneuron
  reconstructions.
- **Allen Cell Types Database** (https://celltypes.brain-map.org): spiny cells
  mapped to pyramidal, aspiny and sparsely spiny mapped to interneuron.
- **In-house hpf_ca1**: hippocampal CA1 pyramidal reconstructions.

## Layout

```
pyramidal/      *.swc
interneuron/    *.swc
metadata.csv    one row per file (provenance + acquisition metadata)
neuromorpho_references.csv  original publication of each NeuroMorpho.Org file
allen_specimens.csv  specimen ID and direct download of each Allen Cell Types file
neuromorpho_allen_archive.csv  direct download of each file from NeuroMorpho.Org's Allen Cell Types archive
README.md
```

Filenames follow `<source>__<original_name>.swc` to preserve provenance.

## metadata.csv

One row per SWC file in this folder. Columns:

| Column        | Meaning                                              |
|---------------|------------------------------------------------------|
| cell_type     | pyramidal or interneuron                             |
| source        | allen, neuromorpho, or hpf_ca1                       |
| file_name     | SWC filename in this folder                          |
| rel_path      | path relative to `data/` (`<cell_type>/<file_name>`) |
| original_name | filename as distributed by the source                |
| specimen_id   | source specimen identifier, when available           |
| neuron_name   | source neuron name, when available                   |
| archive       | NeuroMorpho archive, when available                  |
| species       | species, when available                              |
| structure     | brain region, when available                         |
| layer         | cortical layer, when available                       |
| dendrite_type | spiny / aspiny / sparsely spiny, when available      |

## neuromorpho_references.csv

One row per NeuroMorpho.Org file: its `file_name` and `neuron_name` as in `metadata.csv`, its archive, and the PubMed ID and DOI of the publication NeuroMorpho.Org lists for it, so that each reconstruction can be credited to its source. Files of the NeuroMorpho.Org "Allen Cell Types" archive list no publication; they come from the Allen Cell Types Database.

## allen_specimens.csv

One row per Allen Cell Types Database file: its `file_name` as in `metadata.csv`, its cell type, its specimen ID, the direct download of its SWC from the Allen Cell Types Database, and its specimen page. These files are not redistributed with the corpus; each one downloads from these links. On 2026-10-14, 202 of the 203 downloads were identical to the corpus files; the `note` column marks the one the Allen API listed but did not serve.

## neuromorpho_allen_archive.csv

One row per corpus file from NeuroMorpho.Org's "Allen Cell Types" archive, which re-hosts Allen Institute reconstructions: its `file_name` as in `metadata.csv`, its cell type, its NeuroMorpho.Org `neuron_name`, the direct download of its SWC from NeuroMorpho.Org, and its NeuroMorpho.Org page. Like the files in `allen_specimens.csv`, these are not redistributed with the corpus; each one downloads from these links. On 2026-10-14, all 141 downloads were identical to the corpus files.
