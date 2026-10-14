# External validation set

Cells from laboratories that contributed nothing to the training corpus, used to
measure how the shipped models transfer and how much a laboratory recovers by
retraining on its own data. Assembled by
`evaluation/generalization/fetch_external.py`,
`evaluation/generalization/new_class.py` and
`evaluation/generalization/more_cells.py`.

| folder | cells | contents |
| --- | --- | --- |
| `pyramidal/` | 1,500 | 10 NeuroMorpho archives, 150 each, mouse / rat / human / chimpanzee / monkey |
| `interneuron/` | 467 | 5 NeuroMorpho archives, mouse / rat / human |
| `granule/` | 150 | one archive (Rodrigues, rat), a cell class the pipeline never trained on |
| `ganglion/` | 150 | one archive (Rodger, rat), retinal ganglion cells, never trained on |
| `mitral_tufted/` | 63 | one archive (Urban, mouse), olfactory bulb mitral/tufted cells, never trained on |
| `more_cells/` | 561 | the rest of two archives above that pass the same selection: 366 Bianchi (chimpanzee) and 195 Rodger (retinal ganglion) cells |

Per-cell provenance is in the `.csv` beside each folder (`pyramidal.csv` for
`pyramidal/`, and so on): neuron name, archive, species, brain region, node
count, and which compartments the ground truth actually carries. `manifests/` holds the candidate lists the census
produced and the per-archive probe that selected from them.

## Selection

NeuroMorpho's `domain` field claims an axon for every candidate archive, but the
published CNG files often carry none, so selection is verified against file
content rather than metadata:

1. archive and neuron name absent from `data/metadata.csv`;
2. the file parses, has a soma, one root, no orphans, finite coordinates and
   radii — the same `pipeline._scan_corpus_qc` gate the training corpus passed;
3. basal and apical ground truth present for pyramidal, axon and dendrite for
   granule. Axon ground truth is recorded per cell in the metadata and scored
   only where it exists.

Purkinje cells were queried and rejected: the candidate archives store axon and
dendrites as separate files, or omit soma nodes, so no archive yields a
reconstruction the pipeline's input contract accepts.

## Terms

Every file is a NeuroMorpho.Org CNG-version reconstruction, the same source and
format as most of the training corpus. Attribution is per archive; see
`../LICENSE.md` for terms and the archive list. The `.swc` files are gitignored;
the manifests and metadata ship so the set can be rebuilt from the drivers.
