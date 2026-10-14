# Autolabeler

Labels every node of an SWC neuron reconstruction as soma, axon, basal dendrite
or apical dendrite, and flags the cells most likely to be mislabelled.

Five stages: quality control, cell type, subtree classification, topology rules
with a learned per-branch correction, and a quality flag for each cell type.

## Install

Python 3.10.

```bash
pip install -r requirements.txt
```

Run every command below from the repository root. They are written for bash; on
Windows, use Git Bash or WSL.

Labelling runs on the CPU alone and needs no GPU. Training fits Stages 1 and 2
on an NVIDIA GPU by default, which is how the released models were fitted;
`--device cpu` on `pipeline.train_pipeline` fits them on the CPU instead, for a
machine without one. A CPU fit is close to a GPU fit but not bit-identical,
so rebuilding the released models exactly needs the GPU. The other stages
always train on the CPU.

## Data

The reconstructions are not in this repository. The corpus and the external
test sets are deposited at Zenodo under CC BY 4.0 (the record is linked here once
it is published); unpack the deposit so the files sit at `data/pyramidal/*.swc`,
`data/interneuron/*.swc` and `data/external/<set>/*.swc`. The 344 reconstructions
produced by the Allen Institute are not redistributed: download each from the
link in `data/allen_specimens.csv` (203 files) or
`data/neuromorpho_allen_archive.csv` (141 files) and save it under its
`file_name` in the folder of its `cell_type`. `data/metadata.csv` identifies every
file by original name and source archive, and `data/LICENSE.md` gives each
source's terms and the citations it asks for.

**Coordinates and radii must be in micrometres.** Nothing detects otherwise: a
file in nanometres is labelled without complaint and the output looks ordinary.

## Label a cell

```bash
python -m pipeline.infer_pipeline cell.swc -o labelled.swc
```

The file need not carry labels. With no node typed soma, the root is taken as the
soma; with no neurite labels, every other node is an unlabeled neurite. A file
that is not one connected cell with a node beyond the soma, has a negative or
non-finite radius or a non-finite coordinate, or falls outside 10 to 200,000
nodes is refused; `--force` labels it anyway. These are the same checks
that chose the training corpus, where a file must also carry soma and neurite
labels to serve as ground truth.

When the cell type is known, give it and Stage 1 is skipped. The cell type sets
the label set: pyramidal is axon, basal and apical, interneuron is axon and
dendrite. The quality flag, trained on Stage 1's predictions, is then not scored.

```bash
python -m pipeline.infer_pipeline cell.swc -o labelled.swc --cell-type pyramidal
```

A cell of another type can take the label set its compartments fit, but the
shipped models are trained on pyramidal cells and interneurons only; other
classes need the labeling stages retrained on their own cells, as
`evaluation.generalization.new_class run` does.

The unstaged model, one model per cell type over all the features, labels the
same way but carries no quality flag:

```bash
python -m pipeline.infer_unstaged cell.swc -o labelled.swc
```

## Train

Roughly 45 minutes per seed. Run seed 123 first if you want the training-cost
table, and copy its manifest, which is where that table comes from. On a
machine without an NVIDIA GPU add `--device cpu`; training takes about as long.

```bash
python -m pipeline.train_pipeline --data-dir data --seed 123
```

```bash
cp pipeline/models/seed123/train_manifest.json evaluation/results/train_timing_full.json
```

Repeat for seeds 42 and 789, then build the all-in-one model for each seed:

```bash
python -m pipeline.train_unstaged --data-dir data --seed 123
```

Each seed directory should end up with `cell_type_classifier.pkl`,
`branch_classifier.pkl`, `branch_correction.joblib`, `flag_model.joblib` and
`unstaged_model.joblib`. A missing `branch_correction.joblib` means Stage 3 did
not fit; stop rather than continue.

To train on your own corpus, point `--data-dir` at it and give a new `--out-dir`,
which `infer_pipeline --model-dir` then reads. The released `seed42`, `seed123`
and `seed789` folders are rebuilt only from `data/`: a run on another corpus that
would land in one of them stops before writing anything unless `--overwrite` is
given.

## Evaluate

```bash
for s in 42 123 789; do python -m evaluation.eval_stage1 --seed $s; done
```

```bash
for s in 42 123 789; do python -m evaluation.eval_pipeline --seed $s; done
```

```bash
python -m report.aggregate_pipeline
```

Results land in `evaluation/results/`; `report/` turns them into tables.

## Evaluate the quality flag

About 35 minutes, almost all of it in the first step, which replays the whole
pipeline over each seed's held-out cells to build the flag's training rows.

```bash
for s in 42 123 789; do python -m evaluation.stage4.score_flag_dataset --seed $s; done
```

```bash
python -m evaluation.stage4.combine_flag_multiseed
```

The combine step checks that no cell reaches a flag model that already saw it;
it prints the check and refuses to write if it fails.

```bash
python -m evaluation.stage4.eval_flag
python -m evaluation.stage4.flag_oracle
python -m evaluation.stage4.train_flag_deploy
python -m evaluation.stage4.eval_single_seed_flag
python -m evaluation.analyze_failure_modes
python -m evaluation.stage4.flag_deployment
```

There is one flag per cell type, pyramidal and interneuron. `eval_flag` and
`eval_single_seed_flag` score each on held-out cells of its own type.
`train_flag_deploy` writes the multi-seed flag bundles, one per cell type, trained
on all three seeds' held-out cells, to `pipeline/models/flag_model/`; the rest only
measure. The labeling command scores a cell with the single-seed flag of the type
Stage 1 assigns it, from its model folder, which `train_pipeline` fits on that
seed's held-out cells. `flag_deployment` is the mixed-cell case, where Stage 1
decides which flag scores each cell. For the feature-set comparison:

```bash
for f in confidence_only mean_confidence; do python -m evaluation.stage4.eval_flag --feature-set $f; done
```

```bash
python -m report.flag_ablation_table
```

## Compare against the baselines

About 15 hours, and it writes roughly 25 GB of fitted models under
`evaluation/results/baselines/`. The first command is most of that time; the GCN
needs a GPU.

```bash
for s in 42 123 789; do python -m evaluation.baselines.eval_baselines --seed $s; done
```

```bash
python -m evaluation.baselines.emissah_gcn --seeds 42,123,789
```

```bash
python -m evaluation.baselines.xgb_full --seeds 42,123,789
```

NPIN (Su et al. 2021), the one released method that labels axon and dendrite
from the skeleton alone, is retrained on the same splits. Its code states no
licence, so it is not included here: clone it next to this repository. It also
imports graphviz and tqdm, and seaborn older than 0.14, whose `distplot` it uses
to set a length threshold. About 40 minutes on 26 cores, most of it NPIN's own
feature code.

```bash
git clone https://gitlab.com/czsu32/npin.git ../npin
git -C ../npin checkout 8d632fe
```

```bash
pip install "seaborn<0.14" graphviz tqdm
```

```bash
python -m evaluation.baselines.npin --npin-dir ../npin --seeds 42,123,789
```

The stage ablation runs next because the paired test reads the per-cell scores
it writes:

```bash
python -m evaluation.eval_ablation --seeds 42,123,789
```

```bash
python -m evaluation.paired_stats
```

```bash
python -m report.aggregate_baselines
```

## Measure the pipeline's own parts

About 3 hours. Each command stands alone except the last two, which are the
slow ones.

```bash
for s in 42 123 789; do python -m evaluation.eval_stage2 --seed $s; done
```

```bash
for s in 42 123 789; do python -m evaluation.eval_stage3_rules --seed $s; done
```

```bash
python -m report.stage3_rules_table
```

```bash
python -m evaluation.factorial_ablation --seeds 42,123,789
```

`factorial_ablation` checks itself against `ablation_ladder.json` and refuses to
agree with a ladder it cannot reproduce, so run the baselines section first or
pass `--no-ladder-check`.

```bash
for s in 42 123 789; do python -m evaluation.absent_class_fp --seed $s; done
```

```bash
python -m evaluation.correction_scope_ladder --seed 42,123,789
```

```bash
python -m evaluation.calibration
```

The threshold sweep caches one record per held-out cell, about 250 MB across the
three seeds, and reuses it with `--from-cache`:

```bash
python -m evaluation.threshold_sensitivity
```

## Curator feedback loop

About 4 hours, almost all of it in the loop itself: three arms over eight rounds
per seed, each round retraining Stage 2 and the pyramidal flag and relabelling the test
set. The defaults are the published protocol, 5,000 cells with 1,200 seen at the
start and 200 reviewed per round, so no arguments are needed.

```bash
for s in 42 123 789; do python -m evaluation.active_learning.curator_loop --seed $s; done
```

Each seed also writes a curator checkpoint, a drop-in Stage 2 trained on the
reviewed cells. The factory checkpoint is left alone.

```bash
python -m evaluation.active_learning.flag_feedback
```

```bash
python -m report.curator_loop_table
```

```bash
python -m report.make_curator_figures
```

## Laboratories outside the corpus

About 20 minutes once the cells are on disk. Like the corpus, the external cells
are not in this repository: they are in the Zenodo deposit, and `data/external/`
holds their lists. The commands below download them again instead and need
network access. `fetch_external` rebuilds the published selection from the
committed candidate lists; `new_class fetch` queries NeuroMorpho.Org again for
the three cell classes; `more_cells fetch` downloads the rest of the Bianchi and
Rodger archives from the committed candidate lists. All three refuse to write a
smaller set over a published list.

```bash
python -m evaluation.generalization.fetch_external download
python -m evaluation.generalization.fetch_external download --cell-type interneuron
python -m evaluation.generalization.new_class fetch
python -m evaluation.generalization.more_cells fetch
```

```bash
python -m evaluation.generalization.eval_external --seed 123
python -m evaluation.generalization.eval_external --seed 123 --gt-ct
python -m evaluation.generalization.eval_external --seed 123 --cell-type interneuron
python -m evaluation.generalization.eval_external --seed 123 --cell-type interneuron --gt-ct
```

All four passes are needed: the tables compare deployment against a given cell
type, for both cell types.

```bash
python -m evaluation.generalization.flag_transfer --seed 123
```

```bash
python -m evaluation.generalization.retrain_per_lab
python -m evaluation.generalization.retrain_per_lab --device cpu --out evaluation/results/generalization/lab_retrain_cpu.json
```

The second run fits the same laboratory models on the CPU, for Supplementary Section S2.1's
comparison of CPU and GPU fits.

```bash
python -m evaluation.generalization.new_class run
```

A corpus with no axon cannot fit the branch-correction head. That laboratory is
retrained without the correction and the table carries a note; it is not an
error.

```bash
python -m evaluation.generalization.more_cells run
```

Bianchi and Rodger come from archives that hold more usable cells than the
external set samples. This retrains both with the same recipe on every usable
cell outside the same held-out third, for the last two rows of Table 12.

```bash
python -m evaluation.generalization.eval_unstaged_external --seed 123
```

```bash
python -m report.generalization_tables
```

## Corpus survey, morphometrics, runtime and the unstaged model

About 90 minutes. The morphometrics step is most of it, and for the four
baselines it reads the fitted models the baselines section saves under
`evaluation/results/baselines/seed123/`, so run that section first.

```bash
python -m pipeline._scan_corpus_qc
python -m evaluation.corpus_label_survey
python -m evaluation.corpus_class_composition
python -m evaluation.corpus_species
python -m evaluation.mixed_label_structure
```

The QC scan reports the corpus as released, which holds only the cells that
passed. The collected counts and pass rates in Table 1 describe files that were
dropped before assembly and are not redistributed, so they cannot be
regenerated here.

```bash
for m in ours neurom_rf sholl_mlp sholl_rf lmeasure_rf; do python -m evaluation.downstream_morphometrics --method $m --seed 123; done
```

```bash
python -m evaluation.morphometric_utility --seed 123
```

`profile_runtime` times each component on 125 cells spread over the size range, then the labeling
command end to end (read, check, label, write, flag) on 1,000 random corpus cells; the second pass
gives the table's two end-to-end rows and takes a few minutes.

```bash
python -m evaluation.profile_runtime --seed 123
python -m report.runtime_table
```

The training half of Supplementary Table S11 reads the manifest copied in the Train section. The
text also quotes the same run with every stage on the CPU:

```bash
python -m pipeline.train_pipeline --data-dir data --seed 123 --device cpu --out-dir timing_run_cpu
cp timing_run_cpu/train_manifest.json evaluation/results/train_timing_full_cpu.json
```

```bash
for s in 42 123 789; do python -m evaluation.eval_unstaged --seed $s; done
```

## One laboratory's own corpus

About 7 hours, almost all of it in the last step. The lab-only training reads a
corpus folder of its own holding the 1,358 CA1 cells:

```bash
mkdir -p data_lab_only/pyramidal data_lab_only/interneuron
cp data/pyramidal/hpf_ca1__*.swc data_lab_only/pyramidal/
```

```bash
for s in 42 123 789; do python -m evaluation.eval_pipeline --seed $s --source-prefix hpf_ca1__; done
```

```bash
for s in 42 123 789; do
  python -m pipeline.train_pipeline --data-dir data_lab_only --seed $s --out-dir pipeline/models/lab_seed$s --skip stage1,flag
  python -m evaluation.eval_pipeline --seed $s --source-prefix hpf_ca1__ --model-dir pipeline/models/lab_seed$s --out-dir evaluation/results/lab_retrain
done
```

The file-hash split depends on the file name, not the directory, so a lab-only
model trains on exactly the lab cells the full-corpus split holds in and is
scored on the ones it holds out.

```bash
for s in 42 123 789; do python -m evaluation.baselines.eval_baselines --seed $s --source-prefix hpf_ca1__; done
```

```bash
for s in 42 123 789; do python -m evaluation.baselines.eval_baselines --seed $s --source-prefix hpf_ca1__ --train-source-prefix hpf_ca1__; done
```

The last command fits the baselines on lab data alone, into their own cache
directory, so the cross-source models are left intact.

`evaluation.nm_repository_census` is not part of this sequence. It queries
NeuroMorpho.Org live and refuses to overwrite its stored result, because a
re-fetch cannot reproduce the September 2026 snapshot the paper cites.

## Which command makes which table

Each table in the paper and its Supplementary Information (S tables), the command
that writes its numbers, and the section above that runs it.

| Table | Written by | Section above |
|---|---|---|
| 1 | pass counts only: `pipeline._scan_corpus_qc`; the collected counts are not regenerated | Corpus survey |
| 2, 3, 4 | `report.aggregate_pipeline`, `report.aggregate_baselines` | Evaluate; Compare against the baselines |
| 5 | `evaluation.baselines.npin` | Compare against the baselines |
| 6 | `evaluation.morphometric_utility` | Corpus survey |
| 7 | `evaluation.eval_ablation` | Compare against the baselines |
| 8 | `evaluation.stage4.eval_flag`, `evaluation.stage4.flag_oracle` | Evaluate the quality flag |
| 9 | `evaluation.analyze_failure_modes` | Evaluate the quality flag |
| 10 | `report.curator_loop_table` | Curator feedback loop |
| 11, 12, S9 | `report.generalization_tables` | Laboratories outside the corpus |
| 13 | `evaluation.eval_unstaged`, `evaluation.generalization.eval_unstaged_external` | Corpus survey; Laboratories outside the corpus |
| S1 | `evaluation.threshold_sensitivity` | Measure the pipeline's own parts |
| S2 | `evaluation.calibration` | Measure the pipeline's own parts |
| S3, S7 | `evaluation.mixed_label_structure` | Corpus survey |
| S4 | `evaluation.factorial_ablation` | Measure the pipeline's own parts |
| S5 | `evaluation.stage4.flag_deployment` | Evaluate the quality flag |
| S6 | `report.flag_ablation_table` | Evaluate the quality flag |
| S8 | the commands of that section | One laboratory's own corpus |
| S10 | `evaluation.generalization.flag_transfer` | Laboratories outside the corpus |
| S11 | `report.runtime_table` | Corpus survey |
| S12, S14, S15 | not results: the trained models' settings, the QC checks and the baselines' definitions | |
| S13 | `evaluation.correction_scope_ladder` | Measure the pipeline's own parts |

Figure 3 is `report.make_curator_figures`: its panel a comes from the curator
loop, its panel b from `flag_feedback`.

## Layout

```
pipeline/     the labeler and its training
evaluation/   the drivers that measure it
report/       compiles results into tables
data/         corpus provenance
```

`LICENSE` covers this repository, `THIRD_PARTY_NOTICES.md` the code it ports
from other projects, `data/LICENSE.md` the corpus, `CITATION.cff` the citation.

The trained models in `pipeline/models/` were fitted in part on reconstructions from the
Allen Institute for Brain Science; use of these weights should respect the Allen Institute's
terms of use (https://alleninstitute.org/terms-of-use/).
