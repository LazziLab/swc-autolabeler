"""Does the labeler work outside the corpus it was trained on?

Every in-corpus result scores held-out cells from laboratories that contributed
to training, so they share its species, protocols and labeling conventions: that
is interpolation, not transfer. These drivers assemble test sets from archives
the corpus never used and ask what the released models do there, and what a
laboratory could do instead.

Assembling the sets:

    fetch_external      query NeuroMorpho for cells of a type in archives the
                        corpus never used, probe the largest candidates to see
                        which actually carry the ground truth their metadata
                        claims, and download the chosen archives. Metadata is not
                        trusted: every file's own type codes and structural QC
                        decide

Scoring the frozen models:

    eval_external           the staged pipeline on the external pyramidal and
                            interneuron sets, with the cell type given and
                            predicted
    eval_unstaged_external  the single all-in-one model on the same cells, so the
                            two architectures are compared on one set
    flag_transfer           the quality flag out of distribution

What a laboratory can do instead:

    retrain_per_lab     fit the training recipe from scratch on one archive's own
                        cells and score it against the frozen models on that
                        archive's held-out third
    new_class           the same question for cell classes the pipeline never
                        trained on, mapped onto whichever of its two label
                        schemas their ground truth fits

Nothing here is inherited from the pooled models: a retrained model set starts
empty, and the frozen models are never modified.
"""
