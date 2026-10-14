"""Stage 4 evaluation: what the quality flag is worth, and at what budget.

The flag itself lives in ``pipeline.stage4``. These drivers build its training
table, score it, and report the operating points a curator would actually use.
There is one flag per cell type. eval_flag and eval_single_seed_flag score each
on held-out cells of its own type; flag_deployment scores every cell with the
flag of the type Stage 1 assigns it, as a deployment does.

Building the table:

    score_flag_dataset        label one seed's held-out cells with the complete
                              pipeline and record, per cell, the deployment-safe
                              feature row and the ground-truth per-cell F1 the
                              flag regresses onto
    combine_flag_multiseed    join the three seeds' tables into the one pair of
                              CSVs every driver below reads

Scoring it:

    eval_flag                 the released multi-seed flag, fitted
                              leave-one-seed-out, with files shared across
                              seeds' test sets kept out of the training fold.
                              Table 8
    eval_single_seed_flag     the smaller flag ``train_pipeline`` fits from one
                              seed's holdout, scored by grouped cross-validation
                              without the cross-seed features it cannot have
    flag_oracle               the ceiling: rejecting exactly the truly worst
                              cells at the same budget
    flag_deployment           the flag in deployment order, with Stage 1 routing
                              a mixed batch instead of the cell type being given
    train_flag_deploy         fits the bundle that ships in the model directory

``_flag_metrics`` holds what they share: the ground-truth per-cell scoring, the
split-membership test, and the guard that refuses a model trained on a different
split than the one being scored.
"""
