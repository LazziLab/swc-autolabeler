"""What the pipeline is measured against.

No released tool performs this labeling task, so the comparison methods are
constructed here. Each follows one recipe: morphometric features over a labeling
unit, a classifier predicts that unit's label, and the label goes to every node
of the unit. All are trained and scored on the same hash split as the pipeline
and receive the ground-truth cell type as a feature, so the comparison is
controlled.

    baselines           the four learned baselines (NeuroM-RF over branch
                        segments, and L-Measure-style RF, Sholl-RF and Sholl-MLP
                        over primary subtrees) and the two training-free floors:
                        all-basal, the output of a laboratory that draws no
                        distinction, and rules_only, the anatomical priors with
                        no learning at all
    eval_baselines      trains or loads each from a per-seed cache and scores it
                        in the pipeline's own metric pack, so the numbers drop
                        straight into Tables 2 to 4
    baseline_report     the metric pack itself: pooled and per-cell figures, and
                        the per-cell-type breakdown
    emissah_gcn         a faithful port of the one published system for an
                        adjacent task, retrained on our split because its
                        released weights were fitted on cells that overlap any
                        test set drawn from the same repository
    xgb_full            not a baseline but a bound: one tuned model over every
                        feature the pipeline computes, with no subtree stage, no
                        topology rules and no correction, which asks what the
                        staging itself is worth

The two floors matter as much as the learned methods: they fix what the metrics
award to no distinction at all and to anatomy alone, and any learned margin has
to clear both.
"""
