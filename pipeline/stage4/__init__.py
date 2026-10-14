"""Stage 4: the quality flag, a reviewer for the labeling rather than part of it.

Stages 0 to 3 label a cell; this stage scores the finished labeling and ranks a
batch of cells so the least trustworthy can be set aside for a human. Nothing
here changes a node's label.

The flag is a regressor onto per-cell neurite F1 (``flag_metrics``), built from
deployment-safe columns only: predicted labels, per-node confidences and raw
geometry, never ground truth. ``flag_features`` assembles the confidence half of
the row and ``flag_geom_features`` the pyramidal-anatomy half, the two being
disjoint by construction so no quantity reaches the model twice.
``flag_model`` holds the one model definition every driver shares, and
``single_seed_flag`` is the in-pipeline trainer. There is one flag per cell
type, and a cell is scored by the flag of the type Stage 1 assigns it.
"""
