"""Stage 3: close the labeling, first by rule and then by a learned correction.

Two parts, in this order. ``refine`` applies the whole-cell topology rules, the
one view no learned stage has: Stage 2 types each primary subtree in isolation,
so nothing upstream can enforce a property of the whole cell such as a single
axon or a single apical trunk. ``branch_correction`` then re-decides individual
branches near the soma from their own geometry, overriding the subtree's label
only where it disagrees and is confident enough.

The correction is trained on, and applied to, only the branches whose midpoint
lies within a fixed path distance of the soma; the scope is stored in the
released bundle and enforced at inference.
"""
