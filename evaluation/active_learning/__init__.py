"""Stage 5 experiments: the curator feedback loop.

Outer-loop experiments, not part of per-cell inference. They simulate the
deployment cycle in which a curator reviews the cells the flag surfaces, and
measure what those reviews buy:

  flag_feedback   the labeler is the fixed factory pipeline and only the flag
                  learns from the reviews (Sec 5.7)
  curator_loop    both the Stage 2 labeler and the flag are retrained as
                  corrections accumulate, with the endpoints measured through
                  the complete pipeline (Sec 5.7)

Ground truth stands in for the curator: revealing a cell's labels is that cell
being reviewed. Both experiments are pyramidal, matching the deployed flag.
"""
