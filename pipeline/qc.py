"""Structural quality control shared by the corpus scan and the inference command.

A reconstruction is labeled only if, after normalization, it is one connected
cell with a soma, at least one neurite, finite non-negative geometry and a node
count in a workable range. ``qc_violations`` names every condition a parsed
file fails; an empty list is a pass. ``pipeline._scan_corpus_qc`` records the
same checks over the corpus, one row per file.
"""
from __future__ import annotations

import math

from .features import SWCNode

MIN_NODES = 10           # smallest usable reconstruction
MAX_NODES = 200_000      # largest the pipeline is meant to take
NEURITE_TYPES = (2, 3, 4)


def _finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def qc_violations(nodes: list[SWCNode]) -> list[str]:
    """Names of the structural checks ``nodes`` fail; empty when the file passes.

    ``nodes`` must be the normalized parse (``parse_swc`` with its defaults):
    custom type codes rewritten and connected soma nodes merged, the view that
    training and inference see.
    """
    failed: list[str] = []
    n = len(nodes)
    if not (MIN_NODES <= n <= MAX_NODES):
        failed.append(f"node count {n} outside {MIN_NODES}..{MAX_NODES}")
    if not any(nd.type == 1 for nd in nodes):
        failed.append("no soma node")
    if not any(nd.type in NEURITE_TYPES for nd in nodes):
        failed.append("no neurite node")
    ids = {nd.id for nd in nodes}
    roots = sum(1 for nd in nodes if nd.parent == -1)
    orphans = sum(1 for nd in nodes if nd.parent != -1 and nd.parent not in ids)
    if roots != 1 or orphans:
        failed.append(f"not one connected cell ({roots} roots, {orphans} orphan nodes)")
    if not all(_finite(nd.radius) and nd.radius >= 0 for nd in nodes):
        failed.append("a radius is negative or not finite")
    if not all(_finite(nd.x) and _finite(nd.y) and _finite(nd.z) for nd in nodes):
        failed.append("a coordinate is not finite")
    return failed
