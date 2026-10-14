"""Which cells are held out: the one definition, used by training and scoring.

A file is in the test split when the first eight hexadecimal digits of the MD5
digest of ``"{seed}:{filename}"``, read as a fraction of 0xFFFFFFFF, fall below
``test_size``. The file name carries its source prefix, so the assignment
depends on the name alone: no ordering, no shuffling, no state on disk. Any
process that knows a seed can reconstruct the same split, and a cell keeps its
side of it however the corpus grows.

This is the rule the whole evaluation rests on. It decided which cells every
released model never saw, and every reported number is measured on the other
side of it. It is defined once, here, because a trainer and a scorer that
disagree by a single cell would score a model on a cell it had learned. Ten
copies of it agreed when this was written; one definition cannot disagree.
"""
from __future__ import annotations

import hashlib

#: The held-out fraction used throughout the paper.
DEFAULT_TEST_SIZE = 0.20


def in_test(file_name: str, seed: int, test_size: float = DEFAULT_TEST_SIZE) -> bool:
    """True when this file belongs to the seed's held-out set."""
    digest = hashlib.md5(f"{seed}:{file_name}".encode()).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < test_size
