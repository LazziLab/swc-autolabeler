"""Read a model file once and reuse it while the file on disk is unchanged.

Labeling a batch calls the pipeline once per cell, and reading the model files
each time costs more than labeling the cell itself: the Stage 2 bundle takes
about 900 ms to read against roughly 95 ms to label a cell, so a cold read
costs ten cells. The file's size and modification time are part of the key, so
a model retrained into the same path is read again rather than served stale.

The cache holds the ``MAX_ENTRIES`` most recently used files and drops the
rest. A bundle costs around 75 MB resident, and drivers that build a model
directory per round, the curator loop above all, would otherwise accumulate
every one of them for the life of the process. Labeling through one model
directory touches three cached files, the Stage 1 classifier, the Stage 2
bundle and the branch correction, so the bound is set to hold two directories
at once and a driver comparing two of them keeps both warm.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

#: Files kept in memory, most recently used first. One model directory occupies
#: three of these, so six holds two directories: a driver that alternates
#: between them keeps both warm instead of reloading roughly a second's worth of
#: model on every switch.
MAX_ENTRIES = 6

_CACHE: "OrderedDict[str, tuple[tuple[int, int], Any]]" = OrderedDict()
# Guards _CACHE, and is held across the load so two callers cannot read the
# same file at once. Loads of different files serialize as well, which costs
# nothing here: the pipeline parallelizes with processes, not threads.
_LOCK = threading.Lock()


def load_cached(path: str | Path, loader: Callable[[Path], Any]) -> Any:
    """``loader(path)``, remembered until the file at ``path`` changes.

    A missing file goes straight to ``loader``, which reports it in its own words.
    """
    p = Path(path).resolve()
    if not p.is_file():
        return loader(p)
    st = p.stat()
    state = (st.st_mtime_ns, st.st_size)
    key = str(p)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == state:
            _CACHE.move_to_end(key)
            return hit[1]
        value = loader(p)
        _CACHE[key] = (state, value)
        _CACHE.move_to_end(key)
        while len(_CACHE) > MAX_ENTRIES:
            _CACHE.popitem(last=False)
        return value


def clear() -> None:
    """Drop every cached model, releasing the memory they hold."""
    with _LOCK:
        _CACHE.clear()
