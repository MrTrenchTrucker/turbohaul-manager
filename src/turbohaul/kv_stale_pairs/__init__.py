"""Drop the saved KV pairs a thread left behind on other engine ports, as one module.

The slot KV save directory holds, per thread, one pair (bin, checkpoint sidecar, metadata) for each
(port, engine slot) the thread was saved on. Once a thread's new pair has landed on a new port, the
pairs it left on other ports are dead weight on the RAM disk. This module finds them and removes them,
so one pair per thread stays.

The caller supplies the facts it alone knows: the size the engine says it wrote, and which ports a live
turn still serves for this thread. When any of those facts is missing or does not add up, nothing is
deleted. The module imports nothing from turbohaul, uses only the standard library, never raises and
logs nothing: the caller logs from the result.
"""

from __future__ import annotations

from .drop import drop_stale_pairs
from .model import DropResult, StalePair

__all__ = [
    "drop_stale_pairs",
    "DropResult",
    "StalePair",
]
