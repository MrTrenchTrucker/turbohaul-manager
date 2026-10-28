"""The save-time hard RAM cap decision.

Before a slot KV snapshot is written, the caller asks whether the directory is already at
or over its cap. If it is, one GC pass may make room. If the directory is still at or over
the cap after that pass, the snapshot is declined. The live KV stays in the engine; only the
on-disk copy is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable


def cap_exceeded(total_bytes: int, cap_bytes: int) -> bool:
    """True when ``total_bytes`` has reached ``cap_bytes``.

    The comparison is ``>=``: a directory sitting exactly at the cap has no room for the next
    snapshot. A cap of zero or less means no cap, so it is never exceeded.
    """
    return cap_bytes > 0 and total_bytes >= cap_bytes


@dataclass(frozen=True)
class SaveGate:
    """The outcome of :func:`enforce_save_cap`, with the numbers a log line needs."""

    proceed: bool
    total_bytes: int
    cap_bytes: int
    made_room: bool


async def enforce_save_cap(cap_bytes: int, measure: Callable[[], int],
                           make_room: Callable[[], Awaitable[object]]) -> SaveGate:
    """Decide whether a save may proceed, driving ``make_room`` once if the cap is reached.

    ``measure`` returns the directory's bytes now. ``make_room`` runs the caller's GC. It is
    called at most once, and only when the first measurement is at or over the cap. The
    second measurement after it decides the outcome. Exceptions from either callable
    propagate to the caller unchanged.
    """
    if cap_bytes <= 0:
        return SaveGate(True, 0, cap_bytes, False)
    total = measure()
    if not cap_exceeded(total, cap_bytes):
        return SaveGate(True, total, cap_bytes, False)
    await make_room()
    total = measure()
    return SaveGate(not cap_exceeded(total, cap_bytes), total, cap_bytes, True)
