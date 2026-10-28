"""The box-wide gate for the re-check, and the age test for waiting demand.

The gate answers one question for the whole box: may a re-check attempt run now (has the
interval since the last attempt passed)? It is checked before any per-tag work, so a closed
gate costs the dispatch loop almost nothing.
The age test decides which waiting requests count as demand at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import DEMAND_MIN_AGE_S, PROTECTED_DEMAND_MIN_AGE_S


@dataclass(frozen=True)
class RecheckGate:
    """Whether a re-check attempt may run now, with the reason attached for a log."""

    open: bool
    reason: str


def recheck_window(*, now: float, last_recheck_at: float | None, min_interval_s: float) -> RecheckGate:
    """Decide whether the box-wide re-check may run an attempt now: debounce only.

    The gate does not look at the box budget or at how many engines are live. Whether a
    further engine is allowed is decided later, by whether it fits on a card.
    """
    if last_recheck_at is not None and (now - last_recheck_at) < min_interval_s:
        # A negative elapsed time (the clock went backwards) also lands here: it must not
        # read as "long ago" and open the gate.
        return RecheckGate(open=False, reason="debounced")
    return RecheckGate(open=True, reason="window_open")


def demand_is_eligible(*, age_s: float, protected: bool) -> bool:
    """True when a waiting request has waited long enough to count as starved demand.

    A protected request (its thread still has a live grace pair, or it belongs to the
    engine's own idle thread) waits longer, because the engine serves it first. A
    non-finite or negative age is never eligible: a bad clock reading must not trigger a
    spawn.
    """
    if not math.isfinite(age_s):
        return False
    return age_s >= (PROTECTED_DEMAND_MIN_AGE_S if protected else DEMAND_MIN_AGE_S)
