"""Keep a finishing co-resident warm, or let it go: the bounded decision.

A resident whose request just ended on a box that already runs another resident of the
same model would normally be torn down at once when its keep_alive is 0. This module
decides whether it stays warm for a bounded window instead. It never reads live state:
the caller measures every input, under its own lock, and passes the numbers in.

The identity class of the finished request (disposable, curator, ...) is deliberately not
an input. A disposable identity decides only whether its KV cache is saved, never whether
the resident itself lives, so the keep-warm answer must not depend on it.

Holding is bounded. The caller parks the resident IDLE_EVICTABLE with a real deadline, so
make-room can still evict it for a higher-ranked waiter that arrives later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RetentionInputs:
    """Everything the keep-warm decision needs, measured by the caller.

    Every field is required: a default here would let a caller forget a signal and
    silently get the answer for a box that does not have it.
    """

    idle_window_s: float
    """The idle window the resident would use now. Zero or less means unload at once."""

    hold_window_s: float
    """The bounded hold the caller offers if the decision is to hold."""

    max_parallel_sidecars: int
    """The box-wide engine budget. Below 2 there is never a co-resident to keep warm for."""

    live_active_siblings: int
    """Other live residents of the same model that are ACTIVE right now."""

    designated_victim: bool
    """The make-room pass has picked this resident as the next victim."""

    outranking_claim: bool
    """A strictly higher-ranked request is waiting for this resident's capacity."""

    capacity_demand_pending: bool
    """Someone else is waiting for the capacity this resident holds: other-model demand,
    or a pending reclaim on its card. Holding would keep that capacity from them."""


@dataclass(frozen=True)
class RetentionDecision:
    """The keep-warm answer, with the rule that produced it so a log can say why."""

    hold: bool
    """True when the resident should stay warm for window_s."""

    window_s: float
    """The idle window to use. Equal to the input idle window whenever hold is False."""

    reason: str
    """Which rule decided the answer. Never a bare True or False in a log."""


def decide_retention(inputs: RetentionInputs) -> RetentionDecision:
    """Apply the rules in order; the first one that matches decides.

    The order is a contract: a cap of 1 always wins over the rest, and a designated
    victim is never held even when a sibling is active. Every answer that is not a hold
    returns the input idle window unchanged, so a caller that ignores the decision keeps
    the plain idle-window behaviour.
    """
    if inputs.max_parallel_sidecars < 2:
        return _no_hold(inputs, "cap_le_1")
    if inputs.idle_window_s > 0:
        return _no_hold(inputs, "idle_window_positive")
    if inputs.designated_victim:
        return _no_hold(inputs, "designated_victim")
    if inputs.outranking_claim:
        return _no_hold(inputs, "outranking_claim")
    if inputs.capacity_demand_pending:
        return _no_hold(inputs, "capacity_demand_pending")
    if inputs.live_active_siblings < 1:
        return _no_hold(inputs, "no_active_sibling")
    if inputs.hold_window_s <= 0:
        return _no_hold(inputs, "no_hold_window")
    return RetentionDecision(
        hold=True, window_s=inputs.hold_window_s, reason="co_resident_active_sibling",
    )


def _no_hold(inputs: RetentionInputs, reason: str) -> RetentionDecision:
    """The unchanged answer: no hold, the caller's own idle window, and the named rule."""
    return RetentionDecision(hold=False, window_s=inputs.idle_window_s, reason=reason)
