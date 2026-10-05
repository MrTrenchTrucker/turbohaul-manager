"""Engine budget for one model tag.

This module decides how many engine processes (llama-server sidecars) one model tag
may run at the same time. It returns the TOTAL for the tag, once every admitted
engine has loaded.

Only two limits bound that number:

* the box-wide budget, ``queue.max_parallel_sidecars``: the most engines the whole
  box runs at once, for all tags together;
* the cards that can host another copy of the model right now.

The answer is never lower than the engines the tag already runs, so a failed
probe, a spent budget or a starved reading cannot shut down a live engine. When the
caller reports that the manifest could not be read, the module fails safe and holds
the live engines: it neither adds nor removes any.

Windows per engine are a separate setting. ``parallel_width`` carries the manifest's
``llama_server_flags.parallel`` (the caller supplies it); it multiplies into the
context-window total reported by :func:`context_window_capacity` but never changes
the engine count. The per-model ``max_instances`` manifest setting was retired and
this module never reads it.

The module is pure: no I/O, no logging, no locks, no globals. The caller holds the
registry lock and supplies the already-measured inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "EngineBudgetInputs",
    "EngineBudget",
    "effective_engine_cap",
    "context_window_capacity",
]


@dataclass(frozen=True)
class EngineBudgetInputs:
    """Everything the cap needs, measured by the caller.

    ``parallel_width`` is the number of context windows one engine serves (the
    manifest's ``llama_server_flags.parallel``). It is deliberately NOT an input to
    the engine count: it only feeds :func:`context_window_capacity`.
    """

    max_parallel_sidecars: int
    """Global engine budget for the whole box (``queue.max_parallel_sidecars``)."""

    live_residents: int
    """Engines currently loaded for ALL tags."""

    own_instances: int
    """Engines currently loaded for THIS tag."""

    cards_that_fit: int
    """Distinct cards that can host another engine of this model right now."""

    parallel_width: int
    """Context windows per engine (the manifest's ``llama_server_flags.parallel``)."""

    manifest_unreadable: bool = False
    """True when the manifest read raised. The decision then holds the live engines."""


@dataclass(frozen=True)
class EngineBudget:
    """The decision, with the reason attached so a log can say which limit applied."""

    engines: int
    """TOTAL engines this tag may run once every admitted one has loaded.

    Total, not additional: the caller branches on "more than one engine", so this
    must be the tag's whole capacity. The count still to start is available
    separately as ``additional_engines``.
    """

    own_instances: int
    """Engines of this tag already live, so the additional count is derivable."""

    reason: str
    """Which input bound the answer. Never a bare number in a log."""

    @property
    def admits_additional_engine(self) -> bool:
        """True when at least one MORE engine than the tag already runs is allowed."""
        return self.engines > self.own_instances

    @property
    def additional_engines(self) -> int:
        """How many more engines the tag may run beyond the ones already live."""
        return max(0, self.engines - self.own_instances)


def effective_engine_cap(inp: EngineBudgetInputs) -> EngineBudget:
    """TOTAL engines this tag may run, given the measured inputs.

    TOTAL, not additional: the caller branches on "more than one engine", so this
    is the tag's whole capacity. ``additional_engines`` derives the remainder.

    Ceilings: the box-wide engine budget (``queue.max_parallel_sidecars``) and the
    cards that can host the model, with a floor at the engines already live -- a
    starved or unreadable probe must not shrink a running tag, or the caller drops
    its engine.

    ``parallel_width`` (windows per engine) is deliberately NOT an input to the
    engine count; it appears in the reason text only.
    """
    own = max(0, inp.own_instances)

    def budget(total: int, reason: str) -> EngineBudget:
        """Clamp once, so no branch can forget the floor or the reason."""
        return EngineBudget(max(own, min(inp.max_parallel_sidecars, total)), own, reason)

    if inp.manifest_unreadable:
        # A manifest that cannot be read must never propagate out of routing.
        return budget(own, "manifest unreadable -> fail-safe, hold live engines")

    free = inp.max_parallel_sidecars - inp.live_residents
    if free <= 0:
        return budget(own, f"global budget spent (live={inp.live_residents} "
                           f"of {inp.max_parallel_sidecars})")

    if inp.cards_that_fit <= 0:
        # An unreadable probe is not evidence the model cannot fit, so a cold tag
        # still degrades to one pin rather than 0. The spawn-time VRAM gate guards
        # the actual load.
        return budget(1, "no card fits -> single pin")

    # A tag may hold every global slot another tag is not holding.
    return budget(
        own + min(free, inp.cards_that_fit),
        f"admitted: {free} free slot(s), {inp.cards_that_fit} card(s) fit, "
        f"parallel_width={inp.parallel_width}",
    )


def context_window_capacity(inp: EngineBudgetInputs, engines_total: int) -> int:
    """Total concurrent context windows this tag can serve.

    ``engines_total`` is the tag's TOTAL engine count -- exactly what
    :func:`effective_engine_cap` returns -- not a count of engines still to be
    admitted.
    """
    return max(0, engines_total) * max(1, inp.parallel_width)
