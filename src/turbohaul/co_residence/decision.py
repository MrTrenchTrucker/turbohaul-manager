"""The per-tag decision: should the re-check probe this tag now?

The box-wide gate (window.py) has already said an attempt may run. This file looks at one
tag, using numbers the caller measured, and returns a yes or no with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RecheckInputs:
    """What the caller measured for one tag."""

    tag_live_instances: int
    """Engines of this tag that are live."""

    eligible_demand: int
    """Waiting same-model requests that are old enough to count (see ``demand_is_eligible``)."""

    memo_blocked: bool
    """True while the tag's no-room memo entry is still live."""


@dataclass(frozen=True)
class RecheckDecision:
    """The decision, with the reason attached so a log can say which rule applied."""

    run: bool
    reason: str


def decide_recheck(inputs: RecheckInputs) -> RecheckDecision:
    """Say whether to re-ask the second-engine spawn path for this tag.

    Checked in a fixed order, first match wins. A tag with no live engine is a cold start,
    which the normal path already serves, so it is never a co-residence question. The memo
    comes last so the reason shows demand was there and only the memo held it back.
    """
    if inputs.tag_live_instances < 1:
        return RecheckDecision(run=False, reason="no_live_instance")
    if inputs.eligible_demand < 1:
        return RecheckDecision(run=False, reason="no_eligible_demand")
    if inputs.memo_blocked:
        return RecheckDecision(run=False, reason="no_room_memo")
    return RecheckDecision(run=True, reason="recheck_due")
