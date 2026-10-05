"""The FSM gains a legal ACTIVE -> POPPED edge.

WHY THE EDGE EXISTS. The Fast Lane design says the designated
victim "gets NO grace timer", and a design assertion says "assert no grace
state is ever entered for the victim". Honouring that means `_serve_on_resident` must
not move the victim's slot into `SlotState.GRACE` at all -- but that function
always ends in `manager.py` with `transition(slot, SlotState.POPPED)`.
Before this edge existed, a slot that skipped the GRACE entry was still ACTIVE
at that point and `transition()` raised `InvalidTransition`. That is why a
designated-victim check would have had to sit BELOW the grace entry and skip only the WAIT: the
constraint had a structural cause in this table, not a sloppy ordering.

WHY IT IS SAFE. Widening an edge removes a loud failure, so the question is what
that failure was catching. Every `transition(<var>, SlotState.POPPED)` site in
manager.py was checked by deriving the incoming state mechanically (enclosing
function -> every event that can change that variable's state -> the last one
before the site). Seven arrive in GRACE or LOADING_FAIL and cannot be ACTIVE;
one is dynamically guarded by `if SlotState.POPPED in legal` and cannot raise
whatever its incoming state; another already arrives ACTIVE today, where the
raise is swallowed by `except InvalidTransition: pass`. The full per-site table
is not reproduced here.

The controls below matter as much as the new edge: this change must widen
exactly one row and must not weaken transition() into accepting anything else.
"""
from __future__ import annotations

import pytest

from turbohaul.fsm import LEGAL_TRANSITIONS, InvalidTransition, transition
from turbohaul.slot import Slot, SlotState


def _slot_in(state: SlotState) -> Slot:
    """A slot parked in `state` without asserting how it got there."""
    return Slot(slot_id="fsm-probe", model_tag="m", state=state, prompt="p")


def _walk_to_active() -> Slot:
    """A slot driven to ACTIVE through real transitions only -- the same path
    `_serve_on_resident`'s anchor takes (STAGED -> LOADING -> ACTIVE), so the
    state under test is reached the way production reaches it."""
    s = _slot_in(SlotState.RECEIVED)
    for st in (SlotState.STAGED, SlotState.LOADING, SlotState.ACTIVE):
        transition(s, st)
    return s


class TestActiveToPoppedEdge:
    def test_active_to_popped_is_legal(self):
        """THE EDGE. Red before the edge was added: InvalidTransition, "illegal
        transition ACTIVE -> POPPED (legal from ACTIVE: ['ACTIVE_MATCH',
        'GRACE'])" -- which is exactly what blocked the victim from leaving
        _serve_on_resident without first entering grace."""
        s = _walk_to_active()
        transition(s, SlotState.POPPED)
        assert s.state is SlotState.POPPED, (
            "a slot that completed its turn without entering GRACE must be "
            "able to reach POPPED -- the designated victim has no grace state to "
            "pass through"
        )

    def test_active_row_is_exactly_the_widened_set(self):
        """The widening is one edge, named explicitly, not an open row."""
        assert LEGAL_TRANSITIONS[SlotState.ACTIVE] == {
            SlotState.GRACE,
            SlotState.ACTIVE_MATCH,
            SlotState.POPPED,
        }


class TestTheWideningIsSurgical:
    """★ CONTROLS -- every test in this class must pass BOTH before and after
    the edge landed. They share no assertion with the two above: these ask "did
    anything ELSE change", which is a different question from "did the edge
    land", and a change that broke them would be invisible to the tests above.
    """

    def test_active_to_grace_still_legal(self):
        """The ordinary, overwhelmingly common path -- a non-victim finishing
        its turn -- is untouched."""
        s = _walk_to_active()
        transition(s, SlotState.GRACE)
        assert s.state is SlotState.GRACE

    def test_active_to_active_match_still_legal(self):
        """The warm follow-up promotion is untouched."""
        s = _walk_to_active()
        transition(s, SlotState.ACTIVE_MATCH)
        assert s.state is SlotState.ACTIVE_MATCH

    @pytest.mark.parametrize(
        "illegal",
        [
            SlotState.LOADING,
            SlotState.STAGED,
            SlotState.COLD,
            SlotState.IDLE_HOT,
            SlotState.LOADING_FAIL,
            SlotState.GRACE_BUSY,
            SlotState.RECEIVED,
        ],
    )
    def test_transition_still_rejects_every_other_target_from_active(self, illegal):
        """⛔ The load-bearing control. Widening a row must not degrade
        transition() into a permissive no-op: everything ACTIVE could not
        reach before, it still cannot reach. If this passes only because
        transition() stopped validating, the test above would pass too --
        which is why the rejection is asserted per-target, by name."""
        s = _walk_to_active()
        with pytest.raises(InvalidTransition):
            transition(s, illegal)
        assert s.state is SlotState.ACTIVE, (
            "a rejected transition must leave the slot where it was"
        )

    def test_no_row_other_than_active_changed(self):
        """The whole table, pinned except for ACTIVE. This is what makes a
        future accidental widening elsewhere visible in review instead of
        silent -- the same reason the widening above had to be justified."""
        expected = {
            SlotState.RECEIVED: {SlotState.ACCEPT_BUFFER, SlotState.STAGED},
            SlotState.ACCEPT_BUFFER: {SlotState.STAGED, SlotState.COLD},
            SlotState.STAGED: {
                SlotState.LOADING, SlotState.COLD, SlotState.ACTIVE_MATCH,
            },
            SlotState.LOADING: {
                SlotState.ACTIVE, SlotState.LOADING_FAIL, SlotState.COLD,
            },
            SlotState.LOADING_FAIL: {SlotState.STAGED, SlotState.POPPED},
            SlotState.GRACE: {
                SlotState.GRACE_BUSY, SlotState.POPPED, SlotState.ACTIVE,
            },
            SlotState.GRACE_BUSY: {SlotState.GRACE, SlotState.POPPED},
            SlotState.ACTIVE_MATCH: {SlotState.ACTIVE},
            SlotState.POPPED: {
                SlotState.IDLE_HOT, SlotState.STAGED, SlotState.COLD,
            },
            SlotState.IDLE_HOT: {
                SlotState.ACTIVE, SlotState.POPPED, SlotState.COLD,
            },
            SlotState.COLD: set(),
        }
        actual = {k: v for k, v in LEGAL_TRANSITIONS.items()
                  if k is not SlotState.ACTIVE}
        assert actual == expected
