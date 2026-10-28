"""decide_retention precedence, frozenness, keyword-only construction, determinism, no mutation.

resident_exit, see src/turbohaul/resident_exit/AGENTS.md
"""

import dataclasses
import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.resident_exit import RetentionDecision, RetentionInputs, decide_retention  # noqa: E402

CARD = "resident_exit, see src/turbohaul/resident_exit/AGENTS.md"

CLEAR = dict(
    idle_window_s=-1.0,
    hold_window_s=30.0,
    max_parallel_sidecars=2,
    live_active_siblings=1,
    designated_victim=False,
    outranking_claim=False,
    capacity_demand_pending=False,
)

# One value per check that trips it, keyed by the reason it must produce, in precedence order.
TRIP = {
    "cap_le_1": dict(max_parallel_sidecars=1),
    "idle_window_positive": dict(idle_window_s=7.5),
    "designated_victim": dict(designated_victim=True),
    "outranking_claim": dict(outranking_claim=True),
    "capacity_demand_pending": dict(capacity_demand_pending=True),
    "no_active_sibling": dict(live_active_siblings=0),
    "no_hold_window": dict(hold_window_s=0.0),
}
CHECKS = list(TRIP)
ADJACENT = list(zip(CHECKS, CHECKS[1:]))


def _with(*names) -> RetentionInputs:
    kw = dict(CLEAR)
    for name in names:
        kw.update(TRIP[name])
    return RetentionInputs(**kw)


@pytest.mark.parametrize("first,second", ADJACENT, ids=[f"{a}_beats_{b}" for a, b in ADJACENT])
def test_earlier_check_wins_over_the_adjacent_later_check(first, second):
    inp = _with(first, second)
    d = decide_retention(inp)
    assert (d.hold, d.reason) == (False, first), (
        f"{CARD}: with {first} and {second} both true, got reason {d.reason!r}"
    )
    assert d.window_s == inp.idle_window_s, f"{CARD}: non-hold window {d.window_s} is not the input idle window"


def test_every_blocker_true_gives_cap_le_1():
    inp = _with(*CHECKS)
    d = decide_retention(inp)
    assert _view(d) == (False, 7.5, "cap_le_1"), f"{CARD}: all blockers true gave {_view(d)}"


def _view(d: RetentionDecision):
    return (d.hold, d.window_s, d.reason)


def test_decision_is_frozen():
    d = decide_retention(_with())
    try:
        d.hold = True
    except dataclasses.FrozenInstanceError as exc:
        assert type(exc) is dataclasses.FrozenInstanceError, f"{CARD}: RetentionDecision raised {type(exc)!r}"
    else:
        raise AssertionError(f"{CARD}: RetentionDecision.hold was assignable; RetentionDecision must be frozen")


def test_inputs_are_frozen():
    inp = _with()
    try:
        inp.idle_window_s = 1.0
    except dataclasses.FrozenInstanceError as exc:
        assert type(exc) is dataclasses.FrozenInstanceError, f"{CARD}: RetentionInputs raised {type(exc)!r}"
    else:
        raise AssertionError(f"{CARD}: RetentionInputs.idle_window_s was assignable; RetentionInputs must be frozen")


def test_inputs_are_keyword_only():
    try:
        RetentionInputs(-1.0, 30.0, 2, 1, False, False, False)
    except TypeError as exc:
        assert type(exc) is TypeError, f"{CARD}: positional RetentionInputs raised {type(exc)!r}"
    else:
        raise AssertionError(
            f"{CARD}: RetentionInputs accepted positional arguments; it must be keyword-only"
        )


def test_identical_inputs_give_equal_decisions():
    a = decide_retention(_with("designated_victim"))
    b = decide_retention(_with("designated_victim"))
    assert a == b, f"{CARD}: two calls on equal inputs differ"
    assert _view(a) == (False, -1.0, "designated_victim"), f"{CARD}: designated input gave {_view(a)}"
    hold_a = decide_retention(_with())
    hold_b = decide_retention(_with())
    assert hold_a == hold_b == RetentionDecision(hold=True, window_s=30.0, reason="co_resident_active_sibling"), (
        f"{CARD}: hold decisions differ or are wrong: {hold_a!r} {hold_b!r}"
    )


def test_call_does_not_mutate_its_input():
    inp = _with("capacity_demand_pending")
    before = dataclasses.asdict(inp)
    decide_retention(inp)
    assert dataclasses.asdict(inp) == before, f"{CARD}: decide_retention changed its input"
    assert inp == _with("capacity_demand_pending"), f"{CARD}: input no longer equals a fresh copy"
