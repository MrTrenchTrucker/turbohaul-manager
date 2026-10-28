"""decide_retention: the smallest input that reaches each outcome.

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


def _inp(**kw) -> RetentionInputs:
    """A clean hold: cap 2, unload at once, one active sibling, 30 s hold, nothing else pending."""
    base = dict(
        idle_window_s=-1.0,
        hold_window_s=30.0,
        max_parallel_sidecars=2,
        live_active_siblings=1,
        designated_victim=False,
        outranking_claim=False,
        capacity_demand_pending=False,
    )
    base.update(kw)
    return RetentionInputs(**base)


def _view(d: RetentionDecision):
    return (d.hold, d.window_s, d.reason)


def test_input_fields_are_exactly_the_measured_signals_and_all_required():
    fields = dataclasses.fields(RetentionInputs)
    names = [f.name for f in fields]
    assert names == [
        "idle_window_s", "hold_window_s", "max_parallel_sidecars", "live_active_siblings",
        "designated_victim", "outranking_claim", "capacity_demand_pending",
    ], f"{CARD}: RetentionInputs fields are {names}"
    for f in fields:
        assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, (
            f"{CARD}: RetentionInputs.{f.name} has a default"
        )


def test_decision_fields_are_hold_window_and_reason():
    names = [f.name for f in dataclasses.fields(RetentionDecision)]
    assert names == ["hold", "window_s", "reason"], f"{CARD}: RetentionDecision fields are {names}"


@pytest.mark.parametrize("cap,idle", [(1, -1.0), (0, 7.5)])
def test_cap_below_two_never_holds(cap, idle):
    d = decide_retention(_inp(max_parallel_sidecars=cap, idle_window_s=idle))
    assert _view(d) == (False, idle, "cap_le_1"), f"{CARD}: cap {cap} idle {idle} gave {_view(d)}"


@pytest.mark.parametrize("idle", [5.0, 0.001])
def test_positive_idle_window_is_not_held(idle):
    d = decide_retention(_inp(idle_window_s=idle))
    assert _view(d) == (False, idle, "idle_window_positive"), f"{CARD}: idle {idle} gave {_view(d)}"


def test_designated_victim_is_not_held():
    d = decide_retention(_inp(designated_victim=True, idle_window_s=0.0))
    assert _view(d) == (False, 0.0, "designated_victim"), f"{CARD}: gave {_view(d)}"


def test_outranking_claim_is_not_held():
    d = decide_retention(_inp(outranking_claim=True, idle_window_s=-1.0))
    assert _view(d) == (False, -1.0, "outranking_claim"), f"{CARD}: gave {_view(d)}"


def test_capacity_demand_pending_is_not_held():
    d = decide_retention(_inp(capacity_demand_pending=True, idle_window_s=0.0))
    assert _view(d) == (False, 0.0, "capacity_demand_pending"), f"{CARD}: gave {_view(d)}"


def test_no_active_sibling_is_not_held():
    d = decide_retention(_inp(live_active_siblings=0, idle_window_s=-1.0))
    assert _view(d) == (False, -1.0, "no_active_sibling"), f"{CARD}: gave {_view(d)}"


@pytest.mark.parametrize("hold", [0.0, -5.0])
def test_no_hold_window_is_not_held(hold):
    d = decide_retention(_inp(hold_window_s=hold, idle_window_s=0.0))
    assert _view(d) == (False, 0.0, "no_hold_window"), f"{CARD}: hold {hold} gave {_view(d)}"


@pytest.mark.parametrize("idle", [0.0, -1.0])
@pytest.mark.parametrize("siblings", [1, 3])
@pytest.mark.parametrize("cap", [2, 4])
def test_active_sibling_holds_for_an_idle_window_of_zero_or_less(idle, siblings, cap):
    d = decide_retention(_inp(idle_window_s=idle, live_active_siblings=siblings, max_parallel_sidecars=cap))
    assert _view(d) == (True, 30.0, "co_resident_active_sibling"), (
        f"{CARD}: idle {idle} siblings {siblings} cap {cap} gave {_view(d)}"
    )


def test_hold_window_is_the_one_the_caller_offered():
    d = decide_retention(_inp(hold_window_s=12.5, idle_window_s=-1.0))
    assert _view(d) == (True, 12.5, "co_resident_active_sibling"), f"{CARD}: gave {_view(d)}"
