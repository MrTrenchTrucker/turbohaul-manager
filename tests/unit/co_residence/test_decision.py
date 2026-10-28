"""co_residence.decision: the per-tag decision (four outcomes, fixed precedence)."""

import dataclasses
import itertools

import pytest

from turbohaul.co_residence import RecheckDecision, RecheckInputs, decide_recheck


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


def _decide(live=1, demand=1, memo=False) -> RecheckDecision:
    """One live engine, one eligible waiting request, no memo: the re-check is due."""
    return decide_recheck(RecheckInputs(tag_live_instances=live, eligible_demand=demand, memo_blocked=memo))


# -- the four outcomes -----------------------------------------------------------

def test_due_when_live_engine_demand_and_no_memo():
    d = _decide()
    assert d == RecheckDecision(run=True, reason="recheck_due"), why(f"due case gave {d!r}")


def test_no_live_instance():
    d = _decide(live=0)
    assert d == RecheckDecision(run=False, reason="no_live_instance"), why(f"cold tag gave {d!r}")


def test_no_eligible_demand():
    d = _decide(demand=0)
    assert d == RecheckDecision(run=False, reason="no_eligible_demand"), why(f"no demand gave {d!r}")


def test_memo_blocked():
    d = _decide(memo=True)
    assert d == RecheckDecision(run=False, reason="no_room_memo"), why(f"memo gave {d!r}")


# -- boundaries -------------------------------------------------------------------

def test_one_live_instance_is_enough_and_zero_or_negative_is_not():
    assert _decide(live=1).run is True, why("1 live engine is the boundary that runs")
    assert _decide(live=0).reason == "no_live_instance"
    assert _decide(live=-1).reason == "no_live_instance", why("a negative count is no instance")


def test_one_eligible_request_is_enough_and_zero_or_negative_is_not():
    assert _decide(demand=1).run is True, why("1 eligible request is the boundary that runs")
    assert _decide(demand=0).reason == "no_eligible_demand"
    assert _decide(demand=-1).reason == "no_eligible_demand", why("a negative count is no demand")


def test_many_instances_and_much_demand_still_run():
    assert _decide(live=3, demand=50).run is True, why("counts above one change nothing")


# -- precedence: every pair, first match wins -------------------------------------

def test_no_live_instance_beats_no_demand():
    assert _decide(live=0, demand=0).reason == "no_live_instance", why("cold tag with no demand")


def test_no_live_instance_beats_memo():
    assert _decide(live=0, memo=True).reason == "no_live_instance", why("cold tag with a memo entry")


def test_no_eligible_demand_beats_memo():
    assert _decide(demand=0, memo=True).reason == "no_eligible_demand", why("no demand with a memo entry")


def test_all_three_causes_report_no_live_instance():
    assert _decide(live=0, demand=0, memo=True).reason == "no_live_instance"


@pytest.mark.parametrize("live, demand, memo", list(itertools.product([-1, 0, 1, 2], [-1, 0, 1, 4], [False, True])))
def test_full_truth_table(live, demand, memo):
    if live < 1:
        want = ("no_live_instance", False)
    elif demand < 1:
        want = ("no_eligible_demand", False)
    elif memo:
        want = ("no_room_memo", False)
    else:
        want = ("recheck_due", True)
    d = _decide(live=live, demand=demand, memo=memo)
    assert (d.reason, d.run) == want, why(f"live={live} demand={demand} memo={memo} gave {d!r}")


def test_run_is_a_bool_and_only_the_due_answer_runs():
    answers = [_decide(live=0), _decide(demand=0), _decide(memo=True)]
    assert [a.run for a in answers] == [False, False, False], why("every non-run answer has run=False")
    assert type(_decide().run) is bool and _decide().run is True


# -- the dataclasses ---------------------------------------------------------------

def test_decision_fields_are_run_then_reason():
    assert [f.name for f in dataclasses.fields(RecheckDecision)] == ["run", "reason"], why("RecheckDecision order")


def test_inputs_fields_and_all_are_required():
    names = [f.name for f in dataclasses.fields(RecheckInputs)]
    assert names == ["tag_live_instances", "eligible_demand", "memo_blocked"], why("RecheckInputs field order")
    for missing in names:
        kw = dict(tag_live_instances=1, eligible_demand=1, memo_blocked=False)
        del kw[missing]
        with pytest.raises(TypeError):
            RecheckInputs(**kw)


def test_inputs_are_keyword_only():
    with pytest.raises(TypeError):
        RecheckInputs(1, 1, False)  # type: ignore[misc]


def test_inputs_and_decision_are_frozen():
    inp = RecheckInputs(tag_live_instances=1, eligible_demand=1, memo_blocked=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        inp.memo_blocked = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        _decide().run = False  # type: ignore[misc]
