"""co_residence.window: the box-wide debounce (two outcomes) and the demand age test.

The gate does not read the box budget: admission is decided by whether the engine fits on a card.
The reasons ``cap_le_1`` and ``box_full`` and their two parameters do not exist.
"""

import dataclasses

import pytest

from turbohaul.co_residence import RecheckGate, demand_is_eligible, recheck_window


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


NAN = float("nan")
INF = float("inf")


def _gate(**kw) -> RecheckGate:
    """Never re-checked, 10 s interval: the gate is open."""
    args = dict(now=100.0, last_recheck_at=None, min_interval_s=10.0)
    args.update(kw)
    return recheck_window(**args)


# -- the two outcomes --------------------------------------------------------

def test_open_window_has_open_true_and_reason_window_open():
    g = _gate()
    assert g == RecheckGate(open=True, reason="window_open"), why(f"open box gave {g!r}")


def test_recent_attempt_is_debounced():
    g = _gate(now=100.0, last_recheck_at=95.0)
    assert g == RecheckGate(open=False, reason="debounced"), why(f"5 s after an attempt gave {g!r}")


def test_the_gate_takes_no_box_budget_or_resident_count():
    # Fit-based admission: the budget-based parameters must be rejected, not silently ignored.
    for extra in ({"max_parallel_sidecars": 1}, {"live_residents": 5}):
        with pytest.raises(TypeError):
            _gate(**extra)


def test_a_budget_of_one_and_a_full_box_no_longer_close_the_gate():
    # The reasons cap_le_1 and box_full cannot be produced.
    reasons = {_gate().reason, _gate(last_recheck_at=99.0).reason, _gate(last_recheck_at=500.0).reason}
    assert reasons == {"window_open", "debounced"}, why(f"unexpected gate reasons {reasons!r}")


# -- boundaries --------------------------------------------------------------

def test_elapsed_exactly_the_interval_is_not_debounced():
    g = _gate(now=100.0, last_recheck_at=90.0, min_interval_s=10.0)
    assert g.open is True and g.reason == "window_open", why("elapsed == interval opens the gate (strict <)")


def test_elapsed_just_under_the_interval_is_debounced():
    g = _gate(now=99.99, last_recheck_at=90.0, min_interval_s=10.0)
    assert g.reason == "debounced" and g.open is False, why("elapsed 9.99 < 10 is debounced")


def test_zero_elapsed_is_debounced():
    assert _gate(now=100.0, last_recheck_at=100.0).reason == "debounced", why("same instant is debounced")


def test_first_ever_attempt_is_never_debounced():
    # last_recheck_at None means "never", even at a tiny `now` that would look recent.
    g = _gate(now=0.0, last_recheck_at=None)
    assert g.open is True, why("no earlier attempt, nothing to debounce")


def test_last_attempt_of_zero_is_a_real_time_not_none():
    # 0.0 is falsy; a truthiness check would treat it as "never attempted".
    g = _gate(now=5.0, last_recheck_at=0.0)
    assert g.reason == "debounced", why("last_recheck_at=0.0 is an attempt at t=0, not 'never'")


def test_negative_elapsed_time_is_debounced():
    g = _gate(now=100.0, last_recheck_at=105.0)
    assert g == RecheckGate(open=False, reason="debounced"), why(f"clock going backwards gave {g!r}")


def test_interval_comes_from_the_argument():
    assert _gate(now=100.0, last_recheck_at=97.0, min_interval_s=3.0).open is True
    assert _gate(now=100.0, last_recheck_at=97.0, min_interval_s=3.5).reason == "debounced"


# -- the dataclass ------------------------------------------------------------

def test_gate_fields_are_open_then_reason():
    assert [f.name for f in dataclasses.fields(RecheckGate)] == ["open", "reason"], why("RecheckGate field order")


def test_gate_is_frozen():
    g = _gate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.open = False  # type: ignore[misc]


def test_recheck_window_is_keyword_only():
    with pytest.raises(TypeError):
        recheck_window(100.0, None, 10.0)  # type: ignore[misc]


# -- demand_is_eligible ---------------------------------------------------------

@pytest.mark.parametrize(
    "age, expected",
    [(0.0, False), (4.99, False), (4.999999, False), (5.0, True), (5.01, True), (119.99, True), (1e9, True)],
)
def test_unprotected_demand_counts_from_five_seconds(age, expected):
    assert demand_is_eligible(age_s=age, protected=False) is expected, why(f"unprotected age {age}")


@pytest.mark.parametrize(
    "age, expected",
    [(0.0, False), (5.0, False), (60.0, False), (119.99, False), (119.999999, False), (120.0, True), (120.01, True)],
)
def test_protected_demand_counts_from_one_hundred_twenty_seconds(age, expected):
    assert demand_is_eligible(age_s=age, protected=True) is expected, why(f"protected age {age}")


@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("age", [NAN, INF, -INF, -1.0, -0.001, -1e9])
def test_non_finite_or_negative_age_is_never_eligible(age, protected):
    assert demand_is_eligible(age_s=age, protected=protected) is False, why(f"age {age!r} must not trigger a spawn")


def test_demand_is_eligible_is_keyword_only_and_returns_a_bool():
    with pytest.raises(TypeError):
        demand_is_eligible(10.0, False)  # type: ignore[misc]
    assert type(demand_is_eligible(age_s=10.0, protected=False)) is bool
