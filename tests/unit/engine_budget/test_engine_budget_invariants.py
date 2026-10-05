"""The engine budget's hard invariants, swept exhaustively.

Hand-picked inputs only pin arithmetic, so nothing else pinned the two properties
that actually matter:

  * the box must never end up over ``max_parallel_sidecars`` because of a spawn this
    function authorised;
  * a starved budget or an unreadable probe must never shrink a tag below the
    engines it already runs.

The invariant is on the CALLER's view, not the raw number. The caller admits an
engine when ``len(instances) < eff_cap``, so what must never happen is::

    live_engines + (eff_cap - own_instances) > max_parallel_sidecars

A box ALREADY over budget (``live > budget``) is a pre-existing condition and is out
of scope: admitting nothing is correct there.

SCOPE, measured not assumed. Both properties survive two classes of mutation, and
it is worth being exact about which is which:

  * GROWTH TERM (bites). Dropping the free-global-slot term lets a tag grow past
    the budget -> ``test_never_admits_an_engine_past_the_global_budget`` goes red.
    Falsified: 1 failed -> 5 passed.
  * FLOOR (cannot bite, by design). Under-reporting a branch is impossible because
    the floor is applied once, in ``budget()``, and every branch routes through it.
    Mutating a branch to return less still yields ``max(own, ...)``. That is the
    single-clamp design doing its job; ``test_never_shrinks_a_tag_below_the_engines_
    already_running`` is a guard against a future refactor that reintroduces a
    per-branch floor, not a pin on today's arithmetic.
"""

import sys
from itertools import product
from pathlib import Path

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.engine_budget import EngineBudgetInputs, effective_engine_cap  # noqa: E402


def _state(budget, live, own, cards, width=1) -> EngineBudgetInputs:
    """Single construction point, so no caller can forget `parallel_width`."""
    return EngineBudgetInputs(budget, live, own, cards, width)


def _with_width(i: EngineBudgetInputs, width: int) -> EngineBudgetInputs:
    return _state(i.max_parallel_sidecars, i.live_residents,
                  i.own_instances, i.cards_that_fit, width)


def _sweep():
    """Every reachable state: a tag cannot own more engines than exist."""
    for budget, live, own, cards, width in product(
        (1, 2, 3), (0, 1, 2, 3, 4), (0, 1, 2), (0, 1, 2, 3), (1, 2, 4)
    ):
        if own <= live:
            yield _state(budget, live, own, cards, width)


def test_never_admits_an_engine_past_the_global_budget():
    """THE INVARIANT: a spawn this authorises cannot overcommit the box."""
    breaches = [
        (i, b.engines, b.engines - i.own_instances)
        for i in _sweep()
        if (b := effective_engine_cap(i)).engines > i.own_instances
        and i.live_residents + b.engines - i.own_instances > i.max_parallel_sidecars
    ]
    assert not breaches, f"engine budget breached: {breaches[:5]}"


def test_never_shrinks_a_tag_below_the_engines_already_running():
    """A starved budget or unreadable probe must not cost a tag its engines."""
    shrinks = [
        (i, b.engines)
        for i in _sweep()
        if (b := effective_engine_cap(i)).engines < i.own_instances
    ]
    assert not shrinks, f"tag shrunk below its live engines: {shrinks[:5]}"


def test_parallel_width_never_moves_the_answer():
    """Windows per engine never change the engine count, swept rather than sampled."""
    for width in (1, 3, 8):
        for i in _sweep():
            assert (effective_engine_cap(_with_width(i, width)).engines
                    == effective_engine_cap(_with_width(i, 1)).engines), (
                f"parallel_width={width} moved the count at {i}")


def test_a_cold_tag_on_a_full_budget_is_refused_not_floored():
    """0 < 1 would make the caller's spawn gate true past max_parallel_sidecars."""
    b = effective_engine_cap(
        _state(2, 2, 0, 2)
    )
    assert (b.engines, b.additional_engines) == (0, 0)


def test_two_sidecars_one_window_each_still_admit_the_second_sidecar():
    """Stated positively: a box budget of 2 admits the second sidecar, 1 window each."""
    b = effective_engine_cap(
        _state(2, 1, 1, 1)
    )
    assert (b.engines, b.additional_engines) == (2, 1)


# The next four tests pin the "no card reported as fitting" branch. Every expected
# number below is derived from the module card's invariants, not from running the
# code: a tag with nothing running and no fitting card still gets one engine; the box
# never exceeds max_parallel_sidecars; a tag never drops below the engines it runs.

def test_a_cold_tag_with_no_fitting_card_still_gets_one_engine():
    """Nothing running, no card reported as fitting, budget 2: one engine.

    Fails if the no-card branch stops granting the single engine (the spawn-time
    memory check, not this module, guards the actual load).
    """
    b = effective_engine_cap(_state(2, 0, 0, 0))
    assert b.engines == 1
    assert b.additional_engines == 1


def test_a_cold_tag_with_no_fitting_card_gets_one_engine_on_a_budget_of_one():
    """The same, with the smallest box budget: one engine, never more."""
    b = effective_engine_cap(_state(1, 0, 0, 0))
    assert b.engines == 1


def test_a_spent_budget_beats_the_single_pin_for_a_cold_tag():
    """Budget 2 fully held by other tags (live=2), this tag has nothing, no card fits.

    The box-wide budget is a hard limit, so the answer is 0, not the single pin of
    the no-card case. Fails if the spent-budget branch falls through.
    """
    b = effective_engine_cap(_state(2, 2, 0, 0))
    assert b.engines == 0
    assert b.additional_engines == 0


def test_a_running_tag_with_no_fitting_card_keeps_its_engine_without_growing():
    """One engine running, budget 2 with a free slot, no card reported as fitting.

    The tag keeps the engine it runs (floor) and gets no second one (no card fits).
    """
    b = effective_engine_cap(_state(2, 1, 1, 0))
    assert b.engines == 1
    assert b.additional_engines == 0


def test_many_fitting_cards_never_lift_the_total_past_the_box_budget():
    """Six cards could each host a copy, but the box budget is 2.

    The returned number is this tag's TOTAL. Engines of other tags use the same
    budget, so whenever the tag is given more engines than it runs, its total plus
    the other tags' engines must stay within 2. The expected totals are worked out by
    hand: free slots = 2 - (own + others); the tag gains the smaller of the free
    slots and the six cards on top of what it runs; a spent budget holds the tag at
    the engines it already runs.
    """
    expected = {  # (own, others) -> total engines for this tag
        (0, 0): 2, (0, 1): 1, (0, 2): 0, (0, 3): 0,
        (1, 0): 2, (1, 1): 1, (1, 2): 1, (1, 3): 1,
        (2, 0): 2, (2, 1): 2, (2, 2): 2, (2, 3): 2,
    }
    for (own, others), total in expected.items():
        b = effective_engine_cap(_state(2, own + others, own, 6))
        assert b.engines == total, f"own={own} others={others}: got {b.engines}"
        assert b.engines == own or others + b.engines <= 2, (
            f"own={own} others={others}: {others} + {b.engines} is over the box budget"
        )
