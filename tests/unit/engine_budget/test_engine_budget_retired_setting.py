"""The per-model engine limit is gone: engines come from the box budget and the cards.

The manifest used to carry ``max_instances``, a per-model limit on engine processes
that silently overrode ``queue.max_parallel_sidecars``. It is retired. These tests
pin, with real calls, that the engine budget module has no such input, that the
engine count ignores windows per engine, and that windows per engine only feed the
context-window total.

Related sweeps live in ``test_engine_budget_invariants.py``
(``test_parallel_width_never_moves_the_answer`` covers widths 1, 3 and 8 over a
consistent grid; the grid here is wider, adds width 2 and also covers inconsistent
live counts).
"""

import dataclasses
import inspect
import sys
from itertools import product
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

import turbohaul.engine_budget as engine_budget  # noqa: E402
from turbohaul.engine_budget import (  # noqa: E402
    EngineBudget,
    EngineBudgetInputs,
    context_window_capacity,
    effective_engine_cap,
)

RETIRED = "max_instances"


def _inp(**kw) -> EngineBudgetInputs:
    """One tag, one engine already loaded, one more card free, budget of 2."""
    base = dict(
        max_parallel_sidecars=2,
        live_residents=1,
        own_instances=1,
        cards_that_fit=1,
        parallel_width=1,
        manifest_unreadable=False,
    )
    base.update(kw)
    return EngineBudgetInputs(**base)


# -- the retired name is not an input -----------------------------------------

def test_inputs_have_no_retired_field_and_reject_the_keyword():
    """Fails if the field is added back to EngineBudgetInputs."""
    names = {f.name for f in dataclasses.fields(EngineBudgetInputs)}
    assert RETIRED not in names
    with pytest.raises(TypeError):
        EngineBudgetInputs(
            max_parallel_sidecars=2,
            live_residents=1,
            own_instances=1,
            cards_that_fit=1,
            parallel_width=1,
            **{RETIRED: 1},
        )


def test_no_public_name_carries_the_retired_setting():
    """Fails if any public dataclass field, function parameter or module name
    mentions the retired setting. Introspection only, no source text search."""
    fields = {f.name for cls in (EngineBudgetInputs, EngineBudget)
              for f in dataclasses.fields(cls)}
    params = {p for fn in (effective_engine_cap, context_window_capacity)
              for p in inspect.signature(fn).parameters}
    public_names = {n for n in dir(engine_budget) if not n.startswith("_")}
    assert RETIRED not in fields
    assert RETIRED not in params
    assert RETIRED not in public_names
    assert not [n for n in fields | params | public_names if RETIRED in n]


# -- engines come only from the box budget and the cards ----------------------

def test_engine_count_is_identical_for_every_window_width():
    """Fails if the engine count depends on parallel_width."""
    checked = 0
    for budget, live, own, cards in product(range(1, 5), range(4), range(4), range(4)):
        baseline = effective_engine_cap(
            _inp(max_parallel_sidecars=budget, live_residents=live,
                 own_instances=own, cards_that_fit=cards, parallel_width=1)
        ).engines
        for width in (2, 3, 8):
            got = effective_engine_cap(
                _inp(max_parallel_sidecars=budget, live_residents=live,
                     own_instances=own, cards_that_fit=cards, parallel_width=width)
            ).engines
            assert got == baseline, (
                f"width={width} moved engines {baseline} -> {got} at "
                f"budget={budget} live={live} own={own} cards={cards}"
            )
            checked += 1
    assert checked == 4 * 4 * 4 * 4 * 3


def test_reason_text_never_names_the_retired_setting():
    """Fails if a decision reason mentions the retired setting."""
    seen = set()
    for budget, live, own, cards, unreadable in product(
        range(0, 4), range(0, 4), range(0, 3), range(-1, 3), (False, True)
    ):
        b = effective_engine_cap(
            _inp(max_parallel_sidecars=budget, live_residents=live, own_instances=own,
                 cards_that_fit=cards, manifest_unreadable=unreadable)
        )
        assert RETIRED not in b.reason
        seen.add(b.reason.split(" (")[0].split(":")[0])
    # all four decision branches were reached, so the check above covered each one
    assert len(seen) >= 4, seen


# -- windows come from parallel_width ----------------------------------------

@pytest.mark.parametrize("width", [1, 2, 3])
def test_context_windows_are_engines_times_width(width):
    """Fails if context_window_capacity ignores parallel_width."""
    inp = _inp(parallel_width=width)
    for engines in (0, 1, 2, 5):
        assert context_window_capacity(inp, engines) == engines * width


@pytest.mark.parametrize("width", [0, -5])
def test_a_width_below_one_counts_as_one_window(width):
    """Fails if a width under 1 yields zero or negative windows."""
    inp = _inp(parallel_width=width)
    assert context_window_capacity(inp, 3) == 3


def test_negative_engines_count_as_zero_windows():
    """Fails if a negative engine total yields negative windows."""
    assert context_window_capacity(_inp(parallel_width=3), -2) == 0
    assert context_window_capacity(_inp(parallel_width=1), -1) == 0


# -- the reported case -------------------------------------------------------

def test_box_budget_two_gives_a_second_engine_and_budget_one_does_not():
    """A box budget of 2 with one live engine and one fitting card gives two
    engines; the same inputs with a budget of 1 stay at one.

    ``test_second_engine_is_admitted_with_parallel_width_one`` (test_engine_budget.py)
    already pins the budget-2 half; the budget-1 contrast is added here.
    Fails if the box budget stops being the deciding limit.
    """
    two = effective_engine_cap(_inp(max_parallel_sidecars=2, parallel_width=1))
    assert two.engines == 2
    assert two.additional_engines == 1
    assert two.admits_additional_engine is True

    one = effective_engine_cap(_inp(max_parallel_sidecars=1, parallel_width=1))
    assert one.engines == 1
    assert one.additional_engines == 0
    assert one.admits_additional_engine is False


# -- ceilings and the floor ---------------------------------------------------

def test_a_box_budget_of_one_never_yields_more_than_one_engine():
    """Fails if the total can exceed the box budget while the tag is within it."""
    b = effective_engine_cap(
        _inp(max_parallel_sidecars=1, live_residents=1, own_instances=1,
             cards_that_fit=1)
    )
    assert b.engines == 1


def test_a_stale_live_count_cannot_lift_the_total_past_the_budget():
    """The total is capped by the box budget even if the live count reads low.

    Budget 2, this tag runs 2 engines but the box-wide count reads 1: the growth
    term alone would give 3. Fails if the box budget stops acting as a ceiling.
    """
    b = effective_engine_cap(
        _inp(max_parallel_sidecars=2, live_residents=1, own_instances=2,
             cards_that_fit=2)
    )
    assert b.engines == 2


def test_an_unreadable_manifest_holds_the_live_engines():
    """Fails if the unreadable-manifest branch adds or drops an engine."""
    assert effective_engine_cap(
        _inp(manifest_unreadable=True, max_parallel_sidecars=2, live_residents=2,
             own_instances=2, cards_that_fit=2)
    ).engines == 2
    # roomy budget and free cards: still no growth, the answer is the live count
    assert effective_engine_cap(
        _inp(manifest_unreadable=True, max_parallel_sidecars=4, live_residents=1,
             own_instances=1, cards_that_fit=3)
    ).engines == 1


def test_the_floor_keeps_engines_when_the_tag_is_over_the_budget():
    """Budget 1 with two live engines of this tag: the floor holds both.

    Fails if the floor at the engines already live is dropped.
    """
    for unreadable in (True, False):
        b = effective_engine_cap(
            _inp(manifest_unreadable=unreadable, max_parallel_sidecars=1,
                 live_residents=2, own_instances=2, cards_that_fit=1)
        )
        assert b.engines == 2, unreadable
