"""Engine budget: the engine count comes only from the box budget and the cards.

A model's manifest carries ``llama_server_flags.parallel`` (context windows per
engine). With the default of 1 and ``queue.max_parallel_sidecars = 2``, a model must
still get two engines, one context window each: the windows setting never lowers or
raises the engine count.

These tests call the real decision function with plain numbers.
"""

import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.engine_budget import (  # noqa: E402
    EngineBudgetInputs,
    context_window_capacity,
    effective_engine_cap,
)


def _inp(**kw) -> EngineBudgetInputs:
    """One tag, one engine already loaded, one more card free, budget of 2."""
    base = dict(
        max_parallel_sidecars=2,
        live_residents=1,
        own_instances=1,
        cards_that_fit=1,
        parallel_width=1,  # the default: one window per engine
        manifest_unreadable=False,
    )
    base.update(kw)
    return EngineBudgetInputs(**base)


# -- the reported case ------------------------------------------------------

def test_second_engine_is_admitted_with_parallel_width_one():
    """parallel_width=1 must not mean "one engine".

    `engines` is the TOTAL count: a tag already holding one engine is granted one
    more here, which is the second engine the box budget of 2 allows.
    """
    b = effective_engine_cap(_inp())
    # TOTAL capacity: the tag already runs 1, one more is admissible -> total 2.
    assert b.engines == 2, (
        "with 1 free global slot and 1 free card, the tag's TOTAL capacity is 2"
    )
    assert b.additional_engines == 1
    assert b.admits_additional_engine


def test_parallel_width_does_not_reduce_engine_count():
    """The conflation itself, stated as an invariant over a sweep of widths."""
    for width in (1, 2, 3, 4, 8):
        wide = effective_engine_cap(_inp(parallel_width=width))
        assert wide.engines == 2, f"width={width} changed the engine count"


def test_two_free_cards_admits_two_more_engines():
    b = effective_engine_cap(_inp(cards_that_fit=2, live_residents=0, own_instances=0))
    assert b.engines == 2   # no engines of its own yet, two free


# -- the ceilings that must hold --------------------------------------------

def test_global_budget_still_bounds_the_result():
    """A wider manifest cannot lift the box's hard engine cap."""
    b = effective_engine_cap(
        _inp(max_parallel_sidecars=1, live_residents=1, cards_that_fit=2, parallel_width=8)
    )
    # The tag already runs 1 and no further engine is admissible, so TOTAL stays 1:
    # it must not shrink to 0 (the caller would drop its live engine) and must not
    # lift to 2 (which would breach max_parallel_sidecars).
    # max_parallel_sidecars=1 is a HARD ceiling on the total: this tag already
    # runs 1, so it can admit nothing and must not report 2.
    assert b.engines == 1
    assert b.additional_engines == 0
    assert "global budget" in b.reason or "no headroom" in b.reason


def test_own_engines_do_not_count_against_own_allowance():
    """A tag may hold every global slot not held by ANOTHER tag.

    Budget 3, two of this tag's own engines live, no other tag resident: the tag
    may still take the third.
    """
    b = effective_engine_cap(
        _inp(max_parallel_sidecars=3, live_residents=2, own_instances=2, cards_that_fit=1)
    )
    # Two of the three slots are already this tag's and the third is free, so the
    # tag may grow to 3 -- its own engines never count against its own allowance.
    assert b.engines == 3
    assert b.additional_engines == 1


def test_no_fitting_card_degrades_to_single_pin():
    b = effective_engine_cap(_inp(cards_that_fit=0))
    assert b.engines == 1   # cannot admit more, but must not shrink below live
    assert "no card fits" in b.reason


def test_manifest_unreadable_fails_safe_to_one():
    """An unreadable manifest holds the live engines (here one)."""
    b = effective_engine_cap(_inp(manifest_unreadable=True))
    assert b.engines == 1
    assert "fail-safe" in b.reason


# -- two sidecars, one window each, made explicit ---------------------------

def test_two_sidecars_one_window_each_is_the_intended_capacity():
    """max_parallel_sidecars=2 + parallel_width=1 => 2 windows, not 1."""
    inp = _inp(parallel_width=1)
    total_engines = effective_engine_cap(inp).engines
    assert total_engines == 2
    assert context_window_capacity(inp, total_engines) == 2
    # and explicitly: 2 sidecars, ONE window each -- not 1 sidecar with 1 window
    assert context_window_capacity(inp, total_engines) == total_engines * 1


def test_context_window_capacity_scales_with_width():
    inp = _inp(parallel_width=4)
    assert context_window_capacity(inp, 2) == 8


# -- never raises, always explains ------------------------------------------

@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(live_residents=99),
        dict(parallel_width=0),
        dict(parallel_width=-5),
        dict(max_parallel_sidecars=0),
        dict(cards_that_fit=-1),
        dict(manifest_unreadable=True),
    ],
)
def test_decision_never_raises_and_always_explains(kw):
    b = effective_engine_cap(_inp(**kw))
    # never below the engines already live, never negative
    assert isinstance(b.engines, int) and b.engines >= b.own_instances >= 0
    assert b.reason and b.reason.strip()
