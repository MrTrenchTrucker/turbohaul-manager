"""The engine budget and the placement path are different questions.

How many engines one model may run is a BUDGET question: the box-wide
``queue.max_parallel_sidecars`` setting and the cards the model fits on, nothing else.
The manifest has no per-model engine limit.

Whether the routing path may place a SECOND copy of a model is a PLACEMENT question,
and only one kind of manifest can answer yes: one that lets the manager pick the card
(``auto_place: true``) and keeps each engine on a single card (``split_mode: none``).
``_wants_one_card_per_engine`` is that predicate.

A routing gate that reuses the budget number (``eff_cap > 1``) sends every placement-free
or card-spanning manifest down the multi-engine path as soon as a box slot is free, and
that path stamps ``--main-gpu`` on the engine: a single-engine manifest gains a card it
never asked for (``test_ARM4_CONTROL_pinned_not_relocated_argv_byte_identical`` is the
over-broadness control for that). The gate must key on placement intent. Both properties
hold at once, and they are independent:

  1. the budget admits a second engine for the model;
  2. only a one-card-per-engine manifest is placed as more than one engine.

This file covers the predicate and the routing gate: which manifests are placed as more
than one engine, which keep the single-engine path, and how an unreadable manifest and
the other two ways into the multi-engine branch behave. The cold-start and relocation
cases and the refused, failed-to-load and failed-to-start engine cases are in the
sibling files ``test_multinstance_cold_start_relocation.py``,
``test_multinstance_reroute_refused_spawn.py`` and
``test_multinstance_reroute_start_failures.py``. Their shared fakes and helpers are in
``_multiinstance_support.py``.

Everything here runs real code: manifests are read through the real cached accessor,
the decision comes from the real budget function, and the routing tests drive the real
``_route_or_reserve`` with a recording spawn seam. Nothing asserts on source text.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[1]
for _p in (str(_TREE / "src"), str(_TREE / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from turbohaul.engine_budget import (  # noqa: E402
    EngineBudgetInputs,
    effective_engine_cap,
)
from turbohaul.manager import (  # noqa: E402
    ResidentState,
    TurbohaulManager,
    read_manifest_cached,
    _wants_one_card_per_engine,
)
from _multiinstance_support import (  # noqa: E402
    _manifest,
    _argv_value,
    _probe,
    _new_manager,
    _live_engine,
    _request,
    _wait_for_spawn,
)


def _wants(mgr, tag: str) -> bool:
    """The predicate, applied to the manifest read through the real cached accessor."""
    return _wants_one_card_per_engine(read_manifest_cached(mgr.boot.storage.manifests_path, tag))


@pytest.fixture()
def mgr():
    """A manager over a single placement-free manifest, box budget 2."""
    from test_placement_multiinstance import _boot_runtime, _mocks

    with tempfile.TemporaryDirectory() as td:
        boot, runtime = _boot_runtime(Path(td), max_parallel_sidecars=2)
        _manifest(boot, "solo")
        manager = TurbohaulManager(boot, runtime, **_mocks())
        manager.runtime.queue.safety_enabled = False
        yield manager


def _second_engine_budget():
    """Budget 2 with one engine live and a second card free: a second engine fits."""
    return effective_engine_cap(EngineBudgetInputs(
        max_parallel_sidecars=2, live_residents=1, own_instances=1,
        cards_that_fit=1, parallel_width=1,
    ))


# ---- the predicate -------------------------------------------------------------

def test_a_placement_free_manifest_never_wants_a_card_per_engine(mgr):
    """The over-broadness control, at the decision.

    The budget may say a second engine is affordable. A manifest that declares no
    placement must still be placed as the single engine it always was.
    """
    budget = _second_engine_budget()
    assert budget.engines == 2, "precondition: the budget DOES admit a second engine"
    assert _wants(mgr, "solo") is False, (
        "a placement-free manifest must not take the multi-engine placement path just "
        "because the box budget could admit a second engine"
    )


def test_placement_free_stays_false_even_with_split_mode_none(mgr):
    """auto_place is half the predicate: split_mode none alone is not enough.

    Without auto_place the manager does not choose the card, so a second copy has no
    card to be placed on.
    """
    _manifest(mgr.boot, "pinned", auto_place=False,
              llama_server_flags={"split_mode": "none", "main_gpu": 0})
    assert _wants(mgr, "pinned") is False


def test_auto_place_with_split_mode_none_wants_a_card_per_engine(mgr):
    _manifest(mgr.boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    assert _wants(mgr, "duo") is True, (
        "a one-card-per-engine manifest genuinely wants the multi-engine path; the gate "
        "must not be narrowed until such tags stop getting a second engine"
    )


def test_auto_place_with_no_split_mode_is_not_one_card_per_engine(mgr):
    """An ABSENT split_mode is not 'none': the engine may span cards."""
    _manifest(mgr.boot, "implicit", auto_place=True, llama_server_flags={})
    assert _wants(mgr, "implicit") is False


@pytest.mark.parametrize("split_mode", ["layer", "row", "tensor"])
def test_auto_place_with_a_card_spanning_split_is_not_one_card_per_engine(mgr, split_mode):
    _manifest(mgr.boot, "spanning", auto_place=True,
              llama_server_flags={"split_mode": split_mode})
    assert _wants(mgr, "spanning") is False


def test_the_budget_still_admits_a_second_engine_for_a_one_card_per_engine_manifest(mgr):
    """Property 1, with the manifest in hand: two engines for one model, one window each.

    If the gate is ever narrowed into taking this away, this and the routing test below
    catch it.
    """
    _manifest(mgr.boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    assert _wants(mgr, "duo") is True
    budget = _second_engine_budget()
    assert (budget.engines, budget.additional_engines) == (2, 1)


# ---- the real routing path -----------------------------------------------------


async def test_a_one_card_per_engine_tag_with_one_live_engine_gets_a_second_engine(tmp_path):
    """The budget admits a second engine AND the manifest can host it: it spawns.

    Box budget 2, two cards that fit a copy, one engine already on card 0. The new
    engine goes on the card that does not already host one of the tag's engines.
    """
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    first = _live_engine(manager, "duo", gpu=0)
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            assert manager._effective_cap("duo") == 2, "precondition: budget admits it"
            await manager._route_or_reserve(_request("duo"))
            assert await _wait_for_spawn(spawn_calls), "no second engine was spawned"
            assert len(manager._instances_for("duo")) == 2
            assert spawn_calls[0]["model_tag"] == "duo"
            assert _argv_value(spawn_calls[0]["argv"], "main_gpu") == "1", (
                f"the second engine must land on the free card. argv={spawn_calls[0]['argv']}"
            )
            assert first.inbox.empty(), "the request must not have been routed to the old engine"
        finally:
            await manager.shutdown()


async def test_a_retired_key_left_in_a_one_card_per_engine_manifest_does_not_block_the_second_engine(
        tmp_path):
    """A saved manifest that still says ``max_instances: 1`` loads with the key dropped
    and the model still gets its second engine: the old per-model value caps nothing."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, max_instances=1,
              llama_server_flags={"split_mode": "none"})
    _live_engine(manager, "duo", gpu=0)
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(_request("duo"))
            assert await _wait_for_spawn(spawn_calls), "no second engine was spawned"
            assert len(manager._instances_for("duo")) == 2
        finally:
            await manager.shutdown()


@pytest.mark.parametrize("shape", [
    pytest.param({"auto_place": False, "llama_server_flags": {}}, id="placement-free"),
    pytest.param({"auto_place": False,
                  "llama_server_flags": {"split_mode": "none", "main_gpu": 0}},
                 id="pinned-card"),
    pytest.param({"auto_place": True, "llama_server_flags": {}}, id="auto-place-no-split"),
    pytest.param({"auto_place": True, "llama_server_flags": {"split_mode": "layer"}},
                 id="auto-place-layer-split"),
])
async def test_a_tag_that_is_not_one_card_per_engine_keeps_its_single_engine(tmp_path, shape):
    """The over-broadness control, through the real routing path.

    Box budget 2, two cards that fit a copy, one engine already live: the budget DOES
    admit a second engine, but the manifest cannot host one. The request is routed to
    the one engine (the legacy HIT): nothing is spawned, so no second engine and no
    ``--main-gpu`` override appears.
    """
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "solo", **shape)
    engine = _live_engine(manager, "solo", gpu=0)
    slot = _request("solo")
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            assert (len(manager._model_residents())
                    < manager.runtime.queue.max_parallel_sidecars), (
                "precondition: the box budget has a free slot for a second engine"
            )
            assert manager._effective_cap("solo") == 1, (
                "a tag that is not one-card-per-engine is held to its one live engine "
                "even though the box budget has a free slot"
            )
            await manager._route_or_reserve(slot)
            await asyncio.sleep(0.1)
            assert spawn_calls == [], f"a second engine was spawned: {spawn_calls}"
            assert len(manager._instances_for("solo")) == 1
            assert slot.fastlane_relocated_main_gpu is None
            assert engine.inbox.get_nowait() is slot, "the request must go to the one engine"
            assert engine.state is ResidentState.ACTIVE
        finally:
            await manager.shutdown()


async def test_a_full_box_budget_keeps_a_one_card_per_engine_tag_to_its_one_engine(tmp_path):
    """The budget is still the limit: with the box budget spent, no second engine."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=1)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0)
    slot = _request("duo")
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            await asyncio.sleep(0.1)
            assert spawn_calls == []
            assert len(manager._instances_for("duo")) == 1
            assert engine.inbox.get_nowait() is slot
        finally:
            await manager.shutdown()


async def test_a_single_card_keeps_a_one_card_per_engine_tag_to_its_one_engine(tmp_path):
    """The cards are still a limit: one card, already hosting the tag, no second engine."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0)
    slot = _request("duo")
    p1, p2 = _probe([20000])
    with p1, p2:
        try:
            assert manager._effective_cap("duo") == 1
            await manager._route_or_reserve(slot)
            await asyncio.sleep(0.1)
            assert spawn_calls == []
            assert engine.inbox.get_nowait() is slot
        finally:
            await manager.shutdown()


# ---- an unreadable manifest ----------------------------------------------------

class _BranchEntered(Exception):
    """Raised by the spy at the first step of the multi-engine branch."""


async def _route_with_stubbed_cap(tmp_path, cap, *, manifest_body):
    """Route one request for a tag with one live engine, an unreadable manifest and the
    engine count stubbed to ``cap``. Returns True if the multi-engine branch was entered.

    The branch's first step is the sticky lookup; the spy records the entry and stops
    there so nothing after it runs against a manifest that cannot be read. A real
    manifest cannot reach this state by itself (the cap and the gate read the same
    manifest), so the cap is stubbed.
    """
    manager, boot, _spawn_calls = _new_manager(tmp_path, budget=2)
    (boot.storage.manifests_path / "ghost.yaml").write_text(manifest_body)
    engine = _live_engine(manager, "ghost", gpu=0)
    manager._effective_cap = lambda tag: cap

    def _entered(slot, tag):
        raise _BranchEntered(tag)

    manager._sticky_resident_key = _entered
    slot = _request("ghost")
    try:
        await manager._route_or_reserve(slot)
    except _BranchEntered:
        return True
    else:
        assert engine.inbox.get_nowait() is slot, "the legacy path must serve it"
        return False
    finally:
        await manager.shutdown()


async def test_an_unreadable_manifest_enters_the_multi_engine_branch_when_the_cap_allows(
        tmp_path):
    """An unreadable manifest must not change routing: the gate falls back to the engine
    count, so a cap above 1 still enters the branch."""
    assert await _route_with_stubbed_cap(tmp_path, 2, manifest_body="{ not: valid: yaml") is True


async def test_an_unreadable_manifest_keeps_the_single_engine_path_when_the_cap_is_one(
        tmp_path):
    """The fallback is `cap > 1`, not 'always enter': a cap of 1 stays on the legacy path."""
    assert await _route_with_stubbed_cap(tmp_path, 1, manifest_body="{ not: valid: yaml") is False


# ---- the other two ways into the multi-engine branch ----------------------------

async def _gate_enters_branch(tmp_path, engine_keys):
    """Route one request for a PLACEMENT-FREE tag whose live engines have the given
    registry keys. Returns True if the multi-engine branch was entered (the sticky
    lookup, its first step, is where the spy stops)."""
    manager, boot, _spawn_calls = _new_manager(tmp_path, budget=3)
    _manifest(boot, "free", auto_place=False)
    for gpu, key in enumerate(engine_keys):
        _live_engine(manager, "free", gpu=gpu, key=key)

    def _entered(slot, tag):
        raise _BranchEntered(tag)

    manager._sticky_resident_key = _entered
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(_request("free"))
        except _BranchEntered:
            return True
        finally:
            await manager.shutdown()
    return False


async def test_a_placement_free_tag_with_two_live_engines_enters_the_branch(tmp_path):
    """A tag that already runs more than one engine is routed across them whatever its
    manifest says: the legacy path only knows the engine keyed by the tag itself."""
    assert await _gate_enters_branch(tmp_path, ["free", "free#1"]) is True


async def test_a_placement_free_tag_whose_only_engine_has_a_suffixed_key_enters_the_branch(
        tmp_path):
    """A survivor keyed ``tag#1`` (the first engine went away) is invisible to the
    legacy path, which looks the tag up by its bare name."""
    assert await _gate_enters_branch(tmp_path, ["free#1"]) is True


async def test_a_placement_free_tag_with_one_plain_engine_does_not_enter_the_branch(tmp_path):
    """Control for the two tests above: the same setup with the engine under the bare
    tag stays on the legacy path."""
    assert await _gate_enters_branch(tmp_path, ["free"]) is False
