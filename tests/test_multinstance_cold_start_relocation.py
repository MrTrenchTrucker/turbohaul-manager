"""Cold start, Fast Lane relocation and the fit check of the multi-engine branch.

A one-card-per-engine tag with no live engine enters the multi-engine branch, which
remembers the session of the engine it starts. A Fast Lane relocation decides the card
(and split mode) of a cold start only: with an engine already running a relocation never
puts a second copy on the card that hosts it. The branch's fit check is asked about the
relocation card and split mode, never the picker's or the manifest's. A tag with no
placement is started on the single-engine path, and the branch counts the box budget
itself when the cap read failed.

Everything here runs real code with a recording spawn seam; the shared fakes are in
``_multiinstance_support.py``.
"""

import asyncio
import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[1]
for _p in (str(_TREE / "src"), str(_TREE / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unittest.mock import patch  # noqa: E402

from turbohaul.manager import ResidentState  # noqa: E402
from _multiinstance_support import (  # noqa: E402
    _manifest,
    _argv_value,
    _probe,
    _new_manager,
    _live_engine,
    _request,
    _wait_for_spawn,
)


# ---- a cold start ----------------------------------------------------------------

async def test_a_cold_one_card_per_engine_tag_honours_a_fast_lane_relocation(tmp_path):
    """A Fast Lane relocation chosen for the request decides the card of the first
    engine, as on the single-engine path: here card 0 is the most free, but the
    request was relocated to card 1.
    """
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    slot = _request("duo")
    slot.fastlane_relocated_main_gpu = 1
    slot.fastlane_relocated_split_mode = "none"
    p1, p2 = _probe([20000, 8000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            assert await _wait_for_spawn(spawn_calls), "no engine was spawned"
            argv = spawn_calls[0]["argv"]
            assert _argv_value(argv, "main_gpu") == "1", (
                f"the relocation target card was ignored. argv={argv}"
            )
        finally:
            await manager.shutdown()


@pytest.mark.parametrize("relocated_to,expected_card", [
    pytest.param(None, "1", id="no-relocation-the-branch-picks-the-most-free-card"),
    pytest.param(0, "0", id="relocation-target-card-wins"),
])
async def test_a_cold_one_card_per_engine_tag_enters_the_branch_and_remembers_the_session(
        tmp_path, relocated_to, expected_card):
    """A cold one-card-per-engine tag goes through the multi-engine branch: the sticky
    lookup runs, the branch picks the card (the most free one, unless a relocation
    target says otherwise), the first engine is keyed by the bare tag, and the
    session's affinity points at it, so the session's next request sticks to it."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    entered = []
    real_sticky = manager._sticky_resident_key
    real_pick = manager._pick_card_for_new_instance
    manager._sticky_resident_key = lambda *a: (entered.append("sticky"), real_sticky(*a))[1]
    manager._pick_card_for_new_instance = lambda *a: (entered.append("pick"), real_pick(*a))[1]
    slot = _request("duo")
    slot.client_meta = {"session_id": "NEW"}
    if relocated_to is not None:
        slot.fastlane_relocated_main_gpu = relocated_to
        slot.fastlane_relocated_split_mode = "none"
    p1, p2 = _probe([8000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            assert await _wait_for_spawn(spawn_calls), "no engine was spawned"
            assert entered == ["sticky", "pick"], entered
            assert [r.resident_key for r in manager._model_residents()] == ["duo"]
            assert _argv_value(spawn_calls[0]["argv"], "main_gpu") == expected_card
            assert dict(manager._session_affinity) == {("duo", "sess:NEW"): "duo"}
        finally:
            await manager.shutdown()


async def test_a_cold_placement_free_tag_is_started_on_the_legacy_path(tmp_path):
    """A placement-free manifest keeps the single-engine path for its first engine: the
    multi-engine branch is not entered (no sticky lookup, no per-tag card pick), the
    engine is keyed by the bare tag and no card is stamped on it."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=False, llama_server_flags={})
    entered = []
    real_sticky = manager._sticky_resident_key
    real_pick = manager._pick_card_for_new_instance
    manager._sticky_resident_key = lambda *a: (entered.append("sticky"), real_sticky(*a))[1]
    manager._pick_card_for_new_instance = lambda *a: (entered.append("pick"), real_pick(*a))[1]
    p1, p2 = _probe([8000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(_request("duo"))
            assert await _wait_for_spawn(spawn_calls), "no engine was spawned"
            assert entered == [], f"the multi-engine branch ran for a placement-free tag: {entered}"
            assert [r.resident_key for r in manager._model_residents()] == ["duo"]
            assert _argv_value(spawn_calls[0]["argv"], "main_gpu") is None
        finally:
            await manager.shutdown()


# ---- the box budget is counted by the branch itself --------------------------------

async def _route_cold_tag_with_transient_cap_failure(tmp_path):
    """Box budget 2 held by two idle engines of OTHER tags; a cold one-card-per-engine
    tag requests. The cached manifest read inside ``_effective_cap`` (the first one)
    raises once, so the cap falls back to its fail-safe 1 while the gate's own read of
    the same manifest succeeds. Returns (manager, unloads, spawn_calls) after routing."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    manager.runtime.queue.idle_hot_load_seconds = 600  # the new engine stays loaded
    for tag in ("duo", "otherA", "otherB"):
        _manifest(boot, tag, auto_place=True, llama_server_flags={"split_mode": "none"})
    _live_engine(manager, "otherA", gpu=0).last_active_monotonic = 1.0
    _live_engine(manager, "otherB", gpu=1).last_active_monotonic = 2.0
    unloads = []
    real_unload = manager._begin_unload_locked
    manager._begin_unload_locked = lambda r, *a, **k: (unloads.append(r.resident_key),
                                                       real_unload(r, *a, **k))[1]
    import turbohaul.manager as manager_module
    real_read = manager_module.read_manifest_cached
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("transient: manifest being rewritten")
        return real_read(*a, **k)

    p1, p2 = _probe([20000, 20000])
    with p1, p2, patch("turbohaul.manager.read_manifest_cached", flaky):
        try:
            await manager._route_or_reserve(_request("duo"))
            await asyncio.sleep(0.3)
            live = sorted(r.resident_key for r in manager._model_residents()
                          if r.state is not ResidentState.DEAD)
            unloaded = list(unloads)  # before shutdown unloads the rest
        finally:
            await manager.shutdown()
    return live, unloaded, spawn_calls


async def test_a_cold_tag_makes_room_when_the_cap_read_failed_and_the_box_is_full(tmp_path):
    """The multi-engine branch counts the box budget itself. With the cap's manifest read
    failing once (cap falls back to 1) while the box is full, the cold tag must not
    start an engine on top of the full box: the single-engine path's make-room runs, one
    idle engine (the least recently used) is unloaded and the box ends at its budget."""
    live, unloads, spawn_calls = await _route_cold_tag_with_transient_cap_failure(tmp_path)
    assert unloads == ["otherA"], f"expected exactly the LRU idle engine to be unloaded: {unloads}"
    assert live == ["duo", "otherB"], f"the box must end at its budget of 2: {live}"
    assert len(spawn_calls) == 1


# ---- relocation: cold start only; the fit check uses the override --------------------

@pytest.mark.parametrize("split", ["none", "layer"])
async def test_a_relocation_never_puts_a_second_engine_on_the_card_that_hosts_the_first(
        tmp_path, split):
    """One engine of the tag is live on card 0, budget 2, both cards roomy, and the
    request carries a relocation target of card 0. The relocation decides the card of
    a cold start only: a tag that already has an engine keeps the picker's choice, so
    the second engine goes to the unused card 1 and never onto card 0."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    _live_engine(manager, "duo", gpu=0)
    slot = _request("duo")
    slot.client_meta = {"session_id": "NEW"}
    slot.fastlane_relocated_main_gpu = 0
    slot.fastlane_relocated_split_mode = split
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            assert await _wait_for_spawn(spawn_calls), "no second engine was spawned"
            assert _argv_value(spawn_calls[0]["argv"], "main_gpu") == "1", spawn_calls[0]["argv"]
        finally:
            await manager.shutdown()


async def test_the_fit_check_uses_the_relocation_card(tmp_path):
    """Cold one-card-per-engine tag, relocation target card 0, but card 0 is too small
    (2000 MiB free for a ~3000 MiB model) while card 1 is roomy. Nothing may be
    reserved or spawned on the relocation card: the branch's fit check fails and the
    request goes to the single-engine path, which defers it (waiting for room), exactly
    as it does for the same slot on the unmodified code."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    reserved = []
    real_reserve = manager._reserve_and_start_locked
    manager._reserve_and_start_locked = lambda *a, **k: (reserved.append(k), real_reserve(*a, **k))[1]
    deferred = []
    real_defer = manager._defer_unroutable
    manager._defer_unroutable = lambda s, **k: (deferred.append(k), real_defer(s, **k))[1]
    slot = _request("duo")
    slot.fastlane_relocated_main_gpu = 0
    slot.fastlane_relocated_split_mode = "none"
    p1, p2 = _probe([2000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            await asyncio.sleep(0.2)
            assert spawn_calls == []
            assert reserved == [], "an engine was reserved on the relocation card"
            assert deferred, "the request must be deferred, waiting for room"
            assert not slot.completion_future.done()
        finally:
            await manager.shutdown()


async def test_the_relocated_split_mode_reaches_the_fit_check_and_the_spawn(tmp_path):
    """The relocation carries a split mode as well as a card. A card-spanning split
    ("layer") is checked against the free VRAM of all cards together, so here, with card
    0 too small on its own, the request fits and the engine is started with that split
    mode on the relocation card, as the single-engine path would."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    slot = _request("duo")
    slot.fastlane_relocated_main_gpu = 0
    slot.fastlane_relocated_split_mode = "layer"
    p1, p2 = _probe([2000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            assert await _wait_for_spawn(spawn_calls), "no engine was spawned"
            argv = spawn_calls[0]["argv"]
            assert _argv_value(argv, "split_mode") == "layer", argv
            assert _argv_value(argv, "main_gpu") == "0", argv
            (engine,) = manager._model_residents()
            assert (engine.main_gpu, engine.split_mode) == (0, "layer"), (
                "the reservation must carry the relocated card and split mode"
            )
        finally:
            await manager.shutdown()


async def test_the_branch_fit_check_carries_the_relocated_split_mode(tmp_path):
    """Card 0 alone is too small, so the request is only admitted on the relocated
    "layer" split (all cards together). The fit check of the multi-engine branch must
    be asked about exactly that card and split mode, and the branch itself (not the
    single-engine path it falls back to) must start the engine and remember the
    session. A fit check that used the manifest's "none" would refuse, abandon the
    branch, and leave the same engine on the same card, but no session remembered."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    admits = []
    real_admits = manager._vram_admits_locked

    def spy(*args, **kwargs):
        result = real_admits(*args, **kwargs)
        admits.append((args, kwargs, result))
        return result

    manager._vram_admits_locked = spy
    slot = _request("duo")
    slot.client_meta = {"session_id": "alpha"}
    slot.fastlane_relocated_main_gpu = 0
    slot.fastlane_relocated_split_mode = "layer"
    p1, p2 = _probe([2000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            assert await _wait_for_spawn(spawn_calls), "no engine was spawned"
            assert admits, "the fit check was never asked"
            args, _kwargs, result = admits[0]
            assert (args[2], args[3]) == (0, "layer"), (
                "the branch fit check was not asked about the relocation card and "
                "split mode: %r" % (args,)
            )
            assert result is True, "the relocated split must fit across both cards"
            assert dict(manager._session_affinity) == {("duo", "sess:alpha"): "duo"}, (
                "the branch must remember the session of the engine it started"
            )
        finally:
            await manager.shutdown()


async def test_the_relocated_split_mode_decides_whether_a_neighbour_may_share(tmp_path):
    """A card-spanning split ("layer") cannot share the box with another model's
    engine, a one-card split ("none") can. The tag is one-card-per-engine, but the
    relocation asks for "layer" while another model's engine is already loaded, so the
    fit check must refuse: nothing may be started for the request, which waits for
    room. If the check ignored the relocated split mode and used the manifest's "none",
    the engine would be started here."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=3)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    _live_engine(manager, "neighbour", gpu=1)
    reserved = []
    real_reserve = manager._reserve_and_start_locked
    manager._reserve_and_start_locked = lambda *a, **k: (reserved.append(k), real_reserve(*a, **k))[1]
    slot = _request("duo")
    slot.fastlane_relocated_main_gpu = 0
    slot.fastlane_relocated_split_mode = "layer"
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            await asyncio.sleep(0.2)
            assert spawn_calls == []
            assert reserved == [], "an engine was reserved with a split that cannot share"
        finally:
            await manager.shutdown()
