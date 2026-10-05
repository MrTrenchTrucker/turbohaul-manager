"""A second engine that a host or VRAM check refuses falls back to the running one.

When a safety gate refuses the extra engine of a tag that already has a live engine,
the request is handed to that engine instead of failing; a refused first engine still
fails the request, and a refusal at reserve time requeues it. The hand-off helper
(``_fail_or_reroute_refused_spawn``) is also called directly: it parks the request's
Fast Lane claim on the live engine, records the wait, skips engines in a reclaim barrier wait or without an
inbox, marks the request as rerouted as soon as it is enqueued, and takes the registry
lock before it reads or changes anything.

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

from turbohaul.fastlane import FastLaneMatch  # noqa: E402
from turbohaul.manager import (  # noqa: E402
    Resident,
    ResidentState,
)
from turbohaul.safety import GateResult  # noqa: E402
from _multiinstance_support import (  # noqa: E402
    _manifest,
    _probe,
    _new_manager,
    _live_engine,
    _request,
    _track_started,
    _assert_driver_ended_cleanly,
)


# ---- a refused second engine falls back to the running one -----------------------

CPU_BUSY = GateResult("cpu_busy", False, "cpu 95% > max 80%", blocked_on="host")
VRAM_SHORT = GateResult("vram", False, "need 3000 MiB, card short", blocked_on="vram")

REFUSALS = [
    pytest.param("host", [CPU_BUSY], id="host-gate"),
    pytest.param("vram", [VRAM_SHORT], id="vram-gate-no-reclaim-in-flight"),
    pytest.param("barrier", [VRAM_SHORT], id="vram-gate-reclaim-barrier-runs-out"),
]


async def _route_with_refusing_gate(tmp_path, kind, gates, *, live):
    """Route one new-session request for a one-card-per-engine tag, budget 2, two free
    cards, with the host safety gates refusing every spawn. ``live`` puts one idle
    engine of the tag on card 0 first. Returns (slot, engine, spawn_calls, manager)
    once the request has either been handed to the live engine or failed."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    manager.runtime.queue.safety_enabled = True
    manager.runtime.queue.spawn_reclaim_wait_max_s = 0.3 if kind == "barrier" else 0.0
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0) if live else None
    _track_started(manager)
    reclaim = None
    if kind == "barrier":
        # A teardown that never finishes: positive evidence for the reclaim barrier,
        # which then waits out its (short) budget and gives up.
        reclaim = asyncio.get_running_loop().create_future()
        manager._card_release_tasks = {0: {reclaim}, 1: {reclaim}}
    slot = _request("duo")
    slot.client_meta = {"session_id": "NEW"}
    p1, p2 = _probe([20000, 20000])
    with p1, p2, patch("turbohaul.manager.all_safety_gates", return_value=list(gates)):
        try:
            await manager._route_or_reserve(slot)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5.0
            while loop.time() < deadline:
                if slot.completion_future.done() or (engine and not engine.inbox.empty()):
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)  # let the refused engine be reaped
        finally:
            if reclaim is not None:
                reclaim.cancel()
            await manager.shutdown()
    return slot, engine, spawn_calls, manager


@pytest.mark.parametrize("kind,gates", REFUSALS)
async def test_a_refused_second_engine_routes_the_request_to_the_live_engine(
        tmp_path, kind, gates, caplog):
    """One live idle engine, budget 2, a free card, a new session: the branch starts a
    second engine, and a host safety gate refuses it. The request must be served by
    the engine that is already running, not failed."""
    slot, engine, spawn_calls, manager = await _route_with_refusing_gate(
        tmp_path, kind, gates, live=True)
    assert not slot.completion_future.done(), (
        f"the request was failed instead of routed: {slot.completion_future.exception()!r}"
    )
    assert engine.inbox.get_nowait() is slot, "the request must be on the live engine"
    assert engine.state is ResidentState.ACTIVE
    # a parked request keeps its claim until its own turn starts, and the engine is
    # stamped with the client of the turn running now, so the hand-off leaves the stamp
    # alone. The engine here has no turn loop, so only the unchanged part is checked
    # (a fresh engine starts with no stamp).
    assert engine.rank_client_meta is None, (
        "the hand-off must not stamp the engine with the parked request's client; the "
        "stamp is written when the request's own turn starts"
    )
    assert spawn_calls == [], "the refused engine must not have been spawned"
    assert [r.resident_key for r in manager._model_residents()] == ["duo"], (
        "the refused engine must be gone"
    )
    _assert_driver_ended_cleanly(manager, caplog)


@pytest.mark.parametrize("kind,gates", REFUSALS)
async def test_a_refused_first_engine_still_fails_the_request(tmp_path, kind, gates):
    """Control: with no live engine there is nothing to fall back to, so a refused
    cold start fails the request exactly as it always has."""
    slot, _engine, spawn_calls, _manager = await _route_with_refusing_gate(
        tmp_path, kind, gates, live=False)
    assert slot.completion_future.done()
    exc = slot.completion_future.exception()
    assert type(exc) is RuntimeError, repr(exc)
    assert str(exc).startswith("safety gates refused spawn: "), str(exc)
    assert spawn_calls == []


async def test_a_second_engine_refused_at_reserve_time_is_requeued_not_failed(tmp_path):
    """The other way a second engine can be refused: the cross-resident VRAM check inside
    the reservation says no although the check just before it said yes (the card
    filled in between). That refusal never fails the request. The slot is requeued, no
    engine is spawned and the request is not handed to the live engine (it is retried
    through the whole routing path on its next turn).

    The two checks are the same function; the stub admits the first call and refuses the
    second, which is the race in miniature.
    """
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0)
    answers = iter([True, False])
    manager._vram_admits_locked = lambda *a, **k: next(answers)
    deferred = []
    real_defer = manager._defer_unroutable
    manager._defer_unroutable = lambda s, **k: (deferred.append(k), real_defer(s, **k))[1]
    slot = _request("duo")
    slot.client_meta = {"session_id": "NEW"}
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            await asyncio.sleep(0.1)
            assert deferred == [{"evict_pending": True}]
            assert not slot.completion_future.done(), "the request must not be failed"
            assert engine.inbox.empty(), "the request must not be on the live engine"
            assert spawn_calls == []
        finally:
            await manager.shutdown()


# ---- the hand-off helper, called directly ---------------------------------------

async def _refused_spawn_setup(tmp_path, *, engine_inbox=True, barriered=False):
    """A manager with one live engine of "duo" (idle, card 0) and a second engine
    ``refused`` (reserved, loading) whose spawn a safety gate has just refused for a
    Fast Lane request ``slot``. The slot's claim is registered up front and the
    handed-off registry is empty, so only the helper's own calls can change either."""
    manager, boot, _spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0)
    if not engine_inbox:
        engine.inbox = None
    engine.spawn_barrier_active = barriered
    refused = Resident(
        model_tag="duo", resident_key="duo#1", state=ResidentState.RESERVED_LOADING,
        main_gpu=1, split_mode="none", inbox=asyncio.Queue(),
    )
    manager._residents["duo#1"] = refused
    slot = _request("duo")
    slot.client_meta = {"session_id": "NEW", "ip": "10.0.0.7"}
    slot.fastlane = FastLaneMatch(
        rule_index=0, raw_address="10.0.0.7", label="", effective_tag="main", rank=1,
    )
    manager._register_fastlane_claim_locked(slot, "make_room_starved_vram")
    assert manager._fastlane_claims, "setup: the claim must be registered"
    assert all("parked_on" not in c for c in manager._fastlane_claims.values()), (
        "setup: the claim must not be marked as parked yet"
    )
    assert not manager._handed_off_waiting, "setup: nothing handed off yet"
    return manager, engine, refused, slot


async def test_the_handoff_keeps_the_requests_fast_lane_claim_parked_on_the_engine_and_records_the_wait(tmp_path):
    """Once the request sits on the live engine its claim is kept, marked as parked on
    that engine (a parked request keeps its claim until its own turn starts), and
    /status must keep showing it until it is served (it is recorded as handed off).
    The registry starts without the parked mark and the handed-off record, so only
    the helper can have done them."""
    manager, engine, refused, slot = await _refused_spawn_setup(tmp_path)
    try:
        await manager._fail_or_reroute_refused_spawn(refused, slot, RuntimeError("refused"))
        assert engine.inbox.get_nowait() is slot
        key = manager._fastlane_claim_key(slot)
        assert key in manager._fastlane_claims, "the claim must still be registered"
        claim = manager._fastlane_claims[key]
        assert claim["slot"] is slot
        assert claim.get("parked_on") == engine.resident_key, (
            "the claim must be marked as parked on the live engine"
        )
        assert slot.slot_id in manager._handed_off_waiting
        assert refused.rerouted_slot is slot
    finally:
        await manager.shutdown()


async def test_an_engine_in_a_reclaim_barrier_wait_is_not_a_fallback_target(tmp_path):
    """The only other engine is held by a reclaim barrier wait: nothing may be routed
    onto it (a barriered engine must not receive new arrivals), so the request fails
    with the original refusal error."""
    manager, engine, refused, slot = await _refused_spawn_setup(tmp_path, barriered=True)
    original = RuntimeError("safety gates refused spawn: cpu_busy: cpu 95% > max 80%")
    try:
        await manager._fail_or_reroute_refused_spawn(refused, slot, original)
        assert slot.completion_future.exception() is original
        assert engine.inbox.empty()
        assert refused.rerouted_slot is None
    finally:
        await manager.shutdown()


async def test_an_engine_without_an_inbox_is_not_a_fallback_target(tmp_path):
    """An engine that cannot take the request is skipped, so the request fails with the
    original error instead of being silently lost."""
    manager, engine, refused, slot = await _refused_spawn_setup(tmp_path, engine_inbox=False)
    original = RuntimeError("safety gates refused spawn: cpu_busy: cpu 95% > max 80%")
    try:
        await manager._fail_or_reroute_refused_spawn(refused, slot, original)
        done, _pending = await asyncio.wait({slot.completion_future}, timeout=1.0)
        assert done, "the request was lost: its future never resolved"
        assert slot.completion_future.exception() is original
        assert refused.rerouted_slot is None
    finally:
        await manager.shutdown()


async def test_the_slot_is_marked_rerouted_as_soon_as_it_is_enqueued(tmp_path):
    """If a step after the enqueue raises, the request is already on the live engine,
    so the refused engine's driver must already know not to fail it as an orphan."""
    manager, engine, refused, slot = await _refused_spawn_setup(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("late failure")

    manager._park_fastlane_claim_locked = boom
    try:
        with pytest.raises(RuntimeError, match="late failure"):
            await manager._fail_or_reroute_refused_spawn(refused, slot, RuntimeError("refused"))
        assert engine.inbox.get_nowait() is slot
        assert refused.rerouted_slot is slot
    finally:
        await manager.shutdown()


async def _turn_the_loop(times=10):
    """Let every other ready task on the loop run a few times."""
    for _ in range(times):
        await asyncio.sleep(0)


async def test_the_handoff_waits_for_the_registry_lock(tmp_path):
    """The hand-off reads and changes the registry, so it must run under the registry
    lock like every other registry change. While another holder has the lock nothing
    may happen: the request is not queued on the live engine, the refused engine is
    not marked, the request is not failed and the helper does not return. Once the
    lock is free the hand-off completes."""
    manager, engine, refused, slot = await _refused_spawn_setup(tmp_path)
    try:
        await manager._registry_lock.acquire()
        task = asyncio.ensure_future(
            manager._fail_or_reroute_refused_spawn(refused, slot, RuntimeError("refused"))
        )
        try:
            await _turn_the_loop()
            assert not task.done(), "the hand-off ran without taking the registry lock"
            assert engine.inbox.empty()
            assert refused.rerouted_slot is None
            assert not slot.completion_future.done()
        finally:
            manager._registry_lock.release()
        assert await asyncio.wait_for(task, timeout=2.0) is True
        assert engine.inbox.get_nowait() is slot
        assert refused.rerouted_slot is slot
    finally:
        await manager.shutdown()


async def test_the_failure_of_a_request_with_no_fallback_waits_for_the_registry_lock(tmp_path):
    """Same, for a request that cannot be handed over (the only other engine is held by
    a reclaim barrier wait): while the lock is held the request is not failed, after it
    is released the request fails with the original error and the helper returns False."""
    manager, engine, refused, slot = await _refused_spawn_setup(tmp_path, barriered=True)
    original = RuntimeError("refused")
    try:
        await manager._registry_lock.acquire()
        task = asyncio.ensure_future(
            manager._fail_or_reroute_refused_spawn(refused, slot, original)
        )
        try:
            await _turn_the_loop()
            assert not task.done(), "the helper ran without taking the registry lock"
            assert not slot.completion_future.done()
        finally:
            manager._registry_lock.release()
        assert await asyncio.wait_for(task, timeout=2.0) is False
        assert slot.completion_future.exception() is original
        assert engine.inbox.empty()
        assert refused.rerouted_slot is None
    finally:
        await manager.shutdown()
