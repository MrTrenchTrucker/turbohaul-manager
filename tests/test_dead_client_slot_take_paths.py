"""A waiting request whose client has hung up must not be taken for service.

Every place that can take a waiting request (the staging pops of the queue, the
resident driver's inbox drain, the wait for a VRAM reclaim before an engine
start, the Fast Lane claim pruner) has to treat a request whose client is gone
as not takeable: it is never served, no engine is started for it, and its claim
does not keep pointing at it.

Each behaviour is pinned twice, on the same path with the same shape: once with
the client gone, and once with the client still connected (the control), so a
check that simply refuses everything cannot pass.

Part 1 drives the queue's public pop entry points. Part 2 runs the real
resident driver with a fake engine. Part 3 runs the real engine-start path
through the VRAM reclaim wait with a fake engine process. Part 4 reads the Fast
Lane claims snapshot. One process, asyncio only, bounded waits, no network.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from _dead_client_take_paths_support import (
    boot_and_runtime,
    loading_resident,
    manager_with_spawn_log,
    seed_single_card_manifest,
    vram_clear,
    vram_refusal,
    waiting_slot,
)
from _fastlane_fixture import boot_ranked_runtime, ranked_rules

from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import FastLanePopPolicy, TurbohaulQueue
from turbohaul.slot import Slot, SlotEvictedError, SlotState

# ---------------------------------------------------------------------------
# Part 1. The queue's pop entry points
# ---------------------------------------------------------------------------


def _match(rule_index=0, rank=1, address="192.0.2.1"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address=address, label="",
        effective_tag="main", rank=rank,
    )


def _slot(name, *, gone=False, model_tag="m1", thread_id="", client_meta=None,
          fastlane=None, age_s=0.0):
    """A request with a readable id. ``gone`` marks its client as hung up."""
    slot = Slot.new(
        model_tag=model_tag, prompt=name, thread_id=thread_id,
        client_meta=client_meta,
    )
    slot.slot_id = name
    slot.fastlane = fastlane
    slot.created_at -= age_s
    slot.disconnect_event = asyncio.Event()
    if gone:
        slot.disconnect_event.set()
    return slot


@dataclass
class Scenario:
    """A queue holding one request that may be gone, plus how to take from it.

    ``target`` is the request whose client may have hung up. ``neighbours`` are
    requests of connected clients that must still be served. ``take`` performs
    one pop through a public entry point.
    """

    queue: TurbohaulQueue
    target: Slot
    take: Callable[[], Awaitable[Slot | None]]
    neighbours: list = field(default_factory=list)


async def _fill(queue, slots):
    for slot in slots:
        await queue.enqueue(slot)


async def _fifo_head(gone):
    q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
    target = _slot("target", gone=gone)
    neighbour = _slot("neighbour")
    await _fill(q, [target, neighbour])
    return Scenario(q, target, q.pop_next, [neighbour])


async def _fifo_from_acceptance_buffer(gone):
    # Staging holds one request, so the target and its neighbour wait in the
    # acceptance buffer and move into staging one by one as pops make room.
    q = TurbohaulQueue(staging_max=1, main_lane_reserved=False)
    first = _slot("first")
    target = _slot("target", gone=gone)
    neighbour = _slot("neighbour")
    await _fill(q, [first, target, neighbour])
    return Scenario(q, target, q.pop_next, [first, neighbour])


async def _warm_model_affinity(gone):
    q = TurbohaulQueue(
        staging_max=10, max_consecutive_same_model=5,
        max_other_model_wait_s=3600.0, main_lane_reserved=False,
    )
    other = _slot("other", model_tag="m2")
    target = _slot("target", gone=gone)
    neighbour = _slot("neighbour")
    await _fill(q, [other, target, neighbour])

    async def take():
        return await q.pop_next(warm_model_tag="m1")

    return Scenario(q, target, take, [neighbour])


async def _starved_other_model(gone):
    q = TurbohaulQueue(
        staging_max=10, max_consecutive_same_model=5,
        max_other_model_wait_s=20.0, main_lane_reserved=False,
    )
    warm = _slot("warm")
    target = _slot("target", gone=gone, model_tag="m2", age_s=100.0)
    await _fill(q, [warm, target])

    async def take():
        return await q.pop_next(warm_model_tag="m1")

    return Scenario(q, target, take, [warm])


async def _main_lane_reservation(gone):
    q = TurbohaulQueue(staging_max=10, main_lane_reserved=True)
    ordinary = _slot("ordinary")
    target = _slot("target", gone=gone, client_meta={"is_main": True})
    other_main = _slot("other-main", client_meta={"is_main": True})
    await _fill(q, [ordinary, target, other_main])
    return Scenario(q, target, q.pop_next, [ordinary, other_main])


async def _main_lane_from_acceptance_buffer(gone):
    q = TurbohaulQueue(staging_max=1, main_lane_reserved=True)
    ordinary = _slot("ordinary")
    target = _slot("target", gone=gone, client_meta={"is_main": True})
    await _fill(q, [ordinary, target])
    return Scenario(q, target, q.pop_next, [ordinary])


async def _compression_skip_ahead(gone):
    q = TurbohaulQueue(staging_max=10, main_lane_reserved=True)
    ordinary = _slot("ordinary")
    target = _slot("target", gone=gone, client_meta={"is_compression": True})
    neighbour = _slot("neighbour")
    await _fill(q, [ordinary, target, neighbour])
    return Scenario(q, target, q.pop_next, [ordinary, neighbour])


def _policy():
    return FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)


async def _fast_lane_ranked_pick(gone):
    q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
    unlisted = _slot("unlisted")
    target = _slot("target", gone=gone, fastlane=_match(rule_index=0, rank=1))
    neighbour = _slot("neighbour", fastlane=_match(rule_index=0, rank=2))
    await _fill(q, [unlisted, target, neighbour])

    async def take():
        return await q.pop_next(warm_model_tag="m1", fastlane_policy=_policy())

    return Scenario(q, target, take, [unlisted, neighbour])


async def _fast_lane_fairness_floor(gone):
    q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
    listed = _slot("listed", fastlane=_match(rule_index=0, rank=1))
    target = _slot("target", gone=gone, age_s=100.0)
    await _fill(q, [listed, target])
    policy = FastLanePopPolicy(max_normal_wait_s=5.0, cross_model_switches_per_min=60)

    async def take():
        return await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)

    return Scenario(q, target, take, [listed])


async def _same_thread_follow_up(gone):
    q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
    unrelated = _slot("unrelated", thread_id="thread-other")
    target = _slot("target", gone=gone, thread_id="thread-follow-up")
    await _fill(q, [unrelated, target])

    async def take():
        return await q.pop_matched_thread("thread-follow-up", "m1")

    return Scenario(q, target, take, [])


SCENARIOS = {
    "plain_fifo_head": _fifo_head,
    "plain_fifo_refilled_from_acceptance_buffer": _fifo_from_acceptance_buffer,
    "warm_model_affinity": _warm_model_affinity,
    "starved_other_model_request": _starved_other_model,
    "main_lane_reservation": _main_lane_reservation,
    "main_lane_from_acceptance_buffer": _main_lane_from_acceptance_buffer,
    "compression_skip_ahead": _compression_skip_ahead,
    "fast_lane_ranked_pick": _fast_lane_ranked_pick,
    "fast_lane_fairness_floor": _fast_lane_fairness_floor,
    "same_thread_follow_up": _same_thread_follow_up,
}


async def _take_all(scenario, limit=8):
    """Take until the queue is empty (at most ``limit`` calls). A call that hands
    nothing back while requests still wait is retried, as the dispatcher does."""
    taken = []
    for _ in range(limit):
        slot = await scenario.take()
        if slot is not None:
            taken.append(slot)
        if scenario.queue.depth()["queue_depth_total"] == 0:
            break
    return taken


def _ids(slots):
    return [s.slot_id for s in slots]


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_a_request_whose_client_is_gone_is_never_taken_for_service(name):
    """The request at the front of this path has a client that has hung up.
    Whatever the pop hands back, the gone request must be marked evicted (so the
    dispatcher fails it instead of serving it), and the requests of connected
    clients behind it must still come out served.

    Fails when the gone request is handed back as an ordinary servable request,
    or when the requests behind it are lost."""
    scenario = await SCENARIOS[name](gone=True)
    taken = await _take_all(scenario)

    unflagged_gone = [
        s.slot_id for s in taken
        if s.disconnect_event.is_set() and not s.is_evicted
    ]
    assert not unflagged_gone, (
        f"{name}: a request whose client is gone was handed back as servable: "
        f"{unflagged_gone}; taken in order {_ids(taken)}"
    )
    served = [s for s in taken if not s.is_evicted]
    for neighbour in scenario.neighbours:
        assert neighbour in served, (
            f"{name}: the connected request {neighbour.slot_id!r} was not served; "
            f"served {_ids(served)}, taken {_ids(taken)}"
        )


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_a_request_whose_client_is_connected_is_taken_for_service(name):
    """Control for the test above: the same path, the same shape, but the client
    of the request at the front is still connected. It is handed back as an
    ordinary servable request, not marked evicted, and so are its neighbours.

    Fails when the path refuses or evicts a request whose client is present."""
    scenario = await SCENARIOS[name](gone=False)
    taken = await _take_all(scenario)

    served = [s for s in taken if not s.is_evicted]
    assert scenario.target in served, (
        f"{name}: the connected request was not served; taken {_ids(taken)}, "
        f"evicted {_ids([s for s in taken if s.is_evicted])}"
    )
    assert scenario.target.is_evicted is False
    for neighbour in scenario.neighbours:
        assert neighbour in served and neighbour.is_evicted is False, (
            f"{name}: the connected request {neighbour.slot_id!r} was not served; "
            f"served {_ids(served)}"
        )


# ---------------------------------------------------------------------------
# Part 2. The resident driver's inbox drain
# ---------------------------------------------------------------------------

DRIVER_MODEL = "m1"
CLIENT_A = "192.0.2.11"
CLIENT_B = "192.0.2.12"


async def _wait_until(predicate, *, timeout=5.0, what="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"{what} never became true within {timeout}s")


def _rider(name, created_at, address, rank, *, gone=False):
    slot = Slot(
        slot_id=name,
        model_tag=DRIVER_MODEL,
        state=SlotState.ACTIVE,
        created_at=created_at,
        fastlane=_match(rule_index=0 if address == CLIENT_A else 1, rank=rank,
                        address=address),
        client_meta={},
        completion_future=asyncio.Future(),
    )
    slot.disconnect_event = asyncio.Event()
    if gone:
        slot.disconnect_event.set()
    return slot


async def _run_driver_over_inbox(tmp_path, monkeypatch, *, rider_a_gone):
    """Run the real resident driver, with a fake engine and a fake turn, over an
    inbox that already holds the first request and two parked riders.

    Rider A belongs to the better-ranked client and arrived first; rider B is
    from a worse-ranked client. Both hold a Fast Lane claim parked in the inbox.
    Returns what the driver served, in order, and the pieces to inspect."""
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=ranked_rules((CLIENT_A, 1), (CLIENT_B, 1)),
    )
    mgr = TurbohaulManager(boot, runtime)
    resident = Resident(
        resident_key=DRIVER_MODEL, model_tag=DRIVER_MODEL, inbox=asyncio.Queue(),
    )
    first = Slot(
        slot_id="first", model_tag=DRIVER_MODEL, state=SlotState.ACTIVE,
        created_at=0.0, client_meta={}, completion_future=asyncio.Future(),
    )
    rider_a = _rider("rider-a", 1.0, CLIENT_A, 1, gone=rider_a_gone)
    rider_b = _rider("rider-b", 2.0, CLIENT_B, 1)
    for rider in (rider_a, rider_b):
        mgr._register_fastlane_claim_locked(rider, "parked_in_inbox")
        mgr._park_fastlane_claim_locked(rider, resident)
    for slot in (first, rider_a, rider_b):
        resident.inbox.put_nowait(slot)

    served: list[Slot] = []

    class _StoppedEngine:
        """An engine handle that reports itself as not running, so the driver
        skips saving the previous turn's context."""

        def is_alive(self):
            return False

    async def fake_spawn(r_, slot):
        return _StoppedEngine()

    async def fake_serve(r_, slot, handle):
        served.append(slot)

    monkeypatch.setattr(mgr, "_spawn_for_resident", fake_spawn)
    monkeypatch.setattr(mgr, "_serve_on_resident", fake_serve)
    # Keep the engine warm indefinitely, so the only way the driver leaves its
    # park is by taking a request from the inbox.
    resident.latest_keep_alive_s = -1
    resident.sleep_idle_seconds = -1
    task = asyncio.create_task(mgr._drive_resident(resident))
    expected = 2 if rider_a_gone else 3
    try:
        await _wait_until(
            lambda: len(served) >= expected
            and resident.state is ResidentState.IDLE_EVICTABLE
            and resident.inbox.empty(),
            what=f"the driver serving {expected} requests and parking",
        )
        # Give a wrongly admitted extra request the chance to show up.
        for _ in range(5):
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)
    return mgr, resident, served, first, rider_a, rider_b


def _claimed_slots(mgr):
    return [c["slot"] for c in mgr._fastlane_claims.values()]


async def test_a_parked_rider_whose_client_is_gone_is_not_admitted_and_loses_its_claim(
    tmp_path, monkeypatch,
):
    """The better-ranked rider parked in a resident's inbox has hung up while the
    resident was busy. When the resident picks its next request it must serve the
    other rider, never the gone one: the gone rider's own completion fails with
    the evicted error and its Fast Lane claim is released, while the connected
    rider keeps being served normally.

    Fails when the gone rider is served, or keeps its claim, or its completion is
    left pending."""
    mgr, resident, served, first, rider_a, rider_b = await _run_driver_over_inbox(
        tmp_path, monkeypatch, rider_a_gone=True,
    )
    assert _ids(served) == ["first", "rider-b"], (
        f"the driver served {_ids(served)}; the gone rider must be skipped and the "
        "connected rider served"
    )
    assert rider_a.is_evicted, "the gone rider was not marked evicted"
    assert rider_a.completion_future.done() and isinstance(
        rider_a.completion_future.exception(), SlotEvictedError
    ), "the gone rider's completion was not failed with the evicted error"
    assert rider_a not in _claimed_slots(mgr), (
        "the gone rider still holds a Fast Lane claim"
    )
    assert rider_b.is_evicted is False


async def test_a_parked_rider_whose_client_is_connected_is_admitted_first(
    tmp_path, monkeypatch,
):
    """Control for the test above: the same inbox and the same ranks, but the
    better-ranked rider's client is still connected. It is served right after the
    first request, then the other rider; nothing is evicted and no completion is
    failed.

    Fails when the driver skips or evicts a rider whose client is present."""
    mgr, resident, served, first, rider_a, rider_b = await _run_driver_over_inbox(
        tmp_path, monkeypatch, rider_a_gone=False,
    )
    assert _ids(served) == ["first", "rider-a", "rider-b"], (
        f"the driver served {_ids(served)}; expected the better-ranked rider first"
    )
    assert not rider_a.is_evicted and not rider_b.is_evicted
    assert not rider_a.completion_future.done()
    assert not rider_b.completion_future.done()


# ---------------------------------------------------------------------------
# Part 3. The wait for a VRAM reclaim before an engine start
# ---------------------------------------------------------------------------

MODEL = "m1"


async def _release_after(delay_s, hangup=None):
    await asyncio.sleep(delay_s)
    if hangup is not None:
        hangup.set()


async def _start_engine_through_reclaim_wait(
    tmp_path, *, release_s, hangup, hangup_with_release, gate_clears_on_release,
    budget_s, hangup_at_s=None,
):
    """Start an engine for one request whose VRAM safety gate first refuses
    while a release is in flight on its card, so the start enters the reclaim
    wait.

    ``hangup`` is the request's disconnect event (or None for a client that
    stays). ``hangup_with_release`` sets it at the very moment the release
    finishes; ``hangup_at_s`` sets it at that time instead. With
    ``gate_clears_on_release`` the re-run gate passes once the release is done,
    otherwise it keeps refusing. Returns the pieces to inspect."""
    boot, runtime = boot_and_runtime(tmp_path, spawn_reclaim_wait_max_s=budget_s)
    seed_single_card_manifest(boot, MODEL)
    mgr, spawn_log = manager_with_spawn_log(boot, runtime)
    release = asyncio.create_task(
        _release_after(release_s, hangup if hangup_with_release else None)
    )
    mgr._card_release_tasks.setdefault(0, set()).add(release)
    side_tasks = []
    if hangup is not None and hangup_at_s is not None:
        side_tasks.append(asyncio.create_task(_release_after(hangup_at_s, hangup)))

    def fake_gates(**_kw):
        if gate_clears_on_release and release.done():
            return [vram_clear()]
        return [vram_refusal()]

    resident = loading_resident(MODEL)
    mgr._residents[MODEL] = resident
    slot = waiting_slot(MODEL, name="waiting", disconnect_event=hangup)
    try:
        with patch("turbohaul.manager.all_safety_gates", side_effect=fake_gates), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[10_000]), \
             patch("turbohaul.manager._SPAWN_RECLAIM_REGATE_TICK_S", 0.05):
            handle = await asyncio.wait_for(
                mgr._spawn_for_resident(resident, slot), timeout=4.0,
            )
    finally:
        for task in (release, *side_tasks):
            task.cancel()
        await asyncio.gather(release, *side_tasks, return_exceptions=True)
        await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)
    return handle, spawn_log, slot, resident


async def test_no_engine_is_started_for_a_client_that_hangs_up_as_the_reclaim_finishes(
    tmp_path,
):
    """The request waits for a VRAM release that is in flight on its card. The
    client hangs up at the moment that release finishes, so the next safety check
    passes. No engine may be started for a client that is gone: the start ends
    without one, the request's completion fails with the evicted error, and the
    wait leaves no flag behind.

    Fails when the engine is started (and so a model load is spent) for the
    client that hung up."""
    hangup = asyncio.Event()
    handle, spawn_log, slot, resident = await _start_engine_through_reclaim_wait(
        tmp_path, release_s=0.12, hangup=hangup, hangup_with_release=True,
        gate_clears_on_release=True, budget_s=3.0,
    )
    assert spawn_log == [], (
        f"an engine was started for a client that had hung up: {spawn_log}"
    )
    assert handle is None
    assert slot.completion_future.done() and isinstance(
        slot.completion_future.exception(), SlotEvictedError
    ), "the gone request's completion was not failed with the evicted error"
    assert resident.spawn_barrier_active is False


async def test_the_engine_is_started_for_a_client_that_stays_through_the_reclaim_wait(
    tmp_path,
):
    """Control for the test above: the same wait, the same release finishing, the
    same passing re-check, but the client stays connected. The engine is started
    once and the request is not evicted.

    Fails when the wait refuses a client that is present."""
    handle, spawn_log, slot, resident = await _start_engine_through_reclaim_wait(
        tmp_path, release_s=0.12, hangup=asyncio.Event(), hangup_with_release=False,
        gate_clears_on_release=True, budget_s=3.0,
    )
    assert handle is not None, "no engine was started for a client that stayed"
    assert spawn_log == [MODEL]
    assert not slot.is_evicted
    assert not slot.completion_future.done()
    assert resident.spawn_barrier_active is False


async def test_no_engine_is_started_for_a_client_that_hangs_up_during_the_reclaim_wait(
    tmp_path,
):
    """The request waits while the VRAM check keeps refusing and the release is
    slow. The client hangs up early in the wait. The wait ends within a few
    ticks, no engine is started, and the request's completion fails with the
    evicted error rather than with a memory refusal.

    Fails when the wait outlives the hang-up, starts an engine, or reports the
    wrong error."""
    hangup = asyncio.Event()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    handle, spawn_log, slot, resident = await _start_engine_through_reclaim_wait(
        tmp_path, release_s=2.0, hangup=hangup, hangup_with_release=False,
        gate_clears_on_release=False, budget_s=3.0, hangup_at_s=0.06,
    )
    elapsed = loop.time() - t0
    assert spawn_log == [] and handle is None, (
        f"an engine was started for a client that had hung up: {spawn_log}"
    )
    assert elapsed < 1.0, f"the wait went on for {elapsed:.2f}s after the hang-up"
    assert slot.completion_future.done() and isinstance(
        slot.completion_future.exception(), SlotEvictedError
    )
    assert resident.spawn_barrier_active is False


async def test_a_client_that_stays_is_refused_with_a_memory_error_when_the_wait_runs_out(
    tmp_path,
):
    """Control for the test above: the same slow release and the same refusing
    check, but the client stays. When the bounded wait runs out the request is
    refused for memory, not evicted, and no engine is started.

    Fails when a connected client is reported as having hung up."""
    handle, spawn_log, slot, resident = await _start_engine_through_reclaim_wait(
        tmp_path, release_s=2.0, hangup=asyncio.Event(), hangup_with_release=False,
        gate_clears_on_release=False, budget_s=0.3,
    )
    assert spawn_log == [] and handle is None
    assert not slot.is_evicted
    exc = slot.completion_future.exception()
    assert isinstance(exc, RuntimeError) and not isinstance(exc, SlotEvictedError), (
        f"expected a refusal for memory, got {exc!r}"
    )
    assert resident.spawn_barrier_active is False


# ---------------------------------------------------------------------------
# Part 4. The Fast Lane claim pruner
# ---------------------------------------------------------------------------

POINTER_A = "192.0.2.21"
POINTER_B = "192.0.2.22"


def _claim_manager(tmp_path):
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=ranked_rules((POINTER_A, 1), (POINTER_B, 1)),
    )
    return TurbohaulManager(boot, runtime)


def _waiter(name, created_at, address, rule_index, *, gone=False):
    slot = Slot(
        slot_id=name,
        model_tag=MODEL,
        state=SlotState.STAGED,
        created_at=created_at,
        fastlane=_match(rule_index=rule_index, rank=1, address=address),
        client_meta={},
        completion_future=asyncio.Future(),
    )
    slot.disconnect_event = asyncio.Event()
    if gone:
        slot.disconnect_event.set()
    return slot


async def _settle(mgr):
    await asyncio.gather(*list(mgr._bg_tasks), return_exceptions=True)


async def test_a_claim_whose_request_is_gone_moves_to_the_next_live_waiter_of_its_client(
    tmp_path,
):
    """Two requests of one client wait for one model and share one claim, which
    points at the first. The first one's client hangs up. Reading the claims
    snapshot must move the claim to the second request, which is still waiting,
    rather than keep it on the gone request or drop it.

    Fails when the snapshot lists the gone request, or lists no claim for the
    client that still has a waiting request."""
    mgr = _claim_manager(tmp_path)
    first = _waiter("first", 1.0, POINTER_A, 0)
    second = _waiter("second", 2.0, POINTER_A, 0)
    mgr._register_fastlane_claim_locked(first, "staging_arrival")
    mgr._register_fastlane_claim_locked(second, "staging_arrival")
    assert _ids(_claimed_slots(mgr)) == ["first"], "setup: the claim must point at the first"

    first.disconnect_event.set()
    rows = mgr.fastlane_claims_snapshot()
    await _settle(mgr)

    assert [row["slot_id"] for row in rows] == ["second"], (
        f"the snapshot lists {[row['slot_id'] for row in rows]}, expected the claim "
        "to have moved to the waiting request"
    )
    assert _ids(_claimed_slots(mgr)) == ["second"]


async def test_a_claim_whose_request_is_still_connected_stays_where_it_is(tmp_path):
    """Control for the test above: the same two requests, but the first one's
    client is still connected. The snapshot keeps the claim on the first request.

    Fails when the pruner moves or drops the claim of a request that is present."""
    mgr = _claim_manager(tmp_path)
    first = _waiter("first", 1.0, POINTER_A, 0)
    second = _waiter("second", 2.0, POINTER_A, 0)
    mgr._register_fastlane_claim_locked(first, "staging_arrival")
    mgr._register_fastlane_claim_locked(second, "staging_arrival")

    rows = mgr.fastlane_claims_snapshot()
    await _settle(mgr)

    assert [row["slot_id"] for row in rows] == ["first"]
    assert _ids(_claimed_slots(mgr)) == ["first"]


async def test_a_claim_whose_only_request_is_gone_is_removed_and_other_clients_keep_theirs(
    tmp_path,
):
    """Two clients each have one waiting request and one claim. The first
    client's request is gone and nothing else of that client waits. The snapshot
    must drop that client's claim and leave the other client's claim listed.

    Fails when the gone request's claim is still listed, or when the other
    client's claim is dropped with it."""
    mgr = _claim_manager(tmp_path)
    gone = _waiter("gone", 1.0, POINTER_A, 0)
    staying = _waiter("staying", 2.0, POINTER_B, 1)
    mgr._register_fastlane_claim_locked(gone, "staging_arrival")
    mgr._register_fastlane_claim_locked(staying, "staging_arrival")

    gone.disconnect_event.set()
    rows = mgr.fastlane_claims_snapshot()
    await _settle(mgr)

    assert [row["slot_id"] for row in rows] == ["staying"], (
        f"the snapshot lists {[row['slot_id'] for row in rows]}"
    )
    assert _ids(_claimed_slots(mgr)) == ["staying"]


async def test_two_clients_that_both_stay_keep_both_claims(tmp_path):
    """Control for the test above: the same two clients, both still connected.
    Both claims are listed.

    Fails when the pruner removes a claim whose request is present."""
    mgr = _claim_manager(tmp_path)
    first = _waiter("first", 1.0, POINTER_A, 0)
    second = _waiter("second", 2.0, POINTER_B, 1)
    mgr._register_fastlane_claim_locked(first, "staging_arrival")
    mgr._register_fastlane_claim_locked(second, "staging_arrival")

    rows = mgr.fastlane_claims_snapshot()
    await _settle(mgr)

    assert sorted(row["slot_id"] for row in rows) == ["first", "second"]
    assert sorted(_ids(_claimed_slots(mgr))) == ["first", "second"]
