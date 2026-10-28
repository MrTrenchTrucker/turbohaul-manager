"""A follow-up whose client has hung up must not be claimed by a resident's grace loop.

When a resident finishes a turn it waits in a grace window for a follow-up of the
same conversation, and claims one two ways: by the thread id, and by the request's
hash chain extending the finished turn's chain. A follow-up claimed there is served
warm on the engine that is already loaded, so claiming one whose client is gone
spends an engine turn on nobody.

Part 1 calls the two claim entry points of the queue directly, on the shapes they
can reach: the hash-chain drain over a resident's inbox, the hash-chain drain over
the staging queue, and the thread match over the staging queue. What a claim hands
back must never be a gone request that looks servable: it is either not handed back
at all, or it is marked evicted. The requests of connected clients that wait around
it are kept, in order.

Part 2 runs the real resident driver (the grace loop included) over a fake engine
and counts the turns the engine is asked to run. The same follow-up with a gone
client must not get a turn; with a connected client it is served warm.

Every behaviour is pinned twice, on the same path with the same shape: once with the
client gone and once with the client connected (the control), so a check that simply
refuses every follow-up cannot pass. One process, asyncio only, bounded waits, no
network.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest
from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    resident_for,
    seed_manifest,
    wait_until,
)

from turbohaul.kv_policy import _prefix_hash_chain
from turbohaul.manager import TurbohaulManager
from turbohaul.queue import TurbohaulQueue
from turbohaul.slot import Slot, SlotState

MODEL = "m1"
ANCHOR_THREAD = "thread-anchor"
ANCHOR_HISTORY = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
]
ANCHOR_CHAIN = _prefix_hash_chain(ANCHOR_HISTORY)
FOLLOW_UP_HISTORY = ANCHOR_HISTORY + [{"role": "user", "content": "second question"}]
UNRELATED_HISTORY = [{"role": "user", "content": "something else entirely"}]

# ---------------------------------------------------------------------------
# Part 1. The queue's two claim entry points
# ---------------------------------------------------------------------------


def _request(name, *, history, thread_id="", gone=False, age_s=-1.0):
    """A waiting request with a readable id. ``gone`` marks its client as hung up.

    ``age_s`` is how long ago it arrived; the default is a moment after the
    grace window opened (the hash-chain claim only takes later arrivals)."""
    slot = Slot.new(
        model_tag=MODEL, context=list(history), thread_id=thread_id,
        admission_hash_chain=_prefix_hash_chain(history),
    )
    slot.slot_id = name
    slot.created_at = time.monotonic() - age_s
    slot.disconnect_event = asyncio.Event()
    if gone:
        slot.disconnect_event.set()
    return slot


@dataclass
class Scenario:
    """A queue holding one follow-up that may be gone, plus how to claim it.

    ``target`` is the follow-up whose client may have hung up. ``bystanders`` are
    requests of connected clients that are not follow-ups and must stay where they
    wait, in order. ``live_behind`` is, when present, a second matching follow-up
    of a connected client waiting behind the target. ``claim`` makes one claim
    attempt. ``waiting`` lists what still waits, in order.
    """

    target: Slot
    claim: Callable[[], Awaitable[Slot | None]]
    waiting: Callable[[], list]
    bystanders: list = field(default_factory=list)
    live_behind: Slot | None = None


def _drain_inbox(inbox):
    """Everything in the inbox, in order, left in place."""
    items = []
    while not inbox.empty():
        items.append(inbox.get_nowait())
    for item in items:
        inbox.put_nowait(item)
    return items


async def _hash_chain_over_inbox(gone, live_behind=False):
    q = TurbohaulQueue(staging_max=10)
    inbox: asyncio.Queue[Slot] = asyncio.Queue()
    first = _request("bystander-1", history=UNRELATED_HISTORY)
    target = _request("target", history=FOLLOW_UP_HISTORY, gone=gone)
    live = _request("live", history=FOLLOW_UP_HISTORY) if live_behind else None
    last = _request("bystander-2", history=UNRELATED_HISTORY + [{"role": "user", "content": "x"}])
    for slot in (first, target, live, last):
        if slot is not None:
            inbox.put_nowait(slot)
    assert not q.staging_snapshot(), "the inbox shape must not use the staging queue"

    async def claim():
        return await q.drain_inbox_and_staging_match_hash_chain(
            ANCHOR_CHAIN, MODEL, time.monotonic() - 10.0, inbox,
        )

    return Scenario(target, claim, lambda: _drain_inbox(inbox), [first, last], live)


async def _hash_chain_over_staging(gone, live_behind=False):
    q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
    first = _request("bystander-1", history=UNRELATED_HISTORY)
    target = _request("target", history=FOLLOW_UP_HISTORY, gone=gone)
    live = _request("live", history=FOLLOW_UP_HISTORY) if live_behind else None
    last = _request("bystander-2", history=UNRELATED_HISTORY + [{"role": "user", "content": "x"}])
    for slot in (first, target, live, last):
        if slot is not None:
            await q.enqueue(slot)
    inbox: asyncio.Queue[Slot] = asyncio.Queue()

    async def claim():
        return await q.drain_inbox_and_staging_match_hash_chain(
            ANCHOR_CHAIN, MODEL, time.monotonic() - 10.0, inbox,
        )

    return Scenario(target, claim, lambda: list(q.staging_snapshot()), [first, last], live)


async def _thread_match_over_staging(gone, live_behind=False):
    q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
    first = _request("bystander-1", history=UNRELATED_HISTORY, thread_id="thread-other-1")
    target = _request("target", history=FOLLOW_UP_HISTORY, thread_id=ANCHOR_THREAD, gone=gone)
    live = (
        _request("live", history=FOLLOW_UP_HISTORY, thread_id=ANCHOR_THREAD)
        if live_behind else None
    )
    last = _request("bystander-2", history=UNRELATED_HISTORY, thread_id="thread-other-2")
    for slot in (first, target, live, last):
        if slot is not None:
            await q.enqueue(slot)

    async def claim():
        return await q.pop_matched_thread(ANCHOR_THREAD, MODEL)

    return Scenario(target, claim, lambda: list(q.staging_snapshot()), [first, last], live)


SHAPES = {
    "hash_chain_over_inbox": _hash_chain_over_inbox,
    "hash_chain_over_staging": _hash_chain_over_staging,
    "thread_match_over_staging": _thread_match_over_staging,
}


async def _claim_all(scenario, attempts=3):
    """Claim until nothing more is handed back (at most ``attempts`` calls). A call
    that hands nothing back is retried, as the grace loop does on its next poll."""
    claimed = []
    for _ in range(attempts):
        slot = await scenario.claim()
        if slot is not None:
            claimed.append(slot)
    return claimed


def _ids(slots):
    return [s.slot_id for s in slots]


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_a_follow_up_whose_client_is_gone_is_not_claimed_as_servable(shape):
    """The follow-up of the finished turn has a client that has hung up. Whatever
    the claim hands back, the gone follow-up must not come back as an ordinary
    servable request (it is either left out or marked evicted), and the requests
    of connected clients around it must still be waiting, in their order.

    Fails when the gone follow-up is handed back unmarked, or when the requests of
    connected clients are lost or reordered."""
    scenario = await SHAPES[shape](gone=True)
    claimed = await _claim_all(scenario)

    unmarked = [s.slot_id for s in claimed if s is scenario.target and not s.is_evicted]
    assert not unmarked, (
        f"{shape}: a follow-up whose client is gone was handed back as servable; "
        f"claimed in order {_ids(claimed)}"
    )
    still_waiting = [s for s in scenario.waiting() if s is not scenario.target]
    assert still_waiting == scenario.bystanders, (
        f"{shape}: the waiting requests of connected clients changed: expected "
        f"{_ids(scenario.bystanders)}, found {_ids(still_waiting)}"
    )


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_a_follow_up_whose_client_is_connected_is_claimed(shape):
    """Control for the test above: the same path and shape, but the client of the
    follow-up is still connected. It is handed back as an ordinary servable
    request, not marked evicted, and the requests around it stay in place.

    Fails when the claim refuses or evicts a follow-up whose client is present."""
    scenario = await SHAPES[shape](gone=False)
    claimed = await _claim_all(scenario, attempts=1)

    assert claimed == [scenario.target], (
        f"{shape}: the connected follow-up was not the one claimed; claimed "
        f"{_ids(claimed)}"
    )
    assert scenario.target.is_evicted is False
    still_waiting = [s for s in scenario.waiting() if s is not scenario.target]
    assert still_waiting == scenario.bystanders, (
        f"{shape}: the waiting requests of connected clients changed: expected "
        f"{_ids(scenario.bystanders)}, found {_ids(still_waiting)}"
    )


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_a_connected_follow_up_behind_a_gone_one_is_still_claimed(shape):
    """Two follow-ups of the same conversation wait, the first with a client that
    has hung up and the second with a client still connected. Over a few claim
    attempts the connected one is handed back servable, and the gone one never is.

    Fails when the gone one is handed back unmarked, or when the connected one
    behind it is never claimed."""
    scenario = await SHAPES[shape](gone=True, live_behind=True)
    claimed = await _claim_all(scenario)

    unmarked = [s.slot_id for s in claimed if s is scenario.target and not s.is_evicted]
    assert not unmarked, (
        f"{shape}: a follow-up whose client is gone was handed back as servable; "
        f"claimed in order {_ids(claimed)}"
    )
    assert scenario.live_behind in claimed and not scenario.live_behind.is_evicted, (
        f"{shape}: the connected follow-up behind the gone one was not claimed; "
        f"claimed in order {_ids(claimed)}"
    )


# ---------------------------------------------------------------------------
# Part 2. The real resident driver's grace loop over a fake engine
# ---------------------------------------------------------------------------

END_TO_END_SHAPES = ("thread_match_staging", "hash_chain_staging", "hash_chain_inbox")


async def _run_grace_loop_with_follow_up(tmp_path, shape, *, gone):
    """Serve one request on a real resident driver, let it enter its grace window,
    then park a follow-up of the same conversation where ``shape`` says.

    The dispatcher's own pick is switched off, so the only way the follow-up can
    get a turn is the resident's grace loop. Returns ``(follow_up, served,
    resident)``; ``served`` lists every request the fake engine was asked to run.
    """
    boot, runtime = boot_ranked_runtime(tmp_path, grace_seconds=4)
    seed_manifest(boot, MODEL, main_gpu=0)
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})
    served: list[Slot] = []
    release_anchor = asyncio.Event()

    async def complete(slot, handle):
        served.append(slot)
        if slot.prompt == "anchor-turn":
            await release_anchor.wait()
        return {"ok": True}

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
        sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete,
    )
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            await mgr.submit(
                model_tag=MODEL, prompt="anchor-turn", thread_id=ANCHOR_THREAD,
                admission_hash_chain=ANCHOR_CHAIN,
            )
            await wait_until(
                lambda: any(r.model_tag == MODEL for r in mgr._model_residents()),
                timeout=5.0,
            )
            resident = resident_for(mgr, MODEL)
            release_anchor.set()
            await wait_until(lambda: getattr(resident, "in_grace_loop", False), timeout=5.0)

            async def no_pick(*_a, **_k):
                return None

            mgr.queue.pop_next = no_pick

            by_thread = shape == "thread_match_staging"
            follow_up = Slot.new(
                model_tag=MODEL, prompt="follow-up-turn", context=list(FOLLOW_UP_HISTORY),
                thread_id=ANCHOR_THREAD if by_thread else "thread-drifted",
                admission_hash_chain=_prefix_hash_chain(FOLLOW_UP_HISTORY),
            )
            follow_up.state = SlotState.STAGED
            follow_up.disconnect_event = asyncio.Event()
            if gone:
                follow_up.disconnect_event.set()
            if shape == "hash_chain_inbox":
                resident.inbox.put_nowait(follow_up)
            else:
                mgr.queue._staging.append(follow_up)

            if gone:
                # Give a wrongly claimed follow-up the chance to show up: the grace
                # loop polls every 50 ms.
                await asyncio.sleep(0.8)
            else:
                await wait_until(lambda: follow_up.state is SlotState.POPPED, timeout=5.0)
    finally:
        await mgr.shutdown()
    return follow_up, served, resident


@pytest.mark.parametrize("shape", END_TO_END_SHAPES)
async def test_the_grace_loop_runs_no_turn_for_a_follow_up_whose_client_is_gone(
    tmp_path, shape,
):
    """A resident has just finished a turn and is in its grace window. A follow-up
    of the same conversation waits (in staging matched by thread id, in staging
    matched by hash chain, or in the resident's inbox matched by hash chain) and its
    client has hung up. The engine must not be asked to run a turn for it.

    Fails when the fake engine is asked to run a turn for the gone follow-up."""
    follow_up, served, _resident = await _run_grace_loop_with_follow_up(
        tmp_path, shape, gone=True,
    )
    assert follow_up not in served, (
        f"{shape}: the engine ran a turn for a follow-up whose client had hung up; "
        f"turns run: {[s.prompt for s in served]}"
    )


@pytest.mark.parametrize("shape", END_TO_END_SHAPES)
async def test_the_grace_loop_serves_a_follow_up_whose_client_is_connected(
    tmp_path, shape,
):
    """Control for the test above: the same resident, grace window and follow-up
    shape, but the follow-up's client is still connected. It is served warm on the
    engine already loaded: the engine runs exactly the finished turn and then the
    follow-up, and the follow-up is not marked evicted.

    Fails when the grace loop does not serve a follow-up whose client is present."""
    follow_up, served, _resident = await _run_grace_loop_with_follow_up(
        tmp_path, shape, gone=False,
    )
    assert [s.prompt for s in served] == ["anchor-turn", "follow-up-turn"], (
        f"{shape}: unexpected turns run: {[s.prompt for s in served]}"
    )
    assert follow_up.state is SlotState.POPPED
    assert follow_up.is_evicted is False
