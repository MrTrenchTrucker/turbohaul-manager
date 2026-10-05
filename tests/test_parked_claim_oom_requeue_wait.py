"""The claim of a request that waits for room after an out-of-memory load failure.

A request for a cold model is handed to the inbox of a new resident while that resident's
engine starts, and keeps its Fast Lane claim, marked as parked on the resident. When the engine
fails to load for lack of memory the resident is torn down and the request goes to a detached
companion that waits for a make-room wake and then re-enqueues it. While the companion waits,
no inbox holds the request. Its claim therefore has to be an ordinary waiting claim for the
whole wait, not just once the wait ends:

* it carries no mark naming the resident that is gone,
* it can govern the victim designation (the one request that needs room is the one that has to
  make a victim lose its grace), and
* its time-to-live runs again, re-armed from the moment the mark was dropped.

The existing re-queue test in ``test_parked_claim_swap_window_and_requeue`` reads the claim only
after the companion has re-enqueued the request, so it cannot see what the claim looked like
during the wait. The tests here hold the wait on purpose (no make-room wake is sent) and read
the claim while the companion is still waiting.

Everything runs real manager code with a fake engine process whose load fails with the
out-of-memory scan. One process, asyncio only, bounded waits, no sleep decides a result. The
queue's re-enqueue is replaced by a recorder, so nothing re-admits the request.
"""

import asyncio
import contextlib
import time
from types import SimpleNamespace

from _fastlane_fixture import ranked_rules, wait_until
from test_parked_claim_lifetime_and_kv_save import (
    PROMPT_S,
    build_world,
    hold_and_park,
)
from test_oom_requeue_resident_driver import (
    _OOM_SCAN,
    _make_manager,
    _run_worker,
    _wait_until_parked,
)

from turbohaul import manager as manager_module

OOM_IP = "10.0.0.5"
OOM_POLL_S = 0.01

# The claim time-to-live for these tests. It is long on purpose: nothing here waits for it to
# run out, and the claim must still be live when it is read while the companion waits.
HELD_TTL_S = 30.0

# Bounds, not delays.
RELEASE_S = 5.0
SHUTDOWN_S = 1.5


def companion_tasks(mgr):
    """The background tasks that are the out-of-memory re-queue companions and still running."""
    return [
        t for t in list(mgr._bg_tasks)
        if not t.done()
        and getattr(t.get_coro(), "__name__", "") == "_oom_requeue_wait_then_readmit"
    ]


@contextlib.asynccontextmanager
async def held_oom_wait(tmp_path, monkeypatch):
    """A cold reservation whose engine load fails for lack of memory, with the companion's wait
    held: no make-room wake is ever sent here.

    The manager's claim park is spied so the claim's time-to-live deadline can be read at the
    moment the claim was parked, and the queue's re-enqueue is replaced by a recorder. Yields
    ``SimpleNamespace(mgr, slot, enqueued, park_deadline)`` once the resident is gone and the
    companion is waiting. On the way out (also when a check failed) the client is disconnected,
    which ends the companion by itself, and the manager is shut down with a bound, so a test can
    never leave the companion running or hang in teardown.
    """
    monkeypatch.setattr(manager_module, "_FASTLANE_CLAIM_TTL_S", HELD_TTL_S)
    mgr, _calls, _boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN,
        fastlane_rules=ranked_rules((OOM_IP, 1)), poll_s=OOM_POLL_S,
    )
    enqueued = []
    parks = []
    park_deadline = []

    async def record_enqueue_head(slot):
        enqueued.append(slot)

    real_park = mgr._park_fastlane_claim_locked

    def spy_park(slot, r):
        real_park(slot, r)
        claim = mgr._fastlane_claims.get(mgr._fastlane_claim_key(slot))
        parks.append((slot, r.resident_key))
        park_deadline.append(None if claim is None else claim["ttl_deadline_monotonic"])

    mgr.queue.enqueue_head = record_enqueue_head
    mgr._park_fastlane_claim_locked = spy_park
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event, client_meta={"ip": OOM_IP},
        )
        assert slot.fastlane is not None, "setup: an unlisted request holds no claim at all"
        await _run_worker(mgr)
        assert await _wait_until_parked(slot), "setup: the load failure did not re-queue the request"
        await wait_until(lambda: "m1" not in mgr._residents, timeout=PROMPT_S)
        assert [s for s, _key in parks] == [slot], (
            f"setup: the request was not parked on the new resident exactly once: {parks}"
        )
        assert park_deadline and park_deadline[0] is not None, "setup: the parked request holds no claim"
        assert len(companion_tasks(mgr)) == 1, "setup: the re-queue companion is not waiting"
        assert enqueued == [], f"setup: the request was re-enqueued before any make-room wake: {enqueued}"
        yield SimpleNamespace(
            mgr=mgr, slot=slot, enqueued=enqueued, park_deadline=park_deadline[0],
        )
    finally:
        disconnect_event.set()
        await asyncio.wait_for(mgr.shutdown(), timeout=SHUTDOWN_S)


async def release_companion(held):
    """Release the held wait the way production does: real make-room wakes until the companion
    re-enqueues the request (bounded), then let the background tasks settle."""
    mgr = held.mgr
    deadline = time.monotonic() + RELEASE_S
    while not held.enqueued and time.monotonic() < deadline:
        async with mgr._make_room_signal:
            mgr._make_room_signal.notify_all()
        await asyncio.sleep(0.01)
    assert held.enqueued == [held.slot], f"the companion never re-enqueued the request: {held.enqueued}"
    for _ in range(20):
        pending = [t for t in list(mgr._bg_tasks) if not t.done()]
        if not pending:
            break
        await asyncio.wait(pending, timeout=0.25)


def claim_of(mgr, slot):
    key = mgr._fastlane_claim_key(slot)
    claim = None if key is None else mgr._fastlane_claims.get(key)
    assert claim is not None and claim["slot"] is slot, (
        "the waiting request holds no claim of its own: "
        f"{[c['slot'] is slot for c in mgr._fastlane_claims.values()]}"
    )
    return claim


async def read_probes(mgr, slot, park_deadline):
    """Read the claim of ``slot`` the way the two readers of the parked mark do.

    ``governing_slot`` is the slot of the claim that governs when claims parked in an inbox are
    left out (what the queue's hold-aside and the victim designation use). ``exempt_when_old``
    asks whether a copy of the claim with its time-to-live deadline in the past is still
    treated as live, which is what only a claim carrying the mark is. ``rearmed`` is whether the
    deadline is later than it was when the claim was parked. Nothing here changes the claim.
    """
    claim = claim_of(mgr, slot)
    async with mgr._registry_lock:
        governing = mgr._governing_claim_locked(include_parked=False)
    now = time.monotonic()
    deadline = claim["ttl_deadline_monotonic"]
    return SimpleNamespace(
        claim=claim,
        parked_on=claim.get("parked_on"),
        governing_slot=None if governing is None else governing[1],
        deadline=deadline,
        rearmed=deadline > park_deadline,
        within_bound=deadline <= now + manager_module._FASTLANE_CLAIM_TTL_S,
        live=mgr._claim_is_live(claim) is None,
        exempt_when_old=mgr._claim_is_live(dict(claim, ttl_deadline_monotonic=now - 1.0)) is None,
    )


# --------------------------------------------------------------------------------------
# the claim while the companion waits
# --------------------------------------------------------------------------------------

async def test_the_claim_of_a_request_waiting_for_room_after_an_oom_load_failure_is_an_ordinary_claim(
    tmp_path, monkeypatch,
):
    """A cold model is reserved for a listed client, its new engine fails to load for lack of
    memory, and the request goes to the companion that waits for a make-room wake. The wait is
    held: no make-room wake is sent, so the companion is still waiting and nothing has
    re-enqueued the request. During that wait the resident that held the claim is gone, so the
    claim must be an ordinary waiting claim: (a) not marked as parked on the dead resident,
    (b) the governing claim when claims parked in an inbox are left out, which is how the one
    request that needs room gets to designate a victim, and (c) back under the claim
    time-to-live, with the deadline re-armed after the moment the claim was parked and no later
    than a full time-to-live from now. A claim that keeps the mark never governs and never
    expires. The wait is then released with real make-room wakes."""
    async with held_oom_wait(tmp_path, monkeypatch) as held:
        mgr, slot = held.mgr, held.slot
        assert "m1" not in mgr._residents, "setup: the resident that held the claim must be gone"
        assert slot.oom_requeue_pending and len(companion_tasks(mgr)) == 1, (
            "setup: the request must be waiting in the companion"
        )

        p = await read_probes(mgr, slot, held.park_deadline)
        assert held.enqueued == [], "the wait was not held: the request was re-enqueued by itself"
        assert len(companion_tasks(mgr)) == 1, "the wait was not held: the companion has ended"

        # (a) the mark
        assert not p.parked_on, (
            "the claim of a request waiting for room is still marked as parked on a resident "
            f"that is gone: parked_on={p.parked_on!r}, residents {list(mgr._residents)}"
        )
        # (b) governing
        assert p.governing_slot is slot, (
            "the claim of the request that needs room does not govern the victim designation "
            f"while it waits (governing slot is {'none' if p.governing_slot is None else 'another'}): "
            f"parked_on={p.parked_on!r}"
        )
        # (c) the time-to-live
        assert p.rearmed, (
            "the claim time-to-live was not re-armed after the claim was parked: deadline "
            f"{p.deadline!r} is not later than the park-time deadline {held.park_deadline!r}"
        )
        assert p.within_bound, (
            f"the claim deadline {p.deadline!r} is later than a full time-to-live from now"
        )
        assert p.live, "the re-armed claim is not live"
        assert not p.exempt_when_old, (
            "a copy of the claim with its deadline in the past is still treated as live: the "
            "claim is exempt from the time-to-live while it waits"
        )
        assert held.enqueued == [], "reading the claim re-enqueued the request"

        await release_companion(held)


# --------------------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------------------

async def test_control_a_claim_really_parked_in_a_live_inbox_still_stays_out_of_the_designation(tmp_path):
    """Control, green with or without the fix. A request really parked in a live resident's
    inbox (the holder is mid-turn, so the request is still in the inbox) keeps its claim marked
    as parked on that resident. Such a claim needs no room, so it is still left out when claims
    parked in an inbox are excluded, and it is still exempt from the time-to-live: a copy with
    its deadline in the past is live. Without the mark the same copy is expired, so the past
    deadline is really past. This is what the fix must not change."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r = await hold_and_park(world)
        assert claimant in list(r.inbox._queue), "setup: the request must be in the live inbox"
        claim = claim_of(world.mgr, claimant)
        assert claim.get("parked_on") == r.resident_key, (
            f"setup: the parked claim must be marked with the live resident: {claim.get('parked_on')!r}"
        )
        assert world.mgr._residents[r.resident_key] is r, "setup: the resident must be live"

        async with world.mgr._registry_lock:
            outside = world.mgr._governing_claim_locked(include_parked=False)
            everything = world.mgr._governing_claim_locked()
        assert everything is not None and everything[1] is claimant, (
            "setup: the claim must otherwise govern, or leaving it out proves nothing"
        )
        assert outside is None, "a claim parked in a live inbox was offered as the governing claim"

        old = dict(claim, ttl_deadline_monotonic=time.monotonic() - 1.0)
        assert world.mgr._claim_is_live(old) is None, (
            "a claim parked in a live inbox expired by the claim time-to-live"
        )
        unmarked = {k: v for k, v in old.items() if k != "parked_on"}
        assert world.mgr._claim_is_live(unmarked) == "ttl_expired", (
            "setup: the same claim without the mark must be expired, or the exemption proves nothing"
        )


async def test_instrument_control_the_probes_report_a_stale_mark_as_bad_news(tmp_path, monkeypatch):
    """Instrument control, green with or without the fix. The probes the main test reads are
    shown to be able to report the bad state. During the held wait the claim is given the stale
    mark by hand (the dead resident's key) and its deadline is set back to the park-time
    deadline by hand; the code under test is not called. The same probes then report the mark,
    no governing claim, a deadline that was not re-armed, and a past-deadline copy that is
    treated as live. Taking the mark off by hand and re-arming the deadline by hand flips every
    one of them, so none of the probes is stuck on one answer."""
    async with held_oom_wait(tmp_path, monkeypatch) as held:
        mgr, slot = held.mgr, held.slot
        claim = claim_of(mgr, slot)

        claim["parked_on"] = "m1"
        claim["ttl_deadline_monotonic"] = held.park_deadline
        bad = await read_probes(mgr, slot, held.park_deadline)
        assert bad.parked_on == "m1", "the probe does not see the stale mark"
        assert bad.governing_slot is None, "the probe does not see that a marked claim cannot govern"
        assert not bad.rearmed, "the probe reports a re-armed deadline for a deadline that was not re-armed"
        assert bad.exempt_when_old, "the probe does not see that a marked claim is exempt from the time-to-live"
        assert bad.live and bad.within_bound, "setup: the marked claim must still be live and in bounds"

        claim.pop("parked_on")
        claim["ttl_deadline_monotonic"] = time.monotonic() + manager_module._FASTLANE_CLAIM_TTL_S
        good = await read_probes(mgr, slot, held.park_deadline)
        assert not good.parked_on, "the probe still reports a mark after it was taken off"
        assert good.governing_slot is slot, "the probe does not see the unmarked claim govern"
        assert good.rearmed, "the probe does not see a re-armed deadline"
        assert not good.exempt_when_old, "the probe does not see that an unmarked claim can expire"
