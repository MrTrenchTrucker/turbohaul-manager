"""Further behaviours of a Fast Lane claim parked in a resident's inbox.

A request for an already-loaded model tag is handed to the resident's inbox. Its claim is
kept, marked as parked on that resident, until its own turn starts. Five small behaviours
around that, each checked here by its outcome:

1. A parked request that is re-queued to staging (by a resident teardown, or by the head
   re-queue helper the teardown uses) is a waiting request again: its claim is no longer
   marked as parked, the claim time-to-live runs again from that moment (it did not run
   while parked), and once it is past that time-to-live a claim sitting in staging expires
   like any other ("ttl_expired").
2. The same for the TAIL re-queue helper, called on its own so that nothing else can do the
   un-parking.
3. Registering a claim for a request that already owns one (the request comes back through
   the make-room path after waiting in an inbox) un-parks and re-arms that same claim and
   creates no second claim.
4. A holder whose turn ends while a higher-ranked request is parked in its inbox does not
   enter its grace window at all: the audit trail says "grace_designated_victim_skip", the
   holder's slot goes from ACTIVE straight to POPPED, and the parked request is served next
   on the same engine. With a lower-ranked parked request the holder does enter grace (the
   control).
5. A holder whose client asked for keep_alive 0 is normally unloaded right after its turn.
   When a higher-ranked request is parked in its inbox it is NOT unloaded: the parked request
   is served on the same engine, with no respawn. Without a parked request the same holder is
   still unloaded (the control).

The first group calls the re-queue and register helpers directly on a request that was parked
by the real dispatcher route. The second and third group run real manager code (real
dispatcher, drivers, grace loop, claim registry) with a fake engine process and a fake
completion call that blocks per request, so a test decides exactly when each turn ends. One
process, asyncio only, at most three requests per test, bounded waits, no sleep above one
second. The engine's slot save is replaced by a recorder so no request ever leaves the process.
"""

import asyncio
import contextlib
import time

import pytest
from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    ranked_rules,
    resident_for,
    seed_manifest,
    wait_until,
)

from turbohaul import manager as manager_module
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotState

MODEL = "m1"

# Priority order: index 0 outranks index 1, which outranks index 2.
CLAIMANT_IP = "10.0.0.4"
HOLDER_IP = "10.0.0.5"
LOWER_IP = "10.0.0.6"
RULES = ranked_rules((CLAIMANT_IP, 1), (HOLDER_IP, 1), (LOWER_IP, 1))

HOLDER_THREAD = "thread-holder"
CLAIMANT_THREAD = "thread-claimant"

# Bounds for things that must happen promptly or after a short grace window; not delays.
PROMPT_S = 1.5
AFTER_GRACE_S = 8.0

# A claim time-to-live short enough to run out inside one test, and a sleep that outlasts it.
SHORT_TTL_S = 0.3
PAST_TTL_S = 0.5


def meta(ip, **extra):
    out = {"ip": ip, "is_main": True}
    out.update(extra)
    return out


class World:
    """One manager, one box slot, a fake engine whose turns end only when told to.

    ``send`` submits a request through the real ``submit`` and labels it. The fake completion
    call records the order in which turns start and the engine handle each turn ran on, then
    blocks until the test releases that request. ``audits`` records every audit event the
    manager writes (slot id, event type) and ``transitions`` every state change of a slot.
    """

    def __init__(self, mgr, spawns):
        self.mgr = mgr
        self.spawns = spawns
        self.slots = {}          # label -> Slot
        self.labels = {}         # slot_id -> label
        self.starts = []         # slot_ids in the order their turns started
        self.handles = {}        # slot_id -> engine handle the turn ran on
        self.unloads = []        # resident keys passed to _begin_unload_locked
        self.audits = []         # (slot_id, event_type) in the order the manager wrote them
        self.transitions = []    # (slot_id, from_state, to_state)
        self._gates = {}
        self._open = False

    # -- fake completion -------------------------------------------------
    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def complete(self, slot, handle):
        self.starts.append(slot.slot_id)
        self.handles[slot.slot_id] = handle
        if not self._open:
            await self.gate(slot.slot_id).wait()
        return {"ok": True, "model": handle.model_tag}

    def release(self, label):
        self.gate(self.slots[label].slot_id).set()

    def open_everything(self):
        self._open = True
        for ev in self._gates.values():
            ev.set()

    # -- requests --------------------------------------------------------
    async def send(self, label, ip, thread, **extra_meta):
        slot = await self.mgr.submit(
            MODEL, prompt=label, thread_id=thread, client_meta=meta(ip, **extra_meta),
            wait_for_completion=True,
        )
        self.slots[label] = slot
        self.labels[slot.slot_id] = label
        return slot

    def order(self):
        """Labels of the turns that have started, in start order."""
        return [self.labels.get(s, s) for s in self.starts]

    async def started(self, label, timeout=PROMPT_S):
        """True when ``label``'s turn starts within ``timeout`` seconds."""
        try:
            await wait_until(lambda: label in self.order(), timeout=timeout)
        except AssertionError:
            return False
        return True

    async def parked(self, label, timeout=PROMPT_S):
        """Wait until ``label`` sits in the resident's inbox (the real HIT path put it there)."""
        slot = self.slots[label]
        r = resident_for(self.mgr, MODEL)
        await wait_until(lambda: r.inbox.qsize() >= 1 and slot not in self.mgr.queue._staging,
                         timeout=timeout)
        return r

    # -- single-stepping the dispatcher ----------------------------------
    async def pause_dispatcher(self):
        """Stop the dispatcher task (the resident's driver keeps running)."""
        task = self.mgr._worker_task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def dispatch_once(self):
        """One iteration of the dispatcher's own loop body: pop, then route (real HIT path)."""
        mgr = self.mgr
        slot = await mgr.queue.pop_next(
            warm_model_tag=mgr._dispatch_warm_hint(),
            fastlane_policy=mgr._fastlane_pop_kwarg(),
            grace_active=await mgr._grace_active_exclusions(),
            room_available=mgr._dispatch_room_hint(),
        )
        assert slot is not None, "the dispatcher found nothing to pop"
        with mgr._reserving(slot.model_tag):
            await mgr._route_or_reserve(slot)
        return slot

    async def drain_background(self):
        """Let the fire-and-forget tasks (claim events, re-queues) finish."""
        for _ in range(20):
            pending = [t for t in list(self.mgr._bg_tasks) if not t.done()]
            if not pending:
                return
            await asyncio.wait(pending, timeout=0.25)
        raise AssertionError("background tasks did not settle")

    # -- claims and records ----------------------------------------------
    def claims_of(self, label):
        """Every registered claim whose request is ``label`` (a plain read, never prunes)."""
        slot = self.slots[label]
        return [c for c in self.mgr._fastlane_claims.values() if c["slot"] is slot]

    def audit_events(self, label):
        sid = self.slots[label].slot_id
        return [ev for s, ev in self.audits if s == sid]

    def slot_transitions(self, label):
        sid = self.slots[label].slot_id
        return [(f, t) for s, f, t in self.transitions if s == sid]


@contextlib.asynccontextmanager
async def build_world(tmp_path, monkeypatch, *, grace_seconds=2, max_grace_extensions=50,
                      idle_hot_load_seconds=600):
    """A manager with ONE box slot and the dispatcher running, over three ranked clients."""
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=RULES, max_parallel_sidecars=1, grace_seconds=grace_seconds,
        max_grace_extensions=max_grace_extensions, idle_hot_load_seconds=idle_hot_load_seconds,
    )
    seed_manifest(boot, MODEL, main_gpu=0)
    spawns = []
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})

    def counting_spawn(binary, gguf, port, model_tag, argv, **kw):
        spawns.append(model_tag)
        return spawn_fn(binary, gguf, port, model_tag, argv, **kw)

    holder = {}

    async def complete(slot, handle):
        return await holder["world"].complete(slot, handle)

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=counting_spawn, health_fn=health_fn,
        sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete,
    )
    world = World(mgr, spawns)
    holder["world"] = world

    real_unload = mgr._begin_unload_locked
    real_audit = mgr._audit_async
    real_transition = manager_module.transition

    def spy_unload(r):
        world.unloads.append(r.resident_key)
        return real_unload(r)

    async def spy_audit(slot, event_type):
        world.audits.append((slot.slot_id, event_type))
        return await real_audit(slot, event_type)

    def spy_transition(slot, to_state):
        world.transitions.append((slot.slot_id, slot.state, to_state))
        return real_transition(slot, to_state)

    async def fake_save(port, model_tag, slot=None, **kwargs):
        return True

    mgr._begin_unload_locked = spy_unload
    mgr._audit_async = spy_audit
    mgr._save_slot_kv = fake_save
    monkeypatch.setattr(manager_module, "transition", spy_transition)

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            yield world
    finally:
        world.open_everything()
        await mgr.shutdown()


async def park_with_dispatcher_paused(world):
    """The holder's turn is running, then the dispatcher is stopped and driven by hand.

    The claimant is submitted (it lands in staging with its claim) and one dispatcher
    iteration routes it through the real HIT path into the resident's inbox. Nothing else
    moves the request afterwards. Returns (holder slot, claimant slot, resident, claim); the
    claim is checked to be marked as parked on that resident.
    """
    h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
    assert await world.started("H1", timeout=5.0), "setup: the holder's turn never started"
    await world.pause_dispatcher()
    claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
    assert claimant in world.mgr.queue._staging, "setup: the request must be queued, not routed"
    await world.dispatch_once()
    r = resident_for(world.mgr, MODEL)
    assert r.inbox.qsize() == 1 and claimant not in world.mgr.queue._staging, (
        "setup: the request must now sit in the resident's inbox"
    )
    assert r.active_slot is h1
    claims = world.claims_of("C")
    assert len(claims) == 1, "setup: a parked request holds exactly one claim"
    assert claims[0].get("parked_on") == r.resident_key, (
        "setup: the claim must be marked as parked on the resident the request waits on "
        f"(parked_on={claims[0].get('parked_on')!r})"
    )
    return h1, claimant, r, claims[0]


async def age_past_the_ttl_while_parked(world, claim):
    """Let the parked request outlive the (short) claim time-to-live; it must still be live."""
    await asyncio.sleep(PAST_TTL_S)
    assert time.monotonic() > claim["ttl_deadline_monotonic"], "setup: the claim did not age past its TTL"
    assert world.mgr._claim_is_live(claim) is None, (
        "a parked claim was expired by the time-to-live while it was parked"
    )


async def assert_waiting_claim_again(world, claim, r, t_before):
    """``claim`` is an ordinary waiting claim again: un-parked, time-to-live re-armed, and
    expiring like any other claim of a request in staging once that time-to-live is past."""
    assert "parked_on" not in claim, (
        f"the claim is still marked as parked on {claim.get('parked_on')!r} after the request "
        "was taken out of the inbox"
    )
    t_after = time.monotonic()
    deadline = claim["ttl_deadline_monotonic"]
    assert t_before + SHORT_TTL_S <= deadline <= t_after + SHORT_TTL_S, (
        "the claim time-to-live was not re-armed to about now + the claim time-to-live: "
        f"deadline={deadline:.3f} now in [{t_before:.3f}, {t_after:.3f}] ttl={SHORT_TTL_S}"
    )
    assert world.mgr._claim_is_live(claim) is None, "the freshly re-armed claim is already dead"
    await asyncio.sleep(PAST_TTL_S)
    assert world.mgr._claim_is_live(claim) == "ttl_expired", (
        "a claim of a request sitting in staging past its time-to-live did not expire"
    )


# --------------------------------------------------------------------------------------
# 1-3: a parked claim becomes a waiting claim again when the request leaves the inbox
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("how", ["head_requeue_helper", "resident_teardown"])
async def test_a_requeued_parked_request_holds_an_ordinary_waiting_claim_again(
    tmp_path, monkeypatch, how,
):
    """The request was parked in a resident's inbox with its claim kept (the claim time-to-live
    is short here and runs out while it is parked without expiring the claim). The request is
    then re-queued to staging at the head, either by calling the head re-queue helper on the
    drained inbox or by tearing the resident down (which drains and re-queues the inbox). From
    that moment it is a waiting request like any other: its claim must no longer be marked as
    parked, its time-to-live must be re-armed to about now + the time-to-live, and once that has
    passed the claim of a request sitting in staging expires with the reason "ttl_expired". A
    claim that stayed marked as parked would never expire: a zombie that outlives a request
    that nobody serves."""
    monkeypatch.setattr(manager_module, "_FASTLANE_CLAIM_TTL_S", SHORT_TTL_S)
    async with build_world(tmp_path, monkeypatch) as world:
        _h1, claimant, r, claim = await park_with_dispatcher_paused(world)
        await age_past_the_ttl_while_parked(world, claim)

        t_before = time.monotonic()
        if how == "head_requeue_helper":
            drained = [r.inbox.get_nowait()]
            await world.mgr._requeue_slots_or_fail(drained)
        else:
            async with world.mgr._registry_lock:
                world.mgr._begin_unload_locked(r)
            await world.drain_background()
        assert claimant in world.mgr.queue._staging, "setup: the re-queue must put it back in staging"
        assert world.claims_of("C") == [claim], "setup: the request must still hold that one claim"
        await assert_waiting_claim_again(world, claim, r, t_before)


async def test_the_tail_requeue_alone_makes_a_parked_claim_a_waiting_claim_again(
    tmp_path, monkeypatch,
):
    """Same as above, for the TAIL re-queue helper (the path a displaced resident's own backlog
    takes), called by itself on the request drained from the inbox so that no other code path
    (a dispatcher pass, a register call) can do the un-parking. Right after the call the claim
    must be un-parked and its time-to-live re-armed, and it expires in staging like any other."""
    monkeypatch.setattr(manager_module, "_FASTLANE_CLAIM_TTL_S", SHORT_TTL_S)
    async with build_world(tmp_path, monkeypatch) as world:
        _h1, claimant, r, claim = await park_with_dispatcher_paused(world)
        await age_past_the_ttl_while_parked(world, claim)

        t_before = time.monotonic()
        drained = [r.inbox.get_nowait()]
        await world.mgr._requeue_slots_tail_or_fail(drained)
        assert claimant in world.mgr.queue._staging, "setup: the tail re-queue must put it in staging"
        assert world.claims_of("C") == [claim], "setup: the request must still hold that one claim"
        await assert_waiting_claim_again(world, claim, r, t_before)


async def test_registering_again_for_a_parked_request_unparks_its_own_claim_and_adds_none(
    tmp_path, monkeypatch,
):
    """A request that waited in an inbox and then comes back through the make-room path is
    registered again (``_register_fastlane_claim_locked`` for the same request). The claim it
    already owns must become a waiting claim again: no longer marked as parked, its
    time-to-live re-armed. It is the SAME claim (the registry still holds exactly one claim for
    the request, the very same record, and the same number of claims overall): a second claim
    would double count the request."""
    monkeypatch.setattr(manager_module, "_FASTLANE_CLAIM_TTL_S", SHORT_TTL_S)
    async with build_world(tmp_path, monkeypatch) as world:
        _h1, claimant, r, claim = await park_with_dispatcher_paused(world)
        await age_past_the_ttl_while_parked(world, claim)

        r.inbox.get_nowait()  # the request leaves the inbox, as it does before it is deferred
        claims_before = len(world.mgr._fastlane_claims)
        t_before = time.monotonic()
        async with world.mgr._registry_lock:
            world.mgr._register_fastlane_claim_locked(claimant, "make_room_starved_vram")
        assert world.claims_of("C") == [claim] and world.claims_of("C")[0] is claim, (
            "registering again must keep the one existing claim record, not add or replace one"
        )
        assert len(world.mgr._fastlane_claims) == claims_before, (
            "registering again for a request that owns a claim changed the number of claims"
        )
        await assert_waiting_claim_again(world, claim, r, t_before)


# --------------------------------------------------------------------------------------
# 4: a holder with a parked outranker does not enter grace
# --------------------------------------------------------------------------------------

async def test_a_holder_with_a_parked_outranker_skips_grace_and_the_parked_request_is_served_next(
    tmp_path, monkeypatch,
):
    """The holder's turn ends while a higher-ranked request is parked in its inbox. The holder
    must not enter its grace window at all, not even for an instant: its audit trail has
    "grace_designated_victim_skip" and neither "grace_enter" nor the mid-window
    "grace_designated_unload_target_break", and its slot goes from ACTIVE straight to POPPED
    without ever being in GRACE. The parked request is then served next on the same engine
    (same handle, no unload, no respawn). Whether the window is entered and left again within
    one poll is not visible from the order of turns, so this reads the audit trail and the
    state machine."""
    async with build_world(tmp_path, monkeypatch) as world:
        h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=5.0)
        claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
        r = await world.parked("C")
        assert r.active_slot is h1 and claimant.state is SlotState.STAGED

        world.release("H1")
        assert await world.started("C", timeout=AFTER_GRACE_S), (
            f"the parked request was not served; order {world.order()}"
        )
        events = world.audit_events("H1")
        assert "grace_designated_victim_skip" in events, (
            f"the holder did not skip grace for the parked outranker; audit {events}"
        )
        assert "grace_enter" not in events and "grace_designated_unload_target_break" not in events, (
            f"the holder entered its grace window before giving it up; audit {events}"
        )
        moves = world.slot_transitions("H1")
        assert not any(to is SlotState.GRACE for _frm, to in moves), (
            f"the holder's slot was in GRACE; transitions {moves}"
        )
        assert (SlotState.ACTIVE, SlotState.POPPED) in moves, f"transitions {moves}"
        assert world.order() == ["H1", "C"]
        assert world.handles[claimant.slot_id] is world.handles[h1.slot_id]
        assert world.spawns == [MODEL] and world.unloads == [], (
            f"spawns={world.spawns} unloads={world.unloads}"
        )


async def test_control_a_holder_with_a_lower_ranked_parked_request_still_enters_grace(
    tmp_path, monkeypatch,
):
    """Control for the test above: when the request parked in the holder's inbox is LOWER
    ranked, the holder enters its grace window as usual (audit "grace_enter", slot in GRACE) and
    does not skip it. It shows that the audit and state reads of the test above can see a grace
    entry, and that the skip is not applied to every parked request."""
    async with build_world(tmp_path, monkeypatch, grace_seconds=1) as world:
        h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=5.0)
        await world.send("L", LOWER_IP, "thread-lower")
        await world.parked("L")
        world.release("H1")
        await wait_until(lambda: "grace_enter" in world.audit_events("H1"), timeout=PROMPT_S)
        events = world.audit_events("H1")
        assert "grace_designated_victim_skip" not in events, f"audit {events}"
        assert any(to is SlotState.GRACE for _frm, to in world.slot_transitions("H1"))
        assert h1.state in (SlotState.GRACE, SlotState.POPPED)


# --------------------------------------------------------------------------------------
# 5: keep_alive 0 does not unload a holder that has a parked outranker
# --------------------------------------------------------------------------------------

async def test_a_keep_alive_zero_holder_is_not_unloaded_while_an_outranker_is_parked(
    tmp_path, monkeypatch,
):
    """The holder's client asked for keep_alive 0, which unloads the resident as soon as the turn
    ends (see the control below). A higher-ranked request is parked in the holder's inbox. At the
    holder's turn boundary the resident must NOT be unloaded: the parked request is served next
    on the same resident object and the same engine handle, with exactly one spawn and no
    teardown. An unload there would push the parked request back to staging and pay a full
    respawn of a model that is already loaded."""
    async with build_world(tmp_path, monkeypatch) as world:
        h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD, keep_alive_s=0)
        assert await world.started("H1", timeout=5.0)
        claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
        r = await world.parked("C")
        assert r.active_slot is h1 and r.latest_keep_alive_s == 0, "setup: the holder asked for keep_alive 0"

        world.release("H1")
        assert await world.started("C", timeout=AFTER_GRACE_S), (
            f"the parked request was not served; order {world.order()} unloads {world.unloads}"
        )
        assert world.unloads == [], f"the resident was unloaded despite a parked outranker: {world.unloads}"
        assert resident_for(world.mgr, MODEL) is r and r.state is not ResidentState.DEAD
        assert world.spawns == [MODEL], f"the model was respawned: {world.spawns}"
        assert world.handles[claimant.slot_id] is world.handles[h1.slot_id], (
            "the parked request was not served on the holder's engine"
        )


async def test_control_a_keep_alive_zero_holder_with_nothing_parked_is_still_unloaded(
    tmp_path, monkeypatch,
):
    """Control for the test above, unchanged behaviour: the same keep_alive 0 holder with nothing
    parked in its inbox is unloaded after its turn (and its short grace window), so the guard
    against unloading is not a blanket stop."""
    async with build_world(tmp_path, monkeypatch, grace_seconds=1) as world:
        await world.send("H1", HOLDER_IP, HOLDER_THREAD, keep_alive_s=0)
        assert await world.started("H1", timeout=5.0)
        r = resident_for(world.mgr, MODEL)
        world.release("H1")
        await wait_until(lambda: r.resident_key in world.unloads, timeout=AFTER_GRACE_S)
        await wait_until(lambda: r.state is ResidentState.DEAD, timeout=PROMPT_S)
        assert world.spawns == [MODEL]
