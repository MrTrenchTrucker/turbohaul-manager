"""A request parked in a resident's inbox keeps its Fast Lane claim until its own turn starts.

When a request for an already-loaded model tag arrives, the dispatcher hands it to the
resident's inbox (the HIT path of ``_route_or_reserve``). The holder of that resident is
usually mid-turn, or sitting in its grace window waiting for a same-thread follow-up.
The parked request must stay visible to the Fast Lane machinery until the turn that
serves it actually starts:

* the claim stays registered (and listed in the claims snapshot) while the request is parked;
* when the parked request outranks the holder, the holder loses its grace at the turn
  boundary and the parked request is served next on the SAME resident (no unload, no
  respawn, no draining of the inbox), never in the middle of a running turn;
* the holder's own follow-up is not dropped: it waits its turn by rank and is served after;
* an equal or lower ranked parked request must not jump the holder's grace follow-ups;
* the claim is released at the start of the request's own turn, and also when the client
  disconnects, the future fails, or the resident goes away and the request cannot be re-queued;
* a request re-queued from a torn-down resident's inbox is waiting again and keeps its claim;
* a parked claim needs no room, so it must not widen to unrelated residents of other models;
* the resident's rank identity names the client of the CURRENT turn, so it is not stamped
  with the parked request's client while the holder is still mid-turn.

Everything runs real manager code (real dispatcher, drivers, grace loop, claim registry)
with a fake engine process and a fake completion call that blocks per request, so a test
decides exactly when each turn ends. One process, asyncio only, bounded waits.
"""

import asyncio
import contextlib

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
OTHER_THREAD = "thread-other"

# How long a test waits for something that must happen promptly, and for something
# that legitimately waits behind a grace window. Both are bounds, not delays.
PROMPT_S = 1.5
AFTER_GRACE_S = 8.0


def meta(ip):
    return {"ip": ip, "is_main": True}


class World:
    """One manager, one box slot, a fake engine whose turns end only when told to.

    ``send`` submits a request through the real ``submit`` and labels it. The fake
    completion call records the order in which turns start (``starts``), the engine
    handle each turn ran on, and whether the request's claim was still registered at
    the moment its turn began, then blocks until the test releases that request.
    """

    def __init__(self, mgr, spawns):
        self.mgr = mgr
        self.spawns = spawns
        self.slots = {}          # label -> Slot
        self.labels = {}         # slot_id -> label
        self.starts = []         # slot_ids in the order their turns started
        self.handles = {}        # slot_id -> engine handle the turn ran on
        self.claim_at_start = {}  # slot_id -> claim still registered when the turn began
        self.releases = []       # (slot_id, reason, {label: slot state}) per claim actually removed
        self.unloads = []        # resident keys passed to _begin_unload_locked
        self.tail_requeues = []  # slot_ids handed to the tail re-queue helper
        self._gates = {}
        self._open = False

    # -- fake completion -------------------------------------------------
    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def complete(self, slot, handle):
        self.starts.append(slot.slot_id)
        self.handles[slot.slot_id] = handle
        self.claim_at_start[slot.slot_id] = any(
            c["slot"] is slot for c in self.mgr._fastlane_claims.values()
        )
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
    async def send(self, label, ip, thread, **kw):
        slot = await self.mgr.submit(
            MODEL, prompt=label, thread_id=thread, client_meta=meta(ip),
            wait_for_completion=True, **kw,
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

    async def next_start(self, timeout=PROMPT_S):
        """Wait (bounded) until a second turn has started, whoever it belongs to."""
        with contextlib.suppress(AssertionError):
            await wait_until(lambda: len(self.starts) >= 2, timeout=timeout)

    async def parked(self, label, timeout=PROMPT_S):
        """Wait until ``label`` sits in the resident's inbox (the real HIT path put it there)."""
        slot = self.slots[label]
        r = resident_for(self.mgr, MODEL)
        await wait_until(lambda: r.inbox.qsize() >= 1 and slot not in self.mgr.queue._staging,
                         timeout=timeout)
        return r

    async def turn_done(self, label, timeout=PROMPT_S):
        """Wait until ``label``'s turn has completed (it left ACTIVE)."""
        slot = self.slots[label]
        await wait_until(
            lambda: slot.state in (SlotState.GRACE, SlotState.POPPED), timeout=timeout,
        )

    # -- single-stepping the dispatcher ----------------------------------
    async def pause_dispatcher(self):
        """Stop the dispatcher task (the resident's driver keeps running).

        Afterwards a request submitted with ``send`` stays in staging until
        ``dispatch_once`` routes it, so a test can look at the state between "queued" and
        "routed" and between "routed" and "turn started".
        """
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

    def subscribe(self):
        """A real subscriber queue on the event bus; returns a drain function."""
        q = asyncio.Queue()
        self.mgr.event_bus.subscribe(q)

        async def drain():
            await self.drain_background()
            out = []
            while not q.empty():
                out.append(q.get_nowait())
            return out

        return drain

    # -- claims ----------------------------------------------------------
    def claim_slots(self):
        """Slots holding a registered claim right now (a plain read, never prunes)."""
        return [c["slot"] for c in self.mgr._fastlane_claims.values()]

    def holds_claim(self, slot):
        return slot in self.claim_slots()

    def has_claim(self, label):
        return self.holds_claim(self.slots[label])

    def snapshot_ids(self):
        """Slot ids the claims snapshot lists (the surface the queue tab reads)."""
        return [row["slot_id"] for row in self.mgr.fastlane_claims_snapshot()]


@contextlib.asynccontextmanager
async def build_world(tmp_path, *, grace_seconds=2, max_grace_extensions=50,
                      idle_hot_load_seconds=600, model_split_mode="none",
                      max_parallel_sidecars=1):
    """A manager with ONE box slot and the dispatcher running, over three ranked clients."""
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=RULES, max_parallel_sidecars=max_parallel_sidecars,
        grace_seconds=grace_seconds,
        max_grace_extensions=max_grace_extensions, idle_hot_load_seconds=idle_hot_load_seconds,
    )
    seed_manifest(boot, MODEL, split_mode=model_split_mode, main_gpu=0)
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

    real_release = mgr._release_fastlane_claim_locked
    real_unload = mgr._begin_unload_locked
    real_tail = mgr._requeue_slots_tail_or_fail

    def spy_release(slot, reason):
        """Record a release only when it actually removed the slot's claim."""
        held = world.holds_claim(slot)
        states = {lbl: s.state for lbl, s in world.slots.items()}
        out = real_release(slot, reason)
        if held and not world.holds_claim(slot):
            world.releases.append((slot.slot_id, reason, states))
        return out

    def spy_unload(r):
        world.unloads.append(r.resident_key)
        return real_unload(r)

    async def spy_tail(slots):
        world.tail_requeues.extend(s.slot_id for s in slots)
        return await real_tail(slots)

    mgr._release_fastlane_claim_locked = spy_release
    mgr._begin_unload_locked = spy_unload
    mgr._requeue_slots_tail_or_fail = spy_tail

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            yield world
    finally:
        world.open_everything()
        await mgr.shutdown()


def assert_parked_claim_held(world):
    """The parked request "C" still holds its claim."""
    assert world.has_claim("C"), "a parked request must hold its claim; it was released when it was parked"


async def hold_and_park(world, claimant_label, claimant_ip, claimant_thread, **kw):
    """The defect shape: the holder is mid-turn, and a request is parked in its inbox.

    The holder's first turn is running (its fake completion is blocked), then the
    second request is submitted through the real ``submit``; the dispatcher routes it to
    the resident's inbox through the real HIT path. Returns (holder slot, claimant slot,
    resident).
    """
    h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
    assert await world.started("H1", timeout=5.0), "setup: the holder's turn never started"
    claimant = await world.send(claimant_label, claimant_ip, claimant_thread, **kw)
    r = await world.parked(claimant_label)
    assert r.state is ResidentState.ACTIVE and r.active_slot is h1, (
        "setup: the holder must still be mid-turn"
    )
    assert claimant.state is SlotState.STAGED, "setup: a parked request is still queued"
    return h1, claimant, r


# --------------------------------------------------------------------------------------
# (a) the holder's fast follow-ups do not starve a parked higher-ranked request
# --------------------------------------------------------------------------------------

async def test_holder_follow_ups_do_not_starve_a_parked_higher_ranked_request(tmp_path):
    """A lower-ranked holder sends same-thread follow-ups while a higher-ranked request sits
    parked in its inbox. The parked request's turn must start at the holder's NEXT turn
    boundary, before any further warm follow-up of the holder; the holder's running turn is
    never cut; and the holder's follow-up is still served afterwards (re-queued, not
    dropped). If the parked request stayed invisible, the holder's grace window would keep
    serving its follow-ups first and the order below would be H1, H2."""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(world, "C", CLAIMANT_IP, CLAIMANT_THREAD)

        # Never mid-turn: with the claimant parked and the holder's turn running, nothing
        # else starts and the holder's turn is not completed for it.
        await asyncio.sleep(0.3)
        assert world.order() == ["H1"], f"a turn started while the holder was mid-turn: {world.order()}"
        assert not h1.completion_future.done(), "the holder's running turn was cut"

        # The holder's turn ends; its same-thread follow-up arrives right behind it.
        world.release("H1")
        await world.turn_done("H1")
        h2 = await world.send("H2", HOLDER_IP, HOLDER_THREAD)

        await world.next_start()
        assert world.order()[:2] == ["H1", "C"], (
            f"turn start order was {world.order()}: the holder's warm follow-up (H2) was served "
            "ahead of the higher-ranked request parked in its inbox"
        )
        assert h1.completion_future.done() and h1.completion_future.result()["ok"], (
            "the holder's own turn must complete normally, not be cut"
        )
        assert "H2" not in world.order(), "the follow-up started while the claimant's turn was running"

        # The holder's follow-up is not stranded: after the claimant's turn (and its own
        # grace) the follow-up is served and resolves.
        world.release("C")
        assert await world.started("H2", timeout=AFTER_GRACE_S), (
            f"the holder's follow-up was never served: {world.order()}"
        )
        world.release("H2")
        done, _ = await asyncio.wait({h2.completion_future}, timeout=PROMPT_S)
        assert done and h2.completion_future.result()["ok"], "the follow-up's future did not resolve"


async def test_claimants_turn_starts_through_the_boundary_not_a_timer(tmp_path, monkeypatch):
    """Same shape as above, but every timer that could advance a waiting request on its own is
    made unusable inside the bounded wait: the dispatcher's backstop poll, the deferred-request
    backoff, the grace window (600 s) and the idle timeout (1 h). The parked request must
    still start within a short bounded wait of the holder's turn ending, so it can only have
    been started by the holder's turn-boundary decision. (The grace loop's own 50 ms poll is a
    literal in the loop and is not patched; it is not involved here because the request is
    parked before the holder's turn ends, so there is no grace loop to poll.)"""
    for name in ("_DISPATCH_DEFER_BACKOFF_S", "_VRAM_DEFER_BACKOFF_S", "_SAME_MODEL_QUEUED_HOLD_S"):
        monkeypatch.setattr(manager_module, name, 600.0)
    async with build_world(tmp_path, grace_seconds=600, idle_hot_load_seconds=3600) as world:
        await hold_and_park(world, "C", CLAIMANT_IP, CLAIMANT_THREAD)
        world.release("H1")
        assert await world.started("C", timeout=PROMPT_S), (
            f"the parked higher-ranked request did not start within {PROMPT_S}s of the holder's "
            f"turn ending (order {world.order()}); with every timer disabled only the turn-boundary "
            "decision can start it, and the holder sat in its 600 s grace window instead"
        )


async def test_a_request_parked_during_the_holders_grace_window_ends_that_window(tmp_path):
    """The request arrives AFTER the holder's turn ended, while the holder sits in its grace
    window (600 s here) waiting for a same-thread follow-up. It is parked through the real HIT
    path; the holder's grace loop must notice that it is now outranked and give up its window,
    so the parked request starts within a short bounded wait instead of after the window.

    On unchanged code this is not reliably red: for a few milliseconds between the request's
    submit and its being parked its claim is registered and the request is only queued, and
    if the grace loop's 50 ms poll happens to land in that gap it ends the window for that
    reason (the queued-claim designation), after which the request is served. Most runs miss
    the gap and wait out the window."""
    async with build_world(tmp_path, grace_seconds=600) as world:
        await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=5.0)
        world.release("H1")
        await world.turn_done("H1")
        r = resident_for(world.mgr, MODEL)
        await wait_until(lambda: r.in_grace_loop, timeout=PROMPT_S)

        await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
        # Not waited on as "parked": once the holder gives up its window the request leaves the
        # inbox within milliseconds, so the parked state is only reported if it is still there.
        started = await world.started("C", timeout=PROMPT_S)
        assert started, (
            f"the higher-ranked request did not start within {PROMPT_S}s although the holder was "
            f"only waiting in its grace window (600 s); order {world.order()}, "
            f"requests parked in the inbox: {r.inbox.qsize()}"
        )


async def test_same_resident_swap_does_not_unload(tmp_path):
    """The parked request is served by the SAME resident: the same engine handle runs both
    turns, the resident is not torn down, no second engine is spawned, nothing is unloaded and
    the resident's inbox is not drained to the staging tail. Serving a higher-ranked client of
    the very model the holder has loaded needs no room, so unloading the holder would only
    cost a reload. (On unchanged code this fails at the first assertion for the same reason as
    the starvation test: the parked request never starts. The checks after it guard the obvious
    wrong fix, which would designate the holder as an unload victim.)"""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(world, "C", CLAIMANT_IP, CLAIMANT_THREAD)
        world.release("H1")
        assert await world.started("C", timeout=PROMPT_S), (
            f"the parked request never started (order {world.order()})"
        )
        assert world.handles[claimant.slot_id] is world.handles[h1.slot_id], (
            "the parked request ran on a different engine than the holder"
        )
        assert world.mgr._residents.get(r.resident_key) is r and resident_for(world.mgr, MODEL) is r
        assert r.state is not ResidentState.DEAD and not r.torn_down, "the resident was torn down"
        assert world.spawns == [MODEL], f"an engine was (re)spawned: {world.spawns}"
        assert world.unloads == [], f"an unload was started: {world.unloads}"
        assert world.tail_requeues == [], (
            f"the resident's inbox was drained to the staging tail: {world.tail_requeues}"
        )


async def holder_follow_up_goes_first(tmp_path, ip, kind):
    """A request of the client ``ip`` is parked behind the holder's running turn; the holder's
    turn ends and its same-thread follow-up arrives. The follow-up must be served first, and
    the parked request afterwards."""
    async with build_world(tmp_path) as world:
        await hold_and_park(world, "C", ip, OTHER_THREAD)
        world.release("H1")
        await world.turn_done("H1")
        await world.send("H2", HOLDER_IP, HOLDER_THREAD)
        await world.next_start()
        assert world.order()[:2] == ["H1", "H2"], (
            f"{kind}-ranked parked request jumped the holder's grace follow-up: {world.order()}"
        )
        world.release("H2")
        assert await world.started("C", timeout=AFTER_GRACE_S), (
            f"the {kind}-ranked parked request was never served: {world.order()}"
        )


async def test_control_equal_rank_parked_request_does_not_jump_grace(tmp_path):
    """Control. A parked request of EQUAL rank (same client rule, another thread) must not take
    the holder's turn boundary: the holder's warm same-thread follow-up is still served first
    (grace is unconditional unless a strictly higher-ranked request waits), and the parked
    request is served afterwards, never dropped. A change that dropped the rank comparison and
    let any parked request cut the holder's grace would fail this."""
    await holder_follow_up_goes_first(tmp_path, HOLDER_IP, "equal")


async def test_lower_rank_parked_request_does_not_jump_grace(tmp_path):
    """Same as the equal-rank control, with a parked request of LOWER rank. Required behaviour
    either way: the holder's follow-up first, the parked request afterwards.

    On unchanged code this is order-dependent rather than reliably green: parking stamps the
    resident with the parked client's identity while the holder is still mid-turn, so the
    holder is mistaken for the lower-ranked client, its own follow-up's claim then designates
    it, and its grace stops protecting that follow-up (whether the follow-up or the parked
    request starts first depends on which of two tasks runs first)."""
    await holder_follow_up_goes_first(tmp_path, LOWER_IP, "lower")


async def test_parked_claimant_stays_in_the_claim_snapshot(tmp_path):
    """While the request is parked (the holder is mid-turn) the claims snapshot, the surface
    the queue tab reads, still lists it, and so does the claim registry. When the request's own
    turn starts the claim is gone. On unchanged code the claim is released the moment the
    request is put in the inbox."""
    async with build_world(tmp_path) as world:
        _h1, claimant, _r = await hold_and_park(world, "C", CLAIMANT_IP, CLAIMANT_THREAD)
        assert claimant.slot_id in world.snapshot_ids(), (
            "the parked request is missing from the claims snapshot while it waits in the inbox"
        )
        assert world.has_claim("C")
        await asyncio.sleep(0.3)
        assert claimant.slot_id in world.snapshot_ids(), "the claim vanished while the request was parked"

        world.release("H1")
        assert await world.started("C", timeout=PROMPT_S)
        assert claimant.slot_id not in world.snapshot_ids(), (
            "the claim is still listed after the request's own turn started"
        )
        assert not world.has_claim("C")


async def test_claim_is_released_exactly_when_the_turn_starts(tmp_path):
    """No release while the request is parked (so none while the holder's turn runs); one
    "admitted" release once the holder's turn is over and the request's own turn is taking
    the slot, so the claim is gone by the time the request's turn is actually running; the
    admitted event reaches the event bus at that point and not before."""
    async with build_world(tmp_path) as world:
        drain = world.subscribe()
        _h1, claimant, _r = await hold_and_park(world, "C", CLAIMANT_IP, CLAIMANT_THREAD)
        early = [rel for rel in world.releases if rel[0] == claimant.slot_id]
        assert early == [], (
            f"the claim was released while the request was only parked, holder mid-turn: {early}"
        )
        before = [e for e in await drain() if e.get("slot_id") == claimant.slot_id]
        assert "fastlane_admitted" not in [e["event"] for e in before]

        world.release("H1")
        assert await world.started("C", timeout=PROMPT_S)
        assert world.claim_at_start[claimant.slot_id] is False, (
            "the claim was still registered when the request's own turn began"
        )
        mine = [rel for rel in world.releases if rel[0] == claimant.slot_id]
        assert [rel[1] for rel in mine] == ["admitted"], mine
        holder_state = mine[0][2]["H1"]
        assert holder_state in (SlotState.GRACE, SlotState.POPPED), (
            f"the claim was released while the holder's turn was still {holder_state}"
        )
        events = [e for e in await drain() if e.get("slot_id") == claimant.slot_id]
        assert [e["event"] for e in events].count("fastlane_admitted") == 1, events


async def test_rank_stamp_is_not_overwritten_at_park(tmp_path):
    """The resident's rank identity names the client of the turn that is running. While the
    holder is mid-turn and another client's request is parked in the inbox it still names the
    holder; once the parked request's own turn starts it names that client."""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(world, "C", CLAIMANT_IP, CLAIMANT_THREAD)
        assert r.rank_client_meta["ip"] == HOLDER_IP, (
            f"while the holder is mid-turn the resident is stamped with {r.rank_client_meta['ip']}, "
            "the parked request's client"
        )
        assert r.rank_client_meta is h1.client_meta or r.rank_client_meta == h1.client_meta

        world.release("H1")
        assert await world.started("C", timeout=PROMPT_S)
        assert r.rank_client_meta["ip"] == CLAIMANT_IP, (
            "once the parked request's own turn runs the resident must name its client"
        )


# --------------------------------------------------------------------------------------
# (e) a parked claim does not leak when the claimant goes away
# --------------------------------------------------------------------------------------

async def test_claim_is_released_when_the_parked_client_disconnects(tmp_path):
    """The client of a parked request goes away (its disconnect event is set). Trigger: the
    liveness check the claims snapshot runs on every read ("disconnected"). After that read
    the claim is no longer registered, so a request that will never be waited for cannot sit
    in the registry toward the claim-capacity cap."""
    async with build_world(tmp_path) as world:
        gone = asyncio.Event()
        _h1, claimant, _r = await hold_and_park(
            world, "C", CLAIMANT_IP, CLAIMANT_THREAD, disconnect_event=gone,
        )
        assert_parked_claim_held(world)
        drain = world.subscribe()
        gone.set()
        assert claimant.slot_id not in world.snapshot_ids()
        assert not world.has_claim("C"), "a disconnected parked request still holds a claim"
        reasons = [e.get("reason") for e in await drain()
                   if e["event"] == "fastlane_claim_released" and e["slot_id"] == claimant.slot_id]
        assert reasons and "admitted" not in reasons, reasons


@pytest.mark.parametrize("how", ["cancelled", "failed"])
async def test_claim_is_released_when_the_parked_request_future_ends_badly(tmp_path, how):
    """The parked request's completion future is cancelled (the caller timed out) or failed.
    Trigger: the same liveness check on the next claims snapshot read ("failed"). The claim
    must not stay registered afterwards."""
    async with build_world(tmp_path) as world:
        _h1, claimant, _r = await hold_and_park(world, "C", CLAIMANT_IP, CLAIMANT_THREAD)
        assert_parked_claim_held(world)
        if how == "cancelled":
            claimant.completion_future.cancel()
        else:
            claimant.completion_future.set_exception(RuntimeError("caller gave up"))
            claimant.completion_future.exception()  # mark retrieved
        assert claimant.slot_id not in world.snapshot_ids()
        assert not world.has_claim("C"), f"a {how} parked request still holds a claim"


async def park_with_dispatcher_paused(world):
    """Like ``hold_and_park`` but the dispatcher is stopped first and driven by hand.

    The holder's turn is running. The dispatcher is then paused, the claimant is submitted
    (it lands in staging with its claim) and one dispatcher iteration routes it through the
    real HIT path into the resident's inbox. Nothing else moves the request afterwards, so a
    test can act on the parked request (tear the resident down, fail its re-queue) and look at
    the result before anything dispatches it again.
    """
    h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
    assert await world.started("H1", timeout=5.0), "setup: the holder's turn never started"
    await world.pause_dispatcher()
    claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
    assert world.has_claim("C"), "setup: a queued request registers its claim at submit"
    assert claimant in world.mgr.queue._staging, "setup: the request must be queued, not routed"
    await world.dispatch_once()
    r = resident_for(world.mgr, MODEL)
    assert r.inbox.qsize() == 1 and claimant not in world.mgr.queue._staging, (
        "setup: the request must now sit in the resident's inbox"
    )
    assert r.active_slot is h1
    return h1, claimant, r


# --------------------------------------------------------------------------------------
# (e) resident goes away while the request is parked
# --------------------------------------------------------------------------------------

async def test_claim_is_released_when_the_resident_dies_and_the_requeue_fails(tmp_path):
    """The resident the request is parked on is torn down (the entry point every unload and
    driver-death reap goes through) and the re-queue of its inbox fails, so the request's
    future is failed. Trigger: that failed future, seen by the liveness check on the next
    claims snapshot read ("failed"). The claim must not stay registered."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r = await park_with_dispatcher_paused(world)
        assert_parked_claim_held(world)

        async def refuse(slot):
            raise RuntimeError("queue unavailable")

        world.mgr.queue.enqueue_head = refuse
        async with world.mgr._registry_lock:
            world.mgr._begin_unload_locked(r)
        await world.drain_background()
        assert claimant.completion_future.done() and claimant.completion_future.exception() is not None, (
            "setup: the failed re-queue must fail the request's future"
        )
        assert claimant.slot_id not in world.snapshot_ids()
        assert not world.has_claim("C"), "the request's claim outlived its failed future"


# --------------------------------------------------------------------------------------
# (f) re-queued from a torn-down resident's inbox
# --------------------------------------------------------------------------------------

async def test_a_slot_requeued_from_a_torn_down_residents_inbox_keeps_its_claim(tmp_path):
    """The resident is torn down for a reason other than designation (an unload to make room)
    and its inbox is re-queued to the staging queue. The request is waiting again, so it is
    back in staging, still holds its claim and is still listed in the claims snapshot."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r = await park_with_dispatcher_paused(world)
        async with world.mgr._registry_lock:
            world.mgr._begin_unload_locked(r)
        await world.drain_background()
        assert r.state is ResidentState.DEAD and r.inbox.empty()
        assert claimant in world.mgr.queue._staging, "setup: the re-queue must put it back in staging"
        assert world.has_claim("C"), "the request lost its claim when its resident was torn down"
        assert claimant.slot_id in world.snapshot_ids()


async def test_a_failed_tail_requeue_fails_the_future_and_releases_the_claim(tmp_path):
    """The inbox is re-queued at the staging tail (the path a displaced resident's backlog
    takes) and staging is full, so the re-queue raises QueueFull. The request must not be
    dropped silently: its future is failed, and its claim is released (the liveness check on
    the next snapshot read) so it does not leak."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r = await park_with_dispatcher_paused(world)
        assert_parked_claim_held(world)
        drained = [r.inbox.get_nowait()]
        world.mgr.queue.staging_max = 0
        await world.mgr._requeue_slots_tail_or_fail(drained)
        assert claimant.completion_future.done() and claimant.completion_future.exception() is not None, (
            "the request was dropped without failing its future"
        )
        assert claimant.slot_id not in world.snapshot_ids()
        assert not world.has_claim("C"), "the request's claim outlived its failed re-queue"


# --------------------------------------------------------------------------------------
# (g) a parked claim does not widen to unrelated residents (layer-split model)
# --------------------------------------------------------------------------------------

def add_unrelated_resident(world, ip):
    """A second, unrelated resident (another model) that is mid-turn for the client ``ip``."""
    from unittest.mock import MagicMock

    from turbohaul.manager import Resident
    from turbohaul.queue import GraceTimer, IdleHotTimer
    from turbohaul.subprocess_mgr import SidecarHandle

    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    r = Resident(
        model_tag="other", resident_key="other",
        handle=SidecarHandle(proc=proc, port=59999, model_tag="other"), port=59999,
        grace=GraceTimer(grace_seconds=2, max_extensions=50), idle=IdleHotTimer(idle_seconds=600),
        rank_client_meta=meta(ip), last_active_monotonic=1.0, main_gpu=1, split_mode="none",
        inbox=asyncio.Queue(),
    )
    r.state = ResidentState.ACTIVE
    world.mgr._residents["other"] = r
    return r


async def test_a_parked_claim_for_a_layer_split_model_does_not_designate_a_sibling(tmp_path):
    """The model is layer-split, so a claim that NEEDS ROOM for it would make every outranked
    resident of another model a victim (control below, true before and after any change). A
    claim parked in the model's own resident's inbox needs no room: an unrelated, outranked
    resident of another model must not be designated, must keep its grace and must not be
    unloaded.

    On unchanged code a parked request holds no claim, so the second half passes there and
    cannot fail; it fails only on a change that keeps the claim without scoping it to the
    resident it is parked on. The non-vacuity partner (the claim really is kept) is
    ``test_the_parked_layer_split_claim_is_really_kept``."""
    async with build_world(tmp_path, model_split_mode="layer", max_parallel_sidecars=2) as world:
        h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=5.0)
        other = add_unrelated_resident(world, LOWER_IP)
        try:
            await world.pause_dispatcher()
            claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
            mgr = world.mgr

            # Control: while the request is only QUEUED its claim needs room, and for a
            # layer-split model that designates every outranked resident, the sibling included.
            async with mgr._registry_lock:
                assert mgr._is_designated_unload_target_locked(other), (
                    "control: a queued claim for a layer-split model must designate an outranked sibling"
                )

            await world.dispatch_once()
            r = resident_for(mgr, MODEL)
            assert r.inbox.qsize() == 1 and r.active_slot is h1 and claimant not in mgr.queue._staging
            async with mgr._registry_lock:
                designated = mgr._is_designated_unload_target_locked(other)
            assert not designated, (
                "an unrelated resident of another model was designated by a claim parked on the "
                "layer-split model's own resident"
            )
            assert other.state is ResidentState.ACTIVE and "other" not in world.unloads
        finally:
            world.mgr._residents.pop("other", None)


async def test_the_parked_layer_split_claim_is_really_kept(tmp_path):
    """Non-vacuity partner of the test above: in the same layer-split shape the parked
    request still holds its claim. Without this, the test above could pass only because the
    claim had already been released."""
    async with build_world(tmp_path, model_split_mode="layer", max_parallel_sidecars=2) as world:
        await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=5.0)
        await world.pause_dispatcher()
        await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
        await world.dispatch_once()
        assert resident_for(world.mgr, MODEL).inbox.qsize() == 1
        assert world.has_claim("C"), "the parked request's claim was released"


# --------------------------------------------------------------------------------------
# (d) the reroute path (a refused second engine hands the request to a live engine)
# --------------------------------------------------------------------------------------

def _rerouted_claimant(request_factory):
    """A staged Fast Lane request for a tag with a live engine, from a new session."""
    from turbohaul.fastlane import FastLaneMatch

    slot = request_factory("duo")
    slot.client_meta = {"session_id": "NEW", "ip": "10.0.0.7"}
    slot.fastlane = FastLaneMatch(
        rule_index=0, raw_address="10.0.0.7", label="", effective_tag="main", rank=1,
    )
    return slot


async def test_the_reroute_path_keeps_the_claim_until_the_turn_starts(tmp_path):
    """A request for a tag that already has a live engine starts a second engine; a host
    safety gate refuses it and the request is handed to the live engine
    (``_fail_or_reroute_refused_spawn``). It is then parked in the live engine's inbox, so
    its claim must still be registered (and listed in the snapshot) after the hand-off, and
    must be released when the live engine's own turn for that request starts.

    The whole path is real: the claim is registered by the same call ``submit`` makes, the
    second engine is reserved by ``_route_or_reserve``, the refusal comes from the safety
    gates, and the live engine's driver then serves the request."""
    from unittest.mock import patch

    from _multiinstance_support import _live_engine, _manifest, _new_manager, _probe, _request, _track_started

    from turbohaul.queue import GraceTimer, IdleHotTimer
    from turbohaul.safety import GateResult
    from turbohaul.subprocess_mgr import SidecarHandle

    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    manager.runtime.queue.safety_enabled = True
    manager.runtime.queue.spawn_reclaim_wait_max_s = 0.0
    manager.runtime.queue.grace_seconds = 0
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0)
    engine.grace = GraceTimer(grace_seconds=0, max_extensions=0)
    engine.idle = IdleHotTimer(idle_seconds=600)
    _track_started(manager)
    slot = _rerouted_claimant(_request)
    await manager._register_staged_claim(slot, "staging_arrival")

    claim_at_turn = []

    async def complete(s, handle):
        claim_at_turn.append(any(c["slot"] is s for c in manager._fastlane_claims.values()))
        return {"ok": True, "model": handle.model_tag}

    manager._complete_fn = complete
    refusal = GateResult("cpu_busy", False, "cpu 95% > max 80%", blocked_on="host")
    p1, p2 = _probe([20000, 20000])
    try:
        with p1, p2, patch("turbohaul.manager.all_safety_gates", return_value=[refusal]):
            await manager._route_or_reserve(slot)
            await wait_until(lambda: not engine.inbox.empty() or slot.completion_future.done(), timeout=5.0)
        assert not slot.completion_future.done(), f"setup: the request failed: {slot.completion_future.exception()!r}"
        assert engine.inbox.qsize() == 1 and spawn_calls == [], "setup: the request must be on the live engine"

        assert any(c["slot"] is slot for c in manager._fastlane_claims.values()), (
            "the claim was released when the request was handed to the live engine"
        )
        assert slot.slot_id in [row["slot_id"] for row in manager.fastlane_claims_snapshot()]

        async def refuse_spawn_free(r, s):
            return SidecarHandle(proc=None, port=59998, model_tag="duo")

        manager._spawn_for_resident = refuse_spawn_free
        engine.driver_task = asyncio.create_task(manager._drive_resident(engine))
        done, _ = await asyncio.wait({slot.completion_future}, timeout=5.0)
        assert done and slot.completion_future.result()["ok"], "the live engine never served the request"
        assert claim_at_turn == [False], "the claim was still registered when the request's turn ran"
        assert not any(c["slot"] is slot for c in manager._fastlane_claims.values())
    finally:
        await manager.shutdown()


async def test_the_reroute_helper_alone_does_not_release_the_claim(tmp_path):
    """The hand-off helper called directly, for a request that carries its claim into it (the
    state a request is in when the routing before it keeps the claim). Handing the request to
    the live engine is only a move from one queue to another; the helper must leave the claim
    registered and still record the wait."""
    from _multiinstance_support import _live_engine, _manifest, _new_manager, _request

    from turbohaul.manager import Resident

    manager, boot, _spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0)
    refused = Resident(
        model_tag="duo", resident_key="duo#1", state=ResidentState.RESERVED_LOADING,
        main_gpu=1, split_mode="none", inbox=asyncio.Queue(),
    )
    manager._residents["duo#1"] = refused
    slot = _rerouted_claimant(_request)
    await manager._register_staged_claim(slot, "staging_arrival")
    try:
        await manager._fail_or_reroute_refused_spawn(refused, slot, RuntimeError("refused"))
        assert engine.inbox.get_nowait() is slot
        assert slot.slot_id in manager._handed_off_waiting
        assert any(c["slot"] is slot for c in manager._fastlane_claims.values()), (
            "the hand-off helper released the request's claim while it is still waiting"
        )
    finally:
        await manager.shutdown()
