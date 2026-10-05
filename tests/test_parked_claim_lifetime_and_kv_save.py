"""How long a parked claim lives, and what the holder's KV does when a parked request takes over.

A request for an already-loaded model tag is handed to the resident's inbox (the HIT path of
``_route_or_reserve``) while the holder is still mid-turn or waiting in its grace window. The
claim of that parked request has to behave like the claim of a request that is still queued,
with two differences that are pinned here:

* Lifetime. A claim parked in a resident's own inbox is kept while it is parked and is NOT
  expired by the claim time-to-live while parked (the time-to-live stays exactly as it is for
  a claim of a request still sitting in staging). It is released when the request's own turn
  starts, when its client disconnects, when its future is cancelled or failed, or when its
  resident dies. A parked client that disconnects must not leave a zombie designation behind:
  the holder gets its grace window back.
* The holder's KV. When the parked higher-ranked request is served next on the SAME resident,
  the holder's KV is saved first, with the same unconditional slot save the unload path uses
  (``_save_slot_kv`` for the holder's thread), not with the per-turn clean-prefix save, which
  declines under the single-series gate. A failed save is logged loudly and the request is
  still admitted. With a manifest that runs ``--parallel`` above 1 the save is still harmless.

Everything runs real manager code (real dispatcher, drivers, grace loop, claim registry) with a
fake engine process and a fake completion call that blocks per request, so a test decides
exactly when each turn ends. One process, asyncio only, bounded waits, no sleep above 1 s.
The completion call, the slot save and the claim time-to-live are the only seams replaced.
"""

import asyncio
import contextlib
import logging
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
RULES = ranked_rules((CLAIMANT_IP, 1), (HOLDER_IP, 1))

HOLDER_THREAD = "thread-holder"
CLAIMANT_THREAD = "thread-claimant"

# How long a test waits for something that must happen promptly, and for something that
# legitimately waits behind a short grace window. Both are bounds, not delays.
PROMPT_S = 1.5
AFTER_GRACE_S = 8.0

# A claim time-to-live short enough to run out inside one test, and a sleep that outlasts it.
# Both stay far below one second so the whole test never sleeps long.
SHORT_TTL_S = 0.3
PAST_TTL_S = 0.5


def meta(ip):
    return {"ip": ip, "is_main": True}


class SaveCall:
    """One call of the manager's slot save, as the fake recorded it."""

    def __init__(self, port, model_tag, slot, kwargs):
        self.port = port
        self.model_tag = model_tag
        self.slot = slot
        self.kwargs = kwargs
        # The thread the save is keyed on: an explicit override (the unload path's way of
        # naming the idle holder) or else the slot it is called with.
        self.thread = kwargs.get("thread_id_override") or getattr(slot, "thread_id", None)
        self.force_clean = bool(kwargs.get("force_clean"))


class World:
    """One manager, one box slot, a fake engine whose turns end only when told to.

    ``send`` submits a request through the real ``submit`` and labels it. The fake completion
    call stamps ``events`` with ("start", slot_id) when a turn begins and ("end", slot_id) when
    it finishes, and the fake slot save (when installed) stamps ("save", thread), so a test can
    assert the ORDER of the engine-visible steps with one list.
    """

    def __init__(self, mgr, spawns):
        self.mgr = mgr
        self.spawns = spawns
        self.slots = {}          # label -> Slot
        self.labels = {}         # slot_id -> label
        self.events = []         # ("start"|"end", slot_id) and ("save", thread), in order
        self.saves = []          # SaveCall per slot save the manager attempted
        self.handles = {}        # slot_id -> engine handle the turn ran on
        self.claim_at_start = {}  # slot_id -> claim still registered when the turn began
        self.unloads = []        # resident keys passed to _begin_unload_locked
        self.cleanups = []       # callables run when the world is torn down
        self._gates = {}
        self._open = False

    # -- fake completion -------------------------------------------------
    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def complete(self, slot, handle):
        self.events.append(("start", slot.slot_id))
        self.handles[slot.slot_id] = handle
        self.claim_at_start[slot.slot_id] = any(
            c["slot"] is slot for c in self.mgr._fastlane_claims.values()
        )
        if not self._open:
            await self.gate(slot.slot_id).wait()
        self.events.append(("end", slot.slot_id))
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

    def named_events(self):
        """The event list with slot ids replaced by labels (read lazily: a label is only known
        once ``send`` has returned, which can be after the turn already started)."""
        return [(kind, self.labels.get(what, what)) for kind, what in self.events]

    def order(self):
        """Labels of the turns that have started, in start order."""
        return [what for kind, what in self.named_events() if kind == "start"]

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

    async def turn_done(self, label, timeout=PROMPT_S):
        """Wait until ``label``'s turn has completed (it left ACTIVE)."""
        slot = self.slots[label]
        await wait_until(
            lambda: slot.state in (SlotState.GRACE, SlotState.POPPED), timeout=timeout,
        )

    async def pause_dispatcher(self):
        """Stop the dispatcher task (the resident's driver keeps running).

        Afterwards a request submitted with ``send`` stays in staging, so a test can look at
        a request that is queued and has not been routed to the resident.
        """
        task = self.mgr._worker_task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

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

    def has_claim(self, label):
        return self.slots[label] in self.claim_slots()

    def snapshot_ids(self):
        """Slot ids the claims snapshot lists (it prunes dead claims first, like every read)."""
        return [row["slot_id"] for row in self.mgr.fastlane_claims_snapshot()]

    def governing_slot(self):
        """The slot of the governing live claim, or None (a plain read, never prunes)."""
        governing = self.mgr._governing_claim_locked()
        return None if governing is None else governing[1]


@contextlib.asynccontextmanager
async def build_world(tmp_path, *, grace_seconds=2, max_grace_extensions=50,
                      idle_hot_load_seconds=600, engine_parallel=1,
                      record_saves=True, save_fails=False, gate_declines_clean_save=False):
    """A manager with ONE box slot and the dispatcher running, over two ranked clients.

    ``record_saves`` (on by default, so no test ever makes a save request to the fake engine's
    port) replaces the manager's slot save (``_save_slot_kv``, the one writer every engine slot
    save goes through) with a recorder; ``save_fails`` makes that recorder raise.
    ``gate_declines_clean_save`` makes the per-turn clean-prefix save decline at its
    single-series gate (it counts one other active resident), so a slot save recorded at a swap
    cannot have come from the clean-prefix path. ``engine_parallel`` is the width the fake
    engine reports (the manifest's ``--parallel``).
    """
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=RULES, max_parallel_sidecars=1, grace_seconds=grace_seconds,
        max_grace_extensions=max_grace_extensions, idle_hot_load_seconds=idle_hot_load_seconds,
    )
    seed_manifest(boot, MODEL, main_gpu=0)
    spawns = []
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})

    def counting_spawn(binary, gguf, port, model_tag, argv, **kw):
        spawns.append(model_tag)
        handle = spawn_fn(binary, gguf, port, model_tag, argv, **kw)
        handle.parallel = engine_parallel
        return handle

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

    def spy_unload(r):
        world.unloads.append(r.resident_key)
        return real_unload(r)

    mgr._begin_unload_locked = spy_unload

    if record_saves:
        async def fake_save(port, model_tag, slot=None, **kwargs):
            call = SaveCall(port, model_tag, slot, kwargs)
            world.saves.append(call)
            world.events.append(("save", call.thread))
            if save_fails:
                raise OSError("engine refused the slot save")
            return True

        mgr._save_slot_kv = fake_save

    if gate_declines_clean_save:
        mgr._other_active_residents = lambda handle: 1

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            yield world
    finally:
        for undo in world.cleanups:
            undo()
        world.open_everything()
        await mgr.shutdown()


def stamp_loud_save_failures(world):
    """Stamp ("log", "loud") into ``world.events`` whenever the manager logs, at WARNING or
    higher, a message saying a save failed for the model, so a test can order the log line
    against the save attempt and the next turn's start. The logger is shared by the whole
    manager module, so the handler is detached again in the world's teardown (through
    ``world.cleanups``) before the manager shuts down."""
    class Stamp(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if record.levelno >= logging.WARNING and "save failed" in msg.lower() and MODEL in msg:
                world.events.append(("log", "loud"))

    handler = Stamp(level=logging.WARNING)
    target = logging.getLogger("turbohaul.manager")
    target.addHandler(handler)
    world.cleanups.append(lambda: target.removeHandler(handler))


async def hold_and_park(world, claimant_label="C", claimant_ip=CLAIMANT_IP,
                        claimant_thread=CLAIMANT_THREAD, **kw):
    """The defect shape: the holder is mid-turn, and a request is parked in its inbox.

    The holder's first turn is running (its fake completion is blocked), then the second
    request is submitted through the real ``submit``; the dispatcher routes it to the
    resident's inbox through the real HIT path. Nothing is written into the claim registry by
    hand. Returns (holder slot, claimant slot, resident).
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


def assert_parked_claim_held(world, label="C"):
    """The parked request still holds its claim (the precondition of every lifetime check)."""
    assert world.has_claim(label), (
        "a parked request must hold its claim; it was released when it was parked in the inbox"
    )


# --------------------------------------------------------------------------------------
# claim lifetime: the time-to-live
# --------------------------------------------------------------------------------------

async def test_parked_claim_older_than_the_ttl_still_designates_the_holder(tmp_path, monkeypatch):
    """The claim time-to-live is made very short. A higher-ranked request is parked in the
    holder's inbox through the real HIT path and stays parked well past that time-to-live while
    the holder's turn runs. Its claim must still be registered, listed in the claims snapshot
    and the governing claim, so the holder still loses its grace at its next turn boundary: the
    parked request is served next, ahead of the holder's own same-thread follow-up. If the
    time-to-live ran on while parked, the claim would be pruned as expired and the holder's
    follow-ups would keep winning again, which is the starvation the claim exists to prevent.
    (Order is asserted from the engine-visible turn starts; nothing is read from a timer.)"""
    monkeypatch.setattr(manager_module, "_FASTLANE_CLAIM_TTL_S", SHORT_TTL_S)
    async with build_world(tmp_path) as world:
        h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=5.0)
        registered_at = time.monotonic()
        claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
        r = await world.parked("C")
        assert r.active_slot is h1 and claimant.state is SlotState.STAGED
        assert_parked_claim_held(world)  # precondition: without it the aging below proves nothing

        # The request stays parked past its time-to-live (a bounded wait, well under 1 s).
        await asyncio.sleep(PAST_TTL_S)
        assert time.monotonic() - registered_at > SHORT_TTL_S, "setup: the request did not age past the TTL"
        assert world.order() == ["H1"] and r.inbox.qsize() == 1, "setup: the request must still be parked"

        assert world.has_claim("C"), "the parked request's claim was dropped after the claim TTL ran out"
        assert world.governing_slot() is claimant, (
            "the aged parked claim no longer governs: the holder would regain its grace"
        )
        assert claimant.slot_id in world.snapshot_ids(), "the aged parked claim is missing from the claims snapshot"

        # The holder is still the one that loses its grace at its boundary.
        world.release("H1")
        await world.turn_done("H1")
        h2 = await world.send("H2", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("C", timeout=PROMPT_S), (
            f"the aged parked request was not served at the holder's boundary; order {world.order()}"
        )
        assert world.order() == ["H1", "C"], (
            f"the holder's warm follow-up was served ahead of the aged parked request: {world.order()}"
        )
        assert not h2.completion_future.done()


async def test_control_a_staged_claim_older_than_the_ttl_still_expires(tmp_path, monkeypatch):
    """Control for the test above, unchanged behaviour. A claim whose request is still sitting
    in staging (not routed, not parked) and is older than the claim time-to-live DOES expire:
    it no longer governs, the claims snapshot prunes it, and the release is reported with the
    reason "ttl_expired". The same short time-to-live and the same sleep as the parked test,
    so that test cannot pass merely because the time-to-live was never short."""
    monkeypatch.setattr(manager_module, "_FASTLANE_CLAIM_TTL_S", SHORT_TTL_S)
    async with build_world(tmp_path) as world:
        await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=5.0)
        await world.pause_dispatcher()
        registered_at = time.monotonic()
        claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
        assert claimant in world.mgr.queue._staging, "setup: the request must be queued, not routed"
        assert_parked_claim_held(world)
        assert world.governing_slot() is claimant, "setup: a fresh queued claim governs"

        await asyncio.sleep(PAST_TTL_S)
        assert time.monotonic() - registered_at > SHORT_TTL_S, "setup: the claim did not age past the TTL"
        assert world.governing_slot() is None, "an expired queued claim must not govern"
        drain = world.subscribe()
        assert claimant.slot_id not in world.snapshot_ids()
        assert not world.has_claim("C"), "the expired queued claim was not pruned"
        reasons = [e.get("reason") for e in await drain()
                   if e["event"] == "fastlane_claim_released" and e["slot_id"] == claimant.slot_id]
        assert reasons == ["ttl_expired"], reasons


# --------------------------------------------------------------------------------------
# claim lifetime: the parked client disconnects
# --------------------------------------------------------------------------------------

async def test_parked_claimant_that_disconnects_releases_its_claim_and_the_holder_regains_grace(tmp_path):
    """The client of a parked higher-ranked request goes away (its disconnect event is set).
    The next claims snapshot read (the real place that cleans, it prunes every claim whose
    request is dead, reason "disconnected") must leave no claim registered, so the holder is no
    longer designated and gets its grace window back: its next same-thread follow-up is served
    warm, on the same resident, which stays loaded. A claim that outlived its client would be a
    zombie designation: the holder would lose its grace to a request nobody waits for.

    On unchanged code the claim is already gone at park, so only the precondition (the parked
    request holds its claim before the disconnect) fails; the checks after it cannot fail there
    and fail only on code that keeps a disconnected claim."""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(
            world, disconnect_event=(gone := asyncio.Event()),
        )
        assert_parked_claim_held(world)  # precondition
        assert world.governing_slot() is claimant

        gone.set()
        assert claimant.slot_id not in world.snapshot_ids()
        assert not world.has_claim("C"), "a disconnected parked request still holds a claim"
        assert world.governing_slot() is None
        async with world.mgr._registry_lock:
            assert not world.mgr._is_designated_unload_target_locked(r), (
                "the holder is still designated because of a client that disconnected"
            )

        # Grace is back: the holder's warm follow-up is served first, on the same resident.
        world.release("H1")
        await world.turn_done("H1")
        await world.send("H2", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H2", timeout=PROMPT_S), f"order {world.order()}"
        assert world.order() == ["H1", "H2"], (
            f"the holder lost its grace to a parked request whose client had disconnected: {world.order()}"
        )
        assert resident_for(world.mgr, MODEL) is r and r.state is not ResidentState.DEAD
        assert world.spawns == [MODEL] and world.unloads == [], (
            f"the resident was reloaded or unloaded: spawns={world.spawns} unloads={world.unloads}"
        )
        assert world.handles[h1.slot_id] is world.handles[world.slots["H2"].slot_id]


async def test_a_disconnected_parked_claimant_is_not_served(tmp_path):
    """Once the client of a parked request has disconnected nobody waits for its answer, so the
    resident must not spend a turn on it. The holder's turn ends, the holder's grace window (1 s
    here) runs out with no follow-up, and the disconnected request must still never start. A
    separate test from the claim and grace checks above: whether the parked request is served
    is a statement about the admission from the inbox, not about the claim registry."""
    async with build_world(tmp_path, grace_seconds=1) as world:
        await hold_and_park(world, disconnect_event=(gone := asyncio.Event()))
        gone.set()
        world.release("H1")
        await world.turn_done("H1")
        served = await world.started("C", timeout=2.5)
        assert not served, (
            f"a turn was started for a request whose client had disconnected: order {world.order()}"
        )


# --------------------------------------------------------------------------------------
# the holder's KV at a same-resident swap
# --------------------------------------------------------------------------------------

async def test_the_clean_prefix_save_declines_under_the_gate_in_this_fixture(tmp_path, caplog):
    """Fixture check. With the gate flag of ``build_world`` the per-turn clean-prefix save of the
    holder's turn declines at the single-series gate and writes nothing, so a slot save that the
    swap tests below record can only have been made by the unconditional path. Run on the
    holder's own slot, through the real ``_probe_and_save_clean_kv``."""
    caplog.set_level(logging.DEBUG, logger="turbohaul.manager")
    async with build_world(tmp_path, gate_declines_clean_save=True) as world:
        h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=5.0)
        handle = world.handles[h1.slot_id]
        await world.mgr._probe_and_save_clean_kv(handle, h1)
        assert world.saves == [], f"the clean-prefix save wrote despite the gate: {world.saves}"
        declines = [rec.getMessage() for rec in caplog.records if "KV_SAVE_DECLINE" in rec.getMessage()]
        assert any("SINGLE_SERIES_GATE" in msg for msg in declines), declines


async def test_the_holders_kv_is_saved_when_the_claimant_is_served_next_on_the_same_resident(tmp_path):
    """The holder is mid-turn and a higher-ranked request is parked in its inbox. When the
    holder's turn ends and the parked request is served next on the same resident, the holder's
    KV is saved FIRST: one slot save for the holder's thread, on the holder's engine and model,
    stamped after the holder's turn ended and before the parked request's turn started. The
    save is the unconditional one (the unload path's slot save, not the clean-prefix save): the
    fixture makes the clean-prefix save decline at its single-series gate and records every slot
    save, so a save seen here did not come from the clean-prefix path. Without the save the
    holder's cached prefix is lost to the claimant's prompt on the one engine slot."""
    async with build_world(tmp_path, gate_declines_clean_save=True) as world:
        h1, claimant, r = await hold_and_park(world)
        world.release("H1")
        assert await world.started("C", timeout=AFTER_GRACE_S), (
            f"setup: the parked request never started; events {world.named_events()}"
        )
        events = world.named_events()
        c_start = events.index(("start", "C"))
        h1_end = events.index(("end", "H1"))
        before_c = [(i, e) for i, e in enumerate(events[:c_start]) if e[0] == "save"]
        assert before_c, (
            "no slot save was made for the holder before the parked request's turn started; "
            f"events {events}"
        )
        assert [e for _, e in before_c] == [("save", HOLDER_THREAD)], (
            f"the saves before the swap are not exactly the holder's thread: {before_c}"
        )
        save_at = before_c[0][0]
        assert h1_end < save_at < c_start, (
            f"the save is not between the end of the holder's turn and the start of the claimant's: {events}"
        )
        call = world.saves[0]
        assert call.model_tag == MODEL and call.port == r.handle.port, (
            f"the save was not made on the holder's engine: {call.model_tag} port {call.port}"
        )
        assert not [c for c in world.saves if c.force_clean], (
            "a save came from the clean-prefix path, which the single-series gate can decline"
        )
        assert world.handles[claimant.slot_id] is world.handles[h1.slot_id]


async def test_a_failed_holder_save_is_logged_loudly_and_the_claimant_is_still_admitted(tmp_path, caplog):
    """Same swap as above, but the holder's slot save raises. The failure is logged at WARNING
    or higher with a message that says the save failed and names the model (the unload path
    words its own failure "resident KV cache save failed (best-effort): model_tag=..."), the
    save really was attempted for the holder's thread before the swap, and the parked request is
    still admitted and answered: a best-effort save must never block the higher-ranked request.

    On unchanged code nothing is attempted at the swap, so the attempt check is the first to
    fail. The wording check asserts the level, "save failed" and the model tag, not the whole
    sentence, so any of the unload path's phrasings passes."""
    caplog.set_level(logging.DEBUG, logger="turbohaul.manager")
    async with build_world(tmp_path, save_fails=True, gate_declines_clean_save=True) as world:
        stamp_loud_save_failures(world)
        h1, claimant, r = await hold_and_park(world)
        world.release("H1")
        assert await world.started("C", timeout=AFTER_GRACE_S), (
            f"the parked request was not admitted after the failed save; events {world.named_events()}"
        )
        events = world.named_events()
        c_start = events.index(("start", "C"))
        assert ("save", HOLDER_THREAD) in events[:c_start], (
            f"no save was attempted for the holder before the parked request's turn; events {events}"
        )
        loud = [rec for rec in caplog.records
                if rec.levelno >= logging.WARNING
                and "save failed" in rec.getMessage().lower()
                and MODEL in rec.getMessage()]
        assert loud, (
            "the failed holder save was not logged at WARNING or higher naming the model; records: "
            f"{[(rec.levelname, rec.getMessage()) for rec in caplog.records if rec.levelno >= logging.WARNING]}"
        )
        assert events.index(("save", HOLDER_THREAD)) < events.index(("log", "loud")) < c_start, (
            f"the failure was not logged between the failed save and the claimant's turn: {events}"
        )
        world.release("C")
        done, _ = await asyncio.wait({claimant.completion_future}, timeout=PROMPT_S)
        assert done and claimant.completion_future.result()["ok"], "the admitted request was not answered"
        assert h1.completion_future.result()["ok"], "the holder's own turn must not be affected by its failed save"


# --------------------------------------------------------------------------------------
# a resident that runs --parallel above 1
# --------------------------------------------------------------------------------------

async def test_with_parallel_above_one_the_swap_is_harmless(tmp_path):
    """The engine runs two parallel streams, so the parked request is served as a rider of the
    same engine (no swap at all) once the holder's turn ends. Whatever slot save the manager
    makes at that point must be harmless: the holder's own turn completes normally (never cut),
    the parked request is served and answered on the same engine, the resident is neither
    unloaded nor respawned, and any save that is made is for the holder's thread. Green on
    unchanged code (no save is made there); it guards the new save against breaking a
    multi-stream resident."""
    async with build_world(tmp_path, engine_parallel=2) as world:
        h1, claimant, r = await hold_and_park(world)
        assert not h1.completion_future.done()
        world.release("H1")
        assert await world.started("C", timeout=PROMPT_S), f"events {world.named_events()}"
        world.release("C")
        done, _ = await asyncio.wait({h1.completion_future, claimant.completion_future}, timeout=PROMPT_S)
        assert len(done) == 2, "a request was left unanswered"
        assert h1.completion_future.result()["ok"], "the holder's turn was cut or failed"
        assert claimant.completion_future.result()["ok"], "the parked request failed"
        assert world.handles[claimant.slot_id] is world.handles[h1.slot_id]
        assert resident_for(world.mgr, MODEL) is r and r.state is not ResidentState.DEAD
        assert world.spawns == [MODEL] and world.unloads == []
        assert {c.thread for c in world.saves} <= {HOLDER_THREAD}, (
            f"a save was made for a thread other than the holder's: {[c.thread for c in world.saves]}"
        )


async def test_with_parallel_above_one_the_riders_claim_is_gone_when_its_turn_runs(tmp_path):
    """On a resident that runs --parallel above 1 the parked request is served as a rider: it
    never goes through the per-turn entry of the resident's driver. Its claim must still be
    gone by the time its own turn is running, and must not be left registered afterwards, or
    the claim would outlive the request. The first check (the parked request holds its claim)
    keeps the rest from passing vacuously; on unchanged code it fails because the claim is
    released at park. After the change the rest guards a release placed only on the driver's
    per-turn entry."""
    async with build_world(tmp_path, engine_parallel=2) as world:
        _h1, claimant, _r = await hold_and_park(world)
        assert_parked_claim_held(world)  # precondition: on unchanged code the claim is gone already
        world.release("H1")
        assert await world.started("C", timeout=PROMPT_S), f"events {world.named_events()}"
        assert world.claim_at_start[claimant.slot_id] is False, (
            "the rider's claim was still registered when its own turn started"
        )
        world.release("C")
        await asyncio.wait({claimant.completion_future}, timeout=PROMPT_S)
        assert not world.has_claim("C"), "the rider's claim outlived its turn"
        assert claimant.slot_id not in world.snapshot_ids()
