"""A request parked in a resident's inbox whose client has disconnected is dropped, not served.

A request for an already-loaded model tag is handed to the resident's inbox (the HIT path of
``_route_or_reserve``) while the holder is mid-turn or waiting in its grace window. If the
client of that parked request goes away, nobody is left to read the answer, so:

* the request is dropped from the inbox and its future is failed the way an evicted rider is
  failed (``SlotEvictedError``), and its claim is released;
* the holder is NOT preempted for it: no turn is started for it, the resident is not unloaded or
  respawned, no slot save is made on its behalf, and the holder keeps (or regains) its grace
  window, so its same-thread follow-up is still served warm;
* the check is made at admit time at the latest, so a disconnect that lands in the very tick the
  holder's turn ends is handled the same way;
* a live parked request is still served next at the holder's turn boundary (the drop is not a
  blanket break), and a turn that has already started is never cut by the drop.

Everything runs real manager code (real dispatcher, drivers, grace loop, claim registry) with a
fake engine process and a fake completion call that blocks per request, so a test decides exactly
when each turn ends. One process, asyncio only, bounded waits, no sleep above 1 s. The completion
call and the slot save are the only seams replaced.
"""

import asyncio
import contextlib

from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    ranked_rules,
    resident_for,
    seed_manifest,
    wait_until,
)

from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotEvictedError, SlotState

MODEL = "m1"

# Priority order: index 0 outranks index 1, which outranks index 2. Two claimants sit above the
# holder; the second one only matters to the two-parked-requests test (two requests of ONE client
# rule would share a single claim, so they are given two clients).
CLAIMANT_IP = "10.0.0.3"
SECOND_CLAIMANT_IP = "10.0.0.4"
HOLDER_IP = "10.0.0.5"
RULES = ranked_rules((CLAIMANT_IP, 1), (SECOND_CLAIMANT_IP, 1), (HOLDER_IP, 1))

HOLDER_THREAD = "thread-holder"
CLAIMANT_THREAD = "thread-claimant"
SECOND_CLAIMANT_THREAD = "thread-second-claimant"

# How long a test waits for something that must happen promptly, and for something that
# legitimately waits behind a short grace window. Both are bounds, not delays.
PROMPT_S = 1.5
AFTER_GRACE_S = 8.0


def meta(ip):
    return {"ip": ip, "is_main": True}


class SaveCall:
    """One call of the manager's slot save, as the fake recorded it."""

    def __init__(self, port, model_tag, slot, kwargs):
        self.port = port
        self.model_tag = model_tag
        self.slot = slot
        self.kwargs = kwargs
        # The thread the save is keyed on: an explicit override or else the slot it is called with.
        self.thread = kwargs.get("thread_id_override") or getattr(slot, "thread_id", None)


class World:
    """One manager, one box slot, a fake engine whose turns end only when told to.

    ``send`` submits a request through the real ``submit`` and labels it. The fake completion
    call stamps ``events`` with ("start", slot_id) when a turn begins and ("end", slot_id) when
    it finishes, and the fake slot save stamps ("save", thread), so a test can assert the ORDER
    of the engine-visible steps with one list.
    """

    def __init__(self, mgr, spawns):
        self.mgr = mgr
        self.spawns = spawns
        self.slots = {}          # label -> Slot
        self.labels = {}         # slot_id -> label
        self.events = []         # ("start"|"end", slot_id) and ("save", thread), in order
        self.saves = []          # SaveCall per slot save the manager attempted
        self.handles = {}        # slot_id -> engine handle the turn ran on
        self.unloads = []        # resident keys passed to _begin_unload_locked
        self.before_return = {}  # slot_id -> sync callable run right before its turn returns
        self.decisions = []      # (name, marker ran?) per boundary decision, see same_tick_probe
        self.marker = None       # {"ran": bool} set by same_tick_probe
        self._gates = {}
        self._open = False

    # -- fake completion -------------------------------------------------
    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def complete(self, slot, handle):
        self.events.append(("start", slot.slot_id))
        self.handles[slot.slot_id] = handle
        if not self._open:
            await self.gate(slot.slot_id).wait()
        hook = self.before_return.get(slot.slot_id)
        if hook is not None:
            hook()  # sync: nothing can run between this and the return below
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

    async def parked(self, label, timeout=PROMPT_S, count=1):
        """Wait until ``count`` requests sit in the resident's inbox (the real HIT path put them
        there) and ``label`` is no longer in staging."""
        slot = self.slots[label]
        r = resident_for(self.mgr, MODEL)
        await wait_until(lambda: r.inbox.qsize() >= count and slot not in self.mgr.queue._staging,
                         timeout=timeout)
        return r

    async def turn_done(self, label, timeout=PROMPT_S):
        """Wait until ``label``'s turn has completed (it left ACTIVE)."""
        slot = self.slots[label]
        await wait_until(
            lambda: slot.state in (SlotState.GRACE, SlotState.POPPED), timeout=timeout,
        )

    async def settled(self, label, timeout=AFTER_GRACE_S):
        """Wait until ``label`` either started its turn or its future is done (dropped), and
        say which: "started", "answered" (done without ever starting) or "pending" (neither
        within ``timeout``). Bounded, never an unbounded wait."""
        slot = self.slots[label]
        try:
            await wait_until(
                lambda: label in self.order() or slot.completion_future.done(), timeout=timeout,
            )
        except AssertionError:
            return "pending"
        return "started" if label in self.order() else "answered"

    # -- claims ----------------------------------------------------------
    def claim_slots(self):
        """Slots holding a registered claim right now (a plain read, never prunes)."""
        return [c["slot"] for c in self.mgr._fastlane_claims.values()]

    def has_claim(self, label):
        return self.slots[label] in self.claim_slots()

    def snapshot_ids(self):
        """Slot ids the claims snapshot lists (it prunes dead claims first, like every read)."""
        return [row["slot_id"] for row in self.mgr.fastlane_claims_snapshot()]

    def same_tick_probe(self, gone):
        """A hook for ``before_return``: set the disconnect and schedule a marker callback that
        the loop can only run once the current task yields. The boundary-decision spies record
        whether the marker had run when the decision was made, so a test can show that no
        suspension separated the disconnect from the decision."""
        def hook():
            gone.set()
            self.marker = {"ran": False}
            asyncio.get_running_loop().call_soon(self.marker.__setitem__, "ran", True)
        return hook


@contextlib.asynccontextmanager
async def build_world(tmp_path, *, grace_seconds=1, max_grace_extensions=50,
                      idle_hot_load_seconds=600):
    """A manager with ONE box slot and the dispatcher running, over three ranked clients.

    The manager's slot save (``_save_slot_kv``, the one writer every engine slot save goes
    through) is replaced by a recorder, so no test ever makes a save request to the fake
    engine's port. The per-turn clean-prefix save is made to decline at its single-series gate
    (it counts one other active resident), so a slot save seen in ``world.saves`` cannot have
    come from the clean-prefix path.
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

    def spy_unload(r):
        world.unloads.append(r.resident_key)
        return real_unload(r)

    mgr._begin_unload_locked = spy_unload

    async def fake_save(port, model_tag, slot=None, **kwargs):
        call = SaveCall(port, model_tag, slot, kwargs)
        world.saves.append(call)
        world.events.append(("save", call.thread))
        return True

    mgr._save_slot_kv = fake_save
    mgr._other_active_residents = lambda handle: 1

    # Record, for every boundary decision the manager makes about a holder, whether the marker
    # of ``same_tick_probe`` had already run. Only methods that exist are wrapped.
    for name in ("_is_designated_unload_target_locked", "_parked_claim_outranks_holder_locked"):
        original = getattr(mgr, name, None)
        if original is None:
            continue

        def make_spy(fn, label):
            def spy(*args, **kw):
                if world.marker is not None:
                    world.decisions.append((label, world.marker["ran"]))
                return fn(*args, **kw)
            return spy

        setattr(mgr, name, make_spy(original, name))

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            yield world
    finally:
        world.open_everything()
        await mgr.shutdown()


async def hold_and_park(world, label="C", ip=CLAIMANT_IP, thread=CLAIMANT_THREAD, **kw):
    """The defect shape: the holder is mid-turn, and a request is parked in its inbox.

    The holder's first turn is running (its fake completion is blocked), then the second
    request is submitted through the real ``submit``; the dispatcher routes it to the
    resident's inbox through the real HIT path. Nothing is written into the claim registry by
    hand. Returns (holder slot, parked slot, resident).
    """
    h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
    assert await world.started("H1", timeout=5.0), "setup: the holder's turn never started"
    parked = await world.send(label, ip, thread, **kw)
    r = await world.parked(label)
    assert r.state is ResidentState.ACTIVE and r.active_slot is h1, (
        "setup: the holder must still be mid-turn"
    )
    assert parked.state is SlotState.STAGED, "setup: a parked request is still queued"
    return h1, parked, r


def assert_dropped(world, label, r, h1):
    """The parked request ``label`` was dropped and the holder was not preempted for it."""
    slot = world.slots[label]
    assert label not in world.order(), (
        f"a turn was started for a request whose client had disconnected: order {world.order()}"
    )
    fut = slot.completion_future
    assert fut.done() and not fut.cancelled(), (
        "the dropped request's future was not resolved (the client's caller would hang)"
    )
    # The same outcome the existing evicted-rider handling gives: the future fails with
    # SlotEvictedError.
    assert isinstance(fut.exception(), SlotEvictedError), (
        f"the dropped request was resolved some other way: {fut.exception()!r}"
    )
    assert r.inbox.qsize() == 0, "the dropped request is still in the resident's inbox"
    assert slot not in world.mgr.queue._staging and slot not in world.mgr.queue._accept_buf, (
        "the dropped request was put back in the staging queue"
    )
    assert not world.has_claim(label), "the dropped request still holds a claim"
    # The holder is not preempted: the same engine process, never unloaded or respawned.
    assert resident_for(world.mgr, MODEL) is r and r.state is not ResidentState.DEAD
    assert world.spawns == [MODEL] and world.unloads == [], (
        f"the resident was reloaded or unloaded: spawns={world.spawns} unloads={world.unloads}"
    )
    assert r.handle is world.handles[h1.slot_id], "the engine was replaced"


# --------------------------------------------------------------------------------------
# (1)-(4) the drop itself
# --------------------------------------------------------------------------------------

async def test_disconnected_parked_claimant_is_never_served_and_no_swap_happens(tmp_path):
    """A higher-ranked request is parked in the holder's inbox through the real HIT path and its
    client disconnects while the holder's turn is still running. The holder's turn then ends.
    The parked request must never get a turn (it may not appear in the turn-start order), the
    resident must stay the same engine (no unload, no respawn), the request's future must be
    failed with ``SlotEvictedError`` (what the existing evicted-rider handling does) so the
    caller is answered, and the request must be gone from the inbox and the staging queue with
    its claim released. If the dead request were still admitted the order would be H1, C."""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(world, disconnect_event=(gone := asyncio.Event()))
        gone.set()
        world.release("H1")
        await world.turn_done("H1")
        outcome = await world.settled("C")
        assert outcome == "answered", (
            f"the disconnected parked request was not dropped (outcome {outcome}); "
            f"order {world.order()}"
        )
        assert world.order() == ["H1"], f"turn-start order {world.order()}"
        assert_dropped(world, "C", r, h1)


async def test_no_kv_save_is_made_for_a_disconnected_claimant(tmp_path):
    """Same sequence as the test above, watched through the fake slot save. A live higher-ranked
    parked request makes the holder save its KV before the swap (see the control below); a
    disconnected one must not, because the holder is not preempted and nobody takes the engine.
    No slot save may be recorded at all up to the point where the dead request is resolved: the
    fixture makes the per-turn clean-prefix save decline, so no other path writes here either.
    A save stamped in the event list would sit between the end of H1 and the turn of C. This
    test looks only at saves, so it does not itself say whether C was dropped (the test above
    does); the control below shows the same fixture does record a save for a live claimant."""
    async with build_world(tmp_path) as world:
        _h1, _claimant, _r = await hold_and_park(world, disconnect_event=(gone := asyncio.Event()))
        gone.set()
        world.release("H1")
        await world.turn_done("H1")
        # Wait until the dead request is resolved one way or the other (dropped or, wrongly,
        # started), so the save record below covers the whole boundary; "pending" would mean the
        # look happened before anything was decided and proves nothing.
        outcome = await world.settled("C")
        assert outcome != "pending", (
            f"setup: the boundary decision for the parked request never happened; "
            f"events {world.named_events()}"
        )
        assert world.saves == [], (
            f"a slot save was made for a request nobody waits for: {world.named_events()}"
        )
        assert not [e for e in world.named_events() if e[0] == "save"]


async def test_holder_keeps_its_grace_and_its_follow_up_is_served_warm(tmp_path):
    """After the dead request is dropped the holder is an ordinary holder again. Its same-thread
    follow-up (H2), sent right after the turn ended, is served warm on the SAME engine inside the
    grace window. The claims snapshot no longer lists the dead claim, the holder is neither
    designated as an unload target nor outranked by a parked claim, and nothing was unloaded or
    respawned. When the follow-up is done and the grace window runs out, the dead request is
    still never served: it is dropped. (The first half guards the holder's grace; the last
    clause is what fails if the dead request is admitted once the holder goes quiet.)"""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(world, disconnect_event=(gone := asyncio.Event()))
        gone.set()
        world.release("H1")
        await world.turn_done("H1")
        h2 = await world.send("H2", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H2", timeout=PROMPT_S), f"order {world.order()}"
        assert world.order() == ["H1", "H2"], (
            f"the holder lost its grace to a parked request whose client had disconnected: "
            f"{world.order()}"
        )
        assert world.handles[h2.slot_id] is world.handles[h1.slot_id], "the follow-up was not warm"
        assert claimant.slot_id not in world.snapshot_ids(), "the dead claim is still listed"
        assert not world.has_claim("C")
        async with world.mgr._registry_lock:
            assert not world.mgr._is_designated_unload_target_locked(r), (
                "the holder is designated because of a client that disconnected"
            )
            assert not world.mgr._parked_claim_outranks_holder_locked(r), (
                "the holder is outranked by a parked claim of a client that disconnected"
            )
        assert resident_for(world.mgr, MODEL) is r and r.state is not ResidentState.DEAD
        assert world.spawns == [MODEL] and world.unloads == []

        world.release("H2")
        outcome = await world.settled("C")
        assert outcome == "answered", (
            f"the dead request was not dropped once the holder went quiet (outcome {outcome}); "
            f"order {world.order()}"
        )
        assert world.order() == ["H1", "H2"], f"turn-start order {world.order()}"
        assert_dropped(world, "C", r, h1)
        assert world.saves == [], f"a slot save was made: {world.named_events()}"


async def test_disconnect_at_admit_time_not_only_before(tmp_path):
    """The disconnect is set from inside the fake completion call of the holder, right before it
    returns, so it lands in the very tick the holder's turn ends: there is no suspension between
    the disconnect and the holder's boundary decision (checked below with a marker callback that
    the loop could only have run at a suspension). The outcome must be exactly that of the
    earlier disconnect: the dead request never gets a turn, the engine is the same, its future
    is failed with ``SlotEvictedError``, it is out of the inbox and staging with its claim
    released, and no slot save is made for it."""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(world, disconnect_event=(gone := asyncio.Event()))
        assert not gone.is_set(), "setup: the client must still be connected"
        world.before_return[h1.slot_id] = world.same_tick_probe(gone)
        world.release("H1")
        await world.turn_done("H1")
        assert gone.is_set(), "setup: the disconnect hook did not run"
        assert world.decisions, "setup: no boundary decision was consulted after the disconnect"
        assert world.decisions[0][1] is False, (
            f"setup: the loop ran another callback between the disconnect and the holder's "
            f"first boundary decision: {world.decisions}"
        )
        outcome = await world.settled("C")
        assert outcome == "answered", (
            f"the parked request whose client disconnected in the boundary tick was not dropped "
            f"(outcome {outcome}); order {world.order()}"
        )
        assert world.order() == ["H1"], f"turn-start order {world.order()}"
        assert_dropped(world, "C", r, h1)
        assert world.saves == [], f"a slot save was made: {world.named_events()}"


# --------------------------------------------------------------------------------------
# (5)-(6) controls
# --------------------------------------------------------------------------------------

async def test_control_a_live_parked_claimant_is_still_served_next(tmp_path):
    """Control for the drop. The identical sequence with the client still connected: the parked
    higher-ranked request's turn starts at the holder's boundary, the holder's KV is saved first
    (one slot save for the holder's thread, between the end of H1 and the start of C), and the
    holder's same-thread follow-up is served after it, not dropped. This shows the drop above is
    not a blanket break of the parked-request path, and that the same fixture does show a save
    and a swap when there is a live claimant to serve."""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(world, disconnect_event=asyncio.Event())
        world.release("H1")
        await world.turn_done("H1")
        h2 = await world.send("H2", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("C", timeout=PROMPT_S), (
            f"the live parked request was not served at the holder's boundary: order {world.order()}"
        )
        assert world.order() == ["H1", "C"], f"turn-start order {world.order()}"
        events = world.named_events()
        assert ("save", HOLDER_THREAD) in events, f"no KV save for the holder: {events}"
        assert (
            events.index(("end", "H1")) < events.index(("save", HOLDER_THREAD))
            < events.index(("start", "C"))
        ), f"the holder's KV was not saved between its turn and the claimant's: {events}"
        assert world.handles[claimant.slot_id] is world.handles[h1.slot_id]

        world.release("C")
        assert await world.started("H2", timeout=AFTER_GRACE_S), f"order {world.order()}"
        assert world.order() == ["H1", "C", "H2"], f"turn-start order {world.order()}"
        assert h2.state is not SlotState.STAGED, "the holder's follow-up was dropped, not re-queued"
        assert world.spawns == [MODEL] and world.unloads == []
        assert resident_for(world.mgr, MODEL) is r and r.state is not ResidentState.DEAD


async def test_control_disconnect_after_the_turn_started_does_not_cut_the_turn(tmp_path):
    """The parked request's turn has already started when its client disconnects. The drop only
    concerns a request still waiting in the inbox: the running turn is not cut by it. The
    manager has no disconnect handling of its own for a running turn (the real completion call
    owns that), so with this fake the turn keeps running until it is released and then
    completes normally, the future carries the answer, and the engine is untouched."""
    async with build_world(tmp_path) as world:
        h1, claimant, r = await hold_and_park(world, disconnect_event=(gone := asyncio.Event()))
        world.release("H1")
        assert await world.started("C", timeout=PROMPT_S), f"order {world.order()}"
        gone.set()
        # A bounded look, well under 1 s: the running turn must still be running.
        await asyncio.sleep(0.3)
        assert not claimant.completion_future.done(), (
            "the running turn was cut when its client disconnected"
        )
        assert claimant.state is SlotState.ACTIVE
        assert world.order() == ["H1", "C"]

        world.release("C")
        done, _ = await asyncio.wait({claimant.completion_future}, timeout=PROMPT_S)
        assert done and claimant.completion_future.result()["ok"], (
            "the running turn did not complete normally after its client disconnected"
        )
        assert world.spawns == [MODEL] and world.unloads == []
        assert resident_for(world.mgr, MODEL) is r and r.state is not ResidentState.DEAD
        assert world.handles[claimant.slot_id] is world.handles[h1.slot_id]


# --------------------------------------------------------------------------------------
# (7) two parked requests, one dead
# --------------------------------------------------------------------------------------

async def test_two_parked_requests_one_disconnected(tmp_path):
    """Two higher-ranked requests are parked in the holder's inbox, from two different clients:
    C1 (the higher rank) whose client then disconnects, and C2 (live). At the holder's boundary
    C2 is served and C1 is dropped: C1 never starts, C2 does, the turn-start order is H1, C2,
    C1's future is failed with ``SlotEvictedError``, and the engine is the same. Were the dead
    request admitted first (it outranks C2) the order would be H1, C1."""
    async with build_world(tmp_path) as world:
        h1, c1, r = await hold_and_park(
            world, label="C1", ip=CLAIMANT_IP, thread=CLAIMANT_THREAD,
            disconnect_event=(gone := asyncio.Event()),
        )
        c2 = await world.send("C2", SECOND_CLAIMANT_IP, SECOND_CLAIMANT_THREAD)
        await world.parked("C2", count=2)
        assert world.has_claim("C1") and world.has_claim("C2"), (
            "setup: both parked requests must hold a claim before the disconnect"
        )
        gone.set()
        world.release("H1")
        await world.turn_done("H1")
        try:
            await wait_until(lambda: len(world.order()) >= 2, timeout=AFTER_GRACE_S)
        except AssertionError:
            pass
        assert world.order() == ["H1", "C2"], (
            f"the live parked request was not served in place of the dead one: {world.order()}"
        )
        # C1 is resolved at the latest by the admit that picked C2.
        await wait_until(lambda: c1.completion_future.done(), timeout=PROMPT_S)
        assert_dropped(world, "C1", r, h1)
        assert world.handles[c2.slot_id] is world.handles[h1.slot_id]
        assert not world.has_claim("C2"), "C2's claim outlived the start of its own turn"
