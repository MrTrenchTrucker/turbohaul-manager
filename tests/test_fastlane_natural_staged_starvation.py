"""A top-ranked Fast Lane request of the same model while two engines are loaded, reached naturally.

Two real engines of one model are loaded (the first engine's registry key is the model tag, the
second one's is ``<tag>#1``): the first is in the middle of a turn for a lower-ranked client, the
second is inside its grace window after a served turn. A request of the top rule for the same model
then arrives through the real ``submit`` and the real dispatcher. Nothing about the request is forged
and nothing is held aside for it: it takes whatever path the manager's own routing gives it.

What the tests pin, in plain words:

* the engine budget of the box and the fake free-memory reading are explicit pins (a third engine is
  either impossible or possible, on purpose);
* a new arrival is handed to the least loaded engine, and an engine that is only sitting in its grace
  window after a finished turn is not loaded: the arrival goes to it and not to the engine that is
  really in the middle of a turn;
* a turn that is really running still counts as load, whether it is a first turn or a follow-up served
  inside a grace window, and two idle engines still share new sessions one per engine, so the rule
  above does not pile work onto a busy engine;
* the expected outcome: a loaded holder that the request outranks gives up its grace window
  (``grace_designated_victim_skip`` or ``grace_designated_unload_target_break`` on that holder's own
  slot), or the request itself is admitted (``active`` on its own slot), within a bounded time. With
  a chain of follow-ups keeping the grace engine busy the grace engine is really running a turn, so
  the request waits for it: that case is only checked after the busy engine's turn ends;
* every test asserts its setup first (two live engines, which one is mid-turn, which one is in grace),
  and reads what the manager did with the request (verdict, chosen engine, park, defer, make-room
  starvation line) through spies that call the real code.

One process, asyncio only, three engines at most, bounded waits.
"""

import asyncio
import contextlib
import json
import os
import sys
import time

import pytest
from _multiinstance_support import _manifest, _new_manager, _probe

from turbohaul import manager as manager_module
from turbohaul.config import FastLaneConfig, FastLaneRule, FastLaneTagRanks
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db

TAG = "qwen-duo"
SECOND_KEY = f"{TAG}#1"

# Priority order: index 0 outranks index 1, which outranks index 2.
HIGH = "1.1.1.1"
LOW = "2.2.2.2"
LOWEST = "3.3.3.3"
RULES = [
    FastLaneRule(address=HIGH, label="high", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address=LOW, label="low", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address=LOWEST, label="lowest", tag_ranks=FastLaneTagRanks(main=1)),
]

GRACE_S = 5          # grace window of the main tests
CONTROL_GRACE_S = 3  # grace window of the controls
SETUP_S = 5.0        # bound for setup steps that must happen promptly
HANDLED_S = 3.0      # bound for the dispatcher to hand the new request somewhere
RELEASE_AFTER_S = 1.0   # the claimant waits this long before the busy engine's turn is released
RELEASE_WINDOW_S = 2.0  # bound for the outcome after the release
CHAIN_GAP_S = 0.15      # gap between two same-client follow-ups of the chain
EARLY_BREAK_EVENTS = ("grace_designated_victim_skip", "grace_designated_unload_target_break")
CLAIMANT = "C"


def events(boot, slot_id):
    """The audit event types of one slot, oldest first."""
    conn = open_state_db(boot.storage.state_db_path)
    try:
        cur = conn.execute(
            "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id", (slot_id,),
        )
        return [row["event_type"] for row in cur.fetchall()]
    finally:
        conn.close()


class Scene:
    """One manager over a fake engine whose turns end only when told to, with spies that call the real code.

    ``send`` submits a request through the real ``submit``. The fake completion call records, for each
    request, the engine port its turn ran on, then blocks only for requests that were held with
    ``hold``. The spies record, around the real methods: every arrival verdict, every engine the
    routing picks, every claim park, every defer, every make-room starvation line, every engine start,
    every look-up of the instance-0-only resident from the routing routine, and every request the
    picker hands to the dispatcher.
    """

    def __init__(self, tmp_path, *, grace_s, budget=2):
        self.mgr, self.boot, _spawns = _new_manager(tmp_path, budget=budget)
        self.mgr.runtime.fastlane = FastLaneConfig(enabled=True, rules=RULES)
        self.mgr.runtime.queue.grace_seconds = grace_s
        self.mgr.runtime.queue.max_grace_extensions = 200
        self.mgr.runtime.queue.idle_hot_load_seconds = 600
        # One card per engine: this is what lets the model run on several engines at once.
        _manifest(self.boot, TAG, auto_place=True, llama_server_flags={"split_mode": "none"})
        self.slots = {}    # label -> Slot
        self.port_of = {}  # label -> engine port its turn ran on
        self.verdicts = []   # (prompt, admit, decline reason)
        self.picks = []      # (prompt, engine key chosen by the least-loaded pick)
        self.parks = []      # (slot id, engine key, calling function)
        self.defers = []     # (prompt, calling function)
        self.starved = []    # prompt of each make-room starvation line
        self.starts = []     # (prompt, engine key) per engine started for a request
        self.legacy_lookups = []  # calling function of each instance-0-only resident look-up
        self.picked_by_dispatcher = []  # prompt of each request the picker handed out
        self.matched_pops = 0   # pop_matched_thread calls
        self.trace = []      # periodic reads of the designation predicates
        self.loads_at_arrival = {}  # engine key -> load figure read by the least-loaded pick, at the claimant's submit
        self._gates = {}
        self._open = False
        self.mgr._complete_fn = self._complete
        self._install_spies()

    # -- spies ---------------------------------------------------------------------------------
    def _install_spies(self):
        m = self.mgr
        real_park = m._park_fastlane_claim_locked
        real_pick = m._pick_instance_for
        real_defer = m._defer_unroutable
        real_starved = m._log_make_room_starved
        real_reserve = m._reserve_and_start_locked
        real_live = m._live_resident_for
        real_pop_next = m.queue.pop_next
        real_matched = m.queue.pop_matched_thread
        real_verdict = manager_module.arrival_verdict

        def spy_park(slot, r):
            self.parks.append((slot.slot_id, r.resident_key, sys._getframe(1).f_code.co_name))
            return real_park(slot, r)

        def spy_pick(slot, tag, insts):
            chosen = real_pick(slot, tag, insts)
            self.picks.append((slot.prompt, None if chosen is None else chosen.resident_key))
            return chosen

        def spy_defer(slot, *, evict_pending=False):
            self.defers.append((slot.prompt, sys._getframe(1).f_code.co_name))
            return real_defer(slot, evict_pending=evict_pending)

        def spy_starved(slot, reason, attribution):
            self.starved.append(slot.prompt)
            return real_starved(slot, reason, attribution)

        async def spy_reserve(slot, *a, **k):
            r = await real_reserve(slot, *a, **k)
            self.starts.append((slot.prompt, None if r is None else r.resident_key))
            return r

        def spy_live(model_tag):
            self.legacy_lookups.append(sys._getframe(1).f_code.co_name)
            return real_live(model_tag)

        async def spy_pop_next(**kw):
            slot = await real_pop_next(**kw)
            if slot is not None:
                self.picked_by_dispatcher.append(slot.prompt)
            return slot

        async def spy_matched(*a, **k):
            self.matched_pops += 1
            return await real_matched(*a, **k)

        async def spy_verdict(host, tag, insts, eff_cap, slot):
            v = await real_verdict(host, tag, insts, eff_cap, slot)
            self.verdicts.append((slot.prompt, v.admit, None if v.reason is None else str(v.reason)))
            return v

        m._park_fastlane_claim_locked = spy_park
        m._pick_instance_for = spy_pick
        m._defer_unroutable = spy_defer
        m._log_make_room_starved = spy_starved
        m._reserve_and_start_locked = spy_reserve
        m._live_resident_for = spy_live
        m.queue.pop_next = spy_pop_next
        m.queue.pop_matched_thread = spy_matched
        manager_module.arrival_verdict = spy_verdict
        self._restore_verdict = lambda: setattr(manager_module, "arrival_verdict", real_verdict)

    async def _complete(self, slot, handle):
        self.port_of[slot.prompt] = handle.port
        gate = self._gates.get(slot.prompt)
        if gate is not None and not self._open:
            await gate.wait()
        return {"ok": True, "model": handle.model_tag}

    def hold(self, label):
        self._gates[label] = asyncio.Event()

    def release(self, label):
        self._gates[label].set()

    def is_held(self, label):
        gate = self._gates.get(label)
        return gate is not None and not gate.is_set()

    def open_everything(self):
        self._open = True
        for gate in self._gates.values():
            gate.set()

    async def send(self, label, ip, thread):
        slot = await self.mgr.submit(
            TAG, prompt=label, thread_id=thread, client_meta={"ip": ip, "is_main": True},
            wait_for_completion=True,
        )
        self.slots[label] = slot
        return slot

    def ev(self, label):
        return events(self.boot, self.slots[label].slot_id)

    async def until(self, predicate, timeout):
        """True when ``predicate`` holds within ``timeout`` seconds (bounded poll)."""
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)

    def engines(self):
        return {r.resident_key: r for r in self.mgr._instances_for(TAG)}

    def engine_on(self, port):
        return next(r for r in self.engines().values() if r.port == port)

    def claim_of(self, label):
        slot = self.slots[label]
        for c in self.mgr._fastlane_claims.values():
            if c["slot"] is slot:
                return c
        return None

    def holders(self):
        return [label for label in self.slots if label != CLAIMANT]

    # -- reads of the designation predicates, taken on the loop between awaits -----------------
    def sample(self, started):
        m = self.mgr
        governing = m._governing_claim_locked(include_parked=False)
        per_engine = {}
        for key, r in self.engines().items():
            per_engine[key] = {
                "state": r.state.name,
                "in_grace_loop": r.in_grace_loop,
                "inbox": r.inbox.qsize() if r.inbox is not None else None,
                "designated": m._is_designated_unload_target_locked(r),
                "parked_outranks": m._parked_claim_outranks_holder_locked(r),
            }
        victim = m._lru_idle_unloadable()
        claim = self.claim_of(CLAIMANT) if CLAIMANT in self.slots else None
        self.trace.append({
            "t": round(time.monotonic() - started, 2),
            "governing_unparked": None if governing is None else governing[1].prompt,
            "eligible_victim": None if victim is None else victim.resident_key,
            "claimant_state": self.slots[CLAIMANT].state.name if CLAIMANT in self.slots else None,
            "claimant_parked_on": None if claim is None else claim.get("parked_on"),
            "engines": per_engine,
        })

    async def sampler(self, started):
        while True:
            self.sample(started)
            await asyncio.sleep(0.1)

    def report(self):
        return {
            "verdicts": self.verdicts, "picks": self.picks, "parks": self.parks, "defers": self.defers,
            "make_room_starved": self.starved, "engine_starts": self.starts,
            "instance_zero_lookups_from": sorted(set(self.legacy_lookups)),
            "picker_handed_out": self.picked_by_dispatcher, "matched_thread_polls": self.matched_pops,
            "loads_at_arrival": self.loads_at_arrival,
            "events": {label: self.ev(label) for label in [*self.holders(), CLAIMANT] if label in self.slots},
            "trace_tail": self.trace[-3:],
        }


def dump(name, scene):
    """Write the report to a directory named by the environment, when there is one."""
    target = os.environ.get("NATURAL_STAGED_TRACE_DIR")
    if not target:
        return
    with open(os.path.join(target, f"{name}.json"), "w") as fh:
        json.dump({**scene.report(), "trace": scene.trace}, fh, indent=1, default=str)


@contextlib.asynccontextmanager
async def build_scene(tmp_path, request, *, grace_s, cards, budget=2):
    """A manager with the dispatcher running and ``cards`` cards reporting plenty of free memory.

    The card probe is replaced at both places the manager reads it, so no real host reading is
    involved. ``budget`` is the box budget of engines, ``cards`` decides how many can be placed.
    """
    scene = Scene(tmp_path, grace_s=grace_s, budget=budget)
    probe_a, probe_b = _probe([20000] * cards)
    scene.mgr._worker_task = asyncio.create_task(scene.mgr.worker_loop())
    started = time.monotonic()
    sampler = asyncio.create_task(scene.sampler(started))
    try:
        with probe_a, probe_b:
            yield scene
    finally:
        sampler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sampler
        scene.open_everything()
        with contextlib.suppress(Exception):
            dump(request.node.name.replace("/", "_").replace("[", "_").replace("]", ""), scene)
        scene._restore_verdict()
        await scene.mgr.shutdown()


def early_break_seen(s):
    return any(name in s.ev(label) for label in s.holders() for name in EARLY_BREAK_EVENTS)


def outcome(s):
    """The expected outcome: a holder gave up its grace on its own slot, or the claimant was admitted."""
    return early_break_seen(s) or (CLAIMANT in s.slots and "active" in s.ev(CLAIMANT))


# --------------------------------------------------------------------------------------
# Scene builders
# --------------------------------------------------------------------------------------

async def start_two_engines(s, ip_a, ip_b):
    """Two engines of the model, each running one held turn: A1 for ``ip_a`` and B1 for ``ip_b``.
    The turn submitted first gets the first engine (registry key = the tag), the other one the second."""
    s.hold("A1")
    s.hold("B1")
    for label, ip, thread in (("A1", ip_a, "thread-a"), ("B1", ip_b, "thread-b")):
        await s.send(label, ip, thread)
        assert await s.until(lambda label=label: label in s.port_of, SETUP_S), (
            f"setup: the turn {label} never started"
        )
    assert sorted(s.engines()) == [TAG, SECOND_KEY], (
        f"setup: expected two engines of one model, got {sorted(s.engines())}"
    )
    assert s.port_of["A1"] != s.port_of["B1"], "setup: both turns ran on one engine"


async def serve_follow_up(s, label, ip, thread, anchor):
    """A same-thread follow-up served warm on the engine of ``anchor`` inside its grace window."""
    await s.send(label, ip, thread)
    assert await s.until(lambda: label in s.port_of, SETUP_S), (
        f"setup: the follow-up {label} was not served inside the grace window"
    )
    assert s.port_of[label] == s.port_of[anchor], (
        f"setup: the follow-up {label} was not served warm on its own thread's engine"
    )


def grace_slot_is_finished(r):
    """True when engine ``r`` sits in its grace window with the slot of a turn that has finished: the
    grace flag is set, the slot set on the engine is in state GRACE, nothing is in flight and the inbox
    is empty. Right after a follow-up completes, the slot is still counted as running for a few
    milliseconds until the manager moves it to GRACE, so a setup must wait for this state."""
    slot = r.active_slot
    return bool(
        r.in_grace_loop and slot is not None and getattr(slot, "state", None) == "GRACE"
        and not r.inflight and r.inbox is not None and r.inbox.qsize() == 0
    )


async def settle_grace_engine(s, r):
    """Wait (bounded) until ``r`` is in its grace window with a finished slot, and fail the test with a
    setup message when it is not reached."""
    assert await s.until(lambda: grace_slot_is_finished(r), SETUP_S), (
        f"setup: engine {r.resident_key} never settled in its grace window with a finished slot: "
        f"in grace loop={r.in_grace_loop}, slot state="
        f"{getattr(r.active_slot, 'state', None)!r}, in flight={len(r.inflight)}, "
        f"inbox={r.inbox.qsize() if r.inbox is not None else None}"
    )


async def busy_engine_and_grace_engine(s):
    """Engine of A (the first engine) is mid-turn for a lower-ranked client (held); the engine of B
    (the second) is inside its grace window after a served turn, with its inbox empty and the slot of
    that turn finished (waited for, since the manager moves the slot to its finished state a few
    milliseconds after the turn completes). The setup asserts every part of that. Returns the busy
    engine and the grace engine."""
    await start_two_engines(s, LOW, LOW)
    s.release("B1")
    assert await s.until(lambda: "grace_enter" in s.ev("B1"), SETUP_S), (
        f"setup: the second engine's holder never entered grace: {s.ev('B1')}"
    )
    await serve_follow_up(s, "B2", LOW, "thread-b", "B1")
    busy = s.engine_on(s.port_of["A1"])
    grace = s.engine_on(s.port_of["B1"])
    assert busy is not grace, "setup: the running turn and the grace window are on one engine"
    assert (busy.resident_key, grace.resident_key) == (TAG, SECOND_KEY), "setup: unexpected engine keys"
    await settle_grace_engine(s, grace)
    assert s.is_held("A1") and busy.active_slot is s.slots["A1"], (
        f"setup: engine {busy.resident_key} is not mid-turn on A1"
    )
    assert s.slots["B1"].state is SlotState.GRACE and grace.in_grace_loop, (
        f"setup: engine {grace.resident_key} is not in grace: {s.slots['B1'].state.name}"
    )
    assert grace.inbox.qsize() == 0 and busy.inbox.qsize() == 0, "setup: an inbox is not empty"
    assert not s.defers and not s.starved and not s.starts[2:], (
        f"setup: the setup itself deferred or starved a request: {s.report()}"
    )
    return busy, grace


async def submit_claimant(s, ip, *, pin_to=None):
    """A real submit of the claimant. ``pin_to`` pins the claimant's session to that engine with the
    session record a start writes (an explicit choice among the two real engines); without it the
    manager's own pick decides. Waits (bounded) until the dispatcher has handed the request to
    an engine or deferred it, and returns how it ended up."""
    s.loads_at_arrival = {
        k: {"load": engine_load(r), "inflight": len(r.inflight), "has_active_slot": r.active_slot is not None,
            "inbox": r.inbox.qsize(), "in_grace_loop": r.in_grace_loop,
            "last_active": round(r.last_active_monotonic, 3)}
        for k, r in s.engines().items()
    }
    if pin_to is not None:
        async with s.mgr._registry_lock:
            s.mgr._remember_affinity(TAG, "tid:thread-c", pin_to.resident_key)
    await s.send(CLAIMANT, ip, "thread-c")
    slot = s.slots[CLAIMANT]
    handled = await s.until(
        lambda: any(p[0] == slot.slot_id for p in s.parks) or any(d[0] == CLAIMANT for d in s.defers)
        or "active" in s.ev(CLAIMANT),
        HANDLED_S,
    )
    return handled


def engine_load(r):
    """The raw counts of one engine: riders in flight, plus one for a slot set on the engine, plus the
    inbox depth. Only reported in failure messages; the routing decision is read from the spies."""
    return len(r.inflight) + (1 if r.active_slot is not None else 0) + (r.inbox.qsize() if r.inbox is not None else 0)


def where_is_the_claimant(s):
    """What the manager did with the claimant, from the spies and the live state."""
    slot = s.slots[CLAIMANT]
    claim = s.claim_of(CLAIMANT)
    return {
        "state": slot.state.name,
        "in_staging": any(x is slot for x in s.mgr.queue._staging),
        "in_an_inbox": [r.resident_key for r in s.mgr._model_residents() if s.mgr._inbox_holds_slot(r, slot)],
        "claim_parked_on": None if claim is None else claim.get("parked_on"),
        "parks": [p[1:] for p in s.parks if p[0] == slot.slot_id],
        "pick": [p[1] for p in s.picks if p[0] == CLAIMANT],
        "defers": [d for d in s.defers if d[0] == CLAIMANT],
        "verdicts": [v[1:] for v in s.verdicts if v[0] == CLAIMANT],
        "make_room_starved": s.starved.count(CLAIMANT),
    }


async def chain_of_follow_ups(s, stop):
    """The grace engine's client keeps sending same-thread follow-ups, one every CHAIN_GAP_S, each
    served warm on its own engine, until ``stop`` is set (bounded by the caller)."""
    n = 3
    while not stop.is_set():
        label = f"B{n}"
        n += 1
        await s.send(label, LOW, "thread-b")
        await asyncio.sleep(CHAIN_GAP_S)


ROUTES = pytest.mark.parametrize("route", ["least-loaded", "pinned-to-busy-engine"])
CHAIN = pytest.mark.parametrize("chain", [False, True], ids=["no-chain", "follow-up-chain"])


# --------------------------------------------------------------------------------------
# What the manager does with a same-model arrival while two engines are loaded
# --------------------------------------------------------------------------------------

async def test_same_model_arrival_is_routed_to_the_engine_that_is_not_mid_turn(tmp_path, request):
    """Box budget two, so a third engine is impossible by the budget. The claimant is a real submit and
    takes the manager's own route. One engine is in the middle of a turn, the other one only sits in
    its grace window after a finished turn with an empty inbox. Read by the spies: the verdict
    declines a third engine, the least-loaded pick names the engine in its grace window (the finished
    turn there is not load), the claimant's claim is parked in that engine's inbox by the routing
    routine, and none of the other ways a request can stay waiting was taken (no defer, no
    make-room starvation line, no engine start, no look-up of the instance-0-only resident from the
    routing routine)."""
    async with build_scene(tmp_path, request, grace_s=GRACE_S, cards=2) as s:
        busy, grace = await busy_engine_and_grace_engine(s)
        counts = {r.resident_key: engine_load(r) for r in (busy, grace)}
        handled = await submit_claimant(s, HIGH)
        where = where_is_the_claimant(s)
        assert handled, f"the dispatcher never handed the claimant anywhere: {where}"
        claimant_verdicts = [v[1:] for v in s.verdicts if v[0] == CLAIMANT]
        assert len(claimant_verdicts) == 1 and claimant_verdicts[0][0] is False, (
            f"unexpected arrival verdict (a third engine must be declined): {where}"
        )
        assert not where["defers"] and where["make_room_starved"] == 0, f"claimant was deferred: {where}"
        assert not s.starts[2:] and not s.starved, f"an engine start or a starvation line happened: {s.report()}"
        assert "_route_or_reserve" not in s.legacy_lookups, (
            f"the instance-0-only look-up was reached from the routing routine: {s.legacy_lookups}"
        )
        assert where["pick"] == [grace.resident_key], (
            f"the claimant was not routed to the engine that is not mid-turn ({grace.resident_key}); "
            f"the mid-turn engine is {busy.resident_key}; raw counts before the arrival: {counts}; {where}"
        )
        assert where["parks"] == [(grace.resident_key, "_route_to")], (
            f"the claimant was not parked in the inbox of the engine that is not mid-turn: {where}"
        )


# --------------------------------------------------------------------------------------
# The expected outcome for a top-ranked claimant, whichever way it was routed
# --------------------------------------------------------------------------------------

async def test_top_ranked_arrival_gets_an_outcome_before_the_grace_window_ends(tmp_path, request):
    """One engine mid-turn (the turn is never released inside this test), the other in its grace window
    after a finished turn, a top-ranked request of the same model arrives and nothing is held aside; the
    manager's own least-loaded choice routes it to the engine in its grace window. Before the end of the
    grace window either a holder it outranks has given up its grace (on its own slot) or the claimant
    has been admitted (``active`` on its own slot)."""
    async with build_scene(tmp_path, request, grace_s=GRACE_S, cards=2) as s:
        busy, grace = await busy_engine_and_grace_engine(s)
        started = time.monotonic()
        await submit_claimant(s, HIGH)
        where = where_is_the_claimant(s)
        assert s.is_held("A1"), "setup: the running turn ended on its own"
        seen = await s.until(lambda: outcome(s), GRACE_S - 0.5)
        elapsed = time.monotonic() - started
        assert seen, (
            f"the top-ranked claimant was not admitted and no holder gave up its grace within "
            f"{GRACE_S - 0.5:.1f}s; claimant={where}; report={s.report()}"
        )
        assert elapsed < GRACE_S - 1.0, f"the outcome came after {elapsed:.2f}s of a {GRACE_S}s grace window"


@CHAIN
@ROUTES
async def test_top_ranked_arrival_gets_an_outcome_after_the_busy_engines_turn_ends(tmp_path, request, route, chain):
    """Same scene. The claimant waits for a fixed short time, then the busy engine's turn is released.
    Within a bounded time after that, either a holder it outranks has given up its grace or the
    claimant has been admitted, and the whole wait stays inside the grace window."""
    async with build_scene(tmp_path, request, grace_s=GRACE_S, cards=2) as s:
        busy, grace = await busy_engine_and_grace_engine(s)
        stop = asyncio.Event()
        chain_task = asyncio.create_task(chain_of_follow_ups(s, stop)) if chain else None
        try:
            started = time.monotonic()
            await submit_claimant(s, HIGH, pin_to=busy if route == "pinned-to-busy-engine" else None)
            where = where_is_the_claimant(s)
            await asyncio.sleep(max(0.0, RELEASE_AFTER_S - (time.monotonic() - started)))
            seen_before_release = outcome(s)
            s.release("A1")
            seen = await s.until(lambda: outcome(s), RELEASE_WINDOW_S)
            elapsed = time.monotonic() - started
            assert seen, (
                f"after the busy engine's turn ended, the top-ranked claimant was not admitted and no "
                f"holder gave up its grace within {RELEASE_WINDOW_S:.1f}s (route={route}, chain={chain}, "
                f"outcome before the release: {seen_before_release}); claimant={where}; report={s.report()}"
            )
            assert elapsed < GRACE_S - 1.0, f"the outcome came after {elapsed:.2f}s of a {GRACE_S}s window"
        finally:
            stop.set()
            if chain_task is not None:
                await chain_task


# --------------------------------------------------------------------------------------
# Controls (each asserts its setup and must pass)
# --------------------------------------------------------------------------------------

async def test_control_claimant_that_does_not_outrank_the_holders_changes_nothing(tmp_path, request):
    """The same scene with a claimant of the lowest rule: nobody gives up a grace window for it and
    it is not admitted. Asserts that the claimant really was handed to an engine inbox by the routing."""
    async with build_scene(tmp_path, request, grace_s=CONTROL_GRACE_S, cards=2) as s:
        await busy_engine_and_grace_engine(s)
        handled = await submit_claimant(s, LOWEST)
        where = where_is_the_claimant(s)
        assert handled and len(where["parks"]) == 1 and where["parks"][0][1] == "_route_to", (
            f"setup: the claimant was not routed into an engine inbox by the routing: {where}"
        )
        moved = await s.until(lambda: outcome(s), CONTROL_GRACE_S - 1.0)
        assert not moved, f"a holder gave up grace for a claimant that does not outrank it: {s.report()}"
        assert s.slots["B1"].state is SlotState.GRACE, "the grace holder left grace"


async def test_control_with_room_for_a_third_engine_the_claimant_is_simply_admitted(tmp_path, request):
    """Box budget three and three cards with room: the arrival verdict admits a third engine for the
    claimant, which is admitted on it, while both holders keep their state."""
    async with build_scene(tmp_path, request, grace_s=GRACE_S, cards=3, budget=3) as s:
        busy, grace = await busy_engine_and_grace_engine(s)
        handled = await submit_claimant(s, HIGH)
        where = where_is_the_claimant(s)
        assert [v[1] for v in s.verdicts if v[0] == CLAIMANT] == [True], f"setup: no third engine was admitted: {where}"
        assert await s.until(lambda: "active" in s.ev(CLAIMANT), SETUP_S), (
            f"the claimant was not admitted on the third engine (handled={handled}): {s.report()}"
        )
        assert sorted(s.engines()) == [TAG, SECOND_KEY, f"{TAG}#2"], f"expected three engines: {sorted(s.engines())}"
        assert s.is_held("A1") and busy.active_slot is s.slots["A1"], "the busy engine's turn was disturbed"
        assert not any(name in s.ev("B1") + s.ev("B2") for name in EARLY_BREAK_EVENTS), (
            f"the grace holder was disturbed although room existed: {s.report()}"
        )


async def test_control_one_engine_in_grace_ends_its_grace_for_a_routed_claimant(tmp_path, request):
    """One engine only, in grace after a lower-ranked client's turn with a follow-up served warm. A
    top-ranked claimant arrives and is handed to that engine's inbox by the routing: the holder gives
    up its grace within the window."""
    async with build_scene(tmp_path, request, grace_s=GRACE_S, cards=1) as s:
        await s.send("A1", LOW, "thread-a")
        assert await s.until(lambda: "grace_enter" in s.ev("A1"), SETUP_S), f"setup: no grace: {s.ev('A1')}"
        await serve_follow_up(s, "A2", LOW, "thread-a", "A1")
        assert sorted(s.engines()) == [TAG], f"setup: expected one engine, got {sorted(s.engines())}"
        await settle_grace_engine(s, s.engines()[TAG])
        started = time.monotonic()
        await submit_claimant(s, HIGH)
        where = where_is_the_claimant(s)
        assert where["parks"] or "active" in s.ev(CLAIMANT), (
            f"setup: the claimant was not handed to the engine: {where}"
        )
        seen = await s.until(lambda: outcome(s), GRACE_S - 0.5)
        elapsed = time.monotonic() - started
        assert seen, f"single engine: no outcome within {GRACE_S - 0.5:.1f}s: {s.report()}"
        assert elapsed < GRACE_S - 1.0, f"single engine: outcome only after {elapsed:.2f}s"


# --------------------------------------------------------------------------------------
# Controls for the load figure: a really running turn is load, idle engines share new sessions
# --------------------------------------------------------------------------------------

NEW_CLIENT = LOWEST
SHORT_GRACE_S = 1  # the grace window of the engines that are made idle on purpose


async def two_engines_one_turn_held(s, held_label):
    """Two engines of the model, each with one held turn (A1 first, B1 second), then every turn but
    ``held_label`` is released and its engine waited on until it is truly idle: no slot set on it,
    nothing in flight, empty inbox, not in its grace window. Returns the engine whose turn is held
    (still mid-turn) and the one that is truly idle."""
    await start_two_engines(s, LOW, LOW)
    for label in ("A1", "B1"):
        if label != held_label:
            s.release(label)
    mid = s.engine_on(s.port_of[held_label])
    other_label = "B1" if held_label == "A1" else "A1"
    idle = s.engine_on(s.port_of[other_label])
    assert mid is not idle, "setup: both turns ran on one engine"
    assert await s.until(
        lambda: idle.active_slot is None and not idle.in_grace_loop
        and idle.state is manager_module.ResidentState.IDLE_EVICTABLE,
        SETUP_S,
    ), (
        f"setup: engine {idle.resident_key} never became truly idle: state={idle.state.name}, "
        f"slot set={idle.active_slot is not None}, in grace={idle.in_grace_loop}"
    )
    assert not idle.inflight and idle.inbox.qsize() == 0, "setup: the idle engine has pending work"
    return mid, idle


@pytest.mark.parametrize("held_label", ["A1", "B1"], ids=["first-engine-mid-turn", "second-engine-mid-turn"])
async def test_control_a_running_turn_still_counts_as_load(tmp_path, request, held_label):
    """One engine is really in the middle of a turn (a held turn, not in its grace window), the other
    is truly idle. A new session of the same model goes to the idle engine and is served there at
    once, whichever of the two engines is the busy one: a turn that is really running still counts
    as load."""
    async with build_scene(tmp_path, request, grace_s=SHORT_GRACE_S, cards=2) as s:
        mid, idle = await two_engines_one_turn_held(s, held_label)
        assert s.is_held(held_label) and mid.active_slot is s.slots[held_label], (
            f"setup: engine {mid.resident_key} is not mid-turn on {held_label}"
        )
        assert not mid.in_grace_loop, "setup: the mid-turn engine is in its grace window"
        await s.send("N1", NEW_CLIENT, "thread-n1")
        assert await s.until(lambda: any(p[0] == "N1" for p in s.picks), SETUP_S), (
            f"the new session was never routed: {s.report()}"
        )
        picks = [p[1] for p in s.picks if p[0] == "N1"]
        assert picks == [idle.resident_key], (
            f"the new session was not routed to the idle engine {idle.resident_key} "
            f"(mid-turn engine: {mid.resident_key}): picks={picks}"
        )
        assert await s.until(lambda: "N1" in s.port_of, SETUP_S), f"the new session was not served: {s.report()}"
        assert s.port_of["N1"] == idle.port, "the new session ran on the mid-turn engine"
        assert s.is_held(held_label), "the running turn was disturbed"


FOLLOW_UP_GRACE_S = 3  # long enough to make one engine idle and still serve a follow-up in the other's window


@pytest.mark.parametrize("idle_label", ["A1", "B1"], ids=["first-engine-idle", "second-engine-idle"])
async def test_control_a_follow_up_inside_a_grace_window_still_counts_as_load(tmp_path, request, idle_label):
    """One engine is serving a same-thread follow-up inside its grace window (the follow-up's turn is
    held, so the engine is in its grace loop with a running follow-up slot, not in the state a finished
    turn leaves), the other is truly idle.
    A new session of the same model goes to the idle engine: a follow-up turn is a really running
    turn and counts as load, unlike the finished turn that opened the grace window."""
    async with build_scene(tmp_path, request, grace_s=FOLLOW_UP_GRACE_S, cards=2) as s:
        await start_two_engines(s, LOW, LOW)
        follow_label = "B1" if idle_label == "A1" else "A1"
        thread = "thread-b" if follow_label == "B1" else "thread-a"
        s.release(idle_label)
        idle = s.engine_on(s.port_of[idle_label])
        assert await s.until(
            lambda: idle.active_slot is None and not idle.in_grace_loop
            and idle.state is manager_module.ResidentState.IDLE_EVICTABLE,
            SETUP_S + FOLLOW_UP_GRACE_S,
        ), f"setup: engine {idle.resident_key} never became truly idle"
        assert not idle.inflight and idle.inbox.qsize() == 0, "setup: the idle engine has pending work"
        s.release(follow_label)
        assert await s.until(lambda: "grace_enter" in s.ev(follow_label), SETUP_S), (
            f"setup: the first turn of the other engine never entered grace: {s.ev(follow_label)}"
        )
        s.hold("F1")
        await s.send("F1", LOW, thread)
        assert await s.until(lambda: "F1" in s.port_of, SETUP_S), "setup: the follow-up was not served"
        serving = s.engine_on(s.port_of[follow_label])
        assert s.port_of["F1"] == s.port_of[follow_label] and serving is not idle, (
            "setup: the follow-up was not served warm on its own thread's engine"
        )
        assert s.is_held("F1") and serving.in_grace_loop and serving.active_slot is s.slots["F1"], (
            f"setup: engine {serving.resident_key} is not in its grace loop with the follow-up running"
        )
        assert s.slots["F1"].state is SlotState.ACTIVE, (
            f"setup: the running follow-up slot is in state {s.slots['F1'].state.name}, not ACTIVE "
            "(a slot only reads GRACE once its turn has finished)"
        )
        await s.send("N1", NEW_CLIENT, "thread-n1")
        assert await s.until(lambda: any(p[0] == "N1" for p in s.picks), SETUP_S), (
            f"the new session was never routed: {s.report()}"
        )
        picks = [p[1] for p in s.picks if p[0] == "N1"]
        assert picks == [idle.resident_key], (
            f"the new session was not routed to the idle engine {idle.resident_key} "
            f"(engine serving the follow-up: {serving.resident_key}): picks={picks}"
        )
        assert await s.until(lambda: "N1" in s.port_of, SETUP_S), f"the new session was not served: {s.report()}"
        assert s.port_of["N1"] == idle.port, "the new session ran on the engine serving the follow-up"
        assert s.is_held("F1"), "the running follow-up was disturbed"


async def test_control_two_idle_engines_share_new_sessions_one_each(tmp_path, request):
    """Both engines are truly idle (no slot set, nothing in flight, empty inbox, not in grace). Two new
    sessions of the same model, from two different clients and threads, arrive one after the other;
    the first one's turn is held while the second arrives. They end up one per engine, not both on
    one engine."""
    async with build_scene(tmp_path, request, grace_s=SHORT_GRACE_S, cards=2) as s:
        mid, idle = await two_engines_one_turn_held(s, "A1")
        s.release("A1")
        assert await s.until(
            lambda: mid.active_slot is None and not mid.in_grace_loop
            and mid.state is manager_module.ResidentState.IDLE_EVICTABLE,
            SETUP_S,
        ), f"setup: engine {mid.resident_key} never became truly idle"
        for r in (mid, idle):
            assert r.active_slot is None and not r.inflight and r.inbox.qsize() == 0 and not r.in_grace_loop, (
                f"setup: engine {r.resident_key} is not truly idle"
            )
        s.hold("N1")
        await s.send("N1", NEW_CLIENT, "thread-n1")
        assert await s.until(lambda: "N1" in s.port_of, SETUP_S), f"the first new session never started: {s.report()}"
        await s.send("N2", "5.5.5.5", "thread-n2")
        assert await s.until(lambda: any(p[0] == "N2" for p in s.picks), SETUP_S), (
            f"the second new session was never routed: {s.report()}"
        )
        picks = {label: [p[1] for p in s.picks if p[0] == label] for label in ("N1", "N2")}
        assert len(picks["N1"]) == 1 and len(picks["N2"]) == 1 and picks["N1"] != picks["N2"], (
            f"the two new sessions were routed to the same engine, not one per engine: {picks}"
        )
        assert await s.until(lambda: "N2" in s.port_of, SETUP_S), f"the second new session never started: {s.report()}"
        assert {s.port_of["N1"], s.port_of["N2"]} == {mid.port, idle.port}, (
            f"the two new sessions did not end up one per engine: ports={s.port_of['N1']}, {s.port_of['N2']}; "
            f"engine ports={mid.port}, {idle.port}; picks={picks}"
        )
