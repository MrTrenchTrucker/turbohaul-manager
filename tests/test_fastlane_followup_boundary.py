"""A better-ranked claim parked in the holder's own inbox, at the holder's turn boundary.

The Fast Lane grace rules say that a holder whose own inbox contains a live claim that
strictly outranks it stops waiting for follow-ups: the claim needs no room and is admitted on the same
engine at the holder's turn boundary, after the holder's last turn is saved. The running turn still
finishes first. The code has two places where that rule ends a grace wait, and each has its own audit
event on the holder's slot:

  * the check made right after an anchor turn (the turn that opened the holder's slot), before any grace
    window has started: ``grace_designated_victim_skip``, and the slot never shows ``grace_enter``;
  * the check made at the top of every iteration of the grace loop, while the holder serves warm
    follow-ups: ``grace_designated_unload_target_break``, on a slot that did enter grace.

This file guards both. The first test measures the boundary when the holder is serving a warm follow-up
and a second same-thread follow-up is already waiting; the second test measures it when the holder's
anchor turn is the one that is running. Each has a control with a claimant that does not outrank the
holder, and the follow-up shape has a second control with no claim at all.

The scene: one model with two real engines (the first engine's registry key is the model tag, the second
one's is ``<tag>#N`` with N = 1), a real manager, a real dispatcher and a fake completion call whose turns
end only when the test says so. The other engine sits in its own grace window with nothing to do. A
claimant of the top rule is pinned to the busy engine and parked in that engine's own inbox, then the
running turn is released.

What must hold at the boundary, in this order of events:
  * the holder's slot ends its grace wait early (follow-up shape: one of the two early-end events;
    anchor shape: ``grace_designated_victim_skip`` and no ``grace_enter``), and
  * the claimant is admitted on the busy engine (``active`` on its own slot), and
  * between the end of the running turn and that admission, no follow-up of another request starts, takes
    the warm-follow-up promotion (ACTIVE_MATCH), or completes.

Stand-ins (each one is a controlled substitute, not a weakening):
  * The claimant is pinned to the busy engine with the session affinity record (what a spawn writes). An
    unpinned claimant would be routed by load to the other engine; the pin isolates the turn-boundary
    behaviour from engine choice. The park is proven, not assumed: a spy on the claim-parking call records
    the calling function (the multi-engine route's inner function) and the engine key, a spy on the load
    picker proves it was never asked, and the claim must be inside the busy engine's inbox and absent from
    the other engine's.
  * The route's inner function is a closure and cannot be replaced, so the claim-parking call it makes is
    the spied point.
  * Priority is by rule order: the claimant's rule is the first one, the holder's client the second.
  * In the anchor shape the second same-thread request arrives while the holder's turn still runs, so the
    dispatcher routes it, by the same session record, into the busy engine's own inbox (not into staging).

Controls: a claimant of the holder's own rule (not strictly higher) leaves the holder's grace window to run
in full, and in the follow-up shape it leaves the second follow-up served warm; no claim at all leaves the
follow-up served warm. Both show the tests can see a follow-up being served and a grace window being kept.

One process, asyncio only, two engines at most, bounded waits.
"""

import asyncio
import contextlib
import sys
import time
from unittest.mock import patch

from _multiinstance_support import _manifest, _new_manager, _probe

import turbohaul.manager as manager_module
from turbohaul.config import FastLaneConfig, FastLaneRule, FastLaneTagRanks
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db

TAG = "qwen-duo"
SECOND_KEY = f"{TAG}#{1}"  # registry key of the second engine: the tag, '#', the engine number

# Priority order: index 0 outranks index 1, which outranks index 2.
HIGH = "1.1.1.1"
LOW = "2.2.2.2"
LOWEST = "3.3.3.3"
RULES = [
    FastLaneRule(address=HIGH, label="high", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address=LOW, label="low", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address=LOWEST, label="lowest", tag_ranks=FastLaneTagRanks(main=1)),
]

GRACE_S = 5          # grace window
CONTROL_GRACE_S = 3  # grace window of the anchor-shape control
KEPT_GRACE_S = 1.5   # how long the control watches the kept grace window (less than CONTROL_GRACE_S)
SETUP_S = 4.5        # bound for setup steps that must happen promptly
OUTCOME_S = 4.5      # bound for the outcome after the running turn is released
EARLY_END_EVENTS = ("grace_designated_victim_skip", "grace_designated_unload_target_break")
FOLLOW_UP_DONE_EVENTS = ("active_match_completed", "active_match_failed_terminal")


def audit_events(boot, slot_id):
    """The audit event types of one slot as stored, oldest first."""
    conn = open_state_db(boot.storage.state_db_path)
    try:
        cur = conn.execute(
            "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id", (slot_id,),
        )
        return [row["event_type"] for row in cur.fetchall()]
    finally:
        conn.close()


class Scene:
    """One manager over a fake engine whose turns end only when told to.

    ``send`` submits through the real ``submit``; the prompt is the request's label. The fake completion
    call records the engine port of each turn and blocks only for held requests. Spies record, in one
    ordered list (``rows``): every audit event written through the manager, every slot state transition
    made by the manager module, the start and end of every fake turn, and the test's own release marks.
    Two more spies record every claim park (slot id, engine key, calling function) and every call of the
    load picker.
    """

    def __init__(self, tmp_path, *, grace_s=GRACE_S):
        self.mgr, self.boot, _spawns = _new_manager(tmp_path, budget=2)
        self.mgr.runtime.fastlane = FastLaneConfig(enabled=True, rules=RULES)
        self.mgr.runtime.queue.grace_seconds = grace_s
        self.mgr.runtime.queue.max_grace_extensions = 50
        self.mgr.runtime.queue.idle_hot_load_seconds = 600
        # One card per engine: this is what lets the model run on two engines at once.
        _manifest(self.boot, TAG, auto_place=True, llama_server_flags={"split_mode": "none"})
        self.slots = {}     # label -> Slot
        self.port_of = {}   # label -> engine port of its turn
        self.rows = []      # (seconds, kind, label, what), in the order things happened
        self.parks = []     # (slot id, engine key, calling function)
        self.picks = []     # labels of slots the load picker was asked about
        self._gates = {}
        self._open = False
        self._t0 = time.monotonic()
        self.mgr._complete_fn = self._complete
        self._install_spies()

    # -- spies ------------------------------------------------------------
    def mark(self, kind, label, what):
        self.rows.append((round(time.monotonic() - self._t0, 3), kind, label, what))

    def _install_spies(self):
        mgr = self.mgr
        real_park = mgr._park_fastlane_claim_locked
        real_audit = mgr._audit_async
        real_pick = mgr._pick_instance_for

        def spy_park(slot, r):
            self.parks.append((slot.slot_id, r.resident_key, sys._getframe(1).f_code.co_name))
            return real_park(slot, r)

        async def spy_audit(slot, event_type):
            self.mark("audit", getattr(slot, "prompt", None), event_type)
            return await real_audit(slot, event_type)

        def spy_pick(slot, model_tag, instances):
            self.picks.append(getattr(slot, "prompt", None))
            return real_pick(slot, model_tag, instances)

        mgr._park_fastlane_claim_locked = spy_park
        mgr._audit_async = spy_audit
        mgr._pick_instance_for = spy_pick

    def transition_spy(self):
        """A patch for the manager module's slot transition that records the target state."""
        real = manager_module.transition

        def spy(slot, new_state, *a, **k):
            result = real(slot, new_state, *a, **k)
            self.mark("state", getattr(slot, "prompt", None), new_state.name)
            return result

        return patch.object(manager_module, "transition", spy)

    # -- fake completion --------------------------------------------------
    async def _complete(self, slot, handle):
        label = slot.prompt
        self.port_of[label] = handle.port
        self.mark("turn_start", label, handle.port)
        gate = self._gates.get(label)
        if gate is not None and not self._open:
            await gate.wait()
        self.mark("turn_end", label, handle.port)
        return {"ok": True, "model": handle.model_tag}

    def hold(self, label):
        self._gates[label] = asyncio.Event()

    def release(self, label):
        self.mark("release", label, "")
        self._gates[label].set()

    def is_held(self, label):
        gate = self._gates.get(label)
        return gate is not None and not gate.is_set()

    def open_everything(self):
        self._open = True
        for gate in self._gates.values():
            gate.set()

    # -- requests and reads -----------------------------------------------
    async def send(self, label, ip, thread):
        slot = await self.mgr.submit(
            TAG, prompt=label, thread_id=thread, client_meta={"ip": ip, "is_main": True},
            wait_for_completion=True,
        )
        self.slots[label] = slot
        return slot

    def ev(self, label):
        return audit_events(self.boot, self.slots[label].slot_id)

    async def until(self, predicate, timeout):
        """True when ``predicate`` holds within ``timeout`` seconds (bounded poll)."""
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.02)

    def engines(self):
        return {r.resident_key: r for r in self.mgr._instances_for(TAG)}

    def engine_on(self, port):
        return next(r for r in self.engines().values() if r.port == port)

    def claim_of(self, label):
        slot = self.slots[label]
        return next((c for c in self.mgr._fastlane_claims.values() if c["slot"] is slot), None)

    def seen(self, kind, label, what):
        return any(r[1:] == (kind, label, what) for r in self.rows)

    def timeline_text(self):
        return "\n".join(f"  {t:7.3f}s {kind:10s} {label!s:3s} {what}" for t, kind, label, what in self.rows)


@contextlib.asynccontextmanager
async def build_scene(tmp_path, *, grace_s=GRACE_S):
    """A manager with the dispatcher running and two cards reporting plenty of free memory.

    The card probe is replaced at both places the manager reads it, so no real host reading is involved."""
    scene = Scene(tmp_path, grace_s=grace_s)
    probe_a, probe_b = _probe([20000, 20000])
    scene.mgr._worker_task = asyncio.create_task(scene.mgr.worker_loop())
    try:
        with probe_a, probe_b, scene.transition_spy():
            yield scene
    finally:
        scene.open_everything()
        await scene.mgr.shutdown()


# --------------------------------------------------------------------------------------
# Scene builders (every builder asserts its own setup)
# --------------------------------------------------------------------------------------

async def two_engines_with_a_follow_up_running(s, holder_ip):
    """Engine A (first engine, key = tag): the holder's turn ended, its grace window runs, and a held
    same-thread follow-up F1 is running warm on it. Engine B (second engine): its turn ended and it sits in
    its grace window, idle. Returns (engine_a, engine_b)."""
    s.hold("A1")
    s.hold("B1")
    for label, ip, thread in (("A1", holder_ip, "thread-a"), ("B1", LOW, "thread-b")):
        await s.send(label, ip, thread)
        assert await s.until(lambda label=label: label in s.port_of, SETUP_S), (
            f"setup: the turn {label} never started"
        )
    assert sorted(s.engines()) == [TAG, SECOND_KEY], (
        f"setup: expected two live engines of one model, got {sorted(s.engines())}"
    )
    assert s.port_of["A1"] != s.port_of["B1"], "setup: both turns ran on one engine"
    s.release("B1")
    assert await s.until(lambda: "grace_enter" in s.ev("B1"), SETUP_S), (
        f"setup: the second engine never entered grace: {s.ev('B1')}"
    )
    s.release("A1")
    assert await s.until(lambda: "grace_enter" in s.ev("A1"), SETUP_S), (
        f"setup: the first engine's holder never entered grace: {s.ev('A1')}"
    )
    s.hold("F1")
    await s.send("F1", holder_ip, "thread-a")
    assert await s.until(lambda: "F1" in s.port_of, SETUP_S), (
        "setup: the follow-up F1 was not served inside the grace window"
    )
    engine_a, engine_b = s.engine_on(s.port_of["A1"]), s.engine_on(s.port_of["B1"])
    assert engine_a is not engine_b and engine_a.resident_key == TAG and engine_b.resident_key == SECOND_KEY
    assert s.port_of["F1"] == s.port_of["A1"], "setup: F1 was not served warm on the holder's own engine"
    assert engine_a.in_grace_loop and s.slots["A1"].state is SlotState.GRACE, (
        f"setup: the holder is not inside its grace loop: {s.slots['A1'].state.name}"
    )
    assert engine_a.active_slot is s.slots["F1"] and s.slots["F1"].state is SlotState.ACTIVE and s.is_held("F1"), (
        f"setup: the running follow-up is not the busy engine's active slot: {s.slots['F1'].state.name}"
    )
    assert s.seen("state", "F1", "ACTIVE_MATCH"), "setup: F1 was not promoted as a warm follow-up"
    assert engine_b.in_grace_loop and engine_b.inbox.qsize() == 0 and engine_a.inbox.qsize() == 0, (
        "setup: the idle engine is not in its grace window or an inbox is not empty"
    )
    return engine_a, engine_b


async def two_engines_with_an_anchor_turn_running(s, holder_ip):
    """Engine A (first engine, key = tag): the holder's ANCHOR turn A1, the turn that opened its slot, is
    running and held; the holder has not entered any grace window. Engine B (second engine): its turn ended
    and it sits in its grace window, idle. Returns (engine_a, engine_b)."""
    s.hold("A1")
    s.hold("B1")
    for label, ip, thread in (("A1", holder_ip, "thread-a"), ("B1", LOW, "thread-b")):
        await s.send(label, ip, thread)
        assert await s.until(lambda label=label: label in s.port_of, SETUP_S), (
            f"setup: the turn {label} never started"
        )
    assert sorted(s.engines()) == [TAG, SECOND_KEY], (
        f"setup: expected two live engines of one model, got {sorted(s.engines())}"
    )
    assert s.port_of["A1"] != s.port_of["B1"], "setup: both turns ran on one engine"
    s.release("B1")
    assert await s.until(lambda: "grace_enter" in s.ev("B1"), SETUP_S), (
        f"setup: the second engine never entered grace: {s.ev('B1')}"
    )
    engine_a, engine_b = s.engine_on(s.port_of["A1"]), s.engine_on(s.port_of["B1"])
    assert engine_a is not engine_b and engine_a.resident_key == TAG and engine_b.resident_key == SECOND_KEY
    assert engine_a.active_slot is s.slots["A1"] and s.slots["A1"].state is SlotState.ACTIVE and s.is_held("A1"), (
        f"setup: the holder's anchor turn is not the busy engine's running turn: {s.slots['A1'].state.name}"
    )
    assert not engine_a.in_grace_loop and "grace_enter" not in s.ev("A1"), (
        f"setup: the holder is already in a grace window: {s.ev('A1')}"
    )
    assert engine_b.in_grace_loop and engine_b.inbox.qsize() == 0 and engine_a.inbox.qsize() == 0, (
        "setup: the idle engine is not in its grace window or an inbox is not empty"
    )
    return engine_a, engine_b


async def park_claimant_on(s, engine, other, ip):
    """Pin the claimant's session to ``engine`` (as a spawn records it), submit it, and assert with the
    spies that it was parked in that engine's own inbox, by the multi-engine route, and not in ``other``'s."""
    async with s.mgr._registry_lock:
        s.mgr._remember_affinity(TAG, "tid:thread-c", engine.resident_key)
    await s.send("C", ip, "thread-c")
    assert await s.until(lambda: engine.inbox.qsize() == 1, SETUP_S), (
        f"setup: the claimant never reached the inbox of {engine.resident_key}"
    )
    claimant = s.slots["C"]
    assert other.inbox.qsize() == 0 and not s.mgr._inbox_holds_slot(other, claimant), (
        f"setup: the claimant is in the other engine's inbox ({other.resident_key})"
    )
    assert s.mgr._inbox_holds_slot(engine, claimant), "setup: the claimant is not in the busy engine's inbox"
    claim = s.claim_of("C")
    assert claim is not None and claim.get("parked_on") == engine.resident_key, (
        f"setup: the claimant's claim is not parked on {engine.resident_key}: {claim}"
    )
    assert [p[1:] for p in s.parks if p[0] == claimant.slot_id] == [(engine.resident_key, "_route_to")], (
        f"setup: the claimant was not parked by the multi-engine route on the busy engine: {s.parks}"
    )
    assert "C" not in s.picks, "setup: the load picker was asked about the claimant (the pin was not used)"
    assert s.mgr._governing_claim_locked(include_parked=False) is None, (
        "setup: an unparked governing claim exists besides the parked one"
    )


async def stage_second_follow_up(s, ip):
    """Submit F2, a second same-thread follow-up, while F1 is still running. It must wait in the staging
    queue (the dispatcher hides it while the thread's grace window runs), not in any inbox."""
    await s.send("F2", ip, "thread-a")
    await asyncio.sleep(0.3)
    follow_up = s.slots["F2"]
    assert "F2" not in s.port_of, "setup: F2 started before F1 ended"
    assert any(x is follow_up for x in s.mgr.queue._staging), "setup: F2 is not waiting in the staging queue"
    assert not any(s.mgr._inbox_holds_slot(r, follow_up) for r in s.engines().values()), (
        "setup: F2 is inside an engine's inbox"
    )
    assert s.is_held("F1"), "setup: F1 ended on its own"


async def queue_second_request_behind_the_claim(s, engine, other, ip):
    """Submit F2, a second same-thread request, while the holder's anchor turn still runs and the claim is
    already parked. The thread's session record routes it into the busy engine's own inbox; it must not
    have started, and the claim must still be parked there."""
    await s.send("F2", ip, "thread-a")
    follow_up = s.slots["F2"]
    assert await s.until(lambda: s.mgr._inbox_holds_slot(engine, follow_up), SETUP_S), (
        f"setup: F2 never reached the busy engine's inbox: state {follow_up.state.name}"
    )
    assert not s.mgr._inbox_holds_slot(other, follow_up), "setup: F2 is inside the other engine's inbox"
    assert "F2" not in s.port_of, "setup: F2 started before the holder's turn ended"
    assert s.mgr._inbox_holds_slot(engine, s.slots["C"]), "setup: the claimant left the busy engine's inbox"
    assert s.is_held("A1"), "setup: the holder's anchor turn ended on its own"


def boundary_window(s, turn_label="F1"):
    """Positions in ``rows`` of the end of the running turn and of the claimant's admission (None for the
    admission when the claimant was not admitted)."""
    rows = s.rows
    start = next((i for i, r in enumerate(rows) if r[1:3] == ("turn_end", turn_label)), None)
    admitted = next((i for i, r in enumerate(rows) if r[1:] == ("audit", "C", "active")), None)
    return start, admitted


def follow_up_rows(rows, labels=("F2",)):
    """Rows that show a request of ``labels`` being served or completed: the warm-follow-up promotion, a
    turn, a completion."""
    served = []
    for row in rows:
        _t, kind, label, what = row
        if label not in labels:
            continue
        if (kind == "state" and what == "ACTIVE_MATCH") or kind in ("turn_start", "turn_end") or (
            kind == "audit" and what in FOLLOW_UP_DONE_EVENTS
        ):
            served.append(row)
    return served


# --------------------------------------------------------------------------------------
# Follow-up shape: the holder is serving a warm follow-up when the claim is parked
# --------------------------------------------------------------------------------------

async def test_claim_parked_in_the_holders_inbox_is_admitted_at_the_turn_boundary_before_a_second_follow_up(
    tmp_path,
):
    """The holder's engine runs a warm follow-up; a claim that strictly outranks the holder is parked in that
    engine's own inbox; a second same-thread follow-up waits. When the running turn ends: the holder gives up
    its grace wait, the claimant is admitted on that engine, and the second follow-up is not started,
    promoted or completed between the turn end and the admission. It fails if the holder keeps waiting for
    follow-ups, if the claimant is admitted late or elsewhere, or if the second follow-up is served first."""
    async with build_scene(tmp_path) as s:
        engine_a, engine_b = await two_engines_with_a_follow_up_running(s, LOW)
        await park_claimant_on(s, engine_a, engine_b, HIGH)
        claim_key = s.mgr.queue._fastlane_priority_key(s.slots["C"])
        holder_key = s.mgr._resident_priority_key(engine_a, s.mgr._fastlane_table())
        assert claim_key is not None and holder_key is not None and s.mgr.queue._fastlane_strictly_higher(
            claim_key, holder_key,
        ), f"setup: the claim does not strictly outrank the holder: claim {claim_key}, holder {holder_key}"
        async with s.mgr._registry_lock:
            assert s.mgr._parked_claim_outranks_holder_locked(engine_a), (
                "setup: the manager does not see the parked claim as outranking the holder"
            )
        await stage_second_follow_up(s, LOW)
        assert not [e for e in s.ev("A1") if e in EARLY_END_EVENTS], (
            "setup: the holder gave up its grace before its running follow-up ended"
        )
        assert "active" not in s.ev("C"), "setup: the claimant was admitted before the running turn ended"

        s.release("F1")
        admitted = await s.until(lambda: boundary_window(s)[1] is not None, OUTCOME_S)
        await s.until(lambda: "C" in s.port_of, OUTCOME_S)  # the admitted turn records its port a moment later
        start, end = boundary_window(s)
        rows_in_window = s.rows[start + 1:end] if (start is not None and end is not None) else s.rows
        context = (
            f"\nevents on the holder's slot A1: {s.ev('A1')}\nevents on the claimant C: {s.ev('C')}"
            f"\nevents on the second follow-up F2: {s.ev('F2')}\ntimeline:\n{s.timeline_text()}"
        )
        # (ii) the claimant is admitted, on the busy engine, at the boundary.
        assert admitted, f"the claimant was not admitted within {OUTCOME_S}s of the running turn's end{context}"
        assert start is not None, f"the running follow-up's turn end was not recorded{context}"
        assert s.port_of.get("C") == s.port_of["A1"], f"the claimant was admitted on another engine{context}"
        # (i) the holder gave up its grace wait, on its own slot, after the turn end and before the admission.
        early = [e for e in s.ev("A1") if e in EARLY_END_EVENTS]
        assert early, f"the holder's slot shows no early end of its grace wait{context}"
        early_at = next(
            i for i, r in enumerate(s.rows) if r[1] == "audit" and r[2] == "A1" and r[3] in EARLY_END_EVENTS
        )
        assert start < early_at < end, f"the holder's early end is not between the turn end and the admission{context}"
        # (iii) no second follow-up was started, promoted or completed in between.
        served = follow_up_rows(rows_in_window)
        assert not served, (
            f"a matched follow-up was served ahead of the claimant between the turn end and its admission: "
            f"{served}{context}"
        )
        # The second follow-up was not served before the claimant at all.
        before = follow_up_rows(s.rows[:end])
        assert not before, f"the second follow-up was served before the claimant was admitted: {before}{context}"
        print("\nEARLY END EVENT ON THE HOLDER:", early)
        print("EVENT ORDER (turn end of F1 to the claimant's admission):")
        print("\n".join(f"  {r}" for r in s.rows[start:end + 1]))


async def test_control_a_claim_of_the_holders_own_rule_leaves_the_follow_up_served_warm(tmp_path):
    """Same scene, but the claimant belongs to the holder's own rule: it does not strictly outrank the
    holder. The second follow-up is served warm on the holder's engine at the boundary, ahead of the
    claimant, and the holder does not end its grace wait early. It fails if an equal-rank claim ends the
    holder's grace wait, or if the follow-up is not served warm."""
    async with build_scene(tmp_path) as s:
        engine_a, engine_b = await two_engines_with_a_follow_up_running(s, LOW)
        await park_claimant_on(s, engine_a, engine_b, LOW)
        claim_key = s.mgr.queue._fastlane_priority_key(s.slots["C"])
        holder_key = s.mgr._resident_priority_key(engine_a, s.mgr._fastlane_table())
        assert claim_key is not None and holder_key == claim_key and not s.mgr.queue._fastlane_strictly_higher(
            claim_key, holder_key,
        ), f"setup: the claim and the holder do not share a priority: claim {claim_key}, holder {holder_key}"
        async with s.mgr._registry_lock:
            assert not s.mgr._parked_claim_outranks_holder_locked(engine_a), (
                "setup: the manager sees an equal-rank claim as outranking the holder"
            )
        await stage_second_follow_up(s, LOW)
        s.release("F1")
        served = await s.until(
            lambda: s.seen("state", "F2", "ACTIVE_MATCH") and s.seen("turn_end", "F2", s.port_of.get("F2")),
            OUTCOME_S,
        )
        context = f"\nholder A1: {s.ev('A1')}\nF2: {s.ev('F2')}\ntimeline:\n{s.timeline_text()}"
        assert served, f"the second follow-up was not served warm at the boundary{context}"
        assert s.port_of["F2"] == s.port_of["A1"], f"the second follow-up ran on another engine{context}"
        assert not [e for e in s.ev("A1") if e in EARLY_END_EVENTS], (
            f"the holder ended its grace wait for a claim that does not outrank it{context}"
        )
        assert "active" not in s.ev("C") or s.rows.index(next(
            r for r in s.rows if r[1:] == ("turn_end", "F2", s.port_of["F2"]))) < s.rows.index(next(
            r for r in s.rows if r[1:] == ("audit", "C", "active"))), (
            f"the equal-rank claimant was admitted before the second follow-up finished{context}"
        )


async def test_control_without_a_claim_the_second_follow_up_is_served_warm(tmp_path):
    """Same scene with no claimant at all: the second follow-up is served warm on the holder's engine
    right after the running one ends, and nothing ends the holder's grace wait early. It fails if the
    follow-up is not served warm or if the holder ends its grace wait with nothing waiting."""
    async with build_scene(tmp_path) as s:
        engine_a, _engine_b = await two_engines_with_a_follow_up_running(s, LOW)
        assert not s.mgr._fastlane_claims, f"setup: a claim exists: {list(s.mgr._fastlane_claims)}"
        assert engine_a.inbox.qsize() == 0
        await stage_second_follow_up(s, LOW)
        s.release("F1")
        served = await s.until(
            lambda: s.seen("state", "F2", "ACTIVE_MATCH") and s.seen("turn_end", "F2", s.port_of.get("F2")),
            OUTCOME_S,
        )
        context = f"\nholder A1: {s.ev('A1')}\nF2: {s.ev('F2')}\ntimeline:\n{s.timeline_text()}"
        assert served, f"the second follow-up was not served warm{context}"
        assert s.port_of["F2"] == s.port_of["A1"], f"the second follow-up ran on another engine{context}"
        assert not [e for e in s.ev("A1") if e in EARLY_END_EVENTS], f"the holder ended its grace wait{context}"


# --------------------------------------------------------------------------------------
# Anchor shape: the holder's own anchor turn is running when the claim is parked
# --------------------------------------------------------------------------------------

async def test_claim_parked_during_the_holders_anchor_turn_is_admitted_at_the_turn_boundary(tmp_path):
    """The holder's anchor turn is running on its engine; a claim that strictly outranks the holder is parked
    in that engine's own inbox, then a second same-thread request is queued behind it. When the anchor turn
    ends: the holder's slot shows ``grace_designated_victim_skip`` between the turn end and the admission
    and never enters grace, the claimant is admitted on that engine, and no other request is started,
    promoted or completed before the claimant is admitted. It fails if the holder opens a grace window, if the
    claimant is admitted late or elsewhere, or if another request is served first."""
    async with build_scene(tmp_path) as s:
        engine_a, engine_b = await two_engines_with_an_anchor_turn_running(s, LOW)
        await park_claimant_on(s, engine_a, engine_b, HIGH)
        claim_key = s.mgr.queue._fastlane_priority_key(s.slots["C"])
        holder_key = s.mgr._resident_priority_key(engine_a, s.mgr._fastlane_table())
        assert claim_key is not None and holder_key is not None and s.mgr.queue._fastlane_strictly_higher(
            claim_key, holder_key,
        ), f"setup: the claim does not strictly outrank the holder: claim {claim_key}, holder {holder_key}"
        async with s.mgr._registry_lock:
            assert s.mgr._parked_claim_outranks_holder_locked(engine_a), (
                "setup: the manager does not see the parked claim as outranking the holder"
            )
        await queue_second_request_behind_the_claim(s, engine_a, engine_b, LOW)
        assert not [e for e in s.ev("A1") if e in EARLY_END_EVENTS], (
            "setup: the holder gave up its grace before its anchor turn ended"
        )
        assert "active" not in s.ev("C"), "setup: the claimant was admitted before the anchor turn ended"

        s.release("A1")
        admitted = await s.until(lambda: boundary_window(s, "A1")[1] is not None, OUTCOME_S)
        await s.until(lambda: "C" in s.port_of, OUTCOME_S)  # the admitted turn records its port a moment later
        start, end = boundary_window(s, "A1")
        rows_in_window = s.rows[start + 1:end] if (start is not None and end is not None) else s.rows
        context = (
            f"\nevents on the holder's slot A1: {s.ev('A1')}\nevents on the claimant C: {s.ev('C')}"
            f"\nevents on the second request F2: {s.ev('F2')}\ntimeline:\n{s.timeline_text()}"
        )
        # The claimant is admitted, on the busy engine, at the boundary.
        assert admitted, f"the claimant was not admitted within {OUTCOME_S}s of the anchor turn's end{context}"
        assert start is not None, f"the anchor turn's end was not recorded{context}"
        assert s.port_of.get("C") == s.port_of["A1"], f"the claimant was admitted on another engine{context}"
        # The holder's slot skipped grace: the entry-time event, after the turn end and before the admission.
        assert "grace_designated_victim_skip" in s.ev("A1"), (
            f"the holder's slot shows no grace_designated_victim_skip{context}"
        )
        skip_at = next(
            i for i, r in enumerate(s.rows) if r[1:] == ("audit", "A1", "grace_designated_victim_skip")
        )
        assert start < skip_at < end, (
            f"the holder's grace_designated_victim_skip is not between the turn end and the admission{context}"
        )
        assert "grace_enter" not in s.ev("A1") and not s.seen("state", "A1", "GRACE"), (
            f"the holder opened a grace window although a better claim was parked in its inbox{context}"
        )
        assert "grace_designated_unload_target_break" not in s.ev("A1"), (
            f"the holder broke out of a grace loop it should never have entered{context}"
        )
        # No other request was started, promoted or completed in between, or at any time before the admission.
        others = [r[2] for r in s.rows if r[1] == "turn_start" and r[2] not in ("A1", "B1", "C")]
        served = follow_up_rows(rows_in_window, labels=("F2", "B1"))
        assert not served, (
            f"a request was served ahead of the claimant between the turn end and its admission: "
            f"{served}{context}"
        )
        before = follow_up_rows(s.rows[:end], labels=("F2",))
        assert not before and not others, (
            f"the second request was served before the claimant was admitted: {before} {others}{context}"
        )
        print("\nEARLY END EVENT ON THE HOLDER:", [e for e in s.ev("A1") if e in EARLY_END_EVENTS])
        print("EVENT ORDER (turn end of A1 to the claimant's admission):")
        print("\n".join(f"  {r}" for r in s.rows[start:end + 1]))


async def test_control_an_anchor_claim_of_the_holders_own_rule_leaves_the_grace_window_to_run(tmp_path):
    """Same anchor scene, but the claimant belongs to the holder's own rule: it does not strictly outrank the
    holder. When the anchor turn ends the holder enters its grace window (``grace_enter``) and shows no
    early-end event; the claimant is not admitted while that window runs, and is admitted on the same engine
    once it has lapsed. It fails if an equal-rank claim ends the holder's grace wait or is admitted early."""
    async with build_scene(tmp_path, grace_s=CONTROL_GRACE_S) as s:
        engine_a, engine_b = await two_engines_with_an_anchor_turn_running(s, LOW)
        await park_claimant_on(s, engine_a, engine_b, LOW)
        claim_key = s.mgr.queue._fastlane_priority_key(s.slots["C"])
        holder_key = s.mgr._resident_priority_key(engine_a, s.mgr._fastlane_table())
        assert claim_key is not None and holder_key == claim_key and not s.mgr.queue._fastlane_strictly_higher(
            claim_key, holder_key,
        ), f"setup: the claim and the holder do not share a priority: claim {claim_key}, holder {holder_key}"
        async with s.mgr._registry_lock:
            assert not s.mgr._parked_claim_outranks_holder_locked(engine_a), (
                "setup: the manager sees an equal-rank claim as outranking the holder"
            )

        s.release("A1")
        entered = await s.until(lambda: "grace_enter" in s.ev("A1"), SETUP_S)
        admitted_early = await s.until(lambda: "active" in s.ev("C"), KEPT_GRACE_S)
        context = f"\nholder A1: {s.ev('A1')}\nclaimant C: {s.ev('C')}\ntimeline:\n{s.timeline_text()}"
        assert entered, f"the holder did not enter its grace window at the end of its anchor turn{context}"
        assert not admitted_early, (
            f"the equal-rank claimant was admitted while the holder's grace window was running{context}"
        )
        assert not [e for e in s.ev("A1") if e in EARLY_END_EVENTS], (
            f"the holder ended its grace wait for a claim that does not outrank it{context}"
        )
        # Once the window has lapsed the claimant is admitted on the same engine.
        admitted = await s.until(lambda: "active" in s.ev("C"), OUTCOME_S)
        assert admitted, f"the claimant was not admitted after the holder's grace window lapsed{context}"
        await s.until(lambda: "C" in s.port_of, OUTCOME_S)  # the admitted turn records its port a moment later
        assert s.port_of.get("C") == s.port_of["A1"], f"the claimant was admitted on another engine{context}"
        assert not [e for e in s.ev("A1") if e in EARLY_END_EVENTS], (
            f"the holder shows an early end of its grace wait{context}"
        )
