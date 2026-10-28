"""A top-ranked Fast Lane claimant against two engines of one model that are both in grace.

The rule under test is Rule G1 of the Fast Lane design document (grace rules): while a request that
strictly outranks a loaded client is waiting, that client is the designated victim and gives up its
grace window at its own turn boundary, or the waiting request is admitted next. A claimant must
never sit out the full grace window of a client it outranks.

The scene: one model, two engines of it running at once (the first engine's registry key is the model
tag, the second one's is ``<tag>#1``), each held by a lower-ranked client that has finished a turn and
is in its grace window, with a same-thread follow-up already served warm inside that window. Then a
claimant of the top rule arrives.

What the tests read. Every outcome is read from the audit events of the slot concerned, by their real
names: ``grace_enter``, ``grace_designated_victim_skip`` and ``grace_designated_unload_target_break``
on a holder's slot, and ``active`` on the claimant's own slot. Before the claimant arrives the main
test asserts that two live engines of the model really exist, so the situation it describes is the one
it runs. The controls cover the neighbouring behaviour that must hold with or without the rule: grace
is kept when nothing outranks the holder or when Fast Lane is off, a single engine behaves the
same way, and an admission can be observed at all.

Everything is one process: asyncio only, two engines at most, bounded waits.
"""

import asyncio
import contextlib
import time

import pytest
from _multiinstance_support import _manifest, _new_manager, _probe

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

GRACE_S = 5          # grace window of the main test
CONTROL_GRACE_S = 3  # grace window of the controls
SETUP_S = 5.0        # bound for setup steps that must happen promptly
EARLY_BREAK_EVENTS = ("grace_designated_victim_skip", "grace_designated_unload_target_break")


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
    """One manager over a fake engine whose turns end only when told to.

    ``send`` submits a request through the real ``submit``. The fake completion call records, for
    each request, the engine port its turn ran on, then blocks only for requests that were held
    with ``hold``. Requests that were not held complete at once.
    """

    def __init__(self, tmp_path, *, grace_s, fastlane_enabled=True):
        self.mgr, self.boot, _spawns = _new_manager(tmp_path, budget=2)
        self.mgr.runtime.fastlane = FastLaneConfig(enabled=fastlane_enabled, rules=RULES)
        self.mgr.runtime.queue.grace_seconds = grace_s
        self.mgr.runtime.queue.max_grace_extensions = 50
        self.mgr.runtime.queue.idle_hot_load_seconds = 600
        # One card per engine: this is what lets the model run on two engines at once.
        _manifest(self.boot, TAG, auto_place=True, llama_server_flags={"split_mode": "none"})
        self.slots = {}   # label -> Slot
        self.port_of = {}  # label -> engine port its turn ran on
        self._gates = {}
        self._open = False
        self.mgr._complete_fn = self._complete

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


@contextlib.asynccontextmanager
async def build_scene(tmp_path, *, grace_s, cards, fastlane_enabled=True):
    """A manager with the dispatcher running and ``cards`` cards reporting plenty of free memory.

    The card probe is replaced at both places the manager reads it, so no real host reading is
    involved. The manager's box budget is two engines; ``cards`` decides how many can be placed.
    """
    scene = Scene(tmp_path, grace_s=grace_s, fastlane_enabled=fastlane_enabled)
    probe_a, probe_b = _probe([20000] * cards)
    scene.mgr._worker_task = asyncio.create_task(scene.mgr.worker_loop())
    try:
        with probe_a, probe_b:
            yield scene
    finally:
        scene.open_everything()
        await scene.mgr.shutdown()


async def hold_two_engines_in_grace(s):
    """Two engines of the model, each in grace after a lower-ranked client's turn, each with a
    same-thread follow-up already served warm inside that window. Returns the two anchor labels.
    """
    s.hold("A1")
    s.hold("B1")
    await s.send("A1", LOW, "thread-a")
    assert await s.until(lambda: "A1" in s.port_of, SETUP_S), "setup: the first turn never started"
    await s.send("B1", LOW, "thread-b")
    assert await s.until(lambda: "B1" in s.port_of, SETUP_S), "setup: the second turn never started"
    assert sorted(s.engines()) == [TAG, SECOND_KEY], (
        f"setup: expected two engines of one model, got {sorted(s.engines())}"
    )
    assert s.port_of["A1"] != s.port_of["B1"], "setup: both turns ran on one engine"
    s.release("A1")
    s.release("B1")
    assert await s.until(
        lambda: "grace_enter" in s.ev("A1") and "grace_enter" in s.ev("B1"), SETUP_S,
    ), f"setup: the holders never entered grace: A1={s.ev('A1')} B1={s.ev('B1')}"
    await s.send("A2", LOW, "thread-a")
    await s.send("B2", LOW, "thread-b")
    assert await s.until(lambda: "A2" in s.port_of and "B2" in s.port_of, SETUP_S), (
        "setup: the follow-ups were not served inside the grace window"
    )
    assert s.port_of["A2"] == s.port_of["A1"] and s.port_of["B2"] == s.port_of["B1"], (
        "setup: a follow-up was not served warm on its own thread's engine"
    )
    return "A1", "B1"


def early_break_seen(s, holders):
    return any(name in s.ev(label) for label in holders for name in EARLY_BREAK_EVENTS)


# --------------------------------------------------------------------------------------
# The rule under test
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine_key", [TAG, SECOND_KEY], ids=["first-engine", "second-engine"])
async def test_rule0_claimant_ends_the_outranked_holders_grace_or_is_admitted_next(
    tmp_path, engine_key,
):
    """Two engines of one model, both held by lower-ranked clients in grace, follow-ups served
    warm. A claimant of the top rule arrives, its session pinned to ``engine_key`` (the first or
    the second engine, as session stickiness does in normal operation). Before the end of the grace
    window, either one of the holders it outranks must have given up its grace (it was
    designated, on its own slot) or the claimant itself must have been admitted (``active`` on
    its own slot)."""
    async with build_scene(tmp_path, grace_s=GRACE_S, cards=2) as s:
        holders = await hold_two_engines_in_grace(s)

        # The situation the rule is about: two live engines of the one model, at claim time.
        live = s.engines()
        assert sorted(live) == [TAG, SECOND_KEY], (
            f"expected two live engines of {TAG!r} when the claimant arrives, got {sorted(live)}"
        )
        assert len({r.port for r in live.values()}) == 2, "the two engines share one port"

        async with s.mgr._registry_lock:
            s.mgr._remember_affinity(TAG, "tid:thread-c", engine_key)
        started = time.monotonic()
        await s.send("C", HIGH, "thread-c")

        def outcome():
            return early_break_seen(s, holders) or "active" in s.ev("C")

        seen = await s.until(outcome, GRACE_S - 0.5)
        elapsed = time.monotonic() - started

        assert seen, (
            "the top-ranked claimant was not admitted and no holder it outranks gave up its grace "
            f"within {GRACE_S - 0.5:.1f}s, with two live engines {sorted(live)} and the claimant "
            f"pinned to {engine_key!r}: "
            f"holder A1 events={s.ev('A1')}, holder B1 events={s.ev('B1')}, "
            f"claimant events={s.ev('C')}"
        )
        assert elapsed < GRACE_S - 1.0, (
            f"the outcome came after {elapsed:.2f}s of a {GRACE_S}s grace window: grace was held "
            "toward its full deadline although a strictly higher-ranked claim was waiting"
        )


# --------------------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------------------

async def test_control_without_a_claim_the_holders_keep_their_grace(tmp_path):
    """Nothing outranks the two holders: neither gives up grace early. The test also requires that
    both really are in grace, so a silent probe cannot pass it."""
    async with build_scene(tmp_path, grace_s=CONTROL_GRACE_S, cards=2) as s:
        holders = await hold_two_engines_in_grace(s)
        broke = await s.until(lambda: early_break_seen(s, holders), CONTROL_GRACE_S - 1.0)
        assert not broke, (
            f"a holder gave up grace with no claim waiting: A1={s.ev('A1')} B1={s.ev('B1')}"
        )
        for label in holders:
            assert "grace_enter" in s.ev(label), f"{label} never showed grace_enter: {s.ev(label)}"
            assert s.slots[label].state is SlotState.GRACE, (
                f"{label} is not in grace: {s.slots[label].state.name}"
            )


async def test_control_a_claimant_that_does_not_outrank_the_holders_changes_nothing(tmp_path):
    """A claimant of the lowest rule waits behind two holders of a better rule: neither holder is
    asked to give up its grace for it."""
    async with build_scene(tmp_path, grace_s=CONTROL_GRACE_S, cards=2) as s:
        holders = await hold_two_engines_in_grace(s)
        await s.send("C", LOWEST, "thread-c")
        broke = await s.until(lambda: early_break_seen(s, holders), CONTROL_GRACE_S - 1.0)
        assert not broke, (
            f"a holder gave up grace for a claimant that does not outrank it: A1={s.ev('A1')} "
            f"B1={s.ev('B1')}"
        )
        assert "active" not in s.ev("C"), (
            f"the lower-ranked claimant was admitted while both holders were in grace: {s.ev('C')}"
        )


async def test_control_with_fast_lane_off_the_holders_keep_their_grace(tmp_path):
    """Fast Lane off, the same rules configured: a top-rule request does not shorten any grace."""
    async with build_scene(
        tmp_path, grace_s=CONTROL_GRACE_S, cards=2, fastlane_enabled=False,
    ) as s:
        holders = await hold_two_engines_in_grace(s)
        await s.send("C", HIGH, "thread-c")
        broke = await s.until(lambda: early_break_seen(s, holders), CONTROL_GRACE_S - 1.0)
        assert not broke, (
            f"a holder gave up grace with Fast Lane off: A1={s.ev('A1')} B1={s.ev('B1')}"
        )
        for label in holders:
            assert "grace_enter" in s.ev(label), f"{label} never showed grace_enter: {s.ev(label)}"


async def test_control_a_single_engine_gives_up_grace_and_admits_the_claimant(tmp_path):
    """One engine only (one card), one lower-ranked holder in grace with a follow-up served warm.
    A claimant of the top rule arrives: the holder gives up its grace and the claimant is admitted
    before the end of the window. Whatever the number of engines, this is what must hold."""
    async with build_scene(tmp_path, grace_s=GRACE_S, cards=1) as s:
        await s.send("A1", LOW, "thread-a")
        assert await s.until(lambda: "grace_enter" in s.ev("A1"), SETUP_S), (
            f"setup: the holder never entered grace: {s.ev('A1')}"
        )
        await s.send("A2", LOW, "thread-a")
        assert await s.until(lambda: "A2" in s.port_of, SETUP_S), "setup: follow-up not served"
        assert s.port_of["A2"] == s.port_of["A1"], "setup: follow-up was not served warm"
        assert sorted(s.engines()) == [TAG], f"setup: expected one engine, got {sorted(s.engines())}"

        started = time.monotonic()
        await s.send("C", HIGH, "thread-c")

        def done():
            return early_break_seen(s, ["A1"]) and "active" in s.ev("C")

        seen = await s.until(done, GRACE_S - 0.5)
        elapsed = time.monotonic() - started
        assert seen, (
            f"single engine: holder events={s.ev('A1')}, claimant events={s.ev('C')} "
            f"after {elapsed:.2f}s"
        )
        assert elapsed < GRACE_S - 1.0, f"single engine: outcome only after {elapsed:.2f}s"
        assert sorted(s.engines()) == [TAG], "a second engine appeared in the single-engine control"


async def test_control_an_admission_can_be_observed_on_the_claimants_own_slot(tmp_path):
    """The probe can see an admission: a lone claimant with nothing in its way shows ``active`` on
    its own slot, and no ``active`` is read before it is submitted."""
    async with build_scene(tmp_path, grace_s=CONTROL_GRACE_S, cards=2) as s:
        probe_slot = await s.send("C", HIGH, "thread-c")
        assert await s.until(lambda: "active" in s.ev("C"), SETUP_S), (
            f"a lone claimant was never admitted: {s.ev('C')}"
        )
        # Another slot's events are a different list: the claimant's id is not among them.
        assert probe_slot.slot_id not in s.ev("C")
        assert events(s.boot, "no-such-slot") == []
