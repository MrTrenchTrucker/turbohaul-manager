"""A resident that is torn down for a better client is torn down at the END of its follow-up turn.

Shape. Two Fast Lane clients are configured; client 1 is listed first, so it is the better client.
A resident model (client 2) completes a turn and passes through its grace window. A follow-up from
the same client then arrives and is served warm on the same engine. That follow-up turn is held
mid-generation by the fake engine. A request from client 1 for a different model then arrives and
the card cannot hold both models, so the resident is the model that has to go.

Required behaviour (a turn that is generating is never cut, whichever client asked for the room):
  1. while the follow-up turn is held, the resident is not unloaded and its engine is not stopped;
  2. when the turn ends, its response reaches the client intact, THEN the resident is unloaded,
     with its conversation state saved before the engine is stopped, THEN the engine is stopped,
     THEN the better client's model is loaded and its turn runs;
  3. the unload begins at the turn boundary: within a tight bound after the turn ends, from the
     state the resident was in while serving (never parked idle, never left to finish a grace
     window first). The grace window is 10 s and the retry timers are 30 s in the held test, so a
     teardown that waits for either shows up as a late unload;
  4. the better client is advanced by a wake, not by a retry timer.

Everything is asserted by the ORDER of stamped events, not by sleeps. The second test is the
control: the same clients, but the follow-up turn finishes quickly. It shows the hand-over through
the wake works when nothing is held, so the first test fails only because of the held turn.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import pytest
import yaml

from _fastlane_fixture import boot_ranked_runtime, make_fakes, resident_for
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState, TurbohaulManager

IP_1 = "192.0.2.10"
IP_2 = "198.51.100.20"
FOOTPRINT_MIB = 20000
CARD_MIB = 24000
LOADED = (ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE)
TAG_A, TAG_B = "model-a", "model-b"
RANKS = FastLaneTagRanks(main=1, curator=3, unclassified=5)
CLIENT_1_FIRST = [FastLaneRule(address=IP_1, tag_ranks=RANKS),
                  FastLaneRule(address=IP_2, tag_ranks=RANKS)]
CURATOR_2 = {"ip": IP_2, "is_curator": True, "is_sub_agent": True}
MAIN_1 = {"ip": IP_1, "is_main": True}
OBSERVE_S = 3.0          # many polls of the 0.05 s grace and dispatch loops, with margin
RETRY_TIMER_S = 30.0     # the retry timers are pushed far out so only a wake can advance the claimant
WAKE_BOUND_S = 5.0       # the hand-over must land well below RETRY_TIMER_S
LONG_GRACE_S = 10        # grace window for the follow-up turn: far longer than the boundary bound
TURN_BOUNDARY_S = 1.0    # the unload must begin this soon after the follow-up turn ends


@dataclass
class Box:
    mgr: TurbohaulManager
    holds: dict
    t0: float
    events: list = field(default_factory=list)
    tasks: dict = field(default_factory=dict)
    kv_calls: list = field(default_factory=list)
    unload_states: list = field(default_factory=list)  # (event index, resident state name) per unload entry

    def stamp(self, kind, name):
        self.events.append((time.monotonic(), kind, name))

    def seen(self, kind, name):
        return any((k, n) == (kind, name) for _t, k, n in self.events)

    def idx(self, kind, name, nth=0):
        """Position of the nth stamped (kind, name) in the event list, or None."""
        hits = [i for i, (_t, k, n) in enumerate(self.events) if (k, n) == (kind, name)]
        return hits[nth] if len(hits) > nth else None

    def count(self, kind, name):
        return sum(1 for _t, k, n in self.events if (k, n) == (kind, name))

    def when(self, kind, name):
        i = self.idx(kind, name)
        return None if i is None else self.events[i][0]

    def rel(self):
        return [(round(t - self.t0, 3), k, n) for t, k, n in self.events]

    def submit(self, label, tag, meta, thread, hold=False):
        if hold:
            self.holds[label] = asyncio.Event()

        async def run():
            res = await self.mgr.submit_and_wait(
                tag, label, thread_id=thread, client_meta=dict(meta))
            self.stamp("delivered", label)
            return res

        task = asyncio.create_task(asyncio.wait_for(run(), timeout=60))
        self.tasks[label] = task
        return task

    async def until(self, predicate, what, timeout=8.0):
        t0 = time.monotonic()
        while not predicate():
            if time.monotonic() - t0 > timeout:
                raise AssertionError(
                    f"never happened within {timeout}s: {what}; events={self.rel()}")
            await asyncio.sleep(0.005)

    def one_model_fits(self, tag):
        need, parallel, main_gpu, split_mode, *_ = self.mgr._resolve_placement_locked(tag)
        return self.mgr._vram_admits_locked(need, parallel, main_gpu, split_mode)


def _manifest(boot, tag):
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": tag, "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": FOOTPRINT_MIB * 1024 * 1024, "context_size": 2048,
        "expected_vram_bytes": FOOTPRINT_MIB * 1024 * 1024,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }))


@contextlib.asynccontextmanager
async def box(tmp_path, monkeypatch, *, grace_s):
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=CLIENT_1_FIRST, max_parallel_sidecars=2, grace_seconds=grace_s,
        max_grace_extensions=5, idle_hot_load_seconds=120)
    for t in (TAG_A, TAG_B):
        _manifest(boot, t)
    spawn, health, _sigterm, vram, _complete = make_fakes({})
    holder: dict = {}

    def spawn_rec(binary, gguf, port, model_tag, argv, **kw):
        holder["box"].stamp("load", model_tag)
        return spawn(binary, gguf, port, model_tag, argv, **kw)

    async def sigterm_rec(*a, **k):
        holder["box"].stamp("engine_stop", "sigterm")
        return True, "sigterm-clean"

    async def complete_rec(slot, handle):
        b = holder["box"]
        label = slot.prompt
        b.stamp("turn_start", label)
        gate = b.holds.get(label)
        try:
            if gate is not None:
                await gate.wait()
        except asyncio.CancelledError:
            b.stamp("turn_cancelled", label)
            raise
        b.stamp("turn_end", label)
        return {"ok": True, "label": label, "model": handle.model_tag}

    def free_vram():
        loaded = [r for r in holder["mgr"]._model_residents() if r.state in LOADED]
        return [CARD_MIB - FOOTPRINT_MIB * len(loaded)]

    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", free_vram)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", free_vram)
    # Push every retry timer far out: only a wake can advance the waiting claimant inside the bound.
    monkeypatch.setattr("turbohaul.manager._DISPATCH_DEFER_BACKOFF_S", RETRY_TIMER_S)
    monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_BACKOFF_S", RETRY_TIMER_S)
    mgr = TurbohaulManager(boot, runtime, spawn_fn=spawn_rec, health_fn=health,
                           sigterm_fn=sigterm_rec, vram_fn=vram, complete_fn=complete_rec)
    b = Box(mgr=mgr, holds={}, t0=time.monotonic())
    holder["mgr"], holder["box"] = mgr, b

    orig_unload = mgr._begin_unload_locked

    def unload_spy(r):
        b.unload_states.append((len(b.events), r.state.name))
        b.stamp("unload", r.model_tag)
        return orig_unload(r)

    mgr._begin_unload_locked = unload_spy

    async def save_kv_rec(port, model_tag, slot=None, **kw):
        # The conversation-state save the teardown makes before it stops the engine. The fake engine
        # has no HTTP surface, so the recorded call (and its thread id) is the observation.
        b.kv_calls.append({"tag": model_tag, "thread_id": kw.get("thread_id_override")})
        b.stamp("kv_save", f"{model_tag}:{kw.get('thread_id_override')}")
        return True

    mgr._save_slot_kv = save_kv_rec

    orig_set, orig_notify = mgr._dispatch_wake.set, mgr._make_room_signal.notify_all

    def wake_set():
        b.stamp("wake", "dispatch")
        return orig_set()

    def wake_notify(*a, **k):
        b.stamp("wake", "make_room")
        return orig_notify(*a, **k)

    mgr._dispatch_wake.set = wake_set
    mgr._make_room_signal.notify_all = wake_notify

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        yield b
    finally:
        # Release the gates first, then stop the loop, then cancel the worker; every await is bounded.
        for g in list(b.holds.values()):
            g.set()
        mgr._stop_event.set()
        mgr._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(mgr._worker_task, 10)
        for t in b.tasks.values():
            if not t.done():
                t.cancel()
        for t in b.tasks.values():
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(t, 5)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(mgr.shutdown(), 10)


async def _arrange_follow_up(b, *, hold, grace_s=None):
    """Client 2's model completes a turn and its grace window ends; a follow-up is then served warm.

    Returns the resident. With hold=True the follow-up turn is mid-generation and not finished.
    With grace_s the configured grace window is set to that many seconds before the follow-up
    starts (the setting is read at each turn), so the follow-up turn runs with a long window.
    """
    mgr = b.mgr
    b.submit("A1", TAG_A, CURATOR_2, "th-a")
    await b.until(lambda: b.seen("turn_end", "A1"), "the first turn to end")
    v = resident_for(mgr, TAG_A)
    await b.until(lambda: v.grace is not None and v.grace._started_at is not None,
                  "the grace window to start")
    await b.until(lambda: (not v.in_grace_loop) and v.state is ResidentState.IDLE_EVICTABLE,
                  "the grace window to end", 12.0)
    if grace_s is not None:
        mgr.runtime.queue.grace_seconds = grace_s
    b.submit("A2", TAG_A, CURATOR_2, "th-a", hold=hold)
    await b.until(lambda: b.seen("turn_start", "A2"), "the follow-up turn to start")
    assert b.count("load", TAG_A) == 1, (
        f"the follow-up was not served warm on the same engine; events={b.rel()}")
    if hold:
        assert v.state is ResidentState.ACTIVE, f"the resident is not mid-turn: {v.state}"
        assert not b.tasks["A2"].done(), "the held follow-up turn already finished"
    assert b.one_model_fits(TAG_B) is False, "the card admits a second model; no contention"
    return v


_CUT_KINDS = ("unload", "engine_stop", "turn_cancelled")


def _cut_events(b, v):
    """Teardown-type events stamped BEFORE the follow-up turn ended (empty = the turn was not cut).

    Order-aware: an unload, engine stop, cancel, DEAD resident or claimant load stamped before the
    turn_end event is the cut; the same events stamped after it are the hand-over. Before the
    release the turn has not been allowed to end, so an early turn_end is itself a cut.
    """
    i_end = b.idx("turn_end", "A2")
    i_rel = b.idx("release", "A2")
    horizon = i_end if i_end is not None else len(b.events)
    cut = [(k, n) for _t, k, n in b.events[:horizon]
           if k in _CUT_KINDS or (k == "load" and n == TAG_B)]
    if i_end is not None and (i_rel is None or i_end < i_rel):
        cut.append(("turn_end", "A2 before the release"))
    if i_end is None and v.state is ResidentState.DEAD:
        cut.append(("resident.state", "DEAD"))
    if i_rel is None and b.tasks["A2"].done():
        cut.append(("held_task", "done before the release"))
    return cut


async def _observe_cut(b, v):
    """Watch the held turn for OBSERVE_S; return the cut events found (empty = not cut)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < OBSERVE_S:
        cut = _cut_events(b, v)
        if cut:
            return cut
        await asyncio.sleep(0.01)
    return _cut_events(b, v)


def _assert_turn_boundary_unload(b, v, *, since, label, check_cut=True):
    """The unload began soon after `since`, from the state the resident served in.

    A resident that is parked idle first, or that finishes a grace window first, is not torn down
    at the turn boundary: the grace window is LONG_GRACE_S and the retry timers RETRY_TIMER_S, so
    either shows up as a gap far above TURN_BOUNDARY_S, or as a state other than ACTIVE.
    """
    i_unload = b.idx("unload", TAG_A)
    assert i_unload is not None, f"{label}: the resident was never unloaded; events={b.rel()}"
    gap = b.events[i_unload][0] - since
    print(f"{label}: gap from the turn boundary to the unload beginning: {gap:.3f}s "
          f"(bound {TURN_BOUNDARY_S}s)")
    assert gap <= TURN_BOUNDARY_S, (
        f"{label}: the unload began {gap:.3f}s after the turn boundary, later than "
        f"{TURN_BOUNDARY_S}s (grace window {LONG_GRACE_S}s, retry timers {RETRY_TIMER_S}s): "
        f"not a turn-boundary teardown; events={b.rel()}")
    states = [st for i, st in b.unload_states if i == i_unload]
    assert states == ["ACTIVE"], (
        f"{label}: the resident was in state {states} when its unload began, expected ACTIVE "
        f"(no parking idle or in grace between the turn boundary and the unload); "
        f"events={b.rel()}")
    if check_cut:
        assert not _cut_events(b, v), (
            f"{label}: teardown events before the turn ended: {_cut_events(b, v)}; "
            f"events={b.rel()}")


def _assert_hand_over_order(b, *, after, label):
    """Event order of the hand-over, by position in the stamped list.

    `after` is the (kind, name) that must precede the unload: the delivery of the follow-up's
    response. Order required: delivered < unload(resident) < kv_save < engine_stop < load(claimant)
    < claimant's turn ended. A wake must sit between the turn end and the claimant's load.
    """
    ev = b.rel()
    i_after = b.idx(*after)
    i_unload = b.idx("unload", TAG_A)
    i_kv = b.idx("kv_save", f"{TAG_A}:th-a")
    i_stop = b.idx("engine_stop", "sigterm")
    i_load = b.idx("load", TAG_B)
    i_done = b.idx("turn_end", "M1")
    assert i_after is not None and i_unload is not None and i_after < i_unload, (
        f"{label}: the resident's unload began before the follow-up response was delivered "
        f"(delivered at {i_after}, unload at {i_unload}); events={ev}")
    assert i_kv is not None and i_unload < i_kv, (
        f"{label}: no conversation-state save for the resident after its unload began "
        f"(kv at {i_kv}, unload at {i_unload}); calls={b.kv_calls}; events={ev}")
    assert i_stop is not None and i_kv < i_stop, (
        f"{label}: the engine was stopped before the conversation state was saved "
        f"(kv at {i_kv}, stop at {i_stop}); events={ev}")
    assert i_load is not None and i_stop < i_load, (
        f"{label}: the claimant's model loaded before the resident's engine stopped "
        f"(stop at {i_stop}, load at {i_load}); events={ev}")
    assert i_done is not None and i_load < i_done, (
        f"{label}: the claimant's turn ended before its model loaded; events={ev}")
    wakes = [i for i, (_t, k, _n) in enumerate(b.events) if k == "wake"]
    i_turn_end = b.idx("turn_end", "A2")
    assert any(i_turn_end < i < i_load for i in wakes), (
        f"{label}: no wake (dispatch event set or make-room notify) between the follow-up turn "
        f"ending (event {i_turn_end}) and the claimant's model loading (event {i_load}); "
        f"wakes at {wakes}; events={ev}")


@pytest.mark.asyncio
class TestFollowUpTurnBoundary:
    async def test_designated_victim_is_torn_down_at_the_end_of_its_follow_up_turn_and_the_claimant_is_woken(
            self, tmp_path, monkeypatch):
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            v = await _arrange_follow_up(b, hold=True, grace_s=LONG_GRACE_S)
            b.stamp("claimant_submit", "M1")
            b.submit("M1", TAG_B, MAIN_1, "th-main")
            await b.until(lambda: len(b.mgr._fastlane_claims) > 0,
                          "the claimant's claim to register")

            # 1. While the follow-up turn is held nothing may cut it or hand its room away.
            cut = await _observe_cut(b, v)
            assert not cut, (
                f"the follow-up turn was CUT while its generation was still held: {cut}; "
                f"events={b.rel()}")
            assert not b.unload_states, (
                f"the resident's unload began while its follow-up turn was held: "
                f"{b.unload_states}; events={b.rel()}")

            # 2. Release the turn: the response is delivered intact, then the hand-over happens.
            b.stamp("release", "A2")
            b.holds["A2"].set()
            res = await asyncio.wait_for(b.tasks["A2"], 10)
            assert res[1] == {"ok": True, "label": "A2", "model": TAG_A}, (
                f"the held turn did not complete intact: {res[1]!r}")

            # 3. The claimant is advanced by the wake, well below the 30 s retry timers.
            t_release = b.when("release", "A2")
            await b.until(lambda: b.seen("load", TAG_B),
                          f"the claimant's model to load within {WAKE_BOUND_S}s of the release "
                          f"(retry timers are {RETRY_TIMER_S}s)", WAKE_BOUND_S)
            await b.until(lambda: b.seen("turn_end", "M1"), "the claimant's turn to end",
                          WAKE_BOUND_S)
            waited = b.when("load", TAG_B) - t_release
            assert waited < WAKE_BOUND_S, (
                f"the claimant loaded {waited:.2f}s after the release, not inside "
                f"{WAKE_BOUND_S}s; events={b.rel()}")
            assert b.kv_calls and b.kv_calls[0]["thread_id"] == "th-a", (
                f"the save did not carry the resident's conversation id: {b.kv_calls}")

            _assert_turn_boundary_unload(b, v, since=b.when("turn_end", "A2"),
                                         label="held follow-up")

            _assert_hand_over_order(b, after=("delivered", "A2"), label="held follow-up")

    async def test_follow_up_that_finishes_quickly_hands_over_through_the_wake(
            self, tmp_path, monkeypatch):
        async with box(tmp_path, monkeypatch, grace_s=1) as b:
            v = await _arrange_follow_up(b, hold=False, grace_s=LONG_GRACE_S)
            await b.until(lambda: b.seen("delivered", "A2"), "the follow-up response to be delivered")
            assert b.tasks["A2"].done() and b.tasks["A2"].result()[1] == {
                "ok": True, "label": "A2", "model": TAG_A}, "the quick follow-up did not complete intact"
            assert b.one_model_fits(TAG_B) is False, "the card admits a second model; no contention"
            b.stamp("claimant_submit", "M1")
            t_submit = b.when("claimant_submit", "M1")
            b.submit("M1", TAG_B, MAIN_1, "th-main")

            await b.until(lambda: b.seen("load", TAG_B),
                          f"the claimant's model to load within {WAKE_BOUND_S}s of its request "
                          f"(retry timers are {RETRY_TIMER_S}s)", WAKE_BOUND_S)
            await b.until(lambda: b.seen("turn_end", "M1"), "the claimant's turn to end",
                          WAKE_BOUND_S)
            waited = b.when("load", TAG_B) - t_submit
            assert waited < WAKE_BOUND_S, (
                f"the claimant loaded {waited:.2f}s after its request, not inside "
                f"{WAKE_BOUND_S}s; events={b.rel()}")

            ev = b.rel()
            i_deliver, i_submit = b.idx("delivered", "A2"), b.idx("claimant_submit", "M1")
            i_unload, i_kv = b.idx("unload", TAG_A), b.idx("kv_save", f"{TAG_A}:th-a")
            i_stop, i_load = b.idx("engine_stop", "sigterm"), b.idx("load", TAG_B)
            i_done = b.idx("turn_end", "M1")
            assert i_deliver < i_submit < i_unload, (
                f"quick follow-up: unload did not follow the claimant's request; events={ev}")
            _assert_turn_boundary_unload(b, v, since=t_submit, label="quick follow-up",
                                         check_cut=False)
            assert i_kv is not None and i_unload < i_kv < i_stop < i_load < i_done, (
                f"quick follow-up: hand-over order wrong (unload {i_unload}, kv {i_kv}, "
                f"stop {i_stop}, load {i_load}, done {i_done}); events={ev}")
            wakes = [i for i, (_t, k, _n) in enumerate(b.events) if k == "wake"]
            assert any(i_submit < i < i_load for i in wakes), (
                f"quick follow-up: no wake between the claimant's request and its model "
                f"loading; wakes at {wakes}; events={ev}")

    async def test_follow_up_served_inside_the_grace_window_is_torn_down_at_its_turn_end(
            self, tmp_path, monkeypatch):
        """The follow-up arrives while the resident is still inside its grace window and is served
        there. A better client then asks for the room while that turn is held. The turn runs to
        completion, and the grace window (10 s) is surrendered at its end: the unload begins at the
        turn boundary, not after the window runs out."""
        async with box(tmp_path, monkeypatch, grace_s=LONG_GRACE_S) as b:
            b.submit("A1", TAG_A, CURATOR_2, "th-a")
            await b.until(lambda: b.seen("turn_end", "A1"), "the first turn to end")
            v = resident_for(b.mgr, TAG_A)
            await b.until(lambda: v.in_grace_loop, "the resident to be inside its grace window")
            b.submit("A2", TAG_A, CURATOR_2, "th-a", hold=True)
            await b.until(lambda: b.seen("turn_start", "A2"), "the follow-up turn to start")
            assert v.in_grace_loop and v.state is ResidentState.ACTIVE and not b.tasks["A2"].done(), (
                f"the follow-up is not mid-turn inside the grace window: state={v.state}, "
                f"in_grace_loop={v.in_grace_loop}")
            assert b.count("load", TAG_A) == 1, f"a second engine was loaded; events={b.rel()}"
            assert b.one_model_fits(TAG_B) is False, "the card admits a second model; no contention"
            b.submit("M1", TAG_B, MAIN_1, "th-main")
            await b.until(lambda: len(b.mgr._fastlane_claims) > 0,
                          "the claimant's claim to register")

            cut = await _observe_cut(b, v)
            assert not cut, (
                f"the follow-up turn was CUT while its generation was still held: {cut}; "
                f"events={b.rel()}")
            assert not b.unload_states, (
                f"the resident's unload began while its follow-up turn was held: "
                f"{b.unload_states}; events={b.rel()}")

            b.stamp("release", "A2")
            b.holds["A2"].set()
            res = await asyncio.wait_for(b.tasks["A2"], 10)
            assert res[1] == {"ok": True, "label": "A2", "model": TAG_A}, (
                f"the held turn did not complete intact: {res[1]!r}")
            await b.until(lambda: b.seen("unload", TAG_A),
                          f"the unload to begin within {TURN_BOUNDARY_S}s of the turn end "
                          f"(grace window {LONG_GRACE_S}s)", TURN_BOUNDARY_S)
            await b.until(lambda: b.seen("turn_end", "M1"), "the claimant's turn to end",
                          WAKE_BOUND_S)
            _assert_turn_boundary_unload(b, v, since=b.when("turn_end", "A2"),
                                         label="follow-up inside the grace window")
            _assert_hand_over_order(b, after=("delivered", "A2"),
                                    label="follow-up inside the grace window")
