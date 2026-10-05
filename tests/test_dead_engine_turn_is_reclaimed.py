"""A turn whose engine is dead is ended and its resident is reclaimed, and a waiting claimant is woken.

A running turn is never cut by an unload. That protection must never extend to a turn whose engine
is gone: a dead engine ends the turn (the request fails, or the stream cap fires for a hung
stream), the resident is deregistered, and the better client waiting for the card is admitted
through a wake at that moment, not at the end of a long backstop timer.

These tests pin what the code does today on that path, so a busy check on the unload decision can
never keep a dead turn alive:

  * test_dead_engine_mid_turn_ends_the_turn_and_hands_over: the resident's second turn is held
    mid-generation, a better client for another model is waiting, then the engine dies. The held
    request fails with the engine's own error, the resident is deregistered, no KV save is
    attempted against the dead engine, a wake fires, and the waiting client is admitted and
    completes long before the backstop (both defer backoffs are set to 30 s).
  * test_hung_stream_ends_at_the_stream_cap_and_is_reclaimed: the resident's second turn is a
    stream nobody ever finishes. The manager's stream cap (set tiny for the test) ends it, the
    resident is reclaimed at that turn boundary and the waiting client is admitted.
  * test_live_turn_is_still_protected_while_engine_is_alive: the same shape with the engine ALIVE
    and a wake injected at the same moment the death would have produced one: the live turn is not
    cut. It is the negative control of the first test: only the engine's liveness differs.
  * test_kv_spy_sees_the_save_on_a_live_turn_boundary_teardown: shows the KV spies used by the
    first test can see a save when one happens (a live engine torn down at its turn boundary).

Addresses are documentation ranges. Every wait is bounded; no sleep is used as synchronisation.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from dataclasses import dataclass, field

import pytest
import yaml

from _fastlane_fixture import boot_ranked_runtime, make_fakes, resident_for
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState, TurbohaulManager

IP_1 = "192.0.2.10"
IP_2 = "198.51.100.20"
IP_3 = "203.0.113.30"
FOOTPRINT_MIB = 20000
CARD_MIB = 24000
LOADED = (ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE)
TAG_A, TAG_B, TAG_C = "model-a", "model-b", "model-c"
FOOTPRINTS_MIB = {TAG_A: FOOTPRINT_MIB, TAG_B: FOOTPRINT_MIB, TAG_C: 35000}
RANKS = FastLaneTagRanks(main=1, curator=3, unclassified=5)
CLIENT_1_FIRST = [FastLaneRule(address=IP_1, tag_ranks=RANKS),
                  FastLaneRule(address=IP_2, tag_ranks=RANKS)]
CURATOR_2 = {"ip": IP_2, "is_curator": True, "is_sub_agent": True}
MAIN_1 = {"ip": IP_1, "is_main": True}
MAIN_3 = {"ip": IP_3, "is_main": True}
# Three clients in priority order: the one that owns A, the claimant, the one that owns B.
CLIENT_ORDER_A_CLAIMANT_B = [FastLaneRule(address=IP_1, tag_ranks=RANKS),
                             FastLaneRule(address=IP_3, tag_ranks=RANKS),
                             FastLaneRule(address=IP_2, tag_ranks=RANKS)]
LONG_BACKOFF_S = 30.0     # the lost-wake backstop; every handover below must beat it by far
HANDOVER_BOUND_S = 8.0    # well under LONG_BACKOFF_S
OBSERVE_S = 3.0
STREAM_CAP_S = 2.0
ENGINE_ERROR = "engine connection lost"


@dataclass
class Box:
    mgr: TurbohaulManager
    t0: float
    outcomes: dict = field(default_factory=dict)   # label -> future the fake turn waits on
    handles: dict = field(default_factory=dict)    # label -> engine handle that served it
    events: list = field(default_factory=list)     # (monotonic, kind, name, extra)
    tasks: dict = field(default_factory=dict)
    streams: dict = field(default_factory=dict)    # label -> slot of a stream turn
    route_tasks: dict = field(default_factory=dict)

    def stamp(self, kind, name="", extra=None):
        self.events.append((time.monotonic(), kind, name, extra))

    def seen(self, kind, name=None):
        return any(k == kind and (name is None or n == name) for _t, k, n, _x in self.events)

    def when(self, kind, name=None):
        for t, k, n, _x in self.events:
            if k == kind and (name is None or n == name):
                return t
        return None

    def all_when(self, kind, name=None):
        return [t for t, k, n, _x in self.events if k == kind and (name is None or n == name)]

    def rel(self):
        return [(round(t - self.t0, 3), k, n, x) for t, k, n, x in self.events]

    def submit(self, label, tag, meta, thread, hold=False):
        if hold:
            self.outcomes[label] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(asyncio.wait_for(
            self.mgr.submit_and_wait(tag, label, thread_id=thread, client_meta=dict(meta)),
            timeout=60))
        self.tasks[label] = task
        return task

    async def submit_stream(self, label, tag, meta, thread):
        """A stream turn whose route never finishes it (a hung stream)."""
        slot = await self.mgr.submit_for_streaming(
            tag, label, thread_id=thread, client_meta=dict(meta, stream=True))
        self.streams[label] = slot

        async def route():
            await slot.stream_ready_event.wait()
            self.stamp("turn_start", label)
            try:
                await slot.completion_future
            except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
                self.stamp("stream_turn_failed", label, type(exc).__name__)
                raise
            self.stamp("turn_end", label)

        self.route_tasks[label] = asyncio.create_task(route())
        return slot

    def release(self, label):
        fut = self.outcomes.get(label)
        if fut is not None and not fut.done():
            fut.set_result(None)

    def kill_engine(self, label):
        """The engine process is gone: the handle reports not alive and the call fails."""
        handle = self.handles[label]
        handle.proc.poll.return_value = 1
        self.stamp("engine_dead", label)
        fut = self.outcomes[label]
        fut.set_exception(ConnectionError(ENGINE_ERROR))

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
    size = FOOTPRINTS_MIB[tag] * 1024 * 1024
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump({
        "model_tag": tag, "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": size, "context_size": 2048,
        "expected_vram_bytes": size,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }))


def _caller(depth=2):
    f = sys._getframe(depth)
    return f"{f.f_code.co_name}:{f.f_lineno}"


@contextlib.asynccontextmanager
async def box(tmp_path, monkeypatch, *, grace_s, backoff_s=None, stream_cap_s=None,
              card_mib=CARD_MIB, idle_s=120, max_extensions=5, rules=None):
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules or CLIENT_1_FIRST, max_parallel_sidecars=2, grace_seconds=grace_s,
        max_grace_extensions=max_extensions, idle_hot_load_seconds=idle_s)
    for t in (TAG_A, TAG_B, TAG_C):
        _manifest(boot, t)
    if backoff_s is not None:
        monkeypatch.setattr("turbohaul.manager._DISPATCH_DEFER_BACKOFF_S", backoff_s)
        monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_BACKOFF_S", backoff_s)
    if stream_cap_s is not None:
        monkeypatch.setattr("turbohaul.manager._STREAM_TIMEOUT_S", stream_cap_s)
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
        b.handles[label] = handle
        b.stamp("turn_start", label)
        fut = b.outcomes.get(label)
        try:
            if fut is not None:
                await fut
        except asyncio.CancelledError:
            b.stamp("turn_cancelled", label)
            raise
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            b.stamp("turn_failed", label, type(exc).__name__)
            raise
        b.stamp("turn_end", label)
        return {"ok": True, "label": label, "model": handle.model_tag}

    def free_vram():
        loaded = [r for r in holder["mgr"]._model_residents() if r.state in LOADED]
        return [card_mib - sum(FOOTPRINTS_MIB[r.model_tag] for r in loaded)]

    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", free_vram)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", free_vram)
    mgr = TurbohaulManager(boot, runtime, spawn_fn=spawn_rec, health_fn=health,
                           sigterm_fn=sigterm_rec, vram_fn=vram, complete_fn=complete_rec)
    b = Box(mgr=mgr, t0=time.monotonic())
    holder["mgr"], holder["box"] = mgr, b

    orig_unload = mgr._begin_unload_locked

    def unload_spy(r):
        b.stamp("unload", r.model_tag, _caller())
        return orig_unload(r)

    mgr._begin_unload_locked = unload_spy

    orig_set = mgr._dispatch_wake.set
    orig_notify = mgr._make_room_signal.notify_all

    def set_spy():
        b.stamp("wake_set", "", _caller())
        return orig_set()

    def notify_spy(*a, **k):
        b.stamp("wake_notify", "", _caller())
        return orig_notify(*a, **k)

    mgr._dispatch_wake.set = set_spy
    mgr._make_room_signal.notify_all = notify_spy

    orig_save = mgr._save_slot_kv
    orig_flush = mgr._flush_clean_kv_at_unload
    orig_outcome = mgr._log_kv_reuse_outcome

    async def save_spy(port, model_tag, *a, **k):
        b.stamp("kv_save", model_tag, port)
        return await orig_save(port, model_tag, *a, **k)

    async def flush_spy(handle, model_tag, *a, **k):
        b.stamp("kv_flush", model_tag, handle.is_alive())
        return await orig_flush(handle, model_tag, *a, **k)

    def outcome_spy(slot, handle, **k):
        if k.get("wait_state") == "timeout":
            b.stamp("stream_cap", slot.prompt)
        return orig_outcome(slot, handle, **k)

    mgr._save_slot_kv = save_spy
    mgr._flush_clean_kv_at_unload = flush_spy
    mgr._log_kv_reuse_outcome = outcome_spy

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        yield b
    finally:
        # Release every held turn first, then stop the loop, then cancel the worker; all bounded.
        for label in list(b.outcomes):
            b.release(label)
        for slot in b.streams.values():
            if slot.stream_done_event is not None:
                slot.stream_done_event.set()
        mgr._stop_event.set()
        mgr._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(mgr._worker_task, 10)
        for t in [*b.tasks.values(), *b.route_tasks.values()]:
            if not t.done():
                t.cancel()
        for t in [*b.tasks.values(), *b.route_tasks.values()]:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(t, 5)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(mgr.shutdown(), 10)


async def _park_a_after_first_turn(b):
    """Client 2's model A serves one turn, passes through a full grace window and parks."""
    mgr = b.mgr
    b.submit("A1", TAG_A, CURATOR_2, "th-a")
    await b.until(lambda: b.seen("turn_end", "A1"), "A's first turn to end")
    a = resident_for(mgr, TAG_A)
    await b.until(lambda: a.grace is not None and a.grace._started_at is not None,
                  "A's grace window to start")
    await b.until(lambda: a.in_grace_loop, "A to be inside its grace loop")
    await b.until(lambda: (not a.in_grace_loop) and a.state is ResidentState.IDLE_EVICTABLE,
                  "A's grace window to end", 12.0)
    return a


async def _claim_while_turn_runs(b, *, second_turn, stream=False):
    """A's turn is mid-generation; a better client's request for model B then waits.

    second_turn=True: A first served a turn and passed through a full grace window, so the
    running turn is its second. False: the running turn is A's first (A never entered grace).
    Returns (label of the running turn, A's resident object). The object is kept because a
    reclaim removes A from the manager's resident list.
    """
    mgr = b.mgr
    label = "A2" if second_turn else "A1"
    if second_turn:
        await _park_a_after_first_turn(b)
    if stream:
        await b.submit_stream(label, TAG_A, CURATOR_2, "th-a")
    else:
        b.submit(label, TAG_A, CURATOR_2, "th-a", hold=True)
    await b.until(lambda: b.seen("turn_start", label), "A's running turn to start")
    a = resident_for(mgr, TAG_A)
    assert a.state is ResidentState.ACTIVE, f"A is not mid-turn: {a.state}"
    assert a.active_slot is not None, "A has no running slot"
    assert b.one_model_fits(TAG_B) is False, "the card admits a second model; no contention"
    b.stamp("claimant_submit", "M1")
    b.submit("M1", TAG_B, MAIN_1, "th-main")
    await b.until(lambda: len(mgr._fastlane_claims) > 0, "the claimant's claim to register")
    return label, a


def _assert_running_turn_not_yet_cut(b, a, label):
    assert not b.seen("unload", TAG_A) and a.state is ResidentState.ACTIVE, (
        f"the running turn was cut by the waiting claim before anything else happened: "
        f"unload(A) seen={b.seen('unload', TAG_A)}, A.state={a.state}; events={b.rel()}")
    assert not b.tasks[label].done() if label in b.tasks else True


async def _wait_claimant_parked(b, a, label):
    """The waiting claimant is parked on the make-room signal (so a wake can reach it).

    The claim is registered a moment before the claimant starts waiting; a wake signalled in
    that gap reaches nobody. A cut seen while waiting is reported as the cut.
    """
    t0 = time.monotonic()
    while len(b.mgr._make_room_signal._waiters) == 0:
        _assert_running_turn_not_yet_cut(b, a, label)
        if time.monotonic() - t0 > 5.0:
            raise AssertionError(
                f"the claimant never parked on the make-room signal; events={b.rel()}")
        await asyncio.sleep(0.002)
    _assert_running_turn_not_yet_cut(b, a, label)


def _is_deregistered(b, a):
    return a.state is ResidentState.DEAD and all(r is not a for r in b.mgr._model_residents())


TURN = pytest.mark.parametrize("second_turn", [False, True], ids=["first_turn", "second_turn"])


def _grace_for(second_turn):
    return 1 if second_turn else 30


@pytest.mark.asyncio
class TestDeadEngineTurnIsReclaimed:
    @TURN
    async def test_dead_engine_mid_turn_ends_the_turn_and_hands_over(
            self, tmp_path, monkeypatch, second_turn):
        async with box(tmp_path, monkeypatch, grace_s=_grace_for(second_turn),
                       backoff_s=LONG_BACKOFF_S) as b:
            held, a = await _claim_while_turn_runs(b, second_turn=second_turn)
            # Before the death: the claim is registered, the turn runs, nothing was torn down,
            # and the claimant is parked waiting to be woken.
            _assert_running_turn_not_yet_cut(b, a, held)
            await _wait_claimant_parked(b, a, held)

            b.kill_engine(held)

            # 1. the dead turn ENDS, with the engine's own error.
            await b.until(lambda: b.tasks[held].done(), "the dead turn's request to end", 5.0)
            exc = b.tasks[held].exception()
            assert isinstance(exc, ConnectionError) and ENGINE_ERROR in str(exc), (
                f"the dead turn did not end with the engine's error: {exc!r}; events={b.rel()}")

            # 2. the resident is reclaimed.
            await b.until(lambda: _is_deregistered(b, a), "A to be DEAD and deregistered", 5.0)
            assert b.seen("unload", TAG_A), f"A was never unloaded; events={b.rel()}"

            # 3. the end of the dead turn is a decision moment: a wake is signalled. (The
            # claimant is on a 30 s backstop here, so only a wake can advance it in time.)
            t_dead = b.when("engine_dead", held)

            def woken_after_death():
                return any(t >= t_dead for t in
                           (*b.all_when("wake_notify"), *b.all_when("wake_set")))

            await b.until(woken_after_death,
                          "a wake after the dead turn ended and its resident was reclaimed", 3.0)

            # 4. the claimant is admitted and completes, long before the backstop.
            await b.until(lambda: b.seen("turn_end", "M1"),
                          "the claimant to complete after A's turn died", HANDOVER_BOUND_S)
            t_failed = b.when("turn_failed", held)
            t_unload = b.when("unload", TAG_A)
            t_load = b.when("load", TAG_B)
            t_m1 = b.when("turn_end", "M1")
            ev = b.rel()
            assert t_dead <= t_failed <= t_unload <= t_load <= t_m1, (
                f"wrong order (dead, turn failed, unload, load B, M1 done); events={ev}")
            assert t_m1 - t_dead < HANDOVER_BOUND_S < LONG_BACKOFF_S, (
                f"handover took {t_m1 - t_dead:.2f}s, not under {HANDOVER_BOUND_S}s; events={ev}")

            # 5. it was a WAKE that advanced the claimant, not the backstop timer: a wake was
            # signalled after the engine died and before B was loaded.
            wakes = [t for t in (*b.all_when("wake_notify"), *b.all_when("wake_set"))
                     if t_dead <= t <= t_load]
            assert wakes, f"no wake between the engine's death and B's load; events={ev}"

            # 6. no KV save was attempted against the dead engine.
            late_saves = [(round(t - b.t0, 3), k, n) for t, k, n, _x in b.events
                          if k in ("kv_save", "kv_flush") and n == TAG_A and t >= t_dead]
            assert not late_saves, (
                f"a KV save was attempted for A after its engine died: {late_saves}; events={ev}")

    @TURN
    async def test_hung_stream_ends_at_the_stream_cap_and_is_reclaimed(
            self, tmp_path, monkeypatch, second_turn):
        async with box(tmp_path, monkeypatch, grace_s=_grace_for(second_turn),
                       backoff_s=LONG_BACKOFF_S, stream_cap_s=STREAM_CAP_S) as b:
            held, a = await _claim_while_turn_runs(b, second_turn=second_turn, stream=True)
            t_claim = b.when("claimant_submit", "M1")
            _assert_running_turn_not_yet_cut(b, a, held)
            await _wait_claimant_parked(b, a, held)

            # The stream is never finished by its route. The manager's own cap ends the turn.
            await b.until(lambda: b.seen("stream_cap", held),
                          "the stream cap to fire for the hung stream", STREAM_CAP_S + 6.0)
            t_cap = b.when("stream_cap", held)
            assert t_cap - t_claim >= STREAM_CAP_S - 0.5, (
                f"the stream ended after {t_cap - t_claim:.2f}s, before the cap of "
                f"{STREAM_CAP_S}s; events={b.rel()}")

            await b.until(lambda: b.seen("turn_end", held), "the capped stream turn to end", 5.0)
            await b.until(lambda: _is_deregistered(b, a), "A to be DEAD and deregistered", 8.0)
            await b.until(lambda: b.seen("turn_end", "M1"),
                          "the claimant to complete after the stream cap", HANDOVER_BOUND_S)
            t_end = b.when("turn_end", held)
            t_unload = b.when("unload", TAG_A)
            t_load = b.when("load", TAG_B)
            t_m1 = b.when("turn_end", "M1")
            ev = b.rel()
            assert t_cap <= t_end <= t_unload <= t_load <= t_m1, (
                f"wrong order (cap, turn end, unload, load B, M1 done); events={ev}")
            assert t_m1 - t_cap < HANDOVER_BOUND_S < LONG_BACKOFF_S, (
                f"handover took {t_m1 - t_cap:.2f}s after the cap; events={ev}")
            wakes = [t for t in (*b.all_when("wake_notify"), *b.all_when("wake_set"))
                     if t_cap <= t <= t_load]
            assert wakes, f"no wake between the stream cap and B's load; events={ev}"

    @TURN
    async def test_live_turn_is_still_protected_while_engine_is_alive(
            self, tmp_path, monkeypatch, second_turn):
        async with box(tmp_path, monkeypatch, grace_s=_grace_for(second_turn),
                       backoff_s=LONG_BACKOFF_S) as b:
            held, a = await _claim_while_turn_runs(b, second_turn=second_turn)
            _assert_running_turn_not_yet_cut(b, a, held)
            await _wait_claimant_parked(b, a, held)
            assert b.handles[held].is_alive(), "the engine is not alive in the live-engine control"

            # The wake a death would have produced, injected while the engine is ALIVE. The
            # claimant is re-evaluated fresh; the running turn must survive that evaluation.
            async with b.mgr._registry_lock:
                b.stamp("injected_wake", "")
                b.mgr._dispatch_wake.set()
                b.mgr._make_room_signal.notify_all()

            t0 = time.monotonic()
            cut = []
            while time.monotonic() - t0 < OBSERVE_S and not cut:
                if b.seen("unload", TAG_A):
                    cut.append("unload(A)")
                if a.state is ResidentState.DEAD:
                    cut.append("A.state=DEAD")
                if b.seen("engine_stop", "sigterm"):
                    cut.append("engine_stop")
                if b.seen("turn_cancelled", held):
                    cut.append("turn_cancelled")
                if b.seen("turn_end", held) or b.tasks[held].done():
                    cut.append("held_turn_ended")
                await asyncio.sleep(0.01)
            assert not cut, (
                f"the live running turn was CUT after a wake with its engine alive: {cut}; "
                f"events={b.rel()}")
            assert b.seen("wake_notify") and b.seen("injected_wake"), "the injected wake was lost"

            # The turn finishes intact and only then does the handover happen.
            b.release(held)
            res = await asyncio.wait_for(b.tasks[held], 10)
            assert res[1] == {"ok": True, "label": held, "model": TAG_A}, (
                f"the held turn did not complete intact: {res[1]!r}")
            await b.until(lambda: b.seen("turn_end", "M1"), "the claimant to complete", 12.0)
            ev = b.rel()
            assert b.when("turn_end", held) <= b.when("unload", TAG_A) <= b.when("load", TAG_B), (
                f"wrong order (turn end, unload, load B); events={ev}")

    async def test_kv_spy_sees_the_save_on_a_live_turn_boundary_teardown(
            self, tmp_path, monkeypatch):
        """Control for the KV assertion of the dead-engine test: the spies do see a save.

        Same claimant, A's first turn held with a live engine. After the turn completes A is
        torn down at its boundary and the spies must record a KV save or flush for A before the
        engine is stopped.
        """
        async with box(tmp_path, monkeypatch, grace_s=30) as b:
            b.submit("A1", TAG_A, CURATOR_2, "th-a", hold=True)
            await b.until(lambda: b.seen("turn_start", "A1"), "A's turn to start")
            b.submit("M1", TAG_B, MAIN_1, "th-main")
            await b.until(lambda: len(b.mgr._fastlane_claims) > 0, "the claim to register")
            assert not b.seen("unload", TAG_A), f"A was cut mid-turn; events={b.rel()}"
            b.release("A1")
            await b.until(lambda: b.seen("turn_end", "M1"), "the claimant to complete", 15.0)
            saves = b.all_when("kv_save", TAG_A) + b.all_when("kv_flush", TAG_A)
            assert saves, f"the KV spies saw no save for A at its live teardown; events={b.rel()}"
            assert b.when("turn_end", "A1") <= b.when("unload", TAG_A), (
                f"A unloaded before its turn ended; events={b.rel()}")
            assert min(saves) <= b.when("engine_stop", "sigterm"), (
                f"no KV save or flush for A before the engine was stopped; events={b.rel()}")


IDLE_EXIT_CARD_MIB = 50000        # holds A and B together (2 x 20000), not B plus the large C
IDLE_S = 1                        # idle window after which A's driver exits on its own


async def _idle_a_and_held_b(b):
    """A (the best client) serves one turn and idles; B (the worst client) is then mid-turn, held.

    Both models are loaded together. Returns the two resident objects.
    """
    mgr = b.mgr
    b.submit("A1", TAG_A, MAIN_1, "th-a")
    await b.until(lambda: b.seen("turn_end", "A1"), "A's turn to end")
    a = resident_for(mgr, TAG_A)
    b.submit("BTURN", TAG_B, CURATOR_2, "th-b", hold=True)
    await b.until(lambda: b.seen("turn_start", "BTURN"), "B's held turn to start")
    res_b = resident_for(mgr, TAG_B)
    assert a is not res_b and a.state is not ResidentState.DEAD, "A is gone before the scenario"
    assert res_b.state is ResidentState.ACTIVE and res_b.active_slot is not None, (
        f"B is not mid-turn: {res_b.state}")
    return a, res_b


def _idle_exit_done(b, a):
    return (a.driver_task is not None and a.driver_task.done()
            and b.seen("engine_stop", "sigterm")
            and all(r is not a for r in b.mgr._model_residents()))


def _protected_b_violations(b, res_b):
    """Reasons B (mid-turn, held) was touched; empty means B was left alone."""
    out = []
    if res_b.state is not ResidentState.ACTIVE:
        out.append(f"B.state={res_b.state.name}")
    if any(n == TAG_B for _t, k, n, _x in b.events if k == "unload"):
        out.append("unload(B)")
    if len(b.all_when("engine_stop", "sigterm")) != 1:
        out.append(f"engine_stop x{len(b.all_when('engine_stop', 'sigterm'))}")
    if b.seen("turn_cancelled", "BTURN") or b.seen("turn_end", "BTURN") or b.tasks["BTURN"].done():
        out.append("B's held turn ended or was cancelled")
    return out


@pytest.mark.asyncio
class TestDriverExitWakeIsInert:
    async def test_normal_idle_exit_with_nobody_waiting_changes_nothing(
            self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.WARNING)
        async with box(tmp_path, monkeypatch, grace_s=0, max_extensions=0, idle_s=IDLE_S,
                       card_mib=IDLE_EXIT_CARD_MIB, rules=CLIENT_ORDER_A_CLAIMANT_B) as b:
            a, res_b = await _idle_a_and_held_b(b)
            mark_events, mark_logs = len(b.events), len(caplog.records)

            await b.until(lambda: _idle_exit_done(b, a),
                          "A's driver to exit on its own idle expiry and its engine to stop", 10.0)
            # Watch for a while after the exit for anything the wake might have set going.
            t0 = time.monotonic()
            while time.monotonic() - t0 < OBSERVE_S:
                assert not _protected_b_violations(b, res_b), (
                    f"B was touched after A's exit: {_protected_b_violations(b, res_b)}; "
                    f"events={b.rel()}")
                await asyncio.sleep(0.02)

            after = b.events[mark_events:]
            wakes = [(k, x) for _t, k, _n, x in after if k in ("wake_set", "wake_notify")]
            assert wakes, f"A's exit signalled no wake at all; events={b.rel()}"
            assert all(str(x).split(":")[0] in ("_drive_resident", "_unload_teardown")
                       for _k, x in wakes), (
                f"a wake came from outside the driver exit and its teardown: {wakes}; "
                f"events={b.rel()}")
            # Nothing else happened: only A was unloaded, nothing was admitted, nothing queued.
            assert {n for _t, k, n, _x in b.events if k == "unload"} == {TAG_A}, (
                f"an unload other than A's; events={b.rel()}")
            assert [n for _t, k, n, _x in b.events if k == "load"] == [TAG_A, TAG_B], (
                f"a model was admitted after A's exit; events={b.rel()}")
            assert not any(k == "turn_start" for _t, k, _n, _x in after), (
                f"a turn started after A's exit; events={b.rel()}")
            depth = b.mgr.queue.depth()
            assert depth["staging_queue_depth"] == 0 and depth["acceptance_buffer_depth"] == 0, (
                f"the queue is not empty after A's exit: {depth}")
            assert res_b.inbox is None or res_b.inbox.empty(), "B's inbox is not empty"
            assert len(b.mgr._fastlane_claims) == 0, "a claim is registered"
            noisy = [(r.levelname, r.name, r.getMessage()[:120])
                     for r in caplog.records[mark_logs:]]
            assert noisy == [], f"log records at WARNING or above after A's exit: {noisy}"

            b.release("BTURN")
            res = await asyncio.wait_for(b.tasks["BTURN"], 10)
            assert res[1] == {"ok": True, "label": "BTURN", "model": TAG_B}, (
                f"B's held turn did not complete intact: {res[1]!r}")

    async def test_wake_with_an_unsatisfiable_claimant_does_not_cut_or_admit(
            self, tmp_path, monkeypatch):
        async with box(tmp_path, monkeypatch, grace_s=0, max_extensions=0, idle_s=IDLE_S,
                       card_mib=IDLE_EXIT_CARD_MIB, rules=CLIENT_ORDER_A_CLAIMANT_B) as b:
            a, res_b = await _idle_a_and_held_b(b)
            passes = []
            orig_route = b.mgr._route_or_reserve

            async def route_spy(slot):
                passes.append((time.monotonic(), slot.prompt))
                return await orig_route(slot)

            b.mgr._route_or_reserve = route_spy
            # A client better than B's, and worse than A's, wants model C. It does not fit next
            # to the running B even once A has gone (35000 + 20000 is more than the card), and
            # A, belonging to a better client, is not taken to make room for it.
            b.submit("C1", TAG_C, MAIN_3, "th-c")
            await b.until(lambda: len(b.mgr._fastlane_claims) > 0, "the claimant's claim")
            mark = len(b.events)

            await b.until(lambda: _idle_exit_done(b, a),
                          "A's driver to exit on its own idle expiry and its engine to stop", 10.0)
            t_exit = max(b.all_when("engine_stop", "sigterm"))
            wakes = [t for t in (*b.all_when("wake_notify"), *b.all_when("wake_set"))
                     if t >= b.events[mark][0]]
            assert wakes, f"A's exit signalled no wake; events={b.rel()}"

            # The claimant is re-evaluated after the wake, and stays waiting.
            await b.until(lambda: any(t >= t_exit and lbl == "C1" for t, lbl in passes),
                          "the claimant to be re-evaluated after A's exit", 5.0)
            t0 = time.monotonic()
            while time.monotonic() - t0 < OBSERVE_S:
                assert not _protected_b_violations(b, res_b), (
                    f"B was touched: {_protected_b_violations(b, res_b)}; events={b.rel()}")
                assert not b.seen("load", TAG_C) and not b.tasks["C1"].done(), (
                    f"the claimant was admitted while B's turn ran; events={b.rel()}")
                await asyncio.sleep(0.02)
            assert [n for _t, k, n, _x in b.events if k == "load"] == [TAG_A, TAG_B], (
                f"a model was admitted while B's turn ran; events={b.rel()}")

            # Only when B's turn ends does the handover happen, in order.
            b.release("BTURN")
            res = await asyncio.wait_for(b.tasks["BTURN"], 10)
            assert res[1] == {"ok": True, "label": "BTURN", "model": TAG_B}, (
                f"B's held turn did not complete intact: {res[1]!r}")
            await b.until(lambda: b.seen("turn_end", "C1"), "the claimant to complete", 12.0)
            ev = b.rel()
            assert b.when("turn_end", "BTURN") <= b.when("unload", TAG_B) <= b.when("load", TAG_C), (
                f"wrong order (B turn end, B unload, load C); events={ev}")
