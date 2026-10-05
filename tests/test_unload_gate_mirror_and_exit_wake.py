"""Two guards on the "an unload never cuts a running turn" work.

1. The diagnostic that explains a make-room miss (`_starvation_reason`) must
   agree with the real victim choice (`_lru_idle_unloadable`) for a resident
   that is running a turn. The real choice has a fallback arm for a designated
   victim whose grace window has ended; that arm never returns a resident that
   is mid-turn (an active slot, or riders in flight). The diagnostic repeats
   the same fallback, and it must repeat the running-turn exclusion too, or it
   reports "eligible" for a resident the real choice refuses to touch.

   The shape needs: an ACTIVE resident, its grace timer started, not inside
   its grace loop, designated as the victim by a live better-ranked claim, no
   idle-evictable resident anywhere, and a turn that is running right now.
   Each running-turn form (active slot, in-flight riders) is its own case. The
   sibling case with NO running turn is the control: the very same shape is
   eligible in both the real choice and the diagnostic, so the shape is
   reachable and the "not eligible" assertions are not vacuous.

2. When a resident's driver exits through the path that owns its teardown, it
   must set the dispatch wake by itself (and wake claimants parked on the
   make-room signal). Other tests run with the dispatcher alive, and a live
   dispatcher clears the event, so they cannot tell whether the driver's own
   exit set it. Here the dispatcher is stopped before the exit, so nothing can
   clear or consume the event, the event is cleared explicitly right before the
   exit, and it must be set right after. The control exits a driver whose
   teardown is already owned by the detached unload: that exit must leave the
   event clear, so the wake is tied to the owning exit and the test cannot pass
   through an unrelated set().

Addresses are documentation ranges. Every wait is bounded and uses latches or
polling of state; no sleep is used as synchronisation, and nothing here depends
on a wall-clock bound.
"""
from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import sys
import time
from dataclasses import dataclass, field

import pytest
import yaml

from _fastlane_fixture import boot_ranked_runtime, make_fakes, resident_for
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.fastlane import CompiledRule, match_fastlane
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer
from turbohaul.slot import Slot

CLAIMANT_IP = "192.0.2.1"
BETTER_IP = "192.0.2.2"
VICTIM_IP = "192.0.2.3"


def _rule(index, raw_address):
    addr = ipaddress.ip_address(raw_address)
    return CompiledRule(
        index=index,
        raw_address=raw_address,
        address=addr,
        container_name=None,
        match_addresses=frozenset({addr}),
        label=f"rule{index}",
        tag_ranks={"unclassified": 1},
    )


# Priority order: the claimant first (best), the better-ranked resident, the victim (worst).
TABLE = [_rule(0, CLAIMANT_IP), _rule(1, BETTER_IP), _rule(2, VICTIM_IP)]


# ---------------------------------------------------------------------------
# 1. the diagnostic mirror agrees with the real decision for a running turn
# ---------------------------------------------------------------------------

def _boot_runtime_for_mirror(tmp_path):
    rules = [FastLaneRule(address=ip, tag_ranks=FastLaneTagRanks(unclassified=1))
             for ip in (CLAIMANT_IP, BETTER_IP, VICTIM_IP)]
    return boot_ranked_runtime(tmp_path, rules=rules, max_parallel_sidecars=2,
                               grace_seconds=5, max_grace_extensions=50)


def _active_resident(tag, ip, *, grace_started=False):
    r = Resident(
        model_tag=tag,
        resident_key=tag,
        state=ResidentState.ACTIVE,
        main_gpu=0,
        split_mode="none",
        reserved_need_mib=100,
        last_active_monotonic=1000.0,
        rank_client_meta={"ip": ip},
    )
    if grace_started:
        r.grace = GraceTimer(grace_seconds=5, max_extensions=50)
        r.grace.start(f"thread-{tag}", tag)
        # The grace loop has ended (in_grace_loop stays False), but the timer has
        # been started at some point and never goes back to "not started".
    return r


def _running_active_slot(r):
    r.active_slot = Slot.new(r.model_tag)


def _running_inflight(r):
    r.inflight = [Slot.new(r.model_tag)]


def _not_running(r):
    r.active_slot = None
    r.inflight = []


# (id, how the turn is made to run, turn is running)
MIRROR_CASES = [
    pytest.param(_running_active_slot, True, id="running_active_slot"),
    pytest.param(_running_inflight, True, id="running_inflight_riders"),
    pytest.param(_not_running, False, id="control_no_running_turn"),
]


@pytest.mark.parametrize("make_turn_run,is_running", MIRROR_CASES)
async def test_diagnostic_agrees_with_the_real_choice_about_a_running_turn(
        tmp_path, make_turn_run, is_running):
    boot, runtime = _boot_runtime_for_mirror(tmp_path)
    mgr = TurbohaulManager(boot, runtime)
    mgr._fastlane_table = lambda: TABLE
    try:
        # Two ACTIVE residents, neither idle-evictable. The first is better ranked and
        # is never designated; the second is the worst ranked and gets designated.
        other = _active_resident("model-better", BETTER_IP)
        victim = _active_resident("model-victim", VICTIM_IP, grace_started=True)
        mgr._residents["model-better"] = other
        mgr._residents["model-victim"] = victim
        make_turn_run(victim)

        claimant = Slot.new("model-claimant", prompt="hi", thread_id="thread-claimant",
                            client_meta={"ip": CLAIMANT_IP})
        claimant.fastlane = match_fastlane(TABLE, CLAIMANT_IP, {"ip": CLAIMANT_IP})
        async with mgr._registry_lock:
            mgr._register_fastlane_claim_locked(claimant, "mirror_agreement_test")

        # The shape is the one the fallback arm cares about.
        async with mgr._registry_lock:
            assert mgr._is_designated_unload_target_locked(victim) is True, "victim not designated"
            assert mgr._is_designated_unload_target_locked(other) is False
            assert victim.state is ResidentState.ACTIVE
            assert victim.grace is not None and victim.grace._started_at is not None
            assert victim.in_grace_loop is False
            assert not any(r.state is ResidentState.IDLE_EVICTABLE
                           for r in mgr._model_residents()), "an idle-evictable resident exists"
            assert (victim.active_slot is not None or bool(victim.inflight)) is is_running

            real = mgr._lru_idle_unloadable()
            mirror = mgr._starvation_reason()

        by_tag = dict(entry.split(":", 1) for entry in mirror["per_resident"].split(","))

        if is_running:
            # Real choice: a resident with a running turn is never returned.
            assert real is None, (
                f"the real choice returned {getattr(real, 'model_tag', None)!r}, a resident "
                f"with a running turn")
            # The diagnostic must say the same thing: nobody is eligible, and the running
            # resident is not listed as eligible.
            assert mirror["eligible"] == 0, (
                f"the diagnostic counts {mirror['eligible']} eligible resident(s) while the "
                f"real choice returns none; per_resident={mirror['per_resident']}")
            assert by_tag["model-victim"] != "eligible", (
                f"the diagnostic lists the running resident as eligible: {mirror['per_resident']}")
            assert by_tag["model-better"] == "state_not_idle", mirror["per_resident"]
        else:
            # Control: same shape with no running turn. Both name it as the victim.
            assert real is victim, (
                f"control: the real choice should return the designated victim, got "
                f"{getattr(real, 'model_tag', None)!r}")
            assert mirror["eligible"] == 1, mirror
            assert by_tag["model-victim"] == "eligible", mirror["per_resident"]
            assert by_tag["model-better"] == "state_not_idle", mirror["per_resident"]
    finally:
        await mgr.shutdown()


# ---------------------------------------------------------------------------
# 2. a driver exit that owns the teardown sets the dispatch wake by itself
# ---------------------------------------------------------------------------

TAG_A = "model-a"
CARD_MIB = 24000
FOOTPRINT_MIB = 20000
LONG_BACKOFF_S = 600.0     # the retry polls must never fire inside this test
CURATOR = {"ip": BETTER_IP, "is_curator": True, "is_sub_agent": True}
ENGINE_ERROR = "engine connection lost"
WAIT_S = 15.0              # generous bound for state to be reached; never a pass condition
LOADED = (ResidentState.ACTIVE, ResidentState.GRACE, ResidentState.IDLE_EVICTABLE)


@dataclass
class Box:
    mgr: TurbohaulManager
    outcomes: dict = field(default_factory=dict)   # label -> future the fake turn waits on
    handles: dict = field(default_factory=dict)    # label -> engine handle that served it
    events: list = field(default_factory=list)     # (monotonic, kind, name, extra)
    tasks: dict = field(default_factory=dict)

    def stamp(self, kind, name="", extra=None):
        self.events.append((time.monotonic(), kind, name, extra))

    def seen(self, kind, name=None):
        return any(k == kind and (name is None or n == name) for _t, k, n, _x in self.events)

    def rel(self):
        t0 = self.events[0][0] if self.events else 0.0
        return [(round(t - t0, 3), k, n, x) for t, k, n, x in self.events]

    def submit(self, label, hold=False):
        if hold:
            self.outcomes[label] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(asyncio.wait_for(
            self.mgr.submit_and_wait(TAG_A, label, thread_id="thread-a",
                                     client_meta=dict(CURATOR)),
            timeout=60))
        self.tasks[label] = task
        return task

    def kill_engine(self, label):
        """The engine process is gone: the handle reports not alive and the call fails."""
        self.handles[label].proc.poll.return_value = 1
        self.stamp("engine_dead", label)
        self.outcomes[label].set_exception(ConnectionError(ENGINE_ERROR))

    async def until(self, predicate, what, timeout=WAIT_S):
        t0 = time.monotonic()
        while not predicate():
            if time.monotonic() - t0 > timeout:
                raise AssertionError(
                    f"never happened within {timeout}s: {what}; events={self.rel()}")
            await asyncio.sleep(0.005)

    async def stop_dispatcher(self):
        """Stop the dispatch loop so nothing can wait on, clear or consume the wake event."""
        task = self.mgr._worker_task
        # A cancel that lands in the same instant as a wake can be swallowed by the loop's
        # own bounded wait, so the cancel is repeated (a bounded number of times) until the
        # task is really done.
        for _ in range(20):
            task.cancel()
            done, _pending = await asyncio.wait({task}, timeout=0.5)
            if done:
                break
        assert task.done(), "the dispatch loop did not stop"
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()

    async def park_claimant(self):
        """A stand-in claimant parked on the make-room signal. Returns the woken latch."""
        mgr = self.mgr
        parked, woken = asyncio.Event(), asyncio.Event()

        async def claimant():
            async with mgr._make_room_signal:
                parked.set()
                await mgr._make_room_signal.wait()
            woken.set()

        task = asyncio.create_task(claimant())
        self.tasks["parked_claimant"] = task
        await asyncio.wait_for(parked.wait(), 10)
        await self.until(lambda: len(mgr._make_room_signal._waiters) > 0,
                         "the claimant to be waiting on the make-room signal")
        return woken


def _caller():
    f = sys._getframe(2)
    return f.f_code.co_name


@contextlib.asynccontextmanager
async def box(tmp_path, monkeypatch, *, grace_s):
    rules = [FastLaneRule(address=ip, tag_ranks=FastLaneTagRanks(main=1, curator=3, unclassified=5))
             for ip in (CLAIMANT_IP, BETTER_IP)]
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, max_parallel_sidecars=2, grace_seconds=grace_s,
        max_grace_extensions=5, idle_hot_load_seconds=120)
    size = FOOTPRINT_MIB * 1024 * 1024
    (boot.storage.manifests_path / f"{TAG_A}.yaml").write_text(yaml.safe_dump({
        "model_tag": TAG_A, "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": size, "context_size": 2048,
        "expected_vram_bytes": size,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }))
    # No retry poll may fire inside the test.
    monkeypatch.setattr("turbohaul.manager._DISPATCH_DEFER_BACKOFF_S", LONG_BACKOFF_S)
    monkeypatch.setattr("turbohaul.manager._VRAM_DEFER_BACKOFF_S", LONG_BACKOFF_S)
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
        return [CARD_MIB - FOOTPRINT_MIB * len(loaded)]

    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", free_vram)
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", free_vram)
    mgr = TurbohaulManager(boot, runtime, spawn_fn=spawn_rec, health_fn=health,
                           sigterm_fn=sigterm_rec, vram_fn=vram, complete_fn=complete_rec)
    b = Box(mgr=mgr)
    holder["mgr"], holder["box"] = mgr, b
    b.stamp("start")

    orig_set = mgr._dispatch_wake.set
    orig_notify = mgr._make_room_signal.notify_all

    def set_spy():
        b.stamp("wake_set", _caller())
        return orig_set()

    def notify_spy(*a, **k):
        b.stamp("wake_notify", _caller())
        return orig_notify(*a, **k)

    mgr._dispatch_wake.set = set_spy
    mgr._make_room_signal.notify_all = notify_spy

    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        yield b
    finally:
        # Release every held turn first, then stop the loop, then cancel the worker; all bounded.
        for fut in b.outcomes.values():
            if not fut.done():
                fut.set_result(None)
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


async def _park_a_idle(b):
    """Model A serves one turn, its short grace window lapses and it parks idle."""
    b.submit("A1")
    await b.until(lambda: b.seen("turn_end", "A1"), "A's first turn to end")
    a = resident_for(b.mgr, TAG_A)
    await b.until(lambda: a.state is ResidentState.IDLE_EVICTABLE and not a.in_grace_loop
                  and a.grace is not None and a.grace._started_at is not None,
                  "A to park idle after its grace window")
    assert a.driver_task is not None and not a.driver_task.done(), "A's driver is not running"
    return a


async def _run_a_turn_held(b):
    b.submit("A1", hold=True)
    await b.until(lambda: b.seen("turn_start", "A1"), "A's turn to start")
    a = resident_for(b.mgr, TAG_A)
    assert a.state is ResidentState.ACTIVE and a.active_slot is not None, "A is not mid-turn"
    assert a.driver_task is not None and not a.driver_task.done(), "A's driver is not running"
    return a


EXIT_KINDS = pytest.mark.parametrize("exit_kind", ["engine_death", "driver_cancelled_while_parked"])


@EXIT_KINDS
async def test_a_driver_exit_sets_the_dispatch_wake_and_wakes_a_parked_claimant(
        tmp_path, monkeypatch, exit_kind):
    async with box(tmp_path, monkeypatch, grace_s=1) as b:
        mgr = b.mgr
        if exit_kind == "engine_death":
            a = await _run_a_turn_held(b)
        else:
            a = await _park_a_idle(b)

        # Nothing may be able to clear or consume the wake event from here on.
        await b.stop_dispatcher()
        assert not a.driver_task.done(), "stopping the dispatcher also stopped A's driver"
        woken = await b.park_claimant()

        mgr._dispatch_wake.clear()
        assert not mgr._dispatch_wake.is_set()
        b.stamp("before_exit")
        n_before = len(b.events)

        if exit_kind == "engine_death":
            b.kill_engine("A1")
            await b.until(lambda: b.tasks["A1"].done(), "the dead turn's request to end")
            exc = b.tasks["A1"].exception()
            assert isinstance(exc, ConnectionError) and ENGINE_ERROR in str(exc), (
                f"the dead turn did not end with the engine's error: {exc!r}")
        else:
            a.driver_task.cancel()
        await b.until(lambda: a.driver_task.done(), "A's driver to exit")
        b.stamp("driver_exited")

        # The wake is set by the exit itself: the dispatcher is stopped, so nothing else
        # could have set it after the explicit clear above.
        assert mgr._dispatch_wake.is_set(), (
            f"the driver exited and the dispatch wake is still clear; "
            f"events={b.rel()[n_before - 1:]}")
        driver_sets = [e for e in b.events[n_before:]
                       if e[1] == "wake_set" and e[2] == "_drive_resident"]
        assert driver_sets, f"no wake was set from the driver's exit; events={b.rel()}"

        # The claimant parked on the make-room signal was woken.
        try:
            await asyncio.wait_for(woken.wait(), 10)
        except TimeoutError:
            raise AssertionError(
                f"the parked claimant was never woken by the driver's exit; events={b.rel()}")
        assert any(e[1] == "wake_notify" and e[2] == "_drive_resident"
                   for e in b.events[n_before:]), f"events={b.rel()}"


async def test_control_an_exit_owned_by_the_detached_teardown_leaves_the_wake_clear(
        tmp_path, monkeypatch):
    """The detached unload already owns the teardown (it claimed it first), so the
    driver's exit takes no ownership and must not set the wake or wake anyone."""
    async with box(tmp_path, monkeypatch, grace_s=1) as b:
        mgr = b.mgr
        a = await _park_a_idle(b)
        await b.stop_dispatcher()

        # A real unload of the parked resident: the detached teardown claims the teardown.
        async with mgr._registry_lock:
            mgr._begin_unload_locked(a)
        await b.until(lambda: a.torn_down and not mgr._bg_tasks,
                      "the detached teardown to finish")
        assert not a.driver_task.done(), "the driver exited on its own; the control needs it parked"

        woken = await b.park_claimant()
        mgr._dispatch_wake.clear()
        b.stamp("before_exit")
        n_before = len(b.events)

        a.driver_task.cancel()
        await b.until(lambda: a.driver_task.done(), "A's driver to exit")

        assert not mgr._dispatch_wake.is_set(), (
            f"an exit that did not own the teardown set the dispatch wake; "
            f"events={b.rel()[n_before - 1:]}")
        assert not [e for e in b.events[n_before:] if e[1] in ("wake_set", "wake_notify")], (
            f"an exit that did not own the teardown signalled a wake; events={b.rel()}")
        assert not woken.is_set(), "a parked claimant was woken by an exit that owned nothing"
