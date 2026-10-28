"""Race, bookkeeping and wording behaviour of the co-residence re-check.

Same method as the main wiring file: the real ``TurbohaulManager._dispatch_loop`` is stepped one
tick at a time (a gate in front of ``queue.pop_next``), only the outside world is faked (card
picker, VRAM gate, placement resolver, cap answer, spawn), and ages are made by moving the
manager's ``_co_residence_now`` seam. Log lines are read from the ``turbohaul.co_residence``
loggers.

  R1  an engine that is not ACTIVE serves its own inbox: its parked request is not demand
  R2  a Fast Lane claim parked on the engine is un-parked, with its time-to-live reset, by the take
  R3  the re-routed request runs under the reserving marker
  R4  a re-check that ends without a new engine is remembered (no probe storm after a lost race)
  R5  a request whose client left between the probe and the take is not routed
  R6  the arrival decline line says "cap" when the tag is at its cap, "box_full" when the box is
  R7  an arrival spawn clears the tag's no-room memo
  R8  an engine that vanishes while the re-check waits for the registry lock is not probed
"""
from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import MagicMock

import pytest

from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

TAG = "tag-a"
FOREIGN = "tag-b"
FREE_CARD = 1
NEED_MIB = 4000
STEP_TIMEOUT_S = 3.0
DECLINED = "CO_RESIDENCE_DECLINED"


def _boot_runtime(tmp_path, cap):
    root = tmp_path / "state"
    root.mkdir()
    for d in ("blobs", "manifests", "import-staging"):
        (root / d).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs",
            manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging",
            state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=cap, grace_seconds=3600,
            max_grace_extensions=50, idle_hot_load_seconds=3600,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _mocks():
    def spawn(binary, gguf, port, model_tag, argv, **kw):
        proc = MagicMock()
        proc.pid = 93_000
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def health(*a, **k):
        return True

    async def sigterm(handle, **k):
        return True, "sigterm-clean"

    async def vram(**k):
        return True, 100

    async def complete(slot, handle):
        return {"ok": True}

    return dict(spawn_fn=spawn, health_fn=health, sigterm_fn=sigterm, vram_fn=vram, complete_fn=complete)


def _resident(mgr, tag, *, state=ResidentState.ACTIVE, card=0, port=59701):
    r = Resident(
        model_tag=tag, resident_key=tag, port=port, state=state, main_gpu=card, split_mode="none",
        reserved_need_mib=NEED_MIB,
        grace=GraceTimer(grace_seconds=3600, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=3600),
        last_active_monotonic=time.monotonic(),
    )
    r.inbox = asyncio.Queue()
    mgr._residents[tag] = r
    return r


class Box:
    """What the faked outside world was asked, the re-check clock, and the loop stepper."""

    def __init__(self, mgr):
        self.mgr = mgr
        self.clock = [time.monotonic()]
        self.eff_cap = 2
        self.on_cap = None            # called from the cap answer (runs in a worker thread)
        self.reserve_returns_none = False
        self.reserves = []            # (slot, main_gpu_override, split_mode_override)
        self.reserving_seen = []      # mgr._reserving_model_tag at every reserve
        self.caps = []
        self.picks = []
        self.admits = []
        self.admit_result = True
        self.sem = asyncio.Semaphore(0)
        self.parked = 0
        self.task = None

    def probes(self):
        return (len(self.caps), len(self.picks), len(self.admits))

    def now(self):
        return self.clock[0]

    def launch(self, monkeypatch):
        mgr = self.mgr
        monkeypatch.setattr("turbohaul.manager._DISPATCH_DEFER_BACKOFF_S", 0.005)
        real_pop = mgr.queue.pop_next

        async def gated_pop(**kw):
            self.parked += 1
            await self.sem.acquire()
            return await real_pop(**kw)

        monkeypatch.setattr(mgr.queue, "pop_next", gated_pop)
        self.task = asyncio.create_task(mgr._dispatch_loop())

    async def start_loop(self, monkeypatch):
        """Run the loop up to the point where the FIRST iteration's re-check has finished."""
        self.launch(monkeypatch)
        await self._until_parked(1)

    async def _until_parked(self, target):
        deadline = time.monotonic() + STEP_TIMEOUT_S
        while self.parked < target:
            if self.task.done():
                exc = None if self.task.cancelled() else self.task.exception()
                raise AssertionError(f"the dispatch loop ended before the step finished: {exc!r}")
            assert time.monotonic() < deadline, "the dispatch loop did not reach its next tick in time"
            await asyncio.sleep(0.002)

    async def step(self, n=1):
        for _ in range(n):
            target = self.parked + 1
            self.sem.release()
            await self._until_parked(target)

    async def stop_loop(self):
        if self.task is None:
            return
        self.mgr._stop_event.set()
        self.sem.release()
        done, _ = await asyncio.wait({self.task}, timeout=STEP_TIMEOUT_S)
        if not done:
            self.task.cancel()
        elif not self.task.cancelled():
            self.task.exception()     # a loop that died is reported by the step that saw it


async def _world(tmp_path, monkeypatch, *, cap=2):
    boot, runtime = _boot_runtime(tmp_path, cap)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    box = Box(mgr)

    def cap_placement(tag):
        box.caps.append(tag)
        if box.on_cap is not None:
            box.on_cap()
        return box.eff_cap, True

    def resolve(tag):
        return (NEED_MIB, 1, 0, "none", 0, True, False)

    def pick(tag, need):
        box.picks.append((tag, need))
        return FREE_CARD, "none"

    def admits(need, parallel, g, sp, *a, **k):
        box.admits.append((g, sp))
        return box.admit_result

    async def reserve(slot, main_gpu_override=None, split_mode_override=None):
        box.reserves.append((slot, main_gpu_override, split_mode_override))
        box.reserving_seen.append(mgr._reserving_model_tag)
        if box.reserve_returns_none:
            return None
        n = len(mgr._residents)
        r = Resident(
            model_tag=slot.model_tag, resident_key=f"{slot.model_tag}#{n}", port=59800 + n,
            state=ResidentState.RESERVED_LOADING,
            main_gpu=main_gpu_override if main_gpu_override is not None else 0,
            split_mode=split_mode_override or "none", reserved_need_mib=NEED_MIB, instance_idx=n,
            last_active_monotonic=time.monotonic(),
        )
        r.inbox = asyncio.Queue()
        r.inbox.put_nowait(slot)
        mgr._residents[r.resident_key] = r
        return r

    mgr._route_cap_and_placement = cap_placement
    mgr._resolve_placement_locked = resolve
    mgr._pick_card_for_new_instance = pick
    mgr._vram_admits_locked = admits
    mgr._reserve_and_start_locked = reserve
    monkeypatch.setattr(mgr, "_co_residence_now", box.now, raising=False)
    return mgr, box


def _slot(box, tag, thread, age):
    slot = Slot.new(tag, prompt="p", thread_id=thread)
    slot.created_at = box.clock[0] - age
    return slot


def _park(box, r, thread, age):
    slot = _slot(box, r.model_tag, thread, age)
    slot.state = SlotState.STAGED
    r.inbox.put_nowait(slot)
    return slot


def _inbox(r):
    return list(r.inbox._queue)


def _ids(slots):
    return [s.slot_id for s in slots]


def _lines(caplog, prefix):
    return [rec.getMessage() for rec in caplog.records
            if rec.name.startswith("turbohaul.co_residence") and rec.getMessage().startswith(prefix)]


def _assert_starting_scene(mgr, box, r, *, state=ResidentState.ACTIVE):
    """The starting scene before any tick: one live engine of the tag on card 0, a free sidecar
    slot, a free fitting card, and same-model demand parked on that engine."""
    assert mgr.runtime.queue.max_parallel_sidecars == 2
    assert len(mgr._model_residents()) == 1
    assert len(mgr._instances_for(TAG)) == 1 and mgr._instances_for(TAG)[0] is r
    assert r.state is state and r.main_gpu == 0 and box.eff_cap == 2
    assert box.admit_result is True or box.admit_result is False
    assert _inbox(r), "no same-model demand is waiting"


# --------------------------------------------------------------------------------------
# R1: only an ACTIVE engine's inbox is demand
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("state", [ResidentState.IDLE_EVICTABLE, ResidentState.RESERVED_LOADING])
async def test_r1_an_engine_that_is_not_active_serves_its_own_inbox(tmp_path, monkeypatch, state):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG, state=state)
    p = _park(box, r, "t-p", 30.0)
    try:
        _assert_starting_scene(mgr, box, r, state=state)
        await box.start_loop(monkeypatch)
        await box.step(2)
        assert box.reserves == [] and box.probes() == (0, 0, 0), "a not-ACTIVE engine's inbox is not demand"
        assert _ids(_inbox(r)) == [p.slot_id]
        r.state = ResidentState.ACTIVE                      # control: the same request on a busy engine
        box.clock[0] += 11.0
        await box.step()
        assert len(box.reserves) == 1 and box.reserves[0][0] is p, "the busy-engine control never spawned"
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# R2 / R3: what the take leaves behind
# --------------------------------------------------------------------------------------

async def test_r2_the_take_unparks_a_fast_lane_claim_and_resets_its_time_to_live(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG)
    p = _park(box, r, "t-p", 30.0)
    p.fastlane = FastLaneMatch(rule_index=0, raw_address="9.9.9.9", label="", effective_tag="main", rank=1)
    mgr._park_fastlane_claim_locked(p, r)
    claim = mgr._fastlane_claims[mgr._fastlane_claim_key(p)]
    claim["ttl_deadline_monotonic"] = 1.0
    try:
        _assert_starting_scene(mgr, box, r)
        assert claim["slot"] is p and claim["parked_on"] == r.resident_key, "the claim is parked on the engine"
        await box.start_loop(monkeypatch)
        assert len(box.reserves) == 1 and box.reserves[0][0] is p, "the parked demand never got its engine"
        assert mgr._fastlane_claims.get(mgr._fastlane_claim_key(p)) is claim
        assert "parked_on" not in claim, "the slot left the inbox, so its claim is no longer parked there"
        assert claim["ttl_deadline_monotonic"] > time.monotonic() + 1000.0, "the claim's time-to-live was not reset"
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_r3_the_rerouted_request_runs_under_the_reserving_marker(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG)
    p = _park(box, r, "t-p", 30.0)
    try:
        _assert_starting_scene(mgr, box, r)
        assert mgr._reserving_model_tag is None
        await box.start_loop(monkeypatch)
        assert len(box.reserves) == 1 and box.reserves[0][0] is p, "the parked demand never got its engine"
        assert box.reserving_seen == [TAG], "the reserve ran without the reserving marker"
        assert mgr._reserving_model_tag is None, "the marker must be cleared again"
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# R4: a lost race is remembered
# --------------------------------------------------------------------------------------

async def test_r4_a_recheck_that_ends_without_a_new_engine_is_remembered(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path, monkeypatch)
    box.reserve_returns_none = True                        # the spawn is deferred: no new engine appears
    r = _resident(mgr, TAG)
    older = _park(box, r, "t-older", 40.0)
    newer = _park(box, r, "t-newer", 30.0)
    t0 = box.clock[0]
    try:
        _assert_starting_scene(mgr, box, r)
        await box.start_loop(monkeypatch)
        assert len(box.reserves) == 1 and box.reserves[0][0] is older, "the first attempt never reserved"
        assert len(mgr._instances_for(TAG)) == 1 and _ids(_inbox(r)) == [newer.slot_id]
        after_first = box.probes()
        box.clock[0] = t0 + 11.0                           # debounce open, no-room memo still live
        await box.step()
        assert box.probes() == after_first, "a re-check that produced no engine was probed again at once"
        box.clock[0] = t0 + 72.0                           # control: after the memo window it asks again
        await box.step()
        assert len(box.reserves) == 2 and box.reserves[1][0] is newer
        assert box.probes() != after_first
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# R5: the client left between the probe and the take
# --------------------------------------------------------------------------------------

async def test_r5_a_request_whose_client_left_is_not_routed(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG)
    p = _park(box, r, "t-p", 30.0)
    hooked = []

    def client_leaves():
        hooked.append(True)
        p.is_evicted = True                                 # between the scan and the take

    box.on_cap = client_leaves
    try:
        _assert_starting_scene(mgr, box, r)
        assert not p.is_evicted and mgr._unload_count == 0
        await box.start_loop(monkeypatch)
        assert hooked == [True], "the client never left, so this proved nothing"
        assert box.reserves == [], "an engine was started for a client that is gone"
        assert mgr._unload_count == 1, "the departed request was not handled as an unloaded slot"
        assert _inbox(r) == []
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# R6: the arrival decline line names the right cause
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("why", ["cap", "box_full"])
async def test_r6_the_arrival_decline_names_cap_or_box_full(tmp_path, monkeypatch, caplog, why):
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG)
    box.eff_cap = 1 if why == "cap" else 2
    if why == "box_full":
        _resident(mgr, FOREIGN, card=FREE_CARD, port=59702)
    slot = _slot(box, TAG, "t-arrival", 0.0)
    slot.state = SlotState.STAGED
    residents = 1 if why == "cap" else 2
    try:
        assert len(mgr._instances_for(TAG)) == 1 and len(mgr._model_residents()) == residents
        with caplog.at_level(logging.INFO, logger="turbohaul.co_residence"):
            await mgr._route_or_reserve(slot)
            assert box.picks == [] and box.reserves == [], "a tag at its cap or a full box never probes"
            assert _ids(_inbox(r)) == [slot.slot_id], "the arrival waits on the tag's engine"
            assert _lines(caplog, DECLINED) == [
                f"{DECLINED} via=arrival tag={TAG} reason={why} eff_cap={box.eff_cap} instances=1 "
                f"residents={residents} max_sidecars=2 need_mib=- card=- split=-"
            ]
    finally:
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# R7: an arrival spawn clears the no-room memo
# --------------------------------------------------------------------------------------

async def test_r7_an_arrival_spawn_clears_the_no_room_memo(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path, monkeypatch)
    box.admit_result = False
    r = _resident(mgr, TAG)
    p = _park(box, r, "t-p", 30.0)
    t0 = box.clock[0]
    try:
        _assert_starting_scene(mgr, box, r)
        await box.start_loop(monkeypatch)
        assert len(box.picks) == 1 and box.reserves == [], "the declined re-check never set the memo"
        box.admit_result = True
        box.clock[0] = t0 + 1.0
        arrival = _slot(box, TAG, "t-arrival", 0.0)
        await mgr.queue.enqueue(arrival)
        await box.step()                                     # the dispatcher spawns the second engine
        assert len(box.reserves) == 1 and box.reserves[0][0] is arrival
        second = [x for x in mgr._instances_for(TAG) if x is not r]
        assert len(second) == 1
        del mgr._residents[second[0].resident_key]           # that engine goes away again
        probes_before = len(box.picks)
        box.clock[0] = t0 + 12.0                             # debounce open, inside the 60 s memo window
        await box.step()
        assert len(box.picks) > probes_before, "the arrival spawn left the no-room memo standing"
        assert len(box.reserves) == 2 and box.reserves[1][0] is p
    finally:
        await box.stop_loop()
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# R8: the engine vanished while the re-check waited for the registry lock
# --------------------------------------------------------------------------------------

async def _blocked_at_phase_a(tmp_path, monkeypatch, *, remove_engine):
    """Park the re-check at the registry lock after it chose its candidate, then optionally remove
    the engine, then let it go. The hold is taken right after the first grace read, which is the
    last thing the re-check does before it picks."""
    mgr, box = await _world(tmp_path, monkeypatch)
    r = _resident(mgr, TAG)
    p = _park(box, r, "t-p", 30.0)
    state = {"calls": 0, "held": False}
    real = mgr._grace_active_exclusions

    async def grace_then_hold():
        out = await real()
        state["calls"] += 1
        if state["calls"] == 1:
            await mgr._registry_lock.acquire()
            state["held"] = True
        return out

    mgr._grace_active_exclusions = grace_then_hold
    _assert_starting_scene(mgr, box, r)
    box.launch(monkeypatch)
    try:
        deadline = time.monotonic() + STEP_TIMEOUT_S
        while not state["held"]:
            assert time.monotonic() < deadline, "the lock was never taken by the grace read"
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.05)
        assert box.parked == 0, "the re-check was not waiting at the registry lock"
        if remove_engine:
            del mgr._residents[TAG]
        mgr._registry_lock.release()
        await box._until_parked(1)
    finally:
        if mgr._registry_lock.locked():
            mgr._registry_lock.release()
    return mgr, box, p


async def test_r8_an_engine_that_vanished_at_the_lock_is_not_probed(tmp_path, monkeypatch):
    (tmp_path / "c1").mkdir()
    (tmp_path / "c2").mkdir()
    worlds = []
    try:
        mgr1, box1, p1 = await _blocked_at_phase_a(tmp_path / "c1", monkeypatch, remove_engine=False)
        worlds.append((mgr1, box1))
        assert len(box1.reserves) == 1 and box1.reserves[0][0] is p1, "the control never spawned"
        mgr2, box2, _p2 = await _blocked_at_phase_a(tmp_path / "c2", monkeypatch, remove_engine=True)
        worlds.append((mgr2, box2))
        assert mgr2._instances_for(TAG) == []
        assert box2.caps == [] and box2.picks == [] and box2.admits == [], "an engine that is gone was probed"
        assert box2.reserves == [], "a cold engine was started from the re-check"
    finally:
        for mgr, box in worlds:
            await box.stop_loop()
            await mgr.shutdown()
