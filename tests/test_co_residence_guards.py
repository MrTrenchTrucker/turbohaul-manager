"""Guards for the second-engine routing path: what must stay as it is.

Every test here drives the real TurbohaulManager (real ``_route_or_reserve``, real
``_dispatch_loop``, real queue). Only the outside world is faked: the card/VRAM probes
(``_pick_card_for_new_instance``, ``_vram_admits_locked``), the placement resolver, the
cap/placement answer, and the engine spawn (``_reserve_and_start_locked``).

The assertions read STATE only (inbox contents, registry size, staging, call counts of the
faked methods). They do not read log lines. It pins:

  G1  one idle live engine, non-parallel manifest: the legacy hit, nothing reserved or probed
  G2  one live engine, one-card-per-engine manifest, free fitting card: exactly ONE second engine
  G3  same, but no card / VRAM refuses: nothing reserved, the slot parks in the existing inbox
  G4  cap 1: a request for a tag with one live engine probes nothing and reserves nothing
  G5  a fresh same-thread follow-up inside a live grace window is never taken by the dispatcher
  G6  a cold tag still starts its first engine through the same path
  G7  an ordinary fresh arrival popped by the dispatcher still spawns the second engine
  G8  an ordinary fresh arrival that cannot spawn still parks in the busy engine's inbox
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
import yaml

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

TAG = "tag-a"
COLD_TAG = "tag-cold"
FREE_CARD = 1
NEED_MIB = 4000
STEP_TIMEOUT_S = 3.0


# --------------------------------------------------------------------------------------
# fixture: a real manager, hand-built residents, faked probes
# --------------------------------------------------------------------------------------

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


def _seed_manifest(boot, tag, *, auto_place):
    """A manifest on disk. ``auto_place`` + split_mode none = one card per engine."""
    body = {
        "model_tag": tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": NEED_MIB * 1024 * 1024,
        "context_size": 2048,
        "expected_vram_bytes": NEED_MIB * 1024 * 1024,
        "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
    }
    if auto_place:
        body["auto_place"] = True
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump(body))


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


def _resident(mgr, tag, *, state, port=59701, grace_pair=None):
    r = Resident(
        model_tag=tag, resident_key=tag, port=port, state=state, main_gpu=0, split_mode="none",
        reserved_need_mib=NEED_MIB,
        grace=GraceTimer(grace_seconds=3600, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=3600),
        last_active_monotonic=time.monotonic(),
    )
    r.inbox = asyncio.Queue()
    if grace_pair is not None:
        r.grace.start(*grace_pair)
    mgr._residents[tag] = r
    return r


class Box:
    """Counts what the faked outside world was asked, and holds the loop stepper."""

    def __init__(self, mgr):
        self.mgr = mgr
        self.reserves = []        # (slot, main_gpu_override, split_mode_override)
        self.picks = []           # (tag, need)
        self.admits = []          # (card, split)
        self.pick_result = (FREE_CARD, "none")
        self.admit_result = True
        self.sem = asyncio.Semaphore(0)
        self.parked = 0
        self.task = None

    # -- the dispatcher, one iteration at a time ---------------------------------------
    async def start_loop(self, monkeypatch):
        mgr = self.mgr
        monkeypatch.setattr("turbohaul.manager._DISPATCH_DEFER_BACKOFF_S", 0.005)
        real_pop = mgr.queue.pop_next

        async def gated_pop(**kw):
            self.parked += 1
            await self.sem.acquire()
            return await real_pop(**kw)

        monkeypatch.setattr(mgr.queue, "pop_next", gated_pop)
        self.task = asyncio.create_task(mgr._dispatch_loop())
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


def _install_fakes(mgr, box, *, eff_cap=2, wants_one_card=True):
    def cap_placement(tag):
        return eff_cap, wants_one_card

    def resolve(tag):
        return (NEED_MIB, 1, 0, "none", 0, True, False)

    def pick(tag, need):
        box.picks.append((tag, need))
        return box.pick_result

    def admits(need, parallel, g, sp, *a, **k):
        box.admits.append((g, sp))
        return box.admit_result

    async def reserve(slot, main_gpu_override=None, split_mode_override=None):
        box.reserves.append((slot, main_gpu_override, split_mode_override))
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


def _inbox_items(r):
    return list(r.inbox._queue)


def _new_slot(tag, thread):
    slot = Slot.new(tag, prompt="p", thread_id=thread)
    slot.state = SlotState.STAGED
    return slot


async def _world(tmp_path, *, cap=2, manifest=None, fakes=True, **fake_kw):
    boot, runtime = _boot_runtime(tmp_path, cap)
    if manifest is not None:
        _seed_manifest(boot, TAG, auto_place=manifest)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    box = Box(mgr)
    if fakes:
        _install_fakes(mgr, box, **fake_kw)
    return mgr, box


# --------------------------------------------------------------------------------------
# G1 / G4: the legacy hit and the cap-1 box
# --------------------------------------------------------------------------------------

async def test_g1_one_idle_engine_non_parallel_manifest_takes_the_legacy_hit(tmp_path):
    """Cap 2, a manifest that is NOT one-card-per-engine: the arrival goes to the live engine."""
    mgr, box = await _world(tmp_path, cap=2, manifest=False, fakes=False)
    box.reserves, box.picks, box.admits = [], [], []

    async def no_reserve(slot, **kw):
        box.reserves.append(slot)

    mgr._reserve_and_start_locked = no_reserve
    mgr._pick_card_for_new_instance = lambda tag, need: box.picks.append(tag) or (FREE_CARD, "none")
    mgr._vram_admits_locked = lambda *a, **k: box.admits.append(a) or True
    r = _resident(mgr, TAG, state=ResidentState.IDLE_EVICTABLE)
    slot = _new_slot(TAG, "t-arrival")
    try:
        assert len(mgr._model_residents()) == 1 and not _inbox_items(r)  # the shape exists
        await mgr._route_or_reserve(slot)
        assert [s for s in _inbox_items(r)] == [slot], "the slot must land in the live engine's inbox"
        assert box.reserves == [], "nothing may be reserved on the legacy hit"
        assert box.picks == [] and box.admits == [], "the legacy hit probes no card"
        assert len(mgr._model_residents()) == 1
        assert r.state is ResidentState.ACTIVE, "the hit wakes the idle engine"
    finally:
        await mgr.shutdown()


@pytest.mark.parametrize("manifest", [False, True], ids=["legacy_manifest", "one_card_manifest"])
async def test_g4_cap_one_arrival_with_a_live_engine_probes_nothing(tmp_path, manifest):
    """Cap 1 (real cap and placement answers, real manifest): hit or park, never a probe or a reserve."""
    mgr, box = await _world(tmp_path, cap=1, manifest=manifest, fakes=False)

    async def no_reserve(slot, **kw):
        box.reserves.append(slot)

    mgr._reserve_and_start_locked = no_reserve
    mgr._pick_card_for_new_instance = lambda tag, need: box.picks.append(tag) or (FREE_CARD, "none")
    mgr._vram_admits_locked = lambda *a, **k: box.admits.append(a) or True
    r = _resident(mgr, TAG, state=ResidentState.ACTIVE)
    slot = _new_slot(TAG, "t-arrival")
    try:
        assert mgr.runtime.queue.max_parallel_sidecars == 1 and len(mgr._model_residents()) == 1
        await mgr._route_or_reserve(slot)
        assert _inbox_items(r) == [slot]
        assert box.reserves == []
        assert box.picks == [] and box.admits == [], "a cap-1 box must never probe a card for a live tag"
    finally:
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# G2 / G3 / G6: the arrival path through step 2
# --------------------------------------------------------------------------------------

async def test_g2_one_live_engine_free_card_reserves_exactly_one_second_engine(tmp_path):
    mgr, box = await _world(tmp_path)
    r = _resident(mgr, TAG, state=ResidentState.ACTIVE)
    slot = _new_slot(TAG, "t-arrival")
    try:
        assert len(mgr._instances_for(TAG)) == 1 and box.pick_result == (FREE_CARD, "none")
        await mgr._route_or_reserve(slot)
        assert len(box.reserves) == 1, "exactly one second engine"
        got_slot, card, split = box.reserves[0]
        assert got_slot is slot and card == FREE_CARD and split == "none"
        assert len(mgr._instances_for(TAG)) == 2
        assert _inbox_items(r) == [], "the slot must not also be parked on the first engine"
    finally:
        await mgr.shutdown()


@pytest.mark.parametrize("why", ["no_card", "vram_refused"])
async def test_g3_no_room_parks_the_slot_in_the_existing_inbox(tmp_path, why):
    mgr, box = await _world(tmp_path)
    if why == "no_card":
        box.pick_result = (None, "none")
    else:
        box.admit_result = False
    r = _resident(mgr, TAG, state=ResidentState.ACTIVE)
    slot = _new_slot(TAG, "t-arrival")
    try:
        await mgr._route_or_reserve(slot)
        assert box.reserves == []
        assert _inbox_items(r) == [slot], "step 3 parks the slot on the existing engine"
        assert len(mgr._instances_for(TAG)) == 1
        assert len(box.picks) == 1 and len(box.admits) == (0 if why == "no_card" else 1)
    finally:
        await mgr.shutdown()


async def test_g6_a_cold_tag_still_starts_its_first_engine_through_step_two(tmp_path):
    mgr, box = await _world(tmp_path)
    slot = _new_slot(COLD_TAG, "t-arrival")
    try:
        assert mgr._instances_for(COLD_TAG) == []
        await mgr._route_or_reserve(slot)
        assert len(box.reserves) == 1 and box.reserves[0][0] is slot
        assert box.reserves[0][1] == FREE_CARD
        assert len(mgr._instances_for(COLD_TAG)) == 1
    finally:
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# G5 / G7 / G8: through the real dispatch loop
# --------------------------------------------------------------------------------------

async def test_g5_a_fresh_follow_up_inside_a_live_grace_window_is_left_for_the_resident(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path)
    r = _resident(mgr, TAG, state=ResidentState.ACTIVE, grace_pair=("t-warm", TAG))
    follow_up = _new_slot(TAG, "t-warm")
    follow_up.state = SlotState.RECEIVED
    await mgr.queue.enqueue(follow_up)
    try:
        assert follow_up in mgr.queue.staging_snapshot() and len(mgr._model_residents()) == 1
        assert not r.grace.expired() and r.grace.matches("t-warm", TAG)
        await box.start_loop(monkeypatch)
        await box.step(6)
        assert box.reserves == [], "the dispatcher must not start an engine for a warm follow-up"
        assert follow_up in mgr.queue.staging_snapshot(), "the follow-up stays staged for the grace loop"
        assert _inbox_items(r) == []
        assert len(mgr._model_residents()) == 1
        # Control: the same slot with NO grace window to protect it is an ordinary arrival that
        # the dispatcher does take, so the hold above is the grace window and nothing else.
        r.grace = None
        await box.step(2)
        assert len(box.reserves) == 1 and box.reserves[0][0] is follow_up
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_g7_a_fresh_arrival_popped_by_the_loop_spawns_the_second_engine(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path)
    r = _resident(mgr, TAG, state=ResidentState.ACTIVE)
    arrival = _new_slot(TAG, "t-new")
    arrival.state = SlotState.RECEIVED
    await mgr.queue.enqueue(arrival)
    try:
        await box.start_loop(monkeypatch)
        await box.step(2)
        assert len(box.reserves) == 1
        assert box.reserves[0][0] is arrival and box.reserves[0][1] == FREE_CARD
        assert arrival not in mgr.queue.staging_snapshot() and _inbox_items(r) == []
        assert len(mgr._instances_for(TAG)) == 2
    finally:
        await box.stop_loop()
        await mgr.shutdown()


async def test_g8_a_fresh_arrival_that_cannot_spawn_parks_in_the_busy_engine_inbox(tmp_path, monkeypatch):
    mgr, box = await _world(tmp_path)
    box.admit_result = False
    r = _resident(mgr, TAG, state=ResidentState.ACTIVE)
    arrival = _new_slot(TAG, "t-new")
    arrival.state = SlotState.RECEIVED
    await mgr.queue.enqueue(arrival)
    try:
        await box.start_loop(monkeypatch)
        await box.step(2)
        assert box.reserves == []
        assert _inbox_items(r) == [arrival]
        assert arrival not in mgr.queue.staging_snapshot()
        assert len(box.admits) == 1
    finally:
        await box.stop_loop()
        await mgr.shutdown()
