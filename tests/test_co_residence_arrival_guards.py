"""Guards for the arrival path's second-engine step: the conditions that must NOT change.

These pin the second-engine step's conditions: the tag's own cap, the
box-wide sidecar budget, the Fast Lane relocation rule for a tag that already has an engine,
and the session-affinity record after a spawn. Each test drives the real
``TurbohaulManager._route_or_reserve`` with a real queue and hand-built engines; only the
card picker, the VRAM gate, the placement resolver, the cap answer and the spawn are faked.

State only: inbox contents, call counts of the faked probes, the reserve arguments and the
affinity map. No log lines.

  test_s1  the tag's engine count equals its effective cap (box budget still free): park, no probe
  test_s2  the box budget is spent by a foreign engine though the tag's cap allows more: park, no probe
  test_s3  a Fast Lane relocation card never overrides the picker for a tag that has an engine
      (control: for a cold tag it does)
  test_s4  after an arrival spawn the session maps to the new engine
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

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
FOREIGN = "tag-b"
COLD_TAG = "tag-cold"
FREE_CARD = 1
RELOCATED_CARD = 2
NEED_MIB = 4000


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


def _resident(mgr, tag, *, card=0, port=59701):
    r = Resident(
        model_tag=tag, resident_key=tag, port=port, state=ResidentState.ACTIVE, main_gpu=card,
        split_mode="none", reserved_need_mib=NEED_MIB,
        grace=GraceTimer(grace_seconds=3600, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=3600),
        last_active_monotonic=time.monotonic(),
    )
    r.inbox = asyncio.Queue()
    mgr._residents[tag] = r
    return r


class Box:
    """What the faked outside world was asked."""

    def __init__(self):
        self.reserves = []        # (slot, main_gpu_override, split_mode_override)
        self.picks = []
        self.admits = []


async def _world(tmp_path, *, eff_cap, cap=2):
    boot, runtime = _boot_runtime(tmp_path, cap)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    box = Box()

    def cap_placement(tag):
        return eff_cap, True

    def resolve(tag):
        return (NEED_MIB, 1, 0, "none", 0, True, False)

    def pick(tag, need):
        box.picks.append((tag, need))
        return FREE_CARD, "none"

    def admits(need, parallel, g, sp, *a, **k):
        box.admits.append((g, sp))
        return True

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
    return mgr, box


def _slot(tag, thread):
    slot = Slot.new(tag, prompt="p", thread_id=thread)
    slot.state = SlotState.STAGED
    return slot


def _inbox(r):
    return list(r.inbox._queue)


def _ids(slots):
    return [s.slot_id for s in slots]


async def test_s1_engine_count_at_the_tags_cap_parks_without_probing(tmp_path):
    """One engine, effective cap 1, box budget free (1 of 2): the arrival waits on that engine."""
    mgr, box = await _world(tmp_path, eff_cap=1)
    r = _resident(mgr, TAG)
    slot = _slot(TAG, "t-arrival")
    try:
        assert len(mgr._model_residents()) == 1 < mgr.runtime.queue.max_parallel_sidecars
        assert len(mgr._instances_for(TAG)) == 1
        await mgr._route_or_reserve(slot)
        assert box.picks == [] and box.admits == [], "a tag at its cap must not look for another card"
        assert box.reserves == [], "a tag at its cap must not start another engine"
        assert _ids(_inbox(r)) == [slot.slot_id]
    finally:
        await mgr.shutdown()


async def test_s2_a_spent_box_budget_parks_without_probing_even_if_the_tag_cap_is_higher(tmp_path):
    """Cap 2 holds one engine of the tag plus one foreign engine; the tag's own cap says 3."""
    mgr, box = await _world(tmp_path, eff_cap=3)
    r = _resident(mgr, TAG)
    foreign = _resident(mgr, FOREIGN, card=FREE_CARD, port=59702)
    slot = _slot(TAG, "t-arrival")
    try:
        assert len(mgr._model_residents()) == 2 == mgr.runtime.queue.max_parallel_sidecars
        assert len(mgr._instances_for(TAG)) == 1 < 3
        await mgr._route_or_reserve(slot)
        assert box.picks == [] and box.admits == [], "a full box must not look for another card"
        assert box.reserves == [], "a full box must not start a third engine"
        assert _ids(_inbox(r)) == [slot.slot_id] and _inbox(foreign) == []
        assert len(mgr._model_residents()) == 2
    finally:
        await mgr.shutdown()


async def test_s3_a_relocation_card_rules_only_a_cold_start(tmp_path):
    """A tag that already has an engine takes the picker's card; a cold tag takes the relocation card."""
    mgr, box = await _world(tmp_path, eff_cap=2)
    _resident(mgr, TAG)
    warm = _slot(TAG, "t-warm")
    warm.fastlane_relocated_main_gpu = RELOCATED_CARD
    warm.fastlane_relocated_split_mode = "none"
    cold = _slot(COLD_TAG, "t-cold")
    cold.fastlane_relocated_main_gpu = RELOCATED_CARD
    cold.fastlane_relocated_split_mode = "none"
    try:
        assert mgr._instances_for(COLD_TAG) == [] and len(mgr._instances_for(TAG)) == 1
        await mgr._route_or_reserve(cold)               # control: the rule that must keep working
        assert len(box.reserves) == 1 and box.reserves[0][0] is cold
        assert box.reserves[0][1] == RELOCATED_CARD, "a cold start follows the relocation card"
        # The box is now full (2 of 2), so free a slot the way a real unload would.
        del mgr._residents[f"{COLD_TAG}#{len(mgr._residents) - 1}"]
        await mgr._route_or_reserve(warm)
        assert len(box.reserves) == 2 and box.reserves[1][0] is warm
        assert box.reserves[1][1] == FREE_CARD, "a second copy must not land on the relocation card"
        assert box.reserves[1][2] == "none"
    finally:
        await mgr.shutdown()


async def test_s4_an_arrival_spawn_records_the_session_on_the_new_engine(tmp_path):
    mgr, box = await _world(tmp_path, eff_cap=2)
    _resident(mgr, TAG)
    slot = _slot(TAG, "t-session")
    try:
        assert mgr._session_affinity.get((TAG, "tid:t-session")) is None
        await mgr._route_or_reserve(slot)
        assert len(box.reserves) == 1
        new = [x for x in mgr._instances_for(TAG) if x.resident_key != TAG]
        assert len(new) == 1
        assert mgr._session_affinity.get((TAG, "tid:t-session")) == new[0].resident_key
    finally:
        await mgr.shutdown()
