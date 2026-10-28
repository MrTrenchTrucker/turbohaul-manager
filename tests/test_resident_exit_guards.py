"""Resident exit: the manager-level guards that the keep-warm hold must NOT change.

A keep_alive=0 resident is still torn down at once when nothing argues for keeping it:
no same-model ACTIVE sibling, a designated victim, an outranking claim, other-model
capacity demand, a pending reclaim on its card, cap<=1, or a sibling that is not a live
same-model ACTIVE resident. These tests pin that by STATE only (registry membership, FSM
state, the unload counter), never by a log line, so they hold both with and without the
keep-warm hold.

Harness shape is lifted from tests/test_same_model_residency_floor_live_path.py
and tests/test_idle_unload_timer.py: _drive_resident is driven directly with
safety off and no manifest on disk, so no process is launched.
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
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

GRACE_S = 1

# Rule 0 is the resident's own client (the better rank); rule 1 is a worse client, so a
# claim from rule 1 does not outrank a resident that is served under rule 0.
_RULES_RESIDENT_BETTER = [
    FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1)),
]


def _boot_runtime(tmp_path, *, idle_hot_load_seconds, max_parallel_sidecars=2, fastlane_rules=None):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59970,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=GRACE_S, max_grace_extensions=50,
            idle_hot_load_seconds=idle_hot_load_seconds,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=fastlane_rules or []),
    )
    return boot, runtime


def _fake_handle(model_tag, port, pid):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mocks():
    pid = [92_000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **kw):
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True}

    return dict(
        spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
        vram_fn=fake_vram, complete_fn=fake_complete,
    )


def _resident(mgr, model_tag, port, *, resident_key=None, state=ResidentState.ACTIVE,
              rank_client_meta=None):
    key = model_tag if resident_key is None else resident_key
    r = Resident(
        model_tag=model_tag, resident_key=key, port=port,
        grace=GraceTimer(grace_seconds=GRACE_S, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
        rank_client_meta=rank_client_meta,
        last_active_monotonic=time.monotonic(),
        state=state,
    )
    r.inbox = asyncio.Queue()
    mgr._residents[key] = r
    return r


async def _drive_one_turn(mgr, r, *, timeout_s, thread_id="t1", snap=None):
    """Put one anchor slot in r.inbox, run _drive_resident directly, and report the first
    terminal outcome: 'evicted' (deregistered) or 'idle_evictable' (parked warm).

    The driver is cancelled once the outcome is seen, and a cancelled parked driver
    deregisters itself, so ``snap`` (when given) records the registry and FSM state at the
    moment of the outcome, before that cancel."""
    anchor = Slot.new(r.model_tag, prompt="hi", thread_id=thread_id)
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)

    drive_task = asyncio.create_task(mgr._drive_resident(r))
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            cur = mgr._residents.get(r.resident_key)
            if cur is None:
                outcome = "evicted"
            elif cur.state is ResidentState.IDLE_EVICTABLE:
                outcome = "idle_evictable"
            else:
                await asyncio.sleep(0.05)
                continue
            if snap is not None:
                snap.update(
                    outcome=outcome, registered=cur is r,
                    state=None if cur is None else cur.state,
                    idle_expires_at=None if cur is None else cur.idle_expires_at,
                )
            return outcome
        return "timeout"
    finally:
        drive_task.cancel()
        try:
            await drive_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_G1_sole_resident_at_cap2_is_evicted_at_keep_alive_zero(tmp_path):
    boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g1-model", 59971)
    unloads_before = mgr._unload_count
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0)
        assert outcome == "evicted"
        assert "g1-model" not in mgr._residents
        assert r.state is ResidentState.DEAD
        assert mgr._unload_count == unloads_before + 1
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_G2_designated_victim_with_same_model_sibling_is_still_evicted(tmp_path, monkeypatch):
    boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g2-model", 59972)
    _resident(mgr, "g2-model", 59973, resident_key="g2-model#1")
    monkeypatch.setattr(mgr, "_is_designated_unload_target_locked", lambda rr: rr is r)
    unloads_before = mgr._unload_count
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0)
        assert outcome == "evicted"
        assert "g2-model" not in mgr._residents
        assert r.state is ResidentState.DEAD
        assert mgr._unload_count == unloads_before + 1
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_G3a_other_model_fastlane_claim_does_not_keep_a_same_model_sibling_warm(tmp_path):
    boot, runtime = _boot_runtime(
        tmp_path, idle_hot_load_seconds=0, fastlane_rules=_RULES_RESIDENT_BETTER,
    )
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g3a-model", 59974, rank_client_meta={"ip": "1.2.3.4"})
    _resident(mgr, "g3a-model", 59975, resident_key="g3a-model#1")
    claimant = Slot.new("other-model", prompt="waiting", thread_id="claim")
    claimant.fastlane = FastLaneMatch(
        rule_index=1, raw_address="9.9.9.9", label="", effective_tag="main", rank=1,
    )
    async with mgr._registry_lock:
        mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")
        designated = mgr._is_designated_unload_target_locked(r)
    assert designated is False, "precondition: the resident must not be the designated victim"
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0)
        assert outcome == "evicted"
        assert "g3a-model" not in mgr._residents
        assert r.state is ResidentState.DEAD
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_G3b_pending_reclaim_on_the_card_does_not_keep_a_same_model_sibling_warm(tmp_path):
    boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g3b-model", 59976)
    _resident(mgr, "g3b-model", 59977, resident_key="g3b-model#1")
    assert not mgr._fastlane_claims, "precondition: no fast-lane claim may be registered"
    mgr._pending_reclaim_mib[r.main_gpu] = 100
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0)
        assert outcome == "evicted"
        assert "g3b-model" not in mgr._residents
        assert r.state is ResidentState.DEAD
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_G4a_cap1_single_resident_is_evicted_at_keep_alive_zero(tmp_path):
    boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0, max_parallel_sidecars=1)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g4a-model", 59978)
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0)
        assert outcome == "evicted"
        assert "g4a-model" not in mgr._residents
        assert r.state is ResidentState.DEAD
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_G4b_cap1_same_model_sibling_still_evicts_the_resident(tmp_path):
    boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0, max_parallel_sidecars=1)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g4b-model", 59979)
    _resident(mgr, "g4b-model", 59980, resident_key="g4b-model#1")
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0)
        assert outcome == "evicted"
        assert "g4b-model" not in mgr._residents
        assert r.state is ResidentState.DEAD
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_G5_positive_keep_alive_parks_idle_with_a_real_deadline(tmp_path):
    boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=300)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g5-model", 59981)
    _resident(mgr, "g5-model", 59982, resident_key="g5-model#1")
    snap = {}
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0, snap=snap)
        assert outcome == "idle_evictable"
        assert snap["registered"] is True, "a parked resident must stay in the registry"
        assert snap["state"] is ResidentState.IDLE_EVICTABLE
        remaining = snap["idle_expires_at"] - time.monotonic()
        assert abs(remaining - 300.0) <= 5.0, f"idle deadline should be about +300s, got {remaining!r}"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_G6_sibling_of_another_model_does_not_keep_the_resident_warm(tmp_path):
    boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g6-model", 59983)
    _resident(mgr, "g6-other-model", 59984)
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0)
        assert outcome == "evicted"
        assert "g6-model" not in mgr._residents
        assert r.state is ResidentState.DEAD
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_G7_same_model_sibling_that_is_not_active_does_not_keep_the_resident_warm(tmp_path):
    boot, runtime = _boot_runtime(tmp_path, idle_hot_load_seconds=0)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    r = _resident(mgr, "g7-model", 59985)
    _resident(
        mgr, "g7-model", 59986, resident_key="g7-model#1",
        state=ResidentState.IDLE_EVICTABLE,
    )
    try:
        outcome = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0)
        assert outcome == "evicted"
        assert "g7-model" not in mgr._residents
        assert r.state is ResidentState.DEAD
    finally:
        await mgr.shutdown()
