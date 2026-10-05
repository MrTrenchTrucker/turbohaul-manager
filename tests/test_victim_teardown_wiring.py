"""Victim teardown requeue: `_drive_resident`'s immediate-eviction
branch (`idle_window <= 0`, in manager.py) re-checks
`_is_designated_unload_target_locked(r)` FRESH and, when true, drains `r.inbox` into
`_requeue_slots_tail_or_fail` instead of leaving it for `_begin_unload_locked`'s
own (HEAD) drain; the head path stays for non-victim evictions
(that behaviour is unchanged).

Drives `_drive_resident` DIRECTLY as a task, bypassing `submit_and_wait`/
`worker_loop` entirely: the resident's `inbox` is pre-populated with TWO real
`Slot`s before the task starts. `_drive_resident`'s own first line
(`slot = await r.inbox.get()`) pulls slot 1 as the anchor and serves it; with
`idle_hot_load_seconds=0`, `idle_window` resolves `<=0` the INSTANT that serve
completes, so the immediate-eviction branch fires BEFORE slot 2 is ever pulled
from the inbox -- exactly the "victim's own queued follow-up, never served"
scenario this behaviour exists for.
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.fastlane import FastLaneMatch
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, *, fastlane_rules=()):
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
            default_port_base=59800,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=2, grace_seconds=2,
            max_grace_extensions=50, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=bool(fastlane_rules), rules=list(fastlane_rules)),
    )
    return boot, runtime


def _seed_manifest(boot, model_tag, *, main_gpu=0):
    import yaml
    p = boot.storage.manifests_path / f"{model_tag}.yaml"
    p.write_text(yaml.safe_dump({
        "model_tag": model_tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 0,
        "context_size": 2048,
        "expected_vram_bytes": 0,
        "llama_server_flags": {"split_mode": "none", "main_gpu": main_gpu},
    }))


def _high_vram():
    return patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000, 80000])


def _make_fakes(pid_start: int):
    pid = [pid_start]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete


async def _wait_until(predicate, *, timeout=5.0, interval=0.01):
    t0 = time.monotonic()
    while True:
        if predicate():
            return
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"predicate never became true within {timeout}s")
        await asyncio.sleep(interval)


_RULES = (FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),)


@pytest.mark.asyncio
class TestVictimTeardownRequeuesToTail:
    async def test_designated_victims_inbox_backlog_lands_at_tail_not_head(self, tmp_path):
        """MUST FAIL ON UNMODIFIED CODE (before this change): the victim's queued
        follow-up would go through `_begin_unload_locked`'s own HEAD-requeue,
        landing BEFORE the pre-existing marker, not after it."""
        # Designation requires a LIVE claim that
        # STRICTLY outranks the resident (the designation rule's precondition).
        # The module-level _RULES has a single entry, so both residents here
        # resolve to rule_index 0 and NOTHING could outrank them; there would be
        # no legal way to designate a victim at all. Scoped to this test so the
        # siblings' table is untouched: 9.9.9.9 takes index 0 (the claimant),
        # 1.2.3.4 moves to index 1 (both residents).
        _rules = (
            FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1)),
            FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),
        )
        boot, runtime = _boot_runtime(tmp_path, fastlane_rules=_rules)
        _seed_manifest(boot, "victim-model", main_gpu=0)
        _seed_manifest(boot, "survivor-model", main_gpu=1)
        fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete = _make_fakes(90000)
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )

        marker = Slot.new("marker-model", prompt="p", thread_id="t-marker")
        await mgr.queue.enqueue(marker)

        # A second, BETTER-ranked (more-recently-active, same rule_index)
        # resident, loaded, so `victim` unambiguously resolves as the
        # designated victim (worst == least-recently-active at a rule tie,
        # _is_designated_unload_target_locked's own convention).
        survivor = Resident(
            model_tag="survivor-model", resident_key="survivor-model",
            handle=None, port=59811,
            grace=GraceTimer(grace_seconds=2, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
            inbox=asyncio.Queue(),
            rank_client_meta={"ip": "1.2.3.4"},
            last_active_monotonic=time.monotonic() + 1000.0,
        )
        survivor.state = ResidentState.ACTIVE
        mgr._residents["survivor-model"] = survivor

        victim = Resident(
            model_tag="victim-model", resident_key="victim-model",
            handle=None, port=59812,
            grace=GraceTimer(grace_seconds=2, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
            inbox=asyncio.Queue(),
            rank_client_meta={"ip": "1.2.3.4"},
            last_active_monotonic=1.0,
        )
        mgr._residents["victim-model"] = victim

        # The designation rule's precondition: a higher-priority request must actually be
        # waiting. Registered through the production path so this fixture
        # cannot drift from the claim shape `_claim_is_live` reads.
        claimant = Slot.new("claimant-model", prompt="p", thread_id="t-claim")
        claimant.fastlane = FastLaneMatch(
            rule_index=0, raw_address="9.9.9.9", label="",
            effective_tag="main", rank=1,
        )
        mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")

        assert mgr._is_designated_unload_target_locked(victim) is True
        # The precondition is load-bearing, not decoration: drop the claim and
        # the victim stops being one (grace is unconditional in that state).
        mgr._release_fastlane_claim_locked(claimant, "test_control")
        assert mgr._is_designated_unload_target_locked(victim) is False
        mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")

        anchor = Slot.new("victim-model", prompt="p", thread_id="t-anchor")
        anchor.client_meta = {"ip": "1.2.3.4"}
        anchor.state = SlotState.STAGED
        followup = Slot.new("victim-model", prompt="p", thread_id="t-followup")
        followup.state = SlotState.STAGED
        victim.inbox.put_nowait(anchor)
        victim.inbox.put_nowait(followup)

        with _high_vram():
            driver = asyncio.create_task(mgr._drive_resident(victim))
            try:
                await asyncio.wait_for(driver, timeout=6.0)
            finally:
                if not driver.done():
                    driver.cancel()

            await _wait_until(lambda: followup.state is SlotState.STAGED, timeout=2.0)

            staging = list(mgr.queue._staging)
            assert followup in staging, (
                "the victim's queued follow-up must be requeued, not dropped"
            )
            assert marker in staging
            assert staging.index(followup) > staging.index(marker), (
                "a DESIGNATED VICTIM's own backlog must land at the TAIL -- "
                "behind the pre-existing marker entry, not ahead of it"
            )

            await mgr.shutdown()

    async def test_negative_control_non_victims_inbox_backlog_still_lands_at_head(self, tmp_path):
        """NEGATIVE CONTROL: a resident that is NOT the designated victim (here,
        the only loaded resident, unresolvable -- never a candidate) must keep
        the existing HEAD-requeue behaviour, unchanged."""
        boot, runtime = _boot_runtime(tmp_path, fastlane_rules=())
        _seed_manifest(boot, "solo-model", main_gpu=0)
        fake_spawn, fake_health, fake_sigterm, fake_vram, fake_complete = _make_fakes(91000)
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )

        marker = Slot.new("marker-model", prompt="p", thread_id="t-marker")
        await mgr.queue.enqueue(marker)

        solo = Resident(
            model_tag="solo-model", resident_key="solo-model",
            handle=None, port=59813,
            grace=GraceTimer(grace_seconds=2, max_extensions=50),
            idle=IdleHotTimer(idle_seconds=0),
            inbox=asyncio.Queue(),
        )
        mgr._residents["solo-model"] = solo

        assert mgr._is_designated_unload_target_locked(solo) is False

        anchor = Slot.new("solo-model", prompt="p", thread_id="t-anchor")
        anchor.state = SlotState.STAGED
        followup = Slot.new("solo-model", prompt="p", thread_id="t-followup")
        followup.state = SlotState.STAGED
        solo.inbox.put_nowait(anchor)
        solo.inbox.put_nowait(followup)

        with _high_vram():
            driver = asyncio.create_task(mgr._drive_resident(solo))
            try:
                await asyncio.wait_for(driver, timeout=6.0)
            finally:
                if not driver.done():
                    driver.cancel()

            await _wait_until(lambda: followup.state is SlotState.STAGED, timeout=2.0)

            staging = list(mgr.queue._staging)
            assert followup in staging
            assert marker in staging
            assert staging.index(followup) < staging.index(marker), (
                "a NON-victim's own backlog must still land at the HEAD, unchanged"
            )

            await mgr.shutdown()
