"""The idle-unload-timer half of the designated-victim eviction fix.

`_idle_window_seconds` is computed purely from
`r.latest_keep_alive_s` / per-model config -- zero Fast Lane awareness -- so
a designated victim reaches `_drive_resident`'s post-turn idle_window branch
(the post-turn branch in manager.py) exactly like any other resident and only
gets the immediate-evict treatment by coincidence (an explicit keep_alive=0
or a zero per-model idle timeout). Expected behavior: the victim needs to
immediately lose its idle unload timer, but the timer path just did not
activate at all.

Drives `_drive_resident` DIRECTLY, same "no full dispatcher" shape as
the grace-wiring test's direct `_serve_on_resident`
calls -- exercises exactly the widened branch without the full
worker_loop/_dispatch_loop/submit machinery (that harness belongs to the
acceptance suite, not duplicated here). `safety_enabled`
False + no manifest on disk (falls back to `gguf_path=".../missing.gguf"`,
`argv=[]` via `_spawn_for_resident`'s own `except FileNotFoundError` arm) --
this test never actually launches a process, only the FSM/branch decision
around it.

The victim's live outranking claim is registered via
`_register_fastlane_claim_locked` directly, the SAME construction three
independent existing tests already use (the grace-wiring test, a second
fast-lane claim test and the acceptance suite) -- not a shortcut invented for this file.
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


def _boot_runtime(tmp_path, *, grace_seconds=5, idle_hot_load_seconds=120, fastlane_rules=None):
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
            default_port_base=59950,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=2,
            grace_seconds=grace_seconds, max_grace_extensions=50,
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
    pid = [88_000]

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


_RULES = [FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1))]


def _resident(mgr, model_tag, port, *, rank_client_meta, grace_seconds):
    r = Resident(
        model_tag=model_tag, resident_key=model_tag, port=port,
        grace=GraceTimer(grace_seconds=grace_seconds, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
        rank_client_meta=rank_client_meta,
        last_active_monotonic=time.monotonic(),
    )
    r.inbox = asyncio.Queue()
    mgr._residents[model_tag] = r
    return r


async def _drive_one_turn_and_classify(mgr, r, model_tag, *, timeout_s):
    """Puts one anchor slot in r.inbox, runs _drive_resident directly, polls
    for the FIRST of: resident deregistered (evicted) or resident parked
    IDLE_EVICTABLE (ordinary idle window). Cancels the driver task either
    way -- this test only needs one turn's outcome, not the long-lived loop."""
    anchor = Slot.new(model_tag, prompt="hi", thread_id="t1")
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)

    drive_task = asyncio.create_task(mgr._drive_resident(r))
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            cur = mgr._residents.get(model_tag)
            if cur is None:
                return "evicted"
            if cur.state is ResidentState.IDLE_EVICTABLE:
                return "idle_evictable"
            await asyncio.sleep(0.05)
        return "timeout"
    finally:
        drive_task.cancel()
        try:
            await drive_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
class TestIdleUnloadTimerFollowsDesignatedVictim:
    async def test_discriminator_victim_evicted_immediately_not_idle_window(self, tmp_path):
        """A designated victim (worst-ranked,
        outranked by a live claim) must NOT get the ORDINARY per-model idle
        window after its turn completes (parked IDLE_EVICTABLE with a real
        idle_expires_at, not torn down); it goes straight through
        `_begin_unload_locked` instead."""
        grace_seconds = 5
        _rules = [
            FastLaneRule(address="9.9.9.9", tag_ranks=FastLaneTagRanks(main=1)),
            FastLaneRule(address="1.2.3.4", tag_ranks=FastLaneTagRanks(main=1)),
        ]
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            idle_hot_load_seconds=120, fastlane_rules=_rules,
        )
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        victim = _resident(
            mgr, "victim-model", 59951,
            rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
        )
        victim.state = ResidentState.ACTIVE
        victim.last_active_monotonic = 1.0
        survivor = _resident(
            mgr, "survivor-model", 59952,
            rank_client_meta={"ip": "1.2.3.4"}, grace_seconds=grace_seconds,
        )
        survivor.state = ResidentState.ACTIVE
        survivor.last_active_monotonic = 1000.0

        claimant = Slot.new("claimant-model", prompt="hi", thread_id="claim")
        claimant.fastlane = FastLaneMatch(
            rule_index=0, raw_address="9.9.9.9", label="",
            effective_tag="main", rank=1,
        )
        mgr._register_fastlane_claim_locked(claimant, "make_room_starved_count_cap")
        assert mgr._is_designated_unload_target_locked(victim) is True

        try:
            outcome = await _drive_one_turn_and_classify(
                mgr, victim, "victim-model", timeout_s=6.0,
            )
            assert outcome == "evicted", (
                f"designated victim should be evicted immediately after its "
                f"turn, not parked IDLE_EVICTABLE waiting out an ordinary "
                f"120s idle window -- outcome={outcome!r}"
            )
        finally:
            mgr._residents.pop("survivor-model", None)
            await mgr.shutdown()

    async def test_GREEN_CONTROL_non_victim_still_gets_ordinary_idle_window(self, tmp_path):
        """★ CONTROL -- must PASS with or without victim handling. A resident
        that is NOT the designated victim (no outranking live claim) still
        gets its ordinary idle window, not immediate eviction -- proves the
        widened branch does not over-fire for everyone."""
        grace_seconds = 5
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=grace_seconds,
            idle_hot_load_seconds=120, fastlane_rules=_RULES,
        )
        mgr = TurbohaulManager(boot, runtime, **_mocks())

        solo = _resident(
            mgr, "solo-model", 59953,
            rank_client_meta=None, grace_seconds=grace_seconds,
        )
        solo.state = ResidentState.ACTIVE
        assert mgr._is_designated_unload_target_locked(solo) is False

        try:
            # solo is NOT the victim -> normal grace wait holds the full
            # grace_seconds before the idle_window branch even runs.
            outcome = await _drive_one_turn_and_classify(
                mgr, solo, "solo-model", timeout_s=grace_seconds + 3.0,
            )
            assert outcome == "idle_evictable", (
                f"a non-victim resident must still get its ordinary idle "
                f"window, not be evicted immediately -- outcome={outcome!r}"
            )
        finally:
            await mgr.shutdown()
