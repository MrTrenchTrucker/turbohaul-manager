"""Pure extract of worker_loop's inline bin-death-strike /
ENGINE STALLED banner / 3-strike auto-quarantine block (cap<=1, save-timing
attribution, 3-strikes design) into a shared helper (`_record_bin_death_strike`),
so a cap>=2 idle-liveness sweep can call the SAME tested
implementation instead of a drifting copy.

ZERO BEHAVIOUR CHANGE. This file drives the REAL worker_loop cap<=1
dead-idle-holder sweep end to end (same harness shape as
the idle-dead classification test's TestDeadIdleSweepClassifyWiring)
and asserts on the mechanism's actual observable effects: the
`_bin_death_strikes` counter, the FE-visible `_engine_stall` dict shape, the
ENGINE STALLED log line, and the 3-strike `_quarantine_bin_triplet` trigger.
Same test, same assertions, run unchanged before AND after the extraction
(the right instrument, run in both directions) -- proves the refactor
preserved behaviour rather than merely asserting it did.

This mechanism had no test coverage of its own, at either cap: the only
tests that touch these code paths (bin_death_strike/_engine_stall/
quarantine_bin_triplet/save-timing) are the idle-dead classification test
and test_multislot_concurrency.py, and both are about IDLE_DEAD_CLASSIFY
(the separate, log-only classification), never the strike/banner/quarantine
half. This file is the first.
"""
import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

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
from turbohaul.manager import TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, grace_seconds=0, idle_hot_load_seconds=120):
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
            safety_enabled=False,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


async def _boot_manager_with_one_idle_hot_holder(tmp_path):
    """Same shape as the idle-dead classification test's helper: real
    worker_loop, DI fakes, one served request, holder now idle-hot and
    ALIVE."""
    boot, runtime = _boot_runtime(tmp_path)

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_fake_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(*a, **kw):
        return True, None

    async def fake_complete(slot, handle):
        return {"ok": True}

    manifests_dir = boot.storage.manifests_path
    (manifests_dir / "real-model.yaml").write_text(
        "model_tag: real-model\n"
        "gguf_blob_sha256: " + "a" * 64 + "\n"
        "display_name: \"Real Model\"\n"
        "description: test\n"
        "context_size: 2048\n"
        "expected_vram_bytes: 0\n"
        "llama_server_flags: {}\n"
    )

    mgr = TurbohaulManager(
        boot, runtime,
        spawn_fn=fake_spawn, health_fn=fake_health,
        sigterm_fn=fake_sigterm, vram_fn=fake_vram,
        complete_fn=fake_complete,
    )
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    await mgr.submit_and_wait("real-model", "prompt", thread_id="t1")
    await asyncio.sleep(0.2)  # grace + idle-hold
    assert mgr._idle_handle is not None, "idle holder should exist"
    return mgr


async def _wait_for(predicate, timeout_s=5.0, interval_s=0.05):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


def _stall_lines(caplog):
    return [r.message for r in caplog.records if "ENGINE STALLED" in r.message]


@pytest.mark.asyncio
class TestBinDeathStrikeSweepWiring:
    async def test_first_strike_sets_banner_and_increments_counter(
        self, tmp_path, caplog,
    ):
        mgr = await _boot_manager_with_one_idle_hot_holder(tmp_path)
        held = mgr._idle_handle
        port = held.port
        mgr._last_restored_bin = {port: "testbin_abc123.bin"}
        mgr._bin_death_strikes = {}
        held.proc.poll.return_value = -11  # child gone

        try:
            with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
                found = await _wait_for(
                    lambda: mgr._bin_death_strikes.get("testbin_abc123.bin") is not None
                )
            assert found, "sweep never recorded a bin-death-strike on a dead holder"
            assert mgr._bin_death_strikes["testbin_abc123.bin"] == 1
            stall = mgr._engine_stall
            assert stall is not None, "ENGINE STALLED banner (_engine_stall) never set"
            assert stall["active"] is True
            assert stall["attempt"] == 1
            assert stall["max"] == 3
            assert stall["bin"] == "testbin_abc123.bin"
            assert "ts" in stall
            assert _stall_lines(caplog), "no ENGINE STALLED log line emitted"
            # Save-timing's own "one blame per restore" rule: the port's
            # attribution is popped after the strike lands.
            assert port not in mgr._last_restored_bin
        finally:
            mgr._worker_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await mgr._worker_task

    async def test_third_strike_triggers_auto_quarantine(self, tmp_path, caplog):
        mgr = await _boot_manager_with_one_idle_hot_holder(tmp_path)
        held = mgr._idle_handle
        port = held.port
        mgr._last_restored_bin = {port: "repeat_offender.bin"}
        # Seed two PRIOR strikes -- a legitimate, standard way to reach the
        # 3rd-strike branch without three full realistic restore/die cycles
        # (same technique this codebase's own tests use elsewhere to reach a
        # specific state transition directly).
        mgr._bin_death_strikes = {"repeat_offender.bin": 2}
        held.proc.poll.return_value = -11

        with patch.object(mgr, "_quarantine_bin_triplet") as mock_quarantine:
            try:
                with caplog.at_level(logging.ERROR, logger="turbohaul.manager"):
                    found = await _wait_for(
                        lambda: mgr._bin_death_strikes.get("repeat_offender.bin") == 3
                    )
                assert found, "sweep never reached the 3rd strike"
                # _quarantine_bin_triplet is spawned via _spawn_bg(asyncio.to_thread(...))
                # -- give the event loop a beat to actually run it.
                await _wait_for(lambda: mock_quarantine.called, timeout_s=2.0)
                assert mock_quarantine.called, "3rd strike did not trigger auto-quarantine"
                mock_quarantine.assert_called_once_with("repeat_offender.bin")
                assert mgr._engine_stall["attempt"] == 3
            finally:
                mgr._worker_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await mgr._worker_task

    async def test_no_last_restored_bin_means_no_strike(self, tmp_path):
        """GREEN CONTROL: a dead holder with NOTHING recorded in
        _last_restored_bin (never restored a bin onto this port -- e.g. a
        cold spawn) must not fabricate a strike against a bin that was never
        implicated. Shares the dead-holder-detection code path with the
        subject tests but takes the OTHER branch (`if _bfn:` false)."""
        mgr = await _boot_manager_with_one_idle_hot_holder(tmp_path)
        held = mgr._idle_handle
        mgr._last_restored_bin = {}  # nothing restored on this port
        mgr._bin_death_strikes = {}
        held.proc.poll.return_value = -11

        try:
            await asyncio.sleep(0.5)  # let a few sweep ticks pass
            assert mgr._bin_death_strikes == {}, (
                "a strike was recorded with no last-restored-bin attribution"
            )
            assert getattr(mgr, "_engine_stall", None) is None, (
                "ENGINE STALLED banner set with no bin to attribute the death to"
            )
        finally:
            mgr._worker_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await mgr._worker_task
