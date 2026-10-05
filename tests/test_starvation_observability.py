"""Observability for MAKE_ROOM_STARVED, which cannot tell a DEFECT
from a BUSY HOST. LOGGING ONLY -- no admission/eviction/queue/fairness
behaviour changes anywhere in this change.

Motivation: two starvation episodes read IDENTICALLY on
MAKE_ROOM_STARVED; nvidia-smi showed one was a genuinely saturated host
(near-full compute and VRAM) where waiting is correct. Part 1 below adds
per-card GPU util + VRAM used/total + the incoming request's traffic class
to that line, read ONLY from the existing off-loop ~1Hz telemetry cache
(never a live nvidia-smi call on the _registry_lock-held path).

On a saturated host MAKE_ROOM_STARVED fires with residents=1 or
residents=2 -- residents=0 NEVER, not rare,
zero. That is logically necessary (an empty host has no make-room decision
to make), so a report of the form "both cards were EMPTY
and nothing was getting loaded" cannot be seen by that line no matter
what fields it grows. Part 2 adds a SEPARATE admission-path line in
`_dispatch_loop`, checked every loop iteration BEFORE `pop_next`, that
fires exactly when residents==0 and staging_depth>0 -- the join that the
per-opportunity line cannot see, since it occurs on ZERO of the
logged opportunities.

Behaviour is added here (two new call sites, a new cache, a new log
line), so it is tested as a behaviour change, not as a pure
refactor, even though the change is "logging only" in that no
admission/eviction DECISION changes.
"""
import asyncio
import logging
import subprocess
from unittest.mock import patch

import pytest

from turbohaul.safety import _read_gpu_util_all_percent

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot, SlotState


@pytest.fixture
def boot_and_runtime(tmp_path):
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return boot, runtime


def _slot(model_tag="m", ip=None):
    return Slot(
        slot_id="s1",
        model_tag=model_tag,
        state=SlotState.RECEIVED,
        client_meta={"ip": ip} if ip else {},
    )


class TestReadGpuUtilAllPercent:
    """Direct unit coverage of the new probe itself -- mirrors
    _read_free_vram_all_mib's shape exactly, so it is exercised the same
    way (query-gpu flag, csv parse, None-on-failure), independent of the
    manager-level cache tests above."""

    def test_parses_multi_gpu_csv(self):
        with patch(
            "turbohaul.safety.subprocess.check_output",
            return_value="37\n98\n",
        ):
            assert _read_gpu_util_all_percent() == [37, 98]

    def test_queries_utilization_gpu_not_memory(self):
        with patch("turbohaul.safety.subprocess.check_output", return_value="0\n") as m:
            _read_gpu_util_all_percent()
            args = m.call_args[0][0]
            assert "--query-gpu=utilization.gpu" in args

    def test_none_when_nvidia_smi_missing(self):
        with patch(
            "turbohaul.safety.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert _read_gpu_util_all_percent() is None

    def test_none_on_subprocess_error(self):
        with patch(
            "turbohaul.safety.subprocess.check_output",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ):
            assert _read_gpu_util_all_percent() is None

    def test_malformed_line_skipped_not_fatal(self):
        with patch(
            "turbohaul.safety.subprocess.check_output",
            return_value="12\nnot-a-number\n45\n",
        ):
            assert _read_gpu_util_all_percent() == [12, 45]


class TestMakeRoomStarvedTelemetryFields:
    """Part 1: per-card gpu_util_pct/vram_used_mib/vram_total_mib + traffic_class,
    APPENDED after MAKE_ROOM_STARVED's existing contract fields."""

    def test_existing_contract_fields_stay_first_and_in_order(self, boot_and_runtime, caplog):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        mgr._log_make_room_starved(
            _slot("modelA"), "make_room_starved_vram",
            {"why": "reason_unknown", "residents": 1, "eligible": 0},
        )
        line = next(r.message for r in caplog.records if "MAKE_ROOM_STARVED" in r.message)
        assert line.startswith(
            "MAKE_ROOM_STARVED reason=make_room_starved_vram why=reason_unknown "
            "model_tag=modelA fastlane_client=unresolvable rule_index=unresolvable "
        ), line

    def test_new_fields_prove_both_ways_busy_vs_idle(self, boot_and_runtime, caplog):
        """A field that only ever prints one value is a decoration, not an
        instrument -- prove BUSY and IDLE produce genuinely different values."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        caplog.set_level(logging.INFO, logger="turbohaul.manager")

        # BUSY: models a saturated host.
        mgr._gpu_util_pct = [98, 12]
        mgr._vram_free_mib = [1400, 20000]
        mgr._vram_total_mib = [24576, 24576]
        caplog.clear()
        mgr._log_make_room_starved(_slot("modelA"), "make_room_starved_vram", {"why": "x"})
        busy_line = caplog.records[-1].message

        # IDLE: same manager, different cache snapshot (the ~1Hz poller ticked).
        mgr._gpu_util_pct = [3, 0]
        mgr._vram_free_mib = [23000, 24000]
        mgr._vram_total_mib = [24576, 24576]
        caplog.clear()
        mgr._log_make_room_starved(_slot("modelA"), "make_room_starved_vram", {"why": "x"})
        idle_line = caplog.records[-1].message

        assert "gpu0_util_pct=98" in busy_line
        assert "gpu0_vram_used_mib=23176" in busy_line  # 24576-1400
        assert "gpu1_util_pct=12" in busy_line
        assert "gpu0_util_pct=3" in idle_line
        assert "gpu0_vram_used_mib=1576" in idle_line  # 24576-23000
        assert busy_line != idle_line
        # existing fields must be byte-identical across the two ticks -- only
        # the appended telemetry differs.
        prefix = "MAKE_ROOM_STARVED reason=make_room_starved_vram why=x model_tag=modelA "
        assert busy_line.startswith(prefix)
        assert idle_line.startswith(prefix)

    def test_probe_never_ran_emits_sentinel_not_silent_omission(self, boot_and_runtime, caplog):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._gpu_util_pct is None
        assert mgr._vram_free_mib is None
        assert mgr._vram_total_mib is None
        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        mgr._log_make_room_starved(_slot("modelA"), "make_room_starved_vram", {"why": "x"})
        line = caplog.records[-1].message
        assert "gpu_telemetry=unavailable" in line

    def test_partial_probe_failure_per_field_sentinel(self, boot_and_runtime, caplog):
        """util probe failed but VRAM succeeded (or vice versa) -- each
        card's fields degrade independently, never all-or-nothing."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._gpu_util_pct = None
        mgr._vram_free_mib = [1000]
        mgr._vram_total_mib = [24576]
        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        mgr._log_make_room_starved(_slot("modelA"), "make_room_starved_vram", {"why": "x"})
        line = caplog.records[-1].message
        assert "gpu0_util_pct=unavailable" in line
        assert "gpu0_vram_used_mib=23576" in line
        assert "gpu0_vram_total_mib=24576" in line

    def test_traffic_class_registered_when_fastlane_matches(self, boot_and_runtime, caplog):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr.runtime.fastlane = FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address="10.0.0.5", label="known-client")],
        )
        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        mgr._log_make_room_starved(
            _slot("modelA", ip="10.0.0.5"), "make_room_starved_vram", {"why": "x"},
        )
        line = caplog.records[-1].message
        assert "traffic_class=registered" in line
        assert "fastlane_client=10.0.0.5" in line

    def test_traffic_class_unregistered_when_no_match(self, boot_and_runtime, caplog):
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr.runtime.fastlane = FastLaneConfig(
            enabled=True,
            rules=[FastLaneRule(address="10.0.0.5", label="known-client")],
        )
        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        mgr._log_make_room_starved(
            _slot("modelA", ip="203.0.113.9"), "make_room_starved_vram", {"why": "x"},
        )
        line = caplog.records[-1].message
        assert "traffic_class=unregistered" in line

    def test_traffic_class_unregistered_when_fastlane_off(self, boot_and_runtime, caplog):
        """Negative control: no client_meta at all (the ordinary, feature-off
        case) -- must not crash, must read unregistered."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        mgr._log_make_room_starved(_slot("modelA"), "make_room_starved_vram", {"why": "x"})
        line = caplog.records[-1].message
        assert "traffic_class=unregistered" in line

    def test_no_nvidia_smi_shell_out_on_this_path(self, boot_and_runtime, caplog, monkeypatch):
        """The class's own docstring contract: this fires under _registry_lock
        (via _route_or_reserve). A live probe here would be the exact
        hazard this codebase avoids elsewhere."""
        import subprocess as _subprocess

        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        mgr._gpu_util_pct = [50]
        mgr._vram_free_mib = [1000]
        mgr._vram_total_mib = [2000]

        def _boom(*a, **k):
            raise AssertionError("nvidia-smi shelled out from the locked path")

        monkeypatch.setattr(_subprocess, "check_output", _boom)
        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        mgr._log_make_room_starved(_slot("modelA"), "make_room_starved_vram", {"why": "x"})
        assert any("MAKE_ROOM_STARVED" in r.message for r in caplog.records)


class TestDispatchEmptyBoxStagedWork:
    """Part 2: a real dispatch-loop iteration, real queue, real (empty)
    resident registry -- residents=0 and staging_depth>0 produced by
    actually running the production admission code, not hand-typed."""

    @pytest.mark.asyncio
    async def test_fires_when_registry_is_genuinely_empty_and_work_is_staged(
        self, boot_and_runtime, caplog,
    ):
        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 2  # cap>=2 -> the deployed dispatcher path
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._model_residents() == []  # genuinely empty, not asserted-empty

        await mgr.queue.enqueue(_slot("modela"))
        assert mgr.queue.depth()["staging_queue_depth"] == 1

        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        task = asyncio.create_task(mgr._dispatch_loop())
        try:
            await asyncio.sleep(0.05)
        finally:
            mgr._stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        lines = [r.message for r in caplog.records if "DISPATCH_EMPTY_BOX_STAGED_WORK" in r.message]
        assert lines, "expected the empty-box census line to fire on a genuinely staged, resident-less registry"
        assert "residents=0" in lines[0]
        assert "staging_depth=1" in lines[0]

    @pytest.mark.asyncio
    async def test_silent_when_a_resident_already_exists(self, boot_and_runtime, caplog):
        """Negative control: the SAME staged work, but the registry is not
        empty -- the line must not fire (it is not a periodic heartbeat)."""
        from turbohaul.manager import Resident, ResidentState

        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 2
        mgr = TurbohaulManager(boot, runtime)
        mgr._residents["already-here"] = Resident(
            model_tag="already-here", resident_key="already-here",
            state=ResidentState.ACTIVE, last_active_monotonic=1.0,
        )
        await mgr.queue.enqueue(_slot("modela"))

        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        task = asyncio.create_task(mgr._dispatch_loop())
        try:
            await asyncio.sleep(0.05)
        finally:
            mgr._stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        lines = [r.message for r in caplog.records if "DISPATCH_EMPTY_BOX_STAGED_WORK" in r.message]
        assert not lines, f"must not fire with a live resident present: {lines}"

    @pytest.mark.asyncio
    async def test_silent_when_nothing_is_staged(self, boot_and_runtime, caplog):
        """Negative control: empty registry, but nothing staged either --
        the ordinary idle-and-quiet state must stay silent."""
        boot, runtime = boot_and_runtime
        runtime.queue.max_parallel_sidecars = 2
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._model_residents() == []
        assert mgr.queue.depth()["staging_queue_depth"] == 0

        caplog.set_level(logging.INFO, logger="turbohaul.manager")
        task = asyncio.create_task(mgr._dispatch_loop())
        try:
            await asyncio.sleep(0.05)
        finally:
            mgr._stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        lines = [r.message for r in caplog.records if "DISPATCH_EMPTY_BOX_STAGED_WORK" in r.message]
        assert not lines, f"must not fire with nothing staged: {lines}"
