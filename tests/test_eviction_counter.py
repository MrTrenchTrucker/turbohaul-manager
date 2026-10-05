"""The `/status` eviction counter (`_unload_count` /
`_last_unloaded_at_iso`) reported zero after a real resident eviction fired.

Measured, not theorised: `_unload_count`/`_last_unloaded_at_iso` were only ever incremented at the
two client-disconnect QUEUE-SLOT eviction sites (`slot.is_evicted`, dispatcher inline branch and
`_handle_unloaded_slot`) -- never at `_begin_unload_locked`, the sole entry point for every
resident-death path (make-room VRAM eviction, make-room count-cap eviction, designated-victim
idle-timeout, self idle-timeout, driver-death reap). This file drives `_begin_unload_locked` directly
under `_registry_lock` -- the same choke point a `make_room_vram` eviction goes through --
rather than reproducing the full VRAM-over-commit routing decision, since the counter's fix lives
entirely inside that one shared function and a direct call is a more surgical, less flaky proof of
it than driving unrelated routing/gate logic to reach the same line.

Design decision: the fix hooks `_begin_unload_locked` alone, so a driver crash-reap ALSO
increments this counter now -- documented as a known, accepted property, not a discovered one.
"""
import asyncio

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
from turbohaul.manager import TurbohaulManager, Resident, ResidentState


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
            llama_server_binary=tmp_path / "fake_llama_server",  # nonexistent, unused here
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return boot, runtime


@pytest.mark.asyncio
class TestEvictionCounterFiresOnARealEviction:
    async def test_counter_and_timestamp_stay_zero_before_any_eviction(self, boot_and_runtime):
        """Negative control: construction alone (no eviction fires) leaves the
        counter and timestamp at their initial values -- must stay true
        identically before and after the fix, since the fix only touches
        `_begin_unload_locked`, never `__init__`."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._unload_count == 0
        assert mgr._last_unloaded_at_iso is None

    async def test_counter_and_timestamp_update_after_a_real_resident_eviction(
        self, boot_and_runtime,
    ):
        """THE RED: a real resident eviction fires through `_begin_unload_locked`
        (the sole entry point every make-room/staleness/driver-death path
        already funnels through) and the published counter must reflect it."""
        boot, runtime = boot_and_runtime
        mgr = TurbohaulManager(boot, runtime)
        victim = Resident(
            model_tag="m1", resident_key="m1",
            state=ResidentState.IDLE_EVICTABLE, last_active_monotonic=1.0,
        )
        mgr._residents["m1"] = victim
        assert mgr._unload_count == 0
        assert mgr._last_unloaded_at_iso is None

        async with mgr._registry_lock:
            mgr._begin_unload_locked(victim)

        assert mgr._unload_count == 1, (
            "a real MAKE_ROOM_EVICTION-class event fired and the operator's "
            "own watched counter must not still read zero"
        )
        assert mgr._last_unloaded_at_iso is not None

        # The actual /status read-out, not just the internal attributes it reads.
        status = mgr.status_snapshot()
        assert status["evictions"]["total_lifetime"] == 1
        assert status["evictions"]["last_evicted_at"] is not None

        await asyncio.sleep(0.05)  # let the detached teardown bg task settle
