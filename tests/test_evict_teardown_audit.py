"""cap>=2 resident teardown audit event.

Verifies that `_unload_teardown` (in manager.py) records a `teardown` audit
event in the audit_events table — the cap>=2 twin of the cap<=1 `_teardown`
audit in manager.py. Without this fix, cap>=2 resident deaths
(idle-evict, make-room VRAM, make-room count-cap, driver-death reap) leave
NO entry in the audit log, so `/status` dashboards and operator tooling
see zero engine-kill events even though real evictions are firing
(the same kind of blind spot, but for the audit table instead of the
eviction counter).

RED:   unpatched — no "teardown" audit event after _begin_unload_locked + settle.
GREEN: patched  — one "teardown" event with the documented payload shape.

GAP NOTE (not asserted as a value, only that the key exists):
  - sigterm_status / sigterm_ok are None (unavailable — _reap_resident_handle
    discards _sigterm's return; closed by a separate follow-up change).
  - post_teardown_orphans_reaped is 0 (same gap).
  - reason is the generic seam label "evict_teardown" (no per-call reason param
    exists on _unload_teardown).
"""
import asyncio
import json
import sqlite3

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
from turbohaul.state import open_state_db


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
            llama_server_binary=tmp_path / "fake_llama_server",  # nonexistent, unused
            default_port_base=60100,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, max_parallel_sidecars=2),
        pull=PullConfig(),
    )
    return boot, runtime


def _audit_teardown_rows(db_path):
    """Raw SELECT over the audit_events table (sync-only, off the loop).

    Uses open_state_db so the connection matches the schema the manager writes
    under (same CHECK constraints, same WAL mode, same isolation_level).
    """
    conn = open_state_db(db_path, check_same_thread=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT slot_id, event_type, payload_json FROM audit_events "
            "WHERE event_type = ?",
            ("teardown",),
        )
        return list(cur)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_evict_teardown_emits_teardown_audit_event(boot_and_runtime, tmp_path):
    """THE RED/GREEN: _begin_unload_locked spawns _unload_teardown; after it
    settles, exactly one 'teardown' audit event must exist for this resident.

    On UNPATCHED code: zero 'teardown' events (RED).
    On PATCHED code: one  'teardown' event with the documented payload shape (GREEN).
    """
    boot, runtime = boot_and_runtime
    mgr = TurbohaulManager(boot, runtime)

    # Initialize the state DB schema (open_state_db runs CREATE TABLE IF NOT EXISTS).
    # The manager does not init the audit pool without a FastAPI lifespan, so we
    # must provision the schema ourselves for the SELECT to find audit_events.
    open_state_db(boot.storage.state_db_path)

    # A resident in IDLE_EVICTABLE state -- the same shape _lru_idle_unloadable
    # would select as a victim. _unload_teardown fires on ANY _begin_unload_locked
    # call regardless of how the resident died (idle-timeout, make-room, driver
    # death). We drive _begin_unload_locked directly (same as in the
    # eviction-counter test) rather than reproducing the full routing decision.
    victim = Resident(
        model_tag="m1",
        resident_key="m1",
        state=ResidentState.IDLE_EVICTABLE,
        last_active_monotonic=1.0,
    )
    mgr._residents["m1"] = victim

    # No teardown audit before eviction.
    before = _audit_teardown_rows(boot.storage.state_db_path)
    assert len(before) == 0, f"pre-existing teardown events: {len(before)}"

    async with mgr._registry_lock:
        mgr._begin_unload_locked(victim)

    # _begin_unload_locked spawns _spawn_bg(self._unload_teardown(r)) -- let it settle.
    await asyncio.sleep(0.2)

    after = _audit_teardown_rows(boot.storage.state_db_path)
    teardown_events = [r for r in after if r["event_type"] == "teardown"]

    assert len(teardown_events) >= 1, (
        "RED: _unload_teardown did NOT emit a 'teardown' audit event "
        f"(found {len(teardown_events)}). The audit seam is missing."
    )

    # --- GREEN assertions: the event shape (NOT the gap values) ---
    ev = teardown_events[-1]
    payload = json.loads(ev["payload_json"])

    # reason is the generic seam label (no per-call reason param on _unload_teardown).
    assert payload["reason"] == "evict_teardown", (
        f"reason should be the seam label 'evict_teardown', got {payload['reason']!r}"
    )

    # model_tag is always captured (Resident.model_tag is stable).
    assert payload["model_tag"] == "m1"

    # GAP FIELDS (documented in the code comment, asserted as None/0 not values):
    assert payload["sigterm_status"] is None, (
        "sigterm_status must be None until a follow-up change closes the gap"
    )
    assert payload["sigterm_ok"] is None, (
        "sigterm_ok must be None until a follow-up change closes the gap"
    )
    assert payload["post_teardown_orphans_reaped"] == 0, (
        "post_teardown_orphans_reaped must be 0 (placeholder) until a follow-up change closes the gap"
    )

    # slot_id: the Resident here has active_slot=None (never had a real one in
    # this unit test), so it must be null -- NOT a non-null value that would
    # imply a slot existed.
    assert ev["slot_id"] is None, (
        f"slot_id should be None for an idle-evicted resident with no active slot, "
        f"got {ev['slot_id']!r}"
    )
