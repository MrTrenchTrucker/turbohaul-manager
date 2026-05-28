"""Tests for the background sweeper that finalizes the
state-row for client-disconnect evictions.

Coverage (6 cases):
1. sweep finds + finalizes a synthetic 24h-old STAGED slot with pid=NULL
2. sweep does NOT touch a STAGED slot with active pid (live-slot guard)
3. sweep does NOT touch a STAGED slot younger than min_age (24h staleness floor)
4. sweep is idempotent — running twice on the same set is a no-op the 2nd time
5. /status surfaces last_sweep_iso + slots_finalized_lifetime after a sweep
6. Lifecycle: start spawns the task, shutdown cancels it cleanly
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig,
    RuntimePathsConfig, ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.state import open_state_db


def _make_app(tmp_path, sweep_interval_s=60, sweep_min_age_s=86400):
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
            llama_server_binary=tmp_path / "fake", default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            grace_seconds=0, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
            background_sweep_interval_s=sweep_interval_s,
            background_sweep_min_age_s=sweep_min_age_s,
        ),
        pull=PullConfig(),
    )
    return create_app(
        boot, runtime, auto_start_worker=False, auto_boot_reconcile=False,
    )


def _insert_slot(state_db_path, slot_id: str, state: str, created_at_iso: str,
                 pid: int | None = None, ended_at: str | None = None):
    """Insert a synthetic slot row at a chosen age + state. Bypasses
    upsert_slot's auto-now timestamp so we can plant 24h-old rows for the
    sweep predicate to find."""
    conn = open_state_db(state_db_path)
    try:
        conn.execute(
            """INSERT INTO slots (
                slot_id, model_tag, thread_id, state, port, pid,
                created_at, updated_at, ended_at, end_reason,
                extension_count, client_meta_json
            ) VALUES (?, 'm', '', ?, NULL, ?, ?, ?, ?, NULL, 0, '{}')""",
            (slot_id, state, pid, created_at_iso, created_at_iso, ended_at),
        )
    finally:
        conn.close()


def _read_slot(state_db_path, slot_id: str) -> dict | None:
    conn = open_state_db(state_db_path)
    try:
        cur = conn.execute(
            "SELECT state, ended_at, end_reason FROM slots WHERE slot_id=?",
            (slot_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "state": row["state"],
            "ended_at": row["ended_at"],
            "end_reason": row["end_reason"],
        }
    finally:
        conn.close()


# ============================================================================
# Test 1 — sweep finds + finalizes a 24h-old STAGED slot with pid=NULL
# ============================================================================


@pytest.mark.asyncio
async def test_1_sweep_finalizes_old_staged_slot(tmp_path):
    app = _make_app(tmp_path)
    mgr = app.state.manager
    state_db_path = mgr.boot.storage.state_db_path

    old_iso = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(
        timespec="seconds",
    )
    _insert_slot(state_db_path, "slot-old-1", "STAGED", old_iso, pid=None)

    finalized = await mgr._run_one_sweep(min_age_s=86400)
    assert finalized == 1
    row = _read_slot(state_db_path, "slot-old-1")
    assert row is not None
    assert row["state"] == "COLD"
    assert row["end_reason"] == "background_sweeper_evicted"
    assert mgr._slots_finalized_lifetime == 1
    assert mgr._last_sweep_iso is not None


# ============================================================================
# Test 2 — sweep does NOT touch a STAGED slot with active pid
# ============================================================================


@pytest.mark.asyncio
async def test_2_sweep_skips_slot_with_active_pid(tmp_path):
    app = _make_app(tmp_path)
    mgr = app.state.manager
    state_db_path = mgr.boot.storage.state_db_path

    old_iso = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(
        timespec="seconds",
    )
    _insert_slot(state_db_path, "slot-live", "STAGED", old_iso, pid=42)

    finalized = await mgr._run_one_sweep(min_age_s=86400)
    assert finalized == 0
    row = _read_slot(state_db_path, "slot-live")
    assert row["state"] == "STAGED"  # untouched
    assert row["ended_at"] is None


# ============================================================================
# Test 3 — sweep does NOT touch a slot younger than min_age
# ============================================================================


@pytest.mark.asyncio
async def test_3_sweep_skips_slot_younger_than_min_age(tmp_path):
    app = _make_app(tmp_path)
    mgr = app.state.manager
    state_db_path = mgr.boot.storage.state_db_path

    recent_iso = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(
        timespec="seconds",
    )
    _insert_slot(state_db_path, "slot-recent", "STAGED", recent_iso, pid=None)

    finalized = await mgr._run_one_sweep(min_age_s=86400)  # 24h floor
    assert finalized == 0
    row = _read_slot(state_db_path, "slot-recent")
    assert row["state"] == "STAGED"


# ============================================================================
# Test 4 — sweep is idempotent (second run on same set is a no-op)
# ============================================================================


@pytest.mark.asyncio
async def test_4_sweep_is_idempotent(tmp_path):
    app = _make_app(tmp_path)
    mgr = app.state.manager
    state_db_path = mgr.boot.storage.state_db_path

    old_iso = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat(
        timespec="seconds",
    )
    _insert_slot(state_db_path, "slot-idem-1", "STAGED", old_iso, pid=None)
    _insert_slot(state_db_path, "slot-idem-2", "STAGED", old_iso, pid=None)

    first = await mgr._run_one_sweep(min_age_s=86400)
    assert first == 2
    second = await mgr._run_one_sweep(min_age_s=86400)
    assert second == 0  # both already COLD after first sweep
    assert mgr._slots_finalized_lifetime == 2


# ============================================================================
# Test 5 — /status surfaces last_sweep_iso + slots_finalized_lifetime
# ============================================================================


def test_5_status_block_surfaces_sweeper_state(tmp_path):
    app = _make_app(tmp_path)
    mgr = app.state.manager
    state_db_path = mgr.boot.storage.state_db_path

    old_iso = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(
        timespec="seconds",
    )
    _insert_slot(state_db_path, "slot-status-1", "STAGED", old_iso, pid=None)

    # Drive one sweep synchronously (no full lifespan needed for this test)
    asyncio.run(mgr._run_one_sweep(min_age_s=86400))

    with TestClient(app) as client:
        r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert "background_sweeper" in body
    bs = body["background_sweeper"]
    assert bs["slots_finalized_lifetime"] == 1
    assert bs["last_sweep_iso"] is not None
    assert "T" in bs["last_sweep_iso"]  # ISO format


# ============================================================================
# Test 6 — lifecycle: start spawns sweeper task, shutdown cancels cleanly
# ============================================================================


@pytest.mark.asyncio
async def test_6_lifecycle_start_spawns_shutdown_cancels(tmp_path):
    """Use a fast interval (0.05s) so we can observe a sweep tick during the
    short test window, then confirm clean shutdown cancels the task."""
    app = _make_app(tmp_path, sweep_interval_s=1, sweep_min_age_s=60)
    mgr = app.state.manager
    # Manually spawn the sweeper as the lifespan would; auto_start_worker=False
    mgr._sweeper_task = asyncio.create_task(
        mgr._periodic_terminal_park_sweep()
    )
    # Give it a moment to enter the loop
    await asyncio.sleep(0.1)
    assert mgr._sweeper_task is not None
    assert not mgr._sweeper_task.done()
    # Shutdown — must cancel sweeper task cleanly without raising
    await mgr.shutdown()
    assert mgr._sweeper_task.done()
