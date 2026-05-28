"""Tests for the audit-write connection pool.

2 cases:
1. test_concurrent_no_busy_under_load — N=100 concurrent threads issue
   record_audit_event through audit_db_session(); wall budget <2s + zero
   SQLITE_BUSY surfaces + autocommit visibility (post-INSERT raw-conn SELECT
   sees the row without an explicit commit).
2. test_lifespan_init_shutdown — TestClient lifespan init creates the pool,
   shutdown closes it cleanly (no leaked conn).
"""
import asyncio
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

import turbohaul.state as state_mod
from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig,
    RuntimePathsConfig, ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.state import (
    audit_db_session, close_audit_pool, init_audit_pool, record_audit_event,
)


@pytest.fixture
def fresh_db(tmp_path):
    """A clean state.sqlite path with the audit pool guaranteed-closed before/after."""
    close_audit_pool()  # in case a prior test left it open
    db_path = tmp_path / "state.sqlite"
    yield db_path
    close_audit_pool()


def test_concurrent_no_busy_under_load(fresh_db):
    """N=100 concurrent threads write audit events through the pool.

    Asserts: wall budget <2s, zero SQLITE_BUSY/SQLITE_LOCKED errors, and
    autocommit visibility (raw second-conn SELECT sees inserted rows w/o commit).
    """
    init_audit_pool(fresh_db)
    n = 100
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            barrier.wait()
            with audit_db_session() as conn:
                record_audit_event(
                    conn, "concurrent_test", {"i": i}, slot_id=f"slot-{i}"
                )
        except BaseException as e:  # capture for surface in main thread
            with errors_lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    assert errors == [], f"writer errors under concurrent load: {errors[:3]}"
    assert wall < 2.0, f"wall {wall:.3f}s exceeds 2s budget (lock-contention regression?)"

    # Autocommit visibility check: open a SECOND raw conn and SELECT —
    # rows must be visible without an explicit commit from the pool conn.
    raw = sqlite3.connect(str(fresh_db), isolation_level=None)
    try:
        raw.execute("PRAGMA journal_mode=WAL")
        cur = raw.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = ?",
            ("concurrent_test",),
        )
        count = cur.fetchone()[0]
    finally:
        raw.close()
    assert count == n, (
        f"autocommit visibility broke: raw SELECT saw {count}/{n} rows "
        "(post-INSERT row not visible without explicit commit?)"
    )


def test_lifespan_init_shutdown(tmp_path):
    """TestClient enters lifespan startup → init_audit_pool sets _audit_conn;
    lifespan shutdown → close_audit_pool clears it. No leaked conn."""
    close_audit_pool()  # baseline clean

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
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    app = create_app(
        boot, runtime, auto_start_worker=False, auto_boot_reconcile=True,
    )

    assert state_mod._audit_conn is None, "pool dirty before lifespan startup"
    with TestClient(app) as client:
        # Inside lifespan: pool MUST be initialized
        assert state_mod._audit_conn is not None, (
            "init_audit_pool was not called during lifespan startup"
        )
        # And the conn is usable
        r = client.get("/health")
        assert r.status_code == 200
    # After lifespan exit: pool MUST be closed (no leak)
    assert state_mod._audit_conn is None, (
        "close_audit_pool was not called during lifespan shutdown — conn leaked"
    )


def test_sync_only_guard_fires_in_async_context(fresh_db):
    """Bonus F1 boundary: audit_db_session called from async context raises.

    Documents the sync-only contract; if someone removes the guard, this test
    surfaces the regression.
    """
    init_audit_pool(fresh_db)

    async def _bad_call():
        with audit_db_session() as conn:
            record_audit_event(conn, "should-not-fire", {})

    with pytest.raises(RuntimeError, match="sync-only"):
        asyncio.run(_bad_call())
