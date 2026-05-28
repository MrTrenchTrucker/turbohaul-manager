"""Tests for state.sqlite schema + helpers (v0.2 §12)."""
import json

import pytest

from turbohaul.state import (
    SCHEMA_VERSION,
    known_active_pids,
    mark_slot_ended,
    open_state_db,
    reconcile_orphaned_slots,
    record_audit_event,
    state_db_session,
    upsert_slot,
    utcnow_iso,
)


class TestStateDb:
    def test_open_creates_tables(self, tmp_path):
        db_path = tmp_path / "state.sqlite"
        conn = open_state_db(db_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = [r["name"] for r in cur.fetchall()]
        assert "slots" in names
        assert "audit_events" in names
        assert "pull_history" in names
        assert "schema_version" in names
        conn.close()

    def test_schema_version_recorded(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            cur = conn.execute("SELECT version FROM schema_version")
            versions = [r["version"] for r in cur.fetchall()]
            assert SCHEMA_VERSION in versions

    def test_idempotent_init(self, tmp_path):
        db_path = tmp_path / "state.sqlite"
        # Open twice - should not double-insert schema_version
        with state_db_session(db_path):
            pass
        with state_db_session(db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) AS n FROM schema_version")
            n = cur.fetchone()["n"]
            assert n == 1

    def test_wal_journal_mode(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            cur = conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            assert mode.lower() == "wal"


class TestSlotOps:
    def test_upsert_insert_then_update(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            upsert_slot(
                conn,
                {
                    "slot_id": "slot-1",
                    "model_tag": "qwen3.6-35b-moe",
                    "state": "STAGED",
                },
            )
            cur = conn.execute("SELECT state FROM slots WHERE slot_id=?", ("slot-1",))
            assert cur.fetchone()["state"] == "STAGED"
            # Update
            upsert_slot(
                conn,
                {
                    "slot_id": "slot-1",
                    "model_tag": "qwen3.6-35b-moe",
                    "state": "ACTIVE",
                    "pid": 12345,
                    "port": 11500,
                },
            )
            cur = conn.execute(
                "SELECT state, pid, port FROM slots WHERE slot_id=?", ("slot-1",)
            )
            row = cur.fetchone()
            assert row["state"] == "ACTIVE"
            assert row["pid"] == 12345
            assert row["port"] == 11500

    def test_client_meta_persisted_as_json(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            upsert_slot(
                conn,
                {
                    "slot_id": "slot-2",
                    "model_tag": "m",
                    "state": "STAGED",
                    "client_meta": {"requester": "agent-a", "audit_id": "abc-123"},
                },
            )
            cur = conn.execute(
                "SELECT client_meta_json FROM slots WHERE slot_id=?", ("slot-2",)
            )
            meta = json.loads(cur.fetchone()["client_meta_json"])
            assert meta["requester"] == "agent-a"
            assert meta["audit_id"] == "abc-123"

    def test_known_active_pids(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            upsert_slot(
                conn, {"slot_id": "s1", "model_tag": "m", "state": "ACTIVE", "pid": 100}
            )
            upsert_slot(
                conn, {"slot_id": "s2", "model_tag": "m", "state": "GRACE", "pid": 200}
            )
            upsert_slot(
                conn, {"slot_id": "s3", "model_tag": "m", "state": "POPPED", "pid": 300}
            )
            pids = known_active_pids(conn)
            assert 100 in pids
            assert 200 in pids
            assert 300 not in pids  # POPPED excluded

    def test_mark_slot_ended(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            upsert_slot(
                conn, {"slot_id": "s1", "model_tag": "m", "state": "ACTIVE", "pid": 100}
            )
            mark_slot_ended(conn, "s1", "test-cleanup")
            cur = conn.execute(
                "SELECT state, end_reason, ended_at FROM slots WHERE slot_id='s1'"
            )
            row = cur.fetchone()
            assert row["state"] == "COLD"
            assert row["end_reason"] == "test-cleanup"
            assert row["ended_at"] is not None


class TestAuditEvents:
    def test_record_event(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            record_audit_event(conn, "test_event", {"foo": "bar"}, slot_id="s1")
            cur = conn.execute(
                "SELECT event_type, payload_json, slot_id FROM audit_events"
            )
            row = cur.fetchone()
            assert row["event_type"] == "test_event"
            assert json.loads(row["payload_json"])["foo"] == "bar"
            assert row["slot_id"] == "s1"

    def test_audit_without_slot_id(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            record_audit_event(conn, "boot_orphan_reaper", {"reaped": 0})
            cur = conn.execute(
                "SELECT event_type, slot_id FROM audit_events WHERE event_type='boot_orphan_reaper'"
            )
            row = cur.fetchone()
            assert row["slot_id"] is None


class TestReconcileOrphanedSlots:
    def test_marks_dead_pids_cold(self, tmp_path):
        with state_db_session(tmp_path / "state.sqlite") as conn:
            upsert_slot(
                conn, {"slot_id": "s1", "model_tag": "m", "state": "ACTIVE", "pid": 100}
            )
            upsert_slot(
                conn, {"slot_id": "s2", "model_tag": "m", "state": "ACTIVE", "pid": 200}
            )
            # Only 100 is "alive"
            n = reconcile_orphaned_slots(conn, live_pids={100})
            assert n == 1
            cur = conn.execute("SELECT state, end_reason FROM slots WHERE slot_id='s2'")
            row = cur.fetchone()
            assert row["state"] == "COLD"
            assert row["end_reason"] == "boot-reconcile-orphaned-pid"

    def test_marks_pid_null_pre_active_orphans_cold(self, tmp_path):
        """Note: slots with pid=NULL in pre-active state are reconciled."""
        with state_db_session(tmp_path / "state.sqlite") as conn:
            upsert_slot(
                conn, {"slot_id": "s_staged", "model_tag": "m", "state": "STAGED"}
            )
            upsert_slot(
                conn, {"slot_id": "s_received", "model_tag": "m", "state": "RECEIVED"}
            )
            upsert_slot(
                conn, {"slot_id": "s_loading_fail", "model_tag": "m", "state": "LOADING_FAIL"}
            )
            # pre-existing COLD slot must NOT be touched again
            upsert_slot(
                conn, {"slot_id": "s_cold", "model_tag": "m", "state": "COLD"}
            )
            n = reconcile_orphaned_slots(conn, live_pids=set())
            assert n == 3
            for sid in ("s_staged", "s_received", "s_loading_fail"):
                cur = conn.execute(
                    "SELECT state, end_reason FROM slots WHERE slot_id=?", (sid,)
                )
                row = cur.fetchone()
                assert row["state"] == "COLD"
                assert row["end_reason"] == "boot-reconcile-pre-active-orphan"
            cur = conn.execute(
                "SELECT state, end_reason FROM slots WHERE slot_id=?", ("s_cold",)
            )
            row = cur.fetchone()
            assert row["state"] == "COLD"
            # No end_reason rewrite on pre-cold slots
            assert row["end_reason"] is None
