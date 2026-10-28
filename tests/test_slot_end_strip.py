"""When a slot ends, its stored request metadata is reduced to the slot's identity.

A slot row keeps the request body only while the request runs. The write that marks the slot COLD
also rewrites that one row's client_meta_json to the identity keys. Nothing else in the row changes,
other rows are not touched, and a value that is not a JSON object is left exactly as it was.
"""
import json

import pytest

import turbohaul.state as state_mod
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.manager import TurbohaulManager
from turbohaul.state import mark_slot_ended, open_state_db, reconcile_orphaned_slots

IDENTITY = {"ip", "session_id", "role", "is_main", "is_sub_agent", "is_curator", "is_compression", "save_kv"}
OLD = "2025-01-01T00:00:00+00:00"


def _full(ip="192.0.2.9"):
    return {
        "kind": "openai-chat-completion",
        "messages": [{"role": "system", "content": "s" * 3000}, {"role": "user", "content": "hello"}],
        "model": "m", "temperature": 0.2, "tools": [{"type": "function", "function": {"name": "f"}}],
        "ip": ip, "session_id": "sess-1", "role": "main", "is_main": True, "is_sub_agent": False,
        "is_curator": None, "is_compression": False, "save_kv": True,
    }


@pytest.fixture
def conn(tmp_path):
    c = open_state_db(tmp_path / "state.sqlite")
    yield c
    c.close()


def _raw(conn, slot_id, body, *, state="ACTIVE", pid=None, created_at=OLD, updated_at=OLD):
    text = body if (body is None or isinstance(body, str)) else json.dumps(body)
    conn.execute(
        """INSERT INTO slots (slot_id, model_tag, thread_id, state, port, pid, created_at, updated_at,
                              ended_at, end_reason, extension_count, client_meta_json)
           VALUES (?, 'm', 't-' || ?, ?, 11500, ?, ?, ?, NULL, NULL, 2, ?)""",
        (slot_id, slot_id, state, pid, created_at, updated_at, text))


def _row(conn, slot_id):
    return dict(conn.execute("SELECT * FROM slots WHERE slot_id=?", (slot_id,)).fetchone())


def _outcomes():
    counter = getattr(state_mod, "slot_meta_strip_outcomes", None)
    assert counter is not None, "state.py keeps no count of slot metadata strips"
    return counter


def test_ending_a_slot_reduces_its_body_to_the_identity_keys(conn):
    body = _full()
    _raw(conn, "s1", body)
    mark_slot_ended(conn, "s1", "done")
    row = _row(conn, "s1")
    assert row["state"] == "COLD"
    stored = json.loads(row["client_meta_json"])
    assert set(stored) == IDENTITY, f"the stored body must keep exactly the identity keys, got {sorted(stored)}"
    assert stored == {k: body[k] for k in IDENTITY}


def test_ending_a_slot_changes_nothing_else_in_the_row(conn):
    _raw(conn, "s1", _full(), pid=4242)
    before = _row(conn, "s1")
    mark_slot_ended(conn, "s1", "done")
    after = _row(conn, "s1")
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {"state", "ended_at", "end_reason", "updated_at", "client_meta_json"}
    assert after["thread_id"] == "t-s1" and after["port"] == 11500 and after["pid"] == 4242
    assert after["created_at"] == OLD and after["extension_count"] == 2


def test_other_slots_keep_their_full_bodies(conn):
    _raw(conn, "ending", _full("192.0.2.1"))
    _raw(conn, "running", _full("192.0.2.2"))
    mark_slot_ended(conn, "ending", "done")
    assert "messages" in json.loads(_row(conn, "running")["client_meta_json"])
    assert "messages" not in json.loads(_row(conn, "ending")["client_meta_json"])


@pytest.mark.parametrize("text", ["{broken", "[1, 2, 3]", '"just text"', "7"])
def test_a_value_that_is_not_a_json_object_is_left_untouched_and_counted(conn, text):
    _raw(conn, "s1", text)
    invalid_before = _outcomes()["invalid"]
    mark_slot_ended(conn, "s1", "done")
    row = _row(conn, "s1")
    assert row["state"] == "COLD"
    assert row["client_meta_json"] == text, "a value that cannot be read as an object must be left as it was"
    assert _outcomes()["invalid"] - invalid_before == 1


def test_a_null_value_is_left_null(conn):
    _raw(conn, "s1", None)
    mark_slot_ended(conn, "s1", "done")
    row = _row(conn, "s1")
    assert row["state"] == "COLD" and row["client_meta_json"] is None


def test_a_body_that_is_already_identity_only_is_not_rewritten(conn):
    identity = json.dumps({"ip": "192.0.2.1", "is_main": True})
    _raw(conn, "s1", identity)
    unchanged_before = _outcomes()["unchanged"]
    changes_before = conn.total_changes
    mark_slot_ended(conn, "s1", "done")
    assert conn.total_changes - changes_before == 1, "only the state update may write"
    assert _row(conn, "s1")["client_meta_json"] == identity
    assert _outcomes()["unchanged"] - unchanged_before == 1


def test_ending_a_slot_that_does_not_exist_is_harmless(conn):
    mark_slot_ended(conn, "no-such-slot", "done")


def test_a_failure_while_reducing_does_not_stop_the_slot_from_ending(conn, monkeypatch):
    assert hasattr(state_mod, "strip_stored_meta"), "state.py does not reduce stored metadata"
    _raw(conn, "s1", _full())

    def boom(_text):
        raise RuntimeError("cannot reduce")

    monkeypatch.setattr(state_mod, "strip_stored_meta", boom)
    error_before = _outcomes()["error"]
    mark_slot_ended(conn, "s1", "done")
    row = _row(conn, "s1")
    assert row["state"] == "COLD", "ending the slot must not depend on the reduction"
    assert "messages" in json.loads(row["client_meta_json"])
    assert _outcomes()["error"] - error_before == 1


def test_the_boot_reconcile_of_an_orphaned_slot_reduces_it_too(conn):
    _raw(conn, "orphan", _full(), state="ACTIVE", pid=999999)
    assert reconcile_orphaned_slots(conn, set()) == 1
    row = _row(conn, "orphan")
    assert row["state"] == "COLD" and set(json.loads(row["client_meta_json"])) == IDENTITY


@pytest.mark.asyncio
async def test_the_sweeper_that_ends_an_old_staged_slot_reduces_it(tmp_path):
    root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (root / sub).mkdir(parents=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs", manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging", state_db_path=root / "state.sqlite"),
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59700),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    mgr = TurbohaulManager(boot, RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig()))
    c = open_state_db(boot.storage.state_db_path)
    try:
        _raw(c, "old-staged", _full(), state="STAGED", pid=None)
        c.execute("UPDATE slots SET ended_at=NULL WHERE slot_id='old-staged'")
        assert await mgr._run_one_sweep(60) == 1
        row = _row(c, "old-staged")
    finally:
        c.close()
    assert row["state"] == "COLD"
    assert set(json.loads(row["client_meta_json"])) == IDENTITY
