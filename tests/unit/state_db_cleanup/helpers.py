"""Shared fixtures and helpers for the state_db_cleanup tests.

The cleanup rewrites a large state database, so the tests pin its gates as much as its result: nothing is
written before the backup has been verified, a failed check stops the run before the next step, a dry run
writes nothing, and only the stored request value of COLD rows is ever changed. Every test uses a scratch
database built with the real schema.
"""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import hashlib
import importlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import turbohaul.slot_meta_projection as projection
import turbohaul.state_db_cleanup as cleanup
from turbohaul.state import open_state_db

TOOL = Path(__file__).resolve().parents[3] / "tools" / "state_db_cleanup.py"

OLD = "2026-01-01T00:00:00+00:00"
NOW = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC)
RECENT = (NOW - timedelta(minutes=5)).isoformat(timespec="seconds")
MIB = 1024 * 1024
IDENTITY = set(projection.IDENTITY_KEYS)


def _body(i, pad, **extra):
    return json.dumps({
        "ip": f"192.0.2.{i % 250}", "session_id": f"sess-{i}", "role": "main", "is_main": True,
        "save_kv": True, "kind": "chat", "model": "m", "tools": [{"type": "function"}],
        "messages": [{"role": "user", "content": "x" * pad}], **extra})


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.data = self.root / "data"
        self.backups = self.root / "backups"
        self.data.mkdir()
        self.backups.mkdir()
        self.db = self.data / "state.sqlite"
        self.cold_ids = []
        self.invalid = {}
        self.other = {}


def make_fixture(tmp_path, n_cold=12, pad=40_000, extra_audit=40):
    fx = Fixture(tmp_path)
    c = open_state_db(fx.db)
    try:
        def ins(slot_id, state, body, updated=OLD):
            c.execute(
                "INSERT INTO slots (slot_id, model_tag, thread_id, state, port, pid, created_at, updated_at, "
                "ended_at, end_reason, extension_count, client_meta_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (slot_id, "m", "t-" + slot_id, state, 11500, None, OLD, updated, OLD if state == "COLD" else None,
                 "done" if state == "COLD" else None, 2, body))

        for i in range(n_cold):
            fx.cold_ids.append(f"cold-{i:03d}")
            ins(f"cold-{i:03d}", "COLD", _body(i, pad))
        for name, value in (("bad-json", "{broken"), ("array", "[1, 2, 3]"), ("scalar", "7"),
                            ("text", '"just text"'), ("null", None)):
            fx.invalid[f"cold-{name}"] = value
            ins(f"cold-{name}", "COLD", value)
        ident = json.dumps(projection.identity_client_meta({"ip": "192.0.2.77", "is_main": True}))
        ins("cold-identity", "COLD", ident)
        fx.other["cold-identity"] = ident
        for state in ("ACTIVE", "STAGED", "IDLE_HOT", "LOADING"):
            body = _body(99, pad)
            fx.other[f"live-{state}"] = body
            ins(f"live-{state}", state, body)
        for i in range(extra_audit):
            c.execute("INSERT INTO audit_events (slot_id, event_type, payload_json, occurred_at) VALUES (?,?,?,?)",
                      (f"cold-{i % max(n_cold, 1):03d}", "evt", json.dumps({"i": i}), OLD))
        c.execute("INSERT INTO pull_history (requester, url, status, started_at) VALUES ('r','u','done',?)", (OLD,))
    finally:
        c.close()
    return fx


def sub(root, name):
    p = Path(root) / name
    p.mkdir()
    return p


def cfg_for(fx, **kw):
    base = dict(db=fx.db, backup_dir=fx.backups, execute=False, sleep_ms=0, margin_bytes=0)
    base.update(kw)
    return cleanup.Config(**base)


def quiet_env(**kw):
    kw.setdefault("say", lambda m: None)
    kw.setdefault("now", lambda: NOW)
    return cleanup.Env(**kw)


def md5(path):
    return hashlib.md5(Path(path).read_bytes(), usedforsecurity=False).hexdigest()


def read_all(db):
    """Every row of every table, read independently of the tool, keyed for comparison."""
    conn = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    try:
        out = {}
        out["slots_other"] = conn.execute(
            "SELECT slot_id, model_tag, thread_id, state, port, pid, engine_starttime, created_at, updated_at, "
            "ended_at, end_reason, extension_count FROM slots ORDER BY slot_id").fetchall()
        out["bodies"] = dict(conn.execute("SELECT slot_id, client_meta_json FROM slots").fetchall())
        out["audit"] = conn.execute("SELECT * FROM audit_events ORDER BY event_id").fetchall()
        out["pulls"] = conn.execute("SELECT * FROM pull_history ORDER BY pull_id").fetchall()
        out["seq"] = conn.execute("SELECT name, seq FROM sqlite_sequence ORDER BY name").fetchall()
        out["schema"] = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
        out["integrity"] = conn.execute("PRAGMA integrity_check").fetchall()
        return out
    finally:
        conn.close()


def real_run(fx, env=None, **kw):
    return cleanup.run(cfg_for(fx, execute=True, **kw), env or quiet_env())


def report_of(call):
    """Run `call`; a raise is a failed check (the tool must end with a report, not a traceback)."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - any raise is the failure being reported
        raise AssertionError(f"expected the run to end with a report, but it raised {exc!r}") from exc


def assert_stop(report, step, fragment=None):
    """The run ended at this gate. Written as assertions so a missing stop reads as a failed check, not a KeyError."""
    stop = report.get("stop") or {}
    assert report["status"] == "stopped" and stop.get("step") == step, (report["status"], stop)
    if fragment is not None:
        assert fragment in stop.get("message", ""), stop.get("message")
    return stop.get("message", "")


def tracing_env(stmts, **kw):
    return quiet_env(on_connect=lambda conn: conn.set_trace_callback(stmts.append), **kw)


def stopped_cleanly(fx, report, step, md5_before, stmts, backups_ok=False):
    assert_stop(report, step)
    assert md5(fx.db) == md5_before, f"the database changed although the run stopped at {step}"
    assert not [s for s in stmts if s.lstrip().upper().startswith(("UPDATE", "VACUUM"))]
    assert report["db_modified"] is False
    if not backups_ok:
        assert not [p for p in fx.backups.iterdir() if ".pre-cleanup-" in p.name and not p.name.endswith(".partial")]


def internals(name):
    """One of the module's own files, for the tests that pin a gate inside it (patch a step, call one check).

    Looked up by name rather than imported: the package entry point is the interface, and these tests
    reach past it on purpose, to prove a single gate fails on its own.
    """
    return importlib.import_module(f"turbohaul.state_db_cleanup.{name}")
