"""The boot census rebuild reads a few keys of each stored slot body, never the body.

A slot row stores the whole request body, which is almost all of the state database. The census rebuild
at boot needs only the client address and the class labels. These tests pin that the rebuild asks SQLite
for exactly those keys, that its memory does not grow with the stored bodies, and that the census it
builds is the same one the full body gives.
"""
import contextlib
import json
import re
import tracemalloc
from datetime import datetime

import pytest

import turbohaul.manager as manager_mod
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.kv_classify import _class_from_label
from turbohaul.manager import TurbohaulManager
from turbohaul.state import open_state_db, record_audit_event

MIB = 1024 * 1024


def _make_mgr(tmp_path):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs", manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging", state_db_path=storage_root / "state.sqlite"),
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59500),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    return TurbohaulManager(boot, RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig()))


@pytest.fixture
def mgr(tmp_path):
    return _make_mgr(tmp_path)


def _insert(mgr, slot_id, body, *, thread_id="t", model_tag="m", created_at="2026-01-01T10:00:00+00:00"):
    """Write a slot row with ``body`` stored verbatim (a str, or None for SQL NULL)."""
    conn = open_state_db(mgr.boot.storage.state_db_path)
    try:
        conn.execute(
            """INSERT INTO slots (slot_id, model_tag, thread_id, state, port, pid, created_at, updated_at,
                                  ended_at, end_reason, extension_count, client_meta_json)
               VALUES (?, ?, ?, 'COLD', NULL, NULL, ?, ?, NULL, NULL, 0, ?)""",
            (slot_id, model_tag, thread_id, created_at, created_at, body))
        conn.commit()
    finally:
        conn.close()


def _body(ip, **labels):
    return json.dumps({"ip": ip, "kind": "openai-chat-completion",
                       "messages": [{"role": "user", "content": "hello"}], **labels})


def _census(mgr):
    return {ip: row for ip, row in mgr._fastlane_census.items()}


class TestTheCensusNeverLoadsTheBody:
    def test_the_census_statement_does_not_select_the_stored_body(self, mgr, monkeypatch):
        _insert(mgr, "s1", _body("192.0.2.1"))
        statements = []
        real = manager_mod.state_db_session

        @contextlib.contextmanager
        def spying(path):
            with real(path) as conn:
                conn.set_trace_callback(statements.append)
                yield conn

        monkeypatch.setattr(manager_mod, "state_db_session", spying)
        mgr.boot_reconcile()
        census = [s for s in statements if "FROM slots" in s and "thread_id" in s and "created_at" in s]
        assert len(census) == 1, f"expected one census statement over slots, got {census}"
        select_list = census[0].split(" FROM slots")[0]
        stripped = re.sub(r"json_valid\(client_meta_json\)", "", select_list)
        stripped = re.sub(r"json_extract\(client_meta_json,[^)]*\)", "", stripped)
        assert "client_meta_json" not in stripped, (
            f"the census must select an expression over the body, never the body itself: {select_list}")

    def test_memory_does_not_grow_with_the_size_of_the_stored_bodies(self, tmp_path):
        def peak_for(size_each):
            m = _make_mgr(tmp_path / f"db_{size_each}")
            for i in range(6):
                _insert(m, f"s{i}", json.dumps(
                    {"ip": f"192.0.2.{i + 1}", "messages": [{"role": "user", "content": "x" * size_each}]}))
            tracemalloc.start()
            try:
                m.boot_reconcile()
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            assert len(m._fastlane_census) == 6, "the census must still see every client"
            return peak

        (tmp_path / "db_1000").mkdir()
        (tmp_path / "db_6291456").mkdir()
        small = peak_for(1000)
        big = peak_for(6 * MIB)
        assert big - small < 2 * MIB, (
            f"the census rebuild's memory grew by {(big - small) / MIB:.1f} MiB for 36 MiB of stored bodies; "
            "it must not materialize them")


class TestCensusParity:
    def _seed(self, mgr):
        a, b, c = "192.0.2.1", "192.0.2.2", "192.0.2.3"
        rows = [
            ("s1", _body(a, is_main=True, role="x"), "ta1", "m1", "2026-01-01T10:00:00+00:00"),
            ("s2", _body(a, is_curator=True, is_sub_agent=True), "ta2", "m1", "2026-01-01T12:00:00+00:00"),
            ("s3", _body(a, is_compression=True), "ta3", "m2", "2026-01-01T11:00:00+00:00"),
            ("s4", _body(b, is_sub_agent=1), "tb1", "m1", "2026-02-01T00:00:00+00:00"),
            ("s5", _body(b, is_curator=False, is_main=False), "tb2", "m1", "2026-02-02T00:00:00+00:00"),
            ("s6", _body(b, role="user-message"), "tb3", "m3", "2026-02-03T00:00:00+00:00"),
            ("s7", _body(b, role="not-a-class"), "tb4", "m3", "2026-02-04T00:00:00+00:00"),
            ("s8", _body(c, is_curator="0"), "tc1", "m1", "2026-03-01T00:00:00+00:00"),
            ("s9", _body(c, is_curator=[], is_compression=True), "tc2", "m1", "2026-03-02T00:00:00+00:00"),
            ("s10", _body(c, is_main=0, is_sub_agent={}), "tc3", "m1", "2026-03-03T00:00:00+00:00"),
            ("s11", _body(c), "tc4", "m1", "2026-03-04T00:00:00+00:00"),
        ]
        for slot_id, body, tid, model, created in rows:
            _insert(mgr, slot_id, body, thread_id=tid, model_tag=model, created_at=created)
        for slot_id, no_ip_body in [
            ("n1", None), ("n2", ""), ("n3", "{not json"), ("n4", "{}"),
            ("n5", json.dumps({"ip": None})), ("n6", json.dumps({"ip": ""}))]:
            _insert(mgr, slot_id, no_ip_body)
        conn = open_state_db(mgr.boot.storage.state_db_path)
        try:
            record_audit_event(conn, "active", {}, slot_id="s1")
            record_audit_event(conn, "active", {}, slot_id="s1")
            record_audit_event(conn, "fanout_rider_active", {}, slot_id="s2")
            record_audit_event(conn, "submit", {}, slot_id="s3")
            conn.commit()
        finally:
            conn.close()
        return a, b, c

    def test_every_census_field_is_what_the_full_body_gave(self, mgr):
        a, b, c = self._seed(mgr)
        skipped_before = mgr._fastlane_census_skipped_no_ip
        mgr.boot_reconcile()
        census = _census(mgr)
        assert set(census) == {a, b, c}
        ts = lambda s: datetime.fromisoformat(s).timestamp()

        ra = census[a]
        assert ra["request_count"] == 3 and ra["served_count"] == 3
        assert ra["classes"] == {"main": 1, "curator": 1, "compression": 1}
        assert ra["models"] == {"m1", "m2"} and ra["thread_ids"] == {"ta1", "ta2", "ta3"}
        assert ra["first_seen"] == ts("2026-01-01T10:00:00+00:00")
        assert ra["last_seen"] == ts("2026-01-01T12:00:00+00:00")
        assert ra["container_name"] is None and ra["address"] == a

        rb = census[b]
        assert rb["request_count"] == 4 and rb["served_count"] == 0
        assert rb["classes"] == {"sub-agent": 1, "unclassified": 2, "user-message": 1}
        assert rb["models"] == {"m1", "m3"} and rb["thread_ids"] == {"tb1", "tb2", "tb3", "tb4"}

        rc = census[c]
        assert rc["request_count"] == 4
        assert rc["classes"] == {"curator": 1, "compression": 1, "unclassified": 2}, (
            "a string '0' is truthy, an empty list and an empty object are not: the labels keep their JSON types")
        assert mgr._fastlane_census_skipped_no_ip - skipped_before == 6

    def test_only_the_label_keys_the_census_projects_decide_a_class(self):
        class Recording(dict):
            seen = set()

            def get(self, key, default=None):
                Recording.seen.add(key)
                return super().get(key, default)

        _class_from_label(Recording(is_main=False))
        assert Recording.seen <= {"is_curator", "is_compression", "is_sub_agent", "is_main", "role"}, (
            "the class derivation reads a key the census projection does not carry: the boot census would drift")

    def test_labels_on_a_huge_body_still_classify(self, mgr):
        _insert(mgr, "big", json.dumps({
            "ip": "192.0.2.9", "is_curator": True,
            "messages": [{"role": "user", "content": "y" * (3 * MIB)}]}))
        mgr.boot_reconcile()
        assert _census(mgr)["192.0.2.9"]["classes"] == {"curator": 1}

    def test_a_row_with_a_broken_body_does_not_empty_the_census(self, mgr):
        _insert(mgr, "good", _body("192.0.2.1"))
        _insert(mgr, "bad", "{broken")
        mgr.boot_reconcile()
        assert set(_census(mgr)) == {"192.0.2.1"}, "one malformed body must not fail the whole census"


class TestPinnedDivergences:
    """Two stored shapes that reading the body in Python treated differently. Pinned so the change is visible."""

    def test_a_body_that_is_valid_json_but_not_an_object_is_skipped_not_fatal(self, mgr):
        _insert(mgr, "good", _body("192.0.2.1"))
        _insert(mgr, "list", "[1, 2, 3]")
        before = mgr._fastlane_census_skipped_no_ip
        mgr.boot_reconcile()
        assert set(_census(mgr)) == {"192.0.2.1"}
        assert mgr._fastlane_census_skipped_no_ip - before == 1

    def test_a_body_with_a_nan_literal_is_skipped(self, mgr):
        _insert(mgr, "good", _body("192.0.2.1"))
        _insert(mgr, "nan", '{"ip": "192.0.2.2", "x": NaN}')
        mgr.boot_reconcile()
        assert set(_census(mgr)) == {"192.0.2.1"}
