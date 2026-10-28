"""A background pass reduces the stored metadata of old ended (COLD) slots to their identity keys.

``kv.cold_meta_purge_age_hours`` turns it on (0, the default, is off). A run handles a bounded number
of rows after a watermark, never deletes, writes only client_meta_json, leaves recent COLD rows and
every other state alone, and leaves a value that is not a JSON object as it was.
"""
import asyncio
import json
import tracemalloc
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import turbohaul.manager as manager_mod
from turbohaul.config import (
    BootConfig, KVConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.manager import TurbohaulManager
from turbohaul.state import open_state_db

IDENTITY = {"ip", "session_id", "role", "is_main", "is_sub_agent", "is_curator", "is_compression", "save_kv"}
OLD = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds")
RECENT = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(timespec="seconds")
MIB = 1024 * 1024


def _make_mgr(tmp_path):
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
    return TurbohaulManager(boot, RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig()))


@pytest.fixture
def mgr(tmp_path):
    return _make_mgr(tmp_path)


def _body(ip="192.0.2.9", pad=2000, **labels):
    return {"kind": "openai-chat-completion", "messages": [{"role": "user", "content": "x" * pad}],
            "tools": [{"type": "function"}], "model": "m", "ip": ip, "session_id": "sess", "role": "main",
            "save_kv": True, **labels}


def _insert(mgr, slot_id, body, *, state="COLD", updated_at=OLD):
    text = body if (body is None or isinstance(body, str)) else json.dumps(body)
    conn = open_state_db(mgr.boot.storage.state_db_path)
    try:
        conn.execute(
            """INSERT INTO slots (slot_id, model_tag, thread_id, state, port, pid, created_at, updated_at,
                                  ended_at, end_reason, extension_count, client_meta_json)
               VALUES (?, 'm', 't-' || ?, ?, 11500, NULL, ?, ?, ?, 'done', 1, ?)""",
            (slot_id, slot_id, state, OLD, updated_at, OLD, text))
    finally:
        conn.close()


def _rows(mgr):
    conn = open_state_db(mgr.boot.storage.state_db_path)
    try:
        return {r["slot_id"]: dict(r) for r in conn.execute("SELECT * FROM slots ORDER BY rowid").fetchall()}
    finally:
        conn.close()


def _is_identity_only(row):
    try:
        return set(json.loads(row["client_meta_json"])) <= IDENTITY
    except (TypeError, ValueError):
        return False


def _set_age(mgr, hours):
    assert "cold_meta_purge_age_hours" in KVConfig.model_fields, "the setting kv.cold_meta_purge_age_hours does not exist"
    mgr.runtime.kv.cold_meta_purge_age_hours = hours


def _sweep(mgr, hours=1.0):
    run = getattr(mgr, "_cold_meta_sweep_once", None)
    assert run is not None, "the manager has no retention pass over ended slots"
    return run(hours)


class TestTheSetting:
    def test_it_exists_and_is_off_by_default(self):
        assert "cold_meta_purge_age_hours" in KVConfig.model_fields, "the setting does not exist"
        assert KVConfig().cold_meta_purge_age_hours == 0.0

    def test_it_accepts_zero_and_fractions_and_rejects_negatives(self):
        assert "cold_meta_purge_age_hours" in KVConfig.model_fields, "the setting does not exist"
        assert KVConfig(cold_meta_purge_age_hours=0).cold_meta_purge_age_hours == 0
        assert KVConfig(cold_meta_purge_age_hours=0.5).cold_meta_purge_age_hours == 0.5
        with pytest.raises(ValidationError):
            KVConfig(cold_meta_purge_age_hours=-1)

    def test_the_run_size_and_the_idle_time_after_a_full_pass_are_fixed(self):
        assert getattr(TurbohaulManager, "_COLD_META_BATCH", None) == 500
        assert getattr(TurbohaulManager, "_COLD_META_IDLE_AFTER_PASS_S", None) == 3600


class TestWhatTheRunTouches:
    def _seed(self, mgr):
        _insert(mgr, "old-cold", _body("192.0.2.1"))
        _insert(mgr, "recent-cold", _body("192.0.2.2"), updated_at=RECENT)
        for state in ("ACTIVE", "STAGED", "IDLE_HOT", "LOADING", "POPPED"):
            _insert(mgr, f"old-{state.lower()}", _body("192.0.2.3"), state=state)
        _insert(mgr, "old-cold-identity", {"ip": "192.0.2.4", "is_main": True})
        _insert(mgr, "old-cold-bad", "{broken")
        _insert(mgr, "old-cold-list", "[1, 2]")
        _insert(mgr, "old-cold-null", None)

    def test_old_cold_rows_are_reduced_and_nothing_else_is(self, mgr):
        self._seed(mgr)
        before = _rows(mgr)
        counts = _sweep(mgr)
        after = _rows(mgr)
        assert _is_identity_only(after["old-cold"])
        assert json.loads(after["old-cold"]["client_meta_json"])["ip"] == "192.0.2.1"
        for untouched in ("recent-cold", "old-active", "old-staged", "old-idle_hot", "old-loading", "old-popped",
                          "old-cold-identity", "old-cold-bad", "old-cold-list", "old-cold-null"):
            assert after[untouched] == before[untouched], f"{untouched} must not change"
        assert counts["stripped"] == 1 and counts["unchanged"] == 1 and counts["invalid"] == 2

    def test_only_the_body_column_is_written_and_no_row_is_removed(self, mgr):
        self._seed(mgr)
        before = _rows(mgr)
        _sweep(mgr)
        after = _rows(mgr)
        assert list(after) == list(before), "no row may be added or removed, and the order must hold"
        for slot_id, row in before.items():
            changed = {k for k in row if row[k] != after[slot_id][k]}
            assert changed <= {"client_meta_json"}, f"{slot_id} changed {changed - {'client_meta_json'}}"
        assert after["old-cold"]["updated_at"] == OLD, "updated_at must keep its value"

    def test_a_value_that_is_not_a_json_object_is_counted_and_logged_once_per_run(self, mgr, caplog):
        _insert(mgr, "bad1", "{broken")
        _insert(mgr, "bad2", "[1]")
        _insert(mgr, "good", _body())
        _set_age(mgr, 1.0)
        caplog.set_level("INFO")
        counts = asyncio.run(mgr._maybe_sweep_cold_meta())
        assert counts["invalid"] == 2 and counts["stripped"] == 1
        lines = [r.getMessage() for r in caplog.records if "cold slot metadata sweep" in r.getMessage()]
        assert len(lines) == 1 and "invalid=2" in lines[0]
        assert mgr._cold_meta_invalid_lifetime == 2 and mgr._cold_meta_stripped_lifetime == 1

    def test_the_boot_census_is_the_same_before_and_after_the_pass(self, mgr):
        labelled = [("192.0.2.1", {"is_main": True}), ("192.0.2.2", {"is_curator": True}),
                    ("192.0.2.3", {"role": "user-message"}), ("192.0.2.4", {})]
        for i, (ip, labels) in enumerate(labelled):
            _insert(mgr, f"s{i}", _body(ip, **labels))
        mgr.boot_reconcile()
        before = {ip: row["classes"] for ip, row in mgr._fastlane_census.items()}
        _sweep(mgr)
        assert all(_is_identity_only(r) for r in _rows(mgr).values())
        mgr.boot_reconcile()
        after = {ip: row["classes"] for ip, row in mgr._fastlane_census.items()}
        assert before == after and len(after) == 4


class TestTheBoundAndTheWatermark:
    def test_a_run_handles_at_most_the_batch_and_the_next_run_resumes(self, mgr):
        mgr._COLD_META_BATCH = 5
        for i in range(12):
            _insert(mgr, f"s{i:02d}", _body(f"192.0.2.{i + 1}"))
        first = _sweep(mgr)
        assert (first["rows"], first["stripped"], first["pass_done"]) == (5, 5, False)
        assert sum(_is_identity_only(r) for r in _rows(mgr).values()) == 5
        second = _sweep(mgr)
        assert (second["rows"], second["stripped"], second["pass_done"]) == (5, 5, False)
        third = _sweep(mgr)
        assert (third["rows"], third["stripped"], third["pass_done"]) == (2, 2, True)
        assert all(_is_identity_only(r) for r in _rows(mgr).values())
        assert mgr._cold_meta_watermark == 0, "a finished pass starts over from the beginning next time"

    def test_a_row_that_was_too_young_is_picked_up_by_a_later_pass(self, mgr):
        _insert(mgr, "young", _body(), updated_at=RECENT)
        assert _sweep(mgr)["stripped"] == 0
        conn = open_state_db(mgr.boot.storage.state_db_path)
        try:
            conn.execute("UPDATE slots SET updated_at=? WHERE slot_id='young'", (OLD,))
        finally:
            conn.close()
        assert _sweep(mgr)["stripped"] == 1


class TestWhenItRuns:
    def test_age_zero_means_off(self, mgr):
        _insert(mgr, "old-cold", _body())
        before = _rows(mgr)
        _set_age(mgr, 0)
        assert asyncio.run(mgr._maybe_sweep_cold_meta()) is None
        assert _rows(mgr) == before

    def test_a_positive_age_runs_the_pass_in_a_worker_thread(self, mgr, monkeypatch):
        _insert(mgr, "old-cold", _body())
        _set_age(mgr, 1.0)
        threaded = []
        real = asyncio.to_thread

        async def spy(fn, *a, **k):
            threaded.append(getattr(fn, "__name__", str(fn)))
            return await real(fn, *a, **k)

        monkeypatch.setattr(manager_mod.asyncio, "to_thread", spy)
        counts = asyncio.run(mgr._maybe_sweep_cold_meta())
        assert counts["stripped"] == 1
        assert threaded == ["_cold_meta_sweep_once"], f"the pass must run off the event loop, ran: {threaded}"

    def test_after_a_full_pass_the_next_run_waits_out_the_idle_time(self, mgr):
        _insert(mgr, "a", _body())
        _set_age(mgr, 1.0)
        first = asyncio.run(mgr._maybe_sweep_cold_meta())
        assert first["pass_done"] is True
        _insert(mgr, "b", _body())
        assert asyncio.run(mgr._maybe_sweep_cold_meta()) is None, "the idle time after a full pass was not honoured"
        assert not _is_identity_only(_rows(mgr)["b"])
        mgr._cold_meta_next_pass_monotonic = 0.0
        assert asyncio.run(mgr._maybe_sweep_cold_meta())["stripped"] == 1

    def test_the_age_is_read_fresh_on_every_run(self, mgr):
        conn_age = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(timespec="seconds")
        _insert(mgr, "two-hours", _body(), updated_at=conn_age)
        _set_age(mgr, 24.0)
        assert asyncio.run(mgr._maybe_sweep_cold_meta())["stripped"] == 0
        _set_age(mgr, 1.0)
        mgr._cold_meta_next_pass_monotonic = 0.0
        assert asyncio.run(mgr._maybe_sweep_cold_meta())["stripped"] == 1

    @pytest.mark.asyncio
    async def test_after_a_partial_run_the_next_run_follows_without_waiting(self, mgr):
        mgr._COLD_META_BATCH = 5
        for i in range(12):
            _insert(mgr, f"s{i:02d}", _body(f"192.0.2.{i + 1}"))
        _set_age(mgr, 1.0)
        first = await mgr._maybe_sweep_cold_meta()
        assert first["pass_done"] is False and first["stripped"] == 5
        second = await mgr._maybe_sweep_cold_meta()
        assert second is not None, "a pass that has not reached the end must not wait out the idle time"
        assert second["stripped"] == 5

    @pytest.mark.asyncio
    async def test_the_sweeper_tick_runs_the_pass(self, mgr, monkeypatch):
        assert hasattr(mgr, "_maybe_sweep_cold_meta"), "the manager has no retention pass over ended slots"
        calls = []

        async def spy():
            calls.append(1)

        async def stop_after_first_job(*a, **k):
            mgr._stop_event.set()
            return 0

        async def nothing(*a, **k):
            return 0

        monkeypatch.setattr(mgr, "_maybe_sweep_cold_meta", spy)
        monkeypatch.setattr(mgr, "_run_one_sweep", nothing)
        monkeypatch.setattr(mgr, "_reconcile_orphaned_vram_credit", stop_after_first_job)
        monkeypatch.setattr(mgr, "_gc_kv_cache", nothing)
        monkeypatch.setattr(mgr, "_maybe_purge_mismatched_bins", lambda *a, **k: None)
        await asyncio.wait_for(mgr._periodic_terminal_park_sweep(), timeout=20)
        assert calls == [1], "the periodic sweeper tick must run the cold metadata pass once per tick"


class TestMemory:
    def test_memory_stays_small_with_large_stored_bodies(self, mgr):
        for i in range(10):
            _insert(mgr, f"big{i}", _body(f"192.0.2.{i + 1}", pad=3 * MIB))
        run = getattr(mgr, "_cold_meta_sweep_once", None)
        assert run is not None, "the manager has no retention pass over ended slots"
        tracemalloc.start()
        try:
            counts = run(1.0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert counts["stripped"] == 10
        assert peak < 15 * MIB, (
            f"the pass held {peak / MIB:.1f} MiB for 30 MiB of stored bodies; it must read one body at a time")
