"""The strip pass: batches, pauses, checkpoints and atomic batches."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import json
import sqlite3
import tracemalloc
from datetime import timedelta

import pytest

from state_db_cleanup.helpers import (
    IDENTITY,
    MIB,
    NOW,
    internals,
    make_fixture,
    quiet_env,
    read_all,
    real_run,
    sub,
    tracing_env,
)

strip = internals("strip")


class TestBatchingAndPauses:
    def _many(self, tmp_path, n=45):
        return make_fixture(tmp_path, n_cold=n, pad=200, extra_audit=3)

    def test_rows_are_taken_batch_by_batch_with_a_pause_after_each(self, tmp_path):
        fx = self._many(tmp_path, 12)
        pauses = []
        report = real_run(fx, quiet_env(sleep=pauses.append), batch=5, sleep_ms=50)
        assert report["status"] == "completed", report.get("stop")
        total = report["strip"]["rows"]
        assert report["strip"]["batches"] == -(-total // 5) > 1
        assert pauses == [0.05] * report["strip"]["batches"]

    def test_each_batch_is_one_transaction(self, tmp_path):
        fx = self._many(tmp_path, 12)
        stmts = []
        report = real_run(fx, tracing_env(stmts), batch=5)
        begins = sum(1 for s in stmts if s == "BEGIN IMMEDIATE")
        commits = sum(1 for s in stmts if s == "COMMIT")
        assert begins == report["strip"]["batches"] > 1
        assert commits == begins, "every batch must be committed on its own"
        assert not [s for s in stmts if s == "ROLLBACK"]

    def test_a_checkpoint_follows_every_n_batches(self, tmp_path):
        issued = {}
        reported = {}
        for every in (20, 100000):
            fx = self._many(sub(tmp_path, f"every{every}"))
            stmts = []
            report = real_run(fx, tracing_env(stmts), batch=1, checkpoint_every=every)
            assert report["status"] == "completed", report.get("stop")
            issued[every] = sum(1 for s in stmts if "wal_checkpoint" in s)
            reported[every] = (report["strip"]["batches"], report["strip"]["checkpoints"])
        batches, done = reported[20]
        assert batches >= 41, "the fixture must be big enough to cross two checkpoint boundaries"
        assert done == batches // 20 == 2
        assert reported[100000][1] == 0
        assert issued[20] - issued[100000] == 2, "each reported checkpoint must be a real PRAGMA wal_checkpoint"

    def test_a_failure_part_way_through_a_batch_leaves_that_batch_unchanged(self, tmp_path, monkeypatch):
        fx = make_fixture(tmp_path, n_cold=12, pad=500)
        before = read_all(fx.db)["bodies"]
        real = strip._classify
        calls = []

        def fails_on_the_tenth(stored):
            calls.append(1)
            # the first pass (planning) reads every row; fail only while applying, part way through the batch
            if len(calls) == 22 + 7:
                raise RuntimeError("boom")
            return real(stored)

        monkeypatch.setattr(strip, "_classify", fails_on_the_tenth)
        report = real_run(fx, batch=100)
        assert report["status"] == "failed" and report["failure"]["type"] == "RuntimeError"
        assert read_all(fx.db)["bodies"] == before, "a batch must be committed whole or not at all"

    def test_the_age_cutoff_is_fixed_once_so_every_pass_chooses_the_same_rows(self, tmp_path):
        fx = make_fixture(tmp_path, n_cold=4, pad=500)
        edge = (NOW - timedelta(hours=1) + timedelta(minutes=5)).isoformat(timespec="seconds")
        conn = sqlite3.connect(fx.db)
        conn.execute("UPDATE slots SET updated_at=? WHERE slot_id='cold-000'", (edge,))
        conn.commit()
        conn.close()
        ticks = []

        def advancing_clock():
            ticks.append(1)
            return NOW + timedelta(minutes=10 * (len(ticks) - 1))

        report = real_run(fx, quiet_env(now=advancing_clock), min_age_hours=1.0)
        assert report["status"] == "completed", report.get("stop")
        assert len(ticks) >= 3, "the clock must have moved on between the passes"
        edge_keys = set(json.loads(read_all(fx.db)["bodies"]["cold-000"]))
        assert edge_keys - IDENTITY, "the edge row is younger than the cutoff, so it must keep its full value"

    def test_memory_stays_small_while_the_values_are_large(self, tmp_path):
        fx = make_fixture(tmp_path, n_cold=8, pad=3 * MIB, extra_audit=2)
        tracemalloc.start()
        try:
            report = real_run(fx)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert report["status"] == "completed", report.get("stop")
        assert peak < 30 * MIB, (
            f"peak {peak / MIB:.1f} MiB for ~{8 * 3 + 12} MiB of stored values; values must be read one at a time")

    def test_an_interrupted_run_leaves_a_consistent_database_and_a_rerun_finishes_the_job(self, tmp_path):
        fx = self._many(sub(tmp_path, "a"), 12)
        ref = self._many(sub(tmp_path, "b"), 12)
        calls = []

        def interrupt(_):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            real_run(fx, quiet_env(sleep=interrupt), batch=3)
        mid = read_all(fx.db)
        assert mid["integrity"] == [("ok",)]
        reduced = [s for s, b in mid["bodies"].items() if s.startswith("cold-0") and set(json.loads(b)) <= IDENTITY]
        assert 0 < len(reduced) < 12, "the interruption should land part way through"
        assert len(mid["slots_other"]) == len(read_all(ref.db)["slots_other"])
        # a new --execute refuses while the unfinished-run marker is there; the operator looks, then deletes it
        (fx.data / "state.sqlite.cleanup-unfinished").unlink()
        real_run(fx, quiet_env(now=lambda: NOW + timedelta(seconds=9)))
        real_run(ref)
        assert read_all(fx.db)["bodies"] == read_all(ref.db)["bodies"]
