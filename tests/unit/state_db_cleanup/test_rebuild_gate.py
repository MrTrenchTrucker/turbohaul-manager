"""The rebuilt file must verify before the swap."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import os
import sqlite3
from pathlib import Path

from state_db_cleanup.helpers import (
    assert_stop,
    internals,
    make_fixture,
    quiet_env,
    real_run,
    report_of,
    tracing_env,
)

config = internals("config")
swap = internals("swap")
world = internals("world")


class TestRebuiltFileMustVerifyBeforeTheSwap:
    def _run_with_vacuum(self, tmp_path, tamper):
        fx = make_fixture(tmp_path)
        inode = fx.db.stat().st_ino
        stmts = []

        def vacuum(conn, dest):
            world._default_vacuum(conn, dest)
            tamper(Path(dest))

        report = real_run(fx, tracing_env(stmts, vacuum_fn=vacuum))
        return fx, report, inode

    def _assert_not_swapped(self, fx, report, inode, fragment):
        assert_stop(report, "verify-new", fragment)
        assert fx.db.stat().st_ino == inode, "the database file must not have been replaced"
        assert Path(report["new_file"]).exists(), "the rejected file is left for inspection"
        assert report["db_modified"] is True and "backup" in report

    def _edit(self, sql, *params):
        def tamper(p):
            c = sqlite3.connect(p)
            c.execute(sql, params)
            c.commit()
            c.close()
        return tamper

    def test_a_missing_row_blocks_the_swap(self, tmp_path):
        fx, report, inode = self._run_with_vacuum(
            tmp_path, self._edit("DELETE FROM audit_events WHERE event_id = (SELECT max(event_id) FROM audit_events)"))
        self._assert_not_swapped(fx, report, inode, "row_counts differs")

    def test_a_changed_other_column_blocks_the_swap(self, tmp_path):
        fx, report, inode = self._run_with_vacuum(
            tmp_path, self._edit("UPDATE slots SET port = 7 WHERE slot_id='cold-001'"))
        self._assert_not_swapped(fx, report, inode, "fingerprints differs")

    def test_a_cold_value_that_still_holds_more_than_the_identity_keys_blocks_the_swap(self, tmp_path):
        fx, report, inode = self._run_with_vacuum(
            tmp_path, self._edit("UPDATE slots SET client_meta_json = '{\"ip\":\"1\",\"messages\":[]}' "
                                 "WHERE slot_id='cold-002'"))
        self._assert_not_swapped(fx, report, inode, "still hold more than the identity keys")

    def test_an_altered_untouched_value_blocks_the_swap(self, tmp_path):
        fx, report, inode = self._run_with_vacuum(
            tmp_path, self._edit("UPDATE slots SET client_meta_json = 'changed' WHERE slot_id='cold-bad-json'"))
        self._assert_not_swapped(fx, report, inode, "stored values differ")

    def test_an_untouched_value_turned_into_an_object_blocks_the_swap(self, tmp_path):
        fx, report, inode = self._run_with_vacuum(
            tmp_path, self._edit("UPDATE slots SET client_meta_json = '{}' WHERE slot_id='cold-bad-json'"))
        self._assert_not_swapped(fx, report, inode, "count of untouched")

    def test_an_altered_non_cold_value_blocks_the_swap(self, tmp_path):
        fx, report, inode = self._run_with_vacuum(
            tmp_path, self._edit("UPDATE slots SET client_meta_json = '{}' WHERE slot_id='live-ACTIVE'"))
        self._assert_not_swapped(fx, report, inode, "stored values differ")

    def test_a_changed_autoincrement_counter_blocks_the_swap(self, tmp_path):
        fx, report, inode = self._run_with_vacuum(tmp_path, self._edit("UPDATE sqlite_sequence SET seq = seq + 5"))
        self._assert_not_swapped(fx, report, inode, "fingerprints differs")

    def test_a_corrupt_new_file_blocks_the_swap(self, tmp_path):
        def corrupt(p):
            with open(p, "r+b") as f:
                f.seek(p.stat().st_size // 2)
                f.write(os.urandom(4096))
        fx, report, inode = self._run_with_vacuum(tmp_path, corrupt)
        assert_stop(report, "verify-new")
        assert fx.db.stat().st_ino == inode

    def test_an_empty_new_file_blocks_the_swap(self, tmp_path):
        fx, report, inode = self._run_with_vacuum(tmp_path, lambda p: p.write_bytes(b""))
        assert_stop(report, "verify-new")
        assert fx.db.stat().st_ino == inode

    def test_an_existing_rebuild_target_is_not_overwritten(self, tmp_path):
        fx = make_fixture(tmp_path)
        (fx.data / "state.sqlite.rebuilt-20261024T120000Z").write_bytes(b"keep")
        report = report_of(lambda: real_run(fx))
        assert_stop(report, "rebuild")
        assert (fx.data / "state.sqlite.rebuilt-20261024T120000Z").read_bytes() == b"keep"

    def _meddled(self, tmp_path, sql, params=()):
        """Run `sql` on the tool's own connection after the first batch, as if something else had touched the data."""
        fx = make_fixture(tmp_path)
        conns, done = [], []

        def meddle(_):
            if not done:
                done.append(1)
                conns[0].execute(sql, params)

        report = real_run(fx, quiet_env(sleep=meddle, on_connect=conns.append), batch=4)
        message = assert_stop(report, "check-strip")
        assert not [n for n in os.listdir(fx.data) if "rebuilt" in n], "nothing may be rebuilt after a failed check"
        return message

    def test_another_column_changing_during_the_strip_is_caught_before_the_rebuild(self, tmp_path):
        msg = self._meddled(tmp_path, "UPDATE slots SET port = 1 WHERE slot_id = 'live-ACTIVE'")
        assert "a column other than client_meta_json changed" in msg

    def test_a_non_cold_value_changing_during_the_strip_is_caught(self, tmp_path):
        msg = self._meddled(tmp_path, "UPDATE slots SET client_meta_json = '{}' WHERE slot_id = 'live-ACTIVE'")
        assert "a row that is not COLD changed" in msg

    def test_a_row_appearing_during_the_strip_is_caught(self, tmp_path):
        msg = self._meddled(
            tmp_path, "INSERT INTO slots (slot_id, model_tag, state, created_at, updated_at) "
                      "VALUES ('late','m','STAGED','x','x')")
        assert "row counts changed" in msg

    def test_a_state_change_during_the_strip_is_caught(self, tmp_path):
        msg = self._meddled(tmp_path, "UPDATE slots SET state = 'STAGED' WHERE slot_id = 'live-ACTIVE'")
        assert "counts by state changed" in msg

    def test_a_value_that_changed_between_the_plan_and_the_strip_is_caught(self, tmp_path):
        msg = self._meddled(tmp_path, "UPDATE slots SET client_meta_json = 'junk' WHERE slot_id = 'cold-005'")
        assert "planned" in msg and "done" in msg

    def test_the_rebuild_runs_with_a_pinned_temp_directory_on_the_database_volume(self, tmp_path):
        fx = make_fixture(tmp_path)
        seen = {}

        def vacuum(conn, dest):
            tmp = os.environ.get("SQLITE_TMPDIR", "")
            seen["tmp"] = tmp
            seen["exists"] = os.path.isdir(tmp)
            world._default_vacuum(conn, dest)

        stmts = []
        real_run(fx, tracing_env(stmts, vacuum_fn=vacuum))
        assert seen["exists"] and Path(seen["tmp"]).parent == fx.data
        assert Path(seen["tmp"]).name.startswith(config._TMP_PREFIX)
        order = [i for i, s in enumerate(stmts) if "temp_store=FILE" in s]
        into = [i for i, s in enumerate(stmts) if s.upper().startswith("VACUUM INTO")]
        assert order and into and order[0] < into[0], "temp storage must be pinned to files before the rebuild"
