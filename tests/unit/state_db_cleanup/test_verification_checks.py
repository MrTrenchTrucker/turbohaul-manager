"""Each property the verification claims is checked on its own."""
# The SQL in these tests is built from fixed strings; nothing external is interpolated.
# ruff: noqa: S608
import os
import sqlite3

import pytest

from state_db_cleanup.helpers import (
    internals,
    make_fixture,
    quiet_env,
    report_of,
)

checks = internals("checks")
measure = internals("measure")


class TestEachVerificationCheckHasItsOwnTest:
    @pytest.fixture
    def parts(self, tmp_path):
        fx = make_fixture(tmp_path, n_cold=3, pad=500)
        conn = sqlite3.connect(f"file:{fx.db}?immutable=1", uri=True)
        expect = measure._snapshot(conn)
        conn.close()
        copy = tmp_path / "copy.sqlite"
        src, dst = sqlite3.connect(fx.db), sqlite3.connect(copy)
        src.backup(dst)
        src.close()
        dst.close()
        return expect, copy

    def _edit(self, copy, sql):
        c = sqlite3.connect(copy)
        c.execute(sql)
        c.commit()
        c.close()

    def _verify(self, parts, **kw):
        expect, copy = parts
        return checks._verify_copy(copy, expect, quiet_env(), "copy", **kw)

    def test_a_faithful_copy_passes(self, parts):
        result = self._verify(parts)
        assert result["ok"] is True and result["problems"] == []

    def test_an_empty_file_is_reported_as_empty(self, parts):
        expect, copy = parts
        copy.write_bytes(b"")
        assert "file is empty" in checks._verify_copy(copy, expect, quiet_env(), "copy")["problems"]

    def test_a_size_different_from_the_source_is_reported(self, parts):
        expect, copy = parts
        problems = self._verify(parts, check_pages=copy.stat().st_size + 4096)["problems"]
        assert any("differs from the source" in p for p in problems)

    def test_a_schema_difference_is_reported_alone(self, parts):
        self._edit(parts[1], "CREATE INDEX extra_idx ON audit_events(event_type)")
        assert self._verify(parts)["problems"] == ["schema differs"]

    def test_a_missing_row_is_reported_as_a_row_count_difference(self, parts):
        self._edit(parts[1], "DELETE FROM pull_history")
        assert set(self._verify(parts)["problems"]) == {"row_counts differs", "fingerprints differs"}

    def test_a_state_change_is_reported_as_a_state_count_difference(self, parts):
        self._edit(parts[1], "UPDATE slots SET state = 'STAGED' WHERE slot_id = 'cold-000'")
        assert set(self._verify(parts)["problems"]) == {"state_counts differs", "fingerprints differs"}

    def test_a_changed_column_is_reported_as_a_fingerprint_difference_alone(self, parts):
        self._edit(parts[1], "UPDATE slots SET port = 1 WHERE slot_id = 'cold-000'")
        assert self._verify(parts)["problems"] == ["fingerprints differs"]

    def test_a_changed_stored_length_is_reported_alone(self, parts):
        self._edit(parts[1], "UPDATE slots SET client_meta_json = substr(client_meta_json, 1, 50) "
                             "WHERE slot_id = 'cold-000'")
        assert self._verify(parts)["problems"] == ["sum(length(client_meta_json)) differs"]

    def test_a_failed_integrity_check_is_reported(self, parts, monkeypatch):
        monkeypatch.setattr(measure, "_integrity", lambda conn: (False, ["page 3 is damaged"]))
        assert any(p.startswith("integrity_check") for p in self._verify(parts)["problems"])

    def test_a_file_that_cannot_be_read_is_reported_not_raised(self, parts):
        expect, copy = parts
        copy.write_bytes(os.urandom(20_000))
        result = report_of(lambda: checks._verify_copy(copy, expect, quiet_env(), "copy"))
        assert result["ok"] is False and result["problems"]
