"""kv_stale_pairs: the skip reasons that depend on the directory, the echoed arguments, and never raising."""

import dataclasses
import os

import pytest

from kv_stale_pairs.helpers import CARD, NEW_PORT, bin_name, call, listing, write_pair
from turbohaul.kv_stale_pairs import DropResult


def test_a_missing_directory_skips_as_dir_missing(tmp_path):
    res = call(tmp_path / "absent")
    assert res.skipped == "dir_missing" and res.dropped == (), f"no directory, nothing to drop ({CARD})"


def test_a_plain_file_in_place_of_the_directory_skips_as_dir_missing(tmp_path):
    target = tmp_path / "file"
    target.write_text("x")
    assert call(target).skipped == "dir_missing", f"save_dir must be a directory ({CARD})"


def test_a_scan_error_skips_as_scan_failed_and_deletes_nothing(tmp_path, monkeypatch):
    write_pair(tmp_path, 11500)
    before = listing(tmp_path)
    real = os.scandir

    def _boom(path):
        if str(path) == str(tmp_path):
            raise PermissionError("scan denied")
        return real(path)

    monkeypatch.setattr(os, "scandir", _boom)
    res = call(tmp_path)
    monkeypatch.undo()
    assert res.skipped == "scan_failed" and res.dropped == (), f"a failed scan must keep every pair ({CARD})"
    assert listing(tmp_path) == before


def test_a_run_reports_the_echoed_arguments(tmp_path):
    write_pair(tmp_path, 11500)
    res = call(tmp_path, landed=777, n_written=777)
    assert (res.landed_bytes, res.n_written) == (777, 777), f"the arguments are echoed on a run too ({CARD})"


def test_an_empty_directory_runs_and_drops_nothing(tmp_path):
    res = call(tmp_path)
    assert res == DropResult(dropped=(), skipped=None, landed_bytes=1000, n_written=1000), f"empty dir ({CARD})"


def test_a_directory_with_only_the_new_pair_drops_nothing(tmp_path):
    write_pair(tmp_path, NEW_PORT)
    before = listing(tmp_path)
    res = call(tmp_path)
    assert res.skipped is None and res.dropped == () and listing(tmp_path) == before


def test_dropped_is_a_tuple_and_the_result_is_frozen(tmp_path):
    write_pair(tmp_path, 11500)
    res = call(tmp_path)
    assert isinstance(res.dropped, tuple), f"dropped is a tuple ({CARD})"
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.skipped = "x"


def test_a_second_call_after_the_drop_finds_nothing_left(tmp_path):
    write_pair(tmp_path, 11500)
    write_pair(tmp_path, NEW_PORT)
    assert len(call(tmp_path).dropped) == 1
    again = call(tmp_path)
    assert again.skipped is None and again.dropped == (), f"the drop is idempotent ({CARD})"
    assert bin_name(NEW_PORT) in listing(tmp_path)


def test_a_non_string_directory_never_raises(tmp_path):
    res = call(None)
    assert res.skipped in ("dir_missing", "scan_failed") and res.dropped == (), f"never raises ({CARD})"
