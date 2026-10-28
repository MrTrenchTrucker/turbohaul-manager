"""Module-interface tests for the kv_stack_wiring save-path functions.

Covers dir_accounting_logged, save_declined_by_cap, drop_stale_pairs_after_save and
evict_soft_floor. The hosts are plain fakes that count every call, so each test pins the returned
value, the number and order of calls, and the exact log record.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.kv_stack_wiring import (  # noqa: E402
    dir_accounting_logged,
    drop_stale_pairs_after_save,
    evict_soft_floor,
    save_declined_by_cap,
)

CARD = "see src/turbohaul/kv_stack_wiring/AGENTS.md"
LOGGER = "turbohaul.manager"


def _records(caplog, level):
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level]


# --- dir_accounting_logged ---------------------------------------------------------------------

def test_dir_accounting_logged_counts_each_bin_with_its_checkpoint_and_logs_nothing(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "a.bin.ckpt").write_bytes(b"y" * 5)
    (tmp_path / "b.bin").write_bytes(b"z" * 7)
    (tmp_path / "note.txt").write_bytes(b"ignored")
    total, bins = dir_accounting_logged(str(tmp_path))
    assert total == 22, f"bin plus checkpoint plus the lone bin is 22 bytes; got {total} ({CARD})"
    assert sorted((fn, size) for fn, _p, _m, size in bins) == [("a.bin", 15), ("b.bin", 7)]
    assert _records(caplog, logging.DEBUG) == [], "a clean scan must log nothing"


def test_dir_accounting_logged_on_a_failed_scan_returns_zero_and_logs_one_debug_record(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    real = os.listdir

    def _failing(path=".", *a, **k):
        if os.fspath(path) == str(tmp_path):
            raise OSError("listdir boom")
        return real(path, *a, **k)

    monkeypatch.setattr(os, "listdir", _failing)
    assert dir_accounting_logged(str(tmp_path)) == (0, [])
    records = _records(caplog, logging.DEBUG)
    assert len(records) == 1, f"a failed scan must log exactly one DEBUG record; got {len(records)}"
    assert records[0].getMessage() == "slot KV dir accounting failed (best-effort)"
    assert records[0].exc_info is not None and records[0].exc_info[0] is OSError, "exc_info must carry the OSError"


# --- save_declined_by_cap ----------------------------------------------------------------------

class _CapHost:
    def __init__(self, cap, sizes, gc_error=None, measure_error=None):
        self.runtime = SimpleNamespace(kv=SimpleNamespace(ram_cache_max_bytes=cap))
        self._sizes = list(sizes)
        self._gc_error = gc_error
        self._measure_error = measure_error
        self.measure_calls = 0
        self.gc_calls = []

    def _kvcache_dir_accounting(self):
        self.measure_calls += 1
        if self._measure_error is not None:
            raise self._measure_error
        size = self._sizes.pop(0) if len(self._sizes) > 1 else self._sizes[0]
        return size, ["unused bins"]

    async def _gc_kv_cache(self, max_age_hours=None, max_files=100):
        self.gc_calls.append((max_age_hours, max_files))
        if self._gc_error is not None:
            raise self._gc_error
        return 0


def test_no_cap_means_no_measure_no_gc_no_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host = _CapHost(0, [999])
    assert asyncio.run(save_declined_by_cap(host)) is False
    assert host.measure_calls == 0 and host.gc_calls == [], f"a zero cap must call nothing ({CARD})"
    assert _records(caplog, logging.WARNING) == []


def test_under_the_cap_proceeds_after_one_measure(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host = _CapHost(100, [40])
    assert asyncio.run(save_declined_by_cap(host)) is False
    assert host.measure_calls == 1 and host.gc_calls == [], "under the cap: one measure, no GC"
    assert _records(caplog, logging.WARNING) == []


def test_over_the_cap_drives_the_gc_once_with_the_ceiling_only_arguments_and_proceeds_when_room_was_made(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host = _CapHost(100, [200, 50])
    assert asyncio.run(save_declined_by_cap(host)) is False
    assert host.gc_calls == [(1e9, 1_000_000_000)], f"the GC runs once, age and count disabled; got {host.gc_calls}"
    assert host.measure_calls == 2, f"measure before and after the GC; got {host.measure_calls}"
    assert _records(caplog, logging.WARNING) == [], "a save that made room is not declined, so no warning"


def test_still_over_the_cap_after_the_gc_declines_with_one_warning_naming_dir_and_cap(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host = _CapHost(100, [300, 250])
    assert asyncio.run(save_declined_by_cap(host)) is True
    assert host.gc_calls == [(1e9, 1_000_000_000)] and host.measure_calls == 2
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1, f"a declined save writes exactly one warning; got {len(warnings)}"
    assert warnings[0].args == (250, 100), f"warning args are the 2nd measurement and the cap; got {warnings[0].args}"
    message = warnings[0].getMessage()
    assert message.startswith("slot KV save SKIP (hard RAM cap "), message
    assert message.endswith(
        "): dir=250 cap=100 — HARD active/in-flight bins hold the floor; snapshot declined, live KV retained in-engine"
    ), message


def test_a_failing_gc_propagates(caplog):
    host = _CapHost(100, [300], gc_error=RuntimeError("gc boom"))
    with pytest.raises(RuntimeError, match="gc boom"):
        asyncio.run(save_declined_by_cap(host))


def test_a_failing_measure_propagates():
    host = _CapHost(100, [300], measure_error=ValueError("measure boom"))
    with pytest.raises(ValueError, match="measure boom"):
        asyncio.run(save_declined_by_cap(host))


# --- drop_stale_pairs_after_save ---------------------------------------------------------------

class _DropHost:
    def __init__(self, events, groups="GROUPS", groups_error=None):
        self.events = events
        self._groups = groups
        self._groups_error = groups_error
        self.invalidations = 0

    def _live_turn_groups(self):
        self.events.append("groups")
        if self._groups_error is not None:
            raise self._groups_error
        return self._groups

    def _kvcache_scan_cache_invalidate(self):
        self.invalidations += 1


def _result(skipped=None, dropped=()):
    return SimpleNamespace(skipped=skipped, dropped=tuple(dropped))


def _drop_fn(events, result=None, error=None):
    calls = []

    def _fn(*args):
        events.append("drop")
        calls.append(args)
        if error is not None:
            raise error
        return result

    _fn.calls = calls
    return _fn


def _final(tmp_path, size=123):
    path = tmp_path / "final.bin"
    path.write_bytes(b"k" * size)
    return str(path)


def test_the_drop_gets_the_landed_size_then_the_live_groups_and_runs_after_them(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    events = []
    host = _DropHost(events)
    fn = _drop_fn(events, _result())
    drop_stale_pairs_after_save(host, fn, "/save", "model", "thash", 7, _final(tmp_path), 123, "x.bin")
    assert fn.calls == [("/save", "model", "thash", 7, "GROUPS", 123, 123)], f"drop arguments; got {fn.calls} ({CARD})"
    assert events == ["groups", "drop"], f"the live scan runs before the drop; got {events}"
    assert host.invalidations == 0 and _records(caplog, logging.DEBUG) == [], "a drop that did nothing is silent"


def test_a_missing_final_file_passes_landed_none_and_still_drops(tmp_path):
    events = []
    host = _DropHost(events)
    fn = _drop_fn(events, _result())
    drop_stale_pairs_after_save(host, fn, "/save", "model", "thash", 7, str(tmp_path / "gone.bin"), None, "x.bin")
    assert fn.calls == [("/save", "model", "thash", 7, "GROUPS", None, None)], f"got {fn.calls}"
    assert events == ["groups", "drop"]


def test_dropped_pairs_are_logged_one_info_each_and_clear_the_scan_cache_once(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host = _DropHost([])
    dropped = [SimpleNamespace(name="old1.bin", port=1, pair_bytes=11),
               SimpleNamespace(name="old2.bin", port=2, pair_bytes=22)]
    fn = _drop_fn([], _result(dropped=dropped))
    drop_stale_pairs_after_save(host, fn, "/save", "model", "thash", 7, _final(tmp_path), 123, "x.bin")
    infos = _records(caplog, logging.INFO)
    assert [r.getMessage() for r in infos] == [
        "slot KV stale-port pair dropped: old1.bin (port=1, now on port=7, 11 bytes)",
        "slot KV stale-port pair dropped: old2.bin (port=2, now on port=7, 22 bytes)",
    ], f"one INFO line per dropped pair, in order; got {[r.getMessage() for r in infos]}"
    assert host.invalidations == 1, f"the scan cache is cleared once, not per pair; got {host.invalidations}"


def test_a_landed_size_mismatch_logs_one_warning_and_does_not_clear_the_scan_cache(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host = _DropHost([])
    fn = _drop_fn([], _result(skipped="landed_size_mismatch"))
    drop_stale_pairs_after_save(host, fn, "/save", "model", "thash", 7, _final(tmp_path, 50), "oops", "x.bin")
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1, f"one warning; got {len(warnings)}"
    assert warnings[0].getMessage() == (
        "slot KV stale-port drop skipped for x.bin: landed_bytes=50 n_written='oops' differ or are unproven"
        " -- old pair kept"), warnings[0].getMessage()
    assert host.invalidations == 0


def test_any_other_skip_reason_logs_one_debug_line(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host = _DropHost([])
    fn = _drop_fn([], _result(skipped="served_unknown"))
    drop_stale_pairs_after_save(host, fn, "/save", "model", "thash", 7, _final(tmp_path), 123, "x.bin")
    debugs = _records(caplog, logging.DEBUG)
    assert [r.getMessage() for r in debugs] == ["slot KV stale-port drop skipped (served_unknown) for x.bin"]
    assert _records(caplog, logging.WARNING) == []


def test_a_drop_that_raises_is_swallowed_with_one_debug_record(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host = _DropHost([])
    fn = _drop_fn([], error=RuntimeError("drop boom"))
    drop_stale_pairs_after_save(host, fn, "/save", "model", "thash", 7, _final(tmp_path), 123, "x.bin")
    debugs = _records(caplog, logging.DEBUG)
    assert len(debugs) == 1 and debugs[0].getMessage() == "slot KV stale-port drop failed (best-effort)"
    assert debugs[0].exc_info is not None and debugs[0].exc_info[0] is RuntimeError
    assert host.invalidations == 0


def test_a_live_scan_that_raises_is_swallowed_and_the_drop_never_runs(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    events = []
    host = _DropHost(events, groups_error=RuntimeError("scan boom"))
    fn = _drop_fn(events, _result())
    drop_stale_pairs_after_save(host, fn, "/save", "model", "thash", 7, _final(tmp_path), 123, "x.bin")
    assert fn.calls == [], "the drop must not run when the live scan failed"
    debugs = _records(caplog, logging.DEBUG)
    assert len(debugs) == 1 and debugs[0].getMessage() == "slot KV stale-port drop failed (best-effort)"


# --- evict_soft_floor --------------------------------------------------------------------------

def _entry(name, mtime, size):
    return (name, f"/d/{name}", mtime, size)


class _Unlinker:
    def __init__(self, refuse=()):
        self.refuse = set(refuse)
        self.calls = []

    def __call__(self, path):
        self.calls.append(path)
        return path not in self.refuse


def _floor(entries, hashes, total, max_bytes, unlinker, hard=("H",), soft=("S",), pinned=(), deleted=()):
    return evict_soft_floor(entries, set(deleted), set(hard), set(soft), set(pinned), hashes,
                            total, max_bytes, unlinker)


def test_soft_bins_go_oldest_first_until_the_total_is_under_the_cap():
    entries = [_entry("new", 30, 10), _entry("old", 10, 10), _entry("mid", 20, 10)]
    hashes = {"/d/new": "S", "/d/old": "S", "/d/mid": "S"}
    unlinker = _Unlinker()
    removed, total = _floor(entries, hashes, 30, 15, unlinker)
    assert unlinker.calls == ["/d/old", "/d/mid"], f"oldest first, stop once under the cap; got {unlinker.calls}"
    assert (removed, total) == (2, 10)


def test_a_hard_bin_is_never_taken():
    entries = [_entry("hard", 1, 50), _entry("soft", 2, 10)]
    hashes = {"/d/hard": "H", "/d/soft": "S"}
    unlinker = _Unlinker()
    removed, total = _floor(entries, hashes, 60, 0, unlinker)
    assert unlinker.calls == ["/d/soft"], f"only the SOFT bin may be unlinked; got {unlinker.calls}"
    assert (removed, total) == (1, 50)


def test_a_total_already_at_or_under_the_cap_unlinks_nothing():
    entries = [_entry("soft", 1, 10)]
    unlinker = _Unlinker()
    assert _floor(entries, {"/d/soft": "S"}, 15, 15, unlinker) == (0, 15)
    assert unlinker.calls == [], "the check comes before the first unlink"


def test_a_refused_unlink_is_not_counted_and_its_size_stays_in_the_total():
    entries = [_entry("a", 1, 10), _entry("b", 2, 10)]
    hashes = {"/d/a": "S", "/d/b": "S"}
    unlinker = _Unlinker(refuse={"/d/a"})
    removed, total = _floor(entries, hashes, 20, 5, unlinker)
    assert unlinker.calls == ["/d/a", "/d/b"], "a refused bin does not stop the walk"
    assert (removed, total) == (1, 10), f"only the removed bin counts and is subtracted; got {(removed, total)}"


def test_a_pinned_bin_of_no_thread_and_an_already_deleted_bin():
    entries = [_entry("pinned", 1, 10), _entry("gone", 2, 10), _entry("soft", 3, 10)]
    hashes = {"/d/soft": "S"}
    unlinker = _Unlinker()
    removed, total = _floor(entries, hashes, 30, 0, unlinker, pinned=["/d/pinned"], deleted=["/d/gone"])
    assert unlinker.calls == ["/d/pinned", "/d/soft"], f"pinned bins qualify, deleted ones do not; got {unlinker.calls}"
    assert (removed, total) == (2, 10)
