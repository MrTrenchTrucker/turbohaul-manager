"""Module-interface tests for kv_ram_cap.dir_accounting (the save-dir byte accounting).

Each test pins one value the function must return. Imports go through the package entry point.
"""

import os
import sys
from pathlib import Path

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.kv_ram_cap import dir_accounting  # noqa: E402

CARD = "see src/turbohaul/kv_ram_cap/AGENTS.md"


def _bin(d: Path, name: str, size: int, mtime: float = 1_000.0, sidecar: int | None = None) -> Path:
    p = d / name
    p.write_bytes(b"\x00" * size)
    os.utime(p, (mtime, mtime))
    if sidecar is not None:
        (d / (name + ".ckpt")).write_bytes(b"\x01" * sidecar)
    return p


def test_total_adds_the_ckpt_sidecar_bytes_to_the_bin(tmp_path):
    _bin(tmp_path, "a.bin", 100, sidecar=40)
    total, bins = dir_accounting(str(tmp_path))
    assert total == 140, f"dir_accounting: total must be bin + .bin.ckpt bytes ({CARD})"
    assert [b[3] for b in bins] == [140], f"dir_accounting: entry size must be bin + sidecar ({CARD})"


def test_bin_without_a_sidecar_counts_only_its_own_bytes(tmp_path):
    _bin(tmp_path, "a.bin", 70)
    total, bins = dir_accounting(str(tmp_path))
    assert total == 70, f"dir_accounting: a missing sidecar must count as zero bytes ({CARD})"
    assert bins[0][3] == 70, f"dir_accounting: entry size must be 70 with no sidecar ({CARD})"


def test_total_is_the_sum_over_every_bin(tmp_path):
    _bin(tmp_path, "a.bin", 10, sidecar=1)
    _bin(tmp_path, "b.bin", 20, sidecar=2)
    _bin(tmp_path, "c.bin", 30, sidecar=3)
    total, bins = dir_accounting(str(tmp_path))
    assert total == 66, f"dir_accounting: total must sum all bins plus sidecars ({CARD})"
    assert len(bins) == 3, f"dir_accounting: every .bin must have an entry ({CARD})"


def test_names_that_do_not_end_in_bin_are_ignored(tmp_path):
    (tmp_path / "meta.json").write_bytes(b"x" * 500)
    (tmp_path / "half.bin.tmp").write_bytes(b"x" * 300)
    (tmp_path / "orphan.ckpt").write_bytes(b"x" * 200)
    (tmp_path / "orphan.bin.ckpt").write_bytes(b"x" * 999)
    _bin(tmp_path, "keep.bin", 10)
    total, bins = dir_accounting(str(tmp_path))
    assert total == 10, f"dir_accounting: only .bin names may count toward the total ({CARD})"
    assert [b[0] for b in bins] == ["keep.bin"], f"dir_accounting: only keep.bin is an entry ({CARD})"


def test_missing_directory_returns_zero_and_no_entries(tmp_path):
    assert dir_accounting(str(tmp_path / "does-not-exist")) == (0, []), (
        f"dir_accounting: a missing directory must give (0, []) ({CARD})")
    some_file = tmp_path / "plain.txt"
    some_file.write_bytes(b"x")
    assert dir_accounting(str(some_file)) == (0, []), f"dir_accounting: a file is not a directory ({CARD})"


def test_entry_is_a_plain_tuple_of_name_path_mtime_and_bin_plus_sidecar_size(tmp_path):
    _bin(tmp_path, "a.bin", 20, mtime=1_700_000_000.5, sidecar=13)
    _, bins = dir_accounting(str(tmp_path))
    expected = ("a.bin", os.path.join(str(tmp_path), "a.bin"), 1_700_000_000.5, 33)
    assert bins == [expected], f"dir_accounting: entry must be (fn, fpath, mtime of the bin, size) ({CARD})"
    assert type(bins[0]) is tuple, f"dir_accounting: each entry must be a tuple ({CARD})"


def test_failing_listdir_calls_on_error_once_and_returns_empty_totals(tmp_path, monkeypatch):
    _bin(tmp_path, "a.bin", 10)

    def listdir_denied(_path):
        raise PermissionError("listdir denied")

    monkeypatch.setattr(os, "listdir", listdir_denied)
    errors = []
    total, bins = dir_accounting(str(tmp_path), on_error=lambda: errors.append("listdir"))
    assert len(errors) == 1, f"dir_accounting: on_error must run exactly once on a failing listdir ({CARD})"
    assert (total, bins) == (0, []), f"dir_accounting: a failing listdir must give (0, []) ({CARD})"


def test_failure_midway_returns_the_bins_gathered_before_it(tmp_path, monkeypatch):
    _bin(tmp_path, "a.bin", 15, sidecar=0)
    _bin(tmp_path, "b.bin", 20)
    real_stat = os.stat
    bin_stats = []

    def stat_fails_on_second_bin(path, *args, **kwargs):
        if str(path).endswith(".bin"):
            bin_stats.append(str(path))
            if len(bin_stats) == 2:
                raise RuntimeError("stat fault on second bin")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_fails_on_second_bin)
    errors = []
    total, bins = dir_accounting(str(tmp_path), on_error=lambda: errors.append("stat"))
    assert len(errors) == 1, f"dir_accounting: on_error must run once on a mid-scan failure ({CARD})"
    assert len(bins) == 1, f"dir_accounting: the partial result must keep the first bin only ({CARD})"
    size_by_name = {"a.bin": 15, "b.bin": 20}
    assert total == size_by_name[bins[0][0]], f"dir_accounting: partial total must match the kept bin ({CARD})"


def test_bin_that_cannot_be_stat_is_skipped_without_an_error_report(tmp_path, monkeypatch):
    _bin(tmp_path, "gone.bin", 50)
    _bin(tmp_path, "kept.bin", 7)
    real_stat = os.stat

    def stat_gone(path, *args, **kwargs):
        if str(path).endswith("gone.bin"):
            raise FileNotFoundError(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat_gone)
    errors = []
    total, bins = dir_accounting(str(tmp_path), on_error=lambda: errors.append("stat"))
    assert [b[0] for b in bins] == ["kept.bin"], f"dir_accounting: an unstatable bin must be skipped ({CARD})"
    assert total == 7, f"dir_accounting: the skipped bin must not count toward the total ({CARD})"
    assert errors == [], f"dir_accounting: a skipped bin is not an error and must not call on_error ({CARD})"


def test_failure_with_no_on_error_is_tolerated_and_returns_zero(tmp_path, monkeypatch):
    def listdir_denied(_path):
        raise OSError("listdir denied")

    monkeypatch.setattr(os, "listdir", listdir_denied)
    assert dir_accounting(str(tmp_path)) == (0, []), (
        f"dir_accounting: with on_error=None a failure must still return (0, []) ({CARD})")
