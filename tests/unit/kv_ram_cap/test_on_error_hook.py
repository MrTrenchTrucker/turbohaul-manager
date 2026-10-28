"""Module-interface tests for the ``on_error`` hook of kv_ram_cap.

``on_error`` is a zero-argument callable. These tests pin three things: it stays silent when
nothing failed, it runs once per failure, and it runs while the exception is
still being handled, so the caller's logger can attach exc_info.
"""

import os
import sys
from pathlib import Path

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.kv_ram_cap import dir_accounting, split_protection  # noqa: E402

CARD = "see src/turbohaul/kv_ram_cap/AGENTS.md"


def _same_tid_identity(tid, client_meta, chain):
    return tid or ""


def test_dir_accounting_on_a_healthy_dir_never_calls_on_error(tmp_path):
    """Pins the on_error contract for dir_accounting: a healthy scan must not report a failure."""
    (tmp_path / "m.a.bin").write_bytes(b"\x00" * 100)
    (tmp_path / "m.a.bin.ckpt").write_bytes(b"\x00" * 7)
    calls = []
    total, bins = dir_accounting(str(tmp_path), on_error=lambda: calls.append("accounting"))
    assert calls == [], f"dir_accounting: on_error must not run on a healthy directory ({CARD})"
    assert total == 107, f"dir_accounting: total must be the bin plus its sidecar, 100 + 7 ({CARD})"
    assert [b[0] for b in bins] == ["m.a.bin"], f"dir_accounting: bins must name the one .bin file ({CARD})"


def test_dir_accounting_on_a_missing_dir_is_silent_too(tmp_path):
    """Pins the on_error contract for dir_accounting: a missing directory is a normal empty answer, not a failure."""
    calls = []
    result = dir_accounting(str(tmp_path / "does-not-exist"), on_error=lambda: calls.append("accounting"))
    assert calls == [], f"dir_accounting: a missing directory must not call on_error ({CARD})"
    assert result == (0, []), f"dir_accounting: a missing directory must return (0, []) ({CARD})"


def test_split_protection_on_healthy_input_never_calls_on_error():
    """Pins the on_error contract for split_protection: a healthy split must not report a failure."""
    calls = []
    split = split_protection(
        [("t1", None, None)], [("t2", None, None)],
        lambda tid: f"H({tid})", _same_tid_identity,
        on_error=lambda: calls.append("protection"),
    )
    assert calls == [], f"split_protection: on_error must not run on healthy input ({CARD})"
    assert split.hard == {"H(t1)"}, f"split_protection: the HARD thread hash must be H(t1) ({CARD})"
    assert split.soft == {"H(t2)"}, f"split_protection: the SOFT thread hash must be H(t2) ({CARD})"


def test_dir_accounting_on_error_runs_inside_the_except_with_the_listdir_exception(tmp_path, monkeypatch):
    """Pins the inside-the-except placement for dir_accounting: the hook must see the exact OSError
    that listdir raised, through sys.exc_info(), while that exception is being handled."""
    real_listdir = os.listdir
    raised = OSError("listdir boom")
    target = str(tmp_path)

    def failing_listdir(path=".", *args, **kwargs):
        if os.fspath(path) == target:
            raise raised
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "listdir", failing_listdir)
    seen = []
    total, bins = dir_accounting(target, on_error=lambda: seen.append(sys.exc_info()[1]))
    assert len(seen) == 1, f"dir_accounting: on_error must run exactly once on a listdir failure ({CARD})"
    assert seen[0] is raised, (
        f"dir_accounting: on_error must run inside the except block, so sys.exc_info()[1] must be "
        f"the listdir exception; got {seen[0]!r} ({CARD})")
    assert (total, bins) == (0, []), f"dir_accounting: a listdir failure must return (0, []) ({CARD})"


def test_split_protection_on_error_runs_inside_the_except_with_the_hash_exception():
    """Pins the inside-the-except placement for split_protection: the hook must see the exact
    ValueError that thread_hash raised, through sys.exc_info(), while that exception is handled."""
    raised = ValueError("hash boom")

    def thread_hash(tid):
        if tid == "bad":
            raise raised
        return f"H({tid})"

    seen = []
    split = split_protection(
        [("ok", None, None), ("bad", None, None)], [],
        thread_hash, _same_tid_identity,
        on_error=lambda: seen.append(sys.exc_info()[1]),
    )
    assert len(seen) == 1, f"split_protection: on_error must run exactly once on a hash failure ({CARD})"
    assert seen[0] is raised, (
        f"split_protection: on_error must run inside the except block, so sys.exc_info()[1] must be "
        f"the hash exception; got {seen[0]!r} ({CARD})")
    assert split.hard == {"H(ok)"}, (
        f"split_protection: the partial HARD tier must keep the triple before the failure ({CARD})")
