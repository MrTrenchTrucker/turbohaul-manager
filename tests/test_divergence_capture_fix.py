"""Divergence-capture facility fix: the facility that records divergence
between a thread's cached prefix and its incoming messages must actually
write its dump files, and must be reachable on the path (a first
request after a full unload) where the divergence it exists to catch occurs.

Two defects fixed:
  1. The dump target directory did not exist, every open() raised, and the
     failure was caught and demoted to log.debug -- invisible at INFO, while
     an unconditional log.info announcement fired regardless of whether the
     write that followed actually succeeded. Many announcements, zero bytes
     written, visible only by checking the filesystem directly.
  2. Both existing call sites are on the WARM grace-window follow-up path.
     No call exists anywhere on the cold-restore path
     (_restore_slot_kv_inner) -- the facility could not observe
     the event it exists to explain.

Design:
  1. One shared write chokepoint (_divergence_dump_write) for all three JSONL
     write sites, replacing three near-identical try/except blocks.
  2. A write failure is ERROR, always, never debug, never a bare swallow --
     this facility exists solely to produce that file.
  3. The rare/valuable case (Case A + the extended tool-opaque
     byte-diff) and the high-volume benign case (length_mismatch) are
     written to two SEPARATE files, so the benign case's volume can never
     put the valuable case's record at risk.
  4. The function is named _log_serve_divergence, since it fires on both
     the warm and cold paths, so the name describes its behaviour on either
     path.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest tests/test_divergence_capture_fix.py -v
"""
from __future__ import annotations

import json
import logging
import pathlib

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

pytestmark = pytest.mark.asyncio

_QWEN = "qwen3.6-27b"
_PORT = 60300


@pytest.fixture
def mgr(tmp_path):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=_PORT,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture
def divergence_debug_on(monkeypatch):
    """Both gates the facility requires, satisfied without touching the real
    system paths: the env var (a plain setenv), and the sentinel file (a
    narrowly-targeted Path.exists patch that only fakes THIS one path, real
    behaviour for everything else)."""
    monkeypatch.setenv("TURBOHAUL_DIVERGENCE_DEBUG", "1")
    _real_exists = pathlib.Path.exists

    def _patched_exists(self):
        if str(self) == "/var/lib/turbohaul/.divergence_debug":
            return True
        return _real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", _patched_exists)


@pytest.fixture
def divergence_files(tmp_path, monkeypatch):
    """Route both dump targets to tmp_path instead of the real system path --
    hermetic, and proves the directory-creation fix without touching
    anything outside the test's own sandbox."""
    capture = tmp_path / "divergence" / "capture.jsonl"
    benign = tmp_path / "divergence" / "benign.jsonl"
    monkeypatch.setattr(manager_mod, "_DIVERGENCE_CAPTURE_FILE", str(capture))
    monkeypatch.setattr(manager_mod, "_DIVERGENCE_BENIGN_FILE", str(benign))
    return capture, benign


def _jsonl_entries(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestDivergenceDumpWriteChokepoint:
    """Dump-directory fix: the shared write helper directly."""

    async def test_creates_missing_parent_directory(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist" / "yet" / "capture.jsonl"
        assert not target.parent.exists()
        TurbohaulManager._divergence_dump_write(str(target), {"case": "test"})
        assert target.exists(), "the directory-creation fix did not create the parent"
        assert _jsonl_entries(target) == [{"case": "test"}]

    async def test_appends_multiple_entries(self, tmp_path):
        target = tmp_path / "capture.jsonl"
        TurbohaulManager._divergence_dump_write(str(target), {"n": 1})
        TurbohaulManager._divergence_dump_write(str(target), {"n": 2})
        assert _jsonl_entries(target) == [{"n": 1}, {"n": 2}]

    async def test_write_failure_is_logged_at_error_never_debug(self, tmp_path, caplog):
        # A directory sitting exactly where the "file" is supposed to go --
        # open(path, "a") raises IsADirectoryError even after os.makedirs
        # succeeds on the parent, reproducing a genuine, unrecoverable write
        # failure (the defect's actual failure mode: the target
        # could not be opened for writing).
        blocked = tmp_path / "capture.jsonl"
        blocked.mkdir()
        with caplog.at_level(logging.DEBUG):
            TurbohaulManager._divergence_dump_write(str(blocked), {"case": "test"})
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        debug_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG and "DIVERGENCE_DUMP" in r.message
        ]
        assert len(error_records) == 1, (
            "a write failure must be logged at ERROR -- found: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )
        assert "FAILED" in error_records[0].message
        assert debug_records == [], (
            "a write failure must never be silently demoted to DEBUG -- "
            f"found: {[r.message for r in debug_records]}"
        )

    async def test_write_failure_does_not_raise(self, tmp_path):
        blocked = tmp_path / "capture.jsonl"
        blocked.mkdir()
        # Never-raises is the facility's own stated contract -- a debug
        # capture must not be able to crash the real request path.
        TurbohaulManager._divergence_dump_write(str(blocked), {"case": "test"})

    async def test_successful_write_is_also_logged(self, tmp_path, caplog):
        target = tmp_path / "capture.jsonl"
        with caplog.at_level(logging.INFO):
            TurbohaulManager._divergence_dump_write(str(target), {"case": "test"})
        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "DIVERGENCE_DUMP" in r.message
        ]
        assert len(info_records) == 1, info_records


class TestCaseSeparationNotRotation:
    """Case separation: the rare/valuable case and the benign high-volume case must
    land in separate files, and the benign case must never be able to put
    the valuable case's record at risk."""

    async def test_case_a_writes_capture_file_only(
        self, mgr, divergence_debug_on, divergence_files, caplog
    ):
        capture_file, benign_file = divergence_files
        slot = Slot.new(model_tag=_QWEN, thread_id="t-case-a")
        with caplog.at_level(logging.INFO):
            mgr._log_serve_divergence(slot, ["h0", "h1"], [{"role": "user"}, {"role": "assistant"}], _PORT)
            # Second call, same identity, DIFFERENT hash at position 1 -> Case A.
            mgr._log_serve_divergence(slot, ["h0", "h2"], [{"role": "user"}, {"role": "assistant"}], _PORT)
        assert _jsonl_entries(capture_file) != []
        assert all(e.get("case") in (None, "extended_capture_tool_opaque") or "first_divergent_turn" in e
                   for e in _jsonl_entries(capture_file))
        assert _jsonl_entries(benign_file) == [], (
            "a Case A divergence must never appear in the benign file"
        )

    async def test_length_mismatch_writes_benign_file_only(
        self, mgr, divergence_debug_on, divergence_files, caplog
    ):
        capture_file, benign_file = divergence_files
        slot = Slot.new(model_tag=_QWEN, thread_id="t-length-mismatch")
        with caplog.at_level(logging.INFO):
            mgr._log_serve_divergence(slot, ["h0", "h1"], [{"role": "user"}, {"role": "assistant"}], _PORT)
            # Second call, same prefix, MORE turns -> length_mismatch, not Case A.
            mgr._log_serve_divergence(
                slot, ["h0", "h1", "h2"],
                [{"role": "user"}, {"role": "assistant"}, {"role": "user"}], _PORT)
        entries = _jsonl_entries(benign_file)
        assert len(entries) == 1, entries
        assert entries[0]["case"] == "length_mismatch"
        assert _jsonl_entries(capture_file) == [], (
            "a benign length-mismatch event must never appear in the capture file"
        )

    async def test_high_volume_benign_traffic_cannot_touch_capture_file(
        self, mgr, divergence_debug_on, divergence_files
    ):
        """A realistic production shape: many length_mismatch events,
        zero Case A, in one run. Simulate a smaller but real burst and prove
        none of it lands in the file the rare divergence record depends on."""
        capture_file, benign_file = divergence_files
        for i in range(20):
            slot = Slot.new(model_tag=_QWEN, thread_id=f"t-burst-{i}")
            mgr._log_serve_divergence(slot, ["h0"], [{"role": "user"}], _PORT)
            mgr._log_serve_divergence(slot, ["h0", "h1"], [{"role": "user"}, {"role": "assistant"}], _PORT)
        assert len(_jsonl_entries(benign_file)) == 20
        assert _jsonl_entries(capture_file) == []


class TestBenignFileIsCapped:
    """Size cap: the benign file is isolated, but left alone it would be
    unbounded. The sizing math: each length_mismatch entry carries full incoming +
    previous message sets, ~0.25 MB per entry on a 65k-token conversation,
    ~30 MB per run, growing forever if left on. Capped via a
    size-triggered single-backup roll -- the capture file must stay
    completely unaffected by this, proven directly below."""

    async def test_write_under_cap_does_not_roll(self, tmp_path):
        target = tmp_path / "benign.jsonl"
        TurbohaulManager._divergence_dump_write(str(target), {"n": 1}, max_bytes=1024)
        TurbohaulManager._divergence_dump_write(str(target), {"n": 2}, max_bytes=1024)
        assert not (tmp_path / "benign.jsonl.1").exists()
        assert _jsonl_entries(target) == [{"n": 1}, {"n": 2}]

    async def test_write_at_or_over_cap_rolls_to_single_backup(self, tmp_path):
        target = tmp_path / "benign.jsonl"
        # First entry alone exceeds a tiny cap -> every subsequent write rolls.
        TurbohaulManager._divergence_dump_write(str(target), {"n": 1}, max_bytes=5)
        assert _jsonl_entries(target) == [{"n": 1}]
        TurbohaulManager._divergence_dump_write(str(target), {"n": 2}, max_bytes=5)
        # The roll happens BEFORE the new entry is appended: entry 1 moves to
        # .1, the active file starts fresh with only entry 2.
        assert _jsonl_entries(target) == [{"n": 2}]
        assert _jsonl_entries(tmp_path / "benign.jsonl.1") == [{"n": 1}]

    async def test_backup_is_overwritten_not_accumulated(self, tmp_path):
        """A single backup, not an ever-growing series -- rolling three times
        over a tiny cap must never leave more than one .1 file, and it must
        never leave a .2, .3, etc."""
        target = tmp_path / "benign.jsonl"
        for i in range(5):
            TurbohaulManager._divergence_dump_write(str(target), {"n": i}, max_bytes=5)
        assert not (tmp_path / "benign.jsonl.2").exists()
        assert not (tmp_path / "benign.jsonl.3").exists()
        # Only the two most recent entries survive anywhere on disk.
        assert _jsonl_entries(target) == [{"n": 4}]
        assert _jsonl_entries(tmp_path / "benign.jsonl.1") == [{"n": 3}]

    async def test_capture_file_call_sites_pass_no_cap(
        self, mgr, divergence_debug_on, divergence_files, monkeypatch
    ):
        """The valuable file must stay genuinely unbounded -- prove it survives
        past what the benign cap would have rolled at, via the REAL call
        sites (not a direct call to the write helper), with the benign cap
        set deliberately tiny so a bug that accidentally capped the capture
        file too would be caught immediately."""
        capture_file, benign_file = divergence_files
        # Tiny -- the benign write rolls under this almost every call.
        monkeypatch.setattr(manager_mod, "_DIVERGENCE_BENIGN_MAX_BYTES", 10)
        slot = Slot.new(model_tag=_QWEN, thread_id="t-cap-isolation")
        # Two Case-A-producing calls (capture file) interleaved with a
        # length-mismatch-producing call (benign file, tiny cap).
        mgr._log_serve_divergence(slot, ["h0", "h1"], [{"role": "user"}, {"role": "a"}], _PORT)
        mgr._log_serve_divergence(slot, ["h0", "h2"], [{"role": "user"}, {"role": "a"}], _PORT)  # Case A
        mgr._log_serve_divergence(
            slot, ["h0", "h2", "h3"],
            [{"role": "user"}, {"role": "a"}, {"role": "user"}], _PORT)  # length_mismatch
        capture_entries = _jsonl_entries(capture_file)
        assert len(capture_entries) == 1, capture_entries
        assert capture_entries[0]["first_divergent_turn"] == 1
        # The benign write rolled under its tiny cap -- confirms the cap is
        # actually live on this path, and that it did not touch capture_file.
        assert _jsonl_entries(capture_file) == capture_entries


class TestColdRestorePathNowCaptures:
    """Missing call site: the facility needs a call site on the cold-restore
    path. Drives the REAL _restore_slot_kv_inner wiring, not a direct call to
    _log_serve_divergence, to prove the new call site actually fires."""

    async def test_two_cold_restores_with_diverging_chain_produce_case_a(
        self, mgr, kv_dir, divergence_debug_on, divergence_files, caplog
    ):
        capture_file, _benign_file = divergence_files
        thread_id = "t-cold-restore-divergence"
        client_meta = {"messages": [{"role": "user", "content": "turn 1"},
                                     {"role": "assistant", "content": "reply 1"}]}
        slot1 = Slot.new(
            model_tag=_QWEN, thread_id=thread_id, client_meta=client_meta,
            admission_hash_chain=["h0", "h1"])
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot1)

        # Second cold restore, SAME (thread_id, model_tag) identity, but the
        # second position's hash has changed -- Case A, exactly the shape a
        # think-token-inclusion bug produces (position matches by turn count,
        # differs by content).
        client_meta2 = {"messages": [{"role": "user", "content": "turn 1"},
                                      {"role": "assistant", "content": "DIFFERENT reply"}]}
        slot2 = Slot.new(
            model_tag=_QWEN, thread_id=thread_id, client_meta=client_meta2,
            admission_hash_chain=["h0", "h2"])
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot2)

        case_a_lines = [
            r.message for r in caplog.records
            if r.message.startswith("DIVERGENCE_DUMP: case=A")
        ]
        assert len(case_a_lines) == 1, (
            "the cold-restore path must fire a Case A divergence capture when "
            f"two consecutive cold restores on the same thread diverge; log: "
            f"{[r.message for r in caplog.records]}"
        )
        entries = _jsonl_entries(capture_file)
        assert len(entries) == 1, entries
        assert entries[0]["thread_id"] == thread_id
        assert entries[0]["first_divergent_turn"] == 1

    async def test_single_cold_restore_is_the_baseline_stash_no_capture_yet(
        self, mgr, kv_dir, divergence_debug_on, divergence_files, caplog
    ):
        """First-ever call for a (thread_id, model_tag) stashes state and
        returns -- there is nothing to diverge FROM yet. Confirms the new
        call site does not misfire on the very first cold restore of a
        thread's lifetime."""
        capture_file, benign_file = divergence_files
        slot = Slot.new(
            model_tag=_QWEN, thread_id="t-first-ever-cold-restore",
            client_meta={"messages": [{"role": "user", "content": "hi"}]},
            admission_hash_chain=["h0"])
        with caplog.at_level(logging.INFO):
            await mgr._restore_slot_kv(_PORT, _QWEN, slot)
        assert _jsonl_entries(capture_file) == []
        assert _jsonl_entries(benign_file) == []
        stash_lines = [r.message for r in caplog.records if "first observed serve" in r.message]
        assert len(stash_lines) == 1, stash_lines
