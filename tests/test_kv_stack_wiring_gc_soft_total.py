"""The GC's soft fallback hands its remaining total back to the ceiling check.

After the unprotected bins are gone, the GC evicts idle-held bins oldest first while a live turn's
bin holds the floor. The "still over the cap" warning that follows must use the total left after
that fallback. Two cases, one thread, no processes:

* the fallback brings the directory under the cap, so the warning must not be logged;
* the fallback is not enough, so exactly one warning is logged and its byte figure is the true remainder.

The fixtures are copied from the other save-and-GC tests.
"""
import json
import logging
import os

import pytest

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

_QWEN = "example-27b"
_MIB = 1024 * 1024
_LOGGER = "turbohaul.manager"
_WARNING_START = "slot KV GC ceiling: still"


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
            default_port_base=59500,
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


def _write_kv_bin(kv_dir, name: str, size: int, mtime: float, thread_hash: str) -> str:
    """A .bin plus a .json sidecar carrying thread_hash, so the GC can classify the bin."""
    path = kv_dir / name
    path.write_bytes(b"\x00" * size)
    os.utime(path, (mtime, mtime))
    meta = {"model_tag": _QWEN, "port": 59500, "slot_id": 0, "thread_hash": thread_hash}
    (kv_dir / (name[:-4] + ".json")).write_text(json.dumps(meta))
    return name


def _make_active_and_idle(mgr, active_tid: str, idle_tid: str) -> None:
    mgr._active_slot = Slot.new(
        _QWEN, thread_id=active_tid, admission_ctx_len=100,
        client_meta={"session_id": f"sess-{active_tid}"},
    )
    mgr._idle_thread_id = idle_tid
    mgr._idle_client_meta = {"session_id": f"sess-{idle_tid}"}


def _ceiling_warnings(caplog):
    return [r for r in caplog.records
            if r.name == _LOGGER and r.levelno == logging.WARNING and r.getMessage().startswith(_WARNING_START)]


async def test_a_soft_fallback_that_makes_room_logs_no_over_cap_warning(mgr, kv_dir, caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    _make_active_and_idle(mgr, "active-tid", "idle-tid")
    hard = _write_kv_bin(kv_dir, "m.hard.active.bin", 10 * _MIB, 1000.0, mgr._thread_hash("active-tid"))
    soft_old = _write_kv_bin(kv_dir, "m.soft.old.bin", 10 * _MIB, 2000.0, mgr._thread_hash("idle-tid"))
    soft_new = _write_kv_bin(kv_dir, "m.soft.new.bin", 10 * _MIB, 3000.0, mgr._thread_hash("idle-tid"))
    mgr.runtime.kv.ram_cache_max_bytes = 25 * _MIB

    deleted = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000_000)

    assert deleted == 1, f"one soft bin brings 30 MiB under a 25 MiB cap; got {deleted} removals"
    assert not (kv_dir / soft_old).exists(), "the oldest soft bin must be removed"
    assert (kv_dir / soft_new).exists() and (kv_dir / hard).exists(), "the newer soft bin and the live bin stay"
    assert _ceiling_warnings(caplog) == [], (
        "the directory is under the cap after the soft fallback, so no over-cap warning may be logged; got "
        f"{[r.getMessage() for r in _ceiling_warnings(caplog)]}")


async def test_a_soft_fallback_that_is_not_enough_logs_one_warning_with_the_true_remainder(mgr, kv_dir, caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    _make_active_and_idle(mgr, "active-tid", "idle-tid")
    hard_one = _write_kv_bin(kv_dir, "m.hard.one.bin", 10 * _MIB, 1000.0, mgr._thread_hash("active-tid"))
    hard_two = _write_kv_bin(kv_dir, "m.hard.two.bin", 10 * _MIB, 1500.0, mgr._thread_hash("active-tid"))
    soft = _write_kv_bin(kv_dir, "m.soft.only.bin", 10 * _MIB, 2000.0, mgr._thread_hash("idle-tid"))
    mgr.runtime.kv.ram_cache_max_bytes = 5 * _MIB

    deleted = await mgr._gc_kv_cache(max_age_hours=1e9, max_files=1_000_000_000)

    assert deleted == 1, f"only the soft bin may go; got {deleted} removals"
    assert not (kv_dir / soft).exists(), "the soft bin must be removed"
    assert (kv_dir / hard_one).exists() and (kv_dir / hard_two).exists(), "a live turn's bins are never removed"
    warnings = _ceiling_warnings(caplog)
    assert len(warnings) == 1, f"exactly one over-cap warning is expected; got {len(warnings)}"
    assert warnings[0].args == (15 * _MIB,), (
        f"20 MiB remain against a 5 MiB cap, so the warning must report 15 MiB over; got {warnings[0].args}")
