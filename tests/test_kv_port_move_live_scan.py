"""The stale-port pair drop at the save call site: the scan cache, the best-effort wrapper and the live-turn scan.

Four things the other port-move tests do not pin down, each with a control that tells it apart from a
save that never reaches the drop:

* a drop that removed pairs invalidates the scan cache once more than a save that dropped nothing;
* a drop that raises never fails the save and never touches the old pair;
* a live turn whose thread id differs from the saved thread, but whose bin identity is the saved
  thread's, spares the old port;
* with no residents, a live turn that sits only in the in-flight list spares the old port.

The fixtures are copied from the other port-move tests. One thread, no processes.
"""
import json
import logging
import os
from types import SimpleNamespace

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
from turbohaul.kv_policy import kv_meta_fn, kv_save_fn
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

MODEL = "example-27b"
OLD_PORT = 59501
NEW_PORT = 59500
THREAD = "sess-thread-A"
OTHER_THREAD = "sess-thread-B"
MSGS = [{"role": "user", "content": "hi"}]
BIN_SIZE = 400
SESSION_META = {"messages": MSGS, "session_id": "S1", "is_main": True}
SESSION_IDENTITY = "sess:S1:main"


@pytest.fixture
def mgr(tmp_path):
    root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (root / sub).mkdir(parents=True)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs", manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging", state_db_path=root / "state.sqlite"),
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59500),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    return TurbohaulManager(boot, RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig()))


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    monkeypatch.setenv("TURBOHAUL_CKPT_SIDECAR", "1")
    return d


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def _install_engine(monkeypatch, kv_dir, reported=BIN_SIZE, written=BIN_SIZE):
    """Fake engine that writes `written` bytes and reports `reported` as n_written."""

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return _Resp([{"id": 0, "n_prompt_tokens": 100, "thread_id": THREAD,
                           "n_context": 4096, "n_remaining": 2048}])

        async def post(self, url, json=None, **kw):
            if "action=save" in url and json and "filename" in json:
                tmp = os.path.join(str(kv_dir), json["filename"])
                with open(tmp, "wb") as f:
                    f.write(b"k" * written)
                with open(tmp + ".ckpt", "wb") as f:
                    f.write(b"c" * (written // 2))
                return _Resp({"n_saved": 100, "n_written": reported})
            return _Resp({})

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client())
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)


def _write_pair(kv_dir, mgr, port, key=THREAD, bin_size=600):
    """Lay down a complete saved pair for the thread (or bin identity) `key`, the way a finished save leaves it."""
    th = mgr._thread_hash(key)
    bin_fn = kv_save_fn(MODEL, 0, th, port)
    (kv_dir / bin_fn).write_bytes(b"k" * bin_size)
    (kv_dir / (bin_fn + ".ckpt")).write_bytes(b"c" * (bin_size // 2))
    (kv_dir / kv_meta_fn(MODEL, 0, th, port)).write_text(json.dumps({
        "thread_hash": th, "model_tag": MODEL, "port": port, "slot_id": 0,
        "prompt_tokens": 100, "bin_bytes": bin_size}))
    return bin_fn


def _names(kv_dir):
    return set(os.listdir(str(kv_dir)))


def _pair_files(bin_fn):
    return {bin_fn, bin_fn + ".ckpt", bin_fn[:-4] + ".json"}


def _live(tid=THREAD, client_meta=None):
    return Slot.new(MODEL, thread_id=tid, admission_ctx_len=100,
                    client_meta=client_meta if client_meta is not None else {"messages": MSGS})


async def _save(mgr, client_meta=None):
    meta = client_meta if client_meta is not None else {"messages": MSGS}
    slot = Slot.new(MODEL, thread_id=THREAD, admission_ctx_len=100, client_meta=meta)
    return await mgr._save_slot_kv(
        NEW_PORT, MODEL, slot, thread_id_override=THREAD, admission_ctx_len_override=100,
        client_meta_override=meta)


def _new_pair_landed(mgr, kv_dir, key=THREAD):
    new = kv_save_fn(MODEL, 0, mgr._thread_hash(key), NEW_PORT)
    return _pair_files(new) <= _names(kv_dir) and os.path.getsize(str(kv_dir / new)) == BIN_SIZE


async def _invalidations_for_one_save(mgr, kv_dir, monkeypatch, reported):
    """Run one save over an old-port pair; return (scan cache invalidations, whether the old pair is gone)."""
    for leftover in kv_dir.iterdir():
        leftover.unlink()
    old = _write_pair(kv_dir, mgr, OLD_PORT)
    _install_engine(monkeypatch, kv_dir, reported=reported)
    calls = []
    real = type(mgr)._kvcache_scan_cache_invalidate

    def _spy():
        calls.append(1)
        return real(mgr)

    monkeypatch.setattr(mgr, "_kvcache_scan_cache_invalidate", _spy)
    await _save(mgr)
    assert _new_pair_landed(mgr, kv_dir)
    return len(calls), not (_pair_files(old) & _names(kv_dir))


async def test_a_dropped_pair_invalidates_the_scan_cache_once_more_than_a_skipped_drop(mgr, kv_dir, monkeypatch):
    skipped_calls, skipped_gone = await _invalidations_for_one_save(mgr, kv_dir, monkeypatch, BIN_SIZE + 1)
    dropped_calls, dropped_gone = await _invalidations_for_one_save(mgr, kv_dir, monkeypatch, BIN_SIZE)
    assert not skipped_gone, "control: an unproven landing keeps the old pair, so nothing was dropped"
    assert dropped_gone, "the proven landing dropped the old pair"
    assert dropped_calls == skipped_calls + 1, (
        f"a drop that removed files must invalidate the scan cache once more: "
        f"dropped={dropped_calls} skipped={skipped_calls}")


async def test_a_drop_that_raises_is_contained_at_the_drop_and_the_rest_of_the_save_still_runs(
        mgr, kv_dir, monkeypatch, caplog):
    """The step after the drop writes a ring entry; it only runs when the drop's own failure stays inside the drop."""
    monkeypatch.setenv("TURBOHAUL_DURABLE_RING", "1")
    caplog.set_level(logging.DEBUG)
    old = _write_pair(kv_dir, mgr, OLD_PORT, key=SESSION_IDENTITY)
    before_old = {n: (kv_dir / n).read_bytes() for n in _pair_files(old)}
    _install_engine(monkeypatch, kv_dir)
    calls = []

    def _boom(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("drop exploded")

    monkeypatch.setattr(manager_mod, "drop_stale_pairs", _boom)
    result = await _save(mgr, SESSION_META)
    messages = [r.getMessage() for r in caplog.records]
    assert len(calls) == 1, "the drop was reached and raised"
    assert result is True, "a save that landed reports success even when the drop failed"
    assert _new_pair_landed(mgr, kv_dir, key=SESSION_IDENTITY)
    assert {n: (kv_dir / n).read_bytes() for n in _pair_files(old)} == before_old
    assert any("stale-port drop failed (best-effort)" in m for m in messages), messages
    assert not any("slot KV save failed" in m for m in messages), "the failure must not reach the slot-level handler"
    ring = mgr._durable_ring_index.get(("main", "S1"), [])
    assert [e["bin_fn"] for e in ring] == [kv_save_fn(MODEL, 0, mgr._thread_hash(SESSION_IDENTITY), NEW_PORT)], (
        "the step after the drop must still run")


async def test_a_live_turn_with_another_thread_id_but_the_saved_bin_identity_spares_the_old_port(
        mgr, kv_dir, monkeypatch):
    assert manager_mod._bin_identity(OTHER_THREAD, SESSION_META) == SESSION_IDENTITY
    old = _write_pair(kv_dir, mgr, OLD_PORT, key=SESSION_IDENTITY)
    _install_engine(monkeypatch, kv_dir)
    resident = SimpleNamespace(port=OLD_PORT, active_slot=_live(OTHER_THREAD, SESSION_META), inflight=[])
    monkeypatch.setattr(mgr, "_model_residents", lambda: [resident])
    await _save(mgr, SESSION_META)
    assert _new_pair_landed(mgr, kv_dir, key=SESSION_IDENTITY)
    assert _pair_files(old) <= _names(kv_dir), "the live turn's bin identity is the saved thread's"


async def test_a_live_turn_with_another_bin_identity_does_not_spare_the_old_port(mgr, kv_dir, monkeypatch):
    other_meta = {"messages": MSGS, "session_id": "S2", "is_main": True}
    assert manager_mod._bin_identity(OTHER_THREAD, other_meta) != SESSION_IDENTITY
    old = _write_pair(kv_dir, mgr, OLD_PORT, key=SESSION_IDENTITY)
    _install_engine(monkeypatch, kv_dir)
    resident = SimpleNamespace(port=OLD_PORT, active_slot=_live(OTHER_THREAD, other_meta), inflight=[])
    monkeypatch.setattr(mgr, "_model_residents", lambda: [resident])
    await _save(mgr, SESSION_META)
    assert _new_pair_landed(mgr, kv_dir, key=SESSION_IDENTITY)
    assert not (_pair_files(old) & _names(kv_dir)), "another session's turn does not serve this thread"


async def test_with_no_residents_a_turn_only_in_the_in_flight_list_spares_the_old_port(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    monkeypatch.setattr(mgr, "_model_residents", lambda: [])
    mgr._active_handle = SimpleNamespace(port=OLD_PORT)
    mgr._active_slot = None
    mgr._inflight = [_live(THREAD)]
    await _save(mgr)
    assert _new_pair_landed(mgr, kv_dir)
    assert _pair_files(old) <= _names(kv_dir), "the in-flight turn serves the thread on the old port"


async def test_with_no_residents_an_in_flight_turn_of_another_thread_does_not_spare_the_old_port(
        mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    monkeypatch.setattr(mgr, "_model_residents", lambda: [])
    mgr._active_handle = SimpleNamespace(port=OLD_PORT)
    mgr._active_slot = None
    mgr._inflight = [_live(OTHER_THREAD)]
    await _save(mgr)
    assert _new_pair_landed(mgr, kv_dir)
    assert not (_pair_files(old) & _names(kv_dir)), "an in-flight turn of another thread does not serve this one"
