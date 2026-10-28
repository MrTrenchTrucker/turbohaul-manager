"""A thread that moves to another engine port keeps ONE saved pair on the RAM disk.

When a save lands on a new port, the pair (bin, checkpoint sidecar, metadata) the same thread
left on another port is removed, after the new pair has landed. A port a live turn still serves,
other threads, other models and other slots on the new port are never touched.
"""
import json
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
OTHER_MODEL = "llama-8b"
OLD_PORT = 59501
NEW_PORT = 59500
THREAD = "sess-thread-A"
OTHER_THREAD = "sess-thread-B"
MSGS = [{"role": "user", "content": "hi"}]


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


def _install_engine(monkeypatch, kv_dir, tokens=100, bin_size=400, fail=False):
    """Fake engine: GET /slots lists one populated slot; POST action=save writes the bin and its sidecar
    and reports n_saved and n_written (the bin's byte count) as the real engine does."""
    posts = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return _Resp([{"id": 0, "n_prompt_tokens": tokens, "thread_id": THREAD,
                           "n_context": 4096, "n_remaining": 2048}])

        async def post(self, url, json=None, **kw):
            posts.append(url)
            if fail:
                raise RuntimeError("engine save failed")
            if "action=save" in url and json and "filename" in json:
                tmp = os.path.join(str(kv_dir), json["filename"])
                with open(tmp, "wb") as f:
                    f.write(b"k" * bin_size)
                with open(tmp + ".ckpt", "wb") as f:
                    f.write(b"c" * (bin_size // 2))
                # Like the real engine, report the effective token count and the bytes written.
                return _Resp({"n_saved": tokens, "n_written": bin_size})
            return _Resp({})

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client())
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return posts


def _write_pair(kv_dir, mgr, model, tid, port, sid=0, bin_size=600, tokens=100):
    """Lay down a complete saved pair the way a finished save leaves it."""
    th = mgr._thread_hash(tid)
    bin_fn = kv_save_fn(model, sid, th, port)
    (kv_dir / bin_fn).write_bytes(b"k" * bin_size)
    (kv_dir / (bin_fn + ".ckpt")).write_bytes(b"c" * (bin_size // 2))
    (kv_dir / kv_meta_fn(model, sid, th, port)).write_text(json.dumps({
        "thread_hash": th, "model_tag": model, "port": port, "slot_id": sid,
        "prompt_tokens": tokens, "bin_bytes": bin_size}))
    return bin_fn


def _names(kv_dir):
    return sorted(os.listdir(str(kv_dir)))


def _pair_files(bin_fn):
    return {bin_fn, bin_fn + ".ckpt", bin_fn[:-4] + ".json"}


async def _save(mgr):
    slot = Slot.new(MODEL, thread_id=THREAD, admission_ctx_len=100, client_meta={"messages": MSGS})
    return await mgr._save_slot_kv(
        NEW_PORT, MODEL, slot, thread_id_override=THREAD, admission_ctx_len_override=100,
        client_meta_override={"messages": MSGS})


@pytest.mark.asyncio
async def test_port_move_drops_the_old_port_pair_and_keeps_the_new_one(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    await _save(mgr)
    new = kv_save_fn(MODEL, 0, mgr._thread_hash(THREAD), NEW_PORT)
    names = set(_names(kv_dir))
    assert _pair_files(new) <= names, f"the new-port pair must land: {sorted(names)}"
    assert not (_pair_files(old) & names), (
        f"the old-port pair (bin, checkpoint sidecar, metadata) must be dropped once the new pair landed: "
        f"{sorted(_pair_files(old) & names)}")


@pytest.mark.asyncio
async def test_pairs_on_every_stale_port_and_slot_are_dropped(mgr, kv_dir, monkeypatch):
    a = _write_pair(kv_dir, mgr, MODEL, THREAD, 59501, sid=0)
    b = _write_pair(kv_dir, mgr, MODEL, THREAD, 59502, sid=1)
    _install_engine(monkeypatch, kv_dir)
    await _save(mgr)
    left = set(_names(kv_dir))
    assert not (_pair_files(a) | _pair_files(b)) & left


@pytest.mark.asyncio
async def test_other_threads_other_models_and_same_port_slots_are_never_touched(mgr, kv_dir, monkeypatch):
    other_thread = _write_pair(kv_dir, mgr, MODEL, OTHER_THREAD, OLD_PORT)
    other_model = _write_pair(kv_dir, mgr, OTHER_MODEL, THREAD, OLD_PORT)
    same_port_other_slot = _write_pair(kv_dir, mgr, MODEL, THREAD, NEW_PORT, sid=3)
    _install_engine(monkeypatch, kv_dir)
    await _save(mgr)
    left = set(_names(kv_dir))
    for keep in (other_thread, other_model, same_port_other_slot):
        assert _pair_files(keep) <= left, f"{keep} must be kept"


@pytest.mark.asyncio
async def test_a_port_a_live_turn_still_serves_for_the_thread_is_spared(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    mgr._active_slot = Slot.new(MODEL, thread_id=THREAD, admission_ctx_len=100, client_meta={"messages": MSGS})
    mgr._active_handle = SimpleNamespace(port=OLD_PORT)
    await _save(mgr)
    assert _pair_files(old) <= set(_names(kv_dir)), "the pair on a port still serving the thread must be kept"


@pytest.mark.asyncio
async def test_a_live_turn_on_a_different_port_does_not_spare_the_old_port(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    mgr._active_slot = Slot.new(MODEL, thread_id=THREAD, admission_ctx_len=100, client_meta={"messages": MSGS})
    mgr._active_handle = SimpleNamespace(port=NEW_PORT)
    await _save(mgr)
    assert not (_pair_files(old) & set(_names(kv_dir)))


@pytest.mark.asyncio
async def test_a_live_turn_whose_port_cannot_be_read_spares_every_old_pair(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    mgr._active_slot = Slot.new(MODEL, thread_id=THREAD, admission_ctx_len=100, client_meta={"messages": MSGS})
    mgr._active_handle = None
    await _save(mgr)
    assert _pair_files(old) <= set(_names(kv_dir))


@pytest.mark.asyncio
async def test_a_live_turn_of_another_thread_does_not_spare_this_threads_old_pair(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    mgr._active_slot = Slot.new(MODEL, thread_id=OTHER_THREAD, admission_ctx_len=100,
                                client_meta={"messages": MSGS})
    mgr._active_handle = SimpleNamespace(port=OLD_PORT)
    await _save(mgr)
    assert not (_pair_files(old) & set(_names(kv_dir)))


@pytest.mark.asyncio
async def test_a_save_that_fails_leaves_the_old_pair_in_place(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir, fail=True)
    await _save(mgr)
    assert _pair_files(old) <= set(_names(kv_dir)), "the old pair is the thread's only copy until the new one lands"


def _live(tid=THREAD):
    return Slot.new(MODEL, thread_id=tid, admission_ctx_len=100, client_meta={"messages": MSGS})


@pytest.mark.asyncio
async def test_a_resident_whose_active_turn_serves_the_thread_spares_its_port(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    resident = SimpleNamespace(port=OLD_PORT, active_slot=_live(), inflight=[])
    monkeypatch.setattr(mgr, "_model_residents", lambda: [resident])
    await _save(mgr)
    assert _pair_files(old) <= set(_names(kv_dir))


@pytest.mark.asyncio
async def test_a_resident_whose_in_flight_rider_serves_the_thread_spares_its_port(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    resident = SimpleNamespace(port=OLD_PORT, active_slot=None, inflight=[_live()])
    monkeypatch.setattr(mgr, "_model_residents", lambda: [resident])
    await _save(mgr)
    assert _pair_files(old) <= set(_names(kv_dir))


@pytest.mark.asyncio
async def test_a_resident_serving_another_thread_does_not_spare_the_old_port(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    resident = SimpleNamespace(port=OLD_PORT, active_slot=_live(OTHER_THREAD), inflight=[])
    monkeypatch.setattr(mgr, "_model_residents", lambda: [resident])
    await _save(mgr)
    assert not (_pair_files(old) & set(_names(kv_dir)))


@pytest.mark.asyncio
async def test_a_resident_serving_the_thread_with_no_readable_port_spares_every_old_pair(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)
    resident = SimpleNamespace(port=None, active_slot=_live(), inflight=[])
    monkeypatch.setattr(mgr, "_model_residents", lambda: [resident])
    await _save(mgr)
    assert _pair_files(old) <= set(_names(kv_dir))


@pytest.mark.asyncio
async def test_a_failed_live_state_scan_spares_every_old_pair(mgr, kv_dir, monkeypatch):
    old = _write_pair(kv_dir, mgr, MODEL, THREAD, OLD_PORT)
    _install_engine(monkeypatch, kv_dir)

    def _boom():
        raise RuntimeError("live state unreadable")

    monkeypatch.setattr(mgr, "_model_residents", _boom)
    await _save(mgr)
    assert _pair_files(old) <= set(_names(kv_dir)), "when the live ports cannot be read, nothing is dropped"
