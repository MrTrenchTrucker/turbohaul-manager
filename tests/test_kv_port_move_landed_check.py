"""The old-port pair is dropped only when the new pair is proven to have landed whole.

The engine reports the bytes it wrote (n_written). The manager compares that number with the size
of the file on disk. Only when the two are the same positive integer is the pair the thread left on
another port removed. A missing, mistyped or unequal number keeps the old pair and, for a number
that is present but wrong, logs one warning with both figures. The new pair is on disk in every case.
"""
import json
import logging
import os

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
MSGS = [{"role": "user", "content": "hi"}]
BIN_SIZE = 400
_MISSING = object()


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


def _install_engine(monkeypatch, kv_dir, reported, written=BIN_SIZE):
    """Fake engine that writes `written` bytes and reports `reported` as n_written (or none at all)."""

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
                body = {"n_saved": 100}
                if reported is not _MISSING:
                    body["n_written"] = reported
                return _Resp(body)
            return _Resp({})

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client())
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)


def _write_pair(kv_dir, mgr, port, bin_size=600):
    th = mgr._thread_hash(THREAD)
    bin_fn = kv_save_fn(MODEL, 0, th, port)
    (kv_dir / bin_fn).write_bytes(b"k" * bin_size)
    (kv_dir / (bin_fn + ".ckpt")).write_bytes(b"c" * (bin_size // 2))
    (kv_dir / kv_meta_fn(MODEL, 0, th, port)).write_text(json.dumps({
        "thread_hash": th, "model_tag": MODEL, "port": port, "slot_id": 0,
        "prompt_tokens": 100, "bin_bytes": bin_size}))
    return bin_fn


def _pair_files(bin_fn):
    return {bin_fn, bin_fn + ".ckpt", bin_fn[:-4] + ".json"}


async def _save(mgr):
    slot = Slot.new(MODEL, thread_id=THREAD, admission_ctx_len=100, client_meta={"messages": MSGS})
    return await mgr._save_slot_kv(
        NEW_PORT, MODEL, slot, thread_id_override=THREAD, admission_ctx_len_override=100,
        client_meta_override={"messages": MSGS})


async def _run(mgr, kv_dir, monkeypatch, reported, written=BIN_SIZE, with_prior_new=False):
    """Save on the new port with an old-port pair present. Returns (old pair names, names after)."""
    old = _write_pair(kv_dir, mgr, OLD_PORT)
    if with_prior_new:
        _write_pair(kv_dir, mgr, NEW_PORT, bin_size=600)
    _install_engine(monkeypatch, kv_dir, reported, written)
    await _save(mgr)
    return _pair_files(old), set(os.listdir(str(kv_dir)))


def _new_pair_landed(mgr, kv_dir):
    new = kv_save_fn(MODEL, 0, mgr._thread_hash(THREAD), NEW_PORT)
    names = set(os.listdir(str(kv_dir)))
    return _pair_files(new) <= names and os.path.getsize(str(kv_dir / new)) > 0


def _kept_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "old pair kept" in r.getMessage()]


@pytest.mark.asyncio
async def test_n_written_equal_to_the_file_size_drops_the_old_pair(mgr, kv_dir, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    old, names = await _run(mgr, kv_dir, monkeypatch, reported=BIN_SIZE)
    assert _new_pair_landed(mgr, kv_dir)
    assert not (old & names), f"equal sizes prove the landing, so the old pair goes: {sorted(old & names)}"
    assert not _kept_warnings(caplog)


@pytest.mark.asyncio
async def test_n_written_one_byte_too_big_keeps_the_old_pair_and_warns_once(mgr, kv_dir, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    old, names = await _run(mgr, kv_dir, monkeypatch, reported=BIN_SIZE + 1)
    assert _new_pair_landed(mgr, kv_dir)
    assert old <= names
    warned = _kept_warnings(caplog)
    assert len(warned) == 1, [r.getMessage() for r in caplog.records]
    text = warned[0].getMessage()
    assert str(BIN_SIZE) in text and str(BIN_SIZE + 1) in text


@pytest.mark.asyncio
async def test_n_written_one_byte_too_small_keeps_the_old_pair(mgr, kv_dir, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    old, names = await _run(mgr, kv_dir, monkeypatch, reported=BIN_SIZE - 1)
    assert _new_pair_landed(mgr, kv_dir)
    assert old <= names
    assert len(_kept_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_a_missing_n_written_keeps_the_old_pair(mgr, kv_dir, monkeypatch):
    old, names = await _run(mgr, kv_dir, monkeypatch, reported=_MISSING)
    assert _new_pair_landed(mgr, kv_dir)
    assert old <= names


@pytest.mark.asyncio
async def test_a_string_n_written_keeps_the_old_pair(mgr, kv_dir, monkeypatch):
    old, names = await _run(mgr, kv_dir, monkeypatch, reported=str(BIN_SIZE))
    assert _new_pair_landed(mgr, kv_dir)
    assert old <= names


@pytest.mark.asyncio
async def test_a_truncated_landing_that_passes_the_half_size_guard_keeps_the_old_pair(
        mgr, kv_dir, monkeypatch, caplog):
    """The engine reports 600 bytes but only 400 land: above half of the expected 600, so the save
    is accepted, yet the numbers differ and the old pair stays."""
    caplog.set_level(logging.DEBUG)
    old, names = await _run(mgr, kv_dir, monkeypatch, reported=600, written=BIN_SIZE, with_prior_new=True)
    assert _new_pair_landed(mgr, kv_dir)
    new = kv_save_fn(MODEL, 0, mgr._thread_hash(THREAD), NEW_PORT)
    assert os.path.getsize(str(kv_dir / new)) == BIN_SIZE, "the truncated file is the one on disk"
    assert old <= names
    assert len(_kept_warnings(caplog)) == 1
