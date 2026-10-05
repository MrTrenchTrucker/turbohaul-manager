"""Multi-slot KV save identity keying.

RED (on unpatched code): a `--parallel >= 2` engine with TWO populated
slots that have DIFFERENT thread_ids, saved via _save_slot_kv with a
``thread_id_override`` (the idle-holder teardown caller), produces a
WRONG-CONVERSATION bin: slot B's KV lands under slot A's thread_hash
because the override is applied to every engine slot in the loop.

GREEN (on patched code): the override is suppressed when >1 slot is
populated. Each engine slot keys on its OWN thread_id from the /slots
response, so slot A and slot B produce distinct, correctly-keyed bins.

Also proves the single-stream path (len(populated)==1) is byte-unchanged:
the override still applies and the bin key matches the override's hash.
"""
import json
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
from turbohaul.kv_policy import kv_save_fn, kv_meta_fn
from turbohaul.manager import TurbohaulManager


# --- fixtures (mirror test_wave_return.py) -----------------------------------

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


# --- httpx fake: GET /slots returns N populated slots; POST writes the bin ----

@pytest.fixture
def _make_resp():
    """Return a class whose .json() is configurable per request."""
    class _SlotsResp:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            return None
        def json(self):
            return self._payload
    return _SlotsResp


# --- the actual tests --

_QWEN = "qwen3.6-27b"
_PORT = 59500


@pytest.mark.asyncio
async def test_multi_slot_save_keys_each_slot_on_own_thread_id(
    mgr, kv_dir, monkeypatch, _make_resp
):
    """With 2 populated engine slots (different thread_ids) and a
    thread_id_override, each slot's KV bin must be keyed under its OWN
    thread_id — NOT the override's.

    RED: unpatched code saves BOTH under the override's thread_hash.
    GREEN: patched code saves each under its own thread_id's hash.
    """
    slot_a_tid = "sess-thread-A"
    slot_b_tid = "sess-thread-B"
    override_tid = "idle-holder-tid"  # what _unload_teardown stashes

    slots_payload = [
        {"id": 0, "n_prompt_tokens": 100, "thread_id": slot_a_tid,
         "n_context": 4096, "n_remaining": 2048},
        {"id": 1, "n_prompt_tokens": 50, "thread_id": slot_b_tid,
         "n_context": 4096, "n_remaining": 1024},
    ]

    class _Client:
        def __init__(self):
            self.posts = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            if "/slots" in url and "action=save" not in url:
                return _make_resp(slots_payload)
            return _make_resp({})

        async def post(self, url, json=None, **kw):
            self.posts.append((url, json))
            if "action=save" in url and json and "filename" in json:
                tmp_path = os.path.join(str(kv_dir), json["filename"])
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                with open(tmp_path, "wb") as f:
                    f.write(b"dummy kv cache data")
            return _make_resp({})

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client())
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)

    from turbohaul.slot import Slot
    slot = Slot.new(_QWEN, thread_id=override_tid, admission_ctx_len=100,
                     client_meta={"messages": [{"role": "user", "content": "hi"}]})

    result = await mgr._save_slot_kv(
        _PORT, _QWEN, slot,
        thread_id_override=override_tid,
        admission_ctx_len_override=100,
        client_meta_override={"messages": [{"role": "user", "content": "hi"}]},
    )

    # The manager's os.replace moves the temp .bin to the final name.
    # Enumerate what landed in SLOT_SAVE_DIR.
    files = sorted(os.listdir(str(kv_dir)))
    bin_files = [f for f in files if f.endswith(".bin")]

    # Each populated slot should produce its OWN bin, keyed on its own thread_id.
    hash_a = mgr._thread_hash(slot_a_tid)
    hash_b = mgr._thread_hash(slot_b_tid)
    hash_override = mgr._thread_hash(override_tid)

    # Correct behavior: each slot saved under its own thread_hash
    bin_a = set()
    bin_b = set()
    bin_override = set()
    for f in bin_files:
        if hash_a in f:
            bin_a.add(f)
        if hash_b in f:
            bin_b.add(f)
        if hash_override in f:
            bin_override.add(f)

    # Slot A's KV must be saved under slot_a_tid's hash, NOT the override's.
    assert bin_a, f"slot A bin missing under thread_hash({slot_a_tid})"
    assert not (bin_a & bin_override), \
        "SLOT A'S KV WAS SAVED UNDER THE IDLE-HOLDER OVERRIDE HASH — " \
        "wrong-conversation bin produced (RED)."

    # Slot B's KV must be saved under slot_b_tid's hash.
    assert bin_b, f"slot B bin missing under thread_hash({slot_b_tid})"
    assert not (bin_b & bin_override), \
        "SLOT B'S KV WAS SAVED UNDER THE IDLE-HOLDER OVERRIDE HASH — " \
        "wrong-conversation bin produced (RED)."


@pytest.mark.asyncio
async def test_single_slot_path_byte_unchanged_with_override(
    mgr, kv_dir, monkeypatch, _make_resp
):
    """Control: with exactly ONE populated slot, the thread_id_override
    is applied exactly as before — bin key == override thread_hash.
    The keying fix MUST NOT regress the single-stream path."""
    idle_tid = "idle-thread-id-single"
    slots_payload = [
        {"id": 0, "n_prompt_tokens": 100, "thread_id": "some-active-tid",
         "n_context": 4096, "n_remaining": 2048},
    ]

    class _Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, url):
            if "/slots" in url and "action=save" not in url:
                return _make_resp(slots_payload)
            return _make_resp({})
        async def post(self, url, json=None, **kw):
            if "action=save" in url and json and "filename" in json:
                tmp_path = os.path.join(str(kv_dir), json["filename"])
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                with open(tmp_path, "wb") as f:
                    f.write(b"dummy kv cache data")
            return _make_resp({})

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client())
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)

    from turbohaul.slot import Slot
    slot = Slot.new(_QWEN, thread_id=idle_tid, admission_ctx_len=100,
                     client_meta={"messages": [{"role": "user", "content": "hi"}]})

    await mgr._save_slot_kv(
        _PORT, _QWEN, slot,
        thread_id_override=idle_tid,
        admission_ctx_len_override=100,
        client_meta_override={"messages": [{"role": "user", "content": "hi"}]},
    )

    files = sorted(os.listdir(str(kv_dir)))
    bin_files = [f for f in files if f.endswith(".bin")]
    expected_hash = mgr._thread_hash(idle_tid)

    assert expected_hash in bin_files[0], \
        f"single-slot bin should be keyed on override hash ({expected_hash}), " \
        f"got: {bin_files}"
