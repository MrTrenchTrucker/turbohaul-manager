"""Wave-return manager delta — cold-path clean save + restore.

Covers the 2 behavioural changes in manager.py (the WARM path + the kv_policy gate
are unchanged and stay covered by test_ws2_classifier.py / test_kv_policy.py):

  * empty-chain save fallback in _save_slot_kv: when slot.context is empty,
    persist a REAL prefix-comparable chain from the admission messages
    (client_meta['messages']) instead of [] (which is permanently unrestorable).
  * cold-path observable in _restore_slot_kv: a cold restore whose WINNING bin
    is a clean_prefix bin (= a wave-return) tags resolved_from='wave-return-clean-
    restore' and bumps self._kv_classifier_wave_return; a non-clean valid restore
    keeps 'restore-prefix-valid' and leaves the counter untouched.
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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot


# --- fixtures (mirror test_ws2_classifier.py) ------------------------------------
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
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    """Isolated SLOT_SAVE_DIR (re-imported per call inside the manager methods)."""
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


# --- restore-path httpx fake (POST action=restore records + raise_for_status ok) --
class _FakeResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _FakeClient:
    """Records every POST so tests can assert the restore fired (or not)."""

    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        return _FakeResp()


@pytest.fixture
def posts(monkeypatch):
    recorded = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _FakeClient(recorded))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return recorded


# --- save-path httpx fake (GET /slots + POST action=save writes the temp bin) ------
class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _SaveClient:
    """Fake the engine for _save_slot_kv: GET /slots reports one populated slot; the
    action=save POST materialises the temp .bin so the manager's os.replace lands."""

    def __init__(self, slots_payload):
        self._slots_payload = slots_payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url and "action=save" not in url:
            return _SaveResp(self._slots_payload)
        return _SaveResp({})

    async def post(self, url, json=None, **kw):
        if "action=save" in url and json and "filename" in json:
            # Engine writes the KV dump to the temp path; mirror that so os.replace works.
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
        return _SaveResp({"status": "ok"})


@pytest.fixture
def save_httpx(monkeypatch):
    """Patch manager_mod.httpx so _save_slot_kv sees one populated slot (id=0)."""
    payload = [{"id": 0, "n_prompt_tokens": 100, "id_task": 0}]

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _SaveClient(payload))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    return payload


# turn primitives + an example model tag (path-safe: no '/','\\','..')
_SYS = {"role": "system", "content": "system prompt long enough to matter"}
_U1 = {"role": "user", "content": "first user turn"}
_U2 = {"role": "user", "content": "second user turn"}
_A1 = {"role": "assistant", "content": "A"}
_QWEN = "example-27b"


def _write_bin(kv_dir, model_tag, port, thread_id, sid, chain, *, clean,
               prompt_len=40000, prompt_tokens=12345):
    """Write a saved bin (.bin + .json meta) for a thread. clean=True stamps a
    clean_prefix bin (the wave-return anchor); clean=False a normal saved bin."""
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(model_tag, sid, th, port)
    bin_fn = meta_fn[:-5] + ".bin"
    (kv_dir / bin_fn).write_bytes(b"\x00")  # non-empty bin so existence check passes
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": prompt_tokens,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": model_tag, "slot_id": sid, "port": port, "clean_prefix": clean,
    }))
    return bin_fn


# ================================================================================
# empty-chain save fallback in _save_slot_kv
# ================================================================================
@pytest.mark.asyncio
async def test_fix_b_empty_context_falls_back_to_admission_messages(mgr, kv_dir, save_httpx):
    """slot.context empty -> the saved meta chain is derived from the admission
    messages (client_meta['messages']), NOT [] (which would be unrestorable)."""
    messages = [_SYS, _U1, _U2]
    slot = Slot.new(_QWEN, thread_id="t", context=None, client_meta={"messages": messages})

    await mgr._save_slot_kv(59500, _QWEN, slot)

    meta_fn = kv_meta_fn(_QWEN, 0, mgr._thread_hash("t"), 59500)
    meta = json.loads((kv_dir / meta_fn).read_text())
    expected = _prefix_hash_chain(messages)
    assert expected, "sanity: the admission messages produce a non-empty chain"
    assert meta["hash_chain"] == expected              # REAL prefix-comparable chain
    assert meta["hash_chain"] != []                    # not the unrestorable empty case
    assert meta["n_context_turns"] == len(messages)    # auto-follows len(hash_chain)


@pytest.mark.asyncio
async def test_fix_b_present_context_is_used_unchanged(mgr, kv_dir, save_httpx):
    """slot.context present -> chain comes from context; client_meta is NOT consulted
    (fallback only fires when context is empty)."""
    context = [_SYS, _U1, _U2]
    # A DIFFERENT messages list proves precedence: if the fallback wrongly fired we'd
    # see this 1-turn chain instead of the 3-turn context chain.
    slot = Slot.new(_QWEN, thread_id="t", context=context,
                    client_meta={"messages": [_SYS]})

    await mgr._save_slot_kv(59500, _QWEN, slot)

    meta_fn = kv_meta_fn(_QWEN, 0, mgr._thread_hash("t"), 59500)
    meta = json.loads((kv_dir / meta_fn).read_text())
    assert meta["hash_chain"] == _prefix_hash_chain(context)
    assert meta["hash_chain"] != _prefix_hash_chain([_SYS])   # did NOT use client_meta
    assert meta["n_context_turns"] == len(context)


# ================================================================================
# cold-path wave-return observable in _restore_slot_kv
# ================================================================================
@pytest.mark.asyncio
async def test_f2_cold_clean_restore_tags_wave_return(mgr, kv_dir, posts):
    """The winning cold bin is a clean_prefix bin (valid prefix of incoming) ->
    resolved_from='wave-return-clean-restore' + wave_return counter incremented."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain, clean=True)
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])  # clean_chain ⊑ inc (strict extension)
    slot = Slot.new(_QWEN, thread_id="t", admission_ctx_len=50000, admission_hash_chain=inc)

    await mgr._restore_slot_kv(59500, _QWEN, slot)

    assert mgr._kv_classifier_wave_return == 1
    assert mgr._kv_classifier_forced == 0              # WARM counter untouched
    last = mgr._kv_classifier_last
    assert last["resolved_from"] == "wave-return-clean-restore"
    assert last["event_type"] == "continuation"        # events dict unchanged
    assert last["forced_clean_restore"] is False
    assert mgr._kv_classifier_counts.get("continuation") == 1
    # the restore POST actually fired for the clean bin
    assert len(posts) == 1
    url, body = posts[0]
    assert "action=restore" in url and body == {"filename": bin_fn}


@pytest.mark.asyncio
async def test_f2_cold_nonclean_restore_keeps_prefix_valid(mgr, kv_dir, posts):
    """A valid but NON-clean cold restore keeps resolved_from='restore-prefix-valid'
    and does NOT touch the wave_return counter."""
    saved_chain = _prefix_hash_chain([_SYS, _U1])
    bin_fn = _write_bin(kv_dir, _QWEN, 59500, "t", 0, saved_chain, clean=False)
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])  # saved_chain ⊑ inc
    slot = Slot.new(_QWEN, thread_id="t", admission_ctx_len=50000, admission_hash_chain=inc)

    await mgr._restore_slot_kv(59500, _QWEN, slot)

    assert mgr._kv_classifier_wave_return == 0          # counter UNCHANGED
    last = mgr._kv_classifier_last
    assert last["resolved_from"] == "restore-prefix-valid"
    assert last["event_type"] == "continuation"
    assert last["forced_clean_restore"] is False
    assert len(posts) == 1
    url, body = posts[0]
    assert "action=restore" in url and body == {"filename": bin_fn}


@pytest.mark.asyncio
async def test_f2_wave_return_surfaced_on_status(mgr, kv_dir, posts):
    """The cold counter is exposed on /status as kv_classifier.wave_return_restores."""
    clean_chain = _prefix_hash_chain([_SYS, _U1])
    _write_bin(kv_dir, _QWEN, 59500, "t", 0, clean_chain, clean=True)
    inc = _prefix_hash_chain([_SYS, _U1, _A1, _U2])
    slot = Slot.new(_QWEN, thread_id="t", admission_ctx_len=50000, admission_hash_chain=inc)

    await mgr._restore_slot_kv(59500, _QWEN, slot)

    snap = mgr.status_snapshot()
    assert snap["kv_classifier"]["wave_return_restores"] == 1
    assert snap["kv_classifier"]["forced_clean_restores"] == 0
