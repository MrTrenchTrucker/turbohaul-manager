"""Fast-reload KV restore under the multi-process dispatcher.

Problem: an is_main precomputed KV, after being evicted -> RAM, is NOT
reused on reload -> cold re-prefill of the full ~250k ctx. Root cause: under the
dispatcher the clean-prefix save hard-disables (global sidecar-count gate) AND the
teardown bin is written with an EMPTY hash_chain -> the restore side rejects it
(restore-diverged-fresh). This suite pins the SAVE-side behaviour (manager.py only):

  (1) gate on per-ENGINE handle.parallel, not global max_parallel_sidecars
  (2) populate r.idle_client_meta / r.idle_admission_ctx_len (parallel==1)
  (3) honor client_meta_override in the hash_chain writeout (slot=None teardown)
  (4) clean-flush at dispatcher teardown for parallel==1

Acceptance tests: per-engine gate (+ parallel>=2 negative), save+restore (prefix-valid),
cap<=1 regression. kv_policy.py + safety.py are BYTE-UNTOUCHED (restore logic unchanged;
we only feed it a non-empty saved_chain that was always intended).

Fixtures/fakes mirror tests/test_lagreducer.py (the established _probe_and_save_clean_kv
/ _save_slot_kv save-path harness).
"""
import json
import os
import types

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
    KVConfig,
)
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn, resolve_kv
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

_QWEN = "qwen3.6-27b"
_PORT = 59500


# --- fixtures (mirror test_lagreducer.py) ---------------------------------------
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
    # Pin the SAVE-probe scaffold-strip OFF: the flag moved from
    # an env var (TURBOHAUL_COVERED_SCAFFOLD_STRIP, which the old autouse fixture
    # pinned) to runtime.kv.covered_scaffold_strip (DEFAULT ON). This suite's fake
    # httpx client speaks the plain messages /v1/chat/completions probe transport
    # only, so the fixture pins the config flag to keep the flag-OFF probe path —
    # the gate under test (per-engine gate / observed-concurrency) is probe-transport-
    # independent and evaluated before any prefill branch.
    runtime = RuntimeConfig(
        queue=QueueConfig(), pull=PullConfig(),
        kv=KVConfig(covered_scaffold_strip=False),
    )
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ProbeSaveClient:
    def __init__(self, slots_payload, posts):
        self._slots_payload = slots_payload
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url and "action=save" not in url:
            return _SaveResp(self._slots_payload)
        return _SaveResp({})

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        if "action=save" in url and json and "filename" in json:
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
        if "/v1/chat/completions" in url:
            # Plain-probe regime (covered_scaffold_strip off): the real
            # sidecar's /v1/chat/completions reply carries the tokenized
            # prompt's usage. The manager's strict clean stamp
            # keys off usage.prompt_tokens as its save evidence, so the fake
            # reports the slot's own engine-reported count — exactly what the
            # engine would return for the same render.
            _n = (self._slots_payload[0].get("n_prompt_tokens") or 1) if self._slots_payload else 1
            return _SaveResp({"status": "ok",
                               "usage": {"prompt_tokens": _n,
                                         "completion_tokens": 0,
                                         "total_tokens": _n}})
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    def _make(payload):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _ProbeSaveClient(payload, posts))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


# --- helpers --------------------------------------------------------------------
def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _handle(parallel=1, port=_PORT):
    return types.SimpleNamespace(parallel=parallel, port=port)


def _read_meta(kv_dir, model_tag, sid, thread_id, port=_PORT):
    th = TurbohaulManager._thread_hash(thread_id)
    return json.loads((kv_dir / kv_meta_fn(model_tag, sid, th, port)).read_text())


def _n_save_posts(posts):
    return sum(1 for (url, _j) in posts if "action=save" in url)


# ================================================================================
# Test 1 — per-engine gate: the clean-prefix gate keys on per-ENGINE handle.parallel
# ================================================================================
@pytest.mark.asyncio
async def test_proceeds_when_handle_parallel_1_despite_sidecars(mgr, kv_dir, make_httpx):
    """A deployment can run max_parallel_sidecars>1 (different models on different ports), but a
    parallel==1 main owns its sidecar and serves serially. A global `_mp != 1`
    gate would abort every clean save (=> cold restore); the gate is `_hp != 1`, so
    a parallel==1 engine PROCEEDS past it and saves. (Fails with a global gate: posts would be [].)"""
    mgr.runtime.queue.max_parallel_sidecars = 24            # global count != 1
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 5000}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(6)}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(parallel=1), slot, save_to_disk=True)

    assert _n_save_posts(posts) == 1                        # proceeded past the per-engine gate


@pytest.mark.asyncio
async def test_still_skips_parallel_ge2_recurrent_engine(mgr, kv_dir, make_httpx):
    """No-regression for a parallel>=2 (small recurrent sub-agent model) engine: `_hp != 1`
    still returns at the gate -> no clean save attempted (a recurrent engine cannot reuse
    a disk-restored bin past divergence anyway; saving would only burn tmpfs)."""
    mgr.runtime.queue.max_parallel_sidecars = 24
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 5000}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(6)}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(parallel=3), slot, save_to_disk=True)

    assert posts == []                                      # parallel>=2 still gated off


# ================================================================================
# Test 2 — teardown metadata + hash_chain writeout: the dispatcher teardown bin carries a RESTORABLE chain
# ================================================================================
@pytest.mark.asyncio
async def test_fix_b_teardown_bin_is_restorable(mgr, kv_dir, make_httpx):
    """The dispatcher teardown saves with slot=None + client_meta_override (the stashed
    idle meta). Without the override the hash_chain writeout reads only `slot` (None) -> writes
    hash_chain=[] -> the restore side returns restore-diverged-fresh -> cold 250k
    re-prefill. The writeout honors client_meta_override, so the bin carries the
    real chain + prompt_len. Then resolve_kv('restore', ...) accepts it as a valid
    prefix. (Fails without it: the hash_chain assertion is [] and the decision is
    restore-diverged-fresh.)"""
    messages = _msgs(6)
    make_httpx([{"id": 0, "n_prompt_tokens": 5000}])   # installs the fake sidecar (httpx)

    # slot=None teardown save with the stashed idle overrides (as the dispatcher populates them)
    await mgr._save_slot_kv(
        _PORT, _QWEN, None,
        thread_id_override="t",
        admission_ctx_len_override=50000,
        client_meta_override={"messages": messages},
    )

    meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert meta["hash_chain"] == _prefix_hash_chain(messages)   # NON-EMPTY, from override
    assert meta["hash_chain"]                                   # explicit non-empty guard
    assert meta["n_context_turns"] == len(messages)
    assert meta["prompt_len"] == 50000                         # honored admission_ctx_len_override

    # restore side (kv_policy UNTOUCHED): same-session next turn EXTENDS the saved prefix
    incoming = _prefix_hash_chain(messages + [{"role": "user", "content": "next same-session turn"}])
    d = resolve_kv("restore", {"thread_id": "t", "model_tag": _QWEN}, {
        "saved_tokens": meta["prompt_tokens"], "saved_len": meta["prompt_len"],
        "incoming_len": meta["prompt_len"] + 500,
        "saved_thread_id": "t",
        "saved_chain": meta["hash_chain"], "incoming_chain": incoming,
    })
    assert d.do_it and d.resolved_from == "restore-prefix-valid"


def test_fix_b_regression_empty_chain_would_diverge():
    """Regression witness (proves the hash_chain assertion above is non-tautological): feed the
    restore chokepoint an EMPTY saved_chain and confirm it rejects with exactly
    restore-diverged-fresh -> full re-prefill. This is the behavior the writeout change removes.
    Pure kv_policy (byte-locked) chokepoint call — no manager/engine needed."""
    incoming = _prefix_hash_chain(_msgs(6) + [{"role": "user", "content": "next"}])
    d = resolve_kv("restore", {"thread_id": "t", "model_tag": _QWEN}, {
        "saved_tokens": 5000, "saved_len": 50000, "incoming_len": 50500,
        "saved_thread_id": "t",
        "saved_chain": [],              # <- the empty chain a teardown without the override writes
        "incoming_chain": incoming,
    })
    assert not d.do_it and d.resolved_from == "restore-diverged-fresh"


# ================================================================================
# Test 3 — Regression: cap<=1 path is byte-identical (change is inert)
# ================================================================================
@pytest.mark.asyncio
async def test_one_resident_cap_path_unchanged(mgr, kv_dir, make_httpx):
    """max_parallel_sidecars=1 (cap<=1 fixtures/tests) keeps _mp==1, so the old gate
    `_mp!=1 or _hp!=1` and the new gate `_hp!=1` are decision-identical for a
    parallel==1 engine: both PROCEED. The clean save fires exactly as before the change."""
    assert mgr.runtime.queue.max_parallel_sidecars == 1     # default cap<=1
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 5000}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(6)}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(parallel=1), slot, save_to_disk=True)

    assert _n_save_posts(posts) == 1                        # unchanged cap<=1 behavior
