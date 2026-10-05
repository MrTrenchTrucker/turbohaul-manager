"""The engine's action=save persists every populated KV cell, regardless of
the slot's current logical prompt length. A shorter prefill (e.g. the
clean-prefix strip-probe reprefilling over a longer prior generation) only
overwrites cells up to the new length; cells beyond it can still hold the
prior, longer generation's tokens, and a plain action=save writes those
too. The engine already supports capping the write at save_token_limit
tokens; the manager never sent it. This tests the one call site that sends
it: _save_slot_kv_inner's action=save POST, capped at n_prompt_tokens --
the engine's own just-reported prompt length for that slot (read from
/slots, the same value already fed to resolve_kv's "saved_tokens" field)
-- never hardcoded, never derived from turn count.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest tests/test_save_token_limit_capped_write.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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
from turbohaul.subprocess_mgr import SidecarHandle

_MODEL = "test-model-27b"
_PORT = 59900

pytestmark = pytest.mark.asyncio


# --- fixtures (mirror test_kv_save_observability.py) --

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
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, idle_hot_load_seconds=60),
        pull=PullConfig(),
    )
    m = TurbohaulManager(boot, runtime)
    m.runtime.kv.covered_scaffold_strip = False
    return m


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    import turbohaul.subprocess_mgr as subprocess_mgr
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
    def __init__(self, slots_payload, posts, probe_prompt_tokens=None, probe_usage=True):
        self._slots_payload = slots_payload
        self._posts = posts
        self._probe_prompt_tokens = probe_prompt_tokens
        self._probe_usage = probe_usage

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
            import os
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            tmp_path = os.path.join(SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
        if "/v1/chat/completions" in url:
            if not self._probe_usage:
                # Simulates a 200 probe reply whose usage block the manager
                # cannot parse (no token count recoverable) — the production
                # state in which a force_clean save must be declined.
                return _SaveResp({"status": "ok"})
            # Plain-probe regime (covered_scaffold_strip off): the real
            # sidecar's /v1/chat/completions reply carries the tokenized
            # prompt's usage. The manager's strict clean stamp
            # keys off usage.prompt_tokens as its save evidence, so the fake
            # reports the probe's own clean-render count — the test-supplied
            # value when given (that IS the render's count), else the slot's
            # engine-reported count as the default.
            _n = self._probe_prompt_tokens
            if _n is None:
                _n = (self._slots_payload[0].get("n_prompt_tokens") or 1) if self._slots_payload else 1
            return _SaveResp({"status": "ok",
                               "usage": {"prompt_tokens": _n,
                                         "completion_tokens": 0,
                                         "total_tokens": _n}})
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    import turbohaul.manager as manager_mod

    def _make(payload, probe_prompt_tokens=None, probe_usage=True):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _ProbeSaveClient(payload, posts, probe_prompt_tokens, probe_usage))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


def _msgs(k, marker="content"):
    filler = (marker + "-") * 100
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-{filler}"} for i in range(k)]


def _fake_handle(model_tag=_MODEL, port=_PORT):
    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _save_post(posts):
    """The single action=save POST, asserting exactly one fired."""
    saves = [(url, j) for (url, j) in posts if "action=save" in url]
    assert len(saves) == 1, f"expected exactly one action=save POST, got {saves}"
    return saves[0]


async def test_save_token_limit_sent_matches_engine_reported_prompt_tokens(
    mgr, kv_dir, make_httpx
):
    """The clean-prefix save (force_clean, the unload-seam flush's own path)
    must send save_token_limit == whatever the engine's /slots GET reported
    as n_prompt_tokens for that slot -- not hardcoded, not turn-derived."""
    handle = _fake_handle()
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 4321}])
    slot = Slot.new(_MODEL, thread_id="t-cap", admission_ctx_len=45000,
                     client_meta={"messages": _msgs(30, marker="CAPTEST")})
    await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
    _url, payload = _save_post(posts)
    assert payload.get("save_token_limit") == 4321, (
        f"save_token_limit missing or wrong: {payload}"
    )


async def test_save_token_limit_tracks_a_different_engine_value(mgr, kv_dir, make_httpx):
    """Non-vacuous companion: a different engine-reported n_prompt_tokens
    produces a different save_token_limit -- proves the value is actually
    threaded through, not a coincidental match on one fixture number."""
    handle = _fake_handle()
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 17}])
    slot = Slot.new(_MODEL, thread_id="t-two-resident-cap", admission_ctx_len=45000,
                     client_meta={"messages": _msgs(30, marker="CAPTEST2")})
    await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
    _url, payload = _save_post(posts)
    assert payload.get("save_token_limit") == 17, (
        f"save_token_limit missing or wrong: {payload}"
    )


async def test_filename_still_sent_alongside_save_token_limit(mgr, kv_dir, make_httpx):
    """Regression guard: adding save_token_limit must not drop the existing
    filename field the engine also requires on this same POST."""
    handle = _fake_handle()
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 999}])
    slot = Slot.new(_MODEL, thread_id="t-cap3", admission_ctx_len=45000,
                     client_meta={"messages": _msgs(30, marker="CAPTEST3")})
    await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
    _url, payload = _save_post(posts)
    assert isinstance(payload.get("filename"), str) and payload["filename"], payload


async def test_cap_prefers_the_clean_render_over_the_slot_prompt_length(
    mgr, kv_dir, make_httpx
):
    """THE POINT OF THE FIX. The slot's own prompt length also covers whatever was
    generated after the clean render -- tokens the client strips before sending the
    conversation back. Capping there leaves a surplus the restore cannot reconcile.
    When the clean render's own token count is known, the save must stop THERE.

    Extent check: the clean-stamp extent gate requires the render's extent
    and the slot's live extent to be coherent (within EPS) before a clean
    stamp — the old far-apart geometry (190000 vs 178756) is now exactly
    the poisoned-bin shape the gate declines (non-clean, slot cap), pinned by
    the extent-mismatch tests. Here the numbers are coherent-BUT-DISTINCT (178760
    vs 178756, delta 4 <= EPS 8): the render's cap must still win over the
    slot's number, and a fallback to the slot's value fails loudly."""
    handle = _fake_handle()
    slot_msgs = _msgs(30, marker="CAPTEST4")
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 178760}], probe_prompt_tokens=178756)
    # The register holds CleanPrefillEvidence (the probe
    # path re-records the same coherent evidence; the pre-seed documents the
    # shape). Coherent extent: 178756 vs slot 178760 (delta 4 <= EPS 8).
    from turbohaul.manager import CleanPrefillEvidence, _chain_fp, _prefix_hash_chain
    mgr._clean_prefill_tokens[_PORT] = CleanPrefillEvidence(
        count=178756, chain_fp=_chain_fp(_prefix_hash_chain(slot_msgs)), arm="plain")
    slot = Slot.new(_MODEL, thread_id="t-cap4", admission_ctx_len=45000,
                    client_meta={"messages": slot_msgs})
    await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
    _url, payload = _save_post(posts)
    assert payload.get("save_token_limit") == 178756, (
        "cap must come from the clean render, not the slot's prompt length; "
        f"got {payload.get('save_token_limit')}"
    )


async def test_force_clean_without_clean_boundary_is_declined(mgr, kv_dir, make_httpx):
    """A force_clean save with no
    clean-render count known must be DECLINED — no bin written at the slot's
    prompt length. Capping at the slot's value and still stamping the bin clean
    was the exact poison path: the engine byte-verified the restore, the engine
    reused zero of it, and the wave paid the full re-prefill anyway. Here the
    probe 200s but its usage block is unrecoverable (probe_usage=False) — the
    production state in which no clean boundary is known. The non-force_clean
    (teardown) save keeps the fallback cap, un-stamped — that variant is pinned
    in test_clean_stamp_integrity.py case 3."""
    handle = _fake_handle()
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 4242}], probe_usage=False)
    mgr._clean_prefill_tokens.pop(_PORT, None)
    slot = Slot.new(_MODEL, thread_id="t-cap5", admission_ctx_len=45000,
                    client_meta={"messages": _msgs(30, marker="CAPTEST5")})
    await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)
    saves = [(url, j) for (url, j) in posts if "action=save" in url]
    assert saves == [], (
        f"force_clean save without probe evidence must be declined, got {saves}"
    )
