"""Clean-stamp integrity for the evict-return KV path.

Swap-and-return reuse is deterministic on WHICH bin gets saved, not on
chance: a bin saved at a clean turn boundary (seed-only) restores and the
engine reuses the whole saved prefix; a bin saved post-generation (seed +
the entire assistant reply) restores byte-perfect and the engine then
reuses ZERO of it, because the saved token record diverges from the
think-stripped resend near its end. The manager-side defect is the SAVE
side: a force_clean save with no probe evidence fell back to the slot's
own n_prompt_tokens (which covers the generated reply) and still stamped
the bin clean_prefix=True.

This suite pins the stamp invariant and its restore-side backstop:
  1. force_clean save + single populated slot + NO recorded probe evidence
     -> DECLINED: no save POST, no meta on disk (the exact reintroducing
     edit is the old fallback cap + unconditional clean stamp);
  2. force_clean save WITH recorded probe evidence -> saves, meta
     clean_prefix=True, bin_provenance="clean_prefill", cap == the probe
     value (not the slot's post-generation count);
  3. the ordinary (non-force_clean) save is inert: no probe evidence
     needed, clean_prefix=False, cap = n_prompt_tokens (green both sides
     of the fix — regression guard);
  4. the cold-restore size-guard classifier declines a bin that the
     recorded-probe reference cannot cover (realistic field numbers
     as seen in practice), fails open when the reference is unknown, and leaves the
     with-think (polluted) lane out of scope unchanged;
  5. characterization of the pure first-divergent-turn helper.

kv_policy.py is byte-locked: case 4 exercises the classifier through its
existing public surface only, no source changes there.
"""
import json
import os

import pytest

import turbohaul.manager as manager_mod
import turbohaul.subprocess_mgr as subprocess_mgr
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
    KVConfig,
)
from turbohaul.manager import (
    TurbohaulManager, _classify_cold_restore_size, _first_divergent_turn_index,
)
from turbohaul.kv_policy import kv_meta_fn
from turbohaul.slot import Slot

_QWEN = "qwen3.6-27b"
_PORT = 59500

# --- fixtures (mirror the fast-reload KV test fixtures) -------------
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


# --- helpers ----------------------------------------------------------------
def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _slot():
    return Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": _msgs(6)}, admission_ctx_len=50000)


def _evidence(count, msgs, arm="plain"):
    """Register entries are CleanPrefillEvidence — the
    recorded count BOUND to the probed context's turn-hash fingerprint with
    the identical derivation the probe writers use. Imported lazily so the
    file stays importable on the pre-fix base (the tripwire's RED run)."""
    from turbohaul.manager import CleanPrefillEvidence, _chain_fp, _prefix_hash_chain
    return CleanPrefillEvidence(
        count=count, chain_fp=_chain_fp(_prefix_hash_chain(msgs)), arm=arm)


def _read_meta(kv_dir, model_tag, sid, thread_id, port=_PORT):
    th = TurbohaulManager._thread_hash(thread_id)
    return json.loads((kv_dir / kv_meta_fn(model_tag, sid, th, port)).read_text())


def _save_posts(posts):
    return [j for (url, j) in posts if "action=save" in url and j]


# ================================================================================
# case 1 — TRIPWIRE: force_clean without probe evidence is DECLINED
# (reads RED on base: base falls back to the slot count and stamps clean)
# ================================================================================
@pytest.mark.asyncio
async def test_case1_force_clean_without_probe_evidence_is_declined(mgr, kv_dir, make_httpx):
    """The reintroducing edit is the old fallback: cap = n_prompt_tokens +
    unconditional clean_prefix stamp. A force_clean save whose port has NO
    recorded clean-prefill token count must not write a bin at all — the
    slot dump is post-generation and a clean stamp on it is the poison."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 87401}])
    assert mgr._clean_prefill_tokens.get(_PORT) is None

    await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)

    # no engine save POST issued, no bin and no meta sidecar on disk
    assert _save_posts(posts) == []
    assert [f for f in os.listdir(kv_dir) if f.endswith(".bin")] == []
    meta_files = [f for f in os.listdir(kv_dir) if f.endswith(".json")]
    assert meta_files == []


@pytest.mark.asyncio
async def test_case1b_force_clean_decline_keeps_previous_bin(mgr, kv_dir, make_httpx):
    """A prior GOOD bin must survive a declined force_clean re-save (the
    'previous bin kept' half of the decline contract): a clean meta on disk
    is not overwritten, and a second force_clean attempt with no evidence
    writes nothing over it."""
    # seed an existing clean bin + meta pair (as a prior evidenced save wrote)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 81902}])
    mgr._clean_prefill_tokens[_PORT] = _evidence(81902, _slot().client_meta["messages"])
    await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)
    prior_meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert prior_meta["clean_prefix"] is True
    _meta_fn = kv_meta_fn(_QWEN, 0, TurbohaulManager._thread_hash("t"), _PORT)
    _bin_fn = _meta_fn[:-5] + ".bin"
    prior_bin_bytes = (kv_dir / _bin_fn).read_bytes()

    # probe evidence lost (engine respawned / stale entry popped) -> decline
    mgr._clean_prefill_tokens.pop(_PORT, None)
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 87401}])
    await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)
    assert _save_posts(posts) == []

    after = _read_meta(kv_dir, _QWEN, 0, "t")
    assert after["clean_prefix"] is True           # not demoted to False
    assert after["prompt_tokens"] == 81902         # old meta intact
    assert (kv_dir / _bin_fn).read_bytes() == prior_bin_bytes


# ================================================================================
# case 2 — TRIPWIRE: force_clean WITH probe evidence saves a clean bin
# (reads RED on base: base has no bin_provenance meta field)
# ================================================================================
@pytest.mark.asyncio
async def test_case2_force_clean_with_probe_evidence_saves_clean(mgr, kv_dir, make_httpx):
    """With the engine's own recorded think-free render count for the port,
    BOUND to this context's turn-hash fingerprint (coherent evidence —
    the count matches the slot's live count within EPS and the chain fp
    matches the save context), the force_clean save proceeds, caps at the
    PROBE value, and stamps the bin clean with its provenance recorded."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 81902}])
    mgr._clean_prefill_tokens[_PORT] = _evidence(81902, _slot().client_meta["messages"])
    await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)

    sp = _save_posts(posts)
    assert len(sp) == 1
    assert sp[0]["save_token_limit"] == 81902      # probe value (coherent evidence)
    meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert meta["clean_prefix"] is True
    assert meta["bin_provenance"] == "clean_prefill"


# ================================================================================
# case 3 — REGRESSION GUARD (green both sides): the inert path is unchanged
# ================================================================================
@pytest.mark.asyncio
async def test_case3_plain_save_without_probe_stays_inert(mgr, kv_dir, make_httpx):
    """A non-force_clean teardown save needs no probe evidence: it caps at
    the slot's own n_prompt_tokens and stamps clean_prefix=False, exactly
    as before the fix (the strict gate must not reach the inert path)."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 87401}])
    await mgr._save_slot_kv(_PORT, _QWEN, _slot())   # force_clean defaults False

    sp = _save_posts(posts)
    assert len(sp) == 1
    assert sp[0]["save_token_limit"] == 87401        # cap = n_prompt_tokens
    meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert meta["clean_prefix"] is False


# ================================================================================
# case 4 — restore-side backstop: size-guard classifier on the recorded
# probe fallback (pure fn, byte-locked surface)
# ================================================================================
def test_case4_size_guard_recorded_probe_fallback():
    """The fallback reference is the recorded think-free render count. A
    post-generation bin (87,489 saved) cannot be covered by the recorded
    render (87,000) plus margin -> DECLINE (fresh prefill, the safe cost);
    with NO reference at all the guard still fails open; the with-think
    polluted lane stays out of scope, unchanged."""
    margin = 32
    # example field numbers: saved 87,489 vs recorded 87,000 + margin
    p, c = _classify_cold_restore_size("clean", True, 87489, 87000 + margin, margin)
    assert (p, c) == (False, "size_decline")
    # one token under the recorded render (within margin): covered -> proceed
    p, c = _classify_cold_restore_size("clean", True, 87489, 87489 - margin, margin)
    assert (p, c) == (True, "size_ok")
    # no live probe AND no recorded reference: fail open (no fresh-restore regression)
    p, c = _classify_cold_restore_size("clean", True, 87489, None, margin)
    assert (p, c) == (True, "unknown_fail_open")
    # with-think lane: deliberately out of scope, unchanged
    p, c = _classify_cold_restore_size("clean", False, 87489, 81902, margin)
    assert (p, c) == (True, "polluted_out_of_scope")


# ================================================================================
# case 5 — pure helper characterization (characterization only)
# ================================================================================
def test_case5_first_divergent_turn_index():
    a = ["h1", "h2", "h3", "h4"]
    # saved is a valid prefix: divergence sits exactly at the extension boundary
    assert _first_divergent_turn_index(a, a + ["h5"]) == 4
    # mid-chain divergence
    assert _first_divergent_turn_index(a, ["h1", "X", "h3"]) == 1
    # degenerate shapes: never raises
    assert _first_divergent_turn_index([], []) == 0
    assert _first_divergent_turn_index([], ["h1"]) == 0
    assert _first_divergent_turn_index(a, []) == 0
    assert _first_divergent_turn_index(None, None) == 0


# ================================================================================
# case 6 -- follow-up checks: probe-failure evidence hygiene
# + the ops kill-switch semantics
# ================================================================================
def _resp403():
    import httpx
    return httpx.HTTPStatusError(
        "server error",
        request=httpx.Request("POST", f"http://127.0.0.1:{_PORT}/v1/chat/completions"),
        response=httpx.Response(403),
    )


@pytest.mark.asyncio
async def test_case6_plain_probe_4xx_declines_save_and_drops_stale_evidence(
    mgr, kv_dir, monkeypatch, caplog
):
    """A 4xx'd probe must decline the clean save (no save POST) AND drop any
    stale evidence entry for the port — the register's documented invariant
    (a probe removes the entry whenever it cannot obtain a fresh count).
    The import-time _HTTPX_HTTP_STATUS_ERROR binding is what makes this
    branch reachable with the monkeypatched fake httpx below."""
    import httpx
    import logging
    from turbohaul.manager import KV_DECLINE_PROBE_POST_FAILED

    class _Resp403:
        def raise_for_status(self):
            raise _resp403()

        def json(self):
            return {}

    class _Client403:
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
            return _Resp403()

    posts = []

    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client403([{"id": 0, "n_prompt_tokens": 87401}], posts))
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
    # stale evidence from a pre-respawn probe must NOT survive the 4xx
    mgr._clean_prefill_tokens[_PORT] = _evidence(81902, _msgs(6))

    import types
    handle = types.SimpleNamespace(parallel=1, port=_PORT)
    slot = Slot.new(_QWEN, thread_id="t-4xx", context=None,
                    client_meta={"messages": _msgs(6)}, admission_ctx_len=50000)
    with caplog.at_level(logging.INFO):
        await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True)

    assert _save_posts(posts) == []                       # declined, nothing written
    assert mgr._clean_prefill_tokens.get(_PORT) is None   # stale entry dropped
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "KV_SAVE_DECLINE" in joined
    assert KV_DECLINE_PROBE_POST_FAILED in joined


@pytest.mark.asyncio
async def test_case6b_evidence_gate_kill_switch_restores_pre_fix_stamp(
    mgr, kv_dir, make_httpx, monkeypatch
):
    """TURBOHAUL_CLEAN_STAMP_EVIDENCE_GATE=0 is the operator's explicit lever:
    a force_clean save without probe evidence is NOT declined — it saves and
    stamps clean on the fallback cap (pre-fix behaviour), loud in the logs.
    Pinned so a future 'simplification' cannot silently remove the escape
    hatch operators rely on."""
    monkeypatch.setenv("TURBOHAUL_CLEAN_STAMP_EVIDENCE_GATE", "0")
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 87401}])
    assert mgr._clean_prefill_tokens.get(_PORT) is None

    await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)

    sp = _save_posts(posts)
    assert len(sp) == 1                                   # NOT declined
    assert sp[0]["save_token_limit"] == 87401             # fallback cap
    meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert meta["clean_prefix"] is True                   # pre-fix stamp restored


# ================================================================================
# case 1c/2c/3c/6c -- tripwires: the extent
# gate, the context-binding gate, the poison guard, and the single-lever
# kill-switch interaction. All four FAIL on the pre-fix base (the base has
# none of the three checks) and PASS on the fix — the tripwire contract.
# ================================================================================
@pytest.mark.asyncio
async def test_case1c_extent_mismatch_declines_stamp_not_save(mgr, kv_dir, make_httpx, caplog):
    """Extent check: evidence for a DIFFERENT extent than the slot's
    live count (e.g. a poisoned bin whose render count differs from the slot's)
    must decline the CLEAN STAMP, not the save: the
    bin is written NON-CLEAN on the slot cap, attributed KV_SAVE_DECLINE
    EVIDENCE_MISMATCH. (RED on base: the base stamps the probe cap clean.)"""
    import logging
    from turbohaul.manager import KV_DECLINE_EVIDENCE_MISMATCH
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 87401}])
    mgr._clean_prefill_tokens[_PORT] = _evidence(81902, _slot().client_meta["messages"])

    with caplog.at_level(logging.INFO):
        await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)

    sp = _save_posts(posts)
    assert len(sp) == 1                                    # save PROCEEDS
    assert sp[0]["save_token_limit"] == 87401              # slot cap, not the probe value
    meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert meta["clean_prefix"] is False                   # stamp declined
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "KV_SAVE_DECLINE" in joined
    assert KV_DECLINE_EVIDENCE_MISMATCH in joined


@pytest.mark.asyncio
async def test_case2c_chain_mismatch_declines_stamp_not_save(mgr, kv_dir, make_httpx, caplog):
    """Context-binding check: evidence with the RIGHT extent but bound to a
    DIFFERENT context (the chain fp of another messages list) must decline
    the stamp, not the save — non-clean bin on the slot cap, attributed
    KV_SAVE_DECLINE EVIDENCE_MISMATCH. (RED on base: the base never checks
    the binding and stamps the coherent-extent evidence clean.)"""
    import logging
    from turbohaul.manager import (
        CleanPrefillEvidence, KV_DECLINE_EVIDENCE_MISMATCH, _chain_fp, _prefix_hash_chain,
    )
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 81902}])
    other_msgs = [{"role": "user", "content": "a different context entirely"}]
    mgr._clean_prefill_tokens[_PORT] = CleanPrefillEvidence(
        count=81902, chain_fp=_chain_fp(_prefix_hash_chain(other_msgs)), arm="strip")

    with caplog.at_level(logging.INFO):
        await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)

    sp = _save_posts(posts)
    assert len(sp) == 1
    assert sp[0]["save_token_limit"] == 81902
    meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert meta["clean_prefix"] is False                   # stamp declined
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "KV_SAVE_DECLINE" in joined
    assert KV_DECLINE_EVIDENCE_MISMATCH in joined


@pytest.mark.asyncio
async def test_case3c_poisoned_context_declines_stamp(mgr, kv_dir, make_httpx, caplog):
    """Poison guard (anti-churn): a context whose clean bin already
    proved poison (retired after a reuse-zero result) must NOT be re-stamped —
    coherent evidence is not enough; the save proceeds NON-CLEAN and the
    decline is attributed KV_DECLINE_POISON_REPROBE. (RED on base: the base
    has no poison set and stamps the coherent evidence clean.)"""
    import logging
    from turbohaul.manager import KV_DECLINE_POISON_REPROBE, _chain_fp, _prefix_hash_chain
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 81902}])
    slot_msgs = _slot().client_meta["messages"]
    mgr._clean_prefill_tokens[_PORT] = _evidence(81902, slot_msgs)
    mgr._poisoned_chain_fps[_PORT] = {_chain_fp(_prefix_hash_chain(slot_msgs))}

    with caplog.at_level(logging.INFO):
        await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)

    sp = _save_posts(posts)
    assert len(sp) == 1
    meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert meta["clean_prefix"] is False                   # stamp declined
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "KV_SAVE_DECLINE" in joined
    assert KV_DECLINE_POISON_REPROBE in joined


@pytest.mark.asyncio
async def test_case6c_kill_switch_short_circuits_evidence_gate(mgr, kv_dir, make_httpx, monkeypatch):
    """Single-lever: TURBOHAUL_CLEAN_STAMP_EVIDENCE_GATE=0
    must short-circuit the ENTIRE evidence gate — the extent AND context-binding checks both failing
    (count far from the slot count, chain fp of another context) still
    stamps clean on the available (evidence) cap, exactly as the pre-fix
    behaviour would have. The lever is ONE switch, not per-check. (RED on
    base: the base's kill-switch arm only covers the no-evidence fallback.)"""
    monkeypatch.setenv("TURBOHAUL_CLEAN_STAMP_EVIDENCE_GATE", "0")
    from turbohaul.manager import CleanPrefillEvidence, _chain_fp, _prefix_hash_chain
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 87401}])
    other_msgs = [{"role": "user", "content": "a different context entirely"}]
    mgr._clean_prefill_tokens[_PORT] = CleanPrefillEvidence(
        count=81902, chain_fp=_chain_fp(_prefix_hash_chain(other_msgs)), arm="strip")

    await mgr._save_slot_kv(_PORT, _QWEN, _slot(), force_clean=True)

    sp = _save_posts(posts)
    assert len(sp) == 1
    assert sp[0]["save_token_limit"] == 81902              # evidence cap (the available cap)
    meta = _read_meta(kv_dir, _QWEN, 0, "t")
    assert meta["clean_prefix"] is True
