"""SAVE-side shadow-reprefill (manager delta).

Covers `_shadow_reprefill_and_save` + `_save_shadow_slot_kv` (SAVE path only):

  1. FLAG-OFF DEFAULT — with TURBOHAUL_SHADOW_REPREFILL unset the feature ships
     INERT: no reprefill/save POST, no shadow bin, no counts.
  2. THINKING TURN — with the flag on, a non-streaming think turn writes a bin under
     the DISTINCT `.shadow` name with meta shadow=True / clean_prefix=False, and the
     think-free assistant-N is appended to the harness's own messages.
  3. CLEAN ANCHOR INTACT — the pre-existing clean_prefix anchor bin+meta are byte-
     for-byte untouched; the shadow is a SEPARATE file.
  4. GUARDS — tool-call / no-</think> / empty-after-strip / empty-messages all skip
     (no reprefill POST, no shadow bin) and bump the matching skipped_* count.
  5. STREAMING — result=None + slot.streamed_assistant_text drives the same save.
  6. OFF-PATH — parallel>1 single-series gate skips entirely; >1 populated slot
     skips the save (never guesses which slot is ours).

The RESTORE path + the clean_prefix never-demote path are NOT touched
(the shadow bin is written, not consumed). Fixtures mirror test_lagreducer.py.
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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn, kv_save_fn
from turbohaul.manager import (
    TurbohaulManager,
    _kv_shadow_meta_fn,
    _kv_shadow_save_fn,
)


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
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    """Isolated SLOT_SAVE_DIR (re-imported per call inside the manager methods)."""
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture
def shadow_on(monkeypatch):
    """Enable the SAVE-side shadow-reprefill for a test."""
    monkeypatch.setenv("TURBOHAUL_SHADOW_REPREFILL", "1")


@pytest.fixture(autouse=True)
def _shadow_off_by_default(monkeypatch):
    """Guarantee a clean default even if the ambient env has the flag set."""
    monkeypatch.delenv("TURBOHAUL_SHADOW_REPREFILL", raising=False)


@pytest.fixture(autouse=True)
def _covered_scaffold_strip_off(monkeypatch):
    """The covered-scaffold-strip feature ships default-ON, which swaps the save-probe prefill TRANSPORT to
    /apply-template + /completion. This suite verifies the TRANSPORT-INDEPENDENT
    save/meta/guard logic against the flag-OFF (byte-identical-to-today) messages transport;
    the Fix B default-ON render+strip transport + bin-write is covered in
    tests/test_covered_scaffold_strip.py. Pin OFF so the messages-based fake engine here
    keeps serving the probe."""
    monkeypatch.setenv("TURBOHAUL_COVERED_SCAFFOLD_STRIP", "0")


# --- sidecar fake (GET /slots + records every POST; materialises the temp bin) ---
class _SaveResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ShadowClient:
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
                f.write(b"dummy think-free kv cache data")
        return _SaveResp({"status": "ok"})


@pytest.fixture
def make_httpx(monkeypatch):
    def _make(payload):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(lambda *a, **k: _ShadowClient(payload, posts))
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


_QWEN = "example-model-27b"
_PORT = 59500
_THINK = "<think>chain of reasoning here</think>THE_FINAL_ANSWER"


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _result(content=None, tool_calls=None, reasoning=None):
    msg = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}]}


def _slot(k=8, thread_id="t", streamed=None):
    return SimpleNamespace(
        thread_id=thread_id,
        model_tag=_QWEN,
        client_meta={"messages": _msgs(k)},
        streamed_assistant_text=streamed,
    )


def _handle(port=_PORT, parallel=1):
    return SimpleNamespace(port=port, parallel=parallel)


def _shadow_paths(kv_dir, sid=0, thread_id="t", port=_PORT):
    th = TurbohaulManager._thread_hash(thread_id)
    return (kv_dir / _kv_shadow_save_fn(_QWEN, sid, th, port),
            kv_dir / _kv_shadow_meta_fn(_QWEN, sid, th, port))


def _n_save_posts(posts):
    return sum(1 for (url, _j) in posts if "action=save" in url)


def _n_reprefill_posts(posts):
    return sum(1 for (url, _j) in posts if url.endswith("/v1/chat/completions"))


def _write_clean_anchor(kv_dir, sid, thread_id, chain, prompt_len=40000):
    th = TurbohaulManager._thread_hash(thread_id)
    meta_fn = kv_meta_fn(_QWEN, sid, th, _PORT)
    bin_fn = kv_save_fn(_QWEN, sid, th, _PORT)
    (kv_dir / bin_fn).write_bytes(b"CLEAN_ANCHOR_BYTES")
    (kv_dir / meta_fn).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 999,
        "prompt_len": prompt_len, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": _QWEN, "slot_id": sid, "port": _PORT, "clean_prefix": True,
    }))
    return kv_dir / bin_fn, kv_dir / meta_fn


# ================================================================================
# 1. FLAG OFF (default) -> completely INERT
# ================================================================================
@pytest.mark.asyncio
async def test_flag_off_ships_inert(mgr, kv_dir, make_httpx):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(), _slot(), _result(_THINK))
    assert posts == []                                   # no reprefill/save POST at all
    assert list(kv_dir.iterdir()) == []                  # no shadow bin written
    assert mgr._shadow_reprefill_counts == {}            # nothing recorded


# ================================================================================
# 2. THINKING TURN -> DISTINCT `.shadow` bin with shadow:true
# ================================================================================
@pytest.mark.asyncio
async def test_thinking_turn_writes_distinct_shadow_bin(mgr, kv_dir, make_httpx, shadow_on):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(), _slot(k=8), _result(_THINK))

    sbin, smeta = _shadow_paths(kv_dir, sid=0)
    assert sbin.exists() and smeta.exists()              # DISTINCT .shadow bin written
    assert sbin.name.endswith(".slot0.shadow.bin")       # marker in the filename
    meta = json.loads(smeta.read_text())
    assert meta["shadow"] is True                        # DISTINCT MARKER
    assert meta["clean_prefix"] is False                 # NOT a clean anchor
    # messages = 8 harness turns + the predicted think-free assistant-N
    assert meta["n_context_turns"] == 9
    assert meta["hash_chain"] == _prefix_hash_chain(
        _msgs(8) + [{"role": "assistant", "content": "THE_FINAL_ANSWER"}])
    # the plain (clean/normal) bin name was NOT written by the shadow path
    th = TurbohaulManager._thread_hash("t")
    assert not (kv_dir / kv_save_fn(_QWEN, 0, th, _PORT)).exists()
    # POSTs: one reprefill (n_predict=0) + one save
    assert _n_reprefill_posts(posts) == 1
    assert _n_save_posts(posts) == 1
    _url, body = next(p for p in posts if p[0].endswith("/v1/chat/completions"))
    assert body["n_predict"] == 0 and body["max_tokens"] == 0 and body["cache_prompt"] is True
    assert body["messages"][-1] == {"role": "assistant", "content": "THE_FINAL_ANSWER"}
    assert mgr._shadow_reprefill_counts.get("saved") == 1


# ================================================================================
# 3. CLEAN ANCHOR left byte-for-byte INTACT
# ================================================================================
@pytest.mark.asyncio
async def test_clean_anchor_left_intact(mgr, kv_dir, make_httpx, shadow_on):
    chain = _prefix_hash_chain(_msgs(12))
    cbin, cmeta = _write_clean_anchor(kv_dir, 0, "t", chain, prompt_len=100000)
    bin_before, meta_before = cbin.read_bytes(), cmeta.read_text()

    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(), _slot(k=8), _result(_THINK))

    # anchor untouched (bytes + meta identical), still a clean anchor
    assert cbin.read_bytes() == bin_before
    assert cmeta.read_text() == meta_before
    assert json.loads(cmeta.read_text())["clean_prefix"] is True
    # shadow is a SEPARATE file
    sbin, smeta = _shadow_paths(kv_dir, sid=0)
    assert sbin.exists() and sbin != cbin and smeta.exists() and smeta != cmeta
    assert json.loads(smeta.read_text())["shadow"] is True


# ================================================================================
# 4. GUARDS — skip (no POST, no bin) + bump the matching skipped_* count
# ================================================================================
@pytest.mark.asyncio
async def test_guard_skips_tool_call(mgr, kv_dir, make_httpx, shadow_on):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(
        _handle(), _slot(), _result(content=None, tool_calls=[{"id": "c1"}]))
    assert posts == [] and list(kv_dir.iterdir()) == []
    assert mgr._shadow_reprefill_counts.get("skipped_toolcall") == 1


@pytest.mark.asyncio
async def test_no_think_turn_now_saves(mgr, kv_dir, make_httpx, shadow_on):
    """Freshness: a no-`</think>` turn is ALREADY think-free,
    so it is a VALID shadow (esp. the last main turn before a swap) — it now SAVES rather
    than skipping, and appends the (already think-free) content verbatim."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(), _slot(k=8), _result("plain answer no think"))

    sbin, smeta = _shadow_paths(kv_dir, sid=0)
    assert sbin.exists() and smeta.exists()                       # shadow bin IS written
    assert _n_reprefill_posts(posts) == 1 and _n_save_posts(posts) == 1
    meta = json.loads(smeta.read_text())
    assert meta["shadow"] is True and meta["clean_prefix"] is False
    # no </think> -> _strip_thinking_all is a whitespace no-op -> content appended as-is
    assert meta["hash_chain"] == _prefix_hash_chain(
        _msgs(8) + [{"role": "assistant", "content": "plain answer no think"}])
    _url, body = next(p for p in posts if p[0].endswith("/v1/chat/completions"))
    assert body["messages"][-1] == {"role": "assistant", "content": "plain answer no think"}
    assert mgr._shadow_reprefill_counts.get("saved") == 1          # saved, not skipped
    assert mgr._shadow_reprefill_counts.get("no_think_saved") == 1  # observability bump
    assert mgr._shadow_reprefill_counts.get("skipped_no_think") is None  # old skip gone


@pytest.mark.asyncio
async def test_guard_skips_empty_after_strip(mgr, kv_dir, make_httpx, shadow_on):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(), _slot(), _result("<think>only reasoning</think>"))
    assert posts == [] and list(kv_dir.iterdir()) == []
    assert mgr._shadow_reprefill_counts.get("skipped_empty") == 1


@pytest.mark.asyncio
async def test_guard_skips_no_messages(mgr, kv_dir, make_httpx, shadow_on):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = SimpleNamespace(thread_id="t", model_tag=_QWEN,
                           client_meta={"messages": []}, streamed_assistant_text=None)
    await mgr._shadow_reprefill_and_save(_handle(), slot, _result(_THINK))
    assert posts == [] and list(kv_dir.iterdir()) == []
    assert mgr._shadow_reprefill_counts.get("skipped_no_messages") == 1


# ================================================================================
# 5. STREAMING (result=None -> slot.streamed_assistant_text)
# ================================================================================
@pytest.mark.asyncio
async def test_streaming_thinking_turn_writes_shadow(mgr, kv_dir, make_httpx, shadow_on):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = _slot(k=6, streamed="<think>r</think>STREAMED_ANSWER")
    await mgr._shadow_reprefill_and_save(_handle(), slot, None)

    sbin, smeta = _shadow_paths(kv_dir, sid=0)
    assert sbin.exists() and smeta.exists()
    meta = json.loads(smeta.read_text())
    assert meta["shadow"] is True and meta["clean_prefix"] is False
    assert meta["hash_chain"] == _prefix_hash_chain(
        _msgs(6) + [{"role": "assistant", "content": "STREAMED_ANSWER"}])
    assert mgr._shadow_reprefill_counts.get("saved") == 1


@pytest.mark.asyncio
async def test_streaming_empty_text_skips(mgr, kv_dir, make_httpx, shadow_on):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(), _slot(streamed=None), None)
    assert posts == [] and list(kv_dir.iterdir()) == []
    assert mgr._shadow_reprefill_counts.get("skipped_empty") == 1


# ================================================================================
# 6. OFF-PATH — single-series gate + single-populated-slot invariant
# ================================================================================
@pytest.mark.asyncio
async def test_parallel_gate_skips(mgr, kv_dir, make_httpx, shadow_on):
    """handle.parallel>1 -> not a single series -> no work at all."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(parallel=2), _slot(), _result(_THINK))
    assert posts == [] and list(kv_dir.iterdir()) == []
    assert mgr._shadow_reprefill_counts == {}


@pytest.mark.asyncio
async def test_multislot_reprefills_but_skips_save(mgr, kv_dir, make_httpx, shadow_on):
    """>1 populated slot is ambiguous -> the reprefill still fires but the save is
    skipped (never guess which slot is ours) -> NO shadow bin, saved count absent."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100},
                        {"id": 1, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(), _slot(), _result(_THINK))
    assert _n_reprefill_posts(posts) == 1                # reprefill fired
    assert _n_save_posts(posts) == 0                     # save skipped (ambiguous)
    assert not any(p.name.endswith(".shadow.bin") for p in kv_dir.iterdir())
    assert mgr._shadow_reprefill_counts.get("saved") is None


# ================================================================================
# 7. Swap-seam freshness (_shadow_save_at_swap): re-save the freshest
#    think-free shadow at the model-swap teardown, no-downgrade + identity-matched.
# ================================================================================
def _write_existing_shadow(kv_dir, sid, chain, thread_id="t", port=_PORT):
    th = TurbohaulManager._thread_hash(thread_id)
    (kv_dir / _kv_shadow_save_fn(_QWEN, sid, th, port)).write_bytes(b"EXISTING_SHADOW")
    (kv_dir / _kv_shadow_meta_fn(_QWEN, sid, th, port)).write_text(json.dumps({
        "thread_id": thread_id, "thread_hash": th, "prompt_tokens": 600,
        "prompt_len": 45000, "n_context_turns": len(chain), "hash_chain": chain,
        "model_tag": _QWEN, "slot_id": sid, "port": port,
        "clean_prefix": False, "shadow": True,
    }))


def _set_src(mgr, k=8, thread_id="t", port=_PORT, answer="SWAP_ANSWER"):
    msgs = _msgs(k) + [{"role": "assistant", "content": answer}]
    # _last_shadow_src is a (thread_id, model_tag)-keyed OrderedDict.
    mgr._last_shadow_src[(thread_id, _QWEN)] = {"thread_id": thread_id, "model_tag": _QWEN,
                                                "port": port, "messages": msgs}
    return msgs


@pytest.mark.asyncio
async def test_swap_saves_when_no_existing_shadow(mgr, kv_dir, make_httpx, shadow_on):
    """No prior shadow (e.g. the last per-turn save POST silently failed) -> the swap
    seam reprefills + saves the freshest think-free shadow. Recovery belt."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    msgs = _set_src(mgr, k=8)
    await mgr._shadow_save_at_swap(_handle(), _QWEN, "t")

    sbin, smeta = _shadow_paths(kv_dir, sid=0, thread_id="t")
    assert sbin.exists() and smeta.exists()
    assert _n_reprefill_posts(posts) == 1 and _n_save_posts(posts) == 1
    assert json.loads(smeta.read_text())["hash_chain"] == _prefix_hash_chain(msgs)
    assert mgr._shadow_reprefill_counts.get("swap_saved") == 1


@pytest.mark.asyncio
async def test_swap_no_downgrade_when_fresher_exists(mgr, kv_dir, make_httpx, shadow_on):
    """An existing shadow with an EQUAL-OR-LONGER chain is NOT overwritten (the per-turn
    hook already saved the freshest) -> no reprefill, no save; constraint #2 + #5."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    msgs = _set_src(mgr, k=8)                       # new chain len = 9
    _write_existing_shadow(kv_dir, 0, _prefix_hash_chain(msgs))  # existing len 9 (equal)
    await mgr._shadow_save_at_swap(_handle(), _QWEN, "t")

    assert posts == []                              # short-circuited BEFORE any POST
    assert mgr._shadow_reprefill_counts.get("swap_skip_have_fresher") == 1
    assert mgr._shadow_reprefill_counts.get("swap_saved") is None


@pytest.mark.asyncio
async def test_swap_identity_mismatch_skips(mgr, kv_dir, make_httpx, shadow_on):
    """The recorded source is for a DIFFERENT thread/model -> never guess -> skip."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    _set_src(mgr, thread_id="other-thread")
    await mgr._shadow_save_at_swap(_handle(), _QWEN, "t")      # teardown thread 't'
    assert posts == [] and mgr._shadow_reprefill_counts == {}


@pytest.mark.asyncio
async def test_swap_inert_when_flag_off(mgr, kv_dir, make_httpx):
    """No shadow_on fixture -> SHADOW_REPREFILL off -> completely INERT."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    _set_src(mgr)
    await mgr._shadow_save_at_swap(_handle(), _QWEN, "t")
    assert posts == [] and mgr._shadow_reprefill_counts == {}


@pytest.mark.asyncio
async def test_swap_no_source_skips(mgr, kv_dir, make_httpx, shadow_on):
    """No recorded source (feature just enabled, no turn yet) -> skip cleanly."""
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    mgr._last_shadow_src.clear()  # keyed OrderedDict, empty == no source
    await mgr._shadow_save_at_swap(_handle(), _QWEN, "t")
    assert posts == [] and mgr._shadow_reprefill_counts == {}


@pytest.mark.asyncio
async def test_per_turn_save_records_last_shadow_src(mgr, kv_dir, make_httpx, shadow_on):
    """A normal per-turn shadow save records _last_shadow_src (feeds the swap belt)."""
    make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    await mgr._shadow_reprefill_and_save(_handle(), _slot(k=6), _result(_THINK))
    src = mgr._last_shadow_src.get(("t", _QWEN))  # keyed store
    assert src is not None and src["thread_id"] == "t" and src["model_tag"] == _QWEN
    assert src["messages"][-1] == {"role": "assistant", "content": "THE_FINAL_ANSWER"}
# TOOL-KNOB PARITY — the n_predict=0 prefill probe must carry the
# live request's `tools` (some chat templates render them at the FRONT of the prompt); a tools-less
# probe would save a bin diverging from a tools-bearing request -> engine CLEAR. The
# clean probe (_probe_and_save_clean_kv) uses the SAME _KV_PROBE_TOOL_KNOBS constant +
# forwarding pattern (byte-reviewed symmetric); these drive it via the shadow probe.
# ================================================================================
@pytest.mark.asyncio
async def test_probe_forwards_tools_when_request_has_them(mgr, kv_dir, make_httpx, shadow_on):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    tools = [{"type": "function",
              "function": {"name": "search", "parameters": {"type": "object"}}}]
    slot = _slot(k=8)
    slot.client_meta = {"messages": _msgs(8), "tools": tools, "tool_choice": "auto"}
    await mgr._shadow_reprefill_and_save(_handle(), slot, _result(_THINK))
    _url, body = next(p for p in posts if p[0].endswith("/v1/chat/completions"))
    # the probe renders the SAME tool preamble the live request does (verbatim)
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_probe_stays_toolless_when_request_has_no_tools(mgr, kv_dir, make_httpx, shadow_on):
    posts = make_httpx([{"id": 0, "n_prompt_tokens": 100}])
    # _slot's client_meta = messages only, NO tools -> no-regression on tools-less turns
    await mgr._shadow_reprefill_and_save(_handle(), _slot(k=8), _result(_THINK))
    _url, body = next(p for p in posts if p[0].endswith("/v1/chat/completions"))
    for k in ("tools", "tool_choice", "parallel_tool_calls", "function_call", "functions"):
        assert k not in body                             # payload byte-identical to pre-fix
