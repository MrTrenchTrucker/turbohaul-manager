"""Covered-region REASONING-NORMALIZE (Fix B) + skip-wider FLOOR (Fix A).

ROOT (byte-proven via the engine /apply-template): a COVERED assistant tool-call turn is
HASH-INVISIBLE to `_prefix_hash_chain` (kv_policy, UNTOUCHED — it hashes role+content only),
yet the chat template renders a `<think>...</think>\\n\\n` scaffold for it POSITION-based
(any turn after last_query_index). At SAVE time those recent turns sit after
last_query_index -> the saved KV carries the scaffold; on the FUTURE resend the same turns
sit BEFORE last_query_index -> no scaffold. That positional drift makes the saved bin
token-stale -> the engine CLEARs instead of reusing.

Fix B (SAVE-ONLY, default ON, TURBOHAUL_COVERED_SCAFFOLD_STRIP): at each SAVE probe render
the messages via /apply-template, strip `<think>.*?</think>\\n\\n` (DOTALL) from the rendered
PROMPT, then prefill that raw prompt via /completion (n_predict=0, cache_prompt) and save.
The saved KV now byte-matches the historical resend -> MATCH+REUSE. The live-generation
sites (_build_stream_payload / _complete) are UNTOUCHED (they already render historical).

Fix A (emergency FLOOR, default OFF, TURBOHAUL_TOOLTAIL_SCAN_COVERED): widen the crit3
tool-opaque restore-skip scan from start=common to start=0 (the whole bin-covered span).

Covers: the pure strip helper (gate byte-target + empty scaffold + multi-block +
edges), both flag readers (default ON / default OFF), Fix A `scan_covered`, hash-chain
invariance (proves the meta is unaffected), and the render->strip->/completion transport at
`_render_strip_prefill_probe` + both save probes writing their bin under the default-ON path.

Run:  PYTHONPATH=<worktree>/src pytest tests/test_covered_scaffold_strip.py
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
from turbohaul.kv_policy import _prefix_hash_chain, kv_meta_fn
from turbohaul.manager import (
    TurbohaulManager,
    _covered_scaffold_strip_enabled,
    _divergent_tail_is_tool_opaque,
    _kv_shadow_meta_fn,
    _kv_shadow_save_fn,
    _strip_think_scaffold,
    _tooltail_scan_covered_enabled,
)
from turbohaul.slot import Slot

_MODEL_TAG = "example-model-27b"
_PORT = 59500


# ============================================================================
# 1. PURE — _strip_think_scaffold (GATE byte-target + edges)
# ============================================================================
def test_strip_gate_byte_target():
    """GATE: assistant\\n<think>\\n{r}\\n</think>\\n\\n<tool_call> -> assistant\\n<tool_call>."""
    rendered = (
        "<|im_start|>assistant\n"
        "<think>\nlet me check the weather tool\n</think>\n\n"
        "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call><|im_end|>\n"
    )
    expected = (
        "<|im_start|>assistant\n"
        "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call><|im_end|>\n"
    )
    assert _strip_think_scaffold(rendered) == expected


def test_strip_empty_scaffold():
    """The EMPTY `<think>\\n\\n</think>\\n\\n` scaffold (what a field-strip leaves behind)
    is removed too — this is precisely why field-strip alone was insufficient."""
    rendered = "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>x</tool_call>"
    assert _strip_think_scaffold(rendered) == "<|im_start|>assistant\n<tool_call>x</tool_call>"


def test_strip_multi_block_global():
    """Applied globally: two ANCHORED assistant turns each with a scaffold -> BOTH removed;
    the non-greedy `.*?` pairs each open with its own close (never spans across a block)."""
    rendered = (
        "<|im_start|>assistant\n<think>\nr1\n</think>\n\nX<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nr2\n</think>\n\nY<|im_end|>\n"
    )
    expected = (
        "<|im_start|>assistant\nX<|im_end|>\n"
        "<|im_start|>assistant\nY<|im_end|>\n"
    )
    assert _strip_think_scaffold(rendered) == expected


def test_strip_no_think_is_noop():
    plain = "<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\n"
    assert _strip_think_scaffold(plain) == plain


def test_strip_requires_anchor_and_trailing_blank():
    """Un-anchored / malformed blocks are left intact: a `<think>...</think>` NOT preceded by
    `<|im_start|>assistant\\n` (and here also missing the trailing `\\n\\n`) survives -> the
    strip only ever removes a full ASSISTANT template-emitted block, never partial prose."""
    s = "prefix <think>x</think>NO_BLANK_AFTER suffix"
    assert _strip_think_scaffold(s) == s


def test_strip_text_answer_keeps_header():
    """A text answer (no tool call): the `\\1` backref KEEPS the assistant header, drops only
    the think -> `...assistant\\n<think>\\n{r}\\n</think>\\n\\nThe answer.` -> `...assistant\\nThe answer.`."""
    rendered = "<|im_start|>assistant\n<think>\nlong cot\n</think>\n\nThe answer.<|im_end|>"
    assert _strip_think_scaffold(rendered) == "<|im_start|>assistant\nThe answer.<|im_end|>"


def test_strip_idempotent():
    rendered = "<|im_start|>assistant\n<think>\nr\n</think>\n\nb<|im_end|>"
    once = _strip_think_scaffold(rendered)
    assert once == "<|im_start|>assistant\nb<|im_end|>"
    assert _strip_think_scaffold(once) == once


def test_strip_none_and_nonstr_passthrough():
    assert _strip_think_scaffold(None) is None
    assert _strip_think_scaffold(123) == 123


def test_user_and_tool_embedded_think_survive():
    """IMMUNITY: a role=user msg AND a role=tool response each carrying a LITERAL
    `<think>...</think>\\n\\n` (e.g. a user pasting this template + engine logs into the chat)
    SURVIVE strip untouched — ONLY the assistant-anchored scaffold is removed. The
    user/tool blocks here use the FULL structural `<think>\\n...\\n</think>\\n\\n` form, so their
    survival is due SOLELY to the missing `<|im_start|>assistant\\n` anchor (proves the anchor
    is load-bearing, not just the newline structure)."""
    rendered = (
        "<|im_start|>user\npasted log: <think>\ndebug trace\n</think>\n\nplease help<|im_end|>\n"
        "<|im_start|>tool\nresult: <think>\ninner\n</think>\n\ndone<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nmy reasoning\n</think>\n\nThe answer.<|im_end|>\n"
    )
    out = _strip_think_scaffold(rendered)
    # user + tool literal think blocks SURVIVE byte-for-byte
    assert "user\npasted log: <think>\ndebug trace\n</think>\n\nplease help<|im_end|>" in out
    assert "tool\nresult: <think>\ninner\n</think>\n\ndone<|im_end|>" in out
    # ONLY the assistant scaffold is stripped (header kept via \1, reasoning gone)
    assert "<|im_start|>assistant\nThe answer.<|im_end|>\n" in out
    assert "my reasoning" not in out


# ============================================================================
# 2. FLAG READERS — Fix B default ON, Fix A default OFF
# ============================================================================
@pytest.fixture(autouse=True)
def _clean_flag_env(monkeypatch):
    """Each test starts from the shipped defaults regardless of ambient env."""
    monkeypatch.delenv("TURBOHAUL_COVERED_SCAFFOLD_STRIP", raising=False)
    monkeypatch.delenv("TURBOHAUL_TOOLTAIL_SCAN_COVERED", raising=False)


def test_fixb_flag_default_on():
    assert _covered_scaffold_strip_enabled() is True


@pytest.mark.parametrize("v", ["0", "false", "no", "off", "OFF", "False"])
def test_fixb_flag_explicit_off(monkeypatch, v):
    monkeypatch.setenv("TURBOHAUL_COVERED_SCAFFOLD_STRIP", v)
    assert _covered_scaffold_strip_enabled() is False


@pytest.mark.parametrize("v", ["1", "true", "yes", "on", "anything-else"])
def test_fixb_flag_on_values(monkeypatch, v):
    monkeypatch.setenv("TURBOHAUL_COVERED_SCAFFOLD_STRIP", v)
    assert _covered_scaffold_strip_enabled() is True


def test_fixa_flag_default_off():
    assert _tooltail_scan_covered_enabled() is False


@pytest.mark.parametrize("v", ["1", "true", "yes", "on", "ON"])
def test_fixa_flag_explicit_on(monkeypatch, v):
    monkeypatch.setenv("TURBOHAUL_TOOLTAIL_SCAN_COVERED", v)
    assert _tooltail_scan_covered_enabled() is True


@pytest.mark.parametrize("v", ["0", "false", "no", "off", "garbage"])
def test_fixa_flag_off_values(monkeypatch, v):
    monkeypatch.setenv("TURBOHAUL_TOOLTAIL_SCAN_COVERED", v)
    assert _tooltail_scan_covered_enabled() is False


# ============================================================================
# 3. Fix A — _divergent_tail_is_tool_opaque(scan_covered=...)
# ============================================================================
def _tool_call_turn():
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]}


def test_fixa_off_ignores_settled_tool_before_common():
    """Default (scan_covered=False): a tool turn INSIDE the reused prefix (index < common)
    does NOT trigger the skip — a settled tool region never over-triggers."""
    msgs = [{"role": "user", "content": "u0"}, _tool_call_turn(),
            {"role": "tool", "content": "res"}, {"role": "user", "content": "u3"}]
    assert _divergent_tail_is_tool_opaque(msgs, common=4, scan_covered=False) is False


def test_fixa_on_catches_settled_tool_before_common():
    """Fix A ON (scan_covered=True): the SAME settled tool turn is now caught (scan from 0)
    -> the whole covered span falls back to native reuse. This is the emergency floor."""
    msgs = [{"role": "user", "content": "u0"}, _tool_call_turn(),
            {"role": "tool", "content": "res"}, {"role": "user", "content": "u3"}]
    assert _divergent_tail_is_tool_opaque(msgs, common=4, scan_covered=True) is True


def test_fixa_tail_tool_caught_either_way():
    """A tool turn in the divergent tail (index >= common) is caught with OR without Fix A."""
    msgs = [{"role": "user", "content": "u0"}, {"role": "user", "content": "u1"},
            _tool_call_turn()]
    assert _divergent_tail_is_tool_opaque(msgs, common=2, scan_covered=False) is True
    assert _divergent_tail_is_tool_opaque(msgs, common=2, scan_covered=True) is True


def test_fixa_all_text_never_skips():
    msgs = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a1"}]
    assert _divergent_tail_is_tool_opaque(msgs, common=0, scan_covered=True) is False


def test_fixa_empty_messages_false():
    assert _divergent_tail_is_tool_opaque([], common=0, scan_covered=True) is False
    assert _divergent_tail_is_tool_opaque(None, common=3, scan_covered=True) is False


# ============================================================================
# 4. HASH-CHAIN INVARIANCE — proves kv_policy + the saved meta are UNTOUCHED
# ============================================================================
def test_reasoning_content_invisible_to_chain():
    """The saved meta's hash_chain is derived from `messages` (role+content). The Fix B
    strip acts on the RENDERED prompt, and `reasoning_content` is invisible to the chain, so
    the meta hash_chain is byte-identical whether or not the reasoning field/scaffold exists
    -> zero meta drift, kv_policy UNTOUCHED, and the per-turn re-save cadence (n_context_turns
    / no-downgrade length) is unaffected (case (b): still-current tool chain not broken)."""
    with_reasoning = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "answer", "reasoning_content": "long CoT here"},
    ]
    without = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "answer"},
    ]
    assert _prefix_hash_chain(with_reasoning) == _prefix_hash_chain(without)


# ============================================================================
# 5. TRANSPORT — render -> strip -> /completion (fake engine)
# ============================================================================
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
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeEngine:
    """Serves GET /slots, POST /apply-template -> {"prompt": rendered}, POST /completion,
    POST /slots/{id}?action=save (materialises the temp bin). Records every POST as
    (url, json). `rendered=None` -> /apply-template returns {} (no prompt); `fail_apply`
    -> /apply-template raises."""

    def __init__(self, slots_payload, rendered, fail_apply, posts):
        self._slots = slots_payload
        self._rendered = rendered
        self._fail_apply = fail_apply
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if "/slots" in url and "action=save" not in url:
            return _Resp(self._slots)
        return _Resp({})

    async def post(self, url, json=None, **kw):
        self._posts.append((url, json))
        if url.endswith("/apply-template"):
            if self._fail_apply:
                raise RuntimeError("apply-template boom")
            return _Resp({"prompt": self._rendered} if self._rendered is not None else {})
        if "action=save" in url and json and "filename" in json:
            tmp_path = os.path.join(subprocess_mgr.SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"stripped-historical-kv")
            return _Resp({"status": "ok"})
        return _Resp({"tokens_evaluated": 42})  # /completion (and any other)


@pytest.fixture
def make_engine(monkeypatch):
    def _make(slots_payload, rendered="RENDERED", fail_apply=False):
        posts = []

        class _FakeHttpx:
            AsyncClient = staticmethod(
                lambda *a, **k: _FakeEngine(slots_payload, rendered, fail_apply, posts)
            )
            Timeout = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return posts

    return _make


def _apply_posts(posts):
    return [p for p in posts if p[0].endswith("/apply-template")]


def _completion_posts(posts):
    return [p for p in posts if p[0].endswith("/completion")]


def _chat_posts(posts):
    return [p for p in posts if p[0].endswith("/v1/chat/completions")]


def _save_posts(posts):
    return [p for p in posts if "action=save" in p[0]]


@pytest.mark.asyncio
async def test_render_strip_prefill_posts_stripped_prompt(mgr, make_engine):
    """The helper: /apply-template -> strip -> /completion with the STRIPPED prompt,
    n_predict=0, cache_prompt=true. Tool knobs are forwarded into the /apply-template body."""
    rendered = "SYS\n<|im_start|>assistant\n<think>\nreason\n</think>\n\n<tool_call>call</tool_call>\nEND"
    stripped = "SYS\n<|im_start|>assistant\n<tool_call>call</tool_call>\nEND"
    posts = make_engine([{"id": 0, "n_prompt_tokens": 100}], rendered=rendered)

    ok = await mgr._render_strip_prefill_probe(
        _PORT, _MODEL_TAG, [{"role": "user", "content": "u"}],
        {"tools": [{"type": "function"}], "tool_choice": "auto"},
    )

    assert ok is True
    # /apply-template carried the messages + tool preamble (crit1 parity)
    ap = _apply_posts(posts)
    assert len(ap) == 1
    assert ap[0][1]["messages"] == [{"role": "user", "content": "u"}]
    assert ap[0][1]["tools"] == [{"type": "function"}]
    assert ap[0][1]["tool_choice"] == "auto"
    # /completion got the STRIPPED prompt, prefill-only
    cp = _completion_posts(posts)
    assert len(cp) == 1
    assert cp[0][1]["prompt"] == stripped
    assert cp[0][1]["n_predict"] == 0
    assert cp[0][1]["cache_prompt"] is True
    # SAVE-ONLY transport: never touches the live chat-completions endpoint
    assert _chat_posts(posts) == []


@pytest.mark.asyncio
async def test_render_strip_prefill_no_tools_omits_knobs(mgr, make_engine):
    posts = make_engine([{"id": 0, "n_prompt_tokens": 100}], rendered="plain\nprompt")
    ok = await mgr._render_strip_prefill_probe(_PORT, _MODEL_TAG, [{"role": "user", "content": "u"}], {})
    assert ok is True
    body = _apply_posts(posts)[0][1]
    for k in ("tools", "tool_choice", "parallel_tool_calls", "function_call", "functions"):
        assert k not in body


@pytest.mark.asyncio
async def test_render_strip_prefill_false_on_apply_fail(mgr, make_engine):
    posts = make_engine([{"id": 0, "n_prompt_tokens": 100}], fail_apply=True)
    ok = await mgr._render_strip_prefill_probe(_PORT, _MODEL_TAG, [{"role": "user", "content": "u"}], {})
    assert ok is False
    assert _completion_posts(posts) == []  # never reached /completion


@pytest.mark.asyncio
async def test_render_strip_prefill_false_on_missing_prompt(mgr, make_engine):
    posts = make_engine([{"id": 0, "n_prompt_tokens": 100}], rendered=None)  # apply returns {}
    ok = await mgr._render_strip_prefill_probe(_PORT, _MODEL_TAG, [{"role": "user", "content": "u"}], {})
    assert ok is False
    assert _completion_posts(posts) == []


# ============================================================================
# 6. DEFAULT-ON WIRING — both save probes prefill via render+strip and still save
# ============================================================================
def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


@pytest.mark.asyncio
async def test_clean_probe_default_on_saves_via_render_strip(mgr, kv_dir, make_engine):
    """Fix B default ON: _probe_and_save_clean_kv prefills the stripped historical prompt
    (apply-template + /completion) — NOT /v1/chat/completions — and still writes the clean
    bin. Proves case (a) end-to-end at the clean save site."""
    rendered = "<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\nr\n</think>\n\nTAIL"
    stripped = "<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\nTAIL"
    posts = make_engine([{"id": 0, "n_prompt_tokens": 100}], rendered=rendered)
    slot = Slot.new(_MODEL_TAG, thread_id="t", context=None,
                    client_meta={"messages": _msgs(6)}, admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(), slot)

    # transport swapped: rendered + prefilled the stripped prompt, no live chat POST
    assert len(_apply_posts(posts)) == 1
    assert _completion_posts(posts)[0][1]["prompt"] == stripped
    assert _chat_posts(posts) == []
    # clean bin still saved with a clean_prefix meta
    assert len(_save_posts(posts)) == 1
    meta = _read_clean_meta(kv_dir, "t")
    assert meta["clean_prefix"] is True
    assert meta["n_context_turns"] == 6


@pytest.mark.asyncio
async def test_shadow_probe_default_on_saves_via_render_strip(mgr, kv_dir, make_engine, monkeypatch):
    """Fix B default ON at the shadow save site: think-free reprefill goes through
    render+strip and still writes the DISTINCT `.shadow` bin."""
    monkeypatch.setenv("TURBOHAUL_SHADOW_REPREFILL", "1")  # arm the SAVE-side shadow hook
    rendered = "<|im_start|>assistant\n<think>\n\n</think>\n\nDONE"  # empty scaffold on think-free turn N
    stripped = "<|im_start|>assistant\nDONE"
    posts = make_engine([{"id": 0, "n_prompt_tokens": 100}], rendered=rendered)
    slot = SimpleNamespace(thread_id="t", model_tag=_MODEL_TAG,
                           client_meta={"messages": _msgs(8)}, streamed_assistant_text=None)
    result = {"choices": [{"message": {"content": "<think>cot</think>THE_ANSWER"}}]}

    await mgr._shadow_reprefill_and_save(_handle(), slot, result)

    assert len(_apply_posts(posts)) == 1
    assert _completion_posts(posts)[0][1]["prompt"] == stripped
    assert _chat_posts(posts) == []
    # a DISTINCT .shadow bin was written
    th = TurbohaulManager._thread_hash("t")
    shadow_bin = kv_dir / _kv_shadow_save_fn(_MODEL_TAG, 0, th, _PORT)
    shadow_meta = kv_dir / _kv_shadow_meta_fn(_MODEL_TAG, 0, th, _PORT)
    assert shadow_bin.exists() and shadow_meta.exists()
    meta = json.loads(shadow_meta.read_text())
    assert meta["shadow"] is True and meta["clean_prefix"] is False


@pytest.mark.asyncio
async def test_flag_off_uses_legacy_chat_transport(mgr, kv_dir, make_engine):
    """Flag OFF -> byte-identical-to-today: the clean probe uses /v1/chat/completions and
    NEVER calls /apply-template or /completion."""
    posts = make_engine([{"id": 0, "n_prompt_tokens": 100}], rendered="unused")
    import os as _os
    _os.environ["TURBOHAUL_COVERED_SCAFFOLD_STRIP"] = "0"
    try:
        slot = Slot.new(_MODEL_TAG, thread_id="t", context=None,
                        client_meta={"messages": _msgs(6)}, admission_ctx_len=50000)
        await mgr._probe_and_save_clean_kv(_handle(), slot)
    finally:
        _os.environ.pop("TURBOHAUL_COVERED_SCAFFOLD_STRIP", None)

    assert len(_chat_posts(posts)) == 1
    assert _apply_posts(posts) == []
    assert _completion_posts(posts) == []
    assert len(_save_posts(posts)) == 1  # clean bin still saved


def _handle(port=_PORT):
    return SimpleNamespace(parallel=1, port=port)


def _read_clean_meta(kv_dir, thread_id, sid=0, port=_PORT):
    th = TurbohaulManager._thread_hash(thread_id)
    return json.loads((kv_dir / kv_meta_fn(_MODEL_TAG, sid, th, port)).read_text())
