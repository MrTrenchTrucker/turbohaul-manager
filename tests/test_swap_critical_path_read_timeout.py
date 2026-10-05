"""Bound the 900s read timeout on the model-swap critical path.

Root cause (a multi-minute stall on ONE probe call, delaying the
swap): the model-swap teardown AWAITS _probe_and_save_clean_kv (via
_flush_clean_kv_at_unload) before _sigterm. Its two clean-probe transports --
_render_strip_prefill_probe (covered-scaffold-strip default ON)
and the plain /v1/chat/completions probe (flag OFF) -- both used
to build their httpx.AsyncClient with an unbounded 900.0s read timeout (the ON path by
omitting the read_timeout_s kwarg entirely -- positional args only -- so it fell through
to the function's own 900.0 default; the OFF path via an explicit read=900.0 literal).
manager.py (a DIFFERENT call site, the swap-seam shadow-save recovery reprefill)
already carried read_timeout_s=180.0 -- proving the bound belongs on the critical path,
just applied to the wrong two lines.

Non-vacuity: both tests below FAIL on unfixed code (captured read == 900.0, not
180.0) and PASS after the fix. The assertion reads the actual kwarg that reaches
httpx.Timeout's `read=` construction -- not a source-text search for "180.0" -- because
an earlier defect passed exactly that weaker, name-only shape
of check while the real call site still shipped the 900s default.

Run: PYTHONPATH=<worktree>/src pytest tests/test_swap_critical_path_read_timeout.py -v
"""
import os
import types

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
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot

_QWEN = "qwen3.6-27b"
_PORT = 59500


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


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


_PROBE_URL_MARKERS = ("/apply-template", "/completion", "/v1/chat/completions")


class _ClientRecord:
    """One record per httpx.AsyncClient(...) construction: the `timeout=` kwarg it
    was built with, plus every URL it was subsequently used against -- so the test can
    isolate the CLEAN-PROBE client (the one that hits /apply-template, /completion, or
    /v1/chat/completions) from unrelated sidecar clients _probe_and_save_clean_kv's own
    save_to_disk=True path also opens (e.g. _save_slot_kv_inner's GET /slots + action=save
    client, which legitimately uses its own unrelated 120s/1s timeout and is not part of
    this change's scope)."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.urls = []

    def touched_probe_url(self):
        return any(any(m in u for m in _PROBE_URL_MARKERS) for u in self.urls)


class _CapturingClient:
    """Minimal fake sidecar that also records the `timeout=` kwarg every
    httpx.AsyncClient(...) call receives, tagged with the URLs that client instance
    goes on to touch (`records`), so the test can assert on the ACTUAL value reaching
    the client construction for the CORRECT call site rather than trusting a stub that
    silently swallows the kwarg (as the other suites' `*a, **k: ...` fakes do)."""

    def __init__(self, slots_payload, rendered, records, *a, **kw):
        self._slots = slots_payload
        self._rendered = rendered
        self._record = _ClientRecord(kw.get("timeout"))
        records.append(self._record)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        self._record.urls.append(url)
        if "/slots" in url and "action=save" not in url:
            return _Resp(self._slots)
        return _Resp({})

    async def post(self, url, json=None, **kw):
        self._record.urls.append(url)
        if url.endswith("/apply-template"):
            return _Resp({"prompt": self._rendered})
        if "action=save" in url and json and "filename" in json:
            tmp_path = os.path.join(subprocess_mgr.SLOT_SAVE_DIR, json["filename"])
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(b"dummy kv cache data")
            return _Resp({"status": "ok"})
        return _Resp({"tokens_evaluated": 42})  # /completion + /v1/chat/completions


class _FakeTimeout:
    """A REAL capture, not a no-op: stores the exact kwargs httpx.Timeout(...) is built
    with, so the test reads `.kw["read"]` instead of a stub that discards it (the
    name-only-check failure mode to be avoided)."""

    def __init__(self, *a, **kw):
        self.kw = kw


@pytest.fixture
def capturing_httpx(monkeypatch):
    def _make(slots_payload, rendered="plain prompt, no scaffold"):
        records = []

        class _FakeHttpx:
            AsyncClient = staticmethod(
                lambda *a, **k: _CapturingClient(slots_payload, rendered, records, *a, **k)
            )
            Timeout = _FakeTimeout

        monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)
        return records

    return _make


def _handle(port=_PORT):
    return types.SimpleNamespace(parallel=1, port=port)


@pytest.mark.asyncio
async def test_covered_scaffold_strip_on_probe_is_bounded_not_900s(mgr, kv_dir, capturing_httpx):
    """Critical-path site 1 (manager.py): the default-ON transport
    (_render_strip_prefill_probe via /apply-template + /completion) must reach
    httpx.Timeout with read=180.0, not fall through to the function's own 900.0
    default by omitting the kwarg."""
    mgr.runtime.kv.covered_scaffold_strip = True
    records = capturing_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": [{"role": "user", "content": "u"}]},
                    admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(), slot, save_to_disk=True)

    probe_records = [r for r in records if r.touched_probe_url()]
    assert probe_records, f"no client touched a probe URL at all: {[r.urls for r in records]}"
    for r in probe_records:
        assert r.timeout is not None, f"probe client built with no timeout kwarg: urls={r.urls}"
        assert r.timeout.kw.get("read") == 180.0, (
            f"expected read=180.0 on the swap critical path, got {r.timeout.kw} "
            f"for urls={r.urls} (900.0 here means the unbounded default leaked back in)"
        )


@pytest.mark.asyncio
async def test_covered_scaffold_strip_off_probe_is_bounded_not_900s(mgr, kv_dir, capturing_httpx):
    """Critical-path site 2 (manager.py): the flag-OFF plain
    /v1/chat/completions probe must ALSO carry read=180.0 -- this is the else-branch
    the fix covers alongside site 1."""
    mgr.runtime.kv.covered_scaffold_strip = False
    records = capturing_httpx([{"id": 0, "n_prompt_tokens": 100}])
    slot = Slot.new(_QWEN, thread_id="t", context=None,
                    client_meta={"messages": [{"role": "user", "content": "u"}]},
                    admission_ctx_len=50000)

    await mgr._probe_and_save_clean_kv(_handle(), slot, save_to_disk=True)

    probe_records = [r for r in records if r.touched_probe_url()]
    assert probe_records, f"no client touched a probe URL at all: {[r.urls for r in records]}"
    for r in probe_records:
        assert r.timeout is not None, f"probe client built with no timeout kwarg: urls={r.urls}"
        assert r.timeout.kw.get("read") == 180.0, (
            f"expected read=180.0 on the swap critical path, got {r.timeout.kw} "
            f"for urls={r.urls} (900.0 here means the unbounded default leaked back in)"
        )


# ================================================================================
# Third call site of _render_strip_prefill_probe:
# (_shadow_reprefill_and_save) must not silently inherit the function's own
# 900.0 default by accident. Prefill durations are long, so flipping that
# default to 180.0 would have been
# WRONG: this site is a full reprefill fired AFTER set_result (zero added TTFT),
# not a swap-blocking probe like the other two, and long prefills can run
# for many minutes -- 180s would cut off a sizeable share of
# calls. Fix: read_timeout_s is a REQUIRED keyword-only argument (no default),
# so every call site must say what it means explicitly rather than one shared
# number silently covering structurally different workloads. This site passes
# 900.0 explicitly (unchanged VALUE, no longer an accident); the other two keep
# their own 180.0.
# ================================================================================
import inspect  # noqa: E402


def test_render_strip_prefill_probe_requires_read_timeout_s_explicitly():
    """The strongest form of the invariant: Python itself enforces it at every
    call site (a missing kwarg is a TypeError at the call, not a silent 900.0),
    so a future 4th call site fails loudly rather than needing a test to catch it.
    This test is the belt for that enforcement, not the only guard -- see the AST
    test below for the buckle."""
    sig = inspect.signature(TurbohaulManager._render_strip_prefill_probe)
    param = sig.parameters["read_timeout_s"]
    assert param.default is inspect.Parameter.empty, (
        f"read_timeout_s has regained a default ({param.default!r}) -- the original defect "
        f"exists because a shared default silently covered a call site that needed "
        f"a different value; keep it required so that can't happen again"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_every_call_site_passes_read_timeout_s_explicitly():
    """Belt-and-braces alongside the signature test above (which alone would
    already make a missing kwarg a TypeError): re-derive every call site of
    _render_strip_prefill_probe from the actual source via AST -- not a hand-
    maintained line-number list, which is exactly what went stale for the original defect
    -- and confirm each one supplies read_timeout_s. Also pins the count so a new
    (4th) call site is a visible test change, not a silent addition."""
    import ast
    from pathlib import Path

    src_path = Path(inspect.getfile(TurbohaulManager))
    tree = ast.parse(src_path.read_text(), filename=str(src_path))
    call_sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "_render_strip_prefill_probe":
                has_kw = any(kw.arg == "read_timeout_s" for kw in node.keywords)
                call_sites.append((node.lineno, has_kw))

    assert len(call_sites) == 3, (
        f"expected exactly 3 call sites of _render_strip_prefill_probe, found "
        f"{len(call_sites)} at lines {[l for l, _ in call_sites]} -- this test "
        f"assumes exactly 3; a new site changes the count here on purpose"
    )
    missing = [lineno for lineno, has_kw in call_sites if not has_kw]
    assert not missing, (
        f"call site(s) at line(s) {missing} do not pass read_timeout_s explicitly "
        f"-- this can no longer happen silently (Python itself would raise "
        f"TypeError at that call), but if it does, it means the signature's own "
        f"required-kwarg guarantee was somehow bypassed, which is worth knowing"
    )


def _shadow_handle(port=_PORT, parallel=1):
    return types.SimpleNamespace(port=port, parallel=parallel)


def _shadow_slot(thread_id="t-shadow"):
    return types.SimpleNamespace(
        thread_id=thread_id,
        model_tag=_QWEN,
        client_meta={"messages": [{"role": "user", "content": "u"}]},
        streamed_assistant_text=None,
    )


def _shadow_result():
    return {"choices": [{"message": {
        "content": "<think>chain of reasoning here</think>THE_FINAL_ANSWER",
    }}]}


@pytest.mark.asyncio
async def test_shadow_reprefill_full_reprefill_uses_900s_not_180s(
    mgr, kv_dir, capturing_httpx, monkeypatch,
):
    """The third call site (_shadow_reprefill_and_save's render-strip transport,
    manager.py): must reach httpx.Timeout with read=900.0 -- deliberately
    NOT 180.0, unlike the two swap-critical-path sites above.

    NOTE on what this test does and does not prove: unlike the two tests above
    (which discriminate a real pre/post-fix VALUE change, 900->180), this one
    does NOT -- the value at this site is unchanged by the fix (900.0
    before, 900.0 after; only whether it's explicit changed). It would pass
    against unpatched main too, since main's bug was an accidentally-inherited
    900.0 that happens to equal the deliberately-chosen 900.0 here. Its job is
    to guard against a FUTURE regression (someone changing this site's value by
    mistake, e.g. copy-pasting the other two sites' 180.0) -- the tests that
    actually discriminate this fix are
    test_render_strip_prefill_probe_requires_read_timeout_s_explicitly and
    test_every_call_site_passes_read_timeout_s_explicitly, both of which fail
    on unpatched main (default is 900.0 not absent; this call site omits the
    kwarg) and pass here."""
    monkeypatch.setenv("TURBOHAUL_SHADOW_REPREFILL", "1")
    mgr.runtime.kv.covered_scaffold_strip = True
    records = capturing_httpx([{"id": 0, "n_prompt_tokens": 100}])

    await mgr._shadow_reprefill_and_save(_shadow_handle(), _shadow_slot(), _shadow_result())

    probe_records = [r for r in records if r.touched_probe_url()]
    assert probe_records, f"no client touched a probe URL at all: {[r.urls for r in records]}"
    for r in probe_records:
        assert r.timeout is not None, f"probe client built with no timeout kwarg: urls={r.urls}"
        assert r.timeout.kw.get("read") == 900.0, (
            f"expected read=900.0 on the full-reprefill site (the explicit-timeout fix -- this "
            f"site is deliberately NOT bounded to 180.0, see this file's module "
            f"docstring section for the measured-distribution reasoning), got "
            f"{r.timeout.kw} for urls={r.urls}"
        )
