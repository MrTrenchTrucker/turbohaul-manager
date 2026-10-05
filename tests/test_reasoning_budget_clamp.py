"""Request-time CLAMP for a pathological reasoning_budget.

Design: when the effective ceiling (same min-of-manifest-n_predict-and-caller-
max_tokens computation the existing warning uses) would leave no room to answer,
clamp the EFFECTIVE reasoning_budget for THIS request only -- reserving an answer
floor -- while keeping the existing warning. Never touches the save-time
guard (manifest.py:_validate_reasoning_budget) and never rewrites the manifest
file itself; this is a request-path-only, per-call override via the same
_COMMON_FORWARDED_KNOBS mechanism a caller-supplied override already uses.

Answer floor design:
half the effective ceiling -- the documented "half-of-max_tokens
for half-budget" manifest convention (docs/MODEL_CONFIG_REFERENCE.md),
applied here as a request-time safety net rather than invented from scratch --
bounded to [32, 2048] tokens (never too small to be a real answer, never
disproportionate against a very generous ceiling). When even the 32-token
minimum would consume the whole ceiling, reasoning_budget clamps to 0
(documented "thinking off", NEVER -1 "unlimited" -- see
the reasoning-budget validation notes' explicit trap: is_thinking_payload() only
checks reasoning_budget > 0, so -1 would be silently misread as "not thinking"
when it means the opposite).

Three required controls, both as pure-function units and end-to-end through the
real routes (capturing the ACTUAL slot.client_meta the manager would forward to
the engine, not just this function's return value in isolation):
  NEGATIVE -- a normal request (budget well under ceiling) is COMPLETELY
    untouched: no clamp function call downstream, reasoning_budget absent from
    client_meta exactly as if this change never shipped.
  POSITIVE -- a real qwen3.6-style shape (budget 3072 vs
    caller max_tokens 2048) clamps to 1024 AND leaves a 1024-token answer floor.
  EDGE -- caller sends no max_tokens at all, manifest n_predict unbounded (-1):
    no ceiling exists, so no clamp -- matches the pathological
    shape's exempt precondition, must not fire in the absence of a caller cap.
"""
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.chat_completion import clamp_reasoning_budget_for_ceiling
from turbohaul.api.main import create_app
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
from turbohaul.subprocess_mgr import SidecarHandle


@pytest.fixture(autouse=True)
def _clear_warn_cache():
    """The warn-once-per-(model,caller) cache is module-level global
    state (chat_completion._reasoning_budget_warned) -- shared across every
    test in this process, including the reasoning-budget warning test's own
    tests if run in the same session. Isolate each test in THIS file so a
    prior test's warning for the same (model, caller) doesn't silently
    suppress this file's own warning assertions. Same pattern as
    the reasoning-budget warning test's _clear_warn_cache fixture.
    """
    from turbohaul.api import chat_completion as cc
    cache = getattr(cc, "_reasoning_budget_warned", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


# ============================================================================
# A. clamp_reasoning_budget_for_ceiling -- pure function
# ============================================================================


class TestClampReasoningBudgetForCeiling:
    def test_negative_control_budget_under_ceiling_returns_none(self):
        # A normal request: 500 well under a 4096 caller ceiling. Must be a
        # complete no-op -- this IS the "leave it untouched" contract callers
        # rely on to know whether to mutate the payload at all.
        assert clamp_reasoning_budget_for_ceiling(
            {"reasoning_budget": 500, "n_predict": -1}, 4096,
        ) is None

    def test_positive_control_ornith_qwen_shape_clamps_with_floor(self):
        # A realistic shape:
        # budget 3072, caller max_tokens 2048, n_predict
        # unbounded (-1) on the manifest side. floor = 0.5*2048 = 1024;
        # effective = min(3072, 2048-1024) = 1024.
        assert clamp_reasoning_budget_for_ceiling(
            {"reasoning_budget": 3072, "n_predict": -1}, 2048,
        ) == 1024

    def test_edge_no_caller_max_tokens_unbounded_manifest_returns_none(self):
        # The exempt precondition: n_predict <= 0 AND no caller
        # ceiling at all -- there is nothing to clamp against.
        assert clamp_reasoning_budget_for_ceiling(
            {"reasoning_budget": 3072, "n_predict": -1}, None,
        ) is None

    def test_ceiling_from_manifest_alone_clamps_too(self):
        # No caller override; the manifest's own bounded n_predict is the
        # ceiling. budget 5000 >= ceiling 4096 -> floor = 2048 (0.5*4096, at
        # the cap boundary), effective = min(5000, 4096-2048) = 2048.
        assert clamp_reasoning_budget_for_ceiling(
            {"reasoning_budget": 5000, "n_predict": 4096}, None,
        ) == 2048

    def test_floor_cap_engages_for_generous_ceiling(self):
        # A large model's actual reasoning_budget (24576) against a caller
        # max_tokens of 8000: uncapped half would be 4000, but the 2048 cap
        # engages -- effective = min(24576, 8000-2048) = 5952.
        assert clamp_reasoning_budget_for_ceiling(
            {"reasoning_budget": 24576, "n_predict": -1}, 8000,
        ) == 5952

    def test_no_sane_floor_degrades_to_zero_thinking(self):
        # max_tokens=20: even the 32-token minimum floor exceeds the whole
        # ceiling. Disable thinking for this request (0 = documented "off"),
        # rather than reserve a floor too small to be a real answer.
        assert clamp_reasoning_budget_for_ceiling(
            {"reasoning_budget": 500, "n_predict": -1}, 20,
        ) == 0

    def test_boundary_just_above_no_sane_floor_threshold(self):
        # ceiling=33: floor clamps to the 32-token minimum, 33 > 32 so a
        # (tiny, 1-token) thinking allowance still survives instead of
        # degrading fully to 0 -- monotonic, no cliff.
        assert clamp_reasoning_budget_for_ceiling(
            {"reasoning_budget": 500, "n_predict": -1}, 33,
        ) == 1

    def test_zero_budget_returns_none_regardless_of_ceiling(self):
        # reasoning_budget=0 already means "thinking off" -- nothing to clamp.
        assert clamp_reasoning_budget_for_ceiling(
            {"reasoning_budget": 0, "n_predict": -1}, 20,
        ) is None

    def test_never_returns_negative_one(self):
        # The explicit trap: is_thinking_payload()
        # only checks reasoning_budget > 0, so a clamp result of -1 would be
        # silently misread as "not thinking" when -1 actually means
        # "unlimited". Sweep a range of pathological shapes and assert none
        # ever produces -1 (or any negative value).
        for budget, n_predict, caller_max in [
            (3072, -1, 2048), (500, -1, 1), (1_000_000, -1, 5),
            (5000, 4096, None), (24576, -1, 8000), (100, -1, 32),
        ]:
            result = clamp_reasoning_budget_for_ceiling(
                {"reasoning_budget": budget, "n_predict": n_predict}, caller_max,
            )
            assert result is None or result >= 0, (budget, n_predict, caller_max, result)


# ============================================================================
# B. End-to-end: prove the ACTUAL forwarded value, via slot.client_meta
# ============================================================================


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _write_manifest_yaml(manifests_root, tag: str, reasoning_budget, n_predict=None):
    flags_lines = [f"  reasoning_budget: {reasoning_budget}"]
    if n_predict is not None:
        flags_lines.append(f"  n_predict: {n_predict}")
    (manifests_root / f"{tag}.yaml").write_text(
        "model_tag: " + tag + "\n"
        'gguf_blob_sha256: "' + ("a" * 64) + '"\n'
        "gguf_size_bytes: 1000\n"
        "llama_server_flags:\n" + "\n".join(flags_lines) + "\n"
    )


@pytest.fixture
def clamp_e2e_app(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    # normal-model: budget well under any caller ceiling used below -- the
    # negative control.
    _write_manifest_yaml(storage_root / "manifests", "normal-model", reasoning_budget=500)
    # clamp-model: the real qwen3.6-style pathological shape -- budget 3072,
    # n_predict unbounded (-1, omitted -> manifest default is -1).
    _write_manifest_yaml(storage_root / "manifests", "clamp-model", reasoning_budget=3072)
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake",
            default_port_base=59700,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, grace_seconds=0, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
    )
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager

    captured_client_meta = {}

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    async def fake_complete(slot, handle):
        # Capture the ACTUAL client_meta the manager would forward -- this is
        # the real "does the outgoing request carry the clamped value" proof,
        # not just this function's own return value in isolation.
        captured_client_meta["last"] = dict(slot.client_meta or {})
        messages = (slot.client_meta or {}).get("messages") or []
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1700000000,
            "model": slot.model_tag,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"echo: {last_user}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete

    with TestClient(app) as client:
        yield app, client, captured_client_meta


class TestClampEndToEnd:
    def test_negative_control_normal_request_untouched(self, clamp_e2e_app):
        app, client, captured = clamp_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "normal-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
            },
        )
        assert r.status_code == 200, r.text
        # Completely untouched: no clamp was applied, so reasoning_budget was
        # never written into payload, so it never enters _COMMON_FORWARDED_KNOBS
        # -- exactly the same client_meta shape as before this change shipped.
        assert "reasoning_budget" not in captured["last"], captured["last"]

    def test_positive_control_ornith_qwen_shape_clamps_and_leaves_floor(self, clamp_e2e_app):
        app, client, captured = clamp_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "clamp-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2048,
            },
        )
        assert r.status_code == 200, r.text
        assert captured["last"].get("reasoning_budget") == 1024, captured["last"]

    def test_edge_no_max_tokens_unbounded_manifest_untouched(self, clamp_e2e_app):
        app, client, captured = clamp_e2e_app
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "clamp-model",
                "messages": [{"role": "user", "content": "hi"}],
                # no max_tokens at all -- and clamp-model's n_predict is
                # unbounded (-1 default) -- no ceiling exists.
            },
        )
        assert r.status_code == 200, r.text
        assert "reasoning_budget" not in captured["last"], captured["last"]

    def test_warning_and_clamp_coexist_on_the_same_request(self, caplog, clamp_e2e_app):
        # Design: keep the existing warning -- the clamp must not replace or
        # suppress it. Both fire on the exact same pathological request.
        app, client, captured = clamp_e2e_app
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "clamp-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 2048,
                },
            )
        assert r.status_code == 200, r.text
        warn_lines = [rec for rec in caplog.records if "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)" in rec.getMessage()]
        assert len(warn_lines) == 1, caplog.text
        assert captured["last"].get("reasoning_budget") == 1024, captured["last"]

    def test_positive_control_via_ollama_compat_route(self, clamp_e2e_app):
        # The second call site (ollama_chat, /api/chat) must clamp identically
        # -- both sites share the same clamp function and contract.
        app, client, captured = clamp_e2e_app
        r = client.post(
            "/api/chat",
            json={
                "model": "clamp-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2048,
                "stream": False,
            },
        )
        assert r.status_code == 200, r.text
        assert captured["last"].get("reasoning_budget") == 1024, captured["last"]
