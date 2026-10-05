"""reasoning_budget cross-field validation.

Two independent checks:
  A. manifest.py save-time: Manifest construction rejects reasoning_budget >=
     n_predict (n_predict > 0). n_predict == -1 (unbounded) is exempt.
  B. chat_completion.py request-time: warn_if_reasoning_budget_exceeds_ceiling
     logs once per (model, caller) when the effective ceiling (min of a bounded
     manifest n_predict and the caller's max_tokens, whichever is smaller and
     actually set) is <= reasoning_budget. Exercised both as a pure unit and
     end-to-end through /v1/chat/completions (non-stream + stream) and
     /api/chat (including native Ollama options.num_predict), to prove the
     warning is NOT gated behind response_format:json_schema and DOES cover
     streaming -- the case an unconditional request-time check covers.
"""
import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

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
from turbohaul.manifest import Manifest, ModelManifest, ManifestValidationError
from turbohaul.subprocess_mgr import SidecarHandle


# ============================================================================
# A. manifest.py save-time check
# ============================================================================


def _mk_manifest(reasoning_budget, n_predict, tag="rb-test"):
    flags = {}
    if reasoning_budget is not None:
        flags["reasoning_budget"] = reasoning_budget
    if n_predict is not None:
        flags["n_predict"] = n_predict
    return ModelManifest(
        model_tag=tag,
        gguf_blob_sha256="a" * 64,
        gguf_size_bytes=1000,
        llama_server_flags=flags,
    )


class TestManifestSaveTimeCheck:
    def test_budget_equal_to_n_predict_rejected(self):
        # Example: budget 3192 against a bounded ceiling of 2048 is a
        # failing shape.
        with pytest.raises((ManifestValidationError, ValidationError)) as exc:
            _mk_manifest(reasoning_budget=2048, n_predict=2048)
        assert "reasoning_budget=2048" in str(exc.value)
        assert "n_predict=2048" in str(exc.value)
        assert "rb-test" in str(exc.value)

    def test_budget_greater_than_n_predict_rejected(self):
        with pytest.raises((ManifestValidationError, ValidationError)):
            _mk_manifest(reasoning_budget=3192, n_predict=2048)

    def test_budget_below_n_predict_accepted(self):
        # healthy case: budget 3192, ceiling 20480.
        m = _mk_manifest(reasoning_budget=3192, n_predict=20480)
        assert m.llama_server_flags["reasoning_budget"] == 3192

    def test_n_predict_unbounded_exempts_any_budget(self):
        # n_predict: -1 must NEVER trip this, however large reasoning_budget is.
        m = _mk_manifest(reasoning_budget=1_000_000, n_predict=-1)
        assert m.llama_server_flags["n_predict"] == -1

    def test_n_predict_absent_does_not_raise(self):
        m = _mk_manifest(reasoning_budget=5000, n_predict=None)
        assert m.llama_server_flags["reasoning_budget"] == 5000

    def test_reasoning_budget_absent_does_not_raise(self):
        m = _mk_manifest(reasoning_budget=None, n_predict=100)
        assert m.llama_server_flags["n_predict"] == 100

    def test_both_absent_does_not_raise(self):
        # MANIFEST_FLAG_DEFAULTS still injects its own unrelated defaults
        # (ctx_checkpoints/cache_type_k/cache_type_v) -- just confirm neither
        # reasoning_budget nor n_predict got invented out of thin air.
        m = _mk_manifest(reasoning_budget=None, n_predict=None)
        assert "reasoning_budget" not in m.llama_server_flags
        assert "n_predict" not in m.llama_server_flags

    def test_reasoning_budget_zero_does_not_raise_even_if_ge_n_predict(self):
        # 0 means "no thinking" (is_thinking_payload() also treats 0 as off);
        # 0 >= n_predict only when n_predict <= 0, i.e. unbounded -- not a
        # real conflict either way, but confirm no false-positive.
        m = _mk_manifest(reasoning_budget=0, n_predict=100)
        assert m.llama_server_flags["reasoning_budget"] == 0


# ============================================================================
# B1. warn_if_reasoning_budget_exceeds_ceiling -- pure unit tests
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_warn_cache():
    """The warn-once cache is module-level global state; isolate each test.

    getattr-guarded (not a plain attribute access) so a run against a tree
    that does not have the check yet -- where the cache doesn't exist -- lets
    each test fail on its OWN missing behavior instead of every test in this
    file erroring identically at fixture setup.
    """
    from turbohaul.api import chat_completion as cc
    cache = getattr(cc, "_reasoning_budget_warned", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


class TestWarnIfReasoningBudgetExceedsCeiling:
    def test_ceiling_from_caller_warns(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "qwen3.6-35b-mtp", {"reasoning_budget": 3192, "n_predict": -1},
                2048, thread_id="agent-1", ip=None,
            )
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "3192" in msg and "2048" in msg and "caller" in msg and "qwen3.6-35b-mtp" in msg

    def test_ceiling_from_manifest_warns(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "m", {"reasoning_budget": 5000, "n_predict": 4096},
                None, thread_id="agent-1", ip=None,
            )
        assert len(caplog.records) == 1
        assert "manifest" in caplog.records[0].getMessage()

    def test_min_of_both_picks_the_smaller_caller_side(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "m", {"reasoning_budget": 3000, "n_predict": 20000},
                2500, thread_id="agent-1", ip=None,
            )
        assert len(caplog.records) == 1
        assert "2500" in caplog.records[0].getMessage()
        assert "caller" in caplog.records[0].getMessage()

    def test_healthy_ceiling_above_budget_no_warn(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "qwen3.8-27b", {"reasoning_budget": 3192, "n_predict": 20480},
                None, thread_id="agent-1", ip=None,
            )
        assert len(caplog.records) == 0

    def test_zero_budget_never_warns(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "m", {"reasoning_budget": 0, "n_predict": -1},
                1, thread_id="agent-1", ip=None,
            )
        assert len(caplog.records) == 0

    def test_unbounded_everywhere_never_warns(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "m", {"reasoning_budget": 1_000_000, "n_predict": -1},
                None, thread_id="agent-1", ip=None,
            )
        assert len(caplog.records) == 0

    def test_fires_once_per_model_and_caller(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                warn_if_reasoning_budget_exceeds_ceiling(
                    "m", {"reasoning_budget": 3192, "n_predict": -1},
                    2048, thread_id="agent-1", ip=None,
                )
        assert len(caplog.records) == 1

    def test_distinct_caller_warns_again(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "m", {"reasoning_budget": 3192, "n_predict": -1},
                2048, thread_id="agent-1", ip=None,
            )
            warn_if_reasoning_budget_exceeds_ceiling(
                "m", {"reasoning_budget": 3192, "n_predict": -1},
                2048, thread_id="agent-2", ip=None,
            )
        assert len(caplog.records) == 2

    def test_distinct_model_warns_again_for_same_caller(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "model-a", {"reasoning_budget": 3192, "n_predict": -1},
                2048, thread_id="agent-1", ip=None,
            )
            warn_if_reasoning_budget_exceeds_ceiling(
                "model-b", {"reasoning_budget": 3192, "n_predict": -1},
                2048, thread_id="agent-1", ip=None,
            )
        assert len(caplog.records) == 2

    def test_ip_fallback_used_when_no_thread_id(self, caplog):
        from turbohaul.api.chat_completion import warn_if_reasoning_budget_exceeds_ceiling
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_budget_exceeds_ceiling(
                "m", {"reasoning_budget": 3192, "n_predict": -1},
                2048, thread_id=None, ip="10.0.0.5",
            )
            # Same IP, still no thread_id -> same key -> suppressed
            warn_if_reasoning_budget_exceeds_ceiling(
                "m", {"reasoning_budget": 3192, "n_predict": -1},
                2048, thread_id=None, ip="10.0.0.5",
            )
        assert len(caplog.records) == 1

    def test_cache_bounded_by_cap(self, caplog):
        from turbohaul.api import chat_completion as cc
        with caplog.at_level(logging.WARNING):
            for i in range(cc._REASONING_BUDGET_WARN_CAP + 20):
                cc.warn_if_reasoning_budget_exceeds_ceiling(
                    "m", {"reasoning_budget": 3192, "n_predict": -1},
                    2048, thread_id=f"agent-{i}", ip=None,
                )
        assert len(cc._reasoning_budget_warned) <= cc._REASONING_BUDGET_WARN_CAP
        assert len(caplog.records) == cc._REASONING_BUDGET_WARN_CAP + 20


# ============================================================================
# B2. End-to-end coverage proof via TestClient
# ============================================================================


def _make_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 12345
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _write_manifest_yaml_full(manifests_root, tag: str, reasoning_budget=None, n_predict=None):
    flags_lines = []
    if reasoning_budget is not None:
        flags_lines.append(f"  reasoning_budget: {reasoning_budget}")
    if n_predict is not None:
        flags_lines.append(f"  n_predict: {n_predict}")
    flags_block = ("llama_server_flags:\n" + "\n".join(flags_lines) + "\n") if flags_lines else ""
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
{flags_block}"""
    )


@pytest.fixture
def e2e_app(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    # Locked-budget model: reasoning_budget 3192 with NO n_predict set (manifest
    # side unbounded) -- so only a caller-supplied max_tokens can trip the warn,
    # isolating this fixture to the REQUEST-time half of the problem, not the save-time
    # half already covered above.
    _write_manifest_yaml_full(storage_root / "manifests", "thinking-model", reasoning_budget=3192)
    _write_manifest_yaml_full(storage_root / "manifests", "healthy-model", reasoning_budget=3192, n_predict=20480)
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
            default_port_base=59600,
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

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_handle(model_tag, port)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    async def fake_complete(slot, handle):
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
        yield app, client


class TestEndToEndCoverage:
    """Proves the request-time warning fires on an ordinary chat request --
    no response_format -- and on
    the streaming path, neither of which a check gated on response_format
    would cover.
    """

    def test_plain_nonstream_chat_no_response_format_warns(self, caplog, e2e_app):
        app, client = e2e_app
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "thinking-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 2048,
                },
            )
        assert r.status_code == 200, r.text
        warn_lines = [rec for rec in caplog.records if "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)" in rec.getMessage()]
        assert len(warn_lines) == 1, caplog.text
        assert "3192" in warn_lines[0].getMessage()
        assert "2048" in warn_lines[0].getMessage()

    def test_plain_nonstream_chat_healthy_model_no_warn(self, caplog, e2e_app):
        # healthy-model: budget 3192, manifest n_predict 20480 -- caller sends
        # NO max_tokens, so the only ceiling in play is the manifest's own
        # (healthy) one. (A caller-supplied 2048 here would correctly warn --
        # that's a DIFFERENT scenario, covered by test_min_of_both_picks_the_
        # smaller_caller_side above, not this no-warn control.)
        app, client = e2e_app
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "healthy-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert r.status_code == 200, r.text
        warn_lines = [rec for rec in caplog.records if "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)" in rec.getMessage()]
        assert len(warn_lines) == 0, caplog.text

    def test_streaming_chat_warns(self, caplog, e2e_app):
        """The streaming leg has no
        json_schema-gated thinking_manifest read at all, so without the
        unconditional chokepoint placement, nothing here would have warned.
        """
        app, client = e2e_app
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "thinking-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 2048,
                    "stream": True,
                },
            ) as r:
                assert r.status_code == 200
                for _ in r.iter_bytes():
                    pass
        warn_lines = [rec for rec in caplog.records if "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)" in rec.getMessage()]
        assert len(warn_lines) == 1, caplog.text

    def test_ollama_chat_top_level_max_tokens_warns(self, caplog, e2e_app):
        app, client = e2e_app
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/api/chat",
                json={
                    "model": "thinking-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 2048,
                },
            )
        assert r.status_code == 200, r.text
        warn_lines = [rec for rec in caplog.records if "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)" in rec.getMessage()]
        assert len(warn_lines) == 1, caplog.text

    def test_ollama_chat_native_options_num_predict_warns(self, caplog, e2e_app):
        """Native Ollama shape: ceiling nested under options.num_predict, no
        top-level max_tokens at all -- proves the options.* unwrap
        (mirroring the existing keep_alive unwrap) actually works.
        """
        app, client = e2e_app
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/api/chat",
                json={
                    "model": "thinking-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {"num_predict": 2048},
                },
            )
        assert r.status_code == 200, r.text
        warn_lines = [rec for rec in caplog.records if "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)" in rec.getMessage()]
        assert len(warn_lines) == 1, caplog.text

    def test_ollama_chat_no_ceiling_specified_no_warn(self, caplog, e2e_app):
        # thinking-model has NO manifest n_predict and the caller supplies
        # neither max_tokens nor options.num_predict -- unbounded on both
        # sides, nothing to compare, must not warn.
        app, client = e2e_app
        with caplog.at_level(logging.WARNING, logger="turbohaul.api.chat_completion"):
            r = client.post(
                "/api/chat",
                json={
                    "model": "thinking-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert r.status_code == 200, r.text
        warn_lines = [rec for rec in caplog.records if "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)" in rec.getMessage()]
        assert len(warn_lines) == 0, caplog.text
