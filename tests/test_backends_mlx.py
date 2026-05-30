"""Tests for MLX backend (backends/mlx.py), MLX manifest fields, and backend resolution.

These tests verify:
- MLX flag allowlist enforcement
- MLX command building (model_repo vs model_path)
- MLX backend spawn/health/teardown (mocked subprocess)
- Manifest validation for MLX fields
- Backend resolution in the manager
"""
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from turbohaul.backends import MLXBackend, SpawnRequest
from turbohaul.backends.mlx import SAFE_MLX_FLAGS, mlx_flags_to_argv


class TestMlxFlagsToArgv:
    """Test MLX flag-to-argv conversion."""

    def test_empty_flags(self):
        assert mlx_flags_to_argv({}) == []

    def test_single_bool_flag(self):
        assert mlx_flags_to_argv({"pipeline": True}) == ["--pipeline"]

    def test_bool_false_omitted(self):
        assert mlx_flags_to_argv({"pipeline": False}) == []

    def test_flag_with_value(self):
        result = mlx_flags_to_argv({"host": "0.0.0.0", "port": 8080})
        assert "--host" in result
        idx = result.index("--host")
        assert result[idx + 1] == "0.0.0.0"
        assert "--port" in result
        idx = result.index("--port")
        assert result[idx + 1] == "8080"

    def test_mixed_flags(self):
        result = mlx_flags_to_argv({"pipeline": True, "max_tokens": 1024})
        assert "--pipeline" in result
        assert "--max-tokens" in result
        idx = result.index("--max-tokens")
        assert result[idx + 1] == "1024"

    def test_snake_to_kebab(self):
        result = mlx_flags_to_argv({"max_tokens": 512})
        assert "--max-tokens" in result

    def test_unknown_flag_rejected(self):
        with pytest.raises(ValueError, match="allowlist"):
            mlx_flags_to_argv({"evil_flag": "bad"})

    def test_temp_flag(self):
        # mlx_lm uses --temp not --temperature
        result = mlx_flags_to_argv({"temp": 0.7})
        assert "--temp" in result
        idx = result.index("--temp")
        assert result[idx + 1] == "0.7"


class TestSafeMlxFlags:
    """Test MLX flag allowlist."""

    def test_known_flags_present(self):
        assert "host" in SAFE_MLX_FLAGS
        assert "port" in SAFE_MLX_FLAGS
        assert "max_tokens" in SAFE_MLX_FLAGS
        assert "temp" in SAFE_MLX_FLAGS
        assert "pipeline" in SAFE_MLX_FLAGS
        assert "log_level" in SAFE_MLX_FLAGS

    def test_removed_flags_absent(self):
        # These were in the old allowlist but are not real mlx_lm.server flags
        assert "verbose" not in SAFE_MLX_FLAGS
        assert "temperature" not in SAFE_MLX_FLAGS
        assert "cors_origin" not in SAFE_MLX_FLAGS
        assert "adapter" not in SAFE_MLX_FLAGS  # renamed to adapter_path

    def test_type_annotations(self):
        assert SAFE_MLX_FLAGS["host"] is str
        assert SAFE_MLX_FLAGS["port"] is int
        assert SAFE_MLX_FLAGS["max_tokens"] is int
        assert SAFE_MLX_FLAGS["temp"] is float
        assert SAFE_MLX_FLAGS["pipeline"] is bool
        assert SAFE_MLX_FLAGS["log_level"] is str


class TestMlxBackend:
    """Test MLXBackend spawn, health, and teardown (mocked)."""

    def test_backend_name(self):
        backend = MLXBackend()
        assert backend.name == "mlx"

    def test_spawn_uses_new_mlx_lm_invocation(self):
        """mlx_lm >= 0.21 uses `python -m mlx_lm server` not `mlx_lm.server`."""
        backend = MLXBackend()
        req = SpawnRequest(
            port=8080,
            model_tag="test-model",
            model_repo="mlx-community/test",
            python_binary=Path("/usr/bin/python3"),
        )
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            backend.spawn(req)
            cmd = mock_popen.call_args[0][0]
            # Must be: python -m mlx_lm server --model ...
            assert "-m" in cmd
            idx = cmd.index("-m")
            assert cmd[idx + 1] == "mlx_lm"
            assert cmd[idx + 2] == "server"

    def test_spawn_sets_sidecar_model_id_from_repo(self):
        backend = MLXBackend()
        req = SpawnRequest(
            port=8080,
            model_tag="test-model",
            model_repo="mlx-community/Llama-3.2-1B",
            python_binary=Path("/usr/bin/python3"),
        )
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345

        with patch("subprocess.Popen", return_value=mock_proc):
            handle = backend.spawn(req)
            assert handle.sidecar_model_id == "mlx-community/Llama-3.2-1B"

    def test_spawn_sets_sidecar_model_id_from_path(self):
        backend = MLXBackend()
        req = SpawnRequest(
            port=8080,
            model_tag="local-model",
            model_path=Path("/models/my-model"),
            python_binary=Path("/usr/bin/python3"),
        )
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 99

        with patch("subprocess.Popen", return_value=mock_proc):
            handle = backend.spawn(req)
            assert handle.sidecar_model_id == "/models/my-model"

    def test_spawn_returns_handle(self):
        backend = MLXBackend()
        req = SpawnRequest(
            port=8080,
            model_tag="test-model",
            model_repo="mlx-community/test",
            python_binary=Path("/usr/bin/python3"),
        )
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            handle = backend.spawn(req)
            assert handle is not None
            assert handle.port == 8080
            assert handle.model_tag == "test-model"
            assert handle.pid == 12345

    @pytest.mark.asyncio
    async def test_wait_healthy_success(self):
        backend = MLXBackend()
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": "list", "data": [{"id": "test"}]}

        async def mock_get(*args, **kwargs):
            return mock_response

        with patch.object(httpx.AsyncClient, "get", new=mock_get):
            healthy = await backend.wait_healthy(
                8080, timeout_s=1.0, poll_interval_s=0.1,
            )
            assert healthy is True

    @pytest.mark.asyncio
    async def test_wait_healthy_timeout(self):
        backend = MLXBackend()
        import httpx

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        with patch.object(httpx.AsyncClient, "get", new=mock_get):
            healthy = await backend.wait_healthy(
                8080, timeout_s=0.3, poll_interval_s=0.1,
            )
            assert healthy is False

    @pytest.mark.asyncio
    async def test_teardown_sigterm_ok(self):
        backend = MLXBackend()
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        handle = MagicMock()
        handle.proc = mock_proc
        handle.is_alive.return_value = False
        handle.pid = 12345

        ok, status = await backend.teardown(
            handle,
            drained_window_s=5.0,
            is_active=True,
            cold_window_s=2.0,
        )
        assert ok is True


class TestMlxManifest:
    """Test manifest MLX field validation."""

    def test_mlx_backend_default_llama(self):
        from turbohaul.manifest import Manifest

        m = Manifest(
            model_tag="test",
            gguf_blob_sha256="a" * 64,
            backend="llama.cpp",
        )
        assert m.backend == "llama.cpp"
        assert m.requires_llama_fields()
        assert not m.requires_mlx_fields()

    def test_mlx_backend_explicit(self):
        from turbohaul.manifest import Manifest

        m = Manifest(
            model_tag="mlx-model",
            backend="mlx",
            model_repo="mlx-community/Llama-3.2-1B",
        )
        assert m.backend == "mlx"
        assert not m.requires_llama_fields()
        assert m.requires_mlx_fields()

    def test_mlx_backend_invalid(self):
        from turbohaul.manifest import Manifest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="backend"):
            Manifest(
                model_tag="test",
                backend="vllm",
            )

    def test_mlx_model_path(self):
        from turbohaul.manifest import Manifest

        m = Manifest(
            model_tag="local-mlx",
            backend="mlx",
            model_path="/models/my-model",
        )
        assert m.model_path == "/models/my-model"
        assert not m.model_repo

    def test_mlx_flags_validation(self):
        from turbohaul.manifest import Manifest
        from pydantic import ValidationError

        # Valid flags using real allowlist values
        m = Manifest(
            model_tag="test",
            backend="mlx",
            mlx_server_flags={"pipeline": True, "max_tokens": 2048},
        )
        assert m.mlx_server_flags == {"pipeline": True, "max_tokens": 2048}

        # Invalid flag
        with pytest.raises(ValidationError, match="mlx_server_flags"):
            Manifest(
                model_tag="test",
                backend="mlx",
                mlx_server_flags={"evil_flag": "bad"},
            )

    def test_mlx_flag_type_mismatch(self):
        from turbohaul.manifest import Manifest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="mlx_server_flags"):
            Manifest(
                model_tag="test",
                backend="mlx",
                mlx_server_flags={"port": "not-an-int"},
            )

    def test_llama_model_no_gguf_required_for_mlx(self):
        """MLX models don't need gguf_blob_sha256."""
        from turbohaul.manifest import Manifest

        m = Manifest(
            model_tag="mlx-only",
            backend="mlx",
            model_repo="mlx-community/test",
        )
        assert m.gguf_blob_sha256 == ""

    def test_mlx_chat_template_flag(self):
        """chat_template flag lets local models skip HF network lookup."""
        from turbohaul.manifest import Manifest

        m = Manifest(
            model_tag="local-model",
            backend="mlx",
            model_path="/models/VibeThinker",
            mlx_server_flags={
                "chat_template": "/models/VibeThinker/chat_template.jinja",
            },
        )
        assert m.mlx_server_flags["chat_template"].endswith(".jinja")
