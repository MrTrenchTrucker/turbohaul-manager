"""Tests for the v0.6.0 MLX backend (src/turbohaul/mlx_spawn.py) and the MLX
manifest fields.

These verify:
- MLX flag allowlist enforcement (mlx_flags_to_argv / validate_mlx_flags)
- Command building: `python -m mlx_lm server --model <repo> --host 127.0.0.1 --port <p>`
- PYTHONPATH is popped in the child env (pythonpath-shadow-bug)
- Precondition refusal on non-Apple-Silicon hosts
- Manifest backend/model_repo/mlx_server_flags parsing + validation
- Manifest PUT accepts MLX fields via the manifests API
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import subprocess
import pydantic

from turbohaul.manifest import Manifest, ManifestValidationError
from turbohaul.mlx_spawn import (
    SAFE_MLX_FLAGS,
    is_mlx_available,
    mlx_flags_to_argv,
    mlx_spawn,
    validate_mlx_flags,
)


# --------------------------------------------------------------------------
# Flag allowlist
# --------------------------------------------------------------------------
class TestMlxFlagsToArgv:
    def test_empty_flags(self):
        assert mlx_flags_to_argv({}) == []

    def test_single_bool_true(self):
        assert mlx_flags_to_argv({"pipeline": True}) == ["--pipeline"]

    def test_bool_false_omitted(self):
        assert mlx_flags_to_argv({"pipeline": False}) == []

    def test_value_flag_kebab(self):
        result = mlx_flags_to_argv({"max_tokens": 1024, "host": "0.0.0.0", "port": 8080})
        assert result[result.index("--max-tokens") + 1] == "1024"
        assert result[result.index("--host") + 1] == "0.0.0.0"
        assert result[result.index("--port") + 1] == "8080"

    def test_unknown_flag_rejected(self):
        with pytest.raises(ValueError, match="allowlist"):
            mlx_flags_to_argv({"evil_flag": "bad"})

    def test_wrong_type_rejected(self):
        with pytest.raises(ValueError):
            validate_mlx_flags({"port": "not-an-int"})


class TestSafeMlxFlags:
    def test_known_flags_present(self):
        for k in ("host", "port", "max_tokens", "temp", "pipeline", "log_level"):
            assert k in SAFE_MLX_FLAGS

    def test_type_annotations(self):
        assert SAFE_MLX_FLAGS["host"] is str
        assert SAFE_MLX_FLAGS["port"] is int
        assert SAFE_MLX_FLAGS["max_tokens"] is int
        assert SAFE_MLX_FLAGS["temp"] is float
        assert SAFE_MLX_FLAGS["pipeline"] is bool


# --------------------------------------------------------------------------
# Spawn command building (mocked subprocess)
# --------------------------------------------------------------------------
def _mock_proc():
    p = MagicMock(spec=subprocess.Popen)
    p.pid = 12345
    p.poll.return_value = None
    return p


def test_spawn_builds_mlx_lm_server_cmd():
    """mlx_spawn must invoke `python -m mlx_lm server --model <repo>`."""
    proc = _mock_proc()
    with patch("subprocess.Popen", return_value=proc) as mock_popen, patch(
        "turbohaul.mlx_spawn._check_mlx_preconditions"
    ):
        handle = mlx_spawn(
            port=11555,
            model_tag="qwen3-1.7b",
            model_repo="mlx-community/Qwen3-1.7B-4B",
            model_path="",
            mlx_flags={"max_tokens": 4096},
            python_binary=Path("/usr/bin/python3"),
        )
    cmd = mock_popen.call_args[0][0]
    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == "mlx_lm"
    assert cmd[cmd.index("-m") + 2] == "server"
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "mlx-community/Qwen3-1.7B-4B"
    assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "11555"
    # max_tokens flowed through the allowlist
    assert "--max-tokens" in cmd and cmd[cmd.index("--max-tokens") + 1] == "4096"
    # handle shape matches SidecarHandle
    assert handle.port == 11555
    assert handle.model_tag == "qwen3-1.7b"


def test_spawn_pops_pythonpath_in_child_env():
    """pythonpath-shadow-bug: child must not inherit PYTHONPATH."""
    proc = _mock_proc()
    with patch.dict("os.environ", {"PYTHONPATH": "/evil/path"}, clear=False), patch(
        "subprocess.Popen", return_value=proc
    ) as mock_popen, patch("turbohaul.mlx_spawn._check_mlx_preconditions"):
        mlx_spawn(
            port=11556,
            model_tag="m",
            model_repo="mlx-community/M",
            model_path="",
            mlx_flags={},
            python_binary=Path("/usr/bin/python3"),
        )
    child_env = mock_popen.call_args.kwargs.get("env", {})
    assert "PYTHONPATH" not in child_env


def test_spawn_requires_repo_or_path():
    with patch("turbohaul.mlx_spawn._check_mlx_preconditions"):
        with pytest.raises(RuntimeError, match="model_repo"):
            mlx_spawn(11557, "m", "", "", {}, python_binary=Path("/usr/bin/python3"))


def test_spawn_uses_arbitrary_model_repo_verbatim():
    """No model is hardcoded: an arbitrary MLX repo id (e.g. any Qwen build)
    is passed through to --model exactly as given."""
    proc = _mock_proc()
    with patch("subprocess.Popen", return_value=proc) as mock_popen, patch(
        "turbohaul.mlx_spawn._check_mlx_preconditions"
    ):
        handle = mlx_spawn(
            port=11560,
            model_tag="any-tag",
            model_repo="mlx-community/Qwen3-8B-4bit",
            model_path="",
            mlx_flags={},
            python_binary=Path("/usr/bin/python3"),
        )
    cmd = mock_popen.call_args[0][0]
    assert cmd[cmd.index("--model") + 1] == "mlx-community/Qwen3-8B-4bit"
    # model_tag is just Turbohaul's name; the served model must be the repo id
    assert handle.model_id == "mlx-community/Qwen3-8B-4bit"


def test_spawn_uses_local_model_path_verbatim():
    """A local dir is used as --model verbatim (no download, no built-in default)."""
    proc = _mock_proc()
    with patch("subprocess.Popen", return_value=proc) as mock_popen, patch(
        "turbohaul.mlx_spawn._check_mlx_preconditions"
    ):
        mlx_spawn(
            port=11561,
            model_tag="local-tag",
            model_repo="",
            model_path="/Volumes/Models/SomeQwen-MLX",
            mlx_flags={},
            python_binary=Path("/usr/bin/python3"),
        )
    cmd = mock_popen.call_args[0][0]
    assert cmd[cmd.index("--model") + 1] == "/Volumes/Models/SomeQwen-MLX"


# --------------------------------------------------------------------------
# Precondition gating
# --------------------------------------------------------------------------
def test_preconditions_refuse_non_darwin():
    with patch("turbohaul.mlx_spawn.platform") as plat:
        plat.system.return_value = "Linux"
        plat.machine.return_value = "x86_64"
        with pytest.raises(RuntimeError, match="macOS"):
            mlx_spawn(
                11558, "m", "mlx-community/M", "", {}, python_binary=Path("/usr/bin/python3")
            )


def test_preconditions_refuse_non_arm64():
    with patch("turbohaul.mlx_spawn.platform") as plat:
        plat.system.return_value = "Darwin"
        plat.machine.return_value = "x86_64"
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            mlx_spawn(
                11559, "m", "mlx-community/M", "", {}, python_binary=Path("/usr/bin/python3")
            )


# --------------------------------------------------------------------------
# Manifest MLX fields
# --------------------------------------------------------------------------
def test_manifest_mlx_defaults():
    m = Manifest(
        model_tag="x",
        gguf_blob_sha256="",  # empty allowed for MLX
        backend="mlx",
        model_repo="mlx-community/Qwen3-1.7B-4B",
        mlx_server_flags={"max_tokens": 4096},
    )
    assert m.is_mlx()
    assert not m.is_llama_cpp()
    assert m.backend == "mlx"


def test_manifest_llamacpp_still_requires_sha256():
    with pytest.raises((ManifestValidationError, pydantic.ValidationError)):
        Manifest(model_tag="x", gguf_blob_sha256="", backend="llama.cpp")


def test_manifest_bad_backend_rejected():
    with pytest.raises((ManifestValidationError, pydantic.ValidationError)):
        Manifest(model_tag="x", gguf_blob_sha256="", backend="tensorrt")


def test_manifest_bad_mlx_flag_rejected():
    with pytest.raises((ManifestValidationError, pydantic.ValidationError), match="allowlist"):
        Manifest(
            model_tag="x",
            gguf_blob_sha256="",
            backend="mlx",
            model_repo="mlx-community/Qwen3-1.7B-4B",
            mlx_server_flags={"evil": "x"},
        )


# --------------------------------------------------------------------------
# SidecarHandle.model_id (completion proxy rewrite for MLX)
# --------------------------------------------------------------------------
class TestMlxModelId:
    def test_mlx_spawn_sets_model_id_to_model_arg(self):
        # mlx_lm server routes completion by the request `model` field, so the
        # handle must advertise the --model identity (HF repo or local path).
        proc = _mock_proc()
        fake_run = MagicMock()
        fake_run.returncode = 0
        fake_run.stdout = "usage: __main__.py"
        with patch("turbohaul.mlx_spawn.subprocess.Popen", return_value=proc), patch(
            "turbohaul.mlx_spawn.subprocess.run", return_value=fake_run
        ), patch("turbohaul.mlx_spawn._check_mlx_preconditions"):
            h = mlx_spawn(
                11555,
                "tag",
                model_repo="mlx-community/Qwen3-1.7B-4B",
                model_path="",
                mlx_flags={"pipeline": True},
            )
        assert h.model_id == "mlx-community/Qwen3-1.7B-4B"

    def test_mlx_spawn_model_id_prefers_local_path(self):
        proc = _mock_proc()
        fake_run = MagicMock()
        fake_run.returncode = 0
        fake_run.stdout = "usage: __main__.py"
        with patch("turbohaul.mlx_spawn.subprocess.Popen", return_value=proc), patch(
            "turbohaul.mlx_spawn.subprocess.run", return_value=fake_run
        ), patch("turbohaul.mlx_spawn._check_mlx_preconditions"):
            h = mlx_spawn(
                11556,
                "tag",
                model_repo="",
                model_path="/models/foo",
                mlx_flags={},
            )
        assert h.model_id == "/models/foo"


def test_completion_fn_rewrites_model_to_backend_identity():
    # The production completion_fn must send handle.model_id (not the Turbohaul
    # tag) as `model` so mlx_lm server does not try to re-fetch from HF.
    import asyncio

    from turbohaul.api.chat_completion import make_llama_server_complete_fn

    captured: dict = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

    def _factory():
        return _FakeClient()

    slot = MagicMock()
    slot.model_tag = "my-turbohaul-tag"
    slot.client_meta = {"messages": [{"role": "user", "content": "hi"}]}
    handle = MagicMock()
    handle.port = 11557
    handle.model_id = "/models/foo"  # MLX backend identity

    fn = make_llama_server_complete_fn(http_client_factory=_factory)
    asyncio.run(fn(slot, handle))
    assert captured["json"]["model"] == "/models/foo"
    assert captured["json"]["model"] != "my-turbohaul-tag"


def test_llama_cpp_handle_has_no_model_id_backward_compat():
    # llama.cpp SidecarHandles must leave model_id None so the completion proxy
    # keeps using slot.model_tag (unchanged pre-MLX behavior).
    from turbohaul.subprocess_mgr import SidecarHandle

    h = SidecarHandle(proc=MagicMock(spec=subprocess.Popen), port=1234, model_tag="t")
    assert h.model_id is None
