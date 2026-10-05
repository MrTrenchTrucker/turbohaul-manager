"""The non-streaming sidecar completion timeout must be a
REAL config field that actually reaches the httpx call, not a hardcoded
600.0 default argument main.py called with zero arguments (the pre-fix
shape: QueueConfig.sidecar_complete_timeout_s existed nowhere, and
make_llama_server_complete_fn()'s default was the only value that ever ran).

A test asserting the Field exists/parses proves nothing (a sound test bar for
this kind of change) -- both tests below prove the VALUE reaches the code that consumes
it: test_complete_fn_uses_configured_timeout_not_default proves it inside
create_app()'s real production wiring (in main.py), and
test_configured_timeout_reaches_httpx_call proves a non-default value
survives all the way to the timeout= kwarg passed to the httpx client.
"""
import asyncio
import inspect

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.chat_completion import make_llama_server_complete_fn
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
from turbohaul.slot import Slot


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class _FakeAsyncClient:
    """Captures the timeout= kwarg every caller passes, both to the
    constructor (httpx.AsyncClient(timeout=timeout_s)) and to client.post
    (which re-passes timeout_s explicitly, in chat_completion.py)."""

    def __init__(self, captured):
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None, timeout=None):
        self._captured.append(timeout)
        return _FakeResponse()


def test_configured_timeout_reaches_httpx_call():
    """A NON-default timeout_s, threaded through make_llama_server_complete_fn,
    must arrive at the actual httpx client.post call unchanged -- not the
    600.0 (old hardcode) or 3600.0 (new config default) value."""
    NON_DEFAULT_TIMEOUT = 42.0
    captured = []

    def factory():
        return _FakeAsyncClient(captured)

    complete_fn = make_llama_server_complete_fn(
        timeout_s=NON_DEFAULT_TIMEOUT, http_client_factory=factory
    )
    slot = Slot.new(model_tag="m", context=[{"role": "user", "content": "hi"}])

    class _Handle:
        port = 1234

    asyncio.run(complete_fn(slot, _Handle()))

    assert captured == [NON_DEFAULT_TIMEOUT], (
        f"expected the configured timeout {NON_DEFAULT_TIMEOUT} to reach "
        f"client.post's timeout kwarg, got {captured}"
    )


@pytest.fixture
def _boot_and_paths(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
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
    return boot


def test_complete_fn_uses_configured_timeout_not_default(_boot_and_paths):
    """MUTATION PROOF: change RuntimeConfig.queue.
    sidecar_complete_timeout_s away from the default and show the
    production-wired complete_fn's closed-over timeout_s actually changes
    with it -- through create_app()'s real wiring in main.py, not a
    hand-built complete_fn. If this field were decorative (main.py still
    calling make_llama_server_complete_fn() with no arguments), this test
    would see the OLD hardcoded 600.0 / the new field's default 3600.0
    regardless of what we set here, and fail."""
    NON_DEFAULT_TIMEOUT = 1234.5
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            sidecar_complete_timeout_s=NON_DEFAULT_TIMEOUT,
        ),
        pull=PullConfig(),
    )
    app = create_app(
        _boot_and_paths, runtime, auto_start_worker=False, auto_boot_reconcile=False
    )
    with TestClient(app):
        mgr = app.state.manager
        # mgr._complete_fn (in manager.py) is a wrapper
        # (_complete_fn_with_kv_reuse_outcome) that closes over the REAL
        # complete_fn under the name `_real_complete_fn`; that real fn is
        # `_complete`, defined inside make_llama_server_complete_fn, which
        # closes over timeout_s. Unwrap one level, then inspect that
        # closure -- this is how we observe the ACTUAL value bound into the
        # running production callable, not a value we assume was threaded
        # through.
        outer_vars = inspect.getclosurevars(mgr._complete_fn).nonlocals
        real_complete_fn = outer_vars["_real_complete_fn"]
        inner_vars = inspect.getclosurevars(real_complete_fn).nonlocals
        assert inner_vars.get("timeout_s") == NON_DEFAULT_TIMEOUT, (
            f"create_app() did not thread runtime.queue.sidecar_complete_timeout_s "
            f"into make_llama_server_complete_fn -- complete_fn closed over "
            f"{inner_vars.get('timeout_s')!r} instead of {NON_DEFAULT_TIMEOUT!r}. "
            f"The config field is decorative: it parses but the value never "
            f"reaches the runtime call, the exact defect this test guards against."
        )


def test_default_still_gives_streaming_parity(_boot_and_paths):
    """Sanity control: with NO override, the default must be 3600.0 (parity
    with STREAM_TIMEOUT_S), not the old 600.0 hardcode."""
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False), pull=PullConfig()
    )
    app = create_app(
        _boot_and_paths, runtime, auto_start_worker=False, auto_boot_reconcile=False
    )
    with TestClient(app):
        mgr = app.state.manager
        outer_vars = inspect.getclosurevars(mgr._complete_fn).nonlocals
        real_complete_fn = outer_vars["_real_complete_fn"]
        inner_vars = inspect.getclosurevars(real_complete_fn).nonlocals
        assert inner_vars.get("timeout_s") == 3600.0
