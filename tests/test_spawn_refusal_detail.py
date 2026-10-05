"""Refused-spawn reporting:
"stop a refused spawn silently returning HTTP 200" has different
answers depending on transport and which of the two refusal call sites fires.

It was verified in chat_completion.py
that the refusals hit during a smoke run were ALL site 1
(_spawn_for_resident -> _run_spawn_safety_gate) -- discriminated
by the log format string, one token ("for slot-...") vs site 2's two
("for slot slot-..."). Site 1 threw a BARE string, "safety gates refused
spawn", with the actual reason computed six lines earlier (a
server-side WARNING) and then discarded. The client got an error, but one
with no reason in it -- exactly why such a run looks like "nothing is
happening" until nvidia-smi is checked.

Non-stream HTTP status: already correct today (chat_completion.py's
    `except RuntimeError as e: raise HTTPException(status_code=500, ...)`,
    both call sites). Expected GREEN pre-fix. No change needed,
    not a red->green -- see TestNonStreamAlready500.
THE WORK: _run_spawn_safety_gate returns the joined failed-gate
    detail string (not a bare bool), and _spawn_for_resident threads it into
    the RuntimeError instead of a fixed string. RED pre-fix (a bare string
    literal cannot contain "cpu_util"), GREEN post-fix -- see
    TestSite1RuntimeErrorCarriesDetail.
Streaming HTTP status: UNCHANGED, by design (rationale:
    stay consistent with the capacity-failure precedent). No new test --
    the existing tests/test_api_chat_completion.py::TestSseCapacityFrame::
    test_capacity_failure_emits_typed_frame_not_placement_failed already
    covers "refused spawn keeps HTTP 200 + in-band typed frame on this
    transport" for the sibling VramOverCommitError refusal; this change does
    not touch that contract.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import turbohaul.manager as manager_mod
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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.safety import GateResult
from turbohaul.slot import Slot


@pytest.fixture
def mgr(tmp_path):
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
    # safety_enabled=True -- deliberately:
    # a test that only passes with the gate off proves
    # nothing about the gate.
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=True, max_parallel_sidecars=8),
        pull=PullConfig(),
    )
    return TurbohaulManager(boot, runtime)


def _slot(tag):
    s = Slot.new(tag)
    s.completion_future = asyncio.get_event_loop().create_future()
    return s


def _resident(tag):
    return Resident(
        model_tag=tag, resident_key=tag, state=ResidentState.RESERVED_LOADING,
        main_gpu=0, split_mode="none",
    )


@pytest.mark.asyncio
class TestSite1RuntimeErrorCarriesDetail:
    """THE WORK. Drive the real _spawn_for_resident/
    _run_spawn_safety_gate path (site 1), refuse via a mocked
    all_safety_gates carrying a cpu_util refusal (the exact shape the
    live run produced), and check the exception actually landed on
    slot.completion_future carries the reason. RED pre-fix: the old code
    threw RuntimeError("safety gates refused spawn") -- a fixed string
    that cannot contain "cpu_util" no matter what the gates said. GREEN
    post-fix."""

    async def test_refusal_detail_reaches_the_completion_future(self, mgr, monkeypatch):
        def _gates(**kw):
            return [
                GateResult("ram", True, "ok"),
                GateResult(
                    "cpu_util", False,
                    "cpu-busy=95.0% > max 85.0%",
                ),
            ]

        monkeypatch.setattr(manager_mod, "all_safety_gates", _gates)
        monkeypatch.setattr(
            manager_mod, "read_manifest",
            lambda path, tag: (_ for _ in ()).throw(FileNotFoundError()),
        )

        def _spawn_should_not_be_called(*a, **kw):
            raise AssertionError(
                "a refused gate must not reach _spawn -- the RuntimeError "
                "should fire before this"
            )

        mgr._spawn = _spawn_should_not_be_called

        r = _resident("m1")
        slot = _slot("m1")
        result = await mgr._spawn_for_resident(r, slot)

        assert result is None, "a refused spawn must return None, not proceed"
        assert slot.completion_future.done()
        exc = slot.completion_future.exception()
        assert isinstance(exc, RuntimeError)
        msg = str(exc)
        assert "cpu_util" in msg, (
            f"the refusal reason must reach the client -- got {msg!r}, "
            "a bare 'safety gates refused spawn' with no detail is exactly "
            "the silent-200-shaped bug this change exists to fix "
            "(here: silent-no-reason, not silent-200, but the same root "
            "cause -- the detail computed at _run_spawn_safety_gate was "
            "discarded before reaching the caller)"
        )
        assert "95.0%" in msg, f"expected the numeric detail too, got {msg!r}"

    async def test_all_pass_still_reaches_spawn(self, mgr, monkeypatch):
        """Negative control: when every gate passes, _spawn_for_resident
        must still proceed to _spawn -- proves the refusal-detail plumbing
        didn't accidentally invert the gate's pass/fail branch."""
        monkeypatch.setattr(
            manager_mod, "all_safety_gates",
            lambda **kw: [GateResult("ram", True, "ok")],
        )
        monkeypatch.setattr(
            manager_mod, "read_manifest",
            lambda path, tag: (_ for _ in ()).throw(FileNotFoundError()),
        )
        called = {}

        def _spawn(binary, gguf, port, tag, argv, binary_fd=None):
            called["yes"] = True
            raise _StopSpawn

        mgr._spawn = _spawn

        r = _resident("m1")
        slot = _slot("m1")
        with pytest.raises(_StopSpawn):
            await mgr._spawn_for_resident(r, slot)
        assert called.get("yes") is True
        assert not slot.completion_future.done()


class _StopSpawn(Exception):
    """Marker exception to stop _spawn_for_resident right after the real _spawn
    call, without needing a real subprocess -- same idiom as
    the spawn-argv relocation test's _SpawnStop."""


class TestNonStreamAlready500:
    """CONTROL, not the work. The non-stream except-chain
    (in chat_completion.py) already converts any RuntimeError to
    HTTP 500 today, independent of what the message says -- verified on
    the module itself and reproduced here at the
    route level. Expected GREEN on BOTH the pristine tree and the
    fixed tree (the fix changes what the message SAYS, not whether it's a
    500) -- no change needed, not a red->green."""

    def _app(self, tmp_path, detail_msg):
        storage_root = tmp_path / "state"
        storage_root.mkdir()
        (storage_root / "blobs").mkdir()
        manifests_root = storage_root / "manifests"
        manifests_root.mkdir()
        (storage_root / "import-staging").mkdir()
        # chat-completion routes 404 on an unknown model tag before ever
        # reaching submit_and_wait -- a manifest must exist for the tag
        # this test dispatches against.
        (manifests_root / "m1.yaml").write_text(
            f"model_tag: m1\ngguf_blob_sha256: \"{'a' * 64}\"\ngguf_size_bytes: 1000\n"
        )
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
                default_port_base=59500,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        runtime = RuntimeConfig(
            queue=QueueConfig(safety_enabled=True),
            pull=PullConfig(),
        )
        app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
        mgr = app.state.manager

        async def _raise(*a, **kw):
            raise RuntimeError(detail_msg)

        mgr.submit_and_wait = _raise
        return app

    def test_openai_route_returns_500_with_reason(self, tmp_path):
        app = self._app(
            tmp_path, "safety gates refused spawn: cpu_util: cpu-busy=95.0% > max 85.0%",
        )
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 500
        assert "cpu_util" in r.text, (
            f"the reason must reach the HTTP client body -- got {r.text!r}"
        )
