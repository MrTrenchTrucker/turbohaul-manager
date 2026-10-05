"""Without the fix, the reaper (in manager.py) substituted a bare
RuntimeError for the driver's real typed exception (SidecarUnavailableError /
SidecarTimeoutError) when a cap>=2 resident's driver task died, so the route's
typed except-clauses (in chat_completion.py etc.) could never match --
every driver-death caller saw a generic 500 instead of 503+Retry-After or 504.

Note: there are TWO distinct failure paths, and only
ONE was broken.

  PATH A -- failure raised INLINE in the request's own task (e.g. httpx
    RemoteProtocolError surfacing directly from mgr._complete_fn on a cap<=1
    resident) -- already correctly returns 503+Retry-After today, does not
    touch the reaper at all, and is NOT changed by this fix. Already covered
    end-to-end by the sidecar-unavailable Retry-After test;
    ONE lightweight non-regression check is duplicated here (TestPathANonRegression)
    so this file's own RED/GREEN story is self-contained.

  PATH B -- the cap>=2 resident's DRIVER TASK ITSELF dies (an exception
    propagates out of _drive_resident, not out of one request's own await),
    _on_driver_done's done_callback fires, and _reap_dead_resident fails every
    pending future the dead driver owned. THIS is the actual fix.
    TestDriverDeathTypedException below drives it end-to-end (a real driver
    task dies with a real typed exception, through create_app + TestClient),
    not synthetically at the route layer -- so these tests fail against the
    unfixed reaper and pass against the fix. Reverting the
    propagation makes these tests fail.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.chat_completion import (
    SidecarTimeoutError,
    SidecarUnavailableError,
)
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
from unittest.mock import MagicMock


def _write_manifest_yaml(manifests_root, tag: str):
    (manifests_root / f"{tag}.yaml").write_text(
        f"""model_tag: {tag}
gguf_blob_sha256: "{'a' * 64}"
gguf_size_bytes: 1000
"""
    )


def _make_handle(model_tag: str, port: int, pid: int = 12345) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _app_with_dying_driver(tmp_path, *, raise_exc, max_parallel_sidecars):
    """Boots a real app (create_app + TestClient-compatible) whose sidecar
    "completes" by raising ``raise_exc`` for model 'm' -- with
    max_parallel_sidecars=2, this exception propagates out of _drive_resident
    itself (PATH B: kills the driver task, routes through _on_driver_done /
    _reap_dead_resident), not out of the request's own task (PATH A)."""
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    _write_manifest_yaml(storage_root / "manifests", "m")
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
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False,
            max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=0,
            idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=120,
        ),
        pull=PullConfig(),
    )
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        return _make_handle(model_tag, port)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(*a, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete_dies(slot, handle):
        raise raise_exc

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete_dies
    return app, mgr


class TestDriverDeathTypedException:
    """PATH B -- drives a real cap>=2 driver-task death through the reaper."""

    def test_sidecar_unavailable_driver_death_is_503_with_retry_after(self, tmp_path):
        exc = SidecarUnavailableError(
            "sidecar crashed or disconnected (transport error: "
            "RemoteProtocolError: Server disconnected without sending a "
            "response.)",
            cause="sidecar_disconnected_or_crashed",
            retry_after_s=30,
        )
        app, mgr = _app_with_dying_driver(
            tmp_path, raise_exc=exc, max_parallel_sidecars=2,
        )
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 503, r.text
        assert "Retry-After" in r.headers, r.headers
        # detail["error"] is a typed dict, not a
        # bare string.
        assert r.json()["detail"]["error"]["cause"] == "sidecar_disconnected_or_crashed"

    def test_sidecar_timeout_driver_death_is_504(self, tmp_path):
        exc = SidecarTimeoutError(
            "sidecar request timed out", retry_after_s=60,
        )
        app, mgr = _app_with_dying_driver(
            tmp_path, raise_exc=exc, max_parallel_sidecars=2,
        )
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 504, r.text
        assert r.headers.get("Retry-After") == "60"


class TestReaperCancelledPathStaysGeneric:
    """The no-exception fallback (exc is None) must still be used verbatim for
    the genuine no-fault case. Drives a REAL cap>=2 manager, cancels the live
    driver task directly (the actual way a cancellation reaches _on_driver_done
    in production -- e.g. shutdown/teardown racing an in-flight request), and
    checks the pending future's exception is the bare RuntimeError fallback,
    not a masqueraded typed sidecar fault."""

    async def test_cancelled_driver_task_yields_generic_failure_not_typed(self, tmp_path):
        from turbohaul.manager import TurbohaulManager

        storage_root = tmp_path / "state"
        storage_root.mkdir()
        (storage_root / "blobs").mkdir()
        (storage_root / "manifests").mkdir()
        (storage_root / "import-staging").mkdir()
        _write_manifest_yaml(storage_root / "manifests", "m1")
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
                default_port_base=59970,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        runtime = RuntimeConfig(
            queue=QueueConfig(
                safety_enabled=False,
                max_parallel_sidecars=2,
                grace_seconds=0,
                idle_hot_load_seconds=0,
                drained_sigterm_window_active_s=1,
                drained_sigterm_window_cold_s=1,
                loading_health_timeout_s=10,
            ),
            pull=PullConfig(),
        )
        hang_gate = asyncio.Event()

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete_hangs(slot, handle):
            await hang_gate.wait()
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete_hangs,
        )
        mgr.runtime.queue.safety_enabled = False
        from unittest.mock import patch
        with patch(
            "turbohaul.safety._read_free_vram_all_mib",
            return_value=[80000, 80000],
        ):
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                submit_task = asyncio.create_task(
                    mgr.submit_and_wait("m1", "a", thread_id="t1")
                )
                for _ in range(100):
                    await asyncio.sleep(0.02)
                    r = mgr._residents.get("m1")
                    if r is not None and r.driver_task is not None and r.active_slot is not None:
                        break
                assert r is not None and r.driver_task is not None
                pending_future = r.active_slot.completion_future
                assert pending_future is not None and not pending_future.done()

                r.driver_task.cancel()
                with pytest.raises(RuntimeError) as excinfo:
                    await asyncio.wait_for(submit_task, timeout=5)

                fail_exc = excinfo.value
                assert type(fail_exc) is RuntimeError, (
                    f"cancelled driver death must fall back to a bare "
                    f"RuntimeError, not masquerade as a typed sidecar fault; "
                    f"got {type(fail_exc)!r}"
                )
                assert not isinstance(
                    fail_exc, (SidecarUnavailableError, SidecarTimeoutError)
                )
            finally:
                hang_gate.set()
                await mgr.shutdown()


class TestPathANonRegression:
    """PATH A (inline request-task failure, cap<=1) was already correct and is
    untouched by this fix. Duplicated here (already covered by
    the sidecar-unavailable Retry-After test) so this file proves it
    independently."""

    def test_inline_request_task_failure_still_503_with_retry_after(self, tmp_path):
        storage_root = tmp_path / "state"
        storage_root.mkdir()
        (storage_root / "blobs").mkdir()
        (storage_root / "manifests").mkdir()
        (storage_root / "import-staging").mkdir()
        _write_manifest_yaml(storage_root / "manifests", "m")
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
                default_port_base=59950,
            ),
            ui=UIConfig(static_path=tmp_path / "ui"),
        )
        runtime = RuntimeConfig(
            queue=QueueConfig(
                safety_enabled=False,
                # default max_parallel_sidecars=1 -- cap<=1 worker_loop path,
                # deliberately NOT the reaper this fix changed.
                grace_seconds=0,
                idle_hot_load_seconds=0,
                drained_sigterm_window_active_s=1,
                drained_sigterm_window_cold_s=1,
                loading_health_timeout_s=120,
            ),
            pull=PullConfig(),
        )
        app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
        mgr = app.state.manager

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete_crashes(slot, handle):
            raise SidecarUnavailableError(
                "sidecar crashed or disconnected (transport error: "
                "RemoteProtocolError: Server disconnected without sending a "
                "response.)",
                cause="sidecar_disconnected_or_crashed",
                retry_after_s=30,
            )

        mgr._spawn = fake_spawn
        mgr._wait_healthy = fake_health
        mgr._sigterm = fake_sigterm
        mgr._vram_verify = fake_vram
        mgr._complete_fn = fake_complete_crashes

        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 503, r.text
        assert r.headers["Retry-After"] == "120"
        # detail["error"] is a typed dict.
        assert r.json()["detail"]["error"]["cause"] == "sidecar_disconnected_or_crashed"
