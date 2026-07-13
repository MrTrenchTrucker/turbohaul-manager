"""Fix B: at cap<=1 the /status residents list must surface the singleton
sidecar (model_tag / state / pid / port / truthful model_resident) instead of
being empty and relying on the FE fallback. cap>=2 must stay on the registry
snapshot path (untouched)."""

import pytest

from turbohaul import load_verify_log as lv
from turbohaul.config import BootConfig, RuntimeConfig, StorageConfig, RuntimePathsConfig, UIConfig, ServerConfig, QueueConfig, PullConfig
from turbohaul.manager import TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


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
    # cap<=1 (the single-sidecar install case)
    runtime = RuntimeConfig(queue=QueueConfig(max_parallel_sidecars=1), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


def _handle(port=11500, pid=51542):
    fake_proc = type("P", (), {"pid": pid, "poll": lambda self: None})()
    return SidecarHandle(
        proc=fake_proc, port=port, model_tag="darwin-35b-a3b-opus-oq4", parallel=1
    )


def test_singleton_resident_populated_at_cap1(mgr):
    mgr._active_handle = _handle()
    mgr._idle_handle = _handle()
    mgr._idle_model_tag = "darwin-35b-a3b-opus-oq4"
    lv.clear_ring()
    lv.log_load_verify(
        event="model_load", trigger="spawn", model_tag="darwin-35b-a3b-opus-oq4",
        port=11500, pid=51542, process_alive=True, health_200=True,
        model_resident=True,
    )
    snap = mgr.status_snapshot()
    res = snap["residents"]
    assert len(res) == 1, res
    r = res[0]
    assert r["model_tag"] == "darwin-35b-a3b-opus-oq4"
    assert r["pid"] == 51542
    assert r["port"] == 11500
    assert r["model_resident"] is True
    assert r["state"] in ("ACTIVE", "GRACE", "IDLE_HOT", "LOADING")


def test_singleton_resident_empty_when_no_handle(mgr):
    # No sidecar alive → still empty (honest), never fabricated.
    snap = mgr.status_snapshot()
    assert snap["residents"] == []


def test_cap2_uses_registry_snapshot(mgr):
    # Bump to cap>=2: the singleton synthesis must NOT fire; residents comes
    # from _residents_snapshot (the Phase-0 singleton is excluded → empty here).
    mgr.runtime.queue.max_parallel_sidecars = 2
    mgr._active_handle = _handle()
    snap = mgr.status_snapshot()
    # cap>=2 path: registry snapshot, which excludes the Phase-0 singleton → []
    assert snap["residents"] == []
