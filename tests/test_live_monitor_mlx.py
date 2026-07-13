"""Fix C: live tok/s for backends without /slots (MLX).

The LiveSlotsPoller polls GET /slots, which mlx_lm server does NOT implement,
so live tok/s/progress were stuck at the idle default. The fix derives REAL
tok/s from the streamed SSE token stream (note_stream_token) and surfaces it in
the poller even when _active_slot is None (streaming slots aren't registered as
_active_slot at cap<=1). Honest: non-streamed MLX requests have no mid-flight
signal, so tok/s stays null (never fabricated).
"""

import asyncio
import time

import pytest

from turbohaul import load_verify_log as lv
from turbohaul.config import (
    BootConfig, RuntimeConfig, StorageConfig, RuntimePathsConfig, UIConfig,
    ServerConfig, QueueConfig, PullConfig,
)
from turbohaul.live_monitor import LiveSlotsPoller
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
    runtime = RuntimeConfig(queue=QueueConfig(max_parallel_sidecars=1), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


def _fake_handle(port=11500, pid=51542):
    proc = type("P", (), {"pid": pid, "poll": lambda self: None})()
    return SidecarHandle(proc=proc, port=port, model_tag="m", parallel=1)


def test_note_stream_token_accumulates_and_rates(mgr):
    mgr._live_stream_stats = None
    gid = "gen-1"
    mgr.note_stream_token(gid)
    mgr.note_stream_token(gid)
    sts = mgr._live_stream_stats
    assert sts["n_decoded"] == 2
    assert sts["gen_id"] == gid
    assert isinstance(sts["tok_s"], float) and sts["tok_s"] >= 0.0


def test_clear_stream_stats_resets(mgr):
    mgr._live_stream_stats = {"gen_id": "g", "n_decoded": 5, "tok_s": 12.0, "last_t": time.monotonic()}
    mgr.clear_stream_stats()
    assert mgr._live_stream_stats is None


def test_tick_surfaces_fresh_stream_stats_without_active_slot(mgr):
    """Streaming slots at cap<=1 aren't _active_slot; the poller must still show
    live tok/s from _live_stream_stats (the MLX bug)."""
    mgr._active_slot = None
    mgr._active_handle = None
    mgr._live_stream_stats = {
        "gen_id": "gen-stream", "n_decoded": 7, "tok_s": 4.2,
        "last_t": time.monotonic(),
    }
    poller = LiveSlotsPoller(mgr, interval_s=1.0)
    asyncio.get_event_loop().run_until_complete(poller._tick())
    g = mgr.live_generation
    assert g is not None
    assert g["state"] == "generating"
    assert g["tok_s"] == 4.2
    assert g["n_decoded"] == 7


def test_tick_falls_through_when_stream_stats_stale(mgr):
    """Stale stream stats (no live stream) -> don't fabricate; fall to idle."""
    mgr._active_slot = None
    mgr._active_handle = _fake_handle()
    mgr._live_stream_stats = {
        "gen_id": "old", "n_decoded": 3, "tok_s": 9.0,
        "last_t": time.monotonic() - 10.0,  # stale (>3s)
    }
    poller = LiveSlotsPoller(mgr, interval_s=1.0)
    asyncio.get_event_loop().run_until_complete(poller._tick())
    g = mgr.live_generation
    assert g is not None
    # No fresh stream -> honest idle/transitioning, NOT a fabricated generating tok/s.
    assert g["state"] in ("idle", "transitioning")
    assert g["tok_s"] is None


def test_slots_unavailable_gen_fresh_vs_stale(mgr):
    from turbohaul.live_monitor import compute_generation_id
    poller = LiveSlotsPoller(mgr, interval_s=1.0)
    thread = "gen-x"
    gid = compute_generation_id(51542, 0, thread)  # same hash the poller computes
    # fresh
    mgr._live_stream_stats = {"gen_id": gid, "n_decoded": 4, "tok_s": 5.0, "last_t": time.monotonic()}
    fresh = poller._slots_unavailable_gen(mgr, 51542, 0, thread, "ACTIVE")
    assert fresh["state"] == "generating"
    assert fresh["tok_s"] == 5.0
    # stale
    mgr._live_stream_stats = {"gen_id": gid, "n_decoded": 4, "tok_s": 5.0, "last_t": time.monotonic() - 9.0}
    stale = poller._slots_unavailable_gen(mgr, 51542, 0, thread, "ACTIVE")
    assert stale["state"] == "ACTIVE"  # honest FSM state, no tok/s
    assert stale["tok_s"] is None
