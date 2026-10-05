"""Session -> engine-instance affinity map.

Unit tests for the pure affinity helpers on TurbohaulManager:
  _session_key_for / _remember_affinity / _sticky_resident_key / _drop_affinity /
  _pick_instance_for / _purge_affinity_for  + the _session_affinity OrderedDict.

Mirrors tests/test_multislot_concurrency.py conventions (same _boot_runtime_multislot
/ _mocks / _mk shapes) so the same fixture shapes are reused. Uses
SimpleNamespace mock Residents carrying
resident_key — the registry carries Resident.resident_key; decoupling here keeps the
affinity unit independent of the registry re-key. All helpers assume the caller holds
_registry_lock; these single-threaded unit calls satisfy that trivially.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig, RuntimePathsConfig,
    ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime_multislot(tmp_path, *, max_parallel_sidecars=2):
    storage_root = tmp_path / "state"
    storage_root.mkdir(parents=True, exist_ok=True)
    (storage_root / "blobs").mkdir(exist_ok=True)
    (storage_root / "manifests").mkdir(exist_ok=True)
    (storage_root / "import-staging").mkdir(exist_ok=True)
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
    runtime = RuntimeConfig(
        queue=QueueConfig(max_parallel_sidecars=max_parallel_sidecars,
                          grace_seconds=0, idle_hot_load_seconds=0),
        pull=PullConfig(),
    )
    return boot, runtime


def _mocks():
    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = MagicMock(); proc.pid = 1; proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
                vram_fn=fake_vram, complete_fn=fake_complete)


def _mk(tmp_path, **kw):
    boot, runtime = _boot_runtime_multislot(tmp_path, **kw)
    mgr = TurbohaulManager(boot, runtime, **_mocks())
    mgr.runtime.queue.safety_enabled = False
    return mgr


class _FakeQ:
    def __init__(self, n): self._n = n
    def qsize(self): return self._n


def _resident(rk, tag="borage", *, state=ResidentState.ACTIVE, inflight=0,
              active=False, last=0.0, gpu=0, inbox=0):
    """Mock Resident with a resident_key."""
    return SimpleNamespace(
        resident_key=rk, model_tag=tag, state=state,
        inflight=[object()] * inflight, active_slot=(object() if active else None),
        last_active_monotonic=last, main_gpu=gpu, reserved_need_mib=0,
        inbox=(_FakeQ(inbox) if inbox else None),
    )


def _slot(tag="borage", session_id=None, thread_id=None):
    return SimpleNamespace(
        model_tag=tag, thread_id=thread_id or "",
        client_meta={"session_id": session_id} if session_id else {},
    )


# --------------------------------------------------------------------------------
def test_init_creates_session_affinity_map(tmp_path):
    mgr = _mk(tmp_path)
    # __init__ creates the affinity OrderedDict; it starts empty.
    from collections import OrderedDict
    assert isinstance(mgr._session_affinity, OrderedDict)
    assert len(mgr._session_affinity) == 0


def test_session_key_precedence(tmp_path):
    mgr = _mk(tmp_path)
    assert mgr._session_key_for(_slot(session_id="S1")) == "sess:S1"
    assert mgr._session_key_for(_slot(thread_id="TH")) == "tid:TH"
    assert mgr._session_key_for(_slot(session_id="S1", thread_id="TH")) == "sess:S1"
    assert mgr._session_key_for(_slot()) is None                 # unstickable
    assert mgr._session_key_for(_slot(session_id="")) is None     # empty not truthy


def test_session_key_non_dict_client_meta_degrades(tmp_path):
    # A non-dict client_meta must degrade to thread_id/None, never crash.
    mgr = _mk(tmp_path)
    bad = SimpleNamespace(model_tag="borage", thread_id="TH", client_meta="notadict")
    assert mgr._session_key_for(bad) == "tid:TH"
    bad2 = SimpleNamespace(model_tag="borage", thread_id="", client_meta=["x"])
    assert mgr._session_key_for(bad2) is None


def test_pick_records_under_passed_model_tag(tmp_path):
    # Affinity is recorded under the caller-PASSED model_tag, not
    # slot.model_tag — so record-key and lookup-key are the same tag.
    mgr = _mk(tmp_path)
    s = _slot(tag="slot_tag", session_id="AL")
    mgr._pick_instance_for(s, "route_tag", [_resident("route_tag#0", "route_tag", inflight=0)])
    assert mgr._sticky_resident_key(_slot(tag="route_tag", session_id="AL"), "route_tag") == "route_tag#0"
    assert mgr._sticky_resident_key(_slot(tag="slot_tag", session_id="AL"), "slot_tag") is None


def test_pick_least_loaded_and_records_sticky(tmp_path):
    mgr = _mk(tmp_path)
    insts = [_resident("borage#0", inflight=5), _resident("borage#1", inflight=0)]
    chosen = mgr._pick_instance_for(_slot(session_id="SESS"), "borage", insts)
    assert chosen.resident_key == "borage#1"                      # fewest in-flight
    # follow-up sticks to the same instance
    assert mgr._sticky_resident_key(_slot(session_id="SESS"), "borage") == "borage#1"
    # affinity is scoped per (tag, session): a different tag is independent
    assert mgr._sticky_resident_key(_slot(tag="other", session_id="SESS"), "other") is None


def test_active_slot_counts_as_load_and_tiebreak_by_recency(tmp_path):
    mgr = _mk(tmp_path)
    insts = [_resident("borage#0", inflight=1, active=True), _resident("borage#1", inflight=1)]
    assert mgr._pick_instance_for(_slot(session_id="A"), "borage", insts).resident_key == "borage#1"
    mgr2 = _mk(tmp_path / "b")
    insts = [_resident("borage#0", inflight=2, last=300.0), _resident("borage#1", inflight=2, last=50.0)]
    assert mgr2._pick_instance_for(_slot(session_id="A"), "borage", insts).resident_key == "borage#1"


def test_pick_prefers_ready_over_booting(tmp_path):
    # A warm ACTIVE instance (load 1) beats a still-booting
    # RESERVED_LOADING (load 0) — a new session shouldn't wait out a cold boot.
    mgr = _mk(tmp_path)
    insts = [_resident("borage#0", state=ResidentState.RESERVED_LOADING, inflight=0),
             _resident("borage#1", state=ResidentState.ACTIVE, inflight=1)]
    assert mgr._pick_instance_for(_slot(session_id="R"), "borage", insts).resident_key == "borage#1"
    # both booting -> falls through to least-loaded
    insts = [_resident("borage#0", state=ResidentState.RESERVED_LOADING, inflight=3),
             _resident("borage#1", state=ResidentState.RESERVED_LOADING, inflight=0)]
    assert mgr._pick_instance_for(_slot(session_id="R2"), "borage", insts).resident_key == "borage#1"


def test_pick_counts_inbox_depth(tmp_path):
    # Inbox-queued slots count as pending load so back-to-back new
    # sessions spread even before the driver drains the inbox.
    mgr = _mk(tmp_path)
    insts = [_resident("borage#0", inflight=0, inbox=5), _resident("borage#1", inflight=2, inbox=0)]
    assert mgr._pick_instance_for(_slot(session_id="R3"), "borage", insts).resident_key == "borage#1"


def test_purge_empty_resident_key_is_noop(tmp_path):
    # A resident carrying the "" dataclass default must NOT purge
    # every entry (they never map to "" but be defensive).
    mgr = _mk(tmp_path)
    mgr._remember_affinity("borage", "sess:A", "borage#0")
    mgr._purge_affinity_for(_resident("", "borage"))
    assert mgr._session_affinity.get(("borage", "sess:A")) == "borage#0"


def test_pick_returns_none_on_empty_or_all_dead(tmp_path):
    mgr = _mk(tmp_path)
    assert mgr._pick_instance_for(_slot(session_id="S"), "borage", []) is None
    assert mgr._pick_instance_for(_slot(session_id="S"), "borage",
                                  [_resident("borage#0", state=ResidentState.DEAD)]) is None
    # DEAD excluded even if it looks least-loaded
    insts = [_resident("borage#0", inflight=0, state=ResidentState.DEAD),
             _resident("borage#1", inflight=9)]
    assert mgr._pick_instance_for(_slot(session_id="S"), "borage", insts).resident_key == "borage#1"


def test_unstickable_session_routes_but_records_nothing(tmp_path):
    mgr = _mk(tmp_path)
    chosen = mgr._pick_instance_for(_slot(), "borage", [_resident("borage#0")])   # no session/thread
    assert chosen.resident_key == "borage#0"
    assert len(mgr._session_affinity) == 0


def test_purge_affinity_for_drops_only_that_resident(tmp_path):
    mgr = _mk(tmp_path)
    mgr._remember_affinity("borage", "sess:A", "borage#0")
    mgr._remember_affinity("borage", "sess:B", "borage#1")
    mgr._remember_affinity("borage", "sess:C", "borage#0")
    mgr._purge_affinity_for(_resident("borage#0"))
    assert ("borage", "sess:A") not in mgr._session_affinity
    assert ("borage", "sess:C") not in mgr._session_affinity
    assert mgr._session_affinity.get(("borage", "sess:B")) == "borage#1"
    # no resident_key -> no-op, no crash
    mgr._purge_affinity_for(SimpleNamespace())


def test_drop_affinity_single_session(tmp_path):
    mgr = _mk(tmp_path)
    mgr._remember_affinity("borage", "sess:A", "borage#0")
    mgr._drop_affinity(_slot(session_id="A"), "borage")
    assert ("borage", "sess:A") not in mgr._session_affinity


def test_lru_bound_and_recency(tmp_path, monkeypatch):
    import turbohaul.manager as M
    mgr = _mk(tmp_path)
    monkeypatch.setattr(M, "_AFFINITY_MAX", 4, raising=False)
    for i in range(10):
        mgr._remember_affinity("borage", f"sess:{i}", "borage#0")
    assert len(mgr._session_affinity) == 4                 # never exceeds bound
    assert ("borage", "sess:0") not in mgr._session_affinity  # oldest evicted
    assert ("borage", "sess:9") in mgr._session_affinity      # newest kept
    # recency: a read (move_to_end) protects a key from the next eviction
    mgr2 = _mk(tmp_path / "r")
    monkeypatch.setattr(M, "_AFFINITY_MAX", 4, raising=False)
    for i in range(4):
        mgr2._remember_affinity("borage", f"sess:{i}", "borage#0")
    mgr2._sticky_resident_key(_slot(session_id="0"), "borage")   # touch sess:0
    mgr2._remember_affinity("borage", "sess:4", "borage#0")      # overflow
    assert ("borage", "sess:0") in mgr2._session_affinity       # touched survived
    assert ("borage", "sess:1") not in mgr2._session_affinity   # untouched-oldest gone
