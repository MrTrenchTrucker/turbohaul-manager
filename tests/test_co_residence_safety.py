"""A box that holds more engines than its budget must keep every one of them.

The periodic co-residence re-check may start a further engine whenever one fits, so a box can
hold more engines than ``queue.max_parallel_sidecars``. These tests pin what the rest of the
manager does with such a box: nothing may treat an engine over the budget as one to remove.

Only the first group (the orphan reaper) fails if the set of live engine pids handed to the
reaper is built only when the budget is at least 2: at a budget of 1 it would be empty and the
scan that follows a teardown would terminate every healthy engine on the managed ports. The set
must be built at every budget, including 1. The other groups (liveness sweep, room hint, request
hit, arrival eviction) pin current behaviour: nothing in the manager trims a box back to its
budget, and the in-budget controls next to each of them show that the check can fail.

The manager is real; the engines are hand-built residents with live fake processes. The
reapers that scan the host's processes are replaced in the ``turbohaul.manager`` namespace by
recorders that kill nothing.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import MagicMock

import pytest
import yaml

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
from turbohaul.fsm import transition as fsm_transition
from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

MIB = 1024 * 1024
NEED_MIB = 4000
WAIT_S = 3.0
PID_IDLE = 41001
PID_BOOTING = 41002


# --------------------------------------------------------------------------------------
# fixture: a real manager, hand-built residents, recorders instead of the host-scanning reapers
# --------------------------------------------------------------------------------------

def _manager(tmp_path, cap):
    root = tmp_path / "state"
    root.mkdir()
    for d in ("blobs", "manifests", "import-staging"):
        (root / d).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=root / "blobs",
            manifests_path=root / "manifests",
            import_allowed_root=root / "import-staging",
            state_db_path=root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(llama_server_binary=tmp_path / "fake_llama_server", default_port_base=59700),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=cap, grace_seconds=3600,
            max_grace_extensions=50, idle_hot_load_seconds=3600,
        ),
        pull=PullConfig(),
    )

    async def health(*a, **k):
        return True

    async def sigterm(handle, **k):
        return True, "sigterm-clean"

    async def vram(**k):
        return True, 100

    async def complete(slot, handle):
        return {"ok": True}

    def spawn(*a, **k):
        raise AssertionError("no test here may start an engine process")

    mgr = TurbohaulManager(boot, runtime, spawn_fn=spawn, health_fn=health, sigterm_fn=sigterm,
                           vram_fn=vram, complete_fn=complete)
    for tag in ("tag-a", "tag-b", "tag-c", "tag-d"):
        (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump({
            "model_tag": tag, "gguf_blob_sha256": "a" * 64, "gguf_size_bytes": 1000 * MIB,
            "context_size": 512, "expected_vram_bytes": NEED_MIB * MIB, "auto_place": False,
            "llama_server_flags": {"split_mode": "none", "main_gpu": 0},
        }))
    return mgr


class Reapers:
    """Replaces the two host-scanning reapers; records the pid sets each was handed."""

    def __init__(self, monkeypatch):
        self.boot_calls = []
        self.scan_calls = []
        monkeypatch.setattr("turbohaul.manager.boot_orphan_reaper", self._boot)
        monkeypatch.setattr("turbohaul.manager.intra_lifetime_orphan_scan", self._scan)

    def _boot(self, **kw):
        self.boot_calls.append(kw)
        return {"reaped": 0}

    def _scan(self, **kw):
        self.scan_calls.append(kw)
        return {"reaped": 0}


@pytest.fixture(autouse=True)
def reapers(monkeypatch):
    return Reapers(monkeypatch)


@pytest.fixture(autouse=True)
def roomy_cards(monkeypatch):
    """Two cards with plenty of free memory, for the tests that route a request."""
    monkeypatch.setattr("turbohaul.safety._read_free_vram_all_mib", lambda: [40000, 40000])
    monkeypatch.setattr("turbohaul.manager._read_free_vram_all_mib", lambda: [40000, 40000])


def _handle(tag, port, pid):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=tag)


def _resident(mgr, tag, *, pid=None, booting_pid=None, state=ResidentState.IDLE_EVICTABLE, last_active=1.0):
    """One engine of ``tag`` registered under its tag; ``pid`` gives it a live handle."""
    port = 59800 + len(mgr._residents)
    r = Resident(
        model_tag=tag, resident_key=tag, port=port, state=state, main_gpu=0, split_mode="none",
        reserved_need_mib=NEED_MIB, grace=GraceTimer(grace_seconds=3600, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=3600), last_active_monotonic=last_active,
        handle=_handle(tag, port, pid) if pid is not None else None, booting_pid=booting_pid,
    )
    r.inbox = asyncio.Queue()
    mgr._residents[tag] = r
    return r


def _tags(mgr):
    """The model tags of the engines the manager holds, sorted."""
    return sorted(r.model_tag for r in mgr._model_residents())


def _three_idle_engines(mgr):
    """Three idle engines with live processes; last use order: tag-b oldest, then tag-c, then tag-a."""
    a = _resident(mgr, "tag-a", pid=41011, last_active=30.0)
    b = _resident(mgr, "tag-b", pid=41012, last_active=10.0)
    c = _resident(mgr, "tag-c", pid=41013, last_active=20.0)
    return a, b, c


def _request(tag):
    slot = Slot.new(tag, prompt="p", thread_id="t-arrival")
    slot.client_meta = {}
    slot.completion_future = asyncio.get_running_loop().create_future()
    fsm_transition(slot, SlotState.STAGED)
    return slot


def _record_unloads(mgr):
    victims = []
    real = mgr._begin_unload_locked

    def spy(r):
        victims.append(r)
        return real(r)

    mgr._begin_unload_locked = spy
    return victims


def _record_reserves(mgr):
    reserves = []

    async def reserve(slot, **kw):
        reserves.append((slot, kw))

    mgr._reserve_and_start_locked = reserve
    return reserves


async def _until(predicate, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return False


# --------------------------------------------------------------------------------------
# the orphan reaper: every live engine is in the set it must not touch, whatever the budget
# --------------------------------------------------------------------------------------

BUDGETS = [pytest.param(1, id="budget_1_over_budget"), pytest.param(2, id="budget_2_control")]


async def _reap_with_two_engines(tmp_path, cap):
    """One idle engine with a live process and one engine still booting (only its pid is known),
    then the cleanup that follows every engine teardown."""
    mgr = _manager(tmp_path, cap)
    _resident(mgr, "tag-a", pid=PID_IDLE)
    _resident(mgr, "tag-b", booting_pid=PID_BOOTING, state=ResidentState.RESERVED_LOADING)
    assert len(mgr._model_residents()) == 2 and mgr.runtime.queue.max_parallel_sidecars == cap
    await mgr._reap_multi_engine_orphans("test")
    return mgr


ENGINES = [pytest.param(PID_IDLE, id="idle_engine"), pytest.param(PID_BOOTING, id="booting_engine")]


@pytest.mark.parametrize("pid", ENGINES)
@pytest.mark.parametrize("cap", BUDGETS)
async def test_the_scan_after_a_teardown_is_told_about_every_live_and_booting_engine(tmp_path, reapers, cap, pid):
    """The scan terminates any engine on the managed ports whose pid is not in the set it is handed.
    The idle engine's pid and the booting engine's pid must each be in it, at any box budget
    (one test per engine, so the two promises can fail on their own)."""
    mgr = await _reap_with_two_engines(tmp_path, cap)
    try:
        assert len(reapers.scan_calls) == 1, "the cleanup never ran the scan"
        protected = reapers.scan_calls[0]["known_handle_pids"]
        assert pid in protected, f"engine pid {pid} is unprotected: {sorted(protected)}"
    finally:
        await mgr.shutdown()


@pytest.mark.parametrize("pid", ENGINES)
@pytest.mark.parametrize("cap", BUDGETS)
async def test_the_boot_reaper_after_a_teardown_is_told_about_every_live_and_booting_engine(
    tmp_path, reapers, cap, pid,
):
    mgr = await _reap_with_two_engines(tmp_path, cap)
    try:
        assert len(reapers.boot_calls) == 1, "the cleanup never ran the boot reaper"
        protected = reapers.boot_calls[0]["known_pids"]
        assert pid in protected, f"engine pid {pid} is unprotected: {sorted(protected)}"
    finally:
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# the idle liveness sweep unloads the dead engine and nothing else
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cap", [pytest.param(1, id="budget_1_three_engines"), pytest.param(3, id="budget_3_control")],
)
async def test_one_sweep_pass_unloads_exactly_the_dead_idle_engine(tmp_path, monkeypatch, cap):
    mgr = _manager(tmp_path, cap)
    a = _resident(mgr, "tag-a", pid=41011)
    b = _resident(mgr, "tag-b", pid=41012)
    c = _resident(mgr, "tag-c", pid=41013)
    b.handle.proc.poll.return_value = -11            # the middle engine's process has died
    victims = _record_unloads(mgr)
    monkeypatch.setattr("turbohaul.manager._MULTI_ENGINE_IDLE_LIVENESS_SWEEP_INTERVAL_S", 60.0)   # one pass only
    task = asyncio.create_task(mgr._idle_engine_liveness_sweep())
    try:
        assert len(mgr._model_residents()) == 3 and mgr.runtime.queue.max_parallel_sidecars == cap
        assert await _until(lambda: "tag-b" not in mgr._residents), "the dead engine was never unloaded"
        assert victims == [b], f"the sweep unloaded {[v.model_tag for v in victims]}"
        assert mgr._residents.get("tag-a") is a and mgr._residents.get("tag-c") is c, "a live engine was removed"
        assert a.state is ResidentState.IDLE_EVICTABLE and c.state is ResidentState.IDLE_EVICTABLE
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# the room hint tells the picker there is no room, and removes nothing
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("cap", "expected"),
    [pytest.param(1, False, id="budget_1_three_engines"), pytest.param(3, False, id="budget_3_full"),
     pytest.param(4, True, id="budget_4_one_free_control")],
)
async def test_the_room_hint_follows_the_free_budget_and_unloads_nothing(tmp_path, cap, expected):
    mgr = _manager(tmp_path, cap)
    _three_idle_engines(mgr)
    victims = _record_unloads(mgr)
    try:
        assert mgr._dispatch_room_hint() is expected
        assert victims == [] and _tags(mgr) == ["tag-a", "tag-b", "tag-c"]
    finally:
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# a same-model request is served by the idle engine it belongs to; nothing is unloaded for it
# --------------------------------------------------------------------------------------

async def test_a_request_for_an_idle_engine_on_an_over_budget_box_is_served_by_it_and_unloads_nothing(tmp_path):
    mgr = _manager(tmp_path, 1)
    a, _b, _c = _three_idle_engines(mgr)
    victims = _record_unloads(mgr)
    reserves = _record_reserves(mgr)
    slot = _request("tag-a")
    try:
        assert len(mgr._model_residents()) == 3 > mgr.runtime.queue.max_parallel_sidecars
        await mgr._route_or_reserve(slot)
        assert list(a.inbox._queue) == [slot], "the request must land in the idle engine's inbox"
        assert a.state is ResidentState.ACTIVE, "the hit wakes the idle engine"
        assert victims == [], f"an engine was unloaded for a hit: {[v.model_tag for v in victims]}"
        assert reserves == [] and _tags(mgr) == ["tag-a", "tag-b", "tag-c"]
    finally:
        await mgr.shutdown()


# --------------------------------------------------------------------------------------
# a request for another model makes room by unloading ONE idle engine, never a cascade
# --------------------------------------------------------------------------------------

async def test_a_request_for_another_model_on_an_over_budget_box_unloads_exactly_one_least_recent_idle_engine(
    tmp_path,
):
    mgr = _manager(tmp_path, 1)
    a, b, c = _three_idle_engines(mgr)
    victims = _record_unloads(mgr)
    reserves = _record_reserves(mgr)
    slot = _request("tag-d")
    try:
        assert len(mgr._model_residents()) == 3 > mgr.runtime.queue.max_parallel_sidecars
        await mgr._route_or_reserve(slot)
        assert [v.model_tag for v in victims] == ["tag-b"], (
            f"expected exactly the least recently used idle engine, got {[v.model_tag for v in victims]}"
        )
        assert len(reserves) == 1 and reserves[0][0] is slot, "the freed slot was not used for the request"
        assert mgr._residents.get("tag-a") is a and mgr._residents.get("tag-c") is c, "another engine was removed"
    finally:
        await mgr.shutdown()
