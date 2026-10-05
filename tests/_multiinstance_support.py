"""Shared fakes and helpers for the multi-engine routing tests.

A manifest writer, a spawn seam that records every argv the manager builds, a free-VRAM
probe patch, a manager over a fresh runtime with a chosen box budget, live-engine and
request builders, and small trackers for the engines the manager starts. Used by
``test_multinstance_gate_vs_budget.py``, ``test_multinstance_cold_start_relocation.py``,
``test_multinstance_reroute_refused_spawn.py`` and
``test_multinstance_reroute_start_failures.py``. Not a test module itself.
"""

import asyncio
import sys
from pathlib import Path

import yaml

_TREE = Path(__file__).resolve().parents[1]
for _p in (str(_TREE / "src"), str(_TREE / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unittest.mock import MagicMock, patch  # noqa: E402

from turbohaul.fsm import transition as fsm_transition  # noqa: E402
from turbohaul.manager import (  # noqa: E402
    Resident,
    ResidentState,
    TurbohaulManager,
)
from turbohaul.slot import (  # noqa: E402
    Slot,
    SlotState,
)
from turbohaul.subprocess_mgr import SidecarHandle  # noqa: E402


MIB = 1024 * 1024


def _manifest(boot, tag: str, **overrides) -> None:
    """Write a manifest. Defaults describe a tag that declares NO placement."""
    body = {
        "model_tag": tag,
        "gguf_blob_sha256": "a" * 64,
        "gguf_size_bytes": 3000 * MIB,
        "context_size": 2048,
        "expected_vram_bytes": 3000 * MIB,
        "auto_place": False,
        "llama_server_flags": {},
    }
    body.update(overrides)
    (boot.storage.manifests_path / f"{tag}.yaml").write_text(yaml.safe_dump(body))


def _capturing_mocks(spawn_calls, healthy=True, spawn_raises=None):
    """Spawn seam that records every argv the manager builds for an engine."""
    pid = [97000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        spawn_calls.append({"model_tag": model_tag, "port": port, "argv": list(argv)})
        if spawn_raises is not None:
            raise spawn_raises("cannot exec the engine")
        pid[0] += 1
        proc = MagicMock()
        proc.pid = pid[0]
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(*a, **k):
        return healthy

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True, "model": handle.model_tag}

    return dict(spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
                vram_fn=fake_vram, complete_fn=fake_complete)


def _argv_value(argv, flag):
    """None => the flag is ABSENT from argv (only keys in the flags dict are emitted)."""
    key = "--" + flag.replace("_", "-")
    if key not in argv:
        return None
    return argv[argv.index(key) + 1]


def _probe(values):
    """Free-VRAM probe, patched at both binding points."""
    return (patch("turbohaul.safety._read_free_vram_all_mib", return_value=values),
            patch("turbohaul.manager._read_free_vram_all_mib", return_value=values))


def _new_manager(tmp_path, *, budget, healthy=True, spawn_raises=None):
    from test_placement_multiinstance import _boot_runtime

    boot, runtime = _boot_runtime(tmp_path, max_parallel_sidecars=budget)
    spawn_calls = []
    manager = TurbohaulManager(boot, runtime, **_capturing_mocks(spawn_calls, healthy, spawn_raises))
    manager.runtime.queue.safety_enabled = False
    return manager, boot, spawn_calls


def _live_engine(manager, tag, *, gpu=0, key=None):
    """An engine of ``tag`` already loaded and idle on card ``gpu``, with an inbox.

    ``key`` is its registry key: the tag itself for the first engine, ``tag#N`` for
    a later one."""
    key = key or tag
    r = Resident(
        model_tag=tag, resident_key=key, state=ResidentState.IDLE_EVICTABLE,
        last_active_monotonic=1.0, main_gpu=gpu, split_mode="none",
        inbox=asyncio.Queue(),
    )
    manager._residents[key] = r
    return r


def _request(tag):
    """A staged request, as the queue hands it to _route_or_reserve."""
    s = Slot.new(tag)
    s.client_meta = {}
    s.completion_future = asyncio.get_running_loop().create_future()
    fsm_transition(s, SlotState.STAGED)
    return s


def _track_started(manager):
    """Record every engine the manager starts for a request (the value returned by
    ``_reserve_and_start_locked``), delegating to the real method. The list is also
    kept on the manager as ``started_engines``."""
    started = []
    real_reserve = manager._reserve_and_start_locked

    async def reserve(*a, **k):
        r = await real_reserve(*a, **k)
        started.append(r)
        return r

    manager._reserve_and_start_locked = reserve
    manager.started_engines = started
    return started


def _assert_driver_ended_cleanly(manager, caplog):
    """The refused/failed engine's driver ended normally: it did not die of an
    exception, and nothing was logged about a driver dying."""
    (dead,) = manager.started_engines
    assert dead.driver_task.done() and not dead.driver_task.cancelled()
    assert dead.driver_task.exception() is None
    assert not [r for r in caplog.records if "driver for" in r.getMessage()
                and "died" in r.getMessage()]


async def _wait_for_spawn(spawn_calls, *, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if spawn_calls:
            return True
        await asyncio.sleep(0.01)
    return False
