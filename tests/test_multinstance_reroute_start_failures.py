"""A second engine that fails to load or cannot be started falls back to the running one.

Same hand-off as for a refused engine, for the other two ways an extra engine can fail:
it starts but never becomes healthy, or starting it raises. With a live engine the
request is handed to it, the dead engine's driver ends cleanly, and the request does
not keep the dead engine's port and pid; with no other engine the request fails as it
always has.

Everything here runs real code with a recording spawn seam; the shared fakes are in
``_multiinstance_support.py``.
"""

import asyncio
import logging
import sys
from pathlib import Path

import pytest

_TREE = Path(__file__).resolve().parents[1]
for _p in (str(_TREE / "src"), str(_TREE / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from turbohaul.manager import ResidentState  # noqa: E402
from _multiinstance_support import (  # noqa: E402
    _manifest,
    _probe,
    _new_manager,
    _live_engine,
    _request,
    _track_started,
    _assert_driver_ended_cleanly,
)


# ---- a second engine that fails to load -----------------------------------------

async def _route_with_failing_load(tmp_path, *, live):
    """Route one new-session request for a one-card-per-engine tag, budget 2, two free
    cards, where every engine that is started never becomes healthy. ``live`` puts one
    idle engine of the tag on card 0 first."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2, healthy=False)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0) if live else None
    _track_started(manager)
    slot = _request("duo")
    slot.client_meta = {"session_id": "NEW"}
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5.0
            while loop.time() < deadline:
                if slot.completion_future.done() or (engine and not engine.inbox.empty()):
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)  # let the dead engine be reaped
        finally:
            await manager.shutdown()
    return slot, engine, spawn_calls, manager


async def test_a_second_engine_that_fails_to_load_routes_the_request_to_the_live_engine(
        tmp_path, caplog):
    """The second engine is spawned and never becomes healthy (a load timeout, or the
    process dying while it loads). The request has not been served by it, so it must
    be served by the engine that is already running, not failed."""
    slot, engine, spawn_calls, manager = await _route_with_failing_load(tmp_path, live=True)
    assert len(spawn_calls) == 1, "precondition: the second engine really was started"
    assert not slot.completion_future.done(), (
        f"the request was failed instead of routed: {slot.completion_future.exception()!r}"
    )
    assert engine.inbox.get_nowait() is slot
    assert [r.resident_key for r in manager._model_residents()] == ["duo"], (
        "the engine that failed to load must be gone"
    )
    _assert_driver_ended_cleanly(manager, caplog)


async def test_a_first_engine_that_fails_to_load_still_fails_the_request(tmp_path):
    """Control: with no other live engine there is nothing to fall back to, so a failed
    cold load fails the request as it always has."""
    slot, _engine, spawn_calls, _manager = await _route_with_failing_load(tmp_path, live=False)
    assert len(spawn_calls) == 1
    assert slot.completion_future.done()
    exc = slot.completion_future.exception()
    assert type(exc) is RuntimeError, repr(exc)
    assert str(exc) == "loading-fail-health-timeout"


# ---- a second engine that cannot even be started ---------------------------------

async def _route_with_raising_spawn(tmp_path, exc_type, *, live):
    """Route one new-session request for a one-card-per-engine tag, budget 2, two free
    cards, where starting any engine raises ``exc_type``. ``live`` puts one idle engine
    of the tag on card 0 first. Also returns the engines the manager tried to start."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2, spawn_raises=exc_type)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0) if live else None
    started = _track_started(manager)
    slot = _request("duo")
    slot.client_meta = {"session_id": "NEW"}
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5.0
            while loop.time() < deadline:
                if slot.completion_future.done() or (engine and not engine.inbox.empty()):
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.2)  # let the dead engine's driver finish its cleanup
            port_after = manager._alloc_port()
        finally:
            await manager.shutdown()
    return slot, engine, spawn_calls, manager, started, port_after


@pytest.mark.parametrize("exc_type", [OSError, RuntimeError])
async def test_a_second_engine_that_cannot_be_started_routes_the_request_to_the_live_engine(
        tmp_path, exc_type, caplog):
    """Starting the second engine raises. The request must be served by the engine that
    is already running, and the engine that failed to start must be cleaned up like any
    engine that dies early: out of the registry, its port free again, no affinity left
    pointing at it."""
    slot, engine, spawn_calls, manager, started, port_after = await _route_with_raising_spawn(
        tmp_path, exc_type, live=True)
    assert len(spawn_calls) == 1, "precondition: starting the second engine was attempted"
    assert not slot.completion_future.done(), (
        f"the request was failed instead of routed: {slot.completion_future.exception()!r}"
    )
    assert engine.inbox.get_nowait() is slot
    (dead,) = started
    assert dead.rerouted_slot is slot
    assert dead.state is ResidentState.DEAD and dead.torn_down
    assert [r.resident_key for r in manager._model_residents()] == ["duo"]
    assert port_after == dead.port, "the failed engine's port must be free again"
    assert "duo#1" not in set(manager._session_affinity.values())
    _assert_driver_ended_cleanly(manager, caplog)


@pytest.mark.parametrize("exc_type", [OSError, RuntimeError])
async def test_a_first_engine_that_cannot_be_started_fails_as_it_always_has(
        tmp_path, exc_type, caplog):
    """Control: with no other live engine the exception escapes the driver as it always
    did (the driver is reported as having died of it), and the request is failed by
    the driver's orphan check."""
    with caplog.at_level(logging.WARNING, logger="turbohaul.manager"):
        slot, _engine, spawn_calls, manager, started, _port = await _route_with_raising_spawn(
            tmp_path, exc_type, live=False)
    assert any(
        "driver for duo died" in r.getMessage() and exc_type.__name__ in r.getMessage()
        for r in caplog.records
    ), "the exception must escape the driver unchanged"
    assert len(spawn_calls) == 1
    assert slot.completion_future.done()
    exc = slot.completion_future.exception()
    assert type(exc) is RuntimeError, repr(exc)
    assert str(exc) == "resident 'duo' driver exited before serve"
    (dead,) = started
    assert dead.rerouted_slot is None


# ---- a rerouted request does not keep the dead engine's identifiers -----------------

async def test_a_rerouted_request_drops_the_identifiers_of_the_engine_that_died(
        tmp_path, monkeypatch):
    """The second engine starts, then something raises before it is recorded. The
    request is served by the live engine and no longer carries the dead engine's pid
    and port (they are back to what they were before the attempt: none)."""
    manager, boot, spawn_calls = _new_manager(tmp_path, budget=2)
    _manifest(boot, "duo", auto_place=True, llama_server_flags={"split_mode": "none"})
    engine = _live_engine(manager, "duo", gpu=0)

    def boom(slot):
        raise OSError("cannot record the identity")

    monkeypatch.setattr(manager, "_record_engine_identity", boom)
    slot = _request("duo")
    slot.client_meta = {"session_id": "NEW"}
    assert (slot.port, slot.pid) == (None, None)
    p1, p2 = _probe([20000, 20000])
    with p1, p2:
        try:
            await manager._route_or_reserve(slot)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5.0
            while loop.time() < deadline and engine.inbox.empty() and not slot.completion_future.done():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)
            assert len(spawn_calls) == 1, "precondition: the second engine really started"
            assert engine.inbox.get_nowait() is slot
            assert not slot.completion_future.done()
            assert (slot.port, slot.pid) == (None, None), "the dead engine's identifiers stayed"
        finally:
            await manager.shutdown()
