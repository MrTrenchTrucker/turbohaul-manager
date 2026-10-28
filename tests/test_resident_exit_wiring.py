"""Resident exit behaviour, pinned on the real TurbohaulManager.

Two behaviours are pinned. (1) A keep_alive=0 resident with a live same-model ACTIVE sibling is
held warm for a bounded window instead of torn down at once, with a hold-warm log line.
(2) Every resident death writes exactly ONE RESIDENT_EXIT line, except make-room deaths,
whose MAKE_ROOM_EVICTION line is written by the caller. These tests read the log, so they
would fail without the hold and the exit line.

This file imports nothing from turbohaul.resident_exit: it pins the manager's behaviour,
not the module's interface.
"""
import ast
import asyncio
import contextlib
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import turbohaul.manager as manager_mod
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
from turbohaul.queue import GraceTimer, IdleHotTimer
from turbohaul.slot import Slot, SlotState
from turbohaul.subprocess_mgr import SidecarHandle

GRACE_S = 1
MANAGER_SRC = Path(__file__).resolve().parents[1] / "src" / "turbohaul" / "manager.py"


def _boot_runtime(tmp_path, *, idle_hot_load_seconds, max_parallel_sidecars=2):
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
            default_port_base=59940,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            safety_enabled=False, max_parallel_sidecars=max_parallel_sidecars,
            grace_seconds=GRACE_S, max_grace_extensions=50,
            idle_hot_load_seconds=idle_hot_load_seconds,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _fake_handle(model_tag, port, pid):
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _mocks():
    pid = [93_000]

    def fake_spawn(binary, gguf, port, model_tag, argv, **kw):
        pid[0] += 1
        return _fake_handle(model_tag, port, pid[0])

    async def fake_health(*a, **k):
        return True

    async def fake_sigterm(handle, **k):
        return True, "sigterm-clean"

    async def fake_vram(**k):
        return True, 100

    async def fake_complete(slot, handle):
        return {"ok": True}

    return dict(
        spawn_fn=fake_spawn, health_fn=fake_health, sigterm_fn=fake_sigterm,
        vram_fn=fake_vram, complete_fn=fake_complete,
    )


def _manager(tmp_path, **kw):
    boot, runtime = _boot_runtime(tmp_path, **kw)
    return TurbohaulManager(boot, runtime, **_mocks())


def _resident(mgr, model_tag, port, *, resident_key=None, state=ResidentState.ACTIVE):
    key = model_tag if resident_key is None else resident_key
    r = Resident(
        model_tag=model_tag, resident_key=key, port=port,
        grace=GraceTimer(grace_seconds=GRACE_S, max_extensions=50),
        idle=IdleHotTimer(idle_seconds=0),
        rank_client_meta=None,
        last_active_monotonic=time.monotonic(),
        state=state,
    )
    r.inbox = asyncio.Queue()
    mgr._residents[key] = r
    return r


def _exit_lines(caplog):
    return [m for m in (rec.getMessage() for rec in caplog.records) if m.startswith("RESIDENT_EXIT ")]


def _hold_lines(caplog):
    return [m for m in (rec.getMessage() for rec in caplog.records) if "resident hold-warm" in m]


async def _drive_one_turn(mgr, r, *, timeout_s, thread_id="t1", client_meta=None, observe=None):
    """Put one anchor slot in r.inbox, run _drive_resident directly, report the first terminal
    outcome ('evicted' or 'idle_evictable') and call ``observe()`` at that moment, BEFORE the
    driver is cancelled. A cancelled parked driver writes its own exit line, so anything that
    must not appear has to be read here, not after the cancel."""
    anchor = Slot.new(r.model_tag, prompt="hi", thread_id=thread_id, client_meta=client_meta)
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)

    drive_task = asyncio.create_task(mgr._drive_resident(r))
    snap = {}
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            cur = mgr._residents.get(r.resident_key)
            if cur is None:
                outcome = "evicted"
            elif cur.state is ResidentState.IDLE_EVICTABLE:
                outcome = "idle_evictable"
            else:
                await asyncio.sleep(0.05)
                continue
            snap.update(
                outcome=outcome, registered=cur is r,
                state=None if cur is None else cur.state,
                idle_expires_at=None if cur is None else cur.idle_expires_at,
                observed=None if observe is None else observe(),
            )
            return snap
        snap.update(outcome="timeout", registered=False, state=None, idle_expires_at=None, observed=None)
        return snap
    finally:
        drive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task


def _sibling(mgr, model_tag, port, *, state=ResidentState.ACTIVE):
    return _resident(mgr, model_tag, port, resident_key=f"{model_tag}#1", state=state)


@pytest.mark.asyncio
async def test_W1_same_model_sibling_holds_a_keep_alive_zero_resident_warm(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w1-model", 59941)
    _sibling(mgr, "w1-model", 59942)
    unloads_before = mgr._unload_count
    try:
        snap = await _drive_one_turn(
            mgr, r, timeout_s=GRACE_S + 4.0,
            observe=lambda: {"exit": _exit_lines(caplog), "hold": _hold_lines(caplog)},
        )
        assert snap["outcome"] == "idle_evictable", (
            "a keep_alive=0 resident with a same-model ACTIVE sibling must be held warm, "
            f"not torn down -- outcome={snap['outcome']!r}"
        )
        assert snap["registered"] is True
        assert snap["state"] is ResidentState.IDLE_EVICTABLE
        assert mgr._unload_count == unloads_before
        remaining = snap["idle_expires_at"] - time.monotonic()
        assert 25.0 <= remaining <= 31.0, f"the hold must be bounded to about 30s, got {remaining!r}"
        holds = snap["observed"]["hold"]
        assert len(holds) == 1, f"expected one hold-warm line, got {holds!r}"
        assert "model_tag=w1-model" in holds[0]
        assert "active_siblings=1" in holds[0]
        assert snap["observed"]["exit"] == [], "a held resident is not dead, so no exit line"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_meta",
    [{"is_curator": True, "role": "curator"}, {"is_compression": True}],
    ids=["curator", "compression"],
)
async def test_W2_the_identity_class_of_the_finished_request_does_not_change_the_hold(
    tmp_path, caplog, client_meta,
):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w2-model", 59943)
    _sibling(mgr, "w2-model", 59944)
    try:
        snap = await _drive_one_turn(
            mgr, r, timeout_s=GRACE_S + 4.0,
            thread_id="hermes-main-20260101_000000_a1b2c3", client_meta=client_meta,
            observe=lambda: {"hold": _hold_lines(caplog)},
        )
        assert snap["outcome"] == "idle_evictable", (
            f"a disposable identity must not decide whether the resident lives -- outcome={snap['outcome']!r}"
        )
        assert snap["state"] is ResidentState.IDLE_EVICTABLE
        assert len(snap["observed"]["hold"]) == 1
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W3_the_hold_is_bounded_and_ends_in_one_self_timeout_line(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    monkeypatch.setattr(manager_mod, "_SAME_MODEL_QUEUED_HOLD_S", 0.2)
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w3-model", 59945)
    _sibling(mgr, "w3-model", 59946)
    unloads_before = mgr._unload_count
    anchor = Slot.new("w3-model", prompt="hi", thread_id="t1")
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)
    drive_task = asyncio.create_task(mgr._drive_resident(r))
    saw_held = False
    try:
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and mgr._residents.get("w3-model") is r:
            if r.state is ResidentState.IDLE_EVICTABLE:
                saw_held = True
            await asyncio.sleep(0.01)
        assert saw_held, "the same-model co-resident was never held warm before its window"
        assert mgr._residents.get("w3-model") is None, "the held resident outlived its bounded window"
        assert r.state is ResidentState.DEAD
        assert mgr._unload_count == unloads_before + 1
        lines = _exit_lines(caplog)
        assert len(lines) == 1, f"expected exactly one RESIDENT_EXIT line, got {lines!r}"
        assert "reason=self_idle_timeout" in lines[0]
        assert "state_before=IDLE_EVICTABLE" in lines[0]
        assert "state_after=DEAD" in lines[0]
    finally:
        drive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W4_a_lone_keep_alive_zero_resident_writes_the_exact_exit_line(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w4-model", 59961)
    try:
        snap = await _drive_one_turn(mgr, r, timeout_s=GRACE_S + 4.0, thread_id="thread-abc")
        assert snap["outcome"] == "evicted"
        assert _exit_lines(caplog) == [
            "RESIDENT_EXIT reason=immediate_evict_keep_alive_0 resident_key=w4-model "
            "model_tag=w4-model thread=thread-abc port=59961 state_before=ACTIVE state_after=DEAD"
        ]
        assert _hold_lines(caplog) == [], "with no sibling there is nothing to hold for"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W5a_designated_victim_with_positive_keep_alive_is_unloaded_with_the_designation_reason(
    tmp_path, caplog, monkeypatch,
):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=300)
    r = _resident(mgr, "w5a-model", 59963)
    _sibling(mgr, "w5a-model", 59964)
    monkeypatch.setattr(mgr, "_is_designated_unload_target_locked", lambda rr: rr is r)
    try:
        snap = await _drive_one_turn(
            mgr, r, timeout_s=GRACE_S + 4.0, thread_id="thread-w5a",
            observe=lambda: _exit_lines(caplog),
        )
        assert snap["outcome"] == "evicted", (
            "a designated victim must be unloaded at once, not parked for its keep_alive window -- "
            f"outcome={snap['outcome']!r}"
        )
        lines = snap["observed"]
        assert len(lines) == 1, f"expected exactly one RESIDENT_EXIT line, got {lines!r}"
        assert "reason=designated_victim_idle_break" in lines[0]
        assert "state_before=ACTIVE" in lines[0]
        assert "state_after=DEAD" in lines[0]
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W5b_designated_victim_at_keep_alive_zero_logs_the_immediate_reason(
    tmp_path, caplog, monkeypatch,
):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w5b-model", 59965)
    _sibling(mgr, "w5b-model", 59966)
    monkeypatch.setattr(mgr, "_is_designated_unload_target_locked", lambda rr: rr is r)
    try:
        snap = await _drive_one_turn(
            mgr, r, timeout_s=GRACE_S + 4.0, thread_id="thread-w5b",
            observe=lambda: _exit_lines(caplog),
        )
        assert snap["outcome"] == "evicted"
        assert snap["observed"] == [
            "RESIDENT_EXIT reason=immediate_evict_keep_alive_0 resident_key=w5b-model "
            "model_tag=w5b-model thread=thread-w5b port=59965 state_before=ACTIVE state_after=DEAD"
        ]
        assert _hold_lines(caplog) == [], (
            "a designated victim must never be reported as held warm, even with a same-model "
            f"sibling: {_hold_lines(caplog)!r}"
        )
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W6_reap_of_a_live_resident_writes_one_driver_reap_line(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w6-model", 59947)
    try:
        await mgr._reap_dead_resident(r, RuntimeError("x"))
        lines = _exit_lines(caplog)
        assert len(lines) == 1, f"expected exactly one RESIDENT_EXIT line, got {lines!r}"
        assert "reason=driver_reap" in lines[0]
        assert "state_before=ACTIVE" in lines[0]
        assert r.state is ResidentState.DEAD
        assert "w6-model" not in mgr._residents
        r.exit_reason = "self_idle_timeout"
        async with mgr._registry_lock:
            mgr._begin_unload_locked(r)
        assert len(_exit_lines(caplog)) == 1, "a dead resident must log nothing on a second death call"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W7_driver_finally_death_writes_one_driver_exit_line(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=300)
    r = _resident(mgr, "w7-model", 59948)
    anchor = Slot.new("w7-model", prompt="hi", thread_id="thread-w7")
    anchor.state = SlotState.STAGED
    await r.inbox.put(anchor)
    drive_task = asyncio.create_task(mgr._drive_resident(r))
    try:
        deadline = time.monotonic() + GRACE_S + 6.0
        while time.monotonic() < deadline and r.state is not ResidentState.IDLE_EVICTABLE:
            await asyncio.sleep(0.05)
        assert r.state is ResidentState.IDLE_EVICTABLE, "precondition: the resident must be parked"
        drive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task
        assert mgr._residents.get("w7-model") is None, "the cancelled driver must deregister its resident"
        assert r.state is ResidentState.DEAD
        lines = _exit_lines(caplog)
        assert len(lines) == 1, f"expected exactly one RESIDENT_EXIT line, got {lines!r}"
        assert "reason=driver_exit" in lines[0]
        assert "state_before=IDLE_EVICTABLE" in lines[0]
        assert "port=59948" in lines[0], f"the driver_exit line must carry the handle's port: {lines[0]!r}"
        assert "thread=thread-w7" in lines[0]
        await mgr._reap_dead_resident(r)
        assert len(_exit_lines(caplog)) == 1, "the supervisor reap must add no second line"
    finally:
        drive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drive_task
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W8_make_room_unload_writes_no_exit_line_of_its_own(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    victim = _resident(mgr, "w8-model", 59949)
    control = _resident(mgr, "w8-control", 59955)
    try:
        victim.exit_reason = "make_room"
        async with mgr._registry_lock:
            mgr._begin_unload_locked(victim)
        assert victim.state is ResidentState.DEAD
        assert "w8-model" not in mgr._residents
        assert _exit_lines(caplog) == [], "the make-room caller writes its own MAKE_ROOM_EVICTION line"
        # Control on the same call path: a non-make-room stamp must write its line, so the silence
        # above belongs to the make-room stamp and not to a logger that writes nothing at all.
        control.exit_reason = "driver_reap"
        async with mgr._registry_lock:
            mgr._begin_unload_locked(control)
        assert _exit_lines(caplog) == [
            "RESIDENT_EXIT reason=driver_reap resident_key=w8-control model_tag=w8-control "
            "thread=- port=- state_before=ACTIVE state_after=DEAD"
        ]
    finally:
        await mgr.shutdown()


def _is_begin_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_begin_unload_locked"
    )


def _is_one_arg_call(call):
    return len(call.args) == 1 and not call.keywords


def _block_of(tree, call):
    """The statement list that holds this call as an expression statement, and the call's index."""
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(parent, field, None)
            if not isinstance(stmts, list):
                continue
            for i, stmt in enumerate(stmts):
                if isinstance(stmt, ast.Expr) and stmt.value is call:
                    return stmts, i
    raise AssertionError("the _begin_unload_locked call is not an expression statement in a block")


def _stamp_for(stmts, i, call):
    """The ``<name>.exit_reason = ...`` assignment on the line right before the call, on the same
    name the call passes, or None."""
    if i == 0 or not isinstance(stmts[i - 1], ast.Assign) or len(stmts[i - 1].targets) != 1:
        return None
    target = stmts[i - 1].targets[0]
    if not (isinstance(target, ast.Attribute) and target.attr == "exit_reason"):
        return None
    if not (isinstance(target.value, ast.Name) and isinstance(call.args[0], ast.Name)):
        return None
    if target.value.id != call.args[0].id:
        return None
    return stmts[i - 1]


def _is_make_room_log_at(stmts, index):
    if index < 0:
        return False
    stmt = stmts[index]
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "_log_make_room_unload"
    )


def _in_timeout_handler_of_drive_resident(parents, node):
    in_timeout = False
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.ExceptHandler) and isinstance(cur.type, ast.Name) and cur.type.id == "TimeoutError":
            in_timeout = True
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return in_timeout and cur.name == "_drive_resident"
    return False




def test_W9_every_begin_unload_caller_is_one_argument_and_stamps_its_reason():
    """Rule: the reason travels on the resident (``<name>.exit_reason = ...`` on the line
    before the call), and every call stays one positional argument, so one-argument fakes keep working."""
    tree = ast.parse(MANAGER_SRC.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if _is_begin_call(n)]
    assert len(calls) == 8, f"expected 8 _begin_unload_locked call sites, found {len(calls)}"
    not_one_arg = [ast.unparse(c) for c in calls if not _is_one_arg_call(c)]
    assert not not_one_arg, f"every call must be one positional argument, no keywords: {not_one_arg}"
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    stamped, unstamped = [], []
    for call in calls:
        stmts, i = _block_of(tree, call)
        stamp = _stamp_for(stmts, i, call)
        if stamp is None:
            unstamped.append(call)
        else:
            stamped.append((call, stmts, i, stamp))
    assert len(stamped) == 7, f"expected 7 callers stamped on the line before the call, found {len(stamped)}"
    assert len(unstamped) == 1
    assert _in_timeout_handler_of_drive_resident(parents, unstamped[0]), (
        "the one unstamped caller must be the self idle-timeout in _drive_resident's "
        "except TimeoutError handler (pinned by tests/test_idle_unload_wedge.py)"
    )
    make_room = [(call, stmts, i, stamp) for call, stmts, i, stamp in stamped
                 if "MAKE_ROOM" in ast.unparse(stamp.value)]
    assert len(make_room) == 4, f"expected 4 make-room stamps, found {len(make_room)}"
    for call, stmts, i, stamp in make_room:
        assert _is_make_room_log_at(stmts, i - 2), (
            "every make-room stamp must come right after its _log_make_room_unload line: "
            f"{ast.unparse(stamp)}"
        )


@pytest.mark.asyncio
async def test_W10_a_failing_exit_line_is_swallowed_and_the_death_still_happens(
    tmp_path, caplog, monkeypatch,
):
    caplog.set_level(logging.DEBUG, logger="turbohaul.manager")

    def boom(**kw):
        raise ValueError("boom")

    monkeypatch.setattr(manager_mod, "format_resident_exit", boom, raising=False)
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w10-model", 59950)
    unloads_before = mgr._unload_count
    try:
        try:
            await mgr._reap_dead_resident(r, RuntimeError("x"))
        except Exception as exc:
            raise AssertionError("a failing exit line must not escape the death path: " + repr(exc)) from exc
        assert r.state is ResidentState.DEAD
        assert "w10-model" not in mgr._residents
        assert mgr._unload_count == unloads_before + 1
        failed = [
            rec for rec in caplog.records
            if rec.levelno == logging.DEBUG and "resident exit line failed" in rec.getMessage()
        ]
        assert len(failed) == 1, f"expected one DEBUG 'resident exit line failed' record, got {len(failed)}"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tip_thread", "slot_thread", "idle_thread", "expected"),
    [
        ("tip-thread", "slot-thread", "idle-thread", "tip-thread"),
        (None, "slot-thread", "idle-thread", "slot-thread"),
        (None, None, "idle-thread", "idle-thread"),
        (None, None, None, "-"),
    ],
    ids=["grace-tip", "active-slot", "idle-thread", "none"],
)
async def test_W11_the_exit_line_thread_falls_back_in_order(
    tmp_path, caplog, tip_thread, slot_thread, idle_thread, expected,
):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    assert hasattr(TurbohaulManager, "_log_resident_exit"), "the exit line must be written by _log_resident_exit"
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = Resident(model_tag="w11-model", resident_key="w11-model", state=ResidentState.ACTIVE)
    r.grace_tip = None if tip_thread is None else Slot.new("w11-model", prompt="x", thread_id=tip_thread)
    r.active_slot = None if slot_thread is None else Slot.new("w11-model", prompt="x", thread_id=slot_thread)
    r.idle_thread_id = idle_thread
    mgr._log_resident_exit(r, "driver_exit", ResidentState.ACTIVE)
    assert _exit_lines(caplog) == [
        f"RESIDENT_EXIT reason=driver_exit resident_key=w11-model model_tag=w11-model "
        f"thread={expected} port=- state_before=ACTIVE state_after=DEAD"
    ]


@pytest.mark.asyncio
async def test_W12a_reasonless_unload_of_an_active_resident_logs_unspecified(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w12a-model", 59951)
    try:
        async with mgr._registry_lock:
            mgr._begin_unload_locked(r)
        assert r.state is ResidentState.DEAD
        assert "w12a-model" not in mgr._residents
        assert _exit_lines(caplog) == [
            "RESIDENT_EXIT reason=unspecified resident_key=w12a-model model_tag=w12a-model "
            "thread=- port=- state_before=ACTIVE state_after=DEAD"
        ]
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W12b_reasonless_unload_of_an_idle_resident_logs_self_idle_timeout(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w12b-model", 59952, state=ResidentState.IDLE_EVICTABLE)
    try:
        async with mgr._registry_lock:
            mgr._begin_unload_locked(r)
        assert r.state is ResidentState.DEAD
        assert "w12b-model" not in mgr._residents
        assert _exit_lines(caplog) == [
            "RESIDENT_EXIT reason=self_idle_timeout resident_key=w12b-model model_tag=w12b-model "
            "thread=- port=- state_before=IDLE_EVICTABLE state_after=DEAD"
        ]
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W13_cap2_liveness_sweep_logs_idle_engine_dead_for_a_dead_idle_resident(
    tmp_path, caplog, monkeypatch,
):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    monkeypatch.setattr(manager_mod, "_MULTI_ENGINE_IDLE_LIVENESS_SWEEP_INTERVAL_S", 0.05)
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w13-model", 59950, state=ResidentState.IDLE_EVICTABLE)
    dead = MagicMock()
    dead.port = 59950
    dead.pid = 4242
    dead.is_alive.return_value = False
    r.handle = dead
    sweep = asyncio.create_task(mgr._idle_engine_liveness_sweep())
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and mgr._residents.get("w13-model") is r:
            await asyncio.sleep(0.02)
        assert mgr._residents.get("w13-model") is None, (
            "the liveness sweep did not deregister the dead idle resident within 5s"
        )
        assert r.state is ResidentState.DEAD
        assert _exit_lines(caplog) == [
            "RESIDENT_EXIT reason=idle_engine_dead resident_key=w13-model model_tag=w13-model "
            "thread=- port=59950 state_before=IDLE_EVICTABLE state_after=DEAD"
        ]
    finally:
        mgr._stop_event.set()
        sweep.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweep
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W12c_a_stamp_wins_over_the_state_resolution(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="turbohaul.manager")
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w12c-model", 59953, state=ResidentState.IDLE_EVICTABLE)
    try:
        r.exit_reason = "driver_reap"
        async with mgr._registry_lock:
            mgr._begin_unload_locked(r)
        assert r.state is ResidentState.DEAD
        assert _exit_lines(caplog) == [
            "RESIDENT_EXIT reason=driver_reap resident_key=w12c-model model_tag=w12c-model "
            "thread=- port=- state_before=IDLE_EVICTABLE state_after=DEAD"
        ]
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_W14_a_one_argument_fake_of_begin_unload_locked_still_receives_the_reap(
    tmp_path, monkeypatch,
):
    """Many existing tests replace the method with one-argument fakes, so the
    supervisor's reap must still reach it with exactly one argument and must not raise."""
    mgr = _manager(tmp_path, idle_hot_load_seconds=0)
    r = _resident(mgr, "w14-model", 59954)
    calls = []
    monkeypatch.setattr(mgr, "_begin_unload_locked", lambda rr: calls.append(rr))
    try:
        await mgr._reap_dead_resident(r)
        assert len(calls) == 1, f"the fake must be called exactly once, got {len(calls)} calls"
        assert calls[0] is r
    finally:
        await mgr.shutdown()
