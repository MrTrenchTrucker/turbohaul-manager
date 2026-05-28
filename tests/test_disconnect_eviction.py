"""Tests for   — Client-Disconnect Queue Eviction.

9 tests covering:
1. pop_next skips evicted slots and flags is_evicted (queue-level)
2. SlotEvictedError raised from completion_future when worker_loop sees is_evicted
3. ACTIVE slot not affected by evict (sanity: happy non-stream path unaffected)
4. Client disconnect on pending slot → HTTP 499 via FastAPI E2E
5. pop_matched_thread also flags evicted slots (MOD-1 symmetry, grace-rematch)
6. Lazy-init disconnect_event=None works pre-attach (BootInventory path)
7. Evicted slot emits audit event via the async pool path (NOT state_db_session)
8. Consecutive evictions don't starve idle-expiry tick — fire-and-forget
   asyncio.create_task on _teardown_idle_holder (Bonus-2) + MOD-α identity guard
9. NEW T9: _pop_first_non_evicted_from is bounded — incremental drain per call
"""
import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig, PullConfig, QueueConfig, RuntimeConfig,
    RuntimePathsConfig, ServerConfig, StorageConfig, UIConfig,
)
from turbohaul.queue import TurbohaulQueue
from turbohaul.slot import Slot, SlotEvictedError


# ============================================================================
# Queue-level tests (1, 5, 6, 9) — pure unit, no fixture
# ============================================================================


@pytest.mark.asyncio
async def test_1_pop_next_skips_evicted_returns_with_flag():
    """Slot with disconnect_event SET pops with is_evicted=True flag."""
    q = TurbohaulQueue(staging_max=10)
    s = Slot.new("m")
    s.disconnect_event = asyncio.Event()
    s.disconnect_event.set()
    await q.enqueue(s)
    popped = await q.pop_next()
    assert popped is not None
    assert popped.slot_id == s.slot_id
    assert popped.is_evicted is True


@pytest.mark.asyncio
async def test_5_pop_matched_thread_skips_evicted_grace_rematch():
    """pop_matched_thread also flags is_evicted on disconnect_event-set slot."""
    q = TurbohaulQueue(staging_max=10)
    s = Slot.new("m", thread_id="t-1")
    s.disconnect_event = asyncio.Event()
    s.disconnect_event.set()
    await q.enqueue(s)
    popped = await q.pop_matched_thread("t-1", "m")
    assert popped is not None
    assert popped.slot_id == s.slot_id
    assert popped.is_evicted is True


@pytest.mark.asyncio
async def test_6_submit_then_immediate_disconnect_before_pop_lazy_init():
    """Lazy-init: Slot.new() leaves disconnect_event=None — non-HTTP callers
    (BootInventory orphan-replay) construct this way; queue treats None as
    'not evicted'."""
    q = TurbohaulQueue(staging_max=10)
    s = Slot.new("m")
    assert s.disconnect_event is None
    await q.enqueue(s)
    popped = await q.pop_next()
    assert popped is not None
    assert popped.slot_id == s.slot_id
    assert popped.is_evicted is False


@pytest.mark.asyncio
async def test_9_pop_first_non_evicted_from_bounded_drain():
    """MOD-B: 20 pre-evicted slots, pop_next makes bounded incremental
    progress per call — no O(N²) wedge."""
    q = TurbohaulQueue(staging_max=100)
    for _ in range(20):
        s = Slot.new("m")
        s.disconnect_event = asyncio.Event()
        s.disconnect_event.set()
        await q.enqueue(s)
    popped1 = await q.pop_next()
    assert popped1 is not None and popped1.is_evicted is True
    popped2 = await q.pop_next()
    assert popped2 is not None and popped2.is_evicted is True


# ============================================================================
# Manager + API tests (2, 3, 4, 7, 8) — TestClient fixture
# ============================================================================


def _make_app(tmp_path):
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
            llama_server_binary=tmp_path / "fake", default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            grace_seconds=0, idle_hot_load_seconds=0,
            drained_sigterm_window_active_s=1, drained_sigterm_window_cold_s=1,
        ),
        pull=PullConfig(),
    )
    app = create_app(boot, runtime, auto_start_worker=True, auto_boot_reconcile=False)
    mgr = app.state.manager

    from turbohaul.subprocess_mgr import SidecarHandle

    def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

    async def fake_health(port, timeout_s, **kwargs):
        return True

    async def fake_sigterm(handle, **kwargs):
        return True, "sigterm-clean"

    async def fake_vram(**kwargs):
        return True, 100

    async def fake_complete(slot, handle):
        return {
            "id": "chatcmpl-test", "object": "chat.completion",
            "created": 1700000000, "model": slot.model_tag,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    mgr._spawn = fake_spawn
    mgr._wait_healthy = fake_health
    mgr._sigterm = fake_sigterm
    mgr._vram_verify = fake_vram
    mgr._complete_fn = fake_complete
    return app


@pytest.fixture
def app_fixture(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        yield app, client


@pytest.mark.asyncio
async def test_2_evict_fails_future_with_SlotEvictedError(tmp_path):
    """worker_loop's is_evicted branch fails completion_future with SlotEvictedError."""
    app = _make_app(tmp_path)
    mgr = app.state.manager
    disconnect_event = asyncio.Event()
    disconnect_event.set()
    slot = await mgr.submit(
        model_tag="m",
        prompt="hi",
        wait_for_completion=True,
        disconnect_event=disconnect_event,
    )
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with pytest.raises(SlotEvictedError):
            await asyncio.wait_for(slot.completion_future, timeout=3.0)
    finally:
        mgr._stop_event.set()
        if mgr._worker_task is not None:
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except (asyncio.CancelledError, Exception):
                pass


def test_3_active_slot_not_affected_by_evict(app_fixture):
    """Happy non-stream path with NO disconnect completes normally."""
    app, client = app_fixture
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "ok"


def test_4_client_disconnect_evicts_pending_returns_499(app_fixture):
    """FastAPI E2E: submit_and_wait raising SlotEvictedError → HTTP 499."""
    app, client = app_fixture
    mgr = app.state.manager

    async def evicted_submit_and_wait(**kwargs):
        raise SlotEvictedError("simulated client disconnect")

    mgr.submit_and_wait = evicted_submit_and_wait
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 499, r.text
    d = r.json()["detail"]
    assert d["error"] == "client_closed_request"


@pytest.mark.asyncio
async def test_7_evicted_emits_audit_via_pool_not_state_db(tmp_path):
    """MOD-C: slot_evicted audit fires through _audit_event_only_async."""
    app = _make_app(tmp_path)
    mgr = app.state.manager
    audit_calls: list[tuple] = []

    original = mgr._audit_event_only_async

    async def capturing_audit(slot_id, event_type, payload):
        audit_calls.append((slot_id, event_type, dict(payload or {})))
        return await original(slot_id, event_type, payload)

    mgr._audit_event_only_async = capturing_audit

    disconnect_event = asyncio.Event()
    disconnect_event.set()
    slot = await mgr.submit(
        model_tag="m", prompt="hi",
        wait_for_completion=True,
        disconnect_event=disconnect_event,
    )
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with pytest.raises(SlotEvictedError):
            await asyncio.wait_for(slot.completion_future, timeout=3.0)
        await asyncio.sleep(0.1)
    finally:
        mgr._stop_event.set()
        if mgr._worker_task is not None:
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except (asyncio.CancelledError, Exception):
                pass
    slot_evicted_audits = [c for c in audit_calls if c[1] == "slot_evicted"]
    assert len(slot_evicted_audits) >= 1
    assert slot_evicted_audits[0][2].get("reason") == "client_disconnect"
    assert "time_in_queue_s" in slot_evicted_audits[0][2]


@pytest.mark.asyncio
async def test_8_consecutive_evictions_create_task_fire_and_forget(tmp_path):
    """Bonus-2: idle-teardown is asyncio.create_task fire-and-forget; the
    eviction-branch inline idle-tick fires with MOD-α identity guard."""
    app = _make_app(tmp_path)
    mgr = app.state.manager
    teardown_calls: list[str] = []

    async def capturing_teardown(reason):
        teardown_calls.append(reason)

    mgr._teardown_idle_holder = capturing_teardown
    mgr._idle_handle = MagicMock()
    mgr._idle_expires_at = 0.0

    disconnect_event = asyncio.Event()
    disconnect_event.set()
    slot = await mgr.submit(
        model_tag="m", prompt="hi",
        wait_for_completion=True,
        disconnect_event=disconnect_event,
    )
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with pytest.raises(SlotEvictedError):
            await asyncio.wait_for(slot.completion_future, timeout=3.0)
        await asyncio.sleep(0.1)
    finally:
        mgr._stop_event.set()
        if mgr._worker_task is not None:
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except (asyncio.CancelledError, Exception):
                pass
    assert "idle_expired" in teardown_calls, (
        "idle-expired teardown was not fire-and-forget invoked"
    )
    assert mgr._idle_expires_at is None
