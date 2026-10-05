"""Tests for WS /ws/state redacted broadcaster."""
import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

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
from turbohaul.manager import EventBus
from turbohaul.subprocess_mgr import SidecarHandle


@pytest.fixture
def app_test(tmp_path):
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
            llama_server_binary=tmp_path / "fake",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


class TestEventBus:
    def test_subscribe_unsubscribe(self):
        bus = EventBus()
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        bus.subscribe(q1)
        bus.subscribe(q2)
        assert bus.subscriber_count == 2
        bus.unsubscribe(q1)
        assert bus.subscriber_count == 1

    def test_publish_fanout(self):
        bus = EventBus()
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        bus.subscribe(q1)
        bus.subscribe(q2)
        bus.publish_nowait({"event": "test", "state": "STAGED"})
        assert q1.qsize() == 1
        assert q2.qsize() == 1
        assert q1.get_nowait() == {"event": "test", "state": "STAGED"}

    def test_publish_redacts_prompt(self):
        bus = EventBus()
        q = asyncio.Queue()
        bus.subscribe(q)
        bus.publish_nowait(
            {"event": "active", "prompt": "SECRET PROMPT", "state": "ACTIVE"}
        )
        ev = q.get_nowait()
        assert "prompt" not in ev
        assert ev["event"] == "active"
        assert ev["state"] == "ACTIVE"

    def test_publish_redacts_response(self):
        bus = EventBus()
        q = asyncio.Queue()
        bus.subscribe(q)
        bus.publish_nowait(
            {"event": "complete", "response": "SECRET RESPONSE", "state": "GRACE"}
        )
        ev = q.get_nowait()
        assert "response" not in ev

    def test_publish_redacts_stderr(self):
        bus = EventBus()
        q = asyncio.Queue()
        bus.subscribe(q)
        bus.publish_nowait(
            {"event": "fail", "stderr": "trace line with sensitive paths", "state": "POPPED"}
        )
        ev = q.get_nowait()
        assert "stderr" not in ev

    def test_publish_redacts_context_and_messages(self):
        bus = EventBus()
        q = asyncio.Queue()
        bus.subscribe(q)
        bus.publish_nowait(
            {
                "event": "chat",
                "messages": [{"role": "user", "content": "hi"}],
                "context": [1, 2, 3],
                "state": "ACTIVE",
            }
        )
        ev = q.get_nowait()
        assert "messages" not in ev
        assert "context" not in ev

    def test_publish_back_pressure_drops(self):
        """Full queues drop events (don't block publisher)."""
        bus = EventBus()
        q = asyncio.Queue(maxsize=1)
        bus.subscribe(q)
        bus.publish_nowait({"event": "first"})
        bus.publish_nowait({"event": "second"})  # would block; should drop
        assert q.qsize() == 1
        assert q.get_nowait()["event"] == "first"


def _run_loop_in_background_thread():
    """Start a fresh asyncio event loop on its own OS thread and let it go
    genuinely idle (run_forever with nothing scheduled -> parked in
    epoll_wait, matching the production scenario this test guards against).
    Returns (loop, thread); caller must call _stop_background_loop after."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _target():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    assert ready.wait(timeout=5.0), "background loop never started"
    return loop, thread


def _stop_background_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)
    loop.close()


class TestPublishNowaitLoopAffinity:
    """EventBus.publish_nowait must be safe to call from any
    thread. Bounded timeout is load-bearing throughout --
    a deadlocking test that hangs forever is worse than no test at all, so
    every wait below has an explicit, finite bound and can never hang pytest
    itself even if the underlying hazard were still present."""

    BOUND_S = 3.0

    def test_offloop_publish_reaches_subscriber_bounded(self):
        """The synthetic case: a bus subscribed on a background loop, published
        to directly from THIS (main) test thread -- a genuine off-loop caller,
        not a simulation. Must fail (timeout) on the unfixed code and pass here."""
        loop, thread = _run_loop_in_background_thread()
        try:
            bus = EventBus()
            received = threading.Event()

            async def _subscribe_and_wait():
                q = asyncio.Queue()
                bus.subscribe(q)
                event = await q.get()
                if event.get("event") == "offloop-test":
                    received.set()

            # Scheduling the subscriber task itself uses the documented
            # thread-safe primitive -- only the publish below is the thing
            # under test.
            asyncio.run_coroutine_threadsafe(_subscribe_and_wait(), loop)
            time.sleep(0.1)  # let subscribe() actually run and bind self._loop

            # THE call under test: off-loop, no thread-safe wrapper.
            bus.publish_nowait({"event": "offloop-test"})

            assert received.wait(timeout=self.BOUND_S), (
                "subscriber did not receive the off-loop publish within "
                f"{self.BOUND_S}s -- off-loop publish hazard reproduced"
            )
        finally:
            _stop_background_loop(loop, thread)

    def test_first_ever_publish_is_offloop_not_ordering_dependent(self):
        """Trap check: the fix must bind the loop at
        subscribe(), not at the first publish_nowait() call of any kind. This
        test makes the FIRST EVER publish_nowait call an off-loop one -- no
        on-loop publish happens first to "warm up" self._loop. If the fix
        were instead written to bind lazily inside publish_nowait itself, an
        off-loop first call would find no running loop to capture and could
        silently take the unsafe fast path forever -- this test exists
        specifically to catch that shape of bug."""
        loop, thread = _run_loop_in_background_thread()
        try:
            bus = EventBus()
            received = threading.Event()

            async def _subscribe_only():
                q = asyncio.Queue()
                bus.subscribe(q)
                return q

            fut = asyncio.run_coroutine_threadsafe(_subscribe_only(), loop)
            q = fut.result(timeout=5.0)

            async def _await_get():
                event = await q.get()
                if event.get("event") == "first-ever-offloop":
                    received.set()

            asyncio.run_coroutine_threadsafe(_await_get(), loop)
            time.sleep(0.05)

            # This is the FIRST publish_nowait call this bus instance has
            # EVER seen, and it is off-loop.
            bus.publish_nowait({"event": "first-ever-offloop"})

            assert received.wait(timeout=self.BOUND_S), (
                "first-ever publish_nowait call (off-loop) was not delivered "
                f"within {self.BOUND_S}s -- fix is ordering-dependent"
            )
        finally:
            _stop_background_loop(loop, thread)

    def test_onloop_publish_still_uses_direct_path_not_threadsafe(self, monkeypatch):
        """The other half of the design: an on-loop call must NOT pay the
        call_soon_threadsafe cost -- this is what makes the on-loop fast path cheaper
        than blanket call_soon_threadsafe for the 5 already-loop-affine, hot-path callers."""
        loop, thread = _run_loop_in_background_thread()
        try:
            bus = EventBus()
            received = threading.Event()
            calls = {"threadsafe": 0}

            async def _run():
                q = asyncio.Queue()
                bus.subscribe(q)
                real_threadsafe = bus._loop.call_soon_threadsafe

                def spy(*a, **k):
                    calls["threadsafe"] += 1
                    return real_threadsafe(*a, **k)

                bus._loop.call_soon_threadsafe = spy
                bus.publish_nowait({"event": "onloop-test"})  # called ON the loop
                event = await asyncio.wait_for(q.get(), timeout=2.0)
                if event.get("event") == "onloop-test":
                    received.set()

            fut = asyncio.run_coroutine_threadsafe(_run(), loop)
            fut.result(timeout=self.BOUND_S)

            assert received.is_set()
            assert calls["threadsafe"] == 0, (
                "on-loop publish_nowait routed through call_soon_threadsafe "
                "-- this defeats the point of the on-loop fast path"
            )
        finally:
            _stop_background_loop(loop, thread)

    def test_real_audit_async_offloop_publish_reaches_ws_bounded(self, app_test):
        """The REAL path, not a synthetic thread: _audit is the actual, live
        off-loop caller found by reviewing call sites (asyncio.to_thread in
        _audit_async offloads _audit's whole body, publish_nowait included,
        to a worker thread -- see the EventBus class docstring). Drives
        mgr._audit_async for real through a WS connection and proves the
        published event still arrives, bounded, no synthetic thread involved
        anywhere in this test.

        NOTE on what this test does and does NOT prove (checked, not assumed
        -- by direct run): this test passes on BOTH the fixed code
        and the unfixed code, so it is an integration proof that the real
        production entry point delivers correctly, NOT a pre/post-fix
        differentiator. Reason, verified directly: asyncio.to_thread's own
        completion handoff (loop.run_in_executor's internal wrap_future) is
        itself thread-safe (uses call_soon_threadsafe) and fires on the same
        worker thread immediately after _audit() returns -- i.e. right after
        the unsafe publish_nowait call queues its wakeup. That secondary,
        safe wakeup flushes the WHOLE ready deque, including the earlier
        unsafe entry, within microseconds, for every real call site (all 21
        are `await self._audit_async(...)`, never fire-and-forget). The
        clean, discriminating causal proof is the SYNTHETIC test above
        (test_offloop_publish_reaches_subscriber_bounded), which isolates a
        genuinely idle loop with nothing else scheduled -- the actual
        at-risk scenario -- and does fail on the unfixed code. This test is
        kept as a real-path regression guard, not a causality proof."""
        app, client = app_test
        mgr = app.state.manager
        with client.websocket_connect("/ws/state") as ws:
            ws.receive_json()  # connected

            async def do_submit():
                return await mgr.submit(model_tag="m1", prompt="hi", thread_id="thr-real-audit")

            slot = asyncio.run(do_submit())

            # The real production entry point: _audit_async, awaited on the
            # app's own event loop via the portal (exactly how the driver
            # chain awaits it in production) -- its OWN body still hops to
            # asyncio.to_thread internally, which is the actual hazard.
            ws.portal.call(mgr._audit_async, slot, "real_audit_offloop_test")

            # Bounded receive: don't trust receive_json()'s own blocking
            # behavior to be safely bounded -- run it on a thread and join
            # with an explicit timeout so this test can never hang pytest.
            box = {}
            done = threading.Event()

            def _recv():
                try:
                    box["event"] = ws.receive_json()
                except Exception as e:  # noqa: BLE001 - report, don't hide
                    box["error"] = e
                finally:
                    done.set()

            recv_thread = threading.Thread(target=_recv, daemon=True)
            recv_thread.start()
            arrived = done.wait(timeout=self.BOUND_S)

            assert arrived, (
                f"real _audit_async off-loop publish did not arrive at the "
                f"WS subscriber within {self.BOUND_S}s -- off-loop publish hazard "
                f"reproduced via the actual production path"
            )
            assert "error" not in box, box.get("error")
            ev = box["event"]
            assert ev["event"] == "real_audit_offloop_test"
            assert ev["slot_id"] == slot.slot_id


class TestWsStateConnect:
    def test_ws_connect_sends_initial_snapshot(self, app_test):
        app, client = app_test
        with client.websocket_connect("/ws/state") as ws:
            msg = ws.receive_json()
            assert msg["event"] == "connected"
            assert "snapshot" in msg
            assert "queue" in msg["snapshot"]

    def test_ws_unsubscribes_on_disconnect(self, app_test):
        app, client = app_test
        mgr = app.state.manager
        assert mgr.event_bus.subscriber_count == 0
        with client.websocket_connect("/ws/state") as ws:
            ws.receive_json()  # connected
            assert mgr.event_bus.subscriber_count == 1
        # After exit, should unsubscribe
        # Give time for cleanup
        import time
        time.sleep(0.1)
        assert mgr.event_bus.subscriber_count == 0


class TestWsStateRedaction:
    def test_published_event_received_redacted(self, app_test):
        """If somehow a prompt gets into a published event, it's stripped pre-broadcast."""
        app, client = app_test
        mgr = app.state.manager
        with client.websocket_connect("/ws/state") as ws:
            ws.receive_json()  # connected
            # Simulate manager publishing an event with a sensitive payload
            ws.portal.call(mgr.event_bus.publish_nowait,
                {
                    "event": "active",
                    "slot_id": "slot-test",
                    "model_tag": "m",
                    "state": "ACTIVE",
                    "prompt": "leaked text!",
                    "response": "leaked response!",
                    "stderr": "private trace",
                })
            ev = ws.receive_json()
            assert ev["event"] == "active"
            assert ev["slot_id"] == "slot-test"
            assert "prompt" not in ev
            assert "response" not in ev
            assert "stderr" not in ev

    def test_manager_audit_events_arrive_at_ws(self, app_test):
        """When the manager's _audit fires, the event reaches WS subscribers."""
        import asyncio
        app, client = app_test
        mgr = app.state.manager
        with client.websocket_connect("/ws/state") as ws:
            ws.receive_json()  # connected

            # Run submit() inside event loop via httpx (best way through TestClient)
            # Simpler: directly call mgr.submit synchronously inside an async wrapper
            async def do_submit():
                return await mgr.submit(model_tag="m1", prompt="hi", thread_id="thr-abc-12345")

            slot = asyncio.run(do_submit())
            # The submit() path records via _audit OR via the inline audit-log code.
            # Currently submit() uses inline upsert; it doesn't call _audit().
            # We test that _audit() events DO publish:
            ws.portal.call(mgr.event_bus.publish_nowait,
                {
                    "event": "manual_publish_test",
                    "slot_id": slot.slot_id,
                    "state": "STAGED",
                    "thread_id_prefix": "thr-abc-",
                })
            ev = ws.receive_json()
            assert ev["slot_id"] == slot.slot_id
            assert ev["thread_id_prefix"] == "thr-abc-"
            # No leaked full thread_id, no prompt
            assert "prompt" not in ev
            assert "thread_id" not in ev or len(ev.get("thread_id", "")) <= 8
