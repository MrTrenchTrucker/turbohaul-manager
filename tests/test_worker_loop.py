"""Integration tests for the full worker_loop FSM cycle ().

Uses DI to inject mocks for spawn / health / sigterm / vram / complete so no real
llama-server is spawned. Phase 6 smoke E2E uses the real backend.
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

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
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(tmp_path, grace_seconds=0, idle_hot_load_seconds=0):
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


@pytest.mark.asyncio
class TestWorkerLoopFullCycle:
    async def test_full_cycle_happy_path(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)
        spawn_calls = []
        sigterm_calls = []
        vram_calls = []
        complete_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append({"model_tag": model_tag, "port": port, "argv": argv})
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, *, drained_window_s, is_active, **kwargs):
            sigterm_calls.append({"model_tag": handle.model_tag, "is_active": is_active})
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            vram_calls.append(kwargs)
            return True, 100

        async def fake_complete(slot, handle):
            complete_calls.append(slot.slot_id)

        mgr = TurbohaulManager(
            boot,
            runtime,
            spawn_fn=fake_spawn,
            health_fn=fake_health,
            sigterm_fn=fake_sigterm,
            vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        slot = await mgr.submit(model_tag="m1", prompt="hi")
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        # Allow enough time for: pop → load → active → complete → grace(0s) → pop → idle
        await asyncio.sleep(0.5)
        await mgr.shutdown()

        assert len(spawn_calls) == 1
        assert spawn_calls[0]["model_tag"] == "m1"
        assert spawn_calls[0]["port"] == boot.runtime.default_port_base
        assert len(complete_calls) == 1
        assert complete_calls[0] == slot.slot_id
        assert len(sigterm_calls) == 1
        assert len(vram_calls) == 1

        # Verify slot ended COLD via teardown
        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute(
            "SELECT state, end_reason FROM slots WHERE slot_id=?", (slot.slot_id,)
        )
        row = cur.fetchone()
        assert row["state"] == "COLD"
        assert "grace-expired" in row["end_reason"]
        conn.close()

    async def test_full_cycle_records_fsm_transitions(self, tmp_path):
        """Verify the FSM transition events land in audit_events table."""
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            pass

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot = await mgr.submit(model_tag="m1", prompt="hi")
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.5)
        await mgr.shutdown()

        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute(
            "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
            (slot.slot_id,),
        )
        events = [r["event_type"] for r in cur.fetchall()]
        conn.close()

        assert "submit" in events
        assert "stage_to_loading" in events
        assert "active" in events
        assert "grace_enter" in events
        assert "teardown" in events
        assert "idle_hot_enter" in events

    async def test_loading_fail_health_timeout_pops(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health_timeout(port, timeout_s, **kwargs):
            return False  # never healthy

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(**kwargs):
            return True, 100

        async def fake_complete(slot, handle):
            pass  # never reached

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health_timeout,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        slot = await mgr.submit(model_tag="m1", prompt="hi")
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.5)
        await mgr.shutdown()

        conn = open_state_db(boot.storage.state_db_path)
        cur = conn.execute(
            "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
            (slot.slot_id,),
        )
        events = [r["event_type"] for r in cur.fetchall()]
        assert "loading_fail_health_timeout" in events

        cur2 = conn.execute("SELECT state, end_reason FROM slots WHERE slot_id=?", (slot.slot_id,))
        row = cur2.fetchone()
        assert row["state"] == "COLD"
        assert "loading-fail" in row["end_reason"]
        conn.close()

    async def test_two_slots_processed_sequentially(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)
        spawn_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            await asyncio.sleep(0.01)

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        s1 = await mgr.submit(model_tag="m1", prompt="first")
        s2 = await mgr.submit(model_tag="m2", prompt="second")
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.8)
        await mgr.shutdown()

        assert spawn_calls == ["m1", "m2"]  # FIFO order

    async def test_worker_loop_exits_on_shutdown(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        mgr = TurbohaulManager(boot, runtime)
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        await asyncio.sleep(0.1)
        await mgr.shutdown()
        assert mgr._worker_task.done() or mgr._worker_task.cancelled()


@pytest.mark.asyncio
class TestIdleHotWire:
    """Note: idle-hot warm-hold + model-swap + expiry ."""

    async def test_grace_expiry_holds_warm_idle_when_idle_seconds_gt0(self, tmp_path):
        """After grace expires WITHOUT match, _idle_handle is held + sigterm NOT called yet."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=120
        )
        spawn_call_count = [0]
        sigterm_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_call_count[0] += 1
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            sigterm_calls.append(handle.model_tag)
            return True, "sigterm-clean"

        async def fake_vram(*a, **kw):
            return True, None

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            await mgr.submit_and_wait("gpt-x", "prompt-1", thread_id="t1")
            # Wait for grace expiry to enter idle-hold
            await asyncio.sleep(0.4)
        finally:
            await mgr.shutdown()

        assert spawn_call_count[0] == 1
        # The sidecar SHOULD have been torn down on shutdown (not before)
        assert sigterm_calls == ["gpt-x"]

    async def test_warm_inherit_same_model_skips_spawn(self, tmp_path):
        """Second request for SAME model_tag inherits the warm handle."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=120
        )
        spawn_call_count = [0]

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_call_count[0] += 1
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            return True, "sigterm-clean"

        async def fake_vram(*a, **kw):
            return True, None

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            await mgr.submit_and_wait("gpt-x", "prompt-1", thread_id="t1")
            await asyncio.sleep(0.1)  # let grace expire + idle hold
            await mgr.submit_and_wait("gpt-x", "prompt-2", thread_id="t2")
        finally:
            await mgr.shutdown()

        # Only ONE spawn call (second slot inherited warm handle)
        assert spawn_call_count[0] == 1

    async def test_different_model_tears_down_idle_then_spawns(self, tmp_path):
        """Second request for DIFFERENT model_tag tears down idle holder first."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=120
        )
        spawn_calls = []
        sigterm_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            sigterm_calls.append(handle.model_tag)
            return True, "sigterm-clean"

        async def fake_vram(*a, **kw):
            return True, None

        async def fake_complete(slot, handle):
            return {"ok": True}

        # H-4 polish: seed manifests so manifest_found=True keeps the
        # holder-at-risk bail-fast path inactive; tests the actual swap.
        manifests_dir = boot.storage.manifests_path
        manifests_dir.mkdir(parents=True, exist_ok=True)
        for tag in ("gpt-x", "gpt-y"):
            (manifests_dir / f"{tag}.yaml").write_text(
                f"model_tag: {tag}\n"
                "gguf_blob_sha256: " + "a" * 64 + "\n"
                "context_size: 2048\n"
                "expected_vram_bytes: 0\n"
                "llama_server_flags: {}\n"
            )

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            await mgr.submit_and_wait("gpt-x", "prompt-1", thread_id="t1")
            await asyncio.sleep(0.1)  # idle-hold gpt-x
            await mgr.submit_and_wait("gpt-y", "prompt-2", thread_id="t2")
        finally:
            await mgr.shutdown()

        # Two spawn calls (gpt-x then gpt-y)
        assert spawn_calls == ["gpt-x", "gpt-y"]
        # gpt-x sigterm fired (model swap teardown)
        assert "gpt-x" in sigterm_calls
        # gpt-y also sigterm at shutdown
        assert sigterm_calls.count("gpt-y") >= 1
    async def test_bogus_model_tag_preserves_idle_holder(self, tmp_path):
        """Review note polish : bogus model_tag must NOT tear down idle holder."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=120,
        )
        spawn_calls = []
        sigterm_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            return _make_fake_handle(model_tag, port)

        async def fake_health(port, timeout_s, **kwargs):
            return True

        async def fake_sigterm(handle, **kwargs):
            sigterm_calls.append(handle.model_tag)
            return True, "sigterm-clean"

        async def fake_vram(*a, **kw):
            return True, None

        async def fake_complete(slot, handle):
            return {"ok": True}

        # Pre-seed a manifest for "real-model" so it can spawn cleanly
        manifests_dir = boot.storage.manifests_path
        manifests_dir.mkdir(parents=True, exist_ok=True)
        (manifests_dir / "real-model.yaml").write_text(
            "model_tag: real-model\n"
            "gguf_blob_sha256: " + "a" * 64 + "\n"
            "display_name: \"Real Model\"\n"
            "description: test\n"
            "context_size: 2048\n"
            "expected_vram_bytes: 0\n"
            "llama_server_flags: {}\n"
        )

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        # Force safety_enabled=False to isolate the manifest-not-found check
        mgr.runtime.queue.safety_enabled = False
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            # 1st request: real-model — fills the warm idle holder post-grace
            await mgr.submit_and_wait("real-model", "prompt", thread_id="t1")
            await asyncio.sleep(0.2)  # let grace expire + idle hold
            assert mgr._idle_handle is not None, "idle holder should exist"
            assert mgr._idle_model_tag == "real-model"

            # 2nd request: BOGUS model_tag (no manifest) — must fail fast
            # WITHOUT tearing down the idle holder.
            with pytest.raises(RuntimeError, match="no manifest"):
                await mgr.submit_and_wait(
                    "qwen-pretend", "prompt-bogus", thread_id="t2",
                )
            # Holder must STILL be the real-model warm sidecar
            assert mgr._idle_handle is not None, (
                "idle holder was wrongly torn down on bogus-model fail"
            )
            assert mgr._idle_model_tag == "real-model"
            # No bogus-spawn fired (we bailed before spawn)
            assert "qwen-pretend" not in spawn_calls
            # No sigterm fired on the holder
            assert "real-model" not in sigterm_calls
        finally:
            await mgr.shutdown()


# ===========================================================================
# spec : ACTIVE_MATCH streaming warm-reuse + non-streaming regression
# ===========================================================================
#
# Regression context ( pre-de0602e):
#   The streaming ACTIVE_MATCH branch in worker_loop unconditionally called
#   _complete_fn for the matched slot, ignoring its stream=True flag. The
#   matched slot's stream_ready_event was never set; the route then waited
#   for SLOT_READY_TIMEOUT_S=600s before failing. Every turn ≥ 2 of a an agent
#   multi-tool agent loop hit this and hung for 10 minutes.
#
# These two tests pin the contract:
#   Case 1 — streaming ACTIVE_MATCH propagates handle + sets ready event,
#            and does NOT call _complete_fn (single-sidecar invariant).
#   Case 2 — non-streaming ACTIVE_MATCH still calls _complete_fn (would catch
#            a future fix that broke the non-streaming path).
#
# Mock fidelity (F1): we use the REAL asyncio.Event() instances that
# submit_for_streaming creates on the slot — not MagicMock(spec=Event). A
# Mock .set() returns silently regardless of state, which would mask the
# very regression we're guarding against.


def _seed_manifest(boot, model_tag: str) -> None:
    """Pre-seed a manifest so manifest_found=True keeps holder-at-risk inert."""
    manifests_dir = boot.storage.manifests_path
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{model_tag}.yaml").write_text(
        f"model_tag: {model_tag}\n"
        "gguf_blob_sha256: " + "a" * 64 + "\n"
        "context_size: 2048\n"
        "expected_vram_bytes: 0\n"
        "llama_server_flags: {}\n"
    )


@pytest.mark.asyncio
class TestActiveMatchStreaming:
    """spec  — streaming warm-reuse + non-streaming regression guard."""

    async def test_streaming_active_match_warm_reuse_passes_handle(self, tmp_path):
        """A streaming follow-up on the same (thread_id, model_tag) during the
        anchor's GRACE window must: (a) propagate the anchor's SidecarHandle to
        the matched slot via stream_handle, (b) set stream_ready_event so the
        route unblocks, (c) NOT call _complete_fn (would open a 2nd sidecar
        connection and break the single-slot invariant).

        Regression guarded: pre-de0602e the matched slot's stream_ready_event
        was never set; routes hung 600s on SLOT_READY_TIMEOUT_S.
        """
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=5, idle_hot_load_seconds=0,
        )
        _seed_manifest(boot, "m1")

        spawn_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(*a, **k):
            return True, None

        # complete_fn is a spy: must NOT be called on the streaming branches.
        complete_spy = MagicMock()

        async def fake_complete(slot, handle):
            complete_spy(slot.slot_id)
            return {"_should_not_be_called": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            # ── Anchor: first streaming submit on (m1, t1) ──
            anchor = await mgr.submit_for_streaming(
                model_tag="m1", prompt="anchor",
                thread_id="t1",
                client_meta={"kind": "openai-chat-completion-stream", "stream": True},
            )
            # Wait for worker to bring anchor to ACTIVE + set stream_ready_event.
            # The 5s wait_for cap is the F3 timeout; pre-fix this would have
            # waited 600s. Anything > 0.5s here is a CI red flag, see below.
            t0_anchor = time.monotonic()
            await asyncio.wait_for(anchor.stream_ready_event.wait(), timeout=5.0)
            assert anchor.stream_handle is not None, (
                "anchor stream_handle not set when stream_ready_event fired"
            )
            assert (time.monotonic() - t0_anchor) < 2.0, (
                "anchor took > 2s to reach stream_ready; CI flake or regression"
            )

            # Release the anchor: simulate the route signalling end-of-stream.
            # Worker now advances anchor ACTIVE → GRACE.
            anchor.stream_done_event.set()
            # Tiny breather so the worker enters the GRACE-loop before we submit
            # the matched follow-up.
            await asyncio.sleep(0.05)

            # ── Matched: second streaming submit on SAME (m1, t1) ──
            matched = await mgr.submit_for_streaming(
                model_tag="m1", prompt="matched-followup",
                thread_id="t1",
                client_meta={"kind": "openai-chat-completion-stream", "stream": True},
            )
            # F3 instrumented assertion: the ACTIVE_MATCH branch must set
            # stream_ready_event promptly. Pre-fix this never fired and the
            # wait below would have hit the 5s timeout = test fail = regression
            # caught. Tight 0.5s sub-bound decouples from CI flake (the only
            # work between anchor-release and matched-ready is one GRACE-loop
            # iteration + one transition pair + one event .set()).
            t0_matched = time.monotonic()
            await asyncio.wait_for(matched.stream_ready_event.wait(), timeout=5.0)
            t_set = time.monotonic()
            assert (t_set - t0_matched) < 0.5, (
                f"matched stream_ready took {t_set - t0_matched:.3f}s; "
                "expected < 0.5s — slow path / regression"
            )

            # Contract: anchor's handle was reused (single-slot invariant)
            assert matched.stream_handle is anchor.stream_handle, (
                "ACTIVE_MATCH must reuse anchor SidecarHandle; matched got a "
                "different handle which implies a second spawn"
            )
            # Contract: NO new spawn happened for the matched slot
            assert spawn_calls == ["m1"], (
                f"expected 1 spawn for the anchor only; got {spawn_calls}"
            )
            # Contract: _complete_fn was NOT called on the streaming path —
            # not for the anchor and not for the matched slot. The route owns
            # the upstream connection.
            assert complete_spy.call_count == 0, (
                f"_complete_fn invoked {complete_spy.call_count}x on streaming "
                "path; must remain 0 (single-sidecar invariant). Calls: "
                f"{complete_spy.call_args_list!r}"
            )

            # Release matched so worker can finish + shutdown cleanly.
            matched.stream_done_event.set()
        finally:
            await mgr.shutdown()

    async def test_active_match_non_streaming_still_completes_via_complete_fn(
        self, tmp_path,
    ):
        """Non-streaming ACTIVE_MATCH must still invoke _complete_fn for the
        matched slot. Regression guard against a future fix that accidentally
        routes non-streaming follow-ups through the streaming branch.
        """
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=5, idle_hot_load_seconds=0,
        )
        _seed_manifest(boot, "m1")

        spawn_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(*a, **k):
            return True, None

        complete_spy = MagicMock()

        async def fake_complete(slot, handle):
            complete_spy(slot.slot_id)
            return {"ok": True, "for_slot": slot.slot_id}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            # ── Anchor: first NON-streaming submit on (m1, t1) ──
            # submit() (not submit_and_wait) so we don't block this coroutine
            # while the worker enters its GRACE loop. We'll await the future
            # explicitly with a bounded timeout.
            anchor = await mgr.submit(
                model_tag="m1", prompt="anchor",
                thread_id="t1",
                client_meta={"kind": "openai-chat-completion"},
                wait_for_completion=True,
            )
            await asyncio.wait_for(anchor.completion_future, timeout=5.0)

            # Tiny breather to land in the GRACE-loop.
            await asyncio.sleep(0.05)

            # ── Matched: second NON-streaming submit on SAME (m1, t1) ──
            matched = await mgr.submit(
                model_tag="m1", prompt="matched-followup",
                thread_id="t1",
                client_meta={"kind": "openai-chat-completion"},
                wait_for_completion=True,
            )
            t0 = time.monotonic()
            result = await asyncio.wait_for(matched.completion_future, timeout=5.0)
            elapsed = time.monotonic() - t0
            assert elapsed < 2.0, (
                f"matched non-streaming took {elapsed:.3f}s; "
                "expected sub-second under ACTIVE_MATCH warm-reuse"
            )

            # Contract: _complete_fn was called for BOTH anchor and matched —
            # this is what makes ACTIVE_MATCH worth the optimization on the
            # non-streaming path.
            slot_ids_completed = [c.args[0] for c in complete_spy.call_args_list]
            assert anchor.slot_id in slot_ids_completed, (
                f"_complete_fn never called for anchor slot {anchor.slot_id}; "
                f"got calls for: {slot_ids_completed}"
            )
            assert matched.slot_id in slot_ids_completed, (
                f"_complete_fn never called for matched slot {matched.slot_id}; "
                "ACTIVE_MATCH non-streaming branch regression — A fix "
                "must not bleed into this code path. "
                f"Got calls for: {slot_ids_completed}"
            )
            # Contract: warm-reuse → only ONE spawn for both slots
            assert spawn_calls == ["m1"], (
                f"expected single spawn under ACTIVE_MATCH; got {spawn_calls}"
            )
            # Sanity: matched got the canned result
            assert result == {"ok": True, "for_slot": matched.slot_id}
        finally:
            await mgr.shutdown()

