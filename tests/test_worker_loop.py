"""Integration tests for the full worker_loop FSM cycle.

Uses DI to inject mocks for spawn / health / sigterm / vram / complete so no real
llama-server is spawned. Phase 6 smoke E2E uses the real backend.
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

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
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import SlotState
from turbohaul.state import open_state_db
from turbohaul.subprocess_mgr import SidecarHandle


def _boot_runtime(
    tmp_path, grace_seconds=0, idle_hot_load_seconds=0, max_parallel_sidecars=1,
):
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
            safety_enabled=False,
            grace_seconds=grace_seconds,
            idle_hot_load_seconds=idle_hot_load_seconds,
            drained_sigterm_window_active_s=1,
            drained_sigterm_window_cold_s=1,
            loading_health_timeout_s=10,
            max_parallel_sidecars=max_parallel_sidecars,
        ),
        pull=PullConfig(),
    )
    return boot, runtime


def _make_fake_handle(model_tag: str, port: int) -> SidecarHandle:
    proc = MagicMock()
    proc.pid = 88_888
    proc.poll.return_value = None
    return SidecarHandle(proc=proc, port=port, model_tag=model_tag)


def _make_dead_handle(model_tag: str, port: int) -> SidecarHandle:
    """A handle whose child has ALREADY EXITED (poll() returns an exit code, so
    is_alive() is False). Drives the FSM-wedge fast-fail path."""
    proc = MagicMock()
    proc.pid = 88_889
    proc.poll.return_value = 1  # exited non-zero -> is_alive() is False
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
        both_spawned = asyncio.Event()

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            if len(spawn_calls) >= 2:
                both_spawned.set()
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
        # Wait on the actual condition (both spawns landed), not a fixed
        # sleep — the drained_sigterm_window_active_s=1 config above (plus
        # fixed startup latency) leaves under 0.4s of margin against a fixed
        # 0.8s sleep, so a wall-clock guess is inherently marginal here. The
        # timeout below is a failure bound, not the success condition.
        await asyncio.wait_for(both_spawned.wait(), timeout=5.0)
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
    """Idle-hot warm-hold + model-swap + expiry."""

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

    async def test_same_model_queued_stays_warm_with_idle_disabled(self, tmp_path):
        """Residency floor: two SAME-model requests (distinct
        threads) served back-to-back with idle disabled (idle_hot_load_seconds=0,
        i.e. the keep_alive=0 edge) must NOT teardown+respawn between them — the
        second warm-inherits because the residency floor keeps the resident model
        warm while a same-model request is queued.

        FAILS before the floor (2 spawns + a grace-expired sigterm BETWEEN the two
        = churn); PASSES after (1 spawn, the only sigterm is the final shutdown).
        The 1-slot invariant is preserved: exactly one live sidecar throughout."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=0
        )
        spawn_calls = []
        sigterm_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(handle, **k):
            sigterm_calls.append(handle.model_tag)
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            await asyncio.sleep(0.01)
            return {"ok": True, "who": slot.prompt}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            # Two SAME-model requests, distinct threads, submitted together so the
            # 2nd is queued (head_model_tag == m1) when the 1st finishes grace.
            r1, r2 = await asyncio.gather(
                mgr.submit_and_wait("m1", "first", thread_id="t1"),
                mgr.submit_and_wait("m1", "second", thread_id="t2"),
            )
        finally:
            await mgr.shutdown()

        # Both requests served successfully.
        assert r1[1]["ok"] and r2[1]["ok"]
        # ONE spawn: the second warm-inherited (no swap/respawn).
        assert spawn_calls == ["m1"]
        # No mid-stream churn: the only teardown is the final shutdown.
        assert sigterm_calls.count("m1") == 1
        # 1-slot invariant preserved: never more than one live sidecar.
        assert mgr.runtime.queue.max_parallel_sidecars == 1

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
        """Bogus model_tag must NOT tear down idle holder."""
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
# ACTIVE_MATCH streaming warm-reuse + non-streaming regression
# ===========================================================================
#
# Regression context:
#   The streaming ACTIVE_MATCH branch in worker_loop unconditionally called
#   _complete_fn for the matched slot, ignoring its stream=True flag. The
#   matched slot's stream_ready_event was never set; the route then waited
#   for SLOT_READY_TIMEOUT_S=600s before failing. Every turn ≥ 2 of an AI harness
#   multi-tool agent loop hit this and hung for 10 minutes.
#
# These two tests pin the contract:
#   Case 1 — streaming ACTIVE_MATCH propagates handle + sets ready event,
#            and does NOT call _complete_fn (single-sidecar invariant).
#   Case 2 — non-streaming ACTIVE_MATCH still calls _complete_fn (would catch
#            a future fix that broke the non-streaming path).
#
# Mock fidelity: we use the REAL asyncio.Event() instances that
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
    """Streaming warm-reuse + non-streaming regression guard."""

    async def test_streaming_active_match_warm_reuse_passes_handle(self, tmp_path):
        """A streaming follow-up on the same (thread_id, model_tag) during the
        anchor's GRACE window must: (a) propagate the anchor's SidecarHandle to
        the matched slot via stream_handle, (b) set stream_ready_event so the
        route unblocks, (c) NOT call _complete_fn (would open a 2nd sidecar
        connection and break the single-slot invariant).

        Regression guarded: before the fix the matched slot's stream_ready_event
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
            # The 5s wait_for cap is the failure timeout; pre-fix this would have
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
            # Instrumented assertion: the ACTIVE_MATCH branch must set
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
                "ACTIVE_MATCH non-streaming branch regression — the streaming fix "
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


@pytest.mark.asyncio
class TestKeepAliveActiveMatchGuard:
    """cap<=1 path (_process_slot's ACTIVE_MATCH
    promotion). A request that omits keep_alive_s must not
    overwrite an already-stored EXPLICIT keep_alive intent from the anchor
    (or a prior matched follow-up) — last-EXPLICIT-writer-wins, not
    last-writer-wins. Explicit values must still win (regression guard).
    """

    async def _mgr_with_worker(self, tmp_path, grace_seconds=5):
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=grace_seconds)
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

        async def fake_complete(slot, handle):
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        return mgr, spawn_calls

    async def test_omitted_keep_alive_preserves_anchor_intent(self, tmp_path):
        """Anchor sends keep_alive_s=0 (explicit). A same-thread ACTIVE_MATCH
        follow-up with NO keep_alive_s in client_meta must leave the stored
        intent at 0, not reset it to None. Pre-fix: RED (unconditional
        overwrite clobbers 0 -> None)."""
        mgr, spawn_calls = await self._mgr_with_worker(tmp_path)
        try:
            anchor = await mgr.submit(
                model_tag="m1", prompt="anchor", thread_id="t1",
                client_meta={"kind": "openai-chat-completion", "keep_alive_s": 0},
                wait_for_completion=True,
            )
            await asyncio.wait_for(anchor.completion_future, timeout=5.0)
            assert mgr._latest_keep_alive_s == 0, (
                "anchor's explicit keep_alive=0 must be captured"
            )
            await asyncio.sleep(0.05)  # breather: let the GRACE loop start polling

            matched = await mgr.submit(
                model_tag="m1", prompt="followup", thread_id="t1",
                client_meta={"kind": "openai-chat-completion"},  # no keep_alive_s
                wait_for_completion=True,
            )
            await asyncio.wait_for(matched.completion_future, timeout=5.0)

            assert mgr._latest_keep_alive_s == 0, (
                "ACTIVE_MATCH follow-up with omitted keep_alive_s must NOT reset "
                f"the anchor's explicit intent; got {mgr._latest_keep_alive_s!r}"
            )
            assert spawn_calls == ["m1"], (
                f"expected warm ACTIVE_MATCH reuse, no second spawn; got {spawn_calls}"
            )
        finally:
            await mgr.shutdown()

    async def test_explicit_keep_alive_still_overwrites(self, tmp_path):
        """Regression guard: an EXPLICIT follow-up keep_alive_s must still win
        over the anchor's. Already-correct pre-fix; must stay correct post-fix."""
        mgr, spawn_calls = await self._mgr_with_worker(tmp_path)
        try:
            anchor = await mgr.submit(
                model_tag="m1", prompt="anchor", thread_id="t1",
                client_meta={"kind": "openai-chat-completion", "keep_alive_s": 0},
                wait_for_completion=True,
            )
            await asyncio.wait_for(anchor.completion_future, timeout=5.0)
            assert mgr._latest_keep_alive_s == 0
            await asyncio.sleep(0.05)

            matched = await mgr.submit(
                model_tag="m1", prompt="followup", thread_id="t1",
                client_meta={"kind": "openai-chat-completion", "keep_alive_s": 600},
                wait_for_completion=True,
            )
            await asyncio.wait_for(matched.completion_future, timeout=5.0)

            assert mgr._latest_keep_alive_s == 600, (
                "explicit follow-up keep_alive_s=600 must overwrite the anchor's 0; "
                f"got {mgr._latest_keep_alive_s!r}"
            )
        finally:
            await mgr.shutdown()

    async def test_new_anchor_does_not_inherit_stale_keep_alive(self, tmp_path):
        """Entry-clear regression guard (untouched by this fix): a brand
        new anchor cycle on a DIFFERENT thread, itself omitting keep_alive_s,
        must not inherit a stale nonzero value left behind by a fully-popped
        prior chain."""
        mgr, spawn_calls = await self._mgr_with_worker(tmp_path, grace_seconds=0)
        try:
            chain_a = await mgr.submit(
                model_tag="m1", prompt="chain-a", thread_id="t1",
                client_meta={"kind": "openai-chat-completion", "keep_alive_s": 300},
                wait_for_completion=True,
            )
            await asyncio.wait_for(chain_a.completion_future, timeout=5.0)
            # grace_seconds=0 -> chain A's GRACE loop exits immediately and
            # clears _latest_keep_alive_s (POPPED) before chain B.
            for _ in range(100):
                if mgr._latest_keep_alive_s is None:
                    break
                await asyncio.sleep(0.02)
            assert mgr._latest_keep_alive_s is None, (
                "chain A's POPPED clear should have reset the scalar to None "
                "before chain B starts"
            )

            chain_b = await mgr.submit(
                model_tag="m1", prompt="chain-b", thread_id="t2",
                client_meta={"kind": "openai-chat-completion"},  # no keep_alive_s
                wait_for_completion=True,
            )
            await asyncio.wait_for(chain_b.completion_future, timeout=5.0)
            assert mgr._latest_keep_alive_s is None, (
                "chain B (no keep_alive_s) must not inherit chain A's stale 300; "
                f"got {mgr._latest_keep_alive_s!r}"
            )
        finally:
            await mgr.shutdown()


@pytest.mark.asyncio
class TestLoadingWedgeFix:
    """FSM-wedge fix: a model-load that CRASHES (child exits) must not pin the
    single slot in LOADING for the full loading_health_timeout_s. The REAL
    wait_until_healthy (NOT injected) receives handle.is_alive and bails fast, so
    the existing LOADING_FAIL->POPPED cleanup fires in ~one poll interval and the
    worker_loop stays free to drain the queue. Pre-fix this hung ~600s = the
    no-reply an operator saw.
    """

    async def test_dead_child_during_load_fails_fast_and_queue_drains(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path, grace_seconds=0)
        # A LONG timeout makes the wedge obvious: pre-fix this test would take 60s
        # per dead load (and blow the 5s wait_for guards); post-fix both fail in ms.
        runtime.queue.loading_health_timeout_s = 60

        spawn_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_calls.append(model_tag)
            return _make_dead_handle(model_tag, port)

        async def fake_sigterm(handle, **kwargs):
            return True, "already-gone"

        async def fake_vram(**kwargs):
            return True, 0

        async def fake_complete(slot, handle):  # never reached
            pass

        # health_fn intentionally NOT injected -> the REAL wait_until_healthy runs
        # so the manager is_alive wiring is exercised end-to-end. safety_enabled
        # off so we reach spawn deterministically (no manifest needed).
        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr.runtime.queue.safety_enabled = False

        s1 = await mgr.submit(model_tag="m1", prompt="hi", wait_for_completion=True)
        s2 = await mgr.submit(model_tag="m2", prompt="hi", wait_for_completion=True)
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        t0 = time.monotonic()
        try:
            with pytest.raises(RuntimeError, match="loading-fail"):
                await asyncio.wait_for(s1.completion_future, timeout=5.0)
            with pytest.raises(RuntimeError, match="loading-fail"):
                await asyncio.wait_for(s2.completion_future, timeout=5.0)
            elapsed = time.monotonic() - t0
            assert elapsed < 4.0, (
                f"dead-child loads took {elapsed:.2f}s; FSM-wedge fix not firing "
                "(expected sub-second per load, not loading_health_timeout_s)"
            )
            assert spawn_calls == ["m1", "m2"], (
                f"expected both crashed loads processed FIFO; got {spawn_calls}"
            )
        finally:
            await mgr.shutdown()

        conn = open_state_db(boot.storage.state_db_path)
        try:
            for s in (s1, s2):
                cur = conn.execute(
                    "SELECT event_type FROM audit_events WHERE slot_id=? ORDER BY event_id",
                    (s.slot_id,),
                )
                events = [r["event_type"] for r in cur.fetchall()]
                assert "loading_fail_health_timeout" in events
                cur2 = conn.execute(
                    "SELECT state, end_reason FROM slots WHERE slot_id=?", (s.slot_id,)
                )
                row = cur2.fetchone()
                assert row["state"] == "COLD"
                assert "loading-fail" in row["end_reason"]
        finally:
            conn.close()


@pytest.mark.asyncio
class TestWarmInheritIdentityIsolation:
    """The idle-hot warm-inherit block (same-model second request reusing the
    idle sidecar) must only ever adopt the idle holder's IDENTITY fields
    (session_id/role/is_* labels) on a genuine continuation signal -- a
    same-thread re-cue that lost its session_id, or an explicit same-session
    match -- and never on session-less absence alone (that admits any
    brand-new unrelated caller). It must also never adopt `messages`: the
    outbound source is slot.context (see test_complete_messages_source.py),
    a separate field this block does not touch."""

    async def _run_two_requests(
        self, tmp_path, *, first_thread, second_thread,
        first_client_meta, second_client_meta,
    ):
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=120
        )
        captured = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(handle, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            captured.append({
                "client_meta": dict(slot.client_meta or {}),
                "context": slot.context,
            })
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            await mgr.submit_and_wait(
                "gpt-x", "prompt-1", thread_id=first_thread,
                context=[{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
                client_meta=first_client_meta,
            )
            await asyncio.sleep(0.05)  # let the first slot enter idle-hot
            await mgr.submit_and_wait(
                "gpt-x", "prompt-2", thread_id=second_thread,
                context=[{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
                client_meta=second_client_meta,
            )
        finally:
            await mgr.shutdown()
        return captured

    async def test_substitution_and_isolation_sessionless_new_caller(self, tmp_path):
        """SUBSTITUTION + ISOLATION: a session-less request on a DIFFERENT
        thread than the idle holder is a brand-new unrelated caller. It must
        get its OWN prompt (context) -- reproducing what the original bug
        would have substituted -- and must NOT inherit the idle holder's
        session_id or role."""
        captured = await self._run_two_requests(
            tmp_path,
            first_thread="t1", second_thread="t2",  # DIFFERENT thread
            first_client_meta={
                "session_id": "s1", "role": "main",
                "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
            },
            second_client_meta={
                "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
            },
        )
        second = captured[1]
        # SUBSTITUTION: own prompt, not the idle holder's.
        assert second["context"] == [{"role": "user", "content": "SECOND_CALLER_CONTENT"}]
        # ISOLATION: no identity crossed over from the unrelated first caller.
        assert second["client_meta"].get("session_id") is None
        assert second["client_meta"].get("role") is None

    async def test_both_sessionless_different_threads_no_identity_adoption(self, tmp_path):
        """Two absent session ids must not compare equal: a session-less
        caller on a DIFFERENT thread than a session-less idle holder is an
        unrelated caller (thread mismatch also keeps the re-cue disjunct
        False) and must keep its own identity/classification fields, not
        adopt the idle holder's."""
        captured = await self._run_two_requests(
            tmp_path,
            first_thread="t1", second_thread="t2",  # DIFFERENT thread
            first_client_meta={
                "role": "main", "is_curator": True,
                "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
            },
            second_client_meta={
                "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
            },
        )
        second = captured[1]
        assert second["client_meta"].get("session_id") is None
        assert second["client_meta"].get("role") is None
        assert second["client_meta"].get("is_curator") is None
        assert second["context"] == [{"role": "user", "content": "SECOND_CALLER_CONTENT"}]

    async def test_recue_preserved_matching_thread_adopts_identity_not_messages(
        self, tmp_path,
    ):
        """RE-CUE PRESERVED: a legitimate re-cue -- SAME thread_id as the idle
        holder, but the incoming request lost its session_id (the tool-call
        re-cue case) -- still adopts identity (session_id/role), and still
        does NOT adopt messages (its own context is used instead)."""
        captured = await self._run_two_requests(
            tmp_path,
            first_thread="t1", second_thread="t1",  # SAME thread -- genuine re-cue
            first_client_meta={
                "session_id": "s1", "role": "main",
                "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
            },
            second_client_meta={
                "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
            },
        )
        second = captured[1]
        assert second["client_meta"].get("session_id") == "s1"
        assert second["client_meta"].get("role") == "main"
        # Identity adopted, but content is still the caller's own.
        assert second["context"] == [{"role": "user", "content": "SECOND_CALLER_CONTENT"}]

    async def test_recue_preserved_label_classified_idle_holder(self, tmp_path):
        """RE-CUE PRESERVED, label case: the idle holder was LABEL-classified
        (is_curator), not literal-role. A same-thread re-cue must come out of
        the adopt with the SAME _bin_role classification as before the fix --
        a session_id+literal-role-only copy would silently demote it."""
        from turbohaul.manager import _bin_role

        first_client_meta = {
            "session_id": "s1", "is_curator": True,
            "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
        }
        captured = await self._run_two_requests(
            tmp_path,
            first_thread="t1", second_thread="t1",
            first_client_meta=first_client_meta,
            second_client_meta={
                "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
            },
        )
        second = captured[1]
        assert second["client_meta"].get("is_curator") is True
        assert _bin_role(second["client_meta"]) == _bin_role(first_client_meta) == "curator"

    async def test_explicit_same_session_still_adopts_identity(self, tmp_path):
        """The untouched disjunct: an explicit same-session_id match (not a
        re-cue) still adopts identity fields -- unchanged behaviour."""
        captured = await self._run_two_requests(
            tmp_path,
            first_thread="t1", second_thread="t2",  # different thread...
            first_client_meta={
                "session_id": "s1", "role": "main",
                "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
            },
            second_client_meta={
                "session_id": "s1",  # ...but explicit SAME session_id
                "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
            },
        )
        second = captured[1]
        assert second["client_meta"].get("role") == "main"
        assert second["context"] == [{"role": "user", "content": "SECOND_CALLER_CONTENT"}]


class TestWarmInheritRoleClobberFix:
    """Role-clobber guard (label-copy-without-messages poisoning).
    _is_same_session only
    proves the incoming request shares the idle holder's session_id -- NOT
    that it is the same role. A curator can legitimately share session_id
    with its parent main session (spawned "for" that session). Before the
    fix, the warm-inherit copy loop only overwrites keys PRESENT in the idle
    holder's client_meta -- it never clears keys absent there, so an
    incoming curator identified via is_curator=True survives untouched
    (is_curator still wins _class_from_label's priority order even with a
    stray is_main=True also present). The real clobber needs the incoming's
    OWN identity to be establishable ONLY via the literal-role back-compat
    fallback (no is_curator boolean) -- _class_from_label checks
    is_curator > is_compression > is_sub_agent > is_main BEFORE ever
    consulting a literal role string, so copying the idle holder's
    is_main=True onto an incoming curator identified only by role="curator"
    makes _class_from_label resolve to CLASS_MAIN, completely silently
    overriding the incoming's own correct classification -- and the
    last_main_client_meta refresh would then trust the (real) curator's own
    messages as main's own transcript."""

    async def _run_two_requests(
        self, tmp_path, *, first_thread, second_thread,
        first_client_meta, second_client_meta,
    ):
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=120
        )
        captured = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(handle, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            captured.append({
                "client_meta": dict(slot.client_meta or {}),
                "context": slot.context,
            })
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            await mgr.submit_and_wait(
                "gpt-x", "prompt-1", thread_id=first_thread,
                context=[{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
                client_meta=first_client_meta,
            )
            await asyncio.sleep(0.05)
            await mgr.submit_and_wait(
                "gpt-x", "prompt-2", thread_id=second_thread,
                context=[{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
                client_meta=second_client_meta,
            )
        finally:
            await mgr.shutdown()
        return captured

    async def test_discriminator_curator_sharing_main_session_keeps_own_role(
        self, tmp_path,
    ):
        """MUST FAIL before the fix: idle holder is main (is_main=True, no
        literal role). Incoming curator shares the SAME session_id (legit --
        spawned for that session) and identifies itself ONLY via the
        back-compat literal role="curator" (no is_curator boolean, exactly
        the case _class_from_label's own docstring says exists for back-
        compat). _is_same_session is True, so the pre-fix loop copies
        is_main=True from idle onto the incoming curator's client_meta --
        which now resolves to CLASS_MAIN via _class_from_label's own
        priority order, completely silently discarding the curator's own
        role="curator" literal."""
        from turbohaul.manager import _bin_role

        second_client_meta = {
            "session_id": "s-shared", "role": "curator",
            "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
        }
        captured = await self._run_two_requests(
            tmp_path,
            first_thread="t1", second_thread="t2",  # different thread, SAME session
            first_client_meta={
                "session_id": "s-shared", "is_main": True,
                "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
            },
            second_client_meta=second_client_meta,
        )
        second = captured[1]
        assert second["client_meta"].get("role") == "curator"
        assert second["client_meta"].get("is_main") is not True, (
            "idle holder's is_main=True was copied onto the incoming "
            "curator's client_meta -- role clobber via _class_from_label's "
            "priority order (is_main now beats the curator's own literal "
            "role fallback)"
        )
        assert _bin_role(second["client_meta"]) == "curator", (
            "the incoming curator's own identity must still resolve to "
            "'curator' after warm-inherit, not 'main'"
        )

    async def test_good_shape_survives_genuine_recue_still_adopts_identity(
        self, tmp_path,
    ):
        """Pairing assertion: this fix must not break the LEGITIMATE re-cue
        case (already covered by TestWarmInheritIdentityIsolation, repeated
        here as a same-file belt-and-suspenders check) -- a session-less
        same-thread continuation with NO classification of its own still
        adopts the idle holder's identity."""
        captured = await self._run_two_requests(
            tmp_path,
            first_thread="t1", second_thread="t1",  # SAME thread -- genuine re-cue
            first_client_meta={
                "session_id": "s1", "is_main": True,
                "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
            },
            second_client_meta={
                "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
            },
        )
        second = captured[1]
        assert second["client_meta"].get("session_id") == "s1"
        assert second["client_meta"].get("is_main") is True


@pytest.mark.asyncio
class TestWarmRecueIdentityLiveTwoResidentCap:
    """The SAME warm-inherit identity guard as the two
    classes above, but proven on the LIVE cap>=2 dispatch path
    (_route_or_reserve -> _restore_warm_recue_identity), not the dead
    _process_slot body those classes exercise. Two things the classes above
    cannot prove on their own: (1) that a resident's per-resident
    idle_client_meta/idle_thread_id -- a DIFFERENT data source than
    _process_slot's manager-global singleton -- actually carries the parked
    occupant's identity by the time the new call site reads it (not just that
    the observed OUTCOME looks right, which could pass for an unrelated
    reason); (2) that the must-not-adopt assertions have real teeth now, by
    showing them fail against a deliberately unconditional-adopt mutant."""

    async def _run_two_requests_spied(
        self, tmp_path, *, first_thread, second_thread,
        first_client_meta, second_client_meta, max_parallel_sidecars,
        restore_fn=None,
    ):
        """Same harness shape as the two classes above's own _run_two_requests,
        with two additions: an explicit max_parallel_sidecars (the scenario
        requires cap=2), and a spy wrapped around
        TurbohaulManager._restore_warm_recue_identity so the test can assert
        it was actually CALLED with the expected arguments -- not just that
        the final client_meta happened to look right."""
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=120,
            max_parallel_sidecars=max_parallel_sidecars,
        )
        captured = []
        recue_calls = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(handle, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            captured.append({
                "client_meta": dict(slot.client_meta or {}),
                "context": slot.context,
            })
            return {"ok": True}

        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram, complete_fn=fake_complete,
        )
        real_restore = TurbohaulManager._restore_warm_recue_identity
        target = restore_fn or real_restore

        def spy(self_mgr, idle_client_meta, idle_thread_id, slot):
            recue_calls.append({
                "idle_client_meta": dict(idle_client_meta) if idle_client_meta else idle_client_meta,
                "idle_thread_id": idle_thread_id,
            })
            return target(self_mgr, idle_client_meta, idle_thread_id, slot)

        mgr._restore_warm_recue_identity = spy.__get__(mgr, TurbohaulManager)

        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            await mgr.submit_and_wait(
                "gpt-x", "prompt-1", thread_id=first_thread,
                context=[{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
                client_meta=first_client_meta,
            )
            await asyncio.sleep(0.05)  # let the first slot enter idle-hot
            await mgr.submit_and_wait(
                "gpt-x", "prompt-2", thread_id=second_thread,
                context=[{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
                client_meta=second_client_meta,
            )
        finally:
            await mgr.shutdown()
        assert mgr.runtime.queue.max_parallel_sidecars == max_parallel_sidecars, (
            "cap must be set explicitly in the test, never inherited"
        )
        return captured, recue_calls

    async def test_two_resident_cap_explicit_recue_adopts_identity_via_the_real_call_site(
        self, tmp_path,
    ):
        """Boots at max_parallel_sidecars=2 EXPLICITLY (the
        deployed cap). Drives a REAL same-thread warm re-cue through
        worker_loop -> _dispatch_loop -> _route_or_reserve -- no hand-built
        Resident, no hand-set client_meta on the manager. Asserts BOTH the
        integration (the new call site was actually invoked with the parked
        occupant's real identity, proving the per-resident idle_client_meta/
        idle_thread_id data source is populated and read correctly --
        a key check) AND the outcome (session_id/role/is_curator survive)."""
        captured, recue_calls = await self._run_two_requests_spied(
            tmp_path,
            first_thread="t1", second_thread="t1",  # SAME thread -- genuine re-cue
            first_client_meta={
                "session_id": "s-two-resident-cap", "role": "main", "is_curator": False,
                "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
            },
            second_client_meta={
                "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
            },
            max_parallel_sidecars=2,
        )
        # INTEGRATION proof: the helper was actually called, with
        # the FIRST caller's real identity already parked as idle_client_meta
        # -- not None, not empty, not some other resident's leftover state.
        assert len(recue_calls) >= 1, (
            "_restore_warm_recue_identity was never called -- the new call "
            "site in _route_or_reserve did not run"
        )
        seen_real_idle_meta = [
            c for c in recue_calls
            if c["idle_client_meta"] and c["idle_client_meta"].get("session_id") == "s-two-resident-cap"
        ]
        assert seen_real_idle_meta, (
            f"helper was called but never with the first caller's parked "
            f"identity (session_id=s-two-resident-cap) -- calls were: {recue_calls}"
        )
        assert any(c["idle_thread_id"] == "t1" for c in seen_real_idle_meta), (
            "idle_thread_id did not carry the parked occupant's real thread_id"
        )
        # OUTCOME: session_id/role/is_curator survive the re-cue.
        second = captured[1]
        assert second["client_meta"].get("session_id") == "s-two-resident-cap"
        assert second["client_meta"].get("role") == "main"
        assert second["client_meta"].get("is_curator") is False
        assert second["context"] == [{"role": "user", "content": "SECOND_CALLER_CONTENT"}]

    async def test_over_eager_mutant_fuses_two_clients_identity_RED(self, tmp_path):
        """The trap to guard against: a fix that adopts identity
        UNCONDITIONALLY -- including across genuinely different clients --
        passes a naive must-adopt assertion and is a WORSE bug than the original
        defect (it fuses two conversations' identities). This mutant skips
        every guard (_is_recue / _is_same_session / cross-role-clobber
        refusal) and copies the idle holder's identity onto ANY incoming
        slot unconditionally. Run against the EXACT scenario
        test_substitution_and_isolation_sessionless_new_caller (a different
        thread, unrelated session-less caller) -- with the real guard this
        must NOT adopt; this test proves that assertion FAILS against the
        mutant, which is what makes the real guard's passing version
        meaningful evidence rather than a vacuous green."""
        def over_eager_unconditional_adopt(self_mgr, idle_client_meta, idle_thread_id, slot):
            if not idle_client_meta:
                return
            inc_cm = slot.client_meta if isinstance(slot.client_meta, dict) else {}
            for k in (
                "session_id", "role", "is_main", "is_sub_agent",
                "is_curator", "is_compression",
            ):
                if k in idle_client_meta:
                    inc_cm[k] = idle_client_meta[k]
            slot.client_meta = inc_cm

        captured, _ = await self._run_two_requests_spied(
            tmp_path,
            first_thread="t1", second_thread="t2",  # DIFFERENT thread -- unrelated caller
            first_client_meta={
                "session_id": "s1", "role": "main",
                "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
            },
            second_client_meta={
                "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
            },
            max_parallel_sidecars=2,
            restore_fn=over_eager_unconditional_adopt,
        )
        second = captured[1]
        # This is the SAME assertion test_substitution_and_isolation_sessionless_new_caller
        # makes with the REAL guard (and which passes there). Against the
        # over-eager mutant it must FAIL -- proving the must-not-adopt
        # assertion is not vacuous: it CAN go red, and it does exactly when
        # identity is fused across two different clients.
        with pytest.raises(AssertionError):
            assert second["client_meta"].get("session_id") is None
            assert second["client_meta"].get("role") is None

    async def test_two_resident_cap_sticky_multiinstance_recue_adopts_identity_via_route_to(
        self, tmp_path,
    ):
        """Mutant-testing follow-up: pins the OTHER
        live call site -- _route_to()'s closure inside _route_or_reserve's
        multi-instance gate, reached via the STICKY branch.
        Every other test in this class/file uses model_tag
        "gpt-x", which has no manifest on disk -- _effective_cap("gpt-x")
        therefore returns 1 (default-inert), the multi-instance gate
        `if eff_cap > 1 or len(insts) > 1 or ...` is never entered, and
        every scenario falls through to the plain HIT branch.
        A mutant proved this directly: unwiring the
        _restore_warm_recue_identity call inside _route_to
        left the whole 7-test suite green, because nothing
        drives that closure.

        This test drives it HONESTLY -- a real on-disk manifest
        (one card per engine: auto_place with split_mode none) + a patched VRAM probe
        with room for two cards, both real state the routing code itself
        reads, not a hand-built Resident or hand-set client_meta:

          1st submit_and_wait (session_id=s-sticky, thread_id=t1): no
          instance exists yet -> _sticky_resident_key finds nothing ->
          falls to the SPAWN branch, which reserves+starts a
          new Resident AND calls _remember_affinity(tag, "sess:s-sticky",
          resident_key) itself -- the affinity map entry this test depends
          on is written by production code, not seeded by the test.

          (sleep to let the resident finish serving and park
          IDLE_EVICTABLE, stamping idle_client_meta/idle_thread_id from
          grace_tip -- same wait the sibling cap=2 test above uses.)

          2nd submit_and_wait (SAME session_id, so SAME sticky key): the
          multi-instance gate is entered again, _sticky_resident_key(slot,
          tag) now HITS the affinity entry from step 1, the resident is
          not DEAD -> _route_to(rr) runs -- THIS is the call site
          the mutant showed was unpinned. Only one instance is ever
          spawned (asserted below) -- proving the second request routed
          via STICKY re-use, not a second cold spawn that would trivially
          also pick up identity via the SPAWN path's own fresh
          rank_client_meta stamp.
        """
        boot, runtime = _boot_runtime(
            tmp_path, grace_seconds=0, idle_hot_load_seconds=120,
            max_parallel_sidecars=2,
        )
        tag = "gpt-multi-sticky"
        manifest_path = boot.storage.manifests_path / f"{tag}.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "model_tag": tag,
            "gguf_blob_sha256": "c" * 64,
            "gguf_size_bytes": 1024 * 1024 * 1024,
            "context_size": 2048,
            "expected_vram_bytes": 1024 * 1024 * 1024,
            "auto_place": True,
            "llama_server_flags": {"split_mode": "none"},
        }))

        captured = []
        recue_calls = []
        spawn_ports = []

        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            spawn_ports.append(port)
            return _make_fake_handle(model_tag, port)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(handle, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            captured.append({
                "client_meta": dict(slot.client_meta or {}),
                "context": slot.context,
            })
            return {"ok": True}

        with patch(
            "turbohaul.safety._read_free_vram_all_mib", return_value=[8192, 8192],
        ), patch(
            "turbohaul.manager._read_free_vram_all_mib", return_value=[8192, 8192],
        ):
            mgr = TurbohaulManager(
                boot, runtime,
                spawn_fn=fake_spawn, health_fn=fake_health,
                sigterm_fn=fake_sigterm, vram_fn=fake_vram,
                complete_fn=fake_complete,
            )
            real_restore = TurbohaulManager._restore_warm_recue_identity

            def spy(self_mgr, idle_client_meta, idle_thread_id, slot):
                recue_calls.append({
                    "idle_client_meta": (
                        dict(idle_client_meta) if idle_client_meta else idle_client_meta
                    ),
                    "idle_thread_id": idle_thread_id,
                })
                return real_restore(self_mgr, idle_client_meta, idle_thread_id, slot)

            mgr._restore_warm_recue_identity = spy.__get__(mgr, TurbohaulManager)

            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            try:
                await mgr.submit_and_wait(
                    tag, "prompt-1", thread_id="t1",
                    context=[{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
                    client_meta={
                        "session_id": "s-sticky", "role": "main", "is_curator": False,
                        "messages": [{"role": "user", "content": "FIRST_CALLER_CONTENT"}],
                    },
                )
                await asyncio.sleep(0.05)  # let the first slot enter idle-hot
                await mgr.submit_and_wait(
                    tag, "prompt-2", thread_id="t1",
                    context=[{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
                    client_meta={
                        "session_id": "s-sticky",
                        "messages": [{"role": "user", "content": "SECOND_CALLER_CONTENT"}],
                    },
                )
            finally:
                await mgr.shutdown()

        # Precondition: only ONE instance was ever spawned -- the second
        # request routed via STICKY re-use of the first (_route_to), not a
        # second cold spawn. If this fails, the scenario setup is wrong
        # (e.g. VRAM/manifest not actually admitting a sticky hit) and the
        # rest of this test would be proving nothing about _route_to.
        assert len(spawn_ports) == 1, (
            f"expected exactly one spawn (second request should STICKY-route "
            f"via _route_to, not cold-spawn a second instance) -- got "
            f"{len(spawn_ports)} spawns"
        )
        assert mgr.runtime.queue.max_parallel_sidecars == 2

        # INTEGRATION proof: _restore_warm_recue_identity was actually
        # called with the first caller's real parked identity -- this can
        # only have happened via the _route_to closure's call site, since
        # the multi-instance gate (not the legacy HIT branch) is what ran
        # given the one-card-per-engine manifest.
        assert len(recue_calls) >= 1, (
            "_restore_warm_recue_identity was never called -- the _route_to "
            "call site did not run"
        )
        seen_real_idle_meta = [
            c for c in recue_calls
            if c["idle_client_meta"]
            and c["idle_client_meta"].get("session_id") == "s-sticky"
        ]
        assert seen_real_idle_meta, (
            f"helper was called but never with the first caller's parked "
            f"identity (session_id=s-sticky) -- calls were: {recue_calls}"
        )
        assert any(c["idle_thread_id"] == "t1" for c in seen_real_idle_meta), (
            "idle_thread_id did not carry the parked occupant's real thread_id"
        )

        # OUTCOME: session_id/role/is_curator survive the sticky re-cue.
        second = captured[1]
        assert second["client_meta"].get("session_id") == "s-sticky"
        assert second["client_meta"].get("role") == "main"
        assert second["client_meta"].get("is_curator") is False
        assert second["context"] == [{"role": "user", "content": "SECOND_CALLER_CONTENT"}]
