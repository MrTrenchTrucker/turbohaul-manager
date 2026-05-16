"""TurbohaulManager: top-level orchestrator wiring queue + subprocess + state + timers.

Per v0.2 ARCHITECTURE.md - orchestrates the whole lifecycle described in §6 state
machine. Phase 2 Wave 5 ships the foundational interface; the full worker_loop
streaming implementation lands in Wave 6 alongside the API layer that forwards
to llama-server.
"""
import asyncio
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from turbohaul.config import BootConfig, RuntimeConfig
from turbohaul.fsm import InvalidTransition, is_terminal, transition
from turbohaul.manifest import flags_to_argv, read_manifest
from turbohaul.queue import GraceTimer, IdleHotTimer, TurbohaulQueue
from turbohaul.singleton import boot_orphan_reaper, detect_foreign_gpu_apps
from turbohaul.slot import Slot, SlotState, derive_thread_id_prefix_hash
from turbohaul.state import (
    known_active_pids,
    mark_slot_ended,
    open_state_db,
    reconcile_orphaned_slots,
    record_audit_event,
    upsert_slot,
)
from turbohaul.subprocess_mgr import (
    SidecarHandle,
    drained_sigterm,
    spawn_sidecar,
    verify_binary_sha256,
    verify_vram_cleared,
    wait_until_healthy,
)


log = logging.getLogger(__name__)


class EventBus:
    """Pub-sub for state-level events broadcast to /ws/state subscribers.

    Per v0.2 §11.1 redaction policy: callers are responsible for emitting only
    safe events. This bus enforces a denylist (prompt/response/stderr/context)
    on publish as defense-in-depth — even if a caller accidentally includes one
    of those keys, it gets stripped before fan-out.
    """

    REDACTED_KEYS: set[str] = {
        "prompt",
        "response",
        "context",
        "stderr",
        "stdout",
        "messages",
    }

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.add(q)

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish_nowait(self, event: dict) -> None:
        """Publish an event. Sensitive keys are stripped (denylist).

        Each subscriber gets a copy. Full subscriber queues drop on back-pressure
        rather than block the publisher (worker_loop must stay responsive).
        """
        safe_event = {k: v for k, v in event.items() if k not in self.REDACTED_KEYS}
        for q in list(self._subscribers):
            try:
                q.put_nowait(safe_event)
            except asyncio.QueueFull:
                log.warning("event_bus subscriber queue full — dropping event")


def _pid_is_alive(pid: int, kill_fn: Callable[[int, int], None] | None = None) -> bool:
    """Defensive check: is pid currently alive on this host?"""
    fn = kill_fn or os.kill
    try:
        fn(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours - conservatively treat as alive


class TurbohaulManager:
    """Top-level orchestrator.

    Responsibilities:
    - Boot reconcile: orphan reap + foreign-GPU detect + state.sqlite slot cleanup
    - Verify binary sha256 pin at boot (v0.2 §7.1)
    - Accept fresh requests via submit() → push to queue (head if grace match)
    - Expose status_snapshot() for /status endpoint
    - Drive the FSM via worker_loop (skeleton in Wave 5; full streaming in Wave 6)
    - Clean shutdown
    """

    def __init__(
        self,
        boot: BootConfig,
        runtime: RuntimeConfig,
        *,
        spawn_fn: Callable | None = None,
        health_fn: Callable | None = None,
        sigterm_fn: Callable | None = None,
        vram_fn: Callable | None = None,
        complete_fn: Callable | None = None,
    ) -> None:
        self.boot = boot
        self.runtime = runtime
        self.queue = TurbohaulQueue(
            staging_max=runtime.queue.staging_queue_depth,
            acceptance_max=runtime.queue.acceptance_buffer_max,
        )
        self.grace = GraceTimer(
            grace_seconds=runtime.queue.grace_seconds,
            max_extensions=runtime.queue.max_grace_extensions,
        )
        self.idle = IdleHotTimer(idle_seconds=runtime.queue.idle_hot_load_seconds)
        self._active_handle: SidecarHandle | None = None
        self._active_slot: Slot | None = None
        self._worker_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        # Event bus for /ws/state subscribers (v0.2 §11.1 redacted)
        self.event_bus = EventBus()
        # Injection points (default = real subprocess_mgr; tests inject mocks)
        self._spawn = spawn_fn or spawn_sidecar
        self._wait_healthy = health_fn or wait_until_healthy
        self._sigterm = sigterm_fn or drained_sigterm
        self._vram_verify = vram_fn or verify_vram_cleared
        # _complete_fn: Phase 3 will replace with httpx → llama-server /v1/chat/completions
        self._complete_fn = complete_fn or self._default_complete

    async def _default_complete(self, slot: Slot, handle: SidecarHandle) -> None:
        """Placeholder for chat-completion forwarding. Phase 3 implements httpx proxy."""
        await asyncio.sleep(0.001)

    # === Boot lifecycle =====================================================

    def boot_reconcile(self, pid_is_alive_fn: Callable[[int], bool] | None = None) -> dict:
        """Run at startup. Returns summary dict for audit logging."""
        port_base = self.boot.runtime.default_port_base

        # 1. orphan reaper (kills /proc/<pid> llama-server orphans w/ PPid=1)
        reap = boot_orphan_reaper(port_base=port_base)

        # 2. foreign GPU detect — informational only (we don't refuse to start here;
        #    that's a CLI-flag decision)
        foreign = detect_foreign_gpu_apps()

        # 3. state.sqlite reconcile: any slot whose pid is no longer alive → COLD
        check_alive = pid_is_alive_fn or _pid_is_alive
        conn = open_state_db(self.boot.storage.state_db_path)
        try:
            stale_pids = known_active_pids(conn)
            live_pids = {pid for pid in stale_pids if check_alive(pid)}
            reconciled = reconcile_orphaned_slots(conn, live_pids)
            record_audit_event(
                conn,
                "boot_reconcile",
                {
                    "orphans_reaped": reap["reaped"],
                    "foreign_gpu_apps_count": len(foreign),
                    "slots_reconciled_to_cold": reconciled,
                },
            )
        finally:
            conn.close()

        return {
            "orphans_reaped": reap["reaped"],
            "orphans_failed": reap["failed"],
            "foreign_gpu_apps": foreign,
            "slots_reconciled_to_cold": reconciled,
        }

    def verify_binary(self) -> bool:
        """Verify llama_server_binary sha256 pin at boot (v0.2 §7.1)."""
        return verify_binary_sha256(
            self.boot.runtime.llama_server_binary,
            self.boot.runtime.llama_server_binary_sha256,
        )

    # === Request acceptance =================================================

    async def submit(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
    ) -> Slot:
        """Accept a fresh inference request.

        - If thread_id is empty, auto-derive from prompt-prefix-hash (Devil F7 fix).
        - If grace window is open for this thread+model → enqueue at FIFO HEAD
          + restart grace timer (max_grace_extensions cap applies).
        - Otherwise → normal FIFO enqueue.
        """
        if not thread_id:
            thread_id = derive_thread_id_prefix_hash(prompt, model_tag)

        slot = Slot.new(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
        )

        # Grace-window matched-thread shortcut
        if self.grace.matches(thread_id, model_tag):
            await self.queue.enqueue_head(slot)
            # restart_for_followup may return False if at extension cap; that's fine,
            # the request still queues at head once, but the slot will pop next cycle.
            self.grace.restart_for_followup()
        else:
            await self.queue.enqueue(slot)

        # Audit
        conn = open_state_db(self.boot.storage.state_db_path)
        try:
            upsert_slot(
                conn,
                {
                    "slot_id": slot.slot_id,
                    "model_tag": slot.model_tag,
                    "thread_id": slot.thread_id,
                    "state": slot.state.value,
                    "client_meta": slot.client_meta,
                },
            )
            record_audit_event(
                conn,
                "submit",
                {"thread_id_prefix": (thread_id or "")[:8], "model_tag": model_tag},
                slot_id=slot.slot_id,
            )
        finally:
            conn.close()

        return slot

    # === Status snapshot =====================================================

    def status_snapshot(self) -> dict:
        """/status payload per v0.2 §9.3."""
        depth = self.queue.depth()

        active_info: dict | None = None
        if self._active_slot is not None and self._active_handle is not None:
            active_info = {
                "slot_id": self._active_slot.slot_id,
                "model_tag": self._active_slot.model_tag,
                "state": self._active_slot.state.value,
                # Redaction: only first 8 chars of thread_id exposed (v0.2 §11.1)
                "thread_id_prefix": (self._active_slot.thread_id or "")[:8],
                "pid": self._active_handle.pid,
                "port": self._active_handle.port,
            }

        grace_info: dict | None = None
        if not self.grace.expired():
            grace_info = {
                "remaining_s": int(self.grace.remaining_s()),
                "extension_count": self.grace.extension_count,
                "max_extensions": self.grace.max_extensions,
                "thread_id_prefix": (self.grace.thread_id or "")[:8] if self.grace.thread_id else "",
                "model_tag": self.grace.model_tag,
            }

        idle_info: dict | None = None
        if not self.idle.expired():
            idle_info = {
                "remaining_s": int(self.idle.remaining_s()),
                "model_tag": self.idle.model_tag,
            }

        return {
            "queue": {
                "acceptance_buffer_depth": depth["acceptance_buffer_depth"],
                "staging_queue_depth": depth["staging_queue_depth"],
                "staging_queue_max": depth["staging_queue_max"],
            },
            "active": active_info,
            "grace": grace_info,
            "idle_hot": idle_info,
            "parallel_slots": {
                "used": 1 if self._active_handle else 0,
                "max": self.runtime.queue.max_parallel_sidecars,
            },
        }

    # === Worker loop (full FSM-driven cycle) =================================

    async def worker_loop(self) -> None:
        """Drive the FSM forever: pop → spawn → active → complete → grace → pop → idle.

        Per v0.2 §6. Subprocess interactions are dependency-injected via ctor (spawn_fn,
        health_fn, sigterm_fn, vram_fn, complete_fn). Default implementations call the
        real subprocess_mgr functions. Tests inject mocks.
        """
        log.info("worker_loop started")
        while not self._stop_event.is_set():
            slot = await self.queue.pop_next()
            if slot is None:
                await asyncio.sleep(0.05)
                continue
            try:
                await self._process_slot(slot)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("slot %s processing failed", slot.slot_id)
                await self._force_cold(slot, "worker-uncaught-exception")
        log.info("worker_loop exited")

    async def _process_slot(self, slot: Slot) -> None:
        """Drive one slot through STAGED → LOADING → ACTIVE → GRACE → POPPED."""
        self._active_slot = slot

        try:
            # Build llama-server argv from manifest if available; tolerate missing
            # manifest for testing convenience.
            argv: list[str] = []
            try:
                manifest = read_manifest(self.boot.storage.manifests_path, slot.model_tag)
                argv = flags_to_argv(manifest.llama_server_flags)
                gguf_path = (
                    self.boot.storage.blob_store_path
                    / "sha256"
                    / manifest.gguf_blob_sha256[:2]
                    / manifest.gguf_blob_sha256
                )
            except FileNotFoundError:
                gguf_path = self.boot.storage.blob_store_path / "missing.gguf"

            port = self.boot.runtime.default_port_base

            # STAGED → LOADING
            transition(slot, SlotState.LOADING)
            self._audit(slot, "stage_to_loading")

            handle = self._spawn(
                self.boot.runtime.llama_server_binary,
                gguf_path,
                port,
                slot.model_tag,
                argv,
            )
            slot.port = handle.port
            slot.pid = handle.pid
            self._active_handle = handle

            # LOADING → ACTIVE (or LOADING_FAIL → POPPED)
            healthy = await self._wait_healthy(
                port, self.runtime.queue.loading_health_timeout_s
            )
            if not healthy:
                transition(slot, SlotState.LOADING_FAIL)
                self._audit(slot, "loading_fail_health_timeout")
                transition(slot, SlotState.POPPED)
                await self._teardown(slot, "loading-fail-health-timeout")
                return

            transition(slot, SlotState.ACTIVE)
            slot.started_active_at = time.monotonic()
            self._audit(slot, "active")

            # Completion (placeholder — Phase 3 implements httpx proxy)
            await self._complete_fn(slot, handle)

            # ACTIVE → GRACE
            transition(slot, SlotState.GRACE)
            slot.grace_started_at = time.monotonic()
            self.grace.start(slot.thread_id, slot.model_tag)
            self._audit(slot, "grace_enter")

            # Wait for grace window (or stop signal); follow-up rematch via queue
            # is a Phase-3 enhancement and isn't wired here yet.
            deadline = time.monotonic() + self.runtime.queue.grace_seconds
            while time.monotonic() < deadline and not self._stop_event.is_set():
                await asyncio.sleep(0.05)

            # GRACE → POPPED
            transition(slot, SlotState.POPPED)
            await self._teardown(slot, "grace-expired")
            # IDLE_HOT applies to the model, not this slot - record event only
            self.idle.start(slot.model_tag)
            self._audit_event_only(slot.slot_id, "idle_hot_enter", {"model_tag": slot.model_tag})
        finally:
            self._active_slot = None
            self._active_handle = None

    async def _teardown(self, slot: Slot, reason: str) -> None:
        """Drained SIGTERM the process group → VRAM verify → audit."""
        if self._active_handle is not None:
            ok, status = await self._sigterm(
                self._active_handle,
                drained_window_s=float(self.runtime.queue.drained_sigterm_window_active_s),
                is_active=False,
                cold_window_s=float(self.runtime.queue.drained_sigterm_window_cold_s),
            )
            # VRAM verify (default expected_drop 1024 MiB; future: read from manifest)
            await self._vram_verify(expected_drop_mib=1024, timeout_s=30.0)
            conn = open_state_db(self.boot.storage.state_db_path)
            try:
                mark_slot_ended(conn, slot.slot_id, reason)
                record_audit_event(
                    conn,
                    "teardown",
                    {"reason": reason, "sigterm_status": status, "sigterm_ok": ok},
                    slot_id=slot.slot_id,
                )
            finally:
                conn.close()

    async def _force_cold(self, slot: Slot, reason: str) -> None:
        """Mark a slot COLD when processing dies mid-flight."""
        if not is_terminal(slot.state):
            try:
                # Best-effort - jump directly to a terminal state via legal path
                if slot.state == SlotState.ACTIVE or slot.state == SlotState.GRACE:
                    transition(slot, SlotState.POPPED)
                if slot.state == SlotState.POPPED:
                    transition(slot, SlotState.COLD)
            except InvalidTransition:
                slot.state = SlotState.COLD
        conn = open_state_db(self.boot.storage.state_db_path)
        try:
            mark_slot_ended(conn, slot.slot_id, reason)
        finally:
            conn.close()

    def _audit(self, slot: Slot, event_type: str) -> None:
        """Audit: upsert current slot row + record event + publish to event bus."""
        conn = open_state_db(self.boot.storage.state_db_path)
        try:
            upsert_slot(
                conn,
                {
                    "slot_id": slot.slot_id,
                    "model_tag": slot.model_tag,
                    "thread_id": slot.thread_id,
                    "state": slot.state.value,
                    "port": slot.port,
                    "pid": slot.pid,
                    "extension_count": slot.extension_count,
                    "client_meta": slot.client_meta,
                },
            )
            record_audit_event(conn, event_type, {"state": slot.state.value}, slot_id=slot.slot_id)
        finally:
            conn.close()
        # Publish redacted event to WS subscribers (v0.2 §11.1)
        self.event_bus.publish_nowait(
            {
                "event": event_type,
                "slot_id": slot.slot_id,
                "model_tag": slot.model_tag,
                "state": slot.state.value,
                # Redaction: only first 8 chars of thread_id exposed
                "thread_id_prefix": (slot.thread_id or "")[:8],
            }
        )

    def _audit_event_only(self, slot_id: str, event_type: str, payload: dict | None = None) -> None:
        """Audit: record event ONLY, no slot row mutation.

        Use after teardown when the slot is already COLD in DB and we don't want
        to clobber that state.
        """
        conn = open_state_db(self.boot.storage.state_db_path)
        try:
            record_audit_event(conn, event_type, payload or {}, slot_id=slot_id)
        finally:
            conn.close()

    # === Shutdown ============================================================

    async def shutdown(self) -> None:
        """Clean tear-down. Stops worker loop + drains queue + closes state db."""
        self._stop_event.set()
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self.queue.close()
