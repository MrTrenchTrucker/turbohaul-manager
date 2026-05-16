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
from turbohaul.fsm import transition
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
    verify_binary_sha256,
)


log = logging.getLogger(__name__)


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

    def __init__(self, boot: BootConfig, runtime: RuntimeConfig) -> None:
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

    # === Worker loop (skeleton; full impl in Wave 6) =========================

    async def worker_loop(self) -> None:
        """Drive the FSM forever. Full implementation in Wave 6 alongside API layer.

        Skeleton just consumes the queue + records state transitions; actual
        subprocess spawn + health-poll + chat-completion forwarding lands next wave.
        """
        log.info("worker_loop started (skeleton mode)")
        while not self._stop_event.is_set():
            try:
                slot = await self.queue.pop_next()
            except Exception as e:
                log.error("queue.pop_next failed: %s", e)
                await asyncio.sleep(0.5)
                continue

            if slot is None:
                # Empty queue; idle-tick
                await asyncio.sleep(0.25)
                continue

            # Skeleton: just record + mark COLD (Wave 6 will run the full state machine)
            log.info("popped slot %s model=%s (skeleton noop)", slot.slot_id, slot.model_tag)
            conn = open_state_db(self.boot.storage.state_db_path)
            try:
                mark_slot_ended(conn, slot.slot_id, "skeleton-mode-noop")
                record_audit_event(conn, "skeleton_consume", {}, slot_id=slot.slot_id)
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
