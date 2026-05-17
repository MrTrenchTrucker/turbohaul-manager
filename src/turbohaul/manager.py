"""TurbohaulManager: top-level orchestrator wiring queue + subprocess + state + timers.

Per v0.2 ARCHITECTURE.md - orchestrates the whole lifecycle described in §6 state
machine. Phase 2 Wave 5 ships the foundational interface; the full worker_loop
streaming implementation lands in Wave 6 alongside the API layer that forwards
to llama-server.
"""
import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from turbohaul.config import BootConfig, RuntimeConfig
from turbohaul.fsm import LEGAL_TRANSITIONS, InvalidTransition, is_terminal, transition
from turbohaul.manifest import flags_to_argv, read_manifest
from turbohaul.queue import GraceTimer, IdleHotTimer, TurbohaulQueue
from turbohaul.safety import all_safety_gates
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
    open_and_verify_binary,
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
        # GRIP H-4 wire: manager-level idle holder (model warm post-grace).
        # When grace expires without a thread match, the sidecar is NOT
        # torn down -- it migrates here and stays alive for idle_seconds.
        # Next slot of same model_tag inherits the handle; different
        # model_tag tears it down first.
        self._idle_handle: SidecarHandle | None = None
        self._idle_model_tag: str | None = None
        self._idle_expires_at: float | None = None
        self._worker_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        # GRIP M-2 fix: removed unused self._lock = asyncio.Lock(). It had
        # zero acquire sites (single-task worker_loop discipline protects
        # _active_handle / _active_slot mutations). If Wave-6 lands a
        # second mutating task, re-add it purposefully and wrap the call
        # sites that actually need it.
        self._binary_fd: int | None = None  # HAUL P-4: TOCTOU-pinned fd
        # Event bus for /ws/state subscribers (v0.2 §11.1 redacted)
        self.event_bus = EventBus()
        # Injection points (default = real subprocess_mgr; tests inject mocks)
        self._spawn = spawn_fn or spawn_sidecar
        self._wait_healthy = health_fn or wait_until_healthy
        self._sigterm = sigterm_fn or drained_sigterm
        self._vram_verify = vram_fn or verify_vram_cleared
        # _complete_fn: Phase 3 will replace with httpx → llama-server /v1/chat/completions
        self._complete_fn = complete_fn or self._default_complete

    async def _default_complete(self, slot: Slot, handle: SidecarHandle) -> dict | None:
        """Default no-op completion (Phase 2). Phase 3 wires httpx proxy via DI."""
        await asyncio.sleep(0.001)
        return None

    async def submit_and_wait(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
        timeout_s: float = 600.0,
    ) -> tuple[Slot, Any]:
        """Submit + await the slot's completion. Returns (slot, completion_result)."""
        slot = await self.submit(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
            wait_for_completion=True,
        )
        try:
            result = await asyncio.wait_for(slot.completion_future, timeout=timeout_s)
        except asyncio.TimeoutError:
            raise
        return slot, result

    # === Error propagation on worker exceptions ==============================

    def _fail_completion_future(self, slot: Slot, exc: BaseException) -> None:
        """If a slot has a pending completion_future, mark it failed (don't hang caller)."""
        fut = slot.completion_future
        if fut is not None and not fut.done():
            fut.set_exception(exc)

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
        """Verify + TOCTOU-pin llama_server_binary at boot (v0.2 §7.1 + HAUL P-4).

        Empty expected_sha256 = dev mode -- verify_binary_sha256 returns True
        with no fd pinning; spawn_sidecar falls back to path-based exec.
        Non-empty + matching hash = an inode-pinned fd is held on
        self._binary_fd and every spawn execs via ``/proc/self/fd/<fd>``,
        closing the swap window.
        """
        binary_path = self.boot.runtime.llama_server_binary
        expected = self.boot.runtime.llama_server_binary_sha256
        if not verify_binary_sha256(binary_path, expected):
            return False
        if self._binary_fd is not None:
            # Defensive: close prior fd if verify_binary called twice
            with contextlib.suppress(OSError):
                os.close(self._binary_fd)
            self._binary_fd = None
        if expected:
            self._binary_fd = open_and_verify_binary(binary_path, expected)
            if self._binary_fd is None:
                log.error(
                    "binary hash drift between verify and fd-pin -- refusing"
                )
                return False
        return True

    # === Request acceptance =================================================

    async def submit(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
        wait_for_completion: bool = False,
    ) -> Slot:
        """Accept a fresh inference request.

        - If thread_id is empty, auto-derive from prompt-prefix-hash (Devil F7 fix).
        - If grace window is open for this thread+model → enqueue at FIFO HEAD
          + restart grace timer (max_grace_extensions cap applies).
        - Otherwise → normal FIFO enqueue.

        Raises RuntimeError if the manager is shutting down (v0.2.1 RC GRIP-C3).
        """
        # C3 fix (GRIP DAG): refuse new submissions after shutdown signal so
        # callers fail fast instead of hanging on a completion_future that
        # will never resolve (worker is dead, queue.close() clears staging).
        # Raises queue.QueueClosed to match existing test_manager contract +
        # the documented queue-closed exception type.
        if self._stop_event.is_set():
            from turbohaul.queue import QueueClosed
            raise QueueClosed(
                "TurbohaulManager is shutting down — new submissions are refused."
            )

        if not thread_id:
            thread_id = derive_thread_id_prefix_hash(prompt, model_tag)

        slot = Slot.new(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
        )

        # Attach a future BEFORE enqueue so worker_loop can resolve it on completion.
        if wait_for_completion:
            slot.completion_future = asyncio.get_event_loop().create_future()

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
        # GRIP H-4 wire: /status idle snapshot reflects the manager-level
        # _idle_* holder (which IS the warm sidecar), not the legacy
        # IdleHotTimer (which only tracks the model name).
        if (
            self._idle_handle is not None
            and self._idle_expires_at is not None
            and time.monotonic() < self._idle_expires_at
        ):
            idle_info = {
                "remaining_s": int(self._idle_expires_at - time.monotonic()),
                "model_tag": self._idle_model_tag,
            }
        elif not self.idle.expired():
            # Backward compat: when idle_seconds=0 (test mode) the warm
            # holder is not used and self.idle still tracks "last model".
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
                # GRIP H-4 wire: check idle holder expiry on idle ticks.
                if (
                    self._idle_handle is not None
                    and self._idle_expires_at is not None
                    and time.monotonic() >= self._idle_expires_at
                ):
                    try:
                        await self._teardown_idle_holder("idle_expired")
                    except Exception:
                        log.exception(
                            "idle expiry teardown failed (best-effort)"
                        )
                await asyncio.sleep(0.05)
                continue
            try:
                await self._process_slot(slot)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("slot %s processing failed", slot.slot_id)
                self._fail_completion_future(slot, e)
                # C2 fix (GRIP DAG): teardown active sidecar BEFORE force_cold
                # to prevent PID leak. _force_cold only updates DB state; without
                # teardown the spawned llama-server keeps running and the
                # single-slot invariant breaks until boot_orphan_reaper at next
                # restart. Best-effort — don't let teardown failure mask the
                # original exception that triggered this path.
                if self._active_handle is not None:
                    try:
                        await self._teardown(slot, "worker-uncaught-exception")
                    except Exception:
                        log.exception(
                            "teardown during worker exception failed (best-effort)"
                        )
                await self._force_cold(slot, "worker-uncaught-exception")
        log.info("worker_loop exited")

    async def _process_slot(self, slot: Slot) -> None:
        """Drive one slot through STAGED → LOADING → ACTIVE → GRACE → POPPED."""
        self._active_slot = slot

        try:
            # Build llama-server argv from manifest if available; tolerate missing
            # manifest for testing convenience.
            argv: list[str] = []
            manifest_found = True
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
                manifest_found = False
                gguf_path = self.boot.storage.blob_store_path / "missing.gguf"

            # GRIP H-4 polish (RELAY stress test #15735): pre-validate the
            # manifest BEFORE the teardown_idle_holder branch when a warm
            # holder for a DIFFERENT model is at risk. A bogus model_tag
            # with no manifest used to fall into the "different model"
            # path and tear down the warm holder, then the spawn would
            # inevitably hang in LOADING (missing.gguf) and health-check
            # timeout. The warm hold was lost for nothing.
            # Now: if manifest is missing AND an idle holder exists for a
            # DIFFERENT model, bail fast (LOADING -> LOADING_FAIL -> POPPED)
            # WITHOUT touching self._idle_handle. Otherwise (no holder OR
            # same-model holder which we would inherit anyway) fall through
            # to the legacy missing.gguf -> spawn-then-LOADING_FAIL path so
            # existing tests that rely on the manifest-missing tolerance
            # continue to work.
            holder_at_risk = (
                not manifest_found
                and self._idle_handle is not None
                and self._idle_model_tag != slot.model_tag
            )
            if holder_at_risk:
                transition(slot, SlotState.LOADING)
                self._audit(slot, "stage_to_loading")
                transition(slot, SlotState.LOADING_FAIL)
                self._audit(slot, "manifest_not_found")
                transition(slot, SlotState.POPPED)
                _mfn_conn = open_state_db(self.boot.storage.state_db_path)
                try:
                    mark_slot_ended(
                        _mfn_conn, slot.slot_id, "manifest_not_found",
                    )
                finally:
                    _mfn_conn.close()
                self._fail_completion_future(
                    slot,
                    RuntimeError(
                        f"model_tag {slot.model_tag!r} has no manifest "
                        f"(idle holder for {self._idle_model_tag!r} preserved)"
                    ),
                )
                return

            port = self.boot.runtime.default_port_base

            # STAGED → LOADING
            transition(slot, SlotState.LOADING)
            self._audit(slot, "stage_to_loading")

            # GRIP H-4 wire: warm-inherit path. If the previous slot left
            # a warm sidecar holding the same model_tag and the idle
            # window has not expired, reuse the handle and skip spawn +
            # health-wait (sidecar is already healthy by construction).
            warm_inherit = (
                self._idle_handle is not None
                and self._idle_model_tag == slot.model_tag
                and self._idle_expires_at is not None
                and time.monotonic() < self._idle_expires_at
            )
            if warm_inherit:
                handle = self._idle_handle
                self._idle_handle = None
                self._idle_model_tag = None
                self._idle_expires_at = None
                slot.port = handle.port
                slot.pid = handle.pid
                self._active_handle = handle
                self._audit(slot, "idle_hot_inherit")
                # Skip spawn + health wait; jump straight to LOADING -> ACTIVE.
                healthy = True
            else:
                # Different model OR no idle holder. If a stale holder
                # exists for a different model, tear it down before
                # spawning new -- immediate switch (Cmdr #15709 intent).
                if self._idle_handle is not None:
                    await self._teardown_idle_holder("model_swap")
                # Cmdr #15653 safety guardrails: pre-spawn host checks
                # (VRAM headroom + RAM + CPU load + IO wait). Refuse here
                # rather than spawning into an OOM / IO-stuck host.
                if self.runtime.queue.safety_enabled:
                    manifest_vram = 0
                    try:
                        m_for_vram = read_manifest(
                            self.boot.storage.manifests_path,
                            slot.model_tag,
                        )
                        manifest_vram = m_for_vram.expected_vram_bytes or 0
                    except FileNotFoundError:
                        manifest_vram = 0
                    gates = all_safety_gates(
                        min_free_ram_mib=self.runtime.queue.safety_min_free_ram_mib,
                        min_free_vram_mib=self.runtime.queue.safety_min_free_vram_mib,
                        max_load_per_core=self.runtime.queue.safety_max_load_per_core,
                        max_iowait_percent=self.runtime.queue.safety_max_iowait_percent,
                        manifest_expected_vram_bytes=manifest_vram,
                        iowait_sample_window_s=self.runtime.queue.safety_iowait_sample_window_s,
                    )
                    failed = [g for g in gates if not g.ok]
                    if failed:
                        # Build a single error message + emit audit detail.
                        detail = "; ".join(
                            f"{g.name}: {g.detail}" for g in failed
                        )
                        log.warning(
                            "safety gates refused spawn for slot %s: %s",
                            slot.slot_id, detail,
                        )
                        transition(slot, SlotState.LOADING_FAIL)
                        self._audit(slot, "safety_gate_refused")
                        self._audit_event_only(
                            slot.slot_id,
                            "safety_gate_detail",
                            {"failed": [
                                {"name": g.name, "detail": g.detail}
                                for g in failed
                            ]},
                        )
                        transition(slot, SlotState.POPPED)
                        # No sidecar spawned; _teardown is a no-op + audit-only.
                        # Just mark slot ended + fail caller future.
                        _sg_conn = open_state_db(
                            self.boot.storage.state_db_path
                        )
                        try:
                            mark_slot_ended(
                                _sg_conn, slot.slot_id, "safety_gate_refused",
                            )
                        finally:
                            _sg_conn.close()
                        self._fail_completion_future(
                            slot,
                            RuntimeError(
                                f"safety gates refused spawn: {detail}",
                            ),
                        )
                        return
                handle = self._spawn(
                    self.boot.runtime.llama_server_binary,
                    gguf_path,
                    port,
                    slot.model_tag,
                    argv,
                    binary_fd=self._binary_fd,
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
                self._fail_completion_future(slot, RuntimeError("loading-fail-health-timeout"))
                return

            transition(slot, SlotState.ACTIVE)
            slot.started_active_at = time.monotonic()
            self._audit(slot, "active")

            # Completion (Phase 3 wires httpx forward; Phase 2 default is noop)
            result = await self._complete_fn(slot, handle)
            if slot.completion_future is not None and not slot.completion_future.done():
                slot.completion_future.set_result(result)

            # ACTIVE → GRACE
            transition(slot, SlotState.GRACE)
            slot.grace_started_at = time.monotonic()
            self.grace.start(slot.thread_id, slot.model_tag)
            self._audit(slot, "grace_enter")

            # Wait for grace window OR promote a matched staging slot via
            # ACTIVE_MATCH (warm-slot reuse). Per v0.2 §6 FSM; this transition
            # cascades same-(thread_id, model_tag) follow-up requests through
            # the warm llama-server without re-spawn.
            # Wired in v0.2.1 RC-CF78A6-ACTIVE-MATCH-WIRE (was Phase 3 W12 stub).
            deadline = time.monotonic() + self.runtime.queue.grace_seconds
            while time.monotonic() < deadline and not self._stop_event.is_set():
                # GRIP H-3 fix: atomic find + remove in one lock acquire
                matched = await self.queue.pop_matched_thread(
                    slot.thread_id, slot.model_tag
                )
                if matched is not None:
                    matched.port = handle.port
                    matched.pid = handle.pid
                    self._active_slot = matched
                    # AM-1 fix (HAUL DAG): state-drift guard. If matched.state
                    # drifted between find_matched_thread + here (concurrent
                    # reconcile, retry path, etc.), transition raises
                    # InvalidTransition which would crash worker_loop. Wrap
                    # the promotion + park-on-drift instead of propagate.
                    try:
                        transition(matched, SlotState.ACTIVE_MATCH)
                        self._audit(matched, "active_match_promoted")
                        transition(matched, SlotState.ACTIVE)
                    except InvalidTransition as drift_err:
                        log.warning(
                            "active_match state drift: slot %s in %s — terminal-park; %s",
                            matched.slot_id, matched.state.value, drift_err,
                        )
                        self._fail_completion_future(matched, drift_err)
                        await self._force_cold(
                            matched,
                            f"active_match_state_drift:{matched.state.value}",
                        )
                        self._active_slot = slot
                        continue
                    matched.started_active_at = time.monotonic()
                    completed_ok = True
                    try:
                        result2 = await self._complete_fn(matched, handle)
                        if (
                            matched.completion_future is not None
                            and not matched.completion_future.done()
                        ):
                            matched.completion_future.set_result(result2)
                    except asyncio.CancelledError:
                        # AM-2 fix (HAUL DAG): cancellation mid-ACTIVE_MATCH
                        # must terminal-park the matched slot so it does not
                        # rot as a zombie ACTIVE row in state.sqlite, then
                        # re-raise so worker_loop's teardown runs cleanly.
                        self._fail_completion_future(
                            matched,
                            asyncio.CancelledError("shutdown during active_match"),
                        )
                        try:
                            transition(matched, SlotState.POPPED)
                        except InvalidTransition:
                            pass
                        self._audit(matched, "active_match_cancelled")
                        try:
                            _am_conn = open_state_db(
                                self.boot.storage.state_db_path
                            )
                            try:
                                mark_slot_ended(
                                    _am_conn,
                                    matched.slot_id,
                                    "active_match_cancelled",
                                )
                            finally:
                                _am_conn.close()
                        except Exception:
                            log.exception(
                                "AM-2 cleanup mark_slot_ended failed for %s",
                                matched.slot_id,
                            )
                        raise
                    except Exception as e:  # noqa: BLE001 -- per-slot isolation
                        completed_ok = False
                        self._fail_completion_future(matched, e)
                        log.exception(
                            "active_match completion failed for slot %s",
                            matched.slot_id,
                        )
                    # GRIP M-1 fix: on completion failure, skip grace pretense
                    # — go ACTIVE → GRACE → POPPED + mark failed, keep state
                    # machine honest. (transition validates each hop.)
                    transition(matched, SlotState.GRACE)
                    if completed_ok:
                        self._audit(matched, "active_match_to_grace")
                        if self.grace.restart_for_followup():
                            # GRIP H-1 fix: also bump per-slot extension_count
                            # (was always 0 in sqlite — only GraceTimer's was).
                            matched.extension_count = self.grace.extension_count
                            deadline = time.monotonic() + self.runtime.queue.grace_seconds
                            self._audit_event_only(
                                matched.slot_id,
                                "grace_extended_via_active_match",
                                {"extension_count": self.grace.extension_count},
                            )
                    else:
                        self._audit(matched, "active_match_failed")
                    # C1 fix (GRIP DAG): matched slot's request is done; its
                    # sidecar was the anchor's warm process (reused, not its own).
                    # Anchor `slot` remains the GRACE driver until grace expiry.
                    transition(matched, SlotState.POPPED)
                    self._audit(matched, "active_match_completed" if completed_ok else "active_match_failed_terminal")
                    _am_conn = open_state_db(self.boot.storage.state_db_path)
                    try:
                        mark_slot_ended(
                            _am_conn,
                            matched.slot_id,
                            "active_match_completed" if completed_ok else "active_match_failed",
                        )
                    finally:
                        _am_conn.close()
                    self._active_slot = slot  # anchor for teardown bookkeeping
                    continue
                await asyncio.sleep(0.05)

            # GRACE → POPPED (slot lifecycle ends here)
            transition(slot, SlotState.POPPED)
            # GRIP H-4 wire: hold the sidecar in idle for follow-up reuse
            # by any same-model_tag request inside idle_hot_load_seconds.
            # When idle_seconds == 0 (test default), this is equivalent to
            # immediate teardown -- preserves "grace-expired" reason on the
            # mark_slot_ended audit (backward-compat with existing tests).
            idle_seconds = self.runtime.queue.idle_hot_load_seconds
            if idle_seconds > 0 and self._active_handle is not None:
                # Hand off the active handle to the manager-level idle holder.
                self._idle_handle = self._active_handle
                self._idle_model_tag = slot.model_tag
                self._idle_expires_at = time.monotonic() + idle_seconds
                self._audit_event_only(
                    slot.slot_id,
                    "idle_hot_enter",
                    {
                        "model_tag": slot.model_tag,
                        "idle_seconds": idle_seconds,
                    },
                )
                # Mark the slot ended at the state.sqlite layer -- the slot
                # is done; only the model stays warm. Audit reason names the
                # warm-hold so post-hoc audits can see the difference vs.
                # plain grace-expired teardown.
                _ih_conn = open_state_db(self.boot.storage.state_db_path)
                try:
                    mark_slot_ended(
                        _ih_conn, slot.slot_id, "grace-expired-held-idle"
                    )
                finally:
                    _ih_conn.close()
            else:
                # idle disabled (idle_seconds=0) or no handle -- immediate teardown.
                await self._teardown(slot, "grace-expired")
                self.idle.start(slot.model_tag)
                self._audit_event_only(
                    slot.slot_id,
                    "idle_hot_enter",
                    {"model_tag": slot.model_tag},
                )
        finally:
            self._active_slot = None
            self._active_handle = None

    async def _teardown(self, slot: Slot, reason: str) -> None:
        """Drained SIGTERM the process group → VRAM verify → orphan reap → audit."""
        if self._active_handle is not None:
            ok, status = await self._sigterm(
                self._active_handle,
                drained_window_s=float(self.runtime.queue.drained_sigterm_window_active_s),
                is_active=False,
                cold_window_s=float(self.runtime.queue.drained_sigterm_window_cold_s),
            )
            # VRAM verify (default expected_drop 1024 MiB; future: read from manifest)
            await self._vram_verify(expected_drop_mib=1024, timeout_s=30.0)
            # HAUL P-2 fix: scan for grandchild orphans left behind by
            # Tom's Fork setsid-detach (killpg never reached them) and
            # reap before the next slot needs the port + VRAM. ~50ms
            # /proc walk; cheap to run on every teardown.
            orphan_reaped = 0
            try:
                orphan_reap_result = boot_orphan_reaper(
                    port_base=self.boot.runtime.default_port_base,
                    known_pids=set(),  # single-slot mode; multi-slot
                                       # Wave-6 will pass live sidecar pids
                )
                orphan_reaped = orphan_reap_result.get("reaped", 0)
            except Exception:
                log.exception(
                    "post-teardown orphan reap failed (best-effort)"
                )
            conn = open_state_db(self.boot.storage.state_db_path)
            try:
                mark_slot_ended(conn, slot.slot_id, reason)
                record_audit_event(
                    conn,
                    "teardown",
                    {
                        "reason": reason,
                        "sigterm_status": status,
                        "sigterm_ok": ok,
                        "post_teardown_orphans_reaped": orphan_reaped,
                    },
                    slot_id=slot.slot_id,
                )
            finally:
                conn.close()

    async def _teardown_idle_holder(self, reason: str) -> None:
        """GRIP H-4 wire: tear down the manager-level idle handle.

        Called when:
        - a slot for a DIFFERENT model_tag arrives (immediate switch path), or
        - the idle timer expires in the worker_loop, or
        - shutdown.
        """
        if self._idle_handle is None:
            return
        held = self._idle_handle
        model_tag = self._idle_model_tag
        self._idle_handle = None
        self._idle_model_tag = None
        self._idle_expires_at = None
        ok, status = await self._sigterm(
            held,
            drained_window_s=float(
                self.runtime.queue.drained_sigterm_window_active_s
            ),
            is_active=False,
            cold_window_s=float(
                self.runtime.queue.drained_sigterm_window_cold_s
            ),
        )
        await self._vram_verify(expected_drop_mib=1024, timeout_s=30.0)
        try:
            boot_orphan_reaper(
                port_base=self.boot.runtime.default_port_base,
                known_pids=set(),
            )
        except Exception:
            log.exception(
                "idle-holder orphan reap failed (best-effort)"
            )
        _ih_conn = open_state_db(self.boot.storage.state_db_path)
        try:
            record_audit_event(
                _ih_conn,
                "teardown_idle_holder",
                {
                    "reason": reason,
                    "model_tag": model_tag,
                    "sigterm_status": status,
                    "sigterm_ok": ok,
                },
            )
        finally:
            _ih_conn.close()

    async def _force_cold(self, slot: Slot, reason: str) -> None:
        """Mark a slot COLD when processing dies mid-flight.

        GRIP H-5 fix: walk legal transitions to COLD from any non-terminal
        state instead of silent direct-mutation. fsm.py LEGAL_TRANSITIONS now
        carries STAGED→COLD, LOADING→COLD, LOADING_FAIL→POPPED, POPPED→COLD,
        ACTIVE→GRACE→POPPED→COLD, IDLE_HOT→COLD. Memory and DB state stay
        in sync (no drift where slot.state stays e.g. LOADING in Python
        while sqlite reads state='COLD').
        """
        # Walk legal hops to COLD per the new FSM table.
        # Worst case: ACTIVE → GRACE → POPPED → COLD (3 hops).
        if not is_terminal(slot.state):
            for _ in range(4):  # bounded — FSM diameter to COLD is 3
                if slot.state == SlotState.COLD:
                    break
                legal = LEGAL_TRANSITIONS.get(slot.state, set())
                if SlotState.COLD in legal:
                    transition(slot, SlotState.COLD)
                    break
                # Step toward COLD via the cheapest-distance hop.
                if SlotState.POPPED in legal:
                    transition(slot, SlotState.POPPED)
                elif SlotState.GRACE in legal:
                    transition(slot, SlotState.GRACE)
                elif SlotState.LOADING_FAIL in legal:
                    transition(slot, SlotState.LOADING_FAIL)
                else:
                    # No legal hop — terminal-park as COLD directly only as
                    # absolute last resort. Log for diagnostics.
                    log.warning(
                        "_force_cold: no legal hop from %s — direct-set COLD",
                        slot.state.value,
                    )
                    slot.state = SlotState.COLD
                    break
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
        # NEMO V2 2.1 fix: close() now returns the slots it cleared; fail
        # their completion_futures so callers get a clean CancelledError
        # instead of hanging until submit_and_wait timeout.
        cleared_slots = await self.queue.close()
        for cleared in cleared_slots:
            self._fail_completion_future(
                cleared,
                asyncio.CancelledError(
                    "manager shutdown -- slot was never processed"
                ),
            )
        # GRIP H-4 wire: tear down any idle holder so VRAM is released
        # and llama-server child is reaped on graceful shutdown.
        if self._idle_handle is not None:
            try:
                await self._teardown_idle_holder("shutdown")
            except Exception:
                log.exception(
                    "idle teardown during shutdown failed (best-effort)"
                )
        # HAUL P-4: release the TOCTOU-pinned binary fd on shutdown
        if self._binary_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._binary_fd)
            self._binary_fd = None
