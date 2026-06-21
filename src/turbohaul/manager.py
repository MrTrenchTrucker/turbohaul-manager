"""TurbohaulManager: top-level orchestrator wiring queue + subprocess + state + timers.

Per v0.2 ARCHITECTURE.md - orchestrates the whole lifecycle described in §6 state
machine. The foundational interface ships first; the full worker_loop streaming
implementation lands later alongside the API layer that forwards to llama-server.
"""
import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turbohaul.config import KEEP_ALIVE_MAX_S, BootConfig, RuntimeConfig
from turbohaul.fsm import LEGAL_TRANSITIONS, InvalidTransition, is_terminal, transition
from turbohaul.live_monitor import LiveOutputBuffer, idle_generation
from turbohaul.manifest import flags_to_argv, read_manifest
from turbohaul.queue import GraceTimer, IdleHotTimer, TurbohaulQueue
from turbohaul.safety import all_safety_gates
from turbohaul.singleton import boot_orphan_reaper, detect_foreign_gpu_apps, intra_lifetime_orphan_scan
from turbohaul.slot import Slot, SlotEvictedError, SlotState, derive_thread_id_prefix_hash
from turbohaul.state import (
    audit_db_session,
    known_active_pids,
    mark_slot_ended,
    open_state_db,
    reconcile_orphaned_slots,
    record_audit_event,
    state_db_session,
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
    - Drive the FSM via worker_loop (skeleton first; full streaming follows)
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
            max_consecutive_same_model=runtime.queue.max_consecutive_same_model,
            max_other_model_wait_s=runtime.queue.max_other_model_wait_s,
        )
        self.grace = GraceTimer(
            grace_seconds=runtime.queue.grace_seconds,
            max_extensions=runtime.queue.max_grace_extensions,
        )
        self.idle = IdleHotTimer(idle_seconds=runtime.queue.idle_hot_load_seconds)
        self._active_handle: SidecarHandle | None = None
        self._active_slot: Slot | None = None
        # Per-model concurrent dispatch (Design #1): the rider set for the
        # CURRENT anchor cycle when the active model's manifest declares
        # parallel>1. Element 0 is the anchor. Mutated ONLY by worker_loop
        # (append at fan-out admit, remove at per-rider drain, cleared after the
        # drain barrier); read await-free by status_snapshot. Stays [] for
        # parallel:1 models, so the single-mutator discipline above (no lock) is
        # preserved verbatim — worker_loop remains the sole writer and routes
        # only ever set their OWN slot.stream_done_event.
        self._inflight: list[Slot] = []
        # Manager-level idle holder (model warm post-grace).
        # When grace expires without a thread match, the sidecar is NOT
        # torn down -- it migrates here and stays alive for idle_seconds.
        # Next slot of same model_tag inherits the handle; different
        # model_tag tears it down first.
        self._idle_handle: SidecarHandle | None = None
        self._idle_model_tag: str | None = None
        self._idle_expires_at: float | None = None
        # Latest request's keep_alive intent across the ACTIVE_MATCH
        # chain on a single warm slot. Reset per anchor (_process_slot entry);
        # captured on ACTIVE for the anchor and on each ACTIVE_MATCH promotion
        # of a matched follow-up; consumed (cleared) at grace→idle entry. The
        # "latest request wins" rule mirrors Ollama keep_alive semantics
        # (timer resets on request receipt, not on response completion).
        # Without this, stale keep_alive from request N leaks into the IDLE_HOT
        # window computed after request N+M (a failure mode caught in review).
        self._latest_keep_alive_s: int | None = None
        # /status metrics counters for client-disconnect
        # eviction observability. Updated in worker_loop's is_evicted branch.
        self._eviction_count: int = 0
        self._last_evicted_at_iso: str | None = None
        # background sweeper state-row finalizer counters.
        # Finalizes evictions that landed audit-only on the hot path
        # (deferred state-row write to keep worker_loop off the SQLite
        # fsync stall). Sweeper is OFF the hot path — its sync SQL is fine.
        self._slots_finalized_lifetime: int = 0
        self._last_sweep_iso: str | None = None
        self._sweeper_task: asyncio.Task | None = None
        self._worker_task: asyncio.Task | None = None
        # Live inference monitor. Pure-observer plane:
        # worker_loop never reads/writes live_generation/live_output, so the
        # single-mutator FSM invariant is structurally untouched. _spawn_seq is
        # bumped by worker_loop at each _active_handle assignment so the poller
        # can detect a fixed-port (11500) sidecar swap across its httpx await.
        self._spawn_seq: int = 0
        self.live_generation: dict | None = None  # written ONLY by LiveSlotsPoller
        self.live_output = LiveOutputBuffer()      # fed ONLY by the streaming tee
        self._live_poller = None
        self._live_poller_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        # No self._lock: it had zero acquire sites (the single-task
        # worker_loop discipline protects _active_handle / _active_slot
        # mutations). If a future second mutating task lands, re-add a lock
        # purposefully and wrap the call sites that actually need it.
        self._binary_fd: int | None = None  # TOCTOU-pinned fd
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
        timeout_s: float = 7200.0,
        disconnect_event: "asyncio.Event | None" = None,
    ) -> tuple[Slot, Any]:
        """Submit + await the slot's completion. Returns (slot, completion_result).

        caller may pass a ``disconnect_event`` that the
        route's ``watch_disconnect`` task sets on client close. The queue
        pops with eviction-awareness; if the slot is evicted before activation,
        worker_loop fails the completion_future with SlotEvictedError which
        propagates out of this ``await`` for the caller to map to HTTP 499.
        """
        slot = await self.submit(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
            wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        try:
            result = await asyncio.wait_for(slot.completion_future, timeout=timeout_s)
        except asyncio.TimeoutError:
            raise
        return slot, result

    async def submit_for_streaming(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
        disconnect_event: "asyncio.Event | None" = None,
    ) -> Slot:
        """SSE streaming pass-through.

        Submit a request that will be consumed by an SSE-streaming route. Unlike
        submit_and_wait, this returns the slot immediately (without awaiting
        the completion_future). The slot has two pre-armed asyncio.Events:

          - ``slot.stream_ready_event``: worker_loop sets this once the slot
            reaches ACTIVE and the SidecarHandle is stored on
            ``slot.stream_handle``. The route awaits this before opening its
            own httpx.stream() to the sidecar's port.
          - ``slot.stream_done_event``: the route sets this when the stream
            closes (normal exhaustion, client disconnect, or error). Only then
            does worker_loop advance the slot ACTIVE → GRACE.

        client_meta MUST include ``"stream": True`` for worker_loop to
        recognise the streaming path. Existing non-streaming callers are
        unaffected.

        Design note: keeping the slot ACTIVE for the full stream lifetime
        prevents ACTIVE_MATCH from promoting a second submission against the
        same sidecar (single-slot invariant preserved).
        """
        slot = await self.submit(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
            wait_for_completion=True,  # still attach future so error paths surface
            disconnect_event=disconnect_event,
        )
        # Pre-arm the streaming coordination events. worker_loop will check for
        # client_meta["stream"] == True and use these instead of calling
        # self._complete_fn.
        slot.stream_ready_event = asyncio.Event()
        slot.stream_done_event = asyncio.Event()
        slot.stream_handle = None
        return slot

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
        # read + slot-write stay on state_db_session; audit-write uses pool.
        with state_db_session(self.boot.storage.state_db_path) as conn:
            stale_pids = known_active_pids(conn)
            live_pids = {pid for pid in stale_pids if check_alive(pid)}
            reconciled = reconcile_orphaned_slots(conn, live_pids)
        with audit_db_session(self.boot.storage.state_db_path) as conn:
            record_audit_event(
                conn,
                "boot_reconcile",
                {
                    "orphans_reaped": reap["reaped"],
                    "foreign_gpu_apps_count": len(foreign),
                    "slots_reconciled_to_cold": reconciled,
                },
            )

        return {
            "orphans_reaped": reap["reaped"],
            "orphans_failed": reap["failed"],
            "foreign_gpu_apps": foreign,
            "slots_reconciled_to_cold": reconciled,
        }

    def verify_binary(self) -> bool:
        """Verify + TOCTOU-pin llama_server_binary at boot (v0.2 §7.1).

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
        disconnect_event: "asyncio.Event | None" = None,
    ) -> Slot:
        """Accept a fresh inference request.

        - If thread_id is empty, auto-derive from a prompt-prefix hash.
        - If grace window is open for this thread+model → enqueue at FIFO HEAD
          + restart grace timer (max_grace_extensions cap applies).
        - Otherwise → normal FIFO enqueue.

        Raises RuntimeError if the manager is shutting down.
        """
        # Refuse new submissions after the shutdown signal so
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
        # caller-attached disconnect_event (lazy-init in
        # the route's own loop). None for non-HTTP callers
        # (BootInventory orphan-replay, internal probes, tests pre-attach).
        slot.disconnect_event = disconnect_event

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

        # Audit — slot-write stays on state_db_session; audit-write goes
        # through the pool wrapped in asyncio.to_thread (sync-only guard).
        with state_db_session(self.boot.storage.state_db_path) as conn:
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

        def _audit_submit() -> None:
            with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                record_audit_event(
                    audit_conn,
                    "submit",
                    {"thread_id_prefix": (thread_id or "")[:8], "model_tag": model_tag},
                    slot_id=slot.slot_id,
                )

        await asyncio.to_thread(_audit_submit)

        return slot

    # === Status snapshot =====================================================

    def status_snapshot(self) -> dict:
        """/status payload per v0.2 §9.3."""
        depth = self.queue.depth()

        active_info: dict | None = None
        loading_info: dict | None = None
        if self._active_slot is not None:
            slot = self._active_slot
            # FE LOADING transition fix: split status into ACTIVE vs the
            # pre-active transitional states (STAGED / PRE_LOADING /
            # LOADING / READY). Before this split, FE saw active=null
            # for the whole 5-30s cold-load window — reads as a hang.
            state_v = slot.state.value
            if state_v == "ACTIVE" or state_v == "ACTIVE_MATCH":
                if self._active_handle is not None:
                    active_info = {
                        "slot_id": slot.slot_id,
                        "model_tag": slot.model_tag,
                        "state": state_v,
                        # Redaction: only first 8 chars of thread_id exposed (v0.2 §11.1)
                        "thread_id_prefix": (slot.thread_id or "")[:8],
                        "pid": self._active_handle.pid,
                        "port": self._active_handle.port,
                    }
            elif state_v in {"STAGED", "PRE_LOADING", "LOADING", "READY"}:
                elapsed = 0.0
                started = getattr(slot, "started_loading_at", None) or getattr(slot, "received_at", None)
                if started is not None:
                    elapsed = max(0.0, time.monotonic() - started)
                loading_info = {
                    "slot_id": slot.slot_id,
                    "model_tag": slot.model_tag,
                    "state": state_v,
                    "thread_id_prefix": (slot.thread_id or "")[:8],
                    "elapsed_s": round(elapsed, 1),
                    "pid": self._active_handle.pid if self._active_handle else None,
                    "port": self._active_handle.port if self._active_handle else None,
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
        # /status idle snapshot reflects the manager-level
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
            "loading": loading_info,
            "grace": grace_info,
            "idle_hot": idle_info,
            # client-disconnect eviction observability.
            "evictions": {
                "total_lifetime": self._eviction_count,
                "last_evicted_at": self._last_evicted_at_iso,
            },
            # background sweeper that finalizes the
            # state-row for evictions (deferred from the hot path).
            # Sweeper runs every background_sweep_interval_s.
            "background_sweeper": {
                "last_sweep_iso": self._last_sweep_iso,
                "slots_finalized_lifetime": self._slots_finalized_lifetime,
            },
            "parallel_slots": {
                # Design #1: live in-flight rider count when a fan-out is active
                # (best-effort, await-free same-loop read), else 1 if a handle is
                # warm, else 0. `max` is the active sidecar's --parallel width
                # when known (handle.parallel), falling back to the process-count
                # config knob.
                "used": (
                    len(self._inflight)
                    if self._inflight
                    else (1 if self._active_handle else 0)
                ),
                "max": (
                    getattr(self._active_handle, "parallel", None)
                    or self.runtime.queue.max_parallel_sidecars
                ),
            },
            # Live inference monitor: tok/s + progress, written await-free by the
            # LiveSlotsPoller (idle default = single idle_generation() shape).
            # Counts/rates only — no prompt/response/IP/full-thread-id (the
            # 8-char generation_id is non-reversible).
            "generation": self.live_generation or idle_generation(),
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
            # Model-affinity hint for pop_next: the model currently warm in the
            # idle holder (preferred) or the active slot. This is a READ of the
            # manager scalars; a fire-and-forget _teardown_idle_holder task may
            # null _idle_handle/_idle_model_tag concurrently, but the read pair
            # below is AWAIT-FREE (atomic in the single-threaded event loop) and
            # `warm` is only a HINT -- a stale value just falls back to FIFO. No
            # new lock and no new mutator are introduced. warm=None => strict
            # FIFO (back-compat). _idle_model_tag is the confirmed attribute
            # (manager.__init__) for the idle holder's model tag.
            if self._idle_handle is not None:
                warm = self._idle_model_tag
            elif self._active_slot is not None:
                warm = self._active_slot.model_tag
            else:
                warm = None
            slot = await self.queue.pop_next(warm_model_tag=warm)
            if slot is None:
                # inline idle-expiry tick + fire-and-forget teardown
                # + identity-guarded debounce.
                # Capture `expires` into a local; only reset _idle_expires_at
                # if it is STILL the same object we observed. Prevents the race where a
                # concurrent reset (request promotion repopulates _idle_expires_at to a
                # fresh T+120 window) would otherwise be wiped by our stale-T0 debounce
                # → teardown fires on a legitimate fresh window → warm holder killed
                # mid-promotion.
                expires = self._idle_expires_at
                if (
                    self._idle_handle is not None
                    and expires is not None
                    and time.monotonic() >= expires
                ):
                    if self._idle_expires_at is expires:  # identity check
                        self._idle_expires_at = None
                        # Fire-and-forget — don't block worker_loop on
                        # the 5s SIGTERM grace + wait4 of the llama-server child.
                        asyncio.create_task(
                            self._teardown_idle_holder("idle_expired")
                        )
                await asyncio.sleep(0.05)
                continue

            # client-disconnect eviction handling.
            if slot.is_evicted:
                self._fail_completion_future(
                    slot,
                    SlotEvictedError(
                        f"slot {slot.slot_id} evicted: client disconnect"
                    ),
                )
                # Audit-emit via the audit pool path; NO sync
                # state_db_session(mark_slot_ended) on the hot path —
                # SQLite fsync 1-3s stalls would bypass the pool entirely.
                # State-row finalization defers to terminal-park /
                # background sweeper.
                try:
                    await self._audit_event_only_async(
                        slot.slot_id,
                        "slot_evicted",
                        {
                            "reason": "client_disconnect",
                            "time_in_queue_s": time.monotonic() - slot.created_at,
                        },
                    )
                except Exception:
                    log.exception(
                        "slot_evicted audit emit failed (best-effort)"
                    )
                # /status metric bookkeeping
                self._eviction_count += 1
                self._last_evicted_at_iso = datetime.now(
                    timezone.utc,
                ).isoformat()
                # Inline idle-tick mirror + identity guard — same
                # idle-tick block on the eviction branch so consecutive
                # evictions don't starve idle expiry.
                expires = self._idle_expires_at
                if (
                    self._idle_handle is not None
                    and expires is not None
                    and time.monotonic() >= expires
                ):
                    if self._idle_expires_at is expires:
                        self._idle_expires_at = None
                        asyncio.create_task(
                            self._teardown_idle_holder("idle_expired")
                        )
                continue

            try:
                await self._process_slot(slot)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("slot %s processing failed", slot.slot_id)
                self._fail_completion_future(slot, e)
                # Teardown active sidecar BEFORE force_cold to prevent a PID
                # leak. _force_cold only updates DB state; without teardown the
                # spawned llama-server keeps running and the single-slot
                # invariant breaks until boot_orphan_reaper at next restart.
                # Best-effort — don't let teardown failure mask the original
                # exception that triggered this path.
                if self._active_handle is not None:
                    try:
                        await self._teardown(slot, "worker-uncaught-exception")
                    except Exception:
                        log.exception(
                            "teardown during worker exception failed (best-effort)"
                        )
                await self._force_cold(slot, "worker-uncaught-exception")
        log.info("worker_loop exited")

    async def _fan_out_and_drain(
        self, anchor: Slot, handle, n_parallel: int,
    ) -> None:
        """Per-model concurrent dispatch (Design #1): serve up to ``n_parallel``
        same-model STREAMING requests concurrently on ONE shared sidecar.

        Reached ONLY from the ACTIVE branch when the active model's manifest
        declares parallel>1 (``n_parallel == handle.parallel``). parallel:1
        models never enter here -- they take the verbatim serial block, which
        stays byte-identical to today.

        Shape: a ONE-SHOT fan-out (admit up to ``n_parallel - 1`` extra riders
        from the queue exactly once -- never continuously refilling, so it
        cannot livelock the model swap) followed by a per-rider DRAIN barrier.
        Returns only when EVERY rider's route has genuinely finished its stream
        (``self._inflight`` empty), so the caller can safely advance
        ACTIVE->GRACE and (later) tear the sidecar down.

        Single-mutator discipline preserved: worker_loop is the sole writer of
        ``self._inflight``; routes only ever set their OWN
        ``slot.stream_done_event``. No lock, and the drain waiter tasks are
        owned + awaited here (cancelled on every exit), never detached.
        """
        # The anchor's mode decides the whole fan-out: STREAMING riders are
        # served by their own HTTP route (stream_ready/stream_done events);
        # NON-STREAMING riders by concurrent manager-owned _complete_fn calls.
        # Many sub-agent callers are non-streaming, so the non-streaming path is
        # the common one in practice. Riders must match the anchor's mode (the
        # admit loop enforces it; mixed-mode riders go back to the queue).
        anchor_streaming = bool(
            isinstance(anchor.client_meta, dict)
            and anchor.client_meta.get("stream", False)
        )
        # (1) The anchor is rider 0 -- already ACTIVE on `handle`.
        self._inflight = [anchor]
        anchor.stream_handle = handle
        if anchor_streaming and anchor.stream_ready_event is not None:
            anchor.stream_ready_event.set()
        keep_alives: list[int | None] = [
            (anchor.client_meta or {}).get("keep_alive_s")
        ]

        # (2) ONE-SHOT admit loop: pull up to n_parallel-1 MORE same-model
        # streaming riders and attach them to the SAME sidecar handle.
        while len(self._inflight) < n_parallel:
            extra = await self.queue.pop_next(warm_model_tag=anchor.model_tag)
            if extra is None:
                break  # queue has no more admissible riders right now
            extra_streaming = (
                isinstance(extra.client_meta, dict)
                and bool(extra.client_meta.get("stream", False))
            )
            if (
                extra.model_tag != anchor.model_tag
                or extra_streaming != anchor_streaming
            ):
                # Wrong model OR a different streaming-mode than the anchor: not
                # a rider for THIS fan-out. Push it back to the HEAD (the serial
                # path / next fan-out handles it) and stop -- this is one-shot.
                await self.queue.enqueue_head(extra)
                break
            if extra.is_evicted:
                # Client disconnected before admit -> do NOT burn a slot on it.
                self._fail_completion_future(
                    extra,
                    SlotEvictedError("client disconnected before fan-out admit"),
                )
                await self._audit_async(extra, "evicted_pre_fanout")
                continue
            # Promote the rider onto the shared sidecar. Drift-guard mirrors the
            # ACTIVE_MATCH promotion: if the slot's state drifted, reset its pid
            # to None FIRST (so _force_cold can NEVER sigterm the shared sidecar)
            # then cold-park it and drop it from the batch.
            extra.port = handle.port
            extra.pid = handle.pid
            try:
                transition(extra, SlotState.LOADING)
                transition(extra, SlotState.ACTIVE)
            except InvalidTransition as drift_err:
                log.warning(
                    "fan-out rider %s state drift (%s) -- dropping; %s",
                    extra.slot_id, extra.state.value, drift_err,
                )
                extra.pid = None  # MUST precede _force_cold: never reap shared sidecar
                self._fail_completion_future(extra, drift_err)
                await self._force_cold(
                    extra, f"fanout_rider_drift:{extra.state.value}"
                )
                continue
            extra.started_active_at = time.monotonic()
            extra.stream_handle = handle
            # Append to the rider set BEFORE signalling ready, so any observer
            # that sees the route unblock also sees the slot in self._inflight
            # (consistent state; keeps the drain-before-swap invariant exact).
            self._inflight.append(extra)
            keep_alives.append((extra.client_meta or {}).get("keep_alive_s"))
            await self._audit_async(extra, "fanout_rider_active")
            if anchor_streaming and extra.stream_ready_event is not None:
                extra.stream_ready_event.set()

        # keep_alive aggregate = MAX across riders (deterministic; the longest
        # warm-hold any rider asked for wins). Mirrors the serial anchor capture.
        _ka = [k for k in keep_alives if k is not None]
        self._latest_keep_alive_s = max(_ka) if _ka else None

        # (3) DISPATCH + DRAIN, branched on the anchor's mode. Non-streaming
        # riders (hermes sub-agents) are served by concurrent manager-owned
        # _complete_fn calls; streaming riders by their own routes (the barrier
        # below). Every admitted rider matches the anchor's mode.
        if not anchor_streaming:
            await self._drain_nonstreaming_riders(anchor, handle)
            return

        # (3-stream) DRAIN BARRIER. Each rider's route sets its OWN stream_done_event in
        # its finally (normal end, client-disconnect CancelledError, typed httpx
        # error, OR the route's own STREAM_TIMEOUT_S). We wait on the GENUINE
        # done-event -- never force-removing a rider while its route's httpx may
        # still be open (teardown gates on real route close, not an accounting
        # flag). A per-rider 3600s safety cap matches the
        # serial path's wait_for; on a cap, ONLY that rider is cooperatively
        # unwound -- siblings keep streaming. Returns when self._inflight is [].
        # Keyed by slot_id (Slot is a non-frozen dataclass and thus unhashable,
        # so it cannot be a dict key directly).
        waiters: dict[str, asyncio.Task] = {}
        for s in self._inflight:
            assert s.stream_done_event is not None
            waiters[s.slot_id] = asyncio.create_task(
                asyncio.wait_for(s.stream_done_event.wait(), timeout=3600.0)
            )
        try:
            pending = set(waiters.values())
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED,
                )
                for s in list(self._inflight):
                    w = waiters[s.slot_id]
                    if w not in done:
                        continue
                    # Distinguish a genuine stream_done from the 3600s cap.
                    if not w.cancelled() and isinstance(
                        w.exception(), asyncio.TimeoutError
                    ):
                        log.warning(
                            "fan-out rider %s hit 3600s drain cap; cooperative "
                            "unwind (siblings unaffected)", s.slot_id,
                        )
                        if (
                            s.stream_done_event is not None
                            and not s.stream_done_event.is_set()
                        ):
                            s.stream_done_event.set()
                    if (
                        s.completion_future is not None
                        and not s.completion_future.done()
                    ):
                        s.completion_future.set_result({"_streamed": True})
                    if s is not anchor:
                        # Rider terminal cleanup: the ANCHOR continues through
                        # _process_slot's normal GRACE/teardown flow, but a rider
                        # is done the instant its own route closes. Decouple it
                        # from the shared sidecar (pid=None so _force_cold can
                        # never reap the handle) and walk it to COLD so no zombie
                        # ACTIVE slot lingers.
                        s.pid = None
                        await self._force_cold(s, "fanout_rider_drained")
                    self._inflight.remove(s)
        finally:
            # On NORMAL completion self._inflight is already empty. On an
            # exception/cancel mid-drain it may still hold live riders -- set
            # their stream_done_event so their routes unwind (a truncated stream
            # when the sidecar is later reaped on the error path is acceptable
            # degraded behaviour, never a hang or zombie).
            for s in self._inflight:
                if (
                    s.stream_done_event is not None
                    and not s.stream_done_event.is_set()
                ):
                    s.stream_done_event.set()
            # No orphaned waiters: cancel + await every still-pending task.
            for w in waiters.values():
                if not w.done():
                    w.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await w
            self._inflight = []

    async def _drain_nonstreaming_riders(self, anchor: Slot, handle) -> None:
        """Non-streaming concurrent dispatch (Design #1): fire ``_complete_fn``
        for EVERY rider at once -- each is its own httpx POST to the sidecar's
        ``--parallel`` slots, so the engine serves them concurrently via
        continuous batching. Await all, resolve each ``completion_future`` with
        its result (or fail ONLY that rider on a typed sidecar error -- siblings
        keep going), terminal-clean the riders, and clear ``self._inflight``.
        Returns only when every rider has completed (drain-before-swap holds).
        This is the path non-streaming sub-agent callers take.
        """
        # One concurrent completion task per rider, keyed by slot_id (Slot is a
        # non-frozen dataclass and thus unhashable).
        ctasks: dict[str, asyncio.Task] = {
            s.slot_id: asyncio.create_task(self._complete_fn(s, handle))
            for s in self._inflight
        }
        try:
            pending = set(ctasks.values())
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED,
                )
                for s in list(self._inflight):
                    t = ctasks[s.slot_id]
                    if t not in done:
                        continue
                    if t.cancelled():
                        pass  # cancelled on teardown; future handled in finally
                    elif t.exception() is not None:
                        # A typed sidecar error fails ONLY this rider's caller.
                        self._fail_completion_future(s, t.exception())
                    else:
                        result = t.result()
                        if (
                            s.completion_future is not None
                            and not s.completion_future.done()
                        ):
                            s.completion_future.set_result(result)
                    if s is not anchor:
                        # Rider terminal cleanup (the anchor continues through
                        # _process_slot's GRACE/teardown). Decouple from the
                        # shared sidecar (pid=None) and walk to COLD.
                        s.pid = None
                        await self._force_cold(s, "fanout_rider_drained")
                    self._inflight.remove(s)
        finally:
            # Cancel + await any still-pending task; fail any unresolved rider
            # future so no caller hangs. Clear the rider set.
            for s in list(self._inflight):
                t = ctasks.get(s.slot_id)
                if t is not None and not t.done():
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await t
                if (
                    s.completion_future is not None
                    and not s.completion_future.done()
                ):
                    self._fail_completion_future(
                        s, RuntimeError("fan-out drain aborted"),
                    )
            self._inflight = []

    async def _process_slot(self, slot: Slot) -> None:
        """Drive one slot through STAGED → LOADING → ACTIVE → GRACE → POPPED."""
        self._active_slot = slot
        # Clear cross-slot keep_alive leakage. A previous slot's
        # ACTIVE_MATCH chain may have left a value here even though that
        # slot's grace→idle consumed it; defensive reset keeps the invariant
        # "value reflects this anchor cycle only" honest.
        self._latest_keep_alive_s = None

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

            # Stress-test hardening: pre-validate the
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
                await self._audit_async(slot, "stage_to_loading")
                transition(slot, SlotState.LOADING_FAIL)
                await self._audit_async(slot, "manifest_not_found")
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
            await self._audit_async(slot, "stage_to_loading")

            # Warm-inherit path. If the previous slot left
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
                self._spawn_seq += 1  # live-monitor: mark new active handle (sole writer = worker_loop)
                await self._audit_async(slot, "idle_hot_inherit")
                # Skip spawn + health wait; jump straight to LOADING -> ACTIVE.
                healthy = True
            else:
                # Different model OR no idle holder. If a stale holder
                # exists for a different model, tear it down before
                # spawning new -- immediate switch.
                if self._idle_handle is not None:
                    await self._teardown_idle_holder("model_swap")
                # Safety guardrails: pre-spawn host checks
                # (VRAM headroom + RAM + CPU load + IO wait). Refuse here
                # rather than spawning into an OOM / IO-stuck host.
                if self.runtime.queue.safety_enabled:
                    manifest_vram = 0
                    manifest_ctx = 0
                    manifest_gguf_bytes = 0
                    manifest_kv_quant = "f16"
                    manifest_no_kv_offload = False
                    manifest_parallel = 1
                    try:
                        m_for_vram = read_manifest(
                            self.boot.storage.manifests_path,
                            slot.model_tag,
                        )
                        manifest_vram = m_for_vram.expected_vram_bytes or 0
                        manifest_gguf_bytes = m_for_vram.gguf_size_bytes or 0
                        # ctx_size: prefer llama_server_flags.ctx_size (what
                        # actually gets passed to llama-server CLI); fall
                        # back to manifest.context_size.
                        manifest_ctx = (
                            m_for_vram.llama_server_flags.get("ctx_size")
                            or m_for_vram.context_size
                            or 0
                        )
                        # KV quant: derive from cache_type_k (cache_type_v
                        # assumed to match; if different, picks the larger).
                        manifest_kv_quant = (
                            m_for_vram.llama_server_flags.get("cache_type_k")
                            or "f16"
                        )
                        # --no-kv-offload: KV in host RAM, so the kv_cache_fit
                        # gate must NOT count it against VRAM (it re-checks RAM).
                        manifest_no_kv_offload = bool(
                            m_for_vram.llama_server_flags.get("no_kv_offload", False)
                        )
                        # parallel: extra concurrent llama.cpp slots add a flat
                        # per-slot compute floor to the VRAM gate (safety.py).
                        manifest_parallel = int(
                            m_for_vram.llama_server_flags.get("parallel", 1) or 1
                        )
                    except FileNotFoundError:
                        manifest_vram = 0
                    gates = all_safety_gates(
                        min_free_ram_mib=self.runtime.queue.safety_min_free_ram_mib,
                        min_free_vram_mib=self.runtime.queue.safety_min_free_vram_mib,
                        max_load_per_core=self.runtime.queue.safety_max_load_per_core,
                        max_iowait_percent=self.runtime.queue.safety_max_iowait_percent,
                        manifest_expected_vram_bytes=manifest_vram,
                        iowait_sample_window_s=self.runtime.queue.safety_iowait_sample_window_s,
                        ctx_size=manifest_ctx,
                        gguf_size_bytes=manifest_gguf_bytes,
                        kv_cache_quant=manifest_kv_quant,
                        no_kv_offload=manifest_no_kv_offload,
                        parallel=manifest_parallel,
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
                        await self._audit_async(slot, "safety_gate_refused")
                        await self._audit_event_only_async(
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
                self._spawn_seq += 1  # live-monitor: mark new active handle (sole writer = worker_loop)
                # LOADING → ACTIVE (or LOADING_FAIL → POPPED)
                healthy = await self._wait_healthy(
                    port, self.runtime.queue.loading_health_timeout_s,
                    is_alive=handle.is_alive,
                )
            if not healthy:
                transition(slot, SlotState.LOADING_FAIL)
                await self._audit_async(slot, "loading_fail_health_timeout")
                transition(slot, SlotState.POPPED)
                await self._teardown(slot, "loading-fail-health-timeout")
                self._fail_completion_future(slot, RuntimeError("loading-fail-health-timeout"))
                return

            # Per-model concurrent-dispatch admission cap (Design #1). Pinned on
            # the handle at spawn from the actual --parallel argv, so it is
            # drift-proof vs a fresh manifest read across a warm-inherit reuse
            # cycle. NOTE: the cap is handle.parallel DIRECTLY -- NOT
            # max_parallel_sidecars (that config is the separate multi-PROCESS
            # stub, default 1; clamping to it would disable fan-out entirely).
            # For parallel:1 models this is 1, so the serial path below is taken
            # verbatim and byte-identical to today.
            n_parallel = max(1, getattr(handle, "parallel", 1))

            transition(slot, SlotState.ACTIVE)
            slot.started_active_at = time.monotonic()
            await self._audit_async(slot, "active")
            # Capture anchor's keep_alive intent for the grace→idle decision.
            self._latest_keep_alive_s = (slot.client_meta or {}).get("keep_alive_s")

            # Branch on streaming mode.
            #
            # Non-streaming (existing): await self._complete_fn(slot, handle) to
            # post chat-completion, set completion_future, advance to GRACE.
            #
            # Streaming: client_meta["stream"] is True AND submit_for_streaming()
            # pre-armed slot.stream_ready_event + slot.stream_done_event. We
            # SKIP _complete_fn (the route owns the httpx streaming connection).
            # Instead: hand the SidecarHandle to the route via slot.stream_handle,
            # signal stream_ready_event so the route can open its httpx.stream(),
            # then BLOCK here on stream_done_event until the route reports the
            # stream has finished (normal close, client disconnect, or error).
            # Slot stays in ACTIVE the entire time so ACTIVE_MATCH cannot promote
            # a second submission against the same sidecar (a critical review
            # catch — single-slot invariant preserved).
            is_streaming = (
                isinstance(slot.client_meta, dict)
                and bool(slot.client_meta.get("stream", False))
                and slot.stream_ready_event is not None
                and slot.stream_done_event is not None
            )
            if n_parallel > 1:
                # Design #1: concurrent fan-out. Serve up to n_parallel
                # same-model riders (streaming OR non-streaming) on this ONE
                # shared sidecar, then drain them all before advancing to GRACE.
                # Reached ONLY for parallel:N models; parallel:1 falls to the
                # verbatim serial block below (byte-identical to today).
                await self._fan_out_and_drain(slot, handle, n_parallel)
            elif is_streaming:
                # Hand the sidecar handle to the route, signal ready.
                slot.stream_handle = handle
                slot.stream_ready_event.set()
                # Wait for the route to finish streaming. Stream timeout is
                # bounded — same default as non-streaming complete_fn — but
                # very long in practice (1h+ for slow-thinking models on big
                # context). If the route never signals, worker_loop unblocks
                # via timeout and proceeds to GRACE (slot already drained).
                try:
                    await asyncio.wait_for(
                        slot.stream_done_event.wait(),
                        timeout=3600.0,  # 1 hour cap; routes typically signal in seconds
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "streaming slot %s exceeded 3600s waiting for stream_done_event; "
                        "advancing to GRACE anyway",
                        slot.slot_id,
                    )
                # Resolve the completion_future so any caller awaiting the
                # slot (e.g. tests or programmatic await) is unblocked.
                if (
                    slot.completion_future is not None
                    and not slot.completion_future.done()
                ):
                    slot.completion_future.set_result({"_streamed": True})
            else:
                # Non-streaming path (existing behaviour, unchanged).
                # Completion (Phase 3 wires httpx forward; Phase 2 default is noop)
                result = await self._complete_fn(slot, handle)
                if slot.completion_future is not None and not slot.completion_future.done():
                    slot.completion_future.set_result(result)

            # Drain-before-swap hard gate (Design #1): every fan-out rider must
            # have finished before the anchor advances to GRACE / the sidecar is
            # torn down. _fan_out_and_drain returns only when self._inflight is
            # empty; for parallel:1 it is never populated, so this is a no-op.
            assert not self._inflight, (
                "Design #1 invariant: riders must drain before ACTIVE->GRACE"
            )
            # ACTIVE → GRACE
            transition(slot, SlotState.GRACE)
            slot.grace_started_at = time.monotonic()
            self.grace.start(slot.thread_id, slot.model_tag)
            await self._audit_async(slot, "grace_enter")

            # Wait for grace window OR promote a matched staging slot via
            # ACTIVE_MATCH (warm-slot reuse). Per v0.2 §6 FSM; this transition
            # cascades same-(thread_id, model_tag) follow-up requests through
            # the warm llama-server without re-spawn.
            deadline = time.monotonic() + self.runtime.queue.grace_seconds
            while time.monotonic() < deadline and not self._stop_event.is_set():
                # Atomic find + remove in one lock acquire
                matched = await self.queue.pop_matched_thread(
                    slot.thread_id, slot.model_tag
                )
                if matched is not None:
                    matched.port = handle.port
                    matched.pid = handle.pid
                    self._active_slot = matched
                    # State-drift guard. If matched.state
                    # drifted between find_matched_thread + here (concurrent
                    # reconcile, retry path, etc.), transition raises
                    # InvalidTransition which would crash worker_loop. Wrap
                    # the promotion + park-on-drift instead of propagate.
                    try:
                        transition(matched, SlotState.ACTIVE_MATCH)
                        await self._audit_async(matched, "active_match_promoted")
                        transition(matched, SlotState.ACTIVE)
                        # Each matched-follow-up's keep_alive overrides
                        # the anchor's for the next grace→idle calculation. Mirrors
                        # Ollama "timer resets on request receipt" rule.
                        self._latest_keep_alive_s = (
                            matched.client_meta or {}
                        ).get("keep_alive_s")
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
                    # Streaming-path warm-reuse. When a streaming submit lands on an
                    # already-active matched slot, the HTTP route owns the upstream connection via
                    # stream_handle. Worker MUST NOT call _complete_fn (would open a 2nd sidecar
                    # connection and violate the single-slot invariant). Hand
                    # off via stream_ready_event and block on stream_done_event until route drains.
                    # Prior bug: this branch unconditionally called _complete_fn → matched slot's
                    # stream_ready_event was never set → route's SLOT_READY_TIMEOUT_S fired at 600s
                    # every turn ≥ 2 of a multi-tool-call agent loop.
                    matched_is_streaming = bool(
                        isinstance(matched.client_meta, dict)
                        and matched.client_meta.get("stream", False)
                    )
                    try:
                        if matched_is_streaming:
                            assert (
                                matched.stream_ready_event is not None
                                and matched.stream_done_event is not None
                            ), (
                                f"streaming slot {matched.slot_id} missing events at ACTIVE_MATCH promotion"
                            )
                            matched.stream_handle = handle
                            matched.stream_ready_event.set()
                            try:
                                await asyncio.wait_for(
                                    matched.stream_done_event.wait(),
                                    timeout=3600.0,
                                )
                            except asyncio.TimeoutError:
                                log.warning(
                                    "active_match streaming slot %s exceeded 3600s waiting for stream_done_event",
                                    matched.slot_id,
                                )
                            if matched.completion_future is not None and not matched.completion_future.done():
                                matched.completion_future.set_result({"_streamed": True})
                        else:
                            result2 = await self._complete_fn(matched, handle)
                            if (
                                matched.completion_future is not None
                                and not matched.completion_future.done()
                            ):
                                matched.completion_future.set_result(result2)
                    except asyncio.CancelledError:
                        # Cooperatively unwind the route's blocking httpx call
                        # by signaling stream_done before terminal-park (avoids zombie
                        # route + dead slot drift).
                        if (
                            matched_is_streaming
                            and matched.stream_done_event is not None
                            and not matched.stream_done_event.is_set()
                        ):
                            matched.stream_done_event.set()
                        # Cancellation mid-ACTIVE_MATCH must terminal-park the
                        # matched slot so it does not rot as a zombie ACTIVE row
                        # in state.sqlite, then re-raise so worker_loop's
                        # teardown runs cleanly.
                        self._fail_completion_future(
                            matched,
                            asyncio.CancelledError("shutdown during active_match"),
                        )
                        try:
                            transition(matched, SlotState.POPPED)
                        except InvalidTransition:
                            pass
                        await self._audit_async(matched, "active_match_cancelled")
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
                                "active_match cleanup mark_slot_ended failed for %s",
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
                    # On completion failure, skip grace pretense
                    # — go ACTIVE → GRACE → POPPED + mark failed, keep state
                    # machine honest. (transition validates each hop.)
                    transition(matched, SlotState.GRACE)
                    if completed_ok:
                        await self._audit_async(matched, "active_match_to_grace")
                        if self.grace.restart_for_followup():
                            # Also bump per-slot extension_count
                            # (was always 0 in sqlite — only GraceTimer's was).
                            matched.extension_count = self.grace.extension_count
                            deadline = time.monotonic() + self.runtime.queue.grace_seconds
                            await self._audit_event_only_async(
                                matched.slot_id,
                                "grace_extended_via_active_match",
                                {"extension_count": self.grace.extension_count},
                            )
                    else:
                        await self._audit_async(matched, "active_match_failed")
                    # Matched slot's request is done; its sidecar was the
                    # anchor's warm process (reused, not its own). Anchor `slot`
                    # remains the GRACE driver until grace expiry.
                    transition(matched, SlotState.POPPED)
                    await self._audit_async(matched, "active_match_completed" if completed_ok else "active_match_failed_terminal")
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
            # Hold the sidecar in idle for follow-up reuse
            # by any same-model_tag request inside idle_hot_load_seconds.
            # When idle_seconds == 0 (test default), this is equivalent to
            # immediate teardown -- preserves "grace-expired" reason on the
            # mark_slot_ended audit (backward-compat with existing tests).
            #
            # Honor the latest request's keep_alive intent as an IDLE_HOT extension.
            # `_latest_keep_alive_s` was set on the anchor's ACTIVE and refreshed
            # on each ACTIVE_MATCH promotion — so it reflects the most recent
            # request that touched the warm slot (Ollama timer-resets-on-receipt
            # semantics). After consumption it's cleared so a stale value can't
            # leak into the next anchor cycle.
            keep_alive_s = self._latest_keep_alive_s
            default_idle = self.runtime.queue.idle_hot_load_seconds
            if keep_alive_s is None:
                idle_seconds = default_idle
            elif keep_alive_s < 0:
                # Ollama -1 = "pin until VRAM pressure"; we cap at KEEP_ALIVE_MAX_S
                # (never indefinite on a single GPU).
                idle_seconds = KEEP_ALIVE_MAX_S
            else:
                # 0 falls through this expression cleanly → idle disabled.
                idle_seconds = min(keep_alive_s, KEEP_ALIVE_MAX_S)
            ka_clamped = (
                keep_alive_s is not None
                and keep_alive_s >= 0
                and keep_alive_s > KEEP_ALIVE_MAX_S
            )
            # Consumed — clear before any further decisions so the next anchor
            # starts cleanly (defense-in-depth on top of _process_slot reset).
            self._latest_keep_alive_s = None
            if idle_seconds > 0 and self._active_handle is not None:
                # Hand off the active handle to the manager-level idle holder.
                self._idle_handle = self._active_handle
                self._idle_model_tag = slot.model_tag
                self._idle_expires_at = time.monotonic() + idle_seconds
                await self._audit_event_only_async(
                    slot.slot_id,
                    "idle_hot_enter",
                    {
                        "model_tag": slot.model_tag,
                        "idle_seconds": idle_seconds,
                        # Audit visibility into when client keep_alive
                        # overrode the default + when the cap fired.
                        "keep_alive_requested": keep_alive_s,
                        "keep_alive_clamped": ka_clamped,
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
                await self._audit_event_only_async(
                    slot.slot_id,
                    "idle_hot_enter",
                    {"model_tag": slot.model_tag},
                )
        finally:
            # If unwind reaches here with a live handle, the IDLE_HOT
            # entry did NOT complete (most often CancelledError during
            # shutdown or mid-_complete_fn). MUST teardown the handle,
            # not just drop the reference, or llama-server orphans with
            # the full model in VRAM and no parent reference anywhere.
            #
            # Two-part safeguard:
            # 1. Diagnostic log at entry — surfaces the leak path under repro.
            # 2. Do NOT null _active_handle until sigterm SUCCEEDS. If the
            #    sigterm helper raises (drained_sigterm internal failure,
            #    process already dead/zombie, etc.) leaving _active_handle
            #    set lets worker_loop's per-slot exception handler
            #    fire its safety-net _teardown(). Otherwise the
            #    null-before-success ordering bypassed that safety net.
            handle_to_reap = self._active_handle
            self._active_slot = None
            log.warning(
                "process_slot finally reached: slot=%s active_handle_pid=%s alive=%s idle_match=%s",
                getattr(slot, "slot_id", "?"),
                getattr(handle_to_reap, "pid", None) if handle_to_reap is not None else None,
                handle_to_reap.is_alive() if handle_to_reap is not None else False,
                handle_to_reap is self._idle_handle if handle_to_reap is not None else False,
            )
            # Skip defensive sigterm if the handle was promoted to the
            # IDLE_HOT holder — that promotion is by design; killing it
            # would defeat the warm-hold purpose.
            if (
                handle_to_reap is not None
                and handle_to_reap is not self._idle_handle
                and handle_to_reap.is_alive()
            ):
                sigterm_ok = False
                try:
                    await asyncio.shield(
                        self._sigterm(
                            handle_to_reap,
                            drained_window_s=float(
                                self.runtime.queue.drained_sigterm_window_active_s
                            ),
                            is_active=False,
                            cold_window_s=float(
                                self.runtime.queue.drained_sigterm_window_cold_s
                            ),
                        )
                    )
                    sigterm_ok = True
                except Exception:
                    log.exception(
                        "cancellation-unwind teardown FAILED — leaving "
                        "_active_handle set so worker_loop safety-net can retry"
                    )
                if sigterm_ok:
                    self._active_handle = None
                # else: keep _active_handle so worker_loop's except handler
                # (which calls _teardown with reason="worker-uncaught-exception")
                # has a second chance to reap. If that ALSO fails, the
                # intra_lifetime_orphan_scan on the next /ensure tick is
                # the final safety net (singleton.py).
            else:
                # Handle absent or promoted to idle_holder — safe to null.
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
            # Dynamic expected_drop_mib derived from manifest
            # expected_vram_bytes. Was hardcoded 1024 MiB — that let a 921 MiB
            # drop "verify clear" while a 17 GiB model was still resident.
            expected_drop_mib = self._compute_expected_drop_mib(slot.model_tag)
            await self._vram_verify(
                expected_drop_mib=expected_drop_mib, timeout_s=30.0,
            )
            # Scan for grandchild orphans left behind by the upstream
            # setsid-detach (killpg never reached them) and reap before the
            # next slot needs the port + VRAM. ~50ms /proc walk; cheap to run
            # on every teardown.
            orphan_reaped = 0
            try:
                orphan_reap_result = boot_orphan_reaper(
                    port_base=self.boot.runtime.default_port_base,
                    known_pids=set(),  # single-slot mode; a future multi-slot
                                       # path will pass live sidecar pids
                )
                orphan_reaped = orphan_reap_result.get("reaped", 0)
            except Exception:
                log.exception(
                    "post-teardown orphan reap failed (best-effort)"
                )
            # Intra-lifetime port-bound reaper. Catches
            # orphans whose parent IS still the running manager (PPid !=
            # 1 so boot_orphan_reaper misses them) — e.g. handle dropped
            # without sigterm via lost reference or finally-clear bug.
            try:
                live_pids = self._live_handle_pids()
                il_result = intra_lifetime_orphan_scan(
                    port_base=self.boot.runtime.default_port_base,
                    known_handle_pids=live_pids,
                )
                if il_result.get("reaped", 0) > 0:
                    log.warning(
                        "intra-lifetime reap caught orphans post-teardown: %s",
                        il_result,
                    )
            except Exception:
                log.exception(
                    "intra-lifetime orphan scan failed (best-effort)"
                )
            # slot-write stays on state_db_session; audit-write goes
            # through the pool wrapped in asyncio.to_thread (sync-only).
            with state_db_session(self.boot.storage.state_db_path) as conn:
                mark_slot_ended(conn, slot.slot_id, reason)

            def _audit_teardown() -> None:
                with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                    record_audit_event(
                        audit_conn,
                        "teardown",
                        {
                            "reason": reason,
                            "sigterm_status": status,
                            "sigterm_ok": ok,
                            "post_teardown_orphans_reaped": orphan_reaped,
                        },
                        slot_id=slot.slot_id,
                    )

            await asyncio.to_thread(_audit_teardown)
            # Clear _active_handle after successful teardown so the outer
            # finally's defensive sigterm net does not double-fire on normal
            # flow. Owner contract: "if you called _teardown you have handed
            # off the handle."
            self._active_handle = None

    async def _teardown_idle_holder(self, reason: str) -> None:
        """Tear down the manager-level idle handle.

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
        # Dynamic expected_drop_mib for idle holder teardown.
        expected_drop_mib = self._compute_expected_drop_mib(
            model_tag or ""
        )
        await self._vram_verify(
            expected_drop_mib=expected_drop_mib, timeout_s=30.0,
        )
        try:
            boot_orphan_reaper(
                port_base=self.boot.runtime.default_port_base,
                known_pids=set(),
            )
        except Exception:
            log.exception(
                "idle-holder orphan reap failed (best-effort)"
            )
        # Intra-lifetime port-bound reaper here too.
        try:
            live_pids = self._live_handle_pids()
            intra_lifetime_orphan_scan(
                port_base=self.boot.runtime.default_port_base,
                known_handle_pids=live_pids,
            )
        except Exception:
            log.exception(
                "intra-lifetime orphan scan failed (best-effort)"
            )
        # audit-only write via pool, wrapped to_thread (sync-only).
        def _audit_idle_holder() -> None:
            with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                record_audit_event(
                    audit_conn,
                    "teardown_idle_holder",
                    {
                        "reason": reason,
                        "model_tag": model_tag,
                        "sigterm_status": status,
                        "sigterm_ok": ok,
                    },
                )

        await asyncio.to_thread(_audit_idle_holder)

    async def _force_cold(self, slot: Slot, reason: str) -> None:
        """Mark a slot COLD when processing dies mid-flight.

        Walk legal transitions to COLD from any non-terminal
        state instead of silent direct-mutation. fsm.py LEGAL_TRANSITIONS now
        carries STAGED→COLD, LOADING→COLD, LOADING_FAIL→POPPED, POPPED→COLD,
        ACTIVE→GRACE→POPPED→COLD, IDLE_HOT→COLD. Memory and DB state stay
        in sync (no drift where slot.state stays e.g. LOADING in Python
        while sqlite reads state='COLD').

        Defensive teardown of any live handle attached
        to this slot BEFORE forcing COLD. Closes the footgun where a
        caller forgot to teardown (the active_match_cancelled path
        raises after _force_cold(matched, ...) without sigterm'ing the
        anchor sidecar). Defense-in-depth.

        Annotate the audit with pid_source so post-hoc tooling can
        distinguish matched-row-on-anchor-pid (anchor_shared) from genuine
        standalone (self).
        """
        # Identify pid_source, defensive sigterm if the slot owns its own
        # live handle.
        pid_source = "self"
        if (
            self._active_handle is not None
            and slot.pid
            and slot.pid == self._active_handle.pid
            and len(self._inflight) > 1
        ):
            # Design #1 hard requirement: the sidecar is serving MULTIPLE
            # concurrent fan-out riders. Force-colding ONE of them (the ANCHOR or a rider)
            # must NEVER sigterm the shared handle while siblings are still
            # streaming. Identity-INDEPENDENT: gate on shared-pid + rider-set
            # size, NOT slot_id (the anchor IS _active_slot, so the slot_id
            # check below would let it fall through to a teardown). Belt-and-
            # suspenders with the admit-path pid=None reset on rider drift.
            pid_source = "fanout_shared_no_teardown"
        elif (
            self._active_slot is not None
            and slot.slot_id != self._active_slot.slot_id
            and self._active_handle is not None
            and slot.pid
            and slot.pid == self._active_handle.pid
        ):
            # matched.pid == anchor.pid via shared warm sidecar
            # (active_match drift path). Do NOT teardown — anchor owns it.
            pid_source = "anchor_shared"
        elif (
            slot.pid
            and self._active_handle is not None
            and slot.pid == self._active_handle.pid
        ):
            # Slot owns the active handle. Defensive teardown.
            try:
                await self._sigterm(
                    self._active_handle,
                    drained_window_s=float(
                        self.runtime.queue.drained_sigterm_window_active_s
                    ),
                    is_active=False,
                    cold_window_s=float(
                        self.runtime.queue.drained_sigterm_window_cold_s
                    ),
                )
                self._active_handle = None
            except Exception:
                log.exception(
                    "_force_cold defensive teardown failed (best-effort)"
                )

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
        # slot-write stays on state_db_session; audit-write via pool.
        with state_db_session(self.boot.storage.state_db_path) as conn:
            mark_slot_ended(conn, slot.slot_id, reason)

        def _audit_force_cold() -> None:
            with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                # pid_source annotation
                record_audit_event(
                    audit_conn,
                    "force_cold",
                    {"reason": reason, "pid_source": pid_source},
                    slot_id=slot.slot_id,
                )

        await asyncio.to_thread(_audit_force_cold)

    def _compute_expected_drop_mib(self, model_tag: str) -> int:
        """Dynamic expected_drop_mib derived from the manifest.

        Returns max(2048, manifest.expected_vram_bytes/1024**2). Floor
        at 2 GiB absorbs page-cache noise on small models. If manifest
        cannot be read (race during teardown of a just-deleted manifest),
        falls back to 2048 MiB rather than blocking teardown.
        """
        try:
            m = read_manifest(self.boot.storage.manifests_path, model_tag)
            return max(2048, int(m.expected_vram_bytes / (1024 * 1024)))
        except Exception:
            log.warning(
                "_compute_expected_drop_mib: manifest read failed for %s, "
                "falling back to 2048 MiB",
                model_tag,
            )
            return 2048

    def _live_handle_pids(self) -> set[int]:
        """Helper: set of currently-known live llama-server pids.

        Used by intra_lifetime_orphan_scan to distinguish managed
        sidecars from leaked orphans on our port range. Includes
        _active_handle and _idle_handle if alive.
        """
        live: set[int] = set()
        if (
            self._active_handle is not None
            and self._active_handle.is_alive()
        ):
            live.add(self._active_handle.pid)
        if (
            self._idle_handle is not None
            and self._idle_handle.is_alive()
        ):
            live.add(self._idle_handle.pid)
        return live

    async def _audit_async(self, slot: Slot, event_type: str) -> None:
        """Async-context wrapper for `_audit`.

        audit_db_session is sync-only. When called from an
        async function (e.g. `_process_slot`), the sync `_audit` must be
        offloaded to a worker thread or the sync-only guard raises RuntimeError.
        Call this from async code; call `_audit` directly from sync code.
        """
        await asyncio.to_thread(self._audit, slot, event_type)

    async def _audit_event_only_async(
        self,
        slot_id: str,
        event_type: str,
        payload: dict | None = None,
    ) -> None:
        """Async-context wrapper for `_audit_event_only`. See `_audit_async`."""
        await asyncio.to_thread(
            self._audit_event_only, slot_id, event_type, payload
        )

    def _audit(self, slot: Slot, event_type: str) -> None:
        """Audit: upsert current slot row + record event + publish to event bus.

        slot-write stays on state_db_session; audit-write via pool. Both
        calls are sync (this method is `def`, not `async def`), so the
        audit_db_session sync-only guard is satisfied without to_thread.

        ★ do NOT call this directly from async code — use
        `_audit_async` instead (the sync-only guard catches direct calls from a
        running event loop).
        """
        with state_db_session(self.boot.storage.state_db_path) as conn:
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
        with audit_db_session(self.boot.storage.state_db_path) as conn:
            record_audit_event(conn, event_type, {"state": slot.state.value}, slot_id=slot.slot_id)
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
        to clobber that state. Routed through audit pool (sync call).
        """
        with audit_db_session(self.boot.storage.state_db_path) as conn:
            record_audit_event(conn, event_type, payload or {}, slot_id=slot_id)

    # === background sweeper =================================================

    async def _periodic_terminal_park_sweep(self) -> None:
        """Periodically finalize the state-row for client-disconnect evictions.

        Deferred state-row finalization OFF the
        worker_loop hot path to avoid the 1-3s SQLite fsync stall that
        bypassed the audit pool. Audit-emit fires immediately via
        ``_audit_event_only_async``; the state-row finalize lands here.

        Loop: every ``background_sweep_interval_s`` (default 60s — matches
        the audit pool rhythm), the sweeper queries for slots that are:
          - in STAGED state (never reached ACTIVE)
          - older than ``background_sweep_min_age_s`` (default 24h)
          - have no live ``pid`` (no sidecar attached)
          - have ``ended_at IS NULL`` (not already terminal-parked)
        Each match is mark_slot_ended'd with reason
        ``background_sweeper_evicted``. mark_slot_ended sets state=COLD +
        ended_at=now, so the SELECT predicate becomes false on the next
        sweep — naturally idempotent.

        Triple-gate (state=STAGED + age + pid IS NULL) eliminates false
        positives. An in-flight slot has either a non-NULL pid or has
        already transitioned past STAGED; either disqualifies it from the
        sweep. The 24h staleness floor is defense-in-depth — even if a
        pid-check edge case slips through, a 24h+ STAGED-state slot is
        overwhelmingly likely to be a client-disconnect eviction casualty.
        """
        interval = max(1, int(self.runtime.queue.background_sweep_interval_s))
        min_age = max(60, int(self.runtime.queue.background_sweep_min_age_s))
        log.info(
            "periodic_terminal_park_sweep started (interval=%ds, min_age=%ds)",
            interval, min_age,
        )
        while not self._stop_event.is_set():
            try:
                await self._run_one_sweep(min_age)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "periodic_terminal_park_sweep iteration failed (best-effort)"
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval,
                )
                # _stop_event fired during sleep — exit cleanly
                break
            except asyncio.TimeoutError:
                continue  # normal cadence tick — run next sweep
        log.info("periodic_terminal_park_sweep exited")

    async def _run_one_sweep(self, min_age_s: int) -> int:
        """Run one sweep iteration. Returns count of slots finalized.

        Synchronous SQL inside ``state_db_session`` — OK because this method
        runs on the background sweeper task, NOT the worker_loop hot path.
        """
        from datetime import timedelta
        cutoff_iso = (
            datetime.now(timezone.utc) - timedelta(seconds=min_age_s)
        ).isoformat(timespec="seconds")
        finalized: list[str] = []
        with state_db_session(self.boot.storage.state_db_path) as conn:
            cur = conn.execute(
                """SELECT slot_id FROM slots
                   WHERE state = 'STAGED'
                     AND created_at < ?
                     AND pid IS NULL
                     AND ended_at IS NULL
                   LIMIT 100""",  # batch cap: bounds writer-lock hold time under a storm pattern
                (cutoff_iso,),
            )
            stale_slot_ids = [row["slot_id"] for row in cur.fetchall()]
            for slot_id in stale_slot_ids:
                mark_slot_ended(conn, slot_id, "background_sweeper_evicted")
                finalized.append(slot_id)
        # Counters + /status surfacing (single per-run update, not per-row)
        self._slots_finalized_lifetime += len(finalized)
        self._last_sweep_iso = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
        # Single audit per sweep run keeps audit volume bounded under storm.
        try:
            await self._audit_event_only_async(
                None,
                "background_sweeper_run",
                {
                    "slots_finalized": len(finalized),
                    "sweep_ts": self._last_sweep_iso,
                },
            )
        except Exception:
            log.exception(
                "background_sweeper_run audit emit failed (best-effort)"
            )
        return len(finalized)

    # === Shutdown ============================================================

    async def shutdown(self) -> None:
        """Clean tear-down. Stops worker loop + sweeper + drains queue + closes state db."""
        self._stop_event.set()
        # Live monitor: stop the OBSERVER first so it cannot issue a /slots probe
        # against a sidecar being torn down, nor publish a generation tick after
        # the worker_loop is gone (a high-severity review catch).
        if self._live_poller_task is not None and not self._live_poller_task.done():
            self._live_poller_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._live_poller_task
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        # cancel + await the background sweeper task.
        if self._sweeper_task is not None and not self._sweeper_task.done():
            self._sweeper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._sweeper_task
        # close() now returns the slots it cleared; fail their
        # completion_futures so callers get a clean CancelledError instead of
        # hanging until the submit_and_wait timeout.
        cleared_slots = await self.queue.close()
        for cleared in cleared_slots:
            self._fail_completion_future(
                cleared,
                asyncio.CancelledError(
                    "manager shutdown -- slot was never processed"
                ),
            )
        # Tear down any idle holder so VRAM is released
        # and the llama-server child is reaped on graceful shutdown.
        if self._idle_handle is not None:
            try:
                await self._teardown_idle_holder("shutdown")
            except Exception:
                log.exception(
                    "idle teardown during shutdown failed (best-effort)"
                )
        # Release the TOCTOU-pinned binary fd on shutdown
        if self._binary_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._binary_fd)
            self._binary_fd = None
