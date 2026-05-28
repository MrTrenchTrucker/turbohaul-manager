"""TurbohaulQueue: two-tier (unbounded acceptance buffer + capped staging) + grace/idle timers.

Per v0.2 ARCHITECTURE.md §5 + §6.
"""
import asyncio
import logging
import time
from collections import deque

from turbohaul.slot import Slot, SlotState

log = logging.getLogger(__name__)


class QueueClosed(RuntimeError):
    pass


class QueueFull(RuntimeError):
    pass


class TurbohaulQueue:
    """Two-tier queue.

    - Acceptance buffer: capped at acceptance_max (default 10k). Receives all fresh
      requests. Never blocks the API caller until cap hit.
    - Staging queue: capped at staging_max (default 100). FIFO.
    - On enqueue: slot goes to staging if room, else to acceptance buffer.
    - On pop: drain from staging head; buffer feeds staging tail when staging has room.
    """

    def __init__(self, staging_max: int = 100, acceptance_max: int = 10000) -> None:
        self.staging_max = staging_max
        self.acceptance_max = acceptance_max
        self._accept_buf: deque[Slot] = deque()
        self._staging: deque[Slot] = deque()
        self._lock = asyncio.Lock()
        self._closed = False

    async def enqueue(self, slot: Slot) -> None:
        """Add a fresh slot. Promotes to staging if room; else accept-buffer."""
        if self._closed:
            raise QueueClosed("queue closed")
        async with self._lock:
            if len(self._staging) < self.staging_max:
                slot.state = SlotState.STAGED
                self._staging.append(slot)
                return
            if len(self._accept_buf) >= self.acceptance_max:
                raise QueueFull(
                    f"acceptance buffer at max {self.acceptance_max}"
                )
            slot.state = SlotState.ACCEPT_BUFFER
            self._accept_buf.append(slot)

    def _pop_first_non_evicted_from(
        self, buf: deque, max_drain: int = 10,
    ) -> Slot | None:
        """MOD-B: bounded eviction-aware pop.

        Pop entries from ``buf`` left-to-right; examine at most ``max_drain``
        per call. Returns:
        - the first slot whose disconnect_event is SET — caller treats this as
          an EVICTION (slot.is_evicted is flagged True so worker_loop's
          is_evicted branch fires).
        - OR the first slot whose disconnect_event is NOT set — caller processes
          normally.
        - OR None when ``buf`` is empty OR all ``max_drain`` examined entries
          were already evicted-by-someone-else (next tick retries).

        MOD-B rationale: unbounded drain under a storm pattern (100 dead
        clients × every pop_next tick × symmetric use in pop_matched_thread)
        = O(N²) wedge. Bounded ≤10 examinations gives predictable cost and
        eventual progress.

        ⚠ Caller MUST hold ``self._lock``.
        """
        examined = 0
        while buf and examined < max_drain:
            slot = buf.popleft()
            examined += 1
            if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                # Evicted in flight — flag + return for caller-handled audit.
                slot.is_evicted = True
                return slot
            return slot
        return None

    async def pop_next(self) -> Slot | None:
        """Pop the next STAGED slot for activation. Returns None if empty.

         (REC-1 + MOD-B): LOOP form, no recursion. After consulting
        the bounded eviction-aware helper, if the result is None (either buf
        empty or all examined entries pre-evicted), drain accept-buffer into
        staging and retry once with the helper. Bounded by helper's max_drain.
        """
        async with self._lock:
            slot = self._pop_first_non_evicted_from(self._staging)
            if slot is not None:
                # Replenish staging from buffer if there's room.
                if self._accept_buf and len(self._staging) < self.staging_max:
                    tail = self._accept_buf.popleft()
                    tail.state = SlotState.STAGED
                    self._staging.append(tail)
                return slot
            # Staging empty OR all examined were pre-evicted; drain buffer + retry.
            while self._accept_buf and len(self._staging) < self.staging_max:
                s = self._accept_buf.popleft()
                s.state = SlotState.STAGED
                self._staging.append(s)
            return self._pop_first_non_evicted_from(self._staging)

    async def enqueue_head(self, slot: Slot) -> None:
        """Insert at FIFO head — used for ACTIVE-MATCH mid-stream same-thread arrivals (v0.2 §6)."""
        if self._closed:
            raise QueueClosed("queue closed")
        async with self._lock:
            slot.state = SlotState.STAGED
            self._staging.appendleft(slot)

    async def find_matched_thread(self, thread_id: str, model_tag: str) -> Slot | None:
        """Locate a staged slot with same (thread_id, model_tag) for grace-window rematch.

        Kept for read-only callers (introspection); the production fast path now uses
        ``pop_matched_thread`` which atomically pops in one lock acquire (review fix).
        """
        if not thread_id:
            return None
        async with self._lock:
            for slot in self._staging:
                if slot.thread_id == thread_id and slot.model_tag == model_tag:
                    return slot
        return None

    async def pop_matched_thread(
        self, thread_id: str, model_tag: str
    ) -> Slot | None:
        """Fix + MOD-1: atomic find + remove + eviction check.

        Scans staging in order, deletes the first (thread_id, model_tag) match,
        then performs the eviction-check INLINE before returning. If the
        matched slot's disconnect_event is set, flags is_evicted=True so the
        worker_loop's is_evicted branch handles it (audit + fail-future).
        Bounded by len(staging) ≤ staging_max=100, so no extra drain cap needed.

        Symmetric with pop_next's eviction handling (MOD-B + MOD-1
        consistency — eviction can land on a grace-rematch slot just as
        easily as a fresh-staging slot).
        """
        if not thread_id:
            return None
        async with self._lock:
            for i, slot in enumerate(self._staging):
                if (
                    slot.thread_id == thread_id
                    and slot.model_tag == model_tag
                ):
                    del self._staging[i]
                    if (
                        slot.disconnect_event is not None
                        and slot.disconnect_event.is_set()
                    ):
                        slot.is_evicted = True
                    return slot
        return None

    async def remove(self, slot_id: str) -> Slot | None:
        """Remove a specific slot by id from either buffer."""
        async with self._lock:
            for buf in (self._staging, self._accept_buf):
                for i, s in enumerate(buf):
                    if s.slot_id == slot_id:
                        del buf[i]
                        return s
        return None

    async def peek_staging(self) -> list[Slot]:
        async with self._lock:
            return list(self._staging)

    def depth(self) -> dict:
        """Sync snapshot of queue depths. Minor lock-skip OK for /status."""
        return {
            "acceptance_buffer_depth": len(self._accept_buf),
            "staging_queue_depth": len(self._staging),
            "staging_queue_max": self.staging_max,
            "acceptance_buffer_max": self.acceptance_max,
        }

    async def close(self) -> list[Slot]:
        """NEMO V2 2.1 fix: return the cleared slots so manager.shutdown can
        fail their pending completion_futures. Previously close() silently
        clobbered _staging + _accept_buf -- every awaiting caller hung until
        the submit_and_wait timeout (default 600s) fired or never returned.
        """
        async with self._lock:
            self._closed = True
            cleared: list[Slot] = list(self._staging) + list(self._accept_buf)
            self._accept_buf.clear()
            self._staging.clear()
            return cleared


class GraceTimer:
    """Tracks the GRACE window after slot completion.

    Per v0.2 §6: follow-up with matching thread_id within window → warm-slot reuse.
    Bounded by max_extensions to prevent starvation (v0.2 §4 + §6).
    """

    def __init__(self, grace_seconds: float, max_extensions: int = 5) -> None:
        self.grace_seconds = grace_seconds
        self.max_extensions = max_extensions
        self._started_at: float | None = None
        self.thread_id: str | None = None
        self.model_tag: str | None = None
        self.extension_count = 0

    def start(self, thread_id: str, model_tag: str) -> None:
        self._started_at = time.monotonic()
        self.thread_id = thread_id
        self.model_tag = model_tag
        self.extension_count = 0

    def restart_for_followup(self) -> bool:
        """Reset start time for a matched follow-up. Returns False if extension cap exceeded."""
        if self.extension_count >= self.max_extensions:
            return False
        self.extension_count += 1
        self._started_at = time.monotonic()
        return True

    def remaining_s(self) -> float:
        if self._started_at is None:
            return 0.0
        elapsed = time.monotonic() - self._started_at
        return max(0.0, self.grace_seconds - elapsed)

    def expired(self) -> bool:
        return self._started_at is None or self.remaining_s() <= 0.0

    def matches(self, thread_id: str, model_tag: str) -> bool:
        return (
            self._started_at is not None
            and self.thread_id == thread_id
            and self.model_tag == model_tag
            and not self.expired()
        )

    def reset(self) -> None:
        self._started_at = None
        self.thread_id = None
        self.model_tag = None
        self.extension_count = 0


class IdleHotTimer:
    """Tracks the IDLE_HOT window after the queue drains.

    Per v0.2 §6: fresh request with same model_tag → ACTIVE on warm slot.
    """

    def __init__(self, idle_seconds: float) -> None:
        self.idle_seconds = idle_seconds
        self._started_at: float | None = None
        self.model_tag: str | None = None

    def start(self, model_tag: str) -> None:
        self._started_at = time.monotonic()
        self.model_tag = model_tag

    def remaining_s(self) -> float:
        if self._started_at is None:
            return 0.0
        elapsed = time.monotonic() - self._started_at
        return max(0.0, self.idle_seconds - elapsed)

    def expired(self) -> bool:
        return self._started_at is None or self.remaining_s() <= 0.0

    def matches_same_model(self, model_tag: str) -> bool:
        return (
            self._started_at is not None
            and self.model_tag == model_tag
            and not self.expired()
        )

    def reset(self) -> None:
        self._started_at = None
        self.model_tag = None
