"""TurbohaulQueue: two-tier (unbounded acceptance buffer + capped staging) + grace/idle timers.

See ARCHITECTURE.md for the queueing and grace/idle-window design.
"""
import asyncio
import logging
import time
from collections import deque

from turbohaul.slot import Slot, SlotState
from turbohaul.fsm import transition as fsm_transition

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

    def __init__(
        self,
        staging_max: int = 100,
        acceptance_max: int = 10000,
        *,
        max_consecutive_same_model: int = 1,
        max_other_model_wait_s: float = 0.0,
        main_lane_reserved: bool = True,
        main_lane_identity_keys: tuple[str, ...] = ("is_main",),
    ) -> None:
        self.staging_max = staging_max
        self.acceptance_max = acceptance_max
        self._accept_buf: deque[Slot] = deque()
        self._staging: deque[Slot] = deque()
        self._lock = asyncio.Lock()
        self._closed = False
        # Model-affinity pop tuning. Defaults (cap=1, wait=0.0) make pop_next
        # a strict-FIFO no-op even when warm_model_tag is supplied: cap=1 forces
        # the FIFO head every call and wait=0.0 starves any other-model head
        # immediately. Real values come from QueueConfig via the manager ctor.
        self.max_consecutive_same_model = max_consecutive_same_model
        self.max_other_model_wait_s = max_other_model_wait_s
        self.main_lane_reserved = main_lane_reserved
        self.main_lane_identity_keys = tuple(main_lane_identity_keys)
        # Run-length bookkeeping for the affinity path. Sole mutator is
        # pop_next (under self._lock); no second concurrent writer.
        self._consecutive_same_model: int = 0
        self._last_popped_model_tag: str | None = None

    async def enqueue(self, slot: Slot) -> None:
        """Add a fresh slot. Promotes to staging if room; else accept-buffer."""
        if self._closed:
            raise QueueClosed("queue closed")
        async with self._lock:
            if len(self._staging) < self.staging_max:
                fsm_transition(slot, SlotState.STAGED)
                self._staging.append(slot)
                return
            if len(self._accept_buf) >= self.acceptance_max:
                raise QueueFull(
                    f"acceptance buffer at max {self.acceptance_max}"
                )
            fsm_transition(slot, SlotState.ACCEPT_BUFFER)
            self._accept_buf.append(slot)

    def _pop_first_non_evicted_from(
        self, buf: deque, max_drain: int = 10,
    ) -> Slot | None:
        """Bounded eviction-aware pop.

        Pop entries from ``buf`` left-to-right; examine at most ``max_drain``
        per call. Returns:
        - the first slot whose disconnect_event is SET — caller treats this as
          an EVICTION (slot.is_evicted is flagged True so worker_loop's
          is_evicted branch fires).
        - OR the first slot whose disconnect_event is NOT set — caller processes
          normally.
        - OR None when ``buf`` is empty OR all ``max_drain`` examined entries
          were already evicted-by-someone-else (next tick retries).

        Bounding rationale: an unbounded drain under a storm pattern (100 dead
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

    def _pop_first_matching_non_evicted_from(
        self, buf: deque, model_tag: str, max_scan: int = 100,
    ) -> Slot | None:
        """Affinity sibling of ``_pop_first_non_evicted_from``: pop the first
        entry whose ``model_tag`` matches, SKIPPING (NOT removing) entries for
        other models.

        Eviction handling MIRRORS ``_pop_first_non_evicted_from`` EXACTLY: if
        the matched slot's ``disconnect_event`` is SET it is flagged
        ``is_evicted=True`` and returned for caller-handled audit (the
        worker_loop's is_evicted branch then fires). Non-matching entries are
        left in place — they keep their FIFO position for a future
        forced-head pop.

        Scans left-to-right examining at most ``max_scan`` entries (default =
        staging_max=100). Returns:
        - the first matching slot (possibly flagged is_evicted), removed from
          ``buf`` in place; OR
        - None when no matching entry is found within ``max_scan`` examinations
          (caller falls back to the strict-FIFO head pop).

        Bounded ``max_scan`` keeps cost predictable under a staging full of
        non-matching entries (no O(N²) wedge). Removal mid-deque is O(N) via
        ``del buf[i]`` but bounded by max_scan.

        ⚠ Caller MUST hold ``self._lock``.
        """
        examined = 0
        for i, slot in enumerate(buf):
            if examined >= max_scan:
                break
            examined += 1
            if slot.model_tag != model_tag:
                # Skip — leave it in place, keeping its FIFO position.
                continue
            del buf[i]
            if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                # Evicted in flight — flag + return for caller-handled audit
                # (symmetric with _pop_first_non_evicted_from).
                slot.is_evicted = True
            return slot
        return None

    def _is_main_lane(self, slot: Slot) -> bool:
        """True when explicit client metadata identifies interactive main work."""
        meta = slot.client_meta or {}
        return any(bool(meta.get(key)) for key in self.main_lane_identity_keys)

    def _pop_first_main_locked(self) -> Slot | None:
        """Remove the oldest main-lane request across staging then acceptance.

        Main admission is a queue-level reservation only: it does not interrupt
        an active sidecar and therefore preserves the single-sidecar invariant.
        """
        for buf in (self._staging, self._accept_buf):
            for i, slot in enumerate(buf):
                if not self._is_main_lane(slot):
                    continue
                del buf[i]
                if slot.state is SlotState.ACCEPT_BUFFER:
                    fsm_transition(slot, SlotState.STAGED)
                if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                    slot.is_evicted = True
                return slot
        return None

    async def pop_next(self, *, warm_model_tag: str | None = None) -> Slot | None:
        """Pop the next STAGED slot for activation. Returns None if empty.

        LOOP form, no recursion. After consulting the bounded eviction-aware
        helper, if the result is None (either buf empty or all examined entries
        pre-evicted), drain accept-buffer into staging and retry once with the
        helper. Bounded by helper's max_drain.

        Model-affinity (single-mutator-safe parallelism support): when the
        worker_loop passes ``warm_model_tag`` (the model currently warm in the
        active/idle holder) AND staging is non-empty, prefer popping a slot for
        that same model so the warm sidecar is reused — UNLESS:
          - the FIFO-head other-model request has aged past
            ``max_other_model_wait_s`` (head starvation), OR
          - we've already popped ``max_consecutive_same_model`` of this model
            in a row (batch cap — fairness),
        in which case the FIFO head is forced to drain the starved request.
        ``warm_model_tag=None`` (default for every existing caller) preserves
        the exact pre-existing FIFO path with ZERO behavior change. cap=1 +
        wait=0.0 also collapse to strict FIFO even when a tag is supplied.
        """
        async with self._lock:
            # Strict main reservation wins over FIFO and warm-model affinity.
            # It is intentionally admission-only: a running auxiliary request
            # completes normally, then the oldest queued main request is next.
            if self.main_lane_reserved:
                main = self._pop_first_main_locked()
                if main is not None:
                    if self._accept_buf and len(self._staging) < self.staging_max:
                        tail = self._accept_buf.popleft()
                        fsm_transition(tail, SlotState.STAGED)
                        self._staging.append(tail)
                    self._update_run_length_locked(main)
                    return main
            if warm_model_tag is None or not self._staging:
                # === Existing FIFO path — ZERO behavior change. ===
                slot = self._pop_first_non_evicted_from(self._staging)
                if slot is not None:
                    # Replenish staging from buffer if there's room.
                    if self._accept_buf and len(self._staging) < self.staging_max:
                        tail = self._accept_buf.popleft()
                        fsm_transition(tail, SlotState.STAGED)
                        self._staging.append(tail)
                    self._update_run_length_locked(slot)
                    return slot
                # Staging empty OR all examined were pre-evicted; drain buffer + retry.
                while self._accept_buf and len(self._staging) < self.staging_max:
                    s = self._accept_buf.popleft()
                    fsm_transition(s, SlotState.STAGED)
                    self._staging.append(s)
                returned = self._pop_first_non_evicted_from(self._staging)
                self._update_run_length_locked(returned)
                return returned

            # === Model-affinity path (warm_model_tag supplied, staging non-empty). ===
            # Decide BEFORE popping whether affinity applies or the FIFO head is
            # forced. Both branches replenish staging from the buffer afterward
            # (one entry, mirroring the FIFO path) so depth invariants hold.
            head = self._staging[0]
            head_is_other = head.model_tag != warm_model_tag
            head_starved = (
                head_is_other
                and (time.monotonic() - head.created_at) > self.max_other_model_wait_s
            )
            batch_cap_hit = (
                self._consecutive_same_model >= self.max_consecutive_same_model
            )
            # Residency guard: the count cap must NOT force a MODEL
            # SWAP while same-model work is still queued for the resident and the
            # other-model head has not aged past the starvation window. At one GPU
            # slot (max_parallel_sidecars=1) a forced swap M->N->M evicts + reloads
            # a model the very next queued item wants back — avoidable churn.
            # Genuine head-starvation (time-based, `head_starved`) STILL forces the
            # swap so a truly-waiting other-model request is never starved. The
            # count cap now only forces the FIFO head when doing so causes NO
            # avoidable swap — i.e. the head is already the resident's model
            # (`not head_is_other`, a harmless same-model drain). When the head is
            # a DIFFERENT, not-yet-starved model we fall to the affinity branch,
            # which drains queued same-model work and only pops the other-model
            # head if no same-model work remains. Same-model run-length under real
            # contention is thus bounded by max_other_model_wait_s (time) rather
            # than by max_consecutive_same_model (count).
            force_head = head_starved or (batch_cap_hit and not head_is_other)
            if force_head:
                # Force the FIFO head to drain the starved (or same-model) request.
                returned = self._pop_first_non_evicted_from(self._staging)
            else:
                # Prefer a same-model staging entry; fall back to FIFO head if
                # no same-model entry within the scan bound.
                returned = self._pop_first_matching_non_evicted_from(
                    self._staging, warm_model_tag, max_scan=self.staging_max,
                )
                if returned is None:
                    returned = self._pop_first_non_evicted_from(self._staging)
            # Replenish staging from buffer (mirror FIFO path: one entry if room).
            if returned is not None and self._accept_buf and len(self._staging) < self.staging_max:
                tail = self._accept_buf.popleft()
                fsm_transition(tail, SlotState.STAGED)
                self._staging.append(tail)
            self._update_run_length_locked(returned)
            return returned

    def _update_run_length_locked(self, returned: "Slot | None") -> None:
        """Update the same-model run-length counter after a pop.

        Increment when the returned slot's model matches the last popped model,
        else reset to 1. None (empty pop) leaves the counter untouched. ⚠
        Caller MUST hold ``self._lock``.
        """
        if returned is None:
            return
        if returned.model_tag == self._last_popped_model_tag:
            self._consecutive_same_model += 1
        else:
            self._consecutive_same_model = 1
        self._last_popped_model_tag = returned.model_tag

    async def enqueue_head(self, slot: Slot) -> None:
        """Insert at FIFO head — used for ACTIVE-MATCH mid-stream same-thread arrivals.

        Guard: only transition to STAGED if the slot is not already STAGED.
        enqueue_head is called on ALREADY-STAGED slots (fan-out rider push-back,
        unroutable requeue) — fsm_transition(STAGED->STAGED) is ILLEGAL and
        raises InvalidTransition, breaking the fan-out loop.
        """
        if self._closed:
            raise QueueClosed("queue closed")
        async with self._lock:
            if slot.state is not SlotState.STAGED:
                fsm_transition(slot, SlotState.STAGED)
            self._staging.appendleft(slot)

    async def find_matched_thread(self, thread_id: str, model_tag: str) -> Slot | None:
        """Locate a staged slot with same (thread_id, model_tag) for grace-window rematch.

        Kept for read-only callers (introspection); the production fast path now uses
        ``pop_matched_thread`` which atomically pops in one lock acquire.
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
        """Atomic find + remove + eviction check.

        Scans staging in order, deletes the first (thread_id, model_tag) match,
        then performs the eviction-check INLINE before returning. If the
        matched slot's disconnect_event is set, flags is_evicted=True so the
        worker_loop's is_evicted branch handles it (audit + fail-future).
        Bounded by len(staging) ≤ staging_max=100, so no extra drain cap needed.

        Symmetric with pop_next's eviction handling for consistency — eviction
        can land on a grace-rematch slot just as easily as a fresh-staging
        slot.
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

    async def head_model_tag(self) -> str | None:
        """Non-destructive peek: model_tag of the current staging head (the FIFO
        next-to-pop), or None when staging is empty.

        Residency guard: the worker_loop consults this to keep the
        resident model warm when the very next queued request is the SAME model
        (intent: "if the next request in the queue is the same model, it
        should stay loaded"). PURE READ under the same ``self._lock`` discipline
        pop_next uses — it examines only ``_staging[0]`` and mutates NOTHING
        (queue depth unchanged), so it never perturbs FIFO order or run-length
        bookkeeping.
        """
        async with self._lock:
            return self._staging[0].model_tag if self._staging else None

    def depth(self) -> dict:
        """Sync snapshot of queue depths. Minor lock-skip OK for /status."""
        return {
            "acceptance_buffer_depth": len(self._accept_buf),
            "staging_queue_depth": len(self._staging),
            "staging_queue_max": self.staging_max,
            "acceptance_buffer_max": self.acceptance_max,
        }

    async def close(self) -> list[Slot]:
        """Return the cleared slots so manager.shutdown can
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

    A follow-up with a matching thread_id within the window → warm-slot reuse.
    Bounded by max_extensions to prevent starvation.
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

    A fresh request with the same model_tag → ACTIVE on the warm slot.
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
