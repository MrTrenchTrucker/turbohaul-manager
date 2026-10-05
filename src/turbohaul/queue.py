"""TurbohaulQueue: two-tier (unbounded acceptance buffer + capped staging) + grace/idle timers.

See ARCHITECTURE.md for the queueing and grace/idle-window design.
"""
import asyncio
import dataclasses
import hashlib
import logging
import time
from collections import deque
from typing import Callable, Optional

from turbohaul.slot import Slot, SlotState
from turbohaul.fsm import transition as fsm_transition
from turbohaul.kv_classify import CLASS_COMPRESSION, CLASS_MAIN, _class_from_label
# Single source of truth for chain-prefix comparison (do NOT
# re-implement -- kv_classify.py says so for the KV-restore consumer,
# and this is the second consumer, not a fork).
from turbohaul.kv_policy import _is_prefix_match
import ipaddress

log = logging.getLogger(__name__)


class QueueClosed(RuntimeError):
    pass


class QueueFull(RuntimeError):
    pass


# Size at which the fairness floor's per-client serve ledger
# is walked for already-re-armed entries. A PERFORMANCE threshold only -- see
# _prune_floor_ledger_locked for why what the prune removes is invisible to
# every reader of the ledger, at any threshold.
_FLOOR_LEDGER_PRUNE_AT = 128


@dataclasses.dataclass(frozen=True)
class FastLanePopPolicy:
    """Fast Lane: bundles the two knobs _pick_fastlane_locked needs,
    passed as ONE pop_next kwarg -- growing the signature by a second raw
    kwarg for the swap budget would duplicate the same
    read-live-off-runtime.fastlane plumbing twice; manager._fastlane_pop_kwarg()
    builds this bundle in one place instead. None (the default everywhere
    except the admission stamp site) means "feature off / no policy" and
    pop_next's fastlane branch is skipped entirely -- ZERO behavior change
    from today when fastlane_policy is None.
    """

    max_normal_wait_s: float
    cross_model_switches_per_min: int
    # (rule_index, rank) for every currently-loaded resident that resolves to a
    # Fast Lane rule (an unresolvable/unlisted resident contributes nothing --
    # any registered candidate already outranks it). Built fresh by
    # manager._fastlane_pop_kwarg() on every call, same live-read discipline as
    # the two scalars above.
    #
    # A value of () exempts EVERY candidate from the swap budget, not none.
    # This follows from the field's only consumer:
    # _pick_fastlane_locked computes
    # `all(_fastlane_strictly_higher(cand_key, k) for k in ...)` over this
    # field, and `all(())` is True. The consumer at that site is the
    # authority on this and states it directly ("Vacuously exempt when
    # loaded_priority_keys is empty"), so there is one answer, not two.
    #
    # Whether () should be able to mean "the caller does not know"
    # as distinct from "nothing resolves" is a possible refinement
    # (None vs ()); this comment only describes what the code does
    # today.
    loaded_priority_keys: "tuple[tuple[int, int], ...]" = ()
    # The (rule_index, rank) of the best
    # LIVE Fast Lane claim right now, or None when nothing is claiming.
    #
    # WHY THIS FIELD EXISTS AT ALL. A deferred claimant is NOT in either buffer
    # while it waits: ``manager._defer_unroutable`` registers its claim and
    # hands the slot to ``_requeue_after_backoff``, which AWAITS the backoff
    # BEFORE calling ``enqueue_head``. In practice a claimant can be
    # absent from both buffers for most of its wait, and one contested
    # admission can be decided by a few MILLISECONDS of backoff phase between two
    # clients one rank apart. The sort below was rank-correct the whole time --
    # over a candidate set that did not contain the higher-ranked waiter.
    #
    # ⛔ LIVE claims only, and the liveness is NOT decided here. The manager
    # resolves it with ``_governing_claim_priority_key_locked`` ->
    # ``_governing_claim_locked``, which re-derives liveness per call via
    # ``_claim_is_live`` -- the SAME predicate the eviction path uses. A second
    # liveness notion is exactly what must not be invented: the claim registry
    # has no periodic sweep (its only pruner runs from ``status_snapshot``), so
    # a claim this queue treated as live-by-presence could block service
    # forever with nobody polling /status. That would convert an ordering
    # defect into a silent liveness defect, which is strictly worse.
    #
    # None means "no live claim" and restores the pre-existing behaviour exactly.
    governing_claim_key: "tuple[int, int] | None" = None


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
        on_enqueue: "Callable[[], None] | None" = None,
    ) -> None:
        self.staging_max = staging_max
        self.acceptance_max = acceptance_max
        self._accept_buf: deque[Slot] = deque()
        self._staging: deque[Slot] = deque()
        self._lock = asyncio.Lock()
        self._closed = False
        # Dispatcher wake hook: fired after EVERY successful append to either
        # buffer (enqueue and enqueue_head both call it -- enqueue_head is the
        # re-entry point for an unroutable/evict-pending requeue, so hooking
        # enqueue alone would leave the dispatcher unable to notice a slot
        # returning from its own backoff timer). Sync, await-free by contract;
        # exceptions are caught and logged here so a hook failure can never
        # surface as an enqueue/enqueue_head failure.
        self._on_enqueue = on_enqueue
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
        # Fast Lane: rolling 60s window of GRANTED cross-model swap
        # timestamps (monotonic), for the per-minute swap budget in
        # _pick_fastlane_locked. Sole mutator is pop_next (under self._lock).
        self._fastlane_swap_times: deque[float] = deque()
        # Fast Lane: refusal counts by reason, observability for why a pick
        # was refused. The lane-side reason "budget" (a matched priority slot
        # skipped) cannot fire -- see _fastlane_swap_allowed_locked's docstring --
        # it is structurally unreachable, not "never happened to trigger."
        # The reason
        # "ordinary_budget" covers the opposite
        # population: a non-lane cross-model swap refused by the same
        # per-minute budget, see the `else` arm in pop_next's affinity
        # section.
        self._fastlane_refusal_counts: dict[str, int] = {}
        # Fast Lane: GRANTED-swap counts by reason ("warm" = no jump needed,
        # "exempt" = priority-exempt from the budget, "budget_exempt" =
        # the unconditional Fast-Lane override of the per-minute
        # budget) -- the counterpart to _fastlane_refusal_counts above (that
        # one counts DENIED jumps). A "budget" reason (rate-check
        # passed) does not occur; see _pick_fastlane_locked.
        self._fastlane_swap_counts: dict[str, int] = {}
        # The one-turn wall-clock fairness floor's per-CLIENT
        # serve ledger -- client key (see _floor_client_key) -> the monotonic
        # time that client was last SERVED a turn by this queue, by ANY path.
        # The fairness rule admits the longest-waiting queued CLIENT for
        # exactly one full turn and then re-arms its timer, which bounds the
        # worst-case time a queued client goes without work; without this
        # ledger the floor would have no per-client accounting and no re-arm state at
        # all, so one client with N aged requests would take N consecutive turns.
        # Design decision: the clock is time-since-last-SERVED-by-any-path,
        # not time-since-last-floor-promotion, so
        # every pop credits it, not only a floor promotion. Sole mutators are
        # _credit_floor_serve_locked and the prune in _pick_fastlane_locked,
        # both under self._lock.
        self._floor_last_served_at: dict[str, float] = {}
        # Set the first time pop_next is handed a FastLanePopPolicy. The floor
        # is structurally unreachable without one, so a process that never
        # enables Fast Lane writes nothing to the ledger at all -- feature-off
        # stays byte-for-byte today's behavior AND costs no memory, matching
        # this file's own feature-off convention.
        self._floor_ledger_active: bool = False

    async def drain_inbox_and_staging_match_ip(
        self,
        ip: str,
        model_tag: str,
        grace_started_at: float,
        inbox: "Optional[asyncio.Queue[Slot]]",
        *,
        decline_sink: "dict | None" = None,
    ) -> Slot | None:
        """Match _staging AND inbox under ONE lock hold.

        The HIT route (_route_or_resident -> r.inbox.put_nowait) bypasses
        enqueue(), so a returning slot sits in r.inbox, NOT _staging. The
        grace-loop matchers scan _staging and miss.

        This matches _staging first (normal path), then peeks at inbox
        contents (HIT route) -- all under ONE lock hold. If a match is
        found in the inbox, it is removed from there.

        Returns the matched Slot or None. The decline_sink is populated on
        miss (same contract as pop_matched_ip).

        A candidate, in staging or in the inbox, is skipped while a
        better-ranked request of the same client (same ``rule_index``,
        strictly lower ``rank``) is waiting anywhere in staging, the
        acceptance buffer or the inbox; the skipped slot stays where it is.

        NOTE: inbox is Optional[asyncio.Queue[Slot]] -- r.inbox is Optional
        on the Resident model. If inbox is None, the inbox-peek is skipped
        and only _staging is matched.
        """
        if not ip:
            return None
        try:
            anchor_ip = ipaddress.ip_address(ip)
        except ValueError:
            return None

        def _slot_ip_ok(slot: Slot) -> bool:
            raw = (slot.client_meta or {}).get("ip")
            if not raw:
                return False
            try:
                return ipaddress.ip_address(raw) == anchor_ip
            except ValueError:
                return False

        def _predicate(slot: Slot) -> bool:
            if slot.model_tag != model_tag:
                return False
            if slot.created_at <= grace_started_at:
                return False
            return _slot_ip_ok(slot)

        async with self._lock:
            # Take the inbox contents first (public queue API only) so a
            # staged candidate can be checked against better-ranked requests
            # of the same client that sit in the inbox. On a staging hit, or
            # on an exception, the inbox is put back exactly as it was.
            depth_at_entry = inbox.qsize() if inbox is not None else 0
            _inbox_all: list[Slot] = []
            if inbox is not None:
                while not inbox.empty():
                    try:
                        _inbox_all.append(inbox.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            try:
                # Match _staging first (normal path)
                matched = self._pop_first_matching_locked(
                    _predicate, via="pop_matched_ip", guard_anchor=grace_started_at,
                    also_waiting=_inbox_all,
                )
            except BaseException:
                for s in _inbox_all:
                    inbox.put_nowait(s)
                raise
            if matched is not None:
                for s in _inbox_all:
                    inbox.put_nowait(s)
                if decline_sink is not None:
                    decline_sink["reason"] = "matched"
                return matched

            # No match in _staging -- check the inbox contents taken above
            # (HIT route), re-enqueue non-matches.
            # Under queue lock so the matcher sees a consistent snapshot.
            if inbox is not None:
                _peeked = []
                _match_from_inbox = None
                try:
                    for s in _inbox_all:
                        if (
                            _match_from_inbox is None
                            and _predicate(s)
                            and not self._decline_for_better_same_client_locked(
                                s, also_waiting=_inbox_all, via="pop_matched_ip_inbox"
                            )
                        ):
                            _match_from_inbox = s
                        else:
                            _peeked.append(s)
                except BaseException:
                    for s in _inbox_all:
                        inbox.put_nowait(s)
                    raise
                # Re-enqueue non-matching slots at FIFO tail
                for s in _peeked:
                    inbox.put_nowait(s)
                # POST-CONDITION: every slot present at entry was
                # EXAMINED -- not that the inbox ends up empty. This function is DESIGNED to
                # re-enqueue non-matching slots, so inbox-empty and its own contract cannot
                # both hold; the invariant it was reaching for is examined == depth_at_entry,
                # which closes the same head-only-drain hole (a slot silently never popped)
                # without firing on the re-enqueue this function exists to perform.
                # Not `assert`: stripped under `python -O`, silently turning a real corruption
                # signal into nothing (this file raises nothing else via bare `assert`). Raised
                # loud on purpose, not logged-only: no `await` executes between depth_at_entry
                # above and here (every helper this block calls -- _pop_first_matching_locked,
                # _credit_floor_serve_locked, _log_admitted_without_pick -- is a plain sync
                # `def`), so under asyncio's single-threaded scheduling a concurrent
                # r.inbox.put_nowait cannot land inside this span regardless of lock scope --
                # a firing here is a genuine deterministic bug, not environment noise.
                # ⚠ THIS DEPENDS ON THE ABSENCE OF `await` IN THIS SPAN -- adding one (here or
                # in a helper this calls) reopens exactly that race; re-derive before doing so.
                examined = len(_peeked) + (1 if _match_from_inbox is not None else 0)
                if examined != depth_at_entry:
                    raise AssertionError(
                        f"inbox drain examined {examined} of {depth_at_entry} slots present "
                        f"at entry (head-only drain bug pattern)"
                    )
                if _match_from_inbox is not None:
                    self._credit_floor_serve_locked(_match_from_inbox)
                    self._log_admitted_without_pick(_match_from_inbox, via="pop_matched_ip_inbox", guard_anchor=grace_started_at)
                    if decline_sink is not None:
                        decline_sink["reason"] = "matched"
                    return _match_from_inbox

        if decline_sink is not None:
            decline_sink["reason"] = "no_candidate_matched"
            decline_sink["staging_depth"] = len(self._staging)
            decline_sink["inbox_depth"] = inbox.qsize() if inbox is not None else 0
        return None

    async def drain_inbox_and_staging_match_hash_chain(
        self,
        anchor_chain: "list[str]",
        model_tag: str,
        grace_started_at: float,
        inbox: "Optional[asyncio.Queue[Slot]]",
    ) -> Slot | None:
        """Content-derived-identity counterpart of
        ``drain_inbox_and_staging_match_ip``, same shape (atomic
        inbox-drain), different identity signal.

        WHY THIS EXISTS: the IP path exists because ``pop_matched_thread``
        can miss -- ``thread_id`` is a hash of the WHOLE prompt when the
        conversation is still under the prefix-token cutoff, so it changes on
        every turn (see ``derive_thread_id_prefix_hash``). The population that
        takes this fallback is, by construction, exactly the population where
        thread-id identity already failed once. A second identity signal that
        also hashes "the whole thing" would fail the same way; this one does
        not, because ``admission_hash_chain`` is an INCREMENTAL, ORDER-SENSITIVE
        chain (``kv_policy._prefix_hash_chain``) -- a growing conversation
        EXTENDS its chain, so the anchor's saved chain stays a genuine PREFIX
        of the incoming one even though the derived thread_id churns. That
        difference is why this can succeed where thread_id matching failed.

        MATCH SEMANTICS -- routed through the existing chokepoint, not
        re-implemented (kv_classify.py's rule applies here as the second
        consumer): a candidate matches iff ``kv_policy._is_prefix_match
        (anchor_chain, candidate.admission_hash_chain)`` is True, i.e. the
        anchor's own chain, element-for-element, is a prefix of the
        candidate's. ``_is_prefix_match`` already fails safe on an empty
        ``anchor_chain`` (returns False outright) and on a candidate chain
        shorter than the anchor's -- exactly the "uncertain -> refuse, never
        guess" posture the IP path used a denylist for; a content-derived
        chain does not need one, because two unrelated conversations
        producing an identical SHA256 chain prefix is not a realistic risk
        the way one IP address covering two clients is.

        Same lock discipline, same inbox-peek-and-restore, same
        head-only-drain post-condition as ``drain_inbox_and_staging_match_ip``
        -- see that method's docstring for the full contract; only the
        predicate differs.

        NO ``decline_sink``: the IP sibling's sink is
        write-only in production (the manager never
        reads it back) and exists only so one test (the
        drain examined-invariant test) can use it as an
        assertion oracle for THAT function; this method
        does not carry it. The bounded-drain invariant
        this method still enforces (below) is checked
        instead through the return value and the
        observed inbox contents and order across a
        multi-candidate inbox, rather than through a
        sink dict.
        """
        if not anchor_chain:
            return None

        def _predicate(slot: Slot) -> bool:
            if slot.model_tag != model_tag:
                return False
            if slot.created_at <= grace_started_at:
                return False
            candidate_chain = getattr(slot, "admission_hash_chain", None) or []
            return _is_prefix_match(anchor_chain, candidate_chain)

        async with self._lock:
            # Take the inbox contents first (public queue API only) so a
            # staged candidate can be checked against better-ranked requests
            # of the same client that sit in the inbox. On a staging hit, or
            # on an exception, the inbox is put back exactly as it was.
            depth_at_entry = inbox.qsize() if inbox is not None else 0
            _inbox_all: list[Slot] = []
            if inbox is not None:
                while not inbox.empty():
                    try:
                        _inbox_all.append(inbox.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            try:
                matched = self._pop_first_matching_locked(
                    _predicate, via="pop_matched_hash_chain", guard_anchor=grace_started_at,
                    also_waiting=_inbox_all,
                )
            except BaseException:
                for s in _inbox_all:
                    inbox.put_nowait(s)
                raise
            if matched is not None:
                for s in _inbox_all:
                    inbox.put_nowait(s)
                return matched

            if inbox is not None:
                _peeked = []
                _match_from_inbox = None
                try:
                    for s in _inbox_all:
                        if (
                            _match_from_inbox is None
                            and _predicate(s)
                            and not self._decline_for_better_same_client_locked(
                                s, also_waiting=_inbox_all, via="pop_matched_hash_chain_inbox"
                            )
                        ):
                            _match_from_inbox = s
                        else:
                            _peeked.append(s)
                except BaseException:
                    for s in _inbox_all:
                        inbox.put_nowait(s)
                    raise
                for s in _peeked:
                    inbox.put_nowait(s)
                # Same head-only-drain invariant as drain_inbox_and_staging_match_ip
                # every slot present at entry was EXAMINED.
                # No `await` executes between depth_at_entry above and here, so under
                # asyncio's single-threaded scheduling a concurrent put_nowait cannot
                # land inside this span regardless of lock scope -- a firing here is a
                # genuine deterministic bug, not environment noise.
                examined = len(_peeked) + (1 if _match_from_inbox is not None else 0)
                if examined != depth_at_entry:
                    raise AssertionError(
                        f"inbox drain examined {examined} of {depth_at_entry} slots present "
                        f"at entry (head-only drain bug pattern)"
                    )
                if _match_from_inbox is not None:
                    self._credit_floor_serve_locked(_match_from_inbox)
                    self._log_admitted_without_pick(_match_from_inbox, via="pop_matched_hash_chain_inbox", guard_anchor=grace_started_at)
                    return _match_from_inbox

        return None

    async def enqueue(self, slot: Slot) -> None:
        """Add a fresh slot. Promotes to staging if room; else accept-buffer."""
        if self._closed:
            raise QueueClosed("queue closed")
        async with self._lock:
            if len(self._staging) < self.staging_max:
                fsm_transition(slot, SlotState.STAGED)
                self._staging.append(slot)
                self._log_queue_arrival(slot, landed="staging", via="enqueue")
                self._invoke_on_enqueue()
                return
            if len(self._accept_buf) >= self.acceptance_max:
                raise QueueFull(
                    f"acceptance buffer at max {self.acceptance_max}"
                )
            fsm_transition(slot, SlotState.ACCEPT_BUFFER)
            self._accept_buf.append(slot)
            self._log_queue_arrival(slot, landed="accept_buf", via="enqueue")
            self._invoke_on_enqueue()

    async def enqueue_tail(self, slot: Slot) -> None:
        """Re-insert at FIFO tail:
        a DESIGNATED VICTIM's own displaced (already-STAGED) inbox backlog,
        requeued behind everything already staged rather than cutting to the
        front via ``enqueue_head`` — landing it ahead of other work would
        silently undo the entire point of designating that resident the
        victim in the first place.

        Guard: only transition to STAGED if the slot is not already STAGED —
        same reasoning as ``enqueue_head``'s own guard
        (``fsm_transition(STAGED -> STAGED)`` is illegal, and every real
        inbox-drained slot IS already STAGED — see the real
        dispatcher call chain: ``pop_next`` pops an already-STAGED slot,
        ``_route_or_reserve`` puts it straight into ``r.inbox`` with no
        transition in between).

        Unlike ``enqueue_head``, this DOES check capacity: a tail-place is
        competing with fresh admissions for the same bounded deque, not
        skipping ahead of them the way a grace-window head-insert does.

        ⚠ DELIBERATELY DROPS the accept-buffer fallback that plain
        ``enqueue()`` has, although a straightforward implementation (calling
        ``enqueue()`` directly) would keep it. That approach has the same
        failure mode this method exists to avoid, one hop downstream:
        plain ``enqueue()`` unconditionally re-transitions to STAGED (illegal
        on an already-STAGED slot). Guarding only THAT
        transition and keeping the accept-buffer fallback does not close the
        hole, it defers it — when staging is full, the slot would fall to the
        accept-buffer branch, and ``fsm_transition(slot, ACCEPT_BUFFER)`` on
        an already-STAGED slot is ALSO illegal (``fsm.py``'s
        ``LEGAL_TRANSITIONS``: legal-from-STAGED is only
        ``{LOADING, COLD, ACTIVE_MATCH}``). Guarding that second transition
        too would still not be safe: 4+ other call sites in this file drain
        ``_accept_buf`` back into ``_staging`` with an unconditional
        ``fsm_transition(s, SlotState.STAGED)``, correctly assuming (for
        every OTHER producer of ``_accept_buf`` entries) that the slot is
        genuinely ``ACCEPT_BUFFER`` — a STAGED slot parked there by this
        method would raise the identical ``InvalidTransition`` later, at pop
        time, at a call site with no idea this method ever touched it.

        So: staging has room → tail-append (guarded transition). Staging
        full → ``QueueFull`` directly, no accept-buffer detour — the same
        exception plain ``enqueue()`` raises today when BOTH tiers are
        exhausted, just reached one tier sooner here. The caller
        (``_requeue_slots_tail_or_fail``) already treats a raised exception as
        an ordinary per-slot failure (fails that slot's completion_future,
        continues to the next) via its existing ``except Exception`` handler
        — no call-site change is needed for this narrower failure mode.
        """
        if self._closed:
            raise QueueClosed("queue closed")
        async with self._lock:
            if len(self._staging) >= self.staging_max:
                raise QueueFull(f"staging at max {self.staging_max}")
            if slot.state is not SlotState.STAGED:
                fsm_transition(slot, SlotState.STAGED)
            self._staging.append(slot)
            self._log_queue_arrival(slot, landed="staging", via="enqueue_tail")
            self._invoke_on_enqueue()

    def _invoke_on_enqueue(self) -> None:
        """Fire the dispatcher-wake hook, if one was supplied. MUST be called
        AFTER the append it announces -- every calling method (``enqueue``,
        ``enqueue_tail``, ``enqueue_head``) already holds ``self._lock``, so
        this never races the state it is reporting on.
        Sync and await-free by contract; a raising hook is caught and logged
        rather than propagated, so a wake-side bug can never surface as a
        queue-arrival failure."""
        if self._on_enqueue is None:
            return
        try:
            self._on_enqueue()
        except Exception:
            log.exception("QUEUE_WAKE_HOOK_FAILED")

    def _log_queue_arrival(self, slot: Slot, *, landed: str, via: str) -> None:
        """QUEUE_ARRIVAL: a slot became known to the queue's own
        state, the instant it did -- distinct from whatever admission-layer
        log fired before this (e.g. chat_completion.py's identity_recompose),
        and from QUEUE_PRIORITY (which only fires when the queue actually
        POPS something). Closes a diagnostic gap: an absence of
        QUEUE_PRIORITY lines over a long window could be read
        as "the pick never engaged" when it actually meant "nothing has
        finished yet to trigger a pick at all" -- a waiting request was
        invisible to the logs for the entire time it waited.

        ``landed`` is "staging" (immediately schedulable) or "accept_buf"
        (admitted but capacity-bound) -- the two outcomes enqueue() itself
        already discriminates between. ``via`` is "enqueue" (normal
        admission), "enqueue_head" (grace-window head-insert), or
        "enqueue_tail" (a designated victim's own
        displaced backlog, requeued behind everything already staged) -- a
        slot that arrives via enqueue_head is exactly the shape
        pop_matched_thread later serves without ever reaching
        _pick_fastlane_locked (see _log_admitted_without_pick), so this is
        the earliest point that distinction is observable.

        ⚠ Caller MUST hold ``self._lock`` (called from inside enqueue's /
        enqueue_head's own lock scope, never a second acquire).
        """
        log.info(
            "QUEUE_ARRIVAL model=%s fastlane_matched=%s landed=%s via=%s "
            "staging_depth=%d accept_depth=%d thread=%s",
            slot.model_tag,
            slot.fastlane is not None,
            landed,
            via,
            len(self._staging),
            len(self._accept_buf),
            self._thread_hash(slot),
        )

    def _pop_first_non_unloaded_from(
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
        - OR None when ``buf`` is empty. **That is the ONLY input that yields
          None** -- see the correction below.

        ⚠ WHAT THE LOOP ACTUALLY DOES. The description above gives the
        intent of this helper; the loop below is deliberately written as it is
        and this docstring only describes it. Two claims one might expect are
        not implemented:

        1. The bullet above must not be read as also returning None when all
           ``max_drain`` examined entries were already
           evicted-by-someone-else (next tick retries). That case cannot
           happen: BOTH branches of the loop's ``if`` return, so the loop never
           reaches a second iteration. An all-evicted deque returns the first slot,
           FLAGGED, never None.
        2. A bounded-drain rationale (unbounded drain under a storm pattern --
           100 dead clients x every pop_next tick x symmetric use in
           pop_matched_thread -- would be an O(N^2) wedge; bounding it to <=10
           examinations would give predictable cost and eventual progress) does
           not apply: the function examines exactly ONE entry, so ``max_drain``
           is INERT and that bounded-drain mechanism does not exist. The O(N^2)
           wedge is averted by returning immediately, not by the bound.

        Whether the loop should drain more than one entry is a separate
        design question: it is deliberately left unchanged here,
        because this docstring only describes the loop as it is;
        changing the loop would be a separate behaviour change.
        This describes the code as it is; it does not endorse it. ``max_drain`` is
        left in the signature because callers pass it; removing it belongs with a change to that loop.

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

    def _pop_first_matching_non_unloaded_from(
        self, buf: deque, model_tag: str, max_scan: int = 100,
    ) -> Slot | None:
        """Affinity sibling of ``_pop_first_non_unloaded_from``: pop the first
        entry whose ``model_tag`` matches, SKIPPING (NOT removing) entries for
        other models.

        Eviction handling MIRRORS ``_pop_first_non_unloaded_from`` EXACTLY: if
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
                # (symmetric with _pop_first_non_unloaded_from).
                slot.is_evicted = True
            return slot
        return None

    def _pop_first_matching_locked(
        self, predicate, *, via: str, max_scan: int = 100, guard_anchor: float | None = None,
        also_waiting: "tuple[Slot, ...] | list[Slot]" = (),
    ) -> Slot | None:
        """The ONE chokepoint every
        grace-window staging matcher scans through, replacing hand-rolled
        per-matcher copies of this same loop. It has two callers —
        ``pop_matched_thread`` (the first migration) and ``pop_matched_ip``
        (the second port). It is a pure extraction of that loop; it
        adds no behaviour of its own and changes none of the behaviour of the
        matchers that call it.

        Scans ``self._staging`` left-to-right, examining at most ``max_scan``
        entries, popping (removing) the FIRST slot for which ``predicate(slot)``
        is True. Non-matching entries are left in place, keeping their FIFO
        position — modelled on ``_pop_first_matching_non_unloaded_from`` above,
        NOT on ``_pop_first_non_unloaded_from``'s known-dead single-entry drain
        (see that method's own correction note: its ``max_drain`` is inert).

        Eviction handling mirrors both siblings above: if the matched slot's
        ``disconnect_event`` is SET, it is flagged ``is_evicted=True`` and
        still returned (caller-handled audit via worker_loop's is_evicted
        branch). On a match this also performs the full serve epilogue —
        ``_credit_floor_serve_locked`` (this path bypasses
        ``_update_run_length_locked`` too, same as pop_matched_thread did
        standalone, so it must self-credit) and ``_log_admitted_without_pick``
        with the caller's ``via`` label (provenance — WHICH
        mechanism served this slot, not just that one did). Both live here once,
        rather than being duplicated inline in every matcher.

        Same-client rank rule: a candidate whose predicate is true is SKIPPED
        (left in place) while a better-ranked request of the same client
        (same ``rule_index``, strictly lower ``rank``) is waiting in staging,
        the acceptance buffer or ``also_waiting`` (resident-inbox entries the
        caller has taken); the scan continues with later candidates and
        returns None if none remains. See
        ``_better_ranked_same_client_waiting_locked``.

        ⭐ THE WRITTEN REASON this chokepoint takes a bare ``predicate`` and
        applies no identity guard of its own (why one matcher needs a
        temporal condition and the other does not; the same-client rank skip
        above orders waiting requests, it does not judge identity):
        A matcher's predicate is expected to encode its own FULL identity
        contract, guards included. ``pop_matched_thread``'s predicate matches
        on ``thread_id`` — a content hash, so an accidental collision between
        two unrelated conversations is not a realistic risk, and ANY staging
        slot carrying that exact id is safe to match regardless of when it
        arrived. A predicate built on a weaker identity signal — an IP
        address, for example: a NETWORK identity, not a conversation one,
        which multiple unrelated clients or sessions can share (NAT, a
        proxy, one client running several sessions at once) — cannot make
        that same assumption, and needs to narrow itself, e.g. to slots that
        arrived after a specific anchor, or it risks adopting an
        already-queued, unrelated request that merely happens to share the
        weaker signal. That narrowing belongs in the predicate's own closure
        (composed in, visibly, at the call site), not here: a guard baked
        silently into this shared chokepoint would apply to every future
        matcher whether its identity signal needed it or not, and a matcher
        that DOES need one but forgets to compose it in gets no help from a
        chokepoint that never checks — this is deliberately a scan-and-serve
        primitive, not an identity-strength judge.

        PRESERVE the fail-safe direction of every predicate passed here:
        unreadable/missing/malformed inputs must make the predicate REFUSE
        (return False), never match by accident — "uncertainty costs one
        slow turn; a wrong exclusion fuses two conversations onto one KV
        cache" (the same standard every other match path in this file holds
        to).

        ⚠ Caller MUST hold ``self._lock`` — a plain ``def``, this does not
        acquire it itself. Matches ``_pop_first_matching_non_unloaded_from``'s
        and ``_pop_first_non_unloaded_from``'s existing ``_locked``-suffix
        contract exactly (caller-holds-lock, not callee-acquires), so
        ``pop_matched_thread``'s own ``async with self._lock:`` wrapper is
        unchanged by adopting this -- its body shrinks to a predicate plus
        this call (and, for the inbox-aware callers, the inbox hand-over).
        """
        examined = 0
        for i, slot in enumerate(self._staging):
            if examined >= max_scan:
                break
            examined += 1
            if not predicate(slot):
                continue
            if self._decline_for_better_same_client_locked(
                slot, also_waiting=also_waiting, via=via,
            ):
                continue
            del self._staging[i]
            if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                slot.is_evicted = True
            self._credit_floor_serve_locked(slot)
            self._log_admitted_without_pick(slot, via=via, guard_anchor=guard_anchor)
            return slot
        return None

    def _better_ranked_same_client_waiting_locked(
        self, slot: Slot, *, also_waiting: "tuple[Slot, ...] | list[Slot]" = (),
    ) -> bool:
        """True iff ``slot`` is listed and some other waiting slot is listed
        with the SAME ``rule_index`` (the same client) and a STRICTLY lower
        ``rank``. An equal rank is not better; a better rank under another
        ``rule_index`` never counts (rank is only compared inside one
        client); an unlisted ``slot`` is never blocked. Non-mutating.

        Scans ``_staging`` and ``_accept_buf``, each bounded by its own cap
        (same convention as ``_has_strictly_higher_priority_waiting_locked``),
        plus ``also_waiting`` (resident-inbox entries the caller has taken).

        Caller MUST hold ``self._lock``.
        """
        key = self._fastlane_priority_key(slot)
        if key is None:
            return False
        rule_index, rank = key
        for buf, cap in (
            (self._staging, self.staging_max),
            (self._accept_buf, self.acceptance_max),
            (also_waiting, len(also_waiting)),
        ):
            examined = 0
            for other in buf:
                if examined >= cap:
                    break
                examined += 1
                other_key = self._fastlane_priority_key(other)
                if (
                    other_key is not None
                    and other_key[0] == rule_index
                    and other_key[1] < rank
                ):
                    return True
        return False

    def _decline_for_better_same_client_locked(
        self, slot: Slot, *, also_waiting: "tuple[Slot, ...] | list[Slot]" = (), via: str,
    ) -> bool:
        """The matcher-side use of ``_better_ranked_same_client_waiting_locked``:
        True means the matcher must skip ``slot`` (it stays where it is and is
        served later in rank order); logs at debug level when it does.

        Caller MUST hold ``self._lock``.
        """
        if not self._better_ranked_same_client_waiting_locked(
            slot, also_waiting=also_waiting,
        ):
            return False
        log.debug(
            "MATCH_DECLINED_BETTER_RANK_WAITING via=%s staging_depth=%d accept_depth=%d",
            via, len(self._staging), len(self._accept_buf),
        )
        return True

    def _fastlane_swap_allowed_locked(self, budget_per_min: int) -> bool:
        """Rolling 60s-window rate check for a GRANTED cross-model swap. This
        is a genuine per-minute budget (distinct from the count-based
        fairness guard, which would be unsuitable for
        THAT purpose) -- `cross_model_switches_per_min`
        is inherently a rate, not a fairness floor. budget_per_min <= 0
        (config default) means never allowed. ⚠ Caller MUST hold self._lock.

        Design decision: `_pick_fastlane_locked` does not
        call this -- a Fast-Lane-matched candidate must never be refused a
        cross-model swap on budget grounds, at any value. Left in place,
        still directly unit-tested
        (`test_budget_window_is_rolling_60s_not_a_hard_count_cap`) -- its own
        rolling-window math is unchanged and correct -- and it is
        permanently unreachable FROM THE PICK PATH. A caller reading
        `fastlane_refusal_counts()["budget"]` as always 0/absent should read
        that as STRUCTURALLY UNREACHABLE, not as "never happened to fire."

        ⚠ SCOPE. The design decision above applies to the pick path only.
        It does NOT mean this budget never governs ordinary/unregistered
        traffic, nor that the pick path is this function's only call site:
        `pop_next`'s ordinary ladder calls this gate (see the
        `ordinary_budget` refusal below the affinity fallback), and that IS
        unregistered traffic. That is where the cap belongs in the first place:
        it governs unregistered traffic instead of Fast-Lane-matched traffic.
        The gate stays live for the ordinary ladder; only the pick path
        no longer reaches it. The statement above about the pick path says
        nothing about the ordinary-ladder call site (the `ordinary_budget`
        refusal in pop_next), which is a separate call of the
        same gate.

        ⚠ THE TOKEN IS TAKEN HERE, ON THE GRANT, AND THAT IS DELIBERATE --
        but a grant is not yet a swap. The caller MUST hand the
        token back via `_fastlane_swap_refund_locked` when its grant did not
        become one. Do not move the append out of this check to "fix" that:
        the direct unit test above uses this side effect to exhaust the
        budget, so consume-then-refund is the shape used here:
        it keeps that test valid.
        """
        if budget_per_min <= 0:
            return False
        now = time.monotonic()
        while self._fastlane_swap_times and now - self._fastlane_swap_times[0] >= 60.0:
            self._fastlane_swap_times.popleft()
        if len(self._fastlane_swap_times) < budget_per_min:
            self._fastlane_swap_times.append(now)
            return True
        return False

    def _fastlane_swap_refund_locked(self) -> None:
        """Give back a token that
        `_fastlane_swap_allowed_locked` took for a grant which did NOT become
        a swap.

        The budget's subject is the swap itself: a swap is expensive (an
        unload, a load and a re-prefill), and the budget already declines to
        charge a pick that costs none of that: a pick that lands
        in a genuinely free sidecar slot is not a swap at all (nothing is
        unloaded), so the cap does not apply to it either. A grant that
        produced nothing routable costs none of it either, so keeping its
        token refuses a LATER, REAL swap by an UNREGISTERED client -- and
        the cap exists on that client's traffic to throttle real churn,
        not to manufacture refusals out of grants that did nothing.

        Pops the MOST RECENT token, not the oldest. The oldest belongs to a
        different, genuinely-completed swap whose own 60s window must keep
        expiring on schedule; taking that one back would silently extend the
        earlier swap's charge by the age gap between them.

        No-op on an empty window, so a double refund can never credit the
        budget below zero and turn the cap into a free pass.

        ⚠ Caller MUST hold self._lock.
        """
        if self._fastlane_swap_times:
            self._fastlane_swap_times.pop()

    def fastlane_refusal_counts(self) -> "dict[str, int]":
        """Sync snapshot of Fast Lane refusal counts by reason, e.g.
        {"ordinary_budget": 3}. Minor lock-skip OK for /status -- same
        convention depth() documents for the same reason. This is the public
        accessor manager.py's _fastlane_status_badge() reads across the
        queue/manager boundary; it never reaches into _fastlane_refusal_counts
        directly. Returns a copy, not the live dict, so a caller can never
        mutate queue-internal state. A lane-side "budget"
        reason cannot occur -- see _fastlane_swap_allowed_locked.
        """
        return dict(self._fastlane_refusal_counts)

    def fastlane_swap_counts(self) -> "dict[str, int]":
        """Sync snapshot of Fast Lane GRANTED-swap counts by reason, e.g.
        {"warm": 5, "exempt": 1, "budget_exempt": 2}. Same convention and the
        same queue/manager boundary as fastlane_refusal_counts() above --
        returns a copy, never the live dict. "budget_exempt" is the
        reason that covers an override of the budget -- see _pick_fastlane_locked."""
        return dict(self._fastlane_swap_counts)

    def _log_fastlane_declined(self, reason: str) -> None:
        """FASTLANE_DECLINED: _pick_fastlane_locked returned None
        without picking anything -- names WHY, since both would look identical
        (silence) otherwise. reason="staging_empty" means the pick never
        had anything to evaluate at all -- the exact ambiguity behind
        a common misreading of the logs: "no QUEUE_PRIORITY lines" could
        mean either "nothing finished" or "the pick never ran", and this
        line settles it. reason="no_candidate" means it ran a real scan
        (staging had entries) and genuinely found nothing to prioritize --
        no slot matched a rule. (A matched slot cannot be
        swap-budget-refused, so that is not a second cause of
        "no_candidate" -- see _fastlane_refusal_counts and
        _fastlane_swap_allowed_locked's docstrings.) Emitted from
        every return-None point that can actually be reached in normal
        operation, so an absent FASTLANE_DECLINED line is never itself
        ambiguous -- mirrors _log_queue_priority's own reasoning for why
        QUEUE_PRIORITY fires even on an eviction.

        This logs at DEBUG, not INFO: at INFO it would fire on every poll tick regardless
        of whether anything changed -- a large share of the entire log,
        at the poll's own fixed cadence, not correlated with real traffic.
        Not a CPU problem --
        purely a log-volume one. It is at DEBUG rather than gated on state
        change, because state-change-only would have weakened the
        contract above, not merely relocated it: _pick_fastlane_locked's
        very first guard -- the `staging_empty` early return, taken on
        every ordinary idle poll where nothing is queued -- reports the
        identical reason and identical depths for as long as the queue
        stays idle. Gating on change would emit once at the start of any
        such stretch and then go silent for its entire duration, however
        long -- reintroducing, one level up, exactly the kind of
        silence-is-ambiguous gap this log line exists to close (poll stalled vs.
        poll fine and nothing changing become indistinguishable again).
        DEBUG does not have that failure mode: emission still never
        depends on anything changing, only on-screen VISIBILITY does, and
        that is fully recoverable without touching this function again --
        this logger (turbohaul.queue) has no level of its own, so it
        inherits the root level `__main__.py` sets from `--log-level`/
        TURBOHAUL_LOG_LEVEL at process start (logging.basicConfig,
        __main__.py); restart with `--log-level debug` and this
        line returns at the same ~1Hz cadence, no code change needed. The
        cost is the full DEBUG firehose from every other DEBUG-gated
        emitter in the process for as long as that runs, not just this one
        -- there is no per-logger override wired up today, so getting this
        one line back means accepting all of them (the mechanism is the
        root-level log-level setting described above; see __main__.py for
        the logging setup, as cited
        above).

        ⚠ Caller MUST hold ``self._lock``.
        """
        log.debug(
            "FASTLANE_DECLINED reason=%s staging_depth=%d accept_depth=%d",
            reason, len(self._staging), len(self._accept_buf),
        )

    def _fastlane_priority_key(self, slot: Slot) -> "tuple[int, int] | None":
        """Single source of truth for Fast Lane priority
        ORDERING (distinct from the wall-clock fairness floor in
        _pick_fastlane_locked below, which is a time-based anti-starvation
        rescue, not a priority comparison -- do not conflate the two).

        Returns ``(rule_index, rank)`` when ``slot.fastlane`` is set, else
        ``None``. Matches the documented total order exactly:
        ``rule_index`` is the slot's position in the configured address
        list (config.py: "rules: list[FastLaneRule] -- LIST INDEX IS THE
        PRIORITY"; lower index = earlier in the list = higher priority);
        ``rank`` is the within-IP tag sub-rank (main/curator/sub_agent/
        compression). ``None`` (no rule matched -- an unlisted address)
        is, by construction, never a member of ``_pick_fastlane_locked``'s
        own candidate list either (see the ``slot.fastlane is not None``
        filter a few lines below) -- so "listed beats unlisted" is already
        this file's existing structural default; this function does not
        invent that rule, it just names the same tuple ``_pick_fastlane_
        locked``'s candidate sort already builds, so both call sites stay
        provably in agreement.

        Lower tuples sort first (better priority) -- same direction as the
        existing candidate sort below. Callers comparing two keys where
        either may be ``None`` MUST branch explicitly (see
        ``_fastlane_strictly_higher``); Python does not order ``None``
        against a tuple.
        """
        if slot.fastlane is None:
            return None
        return (slot.fastlane.rule_index, slot.fastlane.rank)

    def _fastlane_strictly_higher(
        self, candidate_key: "tuple[int, int] | None", holder_key: "tuple[int, int] | None",
    ) -> bool:
        """True iff ``candidate_key`` is STRICTLY higher Fast Lane priority
        than ``holder_key`` (both from ``_fastlane_priority_key``). Per the
        documented ordering: listed beats unlisted absolutely;
        between two listed candidates, earlier ``rule_index`` wins, then
        lower ``rank``; an EQUAL ``(rule_index, rank)`` is a real tie and is
        deliberately NOT "strictly higher" (a follow-up turn of equal rank
        keeps the warm slot; two equal-ranked agents don't bump each other).
        ``created_at`` is deliberately excluded here, unlike
        ``_pick_fastlane_locked``'s own candidate sort -- that field only
        breaks a tie between candidates BOTH already being considered for
        the SAME pick; it must never manufacture a priority difference
        between two slots that are otherwise equal.
        """
        if candidate_key is None:
            return False
        if holder_key is None:
            return True
        return candidate_key < holder_key

    def _has_strictly_higher_priority_waiting_locked(self, holder: Slot) -> bool:
        """Is there a slot WAITING whose Fast Lane priority is
        STRICTLY higher than ``holder``'s? Non-mutating -- does not remove,
        reorder, or otherwise touch anything. Bounded per buffer by that
        buffer's own cap (same convention ``_starved_other_model_locked``/
        ``_pop_first_matching_non_unloaded_from`` already use for their own
        scans) so cost stays predictable and visible rather than implicit.
        Assumes ``self._lock`` is HELD.

        Scans BOTH ``_staging`` AND ``_accept_buf``.
        ``enqueue`` parks a fresh slot in the acceptance buffer whenever
        staging is already full, so a strictly-higher LISTED claimant can be
        waiting somewhere this predicate would otherwise never look -- the same
        "mechanism present, population unreachable" shape as
        the fairness floor. Missing such a claimant
        answers False, and False is the PERMISSIVE direction (admit the
        lower-ranked client), so the blind spot would produce a priority inversion
        rather than a safe default. Mirrors ``_has_listed_waiter_for_locked``
        below, which already scans both buffers for this same reason.

        The ``examined`` budget is PER-BUFFER: a single
        counter shared across the concatenation is spent by a full staging
        buffer before the acceptance buffer is examined even once, and would
        silently double the bound whenever both are non-empty.

        ⚠ Still bounded, so still not total: a strictly-higher claimant sitting
        BEYOND either cap is not seen, and that residue also fails permissive.
        Scanning both buffers shrinks the blind spot; it does not remove it.
        """
        holder_key = self._fastlane_priority_key(holder)
        for buf, cap in ((self._staging, self.staging_max), (self._accept_buf, self.acceptance_max)):
            examined = 0
            for candidate in buf:
                if examined >= cap:
                    break
                examined += 1
                if self._fastlane_strictly_higher(
                    self._fastlane_priority_key(candidate), holder_key
                ):
                    return True
        return False

    async def has_strictly_higher_priority_waiting(self, holder: Slot) -> bool:
        """Lock-acquiring wrapper for callers outside pop_next (the grace
        loops) -- mirrors ``starved_other_model``'s exact shape (queue.py,
        the adjacent precedent for exactly this "peek from outside the
        lock, without duplicating a second copy of the underlying rule"
        need)."""
        async with self._lock:
            return self._has_strictly_higher_priority_waiting_locked(holder)

    def _has_listed_waiter_for_locked(self, model_tag: str) -> bool:
        """Is a LISTED request (``slot.fastlane is
        not None``) waiting for ``model_tag``, in staging or the acceptance
        buffer? Mirrors ``_has_strictly_higher_priority_waiting_locked``
        (above): non-mutating, bounded per buffer by that buffer's own cap
        so cost stays predictable, assumes ``self._lock`` is HELD.
        """
        for buf, cap in ((self._staging, self.staging_max), (self._accept_buf, self.acceptance_max)):
            examined = 0
            for slot in buf:
                if examined >= cap:
                    break
                examined += 1
                if slot.fastlane is not None and slot.model_tag == model_tag:
                    return True
        return False

    async def listed_waiter_model_tags(self) -> "set[str]":
        """Snapshot of every model_tag with >=1 listed waiter
        right now, taken under ``self._lock``. Callers use this BEFORE
        entering ``_registry_lock`` (lock order ``queue._lock`` ->
        ``_registry_lock``, never reversed) -- ``_lru_idle_unloadable`` runs
        under ``_registry_lock`` and must not itself await ``self._lock``.
        Exhaustive, not bounded like ``_has_listed_waiter_for_locked``: a
        truncated set here would silently under-protect, the opposite of
        this feature's purpose.
        """
        async with self._lock:
            return {
                slot.model_tag
                for buf in (self._staging, self._accept_buf)
                for slot in buf
                if slot.fastlane is not None
            }

    async def listed_waiter_priority_keys(self) -> "dict[str, tuple[int, int]]":
        """Tag -> the BEST-ranked listed
        waiter's Fast Lane priority key, so the shield can be compared
        against the party it actually protects (a queued WAITER) instead
        of a resident's last holder -- two different parties, since
        residents are keyed by model_tag, not by client, and the same tag
        can be held by more than one listed
        client at once.

        Same lock, same two buffers in the same order, and the same
        ``slot.fastlane is not None`` filter as ``listed_waiter_model_tags``
        (a sibling this method does not modify) -- same population, same
        lock-order safety (``queue._lock`` -> ``_registry_lock``, never
        reversed; called from the same pre-``_registry_lock`` site).
        Exhaustive, not bounded, for the same reason that function's own
        docstring gives: a truncated result would silently under-protect.

        "Best" is decided by literally folding through
        ``_fastlane_strictly_higher`` (the shared strict-priority comparator), not by
        reimplementing its ordering -- an equal ``(rule_index, rank)`` is a
        real tie per that function's own docstring and must not flip the
        result depending on scan order.

        NO ``None`` VALUES, EVER (a tag maps to a real
        ``(rule_index, rank)`` tuple or is simply ABSENT -- never present
        with an unresolvable value; a ``None`` value is what this
        method exists to prevent). Under the current filter,
        ``_fastlane_priority_key`` can only return ``None`` when
        ``slot.fastlane`` is ``None``, which this filter already excludes,
        so no reachable input produces one today. If a filtered-in slot's
        key were ever unresolvable anyway, that SLOT is skipped in the fold
        (logged, not silently dropped) rather than the tag being mapped to
        ``None`` or the tag being hidden -- if that left a tag with no
        resolvable waiter at all, the tag is simply absent, same as a tag
        with zero listed waiters.
        """
        async with self._lock:
            best: "dict[str, tuple[int, int]]" = {}
            for buf in (self._staging, self._accept_buf):
                for slot in buf:
                    if slot.fastlane is None:
                        continue
                    key = self._fastlane_priority_key(slot)
                    if key is None:
                        log.warning(
                            "listed_waiter_priority_keys: listed slot for "
                            "model_tag=%s has an unresolvable priority key, "
                            "skipping it in the fold",
                            slot.model_tag,
                        )
                        continue
                    current = best.get(slot.model_tag)
                    if current is None or self._fastlane_strictly_higher(key, current):
                        best[slot.model_tag] = key
            return best

    def _pick_fastlane_locked(
        self, policy: FastLanePopPolicy, warm_model_tag: str | None,
        max_scan: int = 100,
    ) -> Slot | None:
        """Fast Lane priority pick. Returns the slot to serve next, or
        None to defer entirely to the existing FIFO/affinity path below (no
        mutation in that case -- staging is left byte-identical).

        Order of operations:
        1. WALL-CLOCK FAIRNESS FIRST, measured on the OLDEST waiting slot
           with no fastlane match ("normal") WHOSE CLIENT'S FLOOR TIMER HAS
           RE-ARMED. If it has waited longer than policy.max_normal_wait_s, it
           is served NEXT regardless of any staged priority work -- this
           overrides priority order entirely. Deliberately NOT count-based
           (a count-based guard would not reliably fire under load).
           A per-client re-arm applies as well: the fairness rule grants the
           longest-waiting CLIENT exactly one full turn and then the timer
           re-arms, so a client that has been served within the window is
           passed over and the scan continues to the next-oldest normal slot.
           Serving is credited on EVERY path (a deliberately wide rule), not only
           on a floor promotion -- see _credit_floor_serve_locked.
        2. Otherwise, among staged slots that DID match a rule
           (slot.fastlane is not None), pick by (rule_index, rank, arrival
           order) -- lowest rule_index wins, then lowest rank, then earliest
           created_at breaks ties (first-come-first-served on a tie).
        3. WARM-PREFERRING, then UNCONDITIONAL: the best-priority
           candidate is checked for a cold-model jump (its model_tag !=
           warm_model_tag, and warm_model_tag is not None -- nothing is
           resident yet when it IS None, so there is no swap cost to weigh).
           No jump: granted immediately, reason "warm". A jump that is
           strictly higher-priority than every loaded resident: granted,
           reason "exempt" (the pre-existing rank exemption). Any OTHER jump
           -- the case that would otherwise be refused and skipped -- is ALSO
           granted, unconditionally, reason "budget_exempt": the intended
           behaviour (a Fast Lane match overrides the swap budget entirely)
           means a Fast-Lane-matched candidate is never refused a
           cross-model swap on budget grounds, at any
           cross_model_switches_per_min value including 0. Every branch
           therefore resolves the FIRST candidate in priority order, so the
           "try the next-best candidate on refusal" scan this step could
           otherwise describe cannot happen -- the loop's per-candidate `for`
           structure and its non-destructive-skip shape are kept rather than
           collapsed to a single check, both because collapsing would be a
           behavior-neutral rewrite that is not needed, and
           in case a future rule reintroduces a genuine refusal for
           some other candidate class, at which point the scan-and-skip shape
           is already there to receive it. See _pick_fastlane_locked's
           candidate loop below and _fastlane_swap_allowed_locked's own
           docstring for the detail.
        Bounded by max_scan, matching the sibling helpers. The winning slot
        is removed in place (remove-and-return) with the same eviction
        handling as its siblings: if disconnect_event is set it is still
        returned, flagged is_evicted=True, for caller-handled audit -- never
        silently skipped.

        ⚠ Caller MUST hold self._lock.
        """
        if not self._staging:
            self._log_fastlane_declined("staging_empty")
            return None

        now = time.monotonic()

        # --- 1. Wall-clock fairness floor (checked FIRST). ---
        # The floor grants ONE turn per CLIENT and then
        # re-arms that client's own timer, per the fairness rule: the
        # longest-waiting queued CLIENT is admitted for exactly one full turn, then
        # the timer re-arms, bounding the worst-case time a queued CLIENT goes
        # without work. A bare age test on the winning REQUEST would hold no
        # state and be recomputed from scratch every call -- so nothing would record
        # that a promotion had happened and nothing could re-arm, and one
        # client with N aged unregistered requests would take N consecutive turns
        # ahead of a rank-0 registered claimant. The rule's subject is a
        # client, so the accounting is per client, not per request.
        #
        # A client inside its re-arm window is passed over and the scan
        # CONTINUES to the next-oldest normal slot, rather than the whole floor
        # returning None: one client on cooldown must never block a DIFFERENT
        # aged client's own floor turn. That is the same failure this mechanism exists
        # to remove, pointed the other way -- and it is exactly what a GLOBAL
        # (one-promotion-at-a-time) re-arm would have produced, which is why
        # per-client is the chosen design.
        #
        # WHO is eligible is untouched: unregistered (slot.fastlane is None)
        # traffic only, by design. `examined` still counts
        # every scanned entry, so the max_scan bound is unchanged.
        self._prune_floor_ledger_locked(now, policy.max_normal_wait_s)
        oldest_normal = None
        # Scan BOTH _staging AND _accept_buf for the
        # fairness floor. An unregistered (fastlane=None) waiter aged past
        # max_normal_wait_s may reside in _accept_buf when staging was full
        # at enqueue time (see ``enqueue``). Returns the OLDEST by
        # created_at -- concatention order is not arrival order because
        # enqueue_head appendleft can place a younger entry ahead of
        # an older one. Per-buffer examined counter so max_scan bounds
        # EACH buffer, not their concatenation.
        _floor_scan_buffers = (self._staging, self._accept_buf)
        for _buf in _floor_scan_buffers:
            examined = 0
            for slot in _buf:
                if examined >= max_scan:
                    break
                examined += 1
                if slot.fastlane is None:
                    if not self._floor_timer_armed_locked(
                        slot, now, policy.max_normal_wait_s
                    ):
                        continue
                    if oldest_normal is None or slot.created_at < oldest_normal.created_at:
                        oldest_normal = slot
        if (
            oldest_normal is not None
            and (now - oldest_normal.created_at) > policy.max_normal_wait_s
        ):
            # Find oldest_normal in whichever buffer holds it and remove it.
            for buf in _floor_scan_buffers:
                for i, slot in enumerate(buf):
                    if slot is oldest_normal:
                        # The floor promotion emits a
                        # QUEUE_PRIORITY line, so that promoting an aged normal slot is never
                        # silent -- a fastlane-enabled pick must be
                        # visible in the logs. skipped/skipped_classes
                        # are the entries currently ahead of this one in staging
                        # order (mirrors _pop_first_main_locked/_compression's
                        # scanned-and-passed-over counting); reason is distinct
                        # from the rule-win path below so a fairness rescue is
                        # never mistaken for a priority-rule match.
                        skipped_classes = [
                            self._served_class_for(s) for s in list(self._staging)[:i] if s is not oldest_normal
                        ]
                        del buf[i]
                        if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                            slot.is_evicted = True
                        # This IS the one-turn wall-clock fairness floor promoting
                        # an unregistered slot (slot.fastlane is None by this
                        # branch's own oldest_normal filter above) -- observability
                        # only, never consulted for scheduling.
                        slot.floor_promoted = True
                        self._log_queue_priority(
                            slot,
                            reason="fastlane-floor",
                            skipped=i,
                            skipped_classes=skipped_classes,
                        )
                        # A SEPARATE line, deliberately not folded
                        # into QUEUE_PRIORITY above -- that record's fields are
                        # consumed elsewhere and its shape is not this code's to
                        # change. Says which client drew the turn and how long its
                        # floor is now closed for, so a re-arm can be read off the
                        # log instead of inferred from an absence. The key is
                        # hashed (_floor_key_hash); the raw IP never appears.
                        log.info(
                            "QUEUE_FLOOR_REARM client=%s window_s=%.1f waited_s=%.1f",
                            self._floor_key_hash(self._floor_client_key(slot)),
                            policy.max_normal_wait_s,
                            now - slot.created_at,
                        )
                        return slot
            return None  # pragma: no cover — oldest_normal always in a buffer

        # --- 2. Priority candidates, sorted by (rule_index, rank, arrival). ---
        candidates = []
        # Include _accept_buf — Fast Lane claims may
        # also be parked there when staging was full at enqueue time.
        for _buf in _floor_scan_buffers:
            examined = 0
            for slot in _buf:
                if examined >= max_scan:
                    break
                examined += 1
                if slot.fastlane is not None:
                    candidates.append(slot)
        if not candidates:
            self._log_fastlane_declined("no_candidate")
            return None
        # (rule_index, rank) comes from _fastlane_priority_key
        # -- the SAME function has_strictly_higher_priority_waiting uses --
        # rather than a second, separately-written tuple. created_at stays
        # inline (not part of the shared function): it only breaks a tie
        # between candidates already being considered for THIS pick, a
        # concern that function's other caller (the grace loops) must NOT
        # share -- see its own docstring for why.
        candidates.sort(key=lambda s: (*self._fastlane_priority_key(s), s.created_at))

        # --- 3. Warm-preferring scan with swap-budget gate. ---
        winner = None
        winner_swap_reason = None
        for cand in candidates:
            is_cold_jump = warm_model_tag is not None and cand.model_tag != warm_model_tag
            if not is_cold_jump:
                winner = cand
                winner_swap_reason = "warm"
                break
            # A claimant strictly higher-priority than EVERY currently loaded
            # resident is exempt from the swap budget entirely -- it never
            # calls (and never consumes) _fastlane_swap_allowed_locked, so an
            # exempt jump costs nothing against the rolling window. Vacuously
            # exempt when loaded_priority_keys is empty: an unresolvable/
            # unlisted loaded resident contributes no key, and "listed beats
            # unlisted absolutely" (this file's own _fastlane_priority_key
            # docstring) means any registered candidate already outranks it.
            cand_key = self._fastlane_priority_key(cand)
            # Route through the shared strict-priority comparator instead of
            # inlining its return statement without the None guards -- see
            # _fastlane_strictly_higher's own docstring for the documented
            # tie rule this must keep following if it ever changes.
            exempt = all(
                self._fastlane_strictly_higher(cand_key, loaded_key)
                for loaded_key in policy.loaded_priority_keys
            )
            if exempt:
                winner = cand
                winner_swap_reason = "exempt"
                break
            # Design rule (a Fast Lane match overrides the swap budget
            # entirely): `cand` here is ALREADY
            # Fast-Lane-matched -- every entry in `candidates` was filtered
            # to `slot.fastlane is not None` above -- so the per-minute swap
            # budget must never refuse it, at any `cross_model_switches_per_min`
            # value including 0. This branch only ever sees Fast-Lane-matched
            # candidates, so the budget never governs ordinary/unregistered
            # traffic here either (that traffic is
            # served by the entirely separate fairness-floor and ordinary
            # FIFO/main-lane/batch-cap paths, neither of which reaches this
            # loop) -- there is no "still capped" ordinary-traffic behavior
            # for this branch to preserve. `winner_swap_reason="budget_exempt"`
            # is distinct from a plain "budget" (rate-check-passed)
            # reason so `fastlane_swap_counts()` stays honest about WHY a
            # jump happened. `_fastlane_swap_allowed_locked` and
            # `_fastlane_swap_times` are left in place (still directly
            # unit-tested) but are structurally unreachable from here --
            # see that function's own docstring.
            winner = cand
            winner_swap_reason = "budget_exempt"
            break
        # Structural consequence, not a separate bug: `warm`, `exempt`, and
        # the unconditional grant above cover every candidate,
        # so this loop always resolves on its first iteration and `winner`
        # can never be None here when `candidates` is non-empty (checked
        # above) -- the branch below and the "try the next candidate"
        # fallback are dead code for a lane-only reason, kept
        # rather than deleted, so the mechanism is already in place
        # in case a future rule reintroduces a genuine
        # refusal reason for some other candidate class.
        if winner is None:
            self._log_fastlane_declined("no_candidate")
            return None
        self._fastlane_swap_counts[winner_swap_reason] = (
            self._fastlane_swap_counts.get(winner_swap_reason, 0) + 1
        )

        for i, slot in enumerate(self._staging):
            if slot is winner:
                # As with the
                # fairness-floor return above — this path (an actual rule
                # match) emits a QUEUE_PRIORITY line so it is never silent. reason="fastlane" is distinct
                # from "fastlane-floor" so the two guarantees (targeting
                # works vs. the starvation backstop works) stay tellable
                # apart; skipped/skipped_classes are the entries ahead of
                # the winner in current staging order.
                skipped_classes = [
                    self._served_class_for(s) for s in list(self._staging)[:i]
                ]
                del self._staging[i]
                if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                    slot.is_evicted = True
                self._log_queue_priority(
                    slot,
                    reason="fastlane",
                    skipped=i,
                    skipped_classes=skipped_classes,
                )
                return slot
        return None  # pragma: no cover — winner always in _staging

    def _pop_specific_non_unloaded(self, buf: deque, target: "Slot") -> "Slot | None":
        """Pop a SPECIFIC slot out of the middle of
        ``buf``, wherever it sits — used when starvation has already identified
        the exact aged other-model entry to drain (as opposed to the FIFO head).

        Eviction handling MIRRORS ``_pop_first_non_unloaded_from`` /
        ``_pop_first_matching_non_unloaded_from`` EXACTLY: if ``target``'s
        ``disconnect_event`` is SET it is flagged ``is_evicted=True`` and
        returned for caller-handled audit. Returns None if ``target`` is no
        longer present (defensive — cannot happen today since the caller holds
        ``self._lock`` across the whole decide-then-pop sequence in
        ``pop_next``, so nothing else can have removed it in between).

        O(N) removal (``del buf[i]``), same cost class as
        ``_pop_first_matching_non_unloaded_from``'s match removal — called AT
        MOST ONCE per ``pop_next`` invocation (only on the starvation branch),
        never inside a scan loop, so it introduces no new unbounded scan.

        ⚠ Caller MUST hold ``self._lock``.
        """
        for i, slot in enumerate(buf):
            if slot is target:
                del buf[i]
                if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                    slot.is_evicted = True
                return slot
        return None

    def _served_class_for(self, slot: Slot) -> str:
        """Best-effort served_class for QUEUE_PRIORITY logging,
        derived from the slot's client labels only.

        Resolves via the same labels-only priority ``_class_from_label`` uses
        (curator > compression > sub-agent > main) — the queue layer has no
        KV chain state, so it cannot run ``classify_event``'s full 5-way
        chain inference, only this label-only subset. An unlabeled slot
        falls back to CLASS_MAIN: ``kv_classify.EVENT_TO_CLASS`` itself maps
        both label-absent "normal chat" event types (continuation,
        guard-skip) to CLASS_MAIN.

        ⚠ NOTE: this is a DIFFERENT, narrower resolution than
        the manager's ``classify_event``-based ``resolved_class`` for the
        same request — the two CAN disagree for a given slot. Correlate the
        two logs by ``thread`` + timestamp, never by assuming equality.
        """
        return _class_from_label(slot.client_meta) or CLASS_MAIN

    def _thread_hash(self, slot: Slot) -> str:
        """Short, non-reversible correlation id for QUEUE_PRIORITY logging.

        Never the raw ``thread_id``: it can itself be an IP or first-message
        fingerprint (see ``kv_classify`` module docstring). Matches
        ``kv_classify.recompose_identity``'s existing sha256[:12] convention.
        """
        return hashlib.sha256((slot.thread_id or "").encode()).hexdigest()[:12]

    def _floor_client_key(self, slot: Slot) -> str:
        """The CLIENT identity the one-turn fairness floor accounts against.

        The fairness rule's subject is a CLIENT throughout
        (the longest-waiting queued client; how long a client has gone without
        being served a turn), but this module has no client identity of its
        own: it never sees an IP or a container name, and the module boundary forbids reaching
        into the manager's registry for one. What it DOES have is already on
        the slot:

        1. ``client_meta["ip"]`` -- stamped at admission by
           ``api/chat_completion._derive_client_meta_identity`` from
           ``request.client.host``; the same field manager.py already calls
           "client IDENTITY", and the same surface Fast Lane's own registry
           keys on for the ADDRESS half of its rule syntax (an IP address, or a
           container name). The NAME half is deliberately invisible here --
           a rule may name a client by container and another by address, and
           from this module they are one IP, which for accounting a client's
           turns is the right answer anyway.
        2. ``thread_id`` -- narrower (one conversation, not one client), used
           only when no IP was stamped.
        3. A single shared ``"unresolved"`` bucket for a slot carrying
           NEITHER. Disclosed consequence, accepted as a trade-off:
           N wholly-unidentifiable clients then share ONE
           floor timer and collectively draw one floor turn per window rather
           than N. That is strictly MORE restrictive than per-client, never
           less, and it points the same way the fairness rule already points for an
           unresolvable identity (explicitly excluded from this
           protection: exclusion is the decision, not a gap).

        Prefixed per source so an IP can never collide with a thread_id that
        happens to equal it (``kv_classify``'s module docstring notes a
        thread_id can itself BE an IP).
        """
        meta = slot.client_meta or {}
        ip = meta.get("ip")
        if ip:
            return f"ip:{ip}"
        if slot.thread_id:
            return f"thread:{slot.thread_id}"
        return "unresolved"

    def _floor_key_hash(self, key: str) -> str:
        """Short, non-reversible correlation id for a client key in logs.

        Same sha256[:12] convention (and the same reason) as ``_thread_hash``:
        the raw value can be a client IP, and this module's own security
        requirement keeps ``client_meta``/``ip`` out of anything this
        module emits. The literal ``"unresolved"`` bucket is passed through
        unhashed -- it identifies nobody, and hashing it would only make the
        one case a reader most needs to recognise unreadable.
        """
        if key == "unresolved":
            return key
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def _credit_floor_serve_locked(self, slot: Slot) -> None:
        """Record that ``slot``'s client was SERVED a turn, re-arming its
        one-turn fairness floor timer.

        Wide rule: the fairness condition is that a client
        has gone that long without being *served a turn* -- served by ANY
        path, not only floor-promoted -- so every serve resets the clock. Under
        the narrower reading a client fed steadily by ordinary FIFO would keep
        its floor clock running and draw floor turns anyway, which is the
        displace-by-waiting shape the fairness rationale exists to forbid.

        Called from ``_update_run_length_locked`` (the single choke point every
        ``pop_next`` return path already passes through) and from
        ``pop_matched_thread`` (a SECOND, SEPARATE removal path that never
        reaches ``pop_next`` -- see its own docstring). Those two are the whole
        set of serve paths in this module; ``close()`` returns slots on
        shutdown, which is not a serve.

        Known, bounded imprecision, disclosed rather than papered over: a slot
        popped here and then handed BACK via ``enqueue_head`` (the manager's
        unroutable/evict-pending requeue) has credited its client without the
        client actually receiving work. This module cannot tell the two apart
        -- whether a pop became work is decided downstream, past this module's boundary
        -- and the error is in the restrictive direction (a client waits
        longer, never shorter). Documented here.

        ⚠ Caller MUST hold ``self._lock``.
        """
        if not self._floor_ledger_active:
            return
        self._floor_last_served_at[self._floor_client_key(slot)] = time.monotonic()

    def _floor_timer_armed_locked(
        self, slot: Slot, now: float, max_normal_wait_s: float,
    ) -> bool:
        """True when the fairness floor has RE-ARMED for ``slot``'s client.

        A client with no ledger entry has never been served by
        this queue, so its floor is armed -- which makes a client's FIRST
        promotion byte-identical to the pre-existing behavior; only its 2nd-and-later
        ones become gated. Otherwise the same window the eligibility test uses
        must have elapsed since that client was last served.

        This does NOT change WHO is eligible: the floor still promotes
        unregistered (``slot.fastlane is None``) traffic only, by
        design. Eligibility is who;
        this is the re-arm.
        """
        last_served = self._floor_last_served_at.get(self._floor_client_key(slot))
        if last_served is None:
            return True
        return (now - last_served) > max_normal_wait_s

    def _prune_floor_ledger_locked(self, now: float, max_normal_wait_s: float) -> None:
        """Drop ledger entries that have already re-armed.

        Behaviour-neutral BY CONSTRUCTION, not by measurement:
        ``_floor_timer_armed_locked`` returns True for an absent entry and True
        for an entry older than the window, so the two are indistinguishable to
        every reader -- dropping one can never change a pick. It exists only to
        stop the ledger growing without bound in a long-lived process (the
        ``thread_id`` fallback keys one entry per conversation).

        Gated on a size threshold purely so the common small-ledger case does
        not pay an O(n) walk on every pop; the threshold changes when the walk
        happens, never what it leaves behind.

        ⚠ Caller MUST hold ``self._lock``.
        """
        if len(self._floor_last_served_at) <= _FLOOR_LEDGER_PRUNE_AT:
            return
        self._floor_last_served_at = {
            key: served_at
            for key, served_at in self._floor_last_served_at.items()
            if (now - served_at) <= max_normal_wait_s
        }

    def _log_queue_priority(
        self,
        slot: Slot,
        *,
        reason: str,
        skipped: int,
        skipped_classes: "list[str]",
    ) -> None:
        """QUEUE_PRIORITY: which class the queue served next and how many
        OTHER-class entries it skipped past to do it
        — confirmable by instrument instead of by operator observation.

        ``skipped``/``skipped_classes`` are the load-bearing fields:
        ``served_class`` alone cannot distinguish "compression jumped 3
        queued sub-agents" from "compression happened to be first". Counts
        only entries passed over for PRIORITY during the scan below — NEVER
        evicted/dead entries a scan discards for other reasons; those never
        enter this count. reason=fifo is always skipped=0 by construction
        (the FIFO path pops the literal head, so it structurally cannot skip
        anyone). reason=affinity is a RESERVED, deliberately UNEMITTED value
        for now (see the model-affinity block in ``pop_next`` — separate
        feature area) so an absent affinity line reads as
        "not implemented", never "logging broken". reason=fastlane (an
        actual rule match) and reason=fastlane-floor (the wall-clock
        fairness backstop promoting an aged normal slot instead) are kept
        DISTINCT, not merged like fifo's two return sites — a rule win and
        a starvation rescue are different guarantees (targeting works vs.
        the anti-starvation backstop works) and collapsing them would throw
        away exactly what a caller needs to tell them apart (both
        emitted from ``_pick_fastlane_locked``
        itself, mirroring how ``_pop_first_main_locked``/
        ``_pop_first_compression_locked`` log from inside their own picker
        rather than from ``pop_next``).

        Called even when the popped slot turns out to be evicted-in-flight:
        suppressing the line there would reintroduce the exact
        "absent line is ambiguous" defect this log line exists to remove.

        Depths are read immediately after the slot is removed from its
        buffer, before ``pop_next``'s own buffer-to-staging replenish step —
        consistently across every reason this emits for.

        ⚠ Caller MUST hold ``self._lock`` (same requirement as the pop
        helpers this is called from).
        """
        log.info(
            "QUEUE_PRIORITY served_class=%s reason=%s skipped=%d "
            "skipped_classes=%s staging_depth=%d accept_depth=%d thread=%s",
            self._served_class_for(slot),
            reason,
            skipped,
            ",".join(skipped_classes),
            len(self._staging),
            len(self._accept_buf),
            self._thread_hash(slot),
        )

    def _is_main_lane(self, slot: Slot) -> bool:
        """True when explicit client metadata identifies interactive main work."""
        meta = slot.client_meta or {}
        return any(bool(meta.get(key)) for key in self.main_lane_identity_keys)

    def _pop_first_main_locked(self) -> Slot | None:
        """Remove the oldest main-lane request across staging then acceptance.

        Main admission is a queue-level reservation only: it does not interrupt
        an active sidecar and therefore preserves the single-sidecar invariant.

        Emits QUEUE_PRIORITY on a match,
        reporting how many non-main-lane entries were scanned-and-passed-over
        (for PRIORITY, not eviction) to reach it.
        """
        skipped = 0
        skipped_classes: list[str] = []
        for buf in (self._staging, self._accept_buf):
            for i, slot in enumerate(buf):
                if not self._is_main_lane(slot):
                    skipped += 1
                    skipped_classes.append(self._served_class_for(slot))
                    continue
                del buf[i]
                if slot.state is SlotState.ACCEPT_BUFFER:
                    fsm_transition(slot, SlotState.STAGED)
                if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                    slot.is_evicted = True
                self._log_queue_priority(
                    slot,
                    reason="main-lane",
                    skipped=skipped,
                    skipped_classes=skipped_classes,
                )
                return slot
        return None

    def _is_compression(self, slot: Slot) -> bool:
        """True when the slot's resolved class is compression.

        Mirrors ``_is_main_lane``'s role as a queue-level discriminator, but
        resolves the authoritative class via the same ``_class_from_label``
        priority order the manager uses at admission (is_curator >
        is_compression > is_sub_agent > is_main), so a double-labelled
        request (e.g. a curator that also sets is_compression) resolves
        consistently with the rest of the system. None-safe: absent label
        -> _class_from_label returns None -> False.
        """
        return _class_from_label(slot.client_meta) == CLASS_COMPRESSION

    def _pop_first_compression_locked(self) -> Slot | None:
        """Remove the oldest compression-class request across staging then
        acceptance.

        Compression priority is a queue-level reservation only (exactly like
        main-lane): it does not interrupt an active sidecar, so the
        single-sidecar invariant is preserved. It lets a compression event
        (context growing too large while the harness waits on parallel
        sub-agents) skip ahead of already-queued sub-agent requests, matching
        the intended "compression skips the queue" semantics. This captures the
        same discipline as ``_pop_first_main_locked`` (eviction-aware, FSM
        transition for accept-buffer entries, bounded scan over the two
        bounded deques).

        ⚠ Caller MUST hold ``self._lock``. Emits
        QUEUE_PRIORITY on a match, reporting how many non-compression
        entries were scanned-and-passed-over (for PRIORITY, not eviction) to
        reach it.
        """
        skipped = 0
        skipped_classes: list[str] = []
        for buf in (self._staging, self._accept_buf):
            for i, slot in enumerate(buf):
                if not self._is_compression(slot):
                    skipped += 1
                    skipped_classes.append(self._served_class_for(slot))
                    continue
                del buf[i]
                if slot.state is SlotState.ACCEPT_BUFFER:
                    fsm_transition(slot, SlotState.STAGED)
                if slot.disconnect_event is not None and slot.disconnect_event.is_set():
                    slot.is_evicted = True
                self._log_queue_priority(
                    slot,
                    reason="compression",
                    skipped=skipped,
                    skipped_classes=skipped_classes,
                )
                return slot
        return None

    def _starved_other_model_locked(self, warm_model_tag: str | None) -> Slot | None:
        """The oldest staging/accept-buffer entry for a DIFFERENT model aged
        past max_other_model_wait_s, else None. Assumes self._lock is HELD.
        ONE source of truth: pop_next drains it, the grace loops break out on it.

        Scans BOTH ``self._staging`` and ``self._accept_buf`` -- a starvation-
        eligible waiter may reside in either buffer depending on admission
        pressure (see ``enqueue``, overflow into _accept_buf when
        staging is full). Returns the OLDEST by ``created_at``, not the
        first hit: ``enqueue_head`` does ``appendleft`` and can place
        a younger entry at position 0 of _staging; the first other-model entry
        found by a left-to-right scan is therefore NOT guaranteed to be the
        oldest across concurrent insertions. An unconditional ``break``
        after the first starved or non-starved other-model entry would suppress a
        genuinely starved older waiter parked behind a younger one -- a
        starvation blind spot. Bounded by ``self.staging_max`` per buffer, same
        convention ``_pop_first_matching_non_unloaded_from`` already uses.
        """
        starved_slot = None
        # Scan BOTH buffers; track oldest by
        # created_at, not first-match (a naive concat breaks FIFO).
        _buffers = (
            (self._staging, self.staging_max),
            (self._accept_buf, self.acceptance_max),
        )
        for _buf, _max in _buffers:
            examined = 0
            for candidate in _buf:
                if examined >= _max:
                    break
                examined += 1
                if candidate.model_tag == warm_model_tag:
                    continue
                if (time.monotonic() - candidate.created_at) > self.max_other_model_wait_s:
                    if starved_slot is None or candidate.created_at < starved_slot.created_at:
                        starved_slot = candidate
        return starved_slot

    async def starved_other_model(self, warm_model_tag: str | None) -> Slot | None:
        """Lock-acquiring wrapper for callers outside pop_next (the grace loops)."""
        async with self._lock:
            return self._starved_other_model_locked(warm_model_tag)

    async def pop_next(
        self,
        *,
        warm_model_tag: str | None = None,
        fastlane_policy: "FastLanePopPolicy | None" = None,
        grace_active: "frozenset[tuple[str, str]] | None" = None,
        room_available: bool = False,
    ) -> Slot | None:
        """Pop the next STAGED slot for activation. Returns None if empty.

        LOOP form, no recursion. After consulting
        the bounded eviction-aware helper, if the result is None (either buf
        empty or all examined entries pre-evicted), drain accept-buffer into
        staging and retry once with the helper. Bounded by helper's max_drain.

        Model-affinity (single-mutator-safe parallelism support): when the
        worker_loop passes ``warm_model_tag`` (the model currently warm in the
        active/idle holder) AND staging is non-empty, prefer popping a slot for
        that same model so the warm sidecar is reused — UNLESS:
          - the OLDEST other-model request anywhere in staging has aged past
            ``max_other_model_wait_s`` (head starvation: scanned
            across the whole staging deque, not just index 0 — a same-model
            follow-up is appendleft'd to the head on every grace-match, so
            index 0 alone can never surface a starved other-model entry), OR
          - we've already popped ``max_consecutive_same_model`` of this model
            in a row (batch cap — fairness; still keyed off the literal FIFO
            head, unchanged),
        in which case the FIFO head (batch-cap case) or the specific aged
        entry (starvation case) is forced to drain.
        ``warm_model_tag=None`` (default for every existing caller) preserves
        the exact pre-existing FIFO path with ZERO behavior change. cap=1 +
        wait=0.0 also collapse to strict FIFO even when a tag is supplied.

        ``grace_active`` (victim-aware grace protection): a set of (thread_id,
        model_tag) pairs each currently inside a resident's grace window.
        This file never resolves that set itself -- the manager builds it
        fresh on every call (victim-aware: the designated victim gets NO
        grace timer, since its context is saved and the slot
        is torn down immediately -- a victim's pair is never included) and
        hands it in, same shape as ``warm_model_tag``/``fastlane_policy``.
        A matching staging entry is made INVISIBLE to the ENTIRE pick ladder
        below for the duration of this one call -- held aside before the
        ladder runs, spliced back in the ``finally`` below no matter what the
        ladder returned, AT ITS ORIGINAL POSITION relative to the entries that
        stayed visible. It can never be the value THIS call returns, whichever
        rung of the ladder would otherwise have taken it -- the guarantee that every
        other loaded client keeps its normal grace window does not
        care which caller or which branch pops.

        The splice restores POSITION as well as membership, and that matters.
        A splice that appended to the TAIL would let a protected entry drift
        toward the tail across repeated protected calls. That can look
        harmless: it can never starve (an ordinary poppable tail entry the
        instant grace lapses). But the harm is real, and it is a different harm.
        A held entry's ``created_at`` is untouched, so a
        registered slot's rank genuinely is unharmed -- but
        ``_starved_other_model_locked`` does not read rank. It `break`s at the
        FIRST other-model entry in staging ORDER, on its own stated
        precondition that staging preserves FIFO order for every entry that
        is never re-inserted at the head -- the precondition a tail splice
        would falsify. Drift an aged other-model entry behind a newer one and the
        newer one settles that scan, is found not aged, and the ordinary
        starvation breakout reads SATISFIED while it is violated: blind to the
        exact request it exists to rescue. So the entry could not starve for
        the reason the drift argument checks, and could starve for a reason it
        does not check.
        See ``_restore_held_aside_locked``.
        ``grace_active=None``/empty (every existing caller) is byte-for-byte
        today's behavior: nothing is held aside, nothing is restored.
        Scoped to ``_staging`` only (pop_next skips only those staging
        entries) -- a grace-held
        follow-up sitting in ``_accept_buf`` instead (only reachable via a
        staging-at-capacity overflow, scanned only by the main-lane and
        compression-class reservation rungs) is a disclosed, out-of-scope
        edge for now.

        ``room_available`` (room is checked first: if there is a free
        sidecar slot, nothing is evicted): True iff
        the manager's registry has a free slot RIGHT NOW. This file never
        computes room itself (the module boundary forbids the cross-lock reach into the
        manager's registry) -- the manager resolves it and hands it in, same
        shape/staleness contract as ``warm_model_tag`` (a HINT; a stale value
        is corrected downstream, never fatal). Consulted ONLY at the ordinary
        cross-model swap-budget gate below: the budget's subject is
        a swap costing an unload+load+re-prefill, so a pick that
        lands in genuinely free room is not a swap at all and the budget has
        no standing to refuse it, regardless of how exhausted it is.
        ``room_available=False`` (every caller except the cap>=2 dispatch
        loop) preserves the exact pre-``room_available`` gate behavior --
        those callers (the cap<=1 body, the same-model fan-out refills) only
        ever reach this gate when a resident already occupies the relevant
        slot, so "no known free room" is also the correct fact for them, not
        merely a safe default.
        """
        async with self._lock:
            held_aside: list[Slot] = []
            order_before_hold: list[Slot] = []
            # A slot that a strictly-higher LIVE
            # CLAIM outranks is held aside for this ONE call, exactly as a
            # grace-protected slot is. The rule: a lower-ranked client is not
            # admitted ahead of a higher-ranked client that is already
            # waiting. The claimant is already waiting -- it is simply not in
            # either buffer, because ``_defer_unroutable`` hands it to
            # ``_requeue_after_backoff``, which awaits the backoff BEFORE it
            # re-enqueues. In practice a claimant can be absent
            # most of its wait, and one admission can be decided by a few ms of
            # backoff phase between two clients one rank apart.
            #
            # Reusing the hold-aside rather than adding a second mechanism is
            # deliberate: it already makes an entry invisible to the ENTIRE
            # ladder (not just the Fast Lane rung) and already splices it back
            # in the ``finally`` AT ITS ORIGINAL POSITION, which is precisely
            # the two properties this needs.
            #
            # ⛔ LISTED SLOTS ONLY (``s.fastlane is not None``). Unlisted
            # traffic is never held: it is not what that rule orders, and holding it
            # would let one listed claimant starve ordinary work -- converting
            # an ordering fix into a starvation bug.
            # ⛔ STRICTLY higher, via the shared comparator: an EQUAL
            # ``(rule_index, rank)`` is deliberately not "higher", so a peer
            # never holds a peer.
            _gov_key = (
                fastlane_policy.governing_claim_key
                if fastlane_policy is not None else None
            )
            if grace_active or _gov_key is not None:
                kept: deque = deque()
                for s in self._staging:
                    _grace_held = bool(
                        grace_active and (s.thread_id, s.model_tag) in grace_active
                    )
                    _claim_held = bool(
                        _gov_key is not None
                        and s.fastlane is not None
                        and self._fastlane_strictly_higher(
                            _gov_key, self._fastlane_priority_key(s)
                        )
                    )
                    if _grace_held or _claim_held:
                        held_aside.append(s)
                    else:
                        kept.append(s)
                if held_aside:
                    # Snapshot the pre-split order BEFORE the
                    # swap below. This is what the `finally` restores against;
                    # taken here because it is the last instant the original
                    # arrangement of held and visible entries still exists.
                    order_before_hold = list(self._staging)
                    # Shrink the visible cap by the same count so the ladder's
                    # own replenish-from-accept_buf steps below can never let
                    # the deque grow past its real staging_max once the
                    # held-aside entries are spliced back in the `finally`.
                    self._staging = kept
                    self.staging_max -= len(held_aside)
            try:
                # === Fast Lane guarded pick — ABOVE the rest of the ladder
                # so it is reachable regardless of warm_model_tag/staging state. An
                # equivalent scan placed INSIDE the branch below would be
                # unreachable in production (every real caller
                # eventually passes a non-None warm_model_tag once anything is
                # loaded). fastlane_policy=None (every existing caller, and this
                # one when the feature is off) skips this entirely — ZERO
                # behavior change from today. ===
                if fastlane_policy is not None:
                    # Arm the floor's per-client serve ledger
                    # the first time a policy is handed in. The floor is
                    # structurally unreachable without one, so a process that
                    # never enables Fast Lane never writes an entry -- keeping
                    # the feature-off path free of both behavior change and
                    # memory cost, this file's own convention. Set BEFORE the
                    # pick so the credit that follows this call lands, and
                    # harmless on the first call: an empty ledger reads as
                    # "armed" for every client, which is exactly the pre-existing
                    # behavior.
                    self._floor_ledger_active = True
                    # Secondary hardening, separate from the two log helpers above:
                    # top up staging
                    # from accept_buf BEFORE the pick runs, matching the FIFO
                    # path's own retry-drain below (a full while-drain, not the
                    # single-item `if` used at every post-pick replenish site) --
                    # so fastlane's view can never be stale relative to what has
                    # actually been admitted. Only fires when the feature is on
                    # (inside this same guard), so a feature-off caller sees ZERO
                    # behavior change, matching the existing contract. This covers a
                    # narrow case: staging can only be simultaneously
                    # empty-at-pick-time while accept_buf holds something when
                    # staging was AT CAPACITY at some earlier admission (an
                    # overflow condition), not under ordinary one-at-a-time
                    # trickling.
                    while self._accept_buf and len(self._staging) < self.staging_max:
                        s = self._accept_buf.popleft()
                        fsm_transition(s, SlotState.STAGED)
                        self._staging.append(s)
                    picked = self._pick_fastlane_locked(fastlane_policy, warm_model_tag)
                    if picked is not None:
                        if self._accept_buf and len(self._staging) < self.staging_max:
                            tail = self._accept_buf.popleft()
                            fsm_transition(tail, SlotState.STAGED)
                            self._staging.append(tail)
                        self._update_run_length_locked(picked)
                        return picked
                # fastlane_policy is None (feature off), OR the pick above found no
                # match this call (_pick_fastlane_locked's own contract: returns
                # None to defer entirely to the path below, no mutation) -> fall
                # through to today's ladder, byte-for-byte unchanged: main-lane
                # reservation, then compression skip, then affinity/FIFO/
                # starvation/batch-cap. Re-evaluated fresh every call; nothing
                # here is sticky.
                #
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
                # A compression-class request skips ahead of queued
                # sub-agent requests (and everything else) so context can't grow
                # unchecked while the harness waits on parallel work. Sits behind
                # the main-lane reservation (the active agent's slot is served
                # first, by design), ahead of affinity/FIFO. Admission-only:
                # never interrupts an active sidecar.
                compression = self._pop_first_compression_locked()
                if compression is not None:
                    if self._accept_buf and len(self._staging) < self.staging_max:
                        tail = self._accept_buf.popleft()
                        fsm_transition(tail, SlotState.STAGED)
                        self._staging.append(tail)
                    self._update_run_length_locked(compression)
                    return compression
                if warm_model_tag is None or not self._staging:
                    # === Existing FIFO path — ZERO behavior change. ===
                    slot = self._pop_first_non_unloaded_from(self._staging)
                    if slot is not None:
                        # reason=fifo is always
                        # skipped=0 by construction — this path pops the literal
                        # head, so it structurally cannot skip anyone. Logged
                        # here, before the replenish below, to match the
                        # post-removal/pre-replenish depth convention the
                        # main-lane/compression reasons also use.
                        self._log_queue_priority(
                            slot, reason="fifo", skipped=0, skipped_classes=[],
                        )
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
                    returned = self._pop_first_non_unloaded_from(self._staging)
                    if returned is not None:
                        self._log_queue_priority(
                            returned, reason="fifo", skipped=0, skipped_classes=[],
                        )
                    self._update_run_length_locked(returned)
                    return returned

                # === Model-affinity path (warm_model_tag supplied, staging non-empty). ===
                # reason="affinity" is a RESERVED, deliberately UNEMITTED
                # QUEUE_PRIORITY value for now — this
                # branch belongs to the starvation
                # logic and is not handled here. Left as a comment, not silently
                # dropped, so an absent affinity line reads as "not implemented
                # yet", never "logging broken" — the exact ambiguity this log
                # exists to remove, one level up.
                # Decide BEFORE popping whether affinity applies or the FIFO head is
                # forced. Both branches replenish staging from the buffer afterward
                # (one entry, mirroring the FIFO path) so depth invariants hold.
                #
                # Starvation must look at the OLDEST
                # OTHER-MODEL entry anywhere in staging, not just index 0 — a
                # same-thread/same-model follow-up is appendleft'd to the head on
                # every grace-match (manager.py's enqueue_head), so index 0 is
                # almost always same-model and a genuinely starved other-model
                # request sitting at index >= 1 was invisible to an index-0-only
                # check, so the starvation timer (max_other_model_wait_s) could
                # effectively never fire for such a request as long as
                # index 0 stayed same-model.
                # Staging preserves FIFO order for every entry that is never
                # re-inserted at the head, so the FIRST other-model entry hit while
                # scanning left-to-right IS the oldest other-model entry present —
                # no need to keep scanning once it's found. Bounded by
                # self.staging_max, same bound _pop_first_matching_non_unloaded_from
                # already uses for its own scan (staging can never exceed
                # staging_max by construction, but the explicit bound keeps this
                # scan's cost visible/obvious rather than implicit).
                #
                # This call is deliberately unguarded: it has no `.fastlane`
                # check, unlike the two grace loops that are guarded
                # (_starvation_breakout_candidate, manager.py). This predicate
                # has exactly THREE consumers; two of them are guarded.
                # Left unguarded here by design rather than omission --
                # the two grace loops and this one do
                # OPPOSITE things with the same predicate:
                #   - the grace loops SHORTEN a resident's grace window --
                #     guarding them stops a claim from cutting a non-victim's
                #     grace short.
                #   - this call DRAINS the starved request itself, off
                #     staging, to be served next -- it does not touch any
                #     resident's grace at all. Guarding it would mean a
                #     Fast-Lane-tagged request aged past
                #     max_other_model_wait_s is NEVER drained here -- listed
                #     clients starving in exactly the scenario unlisted ones
                #     are rescued from, the inversion this feature removes,
                #     not an instance of it.
                # Ordinary FIFO starvation breakout is unchanged, and it
                # protects exactly this call.
                # Guarding it would look like closing an asymmetry while
                # actually breaking the fairness floor the rule was built to keep.
                # This is not an exempted path.
                # The branch runs whenever _starved_other_model_locked returns an
                # entry whose age exceeds self.max_other_model_wait_s (a constructor
                # argument of this class); no .fastlane check and no carve-out
                # applies to it.
                # This branch still never reaches
                # the ordinary-traffic swap cap below, but that is a
                # STRUCTURAL fact of the if/elif/else ladder
                # (the cap is only gated in the `else` arm,
                # which this `if` branch never falls into),
                # not a designed protection --
                # it is not an
                # intentional exemption.
                starved_slot = self._starved_other_model_locked(warm_model_tag)
                head = self._staging[0]
                head_is_other = head.model_tag != warm_model_tag
                batch_cap_hit = (
                    self._consecutive_same_model >= self.max_consecutive_same_model
                )
                # Residency: the count cap must NOT force a MODEL
                # SWAP while same-model work is still queued for the resident and the
                # other-model head has not aged past the starvation window. At one GPU
                # slot (max_parallel_sidecars=1) a forced swap M->N->M evicts + reloads
                # a model the very next queued item wants back — avoidable churn.
                # Genuine head-starvation (time-based, `starved_slot`:
                # detected ANYWHERE in staging, not just index 0) STILL forces the
                # swap so a truly-waiting other-model request is never starved. The
                # count cap now only forces the FIFO head when doing so causes NO
                # avoidable swap — i.e. the head is already the resident's model
                # (`not head_is_other`, a harmless same-model drain). When the head is
                # a DIFFERENT, not-yet-starved model we fall to the affinity branch,
                # which drains queued same-model work and only pops the other-model
                # head if no same-model work remains. Same-model run-length under real
                # contention is thus bounded by max_other_model_wait_s
                # (time) rather than by max_consecutive_same_model (count), which
                # is the bound the code actually enforces.
                force_head = starved_slot is not None or (batch_cap_hit and not head_is_other)
                if starved_slot is not None:
                    # Genuine starvation: drain the SPECIFIC aged other-model entry,
                    # wherever it sits — not necessarily index 0.
                    returned = self._pop_specific_non_unloaded(self._staging, starved_slot)
                elif force_head:
                    # batch_cap_hit-only case (unchanged behavior): force the FIFO
                    # head to drain the same-model request. head_is_other is False
                    # here by construction of force_head, so this is always a
                    # harmless same-model drain, never an avoidable swap.
                    returned = self._pop_first_non_unloaded_from(self._staging)
                else:
                    # Prefer a same-model staging entry; fall back to FIFO head if
                    # no same-model entry within the scan bound.
                    returned = self._pop_first_matching_non_unloaded_from(
                        self._staging, warm_model_tag, max_scan=self.staging_max,
                    )
                    if returned is None:
                        # Why the gate is skipped when there is free room: the cap gates
                        # unregistered/non-lane cross-model swaps. The one cross-model case
                        # that stays outside it is the one-turn floor's designated client --
                        # the only exemption by design, and it is handled structurally (see
                        # below, where the floor client is returned before this ladder).
                        # (The starvation timer max_other_model_wait_s is not an exemption;
                        # see the comment at `starved_slot`, above, for why.)
                        # _pop_first_matching_non_unloaded_from just scanned
                        # the FULL staging_max bound for a same-model entry
                        # and found none, so the FIFO head about to be popped
                        # here is GUARANTEED cross-MODEL -- but cross-model is
                        # NOT the same claim as "costs a swap": the budget's subject is
                        # unload+load+
                        # re-prefill, and room is checked first --
                        # a pick landing in a slot that is genuinely FREE
                        # right now evicts nothing, so it is not a swap and
                        # the budget has no standing over it. Gating
                        # that case would refuse or miscount exactly this
                        # case --
                        # a routine parallel placement into an already-free
                        # cap>=2 slot -- because it could not tell "genuinely
                        # swapping the one warm resident" from "filling room
                        # a DIFFERENT resident's departure just freed" (covered by the
                        # end-to-end test that parks and wakes a waiter who still skips a
                        # protected sibling, and by the free-room
                        # tests of the ordinary gate). ``room_available``
                        # is that fact, handed down by the manager
                        # the same way ``warm_model_tag`` already is (the module boundary: this
                        # file never computes room itself). The floor-client exemption
                        # is carried for free by NOT gating here: the
                        # one-turn floor client is returned from
                        # _pick_fastlane_locked before pop_next ever reaches
                        # this ladder (mutually exclusive return points -- see
                        # that function's own top section). A genuinely
                        # starved other-model entry (the `if starved_slot is
                        # not None` arm above) also never falls into this
                        # `else`, but that is a structural fact of the
                        # ladder, not a second exemption -- see the
                        # comment there.
                        # Feature-off = zero behavior change (this file's own
                        # convention): only gated when Fast Lane is enabled,
                        # since cross_model_switches_per_min lives inside
                        # FastLaneConfig and has no meaning otherwise.
                        # The gate's decision and the token
                        # it spends are two separate facts, because a
                        # GRANT is not yet a SWAP. `gate_applies` is the
                        # short-circuit: when the feature
                        # is off or there is free room the gate is not called
                        # at all, so no token is taken and none can be owed.
                        gate_applies = (
                            fastlane_policy is not None and not room_available
                        )
                        budget_taken = (
                            gate_applies
                            and self._fastlane_swap_allowed_locked(
                                fastlane_policy.cross_model_switches_per_min
                            )
                        )
                        if gate_applies and not budget_taken:
                            self._fastlane_refusal_counts["ordinary_budget"] = (
                                self._fastlane_refusal_counts.get("ordinary_budget", 0) + 1
                            )
                            returned = None
                        else:
                            returned = self._pop_first_non_unloaded_from(self._staging)
                            # A token pays for an
                            # unload+load+re-prefill. Two outcomes cost none
                            # of that and must hand it back:
                            #   returned.is_evicted -- the helper flagged this
                            #     slot dead and handed it back anyway;
                            #     manager.py's `if slot.is_evicted: ...;
                            #     continue` then skips _route_or_reserve
                            #     entirely, so nothing is ever swapped.
                            #   returned is None -- no slot at all.
                            # ⚠ The `is None` arm is UNREACHABLE FROM HERE and
                            # is retained deliberately: staging is
                            # non-empty at this point (`self._staging[0]` was
                            # read above and the affinity scan removes nothing
                            # when it returns None), and
                            # _pop_first_non_unloaded_from only returns None on
                            # an EMPTY deque. It costs one comparison and
                            # guards a future caller that does reach it. A dedicated
                            # swap-budget refund test
                            # (the is-None disjunct)
                            # pins both structural facts rather than
                            # pretending to exercise the branch.
                            if budget_taken and (
                                returned is None or returned.is_evicted
                            ):
                                self._fastlane_swap_refund_locked()
                # Replenish staging from buffer (mirror FIFO path: one entry if room).
                if returned is not None and self._accept_buf and len(self._staging) < self.staging_max:
                    tail = self._accept_buf.popleft()
                    fsm_transition(tail, SlotState.STAGED)
                    self._staging.append(tail)
                self._update_run_length_locked(returned)
                return returned
            finally:
                if held_aside:
                    # Restore BEFORE splicing back, so the invariant
                    # len(self._staging) <= self.staging_max holds at every
                    # instant an observer could see it.
                    self.staging_max += len(held_aside)
                    self._staging = self._restore_held_aside_locked(
                        order_before_hold, held_aside,
                    )

    def _restore_held_aside_locked(
        self, order_before_hold: "list[Slot]", held_aside: "list[Slot]",
    ) -> "deque[Slot]":
        """Splice grace-held entries back at their ORIGINAL position.

        This is used instead of ``self._staging.extend(held_aside)``,
        which would put them on the TAIL. ``created_at`` is not what a tail splice
        damages -- a held entry's age, and so a registered slot's Fast Lane rank,
        is always untouched. What it damages is POSITION, and that lands on
        ``_starved_other_model_locked``, which `break`s at the FIRST
        other-model entry it meets on its own stated precondition that
        staging preserves FIFO order for every entry that is never
        re-inserted at the head. Tail-splicing an older other-model entry
        behind a newer one lets the newer one settle that scan; not being aged,
        it produces None, and the ordinary starvation breakout reads SATISFIED
        while it is violated. Restoring position makes that precondition true
        again instead of working around it, which is why that function needs
        no change.

        Rebuilds rather than inserts, because index arithmetic cannot survive
        what the ladder does in between (entries deleted from the middle by
        the main-lane/compression rungs, entries appended from ``_accept_buf``
        by any replenish step). Walking ``order_before_hold`` and keeping
        whatever still exists is exact regardless of how many left or from
        where:

        * a held entry is always kept -- the ladder cannot have popped one, it
          could not see them;
        * a formerly-visible entry is kept only if it survived the ladder;
        * an entry that entered staging DURING the call is in neither set, and
          is appended last -- correct, not a fallback: it is genuinely the
          newest thing present and the tail is where a fresh admission belongs.

        Identity is compared with ``id()``, which is sound here specifically
        because every slot involved is strongly referenced for the whole call
        by ``order_before_hold``, ``held_aside`` or ``self._staging`` -- none
        can be collected mid-call and have its id reused. Slot has no
        ``__eq__``/``__hash__`` of its own, and equality on ``slot_id`` would
        be a weaker claim than the object identity actually being tracked.

        ⚠ Caller MUST hold ``self._lock``.
        """
        held_ids = {id(s) for s in held_aside}
        survivor_ids = {id(s) for s in self._staging}
        ids_before_hold = {id(s) for s in order_before_hold}
        restored: deque = deque()
        for slot in order_before_hold:
            if id(slot) in held_ids or id(slot) in survivor_ids:
                restored.append(slot)
        for slot in self._staging:
            if id(slot) not in ids_before_hold:
                restored.append(slot)
        return restored

    def _update_run_length_locked(self, returned: "Slot | None") -> None:
        """Post-pop bookkeeping: the same-model run-length counter, and the
        per-client fairness-floor serve ledger.

        Increment when the returned slot's model matches the last popped model,
        else reset to 1. None (empty pop) leaves both untouched.

        The floor credit lives HERE rather than at a second,
        parallel set of call sites because this function is already the single
        choke point every one of ``pop_next``'s return paths passes through
        (fastlane pick, main-lane reservation, compression, FIFO, FIFO retry,
        and the affinity/starvation/batch-cap ladder). Adding six more call
        sites would create exactly the drift risk a choke point exists to
        prevent -- one new rung added later without its credit, and the ledger
        silently under-counts. The floor re-arm test pins every
        rung individually so a missed one fails loudly rather than quietly.

        ⚠ Caller MUST hold ``self._lock``.
        """
        if returned is None:
            return
        self._credit_floor_serve_locked(returned)
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
            self._log_queue_arrival(slot, landed="staging", via="enqueue_head")
            self._invoke_on_enqueue()

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
        self,
        thread_id: str,
        model_tag: str,
        *,
        inbox: "Optional[asyncio.Queue[Slot]]" = None,
    ) -> Slot | None:
        """Atomic find + remove + eviction check.

        Built on the shared chokepoint: the scan lives in
        `_pop_first_matching_locked`. This method supplies
        only its match
        predicate and its `via` label — the scan, removal, eviction-flag,
        floor-credit, and admitted-without-pick epilogue described below all
        run inside the chokepoint, once, instead of duplicated here.
        See `_pop_first_matching_locked`'s own docstring for the WRITTEN
        REASON this matcher needs no temporal guard (an unrelated future
        matcher on a weaker identity signal, e.g. IP, will).

        Scans staging in order, deletes the first (thread_id, model_tag) match,
        then performs the eviction-check INLINE before returning. If the
        matched slot's disconnect_event is set, flags is_evicted=True so the
        worker_loop's is_evicted branch handles it (audit + fail-future).
        Bounded by len(staging) ≤ staging_max=100, so no extra drain cap needed.

        Symmetric with pop_next's eviction handling (for
        consistency — eviction can land on a grace-rematch slot just as
        easily as a fresh-staging slot).

        This is a SECOND, SEPARATE removal path from pop_next --
        a match here is served WITHOUT pop_next, _pick_fastlane_locked, or
        any of the main-lane/compression/FIFO ladder ever running. That is a
        property of the code above, not of any measurement: a grace-window
        same-thread continuation -- the ordinary shape of one agent's
        sequential turns -- is served the instant the engine frees up and
        this match succeeds, never reaching the picker. The one ordering it
        does apply is the same-client rank rule below: a follow-up that a
        better-ranked request of its own client is waiting behind is declined.
        (A QUEUE_PRIORITY line count is a floor on picks, not a total;
        see _log_admitted_without_pick for the reason. This claim about
        the mechanism stands on the code alone and needs no log
        evidence.) Deliberately NOT
        changed to consult
        _pick_fastlane_locked here: that would make Fast Lane preempt an
        already-in-progress grace-window match for any other client
        (admission-only, never preempt). Made OBSERVABLE instead --
        see _log_admitted_without_pick.

        Because this path bypasses pop_next entirely it also
        bypasses ``_update_run_length_locked``, where every other serve credits
        the fairness floor's per-client ledger -- so it credits directly. It
        must: the fairness clock is time without being SERVED a turn, by
        ANY path (a deliberately wide rule), and a grace-window
        continuation is the ORDINARY shape of one agent's sequential turns.
        Left uncredited, a client whose whole workload is same-thread
        continuations would look permanently unserved to the floor and could
        keep drawing floor turns on top of the work it was already getting --
        precisely the hole the wide reading closes.

        Same-client rank rule: a same-thread follow-up yields to a
        better-ranked waiting request of the same client (same ``rule_index``,
        strictly lower ``rank``). Such a follow-up is skipped, stays waiting
        in its buffer at its position, and is served later: by pop_next in
        rank order, or by this matcher once nothing better waits. Equal rank,
        a better rank under another ``rule_index``, and unlisted traffic never
        cause a skip. A request that is already RUNNING is never affected;
        this only decides which waiting request is served next.

        Resident inbox: when ``inbox`` is given, the follow-up is also weighed
        against the requests waiting in it (a request routed to the loaded
        model's inbox never sits in staging). The inbox contents are taken
        with the public queue API under the queue lock, passed to the
        same-client rank check as ``also_waiting``, and put back in their
        original order on every exit, including an exception. Nothing is
        served from the inbox here; only the staged follow-up can be returned.
        With ``inbox=None`` (the default) only staging and the acceptance
        buffer are weighed.
        """
        if not thread_id:
            return None

        def _predicate(slot: Slot) -> bool:
            return slot.thread_id == thread_id and slot.model_tag == model_tag

        async with self._lock:
            _inbox_all: list[Slot] = []
            if inbox is not None:
                while not inbox.empty():
                    try:
                        _inbox_all.append(inbox.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            try:
                return self._pop_first_matching_locked(
                    _predicate,
                    via="pop_matched_thread",
                    also_waiting=_inbox_all,
                )
            finally:
                for s in _inbox_all:
                    inbox.put_nowait(s)

    async def pop_matched_ip(
        self,
        ip: str,
        model_tag: str,
        grace_started_at: float,
        *,
        decline_sink: "dict | None" = None,
    ) -> Slot | None:
        """Secondary grace-loop matcher for returning clients,
        ported onto the shared chokepoint.

        Matches a queued request by (ip, model_tag) when thread_id differs
        (conversation grew past anchor) but the client IP matches the
        completing slot's client. Temporal guard: only matches slots
        submitted AFTER the completing slot entered grace (genuine
        returning clients, not pre-existing queued slots).

        Port shape: the scan itself is the SHARED CHOKEPOINT
        (``_pop_first_matching_locked``), the same one ``pop_matched_thread``
        uses. The full identity contract, including the
        temporal guard, is composed into the predicate closure below,
        exactly as the chokepoint's own docstring prescribes for a weaker
        identity signal. The per-slot decline bookkeeping the pre-chokepoint
        hand-rolled scan carried is filled post-hoc from the live staging
        state (same lock: this call holds ``self._lock`` throughout).

        IPs are parsed via ipaddress.ip_address() and compared as parsed
        objects — raw-string comparison is a bypass risk because it
        ignores format variations. A malformed IP is rejected (no match).
        No denylist is applied: the relevant denylist (ssrf_guard) is an
        SSRF deny-private check that blocks RFC1918/loopback/IMDS — the
        exact range typical clients use. Applying it here
        would make the same-IP match never succeed. Outbound-connection protection is
        the gateway's job, not queue.py's.
        Atomic find+remove under self._lock — no double-claim possible.

        A candidate is skipped while a better-ranked request of the same
        client (same ``rule_index``, strictly lower ``rank``) is waiting; it
        stays in staging and is served later (see ``pop_matched_thread``).
        """
        if not ip:
            return None
        # Parse the completing slot's IP up front. Malformed → no match.
        # No denylist here: is_blocked_ip is an SSRF deny-private check
        # (blocks RFC1918/loopback/IMDS). Typical clients ARE on private IPs,
        # so that check would make the same-IP match never succeed.
        # Parsed comparison (not raw-string) prevents format-based bypass.
        try:
            anchor_ip = ipaddress.ip_address(ip)
        except ValueError:
            return None

        def _slot_ip_ok(slot: Slot) -> bool:
            raw = (slot.client_meta or {}).get("ip")
            if not raw:
                return False
            try:
                return ipaddress.ip_address(raw) == anchor_ip
            except ValueError:
                return False  # malformed slot IP → no match (fail-safe)

        def _predicate(slot: Slot) -> bool:
            # Identity-contract order mirrors the pre-chokepoint hand-rolled
            # scan: model first (cheap, no side effects), then the temporal
            # guard, then the parsed-IP comparison.
            if slot.model_tag != model_tag:
                return False
            if slot.created_at <= grace_started_at:

                return False
            return _slot_ip_ok(slot)

        async with self._lock:
            # The model-mismatch count: the one decline we
            # can trigger on demand. Counted on the same staging state the
            # chokepoint is about to scan (no await in between: same lock, same
            # contents), so at the fall-through it equals the pre-chokepoint
            # hand-rolled scan's per-iteration count — the only sink state the
            # manager ever reads.
            _model_mismatch = sum(
                1 for s in self._staging if s.model_tag != model_tag
            )
            matched = self._pop_first_matching_locked(
                _predicate,
                via="pop_matched_ip",
                guard_anchor=grace_started_at,
            )
        if matched is not None:
            if decline_sink is not None:
                # baseline on SERVED turns — an instrument that only speaks
                # during failure has no baseline.
                decline_sink["reason"] = "matched"
            return matched
        if decline_sink is not None:
            # THE FALL-THROUGH, highest-value instrument.
            # ⛔ CONTENTS AND TIMESTAMPS, NEVER A BARE DEPTH COUNT: arrivals
            # usually land in staging, so "not in staging" and "in staging but the scan
            # did not see it" are BOTH live and a count cannot separate them. These do:
            #   landed AFTER the scan   -> scan_ts vs the waiter's created_at
            #   snapshot was STALE      -> contents disagree with staging_depth
            #   present and passed over -> contents show it, no reason fired => an UNKNOWN
            #                              decline path nobody has named
            # Port note: ``examined`` is the staging depth at the
            # fall-through — with the chokepoint's default max_scan (100) and the
            # staging cap the scan and the depth coincide, and the wrapper no
            # longer sees the per-iteration counter. That is the value the
            # read surface needs to separate the cases above.
            decline_sink["reason"] = "no_candidate_matched"
            decline_sink["scan_ts"] = time.time()
            decline_sink["model_tag_mismatch"] = _model_mismatch
            decline_sink["staging_depth"] = len(self._staging)
            decline_sink["examined"] = len(self._staging)
            decline_sink["contents"] = [
                (s.slot_id, s.model_tag, s.created_at,
                 (s.client_meta or {}).get("ip"))
                for s in list(self._staging)[:12]
            ]
        return None


    def _log_admitted_without_pick(self, slot: Slot, *, via: str, guard_anchor: float | None = None) -> None:
        """QUEUE_ADMITTED_NO_PICK: a slot was served WITHOUT
        pop_next's ladder ever running for it, so _pick_fastlane_locked (and
        main-lane/compression/FIFO) never ran either. ``via`` NAMES the
        specific queue.py mechanism responsible -- a bare "no pick ran" flag
        would be exactly as unidentifying as today's silence; the whole
        point is WHICH path, not just that one exists.

        A QUEUE_PRIORITY line count is a FLOOR on picks, never a
        total: some admissions reach the engine via something
        that logged nothing identifying it. pop_next's
        model-affinity exit returns a slot WITHOUT calling
        _log_queue_priority on any of its three sub-paths, and
        does so ON PURPOSE: see the banner comment on that
        branch, `reason="affinity" is a RESERVED, deliberately
        UNEMITTED QUEUE_PRIORITY value`. Every other exit logs;
        that one does not.

        On a single-sidecar deployment with a warm resident the
        affinity exit is the COMMON path, so many served
        requests have no QUEUE_PRIORITY line by design, and the
        number of requests served can exceed the number of
        QUEUE_PRIORITY lines. A request taken by the manager's
        HIT route (described below) never enters this queue at
        all.

        Both facts are accounted for in the code, not in this
        log line: the affinity exit is in pop_next, and the HIT
        route is in the manager's _route_or_reserve.

        Call site: it is called
        from EXACTLY ONE place, ``_pop_first_matching_locked`` (the shared
        chokepoint), which passes through whatever ``via`` its caller gave
        it -- pop_matched_thread's is still "pop_matched_thread", unchanged,
        since it still names the matcher, not the chokepoint doing the
        calling on its behalf. This method's own promise -- that
        pop_matched_thread is the ONLY non-pop_next way a slot leaves this
        queue and gets served -- holds, and the set of ``via`` values that
        can reach the call is just each matcher's own label;
        `_pop_first_matching_locked` is the only place that calls it. A
        structural test (the chokepoint-completeness test
        file) pins the "exactly one Python call site, and it is this
        chokepoint" half of that claim, so a future matcher that hand-rolls
        its own scan-and-serve epilogue instead of going through the
        chokepoint fails a test rather than shipping unnoticed -- the class
        of defect (a private, un-unified copy) this whole unification exists to make
        structurally harder to write. If a caller adds another such path in
        the future, it MUST call this with its own distinct ``via`` value,
        not reuse "pop_matched_thread".

        How OFTEN this fires depends on
        the traffic shape and on the
        grace-loop configuration: whether
        grace-window continuations are
        common or rare is not estimated
        here.

        The main way requests get served without a QUEUE_PRIORITY line is
        real and identifiable in the code, independently of this
        docstring:
        it is manager.py's **HIT route**. In
        _route_or_reserve, a request whose model already has a live
        resident is handed straight to that resident's inbox
        (`r.inbox.put_nowait(slot)`) under the registry lock and RETURNS --
        this module's enqueue() is never called. Such a request therefore
        never enters _staging/_accept_buf and is invisible to queue.py end
        to end, rather than merely unlogged by it.

        ⇒ The two facts compose: warm-resident
        traffic bypasses this queue entirely (HIT route), and traffic that
        DOES reach pop_next can still be served by its affinity exit
        without a QUEUE_PRIORITY line. Between them they account for the
        requests that are served without one. That
        decision code is NOT in queue.py; it is
        named here so that a reader can find the mechanism
        in the code.

        ⚠ Caller MUST hold ``self._lock``.
        """
        if guard_anchor is None:
            log.info(
                "QUEUE_ADMITTED_NO_PICK via=%s model=%s fastlane_matched=%s thread=%s",
                via,
                slot.model_tag, slot.fastlane is not None, self._thread_hash(slot),
            )
        else:
            # The two timestamps the temporal guard compares,
            # on the line itself — a later regression of the guard is
            # then visible as an admission with created_at <= grace_started_at.
            log.info(
                "QUEUE_ADMITTED_NO_PICK via=%s model=%s fastlane_matched=%s thread=%s "
                "created_at=%.3f grace_started_at=%.3f",
                via,
                slot.model_tag, slot.fastlane is not None, self._thread_hash(slot),
            slot.created_at, guard_anchor,
            )

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

    def staging_snapshot(self) -> "list[Slot]":
        """AWAIT-FREE ``list()`` snapshot of ``_staging``.

        The sync twin of :meth:`peek_staging`, for a caller that must observe the
        queue from a synchronous stretch and cannot ``await`` the lock. Same
        "minor lock-skip OK" precedent ``depth()`` and ``queue_snapshot()``
        already establish for this exact class of read.

        Its consumer is the grace-loop admission gate
        (``slot.observe_for_grace_match`` -> ``client_has_outstanding_work``),
        which decides whether a returning request may claim a warm resident by
        IP. Two properties are LOAD-BEARING for that consumer, and both fail
        SILENTLY toward "this client is alone" — the permissive direction — if
        they are ever broken:

        ⚠ **RETURNS THE LIVE ``Slot`` OBJECTS. Never copy, wrap or rowify them.**
        The caller excludes the candidate request from its own observation by
        OBJECT IDENTITY (``slot is candidate``: two slots from one
        client can be value-identical, so a value-based exclusion would drop a
        genuine sibling). Hand back dicts and the candidate can never match, so
        it counts ITSELF as outstanding work, the gate answers "concurrent"
        forever and the fix it guards becomes an inert no-op with a green suite.
        ⇒ This is why ``queue_snapshot()`` — which returns ``list[dict]`` rows —
        was REJECTED for this call site rather than reused.

        ⚠ **``_staging`` ONLY, and NEVER capped.** Not ``_accept_buf`` (a
        different set from the one ``pop_matched_ip`` scans), and no ``limit``:
        a truncated view under-counts outstanding work, which again reads as
        "alone". ``queue_snapshot()``'s ``limit=50`` is correct for /status and
        wrong here — the same field, judged by what THIS consumer does when the
        value is wrong.
        """
        return list(self._staging)


    def depth(self, *, inbox_waiting: int = 0, claims_waiting: int = 0) -> dict:
        """Sync snapshot of queue depths. Minor lock-skip OK for /status.

        The undercount addressed here: a
        request whose model already has a live resident is handed straight
        to that resident's inbox by manager._route_or_reserve's HIT route
        and never reaches this module's enqueue() at all (see
        _log_admitted_without_pick's docstring above) -- so ``_staging`` +
        ``_accept_buf`` alone undercount how many requests are actually
        waiting. queue.py has no visibility into resident inboxes and must
        not import them; the caller (manager.py, which owns the resident
        registry) supplies the live inbox count. Default 0 keeps every
        existing caller (including the no-arg ``self.queue.depth()`` calls
        already in this codebase) byte-identical to today's output.

        ``queue_depth_total`` is an ADDITIONAL field, not a redefinition of
        ``staging_queue_depth``: telemetry.py's on_queue_state persists
        ``staging_queue_depth`` verbatim into a durable per-event log keyed
        on that exact meaning (current _staging length), and the FE renders
        it as a ratio against ``staging_queue_max`` (the "1/100" bar) --
        silently folding inbox waiters into either number would corrupt
        both a durable historical field and a capacity percentage that has
        nothing to do with resident inboxes. A caller that wants the
        corrected total reads the new key; nothing existing changes meaning.

        The parallel blind spot -- requests that
        FAILED to route (MISS / count-cap / VRAM-over-commit) are parked
        in the manager's ``_fastlane_claims`` registry by
        ``_defer_unroutable`` and never re-enter ``_staging`` until the
        backoff requeue fires. During that window they are invisible to
        staging + accepted + inbox_waiting, so the operator reads 0 while
        claims genuinely wait. ``claims_waiting`` closes that gap.
        A request waiting in a resident's inbox is counted exactly once. It
        keeps its claim, marked as parked on that resident, until its own turn
        starts; that claim is counted here (the claims snapshot includes it),
        and the manager's inbox count, which feeds inbox_waiting, leaves it
        out while the request really sits in that inbox, so it is never counted
        by both inbox_waiting and claims_waiting. Only live
        claims are counted here. Default 0 keeps callers that do not supply
        it byte-identical to prior output.
        """
        staging = len(self._staging)
        accepted = len(self._accept_buf)
        return {
            "acceptance_buffer_depth": accepted,
            "staging_queue_depth": staging,
            "staging_queue_max": self.staging_max,
            "acceptance_buffer_max": self.acceptance_max,
            "queue_depth_total": staging + accepted + inbox_waiting + claims_waiting,
        }

    def queue_snapshot(self, *, limit: int = 50) -> "list[dict]":
        """Sync snapshot of waiting requests
        for /status's ``queue.waiting[]`` -- same "lock-skip OK for /status"
        precedent ``depth()`` just above already establishes for this exact
        class of read.

        Enumerates ``_staging`` THEN ``_accept_buf``, each in current FIFO
        order, combined and capped at ``limit`` -- by design;
        an accept-buffer slot already carries a resolved
        ``slot.fastlane`` (``match_fastlane`` runs BEFORE either buffer
        decision is made), so excluding it would withhold data already
        known, and a waiting request is meant to be visible the whole
        time, not only once it clears an internal capacity technicality.
        ``state`` is what distinguishes the two populations on the row --
        literally ``SlotState.ACCEPT_BUFFER`` vs ``STAGED`` -- no separate
        field needed.

        ``position`` is this row's index in the combined, capped snapshot --
        an honest label for "where it sits right now", not a promise of
        eventual serve order: Fast Lane priority is only applied inside
        ``pop_next``'s pick, never while a slot rests in either deque.

        By design (visible the WHOLE time): a row whose
        ``fastlane`` is not None is NEVER dropped by ``limit`` -- every Fast
        Lane claimant is emitted, and ``limit`` then governs only how many
        NORMAL rows ride along. Two consequences, both deliberate and
        neither a redefinition of ``position``, which still carries its
        TRUE pre-cap index exactly as described above:
          * the returned rows are no longer necessarily a CONTIGUOUS prefix
            -- a preserved claimant past the cap leaves a gap in
            ``position`` where the normal rows it outranks were dropped;
          * the returned row COUNT can exceed ``limit`` when more than
            ``limit`` claimants are waiting at once. That is the bound the
            "never truncate a claimant" rule buys, and it is bounded by the
            claimant population rather than by the deque, so it cannot be
            driven up by ordinary traffic.
        Why this is needed at all: because Fast Lane priority is applied
        only inside ``pop_next``, a claimant has NO ordering privilege while
        it rests here -- it sits wherever it was enqueued, so at depth > 50
        a plain prefix cap could truncate the one row that must stay visible.

        Redacted for security: explicit
        dict only, ``thread_id_prefix`` (never a full ``thread_id``), never
        ``client_meta``/``ip``/``asdict(slot)``/``__dict__``. ``fastlane``'s
        ``label`` here is the frozen match's own value -- ``queue.py`` has
        no visibility into ``_live_fastlane_label`` (a manager-only method,
        same import boundary ``depth()``'s own ``inbox_waiting`` parameter
        already respects) and must not import it. manager.py's
        ``status_snapshot()`` overwrites ``label`` with the LIVE-resolved
        value on every row before this reaches ``/status`` (status rows show
        the live-resolved label).

        Does NOT include ``likely_victim``: that needs the resident
        registry (``_is_designated_unload_target_locked``), which this module has
        no access to for the same reason. manager.py's ``status_snapshot()``
        attaches it to every row after calling this.
        """
        fastlane_rows: list[dict] = []
        normal_rows: list[dict] = []
        for position, slot in enumerate(list(self._staging) + list(self._accept_buf)):
            fl = slot.fastlane
            # A claimant is never dropped, so only NORMAL rows are
            # budget-checked here -- and the check happens BEFORE the row dict
            # is built, so a deep queue costs an iteration, not an allocation.
            if fl is None and len(normal_rows) >= limit:
                continue
            fastlane_view = (
                {
                    "rule_index": fl.rule_index,
                    "rank": fl.rank,
                    "label": fl.label,
                    "fastlane_rule": fl.raw_address,
                }
                if fl is not None
                else None
            )
            row = {
                "position": position,
                "slot_id": slot.slot_id,
                "model_tag": slot.model_tag,
                "thread_id_prefix": (slot.thread_id or "")[:8],
                "state": slot.state.value,
                "waited_s": round(time.monotonic() - slot.created_at, 1),
                "fastlane": fastlane_view,
                "floor_promoted": slot.floor_promoted,
            }
            if fl is not None:
                fastlane_rows.append(row)
            else:
                normal_rows.append(row)
        # Claimants first (all of them), then whatever normal-row
        # budget survives them; re-sorted so the surface still reads in true
        # queue order. With no claimants past the cap this is byte-identical
        # to a plain prefix cap -- positions 0..limit-1, contiguous.
        budget = max(0, limit - len(fastlane_rows))
        rows = fastlane_rows + normal_rows[:budget]
        rows.sort(key=lambda r: r["position"])
        return rows

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

    Follow-up with matching thread_id within window → warm-slot reuse.
    Bounded by max_extensions to prevent starvation.
    """

    def __init__(self, grace_seconds: float, max_extensions: int = 5) -> None:
        self.grace_seconds = grace_seconds
        self.max_extensions = max_extensions
        self._started_at: float | None = None
        self.thread_id: str | None = None
        self.model_tag: str | None = None
        self.extension_count = 0
        # Paused countdown — stores the remaining_s() at
        # pause time so the countdown FREEZES during active token generation
        # without touching _started_at (which _grace_active_exclusions reads
        # for victim protection). None = not paused.
        self._paused_remaining: float | None = None

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
        # When paused (active serving), return the frozen
        # remaining value instead of free-running wall-clock. This freezes the
        # countdown during token generation WITHOUT touching _started_at.
        if self._paused_remaining is not None:
            return self._paused_remaining
        elapsed = time.monotonic() - self._started_at
        return max(0.0, self.grace_seconds - elapsed)

    def is_paused(self) -> bool:
        """True when the countdown is frozen mid-generation."""
        return self._paused_remaining is not None

    def pause(self) -> None:
        """Freeze the countdown at its current remaining
        value. Called when a grace-window follow-up begins active prefill/generation.
        Does NOT modify _started_at — only captures the remaining time, so
        _grace_active_exclusions's victim-protection is unaffected.
        Idempotent: a double pause is a no-op (keeps the first freeze point)."""
        if self._paused_remaining is None:
            self._paused_remaining = self.remaining_s()

    def resume(self) -> None:
        """Resume a paused countdown from its frozen point.
        Re-anchors _started_at so the window continues counting from where it
        was paused (not from the original start). No-op if not paused."""
        if self._paused_remaining is not None:
            self._started_at = time.monotonic() - (
                self.grace_seconds - self._paused_remaining
            )
            self._paused_remaining = None

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
        self._paused_remaining = None


class IdleHotTimer:
    """Tracks the IDLE_HOT window after the queue drains.

    Fresh request with same model_tag → ACTIVE on warm slot.
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