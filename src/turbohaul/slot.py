"""Slot dataclass + state enum + thread_id derivation.

The slot, its state machine, and the thread_id prefix-hash fallback.
"""
import asyncio
import dataclasses
import enum
import hashlib
import time
import uuid
from typing import Any, Optional


class SlotEvictedError(Exception):
    """Raised when a slot's completion_future is failed due
    to client-disconnect eviction in the queue.

    DEDICATED Exception subclass — NOT a reuse of asyncio.CancelledError —
    because:
    - CancelledError inherits from BaseException (Py 3.8+), which slips past
      ``except Exception:`` handlers and ``raise from None`` chains. Routes
      catching ``Exception`` would silently drop CancelledError-shaped
      eviction signals.
    - Semantic clarity: 'we evicted because the client disconnected' is a
      different fault domain from 'event loop cancelled this task'. Mixing
      them costs us metric / 4xx-vs-5xx clarity.

    Routes catch this in their non-streaming await chain and surface HTTP 499
    (client_closed_request).
    """


class VramOverCommitError(RuntimeError):
    """The manager cannot admit a (non-resident) model because the target
    GPU is VRAM-over-committed by live co-residents, and no idle resident could be
    evicted to free the card within the bounded evict-pending retry budget (a full
    grace cycle). A TRANSIENT capacity condition — the client should retry after
    ``retry_after_s``. Subclasses RuntimeError so existing ``except RuntimeError``
    handlers still catch it, but the routes map it to HTTP 503 + Retry-After (a real
    backpressure signal) instead of a silent HTTP 200 / 500.
    """

    def __init__(self, message: str, cause: str = "vram_over_commit",
                 retry_after_s: int = 10):
        super().__init__(message)
        self.cause = cause
        self.retry_after_s = retry_after_s


class SlotState(str, enum.Enum):
    """States of the slot state machine (10 states total)."""

    RECEIVED = "RECEIVED"
    ACCEPT_BUFFER = "ACCEPT_BUFFER"
    STAGED = "STAGED"
    LOADING = "LOADING"
    LOADING_FAIL = "LOADING_FAIL"
    ACTIVE = "ACTIVE"
    GRACE = "GRACE"
    GRACE_BUSY = "GRACE_BUSY"
    ACTIVE_MATCH = "ACTIVE_MATCH"
    POPPED = "POPPED"
    IDLE_HOT = "IDLE_HOT"
    COLD = "COLD"


@dataclasses.dataclass
class Slot:
    """A single queued or active request slot."""

    slot_id: str
    model_tag: str
    state: SlotState
    prompt: str = ""
    context: list[dict] | None = None
    thread_id: str = ""
    port: int | None = None
    pid: int | None = None
    extension_count: int = 0
    client_meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    created_at: float = 0.0  # monotonic time at creation
    started_active_at: float = 0.0  # monotonic when entered ACTIVE
    grace_started_at: float = 0.0  # monotonic when entered GRACE
    # Monotonic timestamp of entering LOADING. manager.py's /status
    # elapsed_s computation reads this via getattr; elapsed_s stays 0.0 until it
    # is assigned, however long the load runs. Stamped at the stage_to_loading
    # audit point (manager.py,
    # the real load path — not the manifest-not-found bail-out).
    started_loading_at: float = 0.0
    admission_ctx_len: int = 0  # incoming context size recorded at admission
    # Restore classifier: incoming turn-hash chain recorded at admission
    # (parallel to admission_ctx_len). _prefix_hash_chain(messages) computed at the
    # chat_completion admission site so the manager restore path can classify the
    # request by prefix-VALIDITY vs the pinned clean bin (not by char length alone).
    # Default [] = no chain threaded -> restore gate SKIPS (fail-safe, never blind).
    admission_hash_chain: list[str] = dataclasses.field(default_factory=list)
    # Named engine operation for the dashboard pill.
    # Tracks the high-level engine phase: "kv_restore" | "kv_save" | "prefill" | "decode" | "idle" | "stream" | "unload".
    engine_op: str = "idle"
    # Optional asyncio.Future, set by submit(wait_for_completion=True) so caller can
    # await the slot's completion response. worker_loop sets the result after
    # completion_fn returns.
    completion_future: Any = None
    # SSE streaming pass-through:
    # When client sends stream:true, the route uses submit_for_streaming() instead
    # of submit_and_wait(). The slot stays ACTIVE for the full stream lifetime —
    # worker_loop sets stream_ready_event after llama-server health-200 (handle
    # stored on stream_handle), the route opens its own httpx.stream() to the
    # sidecar, yields SSE chunks, and signals stream_done_event when the gen
    # exhausts (or on client disconnect / error). Only then does worker_loop
    # advance ACTIVE → GRACE.
    stream_ready_event: Any = None  # asyncio.Event, set by worker_loop on ACTIVE
    # stream_ready_event ALSO gets force-set on a placement
    # FAILURE (via _fail_completion_future, manager.py), not only on genuine
    # ACTIVE -- mirrors stream_done_forced_reason immediately below: stamped
    # FIRST so there is no race between the reason and the event (asyncio is
    # single-threaded). None means "genuinely ready" (stream_handle IS set --
    # every success path assigns it strictly before calling .set(), so the two
    # can never disagree); non-None means the route's ready-wait unblocked on
    # a failure instead, and chat_completion.py's existing stream_handle-is-
    # None guard reads this to show the real reason instead of its original
    # generic fallback message.
    stream_ready_failed_reason: str | None = None
    stream_done_event: Any = None   # asyncio.Event, set by route on stream close
    stream_handle: Any = None       # SidecarHandle assigned when ACTIVE
    # stream_done_event carries no "who/why" of its own -- the manager itself
    # force-sets it in a few cooperative-unwind paths (a drain-cap, an outer
    # cancel/exception sweep, a cancelled active-match) to unblock a route
    # that may never actually finish. A reader downstream of the wait needs
    # to tell that apart from the route's own genuine done signal, so each
    # force-set site stamps a short reason here FIRST (asyncio is
    # single-threaded, so there is no race between the reason and the event).
    # None means "not forced" -- either genuinely route-completed, or never
    # touched by a force-set path at all.
    stream_done_forced_reason: str | None = None
    # The SSE route accumulates the streamed generated
    # assistant text (content + reasoning merged as <think>...</think>{content},
    # mirroring _merge_reasoning_into_content) and stashes it here BEFORE setting
    # stream_done_event, so the manager can reconstruct the engine-view warm_chain
    # for the STREAMING forced clean-restore (such workloads stream). None
    # when unknown (non-thinking/tool-call/parse-miss) -> gate safe-degrades.
    streamed_assistant_text: str | None = None
    # Client-disconnect eviction signal.
    # Lazy-init `None` (NOT default_factory=asyncio.Event).
    # default_factory binds the Event to whatever loop is current at dataclass
    # construction time — wrong-loop fragility in tests + BootInventory replay.
    # Routes attach an Event constructed FROM their own request handler scope
    # (correct loop). Non-HTTP callers (BootInventory, internal probes) pass None.
    disconnect_event: Optional[asyncio.Event] = None
    is_evicted: bool = False  # set by pop_*_non_evicted_from when caller-disconnected
    # Completion-cache single-flight carrier. When a
    # non-streaming request becomes the cache LEADER, submit_and_wait stashes the
    # completion-key here BEFORE enqueue so the _process_slot WRITE site can cache
    # the result + resolve the leader's single-flight future keyed by it. None for
    # riders / streaming / cache-disabled requests -> the WRITE site is a no-op.
    completion_cache_key: str | None = None
    # Fast Lane: admission-time priority match. None = no rule matched / feature off.
    fastlane: Any = None  # FastLaneMatch | None
    # Fast Lane: True only when the one-turn wall-clock fairness floor is what
    # served this slot (queue.py's _pick_fastlane_locked, reason="fastlane-floor").
    # That floor promotes UNREGISTERED (no-rule-matched) traffic only -- never set
    # for a slot that won on its own fastlane rule match. Observability only; does
    # not affect scheduling.
    floor_promoted: bool = False
    # Set when a VRAM-over-commit eviction picked
    # for this slot's Fast Lane claim (the off-manifest-card widening) landed
    # on a DIFFERENT card than the manifest pin. None = no relocation, the
    # manifest's own main_gpu/split_mode governs (today's behavior, unchanged).
    # Consumed by _route_or_reserve's retry pass (placement re-resolution) and
    # threaded into _reserve_and_start_locked via its existing main_gpu_override/
    # split_mode_override params -- not a new mechanism.
    fastlane_relocated_main_gpu: int | None = None
    fastlane_relocated_split_mode: str | None = None
    # True while the two fields above were set by the reclaim-first decision
    # (a one-card placement chosen because the card fits once its idle models
    # are unloaded), not by the eviction path above. The decision is taken again
    # on every routing pass; this flag lets it clear its own override when it
    # no longer holds, and keeps it from touching an override it did not set.
    fastlane_reclaim_first: bool = False
    # True while this slot sits
    # back in the staging queue after a load failure classified as OOM
    # (load_verify_log.classify_load_failure) with a live disconnect_event.
    # Set at the OOM-requeue point on the live spawn path (_spawn_for_
    # resident -- the only reachable "if not healthy:" branch); cleared the moment
    # the slot is re-admitted (a fresh LOADING attempt starts) or fails for
    # any non-requeue reason (non-OOM, unknown cause, or never-fits).
    # submit_and_wait reads this to decide whether the 7200s timeout_s
    # applies to the current wait --
    # never consulted anywhere else.
    oom_requeue_pending: bool = False

    @classmethod
    def new(
        cls,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict[str, Any] | None = None,
        admission_ctx_len: int = 0,
        admission_hash_chain: list[str] | None = None,
    ) -> "Slot":
        return cls(
            slot_id=f"slot-{uuid.uuid4().hex[:12]}",
            model_tag=model_tag,
            state=SlotState.RECEIVED,
            prompt=prompt,
            context=context,
            thread_id=thread_id,
            client_meta=client_meta or {},
            created_at=time.monotonic(),
            admission_ctx_len=admission_ctx_len,
            admission_hash_chain=admission_hash_chain or [],
        )


def _normalise_ip_for_exclusion(ip: str) -> str:
    """Fold an address to one form FOR EXCLUSION CHECKS ONLY.

    ⚠ READ THIS BEFORE REUSING IT. Normalising an address widens whatever it is
    fed to, and "wider" has OPPOSITE safety directions for the two things this
    module could use it for:

      * widening the MATCH key   -> more requests match a warm resident that is
                                    not theirs.               ⛔ UNSAFE
      * widening the EXCLUSION   -> more addresses are refused a match.  ✅ SAFE

    So this is used ONLY on the denylist side of
    :func:`grace_ip_match_eligible`. The match key itself is compared verbatim.
    The design decision is that the IPv4-mapped/IPv6 SPLIT stays: a split costs a
    missed optimisation, a fusion costs correctness. That decision forbids
    normalising to make MORE things match. It does not forbid — it requires —
    that a denylisted address still be caught when it arrives in its other
    notation, or the guard is defeated by spelling.

    ⛔ TO WHOEVER IS HERE TO DELETE ONE HALF AS REDUNDANT — READ THIS FIRST.
    A config that lists BOTH spellings of a gateway makes this redundant, so for those
    addresses this normalisation is genuinely redundant. That is not an argument
    for removing it:

        Listing both spellings makes the normalisation redundant ONLY for the
        addresses someone listed. An operator who lists ONE spelling is
        protected by the normalisation alone.

    Belt-and-braces, not duplication. Remove the normalisation and a
    single-spelling denylist entry silently stops protecting; remove the
    second spelling from the config and you are relying on this function.
    (Applying this call on ONE side only is caught by
    ``test_a_denylisted_mapped_address_catches_the_bare_form``.)
    """
    s = ip.strip().lower()
    if s.startswith("::ffff:"):
        s = s[len("::ffff:"):]
    return s


def _arrived_after_grace_started(slot, grace_started_at) -> bool:
    """Did this waiter arrive AFTER the anchor entered grace?

    ⭐ WHY THIS PREDICATE AND NOT A COUNT. A plain count asks "does this client have
    other work outstanding?" and refuses if so -- but when the client's own next turn
    arrives, THAT TURN IS THE OUTSTANDING WORK. One waiter means two opposite things:

        1 waiter = my own next turn        -> must ALLOW
        1 waiter = one concurrent sibling  -> must REFUSE   (a fan-out of TWO)

    A count cannot separate them; loosening to ">1 refuses" would ALLOW the
    fan-out of two, which is the base case of the fusion the cap>=2 gate exists to
    prevent. The separating fact is WHEN the waiter arrived.

    ⭐ AND THIS IS NOT A NEW DISCRIMINATOR. It is the SAME rule
    ``pop_matched_ip(ip, model_tag, grace_started_at)`` already uses to PICK. The failure mode
    is not the threshold -- it is THE COUNT AND THE MATCH USING DIFFERENT RULES,
    so the guard vetoed candidates the matcher would never have chosen. Aligning them
    adds no exposure the matcher does not already carry: a sibling genuinely dispatched
    after the anchor entered grace is indistinguishable from a returning turn by every
    signal available, and ``pop_matched_ip`` already has exactly that residual.

    ⚠ THREE PROPERTIES THIS RELIES ON, any of which would make it silently
    wrong:
      * SAME CLOCK. ``Slot.created_at`` is ``time.monotonic()`` (the factory below) and
        ``slot.grace_started_at`` is ``time.monotonic()`` (manager.py). A wall-vs-
        monotonic mix looks identical and compares as garbage.
      * FAIL-SAFE DEFAULT. ``created_at`` defaults to ``0.0``, which is <= any real
        grace start, so an unset value reads as PRE-EXISTING -> counted -> REFUSE.
      * EVERY CONSUMER CAN APPLY IT. The staging side can; a bare ``qsize()`` inbox
        count cannot, which is why the inbox half needs separate handling. Half an
        alignment still refuses.

    ⛔ AFFIRMATIVE RULE, as everywhere else in this gate: returning ``True`` (exclude
    this waiter from the sibling count) requires POSITIVE PROOF. Anything unreadable,
    missing, non-numeric or NaN returns ``False`` -> the waiter is counted -> the match
    is refused. Uncertainty costs one slow turn; a wrong exclusion fuses two
    conversations onto one KV cache.
    """
    if not isinstance(grace_started_at, (int, float)) or isinstance(grace_started_at, bool):
        return False
    created = getattr(slot, "created_at", None)
    if not isinstance(created, (int, float)) or isinstance(created, bool):
        return False
    if created != created or grace_started_at != grace_started_at:  # NaN
        return False
    return created > grace_started_at


def observe_for_grace_match(
    other_staging_slots, candidate, inbox_waiting: int, *, grace_started_at
):
    """Build the outstanding-work observation FOR THE GRACE-LOOP site.

    ⛔ THE TRAP THIS EXISTS TO PREVENT, and it is silent. The route-site observer
    (``api.chat_completion.observe_outstanding_work``) reports
    ``queue_depth_total`` = staging + accepted + inbox waiters. That is right at the
    ROUTE, where identity is derived BEFORE ``submit()`` so the request is not yet in
    staging and cannot count itself.

    **At the grace loop the candidate IS in staging by construction** — that is what
    the matcher scans. Reuse the route-site observer here and the count is never
    zero, the discriminator always answers "concurrent", the IP match never fires,
    and **the IP match is an inert no-op with no exception and no log.**
    That is the same shape as a guard that cannot fire; it just arrives one level
    down.

    So the split differs by site:

        staging          -> VISIBLE here (the matcher walks it). Enumerate it, and
                            EXCLUDE the candidate.
        resident inboxes -> UNSEEN at both sites (``Resident.inbox`` exposes
                            ``qsize()`` only). Passed in, per the same rule
                            ``queue.depth(inbox_waiting=...)`` already follows:
                            the caller supplies what this layer must not import.

    Returns ``(visible_client_metas, unseen_outstanding)`` for
    ``client_has_outstanding_work``.

    ⚠ The candidate is excluded by OBJECT IDENTITY (``is``), never by ``slot_id`` or
    by ``(ip, model_tag)``. Two slots from one client can be indistinguishable by
    value, and an exclusion that is too WIDE drops a genuine sibling and lets the
    fusion through — the same failure-direction rule that governs the shared-address check.
    """
    metas = []
    for slot in other_staging_slots or ():
        if slot is candidate:
            continue
        if _arrived_after_grace_started(slot, grace_started_at):
            # This waiter did not exist when the anchor
            # entered grace, so it cannot be a member of a fan-out dispatched
            # alongside it -- it is this conversation's own next turn. Counting
            # it would make the guard refuse the exact case this exclusion serves.
            continue
        cm = getattr(slot, "client_meta", None)
        if isinstance(cm, dict):
            metas.append(cm)
        else:
            metas.append({})  # unattributable -> the predicate refuses on it
    try:
        unseen = int(inbox_waiting or 0)
    except (TypeError, ValueError):
        unseen = 1  # uncountable waiters are still waiters — refuse, never assume 0
    return metas, unseen


def parse_shared_addresses(raw) -> tuple[str, ...]:
    """Parse the comma-separated shared-address denylist from config/env.

    One parser so two call sites cannot drift into two different splitting rules.
    Blank entries are dropped; nothing else is interpreted, because an address is
    matched by :func:`_normalise_ip_for_exclusion`, not by this.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def grace_ip_match_eligible(ip, shared_addresses=()) -> bool:
    """May this client address be used to match a GRACE window?

    Address matching lets a returning request claim its own resident's grace window on
    ``ip + model_tag`` instead of ``thread_id + model_tag``, because the thread_id
    churns. This is the guard on that key.

    ⭐ WHY THIS IS STRICTER THAN THE FAIRNESS FLOOR, WHICH KEYS ON THE SAME FIELD.
    ``queue._floor_client_key`` already treats one IP as one client and tolerates
    both a shared gateway address and an ``"unresolved"`` shared bucket. That is
    correct THERE and wrong HERE, and the reason is not the key — it is what each
    consumer does when the key is wrong:

      * fairness floor : two fused clients share one floor timer and collectively
                         draw FEWER turns. Its own docstring: "strictly MORE
                         restrictive, never less."                    ✅ fails safe
      * grace matching : two fused clients means one client's request is served on
                         ANOTHER client's warm resident.        ⛔ fails permissive

    ⇒ **A known-shared address must NOT match in the grace loop even though it is
    tolerated in the floor.** That is not an inconsistency to harmonise away; it is
    the correct consequence of judging a key by what its consumer does when the key
    is wrong. (This is a deliberate design decision. If you are here to make these two
    agree, this paragraph is why you should not.)

    ⛔ AND THIS GUARD IS FAIL-OPEN BY CONSTRUCTION. ``shared_addresses`` is a
    DENYLIST: it excludes the addresses someone thought of, when they thought of
    them. An address that is shared but unlisted is matched, silently, with no
    error and no log. It is evidence someone took the boundary seriously; it is not
    evidence the boundary covers your deployment. Treat an empty list as "nobody has
    told me about any gateways yet", never as "there are none".

    Returns True only for an address that is usable AND not known-shared. Every
    other case returns False = no IP match = today's behaviour.
    """
    if not isinstance(ip, str):
        return False           # None, or anything non-textual
    if not ip.strip():
        return False           # "", "   ", "\t\n" — never a shared bucket
    if shared_addresses:
        needle = _normalise_ip_for_exclusion(ip)
        for known in shared_addresses:
            if isinstance(known, str) and known.strip() and _normalise_ip_for_exclusion(known) == needle:
                return False   # known-shared gateway — fusion risk
    return True


def derive_thread_id_prefix_hash(
    prompt: str, model_tag: str, prefix_tokens: int | None = None
) -> str:
    """Auto-derive thread_id for clients that send no explicit thread_id.

    Prefix-token keying: hash only the first N tokens of the
    normalized prompt rather than the full prompt. This ensures that conversation
    extensions — same prefix, more tokens appended — produce the SAME thread_id,
    allowing the grace window to match and the KV cache restore to fire.

    ``prefix_tokens`` controls the cutoff (default 256). When None, the default
    is used. When supplied (e.g. via config), it overrides the default.

    ``prefix_tokens`` defaults to 256 which captures a typical system prompt +
    initial context. Adjust via QueueConfig.prefix_token_count /
    TURBOHAUL_PREFIX_TOKEN_COUNT if needed.

    Normalization is word-based (whitespace split) so incidental whitespace
    differences still map a semantically-identical prompt to one thread_id.
    """
    n = prefix_tokens if prefix_tokens is not None else 256
    normalized = " ".join(prompt.split())
    words = normalized.split()
    prefix = " ".join(words[:n]) if len(words) > n else normalized
    payload = (model_tag + "\0" + prefix).encode("utf-8")
    return "auto-" + hashlib.sha256(payload).hexdigest()[:24]
