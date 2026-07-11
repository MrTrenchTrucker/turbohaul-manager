"""The single KV request-classification chokepoint (Phase 1).

WHY THIS MODULE EXISTS
======================
Today, "how the manager classifies an incoming request for KV reuse" is scattered
across ~5 call sites / 3 mechanisms:

  1. event_type      -> ``manager.TurbohaulManager._classify_event`` (chain relations
                        -> guard-skip / sub-agent / compression / continuation /
                        user-message).
  2. identity        -> ``slot.derive_thread_id_prefix_hash`` + the two duplicated
                        admission blocks in ``api/chat_completion.py`` (IP + first-msg
                        fingerprint) + the DORMANT ``_shadow_recompose_identity``
                        (role+session; default-OFF via ``_m2b_active``).
  3. reuse (cold)    -> ``manager._restore_slot_kv`` delegating each candidate bin to
                        ``kv_policy.resolve_kv("restore", ...)`` + the ``if not restore:``
                        aggregate (fresh vs restore).
  4. labels          -> ``api/chat_completion._derive_client_meta_identity`` parses
                        {turn0_meta, role, session_id, is_compression, context_size}
                        into ``client_meta`` — but NOTHING downstream reads them today.

Phase 2 (LATER, gated) unifies those sites onto ONE ``classify_request`` chokepoint
and wires the parsed labels + per-role routes (curator -> reuse-main, save_ok=False,
role-keyed identity remap, ...).

PHASE 1 (THIS MODULE) — CHARACTERIZE-FIRST + INERT SCAFFOLD, ZERO BEHAVIOR CHANGE
================================================================================
Per the code-unification "safe live refactor" steps 1-2 (characterize today, then
stand up an empty-but-wired scaffold that returns the SAME values):

  * This module RE-EXPRESSES today's classification as PURE functions and returns
    TODAY's values. It is byte-for-byte equivalence-tested against the scattered
    logic (``tests/test_kv_classify_golden.py`` + ``tests/test_kv_classify_scaffold.py``).
  * It is IMPORTABLE + TESTED but effectively INERT: NOTHING in the running manager
    imports or calls it. The 5 scattered sites keep using their current logic. Zero
    behavior change.
  * The ``POLICIES`` registry has all 5 target classes, but each entry encodes ONLY
    today's behavior (``save_ok=True`` for every class — today nothing gates saves by
    role). The NEW routes (curator->reuse-main, save_ok=False, identity remap) are
    Phase 2 and are explicitly NOT encoded here.

INVARIANTS (Phase 1)
--------------------
  * Pure, None-safe, NO I/O. Imports only ``kv_policy`` (itself pure — hashlib, no
    engine/manager state) so ``_is_prefix_match`` / ``_prefix_hash_chain`` stay a
    single source of truth rather than a re-implemented copy.
  * Must NOT import or call anything that mutates manager/engine state.
  * ``resolve_kv`` (the reuse decision), the P1 warm-force gate, and the P2 residency
    fields are UNTOUCHED — ``classify_request`` will FEED ``resolve_kv`` in Phase 2,
    never edit it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Single source of truth for the prefix-validity + chain helpers (kv_policy is PURE
# and BYTE-UNTOUCHED — importing it is exactly the anti-duplication this refactor is
# about; do NOT re-implement _is_prefix_match here).
from turbohaul.kv_policy import _is_prefix_match, _prefix_hash_chain  # noqa: F401  (re-export for tests)

# ============================================================================
# Canonical vocabularies (validated at import — fail fast on boot)
# ============================================================================

# The event taxonomy — the exact strings ``_classify_event`` returns today.
EVENT_GUARD_SKIP = "guard-skip"
EVENT_SUB_AGENT = "sub-agent"
EVENT_COMPRESSION = "compression"
EVENT_CONTINUATION = "continuation"
EVENT_USER_MESSAGE = "user-message"
EVENTS = frozenset({
    EVENT_GUARD_SKIP,
    EVENT_SUB_AGENT,
    EVENT_COMPRESSION,
    EVENT_CONTINUATION,
    EVENT_USER_MESSAGE,
})

# reuse_intent — today's uniform outcome vocabulary (mirrors resolve_kv do_it:
# True -> "restore", False -> "fresh"). Phase 2 adds "reuse-main" for curator.
REUSE_FRESH = "fresh"
REUSE_RESTORE = "restore"
REUSE_INTENTS = frozenset({REUSE_FRESH, REUSE_RESTORE})

# The 5 Phase-2 TARGET classes. They exist in the registry NOW (structure) so Phase 2
# only has to change the entry VALUES, never the shape. "main" folds today's
# continuation/guard-skip (the primary continuing thread). "curator" is a Phase-2
# label-only class — UNREACHABLE with today's labels-absent traffic (documented).
CLASS_MAIN = "main"
CLASS_USER_MESSAGE = "user-message"
CLASS_SUB_AGENT = "sub-agent"
CLASS_CURATOR = "curator"
CLASS_COMPRESSION = "compression"
CLASSES = frozenset({
    CLASS_MAIN,
    CLASS_USER_MESSAGE,
    CLASS_SUB_AGENT,
    CLASS_CURATOR,
    CLASS_COMPRESSION,
})

# event_type -> class map (labels-absent inference). curator is NOT reachable from an
# event_type; it only arrives via an explicit role label (Phase 2 semantics).
EVENT_TO_CLASS = {
    EVENT_CONTINUATION: CLASS_MAIN,
    EVENT_GUARD_SKIP: CLASS_MAIN,
    EVENT_USER_MESSAGE: CLASS_USER_MESSAGE,
    EVENT_SUB_AGENT: CLASS_SUB_AGENT,
    EVENT_COMPRESSION: CLASS_COMPRESSION,
}


# ============================================================================
# Registry (config-as-data) — declarative, greppable, one entry per class
# ============================================================================

@dataclass(frozen=True)
class RoutePolicy:
    """Per-class KV policy. Phase 1 encodes ONLY today's behavior.

    ``save_ok`` is ``True`` for EVERY class EXCEPT curator: nothing in the manager
    gates a non-curator save by role (a save is gated only by ``resolve_kv`` on
    token-count/identity, not by class). ``note`` documents that class's reuse
    behavior.

    Curator keeps ``save_ok=False`` so its context-specific turn is never persisted
    into the shared bin it rides on. Curator reuse itself comes from the calling
    harness thread_id (it sends the curator's thread_id = main's), NOT a manager-side
    identity remap — so there is no ``reuse_target`` here.
    """

    save_ok: bool
    note: str


# Declarative POLICIES registry — all 5 classes side-by-side (add a class = one entry).
POLICIES: dict[str, RoutePolicy] = {
    CLASS_MAIN: RoutePolicy(
        save_ok=True,
        note="today: primary continuation/guard-skip thread; native in-RAM or "
             "cold clean-restore reuse when the saved chain is a valid prefix.",
    ),
    CLASS_USER_MESSAGE: RoutePolicy(
        save_ok=True,
        note="today: think-strip follow-up (clean valid, warm diverges); WARM force "
             "is default-OFF -> native reuse; cold restores the valid clean prefix.",
    ),
    CLASS_SUB_AGENT: RoutePolicy(
        save_ok=True,
        note="today: no clean anchor for this identity -> fresh until it saves its "
             "own anchor; never cross-restored.",
    ),
    CLASS_CURATOR: RoutePolicy(
        save_ok=False,
        note="Phase 2: rides the MAIN bin via the calling harness thread_id "
             "(curator thread_id = main's = main-<session_id>); save_ok=False so "
             "the curator's context-specific tail never overwrites main's anchor. The "
             "save-gate is flag-gated (TURBOHAUL_CURATOR_REUSE_MAIN) — inert until on.",
    ),
    CLASS_COMPRESSION: RoutePolicy(
        save_ok=True,
        note="today: an early turn was rewritten/summarized (clean not a prefix of "
             "incoming) -> reprefill fresh.",
    ),
}


# ============================================================================
# Import-time completeness asserts — fail fast on boot if the registry drifts
# ============================================================================
assert set(POLICIES) == set(CLASSES), (
    f"POLICIES keys {sorted(POLICIES)} != CLASSES {sorted(CLASSES)} — every class "
    "must have exactly one registry entry (single source of truth)."
)
assert set(EVENT_TO_CLASS) == set(EVENTS), (
    f"EVENT_TO_CLASS keys {sorted(EVENT_TO_CLASS)} != EVENTS {sorted(EVENTS)} — every "
    "event_type must map to a class."
)
assert set(EVENT_TO_CLASS.values()) <= set(CLASSES), (
    "EVENT_TO_CLASS maps to a class not in the registry."
)


# ============================================================================
# The output record
# ============================================================================

@dataclass(frozen=True)
class RequestClass:
    """The unified classification of one request. Immutable + loggable + testable.

    Phase 1 fields all carry TODAY's values:
      * identity     — the resolved thread_id key (today's labels-absent value = the
                       base thread_id; the role-keyed remap is Phase 2).
      * event_type   — one of EVENTS (byte-identical to ``_classify_event``).
      * reuse_intent — one of REUSE_INTENTS (mirrors ``resolve_kv`` do_it).
      * resolved_from— provenance: ``"POLICIES[<class>]"`` when an explicit role label
                       selected the class, else ``"chain-inference"``.
      * save_ok      — today: True for every class (no role gates saves yet).
    """

    identity: str | None
    event_type: str
    reuse_intent: str
    resolved_from: str
    save_ok: bool


# ============================================================================
# Signals — the pure inputs classify_request reads (no request/socket/FS objects)
# ============================================================================

@dataclass(frozen=True)
class RequestSignals:
    """Pure, already-extracted inputs for classification.

    The manager/admission code computes these today; Phase 1 keeps that extraction at
    the call site (it reads ``request.client.host``, the payload, config, and the
    on-disk clean-bin anchor — all runtime/IO concerns). ``classify_request`` receives
    the DERIVED signals and stays pure.

      * thread_id    — the base thread_id already derived at admission (payload
                       thread_id, else IP + first-msg fingerprint). None/"" allowed.
      * inc_chain    — incoming turn-hash chain (``slot.admission_hash_chain``).
      * saved_chain  — the pinned clean-anchor chain for this identity, or None when
                       there is no anchor (first-seen / sub-agent).
      * warm_chain   — the engine warm-state chain (``_engine_view_chain``), or None/[]
                       when unknown (noop / streamed / tool-call).
      * size         — admission context size (``slot.admission_ctx_len`` /
                       ``context_size``); the inc_len==0 fail-safe input.
    """

    thread_id: str | None = None
    inc_chain: list[str] = field(default_factory=list)
    saved_chain: list[str] | None = None
    warm_chain: list[str] | None = None
    size: int = 0


# ============================================================================
# Pure re-expressions of today's scattered logic
# ============================================================================

def classify_event(saved_chain, inc_chain, warm_chain) -> str:
    """Byte-for-byte re-expression of ``manager.TurbohaulManager._classify_event``.

    - guard-skip : no incoming chain (can't classify).
    - sub-agent  : no clean anchor for this identity (``saved_chain is None``).
    - compression: clean anchor exists but is NOT a prefix of incoming.
    - continuation: clean ⊑ incoming AND warm is an equal-or-longer valid prefix.
    - user-message: clean ⊑ incoming but the warm state diverges.

    None-safe. Golden-tested to MATCH the manager method across the full matrix.
    """
    if not inc_chain:
        return EVENT_GUARD_SKIP
    if saved_chain is None:
        return EVENT_SUB_AGENT
    if not _is_prefix_match(saved_chain, inc_chain):
        return EVENT_COMPRESSION
    warm_covers = (
        bool(warm_chain)
        and _is_prefix_match(warm_chain, inc_chain)
        and len(warm_chain) >= len(saved_chain)
    )
    return EVENT_CONTINUATION if warm_covers else EVENT_USER_MESSAGE


def infer_reuse_intent(saved_chain, inc_chain, size) -> str:
    """Re-expression of today's cold reuse outcome (``resolve_kv`` do_it -> fresh/restore).

    Mirrors ``kv_policy.resolve_kv("restore", ...)`` for the owner-matched,
    positive-token baseline: restore IFF the saved chain is a valid prefix of the
    incoming chain (the engine strict-extends). The ``size <= 0`` and empty-``inc_chain``
    branches reproduce resolve_kv's inc_len==0 / empty-incoming-chain fail-safes
    (never restore blindly). ``saved_chain is None`` (no anchor) -> fresh (sub-agent).
    """
    if saved_chain is None or not inc_chain or (size or 0) <= 0:
        return REUSE_FRESH
    return REUSE_RESTORE if _is_prefix_match(saved_chain, inc_chain) else REUSE_FRESH


def recompose_identity(base_thread_id, role, session_id) -> str:
    """Pure re-expression of ``chat_completion._shadow_recompose_identity``.

    CHARACTERIZATION ONLY (Phase 1): documents today's DORMANT, append-only role+session
    identity-shadow so Phase 2 has a single source for the remap rule. ``classify_request``
    does NOT apply this today — the shadow never drives thread_id (``_m2b_active`` is
    default-OFF), so today's identity is the base thread_id. With ``role``/``session_id``
    both falsy (labels absent) this returns ``base_thread_id`` unchanged (identity).
    """
    base = base_thread_id or ""
    parts = []
    if session_id:
        parts.append("s=" + hashlib.sha256(str(session_id).encode()).hexdigest()[:12])
    if role:
        parts.append("r=" + hashlib.sha256(str(role).encode()).hexdigest()[:8])
    return base + "".join("-" + p for p in parts)


def _class_from_label(labels) -> str | None:
    """Explicit label -> class. Consumes the caller-supplied is_* BOOLEANS from
    client_meta (is_curator/is_compression/is_sub_agent/is_main). Resolved by
    PRIORITY — the labels are NOT mutually exclusive (an upstream caller can emit a
    curator that carries BOTH is_sub_agent=True AND is_curator=True). Priority order:
    is_curator > is_compression > is_sub_agent > is_main. is_curator MUST be checked
    before is_sub_agent (so a double-labelled curator resolves to CURATOR, never
    sub_agent) AND before is_compression. Back-compat: falls back to a literal role
    string that names a class. Labels absent -> None -> chain-inference.

    Only provenance + class selection here — whether the curator route is APPLIED is
    the admission site's flag-gated choice (E5/E6). None-safe.
    """
    if not labels:
        return None
    if labels.get("is_curator"):
        return CLASS_CURATOR
    if labels.get("is_compression"):
        return CLASS_COMPRESSION
    if labels.get("is_sub_agent"):
        return CLASS_SUB_AGENT
    if labels.get("is_main"):
        return CLASS_MAIN
    role = labels.get("role")
    if role in POLICIES:
        return role
    return None


# ============================================================================
# THE chokepoint (Phase 1: reproduces today; INERT — nothing in the manager calls it)
# ============================================================================

def classify_request(labels: dict | None, signals: RequestSignals) -> RequestClass:
    """Single classification chokepoint. Phase 1 REPRODUCES today's classification.

    Pure + None-safe + no I/O. Derives event_type/identity/reuse_intent exactly the way
    the current scattered code does, looks up the (today-behavior) ``POLICIES`` entry
    for the resolved class, and records provenance.

    Phase 2 migrates the scattered call sites onto this function; POLICIES[curator]
    carries save_ok=False (the manager save-gate reads it). Identity is NOT remapped
    here — curator reuse-main rides the calling harness thread_id, not a manager remap.
    """
    inc_chain = signals.inc_chain or []
    saved_chain = signals.saved_chain  # None => no anchor (sub-agent)
    warm_chain = signals.warm_chain
    size = signals.size or 0

    event_type = classify_event(saved_chain, inc_chain, warm_chain)

    # Class resolution: an explicit role label selects the class (provenance
    # POLICIES[<class>]); otherwise infer from the event_type (chain-inference).
    label_cls = _class_from_label(labels)
    if label_cls is not None:
        cls = label_cls
        resolved_from = f"POLICIES[{cls}]"
    else:
        cls = EVENT_TO_CLASS[event_type]
        resolved_from = "chain-inference"

    policy = POLICIES[cls]  # KeyError impossible — import-time completeness assert.

    # Identity — always the base thread_id. Curator reuse-main is achieved by the
    # calling harness sending the curator thread_id = main's thread_id (main-
    # <session_id>) so resolve_kv owner-matches main's bin naturally; the manager does
    # NOT remap identity (a manager-side remap would OVERRIDE + break the harness id).
    identity = signals.thread_id

    reuse_intent = infer_reuse_intent(saved_chain, inc_chain, size)

    # save_ok — today: uniformly True (no class gates a save). Sourced from the
    # registry so Phase 2 flips it in one place.
    save_ok = policy.save_ok

    return RequestClass(
        identity=identity,
        event_type=event_type,
        reuse_intent=reuse_intent,
        resolved_from=resolved_from,
        save_ok=save_ok,
    )


def explain(labels: dict | None, signals: RequestSignals) -> dict:
    """Observability helper: the RequestClass plus the resolved class + policy note.

    One place for a structured log / test-double at the chokepoint (the payoff of a
    single funnel). Pure; safe to call anywhere.
    """
    rc = classify_request(labels, signals)
    label_cls = _class_from_label(labels)
    cls = label_cls if label_cls is not None else EVENT_TO_CLASS[rc.event_type]
    return {
        "class": cls,
        "identity": rc.identity,
        "event_type": rc.event_type,
        "reuse_intent": rc.reuse_intent,
        "resolved_from": rc.resolved_from,
        "save_ok": rc.save_ok,
        "policy_note": POLICIES[cls].note,
    }
