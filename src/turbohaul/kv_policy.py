"""KV save/restore policy — ONE chokepoint for all save/restore decisions.

Unified identity-keyed save/restore decision logic.
Every save and restore call site routes through resolve_kv() so that:
  1. There is exactly ONE place that decides RESTORE vs SKIP vs SAVE.
  2. Every decision is logged with provenance (kills silent-skip blindness).
  3. The same logic serves all modes (single-series, series-parallel, double-parallel).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KVDecision:
    """The decision from resolve_kv(). Immutable, loggable, testable."""
    do_it: bool
    action: str  # 'restore' | 'save' | 'skip'
    reason: str  # human-readable why
    resolved_from: str  # which rule branch decided this

    def __repr__(self) -> str:
        return f"KVDecision(do_it={self.do_it}, action='{self.action}', reason='{self.reason}', from='{self.resolved_from}')"


def compute_ctx_len(messages) -> int:
    """Single source of truth for context size (char count of message content).

    BOTH the admission-time incoming size (chat_completion, before VRAM)
    AND the save-time saved size (manager._save_slot_kv) call THIS function, so the
    two numbers are guaranteed comparable for the extension-vs-compaction decision.
    Do NOT inline a second copy of this rule — drift here silently breaks restore.

    None-safe: tool-call turns legitimately send content=null. Non-str content
    (e.g. multimodal list blocks) is coerced via str() so this never raises.
    """
    total = 0
    for m in (messages or []):
        if isinstance(m, dict):
            c = m.get("content")
            total += len(c) if isinstance(c, str) else (0 if c is None else len(str(c)))
        else:
            total += len(str(m))
    return total


def _thread_hash(thread_id: str) -> str:
    """Safe filename component from thread_id."""
    if not thread_id:
        return "nothread"
    return hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]


def _prefix_hash_chain(context: list) -> list[str]:
    """Compute rolling hash chain for context turns.
    H_i = SHA256(H_{i-1} + \\x00 + role_i + \\x00 + content_i).
    Each turn's hash includes the previous hash, so the chain is order-sensitive.
    Empty/None context returns []. Monolithic-message clients get a 1-element list.

    Canonical form (unifies the two prior divergent copies —
    kv_policy's non-live copy and manager.py's live copy). Hashing uses
    manager's ROBUST form (\\x00 delimiter avoids ':'-in-content collisions;
    json.dumps(sort_keys=True) canonicalizes list/dict content deterministically
    so this is the form that produces today's on-disk saved metas — zero
    meta-format regression). Non-dict turns are coerced via str() so this never
    raises (kv_policy's input-safety, which manager's copy lacked).
    """
    if not context:
        return []
    chain = []
    prev = ""
    for turn in context:
        if isinstance(turn, dict):
            content = turn.get("content", "")
            role = turn.get("role", "")
        else:
            content = str(turn)
            role = ""
        if isinstance(content, (list, dict)):
            content = json.dumps(content, sort_keys=True, separators=(',', ':'))
        raw = f"{prev}\x00{role}\x00{content}"
        h = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
        chain.append(h)
        prev = h
    return chain


def _is_prefix_match(saved_chain: list[str], incoming_chain: list[str]) -> bool:
    """True when saved context is a strict-or-equal prefix of incoming.
    Handles the strict-extension case: saved=10 turns, incoming=12, first 10 match."""
    if not saved_chain:
        return False
    if len(saved_chain) > len(incoming_chain):
        return False
    return all(saved_chain[i] == incoming_chain[i] for i in range(len(saved_chain)))


def resolve_kv(
    action: str,
    identity: dict,
    sizes: dict,
) -> KVDecision:
    """Single chokepoint for ALL save/restore decisions.

    Args:
        action: 'save' or 'restore'
        identity: {
            'thread_id': str,        # caller identity (e.g. 'agent-ip-172.16.31.4')
            'model_tag': str,        # model name (e.g. 'my-model-27b')
            'slot_id': str | None,   # engine slot id (for per-slot keying)
            'port': int | None,      # sidecar port (for cross-model isolation)
        }
        sizes: {
            'saved_tokens': int | None,   # tokens in saved KV (None if unknown)
            'saved_len': int | None,      # char length of saved context
            'incoming_len': int,          # char length of incoming context
            'saved_chain': list[str],     # hash chain of saved context
            'incoming_chain': list[str],  # hash chain of incoming context
        }

    Returns:
        KVDecision with do_it, action, reason, resolved_from.
    """
    # --- Common guards ---
    thread_id = identity.get("thread_id", "")
    model_tag = identity.get("model_tag", "")

    if action == "save":
        return _resolve_save(thread_id, model_tag, sizes)
    elif action == "restore":
        return _resolve_restore(thread_id, model_tag, sizes)
    else:
        return KVDecision(False, action, f"unknown action '{action}'", "invalid-action")


def _resolve_save(thread_id: str, model_tag: str, sizes: dict) -> KVDecision:
    """Decide whether to save KV for this slot."""
    saved_tokens = sizes.get("saved_tokens") or 0

    # Don't save zero-token entries (degenerate)
    if saved_tokens <= 0:
        return KVDecision(False, "save", "saved_tokens=0 (degenerate)", "save-zero-tokens")

    # Don't save without identity (would collide on nothread)
    if not thread_id:
        return KVDecision(False, "save", "thread_id empty (no identity)", "save-no-identity")

    return KVDecision(True, "save", f"thread_id='{thread_id}' tokens={saved_tokens}", "save-ok")


def _resolve_restore(thread_id: str, model_tag: str, sizes: dict) -> KVDecision:
    """Decide whether to restore KV for this request.

    Prefix-validity classifier (replaces the earlier length compaction gate
    with prefix-VALIDITY vs the pinned clean bin).

    The engine reuses KV ONLY when the saved KV is a valid prefix of the incoming
    context (``saved ⊑ incoming`` → strict extension, stale ≤ 0). If saved diverges
    or is LONGER than incoming, the engine CLEARs + reprefills. So the restore
    decision is now the SAME prefix-validity the engine will enforce: restore iff
    the saved turn-hash chain is a valid prefix of the incoming turn-hash chain.

    Guards (fail-safes — KEPT verbatim, do NOT remove):
    - owner mismatch (saved_thread_id != thread_id) → reject
    - no saved data (saved_tokens<=0) → skip
    - no incoming identity (thread_id empty) → skip
    - inc_len==0 (admission size never threaded) → skip (never restore blindly)
    - empty incoming_chain (no admission chain to validate against) → skip

    Gate (REPLACES ``incoming_len < saved_len → discard``):
    - PHYSICS BELT: len(saved_chain) > len(incoming_chain) → FRESH (never restore a
      bin longer than incoming; the engine would CLEAR). ``_is_prefix_match`` already
      enforces this, but it is asserted explicitly per the physics law + design.
    - ``saved_chain ⊑ incoming_chain`` (valid prefix; covers EQUAL + EXTENSION) →
      RESTORE, ``resolved_from='restore-prefix-valid'``.
    - else (incoming diverges before saved_chain ends) → FRESH,
      ``resolved_from='restore-diverged-fresh'``.

    ``saved_len``/``incoming_len`` are retained ONLY for the inc_len==0 fail-safe +
    the human-readable reason; the length COMPARISON no longer gates the decision.
    """
    saved_tokens = sizes.get("saved_tokens") or 0
    saved_len = sizes.get("saved_len") or 0
    incoming_len = sizes.get("incoming_len", 0)
    saved_tid = sizes.get("saved_thread_id", "")
    saved_chain = sizes.get("saved_chain") or []
    incoming_chain = sizes.get("incoming_chain") or []

    # --- Owner validation: restore must validate against SAVED file's owner ---
    if saved_tid and thread_id and saved_tid != thread_id:
        return KVDecision(False, "restore",
                         f"owner mismatch (saved='{saved_tid}' inc='{thread_id}')",
                         "restore-owner-mismatch")

    # --- Guard: no saved data ---
    if saved_tokens <= 0:
        return KVDecision(False, "restore", "saved_tokens=0 (nothing to restore)", "restore-no-data")

    # --- Guard: no identity match ---
    if not thread_id:
        return KVDecision(False, "restore", "thread_id empty (no identity)", "restore-no-identity")

    # --- inc_len==0: no measurable incoming size ---
    # Hardening: with admission-time recording, a REAL
    # request always has incoming_len > 0. inc_len==0 means an empty request OR a
    # submit path that failed to thread admission_ctx_len (slot default). In NO
    # case can we confirm a valid prefix, so SKIP -- never restore blindly onto an
    # unknown prompt. Conservative: worst case is one fresh prefill. This makes a
    # wiring miss fail safe (skip) instead of masquerading as a fresh-cache restore.
    if incoming_len == 0:
        return KVDecision(False, "restore",
                         "inc_len=0 (no admission size recorded) -> cannot confirm prefix, skip",
                         "restore-no-incoming-size")

    # --- empty incoming_chain: no admission hash chain to validate against ---
    # The prefix-validity gate REQUIRES the incoming chain (threaded from the
    # admission site as slot.admission_hash_chain). If it is absent (a submit path
    # that did not thread it, or a monolithic client with no structured context),
    # we cannot confirm prefix-validity -> SKIP (same fail-safe posture as
    # inc_len==0; never restore blindly). Preserves the fail-safe guard.
    if not incoming_chain:
        return KVDecision(False, "restore",
                         "empty incoming_chain (no admission chain) -> cannot confirm prefix, skip",
                         "restore-no-incoming-chain")

    # --- PHYSICS BELT: never restore a bin LONGER than incoming ---
    # Proven in the engine logs: a saved KV longer than incoming diverges past the
    # incoming tail (stale > n_rs_seq) -> CLEAR + reprefill. _is_prefix_match below
    # already rejects this (returns False when saved is longer), but the physics law
    # is asserted as an EXPLICIT belt with its own provenance for observability.
    if len(saved_chain) > len(incoming_chain):
        return KVDecision(False, "restore",
                         f"PHYSICS-BELT (saved_turns={len(saved_chain)} > incoming_turns="
                         f"{len(incoming_chain)}) -> bin longer than incoming, would CLEAR; fresh",
                         "restore-physics-belt-saved-longer")

    # --- PREFIX-VALIDITY (the core; folds restore-equal + restore-extension) ---
    # saved_chain ⊑ incoming_chain => the saved KV is a valid prefix the engine will
    # reuse as a strict extension (stale ≤ 0); it prefills only the incoming tail.
    if _is_prefix_match(saved_chain, incoming_chain):
        return KVDecision(True, "restore",
                         f"PREFIX-VALID (saved_turns={len(saved_chain)} <= incoming_turns="
                         f"{len(incoming_chain)}, thread_id='{thread_id}')",
                         "restore-prefix-valid")

    # else: incoming diverges before saved_chain ends (NOT a prefix) — e.g. the
    # harness compressed/rewrote an early turn. Restoring would CLEAR; reprefill fresh.
    return KVDecision(False, "restore",
                     f"DIVERGED (saved_turns={len(saved_chain)} is NOT a prefix of incoming_turns="
                     f"{len(incoming_chain)}) -> reprefill fresh",
                     "restore-diverged-fresh")


def kv_save_fn(model_tag: str, sid: int, thread_hash: str, port: int = 0) -> str:
    """Globally-unique save filename: model + port + slot + thread hash."""
    return f"{model_tag}.p{port}.{thread_hash}.slot{sid}.bin"


def kv_meta_fn(model_tag: str, sid: int, thread_hash: str, port: int = 0) -> str:
    """Sidecar metadata filename."""
    return f"{model_tag}.p{port}.{thread_hash}.slot{sid}.json"
