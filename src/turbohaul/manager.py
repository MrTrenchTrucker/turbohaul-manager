"""TurbohaulManager: top-level orchestrator wiring queue + subprocess + state + timers.

Per ARCHITECTURE.md - orchestrates the whole lifecycle described in the state
machine. The foundational interface plus the full worker_loop streaming
implementation work alongside the API layer that forwards to llama-server.
"""
import asyncio
import enum
import contextlib
import hashlib
import httpx
import json
import logging
import os
import re
import signal
import time
from types import SimpleNamespace
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from turbohaul.kv_policy import (
    resolve_kv, kv_save_fn, kv_meta_fn, compute_ctx_len,
    _prefix_hash_chain, _is_prefix_match,
)
from typing import Any

from turbohaul.config import KEEP_ALIVE_MAX_S, BootConfig, RuntimeConfig
from turbohaul.fsm import LEGAL_TRANSITIONS, InvalidTransition, is_terminal, transition
from turbohaul.live_monitor import LiveOutputBuffer, idle_generation
from turbohaul.manifest import flags_to_argv, read_manifest
from turbohaul.queue import GraceTimer, IdleHotTimer, TurbohaulQueue
from turbohaul.safety import (
    all_safety_gates,
    estimate_kv_cache_mib,
    PER_SLOT_COMPUTE_FLOOR_MIB,
    _vram_budget,
)
from turbohaul.singleton import (
    boot_orphan_reaper,
    detect_foreign_gpu_apps,
    intra_lifetime_orphan_scan,
)
from turbohaul.slot import Slot, SlotEvictedError, SlotState, derive_thread_id_prefix_hash
from turbohaul import load_verify_log  # observability emitter + /status read (display-only)
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
from turbohaul.mlx_spawn import mlx_spawn, mlx_flags_to_argv
from turbohaul.telemetry import FlapTelemetry, init_telemetry


log = logging.getLogger(__name__)


# === WIN 4 KV build/model/ctx FINGERPRINT + mismatched-bin purge ===
# A saved KV .bin restored into a DIFFERENT engine build / model / ctx-config is
# GARBAGE KV = a silent wrong answer. Every sidecar is stamped at SAVE with the
# loaded engine's fingerprint (gguf sha + binary sha + n_ctx + recurrent seq
# width); a file-level sweep then deletes any bin whose stamp != the current
# engine's. STRICTLY save/sidecar/sweeper-side — it NEVER touches resolve_kv / the
# restore-decision gate (/ M4, separately gated + off-limits).
FINGERPRINT_PURGE_MIN_INTERVAL_S = 300  # sweeper throttle floor; startup call is unthrottled


def _fingerprint_purge_enabled() -> bool:
    """True IFF TURBOHAUL_FINGERPRINT_PURGE is set truthy. Default OFF = zero cost
    + full back-compat (bins are only ever deleted when an operator opts in)."""
    return os.environ.get("TURBOHAUL_FINGERPRINT_PURGE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# === (DURABLE MANAGER B): per-(role,session) KV ring + reload-before-serve ===
# P1 (INERT observe-only): ring store on _save_slot_kv + Resident.resident_state_tag + LOG would-be
# reload decisions at serve sites (@3872 streaming, @3920 non-streaming). ZERO behavior change.
# Flag: TURBOHAUL_DURABLE_RING (default OFF). kv_policy.py BYTE-LOCKED (no edits).


def _durable_ring_enabled() -> bool:
    """True IFF TURBOHAUL_DURABLE_RING is set truthy. Default OFF = P1 INERT (observability only).
    When ON (1/true/yes/on): enables ring writes, residency tag, and would-be-reload logging.
    When OFF (default): zero behavior change — all DURABLE_RING code paths are best-effort no-ops."""
    return os.environ.get("TURBOHAUL_DURABLE_RING", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _durable_ring_key(client_meta: dict | None) -> tuple[str, str] | None:
    """Centralized ring-key derivation.
    Returns (role, session_id) or None if missing. Mirrors save/restore client_meta usage.
    
    Derives role from the harness's is_* BOOLEANS via _class_from_label (priority:
    is_curator > is_compression > is_sub_agent > is_main). Falls back to literal
    'role' string if present, then 'main'. This ensures curator/sub/main are
    DISTINCT keys even when the harness doesn't send a top-level 'role' string."""
    if not client_meta:
        return None
    from turbohaul.kv_classify import _class_from_label
    role = _class_from_label(client_meta) or client_meta.get("role") or "main"
    session_id = client_meta.get("session_id")
    if not session_id:
        return None
    return (role, str(session_id))


def _bin_role(client_meta: dict | None) -> str | None:
    """SPEC-V2 (RED-HAT design note): trusted role for bin keying. Label-derived first
    (is_curator > is_compression > is_sub_agent > is_main via _class_from_label).
    A literal client_meta['role'] string is honored for NON-main roles only —
    'main' is reachable ONLY via an explicit is_main label, NEVER by default or
    by a bare literal. Returns None when the request is unlabeled, which means
    RAW thread_id keying (today's behavior) — an unlabeled variant can therefore
    never collide with, restore, or reset the main session bin."""
    if not client_meta:
        return None
    from turbohaul.kv_classify import _class_from_label
    role = _class_from_label(client_meta)
    if role is not None:
        return str(role)
    r = client_meta.get("role")
    if r and str(r) != "main":
        return str(r)
    return None


def _bin_identity(thread_id: str, client_meta: dict | None, chain: list | None = None) -> str:
    """SPEC-V2 bin key: ONE KV copy per (session_id, role). Fed to
    Manager._thread_hash for filenames AND stamped as meta 'thread_id', so the
    byte-locked kv_policy.resolve_kv owner string-compare keys on (session,
    role) with ZERO kv_policy/kv_classify edits.

    - session_id absent OR role unlabeled -> raw thread_id (today, byte-for-byte).
    - sub-agent / curator roles carry an 8-hex conversation fingerprint from the
      first TWO admission-chain entries (system + first user turn — chain[0]
      alone can be a sibling-shared system prompt), so concurrent same-role
      siblings in one session get DISTINCT bins (RED-HAT design note — no
      last-longest-wins starvation, no sibling reset ping-pong). chain
      unavailable -> raw thread_id (safe per-thread bin).
    """
    cm = client_meta or {}
    session_id = cm.get("session_id")
    if not session_id:
        return thread_id or ""
    role = _bin_role(cm)
    if role is None:
        return thread_id or ""
    from turbohaul.kv_classify import CLASS_SUB_AGENT, CLASS_CURATOR
    if role in (CLASS_SUB_AGENT, CLASS_CURATOR):
        if not chain:
            return thread_id or ""
        fp = hashlib.sha256("|".join(str(h) for h in chain[:2]).encode()).hexdigest()[:8]
        return f"sess:{session_id}:{role}:{fp}"
    return f"sess:{session_id}:{role}"


def _select_ring_bin(ring: list, inc_chain: list, model_tag: str):
    """Select the first ring entry (newest-first) that matches ALL criteria:
    (a) same model_tag,
    (b) NON-EMPTY hash_chain (guard against [] -> trivially prefix of anything but yields 0 common-prefix),
    (c) valid PREFIX of inc_chain (reuses _is_prefix_match, the SAME check default path uses).
    
    Returns the matching ring_entry or None. Prefers clean_prefix=True when multiple qualify.
    """
    from turbohaul.kv_classify import _is_prefix_match
    
    clean_matches = []
    other_matches = []
    
    for entry in ring:
        # (a) same model_tag
        if entry.get("model_tag") != model_tag:
            continue
        # (b) NON-EMPTY hash_chain
        chain = entry.get("hash_chain", [])
        if not chain:
            continue
        # (c) valid PREFIX of inc_chain (same check as default path)
        if _is_prefix_match(chain, inc_chain):
            if entry.get("clean_prefix", False):
                clean_matches.append(entry)
            else:
                other_matches.append(entry)
    
    # Prefer clean_prefix=True when multiple qualify (don't let polluted bin clobber clean default)
    if clean_matches:
        return clean_matches[0]  # newest clean (ring is newest-first)
    if other_matches:
        return other_matches[0]  # newest non-clean
    return None


def _shadow_reprefill_enabled() -> bool:
    """True IFF TURBOHAUL_SHADOW_REPREFILL is set truthy. Default OFF so
    Option A (SAVE-side shadow-reprefill) ships completely INERT: no shadow bin is
    written and zero extra work runs until PL opts in for a smoke. This gate is
    SAVE-side only — the restore gate is unchanged and never consumes the
    shadow bin (that restore-preference is a SEPARATE, PL-owned step)."""
    return os.environ.get("TURBOHAUL_SHADOW_REPREFILL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _shadow_restore_prefer_enabled() -> bool:
    """True IFF TURBOHAUL_SHADOW_RESTORE_PREFER is set truthy. Default OFF so
    step (d) (the shadow RESTORE-preference) ships completely INERT: the restore path
    is byte-identical to today (the clean anchor is chosen) until PL opts in for a
    smoke. This gate is RESTORE-side only and mirrors the SAVE-side
    _shadow_reprefill_enabled; the two are independent (no shadow bin exists to
    prefer unless the SAVE gate wrote one, so with SAVE off this reader is a no-op).

    When ON it ONLY adds a preference for a PROVEN-PREFIX `.shadow` bin over the clean
    anchor at the one warm forced-restore seam (_maybe_force_clean_restore). It does
    NOT weaken any prefix-validity/owner check: a shadow bin is eligible only if it
    passes the SAME _is_prefix_match bar the clean path uses, and even a wrongly-
    preferred bin only ever costs a reprefill via the engine's get_common_prefix
    backstop — never a wrong answer (see _maybe_force_clean_restore)."""
    return os.environ.get("TURBOHAUL_SHADOW_RESTORE_PREFER", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _shadow_cold_restore_enabled() -> bool:
    """critical item (the cold-path preference) — True IFF the COLD-path (wave-return / swap-back)
    shadow restore-preference is active.

    DISTINCT gate from the WARM ``_shadow_restore_prefer_enabled``
    (TURBOHAUL_SHADOW_RESTORE_PREFER, held 0 on purpose): at the WARM grace-follow-up
    seam the engine already holds a warm KV, and native warm reuse (an equal-or-longer
    valid prefix) should win — so preferring a shadow there is disabled. The COLD
    swap-back has NO warm KV at all (the sidecar was just re-spawned), so the
    byte-matching think-free ``.shadow`` bin (turns 1..N think-free) is the CORRECT
    cold restore target: restoring it lets the next decode STRICT-EXTEND the think-free
    state (reprefill only the appended sub-result tail) instead of a fresh full
    prefill. Because the two seams want opposite defaults, they MUST be independent
    gates (do not fold the cold path under the warm flag).

    Default: ON whenever the SAVE-side ``TURBOHAUL_SHADOW_REPREFILL`` is enabled — a
    shadow bin only exists to prefer when the save gate wrote one, so the cold
    preference should activate exactly when (and only when) shadows are being written.
    An explicit ``TURBOHAUL_SHADOW_COLD_RESTORE`` overrides the default in EITHER
    direction (``1/true/yes/on`` force-ON; ``0/false/no/off`` force-OFF — e.g. to A/B
    the SAVE side without the cold restore). With SHADOW_REPREFILL off (default) this
    ships INERT: no shadow bins exist, so the cold path is byte-identical to today."""
    v = os.environ.get("TURBOHAUL_SHADOW_COLD_RESTORE", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return _shadow_reprefill_enabled()


def _tooltail_restore_skip_enabled() -> bool:
    """crit3 — True IFF the TOOL-tail restore-skip guard is active.

    DEFAULT OFF (unset / 0/false/no/off): the forced/cold restore fires normally even
    when the divergent tail (turns beyond the common prefix) contains a hash-INVISIBLE
    tool turn. ON (1/true/yes/on): when that tail is tool-opaque, SKIP the restore and
    safe-degrade to the engine's native get_common_prefix checkpoint reuse.

    Why the default is now OFF: the guard was ON to prevent a token-STALE POST -> engine
    CLEAR on a tool tail. That failure mode is now closed upstream by Fix B
    (`_covered_scaffold_strip_enabled`, DEFAULT ON): the SAVE probes prefill the
    think-stripped HISTORICAL-form prompt, so the saved clean/shadow bin now BYTE-MATCHES
    the harness's future think-stripped resend across the covered tool region. With the
    tokens matching, the FALSE-POSITIVE chain "match" no longer POSTs a stale bin -> the
    engine can no longer hit stale > n_rs_seq and CLEAR. So this skip is now REDUNDANT
    (Fix B already prevents the CLEAR) AND HARMFUL: it force-degrades to a full
    reprocess instead of the cheap clean-restore reuse Fix B enables. PL verified live
    with env=0 that the restore reuses at pe~265 (no CLEAR). Flag-on (`1/true/yes/on`)
    is RETAINED as an emergency A/B-rollback floor if a covered tool region is ever
    found still drifting despite Fix B.

    ⚠ Flag-on is a BLUNT containment lever, NOT a clean-correctness path: it reverts to
    the pre-crit3 behavior, which scans `inc_messages[common:]` — for a validated strict-
    prefix clean bin `common == len(clean_chain)`, so that span is the FRESH APPENDED tail.
    A tool-heavy fresh tail therefore still OVER-SKIPS (forces a full reprocess, forfeiting
    the cheap clean-restore) — that over-skip is the exact residual this default-flip fixes.
    Enable it ONLY to contain a suspected Fix-B regression on a genuinely-drifting COVERED
    tool region, accepting the fresh-tail over-skip as the cost. Coupling invariant (NOT
    code-enforced, operator-owned): default-OFF here is safe BECAUSE `_covered_scaffold_strip
    _enabled` (Fix B) is DEFAULT ON; if Fix B is ever disabled, explicitly set this guard ON.

    Mechanism (unchanged): `_prefix_hash_chain` (kv_policy, UNTOUCHED) hashes role+content
    ONLY, so an assistant turn with truthy tool_calls (content null) and a role=="tool"
    result are invisible to the chain. When ON and the harness re-serializes that tool
    region nondeterministically, the guard treats the tail-tool chain "match" as
    untrustworthy and skips the restore rather than POST a possibly token-STALE bin."""
    v = os.environ.get("TURBOHAUL_TOOLTAIL_RESTORE_SKIP", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _warm_force_clean_restore_enabled() -> bool:
    """severity item P1 — True IFF the WARM-path forced clean-bin restore is active.

    DEFAULT ON (2026-07-08 single-copy flip; was OFF): when OFF, the warm grace follow-up decodes on the engine's native in-RAM
    get_common_prefix reuse (clean v0.5.8 parity) — the forced DISK restore
    (_maybe_force_clean_restore, wired at the 2 warm sites) is SKIPPED. This restores
    FULL in-RAM KV reuse for follow-up tool calls + same-model continuations: warm_covers
    was structurally ~always False (warm_chain is WITH-<think>; the harness resends
    think-STRIPPED), so the force fired every warm follow-up -> byte-mismatch -> engine
    CLEAR -> full reprefill (~16k of ~50k). ON (1/true/yes/on): re-enable the forced
    restore (emergency A/B rollback). Does NOT affect the COLD swap-back / wave-return
    (_restore_slot_kv), which legitimately needs the disk restore and is UNTOUCHED."""
    # SINGLE-COPY (2026-07-08): DEFAULT ON. The clean bin is now a TRUE
    # byte-prefix of the harness's think-free resend (add_generation_prompt=False +
    # trailing-primer strip in _render_strip_prefill_probe), so forcing its restore
    # on the warm follow-up strict-extends (stale<=1 < n_rs_seq=2, no CLEAR) instead
    # of letting native in-VRAM reuse re-diverge at the assistant-turn boundary.
    # A/B-proven live 2026-07-08 (sentinel run): forced_clean_restores 0->5, zero
    # 38973-pin, zero forcing-full, reprefills 27k -> delta-only. Safety coupling:
    # force-restore is HARD-OFF whenever the covered-scaffold strip is disabled
    # (unstripped saves must never be force-restored). Ops kill-switches (no
    # container recreate needed):
    # hot: touch /var/lib/turbohaul/.warm_force_clean_off
    # env: TURBOHAUL_WARM_FORCE_CLEAN_RESTORE=0
    if os.path.exists("/var/lib/turbohaul/.warm_force_clean_off"):
        return False
    if not _covered_scaffold_strip_enabled():
        return False
    v = os.environ.get("TURBOHAUL_WARM_FORCE_CLEAN_RESTORE", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _ram_reanchor_enabled() -> bool:
    """Option C — per-turn RAM save+restore re-anchor. DEFAULT ON when
    SLOT_SAVE_DIR is tmpfs. Kill-switches (no recreate):
      hot: touch /var/lib/turbohaul/.ram_reanchor_off
      env: TURBOHAUL_RAM_REANCHOR=0
    """
    if os.path.exists("/var/lib/turbohaul/.ram_reanchor_off"):
        return False
    v = os.environ.get("TURBOHAUL_RAM_REANCHOR", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _warm_natural_skip_enabled() -> bool:
    """a later phase — gate the warm-anchor-natural-skip. DEFAULT OFF.

    The skip serves the engine's NATURAL slot state (which, for a thinking model,
    carries the generated ``<think>...</think>`` tokens) instead of restoring the
    think-STRIPPED clean bin. When the harness resends assistant turns WITHOUT
    reasoning (the Hermes case — proven: incoming assistant turns have no
    reasoning_content), the think-full natural state diverges from the think-less
    resend at the first generated think block -> get_common_prefix truncates to the
    last-user boundary -> the whole since-last-user span reprefills every turn.
    Restoring the stripped clean bin (crash-free since a later phase) matches the resend
    byte-for-byte -> delta-only. So default OFF. Set ON only for a harness that
    PRESERVES reasoning_content on resent turns (natural == resend), where the
    skip validly avoids a redundant restore."""
    return os.environ.get("TURBOHAUL_WARM_NATURAL_SKIP", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _curator_reuse_main_active() -> bool:
    """True IFF the curator reuse-main route is active.

    DEFAULT OFF: a curator turn is classified but NOT remapped — it keeps its own
    identity + saves normally, byte-identical to today. ON (1/true/yes/on): a
    labelled curator RESTORES the shared main bin (identity hermes-main-<sid>, via
    the admission remap) and its slot save is SKIPPED (save_ok=False, this file's
    save-gate) so it never overwrites main's anchor. resolve_kv / kv_policy are
    byte-identical either way — only the thread_id fed IN and whether a save fires
    change. Flag- AND label-gated: with the flag off OR the labels absent it is
    fully inert."""
    return os.environ.get("TURBOHAUL_CURATOR_REUSE_MAIN", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# crit2/crit3 (Fix B — SAVE-ONLY reasoning-scaffold strip). Env name is
# CENTRALIZED here so a rename is a one-line change (flag name pending confirmation).
_COVERED_SCAFFOLD_STRIP_ENV = "TURBOHAUL_COVERED_SCAFFOLD_STRIP"


def _covered_scaffold_strip_enabled() -> bool:
    """crit2/crit3 (Fix B) — True IFF the SAVE-probe reasoning-scaffold strip
    is active.

    DEFAULT ON (unset / 1/true/yes/on): the two SAVE probes prefill the HISTORICAL-form
    prompt (engine ``/apply-template`` render -> strip ``<think>...</think>\\n\\n`` -> engine
    ``/completion`` prefill n_predict=0) so the saved KV byte-matches the harness's FUTURE
    think-stripped resend of the covered turns -> the cold/forced restore MATCHES+REUSES
    instead of hitting a position-drifted prefix and CLEARing.

    OFF (0/false/no/off): pre-Fix-B behavior — the plain messages
    ``/v1/chat/completions`` n_predict=0 probe, byte-identical to today.

    SAVE-side ONLY: the two live-generation sites (_build_stream_payload / _complete) are
    UNTOUCHED (they already render covered turns historical via the engine's position
    logic). kv_policy is UNTOUCHED: the strip acts on the rendered PROMPT, never on the
    role/content inputs to _prefix_hash_chain, so the saved meta's hash_chain is identical
    with the flag on or off (the strip is hash-chain-invariant)."""
    v = os.environ.get(_COVERED_SCAFFOLD_STRIP_ENV, "").strip().lower()
    return v not in ("0", "false", "no", "off")


def _role_save_enabled(role, client_meta=None) -> bool:
    """per-role KV-save toggle. HARNESS-LEVEL ONLY — the sole
    source is the FRONT-END/backend setting delivered on the request as
    ``client_meta["save_kv"]`` (bool). design rule: NO environment variables. The harness wires
    the Hermes per-role UI toggle to this field; if present it is authoritative for THIS
    request's role. DEFAULT (field absent) = False = DISPOSABLE ROLE IS NOT SAVED — the operator's
    default (main always saves; curator/sub-agent/compression off unless the front-end turns
    that role on, e.g. a big-VRAM box that wants to keep everything). Called ONLY for disposable
    roles (the gate is `_disp_role is not None`); main never reaches here, so main ALWAYS saves."""
    if isinstance(client_meta, dict) and isinstance(client_meta.get("save_kv"), bool):
        return client_meta["save_kv"]
    return str(role) == "main"


def _seam_flush_allowed(thread_id: str, client_meta: dict | None) -> tuple[bool, str | None]:
    """D2 SINGLE resolve point for 'may an unload seam persist this
    identity?' (code-unification doctrine — no scattered role-ifs). Labels first: a
    disposable role (curator/compression/sub-agent) whose per-role save is OFF
    (client_meta['save_kv'] contract, Stage 1 — honored via
    _role_save_enabled so the FE toggle keeps working) never flushes — kills the
    junk sess:*:sub-agent:* seam bins. Unlabeled (role=None) traffic flushes
    (raw-thread main back-compat; the operator: main always saves) UNLESS the thread name
    proves disposable/auto provenance: hermes-sub-* (harness contract) or the
    auto-minted walk-in fallback agent-ip-*/auto-*. Returns (allowed, resolved_role)
    so every caller logs the decision with identifiers."""
    role = _bin_role(client_meta)
    if role not in (None, "main") and not _role_save_enabled(role, client_meta):
        return False, role
    _tid = thread_id or ""
    if role is None and _tid.startswith(("hermes-sub-", "agent-ip-", "auto-")):
        return False, role
    return True, role


def _tooltail_scan_covered_enabled() -> bool:
    """crit2/crit3 (Fix A — emergency FLOOR, default OFF) — True IFF the crit3
    tool-opaque restore-skip should scan the ENTIRE bin-covered span (start=0) instead of
    only the divergent tail (start=common).

    DEFAULT OFF (unset / 0/false/no/off): pre-Fix-A behavior — the skip scans only
    ``inc_messages[common:]`` (the tail beyond the settled common prefix). ON
    (1/true/yes/on): scan ``inc_messages[0:]``, so a tool-opaque turn ANYWHERE the
    clean/shadow bin covers (even inside the reused prefix) forces the restore to
    safe-degrade to the engine's native get_common_prefix reuse.

    This is the EMERGENCY FLOOR behind Fix B: with Fix B (default ON) the saved bin already
    byte-matches the historical resend across the covered tool region, so the primary path
    is MATCH+REUSE and this wider skip stays OFF. Turn it ON only to fall back to native
    reuse for the whole covered span if a covered tool region is still drifting (Fix B is
    the proof path; this is the rollback safety net)."""
    v = os.environ.get("TURBOHAUL_TOOLTAIL_SCAN_COVERED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


# crit2/crit3 (Fix B): the ``<think>\n...\n</think>\n\n`` block the chat template
# emits position-based for the ASSISTANT turn after last_query_index. ANCHORED to the
# ``<|im_start|>assistant\n`` header (design review): an UN-anchored global strip also nukes a
# LITERAL ``<think>...</think>\n\n`` embedded in USER/TOOL content, and it is commonly dogfooded
# by pasting this very template + engine logs -> those turns are saturated with
# literal think blocks -> the strip would corrupt the save prompt -> REUSE->CLEAR regression
# (silent Fix-B failure on the exact threads it targets). The capture group keeps the header
# (see the ``\1`` backref in _strip_think_scaffold); ONLY the assistant think block is
# removed. Non-greedy + DOTALL pairs each open with its own close and spans the interior
# newlines, so it removes the FILLED block AND the empty ``<think>\n\n</think>\n\n`` scaffold.
_THINK_SCAFFOLD_RE = re.compile(r"(<\|im_start\|>assistant\n)<think>\n.*?\n</think>\n\n", re.DOTALL)


def _strip_think_scaffold(rendered: str, messages: list | None = None) -> str:
    """crit2/crit3 (Fix B) — remove the assistant ``<think>\\n...\\n</think>\\n\\n``
    scaffold from an ALREADY-RENDERED chat-template prompt so a SAVE probe's prefill
    byte-matches the render those SAME covered turns receive once they fall BEFORE
    last_query_index on the next resend (historical position -> the template emits no
    scaffold for them).

    ANCHORED (design review): only a think block IMMEDIATELY preceded by the
    ``<|im_start|>assistant\\n`` header is removed; the ``\\1`` backref KEEPS that header and
    drops only the think, so a literal ``<think>...</think>\\n\\n`` sitting in USER/TOOL
    content (e.g. a user pasting this template + logs into a chat) SURVIVES untouched
    -> no self-inflicted REUSE->CLEAR on those threads.

    GATE (byte target): ``<|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n\\n<tool_call>``
    -> ``<|im_start|>assistant\\n<tool_call>``; the empty scaffold ``<think>\\n\\n</think>\\n\\n``
    and a text answer ``...assistant\\n<think>\\n{r}\\n</think>\\n\\nThe answer.`` ->
    ``...assistant\\nThe answer.`` collapse identically.

    PURE + None/non-str safe (passthrough) + idempotent (after one pass the header remains but
    the think is gone, so a second pass finds no ``<think>`` after the header -> no-op)."""
    if not isinstance(rendered, str):
        return rendered
    _preserved = set()
    for _m in (messages or []):
        if isinstance(_m, dict) and _m.get("role") == "assistant":
            _rc = _m.get("reasoning_content")
            if isinstance(_rc, str) and _rc.strip():
                _preserved.add(_rc.strip())
    if not _preserved:
        return _THINK_SCAFFOLD_RE.sub(r"\1", rendered)

    def _sub(mo):
        _block = mo.group(0)[len(mo.group(1)):]  # "<think>\n{body}\n</think>\n\n"
        _body = _block[len("<think>\n"):-len("\n</think>\n\n")]
        if _body.strip() in _preserved:
            return mo.group(0)  # preserved reasoning (DATA) — the resend re-emits it
        return mo.group(1)      # positional scaffold — strip (original behavior)

    return _THINK_SCAFFOLD_RE.sub(_sub, rendered)


def _turn_is_tool_opaque(turn) -> bool:
    """True IFF a single incoming turn is HASH-INVISIBLE to _prefix_hash_chain in a way
    that lets its tokens drift undetected (see _tooltail_restore_skip_enabled).

    Tool-opaque when ANY of (crit3 constraint #2):
      - role == "tool"                            (a tool RESULT turn)
      - role == "assistant" AND truthy tool_calls (a tool-CALL turn; content null)
      - content is null / empty                   (contributes nothing distinguishing
                                                   to the chain hash -> unverifiable)
    `content` is read exactly as _prefix_hash_chain reads it (turn.get("content", "")),
    so the null/empty test matches what the chain would (fail to) distinguish. A
    non-dict turn is coerced to str() by the chain (always non-empty) so it is never
    tool-opaque here. PURE; never raises."""
    if not isinstance(turn, dict):
        return False
    role = turn.get("role", "")
    if role == "tool":
        return True
    if role == "assistant" and turn.get("tool_calls"):
        return True
    # review note D: legacy OpenAI function-calling shapes (pre-tool_calls) are equally
    # hash-opaque and drift the same way — cover them too.
    if role == "function":
        return True
    if role == "assistant" and turn.get("function_call"):
        return True
    content = turn.get("content", "")
    if content is None:
        return True
    if isinstance(content, str) and not content.strip():
        return True
    if isinstance(content, (list, dict)) and not content:
        return True
    return False


def _divergent_tail_is_tool_opaque(inc_messages, common: int, *, scan_covered: bool = False) -> bool:
    """SINGLE source (crit3 code-unification) called by BOTH the warm force gate
    (_maybe_force_clean_restore) AND the cold gate (_restore_slot_kv): True IFF the
    span a restore would rely on contains a tool-opaque turn.

    Scans `inc_messages[common:]` — the first diverging turn and everything the
    clean/shadow bin covers beyond the settled common prefix (constraint #2: a tool
    turn strictly BEFORE `common`, already inside the stable reused prefix, does NOT
    count -> a settled tool region never over-triggers the skip). `inc_messages` is
    the admission messages (client_meta["messages"]) which are 1:1 aligned with the
    inc_chain `common` indexes into. PURE: no I/O; empty/None messages (e.g. a streamed
    submit that carried none) -> False = safe-degrade to the pre-crit3 restore (never
    worse than today). `common` is clamped to >= 0.

    crit2/crit3 (Fix A — emergency FLOOR): when `scan_covered` is True the scan
    starts at 0 (the ENTIRE bin-covered span, including the reused prefix) instead of
    `common`, so a tool-opaque turn ANYWHERE the bin covers forces the skip. The caller
    passes `scan_covered=_tooltail_scan_covered_enabled` (default OFF); the env read is
    kept OUT of this function so it stays PURE + unit-testable by passing the bool
    directly."""
    if not inc_messages:
        return False
    start = 0 if scan_covered else (common if common > 0 else 0)
    for turn in inc_messages[start:]:
        if _turn_is_tool_opaque(turn):
            return True
    return False


def _kv_shadow_save_fn(model_tag: str, sid: int, thread_hash: str, port: int) -> str:
    """Option A shadow-reprefill save filename. DISTINCT `.shadow` marker
    (load-bearing): derived from the canonical clean/normal bin name (kv_save_fn,
    single source of the base scheme) with `.shadow` inserted before `.bin`. Placing
    `.shadow` AFTER the slot number keeps the shadow bin a SEPARATE artifact from the
    clean_prefix anchor (`.slot{sid}.bin`) so a mispredicted shadow can never
    overwrite/demote it, AND makes _restore_slot_kv's bin parser reject it (its slot
    token parses to `{sid}.shadow`, which is not all-digits -> skipped, never
    restored)."""
    return kv_save_fn(model_tag, sid, thread_hash, port)[:-len(".bin")] + ".shadow.bin"


def _kv_shadow_meta_fn(model_tag: str, sid: int, thread_hash: str, port: int) -> str:
    """Sidecar metadata filename for the shadow-reprefill bin (see _kv_shadow_save_fn
    for the distinct-marker rationale)."""
    return kv_meta_fn(model_tag, sid, thread_hash, port)[:-len(".json")] + ".shadow.json"


def _ckpt_sidecar_enabled() -> bool:
    """pin-and-ship: mirror the ENGINE's TURBOHAUL_CKPT_SIDECAR gate
    (same env var, same container) so the manager only finalizes the checkpoint-
    ladder .ckpt sidecar name when the engine is actually writing one. Absent/off
    -> no .tmp.ckpt is produced, so the finalize is a no-op either way; the flag
    just skips the os.path.exists probe on the OFF path."""
    return os.environ.get("TURBOHAUL_CKPT_SIDECAR", "").strip().lower() in ("1", "true", "yes", "on")


def _finalize_ckpt_sidecar(tmp_path: str, final_path: str, bin_fn: str) -> None:
    """pin-and-ship fast-follow: finalize the engine's checkpoint-ladder
    sidecar NAME alongside the atomic .bin rename. The engine writes the sidecar to
    ``<save-filename>.ckpt`` == ``<bin>.tmp.ckpt`` (save filename is ``<bin>.tmp``);
    the atomic ``os.replace(<bin>.tmp -> <bin>)`` finalizes ONLY the .bin, orphaning
    ``<bin>.tmp.ckpt``. SLOT_RESTORE reads ``<bin>.ckpt`` -> ABSENT -> empty ladder ->
    stale-too-large CLEAR+reprefill (the pre-fix live behavior). This renames it to
    ``<bin>.ckpt``.

    BEST-EFFORT + called AFTER the .bin rename (bin FIRST): a crash/restore in the
    window safely CLEARs (a .bin can exist with NO/older .ckpt, never a .bin with a
    mismatched newer .ckpt) — and the sidecar's own bin_nwrite header ties it to the
    .bin so a stale .ckpt can never mispair (engine rejects it). Flag-gated; any
    failure is logged + swallowed (never breaks the save)."""
    if not _ckpt_sidecar_enabled():
        return
    ckpt_tmp = tmp_path + ".ckpt"
    if not os.path.exists(ckpt_tmp):
        return
    try:
        os.replace(ckpt_tmp, final_path + ".ckpt")
    except OSError as e:  # noqa: BLE001 — best-effort; restore safely CLEARs on absence
        log.warning("ckpt sidecar finalize failed for %s: %s (restore will CLEAR-safe)", bin_fn, e)


class EventBus:
    """Pub-sub for state-level events broadcast to /ws/state subscribers.

    Per the redaction policy: callers are responsible for emitting only
    safe events. This bus enforces a denylist (prompt/response/stderr/context)
    on publish as defense-in-depth — even if a caller accidentally includes one
    of those keys, it gets stripped before fan-out.
    """

    REDACTED_KEYS: frozenset[str] = frozenset({
        "prompt",
        "response",
        "context",
        "stderr",
        "stdout",
        "messages",
    })

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


# Multi-slot refactor PHASE-0 (resident-registry scaffold). The manager today
# runs a single-sidecar invariant (one model loaded at a time); the eventual
# multi-slot work needs a per-model resident registry so >1 sidecar can be
# tracked concurrently. This constant PINS the registry to exactly ONE resident
# for Phase-0 so runtime behaviour is byte-for-byte identical to the deployed
# v0.3.8 single-sidecar manager. Phase-1 (dispatcher / driver-tasks / LRU-evict
# / multi-spawn — explicitly OUT of Phase-0 scope) is the place that raises this
# and migrates the live FSM state onto the registry behind its own concurrency
# tests. Do NOT bump this in Phase-0.
MAX_PARALLEL_SIDECARS = 1

# P1d max times the dispatcher re-queues an unroutable slot (all
# residents busy, no idle victim to evict) before failing its future (a 503-
# equivalent) so an unroutable slot can't busy-loop the dispatcher / starve.
_MAX_DISPATCH_DEFERS = 50

# bounded poll interval for the cap<=1 fan-out drain's
# continuous rider-admit. The drain wakes on EITHER a slot completion OR this
# timeout, so a same-model rider that arrives mid-burst (after the one-shot
# admit already ran, while a --parallel slot is still free) joins within this
# bound instead of waiting out the anchor. Small relative to any real request
# (sub-agents run for seconds); the wake only does a cheap non-blocking
# pop_next while a fan-out is already active.
_FANOUT_ADMIT_POLL_S = 0.1

# severity item P2 residency bounded warm-hold applied when a client's
# keep_alive would unload the model NOW (idle_seconds <= 0) BUT the very next
# queued request is the SAME model. Keeps the resident loaded just long enough
# for that queued same-model request to warm-inherit on the next worker tick
# (it is popped within ~_FANOUT_ADMIT_POLL_S), instead of paying a teardown +
# respawn. Never indefinite — if the queued request vanishes before it is
# popped, this window idle-expires and the model tears down normally.
_SAME_MODEL_QUEUED_HOLD_S = 30.0

# The single registry key under which the lone Phase-0 resident lives. Phase-1
# keys residents by ``model_tag``; in Phase-0 the registry holds exactly one
# entry under this sentinel so ``_residents`` is non-empty and shape-correct
# without claiming a model binding the FSM hasn't actually made yet.
_SINGLETON_RESIDENT_KEY = "__phase0_singleton__"


class ResidentState(enum.StrEnum):
    """P1d Resident lifecycle. RESERVED_LOADING: budget claimed under
    _registry_lock, sidecar spawning. ACTIVE: serving. GRACE: warm grace window.
    IDLE_EVICTABLE: warm-idle + LRU-evictable. DEAD: driver died/evicted, pending
    deregister. The dispatcher is the SOLE writer of the _residents dict + each
    resident's ``state`` (under _registry_lock); the per-resident driver owns r.*."""

    RESERVED_LOADING = "RESERVED_LOADING"
    ACTIVE = "ACTIVE"
    GRACE = "GRACE"
    IDLE_EVICTABLE = "IDLE_EVICTABLE"
    DEAD = "DEAD"

# LEGAL_TRANSITIONS: ResidentState transition guard (a tracked issue)
# Prevents impossible state transitions during concurrent access.
_RESIDENT_LEGAL_TRANSITIONS: dict[ResidentState, frozenset[ResidentState]] = {
    ResidentState.RESERVED_LOADING: frozenset({
        ResidentState.ACTIVE,
        ResidentState.DEAD,
    }),
    ResidentState.ACTIVE: frozenset({
        ResidentState.GRACE,
        ResidentState.IDLE_EVICTABLE,
        ResidentState.DEAD,
    }),
    ResidentState.GRACE: frozenset({
        ResidentState.IDLE_EVICTABLE,
        ResidentState.DEAD,
    }),
    ResidentState.IDLE_EVICTABLE: frozenset({
        ResidentState.ACTIVE,
        ResidentState.DEAD,
    }),
    ResidentState.DEAD: frozenset(),  # terminal state
}

def _transition_resident_state(state: ResidentState, new_state: ResidentState) -> None:
    """Validate a resident state transition; log warning instead of raising.

    Self-transitions (new_state == state) are always permitted and return
    immediately. Genuinely illegal transitions are logged as warnings instead
    of raising ValueError, so detached asyncio worker/teardown tasks don't
    strand the resident on a throw. The caller is responsible for actually
    assigning r.state = new_state after this function returns.
    """
    if new_state == state:
        return
    allowed = _RESIDENT_LEGAL_TRANSITIONS.get(state, frozenset())
    if new_state not in allowed:
        log.warning(
            "Illegal resident state transition: %s -> %s (legal targets: %s)",
            state,
            new_state,
            sorted(allowed),
        )
        return
# Stream timeout constant (a tracked issue: eliminates duplicate 3600.0 literals)
# Previously hardcoded in 7+ places across manager.py
_STREAM_TIMEOUT_S: float = 3600.0  # 1 hour cap on stream waits

# M5 (WIN 2): idempotent completion-cache — kills the retry-storm
# RE-DECODE. When a NON-streaming request completes but its client already timed
# out / disconnected, the answer is discarded and the client's byte-identical
# retry re-decodes the SAME answer. We cache the completed result keyed by the
# FULL messages exact-bytes ⊕ thread_id ⊕ model_tag, so an identical retry
# returns it instantly (no enqueue, no engine call). Bounded LRU + TTL.
# Correctness > hit-rate: a MISS always falls through to normal processing; a
# stale / model-mismatched entry is NEVER served (swap-clear + model_tag-in-key
# + TTL guarantee this). Efficacy is gated on the harness resending byte-
# identical retries; if it does not, the exact key simply MISSES (harmless).
_COMPLETION_CACHE_MAX = 64  # max resident cached completions (oldest LRU-evicted)
_COMPLETION_CACHE_TTL_DEFAULT_S = 120.0

# Sentinel resolved onto a single-flight future when the LEADER's flight ended
# WITHOUT a cache write (eviction / timeout / exception / a cap>=2 path that did
# not write). A rider awaiting the flight sees it and falls through to normal
# processing instead of hanging. A plain object (NOT an exception) so a
# released-but-unawaited future never logs "exception was never retrieved".
_FLIGHT_FAILED = object()

# crit1 (tool-call reuse): the n_predict=0 KV prefill PROBES
# (_probe_and_save_clean_kv + _shadow_reprefill_and_save) must render the SAME
# tool-definitions preamble the live streaming request renders. Qwen's chat template
# injects `tools` at the FRONT of the prompt, so a tools-LESS probe saves a bin that
# diverges from a tools-BEARING request ~3-40 tokens in -> engine "large stale, CLEAR"
# + full reprefill on EVERY tool turn (byte-proven: C1-toolcall/toolresult pe~82k).
# Forward these knobs VERBATIM from client_meta so the probe tokenizes byte-identically
# to the live request (mirrors chat_completion._STREAM_FORWARDED_KNOBS's tool subset;
# duplicated here — not imported — to avoid the manager<->api import cycle). `tools` is
# NOT in the hash chain (kv_policy untouched), so this changes ONLY prefill rendering;
# a request WITHOUT tools -> every knob None -> payload byte-identical to today (tools-
# less turns cannot regress).
_KV_PROBE_TOOL_KNOBS = ("tools", "tool_choice", "parallel_tool_calls", "function_call", "functions")

# shadow byte-match self-check (DORMANT observability): max distinct
# threads kept in the per-thread probe stash. Bounded FIFO — the oldest write is
# evicted past this cap so the dict can never leak (see _record/_compare below).
_SHADOW_BYTEMATCH_CAP: int = 64


def _read_completion_cache_ttl_s() -> float:
    """TTL (seconds) for the completion-cache from TURBOHAUL_COMPLETION_CACHE_TTL_S
    (default 120.0). A malformed value falls back to the default — never raise at
    manager construction."""
    try:
        return float(
            os.environ.get(
                "TURBOHAUL_COMPLETION_CACHE_TTL_S", _COMPLETION_CACHE_TTL_DEFAULT_S
            )
        )
    except (TypeError, ValueError):
        return _COMPLETION_CACHE_TTL_DEFAULT_S


@dataclass
class Resident:
    """A single loaded-model sidecar tracked by the manager (PHASE-0 scaffold).

    This is the data structure the multi-slot refactor will key by ``model_tag``
    in ``TurbohaulManager._residents``. In PHASE-0 the registry is pinned to one
    resident (``MAX_PARALLEL_SIDECARS == 1``) and is NOT yet the authoritative
    source for the live FSM scalars — the manager continues to drive the
    deployed single-sidecar attributes (``_active_handle`` / ``_active_slot`` /
    ``_spawn_seq`` / the ``_idle_*`` holder) verbatim. The fields below mirror
    the exact shape Phase-1 will adopt; only ``inflight`` is wired to the live
    list (shared by reference with ``_inflight`` so in-place ``append``/``remove``
    on the manager's fan-out rider set stays visible through the resident view).
    The scalar fields stay at their construction placeholders in Phase-0 and
    become authoritative only when Phase-1 routes the FSM through the registry.

    Fields:
      model_tag:       the model this resident serves (None until Phase-1 binds).
      handle:          the live ``SidecarHandle`` (Phase-1 authoritative).
      port:            the sidecar's listen port (Phase-1 authoritative).
      inflight:        the concurrent fan-out rider Slots (anchor at index 0).
      spawn_seq:       monotonic spawn counter for fixed-port swap detection.
      idle_expires_at: monotonic deadline of the warm-idle hold, if any.
      active_slot:     the anchor Slot currently driven on this resident.
    """

    model_tag: str | None = None
    handle: "SidecarHandle | None" = None
    port: int | None = None
    inflight: list[Slot] = field(default_factory=list)
    spawn_seq: int = 0
    idle_expires_at: float | None = None
    active_slot: "Slot | None" = None
    # P1c state-migration: per-resident mirrors of the manager-global
    # idle holder (``_idle_handle`` / ``_idle_model_tag``) + the latest keep_alive
    # intent + this resident's own grace/idle timers. At ``MAX_PARALLEL_SIDECARS
    # == 1`` they mirror the singleton manager scalars 1:1 (byte-identical); P1d's
    # dispatcher makes them authoritative per concurrent resident. ``grace``/
    # ``idle`` are constructed + wired in ``TurbohaulManager.__init__`` (the
    # GraceTimer/IdleHotTimer classes live in queue.py, imported — queue.py
    # untouched per the RC guardrail).
    idle_handle: "SidecarHandle | None" = None
    idle_model_tag: str | None = None
    idle_thread_id: str | None = None
    idle_client_meta: dict | None = None
    idle_admission_ctx_len: int = 0
    latest_keep_alive_s: int | None = None
    grace: "GraceTimer | None" = None
    idle: "IdleHotTimer | None" = None
    # P1d dispatcher/concurrency fields. Populated under _registry_lock
    # at reservation (state / reserved_need_mib / parallel / main_gpu / port from the
    # in-lock manifest read) and on spawn (booting_pid then handle). ``inbox`` is the
    # dispatcher->driver slot hand-off queue; ``driver_task`` is this resident's
    # _drive_resident task; ``torn_down`` is the lock-guarded exactly-once teardown
    # claim shared by the driver finally + the death-supervisor reaper.
    state: ResidentState = ResidentState.ACTIVE
    reserved_need_mib: int = 0
    booting_pid: int | None = None
    parallel: int = 1
    main_gpu: int = 0
    # N1 the manifest split_mode this resident loaded under. Co-residence
    # is supported ONLY for single-GPU-pinned ('none') models on DISTINCT cards; a
    # layer/row/tensor-split sibling spans all cards (no free distinct card to
    # guarantee) so the cross-resident gate refuse-blinds against it (interim until
    # per-card layer-split accounting lands). Default 'layer' matches the footprint
    # degrade-open default.
    split_mode: str = "layer"
    # Per-model sleep_idle_seconds from the manifest. -1 = pin/keep-warm (never
    # idle-unload this model), 0 = unload immediately after request, N>0 = idle
    # timeout in seconds. Default 0 means "use global default" — the driver will
    # fall back to runtime.queue.idle_hot_load_seconds.
    sleep_idle_seconds: int = 0
    last_active_monotonic: float = 0.0
    torn_down: bool = False
    driver_task: "asyncio.Task | None" = None
    inbox: "asyncio.Queue | None" = None
    # P1 (DURABLE MANAGER B): residency tag for per-(role,session) ring.
    # Set on cold restore success to the ring_key of the restored session.
    # Used by warm paths to log would-reload when resident_tag != request_tag.
    resident_state_tag: "tuple[str, str] | None" = None


def _kvcache_ceiling_bytes() -> int:
    """Global byte-ceiling for the KV-cache save dir (sum of ``.bin`` files).

    Env ``TURBOHAUL_KVCACHE_MAX_BYTES`` (integer bytes); default 20 GiB. A value
    ``<= 0`` DISABLES the ceiling (the age/count knobs still apply). Best-effort:
    an unparseable value logs a warning and falls back to the default.
    """
    raw = os.environ.get("TURBOHAUL_KVCACHE_MAX_BYTES")
    if raw is None or not raw.strip():
        return 20 * 1024 ** 3
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        log.warning(
            "invalid TURBOHAUL_KVCACHE_MAX_BYTES=%r; falling back to 20 GiB", raw,
        )
        return 20 * 1024 ** 3


# === shadow-diagnosis INSTRUMENTATION helpers ======================
# Cheap, pure, never-raise fingerprints for the SHADOW_SAVE / KV_RESTORE /
# SHADOW_BYTEPARITY structured log lines. They emit LENGTHS + HASHES + offsets
# only (never raw prompt content), and are called ONLY from the off-hot-path
# save/restore/GC code (never on the response TTFT path). Instrumentation only —
# they drive NO decision.
def _fnv1a_64(s: str) -> str:
    """64-bit FNV-1a hex digest of a string. Non-crypto, allocation-light — used
    to fingerprint a think-free / resend region in a log line WITHOUT logging the
    raw content. Never raises (returns 'fnverr' on any failure)."""
    try:
        h = 0xCBF29CE484222325
        for b in s.encode("utf-8", errors="replace"):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f"{h:016x}"
    except Exception:
        return "fnverr"


def _first_diff_offset(a: str, b: str) -> int:
    """First character index at which ``a`` and ``b`` differ, or -1 if they are
    byte-identical. If one is a strict prefix of the other, returns the length of
    the shorter (the offset where the longer one still has bytes). Never raises."""
    try:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return -1 if len(a) == len(b) else n
    except Exception:
        return -2


class TurbohaulManager:
    """Top-level orchestrator.

    Responsibilities:
    - Boot reconcile: orphan reap + foreign-GPU detect + state.sqlite slot cleanup
    - Verify binary sha256 pin at boot
    - Accept fresh requests via submit → push to queue (head if grace match)
    - Expose status_snapshot for /status endpoint
    - Drive the FSM via worker_loop (skeleton in the initial phase; full streaming in a follow-up phase)
    - Clean shutdown
    """

    # lag-reducer (SAVE-path THROTTLE, a failure-mode guard). Re-save the clean-prefix
    # KV anchor only once the incoming clean-prefix chain has grown by >= this many
    # turns beyond the saved anchor's n_context_turns. Bounds BOTH the multi-GB re-save
    # cost AND the sawtooth reuse lag to ~this many turns. Documented tunable.
    LAGREDUCER_MIN_GROWTH_TURNS = int(os.environ.get("TURBOHAUL_LAGREDUCER_MIN_GROWTH_TURNS", "1"))  # SPEC-V2: per-turn re-save (the clean bin IS the always-fresh single copy); RED-HAT design note — env-tunable so the operator owns the SSD-write budget without a redeploy

    # ceiling-GC: minimum seconds between KV-cache file-GC passes. The
    # GC piggybacks on the park-sweep loop but is THROTTLED to this floor so a
    # scandir of the multi-GB save dir never runs on the 60s park cadence.
    _KV_GC_MIN_INTERVAL_S = 300

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
        # idle-holder wiring: manager-level idle holder (model warm post-grace).
        # When grace expires without a thread match, the sidecar is NOT
        # torn down -- it migrates here and stays alive for idle_seconds.
        # Next slot of same model_tag inherits the handle; different
        # model_tag tears it down first.
        self._idle_handle: SidecarHandle | None = None
        self._idle_model_tag: str | None = None
        self._idle_expires_at: float | None = None
        # Bug 3 fix: track thread_id through idle holder so the
        # C++ engine can save KV cache during idle teardown (it needs a
        # valid thread_id to do it — empty thread_id returns do_it=False).
        self._idle_thread_id: str | None = None
        # track admission_ctx_len through idle holder so KV save
        # uses the admission-time context length for prompt_len in metadata.
        self._idle_admission_ctx_len: int = 0
        # track client_meta through idle holder so warm-inherit can
        # restore session_id and other identity labels on cold-start requests.
        # Initialized to None to avoid AttributeError when no idle holder exists yet.
        self._idle_client_meta: dict | None = None
        # keep-alive handling: latest request's keep_alive intent across the ACTIVE_MATCH
        # chain on a single warm slot. Reset per anchor (_process_slot entry);
        # captured on ACTIVE for the anchor and on each ACTIVE_MATCH promotion
        # of a matched follow-up; consumed (cleared) at grace→idle entry. The
        # "latest request wins" rule mirrors Ollama keep_alive semantics
        # (timer resets on request receipt, not on response completion).
        # Without this, stale keep_alive from request N leaks into IDLE_HOT
        # window computed after request N+M — a failure-mode review.
        self._latest_keep_alive_s: int | None = None
        # the design /status metrics counters for client-disconnect
        # eviction observability. Updated in worker_loop's is_evicted branch.
        self._eviction_count: int = 0
        self._last_evicted_at_iso: str | None = None
        # The classifier (operator request) observability: per-event decision counts +
        # forced-clean-restore total + last decision, surfaced on /status. PROVES
        # "Turbohaul determines each event" (continuation / user-message /
        # compression / sub-agent / guard-skip) and whether it forced the clean bin.
        self._kv_classifier_counts: dict[str, int] = {}
        self._kv_classifier_forced: int = 0
        # F2 distinct COLD-path counter. forced_clean_restores is WARM-
        # only (set in _maybe_force_clean_restore); a working wave-return (cold clean-
        # restore in _restore_slot_kv) leaves it 0, so it is invisible. This counter
        # makes the cold clean-restore PROVABLE in PL's smoke.
        self._kv_classifier_wave_return: int = 0
        self._kv_classifier_last: dict | None = None
        # -PROOF (the operator absolute-proof surface): last emitted
        # per-request structured identity dict (see _emit_request_identity),
        # surfaced on /status so the FE can render it live. DISPLAY/
        # OBSERVABILITY ONLY — read-only surface of the class the manager
        # already resolved; drives no decision.
        self._last_request_identity: dict | None = None
        # M5 (WIN 2): idempotent completion-cache + single-flight.
        # _completion_cache maps completion-key -> (result, expiry_monotonic); an
        # OrderedDict so the oldest is LRU-evicted via popitem(last=False) once it
        # exceeds _COMPLETION_CACHE_MAX. _completion_inflight maps completion-key ->
        # the LEADER request's Future, so a concurrent byte-identical retry rides
        # the first flight (single-flight dedupe) instead of launching a 2nd decode.
        # SINGLE-MUTATOR NOTE: both dicts are touched ONLY on the event loop —
        # written by _process_slot (worker_loop) at completion, by submit_and_wait
        # on admission, and cleared on the teardown/swap paths — all on the one
        # loop with no await between check-and-register, so NO lock is needed (same
        # discipline as _active_*/_inflight). NON-STREAMING ONLY: the streaming
        # path (submit_for_streaming) never reads or writes either dict.
        self._completion_cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._completion_inflight: dict[str, asyncio.Future] = {}
        self._completion_cache_ttl_s: float = _read_completion_cache_ttl_s()
        # shadow byte-match self-check (DORMANT observability). Measures the
        # ONE assumption the shadow-reprefill reuse hinges on: does the manager's THINK-
        # FREE assistant-turn strip byte-match the harness's think-STRIPPED resend of the
        # SAME turn next turn? Turn N stashes its think-free assistant hash; turn N+1
        # admission compares the resent assistant turn against it. PURE read+hash+log+
        # count — writes NO KV, changes NO routing/restore/admission/save decision.
        # Bounded FIFO (evict oldest write past _SHADOW_BYTEMATCH_CAP) so it cannot leak.
        self._shadow_bytematch_probe: dict[str, dict] = {}
        self._shadow_bytematch_counts: dict[str, int] = {}
        # Option A shadow-reprefill (SAVE-side; INERT unless env
        # TURBOHAUL_SHADOW_REPREFILL). Per-(port, thread_hash) save-in-flight Lock
        # so a shadow reprefill+save can never overlap the next turn reading a
        # half-written shadow bin for the same thread (atomic tmp+os.replace already
        # protects readers; the lock serialises the WORK vs the single series).
        # Bounded (UNHELD locks pruned past _SHADOW_BYTEMATCH_CAP) so it cannot leak.
        # Counts (saved / skipped_* / reprefill_post_failed) surfaced on /status.
        self._shadow_reprefill_locks: dict[tuple, asyncio.Lock] = {}
        self._shadow_reprefill_counts: dict[str, int] = {}
        # (critical item freshness): the FRESHEST think-free shadow source
        # (think-free `messages` ready to re-save) from the most recent per-turn shadow
        # save, keyed by {thread_id, model_tag, port}. Updated by _shadow_reprefill_and_
        # save on EVERY reconstructable main/grace turn (so it tracks the LATEST turn's
        # slot, not the stale anchor — the outgoing slot object is nulled at swap time).
        # _shadow_save_at_swap reads it at the model-swap teardown to re-save a fresh
        # shadow AS CLOSE TO SIGTERM as possible. Re-keyed per (thread_id, model_tag)
        # so an intervening OTHER-model save can't clobber the outgoing
        # thread's source; bounded evict-oldest (cap 64). INERT unless SHADOW_REPREFILL.
        self._last_shadow_src: OrderedDict[tuple, dict] = OrderedDict()
        # shadow-diag (INSTRUMENTATION-ONLY): per-(thread_id, model_tag)
        # think-free recon text for SHADOW_BYTEPARITY, so candidate (d) survives
        # intervening shadow saves of OTHER threads (on a swap-back a 27b source is not
        # clobbered by 35b sub-agent saves the way the single-global _last_shadow_src
        # is). Bounded (evict-oldest). NEVER read by _shadow_save_at_swap / any
        # decision — pure observability; _last_shadow_src stays untouched.
        self._byteparity_recon_by_key: dict[tuple, str] = {}
        # previous-turn state for divergence capture on warm serve.
        # Keyed by (thread_id, model_tag). Gated by TURBOHAUL_DIVERGENCE_DEBUG=1 (OFF by default).
        self._divergence_prev: dict[tuple, dict] = {}
        # step (d) shadow-restore PREFERENCE (RESTORE-side; INERT unless env
        # TURBOHAUL_SHADOW_RESTORE_PREFER). Counts which bin the ONE warm forced-restore
        # seam preferred: {preferred: chose the think-free .shadow bin (a longer valid
        # prefix -> next decode strict-extends), clean_fallback: no valid+long-enough
        # shadow -> today's clean anchor}. Belt observability only; drives NO decision —
        # the force gate + prefix-validity + owner checks are UNCHANGED.
        self._kv_shadow_restore_counts: dict[str, int] = {}
        # crit3 (TOOL-tail restore skip): counts of forced/cold restores
        # SKIPPED because their divergent tail beyond the common prefix was tool-opaque
        # (hash-invisible -> the chain "match" is untrustworthy). {warm: the
        # _maybe_force_clean_restore force path, cold: the _restore_slot_kv wave-return
        # path}. Obs-only belt; the guard's real effect is the force=False / POST-skip
        # itself. Default OFF (TURBOHAUL_TOOLTAIL_RESTORE_SKIP) now that Fix B byte-matches
        # the covered tool region; flag-on is the emergency A/B-rollback floor.
        self._kv_tooltail_skip_counts: dict[str, int] = {}
        # I/O optimization: cache of SLOT_SAVE_DIR .json sidecar scans so
        # the O(N) listdir+load loop is NOT repeated on every probe/restore call.
        # _kvcache_dir_mtime tracks the last known SLOT_SAVE_DIR mtime at scan time;
        # the cache is invalidated whenever a write/delete in SLOT_SAVE_DIR occurs.
        self._kvcache_dir_mtime: float | None = None
        self._kvcache_clean_bins: dict[tuple, tuple[int, str, list, int] | None] = {}
        self._kvcache_shadow_bins: dict[tuple, tuple[int, str, list, int] | None] = {}
        # SPEC-V2 REWORK (disk-at-unload): live in-VRAM clean-anchor registry.
        # port -> {thread_hash, model_tag, pid, chain, prompt_len, stamp}. Written by
        # _probe_and_save_clean_kv after EVERY successful per-turn normalizing
        # prefill (which physically overwrites the engine slot -> keyed by port).
        # Read by _maybe_force_clean_restore (staleness/ownership gate) + the the classifier
        # classifier overlay. pid stamps the sidecar generation: a respawned engine
        # on the same port has a different pid -> record automatically FOREIGN (no
        # invalidation hooks). NEVER persisted: restart = empty = cold path.
        self._kv_vram_anchor: dict[int, dict] = {}
        # SHADOW-DIAGNOSIS instrumentation (INSTRUMENTATION ONLY, Sev-3):
        # per-cold-restore + byte-parity counters surfaced under /status.shadow_diag
        # so a forcing-full swap-back CONCRETELY names WHY (candidate a/b/c/d). These
        # drive NO decision — every writer is a best-effort log/counter belt.
        # restores: which bin the cold _restore_slot_kv chose per pass
        # (chose_shadow / chose_clean / chose_clean_withthink / fresh).
        # byteparity: manager reconstructed-think-free vs harness resend region
        # (match / diverge / no_recon_src) — makes candidate (d)
        # saved-but-BYTE-DIVERGES directly observable vs (c) wrong-bin.
        # evictions: a `.shadow` bin reaped by the ceiling/over-cap GC
        # (shadow_evicted) — candidate (b) saved-but-EVICTED: the
        # unprotected shadow is deleted FIRST under over-cap while
        # pinned dead-session clean anchors hold the floor.
        self._shadow_diag_counts: dict[str, dict[str, int]] = {
            "restores": {},
            "byteparity": {},
            "evictions": {},
        }
        # Latest ceiling-GC snapshot (total/over_cap/protected-bin bytes + the
        # top protected thread_hashes that hold the 64GB floor). Written once per
        # throttled GC pass by _gc_kv_cache (off the hot path); read into the
        # KV_RESTORE line + /status.shadow_diag.kvgc. None until the first pass.
        self._last_kvgc_snapshot: dict | None = None
        # background sweeper state-row finalizer counters.
        # Finalizes the design evictions that landed audit-only on the hot path
        # (a design note deferred state-row write to keep worker_loop off SQLite
        # fsync stall). Sweeper is OFF the hot path — its sync SQL is fine.
        self._slots_finalized_lifetime: int = 0
        self._last_sweep_iso: str | None = None
        self._sweeper_task: asyncio.Task | None = None
        # ceiling-GC throttle clock. Init to NOW so the first file-GC
        # fires one _KV_GC_MIN_INTERVAL_S AFTER startup (never at t=0), which
        # also keeps short-lived unit-test sweeper spins from touching the dir.
        self._last_kv_gc_monotonic: float = time.monotonic()
        # WIN 4 monotonic throttle for the sweeper-side fingerprint
        # purge (the startup call is unthrottled). 0.0 = never run -> first sweep
        # after enable purges immediately.
        self._last_fp_purge_mono: float = 0.0
        self._worker_task: asyncio.Task | None = None
        # P1 (DURABLE MANAGER B): per-(role,session) ring index + counters.
        # Ring: last-3 newest-overrides-oldest, keyed by (role, session_id) from client_meta.
        # Counters: ring_write, would_reload (warm/cold), residency_mismatch.
        # INERT when TURBOHAUL_DURABLE_RING=OFF (default).
        self._durable_ring_index: dict[tuple[str, str], list[dict]] = {}
        self._durable_ring_counts: dict[str, int] = {}
        # Live inference monitor (follow-up). Pure-observer plane:
        # worker_loop never reads/writes live_generation/live_output, so the
        # single-mutator FSM invariant is structurally untouched. _spawn_seq is
        # bumped by worker_loop at each _active_handle assignment so the poller
        # can detect a fixed-port (11500) sidecar swap across its httpx await.
        self._spawn_seq: int = 0
        self.live_generation: dict | None = None  # written ONLY by LiveSlotsPoller
        self.live_output = LiveOutputBuffer()      # fed ONLY by the streaming tee
        self._live_poller = None
        self._live_poller_task: asyncio.Task | None = None
        # P1e per-resident live-inference monitor (cap>=2). The single
        # LiveSlotsPoller above stays the cap<=1 path (byte-identical). At cap>=2 a
        # LiveResidentsSupervisor task polls EACH resident's /slots and writes its
        # generation into ``live_generations`` (keyed by model_tag), mirroring the
        # most-recently-active resident into ``live_generation`` (the back-compat
        # alias status_snapshot + live_stream.py already read). It also caches per-GPU
        # free VRAM off the hot path so status_snapshot can emit ``vram[]`` await-free
        # (status_snapshot is LOCK-FREE and must NEVER call nvidia-smi synchronously).
        self.live_generations: dict[str, dict] = {}   # model_tag -> generation block
        self._vram_free_mib: list[int] | None = None  # per-GPU free MiB (cached ~1Hz)
        self._vram_total_mib: list[int] | None = None  # per-GPU total MiB (boot-time read)
        self._live_supervisor = None
        self._live_supervisor_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        # fix: removed unused self._lock = asyncio.Lock. It had
        # zero acquire sites (single-task worker_loop discipline protects
        # _active_handle / _active_slot mutations). If Wave-6 lands a
        # second mutating task, re-add it purposefully and wrap the call
        # sites that actually need it.
        #
        # MULTI-SLOT PHASE-0 (resident-registry scaffold): introduce the
        # registry data structure the design anticipated, PINNED to one resident
        # (``MAX_PARALLEL_SIDECARS == 1``) so runtime behaviour is byte-for-byte
        # identical to the deployed single-sidecar manager. In Phase-0 the
        # registry is NOT yet authoritative — the FSM keeps driving the
        # ``_active_*`` / ``_inflight`` / ``_spawn_seq`` / ``_idle_*`` attributes
        # verbatim (zero call-site churn = the 480-green proof is a tautology).
        # The lone resident's ``inflight`` SHARES the live ``self._inflight``
        # list object so the registry view never diverges from the fan-out rider
        # set; the scalar fields stay at placeholders until Phase-1 migrates the
        # FSM onto the registry behind its own concurrency tests.
        #
        # ``_registry_lock`` is the lock the design said to "re-add purposefully"
        # when a second mutating task lands. It is DEFINED here but INTENTIONALLY
        # UNACQUIRED in Phase-0 (single-task worker_loop discipline is unchanged);
        # the Phase-1 dispatcher/driver-tasks that introduce a second mutator MUST
        # acquire it around every ``_residents`` read-modify-write.
        self._residents: dict[str, Resident] = {}
        self._registry_lock = asyncio.Lock()
        # N2 strong refs to load-bearing fire-and-forget tasks
        # (teardown / inbox-requeue / death-reap). asyncio holds only a WEAK ref to
        # a bare create_task result, so an unreferenced task can be GC'd mid-flight,
        # silently cancelling the cleanup. _spawn_bg parks each here and discards
        # on completion. Only used on the cap>=2 dispatcher path.
        self._bg_tasks: set[asyncio.Task] = set()
        # Eagerly create the one permanent Phase-0 resident (never dropped in
        # Phase-0 — its lifetime is the manager's). ``inflight`` is the SAME list
        # object as ``self._inflight`` so in-place append/remove stay in sync.
        self._residents[_SINGLETON_RESIDENT_KEY] = Resident(
            inflight=self._inflight,
            # P1c: the lone resident HOLDS the manager's grace/idle timers (same
            # objects at max=1, so every existing self.grace/self.idle read stays
            # byte-identical). P1d gives each concurrent resident its OWN timer
            # pair + routes the FSM through the active resident's pair.
            grace=self.grace,
            idle=self.idle,
        )
        self._binary_fd: int | None = None  # lifecycle hardening: TOCTOU-pinned fd
        # Event bus for /ws/state subscribers (redacted)
        self.event_bus = EventBus()
        # Injection points (default = real subprocess_mgr; tests inject mocks)
        self._spawn = spawn_fn or spawn_sidecar
        self._wait_healthy = health_fn or wait_until_healthy
        self._sigterm = sigterm_fn or drained_sigterm
        self._vram_verify = vram_fn or verify_vram_cleared
        # _complete_fn: replaced with httpx → llama-server /v1/chat/completions
        self._complete_fn = complete_fn or self._default_complete
        # flap/degradation telemetry (observe-only)
        self._telemetry = init_telemetry(
            log_dir=getattr(boot.storage, "state_db_path", None) and
                     boot.storage.state_db_path.parent / "telemetry",
            enabled=True,
        )

    async def _default_complete(self, slot: Slot, handle: SidecarHandle) -> dict | None:
        """Default no-op completion. The httpx proxy is wired via DI."""
        await asyncio.sleep(0.001)
        return None

    # === M5 (WIN 2): idempotent completion-cache + single-flight ======
    # Kills the retry-storm RE-DECODE: a non-streaming completion whose client
    # already gave up is cached so a byte-identical retry replays instantly. All
    # four helpers run ONLY on the event loop (no lock — see the __init__ note).

    def _completion_cache_key(
        self,
        model_tag: str,
        thread_id: str,
        context: "list[dict] | None",
        client_meta: "dict | None",
    ) -> "str | None":
        """sha256 of a CANONICAL serialization of the FULL messages list ⊕
        thread_id ⊕ model_tag ⊕ output-knobs. NON-STREAMING ONLY. Returns None (cache disabled
        for this request) on a streaming request, absent context, or ANY
        serialization error — the caller then falls through to normal processing.

        RED-HAT: the key hashes the EXACT bytes of the full messages list (NOT the
        lossy _prefix_hash_chain, NOT a join) plus thread_id AND model_tag, so a
        different thread or a mutated message can NEVER collide onto another
        request's answer, and a result is never served for the wrong model. The
        dict wrapper + sort_keys makes the three fields unambiguous (no delimiter-
        collision between messages and thread_id) and the serialization canonical."""
        if context is None:
            return None
        if isinstance(client_meta, dict) and client_meta.get("stream"):
            return None  # defensive — submit_and_wait is the non-streaming entry
        try:
            # M5 MOD (PL byte-review): fold the output-affecting generation
            # knobs into the key so two requests identical in messages+thread+model
            # but differing in ANY sampling/output knob (temperature / top_p / top_k /
            # max_tokens / seed / response_format / tools / reasoning_budget /
            # penalties / ...) do NOT collide onto one cached result. Local import
            # breaks the manager<-api circular dep (chat_completion imports manager,
            # not vice-versa) + keeps ONE source of truth for the knob list (no drift).
            # Absent knobs -> None uniformly, so the retry-storm case (a byte-identical
            # retry carrying identical knobs) still HITS.
            from turbohaul.api.chat_completion import _COMMON_FORWARDED_KNOBS
            knobs = (
                {k: client_meta.get(k) for k in _COMMON_FORWARDED_KNOBS}
                if isinstance(client_meta, dict)
                else {}
            )
            payload = json.dumps(
                {
                    "messages": context,
                    "thread_id": thread_id,
                    "model_tag": model_tag,
                    "knobs": knobs,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except (TypeError, ValueError, ImportError, AttributeError):
            return None  # unserializable/uninit -> no cache, never fail the request

    def _completion_cache_lookup(self, key: str) -> "dict | None":
        """Return a FRESH cached result for key, or None on miss / expiry. An
        expired entry is dropped on access; a hit is LRU-touched (move_to_end)."""
        entry = self._completion_cache.get(key)
        if entry is None:
            return None
        result, expiry = entry
        if time.monotonic() >= expiry:
            self._completion_cache.pop(key, None)
            return None
        self._completion_cache.move_to_end(key)
        return result

    def _completion_cache_store(self, slot: Slot, result: "dict | None") -> None:
        """WRITE site helper: cache a NON-streaming completion + release its single-
        flight future. No-op unless this slot is a cache LEADER (its submit_and_wait
        stashed ``completion_cache_key``). Only a real dict result is cached; a None
        result (no backend wired) is not cached but still releases riders so they do
        not hang. Bounded LRU: evict the oldest when over _COMPLETION_CACHE_MAX. Any
        bookkeeping error is swallowed — the cache must NEVER break a completion."""
        key = getattr(slot, "completion_cache_key", None)
        if key is None:
            return
        try:
            if isinstance(result, dict):
                self._completion_cache[key] = (
                    result, time.monotonic() + self._completion_cache_ttl_s,
                )
                self._completion_cache.move_to_end(key)
                while len(self._completion_cache) > _COMPLETION_CACHE_MAX:
                    self._completion_cache.popitem(last=False)  # drop the oldest
            # Resolve + retire the single-flight future so any riders get the answer.
            fut = self._completion_inflight.pop(key, None)
            if fut is not None and not fut.done():
                fut.set_result(result if isinstance(result, dict) else _FLIGHT_FAILED)
        except Exception:  # noqa: BLE001 — cache is best-effort, never fail a request
            log.exception("completion-cache store failed (best-effort)")

    def _completion_cache_clear(self, reason: str = "") -> None:
        """SWAP-CLEAR: drop every cached completion + release every single-flight
        rider. MUST fire whenever the model swaps / the active sidecar is torn down
        — cached results are engine/build-specific, so an answer must NEVER survive
        a swap (RED-HAT). Released riders fall through to normal processing against
        the NEW engine. Idempotent (clearing empty dicts is a no-op)."""
        if self._completion_cache:
            self._completion_cache.clear()
        if self._completion_inflight:
            for fut in list(self._completion_inflight.values()):
                if not fut.done():
                    fut.set_result(_FLIGHT_FAILED)
            self._completion_inflight.clear()

    def _emit_request_identity(
        self, *, ip, model_tag: str, client_meta: dict | None, thread_id: str,
    ) -> None:
        """-PROOF (the operator absolute-proof surface): DISPLAY/
        OBSERVABILITY ONLY. Emits ONE greppable structured log line
        (``R2B_REQ_IDENTITY``) + stashes the same dict on
        ``self._last_request_identity`` for ``/status`` — proof that the
        manager reads + trusts the structured client_meta identity fields
        instead of guessing off the thread_id prefix.

        ``resolved_class`` is a READ-ONLY surface of ``kv_classify.
        _class_from_label`` — the SAME read already used elsewhere (e.g.
        ``_durable_ring_key``); this call makes NO decision and drives
        nothing. Best-effort / None-safe: wrapped so a malformed
        ``client_meta`` can NEVER raise into the decode path (mirrors the
        existing best-effort ``identity_recompose`` log pattern in
        chat_completion.py).
        """
        try:
            from turbohaul.kv_classify import _class_from_label
            cm = client_meta or {}
            identity = {
                "ip": ip,
                "model_tag": model_tag,
                "session_id": cm.get("session_id"),
                "is_main": cm.get("is_main"),
                "is_sub_agent": cm.get("is_sub_agent"),
                "is_curator": cm.get("is_curator"),
                "is_compression": cm.get("is_compression"),
                "resolved_class": _class_from_label(cm),
                "thread_id": thread_id,
            }
            log.info("R2B_REQ_IDENTITY %s", json.dumps(identity))
            self._last_request_identity = identity
        except Exception:  # noqa: BLE001 — display-only, never fail a request
            log.debug("R2B_REQ_IDENTITY emit failed (ignored)", exc_info=True)

    async def submit_and_wait(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
        timeout_s: float = 7200.0,
        disconnect_event: "asyncio.Event | None" = None,
        admission_ctx_len: int = 0,
        admission_hash_chain: list[str] | None = None,
    ) -> tuple[Slot, Any]:
        """Submit + await the slot's completion. Returns (slot, completion_result).

        the design caller may pass a ``disconnect_event`` that the
        route's ``watch_disconnect`` task sets on client close. The queue
        pops with eviction-awareness; if the slot is evicted before activation,
        worker_loop fails the completion_future with SlotEvictedError which
        propagates out of this ``await`` for the caller to map to HTTP 499.

        M5 (WIN 2): idempotent completion-cache + single-flight, on the
        NON-streaming path only. BEFORE enqueue we compute the completion-key and
        either (a) return a fresh cached result instantly on a HIT (no enqueue, no
        engine call), (b) ride an in-flight byte-identical request's future
        (single-flight dedupe), or (c) become the flight LEADER — register the
        future, process normally, and let the _process_slot WRITE site cache the
        result + resolve the future at completion. A cache HIT / rider returns
        ``(None, result)``; callers on this path use only ``result`` and never
        dereference the slot. Any cache-path error falls through to normal
        processing (correctness > hit-rate). The check-and-register is await-free
        so two identical requests can never both become leader.
        """
        # --- M5 completion-cache READ + single-flight admission (non-streaming) ---
        cache_key = self._completion_cache_key(
            model_tag, thread_id, context, client_meta
        )
        am_leader = False
        if cache_key is not None:
            hit = self._completion_cache_lookup(cache_key)
            if hit is not None:
                return None, hit  # instant retry replay — no enqueue, no decode
            existing = self._completion_inflight.get(cache_key)
            if existing is not None:
                # A byte-identical request is already in flight — ride its future.
                try:
                    riden = await existing
                except Exception:  # noqa: BLE001 — leader failed; re-submit below
                    riden = _FLIGHT_FAILED
                if riden is not _FLIGHT_FAILED:
                    return None, riden
                # Leader's flight failed -> fall through as an ordinary request. Do
                # NOT re-lead (worst case mirrors no-single-flight, still correct).
            else:
                # Become the flight leader: register BEFORE enqueue (await-free from
                # the get above) so concurrent siblings dedupe onto this future.
                self._completion_inflight[cache_key] = (
                    asyncio.get_running_loop().create_future()
                )
                am_leader = True

        try:
            slot = await self.submit(
                model_tag=model_tag,
                prompt=prompt,
                thread_id=thread_id,
                context=context,
                client_meta=client_meta,
                wait_for_completion=True,
                disconnect_event=disconnect_event,
                admission_ctx_len=admission_ctx_len,
                admission_hash_chain=admission_hash_chain,
            )
            # -PROOF: emit the proof-surface identity line ONCE per
            # admitted request, at the chokepoint where client_meta + thread_id +
            # model_tag co-exist. slot.thread_id (not the raw param) — submit
            # may have auto-derived it. Best-effort (see _emit_request_identity).
            self._emit_request_identity(
                ip=(slot.client_meta or {}).get("ip"),
                model_tag=model_tag,
                client_meta=slot.client_meta,
                thread_id=slot.thread_id,
            )
            if am_leader:
                # Stash the key so the _process_slot WRITE site can cache the result
                # and resolve this leader's single-flight future keyed by it.
                slot.completion_cache_key = cache_key
            result = await asyncio.wait_for(slot.completion_future, timeout=timeout_s)
            return slot, result
        finally:
            # LEADER cleanup (try/finally so it can't leak): if the flight ended
            # WITHOUT a cache write (eviction / timeout / exception / a cap>=2 path
            # that did not write), _process_slot never resolved the inflight future
            # -> release riders here so they fall through instead of hanging. On the
            # normal cap<=1 SUCCESS path _process_slot already popped it, so
            # ``pop(..., None)`` returns None and this is a no-op.
            if am_leader:
                pending = self._completion_inflight.pop(cache_key, None)
                if pending is not None and not pending.done():
                    pending.set_result(_FLIGHT_FAILED)

    async def submit_for_streaming(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
        disconnect_event: "asyncio.Event | None" = None,
        admission_ctx_len: int = 0,
        admission_hash_chain: list[str] | None = None,
    ) -> Slot:
        """SSE streaming pass-through.

        Submit a request that will be consumed by an SSE-streaming route. Unlike
        submit_and_wait, this returns the slot immediately (without awaiting
        the completion_future). Streaming coordination events are armed in
        submit BEFORE the slot is enqueued, so the worker can NEVER observe
        None events (root-fix). This method makes the arming
        idempotent — if events are already present (set by submit), it
        preserves them; otherwise it creates them as a safety net.

          - ``slot.stream_ready_event``: worker_loop sets this once the slot
            reaches ACTIVE and the SidecarHandle is stored on
            ``slot.stream_handle``. The route awaits this before opening its
            own httpx.stream to the sidecar's port.
          - ``slot.stream_done_event``: the route sets this when the stream
            closes (normal exhaustion, client disconnect, or error). Only then
            does worker_loop advance the slot ACTIVE → GRACE.

        client_meta MUST include ``"stream": True`` for worker_loop to
        recognise the streaming path. Existing non-streaming callers are
        unaffected.

        A design review verdict on this design: keeping the slot
        ACTIVE for the full stream lifetime prevents ACTIVE_MATCH from
        promoting a second submission against the same sidecar (single-slot
        invariant preserved).
        """
        slot = await self.submit(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
            wait_for_completion=True,  # still attach future so error paths surface
            disconnect_event=disconnect_event,
            admission_ctx_len=admission_ctx_len,
            admission_hash_chain=admission_hash_chain,
        )
        # events are armed in submit before enqueue. Make this
        # idempotent — only create if somehow still None (defensive).
        if slot.stream_ready_event is None:
            slot.stream_ready_event = asyncio.Event()
        if slot.stream_done_event is None:
            slot.stream_done_event = asyncio.Event()
        # -PROOF: emit the proof-surface identity line ONCE per
        # admitted streaming request (mirrors submit_and_wait). Best-effort.
        self._emit_request_identity(
            ip=(slot.client_meta or {}).get("ip"),
            model_tag=model_tag,
            client_meta=slot.client_meta,
            thread_id=slot.thread_id,
        )
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
        # that's a CLI-flag decision)
        foreign = detect_foreign_gpu_apps()

        # 3. state.sqlite reconcile: any slot whose pid is no longer alive → COLD
        check_alive = pid_is_alive_fn or _pid_is_alive
        # the design: read + slot-write stay on state_db_session; audit-write uses pool.
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
        """Verify + TOCTOU-pin llama_server_binary at boot (lifecycle hardening).

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
        admission_ctx_len: int = 0,
        admission_hash_chain: list[str] | None = None,
    ) -> Slot:
        """Accept a fresh inference request.

        - If thread_id is empty, auto-derive from prompt-prefix-hash (edge-case fix).
        - If grace window is open for this thread+model → enqueue at FIFO HEAD
          + restart grace timer (max_grace_extensions cap applies).
        - admission_ctx_len: incoming context size recorded at admission
        - admission_hash_chain: incoming turn-hash chain recorded at admission
        - Otherwise → normal FIFO enqueue.

        Raises RuntimeError if the manager is shutting down.
        """
        # C3 fix (design review): refuse new submissions after shutdown signal so
        # callers fail fast instead of hanging on a completion_future that
        # will never resolve (worker is dead, queue.close clears staging).
        # Raises queue.QueueClosed to match existing test_manager contract +
        # the documented queue-closed exception type.
        if self._stop_event.is_set():
            from turbohaul.queue import QueueClosed
            raise QueueClosed(
                "TurbohaulManager is shutting down — new submissions are refused."
            )

        if not thread_id:
            thread_id = derive_thread_id_prefix_hash(
                prompt, model_tag,
                prefix_tokens=self.runtime.queue.prefix_token_count,
            )

        slot = Slot.new(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
            admission_ctx_len=admission_ctx_len,
            admission_hash_chain=admission_hash_chain,
        )
        # PIN#1: stamp the ADMISSION-resolved role BEFORE enqueue —
        # the warm-inherit clobber (T3 restore) replaces slot.client_meta wholesale
        # for same-session inherits, so a role read there can mis-see a disposable
        # as "main" (FP F1). This attribute survives the clobber (client_meta is
        # the only field it replaces) and equals _bin_role-at-admission by
        # construction (same vocabulary as _disp_role).
        slot.admission_role = _bin_role(client_meta)
        # the design caller-attached disconnect_event (lazy-init in
        # the route's own loop per a design note). None for non-HTTP callers
        # (BootInventory orphan-replay, internal probes, tests pre-attach).
        slot.disconnect_event = disconnect_event

        # root-fix: arm streaming events BEFORE enqueue so the worker
        # can NEVER see None. submit_for_streaming previously armed these AFTER
        # submit returned — leaving a window where ACTIVE_MATCH promotion
        # could observe None events and assert-crash. Arming here closes the
        # race entirely (events always exist before the slot is visible to worker).
        if isinstance(client_meta, dict) and client_meta.get("stream"):
            slot.stream_ready_event = asyncio.Event()
            slot.stream_done_event = asyncio.Event()
            slot.stream_handle = None

        # Attach a future BEFORE enqueue so worker_loop can resolve it on completion.
        if wait_for_completion:
            slot.completion_future = asyncio.get_running_loop().create_future()

        # Grace-window matched-thread shortcut
        if self.grace.matches(thread_id, model_tag):
            await self.queue.enqueue_head(slot)
            # restart_for_followup may return False if at extension cap; that's fine,
            # the request still queues at head once, but the slot will pop next cycle.
            self.grace.restart_for_followup()
        else:
            await self.queue.enqueue(slot)

        # Audit — the design: slot-write stays on state_db_session; audit-write goes
        # through the pool wrapped in asyncio.to_thread (a review guard sync-only guard).
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

        # telemetry — capture request arrival + queue state
        try:
            self._telemetry.on_request_arrival(slot)
            self._telemetry.on_queue_state(self.queue.depth(), slot)
        except Exception:
            pass  # observe-only: never break the hot path

        # shadow byte-match self-check (DORMANT, observability ONLY): on this
        # (N+1) admission, compare the harness's resent think-stripped assistant-N turn
        # against turn N's stashed think-free hash. PURE measurement — the slot is already
        # enqueued above; this only reads/hashes/logs/counts and changes NO decision.
        # Self-protecting (internal try/except) so it can never raise into submit.
        self._compare_shadow_bytematch_probe(thread_id, context, client_meta)

        # R-COMP (the operator): a LABELED is_compression request for a session marks
        # that session's MAIN clean anchor stale at ADMISSION. Label-gated
        # (resolved class must be compression per the PL priority order) ->
        # zero cost for normal traffic; best-effort, never raises into submit.
        try:
            _cm = client_meta or {}
            if _cm.get("is_compression") and _cm.get("session_id"):
                from turbohaul.kv_classify import _class_from_label, CLASS_COMPRESSION
                if _class_from_label(_cm) == CLASS_COMPRESSION:
                    self._mark_main_bin_stale_for_session(
                        str(_cm.get("session_id")), reason="is_compression_submit")
        except Exception:
            log.debug("R-COMP submit-time stale-mark failed (best-effort)", exc_info=True)

        return slot

    # === Status snapshot =====================================================

    def status_snapshot(self) -> dict:
        """/status payload."""
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
                        # Redaction: only first 8 chars of thread_id exposed
                        "thread_id_prefix": (slot.thread_id or "")[:8],
                        "pid": self._active_handle.pid,
                        "port": self._active_handle.port,
                        # the engine-op badge work: named engine operation for FE Dashboard pill
                        "engine_op": getattr(slot, "engine_op", "idle"),
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
                    # the engine-op badge work: named engine operation for FE Dashboard pill
                    "engine_op": getattr(slot, "engine_op", "idle"),
                }

        grace_info: dict | None = None
        # SPEC-V2 WAVE D (grace display truth): the advertised GraceTimer keeps
        # ticking through a follow-up's entire prefill+decode (re-stamped only at
        # arrival and post-completion), while the REAL unload deadline is a serial
        # worker_loop local that can never fire mid-serve — the BE never unloads
        # mid-prefill. Suppress the advertised clock while a serve holds the
        # engine so the FE can never render a mid-prefill 'unload in Ns'
        # countdown. active_info is non-None exactly when an ACTIVE/ACTIVE_MATCH
        # slot has a live handle (built above).
        if not self.grace.expired() and active_info is None:
            grace_info = {
                "remaining_s": int(self.grace.remaining_s()),
                "extension_count": self.grace.extension_count,
                "max_extensions": self.grace.max_extensions,
                "thread_id_prefix": (self.grace.thread_id or "")[:8] if self.grace.thread_id else "",
                "model_tag": self.grace.model_tag,
            }

        idle_info: dict | None = None
        # idle-holder wiring: /status idle snapshot reflects the manager-level
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

        # Cache the vram ref ONCE so the null-check + list see the same value.
        # (status_snapshot is a sync def with no await, so single-threaded asyncio
        # already makes it atomic vs the supervisor's _vram_free_mib write — this is
        # belt-and-suspenders against a future await/second-writer ever sneaking in.)
        vram_cache = self._vram_free_mib
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
            # the design client-disconnect eviction observability.
            "evictions": {
                "total_lifetime": self._eviction_count,
                "last_evicted_at": self._last_evicted_at_iso,
            },
            # The classifier (operator request): per-event decision counts + forced-clean-
            # restore total + last decision. PROVES "Turbohaul determines each event"
            # and reused the clean bin (vs CLEAR). event_type ∈ continuation |
            # user-message | compression | sub-agent | guard-skip. clean_bin_id is a
            # thread-HASHED filename (non-reversible; no raw thread_id exposed).
            "kv_classifier": {
                "events": dict(self._kv_classifier_counts),
                "forced_clean_restores": self._kv_classifier_forced,
                # F2 cold-path clean-restore (wave-return) total. Separate
                # from the WARM forced_clean_restores so a working wave-return proves out.
                "wave_return_restores": self._kv_classifier_wave_return,
                "last": self._kv_classifier_last,
            },
            # -PROOF (the operator absolute-proof surface): last per-request
            # structured identity dict (ip/model_tag/session_id/is_*/
            # resolved_class/thread_id). Additive; existing keys unchanged.
            "request_identity": self._last_request_identity,
            # (observability): last N LOAD_VERIFY records — per model
            # (re)spawn + KV restore, the REAL end-state (process_alive, model_resident,
            # kv_actual_n_past vs expected). Additive read-only surface; the emitter +
            # ring live in turbohaul.load_verify_log (module scope, no instance state).
            "load_verify": load_verify_log.get_recent(20),
            # D3 (ENGINE STALLED): set at the death-strike increment in the
            # worker_loop dead-idle sweep (restore-implicated engine death);
            # cleared on the next successful serve (ACTIVE->GRACE). None when
            # healthy. {active, attempt, max, bin, ts}. Additive, display-only —
            # FE renders "ENGINE STALLED — Retrying {attempt}/{max}" off .active.
            "engine_stall": getattr(self, "_engine_stall", None),
            # shadow byte-match self-check (DORMANT observability): counts of
            # match / mismatch / skipped_{no_think,toolcall,empty}. PROVES whether the
            # manager's think-free strip byte-matches the harness's think-stripped resend
            # the assumption shadow-reprefill reuse hinges on. Drives NO decision.
            "shadow_bytematch": dict(self._shadow_bytematch_counts),
            # Option A shadow-reprefill (SAVE-side): counts of shadow bins
            # written (saved) + guard skips (skipped_{toolcall,no_think,empty,
            # no_messages}) + reprefill_post_failed. INERT unless TURBOHAUL_SHADOW_
            # REPREFILL is on; drives NO restore/admission/routing decision.
            "shadow_reprefill": dict(self._shadow_reprefill_counts),
            # step (d) shadow-restore PREFERENCE (RESTORE-side): counts of
            # warm forced-restores that PREFERRED the think-free .shadow bin (preferred)
            # vs fell back to the clean anchor (clean_fallback). INERT unless
            # TURBOHAUL_SHADOW_RESTORE_PREFER; drives NO restore/admission/routing
            # decision (obs-only belt, mirrors shadow_reprefill).
            "shadow_restore_prefer": dict(self._kv_shadow_restore_counts),
            # crit3 (TOOL-tail restore skip): counts of restores skipped
            # because the divergent tail was tool-opaque (hash-invisible), split
            # warm (forced) vs cold (wave-return). Default ON; drives NO other
            # decision (the skip itself is the effect).
            "tooltail_restore_skip": dict(self._kv_tooltail_skip_counts),
            # SHADOW-DIAGNOSIS (INSTRUMENTATION ONLY): one block that lets
            # a forcing-full swap-back name WHY. `saves` (== _shadow_reprefill_counts)
            # carries the per-turn save outcomes {saved, skipped_*, reprefill_post_
            # failed} AND the swap-belt {swap_saved, swap_skip_have_fresher, swap_
            # reprefill_post_failed} = candidate (a) shadow-never-saved. `restores`
            # = which bin the cold restore chose (candidate (c) present-but-not-
            # selected). `byteparity` = candidate (d) selected-but-BYTE-DIVERGES.
            # `kvgc` = candidate (b) evicted-under-cap + the 64GB-floor characterization
            # (over_cap + protected bins/bytes + top protected thread_hashes). Drives
            # NO decision.
            "shadow_diag": {
                "saves": dict(self._shadow_reprefill_counts),
                "restores": dict(self._shadow_diag_counts["restores"]),
                "byteparity": dict(self._shadow_diag_counts["byteparity"]),
                # candidate (b): a `.shadow` bin reaped by the ceiling/over-cap GC
                # while dead-session clean anchors held the floor.
                "evictions": dict(self._shadow_diag_counts["evictions"]),
                "kvgc": dict(self._last_kvgc_snapshot) if self._last_kvgc_snapshot else {},
            },
            # background sweeper that finalizes the
            # state-row for the design evictions (deferred from the hot path per
            # a design note). Sweeper runs every background_sweep_interval_s.
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
            # LiveSlotsPoller (idle default = single idle_generation shape).
            # Counts/rates only — no prompt/response/IP/full-thread-id (the
            # 8-char generation_id is non-reversible). At cap>=2 this is the
            # back-compat ALIAS = the most-recently-active resident's generation
            # (the supervisor mirrors it); the per-resident blocks ride residents[].
            "generation": self.live_generation or idle_generation(),
            # P1e multi-slot observability. ``residents`` = the live
            # per-model sidecars (EMPTY at cap<=1: the legacy singleton is excluded —
            # active/loading/grace above carry that state). ``vram`` = per-GPU free
            # MiB cached off the hot path by the supervisor (null at cap<=1 / probe-
            # down). BOTH are await-free + lock-free (status_snapshot stays sync).
            "residents": self._residents_snapshot(),
            "vram": list(vram_cache) if vram_cache is not None else None,
            "vram_total_mib": list(self._vram_total_mib) if self._vram_total_mib is not None else None,
            # a later phase: persist KV cache SSD usage snapshot for FE Settings cap display
            "persist_kvcache": self._persist_kvcache_snapshot(),
        }

    def _residents_snapshot(self) -> "list[dict]":
        """Await-free per-resident view for /status (cap>=2 multi-slot observability).

        EMPTY at cap<=1 (``_SINGLETON_RESIDENT_KEY`` + DEAD residents excluded — the
        legacy active/loading/grace/idle_hot fields carry the single-sidecar state).
        Snapshots ``_residents`` via ``list()`` so a concurrent dispatcher mutation
        can't 'dict changed size during iteration'; reads each resident's scalars
        directly (sole-writer driver discipline). LOCK-FREE — never acquires
        ``_registry_lock`` (mirrors status_snapshot's hot-path-safe contract). Each
        entry carries the resident's live generation block (from ``live_generations``)
        so the FE shows per-model tok/s without a second round-trip."""
        out: list[dict] = []
        for k, r in list(self._residents.items()):
            if k == _SINGLETON_RESIDENT_KEY or r.state is ResidentState.DEAD:
                continue
            handle = r.handle
            pid = handle.pid if handle is not None else r.booting_pid
            idle_in = None
            if (
                r.state is ResidentState.IDLE_EVICTABLE
                and r.idle_expires_at is not None
            ):
                idle_in = max(0, int(r.idle_expires_at - time.monotonic()))
            out.append({
                "model_tag": r.model_tag,
                "state": r.state.value,
                "port": r.port,
                "pid": pid,
                "spawn_seq": r.spawn_seq,
                "reserved_need_mib": r.reserved_need_mib,
                "parallel": r.parallel,
                "main_gpu": r.main_gpu,
                "split_mode": r.split_mode,
                "inflight": len(r.inflight),
                "idle_expires_in_s": idle_in,
                "generation": self.live_generations.get(r.model_tag),
            })
        return out

    # === Port allocation =====================================================

    def _alloc_port(self) -> int:
        """Lowest free sidecar port in ``[default_port_base, +100)``.

        A port is "held" if any LIVE resident in ``self._residents`` reports it
        via ``Resident.port`` (the Phase-1 authoritative listen port). Phase-0
        keeps ``MAX_PARALLEL_SIDECARS == 1`` and the lone resident's ``port``
        stays at its placeholder (``None``) because the FSM still drives the
        listen port through ``_active_handle.port``, not the registry scalar —
        so with one resident the window is empty of holds and this returns
        ``default_port_base`` verbatim (== the deployed hard-coded 11500). The
        scan only does real work once Phase-1 binds resident ports for a second
        concurrent sidecar; introducing it now means the spawn path is already
        port-registry aware with ZERO behaviour change at max=1.
        """
        base = self.boot.runtime.default_port_base
        held = {
            r.port
            for r in self._residents.values()
            if r.port is not None
        }
        for port in range(base, base + 100):
            if port not in held:
                return port
        # Window exhausted (>=100 live residents on contiguous ports). Phase-0
        # can never reach this (single resident); Phase-1's dispatcher gates
        # spawns on MAX_PARALLEL_SIDECARS long before 100 ports are claimed, so
        # this is a defensive fallback, not a live path. Return base so the
        # caller's spawn attempt fails fast on a real bind collision rather than
        # silently picking an out-of-window port.
        return base

    # === Spawn-sequence (live-monitor generation_id) =========================

    def _active_resident(self) -> "Resident | None":
        """The resident that owns the CURRENT active sidecar.

        Phase-0 (``MAX_PARALLEL_SIDECARS == 1``) has exactly one resident under
        ``_SINGLETON_RESIDENT_KEY``, so the active resident is unambiguously that
        singleton. Phase-1 keys residents by ``model_tag`` and resolves the
        active one from the FSM's current model; until then return the singleton
        (or ``None`` if the registry is somehow empty, so callers degrade-open).
        """
        return self._residents.get(_SINGLETON_RESIDENT_KEY)

    def _bump_spawn_seq(self) -> None:
        """Advance the spawn counter for a new active handle (worker_loop only).

        Single chokepoint: bumps the legacy global ``_spawn_seq`` AND mirrors it
        onto the active resident's ``spawn_seq`` so the registry view never
        diverges from the global. At max=1 the resident value equals the global
        exactly, so the live-monitor generation_id is unchanged. Keeping the
        global write here preserves every existing read site that still reads
        ``_spawn_seq`` directly (zero churn, identical behaviour); the per-read
        migration to ``_active_spawn_seq`` happens incrementally.
        """
        self._spawn_seq += 1
        resident = self._active_resident()
        if resident is not None:
            resident.spawn_seq = self._spawn_seq

    def _active_spawn_seq(self) -> int:
        """The active resident's spawn_seq (live-monitor generation_id input).

        Returns the active resident's mirrored ``spawn_seq``; falls back to the
        legacy global ``_spawn_seq`` if no resident is registered. At max=1 the
        two are identical (``_bump_spawn_seq`` keeps them in lock-step), so the
        unified generation_id computed by the metrics poller and the streaming
        tee is byte-for-byte unchanged from the deployed manager.
        """
        resident = self._active_resident()
        if resident is not None:
            return resident.spawn_seq
        return self._spawn_seq

    def _spawn_seq_for_model(self, model_tag: "str | None") -> int:
        """spawn_seq of the resident actually serving ``model_tag`` (live-monitor
        generation_id input for the streaming text tee).

        At cap>=2 the dispatcher path never bumps the singleton's spawn_seq, so
        ``_active_spawn_seq`` (which resolves the singleton) stays 0 while the
        metrics supervisor hashes the model_tag resident's bumped spawn_seq. The
        text-plane tee must use THIS value so its generation_id matches the
        LiveOutputBuffer key the supervisor publishes as the anchor -- otherwise
        the live pane subscribes to an unfed buffer and shows nothing (
        bug B). Falls back to ``_active_spawn_seq`` at cap<=1, where no
        model_tag-keyed resident exists -> byte-identical generation_id.
        """
        r = self._live_resident_for(model_tag)
        if r is not None:
            return r.spawn_seq
        return self._active_spawn_seq()

    # === P1c state-migration mirror chokepoints ==================
    # Each writes the legacy manager-global scalar (AUTHORITATIVE — zero churn
    # for existing readers) AND mirrors it onto the active resident, exactly
    # like ``_bump_spawn_seq`` does for spawn_seq. At MAX_PARALLEL_SIDECARS == 1
    # the resident value tracks the global 1:1 (byte-identical); P1d's dispatcher
    # flips the resident copy to authoritative + migrates the read sites.

    def _set_active_handle(self, handle: "SidecarHandle | None") -> None:
        self._active_handle = handle
        r = self._active_resident()
        if r is not None:
            r.handle = handle

    def _set_active_slot(self, slot: "Slot | None") -> None:
        self._active_slot = slot
        r = self._active_resident()
        if r is not None:
            r.active_slot = slot

    def _set_idle_holder(
        self,
        handle: "SidecarHandle | None",
        model_tag: str | None,
        expires_at: float | None,
        thread_id: str | None = None,
        admission_ctx_len: int = 0,
        client_meta: dict | None = None,
    ) -> None:
        self._idle_handle = handle
        self._idle_model_tag = model_tag
        self._idle_expires_at = expires_at
        self._idle_thread_id = thread_id
        self._idle_admission_ctx_len = admission_ctx_len
        self._idle_client_meta = client_meta
        r = self._active_resident()
        if r is not None:
            r.idle_handle = handle
            r.idle_model_tag = model_tag
            r.idle_expires_at = expires_at
            r.idle_thread_id = thread_id
            r.idle_admission_ctx_len = admission_ctx_len
            r.idle_client_meta = client_meta

    def _clear_idle_holder(self) -> None:
        self._set_idle_holder(None, None, None, None, 0)

    def _set_idle_expires_at(self, expires_at: float | None) -> None:
        self._idle_expires_at = expires_at
        r = self._active_resident()
        if r is not None:
            r.idle_expires_at = expires_at

    def _set_latest_keep_alive(self, value: int | None) -> None:
        self._latest_keep_alive_s = value
        r = self._active_resident()
        if r is not None:
            r.latest_keep_alive_s = value

    # === Worker loop (full FSM-driven cycle) =================================

    async def worker_loop(self) -> None:
        """Drive the FSM forever: pop → spawn → active → complete → grace → pop → idle.

        Subprocess interactions are dependency-injected via ctor (spawn_fn,
        health_fn, sigterm_fn, vram_fn, complete_fn). Default implementations call the
        real subprocess_mgr functions. Tests inject mocks.
        """
        log.info("worker_loop started")
        # P1d cap>=2 routes to the multi-slot dispatcher; the cap<=1
        # body below stays byte-identical (the 493-forked gate). The cap is the
        # CONFIG KNOB (default 1; flipped to 2 in prod at cutover, and overridden
        # to 2 by the new concurrency tests) — NOT the module constant — so every
        # existing fixture + the deployed runtime stay single-slot.
        if self.runtime.queue.max_parallel_sidecars >= 2:
            return await self._dispatch_loop()
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
                # the design SA Simp #3 inline + FP Bonus-2 fire-and-forget
                # + MOD-α identity-guarded debounce.
                # MOD-α: capture `expires` into a local; only reset _idle_expires_at
                # if it is STILL the same object we observed. Prevents the race where a
                # concurrent reset (request promotion repopulates _idle_expires_at to a
                # fresh T+120 window) would otherwise be wiped by our stale-T0 debounce
                # → teardown fires on a legitimate fresh window → warm holder killed
                # mid-promotion. PL #16848 mandate.
                # RC stuck-handle design note (a review REQUIRED): proactive DEAD-idle sweep.
                # The observed death was BETWEEN requests (llama-server died ~2min
                # after a wave-return while idle-hot); the reactive warm_inherit
                # is_alive gate only fires on the NEXT request. Sweep holder
                # liveness every idle tick so a dead holder is torn down (dead-pid
                # teardown path: skip flush, reap) and the next request cold-spawns
                # clean instead of stranding on active_handle_pid=X alive=False.
                if (
                    self._idle_handle is not None
                    and not self._idle_handle.is_alive()
                ):
                    log.warning(
                        "Idle-hot holder DEAD between requests (model=%s pid=%s) — "
                        "proactive teardown (idle_dead)",
                        self._idle_model_tag,
                        getattr(self._idle_handle, "pid", None),
                    )
                    # C (the operator): attribute the death to the bin this
                    # port last restored; 3 strikes -> auto-quarantine the triplet
                    # (both tiers) so the next load goes FRESH instead of looping.
                    try:
                        _dp = getattr(self._idle_handle, "port", None)
                        _bfn = getattr(self, "_last_restored_bin", {}).get(_dp)
                        if _bfn:
                            _st = getattr(self, "_bin_death_strikes", {})
                            _st[_bfn] = _st.get(_bfn, 0) + 1
                            self._bin_death_strikes = _st
                            log.warning("bin death-strike %d/3: %s", _st[_bfn], _bfn)
                            log.warning(
                                "ENGINE STALLED — restore-implicated death, "
                                "retry %d/3 (bin=%s)", _st[_bfn], _bfn)
                            # D3 FE-visible stall banner.
                            # REPLACE-ONLY — always assign a fresh dict so the
                            # lock-free status_snapshot reader can never see a
                            # half-written record. Lazily created; read via
                            # getattr. Never written from a to_thread body.
                            self._engine_stall = {
                                "active": True,
                                "attempt": _st[_bfn],
                                "max": 3,
                                "bin": _bfn,
                                "ts": datetime.now(timezone.utc).isoformat(),
                            }
                            # review note 4: one blame per restore — clear the
                            # attribution so unrelated later deaths (fresh loads,
                            # OOM) can never strike a long-dead restore's bin.
                            self._last_restored_bin.pop(_dp, None)
                            if _st[_bfn] >= 3:
                                log.error(
                                    "BIN 3-STRIKES — AUTO-QUARANTINE %s (implicated in "
                                    "%d deaths; by design): next load goes FRESH",
                                    _bfn, _st[_bfn])
                                self._spawn_bg(asyncio.to_thread(
                                    self._quarantine_bin_triplet, _bfn))
                    except Exception:
                        log.debug("bin strike accounting failed (best-effort)", exc_info=True)
                    self._set_idle_expires_at(None)
                    _dtask = asyncio.create_task(
                        self._teardown_idle_holder("idle_dead")
                    )
                    _dtask.add_done_callback(
                        lambda t: t.exception() and log.error(
                            "dead-idle teardown failed: %s", t.exception()
                        )
                    )
                expires = self._idle_expires_at
                if (
                    self._idle_handle is not None
                    and expires is not None
                    and time.monotonic() >= expires
                ):
                    if self._idle_expires_at is expires:  # MOD-α identity check
                        self._set_idle_expires_at(None)
                        # FP Bonus-2: fire-and-forget — don't block worker_loop on
                        # the 5s SIGTERM grace + wait4 of the llama-server child.
                        # a tracked issue: add done callback to log failures instead of silently swallowing.
                        _task = asyncio.create_task(
                            self._teardown_idle_holder("idle_expired")
                        )
                        _task.add_done_callback(
                            lambda t: t.exception() and log.error(
                                "idle teardown failed: %s", t.exception()
                            )
                        )
                await asyncio.sleep(0.05)
                continue

            # the design client-disconnect eviction handling.
            if slot.is_evicted:
                self._fail_completion_future(
                    slot,
                    SlotEvictedError(
                        f"slot {slot.slot_id} evicted: client disconnect"
                    ),
                )
                # a design note: audit-emit via the the design pool path; NO sync
                # state_db_session(mark_slot_ended) on the hot path —
                # SQLite fsync 1-3s stalls would bypass the pool entirely.
                # State-row finalization defers to terminal-park
                # background sweeper RC stub.
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
                # SA Simp #3 (inline mirror) + MOD-α identity guard — same
                # idle-tick block on the eviction branch so consecutive
                # evictions don't starve idle expiry.
                expires = self._idle_expires_at
                if (
                    self._idle_handle is not None
                    and expires is not None
                    and time.monotonic() >= expires
                ):
                    if self._idle_expires_at is expires:
                        self._set_idle_expires_at(None)
                        # a tracked issue: add done callback to log failures instead of silently swallowing.
                        _task2 = asyncio.create_task(
                            self._teardown_idle_holder("idle_expired")
                        )
                        _task2.add_done_callback(
                            lambda t: t.exception() and log.error(
                                "idle teardown failed: %s", t.exception()
                            )
                        )
                continue

            try:
                await self._process_slot(slot)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("slot %s processing failed", slot.slot_id)
                self._fail_completion_future(slot, e)
                # C2 fix (design review): teardown active sidecar BEFORE force_cold
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

    # === P1d multi-slot dispatcher (cap>=2 path) ================
    # worker_loop branches into _dispatch_loop when max_parallel_sidecars>=2. The
    # cap<=1 path above is UNTOUCHED (byte-identical, the 493-forked gate). The
    # dispatcher is the SOLE writer of the _residents dict + each resident's
    # ``state``, ALWAYS under _registry_lock. Each resident is driven by its own
    # long-lived _drive_resident task (sole writer of that r.* lock-free in ACTIVE).
    # The request-route (submit/chat_completion) NEVER takes _registry_lock.

    def _model_residents(self) -> "list[Resident]":
        """Live model_tag-keyed residents (EXCLUDES the legacy singleton + DEAD).
        Await-free ``list()`` snapshot so a concurrent mutation can't error."""
        return [
            r
            for k, r in list(self._residents.items())
            if k != _SINGLETON_RESIDENT_KEY and r.state is not ResidentState.DEAD
        ]

    def _dispatch_warm_hint(self) -> "str | None":
        """pop_next affinity hint at cap>=2: the most-recently-active live
        resident's model_tag. HINT only — stale value falls back to FIFO, never
        starves a routable follower."""
        live = self._model_residents()
        if not live:
            return None
        return max(live, key=lambda r: r.last_active_monotonic).model_tag

    async def _dispatch_loop(self) -> None:
        """The cap>=2 dispatcher: pop -> route-or-reserve -> loop. Never blocks on
        an ACTIVE wait (the per-resident drivers own that)."""
        log.info(
            "dispatcher started (max_parallel_sidecars=%d)",
            self.runtime.queue.max_parallel_sidecars,
        )
        while not self._stop_event.is_set():
            slot = await self.queue.pop_next(
                warm_model_tag=self._dispatch_warm_hint()
            )
            if slot is None:
                await asyncio.sleep(0.05)
                continue
            if slot.is_evicted:
                self._handle_evicted_slot(slot)
                continue
            try:
                await self._route_or_reserve(slot)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("dispatch of slot %s failed", slot.slot_id)
                self._fail_completion_future(
                    slot, RuntimeError("dispatch routing failed")
                )
        log.info("dispatcher exited")

    def _handle_evicted_slot(self, slot: Slot) -> None:
        """Client-disconnect eviction (mirrors worker_loop's is_evicted branch).
        The per-resident idle tick is owned by the drivers, so it is NOT here."""
        self._fail_completion_future(
            slot,
            SlotEvictedError(f"slot {slot.slot_id} evicted: client disconnect"),
        )
        self._spawn_bg(self._audit_evicted_slot(slot))
        self._eviction_count += 1
        self._last_evicted_at_iso = datetime.now(timezone.utc).isoformat()  # noqa: UP017

    async def _audit_evicted_slot(self, slot: Slot) -> None:
        try:
            await self._audit_event_only_async(
                slot.slot_id,
                "slot_evicted",
                {
                    "reason": "client_disconnect",
                    "time_in_queue_s": time.monotonic() - slot.created_at,
                },
            )
            # telemetry — client disconnect
            try:
                elapsed = time.monotonic() - slot.created_at
                self._telemetry.on_client_disconnect(slot, "client_disconnect", elapsed)
            except Exception:
                pass
        except Exception:
            log.exception("slot_evicted audit emit failed (best-effort)")

    def _live_resident_for(self, model_tag: "str | None") -> "Resident | None":
        """The live (non-DEAD) model_tag-keyed resident, or None."""
        if model_tag is None:
            return None
        r = self._residents.get(model_tag)
        if r is None or r.state is ResidentState.DEAD:
            return None
        return r

    def _lru_idle_evictable(self) -> "Resident | None":
        """Least-recently-active resident that is IDLE_EVICTABLE with NO active
        slot and NO inflight riders. Busy residents are NEVER evictable."""
        cands = [
            r
            for r in self._model_residents()
            if r.state is ResidentState.IDLE_EVICTABLE
            and r.active_slot is None
            and not r.inflight
        ]
        if not cands:
            return None
        return min(cands, key=lambda r: r.last_active_monotonic)

    def _spawn_bg(self, coro) -> "asyncio.Task":
        """Fire-and-forget a load-bearing background coroutine while holding a STRONG
        reference (N2). asyncio keeps only a weak ref to a bare ``create_task``
        result, so an unreferenced teardown/requeue/reap task can be GC-cancelled
        mid-flight. The task is parked in ``self._bg_tasks`` and self-removes on
        completion."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    def _defer_unroutable(self, slot: Slot) -> None:
        """MISS+full with no idle victim: bounded re-queue. A per-slot defer
        counter caps the spins so an unroutable slot can't busy-loop forever; on
        cap exhaustion the caller's future fails (503-equivalent) instead of
        starving. enqueue_head preserves FIFO for when a resident frees."""
        n = getattr(slot, "_dispatch_defer_count", 0) + 1
        slot._dispatch_defer_count = n
        if n > _MAX_DISPATCH_DEFERS:
            self._fail_completion_future(
                slot,
                RuntimeError(
                    f"no capacity for model {slot.model_tag!r} after "
                    f"{_MAX_DISPATCH_DEFERS} defers (all residents busy)"
                ),
            )
            return
        self._spawn_bg(self._requeue_after_backoff(slot))

    async def _requeue_after_backoff(self, slot: Slot) -> None:
        """Re-enqueue an unroutable slot at the HEAD after a short backoff so the
        dispatcher can't burn its whole defer budget in a sub-second busy-loop
        while residents are transiently busy. Detached, so it MUST fail the slot's
        future on ANY failure (cancel during the sleep at shutdown, or enqueue_head
        raising on a closing queue) rather than silently dropping it — otherwise the
        client hangs until upstream timeout (B4). Mirrors _route_or_reserve's
        except-fails-the-future safety net."""
        try:
            await asyncio.sleep(0.05)
            await self.queue.enqueue_head(slot)
        except asyncio.CancelledError:
            self._fail_completion_future(
                slot, RuntimeError("requeue cancelled during shutdown")
            )
            raise
        except Exception as e:  # noqa: BLE001 -- detached path must not swallow
            log.exception("requeue-after-backoff failed for slot %s", slot.slot_id)
            self._fail_completion_future(slot, e)

    async def _requeue_slots_or_fail(self, slots: "list[Slot]") -> None:
        """Re-enqueue a BATCH of drained inbox slots at the head IN ORDER. ONE task
        awaiting sequentially preserves FIFO among them — fanning out one _spawn_bg
        per slot would let N tasks race on the queue lock and scramble the order
        (a fairness bug). Every slot that can't be re-enqueued — queue closing at
        shutdown (enqueue_head raises) or cancel — has its future FAILED, never
        dropped (the B3-drain analogue of the B4 _requeue_after_backoff safety net).
        On cancel, the remaining batch is failed too so nothing hangs."""
        for i, slot in enumerate(slots):
            try:
                await self.queue.enqueue_head(slot)
            except asyncio.CancelledError:
                for s in slots[i:]:
                    self._fail_completion_future(
                        s, RuntimeError("inbox-drain requeue cancelled during shutdown")
                    )
                raise
            except Exception as e:  # noqa: BLE001 -- detached path must not swallow
                log.exception("inbox-drain requeue failed for slot %s", slot.slot_id)
                self._fail_completion_future(slot, e)

    async def _route_or_reserve(self, slot: Slot) -> None:
        """ONE atomic _registry_lock critical section: HIT route / MISS+capacity
        reserve / MISS+full LRU-evict-then-reserve / else bounded-defer. The
        IDLE_EVICTABLE->ACTIVE flip and the inbox.put are CO-LOCATED under the lock
        (closes the lost-slot race); the 5s nvidia-smi is scoped to the reserve
        branch only (PL D2)."""
        async with self._registry_lock:
            r = self._live_resident_for(slot.model_tag)
            if r is not None:
                # HIT: reclaim from idle + hand to the driver, all under the lock.
                if r.state is ResidentState.IDLE_EVICTABLE:
                    _transition_resident_state(r.state, ResidentState.ACTIVE)
                    r.state = ResidentState.ACTIVE
                r.last_active_monotonic = time.monotonic()
                if r.inbox is not None:
                    r.inbox.put_nowait(slot)
                return
            cap = self.runtime.queue.max_parallel_sidecars
            if len(self._model_residents()) >= cap:
                victim = self._lru_idle_evictable()
                if victim is None:
                    self._defer_unroutable(slot)
                    return
                self._begin_evict_locked(victim)
            await self._reserve_and_start_locked(slot)

    async def _reserve_and_start_locked(self, slot: Slot) -> None:
        """CALLER HOLDS _registry_lock. Read the model footprint, run the
        cross-resident VRAM gate, alloc a port, insert a RESERVED_LOADING
        placeholder (reserving its budget against concurrent reserves), and start
        its driver. The slow spawn+health happen OUTSIDE the lock inside the driver
        (the placeholder already reserves the budget)."""
        need, parallel, main_gpu, split_mode, sleep_idle_s = self._read_model_footprint(
            slot.model_tag
        )
        if not self._vram_admits_locked(need, parallel, main_gpu, split_mode):
            # Refuse (cross-resident over-commit) — mirror the safety-gate refusal.
            log.warning(
                "cross-resident VRAM gate refused spawn for %s "
                "(need=%dMiB parallel=%d gpu=%d split=%s)",
                slot.model_tag, need, parallel, main_gpu, split_mode,
            )
            self._fail_completion_future(
                slot,
                RuntimeError(
                    f"cross-resident VRAM gate refused {slot.model_tag!r}: "
                    f"need {need} MiB would over-commit GPU {main_gpu}"
                ),
            )
            return
        port = self._alloc_port()
        r = Resident(
            model_tag=slot.model_tag,
            port=port,
            state=ResidentState.RESERVED_LOADING,
            reserved_need_mib=need,
            parallel=parallel,
            main_gpu=main_gpu,
            split_mode=split_mode,
            sleep_idle_seconds=sleep_idle_s,
            last_active_monotonic=time.monotonic(),
            grace=GraceTimer(
                grace_seconds=self.runtime.queue.grace_seconds,
                max_extensions=self.runtime.queue.max_grace_extensions,
            ),
            idle=IdleHotTimer(
                idle_seconds=self.runtime.queue.idle_hot_load_seconds
            ),
            inbox=asyncio.Queue(),
        )
        r.inbox.put_nowait(slot)  # the slot that triggered the reservation
        self._residents[slot.model_tag] = r
        r.driver_task = asyncio.create_task(self._drive_resident(r))
        r.driver_task.add_done_callback(
            lambda t, rr=r: self._on_driver_done(rr, t)
        )

    def _read_model_footprint(
        self, model_tag: "str | None"
    ) -> "tuple[int, int, int, str, int]":
        """(reserved_need_mib, parallel, main_gpu, split_mode, sleep_idle_seconds)
        from the manifest. Sync file read; the dispatcher calls it under the lock
        so the placeholder's reserved budget is exact. Missing manifest ->
        (0,1,0,'layer',0) = degrade-open for the footprint (the per-spawn
        all_safety_gates still runs in the driver). sleep_idle_seconds=0 means
        'use global default' — the driver falls back to
        runtime.queue.idle_hot_load_seconds."""
        try:
            m = read_manifest(self.boot.storage.manifests_path, model_tag)
        except FileNotFoundError:
            return 0, 1, 0, "layer", 0
        flags = m.llama_server_flags or {}
        gguf_mib = int((m.gguf_size_bytes or 0) // (1024 * 1024))
        ctx = int(flags.get("ctx_size") or m.context_size or 0)
        kv_quant = flags.get("cache_type_k") or "f16"
        kv_quant_v = flags.get("cache_type_v") or kv_quant
        kv_mib = estimate_kv_cache_mib(ctx, m.gguf_size_bytes or 0, kv_quant, kv_quant_v)
        parallel = max(1, int(flags.get("parallel", 1) or 1))
        # par_extra: the marginal per-slot compute floor for parallel>1, ON TOP of
        # the model footprint (the red-team's fix: reserve the FULL body+KV, not
        # just the compute floor).
        par_extra = (parallel - 1) * PER_SLOT_COMPUTE_FLOOR_MIB
        expected_vram_mib = int((m.expected_vram_bytes or 0) // (1024 * 1024))
        # cpu_moe / n_cpu_moe offload expert weights to HOST RAM, so the gguf+kv
        # heuristic (which counts EVERY weight as GPU-resident) grossly over-reserves an
        # expert-offload model and wrong-refuses a co-resident that actually fits
        # (live-E2E 2026-06-25: 35b n-cpu-moe @500K reserved 29.9GiB vs a proven
        # 19.4GiB). For those configs the operator's MEASURED expected_vram_bytes is the
        # only accurate footprint -> trust it. Normal models keep the conservative
        # max(declared, gguf+kv). The driver's per-spawn all_safety_gates re-checks LIVE
        # free VRAM before the sidecar binds either way.
        cpu_moe = bool(
            flags.get("cpu_moe") or int(flags.get("n_cpu_moe", 0) or 0) > 0
        )
        if cpu_moe and expected_vram_mib > 0:
            need = expected_vram_mib + par_extra
        else:
            need = max(expected_vram_mib, gguf_mib + kv_mib) + par_extra
        main_gpu = int(flags.get("main_gpu", 0) or 0)
        split_mode = str(flags.get("split_mode", "layer") or "layer")
        sleep_idle_s = int(flags.get("sleep_idle_seconds") or 0)
        return need, parallel, main_gpu, split_mode, sleep_idle_s

    def _vram_admits_locked(
        self, need: int, parallel: int, main_gpu: int, split_mode: str
    ) -> bool:
        """CALLER HOLDS _registry_lock. Cross-resident over-commit gate.

        N1 (interim): co-residence is supported ONLY for single-GPU-pinned
        (``split_mode='none'``) models on DISTINCT cards. A layer/row/tensor-split
        sibling spans every visible GPU, so no "free distinct card" can be
        guaranteed AND the aggregate-budget reserve math would double-count an
        already-loaded sibling (B1). Until per-card layer-split accounting lands,
        refuse-blind any co-residence that isn't tensor-isolated on distinct cards.
        The FIRST resident (no sibling) admits regardless of split_mode — it keeps
        the legacy degrade-open(parallel:1)/refuse-blind(parallel>1) doctrine via
        the per-spawn all_safety_gates in the driver.

        B1: for the admitted (split=none/distinct-card) shape the reserve only
        charges siblings whose VRAM is NOT YET reflected in the live nvidia-smi
        probe — i.e. those still RESERVED_LOADING (spawned, weights not loaded) on
        THIS card. An ACTIVE/GRACE/IDLE_EVICTABLE sibling is already loaded =>
        already absent from the free reading; also subtracting its reserved_need_mib
        double-charges it and wrong-refuses the steady state (two warm models is the
        whole point of the feature). With N1 every admitted sibling is on a distinct
        card so this normally contributes 0; the same-card term is defensive against
        a future relaxation."""
        siblings = self._model_residents()
        new_split = (split_mode or "layer").lower()
        if siblings:
            # N1 refuse-blind: incoming must be tensor-isolated...
            if new_split != "none":
                return False
            for r in siblings:
                # ...and every existing sibling must be tensor-isolated on a
                # DIFFERENT card (else its weights occupy this card too).
                if (r.split_mode or "layer").lower() != "none":
                    return False
                if r.main_gpu == main_gpu:
                    return False
        free_fit, _min_card, _n = _vram_budget(split_mode, main_gpu)
        if free_fit is None:
            # nvidia-smi unreadable.
            if siblings:
                return False  # co-residence without a probe = refuse-blind
            return True  # lone spawn -> driver's all_safety_gates owns the doctrine
        # Reserve ONLY still-booting (not-yet-in-probe) siblings on THIS card (B1).
        reserve = 0
        for r in siblings:
            if r.state is ResidentState.RESERVED_LOADING and r.main_gpu == main_gpu:
                reserve += r.reserved_need_mib
        return (free_fit - reserve) >= need

    def _begin_evict_locked(self, r: "Resident") -> None:
        """CALLER HOLDS _registry_lock. LRU-evict (or driver-death reap): drain any
        inbox slots back to the queue, mark DEAD, deregister from _residents (so
        capacity frees + dispatcher stops routing), then teardown the captured
        handle on a DETACHED task (keyed off the captured ref, not the dict entry).

        B3: the inbox drain happens BEFORE the DEAD early-return — a driver that
        already died via its own finally set DEAD without draining (or only partly),
        and this reaper path is the only other drain site, so an already-DEAD
        resident must still surrender its queued slots. The queue hand-off is
        one-shot, so re-draining an already-empty inbox is a harmless no-op."""
        # Drain unstarted inbox slots back to the main queue (not lost) — EVERY call.
        # This method is sync (under _registry_lock) so it can't await; collect the
        # slots IN ORDER and hand them to ONE ordered bg re-enqueue task (not one
        # task per slot, which would race the queue lock and scramble FIFO).
        if r.inbox is not None and not r.inbox.empty():
            drained: list[Slot] = []
            while not r.inbox.empty():
                try:
                    drained.append(r.inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if drained:
                self._spawn_bg(self._requeue_slots_or_fail(drained))
        if r.state is ResidentState.DEAD:
            return
        _transition_resident_state(r.state, ResidentState.DEAD)
        r.state = ResidentState.DEAD
        if self._residents.get(r.model_tag) is r:
            del self._residents[r.model_tag]
        self._spawn_bg(self._evict_teardown(r))

    async def _evict_teardown(self, r: "Resident") -> None:
        """Detached: claim torn_down (exactly-once vs the driver finally) then
        teardown r's live handle off the hot path. B2: if r is still RESERVED_LOADING
        (sidecar spawned but handle not yet published) the handle is None and
        ``booting_pid`` is the ONLY reference to the live process — reap by pid so
        an LRU/death teardown of a booting resident can't leak it.

        rework: trigger KV cache save BEFORE SIGTERM with the resident's
        idle_thread_id (mirrors _teardown_idle_holder for cap<=1). The C++ engine
        checks for thread_id during save — empty thread_id returns do_it=False.
        """
        async with self._registry_lock:
            if r.torn_down:
                return
            r.torn_down = True
            handle = r.handle
            model_tag = r.model_tag
            idle_thread_id = r.idle_thread_id
            r.handle = None
            booting_pid = r.booting_pid
            r.booting_pid = None
        # M5 (WIN 2) SWAP-CLEAR (cap>=2): a resident sidecar is being
        # torn down (LRU-evict / driver-death). Invalidate the completion-cache so
        # a cached answer can never survive a resident swap. Cleared OUTSIDE the
        # registry lock (the cache is not registry-lock-protected — same no-lock,
        # single-loop discipline as the submit_and_wait / _process_slot accesses).
        self._completion_cache_clear("evict_teardown")
        # rework: trigger KV cache save BEFORE SIGTERM.
        # The C++ engine checks for thread_id during save — empty thread_id
        # returns do_it=False. Now we have the thread_id from idle entry,
        # so the save can succeed and the cache will be fresh on resume.
        # use _save_slot_kv instead of bolt-on POST.
        if handle is not None and handle.is_alive() and idle_thread_id:
            try:
                await self._save_slot_kv(
                    handle.port,
                    model_tag,
                    r.active_slot,
                    thread_id_override=idle_thread_id,
                    admission_ctx_len_override=r.idle_admission_ctx_len,
                    # a later phase a design note: same raw-keying bug as the singleton seam —
                    # the _idle_thread_id fallback compares against the SINGLETON
                    # holder's field, never this resident's; without the explicit
                    # meta the save keys on raw thread_id (duplicate bin) and
                    # T-GUARD has no prior meta to calibrate against.
                    client_meta_override=r.idle_client_meta,
                )
                log.info(
                    "resident KV cache saved before teardown "
                    "(thread_id=%s, model_tag=%s)",
                    idle_thread_id[:8] if idle_thread_id else "none",
                    model_tag,
                )
            except Exception:
                log.warning(
                    "resident KV cache save failed (best-effort): model_tag=%r",
                    model_tag,
                    exc_info=True,
                )
        if handle is not None and handle.is_alive():
            await self._reap_resident_handle(handle)
        elif booting_pid is not None:
            await self._reap_booting_pid(booting_pid)

    async def _reap_booting_pid(self, pid: int) -> None:
        """SIGTERM-then-REAP a sidecar that spawned but whose handle was never
        published (driver cancelled/failed mid ``_wait_healthy``). ``booting_pid`` is
        the only reference to that process — and ``_live_handle_pids`` actively
        PROTECTS it from the orphan reaper — so without this it leaks VRAM+PID forever
        (B2). The sidecar is OUR child (same-process spawn at cap>=2), so we MUST
        ``waitpid`` it or it lingers as a ZOMBIE/defunct entry that keeps the PID
        allocated (P1e fast-follow #1). Bounded: WNOHANG-poll for a grace window,
        escalate to SIGKILL, final blocking reap. Best-effort, off the event loop (the
        sleeps run in the worker thread, never blocking the loop).

        PID-RECYCLE GUARD (PL pre-cutover polish #4): a child that exits becomes a
        zombie holding its PID until reaped — but a COMPETING reaper (the lost
        ``SidecarHandle``'s ``Popen`` being GC-reaped by CPython after the cancel
        unwinds its frame) can reap it first, freeing the PID for the OS to recycle to
        an UNRELATED process. So before EVERY signal we re-confirm pid is still our
        ALIVE child via ``waitpid(WNOHANG)``: ``ChildProcessError`` (ECHILD) => not
        ours / already reaped => STOP (never signal a recycled pid); ``(pid, _)`` =>
        ours, just exited => reaped here => done; ``(0, 0)`` => ours, alive => safe to
        signal. This narrows the recycle window to the microseconds between the check
        and the kill (the standard best-effort bound for raw-pid reaping)."""
        def _own_live_child() -> bool:
            """True iff pid is our ALIVE child (safe to signal). False if it is gone,
            already reaped (zombie collected here), or no longer ours (ECHILD)."""
            try:
                wpid, _ = os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                return False  # not ours / already reaped -> do NOT signal a recycled pid
            return wpid == 0  # 0 = still running; pid = exited+reaped just now (done)

        def _term_and_reap() -> None:
            if not _own_live_child():
                return
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            for _ in range(30):  # ~3s grace for SIGTERM, re-checking ownership each iter
                if not _own_live_child():
                    return  # exited (reaped) or no longer ours
                time.sleep(0.1)
            if not _own_live_child():
                return
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            with contextlib.suppress(ChildProcessError, ProcessLookupError, OSError):
                os.waitpid(pid, 0)  # final blocking reap of the SIGKILL'd child
        try:
            await asyncio.to_thread(_term_and_reap)
        except Exception:
            log.exception("booting_pid reap failed (best-effort) pid=%d", pid)

    async def _reap_resident_handle(self, handle) -> None:
        """Drained-SIGTERM a resident's handle via the DI _sigterm seam (consistent
        with the legacy _teardown; async + non-blocking + mockable). Best-effort.
        NOTE: No KV save here — this is a safety-net reap without slot identity.
        The proper save happens in _teardown_idle_holder / _teardown WITH the slot."""
        try:
            await self._sigterm(
                handle,
                drained_window_s=float(
                    self.runtime.queue.drained_sigterm_window_active_s
                ),
                is_active=False,
                cold_window_s=float(
                    self.runtime.queue.drained_sigterm_window_cold_s
                ),
            )
        except Exception:
            log.exception("resident handle reap failed (best-effort)")

    def _on_driver_done(self, r: "Resident", task: "asyncio.Task") -> None:
        """done_callback (SYNC — cannot await): schedule the supervisor reaper on
        EVERY driver exit, including CANCELLED. B2: a driver cancelled during
        RESERVED_LOADING leaves a live sidecar (handle still None, ``booting_pid``
        set) — bailing on ``task.cancelled()`` meant the reaper never ran and the
        process leaked. The driver finally already reaps + claims torn_down, so the
        reaper is exactly-once-safe (it no-ops when torn_down is already set) and
        only ADDS the failed-future / booting_pid safety net on the cancel path.
        Normal completion (idle-evict self-teardown) likewise no-ops."""
        cancelled = task.cancelled()
        exc = None if cancelled else task.exception()
        if cancelled:
            log.warning("driver for %s cancelled — scheduling reap", r.model_tag)
        elif exc is not None:
            log.warning("driver for %s died: %r — scheduling reap", r.model_tag, exc)
        self._spawn_bg(self._reap_dead_resident(r))

    async def _reap_dead_resident(self, r: "Resident") -> None:
        """Supervisor reaper: fail every pending future the dead driver owned
        (active_slot + ALL inflight riders) + unblock streaming routes, then
        mark DEAD + deregister + teardown (exactly-once via torn_down)."""
        # Always fail the pending futures (the driver abandoned them on death);
        # teardown stays exactly-once via the torn_down claim in _begin_evict_locked.
        async with self._registry_lock:
            pend = [r.active_slot, *list(r.inflight)]
        for s in pend:
            if s is None:
                continue
            if s.completion_future is not None and not s.completion_future.done():
                self._fail_completion_future(
                    s, RuntimeError(f"resident {r.model_tag!r} driver died")
                )
            ev = getattr(s, "stream_done_event", None)
            if ev is not None and not ev.is_set():
                ev.set()
        async with self._registry_lock:
            self._begin_evict_locked(r)

    def _idle_window_seconds(self, keep_alive_s: "int | None", default: int) -> int:
        """Per-resident idle-hot window from the latest keep_alive intent (mirrors
        the cap<=1 grace->idle math: None->default, <0->KEEP_ALIVE_MAX_S cap,
        else min(keep,cap); 0 disables idle)."""
        if keep_alive_s is None:
            return default
        if keep_alive_s < 0:
            return KEEP_ALIVE_MAX_S
        return min(keep_alive_s, KEEP_ALIVE_MAX_S)

    async def _spawn_for_resident(
        self, r: "Resident", slot: Slot
    ) -> "SidecarHandle | None":
        """Spawn r's sidecar OUTSIDE _registry_lock (the placeholder already
        reserved the budget). Per-spawn host safety gate (per-model), spawn,
        capture booting_pid under the lock (closes the reaper window), health-wait,
        then publish r.handle + state=ACTIVE under the lock. On failure: fail the
        slot future and return None (the finally/supervisor reaps the resident)."""
        argv: list[str] = []
        is_mlx = False
        try:
            manifest = read_manifest(self.boot.storage.manifests_path, slot.model_tag)
            is_mlx = manifest.is_mlx()
            if is_mlx:
                # MLX: no GGUF blob, no slot-save KV. Build argv from the MLX
                # closed allowlist (validated again at spawn time in mlx_spawn).
                argv = mlx_flags_to_argv(manifest.mlx_server_flags)
            else:
                argv = flags_to_argv(manifest.llama_server_flags)
                gguf_path = (
                    self.boot.storage.blob_store_path
                    / "sha256"
                    / manifest.gguf_blob_sha256[:2]
                    / manifest.gguf_blob_sha256
                )
        except FileNotFoundError:
            gguf_path = self.boot.storage.blob_store_path / "missing.gguf"
        # Per-model host safety gate (RAM/IO/load + the per-spawn VRAM/KV gate). The
        # CROSS-resident over-commit gate already ran under the lock at reserve.
        # For MLX, expected_vram_bytes is 0 (unified memory), so the VRAM pre-check
        # is a no-op and RAM/IO gates still apply.
        if self.runtime.queue.safety_enabled:
            gate_ok = await self._run_spawn_safety_gate(slot)
            if not gate_ok:
                self._fail_completion_future(
                    slot, RuntimeError("safety gates refused spawn")
                )
                return None
        if is_mlx:
            handle = mlx_spawn(
                r.port,
                slot.model_tag,
                manifest.model_repo,
                manifest.model_path,
                manifest.mlx_server_flags,
                python_binary=self.boot.runtime.mlx_python_binary,
            )
        else:
            handle = self._spawn(
                self.boot.runtime.llama_server_binary,
                gguf_path,
                r.port,
                slot.model_tag,
                argv,
                binary_fd=self._binary_fd,
            )
        async with self._registry_lock:
            r.booting_pid = handle.pid  # in the reaper union before handle is set
        slot.port = handle.port
        slot.pid = handle.pid
        healthy = await self._wait_healthy(
            r.port, self.runtime.queue.loading_health_timeout_s,
            is_alive=handle.is_alive,
        )
        if not healthy:
            self._fail_completion_future(
                slot, RuntimeError("loading-fail-health-timeout")
            )
            # Reap the spawned-but-unhealthy sidecar + clear booting_pid, else it
            # lingers in _live_handle_pids (so the orphan reaper SKIPS it) =
            # permanent PID+VRAM leak. Mirrors the cap<=1 path's _teardown here.
            async with self._registry_lock:
                r.booting_pid = None
            if handle.is_alive():
                await self._reap_resident_handle(handle)
            return None
        # Best-effort KV restore after a healthy (re)spawn so the next same-thread
        # request reuses the slot KV (prefix-match) instead of re-prefilling.
        # MLX has no slot-save KV cache (unified memory, no /slot-save-path), so
        # skip it there.
        if not is_mlx:
            await self._restore_slot_kv(r.port, r.model_tag, slot)
        async with self._registry_lock:
            r.handle = handle
            r.booting_pid = None
            _transition_resident_state(r.state, ResidentState.ACTIVE)
            r.state = ResidentState.ACTIVE
            r.spawn_seq += 1  # live-monitor: new active handle on this resident
        return handle

    async def _run_spawn_safety_gate(self, slot: Slot) -> bool:
        """Run all_safety_gates for slot.model_tag exactly as the cap<=1 path does
        (manifest-derived params). Returns True on all-pass, False on any refusal
        (logged + audited)."""
        mv = mc = mg = 0
        mq = "f16"
        mqv = "f16"
        mnk = False
        mp = 1
        msm = "layer"
        mmg = 0
        mcm = False
        try:
            m = read_manifest(self.boot.storage.manifests_path, slot.model_tag)
            mv = m.expected_vram_bytes or 0
            mg = m.gguf_size_bytes or 0
            mc = m.llama_server_flags.get("ctx_size") or m.context_size or 0
            mq = m.llama_server_flags.get("cache_type_k") or "f16"
            mqv = m.llama_server_flags.get("cache_type_v") or mq
            mnk = bool(m.llama_server_flags.get("no_kv_offload", False))
            mp = int(m.llama_server_flags.get("parallel", 1) or 1)
            msm = str(m.llama_server_flags.get("split_mode", "layer") or "layer")
            mmg = int(m.llama_server_flags.get("main_gpu", 0) or 0)
            mcm = bool(
                m.llama_server_flags.get("cpu_moe")
                or int(m.llama_server_flags.get("n_cpu_moe", 0) or 0) > 0
            )
        except FileNotFoundError:
            mv = 0
        gates = await asyncio.to_thread(all_safety_gates,
            min_free_ram_mib=self.runtime.queue.safety_min_free_ram_mib,
            min_free_vram_mib=self.runtime.queue.safety_min_free_vram_mib,
            max_load_per_core=self.runtime.queue.safety_max_load_per_core,
            max_iowait_percent=self.runtime.queue.safety_max_iowait_percent,
            manifest_expected_vram_bytes=mv,
            iowait_sample_window_s=self.runtime.queue.safety_iowait_sample_window_s,
            ctx_size=mc, gguf_size_bytes=mg, kv_cache_quant=mq,
            kv_cache_quant_v=mqv,
            no_kv_offload=mnk, parallel=mp, split_mode=msm, main_gpu=mmg,
            cpu_moe_offload=mcm,
        )
        failed = [g for g in gates if not g.ok]
        if failed:
            log.warning(
                "safety gates refused spawn for %s: %s",
                slot.slot_id, "; ".join(f"{g.name}: {g.detail}" for g in failed),
            )
            return False
        return True

    async def _drive_resident(self, r: "Resident") -> None:
        """Long-lived per-resident driver (cap>=2). Spawns r's sidecar once, then
        serves slots from r.inbox through ACTIVE->GRACE, parking in IDLE_EVICTABLE
        between requests until idle expiry / eviction / death. SOLE writer of r.*
        lock-free in ACTIVE; takes _registry_lock only for the idle<->active<->dead
        transitions the dispatcher also touches."""
        handle = None
        slot = None  # bound before the first await so the finally can fail it
        try:
            slot = await r.inbox.get()  # the reservation's first slot
            handle = await self._spawn_for_resident(r, slot)
            if handle is None:
                return
            # Per-model idle timeout: read from the manifest's sleep_idle_seconds
            # (threaded through Resident.sleep_idle_seconds). -1 = pin/keep-warm
            # (never idle-unload), 0 = fall back to global default, N>0 = that
            # model's idle timeout in seconds. This is what evicts the 35b sub-agent
            # model too early (Bug C: global 120s was used for everything).
            if r.sleep_idle_seconds == -1:
                per_model_idle = KEEP_ALIVE_MAX_S  # pin-warm, never idle-unload
            elif r.sleep_idle_seconds > 0:
                per_model_idle = r.sleep_idle_seconds
            else:
                per_model_idle = self.runtime.queue.idle_hot_load_seconds
            while not self._stop_event.is_set():
                async with self._registry_lock:
                    if r.state is ResidentState.DEAD:
                        break
                    _transition_resident_state(r.state, ResidentState.ACTIVE)
                    r.state = ResidentState.ACTIVE
                    r.active_slot = slot
                    r.last_active_monotonic = time.monotonic()
                await self._serve_on_resident(r, slot, handle)
                idle_window = self._idle_window_seconds(
                    r.latest_keep_alive_s, per_model_idle
                )
                async with self._registry_lock:
                    if r.state is ResidentState.DEAD:
                        break
                    r.active_slot = None
                    r.last_active_monotonic = time.monotonic()
                    if idle_window <= 0:
                        # rework: capture thread_id for KV save on immediate eviction
                        r.idle_thread_id = slot.thread_id if slot else None
                        self._begin_evict_locked(r)
                        break
                    _transition_resident_state(r.state, ResidentState.IDLE_EVICTABLE)
                    r.state = ResidentState.IDLE_EVICTABLE
                    r.idle_expires_at = time.monotonic() + idle_window
                    # rework: capture thread_id for KV cache save on idle teardown
                    r.idle_thread_id = slot.thread_id if slot else None
                try:
                    slot = await asyncio.wait_for(
                        r.inbox.get(), timeout=idle_window
                    )
                except TimeoutError:
                    async with self._registry_lock:
                        if (
                            r.state is ResidentState.IDLE_EVICTABLE
                            and r.inbox.empty()
                        ):
                            self._begin_evict_locked(r)
                            break
                        if not r.inbox.empty():
                            slot = r.inbox.get_nowait()
                            continue
                        continue
        except asyncio.CancelledError:
            raise
        finally:
            # EXACTLY-ONCE teardown claim (vs _evict_teardown). Whoever claims
            # torn_down owns the FULL cleanup: reap the live process (handle OR the
            # still-booting pid — B2), drain unstarted inbox riders back to the queue
            # (B3), and fail the anchor slot if it died before the serve loop owned it.
            booting_pid = None
            pending_slots: list[Slot] = []
            async with self._registry_lock:
                claim = not r.torn_down
                if claim:
                    r.torn_down = True
                    handle = r.handle
                    r.handle = None
                    booting_pid = r.booting_pid
                    r.booting_pid = None
                    if r.inbox is not None:
                        while not r.inbox.empty():
                            try:
                                pending_slots.append(r.inbox.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                    if self._residents.get(r.model_tag) is r:
                        _transition_resident_state(r.state, ResidentState.DEAD)
                        r.state = ResidentState.DEAD
                        del self._residents[r.model_tag]
            if claim:
                # rework: capture thread_id for KV save before teardown
                idle_thread_id = r.idle_thread_id
                model_tag = r.model_tag
                if handle is not None and handle.is_alive() and idle_thread_id:
                    try:
                        # use _save_slot_kv instead of bolt-on POST.
                        # The idle holder teardown already saved KV, but this is a
                        # separate path (resident teardown) so we also save.
                        await self._save_slot_kv(
                            handle.port,
                            model_tag,
                            r.active_slot,
                            thread_id_override=idle_thread_id,
                            admission_ctx_len_override=r.idle_admission_ctx_len,
                            # a later phase a design note: see the evict_teardown site — explicit
                            # meta keeps the identity keying + arms T-GUARD here.
                            client_meta_override=r.idle_client_meta,
                        )
                        log.info(
                            "resident KV cache saved before teardown "
                            "(thread_id=%s, model_tag=%s)",
                            idle_thread_id[:8] if idle_thread_id else "none",
                            model_tag,
                        )
                    except Exception:
                        log.warning(
                            "resident KV cache save failed (best-effort): model_tag=%r",
                            model_tag,
                            exc_info=True,
                        )
                if handle is not None and handle.is_alive():
                    await self._reap_resident_handle(handle)
                elif booting_pid is not None:
                    await self._reap_booting_pid(booting_pid)  # B2: handle never set
                # Re-queue inbox riders that never started — another resident serves
                # (or fail their future if the queue is closing at shutdown). Async
                # context here (lock released), so await the ordered batch directly
                # to preserve FIFO among the riders.
                await self._requeue_slots_or_fail(pending_slots)
                # Fail the anchor slot if it died before the serve loop took
                # ownership (cancel/spawn-fail during RESERVED_LOADING) — the
                # supervisor's active_slot/inflight sweep can't see it then.
                # _spawn_for_resident already fails it on the health-timeout path,
                # so the done guard in _fail_completion_future keeps this idempotent.
                if (
                    slot is not None
                    and slot is not r.active_slot
                    and slot not in r.inflight
                ):
                    self._fail_completion_future(
                        slot,
                        RuntimeError(
                            f"resident {r.model_tag!r} driver exited before serve"
                        ),
                    )

    async def _serve_on_resident(
        self, r: "Resident", slot: Slot, handle
    ) -> None:
        """Serve one anchor slot on r's warm handle: ACTIVE (parallel fan-out /
        streaming / non-streaming complete) -> drain -> GRACE (ACTIVE_MATCH warm
        reuse within the grace window). Writes r.* directly (sole-writer). Does NOT
        handle idle handoff (the driver loop owns IDLE_EVICTABLE)."""
        n_parallel = max(1, getattr(handle, "parallel", 1))
        if slot.state is SlotState.STAGED:
            transition(slot, SlotState.LOADING)
        transition(slot, SlotState.ACTIVE)
        slot.started_active_at = time.monotonic()
        await self._audit_async(slot, "active")
        r.latest_keep_alive_s = (slot.client_meta or {}).get("keep_alive_s")
        is_streaming = (
            isinstance(slot.client_meta, dict)
            and bool(slot.client_meta.get("stream", False))
            and slot.stream_ready_event is not None
            and slot.stream_done_event is not None
        )
        if n_parallel > 1:
            slot.engine_op = "prefill"
            await self._fan_out_on_resident(r, slot, handle, n_parallel)
        elif is_streaming:
            slot.engine_op = "stream"
            slot.stream_handle = handle
            slot.stream_ready_event.set()
            try:
                await asyncio.wait_for(slot.stream_done_event.wait(), timeout=_STREAM_TIMEOUT_S)
            except TimeoutError:
                log.warning("streaming slot %s exceeded 3600s", slot.slot_id)
            if slot.completion_future is not None and not slot.completion_future.done():
                slot.completion_future.set_result({"_streamed": True})
        else:
            slot.engine_op = "prefill"
            # Option A: capture a clean-prefix KV (pre-generation) so the
            # next cold restore reuses instead of CLEAR+reprefill. No-op unless
            # single-series + large ctx + no equal/larger clean bin already saved.
            await self._probe_and_save_clean_kv(handle, slot)
            slot.engine_op = "decode"
            result = await self._complete_fn(slot, handle)
            if slot.completion_future is not None and not slot.completion_future.done():
                slot.completion_future.set_result(result)
        # GRACE + ACTIVE_MATCH warm reuse within the grace window (per-resident).
        transition(slot, SlotState.GRACE)
        slot.grace_started_at = time.monotonic()
        slot.engine_op = "idle"
        r.grace.start(slot.thread_id, slot.model_tag)
        await self._audit_async(slot, "grace_enter")
        deadline = time.monotonic() + self.runtime.queue.grace_seconds
        while time.monotonic() < deadline and not self._stop_event.is_set():
            matched = await self.queue.pop_matched_thread(
                slot.thread_id, slot.model_tag
            )
            if matched is not None:
                matched.port = handle.port
                matched.pid = handle.pid
                r.active_slot = matched
                try:
                    transition(matched, SlotState.ACTIVE_MATCH)
                    transition(matched, SlotState.ACTIVE)
                    r.latest_keep_alive_s = (
                        matched.client_meta or {}
                    ).get("keep_alive_s")
                    m_stream = (
                        isinstance(matched.client_meta, dict)
                        and matched.client_meta.get("stream", False)
                        and matched.stream_ready_event is not None
                        and matched.stream_done_event is not None
                    )
                    if m_stream:
                        matched.stream_handle = handle
                        matched.stream_ready_event.set()
                        matched.engine_op = "stream"
                        try:
                            await asyncio.wait_for(
                                matched.stream_done_event.wait(), timeout=_STREAM_TIMEOUT_S
                            )
                        except TimeoutError:
                            pass
                        if matched.completion_future is not None and not matched.completion_future.done():
                            matched.completion_future.set_result({"_streamed": True})
                    else:
                        matched.engine_op = "prefill"
                        await self._probe_and_save_clean_kv(handle, matched)
                        matched.engine_op = "decode"
                        res = await self._complete_fn(matched, handle)
                        if matched.completion_future is not None and not matched.completion_future.done():
                            matched.completion_future.set_result(res)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 -- per-slot isolation
                    self._fail_completion_future(matched, e)
                    log.exception("active_match completion failed %s", matched.slot_id)
                transition(matched, SlotState.GRACE)
                if r.grace.restart_for_followup():
                    deadline = time.monotonic() + self.runtime.queue.grace_seconds
                transition(matched, SlotState.POPPED)
                with state_db_session(self.boot.storage.state_db_path) as conn:
                    mark_slot_ended(conn, matched.slot_id, "active_match_completed")
                r.active_slot = slot
                continue
            await asyncio.sleep(0.05)
        transition(slot, SlotState.POPPED)
        assert not r.inflight, "Design #1 invariant: riders drain before GRACE exit"

    async def _fan_out_on_resident(
        self, r: "Resident", anchor: Slot, handle, n_parallel: int
    ) -> None:
        """Per-resident CONTINUOUS concurrent serve, up to ``n_parallel`` in-flight,
        riding ``r.inflight`` (IN-PLACE -- never rebind).

        LAYER-2 (per-model parallelism): same-model requests are routed by the
        dispatcher to THIS resident's ``r.inbox`` (model-pure), NOT left in the global
        staging queue -- so riders are pulled from ``r.inbox`` and the pipe is kept full
        up to ``n_parallel`` WHILE any slot is still generating, so a request arriving
        mid-burst joins immediately (llama-server ``--parallel N`` serves them). Each
        slot runs its own stream/non-stream completion, so a mixed-mode burst is fine.
        Drains every in-flight slot before returning so GRACE/teardown stays safe
        (Design #1 invariant: ``r.inflight`` empty on exit)."""
        r.inflight[:] = []
        tasks: dict[asyncio.Task, Slot] = {}

        def _launch(s: Slot) -> None:
            s.stream_handle = handle
            r.inflight.append(s)
            s_stream = (
                isinstance(s.client_meta, dict)
                and bool(s.client_meta.get("stream", False))
                and s.stream_ready_event is not None
                and s.stream_done_event is not None
            )
            if s_stream:
                s.stream_ready_event.set()
                t = asyncio.create_task(self._await_streamed_slot(s))
            else:
                t = asyncio.create_task(self._complete_one_slot(s, handle))
            tasks[t] = s

        def _admit_from_inbox() -> None:
            # Pull queued same-model riders (r.inbox is model-pure: the dispatcher only
            # routes THIS resident's model here) up to n_parallel concurrent in-flight.
            while len(tasks) < n_parallel:
                try:
                    extra = r.inbox.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if extra.is_evicted:
                    self._fail_completion_future(
                        extra, SlotEvictedError(f"slot {extra.slot_id} evicted")
                    )
                    continue
                _launch(extra)

        try:
            _launch(anchor)
            _admit_from_inbox()
            while tasks:
                done, _pending = await asyncio.wait(
                    set(tasks), return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    s = tasks.pop(t)
                    if s in r.inflight:
                        r.inflight.remove(s)
                    exc = t.exception()
                    if exc is not None:
                        self._fail_completion_future(s, exc)
                # Keep the pipe full: admit any riders that arrived mid-burst.
                _admit_from_inbox()
        finally:
            # Driver cancelled / unexpected raise mid-burst: cancel the still-running
            # per-slot tasks and fail their futures so no client hangs and no task is
            # orphaned. Normal exit leaves ``tasks`` empty -> no-op.
            if tasks:
                for t, s in list(tasks.items()):
                    t.cancel()
                    self._fail_completion_future(
                        s, RuntimeError(f"resident {r.model_tag!r} fan-out interrupted")
                    )
            r.inflight.clear()

    async def _complete_one_slot(self, s: Slot, handle) -> None:
        """Serve one NON-streaming slot on the shared handle + resolve its future."""
        try:
            res = await self._complete_fn(s, handle)
            if s.completion_future is not None and not s.completion_future.done():
                s.completion_future.set_result(res)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- per-slot isolation
            self._fail_completion_future(s, e)

    async def _await_streamed_slot(self, s: Slot) -> None:
        """Wait for a STREAMING slot's HTTP handler to finish, then resolve its future."""
        try:
            if s.stream_done_event is not None:
                await asyncio.wait_for(s.stream_done_event.wait(), timeout=_STREAM_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            pass
        if s.completion_future is not None and not s.completion_future.done():
            s.completion_future.set_result({"_streamed": True})

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
        # Hermes sub-agents are non-streaming, so the non-streaming path is the
        # one that matters most in practice. Riders must match the anchor's mode
        # (the admit loop enforces it; mixed-mode riders go back to the queue).
        anchor_streaming = bool(
            isinstance(anchor.client_meta, dict)
            and anchor.client_meta.get("stream", False)
        )
        # (1) The anchor is rider 0 -- already ACTIVE on `handle`.
        # P1c MUTATE IN PLACE, never rebind self._inflight. The lone
        # resident's ``inflight`` is the SAME list object (Resident(inflight=
        # self._inflight)); rebinding would orphan that view. ``[:] =`` keeps the
        # one canonical list so r.inflight never diverges (P1d reads r.inflight).
        self._inflight[:] = [anchor]
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
            # ⚠ SAFETY SEQUENCE MIRRORED in _admit_nonstreaming_riders (the
            # continuous-admit refill). Keep the pid=None-BEFORE-_force_cold
            # ordering identical in BOTH sites if you ever touch one.
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
        self._set_latest_keep_alive(max(_ka) if _ka else None)

        # (3) DISPATCH + DRAIN, branched on the anchor's mode. Non-streaming
        # riders (hermes sub-agents) are served by concurrent manager-owned
        # _complete_fn calls; streaming riders by their own routes (the barrier
        # below). Every admitted rider matches the anchor's mode.
        if not anchor_streaming:
            await self._drain_nonstreaming_riders(anchor, handle, n_parallel)
            return

        # (3-stream) DRAIN BARRIER. Each rider's route sets its OWN stream_done_event in
        # its finally (normal end, client-disconnect CancelledError, typed httpx
        # error, OR the route's own STREAM_TIMEOUT_S). We wait on the GENUINE
        # done-event -- never force-removing a rider while its route's httpx may
        # still be open (red-team must-fix: teardown gates on real route close,
        # not an accounting flag). A per-rider 3600s safety cap matches the
        # serial path's wait_for; on a cap, ONLY that rider is cooperatively
        # unwound -- siblings keep streaming. Returns when self._inflight is [].
        # Keyed by slot_id (Slot is a non-frozen dataclass and thus unhashable,
        # so it cannot be a dict key directly).
        waiters: dict[str, asyncio.Task] = {}
        for s in self._inflight:
            assert s.stream_done_event is not None
            waiters[s.slot_id] = asyncio.create_task(
                asyncio.wait_for(s.stream_done_event.wait(), timeout=_STREAM_TIMEOUT_S)
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
            self._inflight.clear()  # P1c: in-place, keep r.inflight alias canonical

    async def _admit_nonstreaming_riders(
        self, anchor: Slot, handle, n_parallel: int,
    ) -> list[Slot]:
        """: admit MORE same-model NON-streaming riders onto
        the shared sidecar, up to ``n_parallel`` total in-flight. Returns the
        slots newly promoted (each appended to ``self._inflight`` and ACTIVE on
        ``handle``); ``[]`` when none are admissible right now.

        Mirrors the one-shot admit promotion in ``_fan_out_and_drain`` (same
        drift-guard, evicted-skip, audit) but is called REPEATEDLY by the drain
        so a rider that arrives mid-burst joins the free ``--parallel`` slot
        immediately instead of waiting out the anchor (fixes the 4.5s/9.1s
        serialization the one-shot admit left when req2 hadn't yet reached the
        queue at anchor launch).

        Single-mutator preserved: runs in the worker_loop task -- the sole
        writer of ``self._inflight``. A wrong-model OR streaming pop is pushed
        back to the queue HEAD and STOPS the refill; that same force-FIFO-head
        path is how the queue lets a starved OTHER-model request (e.g. the main
        27b, after ``max_other_model_wait_s``) break the same-model batch so the
        model can still swap back -- so this can never livelock the swap.
        """
        admitted: list[Slot] = []
        while len(self._inflight) < n_parallel:
            extra = await self.queue.pop_next(warm_model_tag=anchor.model_tag)
            if extra is None:
                break  # nothing queued for this model right now
            extra_streaming = (
                isinstance(extra.client_meta, dict)
                and bool(extra.client_meta.get("stream", False))
            )
            if extra.model_tag != anchor.model_tag or extra_streaming:
                # Not a rider for THIS non-streaming fan-out: push back to HEAD
                # (the next worker_loop pop / fresh fan-out / model swap handles
                # it) and STOP -- one bounded refill, never a queue busy-spin.
                await self.queue.enqueue_head(extra)
                break
            if extra.is_evicted:
                self._fail_completion_future(
                    extra,
                    SlotEvictedError("client disconnected before fan-out admit"),
                )
                await self._audit_async(extra, "evicted_pre_fanout")
                continue
            # Promote onto the shared sidecar (identical to the one-shot path).
            extra.port = handle.port
            extra.pid = handle.pid
            try:
                transition(extra, SlotState.LOADING)
                transition(extra, SlotState.ACTIVE)
            except InvalidTransition as drift_err:
                log.warning(
                    "fan-out refill rider %s state drift (%s) -- dropping; %s",
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
            self._inflight.append(extra)
            await self._audit_async(extra, "fanout_rider_active")
            admitted.append(extra)
        return admitted

    async def _drain_nonstreaming_riders(
        self, anchor: Slot, handle, n_parallel: int,
    ) -> None:
        """Non-streaming concurrent dispatch (Design #1): fire ``_complete_fn``
        for EVERY rider at once -- each is its own httpx POST to the sidecar's
        ``--parallel`` slots, so the engine serves them concurrently via
        continuous batching. Await all, resolve each ``completion_future`` with
        its result (or fail ONLY that rider on a typed sidecar error -- siblings
        keep going), terminal-clean the riders, and clear ``self._inflight``.
        Returns only when every rider has completed (drain-before-swap holds).
        This is the path hermes sub-agents take (they call non-streaming).

        : while draining, keep the sidecar's ``--parallel``
        slots full by admitting same-model riders that arrive mid-burst -- the
        loop wakes on a completion OR a short poll (``_FANOUT_ADMIT_POLL_S``),
        then refills up to ``n_parallel`` via ``_admit_nonstreaming_riders``. So
        N sub-agents fired ~together run ``n_parallel``-at-a-time (the rest
        queue) instead of serializing, while the queue's same-model batch cap
        still lets the model swap back.
        """
        # One concurrent completion task per rider, keyed by slot_id (Slot is a
        # non-frozen dataclass and thus unhashable).
        ctasks: dict[str, asyncio.Task] = {
            s.slot_id: asyncio.create_task(self._complete_fn(s, handle))
            for s in self._inflight
        }
        try:
            # Admit any same-model riders ALREADY queued at drain start. Covers
            # the one-shot admit race: req2 reaches the queue micro-seconds after
            # the anchor's one-shot admit ran, so it would otherwise idle a free
            # slot until the anchor finished.
            for s in await self._admit_nonstreaming_riders(
                anchor, handle, n_parallel
            ):
                ctasks[s.slot_id] = asyncio.create_task(
                    self._complete_fn(s, handle)
                )
            pending = set(ctasks.values())
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=_FANOUT_ADMIT_POLL_S,
                    return_when=asyncio.FIRST_COMPLETED,
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
                # Refill any slot freed this tick (and admit a mid-burst arrival
                # that landed while a slot was free) up to n_parallel.
                for s in await self._admit_nonstreaming_riders(
                    anchor, handle, n_parallel
                ):
                    t = asyncio.create_task(self._complete_fn(s, handle))
                    ctasks[s.slot_id] = t
                    pending.add(t)
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
            self._inflight.clear()  # P1c: in-place, keep r.inflight alias canonical

    async def _process_slot(self, slot: Slot) -> None:
        """Drive one slot through STAGED → LOADING → ACTIVE → GRACE → POPPED."""
        self._set_active_slot(slot)
        # keep-alive handling: clear cross-slot keep_alive leakage. A previous slot's
        # ACTIVE_MATCH chain may have left a value here even though that
        # slot's grace→idle consumed it; defensive reset keeps the invariant
        # "value reflects this anchor cycle only" honest.
        self._set_latest_keep_alive(None)

        try:
            # Build backend argv from manifest if available; tolerate missing
            # manifest for testing convenience.
            argv: list[str] = []
            manifest_is_mlx = False
            manifest_found = True
            try:
                manifest = read_manifest(self.boot.storage.manifests_path, slot.model_tag)
                manifest_is_mlx = manifest.is_mlx()
                if manifest_is_mlx:
                    # MLX: no GGUF blob. Build argv from the MLX closed allowlist.
                    argv = mlx_flags_to_argv(manifest.mlx_server_flags)
                else:
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

            # idle-holder polish (stress test): pre-validate the
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

            # P1a: allocate the listen port from the resident registry instead
            # of hard-coding default_port_base. At MAX_PARALLEL_SIDECARS == 1
            # this returns default_port_base (11500) verbatim -- behaviour is
            # byte-for-byte identical to the deployed manager; the indirection
            # is what lets a Phase-1 second sidecar land on the next free port.
            port = self._alloc_port()

            # STAGED → LOADING
            transition(slot, SlotState.LOADING)
            await self._audit_async(slot, "stage_to_loading")

            # idle-holder wiring: warm-inherit path. If the previous slot left
            # a warm sidecar holding the same model_tag and the idle
            # window has not expired, reuse the handle and skip spawn +
            # health-wait (sidecar is already healthy by construction).
            # a tracked issue: capture idle_handle in local var BEFORE the check to
            # avoid TOCTOU — another coroutine could change _idle_handle
            # between the check and the warm_inherit block.
            _idle_h = self._idle_handle
            _idle_mt = self._idle_model_tag
            _idle_exp = self._idle_expires_at
            _idle_client_meta = self._idle_client_meta
            # RC (stuck-handle root fix) Bug A: NEVER warm-inherit a DEAD idle handle.
            # If the idle-hot holder's llama-server died (silent crash / OOM / premature
            # unload), reusing it strands every request on a dead pid with no re-spawn
            # (observed: `active_handle_pid=X alive=False idle_match=True` looping for 10+
            # min, main never reloads). Gating on is_alive drops a dead handle into the
            # `else` branch, which tears down the dead holder (_teardown_idle_holder: clears
            # + skips flush on a dead pid) and does a safety-checked FRESH spawn = recovery.
            warm_inherit = (
                _idle_h is not None
                and _idle_mt == slot.model_tag
                and _idle_exp is not None
                and time.monotonic() < _idle_exp
                and _idle_h.is_alive()
            )
            if (
                not warm_inherit
                and _idle_h is not None
                and _idle_mt == slot.model_tag
                and _idle_exp is not None
                and time.monotonic() < _idle_exp
                and not _idle_h.is_alive()
            ):
                log.warning(
                    "Idle-hot holder DEAD (pid=%s, model=%s) within window — clearing + "
                    "cold-spawning fresh (RC stuck-handle Bug A dead-handle recovery)",
                    getattr(_idle_h, "pid", None), slot.model_tag)
            if warm_inherit:
                handle = _idle_h
                self._clear_idle_holder()
                slot.port = handle.port
                slot.pid = handle.pid
                self._set_active_handle(handle)
                self._bump_spawn_seq()  # live-monitor: mark new active handle + mirror to active resident (sole writer = worker_loop)
                await self._audit_async(slot, "idle_hot_inherit")
                # T3: restore client_meta alongside thread_id (fixes session_id loss + KV cache bin mismatch).
                # ROOT FIX (2026-07-09): restore the idle-holder meta ONLY when the
                # INCOMING request is the SAME session (or carries no session_id of its own).
                # A DIFFERENT session that warm-inherits this idle sidecar (curator/sub-agent
                # <-> main handoff) MUST keep ITS OWN complete client_meta (Slot.new already set
                # it) — clobbering it swaps session_id/role onto the wrong identity => wrong
                # KV-bin ownership (main returns after a sub-agent and can't find its own bin:
                # "reloaded but no context") AND a per-role save-gate misreads the role.
                # thread_id already stays the incoming's, so ONLY client_meta was inconsistent.
                if _idle_client_meta:
                    _inc_cm = slot.client_meta if isinstance(slot.client_meta, dict) else {}
                    _inc_sid = _inc_cm.get("session_id")
                    _idle_sid = (_idle_client_meta.get("session_id")
                                 if isinstance(_idle_client_meta, dict) else None)
                    if isinstance(_idle_client_meta, dict) and (not _inc_sid or _inc_sid == _idle_sid):
                        slot.client_meta = _idle_client_meta
                        log.debug(
                            "Idle-hot warm-inherit: restored client_meta with session_id=%s",
                            _idle_sid if _idle_sid is not None else "N/A",
                        )
                    else:
                        log.info(
                            "Idle-hot warm-inherit: KEPT incoming client_meta (incoming "
                            "session_id=%s != idle-holder session_id=%s) — no cross-session "
                            "identity clobber (root fix)", _inc_sid, _idle_sid)
                # Skip spawn + health wait; jump straight to LOADING -> ACTIVE.
                healthy = True
            else:
                # Different model OR no idle holder. If a stale holder
                # exists for a different model, tear it down before
                # spawning new -- immediate switch (operator request intent).
                if self._idle_handle is not None:
                    await self._teardown_idle_holder("model_swap")
                # operator request safety guardrails: pre-spawn host checks
                # (VRAM headroom + RAM + CPU load + IO wait). Refuse here
                # rather than spawning into an OOM / IO-stuck host.
                if self.runtime.queue.safety_enabled:
                    manifest_vram = 0
                    manifest_ctx = 0
                    manifest_gguf_bytes = 0
                    manifest_kv_quant = "f16"
                    manifest_kv_quant_v = "f16"
                    manifest_no_kv_offload = False
                    manifest_parallel = 1
                    manifest_split_mode = "layer"
                    manifest_main_gpu = 0
                    manifest_cpu_moe = False
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
                        # KV quant: K half scaled by cache_type_k, V half by
                        # cache_type_v (fix; V used to be ignored, so a
                        # K=f16 + V=turbo3 config was over-counted as full-f16).
                        manifest_kv_quant = (
                            m_for_vram.llama_server_flags.get("cache_type_k")
                            or "f16"
                        )
                        manifest_kv_quant_v = (
                            m_for_vram.llama_server_flags.get("cache_type_v")
                            or manifest_kv_quant
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
                        # GPU placement (per-model manifest flags): split_mode
                        # drives the VRAM gate's aggregate-vs-single-card budget;
                        # main_gpu picks the card for split_mode:none. Absent ->
                        # "layer" = llama.cpp default (aggregate across all GPUs).
                        manifest_split_mode = str(
                            m_for_vram.llama_server_flags.get("split_mode", "layer")
                            or "layer"
                        )
                        manifest_main_gpu = int(
                            m_for_vram.llama_server_flags.get("main_gpu", 0) or 0
                        )
                        manifest_is_mlx = m_for_vram.is_mlx()
                        # cpu_moe / n_cpu_moe offload experts to RAM -> the closed-form
                        # body=gguf over-counts; the cpu-moe gate branch trusts the
                        # manifest's measured expected_vram for those configs (parity
                        # with the cap>=2 driver path).
                        manifest_cpu_moe = bool(
                            m_for_vram.llama_server_flags.get("cpu_moe")
                            or int(
                                m_for_vram.llama_server_flags.get("n_cpu_moe", 0) or 0
                            )
                            > 0
                        )
                    except FileNotFoundError:
                        manifest_vram = 0
                    gates = await asyncio.to_thread(all_safety_gates,
                        min_free_ram_mib=self.runtime.queue.safety_min_free_ram_mib,
                        min_free_vram_mib=self.runtime.queue.safety_min_free_vram_mib,
                        max_load_per_core=self.runtime.queue.safety_max_load_per_core,
                        max_iowait_percent=self.runtime.queue.safety_max_iowait_percent,
                        manifest_expected_vram_bytes=manifest_vram,
                        iowait_sample_window_s=self.runtime.queue.safety_iowait_sample_window_s,
                        ctx_size=manifest_ctx,
                        gguf_size_bytes=manifest_gguf_bytes,
                        kv_cache_quant=manifest_kv_quant,
                        kv_cache_quant_v=manifest_kv_quant_v,
                        no_kv_offload=manifest_no_kv_offload,
                        parallel=manifest_parallel,
                        split_mode=manifest_split_mode,
                        main_gpu=manifest_main_gpu,
                        cpu_moe_offload=manifest_cpu_moe,
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
                handle: SidecarHandle
                if manifest_is_mlx:
                    handle = mlx_spawn(
                        port,
                        slot.model_tag,
                        manifest.model_repo if manifest_found else "",
                        manifest.model_path if manifest_found else "",
                        manifest.mlx_server_flags if manifest_found else {},
                        python_binary=self.boot.runtime.mlx_python_binary,
                    )
                else:
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
                self._set_active_handle(handle)
                self._bump_spawn_seq()  # live-monitor: mark new active handle + mirror to active resident (sole writer = worker_loop)
                # LOADING → ACTIVE (or LOADING_FAIL → POPPED)
                healthy = await self._wait_healthy(
                    port, self.runtime.queue.loading_health_timeout_s,
                    is_alive=handle.is_alive,
                )
                # V1 (the operator): verify the MODEL actually loaded after the
                # (swap/cold) spawn + emit a LOAD_VERIFY record. Observability ONLY —
                # never alters the decode path (helpers are read-only + never-raise;
                # belt-wrapped anyway). The bounded verify+RETRY loop is the follow-up
                # behavior step once these records show the real failure modes.
                try:
                    _mv = await load_verify_log.verify_model_resident(
                        handle, mlx=manifest_is_mlx
                    )
                    load_verify_log.log_load_verify(
                        event="model_load", trigger="spawn",
                        model_tag=slot.model_tag, port=port,
                        pid=getattr(handle, "pid", None),
                        process_alive=_mv.get("process_alive"),
                        health_200=_mv.get("health_200") if _mv.get("health_200") is not None else bool(healthy),
                        model_resident=_mv.get("model_resident"),
                        retry_count=0,
                        final_status="ok" if healthy else "failed",
                        reason=None if healthy else "health-timeout",
                        session_id=(slot.client_meta or {}).get("session_id") if isinstance(slot.client_meta, dict) else None,
                    )
                except Exception:
                    log.debug("LOAD_VERIFY model_load emit failed (best-effort)", exc_info=True)
                # Cold-spawn only: best-effort KV restore after fresh sidecar passes health.
                if healthy:
                    await self._restore_slot_kv(port, slot.model_tag, slot)
                    # V1: record the engine's ACTUAL post-restore n_past
                    # (the manager previously TRUSTED the restore POST 200 — "engine
                    # determines actual n_past" — with no check). expected_tokens=None
                    # in V1 (restore internals own the real expected; V2 threads it).
                    try:
                        _kv = await load_verify_log.verify_kv_restored(
                            handle, 0, None,
                        )
                        load_verify_log.log_load_verify(
                            event="kv_restore", trigger="spawn",
                            model_tag=slot.model_tag, port=port,
                            pid=getattr(handle, "pid", None),
                            process_alive=True, health_200=True, model_resident=True,
                            kv_expected_tokens=None,
                            kv_actual_n_past=_kv.get("kv_actual_n_past"),
                            kv_restore_ok=_kv.get("kv_restore_ok"),
                            retry_count=0, final_status="ok",
                            session_id=(slot.client_meta or {}).get("session_id") if isinstance(slot.client_meta, dict) else None,
                        )
                    except Exception:
                        log.debug("LOAD_VERIFY kv_restore emit failed (best-effort)", exc_info=True)
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
            # keep-alive handling: capture anchor's keep_alive intent for grace→idle decision.
            self._set_latest_keep_alive((slot.client_meta or {}).get("keep_alive_s"))

            # telemetry — slot assign + prefill start
            try:
                self._telemetry.on_slot_assign(slot)
                self._telemetry.on_prefill_start(slot)
            except Exception:
                pass

            # Option A: capture a clean-prefix KV (pre-generation) via a
            # prefill-only probe BEFORE the real request generates, so the next cold
            # restore reuses instead of CLEAR+reprefill. Placed here (after ACTIVE +
            # restore, before the streaming/non-streaming branch) so it fires for BOTH
            # streaming and non-streaming requests on the live _process_slot path. The
            # real request below then warm-reuses this prefill (no double prefill).
            # No-op unless single-series + large ctx + no equal/larger clean bin exists.
            await self._probe_and_save_clean_kv(handle, slot)

            # SSE streaming: branch on streaming mode.
            #
            # Non-streaming (existing): await self._complete_fn(slot, handle) to
            # post chat-completion, set completion_future, advance to GRACE.
            #
            # Streaming: client_meta["stream"] is True AND submit_for_streaming
            # pre-armed slot.stream_ready_event + slot.stream_done_event. We
            # SKIP _complete_fn (the route owns the httpx streaming connection).
            # Instead: hand the SidecarHandle to the route via slot.stream_handle,
            # signal stream_ready_event so the route can open its httpx.stream,
            # then BLOCK here on stream_done_event until the route reports the
            # stream has finished (normal close, client disconnect, or error).
            # Slot stays in ACTIVE the entire time so ACTIVE_MATCH cannot promote
            # a second submission against the same sidecar (a design review
            # critical catch — single-slot invariant preserved).
            is_streaming = (
                isinstance(slot.client_meta, dict)
                and bool(slot.client_meta.get("stream", False))
                and slot.stream_ready_event is not None
                and slot.stream_done_event is not None
            )
            # The classifier: the hash chain the engine's warm KV holds after the
            # most recent decode on THIS handle (= that request's prompt + generated
            # <think> turn). Feeds the NO-DOWNGRADE gate before each grace follow-up
            # decode. [] = unknown (fan-out/streaming/noop) -> gate never forces
            # (safe). Local to this anchor cycle = scoped exactly to the warm slot's
            # grace lifetime (lifecycle-accurate; no cross-cycle staleness).
            warm_chain: list[str] = []
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
                        timeout=_STREAM_TIMEOUT_S,  # 1 hour cap; routes typically signal in seconds
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
                # Streaming path: the route stashed the streamed generated text on the slot
                # before signaling done -> record what the engine's warm KV now holds
                # (anchor prompt + generated <think> turn) for the first grace
                # follow-up's no-downgrade gate. Same as the non-streaming anchor.
                warm_chain = self._engine_view_chain(slot, None)
                # shadow byte-match self-check (DORMANT, obs-only): stash this
                # streamed anchor turn's think-free assistant hash for next-turn compare.
                self._record_shadow_bytematch_probe(slot, None)
                # Option A shadow-reprefill (SAVE-side; INERT unless
                # TURBOHAUL_SHADOW_REPREFILL). Fired AFTER set_result above -> zero
                # added response TTFT. Persists the think-free end-of-turn-N state
                # under a DISTINCT .shadow bin; restore path UNCHANGED (PL's step).
                await self._shadow_reprefill_and_save(handle, slot, None)
            else:
                # Non-streaming path (existing behaviour, unchanged).
                # Completion (httpx forward when wired; default is noop)
                result = await self._complete_fn(slot, handle)
                if slot.completion_future is not None and not slot.completion_future.done():
                    slot.completion_future.set_result(result)
                # M5 (WIN 2): WRITE site. Cache this non-streaming result +
                # resolve the leader's single-flight future so a byte-identical retry
                # replays instantly. No-op unless the slot is a cache leader (its
                # submit_and_wait stashed completion_cache_key). Runs on the single
                # worker_loop, between set_result and the next await, so it stays
                # atomic vs the leader's submit_and_wait finally (no double-resolve).
                self._completion_cache_store(slot, result)
                # the classifier: record what the engine's warm KV now holds (anchor prompt +
                # generated <think> turn) so the first grace follow-up's no-downgrade
                # gate can compare against it.
                warm_chain = self._engine_view_chain(slot, result)
                # shadow byte-match self-check (DORMANT, obs-only): stash this
                # non-streamed anchor turn's think-free assistant hash for next-turn compare.
                self._record_shadow_bytematch_probe(slot, result)
                # Option A shadow-reprefill (SAVE-side; INERT unless
                # TURBOHAUL_SHADOW_REPREFILL). Fired AFTER set_result above -> zero
                # added response TTFT. Persists the think-free end-of-turn-N state
                # under a DISTINCT .shadow bin; restore path UNCHANGED (PL's step).
                await self._shadow_reprefill_and_save(handle, slot, result)

            # Drain-before-swap hard gate (Design #1): every fan-out rider must
            # have finished before the anchor advances to GRACE / the sidecar is
            # torn down. _fan_out_and_drain returns only when self._inflight is
            # empty; for parallel:1 it is never populated, so this is a no-op.
            assert not self._inflight, (
                "Design #1 invariant: riders must drain before ACTIVE->GRACE"
            )
            # D3 a completed serve proves the engine is healthy
            # again — drop the ENGINE STALLED banner (replace-only; sole writer =
            # worker_loop, same task as the strike site, so no write-write race).
            if getattr(self, "_engine_stall", None) is not None:
                log.info("ENGINE STALL cleared — serve completed OK (slot=%s thread=%s)",
                         slot.slot_id, (slot.thread_id or "?")[:40])
                self._engine_stall = None
            # ACTIVE → GRACE
            transition(slot, SlotState.GRACE)
            slot.grace_started_at = time.monotonic()
            self.grace.start(slot.thread_id, slot.model_tag)
            await self._audit_async(slot, "grace_enter")

            # telemetry — completion
            try:
                self._telemetry.on_completion(slot, "grace_enter")
            except Exception:
                pass

            # Wait for grace window OR promote a matched staging slot via
            # ACTIVE_MATCH (warm-slot reuse). Per the FSM; this transition
            # cascades same-(thread_id, model_tag) follow-up requests through
            # the warm llama-server without re-spawn.
            deadline = time.monotonic() + self.runtime.queue.grace_seconds
            while time.monotonic() < deadline and not self._stop_event.is_set():
                # idle-holder fix: atomic find + remove in one lock acquire
                matched = await self.queue.pop_matched_thread(
                    slot.thread_id, slot.model_tag
                )
                if matched is not None:
                    matched.port = handle.port
                    matched.pid = handle.pid
                    self._set_active_slot(matched)
                    # a state-drift fix (design review): state-drift guard. If matched.state
                    # drifted between find_matched_thread + here (concurrent
                    # reconcile, retry path, etc.), transition raises
                    # InvalidTransition which would crash worker_loop. Wrap
                    # the promotion + park-on-drift instead of propagate.
                    try:
                        transition(matched, SlotState.ACTIVE_MATCH)
                        await self._audit_async(matched, "active_match_promoted")
                        transition(matched, SlotState.ACTIVE)
                        # keep-alive handling: each matched-follow-up's keep_alive overrides
                        # the anchor's for the next grace→idle calculation. Mirrors
                        # Ollama "timer resets on request receipt" rule.
                        self._set_latest_keep_alive(
                            (matched.client_meta or {}).get("keep_alive_s")
                        )
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
                        self._set_active_slot(slot)
                        continue
                    matched.started_active_at = time.monotonic()
                    completed_ok = True
                    # Round 8 fix: streaming-path warm-reuse. When a streaming submit lands on an
                    # already-active matched slot, the HTTP route owns the upstream connection via
                    # stream_handle. Worker MUST NOT call _complete_fn (would open a 2nd sidecar
                    # connection and violate the design's single-slot invariant). Hand
                    # off via stream_ready_event and block on stream_done_event until route drains.
                    # Prior bug: this branch unconditionally called _complete_fn → matched slot's
                    # stream_ready_event was never set → route's SLOT_READY_TIMEOUT_S fired at 600s
                    # every turn ≥ 2 of a Hermes multi-tool-call agent loop.
                    matched_is_streaming = bool(
                        isinstance(matched.client_meta, dict)
                        and matched.client_meta.get("stream", False)
                    )
                    try:
                        if matched_is_streaming:
                            # events are armed in submit before
                            # enqueue, so they SHOULD always exist here. This
                            # defensive guard handles the never-hit case where
                            # they are somehow still None (e.g. non-streaming
                            # submit that set stream flag in client_meta). Fail
                            # cleanly WITHOUT opening a 2nd sidecar connection.
                            if (
                                matched.stream_ready_event is None
                                or matched.stream_done_event is None
                            ):
                                log.warning(
                                    "active_match streaming slot %s events None "
                                    "(should not happen) — failing promotion cleanly",
                                    matched.slot_id,
                                )
                                self._fail_completion_future(
                                    matched,
                                    RuntimeError("streaming events not armed"),
                                )
                                continue
                            # Streaming path: THE STREAMING warm grace follow-up (streaming path).
                            # The engine's KV holds the anchor's just-generated <think>
                            # turn (warm_chain); the incoming user turn is think-
                            # stripped. Force-restore the pinned clean bin BEFORE we
                            # signal the route to open its stream, so the engine's next
                            # get_common_prefix runs against the clean prefix (strict
                            # extension, no CLEAR). Same no-downgrade gate + safe-degrade
                            # as the non-streaming path.
                            # log-only divergence capture on WARM streaming path.
                            inc_div_chain = getattr(matched, "admission_hash_chain", []) or []
                            inc_div_msg = (getattr(matched, "client_meta", None) or {}).get("messages") or []
                            self._log_warm_serve_divergence(matched, inc_div_chain, inc_div_msg, handle.port)
                            # P3 (DURABLE MANAGER B): reload-before-serve on WARM streaming path.
                            # When TURBOHAUL_DURABLE_RING=ON, if the physically-resident (role,session)
                            # does NOT match the incoming request's (role,session), RELOAD the matching
                            # ring state so engine residency matches the request head. THEN proceed with
                            # existing warm path (_maybe_force_clean_restore). Fires ONLY on explicit
                            # residency mismatch (old_tag SET and != ring_key); old_tag=None -> skip (safe).
                            # ORDER: read old_tag -> decide/do reload -> THEN advance tag = current request's key
                            if _durable_ring_enabled():
                                inc_chain_stream = getattr(matched, "admission_hash_chain", []) or []
                                inc_messages_stream = (getattr(matched, "client_meta", None) or {}).get("messages") or []
                                reloaded, ring_key = await self._reload_matching_state_before_serve(
                                    handle.port, matched.model_tag, matched, inc_chain_stream, inc_messages_stream
                                )
                                if reloaded:
                                    log.info("DURABLE_RING warm reload: streaming path reloaded key=%s before _maybe_force_clean_restore", ring_key)
                                # Advance resident tag to current request's key (regardless of reload)
                                # so subsequent requests see this as the resident
                                if ring_key:
                                    resident = self._residents.get(matched.model_tag)
                                    if resident:
                                        resident.resident_state_tag = ring_key
                            await self._maybe_force_clean_restore(
                                handle.port, matched.model_tag, matched, warm_chain,
                            )
                            # (per the "scale as context
                            # grows" design): re-save/extend the clean bin on the per-turn WARM
                            # path. Track-1 keep-warm (IDLE_HOT_S=1800) removed the
                            # eviction-saves the clean re-save rode -> it froze @26 while
                            # incoming grew (forced-restore then reprocessed a growing
                            # tail = the operator's "slow, getting worse"). Decouple it: fire the
                            # same pre-generation clean probe as the _process_slot path
                            # (2913) so the clean bin grows with the conversation; the
                            # existing throttle bounds the actual re-save to every
                            # LAGREDUCER_MIN_GROWTH_TURNS. Pre-stream so the decode warm-
                            # reuses the probe's prefill (no double prefill).
                            await self._probe_and_save_clean_kv(handle, matched)
                            matched.stream_handle = handle
                            matched.stream_ready_event.set()
                            try:
                                await asyncio.wait_for(
                                    matched.stream_done_event.wait(),
                                    timeout=_STREAM_TIMEOUT_S,
                                )
                            except asyncio.TimeoutError:
                                log.warning(
                                    "active_match streaming slot %s exceeded 3600s waiting for stream_done_event",
                                    matched.slot_id,
                                )
                            if matched.completion_future is not None and not matched.completion_future.done():
                                matched.completion_future.set_result({"_streamed": True})
                            # Streaming path: refresh warm_chain from the route-stashed streamed
                            # text (prompt + generated <think> turn) for the NEXT grace
                            # follow-up's gate. [] if the route couldn't reconstruct it
                            # (parse-miss/tool-call) -> next gate safe-degrades.
                            warm_chain = self._engine_view_chain(matched, None)
                            # shadow byte-match self-check (DORMANT, obs-only).
                            self._record_shadow_bytematch_probe(matched, None)
                            # Option A shadow-reprefill (SAVE-side; INERT
                            # unless TURBOHAUL_SHADOW_REPREFILL). AFTER set_result ->
                            # zero added TTFT; DISTINCT .shadow bin; restore UNCHANGED.
                            await self._shadow_reprefill_and_save(handle, matched, None)
                        else:
                            # The classifier: this is THE warm grace follow-up. The
                            # engine's KV holds the anchor's just-generated <think>
                            # turn (warm_chain); the incoming user turn is think-
                            # stripped. Force-restore the pinned clean bin BEFORE the
                            # decode when (and only when) the no-downgrade gate says
                            # it improves reuse — so get_common_prefix runs against
                            # the clean prefix (strict extension, no CLEAR).
                            # log-only divergence capture on WARM non-streaming path.
                            inc_div_chain = getattr(matched, "admission_hash_chain", []) or []
                            inc_div_msg = (getattr(matched, "client_meta", None) or {}).get("messages") or []
                            self._log_warm_serve_divergence(matched, inc_div_chain, inc_div_msg, handle.port)
                            # P3 (DURABLE MANAGER B): reload-before-serve on WARM non-streaming path.
                            # When TURBOHAUL_DURABLE_RING=ON, if the physically-resident (role,session)
                            # does NOT match the incoming request's (role,session), RELOAD the matching
                            # ring state so engine residency matches the request head. THEN proceed with
                            # existing warm path (_maybe_force_clean_restore). Fires ONLY on explicit
                            # residency mismatch (old_tag SET and != ring_key); old_tag=None -> skip (safe).
                            # ORDER: read old_tag -> decide/do reload -> THEN advance tag = current request's key
                            if _durable_ring_enabled():
                                inc_chain_nonstream = getattr(matched, "admission_hash_chain", []) or []
                                inc_messages_nonstream = (getattr(matched, "client_meta", None) or {}).get("messages") or []
                                reloaded, ring_key = await self._reload_matching_state_before_serve(
                                    handle.port, matched.model_tag, matched, inc_chain_nonstream, inc_messages_nonstream
                                )
                                if reloaded:
                                    log.info("DURABLE_RING warm reload: non-streaming path reloaded key=%s before _maybe_force_clean_restore", ring_key)
                                # Advance resident tag to current request's key (regardless of reload)
                                # so subsequent requests see this as the resident
                                if ring_key:
                                    resident = self._residents.get(matched.model_tag)
                                    if resident:
                                        resident.resident_state_tag = ring_key
                            await self._maybe_force_clean_restore(
                                handle.port, matched.model_tag, matched, warm_chain,
                            )
                            # re-save/extend the clean bin on the
                            # per-turn warm path (see streaming site above for rationale).
                            # Pre-decode so _complete_fn warm-reuses the probe's prefill.
                            await self._probe_and_save_clean_kv(handle, matched)
                            result2 = await self._complete_fn(matched, handle)
                            if (
                                matched.completion_future is not None
                                and not matched.completion_future.done()
                            ):
                                matched.completion_future.set_result(result2)
                            # M5 (WIN 2): WRITE site for the warm grace
                            # follow-up (its own submit_and_wait was the leader for
                            # this turn's key) — a follow-up turn that times out then
                            # retries also replays instantly. Same guard/atomicity as
                            # the anchor WRITE above.
                            self._completion_cache_store(matched, result2)
                            # the classifier: refresh warm_chain to what the engine now holds
                            # after THIS follow-up's decode (its prompt + generated
                            # <think> turn), for the next grace follow-up's gate.
                            warm_chain = self._engine_view_chain(matched, result2)
                            # shadow byte-match self-check (DORMANT, obs-only).
                            self._record_shadow_bytematch_probe(matched, result2)
                            # Option A shadow-reprefill (SAVE-side; INERT
                            # unless TURBOHAUL_SHADOW_REPREFILL). AFTER set_result ->
                            # zero added TTFT; DISTINCT .shadow bin; restore UNCHANGED.
                            await self._shadow_reprefill_and_save(handle, matched, result2)
                    except asyncio.CancelledError:
                        # Round 8 design note: cooperatively unwind route's blocking httpx call
                        # by signaling stream_done before terminal-park (avoids zombie
                        # route + dead slot drift).
                        if (
                            matched_is_streaming
                            and matched.stream_done_event is not None
                            and not matched.stream_done_event.is_set()
                        ):
                            matched.stream_done_event.set()
                        # a state-drift fix (design review): cancellation mid-ACTIVE_MATCH
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
                                "cleanup mark_slot_ended failed for %s",
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
                    # fix: on completion failure, skip grace pretense
                    # go ACTIVE → GRACE → POPPED + mark failed, keep state
                    # machine honest. (transition validates each hop.)
                    transition(matched, SlotState.GRACE)
                    if completed_ok:
                        await self._audit_async(matched, "active_match_to_grace")
                        if self.grace.restart_for_followup():
                            # idle-holder fix: also bump per-slot extension_count
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
                    # C1 fix (design review): matched slot's request is done; its
                    # sidecar was the anchor's warm process (reused, not its own).
                    # Anchor `slot` remains the GRACE driver until grace expiry.
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
                    self._set_active_slot(slot)  # anchor for teardown bookkeeping
                    continue
                await asyncio.sleep(0.05)

            # GRACE → POPPED (slot lifecycle ends here)
            transition(slot, SlotState.POPPED)
            # idle-holder wiring: hold the sidecar in idle for follow-up reuse
            # by any same-model_tag request inside idle_hot_load_seconds.
            # When idle_seconds == 0 (test default), this is equivalent to
            # immediate teardown -- preserves "grace-expired" reason on the
            # mark_slot_ended audit (backward-compat with existing tests).
            #
            # keep-alive handling (design option E + the operator "fully automatic" 20:25Z):
            # honor the latest request's keep_alive intent as IDLE_HOT extension.
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
                # (design review spec — never indefinite on single-GPU).
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
            self._set_latest_keep_alive(None)
            # severity item P2 residency — cheap insurance floor: even when the
            # client's keep_alive would unload the model NOW (idle_seconds <= 0),
            # keep it warm if the NEXT queued request is the SAME model (the operator
            # intent). The queued same-model request then warm-inherits on the next
            # worker tick instead of paying teardown + respawn. Single-slot-safe:
            # only the resident model is ever held, and only for a bounded window
            # (_SAME_MODEL_QUEUED_HOLD_S) that still idle-expires if the queued
            # request vanishes before it is popped. head_model_tag is a pure,
            # non-destructive peek — no queue mutation, no reentrancy (worker_loop
            # holds no queue lock here). The real swap-churn fix is the queue-side
            # affinity narrowing in pop_next; this only covers the keep_alive=0 edge.
            if idle_seconds <= 0 and self._active_handle is not None:
                try:
                    same_model_queued = (
                        await self.queue.head_model_tag() == slot.model_tag
                    )
                except Exception:
                    log.exception("head_model_tag peek failed (best-effort)")
                    same_model_queued = False
                if same_model_queued:
                    idle_seconds = _SAME_MODEL_QUEUED_HOLD_S
            if idle_seconds > 0 and self._active_handle is not None:
                # Hand off the active handle to the manager-level idle holder.
                # Bug 3: pass thread_id so idle teardown can save KV cache.
                # pass admission_ctx_len so KV save uses admission-time context length.
                # T3: pass client_meta so warm-inherit path can restore session_id (fixes KV cache bin mismatch).
                self._set_idle_holder(
                    self._active_handle,
                    slot.model_tag,
                    time.monotonic() + idle_seconds,
                    slot.thread_id,
                    slot.admission_ctx_len,
                    slot.client_meta,
                )
                await self._audit_event_only_async(
                    slot.slot_id,
                    "idle_hot_enter",
                    {
                        "model_tag": slot.model_tag,
                        "idle_seconds": idle_seconds,
                        # keep-alive handling audit (a review item): visibility into when
                        # client keep_alive overrode the default + when the cap fired.
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
            # fix (closes a high-priority item + a medium-priority item) + design fix:
            # If unwind reaches here with a live handle, the IDLE_HOT
            # entry did NOT complete (most often CancelledError during
            # shutdown or mid-_complete_fn). MUST teardown the handle,
            # not just drop the reference, or llama-server orphans with
            # the full model in VRAM and no parent reference anywhere.
            #
            # design synthesis 2026-05-17:
            # 1. Diagnostic log at entry — surfaces leak path under repro.
            # 2. Do NOT null _active_handle until sigterm SUCCEEDS. If the
            # sigterm helper raises (drained_sigterm internal failure,
            # process already dead/zombie, etc.) leaving _active_handle
            # set lets worker_loop's per-slot exception handler
            # (line 466-481) fire its safety-net _teardown. Otherwise
            # the null-before-success ordering bypassed that safety net.
            handle_to_reap = self._active_handle
            self._set_active_slot(None)
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
            # a tracked issue: removed is_alive gate — TOCTOU: process could die
            # between the check and _sigterm, causing the sigterm to be
            # skipped entirely. _sigterm handles a dead process gracefully;
            # the safety-net _teardown in worker_loop catches any missed reaps.
            if (
                handle_to_reap is not None
                and handle_to_reap is not self._idle_handle
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
                    self._set_active_handle(None)
                # else: keep _active_handle so worker_loop's except handler
                # (which calls _teardown with reason="worker-uncaught-exception")
                # has a second chance to reap. If that ALSO fails, the
                # intra_lifetime_orphan_scan on the next /ensure tick is
                # the final safety net (a high-priority item, singleton.py).
            else:
                # Handle absent or promoted to idle_holder — safe to null.
                self._set_active_handle(None)

    async def _teardown(self, slot: Slot, reason: str) -> None:
        """Drained SIGTERM the process group → VRAM verify → orphan reap → audit."""
        if self._active_handle is not None:
            if self._active_handle.is_alive():
                # SPEC-V2 REWORK R3b: unload-time clean flush (grace-expired with
                # idle disabled / failure teardowns). Real slot available -> call
                # the probe directly. Best-effort; sequenced BEFORE _sigterm.
                await self._probe_and_save_clean_kv(
                    self._active_handle, slot, save_to_disk=True)
                # Option C (U4 seam): mirror the idle-holder teardown's
                # RAM->SSD persist — grace-expired-no-idle / failure teardowns are
                # unload moments too. Keyed by the (session,role[,fp8])
                # _bin_identity string, NOT the raw thread_id; chain source
                # mirrors _save_slot_kv's slot path (admission_hash_chain, falling
                # back to _prefix_hash_chain(client_meta messages) like the flush
                # shim) so the persisted filenames match what was saved.
                _p_cm = getattr(slot, "client_meta", None)
                _p_chain = getattr(slot, "admission_hash_chain", None)
                if not _p_chain:
                    _p_msgs = (_p_cm or {}).get("messages")
                    _p_chain = _prefix_hash_chain(_p_msgs) if _p_msgs else None
                await asyncio.to_thread(
                    self._persist_clean_bin_to_ssd, slot.model_tag or "",
                    _bin_identity(getattr(slot, "thread_id", "") or "", _p_cm, _p_chain),
                    self._active_handle.port,
                )
                await self._save_slot_kv(self._active_handle.port, slot.model_tag, slot)
            ok, status = await self._sigterm(
                self._active_handle,
                drained_window_s=float(self.runtime.queue.drained_sigterm_window_active_s),
                is_active=False,
                cold_window_s=float(self.runtime.queue.drained_sigterm_window_cold_s),
            )
            # fix: dynamic expected_drop_mib derived from manifest
            # expected_vram_bytes. Was hardcoded 1024 MiB — let a 921 MiB
            # drop "verify clear" while 17 GiB a 35B model still resident.
            expected_drop_mib = self._compute_expected_drop_mib(slot.model_tag)
            await self._vram_verify(
                expected_drop_mib=expected_drop_mib, timeout_s=30.0,
            )
            # lifecycle hardening fix: scan for grandchild orphans left behind by
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
            # design review a high-priority item fix: intra-lifetime port-bound reaper. Catches
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
            # the design: slot-write stays on state_db_session; audit-write goes
            # through the pool wrapped in asyncio.to_thread (a review guard sync-only).
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
            # design review critical item + test expectation: clear _active_handle after
            # successful teardown so the outer finally's defensive sigterm
            # net does not double-fire on normal flow. Owner contract:
            # "if you called _teardown you have handed off the handle."
            self._set_active_handle(None)
            # M5 (WIN 2) SWAP-CLEAR (cap<=1): the active sidecar that
            # produced any cached completions is gone -> invalidate them so a later
            # fresh spawn of the same model can NEVER serve an answer from the dead
            # engine instance (results are engine/build-specific).
            self._completion_cache_clear("teardown")

    async def _teardown_idle_holder(self, reason: str) -> None:
        """idle-holder wiring: tear down the manager-level idle handle.

        Called when:
        - a slot for a DIFFERENT model_tag arrives (immediate switch path), or
        - the idle timer expires in the worker_loop, or
        - shutdown.
        """
        if self._idle_handle is None:
            return
        held = self._idle_handle
        # F1: capture aliveness at ENTRY — the flush/sigterm below
        # mutate it, and the VRAM-verify at the tail must know whether a drop is
        # even expected (a DEAD holder freed its VRAM at death; expecting a drop
        # burns the full 30s poll + logs a false "not cleared").
        _held_was_alive = held.is_alive()
        model_tag = self._idle_model_tag
        idle_thread_id = self._idle_thread_id
        idle_admission_ctx_len = self._idle_admission_ctx_len
        idle_client_meta = self._idle_client_meta  # SPEC-V2 REWORK R3: capture BEFORE clear (flush needs messages)
        self._clear_idle_holder()
        # D2 a disposable-owned idle stash never flushes/persists
        # at the seam — kills the junk 35b bins (live receipts 16:02:36Z: owner-
        # mismatch dances on sess:*:sub-agent:* and agent-ip-*-auto-* bins).
        # Single resolve point: _seam_flush_allowed. Skipping HERE (not only at
        # the save) also skips the wasted strip-probe FULL PREFILL of a
        # disposable transcript + the Option C SSD copy + the direct
        # _save_slot_kv + the shadow swap-save. Belt AND enforcement:
        # _save_slot_kv_inner re-checks the same predicate.
        _flush_ok, _idle_role = _seam_flush_allowed(idle_thread_id or "", idle_client_meta)
        if held.is_alive() and not _flush_ok:
            _p_msgs_d2 = (idle_client_meta or {}).get("messages")
            log.warning(
                "unload-seam KV flush SKIPPED (D2 disposable-owned stash): role=%s "
                "thread=%s identity=%s — no bin written/persisted",
                _idle_role, (idle_thread_id or "?")[:60],
                _bin_identity(idle_thread_id or "", idle_client_meta,
                              _prefix_hash_chain(_p_msgs_d2) if _p_msgs_d2 else None))
        # reroute KV save through _save_slot_kv with overrides.
        # This replaces the bolt-on POST that sent {"thread_id": ...} directly.
        # _save_slot_kv will POST {"filename": "..."} to the engine (correct API)
        # and write metadata with prompt_len = idle_admission_ctx_len.
        if held.is_alive() and _flush_ok:
            # SPEC-V2 REWORK R3 (disk-at-unload): THIS is the disk-write
            # moment — grace expiry + swap ("model_swap"), idle/full-timer expiry
            # ("idle_expired"), and "shutdown" all land here. The flush MUST be
            # the strip-probe force_clean form (the VRAM tail holds the just-
            # generated with-<think> turn; a plain action=save would either be
            # skipped by the clean-present guard or persist polluted state).
            # Sequenced BEFORE _sigterm below — the engine must be alive to serve
            # apply-template + prefill + action=save.
            await self._flush_clean_kv_at_unload(
                held, model_tag or "", idle_thread_id or "",
                idle_admission_ctx_len, idle_client_meta,
            )
            # Option C: THE SINGLE SSD WRITE PER SESSION. Copy the final
            # clean bin (+ .ckpt sidecar + meta) from the tmpfs SLOT_SAVE_DIR to the
            # SSD SLOT_PERSIST_DIR so a controlled swap/idle warm-reloads from SSD.
            # A hard container restart legitimately loses the RAM bin (recompute
            # fresh, per the operator). Best-effort; never blocks teardown. Bins are keyed
            # by the (session,role[,fp8]) _bin_identity string, NOT the raw
            # thread_id — mirror the flush shim's derivation exactly (chain =
            # _prefix_hash_chain(stashed idle messages)) so the persisted
            # filenames match what _save_slot_kv just wrote.
            _p_msgs = (idle_client_meta or {}).get("messages")
            await asyncio.to_thread(
                self._persist_clean_bin_to_ssd, model_tag or "",
                _bin_identity(idle_thread_id or "", idle_client_meta,
                              _prefix_hash_chain(_p_msgs) if _p_msgs else None),
                getattr(held, "port", None),
            )
            await self._save_slot_kv(
                held.port,
                model_tag,
                self._active_slot,
                thread_id_override=idle_thread_id,
                admission_ctx_len_override=idle_admission_ctx_len,
                client_meta_override=idle_client_meta,
            )
            # (critical item freshness): while the outgoing sidecar is still
            # ALIVE with its KV populated (BEFORE _sigterm), re-save a FRESH think-free
            # shadow of the outgoing model at the swap seam. INERT unless SHADOW_REPREFILL;
            # no-downgrade + identity-matched inside (never overwrites a fresher per-turn
            # shadow, never a different thread) -> best-effort belt, safe if it no-ops.
            await self._shadow_save_at_swap(held, model_tag, idle_thread_id or "")
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
        # fix: dynamic expected_drop_mib for idle holder teardown.
        # F1: a holder that was DEAD at teardown entry already
        # released its VRAM — expect a 0 drop so the verify returns on the first
        # poll instead of burning 30s on the recovery path (review note).
        expected_drop_mib = (
            self._compute_expected_drop_mib(model_tag or "")
            if _held_was_alive else 0
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
        # design review a high-priority item fix: intra-lifetime port-bound reaper here too.
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
        # PIN#2: dirty-tip describes a LIVE engine's VRAM; this
        # engine is now gone — drop the flag with it (flag lifecycle = engine
        # lifecycle; closes FP F2 stale-flag lost parking).
        _p = getattr(held, "port", None)
        if _p is not None and (getattr(self, "_kv_dirty_tail", {}) or {}).get(_p) is not None:
            _dt = dict(self._kv_dirty_tail)
            _dt.pop(_p, None)
            self._kv_dirty_tail = _dt
            log.info("KV dirty-tip DROPPED (engine teardown): port=%s", _p)
        # the design: audit-only write via pool, wrapped to_thread (a review guard sync-only).
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
        # M5 (WIN 2) SWAP-CLEAR (cap<=1): the idle-held sidecar (the
        # formerly-active model kept warm) is now torn down -> its cached
        # completions are stale (that engine instance is gone). This path does NOT
        # go through _teardown, so clear here too, else a later fresh spawn of the
        # same model could serve a dead-engine answer.
        self._completion_cache_clear("teardown_idle_holder")
        # WIN 4 a model SWAP just happened — this teardown fires on the
        # "different model_tag arrives" switch path (see docstring). Fire a throttled,
        # best-effort fingerprint purge so a reloaded / re-quantized / rebuilt
        # engine's now-stale bins are reclaimed PROMPTLY (not only at the next sweep
        # tick). Fire-and-forget on a bg task (parked in _bg_tasks -> drained at
        # shutdown) + OFF the request path; skipped once _stop_event is set so it
        # never races the shutdown-path call. STRICTLY a file sweep — it never
        # touches the restore decision.
        if not self._stop_event.is_set():
            self._spawn_bg(
                asyncio.to_thread(self._maybe_purge_mismatched_bins, "swap")
            )

    # ============================================================
    # v2: Prefix-containment KV restore (review fix)
    # ============================================================

    @staticmethod
    def _turn_hash(role: str, content) -> str:
        """Deterministic hash of a single conversation turn (full SHA-256)."""
        if isinstance(content, list):
            content = json.dumps(content, sort_keys=True, separators=(',', ':'))
        raw = f"{role}\x00{content}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _thread_hash(thread_id: str) -> str:
        """Safe filename component from thread_id."""
        if not thread_id:
            return "nothread"
        return hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _kv_save_fn(model_tag: str, sid: int, thread_hash: str) -> str:
        return f"{model_tag}.{thread_hash}.slot{sid}.bin"

    @staticmethod
    def _kv_meta_fn(model_tag: str, sid: int, thread_hash: str) -> str:
        return f"{model_tag}.{thread_hash}.slot{sid}.json"

    async def _render_strip_prefill_probe(
        self, port: int, model_tag: str, messages: list, meta: dict,
        *, read_timeout_s: float = 900.0,
    ) -> bool:
        """crit2/crit3 (Fix B, SAVE-ONLY) — prefill the warm slot with the
        HISTORICAL-form prompt so the saved KV byte-matches the harness's future
        think-stripped resend of the covered turns (=> the restore MATCHES+REUSES instead
        of hitting a position-drifted prefix and CLEARing).

        Replaces the plain messages ``/v1/chat/completions`` n_predict=0 probe with:
          1. ``/apply-template`` renders {messages (+ the crit1 tool preamble)} into the
             EXACT prompt string the engine would prefix — INCLUDING the
             ``<think>...</think>\\n\\n`` scaffold the template emits position-based for the
             covered tool turns (those after last_query_index) that drift.
          2.:func:`_strip_think_scaffold` removes every such scaffold, so the prompt equals
             the render those SAME turns get once they fall BEFORE last_query_index next
             resend (historical position -> no scaffold).
          3. native ``/completion`` prefills that raw prompt (n_predict=0, cache_prompt) so
             the warm slot holds the historical-form KV -> the CALLER saves THAT.

        SAVE-ONLY / never-generates: only the two n_predict=0 save probes call this and
        n_predict is HARD-SET to 0, so a live-generation request can never be routed here
        or stripped. Tool knobs are carried verbatim (same _KV_PROBE_TOOL_KNOBS the plain
        probe used) so the rendered tools preamble byte-matches a tools-bearing live
        request (crit1 parity); a tools-less request -> knobs all None -> unchanged.

        Best-effort: returns True iff both engine calls succeeded; False (never raises) on
        any apply-template / completion failure so the caller simply skips the save,
        exactly as the plain probe returns on a failed POST."""
        apply_payload: dict = {"messages": messages, "add_generation_prompt": False}
        for _k in _KV_PROBE_TOOL_KNOBS:
            _v = (meta or {}).get(_k)
            if _v is not None:
                apply_payload[_k] = _v
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=read_timeout_s, write=30.0, pool=2.0)
            ) as client:
                r = await client.post(f"{base}/apply-template", json=apply_payload)
                r.raise_for_status()
                rendered = (r.json() or {}).get("prompt")
                if not isinstance(rendered, str) or not rendered:
                    log.debug("Fix B: /apply-template returned no prompt (best-effort skip)")
                    return False
                stripped = _strip_think_scaffold(rendered, apply_payload.get("messages"))
                # ROOT#2 (Layer 4): strip trailing UNCLOSED generation primer
                # on tool-call turns. The Qwen template emits an open `think` block
                # (no closing `</think>`) when a tool-call turn is CURRENT. The save
                # probe renders at SAVE time (turn is current -> open primer), but the
                # future resend renders HISTORICAL (turn is past -> no primer). If the
                # open primer survives into the saved KV bin, it diverges mid-prefix
                # vs the resend -> n_rs_seq=2 wall -> CLEAR. Strip it HERE so the
                # clean bin ends at the tool_result boundary = true byte-prefix.
                stripped = re.sub(
                    r"(<\|im_start\|>assistant\n)(?:<think>\n)?\s*\Z",
                    r"\1",
                    stripped
                )
                # S0 BYTE-TAIL GATE (load-bearing for the forced warm
                # restore): the clean bin is only a safe force-restore target while
                # its tail is a true byte-prefix of the resend. If a trailing
                # <think> primer SURVIVED the strip (template change / CRLF /
                # multi-token preamble), saving it would arm a mid-prefix divergence
                # > n_rs_seq=2 -> full recurrent CLEAR on the next forced restore =
                # WORSE than native reuse. Refuse the save; the older good bin stays
                # and the warm path degrades safely (warm-no-clean-bin / not-prefix).
                if re.search(r"<think>\s*\Z", stripped):
                    self._kv_s0_tail_fail = getattr(self, "_kv_s0_tail_fail", 0) + 1
                    log.warning(
                        "the classifier S0 gate: unstripped <think> primer at render tail; "
                        "refusing clean-KV save (count=%d) tail=%r",
                        self._kv_s0_tail_fail, stripped[-64:],
                    )
                    return False
                # n_predict HARD-SET to 0: this path only ever prefills, never generates.
                cr = await client.post(
                    f"{base}/completion",
                    json={"prompt": stripped, "n_predict": 0, "cache_prompt": True},
                )
                cr.raise_for_status()
            return True
        except Exception:
            log.debug("Fix B render+strip+prefill probe failed (best-effort)", exc_info=True)
            return False

    def _kvcache_scan_cache_invalidate(self) -> None:
        """Invalidate the SLOT_SAVE_DIR scan cache.

        Call this AFTER EVERY write/delete/rename in SLOT_SAVE_DIR so the next
        scan will re-read the directory. Directory mtime can be unreliable
        (coarse resolution, same-second writes), so we double-invalidate:
        set _kvcache_dir_mtime to None AND clear the cached dicts.
        """
        self._kvcache_dir_mtime = None
        self._kvcache_clean_bins.clear()
        self._kvcache_shadow_bins.clear()

    def _hydrate_ram_from_persist(self, save_dir: str, persist_dir: str) -> None:
        """Option C: hydrate the tmpfs SLOT_SAVE_DIR from the SSD persist
        archive (direction-reversed mirror of _persist_clean_bin_to_ssd). For each
        *.bin / *.json / *.bin.ckpt in persist_dir, copy into save_dir ONLY when
        the RAM copy is ABSENT (RAM always wins while present — it is per-turn
        fresh; SSD is only ever written FROM RAM so it can never be fresher than
        an existing RAM file).
        Atomic via .tmp + os.replace (same filesystem) so a concurrent reader never
        sees a partial. Idempotent + best-effort: never raises; a missing persist
        dir is a no-op (hard restart = recompute fresh, per the operator)."""
        import shutil
        try:
            if not os.path.isdir(persist_dir):
                return
            copied = 0
            for fn in os.listdir(persist_dir):
                if not (fn.endswith(".bin") or fn.endswith(".json")
                        or fn.endswith(".bin.ckpt")):
                    continue
                src = os.path.join(persist_dir, fn)
                dst = os.path.join(save_dir, fn)
                try:
                    # a later phase ROOT-CAUSE FIX: NEVER overwrite an existing
                    # RAM copy. Mid-session the RAM pair is per-turn FRESH while
                    # the SSD archive lags at last-unload/swap state (and may be
                    # R-COMP stale-marked); the size-difference "repair" clobbered
                    # the fresh post-compression anchor back to the stale pair on
                    # EVERY scan (found=None -> native-reuse pinned at the
                    # compression boundary, ~18k reprefill/turn). Saves are atomic
                    # (.tmp+os.replace) so an existing dst is always complete;
                    # SSD is only ever written FROM RAM, so it can never be
                    # fresher than an existing RAM file. Absent-only IS the
                    # documented one-time-per-gap intent.
                    if os.path.exists(dst):
                        continue
                    # Belt (R7 design note): never hydrate ANY member of a stale-marked
                    # triplet (R-COMP marked it dead) — the meta would be skipped
                    # by the finder anyway, and the bin+ckpt are multi-GB dead
                    # weight in the exact tmpfs that ENOSPC'd. Sibling json is
                    # derived per member; missing/corrupt json falls through to
                    # copy (pre-belt behavior).
                    if fn.endswith(".bin.ckpt"):
                        _meta_fn = fn[:-len(".bin.ckpt")] + ".json"
                    elif fn.endswith(".bin"):
                        _meta_fn = fn[:-len(".bin")] + ".json"
                    else:
                        _meta_fn = fn
                    try:
                        with open(os.path.join(persist_dir, _meta_fn)) as _hf:
                            if json.load(_hf).get("stale"):
                                continue
                    except Exception:
                        pass
                    os.makedirs(save_dir, exist_ok=True)
                    tmp = dst + ".tmp"
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
                    copied += 1
                except OSError:
                    log.debug("Option C hydrate: copy failed for %s (best-effort)",
                              fn, exc_info=True)
            if copied:
                log.info("Option C: hydrated %d file(s) SSD persist -> RAM kvcache",
                         copied)
        except Exception:
            log.debug("Option C RAM hydrate best-effort failed", exc_info=True)

    def _scan_kvcache_if_stale(self) -> tuple[
        dict[tuple, tuple[int, str, list, int, int] | None],
        dict[tuple, tuple[int, str, list, int] | None],
    ]:
        """Scan SLOT_SAVE_DIR and return (clean_bins, shadow_bins) dicts.

        Returns cached dicts if the directory mtime has not changed since the
        last scan. If stale or missing, re-scans the directory, populates both
        dicts, updates the mtime, and returns the fresh dicts.

        clean_bins maps (model_tag, thread_hash, port) -> (chain_len, bin_fn, chain, sid, prompt_len)
        or None if no matching bin found for that key.

        shadow_bins maps (model_tag, thread_hash, port) -> (chain_len, bin_fn, chain, sid)
        or None if no matching bin found for that key.
        """
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR, SLOT_PERSIST_DIR
        # Option C: hydrate the tmpfs from the SSD persist archive when a
        # controlled swap/idle emptied RAM but the SSD copy survives. One-time per
        # gap; a hard restart intentionally leaves both empty (recompute fresh).
        if _ram_reanchor_enabled():
            try:
                self._hydrate_ram_from_persist(SLOT_SAVE_DIR, SLOT_PERSIST_DIR)
            except Exception:
                log.debug("Option C RAM hydrate best-effort failed", exc_info=True)
        try:
            dir_mtime = os.path.getmtime(SLOT_SAVE_DIR)
        except OSError:
            dir_mtime = None
        if dir_mtime == self._kvcache_dir_mtime:
            return self._kvcache_clean_bins, self._kvcache_shadow_bins
        # Stale or missing — re-scan
        self._kvcache_clean_bins.clear()
        self._kvcache_shadow_bins.clear()
        try:
            for fn in os.listdir(SLOT_SAVE_DIR):
                if not fn.endswith(".json"):
                    continue
                # Parse sidecar meta
                try:
                    with open(os.path.join(SLOT_SAVE_DIR, fn)) as f:
                        m = json.load(f)
                except (OSError, ValueError):
                    continue
                model_tag = m.get("model_tag") or fn.split(".")[0]
                thread_hash = m.get("thread_hash")
                port = m.get("port")
                if not (model_tag and thread_hash and port):
                    continue
                key = (model_tag, thread_hash, port)
                chain = m.get("hash_chain") or []
                sid = int(m.get("slot_id", 0) or 0)
                chain_len = len(chain)
                prompt_len = int(m.get("prompt_len", 0) or 0)
                bin_fn = fn[:-5] + ".bin"
                _bin_path = os.path.join(SLOT_SAVE_DIR, bin_fn)
                if not os.path.exists(_bin_path):
                    continue
                # SPEC-V2 consistency stamp: meta claims the byte-size of the bin it
                # was written for; mismatch => treat the pair as absent (skip).
                _claim = m.get("bin_bytes")
                if _claim is not None:
                    try:
                        if os.path.getsize(_bin_path) != int(_claim):
                            continue
                    except (OSError, ValueError):
                        continue
                # Check if clean_prefix meta is present (R-COMP: a stale-marked
                # sidecar — compression re-delivery — is treated as absent so
                # _find_clean_bin / probe pre-check / warm gate never see the
                # pre-compression anchor).
                if m.get("clean_prefix") and not m.get("stale"):
                    # Store (chain_len, bin_fn, chain, sid, prompt_len)
                    if key not in self._kvcache_clean_bins:
                        self._kvcache_clean_bins[key] = (chain_len, bin_fn, chain, sid, prompt_len)
                    elif chain_len > self._kvcache_clean_bins[key][0]:
                        self._kvcache_clean_bins[key] = (chain_len, bin_fn, chain, sid, prompt_len)
                # Check if shadow meta is present
                if m.get("shadow"):
                    if key not in self._kvcache_shadow_bins:
                        self._kvcache_shadow_bins[key] = (chain_len, bin_fn, chain, sid)
                    elif chain_len > self._kvcache_shadow_bins[key][0]:
                        self._kvcache_shadow_bins[key] = (chain_len, bin_fn, chain, sid)
        except (FileNotFoundError, OSError):
            pass
        self._kvcache_dir_mtime = dir_mtime
        return self._kvcache_clean_bins, self._kvcache_shadow_bins

    async def _probe_and_save_clean_kv(self, handle, slot, *, save_to_disk: bool = False) -> None:
        """Option A (#103/#87): persist a CLEAN-PREFIX KV (no generated
        <think> tail) so a cold restore REUSES instead of CLEAR+reprefill. On the
        qwen MTP hybrid recurrent ctx (n_rs_seq=2) the generated tail CANNOT be
        trimmed post-hoc (seq_rm aborts/corrupts), so the clean prefix must be
        captured BEFORE generation: fire a prefill-only probe (n_predict=0) DIRECT
        to the sidecar (bypasses admission/curator/restore recursion), then save.
        Best-effort, single-series only. the design plan "save before the turn generates".
        """
        _MIN_CTX_LEN = 40000  # chars (~13k tok): only clean-save substantial contexts
        try:
            _mp = int(getattr(self.runtime.queue, "max_parallel_sidecars", 1) or 1)
            _hp = int(max(1, getattr(handle, "parallel", 1)))
            tid = getattr(slot, "thread_id", "") or ""
            inc_len = int(getattr(slot, "admission_ctx_len", 0) or 0)
            if not getattr(self, "_clean_prefix_save_enabled", True):
                return
            # single-series only: no concurrent eviction between probe, save, real req
            if _mp != 1 or _hp != 1:
                # F7 warn when the single-series gate silently self-disables
                # clean-save on a SUBSTANTIAL ctx — otherwise operators are blind to why
                # NO clean bin was saved (=> every cold restore/wave-return goes FRESH).
                if tid and inc_len >= _MIN_CTX_LEN:
                    log.warning("clean-prefix save SKIPPED: single-series gate (max_parallel_sidecars=%s, handle.parallel=%s != 1) on a %d-char ctx -> no clean bin saved; cold restore/wave-return will FRESH for thread=%s", _mp, _hp, inc_len, (tid or "?")[:12])
                return
            if not tid or inc_len < _MIN_CTX_LEN:
                return
            model_tag = slot.model_tag
            port = handle.port
            client_meta = slot.client_meta or {}
            messages = client_meta.get("messages")
            if not messages:
                return
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            # SPEC-V2 WAVE A: bin key = (session_id, role[, fp8]) from harness
            # labels; raw thread_id when session_id absent OR role unlabeled
            # (RED-HAT design note — never default to main).
            _id_chain = _prefix_hash_chain(messages)
            th = self._thread_hash(_bin_identity(tid, client_meta, _id_chain))
            # Stage 1 (the operator contract): curator/compression/sub-agent KV is
            # DISPOSABLE per-role (toggle ONLY via the harness/front-end client_meta["save_kv"];
            # NO env vars; main always saves). When a disposable role's save is OFF, skip its PER-TURN save
            # (no bin accumulates) but STILL park the OUTGOING main identity (ownership-transfer,
            # same fire-and-forget path as the main flow) + stamp the anchor. Main's OWN save
            # (role=main -> _disp_role None) is UNTOUCHED (no failure-mode guard). Main restores on return via
            # its OWN disk bin (identity owner-match in _restore_slot_kv_inner), NOT this anchor;
            # the stamp only keeps _vram_ours honest so main safe-degrades to disk/native, never
            # a wrong reuse (no failure-mode guard). review note: `not save_to_disk` — NEVER gate the unload/
            # disk-flush seam (save_to_disk=True), so a disposable last-holder cannot suppress
            # main's SSD persist (FM-UNLOAD). Zero-change while the per-role default stays SAVE.
            # `_disp_role not in (None, "main")`: _bin_role returns "main" for an EXPLICIT
            # is_main=true (kv_classify CLASS_MAIN), so main MUST be excluded here or it would
            # be gated (deal-killer regression). None = unlabeled (today's main) — also excluded.
            _disp_role = _bin_role(client_meta)
            # D1 per-port DIRTY-TIP tracker. A non-main role's
            # serve EXTENDS the shared VRAM tip past main's canonical chain, and
            # the MTP hybrid-recurrent ctx cannot trim that tail post-hoc (see
            # docstring above: seq_rm aborts). Track it at THIS single role-
            # resolve point so the unload seam knows the tip is a SUPERSET of the
            # canonical render before trusting a prefix-hit prefill + action=save.
            # Role source = ADMISSION stamp (survives the warm-inherit clobber,
            # PIN#1); fallback = _disp_role for shim/legacy slots. SET on every
            # disposable turn (save_kv toggle irrelevant: the tip is foreign
            # either way); CLEARED on every main/unlabeled turn (the engine
            # reprocessed from the divergence point -> tip canonical again).
            # pid-stamped so a respawned engine on a reused port never inherits a
            # stale flag. Replace-only dict writes (lock-free readers). Per-turn
            # only (`not save_to_disk`) so the seam's own main-labeled shim can
            # never self-clear before the remediation below runs.
            _adm_role = getattr(slot, "admission_role", "__unset__")
            _tip_role = _adm_role if _adm_role != "__unset__" else _disp_role
            if not save_to_disk:
                if _tip_role not in (None, "main"):
                    _dt = dict(getattr(self, "_kv_dirty_tail", {}) or {})
                    _dt[port] = {"role": _tip_role,
                                 "pid": getattr(handle, "pid", None),
                                 "thread": tid, "ts": time.time()}
                    self._kv_dirty_tail = _dt
                    log.info("KV dirty-tip SET: port=%s role=%s thread=%s "
                             "(foreign tokens now on shared VRAM tip)",
                             port, _tip_role, (tid or "?")[:40])
                elif (getattr(self, "_kv_dirty_tail", {}) or {}).get(port) is not None:
                    _dt = dict(self._kv_dirty_tail)
                    _dt.pop(port, None)
                    self._kv_dirty_tail = _dt
                    log.info("KV dirty-tip CLEARED: port=%s thread=%s "
                             "(main serve reprocessed the tip)", port, (tid or "?")[:40])
            if (not save_to_disk and _disp_role not in (None, "main")
                    and not _role_save_enabled(_disp_role, client_meta)):
                _old_disp = self._kv_vram_anchor.get(port)
                if (_old_disp and _old_disp.get("thread_hash")
                        and _old_disp.get("thread_hash") != th
                        and _old_disp.get("model_tag")):
                    self._spawn_bg(asyncio.to_thread(
                        self._persist_clean_bin_to_ssd_by_hash,
                        _old_disp.get("model_tag") or "", _old_disp.get("thread_hash") or "",
                        port))
                self._kv_vram_anchor[port] = {
                    "thread_hash": th, "model_tag": model_tag,
                    "pid": getattr(handle, "pid", None), "chain": list(_id_chain),
                    "prompt_len": inc_len, "stamp": time.time(),
                }
                log.info(
                    "clean-prefix save GATED (disposable role=%s, per-role save OFF) "
                    "— no bin saved, outgoing main parked; thread=%s", _disp_role, (tid or "?")[:40])
                return
            # (the operator verbatim 2026-07-10 05:45Z: "KV Save should happen
            # right before the main model is unloaded NOT at the time of sub agent
            # spawn in the cue"): MAIN's per-turn clean-save is OFF — the single KV
            # writer is the unload seam (save_to_disk=True via _flush_clean_kv_at_unload,
            # which re-renders the historical transcript = clean by construction; its
            # "clean bin present" skip-guard now finds none, so it RUNS at unload).
            # AUTOPSY (wf_2fff2359): the poisoned bin was created by per-turn probe
            # saves capturing post-curator/post-rewind engine state as clean=True —
            # this closes that window. Park + anchor stamp preserved (mirrors the
            # disposable gate above) so _vram_ours stays honest; warm tool-call reuse
            # (the deal-killer fix) is native-VRAM and never depended on these saves.
            if not save_to_disk:
                _old_m = self._kv_vram_anchor.get(port)
                if (_old_m and _old_m.get("thread_hash")
                        and _old_m.get("thread_hash") != th
                        and _old_m.get("model_tag")):
                    self._spawn_bg(asyncio.to_thread(
                        self._persist_clean_bin_to_ssd_by_hash,
                        _old_m.get("model_tag") or "", _old_m.get("thread_hash") or "",
                        port))
                self._kv_vram_anchor[port] = {
                    "thread_hash": th, "model_tag": model_tag,
                    "pid": getattr(handle, "pid", None), "chain": list(_id_chain),
                    "prompt_len": inc_len, "stamp": time.time(),
                }
                # NOTE (review note): this stamp is no longer coupled to a prefill+
                # save, so the Wave-K natural-skip proof (anchor.chain == bin.chain
                # => byte-match) has lost its provenance. _warm_natural_skip_enabled
                # ships default OFF; if that env is ever flipped, re-derive the proof
                # before trusting the skip.
                log.debug(
                    "per-turn clean-save OFF (unload-seam-only) — anchor "
                    "stamped; thread=%s", (tid or "?")[:40])
                return
            # D1 SEAM REMEDIATION — save_to_disk=True (unload/
            # teardown seam). If a disposable role extended this port's tip, the
            # strip-probe's cache_prompt prefill of the SHORTER canonical render is
            # a pure prefix-hit: the engine reuses and CANNOT trim the foreign
            # tail, so action=save would persist tip-tokens > canonical-chain
            # tokens (bin != stamped hash_chain = the wave-return strict-extension
            # abort common.cpp:1498 x3 -> 3-strikes -> fresh slow-load, live
            # 2026-07-10 16:05-16:09Z). Fix: ERASE the whole sequence (whole-seq
            # rm is legal on the recurrent ctx; only mid-seq trims abort), then
            # the UNCHANGED strip-probe below FULL-reprefills the canonical render
            # onto the empty slot -> bin == chain by construction. One-time seam
            # cost only when a disposable rode main's VRAM. Erase failure -> flag
            # stays -> _save_slot_kv_inner refuses EVERY save on this port
            # (previous good bin kept; wave-return pays a delta reprefill instead
            # of a poisoned restore).
            _dirty = (getattr(self, "_kv_dirty_tail", {}) or {}).get(port)
            if _dirty and _dirty.get("pid") not in (None, getattr(handle, "pid", None)):
                # F2: stale record from a previous engine on a reused port — POP it
                _dt = dict(self._kv_dirty_tail)
                _dt.pop(port, None)
                self._kv_dirty_tail = _dt
                log.info("KV dirty-tip DROPPED (stale pid, engine respawned): port=%s", port)
                _dirty = None
            if _dirty:
                # F3: no remediation when the seam identity ITSELF is disposable —
                # its save is refused at the chokepoint anyway; flag drops via
                # stale-pop / engine-teardown pop.
                _flush_ok_seam, _seam_role = _seam_flush_allowed(tid, client_meta)
                if not _flush_ok_seam:
                    log.info("dirty-tip remediation SKIPPED (seam identity disposable "
                             "role=%s) — saves refused by chokepoint; port=%s",
                             _seam_role, port)
                    _dirty = None
            if _dirty:
                log.warning(
                    "unload-seam DIRTY TIP: port=%s foreign role=%s (via thread=%s) — "
                    "erase + full canonical reprefill before save; identity=%s thread_hash=%s",
                    port, _dirty.get("role"), str(_dirty.get("thread"))[:40],
                    _bin_identity(tid, client_meta, _id_chain), th)
                _erased = False
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(connect=2.0, read=60.0, write=10.0, pool=2.0)
                    ) as _ec:
                        _sl = await _ec.get(f"http://127.0.0.1:{port}/slots")
                        _sl.raise_for_status()
                        _pop = [s for s in (_sl.json() or [])
                                if (s.get("n_prompt_tokens") or 0) > 0
                                and s.get("id") is not None]
                        if len(_pop) == 1:
                            _er = await _ec.post(
                                f"http://127.0.0.1:{port}/slots/{int(_pop[0]['id'])}?action=erase")
                            _er.raise_for_status()
                            _erased = True
                except Exception:
                    log.debug("dirty-tip erase attempt failed (best-effort)", exc_info=True)
                if not _erased:
                    log.warning(
                        "dirty-tip erase FAILED — unload-seam save REFUSED, previous bin "
                        "kept (port=%s identity=%s thread_hash=%s)",
                        port, _bin_identity(tid, client_meta, _id_chain), th)
                    return
                _dt = dict(self._kv_dirty_tail)
                _dt.pop(port, None)
                self._kv_dirty_tail = _dt
                log.info(
                    "dirty-tip ERASED: port=%s — canonical %d-turn chain will FULL-"
                    "reprefill (one-time seam cost) thread_hash=%s",
                    port, len(_id_chain), th)
            # lag-reducer THROTTLE (a failure-mode guard). Source the incoming clean-
            # prefix turn count from the SAME structured messages the probe below
            # prefills, so it matches the n_context_turns the force_clean save will
            # stamp (_save_slot_kv derives its hash_chain from these same admission
            # messages when slot.context is empty).
            inc_turns = len(_id_chain)
            # a failure-mode guard: never force a save on a degenerate empty chain (0 turns). Belt
            # for a future/edge messages value that is truthy but yields no chain (the
            # `if not messages` guard above already covers the empty-list case).
            if inc_turns <= 0:
                return
            # Skip if we already hold a clean bin for this thread whose prefix >= current
            # (never-overwrite-with-smaller belt), else remember the anchor's grown turn
            # count (longest clean chain) so the throttle can bound the re-save cadence.
            saved_clean_turns = None
            saved_clean_prompt_len = None
            try:
                # Use the scan cache — no O(N) listdir loop.
                clean_bins, _ = self._scan_kvcache_if_stale()
                key = (model_tag, th, port)
                best = clean_bins.get(key)
                if best is not None:
                    # best = (chain_len, bin_fn, chain, sid, prompt_len)
                    # PL v3 defense-in-depth: recheck bin exists before trusting cache
                    # (closes same-mtime-tick window where GC invalidated but cache
                    # wasn't refreshed yet, and finders' exists-recheck doesn't cover
                    # this direct cache read).
                    bin_fn = best[1]
                    if not os.path.exists(os.path.join(SLOT_SAVE_DIR, bin_fn)):
                        self._kvcache_scan_cache_invalidate()
                        best = None
                    else:
                        saved_clean_prompt_len = best[4]
                        if saved_clean_prompt_len >= inc_len:
                            return  # never-overwrite-with-smaller belt
                        saved_clean_turns = best[0]
            except Exception:
                log.debug("clean-prefix pre-check failed (best-effort)", exc_info=True)
            # THROTTLE: a clean anchor already exists -> re-save (multi-GB) ONLY when the
            # clean-prefix chain grew by >= LAGREDUCER_MIN_GROWTH_TURNS turns. Re-saving
            # every +1 turn burns the request critical path; too rarely widens the
            # sawtooth lag. This bounds the lag to ~MIN_GROWTH_TURNS turns.
            _min_growth = 1 if save_to_disk else self.LAGREDUCER_MIN_GROWTH_TURNS
            if (saved_clean_turns is not None
                    and (inc_turns - saved_clean_turns) < _min_growth):
                log.debug(
                    "clean-prefix save THROTTLED: incoming_turns=%d saved n_context_turns=%d "
                    "(growth < %d) thread_hash=%s",
                    inc_turns, saved_clean_turns, self.LAGREDUCER_MIN_GROWTH_TURNS, th)
                return
            if _covered_scaffold_strip_enabled():
                # crit2/crit3 (Fix B): prefill the HISTORICAL-form prompt (render
                # -> strip the covered-turn <think>...</think>\n\n scaffolds -> /completion)
                # so the saved clean bin byte-matches the harness's future think-stripped
                # resend of those covered turns. A failed engine call returns False -> skip
                # the save, exactly as the plain probe returns on a failed POST.
                if not await self._render_strip_prefill_probe(
                    port, model_tag, messages, client_meta
                ):
                    return
            else:
                # Flag OFF -> pre-Fix-B behavior, byte-identical to today: plain messages
                # /v1/chat/completions n_predict=0 probe.
                payload = {
                    "model": model_tag,
                    "messages": messages,
                    "stream": False,
                    "n_predict": 0,
                    "max_tokens": 0,
                    "cache_prompt": True,
                }
                # crit1: carry the live request's tool preamble (verbatim from client_meta)
                # so the saved clean bin's front-of-prompt byte-matches a tools-bearing
                # request; None when the request has no tools -> payload unchanged.
                for _k in _KV_PROBE_TOOL_KNOBS:
                    _v = client_meta.get(_k)
                    if _v is not None:
                        payload[_k] = _v
                url = f"http://127.0.0.1:{port}/v1/chat/completions"
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(connect=2.0, read=900.0, write=30.0, pool=2.0)
                    ) as client:
                        await client.post(url, json=payload)
                except Exception:
                    log.debug("clean-prefix probe POST failed (best-effort)", exc_info=True)
                    return
            # Slot now holds the clean prefix [prompt (+<=1 uncommitted tok)].
            # Option C (per-turn RAM re-anchor): the probe's concurrent
            # n_predict:0 prefill CANNOT occupy the --parallel 1 slot (it is
            # cancelled unlaunched), so it never re-anchors the engine restore-base
            # (frozen@130514 -> growing reprefill). Restore Phase-1's proven
            # round-trip, but to the tmpfs SLOT_SAVE_DIR (RAM, zero SSD wear):
            # action=save the clean prefix, then action=restore it -> the restore
            # SYNCHRONOUSLY re-materializes the slot from the clean state and
            # re-commits the engine's restorable checkpoint at the GROWN clean
            # position, so the NEXT turn's get_common_prefix strict-extends
            # (delta-only) instead of diverging at the with-<think> boundary.
            #
            # tier rule (GAP-1, ownership transfer): another
            # identity takes the slot => the OUTGOING agent's RAM copy goes to
            # SSD. BEFORE the anchor record is overwritten with the NEW identity,
            # schedule a best-effort background persist of the OLD identity's
            # bins (RAM->SSD, keyed by the anchor's already-hashed thread_hash).
            # Off the request critical path (_spawn_bg + to_thread); never blocks.
            _old = self._kv_vram_anchor.get(port)
            if (_old and _old.get("thread_hash") and _old.get("thread_hash") != th
                    and _old.get("model_tag")):
                self._spawn_bg(asyncio.to_thread(
                    self._persist_clean_bin_to_ssd_by_hash,
                    _old.get("model_tag") or "", _old.get("thread_hash") or "",
                    port))
            self._kv_vram_anchor[port] = {
                "thread_hash": th,
                "model_tag": model_tag,
                "pid": getattr(handle, "pid", None),
                "chain": list(_id_chain),
                "prompt_len": inc_len,
                "stamp": time.time(),
            }
            # Per-turn RAM save (SLOT_SAVE_DIR is tmpfs). force_clean=True keeps the
            # existing (session,role[,fp8]) keying, never-overwrite-with-smaller
            # belt, never-demote invariant, and confirmed-save logging INTACT.
            _saved = await self._save_slot_kv(port, model_tag, slot, force_clean=True)
            if not _saved:
                # SPEC-V2 consistency stamp: NEVER log saved/GREW on an unconfirmed
                # save — the pre-Phase-1 era unconditionally logged GREW 29..49 while
                # the on-disk meta was honest at 25 (the '49-turn meta' misreport).
                # FP design note (a later phase gate): ROLL BACK the optimistic anchor stamp —
                # leaving it AHEAD of the on-disk bin makes the a later phase
                # skip-redundant-restore chain check miss on every later turn
                # (anchor chain != disk chain), permanently re-arming the
                # restore->M-RoPE-crash path this wave exists to end.
                if _old is not None:
                    self._kv_vram_anchor[port] = _old
                else:
                    self._kv_vram_anchor.pop(port, None)
                log.warning("clean-prefix KV save NOT CONFIRMED (thread=%s, ctx_len=%d) — anchor rolled back to pre-stamp value", tid[:12], inc_len)
                return
            log.info("clean-prefix KV saved (thread=%s, ctx_len=%d)", tid[:12], inc_len)
            # OBSERVABILITY (reuse existing n_context_turns; NO new clean_turns
            # field per PL). Surface the clean-bin growth so the sawtooth lag is
            # measurable from logs. n_context_turns == incoming_turns here by
            # construction (both = len(_prefix_hash_chain(messages)) = the chain the save
            # just stamped); `was` is the prior anchor (None on first clean save).
            log.info(
                "clean-bin GREW: n_context_turns=%d (was %s) incoming_turns=%d thread_hash=%s",
                inc_turns,
                ("none" if saved_clean_turns is None else saved_clean_turns),
                inc_turns, th)
        except Exception:
            log.debug("clean-prefix probe+save best-effort failed", exc_info=True)

    async def _flush_clean_kv_at_unload(
        self, handle, model_tag: str, thread_id: str,
        admission_ctx_len: int, client_meta: dict | None,
    ) -> None:
        """SPEC-V2 REWORK R3 (disk-at-unload): unload-time DISK flush of the
        clean KV. Reuses _probe_and_save_clean_kv(save_to_disk=True) so the flush
        is the SAME proven strip-probe (render think-stripped -> S0 gate ->
        prefill -> force_clean save) and inherits every guard (min-ctx, identity
        keying incl. the sub-agent fp8 chain, never-overwrite-with-smaller,
        zero-growth dedupe, confirmed-save logging). The idle holder has no live
        Slot object at this seam, so a SimpleNamespace shim carries the stashed
        idle identity fields. admission_hash_chain is recomputed from the stashed
        messages — the SAME derivation the probe/save use — so the (session,
        role[, fp8]) bin identity matches the per-turn key exactly. Best-effort;
        the engine handle must still be alive (call BEFORE _sigterm)."""
        try:
            messages = (client_meta or {}).get("messages")
            if not model_tag or not thread_id or not messages:
                return
            shim = SimpleNamespace(
                thread_id=thread_id,
                model_tag=model_tag,
                admission_ctx_len=admission_ctx_len,
                admission_hash_chain=_prefix_hash_chain(messages),
                client_meta=client_meta,
                context=None,
                prompt="",
                port=getattr(handle, "port", None),
                pid=getattr(handle, "pid", None),
            )
            await self._probe_and_save_clean_kv(handle, shim, save_to_disk=True)
        except Exception:
            log.debug("unload-time clean KV flush best-effort failed", exc_info=True)

    def _persist_clean_bin_to_ssd(self, model_tag: str, thread_id: str, port) -> None:
        """Option C: copy the tmpfs clean bin (+.ckpt+meta) to the SSD
        persist dir at unload. The ONLY SSD write per session. shutil.copy2
        preserves mtime; atomic via .tmp+os.replace so a reader never sees a
        partial. Best-effort — any error leaves the RAM bin as the live source and
        simply forgoes cross-restart durability (safe).

        ``thread_id`` MUST be the (session,role[,fp8]) ``_bin_identity`` string,
        NOT the raw thread_id — _save_slot_kv keyed the filenames on
        _thread_hash(_bin_identity(...)), so hashing anything else here would
        filter for filenames that were never written."""
        try:
            if not (model_tag and thread_id and port):
                return
            self._persist_clean_bin_to_ssd_by_hash(
                model_tag, self._thread_hash(thread_id), port)
        except Exception:
            log.debug("Option C RAM->SSD persist best-effort failed", exc_info=True)

    def _quarantine_bin_triplet(self, bin_fn: str) -> None:
        """C (3-strikes design): move a repeat-offender bin's
        triplet (.bin/.bin.ckpt/.json) out of BOTH tiers into kv_quarantine so the
        next load goes FRESH (the manual 2026-07-10 loop-break, automated). The
        evidence is preserved for autopsy, never deleted. Sync + best-effort —
        runs via _spawn_bg/to_thread off the hot path."""
        import shutil
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR, SLOT_PERSIST_DIR
        qdir = "/var/lib/turbohaul/kv_quarantine"
        base = bin_fn[:-4] if bin_fn.endswith(".bin") else bin_fn
        try:
            os.makedirs(qdir, exist_ok=True)
        except Exception:
            log.exception("quarantine dir create failed")
            return
        moved = 0
        for d in (SLOT_SAVE_DIR, SLOT_PERSIST_DIR):
            for suf in (".bin", ".bin.ckpt", ".json"):
                src = os.path.join(d, base + suf)
                if os.path.exists(src):
                    try:
                        shutil.move(src, os.path.join(qdir, base + suf))
                        moved += 1
                    except Exception:
                        log.exception("quarantine move failed: %s", src)
        self._kvcache_scan_cache_invalidate()
        log.error("QUARANTINED %d files for %s -> %s", moved, base, qdir)

    def _persist_clean_bin_to_ssd_by_hash(self, model_tag: str, th: str, port) -> None:
        """Option C (+ GAP-1 tier rule — ownership transfer): same
        as _persist_clean_bin_to_ssd but takes the ALREADY-HASHED thread hash.
        The _kv_vram_anchor record stores thread_hash (not the identity string),
        so the ownership-transfer seam can only key by hash. Best-effort; never
        raises."""
        import shutil
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR, SLOT_PERSIST_DIR
        try:
            if not (model_tag and th and port):
                return
            os.makedirs(SLOT_PERSIST_DIR, exist_ok=True)
            copied = 0
            for fn in os.listdir(SLOT_SAVE_DIR):
                # only THIS identity's bin/ckpt/meta (session,role keyed filename)
                if not (fn.startswith(f"{model_tag}.")
                        and f".p{port}.{th}." in fn):
                    continue
                if fn.endswith(".tmp"):
                    continue  # never persist a mid-write temp
                src = os.path.join(SLOT_SAVE_DIR, fn)
                dst = os.path.join(SLOT_PERSIST_DIR, fn)
                tmp = dst + ".tmp"
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                copied += 1
            if copied:
                log.info(
                    "Option C: clean bin persisted RAM->SSD (%d file(s), "
                    "thread_hash=%s, port=%s)", copied, th[:12], port)
        except Exception:
            log.debug("Option C RAM->SSD persist best-effort failed", exc_info=True)

    def _shadow_lock_for(self, port: int, thread_hash: str) -> "asyncio.Lock":
        """Per-(port, thread_hash) shadow save-in-flight lock (see __init__). Creates
        it on first use. Opportunistically prunes UNHELD locks past the cap so the
        dict cannot leak across many distinct threads — a currently-HELD lock is
        never evicted (the single-series inline await holds at most one at a time)."""
        key = (port, thread_hash)
        locks = self._shadow_reprefill_locks
        lock = locks.get(key)
        if lock is None:
            if len(locks) >= _SHADOW_BYTEMATCH_CAP:
                for k in [k for k, lk in locks.items() if not lk.locked()]:
                    del locks[k]
            lock = locks.setdefault(key, asyncio.Lock())
        return lock

    async def _shadow_reprefill_and_save(self, handle, slot, result) -> None:
        """Option A (SAVE-side): after a MAIN generation, advance the warm
        slot to the THINK-FREE end-of-turn-N state and persist it under a DISTINCT
        `.shadow` bin, so a LATER restore (PL's SEPARATE restore-preference step)
        strict-extends instead of CLEAR+reprefilling.

        The qwen MTP hybrid recurrent ctx (n_rs_seq=2) bakes the generated <think>
        into the saved KV and CANNOT be trimmed post-hoc (seq_rm aborts/corrupts);
        the harness resends history think-STRIPPED next turn, so the saved state
        diverges at the generated <think> -> do_reset -> against-3 CLEAR. Fix: fire a
        prefill-only probe (n_predict=0) of the think-free history against the current
        warm slot — the LCP diverges at the just-generated <think> so the engine
        do_resets and fully reprefills the think-free prefix — then save THAT.

        Fired at the 4 post-completion hooks (streaming/non-streaming x anchor/active-
        match) AFTER the client already has the answer (set_result) => ZERO added
        response TTFT. ``result`` is the completion dict (non-streaming) or None
        (streaming; the with-<think> answer then comes from slot.streamed_assistant_
        text) — the SAME source the byte-match self-check + _engine_view_chain use.

        SAVE-ONLY, defense-in-depth:
        - INERT unless env TURBOHAUL_SHADOW_REPREFILL (ships off).
        - single-series gate ONLY (mirrors _probe_and_save_clean_kv): no concurrent
          eviction/fan-out between the reprefill, the save, and the next request.
        - DISTINCT `.shadow` bin + `shadow:true` meta: the clean_prefix anchor is left
          COMPLETELY intact (never routed through the never-demote clean path; a
          mispredicted shadow can never overwrite/demote the proven anchor).
        - the restore gate is NOT touched — the shadow bin is WRITTEN, not
          consumed here.
        - best-effort: NEVER raises into _process_slot, never delays the response."""
        try:
            if not _shadow_reprefill_enabled():
                return
            # OFF-PATH: single-series only (same gate _probe_and_save_clean_kv uses).
            _mp = int(getattr(self.runtime.queue, "max_parallel_sidecars", 1) or 1)
            _hp = int(max(1, getattr(handle, "parallel", 1)))
            if _mp != 1 or _hp != 1:
                return
            tid = getattr(slot, "thread_id", "") or ""
            if not tid:
                return
            c = self._shadow_reprefill_counts
            # --- GUARDS (mirror _record_shadow_bytematch_probe EXACTLY): fire only on a
            # positively-reconstructable, think-bearing, non-empty text turn. ------
            if result is None:
                # STREAMING: the SSE route stashed the merged <think> text on the slot.
                with_think = getattr(slot, "streamed_assistant_text", None)
            else:
                # NON-STREAMING: None => tool-call / noop / empty (not reconstructable).
                gen = self._generated_assistant_msg(result)
                if gen is None:
                    c["skipped_toolcall"] = c.get("skipped_toolcall", 0) + 1
                    log.info("SHADOW_SAVE th=%s model=%s slot=%s outcome=skipped:toolcall",
                             self._thread_hash(tid), getattr(slot, "model_tag", ""),
                             getattr(slot, "slot_id", None))
                    return
                with_think = gen.get("content")
            if not isinstance(with_think, str) or not with_think:
                c["skipped_empty"] = c.get("skipped_empty", 0) + 1
                log.info("SHADOW_SAVE th=%s model=%s slot=%s outcome=skipped:empty",
                         self._thread_hash(tid), getattr(slot, "model_tag", ""),
                         getattr(slot, "slot_id", None))
                return
            if "</think>" not in with_think:
                # (critical item freshness): a no-think turn is ALREADY
                # think-free -> it is a VALID (and often the FRESHEST) shadow, especially
                # the last main turn before a model swap. Do NOT skip it: _strip_thinking_
                # all is a whitespace-only no-op here, so the save still yields the correct
                # think-free end-of-turn-N state that the harness will resend. (The tool-
                # call / empty / empty-after-strip / no-messages guards STILL skip below —
                # only the </think>-required gate is relaxed.) Count for observability.
                c["no_think_saved"] = c.get("no_think_saved", 0) + 1
            # Mirror the harness's remove-ALL think-strip so the shadow prefix byte-
            # matches the harness's think-stripped resend next turn (shadow_bytematch =
            # MATCH) — including multi-block / pre-<think> content, where the old
            # rsplit-last _strip_thinking_wrapper diverged (thinkstrip-
            # multiblock). Lazy import avoids a manager<->api cycle.
            from turbohaul.api.chat_completion import _strip_thinking_all
            think_free = _strip_thinking_all(with_think)
            if not think_free:
                c["skipped_empty"] = c.get("skipped_empty", 0) + 1
                log.info("SHADOW_SAVE th=%s model=%s slot=%s outcome=skipped:empty_after_strip",
                         self._thread_hash(tid), getattr(slot, "model_tag", ""),
                         getattr(slot, "slot_id", None))
                return
            # messages = the harness's OWN bytes for turns 1..N + the predicted think-
            # free assistant-N (== exactly what the harness resends next turn, minus
            # the not-yet-known user turn N+1).
            base_msgs = (getattr(slot, "client_meta", None) or {}).get("messages")
            if not base_msgs:
                c["skipped_no_messages"] = c.get("skipped_no_messages", 0) + 1
                log.info("SHADOW_SAVE th=%s model=%s slot=%s outcome=skipped:no_messages",
                         self._thread_hash(tid), getattr(slot, "model_tag", ""),
                         getattr(slot, "slot_id", None))
                return
            messages = list(base_msgs) + [{"role": "assistant", "content": think_free}]

            model_tag = slot.model_tag
            port = handle.port
            # (critical item freshness): record the freshest think-free shadow
            # SOURCE for the model-swap teardown belt (_shadow_save_at_swap). Set BEFORE
            # the reprefill/save POST so a silently-failed save here can still be recovered
            # at the swap seam. Manager-level (not on the slot) so it tracks the LATEST
            # turn across anchor + grace-follow-up (active-match) slots — the outgoing slot
            # object is nulled at _process_slot finally, so a per-slot ref would go stale.
            # Keyed by identity so the teardown never re-saves a DIFFERENT thread's source.
            _ls_key = (tid, model_tag)
            if len(self._last_shadow_src) >= 64 and _ls_key not in self._last_shadow_src:
                self._last_shadow_src.popitem(last=False)
            self._last_shadow_src[_ls_key] = {
                "thread_id": tid, "model_tag": model_tag, "port": port,
                "messages": messages,
            }
            # (adversarial-verify hardening): move_to_end so the bounded
            # evict-oldest is TRUE LRU-by-use — a re-saving thread must not sit at
            # position 0 and get popitem(last=False)-evicted while actively fresh.
            self._last_shadow_src.move_to_end(_ls_key)
            # shadow-diag (INSTRUMENTATION-ONLY): also stash the think-free
            # recon PER (thread_id, model_tag) so SHADOW_BYTEPARITY on a swap-back reads
            # THIS thread's source instead of a clobbering intervening save's. Bounded
            # (evict-oldest); best-effort; does NOT touch _last_shadow_src or any logic.
            try:
                _bpk = self._byteparity_recon_by_key
                if len(_bpk) >= 64 and (tid, model_tag) not in _bpk:
                    _bpk.pop(next(iter(_bpk)))
                _bpk[(tid, model_tag)] = think_free
            except Exception:
                pass
            th = self._thread_hash(tid)
            # Serialise the whole shadow WORK for THIS (port, thread) vs the series so a
            # save can't overlap the next turn reading a half-written shadow bin.
            async with self._shadow_lock_for(port, th):
                # 1. n_predict=0 SHADOW-PREFILL of the think-free messages against the
                # current warm slot (Option A). SAME POST mechanism / params /
                # timeouts as _probe_and_save_clean_kv's clean probe.
                _sh_meta = getattr(slot, "client_meta", None) or {}
                if _covered_scaffold_strip_enabled():
                    # crit2/crit3 (Fix B): render the shadow messages -> strip the
                    # covered-turn <think>...</think>\n\n scaffolds -> /completion prefill, so
                    # the .shadow bin byte-matches the harness's future think-stripped resend
                    # (including the empty scaffold the template emits for the appended
                    # think-free turn N in its still-current position). A failed engine call
                    # -> skip the save, exactly as the plain probe returns on a failed POST.
                    if not await self._render_strip_prefill_probe(
                        port, model_tag, messages, _sh_meta
                    ):
                        c["reprefill_post_failed"] = c.get("reprefill_post_failed", 0) + 1
                        log.info("SHADOW_SAVE th=%s model=%s slot=%s outcome=post_failed turns=%d",
                                 th, model_tag, getattr(slot, "slot_id", None), len(messages))
                        return
                else:
                    # Flag OFF -> pre-Fix-B behavior, byte-identical to today.
                    payload = {
                        "model": model_tag,
                        "messages": messages,
                        "stream": False,
                        "n_predict": 0,
                        "max_tokens": 0,
                        "cache_prompt": True,
                    }
                    # crit1: same tool-preamble parity as the clean probe — the shadow bin
                    # must also match a tools-bearing request's front-of-prompt (else the
                    # cold/forced restore of the shadow CLEARs on every tool turn).
                    for _k in _KV_PROBE_TOOL_KNOBS:
                        _v = _sh_meta.get(_k)
                        if _v is not None:
                            payload[_k] = _v
                    url = f"http://127.0.0.1:{port}/v1/chat/completions"
                    try:
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(connect=2.0, read=900.0, write=30.0, pool=2.0)
                        ) as client:
                            await client.post(url, json=payload)
                    except Exception:
                        c["reprefill_post_failed"] = c.get("reprefill_post_failed", 0) + 1
                        log.debug("shadow reprefill POST failed (best-effort)", exc_info=True)
                        log.info("SHADOW_SAVE th=%s model=%s slot=%s outcome=post_failed turns=%d",
                                 th, model_tag, getattr(slot, "slot_id", None), len(messages))
                        return
                # 2. SAVE the now-think-free slot under the DISTINCT .shadow marker.
                if await self._save_shadow_slot_kv(port, model_tag, tid, messages):
                    c["saved"] = c.get("saved", 0) + 1
                    log.info("shadow-reprefill KV saved (thread=%s, turns=%d)",
                             tid[:12], len(messages))
                    # shadow-diag: per-call SAVED outcome — think_free_hash is
                    # computed here in the OFF-hot-path save code (this whole method runs
                    # AFTER set_result), NEVER on the response TTFT path. path/size ride
                    # the SHADOW_SAVE write=ok line inside _save_shadow_slot_kv.
                    log.info(
                        "SHADOW_SAVE th=%s model=%s slot=%s outcome=saved turns=%d "
                        "think_free_len=%d think_free_hash=%s",
                        th, model_tag, getattr(slot, "slot_id", None), len(messages),
                        len(think_free), _fnv1a_64(think_free))
        except Exception:
            log.debug("shadow reprefill+save best-effort failed", exc_info=True)

    async def _shadow_save_at_swap(self, handle, model_tag: str, thread_id: str) -> None:
        """(critical item freshness): re-save a FRESH think-free shadow of
        the OUTGOING slot at the model-swap teardown seam — as close to SIGTERM as
        possible so swap-back strict-extends the byte-matching think-free [1..N] state
        instead of a fresh full prefill.

        WHY read the manager field, not the slot (a real finding): at
        _teardown_idle_holder the outgoing slot OBJECT is already gone (POPPED ->
        _set_active_slot(None) in _process_slot's finally), and the idle holder retains
        only handle/model_tag/thread_id/ctx_len — NOT client_meta.messages. So this reads
        the manager-level ``_last_shadow_src`` (the freshest think-free ``messages``
        recorded by the last per-turn _shadow_reprefill_and_save — which also correctly
        tracks the LATEST grace-follow-up turn, not the stale anchor slot).

        SAFE by construction (constraints #2 + #5):
        - INERT unless TURBOHAUL_SHADOW_REPREFILL (same SAVE gate as the per-turn hook).
        - identity-matched: only re-saves the source recorded for THIS outgoing
          (thread_id, model_tag) — never guesses / never cross-thread.
        - NO-DOWNGRADE: if a shadow with an EQUAL-OR-LONGER chain already exists (the
          per-turn hook already saved the freshest, or a later grace turn extended it),
          it does NOTHING — a stale/shorter source can NEVER overwrite a fresher shadow.
          So it only ADDS value when the last per-turn save POST silently failed
          (existing shadow shorter/absent), recovering it right at the swap seam.
        - single-bin overwrite via _save_shadow_slot_kv's os.replace (constraint #5, NO
          ring); serialized under the same per-(port,thread) shadow lock.
        - best-effort: never raises into teardown; in the COMMON case the no-downgrade
          check short-circuits BEFORE any reprefill POST, so it adds ZERO swap latency
          (a reprefill only runs in the rare failed-save recovery).

        Reuses the exact reprefill(n_predict=0)+save mechanism as _shadow_reprefill_and_
        save, so a bin it writes is byte-identical to what the per-turn path would."""
        try:
            if not _shadow_reprefill_enabled():
                return
            src = self._last_shadow_src.get((thread_id, model_tag))
            if not src:
                return  # no fresh source for THIS outgoing thread -> skip (never guess)
            messages = src.get("messages")
            if not messages:
                return
            port = handle.port
            new_len = len(_prefix_hash_chain(messages))
            if new_len <= 0:
                return
            c = self._shadow_reprefill_counts
            log.info("SHADOW_SAVE swapbelt=fired th=%s model=%s new_len=%d",
                     self._thread_hash(thread_id), model_tag, new_len)
            # NO-DOWNGRADE: never overwrite an equal-or-longer existing shadow (protects
            # the freshest per-turn shadow; a stale/same-length source is refused here).
            existing = self._find_shadow_bin(port, model_tag, thread_id)
            if existing is not None and len(existing[1]) >= new_len:
                c["swap_skip_have_fresher"] = c.get("swap_skip_have_fresher", 0) + 1
                log.info("SHADOW_SAVE swapbelt=short_circuit_have_fresher th=%s model=%s "
                         "existing_len=%d new_len=%d",
                         self._thread_hash(thread_id), model_tag, len(existing[1]), new_len)
                return
            th = self._thread_hash(thread_id)
            async with self._shadow_lock_for(port, th):
                # reprefill the think-free messages (n_predict=0) so the engine slot holds
                # the think-free state, then SAVE — SAME mechanism/params as the per-turn
                # hook. Reached only when the per-turn save is missing/shorter (rare).
                # review note 2: this recovery reprefill is AWAITED before the swap's
                # _sigterm, so an unbounded read (900s) could STALL the model swap. Bound it
                # to 180s (a healthy think-free reprefill is ~158s) so the worst-case teardown
                # block stays far below 900s; a slow reprefill times out -> best-effort give-up.
                if _covered_scaffold_strip_enabled():
                    # crit2/crit3 (Fix B): the swap-seam recovery re-saves the SAME
                    # think-free messages, so it MUST render+strip identically to the per-turn
                    # _shadow_reprefill_and_save above — else this recovery bin would carry the
                    # covered-turn scaffold Fix B removes and become a position-drifted restore
                    # target (breaking this method's "byte-identical to the per-turn path"
                    # contract). No client_meta at the swap seam (the slot object is gone), so
                    # no tool-knob source -> renders tools-less, matching the pre-Fix-B swap
                    # payload (which also omitted _KV_PROBE_TOOL_KNOBS). Bounded to 180s.
                    if not await self._render_strip_prefill_probe(
                        port, model_tag, messages, {}, read_timeout_s=180.0
                    ):
                        c["swap_reprefill_post_failed"] = c.get("swap_reprefill_post_failed", 0) + 1
                        log.info("SHADOW_SAVE swapbelt=post_failed th=%s model=%s new_len=%d",
                                 th, model_tag, new_len)
                        return
                else:
                    # Flag OFF -> pre-Fix-B behavior, byte-identical to today.
                    payload = {
                        "model": model_tag, "messages": messages, "stream": False,
                        "n_predict": 0, "max_tokens": 0, "cache_prompt": True,
                    }
                    url = f"http://127.0.0.1:{port}/v1/chat/completions"
                    try:
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(connect=2.0, read=180.0, write=30.0, pool=2.0)
                        ) as client:
                            await client.post(url, json=payload)
                    except Exception:
                        c["swap_reprefill_post_failed"] = c.get("swap_reprefill_post_failed", 0) + 1
                        log.debug("swap-seam shadow reprefill POST failed (best-effort)", exc_info=True)
                        log.info("SHADOW_SAVE swapbelt=post_failed th=%s model=%s new_len=%d",
                                 th, model_tag, new_len)
                        return
                if await self._save_shadow_slot_kv(port, model_tag, thread_id, messages):
                    c["swap_saved"] = c.get("swap_saved", 0) + 1
                    log.info("swap-seam shadow KV re-saved (thread=%s, turns=%d)",
                             (thread_id or "")[:12], len(messages))
                    log.info("SHADOW_SAVE swapbelt=resaved th=%s model=%s new_len=%d turns=%d",
                             th, model_tag, new_len, len(messages))
        except Exception:
            log.debug("swap-seam shadow save best-effort failed", exc_info=True)

    async def _save_shadow_slot_kv(
        self, port: int, model_tag: str, thread_id: str, messages: list
    ) -> bool:
        """Atomic tmp+os.replace persist of the CURRENT (think-free) slot KV under the
        DISTINCT `.shadow` bin + a `shadow:true` meta sidecar carrying the think-free
        hash_chain / fingerprint. Deliberately does NOT reuse _save_slot_kv: the
        shadow bin must NEVER route through the clean_prefix never-demote/skip path —
        it is a SEPARATE artifact from the anchor, so a mispredicted shadow can never
        overwrite or demote the proven clean_prefix anchor. Requires exactly ONE
        populated slot (the single-series invariant) so the save maps unambiguously to
        this thread; anything else -> skip (never guess). Returns True iff a shadow bin
        was written. Best-effort; never raises."""
        if '/' in model_tag or '\\' in model_tag or '..' in model_tag:
            log.warning("shadow KV: refusing model_tag with unsafe path chars: %r", model_tag)
            return False
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        th = self._thread_hash(thread_id)
        # think-free hash_chain (SAME chokepoint + shape as the clean/normal save);
        # an empty chain is unrestorable -> never write a degenerate shadow bin.
        hash_chain = _prefix_hash_chain(messages)
        if not hash_chain:
            return False
        prompt_len_val = compute_ctx_len(messages)
        _fp = self._engine_fingerprint(model_tag)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=1.0, read=120.0, write=120.0, pool=1.0)
            ) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/slots")
                resp.raise_for_status()
                slots_data = resp.json()
                populated = [s for s in slots_data
                             if (s.get("n_prompt_tokens") or 0) > 0 and s.get("id") is not None]
                if len(populated) != 1:
                    # ambiguous (co-resident / none) — do NOT guess which slot is ours.
                    log.debug("shadow KV save skip: %d populated slots (want exactly 1)",
                              len(populated))
                    return False
                s = populated[0]
                sid = int(s["id"])
                n_prompt_tokens = s.get("n_prompt_tokens") or 0
                os.makedirs(SLOT_SAVE_DIR, exist_ok=True)
                bin_fn = _kv_shadow_save_fn(model_tag, sid, th, port)
                tmp_path = os.path.join(SLOT_SAVE_DIR, f"{bin_fn}.tmp")
                final_path = os.path.join(SLOT_SAVE_DIR, bin_fn)
                # Engine save to temp, then atomic rename (a reader sees the old bin or
                # the fully-written new one, never a partial).
                save_resp = await client.post(
                    f"http://127.0.0.1:{port}/slots/{sid}?action=save",
                    json={"filename": f"{bin_fn}.tmp"},
                )
                save_resp.raise_for_status()
                os.replace(tmp_path, final_path)
                # pin-and-ship: finalize the .ckpt ladder sidecar (bin FIRST, this best-effort).
                _finalize_ckpt_sidecar(tmp_path, final_path, bin_fn)
                # Invalidate scan cache after writing a bin/sidecar in SLOT_SAVE_DIR.
                self._kvcache_scan_cache_invalidate()
                meta = {
                    "thread_id": thread_id,
                    "thread_hash": th,
                    "prompt_tokens": n_prompt_tokens,
                    "prompt_len": prompt_len_val,
                    "n_context_turns": len(hash_chain),
                    "hash_chain": hash_chain,
                    "prompt_hash": "",
                    "model_tag": model_tag,
                    "slot_id": sid,
                    "port": port,
                    # DISTINCT MARKER: NOT a clean anchor. clean_prefix=False keeps it
                    # invisible to _find_clean_bin / the GC clean-pin / the clean-save
                    # skip+never-demote checks (all gate on clean_prefix); shadow=True
                    # tags it for PL's separate restore-preference step.
                    "clean_prefix": False,
                    "shadow": True,
                    # WIN 4 fingerprint (SAME as clean/normal save) so a stale shadow
                    # bin from a different build/model/ctx is purged, never restored.
                    "gguf_sha256": _fp.get("gguf_sha256"),
                    "engine_build_id": _fp.get("engine_build_id"),
                    "n_ctx": _fp.get("n_ctx"),
                    "n_rs_seq": _fp.get("n_rs_seq"),
                }
                meta_fn = _kv_shadow_meta_fn(model_tag, sid, th, port)
                meta_tmp = os.path.join(SLOT_SAVE_DIR, f"{meta_fn}.tmp")
                meta_path = os.path.join(SLOT_SAVE_DIR, meta_fn)
                with open(meta_tmp, "w") as f:
                    json.dump(meta, f)
                os.replace(meta_tmp, meta_path)
                # Invalidate scan cache after writing a bin/sidecar in SLOT_SAVE_DIR.
                self._kvcache_scan_cache_invalidate()
                log.info("shadow KV saved: %s (turns=%d, tokens=%d)",
                         bin_fn, len(hash_chain), n_prompt_tokens)
                # shadow-diag: the actual write result — path + on-disk byte
                # size (candidate (a) distinguisher: WAS a shadow bin written, and how
                # big). Off the hot path (this method runs from the post-set_result
                # save code). getsize best-effort so it can never fault the save.
                try:
                    _bin_size = os.path.getsize(final_path)
                except OSError:
                    _bin_size = -1
                log.info("SHADOW_SAVE write=ok th=%s model=%s path=%s size=%d turns=%d tokens=%d",
                         th, model_tag, bin_fn, _bin_size, len(hash_chain), n_prompt_tokens)
                return True
        except Exception:
            log.debug("shadow KV save best-effort failed for %s", model_tag, exc_info=True)
            log.info("SHADOW_SAVE write=fail th=%s model=%s", th, model_tag)
            return False

    async def _save_slot_kv(
        self, port: int, model_tag: str, slot=None, *,
        thread_id_override: str | None = None,
        admission_ctx_len_override: int | None = None,
        force_clean: bool = False,
        client_meta_override: dict | None = None,
    ) -> bool:
        """the engine-op badge work + FP R4 design note: the engine_op badge is scoped to the op —
        set on entry, ALWAYS reset in the finally (ReadTimeout/4xx/early-return
        included), or /status advertises kv_save forever on the worker_loop
        path (a display lie of the class Waves D/E killed)."""
        if slot is not None:
            slot.engine_op = "kv_save"
        try:
            return await self._save_slot_kv_inner(
                port, model_tag, slot,
                thread_id_override=thread_id_override,
                admission_ctx_len_override=admission_ctx_len_override,
                force_clean=force_clean,
                client_meta_override=client_meta_override,
            )
        finally:
            if slot is not None:
                slot.engine_op = "idle"

    async def _save_slot_kv_inner(
        self, port: int, model_tag: str, slot=None, *,
        thread_id_override: str | None = None,
        admission_ctx_len_override: int | None = None,
        force_clean: bool = False,
        client_meta_override: dict | None = None,
    ) -> bool:
        """Best-effort persist KV + metadata sidecar before sidecar reaped.
        SPEC-V2: returns True iff at least ONE slot's bin+meta pair was fully
        persisted (engine save 2xx + atomic bin rename + meta rename).
        Uses resolve_kv() chokepoint for decision + provenance logging.

        Args:
            thread_id_override: If set, use this thread_id instead of slot.thread_id.
                Used by idle-holder teardown to save with the original thread_id.
            admission_ctx_len_override: If set, use this for prompt_len in metadata
                instead of computing from slot.context/prompt. Used by idle-holder
                teardown to preserve the admission-time context length.
        """
        if '/' in model_tag or '\\' in model_tag or '..' in model_tag:
            log.warning("slot KV: refusing model_tag with unsafe path chars: %r", model_tag)
            return
        # ENFORCEMENT CHOKEPOINT (D1+D2). This is the SINGLE funnel
        # for every engine /slots?action=save (probe force_clean, idle-teardown
        # direct, _teardown, resident teardowns). (D1) Never save while this
        # port's VRAM tip holds foreign non-main tokens: the bin would be a
        # superset of the stamped hash_chain — an internally inconsistent triplet
        # that PASSES the restore prefix gate and aborts the engine at strict
        # extension. Remediation (erase + canonical reprefill) lives in the probe
        # seam path and clears the flag before calling us. (D2) Never save a
        # disposable-owned identity (single resolve point _seam_flush_allowed).
        # False = unconfirmed save, same contract as every other refusal.
        _dirty = (getattr(self, "_kv_dirty_tail", {}) or {}).get(port)
        _eff_tid = (thread_id_override if thread_id_override is not None
                    else (getattr(slot, "thread_id", "") if slot else "")) or ""
        if _dirty:
            log.warning(
                "slot KV save REFUSED (dirty tip): port=%s foreign role=%s (via "
                "thread=%s) would poison bin for thread_id=%s — previous bin kept",
                port, _dirty.get("role"), str(_dirty.get("thread"))[:40],
                (_eff_tid or "?")[:60])
            return False
        _eff_meta = (client_meta_override if client_meta_override is not None
                     else (getattr(slot, "client_meta", None) if slot else None))
        _allowed, _save_role = _seam_flush_allowed(_eff_tid, _eff_meta)
        if not _allowed:
            log.warning(
                "slot KV save REFUSED (D2 disposable identity): role=%s thread=%s "
                "— no bin written", _save_role, (_eff_tid or "?")[:60])
            return False
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        try:
            # F2: 10s read starved multi-GB engine action=save mid-decode
            # (chronic 'slot KV save failed' -> frozen anchor + .tmp orphan leak).
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=1.0, read=120.0, write=120.0, pool=1.0)) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/slots")
                resp.raise_for_status()
                slots_data = resp.json()
                # Save ALL populated slots (each keyed by its own identity)
                populated = [s for s in slots_data if (s.get("n_prompt_tokens") or 0) > 0 and s.get("id") is not None]
                if not populated:
                    log.debug("slot KV save: no populated slots")
                    return
                # Option A: only stamp clean_prefix when exactly ONE slot is
                # populated (unambiguous single-thread mapping) to avoid mis-marking a
                # co-resident thread's polluted KV as clean.
                _eff_force_clean = bool(force_clean) and len(populated) == 1
                # WIN 4 compute the loaded engine's build/model/ctx
                # fingerprint ONCE per save (not per slot). Sources are already
                # known to the manager (manifest gguf sha + boot.runtime binary sha
                # + manifest ctx/parallel) — this NEVER hashes the multi-GB gguf
                # (it uses the manifest's precomputed gguf_blob_sha256). Stamped
                # into each slot's meta below + compared by _purge_mismatched_bins.
                _fp = self._engine_fingerprint(model_tag)
                saved_any = False  # SPEC-V2: confirmed-save signal (A10 return)
                for s in populated:
                    sid = int(s["id"])
                    n_prompt_tokens = s.get("n_prompt_tokens") or 0
                    # Use override if provided, else fall back to slot
                    thread_id = thread_id_override if thread_id_override is not None else (getattr(slot, "thread_id", "") if slot else "")
                    # SPEC-V2 WAVE A: key the bin on the (session_id, role[, fp8])
                    # identity. Idle-holder teardown (slot=None) falls back to the
                    # manager-held idle client_meta ONLY when the override tid IS the
                    # idle holder's tid (plumbing) — any other slot=None
                    # caller keys on the raw thread_id (today's behavior).
                    if client_meta_override is not None:
                        # a later phase K-FIX: _teardown_idle_holder clears
                        # self._idle_thread_id BEFORE the flush/save block, so the
                        # elif below can never match at that seam — the save fell
                        # to RAW thread_id keying, wrote a ~10GB duplicate bin
                        # (defeating the clean-present skip), exhausted the tmpfs
                        # and caused the 22:30Z ENOSPC truncation. The teardown
                        # now passes its stashed idle meta explicitly; explicit
                        # caller intent outranks whatever _active_slot holds.
                        _cm = client_meta_override
                        _p_msgs = (client_meta_override or {}).get("messages")
                        _idc = _prefix_hash_chain(_p_msgs) if _p_msgs else None
                    elif slot is not None:
                        _cm = getattr(slot, "client_meta", None)
                        _idc = getattr(slot, "admission_hash_chain", None)
                    elif thread_id and thread_id == self._idle_thread_id:
                        _cm = self._idle_client_meta
                        _idc = None
                    else:
                        _cm, _idc = None, None
                    bin_ident = _bin_identity(thread_id, _cm, _idc)
                    thread_hash = self._thread_hash(bin_ident)
                    # Option A: a teardown/normal save (NOT force_clean) must
                    # NOT overwrite an existing clean-prefix bin with a post-generation
                    # (think-polluted) KV. Only the explicit probe-save (force_clean)
                    # may (over)write clean bins. Skip this slot if a clean bin exists.
                    if not force_clean:
                        _has_clean = False
                        try:
                            for _fn in os.listdir(SLOT_SAVE_DIR):
                                if (_fn.startswith(f"{model_tag}.") and _fn.endswith(".json")
                                        and f".p{port}.{thread_hash}." in _fn):
                                    with open(os.path.join(SLOT_SAVE_DIR, _fn)) as _mf:
                                        if json.load(_mf).get("clean_prefix"):
                                            _has_clean = True
                                            break
                        except Exception:
                            pass
                        if _has_clean:
                            log.info("slot KV save skip: clean_prefix bin present (thread_hash=%s)",
                                     thread_hash)
                            continue
                    # NEVER-DEMOTE invariant (a failure-mode guard, critical safety). A
                    # force_clean save that resolved to _eff_force_clean=False (i.e.
                    # len(populated) != 1) would stamp clean_prefix=False below and
                    # OVERWRITE the anchor bin -> demote clean_prefix True->False.
                    # _find_clean_bin + the GC pin then lose the anchor, the the classifier
                    # classifier disarms, and EVERY request goes FRESH (worse than the
                    # sawtooth lag). Abort THIS slot's save when a clean anchor already
                    # exists for this (model_tag, thread_hash, port). (The force_clean=
                    # False path is already covered by the skip guard above; this closes
                    # the force_clean=True-but-multi-slot hole.)
                    if force_clean and not _eff_force_clean:
                        _demote_clean = False
                        try:
                            for _fn in os.listdir(SLOT_SAVE_DIR):
                                if (_fn.startswith(f"{model_tag}.") and _fn.endswith(".json")
                                        and f".p{port}.{thread_hash}." in _fn):
                                    with open(os.path.join(SLOT_SAVE_DIR, _fn)) as _mf:
                                        if json.load(_mf).get("clean_prefix"):
                                            _demote_clean = True
                                            break
                        except Exception:
                            pass
                        if _demote_clean:
                            log.warning("clean-bin re-save ABORTED (never-demote): would lower "
                                        "clean_prefix True->False for thread_hash=%s", thread_hash)
                            continue
                    # curator save-gate. When the curator
                    # reuse-main route is active, a curator turn must NOT save/overwrite
                    # a bin (it rides on main's anchor; saving would poison it). Gate is
                    # flag-gated + label-gated -> inert with the flag off or labels absent.
                    if _curator_reuse_main_active():
                        from turbohaul.kv_classify import POLICIES, _class_from_label
                        _labels = (getattr(slot, "client_meta", None) or {}) if slot else {}
                        _lc = _class_from_label(_labels)
                        if _lc is not None and not POLICIES[_lc].save_ok:
                            log.info("slot KV save skip: curator route save_ok=False (thread_hash=%s)", thread_hash)
                            continue
                    # Policy decision
                    decision = resolve_kv("save", {
                        "thread_id": bin_ident,
                        "model_tag": model_tag,
                        "slot_id": sid,
                        "port": port,
                    }, {
                        "saved_tokens": n_prompt_tokens,
                    })
                    log.info("slot KV save decision: %s", decision)
                    if not decision.do_it:
                        continue
                    # Per-slot try/except: one bad slot doesn't kill the others
                    try:
                        bin_fn = kv_save_fn(model_tag, sid, thread_hash, port)
                        # Atomic save: write temp then rename
                        tmp_path = os.path.join(SLOT_SAVE_DIR, f"{bin_fn}.tmp")
                        final_path = os.path.join(SLOT_SAVE_DIR, bin_fn)
                        os.makedirs(SLOT_SAVE_DIR, exist_ok=True)
                        # a later phase T-GUARD (ENOSPC truncation, proven
                        # 22:30Z): a full tmpfs let the engine short-write a
                        # 1.17GB bin for a 198k-token state and still return
                        # 200; the truncated bin then overwrote the good SSD
                        # copy. Expected size is self-calibrated from THIS
                        # thread's previous meta (bytes scale ~linearly with
                        # tokens for a given model; no absolute floor — unit
                        # tests legitimately save tiny bins). No prior meta ->
                        # no estimate -> guards skip (first-save residual
                        # accepted).
                        _exp_bytes = None
                        try:
                            _old_meta_p = os.path.join(
                                SLOT_SAVE_DIR, kv_meta_fn(model_tag, sid, thread_hash, port))
                            if os.path.exists(_old_meta_p):
                                with open(_old_meta_p) as _omf:
                                    _om = json.load(_omf)
                                _ob, _ot = _om.get("bin_bytes"), _om.get("prompt_tokens")
                                if _ob and _ot and n_prompt_tokens:
                                    _exp_bytes = float(_ob) * (float(n_prompt_tokens) / float(_ot))
                        except Exception:
                            _exp_bytes = None
                        if _exp_bytes:
                            try:
                                _st = os.statvfs(SLOT_SAVE_DIR)
                                _avail = _st.f_bavail * _st.f_frsize
                            except OSError:
                                _avail = None
                            if _avail is not None and _avail < _exp_bytes * 1.05:
                                log.error(
                                    "slot KV save SKIPPED (T-GUARD pre-flight): avail=%d < need~%d "
                                    "for %s — keeping previous intact bin",
                                    _avail, int(_exp_bytes * 1.05), bin_fn)
                                continue
                        # Trigger engine save to temp file first
                        save_resp = await client.post(
                            f"http://127.0.0.1:{port}/slots/{sid}?action=save",
                            json={"filename": f"{bin_fn}.tmp"},
                        )
                        save_resp.raise_for_status()
                        if _exp_bytes:
                            try:
                                _new_bytes = os.path.getsize(tmp_path)
                            except OSError:
                                _new_bytes = None
                            if _new_bytes is not None and _new_bytes < 0.5 * _exp_bytes:
                                log.error(
                                    "slot KV save REJECTED (T-GUARD): %s bytes=%d expected~%d "
                                    "— truncated write (ENOSPC?); previous bin kept",
                                    bin_fn, _new_bytes, int(_exp_bytes))
                                # a design note: the engine also writes a .tmp.ckpt ladder
                                # sidecar (finalized only on success) — remove the
                                # twin too or the reject leaks tmpfs space exactly
                                # when space is already critical.
                                for _orph in (tmp_path, tmp_path + ".ckpt"):
                                    try:
                                        os.remove(_orph)
                                    except OSError:
                                        pass
                                continue
                        # a later phase PAIR-GUARD: the .ckpt ladder sidecar must
                        # land WITH the bin (KV contract: the pair moves together). A
                        # full tmpfs let a bin land while its ckpt silently truncated
                        # or vanished (live pair: bin@07:54 vs ckpt@07:45) — restored
                        # ladders then break and every rollback collapses to an
                        # ancient checkpoint or pos 0. Reject the PAIR unless the tmp
                        # ckpt is plausibly sized (>=10%% of bin; observed 0.5-2.2x).
                        # A model that never produces ckpts (no prior final ckpt and
                        # no tmp ckpt) passes untouched.
                        if _ckpt_sidecar_enabled():
                            try:
                                _ck_bytes = os.path.getsize(tmp_path + ".ckpt")
                            except OSError:
                                _ck_bytes = None
                            try:
                                _bin_bytes_now = os.path.getsize(tmp_path)
                            except OSError:
                                _bin_bytes_now = None
                            _ck_expected = os.path.exists(final_path + ".ckpt")
                            _ck_bad = (
                                (_ck_bytes is None and _ck_expected)
                                or (_ck_bytes is not None and _bin_bytes_now
                                    and _ck_bytes < 0.10 * _bin_bytes_now)
                            )
                            if _ck_bad:
                                log.error(
                                    "slot KV save REJECTED (PAIR-GUARD): %s ckpt_bytes=%s "
                                    "bin_bytes=%s — ckpt sidecar missing/truncated (ENOSPC?); "
                                    "previous pair kept", bin_fn, _ck_bytes, _bin_bytes_now)
                                for _orph in (tmp_path, tmp_path + ".ckpt"):
                                    try:
                                        os.remove(_orph)
                                    except OSError:
                                        pass
                                continue
                        # Atomic rename (overwrite if exists)
                        os.replace(tmp_path, final_path)
                        # design note: invalidate after .bin write
                        self._kvcache_scan_cache_invalidate()
                        # pin-and-ship: finalize the .ckpt ladder sidecar (bin FIRST, this best-effort).
                        _finalize_ckpt_sidecar(tmp_path, final_path, bin_fn)
                        # Write metadata sidecar (atomic: temp + replace)
                        context = getattr(slot, "context", None) if slot else None
                        prompt = getattr(slot, "prompt", "") if slot else ""
                        # FIX B fall back to the admission messages when
                        # slot.context is empty, so a >=40k save persists a REAL prefix-
                        # comparable chain (not [] which is permanently unrestorable —
                        # _is_prefix_match rejects an empty saved_chain). Same source +
                        # shape as admission_hash_chain / _engine_view_chain (:3839).
                        _ctx_src = context
                        if not _ctx_src and slot is not None:
                            _ctx_src = (getattr(slot, "client_meta", None) or {}).get("messages")
                        hash_chain = _prefix_hash_chain(_ctx_src) if _ctx_src else []
                        context_for_len = _ctx_src if _ctx_src else None
                        # saved_len uses the SAME chokepoint as the
                        # admission incoming_len (kv_policy.compute_ctx_len) so the
                        # two are always comparable. Fallback to len(prompt) only when
                        # no structured context was captured.
                        # use admission_ctx_len_override if provided.
                        if admission_ctx_len_override is not None:
                            prompt_len_val = admission_ctx_len_override
                        else:
                            prompt_len_val = compute_ctx_len(context_for_len) if context_for_len else len(prompt)
                        # FIX B guard on _ctx_src (the effective chain source)
                        # so the raw-prompt hash is only stamped when NEITHER context NOR
                        # admission messages produced a chain.
                        prompt_hash = hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest() if (not _ctx_src and prompt) else ""
                        meta = {
                            # SPEC-V2 WAVE A: owner identity = (session,role[,fp8])
                            # string — the byte-locked resolve_kv restore owner
                            # string-compare therefore keys on session+role. Raw tid
                            # preserved for diagnostics.
                            "thread_id": bin_ident,
                            "raw_thread_id": thread_id,
                            "session_id": (_cm or {}).get("session_id"),
                            "role": _bin_role(_cm),
                            # SPEC-V2 consistency stamp: byte-truth of the bin this
                            # meta was written for (bin os.replace succeeded above).
                            "bin_bytes": os.path.getsize(final_path),
                            "bin_mtime": os.path.getmtime(final_path),
                            "thread_hash": thread_hash,
                            "prompt_tokens": n_prompt_tokens,
                            "prompt_len": prompt_len_val,
                            "n_context_turns": len(hash_chain),
                            "hash_chain": hash_chain,
                            "prompt_hash": prompt_hash,
                            "model_tag": model_tag,
                            "slot_id": sid,
                            "port": port,
                            "clean_prefix": bool(_eff_force_clean),
                            # WIN 4 build/model/ctx fingerprint so a
                            # bin can never be silently restored into a different
                            # engine build / model / ctx (= garbage KV). Sourced
                            # once per save from the loaded engine — see
                            # _engine_fingerprint; NO multi-GB gguf hashing.
                            "gguf_sha256": _fp.get("gguf_sha256"),
                            "engine_build_id": _fp.get("engine_build_id"),
                            "n_ctx": _fp.get("n_ctx"),
                            "n_rs_seq": _fp.get("n_rs_seq"),
                        }
                        meta_fn = kv_meta_fn(model_tag, sid, thread_hash, port)
                        meta_tmp = os.path.join(SLOT_SAVE_DIR, f"{meta_fn}.tmp")
                        meta_path = os.path.join(SLOT_SAVE_DIR, meta_fn)
                        with open(meta_tmp, "w") as f:
                            json.dump(meta, f)
                        os.replace(meta_tmp, meta_path)
                        saved_any = True  # SPEC-V2: bin+meta both durably on disk
                        # design note: invalidate after .json sidecar write
                        self._kvcache_scan_cache_invalidate()
                        log.info("slot KV saved: %s (reason: %s)", bin_fn, decision.reason)
                        # P1 (DURABLE MANAGER B): INERT ring store + residency tag + LOG would-be reload.
                        # Reads client_meta labels (role, session_id) from slot to key the per-(role,session) ring.
                        # ZERO behavior change when TURBOHAUL_DURABLE_RING=OFF (default); just observability.
                        if _durable_ring_enabled() and slot is not None:
                            try:
                                ring_key = _durable_ring_key(getattr(slot, "client_meta", None))
                                if ring_key:
                                    ring_entry = {
                                        "bin_fn": bin_fn,
                                        "meta_fn": meta_fn,
                                        "thread_id": thread_id,
                                        "model_tag": model_tag,
                                        "port": port,
                                        "slot_id": sid,
                                        "clean_prefix": bool(_eff_force_clean),
                                        "n_context_turns": len(hash_chain),
                                        "hash_chain": hash_chain,
                                        "prompt_len": prompt_len_val,
                                        "saved_tokens": n_prompt_tokens,  # token count for restore saved_n (units match default path)
                                        "prompt_hash": prompt_hash,
                                        "saved_at": time.time(),
                                    }
                                    # Ring: last-3, newest-overrides-oldest (prepend + cap)
                                    ring_list = self._durable_ring_index.get(ring_key, [])
                                    ring_list.insert(0, ring_entry)
                                    if len(ring_list) > 3:
                                        ring_list.pop()  # evict oldest
                                    self._durable_ring_index[ring_key] = ring_list
                                    self._durable_ring_counts["ring_write"] = self._durable_ring_counts.get("ring_write", 0) + 1
                                    log.info("DURABLE_RING write: key=%s ring_len=%d entry=bin=%s clean=%s turns=%d",
                                             ring_key, len(ring_list), bin_fn, bool(_eff_force_clean), len(hash_chain))
                            except Exception:
                                log.debug("DURABLE_RING ring-store best-effort failed", exc_info=True)
                    except Exception as e:
                        log.warning("slot KV save failed for slot %d: %r", sid, e, exc_info=True)
                        continue
            return saved_any
        except Exception:
            log.debug("slot KV save best-effort failed for %s", model_tag, exc_info=True)
            return False

    # === The classifier (operator request): warm-path forced clean restore ==========
    # The pinned clean bin is BOTH the classifier anchor AND the warm-path restore
    # source. On a warm grace follow-up the engine's in-memory KV holds the just-
    # generated turn WITH <think>; the incoming user turn is think-STRIPPED, so the
    # engine's native LCP diverges at the first <think> -> minimal reuse OR a full
    # CLEAR. Forcing an `action=restore` of the think-free clean bin BEFORE the
    # follow-up decodes makes the engine's next get_common_prefix run against the
    # clean prefix -> strict extension (stale <= 0), no CLEAR (proven under load). The
    # NO-DOWNGRADE gate fires the force ONLY when it improves reuse (never regresses
    # a true continuation whose warm state is an equal-or-longer valid prefix).

    def _find_clean_bin(self, port: int, model_tag: str, thread_id: str):
        """Locate the pinned clean_prefix bin for (model_tag, thread_id, port).

        Returns (bin_filename, saved_chain, sid) for the LONGEST clean chain, or
        None. Uses the scan cache — no O(N) listdir loop. Best-effort — any
        FS/JSON error yields None (no force).
        """
        th = self._thread_hash(thread_id or "")
        try:
            clean_bins, _ = self._scan_kvcache_if_stale()
        except Exception:
            return None
        key = (model_tag, th, port)
        best = clean_bins.get(key)
        if best is None:
            return None
        # design note: defensive os.path.exists — cache may be stale between
        # invalidation sites (race between delete → next lookup). If the bin
        # vanished, invalidate + return None so the caller falls through.
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        _, bin_fn, _chain, _sid = best[1], best[1], best[2], best[3]
        if not os.path.exists(os.path.join(SLOT_SAVE_DIR, bin_fn)):
            self._kvcache_scan_cache_invalidate()
            return None
        # best = (chain_len, bin_fn, chain, sid, prompt_len)
        return best[1], best[2], best[3]

    def _find_shadow_bin(self, port: int, model_tag: str, thread_id: str):
        """step (d): locate the think-free `.shadow` bin for (model_tag,
        thread_id, port) — the SAVE-side artifact step (c) wrote (turns 1..N + the
        think-free assistant-N).

        Returns (bin_filename, saved_chain, sid) for the LONGEST shadow chain, or
        None. Uses the scan cache — no O(N) listdir loop. Mirrors _find_clean_bin's
        file layout EXACTLY, differing ONLY in the marker it matches: `.shadow.json`
        metas carrying `shadow:true` (vs the clean anchor's `clean_prefix:true`).
        The shadow bin is a SEPARATE artifact from the clean anchor, so the two
        finders never cross: `.shadow` metas have clean_prefix=False (invisible to
        _find_clean_bin), and a normal clean/meta never ends in `.shadow.json`.
        Best-effort — any FS/JSON error yields None so the caller falls through to
        EXACTLY today's clean-anchor path."""
        th = self._thread_hash(thread_id or "")
        try:
            _, shadow_bins = self._scan_kvcache_if_stale()
        except Exception:
            return None
        key = (model_tag, th, port)
        best = shadow_bins.get(key)
        if best is None:
            return None
        # design note: defensive os.path.exists — cache may be stale between
        # invalidation sites (race between delete → next lookup).
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        _, bin_fn, _chain, _sid = best[1], best[1], best[2], best[3]
        if not os.path.exists(os.path.join(SLOT_SAVE_DIR, bin_fn)):
            self._kvcache_scan_cache_invalidate()
            return None
        # best = (chain_len, bin_fn, chain, sid)
        return best[1], best[2], best[3]

    @staticmethod
    def _generated_assistant_msg(result) -> "dict | None":
        """Reconstruct the assistant turn the engine's warm KV now holds from a
        completion result, so the warm-state hash chain can be rebuilt.

        The engine's KV holds the FULL generation INCLUDING the <think> block; the
        client received (and the harness stores in history) reasoning merged inline
        as ``<think>...</think>{content}`` (mirrors chat_completion._merge_reasoning_
        into_content). Returns {'role':'assistant','content': <with-think>} or None.

        Returns None (caller SAFE-DEGRADES = no forced restore, no regression) when:
        - result is the noop/None default_complete or a streamed placeholder;
        - the turn is a TOOL CALL (content often null, tool_calls carry the payload,
          and _prefix_hash_chain hashes role+content only) — its history form cannot
          be reconstructed to hash-match, so we must not risk a false divergence;
        - content is empty. Only a positively-reconstructable text turn arms force."""
        if not isinstance(result, dict):
            return None
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        c0 = choices[0]
        msg = c0.get("message") if isinstance(c0, dict) else None
        if not isinstance(msg, dict):
            return None
        if msg.get("tool_calls"):
            return None  # tool-call continuation -> safe-degrade (see docstring)
        content = msg.get("content")
        if content is None:
            content = ""
        reasoning = msg.get("reasoning_content")
        if reasoning and isinstance(content, str) and "<think>" not in content:
            # critical item: SINGLE-SOURCE wrap (the exact _merge_reasoning_into_
            # content newline form) so this reconstruction byte-matches what the harness
            # stores/resends -> _strip_thinking_all(this) == the harness's think-strip.
            # The old no-newline `<think>{reasoning}</think>{content}` form diverged on
            # multi-block / stray-`</think>` content. Lazy import avoids a manager<->api
            # cycle (same pattern as the _strip_thinking_all import below).
            from turbohaul.api.chat_completion import wrap_reasoning_think
            content = wrap_reasoning_think(reasoning, content)
        if not content:
            return None
        return {"role": "assistant", "content": content}

    def _engine_view_chain(self, slot, result) -> list[str]:
        """Hash chain of what the engine's warm KV holds after a decode =
        prompt messages + the generated assistant turn (with <think>).

        Two sources for the generated turn:
        - NON-streaming: parse it out of the completion ``result`` dict.
        - STREAMING (the streaming path): the SSE route accumulated the generated text onto
          ``slot.streamed_assistant_text`` (content + reasoning merged as <think>...)
          before signaling stream_done; use it when ``result`` yields nothing.

        Returns [] (= 'warm state UNKNOWN' -> the gate will NOT force, safe) when
        the generated turn can't be positively reconstructed (noop/tool-call/parse
        miss). A known chain arms the no-downgrade comparison for the next turn."""
        gen = self._generated_assistant_msg(result)
        if gen is None:
            # STREAMING path: the route stashed the accumulated generated text.
            streamed = getattr(slot, "streamed_assistant_text", None)
            if streamed:
                gen = {"role": "assistant", "content": streamed}
        if gen is None:
            return []
        msgs = (getattr(slot, "client_meta", None) or {}).get("messages")
        if not msgs:
            msgs = getattr(slot, "context", None)
        if not msgs:
            return []
        return _prefix_hash_chain(list(msgs) + [gen])

    def _classify_event(self, clean_chain, inc_chain, warm_chain) -> str:
        """Map the chain relationships to the operator's 4 event types (+ guard-skip).

        - guard-skip: no incoming chain (can't classify).
        - sub-agent: no clean anchor for this identity (distinct/first-seen thread
                       — the design nonce sub-agents land here until they save their own
                       anchor; classified independently, never cross-restored).
        - compression: clean anchor exists but is NOT a prefix of incoming (an early
                       turn was rewritten/summarized).
        - continuation: clean ⊑ incoming AND the warm state is an equal-or-longer
                       valid prefix (true continuation — native warm reuse wins).
        - user-message: clean ⊑ incoming but the warm state diverges (the think-strip
                       follow-up — the forced clean restore fires here).

        Unified: delegates to kv_classify.classify_event (single source of truth —
        that pure function is byte-identical to the logic that lived here, and the
        golden suite pins classify_event == _classify_event across the matrix)."""
        from turbohaul.kv_classify import classify_event
        return classify_event(clean_chain, inc_chain, warm_chain)

    def _emit_classifier_decision(self, decision: dict) -> None:
        """P5 observability: structured log + manager metric per restore-relevant
        request. PROVES 'Turbohaul determines each event' (which event + whether it
        forced the clean restore). Counts by event_type; keeps the last decision."""
        et = decision.get("event_type", "guard-skip")
        self._kv_classifier_counts[et] = self._kv_classifier_counts.get(et, 0) + 1
        if decision.get("forced_clean_restore"):
            self._kv_classifier_forced += 1
        self._kv_classifier_last = decision
        log.info("the classifier classifier decision: %s", decision)

    # ============================================================
    # shadow byte-match self-check (DORMANT — observability ONLY)
    # ============================================================
    # De-risks the shadow-reprefill feature by MEASURING its single load-bearing
    # assumption: that the manager's THINK-FREE assistant-turn content byte-matches
    # what the Hermes harness resends (think-STRIPPED) next turn. Reuse is decided on
    # a turn-hash chain, so if the manager's strip != the harness's strip the saved
    # state silently won't reuse. This pair only reads, hashes, logs, and counts — it
    # writes NO KV, touches NO restore/admission/routing/save decision, and never
    # raises into the hot path (both methods are self-protecting best-effort).

    def _record_shadow_bytematch_probe(self, slot, result) -> None:
        """Stash turn N's THINK-FREE assistant hash for the next-turn compare.

        Called at the 4 main-generation completion points (streaming + non-streaming,
        anchor + active-match), mirroring the ``_engine_view_chain`` calls. ``result``
        is the completion dict on the non-streaming path and ``None`` on the streaming
        path (then the route-stashed ``slot.streamed_assistant_text`` carries the
        with-<think> answer, same source ``_engine_view_chain`` uses).

        Records a ``skipped_*`` count and returns (no stash) when there is nothing
        probe-able: a tool-call/noop turn (non-streaming ``_generated_assistant_msg``
        -> None => skipped_toolcall), content lacking a ``</think>`` block
        (skipped_no_think), or an empty think-free strip / absent streamed text
        (skipped_empty). Best-effort: any exception is swallowed (never raises).
        """
        try:
            c = self._shadow_bytematch_counts
            # 1. Recover the full WITH-<think> assistant content (mirror _engine_view_chain).
            if result is None:
                # STREAMING: the SSE route stashed the merged <think> text on the slot.
                content = getattr(slot, "streamed_assistant_text", None)
            else:
                # NON-STREAMING: reconstruct from the completion dict. None => tool-call /
                # noop / empty => not a positively-reconstructable text turn -> safe skip.
                gen = self._generated_assistant_msg(result)
                if gen is None:
                    c["skipped_toolcall"] = c.get("skipped_toolcall", 0) + 1
                    return
                content = gen.get("content")
            if not isinstance(content, str) or not content:
                # streaming parse-miss / tool-call stream / empty answer -> nothing to probe
                c["skipped_empty"] = c.get("skipped_empty", 0) + 1
                return
            if "</think>" not in content:
                c["skipped_no_think"] = c.get("skipped_no_think", 0) + 1
                return
            # Mirror the harness's remove-ALL think-strip so the manager side hashes
            # EXACTLY what the harness would resend — including multi-block / pre-<think>
            # content, where the old rsplit-last diverged (thinkstrip-multiblock;
            # 2/11 live MISMATCH). SAME strip fn the shadow-save uses, so this probe measures
            # exactly what the save produces. Lazy import avoids any manager<->api import
            # cycle (idiomatic here, cf. the QueueClosed / SLOT_SAVE_DIR local imports).
            from turbohaul.api.chat_completion import _strip_thinking_all
            think_free = _strip_thinking_all(content)
            if not think_free:
                c["skipped_empty"] = c.get("skipped_empty", 0) + 1
                return
            th = self._thread_hash(getattr(slot, "thread_id", "") or "")
            probe = self._shadow_bytematch_probe
            # pop-then-set so a re-stashed active thread moves to most-recent (true
            # write-LRU eviction below; never evicts a thread that just wrote).
            probe.pop(th, None)
            probe[th] = {
                "assistant_hash": self._turn_hash("assistant", think_free),
                "sample": think_free[:120],
            }
            # Bounded FIFO: evict oldest write(s) so the stash can never leak.
            while len(probe) > _SHADOW_BYTEMATCH_CAP:
                probe.pop(next(iter(probe)))
        except Exception:
            log.debug("shadow_bytematch record best-effort failed", exc_info=True)

    def _compare_shadow_bytematch_probe(self, thread_id, context, client_meta) -> None:
        """On turn N+1 admission, compare the harness's resent (think-stripped)
        assistant-N turn against turn N's stashed think-free hash.

        The incoming assistant-N turn = the LAST ``role == 'assistant'`` message in the
        resent history (the harness appends the new user turn AFTER it, so the most-
        recent assistant turn is exactly the one we generated last turn; tool-call turns
        never stash, so they can't leave a stale probe). Both sides are hashed with
        ``_turn_hash('assistant', ...)`` — a STANDALONE single-turn hash (NOT the rolling
        ``_prefix_hash_chain``), so hash-equality == byte-equality of the two strips.
        Logs match/mismatch (+ samples on mismatch to diagnose the byte-delta), counts
        it, and pops the probe (consumed).

        DORMANT: changes NO decision (``submit`` already enqueued the slot above).
        Best-effort; never raises into ``submit``.
        """
        try:
            probe = self._shadow_bytematch_probe
            if not probe:
                return
            th = self._thread_hash(thread_id or "")
            stash = probe.get(th)
            if stash is None:
                return
            # Incoming messages: client_meta["messages"] else context (mirror _engine_view_chain).
            msgs = None
            if isinstance(client_meta, dict):
                msgs = client_meta.get("messages")
            if not msgs:
                msgs = context
            if not msgs:
                return
            inc_content = None
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    inc_content = m.get("content")
                    break
            if inc_content is None:
                return  # no assistant turn resent yet — leave the stash for a later turn
            inc_h = self._turn_hash("assistant", inc_content)
            c = self._shadow_bytematch_counts
            if inc_h == stash.get("assistant_hash"):
                c["match"] = c.get("match", 0) + 1
                log.info("shadow_bytematch MATCH thread=%s", th)
            else:
                c["mismatch"] = c.get("mismatch", 0) + 1
                inc_sample = inc_content if isinstance(inc_content, str) else str(inc_content)
                log.info(
                    "shadow_bytematch MISMATCH thread=%s saved=%s inc=%s "
                    "saved_sample=%r inc_sample=%r",
                    th, stash.get("assistant_hash"), inc_h,
                    stash.get("sample"), inc_sample[:120],
                )
            probe.pop(th, None)  # consumed
        except Exception:
            log.debug("shadow_bytematch compare best-effort failed", exc_info=True)

    @staticmethod
    def _is_qwen_family(model_tag: str) -> bool:
        """Streaming path: the turn-hash⊑ → engine-token-prefix equivalence that makes the
        forced clean-restore SAFE was proven ONLY on the qwen MTP family (divergence
        lands at the primer boundary, stale=1, one token off the n_rs_seq=2 CLEAR
        threshold). A different template/tokenizer (longer primer, multi-token think
        preamble, date/counter in the system render) could keep the turn-hash valid
        while pushing token divergence > 2 → CLEAR. The physics belt guards turn-
        COUNT, not the token prefix, so it can't catch that class. So the force path
        is gated to qwen explicitly; any other family safe-degrades to no-force."""
        m = (model_tag or "").lower()
        return "qwen" in m or "qwq" in m

    def _reset_clean_bin(self, port: int, model_tag: str, thread_id: str, bin_fn: str, requester_is_labeled_main: bool = False) -> None:
        """: after a COMPRESSION event (clean anchor no longer a
        prefix of the incoming context), drop the stale clean bin (+ its clean-prefix
        .json sidecar) for this thread so the next turn's scale-up probe re-anchors a
        fresh clean bin at the new compressed baseline. Best-effort; never raises into
        the decode path. Does NOT touch the restore gate — it only removes a
        file the gate already refuses to restore."""
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        th = self._thread_hash(thread_id)
        # SPEC-V2 (RED-HAT design note belt): a bin stamped role=main may only be reset
        # by a requester whose OWN role came from an explicit is_main label —
        # never a defaulted or chain-inferred identity. Legacy metas without a
        # role stamp pass through unchanged (one-time migration window).
        try:
            with open(os.path.join(SLOT_SAVE_DIR, bin_fn[:-4] + ".json")) as _af:
                _am = json.load(_af)
            if _am.get("role") == "main" and not requester_is_labeled_main:
                log.warning(
                    "clean-bin RESET REFUSED: anchor role=main but requester role is "
                    "not label-derived main (bin=%s thread=%s)", bin_fn, (thread_id or "?")[:16])
                return
        except FileNotFoundError:
            pass
        except Exception:
            log.debug("clean-bin reset: anchor meta read skipped", exc_info=True)
        removed = 0
        try:
            try:
                os.remove(os.path.join(SLOT_SAVE_DIR, bin_fn))
                # design note: invalidate after bin delete
                self._kvcache_scan_cache_invalidate()
                removed += 1
            except FileNotFoundError:
                pass
            for fn in list(os.listdir(SLOT_SAVE_DIR)):
                if not (fn.startswith(f"{model_tag}.") and fn.endswith(".json")
                        and f".p{port}.{th}." in fn):
                    continue
                try:
                    with open(os.path.join(SLOT_SAVE_DIR, fn)) as f:
                        m = json.load(f)
                    # SPEC-V2 WAVE A belts (RED-HAT MODS 1+2): only reset sidecars
                    # whose STAMPED owner identity equals the requester's (an
                    # interleaved variant can never delete main's anchor — the
                    # turn-52 residual — even if hashes ever collide); and a sidecar
                    # stamped role=main may only be reset by a label-derived main.
                    if m.get("clean_prefix") and m.get("thread_id", "") == thread_id:
                        if m.get("role") == "main" and not requester_is_labeled_main:
                            log.warning("clean-bin RESET REFUSED (sidecar role=main, requester not labeled main): %s", fn)
                        else:
                            os.remove(os.path.join(SLOT_SAVE_DIR, fn))
                            removed += 1
                except FileNotFoundError:
                    pass
                except Exception:
                    log.debug("clean-bin reset: sidecar read/remove skipped", exc_info=True)
            log.info(
                "clean-bin RESET (compression): dropped %d file(s) anchor=%s thread=%s",
                removed, bin_fn, (thread_id or "?")[:12])
        except Exception:
            log.debug("clean-bin reset best-effort failed", exc_info=True)

    def _mark_main_bin_stale_for_session(self, session_id: str, reason: str = "is_compression") -> int:
        """R-COMP (per the operator: compression re-delivers, so Turbohaul just
        re-computes the session's is_main fresh next time). NON-DESTRUCTIVE:
        stamp stale:true into every clean_prefix sidecar whose STAMPED owner
        identity is exactly sess:<session_id>:main (atomic tmp+replace). Scan /
        find / cold-restore treat a stale meta as ABSENT; the bin stays on disk
        (crash-safe, no delete race) and the NEXT confirmed main force_clean
        save overwrites the same deterministic filename, clearing the marker.
        _reset_clean_bin's design note main-guard is deliberately untouched. Returns
        sidecars marked; best-effort, never raises."""
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        marked = 0
        try:
            ident = f"sess:{session_id}:main"
            for fn in list(os.listdir(SLOT_SAVE_DIR)):
                if not fn.endswith(".json"):
                    continue
                fpath = os.path.join(SLOT_SAVE_DIR, fn)
                try:
                    with open(fpath) as f:
                        m = json.load(f)
                    if not (m.get("clean_prefix") and not m.get("stale")
                            and m.get("thread_id") == ident):
                        continue
                    m["stale"] = True
                    m["stale_reason"] = reason
                    m["stale_at"] = time.time()
                    tmp = fpath + ".stale.tmp"
                    with open(tmp, "w") as f:
                        json.dump(m, f)
                    os.replace(tmp, fpath)
                    marked += 1
                except (OSError, ValueError):
                    continue
            if marked:
                self._kvcache_scan_cache_invalidate()
                # Drop any live VRAM anchor for the main identity too, so the R2
                # gate can never skip-force on the pre-compression state.
                th = self._thread_hash(ident)
                for _p, _rec in list((getattr(self, "_kv_vram_anchor", None) or {}).items()):
                    if _rec.get("thread_hash") == th:
                        self._kv_vram_anchor.pop(_p, None)
                log.info("R-COMP: marked %d main clean sidecar(s) STALE for session=%s (%s)",
                         marked, str(session_id)[:16], reason)
        except Exception:
            log.debug("R-COMP stale-mark best-effort failed", exc_info=True)
        return marked

    async def _maybe_force_clean_restore(
        self, port: int, model_tag: str, slot, warm_chain: list[str]
    ) -> dict:
        """the classifier warm-path forced clean-bin restore + NO-DOWNGRADE gate.

        Runs BEFORE a warm grace follow-up decodes. Gate:
            force = (clean ⊑ incoming) AND warm_state_known AND NOT warm_covers
        where warm_covers = (warm ⊑ incoming) AND len(warm) >= len(clean).

        - clean ⊑ incoming: the think-free clean bin is a valid prefix -> restoring
          it yields a guaranteed strict-extension reuse (physics belt: _is_prefix_
          match rejects a clean bin LONGER than incoming, so we never restore a bin
          the engine would CLEAR).
        - warm_covers: the engine's current warm KV already gives an equal-or-
          longer valid prefix (a TRUE continuation — no think divergence). Forcing
          the shorter clean bin here would restore LESS + reprefill recent turns =
          REGRESSION, so we DO NOT force. Native warm reuse wins.
        - warm_state_known: if the warm chain is unknown ([] — noop/streamed/tool
          call), we DO NOT force (safe-degrade to today's behavior; no regression).

        Only when clean is valid AND the warm state is KNOWN to diverge do we POST
        `action=restore` of the clean bin into the engine slot. Best-effort; never
        raises into the decode path. Returns the observability decision dict."""
        inc_chain = getattr(slot, "admission_hash_chain", []) or []
        tid = getattr(slot, "thread_id", "") or ""
        # SPEC-V2 WAVE A: every bin lookup/reset in this method keys on the
        # (session_id, role[, fp8]) identity (raw tid when unlabeled/no session).
        tid = _bin_identity(tid, getattr(slot, "client_meta", None), inc_chain)
        # crit3: the incoming turns (client_meta["messages"], 1:1 aligned
        # with inc_chain — both derived from the same admission messages) for the
        # tool-opaque tail guard below. Absent (streamed submit w/o messages) -> []
        # -> guard inert (safe-degrade to today's force behavior).
        inc_messages = (getattr(slot, "client_meta", None) or {}).get("messages") or []
        decision = {
            "event_type": "guard-skip",
            "action": "fresh",
            "resolved_from": "warm-no-incoming-chain",
            "clean_bin_id": None,
            "common_prefix_turns": 0,
            "incoming_turns": len(inc_chain),
            "forced_clean_restore": False,
        }
        try:
            # severity item P1 — WARM forced clean-bin restore is DEFAULT OFF.
            # Short-circuit BEFORE any restore POST (equivalent to gating the 2 warm
            # callers, but unit-testable + observable): the warm grace follow-up decodes
            # on the engine's native in-RAM get_common_prefix reuse (clean v0.5.8 parity).
            # warm_covers was structurally ~always False (warm_chain is WITH-<think>; the
            # harness resends think-STRIPPED) -> the force fired every warm follow-up ->
            # byte-mismatch -> engine CLEAR -> full reprefill (~16k of ~50k). The COLD
            # swap-back / wave-return (_restore_slot_kv) is a SEPARATE path, NOT gated by
            # this flag. ON (emergency A/B rollback) re-enables the force below (the crit3
            # tool-tail guard, shadow-restore-preference + no-downgrade gate all preserved).
            # The `finally` still emits this decision so /status reflects the gate.
            if not _warm_force_clean_restore_enabled():
                decision["action"] = "fresh"
                decision["resolved_from"] = "warm-force-gated-native-reuse"
                decision["forced_clean_restore"] = False
                return decision
            if not inc_chain or not tid:
                return decision
            found = self._find_clean_bin(port, model_tag, tid)
            # SPEC-V2 REWORK R2: live in-VRAM anchor overlay. With per-turn disk
            # saves gone the disk bin LAGS until unload; the classifier anchor and
            # the force decision ride the in-memory chain whenever the VRAM state
            # provably belongs to THIS identity on THIS live sidecar (identity
            # hash + model_tag + pid all match — a respawn/foreign takeover fails
            # the pid/identity compare and falls through to today's disk logic).
            _mem = (getattr(self, "_kv_vram_anchor", None) or {}).get(port) or {}
            _vram_ours = (
                _mem.get("model_tag") == model_tag
                and _mem.get("thread_hash") == self._thread_hash(tid)
                and _mem.get("pid") is not None
                and _mem.get("pid") == getattr(slot, "pid", None)
            )
            _mem_chain = list(_mem.get("chain") or []) if _vram_ours else []
            if found is None:
                decision["event_type"] = self._classify_event(
                    _mem_chain or None, inc_chain, warm_chain)
                decision["resolved_from"] = (
                    "warm-vram-anchor-native-reuse" if _mem_chain
                    else "warm-no-clean-bin")
                if _mem_chain:
                    _mc = 0
                    for a, b in zip(_mem_chain, inc_chain, strict=False):
                        if a != b:
                            break
                        _mc += 1
                    decision["common_prefix_turns"] = _mc
                return decision
            bin_fn, clean_chain, sid = found
            # SPEC-V2 REWORK R2 (staleness gate): NEVER force-restore a disk bin
            # that is staler than the live VRAM state — restoring it would REWIND
            # the slot and throw away the per-turn probe prefill (forbidden by
            # the operator's spec). Skip the force IFF the VRAM anchor is OURS, still a
            # valid prefix of the incoming chain, AND at least as long as the
            # disk chain. Same identity => native reuse + the per-turn probe
            # handles the think-tail deltas; foreign/empty VRAM falls through to
            # the disk-restore machinery below unchanged.
            # Option C: when per-turn RAM re-anchor is active, the RAM
            # clean bin IS the fresh state and the restore is REQUIRED to re-commit
            # the engine base (the VRAM tail is think-polluted post-generation and
            # will diverge natively). Do NOT skip the force in that mode.
            if (not _ram_reanchor_enabled()
                    and _mem_chain
                    and _is_prefix_match(_mem_chain, inc_chain)
                    and len(_mem_chain) >= len(clean_chain)):
                decision["event_type"] = self._classify_event(
                    _mem_chain, inc_chain, warm_chain)
                decision["clean_bin_id"] = bin_fn
                _mc = 0
                for a, b in zip(_mem_chain, inc_chain, strict=False):
                    if a != b:
                        break
                    _mc += 1
                decision["common_prefix_turns"] = _mc
                decision["resolved_from"] = "warm-vram-fresher-skip"
                return decision
            clean_valid = _is_prefix_match(clean_chain, inc_chain)
            warm_known = bool(warm_chain)
            warm_covers = (
                warm_known
                and _is_prefix_match(warm_chain, inc_chain)
                and len(warm_chain) >= len(clean_chain)
            )
            # The gate would fire on prefix-validity + warm divergence...
            would_force = clean_valid and warm_known and not warm_covers
            # ...but F2 gates the actual force to the qwen MTP family only.
            qwen_ok = self._is_qwen_family(model_tag)
            force = would_force and qwen_ok
            # common prefix depth (turns) for observability
            common = 0
            for a, b in zip(clean_chain, inc_chain, strict=False):
                if a != b:
                    break
                common += 1
            decision["event_type"] = self._classify_event(clean_chain, inc_chain, warm_chain)
            decision["clean_bin_id"] = bin_fn
            decision["common_prefix_turns"] = common
            # forced_clean_restore stays False until a 2xx restore POST (F5).
            decision["forced_clean_restore"] = False
            if would_force and not qwen_ok:
                # would force, but the model family is out of the proven-safe scope
                decision["resolved_from"] = "warm-non-qwen-skip"
            elif not clean_valid:
                decision["resolved_from"] = "warm-clean-not-prefix"
                # (per the "go back down after a
                # compression event" design): clean anchor is NOT a prefix of incoming ==
                # compression (an early turn was summarized/rewritten). The stale
                # longer/diverged bin can never validly restore now (the physics belt
                # already refuses it) and would keep every turn FRESH. RESET it so the
                # next turn's SCALEUP probe re-anchors a fresh clean bin at the NEW
                # compressed baseline. Best-effort; restore-gate logic
                # UNTOUCHED (this only removes a file the gate already rejects).
                # F3 guard: turn-0 divergence (common == 0) is a thread-id
                # COLLISION (two conversations coalesced onto one key), NOT compression.
                # Resetting there would delete/re-save the shared anchor on every
                # interleave (GB-scale churn). Reset only on a real mid-thread rewrite.
                if decision.get("event_type") == "compression" and common > 0:
                    # SPEC-V2 (RED-HAT design note): prove the requester's role came from an
                    # explicit is_main LABEL (not default, not literal, not inference)
                    # before the reset may touch a main-role bin.
                    from turbohaul.kv_classify import _class_from_label, CLASS_MAIN
                    self._reset_clean_bin(
                        port, model_tag, tid, bin_fn,
                        requester_is_labeled_main=(
                            _class_from_label(getattr(slot, "client_meta", None) or {}) == CLASS_MAIN))
            elif warm_covers:
                decision["resolved_from"] = "warm-native-reuse-longer"
            elif not force:
                decision["resolved_from"] = "warm-state-unknown-safe"
            # --- crit3: TOOL-tail restore SKIP (Option A) ---------------
            # The force gate above trusts the chain-prefix match, but _prefix_hash_
            # chain is BLIND to tool turns (assistant tool_calls / role=="tool" / null
            # content). If the divergent tail beyond `common` is tool-opaque, the
            # harness re-serializes that region nondeterministically -> forcing the
            # clean/shadow bin would POST token-stale KV -> engine stale>n_rs_seq
            # CLEAR -> full reprefill (a REGRESSION vs native warm reuse). Safe-
            # degrade: DON'T force -> the engine's own get_common_prefix checkpoint
            # reuse runs. Only skips the FORCE; a TEXT/think-strip tail (hash-
            # verifiable, no tool turn beyond common) STILL forces + still gets the
            # crit2 shadow-restore-preference. Flag-gated (default ON) for A/B
            # rollback. Runs AFTER the resolved_from chain so its distinct reason is
            # not overwritten, and BEFORE the `if force:` restore block so the POST
            # (clean AND the shadow-preferred target inside it) is fully gated.
            if (force and _tooltail_restore_skip_enabled()
                    and _divergent_tail_is_tool_opaque(
                        inc_messages, common,
                        scan_covered=_tooltail_scan_covered_enabled())):
                force = False
                self._kv_tooltail_skip_counts["warm"] = (
                    self._kv_tooltail_skip_counts.get("warm", 0) + 1)
                decision["resolved_from"] = "warm-tooltail-skip"
                log.info(
                    "crit3 warm TOOL-tail restore SKIP: divergent tail "
                    "beyond common=%d is tool-opaque (hash-invisible) -> NOT forcing "
                    "clean/shadow restore; safe-degrade to engine native reuse "
                    "(thread=%s, incoming_turns=%d, clean_bin=%s)",
                    common, tid[:12], len(inc_chain), bin_fn,
                )
            if force:
                # F5: only count the forced restore AFTER the engine ACKs it (2xx).
                # A 500'd restore leaves the decode on the polluted warm KV, so the
                # metric must NOT claim a clean restore that never landed.
                decision["action"] = "restore"
                # --- step (d): SHADOW-BIN RESTORE-PREFERENCE (flag-gated) ---
                # The gate ABOVE is UNCHANGED and alone decides WHETHER to force + which
                # CLEAN anchor (bin_fn/clean_chain/sid). Here we ONLY upgrade the restore
                # TARGET: when TURBOHAUL_SHADOW_RESTORE_PREFER is ON, prefer the think-free
                # `.shadow` bin (turns 1..N + think-free assistant-N, saved by step c) over
                # the shorter clean anchor (turns 1..N) IFF the shadow bin passes the SAME
                # prefix-validity bar the clean path uses (_is_prefix_match vs inc_chain)
                # AND is at least as long (no-downgrade — mirrors warm_covers' >= rule so a
                # preference can only ever GROW the reused prefix, never shrink it). A
                # longer valid prefix => the next decode STRICT-EXTENDS the think-free state
                # (it skips reprefilling assistant-N) instead of the against-3 collapse.
                # A shadow that is stale / not a prefix / shorter is NOT preferred -> we
                # fall through to EXACTLY today's clean anchor. Flag OFF => this whole block
                # is skipped and r_* stay the clean values => the restore is byte-identical
                # to today.
                #
                # CORRECTNESS BACKSTOP (why the preference is safe): the engine's
                # get_common_prefix is authoritative on the next decode — even a wrongly-
                # preferred shadow bin only makes the engine find the REAL common prefix and
                # reprefill the divergent tail (a REPREFILL, never a wrong answer; identical
                # safety to today's clean restore). Do NOT bypass or alter that behavior; it
                # is what lets this stay a pure WHICH-VALID-BIN preference.
                r_bin, r_sid, r_chain, r_kind = bin_fn, sid, clean_chain, "clean"
                if _shadow_restore_prefer_enabled():
                    try:
                        _sh = self._find_shadow_bin(port, model_tag, tid)
                        if (_sh is not None
                                and _is_prefix_match(_sh[1], inc_chain)
                                and len(_sh[1]) >= len(clean_chain)):
                            r_bin, r_chain, r_sid, r_kind = _sh[0], _sh[1], _sh[2], "shadow"
                            self._kv_shadow_restore_counts["preferred"] = (
                                self._kv_shadow_restore_counts.get("preferred", 0) + 1)
                        else:
                            self._kv_shadow_restore_counts["clean_fallback"] = (
                                self._kv_shadow_restore_counts.get("clean_fallback", 0) + 1)
                            if _sh is not None:
                                log.info(
                                    "shadow restore-preference: shadow bin present but NOT "
                                    "preferred (shadow_turns=%d, clean_turns=%d, prefix=%s) "
                                    "-> using clean anchor for thread=%s",
                                    len(_sh[1]), len(clean_chain),
                                    _is_prefix_match(_sh[1], inc_chain), tid[:12])
                    except Exception:
                        log.debug("shadow restore-preference lookup failed (best-effort); "
                                  "using clean anchor", exc_info=True)
                # a later phase SKIP-REDUNDANT-RESTORE: when the engine's natural
                # slot state IS already this thread's clean anchor (the per-turn
                # clean-prefix save is taken FROM the natural state and stamps
                # _kv_vram_anchor with the exact saved chain), re-POSTing
                # action=restore is a content no-op but arms the engine's
                # restored-slot handling — and a chat serve on a restored slot
                # deterministically fails M-RoPE batch init (X<Y) -> 500 ->
                # prompt_clear -> FULL reprefill (100-160s/turn; 2026-07-09
                # receipts: serve tasks 746/1523/2523 canceled, 777/1531/2555 full
                # clears). Chain equality proves the natural state matches the bin
                # byte-for-byte at save time; any OTHER identity's confirmed save
                # overwrites the anchor, so a NEEDED restore is never skipped. An
                # engine relaunch leaves a stale anchor -> the skip serves on an
                # empty slot -> one full native reprefill (the cost the crash path
                # already paid, minus the 500).
                # a later phase: the skip is only VALID when the natural state
                # equals the RESEND byte-for-byte. For a thinking model whose harness
                # drops reasoning on resent turns, the natural state carries generated
                # <think> the resend lacks, so the stripped clean bin (NOT natural) is
                # what matches -> default OFF -> fall through to forced-clean-restore.
                _anch = self._kv_vram_anchor.get(port) or {}
                if (_warm_natural_skip_enabled()
                        and r_kind == "clean"
                        and _anch.get("model_tag") == model_tag
                        and list(_anch.get("chain") or []) == list(clean_chain)):
                    decision["forced_clean_restore"] = False
                    decision["clean_bin_id"] = bin_fn
                    decision["resolved_from"] = "warm-anchor-natural-skip"
                    log.info(
                        "the classifier warm-path restore SKIPPED (natural state == clean anchor): "
                        "%s (engine_slot=%d, thread=%s, clean_turns=%d <= incoming_turns=%d)",
                        bin_fn, sid, tid[:12], len(clean_chain), len(inc_chain))
                    return decision
                posted_ok = False
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(connect=2.0, read=120.0, write=10.0, pool=1.0)
                    ) as client:
                        resp = await client.post(
                            f"http://127.0.0.1:{port}/slots/{r_sid}?action=restore",
                            json={"filename": r_bin},
                        )
                        resp.raise_for_status()
                    posted_ok = True
                except Exception:
                    log.warning(
                        "the classifier forced clean restore POST failed: %s (engine_slot=%d, thread=%s)",
                        r_bin, r_sid, tid[:12], exc_info=True,
                    )
                if posted_ok:
                    decision["forced_clean_restore"] = True
                    decision["clean_bin_id"] = r_bin  # reflect what actually restored
                    if r_kind == "shadow":
                        decision["resolved_from"] = "warm-force-shadow-restore"
                        log.info(
                            "the classifier warm-path forced SHADOW restore: %s (engine_slot=%d, "
                            "thread=%s, shadow_turns=%d <= incoming_turns=%d; clean anchor "
                            "%s had %d turns)",
                            r_bin, r_sid, tid[:12], len(r_chain), len(inc_chain),
                            bin_fn, len(clean_chain),
                        )
                    else:
                        decision["resolved_from"] = "warm-force-clean-restore"
                        log.info(
                            "the classifier warm-path forced clean restore: %s (engine_slot=%d, thread=%s, "
                            "clean_turns=%d <= incoming_turns=%d)",
                            r_bin, r_sid, tid[:12], len(r_chain), len(inc_chain),
                        )
                else:
                    # attempted-but-failed: metric stays truthful (not counted)
                    decision["resolved_from"] = "warm-force-restore-failed"
            return decision
        except Exception:
            log.debug("the classifier warm-path force restore best-effort failed", exc_info=True)
            return decision
        finally:
            self._emit_classifier_decision(decision)

    def _log_kv_restore_diag(self, *, thread_hash: str, model_tag: str, slot_id,
                             chosen: str, resolved_from: str, clean_bin_present: bool,
                             shadow_bin_present: bool, common_prefix_turns: int,
                             incoming_turns: int, restore_bin) -> None:
        """shadow-diag: one grep-able ``KV_RESTORE`` line per cold restore
        decision + bump the /status shadow_diag ``restores`` counter, correlatable to
        the engine's per-task outcome line via {thread_hash, model_tag, restore_bin}.
        Snapshots the ceiling-GC memory-pressure state (total/over_cap from the last
        GC pass) so a forcing-full restore is joinable to KV pressure at that instant.
        INSTRUMENTATION ONLY — reads state, emits a line, never changes a decision;
        best-effort (never raises). Off the response hot path (spawn/swap-back)."""
        try:
            self._shadow_diag_counts["restores"][chosen] = (
                self._shadow_diag_counts["restores"].get(chosen, 0) + 1)
            snap = self._last_kvgc_snapshot or {}
            # divergence_pos == the first diverging TURN index (== common length);
            # the engine's own get_common_prefix stays the final byte-level authority.
            log.info(
                "KV_RESTORE th=%s model=%s slot=%s chosen=%s resolved_from=%s "
                "clean_bin_present=%s shadow_bin_present=%s common_prefix_turns=%d "
                "divergence_pos=%d incoming_turns=%d restore_bin=%s "
                "kv_total_bytes=%d kv_over_cap=%d",
                thread_hash, model_tag, slot_id, chosen, resolved_from,
                clean_bin_present, shadow_bin_present, common_prefix_turns,
                common_prefix_turns, incoming_turns, restore_bin,
                snap.get("total_bytes", -1), snap.get("over_cap", -1),
            )
        except Exception:
            log.debug("KV_RESTORE diag log best-effort failed", exc_info=True)

    def _log_shadow_byteparity(self, *, thread_id: str, thread_hash: str,
                               model_tag: str, inc_messages: list) -> None:
        """shadow-diag PL-refinement (the load-bearing one): when a
        ``.shadow`` bin is in play, compare the manager's RECONSTRUCTED think-free
        assistant render (from the per-(thread,model) shadow-diag recon store, so it
        survives intervening saves of other threads on a swap-back)
        against the harness's INCOMING resend of that same turn (from
        ``inc_messages``) — {len, FNV hash, match bool, first-diverging byte offset}.

        A ``first_diff_offset >= 0`` points straight at candidate (d)
        saved-but-BYTE-DIVERGES (the reconstructed merges reasoning_content+content
        into ONE <think> block vs the harness's interleaved multi-block) versus (c)
        wrong-bin (where the shadow simply was not selected). Both sides pass through
        the SAME ``_strip_thinking_all`` the shadow SAVE uses, so this measures
        exactly what the save produced. INSTRUMENTATION ONLY: no engine call, no raw
        content logged (lengths + FNV + offset only), best-effort (never raises),
        off the response hot path."""
        try:
            from turbohaul.api.chat_completion import _strip_thinking_all
            bp = self._shadow_diag_counts["byteparity"]
            # recon = the think-free assistant turn the manager SAVED into the shadow,
            # read from the per-(thread_id, model_tag) shadow-diag store so it is CLOBBER-
            # PROOF across intervening shadow saves of OTHER threads. On a swap-back
            # (27b saved -> 35b sub-agent saves -> 27b restore) the single-global
            # _last_shadow_src would already hold the 35b source, blinding (d); this
            # per-key store keeps 27b's. Pure instrumentation — _last_shadow_src + the
            # swap-belt are untouched. None -> no reconstruct source recorded.
            recon = self._byteparity_recon_by_key.get((thread_id, model_tag))
            # resend = the harness's resent assistant-N turn (already think-stripped).
            resend = None
            for m in reversed(inc_messages or []):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    resend = m.get("content")
                    break
            if not isinstance(recon, str) or not isinstance(resend, str):
                bp["no_recon_src"] = bp.get("no_recon_src", 0) + 1
                log.info(
                    "SHADOW_BYTEPARITY th=%s model=%s recon_len=-1 resend_len=%d "
                    "match=na first_diff_offset=-1 note=no-reconstruct-source",
                    thread_hash, model_tag,
                    len(resend) if isinstance(resend, str) else -1)
                return
            recon_tf = _strip_thinking_all(recon)
            resend_tf = _strip_thinking_all(resend)
            fdo = _first_diff_offset(recon_tf, resend_tf)
            match = fdo == -1
            key = "match" if match else "diverge"
            bp[key] = bp.get(key, 0) + 1
            log.info(
                "SHADOW_BYTEPARITY th=%s model=%s recon_len=%d recon_hash=%s "
                "resend_len=%d resend_hash=%s match=%s first_diff_offset=%d",
                thread_hash, model_tag, len(recon_tf), _fnv1a_64(recon_tf),
                len(resend_tf), _fnv1a_64(resend_tf), match, fdo,
            )
        except Exception:
            log.debug("SHADOW_BYTEPARITY diag best-effort failed", exc_info=True)

    def _log_warm_serve_divergence(self, matched, incoming_chain: list, incoming_messages: list, port: int) -> None:
            """log-only divergence capture for prompt-determinism debugging.

            GATED OFF by default (TURBOHAUL_DIVERGENCE_DEBUG != '1' → instant return).
            When opted in: stash previous-turn chain + messages in self._divergence_prev
            (keyed by (thread_id, model_tag)), diff incoming vs previous element-by-element,
            and — ONLY when divergence is detected (Case A) or length-mismatch — dump full
            messages to /var/log/turbohaul/divergence-debug.jsonl (never on healthy turns).

            Case A: chains differ at turn K → divergence in role or content (both hashed).
            Neutral (was 'Case B'): chains identical → log only, no file write (may be
              healthy reuse or Case-B; confirm requires cross-ref engine log for THIS turn
              restored pos_min vs prompt tokens).
            length_mismatch: turn count changed but common prefix matched.

            Extended confirm-capture (tool-opaque divergence only): when Case A divergence
            occurs at a tool-opaque turn (assistant+tool_calls or role=tool), dump both
            the /apply-template render of the FULL incoming prompt AND the clean-bin's
            saved render (via /apply-template on clean_messages slice), byte-diff them,
            and log first differing offset + snippets. This isolates whether the bug is:
            - MATCH → native-reuse is the bug, force clean-restore on tool-call turns
            - DIFFER → apply-template/strip render is non-deterministic, fix the render

            Never-raises: all errors swallowed. No KV changes.
            """
            # Gate: opt-in debug only (never fires in production).
            if os.environ.get("TURBOHAUL_DIVERGENCE_DEBUG", "") != "1":
                return
            # Second gate: deployed sentinel file (PL controls when capture runs).
            import pathlib
            if not pathlib.Path("/var/lib/turbohaul/.divergence_debug").exists():
                return
            try:
                thread_id = getattr(matched, "thread_id", "") or ""
                model_tag = getattr(matched, "model_tag", "") or ""
                key = (thread_id, model_tag)
                prev = self._divergence_prev.get(key)
                prev_chain = prev["chain"] if prev else None
                prev_messages = prev["messages"] if prev else None

                if prev_chain is None:
                    log.info("DIVERGENCE_DUMP: first warm serve for thread=%s model=%s, stashing initial state", thread_id, model_tag)
                    self._divergence_prev[key] = {"chain": list(incoming_chain), "messages": list(incoming_messages)}
                    return

                # Diff chains element-by-element
                min_len = min(len(prev_chain), len(incoming_chain))
                first_divergent_turn = -1
                for i in range(min_len):
                    if prev_chain[i] != incoming_chain[i]:
                        first_divergent_turn = i
                        break

                if first_divergent_turn >= 0:
                    # Case A: divergence in role or content (both hashed)
                    prev_msg = prev_messages[first_divergent_turn] if first_divergent_turn < len(prev_messages) else {}
                    inc_msg = incoming_messages[first_divergent_turn] if first_divergent_turn < len(incoming_messages) else {}
                    def _scan_think(content):
                        return "think" in repr(content) if content else False
                    log.info(
                        "DIVERGENCE_DUMP: case=A thread=%s model=%s first_divergent_turn=%d "
                        "prev_role=%s incoming_role=%s prev_tool_calls=%s incoming_tool_calls=%s "
                        "prev_think=%s incoming_think=%s "
                        "prev_content=%s incoming_content=%s "
                        "prev_chain=%s incoming_chain=%s",
                        thread_id, model_tag, first_divergent_turn,
                        prev_msg.get("role", "?"), inc_msg.get("role", "?"),
                        repr(prev_msg.get("tool_calls"))[:200] if prev_msg.get("tool_calls") else "None",
                        repr(inc_msg.get("tool_calls"))[:200] if inc_msg.get("tool_calls") else "None",
                        _scan_think(prev_msg.get("content")), _scan_think(inc_msg.get("content")),
                        repr(prev_msg.get("content", ""))[:400], repr(inc_msg.get("content", ""))[:400],
                        prev_chain[first_divergent_turn][:16], incoming_chain[first_divergent_turn][:16]
                    )
                    # Dump full messages to JSONL (ONLY when divergence detected — not every turn).
                    debug_file = "/var/log/turbohaul/divergence-debug.jsonl"
                    try:
                        with open(debug_file, "a") as f:
                            entry = {
                                "ts": datetime.now().isoformat(sep=" ", timespec="seconds"),
                                "thread_id": thread_id,
                                "model_tag": model_tag,
                                "first_divergent_turn": first_divergent_turn,
                                "turns": len(incoming_chain),
                                "incoming_chain": incoming_chain,
                                "incoming_messages": incoming_messages,
                                "prev_chain": prev_chain,
                                "prev_messages": prev_messages,
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        log.info("DIVERGENCE_DUMP: full messages dumped to %s", debug_file)
                    except Exception:
                        log.debug("DIVERGENCE_DUMP: file write failed (best-effort)", exc_info=True)

                elif len(prev_chain) != len(incoming_chain):
                    # Length mismatch: turn count changed but common prefix matched.
                    log.info(
                        "DIVERGENCE_DUMP: case=length_mismatch thread=%s model=%s turns_prev=%d turns_incoming=%d",
                        thread_id, model_tag, len(prev_chain), len(incoming_chain)
                    )
                    debug_file = "/var/log/turbohaul/divergence-debug.jsonl"
                    try:
                        with open(debug_file, "a") as f:
                            entry = {
                                "ts": datetime.now().isoformat(sep=" ", timespec="seconds"),
                                "thread_id": thread_id,
                                "model_tag": model_tag,
                                "case": "length_mismatch",
                                "turns_prev": len(prev_chain),
                                "turns_incoming": len(incoming_chain),
                                "incoming_messages": incoming_messages,
                                "prev_messages": prev_messages,
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception:
                        log.debug("DIVERGENCE_DUMP: length_mismatch file write failed", exc_info=True)

                else:
                    # Neutral: chains identical. May be healthy reuse or Case-B (unhashed field
                    # divergence). Confirm Case-B requires cross-ref engine log: restored pos_min
                    # vs prompt tokens on THIS turn. Log only, no file write (would spam on healthy).
                    log.info(
                        "DIVERGENCE_DUMP: chains_identical=True thread=%s model=%s turns=%d "
                        "(Case-B confirm requires cross-ref engine log restored pos_min for this turn)",
                        thread_id, model_tag, len(incoming_chain)
                    )

                # EXTENDED CONFIRM-CAPTURE (B4): decoupled from case=A gate.
                # Fire on ALL branches whenever incoming has a tool-opaque tail
                # beyond the clean-bin prefix (sub-hash divergence, invisible to
                # _prefix_hash_chain). Uses the canonical helper.
                try:
                    if port:
                        slot_meta = getattr(matched, "client_meta", None) or {}
                        found = self._find_clean_bin(port, model_tag, thread_id)
                        if found:
                            _bin_fn, clean_chain, _sid = found
                            clean_len = len(clean_chain)
                            if clean_len > 0 and clean_len <= len(incoming_messages):
                                if _divergent_tail_is_tool_opaque(incoming_messages, clean_len):
                                    # Find the last tool-opaque turn index in the tail for logging
                                    tail_idx = -1
                                    for i in range(clean_len, len(incoming_messages)):
                                        if _turn_is_tool_opaque(incoming_messages[i]):
                                            tail_idx = i
                                    self._spawn_bg(self._dump_apply_template_byte_diff(
                                        port, model_tag, thread_id, incoming_messages, incoming_chain, tail_idx, slot_meta
                                    ))
                except Exception:
                    log.debug("EXTENDED_CAPTURE: tool-opaque tail check best-effort failed", exc_info=True)

                # Stash current state for next turn (keyed by identity, survives Slot replacement).
                self._divergence_prev[key] = {"chain": list(incoming_chain), "messages": list(incoming_messages)}

            except Exception:
                log.debug("DIVERGENCE_DUMP: logging best-effort failed", exc_info=True)

    async def _dump_apply_template_byte_diff(self, port: int, model_tag: str, thread_id: str,
                                             incoming_messages: list, incoming_chain: list,
                                             divergent_turn: int, meta: dict) -> None:
            """extended confirm-capture: byte-diff /apply-template render of FULL
            incoming prompt vs clean-bin slice on tool-opaque divergence turn.

            When clean ⊑ incoming, the clean bin's messages = incoming_messages[:len(clean_chain)].
            We render both via /apply-template (same tool knobs, same strip) and diff the
            resulting prompts. If they MATCH → native-reuse is the bug (force clean-restore).
            If they DIFFER → apply-template/strip is non-deterministic (fix the render).

            Log-only, gated, never-raises.
            """
            try:
                # Find the clean bin for this (thread, model, port)
                found = self._find_clean_bin(port, model_tag, thread_id)
                if not found:
                    return
                _bin_fn, clean_chain, _sid = found
                clean_len = len(clean_chain)
                if clean_len == 0 or clean_len > len(incoming_messages):
                    return

                # Build meta with tool knobs (same as _render_and_prefill_clean_kv)
                meta = meta or {}
                apply_meta = {}
                for _k in _KV_PROBE_TOOL_KNOBS:
                    _v = meta.get(_k)
                    if _v is not None:
                        apply_meta[_k] = _v

                base = f"http://127.0.0.1:{port}"
                import httpx

                async def _render_via_apply_template(messages_subset: list) -> str | None:
                    payload = {"messages": messages_subset}
                    payload.update(apply_meta)
                    try:
                        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=1.0)) as client:
                            r = await client.post(f"{base}/apply-template", json=payload)
                            r.raise_for_status()
                            rendered = (r.json() or {}).get("prompt")
                            if not isinstance(rendered, str) or not rendered:
                                return None
                            return _strip_think_scaffold(rendered, messages_subset)
                    except Exception:
                        return None

                # Full incoming render
                full_render = await _render_via_apply_template(incoming_messages)
                if full_render is None:
                    return

                # Clean-bin slice render (first clean_len messages)
                clean_messages = incoming_messages[:clean_len]
                clean_render = await _render_via_apply_template(clean_messages)
                if clean_render is None:
                    return

                # Byte-diff
                if full_render == clean_render:
                    log.info(
                        "EXTENDED_CAPTURE: MATCH — /apply-template(full) == /apply-template(clean_slice) "
                        "thread=%s model=%s divergent_turn=%d clean_len=%d full_len=%d "
                        "=> native-reuse is the bug (force clean-restore on tool-call turns)",
                        thread_id, model_tag, divergent_turn, clean_len, len(incoming_messages)
                    )
                else:
                    # Find first differing byte
                    first_diff = -1
                    min_len = min(len(full_render), len(clean_render))
                    for i in range(min_len):
                        if full_render[i] != clean_render[i]:
                            first_diff = i
                            break
                    if first_diff == -1 and len(full_render) != len(clean_render):
                        first_diff = min_len

                    # Log snippets around first diff (repr to expose whitespace)
                    snippet_full = repr(full_render[max(0, first_diff-80):first_diff+80])
                    snippet_clean = repr(clean_render[max(0, first_diff-80):first_diff+80])
                    log.info(
                        "EXTENDED_CAPTURE: DIFFER — /apply-template(full) != /apply-template(clean_slice) "
                        "thread=%s model=%s divergent_turn=%d clean_len=%d full_len=%d "
                        "first_diff=%d full_snippet=%s clean_snippet=%s "
                        "=> apply-template/strip render is non-deterministic (fix the render)",
                        thread_id, model_tag, divergent_turn, clean_len, len(incoming_messages),
                        first_diff, snippet_full, snippet_clean
                    )

                    # Also dump full renders to JSONL for offline byte-diff
                    debug_file = "/var/log/turbohaul/divergence-debug.jsonl"
                    try:
                        with open(debug_file, "a") as f:
                            entry = {
                                "ts": datetime.now().isoformat(sep=" ", timespec="seconds"),
                                "thread_id": thread_id,
                                "model_tag": model_tag,
                                "case": "extended_capture_tool_opaque",
                                "divergent_turn": divergent_turn,
                                "clean_len": clean_len,
                                "full_len": len(incoming_messages),
                                "first_diff": first_diff,
                                "full_render": full_render,
                                "clean_render": clean_render,
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception:
                        log.debug("EXTENDED_CAPTURE: jsonl write failed", exc_info=True)

            except Exception:
                log.debug("EXTENDED_CAPTURE: outer best-effort failed", exc_info=True)

    async def _restore_slot_kv(self, port: int, model_tag: str, slot=None) -> None:
        """the engine-op badge work + FP R4 design note: engine_op scoped to the op (see
        _save_slot_kv) — set on entry, reset in the finally on every exit."""
        if slot is not None:
            slot.engine_op = "kv_restore"
        try:
            return await self._restore_slot_kv_inner(port, model_tag, slot)
        finally:
            if slot is not None:
                slot.engine_op = "idle"

    async def _restore_slot_kv_inner(self, port: int, model_tag: str, slot=None) -> None:
        """KV restore using resolve_kv() chokepoint for decision + provenance."""
        if slot is None:
            return
        if '/' in model_tag or '\\' in model_tag or '..' in model_tag:
            log.warning("slot KV: refusing model_tag with unsafe path chars: %r", model_tag)
            return
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        try:
            if not os.path.isdir(SLOT_SAVE_DIR):
                return
            # Incoming identity — SPEC-V2 WAVE A: (session_id, role[, fp8]) via
            # _bin_identity; flows into resolve_kv owner-match + the diag hash, so
            # the whole cold path keys on the session bin. Raw-tid fallback (no
            # session_id OR unlabeled role) = today's behavior byte-for-byte.
            inc_tid = _bin_identity(
                getattr(slot, "thread_id", ""),
                getattr(slot, "client_meta", None),
                getattr(slot, "admission_hash_chain", None))
            # use admission-recorded context size (NOT slot.context at spawn)
            inc_len = getattr(slot, "admission_ctx_len", 0)
            # the classifier: admission-recorded incoming turn-hash chain — the prefix-validity
            # gate compares this against each bin's saved hash_chain. Absent (older
            # submit path / monolithic client) -> [] -> gate SKIPS (fail-safe).
            inc_chain = getattr(slot, "admission_hash_chain", []) or []
            # crit3: incoming turns (client_meta["messages"], 1:1 aligned
            # with inc_chain) for the tool-opaque tail guard before the POST below.
            # Absent -> [] -> guard inert (safe-degrade to today's restore).
            inc_messages = (getattr(slot, "client_meta", None) or {}).get("messages") or []

            files = os.listdir(SLOT_SAVE_DIR)
            bins = [fn for fn in files if fn.startswith(f"{model_tag}.") and fn.endswith(".bin")]
            if not bins:
                # the classifier P5: no saved bin for this model at all -> a first-seen identity
                # (or a distinct the design sub-agent nonce) with no anchor yet. FRESH.
                self._emit_classifier_decision({
                    "event_type": "sub-agent", "action": "fresh",
                    "resolved_from": "restore-no-bins", "clean_bin_id": None,
                    "common_prefix_turns": 0, "incoming_turns": len(inc_chain),
                    "forced_clean_restore": False,
                })
                self._log_kv_restore_diag(
                    thread_hash=self._thread_hash(inc_tid or ""), model_tag=model_tag,
                    slot_id=getattr(slot, "slot_id", None), chosen="fresh",
                    resolved_from="restore-no-bins", clean_bin_present=False,
                    shadow_bin_present=False, common_prefix_turns=0,
                    incoming_turns=len(inc_chain), restore_bin=None)
                return

            # the classifier P5 cold-path observability accumulators (see emit after the loop).
            owned_seen = False    # any bin for THIS identity (owner ok)
            diverged_seen = False  # any owned bin that was NOT a valid prefix
            restore = []
            for bin_fn in bins:
                inner = bin_fn[len(f"{model_tag}."):-4]
                dot = inner.find(".slot")
                if dot < 0:
                    continue
                f_th = inner[:dot]
                sid_str = inner[dot+5:]
                if not sid_str.isdigit():
                    continue
                sid = int(sid_str)
                # Meta file is always the bin file with .json extension (round-trip-safe)
                jpath = os.path.join(SLOT_SAVE_DIR, bin_fn[:-4] + ".json")
                meta = None
                if os.path.exists(jpath):
                    try:
                        with open(jpath) as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                if meta is None:
                    continue
                # SPEC-V2 consistency stamp: reject a candidate whose bin bytes don't
                # match what the meta was stamped against (absent field => no check).
                _claim = meta.get("bin_bytes")
                if _claim is not None:
                    try:
                        if os.path.getsize(os.path.join(SLOT_SAVE_DIR, bin_fn)) != int(_claim):
                            continue
                    except (OSError, ValueError):
                        continue

                # R-COMP (the operator): a stale-marked sidecar (labeled is_compression
                # arrived for this session) is ABSENT — the next is_main turn
                # recomputes fresh instead of restoring the pre-compression copy
                # (even where it would still prefix-match).
                if meta.get("stale"):
                    continue
                # Owner validation: restore must validate against SAVED file's owner
                saved_tid = meta.get("thread_id", "")
                saved_tokens = meta.get("prompt_tokens", 0)
                saved_len = meta.get("prompt_len", 0)
                saved_chain = meta.get("hash_chain", [])
                saved_ph = meta.get("prompt_hash", "")
                saved_clean = bool(meta.get("clean_prefix", False))

                # Compute cache age for timing-gap guard
                try:
                    cache_age = time.time() - os.path.getmtime(os.path.join(SLOT_SAVE_DIR, bin_fn))
                except OSError:
                    cache_age = float("inf")

                # Policy decision (admission-size based, no hash gates)
                decision = resolve_kv("restore", {
                    "thread_id": inc_tid,
                    "model_tag": model_tag,
                    "slot_id": sid,
                    "port": port,
                }, {
                    "saved_tokens": saved_tokens,
                    "saved_len": saved_len,
                    "incoming_len": inc_len,
                    "cache_age_s": cache_age,
                    "saved_thread_id": saved_tid,
                    # the classifier: prefix-validity inputs (replaces the length compaction gate).
                    "saved_chain": saved_chain,
                    "incoming_chain": inc_chain,
                })
                log.info("slot KV restore decision: %s (saved_tid='%s')", decision, saved_tid)
                # the classifier P5: bucket this bin's outcome for the cold-path event emit.
                rf = decision.resolved_from
                if rf != "restore-owner-mismatch":
                    owned_seen = True
                if rf in ("restore-diverged-fresh", "restore-physics-belt-saved-longer"):
                    diverged_seen = True
                if decision.do_it:
                    restore.append((sid, bin_fn, saved_tokens, saved_clean, saved_chain))
            if not restore:
                # the classifier P5: bins existed but none is a valid prefix. If an owned bin
                # diverged -> COMPRESSION (harness rewrote an early turn). Else the
                # bins belong to other identities -> SUB-AGENT (own-anchor, no cross-
                # restore). Either way: FRESH.
                if diverged_seen:
                    et, rf = "compression", "restore-diverged-fresh"
                elif owned_seen:
                    et, rf = "compression", "restore-guard-skip"
                else:
                    et, rf = "sub-agent", "restore-no-anchor-for-identity"
                self._emit_classifier_decision({
                    "event_type": et, "action": "fresh", "resolved_from": rf,
                    "clean_bin_id": None, "common_prefix_turns": 0,
                    "incoming_turns": len(inc_chain), "forced_clean_restore": False,
                })
                # _find_shadow_bin is documented never-raises (FS/JSON errors -> None);
                # already inside this method's outer best-effort try.
                _sh_present = self._find_shadow_bin(port, model_tag, inc_tid) is not None
                self._log_kv_restore_diag(
                    thread_hash=self._thread_hash(inc_tid or ""), model_tag=model_tag,
                    slot_id=getattr(slot, "slot_id", None), chosen="fresh",
                    resolved_from=rf, clean_bin_present=owned_seen,
                    shadow_bin_present=_sh_present, common_prefix_turns=0,
                    incoming_turns=len(inc_chain), restore_bin=None)
                return
            # B2 (#103): restore ONLY the single best bin.
            # Restoring multiple mismatched bins under one identity produced +27038 stale -> CLEAR.
            # Option A: PREFER a clean_prefix bin (no generated <think> tail) over a
            # possibly-larger polluted one — a leftover polluted bin would otherwise win on token
            # count and re-introduce the stale>n_rs_seq CLEAR. Tie-break by largest token count.
            restore.sort(key=lambda r: (r[3], r[2]), reverse=True)
            sid, bin_fn, saved_n, _clean, win_chain = restore[0]
            # --- P2 (DURABLE MANAGER B): ring-aware COLD restore (flag-gated) ---
            # When TURBOHAUL_DURABLE_RING=ON, select the newest matching (role,session) ring bin
            # that is a VALID PREFIX of inc_chain (same gate as default path), via _select_ring_bin.
            # Cold/swap-back family ONLY. Default-OFF = byte-identical.
            # Option A (per PL): OVERRIDE the default-selection winner BEFORE the cold-path preference so the ring bin
            # flows through the whole pipeline. Keep default sid (target slot for POST); only override
            # bin_fn, saved_n, _clean, win_chain from ring. Shadow-preference block then operates on
            # the ring's chain (ring vs shadow precedence: shadow of ring bin still wins for strict-extension).
            if _durable_ring_enabled() and slot is not None:
                try:
                    ring_key = _durable_ring_key(getattr(slot, "client_meta", None))
                    ring = self._durable_ring_index.get(ring_key, []) if ring_key else []
                    if ring_key and ring:
                        # Select ring entry that matches model_tag + non-empty chain + valid PREFIX of inc_chain
                        ring_entry = _select_ring_bin(ring, inc_chain, model_tag)
                        if ring_entry:
                            ring_bin_fn = ring_entry.get("bin_fn")
                            ring_meta_fn = ring_entry.get("meta_fn")
                            ring_clean = bool(ring_entry.get("clean_prefix", False))
                            ring_chain = ring_entry.get("hash_chain", [])
                            # Use saved_tokens for token count consistency with default path (prompt_len is turn count)
                            ring_saved_n = ring_entry.get("saved_tokens", ring_entry.get("prompt_len", 0))
                            # Validate ring bin exists on disk and matches this model_tag
                            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
                            ring_bin_path = os.path.join(SLOT_SAVE_DIR, ring_bin_fn) if ring_bin_fn else None
                            ring_meta_path = os.path.join(SLOT_SAVE_DIR, ring_meta_fn) if ring_meta_fn else None
                            if (ring_bin_path and ring_meta_path and
                                os.path.exists(ring_bin_path) and os.path.exists(ring_meta_path)):
                                # OVERRIDE the default-selection winner (keep default sid = target slot for POST)
                                bin_fn = ring_bin_fn
                                saved_n = ring_saved_n
                                _clean = ring_clean
                                win_chain = ring_chain
                                log.info("DURABLE_RING cold restore: ring bin overrides default key=%s bin=%s clean=%s turns=%d",
                                         ring_key, ring_bin_fn, ring_clean, len(ring_chain))
                        else:
                            log.info("DURABLE_RING cold restore: no ring entry passes prefix gate for key=%s -> keep default",
                                     ring_key)
                except Exception:
                    log.debug("DURABLE_RING cold restore ring lookup failed (best-effort); using default selection", exc_info=True)
            # --- (critical item): COLD-path SHADOW restore-preference -------
            # The clean/normal selection ABOVE is UNCHANGED and alone decides the winning
            # clean anchor (sid/bin_fn/win_chain). Here we ONLY upgrade the restore TARGET
            # on the COLD (wave-return / swap-back) path: prefer the byte-matching think-
            # free `.shadow` bin over the winning clean anchor IFF it passes the SAME
            # prefix-validity bar the clean path uses (_is_prefix_match vs inc_chain).
            # COLD-FRESHNESS (Option A, PL-greenlit): NO length/no-downgrade condition
            # on cold — a valid-prefix think-free shadow that is SHORTER than the clean
            # STILL WINS (rationale at the gate below). A valid prefix => the next
            # decode STRICT-EXTENDS the think-free state (skips reprefilling the <think>
            # tail), reprefilling ONLY the appended/missing tail.
            #
            # DISTINCT GATE (_shadow_cold_restore_enabled == TURBOHAUL_SHADOW_COLD_RESTORE,
            # default ON when SHADOW_REPREFILL=1) — NOT the WARM _shadow_restore_prefer_
            # enabled (held 0): the warm seam wants native reuse, the cold swap-back has
            # NO warm KV so it wants the shadow. The `.shadow` bin is found via
            # _find_shadow_bin (its OWN `.shadow.json` matcher) so the numeric-sid
            # `.isdigit` parse for normal bins above is UNTOUCHED — the `.shadow` bypass
            # is scoped to the shadow-marker lookup only (constraint #3).
            #
            # UPGRADE-ONLY / never worse than today (constraint #2): a missing / non-
            # prefix / stale shadow -> keep EXACTLY the clean anchor chosen above (a
            # SHORTER valid shadow is now PREFERRED on cold, see freshness rationale).
            # And the engine's get_common_prefix stays the FINAL authority on the
            # next decode — even a wrongly-preferred shadow only makes the engine reprefill
            # the divergent tail (a REPREFILL, never a wrong answer), identical safety to
            # the clean restore.
            #
            # ENGINE-LIMIT CAVEAT (honest — NOT a fix miss): restoring the think-free
            # [1..N] shadow lets the MAIN history reuse, but the qwen MTP recurrent ctx
            # (n_rs_seq=2) can only rewind so far — if the appended sub-result TAIL exceeds
            # n_rs_seq=2 the engine CLEARs regardless, and swap-back prompt_eval == full
            # context. That is the recurrent-rewind LIMIT (the operator's "only the extra sub-
            # context slow-loads" boundary), not this hunk failing. SUCCESS = engine log
            # "strict extension", prompt_eval == sub-result TAIL tokens; recurrent-limit =
            # engine log "CLEAR"/do_reset, full prefill. The manager restores the RIGHT
            # bin; the rewind depth is the engine's.
            r_bin, r_sid, r_chain, r_kind = bin_fn, sid, win_chain, "clean"
            # --- crit3: COLD-path TOOL-tail restore SKIP (Option A) ------
            # GUARD FIRST — BEFORE the shadow-preference below — against the winning
            # CLEAN anchor's common depth (win_chain: the deterministic think-free
            # prefix; same `common` semantics as the warm seam). _prefix_hash_chain is
            # BLIND to tool turns (assistant tool_calls / role=="tool" / null content),
            # so if the divergent tail beyond that common prefix is tool-opaque, the
            # harness re-serializes that region nondeterministically -> restoring ANY
            # bin here (the clean anchor OR the cold-wire shadow below — its tool turns
            # are EQUALLY hash-invisible, constraint #3) would POST token-stale KV ->
            # engine stale>n_rs_seq CLEAR -> full reprefill. SKIP the whole restore ->
            # the re-spawned slot stays un-restored -> fresh prefill / engine native
            # checkpoint (safe-degrade). Returning here gates BOTH POSTs. A TEXT/think
            # tail (no tool turn beyond common) falls THROUGH to the freshness gate and
            # still restores, so crit2's cold shadow reuse stays intact. Flag-gated
            # (default ON). Gating on win_chain (not the post-preference target) makes
            # the skip decision independent of which bin the freshness gate picks.
            if _tooltail_restore_skip_enabled():
                _common_tt = 0
                for a, b in zip(win_chain or [], inc_chain, strict=False):
                    if a != b:
                        break
                    _common_tt += 1
                if _divergent_tail_is_tool_opaque(
                        inc_messages, _common_tt,
                        scan_covered=_tooltail_scan_covered_enabled()):
                    self._kv_tooltail_skip_counts["cold"] = (
                        self._kv_tooltail_skip_counts.get("cold", 0) + 1)
                    log.info(
                        "crit3 cold TOOL-tail restore SKIP: divergent tail "
                        "beyond common=%d is tool-opaque -> NOT restoring %s; safe-"
                        "degrade to fresh prefill (thread=%s, incoming_turns=%d)",
                        _common_tt, bin_fn, (inc_tid or "")[:12], len(inc_chain),
                    )
                    self._emit_classifier_decision({
                        "event_type": "guard-skip", "action": "fresh",
                        "resolved_from": "cold-tooltail-skip", "clean_bin_id": bin_fn,
                        "common_prefix_turns": _common_tt,
                        "incoming_turns": len(inc_chain),
                        "forced_clean_restore": False,
                    })
                    _sh_present_tt = (
                        self._find_shadow_bin(port, model_tag, inc_tid) is not None)
                    self._log_kv_restore_diag(
                        thread_hash=self._thread_hash(inc_tid or ""),
                        model_tag=model_tag, slot_id=getattr(slot, "slot_id", None),
                        chosen="fresh", resolved_from="cold-tooltail-skip",
                        clean_bin_present=True, shadow_bin_present=_sh_present_tt,
                        common_prefix_turns=_common_tt,
                        incoming_turns=len(inc_chain), restore_bin=bin_fn)
                    return
            # --- COLD-FRESHNESS (Option A, PL-greenlit; distinct from crit3
            # above — this is the crit2 ROOT). On a NON-tool tail, prefer a valid-PREFIX
            # think-free `.shadow` over the winning clean anchor REGARDLESS OF LENGTH: a
            # SHORTER valid shadow STILL WINS on cold. Its missing tail just reprefills
            # (small, bounded seq_rm), whereas the with-<think> clean coin-flips a FULL
            # ~81k CLEAR (the generated <think> tail > n_rs_seq=2). Think-free-
            # correctness beats raw length on cold. The `>= len(clean)` no-downgrade was
            # a WARM-path heuristic (prefer the longer valid bin when a warm KV already
            # holds a continuation) mis-applied to cold — the cold swap-back has NO warm
            # KV, so freshness is what matters, not length. KEPT: _is_prefix_match
            # (valid-prefix ONLY), the owner/model/port scoping via _find_shadow_bin,
            # clean-fallback when there is NO valid-prefix shadow, and the engine
            # get_common_prefix backstop (FINAL authority — a wrongly-preferred shadow
            # only costs a reprefill, never a wrong answer). The WARM path's no-
            # downgrade (_maybe_force_clean_restore) is UNTOUCHED.
            if _shadow_cold_restore_enabled():
                try:
                    _sh = self._find_shadow_bin(port, model_tag, inc_tid)
                    # review note B: drop the length no-downgrade ONLY when the winning anchor is a
                    # WITH-<think> (polluted) bin (_clean == False) -> any valid-prefix shadow beats
                    # its think-tail coin-flip regardless of length. If the anchor is a THINK-FREE
                    # clean_prefix bin (_clean == True) that is LONGER, KEEP the length-guard: the
                    # longer clean strict-extends, whereas a shorter shadow would reprefill the gap
                    # for nothing (latency regression). So: prefer shadow iff valid-prefix AND
                    # (anchor is polluted OR shadow is not shorter than the clean).
                    if (_sh is not None and _is_prefix_match(_sh[1], inc_chain)
                            and (not _clean or len(_sh[1]) >= len(win_chain or []))):
                        r_bin, r_chain, r_sid, r_kind = _sh[0], _sh[1], _sh[2], "shadow"
                        self._kv_shadow_restore_counts["cold_preferred"] = (
                            self._kv_shadow_restore_counts.get("cold_preferred", 0) + 1)
                    else:
                        self._kv_shadow_restore_counts["cold_clean_fallback"] = (
                            self._kv_shadow_restore_counts.get("cold_clean_fallback", 0) + 1)
                        if _sh is not None:
                            log.info(
                                "cold shadow restore-preference: shadow present but NOT "
                                "a valid prefix (shadow_turns=%d, clean_turns=%d) -> "
                                "clean anchor for thread=%s",
                                len(_sh[1]), len(win_chain or []), (inc_tid or "")[:12])
                except Exception:
                    log.debug("cold shadow restore-preference lookup failed (best-effort); "
                              "using clean anchor", exc_info=True)
            # F5: status-check the restore POST so the metric is truthful — a 500'd
            # restore leaves the slot un-restored (fresh prefill), NOT a continuation.
            # Caught locally so the truthful (failed) decision still emits below.
            restore_ok = False
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=120.0, write=10.0, pool=1.0)) as client:
                    resp = await client.post(f"http://127.0.0.1:{port}/slots/{r_sid}?action=restore", json={"filename": r_bin})
                    resp.raise_for_status()
                restore_ok = True
                # B3: truthful — this is the restore POST; actual reuse is decided by the engine's
                # get_common_prefix on next decode (engine logs "strict extension" reuse vs "CLEAR").
                log.info("slot KV restore POSTed: %s (kind=%s, saved ~%d tokens; engine determines actual n_past)", r_bin, r_kind, saved_n)
                # C (3-strikes design): remember which bin this
                # port last restored so a subsequent dead-holder detection can
                # attribute the death and auto-quarantine a repeat offender.
                try:
                    if not hasattr(self, "_last_restored_bin"):
                        self._last_restored_bin = {}
                        self._bin_death_strikes = {}
                    self._last_restored_bin[port] = r_bin
                except Exception:
                    pass
                # P1 (DURABLE MANAGER B): residency tag update + would-be reload log at cold restore.
                if _durable_ring_enabled() and slot is not None:
                    try:
                        ring_key = _durable_ring_key(getattr(slot, "client_meta", None))
                        resident = self._residents.get(model_tag)
                        if ring_key and resident:
                            old_tag = resident.resident_state_tag
                            resident.resident_state_tag = ring_key
                            # Would-reload check (logs if we WOULD reload when resident_tag != request_tag)
                            # Fires ONLY when old_tag is SET and differs (skip None) — consistent with warm sites
                            if old_tag and old_tag != ring_key:
                                self._durable_ring_counts["would_reload"] = self._durable_ring_counts.get("would_reload", 0) + 1
                                log.info("DURABLE_RING would_reload (site=cold): key=%s resident_old_tag=%s new_tag=%s ring_len=%d",
                                         ring_key, old_tag, ring_key, len(self._durable_ring_index.get(ring_key, [])))
                    except Exception:
                        log.debug("DURABLE_RING residency tag update best-effort failed (site=cold)", exc_info=True)
            except Exception:
                log.warning("slot KV restore POST failed: %s (engine_slot=%d)", r_bin, r_sid, exc_info=True)
            # the classifier P5: a cold/swap-back restore that extends a saved prefix = a
            # CONTINUATION reuse (the incoming extends persisted context). warm_chain
            # is unknown on a fresh spawn, so classify vs the RESTORED bin's chain.
            _common = 0
            for a, b in zip(r_chain or [], inc_chain, strict=False):
                if a != b:
                    break
                _common += 1
            # F2 wave-return IS the cold path. When the restore POST landed
            # and the restored bin is think-free (the clean_prefix anchor OR a `.shadow`
            # bin), this cold restore = a wave-return. Tag it distinctly + bump the cold
            # counter. event_type stays "continuation" so the events dict + forced counter
            # are UNCHANGED (no double-count: _emit_classifier_decision counts by
            # event_type / forced_clean_restore, never by resolved_from). shadow vs clean
            # are MUTUALLY EXCLUSIVE (one restored bin) so wave_return bumps exactly once.
            # Separate/additional to forced (WARM-only).
            _cold_shadow = bool(restore_ok and r_kind == "shadow")
            _cold_clean = bool(restore_ok and _clean and not _cold_shadow)
            if _cold_shadow or _cold_clean:
                self._kv_classifier_wave_return += 1
            _resolved_from = ("wave-return-shadow-restore" if _cold_shadow
                              else ("wave-return-clean-restore" if _cold_clean
                                    else ("restore-prefix-valid" if restore_ok
                                          else "restore-post-failed")))
            self._emit_classifier_decision({
                "event_type": "continuation" if restore_ok else "guard-skip",
                "action": "restore" if restore_ok else "fresh",
                "resolved_from": _resolved_from,
                "clean_bin_id": r_bin,
                "common_prefix_turns": _common, "incoming_turns": len(inc_chain),
                "forced_clean_restore": False,
            })
            # shadow-diag: name the cold-restore choice (candidate c/d) +
            # correlate to KV pressure. chosen distinguishes a think-free shadow, a
            # think-free clean anchor, a with-<think> clean bin, or fresh (post-fail).
            if not restore_ok:
                _chosen = "fresh"
            elif r_kind == "shadow":
                _chosen = "chose_shadow"
            elif _clean:
                _chosen = "chose_clean"
            else:
                _chosen = "chose_clean_withthink"
            # _find_shadow_bin is documented never-raises; inside the outer best-effort try.
            _shadow_present = self._find_shadow_bin(port, model_tag, inc_tid) is not None
            self._log_kv_restore_diag(
                thread_hash=self._thread_hash(inc_tid or ""), model_tag=model_tag,
                slot_id=getattr(slot, "slot_id", None), chosen=_chosen,
                resolved_from=_resolved_from, clean_bin_present=bool(_clean),
                shadow_bin_present=_shadow_present, common_prefix_turns=_common,
                incoming_turns=len(inc_chain), restore_bin=r_bin)
            # PL KEY REFINEMENT: when a `.shadow` bin is in play, byte-compare the
            # manager's reconstructed think-free render vs the harness resend region
            # so candidate (d) saved-but-DIVERGES is observable vs (c) wrong-bin.
            if _shadow_present:
                self._log_shadow_byteparity(
                    thread_id=inc_tid, thread_hash=self._thread_hash(inc_tid or ""),
                    model_tag=model_tag, inc_messages=inc_messages)
        except Exception:
            log.debug("slot KV restore best-effort failed for %s", model_tag, exc_info=True)

    async def _reload_matching_state_before_serve(
        self, port: int, model_tag: str, slot, inc_chain: list, inc_messages: list | None = None
    ) -> tuple[bool, tuple[str, str] | None]:
        """P3 (DURABLE MANAGER B): reload-before-serve on the WARM path.
        
        When TURBOHAUL_DURABLE_RING=ON, BEFORE serving a WARM follow-up, if the
        physically-resident (role,session) does NOT match the incoming request's
        (role,session), RELOAD the matching ring state so the engine residency
        matches the request head — THEN proceed with the existing warm path.
        
        CRITICAL: Only fires on RESIDENCY-TAG MISMATCH (old_tag is SET and != ring_key).
        If old_tag == ring_key (resident already matches), do NOTHING (let existing
        warm path run untouched). If old_tag is None (unknown resident), skip - safe default to avoid
        spurious CLEAR).
        
        Returns (reloaded: bool, ring_key: tuple[str,str] | None).
        Caller MUST advance resident_state_tag to ring_key after this call if ring_key is set.
        """
        if not _durable_ring_enabled():
            return (False, None)
        try:
            ring_key = _durable_ring_key(getattr(slot, "client_meta", None))
            resident = self._residents.get(model_tag)
            old_tag = getattr(resident, "resident_state_tag", None) if resident else None
            
            # ONLY fire on explicit mismatch (old_tag SET and != ring_key)
            # old_tag is None -> skip (unknown resident; safe default avoids spurious CLEAR)
            if not ring_key or not old_tag or old_tag == ring_key:
                return (False, ring_key)
            
            ring = self._durable_ring_index.get(ring_key, [])
            if not ring:
                return (False, ring_key)
            
            # Select ring entry that matches model_tag + non-empty chain + valid PREFIX of inc_chain
            ring_entry = _select_ring_bin(ring, inc_chain, model_tag)
            if not ring_entry:
                log.debug("DURABLE_RING warm reload: no ring entry passes prefix gate for key=%s", ring_key)
                return (False, ring_key)
            
            ring_bin_fn = ring_entry.get("bin_fn")
            ring_meta_fn = ring_entry.get("meta_fn")
            ring_chain = ring_entry.get("hash_chain", [])
            ring_clean = bool(ring_entry.get("clean_prefix", False))
            
            # Validate ring bin exists on disk
            from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
            ring_bin_path = os.path.join(SLOT_SAVE_DIR, ring_bin_fn) if ring_bin_fn else None
            ring_meta_path = os.path.join(SLOT_SAVE_DIR, ring_meta_fn) if ring_meta_fn else None
            if not (ring_bin_path and ring_meta_path and os.path.exists(ring_bin_path) and os.path.exists(ring_meta_path)):
                log.debug("DURABLE_RING warm reload: ring bin/meta not found on disk for key=%s", ring_key)
                return (False, ring_key)
            
            # TOOL-OPAQUE-TAIL GUARD (mirrors COLD path): if divergent tail is tool-opaque,
            # skip reload to avoid engine CLEAR (hash validates prefix but byte-diverges on tool turns)
            if inc_messages is not None:
                from turbohaul.kv_classify import _is_prefix_match
                # Find common prefix length between ring_chain and inc_chain
                common = 0
                for a, b in zip(ring_chain, inc_chain, strict=False):
                    if a != b:
                        break
                    common += 1
                if common < len(ring_chain):
                    # There IS a divergent tail beyond common prefix
                    from turbohaul.kv_classify import _divergent_tail_is_tool_opaque
                    if _divergent_tail_is_tool_opaque(inc_messages, common):
                        log.info("DURABLE_RING warm reload: tool-opaque tail beyond common=%d -> SKIP reload for key=%s", common, ring_key)
                        return (False, ring_key)
            
            # Perform the restore POST (same mechanism as cold path)
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=120.0, write=10.0, pool=1.0)) as client:
                    # Use slot_id from slot (target slot for POST)
                    sid = getattr(slot, "slot_id", None)
                    if sid is None:
                        return (False, ring_key)
                    resp = await client.post(
                        f"http://127.0.0.1:{port}/slots/{sid}?action=restore",
                        json={"filename": ring_bin_fn}
                    )
                    resp.raise_for_status()
                
                # Update residency tag (caller will also advance after this returns)
                resident.resident_state_tag = ring_key
                log.info("DURABLE_RING warm reload: key=%s bin=%s clean=%s turns=%d -> resident updated",
                         ring_key, ring_bin_fn, ring_clean, len(ring_chain))
                return (True, ring_key)
                
            except Exception:
                log.debug("DURABLE_RING warm reload: restore POST failed for key=%s", ring_key, exc_info=True)
                return (False, ring_key)
                
        except Exception:
            log.debug("DURABLE_RING warm reload best-effort failed", exc_info=True)
            return (False, None)

    def _live_protected_thread_hashes(self) -> set[str]:
        """thread_hash set whose KV ``.bin`` + sidecar MUST NOT be GC-evicted:
        the active slot, every in-flight slot, the idle-held thread, and every
        resident's idle thread (grep ``_active_slot`` / ``_idle_thread_id``).

        Protecting the whole live thread (not just its clean anchor) covers the
        constraint "never evict a clean_prefix bin belonging to a live thread"
        AND its non-clean bins. The set is transient — a thread drops out once it
        goes idle/dead, so this cannot leak unboundedly. Best-effort: never
        raises; an empty/unknown thread_id is skipped (``nothread`` is never
        protected, so it cannot silently defeat the ceiling).
        """
        hashes: set[str] = set()
        try:
            def _add(tid: str | None, cm: dict | None = None, chain: list | None = None) -> None:
                # SPEC-V2 WAVE A: protect BOTH the (session,role) identity hash and
                # the raw thread_id hash (legacy bins during the one-time migration).
                if tid:
                    hashes.add(self._thread_hash(tid))
                ident = _bin_identity(tid or "", cm, chain)
                if ident and ident != (tid or ""):
                    hashes.add(self._thread_hash(ident))
            active = self._active_slot
            _add(getattr(active, "thread_id", None) if active is not None else None,
                 getattr(active, "client_meta", None) if active is not None else None,
                 getattr(active, "admission_hash_chain", None) if active is not None else None)
            for s in list(self._inflight):
                _add(getattr(s, "thread_id", None), getattr(s, "client_meta", None),
                     getattr(s, "admission_hash_chain", None))
            _add(self._idle_thread_id, self._idle_client_meta)
            for r in list(self._residents.values()):
                _add(getattr(r, "idle_thread_id", None), getattr(r, "idle_client_meta", None))
        except Exception:
            log.debug("live protected thread-hash scan failed (best-effort)", exc_info=True)
        return hashes

    def _persist_kvcache_snapshot(self) -> dict:
        """Best-effort snapshot of the persist KV cache directory (SSD) for FE display.

        Returns {total_bytes, cap_bytes, file_count, headroom_bytes, over_cap: bool}.
        Uses the runtime config `persist.max_bytes` (default 40 GiB). All I/O is
        synchronous + best-effort; an error returns zeros (FE shows '—').
        """
        try:
            from turbohaul.subprocess_mgr import SLOT_PERSIST_DIR
            if not os.path.isdir(SLOT_PERSIST_DIR):
                return {
                    "total_bytes": 0,
                    "cap_bytes": int(self.runtime.persist.max_bytes),
                    "file_count": 0,
                    "headroom_bytes": int(self.runtime.persist.max_bytes),
                    "over_cap": False,
                }
            cap = int(self.runtime.persist.max_bytes)
            total = 0
            count = 0
            for fn in os.listdir(SLOT_PERSIST_DIR):
                if not fn.endswith(".bin"):
                    continue
                fpath = os.path.join(SLOT_PERSIST_DIR, fn)
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                total += st.st_size
                count += 1
            headroom = cap - total if cap > 0 else 0
            return {
                "total_bytes": total,
                "cap_bytes": cap,
                "file_count": count,
                "headroom_bytes": max(0, headroom),
                "over_cap": cap > 0 and total > cap,
            }
        except Exception:
            log.debug("persist KV cache snapshot best-effort failed", exc_info=True)
            return {
                "total_bytes": 0,
                "cap_bytes": 0,
                "file_count": 0,
                "headroom_bytes": 0,
                "over_cap": False,
            }

    async def _gc_kv_cache(self, max_age_hours: float = 6.0, max_files: int = 100) -> int:
        """Background GC for orphaned KV cache files.

        Reclaims .bin/.json pairs that are older than max_age_hours or exceed
        max_files (LRU by mtime), AND enforces a global TOTAL-BYTES ceiling
        (env ``TURBOHAUL_KVCACHE_MAX_BYTES``, default 20 GiB) by evicting
        oldest-first until the summed .bin size is back under the cap. OFF the
        hot path — call periodically from the background sweeper, NOT from
        save/restore.

        NEVER evicts the per-group pinned clean_prefix anchor, nor ANY bin
        owned by a live thread (active / in-flight / idle-held / resident-idle,
        see ``_live_protected_thread_hashes``). Best-effort throughout: a
        vanished file, a bad stat, or a permission error is caught + logged,
        never raised into the sweeper loop.

        Returns number of .bin files deleted.
        """
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        try:
            if not os.path.isdir(SLOT_SAVE_DIR):
                return 0
            now = time.time()
            # Live-thread guard (constraint): resolve the protected thread_hash
            # set ONCE per GC pass from the manager's live state.
            protected_hashes = self._live_protected_thread_hashes()
            # (a): the keep-warm recency window ("IDLE_HOT_S", ~1800s)
            # used to gate the clean-anchor pin below — a DEAD anchor still inside
            # this window stays pinned (may resume shortly); once outside it, a
            # dead anchor is no longer exempt from age-prune/ceiling.
            idle_hot_s = self.runtime.queue.idle_hot_load_seconds
            # the classifier #2 + F4 (operator request): PIN the clean_prefix bin — it is BOTH the
            # classifier's comparison anchor AND the warm-path restore source; evicting
            # it silently disarms the whole classifier (no anchor -> every request
            # FRESH). BUT (F4) pin only the ONE bin `_find_clean_bin` would actually
            # pick per (model_tag, thread_hash) = NEWEST/LONGEST; every OTHER clean bin
            # (superseded same-thread saves, older-port orphans — never selected) is a
            # NORMAL evictable candidate, else one full-KV dump leaks per save forever
            # (kvcache already ~39GB). First pass: read metas, elect the per-group pin,
            # and remember each bin's thread_hash so the live-thread guard can match it.
            entries = []  # (fn, fpath, mtime, size)
            hash_by_path: dict[str, str] = {}  # fpath -> thread_hash (from sidecar)
            # shadow-diag: fpath -> model_tag (from sidecar) for the
            # SHADOW_EVICT line (the sidecar is unlinked with the bin, so capture now).
            model_by_path: dict[str, str] = {}
            clean_pin = {}  # (model_tag, thread_hash) -> ((prompt_len, mtime), fpath)
            for fn in os.listdir(SLOT_SAVE_DIR):
                if not fn.endswith(".bin"):
                    continue
                fpath = os.path.join(SLOT_SAVE_DIR, fn)
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                # a later phase: account the PAIR (.bin + .bin.ckpt), not just
                # the .bin — the engine's ckpt ladder sidecar is comparable in size
                # (observed 0.5-2.2x of the bin) and was INVISIBLE to the bytes
                # ceiling: bins summed 'under cap' while the tmpfs sat at 100%,
                # so saves/ckpt-writes silently truncated (ENOSPC) and restored
                # ladders broke -> mass rollbacks (the 2026-07-09 violation).
                _sz = st.st_size
                try:
                    _sz += os.stat(fpath + ".ckpt").st_size
                except OSError:
                    pass
                entries.append((fn, fpath, st.st_mtime, _sz))
                meta = None
                try:
                    with open(fpath[:-4] + ".json") as _mf:
                        meta = json.load(_mf)
                except (OSError, ValueError):
                    meta = None
                if meta:
                    th = meta.get("thread_hash")
                    if th:
                        hash_by_path[fpath] = th
                    _mtag = meta.get("model_tag")
                    if _mtag:
                        model_by_path[fpath] = _mtag
                    # Only a bin we can POSITIVELY confirm clean (+ has a thread_hash to
                    # group on) is eligible for the per-group pin; anything else evictable.
                    if meta.get("clean_prefix") and th:
                        # (a): elect into the pin only if `th` is LIVE or
                        # still inside the keep-warm recency window — a DEAD anchor
                        # past that window is NOT pinned, so it falls through
                        # `_protected` to the age-prune + ceiling below.
                        if th in protected_hashes or st.st_mtime > (now - idle_hot_s):
                            grp = (meta.get("model_tag", ""), th)
                            rank = (int(meta.get("prompt_len", 0) or 0), st.st_mtime)  # longest, tie newest
                            cur = clean_pin.get(grp)
                            if cur is None or rank > cur[0]:
                                clean_pin[grp] = (rank, fpath)
            pinned_paths = {v[1] for v in clean_pin.values()}
            deleted_paths: set[str] = set()

            def _unlink_pair(fpath: str) -> bool:
                """Delete a .bin + its .json sidecar, best-effort. Returns True iff
                the .bin was removed. Tolerates a concurrent unlink (the file may
                vanish between scandir and here) and a permission error — neither
                aborts the GC pass.
                
                MOD (PL v3): invalidate scan cache AFTER every successful unlink,
                so all 3 GC callers (age-based, LRU-count, bytes-ceiling) invalidate
                automatically. Single-source, covers all current + future callers.
                """
                removed = False
                try:
                    os.remove(fpath)
                    removed = True
                except FileNotFoundError:
                    pass
                except OSError:
                    log.debug("slot KV GC: cannot remove %s (best-effort)", fpath, exc_info=True)
                # a later phase: the engine-state sidecar pairs with the bin
                # (KV contract: every KV move moves the .ckpt with it) — evicting
                # the bin without it leaked a multi-GB orphan ckpt per eviction.
                # a review guard: remove the ckpt ONLY while the bin is gone — if a
                # concurrent re-save landed a fresh bin between the two removes,
                # its brand-new ladder must survive (the age-floored orphan sweep
                # handles any genuine leftover later).
                if removed or not os.path.exists(fpath):
                    ckpt_path = fpath + ".ckpt"
                    try:
                        os.remove(ckpt_path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        log.debug("slot KV GC: cannot remove sidecar %s (best-effort)", ckpt_path, exc_info=True)
                json_path = fpath[:-4] + ".json"
                try:
                    os.remove(json_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    log.debug("slot KV GC: cannot remove sidecar %s (best-effort)", json_path, exc_info=True)
                if removed:
                    deleted_paths.add(fpath)
                    # MOD: invalidate cache after every GC unlink
                    self._kvcache_scan_cache_invalidate()
                return removed

            def _protected(fpath: str) -> bool:
                # per-group clean anchor OR a bin owned by a live thread
                return fpath in pinned_paths or hash_by_path.get(fpath) in protected_hashes

            # candidates = everything EXCEPT the pinned clean winners AND live-thread bins
            candidates = [(fn, fpath, mtime, size) for (fn, fpath, mtime, size) in entries
                          if not _protected(fpath)]
            deleted = 0
            # a later phase: orphan-ckpt sweep — a .bin.ckpt whose .bin is gone
            # can never be restored (the pair is atomic per the KV contract) = pure
            # dead bytes. Pre-Wave-K evictions and ENOSPC/crash windows strand them.
            try:
                for _cfn in os.listdir(SLOT_SAVE_DIR):
                    if not _cfn.endswith(".bin.ckpt"):
                        continue
                    _cfp = os.path.join(SLOT_SAVE_DIR, _cfn)
                    # FP design note (a later phase gate): existence-check the bin at SWEEP time,
                    # NOT via the entries snapshot — a pair finalizing between the
                    # snapshot and this sweep must not lose its ladder. Plus a
                    # 10-min age floor so an engine-side finalization in flight is
                    # never raced (the ckpt is written by a different process).
                    if os.path.exists(_cfp[:-len(".ckpt")]):
                        continue
                    try:
                        if (now - os.stat(_cfp).st_mtime) < 600:
                            continue
                    except OSError:
                        continue
                    try:
                        os.remove(_cfp)
                        self._kvcache_scan_cache_invalidate()
                        log.info("slot KV GC: reaped orphan ckpt %s", _cfn)
                    except OSError:
                        log.debug("slot KV GC: cannot remove orphan ckpt %s (best-effort)", _cfn, exc_info=True)
                # a review (a later phase gate): reap stranded save temporaries (manager died
                # between the engine save and finalize) — .tmp/.tmp.ckpt bytes are
                # invisible to the ceiling accounting and no other path removes them.
                for _tfn in os.listdir(SLOT_SAVE_DIR):
                    if not (_tfn.endswith(".tmp") or _tfn.endswith(".tmp.ckpt")):
                        continue
                    _tfp = os.path.join(SLOT_SAVE_DIR, _tfn)
                    try:
                        if (now - os.stat(_tfp).st_mtime) > 3600:
                            os.remove(_tfp)
                            log.info("slot KV GC: reaped stranded temp %s", _tfn)
                    except OSError:
                        pass
            except Exception:
                log.debug("slot KV GC: orphan-ckpt sweep failed (best-effort)", exc_info=True)
            # --- Age-based eviction (existing knob) ---
            age_cutoff = now - (max_age_hours * 3600)
            for fn, fpath, mtime, size in candidates:
                if mtime < age_cutoff and _unlink_pair(fpath):
                    deleted += 1
            # --- LRU count eviction (existing knob) if still over limit ---
            remaining = [(fn, fp, mt, sz) for (fn, fp, mt, sz) in candidates
                         if fp not in deleted_paths]
            if len(remaining) > max_files:
                remaining.sort(key=lambda x: x[2])  # oldest first
                to_remove = len(remaining) - max_files
                for fn, fpath, mtime, size in remaining[:to_remove]:
                    if _unlink_pair(fpath):
                        deleted += 1
            # --- Global TOTAL-BYTES ceiling (WIN 1) ---
            max_bytes = _kvcache_ceiling_bytes()
            if max_bytes > 0:
                # Total counts ALL surviving bins (pinned/protected included — they
                # occupy disk too); we only EVICT unprotected bins, oldest-first,
                # and stop the instant we're back at/under the cap (no over-evict).
                surviving = [(fn, fp, mt, sz) for (fn, fp, mt, sz) in entries
                             if fp not in deleted_paths]
                total = sum(sz for _, _, _, sz in surviving)
                if total > max_bytes:
                    # (b1): explicit belt-guard — the freshest `.shadow.`
                    # bin per LIVE thread_hash must never be over-cap evicted (it is
                    # what makes the next swap-back byte-match), even though
                    # `_protected`'s live-thread guard already covers every bin
                    # (incl. all shadows) owned by a live thread.
                    live_shadow_paths: dict[str, tuple[float, str]] = {}
                    for _fn, _fp, _mt, _sz in surviving:
                        if ".shadow." not in _fn:
                            continue
                        _th = hash_by_path.get(_fp)
                        if _th not in protected_hashes:
                            continue
                        _cur = live_shadow_paths.get(_th)
                        if _cur is None or _mt > _cur[0]:
                            live_shadow_paths[_th] = (_mt, _fp)
                    protected_shadow_paths = {v[1] for v in live_shadow_paths.values()}
                    evictable = [(fn, fp, mt, sz) for (fn, fp, mt, sz) in surviving
                                 if not _protected(fp) and fp not in protected_shadow_paths]
                    # (b2): reap DEAD clean_prefix anchors before any
                    # `.shadow.` bin; oldest-first within each group.
                    evictable.sort(key=lambda x: (".shadow." in x[0], x[2]))
                    for fn, fpath, mtime, size in evictable:
                        if total <= max_bytes:
                            break
                        if _unlink_pair(fpath):
                            deleted += 1
                            total -= size
                            # shadow-diag: candidate (b) saved-but-EVICTED —
                            # a `.shadow` bin (born via `SHADOW_SAVE write=ok`) reaped
                            # HERE under over-cap while pinned dead-session clean anchors
                            # hold the floor. LOG ONLY (does NOT change what is deleted —
                            # the delete already happened above). Grep-join SHADOW_SAVE ->
                            # SHADOW_EVICT on th to see born-then-reaped.
                            if ".shadow." in fn:
                                self._shadow_diag_counts["evictions"]["shadow_evicted"] = (
                                    self._shadow_diag_counts["evictions"].get(
                                        "shadow_evicted", 0) + 1)
                                log.info(
                                    "SHADOW_EVICT th=%s model=%s path=%s age_s=%d "
                                    "size=%d reason=unprotected-over-cap",
                                    hash_by_path.get(fpath, "unknown"),
                                    model_by_path.get(fpath, "unknown"), fn,
                                    int(now - mtime), size)
                    if total > max_bytes:
                        log.warning(
                            "slot KV GC ceiling: still %d byte(s) over cap after evicting "
                            "every unprotected bin (pinned/live-thread bins hold the floor)",
                            total - max_bytes,
                        )
            # --- GAP-2 (tier rule): SSD persist-dir sweep ----------
            # SLOT_PERSIST_DIR is the Option-C unload / ownership-transfer SSD
            # archive. Best-effort second sweep mirroring the SLOT_SAVE_DIR pass:
            # (1) age-prune triplets older than the SAME max_age_hours; (2) enforce
            # a bytes ceiling (env TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES, default
            # 42949672960 = 40GiB per the operator) oldest-first by mtime. Triplets
            # (.bin + .bin.ckpt + .json) are deleted together (mirror of
            # _unlink_pair, + the ckpt the persist copy carries). No protected-pins
            # for the persist tier — the tmpfs copy is the live source — EXCEPT a
            # triplet whose .bin mtime is < 60s old is never touched (mid-copy
            # safety: a RAM->SSD persist may be mid-flight). Counted separately
            # from `deleted` so the SLOT_SAVE_DIR diag snapshot stays honest.
            try:
                from turbohaul.subprocess_mgr import SLOT_PERSIST_DIR
                if os.path.isdir(SLOT_PERSIST_DIR):
                    # a later phase: use runtime config (FE-adjustable via PUT /api/config)
                    p_max_bytes = int(self.runtime.persist.max_bytes)
                    p_entries = []  # (fpath, mtime, size) — .bin only (triplet head)
                    for fn in os.listdir(SLOT_PERSIST_DIR):
                        if not fn.endswith(".bin"):
                            continue
                        fpath = os.path.join(SLOT_PERSIST_DIR, fn)
                        try:
                            st = os.stat(fpath)
                        except OSError:
                            continue
                        p_entries.append((fpath, st.st_mtime, st.st_size))

                    def _unlink_persist_triplet(fpath: str) -> bool:
                        """Delete a persist .bin + .bin.ckpt + .json together,
                        best-effort (mirror of _unlink_pair's tolerance)."""
                        removed = False
                        try:
                            os.remove(fpath)
                            removed = True
                        except FileNotFoundError:
                            pass
                        except OSError:
                            log.debug("persist KV GC: cannot remove %s (best-effort)",
                                      fpath, exc_info=True)
                        for side in (fpath + ".ckpt", fpath[:-4] + ".json"):
                            try:
                                os.remove(side)
                            except FileNotFoundError:
                                pass
                            except OSError:
                                log.debug(
                                    "persist KV GC: cannot remove sidecar %s "
                                    "(best-effort)", side, exc_info=True)
                        return removed

                    p_deleted = 0
                    fresh_floor = now - 60.0  # mid-copy safety window
                    p_age_cutoff = now - (max_age_hours * 3600)
                    for fpath, mtime, size in p_entries:
                        if mtime < p_age_cutoff and mtime < fresh_floor:
                            if _unlink_persist_triplet(fpath):
                                p_deleted += 1
                    p_surviving = [(fp, mt, sz) for (fp, mt, sz) in p_entries
                                   if os.path.exists(fp)]
                    p_total = sum(sz for _, _, sz in p_surviving)
                    if p_max_bytes > 0 and p_total > p_max_bytes:
                        p_surviving.sort(key=lambda x: x[1])  # oldest first
                        for fpath, mtime, size in p_surviving:
                            if p_total <= p_max_bytes:
                                break
                            if mtime >= fresh_floor:
                                continue  # mid-copy safety: never delete <60s-old
                            if _unlink_persist_triplet(fpath):
                                p_deleted += 1
                                p_total -= size
                    log.info(
                        "persist KV GC: total_bytes=%d ceiling=%d deleted=%d",
                        p_total, p_max_bytes, p_deleted)
            except Exception:
                log.debug("persist KV GC best-effort failed", exc_info=True)
            # --- shadow-diag: KVGC observability (LEAD #1). One structured
            # line + snapshot per GC pass naming WHAT pins the floor: the total on-disk
            # bytes, over_cap, and the PROTECTED (pinned clean-anchor OR live-thread)
            # bins + bytes, with the top protected thread_hashes. If protected_bytes
            # alone exceeds the cap, GC CANNOT evict under it = the 64GB floor. Sizes
            # come from the already-stat'd `entries` (NO new stat). Reused by the
            # KV_RESTORE line to correlate a forcing-full restore to pressure. Off the
            # hot path (throttled sweeper). Best-effort — never faults the GC pass.
            try:
                _surv = [(fn, fp, mt, sz) for (fn, fp, mt, sz) in entries
                         if fp not in deleted_paths]
                _total_bytes = sum(sz for _, _, _, sz in _surv)
                _over_cap = max(0, _total_bytes - max_bytes) if max_bytes > 0 else 0
                _prot = [(fn, fp, mt, sz) for (fn, fp, mt, sz) in _surv if _protected(fp)]
                _protected_bytes = sum(sz for _, _, _, sz in _prot)
                # A protected bin is LIVE iff its thread_hash is in the live set; a
                # protected bin that is NOT live = a pinned clean-anchor of a DEAD
                # session (the leak — these hold the floor but should be prunable).
                _dead = [(fn, fp, mt, sz) for (fn, fp, mt, sz) in _prot
                         if hash_by_path.get(fp) not in protected_hashes]
                _dead_bytes = sum(sz for _, _, _, sz in _dead)
                # Per-thread_hash aggregate: bytes + OLDEST mtime (largest age = most
                # stale) + liveness. `now` from the top of the GC pass.
                _agg: dict[str, list] = {}  # th -> [bytes, oldest_mtime, live]
                for _fn, _fp, _mt, _sz in _prot:
                    _th = hash_by_path.get(_fp)
                    _tk = _th or ("pinned-anchor" if _fp in pinned_paths else "unknown")
                    _live = _th in protected_hashes
                    rec = _agg.get(_tk)
                    if rec is None:
                        _agg[_tk] = [_sz, _mt, _live]
                    else:
                        rec[0] += _sz
                        rec[1] = min(rec[1], _mt)
                        rec[2] = rec[2] or _live
                _top = sorted(_agg.items(), key=lambda kv: kv[1][0], reverse=True)[:8]
                _top_fmt = ",".join(
                    f"{k}:{v[0]}:{int(now - v[1])}:{'live' if v[2] else 'dead'}"
                    for k, v in _top)
                _floor_reason = ("live-protected-floor"
                                 if _over_cap > 0 and _protected_bytes > max_bytes
                                 else ("over-cap-evicting" if _over_cap > 0 else "under-cap"))
                self._last_kvgc_snapshot = {
                    "total_bytes": _total_bytes, "ceiling": max_bytes,
                    "over_cap": _over_cap, "deleted": deleted,
                    "protected_bin_count": len(_prot),
                    "protected_bytes": _protected_bytes,
                    "dead_protected_bins": len(_dead),
                    "dead_protected_bytes": _dead_bytes,
                    # th -> {bytes, age_s, live} (top-8 by bytes) so PL/the operator SEE which
                    # dead-session anchors pin the floor.
                    "protected_threads": {
                        k: {"bytes": v[0], "age_s": int(now - v[1]),
                            "live": bool(v[2])} for k, v in _top},
                    "floor_reason": _floor_reason,
                    "ts": time.time(),
                }
                log.info(
                    "KVGC total_bytes=%d ceiling=%d over_cap=%d deleted=%d "
                    "protected_bins=%d protected_bytes=%d dead_protected_bins=%d "
                    "dead_protected_bytes=%d protected_threads=[%s] floor_reason=%s",
                    _total_bytes, max_bytes, _over_cap, deleted, len(_prot),
                    _protected_bytes, len(_dead), _dead_bytes, _top_fmt, _floor_reason,
                )
            except Exception:
                log.debug("KVGC diag snapshot best-effort failed", exc_info=True)
            if deleted:
                log.info("slot KV GC: deleted %d stale file(s)", deleted)
            return deleted
        except Exception:
            log.debug("slot KV GC failed", exc_info=True)
            return 0

    # === WIN 4 build/model/ctx fingerprint + mismatched-bin purge ==
    # File-hygiene / defense-in-depth ONLY. Stamps every sidecar at save + deletes
    # bins that can't belong to the current engine. Reads NO restore decision and
    # changes NONE (resolve_kv / kv_policy / the M4 restore gate are off-limits).

    def _engine_fingerprint(self, model_tag: str) -> dict:
        """Best-effort build/model/ctx fingerprint of the CURRENTLY-loaded engine
        for ``model_tag``. Stamped into every KV sidecar at save (:_save_slot_kv)
        and compared by the purge sweep so a bin can NEVER be silently restored
        into a different engine build / model / ctx-config (= garbage KV).

        Every source is ALREADY known to the manager — there is NO multi-GB gguf
        hashing on the save path:
          - gguf_sha256     = manifest.gguf_blob_sha256 (PRECOMPUTED 64-hex blob
                              digest = the model identity).
          - engine_build_id = boot.runtime.llama_server_binary_sha256 (the pinned
                              llama-server binary digest = the ENGINE BUILD id;
                              empty in dev mode -> None).
          - n_ctx           = llama_server_flags.ctx_size or manifest.context_size
                              (the SAME chokepoint as :1512 / :2749).
          - n_rs_seq        = llama_server_flags.parallel (= n_seq_max, the width
                              that dimensions llama.cpp's recurrent/hybrid-SSM state
                              cache). NOTE: a model's INTRINSIC per-architecture
                              recurrent count (e.g. the qwen MTP hybrid's 2) is
                              fully determined by the gguf, so it already rides
                              along in gguf_sha256; this field captures the runtime
                              seq WIDTH. TODO: prefer a true engine-reported
                              n_rs_seq if the sidecar /props ever exposes it.

        A field is None when its source is genuinely unavailable (dev mode / no
        manifest). Never raises."""
        fp: dict[str, Any] = {
            "gguf_sha256": None,
            "engine_build_id": None,
            "n_ctx": None,
            "n_rs_seq": None,
        }
        # engine_build_id: global to the manager (one binary serves every model).
        try:
            bid = getattr(self.boot.runtime, "llama_server_binary_sha256", "") or ""
            fp["engine_build_id"] = bid or None
        except Exception:
            pass
        # model + ctx + seq-width from the loaded model's manifest (small YAML read).
        try:
            m = read_manifest(self.boot.storage.manifests_path, model_tag)
            flags = m.llama_server_flags or {}
            fp["gguf_sha256"] = (m.gguf_blob_sha256 or None)
            _ctx = int(flags.get("ctx_size") or m.context_size or 0)
            fp["n_ctx"] = _ctx or None
            fp["n_rs_seq"] = int(flags.get("parallel", 1) or 1)
        except Exception:
            # Missing manifest (dev/test) or parse error: keep whatever we have.
            pass
        return fp

    @staticmethod
    def _fingerprint_matches(saved_meta: dict, current_fp: dict) -> bool:
        """True IFF the saved sidecar's stamp PROVES same build/model/ctx/seq as the
        current engine. A missing/None field on EITHER side = cannot prove same-
        build = NO MATCH (an unstamped legacy bin can't prove same-build -> unsafe
        to keep). String-normalized compare so None-vs-"" / int-vs-str can't
        false-match into a wrongful keep."""
        for k in ("gguf_sha256", "engine_build_id", "n_ctx", "n_rs_seq"):
            sv = saved_meta.get(k)
            cv = current_fp.get(k)
            if sv is None or cv is None:
                return False
            if str(sv) != str(cv):
                return False
        return True

    @staticmethod
    def _fp_summary(fp: dict) -> dict:
        """Compact fingerprint for logs — truncate the 64-hex digests to 12 chars."""
        out = {}
        for k in ("gguf_sha256", "engine_build_id", "n_ctx", "n_rs_seq"):
            v = fp.get(k)
            out[k] = v[:12] if isinstance(v, str) and len(v) > 12 else v
        return out

    def _purge_protected_basenames(self) -> set:
        """Basenames (no extension) of KV bins the LIVE idle holder is anchored on
        RIGHT NOW — never delete these even on a fingerprint miss. Yanking a live
        warm-path anchor mid-flight silently disarms the the classifier classifier (the exact
        harm the GC's clean-pin guards against,:_gc_kv_cache). A current-build
        anchor is ALSO preserved by the fingerprint match; this is the extra belt
        for a legacy (unstamped) anchor a live idle thread still depends on.
        Best-effort -> empty set on any error. Reads _find_clean_bin ONLY to build
        a PROTECT list (never makes a restore decision)."""
        protected: set = set()
        try:
            h = self._idle_handle
            mt = self._idle_model_tag
            if h is not None and mt:
                found = self._find_clean_bin(h.port, mt, self._idle_thread_id or "")
                if found:
                    bin_fn = found[0]
                    protected.add(
                        bin_fn[:-4] if bin_fn.endswith(".bin") else bin_fn
                    )
        except Exception:
            pass
        return protected

    def _maybe_purge_mismatched_bins(self, reason: str) -> int:
        """Throttled wrapper for the sweeper: run the fingerprint purge at most once
        per FINGERPRINT_PURGE_MIN_INTERVAL_S. The flag is checked FIRST so the
        throttle clock only advances when purge is enabled (a fresh enable purges on
        the next sweep, not one interval later)."""
        if not _fingerprint_purge_enabled():
            return 0
        now = time.monotonic()
        if (now - self._last_fp_purge_mono) < FINGERPRINT_PURGE_MIN_INTERVAL_S:
            return 0
        self._last_fp_purge_mono = now
        return self._purge_mismatched_bins(reason=reason)

    def _purge_mismatched_bins(self, *, reason: str) -> int:
        """FILE-level sweep: delete every KV .bin+.json sidecar whose stamped
        fingerprint != its model's CURRENT engine fingerprint (different build /
        model / ctx / seq-width = garbage KV). Unstamped legacy bins (missing the
        fields) = UNKNOWN build = purged (a missing stamp can't PROVE same-build ->
        unsafe to keep). Gated by TURBOHAUL_FINGERPRINT_PURGE; best-effort (never
        raises); returns the count of bins removed.

        PURELY a file-deletion sweep. It reads NO restore decision and changes
        NONE (resolve_kv / kv_policy / the M4 restore gate are separately gated +
        OFF-LIMITS). Each sidecar is evaluated against ITS OWN model's current
        manifest (grouped by the stamped model_tag) — so a swap/reload that bumped
        a model's ctx or re-quantized its gguf reclaims THAT model's stale bins
        without touching a co-resident other model's still-valid bins."""
        if not _fingerprint_purge_enabled():
            return 0
        from turbohaul.subprocess_mgr import SLOT_SAVE_DIR
        removed = 0
        try:
            entries = os.listdir(SLOT_SAVE_DIR)
        except OSError:
            return 0
        protected = self._purge_protected_basenames()
        fp_cache: dict[str, dict] = {}
        for fn in entries:
            if not fn.endswith(".json"):
                continue
            base = fn[:-5]
            if base in protected:
                continue
            meta_path = os.path.join(SLOT_SAVE_DIR, fn)
            try:
                with open(meta_path) as f:
                    m = json.load(f)
            except (OSError, ValueError):
                # Unreadable/half-written sidecar -> leave it. The atomic .tmp +
                # os.replace save means a real sidecar is fully present or absent;
                # a transient read error must never trigger a delete.
                continue
            mt = m.get("model_tag")
            if not isinstance(mt, str) or not mt:
                # Can't identify the owning model -> can't compare -> leave it.
                continue
            cur = fp_cache.get(mt)
            if cur is None:
                cur = self._engine_fingerprint(mt)
                fp_cache[mt] = cur
            # Can't establish this model's CURRENT identity -> can't prove mismatch
            # -> skip (never purge blind on an unknown current engine). Guard on BOTH
            # gguf_sha256 (manifest gone / dev mode) AND engine_build_id (PL byte-
            # review MOD a): an UNPINNED/dev binary has llama_server_binary_sha256=""
            # -> engine_build_id=None -> _fingerprint_matches returns False for EVERY
            # bin -> would purge ALL bins incl fresh valid ones = reuse-defeating
            # footgun that would nuke the smoke. Skipping when either identity
            # dimension is unknown is the safe direction.
            if not cur.get("gguf_sha256") or not cur.get("engine_build_id"):
                continue
            if self._fingerprint_matches(m, cur):
                continue
            # Mismatch OR unstamped -> purge the .bin + .json pair.
            bin_path = os.path.join(SLOT_SAVE_DIR, base + ".bin")
            ok = True
            for p in (bin_path, meta_path):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    ok = False
                    log.warning("fingerprint purge (%s): failed to remove %s: %s",
                                reason, p, e)
            if ok:
                removed += 1
                # design note: invalidate after fingerprint purge
                self._kvcache_scan_cache_invalidate()
                log.info(
                    "fingerprint purge (%s): removed mismatched KV %s "
                    "(saved=%s vs current=%s)",
                    reason, base, self._fp_summary(m), self._fp_summary(cur),
                )
        if removed:
            log.info("fingerprint purge (%s): removed %d mismatched KV bin(s)",
                     reason, removed)
        return removed

    async def _force_cold(self, slot: Slot, reason: str) -> None:
        """Mark a slot COLD when processing dies mid-flight.

        idle-holder fix: walk legal transitions to COLD from any non-terminal
        state instead of silent direct-mutation. fsm.py LEGAL_TRANSITIONS now
        carries STAGED→COLD, LOADING→COLD, LOADING_FAIL→POPPED, POPPED→COLD,
        ACTIVE→GRACE→POPPED→COLD, IDLE_HOT→COLD. Memory and DB state stay
        in sync (no drift where slot.state stays e.g. LOADING in Python
        while sqlite reads state='COLD').

        fix: defensive teardown of any live handle attached
        to this slot BEFORE forcing COLD. Closes the footgun where a
        caller forgot to teardown (the state-drift guard active_match_cancelled path
        raises after _force_cold(matched, ...) without sigterm'ing the
        anchor sidecar). Defense-in-depth — design review #1-priority fix.

        design review a high-priority item fix: annotate audit with pid_source so post-hoc
        tooling can distinguish matched-row-on-anchor-pid (anchor_shared)
        from genuine standalone (self).
        """
        # M5 (WIN 2) SWAP-CLEAR (cap<=1): processing died mid-flight, so
        # the engine/KV state that would produce a consistent cached answer is
        # disrupted -> invalidate the completion-cache (correctness > hit-rate).
        self._completion_cache_clear("force_cold")
        # design review critical item + a high-priority item: identify pid_source, defensive sigterm
        # if the slot owns its own live handle.
        pid_source = "self"
        if (
            self._active_handle is not None
            and slot.pid
            and slot.pid == self._active_handle.pid
            and len(self._inflight) > 1
        ):
            # Design #1 must-fix: the sidecar is serving MULTIPLE concurrent
            # fan-out riders. Force-colding ONE of them (the ANCHOR or a rider)
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
            # (the state-drift guard drift path). Do NOT teardown — anchor owns it.
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
                self._set_active_handle(None)
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
        # the design: slot-write stays on state_db_session; audit-write via pool.
        with state_db_session(self.boot.storage.state_db_path) as conn:
            mark_slot_ended(conn, slot.slot_id, reason)

        def _audit_force_cold() -> None:
            with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                # design review a high-priority item: pid_source annotation
                record_audit_event(
                    audit_conn,
                    "force_cold",
                    {"reason": reason, "pid_source": pid_source},
                    slot_id=slot.slot_id,
                )

        await asyncio.to_thread(_audit_force_cold)

    def _compute_expected_drop_mib(self, model_tag: str) -> int:
        """fix: dynamic expected_drop_mib derived from manifest.

        Returns max(2048, manifest.expected_vram_bytes/1024**2). Floor
        at 2 GiB absorbs page-cache noise on small models. If manifest
        cannot be read (race during teardown of a just-deleted manifest),
        falls back to 2048 MiB rather than blocking teardown.
        """
        try:
            m = read_manifest(self.boot.storage.manifests_path, model_tag)
            return max(2048, int(m.expected_vram_bytes / (1024 * 1024)))
        except (FileNotFoundError, OSError):
            log.warning(
                "_compute_expected_drop_mib: manifest read failed for %s, "
                "falling back to 2048 MiB",
                model_tag,
            )
            return 2048

    def _live_handle_pids(self) -> set[int]:
        """Set of currently-known live llama-server pids across ALL residents.

        Used by intra_lifetime_orphan_scan + boot_orphan_reaper to tell managed
        sidecars from leaked orphans on our port range. P1d UNION over
        EVERY resident's live handle pid + idle_handle pid + each reserving resident's
        ``booting_pid`` (the spawned-but-handle-not-yet-set pid), read from an
        AWAIT-FREE ``list`` snapshot of the registry so a concurrent dispatcher
        mutation can't 'dict changed size during iteration'. At cap==1 the lone
        resident mirrors ``_active_handle``/``_idle_handle`` so this equals the legacy
        {_active,_idle} set (byte-identical); at cap>=2 the union-ALL is the critical
        fix that stops a co-resident sidecar being reaped as an orphan during another
        resident's teardown, and ``booting_pid`` closes the still-booting-sibling
        window before its handle is set.
        """
        live: set[int] = set()
        # cap-gate the multi-resident union so at cap<=1 this function is LITERALLY
        # the legacy two-if body below (structural byte-identity, not argued — the
        # entire dispatcher/registry surface is unreachable at cap=1, and the
        # _residents singleton's booting_pid is never written on that path).
        if self.runtime.queue.max_parallel_sidecars >= 2:
            for r in list(self._residents.values()):  # await-free atomic snapshot
                h = r.handle
                if h is not None and h.is_alive():
                    live.add(h.pid)
                ih = r.idle_handle
                if ih is not None and ih.is_alive():
                    live.add(ih.pid)
                if r.booting_pid is not None:
                    live.add(r.booting_pid)
        # Legacy manager-global holders (primary mirror) — belt-and-suspenders so
        # cap==1 stays byte-identical even if a singleton mirror ever lagged.
        # a tracked issue: read handle into local var first to avoid TOCTOU between
        # is_alive check and .pid read (handle could change between reads).
        ah = self._active_handle
        if ah is not None and ah.is_alive():
            live.add(ah.pid)
        ih = self._idle_handle
        if ih is not None and ih.is_alive():
            live.add(ih.pid)
        return live

    async def _audit_async(self, slot: Slot, event_type: str) -> None:
        """Async-context wrapper for `_audit`.

        the design design note: audit_db_session is sync-only (a review guard). When called from an
        async function (e.g. `_process_slot`), the sync `_audit` must be
        offloaded to a worker thread or the a review guard guard raises RuntimeError.
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

        the design: slot-write stays on state_db_session; audit-write via pool. Both
        calls are sync (this method is `def`, not `async def`), so the
        audit_db_session a review guard sync-only guard is satisfied without to_thread.

        ★ design note: do NOT call this directly from async code — use
        `_audit_async` instead (the a review guard guard catches direct calls from a
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
        # Publish redacted event to WS subscribers
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
        to clobber that state. the design: routed through audit pool (sync call).
        """
        with audit_db_session(self.boot.storage.state_db_path) as conn:
            record_audit_event(conn, event_type, payload or {}, slot_id=slot_id)

    # === background sweeper ==============================

    async def _periodic_terminal_park_sweep(self) -> None:
        """Periodically finalize the state-row for the design evictions.

        the design a design note deferred state-row finalization OFF the
        worker_loop hot path to avoid the 1-3s SQLite fsync stall that
        bypassed the the design audit pool. Audit-emit fires immediately via
        ``_audit_event_only_async``; the state-row finalize lands here.

        Loop: every ``background_sweep_interval_s`` (default 60s — matches
        the design audit pool rhythm), the sweeper queries for slots that are:
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
        overwhelmingly likely to be an the design eviction casualty.
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
            # ceiling-GC: piggyback the KV-cache FILE gc on the park-sweep
            # cadence, but THROTTLED to _KV_GC_MIN_INTERVAL_S (a scandir of the multi-
            # GB save dir must not run every park tick). Best-effort — a GC failure
            # NEVER breaks the park loop (own try/except; _gc_kv_cache self-guards too).
            # NOTE: wired into the LOOP, not _run_one_sweep, so unit tests that call
            # _run_one_sweep directly never touch the real on-disk kvcache dir.
            try:
                _mono = time.monotonic()
                if _mono - self._last_kv_gc_monotonic >= self._KV_GC_MIN_INTERVAL_S:
                    self._last_kv_gc_monotonic = _mono
                    await self._gc_kv_cache()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("periodic KV-cache ceiling-GC failed (best-effort)")
            # WIN 4 reclaim stale KV bins whose stamped build/model/
            # ctx fingerprint no longer matches the current engine (a model/binary
            # rebuild or a manifest ctx-change leaves a swapped/reloaded model's old
            # bins mismatched). Runs on the sweeper task OFF the hot path — gated +
            # throttled + best-effort; offloaded to a thread so the file I/O never
            # blocks the loop. STRICTLY a file sweep — never touches the restore
            # decision. (The M4 restore gate is the synchronous correctness backstop
            # for the sub-interval window; this is disk hygiene / defense-in-depth.)
            try:
                await asyncio.to_thread(
                    self._maybe_purge_mismatched_bins, "sweeper"
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "fingerprint purge (sweeper) failed (best-effort)"
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
                   LIMIT 100""",  # PL a design note: batch cap, bounds writer-lock hold time under storm pattern
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

    async def _drain_bg_tasks(self, grace_s: float = 10.0) -> None:
        """Await the detached ``_spawn_bg`` teardown/requeue/reap tasks at shutdown so
        an in-flight ``_evict_teardown`` (sidecar reap) can't leak past process exit
        (P1d N2 fast-follow #2). Snapshot first (the set self-mutates as tasks finish);
        bounded by ``grace_s``; ``gather(return_exceptions=True)`` so one failing task
        can't abort the drain. A timeout cancels the stragglers (acceptable — they are
        best-effort cleanup and the process is exiting).

        NOTE on timing: the SLOW reap (``_reap_booting_pid``'s up-to-3s SIGTERM-poll +
        SIGKILL) runs INLINE in the driver finally, which the sweep already awaits
        with NO timeout in the DRIVERS step BEFORE this drain — and that finally claims
        ``torn_down`` first, so any bg ``_evict_teardown`` here finds it already claimed
        and no-ops. So this drain only awaits the FAST bg tasks (future-fails, requeues,
        no-op teardowns); the generous 10s grace is pure insurance."""
        pending = [t for t in list(self._bg_tasks) if not t.done()]
        if not pending:
            return
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=grace_s
            )

    async def shutdown(self) -> None:
        """Clean tear-down. Stops worker loop + sweeper + drains queue + closes state db."""
        self._stop_event.set()
        # === P1e shutdown-sweep: OBSERVERS -> dispatcher -> DRIVERS -> bg-drain ===
        # 1) OBSERVERS FIRST (both the cap<=1 single poller AND the cap>=2 resident
        # supervisor) so neither issues a /slots probe against a sidecar being torn
        # down, nor publishes a generation tick after the loop is gone (a review Failure
        # Predictor #16 HIGH). Cancelling the supervisor stops ALL N per-resident
        # polls at once.
        for obs in (self._live_poller_task, self._live_supervisor_task):
            if obs is not None and not obs.done():
                obs.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await obs
        # 2) DISPATCHER next: stop routing new work onto residents.
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        # 3) DRIVERS (P1e): deterministically cancel every per-resident driver so its
        # finally reaps the handle / booting_pid + drains its inbox (the cap>=2
        # teardown). P1d left drivers to NOTICE _stop_event on their next loop turn;
        # the sweep cancels them so a driver parked in inbox.get/health-wait tears
        # down NOW (and before queue.close, so drained inbox slots can re-enqueue).
        driver_tasks = [
            r.driver_task
            for r in list(self._residents.values())
            if r.driver_task is not None and not r.driver_task.done()
        ]
        for dt in driver_tasks:
            dt.cancel()
        if driver_tasks:
            # Await the driver teardowns in PARALLEL (gather), not sequentially: each
            # finally may run an up-to-3s SIGTERM-grace reap, so a sequential
            # `for dt: await dt` would serialize N residents to ~N*3s at shutdown.
            # return_exceptions so one failing teardown can't abort the rest (the
            # CancelledError each cancelled driver raises is captured, not propagated)
            # (PL pre-cutover polish #3).
            await asyncio.gather(*driver_tasks, return_exceptions=True)
        # 4) DRAIN the detached bg-tasks (P1d N2 fast-follow #2): the driver finallys +
        # death-reapers schedule teardown/requeue/reap coroutines through _spawn_bg;
        # await them (bounded) so an in-flight _evict_teardown can't LEAK a sidecar
        # past process exit. Must come AFTER the driver awaits (which is when those
        # reapers get scheduled).
        await self._drain_bg_tasks()
        # cancel + await the background sweeper task.
        if self._sweeper_task is not None and not self._sweeper_task.done():
            self._sweeper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._sweeper_task
        # NEMO V2 2.1 fix: close now returns the slots it cleared; fail
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
        # idle-holder wiring: tear down any idle holder so VRAM is released
        # and llama-server child is reaped on graceful shutdown.
        if self._idle_handle is not None:
            try:
                await self._teardown_idle_holder("shutdown")
            except Exception:
                log.exception(
                    "idle teardown during shutdown failed (best-effort)"
                )
        # lifecycle hardening: release the TOCTOU-pinned binary fd on shutdown
        if self._binary_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._binary_fd)
            self._binary_fd = None
