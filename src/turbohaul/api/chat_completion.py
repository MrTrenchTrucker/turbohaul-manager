"""Chat-completion API routes - Ollama-compat + OpenAI-compat.

This module ships the non-streaming completion path. Streaming SSE comes in a
future polish pass; the existing manager.submit_and_wait + completion_fn DI is
streaming-ready (just return an async generator from completion_fn and adapt
the route).

The completion_fn is wired into TurbohaulManager via DI. Production uses
make_llama_server_complete_fn() which httpx-POSTs to the spawned llama-server's
/v1/chat/completions on its assigned port. Tests inject a fake completion_fn
that returns a canned response without spawning anything real.

Typed upstream errors: in practice, the live sidecar exhibits
RemoteProtocolError (sidecar OOM-crash during inference) much more often than
HTTPStatusError 4xx. These need different client-facing status codes:
  - 503 Service Unavailable + Retry-After  → sidecar disconnected / crashed
  - 502 Bad Gateway                         → sidecar returned upstream 4xx/5xx
  - 504 Gateway Timeout                     → request timed out at sidecar
  - 500 Internal Server Error               → genuine Turbohaul bug (fallback)
  - 422 RESERVED for input-validation only (NOT used for upstream errors)
"""
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from turbohaul.config import KEEP_ALIVE_MAX_S  # re-exported for tests + manager
from turbohaul.kv_policy import (
    compute_ctx_len,       # shared admission/save size rule
    _prefix_hash_chain,    # shared admission/save turn-hash chain (unified)
)
from turbohaul.live_monitor import (  # live-monitor text tee identity
    compute_generation_id,
    _read_spawn_seq,
)
from turbohaul.manifest import (  # handler-entry manifest read for thinking detect
    ManifestValidationError,
    read_manifest,
)
from turbohaul.slot import SlotEvictedError, derive_thread_id_prefix_hash  # first-message fingerprint
from turbohaul.api.tool_call_recovery import maybe_recover_tool_calls


# === Client-disconnect watcher =============================================
# Constant 2s cadence + direct request.is_disconnected() call. Wrapping
# is_disconnected in asyncio.wait_for can leak the underlying ASGI receive()
# coroutine on cancellation; Starlette already implements is_disconnected as
# non-blocking-fast, so the wrapper is both unnecessary and harmful.
_DISCONNECT_POLL_INTERVAL_S = 2.0


async def watch_disconnect(
    request: Request,
    disconnect_event: asyncio.Event,
) -> None:
    """Poll ``request.is_disconnected()`` every ~2s; signal ``disconnect_event``
    when the client closes the connection.

    Constant cadence, direct poll, no ``asyncio.wait_for`` wrap. Tolerates
    transient ASGI receive() errors (returns True is the only state we care
    about).
    """
    while not disconnect_event.is_set():
        try:
            if await request.is_disconnected():
                disconnect_event.set()
                return
        except Exception:
            # Transient ASGI receive() errors are non-fatal; keep polling.
            # On py3.11 CancelledError is BaseException, so this already never
            # catches KI/SE/Cancel — narrowing would be a regression.
            pass
        await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)


log = logging.getLogger(__name__)
router = APIRouter()


# === JSON Schema validation constants =======================================
# Validator-side DoS guards — caller-supplied schemas are size/shape-bounded
# BEFORE we compile via jsonschema.
_SCHEMA_MAX_BYTES = 65536  # 64 KiB serialized schema cap
_SCHEMA_MAX_DEPTH = 16  # recursive object/array nesting cap
_SCHEMA_MAX_PROPERTIES = 64  # total property count across the schema tree
_BODY_MAX_BYTES = 4194304  # 4 MiB total request body cap (out-of-scope guard hook)
_COMPILE_TIMEOUT_SEC = 0.5  # Draft202012Validator construction budget


def _schema_depth(node: Any, depth: int = 0) -> int:
    """Recursive max-nesting walker for object/array structures in a schema."""
    if depth > _SCHEMA_MAX_DEPTH:
        return depth
    if isinstance(node, dict):
        return max(
            [depth] + [_schema_depth(v, depth + 1) for v in node.values()]
        )
    if isinstance(node, list):
        return max(
            [depth] + [_schema_depth(item, depth + 1) for item in node]
        )
    return depth


def _schema_property_count(node: Any) -> int:
    """Total property-name count across the schema. Walks dict + list."""
    if isinstance(node, dict):
        local = len(node.get("properties", {})) if isinstance(node.get("properties"), dict) else 0
        return local + sum(_schema_property_count(v) for v in node.values())
    if isinstance(node, list):
        return sum(_schema_property_count(item) for item in node)
    return 0


def _schema_has_ref(node: Any) -> bool:
    """Reject ANY $ref for MVP — eliminates cycle + remote-fetch attack surface."""
    if isinstance(node, dict):
        if "$ref" in node:
            return True
        return any(_schema_has_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_schema_has_ref(item) for item in node)
    return False


def _schema_missing_additional_properties_guard(node: Any) -> bool:
    """Object schemas MUST set additionalProperties: false (avoids implicit-anything)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            return True
        return any(_schema_missing_additional_properties_guard(v) for v in node.values())
    if isinstance(node, list):
        return any(_schema_missing_additional_properties_guard(item) for item in node)
    return False


def _validate_json_schema(rf: dict) -> tuple[bool, str | None]:
    """Validate a caller-supplied json_schema response_format.

    Returns ``(ok, reason)``. On not-ok, ``reason`` is a short machine-readable
    string the caller surfaces in the HTTP 422 error body. On ok, the schema is
    safe to forward to llama-server and (later, in `_complete`) to use for
    Draft202012Validator.validate against the model's returned JSON.

    The jsonschema import is LAZY inside the function body — fail-soft
    against the dep being absent during an early boot window (a writable-layer
    pip-install may precede the image bake).

    `rf` is the FULL response_format dict; the schema lives at
    rf["json_schema"]["schema"] per the OpenAI structured-outputs envelope.
    """
    try:
        from jsonschema import Draft202012Validator  # noqa: F401  (compile only)
    except ImportError:
        return (False, "jsonschema_lib_unavailable")

    if not isinstance(rf.get("json_schema"), dict):
        return (False, "missing_or_malformed_json_schema_field")
    schema = rf["json_schema"].get("schema")
    if not isinstance(schema, dict):
        return (False, "missing_or_malformed_schema_field")

    # Size check
    try:
        schema_bytes = len(json.dumps(schema))
    except (TypeError, ValueError):
        return (False, "schema_not_json_serializable")
    if schema_bytes > _SCHEMA_MAX_BYTES:
        return (False, f"schema_size_exceeded:{schema_bytes}")

    # Depth check
    if _schema_depth(schema) > _SCHEMA_MAX_DEPTH:
        return (False, "schema_depth_exceeded")

    # Property count check
    if _schema_property_count(schema) > _SCHEMA_MAX_PROPERTIES:
        return (False, "schema_property_count_exceeded")

    # $ref rejection (no cycle / no remote fetch)
    if _schema_has_ref(schema):
        return (False, "schema_contains_ref_unsupported")

    # additionalProperties guard on object schemas
    if _schema_missing_additional_properties_guard(schema):
        return (False, "schema_missing_additionalProperties_guard")

    # Compile attempt (synchronous; bounded by _COMPILE_TIMEOUT_SEC budget upstream)
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator(schema)
    except Exception as e:  # noqa: BLE001 — compile errors are caller's input fault
        return (False, f"schema_compile_failed:{type(e).__name__}")

    return (True, None)


def is_thinking_payload(payload: dict, manifest: dict) -> bool:
    """Thinking-mode detection.

    `chat_template_kwargs` lives in manifest.py DENIED_FLAGS so a payload-side
    check would always be False (the field never reaches the outgoing request).
    Detection is manifest-only via reasoning_budget > 0. The `payload` arg is
    present for interface parity with the spec; not consulted.

    `manifest` is dict-shaped; callers pass `Manifest.llama_server_flags` (or
    any dict carrying a `reasoning_budget` key).
    """
    rb = manifest.get("reasoning_budget", 0)
    try:
        return int(rb) > 0
    except (TypeError, ValueError):
        return False


def _strip_thinking_wrapper(content: str) -> str:
    """Strip `<think>...</think>` wrapper to surface the post-think payload.

    Uses `rsplit('</think>', 1)` so even malformed multi-tag content (an
    aborted think block followed by a real one) surfaces the LAST post-think
    payload. Returns the original string when no closing tag is present.

    NOTE — rsplit-last is DELIBERATE here: the callers (`json_schema` validate +
    retry) `json.loads` the single trailing payload, so keeping only the tail
    after the FINAL `</think>` is exactly right. Do NOT switch this to remove-all
    — the shadow-save + byte-match probe, which must instead mirror the client
    harness's remove-ALL resend, use :func:`_strip_thinking_all`.
    """
    if not isinstance(content, str):
        return content
    if "</think>" in content:
        return content.rsplit("</think>", 1)[-1].lstrip()
    return content


# Mirror the client harness's think-strip so the shadow-save + byte-match probe
# predict the harness's think-STRIPPED resend byte-for-byte. The harness REMOVES
# every `<think>...</think>` block via `re.sub(r'<think>.*?</think>', '',
# flags=re.DOTALL | re.IGNORECASE)` and its callers then `.strip()` the result.
# `_strip_thinking_wrapper`'s rsplit-last keeps only the tail after the LAST
# `</think>`, which DIFFERS from remove-all on multi-block / pre-`<think>` content
# and silently dropped the shadow byte-match (observed: single-block MATCH,
# multi-block MISMATCH). Non-greedy `.*?` pairs each open with its OWN close (never
# spans across a block); `re.IGNORECASE` matches the harness flag. Scoped to
# `<think>` only: the shadow callers pre-guard on a literal `</think>` and only the
# force-restore family emits that tag — the harness's other passes
# (`<thinking>`/`<reasoning>`/tool-call XML/unterminated-tag) are no-ops on it, so
# this single pass is a faithful subset (any residual under-strip safe-degrades to a
# clean-restore + engine reprefill backstop, never a wrong answer).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking_all(content: str) -> str:
    """Remove ALL `<think>...</think>` blocks then `.strip()` — mirrors the client
    harness resend (`strip_think_blocks(x).strip()`).

    Unlike :func:`_strip_thinking_wrapper` (rsplit-last, keeps only the tail after
    the FINAL `</think>`), this removes EVERY think block and preserves the visible
    text between/before blocks, so multi-block (`<think>a</think>X<think>b</think>Y`
    -> `XY`) and pre-`<think>` prose (`intro<think>t</think>ans` -> `introans`)
    byte-match what the harness resends next turn.

    Best-effort + None-safe (mirrors `_strip_thinking_wrapper`): a non-str passes
    through unchanged; single-block content strips identically to the harness; no
    think block -> the input with surrounding whitespace stripped (the harness always
    `.strip()`s — the shadow callers never reach this branch, they pre-guard on
    `</think>`); empty -> empty.
    """
    if not isinstance(content, str):
        return content
    return _THINK_BLOCK_RE.sub("", content).strip()


def wrap_reasoning_think(reasoning: str, content: str) -> str:
    """SINGLE SOURCE OF TRUTH for the ``<think>``-wrapped reasoning+content
    history form.

    Returns EXACTLY the bytes :func:`_merge_reasoning_into_content` writes to the
    client response — which the client harness receives, strips at its storage
    boundary (``_strip_think_blocks(content).strip()``) and resends think-free
    next turn. The SAVE-side shadow reconstructions (manager
    ``_generated_assistant_msg`` + the streaming accumulator) MUST emit these SAME
    bytes so that ``_strip_thinking_all(shadow_form)`` equals the harness's
    ``strip_think_blocks(merge_form).strip()`` BYTE-FOR-BYTE. The old no-newline
    ``<think>{r}</think>{c}`` shadow form diverged from this newline merge form under
    the remove-ALL strip on multi-block / stray-``</think>`` content (their strips
    differ on the whitespace the strip leaves behind), which silently broke the cold
    shadow restore's byte-match -> a reprefill instead of a strict-extension.

    Form (identical bytes to the merge site — do NOT change):
      reasoning + non-empty content -> ``<think>\\n{r.strip()}\\n</think>\\n\\n{content}``
      reasoning + empty content     -> ``<think>\\n{r.strip()}\\n</think>``
    The empty-vs-present branch is decided on ``content.strip()`` (mirrors the merge
    site's ``if ct.strip():``), so all-whitespace content collapses to the empty form
    exactly as the merge site does.

    PURE: no I/O, no mutation, deterministic on ``(reasoning, content)``. The
    ``reasoning`` truthiness + ``"<think>" not in content`` GUARD stays at each CALL
    site (the three sites guard differently); this helper only FORMATS.
    """
    rc_stripped = reasoning.strip()
    if content.strip():
        return f"<think>\n{rc_stripped}\n</think>\n\n{content}"
    return f"<think>\n{rc_stripped}\n</think>"


# Cap on per-tool_call argument string size to bound memory + log spam when a
# model emits runaway args. 256 KiB tolerates realistic large payloads (image
# data URIs, long structured inputs) while stopping unbounded growth.
MAX_TOOL_ARG_CHARS = 262144


def _coerce_created_at(created_value: Any) -> Any:
    """Coerce OpenAI int unix-epoch ``created`` into ISO-8601 string for
    Ollama-compat ``created_at`` field. None and pre-formatted strings pass
    through unchanged.
    """
    if isinstance(created_value, int):
        return datetime.fromtimestamp(created_value, tz=timezone.utc).isoformat()
    return created_value


# === Ollama-style keep_alive parser =========================================

_KEEP_ALIVE_UNITS = {"s": 1, "m": 60, "h": 3600}  # read-only contract; do not mutate


def parse_keep_alive(value: Any) -> int | None:
    """Parse Ollama-style ``keep_alive`` field. Returns int seconds or None.

    Semantics (matches Ollama upstream):
      - ``None`` / unparseable → caller uses default ``idle_hot_load_seconds``
      - ``0`` (int/float/str ``"0"``/bool ``False``) → unload immediately
      - ``-1`` → pin (caller treats as :data:`KEEP_ALIVE_MAX_S`)
      - positive int seconds → clamped to ``[0, KEEP_ALIVE_MAX_S]`` by caller
      - Ollama-suffix strings ``"30s"``/``"5m"``/``"2h"`` → equivalent int seconds
      - bool ``True`` → ``None`` (Ollama "on" means "use server default")

    Single-layer clamp: this helper only normalises types; clamping to
    ``KEEP_ALIVE_MAX_S`` lives in :class:`TurbohaulManager` so there's one
    source of truth.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 0 if value is False else None
    if isinstance(value, (int, float)):
        v = int(value)
        if v < 0:
            return KEEP_ALIVE_MAX_S
        return v
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            v = int(s)
            if v < 0:
                return KEEP_ALIVE_MAX_S
            return v
        except ValueError:
            pass
        if (
            len(s) >= 2
            and s[-1].lower() in _KEEP_ALIVE_UNITS
            and s[:-1].lstrip("-").isdigit()
        ):
            return int(s[:-1]) * _KEEP_ALIVE_UNITS[s[-1].lower()]
    return None


def _derive_client_meta_identity(payload: dict, messages: list, ip: str | None = None) -> dict:
    """Parse OPTIONAL identity/classification fields out of an incoming request
    into a client_meta fragment.

    DORMANT FOUNDATION — decision-neutral. Nothing downstream reads these keys
    yet; they are stored on ``client_meta`` only. Absent fields → ``None``, so
    this is fully back-compat (zero behaviour change). Single source of truth so
    the three ``client_meta`` build sites stay byte-identical (no drift).

    The ``is_main`` / ``is_sub_agent`` / ``is_curator`` caller booleans are ALSO
    dormant/decision-neutral here (explicit payload field only, None when the
    caller omits them). A flag-gated curator route reads them off
    ``client_meta`` via ``kv_classify._class_from_label``; with the flag OFF or
    the fields absent they stay inert.

    Knob-leak guard: NONE of these keys are in ``_COMMON_FORWARDED_KNOBS`` /
    ``_STREAM_FORWARDED_KNOBS``, and BOTH the ``_complete`` loop and
    ``_build_stream_payload`` forward to llama-server by ITERATING that
    allow-list — so these identity keys can never reach the sidecar payload. Do
    NOT add any of them to the knob allow-list.

    All values are best-effort and None-safe; never raises on a missing or
    malformed field.

    NESTING: some callers send identity fields NESTED under
    payload["client_meta"] (session_id, is_main, is_sub_agent, is_curator,
    is_compression). This function reads from the nested client_meta with a
    TOP-LEVEL fallback (back-compat), activating the is_* labels that were
    previously dormant (never reached manager).

    ``ip`` (an observability surface) is captured here — the caller's
    ``request.client.host`` — from this ONE source of truth so all 3
    client_meta build sites stay byte-identical (no drift). Same
    dormant/decision-neutral contract as session_id/is_*: stored on
    client_meta only, never referenced by the knob-forwarding allow-lists, so
    it can never reach the llama-server payload.
    """
    # Read from nested client_meta with top-level fallback
    cm = payload.get("client_meta") if isinstance(payload.get("client_meta"), dict) else {}
    first = messages[0] if (messages and isinstance(messages[0], dict)) else None
    return {
        # Metadata about turn 0 / the system prompt. Explicit payload field wins;
        # else minimally derive {role, content_len} from the first message.
        # content_len reuses compute_ctx_len (the shared char-count rule) on the
        # first turn so non-str / None content never raises. None if underivable.
        "turn0_meta": payload.get("turn0_meta") or (
            {"role": first.get("role"), "content_len": compute_ctx_len([first])}
            if first is not None else None
        ),
        # Agent role — explicit harness field ONLY; do NOT infer from content.
        "role": payload.get("role") if payload.get("role") is not None else cm.get("role"),
        # Session identifier — DISTINCT from the derived thread_id; explicit only.
        "session_id": payload.get("session_id") if payload.get("session_id") is not None else cm.get("session_id"),
        # Compression-turn flag — explicit only; None when the harness omits it.
        "is_compression": payload.get("is_compression") if payload.get("is_compression") is not None else cm.get("is_compression"),
        # Identity-class booleans — explicit payload field ONLY, None when
        # absent (mirrors is_compression). Dormant/back-compat: NOT in the knob
        # allow-lists so they never reach llama-server; a flag-gated curator
        # route consumes them via kv_classify._class_from_label.
        "is_main": payload.get("is_main") if payload.get("is_main") is not None else cm.get("is_main"),
        "is_sub_agent": payload.get("is_sub_agent") if payload.get("is_sub_agent") is not None else cm.get("is_sub_agent"),
        "is_curator": payload.get("is_curator") if payload.get("is_curator") is not None else cm.get("is_curator"),
        # The front-end per-role KV-save toggle. Manager-side ONLY (like the is_*
        # labels) — NEVER in the knob allow-lists / never to llama-server.
        # Read by _role_save_enabled off slot.client_meta; absent -> per-role default.
        "save_kv": payload.get("save_kv") if payload.get("save_kv") is not None else cm.get("save_kv"),
        # Context size — explicit field, else the cheap serialized-context char
        # count (same rule as admission_ctx_len). Int (0 for empty), never None
        # once messages exist.
        "context_size": payload.get("context_size")
        if payload.get("context_size") is not None
        else compute_ctx_len(messages),
        # Source IP, captured by the caller from request.client.host.
        # Display/observability only — never in the knob allow-lists, never
        # forwarded to llama-server.
        "ip": ip,
    }


def _shadow_recompose_identity(base_thread_id: str, role, session_id) -> str:
    """DORMANT identity-shadow / activation precursor. Computes a role-keyed
    thread_id CANDIDATE that is LOGGED ONLY, never used. Today the manager
    IP-fallback collapses every caller role (main/sub/compression) onto ONE
    thread_id; the activation stage will fix that with a role-keyed thread_id.
    This dormant stage first proves — from the emitted corpus — that this
    recomposition only ever SPLITS an identity (never MERGES two) before we
    switch it on.

    Safe-by-construction:
      * APPEND-ONLY — new_key STARTS WITH the full ``base_thread_id`` (today's
        derived thread_id) and only APPENDS a suffix; it NEVER replaces the base,
        so two distinct bases can never recompose to the same key (no MERGE).
      * HASHED suffix — each field is a short SHA-256 hex prefix, so raw
        session_id / role never leak and a value containing the '-'/'=' delimiter
        cannot forge another field (delimiter-injection-proof).
      * No fields present -> new_key == base (identity). Best-effort / None-safe;
        a falsy role or session_id contributes nothing.

    Unified: delegates to kv_classify.recompose_identity (single source of truth
    — that pure function is byte-identical to the logic that lived here).
    """
    from turbohaul.kv_classify import recompose_identity
    return recompose_identity(base_thread_id, role, session_id)


def _m2b_active() -> bool:
    """Is role-keyed identity ACTIVATION on? When True the computed
    role+session-keyed new_key DRIVES the slot identity (thread_id); when False
    it stays dormant (log-only). env TURBOHAUL_M2B_ACTIVE (default OFF) — a
    runtime flag so it can be flipped OFF instantly if a MISMATCH/merge ever
    shows. resolve_kv / the restore gate CODE is byte-identical either way;
    ONLY the identity fed IN changes."""
    return os.environ.get("TURBOHAUL_M2B_ACTIVE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# === SSE tuning constants (module-level for monkeypatch-in-tests) ===

# How long to wait for the slot to actually reach ACTIVE before we give up.
# Cold-load of a 27B GGUF can take 30-60s; pre-stream wait should be much
# longer than that since the route is held open.
SLOT_READY_TIMEOUT_S = 7200.0
# httpx.stream timeout for the actual sidecar connection — keep generous for
# slow-thinking models on large contexts.
STREAM_TIMEOUT_S = 3600.0
# Emit `: keep-alive\n\n` SSE comments at this cadence while waiting for
# `slot.stream_ready_event` to fire. Many clients set 30-60s read-timeouts on
# streaming responses; without intermittent bytes the client disconnects during
# cold-load (a 27B GGUF takes 30-60s to load). SSE comments are RFC 8895 /
# EventSource-compliant; clients silently consume them and the connection stays
# warm.
HEARTBEAT_INTERVAL_S = 12.0


# === Typed upstream errors ===

class SidecarUnavailableError(RuntimeError):
    """Sidecar process disconnected, crashed, or is otherwise unreachable.

    Examples: httpx.RemoteProtocolError (server disconnected mid-response,
    typically OOM-crash from KV-cache pressure), ConnectError (port closed),
    ReadError (read failed). Maps to HTTP 503 + Retry-After at the route.
    """

    def __init__(self, message: str, cause: str = "sidecar_disconnected", retry_after_s: int = 30):
        super().__init__(message)
        self.cause = cause
        self.retry_after_s = retry_after_s


class SidecarUpstreamError(RuntimeError):
    """Sidecar accepted the request and returned a structured error response.

    Example: httpx.HTTPStatusError on 4xx (context overflow, malformed
    payload, etc.). Maps to HTTP 502 Bad Gateway at the route. The
    upstream status + (truncated) body are preserved for client diagnosis.
    """

    def __init__(self, message: str, upstream_status: int, upstream_body: str = ""):
        super().__init__(message)
        self.upstream_status = upstream_status
        self.upstream_body = upstream_body[:500]


class SidecarTimeoutError(RuntimeError):
    """Sidecar request timed out (httpx.TimeoutException).

    Maps to HTTP 504 Gateway Timeout. Client may retry but should consider
    reducing request size first.
    """

    def __init__(self, message: str, retry_after_s: int = 60):
        super().__init__(message)
        self.retry_after_s = retry_after_s


# ============================================================================
# OpenAI-compat /v1/chat/completions
# ============================================================================


@router.post("/v1/chat/completions")
async def openai_chat_completions(payload: dict, request: Request):
    """OpenAI-shape chat completion. Forwarded through manager.submit_and_wait
    for non-streaming requests; through manager.submit_for_streaming + an SSE
    pass-through generator for streaming requests.

    Return type is ``dict`` for non-streaming or ``fastapi.responses.StreamingResponse``
    for streaming.
    """
    # shallow copy to avoid mutating caller's payload dict
    payload = dict(payload)
    mgr = request.app.state.manager
    model = payload.get("model")
    messages = payload.get("messages")
    if not model:
        raise HTTPException(status_code=400, detail="`model` field required")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")
    # model tag must exist in the manifest store — mirrors the
    # read_manifest -> 404 pattern at embeddings.py:119 / ollama.py:68.
    try:
        read_manifest(mgr.boot.storage.manifests_path, model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {model}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Best-effort prompt extraction for thread-id derivation.
    # OpenAI tool_call messages have content=null (None), not "". The
    # .get("content", "") pattern returns None (not the default "") when content
    # is explicitly null, which crashes " ".join() with TypeError. Coerce None
    # to "" to tolerate tool-call assistant turns in multi-turn conversations.
    prompt = " ".join(
        (c if isinstance(c := m.get("content"), str) else "")
        for m in messages if isinstance(m, dict)
    )
    thread_id = payload.get("thread_id") or ""
    # Stable per-agent identity = the client container source IP, so an agent's
    # sequential turns grace-MATCH the same warm slot and reuse its KV prefix
    # cache (cache_reuse) instead of re-prefilling the full context each turn.
    # GATED on single-residency (cap<=1): at cap>=2 the full-prompt-hash identity
    # (manager.submit) is kept so concurrent sub-agent fan-out is NOT regressed.
    # Falls back to hash if no IP.
    try:
        _single_residency = request.app.state.manager.runtime.queue.max_parallel_sidecars <= 1
    except Exception:
        _single_residency = True
    if not thread_id and _single_residency:
        # MULTI-SIGNAL identity: IP (role tier) + fingerprint of the STATIC first
        # message (instance identity). IP-only collapses concurrent sub-agents
        # into ONE thread_id -> cross-context KV mismatch -> positive stale ->
        # CLEAR+reprefill. First-message fingerprint distinguishes sub-agents by
        # persona AND keeps a conversation's follow-ups (same first msg + larger
        # ctx) on ONE id -> KV reuse fires.
        _ip = request.client.host if (getattr(request, "client", None) and request.client.host) else ""
        _first_msg = next(
            (m.get("content", "") for m in (messages or [])
             if isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("content").strip()),
            "",
        )
        if _first_msg:
            _fp = derive_thread_id_prefix_hash(_first_msg, payload.get("model", ""))
            thread_id = ("agent-ip-" + _ip + "-" if _ip else "") + _fp
        elif _ip:
            thread_id = "agent-ip-" + _ip
    # === DORMANT identity-shadow (decision-neutral) ==========================
    # thread_id (old_key) is now finalized. Compute a role-keyed CANDIDATE and
    # LOG old-vs-new ONLY — the shadow is NEVER assigned to thread_id nor passed
    # anywhere (the activation stage enables it once this corpus proves it only
    # SPLITS, never MERGES). Best-effort: any failure is swallowed so it can't
    # break a request.
    try:
        old_key = thread_id
        role = payload.get("role")
        session_id = payload.get("session_id")
        # Recover session_id from idle_hot resident when payload omits it.
        # Tool-call re-cues arrive with session_id=None after the grace timer
        # fires, but the warm idle resident has the original session_id. Match
        # the incoming IP+fingerprint base thread_id against the idle resident's
        # thread_id (which has the s={sha256} suffix). Without this recovery,
        # _shadow_recompose_identity produces a session-LESS thread_id,
        # durable_ring_key returns None, KV-bin owner-match fails vs the
        # session-present bin → forced-full → full-context collapse.
        if session_id is None:
            try:
                mgr = request.app.state.manager
                if (getattr(mgr, "_idle_handle", None) is not None
                        and getattr(mgr, "_idle_client_meta", None)
                        and getattr(mgr, "_idle_thread_id", None)):
                    _idle_cm = mgr._idle_client_meta
                    _recovered = _idle_cm.get("session_id") if isinstance(_idle_cm, dict) else None
                    _idle_tid = mgr._idle_thread_id
                    # Match: incoming base thread_id is a prefix of the idle
                    # thread_id (idle has the s-suffix from its original
                    # derivation, incoming doesn't yet). Only recover when the
                    # base matches — don't hijack a different agent's session.
                    if _recovered and _idle_tid.startswith(old_key):
                        session_id = _recovered
                        log.info(
                            "session_id recovered from idle-hot resident: "
                            "thread_base=%s idle_thread=%s",
                            old_key[:24], _idle_tid[:24],
                        )
            except Exception:
                log.debug("session_id recovery failed (ignored)", exc_info=True)
        new_key = _shadow_recompose_identity(old_key, role, session_id)
        # When TURBOHAUL_M2B_ACTIVE, the role+session-keyed new_key DRIVES the
        # slot identity (warm-path role isolation). resolve_kv / the restore gate
        # CODE is byte-identical — ONLY the thread_id fed IN changes. Append-only
        # (new_key STARTS WITH old_key) + proven no-merge on live corpus: a SPLIT
        # mints a NEW identity (-> no bin -> fresh reprefill + saves own, never a
        # wrong-restore); a stable-key continuation keeps warm reuse. Flag-gated
        # so it can be flipped OFF instantly on any MISMATCH/merge.
        _m2b = _m2b_active()
        if _m2b and new_key != old_key:
            thread_id = new_key
        log.info(
            "identity_recompose old_h=%s new_h=%s role=%s session_present=%s "
            "ip_present=%s m2b_active=%s",
            hashlib.sha256(old_key.encode()).hexdigest()[:12],
            hashlib.sha256(new_key.encode()).hexdigest()[:12],
            role if role else "-",
            bool(session_id),
            bool(locals().get("_ip")),
            _m2b,
        )
    except Exception:
        log.debug("identity_recompose failed (ignored)", exc_info=True)
    # response_format pre-validation. Fires for BOTH stream + non-stream so SSE
    # clients cannot bypass via the wants_stream fork. Strict shape:
    # {type:"json_object"} accepted, {type:"text"} normalized to None (OpenAI
    # default — no-op pass-through), anything else REJECTED 400.
    rf = payload.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "response_format_unsupported_type",
                    "message": "response_format must be an object",
                    "received": type(rf).__name__,
                },
            )
        rf_type = rf.get("type")
        if rf_type == "text":
            payload["response_format"] = None  # OpenAI default — no-op
        elif rf_type == "json_object":
            pass  # accept-and-forward — no validation; manifest decides if model honors
        elif rf_type == "json_schema":
            # Validate caller schema; on bad → 422 schema_validation_failed.
            # On ok, response_format propagates via the existing
            # _COMMON_FORWARDED_KNOBS tuple (no separate forwarding code path needed).
            ok, reason = _validate_json_schema(rf)
            if not ok:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "schema_validation_failed",
                        "message": f"json_schema validation failed: {reason}",
                    },
                )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "response_format_unsupported_type",
                    "message": (
                        "response_format type must be one of "
                        "'text', 'json_object', 'json_schema'"
                    ),
                    "received_type": str(rf_type),
                },
            )
    wants_stream = bool(payload.get("stream", False))

    # SSE streaming pass-through: when the client sends stream=true, branch to
    # the streaming helper which opens its own httpx.stream() to the sidecar and
    # yields SSE chunks back to the client. The non-streaming path below is
    # unchanged.
    if wants_stream:
        return await _openai_chat_completions_stream(
            request, mgr, model, messages, prompt, thread_id, payload,
        )

    # When caller requested json_schema, read the manifest to slice out
    # reasoning_budget for the in-_complete retry-path gate. Only fires for
    # json_schema requests (narrow scope — text/json_object/no-response_format
    # never need this read). Read failure is non-fatal: thinking_manifest stays
    # empty + retry path is disabled, primary forward still proceeds via
    # _COMMON_FORWARDED_KNOBS.
    thinking_manifest: dict = {}
    if (
        isinstance(payload.get("response_format"), dict)
        and payload["response_format"].get("type") == "json_schema"
    ):
        try:
            _m = read_manifest(mgr.boot.storage.manifests_path, model)
            thinking_manifest = {
                "reasoning_budget": (_m.llama_server_flags or {}).get(
                    "reasoning_budget", 0,
                ),
            }
        except Exception:
            log.exception(
                "manifest read failed for model=%s; retry-path disabled",
                model,
            )

    client_meta = {
        "kind": "openai-chat-completion",
        "messages": messages,  # carried for the completion_fn to forward; redacted from /ws/state
        "model": model,
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "stream": False,
        "max_tokens": payload.get("max_tokens"),
        # Ollama-style keep_alive → IDLE_HOT extension hint
        "keep_alive_s": parse_keep_alive(payload.get("keep_alive")),
        # Forward validated response_format to llama-server via _complete's
        # _COMMON_FORWARDED_KNOBS loop. Skipping this line would silently drop
        # the field on the non-stream path despite the validator accepting it.
        "response_format": payload.get("response_format"),
        # Manifest reasoning_budget slice for in-_complete is_thinking_payload
        # check. {} ⇒ retry disabled (non-thinking or read-fail).
        "thinking_manifest": thinking_manifest,
        # Forward tools-family knobs into client_meta so the _complete closure
        # includes them in the llama-server payload AND so
        # maybe_recover_tool_calls can read the advertised tools allowlist.
        # Mirrors the /api/chat endpoint client_meta build — the OpenAI endpoint
        # otherwise silently dropped tools on the floor.
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
        "parallel_tool_calls": payload.get("parallel_tool_calls"),
        "function_call": payload.get("function_call"),
        "functions": payload.get("functions"),
        # DORMANT identity/classification foundation. Decision-neutral: stored
        # only, read nowhere; NOT in the knob allow-list so it never reaches the
        # llama-server payload. See _derive_client_meta_identity.
        **_derive_client_meta_identity(
            payload, messages,
            ip=request.client.host if (getattr(request, "client", None) and request.client.host) else None,
        ),
    }

    # Client-disconnect watcher. Event constructed IN-HANDLER so it's bound to
    # the route's request-loop (correct-loop guarantee). watch_disconnect polls
    # request.is_disconnected() every 2s; if the client closes, it sets the Event
    # which queue.pop_next sees and evicts the slot.
    disconnect_event = asyncio.Event()
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))
    try:
        slot, result = await mgr.submit_and_wait(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            context=list(messages),
            client_meta=client_meta,
            disconnect_event=disconnect_event,
            admission_ctx_len=compute_ctx_len(messages),
            # Incoming turn-hash chain (parallel to admission_ctx_len) so the
            # manager restore path classifies by prefix-VALIDITY vs the pinned
            # clean bin. Same _prefix_hash_chain used at save time, so both sides
            # of the comparison are byte-identical hashes.
            admission_hash_chain=_prefix_hash_chain(messages),
        )
    except SlotEvictedError as e:
        # Client closed connection before slot activated; surface as HTTP 499
        # (client_closed_request) so monitoring distinguishes client-side close
        # from a real 500 server fault.
        raise HTTPException(
            status_code=499,
            detail={"error": "client_closed_request", "message": str(e)},
        ) from e
    except SidecarUnavailableError as e:
        # 503 — sidecar crashed/disconnected (likely KV-cache OOM mid-response)
        raise HTTPException(
            status_code=503,
            detail={"error": "sidecar_unavailable", "cause": e.cause, "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarTimeoutError as e:
        # 504 — request exceeded sidecar timeout
        raise HTTPException(
            status_code=504,
            detail={"error": "sidecar_timeout", "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarUpstreamError as e:
        # 502 — sidecar returned an upstream 4xx/5xx (context overflow,
        # malformed payload, etc.)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_sidecar_error",
                "upstream_status": e.upstream_status,
                "upstream_body": e.upstream_body,
                "message": str(e),
            },
        ) from e
    except RuntimeError as e:
        # loading-fail / safety-gate-refused / unknown worker exception → 500
        raise HTTPException(status_code=500, detail=f"sidecar failed: {e}") from e
    finally:
        # Tear down the disconnect watcher cleanly. contextlib.suppress so a
        # normal cancellation doesn't surface from finally.
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watch_task

    if result is None:
        # Default completion_fn (no real backend wired) — return an empty echo
        raise HTTPException(
            status_code=503,
            detail="no completion_fn wired - production needs make_llama_server_complete_fn",
        )
    return result


# ============================================================================
# Streaming helper
# ============================================================================


# Knobs forwarded to llama-server from client_meta. Split into _COMMON
# (forwarded everywhere) and _STREAM_ONLY (only included in the streaming
# payload). `_complete` iterates _COMMON only so non-streaming requests never
# inherit stream-only keys; the streaming payload-build helper uses the derived
# _STREAM_FORWARDED_KNOBS alias. Drift between the two lists is now
# structurally impossible.
_COMMON_FORWARDED_KNOBS = (
    # Core OpenAI-compat
    "temperature", "top_p", "top_k", "max_tokens", "min_p",
    # Accept-and-forward only; the handler-entry validator rejects json_schema
    # as deferred. Thinking-mode JSON guarantee blocked upstream on
    # llama.cpp #20345 + Ollama #10538.
    "response_format",
    # Preserved-thinking controls
    "thinking_budget_tokens", "reasoning_budget", "reasoning",
    # Ollama-parity samplers
    "presence_penalty", "frequency_penalty", "repeat_penalty",
    "repeat_last_n", "typical_p", "seed",
    "mirostat", "mirostat_lr", "mirostat_ent",
    # max-output alias
    "n_predict",
    # Tool-call pass-through (minimal — full capability advertisement +
    # size-cap + per-model gating deferred to a follow-on; this is "forward the
    # field to a model that supports tool_calls natively, e.g. a 27B dense
    # model"). llama-server mirrors OpenAI's schema, so structured values
    # (list/dict/string) just pass through unchanged.
    "tools", "tool_choice", "parallel_tool_calls",
    "function_call", "functions",
)
_STREAM_ONLY_KNOBS = ("stream", "stream_options")
# Back-compat alias — `_build_stream_payload` reads this for the streaming
# llama-server call where `stream=True` must be forwarded.
_STREAM_FORWARDED_KNOBS = _COMMON_FORWARDED_KNOBS + _STREAM_ONLY_KNOBS


def _resolve_model_identity(
    model_tag: str,
    handle_model_id: "str | None",
    manifests_path: "Path | None",
) -> str:
    """Resolve the model identity sent to the sidecar's `model` field.

    mlx_lm server routes each completion by the request `model` field and
    re-resolves it as a HuggingFace repo. The Turbohaul tag (e.g.
    "darwin-35b-a3b-opus-oq4") is NOT a valid repo id or local path, so sending
    it yields a 404 "Repository Not Found". Always prefer a real backend
    identity: the sidecar handle's model_id (set by mlx_spawn to the --model
    arg), else the manifest's model_path / model_repo, else the tag as a last
    resort (llama.cpp serves by tag).
    """
    if handle_model_id:
        return handle_model_id
    if manifests_path:
        try:
            _m = read_manifest(manifests_path, model_tag)
            identity = _m.model_path or _m.model_repo
            if identity:
                return identity
        except Exception:
            pass
    return model_tag


def _build_stream_payload(
    client_meta: dict, model: str, messages: list, model_identity: str | None = None
) -> dict:
    """Build the streaming chat-completions payload sent to the sidecar.

    `model_identity` (when provided) is the real backend identity (local path /
    HF repo id); falls back to `model` (the Turbohaul tag) otherwise.
    """
    payload: dict = {
        "model": model_identity or model,
        "messages": messages,
        "stream": True,
    }
    for k in _STREAM_FORWARDED_KNOBS:
        v = client_meta.get(k)
        if v is not None:
            payload[k] = v
    return payload


def _stream_error_frame(error: str, message: str, **extra: Any) -> bytes:
    """Build a synthetic OpenAI-compat SSE error frame.

    OpenAI's streaming wire-format expresses mid-stream errors as a final
    ``data: {"error": {...}}\\n\\n`` chunk followed by ``data: [DONE]\\n\\n``.
    Keeps HTTP 200 once the response has started; the client SDK surfaces
    the error during chunk iteration.
    """
    body: dict[str, Any] = {"error": {"type": error, "message": message[:500]}}
    body["error"].update(extra)
    return f"data: {json.dumps(body)}\n\n".encode()


async def _openai_chat_completions_stream(
    request: Request,
    mgr,
    model: str,
    messages: list,
    prompt: str,
    thread_id: str,
    payload: dict,
) -> StreamingResponse:
    """SSE streaming pass-through.

    Submits via ``manager.submit_for_streaming`` (slot held ACTIVE for full
    stream lifetime — single-slot invariant preserved). Awaits
    ``slot.stream_ready_event`` so we know the sidecar is up and
    ``slot.stream_handle`` is populated. Then opens our own
    ``httpx.stream("POST", url, ...)`` to the sidecar and pipes raw SSE bytes
    back to the client. On end-of-stream / disconnect / error sets
    ``slot.stream_done_event`` so the manager can advance ACTIVE → GRACE.

    Wrapper ``_merge_reasoning_into_content`` is intentionally SKIPPED on the
    streaming path: most modern streaming consumers (agent harnesses, langchain,
    Open WebUI, OpenAI SDK) parse ``delta.content`` and
    ``delta.reasoning_content`` independently. Per-chunk merge would require
    accumulator/reorder state and would break the token-by-token UX.
    """
    client_meta = {
        "kind": "openai-chat-completion-stream",
        "messages": messages,
        "model": model,
        "stream": True,
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "max_tokens": payload.get("max_tokens"),
        # Ollama-style keep_alive → IDLE_HOT extension hint. The streaming path
        # is agent clients' primary entry, so this hint matters most here.
        "keep_alive_s": parse_keep_alive(payload.get("keep_alive")),
        # All forwardable knobs carried for the streaming payload helper.
        **{k: payload.get(k) for k in _STREAM_FORWARDED_KNOBS if payload.get(k) is not None},
        # DORMANT identity/classification foundation. Decision-neutral: stored
        # only, read nowhere; NOT in the knob allow-list so it never reaches the
        # llama-server payload. See _derive_client_meta_identity.
        **_derive_client_meta_identity(
            payload, messages,
            ip=request.client.host if (getattr(request, "client", None) and request.client.host) else None,
        ),
    }

    # Client-disconnect eviction for the STREAMING path. The non-streaming path
    # wires this too, but the streaming path historically did NOT, so a streaming
    # slot that sat QUEUED behind a client that already hung up could never be
    # evicted. Carry a disconnect_event into
    # submit so the queue marks the slot is_evicted on disconnect; the fan-out
    # admit then SKIPS dead-client riders instead of burning a --parallel slot.
    # The watcher task is started after a successful submit (covers the
    # queue-wait window) and cancelled in stream_gen's finally.
    disconnect_event = asyncio.Event()
    # Pre-stream submission errors → standard HTTPException with proper status code.
    try:
        slot = await mgr.submit_for_streaming(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            context=list(messages),
            client_meta=client_meta,
            disconnect_event=disconnect_event,
            admission_ctx_len=compute_ctx_len(messages),
            # Incoming turn-hash chain (parallel to admission_ctx_len) so the
            # manager restore path classifies by prefix-VALIDITY vs the pinned
            # clean bin. Same _prefix_hash_chain used at save time, so both sides
            # of the comparison are byte-identical hashes.
            admission_hash_chain=_prefix_hash_chain(messages),
        )
    except SidecarUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "sidecar_unavailable", "cause": e.cause, "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "sidecar_timeout", "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarUpstreamError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_sidecar_error",
                "upstream_status": e.upstream_status,
                "upstream_body": e.upstream_body,
                "message": str(e),
            },
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"sidecar failed: {e}") from e

    # Submit succeeded: start the disconnect watcher now (covers the queue-wait
    # window before the slot reaches ACTIVE). Created here rather than before the
    # submit so a pre-stream submit failure cannot leak a watcher task. Cancelled
    # in stream_gen's finally.
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))

    async def stream_gen():
        gen_id_for_tee = None
        first_token_received = False
        prefill_start = time.monotonic()
        # Accumulate the streamed generated assistant text (content +
        # reasoning) so the manager can reconstruct the engine-view warm_chain for
        # the STREAMING forced clean-restore. Best-effort, fail-open — a parse miss
        # just leaves the warm state unknown (gate safe-degrades to no-force).
        _gen_content: list[str] = []
        _gen_reasoning: list[str] = []
        _sse_carry = ""
        try:
            # Wait for worker_loop to bring slot to ACTIVE + assign handle.
            # Emit `: keep-alive\n\n` SSE comments every HEARTBEAT_INTERVAL_S so
            # clients with 30-60s read-timeouts don't disconnect during
            # cold-load. asyncio.shield prevents the heartbeat wait_for from
            # cancelling the underlying ready_task.
            ready_task = asyncio.create_task(slot.stream_ready_event.wait())
            loop = asyncio.get_running_loop()
            deadline = loop.time() + SLOT_READY_TIMEOUT_S
            while not ready_task.done():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    ready_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ready_task
                    yield _stream_error_frame(
                        "slot_ready_timeout",
                        f"Slot did not reach ACTIVE within {SLOT_READY_TIMEOUT_S}s",
                    )
                    yield b"data: [DONE]\n\n"
                    return
                try:
                    await asyncio.wait_for(
                        asyncio.shield(ready_task),
                        timeout=min(HEARTBEAT_INTERVAL_S, remaining),
                    )
                except asyncio.TimeoutError:
                    if not ready_task.done():
                        yield b": keep-alive\n\n"

            handle = slot.stream_handle
            if handle is None:
                yield _stream_error_frame(
                    "no_sidecar_handle",
                    "Slot reached ACTIVE but stream_handle is None",
                )
                yield b"data: [DONE]\n\n"
                return

            # Live monitor: unify text-plane identity with the metrics-plane
            # poller (same pid:spawn_seq:slot_id -> same generation_id). slot_id
            # is per-request unique, so turns of one conversation never collide.
            # Use the spawn_seq of the resident actually serving THIS model_tag,
            # not the singleton. At cap>=2 the dispatcher never bumps the
            # singleton's spawn_seq (stays 0) while the metrics supervisor hashes
            # the model_tag resident's bumped spawn_seq; using _read_spawn_seq(mgr)
            # (=singleton) here made the text-plane tee feed a DIFFERENT
            # generation_id than the SSE anchor, so the live pane subscribed to an
            # unfed buffer and showed nothing. _spawn_seq_for_model unifies both
            # planes (byte-identical at cap<=1).
            gen_id_for_tee = compute_generation_id(
                handle.pid, mgr._spawn_seq_for_model(model), slot.slot_id or slot.thread_id
            )

            model_identity = _resolve_model_identity(
                model, handle.model_id, mgr.boot.storage.manifests_path
            )
            stream_payload = _build_stream_payload(
                client_meta, model, messages, model_identity
            )
            url = f"http://127.0.0.1:{handle.port}/v1/chat/completions"

            async with httpx.AsyncClient(timeout=STREAM_TIMEOUT_S) as client:
                # Prefill keep-alive: the silence that disconnects clients is
                # during the STREAM OPEN, not the byte loop.
                # ``client.stream(...).__aenter__()`` blocks until llama-server
                # returns the response, which it withholds for the ENTIRE prefill
                # (no headers/bytes until generation begins — observed: ~55s
                # silent on a 60K prompt, ALL of it inside the open). So we open
                # the stream MANUALLY and emit ': keep-alive' every
                # HEARTBEAT_INTERVAL_S while the
                # open is pending. asyncio.shield keeps the open alive across a tick
                # (never cancel it on a tick). httpx errors from the open still
                # propagate to the outer except handlers below. Mirrors the existing
                # slot-ready heartbeat pattern.
                stream_cm = client.stream(
                    "POST", url, json=stream_payload, timeout=STREAM_TIMEOUT_S,
                )
                open_task = asyncio.ensure_future(stream_cm.__aenter__())
                try:
                    r = None
                    while r is None:
                        try:
                            r = await asyncio.wait_for(
                                asyncio.shield(open_task), timeout=HEARTBEAT_INTERVAL_S
                            )
                        except asyncio.TimeoutError:
                            # Sidecar still prefilling (no response yet): keep the
                            # client connection warm; keep waiting on the SAME open.
                            yield b": keep-alive\n\n"
                            # telemetry — keep-alive emission
                            try:
                                mgr_telemetry = getattr(mgr, "_telemetry", None)
                                if mgr_telemetry is not None:
                                    mgr_telemetry.on_keep_alive_emitted(
                                        slot_id=slot.slot_id,
                                        is_queued=False,
                                        thread_id=slot.thread_id,
                                    )
                            except Exception:
                                pass  # observe-only

                    # raise_for_status is NOT auto-called by httpx.stream;
                    # the body may not be loaded yet so we read it manually if
                    # the upstream returned a 4xx/5xx.
                    if r.status_code >= 400:
                        # Cap the error-body read so a stalled 4xx/5xx body can't
                        # block un-heartbeated until STREAM_TIMEOUT_S (error bodies
                        # are tiny + already generated; empty on timeout).
                        try:
                            body_bytes = await asyncio.wait_for(
                                r.aread(), timeout=HEARTBEAT_INTERVAL_S
                            )
                        except asyncio.TimeoutError:
                            body_bytes = b""
                        body_str = body_bytes.decode("utf-8", errors="replace")[:500]
                        yield _stream_error_frame(
                            "upstream_sidecar_error",
                            f"sidecar returned HTTP {r.status_code}",
                            upstream_status=r.status_code,
                            upstream_body=body_str,
                        )
                        yield b"data: [DONE]\n\n"
                        return

                    # Pipe raw SSE bytes from llama-server straight through. Also
                    # heartbeat-guarded (a decode that stalls >12s gets keep-alives);
                    # shield keeps the in-flight read alive across a tick — NEVER
                    # cancel a mid-flight read (could drop bytes / corrupt the stream).
                    aiter = r.aiter_bytes()
                    next_read = asyncio.ensure_future(aiter.__anext__())
                    try:
                        while True:
                            try:
                                chunk_bytes = await asyncio.wait_for(
                                    asyncio.shield(next_read),
                                    timeout=HEARTBEAT_INTERVAL_S,
                                )
                            except asyncio.TimeoutError:
                                yield b": keep-alive\n\n"
                                continue
                            except StopAsyncIteration:
                                break
                            if chunk_bytes:
                                yield chunk_bytes
                                # telemetry — first token (TTFT)
                                if not first_token_received:
                                    first_token_received = True
                                    ttft_ms = (time.monotonic() - prefill_start) * 1000.0
                                    try:
                                        mgr_telemetry = getattr(mgr, "_telemetry", None)
                                        if mgr_telemetry is not None:
                                            mgr_telemetry.on_first_token(slot, ttft_ms)
                                    except Exception:
                                        pass
                                # Live monitor: passive output-text tee. Client yield
                                # FIRST, tee SECOND, fail-open — never delays/reorders/
                                # corrupts the client stream.
                                try:
                                    mgr.live_output.feed(gen_id_for_tee, chunk_bytes)
                                except Exception:
                                    pass
                                # Parse the SSE deltas to accumulate the
                                # generated content + reasoning (for warm_chain). Yield
                                # + tee happen FIRST; this is passive + fail-open and
                                # never touches the bytes handed to the client.
                                try:
                                    _txt = _sse_carry + chunk_bytes.decode("utf-8", errors="replace")
                                    _lines = _txt.split("\n")
                                    _sse_carry = _lines.pop()  # last (possibly partial) line
                                    for _ln in _lines:
                                        _ln = _ln.strip()
                                        if not _ln.startswith("data:"):
                                            continue
                                        _pl = _ln[5:].strip()
                                        if not _pl or _pl == "[DONE]":
                                            continue
                                        _delta = (json.loads(_pl).get("choices") or [{}])[0].get("delta") or {}
                                        if _delta.get("content"):
                                            _gen_content.append(_delta["content"])
                                        if _delta.get("reasoning_content"):
                                            _gen_reasoning.append(_delta["reasoning_content"])
                                except Exception:
                                    pass  # fail-open: warm state stays unknown -> no force
                            next_read = asyncio.ensure_future(aiter.__anext__())
                    finally:
                        if not next_read.done():
                            next_read.cancel()
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await next_read
                finally:
                    # Always release the stream we opened manually. If the open never
                    # completed (client disconnect mid-prefill), cancel+reap it; ONLY
                    # call __aexit__ when __aenter__ actually succeeded (else there is
                    # nothing entered to exit).
                    if not open_task.done():
                        open_task.cancel()
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await open_task
                    # Compute opened_ok ORDERING-INDEPENDENTLY (a cancelled/pending
                    # task's .exception() RAISES — don't rely on short-circuit term
                    # order surviving a future refactor).
                    opened_ok = False
                    if open_task.done() and not open_task.cancelled():
                        try:
                            opened_ok = open_task.exception() is None
                        except asyncio.CancelledError:
                            opened_ok = False
                    if opened_ok:
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await stream_cm.__aexit__(None, None, None)
        except (
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.CloseError,
            httpx.ProtocolError,
        ) as e:
            yield _stream_error_frame(
                "sidecar_unavailable",
                f"{type(e).__name__}: {e}",
                cause="sidecar_disconnected_or_crashed",
            )
            yield b"data: [DONE]\n\n"
        except httpx.TimeoutException as e:
            yield _stream_error_frame(
                "sidecar_timeout",
                f"{type(e).__name__}: {e}",
            )
            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            # Client disconnected mid-stream — propagate cancellation but
            # ensure cleanup runs in `finally` below. Do NOT yield any
            # additional frames after cancellation (the connection is dead).
            log.info(
                "client disconnect during stream slot=%s thread=%s",
                slot.slot_id, slot.thread_id,
            )
            raise
        except Exception as e:  # pragma: no cover — defensive
            log.exception(
                "unexpected error in stream_gen slot=%s", slot.slot_id,
            )
            yield _stream_error_frame("internal_error", str(e))
            yield b"data: [DONE]\n\n"
        finally:
            # Tear down the disconnect watcher. Cancel + await so no "Task was
            # destroyed but it is pending" leak.
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watch_task
            # Stash the accumulated generated turn (with <think>) on the slot
            # BEFORE signaling done, so the manager (which wakes on
            # stream_done_event) computes the engine-view warm_chain for the
            # streaming no-downgrade gate. Use the SINGLE-SOURCE
            # wrap_reasoning_think (the exact _merge_reasoning_into_content
            # newline form) so this shadow reconstruction byte-matches what the
            # harness stores/resends after its think-strip — the old no-newline
            # `<think>{r}</think>{c}` form diverged on multi-block / stray-tag
            # content. Empty -> left None (warm unknown).
            try:
                _c = "".join(_gen_content)
                _r = "".join(_gen_reasoning)
                _merged = wrap_reasoning_think(_r, _c) if (_r and "<think>" not in _c) else _c
                if _merged:
                    slot.streamed_assistant_text = _merged
            except Exception:
                pass
            # Signal worker_loop to advance the slot ACTIVE → GRACE.
            # Idempotent: setting an already-set Event is a no-op.
            if slot.stream_done_event is not None and not slot.stream_done_event.is_set():
                slot.stream_done_event.set()
            # Live monitor: close the output buffer so SSE subscribers get `done`.
            if gen_id_for_tee is not None:
                with contextlib.suppress(Exception):
                    mgr.live_output.mark_done(gen_id_for_tee)

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable buffering at nginx / other reverse proxies so chunks
            # reach the client in real time.
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Ollama-compat /api/chat
# ============================================================================


@router.post("/api/chat")
async def ollama_chat(payload: dict, request: Request) -> dict:
    """Ollama-shape chat. Internally forwarded as OpenAI to llama-server then
    re-shaped to Ollama on return."""
    # shallow copy to avoid mutating caller's payload dict
    payload = dict(payload)
    mgr = request.app.state.manager
    model = payload.get("model")
    messages = payload.get("messages")
    if not model or not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`model` + `messages` required")
    # model tag must exist in the manifest store — mirrors the
    # read_manifest -> 404 pattern at embeddings.py:119 / ollama.py:68.
    try:
        read_manifest(mgr.boot.storage.manifests_path, model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {model}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # OpenAI tool_call messages have content=null (None), not "". Coerce None to
    # "" to tolerate tool-call assistant turns.
    prompt = " ".join(
        (c if isinstance(c := m.get("content"), str) else "")
        for m in messages if isinstance(m, dict)
    )
    thread_id = payload.get("thread_id") or ""
    # Stable per-agent identity = the client container source IP, so an agent's
    # sequential turns grace-MATCH the same warm slot and reuse its KV prefix
    # cache (cache_reuse) instead of re-prefilling the full context each turn.
    # GATED on single-residency (cap<=1): at cap>=2 the full-prompt-hash identity
    # (manager.submit) is kept so concurrent sub-agent fan-out is NOT regressed.
    # Falls back to hash if no IP.
    try:
        _single_residency = request.app.state.manager.runtime.queue.max_parallel_sidecars <= 1
    except Exception:
        _single_residency = True
    if not thread_id and _single_residency:
        # MULTI-SIGNAL identity: IP (role tier) + fingerprint of the STATIC first
        # message (instance identity). IP-only collapses concurrent sub-agents
        # into ONE thread_id -> cross-context KV mismatch -> positive stale ->
        # CLEAR+reprefill. First-message fingerprint distinguishes sub-agents by
        # persona AND keeps a conversation's follow-ups (same first msg + larger
        # ctx) on ONE id -> KV reuse fires.
        _ip = request.client.host if (getattr(request, "client", None) and request.client.host) else ""
        _first_msg = next(
            (m.get("content", "") for m in (messages or [])
             if isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("content").strip()),
            "",
        )
        if _first_msg:
            _fp = derive_thread_id_prefix_hash(_first_msg, payload.get("model", ""))
            thread_id = ("agent-ip-" + _ip + "-" if _ip else "") + _fp
        elif _ip:
            thread_id = "agent-ip-" + _ip
    # === DORMANT identity-shadow (decision-neutral) ==========================
    # thread_id (old_key) is now finalized. Compute a role-keyed CANDIDATE and
    # LOG old-vs-new ONLY — the shadow is NEVER assigned to thread_id nor passed
    # anywhere (the activation stage enables it once this corpus proves it only
    # SPLITS, never MERGES). Best-effort: any failure is swallowed so it can't
    # break a request.
    try:
        old_key = thread_id
        role = payload.get("role")
        session_id = payload.get("session_id")
        # Recover session_id from idle_hot resident when payload omits it.
        # Tool-call re-cues arrive with session_id=None after the grace timer
        # fires, but the warm idle resident has the original session_id. Match
        # the incoming IP+fingerprint base thread_id against the idle resident's
        # thread_id (which has the s={sha256} suffix). Without this recovery,
        # _shadow_recompose_identity produces a session-LESS thread_id,
        # durable_ring_key returns None, KV-bin owner-match fails vs the
        # session-present bin → forced-full → full-context collapse.
        if session_id is None:
            try:
                mgr = request.app.state.manager
                if (getattr(mgr, "_idle_handle", None) is not None
                        and getattr(mgr, "_idle_client_meta", None)
                        and getattr(mgr, "_idle_thread_id", None)):
                    _idle_cm = mgr._idle_client_meta
                    _recovered = _idle_cm.get("session_id") if isinstance(_idle_cm, dict) else None
                    _idle_tid = mgr._idle_thread_id
                    # Match: incoming base thread_id is a prefix of the idle
                    # thread_id (idle has the s-suffix from its original
                    # derivation, incoming doesn't yet). Only recover when the
                    # base matches — don't hijack a different agent's session.
                    if _recovered and _idle_tid.startswith(old_key):
                        session_id = _recovered
                        log.info(
                            "session_id recovered from idle-hot resident: "
                            "thread_base=%s idle_thread=%s",
                            old_key[:24], _idle_tid[:24],
                        )
            except Exception:
                log.debug("session_id recovery failed (ignored)", exc_info=True)
        new_key = _shadow_recompose_identity(old_key, role, session_id)
        # When TURBOHAUL_M2B_ACTIVE, the role+session-keyed new_key DRIVES the
        # slot identity (warm-path role isolation). resolve_kv / the restore gate
        # CODE is byte-identical — ONLY the thread_id fed IN changes. Append-only
        # (new_key STARTS WITH old_key) + proven no-merge on live corpus: a SPLIT
        # mints a NEW identity (-> no bin -> fresh reprefill + saves own, never a
        # wrong-restore); a stable-key continuation keeps warm reuse. Flag-gated
        # so it can be flipped OFF instantly on any MISMATCH/merge.
        _m2b = _m2b_active()
        if _m2b and new_key != old_key:
            thread_id = new_key
        log.info(
            "identity_recompose old_h=%s new_h=%s role=%s session_present=%s "
            "ip_present=%s m2b_active=%s",
            hashlib.sha256(old_key.encode()).hexdigest()[:12],
            hashlib.sha256(new_key.encode()).hexdigest()[:12],
            role if role else "-",
            bool(session_id),
            bool(locals().get("_ip")),
            _m2b,
        )
    except Exception:
        log.debug("identity_recompose failed (ignored)", exc_info=True)
    # Ollama accepts keep_alive at top level OR nested under options.
    ka_raw = payload.get("keep_alive")
    if ka_raw is None and isinstance(payload.get("options"), dict):
        ka_raw = payload["options"].get("keep_alive")
    # response_format pre-validation. Fires for BOTH stream + non-stream and
    # BEFORE the streaming-tools guard below so the deferred type ('json_schema')
    # gets a clean 400 regardless of the stream flag. Mirror of
    # openai_chat_completions validator; minor wording tweak in the detail for
    # endpoint clarity.
    rf = payload.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "response_format_unsupported_type",
                    "message": "response_format must be an object",
                    "received": type(rf).__name__,
                },
            )
        rf_type = rf.get("type")
        if rf_type == "text":
            payload["response_format"] = None  # OpenAI default — no-op
        elif rf_type == "json_object":
            pass  # accept-and-forward
        elif rf_type == "json_schema":
            # Mirror of openai_chat_completions json_schema branch.
            ok, reason = _validate_json_schema(rf)
            if not ok:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "schema_validation_failed",
                        "message": f"json_schema validation failed: {reason}",
                    },
                )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "response_format_unsupported_type",
                    "message": (
                        "response_format type must be one of "
                        "'text', 'json_object', 'json_schema'"
                    ),
                    "received_type": str(rf_type),
                },
            )
    # Streaming + tools is not yet supported on this endpoint. Cheap defensive
    # guard before submit_and_wait so callers get a clean 400 instead of a
    # confusing partial-tool stream.
    if payload.get("stream") and any(
        payload.get(k)
        for k in ("tools", "tool_choice", "parallel_tool_calls",
                  "function_call", "functions")
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "streaming_with_tools_deferred",
                "message": (
                    "Ollama-shape streaming + tool_calls is not yet supported. "
                    "Use stream=false for tool requests or /v1/chat/completions "
                    "for OpenAI-shape streaming-tools."
                ),
                "follow_on": "planned",
            },
        )
    # Mirror of Site A manifest read — slice reasoning_budget for in-_complete
    # retry-path gate. Only fires for json_schema requests.
    thinking_manifest: dict = {}
    if (
        isinstance(payload.get("response_format"), dict)
        and payload["response_format"].get("type") == "json_schema"
    ):
        try:
            _m = read_manifest(mgr.boot.storage.manifests_path, model)
            thinking_manifest = {
                "reasoning_budget": (_m.llama_server_flags or {}).get(
                    "reasoning_budget", 0,
                ),
            }
        except Exception:
            log.exception(
                "manifest read failed for model=%s (ollama_chat); retry-path disabled",
                model,
            )
    client_meta = {
        "kind": "ollama-chat",
        "messages": messages,
        "model": model,
        "stream": bool(payload.get("stream", False)),
        "options": payload.get("options"),
        "keep_alive_s": parse_keep_alive(ka_raw),
        # Forward Ollama-shape tool knobs into client_meta so _complete can pass
        # them through to llama-server. Without this, ollama_chat tool requests
        # silently dropped tools on the floor before reaching the sidecar.
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
        "parallel_tool_calls": payload.get("parallel_tool_calls"),
        "function_call": payload.get("function_call"),
        "functions": payload.get("functions"),
        # Forward validated response_format. Same rationale as the OpenAI path —
        # the handler-entry validator alone does not propagate the field; the
        # explicit add line keeps the non-stream ollama path honest.
        "response_format": payload.get("response_format"),
        # Manifest reasoning_budget slice for in-_complete is_thinking_payload
        # gate. Mirrors the OpenAI path; {} ⇒ retry disabled.
        "thinking_manifest": thinking_manifest,
        # DORMANT identity/classification foundation. Decision-neutral: stored
        # only, read nowhere; NOT in the knob allow-list so it never reaches the
        # llama-server payload. See _derive_client_meta_identity.
        **_derive_client_meta_identity(
            payload, messages,
            ip=request.client.host if (getattr(request, "client", None) and request.client.host) else None,
        ),
    }
    # Client-disconnect watcher (ollama_chat mirror of the OpenAI path).
    disconnect_event = asyncio.Event()
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))
    try:
        slot, result = await mgr.submit_and_wait(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            context=list(messages),
            client_meta=client_meta,
            disconnect_event=disconnect_event,
            admission_ctx_len=compute_ctx_len(messages),
            # Incoming turn-hash chain (parallel to admission_ctx_len) so the
            # manager restore path classifies by prefix-VALIDITY vs the pinned
            # clean bin. Same _prefix_hash_chain used at save time, so both sides
            # of the comparison are byte-identical hashes.
            admission_hash_chain=_prefix_hash_chain(messages),
        )
    except SlotEvictedError as e:
        # Client closed before activation → HTTP 499
        raise HTTPException(
            status_code=499,
            detail={"error": "client_closed_request", "message": str(e)},
        ) from e
    except SidecarUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "sidecar_unavailable", "cause": e.cause, "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "sidecar_timeout", "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarUpstreamError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_sidecar_error",
                "upstream_status": e.upstream_status,
                "upstream_body": e.upstream_body,
                "message": str(e),
            },
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"sidecar failed: {e}") from e
    finally:
        # Tear down the disconnect watcher (ollama_chat).
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watch_task

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="no completion_fn wired - production needs make_llama_server_complete_fn",
        )

    # Adapt OpenAI-shape response → Ollama shape with full tool-call
    # translation, arg JSON coercion, size-cap, id preservation, finish_reason
    # → done_reason mapping, and ISO-8601 created_at.
    # Error guard: if completion_fn surfaced an error dict (no choices), pass it
    # through unchanged so the caller still sees the upstream detail.
    if isinstance(result, dict) and "error" in result:
        return result
    if "choices" in result and result["choices"]:
        choice = result["choices"][0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content", "")
        ollama_msg: dict[str, Any] = {"role": "assistant", "content": content}

        # tool_calls translation: OpenAI emits arguments as a JSON-encoded
        # string; Ollama clients expect a parsed object. Enrich the warning
        # with model + truncated args so we can identify which model is
        # misbehaving in the field.
        openai_tcs = msg.get("tool_calls")
        if openai_tcs:  # only emit the key when there's at least one call
            ollama_tcs = []
            for tc in openai_tcs:
                fn = tc.get("function") or {}
                args_str = fn.get("arguments", "{}")
                if isinstance(args_str, str) and len(args_str) > MAX_TOOL_ARG_CHARS:
                    log.warning(
                        "ollama_chat tool_call args >%dB (got %dB) for model=%r, truncating",
                        MAX_TOOL_ARG_CHARS, len(args_str), model,
                    )
                    args_str = args_str[:MAX_TOOL_ARG_CHARS]
                try:
                    args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError as e:
                    truncated = args_str[:80] if isinstance(args_str, str) else repr(args_str)[:80]
                    log.warning(
                        "ollama_chat: json.loads failed on tool_call args "
                        "for model=%r args[:80]=%r err=%s",
                        model, truncated, e,
                    )
                    args_obj = args_str  # lenient: pass raw string
                tc_entry: dict[str, Any] = {
                    "function": {"name": fn.get("name"), "arguments": args_obj},
                }
                if tc.get("id"):  # preserve id when present
                    tc_entry["id"] = tc["id"]
                ollama_tcs.append(tc_entry)
            ollama_msg["tool_calls"] = ollama_tcs

        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        done_reason_map = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "stop",
            "function_call": "stop",
        }
        done_reason = done_reason_map.get(finish_reason, "stop")

        return {
            "model": model,
            "created_at": _coerce_created_at(result.get("created")),
            "message": ollama_msg,
            "done": True,
            "done_reason": done_reason,
            "thread_id": slot.thread_id,
        }
    # Pass-through if completion_fn returned Ollama-native
    return result


# ============================================================================
# Production completion_fn factory: httpx → llama-server child port
# ============================================================================


def _merge_reasoning_into_content(
    result: Any, response_format: dict | None = None,
) -> None:
    """Merge thinking-model reasoning_content into content.

    Thinking-models (reasoning-style models in general) split output between
    `message.content` (final answer, often empty during thinking) and
    `message.reasoning_content` (the chain-of-thought). Client parsers that read
    only `.content` (many agent harnesses, langchain default, OpenAI SDK) see
    empty and bail → retry storm → no usable output.

    Wrap reasoning_content inline as `<think>...</think>` tags so EVERY
    client sees a non-empty content string. Preserve reasoning_content
    untouched for clients that explicitly read it (Open WebUI, etc.).

    No-op if reasoning_content is empty (non-thinking models) or content
    is already populated alongside reasoning_content (some configs).

    Guards against an empty-response edge case observed under load.

    Skip the merge when the caller requested
    `response_format = {"type": "json_object"}`. The `<think>...</think>`
    wrapper would prepend non-JSON tokens to the content and break the
    contract that response_format clients depend on. The json_schema retry path
    owns the thinking-mode JSON guarantee (blocked upstream on llama.cpp #20345 +
    Ollama #10538); this merge's job is at minimum not to corrupt the path.
    """
    if not isinstance(result, dict):
        return
    # Structured-output skip-branch — see docstring. The skip covers both
    # json_object and json_schema; prepending <think>...</think> would corrupt
    # JSON under either response_format type.
    if (
        isinstance(response_format, dict)
        and response_format.get("type") in ("json_object", "json_schema")
    ):
        return
    choices = result.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        rc = msg.get("reasoning_content") or ""
        ct = msg.get("content") or ""
        if not isinstance(rc, str) or not isinstance(ct, str):
            continue
        rc_stripped = rc.strip()
        if not rc_stripped:
            continue  # no thinking to merge
        # Single-source formatter. Behaviour-preserving extraction —
        # wrap_reasoning_think reproduces this site's exact bytes (rc.strip() + the
        # ct.strip() with/without-content branch), so the client-facing response
        # does not change one byte. The shadow sites now call the SAME helper, so a
        # shadow reconstruction can never drift from what the harness
        # stores/resends.
        msg["content"] = wrap_reasoning_think(rc, ct)


def make_llama_server_complete_fn(
    timeout_s: float = 600.0,
    http_client_factory=None,
    manifests_path: Path | None = None,
):
    """Build a completion_fn that forwards to the active sidecar's port via httpx.

    Used by main.py to wire the production completion_fn. Tests typically inject
    a simpler fake instead.
    """
    async def _complete(slot, handle):
        client_meta = slot.client_meta or {}
        messages = client_meta.get("messages")
        if not messages:
            return None
        model_identity = _resolve_model_identity(
            slot.model_tag, handle.model_id, manifests_path
        )
        payload = {
            "model": model_identity,
            "messages": messages,
            "stream": False,  # streaming SSE is a follow-on polish wave
        }
        # Iterate the canonical _COMMON_FORWARDED_KNOBS list (which already
        # covers tools/tool_choice/parallel_tool_calls/function_call/functions).
        # Previously this loop duplicated an open-coded knob tuple that had
        # drifted from the source of truth — tools knobs were dropped here.
        # Single-list invariant now. Stream-only knobs (`stream`,
        # `stream_options`) are deliberately excluded; this path is non-streaming.
        for k in _COMMON_FORWARDED_KNOBS:
            v = client_meta.get(k)
            if v is not None:
                payload[k] = v
        url = f"http://127.0.0.1:{handle.port}/v1/chat/completions"
        if http_client_factory is not None:
            client_cm = http_client_factory()
        else:
            client_cm = httpx.AsyncClient(timeout=timeout_s)
        try:
            async with client_cm as client:
                r = await client.post(url, json=payload, timeout=timeout_s)
                r.raise_for_status()
                result = r.json()

                # Bounded validate+retry for thinking-mode json_schema callers.
                # ONE retry max with enable_thinking=False overlay (works around
                # llama.cpp #20345 grammar-inactive bug). Retry shares this
                # try/except so httpx errors map to existing
                # SidecarUnavailableError / SidecarUpstreamError /
                # SidecarTimeoutError handlers below — no duplicate classifier.
                rf = client_meta.get("response_format")
                thinking_manifest = client_meta.get("thinking_manifest") or {}
                if (
                    isinstance(rf, dict)
                    and rf.get("type") == "json_schema"
                    and is_thinking_payload(payload, thinking_manifest)
                ):
                    # Lazy import — fail-soft if dep absent.
                    try:
                        from jsonschema import (
                            Draft202012Validator,
                            ValidationError,
                        )
                    except ImportError:
                        Draft202012Validator = None
                        ValidationError = Exception  # type: ignore[assignment,misc]
                    if Draft202012Validator is not None:
                        schema = rf.get("json_schema", {}).get("schema") or {}
                        validate_failed = False
                        try:
                            first_content = _strip_thinking_wrapper(
                                result["choices"][0]["message"].get("content") or ""
                            )
                            Draft202012Validator(schema).validate(
                                json.loads(first_content)
                            )
                        except (
                            json.JSONDecodeError,
                            ValidationError,
                            KeyError,
                            IndexError,
                            TypeError,
                        ):
                            validate_failed = True

                        if validate_failed:
                            # ONE retry: enable_thinking=False overlay.
                            retry_payload = dict(payload)
                            # Validate chat_template_kwargs before forwarding to
                            # the sidecar — only allow known safe keys, strip Jinja
                            # constructs that could SSTI via llama-server.
                            _raw_ctk = payload.get("chat_template_kwargs") or {}
                            _safe_ctk = {
                                "enable_thinking": False,
                            }
                            if isinstance(_raw_ctk, dict):
                                for k, v in _raw_ctk.items():
                                    if not isinstance(k, str):
                                        continue
                                    # Reject any value containing Jinja constructs
                                    if isinstance(v, str) and ("{{" in v or "{%" in v):
                                        continue
                                    # Only allow known safe keys (whitelist)
                                    if k in ("enable_thinking",):
                                        _safe_ctk[k] = v
                            retry_payload["chat_template_kwargs"] = _safe_ctk
                            retry_r = await client.post(
                                url, json=retry_payload, timeout=timeout_s,
                            )
                            retry_r.raise_for_status()
                            retry_result = retry_r.json()
                            try:
                                retry_content = _strip_thinking_wrapper(
                                    retry_result["choices"][0]["message"].get("content") or ""
                                )
                                Draft202012Validator(schema).validate(
                                    json.loads(retry_content)
                                )
                            except (
                                json.JSONDecodeError,
                                ValidationError,
                                KeyError,
                                IndexError,
                                TypeError,
                            ):
                                raise SidecarUpstreamError(
                                    "model_jsonschema_noncompliance_after_retry",
                                    upstream_status=200,
                                    upstream_body=str(retry_result)[:500],
                                )
                            # Retry validated — use retry result.
                            result = retry_result

                # Pass response_format through so the merger can short-circuit
                # for json_object + json_schema callers.
                _merge_reasoning_into_content(
                    result, client_meta.get("response_format"),
                )
                # Recover text-JSON tool calls into structured `tool_calls`.
                # No-op when upstream already populated tool_calls (idempotency)
                # or no tools advertised.
                maybe_recover_tool_calls(result, client_meta.get("tools"))
                return result
        except httpx.HTTPStatusError as e:
            # Sidecar accepted the request but returned 4xx/5xx (context
            # overflow, malformed payload, etc.). Convert to typed error
            # so the route handler can return 502 Bad Gateway with the
            # upstream status preserved.
            raise SidecarUpstreamError(
                f"sidecar returned HTTP {e.response.status_code}",
                upstream_status=e.response.status_code,
                upstream_body=e.response.text,
            ) from e
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
                httpx.ConnectError, httpx.NetworkError, httpx.CloseError,
                httpx.ProtocolError) as e:
            # Sidecar disconnected / crashed / port closed. Most often
            # KV-cache OOM mid-response under stress. Convert to 503 +
            # Retry-After.
            raise SidecarUnavailableError(
                f"sidecar unavailable: {type(e).__name__}: {e}",
                cause="sidecar_disconnected_or_crashed",
                retry_after_s=30,
            ) from e
        except (httpx.TimeoutException,) as e:
            # Includes ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout.
            raise SidecarTimeoutError(
                f"sidecar request timed out after {timeout_s}s: {type(e).__name__}",
                retry_after_s=60,
            ) from e

    return _complete
