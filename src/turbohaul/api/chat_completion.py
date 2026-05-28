"""Chat-completion API routes - Ollama-compat + OpenAI-compat (v0.2 §9).

This module ships the non-streaming completion path. Streaming SSE comes in a
future polish phase; the existing manager.submit_and_wait + completion_fn DI is
streaming-ready (just return an async generator from completion_fn and adapt
the route).

The completion_fn is wired into TurbohaulManager via DI. Production uses
make_llama_server_complete_fn() which httpx-POSTs to the spawned llama-server's
/v1/chat/completions on its assigned port. Tests inject a fake completion_fn
that returns a canned response without spawning anything real.

Typed upstream errors: in practice the live container exhibits RemoteProtocolError (sidecar
OOM-crash during inference) much more often than HTTPStatusError 4xx. These
need different client-facing status codes:
  - 503 Service Unavailable + Retry-After  → sidecar disconnected / crashed
  - 502 Bad Gateway                         → sidecar returned upstream 4xx/5xx
  - 504 Gateway Timeout                     → request timed out at sidecar
  - 500 Internal Server Error               → genuine Turbohaul bug (fallback)
  - 422 RESERVED for input-validation only (NOT used for upstream errors)
"""
import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from turbohaul.config import KEEP_ALIVE_MAX_S  # re-exported for tests + manager
from turbohaul.manifest import read_manifest  # handler-entry manifest read for thinking detect
from turbohaul.slot import SlotEvictedError  # client-disconnect eviction exception
from turbohaul.api.tool_call_recovery import maybe_recover_tool_calls  #


# === client-disconnect watcher =======================
# Constant 2s cadence + direct request.is_disconnected() call per MOD-D /
# FAIL-E + simplification #1. Wrapping is_disconnected in asyncio.wait_for can leak
# the underlying ASGI receive() coroutine on cancellation; Starlette already
# implements is_disconnected as non-blocking-fast, so the wrapper is both
# unnecessary and harmful.
_DISCONNECT_POLL_INTERVAL_S = 2.0


async def watch_disconnect(
    request: Request,
    disconnect_event: asyncio.Event,
) -> None:
    """Poll ``request.is_disconnected()`` every ~2s; signal ``disconnect_event``
    when the client closes the connection.

     MOD-D + simplification #1: constant cadence, direct
    poll, no ``asyncio.wait_for`` wrap. Tolerates transient ASGI receive()
    errors (returns True is the only state we care about).
    """
    while not disconnect_event.is_set():
        try:
            if await request.is_disconnected():
                disconnect_event.set()
                return
        except Exception:
            # Transient ASGI receive() errors are non-fatal; keep polling.
            pass
        await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)


log = logging.getLogger(__name__)
router = APIRouter()


# === JSON Schema validation constants ===================
# Validator-side DoS guards — caller-supplied schemas are size/shape-bounded
# BEFORE we compile via jsonschema. Validated against malformed-schema edge cases.
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

    Obs-1: jsonschema import is LAZY inside the function body — fail-soft
    against the dep being absent during the Step 0a/0b window (writable-layer
    pip-install precedes the image-bake by up to 24h).

    N6: `rf` is the FULL response_format dict; the schema lives at
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
    """ thinking-mode detection (N2 simplification).

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
    payload — Returns the original string when no closing
    tag is present.
    """
    if not isinstance(content, str):
        return content
    if "</think>" in content:
        return content.rsplit("</think>", 1)[-1].lstrip()
    return content


#  (LOW-6): cap on per-tool_call argument string size to
# bound memory + log spam when a model emits runaway args. 256 KiB tolerates
# realistic large payloads (image data URIs, long structured inputs) while
# stopping unbounded growth.
MAX_TOOL_ARG_CHARS = 262144


def _coerce_created_at(created_value: Any) -> Any:
    """  (LOW-7): coerce OpenAI int unix-epoch ``created``
    into ISO-8601 string for Ollama-compat ``created_at`` field. None and
    pre-formatted strings pass through unchanged.
    """
    if isinstance(created_value, int):
        return datetime.fromtimestamp(created_value, tz=timezone.utc).isoformat()
    return created_value


# === Ollama-style keep_alive parser ==============================

_KEEP_ALIVE_UNITS = {"s": 1, "m": 60, "h": 3600}


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
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            pass
        if (
            len(s) >= 2
            and s[-1].lower() in _KEEP_ALIVE_UNITS
            and s[:-1].lstrip("-").isdigit()
        ):
            return int(s[:-1]) * _KEEP_ALIVE_UNITS[s[-1].lower()]
    return None


# ===  SSE tuning constants (module-level for monkeypatch-in-tests) ===

# How long to wait for the slot to actually reach ACTIVE before we give up.
# Cold-load of a 27B GGUF can take 30-60s; pre-stream wait should be much
# longer than that since the route is held open.
SLOT_READY_TIMEOUT_S = 600.0
# httpx.stream timeout for the actual sidecar connection — keep generous for
# slow-thinking models on large contexts.
STREAM_TIMEOUT_S = 3600.0
# emit `: keep-alive\n\n` SSE comments at this cadence while waiting
# for `slot.stream_ready_event` to fire. Many clients set 30-60s read-timeouts
# on streaming responses; without intermittent bytes the client disconnects
# during cold-load (a 27B GGUF takes 30-60s to load). SSE comments are RFC
# 8895 / EventSource-compliant; clients silently consume them and the
# connection stays warm.
HEARTBEAT_INTERVAL_S = 12.0


# ===  typed upstream errors ===

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
    mgr = request.app.state.manager
    model = payload.get("model")
    messages = payload.get("messages")
    if not model:
        raise HTTPException(status_code=400, detail="`model` field required")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")
    # Best-effort prompt extraction for thread-id derivation
    prompt = " ".join(
        m.get("content", "") for m in messages if isinstance(m, dict)
    )
    thread_id = payload.get("thread_id") or ""
    # response_format pre-validation. Fires for BOTH stream
    # + non-stream so SSE clients cannot bypass via the wants_stream fork.
    # Strict shape: {type:"json_object"} accepted, {type:"text"} normalized to
    # None (OpenAI default — no-op pass-through), anything else REJECTED 400.
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
            pass  #  accept-and-forward — no validation; manifest decides if model honors
        elif rf_type == "json_schema":
            # validate caller schema; on bad → 422 schema_validation_failed.
            # On ok, response_format propagates via the existing  _COMMON_FORWARDED_KNOBS
            # tuple (no separate forwarding code path needed).
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

    # SSE streaming pass-through:
    # when the client sends stream=true, branch to the streaming helper which
    # opens its own httpx.stream() to the sidecar and yields SSE chunks back
    # to the client. The non-streaming path below is unchanged.
    if wants_stream:
        return await _openai_chat_completions_stream(
            request, mgr, model, messages, prompt, thread_id, payload,
        )

    # when caller requested json_schema, read the manifest
    # to slice out reasoning_budget for the in-_complete retry-path gate. Only
    # fires for json_schema requests (F5 narrow scope — text/json_object/
    # no-response_format never need this read). Read failure is non-fatal:
    # thinking_manifest stays empty + retry path is disabled, primary forward
    # still proceeds via _COMMON_FORWARDED_KNOBS.
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
                ": manifest read failed for model=%s; retry-path disabled",
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
        # Ollama-style keep_alive → IDLE_HOT extension hint (advisor Option E)
        "keep_alive_s": parse_keep_alive(payload.get("keep_alive")),
        #  Site A: forward validated response_format to
        # llama-server via _complete's _COMMON_FORWARDED_KNOBS loop. Skipping
        # this line would silently drop the field on the non-stream path
        # despite the validator accepting it.
        "response_format": payload.get("response_format"),
        # manifest reasoning_budget slice for in-_complete
        # is_thinking_payload check. {} ⇒ retry disabled (non-thinking or read-fail).
        "thinking_manifest": thinking_manifest,
        # closure-fix: forward tools-family knobs into client_meta so
        # the _complete closure includes them in the llama-server payload AND
        # so maybe_recover_tool_calls can read the advertised tools allowlist.
        # Mirrors the /api/chat endpoint client_meta build ( Change 4 /
        #) — the OpenAI endpoint silently dropped tools on the floor.
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
        "parallel_tool_calls": payload.get("parallel_tool_calls"),
        "function_call": payload.get("function_call"),
        "functions": payload.get("functions"),
    }

    # client-disconnect watcher. Event constructed
    # IN-HANDLER so it's bound to the route's request-loop (MOD-A correct-loop
    # guarantee). watch_disconnect polls request.is_disconnected() every 2s; if
    # client closes, sets the Event which queue.pop_next sees and evicts the slot.
    disconnect_event = asyncio.Event()
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))
    try:
        slot, result = await mgr.submit_and_wait(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            client_meta=client_meta,
            disconnect_event=disconnect_event,
        )
    except SlotEvictedError as e:
        # Bonus-1: client closed connection before slot activated; surface as
        # HTTP 499 (client_closed_request) so monitoring distinguishes client-side
        # close from a real 500 server fault.
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
        # tear down the disconnect watcher cleanly.
        # contextlib.suppress so a normal cancellation doesn't surface from finally.
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
# streaming helper
# ============================================================================


# Knobs forwarded to llama-server from client_meta.
# (note-1 + MED-5 + MED-1): split into _COMMON (forwarded everywhere)
# and _STREAM_ONLY (only included in the streaming payload). `_complete`
# iterates _COMMON only so non-streaming requests never inherit stream-only
# keys; the streaming payload-build helper uses the derived
# _STREAM_FORWARDED_KNOBS alias. Drift between the two lists is now
# structurally impossible.
_COMMON_FORWARDED_KNOBS = (
    # Core OpenAI-compat
    "temperature", "top_p", "top_k", "max_tokens", "min_p",
    #  — accept-and-forward only; handler-entry validator
    # rejects json_schema as deferred-to-. Thinking-mode JSON guarantee
    # blocked upstream on llama.cpp #20345 + Ollama #10538.
    "response_format",
    # an agent preserved-thinking controls
    "thinking_budget_tokens", "reasoning_budget", "reasoning",
    # Ollama-parity samplers
    "presence_penalty", "frequency_penalty", "repeat_penalty",
    "repeat_last_n", "typical_p", "seed",
    "mirostat", "mirostat_lr", "mirostat_ent",
    # max-output alias
    "n_predict",
    # tool-call pass-through (minimal — full capability
    # advertisement + size-cap + per-model gating deferred to /4b-proper
    # per advisor's 5-hazard analysis; this is "forward the field to a model
    # that supports tool_calls natively, e.g. Qwen3.6-27b-dense"). llama-server
    # mirrors OpenAI's schema, so structured values (list/dict/string) just
    # pass through unchanged.
    "tools", "tool_choice", "parallel_tool_calls",
    "function_call", "functions",
)
_STREAM_ONLY_KNOBS = ("stream", "stream_options")
# Back-compat alias — `_build_stream_payload` reads this for the streaming
# llama-server call where `stream=True` must be forwarded.
_STREAM_FORWARDED_KNOBS = _COMMON_FORWARDED_KNOBS + _STREAM_ONLY_KNOBS


def _build_stream_payload(client_meta: dict, model: str, messages: list) -> dict:
    """Build the streaming chat-completions payload sent to llama-server."""
    payload: dict = {
        "model": model,
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
    stream lifetime — single-slot invariant preserved by design). Awaits ``slot.stream_ready_event`` so we know the sidecar
    is up and ``slot.stream_handle`` is populated. Then opens our own
    ``httpx.stream("POST", url, ...)`` to the sidecar and pipes raw SSE bytes
    back to the client. On end-of-stream / disconnect / error sets
    ``slot.stream_done_event`` so the manager can advance ACTIVE → GRACE.

    Wrapper ``_merge_reasoning_into_content`` is intentionally SKIPPED on the
    streaming path: most modern streaming consumers (an agent, langchain, Open
    WebUI, OpenAI SDK) parse ``delta.content`` and ``delta.reasoning_content``
    independently. Per-chunk merge would require accumulator/reorder state
    and would break the token-by-token UX.
    """
    client_meta = {
        "kind": "openai-chat-completion-stream",
        "messages": messages,
        "model": model,
        "stream": True,
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "max_tokens": payload.get("max_tokens"),
        # Ollama-style keep_alive → IDLE_HOT extension hint (advisor Option E).
        # Streaming path is agent-class agents' primary entry; this fix is the
        # whole point of the wave.
        "keep_alive_s": parse_keep_alive(payload.get("keep_alive")),
        # All forwardable knobs carried for the streaming payload helper.
        **{k: payload.get(k) for k in _STREAM_FORWARDED_KNOBS if payload.get(k) is not None},
    }

    # Pre-stream submission errors → standard HTTPException with proper status code.
    try:
        slot = await mgr.submit_for_streaming(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            client_meta=client_meta,
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

    async def stream_gen():
        try:
            # Wait for worker_loop to bring slot to ACTIVE + assign handle.
            # emit `: keep-alive\n\n` SSE comments every
            # HEARTBEAT_INTERVAL_S so clients with 30-60s read-timeouts don't
            # disconnect during cold-load. asyncio.shield prevents the
            # heartbeat wait_for from cancelling the underlying ready_task.
            ready_task = asyncio.create_task(slot.stream_ready_event.wait())
            loop = asyncio.get_event_loop()
            deadline = loop.time() + SLOT_READY_TIMEOUT_S
            while not ready_task.done():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    ready_task.cancel()
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

            stream_payload = _build_stream_payload(client_meta, model, messages)
            url = f"http://127.0.0.1:{handle.port}/v1/chat/completions"

            async with httpx.AsyncClient(timeout=STREAM_TIMEOUT_S) as client:
                async with client.stream(
                    "POST", url, json=stream_payload, timeout=STREAM_TIMEOUT_S,
                ) as r:
                    # raise_for_status is NOT auto-called by httpx.stream;
                    # the body may not be loaded yet so we read it manually if
                    # the upstream returned a 4xx/5xx.
                    if r.status_code >= 400:
                        body_bytes = await r.aread()
                        body_str = body_bytes.decode("utf-8", errors="replace")[:500]
                        yield _stream_error_frame(
                            "upstream_sidecar_error",
                            f"sidecar returned HTTP {r.status_code}",
                            upstream_status=r.status_code,
                            upstream_body=body_str,
                        )
                        yield b"data: [DONE]\n\n"
                        return

                    # Pipe raw SSE bytes from llama-server straight through to
                    # the client. llama-server already emits
                    # ``data: {...}\n\ndata: [DONE]\n\n`` shaped chunks.
                    async for chunk_bytes in r.aiter_bytes():
                        if chunk_bytes:
                            yield chunk_bytes
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
            # Signal worker_loop to advance the slot ACTIVE → GRACE.
            # Idempotent: setting an already-set Event is a no-op.
            if slot.stream_done_event is not None and not slot.stream_done_event.is_set():
                slot.stream_done_event.set()

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable buffering at nginx / other reverse proxies so chunks
            # reach the client in real time (caught in review).
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
    mgr = request.app.state.manager
    model = payload.get("model")
    messages = payload.get("messages")
    if not model or not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`model` + `messages` required")
    prompt = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
    thread_id = payload.get("thread_id") or ""
    # Ollama accepts keep_alive at top level OR nested under options.
    ka_raw = payload.get("keep_alive")
    if ka_raw is None and isinstance(payload.get("options"), dict):
        ka_raw = payload["options"].get("keep_alive")
    # response_format pre-validation. Fires for BOTH stream
    # + non-stream and BEFORE the streaming-tools guard below so the deferred
    # type ('json_schema') gets a clean 400 regardless of the stream flag.
    # Mirror of openai_chat_completions validator; minor wording tweak in the
    # detail for endpoint clarity.
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
            pass  #  accept-and-forward
        elif rf_type == "json_schema":
            # mirror of openai_chat_completions json_schema branch.
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
    #  Change 4 (HIGH-1): streaming + tools is deferred
    # to follow-on . Cheap defensive guard before submit_and_wait so
    # callers get a clean 400 instead of a confusing partial-tool stream.
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
                    "Ollama-shape streaming + tool_calls deferred to follow-on RC. "
                    "Use stream=false for tool requests or /v1/chat/completions "
                    "for OpenAI-shape streaming-tools."
                ),
                "follow_on": "streaming-tools",
            },
        )
    # mirror of Site A manifest read — slice reasoning_budget
    # for in-_complete retry-path gate. Only fires for json_schema requests.
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
                ": manifest read failed for model=%s (ollama_chat); retry-path disabled",
                model,
            )
    client_meta = {
        "kind": "ollama-chat",
        "messages": messages,
        "model": model,
        "stream": bool(payload.get("stream", False)),
        "options": payload.get("options"),
        "keep_alive_s": parse_keep_alive(ka_raw),
        #  Change 3 (BUG-1 fix): forward Ollama-shape tool
        # knobs into client_meta so _complete can pass them through to
        # llama-server. Without this, ollama_chat tool requests silently
        # dropped tools on the floor before reaching the sidecar.
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
        "parallel_tool_calls": payload.get("parallel_tool_calls"),
        "function_call": payload.get("function_call"),
        "functions": payload.get("functions"),
        #  Site C: forward validated response_format. Same
        # rationale as Site A — handler-entry validator alone does not propagate
        # the field; the explicit add line keeps the non-stream ollama path
        # honest.
        "response_format": payload.get("response_format"),
        # manifest reasoning_budget slice for in-_complete
        # is_thinking_payload gate. Mirrors Site A; {} ⇒ retry disabled.
        "thinking_manifest": thinking_manifest,
    }
    # client-disconnect watcher (ollama_chat mirror of Site A).
    disconnect_event = asyncio.Event()
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))
    try:
        slot, result = await mgr.submit_and_wait(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            client_meta=client_meta,
            disconnect_event=disconnect_event,
        )
    except SlotEvictedError as e:
        # Bonus-1: client closed before activation → HTTP 499
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
        # tear down disconnect watcher (ollama_chat).
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watch_task

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="no completion_fn wired - production needs make_llama_server_complete_fn",
        )

    #  Change 5: adapt OpenAI-shape response → Ollama shape
    # with full tool-call translation (BUG-3 fix), arg JSON coercion (BUG-4 /
    # note-3), size-cap (LOW-6), id preservation (MED-3), finish_reason
    # → done_reason mapping (MED-4), and ISO-8601 created_at (LOW-7).
    # note-2 error guard: if completion_fn surfaced an error dict (no choices),
    # pass it through unchanged so the caller still sees the upstream detail.
    if isinstance(result, dict) and "error" in result:
        return result
    if "choices" in result:
        choice = result["choices"][0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content", "")
        ollama_msg: dict[str, Any] = {"role": "assistant", "content": content}

        # tool_calls translation: OpenAI emits arguments as a JSON-encoded
        # string; Ollama clients expect a parsed object. LOW-5: enrich
        # the warning with model + truncated args so we can identify which
        # model is misbehaving in the field.
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
                    args_obj = args_str  # note-3 lenient: pass raw string
                tc_entry: dict[str, Any] = {
                    "function": {"name": fn.get("name"), "arguments": args_obj},
                }
                if tc.get("id"):  # MED-3: preserve id when present
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
    """: merge thinking-model reasoning_content into content.

    Thinking-models (Qwen3, deepseek-r1, Gemma-thinking, etc.) split output
    between `message.content` (final answer, often empty during thinking)
    and `message.reasoning_content` (the chain-of-thought). Client parsers
    that read only `.content` (agent-class workers, langchain default,
    OpenAI SDK) see empty and bail → retry storm → no usable output.

    Wrap reasoning_content inline as `<think>...</think>` tags so EVERY
    client sees a non-empty content string. Preserve reasoning_content
    untouched for clients that explicitly read it (Open WebUI, etc.).

    No-op if reasoning_content is empty (non-thinking models) or content
    is already populated alongside reasoning_content (some configs).

    Handles an observed empty-response edge case.

    : skip merge when the caller requested
    `response_format = {"type": "json_object"}`. The `<think>...</think>`
    wrapper would prepend non-JSON tokens to the content and break the
    contract that response_format clients depend on. No upstream component yet
    owns the thinking-mode JSON guarantee (blocked on llama.cpp #20345 +
    Ollama #10538); Turbohaul's job is at minimum not to corrupt the path.
    """
    if not isinstance(result, dict):
        return
    #  + : structured-output skip-branch —
    # see docstring.  widens the skip from json_object alone to also
    # cover json_schema; prepending <think>...</think> would corrupt JSON
    # under either response_format type.
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
        if ct.strip():
            # Final answer already populated — prepend thinking as context
            msg["content"] = f"<think>\n{rc_stripped}\n</think>\n\n{ct}"
        else:
            # Final answer empty — surface the thinking so client sees something
            msg["content"] = f"<think>\n{rc_stripped}\n</think>"


def make_llama_server_complete_fn(
    timeout_s: float = 600.0,
    http_client_factory=None,
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
        payload = {
            "model": slot.model_tag,
            "messages": messages,
            "stream": False,  # streaming SSE is a follow-on polish wave
        }
        #  Change 1 (BUG-2 fix): iterate the canonical
        # _COMMON_FORWARDED_KNOBS list (which already covers tools/tool_choice/
        # parallel_tool_calls/function_call/functions). Previously this loop
        # duplicated an open-coded knob tuple that had drifted from the source
        # of truth — tools knobs were dropped here. Single-list invariant now.
        # Stream-only knobs (`stream`, `stream_options`) are deliberately
        # excluded; this path is non-streaming.
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

                # bounded validate+retry for thinking-mode
                # json_schema callers. ONE retry max with enable_thinking=False
                # overlay (works around llama.cpp #20345 grammar-inactive bug).
                # Retry shares this try/except so httpx errors map to existing
                # SidecarUnavailableError / SidecarUpstreamError /
                # SidecarTimeoutError handlers below — no duplicate classifier.
                rf = client_meta.get("response_format")
                thinking_manifest = client_meta.get("thinking_manifest") or {}
                if (
                    isinstance(rf, dict)
                    and rf.get("type") == "json_schema"
                    and is_thinking_payload(payload, thinking_manifest)
                ):
                    # Obs-1 lazy import — fail-soft if dep absent.
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
                            retry_payload["chat_template_kwargs"] = {
                                **(payload.get("chat_template_kwargs") or {}),
                                "enable_thinking": False,
                            }
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

                # pass response_format through so the
                # merger can short-circuit for json_object + json_schema callers.
                _merge_reasoning_into_content(
                    result, client_meta.get("response_format"),
                )
                # recover Qwen-class text-JSON tool calls into
                # structured `tool_calls`. No-op when upstream already
                # populated tool_calls (idempotency) or no tools advertised.
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
            # KV-cache OOM mid-response observed under KV-cache pressure.
            # Convert to 503 + Retry-After.
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
