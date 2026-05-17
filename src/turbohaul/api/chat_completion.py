"""Chat-completion API routes - Ollama-compat + OpenAI-compat (v0.2 §9).

Phase 3 Wave 12 ships the non-streaming completion path. Streaming SSE comes in a
future polish wave; the existing manager.submit_and_wait + completion_fn DI is
streaming-ready (just return an async generator from completion_fn and adapt
the route).

The completion_fn is wired into TurbohaulManager via DI. Production uses
make_llama_server_complete_fn() which httpx-POSTs to the spawned llama-server's
/v1/chat/completions on its assigned port. Tests inject a fake completion_fn
that returns a canned response without spawning anything real.

Wave 1.5-D (DAG synthesis 2026-05-17): typed upstream errors. Per Devil's
Advocate verdict, the live container exhibits RemoteProtocolError (sidecar
OOM-crash during inference) much more often than HTTPStatusError 4xx. These
need different client-facing status codes:
  - 503 Service Unavailable + Retry-After  → sidecar disconnected / crashed
  - 502 Bad Gateway                         → sidecar returned upstream 4xx/5xx
  - 504 Gateway Timeout                     → request timed out at sidecar
  - 500 Internal Server Error               → genuine Turbohaul bug (fallback)
  - 422 RESERVED for input-validation only (NOT used for upstream errors)
"""
import asyncio
import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse


log = logging.getLogger(__name__)
router = APIRouter()


# === Wave 3.1 SSE tuning constants (module-level for monkeypatch-in-tests) ===

# How long to wait for the slot to actually reach ACTIVE before we give up.
# Cold-load of a 27B GGUF can take 30-60s; pre-stream wait should be much
# longer than that since the route is held open.
SLOT_READY_TIMEOUT_S = 600.0
# httpx.stream timeout for the actual sidecar connection — keep generous for
# slow-thinking models on large contexts.
STREAM_TIMEOUT_S = 3600.0
# Wave 3.1: emit `: keep-alive\n\n` SSE comments at this cadence while waiting
# for `slot.stream_ready_event` to fire. Many clients set 30-60s read-timeouts
# on streaming responses; without intermittent bytes the client disconnects
# during cold-load (a 27B GGUF takes 30-60s to load). SSE comments are RFC
# 8895 / EventSource-compliant; clients silently consume them and the
# connection stays warm.
HEARTBEAT_INTERVAL_S = 12.0


# === Wave 1.5-D typed upstream errors ===

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
    pass-through generator for streaming requests (Wave 3, Cmdr 2026-05-17
    16:48Z).

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
    wants_stream = bool(payload.get("stream", False))

    # Wave 3 SSE streaming pass-through (Cmdr 2026-05-17 16:48Z directive):
    # when the client sends stream=true, branch to the streaming helper which
    # opens its own httpx.stream() to the sidecar and yields SSE chunks back
    # to the client. The non-streaming path below is unchanged.
    if wants_stream:
        return await _openai_chat_completions_stream(
            request, mgr, model, messages, prompt, thread_id, payload,
        )

    client_meta = {
        "kind": "openai-chat-completion",
        "messages": messages,  # carried for the completion_fn to forward; redacted from /ws/state
        "model": model,
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "stream": False,
        "max_tokens": payload.get("max_tokens"),
    }

    try:
        slot, result = await mgr.submit_and_wait(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            client_meta=client_meta,
        )
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

    if result is None:
        # Default completion_fn (no real backend wired) — return an empty echo
        raise HTTPException(
            status_code=503,
            detail="no completion_fn wired - production needs make_llama_server_complete_fn",
        )
    return result


# ============================================================================
# Wave 3 streaming helper (Cmdr 2026-05-17 16:48Z)
# ============================================================================


# Knobs forwarded to llama-server from client_meta (kept in sync with
# make_llama_server_complete_fn._complete). The streaming payload-build helper
# mirrors the non-streaming forwarder so reasoning/sampling parity holds.
_STREAM_FORWARDED_KNOBS = (
    # Core OpenAI-compat
    "temperature", "top_p", "top_k", "max_tokens", "min_p",
    # Wave 2.2: Hermes preserved-thinking controls
    "thinking_budget_tokens", "reasoning_budget", "reasoning",
    # Wave 2.2: Ollama-parity samplers
    "presence_penalty", "frequency_penalty", "repeat_penalty",
    "repeat_last_n", "typical_p", "seed",
    "mirostat", "mirostat_lr", "mirostat_ent",
    # Wave 2.2: max-output alias
    "n_predict",
    # Wave 4b-light: tool-call pass-through (minimal — full capability
    # advertisement + size-cap + per-model gating deferred to Wave 4a/4b-proper
    # per advisor's 5-hazard analysis; this is "forward the field to a model
    # that supports tool_calls natively, e.g. Qwen3.6-27b-dense"). llama-server
    # mirrors OpenAI's schema, so structured values (list/dict/string) just
    # pass through unchanged.
    "tools", "tool_choice", "parallel_tool_calls",
    "function_call", "functions",
)


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
    """Wave 3 SSE streaming pass-through (Cmdr 2026-05-17 16:48Z).

    Submits via ``manager.submit_for_streaming`` (slot held ACTIVE for full
    stream lifetime — single-slot invariant preserved per Failure Predictor
    #16 verdict). Awaits ``slot.stream_ready_event`` so we know the sidecar
    is up and ``slot.stream_handle`` is populated. Then opens our own
    ``httpx.stream("POST", url, ...)`` to the sidecar and pipes raw SSE bytes
    back to the client. On end-of-stream / disconnect / error sets
    ``slot.stream_done_event`` so the manager can advance ACTIVE → GRACE.

    Wrapper ``_merge_reasoning_into_content`` is intentionally SKIPPED on the
    streaming path: most modern streaming consumers (Hermes, langchain, Open
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
            # Wave 3.1: emit `: keep-alive\n\n` SSE comments every
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
            # reach the client in real time (Failure Predictor #16 catch).
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
    client_meta = {
        "kind": "ollama-chat",
        "messages": messages,
        "model": model,
        "stream": bool(payload.get("stream", False)),
        "options": payload.get("options"),
    }
    try:
        slot, result = await mgr.submit_and_wait(
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

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="no completion_fn wired - production needs make_llama_server_complete_fn",
        )

    # Adapt OpenAI-shape response → Ollama shape
    if "choices" in result:
        choice = result["choices"][0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content", "")
        return {
            "model": model,
            "created_at": result.get("created"),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "thread_id": slot.thread_id,
        }
    # Pass-through if completion_fn returned Ollama-native
    return result


# ============================================================================
# Production completion_fn factory: httpx → llama-server child port
# ============================================================================


def _merge_reasoning_into_content(result: Any) -> None:
    """Wave 2.1: merge thinking-model reasoning_content into content.

    Thinking-models (Qwen3, deepseek-r1, Gemma-thinking, etc.) split output
    between `message.content` (final answer, often empty during thinking)
    and `message.reasoning_content` (the chain-of-thought). Client parsers
    that read only `.content` (Hermes-class workers, langchain default,
    OpenAI SDK) see empty and bail → retry storm → no usable output.

    Wrap reasoning_content inline as `<think>...</think>` tags so EVERY
    client sees a non-empty content string. Preserve reasoning_content
    untouched for clients that explicitly read it (Open WebUI, etc.).

    No-op if reasoning_content is empty (non-thinking models) or content
    is already populated alongside reasoning_content (some configs).

    Per RELAY DAG finding 2026-05-17 + Cmdr empty-response observation.
    """
    if not isinstance(result, dict):
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
        # Pass-through tuning knobs if present.
        # Wave 2.2: extended the forwarded list to include reasoning-budget knobs
        # (Hermes-class clients control thinking depth per-request) + Ollama-parity
        # sampling knobs (presence/frequency/seed/repeat_penalty + mirostat trio).
        for k in (
            # Core OpenAI-compat
            "temperature", "top_p", "top_k", "max_tokens", "min_p",
            # Wave 2.2: Hermes preserved-thinking controls
            "thinking_budget_tokens", "reasoning_budget", "reasoning",
            # Wave 2.2: Ollama-parity samplers (clients sending these get them honored)
            "presence_penalty", "frequency_penalty", "repeat_penalty",
            "repeat_last_n", "typical_p", "seed",
            "mirostat", "mirostat_lr", "mirostat_ent",
            # Wave 2.2: alias for max output budget (some clients send n_predict)
            "n_predict",
        ):
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
                _merge_reasoning_into_content(result)
                return result
        except httpx.HTTPStatusError as e:
            # Sidecar accepted the request but returned 4xx/5xx (context
            # overflow, malformed payload, etc.). Convert to typed error
            # so the route handler can return 502 Bad Gateway with the
            # upstream status preserved. (Wave 1.5-D DAG synthesis.)
            raise SidecarUpstreamError(
                f"sidecar returned HTTP {e.response.status_code}",
                upstream_status=e.response.status_code,
                upstream_body=e.response.text,
            ) from e
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
                httpx.ConnectError, httpx.NetworkError, httpx.CloseError,
                httpx.ProtocolError) as e:
            # Sidecar disconnected / crashed / port closed. Most often
            # KV-cache OOM mid-response per RELAY's stress observations.
            # Convert to 503 + Retry-After. (Wave 1.5-D DAG synthesis.)
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
