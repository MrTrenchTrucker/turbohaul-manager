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
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request


log = logging.getLogger(__name__)
router = APIRouter()


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
async def openai_chat_completions(payload: dict, request: Request) -> dict:
    """OpenAI-shape chat completion. Forwarded through manager.submit_and_wait."""
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
    client_meta = {
        "kind": "openai-chat-completion",
        "messages": messages,  # carried for the completion_fn to forward; redacted from /ws/state
        "model": model,
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "stream": bool(payload.get("stream", False)),
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
