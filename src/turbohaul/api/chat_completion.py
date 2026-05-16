"""Chat-completion API routes - Ollama-compat + OpenAI-compat (v0.2 §9).

Phase 3 Wave 12 ships the non-streaming completion path. Streaming SSE comes in a
future polish wave; the existing manager.submit_and_wait + completion_fn DI is
streaming-ready (just return an async generator from completion_fn and adapt
the route).

The completion_fn is wired into TurbohaulManager via DI. Production uses
make_llama_server_complete_fn() which httpx-POSTs to the spawned llama-server's
/v1/chat/completions on its assigned port. Tests inject a fake completion_fn
that returns a canned response without spawning anything real.
"""
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request


log = logging.getLogger(__name__)
router = APIRouter()


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
    except RuntimeError as e:
        # loading-fail / worker exception → 500
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
        # Pass-through tuning knobs if present
        for k in ("temperature", "top_p", "top_k", "max_tokens", "min_p"):
            v = client_meta.get(k)
            if v is not None:
                payload[k] = v
        url = f"http://127.0.0.1:{handle.port}/v1/chat/completions"
        if http_client_factory is not None:
            client_cm = http_client_factory()
        else:
            client_cm = httpx.AsyncClient(timeout=timeout_s)
        async with client_cm as client:
            r = await client.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            return r.json()

    return _complete
