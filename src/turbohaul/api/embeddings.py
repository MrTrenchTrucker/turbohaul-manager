"""POST /v1/embeddings — OpenAI-compat embeddings forwarder.

Embeddings passthrough endpoint.

Route owns its own slot lease via manager.submit_for_streaming (mirrors 
SSE chat pattern — route-owned upstream dispatch, manager skips _complete_fn).
Validation order: Content-Length gate → pydantic parse → manifest capability
pre-flight → encoding_format/dimensions guards → batch cap → slot acquire →
sidecar /v1/embeddings forward → upstream JSON pass-through.

Auth: NO app-layer auth — perimeter-trust model per ARCHITECTURE.md §11.3.
"""
import asyncio
import contextlib
import logging
from typing import Union

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from turbohaul.api.chat_completion import watch_disconnect  #  shared helper
from turbohaul.manifest import read_manifest
from turbohaul.slot import SlotEvictedError  # client-disconnect eviction exception


log = logging.getLogger(__name__)
router = APIRouter()


# Constants
_MAX_REQUEST_BYTES = 32_768 * 64  # ~2MB Content-Length ceiling
_BATCH_CAP = 64  # 96→64 power-of-2
_UPSTREAM_TIMEOUT_S = 120.0
_SLOT_READY_TIMEOUT_S = 600.0

# Module-scope httpx client (NO factory pattern)
_HTTPX_CLIENT: httpx.AsyncClient | None = None


def _get_httpx_client() -> httpx.AsyncClient:
    global _HTTPX_CLIENT
    if _HTTPX_CLIENT is None or _HTTPX_CLIENT.is_closed:
        _HTTPX_CLIENT = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT_S)
    return _HTTPX_CLIENT


async def _content_length_gate(request: Request) -> None:
    """Reject HTTP 413 before pydantic parse if Content-Length > ceiling (F3)."""
    cl = request.headers.get("content-length")
    if cl is None:
        return  # chunked encoding falls through to batch-cap + per-item limits
    try:
        n = int(cl)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid Content-Length header")
    if n > _MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"request body {n} bytes exceeds {_MAX_REQUEST_BYTES} ceiling",
        )


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    input: Union[str, list[str]]
    encoding_format: str | None = None
    dimensions: int | None = None


@router.post("/v1/embeddings", dependencies=[Depends(_content_length_gate)])
async def post_embeddings(req: EmbeddingsRequest, request: Request) -> dict:
    """POST /v1/embeddings — forward to model's llama-server /v1/embeddings.

    Validation order: Content-Length (dep) → pydantic (auto) → manifest
    capability → encoding_format/dimensions → batch cap → slot acquire.
    """
    mgr = request.app.state.manager

    # Gate 3: encoding_format=base64 → 400 plain-string (Q-4 lock)
    if req.encoding_format is not None and req.encoding_format.lower() == "base64":
        raise HTTPException(
            status_code=400,
            detail="encoding_format='base64' not supported; use 'float'",
        )

    # Gate 4: dimensions param present → 400 plain-string (Q-5 lock)
    if req.dimensions is not None:
        raise HTTPException(
            status_code=400,
            detail="dimensions param not supported; embedding dim is model-defined",
        )

    # Gate 5: batch cap
    inputs: list[str] = [req.input] if isinstance(req.input, str) else list(req.input)
    if len(inputs) > _BATCH_CAP:
        raise HTTPException(
            status_code=413,
            detail=f"input batch size {len(inputs)} exceeds {_BATCH_CAP}-item cap",
        )

    # Gate 2: manifest capability pre-flight (embeddings.py owns this)
    try:
        manifest = read_manifest(mgr.boot.storage.manifests_path, req.model)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"model '{req.model}' has no manifest"
        )
    if not manifest.llama_server_flags.get("embeddings"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"model '{req.model}' does not expose embeddings; "
                "manifest.llama_server_flags.embeddings is false"
            ),
        )

    # Slot acquire via streaming primitive (route owns upstream dispatch)
    client_meta = {
        "kind": "openai-embeddings",
        "model": req.model,
        "stream": True,  # reuses the route-owned-dispatch plumbing
        "keep_alive_s": 0,  # RAG batches do NOT need keep_alive carry-over
    }
    thread_hint = inputs[0][:256] if inputs else ""
    # client-disconnect watcher for embeddings.
    disconnect_event = asyncio.Event()
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))
    slot = None  # bound by submit_for_streaming below; finally guards on None
    try:
        try:
            slot = await mgr.submit_for_streaming(
                model_tag=req.model,
                prompt=thread_hint,
                thread_id="",
                client_meta=client_meta,
                disconnect_event=disconnect_event,
            )
        except SlotEvictedError as e:
            # Bonus-1: client closed before activation → HTTP 499
            raise HTTPException(
                status_code=499,
                detail={"error": "client_closed_request", "message": str(e)},
            ) from e
        except RuntimeError as e:
            raise HTTPException(
                status_code=503,
                detail=f"sidecar unavailable: {e}",
                headers={"Retry-After": "5"},
            ) from e

        try:
            await asyncio.wait_for(
                slot.stream_ready_event.wait(),
                timeout=_SLOT_READY_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"slot did not reach ACTIVE within {_SLOT_READY_TIMEOUT_S}s",
            )

        handle = slot.stream_handle
        if handle is None:
            raise HTTPException(
                status_code=500,
                detail="slot reached ACTIVE but stream_handle is None",
            )

        upstream_url = f"http://127.0.0.1:{handle.port}/v1/embeddings"
        upstream_payload = {"model": req.model, "input": req.input}
        client = _get_httpx_client()
        try:
            r = await client.post(
                upstream_url, json=upstream_payload, timeout=_UPSTREAM_TIMEOUT_S,
            )
        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=504, detail=f"upstream embeddings timeout: {e}",
            ) from e
        except (httpx.NetworkError, httpx.ProtocolError) as e:
            raise HTTPException(
                status_code=502, detail=f"upstream transport error: {e}",
            ) from e
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"upstream HTTP {r.status_code}: {r.text[:500]}",
            )
        return r.json()
    finally:
        # tear down disconnect watcher cleanly.
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watch_task
        # Guard: slot is None if submit_for_streaming raised pre-acquire
        # (SlotEvictedError, RuntimeError) — no stream_done_event to flip.
        if (
            slot is not None
            and slot.stream_done_event is not None
            and not slot.stream_done_event.is_set()
        ):
            slot.stream_done_event.set()
