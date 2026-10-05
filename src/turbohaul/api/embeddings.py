"""POST /v1/embeddings — OpenAI-compat embeddings forwarder.

Request limits (body size, batch cap) are enforced before any slot is leased.

Route owns its own slot lease via manager.submit_for_streaming (mirrors the
SSE chat pattern — route-owned upstream dispatch, manager skips _complete_fn).
Validation order: Content-Length gate → pydantic parse → manifest capability
pre-flight → encoding_format/dimensions guards → batch cap → slot acquire →
sidecar /v1/embeddings forward → upstream JSON pass-through.

Auth: NO app-layer auth — perimeter-trust model per ARCHITECTURE.md §8.
"""
import asyncio
import contextlib
import logging
from typing import Union

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from turbohaul.api.chat_completion import (  # shared helpers for error bodies and disconnect watching
    SidecarUnavailableError,
    capacity_unavailable_error_body,
    sidecar_unavailable_error_body,
    watch_disconnect,
)
from turbohaul.manifest import ManifestValidationError, read_manifest
from turbohaul.slot import SlotEvictedError, VramOverCommitError  # shared slot exceptions


log = logging.getLogger(__name__)
router = APIRouter()

# read_manifest raises SIX classes; this tuple covers all of them,
# ManifestValidationError included.
# Restated file-local, not via a shared helper (see
# api/ollama.py) -- kept in sync with the other five copies by the drift
# test's assertion, not an import.
_UNREADABLE_MANIFEST = (
    FileNotFoundError,
    ManifestValidationError,
    yaml.YAMLError,
    ValidationError,
    OSError,
    UnicodeDecodeError,
)


# Constants
# The Content-Length ceiling is a live config field, not a hardcoded module
# constant (config.py's HttpConfig.max_body_bytes,
# shared with api/body_size_limit.py's middleware -- one value, not two) --
# see _content_length_gate below.
_BATCH_CAP = 64  # power-of-2 batch cap
_UPSTREAM_TIMEOUT_S = 120.0
_SLOT_READY_TIMEOUT_S = 7200.0

# Bound on the finally block's watch_task.cancel()+await
# teardown. Reachable production race (see _defer_unroutable/submit()'s own
# real async work between enqueue and return): slot.stream_ready_event can
# already be SET by the time this route reaches its ready/completion_future
# race, so the race falls straight through with NO intervening await ever
# having run -- unlike chat_completion.py's SSE routes, which get a forced
# ASGI yield (StreamingResponse initiation) before their identical race is
# ever evaluated, this JSON route (borrowing the streaming submission
# primitive without returning a stream) does not. watch_task can then reach
# this teardown never having had its OWN first scheduling turn; cancelling
# and bare-awaiting it can wedge the whole response indefinitely (an
# unbounded hang). 1.0s is generous versus normal asyncio scheduling
# latency while keeping the worst-case
# added teardown latency small and fixed -- nowhere near the unbounded
# hang it prevents.
_WATCH_TASK_TEARDOWN_TIMEOUT_S = 1.0

# Module-scope httpx client (no factory pattern)
# asyncio.Lock to prevent TOCTOU between is_closed check and creation.
_HTTPX_CLIENT: httpx.AsyncClient | None = None
_HTTPX_CLIENT_LOCK: asyncio.Lock | None = None  # lazy-init in _get_httpx_client


async def _get_httpx_client() -> httpx.AsyncClient:
    global _HTTPX_CLIENT, _HTTPX_CLIENT_LOCK
    if _HTTPX_CLIENT_LOCK is None:
        _HTTPX_CLIENT_LOCK = asyncio.Lock()
    async with _HTTPX_CLIENT_LOCK:
        if _HTTPX_CLIENT is None or _HTTPX_CLIENT.is_closed:
            # Close the old client before creating a new one to avoid leaks.
            if _HTTPX_CLIENT is not None:
                try:
                    _HTTPX_CLIENT.close()
                except Exception:
                    pass
            _HTTPX_CLIENT = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT_S)
        return _HTTPX_CLIENT


async def _content_length_gate(request: Request) -> None:
    """Reject HTTP 413 before pydantic parse if Content-Length > ceiling."""
    cl = request.headers.get("content-length")
    if cl is None:
        return  # chunked encoding falls through to batch-cap + per-item limits
    try:
        n = int(cl)
    except ValueError:
        # This 400 looks stricter than
        # body_size_limit.py's matching except ValueError, which lets a
        # malformed Content-Length fall through unchecked -- but that
        # asymmetry is inert, not a live bypass. A malformed Content-Length
        # never reaches either guard: uvicorn's h11 AND httptools protocols
        # (both tested -- production resolves "auto" to httptools via
        # uvicorn[standard], not h11) reject it inside
        # Protocol.data_received(), before any ASGI scope is constructed, so
        # neither this dependency nor body_size_limit.py's middleware is ever
        # invoked for such a request. Checked with a raw-socket probe against
        # both parsers; see
        # the uvicorn protocol sources for details. Do NOT
        # remove this 400 to "match" the other file's fall-through.
        raise HTTPException(status_code=400, detail="invalid Content-Length header")
    max_body_bytes = request.app.state.manager.runtime.http.max_body_bytes
    if n > max_body_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"request body {n} bytes exceeds {max_body_bytes} ceiling",
        )


def _raise_capacity_or_routing_failure(cf_exc: BaseException) -> None:
    """The ONE source of truth for the capacity/routing failure
    response, called from both places a routing failure can surface
    (the ``not ready_task.done()`` branch and the
    ``handle is None`` branch below) so the two capacity-503 sites
    cannot drift apart; one helper keeps their bodies byte-identical.
    The 503 body reuses ``chat_completion.capacity_unavailable_error_body``
    (the shared constructor) so this JSON body and the SSE frame carry byte-
    identical nested shape (the shared error contract).
    """
    if isinstance(cf_exc, VramOverCommitError):
        raise HTTPException(
            status_code=503,
            detail={"error": capacity_unavailable_error_body(cf_exc)},
            headers={"Retry-After": str(cf_exc.retry_after_s)},
        ) from cf_exc
    raise HTTPException(
        status_code=500, detail=f"routing failed: {cf_exc}",
    ) from cf_exc


def _consume_abandoned_watch_task_result(task: "asyncio.Task") -> None:
    """If the finally block gave up WAITING on watch_task
    (bounded teardown timed out) rather than actually finishing it, the task
    keeps running in the background and will eventually complete on its own
    time. Without this, a late completion carrying an exception would only
    ever surface as an unretrieved "Task exception was never retrieved"
    warning at GC time -- noisy and untraceable back to this route. This
    retrieves and logs it explicitly instead, so giving up on the WAIT never
    turns into silently losing the task's own eventual result.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning(
            "watch_task (abandoned after teardown timeout) "
            "finished with an exception: %r", exc,
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

    # Gate 3: encoding_format=base64 → 400 plain-string
    if req.encoding_format is not None and req.encoding_format.lower() == "base64":
        raise HTTPException(
            status_code=400,
            detail="encoding_format='base64' not supported; use 'float'",
        )

    # Gate 4: dimensions param present → 400 plain-string
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
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (yaml.YAMLError, ValidationError, UnicodeDecodeError, OSError) as e:
        # Same handled-500 shape as chat_completion.py / models.py's
        # get_model precedent. Fixed message only -- never str(e): pydantic
        # ValidationError echoes manifest field VALUES, yaml.YAMLError leaks
        # parser position/content. Full detail logged server-side instead.
        log.exception("manifest for %r is present but unreadable", req.model)
        raise HTTPException(
            status_code=500,
            detail=f"manifest for '{req.model}' is present but unreadable",
        ) from e
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
    # Client-disconnect watcher for embeddings.
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
            # Client closed before activation → HTTP 499
            raise HTTPException(
                status_code=499,
                detail={"error": "client_closed_request", "message": str(e)},
            ) from e
        except SidecarUnavailableError as e:
            # A typed arm is needed here: a bare
            # `except RuntimeError` arm would mislabel
            # EVERY RuntimeError (including this one, since
            # SidecarUnavailableError subclasses RuntimeError -- see slot.py)
            # as a flat-string "sidecar unavailable: {e}" body with a
            # HARDCODED Retry-After of "5", ignoring the exception's own
            # (correct) retry_after_s. This arm uses the typed dict shape (same
            # shared constructor as chat_completion.py's identical family)
            # and the exception's own Retry-After value.
            raise HTTPException(
                status_code=503,
                detail={"error": sidecar_unavailable_error_body(e)},
                headers={"Retry-After": str(e.retry_after_s)},
            ) from e
        except VramOverCommitError as e:
            # This arm must come ahead of the generic
            # RuntimeError catch below -- VramOverCommitError also
            # subclasses RuntimeError, so it would otherwise be swallowed flat as
            # the same mislabeled 503 above. LATENT, not live -- see
            # capacity_unavailable_error_body's own SCOPE docstring
            # (chat_completion.py) for why submit_for_streaming cannot raise
            # this synchronously today.
            _raise_capacity_or_routing_failure(e)
        except RuntimeError as e:
            # Genuine residual case (e.g. QueueClosed on shutdown) -- no
            # typed cause/retry_after_s to draw from, unlike the two arms
            # above. Retry-After is therefore the fixed "5": the
            # exception's own retry_after_s is used only by the typed arms
            # above. The shape is still wrapped as a typed dict for the
            # same one-parse-rule reason the typed shape exists (clients
            # parse one error shape), even though this specific residual
            # case has no typed cause
            # to report.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "type": "sidecar_unavailable",
                        "cause": "unknown",
                        "message": str(e),
                    }
                },
                headers={"Retry-After": "5"},
            ) from e

        # Race stream_ready_event against the completion_future. A
        # routing failure (VRAM over-commit / evict-pending exhaustion) fails the
        # completion_future WITHOUT ever setting stream_ready_event; a lone
        # wait_for on stream_ready_event would block the full _SLOT_READY_TIMEOUT_S (up to
        # 2h) then mislabel a 504. Mirror the OpenAI-streaming completion_future
        # probe so a capacity failure surfaces immediately as a retryable 503.
        ready_task = asyncio.ensure_future(slot.stream_ready_event.wait())
        # Only race a REAL completion_future (a Future, or None, in production); guard
        # against an unexpected type so asyncio.wait never receives a non-awaitable.
        cf = slot.completion_future
        if not isinstance(cf, asyncio.Future):
            cf = None
        waiters = [ready_task] + ([cf] if cf is not None else [])
        done, _pending = await asyncio.wait(
            waiters,
            timeout=_SLOT_READY_TIMEOUT_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not ready_task.done():
            # stream_ready never fired: cancel OUR wrapper (never the worker-owned
            # completion_future), then surface a routing failure if there is one,
            # else a genuine timeout.
            ready_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ready_task
            cf_exc = (
                cf.exception()
                if (cf is not None and cf in done and not cf.cancelled()
                    and cf.exception() is not None)
                else None
            )
            if cf_exc is not None:
                _raise_capacity_or_routing_failure(cf_exc)  # shared site
            raise HTTPException(
                status_code=504,
                detail=f"slot did not reach ACTIVE within {_SLOT_READY_TIMEOUT_S}s",
            )

        handle = slot.stream_handle
        if handle is None:
            # manager.py's
            # _fail_completion_future also sets stream_ready_event on a capacity
            # failure (not only on a genuine placement
            # success), so ready_task.done() does not discriminate "reached
            # ACTIVE" from "woken by a failure" -- both land here. Mirror
            # chat_completion.py's discrimination (stream_ready_failed_reason
            # non-None => this was a failure wakeup, not a real placement) rather
            # than its transport (embeddings is a JSON route: raise, don't yield).
            # cf_exc extraction reuses the SAME guard as the routing-failure
            # branch above; the raise itself goes through the SAME shared helper
            # so the two capacity-503 sites cannot drift.
            if slot.stream_ready_failed_reason is not None:
                cf_exc = (
                    cf.exception()
                    if (cf is not None and cf in done and not cf.cancelled()
                        and cf.exception() is not None)
                    else None
                )
                if cf_exc is not None:
                    _raise_capacity_or_routing_failure(cf_exc)  # shared site
            raise HTTPException(
                status_code=500,
                detail="slot reached ACTIVE but stream_handle is None",
            )

        upstream_url = f"http://127.0.0.1:{handle.port}/v1/embeddings"
        upstream_payload = {"model": req.model, "input": req.input}
        client = await _get_httpx_client()
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
        # Tear down disconnect watcher cleanly.
        # A BARE await here can wedge the whole response
        # indefinitely -- watch_task can reach this point never having had
        # its own first scheduling turn (see _WATCH_TASK_TEARDOWN_TIMEOUT_S's
        # own comment for the reachable race and why this route differs from
        # chat_completion.py's SSE routes). Bounded so cleanup can never hang
        # the caller; the three outcomes are handled distinctly rather than
        # blanket-suppressed, so a genuine bug in watch_task itself is still
        # visible instead of silently absorbed alongside the expected
        # cancellation/timeout cases.
        watch_task.cancel()
        try:
            await asyncio.wait_for(
                watch_task, timeout=_WATCH_TASK_TEARDOWN_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            pass  # our own cancel() landing -- expected, not an error
        except asyncio.TimeoutError:
            # watch_task never resolved even after cancellation within the
            # bound -- stop waiting rather than hang the response. The task
            # is not abandoned/leaked: it keeps running to completion on its
            # own time, and its eventual result/exception is still retrieved
            # (not silently dropped) via the done-callback below.
            log.warning(
                "watch_task teardown exceeded %.1fs after cancel(); "
                "abandoning the wait so the response is not blocked",
                _WATCH_TASK_TEARDOWN_TIMEOUT_S,
            )
            watch_task.add_done_callback(_consume_abandoned_watch_task_result)
        except Exception:
            # A REAL bug in watch_task itself -- distinct from the expected
            # cancellation/timeout paths above, so it must not be silently
            # swallowed alongside them.
            log.exception(
                "watch_task raised unexpectedly during teardown"
            )
        # Guard: slot is None if submit_for_streaming raised pre-acquire
        # (SlotEvictedError, RuntimeError) — no stream_done_event to flip.
        if (
            slot is not None
            and slot.stream_done_event is not None
            and not slot.stream_done_event.is_set()
        ):
            slot.stream_done_event.set()
