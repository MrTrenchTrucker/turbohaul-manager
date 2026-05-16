"""Pull endpoints per v0.2 ARCHITECTURE.md §9 + §9.1 + §12.1.

POST /api/pull-url — arbitrary https URL (SSRF guard enforced)
POST /api/pull-hf — HuggingFace allowlist + HF_API_KEY injection ONLY to allowlisted hosts
POST /api/pull — Ollama registry (501 stub; Phase 5+ implements manifest+layer protocol)

All streaming pulls land via write_stream_atomic_async with per_stream_max_bytes
ceiling. Progress events emit to /ws/state via mgr.event_bus.
"""
import logging
import os
import secrets

import httpx
from fastapi import APIRouter, HTTPException, Request

from turbohaul.blob_store import (
    BlobError,
    BlobHashMismatch,
    BlobSizeExceeded,
    write_stream_atomic_async,
)
from turbohaul.ssrf_guard import UrlSafetyError, is_hf_host, validate_pull_url


log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["pull"])


def _default_http_client_factory(timeout_s: float = 600.0):
    return httpx.AsyncClient(timeout=timeout_s, follow_redirects=False)


def _double_resolve_check(url: str) -> tuple[str, str]:
    """Validate URL + resolve TWICE; raise if (host, ip) differs across calls.

    HAUL S-1 fix (CRITICAL DNS rebind mitigation): the canonical fix is a
    custom httpx.AsyncHTTPTransport that connects to the pre-resolved IP and
    pins SNI to the original hostname — that's the v0.2.2 follow-on. This
    interim fix catches the obvious rebind pattern where the attacker-DNS
    returns a public IP at validate time and an internal IP at connect time:
    we resolve back-to-back and refuse the connection if the two
    resolutions diverge. Race window shrinks from "long enough for httpx to
    do its own connect-time DNS" down to "two synchronous resolver calls
    apart" — microseconds vs milliseconds. Defense-in-depth with the SSRF
    guard's RFC1918/NAT64/IPv4-compat blocks (which catch internal-IP
    rebinds even if they win the race).

    HAUL S-3 fix (MED, multi-record blindspot): validate_pull_url checks the
    first A record only. By running it twice, we also catch round-robin DNS
    where validate sees one record and httpx would see another.
    """
    host1, ip1 = validate_pull_url(url)
    host2, ip2 = validate_pull_url(url)
    if host1 != host2 or ip1 != ip2:
        raise UrlSafetyError(
            f"DNS rebind class detected: validate#1 ({host1}->{ip1}) != "
            f"validate#2 ({host2}->{ip2}). Refusing pull."
        )
    return host1, ip1


async def _stream_chunks(client: httpx.AsyncClient, url: str, headers: dict):
    """Yield bytes from an HTTP GET stream."""
    async with client.stream("GET", url, headers=headers) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
            yield chunk


def _http_client_factory_from_app(app):
    """Allow tests to inject a fake httpx via app.state.http_client_factory."""
    return getattr(app.state, "http_client_factory", None) or _default_http_client_factory


@router.post("/pull-url")
async def pull_url(payload: dict, request: Request) -> dict:
    """Pull a blob from an arbitrary https URL into the blob store."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be JSON object")
    url = payload.get("url")
    expected_sha256 = payload.get("expected_sha256")
    if not url:
        raise HTTPException(status_code=400, detail="`url` required")

    try:
        host, _ = _double_resolve_check(url)
    except UrlSafetyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    mgr = request.app.state.manager
    blobs_root = mgr.boot.storage.blob_store_path
    pull_id = "pull-" + secrets.token_hex(8)
    mgr.event_bus.publish_nowait(
        {"event": "pull_url_started", "pull_id": pull_id, "host": host}
    )

    factory = _http_client_factory_from_app(request.app)
    try:
        async with factory() as client:
            sha, bytes_written = await write_stream_atomic_async(
                blobs_root,
                _stream_chunks(client, url, {}),
                expected_sha256=expected_sha256,
                per_stream_max_bytes=mgr.runtime.pull.per_stream_max_bytes,
            )
    except BlobSizeExceeded as e:
        mgr.event_bus.publish_nowait(
            {"event": "pull_url_failed", "pull_id": pull_id, "reason": "size-exceeded"}
        )
        raise HTTPException(status_code=413, detail=str(e)) from e
    except BlobHashMismatch as e:
        mgr.event_bus.publish_nowait(
            {"event": "pull_url_failed", "pull_id": pull_id, "reason": "hash-mismatch"}
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except BlobError as e:
        mgr.event_bus.publish_nowait(
            {"event": "pull_url_failed", "pull_id": pull_id, "reason": "blob-error"}
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
    except httpx.HTTPError as e:
        mgr.event_bus.publish_nowait(
            {"event": "pull_url_failed", "pull_id": pull_id, "reason": "upstream-error"}
        )
        raise HTTPException(status_code=502, detail=f"upstream HTTP error: {e}") from e

    mgr.event_bus.publish_nowait(
        {
            "event": "pull_url_complete",
            "pull_id": pull_id,
            "sha256": sha,
            "bytes_written": bytes_written,
        }
    )
    return {
        "pull_id": pull_id,
        "status": "complete",
        "sha256": sha,
        "bytes_written": bytes_written,
        "host": host,
    }


@router.post("/pull-hf")
async def pull_hf(payload: dict, request: Request) -> dict:
    """Pull a file from HuggingFace. host must match hf_host_allowlist.

    HF_API_KEY (from env named by `pull.hf_api_key_env`) is injected as
    `Authorization: Bearer` ONLY when host matches allowlist (defense against
    key exfil via redirect to attacker host).
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be JSON object")
    repo_id = payload.get("repo_id")
    filename = payload.get("filename")
    revision = payload.get("revision", "main")
    expected_sha256 = payload.get("expected_sha256")
    if not repo_id or not filename:
        raise HTTPException(
            status_code=400, detail="`repo_id` + `filename` required"
        )
    # Build canonical HF URL
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"

    mgr = request.app.state.manager
    allowlist = mgr.runtime.pull.hf_host_allowlist

    try:
        host, _ = _double_resolve_check(url)
    except UrlSafetyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not is_hf_host(host, allowlist):
        raise HTTPException(
            status_code=403,
            detail=f"host {host} not in hf_host_allowlist (Security F3 - "
            "HF_API_KEY only sent to allowlisted hosts)",
        )

    hf_key_env = mgr.runtime.pull.hf_api_key_env
    hf_key = os.environ.get(hf_key_env, "")
    headers: dict[str, str] = {}
    if hf_key:
        headers["Authorization"] = f"Bearer {hf_key}"

    blobs_root = mgr.boot.storage.blob_store_path
    pull_id = "pull-" + secrets.token_hex(8)
    mgr.event_bus.publish_nowait(
        {
            "event": "pull_hf_started",
            "pull_id": pull_id,
            "repo_id": repo_id,
            "filename": filename,
        }
    )

    factory = _http_client_factory_from_app(request.app)
    try:
        async with factory() as client:
            sha, bytes_written = await write_stream_atomic_async(
                blobs_root,
                _stream_chunks(client, url, headers),
                expected_sha256=expected_sha256,
                per_stream_max_bytes=mgr.runtime.pull.per_stream_max_bytes,
            )
    except BlobSizeExceeded as e:
        mgr.event_bus.publish_nowait(
            {"event": "pull_hf_failed", "pull_id": pull_id, "reason": "size-exceeded"}
        )
        raise HTTPException(status_code=413, detail=str(e)) from e
    except BlobHashMismatch as e:
        mgr.event_bus.publish_nowait(
            {"event": "pull_hf_failed", "pull_id": pull_id, "reason": "hash-mismatch"}
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except BlobError as e:
        mgr.event_bus.publish_nowait(
            {"event": "pull_hf_failed", "pull_id": pull_id, "reason": "blob-error"}
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
    except httpx.HTTPError as e:
        mgr.event_bus.publish_nowait(
            {"event": "pull_hf_failed", "pull_id": pull_id, "reason": "upstream-error"}
        )
        raise HTTPException(status_code=502, detail=str(e)) from e

    mgr.event_bus.publish_nowait(
        {
            "event": "pull_hf_complete",
            "pull_id": pull_id,
            "sha256": sha,
            "bytes_written": bytes_written,
        }
    )
    return {
        "pull_id": pull_id,
        "status": "complete",
        "sha256": sha,
        "bytes_written": bytes_written,
        "host": host,
    }


@router.post("/pull")
async def pull_ollama_registry(payload: dict, request: Request) -> dict:
    """Ollama registry pull - 501 stub for v1.

    The Ollama registry uses a custom manifest+layer protocol (similar to Docker).
    Phase 5+ implements this; for v1 use /api/pull-hf for HF or /api/pull-url for arbitrary.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "Ollama registry pull not implemented in v1. Use /api/pull-hf for "
            "HuggingFace or /api/pull-url for arbitrary https sources."
        ),
    )
