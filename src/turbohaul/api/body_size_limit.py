"""ASGI middleware enforcing a max request-body size on chat/completion routes.

Reject oversized request bodies with HTTP 413 on
/v1/chat/completions and /api/chat *before* the route handler runs. Mirrors
the Content-Length ceiling already shipped for /v1/embeddings
(turbohaul/api/embeddings.py:_content_length_gate) — same
Content-Length-only check (chunked requests with no Content-Length fall
through, matching the existing embeddings precedent — this leaves that
gap exactly as wide as it already was).
Implemented as ASGI middleware (not a FastAPI dependency) so it runs ahead
of routing/dependency-injection for these two paths specifically.

The ceiling is a live config value rather than a hardcoded ~2MB module constant
(embeddings.py reads the same value), so large multimodal requests
(video/large images) are not silently rejected before the route
handler -- or the model -- sees them. Both read the SAME live config
field, `TurbohaulConfig.runtime.http.max_body_bytes` (config.py's
HttpConfig, default 64 MiB, PUT /api/config + TURBOHAUL_MAX_BODY_BYTES
adjustable). The limit is checked against the declared
Content-Length only.
"""
import logging

from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger(__name__)

_GUARDED_PATHS = frozenset({"/v1/chat/completions", "/api/chat"})


class BodySizeLimitMiddleware:
    """Reject request bodies over runtime.http.max_body_bytes on _GUARDED_PATHS with 413."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in _GUARDED_PATHS:
            await self.app(scope, receive, send)
            return

        content_length = dict(scope.get("headers") or []).get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                # Note: this fall-through looks like it lets a
                # malformed (non-integer) Content-Length through unchecked --
                # contrast embeddings.py:_content_length_gate, which raises 400
                # on the identical ValueError. That asymmetry is real but
                # INERT: it is unreachable. uvicorn's h11 AND httptools
                # protocols (both tested -- production resolves "auto" to
                # httptools via uvicorn[standard], not h11) reject a malformed
                # Content-Length inside Protocol.data_received(), before any
                # ASGI scope is constructed -- this middleware's __call__ is
                # never invoked for such a request in the first place.
                # Both parsers reject a malformed Content-Length before any ASGI scope is
                # built (verified with a raw-socket probe against each parser; the parser
                # source shows the same early rejection). Do NOT "fix" this by aligning
                # the two handlers without first re-checking that the early rejection
                # still holds.
                declared = None
            if declared is not None:
                max_body_bytes = scope["app"].state.manager.runtime.http.max_body_bytes
                if declared > max_body_bytes:
                    await self._reject(send, declared, max_body_bytes)
                    return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send: Send, size: int, max_body_bytes: int) -> None:
        body = (
            f'{{"detail": "request body {size} bytes exceeds {max_body_bytes} ceiling"}}'
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
