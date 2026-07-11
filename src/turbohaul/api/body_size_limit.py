"""ASGI middleware enforcing a max request-body size on chat/completion routes.

Reject oversized request bodies with HTTP 413 on
/v1/chat/completions and /api/chat *before* the route handler runs. Mirrors
the Content-Length ceiling already shipped for /v1/embeddings
(turbohaul/api/embeddings.py:_MAX_REQUEST_BYTES) — same ~2MB default
and same Content-Length-only check (chunked requests with no
Content-Length fall through, matching the existing embeddings precedent).
Implemented as ASGI middleware (not a FastAPI dependency) so it runs ahead
of routing/dependency-injection for these two paths specifically.
"""
import logging

from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger(__name__)

# Mirrors embeddings.py's _MAX_REQUEST_BYTES ceiling: ~2MB.
_MAX_BODY_BYTES = 32_768 * 64

_GUARDED_PATHS = frozenset({"/v1/chat/completions", "/api/chat"})


class BodySizeLimitMiddleware:
    """Reject request bodies over _MAX_BODY_BYTES on _GUARDED_PATHS with 413."""

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
                declared = None
            if declared is not None and declared > _MAX_BODY_BYTES:
                await self._reject(send, declared)
                return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send: Send, size: int) -> None:
        body = (
            f'{{"detail": "request body {size} bytes exceeds {_MAX_BODY_BYTES} ceiling"}}'
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
