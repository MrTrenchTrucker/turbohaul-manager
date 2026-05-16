"""FastAPI app for Turbohaul-Manager.

Per v0.2 ARCHITECTURE.md §9 + §11. Phase 5 Wave 16 adds /ui static-file serving
with SPA fallback + CSP + security headers.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from turbohaul import __version__
from turbohaul.api.chat_completion import router as chat_completion_router
from turbohaul.api.config_put import router as config_put_router
from turbohaul.api.import_ import router as import_router
from turbohaul.api.manifests import router as manifests_router
from turbohaul.api.ollama import router as ollama_router
from turbohaul.api.pull import router as pull_router
from turbohaul.api.ws_state import router as ws_state_router
from turbohaul.config import BootConfig, RuntimeConfig
from turbohaul.manager import TurbohaulManager


log = logging.getLogger(__name__)


# CSP verified against Logbook src/frontend/nginx.conf (production-validated 2026-05).
# Adopted verbatim to inherit prod hardening. Permits same-origin scripts, inline
# styles (Tailwind injects), data: + blob: images, ws/wss connections same-origin,
# self-hosted fonts. Denies object/embed and framing.
_CSP_HEADER = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' ws: wss:; "
    "font-src 'self' data:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "object-src 'none'"
)


def _ui_security_headers() -> dict[str, str]:
    """Headers applied to every /ui/* response per v0.2 §11.2."""
    return {
        "Content-Security-Policy": _CSP_HEADER,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "same-origin",
    }


# Vite-emitted file extensions that carry content-hashes — safe to cache long-term.
_HASHED_ASSET_EXTENSIONS = frozenset(
    {".js", ".css", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".woff", ".woff2"}
)


def create_app(
    boot: BootConfig,
    runtime: RuntimeConfig,
    *,
    auto_start_worker: bool = True,
    auto_boot_reconcile: bool = True,
) -> FastAPI:
    """Create a FastAPI app wired to a TurbohaulManager instance.

    auto_start_worker / auto_boot_reconcile let tests skip lifecycle side effects.
    """
    mgr = TurbohaulManager(boot, runtime)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if auto_boot_reconcile:
            try:
                reconcile = mgr.boot_reconcile()
                log.info("boot_reconcile: %s", reconcile)
            except Exception:
                log.exception("boot_reconcile failed")
            if not mgr.verify_binary():
                log.error(
                    "llama_server_binary sha256 mismatch — set "
                    "runtime.llama_server_binary_sha256 to empty for dev, "
                    "or correct the pinned value."
                )
        if auto_start_worker:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            yield
        finally:
            await mgr.shutdown()

    app = FastAPI(
        title="Turbohaul-Manager",
        description="Ollama-shape inference manager using TurboQuant llama.cpp (v0.2).",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.manager = mgr
    app.include_router(ollama_router)
    app.include_router(manifests_router)
    app.include_router(config_put_router)
    app.include_router(ws_state_router)
    app.include_router(chat_completion_router)
    app.include_router(pull_router)
    app.include_router(import_router)

    @app.get("/health")
    async def health() -> dict:
        """Liveness + version."""
        return {"status": "ok", "version": __version__}

    @app.get("/status")
    async def status() -> dict:
        """Queue + active + grace + idle state per v0.2 §9.3."""
        return mgr.status_snapshot()

    @app.get("/api/version")
    async def api_version() -> dict:
        """User-Agent / version info per v0.2 §9."""
        return {
            "version": __version__,
            "backend": "turboquant-llama-cpp",
            "backend_sha_pinned": bool(boot.runtime.llama_server_binary_sha256),
            "api_compat": "ollama-superset",
            "user_agent": f"Turbohaul-Manager/{__version__} (Ollama-compatible)",
        }

    @app.get("/api/config")
    async def get_config() -> dict:
        """Return current runtime + boot config (read-only view).

        Reads live runtime from mgr.runtime so PUT-mutations are reflected.
        """
        live_runtime = mgr.runtime
        return {
            "server": boot.server.model_dump(mode="json"),
            "storage": {
                "blob_store_path": str(boot.storage.blob_store_path),
                "manifests_path": str(boot.storage.manifests_path),
                "import_allowed_root": str(boot.storage.import_allowed_root),
                "state_db_path": str(boot.storage.state_db_path),
            },
            "runtime": {
                "llama_server_binary": str(boot.runtime.llama_server_binary),
                "llama_server_binary_sha256": boot.runtime.llama_server_binary_sha256,
                "default_port_base": boot.runtime.default_port_base,
            },
            "ui": {
                "enabled": boot.ui.enabled,
                "static_path": str(boot.ui.static_path),
            },
            "queue": live_runtime.queue.model_dump(mode="json"),
            "pull": live_runtime.pull.model_dump(mode="json"),
        }

    # Phase 5 §11: /ui static-file serving with SPA fallback + CSP.
    # Only registered when the bundle is enabled AND the static dir exists,
    # so tests that don't provision a ui_dist see no /ui route.
    if boot.ui.enabled and boot.ui.static_path.exists():
        ui_root = boot.ui.static_path.resolve()
        index_html = ui_root / "index.html"

        async def _serve(full_path: str) -> FileResponse:
            if full_path:
                candidate = (ui_root / full_path).resolve()
                # Path-traversal guard: candidate MUST be under ui_root.
                try:
                    candidate.relative_to(ui_root)
                except ValueError:
                    candidate = None
                if candidate is not None and candidate.is_file():
                    headers = _ui_security_headers()
                    if candidate.suffix.lower() in _HASHED_ASSET_EXTENSIONS:
                        headers["Cache-Control"] = "public, max-age=31536000, immutable"
                    else:
                        headers["Cache-Control"] = "no-cache, must-revalidate"
                    return FileResponse(candidate, headers=headers)
            # SPA fallback (anything not matching a real file → index.html).
            if not index_html.is_file():
                raise HTTPException(
                    status_code=404,
                    detail="UI bundle is enabled but index.html is missing.",
                )
            headers = _ui_security_headers()
            headers["Cache-Control"] = "no-cache, must-revalidate"
            return FileResponse(index_html, headers=headers)

        @app.get("/ui", include_in_schema=False)
        async def serve_ui_root() -> FileResponse:
            return await _serve("")

        @app.get("/ui/", include_in_schema=False)
        async def serve_ui_root_slash() -> FileResponse:
            return await _serve("")

        @app.get("/ui/{full_path:path}", include_in_schema=False)
        async def serve_ui(full_path: str) -> FileResponse:
            return await _serve(full_path)

    return app
