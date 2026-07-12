"""FastAPI app for Turbohaul-Manager.

Serves the API plus /ui static files with SPA fallback, CSP, and security
headers.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from turbohaul import __version__
from turbohaul.api.chat_completion import (
    make_llama_server_complete_fn,
    router as chat_completion_router,
)
from turbohaul.api.body_size_limit import BodySizeLimitMiddleware
from turbohaul.api.config_put import router as config_put_router
from turbohaul.api.embeddings import router as embeddings_router
from turbohaul.api.import_ import router as import_router
from turbohaul.api.live_stream import router as live_stream_router
from turbohaul.api.logging import router as logging_router
from turbohaul.api.manifests import router as manifests_router
from turbohaul.api.ollama import router as ollama_router
from turbohaul.api.telemetry import router as telemetry_router
from turbohaul.api.pull import router as pull_router
from turbohaul.api.ws_state import router as ws_state_router
from turbohaul.config import BootConfig, RuntimeConfig
from turbohaul.live_monitor import LiveResidentsSupervisor, LiveSlotsPoller
from turbohaul.manager import TurbohaulManager
from turbohaul.state import close_audit_pool, init_audit_pool


log = logging.getLogger(__name__)


# CSP adapted from a production-hardened nginx config. Permits same-origin
# scripts, inline styles (Tailwind injects), data: + blob: images, ws/wss
# connections same-origin, self-hosted fonts. Denies object/embed and framing.
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
    """Headers applied to every /ui/* response."""
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
    mgr = TurbohaulManager(
        boot,
        runtime,
        complete_fn=make_llama_server_complete_fn(
            manifests_path=boot.storage.manifests_path,
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Eager-init the audit pool BEFORE boot_reconcile (which writes audit).
        init_audit_pool(boot.storage.state_db_path)
        if auto_boot_reconcile:
            try:
                # boot_reconcile is sync; offload to a worker thread so the
                # audit_db_session sync-only guard doesn't trip on the
                # lifespan event loop.
                reconcile = await asyncio.to_thread(mgr.boot_reconcile)
                log.info("boot_reconcile: %s", reconcile)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                log.exception("boot_reconcile failed")
            if not mgr.verify_binary():
                log.error(
                    "llama_server_binary sha256 mismatch — set "
                    "runtime.llama_server_binary_sha256 to empty for dev, "
                    "or correct the pinned value."
                )
            # Purge KV bins whose stamped build/model/ctx
            # fingerprint no longer matches the current engine BEFORE serving — a
            # llama-server binary rebuild (new engine_build_id) or a manifest
            # ctx/gguf change would otherwise leave stale bins that, if restored,
            # produce garbage KV. Gated by TURBOHAUL_FINGERPRINT_PURGE, best-effort,
            # offloaded (sync file I/O). File sweep only — never touches the restore
            # decision.
            try:
                purged = await asyncio.to_thread(
                    mgr._purge_mismatched_bins, reason="startup"
                )
                if purged:
                    log.info(
                        "startup fingerprint purge: removed %d stale KV bin(s)",
                        purged,
                    )
            except Exception:
                log.exception("startup fingerprint purge failed (best-effort)")
        if auto_start_worker:
            mgr._worker_task = asyncio.create_task(mgr.worker_loop())
            # Spawn the background sweeper alongside the worker_loop.
            # Lifecycle is symmetric — shutdown() cancels both with
            # contextlib.suppress(CancelledError).
            mgr._sweeper_task = asyncio.create_task(
                mgr._periodic_terminal_park_sweep()
            )
            # Live inference monitor (pure observer). At cap>=2 a single
            # supervisor task polls EVERY resident's /slots + caches per-GPU VRAM;
            # at cap<=1 the legacy single-sidecar poller runs verbatim (byte-identical).
            if runtime.monitor.enabled:
                if mgr.runtime.queue.max_parallel_sidecars >= 2:
                    mgr._live_supervisor = LiveResidentsSupervisor(
                        mgr, interval_s=runtime.monitor.poll_interval_s
                    )
                    mgr._live_supervisor_task = asyncio.create_task(
                        mgr._live_supervisor.run()
                    )
                else:
                    mgr._live_poller = LiveSlotsPoller(
                        mgr, interval_s=runtime.monitor.poll_interval_s
                    )
                    mgr._live_poller_task = asyncio.create_task(mgr._live_poller.run())
        try:
            yield
        finally:
            await mgr.shutdown()
            close_audit_pool()

    app = FastAPI(
        title="Turbohaul-Manager",
        description="Ollama-shape inference manager using TurboQuant llama.cpp.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.manager = mgr
    # Reject 413 oversized bodies before routing/handlers on these two paths.
    app.add_middleware(BodySizeLimitMiddleware)
    app.include_router(ollama_router)
    app.include_router(manifests_router)
    app.include_router(config_put_router)
    app.include_router(ws_state_router)
    app.include_router(live_stream_router)
    app.include_router(chat_completion_router)
    app.include_router(pull_router)
    app.include_router(import_router)
    app.include_router(logging_router)
    app.include_router(embeddings_router)
    app.include_router(telemetry_router)

    @app.get("/health")
    async def health() -> dict:
        """Liveness + version."""
        return {"status": "ok", "version": __version__}

    @app.get("/status")
    async def status() -> dict:
        """Queue + active + grace + idle state."""
        return mgr.status_snapshot()

    @app.get("/api/version")
    async def api_version() -> dict:
        """User-Agent / version info."""
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
            # Redact internal paths to basename. Disclosing full absolute
            # paths gave a rebind-pivoting attacker the exact write targets
            # on disk. UI only needs basenames anyway.
            "storage": {
                "blob_store_path": boot.storage.blob_store_path.name,
                "manifests_path": boot.storage.manifests_path.name,
                "import_allowed_root": boot.storage.import_allowed_root.name,
                "state_db_path": boot.storage.state_db_path.name,
            },
            "runtime": {
                "llama_server_binary": boot.runtime.llama_server_binary.name,
                "llama_server_binary_sha256": boot.runtime.llama_server_binary_sha256,
                "default_port_base": boot.runtime.default_port_base,
            },
            "ui": {
                "enabled": boot.ui.enabled,
                "static_path": boot.ui.static_path.name,
            },
            "queue": live_runtime.queue.model_dump(mode="json"),
            "pull": live_runtime.pull.model_dump(mode="json"),
            "persist": live_runtime.persist.model_dump(mode="json"),
        }

    # /ui static-file serving with SPA fallback + CSP.
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
