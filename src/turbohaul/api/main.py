"""FastAPI app for Turbohaul-Manager.

Per v0.2 ARCHITECTURE.md §9. Phase 2 Wave 6 ships the skeleton: /health, /status,
/api/version, /api/config (GET only for now). Phase 3 will add Ollama + OpenAI
compat routes and the worker_loop completion forwarding to llama-server.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

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
    app.state.manager = mgr  # for tests + future routes
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

        Boot fields are exposed for visibility but PUT /api/config will accept
        ONLY runtime fields; boot fields return HTTP 403 on mutation (v0.2 §7.1).

        Reads live runtime config from mgr.runtime so PUT-mutations are reflected.
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

    return app
