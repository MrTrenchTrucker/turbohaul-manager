"""FastAPI app for Turbohaul-Manager.

Serves the HTTP API plus the /ui static bundle (SPA fallback, CSP and security
headers); see ARCHITECTURE.md §9 (Configuration) and §11 (Front-end).
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
from turbohaul.api.config_put import compute_config_provenance, load_runtime_provenance_stamp
from turbohaul.api.config_put import router as config_put_router
from turbohaul.api.config_schema import router as config_schema_router
from turbohaul.api.embeddings import router as embeddings_router
from turbohaul.api.fastlane_census import router as fastlane_census_router
from turbohaul.api.import_ import router as import_router
from turbohaul.api.live_stream import router as live_stream_router
from turbohaul.api.logging import router as logging_router
from turbohaul.api.manifests import router as manifests_router
from turbohaul.api.models import router as models_router
from turbohaul.api.ollama import router as ollama_router
from turbohaul.api.plugins import router as plugins_router
from turbohaul.api.exec_ws import router as exec_ws_router
from turbohaul.api.telemetry import router as telemetry_router
from turbohaul.api.pull import router as pull_router
from turbohaul.api.ws_state import router as ws_state_router
from turbohaul.config import BootConfig, RuntimeConfig
from turbohaul.live_monitor import LiveResidentsSupervisor, LiveSlotsPoller
from turbohaul.manager import TurbohaulManager
from turbohaul.fastlane_resolve import FastLaneClientNamer, FastLaneNameResolver
from turbohaul.plugin_health import PluginHealthMonitor
from turbohaul.state import close_audit_pool, init_audit_pool


log = logging.getLogger(__name__)


# CSP header for the bundled UI, taken from a hardened nginx configuration.
# Adopted verbatim to inherit its hardening. Permits same-origin scripts, inline
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
    """Headers applied to every /ui/* response."""
    return {
        "Content-Security-Policy": _CSP_HEADER,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "same-origin",
    }


def _legacy_wire_live_poller(mgr, runtime) -> None:
    """UNCALLED. RETAINED, NOT DELETED: dead code is marked as dead in the
    comments and kept as it is, because it is the legacy path and may be
    useful again.

    WHY unreachable: this was `create_app`'s lifespan cap<=1 branch --
    construct a `LiveSlotsPoller` directly and run it as its own task.
    `LiveSlotsPoller._tick` reads `mgr._active_slot` / `mgr._active_handle`,
    the manager-global fields only `_process_slot` (manager.py) ever wrote --
    and `_process_slot` has no production callers because
    `worker_loop`'s cap fork is unconditionally `>= 1` (config.py's
    `max_parallel_sidecars` is `ge=1`). So this branch was reading
    permanently-dead state and publishing `state: 'idle'` on every tick,
    even with a resident actively serving -- not a genuine cap-safety
    choice, an accident of which observer got ported to the multi-instance
    design first (same shape as the other cap-gated task, the idle
    liveness sweep in `create_app`'s lifespan).

    NOT MAINTAINED. `LiveSlotsPoller` the CLASS is NOT dead --
    `ResidentSlotsPoller` (live_monitor.py) subclasses it and reuses
    `_fetch_revalidate_compute` / `_compute` / `_derive` / `_reset_samples` /
    `_warn_schema` verbatim; only THIS direct-construct-and-run usage (and,
    transitively, `LiveSlotsPoller.run` / `._tick` / `._store` / its own
    `._refresh_vram`) is unreachable. Verify reachability again before
    reattaching a caller -- it will have rotted like any other unexercised
    path.

    WHAT REPLACED IT: `LiveResidentsSupervisor`, now constructed
    UNCONDITIONALLY whenever `runtime.monitor.enabled` (see the call site in
    `create_app`'s lifespan, this file, same block this function used to sit
    in) -- no cap check. `ResidentSlotsPoller` walks a list of one at cap 1.
    """
    mgr._live_poller = LiveSlotsPoller(
        mgr, interval_s=runtime.monitor.poll_interval_s
    )
    mgr._live_poller_task = asyncio.create_task(mgr._live_poller.run())


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
    fastlane_config_error: str | None = None,
    shipped_yaml_raw: dict | None = None,
) -> FastAPI:
    """Create a FastAPI app wired to a TurbohaulManager instance.

    auto_start_worker / auto_boot_reconcile let tests skip lifecycle side effects.
    fastlane_config_error: rejection text from an isolated fail-loud parse of a
    persisted fastlane block (see __main__.py); surfaced via GET /api/config so
    the FE can show a config-rejected banner.
    shipped_yaml_raw: the raw shipped-yaml dict __main__
    parsed at boot, kept only for GET /api/config's provenance field -- never
    consulted for any effective value. Defaults to {} (no provenance beyond
    "not from the shipped yaml") for callers that don't have a real yaml file
    (most existing tests construct BootConfig/RuntimeConfig by hand).
    """
    _shipped_yaml_raw = shipped_yaml_raw or {}
    mgr = TurbohaulManager(
        boot,
        runtime,
        complete_fn=make_llama_server_complete_fn(
            timeout_s=runtime.queue.sidecar_complete_timeout_s,
        ),
    )
    mgr.fastlane_config_error = fastlane_config_error

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
            # Spawn the background sweeper alongside the worker_loop. Lifecycle is symmetric — shutdown() cancels
            # both with contextlib.suppress(CancelledError).
            mgr._sweeper_task = asyncio.create_task(
                mgr._periodic_terminal_park_sweep()
            )
            # Proactive idle-dead-holder sweep -- ports the
            # ENGINE STALLED bin-death-strike / 3-strike auto-quarantine
            # detection onto Loop A, which was reactive-only (a dead-while-
            # idle resident was invisible until the next request routed onto
            # it).
            # This task's OWN name/history say "cap>=2 twin",
            # but that was never actually load-bearing -- worker_loop's own
            # `>= 1` fork means EVERY cap now drives residents
            # through _route_or_reserve -> _reserve_and_start_locked, which is
            # model-tag-keyed with a real _drive_resident task regardless of
            # cap (there is no cap branch anywhere in that path). The
            # cap<=1-only `_SINGLETON_RESIDENT_KEY` resident this gate used to
            # implicitly assume is the "other" cap-1 case is never routed to
            # now and is unrelated to what this sweep scans
            # (_model_residents() already excludes it by name, on purpose,
            # for its own reasons -- unaffected by this gate). So `>= 2`
            # here was gating a task off at the ONE cap where nothing else
            # detects a dead-while-idle engine at all -- not a genuine
            # cap-safety boundary. `>= 1` is unconditionally true
            # (config.py's max_parallel_sidecars is `ge=1`); kept explicit
            # rather than unconditional to match worker_loop's own retained-
            # explicit-check style for the same always-true condition.
            if mgr.runtime.queue.max_parallel_sidecars >= 1:
                mgr._idle_liveness_task = asyncio.create_task(
                    mgr._idle_engine_liveness_sweep()
                )
            # Live inference monitor (pure observer). ONE supervisor task
            # polls EVERY live resident's /slots + caches per-GPU VRAM, at every
            # cap -- the cap<=1/cap>=2 split was removed: the
            # legacy single-sidecar poller it replaced read manager-global
            # fields (`mgr._active_slot`/`_active_handle`) that nothing writes
            # any more (see `_legacy_wire_live_poller`,
            # this file, for the full trace), so it always published
            # `state: 'idle'` regardless of whether a resident was serving.
            # `ResidentSlotsPoller` walks a list of one at cap 1; the
            # compute/rate-EWMA/fetch core is shared with the legacy poller,
            # so there is one implementation and no cap gate.
            if runtime.monitor.enabled:
                mgr._live_supervisor = LiveResidentsSupervisor(
                    mgr, interval_s=runtime.monitor.poll_interval_s
                )
                mgr._live_supervisor_task = asyncio.create_task(
                    mgr._live_supervisor.run()
                )
        # Plugin registry health probe: boot + periodic
        # probe over mgr.boot.plugins.registry, honouring health_path /
        # auth_token_file the same way invoke_plugin does. Best-effort, same
        # shape as the fingerprint purge above -- a probe failure must never
        # prevent boot. Unconditional (not gated on auto_start_worker): it is
        # independent of the worker/queue, and against the default empty
        # registry it does nothing. Results are in-process only (see
        # plugin_health.py's module docstring) -- this must NEVER feed
        # `configured` or api/plugins.py's listing.
        # ONE long-lived Fast Lane name resolver for this manager. It
        # must be a single instance, not one per caller: it owns the resolved
        # -address cache and the generation counter the rule-table cache keys
        # on, so a fresh instance would start empty and its work would be
        # thrown away. Names are asked fresh each tick, since PUT /api/config
        # can change the rule set underneath it.
        mgr._fastlane_resolver = FastLaneNameResolver(mgr.fastlane_rule_names)
        try:
            # Resolve once before serving, so the first request does not race
            # an empty cache and get treated as unlisted.
            await mgr._fastlane_resolver.resolve_once()
        except Exception:
            log.exception("fastlane name resolve boot pass failed (best-effort)")
        mgr._fastlane_resolver_task = asyncio.create_task(
            mgr._fastlane_resolver.run_periodic()
        )
        # Names the Discovered list shows: a background reverse lookup per
        # address, shown only once a forward lookup confirms it. Display only
        # -- it never feeds the rule matcher, and it runs in an executor of
        # its own, off the request path.
        mgr._fastlane_namer = FastLaneClientNamer(
            mgr.fastlane_census_name_targets, mgr.fastlane_census_set_name
        )
        mgr._fastlane_namer_task = asyncio.create_task(
            mgr._fastlane_namer.run_periodic()
        )

        mgr._plugin_health = PluginHealthMonitor(lambda: mgr.boot.plugins.registry)
        try:
            await mgr._plugin_health.probe_once()
        except Exception:
            log.exception("plugin health boot probe failed (best-effort)")
        mgr._plugin_health_task = asyncio.create_task(mgr._plugin_health.run_periodic())
        try:
            yield
        finally:
            # Cancel the Fast Lane name resolver before shutting the
            # manager down. Without this the periodic task outlives the
            # lifespan and keeps calling getaddrinfo against a manager that is
            # tearing down -- asyncio also logs "Task was destroyed but it is
            # pending" on exit. Best-effort and FIRST, so a failure here can
            # never stop mgr.shutdown() from running.
            _fl_task = getattr(mgr, "_fastlane_resolver_task", None)
            if _fl_task is not None:
                _fl_task.cancel()
                try:
                    await _fl_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Same for the client namer: cancel it first, then close its
            # lookup threads (best-effort, before the manager shuts down).
            _fl_namer_task = getattr(mgr, "_fastlane_namer_task", None)
            if _fl_namer_task is not None:
                _fl_namer_task.cancel()
                try:
                    await _fl_namer_task
                except (asyncio.CancelledError, Exception):
                    pass
            _fl_namer = getattr(mgr, "_fastlane_namer", None)
            if _fl_namer is not None:
                _fl_namer.close()
            await mgr.shutdown()
            close_audit_pool()

    app = FastAPI(
        title="Turbohaul-Manager",
        description="Ollama-shape inference manager using TurboQuant llama.cpp (v0.2).",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.manager = mgr
    # 413 oversized bodies before routing/handlers on these two paths.
    app.add_middleware(BodySizeLimitMiddleware)
    app.include_router(ollama_router)
    app.include_router(manifests_router)
    app.include_router(models_router)
    app.include_router(plugins_router)
    app.include_router(exec_ws_router)  # /api/plugins/{tag}/exec WS
    app.include_router(config_put_router)
    app.include_router(config_schema_router)
    app.include_router(ws_state_router)
    app.include_router(live_stream_router)
    app.include_router(chat_completion_router)
    app.include_router(pull_router)
    app.include_router(import_router)
    app.include_router(logging_router)
    app.include_router(embeddings_router)
    app.include_router(telemetry_router)
    app.include_router(fastlane_census_router)

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

        "_provenance" names, per section+field, which
        precedence layer supplied the effective value ("default" | "yaml" |
        "env" | "persisted") -- recomputed fresh on every call (same
        live-reflects-PUT contract as the values above), additive, does not
        change any existing key's shape or value.

        "_provenance_stamp" is a SEPARATE, additive
        signal -- for fields an actual PUT has touched since stamping
        began, the UTC timestamp of the most recent such PUT. Forward-only:
        a field's absence here says nothing about intent for anything
        written before stamping existed (see load_runtime_provenance_stamp's
        docstring). Named distinctly from "_provenance" above on purpose --
        one says WHICH LAYER governs a value today (recomputed, always
        answerable), the other says WHEN a value was last explicitly PUT
        (recorded going forward only, often unanswerable for old data);
        conflating the two names would suggest a completeness this doesn't
        have.
        """
        live_runtime = mgr.runtime
        return {
            "server": boot.server.model_dump(mode="json"),
            # Redact internal paths to basename. Disclosing full
            # absolute paths gave a rebind-pivoting attacker the exact
            # write targets on disk. UI only needs basenames anyway.
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
            # Same redaction rationale as "storage"/"runtime" above:
            # the plugin registry holds host:port for INTERNAL containers --
            # exactly the network topology a rebind-pivoting attacker wants.
            # The UI only needs to know WHICH plugins exist (to list them and
            # edit their runtime knobs), never where they live, so expose the
            # resource keys and withhold every endpoint field.
            "plugins": {
                "registry_keys": sorted(boot.plugins.registry),
            },
            "queue": live_runtime.queue.model_dump(mode="json"),
            "pull": live_runtime.pull.model_dump(mode="json"),
            "persist": live_runtime.persist.model_dump(mode="json"),
            "monitor": live_runtime.monitor.model_dump(mode="json"),
            "kv": live_runtime.kv.model_dump(mode="json"),
            "http": live_runtime.http.model_dump(mode="json"),
            "plugin_runtime": live_runtime.plugin_runtime.model_dump(mode="json"),
            # config_error is a boot-time diagnostic, not a FastLaneConfig
            # field -- nested here (not as a sibling top-level key) so the
            # registry invariant (top-level keys == RUNTIME_SECTIONS) holds.
            "fastlane": {
                **live_runtime.fastlane.model_dump(mode="json"),
                "config_error": getattr(mgr, "fastlane_config_error", None),
            },
            "_provenance": compute_config_provenance(
                _shipped_yaml_raw, boot.storage.state_db_path
            ),
            "_provenance_stamp": load_runtime_provenance_stamp(
                boot.storage.state_db_path
            ),
        }

    # /ui static-file serving with SPA fallback + CSP (ARCHITECTURE.md §11).
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
