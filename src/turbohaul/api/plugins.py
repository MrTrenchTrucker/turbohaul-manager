"""Agent-facing plugin routes.

The listing is SELF-DESCRIBING. It is for agents, not humans, and
for an agent discoverability IS the product — a capability it cannot find is
indistinguishable from one that was never built. So each entry carries the
plugin's declared routes and executables plus the shape of the invoke call,
and this is done WITHOUT relaxing the host/port property one inch: the
fields are manifest-validated to be relative, scheme-free and traversal-free
at LOAD time, so a host is not a thing that can be stored in them, let alone
returned from here.

GET /api/plugins — discovery listing for resource plugins (media hook), the same
role /v1/models and /api/tags play for chat models: hidden plugins omitted, never
host/port (see get_config()'s "plugins.registry_keys" for the identical redaction
rationale — the plugin registry holds internal container topology, exactly what a
rebind-pivoting attacker wants).

The per-entry status field is `configured`, NOT `resolves` and
never anything reachability-flavoured. This endpoint opens no sockets and so may
not report reachability; see the comment on the field itself in list_plugins()
for the full reasoning, and plugin_invoke.invoke_plugin for where reachability
actually gets established.

POST /api/plugins/{model_tag}/invoke — the only way an agent reaches a plugin.
Addressed by model_tag only; a caller never supplies a host, port, or URL. Resolves
via plugin_invoke.resolve_endpoint/invoke_plugin — never reimplemented here.
Synchronous: by design, results go back inline, not to a file or job
store, and there is no duration timeout (a no-progress check is the right shape for
a hang; the caller's own outer timeout is the real ceiling, out of scope
here, and the same holds for every call made through this
route).
"""
import logging

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from turbohaul.manifest import (
    ManifestValidationError,
    PluginManifest,
    list_manifests,
    read_manifest,
)
from turbohaul.plugin_invoke import PluginInvokeError, invoke_plugin, resolve_endpoint


log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plugins", tags=["plugins"])

# read_manifest can fail SIX ways (FileNotFoundError / OSError /
# UnicodeDecodeError / yaml.YAMLError / ManifestValidationError / pydantic
# ValidationError — see its docstring) — same enumeration api/models.py and
# api/ollama.py already use for their list endpoints, restated here rather than
# imported since it is file-local, not shared plumbing.
# The tuple below covers all six, including OSError and UnicodeDecodeError.
# Restated here, not imported, on purpose.
_UNREADABLE_MANIFEST = (
    FileNotFoundError,
    ManifestValidationError,
    yaml.YAMLError,
    ValidationError,
    OSError,
    UnicodeDecodeError,
)

# PluginInvokeError.reason -> HTTP status, exactly per plugin_invoke's contract. Every
# reason the contract enumerates is listed explicitly; an unrecognized reason
# (a future plugin_invoke addition this file hasn't been updated for) falls back to
# 502 rather than a raw KeyError — still loud, still recoverable, not a 500.
_REASON_TO_STATUS: dict[str, int] = {
    "unknown_resource": 400,
    "blocked_target": 400,
    "unreachable": 502,
    "plugin_error": 502,
    "no_progress": 504,
}
_DEFAULT_REASON_STATUS = 502


def _plugin_invoke_error_to_http(e: PluginInvokeError) -> HTTPException:
    status = _REASON_TO_STATUS.get(e.reason, _DEFAULT_REASON_STATUS)
    return HTTPException(status_code=status, detail={"reason": e.reason, "detail": e.detail})


class InvokeRequest(BaseModel):
    """POST /api/plugins/{model_tag}/invoke body.

    `path` is a route on the ALREADY-RESOLVED plugin container (e.g. "/transcribe"),
    not a URL — a plugin manifest's `capabilities` can list more than one operation,
    so the caller picks which one. Shape validation ("/"-prefixed, no "..") is
    invoke_plugin's own job per its contract, not re-validated here.
    """

    path: str
    payload: dict = Field(default_factory=dict)


@router.get("")
async def list_plugins(request: Request) -> dict:
    """Agent-facing plugin discovery. Hidden plugins omitted (matches /v1/models,
    /api/tags — still individually invokable by exact model_tag either way).
    NEVER returns host or port.
    """
    mgr = request.app.state.manager
    root = mgr.boot.storage.manifests_path
    registry = mgr.boot.plugins.registry
    out = []
    for tag in list_manifests(root):
        try:
            m = read_manifest(root, tag)
        except _UNREADABLE_MANIFEST as exc:
            # One unreadable manifest must not blank the whole listing, but a
            # skip must never be silent (same convention as the other list endpoints).
            log.warning("skipping unreadable manifest %r: %s", tag, type(exc).__name__)
            continue
        if not isinstance(m, PluginManifest):
            continue
        if m.hidden:
            continue
        # Registry resolution at load: computed HERE, via resolve_endpoint
        # (never a hand-rolled `resource_key in registry` check) — the point where
        # a loaded PluginManifest and the loaded boot registry are both already in
        # scope. Design choice: read-time report, not a write-time
        # gate in api/manifests.py — a hard gate there would make first-time setup
        # a chicken-and-egg (can't save a plugin manifest before its registry
        # entry exists).
        #
        # THE FIELD IS NAMED `configured`, AND THAT NAME IS
        # LOAD-BEARING. A name like `resolves` could report true for a capability
        # whose requests all fail: resolve_endpoint is a
        # REGISTRY LOOKUP + address-policy check and NEVER OPENS A SOCKET, so the
        # value would be honest about what it computes and misleading about what it is
        # called. It means exactly: an entry for this resource_key exists in the
        # boot registry AND its host is not in the never-a-valid-destination set.
        # It does NOT mean the container is up, and nothing here may ever make it
        # mean that — resolve_endpoint's own docstring
        # already states that reachability is invoke_plugin's concern, surfaced as
        # `unreachable` -> 502, established BY ACTUALLY CONNECTING at call time.
        # Do not add a probe to this listing: it is agent-facing and polled, and a
        # cached/stale `reachable: true` is the same lie with a timestamp on it.
        try:
            resolve_endpoint(m.resource_key, registry)
            configured = True
        except PluginInvokeError:
            configured = False
        out.append({
            "model_tag": m.model_tag,
            "lane": m.lane,
            "capabilities": m.capabilities,
            "configured": configured,
            # Self-description. Manifest-declared and manifest-
            # validated (PluginManifest._provides_routes_safe / _provides_
            # executables_safe) -- every entry is already known to be a
            # relative, scheme-free, traversal-free route by the time it is
            # loaded, so this cannot become a host/port leak channel: the
            # shapes that could carry a host CANNOT BE STORED, which is a
            # stronger property than redacting them here would be.
            "provides_routes": m.provides_routes,
            "provides_executables": m.provides_executables,
            # How to actually call the routes above. Without this an agent can
            # see that /analyze-audio exists and still have no idea that the
            # way to reach it is this route with {"path": ..., "payload": ...}
            # -- so the capability would otherwise stay effectively invisible.
            # A RELATIVE url by construction (an f-string over model_tag, which
            # TAG_RE has already constrained): this endpoint states WHAT to
            # call, never WHERE, and the registry stays the only place a
            # host/port lives.
            "invoke": {
                "method": "POST",
                "url": f"/api/plugins/{m.model_tag}/invoke",
                "body": {"path": "<one of provides_routes>", "payload": {}},
            },
        })
    return {"plugins": out, "total": len(out)}


@router.post("/{model_tag}/invoke")
async def invoke(model_tag: str, body: InvokeRequest, request: Request) -> dict:
    """Resolve model_tag -> PluginManifest -> resource_key, call
    invoke_plugin, return its JSON body inline. No duration timeout; a hang is
    what no_progress_timeout_s is for, not this route's job.

    No explicit validate_tag() pre-check: read_manifest -> _safe_manifest_path
    -> validate_tag already rejects traversal/invalid tags before any filesystem
    access (same reasoning api/models.py's single-model route documents for the
    identical omission) — the `except ManifestValidationError` branch below is
    what turns that rejection into a clean 400.
    """
    mgr = request.app.state.manager
    root = mgr.boot.storage.manifests_path
    try:
        m = read_manifest(root, model_tag)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"plugin not found: {model_tag}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (yaml.YAMLError, ValidationError, UnicodeDecodeError, OSError) as e:
        # Same handled-500 shape as chat_completion.py / embeddings.py /
        # models.py's get_model precedent. Fixed message only -- never str(e):
        # pydantic ValidationError echoes manifest field VALUES, yaml.YAMLError
        # leaks parser position/content. Full detail logged server-side instead.
        log.exception("manifest for %r is present but unreadable", model_tag)
        raise HTTPException(
            status_code=500,
            detail=f"manifest for '{model_tag}' is present but unreadable",
        ) from e

    if not isinstance(m, PluginManifest):
        raise HTTPException(
            status_code=400,
            detail=f"{model_tag!r} is not a plugin manifest (kind={m.kind!r})",
        )

    try:
        result = await invoke_plugin(
            resource_key=m.resource_key,
            registry=mgr.boot.plugins.registry,
            path=body.path,
            payload=body.payload,
            no_progress_timeout_s=mgr.runtime.plugin_runtime.no_progress_timeout_s,
        )
    except PluginInvokeError as e:
        raise _plugin_invoke_error_to_http(e) from e

    return result
