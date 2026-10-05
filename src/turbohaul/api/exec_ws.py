"""WS exec forwarder (hop 1 of 2) — plugin-native EXEC routing.

The engine image's generic PATH shim (a single
implementation installed under whatever executable name it serves) connects to
/api/plugins/{model_tag}/exec on the manager's own loopback; this route
resolves the tag EXACTLY like the JSON invoke route — read_manifest ->
resource_key -> registry -> resolve_endpoint (the SSRF-guard chokepoint,
shared by every plugin route) — then relays WebSocket frames bidirectionally to the target
plugin's /ws/exec. The forwarder is stateless and near-opaque: frames pass
through unmodified; it only (a) resolves + rejects pre-accept,
(b) extracts run_id from the first text frame for its own log lines,
(c) injects the protocol's dedicated target-lost error frame (code 101)
if the target connection dies mid-stream, and (d) closes both ends when
either side goes away (the target's own disconnect handler then reaps its
child). It is binary-agnostic by construction: the binary
is a parameter in the start frame; the security property (the exec allow-
list) lives in the target endpoint, not here.

Forwarder for the exec protocol, registered in api/main.py: this file
implements the forwarding contract. Known seam:
when auth lands on the invoke path, this
route needs the same gate (the optional EXEC_PLUGIN_AUTH_TOKEN env var is
the forwarder's half of that; the endpoint's half is the verify_key
include in the plugin's exec endpoint, which fails OPEN while /secrets/api_key
is absent and CLOSED when it is).

Frame protocol: the start frame names the binary; error frames carry type, code, stderr and run_id; other frames pass through unmodified.
"""
import asyncio
import ipaddress
import json
import logging
import os
from typing import TYPE_CHECKING

import yaml
from fastapi import APIRouter, WebSocket
from pydantic import ValidationError as PydValidationError

from turbohaul.manifest import ManifestValidationError, PluginManifest, read_manifest
from turbohaul.plugin_invoke import PluginInvokeError, resolve_endpoint

if TYPE_CHECKING:
    from turbohaul.manager import TurbohaulManager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plugins", tags=["plugins"])

# This route handles four of read_manifest's failure modes (FileNotFoundError /
# ManifestValidationError / yaml.YAMLError / pydantic ValidationError), the first
# four entries of api/plugins.py's _UNREADABLE_MANIFEST (that tuple also lists OSError
# and UnicodeDecodeError); restated here, not imported, since that file keeps it
# file-local. The WS route rejects the upgrade on any of them (pre-accept close = HTTP 403 to the shim).
_UNREADABLE_MANIFEST = (
    FileNotFoundError,
    ManifestValidationError,
    yaml.YAMLError,
    PydValidationError,
)

# The target-side path the plugin endpoint exposes (for example the whisperx
# plugin's exec endpoint). Fixed by the contract, not a caller-supplied value —
# no path injection. Generic by design: the endpoint is plugin-local but the
# path is part of the shared frame protocol (the mechanism
# is plugin-generic; this route never sees or cares which binary is named).
_TARGET_PATH = "/ws/exec"

# TCP connect to the target only; mirrors plugin_invoke._CONNECT_TIMEOUT_S
# (deliberately independent of any duration timeout — an internal-network
# connect either succeeds fast or the target is unreachable).
_CONNECT_TIMEOUT_S = 10.0

# Protocol code reserved for "the forwarder lost its target connection
# mid-stream". Distinct from the endpoint's own spawn/mapping errors
# (127/2/70...) which this route forwards verbatim — a consumer can tell
# "target process refused the work" from "target container vanished" from
# the exit code alone.
_TARGET_LOST_CODE = 101


class _DeclaredRefusal(Exception):
    """The forwarder's own policy refusal (hop-1): the start frame names a
    binary the plugin manifest does not declare. Raised by the client pump
    AFTER the error frame has been delivered to the client; the coordinator
    treats it as a clean (non-target-lost) finish."""


def _declared_violation(manifest, binary):
    """Consumption of the manifest schema extension for declared executables:
    if the plugin manifest DECLARES
    `provides_executables` (a non-empty list), a start frame naming a binary
    outside that declaration is refused at hop 1 — defense in depth in front
    of the target endpoint's own allow-list (which remains THE security
    property; this check only fails earlier and closer to the caller).
    Manifests without the field (or with an empty list) impose no hop-1
    restriction — the target's allow-list is still the gate.
    Returns the violating binary, or None when the binary is declared (or
    undeclared-relevant)."""
    declared = getattr(manifest, "provides_executables", None)
    if not declared or not isinstance(declared, (list, tuple)):
        return None
    if isinstance(binary, str) and binary not in declared:
        return binary
    return None


def _host_for_url(host: str) -> str:
    """Bracket an IPv6 literal for URL construction — same precedent as
    plugin_invoke.invoke_plugin (its _parse_ip_literal strips a pre-existing
    bracket pair first, so an already-bracketed host is never double-wrapped;
    the check below replicates that for the URL-only purpose)."""
    unwrapped = host
    if len(host) >= 2 and host.startswith("[") and host.endswith("]"):
        unwrapped = host[1:-1]
    try:
        ip = ipaddress.ip_address(unwrapped)
    except ValueError:
        return host  # hostname — the registry validator already checked it
    return f"[{ip}]" if isinstance(ip, ipaddress.IPv6Address) else host


async def _pump_source_to_target(
    source: WebSocket, target, run_id_box: list, manifest
) -> None:
    """Client (shim) -> target. First text frame is parsed ONLY for run_id
    (logging) and the hop-1 declaration check (provides_executables); every
    frame is forwarded byte-identical regardless."""
    first = True
    try:
        while True:
            msg = await source.receive()
            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                return
            if mtype != "websocket.receive":
                continue
            if msg.get("bytes") is not None:
                await target.send(msg["bytes"])
            elif msg.get("text") is not None:
                if first:
                    first = False
                    try:
                        d = json.loads(msg["text"])
                        run_id_box[0] = str(d.get("run_id") or "?")
                    except (ValueError, TypeError):
                        d = None
                        run_id_box[0] = "?"  # forward opaquely; the target validates
                    # hop-1 declaration check (only when the manifest declares
                    # executables; the target's allow-list remains the gate):
                    if isinstance(d, dict):
                        viol = _declared_violation(manifest, d.get("binary"))
                        if viol is not None:
                            declared = getattr(manifest, "provides_executables", None)
                            log.warning("exec ws: refusing at hop 1 — binary %r "
                                        "not declared by manifest (declares %s) "
                                        "run_id=%s", viol, declared, run_id_box[0])
                            try:
                                await source.send_json({
                                    "type": "error", "code": 127,
                                    "stderr": f"binary {viol!r} not declared by "
                                              "this plugin's manifest "
                                              f"(provides_executables={list(declared)})",
                                    "run_id": run_id_box[0],
                                })
                                await source.close(code=1008)
                            except Exception:
                                pass
                            raise _DeclaredRefusal(viol)
                await target.send(msg["text"])
    except Exception:
        # target died mid-forward: the coordinator turns this into the
        # target-lost error frame on the client side.
        raise


async def _pump_target_to_client(target, dest: WebSocket) -> None:
    """Target (endpoint) -> client. Pure bytes-through; the target's own
    frames (meta/ready/exit/error) are forwarded unmodified. Any raise here
    (ConnectionClosed or transport error) = target lost mid-stream."""
    async for frame in target:
        if isinstance(frame, (bytes, bytearray)):
            await dest.send_bytes(bytes(frame))
        else:
            await dest.send_text(frame)


@router.websocket("/{model_tag}/exec")
async def plugin_exec_ws(model_tag: str, websocket: WebSocket) -> None:
    mgr: "TurbohaulManager" = websocket.app.state.manager

    # ---- resolve BEFORE accept: the exact chokepoint of the JSON invoke
    # route (read_manifest -> resource_key -> registry -> resolve_endpoint,
    # where the SSRF guard lives). A pre-accept close makes the ASGI server
    # answer the upgrade with HTTP 403 — the shim's connect() fails loud.
    endpoint = None
    try:
        manifest = read_manifest(mgr.boot.storage.manifests_path, model_tag)
        if not isinstance(manifest, PluginManifest):
            raise ValueError(f"{model_tag!r} is not a plugin manifest")
        endpoint = resolve_endpoint(manifest.resource_key, mgr.boot.plugins.registry)
    except _UNREADABLE_MANIFEST as e:
        log.warning("exec ws: refusing upgrade model_tag=%r unreadable manifest "
                    "(%s: %s)", model_tag, type(e).__name__, str(e)[:200])
        await websocket.close(code=1008)
        return
    except (ValueError, PluginInvokeError) as e:
        log.warning("exec ws: refusing upgrade model_tag=%r (%s: %s)",
                    model_tag, type(e).__name__, str(e)[:200])
        await websocket.close(code=1008)
        return

    auth_token = os.environ.get("EXEC_PLUGIN_AUTH_TOKEN", "")
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
    target_url = (f"ws://{_host_for_url(endpoint.host)}:{endpoint.port}"
                  f"{_TARGET_PATH}")

    await websocket.accept()
    run_id_box: list = ["?"]
    target = None
    try:
        from websockets.asyncio.client import connect as ws_connect

        try:
            target = await ws_connect(target_url, open_timeout=_CONNECT_TIMEOUT_S,
                                      additional_headers=headers)
        except Exception as e:
            # Address goes to the server log ONLY (the plugins API contract:
            # callers never see the registry's host:port).
            log.error("exec ws: target unreachable model_tag=%r resource=%r "
                      "endpoint=%s:%s error=%s: %s", model_tag,
                      manifest.resource_key, endpoint.host, endpoint.port,
                      type(e).__name__, e)
            await websocket.send_json({
                "type": "error", "code": 1,
                "stderr": "media target unreachable (see turbohaul server log); "
                          "run cannot be served",
                "run_id": None,
            })
            await websocket.close(code=1011)
            return

        pump_ct = asyncio.create_task(
            _pump_source_to_target(websocket, target, run_id_box, manifest))
        pump_tc = asyncio.create_task(
            _pump_target_to_client(target, websocket))
        done, pending = await asyncio.wait(
            {pump_ct, pump_tc}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        # Which side died, and what the client is owed:
        target_lost = pump_tc in done and pump_ct not in done
        # (pump_tc completing normally without pump_ct = the target closed
        # its side — the endpoint does that only AFTER its terminal frame,
        # so a clean target close is NOT an error; a target EXCEPTION IS.)
        target_exc = pump_tc.exception() if (pump_tc in done
                                             and not pump_tc.cancelled()) else None
        client_exc = pump_ct.exception() if (pump_ct in done
                                             and not pump_ct.cancelled()) else None
        if target_exc is not None:
            log.error("exec ws: target lost mid-stream model_tag=%r run_id=%s: "
                      "%s: %s", model_tag, run_id_box[0],
                      type(target_exc).__name__, target_exc)
            try:
                await websocket.send_json({
                    "type": "error", "code": _TARGET_LOST_CODE,
                    "stderr": "media target connection lost mid-stream "
                              "(see turbohaul server log); partial output was "
                              "dropped to the clean-frame boundary by the shim",
                    "run_id": run_id_box[0],
                })
            except Exception:
                pass
            await websocket.close(code=1011)
        else:
            # Clean finish (target closed after its terminal frame), a
            # client-side policy refusal (_DeclaredRefusal: the error frame
            # already went to the client, it is just closing), or the client
            # went away: close the far end; the endpoint's own disconnect
            # handler reaps its child and temp file.
            if client_exc is not None:
                if isinstance(client_exc, _DeclaredRefusal):
                    log.warning("exec ws: refused at hop 1 (binary not declared "
                                "by manifest) model_tag=%r run_id=%s",
                                model_tag, run_id_box[0])
                else:
                    log.warning("exec ws: client pump failed model_tag=%r run_id=%s: "
                                "%s: %s", model_tag, run_id_box[0],
                                type(client_exc).__name__, client_exc)
            try:
                await target.close()
            except Exception:
                pass
            try:
                await websocket.close(code=1000)
            except Exception:
                pass
    finally:
        # target.close() is a COROUTINE on the websockets asyncio client —
        # a bare call would create a never-awaited coroutine and the socket
        # would stay open. None covers the ws_connect-failure path.
        if target is not None:
            try:
                await target.close()
            except Exception:
                pass
