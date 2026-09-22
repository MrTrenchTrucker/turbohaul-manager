"""Turbohaul LOAD_VERIFY observability (DISPLAY / OBSERVABILITY ONLY).

WHY THIS MODULE EXISTS
======================
A live sub-agent handoff exposed a manager blind spot: after a model
(re)spawn the manager POSTs a KV restore, gets ``200``, and *trusts* it — it
logs "engine determines actual n_past" but never checks the engine's ACTUAL
``n_past``. Worse, a silently-dead ``llama-server`` kept being treated as a
live idle-hot resident (``alive=False idle_match=True``) for 10+ minutes and
main never re-spawned. We were blind to whether the model + precomputed KV
truly loaded.

This module is the OBSERVABILITY half of the fix:

  * ``log_load_verify(**fields)`` — emits ONE greppable ``LOAD_VERIFY {json}``
    log line (mirrors the existing ``R2B_REQ_IDENTITY`` / ``KV_RESTORE``
    lines) and stashes the last N records in a module-level ring for a
    ``/status`` read surface.
  * ``verify_model_resident(handle)`` / ``verify_kv_restored(handle, slot_id,
    expected_tokens)`` — PURE async read helpers (os pid-alive + engine
    ``/health`` + ``/slots``) that the root fix CALLS to decide
    whether a load truly took and whether to retry. They make NO decision and
    have NO side effects.

HARD BOUNDARY: this module NEVER changes KV save/restore/gate/unload behavior.
It reads and reports. The retry loop, unload-timing and ``final_status``
verdict are populated by the manager root-fix and passed *into* the emitter;
the emitter only records what it is given. The module keeps its state at
MODULE scope (the ring below) — it adds NO new instance state to the manager,
so it cannot collide with concurrent edits to ``manager.py``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level ring buffer (last N LOAD_VERIFY records) for the /status surface.
# Lives here — NOT on the manager instance — so this module adds zero new
# manager state and cannot conflict with the root-fix edits to manager.py.
# ---------------------------------------------------------------------------
_RING_MAX = 64
_ring: "deque[dict]" = deque(maxlen=_RING_MAX)
_ring_lock = threading.Lock()

# The schema fields, in emit order. Mirrors the brief's LOAD_VERIFY schema.
_FIELDS = (
    "event",              # "model_load" | "kv_restore"
    "trigger",            # "spawn" | "model_swap" | "reload" | "cold" | "warm"
    "model_tag",
    "port",
    "pid",
    "process_alive",      # os-level pid alive
    "health_200",         # GET /health returned ok
    "model_resident",     # /slots|/props confirm model loaded + n_ctx>0
    "kv_expected_tokens",
    "kv_actual_n_past",   # engine-reported context depth AFTER restore
    "kv_restore_ok",      # actual >= expected * threshold
    "retry_count",
    "final_status",       # "ok" | "retried_ok" | "failed"
    "reason",
    "thread_hash",
    "session_id",
)


def log_load_verify(
    *,
    event: str,
    trigger: str,
    model_tag: str,
    port: int,
    pid: int | None = None,
    process_alive: bool | None = None,
    health_200: bool | None = None,
    model_resident: bool | None = None,
    kv_expected_tokens: int | None = None,
    kv_actual_n_past: int | None = None,
    kv_restore_ok: bool | None = None,
    retry_count: int = 0,
    final_status: str = "ok",
    reason: str | None = None,
    thread_hash: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Emit ONE ``LOAD_VERIFY`` json log line + stash it in the module ring.

    Returns the recorded dict (so the caller can reuse it). Best-effort and
    NONE-safe — a malformed field can NEVER raise into the spawn/restore path
    (mirrors the best-effort ``_emit_request_identity`` pattern). ``retry_count``
    / ``final_status`` are supplied BY the caller's verify+retry loop; this
    function does not own or drive any retry logic.
    """
    record: dict[str, Any] = {
        "event": event,
        "trigger": trigger,
        "model_tag": model_tag,
        "port": port,
        "pid": pid,
        "process_alive": process_alive,
        "health_200": health_200,
        "model_resident": model_resident,
        "kv_expected_tokens": kv_expected_tokens,
        "kv_actual_n_past": kv_actual_n_past,
        "kv_restore_ok": kv_restore_ok,
        "retry_count": retry_count,
        "final_status": final_status,
        "reason": reason,
        "thread_hash": thread_hash,
        "session_id": session_id,
    }
    try:
        log.info("LOAD_VERIFY %s", json.dumps(record))
    except Exception:  # noqa: BLE001 — display-only, never fail a load
        log.debug("LOAD_VERIFY emit failed (ignored)", exc_info=True)
    try:
        with _ring_lock:
            # Store a COPY, not the returned object — the caller may reuse/mutate
            # the returned record (docstring invites it); a live alias in the ring
            # could be mutated mid-``json.dumps`` on the /status path.
            _ring.append(dict(record))
    except Exception:  # noqa: BLE001 — ring is observability only
        log.debug("LOAD_VERIFY ring append failed (ignored)", exc_info=True)
    return record


def get_recent(n: int | None = None) -> list[dict]:
    """Return the last ``n`` LOAD_VERIFY records (newest last), for ``/status``.

    Copies the ring under the lock so the caller gets a stable snapshot. ``n``
    defaults to the whole ring.
    """
    with _ring_lock:
        items = list(_ring)
    if n is not None and n >= 0:
        # n==0 must mean "none" — items[-0:] == items[0:] == everything (edge case).
        items = items[-n:] if n else []
    return items


def clear_ring() -> None:
    """Test/ops helper — drop all buffered records. No effect on behavior."""
    with _ring_lock:
        _ring.clear()


# ---------------------------------------------------------------------------
# PURE read helpers — the root fix CALLS these. No side effects.
# ---------------------------------------------------------------------------
def _pid_alive(pid: int | None) -> bool:
    """os-level liveness of ``pid`` (signal 0). False on None / dead / gone."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another uid — still alive.
        return True
    except OSError:
        return False
    return True


def _handle_port(handle: Any) -> int | None:
    """Duck-type the manager handle -> engine port. Accepts a handle object
    with ``.port``, or a bare int port."""
    if isinstance(handle, int):
        return handle
    return getattr(handle, "port", None)


def _handle_pid(handle: Any) -> int | None:
    return None if isinstance(handle, int) else getattr(handle, "pid", None)


async def verify_model_resident(
    handle: Any, *, timeout: float = 2.0, mlx: bool = False
) -> dict:
    """PURE read: is the engine process alive AND the model actually resident?

    Reads: os pid-alive (``handle.pid``), ``GET /health`` (health_200), and
    the engine's model list.

    - llama.cpp (default): ``GET /slots`` (model_resident = 200 and a slot with
      ``n_ctx`` > 0).
    - MLX (``mlx=True``): ``mlx_lm server`` has **no** ``/slots`` endpoint, so we
      probe ``GET /v1/models`` instead and treat the model as resident when the
      sidecar is healthy AND it reports at least one model. (Full-path/id
      matching is intentionally loose — the sidecar serves exactly one model.)

    Returns ``{process_alive, health_200, model_resident, n_ctx, port, pid,
    reason}``. Never raises — on any error the booleans are False and
    ``reason`` carries the cause.
    """
    port = _handle_port(handle)
    pid = _handle_pid(handle)
    out: dict[str, Any] = {
        "process_alive": _pid_alive(pid),
        "health_200": False,
        "model_resident": False,
        "n_ctx": None,
        "port": port,
        "pid": pid,
        "reason": None,
    }
    if not port:
        out["reason"] = "no port on handle"
        return out
    base = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            h = await client.get(f"{base}/health")
            out["health_200"] = h.status_code == 200
            if mlx:
                # mlx_lm server: no /slots. /v1/models lists the loaded model(s).
                m = await client.get(f"{base}/v1/models")
                if m.status_code == 200:
                    try:
                        models = m.json().get("data", [])
                    except Exception:
                        models = []
                    if isinstance(models, list) and models:
                        out["model_resident"] = bool(out["health_200"])
                    else:
                        out["reason"] = "empty /v1/models"
                else:
                    out["reason"] = f"/v1/models http {m.status_code}"
            else:
                s = await client.get(f"{base}/slots")
                if s.status_code == 200:
                    slots = s.json()
                    if isinstance(slots, list) and slots:
                        n_ctx = slots[0].get("n_ctx")
                        out["n_ctx"] = n_ctx
                        out["model_resident"] = bool(out["health_200"] and n_ctx and n_ctx > 0)
                    else:
                        out["reason"] = "empty /slots"
                else:
                    out["reason"] = f"/slots http {s.status_code}"
    except Exception as e:  # noqa: BLE001 — pure read, never raise into caller
        out["reason"] = f"{type(e).__name__}: {e}"
    return out


async def verify_kv_restored(
    handle: Any,
    slot_id: int,
    expected_tokens: int | None,
    *,
    actual_n_past: int | None = None,
    threshold: float = 0.98,
    timeout: float = 2.0,
) -> dict:
    """PURE read: did the KV restore actually land the expected token depth?

    Authoritative read (the normal path) is the slot's ``n_prompt_tokens`` from
    ``GET /slots`` — it reliably reflects the restored context depth.
    ``actual_n_past`` is an OPTIONAL override: if the caller already holds the
    engine's restored-token count from the ``action=restore`` response, pass it
    and it wins. ``kv_restore_ok`` is ``actual >= expected * threshold``.
    Returns ``{kv_expected_tokens, kv_actual_n_past, kv_restore_ok, source,
    reason}`` (``source`` = "caller" override or "slots" read). Never raises.
    """
    out: dict[str, Any] = {
        "kv_expected_tokens": expected_tokens,
        "kv_actual_n_past": actual_n_past,
        "kv_restore_ok": None,
        "source": "caller" if actual_n_past is not None else None,
        "reason": None,
    }
    if actual_n_past is None:
        port = _handle_port(handle)
        if not port:
            out["reason"] = "no port on handle and no actual_n_past passed"
            return out
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                s = await client.get(f"{base}/slots")
                if s.status_code == 200:
                    slots = s.json()
                    match = None
                    if isinstance(slots, list):
                        match = next((sl for sl in slots if sl.get("id") == slot_id), None)
                    if match is not None:
                        out["kv_actual_n_past"] = match.get("n_prompt_tokens")
                        out["source"] = "slots"
                    else:
                        out["reason"] = f"slot {slot_id} not found in /slots"
                        return out
                else:
                    out["reason"] = f"/slots http {s.status_code}"
                    return out
        except Exception as e:  # noqa: BLE001 — pure read, never raise
            out["reason"] = f"{type(e).__name__}: {e}"
            return out

    # NOTE: this arithmetic tail runs OUTSIDE the httpx try — the engine's
    # n_prompt_tokens (or a caller-passed override) is UNTRUSTED, and a str/None
    # would make ``>=`` / ``<=`` raise TypeError straight into the caller's
    # restore+retry loop (the contract's hard "never raise into restore" line).
    # Guard both operands as real numbers (bool excluded — it's an int subclass).
    def _num(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    actual = out["kv_actual_n_past"]
    if not _num(expected_tokens) or expected_tokens <= 0:
        # Nothing meaningfully expected (cold fresh / bad expected) — a present
        # numeric context is trivially ok.
        out["kv_restore_ok"] = _num(actual)
        return out
    if not _num(actual):
        out["kv_restore_ok"] = False
        out["reason"] = out["reason"] or f"non-numeric n_past ({type(actual).__name__})"
        return out
    out["kv_restore_ok"] = actual >= expected_tokens * threshold
    return out
