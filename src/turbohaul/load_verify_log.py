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
  * ``verify_model_identity(handle, expected_model_path, engine_log_path=...)``
    -- PURE async read helper: does the engine's OWN reported
    identity (``GET /props``' ``model_path``, falling back to the engine's own
    logged ``loading model '<path>'`` line) match what was actually requested.
    ``verify_model_resident`` above can only prove *A* model is up; this is
    the check for WHICH one.
  * ``scan_engine_log_for_errors(engine_log_path)`` -- PURE sync
    read helper: does the engine's own log for this spawn contain any E-level
    line, so a degraded-but-"healthy" load is never silently reported clean.
  * ``classify_idle_dead_holder(engine_log_path)`` -- PURE
    sync read helper: when the manager's own dead-idle sweep (worker_loop,
    stuck-handle step) finds an idle-hot holder already dead, this reads
    the SAME engine_log_path to say WHICH of the two shapes it was — the
    engine announcing its own exit vs. a fatal signature with no announcement
    at all — using the same file this module already reads for
    ``scan_engine_log_for_errors``. Naming only; makes no decision and has no
    side effects on the teardown/recovery path that follows it.

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
import re
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
    "trigger",            # "spawn" | "model_swap" | "wave_return" | "cold" | "warm"
    "model_tag",
    "port",
    "pid",
    "process_alive",      # os-level pid alive
    "health_200",         # GET /health returned ok. On a kv_restore record this is
                          # NEVER independently re-measured -- None, not a copy of
                          # the paired model_load record's own reading.
    "model_resident",     # health_200 and n_ctx>0 -- proves *A* model is resident,
                          # NEVER which one. (It does not mean "/slots|/props
                          # confirm model loaded" -- /props is not consulted
                          # here; see identity_verified below for the real check.)
                          # Same never-re-measured-on-kv_restore rule as health_200.
    "identity_verified",  # True/False -- the engine's OWN reported
                          # identity (model_path via /props, or its own "loading
                          # model '<path>'" log line as fallback) compared to the
                          # model actually requested at spawn. None = could not be
                          # established (see identity_reason). Only ever set on
                          # model_load records.
    "identity_source",    # "props" | "log" | None -- which mechanism produced identity_verified.
    "identity_reason",    # why identity_verified is None, or detail of a mismatch.
    "served_model_path",  # the path the ENGINE itself reported/logged, if any.
    "expected_model_path",  # the path the manager requested at spawn time.
    "kv_expected_tokens",
    "kv_actual_n_past",   # engine-reported context depth AFTER restore
    "kv_restore_ok",      # actual >= expected * threshold; None if never verified
    "restore_attempted",  # was there a real expectation to check at all (bool)
    "kv_expected_reason", # why kv_expected_tokens is null (the caller's side) --
                          # populated independently of kv_actual_reason, never
                          # first-write-wins against it.
    "kv_actual_reason",   # why kv_actual_n_past is null (the ENGINE's
                          # side) -- populated independently of kv_expected_reason.
    "retry_count",
    "final_status",       # "ok" | "retried_ok" | "failed" | "unverified" | None
    "reason",
    "source",              # provenance of kv_actual_n_past: "caller" | "slots" | None
    "slots_http_status",   # raw GET /slots status code; None if no response at all
    "slot_id_found",       # was the requested slot id present in /slots; None if never read
    "engine_errors_detected",  # True if the engine's own log for this
                          # spawn contains any E-level line; False if checked and
                          # clean; None if the log could not be read at all.
    "engine_error_lines",  # up to a few of the actual matched E-level lines (evidence).
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
    identity_verified: bool | None = None,
    identity_source: str | None = None,
    identity_reason: str | None = None,
    served_model_path: str | None = None,
    expected_model_path: str | None = None,
    kv_expected_tokens: int | None = None,
    kv_actual_n_past: int | None = None,
    kv_restore_ok: bool | None = None,
    restore_attempted: bool | None = None,
    kv_expected_reason: str | None = None,
    kv_actual_reason: str | None = None,
    retry_count: int = 0,
    final_status: str | None = None,
    reason: str | None = None,
    source: str | None = None,
    slots_http_status: int | None = None,
    slot_id_found: bool | None = None,
    engine_errors_detected: bool | None = None,
    engine_error_lines: list[str] | None = None,
    thread_hash: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Emit ONE ``LOAD_VERIFY`` json log line + stash it in the module ring.

    Returns the recorded dict (so the caller can reuse it). Best-effort and
    NONE-safe — a malformed field can NEVER raise into the spawn/restore path
    (mirrors the best-effort ``_emit_request_identity`` pattern). ``retry_count``
    / ``final_status`` are supplied BY the caller's verify+retry loop; this
    function does not own or drive any retry logic.

    ``final_status`` has NO success-shaped default (it is ``None``, not ``"ok"``
    -- an omitted status must read as "nothing was asserted", never as a pass).
    That alone is defence in depth: a caller can still choose to
    pass a literal ``"ok"``. The load-bearing guarantee is the check below --
    on a ``kv_restore`` record, a passing ``final_status`` is STRUCTURALLY
    incompatible with ``kv_restore_ok`` reading anything other than ``True``
    (``False`` or ``None`` both count -- a verification that was never asked
    is exactly as untrustworthy as one that failed). A caller cannot emit that
    combination even if it tries. The coerced value is DERIVED from
    ``kv_restore_ok``, not flattened to one catch-all string: ``False``
    (actively fell short) coerces to ``"failed"``; ``None`` (never
    established) coerces to ``"unverified"``. Deliberately verdict-aware --
    a coercion that flattened both to the same value would silently
    DEGRADE a real "failed" into a merely-unchecked "unverified", which is
    real information loss, not just a defended-against gap. Never silently
    dropped or raised
    (this function's never-raise contract holds). Scoped to
    ``event == "kv_restore"`` only -- a ``model_load`` record legitimately
    never sets ``kv_restore_ok`` at all, and must not be downgraded for a
    field that was never relevant to it.
    """
    _PASSING_STATUSES = {"ok", "retried_ok"}
    if (
        event == "kv_restore"
        and kv_restore_ok is not True
        and final_status in _PASSING_STATUSES
    ):
        _coerced = "failed" if kv_restore_ok is False else "unverified"
        if reason is None:
            reason = (
                f"final_status={final_status!r} coerced to {_coerced!r}: "
                f"kv_restore_ok={kv_restore_ok!r} is not True"
            )
        final_status = _coerced
    record: dict[str, Any] = {
        "event": event,
        "trigger": trigger,
        "model_tag": model_tag,
        "port": port,
        "pid": pid,
        "process_alive": process_alive,
        "health_200": health_200,
        "model_resident": model_resident,
        "identity_verified": identity_verified,
        "identity_source": identity_source,
        "identity_reason": identity_reason,
        "served_model_path": served_model_path,
        "expected_model_path": expected_model_path,
        "kv_expected_tokens": kv_expected_tokens,
        "kv_actual_n_past": kv_actual_n_past,
        "kv_restore_ok": kv_restore_ok,
        "restore_attempted": restore_attempted,
        "kv_expected_reason": kv_expected_reason,
        "kv_actual_reason": kv_actual_reason,
        "retry_count": retry_count,
        "final_status": final_status,
        "reason": reason,
        "source": source,
        "slots_http_status": slots_http_status,
        "slot_id_found": slot_id_found,
        "engine_errors_detected": engine_errors_detected,
        "engine_error_lines": list(engine_error_lines) if engine_error_lines else engine_error_lines,
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


def _num(x: Any) -> bool:
    """Real number, bool excluded (it's an int subclass). Note: the
    engine's n_prompt_tokens (or a caller-passed override) is UNTRUSTED, and
    a str/None would make >=/<= raise TypeError straight into the caller's
    restore+retry loop (this module's hard "never raise into restore"
    line) -- every arithmetic comparison in this module must guard through
    this first."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _kv_verdict(actual: Any, expected: Any, threshold: float) -> bool | None:
    """The SOLE place ``kv_restore_ok`` is computed, so "unmeasurable" can
    never be reported as a terminal verdict. The same rule is applied in
    three places in this module (``final_status``'s success-shaped
    default, ``verify_kv_restored``'s no-expectation branch, and this
    function's handling of a non-numeric ``actual``). The
    invariant, enforced BY CONSTRUCTION because this is the only function
    permitted to return a bool here:

        kv_restore_ok is False  <=>  a real number was obtained for BOTH
        actual and expected, AND actual fell short of expected * threshold.

    ``None`` means "not measurable" -- no usable expectation, no usable
    actual, or both -- and is returned in EVERY other case, never
    fabricated. ``None`` already means "refuses to claim success" (via the
    ``final_status`` coercion to ``"unverified"``); ``False`` additionally claims
    the restore FAILED, which must only ever be true when a real comparison
    was actually made -- "could not measure" and "measured and it fell
    short" are different facts, and conflating them turns a quiet wrong
    ``"ok"`` into a loud wrong ``"failed"`` instead of fixing anything. Never
    raises."""
    if not _num(expected) or expected <= 0:
        return None
    if not _num(actual):
        return None
    return actual >= expected * threshold


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

    Returns ``{kv_expected_tokens, kv_actual_n_past, kv_restore_ok,
    restore_attempted, source, reason, kv_expected_reason, kv_actual_reason,
    process_alive, slots_http_status, slot_id_found}``.

    ``kv_expected_reason`` / ``kv_actual_reason``: the shared
    ``reason`` field above is first-write-wins -- when BOTH sides are null
    at once (no expectation supplied by the caller, AND the engine's own
    read comes back empty for an unrelated cause), only whichever cause
    happened to be written first ever reaches ``reason``, silently dropping
    the other. These two fields never share that race: ``kv_expected_reason``
    is decided entirely from ``expected_tokens`` up front (the CALLER's side of the
    story), and ``kv_actual_reason`` is set independently at whichever engine-
    side branch actually leaves ``kv_actual_n_past`` null (the ENGINE's side).
    Both are ``None`` exactly when their own field needs no explanation.

    ``restore_attempted`` and ``kv_restore_ok`` answer TWO DIFFERENT
    questions and must never be conflated: ``restore_attempted`` is "was
    there a real expectation to check against" (a positive numeric
    ``expected_tokens``); ``kv_restore_ok`` is "did the actual depth meet
    it". When there is NO usable expectation (``expected_tokens`` absent or
    non-positive -- a cold-fresh session with nothing to restore, or a bad
    caller value), ``kv_restore_ok`` is ``None`` -- NEVER derived from
    whether ``kv_actual_n_past`` merely happens to hold a number. Answering
    "is there a number here" and publishing it as "did the restore succeed"
    would be a fabricated verdict: two different questions collapsed onto
    one field, indistinguishable from a real failure once ``final_status``
    renders it.

    ``kv_actual_n_past is None`` is unambiguous across the four ways it can
    happen via ``process_alive``, ``slots_http_status``, ``slot_id_found``
    -- engine down (``process_alive=False``), slot absent
    (``slot_id_found=False``), read failed (``slots_http_status`` is a
    non-200, or ``None`` if the request never got a response at all), or read
    never attempted (``actual_n_past`` was overridden by the caller, or no
    port on the handle). ``process_alive`` is an independent pid-liveness
    check (mirrors ``verify_model_resident``'s own convention: always a real
    bool, never ``None``, even without a port) -- ``slots_http_status`` and
    ``slot_id_found`` are the ones that stay ``None`` when the /slots read
    itself was never attempted, meaning "not applicable" there, not
    "unknown". ``reason`` always carries a human-readable restatement of the
    same fact when ``kv_actual_n_past`` is ``None`` -- check the structured
    fields for programmatic classification, or ``reason`` for a human, never
    infer from the bare ``None`` alone. Never raises.
    """

    _restore_attempted = _num(expected_tokens) and expected_tokens > 0
    out: dict[str, Any] = {
        "kv_expected_tokens": expected_tokens,
        "kv_actual_n_past": actual_n_past,
        "kv_restore_ok": None,
        # "was there a real expectation" (attempted) is answerable from
        # expected_tokens alone, independent of whether the VERIFICATION
        # read below succeeds -- computed once, up front, never revised.
        "restore_attempted": _restore_attempted,
        "source": "caller" if actual_n_past is not None else None,
        "reason": None,
        # kv_expected_reason is the CALLER's side of the null-reason
        # story and is decided ENTIRELY from expected_tokens, right here --
        # it can never be overwritten or skipped by whatever happens to the
        # engine-side read below (a single shared `reason` slot would let a
        # first-write-wins engine-side reason silently hide this one whenever
        # both were null at once).
        "kv_expected_reason": (
            None if _restore_attempted else
            "no expectation supplied (nothing to restore against)" if expected_tokens is None else
            f"expected_tokens={expected_tokens!r} is not a usable positive expectation"
        ),
        # The ENGINE-side counterpart, set independently at
        # each branch below that leaves kv_actual_n_past null. Never derived
        # from kv_expected_reason or vice versa.
        "kv_actual_reason": None,
        "process_alive": None,
        "slots_http_status": None,
        "slot_id_found": None,
    }
    if actual_n_past is None:
        port = _handle_port(handle)
        # Engine liveness AT THIS INSTANT, on this same record --
        # verify_model_resident captures this only for a DIFFERENT event
        # (model_load) at a DIFFERENT time. Without this, a kv_restore record
        # would have zero liveness signal of its own.
        out["process_alive"] = _pid_alive(_handle_pid(handle))
        if not port:
            out["reason"] = "no port on handle and no actual_n_past passed"
            out["kv_actual_reason"] = out["reason"]
            return out
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                s = await client.get(f"{base}/slots")
                out["slots_http_status"] = s.status_code
                if s.status_code == 200:
                    slots = s.json()
                    match = None
                    if isinstance(slots, list):
                        match = next((sl for sl in slots if sl.get("id") == slot_id), None)
                    out["slot_id_found"] = match is not None
                    if match is not None:
                        out["kv_actual_n_past"] = match.get("n_prompt_tokens")
                        out["source"] = "slots"
                        if out["kv_actual_n_past"] is None:
                            # The matching slot exists but its own record
                            # carries no "n_prompt_tokens" (missing key, or
                            # explicit null) -- an absent number must never be
                            # silently unexplained; name it rather than let a
                            # bare None with no reason look identical to "no
                            # attempt was made".
                            out["reason"] = f"slot {slot_id} present but n_prompt_tokens missing/null"
                            out["kv_actual_reason"] = out["reason"]
                    else:
                        out["reason"] = f"slot {slot_id} not found in /slots"
                        out["kv_actual_reason"] = out["reason"]
                        return out
                else:
                    out["reason"] = f"/slots http {s.status_code}"
                    out["kv_actual_reason"] = out["reason"]
                    return out
        except Exception as e:  # noqa: BLE001 — pure read, never raise
            out["reason"] = f"{type(e).__name__}: {e}"
            out["kv_actual_reason"] = out["reason"]
            return out

    # This runs OUTSIDE the httpx try -- the engine's n_prompt_tokens (or a
    # caller-passed override) is UNTRUSTED, and delegating the ENTIRE verdict
    # to _kv_verdict (the sole permitted producer of kv_restore_ok, see its
    # docstring for the class-level invariant) both guards the comparison
    # (never raises on a non-numeric operand) and structurally forecloses a
    # future branch here from fabricating a False from an
    # unmeasurable operand.
    actual = out["kv_actual_n_past"]
    out["kv_restore_ok"] = _kv_verdict(actual, expected_tokens, threshold)
    if out["kv_restore_ok"] is None and out["reason"] is None:
        if not out["restore_attempted"]:
            out["reason"] = "no usable expected_tokens -- nothing to verify against"
        else:
            # A real expectation existed but actual could not be measured
            # (non-numeric / absent) -- "could not measure" is NOT "measured
            # and it failed"; naming it explicitly keeps that distinction
            # visible in the record, not just in the None/False split.
            out["reason"] = f"attempted but unmeasurable (n_past={actual!r})"
            # Covers the residual path where kv_actual_n_past
            # is a caller-supplied, non-numeric override -- the async /slots
            # branch above never ran, so kv_actual_reason wasn't set there.
            # Backfilled here rather than left silently unexplained.
            if out["kv_actual_reason"] is None:
                out["kv_actual_reason"] = out["reason"]
    return out


def _find_logged_model_path(engine_log_path: str) -> tuple[str | None, str | None]:
    """Best-effort scan of the engine's own log (the structured ``.log``
    target, then its ``.stdio`` sibling) for its ``loading model '<path>'``
    identity line. Returns ``(path, None)`` on a match, or ``(None, reason)``
    when neither file yields one.

    Matches on the MESSAGE body only (``loading model '...'``), not the level
    prefix or the component-name padding in front of it -- those are
    log-formatting details this repo does not control the exact byte-shape
    of (confirmed by reading the vendored engine source this build compiles
    from: the identity line is emitted via a level+component-name prefix
    macro whose padding is a formatting convenience, not a stable contract),
    while the message text itself is the load-bearing part. Never raises."""
    pattern = re.compile(r"loading model '([^']+)'")
    reasons: list[str] = []
    for candidate in (engine_log_path, engine_log_path + ".stdio"):
        try:
            with open(candidate, "r", errors="replace") as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        return m.group(1), None
            reasons.append(f"{candidate}: no identity line found")
        except FileNotFoundError:
            reasons.append(f"{candidate}: not found")
        except OSError as e:  # noqa: BLE001 — pure read, never raise
            reasons.append(f"{candidate}: {type(e).__name__}: {e}")
    return None, "; ".join(reasons)


async def verify_model_identity(
    handle: Any,
    expected_model_path: str | None,
    *,
    engine_log_path: str | None = None,
    timeout: float = 2.0,
) -> dict:
    """PURE read: does the engine's OWN reported identity match what was requested?

    This is the instrument that establishes which model is resident.
    ``verify_model_resident`` proves *A* model is up (``health_200 and
    n_ctx>0``); it can never say WHICH one -- two different models both have
    an ``n_ctx``. This function reads the engine's own answer for which
    model it is actually serving and compares it to ``expected_model_path``
    (the path the manager passed via ``-m`` at spawn time).

    Primary source: ``GET /props``. Confirmed (by reading the vendored
    engine source this build compiles from -- ``tools/server/server-context.cpp``,
    cross-checked against ``Dockerfile.engine-src``) to be registered
    UNCONDITIONALLY for GET in the single-model (non-router) mode this
    manager always spawns in -- only POST /props (runtime mutation) is
    gated behind the ``--props`` flag. The handler always returns
    ``model_path``, the exact on-disk path the engine loaded, regardless of
    that flag. This follows from the engine source itself (the
    vendored copy this build compiles from), not
    taken from upstream llama.cpp docs.

    Fallback source: the engine's OWN log file (the manager already
    constructs its path at spawn time; passed in here as ``engine_log_path``,
    covering both the structured ``.log`` target and its ``.stdio`` sibling
    via ``_find_logged_model_path``), used only when /props is unreachable,
    refused, or its response carries no usable ``model_path``.

    Returns ``{identity_verified, identity_source, identity_reason,
    served_model_path, expected_model_path}``. ``identity_verified`` is
    ``True``/``False`` ONLY when a real served path was actually obtained
    and compared against a real expected path -- ``None`` (never a guessed
    True) when neither source could be read, or no expected path was
    supplied to compare against. Degrading honestly on an absent/unusable
    identity source, rather than assuming one, is the purpose of
    this function. Never raises."""
    expected = str(expected_model_path) if expected_model_path is not None else None
    out: dict[str, Any] = {
        "identity_verified": None,
        "identity_source": None,
        "identity_reason": None,
        "served_model_path": None,
        "expected_model_path": expected,
    }
    port = _handle_port(handle)
    props_reason: str
    if port:
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{base}/props")
            if r.status_code == 200:
                body = r.json()
                served = body.get("model_path") if isinstance(body, dict) else None
                if served:
                    out["served_model_path"] = served
                    out["identity_source"] = "props"
                    out["identity_verified"] = expected is not None and served == expected
                    if not out["identity_verified"]:
                        out["identity_reason"] = (
                            f"engine /props reports model_path={served!r}, expected {expected!r}"
                            if expected is not None else
                            "engine /props reported a model_path but no expected_model_path "
                            "was supplied to compare against"
                        )
                    return out
                props_reason = "/props returned 200 but no usable model_path field"
            else:
                props_reason = f"/props http {r.status_code}"
        except Exception as e:  # noqa: BLE001 — pure read, never raise into caller
            props_reason = f"/props {type(e).__name__}: {e}"
    else:
        props_reason = "no port on handle"

    if engine_log_path:
        served, log_reason = _find_logged_model_path(engine_log_path)
        if served:
            out["served_model_path"] = served
            out["identity_source"] = "log"
            out["identity_verified"] = expected is not None and served == expected
            if not out["identity_verified"]:
                out["identity_reason"] = (
                    f"engine log reports loading model {served!r}, expected {expected!r}"
                    if expected is not None else
                    "engine log reported a loaded model path but no expected_model_path "
                    "was supplied to compare against"
                )
            return out
        out["identity_reason"] = f"{props_reason}; log fallback: {log_reason}"
        return out

    out["identity_reason"] = f"{props_reason}; no engine_log_path supplied for fallback"
    return out


def scan_engine_log_for_errors(
    engine_log_path: str | None,
    *,
    max_lines: int = 5,
) -> dict:
    """Best-effort scan of the engine's own log for any
    E-level line logged during this spawn, so a load that degraded (e.g. a
    CUDA OOM forcing a fallback, the failure shape this scan targets) cannot
    report a clean status with literally no trace of it anywhere on the
    record. Checks BOTH the structured ``.log`` target and its ``.stdio``
    sibling -- the manager constructs the log path itself (see
    subprocess_mgr.py, plus .stdio) and it is confirmed from the vendored
    engine's own source that GGML-level errors do not all reliably reach the
    same one of the two sinks.

    The llama.cpp convention (confirmed from the vendored engine source,
    ``common/log.cpp``, and cross-checked against REAL captured engine logs
    -- a real line is ``H.MM.SSS.uuu E <component>: ...``, so the level letter
    is not the first character of the line; it follows a timestamp
    field, and a LINE PREFIX match on it would silently read every real
    error line as clean) is a single capital level letter --
    ``E``/``W``/``I``/``D`` --
    as its own whitespace-delimited FIELD, either at the very start of the
    line or immediately after a leading timestamp field, with ANSI color
    codes (disabled by default) stripped defensively first. Matching the
    letter as a field rather than anchoring to any specific timestamp shape
    means this does not hardcode the exact ``H.MM.SSS.uuu`` format and still
    recognizes a bare ``E <message>`` line with no timestamp at all.

    Engine-side note: ``~ggml_backend_cuda_buffer_context``'s own cudaFree failure
    does not abort the process -- it logs at ``W``, deliberately, so a
    benign buffer-release-at-shutdown does not read as an E-level fault
    (this file's E-only convention above is otherwise unrelated to that
    behaviour). That correctly stops THIS function from over-counting the
    benign-dominated shape, but on its own it would also
    let the SAME line fall through this function's net entirely for the
    cases where it genuinely does indicate trouble (this destructor also
    runs mid-serving, not only at shutdown). Rather than broadening to
    catch every ``W``
    line -- which would flood this function with unrelated warnings and
    reintroduce noise, the exact mistake this function exists to avoid, just
    on the other side -- this additionally matches ONLY this one
    destructor's own distinctive symbol text. Narrow by design, not a
    general severity change. This does NOT distinguish benign from fatal
    for that line -- that stays ``classify_idle_dead_holder``'s job
    (a separate check), deliberately untouched here -- it only keeps the line
    counted as "an engine error occurred" one layer up, the same as an
    E-level line, instead of silently vanishing from this
    function's view.

    Returns ``{engine_errors_detected, engine_error_lines, reason}``.
    ``engine_errors_detected`` is ``True``/``False`` only when at least one
    of the two files was actually readable AND at least one line was
    actually read from it; ``None`` when NEITHER file could be read at all,
    OR both were readable but contained literally zero lines (indistinguishable
    from a log that was never actually written to -- e.g. a crash before the
    first flush -- from "genuinely clean"; this shape occurs in practice,
    so it is not a
    hypothetical). "Could not check" must never collapse into "checked and
    clean", the same never-fabricate discipline the rest of this module
    already holds ``kv_restore_ok`` and ``model_resident`` to. A further,
    finer distinction -- "readable and non-empty but the E-line pattern
    matched nothing" vs "genuinely clean" -- is NOT attempted: once at least
    one real line has been read, there is no cheap, honest signal left to
    tell those two apart, so this function does not invent one. Line count
    is capped at ``max_lines`` for evidence display only --
    ``engine_errors_detected`` itself reflects whether ANY match existed,
    uncapped. Never raises."""
    # A leading timestamp field, if present, then the level letter as its
    # OWN token (bounded by start-of-line-or-whitespace before, whitespace
    # after) -- never anchored to column 0, and never keyed to the specific
    # timestamp shape.
    level_e = re.compile(r"(?:^|\s)E(?=\s)")
    ansi_strip = re.compile(r"\x1b\[[0-9;]*m")
    # This destructor's own symbol name is distinctive enough on
    # its own -- no other engine log line plausibly contains it -- so this is
    # a narrow, TEXT-based match, not a level-based one, kept local to this
    # function (not a module constant): classify_idle_dead_holder does not
    # need it, since its own _FATAL_INDICATORS already matches "CUDA error"/
    # "illegal memory access" regardless of which function emitted them.
    destructor_warn_mark = "~ggml_backend_cuda_buffer_context"
    matched: list[str] = []
    seen: set[str] = set()  # the .log and .stdio sinks can carry identical content
    readable = False
    lines_read = 0
    reasons: list[str] = []
    if not engine_log_path:
        return {
            "engine_errors_detected": None,
            "engine_error_lines": [],
            "reason": "no engine_log_path supplied",
        }
    for candidate in (engine_log_path, engine_log_path + ".stdio"):
        try:
            with open(candidate, "r", errors="replace") as f:
                readable = True
                for line in f:
                    lines_read += 1
                    stripped = ansi_strip.sub("", line)
                    if level_e.search(stripped) or destructor_warn_mark in stripped:
                        clean = line.rstrip("\n")
                        if clean not in seen:
                            seen.add(clean)
                            if len(matched) < max_lines:
                                matched.append(clean)
        except FileNotFoundError:
            reasons.append(f"{candidate}: not found")
        except OSError as e:  # noqa: BLE001 — pure read, never raise
            reasons.append(f"{candidate}: {type(e).__name__}: {e}")
    if not readable:
        return {
            "engine_errors_detected": None,
            "engine_error_lines": [],
            "reason": "; ".join(reasons) if reasons else "neither log file was readable",
        }
    if lines_read == 0:
        return {
            "engine_errors_detected": None,
            "engine_error_lines": [],
            "reason": (
                "log file(s) present but empty (0 lines read) -- "
                "indistinguishable from a log never actually written to"
            ),
        }
    return {
        "engine_errors_detected": bool(matched),
        "engine_error_lines": matched,
        "reason": None,
    }


# An out-of-memory load failure is requeued: these are the OOM text
# signatures that a scan_engine_log_for_errors() result is classified
# against. They cover the text the engine logs when a CUDA allocation
# fails during model load (the allocation size varies), for
# example:
#   "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating <N> MiB on
#    device 0: cudaMalloc failed: out of memory"
# "out of memory" matches the cudaMalloc wording above and the engine's own
# CUDA_ERROR_OUT_OF_MEMORY text; "failed to allocate" is the ggml
# alloc_tensor_range wording for the same underlying condition on a
# different code path. Matched case-insensitively.
_OOM_INDICATORS = ("out of memory", "failed to allocate")


def classify_load_failure(scan: dict) -> str:
    """Classify a load failure using ``scan_engine_log_for_errors()``'s own
    result -- never re-reads the log itself.

    Returns one of:
      * ``"oom"`` -- at least one line in ``scan["engine_error_lines"]``
        contains an OOM signature (see ``_OOM_INDICATORS``), case-insensitive.
        This is the ONLY verdict a caller may requeue on.
      * ``"unknown"`` -- ``scan["engine_errors_detected"]`` is ``None``: the
        engine log could not be read at all, or was readable but literally
        empty (``scan_engine_log_for_errors``'s own "could not check must
        never collapse into checked and clean" rule). An unprovable cause
        must never be guessed as OOM.
      * ``"other"`` -- the log WAS read (``engine_errors_detected`` is
        ``True`` or ``False``) but no line matched an OOM signature. Covers
        both "errors were found but none were OOM-shaped" and "the log was
        clean but the load still failed for some other reason the log does
        not capture" -- neither is provably OOM, so neither requeues.

    ``"other"`` and ``"unknown"`` are deliberately NOT merged: a caller that
    wants to log or test them separately can, but both are non-OOM for
    admission purposes and both fail once by design. Never raises --
    a malformed ``scan`` (missing keys, wrong types) degrades to
    ``"unknown"``, the same fail-closed-to-fail-once direction as an
    unreadable log, never to ``"oom"``.
    """
    detected = scan.get("engine_errors_detected") if isinstance(scan, dict) else None
    if detected is None:
        return "unknown"
    lines = scan.get("engine_error_lines") or []
    if not isinstance(lines, list):
        return "unknown"
    for line in lines:
        if not isinstance(line, str):
            continue
        lowered = line.lower()
        if any(indicator in lowered for indicator in _OOM_INDICATORS):
            return "oom"
    return "other"


# The engine's own orderly-exit announcement. In engine logs it is the
# discriminator between a recoverable/expected exit and the unannounced
# mid-serving fault: when this line is present
# the process was already on its way out by its own (or the
# manager's) decision; when it is absent the file simply
# stops mid-task.
_CLEAN_EXIT_MARK = "cleaning up before exit"

# The fatal signatures known to occur (CUDA error /
# illegal memory access, firing inside an active tensor op with zero further
# log lines afterward), plus the standard llama.cpp/ggml abort vocabulary as
# future-proofing, so that
# this classifier does not
# need to be revisited if a different fatal shape shows up later.
_FATAL_INDICATORS = (
    "CUDA error",
    "illegal memory access",
    "unspecified launch failure",
    "Segmentation fault",
    "SIGSEGV",
    "SIGABRT",
    "Aborted",
    "core dumped",
    "GGML_ASSERT",
    "GGML_ABORT",
    "terminate called",
    "double free",
)


def classify_idle_dead_holder(engine_log_path: str | None) -> dict:
    """Name what the stuck-handle dead-idle sweep caught.

    The sweep (manager.py worker_loop, the stuck-handle step) already
    recovers correctly from an idle-hot holder found dead between requests --
    this function does not touch that recovery. It answers a narrower
    question the sweep never asked: is this the engine announcing its own
    exit (``_CLEAN_EXIT_MARK`` present), or the unannounced fatal shape
    seen in real engine logs (a ``_FATAL_INDICATORS`` line with no exit
    announcement after it)?

    Reads the SAME ``engine_log_path`` (plus its ``.stdio`` sibling) that
    ``scan_engine_log_for_errors`` already reads for this exact handle, for
    the same reason: some fatal output only reaches one of the two sinks.
    Per file, the discriminator is PRESENCE of the clean-exit mark, not the
    relative order of the two patterns: a real fault kills
    the process outright, with zero further log
    lines of any kind (no completion, no shutdown message, nothing) -- so a
    genuinely unannounced death can never be followed by the exit mark in
    the same file. A fatal indicator that DOES appear in a file that also
    contains the exit mark (the ``~ggml_backend_cuda_buffer_context``
    destructor case: already announced, THEN a
    teardown-side-effect error) is therefore always a symptom of an exit
    already underway, not the cause -- ``"clean_shutdown"``, not
    ``"fault_signature"``, regardless of which line is physically last.
    Across the two files, ANY file reporting a fault with no exit mark of
    its own wins the overall classification -- this errs toward NOT missing
    a real fault over erring toward a quiet clean-shutdown verdict.

    Returns ``{death_class, reason, matched_line}``. ``death_class`` is one
    of four values, deliberately NOT collapsed into three -- doing so would
    silently reintroduce the exact trap this classifier exists to close, one level
    down:
      * ``"clean_shutdown"`` -- the exit mark is the terminal signal.
      * ``"fault_signature"`` -- a fatal indicator is the terminal signal.
      * ``"no_signature_found"`` -- both files were readable and non-empty,
        but neither pattern appears anywhere. This is itself informative: the
        holder died in a shape this classifier does not yet recognize.
      * ``None`` (reason explains why) -- neither file could be read, or
        both were readable but carried zero lines -- indistinguishable from
        "never actually written to" (the same one-line-read edge case
        ``scan_engine_log_for_errors`` already documents). "Could not check"
        must never collapse into any of the three definite verdicts above.

    Never raises. Read-only. No side effects."""
    if not engine_log_path:
        return {
            "death_class": None,
            "reason": "no engine_log_path supplied",
            "matched_line": None,
        }
    per_file_verdicts: list[tuple[str, str | None]] = []  # (verdict, matched_line)
    any_readable = False
    any_lines = False
    read_reasons: list[str] = []
    for candidate in (engine_log_path, engine_log_path + ".stdio"):
        try:
            with open(candidate, "r", errors="replace") as f:
                has_clean = False
                has_fault = False
                fault_line: str | None = None
                for line in f:
                    any_readable = True
                    any_lines = True
                    if _CLEAN_EXIT_MARK in line:
                        has_clean = True
                    if not has_fault:
                        for indicator in _FATAL_INDICATORS:
                            if indicator in line:
                                has_fault = True
                                fault_line = line.rstrip("\n")
                                break
                if not has_clean and not has_fault:
                    continue  # this file had no signal either way
                if has_clean:
                    # present anywhere -- see docstring for why order isn't
                    # the discriminator; an announced exit makes any fatal
                    # indicator in the SAME file a teardown side effect.
                    per_file_verdicts.append(("clean_shutdown", None))
                else:
                    per_file_verdicts.append(("fault_signature", fault_line))
        except FileNotFoundError:
            read_reasons.append(f"{candidate}: not found")
        except OSError as e:  # noqa: BLE001 — pure read, never raise
            read_reasons.append(f"{candidate}: {type(e).__name__}: {e}")
    if not any_readable:
        return {
            "death_class": None,
            "reason": "; ".join(read_reasons) if read_reasons else
                      "neither log file was readable",
            "matched_line": None,
        }
    if not any_lines:
        return {
            "death_class": None,
            "reason": (
                "log file(s) present but empty (0 lines read) -- "
                "indistinguishable from a log never actually written to"
            ),
            "matched_line": None,
        }
    for verdict, matched_line in per_file_verdicts:
        if verdict == "fault_signature":
            return {
                "death_class": "fault_signature",
                "reason": None,
                "matched_line": matched_line,
            }
    if any(v == "clean_shutdown" for v, _ in per_file_verdicts):
        return {"death_class": "clean_shutdown", "reason": None, "matched_line": None}
    return {
        "death_class": "no_signature_found",
        "reason": (
            "log file(s) readable and non-empty but neither the clean-exit "
            "mark nor a known fatal indicator appears anywhere"
        ),
        "matched_line": None,
    }
