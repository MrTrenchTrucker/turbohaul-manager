"""Turbohaul SPEC_DOWNGRADE observability (DISPLAY / OBSERVABILITY ONLY).

WHY THIS MODULE EXISTS
======================
D-Spark/D-Flash need the TARGET model's per-layer hidden state, captured by
per-architecture engine code that exists for only 6 of 135 architectures
(an engine-side limitation). Common architectures such as qwen35,
qwen35moe and qwen2 are not among the six, so a speculative-decode request against
them would otherwise abort the whole engine on the first token. The engine
turns that into a clean DOWNGRADE: the target model still loads and serves
normally, just without acceleration -- and emits one engine-log line saying
so, from ``server_context::load_model()``, after the target has already
loaded successfully and before the real draft context would be constructed.

Requirement: a silent downgrade is nearly as bad as the crash it
replaces, because the user believes they have acceleration they do not have.
This module is the manager-side half of making it visible: it turns the
engine's log line into structured state a ``/status`` consumer (the FE) can
render, mirroring ``load_verify_log.py``'s established shape exactly (same
module-scope ring, same read-only/never-raise discipline) rather than
inventing a second observability mechanism.

  * ``scan_engine_log_for_spec_downgrade(engine_log_path)`` -- PURE sync read
    helper. Reads the SAME two files (``.log`` + ``.stdio``)
    ``scan_engine_log_for_errors`` already reads, for the same reason (GGML/
    server-level log lines do not all reliably reach the same one of the two
    sinks). Matches on the LITERAL TAG ``SPEC_DOWNGRADED`` (the engine's log format fixes
    this token and never changes its wording -- grep on this exact
    token), not on any status/field value -- so a future, separate
    ``SPEC_FAILED`` tag (expected to use an identical key layout, not a status= value
    on this one) slots in later as its own scan without touching this
    parser's logic, the same way this function's own sibling
    ``classify_idle_dead_holder`` reads a second, independent signature out
    of the exact same files ``scan_engine_log_for_errors`` reads.
  * ``log_spec_downgrade(**fields)`` -- emits ONE greppable
    ``SPEC_DOWNGRADE_RECORD {json}`` log line and stashes the last N records
    in a module-level ring for the ``/status`` read surface. Two provenances
    write into the SAME channel: the engine's log line (``source=
    "engine_log"``, via the scan above) and the manager's OWN
    silent downgrade (``source="manager_config"``: a manifest
    sets ``spec_draft_gguf_blob_sha256`` but ``spec_type`` does not qualify
    for a standalone drafter -- the injection is correctly skipped,
    but without this module it would never reach ``/status`` or the user).

HARD BOUNDARY, same as ``load_verify_log.py``: this module NEVER changes
spawn/decode behavior. It reads and reports. The ring lives at MODULE scope
so it adds no new manager instance state and keeps
``manager.py`` free of it.

DOWNGRADED IS NOT FAILED, STRUCTURALLY, NOT JUST BY CONVENTION: this module
has no failure/error status value at all -- it has exactly one shape, an
informational downgrade record. A real spawn/load FAILURE continues to be
represented exclusively by the existing ``ResidentState.LOADING_FAIL``
lifecycle state and ``scan_engine_log_for_errors``'s E-level detection; this
module's records and those are never merged into one field or one tone, by
construction, not by a rule someone has to remember to keep following. Per
the engine's behavior, this line only ever fires in a state where the TARGET model has
ALREADY loaded successfully and WILL serve normally.

KNOWN GAP:
the engine's downgrade interception only runs when
``fit_params`` is ON (the manifest's ``fit`` flag defaults to unset, which
the engine treats as on). Under an explicit ``fit: off`` manifest, no
SPEC_DOWNGRADED line is ever emitted and the original hard abort still
happens on an incompatible architecture. This module cannot distinguish
"aborted because of the fit-off/spec-incompatibility case" from any other
engine abort -- there is no engine-side signal for that specific failure
shape today (no SPEC_FAILED tag exists yet in the engine's log format), and inventing a
heuristic for it here would be exactly the "guess the engine's emitted
contract" this module must not do. A ``fit: off`` deployment
can therefore have a perfectly correct UI that shows nothing, on a model
that actually crashed for this reason. This is a known,
open gap in this module.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level ring buffer (last N SPEC_DOWNGRADE records) for the /status
# surface. Lives here -- NOT on the manager instance -- for the identical
# reason load_verify_log.py's ring does: adds zero new manager state and keeps
# manager.py free of it.
# ---------------------------------------------------------------------------
_RING_MAX = 64
_ring: "deque[dict]" = deque(maxlen=_RING_MAX)
_ring_lock = threading.Lock()

# The engine's fixed log tag -- the ONE thing that must never change
# wording, by design. Matched as a token, not a full-line anchor, so
# this does not assume anything about what precedes it in the real log line
# (timestamp shape, level letter, "[spec] " prefix) -- mirrors
# scan_engine_log_for_errors's own "never anchored to a specific timestamp
# shape" discipline for the exact same reason: the real prefix format is
# engine-owned and can change without this parser caring.
_TAG = "SPEC_DOWNGRADED"
_TAG_RE = re.compile(r"\bSPEC_DOWNGRADED\b(.*)$")
_ARCH_RE = re.compile(r"\barch=(\S+)")
_COMPONENT_RE = re.compile(r"\bcomponent=(\S+)")
_REASON_RE = re.compile(r"\breason=(\S+)")
# detail= is the LAST field on the line and is explicitly raw/unparsed free
# text (the engine's log format says not to build logic against this field's content) -- captured
# greedily to end of line, once "detail=" itself is found, rather than
# \S+-bounded like the three fields above it.
_DETAIL_RE = re.compile(r"\bdetail=(.*)$")

ansi_strip = re.compile(r"\x1b\[[0-9;]*m")


def scan_engine_log_for_spec_downgrade(engine_log_path: str | None) -> dict:
    """Best-effort scan of the engine's own log for a SPEC_DOWNGRADED line.

    Checks BOTH the structured ``.log`` target and its ``.stdio`` sibling,
    same as ``scan_engine_log_for_errors`` and for the same reason. Returns
    the FIRST match (the engine fires this line once, from
    ``load_model()``, per spawn -- there is nothing to reconcile across
    multiple matches).

    Returns ``{spec_downgraded, arch, component, reason, detail,
    matched_line, engine_log_reason}``.

    ``spec_downgraded`` is ``True`` only when the tag was found AND all three
    required fields (arch, component, reason) parsed -- a line that matches
    the tag but fails to parse (a future engine-side wording change to the
    fields, not the tag) is reported as ``None`` with a reason, never
    silently coerced to ``False`` (that would read as "no downgrade
    happened" when the truth is "could not tell"). ``False`` when at least
    one file was read successfully and the tag was genuinely absent.
    ``None`` when neither file could be read at all, or both were readable
    but carried zero lines -- indistinguishable from "never actually written
    to", same non-fabrication discipline as ``scan_engine_log_for_errors``.
    Never raises.
    """
    if not engine_log_path:
        return {
            "spec_downgraded": None, "arch": None, "component": None,
            "reason": None, "detail": None, "matched_line": None,
            "engine_log_reason": "no engine_log_path supplied",
        }
    readable = False
    lines_read = 0
    reasons: list[str] = []
    for candidate in (engine_log_path, engine_log_path + ".stdio"):
        try:
            with open(candidate, "r", errors="replace") as f:
                readable = True
                for line in f:
                    lines_read += 1
                    stripped = ansi_strip.sub("", line)
                    tag_m = _TAG_RE.search(stripped)
                    if not tag_m:
                        continue
                    tail = tag_m.group(1)
                    arch_m = _ARCH_RE.search(tail)
                    component_m = _COMPONENT_RE.search(tail)
                    reason_m = _REASON_RE.search(tail)
                    detail_m = _DETAIL_RE.search(tail)
                    if not (arch_m and component_m and reason_m):
                        return {
                            "spec_downgraded": None, "arch": None,
                            "component": None, "reason": None, "detail": None,
                            "matched_line": line.rstrip("\n"),
                            "engine_log_reason": (
                                f"{_TAG} tag found but arch/component/reason "
                                "did not all parse -- engine wording may "
                                "have changed"
                            ),
                        }
                    return {
                        "spec_downgraded": True,
                        "arch": arch_m.group(1),
                        "component": component_m.group(1),
                        "reason": reason_m.group(1),
                        "detail": detail_m.group(1).strip() if detail_m else None,
                        "matched_line": line.rstrip("\n"),
                        "engine_log_reason": None,
                    }
        except FileNotFoundError:
            reasons.append(f"{candidate}: not found")
        except OSError as e:  # noqa: BLE001 — pure read, never raise
            reasons.append(f"{candidate}: {type(e).__name__}: {e}")
    if not readable:
        return {
            "spec_downgraded": None, "arch": None, "component": None,
            "reason": None, "detail": None, "matched_line": None,
            "engine_log_reason": (
                "; ".join(reasons) if reasons else "neither log file was readable"
            ),
        }
    if lines_read == 0:
        return {
            "spec_downgraded": None, "arch": None, "component": None,
            "reason": None, "detail": None, "matched_line": None,
            "engine_log_reason": (
                "log file(s) present but empty (0 lines read) -- "
                "indistinguishable from a log never actually written to"
            ),
        }
    return {
        "spec_downgraded": False, "arch": None, "component": None,
        "reason": None, "detail": None, "matched_line": None,
        "engine_log_reason": None,
    }


def log_spec_downgrade(
    *,
    model_tag: str,
    source: str,
    arch: str | None = None,
    component: str | None = None,
    reason_code: str | None = None,
    detail: str | None = None,
) -> dict:
    """Emit ONE ``SPEC_DOWNGRADE_RECORD`` json log line + stash it in the
    module ring.

    ``source`` distinguishes provenance: ``"engine_log"`` (parsed from
    the engine's SPEC_DOWNGRADED line via the scan above) or ``"manager_config"``
    (the manager's own manifest/spec_type mismatch, detected before spawn,
    never reaches the engine at all).

    ``arch`` and ``component`` are OPTIONAL, deliberately -- the engine
    line always carries both, but the manager-config tenant fires before a
    GGUF read that would establish arch and must not fabricate one. Absent
    is represented as ``None`` here (not the engine's own "unknown" convention,
    which is ITS fallback for a failed read on ITS side, not a stand-in for
    "not applicable on the manager side" -- collapsing the two would blur a
    real distinction: "the engine tried to read arch and failed" is a
    different fact from "this record's source never has arch at all").

    ``reason_code`` is an OPEN, APPEND-ONLY vocabulary, never validated
    against a closed set here -- the engine has exactly one value today
    (``draft_context_init_failed``) and does not
    invent sub-codes it cannot distinguish; the manager-config
    tenant contributes its own value (``spec_type_mismatch``). Never-raise,
    best-effort, mirrors ``log_load_verify`` exactly.
    """
    record: dict[str, Any] = {
        "model_tag": model_tag,
        "source": source,
        "arch": arch,
        "component": component,
        "reason_code": reason_code,
        "detail": detail,
    }
    try:
        log.warning("SPEC_DOWNGRADE_RECORD %s", json.dumps(record))
    except Exception:  # noqa: BLE001 — display-only, never fail a spawn
        log.debug("SPEC_DOWNGRADE_RECORD emit failed (ignored)", exc_info=True)
    try:
        with _ring_lock:
            # Store a COPY, not the returned object -- same reason
            # as load_verify_log's ring: a live alias could be mutated
            # mid-json.dumps on the /status path.
            _ring.append(dict(record))
    except Exception:  # noqa: BLE001 — ring is observability only
        log.debug("SPEC_DOWNGRADE_RECORD ring append failed (ignored)", exc_info=True)
    return record


def get_recent(n: int | None = None) -> list[dict]:
    """Return the last ``n`` SPEC_DOWNGRADE records (newest last), for
    ``/status``. Copies the ring under the lock so the caller gets a stable
    snapshot. ``n`` defaults to the whole ring."""
    with _ring_lock:
        items = list(_ring)
    if n is not None and n >= 0:
        # n==0 must mean "none" -- items[-0:] == items[0:] == everything.
        items = items[-n:] if n else []
    return items


def clear_ring() -> None:
    """Test/ops helper — drop all buffered records. No effect on behavior."""
    with _ring_lock:
        _ring.clear()
