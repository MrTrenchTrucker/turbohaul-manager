"""GET /v1/logging — paginated audit_events stream (audit-event stream).

Paginated audit-event endpoint.

Auth model: NO app-layer auth — matches project-wide network-perimeter posture
(/api/config, /api/chat, /v1/chat/completions all unauthenticated). See
ARCHITECTURE.md §11 addendum for the documented security model.

Denylist is a tripwire; load-bearing protection is emitter discipline (no
prompts/responses/PII in audit payloads).
"""
import json
import logging as stdlogging
from typing import Any

from fastapi import APIRouter, Query, Request

from turbohaul.manager import EventBus
from turbohaul.state import state_db_session


log = stdlogging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["logging"])


# Token-budget constants (mode 2 — envelope overhead subtracted upfront)
MAX_RESPONSE_TOKENS = 20_000  # hard cap
APPROX_CHARS_PER_TOKEN = 4
MAX_RESPONSE_CHARS = MAX_RESPONSE_TOKENS * APPROX_CHARS_PER_TOKEN  # 80_000
# 512 = JSON envelope overhead allocation (derivation):
#   - {"events":[...], "next_since": N, "oversized": B} key/value framing ~80c
#   - worst-case next_since int64 string ~20c
#   - schema-evolution headroom (future field additions) ~200c
#   - 212c safety margin for delimiter/whitespace variance
ENVELOPE_OVERHEAD_CHARS = 512
EFFECTIVE_BUDGET = MAX_RESPONSE_CHARS - ENVELOPE_OVERHEAD_CHARS  # 79_488

_FILTER_LEN_CAP = 128
_LIMIT_MIN = 1
_LIMIT_MAX = 500
_LIMIT_DEFAULT = 200
_REDACT_DEPTH_CAP = 10


def _redact(value: Any, depth: int = 0) -> Any:
    """Recursively scrub REDACTED_KEYS from dicts (including nested + list items).

    Denylist is a tripwire; load-bearing protection is emitter discipline (no
    prompts/responses/PII in audit payloads).

    Depth-cap guards against pathological deep nesting and reference cycles.
    """
    if depth > _REDACT_DEPTH_CAP:
        return value
    if isinstance(value, dict):
        return {
            k: _redact(v, depth + 1)
            for k, v in value.items()
            if k not in EventBus.REDACTED_KEYS
        }
    if isinstance(value, list):
        return [_redact(item, depth + 1) for item in value]
    return value


def _decode_payload(payload_json: str | None) -> dict:
    """Best-effort JSON decode. Poison rows do not crash the request (mode 1).

    On decode failure or non-dict shape, substitute a sentinel so the row is
    still surfaced (caller sees WHICH event_id is corrupt) without 500'ing the
    whole pagination call.
    """
    if not payload_json:
        return {}
    try:
        loaded = json.loads(payload_json)
    except json.JSONDecodeError:
        return {"_decode_error": True, "_raw_len": len(payload_json)}
    if not isinstance(loaded, dict):
        return {"_decode_error": True, "_raw_len": len(payload_json)}
    return loaded


def _row_to_event(row: Any) -> dict:
    return {
        "event_id": row["event_id"],
        "slot_id": row["slot_id"],
        "event_type": row["event_type"],
        "payload": _redact(_decode_payload(row["payload_json"])),
        "occurred_at": row["occurred_at"],
    }


@router.get("/logging")
async def get_logging(
    request: Request,
    since: int = Query(0, ge=0),
    slot_id: str | None = Query(None, max_length=_FILTER_LEN_CAP),
    event_type: str | None = Query(None, max_length=_FILTER_LEN_CAP),
    limit: int = Query(_LIMIT_DEFAULT),
) -> dict:
    """GET /v1/logging — paginated audit_events stream.

    Network-perimeter auth model: NO app-layer auth — matches project posture.
    See ARCHITECTURE.md §11 addendum for the documented security model.

    Pagination: caller loops `?since=next_since` until `next_since` is null.
    Use `>=` semantics; `next_since = last_event_id + 1` when more remain.
    """
    if limit < _LIMIT_MIN:
        limit = _LIMIT_MIN
    elif limit > _LIMIT_MAX:
        limit = _LIMIT_MAX

    mgr = request.app.state.manager
    state_db_path = mgr.boot.storage.state_db_path

    with state_db_session(state_db_path) as conn:
        cur = conn.execute(
            """SELECT event_id, slot_id, event_type, payload_json, occurred_at
               FROM audit_events
               WHERE event_id >= ?
                 AND (? IS NULL OR slot_id = ?)
                 AND (? IS NULL OR event_type = ?)
               ORDER BY event_id ASC
               LIMIT ?""",
            (since, slot_id, slot_id, event_type, event_type, limit),
        )
        rows = list(cur.fetchall())

    events: list[dict] = []
    used = 0
    next_since: int | None = None
    oversized = False

    for row in rows:
        ev = _row_to_event(row)
        encoded_len = len(json.dumps(ev))
        if used + encoded_len > EFFECTIVE_BUDGET:
            next_since = ev["event_id"]
            break
        events.append(ev)
        used += encoded_len
    else:
        if len(rows) == limit and events:
            next_since = events[-1]["event_id"] + 1

    if not events and rows:
        ev = _row_to_event(rows[0])
        events.append(ev)
        oversized = True
        next_since = rows[0]["event_id"] + 1

    return {
        "events": events,
        "next_since": next_since,
        "oversized": oversized,
    }
