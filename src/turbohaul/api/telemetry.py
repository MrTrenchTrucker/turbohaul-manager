"""GET /v1/telemetry/events — query the flap/degradation JSONL log.

Read endpoint for the telemetry subsystem. Serves both the
in-memory ring buffer (hot, fast) and the persistent JSONL files (full
history).

Auth model: NO app-layer auth — matches project-wide network-perimeter
posture. See ARCHITECTURE.md §8.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/telemetry", tags=["telemetry"])

_LIMIT_MIN = 1
_LIMIT_MAX = 1000
_LIMIT_DEFAULT = 200


@router.get("/events")
async def get_telemetry_events(
    request: Request,
    since: int = Query(0, ge=0, description="Ring-buffer sequence cursor (skip events <= this)"),
    event_type: str | None = Query(None, max_length=64, description="Filter by event_type"),
    limit: int = Query(_LIMIT_DEFAULT, ge=_LIMIT_MIN, le=_LIMIT_MAX),
    source: str = Query("ring", pattern="^(ring|file)$"),
) -> dict:
    """GET /v1/telemetry/events - paginated telemetry stream.

    source=ring (default): reads the in-memory ring buffer (fast, last
    ~10k events). source=file: reads the persistent JSONL files (full
    history, slower scan).

    Pagination: since is a monotonic _seq number. Pass next_since
    from a previous response to get the next page.
    """
    mgr = request.app.state.manager
    telemetry = mgr._telemetry  # type: ignore[attr-defined]

    if telemetry is None:
        return {"events": [], "next_since": None, "error": "telemetry not initialized"}

    if source == "file":
        return telemetry.get_events_from_file(
            limit=limit, event_type=event_type, since_id=since,
        )
    return telemetry.get_events(
        limit=limit, event_type=event_type, since_id=since,
    )


@router.get("/status")
async def get_telemetry_status(request: Request) -> dict:
    """GET /v1/telemetry/status — telemetry subsystem health."""
    mgr = request.app.state.manager
    telemetry = mgr._telemetry  # type: ignore[attr-defined]

    if telemetry is None:
        return {"enabled": False, "error": "not initialized"}
    return telemetry.get_status()
