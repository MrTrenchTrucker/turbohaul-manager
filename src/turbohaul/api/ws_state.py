"""WebSocket /ws/state - redacted event broadcaster per v0.2 §11.1.

Subscribers receive STATE-level events only:
  - connected (one-time, on accept) with initial status snapshot
  - submit / stage_to_loading / active / grace_enter / teardown / idle_hot_enter
  - queue_change (depth changes only)

NEVER broadcasts: prompt text, response text, stderr lines, full thread_ids, IPs.
"""
import asyncio
import logging

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect


log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/state")
async def ws_state(websocket: WebSocket) -> None:
    """Subscribe to redacted state events.

    On accept, sends one `connected` event with the current status snapshot, then
    streams events as the worker_loop publishes them.
    """
    await websocket.accept()
    mgr = websocket.app.state.manager

    subscriber_q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    mgr.event_bus.subscribe(subscriber_q)

    try:
        # Initial snapshot at connect time
        await websocket.send_json(
            {
                "event": "connected",
                "snapshot": mgr.status_snapshot(),
            }
        )
        # Stream events
        while True:
            event = await subscriber_q.get()
            try:
                await websocket.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws_state handler crashed")
    finally:
        mgr.event_bus.unsubscribe(subscriber_q)
