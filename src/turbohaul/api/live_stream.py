"""SSE endpoint for live model OUTPUT TEXT — GET /ui/live/output/stream.

ANCHOR-FOLLOW: a single connection follows whatever generation is CURRENTLY live
(mgr.live_generation.generation_id), switching server-side the instant the anchor
changes. This is the robust fix for a bursty workload where the generation_id
turns over every few seconds — the FE would otherwise lock onto a stale id and
miss most generations ("works randomly"). The client never supplies a gid.

Frame protocol (all JSON in `data:` SSE frames):
  {generation_id, text, done:false, reset:true}   -> NEW generation; FE CLEARS the
                                                      pane and shows `text` (replay tail)
  {generation_id, text, done:false}                -> incremental output delta (append)
  {generation_id, text:"", done:true}              -> current generation ENDED
  {generation_id:null, ..., reset:true, idle:true} -> went idle (no live generation)
  ": keep-alive"                                   -> heartbeat

Carries the high-volume token text OFF the EventBus/(/ws/state) so the redaction
denylist there is preserved. Streams ONLY assistant output deltas (never the
prompt / IPs / full thread-id).
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse


log = logging.getLogger(__name__)
router = APIRouter()

_POLL_S = 1.0   # max wait on the current buffer before re-checking the anchor


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


@router.get("/ui/live/output/stream", include_in_schema=False)
async def live_output_stream(request: Request) -> StreamingResponse:
    """Anchor-follow the current live generation's output text."""
    mgr = request.app.state.manager

    async def gen():
        cur: str | None = None      # the generation_id we are currently following
        q: asyncio.Queue | None = None
        try:
            while True:
                anchor = (mgr.live_generation or {}).get("generation_id")
                # --- anchor changed: switch what we follow ---
                if anchor != cur:
                    if q is not None and cur is not None:
                        mgr.live_output.unsubscribe(cur, q)
                        q = None
                    cur = anchor
                    if cur:
                        # cur IS the current anchor, so allow_create is safe (the
                        # tee may not have fed yet); a client can't inject a gid.
                        q, replay, done0 = mgr.live_output.subscribe(cur, allow_create=True)
                        yield _sse({"generation_id": cur, "text": replay, "done": False, "reset": True})
                        if done0 and q is not None:
                            mgr.live_output.unsubscribe(cur, q)
                            q = None
                    else:
                        yield _sse({"generation_id": None, "text": "", "done": False, "reset": True, "idle": True})
                # --- stream the current generation, or idle-wait + re-check ---
                if q is not None:
                    try:
                        piece = await asyncio.wait_for(q.get(), timeout=_POLL_S)
                    except asyncio.TimeoutError:
                        yield b": keep-alive\n\n"
                        continue
                    if piece is None:  # this generation finished
                        yield _sse({"generation_id": cur, "text": "", "done": True})
                        mgr.live_output.unsubscribe(cur, q)
                        q = None
                        await asyncio.sleep(0.25)  # wait for the anchor to advance
                        continue
                    yield _sse({"generation_id": cur, "text": piece, "done": False})
                else:
                    await asyncio.sleep(0.5)
                    yield b": keep-alive\n\n"
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("live_output_stream handler crashed")
        finally:
            if q is not None and cur is not None:
                mgr.live_output.unsubscribe(cur, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
