"""Live inference monitor — ground-truth tok/s + progress + live output text.

Gives turbohaul its OWN view of what the active sidecar is decoding, independent
of any client CLI. Two cleanly separated data planes:

  1. ROBUST CORE (stream-mode independent): ``LiveSlotsPoller`` polls the active
     llama-server's ``GET /slots`` ~1Hz, derives tok/s + progress from
     ``next_token[0].n_decoded`` deltas, and writes a redacted ``generation``
     block into ``mgr.live_generation`` that ``status_snapshot()`` exposes. It is
     a PURE OBSERVER: it copies (pid, port, spawn_seq) scalars off the manager in
     one synchronous read, RE-VALIDATES handle identity after the httpx await
     (fixed-port 11500 reuse race), never holds a Slot, never blocks/locks the
     worker_loop, never mutates FSM state, never iterates ``_inflight``.

  2. THIN LIVE-TEXT TEE: ``LiveOutputBuffer`` holds per-generation bounded ring
     buffers fed by the streaming tee in ``api/chat_completion.py`` and read ONLY
     by the dedicated SSE endpoint in ``api/live_stream.py`` — token TEXT never
     rides the EventBus, so the /ws/state redaction denylist stays intact.

Both planes key on ONE unified ``generation_id`` = blake2b(pid:spawn_seq:thread).
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from collections import OrderedDict

import httpx

from turbohaul.state import utcnow_iso


log = logging.getLogger(__name__)

# --- tuning constants (module-level; only enabled/poll_interval_s are config) ---
SLOTS_TIMEOUT_S = 1.5          # /slots GET timeout — caps a hung sidecar
STALL_AFTER_S = 2.0           # frozen n_decoded this long while processing => stalled/finishing
STALE_GAP_S = 5.0             # poll gap beyond this => rate-only reset (keep generation)
EWMA_ALPHA = 0.4             # tok/s smoothing
TEXT_TAIL_BYTES = 16384       # live output is a TAIL (last ~few hundred tokens), not a transcript
MAX_LIVE_KEYS = 8            # LRU cap on concurrent per-generation text buffers
SSE_HEARTBEAT_S = 15.0        # ': keep-alive' cadence on the output SSE
MAX_SUBS_PER_GEN = 32        # cap concurrent SSE subscribers per generation (DoS guard)
CARRY_MAX_BYTES = 65536       # drop carry-over if a sidecar never frames (defensive)


def compute_generation_id(pid, spawn_seq, thread_or_slot) -> str:
    """Stable, non-reversible 8-hex id for a single generation.

    The SAME tuple is hashed by the poller (metrics plane) and the tee (text
    plane) so the FE never shows generation A's text under generation B's tok/s.
    ``spawn_seq`` makes a fixed-port (11500) sidecar reuse non-colliding.
    """
    raw = f"{pid}:{spawn_seq}:{thread_or_slot}".encode("utf-8", "replace")
    return hashlib.blake2b(raw, digest_size=4).hexdigest()  # 8 hex chars


def _base_generation(state: str, generation_id: str | None = None) -> dict:
    """Canonical generation-block shape. Single source for idle/transition/loading."""
    return {
        "state": state,
        "tok_s": 0.0 if state in ("idle", "stalled", "finishing") else None,
        "tok_s_instant": 0.0,
        "n_decoded": 0,
        "max_tokens": None,
        "n_remain": None,
        "n_prompt_tokens": 0,
        "n_ctx": None,
        "prompt_progress": None,
        "pct": None,
        "eta_s": None,
        "stalled": False,
        "streaming": False,
        "generation_id": generation_id,
        "riders": 0,
        "measured_at_iso": utcnow_iso(),
    }


def idle_generation() -> dict:
    """The /status fallback when the poller has not written anything yet."""
    return _base_generation("idle")


# ============================================================================
# Live output text — per-generation bounded ring buffers (text plane)
# ============================================================================
class _GenBuf:
    __slots__ = ("tail", "carry", "done", "subs")

    def __init__(self) -> None:
        self.tail: str = ""        # bounded replay tail
        self.carry: bytes = b""    # incomplete-SSE-frame carry-over
        self.done: bool = False
        self.subs: set[asyncio.Queue] = set()


class LiveOutputBuffer:
    """Dict of per-generation_id bounded text buffers.

    feed() is called by the streaming tee (cooperative, same event loop); it
    re-frames arbitrary ``aiter_bytes`` TCP boundaries on the SSE ``\\n\\n``
    delimiter, extracts ONLY assistant ``delta.content`` / ``reasoning_content``
    (output text, never the prompt), appends to a bounded tail, and fans the new
    piece out to each subscriber's queue (drop-on-full, never blocks the stream).
    """

    def __init__(self, max_keys: int = MAX_LIVE_KEYS, tail_bytes: int = TEXT_TAIL_BYTES) -> None:
        self._max_keys = max_keys
        self._tail_bytes = tail_bytes
        self._buffers: "OrderedDict[str, _GenBuf]" = OrderedDict()

    def _get(self, generation_id: str, *, create: bool) -> _GenBuf | None:
        buf = self._buffers.get(generation_id)
        if buf is None and create:
            buf = _GenBuf()
            self._buffers[generation_id] = buf
            self._evict()
        if buf is not None:
            self._buffers.move_to_end(generation_id)
        return buf

    def _evict(self) -> None:
        # LRU-evict oldest buffers with NO active subscribers, keeping <= max_keys.
        while len(self._buffers) > self._max_keys:
            for gid, buf in list(self._buffers.items()):
                if not buf.subs:
                    del self._buffers[gid]
                    break
            else:
                break  # all remaining have subscribers; do not orphan them

    def feed(self, generation_id: str, sse_bytes: bytes) -> None:
        buf = self._get(generation_id, create=True)
        if buf is None:
            return
        buf.carry += sse_bytes
        while b"\n\n" in buf.carry:
            frame, buf.carry = buf.carry.split(b"\n\n", 1)
            self._consume_frame(buf, frame)
        # defensive: a sidecar that never emits a frame delimiter must not grow
        # carry without bound (the tail is capped, carry was not).
        if len(buf.carry) > CARRY_MAX_BYTES:
            buf.carry = b""

    def _consume_frame(self, buf: _GenBuf, frame: bytes) -> None:
        try:
            pieces: list[str] = []
            for raw_line in frame.split(b"\n"):
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    c = delta.get("content")
                    if c:
                        pieces.append(c)
                    rc = delta.get("reasoning_content")
                    if rc:
                        pieces.append(rc)
                    # Tool calls stream incrementally in delta.tool_calls[*].function
                    # (name arrives once, then arguments in fragments). Surface them
                    # so the live pane shows the model's ACTIONS, not just its thoughts.
                    for tc in delta.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        name = fn.get("name")
                        if name:
                            pieces.append("\n→ tool_call: %s " % name)
                        arg = fn.get("arguments")
                        # llama.cpp may emit `arguments` as an already-parsed JSON
                        # object (dict) rather than a JSON string (issue #20198 /
                        # PR #18675's common/chat.cpp json::parse refactor). A dict
                        # in `pieces` would make the str.join below raise TypeError,
                        # and the fail-open `except` would then SILENTLY swallow the
                        # whole frame — the exact live symptom "tool calls show
                        # nothing". Coerce non-str arguments to JSON text.
                        if arg is not None:
                            pieces.append(arg if isinstance(arg, str) else json.dumps(arg))
            if pieces:
                # Coerce every piece: one non-str field (upstream type drift in
                # content/reasoning_content/name/arguments) must never nuke an
                # entire frame's output via a join TypeError.
                self._append(buf, "".join(str(p) for p in pieces))
        except Exception:
            pass  # the tee is fail-open; never disturb the client stream

    def _append(self, buf: _GenBuf, piece: str) -> None:
        buf.tail = (buf.tail + piece)[-self._tail_bytes:]
        for q in list(buf.subs):
            try:
                q.put_nowait(piece)
            except asyncio.QueueFull:
                pass

    def subscribe(self, generation_id: str, *, allow_create: bool = False):
        """Returns (queue|None, replay_tail, done). New deltas arrive on the queue.

        queue is None when the generation is unknown (and creation is not
        allowed) or the per-generation subscriber cap is hit — the caller then
        emits a terminal frame. ONLY the trusted tee (feed) creates buffers; an
        arbitrary client ?generation_id= cannot allocate a buffer or pin memory.
        """
        buf = self._get(generation_id, create=allow_create)
        if buf is None:
            return None, "", True
        if len(buf.subs) >= MAX_SUBS_PER_GEN:
            return None, buf.tail, buf.done
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        buf.subs.add(q)
        return q, buf.tail, buf.done

    def unsubscribe(self, generation_id: str, q: asyncio.Queue) -> None:
        buf = self._buffers.get(generation_id)
        if buf is None:
            return
        buf.subs.discard(q)
        # free a completed generation's buffer once its last watcher leaves
        # (bounds retention of finished output text beyond the LRU cap).
        if buf.done and not buf.subs:
            self._buffers.pop(generation_id, None)

    def mark_done(self, generation_id: str) -> None:
        buf = self._buffers.get(generation_id)
        if buf is None:
            return
        buf.done = True
        for q in list(buf.subs):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(None)  # done sentinel


# ============================================================================
# Live slots poller — tok/s + progress (metrics plane)
# ============================================================================
_ACTIVE_STATES = {"ACTIVE", "ACTIVE_MATCH"}
_LOADING_STATES = {"STAGED", "PRE_LOADING", "LOADING", "READY"}


class LiveSlotsPoller:
    """Single ~1Hz observer of the active sidecar's /slots. ONE reader regardless
    of FE client count (clients read mgr.live_generation via /status)."""

    def __init__(self, mgr, interval_s: float = 1.0) -> None:
        self._mgr = mgr
        self._interval = max(0.1, float(interval_s))
        self._client = httpx.AsyncClient(timeout=SLOTS_TIMEOUT_S)
        self._samples: dict = {}          # id_task -> baseline {last_n_decoded,last_t,n_prompt,max_tokens,last_change_t}
        self._agg_ewma: float | None = None
        self._agg_gen_id: str | None = None
        self._schema_warned = False

    async def run(self) -> None:
        try:
            while not self._mgr._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.warning("live poller tick failed", exc_info=True)
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, self._interval - elapsed))
        except asyncio.CancelledError:
            raise
        finally:
            with contextlib.suppress(Exception):
                await self._client.aclose()

    def _store(self, gen: dict) -> None:
        mgr = self._mgr
        prev = mgr.live_generation
        mgr.live_generation = gen
        state = gen["state"]
        changed = (prev or {}).get("state") != state
        active = state in ("generating", "prefill", "finishing", "stalled", "loading")
        if active or changed:
            # contentless nudge so the existing useStatus WS-refetch pulls fresh
            # /status at ~1Hz; carries NO metrics/text (denylist untouched).
            with contextlib.suppress(Exception):
                mgr.event_bus.publish_nowait({"event": "generation_tick"})

    async def _tick(self) -> None:
        mgr = self._mgr
        # --- await-free identity snapshot (the SAME read status_snapshot does) ---
        slot = mgr._active_slot
        handle = mgr._active_handle
        if slot is None or handle is None:
            # _active_slot nulls before _active_handle during teardown -> a
            # lingering handle with no slot is the transition window.
            self._reset_samples()
            self._store(_base_generation("transitioning" if handle is not None else "idle"))
            return
        snap_handle = handle
        pid = getattr(handle, "pid", None)
        port = getattr(handle, "port", None)
        spawn_seq = getattr(mgr, "_spawn_seq", 0)
        # slot_id is a fresh uuid PER REQUEST (unique per generation), and the
        # SAME value the streaming tee reads off this same active slot. thread_id
        # is reused across grace-rematch turns of one conversation, so keying on
        # it would collide gen A's text with gen B — slot_id is the correct key.
        thread_or_slot = getattr(slot, "slot_id", None) or getattr(slot, "thread_id", None)
        state_v = getattr(getattr(slot, "state", None), "value", None)
        if port is None:
            self._store(_base_generation("transitioning"))
            return
        if state_v in _LOADING_STATES:
            self._reset_samples()
            self._store(_base_generation("loading"))
            return
        if state_v not in _ACTIVE_STATES:
            self._store(_base_generation("transitioning"))
            return

        # --- fetch /slots (the only place we await on external IO) ---
        try:
            resp = await self._client.get(f"http://127.0.0.1:{port}/slots")
            resp_t = time.monotonic()
            data = resp.json()
        except Exception:
            # slot vanished mid-swap/grace/teardown is EXPECTED; never crash.
            self._store(_base_generation("transitioning"))
            return

        # --- post-await identity re-validate (fixed-port 11500 reuse race) ---
        cur_handle = mgr._active_handle
        if (
            cur_handle is not snap_handle
            or cur_handle is None
            or getattr(cur_handle, "pid", None) != pid
            or getattr(mgr, "_spawn_seq", 0) != spawn_seq
        ):
            self._store(_base_generation("transitioning"))
            return

        gen = self._compute(data, resp_t, pid, spawn_seq, thread_or_slot)
        self._store(gen)

    def _reset_samples(self) -> None:
        self._samples.clear()
        self._agg_ewma = None
        self._agg_gen_id = None

    def _warn_schema(self, what: str) -> None:
        if not self._schema_warned:
            self._schema_warned = True
            log.warning("live poller: /slots schema mismatch (%s) — tok/s null until parser updated", what)

    def _compute(self, data, resp_t: float, pid, spawn_seq, thread_or_slot) -> dict:
        gen_id = compute_generation_id(pid, spawn_seq, thread_or_slot)
        proc = [s for s in (data or []) if isinstance(s, dict) and s.get("is_processing")]
        if not proc:
            self._reset_samples()
            return _base_generation("idle", generation_id=gen_id)

        total_inst = 0.0
        did_reset = False
        streaming = False
        headline: dict | None = None
        live_tasks: set = set()

        for s in proc:
            id_task = s.get("id_task")
            live_tasks.add(id_task)
            params = s.get("params") or {}
            nt = s.get("next_token") or []
            nt0 = nt[0] if nt else {}
            n_decoded = nt0.get("n_decoded")
            n_remain = nt0.get("n_remain")
            has_next = nt0.get("has_next_token")
            n_prompt = s.get("n_prompt_tokens")
            n_prompt_proc = s.get("n_prompt_tokens_processed")
            n_ctx = s.get("n_ctx")  # model context-window capacity (e.g. 250112)
            max_tokens = params.get("max_tokens")
            n_predict = params.get("n_predict")
            if params.get("stream"):
                streaming = True
            if n_decoded is None:
                self._warn_schema("next_token[0].n_decoded missing")
                continue

            smp = self._samples.get(id_task)
            # n_prompt_tokens GROWS during generation (= original prompt + tokens
            # generated so far), so it is NOT a reset signal — using it stalls
            # tok/s at "pending" forever. A new request on a recycled id_task is
            # caught by n_decoded regressing (a fresh generation starts low) or
            # by max_tokens changing.
            reset = (
                smp is None
                or n_decoded < smp["last_n_decoded"]
                or max_tokens != smp["max_tokens"]
            )
            if reset:
                did_reset = True
                self._samples[id_task] = {
                    "last_n_decoded": n_decoded, "last_t": resp_t,
                    "max_tokens": max_tokens,
                    "last_change_t": resp_t,
                }
                inst = 0.0
            else:
                dt = resp_t - smp["last_t"]
                if dt > STALE_GAP_S:
                    self._agg_ewma = None  # starvation: rate-only reset, keep generation
                dn = max(0, n_decoded - smp["last_n_decoded"])
                inst = dn / dt if dt > 1e-3 else 0.0
                if dn > 0:
                    smp["last_change_t"] = resp_t
                smp["last_n_decoded"] = n_decoded
                smp["last_t"] = resp_t
            total_inst += inst

            if headline is None or (n_decoded or 0) > headline["n_decoded"]:
                headline = {
                    "n_decoded": n_decoded or 0, "n_remain": n_remain,
                    "n_prompt": n_prompt, "n_prompt_proc": n_prompt_proc,
                    "n_ctx": n_ctx,
                    "max_tokens": max_tokens, "n_predict": n_predict,
                    "has_next": has_next,
                    "last_change_t": self._samples[id_task]["last_change_t"],
                }

        # GC vanished tasks
        for k in [k for k in self._samples if k not in live_tasks]:
            del self._samples[k]

        if headline is None:
            return _base_generation("transitioning", generation_id=gen_id)

        # aggregate EWMA, rebaselined on any generation boundary
        if did_reset or self._agg_gen_id != gen_id:
            self._agg_ewma = None
            self._agg_gen_id = gen_id
        if self._agg_ewma is None:
            if total_inst > 0:
                self._agg_ewma = total_inst
        else:
            self._agg_ewma = EWMA_ALPHA * total_inst + (1 - EWMA_ALPHA) * self._agg_ewma

        return self._derive(gen_id, headline, total_inst, streaming, len(proc), resp_t)

    def _derive(self, gen_id, hd, total_inst, streaming, riders, resp_t) -> dict:
        nd = hd["n_decoded"]
        max_tokens = hd["max_tokens"] if (hd["max_tokens"] or 0) > 0 else None
        effective_max = max_tokens if max_tokens else (hd["n_predict"] if (hd["n_predict"] or 0) > 0 else None)

        state = "generating"
        prompt_progress = None
        stalled = False
        tok_s: float | None = None

        if nd == 0 and hd["n_prompt"] and hd["n_prompt_proc"] is not None and hd["n_prompt_proc"] < hd["n_prompt"]:
            state = "prefill"
            prompt_progress = round(hd["n_prompt_proc"] / max(1, hd["n_prompt"]), 3)
        else:
            frozen_for = resp_t - hd["last_change_t"]
            if total_inst == 0.0 and frozen_for >= STALL_AFTER_S:
                if hd["has_next"] is False:
                    state, tok_s, stalled = "finishing", 0.0, False
                elif nd > 0:
                    state, tok_s, stalled = "stalled", 0.0, True
                else:
                    state, tok_s = "generating", None
            else:
                state = "generating"
                tok_s = None if self._agg_ewma is None else round(min(max(self._agg_ewma, 0.0), 10000.0), 1)

        tok_s_instant = round(min(max(total_inst, 0.0), 10000.0), 1)
        pct = round(min(max(nd / effective_max * 100.0, 0.0), 100.0), 1) if effective_max else None
        n_remain = hd["n_remain"] if isinstance(hd["n_remain"], int) and hd["n_remain"] >= 0 else None
        eta_s = None
        if (
            effective_max and self._agg_ewma and self._agg_ewma > 0
            and isinstance(n_remain, int) and 0 <= n_remain <= effective_max
        ):
            eta_s = round(n_remain / self._agg_ewma, 1)

        return {
            "state": state,
            "tok_s": tok_s,
            "tok_s_instant": tok_s_instant,
            "n_decoded": nd,
            "max_tokens": max_tokens,
            "n_remain": n_remain,
            "n_prompt_tokens": hd["n_prompt"] or 0,
            "n_ctx": hd.get("n_ctx"),
            "prompt_progress": prompt_progress,
            "pct": pct,
            "eta_s": eta_s,
            "stalled": stalled,
            "streaming": streaming,
            "generation_id": gen_id,
            "riders": riders,
            "measured_at_iso": utcnow_iso(),
        }
