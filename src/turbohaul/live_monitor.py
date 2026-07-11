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

from turbohaul.safety import _read_free_vram_all_mib, _read_total_vram_all_mib
from turbohaul.state import utcnow_iso


log = logging.getLogger(__name__)

# --- tuning constants (module-level; only enabled/poll_interval_s are config) ---
SLOTS_TIMEOUT_S = 1.5          # /slots GET timeout — caps a hung sidecar
STALL_AFTER_S = 10.0          # frozen (no decode AND no prompt progress) this long => stalled/finishing (2.0 was ~2 orders below legit warm prefill)
STALE_GAP_S = 5.0             # poll gap beyond this => rate-only reset (keep generation)
TRANSIENT_HOLD_TICKS = 3      # consecutive down-ticks before honoring a 'transitioning'/'idle' flip (debounce the generating<->transitioning flap)
EWMA_ALPHA = 0.4             # tok/s smoothing
TEXT_TAIL_BYTES = 16384       # live output is a TAIL (last ~few hundred tokens), not a transcript
MAX_LIVE_KEYS = 8            # LRU cap on concurrent per-generation text buffers
SSE_HEARTBEAT_S = 15.0        # ': keep-alive' cadence on the output SSE
MAX_SUBS_PER_GEN = 32        # cap concurrent SSE subscribers per generation (DoS guard)
CARRY_MAX_BYTES = 65536       # drop carry-over if a sidecar never frames (defensive)
# Mid-prefill hang alarm — separate clock for prefill stalls
# (decoding stalls use STALL_AFTER_S; prefill can legitimately run 30-60s on big contexts)
PREFILL_STALL_AFTER_S = 60.0  # prefill heartbeat frozen this long => alarm


def compute_generation_id(pid, spawn_seq, thread_or_slot) -> str:
    """Stable, non-reversible 8-hex id for a single generation.

    The SAME tuple is hashed by the poller (metrics plane) and the tee (text
    plane) so the FE never shows generation A's text under generation B's tok/s.
    ``spawn_seq`` makes a fixed-port (11500) sidecar reuse non-colliding.
    """
    raw = f"{pid}:{spawn_seq}:{thread_or_slot}".encode("utf-8", "replace")
    return hashlib.blake2b(raw, digest_size=4).hexdigest()  # 8 hex chars


def _read_spawn_seq(mgr) -> int:
    """Active resident's spawn_seq, defensively.

    The spawn counter for the live-monitor generation_id now lives on the
    active ``Resident`` (mirrored from the legacy global by the manager). Prefer
    the manager's ``_active_spawn_seq()`` accessor; fall back to the legacy
    ``_spawn_seq`` attribute when the manager double doesn't expose the method
    (older test stubs). At ``MAX_PARALLEL_SIDECARS == 1`` both yield the same
    value, so the generation_id is unchanged.
    """
    accessor = getattr(mgr, "_active_spawn_seq", None)
    if callable(accessor):
        return accessor()
    return getattr(mgr, "_spawn_seq", 0)


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
        "n_prompt_proc": None,
        "n_prompt_cache": None,
        "n_ctx": None,
        "prompt_progress": None,
        # Telemetry (KV-decoupled): integer prefill percentage
        # = round(n_prompt_tokens_processed / n_prompt_tokens * 100) while
        # is_processing and n_prompt_tokens > 0, else null. DISTINCT from
        # ``prompt_progress`` (a fraction shown only during the prefill state);
        # ``prefill_pct`` is populated whenever a slot is processing.
        "prefill_pct": None,
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
        # Flap debounce: hold the last ACTIVE generation across a
        # transient 'transitioning'/'idle' tick so the FE state doesn't flap
        # (see _store). Inherited by ResidentSlotsPoller but unused there (the
        # supervisor stores poll_once()'s return directly, never via _store).
        self._held_active_gen: dict | None = None
        self._transient_ticks = 0

    async def run(self) -> None:
        try:
            while not self._mgr._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    await self._tick()
                    # Cache per-GPU free VRAM at cap<=1 too, so
                    # status_snapshot can emit vram[]. The cap>=2 supervisor
                    # already does this; the legacy single poller did NOT, so
                    # /status.vram was null under single-residency (blank bars).
                    await self._refresh_vram()
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

    async def _refresh_vram(self) -> None:
        """Cache per-GPU free VRAM off the hot path so status_snapshot emits
        vram[] await-free at cap<=1 too (the cap>=2 LiveResidentsSupervisor does
        the same via its own _refresh_vram). Probe failure -> None (vram[] null,
        never stale). Skipped once shutdown begins. Previously only the
        supervisor refreshed VRAM, so /status.vram was null under single-residency."""
        if self._mgr._stop_event.is_set():
            return
        try:
            self._mgr._vram_free_mib = await asyncio.to_thread(_read_free_vram_all_mib)
        except Exception:  # noqa: BLE001 -- probe failure -> null, never stale/raise
            self._mgr._vram_free_mib = None
            log.warning("VRAM free probe failed — setting _vram_free_mib=None")

        # One-time boot read of total VRAM per GPU (never changes at runtime)
        if self._mgr._vram_total_mib is None:
            try:
                self._mgr._vram_total_mib = await asyncio.to_thread(_read_total_vram_all_mib)
            except Exception:  # noqa: BLE001
                self._mgr._vram_total_mib = None
                log.warning("VRAM total probe failed — setting _vram_total_mib=None")

        # Telemetry — VRAM + process memory sample
        try:
            tel = getattr(self._mgr, "_telemetry", None)
            if tel is not None:
                import resource
                rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                tel.on_vram_sample(self._mgr._vram_free_mib, rss_mib)
        except Exception:
            pass  # observe-only

    _ACTIVE_GEN_STATES = ("generating", "prefill", "finishing", "stalled", "loading")

    def _store(self, gen: dict) -> None:
        mgr = self._mgr
        state = gen["state"]
        active = state in self._ACTIVE_GEN_STATES
        if active:
            # Active tick: snap UP instantly (never debounce responsiveness),
            # remember the real generation, reset the down-edge debounce.
            self._held_active_gen = gen
            self._transient_ticks = 0
            out = gen
        elif (
            self._held_active_gen is not None
            and gen.get("generation_id") is not None
            and gen.get("generation_id") != self._held_active_gen.get("generation_id")
        ):
            # A DIFFERENT generation is already visible to the poller: do NOT keep
            # holding the old one — its generation_id is the
            # anchor the live-output SSE follows, so holding it while a new gen
            # decodes would mis-key the new gen's text under the old gen's id (the
            # exact stale-generation bug the anchor-follow design prevents). Publish
            # through immediately and drop the hold.
            out = gen
            self._held_active_gen = None
            self._transient_ticks = 0
        elif self._held_active_gen is not None:
            # FLAP DEBOUNCE (guards against a flaky-network flap): a single
            # 'transitioning'/'idle' tick right after an active one is almost
            # always a TRANSIENT — a post-await identity-revalidate miss (a
            # concurrent spawn_seq bump under rapid concurrent traffic) or a
            # momentary /slots gap — not a real stop. Honoring it instantly made
            # the FE flap generating<->transitioning every ~1s AND reset the
            # tok/s EWMA so it never populated. Hold the last REAL active gen
            # until TRANSIENT_HOLD_TICKS consecutive down-ticks confirm a genuine
            # stop (the software analog of raising a failover threshold). A real
            # active tick above resets the counter, so a true end still resolves
            # to idle/transitioning within ~TRANSIENT_HOLD_TICKS polls (≤~3s lag
            # on a genuine END — an accepted tradeoff vs the flap; the SSE text
            # plane ends independently via its own done sentinel).
            self._transient_ticks += 1
            if self._transient_ticks < TRANSIENT_HOLD_TICKS:
                out = self._held_active_gen
            else:
                out = gen
                self._held_active_gen = None
        else:
            out = gen
        prev = mgr.live_generation
        mgr.live_generation = out
        out_state = out["state"]
        changed = (prev or {}).get("state") != out_state
        active_out = out_state in self._ACTIVE_GEN_STATES
        if active_out or changed:
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
        gen = await self._fetch_revalidate_compute(
            slot, handle, _read_spawn_seq(mgr),
            cur_handle_getter=lambda: mgr._active_handle,
            cur_spawn_getter=lambda: _read_spawn_seq(mgr),
        )
        self._store(gen)

    async def _fetch_revalidate_compute(
        self, slot, handle, spawn_seq, *, cur_handle_getter, cur_spawn_getter
    ) -> dict:
        """Shared metrics core (legacy single-poller AND the per-resident poller).

        Given a CAPTURED ``(slot, handle, spawn_seq)`` identity, run the loading/active
        state machine, fetch ``/slots`` (the only external await), re-validate identity
        AFTER the await against the live source (the fixed-port 11500 reuse race), and
        return the generation dict. PURE: never stores — the caller routes the result
        to ``mgr.live_generation`` (legacy) or ``mgr.live_generations[tag]`` (per
        resident). ``cur_handle_getter``/``cur_spawn_getter`` re-read the CURRENT handle
        + spawn_seq (the legacy globals, or the resident's fields)."""
        snap_handle = handle
        pid = getattr(handle, "pid", None)
        port = getattr(handle, "port", None)
        # slot_id is a fresh uuid PER REQUEST (unique per generation), and the
        # SAME value the streaming tee reads off this same active slot. thread_id
        # is reused across grace-rematch turns of one conversation, so keying on
        # it would collide gen A's text with gen B — slot_id is the correct key.
        thread_or_slot = getattr(slot, "slot_id", None) or getattr(slot, "thread_id", None)
        state_v = getattr(getattr(slot, "state", None), "value", None)
        if port is None:
            return _base_generation("transitioning")
        if state_v in _LOADING_STATES:
            self._reset_samples()
            return _base_generation("loading")
        if state_v not in _ACTIVE_STATES:
            return _base_generation("transitioning")

        # --- fetch /slots (the only place we await on external IO) ---
        try:
            resp = await self._client.get(f"http://127.0.0.1:{port}/slots")
            resp_t = time.monotonic()
            data = resp.json()
        except Exception:
            # slot vanished mid-swap/grace/teardown is EXPECTED; never crash.
            return _base_generation("transitioning")

        # --- post-await identity re-validate (fixed-port 11500 reuse race) ---
        cur_handle = cur_handle_getter()
        if (
            cur_handle is not snap_handle
            or cur_handle is None
            or getattr(cur_handle, "pid", None) != pid
            or cur_spawn_getter() != spawn_seq
        ):
            return _base_generation("transitioning")

        return self._compute(data, resp_t, pid, spawn_seq, thread_or_slot)

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
            n_prompt_cache = s.get("n_prompt_tokens_cache")
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
                    "last_n_prompt_proc": n_prompt_proc,
                    "dproc": 0,
                }
                inst = 0.0
            else:
                dt = resp_t - smp["last_t"]
                if dt > STALE_GAP_S:
                    self._agg_ewma = None  # starvation: rate-only reset, keep generation
                dn = max(0, n_decoded - smp["last_n_decoded"])
                inst = dn / dt if dt > 1e-3 else 0.0
                # Prompt-processing progress IS activity. The engine
                # resets n_decoded only at DONE_PROMPT (server-context.cpp:4049), so a
                # warm follow-up's whole prefill reads as 'frozen n_decoded>0' and
                # false-alarmed STALLED after 2s. Rising n_prompt_tokens_processed
                # resets the freeze clock exactly like decoded tokens do.
                _prev_proc = smp.get("last_n_prompt_proc")
                _dproc = (n_prompt_proc - _prev_proc) if (n_prompt_proc is not None and _prev_proc is not None) else 0
                if dn > 0 or _dproc > 0:
                    smp["last_change_t"] = resp_t
                smp["dproc"] = _dproc
                smp["last_n_prompt_proc"] = n_prompt_proc
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
                    "dproc": self._samples[id_task].get("dproc", 0),
                    "n_prompt_cache": n_prompt_cache,
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

        # Prefill keys on LIVE prompt
        # progress THIS tick (dproc>0) — never on proc<n_prompt. On this fork
        # n_prompt_tokens includes cached+generated tokens and grows through
        # decode, while n_prompt_tokens_processed counts only NEW prompt tokens
        # and freezes at DONE_PROMPT (server-context.cpp:855/:3752/:3964/:4004),
        # so proc<n_prompt holds through ALL of decode and would classify a
        # genuine decode-stall as 'prefill' forever, masking the red alarm.
        # Warm prefills stay out of the stall path via the freeze clock (rising
        # proc resets last_change_t above); the stalled guard is the original
        # nd>0 — reachable again during decode.
        if total_inst == 0.0 and (hd.get("dproc") or 0) > 0:
            state = "prefill"
            if hd["n_prompt"] and hd["n_prompt_proc"] is not None:
                prompt_progress = round(hd["n_prompt_proc"] / max(1, hd["n_prompt"]), 3)
        else:
            frozen_for = resp_t - hd["last_change_t"]
            if total_inst == 0.0 and frozen_for >= STALL_AFTER_S:
                if not hd["has_next"]:
                    state, tok_s, stalled = "finishing", 0.0, False
                elif nd > 0:
                    state, tok_s, stalled = "stalled", 0.0, True
                else:
                    state, tok_s = "generating", None
            else:
                state = "generating"
                tok_s = None if self._agg_ewma is None else round(min(max(self._agg_ewma, 0.0), 10000.0), 1)

        tok_s_instant = round(min(max(total_inst, 0.0), 10000.0), 1)
        # Telemetry: integer prefill % from the same /slots headline
        # (is_processing is implied — _derive is only reached with a processing
        # slot). Guard divide-by-zero on n_prompt_tokens; null when the counter
        # is absent. DISTINCT from prompt_progress (fraction, prefill-state only).
        n_prompt_hd = hd["n_prompt"] or 0
        n_prompt_proc_hd = hd["n_prompt_proc"]
        # Respin: proc counts only NEW prompt tokens; the reused prefix lives in
        # n_prompt_tokens_cache. A truthful % = (cache+proc)/n_prompt, clamped
        # (n_prompt grows during decode, drifting the ratio down).
        _cache_hd = hd.get("n_prompt_cache")
        _pref_num = (
            n_prompt_proc_hd + _cache_hd
            if (n_prompt_proc_hd is not None and _cache_hd is not None)
            else n_prompt_proc_hd
        )
        prefill_pct = (
            round(min(_pref_num / n_prompt_hd, 1.0) * 100)
            if (n_prompt_hd > 0 and _pref_num is not None)
            else None
        )
        pct = round(min(max(nd / effective_max * 100.0, 0.0), 100.0), 1) if effective_max else None
        n_remain = hd["n_remain"] if isinstance(hd["n_remain"], int) and hd["n_remain"] >= 0 else None
        eta_s = None
        if (
            effective_max and self._agg_ewma and self._agg_ewma > 0
            and isinstance(n_remain, int) and 0 <= n_remain <= effective_max
        ):
            eta_s = round(n_remain / self._agg_ewma, 1)

        # Prefill-stall alarm flag (separate from decoding stall).
        # Prefill can legitimately run 30-60s on large contexts; the decoding
        # stall clock (STALL_AFTER_S=10s) is too aggressive for prefill.
        # Alarm when prefill heartbeat (n_prompt_proc + n_prompt_cache) is
        # frozen for PREFILL_STALL_AFTER_S.
        # FIX (PL early gate): state=='prefill' requires dproc>0 this tick, which
        # resets last_change_t -> frozen_for≈0. The wedge case (cold prefill,
        # nd==0, proc frozen) classifies as 'generating' in the else branch.
        # Use state-independent check: nd==0 (cold wedge) + proc>0 (prefill started)
        # + frozen >= 60s (stalled). FE banner keys ONLY on prefill_stall_alarm.
        prefill_stall_alarm = False
        _frozen_for = resp_t - hd["last_change_t"]
        if (
            total_inst == 0.0
            and _frozen_for >= PREFILL_STALL_AFTER_S
            and nd == 0
            and (hd.get("n_prompt_proc") or 0) > 0
        ):
            prefill_stall_alarm = True

        return {
            "state": state,
            "tok_s": tok_s,
            "tok_s_instant": tok_s_instant,
            "n_decoded": nd,
            "max_tokens": max_tokens,
            "n_remain": n_remain,
            "n_prompt_tokens": hd["n_prompt"] or 0,
            # Raw prefill counters for the FE growth bar (cache-restored
            # + newly-processed both rise through prefill; n_prompt itself fills
            # as it is processed so it cannot be the denominator mid-flight).
            "n_prompt_proc": hd["n_prompt_proc"],
            "n_prompt_cache": hd.get("n_prompt_cache"),
            "n_ctx": hd.get("n_ctx"),
            "prompt_progress": prompt_progress,
            "prefill_pct": prefill_pct,
            "pct": pct,
            "eta_s": eta_s,
            "stalled": stalled,
            "streaming": streaming,
            "generation_id": gen_id,
            "riders": riders,
            "measured_at_iso": utcnow_iso(),
            # Mid-prefill hang alarm (observability-only, drives
            # no FSM decision). FE renders a red banner when true.
            "prefill_stall_alarm": prefill_stall_alarm,
        }


# ============================================================================
# Per-resident metrics — cap>=2 multi-sidecar live monitor
# ============================================================================
class ResidentSlotsPoller(LiveSlotsPoller):
    """Per-resident /slots poller (cap>=2). Reuses the LiveSlotsPoller compute / rate
    EWMA / fetch+revalidate core VERBATIM (with its OWN ``_samples``/EWMA state, so
    per-model rates never cross-contaminate), but reads identity from a SPECIFIC
    ``Resident`` (its ``handle``/``active_slot``/``spawn_seq``) instead of the legacy
    ``_active_*`` globals, and RETURNS the generation block (the supervisor stores it
    into ``mgr.live_generations[model_tag]``). It never runs its own loop — the
    supervisor owns the ~1Hz cadence and calls :meth:`poll_once`."""

    async def poll_once(self, resident) -> dict:
        """Poll this resident's sidecar once and RETURN its generation block. Reads the
        resident AWAIT-FREE (the same pure-observer discipline the legacy poller uses
        on the globals + status_snapshot uses on the FSM scalars), then re-validates
        the resident's handle + spawn_seq AFTER the /slots await (a resident can be
        evicted + its fixed port reused mid-await)."""
        slot = getattr(resident, "active_slot", None)
        handle = getattr(resident, "handle", None)
        if slot is None or handle is None:
            self._reset_samples()
            return _base_generation("transitioning" if handle is not None else "idle")
        return await self._fetch_revalidate_compute(
            slot, handle, getattr(resident, "spawn_seq", 0),
            cur_handle_getter=lambda: getattr(resident, "handle", None),
            cur_spawn_getter=lambda: getattr(resident, "spawn_seq", 0),
        )


class LiveResidentsSupervisor:
    """cap>=2 live-inference monitor: ONE ~1Hz task that polls EVERY live resident's
    ``/slots`` CONCURRENTLY and writes each resident's generation into
    ``mgr.live_generations`` (keyed by ``model_tag``), mirroring the most-recently-
    active resident's generation into ``mgr.live_generation`` (the back-compat alias
    ``status_snapshot`` + ``live_stream.py`` already read). It also refreshes the
    per-GPU free-VRAM cache (``mgr._vram_free_mib``) OFF the hot path so
    ``status_snapshot`` can emit ``vram[]`` await-free (it must never call nvidia-smi
    synchronously — it is lock-free).

    PURE OBSERVER: never holds ``_registry_lock``, never mutates FSM/resident state.
    Cancelled FIRST in ``shutdown()`` (before the drivers) so no poll races a tearing-
    down sidecar — the multi-resident analogue of 'stop the single observer first'.
    Holds ONE ``ResidentSlotsPoller`` per resident (per-model rate state); GCs a
    poller + its ``live_generations`` entry + closes its httpx client when the
    resident vanishes."""

    def __init__(self, mgr, interval_s: float = 1.0) -> None:
        self._mgr = mgr
        self._interval = max(0.1, float(interval_s))
        self._pollers: dict[str, ResidentSlotsPoller] = {}  # model_tag -> poller

    async def run(self) -> None:
        try:
            while not self._mgr._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 -- observer must survive any tick error
                    log.warning("live residents supervisor tick failed", exc_info=True)
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, self._interval - elapsed))
        except asyncio.CancelledError:
            raise
        finally:
            await self._close_all()

    async def _tick(self) -> None:
        mgr = self._mgr
        residents = mgr._model_residents()  # await-free; excludes singleton + DEAD
        live_tags: set[str] = set()
        polls = []
        for r in residents:
            tag = r.model_tag
            if tag is None:
                continue
            live_tags.add(tag)
            poller = self._pollers.get(tag)
            if poller is None:
                poller = ResidentSlotsPoller(mgr, interval_s=self._interval)
                self._pollers[tag] = poller
            polls.append(self._poll_one(tag, poller, r))
        if polls:
            await asyncio.gather(*polls, return_exceptions=True)
        await self._gc_vanished(live_tags)
        self._update_primary_alias(residents)
        await self._refresh_vram()

    async def _poll_one(self, tag: str, poller: ResidentSlotsPoller, resident) -> None:
        gen = await poller.poll_once(resident)
        # The resident can be EVICTED/DEAD during the /slots await. poll_once's
        # post-await re-validate catches a handle SWAP, but not a same-handle DEAD
        # transition — so re-check liveness here before writing, else we'd publish a
        # ZOMBIE generation that _gc_vanished only clears next tick (the tag is still
        # in this tick's live_tags). No await between the check and the write, so the
        # single-threaded loop makes them atomic.
        if self._mgr._live_resident_for(tag) is None:
            # Evicted/dead mid-poll: DROP any stale gen NOW (not just skip the write).
            # _update_primary_alias runs right after this tick's gather — BEFORE
            # _gc_vanished can act (the tag is still in this tick's live_tags) — so
            # leaving a prior tick's 'generating' entry here would let the alias mirror
            # a DEAD resident's stale gen for one tick (PL pre-cutover polish #2).
            self._mgr.live_generations.pop(tag, None)
            return
        self._mgr.live_generations[tag] = gen
        # contentless WS nudge so the FE refetches /status ~1Hz (denylist untouched).
        if gen.get("state") in ("generating", "prefill", "finishing", "stalled", "loading"):
            with contextlib.suppress(Exception):
                self._mgr.event_bus.publish_nowait({"event": "generation_tick"})

    async def _gc_vanished(self, live_tags: set[str]) -> None:
        for tag in list(self._pollers):
            if tag not in live_tags:
                gone = self._pollers.pop(tag, None)
                self._mgr.live_generations.pop(tag, None)
                if gone is not None:
                    with contextlib.suppress(Exception):
                        await gone._client.aclose()

    def _update_primary_alias(self, residents) -> None:
        """Mirror the most-recently-active resident's generation into the back-compat
        ``live_generation`` alias (None -> status_snapshot/live_stream fall back to
        idle). 'Most recently active' matches the dispatcher's warm-hint notion."""
        mgr = self._mgr
        best = None
        best_t: float | None = None
        for r in residents:
            g = mgr.live_generations.get(r.model_tag)
            if g is None:
                continue
            t = getattr(r, "last_active_monotonic", 0.0)
            if best_t is None or t > best_t:
                best, best_t = g, t
        mgr.live_generation = best

    async def _refresh_vram(self) -> None:
        """Cache per-GPU free VRAM off the hot path (nvidia-smi is a ~subprocess);
        status_snapshot reads the cache await-free. Probe failure -> None (vram[]
        becomes null, never stale). Skip once shutdown has begun so we don't kick off
        a fresh nvidia-smi thread that would outlive the cancelled supervisor (the
        subprocess has its own 5s timeout, but starting one at teardown is wasteful)."""
        if self._mgr._stop_event.is_set():
            return
        try:
            self._mgr._vram_free_mib = await asyncio.to_thread(_read_free_vram_all_mib)
        except Exception:  # noqa: BLE001 -- probe failure -> null, never stale/raise
            self._mgr._vram_free_mib = None
            log.warning("VRAM free probe failed — setting _vram_free_mib=None")

    async def _close_all(self) -> None:
        for poller in list(self._pollers.values()):
            with contextlib.suppress(Exception):
                await poller._client.aclose()
        self._pollers.clear()
        # Drop the per-resident generations so a post-shutdown
        # /ui/live/output/stream?model_tag= can't anchor-follow a DEAD generation
        # (the GC path only runs while the supervisor loops; this is the teardown leg).
        self._mgr.live_generations.clear()
        self._mgr.live_generation = None
