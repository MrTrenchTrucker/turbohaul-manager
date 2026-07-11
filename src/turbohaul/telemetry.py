"""Flap/degradation telemetry — persistent JSONL log + read endpoint.

An OBSERVE-ONLY logging/telemetry subsystem that captures, with timestamps,
the signals needed to diagnose the "flapping over time" degradation in
Turbohaul Manager's single-residency gateway.

Design principles:
  - MUST NOT change inference behavior or break /status
  - Low overhead: append-only JSONL, no locks on the hot path
  - FE-independent: pure backend, no frontend dependency
  - Rotating file: MAX_BYTES per file, MAX_FILES retained
  - Survives restart: written under a mounted volume

Integration points (called from manager.py + chat_completion.py):
  - telemetry.on_request_arrival(slot)
  - telemetry.on_queue_state(queue_depth, slot)
  - telemetry.on_slot_assign(slot)
  - telemetry.on_prefill_start(slot)
  - telemetry.on_first_token(slot, ttft_ms)
  - telemetry.on_generation_tick(slot, n_decoded, tok_s)
  - telemetry.on_keep_alive_emitted(slot_id, is_queued)
  - telemetry.on_client_disconnect(slot, reason, elapsed_s)
  - telemetry.on_completion(slot, reason)
  - telemetry.on_vram_sample()
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
DEFAULT_LOG_DIR = Path("/var/lib/turbohaul/telemetry")
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB per file
MAX_FILES = 5  # retain last 5 rotated files
FLUSH_INTERVAL_S = 5.0  # background flush cadence

# In-memory ring buffer for recent events (read endpoint serves this when
# JSONL files are not available or for real-time dashboards).
RING_BUFFER_SIZE = 10_000


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# JSONL Writer — append-only, rotating
# ---------------------------------------------------------------------------
class _JsonlWriter:
    """Thread-safe, append-only JSONL writer with rotation.

    Writes one JSON object per line. Rotates when current file exceeds
    MAX_FILE_BYTES. Retains up to MAX_FILES old files.
    """

    def __init__(self, log_dir: Path, max_bytes: int = MAX_FILE_BYTES,
                 max_files: int = MAX_FILES) -> None:
        self._log_dir = log_dir
        self._max_bytes = max_bytes
        self._max_files = max_files
        self._lock = threading.Lock()
        self._file: Any = None  # file handle
        self._current_path: Path | None = None
        self._ensure_dir()
        self._open_new()

    def _ensure_dir(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _open_new(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._current_path = self._log_dir / f"flap_{ts}.jsonl"
        self._file = open(self._current_path, "a", buffering=1)  # line-buffered
        log.info("telemetry: opened %s", self._current_path)

    def _rotate(self) -> None:
        if self._file:
            self._file.close()
        self._open_new()
        self._prune()

    def _prune(self) -> None:
        files = sorted(self._log_dir.glob("flap_*.jsonl"))
        while len(files) > self._max_files:
            victim = files.pop(0)
            try:
                victim.unlink()
                log.info("telemetry: pruned old log %s", victim)
            except OSError:
                pass

    def write(self, obj: dict) -> None:
        """Append a JSON-serializable dict as one JSONL line."""
        with self._lock:
            line = json.dumps(obj, ensure_ascii=False, default=str)
            if self._current_path and self._current_path.stat().st_size > self._max_bytes:
                self._rotate()
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None

    def read_recent(self, limit: int = 200, event_type: str | None = None,
                    since_id: int = 0) -> list[dict]:
        """Read recent events from all JSONL files (newest first, then reversed).

        This is a best-effort read — scans files in mtime order. Not intended
        for pagination of millions of events; use the ring buffer for hot data.
        """
        all_lines: list[str] = []
        files = sorted(self._log_dir.glob("flap_*.jsonl"), key=lambda p: p.stat().st_mtime)
        for fp in files:
            try:
                with open(fp) as fh:
                    all_lines.extend(fh.readlines())
            except OSError:
                continue

        results: list[dict] = []
        for line in reversed(all_lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_id and obj.get("_seq", 0) <= since_id:
                continue
            if event_type and obj.get("event_type") != event_type:
                continue
            results.append(obj)
            if len(results) >= limit:
                break

        return results


# ---------------------------------------------------------------------------
# Ring buffer — in-memory, lock-free-ish (single-writer safe via GIL)
# ---------------------------------------------------------------------------
class _RingBuffer:
    """Simple append-only ring buffer for recent events."""

    def __init__(self, capacity: int = RING_BUFFER_SIZE) -> None:
        self._capacity = capacity
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._seq = 0

    def append(self, obj: dict) -> int:
        with self._lock:
            self._seq += 1
            obj["_seq"] = self._seq
            self._buf.append(obj)
            if len(self._buf) > self._capacity:
                self._buf = self._buf[-self._capacity:]
            return self._seq

    def get(self, limit: int = 200, event_type: str | None = None,
            since_id: int = 0) -> tuple[list[dict], int | None]:
        """Return (events, next_since). events are oldest-first."""
        with self._lock:
            filtered = [
                e for e in self._buf
                if e["_seq"] > since_id
                and (event_type is None or e.get("event_type") == event_type)
            ]
            if len(filtered) > limit:
                filtered = filtered[-limit:]
            next_since = filtered[-1]["_seq"] if filtered and len(filtered) == limit else None
            return filtered, next_since


# ---------------------------------------------------------------------------
# Telemetry singleton
# ---------------------------------------------------------------------------
class FlapTelemetry:
    """Main telemetry interface. Instantiate once at manager init."""

    def __init__(self, log_dir: Path | str | None = None,
                 enabled: bool = True) -> None:
        self._enabled = enabled
        self._log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self._writer = _JsonlWriter(self._log_dir) if enabled else None
        self._ring = _RingBuffer() if enabled else None

        # Per-slot tracking (lightweight, keyed by slot_id)
        self._slot_arrival: dict[str, float] = {}
        self._slot_queue_enter: dict[str, float] = {}
        self._slot_prefill_start: dict[str, float] = {}
        self._slot_first_token_at: dict[str, float] = {}
        self._slot_keep_alives: dict[str, int] = defaultdict(int)

        # Background VRAM/memory sampler state
        self._last_vram_sample_iso: str | None = None
        self._vram_sample_errors: int = 0

    def close(self) -> None:
        if self._writer:
            self._writer.close()

    # --- internal helpers ---

    def _emit(self, event_type: str, payload: dict) -> None:
        if not self._enabled:
            return
        payload["event_type"] = event_type
        payload["ts"] = utcnow_iso()
        self._ring.append(payload)
        self._writer.write(payload)

    # --- request lifecycle hooks ---

    def on_request_arrival(self, slot: Any) -> None:
        """Called when a new request arrives at the API layer."""
        slot_id = getattr(slot, "slot_id", "unknown")
        self._slot_arrival[slot_id] = time.monotonic()
        self._emit("request_arrival", {
            "slot_id": slot_id,
            "model_tag": getattr(slot, "model_tag", ""),
            "thread_id": getattr(slot, "thread_id", ""),
            "has_context": bool(getattr(slot, "context", None)),
        })

    def on_queue_state(self, queue_depth: dict, slot: Any) -> None:
        """Called when a slot enters the queue (staging or accept-buffer)."""
        slot_id = getattr(slot, "slot_id", "unknown")
        now = time.monotonic()
        self._slot_queue_enter[slot_id] = now
        self._emit("queue_enter", {
            "slot_id": slot_id,
            "thread_id": getattr(slot, "thread_id", ""),
            "staging_depth": queue_depth.get("staging_queue_depth", 0),
            "staging_max": queue_depth.get("staging_queue_max", 0),
            "acceptance_depth": queue_depth.get("acceptance_buffer_depth", 0),
            "slot_state": getattr(getattr(slot, "state", None), "value", ""),
        })

    def on_slot_assign(self, slot: Any) -> None:
        """Called when a slot is assigned to an active sidecar."""
        slot_id = getattr(slot, "slot_id", "unknown")
        now = time.monotonic()
        wait_s = None
        if slot_id in self._slot_queue_enter:
            wait_s = round(now - self._slot_queue_enter[slot_id], 2)
        self._emit("slot_assign", {
            "slot_id": slot_id,
            "thread_id": getattr(slot, "thread_id", ""),
            "model_tag": getattr(slot, "model_tag", ""),
            "wait_in_queue_s": wait_s,
            "pid": getattr(slot, "pid", None),
            "port": getattr(slot, "port", None),
        })

    def on_prefill_start(self, slot: Any) -> None:
        """Called when prefill begins for a slot."""
        slot_id = getattr(slot, "slot_id", "unknown")
        self._slot_prefill_start[slot_id] = time.monotonic()
        self._emit("prefill_start", {
            "slot_id": slot_id,
            "thread_id": getattr(slot, "thread_id", ""),
            "model_tag": getattr(slot, "model_tag", ""),
        })

    def on_first_token(self, slot: Any, ttft_ms: float) -> None:
        """Called when the first token is received (TTFT measurement)."""
        slot_id = getattr(slot, "slot_id", "unknown")
        self._slot_first_token_at[slot_id] = time.monotonic()
        self._emit("first_token", {
            "slot_id": slot_id,
            "thread_id": getattr(slot, "thread_id", ""),
            "model_tag": getattr(slot, "model_tag", ""),
            "ttft_ms": round(ttft_ms, 1),
        })

    def on_generation_tick(self, slot: Any, n_decoded: int, tok_s: float) -> None:
        """Called periodically during generation (from live_monitor poller).

        NOT emitted on every tick — throttled to avoid log spam.
        Emitted every N seconds per slot.
        """
        slot_id = getattr(slot, "slot_id", "unknown")
        last = self._slot_first_token_at.get(slot_id, 0)
        if last > 0 and (time.monotonic() - last) < 1.0:
            return  # skip sub-second duplicates
        self._emit("generation_tick", {
            "slot_id": slot_id,
            "thread_id": getattr(slot, "thread_id", ""),
            "n_decoded": n_decoded,
            "tok_s": round(tok_s, 1),
        })

    def on_keep_alive_emitted(self, slot_id: str, is_queued: bool,
                               thread_id: str = "") -> None:
        """Called when a keep-alive comment is emitted during stream-OPEN.

        KEY signal for flap diagnosis: are QUEUED requests getting heartbeats?
        """
        self._slot_keep_alives[slot_id] += 1
        self._emit("keep_alive", {
            "slot_id": slot_id,
            "thread_id": thread_id,
            "is_queued": is_queued,
            "count_for_slot": self._slot_keep_alives[slot_id],
        })

    def on_client_disconnect(self, slot: Any, reason: str,
                             elapsed_s: float) -> None:
        """Called when a client disconnects (mid-stream or mid-queue)."""
        slot_id = getattr(slot, "slot_id", "unknown")
        self._emit("client_disconnect", {
            "slot_id": slot_id,
            "thread_id": getattr(slot, "thread_id", ""),
            "model_tag": getattr(slot, "model_tag", ""),
            "reason": reason,
            "elapsed_s": round(elapsed_s, 2),
            "was_in_queue": slot_id in self._slot_queue_enter and slot_id not in self._slot_prefill_start,
            "keep_alives_sent": self._slot_keep_alives.get(slot_id, 0),
        })
        self._cleanup_slot(slot_id)

    def on_completion(self, slot: Any, reason: str) -> None:
        """Called when a slot completes successfully."""
        slot_id = getattr(slot, "slot_id", "unknown")
        now = time.monotonic()
        total_s = None
        if slot_id in self._slot_arrival:
            total_s = round(now - self._slot_arrival[slot_id], 2)
        self._emit("completion", {
            "slot_id": slot_id,
            "thread_id": getattr(slot, "thread_id", ""),
            "model_tag": getattr(slot, "model_tag", ""),
            "reason": reason,
            "total_lifecycle_s": total_s,
            "keep_alives_sent": self._slot_keep_alives.get(slot_id, 0),
        })
        self._cleanup_slot(slot_id)

    def on_vram_sample(self, vram_free_mib: list[int] | None,
                       process_rss_mib: float | None = None) -> None:
        """Called periodically (from supervisor) to log VRAM + memory state."""
        self._emit("resource_sample", {
            "vram_free_mib": vram_free_mib,
            "process_rss_mib": round(process_rss_mib, 1) if process_rss_mib else None,
            "vram_sample_errors": self._vram_sample_errors,
        })
        self._last_vram_sample_iso = utcnow_iso()

    def on_slot_state_change(self, slot: Any, old_state: str,
                              new_state: str) -> None:
        """Called when a slot transitions between states."""
        self._emit("slot_state_change", {
            "slot_id": getattr(slot, "slot_id", "unknown"),
            "thread_id": getattr(slot, "thread_id", ""),
            "model_tag": getattr(slot, "model_tag", ""),
            "old_state": old_state,
            "new_state": new_state,
        })

    # --- read API ---

    def get_events(self, limit: int = 200, event_type: str | None = None,
                   since_id: int = 0) -> dict:
        """Query recent events from the ring buffer (hot)."""
        if not self._ring:
            return {"events": [], "next_since": None, "source": "disabled"}
        events, next_since = self._ring.get(limit=limit, event_type=event_type,
                                            since_id=since_id)
        return {
            "events": events,
            "next_since": next_since,
            "source": "ring_buffer",
        }

    def get_events_from_file(self, limit: int = 200,
                             event_type: str | None = None,
                             since_id: int = 0) -> dict:
        """Query events from JSONL files (persistent, full history)."""
        if not self._writer:
            return {"events": [], "next_since": None, "source": "disabled"}
        events = self._writer.read_recent(limit=limit, event_type=event_type,
                                           since_id=since_id)
        next_since = events[-1].get("_seq") if events else None
        return {
            "events": events,
            "next_since": next_since,
            "source": "jsonl",
        }

    def get_status(self) -> dict:
        """Telemetry subsystem health + stats."""
        return {
            "enabled": self._enabled,
            "log_dir": str(self._log_dir),
            "last_vram_sample_at": self._last_vram_sample_iso,
            "vram_sample_errors": self._vram_sample_errors,
            "active_slots_tracked": len(self._slot_arrival),
            "ring_buffer_size": len(self._ring._buf) if self._ring else 0,
        }

    # --- internal ---

    def _cleanup_slot(self, slot_id: str) -> None:
        self._slot_arrival.pop(slot_id, None)
        self._slot_queue_enter.pop(slot_id, None)
        self._slot_prefill_start.pop(slot_id, None)
        self._slot_first_token_at.pop(slot_id, None)
        self._slot_keep_alives.pop(slot_id, None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_telemetry: FlapTelemetry | None = None
_init_lock = threading.Lock()


def init_telemetry(log_dir: Path | str | None = None,
                   enabled: bool = True) -> FlapTelemetry:
    """Initialize the telemetry singleton (idempotent)."""
    global _telemetry
    with _init_lock:
        if _telemetry is None:
            _telemetry = FlapTelemetry(log_dir=log_dir, enabled=enabled)
        return _telemetry


def get_telemetry() -> FlapTelemetry | None:
    """Return the telemetry singleton (None if not initialized)."""
    return _telemetry
