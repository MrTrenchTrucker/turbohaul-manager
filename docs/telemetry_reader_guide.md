# Turbohaul Flap/Degradation Telemetry — Reader's Guide

## What This Is

An OBSERVE-ONLY JSONL log that captures per-request lifecycle signals in
Turbohaul Manager so you can reconstruct a "flap episode" — queue depth
climbing, queued-request wait exceeding the client timeout, disconnect->retry
churn, and context-size growth — all timestamped.

## Log Location

    {state_db_parent}/telemetry/flap_{YYYYMMDDTHHMMSS}.jsonl

Default: `/var/lib/turbohaul/telemetry/`, next to `state.sqlite` (survives restart when `/var/lib/turbohaul` is on a mounted volume).

Files rotate at 10 MiB, retain 5 files. Each line is one JSON object.

## Read Endpoints

### GET /v1/telemetry/events

Query recent events.

| Param       | Default | Description                                    |
|-------------|---------|------------------------------------------------|
| `source`    | `ring`  | `ring` = in-memory (fast, ~10k events). `file` = JSONL (full history) |
| `since`     | `0`     | Ring-buffer sequence cursor (skip events <= this) |
| `event_type`| null    | Filter by event_type                          |
| `limit`     | `200`   | Max events to return (1-1000)                  |

Response shape:

    {
      "events": [ ... ],
      "next_since": 42,    // pass as `since` for next page, null = end
      "source": "ring_buffer"
    }

`source` in the response is `ring_buffer` for `source=ring` and `jsonl` for
`source=file`. With `source=ring`, events come back oldest-first and
`next_since` is null unless the page is full (`limit` events). With
`source=file`, events come back newest-first and `next_since` is the `_seq` of
the last (oldest) event returned, or null when nothing matched. `_seq` restarts
from 1 whenever the manager restarts.

### GET /v1/telemetry/status

Subsystem health:

    {
      "enabled": true,
      "log_dir": "/var/lib/turbohaul/telemetry",
      "last_vram_sample_at": "2026-06-27T05:30:00.000+00:00",
      "vram_sample_errors": 0,
      "active_slots_tracked": 3,
      "ring_buffer_size": 150
    }

## Event Types

Most events also carry `thread_id` and, where known, `model_tag`; the
examples below show the fields that matter for each event type.

### request_arrival

A new inference request arrived at the API layer.

    {
      "event_type": "request_arrival",
      "ts": "2026-06-27T05:30:00.123+00:00",
      "slot_id": "slot-abc123",
      "model_tag": "my-model-27b",
      "thread_id": "thread-xyz",
      "has_context": true
    }

### queue_enter

A slot entered the queue (staging or acceptance buffer).

    {
      "event_type": "queue_enter",
      "ts": "2026-06-27T05:30:00.124+00:00",
      "slot_id": "slot-abc123",
      "staging_depth": 3,        ← KEY: how many requests are ahead
      "staging_max": 100,
      "acceptance_depth": 0,
      "slot_state": "STAGED"
    }

### slot_assign

A slot was assigned to an active sidecar (left the queue).

    {
      "event_type": "slot_assign",
      "ts": "2026-06-27T05:30:00.130+00:00",
      "slot_id": "slot-abc123",
      "wait_in_queue_s": 0.006,  ← how long it waited
      "pid": 12345,
      "port": 11500
    }

### prefill_start

Prefill began for the slot.

    {
      "event_type": "prefill_start",
      "ts": "2026-06-27T05:30:00.131+00:00",
      "slot_id": "slot-abc123"
    }

### turn_dispatch

A turn's prompt was actually handed to the engine. Unlike `slot_assign` /
`prefill_start` — which fire once, on first assignment, and are silent for
every later turn a well-behaved client sends to the same warm slot — this
fires on every turn served through the four call sites below, including the
warm-reuse case (a turn served through the multi-slot fan-out path, i.e. an
engine running with `llama_server_flags.parallel` > 1, does not emit it). It
splits TTFT:
`request_arrival -> turn_dispatch` is WAIT (queueing/stranding — the defect
surface); `turn_dispatch -> first_token` is PREFILL (compute, scales with
context, not a defect).

    {
      "event_type": "turn_dispatch",
      "ts": "2026-06-27T05:30:00.131+00:00",
      "slot_id": "slot-abc123",
      "prompt_tokens": 84872      ← from slot.admission_ctx_len, known at admission
    }

No `reused_tokens` field: the engine's real reuse figure (`engine_reused_tokens`
on `KV_REUSE_OUTCOME`) is stamped only at completion, off a live-poller sample
that does not exist yet at dispatch time — shipping it here would be null or
wrong on every turn. (A DECISION-time number does exist earlier, on the
`KV_REUSE` line — `restored_tokens`, what the manager OFFERED — but that is not
what the engine KEPT; the two provably diverge, e.g. a poison-bin retirement
can offer a full restore and still be kept at zero. Do not conflate the two
when reading these logs side by side.)

**Four call sites, one event type** — the same anchor on the cold
(first-assignment) path and the warm (grace-window `ACTIVE_MATCH` follow-up)
path, each with a streaming and a non-streaming arm:

| Path (`_serve_on_resident`) | Streaming arm fires at | Non-streaming arm fires at |
|---|---|---|
| Cold (anchor slot) | `slot.stream_ready_event.set()` | immediately before `_complete_fn(slot, handle)`, i.e. AFTER `_probe_and_save_clean_kv` has run |
| Warm (`matched`, `ACTIVE_MATCH`) | `matched.stream_ready_event.set()` | immediately before `_complete_fn(matched, handle)`, i.e. AFTER `_probe_and_save_clean_kv` has run |

**Placed differently from `prefill_start` on purpose — the two are not
interchangeable, do not compare them turn-for-turn.** `prefill_start` fires
immediately after the slot's ACTIVE transition, BEFORE `_probe_and_save_clean_kv`
runs. `turn_dispatch` fires AFTER that probe. The probe is a real async KV
operation, a no-op unless single-series + large ctx + no equal/larger clean bin
is already saved — but on the turns where it is NOT a no-op, a reader who
assumes `prefill_start`'s timing also describes `turn_dispatch` would
misattribute that probe/save latency into PREFILL. `turn_dispatch -> first_token`
is the trustworthy PREFILL window on every turn; `prefill_start` keeps its
existing meaning, timing and call sites unchanged (this is additive, not a
retiming of an existing event).

### first_token (TTFT)

First token received (streaming requests only). The KEY latency metric.

    {
      "event_type": "first_token",
      "ts": "2026-06-27T05:30:05.431+00:00",
      "slot_id": "slot-abc123",
      "ttft_ms": 5300.0         ← TTFT = start of the streaming response to first byte
    }

`ttft_ms` is measured from the moment the streaming response generator starts
(`stream_gen` in `api/chat_completion.py`), so it includes any time the request
spent waiting for its slot, not just prefill; use `turn_dispatch -> first_token`
for the prefill window.

### keep_alive

A `: keep-alive` comment was emitted while the manager was waiting for the
engine to open the response stream (the prefill phase of a streaming request).

Note: only this phase is recorded, so `is_queued` is always `false` today.
Streaming clients that are still waiting for their slot also receive a
`: keep-alive` comment every 12 s (`HEARTBEAT_INTERVAL_S`), but those are not
recorded as telemetry events.

    {
      "event_type": "keep_alive",
      "ts": "2026-06-27T05:30:17.131+00:00",
      "slot_id": "slot-abc123",
      "is_queued": false,        ← always false in the current code
      "count_for_slot": 3
    }

### client_disconnect

A queued request's client disconnected, and the dispatcher evicted the slot.
This is the only place the event is emitted (`_handle_unloaded_slot` in
`manager.py`); a disconnect after the slot has gone ACTIVE is not recorded here.

    {
      "event_type": "client_disconnect",
      "ts": "2026-06-27T05:34:00.500+00:00",
      "slot_id": "slot-abc123",
      "reason": "client_disconnect",
      "elapsed_s": 240.3,       ← seconds from slot creation to eviction
      "was_in_queue": true,
      "keep_alives_sent": 0
    }

### completion

Slot completed successfully.

    {
      "event_type": "completion",
      "ts": "2026-06-27T05:35:00.000+00:00",
      "slot_id": "slot-abc123",
      "reason": "grace_enter",
      "total_lifecycle_s": 60.5,
      "keep_alives_sent": 5
    }

### resource_sample

Periodic VRAM + process memory sample (`FlapTelemetry.on_vram_sample`, meant to
be fed by the live monitor's ~1 Hz poll; `process_rss_mib` is the process's peak
resident size, `ru_maxrss`).

Note: in the current code nothing calls `on_vram_sample` (its only caller sits on
the legacy single-sidecar poller, which is no longer started), so no
`resource_sample` events are written and `last_vram_sample_at` in
`/v1/telemetry/status` stays null.

    {
      "event_type": "resource_sample",
      "ts": "2026-06-27T05:30:05.000+00:00",
      "vram_free_mib": [14230, 14100],  ← per-GPU free MiB
      "process_rss_mib": 512.3,
      "vram_sample_errors": 0
    }

### slot_state_change

Slot FSM state transition (`FlapTelemetry.on_slot_state_change`).

Note: this event type is defined but not emitted by the manager in the current
code, so it does not appear in the log.

    {
      "event_type": "slot_state_change",
      "ts": "2026-06-27T05:30:00.200+00:00",
      "slot_id": "slot-abc123",
      "old_state": "STAGED",
      "new_state": "LOADING"
    }

## Reconstructing a Flap Episode

With the logging active, a single flap episode is visible as:

1. **queue_enter** with `staging_depth` climbing over successive requests
2. **slot_assign** with `wait_in_queue_s` growing (requests stacking)
3. **first_token** with `ttft_ms` growing as prefill takes longer (context grows)
4. **client_disconnect** with `elapsed_s` ≈ the client's timeout and
   `was_in_queue: true` (the request was still queued when the client gave up)
5. **keep_alive** events (all `is_queued: false`) only for requests whose slot
   was ready; keep-alives sent to queued streaming clients are not recorded, so
   a missing event does not mean the client received no bytes
6. **resource_sample** with `vram_free_mib` declining over time (memory leak
   hint), where `resource_sample` events are available (see its note above)

## Query Examples

```bash
# Events for a specific slot (among the newest 1000)
curl 'http://localhost:11401/v1/telemetry/events?limit=1000&source=file' | jq '.events[] | select(.slot_id=="slot-abc123")'

# All client disconnects (the flap signal)
curl 'http://localhost:11401/v1/telemetry/events?event_type=client_disconnect&limit=50&source=file' | jq '.events[] | {ts, elapsed_s, was_in_queue, keep_alives_sent}'

# Queue depth over time (shows stacking)
curl 'http://localhost:11401/v1/telemetry/events?event_type=queue_enter&limit=100&source=file' | jq '.events[] | {ts, staging_depth}'

# VRAM trend (memory leak hint)
curl 'http://localhost:11401/v1/telemetry/events?event_type=resource_sample&limit=100&source=file' | jq '.events[] | {ts, vram_free_mib, process_rss_mib}'

# Subsystem health
curl 'http://localhost:11401/v1/telemetry/status'
```
