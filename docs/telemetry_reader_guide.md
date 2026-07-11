# Turbohaul Flap/Degradation Telemetry — Reader's Guide

## What This Is

An OBSERVE-ONLY JSONL log that captures per-request lifecycle signals in
Turbohaul Manager so we can reconstruct a "flap episode" — queue depth
climbing, queued-request silence exceeding client timeout, disconnect->retry
churn, and context-size growth — all timestamped.

## Log Location

    {state_db_parent}/telemetry/flap_{YYYYMMDDTHHMMSS}.jsonl

Default: `/var/lib/turbohaul/telemetry/` (survives restart, mounted volume).

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
      "source": "ring"
    }

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

### request_arrival

A new inference request arrived at the API layer.

    {
      "event_type": "request_arrival",
      "ts": "2026-06-27T05:30:00.123+00:00",
      "slot_id": "slot-abc123",
      "model_tag": "example-model-27b",
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

### first_token (TTFT)

First token received. The KEY latency metric.

    {
      "event_type": "first_token",
      "ts": "2026-06-27T05:30:05.431+00:00",
      "slot_id": "slot-abc123",
      "ttft_ms": 5300.0         ← TTFT = prefill_start to first byte
    }

### keep_alive

A `: keep-alive` comment was emitted during stream-OPEN.

**KEY signal for flap diagnosis**: if QUEUED requests never get this but the
active request does, the queued clients are sitting in silence.

    {
      "event_type": "keep_alive",
      "ts": "2026-06-27T05:30:17.131+00:00",
      "slot_id": "slot-abc123",
      "is_queued": false,        ← false = active request; true = queued
      "count_for_slot": 3
    }

### client_disconnect

Client disconnected (mid-stream or mid-queue).

    {
      "event_type": "client_disconnect",
      "ts": "2026-06-27T05:34:00.500+00:00",
      "slot_id": "slot-abc123",
      "reason": "client_disconnect",
      "elapsed_s": 240.3,       ← seconds between arrival and disconnect
      "was_in_queue": false,
      "keep_alives_sent": 5
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

Periodic VRAM + process memory sample (from supervisor, ~1Hz).

    {
      "event_type": "resource_sample",
      "ts": "2026-06-27T05:30:05.000+00:00",
      "vram_free_mib": [14230, 14100],  ← per-GPU free MiB
      "process_rss_mib": 512.3,
      "vram_sample_errors": 0
    }

### slot_state_change

Slot FSM state transition.

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
4. **client_disconnect** with `elapsed_s` ≈ 240 (client timeout) and
   `was_in_queue: true` (queued request got ZERO bytes)
5. **keep_alive** with `is_queued: false` only (active request gets heartbeats,
   queued requests get silence — confirms the root cause)
6. **resource_sample** with `vram_free_mib` declining over time (memory leak hint)

## Query Examples

```bash
# All events for a specific slot
curl 'http://localhost:11401/v1/telemetry/events?event_type=request_arrival&limit=100&source=file' | jq '.events[] | select(.slot_id=="slot-abc123")'

# All client disconnects (the flap signal)
curl 'http://localhost:11401/v1/telemetry/events?event_type=client_disconnect&limit=50&source=file' | jq '.events[] | {ts, elapsed_s, was_in_queue, keep_alives_sent}'

# Queue depth over time (shows stacking)
curl 'http://localhost:11401/v1/telemetry/events?event_type=queue_enter&limit=100&source=file' | jq '.events[] | {ts, staging_depth}'

# VRAM trend (memory leak hint)
curl 'http://localhost:11401/v1/telemetry/events?event_type=resource_sample&limit=100&source=file' | jq '.events[] | {ts, vram_free_mib, process_rss_mib}'

# Subsystem health
curl 'http://localhost:11401/v1/telemetry/status'
```
