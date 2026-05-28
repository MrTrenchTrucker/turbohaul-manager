"""Tests for GET /v1/logging ( — spec LOCKED-FINAL).

13 cases: 11 spec-mandated + 2 adversarial-review checks (poison-row + list traversal).
"""
import json

import pytest
from fastapi.testclient import TestClient

from turbohaul.api.logging import (
    EFFECTIVE_BUDGET,
    ENVELOPE_OVERHEAD_CHARS,
    MAX_RESPONSE_CHARS,
    _redact,
)
from turbohaul.api.main import create_app
from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.state import open_state_db, record_audit_event, state_db_session


@pytest.fixture
def app_test(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


def _seed_events(state_db_path, events):
    """Insert audit_events rows. Each event dict: slot_id, event_type, payload, occurred_at?."""
    conn = open_state_db(state_db_path)
    try:
        for ev in events:
            record_audit_event(
                conn,
                event_type=ev["event_type"],
                payload=ev.get("payload", {}),
                slot_id=ev.get("slot_id"),
            )
    finally:
        conn.close()


def _state_db_path(app):
    return app.state.manager.boot.storage.state_db_path


# ============================================================================
# Case 1 — Happy path: 5 events returned w/ correct simplified schema
# ============================================================================


def test_happy_path_five_events_simplified_schema(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {"slot_id": "slot-1", "event_type": "submit", "payload": {"a": 1}},
        {"slot_id": "slot-1", "event_type": "staged", "payload": {}},
        {"slot_id": "slot-1", "event_type": "loading", "payload": {}},
        {"slot_id": "slot-1", "event_type": "active", "payload": {}},
        {"slot_id": "slot-1", "event_type": "grace_enter", "payload": {}},
    ])
    r = client.get("/v1/logging")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"events", "next_since", "oversized"}
    assert "truncated" not in body
    assert "server_now" not in body
    assert len(body["events"]) == 5
    assert body["oversized"] is False
    # next_since is null because limit not hit
    assert body["next_since"] is None
    for ev in body["events"]:
        assert set(ev.keys()) == {
            "event_id", "slot_id", "event_type", "payload", "occurred_at",
        }


# ============================================================================
# Case 2 — Empty store returns canonical empty envelope
# ============================================================================


def test_empty_store(app_test):
    app, client = app_test
    r = client.get("/v1/logging")
    assert r.status_code == 200
    body = r.json()
    assert body == {"events": [], "next_since": None, "oversized": False}


# ============================================================================
# Case 3 — ?since past tail returns empty cleanly
# ============================================================================


def test_since_past_tail(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {"event_type": "x", "payload": {}} for _ in range(3)
    ])
    r = client.get("/v1/logging?since=9999")
    assert r.status_code == 200
    body = r.json()
    assert body["events"] == []
    assert body["next_since"] is None
    assert body["oversized"] is False


# ============================================================================
# Case 4 — ?slot_id filter exact-match subset
# ============================================================================


def test_slot_id_filter(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {"slot_id": "slot-A", "event_type": "x", "payload": {}},
        {"slot_id": "slot-B", "event_type": "x", "payload": {}},
        {"slot_id": "slot-A", "event_type": "y", "payload": {}},
    ])
    r = client.get("/v1/logging?slot_id=slot-A")
    body = r.json()
    assert len(body["events"]) == 2
    assert all(ev["slot_id"] == "slot-A" for ev in body["events"])


# ============================================================================
# Case 5 — ?event_type filter exact-match subset
# ============================================================================


def test_event_type_filter(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {"slot_id": "s", "event_type": "submit", "payload": {}},
        {"slot_id": "s", "event_type": "staged", "payload": {}},
        {"slot_id": "s", "event_type": "submit", "payload": {}},
    ])
    r = client.get("/v1/logging?event_type=submit")
    body = r.json()
    assert len(body["events"]) == 2
    assert all(ev["event_type"] == "submit" for ev in body["events"])


# ============================================================================
# Case 6 — ?limit clamp: 999 → 500 effective
# ============================================================================


def test_limit_clamp(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {"event_type": "x", "payload": {}} for _ in range(600)
    ])
    r = client.get("/v1/logging?limit=999")
    body = r.json()
    assert len(body["events"]) == 500  # clamped
    assert body["next_since"] is not None  # more available


# ============================================================================
# Case 7 — Pagination: 1000-event fixture, walk next_since, no miss/dup
# ============================================================================


def test_pagination_1000_events_no_miss_no_dup(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {"event_type": "x", "payload": {"i": i}} for i in range(1000)
    ])
    seen_ids = []
    since = 0
    safety = 20  # max pages
    while safety > 0:
        safety -= 1
        r = client.get(f"/v1/logging?since={since}&limit=200")
        body = r.json()
        for ev in body["events"]:
            seen_ids.append(ev["event_id"])
        if body["next_since"] is None:
            break
        since = body["next_since"]
    assert safety > 0, "pagination did not terminate"
    assert len(seen_ids) == 1000
    assert seen_ids == sorted(seen_ids)  # ascending
    assert len(set(seen_ids)) == 1000  # no dups


# ============================================================================
# Case 8 — Concurrent write mid-iteration: no exception, no missed rows
# ============================================================================


def test_concurrent_write_mid_iteration(app_test):
    app, client = app_test
    state_db_path = _state_db_path(app)
    _seed_events(state_db_path, [
        {"event_type": "x", "payload": {"i": i}} for i in range(5)
    ])
    # First call
    r1 = client.get("/v1/logging?limit=200")
    body1 = r1.json()
    last_id_first_call = body1["events"][-1]["event_id"]
    # Inject 10 more events between calls
    _seed_events(state_db_path, [
        {"event_type": "y", "payload": {"i": i}} for i in range(10)
    ])
    # Second call from where we left off
    r2 = client.get(f"/v1/logging?since={last_id_first_call + 1}")
    body2 = r2.json()
    assert len(body2["events"]) == 10
    for ev in body2["events"]:
        assert ev["event_id"] > last_id_first_call


# ============================================================================
# Case 9 — Oversized payload (single event >> budget) returns alone w/ oversized=true
# ============================================================================


def test_oversized_single_event_escape(app_test):
    app, client = app_test
    # Build a payload whose serialized form exceeds EFFECTIVE_BUDGET on its own
    big = "x" * (EFFECTIVE_BUDGET + 1000)
    _seed_events(_state_db_path(app), [
        {"event_type": "huge", "payload": {"blob": big}},
        {"event_type": "small", "payload": {}},
    ])
    r = client.get("/v1/logging")
    body = r.json()
    assert body["oversized"] is True
    assert len(body["events"]) == 1
    assert body["events"][0]["event_type"] == "huge"
    assert body["next_since"] == body["events"][0]["event_id"] + 1


# ============================================================================
# Case 10 — Redaction top-level: `prompt` key stripped
# ============================================================================


def test_redaction_top_level(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {"event_type": "x", "payload": {"prompt": "secret", "safe": "ok"}},
    ])
    r = client.get("/v1/logging")
    body = r.json()
    payload = body["events"][0]["payload"]
    assert "prompt" not in payload
    assert payload.get("safe") == "ok"


# ============================================================================
# Case 11 — Monotone ordering: all event_id ascending
# ============================================================================


def test_monotone_ordering(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {"event_type": "x", "payload": {"i": i}} for i in range(50)
    ])
    r = client.get("/v1/logging?limit=500")
    body = r.json()
    ids = [ev["event_id"] for ev in body["events"]]
    assert ids == sorted(ids)


# ============================================================================
# Case 12 — NEW (F1): Poison row decode failure → sentinel, not 500
# ============================================================================


def test_poison_row_returns_sentinel(app_test):
    app, client = app_test
    state_db_path = _state_db_path(app)
    # Seed one good row, one poison row (raw non-JSON), one more good row.
    _seed_events(state_db_path, [{"event_type": "good", "payload": {"i": 1}}])
    # Direct INSERT bypassing record_audit_event to plant raw garbage payload_json
    with state_db_session(state_db_path) as conn:
        conn.execute(
            """INSERT INTO audit_events (slot_id, event_type, payload_json, occurred_at)
               VALUES (?, ?, ?, ?)""",
            (None, "poisoned", "not-valid-json", "2026-05-18T00:00:00+00:00"),
        )
    _seed_events(state_db_path, [{"event_type": "good", "payload": {"i": 2}}])
    r = client.get("/v1/logging")
    assert r.status_code == 200
    body = r.json()
    assert len(body["events"]) == 3
    poisoned = next(ev for ev in body["events"] if ev["event_type"] == "poisoned")
    assert poisoned["payload"] == {
        "_decode_error": True,
        "_raw_len": len("not-valid-json"),
    }


# ============================================================================
# Case 13 — NEW (F3): Redaction list traversal scrubs nested prompts inside list
# ============================================================================


def test_redaction_list_traversal(app_test):
    app, client = app_test
    _seed_events(_state_db_path(app), [
        {
            "event_type": "x",
            "payload": {
                "items": [{"prompt": "leak1"}, {"safe": "ok"}, {"prompt": "leak2"}],
            },
        },
    ])
    r = client.get("/v1/logging")
    body = r.json()
    items = body["events"][0]["payload"]["items"]
    assert items == [{}, {"safe": "ok"}, {}]


# ============================================================================
# Bonus inline: _redact depth-cap returns value unchanged below the cap
# (not one of the 13 mandated tests; defensive coverage for the depth guard)
# ============================================================================


def test_redact_depth_cap_returns_value_unchanged():
    # Build nesting deeper than the cap; deepest dict still contains `prompt`
    # which should NOT be stripped because we stop recursing at depth_cap.
    deep = {"prompt": "untouched-because-too-deep"}
    nested = deep
    for _ in range(15):
        nested = {"inner": nested}
    out = _redact(nested)
    # Walk down 15 levels of "inner"
    cursor = out
    for _ in range(15):
        cursor = cursor["inner"]
    # At depth >10 the scrubber stops recursing — `prompt` remains
    assert "prompt" in cursor


# MOD #4 : depth-cap boundary acceptance + cycle guard
def test_redact_depth_at_exactly_10_still_redacts():
    """At depth == cap (10) the scrubber MUST still apply redaction.

    Boundary test: only depth > 10 short-circuits. depth == 10 is the last
    level that enters the dict branch and filters REDACTED_KEYS.
    """
    inner = {"prompt": "leak-must-strip", "safe": "keep"}
    nested = inner
    for _ in range(10):
        nested = {"wrap": nested}
    out = _redact(nested)
    cursor = out
    for _ in range(10):
        cursor = cursor["wrap"]
    # At depth=10 the function still ran the redaction comprehension
    assert "prompt" not in cursor
    assert cursor.get("safe") == "keep"


def test_redact_cycle_detection_terminates():
    """Cyclic dict reference must not raise RecursionError.

    The depth-cap IS the cycle guard (no explicit visited-set). This test
    documents that contract — if someone removes the depth-cap to add a
    visited-set, the cycle defense must survive the refactor.
    """
    d: dict = {}
    d["self"] = d
    out = _redact(d)
    assert isinstance(out, dict)
    # Depth-cap halts traversal before stack overflow; exact shape depends
    # on cap value but the call MUST return rather than raise.


# MOD #5 : negative-space no-auth posture documentation
def test_get_logging_returns_200_without_authorization_header(app_test):
    """Design-intent: /v1/logging is unauthenticated per §11.3 perimeter model.

    No Authorization header sent. If future hardening silently adds app-layer
    auth on this endpoint, this test surfaces the regression — and the §11.3
    addendum should be updated in the same change.
    """
    app, client = app_test
    r = client.get("/v1/logging")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"events", "next_since", "oversized"}


def test_budget_constants_sane():
    """Sanity: budget arithmetic matches RC spec."""
    assert MAX_RESPONSE_CHARS == 80_000
    assert ENVELOPE_OVERHEAD_CHARS == 512
    assert EFFECTIVE_BUDGET == 79_488
