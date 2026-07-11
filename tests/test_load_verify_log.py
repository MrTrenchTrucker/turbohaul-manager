"""Contract tests for the load-verify observability module.

The load-bearing guarantee: the emitter + read helpers are DISPLAY-ONLY and must
NEVER raise into the caller (the manager's root fix calls them from the live
spawn/restore/retry path). These tests exercise the never-raise boundary + the
three adversarial edge cases (non-numeric n_past, get_recent(0), ring aliasing).
"""

import asyncio

from turbohaul import load_verify_log as lv


def _run(coro):
    return asyncio.run(coro)


class _Handle:
    """Duck-typed stand-in for the manager's _active_handle."""

    def __init__(self, port=None, pid=None):
        self.port = port
        self.pid = pid


def test_emitter_never_raises_and_returns_record():
    lv.clear_ring()
    rec = lv.log_load_verify(
        event="kv_restore", trigger="wave_return", model_tag="m", port=11500,
        kv_expected_tokens=66509, kv_actual_n_past=66509, kv_restore_ok=True,
    )
    assert rec["model_tag"] == "m"
    assert lv.get_recent(1)[-1]["event"] == "kv_restore"


def test_ring_stores_copy_not_alias():
    # Edge case #2: caller mutating the returned record must not corrupt the ring.
    lv.clear_ring()
    rec = lv.log_load_verify(event="model_load", trigger="spawn", model_tag="m", port=1)
    rec["final_status"] = "MUTATED"
    assert lv.get_recent(1)[-1]["final_status"] != "MUTATED"


def test_get_recent_zero_is_empty():
    # Edge case #3: get_recent(0) must mean "none", not "everything" (items[-0:]).
    lv.clear_ring()
    lv.log_load_verify(event="model_load", trigger="spawn", model_tag="m", port=1)
    assert lv.get_recent(0) == []
    assert len(lv.get_recent(1)) == 1


def test_ring_bounded():
    lv.clear_ring()
    for i in range(200):
        lv.log_load_verify(event="model_load", trigger="spawn", model_tag=str(i), port=1)
    assert len(lv.get_recent()) <= lv._RING_MAX


def test_kv_restore_ok_arithmetic():
    # Normal ok / short restore / boundary.
    h = _Handle()
    assert _run(lv.verify_kv_restored(h, 0, 12000, actual_n_past=12839))["kv_restore_ok"] is True
    assert _run(lv.verify_kv_restored(h, 0, 999999, actual_n_past=12839))["kv_restore_ok"] is False
    # exact threshold boundary (0.98)
    assert _run(lv.verify_kv_restored(h, 0, 1000, actual_n_past=980))["kv_restore_ok"] is True
    assert _run(lv.verify_kv_restored(h, 0, 1000, actual_n_past=979))["kv_restore_ok"] is False


def test_kv_restore_non_numeric_never_raises():
    # Edge case #1: untrusted engine/override n_past of a non-numeric type must NOT
    # raise TypeError into the caller's restore/retry path.
    h = _Handle()
    r = _run(lv.verify_kv_restored(h, 0, 12000, actual_n_past="512"))
    assert r["kv_restore_ok"] is False and "non-numeric" in (r["reason"] or "")
    # non-numeric expected -> degrade, present numeric actual is trivially ok
    assert _run(lv.verify_kv_restored(h, 0, "bad", actual_n_past=100))["kv_restore_ok"] is True
    # bool excluded (int subclass)
    assert _run(lv.verify_kv_restored(h, 0, 10, actual_n_past=True))["kv_restore_ok"] is False


def test_verify_model_resident_no_port_never_raises():
    r = _run(lv.verify_model_resident(_Handle(port=None, pid=None)))
    assert r["model_resident"] is False and r["reason"] == "no port on handle"


def test_verify_dead_engine_never_raises():
    # Unreachable port -> degrades to False + reason (the silently-dead-engine
    # blind spot), never raises.
    r = _run(lv.verify_model_resident(_Handle(port=1, pid=None)))
    assert r["model_resident"] is False and r["reason"]
    k = _run(lv.verify_kv_restored(_Handle(port=1), 0, 100))
    assert k["kv_restore_ok"] is None and k["reason"]
