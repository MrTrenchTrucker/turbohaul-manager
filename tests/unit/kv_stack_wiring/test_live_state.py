"""Module-interface tests for kv_stack_wiring.live_kv_protection and live_turn_groups.

The host is a plain fake that records the order of its reads. Each test asserts what the scan
returned, which hashes were computed in which order, and the exact log record of a failed scan.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.kv_stack_wiring import live_kv_protection, live_turn_groups  # noqa: E402

CARD = "see src/turbohaul/kv_stack_wiring/AGENTS.md"
LOGGER = "turbohaul.manager"


def _slot(tid, meta=None, chain=None):
    return SimpleNamespace(thread_id=tid, client_meta=meta, admission_hash_chain=chain)


def _identity(tid, meta, chain):
    """A session in the meta renames the bin; otherwise the identity is the thread id itself."""
    session = (meta or {}).get("session")
    return f"sess:{session}" if session else tid


class _Host:
    def __init__(self, residents=(), active=None, inflight=(), idle=None, idle_meta=None,
                 handle=None, resident_map=None):
        self._residents_list = list(residents)
        self._active_slot = active
        self._inflight = list(inflight)
        self._idle_thread_id = idle
        self._idle_client_meta = idle_meta
        self._active_handle = handle
        self._residents = resident_map if resident_map is not None else {}
        self.hash_calls = []
        self.events = []

    def _model_residents(self):
        self.events.append("residents")
        return self._residents_list

    def _thread_hash(self, tid):
        self.hash_calls.append(tid)
        return f"h:{tid}"


def _resident(port=None, active=None, inflight=(), idle=None, idle_meta=None):
    return SimpleNamespace(port=port, active_slot=active, inflight=list(inflight),
                           idle_thread_id=idle, idle_client_meta=idle_meta)


class _ExplodesOnIteration:
    def __iter__(self):
        raise RuntimeError("inflight boom")


def _scan_records(caplog, text):
    return [r for r in caplog.records
            if r.name == LOGGER and r.levelno == logging.DEBUG and text in r.getMessage()]


def test_resident_active_and_inflight_are_hard_and_idle_threads_are_soft():
    active, flying = _slot("a1"), _slot("f1")
    host = _Host(residents=[_resident(active=active, inflight=[flying])], idle="idle-held",
                 resident_map={"m": _resident(idle="res-idle")})
    hard, soft = live_kv_protection(host, _identity)
    assert hard == {"h:a1", "h:f1"}, f"resident active and in-flight threads must be HARD; got {sorted(hard)} ({CARD})"
    assert soft == {"h:idle-held", "h:res-idle"}, f"idle threads must be SOFT; got {sorted(soft)} ({CARD})"
    assert type(hard) is set and type(soft) is set


def test_hashes_are_taken_hard_first_then_soft_in_gather_order():
    host = _Host(residents=[_resident(active=_slot("a1"), inflight=[_slot("f1"), _slot("f2")])],
                 idle="idle-held", resident_map={"m": _resident(idle="res-idle")})
    live_kv_protection(host, _identity)
    assert host.hash_calls == ["a1", "f1", "f2", "idle-held", "res-idle"], (
        f"HARD threads are hashed before SOFT, each in gather order; got {host.hash_calls} ({CARD})")


def test_no_residents_reads_the_legacy_singleton_and_ignores_resident_fields():
    host = _Host(residents=[], active=_slot("legacy-a"), inflight=[_slot("legacy-f")])
    hard, soft = live_kv_protection(host, _identity)
    assert hard == {"h:legacy-a", "h:legacy-f"}, f"the legacy singleton must supply HARD; got {sorted(hard)}"
    assert soft == set(), f"nothing idle was set, so SOFT must be empty; got {sorted(soft)}"


def test_with_residents_the_legacy_singleton_is_not_read():
    host = _Host(residents=[_resident(active=_slot("res-a"))], active=_slot("legacy-a"),
                 inflight=[_slot("legacy-f")])
    hard, _soft = live_kv_protection(host, _identity)
    assert hard == {"h:res-a"}, f"a resident scan must not also read the singleton; got {sorted(hard)}"


def test_both_the_thread_hash_and_the_identity_hash_are_protected():
    host = _Host(residents=[_resident(active=_slot("t1", meta={"session": "S1"}))])
    hard, _soft = live_kv_protection(host, _identity)
    assert hard == {"h:t1", "h:sess:S1"}, f"the raw and the identity hash must both be HARD; got {sorted(hard)}"


def test_an_empty_thread_id_is_skipped_and_a_none_active_slot_adds_nothing():
    host = _Host(residents=[_resident(active=None), _resident(active=_slot(""))], idle=None)
    hard, soft = live_kv_protection(host, _identity)
    assert hard == set() and soft == set(), f"an unknown thread id must never be protected; got {hard} {soft}"
    assert host.hash_calls == [], f"no hash may be taken for an empty thread id; got {host.hash_calls}"


def test_a_mid_gather_failure_keeps_what_was_gathered_and_logs_one_debug_record(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    broken = SimpleNamespace(port=2, active_slot=_slot("b-active"), inflight=_ExplodesOnIteration(),
                             idle_thread_id=None, idle_client_meta=None)
    host = _Host(residents=[_resident(active=_slot("a1"), inflight=[_slot("f1")]), broken], idle="never-reached")
    hard, soft = live_kv_protection(host, _identity)
    assert hard == {"h:a1", "h:f1", "h:b-active"}, f"triples gathered before the failure are kept; got {sorted(hard)}"
    assert soft == set(), f"the SOFT gather never ran, so SOFT must be empty; got {sorted(soft)}"
    records = _scan_records(caplog, "live KV protection-class scan failed (best-effort)")
    assert len(records) == 1, f"a failed gather must log exactly one DEBUG record; got {len(records)}"
    assert records[0].exc_info is not None and records[0].exc_info[0] is RuntimeError, "exc_info must carry the error"


def test_a_failing_thread_hash_logs_one_debug_record_and_returns_two_empty_sets(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)

    class _Boom(RuntimeError):
        pass

    host = _Host(residents=[_resident(active=_slot("a1"))])

    def _boom(tid):
        raise _Boom("hash boom")

    host._thread_hash = _boom
    hard, soft = live_kv_protection(host, _identity)
    assert hard == set() and soft == set(), f"a failed split must give empty tiers; got {hard} {soft}"
    records = _scan_records(caplog, "live KV protection-class scan failed (best-effort)")
    assert len(records) == 1, f"a failed split must log exactly one DEBUG record; got {len(records)}"
    assert records[0].exc_info[0] is _Boom, "exc_info must carry the raised class"


def test_live_turn_groups_lists_one_entry_per_resident_with_one_set_per_live_turn():
    r1 = _resident(port=100, active=_slot("a1", meta={"session": "S1"}), inflight=[_slot("f1"), None])
    r2 = _resident(port=200, active=None, inflight=[_slot("")])
    host = _Host(residents=[r1, r2])
    groups = live_turn_groups(host, _identity)
    assert groups == [(100, [frozenset({"h:a1", "h:sess:S1"}), frozenset({"h:f1"})]), (200, [])], (
        f"one (port, sets) entry per resident, empty sets and None slots left out; got {groups} ({CARD})")
    assert all(isinstance(g, frozenset) for g in groups[0][1]), "each live turn is a frozenset"


def test_live_turn_groups_without_residents_reads_the_singleton_on_the_active_handle_port():
    host = _Host(residents=[], active=_slot("legacy-a"), inflight=[_slot("legacy-f")],
                 handle=SimpleNamespace(port=300))
    assert live_turn_groups(host, _identity) == [(300, [frozenset({"h:legacy-a"}), frozenset({"h:legacy-f"})])]


def test_live_turn_groups_without_residents_or_handle_gives_port_none():
    host = _Host(residents=[], active=None, inflight=[], handle=None)
    assert live_turn_groups(host, _identity) == [(None, [])], "no handle means the served port is None"


def test_live_turn_groups_failure_returns_none_and_logs_one_debug_record(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    broken = SimpleNamespace(port=1, active_slot=None, inflight=_ExplodesOnIteration())
    host = _Host(residents=[broken])
    assert live_turn_groups(host, _identity) is None, "a failed scan means the served ports are unknown (None)"
    records = _scan_records(caplog, "live turn scan failed (served ports unknown)")
    assert len(records) == 1, f"a failed scan must log exactly one DEBUG record; got {len(records)}"
    assert records[0].exc_info[0] is RuntimeError, "exc_info must carry the error"


def test_idle_threads_are_hashed_with_their_own_client_meta_and_the_identity_hash_lands_in_soft():
    seen = []

    def _recording_identity(tid, meta, chain):
        seen.append((tid, meta, chain))
        return f"ident:{(meta or {}).get('who')}" if meta else tid

    host_meta = {"who": "host-idle"}
    resident_meta = {"who": "resident-idle"}
    host = _Host(residents=[], idle="idle-held", idle_meta=host_meta,
                 resident_map={"m": _resident(idle="res-idle", idle_meta=resident_meta)})
    hard, soft = live_kv_protection(host, _recording_identity)
    assert ("idle-held", host_meta, None) in seen, f"the host's idle thread is judged with its own meta; saw {seen}"
    assert ("res-idle", resident_meta, None) in seen, f"a resident's idle thread is judged with its meta; saw {seen}"
    assert "h:ident:host-idle" in soft and "h:ident:resident-idle" in soft, (
        f"the identity hash of each idle thread must be SOFT; got {sorted(soft)} ({CARD})")
    assert soft == {"h:idle-held", "h:res-idle", "h:ident:host-idle", "h:ident:resident-idle"}
    assert hard == set(), f"idle threads are never HARD; got {sorted(hard)}"
