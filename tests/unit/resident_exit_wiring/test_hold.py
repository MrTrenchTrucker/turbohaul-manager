"""hold_warm_window: the window returned for each branch, and the hold-warm log line.

resident_exit_wiring, see src/turbohaul/resident_exit_wiring/AGENTS.md
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

_TREE = Path(__file__).resolve().parents[3]
if str(_TREE / "src") not in sys.path:
    sys.path.insert(0, str(_TREE / "src"))

from turbohaul.resident_exit_wiring import hold_warm_window  # noqa: E402

CARD = "resident_exit_wiring, see src/turbohaul/resident_exit_wiring/AGENTS.md"
LOGGER = "turbohaul.manager"
ACTIVE = "ACTIVE"
IDLE = "IDLE_EVICTABLE"
HOLD_S = 12.0


def _res(tag="m1", key="m1", state=ACTIVE, gpu=0):
    return SimpleNamespace(model_tag=tag, resident_key=key, state=state, main_gpu=gpu)


class _Host:
    """A plain stand-in for the manager: only what the measurement reads."""

    def __init__(self, residents, *, cap=2, designated=False, claims="unset", pending="unset"):
        self._residents_list = residents
        self._designated = designated
        self.runtime = SimpleNamespace(queue=SimpleNamespace(max_parallel_sidecars=cap))
        if claims != "unset":
            self._fastlane_claims = claims
        if pending != "unset":
            self._pending_reclaim_mib = pending

    def _model_residents(self):
        return list(self._residents_list)

    def _is_designated_unload_target_locked(self, r):
        return self._designated


def _call(host, r, idle_window=-1.0, hold_s=HOLD_S, active_state=ACTIVE):
    return hold_warm_window(host, r, idle_window, hold_s, active_state=active_state)


def _hold_pair(**host_kw):
    r = _res()
    sibling = _res(key="m1#1")
    return _Host([r, sibling], **host_kw), r


def test_a_same_model_active_sibling_holds_for_the_offered_window():
    host, r = _hold_pair()
    got = _call(host, r)
    assert got == HOLD_S, f"{CARD}: expected the offered window {HOLD_S!r}, got {got!r}"


def test_the_hold_writes_one_exact_info_line_on_the_manager_logger(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    host, r = _hold_pair()
    _call(host, r)
    assert [(rec.name, rec.levelno, rec.getMessage()) for rec in caplog.records] == [(
        LOGGER, logging.INFO,
        "resident hold-warm: model_tag=m1 resident_key=m1 window=12s active_siblings=1 "
        "reason=co_resident_active_sibling (keep_alive<=0 on a co-resident; bounded, "
        "make-room can still evict it)",
    )], f"{CARD}: got {[(rec.levelname, rec.getMessage()) for rec in caplog.records]!r}"


def test_the_sibling_count_in_the_line_is_the_live_active_count(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    r = _res()
    host = _Host([r, _res(key="a"), _res(key="b"), _res(key="c", state=IDLE), _res(tag="other", key="d")])
    assert _call(host, r) == HOLD_S
    assert "active_siblings=2 " in caplog.records[0].getMessage()


def test_no_sibling_means_no_hold_and_no_line(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    r = _res()
    assert _call(_Host([r]), r) is None
    assert caplog.records == []


def test_a_sibling_of_another_model_does_not_hold():
    r = _res()
    assert _call(_Host([r, _res(tag="other", key="o")]), r) is None


def test_a_sibling_that_is_not_active_does_not_hold():
    r = _res()
    assert _call(_Host([r, _res(key="m1#1", state=IDLE)]), r) is None


def test_the_resident_itself_is_not_counted_as_its_own_sibling():
    r = _res()
    assert _call(_Host([r, r]), r) is None


def test_active_is_whatever_state_the_caller_passes():
    host, r = _hold_pair()
    assert _call(host, r, active_state="SOME_OTHER_STATE") is None
    sibling = _res(key="m1#1", state="SOME_OTHER_STATE")
    assert _call(_Host([r, sibling]), r, active_state="SOME_OTHER_STATE") == HOLD_S


def test_a_cap_of_one_never_holds():
    host, r = _hold_pair(cap=1)
    assert _call(host, r) is None


def test_a_positive_idle_window_is_not_held_over():
    host, r = _hold_pair()
    assert _call(host, r, idle_window=300.0) is None


def test_a_designated_victim_is_not_held(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    host, r = _hold_pair(designated=True)
    assert _call(host, r) is None
    assert caplog.records == []


def test_a_claim_from_another_model_stops_the_hold():
    host, r = _hold_pair(claims={"c1": {"slot": SimpleNamespace(model_tag="other")}})
    assert _call(host, r) is None


def test_a_claim_for_the_same_model_or_with_no_slot_does_not_stop_the_hold():
    same = {"c1": {"slot": SimpleNamespace(model_tag="m1")}}
    no_slot = {"c1": {}}
    no_tag = {"c1": {"slot": SimpleNamespace(model_tag=None)}}
    for claims in (same, no_slot, no_tag):
        host, r = _hold_pair(claims=claims)
        assert _call(host, r) == HOLD_S, f"{CARD}: claims {claims!r} stopped the hold"


def test_a_pending_reclaim_on_this_resident_card_stops_the_hold():
    host, r = _hold_pair(pending={0: 100})
    assert _call(host, r) is None


def test_a_pending_reclaim_on_another_card_or_of_zero_does_not_stop_the_hold():
    for pending in ({1: 100}, {0: 0}):
        host, r = _hold_pair(pending=pending)
        assert _call(host, r) == HOLD_S, f"{CARD}: pending {pending!r} stopped the hold"


def test_a_host_without_the_claims_and_pending_attributes_is_read_as_having_none():
    host, r = _hold_pair()
    assert not hasattr(host, "_fastlane_claims") and not hasattr(host, "_pending_reclaim_mib")
    assert _call(host, r) == HOLD_S


def test_none_claims_and_none_pending_are_read_as_empty():
    host, r = _hold_pair(claims=None, pending=None)
    assert _call(host, r) == HOLD_S


def test_the_offered_window_is_what_comes_back():
    host, r = _hold_pair()
    assert _call(host, r, hold_s=7.0) == 7.0
