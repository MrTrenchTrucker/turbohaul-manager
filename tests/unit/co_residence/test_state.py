"""co_residence.state: the per-manager re-check state (last attempt time and two memos)."""

from turbohaul.co_residence import NO_ROOM_MEMO_S, NoRoomMemo, RecheckState


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


def test_a_new_state_has_never_rechecked():
    assert RecheckState().last_recheck_at is None, why("no attempt yet means None, not a number")


def test_a_new_state_has_two_empty_memos():
    st = RecheckState()
    assert isinstance(st.norom, NoRoomMemo) and isinstance(st.declog, NoRoomMemo), why("both are NoRoomMemo")
    assert len(st.norom) == 0 and len(st.declog) == 0, why("both start empty")


def test_the_two_memos_are_separate_objects():
    st = RecheckState()
    assert st.norom is not st.declog, why("a no-room verdict must not throttle a decline line, or the reverse")
    st.norom.note("tag", 0.0)
    assert len(st.declog) == 0, why("noting in norom leaked into declog")
    st.declog.note("tag|cap", 0.0)
    assert st.norom.keys() == ["tag"], why("noting in declog leaked into norom")


def test_two_states_share_nothing():
    a, b = RecheckState(), RecheckState()
    a.last_recheck_at = 5.0
    a.norom.note("tag", 0.0)
    a.declog.note("tag|cap", 0.0)
    assert b.last_recheck_at is None, why("the attempt time is per state")
    assert len(b.norom) == 0 and len(b.declog) == 0, why("memos must not be shared between two states")
    assert a.norom is not b.norom and a.declog is not b.declog


def test_both_memos_use_the_documented_expiry():
    st = RecheckState()
    for memo in (st.norom, st.declog):
        memo.note("k", 100.0)
        assert memo.blocked("k", 100.0 + NO_ROOM_MEMO_S - 0.1), why("still blocked just inside the window")
        assert not memo.blocked("k", 100.0 + NO_ROOM_MEMO_S), why("expired at the window end")


def test_the_attempt_time_can_be_set_and_read_back():
    st = RecheckState()
    st.last_recheck_at = 0.0
    assert st.last_recheck_at == 0.0 and st.last_recheck_at is not None
