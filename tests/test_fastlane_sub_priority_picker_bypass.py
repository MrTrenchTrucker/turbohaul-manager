"""Fast Lane sub-priority vs. the two grace-window removal paths.

One client (one rule, one IP) has two requests waiting in staging:
a better-ranked one (class "main", rank 1) and a worse-ranked one
(class "curator", rank 3).  ``pop_matched_thread`` and ``pop_matched_ip``
remove a queued slot without running ``_pick_fastlane_locked``.  The
behaviour under test: even through these paths, the better-ranked request
of the same client is served before the worse-ranked one.  A matcher whose
candidate is the worse-ranked request declines it (returns None and leaves
it waiting) while the better-ranked request of the same client waits; the
next ``pop_next`` then serves the better-ranked request first and the
worse-ranked one after it.  Between requests of equal rank, and for the
temporal guard of the IP matcher, the matcher behaves as before.

Each group carries a control proving the instrument can see the difference
it is about, so a pass is not vacuous.
"""

import logging
import time

import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.queue import FastLanePopPolicy, TurbohaulQueue
from turbohaul.slot import Slot


# The tests in this file never read free VRAM. The manager's binding is pinned
# anyway, so a future change that reaches it fails loudly instead of reading
# the live GPU.
@pytest.fixture(autouse=True)
def _pin_free_vram(monkeypatch):
    import turbohaul.manager as manager_module
    import turbohaul.safety as safety_module

    def _must_not_read(*args, **kwargs):
        raise AssertionError("a test of this file read free VRAM")

    monkeypatch.setattr(manager_module, "_read_free_vram_all_mib", _must_not_read)
    monkeypatch.setattr(safety_module, "_read_free_vram_all_mib", _must_not_read)


IP = "192.0.2.10"
MODEL = "m1"
THREAD_BETTER = "thread-better"
THREAD_WORSE = "thread-worse"


def _match(tag, rank, rule_index=0):
    return FastLaneMatch(
        rule_index=rule_index, raw_address=IP, label="", effective_tag=tag, rank=rank,
    )


def _slot(thread_id, tag, rank, created_at=None, model=MODEL):
    s = Slot.new(model_tag=model, prompt=tag, thread_id=thread_id, client_meta={"ip": IP})
    s.fastlane = _match(tag, rank)
    if created_at is not None:
        s.created_at = created_at
    return s


def _peer(thread_id, created_at=None):
    """A request of the same client with the same rank as ``_worse``."""
    return _slot(thread_id, "curator", 3, created_at)


def _better(created_at=None):
    return _slot(THREAD_BETTER, "main", 1, created_at)


def _worse(created_at=None):
    return _slot(THREAD_WORSE, "curator", 3, created_at)


def _policy():
    return FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)


async def _queue_with(*slots):
    q = TurbohaulQueue(staging_max=10, acceptance_max=10)
    for s in slots:
        await q.enqueue(s)
    return q


def _desc(s):
    return f"{s.prompt}(rank={s.fastlane.rank},thread={s.thread_id})" if s else "None"


def _admitted_lines(caplog):
    return [r.getMessage() for r in caplog.records if "QUEUE_ADMITTED_NO_PICK" in r.getMessage()]


# ---------------------------------------------------------------------------
# 1. pop_matched_thread
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPopMatchedThread:
    async def test_thread_match_on_worse_request_serves_better_ranked_first(self):
        """The thread match on the curator (rank 3) request declines it while the main (rank 1) request waits; the main request is served first, then the curator request."""
        worse, better = _worse(), _better()
        q = await _queue_with(worse, better)
        got = await q.pop_matched_thread(THREAD_WORSE, MODEL)
        assert got is None, f"matcher returned {_desc(got)}; the worse-ranked request must be declined"
        assert list(q._staging) == [worse, better], (
            f"waiting after the declined match: {[_desc(s) for s in q._staging]}"
        )
        first = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        second = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        assert first is better and second is worse, (
            f"service order was {[_desc(first), _desc(second)]}; "
            "the better-ranked request should have been served first"
        )
        assert len(q._staging) == 0

    async def test_thread_match_worse_arrived_after_better(self):
        """When the main (rank 1) request also arrived first, the thread match on the curator (rank 3) request is still declined; the main request is served first, then the curator request."""
        better, worse = _better(), _worse()
        q = await _queue_with(better, worse)
        got = await q.pop_matched_thread(THREAD_WORSE, MODEL)
        assert got is None, f"matcher returned {_desc(got)}; the worse-ranked request must be declined"
        assert list(q._staging) == [better, worse], (
            f"waiting after the declined match: {[_desc(s) for s in q._staging]}"
        )
        first = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        second = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        assert first is better and second is worse, (
            f"service order was {[_desc(first), _desc(second)]}; "
            "the better-ranked request should have been served first"
        )
        assert len(q._staging) == 0

    async def test_control_thread_match_on_better_request_returns_it(self):
        """Control: the thread id of the better-ranked request returns that request, in either arrival order."""
        for order in ("worse_first", "better_first"):
            worse, better = _worse(), _better()
            q = await _queue_with(*((worse, better) if order == "worse_first" else (better, worse)))
            got = await q.pop_matched_thread(THREAD_BETTER, MODEL)
            assert got is better, f"{order}: returned {_desc(got)}"
            assert q._staging and q._staging[0] is worse, f"{order}: worse request not left waiting"

    async def test_control_single_request_is_returned(self):
        """Control: with one waiting request the thread match returns it, so the instrument sees the path."""
        only = _worse()
        q = await _queue_with(only)
        got = await q.pop_matched_thread(THREAD_WORSE, MODEL)
        assert got is only
        assert len(q._staging) == 0

    async def test_control_other_thread_id_changes_which_request_is_returned(self):
        """Control: swapping the thread id swaps the outcome (declined versus returned), proving the observable can tell the two apart."""
        worse, better = _worse(), _better()
        q = await _queue_with(worse, better)
        first = await q.pop_matched_thread(THREAD_WORSE, MODEL)
        better2 = _better()
        q2 = await _queue_with(_worse(), better2)
        second = await q2.pop_matched_thread(THREAD_BETTER, MODEL)
        assert first is None, f"worse-ranked thread id returned {_desc(first)}"
        assert second is better2 and second.thread_id == THREAD_BETTER, f"better-ranked thread id returned {_desc(second)}"
        assert list(q._staging) == [worse, better], "declined match must leave both requests waiting"


# ---------------------------------------------------------------------------
# 2. pop_matched_ip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPopMatchedIp:
    async def test_ip_match_worse_arrived_first_serves_better_ranked_first(self):
        """The main (rank 1) request is returned although the curator (rank 3) request arrived first; the curator request stays waiting."""
        t0 = time.monotonic()
        worse, better = _worse(t0 + 1.0), _better(t0 + 2.0)
        q = await _queue_with(worse, better)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is better, (
            f"returned {_desc(got)} while {[_desc(s) for s in q._staging]} stayed waiting; "
            "the better-ranked request should have been served first"
        )
        assert list(q._staging) == [worse]

    async def test_ip_match_better_arrived_first_serves_better_ranked_first(self):
        """The main (rank 1) request is returned when it arrived first."""
        t0 = time.monotonic()
        better, worse = _better(t0 + 1.0), _worse(t0 + 2.0)
        q = await _queue_with(better, worse)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is better, f"returned {_desc(got)}"

    async def test_control_arrival_order_alone_decides_the_ip_pick(self):
        """Control: between two requests of EQUAL rank, swapping only the arrival order swaps the returned request, so the instrument can see an ordering difference."""
        t0 = time.monotonic()
        q1 = await _queue_with(_peer("thread-a", t0 + 1.0), _peer("thread-b", t0 + 2.0))
        q2 = await _queue_with(_peer("thread-b", t0 + 1.0), _peer("thread-a", t0 + 2.0))
        a = await q1.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        b = await q2.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert a is not None and b is not None, f"equal-rank match returned {_desc(a)} / {_desc(b)}"
        assert (a.thread_id, b.thread_id) == ("thread-a", "thread-b"), (
            f"orders returned {a.thread_id} and {b.thread_id}; the pick does not follow arrival order"
        )

    async def test_ip_match_declines_worse_request_in_either_arrival_order(self):
        """With a better-ranked request of the same client waiting, the IP match returns the better-ranked request in either arrival order, never the worse one first."""
        t0 = time.monotonic()
        for order in ("worse_first", "better_first"):
            worse, better = _worse(t0 + 1.0), _better(t0 + 2.0)
            q = await _queue_with(*((worse, better) if order == "worse_first" else (better, worse)))
            got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
            assert got is better, f"{order}: returned {_desc(got)}"
            assert list(q._staging) == [worse], f"{order}: waiting {[_desc(s) for s in q._staging]}"

    async def test_control_temporal_guard_excludes_request_older_than_grace_start(self):
        """Control: of two requests of EQUAL rank, the one created before grace_started_at must not match, so the later one is returned."""
        t0 = time.monotonic()
        old_peer, new_peer = _peer("thread-old", t0 - 1.0), _peer("thread-new", t0 + 1.0)
        q = await _queue_with(old_peer, new_peer)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is new_peer, f"returned {_desc(got)}"
        assert [x.thread_id for x in q._staging] == ["thread-old"]

    async def test_older_better_ranked_waiter_still_blocks_newer_worse_ip_match(self):
        """A better-ranked request that predates grace_started_at cannot itself match, yet it still blocks the newer worse-ranked match: the matcher returns None, then pop_next serves the older better-ranked request first and the newer worse-ranked one after it."""
        t0 = time.monotonic()
        old_better, new_worse = _better(t0 - 1.0), _worse(t0 + 1.0)
        q = await _queue_with(old_better, new_worse)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is None, f"returned {_desc(got)}; the worse-ranked request must be declined"
        assert list(q._staging) == [old_better, new_worse]
        first = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        second = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        assert first is old_better and second is new_worse, f"order was {[_desc(first), _desc(second)]}"

    async def test_control_temporal_guard_boundary_is_exclusive(self):
        """Control: a request created exactly at grace_started_at does not match (the guard is exclusive at the boundary)."""
        t0 = time.monotonic()
        edge = _better(t0)
        q = await _queue_with(edge)
        assert await q.pop_matched_ip(IP, MODEL, grace_started_at=t0) is None
        assert [x.thread_id for x in q._staging] == [THREAD_BETTER]

    async def test_control_single_request_is_returned(self):
        """Control: with one eligible request the IP match returns it, so the instrument sees the path."""
        t0 = time.monotonic()
        only = _worse(t0 + 1.0)
        q = await _queue_with(only)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is only

    async def test_control_other_ip_is_not_returned(self):
        """Control: a request from a different address never matches."""
        t0 = time.monotonic()
        q = await _queue_with(_better(t0 + 1.0))
        assert await q.pop_matched_ip("198.51.100.7", MODEL, grace_started_at=t0) is None


# ---------------------------------------------------------------------------
# 3. what is served next after a bypass pop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestServedNextAfterBypass:
    async def test_after_bypass_pop_the_worse_request_has_not_jumped_the_better(self):
        """The thread match on the curator (rank 3) request returns None; the next two pop_next calls serve the main (rank 1) request first and the curator request second, so nothing is lost and the worse request does not jump the better one."""
        worse, better = _worse(), _better()
        q = await _queue_with(worse, better)
        bypass = await q.pop_matched_thread(THREAD_WORSE, MODEL)
        nxt = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        last = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        order = [_desc(bypass), _desc(nxt), _desc(last)]
        assert bypass is None and nxt is better and last is worse, f"service order was {order}"
        assert len(q._staging) == 0

    async def test_after_ip_bypass_pop_the_worse_request_has_not_jumped_the_better(self):
        """The main (rank 1) request is served first overall when the IP match runs first, and the curator (rank 3) request is served after it."""
        t0 = time.monotonic()
        worse, better = _worse(t0 + 1.0), _better(t0 + 2.0)
        q = await _queue_with(worse, better)
        bypass = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        nxt = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        order = [_desc(bypass), _desc(nxt)]
        assert bypass is better and nxt is worse, f"service order was {order}"

    async def test_control_pop_next_alone_serves_better_ranked_first(self):
        """Control: with no bypass, pop_next with the same two waiting serves the main (rank 1) request first (the picker reads the request's rank)."""
        worse, better = _worse(), _better()
        q = await _queue_with(worse, better)
        first = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        second = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        assert first is better and second is worse, f"order was {[_desc(first), _desc(second)]}"

    async def test_control_pop_next_serves_other_when_rank_is_swapped(self):
        """Control: swapping the two ranks swaps the request pop_next serves first, so the observable can see a rank difference."""
        a = _slot("thread-a", "main", 3)
        b = _slot("thread-b", "curator", 1)
        q = await _queue_with(a, b)
        first = await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())
        assert first is b, f"served {_desc(first)}"


# ---------------------------------------------------------------------------
# 4. the bypass pop logs "admitted without pick"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBypassLogging:
    async def test_thread_bypass_logs_admitted_without_pick_with_rank_still_unread(self, caplog):
        """When the thread match happens (two requests of equal rank), exactly one admitted-without-pick line naming via=pop_matched_thread is logged and no priority-pick line."""
        mine, other = _peer(THREAD_WORSE), _peer("thread-peer")
        q = await _queue_with(other, mine)
        with caplog.at_level(logging.INFO):
            got = await q.pop_matched_thread(THREAD_WORSE, MODEL)
        assert got is mine, f"returned {_desc(got)}"
        lines = _admitted_lines(caplog)
        assert len(lines) == 1, lines
        assert "via=pop_matched_thread" in lines[0]
        assert "fastlane_matched=True" in lines[0]
        assert not [r for r in caplog.records if "QUEUE_PRIORITY" in r.getMessage()]

    async def test_ip_bypass_logs_admitted_without_pick_with_rank_still_unread(self, caplog):
        """When the IP match happens, exactly one admitted-without-pick line naming via=pop_matched_ip is logged and no priority-pick line."""
        t0 = time.monotonic()
        q = await _queue_with(_worse(t0 + 1.0), _better(t0 + 2.0))
        with caplog.at_level(logging.INFO):
            got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is not None and got.thread_id == THREAD_BETTER, f"returned {_desc(got)}"
        lines = _admitted_lines(caplog)
        assert len(lines) == 1, lines
        assert "via=pop_matched_ip" in lines[0]
        assert "fastlane_matched=True" in lines[0]
        assert not [r for r in caplog.records if "QUEUE_PRIORITY" in r.getMessage()]

    async def test_control_no_match_logs_nothing(self, caplog):
        """Control: a bypass call that matches nothing emits no admitted-without-pick line."""
        q = await _queue_with(_better())
        with caplog.at_level(logging.INFO):
            assert await q.pop_matched_thread("unknown-thread", MODEL) is None
        assert _admitted_lines(caplog) == []

    async def test_declined_thread_match_logs_nothing_as_admitted(self, caplog):
        """A thread match declined because a better-ranked request of the same client waits logs no admitted-without-pick line."""
        q = await _queue_with(_worse(), _better())
        with caplog.at_level(logging.INFO):
            got = await q.pop_matched_thread(THREAD_WORSE, MODEL)
        assert got is None, f"returned {_desc(got)}"
        assert _admitted_lines(caplog) == []
