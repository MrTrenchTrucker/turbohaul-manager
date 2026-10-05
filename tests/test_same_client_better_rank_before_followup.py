"""Same-client rank rule for the grace-window matchers.

Within ONE client (same ``rule_index``) a waiting request with a strictly
better (lower) tag rank is served before a worse-ranked same-thread /
same-address / same-hash-chain follow-up.  The matchers
(``pop_matched_thread``, ``pop_matched_ip``, ``drain_inbox_and_staging_match_ip``,
``drain_inbox_and_staging_match_hash_chain``) skip such a follow-up; it stays
waiting at its position and is served later, in rank order.

Rank is never compared across clients: a better rank under another
``rule_index`` does not hold anything back.  Equal rank is not better.
Unlisted traffic is never held back.

Every test asserts WHICH request was returned (identity), and each group has a
control that shows the blocking arm and the non-blocking arm give different
answers.
"""

import asyncio
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
FOLLOW_THREAD = "thread-follow"
OTHER_THREAD = "thread-other"
ANCHOR_CHAIN = ["h1"]


def _slot(thread_id, rank, *, rule_index=0, listed=True, created_at=None,
          model=MODEL, chain=None):
    s = Slot.new(
        model_tag=model, prompt=f"{thread_id}/r{rank}/i{rule_index}",
        thread_id=thread_id, client_meta={"ip": IP},
        admission_hash_chain=chain,
    )
    if listed:
        s.fastlane = FastLaneMatch(
            rule_index=rule_index, raw_address=IP, label="",
            effective_tag=f"tag{rank}", rank=rank,
        )
    if created_at is not None:
        s.created_at = created_at
    return s


async def _queue_with(*slots, staging_max=10, acceptance_max=10):
    q = TurbohaulQueue(staging_max=staging_max, acceptance_max=acceptance_max)
    for s in slots:
        await q.enqueue(s)
    return q


def _policy():
    return FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)


async def _next(q):
    return await q.pop_next(warm_model_tag=MODEL, fastlane_policy=_policy())


def _d(s):
    return "None" if s is None else s.prompt


def _fresh_inbox(*slots):
    inbox: "asyncio.Queue" = asyncio.Queue()
    for s in slots:
        inbox.put_nowait(s)
    return inbox


def _inbox_contents(inbox):
    out = []
    while not inbox.empty():
        out.append(inbox.get_nowait())
    return out


class _Boom(RuntimeError):
    pass


async def _exception_raised_by(q, inbox, call, *, target):
    """Make ``q.<target>`` raise once, run ``call`` and return the exception it raised."""
    real = getattr(q, target)

    def boom(*args, **kwargs):
        raise _Boom(target)

    setattr(q, target, boom)
    try:
        try:
            await call()
        except _Boom as exc:
            return exc
        return None
    finally:
        setattr(q, target, real)


# ---------------------------------------------------------------------------
# pop_matched_thread
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestThreadMatcher:
    async def test_followup_yields_then_is_served_after_better_is_popped(self):
        """(a) the declined follow-up is not lost: better first, follow-up second."""
        follow = _slot(FOLLOW_THREAD, 3)
        better = _slot(OTHER_THREAD, 1)
        q = await _queue_with(follow, better)
        first_try = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert first_try is None, f"follow-up served while a better one waited: {_d(first_try)}"
        assert list(q._staging) == [follow, better], "declined candidate changed the buffer"
        popped = await _next(q)
        assert popped is better, f"pop_next served {_d(popped)}"
        second_try = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert second_try is follow, f"follow-up lost: matcher returned {_d(second_try)}"
        assert len(q._staging) == 0

    async def test_pop_next_order_is_better_then_followup(self):
        follow = _slot(FOLLOW_THREAD, 3)
        better = _slot(OTHER_THREAD, 1)
        q = await _queue_with(follow, better)
        order = [await _next(q), await _next(q)]
        assert order == [better, follow], [_d(s) for s in order]

    async def test_equal_rank_followup_is_served_by_matcher(self):
        """(b) equal rank is not better: the follow-up is served as before."""
        follow = _slot(FOLLOW_THREAD, 2)
        peer = _slot(OTHER_THREAD, 2)
        q = await _queue_with(peer, follow)
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert got is follow, f"returned {_d(got)}"
        assert list(q._staging) == [peer]

    async def test_control_strictly_better_rank_blocks_equal_rank_does_not(self):
        """Control for (b): only the rank value differs between the two arms."""
        blocked = await _queue_with(_slot(FOLLOW_THREAD, 2), _slot(OTHER_THREAD, 1))
        free = await _queue_with(_slot(FOLLOW_THREAD, 2), _slot(OTHER_THREAD, 2))
        a = await blocked.pop_matched_thread(FOLLOW_THREAD, MODEL)
        b = await free.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert a is None and b is not None and b.thread_id == FOLLOW_THREAD, (_d(a), _d(b))

    async def test_better_rank_of_another_client_does_not_block(self):
        """(c) rank is never compared across rule_index values."""
        follow = _slot(FOLLOW_THREAD, 3, rule_index=1)
        other_client = _slot(OTHER_THREAD, 1, rule_index=0)
        q = await _queue_with(other_client, follow)
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert got is follow, f"returned {_d(got)}"
        assert list(q._staging) == [other_client]

    async def test_control_same_rule_index_blocks_where_other_rule_index_did_not(self):
        """Control for (c): only the other slot's rule_index differs."""
        same = await _queue_with(_slot(FOLLOW_THREAD, 3, rule_index=1),
                                 _slot(OTHER_THREAD, 1, rule_index=1))
        diff = await _queue_with(_slot(FOLLOW_THREAD, 3, rule_index=1),
                                 _slot(OTHER_THREAD, 1, rule_index=0))
        a = await same.pop_matched_thread(FOLLOW_THREAD, MODEL)
        b = await diff.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert a is None and b is not None, (_d(a), _d(b))

    async def test_better_rank_of_a_later_rule_index_does_not_block_either(self):
        """(c) the other client sits LATER in the list and has the better rank."""
        follow = _slot(FOLLOW_THREAD, 3, rule_index=0)
        other_client = _slot(OTHER_THREAD, 0, rule_index=2)
        q = await _queue_with(follow, other_client)
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert got is follow, f"returned {_d(got)}"

    async def test_unlisted_followup_is_never_blocked(self):
        """(d) an unlisted follow-up is served even with a listed request waiting."""
        follow = _slot(FOLLOW_THREAD, 3, listed=False)
        listed = _slot(OTHER_THREAD, 1)
        q = await _queue_with(follow, listed)
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert got is follow, f"returned {_d(got)}"
        assert list(q._staging) == [listed]

    async def test_unlisted_waiter_does_not_block_a_listed_followup(self):
        """(d) an unlisted waiting request never counts as better."""
        follow = _slot(FOLLOW_THREAD, 3)
        unlisted = _slot(OTHER_THREAD, 1, listed=False)
        q = await _queue_with(follow, unlisted)
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert got is follow, f"returned {_d(got)}"

    async def test_better_request_in_acceptance_buffer_blocks(self):
        """The waiting better request may sit in the acceptance buffer."""
        follow = _slot(FOLLOW_THREAD, 3)
        better = _slot(OTHER_THREAD, 1)
        q = await _queue_with(follow, better, staging_max=1)
        assert list(q._staging) == [follow] and list(q._accept_buf) == [better]
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert got is None, f"returned {_d(got)}"
        assert list(q._staging) == [follow] and list(q._accept_buf) == [better]

    async def test_declined_candidate_keeps_its_position(self):
        """(f) the buffer is unchanged after a decline."""
        x = _slot("tx", 5, rule_index=4)
        follow = _slot(FOLLOW_THREAD, 3)
        better = _slot(OTHER_THREAD, 1)
        y = _slot("ty", 5, rule_index=5, listed=False)
        q = await _queue_with(x, follow, better, y)
        before = list(q._staging)
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert got is None
        assert list(q._staging) == before, [_d(s) for s in q._staging]

    async def test_running_request_is_not_a_waiting_better_one(self):
        """A better-ranked request already popped (running) no longer blocks."""
        follow = _slot(FOLLOW_THREAD, 3)
        better = _slot(OTHER_THREAD, 1)
        q = await _queue_with(follow, better)
        running = await _next(q)
        assert running is better
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)
        assert got is follow, f"returned {_d(got)}"


# ---------------------------------------------------------------------------
# pop_matched_ip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestIpMatcher:
    async def test_declined_ip_candidate_is_served_after_better(self):
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, created_at=t0 + 1)
        better = _slot(OTHER_THREAD, 1, created_at=t0 - 1)  # older than grace: not a match candidate
        q = await _queue_with(follow, better)
        assert await q.pop_matched_ip(IP, MODEL, grace_started_at=t0) is None
        assert list(q._staging) == [follow, better]
        assert await _next(q) is better
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is follow, f"returned {_d(got)}"

    async def test_scan_continues_to_a_later_unblocked_candidate(self):
        t0 = time.monotonic()
        worst = _slot("t3", 3, created_at=t0 + 1)
        mid = _slot("t2", 2, created_at=t0 + 2)
        best = _slot("t1", 1, created_at=t0 + 3)
        q = await _queue_with(worst, mid, best)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is best, f"returned {_d(got)}"
        assert list(q._staging) == [worst, mid]

    async def test_equal_rank_first_arrival_is_served(self):
        t0 = time.monotonic()
        a = _slot("ta", 2, created_at=t0 + 1)
        b = _slot("tb", 2, created_at=t0 + 2)
        q = await _queue_with(a, b)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is a, f"returned {_d(got)}"

    async def test_other_client_better_rank_does_not_block(self):
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, rule_index=1, created_at=t0 + 1)
        other_client = _slot(OTHER_THREAD, 1, rule_index=0, created_at=t0 - 1)
        q = await _queue_with(follow, other_client)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is follow, f"returned {_d(got)}"

    async def test_unlisted_candidate_is_never_blocked(self):
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, listed=False, created_at=t0 + 1)
        listed = _slot(OTHER_THREAD, 1, created_at=t0 - 1)
        q = await _queue_with(follow, listed)
        got = await q.pop_matched_ip(IP, MODEL, grace_started_at=t0)
        assert got is follow, f"returned {_d(got)}"


# ---------------------------------------------------------------------------
# drain_inbox_and_staging_match_ip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestIpDrain:
    async def _drain(self, q, inbox, t0):
        return await q.drain_inbox_and_staging_match_ip(IP, MODEL, t0, inbox)

    async def test_staged_followup_yields_to_better_in_staging(self):
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, created_at=t0 + 1)
        better = _slot(OTHER_THREAD, 1, created_at=t0 - 1)
        q = await _queue_with(follow, better)
        inbox = _fresh_inbox()
        assert await self._drain(q, inbox, t0) is None
        assert list(q._staging) == [follow, better]
        assert await _next(q) is better
        got = await self._drain(q, inbox, t0)
        assert got is follow, f"returned {_d(got)}"

    async def test_staged_followup_yields_to_better_in_inbox_which_is_kept(self):
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, created_at=t0 + 1)
        better = _slot(OTHER_THREAD, 1, created_at=t0 - 1)  # not a match candidate itself
        q = await _queue_with(follow)
        inbox = _fresh_inbox(better)
        assert await self._drain(q, inbox, t0) is None
        assert list(q._staging) == [follow]
        assert _inbox_contents(inbox) == [better]

    async def test_inbox_followup_yields_to_better_in_staging(self):
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, created_at=t0 + 1)
        better = _slot(OTHER_THREAD, 1, created_at=t0 - 1)
        q = await _queue_with(better)
        inbox = _fresh_inbox(follow)
        assert await self._drain(q, inbox, t0) is None
        assert _inbox_contents(inbox) == [follow]
        assert list(q._staging) == [better]

    async def test_inbox_better_candidate_is_served_before_staged_worse(self):
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, created_at=t0 + 1)
        better = _slot(OTHER_THREAD, 1, created_at=t0 + 2)
        q = await _queue_with(follow)
        inbox = _fresh_inbox(better)
        got = await self._drain(q, inbox, t0)
        assert got is better, f"returned {_d(got)}"
        assert list(q._staging) == [follow]
        assert _inbox_contents(inbox) == []

    async def test_staging_hit_leaves_inbox_untouched_and_in_order(self):
        t0 = time.monotonic()
        hit = _slot(FOLLOW_THREAD, 2, created_at=t0 + 1)
        i1 = _slot("i1", 2, rule_index=3, created_at=t0 - 1)
        i2 = _slot("i2", 2, rule_index=4, created_at=t0 - 1)
        q = await _queue_with(hit)
        inbox = _fresh_inbox(i1, i2)
        got = await self._drain(q, inbox, t0)
        assert got is hit, f"returned {_d(got)}"
        assert _inbox_contents(inbox) == [i1, i2]

    async def test_equal_rank_and_other_client_and_unlisted_are_not_blocked(self):
        t0 = time.monotonic()
        for label, follow, waiter in (
            ("equal", _slot("f", 2, created_at=t0 + 1), _slot("w", 2, created_at=t0 - 1)),
            ("other_client", _slot("f", 3, rule_index=1, created_at=t0 + 1),
             _slot("w", 1, rule_index=0, created_at=t0 - 1)),
            ("unlisted", _slot("f", 3, listed=False, created_at=t0 + 1),
             _slot("w", 1, created_at=t0 - 1)),
        ):
            q = await _queue_with(follow)
            got = await self._drain(q, _fresh_inbox(waiter), t0)
            assert got is follow, f"{label}: returned {_d(got)}"

    async def test_inbox_worse_is_declined_by_better_that_is_also_in_the_inbox(self):
        """Inbox against inbox: the worse-ranked entry is checked first and the
        better-ranked one of the same client sits LATER in the same inbox (it is
        older than the grace start, so it is not a candidate itself). The worse
        entry is declined, nothing is served and the inbox is intact."""
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, created_at=t0 + 1)
        better = _slot(OTHER_THREAD, 1, created_at=t0 - 1)
        q = await _queue_with()
        inbox = _fresh_inbox(follow, better)
        assert await self._drain(q, inbox, t0) is None
        assert _inbox_contents(inbox) == [follow, better]

    async def test_inbox_better_candidate_is_served_when_the_worse_precedes_it(self):
        """Inbox against inbox, both candidates: the worse entry comes first in the
        inbox, the better one second. The better one is served, the worse one stays."""
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, created_at=t0 + 1)
        better = _slot(OTHER_THREAD, 1, created_at=t0 + 2)
        q = await _queue_with()
        inbox = _fresh_inbox(follow, better)
        got = await self._drain(q, inbox, t0)
        assert got is better, f"returned {_d(got)}"
        assert _inbox_contents(inbox) == [follow]

    async def test_control_equal_rank_inbox_entries_do_not_block_each_other(self):
        """Control for the two tests above: only the rank of the second inbox entry
        differs (equal instead of better); the first entry is then served."""
        t0 = time.monotonic()
        follow = _slot(FOLLOW_THREAD, 3, created_at=t0 + 1)
        peer = _slot(OTHER_THREAD, 3, created_at=t0 - 1)
        q = await _queue_with()
        inbox = _fresh_inbox(follow, peer)
        got = await self._drain(q, inbox, t0)
        assert got is follow, f"returned {_d(got)}"
        assert _inbox_contents(inbox) == [peer]

    async def test_inbox_is_restored_when_the_staging_scan_raises(self):
        """An exception in the staging scan reaches the caller and the inbox still holds
        every entry in the original order."""
        t0 = time.monotonic()
        a = _slot("a", 2, created_at=t0 - 1)
        b = _slot("b", 1, created_at=t0 - 1)
        q = await _queue_with(_slot(FOLLOW_THREAD, 3, created_at=t0 + 1))
        inbox = _fresh_inbox(a, b)
        exc = await _exception_raised_by(
            q, inbox, lambda: self._drain(q, inbox, t0), target="_pop_first_matching_locked")
        assert exc is not None, "the injected failure did not reach the caller"
        assert _inbox_contents(inbox) == [a, b]
        assert not q._lock.locked(), "the queue lock was left held"

    async def test_inbox_is_restored_when_the_inbox_decision_raises(self):
        """An exception while the inbox entries are being weighed reaches the caller and
        the inbox still holds every entry in the original order."""
        t0 = time.monotonic()
        a = _slot("a", 2, created_at=t0 + 1)
        b = _slot("b", 1, created_at=t0 - 1)
        q = await _queue_with()
        inbox = _fresh_inbox(a, b)
        exc = await _exception_raised_by(
            q, inbox, lambda: self._drain(q, inbox, t0),
            target="_decline_for_better_same_client_locked")
        assert exc is not None, "the injected failure did not reach the caller"
        assert _inbox_contents(inbox) == [a, b]
        assert not q._lock.locked(), "the queue lock was left held"


# ---------------------------------------------------------------------------
# drain_inbox_and_staging_match_hash_chain
# ---------------------------------------------------------------------------

def _chain_slot(name, rank, *, rule_index=0, listed=True, created_at, chain):
    return _slot(name, rank, rule_index=rule_index, listed=listed,
                 created_at=created_at, chain=chain)


@pytest.mark.asyncio
class TestHashChainDrain:
    async def _drain(self, q, inbox, t0):
        return await q.drain_inbox_and_staging_match_hash_chain(
            ANCHOR_CHAIN, MODEL, t0, inbox,
        )

    async def test_staged_followup_yields_then_is_served_after_better(self):
        t0 = time.monotonic()
        follow = _chain_slot("f", 3, created_at=t0 + 1, chain=["h1", "h2"])
        better = _chain_slot("b", 1, created_at=t0 - 1, chain=["x"])  # same client, not a chain match
        q = await _queue_with(follow, better)
        inbox = _fresh_inbox()
        assert await self._drain(q, inbox, t0) is None
        assert list(q._staging) == [follow, better]
        assert await _next(q) is better
        got = await self._drain(q, inbox, t0)
        assert got is follow, f"returned {_d(got)}"

    async def test_staged_followup_yields_to_better_in_inbox_which_is_kept(self):
        t0 = time.monotonic()
        follow = _chain_slot("f", 3, created_at=t0 + 1, chain=["h1", "h2"])
        better = _chain_slot("b", 1, created_at=t0 + 1, chain=["x"])
        q = await _queue_with(follow)
        inbox = _fresh_inbox(better)
        assert await self._drain(q, inbox, t0) is None
        assert list(q._staging) == [follow]
        assert _inbox_contents(inbox) == [better]

    async def test_inbox_followup_yields_to_better_in_staging(self):
        t0 = time.monotonic()
        follow = _chain_slot("f", 3, created_at=t0 + 1, chain=["h1", "h2"])
        better = _chain_slot("b", 1, created_at=t0 + 1, chain=["x"])
        q = await _queue_with(better)
        inbox = _fresh_inbox(follow)
        assert await self._drain(q, inbox, t0) is None
        assert _inbox_contents(inbox) == [follow]
        assert list(q._staging) == [better]

    async def test_better_chain_match_in_inbox_is_served_before_staged_worse(self):
        t0 = time.monotonic()
        follow = _chain_slot("f", 3, created_at=t0 + 1, chain=["h1", "h2"])
        better = _chain_slot("b", 1, created_at=t0 + 1, chain=["h1", "h3"])
        q = await _queue_with(follow)
        inbox = _fresh_inbox(better)
        got = await self._drain(q, inbox, t0)
        assert got is better, f"returned {_d(got)}"
        assert list(q._staging) == [follow]

    async def test_equal_rank_other_client_unlisted_are_not_blocked(self):
        t0 = time.monotonic()
        mk = _chain_slot
        for label, follow, waiter in (
            ("equal", mk("f", 2, created_at=t0 + 1, chain=["h1", "h2"]),
             mk("w", 2, created_at=t0 + 1, chain=["x"])),
            ("other_client", mk("f", 3, rule_index=1, created_at=t0 + 1, chain=["h1", "h2"]),
             mk("w", 1, rule_index=0, created_at=t0 + 1, chain=["x"])),
            ("unlisted", mk("f", 3, listed=False, created_at=t0 + 1, chain=["h1", "h2"]),
             mk("w", 1, created_at=t0 + 1, chain=["x"])),
        ):
            q = await _queue_with(follow)
            got = await self._drain(q, _fresh_inbox(waiter), t0)
            assert got is follow, f"{label}: returned {_d(got)}"

    async def test_inbox_worse_is_declined_by_better_that_is_also_in_the_inbox(self):
        """Inbox against inbox: the worse-ranked chain match is first in the inbox, the
        better-ranked entry of the same client (not a chain match) is later in it. The
        worse entry is declined, nothing is served and the inbox is intact."""
        t0 = time.monotonic()
        follow = _chain_slot("f", 3, created_at=t0 + 1, chain=["h1", "h2"])
        better = _chain_slot("b", 1, created_at=t0 + 1, chain=["x"])
        q = await _queue_with()
        inbox = _fresh_inbox(follow, better)
        assert await self._drain(q, inbox, t0) is None
        assert _inbox_contents(inbox) == [follow, better]

    async def test_control_equal_rank_inbox_entries_do_not_block_each_other(self):
        """Control: only the rank of the second inbox entry differs (equal instead of
        better); the first entry, a chain match, is then served."""
        t0 = time.monotonic()
        follow = _chain_slot("f", 3, created_at=t0 + 1, chain=["h1", "h2"])
        peer = _chain_slot("p", 3, created_at=t0 + 1, chain=["x"])
        q = await _queue_with()
        inbox = _fresh_inbox(follow, peer)
        got = await self._drain(q, inbox, t0)
        assert got is follow, f"returned {_d(got)}"
        assert _inbox_contents(inbox) == [peer]

    async def test_inbox_is_restored_when_the_staging_scan_raises(self):
        """An exception in the staging scan reaches the caller and the inbox still holds
        every entry in the original order."""
        t0 = time.monotonic()
        a = _chain_slot("a", 2, created_at=t0 + 1, chain=["x"])
        b = _chain_slot("b", 1, created_at=t0 + 1, chain=["y"])
        q = await _queue_with(_chain_slot("f", 3, created_at=t0 + 1, chain=["h1", "h2"]))
        inbox = _fresh_inbox(a, b)
        exc = await _exception_raised_by(
            q, inbox, lambda: self._drain(q, inbox, t0), target="_pop_first_matching_locked")
        assert exc is not None, "the injected failure did not reach the caller"
        assert _inbox_contents(inbox) == [a, b]
        assert not q._lock.locked(), "the queue lock was left held"

    async def test_inbox_is_restored_when_the_inbox_decision_raises(self):
        """An exception while the inbox entries are being weighed reaches the caller and
        the inbox still holds every entry in the original order."""
        t0 = time.monotonic()
        a = _chain_slot("a", 3, created_at=t0 + 1, chain=["h1", "h2"])
        b = _chain_slot("b", 1, created_at=t0 + 1, chain=["y"])
        q = await _queue_with()
        inbox = _fresh_inbox(a, b)
        exc = await _exception_raised_by(
            q, inbox, lambda: self._drain(q, inbox, t0),
            target="_decline_for_better_same_client_locked")
        assert exc is not None, "the injected failure did not reach the caller"
        assert _inbox_contents(inbox) == [a, b]
        assert not q._lock.locked(), "the queue lock was left held"
