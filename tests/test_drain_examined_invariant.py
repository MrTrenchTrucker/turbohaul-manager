"""drain_inbox_and_staging_match_ip's post-condition.

The old post-condition (`assert inbox.empty()`, queue.py) contradicted the
function's own design: it re-enqueues non-matching slots on purpose, then
asserted the inbox is empty. Measured firing table:

    empty inbox                -> passes
    1 slot, MATCHES             -> passes
    1 slot, does NOT match      -> FIRES
    3 slots, 1 matches          -> FIRES
    2 slots, BOTH MATCH         -> FIRES   <- narrow wording misses this
    2 slots, NEITHER matches    -> FIRES
    inbox=None                  -> passes (peek skipped)

The correct invariant is EVERY SLOT EXAMINED, not inbox-empty:
examined (== len(_peeked) + (1 if matched else 0)) == depth_at_entry
(== inbox.qsize() before the drain loop starts). This file drives the real
TurbohaulQueue.drain_inbox_and_staging_match_ip end-to-end (no mocking of the
method under test) for every arm in the table above, plus a non-vacuity
control that proves the NEW check is live code, not a tautology.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from turbohaul.queue import TurbohaulQueue
from turbohaul.slot import Slot

_MODEL = "test-model"
_ANCHOR_IP = "10.0.0.5"
_OTHER_IP = "10.0.0.9"


def _slot(*, ip: str = _ANCHOR_IP, model_tag: str = _MODEL, created_at: float) -> Slot:
    s = Slot.new(model_tag, client_meta={"ip": ip})
    s.created_at = created_at
    return s


async def _drain(q: TurbohaulQueue, inbox, grace_started_at: float, *, ip: str = _ANCHOR_IP):
    sink: dict = {}
    matched = await q.drain_inbox_and_staging_match_ip(
        ip, _MODEL, grace_started_at, inbox=inbox, decline_sink=sink,
    )
    return matched, sink


# ---------------------------------------------------------------------------
# Controls: must pass under BOTH the old and the new post-condition -- these
# document that the fix changes nothing about the cases that were already
# correct, only the ones that were wrongly firing.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestControlsUnaffectedByFix:
    async def test_empty_inbox_passes(self):
        q = TurbohaulQueue(staging_max=10)
        grace_started_at = time.monotonic()
        inbox: "asyncio.Queue" = asyncio.Queue()
        matched, sink = await _drain(q, inbox, grace_started_at)
        assert matched is None
        assert sink["reason"] == "no_candidate_matched"
        assert inbox.qsize() == 0

    async def test_none_inbox_skips_peek(self):
        q = TurbohaulQueue(staging_max=10)
        grace_started_at = time.monotonic()
        matched, sink = await _drain(q, None, grace_started_at)
        assert matched is None
        assert sink["inbox_depth"] == 0

    async def test_single_slot_matches_passes(self):
        q = TurbohaulQueue(staging_max=10)
        grace_started_at = time.monotonic()
        s = _slot(created_at=grace_started_at + 1)
        inbox: "asyncio.Queue" = asyncio.Queue()
        inbox.put_nowait(s)
        matched, sink = await _drain(q, inbox, grace_started_at)
        assert matched is s
        assert inbox.qsize() == 0


# ---------------------------------------------------------------------------
# The four arms that FIRE the old `assert inbox.empty()` -- RED on the
# unmodified code (the old post-condition raises an AssertionError),
# GREEN on the fixed code (this file, run against the
# current implementation).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
class TestFormerlyFiringArmsNowGreen:
    async def test_single_slot_no_match(self):
        q = TurbohaulQueue(staging_max=10)
        grace_started_at = time.monotonic()
        s = _slot(ip=_OTHER_IP, created_at=grace_started_at + 1)
        inbox: "asyncio.Queue" = asyncio.Queue()
        inbox.put_nowait(s)
        matched, sink = await _drain(q, inbox, grace_started_at)
        assert matched is None
        assert sink["reason"] == "no_candidate_matched"
        # non-match must still be sitting back in the inbox for the next poll
        assert inbox.qsize() == 1
        assert inbox.get_nowait() is s

    async def test_three_slots_one_matches(self):
        q = TurbohaulQueue(staging_max=10)
        grace_started_at = time.monotonic()
        miss1 = _slot(ip=_OTHER_IP, created_at=grace_started_at + 1)
        hit = _slot(created_at=grace_started_at + 2)
        miss2 = _slot(ip=_OTHER_IP, created_at=grace_started_at + 3)
        inbox: "asyncio.Queue" = asyncio.Queue()
        for s in (miss1, hit, miss2):
            inbox.put_nowait(s)
        matched, sink = await _drain(q, inbox, grace_started_at)
        assert matched is hit
        # both non-matches restored, FIFO order preserved
        assert inbox.qsize() == 2
        assert inbox.get_nowait() is miss1
        assert inbox.get_nowait() is miss2

    async def test_two_slots_both_match(self):
        """The both-match arm: the narrow headline ("fires whenever a
        NON-matching slot is present") would predict this passes, since
        nothing here fails to match. Measured: it FIRES on old code anyway,
        because the loop takes only the FIRST match and treats every later
        match as a non-match to re-enqueue."""
        q = TurbohaulQueue(staging_max=10)
        grace_started_at = time.monotonic()
        first = _slot(created_at=grace_started_at + 1)
        second = _slot(created_at=grace_started_at + 2)
        inbox: "asyncio.Queue" = asyncio.Queue()
        inbox.put_nowait(first)
        inbox.put_nowait(second)
        matched, sink = await _drain(q, inbox, grace_started_at)
        # only the FIRST match is served this call -- unchanged semantics
        assert matched is first
        assert inbox.qsize() == 1
        assert inbox.get_nowait() is second

    async def test_two_slots_neither_matches(self):
        q = TurbohaulQueue(staging_max=10)
        grace_started_at = time.monotonic()
        miss1 = _slot(ip=_OTHER_IP, created_at=grace_started_at + 1)
        miss2 = _slot(ip=_OTHER_IP, created_at=grace_started_at + 2)
        inbox: "asyncio.Queue" = asyncio.Queue()
        inbox.put_nowait(miss1)
        inbox.put_nowait(miss2)
        matched, sink = await _drain(q, inbox, grace_started_at)
        assert matched is None
        assert inbox.qsize() == 2


# ---------------------------------------------------------------------------
# NON-VACUITY control: prove the NEW invariant is live code that CAN fail,
# not a tautology that always holds given the current loop body. Analysis
# shows examined == depth_at_entry holds by
# construction of today's loop -- there is no await between capturing
# depth_at_entry and the check, so nothing can reach this function's inbox
# arg mid-drain. That means none of the four legitimate scenarios above can
# ever exercise the new raise. This queue lies about `empty()` to reproduce
# a head-only-drain pattern directly -- a real slot silently left
# un-popped and un-accounted -- independent of whether today's code can
# produce that state on its own.
# ---------------------------------------------------------------------------
class _HeadOnlyDrainQueue(asyncio.Queue):
    def __init__(self, items):
        super().__init__()
        for it in items:
            self.put_nowait(it)
        self._lied_once = False

    def empty(self):
        real_empty = super().empty()
        if not real_empty and self.qsize() == 1 and not self._lied_once:
            self._lied_once = True
            return True  # lie: one real item is still sitting in the queue
        return real_empty


@pytest.mark.asyncio
class TestNonVacuity:
    async def test_new_invariant_fires_on_a_genuinely_lost_slot(self):
        q = TurbohaulQueue(staging_max=10)
        grace_started_at = time.monotonic()
        miss1 = _slot(ip=_OTHER_IP, created_at=grace_started_at + 1)
        miss2 = _slot(ip=_OTHER_IP, created_at=grace_started_at + 2)
        inbox = _HeadOnlyDrainQueue([miss1, miss2])
        with pytest.raises(AssertionError, match=r"examined 1 of 2 slots present at entry"):
            await _drain(q, inbox, grace_started_at)
