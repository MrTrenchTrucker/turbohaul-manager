"""The claimant must stay visible while it waits.

Design requirement --
"The queue tab shows the waiting high-priority request THE WHOLE TIME --
client, tag, rank, and 'waiting for a turn to complete' -- for as long as the
lower-priority turn is in flight." The acceptance test for it
names the same instrument: "The queue surface showed the claimant waiting with its
client, tag, and rank throughout."

Decision: the fix is the QUEUE SURFACE, not the claim's lifetime --
the whole-time duty and its "rank" field both belong to the queue
tab, while the claim record carries neither. The claim's lifetime is
deliberately UNCHANGED by the queue-surface fix these tests cover. (A separate
mechanism keeps a parked request's claim until its own turn starts; the
hand-off sites park the claim instead of releasing it as admitted, and
the structural scan below pins that.)

TWO DEFECTS, both of the visibility requirement, both fixed together (one requirement,
one surface, one change):

  (a)   TRUNCATION (queue.py). ``queue_snapshot``'s prefix cap dropped any row
        past ``limit``. A claimant has NO ordering privilege while it waits --
        the function's own docstring says Fast Lane priority "is only applied
        inside pop_next's pick, never while a slot rests in either deque" --
        so at depth > limit the claimant is simply truncated away.
        => These tests go RED on the REAL pre-fix code.

  (b)   LIFETIME-OF-THE-WAIT (manager.py). All three
        ``_release_fastlane_claim_locked(slot, "admitted")`` sites fired at the
        INBOX HAND-OFF, which is not the end of waiting: the slot has already
        left ``_staging`` (so the queue surface loses it) and its claim was
        deleted (so the claim surface loses it), while it still only sits in
        ``r.inbox``. At the third site it waits out an entire cold spawn.
        => These CANNOT go RED pre-fix for the defect's own reason: the
        registry and its reader do not exist on unfixed code, so the test
        would die on AttributeError, and a test that failed because the
        symbol was missing has tested nothing. The way to pin such a
        test is to prove it instead by MUTANTS of the change's own work.
"""
import inspect
import itertools

import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import TurbohaulManager
from turbohaul.queue import TurbohaulQueue
from turbohaul.slot import Slot, SlotState

_slot_id_seq = itertools.count()


@pytest.fixture
def mgr():
    """A manager built WITHOUT boot config, on purpose.

    Every method under test here (`_note_handed_off_waiting`,
    `_handed_off_waiting_rows`, `_waiting_surface_rows`) reads and writes
    exactly one attribute, `_handed_off_waiting`. Standing up the real
    BootConfig fixture would drag in the sqlite state DB whose first-ever
    `PRAGMA journal_mode=WAL` races its own busy_timeout -- a known flake
    already documented in the fast-lane claims test fixture.
    Importing that risk into tests that cannot exercise it would make a red
    here ambiguous between the code and the harness.
    """
    m = TurbohaulManager.__new__(TurbohaulManager)
    m._handed_off_waiting = {}
    return m


def _match(rule_index=0, rank=1, raw_address="10.0.0.1"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address=raw_address, label="",
        effective_tag="main", rank=rank,
    )


def _slot(model_tag="m1", *, fastlane=None, thread_id="", state=SlotState.STAGED):
    # Sequence, not id(object()) -- a short-lived object's id() can be reused
    # by CPython, which silently collapses two slots onto one slot_id and
    # makes every set-based membership assertion below undercount.
    return Slot(
        slot_id=f"waiting-{next(_slot_id_seq)}",
        model_tag=model_tag,
        state=state,
        thread_id=thread_id,
        client_meta={},
        fastlane=fastlane,
    )


def _queue_with(n_normal, *, claimant_at=None, limit_hint=200):
    """A staging deque of ``n_normal`` ordinary slots, optionally with one
    claimant spliced in at index ``claimant_at``.

    staging_max is set well above the population so nothing is diverted into
    the accept buffer -- this fixture is about the SNAPSHOT's cap, not the
    queue's own admission cap, and conflating the two would make a failure
    here ambiguous between the thing under test and the fixture.
    """
    q = TurbohaulQueue(staging_max=limit_hint, acceptance_max=10000)
    claimant = None
    for i in range(n_normal):
        if claimant_at is not None and i == claimant_at:
            claimant = _slot("fast", fastlane=_match(rank=0))
            q._staging.append(claimant)
        q._staging.append(_slot(f"normal-{i}"))
    if claimant_at is not None and claimant_at >= n_normal:
        claimant = _slot("fast", fastlane=_match(rank=0))
        q._staging.append(claimant)
    return q, claimant


# ---------------------------------------------------------------------------
# Part (a) -- truncation. RED on the real pre-fix code.
# ---------------------------------------------------------------------------

class TestAClaimantIsNeverTruncatedAway:
    def test_a_claimant_past_the_cap_is_still_on_the_surface(self):
        """THE DEFECT. 60 ordinary requests, then the claimant. The old
        prefix cap returned rows 0..49 and the claimant -- at true index 60 --
        was simply absent, which is the "whole time" requirement failing for
        the entire wait rather than for a window at the end of it."""
        q, claimant = _queue_with(60, claimant_at=60)
        rows = q.queue_snapshot(limit=50)
        ids = {r["slot_id"] for r in rows}
        assert claimant.slot_id in ids, (
            "the Fast Lane claimant sat at true index 60 and the cap of 50 "
            "dropped it: the requirement is that it is visible the WHOLE time, and "
            f"it was visible for none of it. got {len(rows)} rows, ids={sorted(ids)[:3]}..."
        )

    def test_the_preserved_claimant_carries_its_true_pre_cap_index(self):
        """``position`` KEEPS its true pre-cap index. A
        preserved claimant's row must say 60, not its ordinal in the returned
        list -- the docstring promises "where it sits right now", and its
        ordinal in a non-contiguous result is not where it sits."""
        q, claimant = _queue_with(60, claimant_at=60)
        rows = q.queue_snapshot(limit=50)
        matches = [r for r in rows if r["slot_id"] == claimant.slot_id]
        # Assert presence FIRST and by hand: `next(...)` on an empty result
        # raises StopIteration, and a test that dies on StopIteration has
        # reported a crash, not a defect.
        assert matches, (
            "the claimant is not on the surface at all, so its position "
            "cannot be checked -- the cap dropped it (see the sibling test)"
        )
        assert matches[0]["position"] == 60, (
            "the claimant's true index in staging is 60; the row reported "
            f"{matches[0]['position']}, which is an ordinal, not a position"
        )

    def test_normal_rows_stay_capped_so_the_payload_stays_bounded(self):
        """The other half of the requirement: never truncating a claimant must not
        turn into never truncating anything. With one claimant preserved, the
        normal rows get limit-1 of the budget, not all 60."""
        q, claimant = _queue_with(60, claimant_at=60)
        rows = q.queue_snapshot(limit=50)
        normal = [r for r in rows if r["fastlane"] is None]
        assert len(normal) == 49, (
            f"expected the claimant to consume one slot of the 50-row budget "
            f"leaving 49 normal rows; got {len(normal)}"
        )
        assert len(rows) == 50

    def test_many_claimants_are_all_kept_even_past_the_cap(self):
        """The disclosed consequence, asserted rather than left to be found:
        when claimants alone exceed ``limit`` the row count exceeds ``limit``
        too. That is what "never truncate a claimant" costs, it is bounded by
        the claimant population and not by the deque, and it is deliberate."""
        q = TurbohaulQueue(staging_max=500, acceptance_max=10000)
        claimants = [_slot("fast", fastlane=_match(rank=0)) for _ in range(60)]
        for c in claimants:
            q._staging.append(c)
        rows = q.queue_snapshot(limit=50)
        ids = {r["slot_id"] for r in rows}
        assert all(c.slot_id in ids for c in claimants), (
            "every claimant must survive the cap, even when they alone "
            f"exceed it; {sum(1 for c in claimants if c.slot_id not in ids)} were dropped"
        )
        assert len(rows) == 60

    def test_rows_are_returned_in_true_queue_order(self):
        """A preserved claimant is spliced back into position order, not
        appended -- an operator reading the surface top-to-bottom must not
        see a row from index 60 sitting above one from index 3.

        NOTE, NON-VACUITY: the ordering claim only means anything once the
        claimant is actually IN the result, and pre-fix it never is -- an
        all-normal prefix is trivially sorted, so a bare sorted() check
        would pass on BOTH arms and discriminate nothing. That is the exact
        shape that makes an ordering test vacuous. The presence
        assertion is what gives this teeth on the pre-fix arm; the sorted()
        assertion is what gives it teeth against a faulty version of the fix that
        appends the claimant instead of splicing it.

        NOTE, THE CLAIMANT MUST SIT EARLY -- that is the whole fixture. With
        the claimant at the LAST index its true position is larger than every
        normal row that survives the budget, so "append to the end" and
        "splice into place" produce the SAME sorted list. Measured, not
        assumed: with claimant_at=60 an appending version would pass every test
        in this file. At index 3 the appended row lands after position 49 and
        the order breaks, which is the only arrangement in which this test
        tests anything.
        """
        q, claimant = _queue_with(60, claimant_at=3)
        rows = q.queue_snapshot(limit=50)
        positions = [r["position"] for r in rows]
        assert claimant.slot_id in {r["slot_id"] for r in rows}, (
            "the claimant is absent, so this ordering check would be vacuous "
            "-- there is nothing here for it to be ordered against"
        )
        assert positions == sorted(positions), (
            f"rows are out of true-queue order -- the claimant was appended "
            f"rather than spliced: {positions[:8]}...{positions[-3:]}"
        )


class TestTheOrdinaryQueueIsUnchanged:
    """GREEN CONTROLS. These pass on BOTH arms by design -- they exist to
    prove the fix did not pay for the claimant with everyone else. Their
    teeth are proven separately by the always-preserve mutant, which reds `test_normal_rows_stay_capped...` above."""

    def test_with_no_claimant_the_snapshot_is_the_old_contiguous_prefix(self):
        q, _ = _queue_with(60)
        rows = q.queue_snapshot(limit=50)
        assert len(rows) == 50
        assert [r["position"] for r in rows] == list(range(50)), (
            "with no claimant present the result must be byte-identical to "
            "the old prefix: 50 rows, positions 0..49, contiguous"
        )
        assert all(r["fastlane"] is None for r in rows)

    def test_a_short_queue_is_returned_whole(self):
        q, _ = _queue_with(5)
        rows = q.queue_snapshot(limit=50)
        assert [r["position"] for r in rows] == [0, 1, 2, 3, 4]

    def test_the_membership_check_can_report_absence(self):
        """KNOWN-ABSENT PROBE. Every assertion above is a membership test, and
        a membership test that cannot say "no" proves nothing when it says
        "yes". This points the same check at a slot that was never enqueued."""
        q, _ = _queue_with(5)
        never_enqueued = _slot("ghost", fastlane=_match(rank=0))
        ids = {r["slot_id"] for r in q.queue_snapshot(limit=50)}
        assert never_enqueued.slot_id not in ids, (
            "the membership check reported a slot that was never enqueued -- "
            "it cannot distinguish present from absent and every green above "
            "is therefore worthless"
        )


# ---------------------------------------------------------------------------
# Part (b) -- the wait does not end at the hand-off.
# ---------------------------------------------------------------------------

class TestTheHandOffSitesAreWired:
    """STRUCTURAL GUARD for an absence-defect: the bug is a MISSING call, and
    a missing call at one of three sites is invisible to any behavioural test
    that happens to drive one of the other two.

    Scoped to the two enclosing functions ON PURPOSE -- ``_note_handed_off_
    waiting``'s own docstring quotes the release call verbatim, so a
    whole-file substring scan would match that prose and pass while the real
    wiring was gone.
    """

    @staticmethod
    def _code_lines(fn):
        return [
            ln.strip() for ln in inspect.getsource(fn).splitlines()
            if not ln.strip().startswith("#")
        ]

    def test_route_or_reserve_notes_both_of_its_hand_offs(self):
        lines = self._code_lines(TurbohaulManager._route_or_reserve)
        # a parked request keeps its claim until its own turn starts, so a
        # hand-off site parks the claim instead of releasing it as "admitted"
        parks = [i for i, ln in enumerate(lines)
                 if ln.startswith("self._park_fastlane_claim_locked(slot, ")]
        releases = [i for i, ln in enumerate(lines)
                    if ln == 'self._release_fastlane_claim_locked(slot, "admitted")']
        notes = [i for i, ln in enumerate(lines)
                 if ln == "self._note_handed_off_waiting(slot)"]
        assert len(parks) == 2, (
            f"_route_or_reserve should hand off at exactly 2 sites, found {len(parks)}"
        )
        assert releases == [], (
            "a hand-off must park the claim, not release it as admitted; "
            f"found {len(releases)} admitted releases"
        )
        assert len(notes) == 2, (
            f"each hand-off must record the claimant as still waiting; "
            f"{len(parks)} parks but only {len(notes)} notes"
        )
        for p, n in zip(parks, notes):
            assert n > p, "the note must follow the park it belongs to"

    def test_reserve_and_start_notes_its_hand_off(self):
        lines = self._code_lines(TurbohaulManager._reserve_and_start_locked)
        # a parked request keeps its claim until its own turn starts
        parks = [i for i, ln in enumerate(lines)
                 if ln.startswith("self._park_fastlane_claim_locked(slot, ")]
        releases = [i for i, ln in enumerate(lines)
                    if ln == 'self._release_fastlane_claim_locked(slot, "admitted")']
        notes = [i for i, ln in enumerate(lines)
                 if ln == "self._note_handed_off_waiting(slot)"]
        assert len(parks) == 1 and len(notes) == 1, (
            "the spawn path is the LONGEST invisible window -- a whole cold "
            f"load -- and it must be wired: {len(parks)} parks, {len(notes)} notes"
        )
        assert releases == [], (
            "the spawn hand-off must park the claim, not release it as "
            f"admitted; found {len(releases)} admitted releases"
        )
        assert notes[0] > parks[0]

    def test_the_scan_can_report_a_missing_note(self):
        """KNOWN-ABSENT PROBE for the guard itself: pointed at a function that
        legitimately has no hand-off, the same scan must find zero. A scan
        that returns a hit everywhere cannot fail and is not a guard."""
        lines = self._code_lines(TurbohaulManager._inbox_waiting_count)
        assert [ln for ln in lines
                if ln == "self._note_handed_off_waiting(slot)"] == []


class TestTheClaimantStaysOnTheSurfaceAfterHandOff:
    """Behaviour of the surface itself. Driven through the registry rather
    than through a live route, because the three real hand-off sites sit
    under _registry_lock inside a spawn path; `TestTheHandOffSitesAreWired`
    above is what proves those three sites actually call this."""

    def test_a_handed_off_claimant_appears_with_its_tag_and_rank(self, mgr):
        slot = _slot("m1", fastlane=_match(rank=0), thread_id="thread-abcdef123")
        mgr._note_handed_off_waiting(slot)
        rows = mgr._handed_off_waiting_rows()
        assert len(rows) == 1, f"expected the claimant on the surface, got {rows}"
        row = rows[0]
        assert row["model_tag"] == "m1"
        assert row["fastlane"]["rank"] == 0
        assert row["handed_off"] is True
        assert row["position"] is None, (
            "a handed-off row holds no index in either deque; inventing one "
            "would be the dishonest number queue_snapshot's docstring forbids"
        )

    def test_an_ordinary_request_is_not_recorded(self, mgr):
        """The requirement scopes the duty to the HIGH-PRIORITY request.
        Recording ordinary traffic would grow the registry with every routed
        request for no requirement."""
        mgr._note_handed_off_waiting(_slot("m1", fastlane=None))
        assert mgr._handed_off_waiting == {}

    def test_the_row_disappears_once_the_slot_is_actually_served(self, mgr):
        """The wait ENDS at the serve, and the surface must end with it --
        otherwise this fix trades an invisible claimant for a permanent
        ghost, which is worse than the defect."""
        slot = _slot("m1", fastlane=_match(rank=0))
        mgr._note_handed_off_waiting(slot)
        assert len(mgr._handed_off_waiting_rows()) == 1
        slot.state = SlotState.ACTIVE
        assert mgr._handed_off_waiting_rows() == []
        assert mgr._handed_off_waiting == {}, (
            "the entry must be PRUNED, not merely filtered -- a filtered-but-"
            "retained entry is an unbounded leak on a long-lived process"
        )

    def test_an_evicted_claimant_is_dropped(self, mgr):
        slot = _slot("m1", fastlane=_match(rank=0))
        mgr._note_handed_off_waiting(slot)
        slot.is_evicted = True
        assert mgr._handed_off_waiting_rows() == []

    def test_the_row_never_carries_client_meta_or_a_full_thread_id(self, mgr):
        """Same redaction rule the rest of this surface already follows."""
        slot = _slot("m1", fastlane=_match(rank=0), thread_id="thread-abcdef123456")
        mgr._note_handed_off_waiting(slot)
        row = mgr._handed_off_waiting_rows()[0]
        assert "client_meta" not in row
        assert "ip" not in row
        assert "thread_id" not in row
        assert row["thread_id_prefix"] == "thread-a"


class TestTheTwoPopulationsAreMergedWithoutDoubleCounting:
    """``_waiting_surface_rows`` is the seam where the two populations meet.
    A requeued claimant is briefly in BOTH, and that is a real state, not a
    defensive hypothetical: an inbox drain re-enqueues at the staging head
    while this registry still holds the slot (its state is STAGED again,
    which is precisely the predicate the registry keeps)."""

    def test_a_requeued_claimant_appears_once_not_twice(self, mgr):
        slot = _slot("m1", fastlane=_match(rank=0))
        mgr._note_handed_off_waiting(slot)
        # the drain put it back in staging, so queue_snapshot sees it again
        queue_rows = [{"slot_id": slot.slot_id, "position": 0, "fastlane": None}]
        rows = mgr._waiting_surface_rows(queue_rows)
        assert len(rows) == 1, (
            "the claimant is in the registry AND back in staging; without a "
            f"dedup the surface reports it twice and overstates the queue: {rows}"
        )
        assert rows[0]["position"] == 0, (
            "the queue row must win -- it carries a real position and the "
            "handed-off row cannot"
        )

    def test_a_claimant_still_in_an_inbox_leads_the_surface(self, mgr):
        slot = _slot("m1", fastlane=_match(rank=0))
        mgr._note_handed_off_waiting(slot)
        queue_rows = [{"slot_id": "other", "position": 0, "fastlane": None}]
        rows = mgr._waiting_surface_rows(queue_rows)
        assert [r["slot_id"] for r in rows] == [slot.slot_id, "other"]
        assert rows[0]["handed_off"] is True
        assert rows[1]["handed_off"] is False

    def test_the_dedup_can_report_a_collision(self, mgr):
        """KNOWN-ABSENT PROBE for the dedup: with NO collision present the
        same code must return both rows. A dedup that always returns one row
        would pass the test above for the wrong reason."""
        slot = _slot("m1", fastlane=_match(rank=0))
        mgr._note_handed_off_waiting(slot)
        rows = mgr._waiting_surface_rows(
            [{"slot_id": "a-different-slot", "position": 0, "fastlane": None}])
        assert len(rows) == 2
