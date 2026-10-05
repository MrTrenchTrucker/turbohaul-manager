"""The "admitted" lifecycle step becomes a registered event.

Design requirement --
"New telemetry events (claim registered, admission pending, admitted,
released) must be registered *and* verified observable on the surface
they're read from -- several existing events go to the audit table only and
never reach the bus, and with multiple residents some events currently
vanish entirely."

MEASURED BEFORE BUILDING, against the tree as it stood. Of the four events
listed above, THREE exist
as real emissions and one does not:

    fastlane_claim_registered    manager.py, _register_fastlane_claim_locked
    fastlane_admission_pending   manager.py, _register_fastlane_claim_locked
    admitted                     ✗ NOWHERE -- only ever the string "admitted"
                                   passed as `reason` to the RELEASE event
    fastlane_claim_released      manager.py, three sites

Two fossils in the tree agree, and neither was written for this change:
  * the fast-lane claims test module has a test NAMED
    ``test_all_four_events_reach_a_subscriber`` whose ``expected`` set
    contained exactly THREE names. The missing one is ``admitted``.
    (That set holds four once this change lands: this change is
    what makes the name true, so it re-points the set. The observation
    above is what the tree looked like BEFORE that re-point -- it is the
    evidence for this change, not a description of the current file.)
  * ``_emit_fastlane_claim_event``'s own docstring says it closes the
    audit-vs-bus gap "for these four events specifically".
So the contract was always four; only three were ever built.

THE OTHER HALF: the
lifecycle was ASYMMETRIC. ``fastlane_admission_pending`` was opened at
registration and closed by nothing -- a consumer could only infer the
admission by filtering RELEASE events on a free-text reason field.

⚠ BUS, NOT JUST AUDIT. The requirement is "registered *and*
verified observable on the surface they're read from", and it names the
exact failure mode: events that "go to the audit table only and never reach
the bus". A test that asserts only that the event was recorded would pass
while the defect the requirement actually describes was still present. Every
assertion below therefore reads a real subscriber queue.
"""
import asyncio
import itertools

import pytest

from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot, SlotState
from turbohaul.state import state_db_session

pytestmark = pytest.mark.asyncio

_RULES = [FastLaneRule(address="10.0.0.1", tag_ranks=FastLaneTagRanks(main=1))]
_slot_id_seq = itertools.count()


@pytest.fixture
def mgr(tmp_path):
    """Same shape as the fast-lane claims test's fixture,
    including its state-DB pre-warm: these tests fire concurrent background
    audit writes, and state.py runs `PRAGMA journal_mode=WAL` BEFORE it sets
    `busy_timeout`, so two threads racing the first-ever write on a new file
    hit a transient "database is locked". Pre-warming here avoids importing
    that flake into this suite; it is not a fix to state.py, which is out of
    scope for this change.
    """
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
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False), pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    m = TurbohaulManager(boot, runtime)
    with state_db_session(m.boot.storage.state_db_path):
        pass
    return m


def _match(rule_index=0, rank=1, raw_address="10.0.0.1"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address=raw_address, label="",
        effective_tag="main", rank=rank,
    )


def _slot(model_tag="m1", *, fastlane=None, thread_id=""):
    return Slot(
        slot_id=f"admitted-{next(_slot_id_seq)}",
        model_tag=model_tag,
        state=SlotState.RECEIVED,
        thread_id=thread_id,
        client_meta={},
        fastlane=fastlane,
    )


async def _drain_bg(mgr):
    """_emit_fastlane_claim_event is spawned off-lock; the bus publish has
    not happened until those tasks actually run."""
    tasks = [t for t in list(mgr._bg_tasks) if not t.done()]
    if tasks:
        await asyncio.gather(*tasks)


async def _events_for(mgr, fn):
    """Subscribe, run ``fn``, drain, and return the event names seen on a
    real subscriber queue -- the BUS, which is the surface the design
    requires these be observable on."""
    q: asyncio.Queue = asyncio.Queue()
    mgr.event_bus.subscribe(q)
    fn()
    await _drain_bg(mgr)
    seen = []
    while not q.empty():
        seen.append(q.get_nowait())
    return seen


class TestAdmittedIsARegisteredEvent:
    async def test_admission_puts_an_admitted_event_on_the_bus(self, mgr):
        """THE DEFECT. Pre-fix, admission produced no event of its own --
        only a `fastlane_claim_released` carrying reason="admitted"."""
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        seen = await _events_for(
            mgr, lambda: mgr._release_fastlane_claim_locked(slot, "admitted"))
        names = [e["event"] for e in seen]
        assert "fastlane_admitted" in names, (
            "The design names four lifecycle events and 'admitted' is one of "
            f"them; the bus carried only {names}. An admission that is only a "
            "reason string on a release event is not a registered event."
        )

    async def test_all_four_lifecycle_events_reach_a_subscriber(self, mgr):
        """The whole list of required events, on the bus, in one claim lifecycle."""
        q: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q)
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        mgr._release_fastlane_claim_locked(slot, "admitted")
        await _drain_bg(mgr)
        names = set()
        while not q.empty():
            names.add(q.get_nowait()["event"])
        expected = {
            "fastlane_claim_registered",
            "fastlane_admission_pending",
            "fastlane_admitted",
            "fastlane_claim_released",
        }
        assert expected <= names, f"missing {expected - names}, saw {names}"

    async def test_the_pending_event_is_actually_closed(self, mgr):
        """The lifecycle asymmetry, asserted directly: an admission_pending
        that nothing ever closes is its own half of the defect."""
        q: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q)
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        mgr._release_fastlane_claim_locked(slot, "admitted")
        await _drain_bg(mgr)
        names = []
        while not q.empty():
            names.append(q.get_nowait()["event"])
        # Both presence checks by hand: list.index() raises ValueError on a
        # miss, and a ValueError is a crash, not a statement about ordering.
        assert "fastlane_admission_pending" in names, f"no pending event: {names}"
        assert "fastlane_admitted" in names, (
            "the admission_pending opened at registration is never closed -- "
            f"nothing admits it. bus carried {names}"
        )
        assert names.index("fastlane_admission_pending") < names.index(
            "fastlane_admitted"), (
            "the pending event must be OPENED before the admitted event "
            f"closes it; order was {names}"
        )

    async def test_the_admitted_event_is_redacted_like_its_siblings(self, mgr):
        slot = _slot(fastlane=_match(), thread_id="thread-abcdef123456")
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        seen = await _events_for(
            mgr, lambda: mgr._release_fastlane_claim_locked(slot, "admitted"))
        # Presence asserted by hand first: `next()` on an empty result raises
        # StopIteration, and a test that dies on StopIteration has reported a
        # crash, not a defect.
        matches = [e for e in seen if e["event"] == "fastlane_admitted"]
        assert matches, (
            "no fastlane_admitted event was published, so there is nothing "
            f"whose redaction can be checked; bus carried {[e['event'] for e in seen]}"
        )
        ev = matches[0]
        assert "client_meta" not in ev
        assert "thread_id" not in ev
        assert ev["thread_id_prefix"] == "thread-a"


class TestAdmittedFiresOnlyForARealAdmission:
    """CONTROLS. These share no code path with the emission they guard: each
    drives a DIFFERENT release reason, or no claim at all, and asserts the
    new event is absent. Each of them fails if the event is emitted
    unconditionally."""

    async def test_a_non_admission_release_emits_no_admitted_event(self, mgr):
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        await _drain_bg(mgr)
        seen = await _events_for(
            mgr, lambda: mgr._release_fastlane_claim_locked(slot, "disconnected"))
        names = [e["event"] for e in seen]
        assert "fastlane_admitted" not in names, (
            f"a disconnect is not an admission; bus carried {names}"
        )
        assert "fastlane_claim_released" in names

    async def test_a_slot_that_never_held_a_claim_emits_nothing(self, mgr):
        """Symmetry with fastlane_admission_pending, which also only fires
        for a slot that actually registered a claim: the pair must open and
        close together or a consumer sees a close with no open."""
        slot = _slot(fastlane=_match())
        seen = await _events_for(
            mgr, lambda: mgr._release_fastlane_claim_locked(slot, "admitted"))
        assert [e["event"] for e in seen] == [], (
            "a slot that never registered a claim has nothing to admit"
        )


class TestTheClaimRecordCarriesTheLikelyVictim:
    """Design requirement: the claim record is "who is waiting, who is the
    likely eviction target, since when"."""

    async def test_the_claim_row_names_the_likely_victim(self, mgr):
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        rows = mgr.fastlane_claims_snapshot()
        assert len(rows) == 1
        assert "likely_victim" in rows[0], (
            "The design requires the claim record to carry the likely "
            f"eviction target; the row has only {sorted(rows[0])}"
        )

    async def test_the_field_reports_the_real_resolver_not_a_constant(self, mgr):
        """KNOWN-ABSENT PROBE. `likely_victim` is None with no evictable
        resident present, so a row that said None could be either the real
        answer or a hardcoded stub. This pins it to the resolver by making
        the resolver answer something else and watching the row follow."""
        slot = _slot(fastlane=_match())
        mgr._register_fastlane_claim_locked(slot, "make_room_starved_count_cap")
        row = mgr.fastlane_claims_snapshot()[0]
        assert "likely_victim" in row, (
            "the field does not exist, so whether it tracks the resolver is "
            f"not yet a question; row has {sorted(row)}"
        )
        baseline = row["likely_victim"]
        mgr._likely_unload_target_model_tag = lambda: "a-specific-victim-tag"
        after = mgr.fastlane_claims_snapshot()[0]["likely_victim"]
        assert after == "a-specific-victim-tag", (
            "the row does not track the resolver, so the field is a stub: "
            f"baseline={baseline!r} after={after!r}"
        )
        assert baseline != after, (
            "baseline and mutated value are identical, so this probe proved "
            "nothing -- it cannot tell a live field from a frozen one"
        )
