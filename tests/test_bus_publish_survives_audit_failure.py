"""A lifecycle event's observability must not
depend on an unrelated database write.

DEFECT: `manager.py`'s `_emit_fastlane_claim_event` runs
    await self._audit_event_only_async(...)
    self.event_bus.publish_nowait({...})
sequentially with NO try/except. If the audit write raises, the publish never
happens and the event is lost from the bus entirely.

REQUIREMENT: lifecycle events must be registered *and* verified
observable on the surface they are read from — several existing events go to the
audit table only and never reach the bus. An event that reaches the bus only
when an unrelated sqlite write succeeds is precisely that failure mode, one
level down.

★ WHY THIS ONE MATTERS: this is the helper that the admission
path routes `fastlane_admitted` through. The admitted event is
losable exactly this way, so the fix belongs in the shared helper rather
than in each caller.

⚠ SCOPE: the helper's docstring rationale is about LOCK
safety ("BOTH off the lock"). That does NOT authorise the failure-coupling —
but the off-lock property must survive the fix. Nothing here acquires a lock.

⚠ NOT ADDRESSED: a HANGING audit write still delays the publish. Fixing
that needs a timeout policy (what value?), which is a config decision, not a
shape choice. Severity is bounded: this helper is always `_spawn_bg`'d, so a
hang parks a background task rather than wedging the manager. Known limitation.
"""
import asyncio
import logging

import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import EventBus, TurbohaulManager
from turbohaul.slot import Slot


class _AuditExploded(RuntimeError):
    """Distinct type so a test can never mistake an unrelated error for the
    injected one."""


@pytest.fixture
def mgr():
    """Deliberately minimal: this slice needs only the bus, the label resolver
    and the audit call, so it is built with __new__ rather than a real boot.

    That is not laziness -- a full TurbohaulManager drags in a real sqlite
    audit DB, and the whole point of these tests is to control precisely
    whether the audit write succeeds or raises. Constructing the real thing
    and then monkeypatching its audit away would give the same coverage with
    more moving parts and a WAL race
    (see the fastlane claims test's fixture comment).
    """
    m = TurbohaulManager.__new__(TurbohaulManager)
    m.event_bus = EventBus()
    m._bg_tasks = set()
    m._fastlane_table = lambda: []  # -> _live_fastlane_label resolves to None
    return m


def _slot():
    s = Slot.new(model_tag="m1", prompt="p")
    s.thread_id = "abcdef1234567890"
    s.fastlane = FastLaneMatch(
        rule_index=0, raw_address="10.0.0.1", label="", effective_tag="main", rank=1,
    )
    return s


def _drain(q: asyncio.Queue):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


@pytest.mark.asyncio
class TestTheBusSurvivesAnAuditFailure:
    async def test_the_event_still_reaches_a_subscriber_when_the_audit_write_raises(self, mgr):
        """RED pre-fix: the audit raises, the publish never runs, the event is
        gone. Asserted by reading a REAL subscriber queue, not by inspecting
        the audit table -- the requirement is about the bus."""
        q: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q)

        async def _boom(*a, **k):
            raise _AuditExploded("audit db unavailable")

        mgr._audit_event_only_async = _boom

        # Pre-fix the exception escapes; suppressing it here is what lets the
        # REAL assertion below be the thing that fails, instead of the test
        # erroring out on the injected exception (a crash is not evidence).
        try:
            await mgr._emit_fastlane_claim_event(_slot(), "fastlane_admitted")
        except _AuditExploded:
            pass

        seen = _drain(q)
        names = {e["event"] for e in seen}
        assert "fastlane_admitted" in names, (
            "The event must be observable on the surface it is read "
            "from. The audit write failed and took the bus publish down with "
            "it, so a subscriber saw NOTHING -- observability that depends on "
            f"an unrelated database write is not observability. Bus carried: "
            f"{sorted(names)}"
        )

    async def test_the_audit_failure_is_surfaced_not_swallowed(self, mgr, caplog):
        """The fix must not trade a lost event for a lost error."""
        q: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q)

        async def _boom(*a, **k):
            raise _AuditExploded("audit db unavailable")

        mgr._audit_event_only_async = _boom

        with caplog.at_level(logging.ERROR, logger="turbohaul.manager"):
            try:
                await mgr._emit_fastlane_claim_event(_slot(), "fastlane_claim_released")
            except _AuditExploded:
                pass

        assert _drain(q), "precondition: the publish must have happened"
        assert any("_AuditExploded" in r.getMessage() or r.exc_info for r in caplog.records), (
            "the audit failure must be LOGGED, not silently swallowed -- a "
            "swap of one invisible failure for another is not a fix"
        )

    async def test_every_lifecycle_event_survives_not_just_admitted(self, mgr):
        """There are four lifecycle events. The fix is in the shared helper, so it must
        hold for all of them -- asserted rather than assumed."""
        for event_type in (
            "fastlane_claim_registered",
            "fastlane_admission_pending",
            "fastlane_admitted",
            "fastlane_claim_released",
        ):
            q: asyncio.Queue = asyncio.Queue()
            mgr.event_bus.subscribe(q)

            async def _boom(*a, **k):
                raise _AuditExploded("audit db unavailable")

            mgr._audit_event_only_async = _boom
            try:
                await mgr._emit_fastlane_claim_event(_slot(), event_type)
            except _AuditExploded:
                pass

            assert event_type in {e["event"] for e in _drain(q)}, (
                f"{event_type} was lost when the audit write failed"
            )


@pytest.mark.asyncio
class TestTheFixDidNotBreakTheHealthyPathOrTheRedaction:
    async def test_a_healthy_audit_still_publishes_exactly_once(self, mgr):
        """GREEN CONTROL. Shares no code path with the failure arm: the audit
        succeeds, so the except branch is never entered."""
        q: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q)
        calls = []

        async def _ok(slot_id, event_type, payload=None):
            calls.append((slot_id, event_type))

        mgr._audit_event_only_async = _ok

        await mgr._emit_fastlane_claim_event(_slot(), "fastlane_admitted")

        assert len(calls) == 1, "the audit write must still be attempted"
        seen = _drain(q)
        assert len(seen) == 1 and seen[0]["event"] == "fastlane_admitted", (
            "a healthy audit must publish exactly one event -- a fix that "
            f"double-published would show here; bus carried {seen}"
        )

    async def test_the_surviving_event_is_still_redacted(self, mgr):
        """The redaction guarantees must hold on the RESCUED event too -- a fix that
        saved the event by publishing a rawer payload would be worse than the
        defect."""
        q: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q)

        async def _boom(*a, **k):
            raise _AuditExploded("audit db unavailable")

        mgr._audit_event_only_async = _boom
        slot = _slot()
        try:
            await mgr._emit_fastlane_claim_event(slot, "fastlane_admitted")
        except _AuditExploded:
            pass

        seen = _drain(q)
        assert seen, "precondition: an event must have been published"
        for e in seen:
            assert e["thread_id_prefix"] == slot.thread_id[:8]
            assert "thread_id" not in e, "full thread_id must never reach the bus"
            assert "client_meta" not in e, "client_meta must never reach the bus"
            assert slot.thread_id not in str(e), (
                "the full thread id must not appear anywhere in the payload"
            )

    async def test_cancellation_is_not_swallowed(self, mgr):
        """A blanket `except Exception` that also ate CancelledError would make
        this helper uncancellable. CancelledError is BaseException on 3.8+, but
        asserting it explicitly is cheap and pins the intent against a future
        edit to `except BaseException`."""
        q: asyncio.Queue = asyncio.Queue()
        mgr.event_bus.subscribe(q)

        async def _cancelled(*a, **k):
            raise asyncio.CancelledError()

        mgr._audit_event_only_async = _cancelled

        with pytest.raises(asyncio.CancelledError):
            await mgr._emit_fastlane_claim_event(_slot(), "fastlane_admitted")

        assert not _drain(q), (
            "a cancelled emit must not publish -- cancellation means the "
            "caller is going away, not that the audit failed"
        )
