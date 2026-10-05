"""`_audit`'s bus publish must survive its DB writes.

DEFECT: `manager.py`'s `_audit` runs

    with state_db_session(...) as conn:  upsert_slot(conn, {...})      # 1st write
    with audit_db_session(...) as conn:  record_audit_event(...)       # 2nd write
    self.event_bus.publish_nowait({...})                               # never reached

sequentially and unguarded. If EITHER write raises, the publish never happens and the event is lost
from the bus. `_audit` is the sibling of the `_emit_fastlane_claim_event` emitter, on the
higher-traffic path: `_audit_async` has 15+ call sites against that emitter's 7.

⚠ TWO WRITES, NOT ONE — the slot upsert and the audit write are independent writes.
The audit write (`audit_db_session`) runs SECOND; the `state_db_session`
slot-row upsert runs FIRST, so guarding only the audit write would leave the more likely killer in
place. Both are in scope. Both are tested below, separately, because one test cannot
attribute two different writes.

=== THE WRITES ARE NOT SWALLOWED; THE PUBLISH IS THE THING THAT IS GUARDED ===
The writes are NOT swallowed. A DB failure still propagates to all 24 `_audit_async` call sites
exactly as it does today — that was the condition for choosing this shape over the more robust
"swallow and continue", which would be an error-propagation POLICY change across the whole system
and would need its own change if wanted.

★ THE SUBTLETY THAT MAKES A BARE `finally` WRONG, and it is easy to miss:
`publish_nowait` is NOT raise-free. `_deliver` swallows only `asyncio.QueueFull` (in manager.py),
while `call_soon_threadsafe` (in manager.py) raises `RuntimeError` on a CLOSED loop. Inside a bare
`finally:` that RuntimeError would REPLACE the in-flight DB exception — destroying the very
propagation property this design preserves, and handing callers a misleading error. So the PUBLISH
is what carries the guard, not the writes. `TestAFailingPublishNeverReplacesTheDbError` is that test.

★ AND THE HAPPY PATH KEEPS ITS CONTRACT TOO: when the writes SUCCEED, a publish failure propagates
exactly as it does today. The guard applies only while an exception is already in flight. Blanket-
swallowing publish errors would be the same kind of quiet policy change this change declines to make for
the DB writes, so it is not made for the publish either.

★ NARROWING MUTANT — THE FIXTURES' EXCEPTION FAMILY IS LOAD-BEARING.
A suite whose fixtures are all `RuntimeError` subclasses would stay green while its evidence
is narrower than its claim: narrowing `_audit`'s outer
`except Exception:` to `except RuntimeError:` would leave every test PASSING while the actual hazard
this suite exists to catch -- a `sqlite3.Error`, which is not a RuntimeError -- would escape unrescued. The
fixture classes at the top of this file are chosen so that no narrowing of EITHER guard can pass;
the comment there is the reasoning, and it is not decoration.
"""
import contextlib
import logging
import queue
import sqlite3
from types import SimpleNamespace

import pytest

import turbohaul.manager as mgr_mod
from turbohaul.manager import EventBus, TurbohaulManager
from turbohaul.slot import Slot


# === WHY THESE CLASSES ARE WHAT THEY ARE =============
# Building all three fixtures on RuntimeError would be insufficient. The
# code could be right and the suite green, but the EVIDENCE would be narrower than
# the claim it backs: narrowing `_audit`'s outer `except Exception:` to
# `except RuntimeError:` would leave all the tests passing, because
# `sqlite3.Error`'s MRO is (Error, Exception, BaseException) -- a real DB
# failure is NOT a RuntimeError. Such a suite would prove the publish survives a
# RuntimeError from a DB write; it would not prove it survives a DB failure.
#
# ⚠ AND SWAPPING RuntimeError -> sqlite3 WOULD ONLY MOVE THE GAP: a guard
# narrowed to `except sqlite3.Error:` would then survive instead. So each
# failure site is exercised with TWO classes whose only common ancestor is
# Exception itself. That pair is what pins the guard AS WRITTEN -- no
# narrowing of it can pass, in either direction.
class _SlotWriteExploded(sqlite3.OperationalError):
    """First write, real DB family -- what the code will actually meet.

    Distinct types per site so a test can never mistake one failure for another.
    """


class _SlotWriteExplodedNotADbError(RuntimeError):
    """First write, non-DB family (a pool/threading failure above sqlite)."""


class _AuditWriteExploded(sqlite3.OperationalError):
    """Second write, real DB family."""


class _AuditWriteExplodedNotADbError(RuntimeError):
    """Second write, non-DB family."""


class _PublishExploded(RuntimeError):
    """★ FAITHFUL AS-IS, and deliberately left a RuntimeError.

    Unlike the DB fixtures, this one already models the hazard the code will
    actually meet: `call_soon_threadsafe` on a closed loop raises a BARE
    `RuntimeError('Event loop is closed')` -- re-derived on a genuinely closed
    loop, not cited from the docs. So the asymmetry is real and it is the
    reason only the DB fixtures need two families.
    """


class _PublishExplodedNotRuntime(queue.Full):
    """The non-RuntimeError escape from `publish_nowait`, and it is a real one.

    `_deliver` catches `asyncio.QueueFull`, which is NOT `queue.Full` -- a
    subscriber holding a threading Queue raises straight through it. That
    needs a caller to violate `subscribe`'s `asyncio.Queue` hint, so this is
    not a live bug; it is what makes the INNER guard's written breadth
    falsifiable, which `_PublishExploded` alone cannot do.
    """


@contextlib.contextmanager
def _ok_session(*a, **k):
    yield object()


def _boom_session(exc):
    @contextlib.contextmanager
    def _cm(*a, **k):
        raise exc
        yield  # pragma: no cover - unreachable, keeps this a generator

    return _cm


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    """Minimal by design: `_audit` needs only `boot.storage.state_db_path` and
    `event_bus`. Both DB sessions are replaced wholesale so each test controls
    exactly which write fails -- a real sqlite would make "the slot write
    failed but the audit write did not" almost impossible to stage."""
    m = TurbohaulManager.__new__(TurbohaulManager)
    m.event_bus = EventBus()
    m.boot = SimpleNamespace(
        storage=SimpleNamespace(state_db_path=tmp_path / "state.sqlite")
    )
    monkeypatch.setattr(mgr_mod, "state_db_session", _ok_session)
    monkeypatch.setattr(mgr_mod, "audit_db_session", _ok_session)
    monkeypatch.setattr(mgr_mod, "upsert_slot", lambda *a, **k: None)
    monkeypatch.setattr(mgr_mod, "record_audit_event", lambda *a, **k: None)
    return m


def _slot():
    s = Slot.new(model_tag="m1", prompt="p")
    s.thread_id = "abcdef1234567890"
    return s


def _drain(bus_q):
    out = []
    while not bus_q.empty():
        out.append(bus_q.get_nowait())
    return out


def _subscribe(m):
    import asyncio

    q: "asyncio.Queue" = asyncio.Queue()
    m.event_bus.subscribe(q)
    return q


class TestTheEventSurvivesADatabaseFailure:
    @pytest.mark.parametrize(
        "exc_cls",
        [_SlotWriteExploded, _SlotWriteExplodedNotADbError],
        ids=["sqlite3.OperationalError", "RuntimeError"],
    )
    def test_the_event_reaches_the_bus_when_the_SLOT_write_fails(
        self, mgr, monkeypatch, exc_cls
    ):
        """The slot-row write, which runs FIRST.

        Both families, because one family alone cannot tell `except Exception:`
        from a narrowing of it -- see the fixture note at the top of this file.
        """
        q = _subscribe(mgr)
        monkeypatch.setattr(
            mgr_mod, "state_db_session", _boom_session(exc_cls("state db down"))
        )

        with pytest.raises(exc_cls):
            mgr._audit(_slot(), "active")

        names = {e["event"] for e in _drain(q)}
        assert "active" in names, (
            "The event must be observable on the surface it is read from. The "
            "state_db_session slot-row write failed and took the bus publish with it, so a "
            f"subscriber saw NOTHING. This is the FIRST of the two writes, failing with "
            f"{exc_cls.__mro__[1].__module__}.{exc_cls.__mro__[1].__name__}. "
            f"Bus carried: {sorted(names)}"
        )

    @pytest.mark.parametrize(
        "exc_cls",
        [_AuditWriteExploded, _AuditWriteExplodedNotADbError],
        ids=["sqlite3.OperationalError", "RuntimeError"],
    )
    def test_the_event_reaches_the_bus_when_the_AUDIT_write_fails(
        self, mgr, monkeypatch, exc_cls
    ):
        """The audit write, which runs SECOND."""
        q = _subscribe(mgr)
        monkeypatch.setattr(
            mgr_mod, "audit_db_session", _boom_session(exc_cls("audit db down"))
        )

        with pytest.raises(exc_cls):
            mgr._audit(_slot(), "grace_enter")

        names = {e["event"] for e in _drain(q)}
        assert "grace_enter" in names, (
            "the audit_db_session write failed and the event was lost from the bus; "
            "observability must not depend on an unrelated database write. Failing with "
            f"{exc_cls.__mro__[1].__module__}.{exc_cls.__mro__[1].__name__}. "
            f"Bus carried: {sorted(names)}"
        )

    def test_the_published_event_is_the_same_shape_on_the_failure_path(self, mgr, monkeypatch):
        """A rescued event must not be a different, rawer event. Redaction and
        fields must match what the healthy path publishes."""
        q_ok = _subscribe(mgr)
        slot = _slot()
        mgr._audit(slot, "active")
        healthy_events = _drain(q_ok)
        assert healthy_events, "fixture precondition: the healthy path must publish"
        healthy = healthy_events[0]

        q_bad = _subscribe(mgr)
        monkeypatch.setattr(
            mgr_mod, "audit_db_session", _boom_session(_AuditWriteExploded("audit db down"))
        )
        with pytest.raises(_AuditWriteExploded):
            mgr._audit(slot, "active")
        # Assert presence BY HAND before indexing: without the guard nothing is published
        # and a bare [0] would fail with IndexError, which is a crash, not
        # evidence of the defect.
        rescued_events = _drain(q_bad)
        assert rescued_events, (
            "the audit write failed and NO event reached the bus, so there is no rescued "
            "event whose shape can be compared -- the publish never happened at all"
        )
        rescued = rescued_events[0]

        assert rescued == healthy, (
            f"the rescued event must be identical to the healthy one; got {rescued} vs {healthy}"
        )
        assert rescued["thread_id_prefix"] == slot.thread_id[:8]
        assert "thread_id" not in rescued
        assert slot.thread_id not in str(rescued), "full thread id must never reach the bus"


class TestPropagationIsPreservedExactly:
    """The whole justification: 24 call sites keep the contract they have."""

    def test_the_slot_write_error_still_propagates(self, mgr, monkeypatch):
        monkeypatch.setattr(
            mgr_mod, "state_db_session", _boom_session(_SlotWriteExploded("state db down"))
        )
        with pytest.raises(_SlotWriteExploded):
            mgr._audit(_slot(), "active")

    def test_the_audit_write_error_still_propagates(self, mgr, monkeypatch):
        monkeypatch.setattr(
            mgr_mod, "audit_db_session", _boom_session(_AuditWriteExploded("audit db down"))
        )
        with pytest.raises(_AuditWriteExploded):
            mgr._audit(_slot(), "active")

    def test_a_failing_slot_write_still_skips_the_audit_write(self, mgr, monkeypatch):
        """The disclosed cost, pinned so nobody mistakes it for an accident: the
        unwind skips the second write, exactly as today. The guard fixes observability,
        NOT write independence -- that would be a separate change."""
        called = []
        monkeypatch.setattr(
            mgr_mod, "state_db_session", _boom_session(_SlotWriteExploded("state db down"))
        )
        monkeypatch.setattr(
            mgr_mod, "record_audit_event", lambda *a, **k: called.append("audit")
        )
        with pytest.raises(_SlotWriteExploded):
            mgr._audit(_slot(), "active")
        assert called == [], (
            "the audit write must still be skipped when the slot write fails -- unchanged "
            "from today; making the writes independent would be a separate change"
        )


class TestAFailingPublishNeverReplacesTheDbError:
    """★ Why the guard sits on the publish: a bare `finally:` would substitute publish_nowait's
    RuntimeError for the DB exception, breaking the property this design exists for."""

    def test_the_db_error_survives_a_publish_that_also_raises(self, mgr, monkeypatch):
        monkeypatch.setattr(
            mgr_mod, "state_db_session", _boom_session(_SlotWriteExploded("state db down"))
        )

        def _boom_publish(_event):
            raise _PublishExploded("event loop is closed")

        monkeypatch.setattr(mgr.event_bus, "publish_nowait", _boom_publish)

        with pytest.raises(_SlotWriteExploded):
            mgr._audit(_slot(), "active")

    def test_the_db_error_survives_a_publish_that_raises_a_NON_RuntimeError(
        self, mgr, monkeypatch
    ):
        """Pins the INNER guard's written breadth (`except Exception:`).

        ⚠ HONEST SCOPE, because this one is NOT the same as the DB-family case: that one
        exposed a hazard the code meets in production (a real sqlite3.Error
        escaping unrescued). This test closes a narrower thing -- the inner
        guard as WRITTEN catches every Exception, and a suite whose only
        publish fixture is a RuntimeError cannot tell it from `except
        RuntimeError:`. `queue.Full` is the one non-RuntimeError escape from
        `_deliver` that is actually identifiable (it is not `asyncio.QueueFull`),
        and it needs a subscriber that violates `subscribe`'s type hint. So:
        the guard is pinned as written, and no live hole is being claimed.
        """
        monkeypatch.setattr(
            mgr_mod, "state_db_session", _boom_session(_SlotWriteExploded("state db down"))
        )

        def _boom_publish(_event):
            raise _PublishExplodedNotRuntime("subscriber queue full")

        monkeypatch.setattr(mgr.event_bus, "publish_nowait", _boom_publish)

        with pytest.raises(_SlotWriteExploded):
            mgr._audit(_slot(), "active")

    def test_the_publish_failure_is_logged_not_swallowed_silently(self, mgr, monkeypatch, caplog):
        monkeypatch.setattr(
            mgr_mod, "state_db_session", _boom_session(_SlotWriteExploded("state db down"))
        )

        def _boom_publish(_event):
            raise _PublishExploded("event loop is closed")

        monkeypatch.setattr(mgr.event_bus, "publish_nowait", _boom_publish)

        with caplog.at_level(logging.ERROR, logger="turbohaul.manager"):
            with pytest.raises(_SlotWriteExploded):
                mgr._audit(_slot(), "active")

        assert any(r.exc_info or "_PublishExploded" in r.getMessage() for r in caplog.records), (
            "swallowing the publish error to protect the DB error must not make the publish "
            "failure invisible -- one hidden failure traded for another is not a fix"
        )


class TestTheHealthyPathIsUnchanged:
    """Green controls. These share no code path with the failure arm: no write
    raises, so the guard is never entered."""

    def test_a_healthy_call_publishes_exactly_once_and_does_not_raise(self, mgr):
        q = _subscribe(mgr)
        mgr._audit(_slot(), "active")
        seen = _drain(q)
        assert len(seen) == 1 and seen[0]["event"] == "active", (
            f"a healthy call must publish exactly one event; bus carried {seen}"
        )

    def test_both_writes_still_happen_on_the_healthy_path(self, mgr, monkeypatch):
        calls = []
        monkeypatch.setattr(mgr_mod, "upsert_slot", lambda *a, **k: calls.append("slot"))
        monkeypatch.setattr(mgr_mod, "record_audit_event", lambda *a, **k: calls.append("audit"))
        mgr._audit(_slot(), "active")
        assert calls == ["slot", "audit"], (
            f"both writes must still run, in order, on the healthy path; got {calls}"
        )

    def test_a_publish_failure_on_the_HAPPY_path_still_propagates(self, mgr, monkeypatch):
        """The guard must apply ONLY while an exception is already in flight.
        Blanket-swallowing publish errors would be the same quiet policy change
        this change declines to make for the DB writes -- so it is not made here."""

        def _boom_publish(_event):
            raise _PublishExploded("event loop is closed")

        monkeypatch.setattr(mgr.event_bus, "publish_nowait", _boom_publish)

        with pytest.raises(_PublishExploded):
            mgr._audit(_slot(), "active")
