"""The same-thread follow-up pop must see the resident's inbox.

A request routed to a loaded model's inbox never sits in staging. When a
worse-ranked same-thread follow-up F waits in staging and a better-ranked
request B of the SAME client (same ``rule_index``, strictly lower ``rank``)
waits in that model's inbox, ``pop_matched_thread`` given the inbox must not
serve F ahead of B: it declines (returns None), F stays in staging, and the
inbox is put back with the same objects in the same order.

Rank is never compared across clients, equal rank is not better, and a call
that passes no inbox keeps the old behaviour. The inbox is restored on every
exit, including an exception raised while matching.

Every test asserts WHICH request is returned (identity) and the buffer
contents and ORDER afterwards.
Free VRAM: none of these tests reaches a code path that reads free VRAM (the
queue methods and a source-shape check only), so no free-VRAM read is
reachable here; the binding is pinned by an autouse fixture anyway.
"""

import asyncio

import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.queue import TurbohaulQueue
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


def _slot(label, rank, *, rule_index=0, listed=True, thread_id=None, model=MODEL):
    s = Slot.new(
        model_tag=model, prompt=label,
        thread_id=thread_id if thread_id is not None else f"thread-{label}",
        client_meta={"ip": IP},
    )
    if listed:
        s.fastlane = FastLaneMatch(
            rule_index=rule_index, raw_address=IP, label="",
            effective_tag=f"tag{rank}", rank=rank,
        )
    return s


def _follow(rank, **kw):
    return _slot("follow", rank, thread_id=FOLLOW_THREAD, **kw)


async def _queue_with(*slots):
    q = TurbohaulQueue(staging_max=10, acceptance_max=10)
    for s in slots:
        await q.enqueue(s)
    return q


def _inbox_of(*slots):
    inbox: "asyncio.Queue" = asyncio.Queue()
    for s in slots:
        inbox.put_nowait(s)
    return inbox


def _inbox_list(inbox):
    """Non-destructive view through the public API only (drain and refill)."""
    out = []
    while not inbox.empty():
        out.append(inbox.get_nowait())
    for s in out:
        inbox.put_nowait(s)
    return out


def _d(s):
    return "None" if s is None else s.prompt


@pytest.mark.asyncio
class TestThreadPopSeesInbox:
    async def test_follow_up_yields_to_better_ranked_same_client_inbox_request(self):
        """(1) F (rank 3) in staging, B (rank 1, same client) in the inbox among
        others: the pop declines, F stays, the inbox is the same objects in the
        same order."""
        follow = _follow(3)
        before = _slot("before", 5, listed=False)
        better = _slot("better", 1)
        after = _slot("after", 4)
        q = await _queue_with(follow)
        inbox = _inbox_of(before, better, after)

        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL, inbox=inbox)

        assert got is None, f"follow-up served ahead of a better inbox request: {_d(got)}"
        assert list(q._staging) == [follow], (
            f"declined follow-up moved: {[_d(s) for s in q._staging]}")
        contents = _inbox_list(inbox)
        assert contents == [before, better, after], (
            f"inbox changed: {[_d(s) for s in contents]}")
        assert all(a is b for a, b in zip(contents, [before, better, after]))

    async def test_control_equal_rank_inbox_request_does_not_hold_follow_up(self):
        """(2) B has the same rank as F: not better, F is returned (old
        behaviour); the inbox is untouched."""
        follow = _follow(3)
        peer = _slot("peer", 3)
        other = _slot("other", 5)
        q = await _queue_with(follow)
        inbox = _inbox_of(peer, other)

        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL, inbox=inbox)

        assert got is follow, f"returned {_d(got)}"
        assert len(q._staging) == 0, "served follow-up is still in staging"
        contents = _inbox_list(inbox)
        assert contents == [peer, other], f"inbox changed: {[_d(s) for s in contents]}"

    async def test_control_better_rank_of_another_client_does_not_hold_follow_up(self):
        """(3) B is a different client (different rule_index) with a better
        rank: rank is never compared across clients, F is returned."""
        follow = _follow(3, rule_index=0)
        other_client = _slot("other-client", 1, rule_index=1)
        q = await _queue_with(follow)
        inbox = _inbox_of(other_client)

        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL, inbox=inbox)

        assert got is follow, f"returned {_d(got)}"
        contents = _inbox_list(inbox)
        assert contents == [other_client], f"inbox changed: {[_d(s) for s in contents]}"

    async def test_no_inbox_argument_keeps_old_behaviour(self):
        """(4) Without the inbox the pop cannot see B and serves F; the same
        slots with the inbox declined in test (1), so this pair differs only by
        the argument. The positional signature still works."""
        follow = _follow(3)
        better = _slot("better", 1)
        q = await _queue_with(follow)
        inbox = _inbox_of(better)

        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL)

        assert got is follow, f"returned {_d(got)}"
        assert _inbox_list(inbox) == [better]

        follow2 = _follow(3)
        q2 = await _queue_with(follow2)
        got2 = await q2.pop_matched_thread(FOLLOW_THREAD, MODEL, inbox=None)
        assert got2 is follow2, f"inbox=None returned {_d(got2)}"

    async def test_inbox_is_complete_and_ordered_after_an_exception(self):
        """(5) The matching step raises once: the exception reaches the caller
        and the inbox still holds every slot in the original order; the queue
        lock is free and the next call works."""
        follow = _follow(3)
        a = _slot("a", 2)
        b = _slot("b", 1)
        c = _slot("c", 4, listed=False)
        q = await _queue_with(follow)
        inbox = _inbox_of(a, b, c)
        real = q._pop_first_matching_locked
        calls = []

        def boom(*args, **kwargs):
            calls.append(kwargs.get("also_waiting"))
            raise RuntimeError("injected matching failure")

        q._pop_first_matching_locked = boom
        try:
            raised = None
            try:
                await q.pop_matched_thread(FOLLOW_THREAD, MODEL, inbox=inbox)
            except RuntimeError as exc:
                raised = exc
        finally:
            q._pop_first_matching_locked = real

        assert raised is not None, "the injected failure did not reach the caller"
        assert len(calls) == 1 and calls[0] is not None and len(calls[0]) == 3, (
            f"the matcher was not handed the inbox contents: {calls}")
        contents = _inbox_list(inbox)
        assert contents == [a, b, c], f"inbox after exception: {[_d(s) for s in contents]}"
        assert list(q._staging) == [follow]
        assert not q._lock.locked(), "the queue lock was left held"
        again = await asyncio.wait_for(
            q.pop_matched_thread(FOLLOW_THREAD, MODEL, inbox=inbox), timeout=5)
        assert again is None, f"the better inbox request no longer holds F: {_d(again)}"
        assert _inbox_list(inbox) == [a, b, c]

    async def test_empty_inbox_and_none_inbox_serve_the_follow_up(self):
        """Edge: an empty inbox is the same as none."""
        follow = _follow(3)
        q = await _queue_with(follow)
        inbox = _inbox_of()
        got = await q.pop_matched_thread(FOLLOW_THREAD, MODEL, inbox=inbox)
        assert got is follow, f"returned {_d(got)}"
        assert inbox.empty()


# ---------------------------------------------------------------------------
# wiring: the grace loop hands the resident's inbox to the pop
# ---------------------------------------------------------------------------

import ast  # noqa: E402
import inspect  # noqa: E402
import textwrap  # noqa: E402

from turbohaul.manager import TurbohaulManager  # noqa: E402


def _pop_calls(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [
        c for c in ast.walk(tree)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "pop_matched_thread"
    ]


class TestGraceLoopWiring:
    def test_grace_loop_passes_the_resident_inbox_to_the_thread_pop(self):
        """WIRING check, not behaviour: it reads the source of the per-resident
        grace loop (``_serve_on_resident``) and asserts the shape of its
        ``pop_matched_thread`` calls. What the pop DOES with an inbox is covered
        by the queue-level tests above; this test only guards that the grace
        loop hands the resident's inbox over at all.

        Expected shape: one call that passes the keyword ``inbox`` with the
        resident's inbox attribute (not a literal), and one plain two-argument
        call for the empty-inbox shape."""
        calls = _pop_calls(TurbohaulManager._serve_on_resident)
        assert calls, "no pop_matched_thread call found in _serve_on_resident"
        with_inbox = [c for c in calls if any(k.arg == "inbox" for k in c.keywords)]
        assert with_inbox, (
            "the grace loop never passes the keyword 'inbox' to pop_matched_thread "
            f"(calls found: {[[k.arg for k in c.keywords] for c in calls]})")
        for c in with_inbox:
            value = next(k.value for k in c.keywords if k.arg == "inbox")
            assert isinstance(value, ast.Attribute) and value.attr == "inbox", (
                "the keyword 'inbox' is not given the resident's inbox attribute: "
                f"{ast.unparse(value)!r}")
        plain = [c for c in calls if not c.keywords and len(c.args) == 2]
        assert plain, (
            "no plain two-argument pop_matched_thread call for the empty-inbox "
            f"shape (calls found: {[ast.unparse(c) for c in calls]})")
