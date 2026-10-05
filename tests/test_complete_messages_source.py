"""SUBSTITUTION regression test for make_llama_server_complete_fn's _complete:
the outbound payload's ``messages`` must come from ``slot.context`` (the
request-scoped field, set once at admission and never reassigned), never from
``slot.client_meta`` (a mutable field a same-model idle-hot warm-inherit can
replace with a different caller's identity + content).

This reproduces the bug directly against the real completion_fn, not a fake:
pre-fix, _complete read messages off client_meta and would have forwarded the
STALE content below; post-fix it must forward the slot's own context.
"""
import asyncio
from unittest.mock import MagicMock

from turbohaul.api.chat_completion import make_llama_server_complete_fn
from turbohaul.slot import Slot


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, captured_payloads):
        self._captured = captured_payloads

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None, timeout=None):
        self._captured.append(json)
        return _FakeResponse({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        })


def _fake_http_client_factory(captured_payloads):
    def factory():
        return _FakeAsyncClient(captured_payloads)
    return factory


def test_complete_reads_messages_from_slot_context_not_client_meta():
    stale_inherited_messages = [{"role": "user", "content": "STALE_INHERITED_PROMPT"}]
    real_request_messages = [{"role": "user", "content": "REAL_REQUEST_PROMPT"}]

    captured = []
    complete_fn = make_llama_server_complete_fn(
        http_client_factory=_fake_http_client_factory(captured)
    )
    # Shape a slot the way a session-less warm-inherit COULD produce pre-fix:
    # client_meta carries a different (stale) caller's messages, but
    # slot.context — set once at THIS request's admission — carries its own.
    slot = Slot.new(
        model_tag="m",
        context=real_request_messages,
        client_meta={"messages": stale_inherited_messages, "session_id": None},
    )
    handle = MagicMock(port=1234)

    asyncio.run(complete_fn(slot, handle))

    assert len(captured) == 1
    assert captured[0]["messages"] == real_request_messages
    assert captured[0]["messages"] != stale_inherited_messages


def test_complete_returns_none_when_context_empty():
    """No fallback to client_meta when context is unset (e.g. a caller that
    never passed context, like embeddings.py's submit_for_streaming today) —
    must degrade to a no-op, never silently substitute a different field."""
    captured = []
    complete_fn = make_llama_server_complete_fn(
        http_client_factory=_fake_http_client_factory(captured)
    )
    slot = Slot.new(
        model_tag="m",
        context=None,
        client_meta={"messages": [{"role": "user", "content": "would-be-wrong-source"}]},
    )
    handle = MagicMock(port=1234)

    result = asyncio.run(complete_fn(slot, handle))

    assert result is None
    assert captured == []
