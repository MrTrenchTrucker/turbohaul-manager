"""manager.submit()'s stream-event arming.

Focused unit-level tier, complementing the end-to-end causal-red proof in
tests/test_api_chat_completion.py::TestResponseFormatContract::
test_e_ollama_stream_flag_does_not_strip_response_format (the existing,
previously-skipped test that hits this bug through the real HTTP route).
This file isolates the exact mechanism at the smallest testable unit: what
manager.submit() itself does to a Slot's stream_ready_event/stream_done_event,
before any worker-loop processing.

THE BUG (fixed, not re-derived here — only its mechanism is stated, for
reference): submit() used to arm the two streaming coordination events
whenever `client_meta["stream"]` was truthy, regardless of which wrapper
method (submit_and_wait vs submit_for_streaming) called it. client_meta is
CLIENT-influenced data; using it alone to control an INTERNAL PROTOCOL
decision (does the manager require the caller's own code to fulfill the
stream_ready_event -> httpx.stream() -> stream_done_event handshake) let
ollama_chat's plain submit_and_wait() calls arm events nothing would ever
resolve, hanging every ollama_chat stream=true request for
manager._STREAM_TIMEOUT_S (3600s) before returning a placeholder result.

THE FIX: submit() now additionally requires an explicit, WRAPPER-controlled
`arm_stream_events` flag. Only submit_for_streaming passes True (it and its
caller-side route are the only code that ever performs the handshake).
submit_and_wait passes nothing (False by default) -- a slot it creates can
never have non-None stream events, so every downstream is_streaming check
(all of which require both events to be non-None) correctly evaluates False
regardless of client_meta["stream"].

CAUSAL BOTH DIRECTIONS:
  * test_arm_stream_events_default_false_never_arms_even_with_stream_true_
    client_meta -- calls the real submit() the exact way submit_and_wait does
    (arm_stream_events omitted). A mutant that defaults arm_stream_events to
    True (or drops the parameter and reverts to the old client_meta-only
    gate) fails this: it would find the events armed.
  * test_arm_stream_events_true_and_client_meta_stream_true_arms_both --
    calls the real submit() the exact way submit_for_streaming does. A mutant
    that always requires arm_stream_events AND ALSO changes the client_meta
    check to something that can never be satisfied (over-tightening) fails
    this: it would find the events still None, breaking the real streaming
    path (the healthy flow that must not be wrongly rejected --
    also directly covered by the still-passing, unmodified
    test_submit_for_streaming_returns_slot_with_armed_events and
    test_a_openai_stream_json_object_in_stream_payload elsewhere in the
    suite).
  * test_arm_stream_events_true_but_client_meta_stream_false_does_not_arm --
    proves the AND is real, not a bypass: wrapper intent alone is not
    sufficient either, matching the pre-existing safety property (a caller
    that forgets to set client_meta["stream"] does not silently start a
    stream it never asked for).

At least one test drives the real, unmocked `submit()` method (all three do)
-- what they construct is a client_meta dict + an arm_stream_events value;
what production DERIVES is the Slot's actual event attributes.
"""
import asyncio

import pytest

from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.manager import TurbohaulManager


# --- fixture (mirror test_cold_restore_tools_fingerprint_guard.py / the
# standard mgr fixture) -- self-contained, does NOT touch anything
# defined in test_api_chat_completion.py (owned elsewhere). submit() enqueues
# onto self.queue and returns without needing a consumer, so no worker loop is
# started or required for what these tests check.
@pytest.fixture
def mgr(tmp_path):
    storage_root = tmp_path / "state"
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir(parents=True)
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.mark.asyncio
async def test_arm_stream_events_default_false_never_arms_even_with_stream_true_client_meta(
        mgr):
    """THE core regression proof, at the smallest unit: this is EXACTLY the
    call shape ollama_chat -> submit_and_wait -> submit() produces
    (client_meta["stream"]=True, arm_stream_events omitted). Both events must
    stay None -- if they don't, _serve_on_resident's is_streaming check would
    take the streaming branch with nothing ever able to set stream_done_event,
    reproducing the hang."""
    slot = await mgr.submit(
        model_tag="test-model",
        prompt="hi",
        thread_id="thr-arming-default",
        client_meta={
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
        },
        wait_for_completion=True,
    )

    assert slot.stream_ready_event is None, (
        "stream_ready_event was armed despite arm_stream_events not being "
        "passed -- this is the exact shape of the hang"
    )
    assert slot.stream_done_event is None, (
        "stream_done_event was armed despite arm_stream_events not being "
        "passed -- this is the exact shape of the hang"
    )


@pytest.mark.asyncio
async def test_arm_stream_events_true_and_client_meta_stream_true_arms_both(
        mgr):
    """THE ONE THAT EARNS ITS KEEP: the real streaming caller's shape
    (submit_for_streaming's own call into submit()) must still arm both
    events -- a fix that closes the hang by breaking real streaming would
    pass the test above while destroying the OpenAI SSE path entirely."""
    slot = await mgr.submit(
        model_tag="test-model",
        prompt="hi",
        thread_id="thr-arming-true",
        client_meta={
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
        },
        wait_for_completion=True,
        arm_stream_events=True,
    )

    assert slot.stream_ready_event is not None
    assert isinstance(slot.stream_ready_event, asyncio.Event)
    assert slot.stream_done_event is not None
    assert isinstance(slot.stream_done_event, asyncio.Event)


@pytest.mark.asyncio
async def test_arm_stream_events_true_but_client_meta_stream_false_does_not_arm(
        mgr):
    """The AND is real: wrapper intent alone (arm_stream_events=True) is not
    sufficient without client_meta also declaring stream=True -- a caller
    that forgets to set the flag does not silently start a stream handshake
    nobody asked for."""
    slot = await mgr.submit(
        model_tag="test-model",
        prompt="hi",
        thread_id="thr-arming-mismatch",
        client_meta={
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
        },
        wait_for_completion=True,
        arm_stream_events=True,
    )

    assert slot.stream_ready_event is None
    assert slot.stream_done_event is None
