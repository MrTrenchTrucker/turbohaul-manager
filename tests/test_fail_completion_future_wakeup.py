"""Gate the streaming wakeup on a placement failure (manager._fail_completion_future).

WHY THIS FILE EXISTS. A placement failure is set on a
completion future with no awaiter, so without a wakeup a STREAMING client waits on
`stream_ready_event` for the full SLOT_READY_TIMEOUT_S (7200s) instead of
failing fast. The failure path therefore force-sets the event, with a reason
(`slot.py` documents that intent).

That wakeup was completely UNGATED before this file existed:
deleting its three lines left every test that
mentions `_fail_completion_future` or `stream_ready_*` green, so
nothing in the suite noticed. This file closes that gap.

WHY EXISTING TESTS MISSED IT — a trap, not an oversight. Three sites re-implement the
fix inside their own fixtures and then assert the behaviour they just performed:
`test_api_chat_completion.py` (one site) and `test_api_embeddings.py` (two sites).
Each replaces `mgr.submit_for_streaming` wholesale
with a fake, so the real manager never runs at all.

Those three are NOT broken and are deliberately left alone. Their subject is a
genuine ROUTE-TIER property — given a slot already woken with a failure reason
and `stream_handle is None`, does the route emit 503 `capacity_unavailable` with
`retry_after=7`. The stamping there is a fixture PRECONDITION, not the thing
under test. (One of the embeddings sites already words it exactly that way, and the
other two are worded to match.) Rewriting them to drive the real manager would destroy
real route coverage and couple two deliberately separated tiers.

The defect-reproduction test is a fourth near-miss worth naming: it DOES take the
real method (`real_fail = mgr._fail_completion_future`) — but its `_miss()` helper
builds a `Slot` that never assigns `stream_ready_event`, so it stays the
dataclass default `None` and the `is not None` guard makes the block a
structural no-op there. That file contains zero occurrences of "stream_ready".

So this file is the manager tier: it drives the REAL bound method on a REAL
`Slot` with a REAL `asyncio.Event`, and asserts the wakeup itself. One test per
condition in the three-line block, because the two guards are exactly what a
future "simplification" would drop:

        if slot.stream_ready_event is not None      -> test 3
        and not slot.stream_ready_event.is_set():   -> test 2
            slot.stream_ready_failed_reason = ...    -> test 1
            slot.stream_ready_event.set()            -> test 1

★ EVERY TEST HERE CARRIES A POSITIVE HALF. Each asserts that the completion
future actually received the exception — which lives in the UNTOUCHED (a) branch
ahead of the block and therefore SURVIVES deleting it. That is deliberate: it means a
RED here can only ever be "the wakeup is missing", never "the fixture never got
there". A test whose RED is ambiguous is not a control.
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
from turbohaul.slot import Slot, SlotState, VramOverCommitError


# --- fixtures (mirror test_tools_fingerprint_wiring.py) ----------------------
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
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


def _streaming_slot(tid="hermes-main-stream"):
    """A REAL Slot as the streaming path builds one: a real asyncio.Event and a
    real pending Future. Nothing here is a MagicMock and nothing pre-stamps what
    the assertions check -- that re-implementation is the defect this test gates."""
    return Slot(
        slot_id="stream-fail",
        model_tag="qwen-test",
        state=SlotState.RECEIVED,
        prompt="p",
        thread_id=tid,
    )


_EXC_MSG = "no VRAM capacity"


@pytest.mark.asyncio
async def test_placement_failure_wakes_a_streaming_slot_with_the_reason(mgr):
    """THE CONTROL for this gate. Deleting the wakeup block in _fail_completion_future must fail THIS.

    A placement failure on a streaming slot must wake `stream_ready_event` and
    stamp the reason -- otherwise the route's ready-wait blocks for the full
    7200s deadline and the client hangs, which is the original hang exactly.
    """
    slot = _streaming_slot()
    slot.stream_ready_event = asyncio.Event()
    slot.completion_future = asyncio.get_running_loop().create_future()
    exc = VramOverCommitError(_EXC_MSG, retry_after_s=7)

    mgr._fail_completion_future(slot, exc)

    # --- POSITIVE HALF: proof the real method ran and reached its body. This
    # lives in the (a) branch ahead of the wakeup block, which the deletion does NOT touch, so it
    # stays GREEN under the mutation and the RED below can only mean the wakeup
    # is missing -- never "the fixture never got there".
    assert slot.completion_future.done(), (
        "_fail_completion_future did not run at all -- this RED would be a "
        "harness problem, not the defect under test")
    assert slot.completion_future.exception() is exc

    # --- THE GATE: the reason stamp and the wakeup.
    assert slot.stream_ready_event.is_set(), (
        "manager.py wakeup -- a placement failure must WAKE the streaming "
        "ready-wait; without it the client blocks for SLOT_READY_TIMEOUT_S "
        "(7200s). This is the original hang.")
    assert slot.stream_ready_failed_reason == str(exc), (
        f"manager.py reason stamp -- the reason must be stamped so the route can report "
        f"the real cause instead of its generic fallback; got "
        f"{slot.stream_ready_failed_reason!r}")


@pytest.mark.asyncio
async def test_failure_after_a_genuine_ready_does_not_overwrite_the_reason(mgr):
    """Gates the `not slot.stream_ready_event.is_set()` half of the wakeup condition.

    A slot that was legitimately readied has its event already set and
    `stream_ready_failed_reason` still None. A failure arriving afterwards must
    NOT stamp a reason: nobody will read it, and a genuinely-served stream must
    never acquire a failure reason retroactively. The manager docstring calls
    this out ("so a failure arriving after a slot was already legitimately
    readied cannot overwrite a reason nobody will read") and nothing enforced it.
    """
    slot = _streaming_slot(tid="hermes-main-already-ready")
    slot.stream_ready_event = asyncio.Event()
    slot.stream_ready_event.set()          # genuinely ready, BEFORE the failure
    slot.completion_future = asyncio.get_running_loop().create_future()
    exc = VramOverCommitError(_EXC_MSG, retry_after_s=7)

    mgr._fail_completion_future(slot, exc)

    # POSITIVE HALF -- the method really did run on this slot.
    assert slot.completion_future.done()
    assert slot.completion_future.exception() is exc

    assert slot.stream_ready_failed_reason is None, (
        "manager.py wakeup guard -- the `not ...is_set()` guard must stop a late "
        "failure from stamping a reason onto an already-ready slot; got "
        f"{slot.stream_ready_failed_reason!r}")
    assert slot.stream_ready_event.is_set(), "the event must remain set"


@pytest.mark.asyncio
async def test_non_streaming_slot_without_the_event_is_not_a_crash(mgr):
    """Gates the `is not None` half of the wakeup condition.

    Non-streaming slots never arm `stream_ready_event` (see
    test_stream_event_arming.py -- `submit()` only arms it when asked), so the
    field is the dataclass default None. The failure path must still complete
    normally rather than raising AttributeError on `.is_set()`.
    """
    slot = _streaming_slot(tid="hermes-main-nonstream")
    assert slot.stream_ready_event is None, "precondition: unarmed slot"
    slot.completion_future = asyncio.get_running_loop().create_future()
    exc = VramOverCommitError(_EXC_MSG, retry_after_s=7)

    mgr._fail_completion_future(slot, exc)   # must not raise

    # POSITIVE HALF -- and the ordinary failure delivery still happened.
    assert slot.completion_future.done()
    assert slot.completion_future.exception() is exc
    assert slot.stream_ready_failed_reason is None
