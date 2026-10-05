"""The tools-change warning must reach EVERY per-turn path.

WHY THE CHECK IS NOT INSIDE THE KV PROBE. The tools check sits above
`_probe_and_save_clean_kv`'s two early returns, so a turn whose clean save is
declined by the single-series gate still warns. But if the check lived INSIDE
that method it would be missed, since `_serve_on_resident` branches three ways with only the
non-streaming `else` arm calling it:

    if n_parallel > 1:   -> _fan_out_on_resident      NO probe call
    elif is_streaming:   -> stream wait               NO probe call
    else:                -> _probe_and_save_clean_kv  the only arm that calls it

So a STREAMING turn served on a warm resident — the ordinary chat shape, and the
path where tool-call complaints surface — would never reach the check at
all. Hoisting cannot fix that: you cannot hoist out of a method that is never
called.

HOW IT IS REACHED. The check lives in `_check_thread_tools_fingerprint`, a
pure zero-I/O method, reached from the PER-TURN seam instead of the
KV-save path: `_mark_slot_active` (5 sites) plus one direct call at
`_serve_on_resident`'s matched promotion, which reaches no `_mark_slot_active`
of its own.

WHY NOT JUST CALL THE PROBE FROM THE TWO MISSING ARMS. That is the obvious
alternative and it is the HIGH-disturbance one, not the low one:
`_probe_and_save_clean_kv` performs real I/O — a direct sidecar prefill POST
(`{"n_predict": 0}`) and a multi-GB `_save_slot_kv` — and the streaming path is
the one place the file explicitly documents as owning its upstream connection
("Worker MUST NOT ... open a 2nd sidecar connection and violate
the single-slot invariant"). Adding a probe there is a production
behaviour change on the hottest path, not an observability fix.

EVERY TEST HERE CARRIES A POSITIVE HALF — an assertion that the path under test
was ACTUALLY taken — so a RED can only ever mean "the warning is missing", never
"the fixture never got there".
"""
from __future__ import annotations

import asyncio

import pytest

import turbohaul.subprocess_mgr as subprocess_mgr
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
from turbohaul.manager import Resident, TurbohaulManager
from turbohaul.queue import GraceTimer
from turbohaul.slot import Slot, SlotState


_QWEN = "qwen3.6-27b"
_PID = 4242


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
            default_port_base=60400,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    return TurbohaulManager(boot, runtime)


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


class _FakeHandle:
    def __init__(self, pid=_PID, parallel=1):
        self.pid = pid
        self.parallel = parallel
        self.port = 60400


def _resident(mgr, handle):
    """A Resident with a real GraceTimer. `grace_seconds` is forced to 0 so
    _serve_on_resident's grace loop exits immediately -- the grace
    window is not what these tests are about, and leaving it open would make
    them wait on wall-clock."""
    mgr.runtime.queue.grace_seconds = 0
    return Resident(
        model_tag=_QWEN,
        handle=handle,
        grace=GraceTimer(grace_seconds=0, max_extensions=0),
    )


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


_BASE = [_tool("a"), _tool("b")]
_GROWN = _BASE + [_tool(f"extra_tool_{i}") for i in range(3)]


def _streaming_slot(tid, tools):
    """A slot shaped exactly as _serve_on_resident's `is_streaming` test
    requires: client_meta['stream'] true AND both events present."""
    slot = Slot.new(model_tag=_QWEN, thread_id=tid)
    slot.state = SlotState.LOADING           # legal predecessor of ACTIVE
    slot.stream_ready_event = asyncio.Event()
    slot.stream_done_event = asyncio.Event()
    slot.stream_done_event.set()             # the route already drained
    slot.client_meta = {"stream": True, "tools": tools}
    slot.completion_future = asyncio.get_running_loop().create_future()
    return slot


def _warnings(caplog):
    return [r.message for r in caplog.records if "TOOLS CHANGED" in r.message]


@pytest.mark.asyncio
async def test_streaming_turn_on_a_warm_resident_warns_on_a_tools_change(
        mgr, kv_dir, caplog):
    """THE CONTROL for this check. RED if the per-turn check is missing, because the streaming arm of
    _serve_on_resident never reaches the check.

    Drives the REAL _serve_on_resident twice on one thread with the tools set
    growing 2 -> 5, through its genuine `elif is_streaming` arm.
    """
    caplog.set_level("WARNING", logger="turbohaul.manager")
    tid = "hermes-main-streaming-thread"
    handle = _FakeHandle()

    # POSITIVE HALF: prove the STREAMING arm is the one taken, by making the
    # non-streaming arm impossible to take silently -- if _serve_on_resident
    # ever falls through to the `else`, it calls the KV probe, and this spy
    # fires. A RED below therefore cannot mean "the fixture took another path".
    probed: list = []

    async def _spy_probe(h, s, **kw):
        probed.append(s)

    mgr._probe_and_save_clean_kv = _spy_probe

    r = _resident(mgr, handle)
    await mgr._serve_on_resident(r, _streaming_slot(tid, _BASE), handle)
    assert _warnings(caplog) == [], (
        "the FIRST turn on a thread must never warn -- nothing to diverge from")

    caplog.clear()
    await mgr._serve_on_resident(r, _streaming_slot(tid, _GROWN), handle)

    assert probed == [], (
        "POSITIVE HALF FAILED: _serve_on_resident called the KV probe, so it "
        "took the non-streaming `else` arm and this test is not exercising the "
        "streaming path it claims to")

    warnings = _warnings(caplog)
    assert len(warnings) == 1, (
        "a streaming turn on a warm resident must produce exactly one TOOLS "
        f"CHANGED warning -- this is the ordinary chat shape: {warnings}")
    msg = warnings[0]
    assert "prev_count=2" in msg and "cur_count=5" in msg, msg
    assert "extra_tool_0" in msg, msg


@pytest.mark.asyncio
async def test_one_turn_produces_exactly_one_warning(mgr, kv_dir, caplog):
    """Once-per-turn, the single-turn shape: the seam must not double-fire.

    Guards against the check being reached twice for one slot -- e.g. if a
    future change added a second call alongside _mark_slot_active.
    """
    caplog.set_level("WARNING", logger="turbohaul.manager")
    tid = "hermes-main-once-thread"
    handle = _FakeHandle()

    probed: list = []

    async def _spy_probe(h, s, **kw):
        probed.append(s)

    mgr._probe_and_save_clean_kv = _spy_probe

    r = _resident(mgr, handle)
    await mgr._serve_on_resident(r, _streaming_slot(tid, _BASE), handle)
    caplog.clear()
    await mgr._serve_on_resident(r, _streaming_slot(tid, _GROWN), handle)

    assert probed == [], "POSITIVE HALF: must still be on the streaming arm"
    assert len(_warnings(caplog)) == 1, (
        f"ONE turn must warn exactly ONCE, not twice: {_warnings(caplog)}")


@pytest.mark.asyncio
async def test_grace_loop_shape_two_turns_produce_exactly_two_warnings(
        mgr, kv_dir, caplog):
    """The GRACE-LOOP shape, tested explicitly: the anchor mark then the follow-up mark inside
    ONE _process_slot call.

    Those two sites are NOT mutually exclusive -- the second lives inside the grace
    `while` loop -- but that loop does `matched = await
    queue.pop_matched_thread(...)`, so `matched` is a SEPARATE request off the
    queue: the NEXT TURN on the same thread. Two marks therefore mean two TURNS,
    and the correct answer is TWO warnings when the tools change between them --
    not one (which would mean a turn was missed) and not three (double-firing).

    Driven at the seam with two DISTINCT slots on one thread, which is exactly
    what those two sites pass.
    """
    caplog.set_level("WARNING", logger="turbohaul.manager")
    tid = "hermes-main-grace-thread"

    anchor = Slot.new(model_tag=_QWEN, thread_id=tid)
    anchor.client_meta = {"tools": _BASE}
    followup = Slot.new(model_tag=_QWEN, thread_id=tid)
    followup.client_meta = {"tools": _GROWN}
    third = Slot.new(model_tag=_QWEN, thread_id=tid)
    third.client_meta = {"tools": _BASE}

    # POSITIVE HALF: distinct objects, one thread -- i.e. genuinely the
    # anchor/matched shape and not the same slot marked twice.
    assert anchor is not followup and anchor.thread_id == followup.thread_id

    mgr._mark_slot_active(anchor)      # turn N   (anchor)
    mgr._mark_slot_active(followup)    # turn N+1 (follow-up, tools changed -> warn)
    mgr._mark_slot_active(third)       # turn N+2 (changed back      -> warn)

    warnings = _warnings(caplog)
    assert len(warnings) == 2, (
        "three turns with tools 2->5->2 must warn exactly twice: once per "
        f"CHANGE, one warning per turn, never one turn counted twice: {warnings}")
    assert "prev_count=2" in warnings[0] and "cur_count=5" in warnings[0], warnings[0]
    assert "prev_count=5" in warnings[1] and "cur_count=2" in warnings[1], warnings[1]


@pytest.mark.asyncio
async def test_seam_path_never_reaches_the_check(mgr, kv_dir, caplog):
    """The save_to_disk=False flag is a PLACEMENT property.

    The unload/seam path can carry a carried-forward last_main_client_meta
    rather than this turn's own sample, so it must not be fingerprinted. That
    is enforced by the seam path never marking a slot ACTIVE, not by a
    `save_to_disk` check. This asserts the property directly rather
    than trusting the placement.
    """
    reached: list = []
    mgr._check_thread_tools_fingerprint = lambda s: reached.append(s)

    slot = Slot.new(model_tag=_QWEN, thread_id="hermes-main-seam")
    slot.client_meta = {"tools": _BASE}
    handle = _FakeHandle()

    await mgr._probe_and_save_clean_kv(handle, slot, save_to_disk=True,
                                       _seam_tag="unload")

    assert reached == [], (
        "the unload/seam path reached the tools-change check -- it must not: its "
        f"client_meta is not a trustworthy sample of this turn: {reached}")


@pytest.mark.asyncio
async def test_grace_matched_promotion_on_a_resident_warns(mgr, kv_dir, caplog):
    """THE SIXTH CALL SITE — _serve_on_resident's matched promotion.

    This is the one site NOT reached via _mark_slot_active: the grace loop
    promotes a follow-up popped by `queue.pop_matched_thread` straight to
    ACTIVE without ever marking it active, unlike _process_slot's
    equivalent. Without the direct call there, this turn is invisible.

    The follow-up here is STREAMING on purpose. That arm never calls
    the KV probe either, so without the direct call it would have no tools check by ANY route —
    placing the call ABOVE the m_stream/else split covers both arms with one
    line, which is why it sits at the promotion rather than in the two branches.
    """
    caplog.set_level("WARNING", logger="turbohaul.manager")
    tid = "hermes-main-matched-thread"
    handle = _FakeHandle()

    mgr.runtime.queue.grace_seconds = 30          # window must stay open
    r = Resident(model_tag=_QWEN, handle=handle,
                 grace=GraceTimer(grace_seconds=30, max_extensions=0))

    followup = _streaming_slot(tid, _GROWN)
    followup.state = SlotState.STAGED             # legal predecessor of ACTIVE_MATCH
    pending = [followup]

    async def _pop(thread_id, model_tag):
        if pending:
            return pending.pop(0)
        mgr._stop_event.set()                     # end the grace loop deterministically
        return None

    mgr.queue.pop_matched_thread = _pop

    probed: list = []

    async def _spy_probe(h, s, **kw):
        probed.append(s)

    mgr._probe_and_save_clean_kv = _spy_probe

    await mgr._serve_on_resident(r, _streaming_slot(tid, _BASE), handle)

    # POSITIVE HALF: the matched follow-up really was promoted and served.
    assert pending == [], "the grace loop never popped the follow-up"
    # ...and -- the point of this whole site -- it was promoted WITHOUT ever
    # being marked active, so _mark_slot_active's five sites could not have
    # covered it. started_active_at is _mark_slot_active's own stamp; if a
    # future change routes this path through it, this assertion fires and the
    # sixth call site can then be removed deliberately rather than by accident.
    assert followup.started_active_at == 0.0, (
        "the follow-up went through _mark_slot_active after all -- the sixth "
        "call site may now be redundant, and a double-warn is possible: "
        f"started_active_at={followup.started_active_at}")
    assert probed == [], (
        "POSITIVE HALF: both turns must be on STREAMING arms, which never call "
        f"the KV probe -- probe was called with {probed}")

    warnings = _warnings(caplog)
    assert len(warnings) == 1, (
        "the grace-window follow-up is its own TURN and must warn when its "
        f"tools differ from the anchor's: {warnings}")
    assert "prev_count=2" in warnings[0] and "cur_count=5" in warnings[0], warnings[0]
