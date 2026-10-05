"""Frozen-prefix re-prefill: end-to-end wiring tier,
driving the real `_probe_and_save_clean_kv` method.

ROOT CAUSE (shown by a direct payload diff of consecutive turns,
not inferred from token counts): a thread's `tools` array is not
stable across its own turns — an optional tool suite (a large set of memory-service tools
on some models) toggles in and out mid-conversation, for the SAME
thread_id, same message count, same system-prompt length. Because Qwen's chat
template renders `tools` at the FRONT of the prompt and kv_policy.py's hash
chain never sees `tools` at all (the same root cause on a second
surface), the engine's own warm-slot common-prefix match silently caps at a
constant early offset the instant the tools set changes, and self-perpetuates
(full re-prefill + checkpoint erasure) until it changes back or the engine
gives up. The fix for THAT belongs upstream of turbohaul (whatever assembles
the outgoing tools payload) — this change does not touch it; that is
out of scope.

What ships here instead is what matters regardless of cause: a
re-prefill of this magnitude must never be silent. `_classify_tools_fingerprint`
(pure, tested in test_classify_tools_fingerprint.py) is wired into
`_probe_and_save_clean_kv` — the one chokepoint that already runs on every
live per-turn probe (seven call sites, per the comment in that
function) — turning what would otherwise require a forensic pass over a large live
database into a single WARNING the instant it happens, carrying the
fingerprint AND the delta (prev/cur tool counts, entered/left names), not just
"something changed".

CAUSAL BOTH DIRECTIONS:
  * test_tools_change_mid_thread_logs_warning_with_delta -- a hook that NEVER
    fires (the wiring block deleted, or the whole fix reverted) fails this:
    no WARNING is ever captured, real assertion mismatch.
  * test_same_tools_resent_does_not_warn -- a hook that fires on EVERY turn
    (the hash-equality short-circuit removed or inverted) fails this instead:
    a WARNING appears where none should, real assertion mismatch.
  Neither test can pass for the wrong reason: both assert on the SPECIFIC log
  message content ("TOOLS CHANGED" / its absence), not merely on whether
  `_probe_and_save_clean_kv` raised or returned. This file imports only
  `TurbohaulManager` (not the new pure functions), so a full revert of
  manager.py makes these tests run for real and fail with real mismatches --
  not a vacuous ImportError.

At least one test drives the real production path: what the tests construct
is two consecutive client_meta payloads for the SAME thread; what production
DERIVES is the hash, the name-sets, the delta, and whether the WARNING fires.
"""
import types

import pytest

import turbohaul.manager as manager_mod
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
from turbohaul.manager import TurbohaulManager


# --- fixtures (mirror test_erase_order.py) -----------------------------------
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


@pytest.fixture
def kv_dir(tmp_path, monkeypatch):
    d = tmp_path / "kvcache"
    d.mkdir()
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _covered_scaffold_strip_off(mgr):
    mgr.runtime.kv.covered_scaffold_strip = False


class _Resp:
    def raise_for_status(self):
        return None

    def json(self):
        return {}


class _Client:
    """Generic fake for _probe_and_save_clean_kv's downstream strip-probe --
    every GET/POST succeeds with an empty/ok body. This check's own assertions
    are all settled BEFORE the strip-probe runs (the fingerprint hook sits
    right after client_meta is resolved, ahead of every later gate), so what
    happens downstream here is irrelevant to what's under test."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kw):
        return _Resp()

    async def post(self, url, json=None, **kw):
        return _Resp()


@pytest.fixture
def make_httpx(monkeypatch):
    class _FakeHttpx:
        AsyncClient = staticmethod(lambda *a, **k: _Client())
        Timeout = staticmethod(lambda *a, **k: None)
    monkeypatch.setattr(manager_mod, "httpx", _FakeHttpx)


_PORT = 59500
_PID = 4242


def _handle():
    return types.SimpleNamespace(parallel=1, port=_PORT, pid=_PID)


def _msgs(k):
    return [{"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"turn-{i}-content"} for i in range(k)]


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _slot(tid, tools):
    return types.SimpleNamespace(
        thread_id=tid,
        model_tag="test-model",
        admission_ctx_len=100,
        client_meta={"messages": _msgs(3), "tools": tools},
        context=None,
        prompt="",
        port=_PORT,
        pid=_PID,
    )


_BASE = [_tool("a"), _tool("b")]
_GROWN = _BASE + [_tool(f"extra_tool_{i}") for i in range(3)]


def _turn(mgr, tid, tools):
    """Drive ONE real per-turn seam.

    These tests drive the per-turn seam rather than
    `_probe_and_save_clean_kv(..., save_to_disk=False)`, because the check is
    not in that method; it lives in
    `_check_thread_tools_fingerprint` and is reached from the PER-TURN seam, so
    that streaming and fan-out turns (which never call the KV probe at all) are not
    blind. Driving the probe here would assert nothing.

    This calls the real production `_mark_slot_active`, which is five of the
    check's six call sites. It is SYNC -- no await.
    """
    mgr._mark_slot_active(_slot(tid, tools))


@pytest.mark.asyncio
async def test_tools_change_mid_thread_logs_warning_with_delta(
        mgr, kv_dir, make_httpx, caplog):
    """THE core proof: two consecutive live turns on the SAME thread, tools
    grows from 2 to 5 -- the real path must log the change WITH the delta
    (counts + entered names), not just a bare "changed" flag."""
    caplog.set_level("WARNING", logger="turbohaul.manager")
    tid = "hermes-main-test-thread"

    _turn(mgr, tid, _BASE)
    assert not any("TOOLS CHANGED" in r.message for r in caplog.records), (
        "the FIRST turn ever seen for a thread must never warn (nothing to "
        "diverge from yet)"
    )

    caplog.clear()
    _turn(mgr, tid, _GROWN)

    warnings = [r.message for r in caplog.records if "TOOLS CHANGED" in r.message]
    assert len(warnings) == 1, f"expected exactly one TOOLS CHANGED warning, got {warnings}"
    msg = warnings[0]
    assert "prev_count=2" in msg and "cur_count=5" in msg, msg
    assert "extra_tool_0" in msg and "extra_tool_1" in msg and "extra_tool_2" in msg, msg
    assert "left=[]" in msg, f"nothing was removed, left must be empty: {msg}"


@pytest.mark.asyncio
async def test_same_tools_resent_does_not_warn(mgr, kv_dir, make_httpx, caplog):
    """Causal counter-direction: the SAME tools list resent turn after turn
    (the ordinary, healthy case -- a client that behaves) must NEVER warn,
    even across several consecutive turns."""
    caplog.set_level("WARNING", logger="turbohaul.manager")
    tid = "hermes-main-stable-thread"

    for _ in range(4):
        _turn(mgr, tid, _BASE)

    assert not any("TOOLS CHANGED" in r.message for r in caplog.records), (
        f"a thread resending byte-identical tools must never warn, got: "
        f"{[r.message for r in caplog.records if 'TOOLS CHANGED' in r.message]}"
    )


@pytest.mark.asyncio
async def test_tools_removed_reports_left_not_entered(mgr, kv_dir, make_httpx, caplog):
    """Mirror image of the growth test: shrinking must report `left`, not
    `entered` -- proves the delta direction isn't hardcoded to one shape."""
    caplog.set_level("WARNING", logger="turbohaul.manager")
    tid = "hermes-main-shrink-thread"

    _turn(mgr, tid, _GROWN)
    caplog.clear()
    _turn(mgr, tid, _BASE)

    warnings = [r.message for r in caplog.records if "TOOLS CHANGED" in r.message]
    assert len(warnings) == 1
    msg = warnings[0]
    assert "prev_count=5" in msg and "cur_count=2" in msg, msg
    assert "entered=[]" in msg, f"nothing was added, entered must be empty: {msg}"
    assert "extra_tool_0" in msg, msg


@pytest.mark.asyncio
async def test_unload_seam_save_to_disk_true_does_not_fire_the_check(
        mgr, kv_dir, make_httpx, caplog):
    """The unload/seam path must never sample tools as if it were a live turn:
    its client_meta can be a last_main_client_meta CARRIED FORWARD through a
    disposable visitor rather than this call's own turn (the comment
    in _probe_and_save_clean_kv), so it is not a trustworthy sample.

    ⛔ THE GUARANTEE IS A PLACEMENT PROPERTY, NOT A FLAG.
    A flag is not what holds this: the check is not behind
    `if not save_to_disk`. The check does not live in that method at all, so
    a bare "no TOOLS CHANGED line" assertion would pass because THE CODE
    IS ELSEWHERE -- which is worse than a deleted test, and is exactly the trap
    this test family exists to remove.

    So it asserts the mechanism directly: the seam path must not REACH
    `_check_thread_tools_fingerprint` at all. That is a PLACEMENT property
    (the seam never marks a slot ACTIVE and never hits the matched promotion)
    rather than a flag check, and this test is what holds it: re-add a call to
    the check anywhere inside the probe/seam path and the spy below fires.
    """
    caplog.set_level("WARNING", logger="turbohaul.manager")
    tid = "hermes-main-seam-thread"

    reached = []
    mgr._check_thread_tools_fingerprint = lambda s: reached.append(s)

    await mgr._probe_and_save_clean_kv(_handle(), _slot(tid, _BASE), save_to_disk=True)
    caplog.clear()
    await mgr._probe_and_save_clean_kv(_handle(), _slot(tid, _GROWN), save_to_disk=True)

    assert reached == [], (
        "the unload/seam path reached the tools check -- it must not. Its "
        "client_meta can be a carried-forward last_main_client_meta, not this "
        f"turn's own sample. Reached with: {reached}")
    assert not any("TOOLS CHANGED" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_fingerprint_state_bounded_fifo_cannot_leak(mgr, kv_dir, make_httpx):
    """Drive _TOOLS_FINGERPRINT_CAP + 5 distinct threads through the real
    path; the stash must never exceed the cap (oldest evicted first)."""
    cap = manager_mod._TOOLS_FINGERPRINT_CAP
    for i in range(cap + 5):
        _turn(mgr, f"thread-{i}", _BASE)
    assert len(mgr._tools_fingerprint_by_thread) <= cap
    # the earliest threads were evicted; the most recent ones were kept
    assert f"thread-{cap + 4}" in mgr._tools_fingerprint_by_thread
    assert "thread-0" not in mgr._tools_fingerprint_by_thread


# --- the warning was itself silenced by a gate above it ----


def _handle_multi():
    """parallel=2 trips the single-series gate HONESTLY at its real condition
    (`_other_active or _hp != 1`), so the gate genuinely declines. Deliberately
    NOT monkeypatched away: a gate patched out is not the gate, and a test that
    removes the obstacle it claims to clear proves nothing."""
    return types.SimpleNamespace(parallel=2, port=_PORT, pid=_PID)


@pytest.mark.asyncio
async def test_tools_change_warns_even_when_single_series_gate_declines(
        mgr, kv_dir, make_httpx, caplog):
    """The tools check is not silenced by the single-series gate.

    The risk: a TOOLS CHANGED check placed BELOW the single-series gate's
    `return` inside `_probe_and_save_clean_kv` would never run: on ordinary multi-model
    operation the method returns before ever reaching it. Hoisting it above
    the gates inside that method would still tie it to the KV probe, so it is reached from the per-turn seam instead.

    THE PROPERTY IS WORTH HOLDING, and it spans BOTH mechanisms: on a
    real turn where the single-series gate declines the clean save, the operator
    must STILL get the warning. Those come from two different places —
    the warning from the per-turn seam, the decline from the KV probe — so the
    test drives BOTH, which is what a real gated turn actually does.

    Both halves asserted, because this must not move a save decision:
      1. the single-series decline STILL happens and is STILL logged, and
      2. the TOOLS CHANGED warning fires ANYWAY.

    Half 1 is a positive control, not decoration: without it this test could
    pass because the gate quietly failed to trip at all, which would silently
    convert it into a re-test of the ungated path it is supposed to cover.
    """
    caplog.set_level("INFO", logger="turbohaul.manager")
    tid = "hermes-main-gated-thread"
    h = _handle_multi()

    # A real gated turn = the per-turn seam runs, AND the KV probe runs and is
    # declined by the gate. Drive both, in that order, exactly as production does.
    _turn(mgr, tid, _BASE)
    await mgr._probe_and_save_clean_kv(h, _slot(tid, _BASE), save_to_disk=False)
    caplog.clear()
    _turn(mgr, tid, _GROWN)
    await mgr._probe_and_save_clean_kv(h, _slot(tid, _GROWN), save_to_disk=False)

    # 1. POSITIVE CONTROL — the gate really did decline, and said so. Asserted
    # against the literal wire value rather than the constant on purpose: if
    # KV_DECLINE_SINGLE_SERIES_GATE's VALUE is ever changed, this must fail
    # loudly rather than keep passing against a renamed symbol.
    declines = [r.message for r in caplog.records
                if "KV_SAVE_DECLINE" in r.message and "SINGLE_SERIES_GATE" in r.message]
    assert len(declines) == 1, (
        "the single-series gate must still decline and still log exactly as it "
        f"does today -- this change must not move a save decision: {declines}")

    # 2. THE FIX — and the warning fires anyway.
    warnings = [r.message for r in caplog.records if "TOOLS CHANGED" in r.message]
    assert len(warnings) == 1, (
        "TOOLS CHANGED must fire even when the single-series gate declines the "
        f"clean save -- that is the whole point of this test: {warnings}")
    msg = warnings[0]
    assert "prev_count=2" in msg and "cur_count=5" in msg, msg
    assert "extra_tool_0" in msg, msg
