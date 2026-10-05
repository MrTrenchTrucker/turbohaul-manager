"""Disk persist ONLY on warm-idle
expiry + shutdown, never on model_swap.

`_teardown_idle_holder`'s flush/persist sequence ran `_persist_clean_bin_to_ssd`
(the SSD copy2, in manager.py) unconditionally for every teardown
`reason`, including `model_swap` — a live model swap does not need cold
storage, and the SSD copy sat on that critical path for nothing. Fix: gate
ONLY that call on `reason != "model_swap"`.

Explicitly NOT touched, by design
(the long swap-time stall is the
reprefill probe inside `_flush_clean_kv_at_unload`, not the SSD copy):
`_flush_clean_kv_at_unload` (the only writer of main's RAM clean bin —
a standing invariant), `_save_slot_kv`, `_shadow_save_at_swap`.
All three stay unconditional for every reason, including model_swap.

Run:
    PYTHONPATH=<repo>/src python3 -m pytest tests/<this test file> -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

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


def _fake_held():
    held = MagicMock()
    held.is_alive.return_value = True
    held.port = 59500
    held.pid = 90001
    return held


def _wire_idle_holder(mgr, *, thread_id="t1", model_tag="qwen3.6-27b"):
    """Set the idle-holder state _teardown_idle_holder reads, with a plain
    unlabeled client_meta (role=None) so _seam_flush_allowed returns True —
    the ordinary main-session case."""
    held = _fake_held()
    mgr._idle_handle = held
    mgr._idle_model_tag = model_tag
    mgr._idle_thread_id = thread_id
    mgr._idle_admission_ctx_len = 40000
    mgr._idle_client_meta = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]}
    return held


@pytest.fixture
def teardown_spies(mgr, monkeypatch):
    """Patch every downstream call inside _teardown_idle_holder's flush block
    (plus _sigterm, which always fires) with recording fakes, so the test
    observes exactly which calls the gate lets through -- without needing a
    real engine."""
    calls = {"flush": 0, "persist": 0, "save_slot_kv": 0, "shadow": 0, "sigterm": 0}
    persist_calls = []  # captured (args, kwargs) per persist call

    async def fake_flush(*a, **k):
        calls["flush"] += 1

    def fake_persist(*a, **k):
        calls["persist"] += 1
        persist_calls.append(k)

    async def fake_save_slot_kv(*a, **k):
        calls["save_slot_kv"] += 1
        return True

    async def fake_shadow(*a, **k):
        calls["shadow"] += 1

    async def fake_sigterm(*a, **k):
        calls["sigterm"] += 1
        return True, "sigterm-clean"

    monkeypatch.setattr(mgr, "_flush_clean_kv_at_unload", fake_flush)
    monkeypatch.setattr(mgr, "_persist_clean_bin_to_ssd", fake_persist)
    monkeypatch.setattr(mgr, "_save_slot_kv", fake_save_slot_kv)
    monkeypatch.setattr(mgr, "_shadow_save_at_swap", fake_shadow)
    monkeypatch.setattr(mgr, "_sigterm", fake_sigterm)
    calls["persist_calls"] = persist_calls
    return calls


# ================================================================================
# Case 1 — model_swap must NOT invoke the SSD persist path.
# This test FAILS on the unmodified code (verified separately).
# ================================================================================

@pytest.mark.asyncio
async def test_model_swap_skips_ssd_persist(mgr, teardown_spies):
    _wire_idle_holder(mgr)
    await mgr._teardown_idle_holder("model_swap")
    assert teardown_spies["persist"] == 0, (
        "model_swap invoked the SSD persist path — it must not, per the "
        "storage-tier policy"
    )


# ================================================================================
# Case 2 — idle_expired and shutdown STILL persist. The arm
# that makes test 1 non-vacuous, and the actual contract.
# ================================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["idle_expired", "shutdown"])
async def test_idle_and_shutdown_still_persist(mgr, teardown_spies, reason):
    _wire_idle_holder(mgr)
    await mgr._teardown_idle_holder(reason)
    assert teardown_spies["persist"] == 1, (
        f"{reason} must still invoke the SSD persist path — this is cold "
        f"storage, not the swap critical path"
    )


# ================================================================================
# Case 3 — model_swap must STILL run the RAM clean-save.
# Proves the fix did not regress the protected behaviour:
# _flush_clean_kv_at_unload stays unconditional for every reason. This test
# PASSES on both base and patched (asserting existing, untouched behavior) —
# stated as such, not claimed as a "fails on base" criterion.
# ================================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["model_swap", "idle_expired", "shutdown"])
async def test_ram_clean_save_stays_unconditional_for_every_reason(
    mgr, teardown_spies, reason
):
    _wire_idle_holder(mgr)
    await mgr._teardown_idle_holder(reason)
    assert teardown_spies["flush"] == 1, (
        f"_flush_clean_kv_at_unload did not run for reason={reason!r} — it "
        f"must be unconditional (the only writer of main's RAM clean bin)"
    )
    assert teardown_spies["save_slot_kv"] == 1
    assert teardown_spies["shadow"] == 1
    assert teardown_spies["sigterm"] == 1


# ================================================================================
# Break-test guard: idle_dead already self-skips via held.is_alive() —
# confirm, not assume. A dead holder must invoke NEITHER flush nor persist.
# ================================================================================

@pytest.mark.asyncio
async def test_idle_dead_self_skips_via_is_alive_gate(mgr, teardown_spies):
    held = _wire_idle_holder(mgr)
    held.is_alive.return_value = False
    await mgr._teardown_idle_holder("idle_dead")
    assert teardown_spies["flush"] == 0
    assert teardown_spies["persist"] == 0
    assert teardown_spies["sigterm"] == 1, "sigterm must still run to reap the dead holder"


# ================================================================================
# The persist log line carries a trigger label. Without it,
# idle_expired and shutdown persists (and everything else that reaches
# _persist_clean_bin_to_ssd_by_hash) would be indistinguishable -- the
# ran-vs-ran-and-found-nothing ambiguity for anyone reading the logs.
# Enumerated (not assumed to be only the two primary reasons) every real call
# site: _teardown's three reasons (worker-uncaught-exception,
# loading-fail-health-timeout, grace-expired), _teardown_idle_holder's two
# reachable reasons (idle_expired, shutdown -- model_swap excluded by its
# own gate above, idle_dead excluded by the is_alive gate just proven),
# and three ownership-transfer sites inside _probe_and_save_clean_kv that
# share one label ("ownership_transfer") because they're the same
# conceptual event at three different points in that function.
# ================================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["idle_expired", "shutdown"])
async def test_idle_expired_and_shutdown_persist_carry_matching_trigger_label(
    mgr, teardown_spies, reason
):
    """The two primary reasons: not just that persist fires
    (already proven above), but that the label reaching it is the reason
    itself, not a generic placeholder."""
    _wire_idle_holder(mgr)
    await mgr._teardown_idle_holder(reason)
    calls = teardown_spies["persist_calls"]
    assert len(calls) == 1, calls
    assert calls[0].get("trigger") == reason, (
        f"expected trigger={reason!r}, got {calls[0]!r}"
    )


def test_persist_trigger_is_required_no_default():
    """Same design as read_timeout_s: a signature constrains
    what can run, a test only asserts what it can see -- prefer the
    stronger guarantee once nothing legitimate depends on a default (there
    never was one for `trigger`; it's a new, deliberately-required
    parameter, not a loosened existing one)."""
    import inspect
    from turbohaul.manager import TurbohaulManager

    for name in ("_persist_clean_bin_to_ssd", "_persist_clean_bin_to_ssd_by_hash"):
        sig = inspect.signature(getattr(TurbohaulManager, name))
        param = sig.parameters["trigger"]
        assert param.default is inspect.Parameter.empty, (
            f"{name}'s trigger parameter has regained a default "
            f"({param.default!r}) -- the requirement exists because callers could "
            f"silently omit context about why a persist happened; keep it "
            f"required so a future 6th call site can't do that again"
        )
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_every_persist_call_site_passes_trigger_explicitly():
    """AST-walks the real source (not a hand-maintained line list -- the
    exact thing that would go stale) for every call site of either persist
    function, INCLUDING the asyncio.to_thread(...)-wrapped shape every real
    call site actually uses (a naive walk matching only direct Call nodes
    misses these entirely -- it would find only 1 of the 6 real sites,
    the one direct call, which is why this walk is
    to_thread-aware). Confirms each of the 6 sites
    supplies `trigger` and pins the count so a 7th site is a visible test
    change, not a silent addition."""
    import ast
    import inspect
    from pathlib import Path
    from turbohaul.manager import TurbohaulManager

    src_path = Path(inspect.getfile(TurbohaulManager))
    tree = ast.parse(src_path.read_text(), filename=str(src_path))
    target_names = {"_persist_clean_bin_to_ssd", "_persist_clean_bin_to_ssd_by_hash"}
    sites = []  # (lineno, has_trigger)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Direct call: self._persist_clean_bin_to_ssd_by_hash(...)
        if isinstance(func, ast.Attribute) and func.attr in target_names:
            has_trigger = any(kw.arg == "trigger" for kw in node.keywords)
            sites.append((node.lineno, has_trigger))
            continue
        # asyncio.to_thread(self._persist_clean_bin_to_ssd, ..., trigger=...)
        # -- every real production call site uses this shape. to_thread
        # forwards its own **kwargs to the wrapped function, so `trigger`
        # must be found among to_thread's OWN keywords, not the (missing)
        # keywords on the plain attribute reference passed as its first arg.
        if (isinstance(func, ast.Attribute) and func.attr == "to_thread"
                and node.args
                and isinstance(node.args[0], ast.Attribute)
                and node.args[0].attr in target_names):
            has_trigger = any(kw.arg == "trigger" for kw in node.keywords)
            sites.append((node.lineno, has_trigger))

    assert len(sites) == 6, (
        f"expected exactly 6 call sites (5 external callers + the "
        f"_persist_clean_bin_to_ssd wrapper's own internal delegation), "
        f"found {len(sites)} at lines {[l for l, _ in sites]} -- the enumeration "
        f"assumes exactly this; a new site changes the count "
        f"here on purpose"
    )
    missing = [lineno for lineno, has_trigger in sites if not has_trigger]
    assert not missing, (
        f"call site(s) at line(s) {missing} do not pass trigger explicitly "
        f"-- Python's own required-kwarg enforcement would already raise "
        f"TypeError at that call, but if this assertion ever fires it means "
        f"that guarantee was somehow bypassed, which is worth knowing"
    )


@pytest.mark.asyncio
async def test_real_log_line_carries_the_trigger_not_just_the_kwarg(mgr, tmp_path, monkeypatch, caplog):
    """The mocked tests above prove the KWARG reaches the (faked) persist
    function. This one drives the REAL, unmocked
    _persist_clean_bin_to_ssd_by_hash and asserts the actual emitted log
    line contains the trigger -- the thing the trigger label is actually about
    (an operator reading logs, not a kwarg reaching a test double)."""
    import logging
    import turbohaul.subprocess_mgr as subprocess_mgr

    save_dir = tmp_path / "kvcache"
    save_dir.mkdir()
    persist_dir = tmp_path / "kvcache_persist"
    # _persist_clean_bin_to_ssd_by_hash imports these names LOCALLY at call
    # time (`from turbohaul.subprocess_mgr import SLOT_SAVE_DIR,
    # SLOT_PERSIST_DIR`) rather than binding them at manager.py module
    # scope, so only the defining module's attributes need patching.
    monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(save_dir))
    monkeypatch.setattr(subprocess_mgr, "SLOT_PERSIST_DIR", str(persist_dir))

    (save_dir / "qwen3.6-27b.p59500.deadbeef01234567.bin").write_bytes(b"KV")

    with caplog.at_level(logging.INFO):
        mgr._persist_clean_bin_to_ssd_by_hash(
            "qwen3.6-27b", "deadbeef01234567", 59500, trigger="idle_expired",
        )

    lines = [r.getMessage() for r in caplog.records if "Option C: clean bin persisted" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "trigger=idle_expired" in lines[0], lines[0]
