"""Save-truthfulness defect: the
UNCONDITIONAL "resident KV cache saved before teardown" log in the
resident-teardown save path.

The save-before-teardown contract is explicit:
"An implementation that reports success when the save did not occur does
not satisfy this section."

The code does exactly that, at both resident-teardown save sites
(manager.py: the cap>=2 ``_unload_teardown`` site and the ``_drive_resident``
twin). Both call ``await self._save_slot_kv(...)`` and then log
``"resident KV cache saved before teardown"`` at INFO -- unconditionally.
But ``_save_slot_kv``'s own contract (its own docstring at the
definition) is: returns True iff at least ONE slot's bin+meta pair was
fully persisted. Its refusal paths return WITHOUT raising:
  - dirty tip                           -> returns False  (warned at source)
  - disposable identity                 -> returns False  (warned at source)
  - no populated slots                   -> returns None   (debug at source)
  - unsafe model_tag path chars          -> returns None   (warned at source)
so the except-branch's "save failed" warning never fires for a refusal,
and the caller logs success anyway. The
"saved" line can fire within milliseconds of an engine 501 -- an unconfirmed save
reported as a fact.

Fix (both sites, identical shape): capture the return value; log
"saved" only when it is True (the confirmed-save contract); otherwise log
an honest "not confirmed" line at DEBUG -- the refusal reason, when there
is one, is already logged at its source with its own WARNING (dirty tip /
disposable identity) or DEBUG (no populated slots), so the call site must not duplicate
the warning, only stop claiming success. The worker_loop singleton site
(manager.py) already follows the honest pattern (no unconditional
success log; declines go through the KV_DECLINE_* structured helper) and
is untouched.

The twin (``_drive_resident``) carries the byte-identical log and gets the
byte-identical edit; driving ``_drive_resident`` end-to-end requires the
full driver loop, so the behavioural arm here exercises the primary
``_unload_teardown`` site (the first of the two) and the twin is
covered by the identical hunk + the suite.

Fails on the pre-fix code, passes after the fix -- the refused-save arm asserts
the operator-visible log line is ABSENT and finds it present.
"""
import logging
import asyncio
from unittest.mock import AsyncMock

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
from turbohaul.manager import Resident, ResidentState, TurbohaulManager

SAVED_LINE = "resident KV cache saved before teardown"


class _FakeTeardownHandle:
    """Just enough of a SidecarHandle for the teardown save block:
    alive, a port, and parallel=2 (so the single-stream clean-flush
    branch is not taken -- this test is about the save-result contract,
    not the flush)."""

    def __init__(self):
        self.port = 19555
        self.parallel = 2

    def is_alive(self):
        return True


@pytest.fixture
def mgr(tmp_path):
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
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False, max_parallel_sidecars=8),
        pull=PullConfig(),
    )
    m = TurbohaulManager(boot, runtime)
    return m


def _resident(mgr, save_result):
    """A cap>=2 resident at the exact shape the teardown save block sees,
    with ``_save_slot_kv`` standing in for the real save (the refusal
    mechanisms it returns are tested at their own sites -- this file tests
    the CALLER'S contract: inspect the result before reporting success).
    reclaim_credited_mib=0 keeps the VRAM-credit finally-block inert, so
    the only observable side effect of the drive is the logging."""
    r = Resident(
        model_tag="victim-model", resident_key="victim-model",
        state=ResidentState.ACTIVE, last_active_monotonic=1.0,
        main_gpu=0, split_mode="none",
    )
    r.handle = _FakeTeardownHandle()
    r.idle_thread_id = "tid-1234567890"
    r.active_slot = None
    r.idle_client_meta = {"ip": "10.0.0.5"}
    r.idle_admission_ctx_len = 0
    r.reclaim_credited_mib = 0
    mgr._save_slot_kv = AsyncMock(return_value=save_result)
    mgr._reap_resident_handle = AsyncMock()
    return r


def _info_lines(caplog):
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.INFO and "turbohaul" in (rec.name or "")
    ]


@pytest.mark.asyncio
class TestSaveTruthfulness:
    async def test_RED_refused_save_must_not_log_saved(self, mgr, caplog):
        """The traced defect, watched. _save_slot_kv is REFUSED (returns
        False -- the dirty-tip / disposable-identity contract: logged at its source, nothing
        persisted, no exception). An implementation that
        reports success when the save did not occur does not satisfy this
        contract.

        PRE-FIX (expected RED): the teardown unconditionally logs
        "resident KV cache saved before teardown" -- the operator sees a
        save that did not happen (for example, milliseconds after an engine 501).
        POST-FIX (expected GREEN): the "saved" line is absent for a
        refused save; the honest "not confirmed" line (DEBUG) carries the
        result instead.
        """
        caplog.set_level(logging.DEBUG)
        r = _resident(mgr, save_result=False)
        await mgr._unload_teardown(r)

        assert mgr._save_slot_kv.await_count == 1, (
            "the teardown must still attempt the save -- the fix is about "
            "reporting the result, not skipping the save"
        )
        saved_lines = [m for m in _info_lines(caplog) if SAVED_LINE in m]
        assert saved_lines == [], (
            f"the save was REFUSED (result=False) but the teardown logged "
            f"success anyway: {saved_lines!r} -- an "
            f"implementation that reports success when the save did not "
            f"occur does not satisfy this section"
        )

    async def test_RED_no_populated_slots_must_not_log_saved(self, mgr, caplog):
        """The other unconfirmed contract value: _save_slot_kv returns
        None when the engine has no populated slots (nothing to save --
        debug-logged at source). Same defect: the unconditional
        "saved" log claims a save that did not occur.

        PRE-FIX (expected RED); POST-FIX (expected GREEN).
        """
        caplog.set_level(logging.DEBUG)
        r = _resident(mgr, save_result=None)
        await mgr._unload_teardown(r)

        saved_lines = [m for m in _info_lines(caplog) if SAVED_LINE in m]
        assert saved_lines == [], (
            f"the save returned None (no populated slots) but the teardown "
            f"logged success anyway: {saved_lines!r}"
        )

    async def test_CONTROL_confirmed_save_still_logs_saved(self, mgr, caplog):
        """The fix must not over-correct: when _save_slot_kv returns True
        (the confirmed-save contract: at least one slot's bin+meta
        fully persisted), the operator MUST still see the "saved" line --
        that is the log's whole reason to exist. Passes pre-fix AND must
        pass post-fix."""
        caplog.set_level(logging.DEBUG)
        r = _resident(mgr, save_result=True)
        await mgr._unload_teardown(r)

        saved_lines = [m for m in _info_lines(caplog) if SAVED_LINE in m]
        assert saved_lines, (
            "a CONFIRMED save (result=True) must still be logged -- "
            "removing the success log entirely would make the teardown "
            "silent about its main job"
        )
        assert "victim-model" in saved_lines[0]

    async def test_CONTROL_raised_save_still_warns_not_claims_saved(self, mgr, caplog):
        """Pre-existing arm, kept as a control: a save that RAISES takes
        the except-branch's "save failed (best-effort)" warning and must
        not also claim success. (The defect is specifically the
        NON-raising refusals, which the except-branch never sees.) Passes
        pre-fix AND must pass post-fix."""
        caplog.set_level(logging.DEBUG)
        r = _resident(mgr, save_result=None)
        mgr._save_slot_kv = AsyncMock(side_effect=OSError("engine 501"))
        await mgr._unload_teardown(r)

        messages = [m for m in caplog.messages if "KV cache" in m]
        saved_lines = [m for m in _info_lines(caplog) if SAVED_LINE in m]
        assert saved_lines == [], f"no success log after a raise: {saved_lines!r}"
        assert any("failed" in m for m in messages), (
            f"the except-branch warning must still fire on a raise: {messages!r}"
        )
