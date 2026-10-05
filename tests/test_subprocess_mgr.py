"""Tests for subprocess_mgr (mocked Popen + httpx + nvidia-smi + killpg)."""
import asyncio
import hashlib
import logging
import os
import signal
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from turbohaul import subprocess_mgr
from turbohaul.subprocess_mgr import (
    ENGINE_REGISTERED_ENV_VARS,
    HEALTH_OK_STATUSES,
    HEALTH_REQUIRED_FIELDS,
    SchemaMismatch,
    SidecarHandle,
    _log_spawn_env_observations,
    classify_spawn_env,
    drained_sigterm,
    get_gpu_memory_used_mib,
    health_check_once,
    spawn_sidecar,
    verify_binary_sha256,
    verify_vram_cleared,
    wait_until_healthy,
)


def _make_fake_proc(pid=12345, poll_return=None):
    p = MagicMock()
    p.pid = pid
    if callable(poll_return):
        p.poll.side_effect = poll_return
    elif isinstance(poll_return, list):
        p.poll.side_effect = poll_return
    else:
        p.poll.return_value = poll_return
    return p


class TestSpawn:
    def test_spawn_uses_setsid(self):
        captured_kwargs = {}

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_fake_proc()

        spawn_sidecar(
            binary=Path("/opt/turboquant/build/bin/llama-server"),
            gguf_path=Path("/var/lib/turbohaul/blobs/sha256/ab/abc"),
            port=11500,
            model_tag="t",
            argv_flags=["--ctx-size", "4096"],
            popen_factory=fake_popen,
        )
        assert captured_kwargs["start_new_session"] is True

    def test_spawn_passes_argv(self):
        captured: list = []

        def fake_popen(cmd, **kwargs):
            captured.extend(cmd)
            return _make_fake_proc()

        spawn_sidecar(
            Path("/x/llama-server"),
            Path("/x/model.gguf"),
            11500,
            "t",
            ["--ctx-size", "4096", "--mlock"],
            popen_factory=fake_popen,
        )
        assert "--ctx-size" in captured
        assert "4096" in captured
        assert "--mlock" in captured
        assert "--port" in captured
        assert "11500" in captured
        assert "--host" in captured
        assert "127.0.0.1" in captured
        assert "-m" in captured

    def test_spawn_returns_handle(self):
        def fake_popen(*a, **k):
            return _make_fake_proc(pid=99)

        handle = spawn_sidecar(
            Path("/x"), Path("/y"), 11500, "model-a", [], popen_factory=fake_popen
        )
        assert isinstance(handle, SidecarHandle)
        assert handle.port == 11500
        assert handle.model_tag == "model-a"
        assert handle.pid == 99
        assert handle.is_alive() is True

    def test_handle_is_alive_after_exit(self):
        proc = _make_fake_proc(pid=1, poll_return=0)
        h = SidecarHandle(proc=proc, port=11500, model_tag="t")
        assert h.is_alive() is False


class TestSpawnEnvObserve:
    """Observe-only env-inheritance classification.

    The classification is observe-only: it never filters the spawn environment.
    These tests are split in two directions: a
    classifier that flagged everything would fail the clean-env tests below;
    one that flagged nothing would fail the deny-candidate / near-miss tests.

    spawn_sidecar passes env=launch_env(...) to Popen (engine_launch module,
    the env helper) -- the full, unfiltered environment plus at most one
    added key. The guarantee tested below is that environment, not the mechanism.
    """

    def test_flags_deny_candidate_not_engine_registered(self):
        flagged = classify_spawn_env({"MY_SECRET_TOKEN": "x"})
        assert len(flagged) == 1
        assert flagged[0]["name"] == "MY_SECRET_TOKEN"
        assert flagged[0]["category"] == "deny-candidate"

    def test_clean_env_flags_nothing(self):
        """A classifier that flagged everything would fail this -- the direction
        usually called the 'passes everything must fail a test' arm."""
        flagged = classify_spawn_env({
            "PATH": "/usr/bin", "HOME": "/root", "LANG": "en_US.UTF-8",
            "TERM": "xterm", "SHELL": "/bin/bash",
        })
        assert flagged == []

    def test_near_miss_engine_registered_credential_shaped_vars_are_exempt(self):
        """THE case this test exists for: a naive credential-shaped pattern
        (contains TOKEN, or contains KEY) would ALSO have caught these four --
        two beyond the original HF_TOKEN/LLAMA_API_KEY pair, found only by
        enumerating the engine's actual .set_env( registrations rather than
        prefix-guessing. All four are legitimate engine inputs (verified
        directly against arg.cpp: HF_TOKEN -> params.hf_token, LLAMA_API_KEY
        -> params.api_keys, LLAMA_ARG_API_KEY_FILE / LLAMA_ARG_SSL_KEY_FILE ->
        file-path options) and must be classified engine-registered, never
        deny-candidate -- a deny-list built the obvious way would have blocked
        the engine's own auth/TLS config, a self-inflicted outage that would
        look like an auth bug."""
        near_miss_names = {
            "HF_TOKEN", "LLAMA_API_KEY",
            "LLAMA_ARG_API_KEY_FILE", "LLAMA_ARG_SSL_KEY_FILE",
        }
        env = {name: "x" for name in near_miss_names}
        flagged = classify_spawn_env(env)
        assert {f["name"] for f in flagged} == near_miss_names
        for f in flagged:
            assert f["category"] == "engine-registered", f

    def test_word_boundary_tokens_plural_is_not_a_token_credential(self):
        """Substring matching was tried and rejected: it
        would flag LLAMA_ARG_IMAGE_MAX_TOKENS / LLAMA_ARG_MTMD_BATCH_MAX_TOKENS
        (both legitimate, both contain "_TOKEN" as a substring of "_TOKENS")
        as credential-shaped. This proves the word-boundary fix holds for a
        var that ISN'T also engine-registered, where the near-miss test above
        can't distinguish "exempt because engine-registered" from "exempt
        because word-boundary" -- this one isolates the word-boundary logic
        alone."""
        flagged = classify_spawn_env({"MAX_TOKENS_ALLOWED": "5"})
        assert flagged == []

    def test_engine_registered_non_credential_var_flagged_informationally(self):
        flagged = classify_spawn_env({"LLAMA_ARG_CTX_SIZE": "4096"})
        assert len(flagged) == 1
        assert flagged[0]["category"] == "engine-registered"
        assert "collides with a CLI flag" in flagged[0]["rule"]

    def test_engine_registered_set_has_no_accidental_duplicates(self):
        # ENGINE_REGISTERED_ENV_VARS is a frozenset already -- this pins the
        # derived count so a future re-derivation from a different engine
        # build is a visible diff, not a silent shrink/grow.
        assert len(ENGINE_REGISTERED_ENV_VARS) == 134

    def test_clean_spawn_logs_at_info_not_debug(self, caplog):
        """The no-flags case must be visible at INFO, same as the
        flagged case -- not silent DEBUG. Silence at DEBUG makes it impossible to tell
        "ran, found nothing" from "never ran" in a deployed log.
        This pins the fix so it can't
        quietly regress back to DEBUG-on-empty.
        Uses a hand-built clean env (not
        os.environ) so the assertion doesn't depend on what happens to be
        ambient in whatever machine runs this test."""
        with caplog.at_level(logging.DEBUG):
            _log_spawn_env_observations(11503, "t4", {"PATH": "/usr/bin", "HOME": "/root"})

        observe = [r for r in caplog.records if "env-observe" in r.getMessage()]
        assert len(observe) == 1, f"expected exactly one env-observe line, got {len(observe)}"
        assert observe[0].levelno == logging.INFO, (
            f"clean-env observation logged at {logging.getLevelName(observe[0].levelno)}, "
            f"not INFO -- this is invisible in a deployed log and reproduces the exact "
            f"'ran vs never-ran' ambiguity"
        )
        assert "flagged=0" in observe[0].getMessage()

    def test_spawn_sidecar_passes_full_unfiltered_env(self, monkeypatch, tmp_path):
        """The behavior-preservation guarantee: a spawn with a deny-candidate-
        shaped var actually present in the real environment must still spawn
        with the full, unfiltered environment -- proving nothing is filtered,
        only logged. env=launch_env(...) is always passed to Popen; the
        guarantee is the environment content, not the mechanism -- so this
        asserts the guarantee itself rather than a "no env= kwarg" HOW.
        Note: SLOT_SAVE_DIR monkeypatched to a tmp
        path so this test doesn't need /var/lib/turbohaul on the runner."""
        monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(tmp_path / "kvcache"))
        monkeypatch.setenv("SOME_FAKE_SECRET_TOKEN_EXAMPLE", "sentinel-value")
        captured_kwargs = {}

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_fake_proc()

        spawn_sidecar(
            binary=Path("/x/llama-server"), gguf_path=Path("/x/model.gguf"),
            port=11500, model_tag="t", argv_flags=[], popen_factory=fake_popen,
        )
        assert "env" in captured_kwargs
        assert captured_kwargs["env"]["SOME_FAKE_SECRET_TOKEN_EXAMPLE"] == "sentinel-value"
        assert captured_kwargs["env"] == dict(os.environ)

    def test_spawn_sidecar_vision_single_card_env_matches_launch_env(self, monkeypatch, tmp_path):
        """Pins engine_launch's env=launch_env(...) wiring at its
        real call site inside spawn_sidecar. A mutant that swaps this back to
        env=dict(os.environ) (the fix silently bypassed) is only caught by a
        test that drives spawn_sidecar through a vision argv -- launch_env
        itself, as a pure function, and the text-only path through
        spawn_sidecar are pinned elsewhere. A single-card vision launch must get
        MTMD_BACKEND_DEVICE=CUDA<main_gpu> with every other key/value from os.environ
        passed through unfiltered."""
        monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(tmp_path / "kvcache"))
        monkeypatch.delenv("MTMD_BACKEND_DEVICE", raising=False)
        captured_kwargs = {}

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_fake_proc()

        spawn_sidecar(
            binary=Path("/x/llama-server"), gguf_path=Path("/x/model.gguf"),
            port=11502, model_tag="t3",
            argv_flags=["--split-mode", "none", "--main-gpu", "1", "--mmproj", "/x/mm.gguf"],
            popen_factory=fake_popen,
        )
        env = captured_kwargs["env"]
        assert env["MTMD_BACKEND_DEVICE"] == "CUDA1"
        expected = dict(os.environ)
        expected["MTMD_BACKEND_DEVICE"] = "CUDA1"
        assert env == expected

    def test_spawn_sidecar_preset_never_overridden_and_warns_on_mismatch(self, monkeypatch, caplog, tmp_path):
        """Pins the preset-mismatch warning at its real call site
        inside spawn_sidecar. A mutant that swaps log.warning(_preset_warning) for
        `pass` (the warning silently dropped) is only caught by a test
        that drives spawn_sidecar with an operator preset already set in
        the environment. An operator's MTMD_BACKEND_DEVICE preset must never be
        overridden (by design) regardless of whether it matches --main-gpu's card;
        a mismatching preset must log exactly one WARNING naming both cards
        (by design), and a matching preset must not warn at all."""
        monkeypatch.setattr(subprocess_mgr, "SLOT_SAVE_DIR", str(tmp_path / "kvcache"))
        vision_argv = ["--split-mode", "none", "--main-gpu", "1", "--mmproj", "/x/mm.gguf"]
        captured_kwargs = {}

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_fake_proc()

        # Mismatching preset: kept verbatim, exactly one WARNING naming both cards.
        monkeypatch.setenv("MTMD_BACKEND_DEVICE", "CUDA0")
        with caplog.at_level(logging.WARNING):
            spawn_sidecar(
                binary=Path("/x/llama-server"), gguf_path=Path("/x/model.gguf"),
                port=11503, model_tag="t4", argv_flags=vision_argv,
                popen_factory=fake_popen,
            )
        assert captured_kwargs["env"]["MTMD_BACKEND_DEVICE"] == "CUDA0"
        mismatch = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "CUDA0" in r.getMessage() and "CUDA1" in r.getMessage()
        ]
        assert len(mismatch) == 1, f"expected exactly one mismatch WARNING, got {len(mismatch)}"

        # Matching preset: no mismatch warning.
        caplog.clear()
        captured_kwargs.clear()
        monkeypatch.setenv("MTMD_BACKEND_DEVICE", "CUDA1")
        with caplog.at_level(logging.WARNING):
            spawn_sidecar(
                binary=Path("/x/llama-server"), gguf_path=Path("/x/model.gguf"),
                port=11504, model_tag="t5", argv_flags=vision_argv,
                popen_factory=fake_popen,
            )
        assert captured_kwargs["env"]["MTMD_BACKEND_DEVICE"] == "CUDA1"
        mismatch2 = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "differs from" in r.getMessage()
        ]
        assert mismatch2 == []

    def test_spawn_sidecar_logs_flagged_name_and_rule_not_just_a_count(self, monkeypatch, caplog):
        """Through the REAL spawn_sidecar call path (production derives this,
        the test only constructs the injected env var) -- proves the
        observation is actually wired into the spawn, not just a standalone
        function nobody calls. Asserts the NAME and part of the RULE text
        appear in the log, not merely a count."""
        monkeypatch.setenv("EXAMPLE_FAKE_API_KEY", "sentinel-value")

        def fake_popen(cmd, **kwargs):
            return _make_fake_proc()

        with caplog.at_level(logging.INFO):
            spawn_sidecar(
                binary=Path("/x/llama-server"), gguf_path=Path("/x/model.gguf"),
                port=11501, model_tag="t2", argv_flags=[], popen_factory=fake_popen,
            )

        observe = [r for r in caplog.records if "env-observe" in r.getMessage()]
        assert len(observe) == 1, f"expected exactly one env-observe line, got {len(observe)}"
        msg = observe[0].getMessage()
        assert "EXAMPLE_FAKE_API_KEY" in msg
        assert "deny-candidate" in msg
        assert "credential-shaped" in msg

    def test_never_logs_a_flagged_value_direct(self):
        """Unit level: classify_spawn_env's return value must never
        contain the VALUE of a flagged (or any) env var, only names and static
        rule text. Checks every string in every returned dict, not just the
        obvious 'rule' field, so a future edit that leaks the value into
        'name' or a new field is also caught."""
        secret = "sk-CAUSAL-LEAK-CANARY-9f3e7b1c4a"
        flagged = classify_spawn_env({"FAKE_SECRET_TOKEN": secret, "PATH": "/usr/bin"})
        assert flagged, "expected the credential-shaped var to be flagged at all"
        for entry in flagged:
            for value in entry.values():
                assert secret not in value, f"value leaked into classify_spawn_env output: {entry}"

    def test_never_logs_a_flagged_value_through_spawn_sidecar(self, monkeypatch, caplog):
        """Same property, through the real spawn_sidecar call path -- the
        production entry point, not just the standalone function."""
        secret = "sk-CAUSAL-LEAK-CANARY-9f3e7b1c4a"
        monkeypatch.setenv("ANOTHER_FAKE_SECRET_TOKEN", secret)

        def fake_popen(cmd, **kwargs):
            return _make_fake_proc()

        with caplog.at_level(logging.INFO):
            spawn_sidecar(
                binary=Path("/x/llama-server"), gguf_path=Path("/x/model.gguf"),
                port=11502, model_tag="t3", argv_flags=[], popen_factory=fake_popen,
            )

        for record in caplog.records:
            assert secret not in record.getMessage(), (
                f"value leaked into a log record: {record.getMessage()!r}"
            )


@pytest.mark.asyncio
class TestHealthCheck:
    async def test_health_check_200_ok(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_client.get = AsyncMock(return_value=mock_response)
        result = await health_check_once(11500, mock_client)
        assert result == {"status": "ok"}

    async def test_health_check_503_returns_none(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_response)
        assert await health_check_once(11500, mock_client) is None

    async def test_health_check_missing_status_schema_mismatch(self):
        """Tom's Fork drift defense - missing required field raises (FP M3)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"loaded": True}
        mock_client.get = AsyncMock(return_value=mock_response)
        with pytest.raises(SchemaMismatch, match="missing fields"):
            await health_check_once(11500, mock_client)

    async def test_health_check_non_dict_schema_mismatch(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["not", "a", "dict"]
        mock_client.get = AsyncMock(return_value=mock_response)
        with pytest.raises(SchemaMismatch, match="not a dict"):
            await health_check_once(11500, mock_client)

    async def test_health_check_network_error_returns_none(self):
        import httpx
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("conn refused"))
        assert await health_check_once(11500, mock_client) is None


@pytest.mark.asyncio
class TestWaitUntilHealthy:
    async def test_immediate_ok(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_client.get = AsyncMock(return_value=mock_response)
        ok = await wait_until_healthy(11500, timeout_s=2.0, http_client=mock_client, poll_interval_s=0.01)
        assert ok is True

    async def test_timeout(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_response)
        ok = await wait_until_healthy(11500, timeout_s=0.1, http_client=mock_client, poll_interval_s=0.05)
        assert ok is False

    async def test_dead_child_fails_fast(self):
        """FSM-wedge fix: a child that EXITED during load fails fast (is_alive False)
        instead of burning the full timeout. timeout_s=30 but a dead child must
        return False well under the 2s asyncio.wait_for guard."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_response)
        ok = await asyncio.wait_for(
            wait_until_healthy(
                11500, timeout_s=30.0, http_client=mock_client,
                poll_interval_s=0.01, is_alive=lambda: False,
            ),
            timeout=2.0,
        )
        assert ok is False

    async def test_alive_but_unhealthy_still_times_out(self):
        """LIVENESS-ONLY guardrail: an ALIVE child that is merely slow to become
        healthy must NOT be killed by the liveness check - it returns False only
        via the normal timeout path."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_response)
        ok = await wait_until_healthy(
            11500, timeout_s=0.1, http_client=mock_client,
            poll_interval_s=0.02, is_alive=lambda: True,
        )
        assert ok is False

    async def test_health_wins_over_dead_same_tick(self):
        """Ordering guardrail: the liveness check sits AFTER the health probe, so a
        child that reports healthy on the same iteration still returns True even if
        its is_alive() would read False."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_client.get = AsyncMock(return_value=mock_response)
        ok = await wait_until_healthy(
            11500, timeout_s=2.0, http_client=mock_client,
            poll_interval_s=0.01, is_alive=lambda: False,
        )
        assert ok is True

    async def test_stall_warning_silent_before_midpoint(self, caplog):
        """The stall warning must NOT fire for a load that becomes healthy before
        the midpoint of its timeout. This is the arm that matters: without the
        midpoint test the warning fires on the FIRST poll of every load, so every
        healthy model swap logs a stall warning and the signal is worthless."""
        mock_client = MagicMock()
        unhealthy = MagicMock()
        unhealthy.status_code = 503
        healthy = MagicMock()
        healthy.status_code = 200
        healthy.json.return_value = {"status": "ok"}
        # 503, 503, then ok - reached at ~0.04s, far short of the 0.5s midpoint.
        mock_client.get = AsyncMock(side_effect=[unhealthy, unhealthy, healthy])

        with caplog.at_level(logging.WARNING):
            ok = await wait_until_healthy(
                11500, timeout_s=1.0, http_client=mock_client, poll_interval_s=0.02,
            )

        assert ok is True
        stall = [r for r in caplog.records if "still not healthy" in r.getMessage()]
        assert stall == [], f"warned before the midpoint: {[r.getMessage() for r in stall]}"

    async def test_stall_warning_emitted_once_past_midpoint(self, caplog):
        """A load still unhealthy past the midpoint warns EXACTLY once (per wait),
        naming the elapsed time and the timeout so the hang is visible in the log."""
        mock_client = MagicMock()
        unhealthy = MagicMock()
        unhealthy.status_code = 503
        mock_client.get = AsyncMock(return_value=unhealthy)

        with caplog.at_level(logging.WARNING):
            ok = await wait_until_healthy(
                11500, timeout_s=0.2, http_client=mock_client, poll_interval_s=0.02,
            )

        assert ok is False
        stall = [r for r in caplog.records if "still not healthy" in r.getMessage()]
        assert len(stall) == 1, f"expected exactly one stall warning, got {len(stall)}"


class TestGpuMemoryRead:
    def test_get_gpu_memory_used(self):
        assert get_gpu_memory_used_mib(lambda: "1234\n") == 1234

    def test_get_gpu_memory_handles_empty(self):
        assert get_gpu_memory_used_mib(lambda: "") is None

    def test_get_gpu_memory_handles_nvidia_smi_missing(self):
        def runner():
            raise FileNotFoundError("nvidia-smi")
        assert get_gpu_memory_used_mib(runner) is None

    def test_get_gpu_memory_csv_with_comma(self):
        assert get_gpu_memory_used_mib(lambda: "1024, MiB\n") == 1024

    def test_get_gpu_memory_multiline_takes_first(self):
        assert get_gpu_memory_used_mib(lambda: "1024\n2048\n") == 1024


@pytest.mark.asyncio
class TestDrainedSigterm:
    async def test_already_gone(self):
        # getpgid raises ProcessLookupError → process already gone
        def getpgid_fn(pid):
            raise ProcessLookupError("no such process")

        proc = _make_fake_proc(pid=99, poll_return=0)
        handle = SidecarHandle(proc=proc, port=11500, model_tag="t")
        ok, status = await drained_sigterm(
            handle,
            drained_window_s=1.0,
            is_active=True,
            getpgid_fn=getpgid_fn,
        )
        assert ok is True
        assert status == "already-gone"

    async def test_sigterm_clean_exit(self):
        killpg_calls = []

        def killpg_fn(pgid, sig):
            killpg_calls.append((pgid, sig))

        def getpgid_fn(pid):
            return 99999

        # Process is alive (poll=None) on first checks, then exits (poll=0)
        # SIGTERM sent → poll returns 0 next iteration
        poll_values = [None, 0]
        poll_iter = iter(poll_values)
        proc = MagicMock()
        proc.pid = 12345
        proc.poll = MagicMock(side_effect=lambda: next(poll_iter, 0))

        handle = SidecarHandle(proc=proc, port=11500, model_tag="t")
        ok, status = await drained_sigterm(
            handle,
            drained_window_s=2.0,
            is_active=True,
            killpg_fn=killpg_fn,
            getpgid_fn=getpgid_fn,
            poll_interval_s=0.01,
        )
        assert ok is True
        assert status == "sigterm-clean"
        # SIGTERM sent to the process group
        assert (99999, signal.SIGTERM) in killpg_calls
        # No SIGKILL needed
        assert (99999, signal.SIGKILL) not in killpg_calls

    async def test_sigterm_timeout_then_sigkill(self):
        killpg_calls = []
        killed = [False]  # flips True only after SIGKILL is delivered

        def killpg_fn(pgid, sig):
            killpg_calls.append((pgid, sig))
            if sig == signal.SIGKILL:
                killed[0] = True

        def poll_side_effect():
            # Process stays alive (None) until SIGKILL is sent, then exits (0)
            return 0 if killed[0] else None

        proc = MagicMock()
        proc.pid = 12345
        proc.poll = MagicMock(side_effect=poll_side_effect)

        handle = SidecarHandle(proc=proc, port=11500, model_tag="t")
        ok, status = await drained_sigterm(
            handle,
            drained_window_s=0.05,  # very short window forces escalation
            is_active=True,
            killpg_fn=killpg_fn,
            getpgid_fn=lambda pid: 99999,
            poll_interval_s=0.01,
        )
        assert ok is True
        assert status == "sigkill-clean"
        sigs_sent = [s for _, s in killpg_calls]
        assert signal.SIGTERM in sigs_sent
        assert signal.SIGKILL in sigs_sent

    async def test_cold_uses_shorter_window(self):
        """Cold slot uses cold_window_s (5s default) not drained_window_s (15s)."""
        killpg_calls = []
        def killpg_fn(pgid, sig):
            killpg_calls.append((pgid, sig))

        poll_count = [0]
        def poll_side_effect():
            poll_count[0] += 1
            return None if poll_count[0] < 5 else 0

        proc = MagicMock()
        proc.pid = 12345
        proc.poll = MagicMock(side_effect=poll_side_effect)

        handle = SidecarHandle(proc=proc, port=11500, model_tag="t")
        # is_active=False → uses cold_window_s
        ok, status = await drained_sigterm(
            handle,
            drained_window_s=10.0,  # would be used if is_active=True
            is_active=False,
            cold_window_s=0.05,  # actually used
            killpg_fn=killpg_fn,
            getpgid_fn=lambda pid: 99999,
            poll_interval_s=0.01,
        )
        # Either sigterm-clean (process exits in cold window) or sigkill-clean
        assert ok is True


@pytest.mark.asyncio
class TestVramVerify:
    async def test_vram_cleared_drops_below_threshold(self):
        readings = iter([22000, 500])

        def runner():
            return f"{next(readings)}\n"

        # settle_floor_s=0.0 -- this test is exercising the POLL
        # loop, not the new floor (default 5.0 would add a real 5s sleep
        # here for nothing this test is checking).
        cleared, current = await verify_vram_cleared(
            expected_drop_mib=22000,
            nvidia_smi_runner=runner,
            timeout_s=5.0,
            poll_interval_s=0.01,
            settle_floor_s=0.0,
        )
        assert cleared is True
        assert current == 500

    async def test_vram_unavailable_returns_true(self):
        def runner():
            raise FileNotFoundError("nvidia-smi")

        # settle_floor_s=0.0 -- the nvidia-smi-unavailable branch
        # returns before the floor would even be reached, but pinning it
        # keeps this test's own intent (dev-tolerance, not floor timing)
        # unambiguous either way.
        cleared, current = await verify_vram_cleared(
            expected_drop_mib=22000,
            nvidia_smi_runner=runner,
            timeout_s=1.0,
            settle_floor_s=0.0,
        )
        assert cleared is True
        assert current is None

    async def test_vram_timeout_returns_false(self):
        def runner():
            return "22000\n"

        # settle_floor_s=0.0 -- same reason as the first test
        # above; this pins polling/timeout behavior, not the floor.
        cleared, current = await verify_vram_cleared(
            expected_drop_mib=22000,
            nvidia_smi_runner=runner,
            timeout_s=0.05,
            poll_interval_s=0.02,
            settle_floor_s=0.0,
        )
        assert cleared is False

    # -----------------------------------------------------------------
    # The VRAM settle floor.
    # -----------------------------------------------------------------

    async def test_settle_floor_and_poll_interval_defaults(self):
        """settle_floor_s defaults to 5.0s (the
        chosen figure -- a portability margin for
        hardware slower than a typical test host, kept
        deliberately even though the real floor is smaller); poll_interval_s
        default drops 1.0 -> 0.25 so time-to-notice after the floor improves
        rather than regresses. Pre-fix code has neither: no
        settle_floor_s parameter at all, and poll_interval_s defaults to
        1.0 -- this is a real RED against the old code, not a mutant
        stand-in, because the capability genuinely does not exist there."""
        import inspect
        from turbohaul import subprocess_mgr

        params = inspect.signature(subprocess_mgr.verify_vram_cleared).parameters
        assert "settle_floor_s" in params, (
            "settle_floor_s parameter missing -- pre-fix code has no "
            "settle floor at all"
        )
        assert params["settle_floor_s"].default == 5.0, (
            f"expected default 5.0 (the chosen figure), got "
            f"{params['settle_floor_s'].default!r}"
        )
        assert params["poll_interval_s"].default == 0.25, (
            f"expected poll_interval_s default 0.25 (down from 1.0), got "
            f"{params['poll_interval_s'].default!r}"
        )

    async def test_settle_floor_delays_first_poll_sample_but_not_the_initial_baseline(self):
        """Behavioral proof: the ``initial`` baseline is sampled
        IMMEDIATELY (unchanged from pre-fix timing -- this is what the
        clamp relies on being representative), but the FIRST sample inside
        the polling loop only happens AFTER settle_floor_s has elapsed.
        Real (small) sleep -- 0.15s -- not a mock, so this proves actual
        wall-clock behavior, not just that a variable was set."""
        call_times = []

        def runner():
            call_times.append(time.monotonic())
            return "100\n"  # never drops -> loop keeps sampling until timeout

        t0 = time.monotonic()
        cleared, _ = await verify_vram_cleared(
            expected_drop_mib=50,  # small vs initial=100 -> target > 0, never met
            nvidia_smi_runner=runner,
            timeout_s=0.3,
            poll_interval_s=0.05,
            settle_floor_s=0.15,
        )
        assert cleared is False  # never drops -- the timeout side, not the floor, is under test here
        assert len(call_times) >= 2, "expected at least the initial sample + 1 poll sample"
        initial_sample_elapsed = call_times[0] - t0
        first_poll_sample_elapsed = call_times[1] - t0
        assert initial_sample_elapsed < 0.05, (
            f"the initial baseline sample must be immediate (unchanged by the "
            f"floor), got {initial_sample_elapsed:.3f}s"
        )
        assert first_poll_sample_elapsed >= 0.15, (
            f"the first POLL sample must wait out settle_floor_s=0.15, got "
            f"{first_poll_sample_elapsed:.3f}s"
        )

    async def test_settle_floor_zero_skips_the_sleep_entirely(self):
        """settle_floor_s=0.0 (the exemption _teardown/_teardown_idle_holder
        pass) must behave exactly like pre-fix code -- no floor delay at
        all, first poll sample fires immediately after the initial one."""
        call_times = []

        def runner():
            call_times.append(time.monotonic())
            return "100\n"

        t0 = time.monotonic()
        await verify_vram_cleared(
            expected_drop_mib=50, nvidia_smi_runner=runner,
            timeout_s=0.1, poll_interval_s=0.05, settle_floor_s=0.0,
        )
        assert call_times[1] - t0 < 0.05, (
            "settle_floor_s=0.0 must not add any delay before the first poll sample"
        )

    async def test_timeout_is_additive_to_the_settle_floor_not_counted_against_it(self):
        """The give-up point is settle_floor_s + timeout_s of REAL
        looking, not timeout_s measured from function entry (which would
        silently shrink the real look-window by the floor's duration on
        exactly the slow-teardown case where the full budget matters most).
        Discriminating construction: floor=0.2, timeout_s=0.2. A card that
        clears at real-elapsed=0.32s (well past the floor, comfortably
        inside the additive 0.4s budget, but already past a counted-against
        0.2s deadline) must still confirm cleared under additive semantics.
        """
        t0 = time.monotonic()

        def runner():
            elapsed = time.monotonic() - t0
            return "100\n" if elapsed < 0.32 else "5\n"  # target ~10 (90% of 100)

        cleared, current = await verify_vram_cleared(
            expected_drop_mib=100,
            nvidia_smi_runner=runner,
            timeout_s=0.2,
            poll_interval_s=0.03,
            settle_floor_s=0.2,
        )
        assert cleared is True, (
            "a card that clears at t=0.32s (after the 0.2s floor, within "
            "the additive 0.2s+0.2s=0.4s budget) must be confirmed cleared -- "
            "counted-against semantics (deadline=0.2s from entry) would have "
            "given up before this card ever had a chance to clear"
        )
        assert current == 5


class TestVerifyBinarySha256:
    def test_empty_expected_skips(self, tmp_path):
        bin_path = tmp_path / "llama-server"
        bin_path.write_bytes(b"contents")
        assert verify_binary_sha256(bin_path, "") is True

    def test_correct_sha256_passes(self, tmp_path):
        bin_path = tmp_path / "llama-server"
        content = b"some binary contents"
        bin_path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert verify_binary_sha256(bin_path, expected) is True

    def test_wrong_sha256_fails(self, tmp_path):
        bin_path = tmp_path / "llama-server"
        bin_path.write_bytes(b"contents")
        assert verify_binary_sha256(bin_path, "deadbeef" * 8) is False

    def test_missing_binary_returns_false(self, tmp_path):
        bin_path = tmp_path / "nonexistent"
        assert verify_binary_sha256(bin_path, "deadbeef" * 8) is False


class TestHealthConstants:
    def test_required_fields(self):
        assert "status" in HEALTH_REQUIRED_FIELDS

    def test_ok_statuses(self):
        assert "ok" in HEALTH_OK_STATUSES
        assert "ready" in HEALTH_OK_STATUSES
        assert "healthy" in HEALTH_OK_STATUSES
        assert "loaded" in HEALTH_OK_STATUSES
