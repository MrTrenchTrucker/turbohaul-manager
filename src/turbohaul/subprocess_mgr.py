"""Supervised subprocess management for llama-server children.

Addresses several failure modes:
- SIGTERM window too short for a large resident model; no orphan reaper.
- Orphaned Popen on parent death.
- GRACE→POPPED race + drained-SIGTERM.
- Upstream llama-server /health contract drift.

Spawn: subprocess.Popen with start_new_session=True (setsid - process group isolation).
Health: poll /health every poll_interval_s, default 60s load timeout.
Pop: drained-SIGTERM on the whole process group; killpg(SIGKILL) on timeout.
VRAM verify: nvidia-smi cross-check after POPPED before next stage.
Binary integrity: sha256 verify at boot (defense-in-depth).
"""
import asyncio
import contextlib
import hashlib
import logging
import os
import signal
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from turbohaul.engine_launch import launch_env, preset_device_mismatch


log = logging.getLogger(__name__)


# Resolve nvidia-smi to an absolute path at module load so a
# later PATH-poisoning attempt (env injection, attacker-controlled $PATH
# entry) cannot redirect the lookup at run time.
_NVIDIA_SMI_PATH = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"


# --- Observe-only env-inheritance visibility ------------------------------
#
# spawn_sidecar passes the full environment to Popen (via engine_launch,
# the full env plus at most one added key), so every engine
# still inherits this process's ENTIRE environment. That is not filtered
# here -- an allowlist
# built from any static source read cannot be certified complete (two
# independent reasons: the set the engine's own arg parser recognizes
# drifts with any vendor rebuild, with zero signal to this project, same
# fragility as the CLI-wins protection it depends on today; and the wider
# set of vars a spawned native binary needs for dynamic linking/locale/GPU
# driver init depends on the specific compiled backend and the real
# deployment environment, neither visible from source at all). Shipping a
# guessed allowlist trades a visible failure for a silent one -- a missing
# var produces a spawn failure that looks like a manifest or VRAM problem.
#
# So: classify and LOG what a candidate filter would flag, change nothing.
# This is deliberately not an allowlist and not (yet) an enforced deny.

# Every name the vendored llama-server engine's own arg parser recognizes
# via a `.set_env(...)` registration in common/arg.cpp (the vendored
# engine applies these BEFORE
# CLI args and warns "will be overwritten by command line
# argument" when a CLI flag for the same option is also
# passed). SNAPSHOT ONLY -- any engine rebuild can add,
# rename, or remove entries here with zero signal to this project. Used only to
# annotate the observation log, never to gate what spawn_sidecar passes.
ENGINE_REGISTERED_ENV_VARS: frozenset[str] = frozenset({
    'HF_TOKEN', 'LLAMA_API_KEY', 'LLAMA_ARG_ALIAS',
    'LLAMA_ARG_API_KEY_FILE', 'LLAMA_ARG_API_PREFIX',
    'LLAMA_ARG_BACKEND_SAMPLING', 'LLAMA_ARG_BATCH',
    'LLAMA_ARG_CACHE_IDLE_SLOTS', 'LLAMA_ARG_CACHE_PROMPT',
    'LLAMA_ARG_CACHE_RAM', 'LLAMA_ARG_CACHE_REUSE',
    'LLAMA_ARG_CACHE_TYPE_K', 'LLAMA_ARG_CACHE_TYPE_V',
    'LLAMA_ARG_CHAT_TEMPLATE', 'LLAMA_ARG_CHAT_TEMPLATE_FILE',
    'LLAMA_ARG_CHAT_TEMPLATE_KWARGS', 'LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT',
    'LLAMA_ARG_CONTEXT_SHIFT', 'LLAMA_ARG_CONT_BATCHING',
    'LLAMA_ARG_CPU_MOE', 'LLAMA_ARG_CTX_CHECKPOINTS', 'LLAMA_ARG_CTX_SIZE',
    'LLAMA_ARG_DEFRAG_THOLD', 'LLAMA_ARG_DEVICE', 'LLAMA_ARG_DIO',
    'LLAMA_ARG_DOCKER_REPO', 'LLAMA_ARG_DRAFT_MAX', 'LLAMA_ARG_DRAFT_MIN',
    'LLAMA_ARG_EMBEDDINGS', 'LLAMA_ARG_ENDPOINT_METRICS',
    'LLAMA_ARG_ENDPOINT_PROPS', 'LLAMA_ARG_ENDPOINT_SLOTS', 'LLAMA_ARG_FIT',
    'LLAMA_ARG_FIT_CTX', 'LLAMA_ARG_FIT_ESTIMATE', 'LLAMA_ARG_FIT_TARGET',
    'LLAMA_ARG_FLASH_ATTN', 'LLAMA_ARG_GRP_ATTN_N', 'LLAMA_ARG_GRP_ATTN_W',
    'LLAMA_ARG_HF_FILE', 'LLAMA_ARG_HF_FILE_V', 'LLAMA_ARG_HF_REPO',
    'LLAMA_ARG_HF_REPO_V', 'LLAMA_ARG_HOST', 'LLAMA_ARG_IMAGE_MAX_TOKENS',
    'LLAMA_ARG_IMAGE_MIN_TOKENS', 'LLAMA_ARG_JINJA', 'LLAMA_ARG_KV_OFFLOAD',
    'LLAMA_ARG_KV_UNIFIED', 'LLAMA_ARG_LOG_COLORS', 'LLAMA_ARG_LOG_FILE',
    'LLAMA_ARG_LOG_PREFIX', 'LLAMA_ARG_LOG_TIMESTAMPS',
    'LLAMA_ARG_LOG_VERBOSITY', 'LLAMA_ARG_MAIN_GPU', 'LLAMA_ARG_MLOCK',
    'LLAMA_ARG_MMAP', 'LLAMA_ARG_MMPROJ', 'LLAMA_ARG_MMPROJ_AUTO',
    'LLAMA_ARG_MMPROJ_OFFLOAD', 'LLAMA_ARG_MMPROJ_URL', 'LLAMA_ARG_MODEL',
    'LLAMA_ARG_MODELS_AUTOLOAD', 'LLAMA_ARG_MODELS_DIR',
    'LLAMA_ARG_MODELS_MAX', 'LLAMA_ARG_MODELS_PRESET',
    'LLAMA_ARG_MODEL_URL', 'LLAMA_ARG_MTMD_BATCH_MAX_TOKENS',
    'LLAMA_ARG_NO_HOST', 'LLAMA_ARG_NUMA', 'LLAMA_ARG_N_CPU_MOE',
    'LLAMA_ARG_N_GPU_LAYERS', 'LLAMA_ARG_N_GPU_LAYERS_DRAFT',
    'LLAMA_ARG_N_PARALLEL', 'LLAMA_ARG_N_PREDICT', 'LLAMA_ARG_OFFLINE',
    'LLAMA_ARG_OVERRIDE_TENSOR', 'LLAMA_ARG_PERF', 'LLAMA_ARG_POOLING',
    'LLAMA_ARG_PORT', 'LLAMA_ARG_PREFILL_ASSISTANT', 'LLAMA_ARG_REASONING',
    'LLAMA_ARG_REPACK', 'LLAMA_ARG_RERANKING', 'LLAMA_ARG_REUSE_PORT',
    'LLAMA_ARG_ROPE_FREQ_BASE', 'LLAMA_ARG_ROPE_FREQ_SCALE',
    'LLAMA_ARG_ROPE_SCALE', 'LLAMA_ARG_ROPE_SCALING_TYPE', 'LLAMA_ARG_RPC',
    'LLAMA_ARG_SHOW_TIMINGS', 'LLAMA_ARG_SKIP_CHAT_PARSING',
    'LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING',
    'LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K',
    'LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V', 'LLAMA_ARG_SPEC_DRAFT_CPU_MOE',
    'LLAMA_ARG_SPEC_DRAFT_HF_REPO', 'LLAMA_ARG_SPEC_DRAFT_MODEL',
    'LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE', 'LLAMA_ARG_SPEC_DRAFT_N_MAX',
    'LLAMA_ARG_SPEC_DRAFT_N_MIN', 'LLAMA_ARG_SPEC_DRAFT_P_MIN',
    'LLAMA_ARG_SPEC_DRAFT_P_SPLIT', 'LLAMA_ARG_SPEC_TYPE',
    'LLAMA_ARG_SPLIT_MODE', 'LLAMA_ARG_SSE_PING_INTERVAL',
    'LLAMA_ARG_SSL_CERT_FILE', 'LLAMA_ARG_SSL_KEY_FILE',
    'LLAMA_ARG_STATIC_PATH', 'LLAMA_ARG_SWA_FULL', 'LLAMA_ARG_TAGS',
    'LLAMA_ARG_TENSOR_SPLIT', 'LLAMA_ARG_THINK', 'LLAMA_ARG_THINK_BUDGET',
    'LLAMA_ARG_THINK_BUDGET_MESSAGE', 'LLAMA_ARG_THREADS',
    'LLAMA_ARG_THREADS_HTTP', 'LLAMA_ARG_TIMEOUT', 'LLAMA_ARG_TOOLS',
    'LLAMA_ARG_TOP_K', 'LLAMA_ARG_UBATCH', 'LLAMA_ARG_UI',
    'LLAMA_ARG_UI_CONFIG', 'LLAMA_ARG_UI_CONFIG_FILE',
    'LLAMA_ARG_UI_MCP_PROXY', 'LLAMA_ARG_WEBUI', 'LLAMA_ARG_WEBUI_CONFIG',
    'LLAMA_ARG_WEBUI_CONFIG_FILE', 'LLAMA_ARG_WEBUI_MCP_PROXY',
    'LLAMA_ARG_YARN_ATTN_FACTOR', 'LLAMA_ARG_YARN_BETA_FAST',
    'LLAMA_ARG_YARN_BETA_SLOW', 'LLAMA_ARG_YARN_EXT_FACTOR',
    'LLAMA_ARG_YARN_ORIG_CTX',
})

# Word components (underscore-delimited, whole-word match only -- "TOKEN"
# must not match "TOKENS", which is a token-COUNT knob, not a credential)
# that mark a var name as a deny-list CANDIDATE. Substring matching would
# wrongly catch LLAMA_ARG_API_KEY_FILE and
# LLAMA_ARG_SSL_KEY_FILE (both contain "_KEY") on top of the real credential names,
# and LLAMA_ARG_IMAGE_MAX_TOKENS / LLAMA_ARG_MTMD_BATCH_MAX_TOKENS
# (both contain "_TOKEN" as a substring of "_TOKENS") -- none of those four
# are credentials.
_DENY_WORDS: frozenset[str] = frozenset({
    "TOKEN", "SECRET", "PASSWORD", "PASS", "CREDENTIAL", "CRED", "AUTH",
    "KEY",
})


def classify_spawn_env(env: dict[str, str]) -> list[dict[str, str]]:
    """Observe-only: classify env var NAMES worth knowing about
    before a spawn. Never filters -- returns what a candidate rule WOULD
    flag and WHY, so the caller can log the name and the matching rule (a
    count alone tells you something happened; the name and rule tell the
    next person whether the rule is right).

    Two independent categories, not stages of one pipeline:
      - "engine-registered": the vendored engine's own arg parser reads this
        name (see ENGINE_REGISTERED_ENV_VARS). Informational -- today's
        protection against this silently overriding a value is a
        third-party CLI-wins behavior outside this project's control.
        STRUCTURALLY exempt from deny-candidate: membership here proves the
        var is a legitimate, engine-defined input, even when the name is
        also credential-shaped (HF_TOKEN, LLAMA_API_KEY, and, among the
        engine's own names, LLAMA_ARG_API_KEY_FILE, LLAMA_ARG_SSL_KEY_FILE
        all match _DENY_WORDS). A hand-maintained exclude list would have
        needed updating for those last two; this exemption doesn't.
      - "deny-candidate": name contains a word from _DENY_WORDS and is NOT
        engine-registered. Candidate for actual denial in a later change;
        nothing is denied here.

    Two safety properties below are load-bearing in a publicly
    visible repo and are held by nothing but this function's own body --
    correct BY CONSTRUCTION (there is no code path that could violate
    them), not by a guard that could rot. Both are pinned by
    the never-logs-a-value tests in
    tests/test_subprocess_mgr.py, which fail if either
    property is broken. A future edit must not
    reintroduce either of these without re-deriving why they were safe:

    1. NO THROWING OPERATIONS. Every operation below is total over
       arbitrary str input: str.upper() and str.split("_") accept any
       unicode content (embedded NUL, newlines, control chars, a 100k-char
       name -- none of it raises), and frozenset.intersection() /
       membership tests never raise on a hashable operand (every str is
       hashable). There is no regex, no external call, no parsing that
       could throw on hostile input -- this is why the function is safe to
       call unguarded, directly on the live spawn path, with no
       try/except around it.
    2. KEYS-ONLY ITERATION -- NEVER LOGS A VALUE. The loop below is
       `for name in env:` -- it iterates the dict's KEYS only. No line in
       this function ever reads env[name], env.values(), or env.items().
       There is no code path that could touch a value, so there is
       nothing to redact and nothing to leak -- this is what makes the
       observation safe to log at INFO in a publicly visible repo. A
       future edit that adds `.items()` or reads a value for a "richer"
       log line would silently break this.
    """
    flagged: list[dict[str, str]] = []
    for name in env:
        words = name.upper().split("_")
        is_engine_registered = name in ENGINE_REGISTERED_ENV_VARS
        matched_words = sorted(_DENY_WORDS.intersection(words))
        if is_engine_registered:
            if matched_words:
                rule = (
                    f"engine-registered, credential-shaped ({'/'.join(matched_words)}) "
                    "-- exempt from deny-candidate: this is the engine's own "
                    "legitimate input (a near-miss: a naive pattern "
                    "match on the name alone would have denied it)"
                )
            else:
                rule = (
                    "engine-registered -- collides with a CLI flag only if "
                    "we also pass the corresponding --flag; the vendored "
                    "engine warns and the CLI value wins (arg.cpp:564)"
                )
            flagged.append({"name": name, "category": "engine-registered", "rule": rule})
        elif matched_words:
            rule = f"deny-candidate: name contains {'/'.join(matched_words)!r} (credential-shaped)"
            flagged.append({"name": name, "category": "deny-candidate", "rule": rule})
    return flagged


def _log_spawn_env_observations(port: int, model_tag: str, env: dict[str, str]) -> None:
    """Both branches log at INFO, not just the flagged one. A
    DEBUG-on-empty design makes "ran, found nothing" and "never
    ran at all" produce IDENTICAL silence, so an operator cannot
    tell the two apart from the log alone and would have to infer
    it structurally (checking that a later, unconditional INFO line
    exists downstream, therefore this one must already have run).
    One spawn is not a hot path -- it
    happens once per model load/swap, not once per token -- and the very
    next line in spawn_sidecar already logs at INFO on every call
    ("spawning llama-server..."), so this matches the existing logging
    density at this exact point rather than introducing a new one. The
    alternative (leave it silent, add a periodic heartbeat instead) would
    still leave any INDIVIDUAL spawn's observer status ambiguous between
    heartbeats, which is the actual question the operator needs answered -- a
    heartbeat answers "is the observer alive," not "did THIS spawn get
    observed."
    """
    flagged = classify_spawn_env(env)
    if not flagged:
        log.info(
            "env-observe (observe-only, nothing filtered): "
            "port=%d model=%s inherited=%d flagged=0 (clean)",
            port, model_tag, len(env),
        )
        return
    details = "; ".join(f"{f['name']} [{f['category']}]: {f['rule']}" for f in flagged)
    log.info(
        "env-observe (observe-only, nothing filtered): "
        "port=%d model=%s inherited=%d flagged=%d: %s",
        port, model_tag, len(env), len(flagged), details,
    )


class HealthCheckFailed(RuntimeError):
    """llama-server failed to become healthy within loading_health_timeout_s."""


class SchemaMismatch(RuntimeError):
    """Upstream /health response shape changed unexpectedly (contract-drift defense)."""


class SidecarHandle:
    """Wraps a running llama-server subprocess + its identity."""

    def __init__(
        self,
        proc: subprocess.Popen,
        port: int,
        model_tag: str,
        parallel: int = 1,
        engine_log_path: str | None = None,
    ) -> None:
        self.proc = proc
        self.port = port
        self.model_tag = model_tag
        # Live concurrency width pinned at spawn from the actual --parallel argv.
        # The manager derives its per-model in-flight admission cap from THIS,
        # never a later manifest read (which can drift across warm-inherit reuse).
        self.parallel = parallel
        self.spawned_at = time.monotonic()
        self.activated_at: float | None = None
        # The same engine log path this spawn already writes to
        # (see _engine_log_file below) -- threaded onto the handle so a later
        # LOAD_VERIFY identity/error check can read it without having to
        # reconstruct the model_tag+port+epoch naming itself (the epoch is
        # only known here, at spawn time). None only for handles built
        # without going through spawn_sidecar (e.g. bare test doubles).
        self.engine_log_path = engine_log_path

    @property
    def pid(self) -> int:
        return self.proc.pid

    def is_alive(self) -> bool:
        return self.proc.poll() is None


SLOT_SAVE_DIR = "/var/lib/turbohaul/kvcache"
# SLOT_SAVE_DIR is a tmpfs (RAM) — per-turn saves = zero SSD
# wear. SLOT_PERSIST_DIR is the SSD archive the unload flush copies the clean bin
# (+ .ckpt sidecar + meta) into, so a controlled swap/idle warm-reloads from SSD.
SLOT_PERSIST_DIR = "/var/lib/turbohaul/kvcache_persist"


def spawn_sidecar(
    binary: Path,
    gguf_path: Path,
    port: int,
    model_tag: str,
    argv_flags: list[str],
    popen_factory: Callable[..., subprocess.Popen] | None = None,
    binary_fd: int | None = None,
) -> SidecarHandle:
    """Spawn a llama-server child in its own process group (setsid).

    popen_factory exists for test injection. Default = subprocess.Popen.
    """
    os.makedirs(SLOT_SAVE_DIR, exist_ok=True)
    # Observe-only -- this logs what a candidate rule would flag
    # and filters nothing. The Popen call below passes env=launch_env(...)
    # -- which is the full environment plus at most one added key.
    _log_spawn_env_observations(port, model_tag, dict(os.environ))
    factory = popen_factory or subprocess.Popen
    # If a pinned fd is provided, exec via /proc/self/fd/<fd>
    # so the inode we hashed at boot is exactly what we exec; the path could
    # have been swapped after verify, but the fd still points to the right
    # inode. Falls back to path-based exec when binary_fd is None (dev mode
    # / empty sha256).
    if binary_fd is not None:
        exec_path = f"/proc/self/fd/{binary_fd}"
        pass_fds: tuple[int, ...] = (binary_fd,)
    else:
        exec_path = str(binary)
        pass_fds = ()
    # Observability: capture engine logs (prefill n_tokens, restored
    # n_past, timings) to a per-model file via llama-server's own --log-file.
    # Keeps stdout/stderr=DEVNULL intact (the pipe-buffer contract below), so this
    # adds visibility into KV restore/reuse without re-introducing PIPE.
    _eng_log_dir = os.path.join(os.path.dirname(SLOT_SAVE_DIR), "engine_logs")
    try:
        os.makedirs(_eng_log_dir, exist_ok=True)
    except OSError:
        pass
    _safe_model = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in str(model_tag))
    _engine_log_file = os.path.join(_eng_log_dir, f"engine_{_safe_model}_p{port}_{int(time.time())}.log")
    cmd = [
        exec_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "-m", str(gguf_path),
        "--slot-save-path", SLOT_SAVE_DIR,
        "--log-file", _engine_log_file,
        *argv_flags,
    ]
    log.info(
        "spawning llama-server pid=? port=%d model=%s pinned_fd=%s",
        port, model_tag, "yes" if binary_fd is not None else "no",
    )
    # stdout/stderr to DEVNULL — PIPE without an active drainer
    # fills the 64KB OS pipe buffer once llama-server emits enough log lines
    # (model load + slot ops + per-token perf), at which point write(2) blocks
    # inside the child and the drained-SIGTERM contract no longer holds.
    # llama-server has its own --log-file argv option if structured log capture
    # is required; wire it via argv_flags rather than re-introducing PIPE here.
    # stdout/stderr -> append FILE (rather than DEVNULL). Captures the
    # GGML abort output that bypasses --log-file (the abort callback prints
    # CMD_CHILD_TO_ROUTER_ERROR to stdout, which DEVNULL would discard on a poisoned-bin crash).
    # A regular file never back-pressures like the 64KB pipe, so the
    # drained-SIGTERM contract is preserved. Best-effort: fall back to DEVNULL.
    try:
        _stdio_f = open(_engine_log_file + ".stdio", "ab")
    except Exception:
        _stdio_f = None
    # engine_launch: the projector's device, added on top of
    # the full, unfiltered environment above. launch_env and
    # preset_device_mismatch are pure (no I/O); this is the one
    # call site that acts on their output -- applying the env and logging the warning.
    _engine_env = launch_env(cmd, os.environ)
    _preset_warning = preset_device_mismatch(cmd, os.environ)
    if _preset_warning is not None:
        log.warning(_preset_warning)
    proc = factory(
        cmd,
        env=_engine_env,
        stdout=_stdio_f if _stdio_f is not None else subprocess.DEVNULL,
        stderr=_stdio_f if _stdio_f is not None else subprocess.DEVNULL,
        start_new_session=True,  # setsid - own process group → killpg works
        pass_fds=pass_fds,
    )
    if _stdio_f is not None:
        try:
            _stdio_f.close()  # child holds its own dup; release ours
        except Exception:
            pass
    # Pin the live --parallel width from the spawn argv (default 1 when absent).
    # Single source of truth for the manager's in-flight admission cap.
    parallel = 1
    for i, tok in enumerate(argv_flags):
        if tok == "--parallel" and i + 1 < len(argv_flags):
            try:
                parallel = max(1, int(argv_flags[i + 1]))
            except (TypeError, ValueError):
                parallel = 1
            break
    return SidecarHandle(
        proc=proc, port=port, model_tag=model_tag, parallel=parallel,
        engine_log_path=_engine_log_file,
    )


# Defense-in-depth schema check for the upstream /health endpoint.
# If the upstream changes the response shape, we want a loud failure not silent
# health-pass.
HEALTH_REQUIRED_FIELDS: set[str] = {"status"}
HEALTH_OK_STATUSES: set[str] = {"ok", "ready", "healthy", "loaded"}


async def health_check_once(port: int, http_client: httpx.AsyncClient) -> dict | None:
    """One health probe. Returns parsed JSON on 200, None on non-200 / network error.

    Raises SchemaMismatch if the response shape is unexpectedly different from
    HEALTH_REQUIRED_FIELDS - this is intentional load-bearing visibility for
    upstream /health contract drift.
    """
    try:
        r = await http_client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
    except (httpx.HTTPError, OSError):
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        raise SchemaMismatch(f"health response not a dict: {type(data).__name__}")
    missing = HEALTH_REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise SchemaMismatch(f"health response missing fields: {missing}")
    return data


async def wait_until_healthy(
    port: int,
    timeout_s: float,
    http_client: httpx.AsyncClient | None = None,
    poll_interval_s: float = 2.0,
    is_alive: Callable[[], bool] | None = None,
) -> bool:
    """Poll /health until 200+ok or timeout. Returns True on healthy, False on timeout.

    is_alive: optional liveness probe for the spawned child (SidecarHandle.is_alive).
    When supplied, a child that has EXITED (is_alive() is False) fails fast with
    False instead of burning the full timeout_s (the FSM-wedge fix). LIVENESS-ONLY:
    a slow-but-alive cold load (poll() is None) never trips this, so a legitimate
    large-context load is not killed. Default None keeps existing callers unchanged."""
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient()
    _warned_slow_load: bool = False  # half-way stall warning (per-wait)
    try:
        started_at = time.monotonic()
        deadline = started_at + timeout_s
        while time.monotonic() < deadline:
            try:
                data = await health_check_once(port, http_client)
            except SchemaMismatch:
                # Re-raise so the manager can surface this as a schema-drift event
                raise
            if data is not None:
                status = (data.get("status") or "").lower()
                if status in HEALTH_OK_STATUSES:
                    return True
            # FSM-wedge fix: bail fast if the spawned child has already exited
            # (crash / OOM / bad-flag). Checked AFTER the health probe so a child
            # that turns healthy in the same tick still wins; runs every iteration.
            if is_alive is not None and not is_alive():
                log.warning(
                    "wait_until_healthy: child for port %d exited during load "
                    "(is_alive=False) - failing fast instead of waiting out timeout",
                    port,
                )
                return False
            # Half-way diagnostic: a load that has not become healthy by the
            # MIDPOINT of its timeout is probably stuck, or is a model much
            # larger than this timeout was sized for. Emit ONE warning per wait
            # so operators see the hang in the log while there is still time to
            # look; the timeout itself still governs the failure.
            # The midpoint test is what keeps this a signal: without it the
            # warning fires on the first poll of every load, including healthy
            # ones, and becomes noise operators learn to ignore.
            elapsed = time.monotonic() - started_at
            if not _warned_slow_load and elapsed >= timeout_s / 2:
                _warned_slow_load = True
                log.warning(
                    "wait_until_healthy: model still not healthy after %.1fs "
                    "for port %d (timeout=%.1fs) - check the engine process "
                    "and GPU memory for a stalled load",
                    elapsed, port, timeout_s,
                )
            await asyncio.sleep(poll_interval_s)
        return False
    finally:
        if own_client:
            await http_client.aclose()


def _default_nvidia_smi_runner(device_index: int = 0) -> str:
    return subprocess.check_output(
        [
            _NVIDIA_SMI_PATH,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i", str(device_index),
        ],
        text=True,
        timeout=5,
    )


def get_gpu_memory_used_mib(
    nvidia_smi_runner: Callable[..., str] | None = None,
    device_index: int = 0,
) -> int | None:
    """Return GPU ``device_index`` `memory.used` in MiB (default 0). None if
    nvidia-smi unavailable. ``device_index`` is ignored when an explicit
    ``nvidia_smi_runner`` is supplied (the caller's runner decides what it
    samples: real per-card sampling only applies to the
    DEFAULT runner; injected test/fake runners keep their existing zero-arg
    signature unchanged)."""
    runner = nvidia_smi_runner or (lambda: _default_nvidia_smi_runner(device_index))
    try:
        out = runner()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None
    try:
        return int(line.strip().split(",")[0].strip())
    except (ValueError, IndexError):
        return None


async def drained_sigterm(
    handle: SidecarHandle,
    drained_window_s: float,
    is_active: bool,
    cold_window_s: float = 5.0,
    killpg_fn: Callable[[int, int], None] | None = None,
    getpgid_fn: Callable[[int], int] | None = None,
    poll_interval_s: float = 0.2,
) -> tuple[bool, str]:
    """SIGTERM the whole process group → wait → SIGKILL on timeout.

    For active slots (is_active=True), use drained_window_s (default 15s
    to allow in-flight decode to complete cleanly on 21GB-resident llama-server).
    For cold/IDLE_HOT slots, use cold_window_s (default 5s).

    Returns (success, status_str). Status strings:
      - "already-gone"   process already exited before SIGTERM
      - "sigterm-clean"  exited during drained window
      - "sigkill-clean"  needed SIGKILL escalation
      - "sigterm-failed-*", "sigkill-permission-denied", "sigkill-failed-still-alive"

    killpg_fn / getpgid_fn allow test injection.
    """
    killpg = killpg_fn or os.killpg
    getpgid = getpgid_fn or os.getpgid

    wait_window = drained_window_s if is_active else cold_window_s

    try:
        pgid = getpgid(handle.pid)
    except ProcessLookupError:
        return True, "already-gone"

    try:
        killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        # Best-effort waitpid on already-gone — reaps zombie
        # if process exited between getpgid and killpg.
        try:
            await asyncio.to_thread(handle.proc.wait, timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        return True, "already-gone-during-sigterm"
    except PermissionError as e:
        return False, f"sigterm-failed-permission-denied"

    deadline = time.monotonic() + wait_window
    while time.monotonic() < deadline:
        if handle.proc.poll() is not None:
            # Explicit waitpid reap. poll() already reaps if
            # exited, but be defensive — the zombie-free invariant that
            # the orphan reaper depends on must be guaranteed here.
            try:
                await asyncio.to_thread(handle.proc.wait, timeout=2.0)
            except subprocess.TimeoutExpired:
                pass  # benign — process exited per poll(), waitpid race
            return True, "sigterm-clean"
        await asyncio.sleep(poll_interval_s)

    # Escalate to SIGKILL
    try:
        killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True, "sigkill-already-gone"
    except PermissionError:
        return False, "sigkill-permission-denied"

    # Brief final settle
    await asyncio.sleep(0.5)
    if handle.proc.poll() is None:
        return False, "sigkill-failed-still-alive"
    # Explicit waitpid reap. Without this, kernel keeps the
    # PID slot occupied as <defunct>; any helper the engine spawned that
    # reparents inherits the zombie as PPid, slipping the PPid==1 reaper
    # filter (false-negative).
    try:
        await asyncio.to_thread(handle.proc.wait, timeout=5.0)
    except subprocess.TimeoutExpired:
        return False, "wait-timeout-after-sigkill"
    return True, "sigkill-clean"


async def verify_vram_cleared(
    expected_drop_mib: int,
    nvidia_smi_runner: Callable[..., str] | None = None,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.25,
    device_index: int = 0,
    settle_floor_s: float = 5.0,
) -> tuple[bool, int | None]:
    """After POPPED, poll until VRAM drops by ≥90% of expected.

    Defends against the CUDA-allocator-stuck failure mode.

    Returns (cleared_ok, current_used_mib).
    nvidia_smi_runner=None and unavailable → returns (True, None) (dev tolerance).
    ``device_index`` samples the victim's OWN card (default 0
    preserves prior single-GPU behavior); ignored when ``nvidia_smi_runner`` is
    explicitly supplied, same rule as ``get_gpu_memory_used_mib``.

    ``settle_floor_s`` is a ONE-TIME sleep taken once, right after the
    ``initial`` baseline sample, before the polling loop below starts. VRAM does
    not physically begin dropping for >2s after a model unload
    — every sample inside that window is a guaranteed miss, so
    skipping them is free. NOT paying for that skip in admission latency means
    ``timeout_s`` stays ADDITIVE to the floor: the deadline below is computed
    AFTER the floor sleep, not before, so the give-up point is still a real
    ``timeout_s`` of actual looking, just shifted later by the floor rather than
    shrunk by it. ``poll_interval_s`` defaults to a short 0.25 for the same
    reason: a single sample is cheap (tens of ms), so polling faster
    after the floor buys back time-to-notice instead of losing it. Pass
    ``settle_floor_s=0.0`` for a call site whose result does not gate a waiting
    admission (see manager.py's ``_teardown``/``_teardown_idle_holder``) — there
    the floor is pure added latency with no faster-notice benefit to offset it.

    ``expected_drop_mib`` may be a MULTI-card model's total
    credit while ``device_index`` scopes every sample to ONE card (e.g.,
    a large split model, credited in full, verified on a
    card that only ever held part of it). Unclamped, ``target`` would go
    negative and be floored to 0 -- and a live CUDA context never reads
    exactly 0 (its floor is a few MiB) -- so the predicate
    would be UNSATISFIABLE: every split-card eviction would poll the full ``timeout_s``
    and report not-cleared even when the card genuinely emptied in under a
    second. Clamping the expected drop to ``initial`` (this card's own actual
    reading) expresses the correct, ACHIEVABLE claim -- "we expect at least a
    90% drop of whatever THIS card actually held" -- instead of an impossible
    one. It is still a real check: a card that releases less than 90% of what
    IT held still correctly fails, clamp or no clamp.
    """
    initial = get_gpu_memory_used_mib(nvidia_smi_runner, device_index)
    if initial is None:
        return True, None  # nvidia-smi unavailable — trust the kill (dev mode)
    if settle_floor_s:
        await asyncio.sleep(settle_floor_s)
    clamped_expected_drop_mib = min(expected_drop_mib, initial)
    target = max(0, initial - int(clamped_expected_drop_mib * 0.9))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = get_gpu_memory_used_mib(nvidia_smi_runner, device_index)
        if current is None:
            return True, None
        if current <= target:
            return True, current
        await asyncio.sleep(poll_interval_s)
    return False, get_gpu_memory_used_mib(nvidia_smi_runner, device_index)


def verify_binary_sha256(binary_path: Path, expected_sha256: str) -> bool:
    """Verify llama-server binary sha256 matches the pinned value.

    Empty expected_sha256 = skip verify (dev mode).
    """
    if not expected_sha256:
        return True
    if not binary_path.exists():
        return False
    h = hashlib.sha256()
    with binary_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def open_and_verify_binary(binary_path: Path, expected_sha256: str) -> int | None:
    """Open the llama-server binary fd at boot + verify sha256 via that fd.

    Returns:
      - the open fd (caller must keep it alive for process lifetime) on hash match
      - None on empty expected_sha256 (dev mode, no pinning needed)
      - None on hash mismatch or path missing

    Pairs with ``spawn_sidecar(..., binary_fd=fd)`` which execs via
    ``/proc/self/fd/<fd>``. Because the fd points at a specific inode that has
    already been hashed, any later attacker-write to ``binary_path`` cannot
    redirect the spawn to a different binary -- even if the path is overwritten,
    renamed, or replaced. The TOCTOU window between hash-check and exec is
    closed at the kernel level (inode pin via open fd).

    O_CLOEXEC is intentionally NOT set: the fd must survive fork+exec so the
    child can resolve ``/proc/self/fd/<fd>`` at exec time. subprocess.Popen
    ``pass_fds`` keeps the fd inheritable across its close-fds sweep.
    """
    if not expected_sha256:
        return None  # dev mode -- no pinning needed
    if not binary_path.exists():
        return None
    fd = os.open(str(binary_path), os.O_RDONLY)
    try:
        h = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        if h.hexdigest() != expected_sha256:
            os.close(fd)
            return None
        # Reset offset (defensive; exec doesn't care).
        os.lseek(fd, 0, 0)
        return fd
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
