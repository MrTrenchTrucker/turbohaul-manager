"""Chat-completion API routes - Ollama-compat + OpenAI-compat.

This module ships the non-streaming completion path. Streaming SSE comes in a
future polish pass; the existing manager.submit_and_wait + completion_fn DI is
streaming-ready (just return an async generator from completion_fn and adapt
the route).

The completion_fn is wired into TurbohaulManager via DI. Production uses
make_llama_server_complete_fn() which httpx-POSTs to the spawned llama-server's
/v1/chat/completions on its assigned port. Tests inject a fake completion_fn
that returns a canned response without spawning anything real.

Typed upstream errors. A sidecar OOM-crash during inference shows up as
RemoteProtocolError, which is a likelier failure than a
HTTPStatusError 4xx. These
need different client-facing status codes:
  - 503 Service Unavailable + Retry-After  → sidecar disconnected / crashed
  - 502 Bad Gateway                         → sidecar returned upstream 4xx/5xx
  - 504 Gateway Timeout                     → request timed out at sidecar
  - 500 Internal Server Error               → genuine Turbohaul bug (fallback)
  - 422 RESERVED for input-validation only (NOT used for upstream errors)
"""
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from turbohaul.config import KEEP_ALIVE_MAX_S  # re-exported for tests + manager
from turbohaul.kv_policy import (
    compute_ctx_len,       # shared admission/save size rule
    _prefix_hash_chain,    # shared admission/save turn-hash chain (unified)
)
from turbohaul.live_monitor import (  # live-monitor text tee identity
    compute_generation_id,
    _read_spawn_seq,
)
from turbohaul.manifest import (  # handler-entry manifest read for thinking detect
    ManifestValidationError,
    read_manifest,
)
from turbohaul.slot import SlotEvictedError, VramOverCommitError, derive_thread_id_prefix_hash  # eviction + VRAM over-commit errors, thread-id prefix-hash fingerprint
from turbohaul.api.tool_call_recovery import maybe_recover_tool_calls  # text-JSON tool-call recovery


# === client-disconnect watcher =======================
# Constant 2s cadence + direct request.is_disconnected() call, with no
# extra wrapper. Wrapping is_disconnected in asyncio.wait_for can leak
# the underlying ASGI receive() coroutine on cancellation; Starlette already
# implements is_disconnected as non-blocking-fast, so the wrapper is both
# unnecessary and harmful.
_DISCONNECT_POLL_INTERVAL_S = 2.0


async def watch_disconnect(
    request: Request,
    disconnect_event: asyncio.Event,
) -> None:
    """Poll ``request.is_disconnected()`` every ~2s; signal ``disconnect_event``
    when the client closes the connection.

    Constant cadence, direct
    poll, no ``asyncio.wait_for`` wrap. Tolerates transient ASGI receive()
    errors (returns True is the only state we care about).
    """
    while not disconnect_event.is_set():
        try:
            if await request.is_disconnected():
                disconnect_event.set()
                return
        except Exception:
            # Transient ASGI receive() errors are non-fatal; keep polling.
            # On py3.11 CancelledError is BaseException, so this
            # already never catches KI/SE/Cancel — narrowing would be a regression.
            pass
        await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)


log = logging.getLogger(__name__)
router = APIRouter()

# read_manifest (manifest.py) raises SIX classes from three libraries
# plus the stdlib. `except ManifestValidationError` alone does NOT cover
# pydantic's ValidationError -- it is a SIBLING under ValueError, not a
# parent -- so a schema-invalid manifest would otherwise surface from openai_chat_completions and
# ollama_chat as a bare, unhandled 500. Restated file-local rather than
# imported (same file-local choice api/ollama.py already
# documents, api/models.py / api/plugins.py / api/manifests.py / api/ollama.py
# / api/exec_ws.py each restate their own copy too) -- the drift test
# (covering manifest-decode blast radius) keeps them all in sync by
# assertion instead.
_UNREADABLE_MANIFEST = (
    FileNotFoundError,
    ManifestValidationError,
    yaml.YAMLError,
    ValidationError,
    OSError,
    UnicodeDecodeError,
)


# === JSON Schema validation constants ===================
# Validator-side DoS guards — caller-supplied schemas are size/shape-bounded
# BEFORE we compile via jsonschema.
_SCHEMA_MAX_BYTES = 65536  # 64 KiB serialized schema cap
_SCHEMA_MAX_DEPTH = 16  # recursive object/array nesting cap
_SCHEMA_MAX_PROPERTIES = 64  # total property count across the schema tree
_BODY_MAX_BYTES = 4194304  # 4 MiB total request body cap (out-of-scope guard hook)
_COMPILE_TIMEOUT_SEC = 0.5  # Draft202012Validator construction budget


def _schema_depth(node: Any, depth: int = 0) -> int:
    """Recursive max-nesting walker for object/array structures in a schema."""
    if depth > _SCHEMA_MAX_DEPTH:
        return depth
    if isinstance(node, dict):
        return max(
            [depth] + [_schema_depth(v, depth + 1) for v in node.values()]
        )
    if isinstance(node, list):
        return max(
            [depth] + [_schema_depth(item, depth + 1) for item in node]
        )
    return depth


def _schema_property_count(node: Any) -> int:
    """Total property-name count across the schema. Walks dict + list."""
    if isinstance(node, dict):
        local = len(node.get("properties", {})) if isinstance(node.get("properties"), dict) else 0
        return local + sum(_schema_property_count(v) for v in node.values())
    if isinstance(node, list):
        return sum(_schema_property_count(item) for item in node)
    return 0


def _schema_has_ref(node: Any) -> bool:
    """Reject ANY $ref for MVP — eliminates cycle + remote-fetch attack surface."""
    if isinstance(node, dict):
        if "$ref" in node:
            return True
        return any(_schema_has_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_schema_has_ref(item) for item in node)
    return False


def _schema_missing_additional_properties_guard(node: Any) -> bool:
    """Object schemas MUST set additionalProperties: false (avoids implicit-anything)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            return True
        return any(_schema_missing_additional_properties_guard(v) for v in node.values())
    if isinstance(node, list):
        return any(_schema_missing_additional_properties_guard(item) for item in node)
    return False


def _validate_json_schema(rf: dict) -> tuple[bool, str | None]:
    """Validate a caller-supplied json_schema response_format.

    Returns ``(ok, reason)``. On not-ok, ``reason`` is a short machine-readable
    string the caller surfaces in the HTTP 422 error body. On ok, the schema is
    safe to forward to llama-server and (later, in `_complete`) to use for
    Draft202012Validator.validate against the model's returned JSON.

    The jsonschema import is LAZY inside the function body — fail-soft
    against the dependency being absent in a not-yet-rebuilt environment
    (a runtime pip-install may precede the image rebuild).

    Note: `rf` is the FULL response_format dict; the schema lives at
    rf["json_schema"]["schema"] per the OpenAI structured-outputs envelope.
    """
    try:
        from jsonschema import Draft202012Validator  # noqa: F401  (compile only)
    except ImportError:
        return (False, "jsonschema_lib_unavailable")

    if not isinstance(rf.get("json_schema"), dict):
        return (False, "missing_or_malformed_json_schema_field")
    schema = rf["json_schema"].get("schema")
    if not isinstance(schema, dict):
        return (False, "missing_or_malformed_schema_field")

    # Size check
    try:
        schema_bytes = len(json.dumps(schema))
    except (TypeError, ValueError):
        return (False, "schema_not_json_serializable")
    if schema_bytes > _SCHEMA_MAX_BYTES:
        return (False, f"schema_size_exceeded:{schema_bytes}")

    # Depth check
    if _schema_depth(schema) > _SCHEMA_MAX_DEPTH:
        return (False, "schema_depth_exceeded")

    # Property count check
    if _schema_property_count(schema) > _SCHEMA_MAX_PROPERTIES:
        return (False, "schema_property_count_exceeded")

    # $ref rejection (no cycle / no remote fetch)
    if _schema_has_ref(schema):
        return (False, "schema_contains_ref_unsupported")

    # additionalProperties guard on object schemas
    if _schema_missing_additional_properties_guard(schema):
        return (False, "schema_missing_additionalProperties_guard")

    # Compile attempt (synchronous; bounded by _COMPILE_TIMEOUT_SEC budget upstream)
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator(schema)
    except Exception as e:  # noqa: BLE001 — compile errors are caller's input fault
        return (False, f"schema_compile_failed:{type(e).__name__}")

    return (True, None)


def is_thinking_payload(payload: dict, manifest: dict) -> bool:
    """Thinking-mode detection (manifest-only simplification).

    `chat_template_kwargs` lives in manifest.py DENIED_FLAGS so a payload-side
    check would always be False (the field never reaches the outgoing request).
    Detection is manifest-only via reasoning_budget > 0. The `payload` arg is
    present for interface parity with the spec; not consulted.

    `manifest` is dict-shaped; callers pass `Manifest.llama_server_flags` (or
    any dict carrying a `reasoning_budget` key).
    """
    rb = manifest.get("reasoning_budget", 0)
    try:
        return int(rb) > 0
    except (TypeError, ValueError):
        return False


# === reasoning_budget vs. effective-ceiling request-time warning ==
# Bounded warn-once-per-(model, caller) latch. Same idiom as manager.py's
# _shadow_bytematch_probe / _shadow_reprefill_locks / tools-fingerprint stash:
# a SHA-256-truncated key + fixed cap + pop-then-reinsert-on-touch, so a
# caller-influenced key can never leak across a long-running process. A
# per-request log on this hot path would itself be an outage risk -- this
# guarantees at most one line per (model, caller) for the life of the process.
_REASONING_BUDGET_WARN_CAP = 64
_reasoning_budget_warned: dict[str, None] = {}


def _reasoning_budget_warn_key(model: str, caller: str) -> str:
    """SHA-256-truncated-to-16-hex dedup key, mirroring manifest._thread_hash's
    convention elsewhere in this codebase for turning an arbitrary-length,
    caller-influenced string into a short fixed-width dict key.
    """
    return hashlib.sha256(f"{model}:{caller}".encode("utf-8")).hexdigest()[:16]


def _effective_reasoning_ceiling(
    manifest_flags: dict, caller_max_tokens: Any,
) -> tuple[int | None, str | None]:
    """The effective output ceiling is whichever of the
    manifest's n_predict (if bounded) and the caller's max_tokens (if given)
    is smaller; n_predict <= 0 (unbounded) drops out of that comparison
    entirely, so an unbounded manifest alone never trips this. Returns
    (ceiling, source) with source in {"manifest", "caller", None}.

    Shared by warn_if_reasoning_budget_exceeds_ceiling and
    clamp_reasoning_budget_for_ceiling so the two can never
    disagree about what counts as pathological -- one computation, two
    consumers.
    """
    ceiling = None
    ceiling_source = None
    n_predict = manifest_flags.get("n_predict", -1)
    if isinstance(n_predict, int) and not isinstance(n_predict, bool) and n_predict > 0:
        ceiling = n_predict
        ceiling_source = "manifest"
    if (
        isinstance(caller_max_tokens, int)
        and not isinstance(caller_max_tokens, bool)
        and caller_max_tokens > 0
        and (ceiling is None or caller_max_tokens < ceiling)
    ):
        ceiling = caller_max_tokens
        ceiling_source = "caller"
    return ceiling, ceiling_source


# Request-time clamp so a pathological
# reasoning_budget leaves an answer floor instead of consuming the whole
# ceiling. Floor = half the effective ceiling -- the project's own
# "half-of-max_tokens for half-budget" manifest convention
# (docs/MODEL_CONFIG_REFERENCE.md, "leaves the other half of the
# token budget for the actual answer, so a thinking model doesn't exhaust
# its window inside <think>" -- existing documented guidance).
# This enforces that existing rule at request time where nothing enforced it
# before; every bounded manifest already sits at or below that ratio
# (typically well below half).
#
# Bounds, each independently justified (a stated judgement, not a magic
# number):
#   _MAX_ANSWER_FLOOR_TOKENS = 2048 -- re-uses the project's OWN already-
#     documented threshold for "enough room that a thinking model's answer
#     doesn't get starved": docs/AI_AGENT_SETUP.md AND
#     docs/MODEL_CONFIG_REFERENCE.md BOTH independently say to
#     keep a thinking-model request's max_tokens >= ~2000 for this exact
#     reason. Capping the floor there means it never reserves more than the
#     project already treats as "plenty," even against a very generous
#     ceiling (e.g. a 35B model's 24576 budget against a 20000-token
#     caller cap: half would be 10000, capped to 2048 -- still a solid
#     answer allowance without eating half the caller's thinking headroom
#     just because they asked for a lot of room).
#   _MIN_ANSWER_FLOOR_TOKENS = 32 -- judgement call, no equivalent doc
#     citation exists for this one. Chosen because it sits well below the
#     smallest max_tokens this codebase's own docs already treat as a
#     legitimate (if terse) real answer -- max_tokens=256 for short
#     tool-call-style responses (docs/AI_AGENT_SETUP.md, TOOL_CALL_HANDLING.md)
#     -- so 32 is not being presented as itself a *good* answer size, only
#     as the smallest reservation worth making at all. Below it, a "floor"
#     would take tokens away from thinking without buying a usable answer
#     in return -- the worst of both, so the code gives up on reserving one
#     and zeroes thinking out entirely instead (see the floor >= ceiling
#     branch below).
_ANSWER_FLOOR_FRACTION = 0.5
_MIN_ANSWER_FLOOR_TOKENS = 32
_MAX_ANSWER_FLOOR_TOKENS = 2048


def clamp_reasoning_budget_for_ceiling(
    manifest_flags: dict, caller_max_tokens: Any,
) -> int | None:
    """Returns the CLAMPED effective reasoning_budget when the
    manifest's locked budget would leave no room to answer, or None when
    nothing needs to change (not pathological, or no ceiling to measure
    against). Callers must leave the request payload completely untouched
    on None -- this is the negative-control contract.

    Reuses _effective_reasoning_ceiling, so this can never disagree with
    warn_if_reasoning_budget_exceeds_ceiling about what is pathological.
    """
    reasoning_budget = manifest_flags.get("reasoning_budget", 0)
    if (
        not isinstance(reasoning_budget, int)
        or isinstance(reasoning_budget, bool)
        or reasoning_budget <= 0
    ):
        return None
    ceiling, _source = _effective_reasoning_ceiling(manifest_flags, caller_max_tokens)
    if ceiling is None or reasoning_budget < ceiling:
        return None

    floor = min(
        max(_MIN_ANSWER_FLOOR_TOKENS, round(_ANSWER_FLOOR_FRACTION * ceiling)),
        _MAX_ANSWER_FLOOR_TOKENS,
    )
    if floor >= ceiling:
        # No sane floor fits: even the minimum answer floor would consume
        # the whole ceiling. Disable thinking for THIS request rather than
        # reserve a floor too small to be a real answer. 0 is the documented
        # "thinking off" value (MODEL_CONFIG_REFERENCE.md, "-1
        # unlimited, 0 off") -- this must NEVER be -1. -1 means UNLIMITED
        # thinking, the opposite of what this branch needs: it would make
        # the original problem infinitely worse, not fixed -- unbounded thinking on
        # exactly the requests this function exists to rescue. It would
        # also be silently misclassified: is_thinking_payload() only checks
        # reasoning_budget > 0 (a known trap of reasoning-budget
        # handling), so -1 reads as "not thinking" to that check even
        # though it means the opposite. This function never returns a
        # negative value.
        return 0
    return min(reasoning_budget, ceiling - floor)


def warn_if_reasoning_budget_exceeds_ceiling(
    model: str,
    manifest_flags: dict,
    caller_max_tokens: Any,
    *,
    thread_id: str | None,
    ip: str | None,
) -> None:
    """A locked reasoning_budget that meets or exceeds the
    effective output ceiling lets the model burn its whole allowance inside
    <think> and return no answer. The clamp (clamp_reasoning_budget_for_ceiling,
    above) prevents the silent-no-answer outcome for this same
    pathological case; this warning stays as the operator-facing signal that
    the manifest itself is still misconfigured for this caller shape.

    Shared by both chat-completion routes (pure computation + the warn-once
    latch); each call site supplies its own locally available caller_max_tokens
    / thread_id / ip, since how those are extracted from a request legitimately
    differs per endpoint shape (see the ollama_chat call site's options.*
    fallback).
    """
    reasoning_budget = manifest_flags.get("reasoning_budget", 0)
    if (
        not isinstance(reasoning_budget, int)
        or isinstance(reasoning_budget, bool)
        or reasoning_budget <= 0
    ):
        return

    ceiling, ceiling_source = _effective_reasoning_ceiling(manifest_flags, caller_max_tokens)
    if ceiling is None or reasoning_budget < ceiling:
        return

    caller = thread_id or ip or ""
    key = _reasoning_budget_warn_key(model, caller)
    if key in _reasoning_budget_warned:
        return
    _reasoning_budget_warned.pop(key, None)
    _reasoning_budget_warned[key] = None
    while len(_reasoning_budget_warned) > _REASONING_BUDGET_WARN_CAP:
        _reasoning_budget_warned.pop(next(iter(_reasoning_budget_warned)))

    log.warning(
        "reasoning_budget=%s >= effective ceiling=%s (from %s) for model=%s; "
        "the model may burn its whole reasoning_budget inside <think> and "
        "return no answer (see reasoning_budget in docs/MODEL_CONFIG_REFERENCE.md)",
        reasoning_budget, ceiling, ceiling_source, model,
    )


# OpenAI reasoning_effort is not applied per request: the think budget can
# only be honoured at engine STARTUP, so there is no per-request path.
# The design does not try to fight the engine in a way it was not
# designed for mid-conversation (a per-request value cannot change a
# budget that is fixed at engine launch).
# `reasoning_effort` sets the engine's `--reasoning-budget`
# LAUNCH flag once, when the resident is first spawned, and is LOCKED for
# that engine's entire life. It is never written into any request payload
# again.
#
# FIVE rungs -- `minimal` is deliberately absent, not special-cased:
#   low .25 · medium .50 · high .75 · xhigh 1.0 · max 1.25
# `low` is the FLOOR of the ladder. `minimal` is absent because it could
# not be delivered SAFELY on every model: on the models tried, at both a
# per-request lever and the launch lever, suppressing a model's
# thinking does not reliably make it answer faster or at all. Very small
# per-request thinking budgets and a launch-time budget of 0 can both make
# some models run to the output cap instead of stopping --
#   body thinking_budget_tokens=1/2/4 -> RUNAWAY to the output cap on some models
#   body thinking_budget_tokens=0     -> WORSE: removes the cap entirely
#                                        (very long reasoning on some models)
#   body thinking_budget_tokens=3     -> STOP  <- UNEXPLAINED. Listed because
#                                        it does not fit the pattern of the
#                                        other small values above.
#   body thinking_budget_tokens=8/32  -> stop cleanly
#   LAUNCH --reasoning-budget 0       -> RUNAWAY, WORSE still (even longer)
#   LAUNCH --reasoning off            -> reasoning=0, but some models still
#                                        ran to the output cap anyway; others
#                                        stopped cleanly.
# The engine's own help text (arg.cpp) says budget 0 means "immediate
# end". It measurably does not, on either lever -- its own log prints
# "forcing immediately" while the model reasons anyway. Do not trust that
# documented contract. Some models
# absolutely require thinking; asking for less than `low` is not a safe
# request this manager can honour uniformly, so it is not offered at all.
# Choosing a model that tolerates reduced thinking is an operator/end-user
# decision -- Turbohaul cannot know in advance which model a caller will
# pick, so it does not pretend to. This DEVIATES FROM THE OpenAI SPEC
# deliberately: `minimal` is a legal, documented OpenAI value, and a
# spec-compliant client sending it gets the 400 below instead of a
# resolution -- a deliberate design decision,
# not an oversight to be discovered later.
#
# ⚠ FIRST CALLER WINS, stated plainly (not just logged, not just a status
# field -- a caller needs telling outright, not just an operator who already
# suspects): whoever's request first spawns a model's engine fixes that
# model's `--reasoning-budget` for the engine's entire life. Every
# subsequent request to that same warm engine gets the FIRST caller's
# resolved budget, however different an effort THIS request asks for --
# there is no per-request override any more, by design (the point of this
# design is to not fight the engine's own startup-only contract).
# A client asking `low` on an engine the previous caller spawned at `max`
# silently gets `max` -- a real answer comes back, just not the one this
# request asked for, which is easy to overlook since no error is returned.
# Surfaced two ways for whoever is watching
# for it (manager.py: a WARNING log on mismatch, mirroring the reasoning_budget warning's
# style, and the per-resident status listing) -- but the documentation and this
# comment are where a CALLER who is not watching logs finds out it can
# happen at all.
# ⚠ KNOWN, UNEXPLAINED, DELIBERATELY NOT FIXED HERE.
# Generation on some models has been observed
# running to whatever output cap it was given instead of stopping on its own.
# What varies is TERMINATION, not effort: reasoning LENGTH is stable across
# samples that stop and samples that do not, and a run that does terminate
# finishes well inside the cap -- so the cap is not the binding constraint and
# "it hit the cap" is a symptom, not the cause.
#
# ⛔ It is NOT established that this ladder causes it, and NOT established that
# it is specific to the `max` rung. The same behaviour has been seen with no
# `reasoning_effort` sent at all, and at the manifest's own budget with the
# spawn-time override path NOT engaged -- i.e. on the path that ships today,
# with none of this code involved.
#
# Cause unknown. Not fixed here deliberately: a
# floor, clamp, guard or retry added while the cause is unknown and the scope
# unestablished would mask the behaviour without explaining it.
#
# ⇒ The observed frequency is deliberately
# NOT recorded here: a comment cannot be updated when new runs land, and
# nothing fails a test when a comment goes stale.
_EFFORT_FRACTION = {
    "low": 0.25, "medium": 0.5, "high": 0.75, "xhigh": 1.0, "max": 1.25,
}


def resolve_reasoning_effort(payload: dict) -> str | None:
    """Validates OpenAI ``reasoning_effort`` and returns the
    case-folded ladder key, or ``None`` if absent/JSON-null. Does NOT resolve
    a token count and does NOT touch ``payload`` -- the design puts the
    arithmetic at spawn time (see ``resolve_reasoning_budget_for_spawn``
    below); this function's job is the validation half, which is done
    FIRST: reject a non-string with 400, case-fold
    (``"HIGH"`` must not silently fail to match), reject an unrecognized
    value with 400 naming the five accepted ones.

    This function does not compare a resolved value
    against an explicit ``thinking_budget_tokens`` the client sent, and never 400s
    on disagreement. There is no "resolved value" at request time to
    compare against at all -- an explicit
    ``thinking_budget_tokens`` the client sends is simply forwarded as
    it always was (via ``_COMMON_FORWARDED_KNOBS``, untouched by
    this function), independent of whatever ``reasoning_effort`` says.
    """
    raw_effort = payload.get("reasoning_effort")
    if raw_effort is None:
        return None  # absent, or JSON null -- byte-identical, same contract as the clamp helper
    if not isinstance(raw_effort, str):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "reasoning_effort_unsupported_value",
                "message": "reasoning_effort must be a string",
                "received": type(raw_effort).__name__,
            },
        )
    effort_key = raw_effort.strip().lower()  # case-fold first -- "HIGH" must not silent-drop
    if effort_key not in _EFFORT_FRACTION:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "reasoning_effort_unsupported_value",
                "message": (
                    "reasoning_effort must be one of: low, medium, high, xhigh, max"
                ),
                "received": raw_effort,
            },
        )
    return effort_key


def resolve_reasoning_budget_for_spawn(effort_key: "str | None", manifest_budget: Any) -> int | None:
    """The arithmetic half of the ladder, kept out
    of request-time entirely. Called ONCE, from ``manager._reserve_and_start_
    locked``, against the FIRST slot to admit a new resident -- its result is
    baked into that resident's ``--reasoning-budget`` launch flag
    (``_spawn_for_resident``) and never recomputed for that engine's life.

    Pure and spawn-agnostic on purpose (``manager.py`` imports it via a
    local/deferred import, the same pattern already used five other places
    in this file -- ``_COMMON_FORWARDED_KNOBS``, ``_strip_thinking_all``
    (x3), ``wrap_reasoning_think``) so the arithmetic is unit-testable with
    no spawn, no Resident, no manager at all.

    Returns ``None`` (meaning: do not override the manifest's own
    ``reasoning_budget`` launch flag -- spawn with whatever the manifest
    already declares, 0/off or -1/unbounded included) when ``effort_key`` is
    ``None`` or unrecognized, or when ``manifest_budget`` is not a positive
    int -- absent (defaults handled by the caller) or non-positive (0 = off,
    -1 = unbounded) alike; there is no finite base to take a fraction of
    either way. No ceiling/floor arithmetic here, unlike a per-request
    design: there is no caller ``max_tokens`` in scope at spawn (a resident
    outlives any one request), and the separate per-request clamp
    (``clamp_reasoning_budget_for_ceiling``, untouched by this function) still
    protects the answer floor on every individual request regardless of
    what got baked in at spawn. Rounds UP (``math.ceil``), so the result is
    never LESS thinking than the ladder
    asked for because the product wasn't a whole number.
    """
    if effort_key not in _EFFORT_FRACTION:
        return None
    if not isinstance(manifest_budget, int) or isinstance(manifest_budget, bool):
        return None
    if manifest_budget <= 0:
        return None
    return math.ceil(_EFFORT_FRACTION[effort_key] * manifest_budget)


def _strip_thinking_wrapper(content: str) -> str:
    """Strip `<think>...</think>` wrapper to surface the post-think payload.

    Uses `rsplit('</think>', 1)` so even malformed multi-tag content (an
    aborted think block followed by a real one) surfaces the LAST post-think
    payload. Returns the original string when no closing
    tag is present.

    NOTE — rsplit-last is DELIBERATE here: the callers (`json_schema` validate +
    retry) `json.loads` the single trailing payload, so keeping only the tail
    after the FINAL `</think>` is exactly right. Do NOT switch this to remove-all
    — the shadow-save + byte-match probe, which must instead mirror the harness's
    remove-ALL resend, use :func:`_strip_thinking_all` (multi-block
    handling).
    """
    if not isinstance(content, str):
        return content
    if "</think>" in content:
        return content.rsplit("</think>", 1)[-1].lstrip()
    return content


# Mirror the AI harness think-strip so the
# shadow-save + byte-match probe predict the harness's think-STRIPPED resend byte-for-
# byte. The harness (`agent_runtime_helpers.strip_think_blocks`) REMOVES every
# `<think>...</think>` block via `re.sub(r'<think>.*?</think>', '', flags=re.DOTALL |
# re.IGNORECASE)` and its callers then `.strip()` the result. `_strip_thinking_wrapper`'s
# rsplit-last keeps only the tail after the LAST `</think>`, which DIFFERS from remove-all
# on multi-block / pre-`<think>` content and would silently drop the shadow byte-match
# (single-block content matches, multi-block content would mismatch). Non-greedy `.*?` pairs each
# open with its OWN close (never spans across a block); `re.IGNORECASE` matches the harness
# flag. Scoped to `<think>` only: the shadow callers pre-guard on a literal `</think>` and
# only qwen (the sole family the force-restore path enables) emits that tag — the harness's
# other passes (`<thinking>`/`<reasoning>`/tool-call XML/unterminated-tag) are no-ops on it,
# so this single pass is a faithful subset (any residual under-strip safe-degrades to a
# clean-restore + engine reprefill backstop, never a wrong answer).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking_all(content: str) -> str:
    """Remove ALL `<think>...</think>` blocks then `.strip()` — mirrors the AI harness
    harness resend (`strip_think_blocks(x).strip()`).

    Unlike :func:`_strip_thinking_wrapper` (rsplit-last, keeps only the tail after
    the FINAL `</think>`), this removes EVERY think block and preserves the visible
    text between/before blocks, so multi-block (`<think>a</think>X<think>b</think>Y`
    -> `XY`) and pre-`<think>` prose (`intro<think>t</think>ans` -> `introans`)
    byte-match what the harness resends next turn.

    Best-effort + None-safe (mirrors `_strip_thinking_wrapper`): a non-str passes
    through unchanged; single-block content strips identically to the harness; no
    think block -> the input with surrounding whitespace stripped (the harness always
    `.strip()`s — the shadow callers never reach this branch, they pre-guard on
    `</think>`); empty -> empty.
    """
    if not isinstance(content, str):
        return content
    return _THINK_BLOCK_RE.sub("", content).strip()


def _collapse_trailing_assistant_messages(messages: list) -> list:
    """Collapse a malformed TRAILING run of 2+ consecutive
    role=="assistant" messages into ONE before the list reaches the engine.

    llama-server's chat template 400s on "Cannot have 2 or more assistant
    messages at the end of the list." — once a session emits this shape the
    engine 400s on EVERY subsequent request (the slot is poisoned; a manager
    restart does not clear it). This is a pure, additive normalizer applied
    at the engine-forward chokepoint, not a change to any KV/thread-id/
    client_meta/prefix decision (those all run upstream of this call, on the
    ORIGINAL messages list).

    ADDITIVE ONLY: any list whose trailing assistant run is < 2 messages
    (including every well-formed list) is returned UNCHANGED — same object,
    no copy — so normal traffic is byte-identical. Never raises: non-list
    input, an empty list, or any non-dict element encountered while scanning
    the trailing run just stops the scan / returns the input unchanged.
    """
    if not isinstance(messages, list) or not messages:
        return messages

    # Find the maximal trailing run of dict messages with role=="assistant".
    run_start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict) or m.get("role") != "assistant":
            break
        run_start = i
    run = messages[run_start:]
    if len(run) < 2:
        return messages

    content_parts: list = []
    content_is_list = False
    tool_calls: list = []
    seen_tool_call_keys: set = set()
    reasoning_parts: list = []

    for m in run:
        c = m.get("content")
        if isinstance(c, list):
            content_is_list = True
            content_parts.extend(c)
        elif isinstance(c, str) and c:
            content_parts.append(c)
        # None / other falsy content contributes nothing (coerced away,
        # mirrors the existing null-content tool-call precedent).

        tc = m.get("tool_calls")
        if isinstance(tc, list):
            for entry in tc:
                key = (
                    json.dumps(entry, sort_keys=True, default=str)
                    if isinstance(entry, dict) else repr(entry)
                )
                if key in seen_tool_call_keys:
                    continue
                seen_tool_call_keys.add(key)
                tool_calls.append(entry)

        rc = m.get("reasoning_content")
        if isinstance(rc, str) and rc:
            reasoning_parts.append(rc)

    merged = dict(run[-1])  # preserve other keys from the LAST message in the run
    merged["role"] = "assistant"
    merged["content"] = content_parts if content_is_list else "\n".join(content_parts)
    if tool_calls:
        merged["tool_calls"] = tool_calls
    elif "tool_calls" in merged:
        del merged["tool_calls"]
    if reasoning_parts:
        merged["reasoning_content"] = "\n".join(reasoning_parts)

    return messages[:run_start] + [merged]


def wrap_reasoning_think(reasoning: str, content: str) -> str:
    """SINGLE SOURCE OF TRUTH for the ``<think>``-wrapped
    reasoning+content history form.

    Returns EXACTLY the bytes :func:`_merge_reasoning_into_content` writes to the
    client response — which the AI harness receives, strips at its storage
    boundary (``chat_completion_helpers.py`` ``_strip_think_blocks(content).strip()``)
    and resends think-free next turn. The SAVE-side shadow reconstructions
    (manager ``_generated_assistant_msg`` + the streaming accumulator) MUST emit these
    SAME bytes so that ``_strip_thinking_all(shadow_form)`` equals the harness's
    ``strip_think_blocks(merge_form).strip()`` BYTE-FOR-BYTE. The old no-newline
    ``<think>{r}</think>{c}`` shadow form diverged from this newline merge form under
    the remove-ALL strip on multi-block / stray-``</think>`` content (their strips
    differ on the whitespace the strip leaves behind), which silently broke the cold
    shadow restore's byte-match -> a reprefill instead of a strict-extension.

    Form (identical bytes to the merge site — do NOT change):
      reasoning + non-empty content -> ``<think>\\n{r.strip()}\\n</think>\\n\\n{content}``
      reasoning + empty content     -> ``<think>\\n{r.strip()}\\n</think>``
    The empty-vs-present branch is decided on ``content.strip()`` (mirrors the merge
    site's ``if ct.strip():``), so all-whitespace content collapses to the empty form
    exactly as the merge site does.

    PURE: no I/O, no mutation, deterministic on ``(reasoning, content)``. The
    ``reasoning`` truthiness + ``"<think>" not in content`` GUARD stays at each CALL
    site (the three sites guard differently); this helper only FORMATS.
    """
    rc_stripped = reasoning.strip()
    if content.strip():
        return f"<think>\n{rc_stripped}\n</think>\n\n{content}"
    return f"<think>\n{rc_stripped}\n</think>"


# Cap on per-tool_call argument string size to
# bound memory + log spam when a model emits runaway args. 256 KiB tolerates
# realistic large payloads (image data URIs, long structured inputs) while
# stopping unbounded growth.
MAX_TOOL_ARG_CHARS = 262144


def _coerce_created_at(created_value: Any) -> Any:
    """Coerce OpenAI int unix-epoch ``created``
    into ISO-8601 string for Ollama-compat ``created_at`` field. None and
    pre-formatted strings pass through unchanged.
    """
    if isinstance(created_value, int):
        return datetime.fromtimestamp(created_value, tz=timezone.utc).isoformat()
    return created_value


# === Ollama-style keep_alive parser ==============================

_KEEP_ALIVE_UNITS = {"s": 1, "m": 60, "h": 3600}  # read-only contract; do not mutate


def parse_keep_alive(value: Any) -> int | None:
    """Parse Ollama-style ``keep_alive`` field. Returns int seconds or None.

    Semantics (matches Ollama upstream):
      - ``None`` / unparseable → caller uses default ``idle_hot_load_seconds``
      - ``0`` (int/float/str ``"0"``/bool ``False``) → unload immediately
      - ``-1`` → pin (caller treats as :data:`KEEP_ALIVE_MAX_S`)
      - positive int seconds → clamped to ``[0, KEEP_ALIVE_MAX_S]`` by caller
      - Ollama-suffix strings ``"30s"``/``"5m"``/``"2h"`` → equivalent int seconds
      - bool ``True`` → ``None`` (Ollama "on" means "use server default")

    Single-layer clamp: this helper only normalises types; clamping to
    ``KEEP_ALIVE_MAX_S`` lives in :class:`TurbohaulManager` so there's one
    source of truth.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 0 if value is False else None
    if isinstance(value, (int, float)):
        v = int(value)
        if v < 0:
            return KEEP_ALIVE_MAX_S
        return v
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            v = int(s)
            if v < 0:
                return KEEP_ALIVE_MAX_S
            return v
        except ValueError:
            pass
        if (
            len(s) >= 2
            and s[-1].lower() in _KEEP_ALIVE_UNITS
            and s[:-1].lstrip("-").isdigit()
        ):
            return int(s[:-1]) * _KEEP_ALIVE_UNITS[s[-1].lower()]
    return None


def _derive_client_meta_identity(payload: dict, messages: list, ip: str | None = None) -> dict:
    """Parse OPTIONAL identity/classification fields
    out of an incoming request into a client_meta fragment.

    DORMANT FOUNDATION — decision-neutral. Nothing downstream reads these keys
    yet; they are stored on ``client_meta`` only. Absent fields → ``None``, so
    this is fully back-compat (zero behaviour change). Single source of truth so
    the three ``client_meta`` build sites stay byte-identical (no drift).

    The ``is_main`` / ``is_sub_agent`` / ``is_curator`` booleans are ALSO
    dormant/decision-neutral here (explicit payload field only, None when the
    harness omits them). The flag-gated curator route reads them off
    ``client_meta`` via ``kv_classify._class_from_label``; with the flag OFF or
    the fields absent they stay inert.

    KNOB-LEAK GUARD: NONE of these keys are in
    ``_COMMON_FORWARDED_KNOBS`` / ``_STREAM_FORWARDED_KNOBS``, and BOTH the
    ``_complete`` loop and ``_build_stream_payload`` forward to llama-server by
    ITERATING that allow-list — so these identity keys can never reach the
    sidecar payload. Do NOT add any of them to the knob allow-list.

    All values are best-effort and None-safe; never raises on a missing or
    malformed field.

    NESTING: The turbohaul provider sends identity fields
    NESTED under payload["client_meta"] (session_id, is_main, is_sub_agent,
    is_curator, is_compression). This function reads from the nested
    client_meta with a TOP-LEVEL fallback (back-compat). This activates the
    is_* labels, which would otherwise stay dormant (never reached manager).

    ``ip`` (the source-address surface) is captured here
    — the caller's ``request.client.host`` — from this ONE source of truth so
    all 3 client_meta build sites stay byte-identical (no drift). Same
    dormant/decision-neutral contract as session_id/is_*: stored on
    client_meta only, never referenced by the knob-forwarding allow-lists, so
    it can never reach the llama-server payload.
    """
    # Read from nested client_meta with top-level fallback
    cm = payload.get("client_meta") if isinstance(payload.get("client_meta"), dict) else {}
    first = messages[0] if (messages and isinstance(messages[0], dict)) else None
    return {
        # Metadata about turn 0 / the system prompt. Explicit payload field wins;
        # else minimally derive {role, content_len} from the first message.
        # content_len reuses compute_ctx_len (the shared char-count rule) on the
        # first turn so non-str / None content never raises. None if underivable.
        "turn0_meta": payload.get("turn0_meta") or (
            {"role": first.get("role"), "content_len": compute_ctx_len([first])}
            if first is not None else None
        ),
        # Agent role — explicit harness field ONLY; do NOT infer from content.
        "role": payload.get("role") if payload.get("role") is not None else cm.get("role"),
        # Session identifier — DISTINCT from the derived thread_id; explicit only.
        "session_id": payload.get("session_id") if payload.get("session_id") is not None else cm.get("session_id"),
        # Compression-turn flag — explicit only; None when the harness omits it.
        "is_compression": payload.get("is_compression") if payload.get("is_compression") is not None else cm.get("is_compression"),
        # Identity-class booleans — explicit payload field ONLY, None when
        # absent (mirrors is_compression). Dormant/back-compat: NOT in the knob
        # allow-lists so they never reach llama-server; the flag-gated
        # curator route consumes them via kv_classify._class_from_label.
        "is_main": payload.get("is_main") if payload.get("is_main") is not None else cm.get("is_main"),
        "is_sub_agent": payload.get("is_sub_agent") if payload.get("is_sub_agent") is not None else cm.get("is_sub_agent"),
        "is_curator": payload.get("is_curator") if payload.get("is_curator") is not None else cm.get("is_curator"),
        # The front-end per-role KV-save toggle. Manager-side
        # ONLY (like the is_* labels) — NEVER in the knob allow-lists / never to llama-server.
        # Read by _role_save_enabled off slot.client_meta; absent -> per-role default.
        "save_kv": payload.get("save_kv") if payload.get("save_kv") is not None else cm.get("save_kv"),
        # Context size — explicit field, else the cheap serialized-context char
        # count (same rule as admission_ctx_len). Int (0 for empty), never None
        # once messages exist.
        "context_size": payload.get("context_size")
        if payload.get("context_size") is not None
        else compute_ctx_len(messages),
        # Source IP, captured by the caller from
        # request.client.host. Display/observability only — never in the knob
        # allow-lists, never forwarded to llama-server.
        "ip": ip,
    }


def client_has_outstanding_work(
    ip,
    *,
    visible_client_metas,
    unseen_outstanding: int = 0,
    observation_failed: bool = False,
) -> bool:
    """The CONCURRENCY DISCRIMINATOR. Does this client already have
    work outstanding on the server RIGHT NOW?

    ⚠ EVERY ARGUMENT AFTER ``ip`` IS KEYWORD-ONLY, AND ``visible_client_metas``
    HAS NO DEFAULT. That is deliberate.
    ``observe_for_grace_match`` returns a ``(metas, unseen)`` tuple, so a caller
    naturally writes ``client_has_outstanding_work(*obs)`` — which looks obviously
    right and silently binds the METAS LIST to ``ip`` and the UNSEEN INT to the
    metas. ``ip`` then fails the ``isinstance(ip, str)`` guard below, the function
    returns True on EVERY call, the caller's match is refused forever, and the fix
    it guards becomes an inert no-op with no exception and a green suite.
    ⇒ Keyword-only makes that miswiring a LOUD ``TypeError`` at the call site
    instead of a silent permanent "concurrent". And the metas are REQUIRED rather
    than defaulted, because a default of ``()`` would read as "no visible work" —
    the PERMISSIVE direction, which is the one this whole function exists to avoid.
    ⛔ Callers must therefore be prepared for this to raise on a WIRING error:
    keep the call inside the caller's own best-effort guard so a mistake degrades
    to "no IP match" rather than failing an admission.

    This is the question the ``max_parallel_sidecars <= 1`` gate below cannot
    ask, and it is the one that actually separates the two cases that gate is
    stuck between:

      * a conversation's OWN NEXT TURN arrives when the previous turn has
        FINISHED -> nothing of this client's is outstanding -> a stable identity
        is what makes the grace window match instead of timing out (the
        ``wait = grace_seconds - client gap`` stall).
      * A concurrent sub-agent FAN-OUT arrives while its siblings are
        still running -> sharing one identity is a cross-context KV
        mismatch (positive stale -> CLEAR + reprefill) the gate exists to stop.

    ⇒ The discriminator is CONCURRENCY, not the sidecar cap. Cap>=2 makes the
    fan-out POSSIBLE; it does not make any particular request part of one.

    AFFIRMATIVE RULE — ``False`` requires POSITIVE PROOF OF SOLITUDE.
    Every uncertainty resolves to ``True``:

      * ``observation_failed``     -> True. We could not look; we do not guess.
      * unknown/blank ``ip``       -> True. An unidentifiable client cannot be
                                      shown to be alone.
      * any ``unseen_outstanding`` -> True. Resident inboxes report ``qsize()``
                                      only (``manager.Resident.inbox``), so a
                                      queued sibling is COUNTABLE but not
                                      INSPECTABLE. Countable-but-not-inspectable
                                      is not the same as absent.
      * a visible meta with no ip  -> True. It cannot be ruled out as this
                                      client's.

    ⚠ WHY THAT DIRECTION: ``True`` means the caller keeps today's
    content-derived identity — i.e. TODAY'S BEHAVIOUR at cap>=2. So a wrong
    ``True`` costs the grace wait that a stable identity avoids (a slow turn,
    already the status quo) while a wrong ``False`` merges two live conversations
    onto one KV cache. The two errors are not symmetric and this function
    fails toward the cheap one.

    ⛔ WHAT THIS IS NOT: it is not a lock and it does not reserve anything. Two
    genuinely simultaneous siblings can each observe no outstanding work before either
    becomes visible, and both would read as solitary. The observation is a
    SNAPSHOT; that is a known limitation. Closing that window means deciding
    identity under the manager's registry lock rather than in the route, which is
    a wiring question and deliberately not settled here.

    ``visible_client_metas``: client_meta dicts of work we CAN inspect.
    ``unseen_outstanding``:   count of outstanding work we CANNOT inspect.
    """
    if observation_failed:
        return True
    if not isinstance(ip, str) or not ip.strip():
        return True
    if unseen_outstanding:
        return True
    for meta in visible_client_metas or ():
        other = meta.get("ip") if isinstance(meta, dict) else None
        if not isinstance(other, str) or not other.strip():
            return True  # unattributable work — cannot rule it out as ours
        if other == ip:
            return True
    return False


def observe_outstanding_work(manager):
    """Best-effort snapshot for :func:`client_has_outstanding_work`.

    Returns ``(visible_client_metas, unseen_outstanding, observation_failed)``.

    ``unseen_outstanding`` reuses the count the manager already maintains for
    exactly this blind spot — ``queue.depth(inbox_waiting=...)``'s
    ``queue_depth_total`` (staging + accepted + resident-inbox waiters, see
    ``TurbohaulManager._inbox_waiting_count``) — rather than re-deriving a
    second, divergent notion of "how much is waiting".

    Any failure degrades to ``observation_failed=True``, which the predicate
    reads as "concurrent" and which costs a slow turn, never a merged identity.
    """
    try:
        metas = []
        for slot in list(getattr(manager, "_inflight", None) or ()):
            cm = getattr(slot, "client_meta", None)
            if isinstance(cm, dict):
                metas.append(cm)
        depth = manager.queue.depth(
            inbox_waiting=manager._inbox_waiting_count(),
            claims_waiting=len(manager.fastlane_claims_snapshot()),
        )
        return metas, int(depth.get("queue_depth_total") or 0), False
    except Exception:
        log.debug("outstanding-work observation failed", exc_info=True)
        return [], 0, True


def _shadow_recompose_identity(base_thread_id: str, role, session_id) -> str:
    """DORMANT identity-shadow stage (precursor to role-keyed identity activation). Computes a
    role-keyed thread_id CANDIDATE that is LOGGED ONLY, never used. Today the
    manager IP-fallback collapses every harness role (main/sub/compression) onto
    ONE thread_id (identity collapse); activation will fix that with a role-keyed
    thread_id. This dormant stage first proves — from the emitted corpus — that this
    recomposition only ever SPLITS an identity (never MERGES two) before we
    switch it on.

    Safe-by-construction:
      * APPEND-ONLY — new_key STARTS WITH the full ``base_thread_id`` (today's
        derived thread_id) and only APPENDS a suffix; it NEVER replaces the base,
        so two distinct bases can never recompose to the same key (no MERGE).
      * HASHED suffix — each field is a short SHA-256 hex prefix, so raw
        session_id / role never leak and a value containing the '-'/'=' delimiter
        cannot forge another field (delimiter-injection-proof).
      * No fields present -> new_key == base (identity). Best-effort / None-safe;
        a falsy role or session_id contributes nothing.

    Unified: delegates to kv_classify.recompose_identity (single source of truth
    — that pure function is byte-identical to the logic that lived here).
    """
    from turbohaul.kv_classify import recompose_identity
    return recompose_identity(base_thread_id, role, session_id)


def _m2b_active() -> bool:
    """Is role-keyed identity ACTIVATION on? When True the shadow-
    computed role+session-keyed new_key DRIVES the slot identity (thread_id);
    when False the shadow stays dormant (log-only). env TURBOHAUL_M2B_ACTIVE (default
    OFF) — a runtime flag so it can be flipped OFF instantly if a MISMATCH/merge
    ever shows. resolve_kv / the restore gate CODE is byte-identical either way;
    ONLY the identity fed IN changes."""
    return os.environ.get("TURBOHAUL_M2B_ACTIVE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# === SSE tuning constants (module-level for monkeypatch-in-tests) ===

# How long to wait for the slot to actually reach ACTIVE before we give up.
# Cold-load of a 27B GGUF can take 30-60s; pre-stream wait should be much
# longer than that since the route is held open.
SLOT_READY_TIMEOUT_S = 7200.0
# httpx.stream timeout for the actual sidecar connection — keep generous for
# slow-thinking models on large contexts.
STREAM_TIMEOUT_S = 3600.0
# Emit `: keep-alive\n\n` SSE comments at this cadence while waiting
# for `slot.stream_ready_event` to fire. Many clients set 30-60s read-timeouts
# on streaming responses; without intermittent bytes the client disconnects
# during cold-load (a large GGUF can take tens of seconds to load). SSE comments are RFC
# 8895 / EventSource-compliant; clients silently consume them and the
# connection stays warm.
HEARTBEAT_INTERVAL_S = 12.0


# === Typed upstream errors ===

class SidecarUnavailableError(RuntimeError):
    """Sidecar process disconnected, crashed, or is otherwise unreachable.

    Examples: httpx.RemoteProtocolError (server disconnected mid-response,
    typically OOM-crash from KV-cache pressure), ConnectError (port closed),
    ReadError (read failed). Maps to HTTP 503 + Retry-After at the route.
    """

    def __init__(self, message: str, cause: str = "sidecar_disconnected", retry_after_s: int = 30):
        super().__init__(message)
        self.cause = cause
        # This class's own default/argument is a FLOOR, not the advertised
        # value -- see _sidecar_unavailable_retry_after_s below, which is
        # what actually computes the HTTP header.
        self.retry_after_s = retry_after_s


def _sidecar_unavailable_retry_after_s(mgr, exc: "SidecarUnavailableError") -> int:
    """NEVER ADVERTISE A Retry-After
    SHORTER THAN WE WOULD OURSELVES WAIT for a model load. An optimistic
    value marches a well-behaved client into a still-dead service and can
    turn one crash into a retry storm; a pessimistic one only costs a small
    delay -- that asymmetry is the whole argument for erring long.

    Derived from queue.loading_health_timeout_s -- the SAME deadline the
    manager already gives itself before declaring a model load failed
    (the model-load timeout) -- rather than a second number nobody can
    defend. This is a derived value, not a guess: the low end of the
    recovery window is close to typical model cold-load times
    (a few tens of seconds), so the load
    timeout is a real, load-bearing deadline for this exact scenario, not an
    unrelated number being repurposed.

    Read live off mgr.runtime (never captured at construction time) so the
    advertised value moves automatically if the timeout is ever retuned --
    api/config_put.py's PUT /api/config REPLACES mgr.runtime wholesale on
    every write (`mgr.runtime = new_runtime`), so a value captured once at
    startup would silently go stale the first time an operator changes it.

    max() with the exception's own retry_after_s: a floor, not a ceiling --
    this can only ever raise the advertised value above the exception's
    default, never lower it, keeping the "never shorter" invariant even if
    the config read fails or returns something surprising.
    """
    try:
        load_timeout_s = int(mgr.runtime.queue.loading_health_timeout_s)
    except Exception:
        return exc.retry_after_s
    return max(exc.retry_after_s, load_timeout_s)


class SidecarUpstreamError(RuntimeError):
    """Sidecar accepted the request and returned a structured error response.

    Example: httpx.HTTPStatusError on 4xx (context overflow, malformed
    payload, etc.). Maps to HTTP 502 Bad Gateway at the route. The
    upstream status + (truncated) body are preserved for client diagnosis.
    """

    def __init__(self, message: str, upstream_status: int, upstream_body: str = ""):
        super().__init__(message)
        self.upstream_status = upstream_status
        self.upstream_body = upstream_body[:500]


class SidecarTimeoutError(RuntimeError):
    """Sidecar request timed out (httpx.TimeoutException).

    Maps to HTTP 504 Gateway Timeout. Client may retry but should consider
    reducing request size first.
    """

    def __init__(self, message: str, retry_after_s: int = 60):
        super().__init__(message)
        self.retry_after_s = retry_after_s


# ============================================================================
# OpenAI-compat /v1/chat/completions
# ============================================================================


@router.post("/v1/chat/completions")
async def openai_chat_completions(payload: dict, request: Request):
    """OpenAI-shape chat completion. Forwarded through manager.submit_and_wait
    for non-streaming requests; through manager.submit_for_streaming + an SSE
    pass-through generator for streaming requests (SSE
    chunks relayed as they arrive).

    Return type is ``dict`` for non-streaming or ``fastapi.responses.StreamingResponse``
    for streaming.
    """
    # Shallow copy to avoid mutating caller's payload dict
    payload = dict(payload)
    mgr = request.app.state.manager
    model = payload.get("model")
    messages = payload.get("messages")
    if not model:
        raise HTTPException(status_code=400, detail="`model` field required")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")
    # The model tag must exist in the manifest store — mirrors the
    # read_manifest -> 404 pattern at embeddings.py:119 / ollama.py:68.
    try:
        _manifest = read_manifest(mgr.boot.storage.manifests_path, model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {model}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (yaml.YAMLError, ValidationError, UnicodeDecodeError, OSError) as e:
        # pydantic's ValidationError is a SIBLING of ManifestValidationError
        # under ValueError, not a subclass -- the arm above never caught it, and
        # with no app-level exception handler it would surface as a bare 500. Same
        # handled-5xx shape as models.py:get_model: the caller named
        # a real, existing model, so not a 4xx; the manifest is permanently
        # broken until an operator fixes it, so not the transient 503/502/504
        # this file reserves for sidecar failures (see module docstring).
        # Fixed message only -- never str(e): yaml.YAMLError leaks parser
        # position/content, pydantic ValidationError echoes manifest field
        # VALUES. Full detail logged server-side instead.
        log.exception("manifest for %r is present but unreadable", model)
        raise HTTPException(
            status_code=500, detail=f"manifest for '{model}' is present but unreadable"
        ) from e
    # Warn (at most once per model+caller) when a locked
    # reasoning_budget meets or exceeds the effective output ceiling. Placed
    # here -- unconditional, before the wants_stream fork below -- so it
    # covers streaming requests too, not only the json_schema-gated
    # thinking_manifest slice further down (that one serves a different,
    # narrower purpose: the in-_complete retry-path gate).
    warn_if_reasoning_budget_exceeds_ceiling(
        model,
        _manifest.llama_server_flags or {},
        payload.get("max_tokens"),
        thread_id=payload.get("thread_id") or None,
        ip=request.client.host if (getattr(request, "client", None) and request.client.host) else None,
    )
    # Request-time clamp for the same pathological case the
    # warning above detects. None ⇒ untouched (negative control) -- only set
    # payload["reasoning_budget"] when a clamp is actually needed, so a
    # normal request's payload shape is unchanged.
    _clamped_reasoning_budget = clamp_reasoning_budget_for_ceiling(
        _manifest.llama_server_flags or {}, payload.get("max_tokens"),
    )
    if _clamped_reasoning_budget is not None:
        payload["reasoning_budget"] = _clamped_reasoning_budget
    # Validate reasoning_effort and carry
    # the case-folded key through to client_meta -- NEVER into payload. The
    # arithmetic (ladder fraction -> token count) happens once, at spawn
    # time, in manager._reserve_and_start_locked, against whichever request
    # first admits a new resident (see resolve_reasoning_budget_for_spawn's
    # own docstring). Raises HTTPException(400) directly for a
    # non-string/unrecognized value.
    _effort_key = resolve_reasoning_effort(payload)
    # Best-effort prompt extraction for thread-id derivation
    # OpenAI tool_call messages have content=null (None),
    # not "". The .get("content", "") pattern returns None (not the default "") when
    # content is explicitly null, which crashes " ".join() with TypeError. Coerce None
    # to "" to tolerate tool-call assistant turns in multi-turn conversations.
    prompt = " ".join(
        (c if isinstance(c := m.get("content"), str) else "")
        for m in messages if isinstance(m, dict)
    )
    thread_id = payload.get("thread_id") or ""
    # Stable per-agent identity = the client
    # container source IP, so an agent's sequential turns grace-MATCH the same warm
    # slot and reuse its KV prefix cache (cache_reuse) instead of re-prefilling the
    # full context each turn. GATED on single-residency (cap<=1): at cap>=2 the
    # full-prompt-hash identity (manager.submit) is kept so concurrent
    # sub-agent fan-out is NOT regressed. Falls back to hash if no IP.
    try:
        _single_residency = request.app.state.manager.runtime.queue.max_parallel_sidecars <= 1
    except Exception:
        _single_residency = True
    if not thread_id and _single_residency:
        # MULTI-SIGNAL identity: IP (role tier) + fingerprint of the
        # STATIC first message (instance identity). IP-only collapses concurrent sub-agents
        # into ONE thread_id -> cross-context KV mismatch -> positive stale -> CLEAR+reprefill.
        # First-message fingerprint distinguishes sub-agents by persona AND keeps a
        # conversation's follow-ups (same first msg + larger ctx) on ONE id -> KV reuse fires.
        _ip = request.client.host if (getattr(request, "client", None) and request.client.host) else ""
        _first_msg = next(
            (m.get("content", "") for m in (messages or [])
             if isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("content").strip()),
            "",
        )
        if _first_msg:
            _fp = derive_thread_id_prefix_hash(_first_msg, payload.get("model", ""))
            thread_id = ("agent-ip-" + _ip + "-" if _ip else "") + _fp
        elif _ip:
            thread_id = "agent-ip-" + _ip
    # === DORMANT identity-shadow (decision-neutral) ==============
    # thread_id (old_key) is now finalized. Compute a role-keyed CANDIDATE and
    # LOG old-vs-new ONLY — the shadow is NEVER assigned to thread_id nor passed
    # anywhere (activation turns it on once it is shown to only SPLIT, never
    # MERGE). Best-effort: any failure is swallowed so it can't break a request.
    #
    # Observability: _req_id is generated OUTSIDE the try so it
    # survives even if the identity_recompose block itself raises -- it is
    # stashed into client_meta below and re-emitted by the manager's
    # R2B_REQ_IDENTITY line, so the two layers' identity views can be joined
    # by grep under load without depending on thread_id staying stable in
    # between (which is exactly what can legitimately change here).
    _req_id = uuid.uuid4().hex[:12]
    _role_source = "absent"
    _session_source = "absent"
    try:
        old_key = thread_id
        role = payload.get("role")
        session_id = payload.get("session_id")
        if role:
            _role_source = "payload"
        if session_id is not None:
            _session_source = "payload"
        # Recover session_id from idle_hot resident when
        # payload omits it. Tool-call re-cues arrive with session_id=None after
        # grace timer fires, but the warm idle resident has the original
        # session_id. Match the incoming IP+fingerprint base thread_id against
        # the idle resident's thread_id (which has the s={sha256} suffix).
        # Without this recovery, _shadow_recompose_identity produces a session-
        # LESS thread_id, durable_ring_key returns None, KV-bin owner-match
        # fails vs the session-present bin -> forced-full -> a full re-prefill.
        if session_id is None:
            try:
                mgr = request.app.state.manager
                if (getattr(mgr, "_idle_handle", None) is not None
                        and getattr(mgr, "_idle_client_meta", None)
                        and getattr(mgr, "_idle_thread_id", None)):
                    _idle_cm = mgr._idle_client_meta
                    _recovered = _idle_cm.get("session_id") if isinstance(_idle_cm, dict) else None
                    _idle_tid = mgr._idle_thread_id
                    # Match: incoming base thread_id is a prefix of the idle
                    # thread_id (idle has the s-suffix from its original
                    # derivation, incoming doesn't yet). Only recover when the
                    # base matches — don't hijack a different agent's session.
                    if _recovered and _idle_tid.startswith(old_key):
                        session_id = _recovered
                        _session_source = "idle-hot-recovery"
                        log.info(
                            "session_id recovered from idle-hot resident: "
                            "thread_base=%s idle_thread=%s",
                            old_key[:24], _idle_tid[:24],
                        )
            except Exception:
                log.debug("session_id recovery failed (ignored)", exc_info=True)
        new_key = _shadow_recompose_identity(old_key, role, session_id)
        # Role-keyed activation: when TURBOHAUL_M2B_ACTIVE, the role+session-keyed new_key
        # DRIVES the slot identity (warm-path role isolation). resolve_kv / the restore
        # gate CODE is byte-identical — ONLY the thread_id fed IN changes. Append-only
        # (new_key STARTS WITH old_key), so it can only split: a SPLIT mints
        # a NEW identity (-> no bin -> fresh reprefill + saves own, never a wrong-
        # restore); a stable-key continuation keeps warm reuse. Flag-gated so it can be
        # flipped OFF instantly on any MISMATCH/merge.
        _m2b = _m2b_active()
        if _m2b and new_key != old_key:
            thread_id = new_key
        log.info(
            "identity_recompose old_h=%s new_h=%s role=%s role_source=%s "
            "session_present=%s session_source=%s ip_present=%s m2b_active=%s "
            "req_id=%s",
            hashlib.sha256(old_key.encode()).hexdigest()[:12],
            hashlib.sha256(new_key.encode()).hexdigest()[:12],
            role if role else "-",
            _role_source,
            bool(session_id),
            _session_source,
            bool(locals().get("_ip")),
            _m2b,
            _req_id,
        )
    except Exception:
        log.debug("identity_recompose failed (ignored)", exc_info=True)
    # response_format pre-validation. Fires for BOTH stream
    # + non-stream so SSE clients cannot bypass via the wants_stream fork.
    # Strict shape: {type:"json_object"} accepted, {type:"text"} normalized to
    # None (OpenAI default — no-op pass-through), anything else REJECTED 400.
    rf = payload.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "response_format_unsupported_type",
                    "message": "response_format must be an object",
                    "received": type(rf).__name__,
                },
            )
        rf_type = rf.get("type")
        if rf_type == "text":
            payload["response_format"] = None  # OpenAI default — no-op
        elif rf_type == "json_object":
            pass  # accept-and-forward — no validation; manifest decides if model honors
        elif rf_type == "json_schema":
            # Validate caller schema; on bad → 422 schema_validation_failed.
            # On ok, response_format propagates via the existing _COMMON_FORWARDED_KNOBS
            # tuple (no separate forwarding code path needed).
            ok, reason = _validate_json_schema(rf)
            if not ok:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "schema_validation_failed",
                        "message": f"json_schema validation failed: {reason}",
                    },
                )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "response_format_unsupported_type",
                    "message": (
                        "response_format type must be one of "
                        "'text', 'json_object', 'json_schema'"
                    ),
                    "received_type": str(rf_type),
                },
            )
    wants_stream = bool(payload.get("stream", False))

    # SSE streaming pass-through:
    # when the client sends stream=true, branch to the streaming helper which
    # opens its own httpx.stream() to the sidecar and yields SSE chunks back
    # to the client. The non-streaming path below is unchanged.
    if wants_stream:
        return await _openai_chat_completions_stream(
            request, mgr, model, messages, prompt, thread_id, payload, _req_id, _effort_key,
        )

    # When caller requested json_schema, read the manifest
    # to slice out reasoning_budget for the in-_complete retry-path gate. Only
    # fires for json_schema requests (narrow scope — text/json_object/
    # no-response_format never need this read). Read failure is non-fatal:
    # thinking_manifest stays empty + retry path is disabled, primary forward
    # still proceeds via _COMMON_FORWARDED_KNOBS.
    thinking_manifest: dict = {}
    if (
        isinstance(payload.get("response_format"), dict)
        and payload["response_format"].get("type") == "json_schema"
    ):
        try:
            _m = read_manifest(mgr.boot.storage.manifests_path, model)
            thinking_manifest = {
                "reasoning_budget": (_m.llama_server_flags or {}).get(
                    "reasoning_budget", 0,
                ),
            }
        except Exception:
            log.exception(
                "manifest read failed for model=%s; retry-path disabled",
                model,
            )

    # Build the knob dict with the
    # SAME comprehension the stream path (_build_stream_payload) uses, so
    # the non-stream client_meta is byte-identical to the stream path.
    # A hand-enumerated list would drop n_predict (and sampler knobs like
    # presence_penalty, frequency_penalty, repeat_penalty, etc.) — fields
    # that ARE in _COMMON_FORWARDED_KNOBS but would never be copied here, so
    # _complete's _COMMON_FORWARDED_KNOBS loop would find nothing and the
    # 300-token cap (n_predict) would be silently dropped on non-stream requests.
    forwarded_knobs = {
        k: payload.get(k) for k in _COMMON_FORWARDED_KNOBS
        if payload.get(k) is not None
    }
    client_meta = {
        "kind": "openai-chat-completion",
        "messages": messages,  # carried for the completion_fn to forward; redacted from /ws/state
        "model": model,
        "stream": False,
        # Ollama-style keep_alive → IDLE_HOT extension hint
        "keep_alive_s": parse_keep_alive(payload.get("keep_alive")),
        # Forward validated response_format to
        # llama-server via _complete's _COMMON_FORWARDED_KNOBS loop. Skipping
        # this line would silently drop the field on the non-stream path
        # despite the validator accepting it.
        "response_format": payload.get("response_format"),
        # Manifest reasoning_budget slice for in-_complete
        # is_thinking_payload check. {} ⇒ retry disabled (non-thinking or read-fail).
        "thinking_manifest": thinking_manifest,
        # All _COMMON_FORWARDED_KNOBS (temperature, top_p, top_k, max_tokens,
        # min_p, thinking_budget_tokens, reasoning_budget, reasoning,
        # presence_penalty, frequency_penalty, repeat_penalty, repeat_last_n,
        # typical_p, seed, mirostat, mirostat_lr, mirostat_ent, n_predict,
        # tools, tool_choice, parallel_tool_calls, function_call, functions)
        # forwarded via the comprehension above — byte-identical to stream path.
        **forwarded_knobs,
        # The validated, case-folded reasoning_effort
        # key, for the manager to resolve at SPAWN time only (see
        # resolve_reasoning_budget_for_spawn). Deliberately NOT in
        # _COMMON_FORWARDED_KNOBS -- same "decision-neutral, not in the knob
        # allow-list" shape as the identity fields just below -- so it never
        # reaches the llama-server payload for a warm resident (first-caller
        # -wins; every request after the first is a no-op read of this key).
        "reasoning_effort": _effort_key,
        # DORMANT identity/classification foundation.
        # Decision-neutral: stored only, read nowhere; NOT in the knob allow-list
        # so it never reaches the llama-server payload. See _derive_client_meta_identity.
        **_derive_client_meta_identity(
            payload, messages,
            ip=request.client.host if (getattr(request, "client", None) and request.client.host) else None,
        ),
        # Observability: carried through to slot.client_meta so
        # _emit_request_identity can re-emit it on R2B_REQ_IDENTITY, joining
        # this layer's identity_recompose line to the manager's.
        "req_id": _req_id,
    }

    # Client-disconnect watcher. Event constructed
    # IN-HANDLER so it's bound to the route's request-loop (correct-loop
    # guarantee). watch_disconnect polls request.is_disconnected() every 2s; if
    # client closes, sets the Event which queue.pop_next sees and evicts the slot.
    disconnect_event = asyncio.Event()
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))
    try:
        slot, result = await mgr.submit_and_wait(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            context=list(messages),
            client_meta=client_meta,
            disconnect_event=disconnect_event,
            admission_ctx_len=compute_ctx_len(messages),
            # CLASSIFIER: incoming turn-hash chain (parallel to admission_ctx_len)
            # so the manager restore path classifies by prefix-VALIDITY vs the pinned
            # clean bin. Same _prefix_hash_chain (unified) used at save time, so
            # both sides of the comparison are byte-identical hashes.
            admission_hash_chain=_prefix_hash_chain(messages),
        )
    except SlotEvictedError as e:
        # Client closed connection before slot activated; surface as
        # HTTP 499 (client_closed_request) so monitoring distinguishes client-side
        # close from a real 500 server fault.
        raise HTTPException(
            status_code=499,
            detail={"error": "client_closed_request", "message": str(e)},
        ) from e
    except SidecarUnavailableError as e:
        # 503 — sidecar crashed/disconnected (likely KV-cache OOM mid-response)
        raise HTTPException(
            status_code=503,
            detail={"error": sidecar_unavailable_error_body(e)},
            headers={"Retry-After": str(_sidecar_unavailable_retry_after_s(mgr, e))},
        ) from e
    except SidecarTimeoutError as e:
        # 504 — request exceeded sidecar timeout
        raise HTTPException(
            status_code=504,
            detail={"error": "sidecar_timeout", "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarUpstreamError as e:
        # 502 — sidecar returned an upstream 4xx/5xx (context overflow,
        # malformed payload, etc.)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_sidecar_error",
                "upstream_status": e.upstream_status,
                "upstream_body": e.upstream_body,
                "message": str(e),
            },
        ) from e
    except VramOverCommitError as e:
        # No VRAM capacity (target card full, nothing idle-evictable in
        # time). Retryable — real 503 + Retry-After, NOT a silent 200/500.
        raise HTTPException(
            status_code=503,
            detail={"error": capacity_unavailable_error_body(e)},  # single source of truth
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except RuntimeError as e:
        # loading-fail / safety-gate-refused / unknown worker exception → 500
        raise HTTPException(status_code=500, detail=f"sidecar failed: {e}") from e
    finally:
        # Tear down the disconnect watcher cleanly.
        # contextlib.suppress so a normal cancellation doesn't surface from finally.
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watch_task

    if result is None:
        # Default completion_fn (no real backend wired) — return an empty echo
        raise HTTPException(
            status_code=503,
            detail="no completion_fn wired - production needs make_llama_server_complete_fn",
        )
    return result


# ============================================================================
# Streaming helper
# ============================================================================


# Knobs forwarded to llama-server from client_meta.
# The knobs are split into _COMMON (forwarded everywhere)
# and _STREAM_ONLY (only included in the streaming payload). `_complete`
# iterates _COMMON only so non-streaming requests never inherit stream-only
# keys; the streaming payload-build helper uses the derived
# _STREAM_FORWARDED_KNOBS alias. Drift between the two lists is now
# structurally impossible.
_COMMON_FORWARDED_KNOBS = (
    # Core OpenAI-compat
    "temperature", "top_p", "top_k", "max_tokens", "min_p",
    # accept-and-forward only; handler-entry validator
    # rejects json_schema as deferred to schema validation. Thinking-mode JSON guarantee
    # blocked upstream on llama.cpp #20345 + Ollama #10538.
    "response_format",
    # Preserved-thinking controls
    "thinking_budget_tokens", "reasoning_budget", "reasoning",
    # Ollama-parity samplers
    "presence_penalty", "frequency_penalty", "repeat_penalty",
    "repeat_last_n", "typical_p", "seed",
    "mirostat", "mirostat_lr", "mirostat_ent",
    # max-output alias
    "n_predict",
    # Tool-call pass-through (minimal — full capability
    # advertisement + size-cap + per-model gating deferred to a later change
    # for now; this is "forward the field to a model
    # that supports tool_calls natively, e.g. a 27b dense model"). llama-server
    # mirrors OpenAI's schema, so structured values (list/dict/string) just
    # pass through unchanged.
    "tools", "tool_choice", "parallel_tool_calls",
    "function_call", "functions",
)
_STREAM_ONLY_KNOBS = ("stream", "stream_options")
# Back-compat alias — `_build_stream_payload` reads this for the streaming
# llama-server call where `stream=True` must be forwarded.
_STREAM_FORWARDED_KNOBS = _COMMON_FORWARDED_KNOBS + _STREAM_ONLY_KNOBS


def _build_stream_payload(client_meta: dict, model: str, messages: list) -> dict:
    """Build the streaming chat-completions payload sent to llama-server."""
    payload: dict = {
        "model": model,
        "messages": _collapse_trailing_assistant_messages(messages),
        "stream": True,
    }
    for k in _STREAM_FORWARDED_KNOBS:
        v = client_meta.get(k)
        if v is not None:
            payload[k] = v
    return payload


def sidecar_unavailable_error_body(exc: "SidecarUnavailableError") -> dict[str, Any]:
    """The ONE construction for the typed
    ``sidecar_unavailable`` error shape (type + cause + message) -- mirrors
    ``capacity_unavailable_error_body``'s own argument: two or three
    byte-identical dict literals are not "cannot
    drift", they are a single source of truth deferred.

    Building this body inline at a sidecar_unavailable site gives
    ``{"error": "sidecar_unavailable", ...}`` with ``error`` as a bare STRING,
    not the typed dict shape every OTHER error family in this API uses (see
    ``capacity_unavailable_error_body``) -- a client parsing ``error.type`` per
    the documented contract breaks on that shape. This helper keeps ``error`` a
    typed object (type + cause + message), so every site returns the same typed
    shape.

    Only builds the ``detail["error"]`` dict body -- callers keep constructing
    their own ``Retry-After`` header via ``_sidecar_unavailable_retry_after_s``
    unchanged (the Retry-After semantics are not changed here).
    """
    return {
        "type": "sidecar_unavailable",
        "cause": exc.cause,
        "message": str(exc),
    }


def capacity_unavailable_error_body(exc: VramOverCommitError) -> dict[str, Any]:
    """The ONE construction for the typed
    ``capacity_unavailable`` error shape (type + cause + message + retry_after).

    Called by all FIVE sites that surface a VramOverCommitError delivered
    through ``manager._fail_completion_future`` -- this module's non-stream 503s
    (openai and ollama routes), its two SSE frames (the timeout branch and
    the handle-is-None branch), and embeddings.py's non-stream 503 (via
    ``_raise_capacity_or_routing_failure``, which both embeddings branches call).
    Enforced by the capacity-error-shape parity test,
    which builds BOTH halves of its parity assertion from this function (a
    comparison against a hardcoded literal would be tautological).

    SCOPE: this is
    not "every code path a VramOverCommitError could ever take". Two exception
    ladders would swallow one into a NON-conforming body, because
    VramOverCommitError subclasses RuntimeError (slot.py) and neither ladder
    has an ``except VramOverCommitError`` ahead of its ``except RuntimeError``:

      * SSE pre-submit -> HTTP 500, flat ``"sidecar failed: {e}"``
               (its non-stream twins DO catch it first)
      * embeddings.py's ladder -> 503 with a flat string body and a HARDCODED
               ``Retry-After: 5``, never ``exc.retry_after_s``

    Both are LATENT, not live: nothing raises VramOverCommitError
    synchronously -- all three producers in manager.py hand the
    exception to ``_fail_completion_future`` rather than raising it, so neither
    ladder can see one today. Deliberately not handled here: adding handlers for
    an unreachable path would be dead code, and the scope here is the capacity
    discrimination, not the submit ladders. If a future change ever raises this
    exception synchronously out of ``submit_for_streaming``, add
    ``except VramOverCommitError`` to both ladders BEFORE their RuntimeError arm.
    """
    return {
        "type": "capacity_unavailable",
        "cause": exc.cause,
        "message": str(exc),
        "retry_after": exc.retry_after_s,
    }


def _stream_error_frame(error: str, message: str, **extra: Any) -> bytes:
    """Build a synthetic OpenAI-compat SSE error frame.

    OpenAI's streaming wire-format expresses mid-stream errors as a final
    ``data: {"error": {...}}\\n\\n`` chunk followed by ``data: [DONE]\\n\\n``.
    Keeps HTTP 200 once the response has started; the client SDK surfaces
    the error during chunk iteration.

    ``message`` is NOT truncated (no ``message[:500]``
    cap). The non-stream error bodies this module and
    embeddings.py build never truncate ``message`` -- same keys, different
    values for a long message would be the actual defect, not "SSE messages must
    be short". A 500-char cap would not be part of a codebase-wide wire-safety
    policy: it would be the ONLY truncating site against many other unbounded
    ``message``-carrying sites in this file, INCLUDING
    this function's own catch-all ``except Exception`` caller
    (``_stream_error_frame("internal_error", str(e))``, defensive/
    pragma-no-cover) -- so truncating only the SSE copy of that same
    unbounded risk would provide no real defense-in-depth, only an inconsistency.
    If SSE frame size ever becomes a genuine, measured concern, that calls
    for a deliberate, SYMMETRIC truncation policy applied consistently across
    both stream and non-stream bodies -- not a silent asymmetry at one
    function.
    """
    body: dict[str, Any] = {"error": {"type": error, "message": message}}
    body["error"].update(extra)
    return f"data: {json.dumps(body)}\n\n".encode()


async def _openai_chat_completions_stream(
    request: Request,
    mgr,
    model: str,
    messages: list,
    prompt: str,
    thread_id: str,
    payload: dict,
    req_id: str,
    effort_key: "str | None" = None,
) -> StreamingResponse:
    """SSE streaming pass-through.

    Submits via ``manager.submit_for_streaming`` (slot held ACTIVE for full
    stream lifetime — single-slot invariant preserved by
    design). Awaits ``slot.stream_ready_event`` so we know the sidecar
    is up and ``slot.stream_handle`` is populated. Then opens our own
    ``httpx.stream("POST", url, ...)`` to the sidecar and pipes raw SSE bytes
    back to the client. On end-of-stream / disconnect / error sets
    ``slot.stream_done_event`` so the manager can advance ACTIVE → GRACE.

    Wrapper ``_merge_reasoning_into_content`` is intentionally SKIPPED on the
    streaming path: most modern streaming consumers (AI harnesses, agent frameworks, chat
    platforms, OpenAI SDK) parse ``delta.content`` and ``delta.reasoning_content``
    independently. Per-chunk merge would require accumulator/reorder state
    and would break the token-by-token UX.

    ``req_id``: the SAME correlation id the caller's identity_recompose line
    already stamped -- passed in explicitly rather than regenerated here, so
    the streaming leg of one request joins to the same id as the rest of it
    instead of silently minting its own.

    ``effort_key``: the caller already validated
    ``reasoning_effort`` via ``resolve_reasoning_effort`` before the
    stream/non-stream fork; passed through explicitly rather than re-read
    from ``payload`` (which never carries a resolved value under this
    design) so this helper doesn't need its own copy of that validation.
    """
    client_meta = {
        "kind": "openai-chat-completion-stream",
        "messages": messages,
        "model": model,
        "stream": True,
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "max_tokens": payload.get("max_tokens"),
        # Ollama-style keep_alive → IDLE_HOT extension hint.
        # Streaming path is AI harnesses' primary entry; this hint is the
        # whole point of this handling.
        "keep_alive_s": parse_keep_alive(payload.get("keep_alive")),
        # All forwardable knobs carried for the streaming payload helper.
        **{k: payload.get(k) for k in _STREAM_FORWARDED_KNOBS if payload.get(k) is not None},
        # See the non-stream client_meta's identical
        # comment -- same decision-neutral shape, threaded in via the
        # `effort_key` parameter above instead of re-reading payload.
        "reasoning_effort": effort_key,
        # DORMANT identity/classification foundation.
        # Decision-neutral: stored only, read nowhere; NOT in the knob allow-list
        # so it never reaches the llama-server payload. See _derive_client_meta_identity.
        **_derive_client_meta_identity(
            payload, messages,
            ip=request.client.host if (getattr(request, "client", None) and request.client.host) else None,
        ),
        # Observability: carried through to slot.client_meta so
        # _emit_request_identity can re-emit it on R2B_REQ_IDENTITY, joining
        # this layer's identity_recompose line to the manager's. req_id is a
        # PARAMETER here (the caller's own id) -- not regenerated, or the
        # streaming leg would silently get a different id than the request
        # that spawned it.
        "req_id": req_id,
    }

    # Client-disconnect eviction for the STREAMING path.
    # The non-streaming path wires this and the streaming path must too:
    # otherwise a streaming slot that sat QUEUED behind a client
    # that already hung up could never be evicted. Carry a disconnect_event into
    # submit so the queue marks the slot is_evicted on disconnect; the fan-out
    # admit then SKIPS dead-client riders instead of burning a --parallel slot.
    # The watcher task is started after a successful submit (covers the
    # queue-wait window) and cancelled in stream_gen's finally.
    disconnect_event = asyncio.Event()
    # Pre-stream submission errors → standard HTTPException with proper status code.
    try:
        slot = await mgr.submit_for_streaming(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            context=list(messages),
            client_meta=client_meta,
            disconnect_event=disconnect_event,
            admission_ctx_len=compute_ctx_len(messages),
            # CLASSIFIER: incoming turn-hash chain (parallel to admission_ctx_len)
            # so the manager restore path classifies by prefix-VALIDITY vs the pinned
            # clean bin. Same _prefix_hash_chain (unified) used at save time, so
            # both sides of the comparison are byte-identical hashes.
            admission_hash_chain=_prefix_hash_chain(messages),
        )
    except SidecarUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": sidecar_unavailable_error_body(e)},
            headers={"Retry-After": str(_sidecar_unavailable_retry_after_s(mgr, e))},
        ) from e
    except SidecarTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "sidecar_timeout", "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarUpstreamError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_sidecar_error",
                "upstream_status": e.upstream_status,
                "upstream_body": e.upstream_body,
                "message": str(e),
            },
        ) from e
    except VramOverCommitError as e:
        # This ladder has an arm for
        # VramOverCommitError ahead of the bare RuntimeError catch below --
        # since VramOverCommitError subclasses RuntimeError (slot.py), it
        # would otherwise be swallowed flat as a generic 500 instead of the
        # typed, retryable 503 the capacity contract promises. LATENT, not
        # live: submit_for_streaming never raises this synchronously (see
        # capacity_unavailable_error_body's own SCOPE docstring above) -- all
        # three real construct sites in manager.py hand it to
        # _fail_completion_future for async delivery instead. Present for
        # correctness/consistency with the other two (already-safe) ladders,
        # not because this path is reachable today.
        raise HTTPException(
            status_code=503,
            detail={"error": capacity_unavailable_error_body(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"sidecar failed: {e}") from e

    # Submit succeeded: start the disconnect watcher now (covers the queue-wait
    # window before the slot reaches ACTIVE). Created here rather than before the
    # submit so a pre-stream submit failure cannot leak a watcher task. Cancelled
    # in stream_gen's finally.
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))

    async def stream_gen():
        gen_id_for_tee = None
        first_token_received = False
        prefill_start = time.monotonic()
        # Accumulate the streamed generated assistant text (content +
        # reasoning) so the manager can reconstruct the engine-view warm_chain for
        # the STREAMING forced clean-restore. Best-effort, fail-open — a parse miss
        # just leaves the warm state unknown (gate safe-degrades to no-force).
        _gen_content: list[str] = []
        _gen_reasoning: list[str] = []
        _sse_carry = ""
        try:
            # Wait for worker_loop to bring slot to ACTIVE + assign handle.
            # Emit `: keep-alive\n\n` SSE comments every
            # HEARTBEAT_INTERVAL_S so clients with 30-60s read-timeouts don't
            # disconnect during cold-load. asyncio.shield prevents the
            # heartbeat wait_for from cancelling the underlying ready_task.
            ready_task = asyncio.create_task(slot.stream_ready_event.wait())
            loop = asyncio.get_running_loop()
            remaining = SLOT_READY_TIMEOUT_S
            while not ready_task.done():
                # An OOM-requeued, parked slot is exempt
                # from this deadline for as long as it's parked -- mirrors
                # _await_completion_with_oom_exemption's pausable-deadline on
                # the non-streaming path (manager.py). Re-checked every
                # iteration since the flag can flip mid-wait (park ->
                # re-admit -> OOM again).
                exempt = getattr(slot, "oom_requeue_pending", False)
                if not exempt and remaining <= 0:
                    ready_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ready_task
                    yield _stream_error_frame(
                        "slot_ready_timeout",
                        f"Slot did not reach ACTIVE within {SLOT_READY_TIMEOUT_S}s",
                    )
                    yield b"data: [DONE]\n\n"
                    return
                wait_for_s = (
                    HEARTBEAT_INTERVAL_S if exempt else min(HEARTBEAT_INTERVAL_S, remaining)
                )
                _iter_started = loop.time()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(ready_task),
                        timeout=wait_for_s,
                    )
                except asyncio.TimeoutError:
                    # A routing failure (VRAM over-commit / evict-pending
                    # exhaustion) fails the completion_future WITHOUT ever setting
                    # stream_ready_event. Surface it as a prompt, correctly-typed SSE
                    # error frame instead of waiting out SLOT_READY_TIMEOUT_S (up to
                    # 2h) and mislabelling it a slot_ready_timeout.
                    cf = slot.completion_future
                    if (cf is not None and cf.done() and not cf.cancelled()
                            and cf.exception() is not None):
                        exc = cf.exception()
                        ready_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await ready_task
                        if isinstance(exc, VramOverCommitError):
                            # Same nested shape as the
                            # non-stream 503 body (error.{type,cause,message,
                            # retry_after}), via the single shared construction, so
                            # a client has ONE parse rule for the capacity signal.
                            body = capacity_unavailable_error_body(exc)
                            yield _stream_error_frame(
                                body["type"], body["message"],
                                cause=body["cause"], retry_after=body["retry_after"],
                            )
                        else:
                            yield _stream_error_frame("routing_failed", str(exc))
                        yield b"data: [DONE]\n\n"
                        return
                    if not ready_task.done():
                        yield b": keep-alive\n\n"
                if not exempt:
                    remaining -= loop.time() - _iter_started

            handle = slot.stream_handle
            if handle is None:
                # The ready-wait above can unblock for TWO
                # reasons -- a genuine placement success (stream_handle is always
                # set before stream_ready_event, on every success path -- see
                # manager.py:_fail_completion_future) or a placement FAILURE that
                # deliberately wakes this same wait instead of hanging until
                # SLOT_READY_TIMEOUT_S. stream_ready_failed_reason distinguishes
                # them: non-None means the failure case, so show the REAL reason
                # instead of this branch's generic fallback message,
                # which would be wrong for that case (the
                # slot never reached ACTIVE at all).
                if slot.stream_ready_failed_reason is not None:
                    # This handle-is-None branch is
                    # the ONLY reachable capacity-failure site on the SSE path
                    # -- the `except asyncio.TimeoutError` branch above does not
                    # fire for an exception-driven wakeup (the event is set on
                    # failure too, so ready_task.done() is already True by the
                    # time asyncio.wait_for would have raised). Without this,
                    # a VramOverCommitError would silently degrade to a message-only
                    # placement_failed frame (no cause/retry_after). So re-derive
                    # the real exception off completion_future (same guard as
                    # embeddings.py's non-stream twin) so a capacity failure still
                    # emits the typed frame here, not just on the non-stream route.
                    cf = slot.completion_future
                    cf_exc = (
                        cf.exception()
                        if (cf is not None and isinstance(cf, asyncio.Future)
                            and cf.done() and not cf.cancelled()
                            and cf.exception() is not None)
                        else None
                    )
                    if isinstance(cf_exc, VramOverCommitError):
                        body = capacity_unavailable_error_body(cf_exc)
                        yield _stream_error_frame(
                            body["type"], body["message"],
                            cause=body["cause"], retry_after=body["retry_after"],
                        )
                        yield b"data: [DONE]\n\n"
                        return
                    # placement_failed STAYS for every non-capacity placement
                    # failure (stream_ready_failed_reason set, but not a
                    # VramOverCommitError, or the exception is unavailable).
                    yield _stream_error_frame(
                        "placement_failed",
                        slot.stream_ready_failed_reason,
                    )
                    yield b"data: [DONE]\n\n"
                    return
                yield _stream_error_frame(
                    "no_sidecar_handle",
                    "Slot reached ACTIVE but stream_handle is None",
                )
                yield b"data: [DONE]\n\n"
                return

            # Live monitor: unify text-plane identity with the metrics-plane
            # poller (same pid:spawn_seq:slot_id -> same generation_id). slot_id
            # is per-request unique, so turns of one conversation never collide.
            # Use the spawn_seq of the resident actually
            # serving THIS model_tag, not the singleton. At cap>=2 the dispatcher
            # never bumps the singleton's spawn_seq (stays 0) while the metrics
            # supervisor hashes the model_tag resident's bumped spawn_seq; using
            # _read_spawn_seq(mgr) (=singleton) here would make the text-plane tee feed a
            # DIFFERENT generation_id than the SSE anchor, so the live pane would
            # subscribe to an unfed buffer and show nothing. _spawn_seq_for_model
            # unifies both planes (byte-identical at cap<=1).
            gen_id_for_tee = compute_generation_id(
                handle.pid, mgr._spawn_seq_for_model(model), slot.slot_id or slot.thread_id
            )

            stream_payload = _build_stream_payload(client_meta, model, messages)
            url = f"http://127.0.0.1:{handle.port}/v1/chat/completions"

            async with httpx.AsyncClient(timeout=STREAM_TIMEOUT_S) as client:
                # Prefill keep-alive: the silence
                # that disconnects clients is during the STREAM OPEN, not the byte
                # loop. ``client.stream(...).__aenter__()`` blocks until llama-server
                # returns the response, which it withholds for the ENTIRE prefill
                # (no headers/bytes until generation begins -- a long prompt can stay silent for
                # tens of seconds, ALL of it inside the open). So we open the stream
                # MANUALLY and emit ': keep-alive' every HEARTBEAT_INTERVAL_S while the
                # open is pending. asyncio.shield keeps the open alive across a tick
                # (never cancel it on a tick). httpx errors from the open still
                # propagate to the outer except handlers below. Mirrors the existing
                # slot-ready heartbeat pattern.
                stream_cm = client.stream(
                    "POST", url, json=stream_payload, timeout=STREAM_TIMEOUT_S,
                )
                open_task = asyncio.ensure_future(stream_cm.__aenter__())
                try:
                    r = None
                    while r is None:
                        try:
                            r = await asyncio.wait_for(
                                asyncio.shield(open_task), timeout=HEARTBEAT_INTERVAL_S
                            )
                        except asyncio.TimeoutError:
                            # Sidecar still prefilling (no response yet): keep the
                            # client connection warm; keep waiting on the SAME open.
                            yield b": keep-alive\n\n"
                            # Telemetry — keep-alive emission
                            try:
                                mgr_telemetry = getattr(mgr, "_telemetry", None)
                                if mgr_telemetry is not None:
                                    mgr_telemetry.on_keep_alive_emitted(
                                        slot_id=slot.slot_id,
                                        is_queued=False,
                                        thread_id=slot.thread_id,
                                    )
                            except Exception:
                                pass  # observe-only

                    # raise_for_status is NOT auto-called by httpx.stream;
                    # the body may not be loaded yet so we read it manually if
                    # the upstream returned a 4xx/5xx.
                    if r.status_code >= 400:
                        # Cap the error-body read so a stalled
                        # 4xx/5xx body can't block un-heartbeated until STREAM_TIMEOUT_S
                        # (error bodies are tiny + already generated; empty on timeout).
                        try:
                            body_bytes = await asyncio.wait_for(
                                r.aread(), timeout=HEARTBEAT_INTERVAL_S
                            )
                        except asyncio.TimeoutError:
                            body_bytes = b""
                        body_str = body_bytes.decode("utf-8", errors="replace")[:500]
                        yield _stream_error_frame(
                            "upstream_sidecar_error",
                            f"sidecar returned HTTP {r.status_code}",
                            upstream_status=r.status_code,
                            upstream_body=body_str,
                        )
                        yield b"data: [DONE]\n\n"
                        return

                    # Pipe raw SSE bytes from llama-server straight through. Also
                    # heartbeat-guarded (a decode that stalls >12s gets keep-alives);
                    # shield keeps the in-flight read alive across a tick — NEVER
                    # cancel a mid-flight read (could drop bytes / corrupt the stream).
                    aiter = r.aiter_bytes()
                    next_read = asyncio.ensure_future(aiter.__anext__())
                    try:
                        while True:
                            try:
                                chunk_bytes = await asyncio.wait_for(
                                    asyncio.shield(next_read),
                                    timeout=HEARTBEAT_INTERVAL_S,
                                )
                            except asyncio.TimeoutError:
                                yield b": keep-alive\n\n"
                                continue
                            except StopAsyncIteration:
                                break
                            if chunk_bytes:
                                yield chunk_bytes
                                # Telemetry — first token (TTFT)
                                if not first_token_received:
                                    first_token_received = True
                                    ttft_ms = (time.monotonic() - prefill_start) * 1000.0
                                    try:
                                        mgr_telemetry = getattr(mgr, "_telemetry", None)
                                        if mgr_telemetry is not None:
                                            mgr_telemetry.on_first_token(slot, ttft_ms)
                                    except Exception:
                                        pass
                                # Live monitor: passive output-text tee. Client yield
                                # FIRST, tee SECOND, fail-open — never delays/reorders/
                                # corrupts the client stream.
                                try:
                                    mgr.live_output.feed(gen_id_for_tee, chunk_bytes)
                                except Exception:
                                    pass
                                # Parse the SSE deltas to accumulate the
                                # generated content + reasoning (for warm_chain). Yield
                                # + tee happen FIRST; this is passive + fail-open and
                                # never touches the bytes handed to the client.
                                try:
                                    _txt = _sse_carry + chunk_bytes.decode("utf-8", errors="replace")
                                    _lines = _txt.split("\n")
                                    _sse_carry = _lines.pop()  # last (possibly partial) line
                                    for _ln in _lines:
                                        _ln = _ln.strip()
                                        if not _ln.startswith("data:"):
                                            continue
                                        _pl = _ln[5:].strip()
                                        if not _pl or _pl == "[DONE]":
                                            continue
                                        _delta = (json.loads(_pl).get("choices") or [{}])[0].get("delta") or {}
                                        if _delta.get("content"):
                                            _gen_content.append(_delta["content"])
                                        if _delta.get("reasoning_content"):
                                            _gen_reasoning.append(_delta["reasoning_content"])
                                except Exception:
                                    pass  # fail-open: warm state stays unknown -> no force
                            next_read = asyncio.ensure_future(aiter.__anext__())
                    finally:
                        if not next_read.done():
                            next_read.cancel()
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await next_read
                finally:
                    # Always release the stream we opened manually. If the open never
                    # completed (client disconnect mid-prefill), cancel+reap it; ONLY
                    # call __aexit__ when __aenter__ actually succeeded (else there is
                    # nothing entered to exit).
                    if not open_task.done():
                        open_task.cancel()
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await open_task
                    # Compute opened_ok ORDERING-INDEPENDENTLY
                    # (a cancelled/pending task's .exception() RAISES — don't rely on
                    # short-circuit term order surviving a future refactor).
                    opened_ok = False
                    if open_task.done() and not open_task.cancelled():
                        try:
                            opened_ok = open_task.exception() is None
                        except asyncio.CancelledError:
                            opened_ok = False
                    if opened_ok:
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await stream_cm.__aexit__(None, None, None)
        except (
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.CloseError,
            httpx.ProtocolError,
        ) as e:
            # Reuse the SAME IDLE_DEAD_CLASSIFY line name and
            # field set as the manager's own site (that site's docstring:
            # "so every existing grep/tool keyed on that line works
            # identically" -- same intent applies across this third,
            # request-time context). Best-effort: a field lookup must never
            # break the request, mirroring the manager site's own
            # try/except Exception -> log.debug(exc_info=True) shape.
            try:
                log.warning(
                    "IDLE_DEAD_CLASSIFY model_tag=%s port=%s pid=%s prompt_tokens=%s",
                    model,
                    getattr(handle, "port", None),
                    getattr(handle, "pid", None),
                    compute_ctx_len(messages),
                )
            except Exception:
                log.debug(
                    "IDLE_DEAD_CLASSIFY death log failed (best-effort)",
                    exc_info=True,
                )
            yield _stream_error_frame(
                "sidecar_unavailable",
                f"{type(e).__name__}: {e}",
                cause="sidecar_disconnected_or_crashed",
            )
            yield b"data: [DONE]\n\n"
        except httpx.TimeoutException as e:
            yield _stream_error_frame(
                "sidecar_timeout",
                f"{type(e).__name__}: {e}",
            )
            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            # Client disconnected mid-stream — propagate cancellation but
            # ensure cleanup runs in `finally` below. Do NOT yield any
            # additional frames after cancellation (the connection is dead).
            log.info(
                "client disconnect during stream slot=%s thread=%s",
                slot.slot_id, slot.thread_id,
            )
            raise
        except Exception as e:  # pragma: no cover — defensive
            log.exception(
                "unexpected error in stream_gen slot=%s", slot.slot_id,
            )
            yield _stream_error_frame("internal_error", str(e))
            yield b"data: [DONE]\n\n"
        finally:
            # A stream that
            # ends while its slot is still OOM-requeue-parked leaves the
            # background companion (_oom_requeue_wait_then_readmit) polling
            # a disconnect_event that watch_disconnect, cancelled two lines
            # below, will now never get the chance to set for a real
            # disconnect either -- so an ended/cancelled stream would
            # otherwise leave that companion parked forever for nobody.
            # SCOPED to oom_requeue_pending only (not every never-reached-
            # ACTIVE exit): an ORDINARY queued slot's disconnect_event is
            # read by queue.py's own eviction scans too (~9 call sites), so
            # setting it unconditionally here would change today's eviction
            # timing for that unrelated population. Read BEFORE watch_task
            # is cancelled so a genuine last-instant real disconnect it
            # already observed isn't clobbered by this check running first.
            if getattr(slot, "oom_requeue_pending", False):
                disconnect_event.set()
            # Tear down the disconnect watcher. Cancel +
            # await so no "Task was destroyed but it is pending" leak.
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watch_task
            # Stash the accumulated generated turn (with <think>) on the slot
            # BEFORE signaling done, so the manager (which wakes on stream_done_event)
            # computes the engine-view warm_chain for the streaming no-downgrade gate.
            # Use the SINGLE-SOURCE wrap_reasoning_think (the exact
            # _merge_reasoning_into_content newline form) so this shadow reconstruction
            # byte-matches what the harness stores/resends after its think-strip — the
            # no-newline `<think>{r}</think>{c}` form would diverge on multi-block /
            # stray-tag content. Empty -> left None (warm unknown).
            try:
                _c = "".join(_gen_content)
                _r = "".join(_gen_reasoning)
                _merged = wrap_reasoning_think(_r, _c) if (_r and "<think>" not in _c) else _c
                if _merged:
                    slot.streamed_assistant_text = _merged
            except Exception:
                pass
            # Signal worker_loop to advance the slot ACTIVE → GRACE.
            # Idempotent: setting an already-set Event is a no-op.
            if slot.stream_done_event is not None and not slot.stream_done_event.is_set():
                slot.stream_done_event.set()
            # Live monitor: close the output buffer so SSE subscribers get `done`.
            if gen_id_for_tee is not None:
                with contextlib.suppress(Exception):
                    mgr.live_output.mark_done(gen_id_for_tee)

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable buffering at nginx / other reverse proxies so chunks
            # reach the client in real time.
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Ollama-compat /api/chat
# ============================================================================


@router.post("/api/chat")
async def ollama_chat(payload: dict, request: Request) -> dict:
    """Ollama-shape chat. Internally forwarded as OpenAI to llama-server then
    re-shaped to Ollama on return."""
    # Shallow copy to avoid mutating caller's payload dict
    payload = dict(payload)
    mgr = request.app.state.manager
    model = payload.get("model")
    messages = payload.get("messages")
    if not model or not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`model` + `messages` required")
    # The model tag must exist in the manifest store — mirrors the
    # read_manifest -> 404 pattern at embeddings.py:119 / ollama.py:68.
    try:
        _manifest = read_manifest(mgr.boot.storage.manifests_path, model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {model}") from e
    except ManifestValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (yaml.YAMLError, ValidationError, UnicodeDecodeError, OSError) as e:
        # See openai_chat_completions above -- identical rule, identical
        # non-leaking handled-500 shape (models.py:get_model precedent).
        log.exception("manifest for %r is present but unreadable", model)
        raise HTTPException(
            status_code=500, detail=f"manifest for '{model}' is present but unreadable"
        ) from e
    # Warn (at most once per model+caller) when a locked
    # reasoning_budget meets or exceeds the effective output ceiling.
    # ollama_chat always calls submit_and_wait regardless of the client's
    # stream flag (see the "stream": False comment further down), so this one
    # unconditional call already covers every ollama_chat request.
    _caller_max_tokens = payload.get("max_tokens")
    if _caller_max_tokens is None and isinstance(payload.get("options"), dict):
        # Native Ollama shape nests the ceiling under options.num_predict
        # instead of a top-level max_tokens -- same unwrap-with-top-level-
        # fallback idiom as the keep_alive/options handling below. Used for
        # the ceiling computation only (this warning AND the
        # clamp below) -- does not change what num_predict/max_tokens
        # themselves forward downstream via _COMMON_FORWARDED_KNOBS.
        _caller_max_tokens = payload["options"].get("num_predict")
    warn_if_reasoning_budget_exceeds_ceiling(
        model,
        _manifest.llama_server_flags or {},
        _caller_max_tokens,
        thread_id=payload.get("thread_id") or None,
        ip=request.client.host if (getattr(request, "client", None) and request.client.host) else None,
    )
    # Request-time clamp, same contract as the openai_chat_completions
    # site -- None ⇒ untouched (negative control), only set
    # payload["reasoning_budget"] when a clamp is actually needed. This DOES
    # affect _COMMON_FORWARDED_KNOBS downstream (unlike the ceiling unwrap
    # above), since reasoning_budget itself is forwarded from payload.
    _clamped_reasoning_budget = clamp_reasoning_budget_for_ceiling(
        _manifest.llama_server_flags or {}, _caller_max_tokens,
    )
    if _clamped_reasoning_budget is not None:
        payload["reasoning_budget"] = _clamped_reasoning_budget
    # Same validation-only call as the
    # openai_chat_completions site -- no arithmetic, no payload write. See
    # that site's comment for the full reasoning.
    _effort_key = resolve_reasoning_effort(payload)
    # OpenAI tool_call messages have content=null (None),
    # not "". Coerce None to "" to tolerate tool-call assistant turns.
    prompt = " ".join(
        (c if isinstance(c := m.get("content"), str) else "")
        for m in messages if isinstance(m, dict)
    )
    thread_id = payload.get("thread_id") or ""
    # Stable per-agent identity = the client
    # container source IP, so an agent's sequential turns grace-MATCH the same warm
    # slot and reuse its KV prefix cache (cache_reuse) instead of re-prefilling the
    # full context each turn. GATED on single-residency (cap<=1): at cap>=2 the
    # full-prompt-hash identity (manager.submit) is kept so concurrent
    # sub-agent fan-out is NOT regressed. Falls back to hash if no IP.
    try:
        _single_residency = request.app.state.manager.runtime.queue.max_parallel_sidecars <= 1
    except Exception:
        _single_residency = True
    if not thread_id and _single_residency:
        # MULTI-SIGNAL identity: IP (role tier) + fingerprint of the
        # STATIC first message (instance identity). IP-only collapses concurrent sub-agents
        # into ONE thread_id -> cross-context KV mismatch -> positive stale -> CLEAR+reprefill.
        # First-message fingerprint distinguishes sub-agents by persona AND keeps a
        # conversation's follow-ups (same first msg + larger ctx) on ONE id -> KV reuse fires.
        _ip = request.client.host if (getattr(request, "client", None) and request.client.host) else ""
        _first_msg = next(
            (m.get("content", "") for m in (messages or [])
             if isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("content").strip()),
            "",
        )
        if _first_msg:
            _fp = derive_thread_id_prefix_hash(_first_msg, payload.get("model", ""))
            thread_id = ("agent-ip-" + _ip + "-" if _ip else "") + _fp
        elif _ip:
            thread_id = "agent-ip-" + _ip
    # === DORMANT identity-shadow (decision-neutral) ==============
    # thread_id (old_key) is now finalized. Compute a role-keyed CANDIDATE and
    # LOG old-vs-new ONLY — the shadow is NEVER assigned to thread_id nor passed
    # anywhere (activation turns it on once it is shown to only SPLIT, never
    # MERGE). Best-effort: any failure is swallowed so it can't break a request.
    #
    # Observability: _req_id is generated OUTSIDE the try so it
    # survives even if the identity_recompose block itself raises -- it is
    # stashed into client_meta below and re-emitted by the manager's
    # R2B_REQ_IDENTITY line, so the two layers' identity views can be joined
    # by grep under load without depending on thread_id staying stable in
    # between (which is exactly what can legitimately change here).
    _req_id = uuid.uuid4().hex[:12]
    _role_source = "absent"
    _session_source = "absent"
    try:
        old_key = thread_id
        role = payload.get("role")
        session_id = payload.get("session_id")
        if role:
            _role_source = "payload"
        if session_id is not None:
            _session_source = "payload"
        # Recover session_id from idle_hot resident when
        # payload omits it. Tool-call re-cues arrive with session_id=None after
        # grace timer fires, but the warm idle resident has the original
        # session_id. Match the incoming IP+fingerprint base thread_id against
        # the idle resident's thread_id (which has the s={sha256} suffix).
        # Without this recovery, _shadow_recompose_identity produces a session-
        # LESS thread_id, durable_ring_key returns None, KV-bin owner-match
        # fails vs the session-present bin -> forced-full -> a full re-prefill.
        if session_id is None:
            try:
                mgr = request.app.state.manager
                if (getattr(mgr, "_idle_handle", None) is not None
                        and getattr(mgr, "_idle_client_meta", None)
                        and getattr(mgr, "_idle_thread_id", None)):
                    _idle_cm = mgr._idle_client_meta
                    _recovered = _idle_cm.get("session_id") if isinstance(_idle_cm, dict) else None
                    _idle_tid = mgr._idle_thread_id
                    # Match: incoming base thread_id is a prefix of the idle
                    # thread_id (idle has the s-suffix from its original
                    # derivation, incoming doesn't yet). Only recover when the
                    # base matches — don't hijack a different agent's session.
                    if _recovered and _idle_tid.startswith(old_key):
                        session_id = _recovered
                        _session_source = "idle-hot-recovery"
                        log.info(
                            "session_id recovered from idle-hot resident: "
                            "thread_base=%s idle_thread=%s",
                            old_key[:24], _idle_tid[:24],
                        )
            except Exception:
                log.debug("session_id recovery failed (ignored)", exc_info=True)
        new_key = _shadow_recompose_identity(old_key, role, session_id)
        # Role-keyed activation: when TURBOHAUL_M2B_ACTIVE, the role+session-keyed new_key
        # DRIVES the slot identity (warm-path role isolation). resolve_kv / the restore
        # gate CODE is byte-identical — ONLY the thread_id fed IN changes. Append-only
        # (new_key STARTS WITH old_key), so it can only split: a SPLIT mints
        # a NEW identity (-> no bin -> fresh reprefill + saves own, never a wrong-
        # restore); a stable-key continuation keeps warm reuse. Flag-gated so it can be
        # flipped OFF instantly on any MISMATCH/merge.
        _m2b = _m2b_active()
        if _m2b and new_key != old_key:
            thread_id = new_key
        log.info(
            "identity_recompose old_h=%s new_h=%s role=%s role_source=%s "
            "session_present=%s session_source=%s ip_present=%s m2b_active=%s "
            "req_id=%s",
            hashlib.sha256(old_key.encode()).hexdigest()[:12],
            hashlib.sha256(new_key.encode()).hexdigest()[:12],
            role if role else "-",
            _role_source,
            bool(session_id),
            _session_source,
            bool(locals().get("_ip")),
            _m2b,
            _req_id,
        )
    except Exception:
        log.debug("identity_recompose failed (ignored)", exc_info=True)
    # Ollama accepts keep_alive at top level OR nested under options.
    ka_raw = payload.get("keep_alive")
    if ka_raw is None and isinstance(payload.get("options"), dict):
        ka_raw = payload["options"].get("keep_alive")
    # response_format pre-validation. Fires for BOTH stream
    # + non-stream and BEFORE the streaming-tools guard below so the deferred
    # type ('json_schema') gets a clean 400 regardless of the stream flag.
    # Mirror of openai_chat_completions validator; minor wording tweak in the
    # detail for endpoint clarity.
    rf = payload.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "response_format_unsupported_type",
                    "message": "response_format must be an object",
                    "received": type(rf).__name__,
                },
            )
        rf_type = rf.get("type")
        if rf_type == "text":
            payload["response_format"] = None  # OpenAI default — no-op
        elif rf_type == "json_object":
            pass  # accept-and-forward
        elif rf_type == "json_schema":
            # Mirror of openai_chat_completions json_schema branch.
            ok, reason = _validate_json_schema(rf)
            if not ok:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "schema_validation_failed",
                        "message": f"json_schema validation failed: {reason}",
                    },
                )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "response_format_unsupported_type",
                    "message": (
                        "response_format type must be one of "
                        "'text', 'json_object', 'json_schema'"
                    ),
                    "received_type": str(rf_type),
                },
            )
    # Streaming + tools is deferred
    # to a follow-on change. Cheap defensive guard before submit_and_wait so
    # callers get a clean 400 instead of a confusing partial-tool stream.
    if payload.get("stream") and any(
        payload.get(k)
        for k in ("tools", "tool_choice", "parallel_tool_calls",
                  "function_call", "functions")
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "streaming_with_tools_deferred",
                "message": (
                    "Ollama-shape streaming + tool_calls is not yet supported. "
                    "Use stream=false for tool requests or /v1/chat/completions "
                    "for OpenAI-shape streaming-tools."
                ),
                "follow_on_rc": "streaming-tools",
            },
        )
    # Mirror of the openai route's manifest read — slice reasoning_budget
    # for in-_complete retry-path gate. Only fires for json_schema requests.
    thinking_manifest: dict = {}
    if (
        isinstance(payload.get("response_format"), dict)
        and payload["response_format"].get("type") == "json_schema"
    ):
        try:
            _m = read_manifest(mgr.boot.storage.manifests_path, model)
            thinking_manifest = {
                "reasoning_budget": (_m.llama_server_flags or {}).get(
                    "reasoning_budget", 0,
                ),
            }
        except Exception:
            log.exception(
                "manifest read failed for model=%s (ollama_chat); retry-path disabled",
                model,
            )
    # Build the knob dict with the
    # SAME comprehension the stream path (_build_stream_payload) uses, so
    # the non-stream ollama_chat client_meta is byte-identical to the stream
    # path. A hand-enumerated list would drop n_predict (and sampler knobs
    # like presence_penalty, frequency_penalty, repeat_penalty, etc.) —
    # fields that ARE in _COMMON_FORWARDED_KNOBS but would never be copied here,
    # so _complete's _COMMON_FORWARDED_KNOBS loop would find nothing and the
    # 300-token cap (n_predict) would be silently dropped on non-stream requests.
    forwarded_knobs = {
        k: payload.get(k) for k in _COMMON_FORWARDED_KNOBS
        if payload.get(k) is not None
    }
    client_meta = {
        "kind": "ollama-chat",
        "messages": messages,
        "model": model,
        # ollama_chat has no streaming implementation --
        # no SSE/NDJSON consumer, no branch to a streaming helper, always calls
        # submit_and_wait (the same entrypoint used for stream=false) -- so this
        # MUST be False regardless of what the client requested. Passing the raw
        # client flag through here would arm manager.submit()'s streaming
        # handshake (stream_ready_event/stream_done_event) that only
        # submit_for_streaming's caller-side protocol ever fulfills, silently
        # deadlocking every ollama_chat stream=true request for
        # manager._STREAM_TIMEOUT_S (3600s) before returning a placeholder
        # {"_streamed": True} instead of the real completion. manager.submit()
        # also gates arming on an explicit, wrapper-controlled
        # arm_stream_events flag (defense in depth), but this field must still
        # tell the truth: no code downstream of submit_and_wait ever streams.
        # stream_requested keeps the client's own intent for observability
        # without re-arming anything that can't be fulfilled.
        "stream": False,
        "stream_requested": bool(payload.get("stream", False)),
        "options": payload.get("options"),
        "keep_alive_s": parse_keep_alive(ka_raw),
        # Forward validated response_format. Same
        # rationale as the openai route — handler-entry validator alone does not propagate
        # the field; the explicit add line keeps the non-stream ollama path
        # honest.
        "response_format": payload.get("response_format"),
        # Manifest reasoning_budget slice for in-_complete
        # is_thinking_payload gate. Mirrors the openai route; {} ⇒ retry disabled.
        "thinking_manifest": thinking_manifest,
        # All _COMMON_FORWARDED_KNOBS (temperature, top_p, top_k, max_tokens,
        # min_p, thinking_budget_tokens, reasoning_budget, reasoning,
        # presence_penalty, frequency_penalty, repeat_penalty, repeat_last_n,
        # typical_p, seed, mirostat, mirostat_lr, mirostat_ent, n_predict,
        # tools, tool_choice, parallel_tool_calls, function_call, functions)
        # forwarded via the comprehension above — byte-identical to stream path.
        **forwarded_knobs,
        # The validated, case-folded reasoning_effort
        # key, for the manager to resolve at SPAWN time only (see
        # resolve_reasoning_budget_for_spawn). Deliberately NOT in
        # _COMMON_FORWARDED_KNOBS -- same "decision-neutral, not in the knob
        # allow-list" shape as the identity fields just below -- so it never
        # reaches the llama-server payload for a warm resident (first-caller
        # -wins; every request after the first is a no-op read of this key).
        "reasoning_effort": _effort_key,
        # DORMANT identity/classification foundation.
        # Decision-neutral: stored only, read nowhere; NOT in the knob allow-list
        # so it never reaches the llama-server payload. See _derive_client_meta_identity.
        **_derive_client_meta_identity(
            payload, messages,
            ip=request.client.host if (getattr(request, "client", None) and request.client.host) else None,
        ),
        # Observability: carried through to slot.client_meta so
        # _emit_request_identity can re-emit it on R2B_REQ_IDENTITY, joining
        # this layer's identity_recompose line to the manager's.
        "req_id": _req_id,
    }
    # Client-disconnect watcher (ollama_chat mirror of the openai route).
    disconnect_event = asyncio.Event()
    watch_task = asyncio.create_task(watch_disconnect(request, disconnect_event))
    try:
        slot, result = await mgr.submit_and_wait(
            model_tag=model,
            prompt=prompt,
            thread_id=thread_id,
            context=list(messages),
            client_meta=client_meta,
            disconnect_event=disconnect_event,
            admission_ctx_len=compute_ctx_len(messages),
            # CLASSIFIER: incoming turn-hash chain (parallel to admission_ctx_len)
            # so the manager restore path classifies by prefix-VALIDITY vs the pinned
            # clean bin. Same _prefix_hash_chain (unified) used at save time, so
            # both sides of the comparison are byte-identical hashes.
            admission_hash_chain=_prefix_hash_chain(messages),
        )
    except SlotEvictedError as e:
        # Client closed before activation → HTTP 499
        raise HTTPException(
            status_code=499,
            detail={"error": "client_closed_request", "message": str(e)},
        ) from e
    except SidecarUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": sidecar_unavailable_error_body(e)},
            headers={"Retry-After": str(_sidecar_unavailable_retry_after_s(mgr, e))},
        ) from e
    except SidecarTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "sidecar_timeout", "message": str(e)},
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except SidecarUpstreamError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_sidecar_error",
                "upstream_status": e.upstream_status,
                "upstream_body": e.upstream_body,
                "message": str(e),
            },
        ) from e
    except VramOverCommitError as e:
        # No VRAM capacity — retryable 503 + Retry-After (mirror of openai).
        raise HTTPException(
            status_code=503,
            detail={"error": capacity_unavailable_error_body(e)},  # single source of truth
            headers={"Retry-After": str(e.retry_after_s)},
        ) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"sidecar failed: {e}") from e
    finally:
        # Tear down disconnect watcher (ollama_chat).
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watch_task

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="no completion_fn wired - production needs make_llama_server_complete_fn",
        )

    # Adapt OpenAI-shape response → Ollama shape
    # with full tool-call translation, arg JSON coercion (lenient on
    # failure), size-cap, id preservation, finish_reason
    # → done_reason mapping, and ISO-8601 created_at.
    # Error guard: if completion_fn surfaced an error dict (no choices),
    # pass it through unchanged so the caller still sees the upstream detail.
    if isinstance(result, dict) and "error" in result:
        return result
    if "choices" in result and result["choices"]:
        choice = result["choices"][0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content", "")
        ollama_msg: dict[str, Any] = {"role": "assistant", "content": content}

        # tool_calls translation: OpenAI emits arguments as a JSON-encoded
        # string; Ollama clients expect a parsed object. Enrich
        # the warning with model + truncated args so we can identify which
        # model is misbehaving in the field.
        openai_tcs = msg.get("tool_calls")
        if openai_tcs:  # only emit the key when there's at least one call
            ollama_tcs = []
            for tc in openai_tcs:
                fn = tc.get("function") or {}
                args_str = fn.get("arguments", "{}")
                if isinstance(args_str, str) and len(args_str) > MAX_TOOL_ARG_CHARS:
                    log.warning(
                        "ollama_chat tool_call args >%dB (got %dB) for model=%r, truncating",
                        MAX_TOOL_ARG_CHARS, len(args_str), model,
                    )
                    args_str = args_str[:MAX_TOOL_ARG_CHARS]
                try:
                    args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError as e:
                    truncated = args_str[:80] if isinstance(args_str, str) else repr(args_str)[:80]
                    log.warning(
                        "ollama_chat: json.loads failed on tool_call args "
                        "for model=%r args[:80]=%r err=%s",
                        model, truncated, e,
                    )
                    args_obj = args_str  # lenient: pass raw string
                tc_entry: dict[str, Any] = {
                    "function": {"name": fn.get("name"), "arguments": args_obj},
                }
                if tc.get("id"):  # preserve id when present
                    tc_entry["id"] = tc["id"]
                ollama_tcs.append(tc_entry)
            ollama_msg["tool_calls"] = ollama_tcs

        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        done_reason_map = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "stop",
            "function_call": "stop",
        }
        done_reason = done_reason_map.get(finish_reason, "stop")

        return {
            "model": model,
            "created_at": _coerce_created_at(result.get("created")),
            "message": ollama_msg,
            "done": True,
            "done_reason": done_reason,
            "thread_id": slot.thread_id,
        }
    # Pass-through if completion_fn returned Ollama-native
    return result


# ============================================================================
# Production completion_fn factory: httpx → llama-server child port
# ============================================================================


def _merge_reasoning_into_content(
    result: Any, response_format: dict | None = None,
) -> None:
    """Merge thinking-model reasoning_content into content.

    Thinking-models (Qwen3, deepseek-r1, Gemma-thinking, etc.) split output
    between `message.content` (final answer, often empty during thinking)
    and `message.reasoning_content` (the chain-of-thought). Client parsers
    that read only `.content` (AI harness workers, agent-framework defaults,
    OpenAI SDK) see empty and bail → retry storm → no usable output.

    Wrap reasoning_content inline as `<think>...</think>` tags so EVERY
    client sees a non-empty content string. Preserve reasoning_content
    untouched for clients that explicitly read it (chat platforms, etc.).

    No-op if reasoning_content is empty (non-thinking models) or content
    is already populated alongside reasoning_content (some configs).

    Clients that read only ``message.content`` otherwise see an empty reply.

    Skip merge when the caller requested
    `response_format = {"type": "json_object"}`. The `<think>...</think>`
    wrapper would prepend non-JSON tokens to the content and break the
    contract that response_format clients depend on. The retry path owns the
    thinking-mode JSON guarantee (blocked upstream on llama.cpp #20345 +
    Ollama #10538); this function's job is at minimum not to corrupt the path.
    """
    if not isinstance(result, dict):
        return
    # Structured-output skip-branch —
    # see docstring. The skip applies to json_object and also
    # json_schema; prepending <think>...</think> would corrupt JSON
    # under either response_format type.
    if (
        isinstance(response_format, dict)
        and response_format.get("type") in ("json_object", "json_schema")
    ):
        return
    choices = result.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        rc = msg.get("reasoning_content") or ""
        ct = msg.get("content") or ""
        if not isinstance(rc, str) or not isinstance(ct, str):
            continue
        rc_stripped = rc.strip()
        if not rc_stripped:
            continue  # no thinking to merge
        # Single-source formatter. Behaviour-preserving extraction —
        # wrap_reasoning_think reproduces this site's exact bytes (rc.strip() + the
        # ct.strip() with/without-content branch), so the client-facing response does
        # not change one byte. The shadow sites call the SAME
        # helper, so a shadow reconstruction can never drift from what the harness
        # stores/resends.
        msg["content"] = wrap_reasoning_think(rc, ct)


def make_llama_server_complete_fn(
    timeout_s: float = 600.0,
    http_client_factory=None,
):
    """Build a completion_fn that forwards to the active sidecar's port via httpx.

    Used by main.py to wire the production completion_fn. Tests typically inject
    a simpler fake instead.
    """
    async def _complete(slot, handle):
        client_meta = slot.client_meta or {}
        # Outbound messages come from the request-scoped slot.context, set once
        # at admission (Slot.new) and never reassigned afterward — NOT from
        # client_meta, which a same-thread warm-inherit may replace with a
        # different caller's identity fields. Reading messages off a mutable,
        # inheritable field would let an identity substitution become a content
        # substitution; this keeps the two decoupled.
        messages = slot.context
        if not messages:
            return None
        payload = {
            "model": slot.model_tag,
            "messages": _collapse_trailing_assistant_messages(messages),
            "stream": False,  # streaming SSE is a follow-on polish wave
        }
        # Iterate the canonical
        # _COMMON_FORWARDED_KNOBS list (which already covers tools/tool_choice/
        # parallel_tool_calls/function_call/functions). A separate
        # open-coded knob tuple here could drift from the source
        # of truth and silently drop tools knobs. Single-list invariant.
        # Stream-only knobs (`stream`, `stream_options`) are deliberately
        # excluded; this path is non-streaming.
        for k in _COMMON_FORWARDED_KNOBS:
            v = client_meta.get(k)
            if v is not None:
                payload[k] = v
        url = f"http://127.0.0.1:{handle.port}/v1/chat/completions"
        if http_client_factory is not None:
            client_cm = http_client_factory()
        else:
            client_cm = httpx.AsyncClient(timeout=timeout_s)
        try:
            async with client_cm as client:
                r = await client.post(url, json=payload, timeout=timeout_s)
                r.raise_for_status()
                result = r.json()

                # Bounded validate+retry for thinking-mode
                # json_schema callers. ONE retry max with enable_thinking=False
                # overlay (works around llama.cpp #20345 grammar-inactive bug).
                # Retry shares this try/except so httpx errors map to existing
                # SidecarUnavailableError / SidecarUpstreamError /
                # SidecarTimeoutError handlers below — no duplicate classifier.
                rf = client_meta.get("response_format")
                thinking_manifest = client_meta.get("thinking_manifest") or {}
                if (
                    isinstance(rf, dict)
                    and rf.get("type") == "json_schema"
                    and is_thinking_payload(payload, thinking_manifest)
                ):
                    # Lazy import — fail-soft if dep absent.
                    try:
                        from jsonschema import (
                            Draft202012Validator,
                            ValidationError,
                        )
                    except ImportError:
                        Draft202012Validator = None
                        ValidationError = Exception  # type: ignore[assignment,misc]
                    if Draft202012Validator is not None:
                        schema = rf.get("json_schema", {}).get("schema") or {}
                        validate_failed = False
                        try:
                            first_content = _strip_thinking_wrapper(
                                result["choices"][0]["message"].get("content") or ""
                            )
                            Draft202012Validator(schema).validate(
                                json.loads(first_content)
                            )
                        except (
                            json.JSONDecodeError,
                            ValidationError,
                            KeyError,
                            IndexError,
                            TypeError,
                        ):
                            validate_failed = True

                        if validate_failed:
                            # ONE retry: enable_thinking=False overlay.
                            retry_payload = dict(payload)
                            # Validate chat_template_kwargs before forwarding
                            # to sidecar — only allow known safe keys, strip Jinja
                            # constructs that could SSTI via llama-server.
                            _raw_ctk = payload.get("chat_template_kwargs") or {}
                            _safe_ctk = {
                                "enable_thinking": False,
                            }
                            if isinstance(_raw_ctk, dict):
                                for k, v in _raw_ctk.items():
                                    if not isinstance(k, str):
                                        continue
                                    # Reject any value containing Jinja constructs
                                    if isinstance(v, str) and ("{{" in v or "{%" in v):
                                        continue
                                    # Only allow known safe keys (whitelist)
                                    if k in ("enable_thinking",):
                                        _safe_ctk[k] = v
                            retry_payload["chat_template_kwargs"] = _safe_ctk
                            retry_r = await client.post(
                                url, json=retry_payload, timeout=timeout_s,
                            )
                            retry_r.raise_for_status()
                            retry_result = retry_r.json()
                            try:
                                retry_content = _strip_thinking_wrapper(
                                    retry_result["choices"][0]["message"].get("content") or ""
                                )
                                Draft202012Validator(schema).validate(
                                    json.loads(retry_content)
                                )
                            except (
                                json.JSONDecodeError,
                                ValidationError,
                                KeyError,
                                IndexError,
                                TypeError,
                            ):
                                raise SidecarUpstreamError(
                                    "model_jsonschema_noncompliance_after_retry",
                                    upstream_status=200,
                                    upstream_body=str(retry_result)[:500],
                                )
                            # Retry validated — use retry result.
                            result = retry_result

                # Pass response_format through so the
                # merger can short-circuit for json_object + json_schema callers.
                _merge_reasoning_into_content(
                    result, client_meta.get("response_format"),
                )
                # Recover Qwen-class text-JSON tool calls into
                # structured `tool_calls`. No-op when upstream already
                # populated tool_calls (idempotency) or no tools advertised.
                maybe_recover_tool_calls(result, client_meta.get("tools"))
                return result
        except httpx.HTTPStatusError as e:
            # Sidecar accepted the request but returned 4xx/5xx (context
            # overflow, malformed payload, etc.). Convert to typed error
            # so the route handler can return 502 Bad Gateway with the
            # upstream status preserved.
            raise SidecarUpstreamError(
                f"sidecar returned HTTP {e.response.status_code}",
                upstream_status=e.response.status_code,
                upstream_body=e.response.text,
            ) from e
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
                httpx.ConnectError, httpx.NetworkError, httpx.CloseError,
                httpx.ProtocolError) as e:
            # Sidecar disconnected / crashed / port closed. Most often
            # KV-cache OOM mid-response under heavy load.
            # Convert to 503 + Retry-After.
            # Reuse the SAME IDLE_DEAD_CLASSIFY line name and
            # field set as the manager's own sites and this file's streaming
            # twin above -- one grep key for every "concluded the engine is
            # dead" site in the codebase. Best-effort, same shape as the
            # manager.py template: a field lookup must never break the
            # request.
            try:
                log.warning(
                    "IDLE_DEAD_CLASSIFY model_tag=%s port=%s pid=%s prompt_tokens=%s",
                    slot.model_tag,
                    getattr(handle, "port", None),
                    getattr(handle, "pid", None),
                    compute_ctx_len(messages),
                )
            except Exception:
                log.debug(
                    "IDLE_DEAD_CLASSIFY death log failed (best-effort)",
                    exc_info=True,
                )
            raise SidecarUnavailableError(
                # Lead with the human-meaningful cause,
                # not the raw transport exception name -- a reader skimming
                # "RemoteProtocolError" reads it as a networking bug, when
                # the actual cause is that the engine
                # crashed. The exception detail is kept, just demoted to a
                # parenthetical for whoever is actually debugging the
                # transport layer. cause= below is the stable machine-readable value --
                # it is not affected by this wording.
                f"sidecar crashed or disconnected "
                f"(transport error: {type(e).__name__}: {e})",
                cause="sidecar_disconnected_or_crashed",
                retry_after_s=30,
            ) from e
        except (httpx.TimeoutException,) as e:
            # Includes ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout.
            raise SidecarTimeoutError(
                f"sidecar request timed out after {timeout_s}s: {type(e).__name__}",
                retry_after_s=60,
            ) from e

    return _complete
