""" tool-call recovery post-processor.

Some chat-templated GGUF builds (notably Qwen3 family on llama.cpp jinja
runners — see upstream issues #20809 / #20837 / #20260) emit tool-call
attempts as text JSON inside ``message.content`` instead of populating
the structured ``message.tool_calls`` field. Clients that read only the
structured field (Hermes-class workers, OpenAI SDK, LangChain default)
see "no tool call" and bail.

This module's :func:`maybe_recover_tool_calls` runs AFTER the slot
returns, AFTER ``_merge_reasoning_into_content``, and BEFORE the route
returns ``result``. It restores structured ``tool_calls`` when:

  * The request advertised tools.
  * Upstream did NOT already populate ``tool_calls`` (idempotency).
  * The content (post-``</think>``) contains JSON whose ``name`` is in
    the request's tools allowlist and whose ``arguments`` parses as JSON.

It strips the matched JSON from ``content`` and flips ``finish_reason``
to ``tool_calls``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)

# Canonical OpenAI/Ollama text-JSON shape: {"name": "...", "arguments": {...}}
# Non-greedy on the body; we revalidate by json.loads on each candidate.
_CANONICAL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"arguments"\s*:\s*(?P<args>\{.*?\})\s*\}',
    re.DOTALL,
)

# Qwen <tool_call>...</tool_call> XML wrapper around canonical JSON.
_XML_RE = re.compile(
    r'<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>',
    re.DOTALL,
)


def _extract_known_names(tools_list: Any) -> set[str]:
    """Return the set of function names advertised in the request's tools list.

    Accepts the raw OpenAI/Ollama shape:
        [{"type": "function", "function": {"name": "...", ...}}, ...]
    """
    names: set[str] = set()
    if not isinstance(tools_list, list):
        return names
    for entry in tools_list:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _balanced_object_at(text: str, start: int) -> int | None:
    """Return the index ONE PAST the matching close-brace for an opening
    brace at ``text[start]``, respecting nested braces and string literals.

    Returns ``None`` if the object is unbalanced.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    i = start
    in_str = False
    escape = False
    while i < len(text):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def _iter_candidates(segment: str) -> list[tuple[int, int, str, str]]:
    """Yield ``(start, end, name, args_json)`` tuples for every text-JSON
    candidate in ``segment``. Handles XML wrapper + canonical shape.

    Uses brace-balancing to capture the FULL arguments object (the regex's
    non-greedy ``.*?`` can clip nested objects); the regex is only used to
    locate candidate starts.
    """
    found: list[tuple[int, int, str, str]] = []

    # XML wrapper first — outer span includes the tags so we strip them too.
    for m in _XML_RE.finditer(segment):
        body_start = m.start("body")
        end = _balanced_object_at(segment, body_start)
        if end is None:
            continue
        body = segment[body_start:end]
        try:
            parsed = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        args = parsed.get("arguments")
        if not isinstance(name, str):
            continue
        # arguments may already be a dict — re-serialize to canonical JSON.
        if isinstance(args, dict):
            args_json = json.dumps(args)
        elif isinstance(args, str):
            try:
                json.loads(args)
            except (ValueError, json.JSONDecodeError):
                continue
            args_json = args
        else:
            continue
        # Outer span: from `<tool_call>` to `</tool_call>` inclusive.
        outer_end = segment.find("</tool_call>", end)
        if outer_end == -1:
            continue
        outer_end += len("</tool_call>")
        found.append((m.start(), outer_end, name, args_json))

    # Canonical shape — skip ranges already covered by XML hits.
    covered = [(s, e) for s, e, _, _ in found]

    def _in_covered(pos: int) -> bool:
        return any(s <= pos < e for s, e in covered)

    for m in _CANONICAL_RE.finditer(segment):
        if _in_covered(m.start()):
            continue
        name = m.group("name")
        args_start = m.start("args")
        args_end = _balanced_object_at(segment, args_start)
        if args_end is None:
            continue
        args_blob = segment[args_start:args_end]
        try:
            parsed_args = json.loads(args_blob)
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(parsed_args, dict):
            continue
        # End of the wrapping object — first '}' after args_end (allowing whitespace).
        outer_end = args_end
        while outer_end < len(segment) and segment[outer_end] in " \t\r\n":
            outer_end += 1
        if outer_end >= len(segment) or segment[outer_end] != "}":
            continue
        outer_end += 1
        found.append((m.start(), outer_end, name, json.dumps(parsed_args)))

    found.sort(key=lambda t: t[0])
    return found


def maybe_recover_tool_calls(
    result: Any,
    tools_list: Any,
) -> None:
    """Mutate ``result`` in place to restore structured ``tool_calls``
    when the upstream model emitted them as text JSON in ``content``.

    No-op (returns unchanged) when:
      * ``result`` is not a dict with at least one choice.
      * Upstream already populated ``choices[0].message.tool_calls``
        (idempotency — protect against future llama.cpp upstream fix).
      * The request advertised no tools.
      * No text-JSON candidate matches the tools allowlist.
    """
    if not isinstance(result, dict):
        return
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        return
    msg = choice.get("message")
    if not isinstance(msg, dict):
        return

    # Idempotency gate — upstream already produced structured tool_calls.
    existing = msg.get("tool_calls")
    if isinstance(existing, list) and len(existing) > 0:
        log.debug(
            "tool_call_recovery: skip — upstream already populated tool_calls "
            "(count=%d)", len(existing),
        )
        return

    known = _extract_known_names(tools_list)
    if not known:
        log.debug("tool_call_recovery: skip — no tools advertised on request")
        return

    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        log.debug("tool_call_recovery: skip — empty content")
        return

    # Reasoning-text guard: only scan AFTER </think>. Pre-</think> mentions
    # are reasoning prose, not real tool calls.
    think_close = "</think>"
    think_idx = content.rfind(think_close)
    if think_idx >= 0:
        segment_start = think_idx + len(think_close)
        segment = content[segment_start:]
    else:
        segment_start = 0
        segment = content

    candidates = _iter_candidates(segment)
    if not candidates:
        log.debug(
            "tool_call_recovery: no-match — known=%s content_len=%d",
            sorted(known), len(content),
        )
        return

    accepted: list[tuple[int, int, str, str]] = []
    for start, end, name, args_json in candidates:
        if name not in known:
            log.debug(
                "tool_call_recovery: reject — name=%r not in allowlist=%s",
                name, sorted(known),
            )
            continue
        # args_json was already validated as parseable JSON object in _iter_candidates.
        log.debug(
            "tool_call_recovery: match — name=%s args_parse_ok=True",
            name,
        )
        accepted.append((start, end, name, args_json))

    if not accepted:
        return

    # Build structured tool_calls (OpenAI shape).
    tool_calls = []
    for _, _, name, args_json in accepted:
        tool_calls.append({
            "id": f"call_{uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args_json,
            },
        })

    # Strip the accepted spans from content (in reverse to keep indices valid).
    new_segment = segment
    for start, end, _, _ in sorted(accepted, key=lambda t: t[0], reverse=True):
        new_segment = new_segment[:start] + new_segment[end:]
    new_content = content[:segment_start] + new_segment
    # Squash double-blank-line gaps that the strip can introduce.
    new_content = re.sub(r"\n{3,}", "\n\n", new_content).strip()

    msg["content"] = new_content
    msg["tool_calls"] = tool_calls
    choice["finish_reason"] = "tool_calls"
    log.debug(
        "tool_call_recovery: applied — extracted=%d names=%s",
        len(tool_calls), [tc["function"]["name"] for tc in tool_calls],
    )
