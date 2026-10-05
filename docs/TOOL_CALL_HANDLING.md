# Tool-Call Handling in Turbohaul-Manager

This document explains how Turbohaul forwards tool-call request fields to `llama-server` and how it recovers tool calls when the model emits them as text JSON inside `message.content` instead of the structured `message.tool_calls` field. Most agents do not need to read this — the default behavior is "just works." Read this if you are debugging a model that the docs claim should support tool calls but is not producing structured `tool_calls`.

---

## Two wire paths

Turbohaul exposes two API surfaces, **both on the same port** (default `11401`):

- `POST /v1/chat/completions` (OpenAI-shape)
- `POST /api/chat` (Ollama-shape)

Both accept and forward the same tool-call fields:

| Field | OpenAI | Ollama | Behavior |
|---|---|---|---|
| `tools` | yes | yes | Forwarded verbatim to `llama-server`. The `name` allowlist is also used by the recovery post-processor (below). |
| `tool_choice` | yes | yes | Forwarded. |
| `parallel_tool_calls` | yes | (custom) | Forwarded. |
| `function_call` (legacy) | yes | (custom) | Forwarded. |
| `functions` (legacy) | yes | (custom) | Forwarded. |

On `/api/chat`, a request with `"stream": true` and any of these fields is rejected with HTTP 400 (`streaming_with_tools_deferred`); use `"stream": false` there, or `/v1/chat/completions` for streaming tool calls.

The model manifest should set `jinja: true` (under `llama_server_flags`; it is emitted as `--jinja`) for any tool-call work; `llama-server` applies the chat-template `tools` placeholder only in jinja mode, and setting the flag keeps that explicit rather than depending on the engine's default.

---

## Path 1 — native (best case)

When `llama-server` parses the model's chat-template output correctly, the response is shaped like OpenAI canonical:

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_abc",
        "type": "function",
        "function": {"name": "list_directory", "arguments": "{\"path\":\"/app\"}"}
      }]
    }
  }]
}
```

Turbohaul forwards this to the client unchanged. No post-processing. This is what well-aligned chat templates produce.

---

## Path 2 — text-JSON recovery (defensive)

Some chat-templated GGUFs — notably the **Qwen3 family on llama.cpp jinja** (upstream issues [#20809](https://github.com/ggml-org/llama.cpp/issues/20809), [#20837](https://github.com/ggml-org/llama.cpp/issues/20837), [#20260](https://github.com/ggml-org/llama.cpp/issues/20260)) — produce the call as JSON text inside `message.content` and leave `message.tool_calls` empty. Clients that only read the structured field see "no tool call" and stop.

Turbohaul's `maybe_recover_tool_calls` post-processor (lives in [`src/turbohaul/api/tool_call_recovery.py`](../src/turbohaul/api/tool_call_recovery.py)) recovers these. It runs AFTER `_merge_reasoning_into_content` and BEFORE the route returns the result. The pipeline is:

```
llama-server response
        |
        v
_merge_reasoning_into_content  (Qwen3 think-block normalizer)
        |
        v
maybe_recover_tool_calls       (text-JSON recovery)
        |
        v
return to client
```

### What the post-processor catches

Two shapes:

**1. OpenAI canonical text-JSON:**
```
{"name": "list_directory", "arguments": {"path": "/app"}}
```

**2. Qwen XML wrapper:**
```
<tool_call>{"name": "search_web", "arguments": {"q": "hello"}}</tool_call>
```

Parallel calls (multiple JSONs in one response) are all captured via `re.finditer` plus a brace-balancer that handles nested argument objects correctly.

### What it does NOT catch (intentional)

- **Calls inside `<think>...</think>` blocks.** Reasoning text often contains hypothetical calls ("I would call list_directory if...") that are not real tool calls. The post-processor only scans content AFTER the last `</think>`.
- **Calls to functions not in the request's `tools` allowlist.** Defense-in-depth against the model hallucinating a tool name.
- **Malformed JSON arguments.** Rejected silently — the call falls through to the client unchanged.

### What happens when it fires

1. `tool_calls` is populated with one entry per recovered call (synthetic `id: "call_<8-hex>"`).
2. The matched JSON spans are stripped from `content` (reasoning blocks and other text outside the matches are preserved).
3. `finish_reason` is set to `tool_calls`.

### Idempotency

If `message.tool_calls` is already populated by upstream (Path 1 above OR a future llama.cpp fix), the post-processor is a no-op. Safe to leave enabled even after upstream issues #20809 / #20837 / #20260 are fixed.

### Scope — non-streaming responses only

The recovery post-processor runs on the **non-streaming** completion path, where there is a whole response object to rewrite. Streaming requests (`"stream": true` on `/v1/chat/completions`) are relayed chunk-by-chunk as they arrive from the engine and are **not** post-processed: a model that emits its tool call as text JSON will deliver that text in `delta.content` on a stream, and the client will see `finish_reason: "stop"`.

That matters because streaming is the common agent-loop shape. If you are running a model in the text-JSON class and depend on recovery, send that request non-streaming — or handle the text-JSON shape client-side. The probe below exercises the *forwarding* fix (that `tools` reaches the engine at all); a model on the native path produces `delta.tool_calls` in a stream on its own, without recovery being involved.

### Silent rejections

The post-processor returns silently (no error, no warning, no INFO log) when:

- The result is not a dict with at least one choice.
- Upstream already populated `tool_calls` (count >= 1).
- The request advertised no tools.
- The content (post-`</think>`) is empty.
- No text-JSON candidate matches the tools allowlist.

To see per-candidate diagnostics during debugging, set the `turbohaul.api.tool_call_recovery` logger to DEBUG:

```python
import logging
logging.getLogger("turbohaul.api.tool_call_recovery").setLevel(logging.DEBUG)
```

---

## Closure fix — `/v1/chat/completions` tools forwarding

An earlier version of the OpenAI endpoint had a separate bug: its `client_meta` dict (the contract between the route handler and the `_complete` closure) did NOT include `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions`. This had two effects:

1. The closure's `_COMMON_FORWARDED_KNOBS` loop silently dropped tools toward `llama-server`.
2. `maybe_recover_tool_calls(result, client_meta.get("tools"))` saw `None` and returned early ("no tools advertised").

The fix carries these five keys in the `client_meta` dict on the OpenAI endpoint, mirroring the `/api/chat` endpoint. In the current code both endpoints build `client_meta` from the shared `_COMMON_FORWARDED_KNOBS` tuple, which includes all five keys; a key is carried only when the request supplied a non-`None` value:

```python
forwarded_knobs = {
    k: payload.get(k) for k in _COMMON_FORWARDED_KNOBS
    if payload.get(k) is not None
}
client_meta = {
    "kind": "openai-chat-completion",
    # ...
    **forwarded_knobs,
    # ...
}
```

The following probe checks that `tools` reach the engine:

```bash
curl -sN http://localhost:11401/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model-27b",
    "messages": [{"role":"user","content":"What is the weather in Boston?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
      }
    }],
    "tool_choice": "auto",
    "stream": true,
    "max_tokens": 256
  }'
```

Expected (with a model on the native path): chunks with `delta.tool_calls = [{...}]`, ending in `finish_reason: "tool_calls"` with a populated `tool_calls` field.

---

## Testing

Tests live in [`tests/test_tool_call_recovery.py`](../tests/test_tool_call_recovery.py) (13 functions, 18 sub-cases counting parametrized variants):

| Test | Asserts |
|---|---|
| `test_canonical_text_json_extracted` | OpenAI canonical shape → structured `tool_calls`, content stripped, reasoning preserved |
| `test_qwen_xml_wrapper_extracted` | Qwen `<tool_call>...</tool_call>` → same; wrapper tags stripped |
| `test_parallel_tool_calls_all_extracted` | Two canonical JSONs in one response → both extracted, distinct synthetic IDs |
| `test_parallel_mixed_xml_and_canonical` | XML + canonical in same response → both extracted |
| `test_idempotency_skip_when_populated` | Pre-existing `tool_calls` → post-processor no-op |
| `test_unknown_name_rejected` | JSON `name` not in allowlist → no-op |
| `test_invalid_json_args_rejected` | Malformed `arguments` → no-op |
| `test_pre_think_close_not_extracted` | JSON inside `<think>...</think>` → no-op |
| `test_no_tools_advertised_noop` | `tools=None` / `[]` → no-op |
| `test_nested_object_arguments_parsed` | Nested arg objects → brace-balancer captures full object |
| `test_extract_known_names` | `_extract_known_names` helper edge cases |
| `test_iter_candidates_empty_on_no_match` | Plain text → empty candidate list |
| `test_malformed_result_noop` (parametrized x6) | Non-dict / empty choices / None / etc. → no raise |

Run locally:

```bash
pytest tests/test_tool_call_recovery.py -v
```

A companion test, `test_a_inbound_tools_forwarded_into_client_meta` in [`tests/test_ollama_tools.py`](../tests/test_ollama_tools.py), covers the same `client_meta` forwarding on the `/api/chat` endpoint (the pattern the OpenAI endpoint mirrors).

---

## Disabling

There is no kill-switch. The post-processor is idempotent and defensive: it only fires when upstream did NOT populate `tool_calls` AND the request advertised tools AND a matching candidate is present in the content. For a request that already works (structured `tool_calls` present, or no matching text-JSON call in the content) it is a no-op.

If you have a use case that needs a kill-switch (for example, a future llama.cpp upstream fix is in production and the recovery layer is no longer reachable), file an issue with the rationale.

---

## See also

- [README.md](../README.md) — quickstart + API surface
- [docs/AI_AGENT_SETUP.md](AI_AGENT_SETUP.md) — per-agent config recipes + multi-tool-call workflow
- [src/turbohaul/api/tool_call_recovery.py](../src/turbohaul/api/tool_call_recovery.py) — source (287 LoC)
- [tests/test_tool_call_recovery.py](../tests/test_tool_call_recovery.py) — tests (13 functions, 18 sub-cases)
- Upstream llama.cpp issues: [#20809](https://github.com/ggml-org/llama.cpp/issues/20809), [#20837](https://github.com/ggml-org/llama.cpp/issues/20837), [#20260](https://github.com/ggml-org/llama.cpp/issues/20260)
- [CHANGELOG.md](../CHANGELOG.md) — `v0.2.3` entry
