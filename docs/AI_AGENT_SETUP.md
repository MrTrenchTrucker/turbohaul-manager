# Turbohaul-Manager — AI Agent Setup Guide

**Audience:** Developers connecting an AI agent (Hermes, LiteLLM-routed, langchain, llama-index, raw OpenAI SDK, Ollama-client, etc.) to a Turbohaul-Manager instance.

**Goal:** Hermes-class **multi-tool-call** agent loops work out of the box, slot-state survives many tool-call turns without 600s timeouts, and you don't have to read Turbohaul's source to wire it up.

---

## TL;DR — The Two-Line Setup

Most agents need this and only this:

```yaml
base_url: http://<turbohaul-host>:11401/v1
api_key: dummy   # any string — Turbohaul doesn't require auth on its API port
```

If your agent is OpenAI-API-shaped (Hermes, OpenAI Python SDK, langchain `ChatOpenAI`, llama-index `OpenAI`, etc.), point it at `:11401/v1`. If it's Ollama-shaped, point it at `:11401` with no `/v1` suffix — both surfaces are served by the same process on the same port, and share the same slot lifecycle.

That's the entire setup. Sane defaults are baked in: idle-hot keeps slots warm 10 min between turns, ACTIVE_MATCH reuses the warm process for same-`thread_id` follow-ups, tool-call fields pass through, and streaming SSE is supported end-to-end.

The rest of this doc is for when "just works" doesn't, or when you want to tune.

---

## Recommended: turn off automatic conversation-title generation

Many agent harnesses generate a short conversation title automatically, as a separate one-shot
request. That request competes for the same slot as the conversation itself — on a long
conversation with a large warm cache, turning the title generator loose means the real cache gets
saved out and restored around a call whose only output is a handful of words for a UI label. The
work is real; the value is cosmetic.

Turn it off for any agent holding long conversations. Leave it on only where the title is
genuinely needed and conversations stay short.

---

## What Turbohaul gives your agent — for free

You don't have to set any of these; they're the default. Listed so you can recognize the behavior you'll see in logs:

| Behavior | Default | What it does for your agent |
|---|---|---|
| `idle_hot_load_seconds` | `600` (10 min) | After a request completes, the slot stays warm. The next request to the same model within 10 min skips cold-load (saves the cold-load time, which can be tens of seconds on a 27B GGUF). |
| `grace_seconds` | `30` | After a request completes, the slot holds for another 30s before transitioning to IDLE_HOT. Within this window, **ACTIVE_MATCH cascade** warm-reuses the slot for same-thread follow-ups (fast handoff, no re-spawn). |
| `keep_alive` | client-overridable | If your agent sends Ollama-style `keep_alive: "10m"` or `1800` (seconds), Turbohaul honors it as the IDLE_HOT extension (capped at 30 min). `keep_alive: 0` = unload immediately. `keep_alive: -1` = pin (max cap). Single line in `extra_body` for OpenAI-SDK clients (see §1 and §2 below). |
| Streaming SSE pass-through | always on | When you send `stream: true`, Turbohaul opens its own `httpx.stream()` to llama-server and pipes raw SSE chunks back. The 12-second keep-alive heartbeat keeps your client socket warm during cold-load. |
| Tool-call fields | pass through | `tools`, `tool_choice`, `parallel_tool_calls`, `function_call`, `functions` forwarded verbatim to llama-server on BOTH `/v1/chat/completions` and `/api/chat`. On `/api/chat`, a request that combines these fields with `stream: true` is rejected with HTTP 400 (`streaming_with_tools_deferred`): use `stream: false` there, or `/v1/chat/completions` for streaming tool calls. Works on any model whose manifest sets `jinja: true`. |
| Tool-call recovery | transparent post-processor | When a jinja-templated model (notably Qwen3-family per upstream llama.cpp issues #20809 / #20837 / #20260) emits a tool call as text JSON inside `message.content` instead of populating `message.tool_calls`, Turbohaul extracts and restores it into the structured field, flips `finish_reason` to `tool_calls`, and strips the matched JSON from `content`. Idempotent — no-op when upstream already populates correctly. Applies to non-streaming responses only; a stream is relayed as the engine emits it. See [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md). |
| Thinking models | reasoning preserved | Qwen3.6, DeepSeek-R1, and similar thinking models keep their structured `reasoning_content`. Non-streaming responses also carry it inline as `<think>...</think>` in `content` (skipped when `response_format` is `json_object` or `json_schema`); streaming responses are relayed as the engine emits them, so read the `reasoning_content` deltas. |
| Per-thread warm reuse | via `thread_id` | If you include a `thread_id` field in the request body, same-thread follow-ups within the grace window hit ACTIVE_MATCH (warm slot, no re-spawn). Many OpenAI-SDK clients can pass arbitrary fields via `extra_body`. |
| Safety guardrails | on by default (`queue.safety_enabled`) | Pre-spawn VRAM/RAM/CPU/IO-wait checks refuse to spawn into an OOM or IO-stuck host (the request fails with HTTP 500 and a `sidecar failed: safety gates refused spawn: ...` detail naming each failed gate, rather than crashing the box). Capacity over-commit (target GPU/card already full, nothing idle-evictable in time) returns the same HTTP 503 + `Retry-After` shape for the same reason — always retryable, never a crash. |

If your agent gets a 503 with `Retry-After`, that's a capacity/over-commit condition or a sidecar that crashed or disconnected — back off and retry; nothing's broken. A host-safety refusal arrives as the HTTP 500 described above and clears once host pressure eases.

---

## Per-Agent Setup

### 1. Hermes Agent

Hermes is an agent harness that talks to any OpenAI-compatible endpoint, so it works with Turbohaul unchanged.

`hermes-config/config.yaml`:

```yaml
model:
  default: my-model-27b           # any model whose manifest you've loaded into Turbohaul
  max_tokens: 8192                     # >= 2000 recommended for thinking models so chain-of-thought has room
  provider: custom
  base_url: http://<turbohaul-host>:11401/v1
  api_key: dummy

providers:
  custom:
    request_timeout_seconds: 7200      # Hermes-side socket timeout; 2h is generous
    stale_timeout_seconds: 7200
    base_url: http://<turbohaul-host>:11401/v1
    api_key: dummy
    api_mode: openai                   # speaks /v1/chat/completions
    streaming: false                   # informational; Hermes' OpenAI SDK chat path always sends stream=true anyway

agent:
  max_turns: 40                        # up to 40 tool-call turns per agent run
  gateway_timeout: 7200
  api_max_retries: 240                 # 240 * 30s = 2h ceiling
  reasoning_effort: low                # IS a Turbohaul knob -- sets the model's
                                        # --reasoning-budget at spawn, locked for that engine's life.
                                        # See docs/REASONING_EFFORT_LADDER.md.
```

**Important nuance about `streaming: false`**: Hermes' OpenAI Python SDK chat-completions code path **always emits `stream: true` on the wire** (the `streaming: false` config flag is for Hermes' display layer, not the request body). Turbohaul handles this: streaming requests are relayed chunk-by-chunk (SSE pass-through), tool-call chunks included.

If you also want to send Ollama-only fields (like `keep_alive`), use `auxiliary.<task>.extra_body`:

```yaml
auxiliary:
  vision:
    extra_body:
      keep_alive: "10m"
```

The `extra_body` map is merged into the request JSON Hermes sends. Per [Ollama Issue #11458](https://github.com/ollama/ollama/issues/11458), this is the usual way for OpenAI-SDK clients to express `keep_alive` — the SDK doesn't natively support it.

### 2. OpenAI Python SDK (raw)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<turbohaul-host>:11401/v1",
    api_key="dummy",
)

resp = client.chat.completions.create(
    model="my-model-27b",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,               # recommended: SSE keep-alive comments hold the connection open during cold-load
    max_tokens=2048,
    extra_body={
        "thread_id": "session-abc-123",   # opt-in: enables ACTIVE_MATCH warm reuse
        "keep_alive": "5m",               # Ollama-compat; OpenAI SDK doesn't have this natively
    },
)
for chunk in resp:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

If you'll do multi-turn agent loops on the same conversation, **always pass `thread_id`**. Without one, Turbohaul falls back to an identity it derives itself (see [TAGS_AND_IDENTITY.md §4](TAGS_AND_IDENTITY.md#4-no-tags-the-guess-ladder)), which can split one conversation or merge two, so the warm-slot match and KV prefix reuse may not fire. With an explicit `thread_id`, turns N>1 hit the warm slot.

### 3. langchain `ChatOpenAI`

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://<turbohaul-host>:11401/v1",
    api_key="dummy",
    model="my-model-27b",
    streaming=True,
    max_tokens=2048,
    model_kwargs={
        "extra_body": {
            "thread_id": "session-abc-123",
            "keep_alive": "5m",
        }
    },
)
```

For tool-calling agents (langgraph, AgentExecutor), bind tools as usual:

```python
llm_with_tools = llm.bind_tools([my_tool_1, my_tool_2])
```

langchain will pass `tools` and `tool_choice` in the request body. Turbohaul forwards them to llama-server, which produces structured `tool_calls` chunks in the SSE stream. langchain's OpenAI tool-call parser handles those automatically.

### 4. llama-index `OpenAI`

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base="http://<turbohaul-host>:11401/v1",
    api_key="dummy",
    model="my-model-27b",
    max_tokens=2048,
    additional_kwargs={
        "extra_body": {
            "thread_id": "rag-session-7",
            "stream": True,
        }
    },
)
```

### 5. LiteLLM (router / proxy)

If you have LiteLLM in front of multiple providers, register Turbohaul as a custom OpenAI-compat provider:

```yaml
# litellm_config.yaml
model_list:
  - model_name: my-model-27b-turbohaul
    litellm_params:
      model: openai/my-model-27b
      api_base: http://<turbohaul-host>:11401/v1
      api_key: dummy
      stream: true
      extra_body:
        keep_alive: "10m"

  - model_name: my-moe-model-26b-turbohaul
    litellm_params:
      model: openai/my-moe-model-26b
      api_base: http://<turbohaul-host>:11401/v1
      api_key: dummy
      stream: true
```

LiteLLM handles fallback / load-balance / retry logic. Turbohaul handles slot lifecycle.

### 6. Ollama-shape clients (Ollama Python, Ollama JS, OpenWebUI, etc.)

Ollama-shape clients talk to the **same port**, `11401`, without the `/v1` suffix:

```python
import ollama

client = ollama.Client(host="http://<turbohaul-host>:11401")

resp = client.chat(
    model="my-model-27b",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    keep_alive="5m",
    options={"num_ctx": 8192},
)
```

The Ollama-shape surface covers `/api/chat`, `/api/tags`, `GET /api/show?name=<tag>`, `/api/pull-hf` and `/api/pull-url`, alongside the manifest and import routes (`/api/pull`, the Ollama registry protocol, is present but returns 501). That is enough for an Ollama chat client that lists models and chats. `/api/generate` (the raw, non-chat completion endpoint) is **not** implemented; use `/api/chat` or the OpenAI-shape `/v1/chat/completions`. Internally Turbohaul converts to its single slot lifecycle.

### 7. Generic HTTP clients (curl, requests, fetch)

```bash
curl -sN http://<turbohaul-host>:11401/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 256,
    "thread_id": "curl-test-1",
    "keep_alive": "1m"
  }'
```

`-N` is important — disables curl's default line-buffering so you see SSE chunks as they arrive.

---

## Multi-Tool-Call Agent Loops — What to know

This is the workflow Turbohaul is built for.

1. **Each turn is a fresh HTTP request.** OpenAI Chat Completions is stateless — your agent re-sends the full message history each turn (including assistant tool-call messages and tool-result messages).
2. **Same-conversation continuity comes from `thread_id`.** If you include the same `thread_id` on every turn, Turbohaul's ACTIVE_MATCH cascade keeps the slot warm for a fast handoff.
3. **No special handling for thinking models.** Qwen3.6 and friends stream their reasoning as structured `delta.reasoning_content` (non-streaming responses also carry it inline as `<think>...</think>` in `content`). Your agent can either:
   - Display reasoning separately (read `reasoning_content`)
   - Strip thinking blocks (regex `<think>.*?</think>` on accumulated content)
   - Use raw content as-is
4. **Tool calls arrive as structured chunks.** When the model decides to call a tool, you'll get SSE chunks with `delta.tool_calls = [{index, id, function: {name, arguments}}]`. Arguments may stream incrementally — accumulate before parsing.
5. **`finish_reason: "tool_calls"` ends the turn.** Execute the tool, append `{"role": "tool", "tool_call_id": "...", "content": "..."}` to your messages, then POST another request. Don't forget the `thread_id`.

### Common pitfall — `max_tokens` too small

Thinking models exhaust `max_tokens` inside `<think>` if `max_tokens` is too small (default 512 in some clients is way too small). **Set `max_tokens` to at least 2000** for any thinking-model agent loop. The Hermes example above uses 8192 for this reason.

### Common pitfall — Missing `thread_id`

If multi-turn loops feel slow (every turn re-prefills the whole conversation or cold-loads), check that you're passing the same `thread_id` on each turn. Without one, Turbohaul falls back to a derived identity (with one resident sidecar: client IP plus a fingerprint of the first message), which may not stay stable across your turns, so ACTIVE_MATCH warm reuse can miss and a turn can pay full re-prefill or cold-load cost.

### Common pitfall — Qwen3 text-JSON tool calls

Qwen3-family GGUFs on llama.cpp jinja templates sometimes emit a tool call as text JSON inside `message.content` (or wrapped in `<tool_call>...</tool_call>`) instead of populating `message.tool_calls`. Without intervention, OpenAI-shape clients that read only the structured field see "no tool call" and the loop stalls. On non-streaming requests Turbohaul recovers these automatically (post-processor runs AFTER `_merge_reasoning_into_content`, BEFORE returning to the client); your agent should see normal structured `tool_calls` and `finish_reason: "tool_calls"`. Streaming responses are not post-processed: a model in this class will deliver the JSON as `delta.content` on a stream, so send such requests with `stream: false` or handle the text-JSON shape client-side. If you want the diagnostic detail (which candidates matched, which were rejected by the allowlist, etc.), set the `turbohaul.api.tool_call_recovery` logger to DEBUG. Full mechanism: [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md).

---

---

## Multi-Role / Aux-Model Setups — one frontier main + local aux roles

Everything above is a single agent talking to one model. Real agent stacks are usually **several roles at once**: a primary agent that does the work, throwaway sub-agents it spawns, a background curator/reviewer, and a context-compression pass. Turbohaul is built to serve all of these behind one endpoint while keeping each role's KV cache in its own bin — so roles never bleed into each other and the ones that *should* share a warm cache actually do.

This section is about wiring a harness so each role's requests carry the right identity. For **why and when** you'd split roles across models (frontier main + local aux, cost/latency trade-offs, GPU-sharing economics), see [DEPLOYMENT_PATTERNS.md](DEPLOYMENT_PATTERNS.md) and the walkthrough in [MULTI_AGENT_SHARING.md](MULTI_AGENT_SHARING.md).

### The four roles

Each role can be a **different model** — and they don't all have to be served by Turbohaul. A common split is a frontier model driving the main agent over any OpenAI-compatible route, while Turbohaul serves the local aux roles on your own GPU:

| Role | Typical model | Served by | Why it's split out |
|---|---|---|---|
| **main** | frontier or your best local model | any OpenAI-compatible endpoint (incl. Turbohaul) | the primary agent loop — its context is the product, so its KV must never silently recompute |
| **sub-agent** | small/fast local model | Turbohaul | disposable workers spawned per task; isolated so they can't muddy main |
| **curator** | small local model | Turbohaul | background reviewer of main's work; wants main's *warm* cache but must never overwrite it |
| **compression** | small local model | Turbohaul | summarizes/compacts main's context; marks main's saved KV stale so the next main turn re-anchors |

The main agent can point at a frontier API and *only* the aux roles at Turbohaul — the identity contract below is per-request, so a mixed stack is just "some requests go to endpoint A, some to endpoint B." When main is *also* on Turbohaul, the same contract gives it stable-thread KV reuse across turns.

### The per-role `thread_id` scheme

Turbohaul keys one KV copy per `(session_id, role)` (see [TAGS_AND_IDENTITY.md §2](TAGS_AND_IDENTITY.md#2-what-each-role-gets)). The lever you pull to get the *behavior* you want per role is the **`thread_id`** you stamp on each request:

| Role | `thread_id` scheme | Effect |
|---|---|---|
| **main** | **stable across turns** — e.g. `main-<session_id>`, reused every turn of the conversation | turn N>1 continues main's thread → warm KV reuse, no cold reprefill |
| **sub-agent** | **distinct per spawned agent** — e.g. `sub-<session_id>` with a fresh child `session_id` per spawn | each sub-agent gets its own bin → isolation, no cross-restore between siblings or into main |
| **curator** | **same as main's** — reuse main's `thread_id` | curator rides main's warm prefix to review it, without paying to reprefill (its own KV is not saved by default, so it cannot overwrite main's saved copy) |
| **compression** | **main's thread**, smaller context | compression pass re-serves main's prefix at a reduced context length; label it `is_compression` so main's saved bin is marked stale |

The rule of thumb: **stable thread = reuse, distinct thread = isolation, shared thread = ride someone's warm cache.** Give every spawned sub-agent its own child `session_id` (a workable pattern is `{parent_session_id}-sub-<nonce>`) so concurrent siblings land in distinct bins — one sibling's context is never restored into another.

### Registering Turbohaul as an inference provider

The clean way to do this in a harness is a small **provider plugin** that, per request, derives the role from whatever your harness already knows (is this a spawned worker? a compression pass? a curator fork?) and stamps the request body with the right `thread_id` + role labels. Everything it emits lands in the request JSON exactly where Turbohaul reads identity — top-level `thread_id`, nested `client_meta` for the role flags (see [TAGS_AND_IDENTITY.md §1.1](TAGS_AND_IDENTITY.md#11-where-exactly-turbohaul-looks-in-your-request) for the exact JSON placement, precedence, and OpenAI-SDK `extra_body` examples).

A genericized provider shape — one function that maps `(session_id, role-context)` → request-body extras:

```python
def build_request_extras(session_id, *, is_sub_agent=False, is_curator=False,
                         is_compression=False, parent_session_id=None):
    """Return dict merged into the request body (SDKs: pass via extra_body).

    Roles resolve by priority: curator > compression > sub_agent > main.
    """
    # 1. Resolve the role (single source of truth) ────────────────────
    if is_curator:
        role = "curator"
    elif is_compression:
        role = "compression"
    elif is_sub_agent or parent_session_id:
        role = "sub_agent"
    else:
        role = "main"

    # 2. Give each spawned sub-agent its OWN session so siblings isolate
    sid = session_id
    if role == "sub_agent" and (sid is None or parent_session_id):
        sid = f"{parent_session_id or session_id}-sub-{short_nonce()}"

    # 3. Derive thread_id from role → this is what drives KV behavior ──
    #    main/curator/compression share the MAIN thread (curator+compression
    #    ride main's warm cache); sub-agents get a DISTINCT thread (isolation).
    if role == "sub_agent":
        thread_id = f"sub-{sid}"          # distinct  → isolated bin
    else:
        thread_id = f"main-{sid}"         # stable    → main's warm cache

    # 4. Emit identity. thread_id top-level; role flags in client_meta.
    #    Always emit exactly one true flag — even if session_id was missing —
    #    so an untagged-looking turn never arrives "all-None".
    return {
        "thread_id": thread_id,
        "client_meta": {
            "session_id": sid,
            "is_main":        role == "main",
            "is_sub_agent":   role == "sub_agent",
            "is_curator":     role == "curator",
            "is_compression": role == "compression",
            # "save_kv": True,   # opt-in: persist a disposable role's KV (§3)
        },
    }
```

The corresponding provider registration is just the usual OpenAI-compatible pointing-at-Turbohaul, plus this per-request body hook:

```yaml
provider: turbohaul
base_url: http://<turbohaul-host>:11401/v1
api_key: dummy
# per-request: harness calls build_request_extras(...) and merges the result
# into the request body (via extra_body for OpenAI-SDK transports).
```

Notes that keep this correct in practice:

- **Curator = main's `thread_id` on purpose.** The curator shares main's thread so it can ride main's warm prefix. By default its own KV is isolated and not saved, so it can never overwrite main's saved copy (see [TAGS_AND_IDENTITY.md §2](TAGS_AND_IDENTITY.md#2-what-each-role-gets)). There's also an opt-in manager-side route (`TURBOHAUL_CURATOR_REUSE_MAIN`, off by default) that lets a labeled curator restore main's saved state read-only; you don't need it to share the warm prefix.
- **Always emit exactly one role flag**, even when `session_id` is momentarily absent. A body with no `session_id` and no flag arrives as an untagged walk-in and drops out of the role contract — the classic "sub-agent turns show up all-None" gap. Synthesize a child `session_id` for a sub-agent rather than emitting an empty body.
- **`save_kv` is a per-request override**, not an env var — flip it per role at runtime when you have the VRAM/RAM to keep a normally-disposable role's KV. Absent → sub-agent/curator/compression KV is thrown away; main always saves. Details in [TAGS_AND_IDENTITY.md §3](TAGS_AND_IDENTITY.md#3-the-save_kv-override).
- **Verify it landed:** grep the manager log for `R2B_REQ_IDENTITY` right after a test call — `resolved_class` should show the role you intended and `session_id` should be non-null ([TAGS_AND_IDENTITY.md §5](TAGS_AND_IDENTITY.md#5-verifying-your-tags-land)).

### If your harness doesn't have a Turbohaul provider yet

Turbohaul-as-a-first-class-provider isn't merged into every upstream harness. **You are not blocked waiting on that.** The provider plugin is only a convenience wrapper — the actual contract is "put `thread_id` + `client_meta` in the request body," and *any* OpenAI-API client can do that today:

- **OpenAI Python/JS SDK** — pass them through `extra_body` (the SDK flattens `extra_body` keys to the top level, so top-level `thread_id` lands where Turbohaul reads it; `client_meta` rides as a nested object).
- **langchain / llama-index** — the same dict via `model_kwargs={"extra_body": {...}}` / `additional_kwargs={"extra_body": {...}}`.
- **LiteLLM** — `extra_body` on the model entry (per-role model aliases each with their own `thread_id`/flags).
- **Raw curl / requests / fetch** — just put the keys in the JSON body directly; no `extra_body` indirection needed.

Copy-paste JSON and SDK snippets for every one of these are in [TAGS_AND_IDENTITY.md §1.1](TAGS_AND_IDENTITY.md#11-where-exactly-turbohaul-looks-in-your-request). In other words: a native provider plugin makes role-tagging automatic, but role-tagging itself is just three body fields — you can wire the full multi-role contract by hand with the client you already have.

## Production Setup Notes

### Docker run with sane defaults (CUDA / NVIDIA host)

```bash
docker run -d --name turbohaul \
  --gpus all \
  -p 11401:11401 \
  -v $(pwd)/state:/var/lib/turbohaul \
  -v $(pwd)/models:/var/lib/turbohaul/import-staging \
  -e TURBOHAUL_IDLE_HOT_S=600 \
  -e TURBOHAUL_GRACE_S=30 \
  ghcr.io/mrtrenchtrucker/turbohaul-manager:v0.8.0
```

Defaults are reasonable; you only need env-overrides if you want different timing.

### Manifest setup (per model)

Each model needs a manifest at `/var/lib/turbohaul/manifests/<model-tag>.yaml`. Example for a 27B model:

```yaml
model_tag: my-model-27b
gguf_blob_sha256: <64-hex sha256 of the model GGUF in the blob store>
context_size: 92160              # declared model context; metadata + fallback for the KV-fit gate
expected_vram_bytes: 22500000000 # real device budget for the VRAM-fit gate

llama_server_flags:
  ctx_size: 92160
  n_gpu_layers: 999
  cache_type_k: q4_0
  cache_type_v: q4_0
  n_predict: -1
  reasoning: auto
  reasoning_budget: 500          # caps thinking depth — tune 200-2000 for tool-loop speed
  jinja: true                    # keep on: tool_calls + Qwen3 thinking-block preservation use the model's Jinja chat template
```

The **`jinja: true`** flag matters for two things:
- `tool_calls` only work when llama-server uses the Jinja chat-template branch
- Qwen3-class thinking models only preserve `<think>` blocks in the response under `--jinja`

The vendored llama-server enables Jinja by default, so this flag makes the choice explicit (`true` becomes `--jinja`; `false` is simply omitted from the command line). If you copy a manifest, keep this flag.

There is no `gguf_path`, `quant`, `context_length` or top-level `gpu_layers` key. The manifest model rejects unknown top-level keys outright, so a manifest carrying any of them fails to load. The model blob is content-addressed by hash, never named by path — path-bearing launch flags such as `mmproj` and `lora` are on the spawn-flag denylist for the same reason. The weight quant is read from the GGUF header, not declared. Layer placement is `llama_server_flags.n_gpu_layers` (`999` = all layers on GPU), and `ctx_size` under `llama_server_flags` is what actually becomes `--ctx-size` — keep it in sync with the top-level `context_size`. Leaving `expected_vram_bytes` at its `0` default means the model declares no budget of its own, so the free-VRAM check falls back to the baseline free-VRAM floor (the separate `ctx_size`-based KV-fit gate still applies).

#### Hybrid (SSM + attention) models — two optional manifest fields

Most models need nothing beyond the fields above. Two **optional** top-level manifest fields exist for models whose architecture differs from a pure-attention transformer — for example a `qwen35` hybrid that interleaves state-space (SSM) layers with attention layers:

```yaml
model_tag: my-hybrid-27b
gguf_blob_sha256: <64-hex sha256 of the model GGUF in the blob store>
context_size: 92160
arch: qwen35              # architecture hint; default "" (not set)
hybrid_kv_ratio: 0.5      # fraction of layers with a GROWING KV cache; default 1.0

llama_server_flags:
  ctx_size: 92160
  n_gpu_layers: 999
  cache_type_k: q4_0
  cache_type_v: q4_0
  n_predict: -1
  jinja: true
```

- **`arch`** (string, default `""`): an architecture hint. Any non-empty value (for example `qwen35` for an SSM/attention hybrid) opts the model into the dimension-aware KV estimate, which is computed from the attention dimensions read out of the GGUF. Left empty, the model keeps the standard estimate — byte-identical to every model configured before this field existed — unless its GGUF shows sliding-window attention layers, which select the dimension-aware estimate on their own.
- **`hybrid_kv_ratio`** (float 0.0–1.0, default `1.0`): the fraction of layers that contribute a **growing** per-token KV cache. In a hybrid, the SSM layers keep a fixed-size recurrent state instead of a cache that grows with context, so the model's per-token KV footprint is smaller than a pure-attention model of the same size. Turbohaul scales its file-size-based VRAM / KV-fit estimate by this ratio, so a hybrid packs more context (or shares a card more comfortably) than the raw parameter count would suggest. (The dimension-aware estimate and an operator-measured `kv_bytes_per_token` already count only the growing attention layers, so they do not apply the ratio a second time.) `1.0` = pure attention = the original estimate, unchanged for every existing model.

**Weight quant is auto-detected — no new flag needed.** The engine reads the weight quant straight from the GGUF header (`general.file_type`), so a model's quant never appears in a manifest and there is nothing to add to `llama_server_flags`. Every weight quant the vendored engine registers (for example the standard K-quants or `TQ4_1S`) loads through the manifest shape above, unchanged.


### Multi-model deployment

You can load multiple manifests; Turbohaul will swap models on demand (tears down the warm holder for one model when a request for a different model arrives). The default is single-slot (one model active at a time), preserving the single-sidecar invariant. Concurrent multi-model residency has shipped: raise `max_parallel_sidecars` (1–32, env `TURBOHAUL_MAX_PARALLEL`) and see ARCHITECTURE.md's "double parallel" mode.

### Health checks

- `GET http://<host>:11401/health` → `{"status":"ok","version":"<version>"}` (lightweight, no slot interaction)
- `GET http://<host>:11401/status` → full slot state, current model, queue depth, idle-hot timer
- `GET http://<host>:11401/api/config` → effective runtime config (for example `queue.idle_hot_load_seconds` and `queue.grace_seconds`)
- `GET http://<host>:11401/api/tags` → list of available models (Ollama-shape)
- `GET http://<host>:11401/v1/models` → list of available models (OpenAI-shape)

---

## Validation Smoke Tests (run after setup)

### Quick streaming smoke

```bash
curl -sN http://<host>:11401/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-model-27b","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":32}' \
  | head -50
```

Expect: if the model has to cold-load first, `: keep-alive` SSE comments every 12s while it loads, then a stream of `data: {"choices":[{"delta":{"content":"..."}}]}` chunks ending in `data: [DONE]`.

### Multi-turn smoke (proves ACTIVE_MATCH works)

```python
import time, requests, json

URL = "http://<host>:11401/v1/chat/completions"
TID = "smoke-thread-" + str(int(time.time()))

def turn(content, history):
    history = history + [{"role": "user", "content": content}]
    body = {
        "model": "my-model-27b",
        "messages": history,
        "stream": True,
        "max_tokens": 64,
        "thread_id": TID,
    }
    start = time.time()
    with requests.post(URL, json=body, stream=True, timeout=180) as r:
        chunks = sum(1 for line in r.iter_lines() if line)
        wall = time.time() - start
    print(f"  wall={wall:.1f}s chunks={chunks}")
    return history + [{"role": "assistant", "content": "<reply>"}]

hist = []
print("Turn 1 (cold load expected on first call; tens of seconds for a large model):")
hist = turn("What is 2+2?", hist)
print("Turn 2 (ACTIVE_MATCH — should be much faster):")
hist = turn("Add 5.", hist)
print("Turn 3 (ACTIVE_MATCH — should be much faster):")
hist = turn("Multiply by 3.", hist)
```

Turn 1: cold load (tens of seconds for a large model). Turns 2+3: much faster wall time (warm reuse). If turns 2/3 take as long as turn 1, your `thread_id` isn't being forwarded — check `extra_body`.

### Tool-call smoke

```bash
curl -sN http://<host>:11401/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model-27b",
    "messages": [{"role":"user","content":"What is the weather in Boston?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
      }
    }],
    "tool_choice": "auto",
    "stream": true,
    "max_tokens": 256
  }'
```

Expect: at least one chunk with `delta.tool_calls = [{...}]` containing `function.name: "get_weather"` and `function.arguments` accumulating JSON. `finish_reason: "tool_calls"` ends the stream. (Keep `jinja: true` in the model's manifest, as in the manifest example above.)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Every turn is slow (long re-prefill or cold-load) | `thread_id` not being forwarded (also check `keep_alive` and `idle_hot_load_seconds`: the model may be unloading between turns) | Use `extra_body: {"thread_id": "..."}` in OpenAI SDK; same in langchain `model_kwargs`. |
| "Slot did not reach ACTIVE within 7200.0s" | (Should not happen in current releases) | Upgrade to the latest release. This was a bug in the ACTIVE_MATCH streaming path of early v0.2.x builds. |
| HTTP 503 with `Retry-After` | Capacity over-commit (target GPU/card full, nothing idle-evictable in time), OR the sidecar itself crashed/disconnected mid-response | Wait and retry; the `Retry-After` header tells you how long. |
| HTTP 500 `sidecar failed: safety gates refused spawn: ...` | Pre-spawn safety guardrail refused the spawn (host RAM / VRAM / CPU / IO-wait over its threshold) | Wait for host pressure to ease and retry; the detail names each failed gate. |
| HTTP 502 with `upstream_status` | llama-server returned an error (context overflow, malformed payload) | Check `upstream_body` in the response; usually means your prompt exceeded the model's `ctx_size`. |
| HTTP 504 | llama-server timed out generating | Reduce `max_tokens` or check that the model isn't stuck in a thinking loop (lower `reasoning_budget`). |
| Tool calls never fire (model just describes the tool instead) | Jinja chat-template handling is not active for the model (manifest without `jinja: true`, or an engine build that doesn't default it on) | Add `jinja: true` to the model's manifest under `llama_server_flags`. Restart container. |
| Tool calls fire on Ollama `/api/chat` but not OpenAI `/v1/chat/completions` | Builds before v0.2.3 silently dropped `tools` on the OpenAI endpoint's `client_meta` | Upgrade to v0.2.3 or later. |
| Tool calls show as text JSON in `message.content`, `message.tool_calls` empty | Either a build before v0.2.3 (no recovery layer), OR the request did not advertise `tools` (recovery requires the allowlist), OR the request was streaming (recovery runs on non-streaming responses only) | Upgrade to v0.2.3 or later AND ensure your request body includes `tools: [...]` and `stream: false`. See [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md). |
| Streaming hangs at start, no chunks | Client doesn't accept SSE Content-Type | Set `Accept: text/event-stream` header, or use a real SSE library (sseclient-py, eventsource). |
| Hermes stuck "pondering..." | Inter-turn slot didn't promote via ACTIVE_MATCH | Make sure your model's manifest has `reasoning_budget` capped (recommend 500 for tool-loops). |
| `tools` field rejected with HTTP 400 | Old Turbohaul version (early v0.2.x builds) | Upgrade to the latest release. |

If the issue isn't here, check the container logs for the wrapper-side view and `/status` for the current slot state.

---

## What Turbohaul does NOT do (yet)

To save you from chasing things that aren't supported:

- **`/v1/completions` (FIM / raw completion).** Not implemented. The chat-completions surface (`/v1/chat/completions` and `/api/chat`) is the generation path; embeddings have their own route.
- **Cross-process queue sharing.** If you spin up multiple Turbohaul instances, each has its own queue. Use a single instance behind a load balancer if you need request-level isolation.
- **Authentication.** The API port has no auth (`dummy` api_key works), so keep it on a trusted network (the `docker-compose.yml` in this repo binds it to `127.0.0.1`). Exposing it beyond that requires bearer-auth + TLS termination in front of it (not built in).
- **GPU mid-flight cancel.** A client disconnect is detected (polled about every 2 s) and drops a request that is still waiting for its slot. A non-streaming generation already running on the engine is not aborted and completes before the slot frees; a streaming relay closes its upstream connection and leaves it to llama-server to stop.

Also supported, so not limitations: multi-model concurrent residency (set `max_parallel_sidecars`, 1–32, and see ARCHITECTURE.md's "double parallel" mode); `/v1/embeddings`; `response_format` with both `json_object` and `json_schema`; and vision models, via the manifest's content-addressed `mmproj_blob_sha256` projector field.

---

## Where to go next

- **Architecture deep-dive:** `ARCHITECTURE.md` in the repo root (queue / slot FSM / IDLE_HOT / ACTIVE_MATCH internals)
- **Tool-call recovery layer:** [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md) (wire shape, recovery post-processor, Qwen3 text-JSON case, testing)
- **GitHub:** `https://github.com/MrTrenchTrucker/turbohaul-manager`

---

*This doc is the contract between Turbohaul and the agents that use it. If you find a wire-shape detail that's not documented here, that's a doc bug — open an issue.*
