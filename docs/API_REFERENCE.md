# API reference

Turbohaul-Manager listens on port `11401` by default and is compatible with both Ollama-shape and OpenAI-shape clients. Every endpoint below is served on that one port. No authentication is performed; the bind address is the security boundary (see [Security model & hardening](../ARCHITECTURE.md)).

This is the one complete list of routes. How the API layer is built (the app factory, the middleware, the chat-completion compatibility layers, typed errors and streaming) is in [ARCHITECTURE.md](../ARCHITECTURE.md), section 7.

To connect an AI agent, see [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md). For the web interface served at `/ui`, see [frontend/README.md](frontend/README.md). To add and configure a model, see [MODELS_AND_MANIFESTS.md](MODELS_AND_MANIFESTS.md).

**Inference**
- `POST /v1/chat/completions` -- OpenAI-shape inference, non-stream and SSE (supports `response_format` `json_object` and `json_schema`)
- `POST /api/chat` -- Ollama-shape inference; never streams; `stream: true` combined with tools returns 400
- `POST /v1/embeddings` -- llama-server embeddings passthrough; a `Content-Length` gate against `http.max_body_bytes`; at most 64 inputs per batch; a manifest capability pre-flight

**Model discovery**
- `GET /api/tags` -- list models, Ollama shape, from the manifests
- `GET /api/show?name=<tag>` -- model detail, Ollama shape
- `GET /v1/models` -- list models, OpenAI shape (also served as `/models`)
- `GET /v1/models/{model}` -- model detail, OpenAI shape (also served as `/models/{model}`)

All of these omit manifests marked `hidden`; a hidden model still serves normally by exact tag.

**Model management**
- `GET /api/manifests` -- list every manifest, including hidden entries; an unreadable file appears as a placeholder row with `error: "unreadable"`
- `GET` / `PUT` / `PATCH` / `DELETE /api/manifests/{tag}` -- read, replace, narrow-update (`hidden` only), or remove a manifest.
  - `GET` returns the ETag revision. `PUT` and `PATCH` use ETag/`If-Match` optimistic concurrency: a mismatched or missing `If-Match` on an update returns 412, and `PUT` writes atomically. `DELETE` removes the file (404 if absent) and takes no `If-Match`.
  - The GGUF digest is validated for shape only -- a manifest naming a blob that is not yet in the store is accepted, and the missing blob surfaces at spawn time.
- `POST /api/manifests/{tag}/restore-defaults` -- drop this model's per-model flag overrides so it follows the built-in defaults again (`MANIFEST_FLAG_DEFAULTS`); operator-tuned fields such as `ctx_size` stay. Uses ETag/`If-Match`.
- `POST /api/pull-hf` -- pull a GGUF from HuggingFace (SSRF-guarded download into the blob store)
- `POST /api/pull-url` -- pull a GGUF from an arbitrary HTTPS URL (SSRF-guarded download into the blob store)
- `POST /api/pull` -- the Ollama registry protocol; present but returns 501 (a stub). Use `/api/pull-hf` or `/api/pull-url` instead.
- `POST /api/import` -- import a local GGUF file (sandboxed to the configured import root)
- `GET /api/blobs` -- list the blob store
- `PUT /api/blobs/{digest}/description` -- annotate a stored blob (an operator description write)
- `DELETE /api/delete` -- delete a blob by sha256 digest; refuses with 409 while any manifest still names it

**Operations**
- `GET /health` -- liveness plus version
- `GET /status` -- live queue + active + idle_hot snapshot (the full `status_snapshot()`)
- `GET /api/version` -- version and backend identification
- `GET /api/config` -- boot values read-only (paths reduced to basenames, plugin registry reduced to its keys), every runtime section, `_provenance` and `_provenance_stamp`
- `PUT /api/config` -- update any runtime section, persisted to `runtime_config.yaml` beside the state database; boot sections return 403, unknown ones 400
- `GET /api/config/schema` -- type, default, minimum and maximum for each editable config field (the bounds for every PUT-able runtime field)
- `GET /v1/logging` -- paginated audit events with server-side redaction; a per-page budget of about 80,000 characters, a single oversized event returned alone with `oversized: true`; `limit` clamped to 1-500
- `GET /v1/telemetry/events` / `GET /v1/telemetry/status` -- telemetry events and telemetry subsystem status (the flap-telemetry reader)
- `GET /api/fastlane/census` -- addresses observed calling the service (the Fast Lane Discovered list); rule assignment and ambiguity are derived at read time
- `WS /ws/state` -- redacted state-event stream: slot and queue transitions plus pull, import, manifest and blob change notices; it carries no prompt or response text
- `GET /ui/live/output/stream` -- server-sent stream of the current generation's output text, used by the web UI
- `GET /ui`, `/ui/`, `/ui/{path}` -- the bundled web UI (single-page app static serving with CSP headers and a path-traversal guard; served when `ui.enabled` is true, the default, and the UI bundle is present)

**Plugins** (work in progress)
- `GET /api/plugins` -- list registered plugin endpoints (resource-plugin discovery); never returns host or port
- `POST /api/plugins/{model_tag}/invoke` -- invoke a plugin; how an agent reaches one, addressed by tag and never by URL
- `WS /api/plugins/{model_tag}/exec` -- streaming plugin execution (a stateless exec-frame forwarder to the plugin container)

**Not implemented**
- The Ollama routes `/api/generate`, `/api/ps`, `/api/create`, `/api/copy` and `/api/push`, and the OpenAI route `/v1/completions`, are not implemented.
- `POST /api/pull` returns 501 (see Model management).
- FastAPI's default `/openapi.json`, `/docs` and `/redoc` routes stay enabled.

**Limits at a glance**
- Request bodies: `BodySizeLimitMiddleware` returns 413 before routing on `/v1/chat/completions` and `/api/chat`, and the embeddings route has the equivalent gate; both read `http.max_body_bytes` (default 64 MiB) and compare the declared `Content-Length`.
- `/v1/embeddings`: at most 64 inputs per batch.
- `/v1/logging`: `limit` clamped to 1-500 and a page budget of about 80,000 characters.
- Manifest writes: 412 on a mismatched or missing `If-Match`. Blob delete: 409 while a manifest still names the digest.
