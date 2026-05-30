![Turbohaul-Manager — hauling inference traffic](docs/banner.png)

# Turbohaul-Manager

Multi-backend inference manager with support for [llama.cpp](https://github.com/ggerganov/llama.cpp) (NVIDIA GPU via CUDA) and [MLX](https://ml-explore.github.io/mlx/) (Apple Silicon native).

FIFO queue + grace + IDLE_HOT hot-hold + model swap. Single-slot serial sidecar per model.

**Platforms:** macOS (Apple Silicon, native MLX) · Linux (Docker + NVIDIA GPU, llama.cpp)

## What it does

- Accepts OpenAI / Ollama-shape `/v1/chat/completions` requests
- Single-slot serial sidecar (one inference child holds the model)
- **Multi-backend:** llama.cpp (CUDA, TurboQuant + MTP) on NVIDIA GPUs · MLX (`mlx_lm.server`) on Apple Silicon
- ACTIVE_MATCH cascade for same-thread follow-ups within a grace window (warm-process reuse)
- IDLE_HOT 5-minute warm-hold after grace expires: same-model follow-ups inherit the warm process; different-model swap tears down + spawns new
- Multiplexed multi-agent serialization on one shared GPU (proven with three concurrent agents — see [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md))
- Transparent tool-call recovery for jinja-templated GGUFs that emit calls as text JSON in `message.content` instead of the structured `tool_calls` field (notably Qwen3-family per upstream llama.cpp issues #20809 / #20837 / #20260) — see [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md)
- Safety guardrails: refuses spawn when VRAM / RAM / CPU / IO-wait would put the host at risk

## Quick start

### macOS (Apple Silicon, MLX backend)

Requires macOS 15.0+ (Sequoia) and Python 3.11+.

```bash
# Install
git clone https://github.com/MrTrenchTrucker/turbohaul-manager.git
cd turbohaul-manager
pip install -e ".[mlx]"

# Run
turbohaul-manager

# Register an MLX model (uses HuggingFace repo ID)
curl -X PUT http://localhost:11401/api/manifests/qwen3-1.7b \
  -H "Content-Type: application/yaml" \
  -d '
model_tag: qwen3-1.7b
backend: mlx
model_repo: mlx-community/Qwen3-1.7B-4B
mlx_server_flags:
  max_tokens: 8192
'

# Inference
curl http://localhost:11401/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"Hello"}]}'
```

MLX models are downloaded from HuggingFace on first use and cached locally (configurable via `mlx_models_dir` in `turbohaul.yaml`). Models use safetensors format — no GGUF blob import needed.

**Local models:** pass the full path as `model_path` instead of `model_repo`. If the model directory doesn't include a `chat_template` in its `tokenizer_config.json`, add `chat_template` to `mlx_server_flags` pointing at the `.jinja` file that ships with the model — otherwise `mlx_lm.server` will attempt a live HuggingFace lookup on every request:

```yaml
model_tag: vibethinker-1.5b
backend: mlx
model_path: /path/to/VibeThinker-1.5B-MLX
mlx_server_flags:
  chat_template: /path/to/VibeThinker-1.5B-MLX/chat_template.jinja
  max_tokens: 8192
```

Clients always use the short `model_tag` name (e.g., `vibethinker-1.5b`) — Turbohaul rewrites the `model` field in the upstream request to the path the sidecar was started with.

### Docker (NVIDIA GPU, llama.cpp backend)

```bash
# Run it (build locally first — see below; no prebuilt registry image is published yet)
docker run --gpus all -p 11401:11401 \
    -v $(pwd)/state:/var/lib/turbohaul \
    -v $(pwd)/models:/var/lib/turbohaul/import-staging \
    turbohaul-manager:v0.3.0

# Build locally (required)
git clone https://github.com/MrTrenchTrucker/turbohaul-manager.git
cd turbohaul-manager
docker build -f Dockerfile.cuda -t turbohaul-manager:v0.3.0 .
```

The `-v $(pwd)/state:/var/lib/turbohaul` mount is **required** for production deployment — without it, `state.sqlite`, `manifests/*.yaml`, and the `blobs/` store live inside the container layer and are destroyed by `docker rm` or container-layer corruption. See [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md) for the full hardening checklist.

## API

Compatible with Ollama-shape clients:
- `GET /api/tags` -- list models
- `GET /api/show?name=<tag>` -- model detail
- `POST /v1/chat/completions` -- OpenAI-shape inference (supports `response_format` json_object + json_schema)
- `POST /api/chat` -- Ollama-shape inference
- `POST /v1/embeddings` -- llama-server embeddings passthrough
- `GET /v1/logging` -- paginated audit events
- `PUT /api/manifests/{tag}` -- register a new model (requires GGUF blob in store; ETag/If-Match atomic concurrency)
- `POST /api/pull-hf` -- pull a GGUF from HuggingFace
- `POST /api/pull-url` -- pull a GGUF from arbitrary HTTPS URL (SSRF-guarded)
- `POST /api/import` -- import a local GGUF file
- `GET /status` -- live queue + active + idle_hot snapshot

## Setting up AI Agents

Pointing an AI agent (Hermes, langchain, llama-index, LiteLLM, raw OpenAI SDK, Ollama clients, etc.) at Turbohaul is two lines:

```yaml
base_url: http://<turbohaul-host>:11401/v1
api_key: dummy   # no auth required on the internal-network port
```

Turbohaul ships with sane defaults for multi-tool-call agent loops — `idle_hot_load_seconds=600`, `grace_seconds=30`, streaming SSE pass-through, tool-call field forwarding on both `/v1/chat/completions` and `/api/chat`, text-JSON tool-call recovery for jinja-template models that emit calls as content text, and ACTIVE_MATCH warm-slot reuse for same-`thread_id` follow-ups (sub-second after the first turn).

**Full guide:** [docs/AI_AGENT_SETUP.md](docs/AI_AGENT_SETUP.md) — per-agent config recipes (Hermes / OpenAI SDK / langchain / llama-index / LiteLLM / Ollama / curl), multi-tool-call workflow notes, production setup, validation smoke tests, and a troubleshooting table. For the recovery layer specifically, see [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).

## Multi-agent shared-GPU

Multiple agents can target the same Turbohaul endpoint at the same time. Turbohaul queues their requests, holds the warm model when possible, and cleanly swaps models when a different agent needs a different one. Proven with three concurrent agents running on one Blackwell card with zero force-evictions during a multi-model serialization smoke.

This is sharing-via-serialization, not concurrent-tensor-parallelism. See [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md) for the architecture, the proof, and when this does (and does not) fit your workload.

## TurboQuant flag doctrine

The Turbohaul manifest schema includes spawn-time TurboQuant KV-cache flags (turbo2/3/4) and an MTP speculative-decoding flag (`--spec-type draft-mtp`); the KV-cache flags below should be on by default for production manifests: `flash_attn`, `no_context_shift`, `cache_reuse: 256`, `slot_prompt_similarity: 0.5`, `no_perf`. These are spawn argv — manifest PUT does not affect a running `llama-server`; a cold-spawn (request with body `"keep_alive": 0`, natural `IDLE_HOT` teardown, or container restart) is required to pick up changes.

See [docs/TURBOQUANT_FLAGS.md](docs/TURBOQUANT_FLAGS.md) for the spawn-vs-request distinction, patching recipe, and verification recipe.

## Persistence

Production deployments must bind-mount `/var/lib/turbohaul`, ship an image tarball backup, mirror configs to a separate host, and have an auto-recovery entry. See [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md) for the full deployment persistence checklist.

## License

MIT (see LICENSE). All third-party deps audited MIT-compatible (see THIRD_PARTY_NOTICES.md).

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md). MrTrench (founder) shipped v0.2.3. Release notes in [CHANGELOG.md](CHANGELOG.md).
