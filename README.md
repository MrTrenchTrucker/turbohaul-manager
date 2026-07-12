![Turbohaul-Manager — cyberpunk GPU rig hauling inference traffic](docs/banner.png)

# Turbohaul-Manager

Ollama-shape inference manager using [Tom's TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) fork of llama.cpp.

FIFO queue + grace + IDLE_HOT hot-hold + model swap on Nvidia RTX GPU's including Blackwell.

**Precomputed KV-cache reuse.** Turbohaul saves each conversation's computed context and restores it across grace windows, model swaps, and even full idle unloads — so a follow-up turn reuses the already-computed prefix instead of re-reading the whole conversation from scratch. A large context that would cost ~327 s to re-prefill comes back in ~0.5 s: in a measured cold restore after a full unload, **154,647 tokens were reused with a 29-token prefill — roughly 629x less prefill work**. See [docs/REASONING_KV_REUSE.md](docs/REASONING_KV_REUSE.md) and [docs/KV_CACHE_MATCHING.md](docs/KV_CACHE_MATCHING.md).

## What it does

- Accepts OpenAI / Ollama-shape `/v1/chat/completions` requests
- Single-slot serial sidecar (one llama-server child holds the model)
- ACTIVE_MATCH cascade for same-thread follow-ups within a grace window (warm-process reuse)
- IDLE_HOT 5-minute warm-hold after grace expires: same-model follow-ups inherit the warm process; different-model swap tears down + spawns new
- Multiplexed multi-agent serialization on one shared GPU (proven with 3 production agents on a single host — see [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md))
- Transparent tool-call recovery for jinja-templated GGUFs that emit calls as text JSON in `message.content` instead of the structured `tool_calls` field (seen on some model families per upstream llama.cpp issues #20809 / #20837 / #20260) — see [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md)
- Safety guardrails: refuses spawn when VRAM / RAM / CPU / IO-wait would put the host at risk

## Quick start

```bash
# Clone, build the self-contained image, and run (Blackwell or older NVIDIA GPU)
git clone https://github.com/MrTrenchTrucker/turbohaul-manager.git
cd turbohaul-manager
docker build -f Dockerfile.engine-src -t turbohaul-manager:v0.6.0 .   # fully offline (vendored engine + wheels)

docker run --gpus all -p 11401:11401 \
    -v $(pwd)/state:/var/lib/turbohaul \
    -v $(pwd)/models:/var/lib/turbohaul/import-staging \
    turbohaul-manager:v0.6.0

# For broad NVIDIA arch support (Turing through Blackwell), build with Dockerfile.cuda-multi instead.
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

## MLX backend (macOS / Apple Silicon)

Turbohaul can run models two ways, chosen per model in the manifest with the
`backend` field:

- `llama.cpp` (default) — the TurboQuant `llama-server` binary. Works on Linux
  with a GPU, or on macOS using Metal. Needs a GGUF file.
- `mlx` — **Apple Silicon only.** Runs `python -m mlx_lm server` (the MLX
  framework), so the model runs natively on the Mac's GPU/Neural Engine with no
  CUDA and no Docker image. No GGUF file needed.

**Any MLX model works — nothing is hardcoded.** You tell Turbohaul which model to
use in the manifest; it does not ship or assume any particular model. Point it at:

- `model_repo` — a Hugging Face MLX model id (e.g. `mlx-community/Qwen3-1.7B-4B`), or
- `model_path` — a local folder with the model files already on disk.

If you set neither, the manifest is rejected with a clear error. This works with the
Qwen family (Qwen2.5 / Qwen3, Instruct and base, 4-bit and 8-bit MLX builds),
Llama, Mistral, Phi, Gemma — anything `mlx_lm` can load.

MLX is **Apple Silicon + macOS only**. On any other machine an `mlx` manifest
refuses to start (clear error), while `llama.cpp` manifests keep working as before.

How it fits in: `mlx_lm server` already speaks the same OpenAI-style API
(`/health` + `/v1/chat/completions`) that Turbohaul uses for llama.cpp, so health
checks and completion need no MLX-specific code. Any extra `mlx_server_flags` you
set are checked against a fixed allowlist (`SAFE_MLX_FLAGS` in
`src/turbohaul/mlx_spawn.py`) before they reach the command line, so a manifest
can't inject arbitrary flags.

Register a model — example: Qwen3 1.7B pulled from the Hugging Face hub:

```bash
curl -s -X PUT http://localhost:11401/api/manifests/qwen3-1.7b \
  -H "Content-Type: application/json" \
  -d '{
    "model_tag": "qwen3-1.7b",
    "backend": "mlx",
    "model_repo": "mlx-community/Qwen3-1.7B-4B",
    "mlx_server_flags": { "max_tokens": 4096, "use_default_chat_template": true }
  }'
```

Other ways to point at a model:

```jsonc
// A larger Qwen pulled from the hub:
{ "backend": "mlx", "model_repo": "mlx-community/Qwen3-8B-4bit" }

// A model you already have on disk (no download):
{ "backend": "mlx", "model_path": "/Volumes/Models/mlx-community/Qwen3-1.7B-4B" }

// Speculative decoding for more speed (draft + target both MLX):
{ "backend": "mlx", "model_repo": "mlx-community/Qwen3-1.7B-4B",
  "mlx_server_flags": { "draft_model": "mlx-community/Qwen3-0.6B-4bit",
                        "num_draft_tokens": 4 } }
```

Install on the Mac:

```bash
pip install -e ".[mlx]"        # adds mlx-lm to Turbohaul's environment
# or, if mlx-lm is already available (e.g. via oMLX):
conda install -c conda-forge mlx-lm
```

The macOS helper script is `turbohaul-launcher.sh`.

## Setting up AI Agents

Pointing an AI agent (langchain, llama-index, LiteLLM, raw OpenAI SDK, Ollama clients, etc.) at Turbohaul is two lines:

```yaml
base_url: http://<turbohaul-host>:11401/v1
api_key: dummy   # no auth required on the internal port
```

Turbohaul ships with sane defaults for multi-tool-call agent loops — `idle_hot_load_seconds=600`, `grace_seconds=30`, streaming SSE pass-through, tool-call field forwarding on both `/v1/chat/completions` and `/api/chat`, text-JSON tool-call recovery for jinja-template models that emit calls as content text, and ACTIVE_MATCH warm-slot reuse for same-`thread_id` follow-ups (sub-second after the first turn).

**Full guide:** [docs/AI_AGENT_SETUP.md](docs/AI_AGENT_SETUP.md) — per-client config recipes (OpenAI SDK / langchain / llama-index / LiteLLM / Ollama / curl), multi-tool-call workflow notes, production setup, validation smoke tests, and a troubleshooting table. For the recovery layer specifically, see [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).

## Multi-agent shared-GPU

Multiple agents can target the same Turbohaul endpoint at the same time. Turbohaul queues their requests, holds the warm model when possible, and cleanly swaps models when a different agent needs a different one. Proven in production with three agents (one primary worker plus two advisor agents) running on one Blackwell card with zero force-evictions during a multi-model serialization smoke.

This is sharing-via-serialization, not concurrent-tensor-parallelism. See [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md) for the architecture, the proof, and when this does (and does not) fit your workload.

## TurboQuant flag doctrine

The Turbohaul manifest schema includes five spawn-time TurboQuant flags that should be on by default for production manifests: `flash_attn`, `no_context_shift`, `cache_reuse: 256`, `slot_prompt_similarity: 0.5`, `no_perf`. These are spawn argv — manifest PUT does not affect a running `llama-server`; a cold-spawn (request with body `"keep_alive": 0`, natural `IDLE_HOT` teardown, or container restart) is required to pick up changes.

See [docs/TURBOQUANT_FLAGS.md](docs/TURBOQUANT_FLAGS.md) for the spawn-vs-request distinction, patching recipe, and verification recipe.

## Persistence

Production deployments must bind-mount `/var/lib/turbohaul`, ship an image tarball backup, mirror configs to a separate host, and have an auto-recovery entry. See [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md) for the full Turbohaul-specific hardening audit.

## License

MIT (see LICENSE). All third-party deps audited MIT-compatible (see THIRD_PARTY_NOTICES.md).

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md). MrTrench (founder) shipped v0.6.0. Release notes in [CHANGELOG.md](CHANGELOG.md).


## Self-contained offline build

The modified TurboQuant engine source (`engine/llama-cpp-turboquant/`), the Python
dependency wheels (`vendor/pywheels/`), and the prebuilt frontend all ship in this
repository, so you can build everything offline -- no external engine clone, PyPI, or
npm required.

### Build the image from source (needs the CUDA base image once)
```bash
docker build -f Dockerfile.engine-src -t turbohaul-manager:v0.6.0 .
```
Compiles the vendored engine (`engine/llama-cpp-turboquant/`) from source, installs Python
from the vendored wheels (`vendor/pywheels/`), and serves the committed frontend `dist` --
no PyPI, npm, or external clone. `Dockerfile.cuda-multi` is the same build with wider GPU
architecture coverage (Turing through Blackwell) if you prefer pulling dependencies from the
network instead of using the vendored wheels.

Requires an NVIDIA GPU. Licensing: all vendored deps are permissive (see THIRD_PARTY_NOTICES.md).
