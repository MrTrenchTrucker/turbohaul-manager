# Turbohaul-Manager

Ollama-shape inference manager using `llama-server` from Tom's TurboQuant fork of `llama.cpp`.

Fleet-internal local inference one-stop-shop: FIFO queue + grace + idle hot-load, BYOM blob store, React+Vite control plane, OpenAI/Ollama-compatible HTTP surface.

**Architecture:** [ARCHITECTURE.md](./ARCHITECTURE.md) (v0.2 lock).
**Phase tracker:** [TODO.md](./TODO.md).

## Status

| Phase | Status |
|---|---|
| 0 — Forgejo + license audit | ✓ DONE 2026-05-16 |
| 1 — Architecture + RBSRS critique | ✓ DONE 2026-05-16 (v0.2 commit `5a001eb`) |
| 2 — Core queue + slot + supervision (7 waves) | ✓ DONE |
| 3 — Ollama + OpenAI API surface (5 waves) | ✓ DONE |
| 4 — Blob store + pull/import (3 waves) | ✓ DONE |
| 5 — React+Vite frontend (4 waves) | ✓ DONE |
| 6 — Dockerfile + smoke E2E + ship | IN PROGRESS — management-plane smoke green |

**Tests:** 330 green (BE pytest, mocked subprocess/GPU).
**Image:** `turbohaul-manager:v0.2` ~192 MB (no llama-server binary baked in).

## Quickstart (Docker)

```bash
# 1. Build the image (or pull when published).
docker build -t turbohaul-manager:v0.2 .

# 2. Supply a llama-server binary (built from Tom's TurboQuant fork).
#    Path will be bind-mounted into the container at /opt/turbohaul/bin/llama-server.
export LLAMA_SERVER_HOST_PATH=/path/to/your/llama-server

# 3. Start.
docker compose up -d

# 4. Sanity check.
curl http://127.0.0.1:11401/health
curl http://127.0.0.1:11401/api/version
open http://127.0.0.1:11401/ui/      # Dashboard / Queue / Blob / Config / Logs / Settings
```

GPU note: Tom's TurboQuant fork uses CUDA — uncomment the `deploy.resources.devices` block in `docker-compose.yml` to expose the host GPU.

## Configuration

`docker/turbohaul.default.yaml` ships inside the image at `/etc/turbohaul/turbohaul.yaml`. Override by bind-mounting your own. Key environment variables:

| Variable | Effect |
|---|---|
| `TURBOHAUL_CONFIG_PATH` | Yaml path (default `/etc/turbohaul/turbohaul.yaml`) |
| `TURBOHAUL_ALLOW_PUBLIC_BIND` | `1` → uvicorn binds 0.0.0.0 (container only — v0.2 §3.2) |
| `TURBOHAUL_LOG_LEVEL` | `info` / `debug` / `warning` etc. |
| `TURBOHAUL_GRACE_S` | Runtime override for grace seconds |
| `TURBOHAUL_IDLE_HOT_S` | Runtime override for idle hot-load seconds |
| `TURBOHAUL_PORT` | Override server port |
| `HF_API_KEY` | Required for `/api/pull-hf` against private repos |

## HTTP surface

See [ARCHITECTURE.md §9](./ARCHITECTURE.md) for the full route table. Highlights:

| Path | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + version |
| `/status` | GET | Queue + active + grace + idle snapshot |
| `/api/version` | GET | User-agent + backend info |
| `/api/config` | GET/PUT | Boot/runtime split — boot fields 403 on PUT |
| `/api/tags` | GET | Installed model list (Ollama-shape) |
| `/api/show` | GET | Per-model detail (Ollama-shape) |
| `/api/manifests/{tag}` | GET/PUT/DELETE | Manifest CRUD with ETag/If-Match |
| `/api/chat` | POST | Chat completion (Ollama-shape) |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-shape) |
| `/api/pull-url` | POST | Pull a model by HTTPS URL (SSRF-guarded) |
| `/api/pull-hf` | POST | Pull from HuggingFace (allowlisted hosts) |
| `/api/import` | POST | Import a local GGUF file (sandboxed root + O_NOFOLLOW) |
| `/api/delete` | DELETE | Delete a blob by sha256 |
| `/ws/state` | WS | Live state events (redacted) |
| `/ui/{...}` | GET | Static UI bundle + SPA fallback + CSP |

## License

Turbohaul-Manager: MIT (see [LICENSE](./LICENSE)).
Inference backend: `llama-server` from Tom's TurboQuant fork of `llama.cpp` (MIT — see [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md)).
Ollama-compatible HTTP API surface — nominative use only; this project is not affiliated with Ollama.
