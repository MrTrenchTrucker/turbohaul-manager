# Turbohaul-Manager — Third-Party Licenses

Turbohaul-Manager itself is licensed under the MIT License (see LICENSE).
The image and source tree include the following third-party components.

## Runtime backend

### llama.cpp (Tom's TurboQuant fork)
- **Project:** llama.cpp (https://github.com/ggerganov/llama.cpp)
- **Fork:** Tom's TurboQuant fork — selectable-quantization inference fork
- **License:** MIT
- **How we use it:** `llama-server` is invoked as a child subprocess. The binary
  is mounted into the slim image from the host at `/opt/turbohaul/bin/llama-server`
  (see `docker-compose.yml`); the supply chain (build flags, pinned source revision) is
  owned by the operator. The CUDA images (`Dockerfile.engine-src`, `Dockerfile.cuda-multi`)
  instead compile the engine from the source vendored in this repo.

## Python dependencies (PyPI, MIT/BSD/Apache 2.0, plus certifi under MPL-2.0)

| Package | License | Purpose |
|---|---|---|
| fastapi | MIT | HTTP framework |
| uvicorn[standard] | BSD-3-Clause | ASGI server |
| pydantic | MIT | Config + schema validation |
| pydantic-settings | MIT | Env-driven settings |
| pyyaml | MIT | YAML config loader |
| aiosqlite | MIT | Async SQLite for state |
| httpx | BSD-3-Clause | HTTP client (pull URLs / llama-server completion proxy) |
| websockets | BSD-3-Clause | WebSocket /ws/state |
| jsonschema | MIT | JSON Schema validation of structured-output requests |
| structlog | Apache-2.0 / MIT | Logging |
| certifi | MPL-2.0 | CA certificate bundle (pulled in by httpx; vendored as the unmodified wheel) |

## Frontend dependencies (npm, MIT)

| Package | License | Purpose |
|---|---|---|
| react / react-dom | MIT | UI runtime |
| react-router-dom | MIT | Client-side routing |
| vite | MIT | Build tool + dev server |
| @vitejs/plugin-react | MIT | React + Vite integration |
| typescript | Apache-2.0 | Type system |
| tailwindcss | MIT | Styling |
| autoprefixer | MIT | CSS post-processor |
| postcss | MIT | CSS pipeline |

## API surface

The HTTP surface is named after Ollama's API. We implement a strict superset:
Turbohaul-Manager exposes Ollama-compatible endpoints (`/api/tags`, `/api/show`,
`/api/chat`, `/api/pull`) under nominative use only. Turbohaul-Manager is not
affiliated with Ollama.

## Container base images

- **CUDA variants** (`Dockerfile.engine-src`, `Dockerfile.cuda-multi`) run on `nvidia/cuda:12.9.0-runtime-ubuntu22.04`:
  - **NVIDIA CUDA runtime** — (c) NVIDIA Corporation, under the NVIDIA Deep Learning Container
    License / CUDA EULA. NVIDIA permits redistribution of the CUDA runtime as part of an
    application container (see https://docs.nvidia.com/cuda/eula/ and the NVIDIA Deep Learning
    Container License). This is NOT an MIT component.
  - **Ubuntu 22.04** base — Canonical; constituent packages under their own licenses
    (GPL/LGPL/MIT/BSD), freely redistributable as a base OS image.
- **Slim / CPU variant** (`Dockerfile`) builds on `python:3.11-slim` (Debian) — no NVIDIA component.

Turbohaul-Manager's own code is MIT. The published CUDA image is a composite: the MIT
application + MIT/BSD/Apache Python & frontend dependencies (plus certifi, MPL-2.0, a file-level weak-copyleft
license, shipped as the unmodified wheel) + the MIT llama.cpp TurboQuant
binary + the NVIDIA CUDA runtime (NVIDIA license) on an Ubuntu base. The slim variant carries
no NVIDIA-licensed component.

## Reproducing license texts

Each runtime dependency carries its full license text in its installed
distribution at `/usr/local/lib/python3.11/site-packages/<pkg>/LICENSE` (or
equivalent). For the frontend bundle, the source tree under `src/frontend/`
plus `npm ls --json` enumerates exact versions; `node_modules/<pkg>/LICENSE`
carries each text.
