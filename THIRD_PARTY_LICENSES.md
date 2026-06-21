# Turbohaul-Manager — Third-Party Licenses

Turbohaul-Manager itself is licensed under the MIT License (see LICENSE).
The image and source tree include the following third-party components.

## Runtime backend

### llama.cpp (Tom's TurboQuant fork)
- **Project:** llama.cpp (https://github.com/ggerganov/llama.cpp)
- **Fork:** Tom's TurboQuant fork — selectable-quantization inference fork
- **License:** MIT
- **How we use it:** `llama-server` is invoked as a child subprocess. The binary
  is provided at `/opt/turbohaul/bin/llama-server` inside the container — either
  built from the pinned upstream fork in the CUDA Dockerfile build stage, or
  bind-mounted from a host-built binary via `docker-compose.yml`. Build flags and
  the pinned upstream version are part of your own image build.

## Python dependencies (PyPI, MIT/BSD/Apache 2.0)

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
| structlog | Apache-2.0 / MIT | Logging |

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

### Full dependency tree + license requirements

The complete `npm install` tree (`package-lock.json`) resolves to permissive licenses only:
**168 MIT**, 10 ISC, 4 Apache-2.0, 1 BSD-3-Clause, and **1 CC-BY-4.0**. Two license requirements
apply to the bundled / build-time dependencies:

1. **MIT** — [license text](https://choosealicense.com/licenses/mit/). Covers the large majority of
   the tree (and Turbohaul-Manager's own code). The only requirement is that the copyright and
   permission notice be preserved; they are retained in each package's distribution and in this file.

2. **Creative Commons Attribution 4.0 International (CC-BY-4.0)** —
   [license text](https://creativecommons.org/licenses/by/4.0/). Covers one build-time data
   dependency, attributed below. CC-BY-4.0 is used in place of the older 3.0: it is the current
   international version and is the version the package itself declares.

#### caniuse-lite — CC-BY-4.0 (attribution required)

- **Package:** `caniuse-lite` — the browser-support database consumed at build time by
  `browserslist` / `autoprefixer` / `postcss` to target the CSS output. Not shipped in the runtime
  bundle; present in the dependency tree.
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0) —
  <https://creativecommons.org/licenses/by/4.0/>
- **Attribution:** browser-compatibility data © the [caniuse.com](https://caniuse.com) project and
  the Browserslist maintainers, licensed under CC-BY-4.0. Used **unmodified**. Per CC-BY-4.0 you must
  give appropriate credit, provide a link to the license, and indicate if changes were made — no
  changes were made.

## API surface

The HTTP surface is named after Ollama's API. We implement a strict superset:
Turbohaul-Manager exposes Ollama-compatible endpoints (`/api/tags`, `/api/show`,
`/api/chat`, `/api/pull`) under nominative use only. Turbohaul-Manager is not
affiliated with Ollama.

## Container base images

- **CUDA variant** (`Dockerfile.cuda`) builds on `nvidia/cuda:12.9.0-runtime-ubuntu22.04`:
  - **NVIDIA CUDA runtime** — (c) NVIDIA Corporation, under the NVIDIA Deep Learning Container
    License / CUDA EULA. NVIDIA permits redistribution of the CUDA runtime as part of an
    application container (see https://docs.nvidia.com/cuda/eula/ and the NVIDIA Deep Learning
    Container License). This is NOT an MIT component.
  - **Ubuntu 22.04** base — Canonical; constituent packages under their own licenses
    (GPL/LGPL/MIT/BSD), freely redistributable as a base OS image.
- **Slim / CPU variant** (`Dockerfile`) builds on `python:3.11-slim` (Debian) — no NVIDIA component.

Turbohaul-Manager's own code is MIT. The published CUDA image is a composite: the MIT
application + MIT/BSD/Apache Python & frontend dependencies + the MIT llama.cpp TurboQuant
binary + the NVIDIA CUDA runtime (NVIDIA license) on an Ubuntu base. The slim variant carries
no NVIDIA-licensed component.

## Reproducing license texts

Each runtime dependency carries its full license text in its installed
distribution under the Python environment's `site-packages/<pkg>/LICENSE` (or
equivalent). For the frontend bundle, the source tree under `src/frontend/`
plus `npm ls --json` enumerates exact versions; `node_modules/<pkg>/LICENSE`
carries each text.
