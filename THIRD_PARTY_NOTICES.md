# Third-Party Notices

Turbohaul-Manager is MIT-licensed (see LICENSE). It depends on the following
third-party components, all under MIT or MIT-compatible permissive licenses.

## Runtime backend (vendored / external)

- **Tom's TurboQuant fork of llama.cpp** (the `llama-server` binary) -- MIT
  - Upstream: ggerganov/llama.cpp (MIT)
  - Fork-of-record: MIT preserved
- **ggml** (compiled into llama-server) -- MIT

## Python runtime dependencies (per pyproject.toml)

| Package           | License        | MIT-compatible |
|-------------------|----------------|----------------|
| fastapi           | MIT            | yes            |
| uvicorn           | BSD-3-Clause   | yes            |
| pydantic          | MIT            | yes            |
| pydantic-settings | MIT            | yes            |
| pyyaml            | MIT            | yes            |
| aiosqlite         | MIT            | yes            |
| httpx             | BSD-3-Clause   | yes            |
| websockets        | BSD-3-Clause   | yes            |
| structlog         | MIT / Apache-2.0 dual | yes     |
| starlette (via FastAPI) | BSD-3-Clause | yes      |

## Frontend dependencies (per src/frontend/package.json)

| Package                | License      | MIT-compatible |
|------------------------|--------------|----------------|
| react / react-dom      | MIT          | yes            |
| react-router-dom       | MIT          | yes            |
| vite                   | MIT          | yes            |
| @vitejs/plugin-react   | MIT          | yes            |
| tailwindcss            | MIT          | yes            |
| typescript             | Apache-2.0   | yes            |
| autoprefixer           | MIT          | yes            |
| postcss                | MIT          | yes            |
| @types/react           | MIT          | yes            |
| @types/react-dom       | MIT          | yes            |

## Dev-only dependencies

| Package           | License | MIT-compatible |
|-------------------|---------|----------------|
| pytest            | MIT     | yes            |
| pytest-asyncio    | Apache-2.0 | yes         |
| pytest-cov        | MIT     | yes            |
| pytest-mock       | MIT     | yes            |
| ruff              | MIT     | yes            |
| setuptools        | MIT     | yes            |
| wheel             | MIT     | yes            |

## Verification method

All licenses listed above are the official upstream licenses as of the package
versions pinned in `pyproject.toml` (Python deps) and `src/frontend/package.json`
(JS deps), audited 2026-05-17 at v0.2.1 ship. No copyleft (GPL/AGPL/LGPL) deps
were detected.


## Vendored engine — self-contained repo
The turboquant llama.cpp fork ("Tom's TurboQuant" + heavy in-house mods) is now
SHIPPED IN THIS REPO at `engine/llama-cpp-turboquant/` (a pinned source snapshot).
See `engine/llama-cpp-turboquant/VENDORED.md`. Build the engine from source (no external
image, PyPI, or npm) with `Dockerfile.engine-src`, or with wider GPU-architecture
coverage using `Dockerfile.cuda-multi`.


## Vendored dependency licenses (full-vendor license audit)
All vendored deps are permissively licensed, freely redistributable (no copyleft):
- **Engine** (engine/llama-cpp-turboquant): MIT (Tom's TurboQuant fork of llama.cpp; MIT preserved).
- **Python** (vendor/pywheels): MIT/BSD (fastapi, pydantic, uvicorn, httpx, etc.).
- **Frontend** (src/frontend/node_modules): MIT/ISC/Apache-2.0/BSD-3-Clause; plus **caniuse-lite** (browser-compat DATA) under **CC-BY-4.0** (attribution: "caniuse-lite (c) caniuse.com, CC-BY-4.0"). No GPL/AGPL/LGPL.
- **Base OS image** (if mirrored/baked to internal registry): NVIDIA CUDA container images under the NVIDIA Deep Learning Container License (internal use of copies + internal derivative images permitted; NO standalone third-party redistribution). Retain NVIDIA + Canonical(Ubuntu) copyright/license notices + EULA in any derived image; keep cuBLAS Modified-BSD attributions; GPU-only execution. CUDA Toolkit redistributables per EULA Attachment A. Ubuntu base internal-mirror permitted (Canonical IP policy).
