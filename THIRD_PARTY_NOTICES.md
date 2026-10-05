# Third-Party Notices

Turbohaul-Manager is MIT-licensed (see LICENSE). It depends on the following
third-party components, all under MIT or MIT-compatible licenses (mostly permissive; the one weak-copyleft exception,
certifi under MPL-2.0 among the vendored Python wheels, is listed below).

## Runtime backend (vendored / external)

- **Tom's TurboQuant fork of llama.cpp** (the `llama-server` binary) -- MIT
  - Upstream: ggerganov/llama.cpp (MIT)
  - The fork keeps the MIT license (`engine/llama-cpp-turboquant/LICENSE`)
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
| jsonschema        | MIT            | yes            |
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

All licenses listed above are the upstream licenses of the packages declared in
`pyproject.toml` (Python deps) and `src/frontend/package.json` (JS deps). No
GPL/AGPL/LGPL packages are among the dependencies declared there.


## Vendored engine -- self-contained repo
The turboquant llama.cpp fork ("Tom's TurboQuant" plus local modifications) is
shipped in this repo at `engine/llama-cpp-turboquant/` (a source snapshot without
history). See `engine/llama-cpp-turboquant/VENDORED.md`. Build the engine from source
with `Dockerfile.engine-src` (or `Dockerfile.cuda-multi`, which also compiles it from
the vendored source).


## Vendored dependency licenses
All vendored dependencies are freely redistributable. None is GPL/AGPL/LGPL, and all are permissively licensed except one weak-copyleft Python package, certifi (MPL-2.0), noted below:
- **Engine** (engine/llama-cpp-turboquant): MIT (Tom's TurboQuant fork of llama.cpp; MIT preserved).
- **Python** (vendor/pywheels, prebuilt wheels as downloaded from PyPI): MIT/BSD/Apache-2.0/PSF-2.0 (fastapi, pydantic, uvicorn, httpx, etc.), plus **certifi** (CA certificate bundle) under **MPL-2.0**, a file-level weak-copyleft license: it is redistributed here as the unmodified wheel.
- **Frontend** (npm packages installed under `src/frontend/node_modules` at build time): MIT/ISC/Apache-2.0/BSD-3-Clause; plus **caniuse-lite** (browser-compat DATA) under **CC-BY-4.0** (attribution: "caniuse-lite (c) caniuse.com, CC-BY-4.0"). No GPL/AGPL/LGPL.
- **Base OS image** (if mirrored or baked into a private registry): NVIDIA CUDA container images under the NVIDIA Deep Learning Container License (internal use of copies + internal derivative images permitted; NO standalone third-party redistribution). Retain NVIDIA + Canonical(Ubuntu) copyright/license notices + EULA in any derived image; keep cuBLAS Modified-BSD attributions; GPU-only execution. CUDA Toolkit redistributables per EULA Attachment A. Ubuntu base internal-mirror permitted (Canonical IP policy).
