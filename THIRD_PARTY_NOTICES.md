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
