# Third-Party Notices

Turbohaul-Manager is MIT-licensed (see LICENSE). It depends on the following
third-party components, all under MIT or MIT-compatible permissive licenses.

## Runtime backend (vendored / external)

- **Tom's TurboQuant fork of llama.cpp** (the `llama-server` binary) -- MIT
  - Source: `github.com/TheTom/llama-cpp-turboquant`
  - Upstream: the llama.cpp project (MIT)
  - MIT license preserved from upstream
- **ggml** (compiled into llama-server) -- MIT

## Python runtime dependencies (per pyproject.toml)

| Package           | License        | MIT-compatible |
|-------------------|----------------|----------------|
| fastapi           | MIT            | yes            |
| uvicorn           | BSD-3-Clause   | yes            |
| pydantic          | MIT            | yes            |
| pydantic-settings | MIT            | yes            |
| pyyaml            | MIT            | yes            |
| jsonschema        | MIT            | yes            |
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

### Transitive / build-time dependencies — license requirements

The full `npm install` tree (`package-lock.json`) resolves to permissive licenses only:
**168 MIT**, 10 ISC, 4 Apache-2.0, 1 BSD-3-Clause, and **1 CC-BY-4.0**. Two license requirements apply:

- **MIT** — [license text](https://choosealicense.com/licenses/mit/). The large majority of the tree (and this project's own code); preserve the copyright + permission notice.
- **Creative Commons Attribution 4.0 International (CC-BY-4.0)** — [license text](https://creativecommons.org/licenses/by/4.0/). One build-time data dependency, attributed below (CC-BY-4.0 is current; used in place of the older 3.0).

| Package | License | Note |
|---|---|---|
| caniuse-lite | CC-BY-4.0 | Browser-support database used at build time by browserslist / autoprefixer / postcss. Browser-compatibility data © the caniuse.com project and the Browserslist maintainers, used unmodified. CC-BY-4.0 requires giving appropriate credit + a link to the license. |

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
(JS deps). No copyleft (GPL/AGPL/LGPL) dependencies were detected. The one CC-BY-4.0
component (`caniuse-lite`) is attribution-only — not copyleft — and is attributed above.
