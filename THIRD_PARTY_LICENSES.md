# Turbohaul-Manager — Third-Party Licenses

Turbohaul-Manager itself is licensed under the MIT License (see LICENSE).
The image and source tree include the following third-party components.

## Runtime backend

### llama.cpp (Tom's TurboQuant fork)
- **Project:** llama.cpp (https://github.com/ggerganov/llama.cpp)
- **Fork:** Tom's TurboQuant fork — selectable-quantization inference fork
- **License:** MIT
- **How we use it:** `llama-server` is invoked as a child subprocess. The binary
  is mounted into the container from the host at `/opt/turbohaul/bin/llama-server`;
  the supply chain (build flags, pinned SHA) is owned by the operator.

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

## API surface

The HTTP surface is named after Ollama's API. We implement a strict superset:
Turbohaul-Manager exposes Ollama-compatible endpoints (`/api/tags`, `/api/show`,
`/api/chat`, `/api/pull`) under nominative use only. Turbohaul-Manager is not
affiliated with Ollama.

## Reproducing license texts

Each runtime dependency carries its full license text in its installed
distribution at `/usr/local/lib/python3.11/site-packages/<pkg>/LICENSE` (or
equivalent). For the frontend bundle, the source tree under `src/frontend/`
plus `npm ls --json` enumerates exact versions; `node_modules/<pkg>/LICENSE`
carries each text.
