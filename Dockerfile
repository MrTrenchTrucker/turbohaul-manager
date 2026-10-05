# Turbohaul-Manager v0.2 multi-stage Dockerfile.
#
# Stage 1: Build the React+Vite frontend bundle.
# Stage 2: Install the Python package and copy the bundle.
#
# NOTE: This image does NOT include the llama-server binary. The binary is mounted
# at runtime from /opt/turbohaul/bin/llama-server (see docker-compose.yml). This
# decouples the TurboQuant engine compile (which requires GPU + CUDA build
# tools) from the management-plane image, and keeps the supply chain simple:
# the binary is shipped and audited separately.

# ----------------------------------------------------------------------------
# Stage 1: Frontend
# ----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-build

WORKDIR /work
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY src/frontend/. ./
RUN npm run typecheck && npm run build

# ----------------------------------------------------------------------------
# Stage 2: Python runtime
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml README.md ./
COPY src/turbohaul/ ./src/turbohaul/
RUN pip install --no-cache-dir .

# Frontend bundle from stage 1.
COPY --from=frontend-build /work/dist /opt/turbohaul/ui_dist

# The generic PATH shim, installed under each executable name it
# serves. The engine resolves `ffmpeg`/`ffprobe` from PATH exactly as if the
# binary were baked in; the shim is LOCAL-FIRST (a real local binary execv's
# with zero hops, so an operator-baked ffmpeg keeps working untouched) and
# otherwise sends the call through the forwarder to the plugin container's
# /ws/exec endpoint. The executable name IS the binary
# parameter (argv[0] of the symlink); optional per-install semantics live in
# /etc/ffshim/<name>.env. One COPY + symlinks (no per-name copies to drift).
COPY engine/ffshim/ffshim.py /opt/turbohaul/shim/ffshim.py
RUN chmod 755 /opt/turbohaul/shim/ffshim.py \
 && mkdir -p /etc/ffshim \
 && for name in ffmpeg ffprobe; do ln -sf /opt/turbohaul/shim/ffshim.py /usr/local/bin/$name; done

# Runtime directories (state + config).
RUN mkdir -p /var/lib/turbohaul /etc/turbohaul /opt/turbohaul/bin \
 && chmod 700 /var/lib/turbohaul

# Default config shipped with the image. Override via bind-mount.
COPY docker/turbohaul.default.yaml /etc/turbohaul/turbohaul.yaml

# Attribution + third-party licenses.
COPY THIRD_PARTY_LICENSES.md /usr/share/doc/turbohaul/THIRD_PARTY_LICENSES.md

EXPOSE 11401

ENV TURBOHAUL_CONFIG_PATH=/etc/turbohaul/turbohaul.yaml \
    TURBOHAUL_ALLOW_PUBLIC_BIND=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Healthcheck via FastAPI /health.
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD curl -fsS http://127.0.0.1:11401/health || exit 1

ENTRYPOINT ["turbohaul-manager"]
