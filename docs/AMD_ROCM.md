# AMD ROCm GPU Support

Turbohaul-Manager supports AMD GPUs via ROCm (Radeon Open Compute) through a pluggable GPU backend abstraction. This document covers setup, configuration, and troubleshooting for AMD GPUs.

## Supported Hardware

| GPU Series | Architecture | ROCm Version | Status |
|---|---|---|---|
| Radeon RX 7900 XTX/XT | gfx1100 | 6.0+ | Supported |
| Radeon RX 7800 XT/7700 XT | gfx1101 | 6.0+ | Supported |
| Radeon RX 6900 XT/6800 XT | gfx1030 | 5.7+ | Supported |
| Instinct MI300X/A | gfx942 | 6.0+ | Supported |
| Instinct MI250X/MI210 | gfx90a | 5.7+ | Supported |

## Quick Start

### Option A: Docker (Native Linux with /dev/kfd)

For native Linux systems with ROCm installed and `/dev/kfd` available:

```bash
docker build -f Dockerfile.rocm -t turbohaul-manager:v0.5-rocm .

docker run \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -p 127.0.0.1:11401:11401 \
  -e TURBOHAUL_RUNTIME__GPU_BACKEND=rocm \
  turbohaul-manager:v0.5-rocm
```

Or use the provided compose file:

```bash
LLAMA_SERVER_HOST_PATH=/path/to/llama-server \
  docker compose -f docker-compose.rocm.yml up -d
```

### Option B: WSL2 with ROCDXG (Windows + AMD GPU)

For running on Windows with an AMD GPU via WSL2 + ROCDXG. This is the recommended path for Windows users with AMD Radeon desktop GPUs.

#### Automated Setup (Recommended)

Run the one-shot setup script from an **interactive WSL2 terminal**:

```bash
# Copy to WSL2
cp /mnt/c/Users/<you>/path/to/turbohaul-manager/turbohaul-rocm-setup.sh /tmp/
cd /tmp && chmod +x turbohaul-rocm-setup.sh

# Run (requires sudo for first-time setup)
sudo bash turbohaul-rocm-setup.sh
```

The script is idempotent — safe to re-run. It handles ROCm installation, libxml2 fixes, llama.cpp build, turbohaul install, and model setup. See below for manual steps if you prefer.

#### Prerequisites

1. **AMD Adrenalin 26.2.2+** driver installed on Windows
2. **Windows SDK 10.0.26100.0** installed on Windows (https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
3. **WSL2** with Ubuntu 22.04 or 24.04
4. **Docker Desktop** with WSL2 integration enabled
5. **Restart Windows** after installing driver + SDK

#### Step 1: Verify GPU passthrough

```bash
# In WSL2 terminal
ls /dev/dxg
# Must exist — this is the ROCDXG GPU passthrough device
```

If `/dev/dxg` doesn't exist, restart WSL: `wsl --shutdown` from PowerShell, then reopen.

#### Step 2: Install ROCm 7.2.1

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake python3-venv wget

wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb
sudo apt update

# IMPORTANT: --usecase=graphics,rocm (NOT wsl) and --no-dkms
sudo amdgpu-install -y --usecase=graphics,rocm --no-dkms
```

#### Step 3: Fix libxml2 (Ubuntu 26.04 only)

ROCm 7.2.1's `lld` linker requires `libxml2.so.2` but Ubuntu 26.04 ships `libxml2.so.16`.

```bash
wget -q https://launchpad.net/ubuntu/+archive/primary/+files/libxml2_2.12.7+dfsg+really2.9.14-0.4ubuntu0.4_amd64.deb
dpkg-deb -x libxml2_2.12.7+dfsg+really2.9.14-0.4ubuntu0.4_amd64.deb /tmp/libxml2-extract
sudo cp /tmp/libxml2-extract/usr/lib/x86_64-linux-gnu/libxml2.so.2.9.14 /lib/x86_64-linux-gnu/
sudo ln -sf /lib/x86_64-linux-gnu/libxml2.so.2.9.14 /lib/x86_64-linux-gnu/libxml2.so.2
sudo ldconfig
```

#### Step 4: Build librocdxg (WSL2 GPU bridge)

```bash
# Symlink Windows SDK headers (no spaces in path)
sudo ln -sf "/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0" /opt/winsdk
sudo mkdir -p /opt/winsdk-combined
for f in /opt/winsdk/shared/*; do sudo ln -sf "$f" /opt/winsdk-combined/ 2>/dev/null; done
for f in /opt/winsdk/um/*; do sudo ln -sf "$f" /opt/winsdk-combined/ 2>/dev/null; done

# Build from source
git clone --depth 1 https://github.com/ROCm/librocdxg.git
cd librocdxg
mkdir -p build && cd build
cmake .. -DWIN_SDK=/opt/winsdk-combined
make -j$(nproc)
sudo make install
```

#### Step 5: Set environment variables

```bash
sudo tee /etc/profile.d/rocm.sh << 'EOF'
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_ENABLE_DXG_DETECTION=1
export HSA_TOOLS_DISABLE_REGISTER=1
export HSA_OVERRIDE_GFX_VERSION=11.0.1
EOF

sudo ldconfig
source /etc/profile.d/rocm.sh
```

#### Step 6: Verify GPU detection

```bash
rocminfo | grep -A5 "Marketing Name"
# Should show: Marketing Name: AMD Radeon RX 7800 XT
```

#### Step 7: Build llama.cpp with ROCm

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_HIP=ON
cmake --build build -j$(nproc)
```

#### Step 8: Run turbohaul-manager

```bash
# Create venv
python3 -m venv /tmp/turbohaul-env
source /tmp/turbohaul-env/bin/activate

# Install turbohaul
cd /path/to/turbohaul-manager
pip install -e .

# Set ROCm env vars
export HSA_OVERRIDE_GFX_VERSION=11.0.1
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib

# Create config
sudo mkdir -p /etc/turbohaul
sudo mkdir -p /var/lib/turbohaul/{blobs,manifests,import-staging}
sudo chown -R $USER:$USER /var/lib/turbohaul

sudo tee /etc/turbohaul/turbohaul.yaml > /dev/null << 'EOF'
server:
  host: 127.0.0.1
  port: 11401
  allow_public_bind: true
storage:
  blob_store_path: /var/lib/turbohaul/blobs
  manifests_path: /var/lib/turbohaul/manifests
  import_allowed_root: /var/lib/turbohaul/import-staging
  state_db_path: /var/lib/turbohaul/state.sqlite
runtime:
  llama_server_binary: /home/$USER/llama.cpp/build/bin/llama-server
  default_port_base: 11500
  gpu_backend: rocm
ui:
  enabled: false
  static_path: /opt/turbohaul/ui_dist
queue:
  max_parallel_sidecars: 1
  staging_queue_depth: 100
  acceptance_buffer_max: 10000
  grace_seconds: 30
  idle_hot_load_seconds: 600
  max_grace_extensions: 5
  loading_health_timeout_s: 600
  drained_sigterm_window_active_s: 15
  drained_sigterm_window_cold_s: 5
  safety_enabled: false
pull:
  hf_host_allowlist:
    - huggingface.co
    - hf.co
  pull_url_https_only: true
  pull_concurrency: 2
  pull_chunk_size_mb: 64
  per_stream_max_bytes: 107374182400
EOF

# Start
turbohaul-manager --allow-public-bind --log-level debug
```

#### Step 9: Load a model and test

```bash
# Download a GGUF model
mkdir -p ~/models
wget -O ~/models/qwen2.5-3b-q4_k_m.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

# Copy to blob store
HASH=$(sha256sum ~/models/qwen2.5-3b-q4_k_m.gguf | awk '{print $1}')
cp ~/models/qwen2.5-3b-q4_k_m.gguf /var/lib/turbohaul/blobs/$HASH

# Create manifest
curl -X PUT http://localhost:11401/api/manifests/default \
  -H "Content-Type: application/json" \
  -d "{
    \"gguf_blob_sha256\": \"$HASH\",
    \"gguf_size_bytes\": $(stat -c%s ~/models/qwen2.5-3b-q4_k_m.gguf),
    \"context_size\": 4096,
    \"expected_vram_bytes\": 2147483648,
    \"llama_server_flags\": {\"parallel\": 1}
  }"

# Test health
curl http://localhost:11401/health

# Test chat
curl http://localhost:11401/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}'
```

## Configuration

### GPU Backend Selection

The `runtime.gpu_backend` setting controls which GPU monitoring backend is used:

| Value | Behavior |
|---|---|
| `"auto"` (default) | Probe nvidia-smi first, then rocm-smi; use first available |
| `"nvidia"` | Force NVIDIA backend (uses nvidia-smi) |
| `"rocm"` | Force AMD ROCm backend (uses rocm-smi) |

Can also be set via environment variable:
```bash
TURBOHAUL_RUNTIME__GPU_BACKEND=rocm
```

### ROCm Environment Variables

| Variable | Description | WSL2 Required |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | Override GPU architecture version | Yes |
| `HSA_ENABLE_DXG_DETECTION` | Enable ROCDXG GPU detection in WSL2 | Yes |
| `HSA_TOOLS_DISABLE_REGISTER` | Disable rocprofiler (avoids WSL2 sysfs conflicts) | Yes |
| `LD_LIBRARY_PATH` | Must include `/opt/rocm/lib:/usr/lib/wsl/lib` | Yes |
| `HIP_VISIBLE_DEVICES` | Comma-separated list of visible GPU indices | No |

### Common `HSA_OVERRIDE_GFX_VERSION` Values

| GPU | Value |
|---|---|
| RX 7900 XTX/XT | `"11.0.0"` |
| RX 7800 XT | `"11.0.1"` |
| RX 7700 XT | `"11.0.1"` |
| RX 6900 XT | `"10.3.0"` |
| RX 6800 XT | `"10.3.0"` |
| MI300X/A | `"9.4.0"` |
| MI250X | `"9.0.10"` |
| MI210 | `"9.0.10"` |

### Manifest Flags

All manifest flags (`llama_server_flags`) are GPU-agnostic. The same flags work for both NVIDIA and AMD:

```json
{
  "model_tag": "qwen3-32b-q4",
  "gguf_blob_sha256": "...",
  "llama_server_flags": {
    "n_gpu_layers": "all",
    "ctx_size": 8192,
    "flash_attn": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0"
  }
}
```

## Docker Test Image

`Dockerfile.rocm-test` is a lightweight image for validating the management plane without the full ROCm stack:

```bash
docker build -f Dockerfile.rocm-test -t turbohaul-manager:rocm-test .

# Native Linux (with /dev/kfd)
docker run -d --name turbohaul-rocm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -p 11401:11401 turbohaul-manager:rocm-test

# WSL2 with ROCDXG
docker run -d --name turbohaul-rocm \
  --device=/dev/dxg \
  -p 11401:11401 turbohaul-manager:rocm-test

# Verify
curl http://localhost:11401/health
```

## Troubleshooting

### rocm-smi: unrecognized arguments

The GPU backend uses `--showpids` (plural) for process scanning. If you see errors from rocm-smi about unrecognized arguments, make sure you're using the version shipped with your ROCm install (`/opt/rocm/bin/rocm-smi`), not a system package.

### "Driver not initialized (amdgpu not found in modules)"

This is a cosmetic warning from rocm-smi when it can't fully query the driver during backend detection. It's harmless — turbohaul-manager degrades gracefully and the GPU still works for inference. If you want a clean startup, set `HSA_TOOLS_DISABLE_REGISTER=1` in your environment before starting turbohaul-manager.

### DNS rebind false positive on HuggingFace pulls

HuggingFace CDN uses round-robin DNS (multiple A records for same host). The SSRF guard allows different IPs from the same host as long as the hostname matches.

### `libxml2.so.2: cannot open shared object file`

Ubuntu 25.10+ ships `libxml2.so.16` but ROCm 7.2.x's `lld` needs `libxml2.so.2`. See Step 3 above for the symlink fix.

### `Driver not initialized (amdgpu not found in modules)`

Ensure `HSA_ENABLE_DXG_DETECTION=1` is set in the environment **before** starting turbohaul-manager. The env vars are inherited by llama-server child processes.

### GPU not detected in WSL2

- Verify `/dev/dxg` exists: `ls /dev/dxg`
- If missing, run `wsl --shutdown` from PowerShell, reopen WSL2
- Ensure AMD Adrenalin 26.2.2+ is installed on Windows
- Ensure Windows SDK 10.0.26100.0 is installed

### librocdxg build fails

- Ensure Windows SDK headers are accessible at `/opt/winsdk-combined/`
- Check that `windows.h` exists: `ls /opt/winsdk-combined/windows.h`
- Use `sudo` for symlink/mkdir commands (permission denied = forgot sudo)

### Container can't find docker

Enable WSL integration in Docker Desktop: Settings > Resources > WSL Integration > enable for your distro.

### Out of Memory

```bash
rocm-smi --showmeminfo vram

# Use KV cache offloading to host RAM
# In manifest: "no_kv_offload": true

# Or reduce context size
# In manifest: "ctx_size": 4096
```

## Safety Gates

Turbohaul's safety gates (VRAM headroom, RAM, CPU load, IO wait) work identically on AMD. The GPU backend abstraction means `safety_enabled: true` works on both NVIDIA and AMD.

When running without ROCm tooling (e.g., dev environments), the VRAM gate degrades gracefully: `passed-no-probe` instead of blocking. You can also disable safety gates entirely:

```yaml
queue:
  safety_enabled: false
```

## Test Results

All GPU backend tests pass on both NVIDIA and AMD codepaths:

```
tests/test_gpu_backend.py   — 30 passed (NvidiaBackend, RocmBackend, detect, singleton, backward-compat)
tests/test_config.py        — 35+ passed (gpu_backend config field, validation, env overrides)
```

7 pre-existing Windows-only failures (`os.killpg`, `signal.SIGKILL`, `os.getloadavg`, `fcntl`, `os.O_NOFOLLOW`) — zero regressions.
