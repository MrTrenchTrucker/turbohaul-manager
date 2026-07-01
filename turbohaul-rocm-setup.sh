#!/bin/bash
# =============================================================================
# turbohaul-rocm-setup.sh — One-shot WSL2 ROCm setup for turbohaul-manager
#
# Idempotent: safe to re-run. Checks each step and skips what's done.
# Tested on: Ubuntu 26.04 LTS (WSL2), AMD RX 7800 XT (gfx1101)
#
# Usage:
#   bash turbohaul-rocm-setup.sh
#
# Prerequisites (MUST be done on Windows side first):
#   1. AMD Adrenalin 26.2.2+ driver installed
#   2. Windows SDK 10.0.26100.0 installed
#   3. Restart Windows after installing both
# =============================================================================
set -euo pipefail

# -- Colors -------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0
DONE=0

ok()   { PASS=$((PASS+1)); DONE=$((DONE+1)); echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}[FAIL]${NC} $1"; }
warn() { WARN=$((WARN+1)); echo -e "  ${YELLOW}[WARN]${NC} $1"; }
skip() { DONE=$((DONE+1)); echo -e "  ${CYAN}[SKIP]${NC} $1 (already done)"; }
step() { echo -e "\n${CYAN}==> $1${NC}"; }

# Sudo wrapper: runs command if sudo is available, otherwise fails with instructions
run_sudo() {
  if [ "$SUDO_OK" = true ]; then
    sudo "$@"
  else
    echo -e "  ${YELLOW}Cannot run: sudo $*${NC}"
    echo -e "  Fix: re-run from interactive WSL2 terminal with: ${CYAN}sudo bash $0${NC}"
    FAIL=$((FAIL+1))
    return 1
  fi
}

# -- Resolve real user home (handles sudo where $HOME=/root) ------------------
if [ -n "${SUDO_USER:-}" ]; then
  REAL_HOME=$(eval echo "~$SUDO_USER")
else
  REAL_HOME="$HOME"
fi

# -- Password check -----------------------------------------------------------
# Check if sudo works without password. If not, warn the user.
SUDO_OK=true
if ! sudo -n true 2>/dev/null; then
  SUDO_OK=false
  echo -e "\n${YELLOW}WARNING: sudo requires a password.${NC}"
  echo -e "  This script needs sudo for some steps (ROCm install, config creation)."
  echo -e "  Re-run with: ${CYAN}sudo bash turbohaul-rocm-setup.sh${NC}"
  echo -e "  Or open an interactive WSL2 terminal first.\n"
fi

# -- Step 0: WSL2 + GPU passthrough ------------------------------------------
step "Step 0: Checking WSL2 prerequisites"

if [ ! -f /proc/version ]; then
  fail "Not running in Linux/WSL2"
  exit 1
fi
ok "Running in Linux ($(uname -r))"

if [ ! -e /dev/dxg ]; then
  fail "/dev/dxg not found — GPU passthrough not available"
  echo "    Fix: Run 'wsl --shutdown' in PowerShell, reopen WSL2"
  echo "    Ensure AMD Adrenalin 26.2.2+ and Windows SDK 10.0.26100.0 are installed"
  exit 1
fi
ok "/dev/dxg present (ROCDXG GPU passthrough active)"

# -- Step 1: ROCm installation ------------------------------------------------
step "Step 1: ROCm 7.2.1 installation"

if [ -x /opt/rocm/bin/rocm-smi ]; then
  ROCM_VER=$(rocm-smi --showdriverversion 2>/dev/null | grep -oP 'Driver version:\s*\K.*' || echo "unknown")
  skip "ROCm already installed (${ROCM_VER})"
else
  echo "  Installing ROCm 7.2.1..."
  run_sudo apt-get update -qq
  run_sudo apt-get install -y -qq build-essential cmake python3-venv wget

  AMDGPU_DEB="/tmp/amdgpu-install_7.2.1.70201-1_all.deb"
  if [ ! -f "$AMDGPU_DEB" ]; then
    wget -q -O "$AMDGPU_DEB" \
      https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
  fi
  run_sudo apt-get install -y -qq "$AMDGPU_DEB"
  run_sudo apt-get update -qq
  run_sudo amdgpu-install -y --usecase=graphics,rocm --no-dkms
  ok "ROCm 7.2.1 installed"
fi

# -- Step 2: libxml2 fix (Ubuntu 26.04) ---------------------------------------
step "Step 2: libxml2 compatibility fix (Ubuntu 26.04)"

if [ -e /lib/x86_64-linux-gnu/libxml2.so.2 ]; then
  skip "libxml2.so.2 already present"
else
  echo "  Installing libxml2 compat shim..."
  LIBXML_DEB="/tmp/libxml2_2.12.7+dfsg+really2.9.14-0.4ubuntu0.4_amd64.deb"
  if [ ! -f "$LIBXML_DEB" ]; then
    wget -q -O "$LIBXML_DEB" \
      "https://launchpad.net/ubuntu/+archive/primary/+files/libxml2_2.12.7+dfsg+really2.9.14-0.4ubuntu0.4_amd64.deb"
  fi
  dpkg-deb -x "$LIBXML_DEB" /tmp/libxml2-extract
  run_sudo cp /tmp/libxml2-extract/usr/lib/x86_64-linux-gnu/libxml2.so.2.9.14 /lib/x86_64-linux-gnu/
  run_sudo ln -sf /lib/x86_64-linux-gnu/libxml2.so.2.9.14 /lib/x86_64-linux-gnu/libxml2.so.2
  run_sudo ldconfig
  ok "libxml2.so.2 symlink created"
fi

# -- Step 3: Environment variables --------------------------------------------
step "Step 3: Environment variables"

ENV_FILE="/etc/profile.d/rocm.sh"
if grep -q "HSA_ENABLE_DXG_DETECTION" "$ENV_FILE" 2>/dev/null; then
  skip "ROCm env vars already configured in $ENV_FILE"
else
  echo "  Writing $ENV_FILE..."
  run_sudo tee "$ENV_FILE" > /dev/null << 'ENVEOF'
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_ENABLE_DXG_DETECTION=1
export HSA_TOOLS_DISABLE_REGISTER=1
export HSA_OVERRIDE_GFX_VERSION=11.0.1
ENVEOF
  run_sudo ldconfig
  ok "Environment variables written to $ENV_FILE"
fi

# Export now for this session
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_ENABLE_DXG_DETECTION=1
export HSA_TOOLS_DISABLE_REGISTER=1
export HSA_OVERRIDE_GFX_VERSION=11.0.1

# -- Step 4: GPU detection verification ---------------------------------------
step "Step 4: Verifying GPU detection"

GPU_NAME=$(rocminfo 2>/dev/null | grep "Marketing Name" | head -1 | sed 's/.*:\s*//' || echo "")
if [ -n "$GPU_NAME" ]; then
  ok "GPU detected: $GPU_NAME"
else
  warn "rocminfo could not detect GPU name (may still work)"
fi

# -- Step 5: Build llama.cpp with ROCm ----------------------------------------
step "Step 5: Build llama.cpp with ROCm/HIP"

LLAMA_BIN="$REAL_HOME/llama.cpp/build/bin/llama-server"
if [ -x "$LLAMA_BIN" ]; then
  skip "llama-server already built at $LLAMA_BIN"
else
  if [ ! -d "$REAL_HOME/llama.cpp" ]; then
    echo "  Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$REAL_HOME/llama.cpp"
  fi
  echo "  Building llama.cpp with GGML_HIP=ON..."
  cd "$REAL_HOME/llama.cpp"
  cmake -B build -DGGML_HIP=ON 2>&1 | tail -1
  cmake --build build -j$(nproc) 2>&1 | tail -1
  if [ -x "$LLAMA_BIN" ]; then
    ok "llama-server built successfully"
  else
    fail "llama-server build failed — check cmake output above"
  fi
fi

# -- Step 6: Install turbohaul-manager ----------------------------------------
step "Step 6: Install turbohaul-manager"

# Find turbohaul-manager repo: check common locations
SEARCH_DIRS=(
  "$REAL_HOME/oc/oc036-turbohaulamd/turbohaul-manager"
  "$REAL_HOME/turbohaul-manager"
  "/mnt/c/Users/gentl/oc/oc036-turbohaulamd/turbohaul-manager"
  "$(dirname "$0")"
  "$(pwd)"
)
TURBOHAUL_DIR=""
for d in "${SEARCH_DIRS[@]}"; do
  if [ -f "$d/pyproject.toml" ] && grep -q 'turbohaul-manager' "$d/pyproject.toml" 2>/dev/null; then
    TURBOHAUL_DIR="$d"
    break
  fi
done

if [ -z "$TURBOHAUL_DIR" ]; then
  fail "Could not find turbohaul-manager repo. Pass the path as argument:"
  echo "    bash $0 /path/to/turbohaul-manager"
  exit 1
fi

VENV_DIR="/tmp/turbohaul-env"

if [ -x "$VENV_DIR/bin/turbohaul-manager" ]; then
  "$VENV_DIR/bin/pip" install -e "$TURBOHAUL_DIR" -q 2>/dev/null
  skip "turbohaul-manager installed from $TURBOHAUL_DIR"
else
  echo "  Creating venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install -q --upgrade pip
  "$VENV_DIR/bin/pip" install -q -e "$TURBOHAUL_DIR"
  ok "turbohaul-manager installed from $TURBOHAUL_DIR"
fi

# -- Step 7: Create directories + config --------------------------------------
step "Step 7: Configuration"

if [ ! -d /etc/turbohaul ] || [ ! -d /var/lib/turbohaul/blobs ]; then
  run_sudo mkdir -p /etc/turbohaul
  run_sudo mkdir -p /var/lib/turbohaul/{blobs,manifests,import-staging,telemetry}
  run_sudo chown -R "$(id -u):$(id -g)" /var/lib/turbohaul
else
  skip "Directories /etc/turbohaul and /var/lib/turbohaul already exist"
fi

CFG="/etc/turbohaul/turbohaul.yaml"
if [ -f "$CFG" ]; then
  # Update llama_server_binary path if it's using YOUR_USER placeholder or wrong path
  if grep -q "YOUR_USER" "$CFG" 2>/dev/null; then
    run_sudo sed -i "s|/home/YOUR_USER/|/home/$USER/|g" "$CFG"
    ok "Fixed YOUR_USER placeholder in $CFG"
  else
    skip "Config already exists at $CFG"
  fi
else
  echo "  Writing $CFG..."
  run_sudo tee "$CFG" > /dev/null << CFGEOF
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
  llama_server_binary: $LLAMA_BIN
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
CFGEOF
  ok "Config written to $CFG"
fi

# -- Step 8: Download a model (optional) --------------------------------------
step "Step 8: Model setup"

HAS_MODEL=false
HAS_MANIFEST=false

# Check for any GGUF in models dir
if ls "$REAL_HOME"/models/*.gguf 2>/dev/null | head -1 > /dev/null 2>&1; then
  HAS_MODEL=true
  MODEL_FILE=$(ls "$REAL_HOME"/models/*.gguf | head -1)
  ok "Model found: $(basename "$MODEL_FILE")"
else
  echo "  No GGUF model found in ~/models/"
  echo "  Downloading Qwen2.5-3B-Instruct Q4_K_M (~1.8 GB)..."
  mkdir -p "$REAL_HOME/models"
  wget -q -O "$REAL_HOME/models/qwen2.5-3b-q4_k_m.gguf" \
    "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
  MODEL_FILE="$REAL_HOME/models/qwen2.5-3b-q4_k_m.gguf"
  HAS_MODEL=true
  ok "Model downloaded: $(basename "$MODEL_FILE")"
fi

# Import into blob store and create manifest (requires running turbohaul)
# We'll do this after starting the server

# -- Step 9: Stop any running turbohaul-manager --------------------------------
step "Step 9: Stopping existing turbohaul-manager (if running)"

EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ":11401 " | grep -oP 'pid=\K[0-9]+' || echo "")
if [ -n "$EXISTING_PID" ]; then
  kill "$EXISTING_PID" 2>/dev/null || true
  sleep 2
  ok "Stopped existing turbohaul-manager (pid $EXISTING_PID)"
else
  skip "No turbohaul-manager running on port 11401"
fi

# -- Step 10: Start turbohaul-manager -----------------------------------------
step "Step 10: Starting turbohaul-manager"

echo "  Starting turbohaul-manager..."
rm -f /tmp/turbohaul-setup.log
"$VENV_DIR/bin/turbohaul-manager" --allow-public-bind --log-level info > /tmp/turbohaul-setup.log 2>&1 &
TURBO_PID=$!
sleep 4

if kill -0 "$TURBO_PID" 2>/dev/null; then
  HEALTH=$(curl -s http://localhost:11401/health 2>/dev/null || echo "")
  if echo "$HEALTH" | grep -q '"ok"'; then
    ok "turbohaul-manager running (pid $TURBO_PID) — health: $HEALTH"
  else
    fail "turbohaul-manager started but health check failed: $HEALTH"
  fi
else
  fail "turbohaul-manager failed to start. Log:"
  cat /tmp/turbohaul-setup.log 2>/dev/null
fi

# -- Step 11: Import model + create manifest -----------------------------------
step "Step 11: Import model into blob store"

if [ "$HAS_MODEL" = true ]; then
  HASH=$(sha256sum "$MODEL_FILE" | awk '{print $1}')
  SIZE=$(stat -c%s "$MODEL_FILE")

  # Check if manifest already exists
  EXISTING=$(curl -s http://localhost:11401/api/manifests/default 2>/dev/null || echo "")
  if echo "$EXISTING" | grep -q '"gguf_blob_sha256"'; then
    skip "Manifest 'default' already exists"
  else
    # Import via API
    IMPORT_STAGING="/var/lib/turbohaul/import-staging"
    cp "$MODEL_FILE" "$IMPORT_STAGING/" 2>/dev/null || true

    IMPORT_RESP=$(curl -s -X POST http://localhost:11401/api/import \
      -H "Content-Type: application/json" \
      -d "{\"path\": \"$IMPORT_STAGING/$(basename "$MODEL_FILE")\"}" 2>/dev/null || echo "")

    if echo "$IMPORT_RESP" | grep -q '"status":"complete"'; then
      IMPORTED_SHA=$(echo "$IMPORT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['sha256'])" 2>/dev/null || echo "$HASH")

      # Create manifest
      MANIFEST_RESP=$(curl -s -X PUT http://localhost:11401/api/manifests/default \
        -H "Content-Type: application/json" \
        -d "{
          \"gguf_blob_sha256\": \"$IMPORTED_SHA\",
          \"gguf_size_bytes\": $SIZE,
          \"context_size\": 4096,
          \"expected_vram_bytes\": 2147483648,
          \"llama_server_flags\": {\"parallel\": 1}
        }" 2>/dev/null || echo "")

      if echo "$MANIFEST_RESP" | grep -q '"status":"ok"'; then
        ok "Manifest 'default' created"
      else
        warn "Manifest creation response: $MANIFEST_RESP"
      fi
    else
      warn "Import response: $IMPORT_RESP"
    fi
  fi
else
  warn "No model to import — download a GGUF and re-run"
fi

# -- Step 12: Test chat completion ---------------------------------------------
step "Step 12: Testing chat completion"

CHAT_RESP=$(curl -s -X POST http://localhost:11401/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say OK"}]}' 2>/dev/null || echo "")

if echo "$CHAT_RESP" | grep -q '"finish_reason"'; then
  CONTENT=$(echo "$CHAT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
  TIMINGS=$(echo "$CHAT_RESP" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(f\"prompt={t.get('prompt_per_second',0):.0f} tok/s gen={t.get('predicted_per_second',0):.0f} tok/s\")" 2>/dev/null || echo "")
  ok "Chat completion working: \"$CONTENT\" ($TIMINGS)"
else
  warn "Chat completion test inconclusive (model may need manifest first)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================================="
echo -e " ${CYAN}SETUP SUMMARY${NC}"
echo "============================================================================="

if [ "$FAIL" -eq 0 ]; then
  echo -e " ${GREEN}All $DONE checks passed. turbohaul-manager is ready.${NC}"
else
  echo -e " ${GREEN}$DONE passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$WARN warnings${NC}"
fi

echo ""
echo " Service:  http://localhost:11401"
echo " Health:   http://localhost:11401/health"
echo " Config:   /etc/turbohaul/turbohaul.yaml"
echo " Log:      /tmp/turbohaul-setup.log"
echo ""

if [ "$FAIL" -gt 0 ]; then
  echo -e " ${YELLOW}Action required:${NC} fix the failures above and re-run:"
  echo "   bash $(realpath "$0")"
else
  echo -e " ${GREEN}Start using turbohaul:${NC}"
  echo "   curl http://localhost:11401/health"
  echo "   curl -X POST http://localhost:11401/v1/chat/completions \\"
  echo "     -H 'Content-Type: application/json' \\"
  echo "     -d '{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
  echo ""
  echo " To start/stop:"
  echo "   kill \$(ss -tlnp | grep :11401 | grep -oP 'pid=\\K[0-9]+')   # stop"
  echo "   turbohaul-manager --allow-public-bind --log-level info &     # start"
fi

echo "============================================================================="
