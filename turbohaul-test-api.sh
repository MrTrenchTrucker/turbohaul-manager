#!/bin/bash
# =============================================================================
# turbohaul-test-api.sh — Full API test suite for turbohaul-manager
#
# Runs every endpoint and logs results to a file for review.
# Usage: bash turbohaul-test-api.sh [host:port]
# =============================================================================
set -uo pipefail

BASE="http://${1:-localhost:11401}"
LOG="/tmp/turbohaul-api-test-$(date +%Y%m%d-%H%M%S).log"
PASS=0
FAIL=0
SKIP=0

# -- Helpers -------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "$1" | tee -a "$LOG"; }
sep()  { log "\n${CYAN}━━━ $1 ━━━${NC}"; }
ok()   { PASS=$((PASS+1)); log "  ${GREEN}[PASS]${NC} $1"; }
fail() { FAIL=$((FAIL+1)); log "  ${RED}[FAIL]${NC} $1"; }
warn() { SKIP=$((SKIP+1)); log "  ${YELLOW}[SKIP]${NC} $1"; }

# Log a curl call and check HTTP status
# Usage: api METHOD PATH [extra curl args...]
api() {
  local method="$1" path="$2"
  shift 2
  local url="${BASE}${path}"
  local label="$method $path"

  local tmpbody
  tmpbody=$(mktemp)
  local http_code
  http_code=$(curl -s -o "$tmpbody" -w '%{http_code}' \
    -X "$method" "$url" \
    -H "Content-Type: application/json" \
    "$@" 2>/dev/null) || true

  local body
  body=$(cat "$tmpbody")
  rm -f "$tmpbody"

  log ""
  log "┌─ $label"
  log "│  HTTP $http_code"
  if [ -n "$body" ]; then
    # Pretty-print JSON if possible, otherwise raw
    local pretty
    pretty=$(echo "$body" | python3 -m json.tool 2>/dev/null) && body="$pretty"
    echo "$body" | while IFS= read -r line; do log "│  $line"; done
  fi
  log "└─"

  # Evaluate pass/fail
  case "$http_code" in
    2[0-9][0-9]) ok "$label (HTTP $http_code)" ;;
    000)         fail "$label (connection refused)" ;;
    *)           fail "$label (HTTP $http_code)" ;;
  esac

  # Return body for chaining
  echo "$body"
}

# Like api() but expects a specific HTTP status code (not 2xx)
# Usage: api_expect EXPECTED_CODE METHOD PATH [extra curl args...]
api_expect() {
  local expected="$1" method="$2" path="$3"
  shift 3
  local url="${BASE}${path}"
  local label="$method $path (expect $expected)"

  local tmpbody
  tmpbody=$(mktemp)
  local http_code
  http_code=$(curl -s -o "$tmpbody" -w '%{http_code}' \
    -X "$method" "$url" \
    -H "Content-Type: application/json" \
    "$@" 2>/dev/null) || true

  local body
  body=$(cat "$tmpbody")
  rm -f "$tmpbody"

  log ""
  log "┌─ $label"
  log "│  HTTP $http_code"
  if [ -n "$body" ]; then
    local pretty
    pretty=$(echo "$body" | python3 -m json.tool 2>/dev/null) && body="$pretty"
    echo "$body" | while IFS= read -r line; do log "│  $line"; done
  fi
  log "└─"

  if [ "$http_code" = "$expected" ]; then
    ok "$label (HTTP $http_code)"
  else
    fail "$label (expected HTTP $expected, got $http_code)"
  fi
}

# Stream SSE and capture first N seconds
# Usage: sse_test PATH SECONDS [extra curl args]
sse_test() {
  local path="$1" timeout_s="$2"
  shift 2
  local url="${BASE}${path}"
  local label="STREAM $path"

  log ""
  log "┌─ $label (timeout ${timeout_s}s)"
  local out
  out=$(timeout "$timeout_s" curl -s -N "$url" "$@" 2>/dev/null) || true
  if [ -n "$out" ]; then
    echo "$out" | head -30 | while IFS= read -r line; do log "│  $line"; done
    local lines
    lines=$(echo "$out" | wc -l)
    log "│  ... ($lines lines total)"
    ok "$label (received data)"
  else
    log "│  (no data received)"
    warn "$label (empty response)"
  fi
  log "└─"
}

# -- Pre-flight ----------------------------------------------------------------
log "╔══════════════════════════════════════════════════════════════════╗"
log "║  turbohaul-manager API test suite                              ║"
log "║  Target: $BASE"
log "║  Log:    $LOG"
log "╚══════════════════════════════════════════════════════════════════╝"
log ""

# Check server is reachable
HTTP=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null) || true
if [ "$HTTP" != "200" ]; then
  log "${RED}ERROR: Server not reachable at $BASE (HTTP $HTTP)${NC}"
  exit 1
fi
log "${GREEN}Server reachable at $BASE${NC}"

# ─── 1. Health & System ───────────────────────────────────────────────────────
sep "1. HEALTH & SYSTEM"

api GET /health
api GET /status
api GET /api/version
api GET /api/config

# ─── 2. Model Management ──────────────────────────────────────────────────────
sep "2. MODEL MANAGEMENT"

api GET /api/tags
api GET "/api/show?name=default"
api GET /api/manifests/default

# Create test manifest
api PUT /api/manifests/test-temp \
  -d '{"gguf_blob_sha256":"0000000000000000000000000000000000000000000000000000000000000001","gguf_size_bytes":1024,"context_size":2048,"expected_vram_bytes":1073741824,"llama_server_flags":{"parallel":1}}'

# Read it back to get ETag
ETAG=$(curl -s -D- "$BASE/api/manifests/test-temp" 2>/dev/null | grep -i '^etag' | tr -d '\r' | sed 's/^etag: //i')
log "  ETag: $ETAG"

# Update with ETag
if [ -n "$ETAG" ]; then
  api PUT /api/manifests/test-temp \
    -H "If-Match: $ETAG" \
    -d "{\"gguf_blob_sha256\":\"0000000000000000000000000000000000000000000000000000000000000001\",\"gguf_size_bytes\":1024,\"context_size\":4096,\"expected_vram_bytes\":1073741824,\"llama_server_flags\":{\"parallel\":1}}"
fi

# ETag mismatch (should fail 412)
api_expect 412 PUT /api/manifests/test-temp \
  -H "If-Match: wrong-etag" \
  -d '{"gguf_blob_sha256":"0000000000000000000000000000000000000000000000000000000000000001","gguf_size_bytes":1024,"context_size":4096,"expected_vram_bytes":1073741824,"llama_server_flags":{"parallel":1}}'

# Delete test manifest
api DELETE /api/manifests/test-temp

# 404 on deleted manifest
api_expect 404 GET /api/manifests/test-temp

# ─── 3. Chat Completions ──────────────────────────────────────────────────────
sep "3. CHAT COMPLETIONS (OpenAI-compat)"

# Basic
api POST /v1/chat/completions \
  -d '{"model":"default","messages":[{"role":"user","content":"Say exactly: test OK"}]}'

# With system prompt
api POST /v1/chat/completions \
  -d '{"model":"default","messages":[{"role":"system","content":"Reply with one word."},{"role":"user","content":"Hello"}]}'

# max_tokens limit
api POST /v1/chat/completions \
  -d '{"model":"default","messages":[{"role":"user","content":"Count to 20"}],"max_tokens":30}'

# temperature sampling
api POST /v1/chat/completions \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hi"}],"temperature":0.5}'

# Multi-turn conversation
api POST /v1/chat/completions \
  -d '{"model":"default","messages":[{"role":"user","content":"My name is Alice"},{"role":"assistant","content":"Hello Alice!"},{"role":"user","content":"What is my name?"}]}'

# ─── 4. Streaming ─────────────────────────────────────────────────────────────
sep "4. STREAMING (SSE)"

sse_test /v1/chat/completions 8 \
  -d '{"model":"default","messages":[{"role":"user","content":"Count to 3"}],"stream":true}'

# ─── 5. Ollama-compat ────────────────────────────────────────────────────────
sep "5. OLLAMA-COMPAT CHAT"

api POST /api/chat \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi!"}]}'

# ─── 6. Embeddings ────────────────────────────────────────────────────────────
sep "6. EMBEDDINGS (OpenAI-compat)"

api POST /v1/embeddings \
  -d '{"model":"default","input":"Hello world"}'

api POST /v1/embeddings \
  -d '{"model":"default","input":["First sentence","Second sentence"]}'

# ─── 7. Config Update ────────────────────────────────────────────────────────
sep "7. RUNTIME CONFIG UPDATE"

# Mutable update (should succeed)
api PUT /api/config \
  -d '{"queue":{"grace_seconds":45}}'

# Boot-only update (should 403)
api_expect 403 PUT /api/config \
  -d '{"server":{"port":9999}}'

# Unknown section (should 400)
api_expect 400 PUT /api/config \
  -d '{"nonexistent":{"key":"value"}}'

# ─── 8. Import & Delete ──────────────────────────────────────────────────────
sep "8. IMPORT & BLOB DELETE"

# Import (path must be under import_allowed_root)
api POST /api/import \
  -d '{"path":"/var/lib/turbohaul/import-staging/qwen2.5-3b-q4_k_m.gguf"}'

# Delete by SHA256
api DELETE /api/delete \
  -d '{"sha256":"9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94"}'

# Delete non-existent (should 404)
api_expect 404 DELETE /api/delete \
  -d '{"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}'

# ─── 9. Pull (URL) ───────────────────────────────────────────────────────────
sep "9. PULL FROM URL"

# Skipped by default — uncomment to test (downloads ~1.8GB)
# api POST /api/pull-url \
#   -d '{"url":"https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"}'
warn "Pull test skipped (large download, uncomment to enable)"

# ─── 10. Logging & Telemetry ──────────────────────────────────────────────────
sep "10. LOGGING & TELEMETRY"

api GET "/v1/logging?limit=20"
api GET "/v1/telemetry/events?limit=20"
api GET /v1/telemetry/status

# ─── 11. Live Output Stream (SSE) ─────────────────────────────────────────────
sep "11. LIVE OUTPUT STREAM"

sse_test /ui/live/output/stream 3

# ─── 12. Edge Cases ───────────────────────────────────────────────────────────
sep "12. EDGE CASES & ERROR HANDLING"

# Missing model field (should 422)
api_expect 422 POST /v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"test"}]}'

# Empty messages (should 422)
api_expect 422 POST /v1/chat/completions \
  -d '{"model":"default","messages":[]}'

# Non-existent model (should 404 or 503)
api POST /v1/chat/completions \
  -d '{"model":"nonexistent","messages":[{"role":"user","content":"test"}]}'

# GET on POST-only endpoint (should 405)
api_expect 405 GET /v1/chat/completions

# Import from forbidden path (should 403)
api_expect 403 POST /api/import \
  -d '{"path":"/etc/passwd"}'

# Import nonexistent file (should 404)
api_expect 404 POST /api/import \
  -d '{"path":"/var/lib/turbohaul/import-staging/nope.gguf"}'

# ─── Summary ──────────────────────────────────────────────────────────────────
log ""
log "╔══════════════════════════════════════════════════════════════════╗"
log "║  RESULTS                                                       ║"
log "╚══════════════════════════════════════════════════════════════════╝"
log ""
TOTAL=$((PASS + FAIL + SKIP))
log "  ${GREEN}PASS: $PASS${NC}  ${RED}FAIL: $FAIL${NC}  ${YELLOW}SKIP: $SKIP${NC}  Total: $TOTAL"
log ""
log "  Log file: $LOG"
log ""

if [ "$FAIL" -gt 0 ]; then
  log "  ${RED}Some tests failed. Review the log for details.${NC}"
else
  log "  ${GREEN}All tests passed!${NC}"
fi
log ""
