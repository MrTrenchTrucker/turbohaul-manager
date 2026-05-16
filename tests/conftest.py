"""Pytest fixtures shared across the test suite."""
import os
import tempfile
import pytest
from pathlib import Path


@pytest.fixture
def temp_state_dir():
    """Sandboxed /var/lib/turbohaul-equivalent for testing."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "blobs" / "sha256" / "incoming").mkdir(parents=True)
        (root / "manifests").mkdir(parents=True)
        (root / "import-staging").mkdir(parents=True)
        (root / "conf").mkdir(parents=True)
        yield root


@pytest.fixture
def temp_etc_config(temp_state_dir, tmp_path):
    """Default turbohaul.yaml in tempdir."""
    cfg_path = tmp_path / "turbohaul.yaml"
    cfg_path.write_text(f"""server:
  host: "127.0.0.1"
  port: 11401
  allow_public_bind: false
storage:
  blob_store_path: {temp_state_dir / "blobs"}
  manifests_path: {temp_state_dir / "manifests"}
  import_allowed_root: {temp_state_dir / "import-staging"}
  state_db_path: {temp_state_dir / "state.sqlite"}
runtime:
  llama_server_binary: /opt/turboquant/build/bin/llama-server
  llama_server_binary_sha256: ""
  default_port_base: 11500
ui:
  enabled: false
  static_path: /opt/turbohaul/ui_dist
queue:
  max_parallel_sidecars: 1
  staging_queue_depth: 100
  acceptance_buffer_max: 10000
  grace_seconds: 30
  idle_hot_load_seconds: 120
  max_grace_extensions: 5
  loading_health_timeout_s: 600
  drained_sigterm_window_active_s: 15
  drained_sigterm_window_cold_s: 5
pull:
  hf_api_key_env: HF_API_KEY
  hf_host_allowlist: ["huggingface.co", "hf.co", "cdn-lfs.huggingface.co"]
  pull_url_https_only: true
  pull_concurrency: 2
  pull_chunk_size_mb: 64
  per_stream_max_bytes: 107374182400
""")
    return cfg_path
