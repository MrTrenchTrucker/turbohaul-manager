#!/bin/bash
# TurboHaul-Manager macOS launcher
#
# This script provides a simple way to run TurboHaul on macOS (Apple Silicon).
# It handles:
#   1. Self-execution detection (copy to ~/bin/turbohaul)
#   2. Argument forwarding (--config, --allow-public-bind, --log-level)
#   3. Default config path: ~/.turbohaul/turbohaul.yaml
#
# Usage:
#   brew install mlx-lm  # MLX backend (optional, for MLX models)
#   pip install -e ".[mlx]"  # Or install via pip if using local checkout
#   ./turbohaul-launcher.sh
#
# Or copy to ~/bin:
#   cp turbohaul-launcher.sh ~/bin/turbohaul
#   chmod +x ~/bin/turbohaul
#   turbohaul --config ~/.turbohaul/turbohaul.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Default config path (can be overridden via --config)
DEFAULT_CONFIG="${HOME}/.turbohaul/turbohaul.yaml"

# Show usage if no args and help requested
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "TurboHaul-Manager macOS Launcher"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config PATH       Path to turbohaul.yaml (default: ~/.turbohaul/turbohaul.yaml)"
    echo "  --allow-public-bind Bind to 0.0.0.0 instead of 127.0.0.1"
    echo "  --log-level LEVEL   Log level: debug, info, warning, error, critical"
    echo "  --help, -h         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --config /etc/turbohaul/turbohaul.yaml"
    echo "  $0 --allow-public-bind"
    exit 0
fi

# Set PYTHONPATH if needed (for local development)
if [[ -d "${PROJECT_ROOT}/src" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
fi

# Build args
ARGS=("-m" "turbohaul.__main__")

# Check if --config was provided
CONFIG_PROVIDED=false
for ((i=0; i<${#@}; i++)); do
    if [[ "${!i}" == "--config" ]]; then
        CONFIG_PROVIDED=true
        break
    fi
done

# Add default config if not provided
if [[ "$CONFIG_PROVIDED" == "false" ]]; then
    ARGS+=("--config" "$DEFAULT_CONFIG")
fi

# Pass through all user args
for arg in "$@"; do
    ARGS+=("$arg")
done

# Run turbohaul
exec python "${ARGS[@]}"
