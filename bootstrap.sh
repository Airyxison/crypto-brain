#!/usr/bin/env bash
# Bootstrap Nova Brain dev environment
# Usage: bash bootstrap.sh
set -e

# Install uv if not already installed
if ! command -v uv &>/dev/null && ! [ -f "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

UV="${HOME}/.local/bin/uv"
if command -v uv &>/dev/null; then
  UV="uv"
fi

# Create venv and install deps
$UV venv .venv
$UV pip install -r requirements.txt

echo ""
echo "Bootstrap complete. Activate with: source .venv/bin/activate"
