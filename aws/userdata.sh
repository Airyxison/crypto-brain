#!/usr/bin/env bash
# EC2 User Data script — runs automatically on first boot as root.
# Paste this into the "User data" field when launching the EC2 instance.
# Edit S3_BUCKET below before pasting.
# -----------------------------------------------------------------------
set -euo pipefail
exec > /var/log/nova-bootstrap.log 2>&1

S3_BUCKET="nova-trader-data-249899228939-us-east-1-an"          # <-- EDIT THIS to your bucket name
AWS_REGION="us-east-1"
GITHUB_REPO="https://github.com/Airyxison/crypto-brain.git"
UBUNTU_HOME="/home/ubuntu"

echo "=== Nova Trader EC2 Bootstrap ==="
echo "Started: $(date)"

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y git curl unzip

# ── Verify GPU ───────────────────────────────────────────────────────────────
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi not found"

# ── Clone repo ───────────────────────────────────────────────────────────────
echo "Cloning repo..."
sudo -u ubuntu git clone "$GITHUB_REPO" "$UBUNTU_HOME/crypto-brain"

# ── Install uv + Python deps ─────────────────────────────────────────────────
echo "Installing Python environment..."
sudo -u ubuntu bash -c "
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    cd $UBUNTU_HOME/crypto-brain
    uv venv .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
"

# ── Download tick data from S3 ───────────────────────────────────────────────
echo "Downloading tick data from S3..."
mkdir -p "$UBUNTU_HOME/crypto-engine"
aws s3 cp "s3://$S3_BUCKET/ticks/ticks.db" "$UBUNTU_HOME/crypto-engine/ticks.db" \
    --region "$AWS_REGION"

chown ubuntu:ubuntu "$UBUNTU_HOME/crypto-engine/ticks.db"

# ── Verify PyTorch sees the GPU ──────────────────────────────────────────────
echo "PyTorch GPU check:"
sudo -u ubuntu bash -c "
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    cd $UBUNTU_HOME/crypto-brain
    source .venv/bin/activate
    python3 -c \"
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
\"
"

# ── Write a convenience env file ─────────────────────────────────────────────
cat > "$UBUNTU_HOME/.nova_env" << EOF
export S3_BUCKET="$S3_BUCKET"
export AWS_REGION="$AWS_REGION"
export PATH="\$HOME/.local/bin:\$PATH"
EOF
chown ubuntu:ubuntu "$UBUNTU_HOME/.nova_env"
echo 'source ~/.nova_env' >> "$UBUNTU_HOME/.bashrc"

echo ""
echo "=== Bootstrap complete — ready to train ==="
echo "SSH in and run: cd ~/crypto-brain && bash aws/run_training.sh"
echo "Finished: $(date)"
