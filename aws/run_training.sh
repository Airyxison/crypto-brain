#!/usr/bin/env bash
# run_training.sh — Start a full training queue on the EC2 instance.
# Run from ~/crypto-brain on the EC2 instance after bootstrap completes.
#
# Features:
#   - Pulls latest code from GitHub before training
#   - Runs all queued jobs sequentially via queue_runner.py
#   - Syncs checkpoints to S3 after each job completes
#   - Handles spot interruption: uploads checkpoints before shutdown
#   - On relaunch, resumes automatically from last S3 checkpoint
#
# Usage:
#   bash aws/run_training.sh             # run full queue
#   bash aws/run_training.sh --dry-run   # preview jobs without training
# -----------------------------------------------------------------------
set -euo pipefail

source ~/.nova_env 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_DIR/.venv/bin/activate"
DB_PATH="$HOME/crypto-engine/ticks.db"

cd "$REPO_DIR"
source "$VENV"

echo "=== Nova Trainer ==="
echo "Host    : $(hostname)"
echo "GPU     : $(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')")"
echo "Branch  : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "DB size : $(du -sh $DB_PATH 2>/dev/null | cut -f1 || echo 'not found')"
echo ""

# ── Pull latest code ──────────────────────────────────────────────────────────
echo "[1/4] Pulling latest code..."
git pull origin master

# ── Restore checkpoints from S3 (enables auto-resume on relaunch) ─────────────
echo "[2/4] Restoring checkpoints from S3..."
aws s3 sync "s3://$S3_BUCKET/checkpoints/" "./checkpoints/" \
    --region "$AWS_REGION" --quiet \
    && echo "  Checkpoints restored." \
    || echo "  No prior checkpoints in S3 — starting fresh."

# ── Spot interruption handler ─────────────────────────────────────────────────
# AWS sends SIGTERM 2 minutes before reclaiming a spot instance.
# We catch it, upload checkpoints, then allow the process to exit cleanly.
_spot_handler() {
    echo ""
    echo "⚠  SPOT INTERRUPTION SIGNAL — uploading checkpoints before shutdown..."
    aws s3 sync "./checkpoints/" "s3://$S3_BUCKET/checkpoints/" \
        --region "$AWS_REGION" --quiet
    echo "Checkpoints saved to S3. Instance will stop shortly."
    exit 0
}
trap '_spot_handler' SIGTERM SIGINT

# ── Run training queue ────────────────────────────────────────────────────────
echo "[3/4] Starting training queue..."
echo ""

# Run queue_runner in background so the trap fires during training
python queue_runner.py "$@" &
QUEUE_PID=$!

# Checkpoint sync loop: push to S3 every 5 minutes while training
while kill -0 "$QUEUE_PID" 2>/dev/null; do
    sleep 300
    if kill -0 "$QUEUE_PID" 2>/dev/null; then
        echo "[sync] Pushing checkpoints to S3..."
        aws s3 sync "./checkpoints/" "s3://$S3_BUCKET/checkpoints/" \
            --region "$AWS_REGION" --quiet
    fi
done

wait "$QUEUE_PID"
QUEUE_EXIT=$?

# ── Final sync ────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Final checkpoint sync to S3..."
aws s3 sync "./checkpoints/" "s3://$S3_BUCKET/checkpoints/" \
    --region "$AWS_REGION" \
    && echo "All checkpoints uploaded."

# Print S3 summary
echo ""
echo "=== Checkpoints in S3 ==="
aws s3 ls "s3://$S3_BUCKET/checkpoints/" --recursive --human-readable \
    | grep "nova_brain_best\|nova_brain_final" | sort

echo ""
echo "Training complete. Exit code: $QUEUE_EXIT"
echo "Download checkpoints to your local machine: bash aws/sync_from_s3.sh"
