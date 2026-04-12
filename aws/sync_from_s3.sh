#!/usr/bin/env bash
# sync_from_s3.sh — Download trained checkpoints from S3 to this machine
# Run from crypto-brain/ directory: bash aws/sync_from_s3.sh
set -euo pipefail

source "$(dirname "$0")/config.env"

echo "Syncing checkpoints from s3://$S3_BUCKET/checkpoints/ → ./checkpoints/"
aws s3 sync "s3://$S3_BUCKET/checkpoints/" "./checkpoints/" \
    --region "$AWS_REGION" \
    --progress

echo ""
echo "Downloaded checkpoints:"
find checkpoints/ -name "nova_brain_best.pt" -o -name "nova_brain_final.pt" 2>/dev/null \
    | sort | xargs ls -lh 2>/dev/null || echo "  (none found)"
