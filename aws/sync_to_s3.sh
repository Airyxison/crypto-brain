#!/usr/bin/env bash
# sync_to_s3.sh — Upload tick data from this machine to S3
# Run from crypto-brain/ directory: bash aws/sync_to_s3.sh
set -euo pipefail

source "$(dirname "$0")/config.env"

DB_PATH="../crypto-engine/ticks.db"

if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: ticks.db not found at $DB_PATH"
    echo "Make sure the Rust engine has collected some data first."
    exit 1
fi

SIZE=$(du -sh "$DB_PATH" | cut -f1)
echo "Uploading ticks.db ($SIZE) → s3://$S3_BUCKET/ticks/ticks.db"
aws s3 cp "$DB_PATH" "s3://$S3_BUCKET/ticks/ticks.db" \
    --region "$AWS_REGION"

echo "Done. Tick data available for EC2 training instances."
