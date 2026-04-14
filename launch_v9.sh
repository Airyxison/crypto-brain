#!/bin/bash
# launch_v9.sh — Kill v8, pull latest code, start v9 training for all 4 symbols
#
# v9 key change: auto-alpha tuning (SAC adaptive temperature)
# No --alpha flags needed — the agent self-tunes to target_entropy ≈ 1.58 nats
# Episode length doubled: 500 → 1000 steps

set -e
cd /root/crypto-brain

echo "=== Stopping any running train.py processes ==="
pkill -f train.py || echo "(none running)"
sleep 3

echo "=== Pulling latest code ==="
git pull origin master

echo "=== Creating log and checkpoint dirs ==="
mkdir -p logs
mkdir -p checkpoints/{btcusdt,ethusdt,solusdt,adausdt}

DB=/root/crypto-engine/ticks.db
STEPS=500000
SAVE_EVERY=10000

echo "=== Launching v9 training ==="

nohup python3 /root/crypto-brain/train.py \
    --symbol BTCUSDT \
    --db $DB \
    --steps $STEPS \
    --save-dir /root/crypto-brain/checkpoints \
    --save-every $SAVE_EVERY \
    --episode-steps 1000 \
    > /root/crypto-brain/logs/btc_v9.log 2>&1 &
echo "BTC pid=$!"

nohup python3 /root/crypto-brain/train.py \
    --symbol ETHUSDT \
    --db $DB \
    --steps $STEPS \
    --save-dir /root/crypto-brain/checkpoints \
    --save-every $SAVE_EVERY \
    --episode-steps 1000 \
    > /root/crypto-brain/logs/eth_v9.log 2>&1 &
echo "ETH pid=$!"

nohup python3 /root/crypto-brain/train.py \
    --symbol SOLUSDT \
    --db $DB \
    --steps $STEPS \
    --save-dir /root/crypto-brain/checkpoints \
    --save-every $SAVE_EVERY \
    --episode-steps 1000 \
    > /root/crypto-brain/logs/sol_v9.log 2>&1 &
echo "SOL pid=$!"

nohup python3 /root/crypto-brain/train.py \
    --symbol ADAUSDT \
    --db $DB \
    --steps $STEPS \
    --save-dir /root/crypto-brain/checkpoints \
    --save-every $SAVE_EVERY \
    --episode-steps 1000 \
    > /root/crypto-brain/logs/ada_v9.log 2>&1 &
echo "ADA pid=$!"

echo ""
echo "=== v9 launched. Monitor with: ==="
echo "  tail -f /root/crypto-brain/logs/btc_v9.log"
echo "  ps aux | grep train.py | grep -v grep"
echo ""
echo "=== W&B: https://wandb.ai/enlnetsol/nova-brain ==="
