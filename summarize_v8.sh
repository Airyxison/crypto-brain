#!/bin/bash
# summarize_v8.sh — Run backtest on v8 best checkpoints for all 4 symbols
# Run this after v8 completes, before launching v9

cd /root/crypto-brain

DB=/root/crypto-engine/ticks.db

for SYM in BTCUSDT ETHUSDT SOLUSDT ADAUSDT; do
    sym=$(echo $SYM | tr '[:upper:]' '[:lower:]')

    # Find best checkpoint (prefer /root/crypto-brain, fallback to /root)
    BEST=""
    for dir in "/root/crypto-brain/checkpoints/$sym" "/root/checkpoints/$sym"; do
        if [ -f "$dir/nova_brain_best.pt" ]; then
            BEST="$dir/nova_brain_best.pt"
            break
        fi
    done

    if [ -z "$BEST" ]; then
        # Fall back to latest step checkpoint
        BEST=$(find /root/crypto-brain/checkpoints/$sym /root/checkpoints/$sym \
               -name 'nova_brain_step*.pt' 2>/dev/null | sort -V | tail -1)
    fi

    if [ -z "$BEST" ]; then
        echo "[$SYM] No checkpoint found — skipping"
        continue
    fi

    echo ""
    echo "========== $SYM =========="
    echo "Checkpoint: $BEST"
    python3 evaluate.py \
        --checkpoint "$BEST" \
        --symbol "$SYM" \
        --db "$DB" \
        --output "reports/v8_${sym}_backtest.png" \
        2>&1 | grep -E "Return|Sortino|Drawdown|Win Rate|Trades|Actions|BACKTEST"
done

echo ""
echo "Reports written to reports/v8_*.png and reports/v8_*.json"
