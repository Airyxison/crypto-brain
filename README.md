# crypto-brain — Nova Trader Position Management AI

SAC-based position management agent. Places limit orders, manages dynamic stops, realizes gains. Trains on live tick data collected by the Rust engine.

## Architecture

```
Tick Data (SQLite from crypto-engine)
         ↓
  Feature Engineering          13-dim state vector, multi-timeframe
         ↓
  Trading Environment          Gym-compatible, limit order simulation
         ↓
  SAC Agent                    Soft Actor-Critic, entropy-regularized
         ↓
  Reward: Sortino-influenced    asymmetric downside punishment
```

## Setup

```bash
pip install -r requirements.txt
```

## Collect Data First

The Rust engine must run long enough to populate the SQLite database (~15+ minutes for the feature window to fill):

```bash
cd ../crypto-engine && cargo run
```

## Train

```bash
python train.py --db ../crypto-engine/ticks.db --steps 50000
```

Options:
- `--steps`      Total training steps (default: 50,000)
- `--save-every` Checkpoint interval (default: 5,000)
- `--resume`     Resume from checkpoint path

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/nova_brain_best.pt --db ../crypto-engine/ticks.db
```

Outputs `backtest_report.png` and `backtest_report.json`.

## Action Space

| Action | Behavior |
|--------|----------|
| HOLD (0) | Do nothing |
| BUY_LIMIT (1) | Place limit buy at current_price × 0.998 |
| ADJUST_STOP (2) | Trail stop-loss upward to lock in gains |
| REALIZE_GAIN (3) | Close position at market |
| CANCEL_ORDER (4) | Cancel unfilled limit order |

## Reward Function

```
R = portfolio_return
    - 0.5 × max(0, drawdown - 2%)   # asymmetric, 2% free zone
    - 0.10 × stop_loss_hit           # capital preservation hard constraint
    - 0.0001 × losing_hold_cost      # discourage holding underwater
```

## Roadmap

- [ ] Multi-asset support (BTC + equities + FOREX)
- [ ] Continuous action space (variable position sizing)
- [ ] Prioritized experience replay
- [ ] Market regime detector (trending / ranging / volatile)
- [ ] Live execution integration with crypto-engine
