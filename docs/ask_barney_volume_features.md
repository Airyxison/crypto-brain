# Question for Barney — Volume Confirmation Features

## Context

We're building a discrete SAC (Soft Actor-Critic) reinforcement learning trading agent
for BTC, ETH, SOL, and ADA. Training on historical tick data from Binance.

**Tick data schema** (1-minute aggregated trades):
- `symbol` — asset identifier
- `price` — trade price
- `quantity` — trade size
- `trade_time` — timestamp (ms)

**Current feature vector: 17 dimensions**
- Multi-timeframe momentum: 1m, 5m, 15m, 30m, 1d, 7d, 30d, 8h
- Volatility (1h realized)
- Volume norm (last tick vs recent average — centered at 0)
- VWAP deviation
- Position state features (in_position, pnl, hold time, stop distance)
- Trade frequency

## What We Found

Consistency testing across 20 random windows showed the agent enters on
**negative momentum_8h in 49/55 cases** — a real, consistent dip-buying signal.

The problem: it can't distinguish a dip with genuine selling pressure behind it
from a dip on thin, low-conviction tape. Both look identical in the current
feature set. The result: it enters on noise as often as it enters on real moves.

## What We Need

One or two features that give the agent **directional volume confirmation**
without order book data (we only have trade prints — no bid/ask spread).

Specifically:

1. **Is this move backed by volume?** — Is the current price move (up or down)
   happening on above-average participation, or thin air?

2. **Which direction is volume flowing?** — Over a rolling window, is more
   quantity trading on down-ticks or up-ticks? (buying pressure vs selling pressure)

## The Question

Given only `price`, `quantity`, and `trade_time` at 1-minute resolution:

- What's the most reliable approximation for directional volume flow without
  order book data? (tick rule? volume-weighted return sign? something else?)

- What window length would you recommend given we need this to work across
  assets with very different volatility profiles?
  - BTC: ~0.00041 avg move/bar
  - ETH: ~0.00060 avg move/bar
  - SOL: ~0.00079 avg move/bar
  - ADA: ~0.00087 avg move/bar (range-bound, structurally different)

- Should this be one feature (net directional flow) or two separate features
  (buying pressure + selling pressure independently)?

## What We're Trying to Teach the Agent

> A dip with volume = real fear, real sellers, potential conviction entry.
> A dip without volume = noise, thin tape, stay out.

We want the agent to discover this distinction through experience —
not hardcode it as a rule. So the feature needs to carry genuine signal,
not just correlate with price.

## Current File Locations

- Feature engineering: `features/engineer.py`
- Reward function: `environment/trading_env.py`
- Current STATE_DIM = 17 in `agent/networks.py`

Adding features requires a checkpoint migration (we have a script for that:
`scripts/migrate_checkpoint_16to17.py` — can be adapted for 17→19).

---
*From Kiran — Nova Trader project, 2026-04-25*
