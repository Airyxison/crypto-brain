# PRD-005 — Multi-Asset Portfolio Manager

**Status:** Future — requires PRD-003 (per-asset policies) as foundation  
**Priority:** P2 — Long-term vision, begin design now  
**Author:** Kiran  
**Date:** 2026-04-10

---

## Vision

Eric's goal: trade a dynamic basket of ~10 assets at a time, drawn from a universe of ~30 candidates, evaluating each asset against the others to determine the best current holds. Run the same framework in parallel across crypto (top 30 by market cap) and equities (Dow 30), with potential for additional buckets (sector ETFs, commodities, etc.).

This is not just "more assets" — it's a fundamentally different problem. The question shifts from:

> "Is BTC going to go up?"

to:

> "Among my current 30 candidates, which 10 deserve capital right now?"

That relative framing changes the architecture.

---

## The Core Idea: Relative Strength Tournament

Each candidate asset runs its per-asset policy (PRD-003) continuously, generating a real-time **quality score** — a rolling estimate of its expected Sortino ratio over the next N bars.

The portfolio manager holds the top-scoring 10. When a bench asset's score exceeds an active holding's score by a threshold margin, it rotates in. This is a tournament: assets compete for slots based on forward-looking quality, not just current price direction.

**Why this works better than absolute signals:**
- In a market-wide crash, all crypto drops. Relative strength is still informative — some assets hold better than others and are worth rotating into.
- In a bull run, relative strength identifies leaders before they move. You want the asset that's accelerating fastest, not the one that already moved.
- It naturally handles regime shifts: when crypto goes risk-off and equities stabilize, the equities bucket scores higher and capital rotates there (if cross-bucket is enabled).

---

## Architecture

### Layer 1 — Per-Asset Policies (PRD-003)

Already being built. Each asset has a trained SAC that manages position entry/exit for that asset in isolation. These run continuously in inference mode, generating:

- Current action recommendation
- Confidence signal (entropy of action distribution — low entropy = high conviction)
- Rolling Sortino estimate over a lookback window

### Layer 2 — Asset Scorer

A lightweight module that wraps each per-asset policy and produces a **composite score** per asset per bar:

```
score = f(rolling_sortino, momentum_rank, volatility_regime, policy_entropy)
```

Components:
- `rolling_sortino` — 30-bar rolling Sortino of the policy's simulated P&L on live data
- `momentum_rank` — percentile rank of recent return vs the full universe (relative, not absolute)
- `volatility_regime` — downweight assets in high-volatility regimes (drawdown risk)
- `policy_entropy` — how confident the per-asset policy is; low entropy = high conviction

> **Nova's idea:** The scorer doesn't need to be learned — a well-designed formula may outperform a learned ranker at this scale (30 assets), because the training signal for "which 10 assets to hold" is very sparse. Start with a formula, add a learned ranker only if the formula plateaus.

### Layer 3 — Portfolio Manager

Holds the active basket (~10 slots). On each bar:

1. Rank all 30 candidates by current score.
2. For each active holding: if its rank has dropped below threshold (e.g., rank 15+), flag for rotation.
3. For each empty/flagged slot: rotate in the highest-scoring bench asset.
4. Enforce rotation costs (don't churn for marginal score differences — require a minimum gap to rotate).

**Rotation throttle:** Minimum N bars between rotations per slot. Prevents thrashing when scores oscillate around the threshold.

> **Nova's idea:** Model rotation cost explicitly. Each rotation incurs fees + tax drag (short-term gain). A rotation that would earn 0.3% more Sortino but costs 0.5% in fees + taxes is a net negative. The portfolio manager should factor this in — especially important for the equity bucket where tax events are even more significant.

### Layer 4 — Bucket Manager (Optional, Future)

If running crypto + equities + other streams as separate buckets, a top-level allocator decides how much capital each bucket gets. This is essentially a macro regime detector:

- Crypto risk-on → overweight crypto bucket
- Equity earnings season → weight toward equities
- Both in drawdown → move to cash / stablecoins

> **Nova's idea — the regime bridge:** BTC dominance (BTC's share of total crypto market cap) is one of the best macro signals in crypto. When BTC dominance rises, altcoins are underperforming and capital is fleeing to safety. When dominance falls, alts are outperforming. This is a free, high-signal input for the bucket allocator that doesn't require any training.

---

## Nova's Injected Ideas

These go beyond Eric's stated vision but fit naturally:

### 1. The "Bench Warmup" Signal

Assets on the bench aren't idle — they're being evaluated continuously. If a bench asset has been scoring in the top 5 for 3+ consecutive bars but isn't active yet (because rotation throttle is still cooling down), flag it as a "warming up" candidate. This gives the portfolio manager early warning before a slot opens.

Analogy: like a coach watching a bench player warm up — you know the substitution is coming before it happens.

### 2. Cross-Asset Divergence as Alpha

BTC and ETH have ~0.85 correlation. When they diverge (e.g., ETH rallies while BTC is flat), that divergence is tradeable. The portfolio manager could track pairwise correlations across the universe and flag when two correlated assets diverge significantly — one of them is mispriced relative to the other.

This is a free alpha signal that the per-asset policies can't see (they're single-asset by design).

### 3. The "Staleness" Penalty

A checkpoint trained on 2025-2026 data will gradually degrade as market structure evolves. Rather than retraining on a fixed schedule, track each per-asset policy's **live vs backtest Sortino gap**. When live performance drops more than X% below backtest expectation, flag the policy as stale and queue a retrain on recent data.

This makes the system self-healing — it knows when its own models are getting outdated and asks to be updated.

### 4. Equity / Crypto Correlation as Risk Signal

When crypto and equities become *highly correlated* (both falling together), it signals a systemic risk-off event. During these periods, no asset class is a safe haven and position sizes should shrink across all buckets. The bucket manager can use a rolling crypto/equity correlation metric as a drawdown early-warning system.

---

## Buckets

### Crypto Bucket
- Universe: Top 30 by market cap (or a curated list)
- Starting universe: BTC, ETH, SOL, ADA (already in training pipeline)
- Data source: Coinbase / Binance public API (already integrated)
- Active slots: ~10

### Equity Bucket
- Universe: Dow 30 (or S&P 100 for broader coverage)
- Data source: Yahoo Finance / Alpaca / Polygon (separate pipeline needed)
- Active slots: ~10
- Key difference from crypto: market hours, earnings events, dividends, T+2 settlement

### Future Buckets
- Sector ETFs (tech, energy, healthcare)
- Commodities (gold, oil — futures-based, different mechanics)
- International indices

---

## Prerequisites

1. PRD-003 complete — per-asset policies for BTC, ETH, SOL, ADA meeting quality bar.
2. A working inference loop (PRD-004 paper trading) validated for at least one asset.
3. Historical data for equity universe (separate backfill pipeline for equities).

---

## What This Is NOT

- Not a high-frequency trading system. Still 1-minute resolution, same as current.
- Not a unified multi-asset SAC (training one network on 30 assets simultaneously). Per-asset policies remain specialized.
- Not fully autonomous capital allocation without human oversight. Eric reviews and approves any significant change to the active basket or capital allocation.

---

## Notes

- The crypto universe is more tractable to start: 24/7 markets, no earnings events, unified data pipeline already built.
- The equity bucket adds complexity (market hours, corporate events) but also diversification. Build crypto first, layer equities in after.
- Tax efficiency matters more at this scale. A 30-asset rotating portfolio with frequent churn could generate significant short-term capital gains. Worth modeling rotation cost explicitly before going live.
- This is the endgame architecture. Each PRD before this is a stepping stone. Don't shortcut the per-asset quality bar (PRD-003) to get here faster — a bad per-asset policy fed into the portfolio manager just amplifies losses.
