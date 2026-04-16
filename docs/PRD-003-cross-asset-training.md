# PRD-003 — Cross-Asset Training: ETH-USD, SOL-USD, and ADA-USD

**Status:** ETH/SOL training in progress; ADA backfill in progress  
**Priority:** P1 — After PRD-001 and PRD-002  
**Author:** Kiran  
**Date:** 2026-04-10

---

## Problem

The current production policy was trained exclusively on BTC-USD tick data (525,200 ticks, 2025-04-10 to 2026-04-10). A policy trained on a single asset:

1. May not generalize to other market structures (ETH and SOL have different volatility profiles, liquidity, and correlation patterns).
2. Creates concentration risk — if the BTC policy degrades in a regime shift, there is no fallback.
3. Limits the system to trading only one asset, which caps the earning potential.
4. Only exposes the agent to trending/momentum regimes — range-bound behavior is not represented.

ETH-USD and SOL-USD backfills are complete (~525k ticks each). ADA-USD backfill is in progress (kicked off 2026-04-10). ADA was selected over DOGE because it exhibits range-bound, mean-reverting price behavior — a regime the model does not currently see in BTC/ETH/SOL training data. This diversity is expected to improve stop discipline and reduce drawdown.

---

## Goal

Train and evaluate the SAC agent on ETH-USD, SOL-USD, and ADA-USD. Produce per-asset checkpoints that meet the same quality bar as BTC (Sortino > 1.5, Return > +5%, Win Rate > 50%, Max Drawdown < -15%).

---

## Prerequisites

1. PRD-001 (best_sortino bug fix) implemented.
2. PRD-002 (drawdown reduction) completed for BTC — establishes the reward config that cross-asset training will inherit.
3. ETH-USD and SOL-USD backfills complete ✓. ADA-USD backfill in progress.

### Verify Backfill Completion

```bash
sqlite3 ../crypto-engine/ticks.db "SELECT symbol, COUNT(*) FROM ticks GROUP BY symbol;"
```

Expected output when ready:
```
BTCUSDT|525200
ETHUSDT|525000
SOLUSDT|525000
ADAUSDT|525000
```

---

## Architecture Changes Required

The current `train.py` accepts `--symbol` but the feature engineering and reward calibration were designed around BTC price scales. ETH and SOL have very different absolute price levels, which could affect:

1. **Feature scaling:** `price_vs_range` (feature [0]) is already normalized to 4h range, so this is safe. `vwap_deviation` is ratio-based, also safe. All 13 features are ratio-normalized — price scale should not matter.

2. **Stop loss distance (`DEFAULT_STOP_PCT = 0.01`):** BTC at ~$85k means 1% = ~$850. ETH at ~$2k means 1% = ~$20. SOL at ~$130 means 1% = ~$1.30. The stop percentage is price-agnostic (it is already a fraction), so no change needed here.

3. **Capital fraction:** `capital_fraction = 0.9` is ratio-based — safe.

4. **`load_ticks_from_db` symbol parameter:** Already supports arbitrary symbols. Pass `--symbol ETHUSDT` or `--symbol SOLUSDT`.

The architecture is already mostly asset-agnostic. The main question is whether the BTC-trained policy transfers, or whether fresh training is needed per asset.

---

## Training Strategy

### Option A — Per-Asset Fresh Training (Recommended)

Train a separate agent from scratch for each asset. This avoids negative transfer between assets with different volatility profiles.

```bash
# ETH
python train.py --db ../crypto-engine/ticks.db --symbol ETHUSDT --steps 200000

# SOL
python train.py --db ../crypto-engine/ticks.db --symbol SOLUSDT --steps 200000
```

**Pros:** Each policy is fully specialized. No cross-contamination.  
**Cons:** Requires 3x the training compute and time.

### Option B — Transfer Learning from BTC Checkpoint

Resume training on ETH/SOL data starting from the BTC checkpoint. The actor and critic networks already learned useful patterns from BTC momentum; ETH/SOL share macro-correlation with BTC and may converge faster.

```bash
python train.py --db ../crypto-engine/ticks.db --symbol ETHUSDT \
    --resume checkpoints/nova_brain_btc_best.pt --steps 100000
```

**Pros:** Potentially faster convergence (shared macro patterns). Smaller training budget needed.  
**Cons:** If ETH/SOL structure differs significantly from BTC, the starting point may bias the policy in unhelpful ways, adding noise to early training.

### Recommendation

Start with Option B for ETH (higher BTC correlation, ~0.85). Use Option A for SOL (lower correlation, more idiosyncratic behavior). Compare convergence speed and final metrics.

---

## Checkpoint Naming Convention

To avoid confusion with BTC checkpoints, use asset-prefixed names:

| Checkpoint | Description |
|-----------|-------------|
| `nova_brain_btc_best.pt` | Current BTC production policy |
| `nova_brain_eth_best.pt` | Best ETH policy |
| `nova_brain_sol_best.pt` | Best SOL policy |
| `nova_brain_ada_best.pt` | Best ADA policy |
| `nova_brain_eth_YYYYMMDD_sortino{X}.pt` | Permanent ETH backup |
| `nova_brain_sol_YYYYMMDD_sortino{X}.pt` | Permanent SOL backup |
| `nova_brain_ada_YYYYMMDD_sortino{X}.pt` | Permanent ADA backup |

Rename the current BTC checkpoint to `nova_brain_btc_best.pt` before starting ETH/SOL runs.

---

## Code Changes Required

### train.py

1. Update `save_dir` naming to include the symbol, or pass a `--save-dir` argument per symbol to avoid checkpoint collisions:
   ```python
   # Suggested: auto-prefix save dir with symbol
   save_dir = Path(args.save_dir) / args.symbol.lower()
   save_dir.mkdir(parents=True, exist_ok=True)
   ```

2. Log the symbol prominently at startup:
   ```python
   print(f"[TRAIN] Asset: {args.symbol}")
   ```

### checkpoints/ directory structure (suggested)

```
checkpoints/
  btcusdt/
    nova_brain_best.pt
    nova_brain_iter4_100k_sortino1.73.pt
    ...
  ethusdt/
    nova_brain_best.pt
    ...
  solusdt/
    nova_brain_best.pt
    ...
```

---

## Evaluation Plan

For each asset, run the standard backtest against the held-out 20% test split:

```bash
python evaluate.py --checkpoint checkpoints/ethusdt/nova_brain_best.pt \
    --db ../crypto-engine/ticks.db --symbol ETHUSDT
```

Report the full metrics table for each asset. The same acceptance criteria apply across all assets.

---

## Acceptance Criteria

For each of ETH-USD, SOL-USD, and ADA-USD:

1. A trained checkpoint exists that achieves Sortino > 1.5 on the held-out test split.
2. Max drawdown < -15%.
3. Total return > +5%.
4. Win rate > 50%.
5. Checkpoint is saved with a permanent timestamped backup.
6. BTC training is not disrupted — separate save dirs prevent cross-asset checkpoint collisions.

---

## Notes

- Do not begin cross-asset training until the ETH/SOL backfills are verified complete. Training on partial data would produce a biased policy that performs well on the available period but has degraded generalization.
- ETH and SOL have historically higher volatility than BTC on a percentage basis. The drawdown target (-15%) may require tighter stops or lower capital fraction for these assets. Inherit the reward config from PRD-002 as the starting point but be prepared to tune per-asset.
- ADA exhibits range-bound, mean-reverting behavior — different from the trending/momentum regime of BTC/ETH/SOL. Expect the policy to converge toward patience-rewarding strategies (smaller moves, tighter exits). The 0.5% stop from PRD-002 may actually suit ADA well given its lower volatility.
- ADA was chosen over DOGE: less volatile, less susceptible to small-dollar liquidity issues, and its range behavior provides genuinely different training signal. DOGE remains a future candidate if liquidity modeling is added.
- The longer-term goal is multi-asset simultaneous trading from crypto-engine. Per-asset policies are the foundation; a unified multi-asset policy is a future milestone beyond this PRD's scope.
