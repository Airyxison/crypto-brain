# PRD-002 — Drawdown Reduction: Target Max Drawdown < -15%

**Status:** Ready — pending PRD-001 completion  
**Priority:** P1 — Required before live capital deployment  
**Author:** Nova  
**Date:** 2026-04-10

---

## Problem

The production policy (`nova_brain_iter4_100k_sortino1.73.pt`) has a max drawdown of **-23.00%**, which exceeds the target of **-15%**. Eric's stated requirement before trading real capital is drawdown under -15%. The system currently has positive Sortino (1.7294) and positive return (+16.40%), but the drawdown disqualifies it from live use.

### Current Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Max Drawdown | -23.00% | < -15% |
| Sortino | +1.7294 | > 1.5 |
| Total Return | +16.40% | > 5% |
| Win Rate | 58.5% | > 50% |

All metrics pass except drawdown. The drawdown problem is the only remaining blocker for live deployment.

---

## Goal

Reduce max drawdown to < -15% without sacrificing Sortino below 1.5 or return below +5%.

---

## Root Cause Analysis

The drawdown is caused by a combination of factors:

1. **Stop distance is 1% (`DEFAULT_STOP_PCT = 0.01`)**: A 1% stop means the agent can sustain up to 1% loss per trade before ejecting. If multiple consecutive trades hit their stops, the cumulative drawdown compounds quickly.

2. **Trailing stop adjusts too slowly**: `ADJUST_STOP` trails based on `peak_price × (1 - stop_pct)`. With stop_pct = 1%, the stop only locks in gains after a 1%+ move. Smaller winning moves exit via stop instead of locking in the profit.

3. **Reward drawdown penalty has a 2% free zone**: The `ALPHA` drawdown penalty only activates beyond 2% portfolio drawdown. The agent is not penalized for the first 2% of drawdown, which may allow it to let losing positions run longer than needed.

4. **No position-sizing asymmetry**: Every buy uses 90% of available cash. A smaller position size on lower-confidence entries would reduce drawdown exposure.

---

## Proposed Changes (ordered by expected impact, lowest risk first)

### Change 1 — Tighten Stop to 0.5% (Recommended First Try)

**File:** `environment/order_book.py`  
**Change:** `DEFAULT_STOP_PCT = 0.01` → `DEFAULT_STOP_PCT = 0.005`

**Rationale:** Halving the stop distance halves the maximum loss per trade. With 58.5% win rate, the agent is profitable — the issue is average loss size, not win rate. A tighter stop should reduce loss magnitude without significantly reducing the number of winning trades, since BTC at 1-min resolution has enough resolution to distinguish signal from noise within 0.5%.

**Risk:** If the market is "noisy" within 0.5% on 1-min candles, stop-outs may increase, reducing win rate and increasing trade frequency. Monitor win rate carefully.

**Metric to watch:** Win rate should stay above 50%. Trade count should not increase more than 2x.

---

### Change 2 — Tighten Drawdown Free Zone from 2% to 1%

**File:** `environment/trading_env.py`  
**Change:** `drawdown_penalty = ALPHA * max(0.0, drawdown - 0.02)` → `max(0.0, drawdown - 0.01)`

**Rationale:** The 2% free zone means the agent can let portfolio value drop 2% from peak before incurring any penalty. Tightening to 1% means the agent is penalized sooner when drawdown starts accumulating, teaching it to cut losses earlier.

**Risk:** May make the agent overly reactive, reducing win rate. Should be evaluated after Change 1 is stable.

---

### Change 3 — Increase ALPHA (Drawdown Penalty Weight)

**File:** `environment/trading_env.py`  
**Change:** `ALPHA = 0.5` → `ALPHA = 0.75`

**Rationale:** Stronger penalty for drawdown beyond the free zone. Combined with Change 2, this creates a much stronger signal to avoid drawdown events.

**Risk:** If too aggressive, the agent may refuse to open positions at all (excessive caution). Watch for HOLD% climbing above 70%.

---

### Change 4 — Reduce Capital Fraction Per Trade

**File:** `environment/order_book.py`  
**Change:** `capital_fraction = 0.9` → `capital_fraction = 0.6` (in `place_buy_limit`)

**Rationale:** Using only 60% of capital per trade means each stop-out produces a smaller absolute loss. This also leaves cash reserves that can compound if the agent opens a second position after recovering.

**Risk:** Lower position size means lower absolute returns (not % returns — the win/loss ratios are unchanged, but the dollar P&L is smaller). Return target of +5% should still be achievable.

---

## Recommended Sequence

1. Implement PRD-001 (best_sortino bug fix) first.
2. Apply Change 1 only. Train from best checkpoint with `--alpha 0.05` for 100k steps.
3. Evaluate at 10k, 20k, 30k steps. If drawdown < -15% with Sortino > 1.5 → done.
4. If drawdown still > -15%, apply Change 2 and re-train.
5. If still not meeting target after Change 2, apply Change 3.
6. Change 4 is a last resort — reduces return alongside drawdown.

---

## Files to Change

| File | Change |
|------|--------|
| `environment/order_book.py` | `DEFAULT_STOP_PCT`: 0.01 → 0.005 (Change 1) |
| `environment/order_book.py` | `capital_fraction`: 0.9 → 0.6 (Change 4, if needed) |
| `environment/trading_env.py` | Drawdown free zone: 0.02 → 0.01 (Change 2, if needed) |
| `environment/trading_env.py` | `ALPHA`: 0.5 → 0.75 (Change 3, if needed) |
| `iter_log.md` | Document each run with full checkpoint table |

---

## Acceptance Criteria

1. A checkpoint is produced with max drawdown < -15%.
2. Sortino ratio for that checkpoint is > 1.5.
3. Total return for that checkpoint is > +5%.
4. Win rate for that checkpoint is > 50%.
5. The checkpoint is saved as a permanent named backup (e.g., `nova_brain_drawdown14_sortino1.6.pt`).

---

## Notes

- Every training run must start from the PRD-001-fixed `train.py`. Never run without the best_sortino fix.
- Always preserve the current best checkpoint (`nova_brain_iter4_100k_sortino1.73.pt`) as a named backup before any run that might overwrite `nova_brain_best.pt`.
- The drawdown target (-15%) is a hard requirement from Eric for live capital deployment. Do not relax it.
