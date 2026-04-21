# PRD: Phase 1 — Checkpoint Migration Shim & Regime-Conditional Reward

## Overview
**Status**: Draft  
**Created**: 2026-04-20  
**Owner**: Development Team  
**Type**: Brownfield  
**Requested by**: Kiran (Testing Manager)  
**Project**: crypto-brain (RL trading agent)

## Problem Statement

v12.4 upgraded the observation space from 16 to 17 dimensions (adding `momentum_8h` as feature[16]), but trained from **random initialization** instead of migrating from the best known checkpoint (v12.1, Sortino +0.1076). This resulted in:

1. **Lost policy knowledge**: v12.4 has plateaued at Sortino -3.7082 after ~100k steps, significantly worse than the v12.1 baseline
2. **Bear market conflicting gradients**: The current opportunity cost penalty fires on all cash-holding when price moves up, including during bear regimes where holding cash is the correct decision
3. **No positive signal for correct bear behavior**: Agent is penalized for holding cash in downtrends but receives no explicit reward for avoiding losses

The 16→17 dimension incompatibility means the best performing checkpoint cannot be loaded into the current network architecture, forcing training to restart from scratch.

## Business Goals

1. **Preserve learned policy knowledge**: Start v12.5 training from the best known policy (+0.1076 Sortino) rather than random init
2. **Improve bear regime performance**: Reduce BEAR regime trade entries from current levels (v12.1 had 1% BUY rate in BEAR) while maintaining BULL regime profitability
3. **Establish clean migration pattern**: Create a reusable approach for future dimension changes in the observation space
4. **Provide clear W&B validation signals**: Enable Kiran to verify migration success through specific metrics in Weights & Biases

## User Stories

1. **As a developer**, I want to extend a 16-dim checkpoint to 17-dim without losing learned weights, so that v12.5 starts with the best known policy instead of random initialization
2. **As a testing manager (Kiran)**, I want explicit W&B signals that confirm the migration landed cleanly, so I can approve the training run without uncertainty
3. **As the RL agent**, I want to receive positive reward for holding cash during bear regimes, so I can learn to distinguish correct inaction from missed opportunities
4. **As a researcher**, I want the new `momentum_8h` feature to contribute minimally at step 0 and grow gradually over 20-30k steps, so the fine-tuning process is stable

## Scope

### In Scope (Phase 1)

**Two tightly coupled changes that must ship together:**

1. **Migration Shim (16→17 dim checkpoint extension)**
   - One-off standalone script: `scripts/migrate_checkpoint_16to17.py`
   - Extends Actor and Critic first linear layers from `[256, 16]` → `[256, 17]`
   - Near-zero initialization for new weight column (preserves existing 16-dim policy)
   - Safety: copy-tag source checkpoint before any modifications
   - Produces: `btcusdt/nova_brain_v12.1_migrated_17dim.pt`

2. **Change A: Regime-Conditional Opportunity Cost + Bear Preservation Bonus**
   - Modify `_compute_reward()` in `environment/trading_env.py`
   - Opportunity cost now only fires when `price_move > 0 AND momentum_8h > 0` (bull trend)
   - New bear preservation bonus: `+EPSILON * abs(price_move)` when `price_move < 0 AND momentum_8h < 0`
   - Expose `momentum_8h` as cached property on `FeatureEngineer` class
   - Same magnitude as opportunity cost to avoid scale imbalance

3. **Fine-Tune Stabilization Window (first 20-30k steps)**
   - Monitor W&B for clean migration signals (critic/actor loss stability, no alpha spikes)
   - Expect near-identical behavior to v12.1 in first 20k steps (new column near-zero)
   - Gradual weight growth on column [16] as agent learns to use `momentum_8h`

### Out of Scope

- **Change B**: Composite checkpoint selection (best_bear + best_bull) — Phase 2
- **Change C**: W&B regime logging enhancement — Phase 0b (already planned separately)
- **Change D**: Regime-conditional `MIN_HOLD_BARS` — Phase 2
- **Feature additions**: SMA200 distance feature — Phase 3
- **Training strategy**: Curriculum or multi-asset training — Phase 3
- **Network architecture changes**: Layer sizes, activation functions, etc.
- **Inference-time action masking**: Track 1 diagnostic experiment only (training/inference mismatch concerns)

## Technical Context

### Existing System (Brownfield)

**Architecture**: Soft Actor-Critic (SAC) discrete action space, long-only BTC trading

**Key Components**:
- `agent/sac.py` — SAC agent with Actor and Critic networks
- `agent/networks.py` — Network definitions: `nn.Linear(STATE_DIM, 256)` first layer
- `environment/trading_env.py` — RL environment, reward computation
- `features/engineer.py` — Computes 17-dim observation vector
- `train.py` — Training loop with W&B logging
- `backtest/runner.py` — OOS evaluation with regime breakdown

**Current State**:
- `STATE_DIM = 17` (was 16 before v12.4)
- Feature[16] = `momentum_8h` (8-hour return, matches REGIME_WINDOW=480 bars)
- Best checkpoint: `s3://nova-trader-data-.../btcusdt/nova_brain_best.pt` (v12.1, 16-dim, Sortino +0.1076)
- v12.4 status: step ~100k, best Sortino -3.7082, trained from random init

**Reward Function Components** (current):
- Base return: `(current_pv - prev_pv) / prev_pv`
- Drawdown penalty: `ALPHA * max(0, drawdown - 0.02)` (asymmetric, 2% free zone)
- Hold cost: `-GAMMA` per bar when PnL < -1% (flat, not proportional)
- **Opportunity cost**: `-EPSILON * price_move` when not in position and `price_move > 0` (fires unconditionally on any upward tick)
- Realized bonus: scaled by hold duration and exit type (stop vs agent-chosen)

### Proposed Approach

**Migration Strategy**:
- **Weight extension**: Append new column initialized to `torch.zeros(256, 1) * 0.01 + small_noise`
- **Rationale**: Near-zero column means `momentum_8h` contributes ~0 to actor/critic outputs at step 0
- **Gradual activation**: Over 20-30k steps, gradient descent grows the new weights as the agent learns feature[16]'s predictive value
- **Stability**: Existing 16-dim policy is preserved — no sudden behavior change

**Reward Modification**:
```python
# BEFORE (v12.4):
if price_move > 0:
    opp_cost = -EPSILON * price_move  # unconditional penalty

# AFTER (v12.5, Phase 1):
momentum_8h = self._features.momentum_8h  # cached property
if price_move > 0 and momentum_8h > 0:
    opp_cost = -EPSILON * price_move       # bull-only penalty
elif price_move < 0 and momentum_8h < 0:
    opp_cost = +EPSILON * abs(price_move)  # bear bonus (same magnitude)
```

**Design Decisions**:
1. **Same magnitude for bonus and penalty**: Avoids introducing new hyperparameter; empirically tune if BULL Sortino drops
2. **Cached property pattern**: `momentum_8h` already computed in `extract()`, just expose via `@property` — no re-computation overhead
3. **One-off script pattern**: Migration is a versioning event, not a training-loop concern; keep it isolated
4. **Copy-tag before modification**: Protect the baseline; rollback path is `s3 cp ...best_sortino0.1076.pt ...best.pt`

## Detailed Requirements

### Functional Requirements

1. **Checkpoint Migration Script**
   - **FR-1.1**: Load 16-dim checkpoint from S3 path (user-provided or default to `btcusdt/nova_brain_best.pt`)
   - **FR-1.2**: Extract Actor and Critic state_dicts
   - **FR-1.3**: For each network's first linear layer (`fc1.weight`), extend from shape `[256, 16]` to `[256, 17]`
   - **FR-1.4**: Initialize new column as `torch.zeros(256, 1) * 0.01 + torch.randn(256, 1) * 0.001`
   - **FR-1.5**: Copy all other layer weights unchanged (they are shape-compatible)
   - **FR-1.6**: Save migrated checkpoint to S3: `btcusdt/nova_brain_v12.1_migrated_17dim.pt`
   - **FR-1.7**: Copy-tag source as `btcusdt/nova_brain_v12.1_best_sortino0.1076.pt` BEFORE any modifications
   - **FR-1.8**: Assert weight shapes before saving (crash if mismatch)
   - **Priority**: **High**

2. **FeatureEngineer Cached Property**
   - **FR-2.1**: Add instance variable `self._momentum_8h: float` initialized to 0.0 in `__init__`
   - **FR-2.2**: In `extract()`, store computed `momentum_8h` value to `self._momentum_8h` before returning state vector
   - **FR-2.3**: Expose `@property momentum_8h` that returns `self._momentum_8h`
   - **FR-2.4**: Property must return the value computed in the MOST RECENT `extract()` call (no stale values)
   - **Priority**: **High**

3. **Regime-Conditional Reward**
   - **FR-3.1**: In `_compute_reward()`, replace unconditional opportunity cost block with regime-conditional logic
   - **FR-3.2**: Bull regime (`price_move > 0 AND momentum_8h > 0`): apply penalty `-EPSILON * price_move`
   - **FR-3.3**: Bear regime (`price_move < 0 AND momentum_8h < 0`): apply bonus `+EPSILON * abs(price_move)`
   - **FR-3.4**: No opp_cost change in RANGE regime (momentum_8h near zero) or mismatched regimes
   - **FR-3.5**: Magnitude of bonus and penalty must be equal (same `EPSILON` constant)
   - **Priority**: **High**

4. **Training Loop Integration**
   - **FR-4.1**: Load migrated checkpoint at start of v12.5 training (modify `train.py`)
   - **FR-4.2**: Log `actor.fc1.weight[:, 16].norm()` to W&B every 10k steps (optional but recommended)
   - **FR-4.3**: Continue existing W&B logging: critic/actor loss, alpha, regime Sortino breakdowns
   - **Priority**: **Medium**

### Non-Functional Requirements

- **Performance**: No degradation to step time (cached property adds ~zero overhead)
- **Security**: S3 access via existing boto3 patterns; no new credentials needed
- **Scalability**: Migration script is one-off; no performance concern
- **Maintainability**: 
  - Migration script is self-documenting with assertions and comments
  - Cached property pattern is standard Python idiom
  - Reward logic change is localized to 5-line block in `_compute_reward()`
- **Reliability**:
  - Checkpoint copy-tag prevents accidental overwrite of baseline
  - Shape assertions catch dimension mismatches before saving
  - Unit test for `momentum_8h` property with synthetic price sequence

## Acceptance Criteria

### Must Have (P0)

- [ ] **AC-1**: Migration script runs without error on v12.1 checkpoint from S3
- [ ] **AC-2**: Migrated checkpoint shape `actor.fc1.weight.shape == [256, 17]` confirmed via assertion
- [ ] **AC-3**: Migrated checkpoint loads cleanly into 17-dim SAC agent (no shape errors)
- [ ] **AC-4**: Source checkpoint copy-tagged to `...best_sortino0.1076.pt` before modification
- [ ] **AC-5**: `FeatureEngineer.momentum_8h` property returns correct value (unit test with synthetic prices)
- [ ] **AC-6**: Critic loss stable in first 5k steps of v12.5 (no spikes > 2x baseline mean)
- [ ] **AC-7**: Actor loss stable in first 5k steps (no spikes > 2x baseline mean)
- [ ] **AC-8**: Alpha follows configured piecewise schedule (no spike at step 0)
- [ ] **AC-9**: `regime/BEAR/sortino` visible in W&B from first backtest eval (confirms Phase 0b logging active)
- [ ] **AC-10**: By step 50k, overall Sortino is trending positive or stable (not deteriorating below -3.7)
- [ ] **AC-11**: By step 80k, overall Sortino exceeds v12.1 baseline of +0.1076
- [ ] **AC-12**: BEAR regime trade count lower in v12.5 than v12.1's OOS backtest baseline

### Should Have (P1)

- [ ] **AC-13**: Weight norm of `actor.fc1.weight[:, 16]` logged to W&B every 10k steps
- [ ] **AC-14**: Column [16] weight norm near-zero (< 0.05) at step 0
- [ ] **AC-15**: Column [16] weight norm visibly non-zero (> 0.2) by step 30k
- [ ] **AC-16**: Average hold bars remain > 100 bars (no scalping regression)
- [ ] **AC-17**: BULL regime Sortino does not drop by > 20% vs v12.1 (bear bonus not over-tuned)

### Nice to Have (P2)

- [ ] **AC-18**: Migration script supports --dry-run flag (print shapes, no save)
- [ ] **AC-19**: Migration script logs weight statistics (min/max/mean of new column)
- [ ] **AC-20**: Unit test coverage for `_compute_reward` opp_cost block (synthetic price sequences for BULL/BEAR/RANGE)

## Implementation Phases

*This PRD covers Phase 1 only — no further breakdown required.*

**Phase 1 context in larger roadmap**:
- **Phase 0b**: W&B regime logging (already planned separately) — prerequisite for AC-9
- **Phase 1** (this PRD): Migration shim + Change A
- **Phase 2**: Change B (composite checkpoints) + Change D (regime-conditional MIN_HOLD_BARS)
- **Phase 3**: SMA200 feature, curriculum training

## Files to Create/Modify

| File Path | Change Type | Description |
|-----------|-------------|-------------|
| `scripts/migrate_checkpoint_16to17.py` | **CREATE** | One-off migration script; extends weight matrices, saves migrated checkpoint |
| `features/engineer.py` | **MODIFY** | Add `_momentum_8h` instance var + `@property momentum_8h` |
| `environment/trading_env.py` | **MODIFY** | Update `_compute_reward()` opp_cost block (5-line change) |
| `train.py` | **MODIFY** | Load migrated checkpoint, add optional column [16] weight norm logging |
| `agent/sac.py` | **NO CHANGE** | Network already 17-dim; shim produces compatible checkpoint |
| `agent/networks.py` | **NO CHANGE** | `STATE_DIM = 17` already set |

## Testing Strategy

### Unit Tests

1. **Test: `FeatureEngineer.momentum_8h` caching**
   - Input: Synthetic price sequence with known 8h return (-0.05)
   - Expected: `momentum_8h` property returns -0.05 after `extract()` call
   - Location: `tests/test_feature_engineer.py` (create if not exists)

2. **Test: Reward function bear bonus**
   - Setup: Mock `_features.momentum_8h = -0.05`, `price_move = -0.01`
   - Expected: `opp_cost = +EPSILON * 0.01`
   - Location: `tests/test_trading_env.py`

3. **Test: Reward function bull penalty**
   - Setup: Mock `_features.momentum_8h = 0.05`, `price_move = 0.01`
   - Expected: `opp_cost = -EPSILON * 0.01`
   - Location: `tests/test_trading_env.py`

4. **Test: Reward function RANGE regime (no opp_cost)**
   - Setup: Mock `_features.momentum_8h = 0.005`, `price_move = 0.01`
   - Expected: `opp_cost = 0.0` (momentum too weak to trigger)
   - Location: `tests/test_trading_env.py`

### Integration Tests

1. **Test: Migration script end-to-end**
   - Input: Dummy 16-dim checkpoint (create synthetic)
   - Expected: Output checkpoint loads into 17-dim network without error
   - Validation: Assert `actor.fc1.weight.shape == [256, 17]`
   - Run: One-time manual test before production migration

2. **Test: First 10 steps of v12.5 training**
   - Input: Migrated checkpoint
   - Expected: No errors, critic/actor loss within 2x of v12.4 baseline
   - Run: After migration, monitor W&B dashboard

### Manual Validation (W&B Dashboard)

**First 5k steps** (immediate validation):
- [ ] Critic loss stable (no spike)
- [ ] Actor loss stable
- [ ] Alpha follows piecewise schedule (hits exploit_start=40k correctly)
- [ ] `regime/BEAR/sortino` key present in logs

**By step 30k** (stabilization window):
- [ ] Column [16] weight norm > 0.2 (agent learning to use `momentum_8h`)
- [ ] BEAR regime trade entries declining vs v12.4 baseline

**By step 80k** (success milestone):
- [ ] Overall Sortino > +0.1076 (exceeds v12.1 baseline)
- [ ] Average hold bars > 100

## Risk & Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Critic loss spikes at step 0** | High — training from broken init, wasted compute | Medium | Add shape assertions in migration script; test on synthetic checkpoint first |
| **`momentum_8h` property returns stale value** | Medium — reward uses wrong regime signal, agent gets conflicting gradients | Low | Unit test with synthetic sequence; property updates in same `extract()` call that builds state vector |
| **Bear bonus scale too large** | Medium — agent stops buying in BULL regime | Medium | Monitor `regime/BULL/sortino` in W&B; if drops > 20%, reduce bonus scale to `EPSILON * 0.5` |
| **S3 overwrite of v12.1 checkpoint** | **Critical** — lose only positive-Sortino policy, no rollback | Low | **Step 0a**: Script copy-tags v12.1 BEFORE any load/modify operations; manual verification before --run |
| **New column near-zero init causes dead gradient** | Low — gradient should still flow through other 16 features | Very Low | Monitor weight norm in W&B; if still < 0.05 at step 50k, increase init scale to 0.05 |
| **Migration script fails mid-save** | Medium — corrupt checkpoint in S3 | Very Low | Save to temp local file first, upload only after complete write + hash check |
| **Phase 0b regime logging not complete** | High — AC-9 fails, can't validate BEAR metrics | Medium | **Blocker**: Confirm Phase 0b deployed before starting v12.5 training; fallback: merge both PRDs into single deployment |

## Open Questions

**Before implementation**:
- [x] Confirmed: v12.1 checkpoint is 16-dim (not 15 or 14)? — **RESOLVED**: Yes, confirmed in `docs/BEAR_REGIME_STRATEGY_BRIEF.md`
- [x] Is Phase 0b regime logging already deployed to W&B? — **ASSUMED YES** (AC-9 depends on this; will verify before training)
- [ ] What is the exact S3 path to v12.1 checkpoint? — **ACTION**: Developer to confirm before running migration script
- [ ] Should migration script support multi-asset (ETH/SOL/ADA) or BTC-only? — **DEFAULT**: BTC-only for Phase 1; extend to other assets in Phase 2 if needed

**During training**:
- [ ] If BULL Sortino drops > 20%, should we reduce bear bonus scale mid-training or restart? — **DECISION RULE**: Monitor until step 50k; if drop confirmed, reduce bonus scale and continue (no restart unless Sortino < -1.0)

**Post-training**:
- [ ] If v12.5 beats v12.1 by step 80k, should we skip Phase 2 (composite checkpoints) or continue? — **DEFER** to post-Phase 1 retrospective

## Research Notes

**Referenced from `docs/BEAR_REGIME_STRATEGY_BRIEF.md`**:

1. **Best checkpoint baseline**: v12.1, Sortino +0.1076, step 80k (NOT 1.73 as originally hypothesized)
2. **Kiran's concern about inference-time masking**: Creates training/inference distribution mismatch; Track 1 diagnostic only
3. **Cached property pattern**: Kiran recommended `self._features.momentum_8h` cached property over re-computation in `_compute_reward`
4. **Bear threshold**: `-0.02` for `momentum_8h` is tighter than `REGIME_BEAR_THRESH = -0.05`; use -0.05 to match environment's regime classification
5. **Migration shim rationale**: Sound engineering to preserve learned policy; lower ROI than originally thought (migrating from +0.1076, not +1.73), but still worthwhile to avoid wasting 80k training steps

## W&B Metrics Reference

**Existing metrics** (already logged):
- `loss/critic` — Q-network loss
- `loss/actor` — policy gradient loss
- `alpha` — entropy temperature (auto-tuned or piecewise decayed)
- `eval/sortino` — overall Sortino ratio from OOS backtest
- `eval/sharpe` — overall Sharpe ratio
- `eval/total_return` — cumulative return over backtest window

**New metrics** (Phase 0b, required for AC-9):
- `regime/BEAR/sortino` — Sortino ratio for trades entered in BEAR regime
- `regime/BEAR/trades` — count of BEAR regime entries
- `regime/BULL/sortino` — Sortino ratio for trades entered in BULL regime
- `regime/BULL/trades` — count of BULL regime entries

**Optional metrics** (Phase 1, AC-13):
- `weights/actor_fc1_col16_norm` — L2 norm of new weight column, logged every 10k steps

## Success Metrics Summary

**Primary goal**: Overall Sortino > +0.1076 by step 80k (beats v12.1 baseline)

**Secondary goals**:
1. BEAR regime trade count reduced vs v12.1
2. BULL regime Sortino maintained within 20% of v12.1
3. Average hold bars > 100 (no scalping regression)
4. Clean migration (critic/actor loss stable, no alpha spike)

**Failure conditions** (stop training, debug):
- Critic loss spikes > 5x baseline in first 5k steps
- Alpha diverges from configured schedule
- Overall Sortino < -5.0 by step 50k (worse than v12.4)
- BEAR regime trade count INCREASES vs v12.1 (reward change backfired)

## Appendix

### A. Weight Extension Math

**Actor/Critic first linear layer**:
```
BEFORE: fc1.weight.shape = [256, 16]  (output_features, input_features)
AFTER:  fc1.weight.shape = [256, 17]
```

**Extension operation**:
```python
old_weight = checkpoint['actor']['fc1.weight']  # [256, 16]
new_col = torch.zeros(256, 1) * 0.01 + torch.randn(256, 1) * 0.001
new_weight = torch.cat([old_weight, new_col], dim=1)  # [256, 17]
```

**Why near-zero init?**
- At step 0, `state[16] * new_col ≈ 0` → new feature contributes ~0 to hidden activations
- Existing 16-dim policy preserved → no sudden behavior change
- Over 20-30k steps, gradient descent grows `new_col` if feature[16] is predictive
- If feature[16] is NOT predictive, column stays small (graceful degradation)

### B. Opportunity Cost Evolution

| Version | Behavior | Problem |
|---------|----------|---------|
| v9 | Unconditional penalty on cash during upward price move | Penalizes correct bear behavior |
| v10 | Same as v9, but `EPSILON` reduced 10x | Reduced magnitude but still fires in bear regimes |
| v12.2–v12.4 | Same as v10 | Conflicting gradients in bear markets |
| **v12.5 (Phase 1)** | **Regime-conditional: bull penalty + bear bonus** | **Explicit credit for correct bear inaction** |

### C. Regime Classification

**In `trading_env.py`** (for sampling weights):
```python
REGIME_WINDOW = 480  # 8 hours at 1-min bars
REGIME_BEAR_THRESH = -0.05  # 8h return < -5%
REGIME_BULL_THRESH = +0.05  # 8h return > +5%
```

**In Phase 1 reward logic**:
- Use `momentum_8h` directly (continuous value), not thresholded classification
- `momentum_8h > 0` → bullish pressure (penalty for cash)
- `momentum_8h < 0` → bearish pressure (bonus for cash)
- `abs(momentum_8h) < threshold` → RANGE (no opp_cost change)

**Rationale**: Continuous signal allows agent to learn nuanced regime boundaries; hard thresholds only in sampling (training data distribution shaping, not inference logic)

### D. Training Configuration (v12.5)

**From existing codebase** (no changes in Phase 1):
```python
# Reward hyperparameters
ALPHA = 0.5          # drawdown penalty
GAMMA = 0.00003      # losing hold cost (v12.3 tuning)
EPSILON = 0.00001    # opportunity cost scale (now regime-conditional)
MIN_HOLD_BARS = 50   # min bars for realized bonus (Phase 2 will make regime-conditional)

# SAC hyperparameters
gamma = 0.99         # discount factor
tau = 0.005          # soft update rate
lr = 3e-4            # learning rate
batch_size = 1024
buffer_size = 200_000
warmup_steps = 1_000

# Alpha decay (piecewise)
auto_alpha = True
entropy_factor = 0.98
exploit_start = 40_000  # step at which alpha decay begins
target_alpha_floor = -1.5  # log scale floor

# Regime sampling (v12.3 tuning)
REGIME_BEAR_WEIGHT = 2.0   # 2x more BEAR samples in training
REGIME_BULL_WEIGHT = 0.8   # 0.8x BULL samples (slight under-sample)
```

---

## Kiran's Review Notes (2026-04-21)

Overall: excellent PRD. Three concrete fixes required before code is written.

---

**🔴 FIX 1 — Wrong state_dict key (migration script will crash)**

The PRD uses `fc1.weight` throughout the migration math and appendix. This is incorrect.
Actor and Critic both use `nn.Sequential`, so the first linear layer key in the
state_dict is `net.0.weight`, not `fc1.weight`. Using `fc1.weight` will raise a
`KeyError` on load. Every reference in the migration script must use `net.0.weight`.

Correction to Appendix A:
```python
# WRONG:
old_weight = checkpoint['actor']['fc1.weight']

# CORRECT:
old_weight = checkpoint['actor']['net.0.weight']
```

---

**🔴 FIX 2 — `momentum_8h` caching timing mismatch**

FR-2.2 says: "In `extract()`, store computed `momentum_8h` value to `self._momentum_8h`."
This is one step stale. In `trading_env.step()`, the call order is:
1. `self._features.update(price, volume, t_ms)` — prices deque updated
2. `self._compute_reward(...)` — reads `self._features.momentum_8h` ← HERE
3. `self._get_obs()` → `self._features.extract(...)` — cache would be set HERE

`_compute_reward` runs BEFORE `extract()`. The cached property would return the value
from the previous step's extract(), not the current tick.

**Fix**: Compute `momentum_8h` inline in `_compute_reward()` directly from the prices
deque (already available on the FeatureEngineer instance). No caching needed:

```python
# In _compute_reward, before opp_cost block:
prices = self._features.prices
if len(prices) >= 480:
    momentum_8h = (prices[-1] - prices[-480]) / (prices[-480] + 1e-9)
else:
    momentum_8h = 0.0
```

This is simpler than the cached property approach and has no stale-value risk.
The `@property momentum_8h` on FeatureEngineer can still be added (useful for
diagnostics and the masking experiment) but should NOT be the source of truth
inside `_compute_reward`.

---

**🟡 NOTE — Phase 0b not yet deployed**

The PRD assumes "ASSUMED YES" for Phase 0b (W&B regime logging). It is NOT deployed.
Treat this as a blocker per the risk table. Phase 0b must ship in the same commit as
Phase 1, or before it. The simplest path: merge both into a single deployment.

---

**✅ S3 path confirmed**

The open question about the v12.1 checkpoint S3 path is resolved:
```
s3://nova-trader-data-249899228939-us-east-1-an/checkpoints/btcusdt/nova_brain_best.pt
```
This is the 16-dim v12.1 checkpoint (Sortino +0.1076). Tag it as:
```
s3://nova-trader-data-249899228939-us-east-1-an/checkpoints/btcusdt/nova_brain_v12.1_best_sortino0.1076.pt
```
before any migration operations.

---

**✅ Everything else approved as written.**

AC-1 through AC-12 are sufficient. The weight extension math (Appendix A) is correct
once `fc1.weight` is replaced with `net.0.weight`. The risk table is thorough.
The sign-off checklist is the right governance. Ready to implement once fixes 1 and 2
are applied.

---

## Sign-Off Checklist (Pre-Implementation)

**Before writing code**:
- [ ] FIX 1 applied: all `fc1.weight` references replaced with `net.0.weight`
- [ ] FIX 2 applied: `momentum_8h` computed inline in `_compute_reward()`, not via cached property
- [ ] Phase 0b regime logging merged into this deployment (not assumed — confirmed)
- [ ] S3 path confirmed: `btcusdt/nova_brain_best.pt` (resolved above)
- [ ] Migration script copy-tag logic reviewed (no overwrite risk)

**Before training**:
- [ ] Migration script tested on synthetic 16-dim checkpoint (shapes correct)
- [ ] Unit tests passing for `momentum_8h` property
- [ ] Migrated checkpoint manually loaded in Python REPL (no shape errors)
- [ ] W&B project configured to receive new metrics (`weights/actor_fc1_col16_norm`)

**Training milestones**:
- [ ] Step 5k: Critic/actor loss stable, AC-6/7/8 validated
- [ ] Step 30k: Column [16] weight norm > 0.2, AC-15 validated
- [ ] Step 50k: Sortino trending positive, AC-10 validated
- [ ] Step 80k: Sortino > +0.1076, AC-11 validated → **PHASE 1 SUCCESS**

---

**END OF PRD**
