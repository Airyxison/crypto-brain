# Nova Brain — Experiment Log

Format: each entry documents **what changed**, **what we were testing**, and **what the results showed**.
Updated after every completed run. Source of truth for "have we tried this before."

---

## v10 — Alpha Explosion Fix
**What changed:** Fixed reward scaling that caused alpha to explode during training  
**What we were testing:** Whether stable training was possible at all  
**Results:** BTC Sortino -3.31  
**Conclusion:** Training stabilized. Baseline established. Still deeply negative.

---

## v11 — AsyncVectorEnv + 16 Parallel Envs
**What changed:** Switched from serial to vectorized training (AsyncVectorEnv, 16 envs). Added `store_batch` for efficient replay buffer population.  
**What we were testing:** Whether more diverse episode sampling improves learning  
**Results:**
- BTC: -0.0382 (best checkpoint: step ~25k)
- ADA: **+0.0538** ← first positive Sortino in project history
**W&B:** (session logs 2026-04-17/18)  
**Conclusion:** Vectorized training helped. **Early-peak pattern confirmed** — best results always at ~25k steps, never improves after. ADA positive result established a real signal exists.

---

## v12.1 — Regime-Aware Episode Sampling
**What changed:** Added BEAR 2x / BULL 0.8x episode sampling weights. Flat GAMMA (removed proportional scaling from v10).  
**What we were testing:** Whether oversampling bear regimes improves risk-adjusted performance  
**Results:** BTC: -0.0382 (same as v11 best — no improvement)  
**Conclusion:** Regime sampling did not help. Matched v11 but didn't exceed it.

---

## v12.2 — Proportional Hold Penalty
**What changed:** Hold penalty scaled proportionally to position loss magnitude  
**What we were testing:** Whether penalizing bad holds more aggressively reduces drawdown  
**Results:** Worse than v12.1. Avg hold time cut to 38 bars (too short).  
**Conclusion:** Proportional penalty overcorrected — agent exited too early, couldn't learn hold discipline. Reverted.

---

## v12.3 — Recalibrated Reward Weights
**What changed:** GAMMA between v10 and v12.2 values. BEAR weight lightened from 3x to 2x.  
**What we were testing:** Whether gentler regime sampling + intermediate hold penalty helps  
**Results:** Worse than v12.1.  
**Conclusion:** Weight tuning at this level has diminishing returns.

---

## v12.4 — 17-dim Feature (Random Init)
**What changed:** Added `momentum_8h` as feature[16] (8h rolling return). Trained from random init.  
**What we were testing:** Whether 8h momentum signal improves regime awareness  
**Results:** BTC: -2.9861. Plateau at ~110k steps.  
**Conclusion:** Random init from 17-dim is too slow to converge. Migration from 16-dim checkpoint needed.

---

## v12.5 — 17-dim Migration + Change A (Regime-Conditional Reward)
**What changed:**
- Migrated best 16-dim checkpoint (BTC step500k, -0.0382) to 17-dim via `scripts/migrate_checkpoint_16to17.py`
- Added regime-conditional opp_cost (Change A): bull regime = cash penalty, bear regime = cash bonus, range = no signal
- `exploit_start=40k`, `exploit_floor=-1.5` (alpha≈0.22)
- `opp_cost_thresh=0` (always fires when momentum_8h crosses zero)

**What we were testing:** Whether regime-conditional reward shaping helps policy stay active in trending markets  
**Results:**
- BTC: **+0.2506** ✅ (best result in project — found at ~25k steps)
- ETH: -1.5898 ❌
- SOL: -2.2347 ❌
- ADA: -0.2261 ❌ (baseline unchanged)
**W&B BTC:** https://wandb.ai/enlnetsol/nova-brain/runs/8ombjyj7  
**Conclusion:** Change A works for BTC, does not generalize. Regime analysis showed all 4 assets have identical regime distributions (~97% BULL/BEAR) — the difference is move magnitude (ADA moves 2.1x BTC per bar). Same EPSILON = 2x signal strength for ADA. **Early-peak pattern persists** — BTC best at ~25k, no improvement after.

---

## v12.6 — Per-Symbol Epsilon + opp_cost Threshold
**What changed:**
- Per-symbol EPSILON normalization based on avg actionable move (BTC=1.0e-5, ETH=6.83e-6, SOL=5.19e-6, ADA=4.71e-6)
- `opp_cost_thresh=0.005`: require |momentum_8h| > 0.5% to trigger opp_cost (filter near-zero noise)
- Intended to normalize signal strength across assets and reduce always-on firing

**What we were testing:** Whether calibrated epsilon + selective regime threshold helps non-BTC assets  
**Results:**
- BTC: best eval **-0.7283** (step 80k) — **WORSE than v12.5** despite same epsilon for BTC
- ETH/SOL/ADA: not run (BTC result was conclusive)
**W&B BTC:** https://wandb.ai/enlnetsol/nova-brain/runs/kle3gpn3  
**Conclusion:** The `opp_cost_thresh=0.005` removed signal that was helping BTC. BTC's v12.5 success depended on always-on opp_cost (thresh=0). The threshold was a wrong turn. Per-symbol epsilon is retained in code but untested in isolation.

---

## v12.7-A — Remove Exploit Schedule (Diagnostic)
**What changed:**
- Removed `--exploit-start` and `--exploit-floor` entirely
- `auto_alpha=True` runs unsupervised for all 200k steps (no entropy decay floor)
- Everything else unchanged (same checkpoint as v12.5/v12.6 BTC runs)

**What we are testing:** Whether the piecewise entropy schedule is the primary cause of post-40k degradation. Independent architecture review identified the exploit phase as the likely culprit — the agent switches to exploit before the critic has reliable Q-estimates, causing mode collapse. Best results in every prior run appeared just *before* exploit kicked in.

**Hypothesis:** Removing the entropy brake will allow the policy to keep exploring past step 40k and either (a) find a better peak later, or (b) maintain a stable policy rather than degrading.

**Success criteria:** Best Sortino found after step 40k AND/OR best Sortino > -0.0382  
**Failure criteria:** Same early-peak pattern, best still at <40k steps  

**Results:** TBD (run in progress as of 2026-04-22 22:19)  
**W&B BTC:** https://wandb.ai/enlnetsol/nova-brain/runs/n64g64yc

---

## v12.7-B — Full Training Stability Overhaul (Queued, pending v12.7-A outcome)
**What changed (planned):**
- Episode length: 500 → 2000 ticks (agent sees full trade lifecycle)
- Q-target clamping: `torch.clamp(q_target, -10, 10)` in critic loss
- Batch size: 1024 → 256 (reduce overfitting on warm replay buffer)
- Exploit schedule: removed (same as v12.7-A)
- Training budget: 200k → 300k steps

**What we are testing:** Whether sparse reward signal + short episodes are preventing stable critic training. Episode length of 500 ticks means most exits are auto-triggered (max_hold_bars), not agent-chosen. The realized_bonus term credits stop-losses, not learned behavior.

**Execute if:** v12.7-A shows same early-peak pattern (i.e., exploit schedule is not the primary cause)  
**Skip if:** v12.7-A succeeds (exploit schedule was the issue)

---

## Key Patterns (Reference)

| Pattern | First observed | Still present |
|---------|---------------|---------------|
| Early-peak (best at <40k steps) | v11 | Every run through v12.6 |
| BTC-only positive Sortino | v12.5 | — |
| Reward complexity growth without stability gain | v12.1→v12.6 | Addressed in PRD-004 |
| col[16] norm growth to ~20+ | v12.5 | Confirms momentum_8h is used |

## Deployment Bar (not met)
Drawdown < -15% AND win rate > 50% AND positive Sortino on genuine OOS data — across multiple assets.
