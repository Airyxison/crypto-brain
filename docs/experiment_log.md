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

**What we tested:** Whether the piecewise entropy schedule is the primary cause of post-40k degradation.

**Results:**
| Step | Sortino |
|------|---------|
| 10k  | **-0.4563** ← best (still an early-peak) |
| 20k  | -2.6909 |
| 30k  | -3.3495 |
| 40k  | -2.7076 |
| 50k  | -3.0299 |
| 60k  | -1.6458 |
| 70k  | -2.4904 |
| 80k  | -1.3935 |
| 90k  | -1.6329 |
| 100k | -2.0987 |
| 110k | -3.0292 |
| 120k | -3.3752 |

**Smoking gun:** `alpha` locked at `7.3891` (= `e^2.0`, the clamp ceiling) from step 1000 onward. Q-values diverged before the first eval — not at step 40k as hypothesized. The policy collapsed to 92% HOLD immediately.

**Conclusion:** OPTION A FAILED. Exploit schedule was NOT the primary cause. The early-peak pattern persisted — best at step 10k, same as all prior runs. Root cause is Q-value divergence from the first training step. Reward magnitude (GAMMA=3e-5, EPSILON=1e-5) produces near-zero targets with high variance; without Q-target clamping, the Bellman backup amplifies noise, critics diverge, alpha spikes to clamp ceiling, policy collapses. Proceed to Option B.

**W&B BTC:** https://wandb.ai/enlnetsol/nova-brain/runs/n64g64yc

---

## v12.7-B — Full Training Stability Overhaul (Code-ready, pending Eric approval)
**What changed:**
- Q-target clamping: `torch.clamp(q_target, -10, 10)` in critic loss ← **implemented in agent/sac.py**
- Episode length: 1000 → 2000 ticks (agent sees full trade lifecycle) ← `--episode-steps 2000`
- Batch size: 1024 → 256 (reduce overfitting on warm replay buffer) ← `--batch-size 256` (**new CLI arg, implemented in train.py**)
- Exploit schedule: removed (same as v12.7-A) ← no `--exploit-start` flag
- Training budget: 200k → 300k steps ← `--steps 300000`

**Train command (ready to run):**
```
python train.py \
  --db /root/ticks.db \
  --symbol BTCUSDT \
  --resume checkpoints/btcusdt/nova_brain_v12.1_migrated_17dim.pt \
  --steps 300000 \
  --episode-steps 2000 \
  --batch-size 256 \
  --num-envs 16 \
  --opp-cost-thresh 0.0 \
  --save-every 10000 \
  2>&1 | tee /root/logs/v12.7b_btcusdt.log
```

**What we are testing:** Whether Q-value divergence (confirmed in v12.7-A: alpha=7.389 from step 1k) is the root cause of the early-peak pattern. Q-clamp prevents the Bellman backup from amplifying reward noise. Longer episodes give the agent a chance to see full trade lifecycles so realized_bonus becomes meaningful signal rather than crediting stop-losses.

**Execute if:** v12.7-A shows same early-peak pattern — CONFIRMED, proceed.  
**Status:** Awaiting Eric approval.

**Results:** TBD

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
