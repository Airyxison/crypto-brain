# PRD-004: Training Stability Investigation

**Date:** 2026-04-22  
**Status:** Option A in progress  
**Author:** Kiran (Nova) + independent architecture review

---

## Background

Across all v12.x runs, a consistent pathology appears:

- Best Sortino is always found at ~25-40k steps (early in training)
- Policy degrades monotonically after ~40k steps regardless of reward changes
- All reward-function iterations (Change A, per-symbol epsilon, opp_cost threshold) failed to break this pattern
- Only BTC v12.5 produced a positive result (+0.2506) — and that peak was at ~25k steps

The v12.x series tuned *what the agent is rewarded for*. This PRD addresses *how the agent learns* — a different and more fundamental problem.

---

## Root Cause Hypothesis (from independent architecture review)

### 1. Piecewise entropy schedule destroys exploration prematurely

The `exploit_start_step` at 40k switches off auto-alpha and begins linear entropy decay toward `floor=-1.5` (alpha≈0.22). This assumes the agent has learned something worth exploiting at step 40k. It hasn't — the critic is still fitting to noise at that point.

The best results at 25-40k steps are the agent *just before* we turn off exploration. We then spend 160k steps in mode-collapse. The early-peak pattern is a direct artifact of the exploit schedule.

### 2. Reward signal too sparse for stable critic training

Episodes are 500 ticks (~8 minutes real-time). Most episodes contain 0-2 trades. The critic sees primarily:
- Tiny hold costs (GAMMA = 0.00003/bar)
- Opp cost noise (EPSILON * price_move ≈ 4e-9/bar)
- Rare realized bonus events

With 6+ reward terms and almost no meaningful events per episode, Q-value estimates are unreliable by step 40k. Switching to exploit at that point exploits garbage V-estimates.

### 3. Episode length too short to learn trade lifecycle

500 ticks = ~8 minutes. Most exits are auto-triggered (max_hold_bars=300), not agent-chosen. The `realized_bonus` term that should credit agent skill is mostly crediting stop-loss hits. The agent never sees a full trade lifecycle within a single episode.

### 4. Gradient overfitting on warm replay buffer

16 parallel envs (AsyncVectorEnv) × 2 grad updates/step = 32 critic updates per step in vectorized mode. Serial mode: 4 updates/step. High update frequency on a warm (but small) replay buffer causes overfitting to the current data window — the critic generalizes poorly beyond the episode distribution it's seen.

---

## Option A — Remove Exploit Schedule (Diagnostic)

**Hypothesis:** The piecewise entropy decay is the primary cause of post-40k degradation. Removing it and letting auto_alpha run freely will produce a more stable policy.

**Changes:**
- Remove `--exploit-start` and `--exploit-floor` from train command
- Let `auto_alpha=True` run unsupervised for all 200k steps
- Everything else unchanged (same checkpoint, same reward, same episode length)

**Test:** BTC only, 200k steps, resume from `nova_brain_v12.1_migrated_17dim.pt`

**Success criteria:** Best Sortino found later than step 40k AND/OR best Sortino > -0.0382 (baseline)

**Failure criteria:** Same early-peak pattern appears, best still at <40k steps

**If A succeeds:** Run ETH/SOL/ADA with same config. Consider as new training baseline.  
**If A fails:** Proceed to Option B.

---

## Option B — Full Training Stability Overhaul (If A fails or is inconclusive)

**Hypothesis:** The combination of sparse reward, short episodes, and gradient overfitting prevents the critic from learning reliable Q-estimates regardless of entropy schedule.

**Changes (in priority order):**

1. **Extend episode length:** `max_episode_steps` 500 → 2000 ticks  
   Gives agent time to see full trade lifecycle. Realized bonus becomes meaningful signal.

2. **Q-target clamping:** Add `torch.clamp(q_target, -10, 10)` in critic loss  
   Prevents Q-divergence when both critics overfit together on stale replay data.

3. **Reduce batch size:** 1024 → 256  
   Reduces overfitting on warm buffer. Forces more diverse sampling.

4. **Disable piecewise entropy decay:** Same as Option A — let auto_alpha run free.

5. **Extend training budget:** 200k → 300k steps  
   Longer episodes mean fewer episodes per step-count; need more steps to see equivalent data.

**Test:** BTC only first, then multi-asset if positive.

**Implementation:** See `agent/sac.py` critic loss, `train.py` episode config, `environment/trading_env.py` max_episode_steps.

---

## What We Are NOT Changing (and why)

- **Reward function terms** — Change A (regime-conditional opp_cost) stays. The v12.5 BTC result (+0.2506) shows it can work. The problem is training stability, not reward design.
- **Feature vector (17-dim)** — Solid. momentum_8h col[16] norm growing to 20+ confirms it's being used.
- **Per-symbol epsilon** — Keep in code (v12.6 change), but not the focus of this PRD.
- **Architecture (SAC, discrete, network size)** — Not changing until A and B are ruled out.

---

## Version Tracking

| Run | Config | BTC Best Sortino | Notes |
|-----|--------|-----------------|-------|
| v12.5 | exploit_start=40k, thresh=0 | +0.2506 | Best result to date |
| v12.6 | exploit_start=40k, thresh=0.005 | -0.7283 | Threshold hurt BTC |
| **v12.7-A** | **no exploit schedule** | **TBD** | **Option A** |
| v12.7-B | no exploit, ep=2000, bs=256, Q-clamp | TBD | Option B if needed |
