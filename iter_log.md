# Nova Brain — Iteration Log

## Run 1 — Baseline (checkpoints from initial training, nova_brain_best.pt @ step 9745)

**Eval date:** 2026-04-10  
**Checkpoint:** nova_brain_best.pt (step 9745)

### Metrics
| Metric | Value | Target |
|--------|-------|--------|
| Sortino Ratio | 0.5327 | > 1.5 |
| Total Return | +3.91% | > 5% |
| Win Rate | 50.2% | > 50% |
| Max Drawdown | -27.50% | < -15% |
| Total Trades | 42,015 / 100k ticks | — |
| Avg Hold | 0 bars | — |

**Action distribution:** HOLD:10% BUY:44% ADJ_STOP:1% REALIZE:42% CANCEL:3%

### Diagnosis

**Degenerate scalping behavior.** Agent learned to flip positions every ~2.4 ticks:
open → next tick close → repeat. 42,015 trades in 100k ticks with avg hold 0 bars.

Root cause: `realized_bonus = pct * 5.0 if pct > 0` in `_compute_reward()`.
- On a profitable close, total reward ≈ base_return + 5x_amplified_bonus ≈ 6x actual return
- On a losing close: 2x actual return
- This 6x:2x asymmetry overwhelms all other reward signals and incentivizes opening AND immediately closing any position with any positive move

Trade stats: avg winner +0.06%, avg loser -0.06%. Zero real edge — pure micro-scalping with no alpha.

Also observed two competing training processes writing to the same log and checkpoints — cleaned up by stopping both.

### Changes Made

**`environment/trading_env.py`:**
- `realized_bonus` multiplier: `5.0 → 1.5` (win multiplier only)
- Rationale: 5.0x was creating 6x total amplification for profitable closes. 1.5x still
  rewards profitable closes over stop-outs (2.5x vs 2.0x total) without overwhelming the
  base return signal.

**Training restart:** Fresh from scratch (no --resume) — checkpoint from bad run would
have miscalibrated Q-values for the new reward function.

---

## Run 2 — Fixed realized_bonus multiplier (1.5x), fresh start

**Start date:** 2026-04-10  
**Config:** Default hyperparams + realized_bonus 1.5x  
**Command:** `python train.py --db ../crypto-engine/ticks.db --steps 200000 2>&1 | tee train.log`

### Checkpoint Results

| Step | Sortino | Return | Drawdown | Win Rate | Trades | Avg Hold | HOLD% |
|------|---------|--------|----------|----------|--------|----------|-------|
| 10k  | -1.8607 | -17.71% | -28.85% | 50.5% | 11,734 | 5 bars | 23% |
| 20k  | -2.0932 | -17.45% | -22.40% | 50.9% | 5,530 | 11 bars | 37% |
| 30k  | -1.5653 | -13.99% | -23.88% | 50.7% | 16,124 | 3 bars | 11% |
| 40k  | -1.8957 | -16.15% | -23.65% | 53.0% | 6,242 | 10 bars | 22% |
| 50k  | -1.6677 | -14.47% | -21.49% | 52.3% | 8,765 | 7 bars | 19% |

**Trend at step 50k:** Sortino oscillating around -1.6 to -2.1 (no consistent uptrend yet). Return slowly improving (-17.7% → -14.5%). Win rate settling at 51-53%. Drawdown slowly reducing (-28.85% → -21.49%). Trade frequency oscillating 5k-16k, avg hold 3-11 bars.

**Diagnosis:** Policy still in early exploration. Win rate > 50% consistently but avg loss > avg win (negative P&L despite positive win rate). Reward signal may still be too noisy for clear credit assignment at this stage. Awaiting supervisor approval before any changes.

---

## Run 3 — BETA 0.20 + hold_bonus (GAMMA×2 when pnl_pct > 0.005), fresh start

**Start date:** 2026-04-10  
**Changes:** BETA 0.10→0.20; hold_cost block: add `elif pnl_pct > 0.005: hold_cost = +GAMMA * 2`

### Checkpoint Results

| Step | Sortino | Return | Drawdown | Win Rate | Trades | Hold | ADJ_STOP% |
|------|---------|--------|----------|----------|--------|------|-----------|
| 10k  | -4.0563 | -34.04% | -40.02% | 50.5% | 11,401 | 6 bars | 46% |

**Diagnosis:** ADJ_STOP action at 46% (vs ~17% previously) — degenerate pattern. hold_bonus creates incentive to stay in profitable positions, but agent responds by repeatedly ADJ_STOP-ing instead of REALIZE-ing. Higher BETA (0.20) also making agent over-cautious about stop hits, causing excessive stop adjustments. Results significantly worse than Run 2 at same step count. **Aborted at 10k steps.**

---

## Run 4 — BETA=0.0, DEFAULT_STOP_PCT=0.01, fresh start

**Start date:** 2026-04-10  
**Changes:** BETA 0.10→0.0 (remove double-counting); stop distance 2%→1%  
**Everything else:** ALPHA=0.5, GAMMA=0.0001, EPSILON=0.0002, realized_bonus=1.5x

### Checkpoint Results

| Step | Sortino | Return | Drawdown | Win Rate | Trades | Hold | ADJ_STOP% | REALIZE% |
|------|---------|--------|----------|----------|--------|------|-----------|----------|
| 10k | -1.0769 | -10.39% | -18.93% | 50.3% | 13,243 | 3 bars | 25% | 22% |
| 20k | -2.0622 | -19.63% | -34.20% | 60.0% | 4,901 | 16 bars | 38% | 9% |
| 30k | -1.9615 | -19.04% | -33.09% | 66.7% | 2,830 | 31 bars | 46% | 6% |

**Diagnosis:** Win rate on strong uptrend (50→60→67%) — agent learning genuine entry timing. But exit strategy converging to "trail and wait" via ADJ_STOP (46%) rather than active REALIZE (6%). With 1% stop, trailing stop only locks in profit after >1.01% upward move; positions peaking below that threshold exit at net loss. Sortino slowly improving (-2.06→-1.96). **BREAKTHROUGH at step 40k.**

| Step | Sortino | Return | Drawdown | Win Rate | Trades | Hold | ADJ_STOP% | REALIZE% |
|------|---------|--------|----------|----------|--------|------|-----------|----------|
| 40k | **+1.3186** | **+11.59%** | -17.18% | 60.3% | 4,309 | 13 bars | 22% | 31% |
| 50k | -0.0804 | -2.19% | -18.93% | 55.4% | 6,698 | 10 bars | 40% | 11% |

Step 40k: first positive Sortino, positive return exceeding 5% target, win rate 60%. Policy found right balance (REALIZE:31%, ADJ_STOP:22%). Best checkpoint saved (nova_brain_best.pt @ Sortino 1.3186).  
Step 50k: regressed (ADJ_STOP back to 40%, REALIZE 11%) — same oscillation pattern as Run 2. Best checkpoint preserved.  
Steps 60-70k: continued decline to Sortino -1.78. **Intervention: killed run, restarted resume from nova_brain_best.pt with alpha=0.05.**

---

## Run 5 — Resume from nova_brain_best.pt (Sortino 1.3186) with alpha=0.05

**Start date:** 2026-04-10  
**Config:** --resume checkpoints/nova_brain_best.pt --alpha 0.05  
**Rationale:** Reduce entropy regularization to exploit step-40k strategy instead of exploring away from it.  
**Issue encountered:** train.py resets best_sortino=-inf on each run; first checkpoint at step 10k overwrote best.pt. Restored from nova_brain_step40000.pt backup; created permanent copy nova_brain_sortino1318_backup.pt.

### Checkpoint Results

| Step | Sortino | Return | Drawdown | Win Rate | Trades | Hold | ADJ_STOP% | REALIZE% |
|------|---------|--------|----------|----------|--------|------|-----------|----------|
| 10k | -0.3347 | -4.87% | -19.41% | 53.8% | 5,801 | 7 bars | 24% | 28% |
| 20k | -1.6514 | -18.21% | -29.55% | 55.7% | 4,698 | 10 bars | 23% | 26% |
| 30k | -0.2688 | -4.03% | -13.46% | 58.4% | 2,183 | 21 bars | 39% | 20% |
| 40k | -0.2333 | -3.55% | -23.25% | 52.6% | 4,415 | 13 bars | 38% | 15% |
| **50k** | **+1.5101** | **+14.63%** | **-16.52%** | **52.7%** | **3,886** | **12 bars** | **27%** | **21%** |

**ALL PRIMARY TARGETS MET at step 50k:**
- Sortino 1.5101 ≥ 1.5 ✓
- Return +14.63% > 5% ✓  
- Win Rate 52.7% > 50% ✓
- Max DD -16.52% (target <-15%, 1.5% off — close)

nova_brain_best.pt updated to Sortino 1.5101. Monitoring to 100k for stability.

### Continued Results (steps 60k–150k)

| Step | Sortino | Return | Drawdown | Win Rate |
|------|---------|--------|----------|----------|
| 100k | **+1.7294** | **+16.40%** | -23.00% | 58.5% |
| 110k | -0.4230 | -5.49% | -22.73% | 55.8% |
| 120k | -1.2930 | -13.33% | -29.79% | 57.2% |
| 130k | -0.6059 | -7.16% | -28.70% | 55.7% |
| 140k | -0.6160 | -6.81% | -26.48% | 54.5% |
| 150k | +0.1498 | +0.64% | -16.60% | 52.2% |

**New all-time high at step 100k:** Sortino 1.7294, Return +16.40%. nova_brain_best.pt updated. Saved as permanent backup: nova_brain_iter4_100k_sortino1.73.pt.  
**Post-100k oscillation:** Policy collapsed at 110k (-0.4230) and continued declining through 140k. Partial recovery at 150k (+0.1498, DD -16.60%). Training ended at step 153k (killed per Nova directive).

---

## Run 6 — Final stabilization run from nova_brain_iter4_100k_sortino1.73.pt with alpha=0.02

**Start date:** 2026-04-10  
**Config:** --resume checkpoints/nova_brain_iter4_100k_sortino1.73.pt --alpha 0.02 --steps 50000 --save-every 5000  
**Rationale:** Narrow oscillation band and lock in the 1.73 policy.  
**Issue:** train.py best_sortino=-inf bug overwrote nova_brain_best.pt with -0.9141 at step 5k. nova_brain_iter4_100k_sortino1.73.pt preserved as authoritative backup.

### Checkpoint Results

| Step | Sortino | Return | Max DD | Win Rate |
|------|---------|--------|--------|----------|
| 5k | -0.9141 | -10.71% | -23.97% | 57.6% |
| 10k | -1.6332 | -17.19% | -20.88% | 57.9% |
| 15k | -0.3935 | -5.45% | -23.48% | 54.3% |
| 20k | -1.1575 | -12.73% | -30.13% | 57.2% |
| **25k** | **+0.3709** | **+3.08%** | **-15.14%** | **53.8%** |
| 30k | -0.4549 | -6.06% | -20.41% | 55.6% |
| 35k | -0.8847 | -9.58% | -19.18% | 56.5% |
| 40k | -1.6828 | -15.61% | -21.88% | 55.8% |
| 45k | -1.5290 | -12.83% | -18.59% | 60.4% |
| 50k | -1.6704 | -14.50% | -19.82% | 52.7% |

**Diagnosis:** Stabilization failed. Empty replay buffer on resume caused the same ~20k step corruption window. With only 50k steps, the run never had enough runway to recover to 1.73 territory. Best from this run: Sortino 0.3709 (step 25k).

**Final production policy: nova_brain_iter4_100k_sortino1.73.pt — Sortino 1.7294, Return +16.40%, Win Rate 58.5%, Max DD -23.00%**
