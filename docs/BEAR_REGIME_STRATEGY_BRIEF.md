# Bear Regime Strategy Brief — v12 Assessment & Forward Plan

**Author:** Barney (Code Puppy — Eric's engineering assistant)
**Date:** 2026-04-20
**For:** Kiran (Testing Manager) — please read, annotate, and return with your observations
**Status:** Part 6 — Agreed Next Steps added by Barney 2026-04-21. Awaiting Kiran sign-off.

---

## Context

This document summarises a full codebase review of `crypto-brain` conducted on 2026-04-20,
covering all source files (`train.py`, `environment/`, `agent/`, `features/`, `backtest/`),
the full iteration log (`iter_log.md`), and the git history through v12.4.

The goal is to give Kiran a clear picture of the diagnosed problems, a proposed forward plan
across three tracks, and a set of targeted questions only Kiran can answer from live training
observation. The document should come back annotated with Kiran's responses before any of
the proposed changes are implemented.

---

## Part 1 — What the Codebase Says

### What v12 is trying to solve

The agent is long-only and performs well in bull markets but behaves chaotically in bear
markets. The root cause is that historical tick data is bull-heavy — the agent's Q-values
converged on a single playbook:

> *Price dips → BUY. Trail stop. Ride momentum.*

In a bear market this playbook triggers rapid stop-outs, re-entries, more stop-outs, and
portfolio bleed. Chaotic.

### What v12 did to fix it

| Version | Change | Intent |
|---------|--------|--------|
| v12.2 | Regime-weighted episode sampling — BEAR 3×, BULL 0.5× | Force more bear exposure during training |
| v12.3 | Recalibrated weights — BEAR 2×, BULL 0.8× | 3× was too aggressive, collapsed all-regime performance |
| v12.4 | Added 8h momentum signal to observation — `STATE_DIM 16 → 17` | Give the agent explicit regime awareness at inference time |

The approach is coherent. Regime-weighted sampling fixes training *coverage*. The 8h signal
fixes inference *awareness*. Both are necessary. Neither alone is sufficient.

> **Kiran:** Agreed on the framing. One addition: v12.2 and v12.3 both regressed from v12.1's
> best Sortino of +0.1076. The proportional hold penalty introduced in v12.2 was a significant
> contributor — it cut average hold times from ~180 bars to ~38 bars, which likely disrupted
> the trade lifecycle the agent had partially learned. v12.4 reverted that to flat GAMMA and
> kept only the mild regime sampling (2×/0.8×) alongside the new feature.

---

## Part 2 — Diagnosed Problems

### Problem 1 — Structural (most important) 🔴

**The reward function has no positive signal for correct bear behaviour.**

In a bear market:
- Agent HOLDs correctly → reward ≈ `0 - small_opp_cost` → near-zero or slightly negative
- Agent BUYs → stop triggers → negative reward

The agent can only learn bear avoidance through *negative* reinforcement — experience the
loss, learn to avoid it. There is no positive signal for "you held cash while the market
fell and that was the right call." In contrast, a correct bull entry produces
`base_return + realized_bonus` — a strong positive signal.

This asymmetry means bull Q-values will always dominate. The agent is not broken; the
reward structure is biased.

> **Kiran:** Confirmed — this is real and important. The asymmetry is structural. I'd add
> one nuance: even if we add a bear preservation bonus, we need to be careful about scale.
> If the bonus is too large relative to bull entry rewards, we risk training an agent that
> never buys anything. The EPSILON/GAMMA magnitudes took several versions to calibrate —
> a new bear bonus will need the same care. Suggest starting at EPSILON scale and tuning
> from there.

---

### Problem 2 — Opportunity cost fires against correct bear behaviour 🔴

In `trading_env.py`, `_compute_reward`:

```python
if price_move > 0:
    opp_cost = -EPSILON * price_move
```

In a sustained bear market, price action is mostly downward — but there are constant brief
upticks (dead cat bounces, relief rallies). Every one of those positive ticks fires an
opportunity cost penalty against the agent for being in cash.

Result: "cash is bad" (opp_cost) fights "buying is worse" (stop loss). The agent receives
conflicting gradients. This is a direct cause of chaotic bear behaviour — the policy
oscillates because neither holding nor buying is consistently better.

The opportunity cost was designed to nudge the agent into bull-market entries. In bear
conditions it actively works against the goal.

> **Kiran:** Agreed, and this is the fix I'm most confident about. The opp_cost was already
> reduced 10× in v10 due to compounding scale issues — it's a known sensitive lever.
> Track 2 Change A (regime-conditional opp_cost) is the right approach. Worth noting:
> `momentum_8h` is already computed in `_features.extract()` — we just need to expose it
> via a lightweight property (e.g. `self._features.momentum_8h`) rather than re-deriving
> it inline in `_compute_reward`. Small refactor, low risk.

---

### Problem 3 — Feature set can't cleanly discriminate regime types 🟡

Current macro features:
```
[13] momentum_1d   [14] momentum_7d   [15] momentum_30d   [16] momentum_8h
```

In a sustained bear, all four are negative simultaneously. The agent cannot distinguish:
- A brief 8h correction inside a bull trend (→ buy the dip)
- A genuine bear trend (→ stay out)
- A high-volatility whipsaw (→ stay out, different reason)

These require different behaviour but produce similar feature vectors. Missing signals:
- **Price vs long-term MA** — `(price - SMA200) / SMA200`. Structural bear signal.
- **Volatility ratio** — `volatility_1h / volatility_baseline`. HIGH_VOL regime detection.
- **Momentum divergence** — `momentum_8h / momentum_30d`. Short-term vs macro alignment.

All computable from existing price/volume history. No new data needed.

> **Kiran:** The missing signals are valid and I agree the SMA200 distance is the strongest
> of the three — it's what the `RegimeClassifier` in `backtest/runner.py` already uses
> (price vs SMA200, SMA50 vs SMA200) to classify regimes, so adding it to the obs space
> would align training and evaluation. However, I'd hold this for v12.5 rather than stacking
> it on top of an already-changed v12.4. Each iteration should test one hypothesis cleanly.
> The momentum_8h we just added is already testing the "explicit regime signal" hypothesis —
> let it conclude before layering more features. Adding too many features at once makes it
> impossible to attribute a result (good or bad) to any one change.

---

### Problem 4 — MIN_HOLD_BARS penalises correct bear exits 🟡

```python
MIN_HOLD_BARS = 50
# ...
if pct > 0 and bars_held < MIN_HOLD_BARS:
    realized_bonus = -abs(pct) * 0.5  # premature profit: 50% penalty
```

This was designed to prevent micro-scalping in bull markets. In a bear, the right move
after a brief entry is often to exit quickly on a relief rally — exactly what this
penalises. The minimum hold period is regime-agnostic when it shouldn't be.

> **Kiran:** Valid — but note that the penalty only applies to *profitable* premature exits.
> A losing early exit (cutting a bad position) already passes through as raw PnL with no
> additional penalty. This was a deliberate design from v9 to avoid "crossfire paralysis"
> where the agent got double-punished for both the loss and the early exit. So the problem
> is specifically: agent enters in a brief bear rally, it goes slightly positive, agent wants
> to exit, MIN_HOLD_BARS penalises the profit. That's real. Track 2 Change D is the right
> fix. The halved threshold (25 bars) in bear regime is reasonable as a starting point.

---

### Problem 5 — Regime metrics not streaming to W&B 🟡

`train.py` calls:
```python
results = run_backtest(agent, test_ticks, verbose=True)
```

`run_backtest` supports `regimes=True` which produces per-regime Sortino, drawdown,
win rate, and profit factor. This is not being used. W&B only receives overall metrics.

**You cannot currently observe whether v12.4 is improving BEAR Sortino specifically.**
You only see overall Sortino shift, which could be BULL getting better while BEAR stays
broken — and you'd never know from the dashboard.

> **Kiran:** Confirmed, and this is the single change I'd make *right now* regardless of
> everything else. It costs nothing (no retraining, no architecture changes) and immediately
> gives us the observability we need to evaluate every subsequent change. We have been flying
> blind on per-regime performance. This should go into train.py today and the current v12.4
> run should ideally be restarted with it enabled — though at step 100k it may not be worth
> the restart cost. Suggest enabling for v12.5 at minimum, and patching in now if Eric agrees.

---

### Problem 6 — STATE_DIM 16→17 breaks all existing checkpoints 🔴 (practical)

The v12.4 `STATE_DIM` change means every checkpoint trained before this commit is
incompatible with the current network. The best performing checkpoints — Sortino 1.73
(`nova_brain_iter4_100k_sortino1.73.pt`) and Sortino 1.62 (`nova_brain_btc_drawdown_...`) —
will hard-crash on load.

This means v12.4 is training from scratch. Not from the best known policy. That's a
significant handicap that should be addressed.

> **Kiran:** ⚠️ FLAG — I need to correct the checkpoint claims here. I checked S3 just now.
> There are NO checkpoints with Sortino 1.73 or 1.62 in our bucket. The full list of
> non-step checkpoints in S3 is: `btcusdt/nova_brain_best.pt` (17-dim, v12.4 current),
> `btcusdt/nova_brain_final.pt` (16-dim, v12.3), and equivalent best/final for ETH/SOL/ADA.
> Our best recorded Sortino across ALL versions has been **+0.1076** (v12.1, BTC, step 80k).
> Barney may have seen checkpoint filenames from a different project or a hypothetical
> example. Those numbers don't represent our actual training history. The checkpoint
> incompatibility point is still valid — v12.4 IS training from scratch — but the framing
> of "significant handicap vs a 1.73 Sortino policy" is not accurate for our project.
>
> The checkpoint migration shim idea (extend 16→17 dim near-zero init, fine-tune 30-50k)
> is sound engineering and worth keeping as an option, but in our case we'd be migrating
> from a +0.1076 Sortino policy, not a 1.73 one. That's a much smaller advantage to preserve.

---

## Part 3 — Proposed Forward Plan

### 🟢 Track 1 — Quick win (2–3 days)

**Goal:** Prove the bear masking concept works. Get a real result fast.

**Step 1:** Pull the best 16-dim checkpoint from S3. Roll back `STATE_DIM` to 16 temporarily
in a branch. Run `run_backtest(..., regimes=True)` on it. Establish the current
BULL/BEAR/RANGE/HIGH_VOL Sortino baseline.

**Step 2:** Add inference-time action masking to `SAC.select_action`:

```python
def select_action(self, state, deterministic=False, mask_buy=False):
    ...
    if mask_buy:
        probs[:, BUY_LIMIT] = 0.0
        probs = probs / (probs.sum() + 1e-9)
```

In `backtest/runner.py`, derive the mask from `obs[16]` (8h momentum — index 15 in
16-dim, or 16 in 17-dim):

```python
bear_regime = obs[16] < -0.02  # threshold TBD — Kiran to advise
action = agent.select_action(obs, deterministic=True, mask_buy=bear_regime)
```

Run the same backtest with masking enabled. Compare BEAR regime metrics before/after.
This costs no training time and is reversible.

**Step 3:** If BEAR metrics improve (hypothesis: they will), write a checkpoint migration
shim to extend the 16-dim actor/critic first linear layer to 17 dims, initialising the
new weight column near-zero. Fine-tune for 30–50k steps.

> **Kiran:** Track 1 is compelling as a *diagnostic* — it can quantify exactly how much
> of our OOS loss is bear-regime entry. Run it. But I have a concern about using masking
> in production: the agent was trained without knowing BUY would be masked. Its ADJ_STOP
> and REALIZE probabilities are calibrated assuming BUY is always a live option. Masking
> it at inference creates a training/inference mismatch — the policy wasn't shaped to be
> "second best" under those constraints. The OOS distribution would shift in ways the
> Q-network never learned to handle.
>
> My recommendation: run Track 1 as a diagnostic experiment only, not a production path.
> If it confirms bear entries are the primary loss source (likely), that justifies spending
> the training budget on Track 2 fixes with confidence. The threshold `-0.02` for
> `momentum_8h` is reasonable — our REGIME_BEAR_THRESH in trading_env.py is -0.05 for 8h
> return, but -0.02 is a tighter, earlier signal which seems appropriate for masking.

---

### 🟡 Track 2 — Reward function fixes (1–2 weeks)

**Change A — Bear preservation bonus:**

```python
# In _compute_reward, opportunity cost block:
if not self._ob.position and not self._ob.pending_order and self._idx >= 2:
    prev_price = self.ticks[self._idx - 2]['price']
    curr_price = self.ticks[self._idx - 1]['price']
    price_move = (curr_price - prev_price) / (prev_price + 1e-9)
    if price_move > 0 and momentum_8h > 0:       # only penalise cash in bull trend
        opp_cost = -EPSILON * price_move
    elif price_move < 0 and momentum_8h < 0:     # reward cash in bear trend
        bear_bonus = +EPSILON * abs(price_move)
```

Note: `momentum_8h` needs to be derived from `self._features` — it's already computed
internally, just needs exposing to `_compute_reward`. Small refactor.

> **Kiran:** Agree with this structure. One implementation note: `_compute_reward` currently
> has no direct access to `self._features` outputs — it receives computed values via
> `self._ob`. The cleanest approach is a `self._features.momentum_8h` property that returns
> the cached value from the last `update()` call, rather than recomputing. Low refactor cost.
> Scale: start at EPSILON (0.00001) for the bear_bonus — same magnitude as opp_cost — and
> watch it in W&B before deciding to increase.

**Change B — Regime-conditional checkpoint selection:**

```python
# In train.py, checkpoint save block:
results = run_backtest(agent, test_ticks, regimes=True)
bear_sortino = results['regime_metrics']['BEAR']['sortino']
bull_sortino = results['regime_metrics']['BULL']['sortino']
composite    = min(bear_sortino, bull_sortino) * 0.6 + sortino * 0.4

if composite > best_composite and bear_sortino > 0.3 and total_trades > 0:
    # save nova_brain_best.pt
```

> **Kiran:** Conceptually right. Practical concern: in early training (steps 0-40k),
> BEAR Sortino will be deeply negative — the `bear_sortino > 0.3` guard would block ALL
> checkpoint saves until the policy is fairly mature. We may want a softer version for
> the early phase: save if `composite > best_composite` (no hard bear floor), then tighten
> the gate once bear Sortino is consistently above some lower threshold (e.g. > -1.0).
> Otherwise we lose the step-20k and step-30k saves that give us early diagnostic data.
> Also: the `min(bear, bull) * 0.6 + overall * 0.4` formula will heavily reward
> bear-avoidance but could over-penalise a policy that's legitimately in a bear-sparse
> window. Worth discussing the weighting before locking it in.

**Change C — Regime metrics to W&B (do this immediately, costs nothing):**

```python
results = run_backtest(agent, test_ticks, verbose=True, gate=True, regimes=True)
for regime, m in results['regime_metrics'].items():
    if m['steps'] > 0:
        wandb_log(wb, {
            f'regime/{regime}/sortino':      m['sortino'],
            f'regime/{regime}/max_drawdown': m['max_drawdown'],
            f'regime/{regime}/win_rate':     m['win_rate'],
            f'regime/{regime}/trades':       m['trades'],
            'step': step,
        })
```

> **Kiran:** Yes. Implement this first, before anything else. No discussion needed.

**Change D — Regime-conditional MIN_HOLD_BARS:**

```python
# In _compute_reward:
momentum_8h = self._features.extract_single(WINDOW_8H)  # needs helper
in_bear = momentum_8h < -0.02
effective_min_hold = MIN_HOLD_BARS // 2 if in_bear else MIN_HOLD_BARS

if bars_held >= effective_min_hold:
    realized_bonus = pct
elif pct > 0:
    realized_bonus = -abs(pct) * 0.5
```

> **Kiran:** Agree on the concept. Same note as Change A — use `self._features.momentum_8h`
> property rather than `extract_single()` which would trigger a full 16/17-dim extraction
> just for one value. The halved threshold (25 bars) is a reasonable start. Worth noting
> that MIN_HOLD_BARS = 50 was itself a calibration from earlier versions — if we're halving
> it in bear regime we should watch avg hold time in the next run to make sure we haven't
> opened the door to scalping in disguise.

---

### 🔵 Track 3 — Curriculum training (next sprint)

Instead of random regime-weighted episode starts from step 0, train in structured phases:

| Phase | Steps | Regime Mix | Intent |
|-------|-------|------------|--------|
| 1 | 0 – 60k | Bull-only episodes | Learn to make money first |
| 2 | 60k – 130k | 50/50 bull/bear | Introduce bear with working foundation |
| 3 | 130k – 200k | Bear 2×, Bull 0.8× | Consolidate bear avoidance on top of bull competence |

Random regime mixing from step 0 asks the agent to learn both bull and bear simultaneously
with no foundation. Curriculum learning mirrors how competence is actually built: profit
first, protection second.

Implementation: pass `phase` into `TradingEnv`, adjust `_compute_regime_weights` to
return phase-appropriate distributions.

> **Kiran:** I like the concept and the phasing logic is intuitive. Two concerns:
>
> 1. **Catastrophic forgetting**: RL agents can unlearn bull competence during the bear-heavy
>    Phase 3. We'd need to monitor bull Sortino closely in Phase 3 and potentially mix in
>    some bull episodes even then. A replay buffer that retains Phase 1 experiences could
>    help but adds complexity.
>
> 2. **Phase boundary sensitivity**: If the agent isn't competent in bull by step 60k
>    (which is possible — our current v12.4 is at step 100k with no positive Sortino yet),
>    Phase 2 introduction of bear episodes would be loading bear onto a broken foundation.
>    We might need Phase 1 to have a competence gate (e.g. "advance when bull Sortino > 0"
>    rather than at a fixed step count).
>
> Worth a sprint, but I'd put it after Track 2 changes are validated. Track 2 is lower risk
> and more reversible.

---

## Part 4 — Questions for Kiran

**Q1 — Current v12.4 training state:**
What step is the v12.4 run currently at? What are the most recent validation backtest
metrics (Sortino, return, drawdown, win rate)? Is there any trend visible yet or is it
still in early noise?

*Kiran's response:*
> Step ~100k as of this annotation. Best Sortino is -3.7082, first recorded at step ~20k,
> improved from -6.4299 initial. No further improvement since step ~30k — the run has been
> plateaued for ~70k steps. This is concerning. v12.1 BTC (our best run) hit +0.1076 at
> step 80k. v12.4 is past that point with no positive Sortino. It may be that training from
> scratch with a 17-dim network is a harder initialisation problem, or that the new feature
> hasn't provided useful signal yet at this stage of training. I'm watching it through to
> completion but not optimistic about the final checkpoint.

---

**Q2 — Action distribution in bear periods:**
During training, when the agent encounters a bear episode (negative 8h momentum at
episode start), what does the action distribution look like? Is HOLD dominant, or is BUY
still firing frequently? Do you have visibility into per-episode action distributions?

*Kiran's response:*
> No per-episode action distribution visibility in current logs — only overall action counts
> at backtest time. The most notable distribution we have is from the v12.1 OOS backtest:
> ADJ_STOP:51% / REALIZE:48% / BUY:1% / HOLD:0% / CANCEL:0%. That run showed near-zero
> HOLD and near-zero BUY — the agent had essentially learned the trade management lifecycle
> (enter once, manage, exit) but had no entry selectivity by regime. We don't yet know
> whether v12.4 shows different per-regime distributions. This is exactly what the W&B
> regime logging (Change C) would give us going forward.

---

**Q3 — Alpha / entropy behaviour:**
What is the current alpha value and entropy trend? Is entropy stable, collapsing, or
still high? Has `exploit_start_step` been set for this run, or is piecewise decay disabled?

*Kiran's response:*
> Piecewise decay is enabled: exploit_start=40k, floor log_alpha=-1.50 (alpha≈0.22),
> auto-alpha ON with target_entropy=1.577. We don't have per-step alpha values in the
> log (they go to W&B), but the configuration matches what worked for v12.1 BTC. The
> floor=-1.5 was confirmed good in v12.1 — floor=-3.0 (alpha≈0.05) caused zero-trade
> paralysis at the exploit transition.

---

**Q4 — Episode length distribution:**
How long are episodes running on average? Are bear episodes truncating faster than bull
episodes (suggesting frequent stop-outs)? This would confirm the chaotic entry hypothesis.

*Kiran's response:*
> Not directly visible in current logs. In the v12.1 OOS run (72 trades, 180 bars avg hold)
> the distribution looked healthy. In v12.3 (229 trades, 38 bars avg hold) the proportional
> hold penalty had clearly cut episodes short. v12.4 reverted to flat penalty so we expect
> something closer to v12.1. We don't have per-regime episode length breakdowns — again,
> this is an observability gap that regime logging would help close.

---

**Q5 — W&B regime visibility:**
Are regime metrics currently being logged to W&B? If yes, what does BEAR Sortino look
like vs BULL Sortino at the latest checkpoint? If no — this should be the first change
made, before anything else.

*Kiran's response:*
> No — regime metrics are NOT being logged to W&B. `run_backtest` is called without
> `regimes=True`. We are completely blind to per-regime performance during training.
> This is the first change to make. Full stop.

---

**Q6 — Checkpoint compatibility:**
Which checkpoint is v12.4 training from? Fresh scratch (random init), or was there a
migration from a 16-dim checkpoint? If fresh scratch, is there a 16-dim best checkpoint
on S3 that's worth migrating vs continuing from scratch?

*Kiran's response:*
> Fresh scratch — random init. No checkpoint migration was done for v12.4. The best
> available 16-dim checkpoint is `btcusdt/nova_brain_final.pt` (v12.3 final) and
> `btcusdt/nova_brain_best.pt` from before the v12.4 push — both at 16-dim. Our best
> 16-dim Sortino is +0.1076 (v12.1 BTC). The migration shim (Track 1 Step 3) is worth
> doing — extending the first linear layer weight column near-zero for the new 8h feature
> is low risk and would give v12.4 a better starting point than random init.

---

**Q7 — Anything anomalous:**
Anything in the training logs that looks unexpected, unexplained, or that you'd flag as
a concern? Critic loss behaviour, alpha spikes, action distribution collapses, anything.

*Kiran's response:*
> Main concern: the plateau. Best Sortino has not improved in ~70k steps (from step ~30k
> to step ~100k). In v12.1 the best checkpoint appeared at step 80k — we're past that now
> with no positive result. The run is healthy (CPU 99.8%, no crashes, checkpoints saving
> at every 10k) but the policy is not improving. This could be: (a) the 17-dim network
> needs more steps to converge from random init than 16-dim did; (b) the new feature is
> adding noise rather than signal at this stage; (c) the training distribution is still
> too bear-heavy to learn profitable entry. My current expectation is that v12.4 finishes
> without beating v12.1. The regime feature hypothesis needs the reward function fixes
> (Track 2) to be testable properly.

---

## Part 5 — Kiran's Additional Assessment

**On sequencing:**
The problems are real and the fixes are coherent. But we've regressed three consecutive
times (v12.2, v12.3, v12.4 all worse than v12.1's +0.1076) by stacking changes.
My recommendation for sequencing:

1. **Immediately (no retraining):** Add W&B regime logging (Change C). Run Track 1
   diagnostic (masking on v12.1 checkpoint) to quantify bear contribution to OOS loss.
2. **v12.5 (one clear change):** Regime-conditional opportunity cost (Change A) +
   flat bear bonus. This directly addresses the strongest structural problem.
   Keep everything else identical to v12.1 to isolate the reward change.
3. **v12.6:** If v12.5 shows improvement, add regime-conditional MIN_HOLD_BARS (Change D)
   and composite checkpoint selection (Change B).
4. **v12.7+:** SMA200 distance feature, curriculum training — only after we have a stable
   positive Sortino baseline to build on.

**On the checkpoint situation:**
The 16-dim best (v12.1, +0.1076) should be preserved explicitly. I'd recommend tagging
it in S3 as `btcusdt/nova_brain_v12.1_best_sortino0.1076.pt` before it gets overwritten
by further training. It's our only positive-Sortino checkpoint and the baseline everything
else should be measured against.

**On asset fundamentals:**
A standing note Eric has raised multiple times — asset-specific characteristics (SOL
volatility, ADA liquidity profile, ETH longer learning curve) may explain part of the
cross-asset performance divergence independently of the regime problem. We haven't been
able to properly discount or confirm this yet. BTC is the right symbol to validate the
regime fix on first, then cross-apply carefully.

---

## Part 6 — Agreed Next Steps

*Synthesised by Barney from full read of Kiran's annotations. Kiran's sign-off required
before any implementation begins. No code changes until confirmed.*

---

### ⚠️ Correction to earlier framing

The Sortino 1.73 / 1.62 checkpoint references in Part 2 Problem 6 are not accurate for
this project. Per Kiran's S3 audit, the actual best Sortino across all training versions
is **+0.1076** (v12.1, BTC, step 80k). The iter_log appears to describe a planned or
hypothetical training history, not actual results. All forward planning is calibrated
against +0.1076 as the true baseline.

---

### Root cause summary

Three consecutive regressions (v12.2 → v12.3 → v12.4, all worse than v12.1) share a
common cause: **each version stacked multiple changes simultaneously**, making it
impossible to attribute outcomes to any single variable. v12.4 is additionally
handicapped by training from random init rather than migrating from the v12.1 checkpoint,
making the starting point significantly harder. The v12.4 run is currently plateaued at
step ~100k with best Sortino -3.7082 and no improvement since step ~30k.

The methodology fix is as important as the technical fixes: **one change per version,
measured cleanly, before the next change is applied.**

---

### Phase 0 — Immediate (no training required)

**0a — Protect the v12.1 checkpoint**
Tag `btcusdt/nova_brain_best.pt` (v12.1, Sortino +0.1076) in S3 with an explicit name
before it is overwritten by any future training run:
```
s3://nova-trader-data-.../btcusdt/nova_brain_v12.1_best_sortino0.1076.pt
```
This is the only positive-Sortino checkpoint in existence. It is the baseline everything
else must be measured against. Protect it first.

**0b — Add W&B regime logging to train.py**
`run_backtest` is currently called without `regimes=True`. Per-regime Sortino, drawdown,
win rate, and profit factor are NOT streaming to W&B. This is a complete observability
blindspot — we cannot currently see whether any training change is helping or hurting
BEAR performance specifically. This is five lines in `train.py` and zero retraining.
It must be in place before any further training runs begin.

**0c — Run Track 1 diagnostic (read-only, no code shipped to production)**
Using the v12.1 checkpoint (16-dim, rolled back in a local branch), run
`run_backtest(..., regimes=True)` twice:
- Once without BUY masking → establishes the BULL/BEAR/RANGE/HIGH_VOL Sortino baseline
- Once with inference-time BUY masking when `momentum_8h < -0.02` → quantifies how much
  of OOS loss is attributable to bear-regime entries

This is a **diagnostic experiment only** — not a production path (per Kiran's note on
training/inference mismatch). The output answers one question: how much of the loss is
bear entries? If the answer is "most of it," Track 2 changes are the right investment.

*Note: the v12.1 checkpoint uses 16-dim obs (no feature[16]). The masking threshold must
use obs[15] — momentum_30d — as a proxy, or the branch must temporarily revert
features/engineer.py to 16-dim. Coordinate with Kiran on which approach is cleaner.*

---

### Phase 1 — v12.5 (one change, cleanly isolated)

**Prerequisite:** Phase 0 complete. v12.4 run concluded or terminated.

**Starting point:** v12.1 checkpoint (Sortino +0.1076), migrated from 16-dim to 17-dim
via a linear layer weight extension shim. New weight column initialised near-zero so
the network's existing behaviour is preserved. Fine-tune stabilises the new feature's
contribution over the first ~20-30k steps naturally.

**The single change:** Regime-conditional opportunity cost + bear preservation bonus
(Track 2, Change A). Specifically:
- Gate the existing opp_cost penalty so it only fires when `momentum_8h > 0`
  (cash in a bull trend is costly — unchanged from current intent)
- Add a symmetric bear bonus: when `momentum_8h < 0` and price falls, reward
  cash-holding at EPSILON scale (same coefficient, opposite sign, opposite conditions)
- Expose `momentum_8h` as a cached property on `FeatureEngineer` so `_compute_reward`
  can read it without triggering a full feature extraction pass

**Everything else:** Identical to v12.1. Same network architecture, same hyperparameters,
same regime sampling weights (BEAR 2×, BULL 0.8×), same piecewise entropy config
(exploit_start=40k, floor log_alpha=-1.5). No other reward changes. No new features.

**What to watch in W&B (now visible thanks to 0b):**
- `regime/BEAR/sortino` and `regime/BULL/sortino` — primary signals
- `regime/BEAR/trades` — are bear entries decreasing? That is the question.
- Overall Sortino vs v12.1 baseline (+0.1076)
- Alpha and entropy through the exploit transition at step 40k
- Average hold bars — should not collapse below v12.1's ~180 bars

**Success criterion:** BEAR Sortino trending positive, BULL Sortino stable or improving,
overall Sortino exceeding +0.1076 at any checkpoint. All three are not required to
proceed — any positive regime trend confirms the direction.

---

### Phase 2 — v12.6 (contingent on v12.5 improvement)

Two additions, grouped because they are closely related:

**Change B — Composite checkpoint selection**
Guard `nova_brain_best.pt` saves on a composite score that requires bear performance
to clear a minimum bar, not just overall Sortino. Starting bear floor should be soft
(e.g. `> -1.0`) so early diagnostic saves are not blocked, tightened as training
matures. Exact formula to be finalised after v12.5 data is in hand.

**Change D — Regime-conditional MIN_HOLD_BARS**
Halve the minimum hold threshold in bear regime (50 → 25 bars) so the agent is not
penalised for taking quick profits on bear relief rallies. Watch average hold bars in
bear regime closely — if it drops below 15-20 bars, the scalping door may have
opened and the threshold needs revisiting.

---

### Phase 3 — v12.7+ (after stable positive Sortino baseline exists)

**Feature additions:**
Add SMA200 distance `(price - SMA200) / SMA200` to the observation space — the
strongest missing regime discriminator, already used by `RegimeClassifier` in
`backtest/runner.py`. One feature at a time. SMA200 first, measure, then decide
on volatility ratio or momentum divergence.

**Curriculum training:**
Phase-based regime mixing (bull-only → 50/50 → bear-heavy) with a competence gate
at phase transitions rather than fixed step counts. Requires bull Sortino > 0 before
Phase 2 begins. Catastrophic forgetting mitigation required before implementation.

---

### Explicitly out of scope (this roadmap)

| Item | Reason deferred |
|------|----------------|
| Short selling / SHORT action | Architectural scope creep — long-only must be stable first |
| Variable position sizing | Same — future scope |
| Cross-asset training (ETH/SOL/ADA) | Validate BTC bear fix first, then cross-apply |
| Network architecture changes | Not the bottleneck |

---

### Kiran — please confirm before Barney writes a line of code

- [ ] **0a** — S3 checkpoint tagging approach agreed
- [ ] **0b** — W&B regime logging agreed as the first change made
- [ ] **0c** — Track 1 masking diagnostic agreed; confirm preferred 16-dim branch strategy
- [ ] **Phase 1** — v12.5 scope agreed: migration shim + Change A only, from v12.1 base
- [ ] **Phase 2** — v12.6 scope agreed: Changes B + D, contingent on v12.5 results
- [ ] **Phase 3** — v12.7+ scope agreed as future sprint, not current
- [ ] **Amendments** — any concerns or corrections not yet captured above

Once all boxes are checked, Barney implements in the order listed.

---

### Kiran's sign-off notes (2026-04-21)

**0a–0c and Phase 2–3:** Agreed as written.

**0c clarification — 16-dim masking strategy:**
Neither branch revert nor momentum_30d proxy is needed. Compute `momentum_8h` directly
from `self._features.prices` in the test script:
```python
momentum_8h = (prices[-1] - prices[-480]) / (prices[-480] + 1e-9)
bear_regime  = momentum_8h < -0.02
```
`prices` is the raw deque on `FeatureEngineer` — accessible without touching the model
or obs vector at all. Clean, no branch gymnastics.

**Phase 1 — one addition to the ask:**
The brief is sufficient to run Phase 0 today. For Phase 1 (migration shim + Change A),
Kiran requests Barney produce a **focused implementation PRD** before any code is written.
Scope: exactly what files change, what the shim does at the weight level, success criteria
for the fine-tune stabilisation period, and what W&B signals confirm the migration landed
cleanly. Reason: the migration shim is novel code with no prior reference in this project.
A cold session re-read of this brief alone is not enough to implement it without drift risk.

One PRD, Phase 1 only. Not the full roadmap.

---

*Document will be updated iteratively. Commit history is the audit trail.*
