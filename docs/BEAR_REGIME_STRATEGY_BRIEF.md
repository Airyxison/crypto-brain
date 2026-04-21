# Bear Regime Strategy Brief — v12 Assessment & Forward Plan

**Author:** Barney (Code Puppy — Eric's engineering assistant)
**Date:** 2026-04-20
**For:** Kiran (Testing Manager) — please read, annotate, and return with your observations
**Status:** Awaiting Kiran's input

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

---

### Problem 6 — STATE_DIM 16→17 breaks all existing checkpoints 🔴 (practical)

The v12.4 `STATE_DIM` change means every checkpoint trained before this commit is
incompatible with the current network. The best performing checkpoints — Sortino 1.73
(`nova_brain_iter4_100k_sortino1.73.pt`) and Sortino 1.62 (`nova_brain_btc_drawdown_...`) —
will hard-crash on load.

This means v12.4 is training from scratch. Not from the best known policy. That's a
significant handicap that should be addressed.

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

This prevents a bear-chaotic checkpoint from being saved as "best" just because it had
a good bull window. The `bear_sortino > 0.3` guard is a minimum bar — tune as needed.

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

---

## Part 4 — Questions for Kiran

*Kiran — please answer these from your live training observations. Your answers will
determine which of the above changes to prioritise and in what order.*

---

**Q1 — Current v12.4 training state:**
What step is the v12.4 run currently at? What are the most recent validation backtest
metrics (Sortino, return, drawdown, win rate)? Is there any trend visible yet or is it
still in early noise?

*Kiran's response:*
> [please fill in]

---

**Q2 — Action distribution in bear periods:**
During training, when the agent encounters a bear episode (negative 8h momentum at
episode start), what does the action distribution look like? Is HOLD dominant, or is BUY
still firing frequently? Do you have visibility into per-episode action distributions?

*Kiran's response:*
> [please fill in]

---

**Q3 — Alpha / entropy behaviour:**
What is the current alpha value and entropy trend? Is entropy stable, collapsing, or
still high? Has `exploit_start_step` been set for this run, or is piecewise decay disabled?

*Kiran's response:*
> [please fill in]

---

**Q4 — Episode length distribution:**
How long are episodes running on average? Are bear episodes truncating faster than bull
episodes (suggesting frequent stop-outs)? This would confirm the chaotic entry hypothesis.

*Kiran's response:*
> [please fill in]

---

**Q5 — W&B regime visibility:**
Are regime metrics currently being logged to W&B? If yes, what does BEAR Sortino look
like vs BULL Sortino at the latest checkpoint? If no — this should be the first change
made, before anything else.

*Kiran's response:*
> [please fill in]

---

**Q6 — Checkpoint compatibility:**
Which checkpoint is v12.4 training from? Fresh scratch (random init), or was there a
migration from a 16-dim checkpoint? If fresh scratch, is there a 16-dim best checkpoint
on S3 that's worth migrating vs continuing from scratch?

*Kiran's response:*
> [please fill in]

---

**Q7 — Anything anomalous:**
Anything in the training logs that looks unexpected, unexplained, or that you'd flag as
a concern? Critic loss behaviour, alpha spikes, action distribution collapses, anything.
Your live observation will catch things static analysis cannot.

*Kiran's response:*
> [please fill in]

---

## Part 5 — Kiran's Additional Assessment

*Kiran — if there are problems, patterns, or opportunities you've observed that are not
addressed above, please document them here. This section is yours.*

---

*[Kiran's additions]*

---

## Part 6 — Agreed Next Steps

*To be filled in after Kiran's response is reviewed by Eric and Barney.*

- [ ] TBD
- [ ] TBD
- [ ] TBD

---

*Document will be updated iteratively. Commit history is the audit trail.*
