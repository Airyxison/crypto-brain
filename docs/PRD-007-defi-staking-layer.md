# DeFi Staking Layer
## Nova Trader — PRD-007

**Status:** Future Consideration — Stub  
**Depends on:** PRD-006 (asset research), PRD-004 (live execution)  
**Prerequisites:** Spot trading layer stable and profitable in live conditions

---

## Injection Sequencing — When This Gets Built

The stablecoin yield layer must be injected **after** the spot model demonstrates it can reliably identify and act on market moves. The sequencing is non-negotiable:

**1. Spot model learns to see moves** *(current work)*  
The model identifies entries, holds appropriately, exits profitably. No yield comparison yet — adding it before the model has edge would teach "never trade" (can't beat guaranteed yield when you have no reliable signal).

**2. Stablecoin yield as a pre-trade gate** *(Phase 0 — inject here first)*  
Before firing a trade: *"Is the expected return on this trade better than holding USDC for the same duration?"*  
- USDC/USDT earning ~4-5% APY (tracks Fed funds rate) becomes the risk-free baseline
- Idle cash earns yield in the order book simulator (small per-bar credit)
- Reward function reframes opportunity cost: not "are you missing the price move?" but "are you beating stablecoin yield?"
- This makes `MIN_HOLD_BARS` enforcement natural rather than artificial — a 2-bar trade making 0.2% doesn't clear the yield hurdle; the model learns this without being told
- Implementation: `STABLECOIN_APY` constant in `order_book.py`, cash earns per-bar yield when idle, reward baseline shifts accordingly

**3. Yield-bearing stablecoin as an explicit action** *(Phase 1 — flexible staking)*  
The model can actively choose: stay in USDC earning yield vs. deploy into a trade. ADA native staking (no unbonding) is the cleanest first vehicle.

**4. Lock-in staking** *(Phase 2+)*  
See below.

---

## Overview

This PRD describes a second trading layer that sits above the spot trading model. Where the spot model decides *whether to hold an asset*, the staking layer decides *whether the yield from locking that asset justifies surrendering liquidity for a defined period*.

These two decisions interact and cannot be treated independently. A staked position cannot be exited if the spot model signals sell. The staking layer must therefore model both the expected yield and the full illiquidity window — including unbonding — before committing capital.

---

## The Core Problem

Lock-in staking creates a position with the same mathematical structure as a **covered call**:

- You sell optionality (the right to exit) in exchange for yield (the premium)
- Break-even condition: price must not fall more than the yield earned over the full illiquidity window
- Example: 15% APY for 28-day lock ≈ 1.15% yield. If price drops 5% during the lock, net result is -3.85% despite "earning yield"

### The Two-Phase Illiquidity Problem

Most traders model only Phase 1. Phase 2 is where unexpected losses occur.

**Phase 1 — Lock period**
- Duration: stated lock-in term (e.g., 28 days)
- Yield: accruing per protocol rate
- Price exposure: full
- Exit: not possible

**Phase 2 — Unbonding period**
- Duration: chain/protocol-specific, often unstated prominently
  - ETH validator: hours to days (queue-dependent)
  - Cosmos-based chains: 21 days standard
  - Curve veCRV: up to 4 years, no early exit
  - Cardano: flexible (no unbonding — notable exception)
- Yield: none (tokens in transit)
- Price exposure: full
- Exit: not possible

**The model must use total illiquidity duration = lock period + unbonding period**, not just the advertised lock period. A 28-day lock with a 7-day unbond is a 35-day position. The expected return calculation must reflect this.

---

## Yield as a Stochastic Variable

Protocol APYs are not fixed. They are determined by supply and demand for liquidity in the pool:

- When few participants stake → APY is high (protocol incentivizing liquidity)
- When many participants stake → APY compresses (yield diluted across more capital)

A protocol advertising 15% today may yield 4% by day 28 because other participants have piled in. The model must:
1. Treat yield rate as a variable, not a constant
2. Learn the historical distribution of yield compression for each protocol
3. Forecast expected yield over the lock period, not assume the current rate persists

---

## Staking Types and Risk Profiles

Different staking mechanisms have fundamentally different risk structures. The model must distinguish between them.

### Single-Asset Staking
- Stake one token, earn yield in same token (e.g., ETH staking → stETH)
- Risk: price exposure only + smart contract risk
- No impermanent loss
- Most tractable for the model — closest to the spot trading structure

### Liquid Staking
- Receive a liquid derivative token (stETH, rETH, etc.) representing the staked position
- Can be sold on secondary market at a potential discount to underlying
- Eliminates Phase 2 unbonding risk but introduces **liquid staking discount risk**
- Enables staking while maintaining some exit optionality

### Liquidity Pool (LP) Staking
- Provide two assets as a pair (e.g., ETH/USDC) to a DEX pool
- Earn trading fees + protocol incentives
- Exposed to **impermanent loss** — if the price ratio between the two assets changes, the LP position underperforms simply holding both assets
- IL formula: if ETH appreciates 50% vs. USDC, the LP position has ~5.7% IL vs. holding
- In effect: LP staking is **short volatility**. High volatility = high IL. The model is not just predicting price direction — it's predicting realized volatility over the lock period.
- Most complex risk structure. Do not model until single-asset staking is stable.

### Fixed-Term Protocol Staking (veCRV style)
- Lock tokens for a defined period (weeks to years) for amplified governance rights + yield
- No early exit under any circumstances
- Yield rate scales with lock duration
- Requires long-horizon price forecasting — outside current model scope until episode structure extends to match

---

## Protocol Risk

All DeFi staking carries smart contract risk. Historical examples:
- **Celsius (2022):** Centralized yield platform, froze withdrawals, went bankrupt. Users lost principal.
- **Terra/Luna (2022):** Algorithmic stablecoin staking offering 20% APY collapsed to zero in 72 hours.
- **Euler Finance (2023):** $197M exploit, funds partially recovered after negotiation.
- **Multiple smaller protocols:** Rug pulls, admin key exploits, oracle manipulation attacks.

Protocol risk is not modelable from price data alone. It requires:
- Protocol audit history and audit quality
- Team doxxing / anonymity
- TVL history and stability (rapid TVL flight = risk signal)
- Smart contract age (older, battle-tested contracts lower risk)
- Insurance protocol coverage (e.g., Nexus Mutual)

**Architectural implication:** Protocol risk must be a manual input or curated allowlist — not learned from price data. The model should only stake into pre-approved, audited protocols. Risk classification is a human/curation problem, not an ML problem.

---

## Required Model Changes

### New Actions (extending the 5-action space)

Current action space: `HOLD, BUY_LIMIT, ADJUST_STOP, REALIZE_GAIN, CANCEL_ORDER`

Staking layer additions:
```
STAKE           — commit current position to staking at current protocol rate
INITIATE_UNSTAKE — begin the unstaking process (starts unbonding clock)
CLAIM_YIELD     — harvest accrued yield without closing position (where supported)
```

Note: `STAKE` is only valid when `in_position = True` and `staking_available = True`.  
`INITIATE_UNSTAKE` is only valid when `in_stake = True`.  
During unbonding, no actions are valid on the staked portion — position is frozen.

### New State Features

```
[16] in_stake              — 0.0 or 1.0
[17] time_until_unlock     — bars remaining in lock period, normalized 0-1
[18] unbonding_remaining   — bars remaining in unbonding period, normalized 0-1
[19] current_yield_rate    — protocol APY at this moment, normalized
[20] accrued_yield_pct     — yield earned so far as fraction of cost basis
[21] total_illiquidity_pct — (lock_remaining + unbonding_total) / total_illiquidity, 0-1
```

State vector grows: 16 → 22 dimensions (when staking layer active).

### New Reward Components

Current reward: base return + drawdown penalty + hold cost + opportunity cost + realized bonus

Staking additions:
```
yield_accrual   — per-bar yield credit during lock period (reduces opportunity cost)
unstake_penalty — penalty for initiating unstake before lock expires (if applicable)
il_cost         — impermanent loss cost for LP positions (if applicable)
```

### New Episode Structure

Current max episode: 200 steps (~3.3 hours of 1-min data)

Staking requires episodes long enough to span a full lock + unbonding cycle:
- Minimum for 28-day lock + 7-day unbond: 35 days = 50,400 bars
- Current episode length is ~200 bars — 250x too short

**This is a fundamental constraint.** Staking cannot be learned in short episodes. Options:
1. Dedicated staking episodes with longer horizons (50k+ bars)
2. Hierarchical policy: spot policy at short horizon, staking policy at long horizon
3. Model the lock-in as a synthetic forward and train on payoff at expiry

Option 3 (synthetic forward framing) is likely the right abstraction — it converts the staking problem into a price-at-expiry prediction problem, which is closer to what the current model already does.

---

## The Synthetic Forward Framing

Rather than modeling the full intra-lock price path, frame lock-in staking as:

**Entry decision:** Given current price P₀, yield Y over period T, and total illiquidity window T+U (lock + unbonding), is E[P_{T+U}] + Y > P₀ + fees?

This decomposes into:
1. **Price forecast at horizon T+U** — can the current model answer this with regime features?
2. **Yield forecast** — expected yield given likely APY compression
3. **Opportunity cost** — what else could the capital earn in T+U days?

The staking model then becomes a **separate forecasting model** trained specifically on T+U-day price horizons per asset, informed by the asset research (PRD-006). BTC at 28-day horizon looks very different from SOL at 28-day horizon.

This framing means the staking layer doesn't require extending the spot model's episode length — it's a parallel model with a different time horizon.

---

## Interaction with Spot Trading Layer

The two layers must coordinate. Key constraints:

1. **Staked capital is unavailable to the spot model.** If 90% of capital is staked, the spot model can only trade the remaining 10%.
2. **Unstaking decision gates the spot exit.** If the spot model signals exit but position is staked, it must wait for unbonding.
3. **Yield modifies the effective entry price.** A position entered at $100 that earns 2% yield has an effective cost basis of $98 at expiry — the stop-loss and profit targets should adjust accordingly.
4. **Capital allocation must be split between layers.** The portfolio manager (PRD-005) must decide what fraction of capital is available for spot trading vs. staking at any time.

---

## Key Protocols to Research (Pre-Implementation)

Before building, research the mechanics of each target protocol:

| Protocol | Asset | Lock Type | Unbonding | Notes |
|---|---|---|---|---|
| Native ETH staking | ETH | Flexible (exit queue) | Hours–days | Queue-dependent |
| Lido (stETH) | ETH | Liquid (no lock) | None | Secondary market discount risk |
| Cardano native staking | ADA | Flexible | None | Best-case: no unbonding at all |
| Cosmos SDK chains | Various | Fixed | 21 days | Standard across Cosmos ecosystem |
| Curve veCRV | CRV | Fixed (weeks–4 years) | None (no early exit) | Governance-weighted yield |
| Aave / Compound | Multi-asset | Flexible | None | Variable rate, no lock |
| Uniswap v3 LP | Multi-asset | Flexible | None | Concentrated liquidity, high IL risk |

ADA and Lido/stETH are the lowest-complexity entry points — no unbonding period, no lock-in, closest to spot trading with yield overlay.

---

## Recommended Phasing

### Phase 1 — Yield overlay on flexible staking (no lock-in)
- ADA native staking (no unbonding) and/or Lido stETH (liquid)
- Reward function gains a per-bar yield credit when staked
- No new episode structure required — staking is just a state flag
- Action space: add STAKE / UNSTAKE to existing 5 actions
- Validates the basic yield-vs-opportunity-cost tradeoff

### Phase 2 — Fixed lock-in staking (synthetic forward model)
- 28-day lock protocols for ETH or SOL
- Separate forecasting model for P_{T+U} horizon
- Full illiquidity window modeled including unbonding
- Yield stochasticity modeled (APY compression distribution)

### Phase 3 — LP staking (short volatility positions)
- Uniswap v3 concentrated liquidity
- Impermanent loss modeling
- Requires volatility forecasting, not just price direction
- Most complex — do not attempt until Phase 2 is stable

### Phase 4 — Cross-layer portfolio optimization
- Spot and staking capital allocation decided by portfolio manager (PRD-005)
- Dynamic rebalancing between liquid and staked positions
- Full integration with live execution (PRD-004)

---

## Open Research Questions

1. **ADA staking as Phase 1 vehicle:** ADA has no unbonding period — staking is effectively a yield overlay on a held position. This may be the cleanest entry point. How does the yield (~5% APY) interact with the spot trading policy? Does it meaningfully change entry/exit decisions?

2. **Yield rate forecasting:** How stable are protocol APYs over 28-day windows historically? Is APY compression fast (days) or slow (weeks)? This determines whether a point estimate or a distribution is needed.

3. **stETH discount dynamics:** Lido's stETH has historically traded at slight discounts to ETH during stress periods (Celsius crisis). How large and frequent are these discounts? Does secondary market liquidity hold under stress?

4. **Tax treatment of staking yield:** In most jurisdictions, staking rewards are taxable as income at the time of receipt — even if not sold. This interacts with the high-frequency staking/unstaking decisions: each claim event is potentially a taxable event. Worth understanding before building the reward function.

5. **Protocol allowlist governance:** Who decides which protocols are approved for staking? What criteria? This is a human process that must be designed before the model can operate safely in live conditions.

---

*This document captures the conceptual framework. Implementation should not begin until the spot trading layer (PRD-001 through PRD-004) is stable in live conditions and the asset research (PRD-006) is sufficiently complete to inform per-asset staking strategy.*
