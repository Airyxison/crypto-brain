# Asset Research & Classification Framework
## Nova Trader — PRD-006

**Status:** Draft  
**Purpose:** Understand the fundamental nature of each training asset before narrowing model architecture. This document should be treated as a living reference — updated as research deepens and as new assets are considered.

---

## Why This Document Exists

We observed that training a single policy across BTC, ETH, SOL, and ADA produced inconsistent results — ADA passed, the others failed. The surface explanation was "different volatility profiles." The deeper explanation is that these assets have different *price-generating processes*. They don't just move differently; they move for different reasons, on different timescales, responding to different signals.

Building the right model requires knowing what you're trading. This document is that foundation.

---

## Classification Framework

Before examining each asset, a working classification:

| Class | Traditional Analog | Price Driver | Supply | Holding Profile |
|---|---|---|---|---|
| **Commodity-analog** | Gold, Oil | Scarcity + macro sentiment | Fixed/known | Medium–long trend |
| **Productive asset** | Tech equity + dividend | Utility demand + yield | Variable | Mixed, event-driven |
| **Ecosystem growth bet** | Small-cap growth equity | Adoption + competitive position | Variable | Shorter, momentum |
| **Range-bound utility** | Value equity / REIT | Stable use case + governance | Slow-release | Mean-reversion |

The four assets we train on map roughly as follows — though none fit perfectly, and that ambiguity is itself important:

| Asset | Closest Class | Confidence | Notes |
|---|---|---|---|
| BTC | Commodity-analog | High | Fixed supply, no yield, pure store-of-value narrative |
| ETH | Productive asset | Medium | Yield via staking, deflationary mechanics, but also macro-correlated |
| SOL | Ecosystem growth bet | Medium | High upside/downside, sentiment-driven, competitive |
| ADA | Range-bound utility | Medium-low | Academic cadence, lower speculative premium, but still correlated to BTC cycle |

---

## Asset Deep Dives

---

### Bitcoin (BTC)

#### What it is
Bitcoin is the original cryptocurrency, designed explicitly as a peer-to-peer electronic cash system with a fixed supply cap of 21 million coins. Approximately 19.7 million are mined as of early 2026. The remaining ~1.3 million will be issued over the next ~120 years via exponentially declining block rewards.

Its value proposition is **scarcity + credible neutrality**. It has no CEO, no foundation that can change the rules, no staking yield, no utility beyond being a trustless, censorship-resistant store of value. That's its design. In traditional finance terms it's the closest thing to **digital gold** — or more precisely, a commodity with a known, immutable supply schedule.

#### What moves it

**Halving cycles (dominant long-term driver)**  
Every ~210,000 blocks (~4 years), the block reward halves. The last halving was April 2024 (6.25 → 3.125 BTC/block). The next is ~2028. Historically each halving has been followed by a major bull market 6–18 months later as the reduced supply issuance creates scarcity pressure against growing demand. This is the most important structural feature of BTC's price action — a 4-year macro cycle that overwhelms short-term signals.

**Miner economics**  
Miners must sell BTC to cover operating costs (electricity, hardware). When price drops below miner cost-of-production, the weakest miners capitulate — a measurable on-chain event (hash rate drops, miner outflows spike). Miner capitulation events have historically marked cycle bottoms. This signal doesn't appear in our feature vector at all.

**Institutional flows**  
The January 2024 ETF approvals (BlackRock, Fidelity, et al.) created a structural new demand channel. ETF daily inflow/outflow data is now publicly available and meaningfully drives short-to-medium term price action. This is a qualitatively new dynamic that didn't exist in prior cycles.

**Macro risk sentiment**  
BTC correlates with Nasdaq (~0.6–0.7 in risk-off periods) — when institutional risk appetite contracts, BTC sells off with equities. This macro correlation strengthens in downturns and loosens in crypto-specific bull markets.

**On-chain accumulation**  
Long-term holder (LTH) supply — coins unmoved for 155+ days — is a sentiment indicator. When LTH supply rises during price declines, smart money is accumulating. When it falls during rallies, distribution is occurring. MVRV ratio (market value vs. realized value) measures whether the average holder is in profit or loss — extreme readings have historically preceded major reversals.

#### Implications for training

- **Optimal strategy type:** Trend-following with long holds. BTC doesn't mean-revert well intraday. Its big moves are directional and sustained.
- **Relevant timescale:** The 30-day momentum feature we added captures the post-halving rally pattern partially, but the 4-year halving cycle requires multi-month context to trade well.
- **Missing signals:** Miner outflow data, LTH supply change, ETF flow data, MVRV ratio, hash rate. None of these are in the current feature vector.
- **Current model failure interpretation:** The "never buy" policy on BTC during the bear phase may be *correct* — in a post-peak bear market with no on-chain accumulation signal and declining LTH supply, staying out IS the right call. The model may have learned something real, just lacked the context to know when to re-enter.
- **What we need before confident BTC training:** At minimum, a 30-day+ momentum signal (now added) and ideally some proxy for market cycle position (MVRV or similar).

---

### Ethereum (ETH)

#### What it is
Ethereum is a programmable blockchain — a decentralized computing platform. Its native token, ETH, serves multiple functions simultaneously: it's the gas fee currency (burned on use), a productive asset (staking yield ~3–4% APY), and the reserve asset of the DeFi ecosystem. After the Merge (September 2022), Ethereum transitioned from proof-of-work to proof-of-stake, making ETH holders validators who earn yield and making the supply deflationary at high gas usage levels.

This is fundamentally different from BTC. ETH is closer to **equity in a platform business** — its value is derived from the utility and activity of the network it powers. If Ethereum's ecosystem grows, ETH is worth more. If it loses developers to competitors, it's worth less.

#### What moves it

**Gas demand / network activity**  
When the Ethereum network is busy (DeFi activity, NFT minting, token launches), gas prices spike and more ETH is burned per block. High burn rates create deflationary pressure. Low activity periods are inflationary. Gas usage is a real-time fundamental indicator with no equivalent in BTC.

**Staking yield vs. risk-free rate**  
ETH staking yields ~3–4%. When macro risk-free rates (US Treasuries) are high, the risk premium for holding ETH instead shrinks — institutional money rotates out. As rates fall, the yield differential improves. This creates an interest-rate sensitivity that BTC lacks.

**L2 ecosystem dynamics**  
Layer 2 networks (Arbitrum, Optimism, Base, zkSync) absorb transaction volume from mainnet. This is a double-edged signal: L2 growth is good for ETH long-term (more users, more ETH secured) but reduces mainnet gas burn short-term. Net effect on price is complex.

**ETH/BTC ratio (relative strength)**  
ETH has its own cycle relative to BTC — periods of ETH outperformance ("alt season") and underperformance. The ETH/BTC ratio is a key signal traders watch. When ETH is losing ground to BTC on a relative basis, it's often a risk-off signal within crypto.

**DeFi Total Value Locked (TVL)**  
Total value locked in Ethereum-based DeFi protocols is a demand proxy. Rising TVL means more ETH is being used as collateral, staked in protocols, etc. — a bullish fundamental.

#### Implications for training

- **Optimal strategy type:** Mixed — trend-following during ecosystem growth phases, mean-reversion during consolidation. More event-responsive than BTC.
- **Relevant timescale:** Medium. 7-day momentum captures ETH's ecosystem cycles better than either 1-day (noise) or 30-day (too slow for fundamental shifts).
- **Missing signals:** Gas price / usage, staking APY, ETH/BTC ratio, DeFi TVL, L2 activity metrics. These are the actual fundamentals driving ETH. Without them we're trading the shadow of the real signal.
- **Current model failure interpretation:** ETH's price action in a bear market looks like noise to a model with only price/volume features. The "never trade" collapse likely reflects that without ecosystem signals, there's no reliable edge — which is honest. ETH is harder to trade on price alone than BTC or ADA.
- **What we need before confident ETH training:** ETH/BTC ratio as a feature is the highest-leverage addition (derivable from existing data). Gas usage and TVL require external data sources.

---

### Solana (SOL)

#### What it is
Solana is a high-performance Layer 1 blockchain targeting the performance limitations of Ethereum — it achieves ~2,000–5,000 real-world transactions per second at sub-cent fees. Its trade-off is a more centralized validator set and a history of network outages (2022–2023) that damaged its reputation before a significant recovery in 2024–2025.

SOL's value proposition is **performance + ecosystem**. It's primarily a bet that Solana's technical architecture wins a competitive race against Ethereum and other L1s. This makes it function more like **early-stage growth equity** — high upside if the ecosystem wins, high downside if it loses developers or suffers another major outage.

#### What moves it

**Ecosystem activity (especially meme coins)**  
In 2024, Solana's low fees and fast finality made it the dominant chain for meme coin launches via Pump.fun. This drove massive transaction volume and fee revenue to validators — a genuine fundamental signal. Meme coin activity is volatile and sentiment-driven, creating sharp SOL price spikes and corrections.

**FTX / Alameda legacy**  
FTX's collapse (November 2022) was catastrophic for SOL specifically because FTX and Alameda Research held massive SOL positions that were sold into bankruptcy. The liquidation pressure lasted through 2023. The recovery from those lows required rebuilding both price and ecosystem credibility.

**BTC cycle amplification**  
SOL is a high-beta asset — it amplifies BTC's moves. In bull markets it tends to outperform BTC significantly; in bear markets it underperforms. Its correlation to BTC is high (~0.75–0.85) but with a much larger amplitude.

**Network performance / developer activity**  
Ecosystem health is measurable via: number of active developers, DApp deployments, validator count and decentralization, network uptime. A major outage would be a fundamental negative signal.

**Competitive positioning**  
SOL's value is partly relative — it's priced against ETH's ecosystem. When ETH fees are high, users and developers migrate to Solana, boosting demand. When ETH L2s become cheap and capable, that migration slows.

#### Implications for training

- **Optimal strategy type:** Momentum-following, shorter holds, higher volatility tolerance. SOL trends hard and reverses hard.
- **Relevant timescale:** 1-day and 7-day momentum are probably the most useful. SOL moves fast.
- **Missing signals:** BTC dominance ratio (risk appetite within crypto), meme coin activity volume, developer growth metric, network uptime.
- **Current model failure interpretation:** SOL's high beta to BTC means the bear market hit it harder. The +2% result at v2 (31 trades, short hold) is interesting — the model found some short-term edges before overtrading destroyed returns. With the adaptive stop and regime features, SOL is a candidate for improvement.
- **What we need before confident SOL training:** BTC dominance ratio is derivable from our existing data (BTC price / total market proxy). Meme coin activity would require external data.

---

### Cardano (ADA)

#### What it is
Cardano is a proof-of-stake blockchain built using formal verification methods and peer-reviewed academic research, primarily developed by IOHK (now IOG). It uses Haskell as its primary development language — chosen for formal correctness guarantees over development speed. The result is a deliberately slower, more methodical approach to blockchain development.

ADA's value proposition is **rigor and long-term correctness**. It's the blockchain equivalent of building infrastructure vs. building a startup — slower, more conservative, but theoretically more robust. In market terms, this makes ADA behave more like a **value stock** with defined ranges rather than a growth equity.

#### What moves it

**BTC correlation (dampened)**  
ADA correlates with BTC macro cycles but with lower beta (~0.5–0.6) than SOL or ETH. It participates in bull markets but less aggressively, and holds up better in bear markets.

**Governance milestones**  
Cardano uses an on-chain governance system (the Voltaire era). Major governance events — treasury funding decisions, protocol upgrades, CIP (Cardano Improvement Proposal) approvals — are scheduled and public. These are knowable catalysts unlike the sentiment-driven moves in SOL.

**Staking participation**  
~67–70% of ADA supply is staked (~5% APY). This creates structural demand and reduces circulating supply. Unlike ETH's staking, ADA staking doesn't lock tokens — staked ADA remains liquid, reducing staking-induced supply constraints.

**Development pace**  
ADA has historically been criticized for slow delivery. Major milestones (smart contracts via Plutus in 2021, Hydra L2 development, Voltaire governance) move the price on delivery or disappointment relative to announced timelines.

**Range-bound character**  
ADA tends to trade in multi-month ranges, mean-reverting more than BTC or SOL. This is consistent with its lower speculative premium and more value-oriented holder base (large % of supply permanently staked).

#### Implications for training

- **Optimal strategy type:** Mean-reversion / range trading. ADA is the most tractable asset for the current model architecture precisely because its range-bound behavior is more predictable.
- **Relevant timescale:** Medium. 7-day and 30-day signals capture ADA's range behavior well.
- **Missing signals:** Governance calendar, staking participation rate changes.
- **Current model success interpretation:** ADA's v4 result (+7.67%, 7 trades, 128 bar avg hold) likely reflects the model finding real mean-reversion edges within ADA's natural trading range. This is encouraging — the model works when the asset's behavior is consistent with the strategy being learned.
- **ADA as control asset:** ADA passing across multiple iterations while others fail isn't random. It's telling us the model architecture is capable of finding edges — just not in assets whose price action requires signals we don't currently have.

---

## What We Have vs. What We Need

| Signal | In Features Now | BTC | ETH | SOL | ADA |
|---|---|---|---|---|---|
| Short-term momentum (1m/5m/15m) | Yes | Low | Medium | High | Medium |
| 1h volatility | Yes | Medium | Medium | High | Low |
| Volume (normalized) | Yes | Low | Medium | High | Low |
| VWAP deviation | Yes | Low | Medium | Medium | Low |
| 24h / 7d / 30d momentum | Yes (v6) | High | High | High | High |
| Miner outflows / hash rate | No | **Critical** | — | — | — |
| ETH/BTC ratio | No | Medium | **High** | High | Medium |
| Gas usage / burn rate | No | — | **Critical** | — | — |
| DeFi TVL | No | Low | High | Medium | Low |
| BTC dominance ratio | No | — | Medium | **High** | Low |
| Staking yield | No | — | High | Medium | Medium |
| Governance calendar | No | — | — | — | Medium |
| Market cycle position (MVRV) | No | **Critical** | High | Medium | Low |

---

## Architectural Implications

### Near-term (v6 and onward)
The regime momentum features (1d/7d/30d) added in v6 are the highest-leverage near-term improvement and address the largest gap across all four assets. They're derivable from existing price data with no external dependencies.

**ETH/BTC ratio** is the next most valuable addition — also derivable from existing data if we load both assets simultaneously. It captures ETH's relative strength signal without requiring external data sources.

### Medium-term
Separate model families per asset class rather than one universal policy:
- **BTC:** Trend-following policy with longer episode horizons. Consider cycle-position features.
- **ETH:** Fundamentals-responsive policy. Needs gas/TVL data to be genuinely effective.
- **SOL:** Momentum policy with tighter stops. High-beta amplifier of BTC regime.
- **ADA:** Mean-reversion policy with range detection. Current architecture closest to correct already.

### Longer-term  
When adding traditional asset classes (equities, ETFs, commodities, forex), the crypto classification work done here directly applies:
- BTC → Commodity bucket (gold, oil, agricultural futures)
- ETH/SOL → Growth equity bucket (tech stocks, high-growth ETFs)
- ADA → Value/income bucket (dividend stocks, REITs)
- Stablecoins → Cash/money market bucket (T-bills, money market funds)

The policy families designed for crypto will transfer to traditional assets within the same class with minimal re-architecture.

---

## Research Gaps & Open Questions

Before confidently narrowing training per asset, these questions need answers:

1. **BTC halving cycle position:** We're currently ~2 years post-halving (April 2024). Historically this is the beginning of a major bull run. Does our training data capture enough of the pre-halving accumulation phase to learn cycle position? Probably not fully.

2. **ETH post-Merge dynamics:** The deflationary/inflationary flip based on gas usage is a new dynamic with limited historical data. Our 1-year training window may not have enough examples of both states.

3. **SOL ecosystem dependency:** How much of SOL's 2024–2025 price action was driven by meme coin activity specifically? If that activity normalizes, the trained policy may be fitting a regime that no longer exists.

4. **ADA's governance calendar:** If governance milestones are scheduled and public, they're a nearly risk-free feature to add. Worth investigating the Cardano roadmap for predictable catalysts.

5. **Cross-asset correlation structure:** How much does ETH/BTC correlation change across market regimes? The model currently has no way to see relative strength — only absolute price action.

---

## Recommended Next Steps

1. **v6 completes** (adaptive stop + regime momentum features) — establishes the baseline with best available features from price data alone
2. **Add ETH/BTC ratio** — derivable from existing data, high value for ETH and SOL classification
3. **Per-asset research deepens** — particularly BTC cycle position (MVRV proxy) and ETH gas as external data sources
4. **Separate model families** — once research confirms the class boundaries, build BTC-specific and ETH-class-specific policies
5. **Traditional asset expansion** — armed with the class framework, onboard equities/commodities/forex into the same architecture

---

*This document should be updated as training results provide new evidence about what each asset's model needs. The v5/v6 results will be the first data points worth incorporating.*
