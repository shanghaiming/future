# Futures Strategy Performance Table

> Last updated: 2026-05-24
> Data: 68 Chinese commodity futures, 3051 days (2016-2026), CASH0=500K, COMM=0.03%, no leverage

## Summary

| Version | Core Idea | Annual | WR | DD | PF | Walk-Forward | Status |
|---------|-----------|--------|-----|-----|-----|-------------|--------|
| **V63** | **Combo: 极端配对(Z=2.0) + 动量(LB3) 1日持仓** | **+369.1%** | **66.1%** | **18.8%** | **2.90** | **+602.8% (WF2023!)** | **WF突破600%!** |
| V57 | Global adaptive pair (LB5-10, EP60, Z1.0) | +315.3% | 65.3% | 7.5% | - | +493.0% (WF2024) | Previous best |
| V55 | Adaptive spread pair trade (pct/log/raw auto-switch, H1) | +307.3% | 64.4% | 8.6% | 3.08 | +355.7% (avg WF) | Previous best |
| V52 | Ultra short-term pair trading (1-day hold, Z=1.0) | +303.5% | 62.9% | 12.5% | 2.85 | +472.6% (WF2024) | Previous best |
| V39 | Supply chain pair trading (3-day hold, Z=1.5) | +188.1% | 60.6% | 15.4% | 2.43 | +376.7% (WF2024) | Previous best |
| V34b | Group momentum lag (commodity catches up to group) | +86.8% | 55.4% | 23.5% | 2.38 | +97.4% (WF2024) | Proven |
| V38-N1 | Multi-position v34b (N=1, same as v34b) | +86.8% | 55.4% | 23.5% | 2.38 | +32.8% (WF2023) | Baseline |
| V34 | Multi-signal ensemble (group lag + VDP + OI) | +80.7% | 57.4% | 27.2% | 1.89 | - | Proven |
| V14b | VDP momentum swing rotation | +73.0% | 49.4% | 150.8% | 1.55 | - | High DD |
| V36 | HAR-RV volatility timing (regime filter) | +72.2% | - | - | - | - | Hurts v34b |
| V33 | Supply chain upstream momentum lag | +50.2% | 56.3% | 36.9% | 1.79 | - | Proven |
| V40 | OI institutional flow (surge + price confirm) | +46.0% | 56.6% | 34.4% | 1.59 | +48.3% (WF2023) | OI filter hurts |
| V34c | Adaptive lookback/KER gate/Sortino | +46.4% | - | - | - | - | All opts hurt |
| V38-N2 | Multi-position v34b (N=2, group div) | +53.4% | 54.8% | 29.5% | - | - | Dilution |
| V38-N3 | Multi-position v34b (N=3) | +39.8% | 54.0% | 24.9% | - | - | Dilution |
| V38-N5 | Multi-position v34b (N=5) | +18.4% | - | 19.2% | - | - | Dilution |
| V37 | Kelly criterion + VaR risk management | +12.1% | 56.2% | 11.5% | 1.71 | +2.7% (WF2023) | Over-risk-mgmt |
| V35 | Path signature features | +5.1% | 49.0% | - | - | +98.9% (WF2025, 59t) | Failed |
| V42 | Combined v34b + v39 portfolio (60/40 alloc) | +51.7% | 61.4% | 6.0% | 2.04 | +111% (WF2024) | Dilutes V39 |
| V42 | Combined v34b + v39 portfolio (30/70 alloc) | +49.8% | 60.7% | 4.7% | 2.38 | +104% (WF2024) | Best PF/DD |
| V43 | Momentum spike reversal (group context) | +12.0% | 53.4% | 28.8% | 1.80 | +26.4% (WF2024) | Weak signal |
| V45 | Cross-timeframe momentum alignment (TF5_10_20) | +11.1% | 52.9% | 25.7% | 1.34 | +14.0% (WF2024) | Failed |
| V48 | Multi-hop supply chain lead-lag (group+chain) | +87.2% | 56.4% | 28.9% | 2.03 | +76.1% (WF2023) | Same as V34b |
| V44 | Seasonal pattern (anti-season reversal) | +43.2% | 55.1% | 25.5% | 1.80 | +106.7% (WF2024) | WF强 |
| V47 | Opening range breakout + OI (body ratio) | +9.7% | 50.9% | 35.9% | 1.21 | +5.5% (WF2024) | Failed |
| V50 | Vol regime rotation (pairs in low-vol, mom in high-vol) | +126.1% | 60.5% | 24.6% | - | +187.6% (WF2024) | Rotation有效 |
| V49-Seas | Seasonal optimized (anti-season, W20) | +72.7% | 57.0% | 35.9% | - | +154.5% (WF2024) | 改进V44 |
| V49-Comb | V39(80%) + V44(20%) combined | +173.1% | 60.2% | 13.8% | - | +371.4% (WF2024) | 配对主导 |
| V51 | V39极限优化 (exit Z=0.3, hold=2) | +96.2%* | 61.7% | 28.1% | 2.80 | +108.6% (WF2022) | *引擎不同 |
| V46 | V39精选配对 (P13, exit Z=0.5, H2, MP3) | +254.3% | 61.6% | 19.8% | 2.79 | +337.5% (WF2024) | exit_z=0.5关键 |
| V54 | V52多配对 (MP1, LB10, Z1.2) | +283.1% | 63.0% | 12.4% | 2.87 | +447.3% (WF2024) | MP1仍最优 |
| V56 | V52+确认 (VDP最佳确认) | +245.3% | 63.8% | 10.5% | - | +489.0% (WF2024) | 确认砍交易量 |
| V58 | V55严格WF验证 (6窗口, avg OOS) | +277.3%* | - | - | - | 6/6窗口正 | V55≈V52, 3:3 |
| V60 | LOG spread (LB15, Z1.5, H1, MP1) | +312.9% | 65.6% | 7.8% | 3.46 | +662.5% (avg WF), 100%正 | PF最高 |
| V66 | V62零手续费理论天花板 | +357.9% | - | 7.8% | - | +538.7% (avg WF, 6/6) | THEORETICAL MAX |
| V59 | Regime组合 (pairs+momentum, 1-day) | +301.4% | 59.6% | 22.8% | 2.17 | +212.2% (WF2024) | 不如纯pairs |
| V41 | V39优化 (Sharpe加权, 20对) | +132.7% | 58.7% | 9.0% | 2.18 | +167.5% (WF2024) | 2对亏损稀释 |

## Key Insights

### What Works
1. **Supply chain pair trading (V39)**: Z-score mean reversion on 13 upstream-downstream pairs. Market-neutral, 60.6% WR, DD only 15.4%. Walk-forward +376.7%. This is the current champion.
2. **Group momentum lag (V34b)**: When a commodity lags its supply-chain group, it catches up. Simple, robust, +86.8% with walk-forward validation.
3. **Signal simplicity**: Every attempt to add complexity (adaptive params, KER gate, Kelly, VaR) has hurt performance.

### What Doesn't Work
1. **Adding filters**: VDP, OI, KER, Sortino — all reduce returns from +86.8% baseline
2. **Multi-position**: N=1 >> N=2 >> N=3 (signal strength drops with rank)
3. **Path signatures**: Too coarse for daily data, needs tick-level
4. **Vol timing (HAR-RV)**: Filters out best trades (high vol = strong trend = most profitable)
5. **Kelly + VaR**: Over-manages risk, chokes trading frequency
6. **Strategy combination (V42)**: V39 alone (+188%) >> combined (+51.7%). Pair trading is strong enough to stand alone.
7. **Pair expansion (V41)**: Adding more pairs from 13→20 hurt. 2 new pairs were net losers (agfi/aufi, srfi/cfi). Simple 13-pair set is optimal.
8. **Cross-timeframe alignment (V45)**: Pure TF alignment produced zero profitable configs. Multi-TF agreement is too rare a signal on daily data.
9. **Exit Z = 0.5 (V46)**: Exiting pair trades when z-score crosses 0.5 (not 0) was the single biggest improvement to V39. Captures more of the move.
10. **1-day hold + daily compounding (V52)**: 1-day hold with 62.9% WR produces +303.5% through rapid capital recycling.
11. **Adaptive spread mode (V55)**: Dynamically switching between raw/pct/log spread every 40-60 days based on recent performance adds +4pp annual return, +1.5pp WR, and dramatically improves walk-forward robustness.
12. **V55 ≈ V52 in rigorous WF (V58)**: Head-to-head 3:3 tie. Adaptive switching adds robustness but no clear superiority. Both strategies are genuine. Log spread dominates in recent years.
13. **LOG spread dominates recent years (V60)**: LOG_LB15_Z1.5 achieves +312.9%, PF=3.46 (highest), DD=7.8%. 100% of 60 WF tests positive. WF avg +662.5%. Strategy edge is STRENGTHENING over time.
14. **LOG-biased adaptive (V61)**: +324.3%, 6/6 WF positive, avg +394.9%. LOG gets 3/5 candidate slots in adaptive selection. 14 pairs with cfi/csfi (75.6% WR) adds value. Z=0.8 optimal for adaptive mode.

### Core Principles
- **No stop loss** — only trailing stop + time exit + signal flip
- **Signal flip exit** = 86% WR (v14b data), dominant profitable exit
- **Simple fixed params** beat adaptive/complex approaches
- **Concentration > diversification** for single-signal strategies
- **Orthogonal signals** (directional + market-neutral) can be combined

## Strategy Details

### V61 — LOG-Biased Adaptive Pair Trading (CHAMPION)
- **Signal**: Adaptive spread mode with LOG bias (LOG gets 3/5 candidate slots: log_LB10, log_LB15, log_LB20 + raw_LB10, pct_LB10)
- **Best config**: T1_LOGBIAS_EP40_Z0.8_MP1_P14 (eval=40d, Z=0.8, max 1 pair, 14 pairs with cfi/csfi)
- **Key**: +324.3% annual, 6/6 WF windows positive, avg WF +394.9%. Mode selection: LOG 35-43%, raw 29-33%, pct 8-11%. Overfitting check passed (r=0.872, decay=0.82).

### V57 — Global Adaptive Pair Trading
- **Signal**: Z-score mean reversion with global adaptive spread mode (raw/pct/log) + adaptive lookback (5/7/10)
- **Best config**: global_adaptive, EP60, Z1.0, LB[5,7,10], equal weight
- **Key**: +315.3% annual, 65.3% WR, 7.5% DD. Global mode beats per-pair mode. Equal weight beats WR-weighted.

### V55 — Adaptive Spread Pair Trading
- **Signal**: Z-score mean reversion with adaptive spread mode (raw/pct/log auto-switch every 60 days)
- **Entry**: z > 1.0 → short downstream + long upstream; z < -1.0 → reverse
- **Exit**: next day (1-day hold), or z crosses 0
- **Best config**: ADG_EP40_Z1.0_Mpct_log (eval every 40 days, Z=1.0, pct+log modes)
- **Key**: Dynamic spread mode selection improves WR to 64.4%, Sharpe to 2.17, DD to 8.6%. WF2022 +859.6%.

### V52 — Ultra Short-Term Pair Trading (Previous Champion)
- **Signal**: Same as V39 but Z=1.0 entry, 1-day hold
- **Entry**: z > 1.0 → short downstream + long upstream; z < -1.0 → reverse
- **Exit**: next day (time exit), or z crosses 0
- **Best config**: LB10_Z1.0_H1_EZ0.0_MP1 (lookback=10, z=1.0, hold=1, exit z=0, max 1 pair)
- **Key**: 1-day hold enables daily compounding at 62.9% WR. ~280 trades/year. All 13 pairs profitable.

### V39 — Supply Chain Pair Trading (Previous Champion)
- **Signal**: Z-score of price spread between upstream/downstream pairs
- **Entry**: z > 1.5 → short downstream + long upstream; z < -1.5 → reverse
- **Exit**: z crosses 0 (mean reversion), or time (3 days)
- **Best config**: LB10_Z1.5_H3_MP2 (lookback=10, z=1.5, hold=3, max 2 pairs)
- **Pairs**: rbfi/ifi, hcfi/ifi, hcfi/rbfi, jfi/jmfi, mafi/scfi, fufi/scfi, bfi/scfi, mfi/afi, yfi/afi, pfi/yfi, ppfi/mafi, vfi/mafi, egfi/mafi
- **Key**: All 13 pairs profitable. 95.8% exits via mean reversion. No stop-loss ever triggered.

### V34b — Group Momentum Lag
- **Signal**: group_mom_excl_self - own_mom (5-day lookback)
- **Entry**: score > 0.003, take top-ranked commodity
- **Exit**: time (3 days), trailing stop (3.0 ATR), signal flip
- **Key**: Every year profitable 2016-2026. Energy group most profitable.

### V42 — Combined Directional + Pair Portfolio
- **Signal A**: V34b group momentum lag (directional)
- **Signal B**: V39 supply chain pair trading (market-neutral)
- **Best alloc**: 60/40 (dir/pairs) = +51.7%, DD 6.0%, PF 2.04
- **Best risk-adjusted**: 30/70 = +49.8%, DD 4.7%, PF 2.38
- **Key**: Both strategies contribute, but combining dilutes V39's +188%. V39 alone is superior.

## Running Experiments
- **V41**: V39 pair trading deep optimization (dynamic z, asymmetric exit, more pairs) — RUNNING
- **V43**: Momentum spike reversal — COMPLETED, +12.0% best. Too weak.
- **V41**: V39 pair trading deep optimization — COMPLETED, +132.7% best. V39 original still better.
- **V44**: Seasonal patterns — COMPLETED, +43.2% full, WF +106.7% strong OOS
- **V44**: Seasonal patterns (agri/energy cycle) — RUNNING
- **V45**: Cross-timeframe momentum alignment — RUNNING
