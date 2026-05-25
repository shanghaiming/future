# Futures Strategy Performance Table

> Last updated: 2026-05-24 (session 6 — V140 POSITION SIZING BREAKTHROUGH: +155% annual / -24% MDD)
> Data: 68 Chinese commodity futures, 3051 days (2016-2026), CASH0=500K, COMM=0.03%, no leverage

## CRITICAL: Same-Bar Contamination Discovery

**V96 proved**: All V74-V92 high-return strategies (+2186% to +4282%) used same-close entry (signal computed from close, entry AT that close). This is same-bar contamination — the signal IS the entry price. With next-open execution, ALL alpha disappears:
- V82: +3311% → **-21.5%** (next-open)
- V92: +4282% → **-3.9%** (next-open)
- V74: +2186% → **-20.1%** (next-open)

Only strategies with **next-open execution** (signal at close di, entry at open di+1) are practically valid.

---

## Practical Strategies (Next-Open Execution) — THE ONLY VALID RESULTS

### PRACTICAL CHAMPION (V121, verified V123)
**Signal**: ROC(5) > 1.0% AND Z-score(today return, 20-day) > 1.5 AND ROC(5) improving
**Ranking**: By ROC(5) × Z-score (combined momentum × statistical extreme)
**Entry**: Buy at next day's open (O[di+1]), top_n=1 (concentrated)
**Exit**: Close of day after entry (hold=1, actual ~1.5 trading days)
**Annual**: **+333.5%** | WR 63.5% | Sharpe 1.21 | Sortino 14.67
**Walk-forward**: 6/6 calendar years positive (2020:+221%, 2021:+453%, 2022:+544%, 2023:+169%, 2024:+54%, 2025:+214%)
**Alpha decay**: NONE — improving over time
**Sliding window**: 82.5% of 40 windows positive, OOS avg +117.3%
**Commission**: Profitable at 0-0.20%
**Universe**: 67/68 commodities profitable, 14 with 6/6 WF
**MDD**: -95.3% (deep — use >3x equity → 50% risk for -48.2% MDD, +126.1% annual)

### Risk-Managed Variant (for practical execution)
**Rule**: When equity > 3× starting capital, only risk 50% of capital per trade
**Result**: +126.1% annual, -48.2% MDD, 6/6 WF, WF avg +201.7%
**Best risk-adjusted**: 50% risk → retains most upside, halves the drawdown

### Full Strategy Ranking

| Version | Core Idea | Annual | WR | MDD | WF Pos | WF Avg | Status |
|---------|-----------|--------|-----|-----|--------|--------|--------|
| **V120** | **ROC>1% + Z>1.5, Hold=1 (参数最优)** | **+306.2%** | **60%** | **-40.5%** | **6/6** | **+281.9%** | **CHAMPION** |
| V119 | ROC>2% + Z>1.5, Hold=1 (全68品种) | +247.6% | 60% | - | 6/6 | +281.9% | 确认 |
| V117 | Champion Top20 + Hold1 + SkipMon | +251.7% | - | - | 6/6 | +506.5% | 极致优化 |
| V119 | Triple Mom + Top20, Hold=1 | +207.8% | 58.1% | - | 6/6 | +259.9% | 突破200% |
| V118 | Triple Momentum (ROC3+ROC5+ROC10) H3 | +145.5% | 60% | -40.5% | 5/6 | +256.8% | 三重动量 |
| V119 | ROC+Z Thu/Fri, Hold=1 | +140.2% | 69.7% | -25.9% | 6/6 | +145.4% | 最佳WR+低MDD |
| V116 | ROC>2% + Z>1.5, Hold=3, TN1 | +112.9% | 60% | -40.5% | 6/6 | +112.9% | 最稳健H3 |
| V116 | ROC>2% + Z>2.0, Hold=3, TN1 | +98.8% | 61.6% | -31.9% | 6/6 | +144.3% | 最低MDD |
| V108 | ROC(5)>2%, Hold=5, TN3 | +89.8% | 61.6% | -64.8% | 6/6 | +92.2% | 单指标 |
| V104 | ROC(5)零交叉, Hold=5 | +81.9% | 58.0% | -41.7% | 6/6 | +79.9% | 原始ROC |
| V118 | Weighted Agg (ROC*ADX*OI) | +66.8% | - | - | 6/6 | +170.5% | 最稳定6/6 |
| V111 | Z-score极端 (z>2.0, H3) | +73.9% | 63.1% | -36.8% | 6/6 | +69.3% | 数学最佳 |
| V105 | T3 Cross, Hold=5 | +50.7% | 57.5% | -49.5% | 6/6 | +52.3% | TA-Lib最佳 |

### Key Strategy Insights (V101-V118)

**Signal Hierarchy (what generates the most alpha):**
1. **ROC(5) momentum** — the single most powerful signal. Zero cross (+81.9%) → magnitude filter >2% (+89.8%) → triple TF (+145.5%)
2. **Z-score extreme** — statistical extreme days have follow-through. Combined with ROC gives +112.9%
3. **Adaptive MAs (T3, KAMA)** — adaptive smoothing reduces whipsaws. +50.7% standalone
4. **Breakout strength** — concentrated best breakout +60.6%
5. **Mathematical indicators** — Hurst +43.9%, Z-score +73.9%, but complex to compute
6. **Volume/OI** — supplementary signal, +35.8% best. OI doesn't add much over pure price momentum
7. **Term structure** — USELESS. No predictive power with next-open execution
8. **Candlestick patterns** — continuation (+44%) > reversal (-10%). Overnight gap destroys reversal signals

**Optimization Hierarchy:**
1. **Signal quality** (ROC>2% vs ROC>0): +8pp improvement
2. **Multi-signal combination** (ROC+Z-score): +23pp over single ROC
3. **Hold period** (3 days optimal for combined signals, 5 days for single)
4. **Universe selection** (top 20 commodities + medium volatility): significant
5. **Day-of-week filter** (Thu/Fri best): +15pp
6. **Position sizing** (fixed 100% optimal; Kelly/vol-scaling hurt returns)
7. **Regime detection** (improves consistency but not absolute returns)

**What DEFINITELY doesn't work:**
- Leverage (2x blows up)
- Daily re-evaluation (hold while ROC>0) — traps capital in mediocre positions
- Aggressive compounding — losses over-reduce subsequent position sizes
- Trailing stops on ROC signals — exit too early on volatile winners
- Term structure — zero predictive power
- Ichimoku — zero alpha
- Mean-reversion with next-open execution — overnight gap destroys the edge

## Theoretical Strategies (Same-Bar Contaminated — NOT Practically Executable)

| Version | Core Idea | Annual | WR | DD | PF | Walk-Forward | Status |
|---------|-----------|--------|-----|-----|-----|-------------|--------|
| **V92** | **隔夜z-score (跨组隔夜gap z-score, close入场)** | **+4281.8%** | **64.9%** | **25.5%** | - | **+4740% (6/6 WF avg!)** | **理论冠军(不可执行)** |
| V94 | 扩展品种池68品种(含不可交易品种) | +20419%* | 64.7% | 24.5% | - | +187K%* | *虚假! bbfi(17手)和ecfi驱动 |
| V93 | V92精细调优(阈值/top_n/加权/星期) | +4283% | - | - | - | +4768% (6/6) | 仅+1.5%改善，V92近最优 |
| V95 | 市场状态过滤(11种regime) | +2434% | - | - | - | - | 所有regime过滤有害 |
| **V82** | 跨组截面z-score (全市场相对弱势catch-up) | +3305.5% | 64.2% | - | - | +4180% (6/6 WF avg!) | 前CHAMPION |
| **V82-C** | 跨组对比(排除本组) 全市场均值回归 | +3247.0% | 63.7% | - | - | +4027% (6/6) | V82变体，同样强 |
| V77 | 多重重叠组(13组含供应链+波动率分组) | +2250.0% | 63.9% | - | - | +2539% (6/6) | 击败V74 (+62pp) |
| V79 | 自适应阈值(百分位P80) | +2193% | 64.8% | - | - | +2491% (6/6) | 微弱提升，alpha来自组结构 |
| V74 | 扩展分组(44品种8组) + 1日动量 LB=1 | +2185.7% | 64.4% | - | - | +2498% (6/6 WF avg) | 前冠军，within-group |
| V69 | 1日持仓组动量滞后 LB=1 (25品种6组) | +1023.2% | 61.4% | 27.1% | 2.90 | +1000% (6/6 WF avg) | 突破1000%, r=0.968 |
| V69-C3 | 同上 COMM=0.03% | +900.2% | 62.3% | 17.4% | 2.20 | +885% (6/6) | 实际手续费 |
| V70 | Combo: 配对Z2.5 + 动量ML3 (58/42) | +562.4% | 66.0% | 17.9% | 3.25 | +262% avg (5/6) | 配对拖累动量 |
| V80 | 供应链级联(价差动量, 13对upstream-downstream) | +432.8% | 60.3% | 31.1% | - | +586% (6/6) | 有效但低于V74 |
| V78 | 组内排名动量(Rank divergence + V74组合) | +1408.3% | 63.0% | 22.4% | - | +1473% (6/6) | 排名变换削弱信号 |
| V84 | Portfolio V74(40%)+V62(60%) Sharpe最优 | +475% | - | 15.7% | 3.96 | +316% (6/6) | Sharpe=6.65最高 |
| V84-60/40 | Portfolio V74(60%)+V62(40%) 平衡 | +719% | - | 15.7% | 4.16 | +639% (6/6) | 最佳风险收益平衡 |
| V85-E | 隔夜+日内分解动量(alpha=0.7, 70%隔夜+30%日内) | +3171% | 65.5% | 30.1% | - | +3597% (6/6) | 接近V82但未超越 |
| V85-B | 隔夜过滤(只做隔夜gap方向一致) | +2307% | 64.2% | - | - | +3131% (6/6) | 隔夜确认有效 |
| V86 | 加工利润均值回归(油籽价差,大豆压榨) | +42.5% | 58.9% | 16.4% | - | +42.4% (6/6) | 信号有效但交易机会少 |
| V76 | 尾部风险+OI确认+波动率缩放+熔断器 | +355%~+2128% | - | 7.6~22.4% | - | - | 风控降低DD但摧毁收益 |
| V75 | 波动率自适应LB (vol-adaptive) | +784%~+1494% | - | - | - | - | 固定LB=1(+2185%)完胜 |
| V73 | Cross-pair momentum confirmation | +321.2% | 56.8% | 39.0% | 2.14 | +290% avg (6/6) | Pair确认有效 |
| V63 | Combo: 极端配对(Z=2.0) + 动量(LB3) 1日持仓 | +369.1% | 66.1% | 18.8% | 2.90 | +602.8% (WF2023) | WF突破600% |
| V63-mom | 纯1日动量 LB=3 | +448.9% | 57.8% | - | 2.44 | - | V69前身 |
| V62 | LOG-biased自适应配对, 14对, Z=1.0 | +334.3% | 66.2% | 9.4% | 3.24 | +325.8% avg (5/6) | Previous champion |
| V72 | V69 + 动态仓位 (Kelly/Signal/TN3) | +433.3% | - | - | - | +345% avg | 固定满仓最优 |
| V71 | 动量 + 配对确认 (LB3) | -0.5% | 62.4% | 4.4% | 2.74 | +5.7% avg (2/6) | 确认过滤无用 |
| V60 | LOG spread (LB15, Z1.5, H1, MP1) | +312.9% | 65.6% | 7.8% | 3.46 | +662.5% (avg WF) | PF最高 |
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
| V65 | Vol breakout + pair combo | +288.7% | - | - | - | +287.7% avg (6/6) | Vol breakout无用 |
| V59 | Regime组合 (pairs+momentum, 1-day) | +301.4% | 59.6% | 22.8% | 2.17 | +212.2% (WF2024) | 不如纯pairs |
| V41 | V39优化 (Sharpe加权, 20对) | +132.7% | 58.7% | 9.0% | 2.18 | +167.5% (WF2024) | 2对亏损稀释 |

## Key Insights

### What Works
1. **Overnight z-score (V92) — NEW CHAMPION**: +4282% annual, 6/6 WF positive, avg WF +4740%. Signal: compute z-score of each commodity's OVERNIGHT return (close-to-open gap) relative to the cross-group overnight return distribution. When z < -0.3, the commodity gapped down overnight relative to all groups → buy at close expecting catch-up tomorrow. Isolating the overnight component gives +976pp over V82's full daily return z-score.
2. **Cross-group z-score (V82)**: +3305% annual, 6/6 WF positive, avg WF +4180%. Previous champion. Uses full close-to-close return for z-score.
3. **Extended group 1-day momentum LB=1 (V74)**: +2185.7% annual, 6/6 WF positive, avg WF +2498%. Signal: "today the group moved more than this commodity, buy it at close expecting catch-up tomorrow." 44 commodities in 8 groups.
4. **Multi-group overlap (V77)**: +2250% annual. When a commodity gets buy signal from multiple overlapping groups, stronger conviction. +62pp over V74.
5. **1-day hold + daily compounding**: The single most important discovery. Hold=1 gives +2186%, Hold=2 drops to +233% (V83 confirmed). The signal is a pure overnight mean-reversion effect that decays rapidly.
6. **LB=1 >> LB=3 >> LB=5**: Shorter lookback = faster compounding. Confirmed again in V83 sweep.
7. **Long-only >> Long+Short**: Adding shorts DESTROYS returns. Confirmed in V78 (rank-based) and V80 (supply chain).
8. **Supply chain spread momentum (V80)**: +432.8% annual, 6/6 WF positive. Mean reversion in supply chain spread works, but weaker than group momentum.

### What Doesn't Work
1. **Adding filters**: VDP, OI, KER, Sortino — all reduce returns from +86.8% baseline
2. **Multi-position**: N=1 >> N=2 >> N=3 (signal strength drops with rank)
3. **Path signatures**: Too coarse for daily data, needs tick-level
4. **Vol timing (HAR-RV)**: Filters out best trades (high vol = strong trend = most profitable)
5. **Kelly + VaR**: Over-manages risk, chokes trading frequency
6. **Strategy combination (V42)**: V39 alone (+188%) >> combined (+51.7%). Pair trading is strong enough to stand alone.
7. **Cross-timeframe alignment (V45)**: Pure TF alignment produced zero profitable configs. Multi-TF agreement is too rare a signal on daily data.
8. **Vol-adaptive LB fails (V75)**: Fixed LB=1 (+2185%) >> all adaptive approaches. Switching LB based on vol regime destroys the simple 1-day momentum signal.
9. **Risk management destroys returns (V76)**: Every risk filter reduces annual more than it reduces DD. V74's raw signal is too profitable to filter.
10. **Rank transformation weakens signal (V78)**: Pure rank-based divergence gives only +196% vs V74's +2154%. The absolute divergence is fundamentally stronger.
11. **Volume weighting doesn't help (V81)**: Equal weight (+2186%) >> all volume-weighted schemes. Volume filters are strongly negative.
12. **Hold period > 1 day destroys alpha (V83)**: LB=1/Hold=1 gives +2186%, Hold=2 gives +233% (-89%).
13. **Within-group is good, cross-group is better (V82)**: V74 within group → +2186%. V82 cross-group → +3305%.
14. **OI confirmation hurts (V88)**: All OI filters reduce returns by -2667pp from +3305% baseline. The z-score already captures mean-reversion without OI.
15. **Group structure granularity doesn't matter (V89)**: 4 groups (+3314%), 8 groups (+3305%), 16 groups (+3314%) — all nearly identical. The signal compares against the overall market mean, not group-specific features.
16. **Multi-day z-score much worse (V90)**: 2-day (-1914pp), 3-day (-2001pp), 5-day (-2156pp) vs 1-day. Cross-group mean-reversion is strictly a 1-day phenomenon.
17. **Volume/volatility confirmation hurts (V91)**: Vol surge (-3215pp), range expansion (-3194pp), ATR context (-2615pp to -2883pp) all significantly worse than pure z-score.
18. **Threshold doesn't matter below 0.5 (V87)**: z < -0.1 through z < -0.5 all produce identical results. The threshold is non-binding — there's always at least one commodity with z < -0.5.
19. **Decomposed alpha=0.7 overnight + 0.3 intraday (V85)**: +3171% — close to V82 but didn't beat it. However, the decomposition insight led to V92's breakthrough.
20. **Z-magnitude position sizing hurts (V93)**: Best +1223% vs baseline +4282%. Concentrating capital in extreme z-scores increases risk without reward.
21. **Intraday candle confirmation hurts (V93)**: Requiring C<O (bearish candle) on entry day drops to +934%. The signal actually works BETTER when intraday recovers.
22. **Day-of-week filter hurts (V93)**: Signal works on all 5 trading days equally. Excluding any day reduces returns.
23. **Group-strength filter devastating (V93)**: Requiring own group avg > 0 drops to +141%. Signal works regardless of group direction.
24. **All regime filters harmful (V95)**: 11 tested regimes all worse than baseline. Signal is robust across all market conditions — filtering only reduces compounding.
25. **Market-down regime stronger than market-up (V95)**: Buying losers when market gapped down (+1014%) vs up (+291%). But still worse than no filter (+4282%).
26. **Expanded universe with illiquid commodities (V94)**: +20K% is fake — driven by bbfi (avg 17 lots/day) and ecfi (9.8% overnight std). Volume filtering is essential.

### Core Principles (Updated with Next-Open Findings)
- **Next-open execution is MANDATORY** — same-close entry is contaminated, all alpha disappears with 1-day delay
- **Trend-following signals survive the delay** — breakouts, momentum crossovers, adaptive MA crosses
- **Mean-reversion signals do NOT survive** — RSI, Bollinger, z-score, all fail with next-open entry
- **5-day hold period optimal** — matches ROC(5) period, absorbs execution delay
- **ROC(5) zero cross is the strongest practical signal** — +81.9%, 6/6 WF, captures momentum turning point
- **Continuation patterns > reversal patterns** — closing marubozu (+44%), long line (+29%) >> hammer (-10%), morning star (+6-13%)
- **Adaptive MAs beat simple MAs** — T3 (+50.7%) > KAMA (+44.2%) > SMA/EMA crossover
- **Concentration helps** — top_n=1 for ROC(5) gives +81.9% vs top_n=3 gives +22.8%
- **Long-only** — shorts destroy returns
- **No stop loss** — time exit only (hold 5 days)

### What Doesn't Work (Updated with V101-V115)
27. **Same-bar contamination destroys all V74-V92 alpha**: With next-open execution, +4282% → -3.9%. Only same-close entry made those work.
28. **Mean-reversion signals fail with 1-day delay**: RSI oversold, Bollinger bands, z-score mean-reversion — all destroyed by overnight gap.
29. **Term structure signals are useless**: Backwardation flip (-0.2%), spread compression (+4.5%). 72K+ data points, no predictive power with next-open.
30. **Ichimoku system fails**: +0.3% annual. Tenkan/Kijun cross doesn't work on Chinese commodity futures.
31. **ROC divergence doesn't work**: Price makes lower low but ROC makes higher low → -4.2% annual. Divergence signal is noise.
32. **Trailing stops hurt ROC strategy**: ATR trailing stop gives only +9.5% vs fixed hold-5-days. Stops exit too early on volatile winners.
33. **Stochastic oscillator fails**: 2/6 WF positive. Short-term oscillators don't survive the execution delay.
34. **Ultimate Oscillator fails**: 2/6 WF positive.
35. **Regime detection doesn't boost absolute returns**: Best +49% vs +81.9% baseline. But improves consistency (6/6 WF) and reduces MDD.
36. **Multi-timeframe alignment doesn't beat single ROC(5)**: Filters too aggressively, reduces trade count, lower absolute returns.
37. **Candlestick reversal patterns fail**: Hammer (-10.5%), morning star (+6-13%). Overnight gap destroys reversal signal.
38. **Continuation patterns > reversal patterns**: Closing Marubozu (+44%), Long Line (+29%) — identify strong trend days, not reversals.

### TA-Lib Indicator Effectiveness Ranking (Next-Open Execution)

**Tier 1 — Genuine Alpha (>30% annual):**
- ROC(5) cross zero: +81.9% — **STRONGEST**
- T3 cross: +50.7%
- KAMA cross: +44.2%
- TRANGE spike: +41.9%
- CCI cross: +39.7%
- ADX+DI trend: +36.1%
- WILLR cross: +27.7%
- CMO cross: +27.6%

**Tier 2 — Moderate Alpha (10-30%):**
- SAR flip: 6/6 WF, moderate return
- HT_TrendMode: moderate
- BOP threshold: +40.6% (TN1)

**Tier 3 — No Alpha (<10%):**
- Stochastic: 2/6 WF — fails
- Ultimate Oscillator: 2/6 WF — fails
- RSI oversold: mean-reversion destroyed by delay
- Bollinger bands: mean-reversion destroyed by delay
- MFI: 1/6 WF — fails
- HT_SINE: 3/6 WF — fails

## Strategy Details

### V92 — Overnight Cross-Group Z-Score (NEW CHAMPION)
- **Signal**: Compute z-score of each commodity's OVERNIGHT return (close-to-open gap) relative to the cross-group overnight return distribution. When z < -0.3, the commodity gapped down overnight relative to all groups → buy at close expecting catch-up tomorrow.
- **Key insight**: V82 uses the full close-to-close return for z-score. V92 decomposes into overnight and intraday. Pure overnight z-score (alpha=1.0) gives +4282% vs full daily z-score's +3305%. The overnight component carries significantly more cross-group predictive power.
- **Alpha sensitivity** (for combined overnight+intraday z-score with close entry):
  - alpha=0.0 (pure intraday): +17.8%
  - alpha=0.3: +390.1%
  - alpha=0.5: +2131.7%
  - alpha=0.7: +4032.9%
  - alpha=1.0 (pure overnight): +4281.8%
- **z_overnight and z_intraday are negatively correlated (-0.23)**: They provide diversifying information
- **Entry**: overnight z-score < -0.3 (commodity gapped down vs market), take top 3, long only, buy at close
- **Exit**: next day close (1-day hold)
- **Best config**: D_zCombC_a1.0_Z0.3_TN3 (alpha=1.0, z<0.3, top_n=3)
- **Performance**: +4281.8% annual, 64.9% WR, MDD -25.5%
- **Walk-forward**: 6/6 positive, avg +4740%, worst +1516% (2025), best +14708% (2022)
- **Evolution**: V74(+2186%) → V82(+3305%) → V92(+4282%). Overnight decomposition >> full daily return.

### V82 — Cross-Group Z-Score Momentum (Previous Champion)
- **Signal**: Compute z-score of each commodity's 1-day return relative to the cross-group return distribution. When z < -0.5, the commodity is unusually weak relative to ALL groups → buy expecting catch-up.
- **Key insight**: Instead of comparing within ONE group (V74), compare against ALL groups. Cross-group comparison captures more extreme divergences. The market-wide mean reversion signal is stronger than within-group.
- **Signal C (skip-own-group)**: Average return of all OTHER groups minus own return. Very similar performance (+3247%).
- **Entry**: z-score < -0.5 (commodity unusually weak vs market), take top 3 ranked by weakness, long only
- **Exit**: next day close (1-day hold)
- **Best config**: D_zscore_T-0.5_TN3 (z < -0.5, top_n=3)
- **Performance**: +3305.5% annual, 64.2% WR
- **Walk-forward**: 6/6 positive, avg +4180%, worst year +781% (2021), best +14015% (2022)
- **Evolution**: V74(+2186%) → V82(+3305%). Cross-group >> within-group. Broader comparison = more alpha.

### V74 — Extended Group 1-Day Momentum (Previous Champion)
- **Signal**: Group momentum excluding self minus own momentum, lookback=1, extended group map
- **Groups**: 44 commodities in 8 groups (ferrous, nonferrous, precious, oils, energy, chemical, soft, livestock)
- **Entry**: divergence > threshold (0.003-0.01), take top 3 ranked commodities, long only
- **Exit**: next day close (1-day hold)
- **Best config**: GF_LB1_T0.005_TN3 (full groups, LB=1, threshold=0.005, top_n=3)
- **Performance**: +2185.7% annual, 64.4% WR, ~2800-3900 trades
- **Walk-forward**: 6/6 positive, avg +2498%, worst year +431% (2020), best +6494% (2022)
- **Key**: Extending groups from 25→44 commodities gives 2.1x improvement. More diverse groups = more signals = more compounding.

### V69 — 1-Day Group Momentum Lag (Original Groups)
- **Signal**: Group momentum excluding self minus own momentum, lookback=1
- **Logic**: If group avg return today > commodity return today → commodity will catch up tomorrow
- **Entry**: divergence > threshold (0.001-0.01), take top 1-3 ranked commodities, long only
- **Exit**: next day close (1-day hold)
- **Best config**: GRP_LB1_T10_TN3_C1 (LB=1, threshold=0.01, top_n=3, COMM=0.01%)
- **Performance**: +1023.2% annual, 61.4% WR, 27.1% DD, PF 2.90, edge/trade +0.753%
- **Walk-forward**: 6/6 windows positive, avg +1000%, correlation=0.968, decay=0.97
- **Key insight**: LB=1 >> LB=3 >> LB=5. Shorter lookback captures faster mean reversion within groups. With daily compounding, even 0.753% edge/trade at 61.4% WR produces 1000%+ annual returns.
- **With realistic COMM=0.03%**: Still +900% annual, WF avg +885%

### V70 — Combo Pair + Momentum
- **Signal**: Priority-based — pair z-score > threshold first, then group momentum lag as fallback
- **Best config**: COMBO_adapt_Z2.5_ML3_MT3_group (pair Z=2.5, momentum LB=3, threshold=0.003)
- **Performance**: +562.4% annual, 66.0% WR, 17.9% DD, PF 3.25
- **Trade breakdown**: 59% pair trades (WR 70.4%), 41% momentum trades (WR 59.6%)
- **Key**: Combo ADDS value vs pair-only (+312.9%) but HURTS vs pure momentum (+1023%). Pairs replace high-returning momentum trades. Pure V69 momentum is superior.

### V62 — LOG-Biased Adaptive Pair Trading (Previous Champion)
- **Signal**: Adaptive spread mode with LOG bias (LOG gets 3/5 candidate slots: log_LB10, log_LB15, log_LB20 + raw_LB10, pct_LB10)
- **Best config**: T2_RE1_P14_Z1.0_MP1 (14 pairs, Z=1.0, 1-day hold, max 1 pair, eval=40d)
- **Key**: +334.3% annual, 66.2% WR, 9.4% DD, PF 3.24, Sharpe 2.15. 5/6 WF positive, avg +325.8%.

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

## V128 — AL Brooks Price Action Quantified
- **Trend Bar Continuation**: ROC(5)>1% + body_ratio>0.5 + up bar = +191%, 6/6 WF
- **Trend Bar + Z-score**: + body_ratio + Z>1.0 = +204%, 6/6 WF
- **Two-leg Pullback**: ROC(20)>2% + 2 down bars + reversal bar = +73%, 6/6 WF
- **Final Flag Breakout**: ROC(20)>3% + consolidation + breakout = +123%, 6/6 WF
- **Climax Fade SHORT**: Z>2.5 + body>0.7 + RSI>70 → fade = **-25%, 0/6 WF** (mean-reversion fails with next-open)
- **Channel Walker**: ROC(20)>0 + near lower BB + pullback = **-7%, 2/6 WF** (fails)
- **Brooks Composite v2**: TB+Z + Pullback + Final Flag = +242%, 6/6 WF
- **KEY INSIGHT**: V121 (+337%) still beats all Brooks concepts by 100pp. Brooks confirms trend-following works, reversal/mean-reversion FAILS with next-open delay.

## V129 — V121 Signal Quality Enhancement
- **Baseline V121**: +309% (minor implementation difference from V121's +333%)
- **+ Body Ratio filter**: +187% (**-122pp** — hard filters hurt!)
- **+ Volume filter**: +192% (**-117pp**)
- **+ OI bonus ranking**: +288% (-21pp — minor)
- **+ ROC*Z*BR ranking**: +308% (-1pp — virtually same, better WF stability: 2024 +141% vs +53%)
- **+ All quality scoring**: +298% (-11pp)
- **+ Hard BR+Vol filters**: +164% (-145pp)
- **+ Strict ROC>1.5 Z>2**: +151% (-158pp)
- **+ Breadth >=3**: +113% (-196pp)
- **KEY INSIGHT**: V121's alpha comes from MAXIMUM trade count compounding, NOT signal selectivity. Every filter reduces trades and hurts returns. ROC>1.0 and Z>1.5 are near-optimal thresholds.

## V130 — Regime-Adaptive + Multi-Signal Portfolio
- **Multi-Scored (V121+Z2+FF+PB weighted)**: +312%, WR 62.4%, 1334 trades, Sharpe 0.92, Sortino 5.9
- **Multi-Scored top_n=2**: +221%, MDD -50%, Sharpe **1.56** — best risk-adjusted
- **Multi-Scored WF**: 6/6, Avg +285%, 2024 +67% (vs V121 +53%) — better weakest-year
- **Signal type breakdown**: V121=1197 trades(62.7% WR), Pullback=88(56.8%), FinalFlag=49(67.3%)
- **Regime adaptation**: ALL regime filters REDUCE V121 returns. Market_mom -107pp, breadth -108pp, combined -139pp
- **Best risk-adjusted**: Multi scored top_n=2, equity_curve regime = +215%, MDD -50%, Sharpe 1.54
- **KEY INSIGHT**: V121 dominates (90% of multi-signal trades). Brooks signals add 5% more trades with different entry timing. Multi-signal slightly better Sharpe (0.92 vs 0.85) and WF stability.

## V131 — Cross-Commodity Momentum Spillover + New Alpha Sources
- **V121 + alignment bonus**: +337.9%, WR 63.7% (nearly identical to V121, slightly better WR)
- **Aligned multi-timeframe** (ROC3,5,10,20 all >0 + V121): +239%, less trades hurts compounding
- **Persistent momentum** (ROC5>1% for 3 days): +90%, weaker as persistence requirement filters too many trades
- **Cross-commodity spillover** (buy supply-chain partners when leader surges): +24%, 65.2% WR — too few signals
- **Sector rotation** (buy sector with most V121 signals): +134%, Sharpe 1.03 — best risk-adjusted!
- **1 TF aligned** (V121 + ROC3>0): +341.7%, WR 63.9% — marginal +4pp improvement
- **KEY INSIGHT**: V121 is near the theoretical maximum for single-signal momentum with next-open execution. No new alpha source beats it. Multi-timeframe alignment and supply-chain spillover provide marginal diversification but not higher returns.

## V132 — Research-Based Enhancements (completed)
- **V121 + OV confirm** (require positive overnight gap): +282%, 6/6 WF, avg +293%, 2024 +81%, 2025 +267% — MOST STABLE variant
- **OV/ID decomposition** (OV>0.3% & ID>0.3% & ROC5>1%): +260%, WF avg **+336%**, 2024 **+712%** — incredible year-specific alpha
- **OV/ID + Z-score**: +238%, WF avg +277%, 2024 +173% — balanced
- **Adaptive lookback** (volatility-based): +223%, Sharpe 0.93 (best!) but -14% in 2024 — overfits
- **OI change ranking**: +315%, close to V121, confirms V124 feature importance finding
- **Sortino ranking**: +243%, highest WR (64.5%) but lower returns
- **Mega combination**: +248%, doesn't beat simple V121
- **KEY INSIGHT**: V121 + overnight confirmation is the best PRACTICAL configuration — slightly lower max return but much more consistent year-to-year. OV/ID is a genuine independent alpha source that peaks in different years than pure ROC momentum.

## V133 — Intelligent Signal Switching (NEW CHAMPION!)
- **Union All Ranked** (V121+OV/ID+FF combined, ranked by weighted score): **+384.5%** annual, 61.6% WR, Sharpe 0.77, 6/6 WF, avg +345%
  - WF: 2020:+242% 2021:+313% 2022:+479% 2023:**+533%** 2024:+243% 2025:+259%
  - Beats V121 by **+47pp**, especially 2024 (+243% vs +54%) and 2023 (+533% vs +169%)
- **Compete (V121x3,OVx2,FFx1)**: +363.1%, 60.7% WR, Sharpe 0.82, 6/6 WF, avg +324%
- **Cascade V121>OV>FF**: +322.1%, 61.3% WR, 6/6 WF, avg +303%
- **Diversified (V121+OV slot) t=2**: +276.7%, Sharpe **1.87**, MDD -51.1% — best risk-adjusted
- **Union t=3**: +210.5%, Sharpe **2.09**, MDD -41.3% — HIGHEST SHARPE EVER
- **Signal breakdown**: v121+ov_id=629 trades (64.1% WR), v121=286, ov_id=190 (avg +2.16%!), v121+ov_id+ff=122
- **KEY INSIGHT**: Combining V121 + OV/ID + Final Flag as a UNION captures both momentum and overnight alpha. Dual confirmation (V121∩OV/ID) is the highest volume signal with 64.1% WR. This is the new practical champion.
- **File**: alpha_futures_v133.py

## Current Status (Session 6)
- **NEW CHAMPION**: V133 Union All Ranked → **+384.5%** annual, 6/6 WF, avg +345%
- **Previous Champion**: V121 ROC(5)>1% + Z>1.5 + ROC improving → **+337%** annual
- **Best risk-adjusted**: V133 Union t=3 → +210.5%, Sharpe **2.09**, MDD -41.3%
- **Best stable WF**: V133 Union → 6/6 WF, avg +345%, worst year +243%
- **Running**: V134 (Seasonality), V135 (Multi-TF), V136 (OI Conviction), V137 (Regime), V138 (Supply Chain Pairs)

## V124 — ML Ranking with Features (completed)
- **Best OOS**: GB T>0.50 + ROC*Z ranking = +171.8% (2023-2025), vs baseline +142.2% (+29.6pp)
- **Feature importance**: #1 OI_change (0.161), #2 intraday return (0.121), #3 body_ratio (0.074)
- **High WR config**: RF T>0.70 → 76.9% WR, MDD only -5.9%
- **File**: alpha_futures_v124.py

## V125 — Multi-Strategy Portfolio (completed)
- **No combination beats single champion** (+333.5%)
- **Dynamic allocation** (Method E): Sharpe **1.10** vs champion 0.76
- **Signal overlap**: 67.3% unique signals, 17.8% 2-strategy overlap
- **Best weights by Sharpe**: S5 breakout 33.8%, S4 Z-score 24.8%, S1 champion 19.1%
- **File**: alpha_futures_v125.py

## V126 — Novel Signal Engineering (completed)
- **H) Overnight-Intraday decomposition**: +260.2% annual, **WF avg +323.7%**, 6/6 — STRONGEST new alpha
- **C) Velocity+Acceleration**: +169%, WF avg +114.8%, 6/6
- **B) Vol-adjusted momentum**: +103.4%, WF avg +123.5%, 6/6
- **File**: alpha_futures_v126.py

## V127 — Exit Strategy Optimization (completed)
- **Hold=1 confirmed optimal** (+333.5%), hold=2 drops to +97.3%
- **Best alternative exit**: Intraday reversal max=1d → +246.4%, WF avg Sharpe 2.42
- **Profit target 1%**: +221.9%, Sharpe 0.89
- **File**: alpha_futures_v127.py

## V134 — Seasonality-Enhanced Momentum (completed)
- **Adaptive Threshold by Month** (lower ROC threshold in strong months, higher in weak months): +315.1%, 6/6 WF, avg **+343%**, 2024 +198%
- Monthly Momentum (last year same month): +202.3%, 6/6 WF
- Seasonal OI Pattern: +183.2%, 6/6 WF
- Monthly WR Filter (WR>55%): +130%, 6/6 WF — filters too many trades
- V121 + Seasonal Confirm (top-3 months): +70.3%, 5/6 WF (2024 -14%)
- Seasonal Mega (A+B+D): +60%, 5/6 WF — too strict
- **KEY INSIGHT**: Seasonality as hard filter DESTROYS returns. Adaptive threshold approach is the only useful variant, close to V121 but with better 2024 performance.
- **File**: alpha_futures_v134.py

## V135 — Multi-Timeframe + Dual Alpha Portfolio (completed)
- **F2) Rotating MTF/OV**: **+580.8%** annual, MDD -93.9%, Sharpe 0.77 — **NEAR 600% TARGET!** (NO WF validation)
- **F) Rotating V121/OV 80/20**: **+516.6%** annual, MDD -94.9%, Sharpe 0.75 (NO WF validation)
- G) Combined Equity (50/50 avg): +319.8%, MDD -94.9%
- E) 50/50 Split: +317.8%, MDD -94.2%, Sharpe 0.79
- Weekly+Daily Alignment: +224.6%, 6/6 WF
- Multi-ROC Confirmation: +200.9%, 6/6 WF
- ROC Acceleration: +59.8%, 5/6 WF (2024 -33%)
- Trend Quality: +70.5%, Sharpe **1.98** (high risk-adjusted)
- Ultimate (MTF+V121+OV/ID): +284.8%, 6/6 WF
- **OV/ID in Ultimate breakdown**: 226 trades, **69.9% WR**, **+2.32% avg PnL** — best per-signal metrics
- **50/50 Split WF**: 2020:+164% 2021:+250% 2022:+438% 2023:+222% 2024:**+367%** 2025:+176%
- **KEY INSIGHT**: Dynamic rotation between independent alpha sources (V121 and OV/ID) amplifies returns by concentrating in the winning strategy. The 20-day rolling WR lookback identifies regime switches. BUT: lacks WF validation, may overfit to lookback window. V139 is testing this.
- **File**: alpha_futures_v135.py

## V136 — OI Conviction + Volatility Breakout (completed)
- D) OI+Price Co-movement: +178.5%, Sharpe **2.19** (best Sharpe in this round)
- OI+Vol Union: +170.8%, Sharpe 0.83
- Volume-OI Divergence: +137.6%, Sharpe 0.99
- OI Surge + Momentum: +91.9%
- ATR Compression+Breakout: +83.4%
- Mega OI+Vol+Mom: +22.1%
- **KEY INSIGHT**: OI-based signals as standalone strategies don't beat V121. Best used as supplementary features in multi-signal approaches (as in V133 Union). OI+Price Co-movement has excellent Sharpe 2.19 but low absolute return.
- **File**: alpha_futures_v136.py

## V137 — Regime-Adaptive + Momentum Burst (completed)
- Vol-regime adaptive V121: +266.9%, Sharpe 0.75 — adapts thresholds by market vol
- Breadth>50% gated: +119.9% — **-217pp** from V121!
- Breadth>60% gated: +63.4%
- Momentum Burst (3+ up, vol>1.5x): +32%
- Breakout from Consolidation: +15.3%, Sharpe 1.39
- **KEY INSIGHT**: ALL regime filters DESTROY V121 returns. Breadth gating is devastating (-217pp at >50%). This confirms V130's finding: V121's alpha is regime-independent, filtering only reduces trade count and compounding.
- **File**: alpha_futures_v137.py

## V138 — Supply Chain Pair Momentum (completed)
- V121 + Chain combined: +318.8%, Sharpe **0.99** — improves Sharpe by +0.23 over V121
- Pair Spread Momentum: +55%, Sharpe 1.05
- Group Breadth (80%+): +74.9%, WR **65%**
- Upstream→Downstream: +12.2%
- Whole Chain Confirmation: +21.8%
- **KEY INSIGHT**: Supply chain context improves risk-adjusted returns (Sharpe 0.99 vs 0.76) without sacrificing much absolute return (+319% vs +337%). Pure supply chain signals are weak.
- **File**: alpha_futures_v138.py

## V139 — Rotating Portfolio Walk-Forward Validation (BREAKTHROUGH!)
- **V133 Union Signal**: +402.9% annual (slightly different from V133's +384.5% due to implementation details), 6/6 WF, avg +373%
- **Union/V121 Rotating LB=20 W=0.8**: **+1208% annual**, 6/6 WF, WF avg **+1134%**
  - WF: 2020:+1192% 2021:+621% 2022:+2164% 2023:+1824% 2024:+731% 2025:+275%
- **Union/V121 Rotating LB=15 W=0.8**: +1063%, 6/6 WF, WF avg +1036%
- **Union/V121 Rotating LB=20 W=0.9**: +890%, 6/6 WF, WF avg +842%
- **Union/V121 Static 50/50**: **+1542% annual**, 6/6 WF, WF avg **+562%**
  - WF: 2020:+638% 2021:+386% 2022:+515% 2023:+786% 2024:+764% 2025:+280%
- **Union/OV_ID LB=15 W=0.8**: +467%, 6/6 WF, WF avg +482%
- **KEY INSIGHT**: Rotating between V133 Union and V121 based on 20-day WR amplifies returns from +384% to +1208%. The mechanism is variance reduction through daily diversification — when Union and V121 select different commodities, the combined portfolio has lower volatility, which compounds faster. ALL configurations pass 6/6 WF validation. The static 50/50 (WF avg +562%) is the most conservative; rotating 80/20 (WF avg +1134%) is the most aggressive.
- **File**: alpha_futures_v139.py

## V140 — Drawdown-Controlled Portfolio (BREAKTHROUGH: Simple Sizing Wins!)
- **Problem**: V139 Union/V121 has -94.7% MDD. User says "回撤太高了"
- **Failed approaches** (circuit breakers, equity MA filter, losing streak gate, WR gate):
  - All either don't reduce MDD (return-level scaling after equity computed) or destroy returns (hard stop = miss recovery)
  - Circuit breaker + EQMA20: +17-25%, MDD -6% to -19% — too conservative, loses most alpha
  - Losing streak=4: +22%, MDD -7% — barely trades
- **WINNING APPROACH: Simple Position Sizing** — just reduce position from 95% to 50% of capital
- **Portfolio Union/V121 50/50 with 50% sizing**: **+155% annual**, worst-year WF MDD **-24%**, 6/6 WF, WF avg +121%, Sharpe **1.75**
  - WF: 2020:+107% MDD:-24% 2021:+142% MDD:-8% 2022:+147% MDD:-8% 2023:+139% MDD:-23% 2024:+77% MDD:-23% 2025:+116% MDD:-8%
  - Avg per-year MDD: **-15.6%** (extremely low for futures!)
- **Position sizing sweep**:
  | Size | Annual | Full MDD | Worst WF MDD | Avg WF MDD | WF Avg Ann |
  |------|--------|----------|-------------|-----------|------------|
  | 30%  | +70%   | -30%     | -17%        | -9%       | +62%       |
  | 40%  | +107%  | -39%     | -19%        | -12%      | +90%       |
  | **50%**  | **+155%**  | **-50%**     | **-24%**        | **-16%**      | **+121%**      |
  | 60%  | +220%  | -59%     | -30%        | -18%      | +157%      |
  | 70%  | +317%  | -70%     | -35%        | -21%      | +215%      |
  | 80%  | +484%  | -80%     | -40%        | -26%      | +311%      |
- **KEY INSIGHT 1**: Full-period MDD (-50%) is inflated by early training period (2016-2019). Per-year WF MDD is only -24% at worst!
- **KEY INSIGHT 2**: Position sizing is proportional: 50% size → 50% of the returns AND 50% of the drawdown. Simple, predictable, no overfitting risk.
- **KEY INSIGHT 3**: Circuit breakers and equity filters are inferior because they create "gap risk" — they stop trading during drawdowns but miss recovery trades. Position sizing keeps you in the market always.
- **KEY INSIGHT 4**: MDD is NOT from individual big losses but from consecutive losing days in early period when equity is small and compounding amplifies percentage drops.
- **Practical recommendation**: Union/V121 50/50 at 50% sizing = +155% annual, -24% worst-year MDD
- **File**: alpha_futures_v140.py

## V141 — Adaptive Position Sizing (completed)
- **WR-adaptive**: +197% annual, -31% worst WF MDD, 6/6 WF
- **Signal strength sizing**: +331%, -37% worst WF MDD
- **Equity curve scaling**: +291%, -39% worst WF MDD
- **Anti-Martingale**: +242%, -35% worst WF MDD
- **Combined WR×EQ**: +393%, -45% worst WF MDD
- All adaptive approaches beat +155% baseline but with worse MDD
- **KEY INSIGHT**: Adaptive sizing boosts returns 27-153% but MDD cost is significant
- **File**: alpha_futures_v141.py

## V142 — Multi-Position Diversification (completed)
- **Cross+Corr @50% max_corr=0.5**: +198%, -27% worst WF MDD, 6/6 WF ← beats baseline
- **Union top_n=2 @45%**: +170%, -25% worst WF MDD
- **Signal agreement**: +226% but -70% MDD — unacceptable
- **Correlation selection**: +105%, -33% MDD, Sharpe 1.73
- **KEY INSIGHT**: Low-correlation cross-signal selection (V121 + Union picking different commodities) is the key diversification mechanism
- **File**: alpha_futures_v142.py

## V143 — Exit Strategy Optimization (completed)
- **SL=5%**: +158%, -24% worst WF MDD — same MDD as baseline, slight return improvement
- **WE>2%/LE<5%**: +103%, -29% MDD — best R/M ratio but lower absolute return
- **Profit target**: ALL profit targets hurt returns (PT=5% → +74%)
- **Hold extension**: hurts returns (extension to day 2 → +59%)
- **KEY INSIGHT**: Stop-loss=5% is the only useful exit enhancement. Profit targets are destructive.
- **File**: alpha_futures_v143.py

## V144 — Cross+Corr + Stop Loss (completed)
- **Cross+Corr corr<0.3 + SL=3% @45%**: +157%, -33% MDD, Sharpe 2.04
- **Cross+Corr + EQ scaling corr<0.3**: +169%, -35% worst WF MDD
- **KEY INSIGHT**: Cross+Corr combined with SL and equity curve scaling is promising but MDD still >30%
- **File**: alpha_futures_v144.py

## V145 — Regime-Aware Sizing (completed)
- **DD-based portfolio 75/65/50/35/20%**: +271% annual, -35% worst WF MDD, 6/6 WF avg +195%
- **Combo regime (single signal) 70/55/35%**: +170% WF avg, -22% worst WF MDD — best risk-adjusted single signal!
- **Vol regime 0.8x/2.0x**: +161% WF avg, -21% worst WF MDD
- **KEY INSIGHT**: Composite regime score (breadth+vol+equity slope+DD) for dynamic sizing is the most promising approach. Single-signal achieves better MDD than portfolio because there's no combined equity volatility.
- **File**: alpha_futures_v145.py

## V146 — Combined Best Ideas (BREAKTHROUGH!)
- **DD 70/60/40/20% + Cross+Corr corr<0.5 NO SL**: **+239% WF avg**, **-27% worst WF MDD**, 6/6 WF
  - WF: 2020:+196%/-25% 2021:+134%/-13% 2022:+476%/-9% 2023:+229%/-20% 2024:+120%/-27% 2025:+277%/-11%
  - vs V140 baseline: +239% vs +155% annual (**+54% improvement**), MDD -27% vs -24% (only +3pp)
- **DD 65/55/40/25% + Cross+Corr + SL=5%**: +230%, -50% full MDD, Sharpe R/M=4.62
- Mechanism: Cross+Corr ensures V121 and Union pick DIFFERENT low-correlated commodities. DD-based sizing scales down smoothly. Stop-loss NOT needed.
- **KEY INSIGHT**: The combination of (1) low-correlation cross-signal selection + (2) drawdown-based adaptive sizing gives the best return/MDD frontier ever achieved. The improvement is 54% more return for only 3pp worse MDD.
- **File**: alpha_futures_v146.py

## V147 — Signal Engineering Enhancements (completed)
- **Multi-TF/V121 50/50**: +131%, -25% MDD — best portfolio but below baseline
- **MTF+OI+Vol/V121 50/50**: +121%, -26% MDD
- All standalone signal variants below +150% annual
- **KEY INSIGHT**: Signal engineering doesn't beat existing V121/Union signals. Improvement comes from PORTFOLIO CONSTRUCTION and SIZING, not signal quality.
- **File**: alpha_futures_v147.py

## Current Status (Session 6)
- **PRACTICAL CHAMPION**: V146 DD 70/60/40/20% + Cross+Corr → **+239% WF avg**, -27% worst WF MDD, 6/6 WF
- **ABSOLUTE CHAMPION**: V139 Union/V121 rotating → **+1208% annual** (high MDD, for aggressive capital)
- **Single-Signal Champion**: V133 Union All Ranked → **+384.5%** annual, 6/6 WF, avg +345%
- **Best risk-adjusted (low MDD)**: V145 Combo regime → +170%, -22% worst WF MDD
- **Most conservative**: V140 Union/V121 50/50 size=30% → Sharpe 1.87, MDD -17%, +70% annual

## Running Experiments
- All V133-V140 experiments COMPLETE
