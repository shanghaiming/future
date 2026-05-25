# Price Action (PA) -- Actionable Summary for Futures

## 1. Core Concepts: Trend / Reversal / Zone

### 1.1 Trend (Al Brooks "Trading Price Action Trends")

**EMA 20 as the primary reference line.** Every bar's relationship to EMA defines the trend state:

| Condition | Classification |
|-----------|---------------|
| Price above EMA + EMA slope rising | Uptrend |
| Price below EMA + EMA slope falling | Downtrend |
| Price oscillating around EMA | Trading range |

**Bar Counting System (High N / Low N)** is the core edge:
- **High 1**: First bar in an uptrend whose low touches/crosses below EMA (pullback starts).
- **High 2**: Price recovers above EMA, then pulls back again -- the highest-probability trend continuation entry.
- **High 3**: Third pullback -- often a wedge flag, lower quality.
- **Low 1/2/3**: Mirror logic for downtrend (bounces to EMA).

Signal quality ranking (Brooks Chapter 11):
1. High2 + 20-gap bar (body entirely above EMA) = quality 9
2. High2 without gap = quality 7
3. Spike pullback (first pullback after a strong move) = quality 6
4. EMA pin bar (reversal bar at EMA during High1) = quality 6
5. High3 wedge = quality 5

**20-gap bar**: When the entire bar body (open to close) sits on one side of EMA without touching it. This confirms a strong, persistent trend. In code: `min(close, open) > ema` (bullish) or `max(close, open) < ema` (bearish).

**Trend Strength Classification**:
- **Spike**: Strong directional move (consecutive large-body bars, little overlap). High probability continuation.
- **Channel**: Trending but with overlapping bars, pullbacks frequent. Trade the pullbacks.
- **Weak**: Frequent EMA crosses, no clear direction. Avoid or use range tactics.

### 1.2 Reversal (Al Brooks "Trading Price Action Reversals")

**Key patterns with highest reliability:**

1. **Double Top/Bottom**: Two peaks/troughs at approximately the same price level (within 2% tolerance). Requires a neckline break for confirmation. Statistical edge: the measured move target (distance from peak to neckline, projected from breakout) is achieved ~60% of the time.

2. **Head and Shoulders**: Three peaks with the middle one highest. The neckline break confirms. More reliable than double top because it shows distribution (institutional selling into strength).

3. **Climax Reversal**: A series of large-range bars (>2x average range) in the trend direction followed by a reversal bar. In code, tracking consecutive climax bars: 3+ consecutive climax bars = high-probability exit signal (quality 9).

4. **Wedge Reversal**: Three pushes in the trend direction where each push is smaller (rising wedge = bearish, falling wedge = bullish). This is essentially a High3/Low3 that fails.

**Multi-dimensional confirmation framework** (from the confirmation system code):
- Volume spike (> avg + 2*std) at pattern completion
- Price action: breakout of support/resistance with ATR threshold
- Multi-timeframe: trend alignment across timeframes using regression slope
- Momentum divergence: price makes new extreme but RSI does not (most powerful)

Confirmation strength levels: weak (1 dimension), moderate (2), strong (3), very_strong (all 4). Only trade on moderate+.

### 1.3 Zone / Trading Range (Al Brooks "Trading Price Action Trading Ranges")

**Market Structure Identification:**

Swing point detection uses dual approach:
1. Windowed comparison: A bar is a swing high if its close is higher than the 5 bars before and after it.
2. ATR threshold: Only count swing points where the move exceeds 1.5 * ATR (filters noise).

**Structure type classification** (using linear regression on swing point prices):
- **Uptrend**: Sequential higher swing highs and higher swing lows
- **Downtrend**: Sequential lower swing highs and lower swing lows
- **Range**: Swing highs and lows oscillate within a horizontal band
- **Transition**: Structure changing from one type to another
- **Complex**: No clear pattern

**Trading rules by structure:**
- In a trend: trade pullbacks (High2/Low2 entries)
- In a range: buy at range bottom, sell at range top (mean reversion)
- On structure break: trade the breakout direction

**Structure integrity** is measured by R-squared of the regression line. R-squared > 0.7 = strong trend, < 0.3 = range/weak.

---

## 2. Key Candlestick Patterns and Statistical Significance

Based on the Brooks methodology and implementation, the actionable patterns are:

| Pattern | Definition | Reliability | Signal |
|---------|-----------|-------------|--------|
| Strong bull bar | Body ratio > 55% of range, close > open | Moderate | Trend continuation |
| Strong bear bar | Body ratio > 55% of range, close < open | Moderate | Trend continuation |
| Bull reversal bar | Lower wick >= 50% of range, upper wick <= 25%, bullish close | High (at EMA) | Buy in uptrend pullback |
| Bear reversal bar | Upper wick >= 50% of range, lower wick <= 25%, bearish close | High (at EMA) | Sell in downtrend bounce |
| 20-gap bar | Entire body on one side of EMA | High | Trend strength confirmation |
| Climax bar | Range > 2x average 20-bar range | High (when consecutive) | Trend exhaustion |
| Double top/bottom | Two extremes within 2% price tolerance, neckline break | ~60% hit target | Reversal entry |
| Pin bar at EMA | Reversal bar formation touching EMA during pullback | High | Early trend re-entry |

**Important caveat**: Single candlestick patterns have low predictive power in isolation. They become statistically significant only in context (at EMA in a trend, at support/resistance, with volume confirmation).

---

## 3. Identifying Institutional Order Flow

Institutions leave footprints that PA detects:

1. **Spike bars**: A sudden large-range bar (range > 2x average) indicates institutional urgency. When multiple spikes occur in the same direction, institutions are building a position.

2. **Breakout with volume**: Volume > avg + 2*std at a key level = institutional participation. Without volume, breakouts fail more often.

3. **Failure at extremes**: When price reaches a new high/low but immediately reverses with a strong opposite bar, institutions were taking profit (distribution/accumulation complete).

4. **Consecutive climax bars**: 3+ large bars in one direction = institutions completing their move. The subsequent reversal is institutional repositioning.

5. **EMA behavior**: In a strong institutional trend, price stays on one side of EMA and pullbacks are shallow. When EMA slope flattens and price starts crossing EMA frequently, institutions have stopped driving the trend.

6. **False breakouts**: Price briefly exceeds a swing point by < 1.5 * ATR then reverses. This is institutional stop-running -- they triggered stops to fill their own positions in the opposite direction.

---

## 4. Codable Signal Rules

These rules are extracted directly from the implementation and can be used as strategy building blocks.

### Entry Rules

```python
# Rule 1: High2 buy (highest probability trend continuation)
# Conditions: trend is up (price > EMA, EMA slope > 0)
#             pullback count == 2 (two touches of EMA completed)
#             not currently in pullback (price recovered above EMA)
# Quality: 7 (9 if 20-gap bar present)

# Rule 2: Low2 sell (mirror of High2 for shorts)
# Conditions: trend is down (price < EMA, EMA slope < 0)
#             bounce count == 2
#             price resumed below EMA
# Quality: 7 (9 if 20-gap bar present)

# Rule 3: EMA pin bar (early entry during first pullback)
# Conditions: pullback count == 1, currently in pullback
#             bar is a reversal bar (long wick toward EMA side)
# Quality: 6

# Rule 4: Structure breakout
# Conditions: price breaks above/below swing point by > 1.5 * ATR
#             volume > avg + 2*std
# Quality: 7
```

### Exit Rules

```python
# Rule 1: Consecutive climax exit (quality 9)
# Conditions: 3+ consecutive bars with range > 2x average AND strong body

# Rule 2: EMA slope flip exit (quality 6)
# Conditions: EMA slope flips negative AND price below EMA for 3+ bars (long)
#             OR EMA slope flips positive AND price above EMA for 3+ bars (short)

# Rule 3: Trailing stop
# Conditions: 2.5 * ATR from high-water mark (long)
#             OR 2.5 * ATR from low-water mark (short)

# Rule 4: Max hold
# Conditions: position held for 60 bars maximum
```

### Bar Classification Rules

```python
# Strong bar: body_ratio = abs(close - open) / (high - low) > 0.55
# Bull reversal: lower_wick >= 50% of range, upper_wick <= 25%, close > open
# Bear reversal: upper_wick >= 50% of range, lower_wick <= 25%, close < open
# 20-gap bar: entire body on one side of EMA (min(close,open) > ema or max(close,open) < ema)
# Climax bar: (high - low) > 2.0 * 20-bar average range
```

### Default Parameters (Brooks Standard)

| Parameter | Value | Notes |
|-----------|-------|-------|
| EMA period | 20 | Brooks standard |
| ATR period | 14 | Standard |
| Strong bar body ratio | 0.55 | >55% body-to-range |
| Climax range multiplier | 2.0 | 2x average range |
| Max climax count | 3 | Exit after 3 consecutive |
| Swing lookback | 5 bars | For swing point detection |
| Structure break threshold | 1.5 * ATR | Filters false breakouts |
| Volume spike threshold | avg + 2*std | Institutional detection |

---

## 5. Futures-Specific Considerations

### What works differently for futures (T+0, continuous trading):

1. **EMA period adjustment**: Futures intraday data is more noisy. Use shorter EMA periods (10-15 for 5-min bars, 20 for daily). The 20-gap bar concept works well on any timeframe.

2. **Session gaps**: Overnight gaps in futures create artificial "climax" bars. Filter: only count climax bars that occur within the same trading session. Alternatively, use session-adjusted ATR.

3. **Continuous contract roll**: When rolling contracts, the price series has discontinuities. Use ratio-adjusted or difference-adjusted continuous series. PA patterns (double tops, swing points) are distorted across roll dates.

4. **Higher noise ratio**: Futures tick data has more false breakouts. Increase the structure break threshold from 1.5 * ATR to 2.0 * ATR. Increase volume spike threshold from 2*std to 2.5*std.

5. **Leverage amplifies stop placement**: The 2.5 * ATR trailing stop is appropriate for stocks. For futures, consider 1.5-2.0 * ATR due to leverage and the resulting risk.

6. **Intraday vs daily**: Brooks' bar counting works on any timeframe. For intraday futures, the "day" concept maps to a session. High2 within a session is a valid scalp signal. Across sessions, use daily bar counting.

### What transfers directly from equities:

- The entire bar counting framework (High N / Low N)
- Reversal pattern recognition (double top/bottom, H&S)
- EMA slope as trend indicator
- Volume spike detection for institutional flow
- Multi-confirmation framework (volume + price action + momentum)
- Structure integrity scoring (R-squared)

### What needs modification:

- Position sizing: futures use fixed contract sizes, not percentage of equity
- Stop loss: tighter due to leverage (reduce ATR multiplier)
- Session handling: must account for trading session boundaries
- Volume: use tick volume or open interest changes instead of share volume
- Bar construction: use rule-based rollover for continuous series

---

## Implementation Reference

The primary code implementation is in `brooks_pa_strategy.py` (BrooksPAStrategy class). Key methods:

- `_analyze_stock()`: Core analysis engine -- computes EMA, ATR, classifies bars, tracks pullback counts, generates signals with quality scores
- `generate_signals()`: Time-iterative signal loop with position tracking and risk management
- `_compute_ema()`, `_compute_atr_array()`: Pure numpy indicator calculations (no dependency on external libraries)

Supporting implementations:
- `price_action_reversals_reversal_pattern_recognition.py`: Double top/bottom pattern detection with confidence scoring
- `price_action_ranges_market_structure_identifier.py`: Swing point detection and structure classification
- `price_action_reversals_reversal_confirmation_system.py`: Multi-dimensional confirmation framework
