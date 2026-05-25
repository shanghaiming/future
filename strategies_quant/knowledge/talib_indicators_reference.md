# TA-Lib Indicators Reference for Futures Trading

TA-Lib v0.6.8 -- 158 functions across 8 categories. This reference focuses on indicators that are genuinely useful for futures strategy development, with practical notes on parameters and interpretation.

## Table of Contents
- [Moving Averages (Overlap Studies)](#moving-averages)
- [Momentum Indicators](#momentum-indicators)
- [Volatility Indicators](#volatility-indicators)
- [Volume Indicators](#volume-indicators)
- [Trend Strength Indicators](#trend-strength)
- [Statistics Functions](#statistics-functions)
- [Cycle Indicators](#cycle-indicators)
- [Candlestick Patterns](#candlestick-patterns)
- [Combinations That Work](#combinations-that-work)
- [Implementation Notes](#implementation-notes)

---

## Moving Averages

The foundation of most strategies. TA-Lib offers 9 MA types via the `matype` parameter (integer 0-8):

| matype | Type | Characteristics |
|--------|------|----------------|
| 0 | SMA | Simple, equal weight, most lag |
| 1 | EMA | Exponential weight, responsive |
| 2 | WMA | Weighted linear, less lag than SMA |
| 3 | DEMA | Double EMA, very responsive |
| 4 | TEMA | Triple EMA, extremely responsive |
| 5 | TRIMA | Triangular, extra smoothed |
| 6 | KAMA | Kaufman adaptive, adjusts to noise |
| 7 | MAMA | Mesa adaptive, frequency-based |
| 8 | T3 | Triple generalization DEMA, smoothest |

### Key Functions

**SMA / EMA** -- The workhorses. Use EMA for faster response, SMA for cleaner signals.
```python
sma = talib.SMA(close, timeperiod=20)
ema = talib.EMA(close, timeperiod=20)
```
Common periods: 5 (micro), 10 (short), 20 (standard), 50 (medium), 60 (for hourly bars = 1 week), 120/200 (long).

**KAMA** -- Kaufman Adaptive Moving Average. Automatically adjusts its speed based on market noise. Excellent for futures because it filters choppy ranges while tracking trends closely. Period 10 is a good default.

**BBANDS** -- Bollinger Bands. Returns upper, middle (SMA), lower bands.
```python
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
```
Default 20-period, 2 standard deviations. For futures, try (20, 2.5) or (10, 1.5) for tighter bands. Useful for mean-reversion entries and volatility expansion detection.

**SAR** -- Parabolic SAR. Excellent trailing stop for trend-following.
```python
sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
```
Acceleration starts at 0.02, increments by 0.02, caps at 0.2. Increase acceleration (0.04) for faster exits in volatile markets.

**DEMA / TEMA** -- Faster alternatives to EMA. Good for crossover systems where lag reduction matters. DEMA is usually sufficient; TEMA can be too responsive.

---

## Momentum Indicators

30 functions -- the largest category. These measure rate of price change.

### Core Momentum

**RSI** -- Relative Strength Index. The most widely used oscillator.
```python
rsi = talib.RSI(close, timeperiod=14)
```
Range 0-100. >70 overbought, <30 oversold. For futures, 14-period is standard but 2-period (Connors RSI style) works for short-term mean-reversion. RSI divergence from price is one of the strongest signals.

**MACD** -- Moving Average Convergence Divergence.
```python
macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
```
Returns three values: MACD line (fast EMA - slow EMA), signal line (EMA of MACD), histogram (MACD - signal). Default 12/26/9 is universal. Histogram crossovers are more timely than line crossovers. MACD is a lagging indicator -- combine with leading indicators.

**STOCH** -- Stochastic Oscillator.
```python
slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
```
Range 0-100. Measures where close sits within recent high-low range. Good for range-bound markets, poor in strong trends. Use (14,3,3) for slower signals, (5,3,3) for faster.

**MOM / ROC** -- Simple momentum measures.
```python
mom = talib.MOM(close, timeperiod=10)  # absolute difference
roc = talib.ROC(close, timeperiod=10)  # percentage change
```
Useful as raw inputs to other calculations. ROC is preferred for cross-instrument comparison.

### Secondary Momentum

**CCI** -- Commodity Channel Index. Originally designed for futures.
```python
cci = talib.CCI(high, low, close, timeperiod=14)
```
Unbounded, typically trades between -200 and +200. >100 overbought, <-100 oversold. Good for identifying cyclical turns in futures. Uses typical price (H+L+C)/3.

**WILLR** -- Williams %R. Similar to Stochastic but inverted (0 to -100).
```python
willr = talib.WILLR(high, low, close, timeperiod=14)
```
Fast oscillator, good for short-term entry timing. >-20 overbought, <-80 oversold.

**TRIX** -- Triple-smoothed EMA rate of change. Excellent noise filter.
```python
trix = talib.TRIX(close, timeperiod=30)
```
Eliminates short-term noise. Zero-line crossovers are trend change signals.

**ULTOSC** -- Ultimate Oscillator. Combines 3 timeframes (7, 14, 28).
```python
ultosc = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
```
Reduces false signals by weighting multiple periods. Divergence signals are reliable.

**CMO** -- Chande Momentum Oscillator.
```python
cmo = talib.CMO(close, timeperiod=14)
```
Range -100 to +100. More sensitive than RSI. Good for detecting regime changes.

---

## Trend Strength

These belong to Momentum in TA-Lib classification but deserve their own section for futures.

**ADX** -- Average Directional Index. THE trend strength indicator.
```python
adx = talib.ADX(high, low, close, timeperiod=14)
```
Range 0-100. Below 20 = no trend (range), above 25 = trending, above 50 = strong trend. Critical for strategy selection: use trend-following when ADX > 25, mean-reversion when ADX < 20.

**PLUS_DI / MINUS_DI** -- Directional Movement. Used with ADX.
```python
plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
```
+DI > -DI = bullish pressure, -DI > +DI = bearish pressure. Crossovers give directional entry signals. Always use alongside ADX to confirm trend exists.

**AROON** -- Aroon Up/Down.
```python
aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
```
Measures time since highest high / lowest low. Aroon Up > 70 and Aroon Down < 30 = strong uptrend. Both near 50 = consolidation. Good for early trend detection.

---

## Volatility Indicators

Critical for futures position sizing and stop placement.

**ATR** -- Average True Range. Essential for futures.
```python
atr = talib.ATR(high, low, close, timeperiod=14)
```
Measures average bar range including gaps. Uses: stop-loss placement (1.5-3x ATR), position sizing (inverse ATR), volatility filter (ATR percentile). 14-period is standard. Always use ATR-based stops, not fixed-point stops.

**NATR** -- Normalized ATR (ATR/close * 100). Useful for cross-contract comparison.
```python
natr = talib.NATR(high, low, close, timeperiod=14)
```
Compare volatility across different futures contracts (e.g., 5-min vs 1-hour, or gold vs crude).

**TRANGE** -- True Range (raw, not averaged).
```python
trange = talib.TRANGE(high, low, close)
```
Max of: (high-low), |high-previous_close|, |low-previous_close|. Building block for ATR.

---

## Volume Indicators

Volume confirmation is important but note: futures volume is tick-based and reported next-day in some markets.

**OBV** -- On Balance Volume. Cumulative volume with sign.
```python
obv = talib.OBV(close, volume)
```
OBV trending up while price flat = accumulation. OBV divergence from price is a leading signal. Simple but effective.

**AD** -- Chaikin Accumulation/Distribution.
```python
ad = talib.AD(high, low, close, volume)
```
Uses money flow multiplier based on close position within the bar range. Better than OBV for futures because it accounts for where price closed within the bar.

**ADOSC** -- Chaikin A/D Oscillator.
```python
adosc = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
```
MACD-style smoothing of AD line. Zero-line crossovers signal money flow shifts.

---

## Statistics Functions

Useful for quantitative strategy development.

**STDDEV** -- Rolling standard deviation.
```python
stddev = talib.STDDEV(close, timeperiod=5, nbdev=1)
```
Foundation of volatility calculations and z-score based signals.

**LINEARREG_SLOPE** -- Rolling linear regression slope.
```python
slope = talib.LINEARREG_SLOPE(close, timeperiod=14)
```
Quantifies trend direction and strength numerically. Positive = uptrend, negative = downtrend. Magnitude indicates trend strength. More precise than visual assessment.

**LINEARREG_ANGLE** -- Slope converted to angle in degrees. Useful for regime detection.

**BETA** -- Rolling beta between two series. Useful for spread trading and hedging.

**CORREL** -- Rolling Pearson correlation. Essential for pair trading and portfolio risk.

---

## Cycle Indicators

Hilbert Transform-based. Advanced and mathematically complex.

**HT_TRENDMODE** -- Returns 0 (no trend) or 1 (trending). Useful as a regime filter.
```python
trendmode = talib.HT_TRENDMODE(close)
```

**HT_DCPERIOD** -- Dominant cycle period. Useful for adaptive strategy parameter selection.

**HT_SINE / HT_PHASOR** -- Generate sine wave overlays. Primarily for cycle analysis. Not commonly used in practical futures trading.

---

## Candlestick Patterns

61 patterns. They return: +100 (bullish), -100 (bearish), 0 (none). Require OHLC data.

### Most Reliable for Futures

1. **CDLENGULFING** -- Engulfing pattern. Strong reversal signal.
2. **CDLHAMMER / CDLSHOOTINGSTAR** -- Classic single-bar reversals.
3. **CDLDOJI** -- Indecision bar, significant at support/resistance.
4. **CDL3BLACKCROWS / CDL3WHITESOLDIERS** -- Three-bar continuation/reversal.
5. **CDLMORNINGSTAR / CDLEVENINGSTAR** -- Three-bar reversal patterns with gaps.
6. **CDLHARAMI** -- Inside bar pattern, suggests consolidation before continuation.

### Usage Notes
- Patterns work best at key support/resistance levels.
- Always confirm with volume and trend context.
- On intraday futures data, many patterns generate too many false signals. Prefer daily bars.
- Do not use patterns as standalone signals -- combine with indicators.

---

## Combinations That Work

### Trend-Following System
```
Entry:  EMA(20) crossover EMA(50) + ADX(14) > 25
Stop:   SAR or 2x ATR(14) trailing
Filter: +DI > -DI for longs, -DI > +DI for shorts
```

### Mean-Reversion System
```
Entry:  RSI(14) < 30 + price touches lower BBAND(20,2)
Exit:   RSI(14) > 70 or price touches upper band
Filter: ADX(14) < 20 (range-bound only)
```

### Momentum Breakout
```
Entry:  Price > upper BBAND(20,2) + ATR(14) expanding + OBV trending up
Stop:   2x ATR(14) from entry
Filter: ADX(14) rising (trend beginning)
```

### Volatility Regime Detection
```
High vol:  ATR(14) > percentile_75 of 50-day ATR
Low vol:   ATR(14) < percentile_25 of 50-day ATR
Regime:    HT_TRENDMODE for trend vs cycle classification
```

### Proven Indicator Pairings
| Indicator A | Indicator B | Why |
|---|---|---|
| EMA crossover | ADX filter | EMA gives direction, ADX confirms trend exists |
| RSI | Bollinger Bands | RSI for momentum, BB for price context |
| MACD | ATR | MACD for entries, ATR for stop sizing |
| CCI | AD/DOSC | CCI for timing, ADOSC for volume confirmation |
| LinearReg Slope | NATR | Slope for direction, NATR for cross-contract comparison |

---

## Implementation Notes

### Lookback Periods by Timeframe

| Timeframe | Fast | Standard | Slow |
|-----------|------|----------|------|
| 1-min / 5-min | 5 | 10-14 | 20 |
| 15-min / 30-min | 10 | 20 | 50 |
| Hourly | 10 | 20 | 60 |
| Daily | 14 | 20 | 50-60 |
| Weekly | 10 | 20 | 40 |

### Data Requirements
- All functions need numpy arrays (float64), not lists.
- Functions return numpy arrays of same length as input. Initial values are NaN (warm-up period).
- Always skip the first `3 * longest_period` bars to avoid NaN contamination.
- For functions needing OHLC: `talib.RSI(close)` vs `talib.STOCH(high, low, close)`.

### Common Pitfalls
1. **Lookahead bias**: Do not use current bar's indicator value for current bar decisions. Use `indicator[i-1]` for signal at bar `i`.
2. **Over-parameterization**: More indicators does not mean better signals. Use 2-4 indicators maximum per strategy.
3. **Ignoring NaN warmup**: Always check `np.isnan()` or slice output after the warmup period.
4. **Same-family redundancy**: Do not combine RSI + Stochastic + CCI -- they measure similar things. Pick one.
5. **Period optimization trap**: Optimizing indicator periods to past data leads to overfitting. Use standard periods.

### Performance
- TA-Lib is C-optimized, very fast even on large arrays.
- Vectorized: compute on entire array at once, do not loop bar-by-bar.
- For real-time: compute on full array, use only the last value.

### Quick Usage Pattern
```python
import talib
import numpy as np

# Compute
atr = talib.ATR(high, low, close, timeperiod=14)
rsi = talib.RSI(close, timeperiod=14)
macd, signal, hist = talib.MACD(close, 12, 26, 9)

# Current values (skip NaN warmup)
idx = -1
current_atr = atr[idx]
current_rsi = rsi[idx]

# Signal example
if not np.isnan(rsi[idx-1]) and not np.isnan(atr[idx-1]):
    if rsi[idx-1] < 30 and close[idx-1] < close[idx-2]:
        stop_loss = close[idx] - 2 * atr[idx]
        # enter long
```

---

## Indicator Selection Guide for Futures

### Always Use
- **ATR**: Position sizing, stop placement, volatility measurement. Non-negotiable.
- **EMA or SMA**: Trend direction, support/resistance, crossover signals.

### Often Useful
- **RSI**: Overbought/oversold, divergence detection.
- **ADX**: Trend vs range regime filter.
- **Bollinger Bands**: Volatility envelope, mean-reversion context.
- **MACD**: Trend momentum confirmation.

### Situational
- **SAR**: Trailing stop mechanism (trend markets only).
- **OBV/AD**: Volume confirmation (when reliable volume data available).
- **CCI**: Cyclical markets, futures-specific design.
- **LinearReg Slope**: Quantitative trend strength measurement.
- **KAMA**: Adaptive smoothing when market regime changes frequently.

### Usually Skip
- Cycle indicators (HT_*): Too noisy for practical use.
- Most candlestick patterns: Too many false signals on intraday data.
- Math transforms (SIN, COS, etc.): Building blocks, not trading signals.
- AROONOSC, DX, BOP: Redundant with better alternatives above.
