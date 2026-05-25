"""
Undercut & Rally (U&R) Pattern Strategy
=========================================
五层决策漏斗策略:
1. 趋势层: EMA stacking (EMA9>EMA21>EMA50 = 多头趋势)
2. 质量层: RS Rating>70 (个股20日涨幅 vs 市场均值) + Kaufman ER>0.2
3. 触发层: 价格跌破近期swing low (Undercut) 然后反弹
4. 形态层: 反弹K线实体>前阴线50% + 成交量>1.5倍均量(RVol)
5. 交易层: 入场 + ATR止损

Undercut & Rally模式: 价格假突破前低后迅速反弹, 这是机构收集筹码的信号。

买入: 5层全部通过
卖出: 趋势破坏(EMA stacking反转) 或 ATR trailing stop

知识来源:
- Oliver Kell / Mark Minervini U&R pattern
- EMA stacking (trend following)
- RS Rating (relative strength)
- Kaufman Efficiency Ratio (trend quality)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class UndercutRallyStrategy(BaseStrategy):
    """Undercut & Rally — 五层决策漏斗: 趋势+质量+触发+形态+交易"""

    strategy_description = "Undercut&Rally模式: EMA stacking趋势 + RS评分 + Undercut触发 + 反弹确认 + ATR止损"
    strategy_category = "price_action"
    strategy_params_schema = {
        "ema_fast": {"type": "int", "default": 9, "label": "快线EMA"},
        "ema_mid": {"type": "int", "default": 21, "label": "中线EMA"},
        "ema_slow": {"type": "int", "default": 50, "label": "慢线EMA"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "rvol_threshold": {"type": "float", "default": 1.5, "label": "相对成交量阈值"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ema_fast = params.get('ema_fast', 9)
        self.ema_mid = params.get('ema_mid', 21)
        self.ema_slow = params.get('ema_slow', 50)
        self.atr_period = params.get('atr_period', 14)
        self.rvol_threshold = params.get('rvol_threshold', 1.5)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'ema_fast': 9,
            'ema_mid': 21,
            'ema_slow': 50,
            'atr_period': 14,
            'rvol_threshold': 1.5,
            'trail_atr_mult': 2.5,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        high_water = 0.0

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_score = -1
                best_sym = None

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    layers = self._evaluate(hist)
                    if layers is None:
                        continue
                    score = sum(layers.values())
                    if score > best_score:
                        best_score = score
                        best_sym = sym

                # All 5 layers must pass (each layer contributes 1 point)
                if best_sym and best_score >= 5:
                    price = float(current_bars[current_bars['symbol'] == best_sym].iloc[0]['close'])
                    self._record_signal(current_time, 'buy', best_sym, price)
                    current_holding = best_sym
                    high_water = price

            else:
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # Update high water mark
                high_water = max(high_water, current_price)

                # ATR trailing stop
                hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                atr_val = self._calc_atr(hist)
                stop_hit = False

                if atr_val > 0 and high_water > 0:
                    if current_price < high_water - self.trail_atr_mult * atr_val:
                        stop_hit = True

                # Trend destruction exit (EMA stacking reversed)
                trend_exit = False
                if len(hist) >= self.ema_slow + 5:
                    close_arr = hist['close'].values
                    ema_f = self._calc_ema(close_arr, self.ema_fast)
                    ema_m = self._calc_ema(close_arr, self.ema_mid)
                    ema_s = self._calc_ema(close_arr, self.ema_slow)
                    n = len(close_arr)
                    # EMA stacking reversed: fast < mid < slow
                    if ema_f[-1] < ema_m[-1] and ema_m[-1] < ema_s[-1]:
                        trend_exit = True

                if stop_hit or trend_exit:
                    self._record_signal(current_time, 'sell', current_holding, current_price)
                    current_holding = None
                    high_water = 0.0

        print(f"UndercutRally: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """五层决策漏斗评估, 每层通过返回1, 总分5=全部通过"""
        min_len = max(self.ema_slow + 5, self.atr_period + 10, 30)
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        open_arr = data['open'].values
        n = len(close)

        # Pre-calculate indicators
        ema_f = self._calc_ema(close, self.ema_fast)
        ema_m = self._calc_ema(close, self.ema_mid)
        ema_s = self._calc_ema(close, self.ema_slow)
        atr = self._calc_atr_from_arrays(high, low, close, n)

        layers = {}

        # === Layer 1: Trend — EMA Stacking (EMA9 > EMA21 > EMA50) ===
        if ema_f[-1] > ema_m[-1] and ema_m[-1] > ema_s[-1]:
            layers['trend'] = 1
        else:
            layers['trend'] = 0
            return layers  # Early exit if no trend

        # === Layer 2: Quality — RS Rating > 70 + Kaufman ER > 0.2 ===
        # RS Rating: 20-day return percentile (simplified: vs 0 as baseline)
        if n >= 20:
            ret_20 = (close[-1] - close[-20]) / close[-20] * 100 if close[-20] > 0 else 0
            # Map to 0-100 range: ret_20 > 10% maps to RS > 70
            rs_rating = min(100, max(0, 50 + ret_20 * 5))
        else:
            rs_rating = 50

        er = self._efficiency_ratio(close)
        quality_pass = rs_rating > 70 and er > 0.2
        layers['quality'] = 1 if quality_pass else 0
        if not quality_pass:
            return layers

        # === Layer 3: Trigger — Undercut (price breaks below recent swing low then bounces) ===
        swing_low = self._find_recent_swing_low(low, n, lookback=20)
        undercut_triggered = False
        if swing_low is not None and n >= 3:
            # Price went below swing low (within last few bars) but closed above
            # Check last 2 bars: bar[-2] low < swing_low, and close[-1] > swing_low
            if low[-2] < swing_low and close[-1] > swing_low:
                undercut_triggered = True
            # Also check: bar[-1] wick below swing low but close above
            elif low[-1] < swing_low and close[-1] > swing_low and close[-1] > open_arr[-1]:
                undercut_triggered = True

        layers['trigger'] = 1 if undercut_triggered else 0
        if not undercut_triggered:
            return layers

        # === Layer 4: Pattern — Rally candle body > 50% of prior bearish candle + RVol > 1.5 ===
        # Prior bearish candle
        prev_body = open_arr[-2] - close[-2]  # Positive if bearish
        curr_body = close[-1] - open_arr[-1]  # Positive if bullish

        pattern_pass = False
        if prev_body > 0 and curr_body > 0:
            # Rally candle body > 50% of prior bearish body
            body_ratio = curr_body / prev_body if prev_body > 0 else 0
            if body_ratio >= 0.5:
                # Volume confirmation
                vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
                rvol_ok = True
                if vol_col and n >= 20:
                    vol = data[vol_col].values
                    vol_ma = np.mean(vol[-20:])
                    if vol_ma > 0:
                        rvol = vol[-1] / vol_ma
                        rvol_ok = rvol >= self.rvol_threshold

                if rvol_ok:
                    pattern_pass = True

        layers['pattern'] = 1 if pattern_pass else 0
        if not pattern_pass:
            return layers

        # === Layer 5: Trade — Entry confirmation (price > EMA_fast, ATR valid) ===
        trade_pass = close[-1] > ema_f[-1] and atr > 0
        layers['trade'] = 1 if trade_pass else 0

        return layers

    def _find_recent_swing_low(self, low, n, lookback=20):
        """Find the most recent swing low (local minimum in the lookback window)."""
        if n < lookback + 5:
            return None

        # Look at bars from n-lookback-5 to n-5 (exclude most recent bars)
        start = max(0, n - lookback - 10)
        end = n - 3  # Exclude last 3 bars

        if end <= start + 2:
            return None

        # Find local minimum
        min_idx = start
        for i in range(start + 1, end):
            if low[i] < low[min_idx]:
                min_idx = i

        # Verify it's a swing low (higher lows on both sides)
        is_swing = True
        window = 2
        for j in range(max(start, min_idx - window), min_idx):
            if low[j] < low[min_idx]:
                is_swing = False
                break
        for j in range(min_idx + 1, min(end, min_idx + window + 1)):
            if low[j] < low[min_idx]:
                is_swing = False
                break

        if is_swing:
            return low[min_idx]
        # Fallback: just return the minimum
        return low[min_idx]

    def _calc_ema(self, values, period):
        """Calculate EMA series."""
        values = np.asarray(values, dtype=float)
        n = len(values)
        result = np.empty(n)
        result[0] = values[0]
        k = 2.0 / (period + 1)
        for i in range(1, n):
            if i < period:
                result[i] = np.mean(values[:i + 1])
            else:
                result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    def _calc_atr(self, data):
        """Calculate current ATR from DataFrame."""
        if len(data) < self.atr_period + 1:
            return 0
        return self._calc_atr_from_arrays(
            data['high'].values, data['low'].values,
            data['close'].values, len(data)
        )

    def _calc_atr_from_arrays(self, high, low, close, n):
        """Calculate ATR from numpy arrays."""
        if n < self.atr_period + 1:
            return 0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        return float(np.mean(tr[-self.atr_period:]))

    def _efficiency_ratio(self, close):
        """Kaufman Efficiency Ratio"""
        window = 20
        if len(close) < window:
            return 0
        recent = close[-window:]
        net = abs(recent[-1] - recent[0])
        total = np.sum(np.abs(np.diff(recent)))
        return net / total if total > 0 else 0

    def screen(self):
        data = self.data.copy()
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        layers = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if layers is None:
            return {'action': 'hold', 'reason': '数据不足评估', 'price': price}

        score = sum(layers.values())
        passed = [k for k, v in layers.items() if v == 1]
        failed = [k for k, v in layers.items() if v == 0]

        if score >= 5:
            return {
                'action': 'buy',
                'reason': f"5/5层全通过 (U&R)",
                'price': price,
            }
        return {
            'action': 'hold',
            'reason': f'{score}/5层 通过={passed} 失败={failed}',
            'price': price,
        }
