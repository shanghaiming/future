"""
Fib+ATR Swing Probability Strategy
=====================================
基于Fibonacci回撤 + Swing质量评分 + Sigmoid概率的策略:
1. 检测Swing Pivot点(近期最高/最低)
2. 在pivot之间画Fibonacci回撤位(0.236, 0.382, 0.5, 0.618, 0.786)
3. Swing Quality Score: 价格运动幅度/ATR + 回撤深度 + 成交量
4. Sigmoid概率: prob = 1/(1+exp(-k*(score-threshold)))
5. SuperTrend方向过滤: 只在SuperTrend方向上做交易

买入: SuperTrend多头 + 价格回撤到Fib 0.382-0.618区域 + 概率>0.6
卖出: SuperTrend翻空 或 ATR trailing stop

知识来源:
- Fibonacci Retracement (经典技术分析)
- SuperTrend indicator
- Sigmoid probability mapping
- Kaufman Efficiency Ratio
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class FibSwingProbabilityStrategy(BaseStrategy):
    """Fib+ATR Swing Probability — Fibonacci回撤+Swing质量+Sigmoid概率策略"""

    strategy_description = "Fibonacci回撤 + Swing质量评分 + Sigmoid概率 + SuperTrend方向过滤 + ATR止损"
    strategy_category = "swing_trading"
    strategy_params_schema = {
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 3.0, "label": "SuperTrend ATR倍数"},
        "swing_length": {"type": "int", "default": 10, "label": "Swing检测长度"},
        "prob_threshold": {"type": "float", "default": 0.6, "label": "概率阈值"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "止损ATR倍数"},
    }

    FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

    def __init__(self, data, params):
        super().__init__(data, params)
        self.atr_period = params.get('atr_period', 14)
        self.atr_mult = params.get('atr_mult', 3.0)
        self.swing_length = params.get('swing_length', 10)
        self.prob_threshold = params.get('prob_threshold', 0.6)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'atr_period': 14,
            'atr_mult': 3.0,
            'swing_length': 10,
            'prob_threshold': 0.6,
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
        position_dir = 0
        high_water = 0.0

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_prob = 0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    prob, direction = result
                    if prob > best_prob:
                        best_prob = prob
                        best_sym = sym
                        best_dir = direction

                if best_sym and best_prob >= self.prob_threshold:
                    price = float(current_bars[current_bars['symbol'] == best_sym].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym, price)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym, price)
                        position_dir = -1
                    current_holding = best_sym
                    high_water = price

            else:
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # Update high/low water mark
                if position_dir == 1:
                    high_water = max(high_water, current_price)
                else:
                    high_water = min(high_water, current_price) if high_water > 0 else current_price

                # ATR trailing stop
                hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                atr_val = self._calc_atr(hist)
                stop_hit = False

                if atr_val > 0 and high_water > 0:
                    if position_dir == 1 and current_price < high_water - self.trail_atr_mult * atr_val:
                        stop_hit = True
                    elif position_dir == -1 and current_price > high_water + self.trail_atr_mult * atr_val:
                        stop_hit = True

                # SuperTrend flip exit
                st_dir = self._calc_supertrend(hist)
                st_exit = False
                if position_dir == 1 and st_dir == -1:
                    st_exit = True
                elif position_dir == -1 and st_dir == 1:
                    st_exit = True

                if stop_hit or st_exit:
                    if position_dir == 1:
                        self._record_signal(current_time, 'sell', current_holding, current_price)
                    else:
                        self._record_signal(current_time, 'buy', current_holding, current_price)
                    current_holding = None
                    position_dir = 0
                    high_water = 0.0

        print(f"FibSwingProbability: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """评估Swing质量 + Fibonacci回撤 + Sigmoid概率"""
        min_len = max(self.atr_period + 2, self.swing_length * 3)
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ATR
        atr = self._calc_atr_from_arrays(high, low, close, n)
        if atr <= 0:
            return None

        # SuperTrend direction
        st_dir = self._calc_supertrend_from_data(high, low, close, n)
        if st_dir == 0:
            return None

        # Detect swing pivots
        swing_highs = []
        swing_lows = []
        sl = self.swing_length

        for i in range(sl, n - sl):
            # Swing high
            is_sh = True
            for j in range(i - sl, i + sl + 1):
                if j != i and high[j] >= high[i]:
                    is_sh = False
                    break
            if is_sh:
                swing_highs.append((i, high[i]))

            # Swing low
            is_sl = True
            for j in range(i - sl, i + sl + 1):
                if j != i and low[j] <= low[i]:
                    is_sl = False
                    break
            if is_sl:
                swing_lows.append((i, low[i]))

        if not swing_highs or not swing_lows:
            return None

        # Find the most recent swing pair
        if st_dir == 1:
            # Bullish: need swing_low -> swing_high -> current pullback
            recent_low = swing_lows[-1]
            # Find a swing high after the most recent swing low
            highs_after = [(i, p) for i, p in swing_highs if i > recent_low[0]]
            if not highs_after:
                return None
            recent_high = highs_after[-1]

            # Fibonacci levels
            swing_range = recent_high[1] - recent_low[1]
            if swing_range <= 0:
                return None

            fib_0382 = recent_high[1] - 0.382 * swing_range
            fib_0618 = recent_high[1] - 0.618 * swing_range
            current_price = close[-1]

            # Check if price is in the golden zone (0.382 - 0.618)
            in_golden_zone = fib_0618 <= current_price <= fib_0382
            if not in_golden_zone:
                return None

            # Swing Quality Score
            score = self._calc_swing_quality(
                recent_low[1], recent_high[1], current_price,
                atr, close, data, n
            )
            # Sigmoid probability
            prob = self._sigmoid_prob(score)

            return prob, 1

        else:
            # Bearish: need swing_high -> swing_low -> current pullback
            recent_high = swing_highs[-1]
            lows_after = [(i, p) for i, p in swing_lows if i > recent_high[0]]
            if not lows_after:
                return None
            recent_low = lows_after[-1]

            swing_range = recent_high[1] - recent_low[1]
            if swing_range <= 0:
                return None

            fib_0382 = recent_low[1] + 0.382 * swing_range
            fib_0618 = recent_low[1] + 0.618 * swing_range
            current_price = close[-1]

            in_golden_zone = fib_0382 <= current_price <= fib_0618
            if not in_golden_zone:
                return None

            score = self._calc_swing_quality(
                recent_high[1], recent_low[1], current_price,
                atr, close, data, n
            )
            prob = self._sigmoid_prob(score)

            return prob, -1

    def _calc_swing_quality(self, start_price, end_price, current_price, atr, close, data, n):
        """计算Swing Quality Score (0-10范围)"""
        score = 0

        # 1. Price movement magnitude / ATR (larger = better, 0-3 points)
        move_pct = abs(end_price - start_price) / atr if atr > 0 else 0
        if move_pct >= 4:
            score += 3
        elif move_pct >= 2:
            score += 2
        elif move_pct >= 1:
            score += 1

        # 2. Retracement depth in golden zone (0.382-0.618 is optimal, 0-3 points)
        swing_range = abs(end_price - start_price)
        if swing_range > 0:
            retrace = abs(current_price - end_price) / swing_range
            if 0.382 <= retrace <= 0.618:
                score += 3  # Perfect golden zone
            elif 0.236 <= retrace <= 0.786:
                score += 1  # Acceptable zone

        # 3. Volume confirmation (0-2 points)
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol_ma > 0:
                rvol = vol[-1] / vol_ma
                if rvol >= 1.5:
                    score += 2
                elif rvol >= 1.0:
                    score += 1

        # 4. Kaufman Efficiency Ratio (0-2 points)
        er = self._efficiency_ratio(close)
        if er > 0.4:
            score += 2
        elif er > 0.2:
            score += 1

        return score

    def _sigmoid_prob(self, score):
        """Sigmoid probability: compress quality score to 0-1 range"""
        k = 1.0  # steepness
        threshold = 5.0  # midpoint
        prob = 1.0 / (1.0 + np.exp(-k * (score - threshold)))
        return prob

    def _efficiency_ratio(self, close):
        """Kaufman Efficiency Ratio"""
        window = 20
        if len(close) < window:
            return 0
        recent = close[-window:]
        net = abs(recent[-1] - recent[0])
        total = np.sum(np.abs(np.diff(recent)))
        return net / total if total > 0 else 0

    def _calc_atr(self, data):
        """Calculate current ATR from DataFrame."""
        if len(data) < self.atr_period + 1:
            return 0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        return self._calc_atr_from_arrays(high, low, close, len(close))

    def _calc_atr_from_arrays(self, high, low, close, n):
        """Calculate ATR from numpy arrays."""
        if n < self.atr_period + 1:
            return 0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        return float(np.mean(tr[-self.atr_period:]))

    def _calc_supertrend(self, data):
        """Calculate SuperTrend direction from DataFrame."""
        if len(data) < self.atr_period + 2:
            return 0
        return self._calc_supertrend_from_data(
            data['high'].values, data['low'].values,
            data['close'].values, len(data)
        )

    def _calc_supertrend_from_data(self, high, low, close, n):
        """Calculate SuperTrend direction from arrays."""
        period = self.atr_period
        mult = self.atr_mult
        if n < period + 2:
            return 0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        atr_arr = np.zeros(n)
        atr_arr[period] = np.mean(tr[:period])
        for i in range(period + 1, n):
            atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i - 1]) / period

        hl2 = (high + low) / 2.0
        upper = hl2 + mult * atr_arr
        lower = hl2 - mult * atr_arr
        direction = 0

        for i in range(period + 1, n):
            if direction == 1 and i > period + 1:
                lower[i] = max(lower[i], lower[i - 1])
            elif direction == -1 and i > period + 1:
                upper[i] = min(upper[i], upper[i - 1])
            if close[i] > upper[i - 1]:
                direction = 1
            elif close[i] < lower[i - 1]:
                direction = -1

        return direction

    def screen(self):
        data = self.data.copy()
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无Fibonacci回撤信号', 'price': price}

        prob, direction = result
        if prob >= self.prob_threshold:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"prob={prob:.2f} (fib_swing)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'prob={prob:.2f}', 'price': price}
