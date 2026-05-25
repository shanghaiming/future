"""
趋势线突破策略 (Clean Trendline Breakout Strategy)
===================================================
自动识别趋势线，价格突破趋势线并伴随成交量确认时入场。

来源: TradingView "Clean Trendline Breakout Detector"

核心逻辑:
  1. 摆动点连接形成趋势线
  2. 价格收盘突破趋势线
  3. 成交量放大确认突破有效性
  4. ATR追踪止损保护利润

技术指标: Swing Pivots, Linear Regression, Volume, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class TrendlineBreakoutStrategy(BaseStrategy):
    """趋势线突破策略 — 自动趋势线 + 量能确认突破"""

    strategy_description = "趋势线突破: 摆动点趋势线 + 量能确认突破"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "vol_mult": {"type": "float", "default": 1.5, "label": "突破量能倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.pivot_len = params.get('pivot_len', 5)
        self.vol_mult = params.get('vol_mult', 1.5)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'pivot_len': 5, 'vol_mult': 1.5,
            'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0
        low_water = float('inf')

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    score, direction, _ = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= 3:
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0
                    low_water = float('inf')

            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) if low_water < float('inf') else current_price

                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                    atr_val = self._calc_atr(hist)
                    should_exit = False

                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    if days_held >= 60:
                        should_exit = True

                    if not should_exit:
                        result = self._evaluate(hist)
                        if result is not None:
                            score, direction, _ = result
                            if position_dir == 1 and direction == -1 and score < -3:
                                should_exit = True
                            elif position_dir == -1 and direction == 1 and score > 3:
                                should_exit = True

                    if should_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding)
                        else:
                            self._record_signal(current_time, 'buy', current_holding)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0
                        low_water = float('inf')

        print(f"TrendlineBreakout: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _find_pivots(self, high, low, n):
        """找摆动高低点"""
        piv_len = self.pivot_len
        if n < piv_len * 2 + 1:
            return [], []

        swing_highs = []
        swing_lows = []
        for i in range(piv_len, n - piv_len):
            is_high = True
            is_low = True
            for j in range(1, piv_len + 1):
                if high[i] < high[i - j] or high[i] < high[i + j]:
                    is_high = False
                if low[i] > low[i - j] or low[i] > low[i + j]:
                    is_low = False
            if is_high:
                swing_highs.append((i, high[i]))
            if is_low:
                swing_lows.append((i, low[i]))
        return swing_highs, swing_lows

    def _calc_trendline(self, pivots, n):
        """从摆动点计算趋势线斜率和截距"""
        if len(pivots) < 2:
            return None
        recent = pivots[-2:]
        x1, y1 = recent[0]
        x2, y2 = recent[1]
        if x2 == x1:
            return None
        slope = (y2 - y1) / (x2 - x1)
        # Extrapolate to current bar
        intercept = y1 - slope * x1
        return slope, intercept

    def _evaluate(self, data):
        if len(data) < self.pivot_len * 2 + 20:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        swing_highs, swing_lows = self._find_pivots(high, low, n)

        # Resistance trendline from swing highs (downtrend line)
        tl_high = self._calc_trendline(swing_highs, n)
        # Support trendline from swing lows (uptrend line)
        tl_low = self._calc_trendline(swing_lows, n)

        if tl_high is not None:
            slope_h, intercept_h = tl_high
            trendline_val = slope_h * (n - 1) + intercept_h
            # Upside breakout of resistance trendline
            if close[-1] > trendline_val and slope_h <= 0:
                score += 5
            elif close[-1] > trendline_val:
                score += 2

        if tl_low is not None:
            slope_l, intercept_l = tl_low
            trendline_val = slope_l * (n - 1) + intercept_l
            # Downside breakout of support trendline
            if close[-1] < trendline_val and slope_l >= 0:
                score -= 5
            elif close[-1] < trendline_val:
                score -= 2

        # Volume confirmation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * self.vol_mult:
                if score > 0:
                    score += 2
                elif score < 0:
                    score -= 2

        # Price momentum confirmation
        if n >= 10:
            ret = (close[-1] / close[-10] - 1) * 100
            if score > 0 and ret > 1:
                score += 1
            elif score < 0 and ret < -1:
                score += 1

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

    def screen(self):
        data = self.data.copy()
        if len(data) < 40:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (trendline_breakout)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
