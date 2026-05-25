"""
高级RSI动量策略 (Advanced RSI Momentum Strategy)
================================================
三层RSI分析: RSI(14) + EMA(9) of RSI + WMA(45) of RSI + 动量色带。

来源: TradingView "Advanced RSI Momentum with Trend Dot [JoeyWave]"

核心逻辑:
  1. RSI(14)基础值
  2. EMA(9) of RSI 短期平滑
  3. WMA(45) of RSI 长期趋势
  4. 三层排列: RSI > EMA > WMA = 强多头
  5. RSI从超卖区回升 + EMA上穿WMA = 买入

技术指标: RSI, EMA of RSI, WMA of RSI
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class AdvancedRsiMomentumStrategy(BaseStrategy):
    """高级RSI动量策略 — 三层RSI分析 + 动量色带"""

    strategy_description = "高级RSI: 三层RSI(原始+EMA+WMA) + 动量方向"
    strategy_category = "momentum"
    strategy_params_schema = {
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "ema_period": {"type": "int", "default": 9, "label": "RSI的EMA"},
        "wma_period": {"type": "int", "default": 45, "label": "RSI的WMA"},
        "rsi_ob": {"type": "int", "default": 70, "label": "超买"},
        "rsi_os": {"type": "int", "default": 30, "label": "超卖"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.rsi_period = params.get('rsi_period', 14)
        self.ema_period = params.get('ema_period', 9)
        self.wma_period = params.get('wma_period', 45)
        self.rsi_ob = params.get('rsi_ob', 70)
        self.rsi_os = params.get('rsi_os', 30)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'rsi_period': 14, 'ema_period': 9, 'wma_period': 45,
            'rsi_ob': 70, 'rsi_os': 30,
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

        print(f"AdvancedRsiMomentum: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = self.wma_period + self.rsi_period + 20
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # 1. Calculate RSI series
        rsi_series = self._calc_rsi_series(close)
        if rsi_series is None:
            return None

        rsi = rsi_series[-1]

        # 2. EMA of RSI
        ema_of_rsi = self._calc_ema(rsi_series, self.ema_period)

        # 3. WMA of RSI
        wma_of_rsi = self._calc_wma(rsi_series, self.wma_period)

        # 4. Triple alignment
        if rsi > ema_of_rsi[-1] > wma_of_rsi[-1]:
            score += 5  # Perfect bullish alignment
        elif rsi < ema_of_rsi[-1] < wma_of_rsi[-1]:
            score -= 5  # Perfect bearish alignment
        elif rsi > ema_of_rsi[-1]:
            score += 2
        elif rsi < ema_of_rsi[-1]:
            score -= 2

        # 5. EMA/WMA crossover
        if len(ema_of_rsi) >= 2 and len(wma_of_rsi) >= 2:
            if ema_of_rsi[-2] <= wma_of_rsi[-2] and ema_of_rsi[-1] > wma_of_rsi[-1]:
                score += 3  # Golden cross of RSI indicators
            elif ema_of_rsi[-2] >= wma_of_rsi[-2] and ema_of_rsi[-1] < wma_of_rsi[-1]:
                score -= 3  # Death cross

        # 6. RSI zone
        if rsi < self.rsi_os:
            score += 2
        elif rsi > self.rsi_ob:
            score -= 2

        # 7. RSI momentum (acceleration)
        if len(rsi_series) >= 3:
            rsi_slope = rsi_series[-1] - rsi_series[-3]
            if rsi_slope > 5:
                score += 1
            elif rsi_slope < -5:
                score -= 1

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_rsi_series(self, close):
        n = len(close)
        if n < self.rsi_period + 1:
            return None
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi_arr = np.full(n, 50.0)
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])
        if avg_loss == 0:
            rsi_arr[self.rsi_period] = 100.0
        else:
            rsi_arr[self.rsi_period] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)

        for i in range(self.rsi_period, len(gains)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i]) / self.rsi_period
            if avg_loss == 0:
                rsi_arr[i + 1] = 100.0
            else:
                rsi_arr[i + 1] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)
        return rsi_arr

    def _calc_ema(self, values, period):
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

    def _calc_wma(self, values, period):
        values = np.asarray(values, dtype=float)
        n = len(values)
        result = np.empty(n)
        weights = np.arange(1, period + 1, dtype=float)
        weight_sum = np.sum(weights)
        for i in range(n):
            if i < period - 1:
                w = np.arange(1, i + 2, dtype=float)
                result[i] = np.sum(values[:i+1] * w) / np.sum(w)
            else:
                result[i] = np.sum(values[i-period+1:i+1] * weights) / weight_sum
        return result

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
        if len(data) < 80:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (adv_rsi)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
