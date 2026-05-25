"""
WaveTrend振荡器策略 (WaveTrend Oscillator Strategy)
===================================================
WaveTrend指标结合主导周期自调节。

来源: TradingView "Adaptive Pivot-Length WaveTrend"

核心逻辑:
  1. WaveTrend = EMA(EMA(tp, n1) - EMA(tp, n2)) / EMA(ATR, n1)
  2. WT > 60 = 超买
  3. WT < -60 = 超卖
  4. WT从极端区交叉信号线 = 反转信号

技术指标: WaveTrend, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class WaveTrendStrategy(BaseStrategy):
    """WaveTrend振荡器策略 — WT超买超卖 + 交叉信号"""

    strategy_description = "WaveTrend: WT振荡器超买超卖 + 交叉反转"
    strategy_category = "momentum"
    strategy_params_schema = {
        "wt_n1": {"type": "int", "default": 10, "label": "WT通道周期"},
        "wt_n2": {"type": "int", "default": 21, "label": "WT平均周期"},
        "wt_ob": {"type": "int", "default": 60, "label": "WT超买"},
        "wt_os": {"type": "int", "default": -60, "label": "WT超卖"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.wt_n1 = params.get('wt_n1', 10)
        self.wt_n2 = params.get('wt_n2', 21)
        self.wt_ob = params.get('wt_ob', 60)
        self.wt_os = params.get('wt_os', -60)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'wt_n1': 10, 'wt_n2': 21, 'wt_ob': 60, 'wt_os': -60,
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

        print(f"WaveTrend: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.wt_n1, self.wt_n2) * 2 + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        # Typical price
        tp = (high + low + close) / 3.0

        # WaveTrend calculation
        esa = self._calc_ema(tp, self.wt_n1)  # EMA of TP
        d = np.abs(tp - esa)
        ci = self._calc_ema(d, self.wt_n1)  # EMA of deviation

        # Avoid division by zero
        tci = np.where(ci > 0, (tp - esa) / (ci * 0.015), 0)
        wt = self._calc_ema(tci, self.wt_n2)

        # Signal line (SMA of WT)
        wt_signal = self._calc_sma(wt, 4)

        current_wt = wt[-1]
        current_signal = wt_signal[-1]

        # WT zones
        if current_wt > self.wt_ob:
            score -= 4
        elif current_wt > 40:
            score -= 2
        elif current_wt < self.wt_os:
            score += 4
        elif current_wt < -40:
            score += 2

        # WT/Signal cross
        if len(wt) >= 2 and len(wt_signal) >= 2:
            if wt[-2] < wt_signal[-2] and current_wt > current_signal:
                score += 3  # Bullish cross
            elif wt[-2] > wt_signal[-2] and current_wt < current_signal:
                score -= 3  # Bearish cross

        # WT extreme reversal
        if len(wt) >= 3:
            prev_wt = wt[-3]
            if prev_wt < self.wt_os and current_wt > self.wt_os:
                score += 2  # Recovered from oversold
            elif prev_wt > self.wt_ob and current_wt < self.wt_ob:
                score -= 2

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

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

    def _calc_sma(self, values, period):
        n = len(values)
        result = np.empty(n)
        for i in range(n):
            if i < period - 1:
                result[i] = np.mean(values[:i + 1])
            else:
                result[i] = np.mean(values[i - period + 1:i + 1])
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
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (wavetrend)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
