"""
EMA超级云带策略 (EMA Super Cloud Strategy)
==========================================
多条EMA形成云带，云带宽度和方向判断趋势。

来源: TradingView "EMA Super Cloud"

核心逻辑:
  1. EMA(9/21/50/100/200)组成云带
  2. 云带宽度反映趋势强度
  3. EMA排列顺序判断方向
  4. 价格回到云带边缘=回调入场

技术指标: EMA(9/21/50/100/200), Cloud Width
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EmaSuperCloudStrategy(BaseStrategy):
    """EMA超级云带策略 — 多EMA云带排列 + 回调入场"""

    strategy_description = "EMA云带: 5条EMA排列 + 云带宽度趋势"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "ema_periods": {"type": "str", "default": "9,21,50,100,200", "label": "EMA周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        periods = params.get('ema_periods', '9,21,50,100,200')
        self.ema_periods = [int(p.strip()) for p in str(periods).split(',')]
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'ema_periods': '9,21,50,100,200',
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

        print(f"EmaSuperCloud: 生成 {len(self.signals)} 个信号")
        return self.signals

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

    def _evaluate(self, data):
        max_period = max(self.ema_periods)
        if len(data) < max_period + 20:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # Calculate all EMAs
        emas = {}
        for p in self.ema_periods:
            emas[p] = self._calc_ema(close, p)

        # 1. Perfect alignment (all EMAs in order)
        ema_vals = [emas[p][-1] for p in self.ema_periods]
        bullish_aligned = all(ema_vals[i] > ema_vals[i + 1] for i in range(len(ema_vals) - 1))
        bearish_aligned = all(ema_vals[i] < ema_vals[i + 1] for i in range(len(ema_vals) - 1))

        if bullish_aligned:
            score += 5
        elif bearish_aligned:
            score -= 5

        # 2. Cloud width (trend strength)
        cloud_width = (ema_vals[0] - ema_vals[-1]) / ema_vals[-1] * 100
        if abs(cloud_width) > 5:
            if cloud_width > 0:
                score += 2
            else:
                score -= 2

        # 3. Price position relative to cloud
        cloud_top = max(ema_vals)
        cloud_bottom = min(ema_vals)
        current_price = close[-1]

        if current_price > cloud_top:
            score += 2  # Above cloud = strong bullish
        elif current_price < cloud_bottom:
            score -= 2
        elif cloud_bottom <= current_price <= cloud_top:
            # Inside cloud - look for pullback entry
            fast_ema = emas[self.ema_periods[0]][-1]
            if current_price > fast_ema and score > 0:
                score += 1
            elif current_price < fast_ema and score < 0:
                score += 1

        # 4. Cloud direction change (EMA cross)
        if n >= 2:
            prev_vals = [emas[p][-2] for p in self.ema_periods]
            was_bullish = all(prev_vals[i] > prev_vals[i + 1] for i in range(len(prev_vals) - 1))
            was_bearish = all(prev_vals[i] < prev_vals[i + 1] for i in range(len(prev_vals) - 1))
            if not was_bullish and bullish_aligned:
                score += 3  # Fresh alignment
            elif not was_bearish and bearish_aligned:
                score -= 3

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
        if len(data) < max(self.ema_periods) + 20:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (ema_cloud)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
