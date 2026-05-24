"""
HA-EMA交叉策略 (Heikin Ashi EMA Cross Strategy)
================================================
使用Heikin Ashi蜡烛图的EMA交叉系统，减少噪音假信号。

来源: TradingView "Volatility Vault - EMA Tap Scanner"

核心逻辑:
  1. 将K线转为Heikin Ashi蜡烛
  2. HA收盘价上穿EMA(9)做多
  3. HA收盘价下穿EMA(9)做空
  4. EMA(21)趋势过滤
  5. HA实体颜色确认方向

技术指标: Heikin Ashi, EMA(9,21)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class HaEmaCrossStrategy(BaseStrategy):
    """HA-EMA交叉策略 — Heikin Ashi + EMA交叉"""

    strategy_description = "HA-EMA: Heikin Ashi蜡烛 + EMA交叉系统"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "ema_fast": {"type": "int", "default": 9, "label": "快EMA"},
        "ema_slow": {"type": "int", "default": 21, "label": "慢EMA"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ema_fast = params.get('ema_fast', 9)
        self.ema_slow = params.get('ema_slow', 21)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'ema_fast': 9, 'ema_slow': 21,
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

        print(f"HaEmaCross: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.ema_slow, self.ema_fast) * 2 + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        open_ = data['open'].values
        n = len(close)
        score = 0

        # Calculate HA candles
        ha_close = np.zeros(n)
        ha_open = np.zeros(n)
        ha_close[0] = (open_[0] + high[0] + low[0] + close[0]) / 4.0
        ha_open[0] = open_[0]
        for i in range(1, n):
            ha_close[i] = (open_[i] + high[i] + low[i] + close[i]) / 4.0
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0

        # EMA on HA close
        ema_f = self._calc_ema(ha_close, self.ema_fast)
        ema_s = self._calc_ema(ha_close, self.ema_slow)

        # HA candle color
        is_green = ha_close[-1] > ha_open[-1]
        is_red = ha_close[-1] < ha_open[-1]

        # 1. EMA trend
        if ema_f[-1] > ema_s[-1]:
            score += 3
        elif ema_f[-1] < ema_s[-1]:
            score -= 3

        # 2. EMA cross
        if n >= 2:
            if ema_f[-2] <= ema_s[-2] and ema_f[-1] > ema_s[-1]:
                score += 3  # Golden cross
            elif ema_f[-2] >= ema_s[-2] and ema_f[-1] < ema_s[-1]:
                score -= 3  # Death cross

        # 3. HA candle color confirmation
        if is_green and score > 0:
            score += 2
        elif is_red and score < 0:
            score += 2  # Stronger signal
        elif is_green and score < 0:
            score += 1  # Weak counter-signal
        elif is_red and score > 0:
            score += 1

        # 4. Price vs fast EMA
        if ha_close[-1] > ema_f[-1]:
            score += 1
        elif ha_close[-1] < ema_f[-1]:
            score -= 1

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
                'reason': f"score={score} (ha_ema)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
