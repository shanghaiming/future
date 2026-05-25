"""
DEMA+SuperTrend策略 (DEMA + SuperTrend Strategy)
================================================
双EMA(DEMA)趋势系统 + SuperTrend过滤。

来源: TradingView "MTF DEMA Trend System + SuperTrend"

核心逻辑:
  1. DEMA(快/慢)交叉判断趋势方向
  2. SuperTrend确认趋势持续
  3. 两者同向才入场
  4. ATR追踪止损

技术指标: DEMA(10,20), SuperTrend(10,3)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class DemaSupertrendStrategy(BaseStrategy):
    """DEMA+SuperTrend策略 — DEMA交叉 + SuperTrend确认"""

    strategy_description = "DEMA+SuperTrend: 双EMA交叉 + SuperTrend过滤"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "dema_fast": {"type": "int", "default": 10, "label": "快DEMA"},
        "dema_slow": {"type": "int", "default": 20, "label": "慢DEMA"},
        "st_period": {"type": "int", "default": 10, "label": "SuperTrend周期"},
        "st_mult": {"type": "float", "default": 3.0, "label": "SuperTrend倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.dema_fast = params.get('dema_fast', 10)
        self.dema_slow = params.get('dema_slow', 20)
        self.st_period = params.get('st_period', 10)
        self.st_mult = params.get('st_mult', 3.0)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'dema_fast': 10, 'dema_slow': 20, 'st_period': 10, 'st_mult': 3.0,
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

        print(f"DemaSupertrend: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.dema_slow, self.st_period) + 20
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        # 1. DEMA (Double EMA = 2*EMA - EMA(EMA))
        dema_f = self._calc_dema(close, self.dema_fast)
        dema_s = self._calc_dema(close, self.dema_slow)

        if dema_f[-1] > dema_s[-1]:
            score += 3
        elif dema_f[-1] < dema_s[-1]:
            score -= 3

        # DEMA cross
        if n >= 2:
            if dema_f[-2] <= dema_s[-2] and dema_f[-1] > dema_s[-1]:
                score += 3  # Golden cross
            elif dema_f[-2] >= dema_s[-2] and dema_f[-1] < dema_s[-1]:
                score -= 3  # Death cross

        # 2. SuperTrend direction
        atr = self._calc_atr(data)
        if atr > 0:
            hl2 = (high[-1] + low[-1]) / 2.0
            upper = hl2 + self.st_mult * atr
            lower = hl2 - self.st_mult * atr

            if close[-1] > upper:
                score += 3  # SuperTrend bullish
            elif close[-1] < lower:
                score -= 3  # SuperTrend bearish
            elif close[-1] > hl2:
                score += 1
            else:
                score -= 1

        direction = 1 if score > 0 else -1
        return score, direction, atr

    def _calc_dema(self, close, period):
        """Calculate DEMA: 2*EMA - EMA(EMA)"""
        ema1 = self._calc_ema(close, period)
        ema2 = self._calc_ema(ema1, period)
        return 2 * ema1 - ema2

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
                'reason': f"score={score} (dema+st)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
