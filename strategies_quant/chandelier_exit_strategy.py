"""
Chandelier Exit策略 (Chandelier Exit PRO Strategy)
==================================================
基于ATR的追踪止损策略，结合ADX过滤避免震荡市假信号。

来源: TradingView "Chandelier Exit PRO (Fixed)"

核心逻辑:
  1. 计算ATR追踪止损线
  2. 多头: 止损 = 最高价 - N*ATR
  3. 空头: 止损 = 最低价 + N*ATR
  4. ADX过滤: ADX<20时不交易(震荡市)
  5. 止损线交叉作为入场/出场信号

技术指标: ATR, ADX, Chandelier Exit
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class ChandelierExitStrategy(BaseStrategy):
    """Chandelier Exit策略 — ATR追踪止损 + ADX过滤"""

    strategy_description = "ChandelierExit: ATR追踪止损 + ADX趋势过滤"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "atr_period": {"type": "int", "default": 22, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 3.0, "label": "ATR倍数"},
        "adx_period": {"type": "int", "default": 14, "label": "ADX周期"},
        "adx_threshold": {"type": "int", "default": 20, "label": "ADX阈值"},
        "lookback": {"type": "int", "default": 22, "label": "最高/最低回望"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "出场止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.atr_period = params.get('atr_period', 22)
        self.atr_mult = params.get('atr_mult', 3.0)
        self.adx_period = params.get('adx_period', 14)
        self.adx_threshold = params.get('adx_threshold', 20)
        self.lookback = params.get('lookback', 22)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'atr_period': 22, 'atr_mult': 3.0, 'adx_period': 14,
            'adx_threshold': 20, 'lookback': 22, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"ChandelierExit: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.lookback, self.atr_period, self.adx_period * 2) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        # 1. ADX filter
        adx = self._calc_adx(data)
        if adx < self.adx_threshold:
            return None  # Ranging market, don't trade

        score += 2  # Trend confirmed by ADX

        # 2. Chandelier lines
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        highest = np.max(high[-self.lookback:])
        lowest = np.min(low[-self.lookback:])

        long_stop = highest - self.atr_mult * atr
        short_stop = lowest + self.atr_mult * atr

        # 3. Signal from chandelier
        if close[-1] > long_stop:
            score += 3
        elif close[-1] < short_stop:
            score -= 3

        # 4. Cross signal
        if n >= 2:
            prev_close = close[-2]
            if prev_close <= long_stop and close[-1] > long_stop:
                score += 3  # Crossed above long stop
            elif prev_close >= short_stop and close[-1] < short_stop:
                score -= 3  # Crossed below short stop

        direction = 1 if score > 0 else -1
        return score, direction, atr

    def _calc_adx(self, data):
        """Calculate ADX"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(high)
        if n < self.adx_period * 2:
            return 0

        # True Range
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        period = self.adx_period
        if len(tr) < period * 2:
            return 0

        atr = np.mean(tr[-period*2:-period])
        if atr == 0:
            return 0

        plus_di = np.mean(plus_dm[-period*2:-period]) / atr * 100
        minus_di = np.mean(minus_dm[-period*2:-period]) / atr * 100

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0

        dx = np.abs(plus_di - minus_di) / di_sum * 100
        return dx

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
        if len(data) < 60:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': 'ADX过低或数据不足', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (chandelier)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
