"""
Seykota趋势系统 (Seykota Trend System)
======================================
经典Ed Seykota风格趋势跟踪: 通道突破 + 金字塔加仓 + 追踪止损。

来源: TradingView "Seykota Trend System"

核心逻辑:
  1. 价格突破N日最高价做多
  2. 每上涨1ATR加仓一次(最多4次)
  3. 追踪止损 = 低点 + 2*ATR
  4. 反向突破N日最低价做空

技术指标: Donchian Channel, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class SeykotaTrendStrategy(BaseStrategy):
    """Seykota趋势系统 — 通道突破 + ATR金字塔 + 追踪止损"""

    strategy_description = "Seykota: Donchian突破 + ATR金字塔加仓 + 追踪止损"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "breakout_period": {"type": "int", "default": 20, "label": "突破周期"},
        "atr_period": {"type": "int", "default": 20, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 2.0, "label": "止损ATR倍数"},
        "pyramid_atr": {"type": "float", "default": 1.0, "label": "加仓ATR间隔"},
        "max_pyramid": {"type": "int", "default": 4, "label": "最大加仓"},
        "hold_min": {"type": "int", "default": 5, "label": "最少持仓天数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.breakout_period = params.get('breakout_period', 20)
        self.atr_period = params.get('atr_period', 20)
        self.atr_mult = params.get('atr_mult', 2.0)
        self.pyramid_atr = params.get('pyramid_atr', 1.0)
        self.max_pyramid = params.get('max_pyramid', 4)
        self.hold_min = params.get('hold_min', 5)

    def get_default_params(self):
        return {
            'breakout_period': 20, 'atr_period': 20, 'atr_mult': 2.0,
            'pyramid_atr': 1.0, 'max_pyramid': 4, 'hold_min': 5,
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
        entry_price = 0.0
        pyramid_count = 0
        last_pyramid_price = 0.0

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_sym = None
                best_dir = 0
                best_strength = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._check_breakout(hist)
                    if result is None:
                        continue
                    direction, strength = result
                    if strength > best_strength:
                        best_strength = strength
                        best_sym = sym
                        best_dir = direction

                if best_sym and best_dir != 0:
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
                    entry_price = 0.0
                    pyramid_count = 1
                    last_pyramid_price = 0.0

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

                if entry_price == 0.0:
                    entry_price = current_price

                hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                should_exit = False
                should_add = False

                if days_held >= self.hold_min:
                    atr_val = self._calc_atr(hist)
                    # Trailing stop
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.atr_mult * atr_val:
                                should_exit = True

                    # Channel exit
                    if not should_exit and len(hist) >= self.breakout_period:
                        low_vals = hist['low'].values
                        high_vals = hist['high'].values
                        if position_dir == 1:
                            channel_low = np.min(low_vals[-self.breakout_period:])
                            if current_price < channel_low:
                                should_exit = True
                        else:
                            channel_high = np.max(high_vals[-self.breakout_period:])
                            if current_price > channel_high:
                                should_exit = True

                    if days_held >= 120:
                        should_exit = True

                # Pyramid add
                if not should_exit and pyramid_count < self.max_pyramid:
                    atr_val = self._calc_atr(hist)
                    if atr_val > 0 and last_pyramid_price > 0:
                        if position_dir == 1 and current_price > last_pyramid_price + self.pyramid_atr * atr_val:
                            should_add = True
                        elif position_dir == -1 and current_price < last_pyramid_price - self.pyramid_atr * atr_val:
                            should_add = True

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
                    entry_price = 0.0
                    pyramid_count = 0

                elif should_add:
                    if position_dir == 1:
                        self._record_signal(current_time, 'buy', current_holding)
                    else:
                        self._record_signal(current_time, 'sell', current_holding)
                    last_pyramid_price = current_price
                    pyramid_count += 1

        print(f"SeykotaTrend: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _check_breakout(self, data):
        if len(data) < self.breakout_period + 5:
            return None

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        prev_high = np.max(high[-self.breakout_period-1:-1])
        prev_low = np.min(low[-self.breakout_period-1:-1])

        atr = self._calc_atr(data)

        if close[-1] > prev_high:
            strength = (close[-1] - prev_high) / atr if atr > 0 else 0.5
            return 1, strength
        if close[-1] < prev_low:
            strength = (prev_low - close[-1]) / atr if atr > 0 else 0.5
            return -1, strength

        return None

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
        if len(data) < 30:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._check_breakout(data)
        price = float(data['close'].iloc[-1])

        if result is not None:
            direction, strength = result
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"breakout strength={strength:.2f}",
                'price': price,
            }
        return {'action': 'hold', 'reason': 'no breakout', 'price': price}
