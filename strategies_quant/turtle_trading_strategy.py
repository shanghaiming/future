"""
海龟交易策略 (Turtle Trading Strategy)
=======================================
经典海龟交易法则的Python实现。

来源: TradingView "Turtle Trading Strategy with ATR Stop + Pyramiding"

核心逻辑:
  入场: 价格突破20日最高价 (需在200MA之上)
  加仓: 每上涨0.5*ATR加仓一次, 最多4次
  出场: 价格跌破10日最低价 或 2x ATR追踪止损

技术指标: Donchian Channel (20/10), 200 SMA, ATR(20)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class TurtleTradingStrategy(BaseStrategy):
    """海龟交易策略 — Donchian突破 + ATR金字塔加仓 + 追踪止损"""

    strategy_description = "海龟交易: Donchian20突破 + 200MA过滤 + ATR加仓止损"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "entry_period": {"type": "int", "default": 20, "label": "入场突破周期"},
        "exit_period": {"type": "int", "default": 10, "label": "出场突破周期"},
        "ma_period": {"type": "int", "default": 200, "label": "趋势过滤MA"},
        "atr_period": {"type": "int", "default": 20, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 2.0, "label": "止损ATR倍数"},
        "pyramid_atr_mult": {"type": "float", "default": 0.5, "label": "加仓ATR间隔"},
        "max_pyramid": {"type": "int", "default": 4, "label": "最大加仓次数"},
        "hold_min": {"type": "int", "default": 5, "label": "最少持仓天数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.entry_period = params.get('entry_period', 20)
        self.exit_period = params.get('exit_period', 10)
        self.ma_period = params.get('ma_period', 200)
        self.atr_period = params.get('atr_period', 20)
        self.atr_mult = params.get('atr_mult', 2.0)
        self.pyramid_atr_mult = params.get('pyramid_atr_mult', 0.5)
        self.max_pyramid = params.get('max_pyramid', 4)
        self.hold_min = params.get('hold_min', 5)

    def get_default_params(self):
        return {
            'entry_period': 20, 'exit_period': 10, 'ma_period': 200,
            'atr_period': 20, 'atr_mult': 2.0, 'pyramid_atr_mult': 0.5,
            'max_pyramid': 4, 'hold_min': 5,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
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
                # === 扫描所有股票寻找突破 ===
                best_sym = None
                best_dir = 0
                best_strength = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._check_entry(hist)
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

                # Track water marks
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
                    # === Exit checks ===
                    # 1. Donchian exit: price breaks exit_period low (long) or high (short)
                    exit_result = self._check_exit(hist)
                    if exit_result:
                        should_exit = True

                    # 2. ATR trailing stop
                    atr_val = self._calc_atr(hist)
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.atr_mult * atr_val:
                                should_exit = True

                    # 3. Max hold
                    if days_held >= 120:
                        should_exit = True

                # === Pyramid add ===
                if not should_exit and pyramid_count < self.max_pyramid:
                    atr_val = self._calc_atr(hist)
                    if atr_val > 0 and last_pyramid_price > 0:
                        if position_dir == 1 and current_price > last_pyramid_price + self.pyramid_atr_mult * atr_val:
                            should_add = True
                        elif position_dir == -1 and current_price < last_pyramid_price - self.pyramid_atr_mult * atr_val:
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
                    # Add position (record additional signal)
                    if position_dir == 1:
                        self._record_signal(current_time, 'buy', current_holding)
                    else:
                        self._record_signal(current_time, 'sell', current_holding)
                    last_pyramid_price = current_price
                    pyramid_count += 1

        print(f"TurtleTrading: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _check_entry(self, data):
        """检查Donchian突破入场"""
        min_len = max(self.entry_period, 60) + 5  # Lower MA requirement
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # MA trend filter (shorter period for more signals)
        ma_period = min(self.ma_period, len(close) - 1)
        ma = self._calc_sma(close, ma_period)
        if ma[-1] == 0:
            return None

        # Donchian channel: use the PREVIOUS bars (not including the last bar)
        # The last bar's close breaking above the previous 20-day high = breakout
        if len(high) < self.entry_period + 1:
            return None

        prev_high = np.max(high[-self.entry_period-1:-1])  # 20-day high excluding last bar
        prev_low = np.min(low[-self.entry_period-1:-1])
        current_close = close[-1]

        strength = 0

        # Long: close breaks above previous 20-day high
        if current_close > prev_high:
            above_ma = current_close > ma[-1]
            near_ma = abs(current_close - ma[-1]) / ma[-1] < 0.10 if ma[-1] > 0 else False

            if above_ma or near_ma:
                atr = self._calc_atr(data)
                strength = (current_close - prev_high) / atr if atr > 0 else 0.5
                return 1, strength

        # Short: close breaks below previous 20-day low
        if current_close < prev_low:
            below_ma = current_close < ma[-1]
            near_ma = abs(current_close - ma[-1]) / ma[-1] < 0.10 if ma[-1] > 0 else False

            if below_ma or near_ma:
                atr = self._calc_atr(data)
                strength = (prev_low - current_close) / atr if atr > 0 else 0.5
                return -1, strength

        return None

    def _check_exit(self, data):
        """检查Donchian出场"""
        if len(data) < self.exit_period + 1:
            return False

        low = data['low'].values
        high = data['high'].values
        close = data['close'].values

        exit_low = np.min(low[-self.exit_period:])
        exit_high = np.max(high[-self.exit_period:])

        # Long exit: close below 10-day low
        if close[-1] < exit_low:
            return True
        # Short exit: close above 10-day high
        if close[-1] > exit_high:
            return True

        return False

    def _calc_sma(self, values, period):
        n = len(values)
        result = np.zeros(n)
        for i in range(n):
            if i < period - 1:
                result[i] = np.mean(values[:i + 1]) if i > 0 else values[0]
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
        min_len = max(self.entry_period, self.ma_period) + 5
        if len(data) < min_len:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._check_entry(data)
        price = float(data['close'].iloc[-1])

        if result is not None:
            direction, strength = result
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"donchian_breakout strength={strength:.2f}",
                'price': price,
            }
        return {'action': 'hold', 'reason': 'no breakout', 'price': price}
