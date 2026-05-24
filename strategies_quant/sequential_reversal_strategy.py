"""
顺序反转策略 (Sequential Reversal Strategy)
============================================
基于Demark Sequential的序列计数反转检测。

来源: TradingView "Sequential Reversal"

核心逻辑:
  1. 连续9根收盘高于/低于4根前收盘=Setup完成
  2. Setup完成后开始Countdown(13根计数)
  3. Countdown完成=价格耗竭信号
  4. 结合RSI确认反转方向

技术指标: Sequential Count, RSI
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class SequentialReversalStrategy(BaseStrategy):
    """顺序反转策略 — Demark Sequential计数 + RSI确认"""

    strategy_description = "顺序反转: Demark Sequential(9/13) + RSI确认"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "setup_len": {"type": "int", "default": 9, "label": "Setup长度"},
        "countdown_len": {"type": "int", "default": 13, "label": "Countdown长度"},
        "compare_offset": {"type": "int", "default": 4, "label": "比较偏移"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.setup_len = params.get('setup_len', 9)
        self.countdown_len = params.get('countdown_len', 13)
        self.compare_offset = params.get('compare_offset', 4)
        self.rsi_period = params.get('rsi_period', 14)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'setup_len': 9, 'countdown_len': 13, 'compare_offset': 4,
            'rsi_period': 14, 'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"SequentialReversal: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _calc_setup(self, close, n):
        """Check if buy/sell setup is completed"""
        if n < self.setup_len + self.compare_offset:
            return 0, 0  # buy_setup_count, sell_setup_count

        buy_count = 0
        sell_count = 0

        # Buy setup: close < close[4] for 9 consecutive bars
        for i in range(n - self.setup_len, n):
            if i >= self.compare_offset:
                if close[i] < close[i - self.compare_offset]:
                    buy_count += 1
                else:
                    buy_count = 0

        # Sell setup: close > close[4] for 9 consecutive bars
        for i in range(n - self.setup_len, n):
            if i >= self.compare_offset:
                if close[i] > close[i - self.compare_offset]:
                    sell_count += 1
                else:
                    sell_count = 0

        return buy_count, sell_count

    def _calc_rsi(self, values, period):
        n = len(values)
        if n < period + 1:
            return 50
        deltas = np.diff(values)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def _evaluate(self, data):
        min_len = self.setup_len + self.compare_offset + 20
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        buy_setup, sell_setup = self._calc_setup(close, n)

        # Setup completion scoring
        if buy_setup >= self.setup_len:
            score += 4  # Buy setup complete = bearish exhaustion
        elif buy_setup >= self.setup_len - 2:
            score += 2  # Near buy setup

        if sell_setup >= self.setup_len:
            score -= 4  # Sell setup complete = bullish exhaustion
        elif sell_setup >= self.setup_len - 2:
            score -= 2

        # Countdown: count bars that meet criteria within active setup
        # Simplified: count consecutive bars meeting condition in recent window
        countdown_buy = 0
        countdown_sell = 0
        for i in range(max(self.compare_offset, n - 30), n):
            if close[i] <= close[i - 2]:  # Countdown buy condition
                countdown_buy += 1
            if close[i] >= close[i - 2]:  # Countdown sell condition
                countdown_sell += 1

        if countdown_buy >= self.countdown_len:
            score += 3
        elif countdown_buy >= self.countdown_len - 3:
            score += 1

        if countdown_sell >= self.countdown_len:
            score -= 3
        elif countdown_sell >= self.countdown_len - 3:
            score -= 1

        # RSI confirmation
        rsi = self._calc_rsi(close, self.rsi_period)
        if score > 0 and rsi < 35:
            score += 2  # Oversold confirms buy
        elif score < 0 and rsi > 65:
            score += 2  # Overbought confirms sell
        elif score > 0 and rsi > 60:
            score -= 1  # RSI not confirming

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
        if len(data) < self.setup_len + self.compare_offset + 20:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (sequential)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
