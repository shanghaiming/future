"""
突破回调反转策略 (Breakout Pullback Reversal Strategy)
======================================================
检测结构性突破后的回调翻转区域入场。

来源: TradingView "Breakout Pullback - Reversal"

核心逻辑:
  1. 价格突破最近摆动高点/低点
  2. 等待回调至突破区域(Flip Zone)
  3. Fibonacci回调深度38.2%-78.6%确认
  4. RSI动量方向确认
  5. 成交量验证

技术指标: Swing Pivots, Fibonacci, RSI, Volume, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class BreakoutPullbackStrategy(BaseStrategy):
    """突破回调反转策略 — 结构突破 + Flip Zone回调入场"""

    strategy_description = "突破回调: 结构突破 + Flip Zone + Fib回调确认"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "fib_min": {"type": "float", "default": 0.382, "label": "Fib回调最小"},
        "fib_max": {"type": "float", "default": 0.786, "label": "Fib回调最大"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.pivot_len = params.get('pivot_len', 5)
        self.rsi_period = params.get('rsi_period', 14)
        self.fib_min = params.get('fib_min', 0.382)
        self.fib_max = params.get('fib_max', 0.786)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'pivot_len': 5, 'rsi_period': 14, 'fib_min': 0.382, 'fib_max': 0.786,
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

        print(f"BreakoutPullback: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _find_pivots(self, high, low, n):
        piv_len = self.pivot_len
        if n < piv_len * 2 + 1:
            return [], []
        swing_highs = []
        swing_lows = []
        for i in range(piv_len, n - piv_len):
            is_high = all(high[i] >= high[i + j] for j in range(-piv_len, piv_len + 1) if j != 0)
            is_low = all(low[i] <= low[i + j] for j in range(-piv_len, piv_len + 1) if j != 0)
            if is_high:
                swing_highs.append((i, high[i]))
            if is_low:
                swing_lows.append((i, low[i]))
        return swing_highs, swing_lows

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
        if len(data) < self.pivot_len * 2 + 20:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        swing_highs, swing_lows = self._find_pivots(high, low, n)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        current_price = close[-1]

        # Bullish breakout pullback: price broke above last swing high, now pulling back
        last_sh = swing_highs[-1]
        prev_sh = swing_highs[-2]
        last_sl = swing_lows[-1]

        # Check if breakout happened recently (close > last swing high in last few bars)
        if len(close) >= 5:
            recent_high = np.max(close[-5:])
            if recent_high > last_sh[1]:
                # Breakout confirmed, check for pullback to flip zone
                pullback_low = np.min(low[-3:])
                pullback_from = recent_high
                if pullback_from > 0:
                    fib_depth = (pullback_from - pullback_low) / (pullback_from - last_sl[1])
                    if self.fib_min <= fib_depth <= self.fib_max:
                        score += 5  # Pullback in golden zone
                    elif fib_depth < self.fib_min:
                        score += 3  # Shallow pullback

        # Bearish breakout pullback
        if len(close) >= 5:
            recent_low = np.min(close[-5:])
            if recent_low < last_sl[1]:
                pullback_high = np.max(high[-3:])
                pullback_from = recent_low
                prev_high = last_sh[1]
                if prev_high > pullback_from:
                    fib_depth = (pullback_high - pullback_from) / (prev_high - pullback_from)
                    if self.fib_min <= fib_depth <= self.fib_max:
                        score -= 5
                    elif fib_depth < self.fib_min:
                        score -= 3

        # RSI momentum confirmation
        rsi = self._calc_rsi(close, self.rsi_period)
        if score > 0 and rsi > 50:
            score += 2
        elif score < 0 and rsi < 50:
            score += 2
        elif score > 0 and rsi < 40:
            score -= 1
        elif score < 0 and rsi > 60:
            score -= 1

        # Volume confirmation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * 1.2:
                if score > 0:
                    score += 1
                elif score < 0:
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
                'reason': f"score={score} (breakout_pullback)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
