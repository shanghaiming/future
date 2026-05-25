"""
Kijun+RSI策略 (Kijun-sen + RSI Strategy)
=========================================
一目均衡表基准线(Kijun-sen)趋势过滤 + RSI时机入场。

来源: TradingView "BTC Kijun + RSI Strategy [75m]"

核心逻辑:
  1. Kijun-sen(26周期)判断中期趋势
  2. 价格在Kijun之上=多头趋势, 之下=空头趋势
  3. RSI(14)从超卖区回升=做多时机
  4. RSI从超买区回落=做空时机

技术指标: Ichimoku Kijun-sen, RSI(14)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class KijunRsiStrategy(BaseStrategy):
    """Kijun+RSI策略 — 一目Kijun趋势 + RSI时机"""

    strategy_description = "Kijun+RSI: 一目基准线趋势 + RSI入场时机"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "kijun_period": {"type": "int", "default": 26, "label": "Kijun周期"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_ob": {"type": "int", "default": 70, "label": "RSI超买"},
        "rsi_os": {"type": "int", "default": 30, "label": "RSI超卖"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.kijun_period = params.get('kijun_period', 26)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_ob = params.get('rsi_ob', 70)
        self.rsi_os = params.get('rsi_os', 30)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'kijun_period': 26, 'rsi_period': 14, 'rsi_ob': 70, 'rsi_os': 30,
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

        print(f"KijunRsi: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.kijun_period, self.rsi_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        # 1. Kijun-sen = (highest high + lowest low) / 2 over kijun_period
        kijun = (np.max(high[-self.kijun_period:]) + np.min(low[-self.kijun_period:])) / 2.0

        # Price vs Kijun
        if close[-1] > kijun:
            score += 4  # Above Kijun = bullish
        elif close[-1] < kijun:
            score -= 4  # Below Kijun = bearish

        # Kijun slope (trend direction)
        if n >= self.kijun_period + 5:
            prev_kijun = (np.max(high[-self.kijun_period-5:-5]) + np.min(low[-self.kijun_period-5:-5])) / 2.0
            if kijun > prev_kijun:
                score += 2
            elif kijun < prev_kijun:
                score -= 2

        # 2. RSI timing
        rsi = self._calc_rsi(close)
        if rsi < self.rsi_os:
            score += 3  # Oversold → buy setup
        elif rsi > self.rsi_ob:
            score -= 3  # Overbought → sell setup

        # RSI crossing back from extreme
        if n >= 2:
            prev_rsi = self._calc_rsi(close[:-1])
            if prev_rsi < self.rsi_os and rsi > self.rsi_os:
                score += 2  # Bouncing from oversold
            elif prev_rsi > self.rsi_ob and rsi < self.rsi_ob:
                score -= 2

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_rsi(self, close):
        if len(close) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        if avg_loss == 0:
            return 100.0
        return 100.0 - 100.0 / (1 + avg_gain / avg_loss)

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
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (kijun+rsi)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
