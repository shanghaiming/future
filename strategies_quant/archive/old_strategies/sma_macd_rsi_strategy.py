"""
SMA+MACD+RSI三重确认策略 (SMA+MACD+RSI Triple Confirmation Strategy)
===================================================================
趋势(SMA) + 动量(MACD) + 强度(RSI)三重确认入场。

来源: TradingView "SMA + MACD + RSI Signals"

核心逻辑:
  1. SMA10 > SMA20 趋势向上
  2. MACD histogram > 0 动量向上
  3. RSI在40-70区间(不超买不超卖)
  4. 三者同时满足才入场

技术指标: SMA(10,20), MACD(12,26,9), RSI(14)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class SmaMacdRsiStrategy(BaseStrategy):
    """SMA+MACD+RSI三重确认策略"""

    strategy_description = "三重确认: SMA趋势 + MACD动量 + RSI强度"
    strategy_category = "momentum"
    strategy_params_schema = {
        "sma_fast": {"type": "int", "default": 10, "label": "快SMA"},
        "sma_slow": {"type": "int", "default": 20, "label": "慢SMA"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_ob": {"type": "int", "default": 70, "label": "RSI超买"},
        "rsi_os": {"type": "int", "default": 30, "label": "RSI超卖"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.sma_fast = params.get('sma_fast', 10)
        self.sma_slow = params.get('sma_slow', 20)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_ob = params.get('rsi_ob', 70)
        self.rsi_os = params.get('rsi_os', 30)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'sma_fast': 10, 'sma_slow': 20,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'rsi_period': 14, 'rsi_ob': 70, 'rsi_os': 30,
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

        print(f"SmaMacdRsi: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.sma_slow, self.macd_slow + self.macd_signal, self.rsi_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # 1. SMA trend
        sma_f = np.mean(close[-self.sma_fast:])
        sma_s = np.mean(close[-self.sma_slow:])
        if sma_f > sma_s:
            score += 3
        elif sma_f < sma_s:
            score -= 3

        # SMA crossover fresh signal
        if n >= self.sma_slow + 2:
            prev_sma_f = np.mean(close[-self.sma_fast-1:-1])
            prev_sma_s = np.mean(close[-self.sma_slow-1:-1])
            if prev_sma_f <= prev_sma_s and sma_f > sma_s:
                score += 2  # Golden cross
            elif prev_sma_f >= prev_sma_s and sma_f < sma_s:
                score -= 2  # Death cross

        # 2. MACD
        macd_hist = self._calc_macd_hist(close)
        if macd_hist > 0:
            score += 3
        elif macd_hist < 0:
            score -= 3

        # 3. RSI
        rsi = self._calc_rsi(close)
        if self.rsi_os < rsi < self.rsi_ob:
            # Neutral zone - confirm direction
            if rsi > 50:
                score += 2
            else:
                score -= 2
        elif rsi >= self.rsi_ob:
            score -= 2  # Overbought, caution
        elif rsi <= self.rsi_os:
            score += 2  # Oversold, potential bounce

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_macd_hist(self, close):
        fast_ema = self._calc_ema(close, self.macd_fast)
        slow_ema = self._calc_ema(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal = self._calc_ema(macd_line, self.macd_signal)
        return float(macd_line[-1] - signal[-1])

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
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1 + rs)

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
                'reason': f"score={score} (sma+macd+rsi)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
