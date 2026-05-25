"""
EMA交叉+Stoch RSI策略 (EMA Cross + Stochastic RSI Strategy)
===========================================================
50 EMA突破入场 + 200 EMA趋势过滤 + Stochastic RSI超卖回调。

来源: TradingView "50 EMA Cross + 200 EMA Filter + Stoch RSI"

核心逻辑:
  1. 200 EMA判断大趋势方向
  2. 价格上穿50 EMA入场做多(反之做空)
  3. Stochastic RSI在超卖区确认回调到位
  4. ATR追踪止损

技术指标: EMA(50,200), Stochastic RSI(14,14,3,3)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EmaStochRsiStrategy(BaseStrategy):
    """EMA交叉+Stoch RSI策略 — EMA趋势过滤 + StochRSI时机"""

    strategy_description = "EMA趋势过滤: 50/200 EMA + StochRSI回调入场"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "ema_fast": {"type": "int", "default": 50, "label": "快EMA"},
        "ema_slow": {"type": "int", "default": 200, "label": "慢EMA"},
        "stoch_k_period": {"type": "int", "default": 14, "label": "Stoch K周期"},
        "stoch_d_period": {"type": "int", "default": 3, "label": "Stoch D平滑"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "stoch_ob": {"type": "int", "default": 80, "label": "Stoch超买"},
        "stoch_os": {"type": "int", "default": 20, "label": "Stoch超卖"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ema_fast = params.get('ema_fast', 50)
        self.ema_slow = params.get('ema_slow', 200)
        self.stoch_k_period = params.get('stoch_k_period', 14)
        self.stoch_d_period = params.get('stoch_d_period', 3)
        self.rsi_period = params.get('rsi_period', 14)
        self.stoch_ob = params.get('stoch_ob', 80)
        self.stoch_os = params.get('stoch_os', 20)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'ema_fast': 50, 'ema_slow': 200,
            'stoch_k_period': 14, 'stoch_d_period': 3, 'rsi_period': 14,
            'stoch_ob': 80, 'stoch_os': 20,
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

        print(f"EmaStochRsi: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.ema_slow, self.stoch_k_period + self.rsi_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # 1. EMA trend
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)

        if close[-1] > ema_f[-1] and ema_f[-1] > ema_s[-1]:
            score += 4  # Strong uptrend
        elif close[-1] < ema_f[-1] and ema_f[-1] < ema_s[-1]:
            score -= 4  # Strong downtrend
        elif close[-1] > ema_f[-1]:
            score += 2
        elif close[-1] < ema_f[-1]:
            score -= 2

        # EMA cross signal
        if n >= 2:
            prev_above = close[-2] > ema_f[-2]
            curr_above = close[-1] > ema_f[-1]
            if not prev_above and curr_above:
                score += 3  # Price crossed above fast EMA
            elif prev_above and not curr_above:
                score -= 3

        # 2. Stochastic RSI
        stoch_k = self._calc_stoch_rsi(close)
        if stoch_k is not None:
            if stoch_k < self.stoch_os:
                score += 2  # Oversold bounce setup
            elif stoch_k > self.stoch_ob:
                score -= 2  # Overbought

            # StochRSI crossing up from oversold
            prev_k = self._calc_stoch_rsi(close[:-1]) if n > self.stoch_k_period + self.rsi_period + 5 else None
            if prev_k is not None:
                if prev_k < self.stoch_os and stoch_k > self.stoch_os:
                    score += 2
                elif prev_k > self.stoch_ob and stoch_k < self.stoch_ob:
                    score -= 2

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_stoch_rsi(self, close):
        """Calculate Stochastic RSI"""
        if len(close) < self.rsi_period + self.stoch_k_period + 1:
            return None

        # First calculate RSI series
        rsi_arr = self._calc_rsi_series(close)
        if rsi_arr is None:
            return None

        # Then apply Stochastic to RSI
        rsi_recent = rsi_arr[-self.stoch_k_period:]
        if len(rsi_recent) < self.stoch_k_period:
            return None

        lowest = np.min(rsi_recent)
        highest = np.max(rsi_recent)
        if highest == lowest:
            return 50.0
        return (rsi_arr[-1] - lowest) / (highest - lowest) * 100.0

    def _calc_rsi_series(self, close):
        n = len(close)
        if n < self.rsi_period + 1:
            return None
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi_arr = np.full(n, 50.0)
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])
        if avg_loss == 0:
            rsi_arr[self.rsi_period] = 100.0
        else:
            rsi_arr[self.rsi_period] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)

        for i in range(self.rsi_period, len(gains)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i]) / self.rsi_period
            if avg_loss == 0:
                rsi_arr[i + 1] = 100.0
            else:
                rsi_arr[i + 1] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)
        return rsi_arr

    def _calc_ema_series(self, values, period):
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
        if len(data) < 250:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (ema+stochrsi)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
