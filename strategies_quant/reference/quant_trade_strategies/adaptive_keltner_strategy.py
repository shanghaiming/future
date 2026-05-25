"""
自适应Keltner通道策略 (Adaptive Keltner Channel Strategy)
=======================================================
ATR百分位自适应Keltner通道宽度 + 状态感知交易信号。

来源: TradingView "Adaptive Keltner Channel [NovaLens]"

核心逻辑:
  1. EMA中心线
  2. ATR百分位自适应带宽(高波动=宽通道)
  3. 价格突破上轨=做多
  4. 价格跌破下轨=做空
  5. BB在KC内=squeeze不交易

技术指标: EMA, ATR百分位, Keltner Channel
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class AdaptiveKeltnerStrategy(BaseStrategy):
    """自适应Keltner通道策略 — ATR%自适应带宽"""

    strategy_description = "自适应KC: ATR%自适应带宽 + 状态交易"
    strategy_category = "volatility"
    strategy_params_schema = {
        "ema_period": {"type": "int", "default": 20, "label": "EMA中心线"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_lookback": {"type": "int", "default": 100, "label": "ATR百分位回望"},
        "kc_mult_low": {"type": "float", "default": 1.5, "label": "低波动KC倍数"},
        "kc_mult_high": {"type": "float", "default": 3.0, "label": "高波动KC倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ema_period = params.get('ema_period', 20)
        self.atr_period = params.get('atr_period', 14)
        self.atr_lookback = params.get('atr_lookback', 100)
        self.kc_mult_low = params.get('kc_mult_low', 1.5)
        self.kc_mult_high = params.get('kc_mult_high', 3.0)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'ema_period': 20, 'atr_period': 14, 'atr_lookback': 100,
            'kc_mult_low': 1.5, 'kc_mult_high': 3.0,
            'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"AdaptiveKeltner: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.ema_period, self.atr_lookback) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # EMA center
        ema = np.mean(close[-self.ema_period:])

        # ATR
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # ATR percentile for adaptive width
        atr_series = self._calc_atr_series(data)
        if atr_series is not None and len(atr_series) >= self.atr_lookback:
            atr_pct = np.sum(atr_series[-self.atr_lookback:] < atr) / self.atr_lookback
        else:
            atr_pct = 0.5

        # Adaptive KC multiplier
        kc_mult = self.kc_mult_low + (self.kc_mult_high - self.kc_mult_low) * atr_pct

        upper = ema + kc_mult * atr
        lower = ema - kc_mult * atr

        current = close[-1]

        # Breakout signals
        if current > upper:
            score += 4
        elif current < lower:
            score -= 4

        # Center reclaim
        if n >= 2:
            prev = close[-2]
            if prev < ema and current > ema:
                score += 3  # Reclaimed above center
            elif prev > ema and current < ema:
                score -= 3

        # Price vs center
        if current > ema:
            score += 1
        else:
            score -= 1

        direction = 1 if score > 0 else -1
        return score, direction, atr

    def _calc_atr_series(self, data):
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        if len(close) < self.atr_period + 1:
            return None
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        # Rolling ATR
        atr_arr = np.convolve(tr, np.ones(self.atr_period) / self.atr_period, mode='valid')
        return atr_arr

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
        if len(data) < 120:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (adaptive_kc)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
