"""
PPO背离策略 (PPO Divergence Strategy)
====================================
百分比价格振荡器(PPO)背离检测，捕捉动量衰减反转。

来源: TradingView "PPO Divergence Strategy"

核心逻辑:
  1. 计算PPO = (EMA12 - EMA26) / EMA26 * 100
  2. 价格新高但PPO未新高 → 看跌背离
  3. 价格新低但PPO未新低 → 看涨背离
  4. 量能确认 + ATR止损

技术指标: PPO(12,26,9), Volume, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class PpoDivergenceStrategy(BaseStrategy):
    """PPO背离策略 — 动量背离反转检测"""

    strategy_description = "PPO背离: 动量背离检测 + 量能确认 + ATR止损"
    strategy_category = "momentum"
    strategy_params_schema = {
        "ppo_fast": {"type": "int", "default": 12, "label": "PPO快线"},
        "ppo_slow": {"type": "int", "default": 26, "label": "PPO慢线"},
        "ppo_signal": {"type": "int", "default": 9, "label": "PPO信号线"},
        "lookback": {"type": "int", "default": 20, "label": "背离回望"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ppo_fast = params.get('ppo_fast', 12)
        self.ppo_slow = params.get('ppo_slow', 26)
        self.ppo_signal = params.get('ppo_signal', 9)
        self.lookback = params.get('lookback', 20)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'ppo_fast': 12, 'ppo_slow': 26, 'ppo_signal': 9,
            'lookback': 20, 'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"PpoDivergence: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = self.ppo_slow + self.ppo_signal + self.lookback + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # PPO calculation
        ppo = self._calc_ppo_series(close)
        if ppo is None:
            return None

        # Divergence detection
        div = self._detect_divergence(close, ppo, min(self.lookback, n - 1))

        if div == 'bullish':
            score += 5
        elif div == 'bearish':
            score -= 5

        # PPO direction
        if len(ppo) >= 2:
            if ppo[-1] > ppo[-2]:
                score += 2
            elif ppo[-1] < ppo[-2]:
                score -= 2

        # PPO histogram
        signal = self._calc_ema(ppo, self.ppo_signal)
        hist = ppo - signal
        if hist[-1] > 0:
            score += 1
        elif hist[-1] < 0:
            score -= 1

        # Volume confirmation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * 1.3:
                score += 1 if score > 0 else -1

        if score == 0:
            return None
        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_ppo_series(self, close):
        n = len(close)
        if n < self.ppo_slow + 1:
            return None
        fast_ema = self._calc_ema(close, self.ppo_fast)
        slow_ema = self._calc_ema(close, self.ppo_slow)
        ppo = np.where(slow_ema != 0, (fast_ema - slow_ema) / slow_ema * 100, 0)
        return ppo

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

    def _detect_divergence(self, price, indicator, lookback):
        if lookback < 5:
            return None
        p = price[-lookback:]
        ind = indicator[-lookback:]

        p_highs, i_at_highs = [], []
        p_lows, i_at_lows = [], []

        for i in range(1, len(p) - 1):
            if p[i] > p[i-1] and p[i] > p[i+1]:
                p_highs.append(p[i])
                i_at_highs.append(ind[i])
            elif p[i] < p[i-1] and p[i] < p[i+1]:
                p_lows.append(p[i])
                i_at_lows.append(ind[i])

        if len(p_highs) >= 2:
            if p_highs[-1] > p_highs[-2] and i_at_highs[-1] < i_at_highs[-2]:
                return 'bearish'
        if len(p_lows) >= 2:
            if p_lows[-1] < p_lows[-2] and i_at_lows[-1] > i_at_lows[-2]:
                return 'bullish'
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
        if len(data) < 60:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (ppo_div)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
