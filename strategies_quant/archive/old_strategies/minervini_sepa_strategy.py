"""
Minervini SEPA策略 (Minervini SEPA System)
============================================
Mark Minervini的SEPA(特定入场点分析)方法论。

来源: TradingView "Minervini SEPA System"

核心逻辑:
  1. 8项Trend Template标准评分
  2. VCP(波动率收缩形态)检测
  3. 成交量确认突破
  4. ATR动态止损

技术指标: SMA(50/150/200), ATR, Volume, ROC
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class MinerviniSepaStrategy(BaseStrategy):
    """Minervini SEPA策略 — Trend Template评分 + VCP突破"""

    strategy_description = "Minervini SEPA: Trend Template 8项评分 + VCP突破"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "sma_fast": {"type": "int", "default": 50, "label": "快SMA"},
        "sma_mid": {"type": "int", "default": 150, "label": "中SMA"},
        "sma_slow": {"type": "int", "default": 200, "label": "慢SMA"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.sma_fast = params.get('sma_fast', 50)
        self.sma_mid = params.get('sma_mid', 150)
        self.sma_slow = params.get('sma_slow', 200)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'sma_fast': 50, 'sma_mid': 150, 'sma_slow': 200,
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

        print(f"MinerviniSEPA: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _calc_sma(self, values, period):
        n = len(values)
        if n < period:
            return None
        return np.mean(values[-period:])

    def _evaluate(self, data):
        min_len = self.sma_slow + 50
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        sma50 = self._calc_sma(close, self.sma_fast)
        sma150 = self._calc_sma(close, self.sma_mid)
        sma200 = self._calc_sma(close, self.sma_slow)

        if sma50 is None or sma150 is None or sma200 is None:
            return None

        # Trend Template 8 criteria (bullish)
        tt_score = 0
        # 1. Price > 150 MA and 200 MA
        if close[-1] > sma150 and close[-1] > sma200:
            tt_score += 1
        # 2. 150 MA > 200 MA
        if sma150 > sma200:
            tt_score += 1
        # 3. 200 MA trending up (20-bar slope)
        sma200_prev = self._calc_sma(close[:-20], self.sma_slow) if n > self.sma_slow + 20 else None
        if sma200_prev and sma200 > sma200_prev:
            tt_score += 1
        # 4. 50 MA > 150 MA and 200 MA
        if sma50 > sma150 and sma50 > sma200:
            tt_score += 1
        # 5. Price > 50 MA
        if close[-1] > sma50:
            tt_score += 1
        # 6. Price > 25% above 52-week low
        low_252 = np.min(low[-min(252, n):])
        if close[-1] > low_252 * 1.25:
            tt_score += 1
        # 7. Price within 25% of 52-week high
        high_252 = np.max(high[-min(252, n):])
        if close[-1] > high_252 * 0.75:
            tt_score += 1
        # 8. Positive momentum (ROC 50 > 0)
        if n > 50 and close[-1] > close[-50]:
            tt_score += 1

        # Score based on TT criteria
        if tt_score >= 7:
            score += 5
        elif tt_score >= 5:
            score += 3
        elif tt_score >= 3:
            score += 1

        # VCP detection (ATR shrinking + volume shrinking)
        atr = self._calc_atr(data)
        if n >= 50 and atr > 0:
            atr_ma = np.mean([self._calc_atr_history(data, i) for i in range(n - 50, n) if self._calc_atr_history(data, i) > 0])
            if atr_ma > 0 and atr < atr_ma * 0.6:
                vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
                if vol_col:
                    vol = data[vol_col].values
                    vol_ma = np.mean(vol[-50:])
                    if vol[-1] < vol_ma * 0.7:
                        if score > 0:
                            score += 3  # VCP detected in uptrend
                        # Can also be bearish VCP (inverse)

        # Volume breakout confirmation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 50:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-50:])
            if vol[-1] > vol_ma * 1.4:
                if score > 0:
                    score += 2

        # Bearish: reverse conditions
        bear_score = 8 - tt_score
        if bear_score >= 7:
            score -= 5
        elif bear_score >= 5:
            score -= 3

        # Direction: whichever is stronger
        direction = 1 if score > 0 else -1
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

    def _calc_atr_history(self, data, n):
        if n < self.atr_period + 1:
            return 0
        high = data['high'].values[:n]
        low = data['low'].values[:n]
        close = data['close'].values[:n]
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

    def screen(self):
        data = self.data.copy()
        if len(data) < self.sma_slow + 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (minervini_sepa)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
