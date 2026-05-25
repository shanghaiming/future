"""
SuperTrend止盈策略 (SuperTrend Take-Profit Strategy)
====================================================
基于SuperTrend指标的趋势跟踪策略，结合ATR动态止损。

来源: TradingView "SuperTrend Take-Profit Dimensions [AlgoAlpha]"

核心逻辑:
  1. SuperTrend判断趋势方向
  2. 价格突破SuperTrend线入场
  3. ATR追踪止损保护利润
  4. 结合成交量确认突破有效性

技术指标: SuperTrend, ATR, 成交量
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class SuperTrendTPStrategy(BaseStrategy):
    """SuperTrend止盈策略 — SuperTrend趋势 + ATR止损 + 量能确认"""

    strategy_description = "SuperTrend: 趋势跟踪 + ATR动态止损 + 量能确认"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "atr_period": {"type": "int", "default": 10, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 3.0, "label": "SuperTrend ATR倍数"},
        "trail_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "vol_ma_period": {"type": "int", "default": 20, "label": "成交量均线周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.atr_period = params.get('atr_period', 10)
        self.atr_mult = params.get('atr_mult', 3.0)
        self.trail_mult = params.get('trail_mult', 2.5)
        self.vol_ma_period = params.get('vol_ma_period', 20)
        self.hold_min = params.get('hold_min', 3)

    def get_default_params(self):
        return {
            'atr_period': 10, 'atr_mult': 3.0, 'trail_mult': 2.5,
            'vol_ma_period': 20, 'hold_min': 3,
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

                    # ATR trailing stop
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_mult * atr_val:
                                should_exit = True

                    # Max hold
                    if days_held >= 60:
                        should_exit = True

                    # SuperTrend reversal exit
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

        print(f"SuperTrendTP: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """评估SuperTrend信号 — 使用追踪SuperTrend线"""
        if len(data) < self.atr_period + 10:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # Calculate ATR series for proper SuperTrend tracking
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Calculate SuperTrend tracking lines
        # Simplified: use last N bars to compute trailing SuperTrend
        lookback = min(n - 1, 50)
        hl2 = (high + low) / 2.0

        # Initialize bands
        upper_band = hl2[-1] + self.atr_mult * atr
        lower_band = hl2[-1] - self.atr_mult * atr

        # Track SuperTrend state backwards
        st_trend = 1  # 1=up, -1=down
        for i in range(max(n - lookback, 1), n):
            atr_i = self._calc_atr_history(data, i + 1)
            if atr_i <= 0:
                continue
            mid = (high[i] + low[i]) / 2.0
            ub = mid + self.atr_mult * atr_i
            lb = mid - self.atr_mult * atr_i

            if st_trend == 1:
                lb = max(lb, lower_band) if i > max(n - lookback, 1) else lb
            else:
                ub = min(ub, upper_band) if i > max(n - lookback, 1) else ub

            if close[i] > ub:
                st_trend = 1
            elif close[i] < lb:
                st_trend = -1

            upper_band = ub
            lower_band = lb

        score = 0
        direction = st_trend

        # SuperTrend direction
        if direction == 1:
            score += 4
        elif direction == -1:
            score -= 4

        # Trend flip detection
        if n >= 2:
            prev_atr = self._calc_atr_history(data, n - 1)
            if prev_atr > 0:
                prev_mid = (high[-2] + low[-2]) / 2.0
                prev_ub = prev_mid + self.atr_mult * prev_atr
                prev_lb = prev_mid - self.atr_mult * prev_atr
                prev_trend = -st_trend  # Assume opposite
                if close[-2] > prev_ub:
                    prev_trend = 1
                elif close[-2] < prev_lb:
                    prev_trend = -1

                if prev_trend != st_trend:
                    score += 3  # Fresh SuperTrend flip

        # Volume confirmation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= self.vol_ma_period:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-self.vol_ma_period:])
            if vol[-1] > vol_ma * 1.2:
                score += 2

        # Price momentum
        if n >= 10:
            ret = (close[-1] / close[-10] - 1) * 100
            if direction == 1 and ret > 2:
                score += 1
            elif direction == -1 and ret < -2:
                score += 1

        return score, direction, atr

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0
        return self._calc_atr_history(data, len(data))

    def _calc_atr_history(self, data, n):
        high = data['high'].values[:n]
        low = data['low'].values[:n]
        close = data['close'].values[:n]
        if len(high) < self.atr_period + 1:
            return 0
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

    def screen(self):
        data = self.data.copy()
        if len(data) < 30:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (supertrend)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
