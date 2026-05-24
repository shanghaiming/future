"""
结构张力振荡器策略 (7-Point Structural Oscillator Strategy)
=========================================================
7个关键结构点计算价格综合位移(结构张力)。

来源: TradingView "7-Point Structural Oscillator [Tension]"

核心逻辑:
  1. 7个结构坐标: 前/当前swing high/low, 摆动间最高/最低/中点
  2. 计算价格相对于7个点的净位移(结构张力)
  3. 正张力=多头主导, 负张力=空头主导
  4. 零线交叉=突破结构均衡

技术指标: Swing Points, Structural Tension, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class StructuralTensionStrategy(BaseStrategy):
    """结构张力振荡器策略 — 7点结构张力评分"""

    strategy_description = "结构张力: 7点结构位移 + 张力方向评分"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "lookback": {"type": "int", "default": 50, "label": "结构回溯"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.pivot_len = params.get('pivot_len', 5)
        self.lookback = params.get('lookback', 50)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'pivot_len': 5, 'lookback': 50,
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

        print(f"StructuralTension: 生成 {len(self.signals)} 个信号")
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

    def _evaluate(self, data):
        if len(data) < self.pivot_len * 2 + 20:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        score = 0

        swing_highs, swing_lows = self._find_pivots(high, low, n)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # 7 structural reference points
        prev_sh = swing_highs[-2][1] if len(swing_highs) >= 2 else high[-1]
        curr_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1] if len(swing_lows) >= 2 else low[-1]
        curr_sl = swing_lows[-1][1]

        inter_high = max(curr_sh, prev_sh)
        inter_low = min(curr_sl, prev_sl)
        inter_mid = (inter_high + inter_low) / 2.0

        current_price = close[-1]

        # Calculate tension: net displacement from 7 points
        ref_points = [prev_sh, curr_sh, prev_sl, curr_sl, inter_high, inter_low, inter_mid]
        displacements = []
        for rp in ref_points:
            if atr > 0:
                displacements.append((current_price - rp) / atr)

        tension = np.mean(displacements)

        # Tension scoring
        if tension > 1.0:
            score += 5
        elif tension > 0.5:
            score += 3
        elif tension > 0.2:
            score += 1
        elif tension < -1.0:
            score -= 5
        elif tension < -0.5:
            score -= 3
        elif tension < -0.2:
            score -= 1

        # Zero-line cross (trend shift)
        if n >= 2:
            prev_close = close[-2]
            prev_displacements = [(prev_close - rp) / atr for rp in ref_points]
            prev_tension = np.mean(prev_displacements)

            if prev_tension < 0 and tension > 0:
                score += 3  # Bullish cross
            elif prev_tension > 0 and tension < 0:
                score -= 3  # Bearish cross

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
                'reason': f"score={score} (structural_tension)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
