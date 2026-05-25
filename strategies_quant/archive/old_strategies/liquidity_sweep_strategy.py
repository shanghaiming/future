"""
流动性扫荡策略 (Liquidity Sweep Detector Strategy)
===================================================
识别流动性池，检测扫荡后反转。

来源: TradingView "Liquidity Sweep Detector [QuantAlgo]"

核心逻辑:
  1. 摆动高低点附近形成流动性池
  2. 价格穿越后快速反转 = 扫荡
  3. 成交量验证扫荡有效性
  4. ATR追踪止损

技术指标: Swing Pivots, Liquidity Pools, Volume, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class LiquiditySweepStrategy(BaseStrategy):
    """流动性扫荡策略 — 流动性池检测 + 扫荡反转"""

    strategy_description = "流动性扫荡: 摆动点流动性池 + 扫荡反转检测"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "reversal_pct": {"type": "float", "default": 0.003, "label": "反转百分比"},
        "lookback": {"type": "int", "default": 50, "label": "流动性池回溯"},
        "vol_mult": {"type": "float", "default": 1.3, "label": "扫荡量能倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.pivot_len = params.get('pivot_len', 5)
        self.reversal_pct = params.get('reversal_pct', 0.003)
        self.lookback = params.get('lookback', 50)
        self.vol_mult = params.get('vol_mult', 1.3)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'pivot_len': 5, 'reversal_pct': 0.003, 'lookback': 50,
            'vol_mult': 1.3, 'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"LiquiditySweep: 生成 {len(self.signals)} 个信号")
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

    def _detect_sweep(self, data):
        """Detect liquidity sweep and reversal"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        if n < self.pivot_len * 2 + 5:
            return 0

        swing_highs, swing_lows = self._find_pivots(high, low, n)
        score = 0

        # Check recent price action against swing levels
        recent_high = high[-1]
        recent_low = low[-1]
        recent_close = close[-1]

        # High sweep: price broke above a swing high then closed below it
        for idx, sh_price in swing_highs[-5:]:
            if recent_high > sh_price and recent_close < sh_price:
                score -= 4  # Bearish sweep → sell signal

        # Low sweep: price broke below a swing low then closed above it
        for idx, sl_price in swing_lows[-5:]:
            if recent_low < sl_price and recent_close > sl_price:
                score += 4  # Bullish sweep → buy signal

        # Volume confirmation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * self.vol_mult:
                if score > 0:
                    score += 2
                elif score < 0:
                    score -= 2

        return score

    def _evaluate(self, data):
        if len(data) < self.pivot_len * 2 + 20:
            return None

        score = self._detect_sweep(data)
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
                'reason': f"score={score} (liquidity_sweep)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
