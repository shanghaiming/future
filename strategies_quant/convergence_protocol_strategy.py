"""
收敛协议策略 (Convergence Protocol Strategy)
=============================================
4模块(结构趋势/波动率/Delta压力/流动性)收敛系统。

来源: TradingView "Convergence Protocol [JOAT]"

核心逻辑:
  1. SMEMA双平滑均线判断趋势方向
  2. ATR波动率状态分类
  3. Delta成交量压力方向
  4. BOS结构突破确认
  5. 0-4共融评分汇总

技术指标: SMEMA, ATR, Delta Volume, BOS
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class ConvergenceProtocolStrategy(BaseStrategy):
    """收敛协议策略 — 4模块收敛评分"""

    strategy_description = "收敛协议: SMEMA趋势 + 波动率 + Delta + 结构BOS"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "smema_period": {"type": "int", "default": 20, "label": "SMEMA周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "delta_lookback": {"type": "int", "default": 20, "label": "Delta回溯"},
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.smema_period = params.get('smema_period', 20)
        self.atr_period = params.get('atr_period', 14)
        self.delta_lookback = params.get('delta_lookback', 20)
        self.pivot_len = params.get('pivot_len', 5)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'smema_period': 20, 'atr_period': 14, 'delta_lookback': 20,
            'pivot_len': 5, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"ConvergenceProtocol: 生成 {len(self.signals)} 个信号")
        return self.signals

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

    def _calc_sma(self, values, period):
        n = len(values)
        result = np.empty(n)
        for i in range(n):
            if i < period - 1:
                result[i] = np.mean(values[:i + 1])
            else:
                result[i] = np.mean(values[i - period + 1:i + 1])
        return result

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
        min_len = self.smema_period * 2 + 20
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # Module 1: SMEMA trend direction
        ema = self._calc_ema(close, self.smema_period)
        smema = self._calc_sma(ema, self.smema_period)
        trend_signal = 1 if smema[-1] > smema[-2] else (-1 if smema[-1] < smema[-2] else 0)

        # Module 2: Volatility state (ATR ratio)
        atr = self._calc_atr(data)
        atr_series = self._calc_atr_series(data)
        if len(atr_series) > 20 and np.mean(atr_series[-20:]) > 0:
            vol_ratio = atr / np.mean(atr_series[-20:])
        else:
            vol_ratio = 1.0
        # High vol = trending, low vol = ranging
        vol_signal = 1 if vol_ratio > 1.1 else (-1 if vol_ratio < 0.8 else 0)

        # Module 3: Delta pressure
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        delta_signal = 0
        if vol_col and n >= self.delta_lookback + 1:
            vol = data[vol_col].values
            buy_vol = np.where(close[1:] > close[:-1], vol[1:], 0)
            sell_vol = np.where(close[1:] < close[:-1], vol[1:], 0)
            cum_buy = np.sum(buy_vol[-self.delta_lookback:])
            cum_sell = np.sum(sell_vol[-self.delta_lookback:])
            total = cum_buy + cum_sell
            if total > 0:
                net = (cum_buy - cum_sell) / total
                delta_signal = 1 if net > 0.1 else (-1 if net < -0.1 else 0)

        # Module 4: BOS (Break of Structure)
        swing_highs, swing_lows = self._find_pivots(high, low, n)
        bos_signal = 0
        if len(swing_highs) >= 2:
            if swing_highs[-1][1] > swing_highs[-2][1]:
                bos_signal = 1  # Bullish BOS
        if len(swing_lows) >= 2:
            if swing_lows[-1][1] < swing_lows[-2][1]:
                bos_signal -= 1  # Bearish BOS

        # Confluence score: count how many modules agree
        signals = [trend_signal, vol_signal, delta_signal, bos_signal]
        bull_count = sum(1 for s in signals if s > 0)
        bear_count = sum(1 for s in signals if s < 0)

        score = 0
        if bull_count >= 3:
            score = 5
        elif bull_count >= 2:
            score = 3
        elif bull_count >= 1:
            score = 1

        if bear_count >= 3:
            score -= 5
        elif bear_count >= 2:
            score -= 3
        elif bear_count >= 1:
            score -= 1

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

    def _calc_atr_series(self, data):
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(close)
        if n < self.atr_period + 1:
            return np.array([])
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        result = np.zeros(n)
        for i in range(self.atr_period, n):
            result[i] = np.mean(tr[i - self.atr_period:i])
        return result

    def screen(self):
        data = self.data.copy()
        if len(data) < 60:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (convergence)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
