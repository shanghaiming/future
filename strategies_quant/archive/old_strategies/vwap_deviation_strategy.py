"""
VWAP偏差策略 (VWAP Deviation Strategy)
======================================
基于VWAP(成交量加权平均价)及其标准差带的均值回归策略。

来源: TradingView "Statistical VWAP Study: Session and RTH VWAP"

核心逻辑:
  1. 计算滚动VWAP
  2. 计算VWAP的标准差带(类似BB但基于VWAP)
  3. 价格触及下轨 = 超卖买入
  4. 价格触及上轨 = 超买卖出
  5. 价格回归VWAP获利

技术指标: VWAP, 标准差带
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class VwapDeviationStrategy(BaseStrategy):
    """VWAP偏差策略 — VWAP标准差带均值回归"""

    strategy_description = "VWAP偏差: VWAP标准差带 + 均值回归"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "vwap_period": {"type": "int", "default": 20, "label": "VWAP周期"},
        "band_mult": {"type": "float", "default": 2.0, "label": "标准差倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.vwap_period = params.get('vwap_period', 20)
        self.band_mult = params.get('band_mult', 2.0)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 2)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.0)

    def get_default_params(self):
        return {
            'vwap_period': 20, 'band_mult': 2.0,
            'atr_period': 14, 'hold_min': 2, 'trail_atr_mult': 2.0,
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

                    if days_held >= 30:
                        should_exit = True

                    # Check if price reverted to VWAP
                    vwap = self._calc_vwap(hist)
                    if vwap is not None:
                        if position_dir == 1 and current_price >= vwap:
                            should_exit = True
                        elif position_dir == -1 and current_price <= vwap:
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

        print(f"VwapDeviation: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        if len(data) < self.vwap_period + 10:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # Calculate VWAP
        vwap = self._calc_vwap(data)
        if vwap is None:
            return None

        # Calculate standard deviation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col:
            typ_price = (data['high'].values + data['low'].values + close) / 3.0
            recent_typ = typ_price[-self.vwap_period:]
            std = np.std(recent_typ, ddof=1)

            upper_band = vwap + self.band_mult * std
            lower_band = vwap - self.band_mult * std

            current = close[-1]

            # Below lower band = oversold
            if current < lower_band:
                score += 5
            elif current < vwap - std:
                score += 3
            # Above upper band = overbought
            elif current > upper_band:
                score -= 5
            elif current > vwap + std:
                score -= 3

            # Distance from VWAP as percentage
            vwap_dist = (current - vwap) / vwap * 100
            if vwap_dist < -2:
                score += 2
            elif vwap_dist > 2:
                score -= 2

        else:
            # Fallback: use SMA as VWAP proxy
            sma = np.mean(close[-self.vwap_period:])
            std = np.std(close[-self.vwap_period:], ddof=1)
            upper = sma + self.band_mult * std
            lower = sma - self.band_mult * std

            if close[-1] < lower:
                score += 5
            elif close[-1] > upper:
                score -= 5

        if score == 0:
            return None
        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_vwap(self, data):
        if len(data) < self.vwap_period:
            return None
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        typ_price = (high + low + close) / 3.0

        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col:
            vol = data[vol_col].values
            total_vol = np.sum(vol[-self.vwap_period:])
            if total_vol == 0:
                return None
            return np.sum(typ_price[-self.vwap_period:] * vol[-self.vwap_period:]) / total_vol
        else:
            return np.mean(typ_price[-self.vwap_period:])

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
                'reason': f"score={score} (vwap_dev)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
