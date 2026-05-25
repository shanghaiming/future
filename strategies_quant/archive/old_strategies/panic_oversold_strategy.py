"""
恐慌超卖策略 (Panic Oversold & Volume Capitulation Strategy)
==========================================================
极端超卖 + 放量投降检测，捕捉恐慌性抛售后的反转。

来源: TradingView "Panic Oversold & Volume Capitulation v3"

核心逻辑:
  1. RSI(14) < 25 极端超卖
  2. RSI(6) < 15 加速超卖
  3. 成交量 > 2x 均量(恐慌放量)
  4. Stochastic K < 10 极端超卖确认
  5. 多维度超卖同时触发 = 恐慌投降 → 反弹做多

技术指标: RSI(6,14), Stochastic, Volume Ratio
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class PanicOversoldStrategy(BaseStrategy):
    """恐慌超卖策略 — 多维极端超卖 + 量能投降检测"""

    strategy_description = "恐慌超卖: RSI极端 + Stoch极端 + 放量投降反转"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_fast": {"type": "int", "default": 6, "label": "快速RSI"},
        "rsi_panic": {"type": "int", "default": 25, "label": "RSI恐慌阈值"},
        "rsi_extreme": {"type": "int", "default": 15, "label": "RSI极端阈值"},
        "stoch_period": {"type": "int", "default": 14, "label": "Stoch周期"},
        "stoch_panic": {"type": "int", "default": 10, "label": "Stoch恐慌阈值"},
        "vol_mult": {"type": "float", "default": 2.0, "label": "放量倍数"},
        "vol_ma_period": {"type": "int", "default": 20, "label": "量均线周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_fast = params.get('rsi_fast', 6)
        self.rsi_panic = params.get('rsi_panic', 25)
        self.rsi_extreme = params.get('rsi_extreme', 15)
        self.stoch_period = params.get('stoch_period', 14)
        self.stoch_panic = params.get('stoch_panic', 10)
        self.vol_mult = params.get('vol_mult', 2.0)
        self.vol_ma_period = params.get('vol_ma_period', 20)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'rsi_period': 14, 'rsi_fast': 6, 'rsi_panic': 25, 'rsi_extreme': 15,
            'stoch_period': 14, 'stoch_panic': 10, 'vol_mult': 2.0, 'vol_ma_period': 20,
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

                    if days_held >= 30:
                        should_exit = True

                    if not should_exit:
                        result = self._evaluate(hist)
                        if result is not None:
                            score, direction, _ = result
                            # Exit if RSI recovers from oversold
                            if position_dir == 1 and score < -2:
                                should_exit = True
                            elif position_dir == -1 and score > 2:
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

        print(f"PanicOversold: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.rsi_period, self.stoch_period, self.vol_ma_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        # 1. RSI(14) - panic oversold
        rsi14 = self._calc_rsi(close, self.rsi_period)
        if rsi14 < self.rsi_extreme:
            score += 5  # Extreme oversold → strong buy
        elif rsi14 < self.rsi_panic:
            score += 3
        elif rsi14 > 100 - self.rsi_extreme:
            score -= 5
        elif rsi14 > 100 - self.rsi_panic:
            score -= 3

        # 2. RSI(6) - fast confirmation
        rsi6 = self._calc_rsi(close, self.rsi_fast)
        if rsi6 < self.rsi_extreme:
            score += 3
        elif rsi6 > 100 - self.rsi_extreme:
            score -= 3

        # 3. Stochastic K
        stoch_k = self._calc_stoch_k(close, high, low)
        if stoch_k < self.stoch_panic:
            score += 3
        elif stoch_k > 100 - self.stoch_panic:
            score -= 3

        # 4. Volume capitulation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= self.vol_ma_period:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-self.vol_ma_period:])
            vol_ratio = vol[-1] / vol_ma if vol_ma > 0 else 1.0
            if vol_ratio > self.vol_mult:
                # High volume + oversold = capitulation (buy signal)
                if score > 0:
                    score += 3
                elif score < 0:
                    score -= 3

        if score == 0:
            return None
        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_rsi(self, close, period):
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        return 100.0 - 100.0 / (1 + avg_gain / avg_loss)

    def _calc_stoch_k(self, close, high, low):
        if len(close) < self.stoch_period:
            return 50.0
        lowest = np.min(low[-self.stoch_period:])
        highest = np.max(high[-self.stoch_period:])
        if highest == lowest:
            return 50.0
        return (close[-1] - lowest) / (highest - lowest) * 100.0

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
            return {'action': 'hold', 'reason': '非恐慌区', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (panic)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
