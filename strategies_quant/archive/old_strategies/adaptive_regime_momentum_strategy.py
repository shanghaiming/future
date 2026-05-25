"""
自适应体制动量策略 (Adaptive Regime Momentum Strategy)
======================================================
三层过滤: ComboMA斜率 + 价格位置 + Volume RSI

来源: TradingView "Adaptive Regime Momentum [JOAT]"

核心逻辑:
  1. ALMA+ZLMA融合均线(ComboMA)方向确认
  2. 价格与ComboMA位置关系
  3. Volume RSI需求过滤
  4. RSI(14) > 50 作为多头门控
  5. 多根K线连续斜率确认避免噪音

技术指标: ALMA, ZLMA, Volume RSI, RSI, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class AdaptiveRegimeMomentumStrategy(BaseStrategy):
    """自适应体制动量策略 — ComboMA + Volume RSI + 三层过滤"""

    strategy_description = "自适应动量: ComboMA(ALMA+ZLMA) + Volume RSI + RSI门控"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "combo_period": {"type": "int", "default": 21, "label": "ComboMA周期"},
        "slope_bars": {"type": "int", "default": 3, "label": "斜率确认K线数"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "vol_rsi_period": {"type": "int", "default": 14, "label": "Volume RSI周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.combo_period = params.get('combo_period', 21)
        self.slope_bars = params.get('slope_bars', 3)
        self.rsi_period = params.get('rsi_period', 14)
        self.vol_rsi_period = params.get('vol_rsi_period', 14)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'combo_period': 21, 'slope_bars': 3, 'rsi_period': 14,
            'vol_rsi_period': 14, 'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"AdaptiveRegimeMomentum: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _calc_alma(self, values, period):
        """ALMA (Arnaud Legoux Moving Average)"""
        n = len(values)
        result = np.empty(n)
        offset = 0.85
        sigma = 6.0
        m = offset * (period - 1)
        s = period / sigma
        weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
        weights = weights / weights.sum()

        for i in range(n):
            if i < period - 1:
                result[i] = np.mean(values[:i + 1])
            else:
                result[i] = np.dot(values[i - period + 1:i + 1], weights)
        return result

    def _calc_zlma(self, values, period):
        """Zero-Lag MA: MA of (values + (values - MA(values)))"""
        ema = self._calc_ema(values, period)
        lag = values - ema
        return self._calc_ema(values + lag, period)

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

    def _calc_rsi(self, values, period):
        n = len(values)
        if n < period + 1:
            return np.full(n, 50.0)
        deltas = np.diff(values)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        result = np.empty(n)
        result[:period] = 50.0
        for i in range(period, n):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            if avg_loss == 0:
                result[i] = 100
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - 100 / (1 + rs)
        return result

    def _calc_volume_rsi(self, data):
        """Volume-weighted RSI"""
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col is None:
            return None
        vol = data[vol_col].values
        close = data['close'].values
        n = len(vol)
        if n < self.vol_rsi_period + 1:
            return None
        # Up volume vs down volume
        up_vol = np.where(close[1:] > close[:-1], vol[1:], 0)
        down_vol = np.where(close[1:] < close[:-1], vol[1:], 0)
        avg_up = np.mean(up_vol[:self.vol_rsi_period])
        avg_down = np.mean(down_vol[:self.vol_rsi_period])
        if avg_down == 0:
            return 100.0
        rs = avg_up / avg_down
        return 100 - 100 / (1 + rs)

    def _evaluate(self, data):
        if len(data) < self.combo_period + self.slope_bars + 20:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # ComboMA = (ALMA + ZLMA) / 2
        alma = self._calc_alma(close, self.combo_period)
        zlma = self._calc_zlma(close, self.combo_period)
        combo = (alma + zlma) / 2.0

        # Layer 1: Multi-bar slope confirmation
        slope_up = 0
        slope_down = 0
        for i in range(1, self.slope_bars + 1):
            if combo[-i] > combo[-i - 1]:
                slope_up += 1
            elif combo[-i] < combo[-i - 1]:
                slope_down += 1

        if slope_up >= self.slope_bars:
            score += 4
        elif slope_up >= self.slope_bars - 1:
            score += 2
        if slope_down >= self.slope_bars:
            score -= 4
        elif slope_down >= self.slope_bars - 1:
            score -= 2

        # Layer 2: Price vs ComboMA position
        if close[-1] > combo[-1]:
            score += 2
        elif close[-1] < combo[-1]:
            score -= 2

        # Layer 3: Volume RSI demand filter
        vol_rsi = self._calc_volume_rsi(data)
        if vol_rsi is not None:
            if score > 0 and vol_rsi > 50:
                score += 2
            elif score < 0 and vol_rsi < 50:
                score += 2
            elif score > 0 and vol_rsi < 50:
                score -= 1
            elif score < 0 and vol_rsi > 50:
                score -= 1

        # RSI(14) gate
        rsi = self._calc_rsi(close, self.rsi_period)
        if rsi[-1] > 50:
            if score > 0:
                score += 1
        elif rsi[-1] < 50:
            if score < 0:
                score += 1

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
                'reason': f"score={score} (adaptive_regime)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
