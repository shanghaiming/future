"""
OBV背离策略 (OBV Divergence Strategy)
=====================================
检测价格与OBV(能量潮)之间的背离，预判趋势反转。

来源: TradingView "OBV Divergence Finder - Jeff 2026"

核心逻辑:
  1. 计算OBV(On Balance Volume)
  2. 价格创新高但OBV未创新高 → 看跌背离 → 做空
  3. 价格创新低但OBV未创新低 → 看涨背离 → 做多
  4. 结合RSI确认 + ATR止损

技术指标: OBV, RSI, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class ObvDivergenceStrategy(BaseStrategy):
    """OBV背离策略 — 价量背离检测 + RSI确认"""

    strategy_description = "OBV背离: 价量背离反转 + RSI确认 + ATR止损"
    strategy_category = "volume"
    strategy_params_schema = {
        "obv_ma_period": {"type": "int", "default": 20, "label": "OBV均线周期"},
        "lookback": {"type": "int", "default": 20, "label": "背离回望周期"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_ob": {"type": "int", "default": 70, "label": "RSI超买"},
        "rsi_os": {"type": "int", "default": 30, "label": "RSI超卖"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.obv_ma_period = params.get('obv_ma_period', 20)
        self.lookback = params.get('lookback', 20)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_ob = params.get('rsi_ob', 70)
        self.rsi_os = params.get('rsi_os', 30)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'obv_ma_period': 20, 'lookback': 20,
            'rsi_period': 14, 'rsi_ob': 70, 'rsi_os': 30,
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

        print(f"ObvDivergence: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = self.lookback + 20
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # 1. Calculate OBV
        obv = self._calc_obv(data)
        if obv is None:
            return None

        # 2. Detect divergence
        div = self._detect_divergence(close, obv, min(self.lookback, n - 1))

        if div == 'bullish':
            score += 5
        elif div == 'bearish':
            score -= 5

        # 3. OBV trend
        obv_ma = np.mean(obv[-self.obv_ma_period:])
        if obv[-1] > obv_ma:
            score += 2
        elif obv[-1] < obv_ma:
            score -= 2

        # 4. RSI confirmation
        rsi = self._calc_rsi(close)
        if score > 0 and rsi < 50:
            score += 1  # Oversold confirmation for bullish div
        elif score < 0 and rsi > 50:
            score -= 1  # Overbought confirmation for bearish div

        if score == 0:
            return None
        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_obv(self, data):
        close = data['close'].values
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if not vol_col:
            return None
        vol = data[vol_col].values
        if len(close) < 2:
            return None

        obv = np.zeros(len(close))
        obv[0] = vol[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + vol[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - vol[i]
            else:
                obv[i] = obv[i-1]
        return obv

    def _detect_divergence(self, price, obv, lookback):
        if lookback < 5:
            return None
        p = price[-lookback:]
        o = obv[-lookback:]

        # Find peaks and troughs
        p_highs, o_at_highs = [], []
        p_lows, o_at_lows = [], []

        for i in range(1, len(p) - 1):
            if p[i] > p[i-1] and p[i] > p[i+1]:
                p_highs.append(p[i])
                o_at_highs.append(o[i])
            elif p[i] < p[i-1] and p[i] < p[i+1]:
                p_lows.append(p[i])
                o_at_lows.append(o[i])

        # Bearish: price higher high, OBV lower high
        if len(p_highs) >= 2:
            if p_highs[-1] > p_highs[-2] and o_at_highs[-1] < o_at_highs[-2]:
                return 'bearish'

        # Bullish: price lower low, OBV higher low
        if len(p_lows) >= 2:
            if p_lows[-1] < p_lows[-2] and o_at_lows[-1] > o_at_lows[-2]:
                return 'bullish'

        return None

    def _calc_rsi(self, close):
        if len(close) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        if avg_loss == 0:
            return 100.0
        return 100.0 - 100.0 / (1 + avg_gain / avg_loss)

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
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无背离信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (obv_div)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
