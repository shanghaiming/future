"""
CCI体制振荡器策略 (CCI Regime Oscillator Strategy)
==================================================
CCI偏离历史趋势的程度判断超买超卖，早期买卖信号。

来源: TradingView "CCI Regime Oscillator"

核心逻辑:
  1. CCI(20)偏离零线程度
  2. CCI > +100 = 超买区
  3. CCI < -100 = 超卖区
  4. CCI从极端区回归 = 反转信号
  5. 结合趋势过滤器避免逆势

技术指标: CCI, EMA趋势过滤
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class CciRegimeStrategy(BaseStrategy):
    """CCI体制振荡器策略 — CCI极端区反转 + 趋势过滤"""

    strategy_description = "CCI体制: CCI极端区反转 + EMA趋势过滤"
    strategy_category = "momentum"
    strategy_params_schema = {
        "cci_period": {"type": "int", "default": 20, "label": "CCI周期"},
        "cci_extreme": {"type": "int", "default": 150, "label": "CCI极端阈值"},
        "cci_ob": {"type": "int", "default": 100, "label": "CCI超买"},
        "cci_os": {"type": "int", "default": -100, "label": "CCI超卖"},
        "ema_period": {"type": "int", "default": 50, "label": "趋势EMA"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.cci_period = params.get('cci_period', 20)
        self.cci_extreme = params.get('cci_extreme', 150)
        self.cci_ob = params.get('cci_ob', 100)
        self.cci_os = params.get('cci_os', -100)
        self.ema_period = params.get('ema_period', 50)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'cci_period': 20, 'cci_extreme': 150, 'cci_ob': 100, 'cci_os': -100,
            'ema_period': 50, 'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"CciRegime: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = max(self.cci_period, self.ema_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        # 1. CCI calculation
        cci = self._calc_cci(close, high, low)
        cci_series = self._calc_cci_series(close, high, low)

        # CCI extreme zones
        if cci > self.cci_extreme:
            score -= 4  # Extreme overbought → sell
        elif cci > self.cci_ob:
            score -= 2
        elif cci < -self.cci_extreme:
            score += 4  # Extreme oversold → buy
        elif cci < self.cci_os:
            score += 2

        # CCI crossing back from extreme
        if cci_series is not None and len(cci_series) >= 2:
            prev_cci = cci_series[-2]
            if prev_cci < self.cci_os and cci > self.cci_os:
                score += 3  # Crossed up from oversold
            elif prev_cci > self.cci_ob and cci < self.cci_ob:
                score -= 3  # Crossed down from overbought

        # 2. EMA trend filter
        ema = np.mean(close[-self.ema_period:])
        if close[-1] > ema:
            score += 1  # Uptrend bias
        else:
            score -= 1

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_cci(self, close, high, low):
        tp = (high + low + close) / 3.0
        period = min(self.cci_period, len(tp))
        sma = np.mean(tp[-period:])
        mad = np.mean(np.abs(tp[-period:] - sma))
        if mad == 0:
            return 0
        return (tp[-1] - sma) / (0.015 * mad)

    def _calc_cci_series(self, close, high, low):
        tp = (high + low + close) / 3.0
        n = len(tp)
        if n < self.cci_period:
            return None
        result = np.zeros(n)
        for i in range(self.cci_period - 1, n):
            sma = np.mean(tp[i-self.cci_period+1:i+1])
            mad = np.mean(np.abs(tp[i-self.cci_period+1:i+1] - sma))
            if mad == 0:
                result[i] = 0
            else:
                result[i] = (tp[i] - sma) / (0.015 * mad)
        return result

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
                'reason': f"score={score} (cci_regime)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
