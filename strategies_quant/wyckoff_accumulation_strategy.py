"""
威科夫累积检测策略 (Wyckoff Accumulation Strategy)
==================================================
检测威科夫累积/派发模式的量化实现。

来源: TradingView "Wyckoff Method"

核心逻辑:
  1. PS(初步支撑): 放量下跌减速
  2. SC(卖出高潮): 恐慌性抛售，巨量
  3. AR(自动反弹): 空头回补
  4. Spring(弹簧): 假跌破SC低点后反弹
  5. SOS(强势信号): 放量突破阻力

技术指标: Volume Analysis, Swing Points, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class WyckoffAccumulationStrategy(BaseStrategy):
    """威科夫累积策略 — 累积/派发模式检测"""

    strategy_description = "威科夫: SC+Spring+SOS累积模式检测"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "window": {"type": "int", "default": 60, "label": "检测窗口"},
        "vol_mult_sc": {"type": "float", "default": 2.0, "label": "SC量能倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.window = params.get('window', 60)
        self.vol_mult_sc = params.get('vol_mult_sc', 2.0)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'window': 60, 'vol_mult_sc': 2.0,
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

        print(f"WyckoffAccumulation: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        if len(data) < self.window + 20:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        vol = data[vol_col].values if vol_col else np.ones(n)

        window_start = max(0, n - self.window)
        window_data = data.iloc[window_start:]

        # SC: Lowest point with high volume in window
        window_low_idx = window_start + np.argmin(low[window_start:])
        sc_low = low[window_low_idx]
        sc_vol = vol[window_low_idx]
        avg_vol = np.mean(vol[window_start:])
        is_sc = sc_vol > avg_vol * self.vol_mult_sc

        # AR: Rally after SC
        after_sc = slice(window_low_idx + 1, n)
        if after_sc.stop > after_sc.start + 2:
            ar_high = np.max(high[after_sc])
        else:
            ar_high = high[-1]

        # Spring: Price dips below SC low then recovers
        is_spring = False
        for i in range(window_low_idx + 1, n):
            if low[i] < sc_low and close[i] > sc_low:
                is_spring = True
                break

        # SOS: Price breaks above AR high with volume
        current_price = close[-1]
        is_sos = current_price > ar_high and vol[-1] > avg_vol

        # Accumulation scoring
        accum_score = sum([is_sc, is_spring, is_sos])
        if accum_score >= 2:
            score += 5
        elif accum_score == 1:
            if is_sos:
                score += 3
            elif is_spring:
                score += 3

        # Distribution detection (mirror image)
        window_high_idx = window_start + np.argmax(high[window_start:])
        dist_high = high[window_high_idx]
        dist_vol = vol[window_high_idx]
        is_dist_climax = dist_vol > avg_vol * self.vol_mult_sc

        after_dist = slice(window_high_idx + 1, n)
        if after_dist.stop > after_dist.start + 2:
            dist_low = np.min(low[after_dist])
        else:
            dist_low = low[-1]

        is_upthrust = False
        for i in range(window_high_idx + 1, n):
            if high[i] > dist_high and close[i] < dist_high:
                is_upthrust = True
                break

        is_sod = current_price < dist_low and vol[-1] > avg_vol

        dist_score = sum([is_dist_climax, is_upthrust, is_sod])
        if dist_score >= 2:
            score -= 5
        elif dist_score == 1:
            if is_sod:
                score -= 3
            elif is_upthrust:
                score -= 3

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
        if len(data) < self.window + 20:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (wyckoff)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
