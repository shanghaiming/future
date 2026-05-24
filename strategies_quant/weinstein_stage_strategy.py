"""
Weinstein阶段策略 (Weinstein Stage Strategy)
=============================================
Stan Weinstein四阶段市场分类系统。

来源: TradingView "EvolveX Weinstein Stages v2"

核心逻辑:
  1. 30周WMA判断长期趋势方向
  2. 阶段1: 底部筑底(均线走平) → 准备
  3. 阶段2: 上升趋势(价格+均线向上) → 做多
  4. 阶段3: 顶部派发(均线走平) → 谨慎
  5. 阶段4: 下降趋势(价格+均线向下) → 做空或观望

技术指标: 30 WMA, 价格vs均线, 均线斜率
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class WeinsteinStageStrategy(BaseStrategy):
    """Weinstein阶段策略 — 四阶段市场分类 + 趋势交易"""

    strategy_description = "Weinstein: 四阶段分类 + 趋势跟随"
    strategy_category = "regime"
    strategy_params_schema = {
        "wma_period": {"type": "int", "default": 30, "label": "WMA周期"},
        "slope_period": {"type": "int", "default": 10, "label": "斜率周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 5, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.wma_period = params.get('wma_period', 30)
        self.slope_period = params.get('slope_period', 10)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 5)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'wma_period': 30, 'slope_period': 10,
            'atr_period': 14, 'hold_min': 5, 'trail_atr_mult': 2.5,
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

                    # Stage-based exit
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

        print(f"WeinsteinStage: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        min_len = self.wma_period + self.slope_period + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # 1. WMA
        wma = self._calc_wma_series(close, self.wma_period)
        if wma is None:
            return None

        # 2. WMA slope
        slope_lookback = min(self.slope_period, len(wma) - 1)
        wma_slope = (wma[-1] - wma[-1-slope_lookback]) / slope_lookback
        slope_pct = wma_slope / wma[-1] * 100 if wma[-1] > 0 else 0

        # 3. Stage classification
        above_wma = close[-1] > wma[-1]

        if slope_pct > 0.05:
            if above_wma:
                stage = 2  # Advancing
                score += 5
            else:
                stage = 2  # Still advancing but below MA
                score += 2
        elif slope_pct < -0.05:
            if not above_wma:
                stage = 4  # Declining
                score -= 5
            else:
                stage = 4
                score -= 2
        else:
            if above_wma:
                stage = 1  # Basing (bottom)
                score += 1
            else:
                stage = 3  # Top
                score -= 1

        # 4. Price momentum confirmation
        if n >= 20:
            ret_20 = (close[-1] / close[-20] - 1) * 100
            if score > 0 and ret_20 > 5:
                score += 2
            elif score < 0 and ret_20 < -5:
                score -= 2

        # 5. Volume trend
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_trend = (np.mean(vol[-5:]) / np.mean(vol[-20:]) - 1) * 100
            if score > 0 and vol_trend > 10:
                score += 1
            elif score < 0 and vol_trend > 10:
                score -= 1

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_wma_series(self, values, period):
        values = np.asarray(values, dtype=float)
        n = len(values)
        if n < period:
            return None
        result = np.empty(n)
        weights = np.arange(1, period + 1, dtype=float)
        weight_sum = np.sum(weights)

        for i in range(n):
            if i < period - 1:
                w = np.arange(1, i + 2, dtype=float)
                result[i] = np.sum(values[:i+1] * w) / np.sum(w)
            else:
                result[i] = np.sum(values[i-period+1:i+1] * weights) / weight_sum
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
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (weinstein)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
