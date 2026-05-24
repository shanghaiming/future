"""
Aegis HMA哨兵策略 (Aegis HMA Sentinel Strategy)
=================================================
Hull MA多层过滤趋势追踪，成交量+K线动量验证。

来源: TradingView "Aegis HMA Sentinel [wjdtks255]"

核心逻辑:
  1. HMA(Hull MA)方向作为核心趋势信号
  2. 长期基线守护(阻止熊市做多/牛市做空)
  3. 成交量过滤(1.5x均值检测机构参与)
  4. K线动量验证(大实体确认方向)
  5. 全部通过才发出信号

技术指标: HMA, Long-term MA, Volume, Candle Momentum
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class AegisHmaSentinelStrategy(BaseStrategy):
    """Aegis HMA哨兵策略 — HMA + 多层过滤"""

    strategy_description = "HMA哨兵: Hull MA + 长期基线守护 + 量能过滤"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "hma_period": {"type": "int", "default": 20, "label": "HMA周期"},
        "baseline_period": {"type": "int", "default": 100, "label": "基线MA周期"},
        "vol_mult": {"type": "float", "default": 1.5, "label": "量能过滤倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.hma_period = params.get('hma_period', 20)
        self.baseline_period = params.get('baseline_period', 100)
        self.vol_mult = params.get('vol_mult', 1.5)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'hma_period': 20, 'baseline_period': 100, 'vol_mult': 1.5,
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

        print(f"AegisHMA: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _calc_wma(self, values, period):
        """Weighted Moving Average"""
        n = len(values)
        result = np.empty(n)
        weights = np.arange(1, period + 1, dtype=float)
        weights = weights / weights.sum()
        for i in range(n):
            if i < period - 1:
                w = np.arange(1, i + 2, dtype=float)
                w = w / w.sum()
                result[i] = np.dot(values[:i + 1], w)
            else:
                result[i] = np.dot(values[i - period + 1:i + 1], weights)
        return result

    def _calc_hma(self, values, period):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        half_period = max(int(period / 2), 2)
        sqrt_period = max(int(np.sqrt(period)), 2)
        wma_half = self._calc_wma(values, half_period)
        wma_full = self._calc_wma(values, period)
        diff = 2 * wma_half - wma_full
        return self._calc_wma(diff, sqrt_period)

    def _evaluate(self, data):
        min_len = max(self.hma_period, self.baseline_period) + 20
        if len(data) < min_len:
            return None

        close = data['close'].values
        open_ = data['open'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        # HMA direction
        hma = self._calc_hma(close, self.hma_period)
        if hma[-1] > hma[-2]:
            score += 3
        elif hma[-1] < hma[-2]:
            score -= 3

        # Baseline guardian: long-term MA
        baseline = np.mean(close[-self.baseline_period:])
        if close[-1] > baseline:
            if score > 0:
                score += 2  # Bullish above baseline
            else:
                score -= 1  # Sell below baseline is ok but less strong
        else:
            if score < 0:
                score += 2  # Bearish below baseline
            else:
                score -= 1

        # Volume filter
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * self.vol_mult:
                if score > 0:
                    score += 2
                elif score < 0:
                    score += 2

        # Candle momentum (large body confirmation)
        body = abs(close[-1] - open_[-1])
        total_range = high[-1] - low[-1]
        if total_range > 0:
            body_ratio = body / total_range
            is_bullish_candle = close[-1] > open_[-1]
            if body_ratio > 0.6:
                if is_bullish_candle and score > 0:
                    score += 2
                elif not is_bullish_candle and score < 0:
                    score += 2

        # Price vs HMA position
        if close[-1] > hma[-1]:
            score += 1
        else:
            score -= 1

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
        if len(data) < max(self.hma_period, self.baseline_period) + 20:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (aegis_hma)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
