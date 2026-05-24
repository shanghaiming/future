"""
Pocket Pivot策略 (Pocket Pivot Strategy)
========================================
检测基底构建期间的大成交量上涨日(机构收集行为)。

来源: TradingView "Advanced Simple Volume With Pocket Pivots"

核心逻辑:
  1. 价格在基底/回调中
  2. 出现成交量大于前10日任何上涨日的上涨日
  3. 收盘价在50日均线之上或附近
  4. 机构收集信号

技术指标: Volume, MA(50), Base Detection
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class PocketPivotStrategy(BaseStrategy):
    """Pocket Pivot策略 — 成交量异常放大+基底检测"""

    strategy_description = "Pocket Pivot: 基底期大成交量突破检测"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "ma_period": {"type": "int", "default": 50, "label": "趋势MA"},
        "vol_lookback": {"type": "int", "default": 10, "label": "量能回溯"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ma_period = params.get('ma_period', 50)
        self.vol_lookback = params.get('vol_lookback', 10)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'ma_period': 50, 'vol_lookback': 10,
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

        print(f"PocketPivot: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        if len(data) < self.ma_period + self.vol_lookback + 10:
            return None

        close = data['close'].values
        open_ = data['open'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col is None:
            return None
        vol = data[vol_col].values

        # MA50 trend
        ma50 = np.mean(close[-self.ma_period:])

        # Current bar characteristics
        is_up_bar = close[-1] > open_[-1]
        current_vol = vol[-1]

        # Bullish Pocket Pivot detection
        # Volume must exceed every up-day volume in the lookback period
        max_up_vol = 0
        for i in range(n - self.vol_lookback - 1, n - 1):
            if close[i] > open_[i]:  # Up day
                max_up_vol = max(max_up_vol, vol[i])

        if is_up_bar and current_vol > max_up_vol and max_up_vol > 0:
            score += 5  # Strong pocket pivot
        elif is_up_bar and current_vol > max_up_vol * 0.9:
            score += 3  # Near pocket pivot

        # Price relative to MA50
        if close[-1] > ma50 * 0.95:  # Within 5% of MA50
            if score > 0:
                score += 2  # Pivot near support
        elif close[-1] > ma50:
            if score > 0:
                score += 1  # Above MA

        # Base detection: price within range for last N bars
        range_high = np.max(high[-20:])
        range_low = np.min(low[-20:])
        range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0
        if range_pct < 15:  # Tight range = base
            if score > 0:
                score += 2

        # Bearish Pocket Pivot (mirror)
        is_down_bar = close[-1] < open_[-1]
        max_down_vol = 0
        for i in range(n - self.vol_lookback - 1, n - 1):
            if close[i] < open_[i]:
                max_down_vol = max(max_down_vol, vol[i])

        if is_down_bar and current_vol > max_down_vol and max_down_vol > 0:
            score -= 5

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
        if len(data) < 70:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (pocket_pivot)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
