"""
机构偏向引擎策略 (Institutional Bias Engine Strategy)
=====================================================
聚合多个机构足迹信号，判断聪明资金方向。

来源: TradingView "Institutional Bias Engine [Confirmed]"

核心逻辑:
  1. Delta Volume(净买卖量)方向
  2. 大实体K线过滤(机构行为)
  3. VWAP偏离度
  4. 订单流不平衡
  5. 多信号加权聚合评分

技术指标: Delta Volume, VWAP, Volume Analysis, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class InstitutionalBiasStrategy(BaseStrategy):
    """机构偏向引擎策略 — 多信号机构足迹聚合"""

    strategy_description = "机构偏向: Delta Volume + VWAP偏离 + 订单流不平衡"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "lookback": {"type": "int", "default": 20, "label": "回溯周期"},
        "vwap_period": {"type": "int", "default": 20, "label": "VWAP周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.lookback = params.get('lookback', 20)
        self.vwap_period = params.get('vwap_period', 20)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'lookback': 20, 'vwap_period': 20,
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

        print(f"InstitutionalBias: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _calc_delta_volume(self, data):
        """Approximate delta volume from OHLCV"""
        close = data['close'].values
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col is None:
            return 0
        vol = data[vol_col].values
        n = len(vol)
        if n < self.lookback:
            return 0
        # Approximate: up-close bars = buy volume, down-close bars = sell volume
        buy_vol = np.where(close[-self.lookback:] > np.roll(close, 1)[-self.lookback:],
                           vol[-self.lookback:], 0)
        sell_vol = np.where(close[-self.lookback:] < np.roll(close, 1)[-self.lookback:],
                            vol[-self.lookback:], 0)
        delta = np.sum(buy_vol) - np.sum(sell_vol)
        total = np.sum(vol[-self.lookback:])
        if total == 0:
            return 0
        return delta / total  # Normalized -1 to +1

    def _calc_vwap_deviation(self, data):
        """Calculate VWAP and price deviation"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col is None:
            return 0
        vol = data[vol_col].values
        n = len(vol)
        period = min(self.vwap_period, n)
        if period < 2:
            return 0
        tp = (high + low + close) / 3.0
        vwap = np.sum(tp[-period:] * vol[-period:]) / np.sum(vol[-period:])
        if vwap == 0:
            return 0
        return (close[-1] - vwap) / vwap * 100  # Percentage deviation

    def _detect_large_bars(self, data):
        """Detect institutional candle patterns"""
        close = data['close'].values
        open_ = data['open'].values
        high = data['high'].values
        low = data['low'].values
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col is None:
            return 0
        vol = data[vol_col].values
        n = len(vol)
        if n < 20:
            return 0

        body = np.abs(close[-1] - open_[-1])
        total_range = high[-1] - low[-1]
        vol_ma = np.mean(vol[-20:])

        if total_range == 0 or vol_ma == 0:
            return 0

        body_ratio = body / total_range
        vol_ratio = vol[-1] / vol_ma

        # Large bullish candle
        if close[-1] > open_[-1] and body_ratio > 0.6 and vol_ratio > 1.5:
            return 1
        # Large bearish candle
        elif close[-1] < open_[-1] and body_ratio > 0.6 and vol_ratio > 1.5:
            return -1
        return 0

    def _evaluate(self, data):
        if len(data) < 30:
            return None

        score = 0

        # Signal 1: Delta Volume direction
        delta = self._calc_delta_volume(data)
        if delta > 0.15:
            score += 3
        elif delta > 0.05:
            score += 1
        elif delta < -0.15:
            score -= 3
        elif delta < -0.05:
            score -= 1

        # Signal 2: VWAP deviation
        vwap_dev = self._calc_vwap_deviation(data)
        if vwap_dev > 1:
            score += 2
        elif vwap_dev > 0.3:
            score += 1
        elif vwap_dev < -1:
            score -= 2
        elif vwap_dev < -0.3:
            score -= 1

        # Signal 3: Large bar detection (institutional)
        large_bar = self._detect_large_bars(data)
        if large_bar == 1:
            score += 2
        elif large_bar == -1:
            score -= 2

        # Signal 4: Order flow imbalance (consecutive direction bars)
        close = data['close'].values
        open_ = data['open'].values
        n = len(close)
        if n >= 5:
            up_bars = np.sum(close[-5:] > open_[-5:])
            down_bars = 5 - up_bars
            if up_bars >= 4:
                score += 2
            elif down_bars >= 4:
                score -= 2

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
        if len(data) < 40:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (institutional_bias)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
