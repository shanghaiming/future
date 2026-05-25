"""
Delta压力仪表策略 (Delta Pressure Gauge Strategy)
=================================================
实体质量加权Delta振荡器，检测成交量方向性压力。

来源: TradingView "Delta Pressure Gauge [JOAT]"

核心逻辑:
  1. 每根K线成交量按方向和实体质量加权
  2. 全实体K线贡献100%，十字星贡献0%
  3. 滚动累积Delta方向判断
  4. 背离检测价格与Delta的不一致

技术指标: Body-Quality Delta, Cumulative Delta, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class DeltaPressureStrategy(BaseStrategy):
    """Delta压力仪表策略 — 实体加权Delta + 累积压力"""

    strategy_description = "Delta压力: 实体质量加权Delta + 累积压力方向"
    strategy_category = "momentum"
    strategy_params_schema = {
        "lookback": {"type": "int", "default": 30, "label": "回溯周期"},
        "ema_period": {"type": "int", "default": 10, "label": "Delta EMA"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.lookback = params.get('lookback', 30)
        self.ema_period = params.get('ema_period', 10)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'lookback': 30, 'ema_period': 10,
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

        print(f"DeltaPressure: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _calc_body_delta(self, data):
        """Calculate body-quality weighted delta"""
        close = data['close'].values
        open_ = data['open'].values
        high = data['high'].values
        low = data['low'].values
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col is None:
            return np.zeros(len(close))
        vol = data[vol_col].values
        n = len(close)

        deltas = np.zeros(n)
        for i in range(n):
            total_range = high[i] - low[i]
            if total_range == 0:
                deltas[i] = 0
                continue
            body_ratio = abs(close[i] - open_[i]) / total_range
            direction = 1 if close[i] > open_[i] else (-1 if close[i] < open_[i] else 0)
            deltas[i] = direction * body_ratio * vol[i]

        return deltas

    def _evaluate(self, data):
        if len(data) < self.lookback + 10:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # Body-quality weighted delta
        deltas = self._calc_body_delta(data)

        # Cumulative delta over lookback
        cum_delta = np.sum(deltas[-self.lookback:])

        # Normalize by total volume
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col:
            vol = data[vol_col].values
            total_vol = np.sum(vol[-self.lookback:])
            if total_vol > 0:
                norm_delta = cum_delta / total_vol
            else:
                norm_delta = 0
        else:
            norm_delta = 0

        # Delta direction scoring
        if norm_delta > 0.3:
            score += 4
        elif norm_delta > 0.15:
            score += 2
        elif norm_delta > 0.05:
            score += 1
        elif norm_delta < -0.3:
            score -= 4
        elif norm_delta < -0.15:
            score -= 2
        elif norm_delta < -0.05:
            score -= 1

        # Delta trend (is it accelerating?)
        half = self.lookback // 2
        if half > 5:
            recent_delta = np.sum(deltas[-half:])
            older_delta = np.sum(deltas[-self.lookback:-half])
            if vol_col:
                vol = data[vol_col].values
                recent_vol = np.sum(vol[-half:])
                older_vol = np.sum(vol[-self.lookback:-half])
                if recent_vol > 0 and older_vol > 0:
                    recent_norm = recent_delta / recent_vol
                    older_norm = older_delta / older_vol
                    if recent_norm > older_norm and score > 0:
                        score += 2  # Accelerating bullish pressure
                    elif recent_norm < older_norm and score < 0:
                        score += 2  # Accelerating bearish pressure

        # Divergence: price down but delta up (bullish) or vice versa
        if n >= 10:
            price_trend = close[-1] - close[-10]
            delta_trend = np.sum(deltas[-10:])
            if price_trend < 0 and delta_trend > 0:
                score += 2  # Bullish divergence
            elif price_trend > 0 and delta_trend < 0:
                score -= 2  # Bearish divergence

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
                'reason': f"score={score} (delta_pressure)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
