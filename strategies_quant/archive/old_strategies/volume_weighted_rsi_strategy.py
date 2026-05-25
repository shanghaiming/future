"""
成交量加权RSI动量策略 (Volume-Weighted RSI Momentum Strategy)
=============================================================
在RSI公式之前用成交量缩放价格行为, 创建高灵敏度动量振荡器。

来源: TradingView "Inflow/Outflow Index (IOI) — Volume-Weighted RSI"

核心逻辑:
  传统RSI只用价格变化, IOI在RSI公式前先用成交量加权:
  1. 买入压力 = max(close-change, 0) * volume
  2. 卖出压力 = max(-close-change, 0) * volume
  3. EMA平滑后用RSI公式组合为0-100
  4. 极端区(>80/<20) = 高敏感度资金流入/流出转折

  结合MACD确认方向, ATR止损控制风险

技术指标: Volume-Weighted RSI, MACD, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class VolumeWeightedRSIStrategy(BaseStrategy):
    """成交量加权RSI动量策略 — VW-RSI + MACD确认 + ATR止损"""

    strategy_description = "VW-RSI: 成交量加权RSI + MACD确认 + ATR止损"
    strategy_category = "momentum"
    strategy_params_schema = {
        "vwrsi_period": {"type": "int", "default": 14, "label": "VW-RSI周期"},
        "ema_period": {"type": "int", "default": 14, "label": "EMA平滑周期"},
        "ob_thresh": {"type": "int", "default": 75, "label": "超买阈值"},
        "os_thresh": {"type": "int", "default": 25, "label": "超卖阈值"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.vwrsi_period = params.get('vwrsi_period', 14)
        self.ema_period = params.get('ema_period', 14)
        self.ob_thresh = params.get('ob_thresh', 75)
        self.os_thresh = params.get('os_thresh', 25)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 2)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'vwrsi_period': 14, 'ema_period': 14,
            'ob_thresh': 75, 'os_thresh': 25,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'atr_period': 14, 'hold_min': 2, 'trail_atr_mult': 2.5,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
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

                if best_sym and abs(best_score) >= 4:
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

                    # ATR trailing stop
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # Max hold
                    if days_held >= 60:
                        should_exit = True

                    # Signal exit
                    result = self._evaluate(hist)
                    if result is not None and not should_exit:
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

        print(f"VolumeWeightedRSI: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """综合评估: VW-RSI + MACD"""
        min_len = max(self.vwrsi_period + 1, self.macd_slow + self.macd_signal) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        n = len(close)

        # ===== 1. Volume-Weighted RSI =====
        vwrsi = self._calc_vwrsi(data)
        score = 0

        # VW-RSI scoring (weight 5)
        if vwrsi < self.os_thresh:
            score += 5  # Oversold → buy
        elif vwrsi < 40:
            score += 2
        elif vwrsi > self.ob_thresh:
            score -= 5  # Overbought → sell
        elif vwrsi > 60:
            score -= 2

        # VW-RSI crossing back from extreme (weight 3)
        if n >= 2:
            prev_vwrsi = self._calc_vwrsi_history(data, n - 1)
            if prev_vwrsi is not None:
                # Crossing up from oversold
                if prev_vwrsi < self.os_thresh and vwrsi > self.os_thresh:
                    score += 3
                # Crossing down from overbought
                if prev_vwrsi > self.ob_thresh and vwrsi < self.ob_thresh:
                    score -= 3

        # ===== 2. MACD confirmation (weight 2) =====
        macd_hist = self._calc_macd_hist(close)
        if macd_hist > 0:
            score += 2
        elif macd_hist < 0:
            score -= 2

        # ===== 3. Price momentum (weight 1) =====
        if n >= 10:
            ret_10 = (close[-1] / close[-10] - 1) * 100
            if ret_10 > 5:
                score += 1
            elif ret_10 < -5:
                score -= 1

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_vwrsi(self, data):
        """Volume-Weighted RSI"""
        close = data['close'].values
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)

        if len(close) < self.vwrsi_period + 1:
            return 50.0

        deltas = np.diff(close)

        if vol_col:
            vol = data[vol_col].values
            if len(vol) >= len(deltas) + 1:
                vol_changes = vol[1:]  # Align with deltas
                # Volume-weighted gains and losses
                gains = np.where(deltas > 0, deltas * vol_changes, 0.0)
                losses = np.where(deltas < 0, -deltas * vol_changes, 0.0)
            else:
                gains = np.where(deltas > 0, deltas, 0.0)
                losses = np.where(deltas < 0, -deltas, 0.0)
        else:
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)

        period = self.vwrsi_period

        # Use EMA smoothing for VW-RSI
        k = 2.0 / (self.ema_period + 1)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = gains[i] * k + avg_gain * (1 - k)
            avg_loss = losses[i] * k + avg_loss * (1 - k)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1 + rs)

    def _calc_vwrsi_history(self, data, n_bars):
        """Calculate VW-RSI using only first n_bars"""
        if n_bars < self.vwrsi_period + 1:
            return None

        close = data['close'].values[:n_bars]
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)

        deltas = np.diff(close)

        if vol_col:
            vol = data[vol_col].values[:n_bars]
            if len(vol) >= len(deltas) + 1:
                vol_changes = vol[1:]
                gains = np.where(deltas > 0, deltas * vol_changes, 0.0)
                losses = np.where(deltas < 0, -deltas * vol_changes, 0.0)
            else:
                gains = np.where(deltas > 0, deltas, 0.0)
                losses = np.where(deltas < 0, -deltas, 0.0)
        else:
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)

        period = self.vwrsi_period
        k = 2.0 / (self.ema_period + 1)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = gains[i] * k + avg_gain * (1 - k)
            avg_loss = losses[i] * k + avg_loss * (1 - k)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1 + rs)

    def _calc_macd_hist(self, close):
        fast_ema = self._calc_ema_series(close, self.macd_fast)
        slow_ema = self._calc_ema_series(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal = self._calc_ema_series(macd_line, self.macd_signal)
        return float(macd_line[-1] - signal[-1])

    def _calc_ema_series(self, values, period):
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
        vwrsi = self._calc_vwrsi(data)

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 4:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} VWRSI={vwrsi:.1f}",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score} VWRSI={vwrsi:.1f}', 'price': price}
