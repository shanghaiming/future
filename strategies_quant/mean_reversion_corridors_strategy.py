"""
均值回归走廊策略 (Mean Reversion Corridors Strategy)
====================================================
基于统计偏离度的均值回归交易策略。

来源: TradingView "AG Pro Mean Reversion Corridors" + "Custom 4 MA & Probability"

核心逻辑:
  当价格偏离均值到统计显著水平时入场, 预期回归均值。
  使用多维度确认:
  1. Z-score极端 (>2σ 或 <-2σ)
  2. RSI超卖/超买 (<30 或 >70)
  3. Bollinger Bands边界
  4. 成交量确认

  出场: 回归到均值附近 (Z-score回到±0.5以内)

技术指标: Z-score, RSI, Bollinger Bands, Volume, ATR止损
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class MeanReversionCorridorsStrategy(BaseStrategy):
    """均值回归走廊策略 — 统计极端偏离 + 多维度确认回归"""

    strategy_description = "均值回归: Z-score极端 + RSI + BB边界 + 成交量确认"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "lookback_period": {"type": "int", "default": 30, "label": "统计窗口"},
        "z_entry_thresh": {"type": "float", "default": 2.0, "label": "Z-score入场阈值"},
        "z_exit_thresh": {"type": "float", "default": 0.5, "label": "Z-score出场阈值"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_oversold": {"type": "int", "default": 30, "label": "RSI超卖"},
        "rsi_overbought": {"type": "int", "default": 70, "label": "RSI超买"},
        "bb_period": {"type": "int", "default": 20, "label": "BB周期"},
        "bb_mult": {"type": "float", "default": 2.0, "label": "BB倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 3.0, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.lookback_period = params.get('lookback_period', 30)
        self.z_entry_thresh = params.get('z_entry_thresh', 2.0)
        self.z_exit_thresh = params.get('z_exit_thresh', 0.5)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.bb_period = params.get('bb_period', 20)
        self.bb_mult = params.get('bb_mult', 2.0)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 3.0)

    def get_default_params(self):
        return {
            'lookback_period': 30, 'z_entry_thresh': 2.0, 'z_exit_thresh': 0.5,
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'bb_period': 20, 'bb_mult': 2.0, 'atr_period': 14,
            'hold_min': 3, 'trail_atr_mult': 3.0,
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

                    # Mean reversion exit: Z-score returned to normal
                    zscore = self._calc_zscore(hist['close'].values)
                    if position_dir == 1 and zscore > -self.z_exit_thresh:
                        should_exit = True
                    elif position_dir == -1 and zscore < self.z_exit_thresh:
                        should_exit = True

                    # ATR trailing stop (wider for mean reversion)
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # Max hold
                    if days_held >= 30:
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

        print(f"MeanReversionCorridors: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """综合评估: 统计极端 + 多维度确认"""
        min_len = max(self.lookback_period, self.bb_period, self.rsi_period) + 5
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ===== 1. Z-score =====
        recent = close[-self.lookback_period:]
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)
        zscore = (close[-1] - mean) / std if std > 0 else 0

        # Need extreme Z-score to trigger
        if abs(zscore) < self.z_entry_thresh * 0.8:
            return 0, 1 if zscore < 0 else -1, 0

        score = 0

        # Z-score (weight 4) - negative z = buy (oversold), positive = sell (overbought)
        z_magnitude = min(abs(zscore) / 3.0, 1.0)  # Normalize to 0-1
        if zscore < -self.z_entry_thresh:
            score += 4 * z_magnitude
        elif zscore > self.z_entry_thresh:
            score -= 4 * z_magnitude
        elif zscore < -self.z_entry_thresh * 0.8:
            score += 2 * z_magnitude
        elif zscore > self.z_entry_thresh * 0.8:
            score -= 2 * z_magnitude

        # ===== 2. RSI confirmation (weight 2) =====
        rsi = self._calc_rsi(close)
        if rsi < self.rsi_oversold:
            score += 2
        elif rsi > self.rsi_overbought:
            score -= 2
        elif rsi < 40:
            score += 1
        elif rsi > 60:
            score -= 1

        # ===== 3. Bollinger Bands (weight 2) =====
        bb_mid = np.mean(close[-self.bb_period:])
        bb_std = np.std(close[-self.bb_period:], ddof=1)
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std

        if close[-1] <= bb_lower:
            score += 2  # Below lower band → buy
        elif close[-1] >= bb_upper:
            score -= 2  # Above upper band → sell

        # ===== 4. Volume confirmation (weight 1) =====
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and len(data) >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * 1.2:
                # High volume at extreme = capitulation = stronger signal
                score += int(np.sign(score)) * 1

        # ===== 5. Price reversal confirmation (weight 1) =====
        if n >= 3:
            # 3-bar reversal pattern
            if close[-1] > close[-2] and close[-2] < close[-3]:
                if zscore < 0:
                    score += 1  # Bullish reversal at low z
            elif close[-1] < close[-2] and close[-2] > close[-3]:
                if zscore > 0:
                    score -= 1  # Bearish reversal at high z

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_rsi(self, close):
        period = self.rsi_period
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        return 100.0 - 100.0 / (1 + avg_gain / avg_loss)

    def _calc_zscore(self, close):
        if len(close) < self.lookback_period:
            return 0.0
        recent = close[-self.lookback_period:]
        std = np.std(recent, ddof=1)
        if std == 0:
            return 0.0
        return (close[-1] - np.mean(recent)) / std

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
        zscore = self._calc_zscore(data['close'].values)
        if abs(score) >= 4:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} z={zscore:.2f} (MR)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score} z={zscore:.2f}', 'price': price}
