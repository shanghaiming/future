"""
Kaufman效率比率门控策略 (Kaufman Efficiency Ratio Gate Strategy)
==============================================================
使用Kaufman效率比率(KER)作为交易门控过滤器。

来源: TradingView "Kaufman Efficiency Ratio Gate [NovaLens]"

核心逻辑:
  KER = |净变化| / Σ|日变化|
  高效率(>0.3) = 趋势市 → 允许趋势跟踪信号
  低效率(<0.15) = 震荡市 → 禁止交易或切换均值回归

结合指标:
  - KER门控: 只在效率高时允许交易
  - 趋势信号: EMA交叉 + MACD确认
  - 均值回归: RSI超卖 + Z-score极端
  - ATR止损: 自适应追踪止损

技术指标: Kaufman Efficiency Ratio, EMA, MACD, RSI, Z-score, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class KauffERGateStrategy(BaseStrategy):
    """Kaufman效率比率门控策略 — KER过滤 + 趋势/回归双模式"""

    strategy_description = "KER门控: 效率比率过滤 + 趋势跟踪/均值回归自适应"
    strategy_category = "adaptive"
    strategy_params_schema = {
        "er_period": {"type": "int", "default": 20, "label": "效率比率周期"},
        "er_trend_thresh": {"type": "float", "default": 0.3, "label": "趋势效率阈值"},
        "er_range_thresh": {"type": "float", "default": 0.15, "label": "震荡效率阈值"},
        "ema_fast": {"type": "int", "default": 10, "label": "快EMA"},
        "ema_slow": {"type": "int", "default": 30, "label": "慢EMA"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.er_period = params.get('er_period', 20)
        self.er_trend_thresh = params.get('er_trend_thresh', 0.3)
        self.er_range_thresh = params.get('er_range_thresh', 0.15)
        self.ema_fast = params.get('ema_fast', 10)
        self.ema_slow = params.get('ema_slow', 30)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.rsi_period = params.get('rsi_period', 14)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 2)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'er_period': 20, 'er_trend_thresh': 0.3, 'er_range_thresh': 0.15,
            'ema_fast': 10, 'ema_slow': 30,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'rsi_period': 14, 'atr_period': 14,
            'hold_min': 2, 'trail_atr_mult': 2.5,
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

        print(f"KauffERGate: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """综合评估: KER门控 + 趋势/回归双模式"""
        min_len = max(self.er_period, self.ema_slow, self.macd_slow + self.macd_signal, self.rsi_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ===== 1. Kaufman Efficiency Ratio =====
        er = self._calc_er(close)
        regime = 'trend' if er > self.er_trend_thresh else ('range' if er < self.er_range_thresh else 'neutral')

        # ===== 2. EMA Cross =====
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)
        ema_cross_up = n >= 2 and ema_f[-2] < ema_s[-2] and ema_f[-1] > ema_s[-1]
        ema_cross_down = n >= 2 and ema_f[-2] > ema_s[-2] and ema_f[-1] < ema_s[-1]
        above_ema = close[-1] > ema_s[-1]

        # EMA slope
        ema_slope = (ema_s[-1] - ema_s[-5]) / ema_s[-5] if n >= 5 and ema_s[-5] > 0 else 0

        # ===== 3. MACD =====
        fast_ema = self._calc_ema_series(close, self.macd_fast)
        slow_ema = self._calc_ema_series(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._calc_ema_series(macd_line, self.macd_signal)
        macd_hist = macd_line[-1] - signal_line[-1]
        macd_positive = macd_hist > 0

        # ===== 4. RSI =====
        rsi = self._calc_rsi(close)

        # ===== 5. Z-score =====
        zscore = self._calc_zscore(close)

        # ===== 6. Volume =====
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        vol_confirm = True
        if vol_col and len(data) >= 20:
            vol = data[vol_col].values
            vol_confirm = vol[-1] >= np.mean(vol[-20:])

        # ===== Scoring (regime-dependent) =====
        score = 0

        if regime == 'trend':
            # TREND MODE: EMA cross + MACD + direction
            if ema_cross_up:
                score += 4
            elif above_ema and ema_slope > 0:
                score += 2

            if ema_cross_down:
                score -= 4
            elif not above_ema and ema_slope < 0:
                score -= 2

            if macd_positive:
                score += 2
            else:
                score -= 2

            # ER strength bonus
            if er > 0.5:
                score += 2 * (1 if score > 0 else -1)

        elif regime == 'range':
            # RANGE MODE: Mean reversion signals
            if rsi < 30:
                score += 4
            elif rsi < 40:
                score += 2
            elif rsi > 70:
                score -= 4
            elif rsi > 60:
                score -= 2

            if zscore < -1.5:
                score += 3
            elif zscore > 1.5:
                score -= 3

            # Suppress trend signals in range
            if ema_cross_up or ema_cross_down:
                score = int(score * 0.5)

        else:
            # NEUTRAL: Require stronger confirmation
            if ema_cross_up and macd_positive and rsi < 60:
                score += 5
            elif ema_cross_down and not macd_positive and rsi > 40:
                score -= 5

            if rsi < 30 and zscore < -1.5:
                score += 3
            elif rsi > 70 and zscore > 1.5:
                score -= 3

        # Volume confirmation (universal)
        if vol_confirm and abs(score) >= 3:
            score += int(np.sign(score)) * 1

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_er(self, close):
        """Kaufman Efficiency Ratio"""
        if len(close) < self.er_period:
            return 0
        recent = close[-self.er_period:]
        net = abs(recent[-1] - recent[0])
        total = np.sum(np.abs(np.diff(recent)))
        return net / total if total > 0 else 0

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
        period = 20
        if len(close) < period:
            return 0.0
        recent = close[-period:]
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
        er = self._calc_er(data['close'].values)
        regime = 'trend' if er > self.er_trend_thresh else ('range' if er < self.er_range_thresh else 'neutral')

        if abs(score) >= 4:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (KER={er:.2f}, regime={regime})",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}, KER={er:.2f}, regime={regime}', 'price': price}
