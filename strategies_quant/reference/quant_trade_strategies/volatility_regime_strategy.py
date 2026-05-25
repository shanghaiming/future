"""
波动率状态策略 (Volatility Regime Strategy)
==========================================
ATR Z-score波动率状态分类 + 自适应参数交易策略。

来源: TradingView "Volatility Regime Classifier [JOAT]"

核心逻辑:
  三指标分类:
  1. ATR Z-score > 2 → VOLATILE (极端波动)
  2. ATR Percentile < 35% → RANGING (低波动震荡)
  3. EMA方向分离 > 1.5 ATR → TRENDING (趋势)
  4. 其他 → MIXED (混合)

  状态映射策略:
  - TRENDING: 趋势跟踪 (EMA交叉 + MACD)
  - RANGING: 均值回归 (Z-score + RSI)
  - VOLATILE: 减仓观望 (等待极端回归)
  - MIXED: 谨慎交易 (需强确认)

技术指标: ATR Z-score, ATR Percentile, EMA方向比率, RSI, MACD, Bollinger Bands
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class VolatilityRegimeStrategy(BaseStrategy):
    """波动率状态策略 — ATR状态分类 + 自适应交易"""

    strategy_description = "波动率状态: ATR Z-score分类 + 状态自适应策略"
    strategy_category = "adaptive"
    strategy_params_schema = {
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_zscore_period": {"type": "int", "default": 100, "label": "ATR Z-score窗口"},
        "ema_fast": {"type": "int", "default": 20, "label": "快EMA"},
        "ema_slow": {"type": "int", "default": 50, "label": "慢EMA"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "bb_period": {"type": "int", "default": 20, "label": "BB周期"},
        "bb_mult": {"type": "float", "default": 2.0, "label": "BB倍数"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.atr_period = params.get('atr_period', 14)
        self.atr_zscore_period = params.get('atr_zscore_period', 100)
        self.ema_fast = params.get('ema_fast', 20)
        self.ema_slow = params.get('ema_slow', 50)
        self.rsi_period = params.get('rsi_period', 14)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.bb_period = params.get('bb_period', 20)
        self.bb_mult = params.get('bb_mult', 2.0)
        self.hold_min = params.get('hold_min', 2)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'atr_period': 14, 'atr_zscore_period': 100,
            'ema_fast': 20, 'ema_slow': 50,
            'rsi_period': 14,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bb_period': 20, 'bb_mult': 2.0,
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

        print(f"VolatilityRegime: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """综合评估: 波动率状态 + 自适应策略"""
        min_len = max(self.atr_zscore_period, self.ema_slow, self.macd_slow + self.macd_signal,
                      self.bb_period, self.rsi_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ===== 1. Volatility Regime Classification =====
        regime = self._classify_regime(close, high, low, n)

        # ===== 2. Common Indicators =====
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)
        above_ema = close[-1] > ema_s[-1]
        ema_slope = (ema_s[-1] - ema_s[-5]) / ema_s[-5] if n >= 5 and ema_s[-5] > 0 else 0

        # MACD
        fast_ema = self._calc_ema_series(close, self.macd_fast)
        slow_ema = self._calc_ema_series(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._calc_ema_series(macd_line, self.macd_signal)
        macd_hist = macd_line[-1] - signal_line[-1]

        # RSI
        rsi = self._calc_rsi(close)

        # Bollinger Bands
        bb_mid = np.mean(close[-self.bb_period:])
        bb_std = np.std(close[-self.bb_period:], ddof=1)
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std
        bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5

        # Z-score
        zscore = self._calc_zscore(close)

        # Volume
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        vol_confirm = True
        if vol_col and len(data) >= 20:
            vol = data[vol_col].values
            vol_confirm = vol[-1] >= np.mean(vol[-20:])

        # ===== 3. Regime-Dependent Scoring =====
        score = 0

        if regime == 'TRENDING':
            # Trend following: EMA + MACD + direction
            if above_ema and ema_slope > 0:
                score += 3
            elif not above_ema and ema_slope < 0:
                score -= 3

            if macd_hist > 0:
                score += 2
            elif macd_hist < 0:
                score -= 2

            if ema_f[-1] > ema_s[-1]:
                score += 1
            else:
                score -= 1

            # Volume confirms trend
            if vol_confirm:
                score += int(np.sign(score))

        elif regime == 'RANGING':
            # Mean reversion: BB + RSI + Z-score
            if bb_position < 0.1:
                score += 4  # Near lower band → buy
            elif bb_position > 0.9:
                score -= 4  # Near upper band → sell

            if rsi < 30:
                score += 3
            elif rsi > 70:
                score -= 3

            if zscore < -1.5:
                score += 2
            elif zscore > 1.5:
                score -= 2

        elif regime == 'VOLATILE':
            # Cautious: only extreme reversion
            if rsi < 20 and bb_position < 0.05:
                score += 5  # Extreme oversold
            elif rsi > 80 and bb_position > 0.95:
                score -= 5  # Extreme overbought

            # Suppress normal signals in volatile regime
            if abs(score) < 5:
                return 0, 1, 0  # No signal in volatile

        else:  # MIXED
            # Require multi-confirmation
            confirm_count = 0
            if above_ema:
                confirm_count += 1
            if macd_hist > 0:
                confirm_count += 1
            if rsi < 60:
                confirm_count += 1
            if vol_confirm:
                confirm_count += 1

            if confirm_count >= 3:
                score += 4
            elif confirm_count <= 1:
                score -= 4

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _classify_regime(self, close, high, low, n):
        """波动率状态分类"""
        if n < self.atr_zscore_period:
            return 'MIXED'

        # ATR series
        atr_series = self._calc_atr_series(high, low, close, n)
        if len(atr_series) < self.atr_zscore_period:
            return 'MIXED'

        recent_atr = atr_series[-self.atr_zscore_period:]
        current_atr = recent_atr[-1]
        atr_mean = np.mean(recent_atr)
        atr_std = np.std(recent_atr, ddof=1)

        # 1. ATR Z-score
        atr_zscore = (current_atr - atr_mean) / atr_std if atr_std > 0 else 0
        if atr_zscore > 2.0:
            return 'VOLATILE'

        # 2. ATR Percentile
        atr_percentile = np.sum(recent_atr < current_atr) / len(recent_atr) * 100
        if atr_percentile < 35:
            return 'RANGING'

        # 3. EMA directional separation
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)
        ema_sep = abs(ema_f[-1] - ema_s[-1])
        if ema_sep > 1.5 * current_atr and current_atr > 0:
            return 'TRENDING'

        return 'MIXED'

    def _calc_atr_series(self, high, low, close, n):
        if n < self.atr_period + 1:
            return np.array([0])
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr = np.zeros(n)
        if len(tr) >= self.atr_period:
            atr[self.atr_period] = np.mean(tr[:self.atr_period])
            for i in range(self.atr_period + 1, n):
                atr[i] = (atr[i - 1] * (self.atr_period - 1) + tr[i - 1]) / self.atr_period
        return atr[self.atr_period:]  # Return only valid values

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
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        regime = self._classify_regime(close, high, low, len(close))

        if abs(score) >= 4:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} regime={regime}",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score} regime={regime}', 'price': price}
