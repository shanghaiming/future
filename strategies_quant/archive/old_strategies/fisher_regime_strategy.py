"""
Fisher Transform + 市场状态自适应策略
=====================================
统计学基础:
- Fisher变换: 将非正态分布的价格数据转为近似高斯分布, 使转折点更清晰
  公式: F = 0.5 * ln((1+X)/(1-X)) + 0.5 * F_{t-1}
  其中X是归一化到[-1,+1]的价格
- Z-score: 标准化价格偏差, 用于均值回归入场
- R²: 衡量趋势强度, 动态切换策略模式

策略逻辑:
- 趋势市 (regime=TREND): Fisher金叉做多 + Supertrend确认 + MACD方向一致
- 震荡市 (regime=RANGE): Z-score均值回归 + Zone边界反转 + Fisher极值反转
- 过渡市 (regime=TRANSITION): 减仓, 只用最强信号

TradingView对应:
- Fisher Transform: ta.fisherTransform(high, low, 10)
- Z-Score: 自定义 rolling_zscore
- Supertrend: ta.supertrend(3, 10)
- MACD: ta.macd(close, 12, 26, 9)

防未来数据泄漏: data[data.index < current_time], 无shift(-n)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy
from core.market_regime import MarketRegimeDetector


class FisherRegimeStrategy(BaseStrategy):
    """Fisher Transform + 市场状态自适应策略"""

    strategy_description = "基于Fisher变换和市场状态识别的自适应策略"
    strategy_category = "adaptive"
    strategy_params_schema = {
        "fisher_period": {"type": "int", "default": 10, "label": "Fisher变换周期"},
        "ema_period": {"type": "int", "default": 20, "label": "趋势EMA周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 3.0, "label": "Supertrend倍数"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "zscore_period": {"type": "int", "default": 20, "label": "Z-score周期"},
        "regime_lookback": {"type": "int", "default": 20, "label": "市场状态回看"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "stop_atr_mult": {"type": "float", "default": 2.0, "label": "止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.fisher_period = params.get('fisher_period', 10)
        self.ema_period = params.get('ema_period', 20)
        self.atr_period = params.get('atr_period', 14)
        self.atr_mult = params.get('atr_mult', 3.0)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.rsi_period = params.get('rsi_period', 14)
        self.zscore_period = params.get('zscore_period', 20)
        self.regime_lookback = params.get('regime_lookback', 20)
        self.hold_min = params.get('hold_min', 2)
        self.stop_atr_mult = params.get('stop_atr_mult', 2.0)
        self.regime_detector = MarketRegimeDetector(lookback=self.regime_lookback)

    def get_default_params(self):
        return {
            'fisher_period': 10, 'ema_period': 20,
            'atr_period': 14, 'atr_mult': 3.0,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'rsi_period': 14, 'zscore_period': 20,
            'regime_lookback': 20, 'hold_min': 2, 'stop_atr_mult': 2.0,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        self.signals = []
        unique_times = sorted(data.index.unique())
        current_holding = None
        buy_time = None

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_stock, stop = self._select_best(current_bars, current_time, data)
                if best_stock:
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'symbol': best_stock,
                    })
                    current_holding = best_stock
                    buy_time = current_time
            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                if days_held >= self.hold_min:
                    if self._should_sell(current_holding, current_time, data):
                        self.signals.append({
                            'timestamp': current_time,
                            'action': 'sell',
                            'symbol': current_holding,
                        })
                        current_holding = None
                        buy_time = None

        print(f"FisherRegime: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _select_best(self, current_bars, current_time, full_data):
        best_score = -float('inf')
        best_stock = None
        best_stop = 0

        for _, bar in current_bars.iterrows():
            symbol = bar['symbol']
            hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
            result = self._evaluate(hist)
            if result is None:
                continue
            score, should_buy, stop = result
            if should_buy and score > best_score:
                best_score = score
                best_stock = symbol
                best_stop = stop

        return best_stock, best_stop

    def _should_sell(self, symbol, current_time, full_data):
        hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
        result = self._evaluate(hist, sell_mode=True)
        if result is None:
            return False
        _, should_sell, _ = result
        return should_sell

    def _evaluate(self, data, sell_mode=False):
        """核心: 根据市场状态自适应切换策略"""
        min_len = max(self.macd_slow + self.macd_signal, self.ema_period, self.fisher_period) + 15
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # === 市场状态识别 ===
        regime, confidence, details = self.regime_detector.detect(close, high, low)
        direction = details.get('direction', 'up')

        # === Fisher Transform ===
        fisher, fisher_signal = self._calc_fisher(high, low, n)

        # === MACD ===
        fast_ema = self._calc_ema_series(close, self.macd_fast)
        slow_ema = self._calc_ema_series(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._calc_ema_series(macd_line, self.macd_signal)
        histogram = macd_line - signal_line

        # === Supertrend ===
        st_dir = self._calc_supertrend(high, low, close)

        # === RSI ===
        rsi = self._calc_rsi(close)

        # === Z-score ===
        zscore = self._calc_zscore(close)

        # === ATR ===
        atr = self._calc_atr(high, low, close)

        current_price = close[-1]

        if sell_mode:
            return self._sell_logic(regime, fisher, fisher_signal, st_dir,
                                     histogram, rsi, zscore, current_price)
        else:
            return self._buy_logic(regime, confidence, direction, fisher, fisher_signal,
                                    st_dir, histogram, rsi, zscore, current_price, atr)

    def _buy_logic(self, regime, confidence, direction, fisher, fisher_signal,
                   st_dir, histogram, rsi, zscore, price, atr):
        """根据市场状态选择不同的买入逻辑"""
        score = 0
        signal_type = None

        if regime == MarketRegimeDetector.TREND:
            # === 趋势模式: Fisher金叉 + Supertrend + MACD方向 ===
            fisher_cross_up = fisher[-2] < fisher_signal[-2] and fisher[-1] > fisher_signal[-1]
            macd_positive = histogram[-1] > 0

            if fisher_cross_up:
                score += 15
                signal_type = 'fisher_cross'
            if st_dir == 1 and direction == 'up':
                score += 15
            if macd_positive:
                score += 5
            if 30 < rsi < 70:
                score += 5

            should_buy = fisher_cross_up and st_dir == 1 and score >= 25

        elif regime == MarketRegimeDetector.RANGE:
            # === 震荡模式: Z-score均值回归 + Fisher极值反转 ===
            fisher_oversold = fisher[-1] < -1.5 and fisher[-1] > fisher_signal[-1]
            zscore_oversold = zscore < -1.8

            if zscore_oversold:
                score += 15
                signal_type = 'zscore_reversion'
            if fisher_oversold:
                score += 12
                if signal_type is None:
                    signal_type = 'fisher_oversold'
            if rsi < 35:
                score += 10

            # Zone边界: 价格在近期区间的下20%
            should_buy = (zscore_oversold or fisher_oversold) and score >= 20

        else:
            # === 过渡模式: 需要最强信号 ===
            fisher_cross_up = fisher[-2] < fisher_signal[-2] and fisher[-1] > fisher_signal[-1]
            zscore_oversold = zscore < -2.0
            macd_cross_up = histogram[-2] < 0 and histogram[-1] > 0

            if fisher_cross_up and st_dir == 1:
                score += 20
                signal_type = 'strong_fisher'
            if zscore_oversold and rsi < 30:
                score += 20
                if signal_type is None:
                    signal_type = 'strong_reversion'
            if macd_cross_up:
                score += 10

            should_buy = score >= 30

        stop = price - self.stop_atr_mult * atr if atr > 0 else price * 0.95
        return score, should_buy, stop

    def _sell_logic(self, regime, fisher, fisher_signal, st_dir,
                    histogram, rsi, zscore, price):
        """根据市场状态选择不同的卖出逻辑"""
        should_sell = False

        # Fisher死叉: 通用卖出信号
        fisher_cross_down = fisher[-2] > fisher_signal[-2] and fisher[-1] < fisher_signal[-1]

        if regime == MarketRegimeDetector.TREND:
            # 趋势模式: Supertrend翻空 或 Fisher死叉
            should_sell = (st_dir == -1) or fisher_cross_down

        elif regime == MarketRegimeDetector.RANGE:
            # 震荡模式: Z-score超买 或 Fisher超买
            should_sell = (zscore > 1.8) or (fisher[-1] > 1.5) or fisher_cross_down

        else:
            # 过渡模式: 任何卖出信号
            should_sell = fisher_cross_down or (st_dir == -1) or (rsi > 70)

        return 0, should_sell, 0

    # ===== Fisher Transform实现 (Ehlers) =====

    def _calc_fisher(self, high, low, n):
        """
        Fisher Transform (Ehlers)
        将价格归一化到[-1,+1], 然后应用费雪变换得到近似高斯分布
        """
        period = self.fisher_period
        median = (high + low) / 2.0

        fisher = np.zeros(n)
        smoothed = np.zeros(n)
        signal = np.zeros(n)

        for i in range(period, n):
            # 归一化到[-1, +1]
            lowest = np.min(median[i - period + 1:i + 1])
            highest = np.max(median[i - period + 1:i + 1])
            price_range = highest - lowest
            if price_range > 0:
                raw = 2.0 * ((median[i] - lowest) / price_range) - 1.0
            else:
                raw = 0.0

            # Clamp to [-0.999, 0.999]
            raw = max(min(raw, 0.999), -0.999)

            # Ehlers平滑
            smoothed[i] = 0.33 * raw + 0.67 * smoothed[i - 1]
            val = max(min(smoothed[i], 0.999), -0.999)

            # Fisher变换: F = 0.5 * ln((1+X)/(1-X))
            if (1 + val) > 0 and (1 - val) > 0:
                fisher[i] = 0.5 * np.log((1 + val) / (1 - val)) + 0.5 * fisher[i - 1]
            else:
                fisher[i] = fisher[i - 1]

            # 信号线 = 前一bar的Fisher值
            signal[i] = fisher[i - 1]

        return fisher, signal

    # ===== 指标计算 =====

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
        """Rolling Z-score: 标准化价格偏差"""
        period = self.zscore_period
        if len(close) < period:
            return 0.0
        recent = close[-period:]
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)
        if std == 0:
            return 0.0
        return (close[-1] - mean) / std

    def _calc_atr(self, high, low, close):
        period = self.atr_period
        if len(close) < period + 1:
            return 0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        return np.mean(tr[-period:])

    def _calc_supertrend(self, high, low, close):
        period = self.atr_period
        mult = self.atr_mult
        if len(close) < period + 2:
            return 0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        atr = np.zeros(len(close))
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
        hl2 = (high + low) / 2.0
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr
        direction = 0
        for i in range(period + 1, len(close)):
            if direction == 1 and i > period + 1:
                lower[i] = max(lower[i], lower[i - 1])
            elif direction == -1 and i > period + 1:
                upper[i] = min(upper[i], upper[i - 1])
            if close[i] > upper[i - 1]:
                direction = 1
            elif close[i] < lower[i - 1]:
                direction = -1
        return direction
