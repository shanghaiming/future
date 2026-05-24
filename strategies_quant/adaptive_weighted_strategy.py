"""
自适应加权融合策略 (Adaptive Weighted Fusion Strategy)
======================================================
融合所有学习成果的最终策略:

知识来源:
1. TradingView Acrypto Weighted Strategy: 5指标加权评分, 权重可配
2. 概率论: Fisher Transform(高斯化), Z-score(标准化), R²(趋势强度)
3. Al Brooks PA: High 2回调, Pin Bar, 失败突破, Zone边界
4. 市场状态识别: ADX+R²+效率率 → 趋势/震荡/过渡

策略架构:
┌─────────────────────────────────────────────┐
│ 市场状态识别器 (ADX + R² + Efficiency)        │
│   → trend / range / transition              │
├─────────────────────────────────────────────┤
│ 信号评估器 (加权评分)                          │
│   ┌─ Fisher Transform  (权重: 3)             │
│   ├─ MACD Histogram    (权重: 2)             │
│   ├─ Supertrend方向    (权重: 2)             │
│   ├─ RSI               (权重: 1)             │
│   ├─ Z-score/PA        (权重: 2)             │
│   └─ 成交量确认        (权重: 1)             │
├─────────────────────────────────────────────┤
│ 风险管理                                     │
│   止损: ATR * 2.0                            │
│   止盈: Fisher极值 或 RSI超买 或 MACD死叉     │
│   仓位: 固定比例 (全仓)                       │
└─────────────────────────────────────────────┘

趋势模式: Fisher金叉 + MACD正向 + Supertrend看涨 (≥6分)
震荡模式: Z-score超卖 + Fisher极值 + Pin Bar (≥5分)
过渡模式: 需要最强确认 (≥8分)

防未来数据泄漏: data[data.index < current_time], 无shift(-n)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy
from core.market_regime import MarketRegimeDetector


class AdaptiveWeightedStrategy(BaseStrategy):
    """自适应加权融合策略 — 市场状态+加权评分+风险管理"""

    strategy_description = "融合Fisher/MACD/RSI/Supertrend/PA的自适应加权策略"
    strategy_category = "adaptive"
    strategy_params_schema = {
        "fisher_period": {"type": "int", "default": 10, "label": "Fisher周期"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 3.0, "label": "Supertrend倍数"},
        "zscore_period": {"type": "int", "default": 20, "label": "Z-score周期"},
        "ema_period": {"type": "int", "default": 20, "label": "趋势EMA"},
        "regime_lookback": {"type": "int", "default": 20, "label": "状态回看"},
        "trend_threshold": {"type": "int", "default": 6, "label": "趋势模式阈值"},
        "range_threshold": {"type": "int", "default": 5, "label": "震荡模式阈值"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "stop_atr_mult": {"type": "float", "default": 2.0, "label": "止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.fisher_period = params.get('fisher_period', 10)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.rsi_period = params.get('rsi_period', 14)
        self.atr_period = params.get('atr_period', 14)
        self.atr_mult = params.get('atr_mult', 3.0)
        self.zscore_period = params.get('zscore_period', 20)
        self.ema_period = params.get('ema_period', 20)
        self.regime_lookback = params.get('regime_lookback', 20)
        self.trend_threshold = params.get('trend_threshold', 6)
        self.range_threshold = params.get('range_threshold', 5)
        self.hold_min = params.get('hold_min', 2)
        self.stop_atr_mult = params.get('stop_atr_mult', 2.0)
        self.regime_detector = MarketRegimeDetector(lookback=self.regime_lookback)

    def get_default_params(self):
        return {
            'fisher_period': 10, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'rsi_period': 14, 'atr_period': 14, 'atr_mult': 3.0,
            'zscore_period': 20, 'ema_period': 20, 'regime_lookback': 20,
            'trend_threshold': 6, 'range_threshold': 5,
            'hold_min': 2, 'stop_atr_mult': 2.0,
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
                        'timestamp': current_time, 'action': 'buy', 'symbol': best_stock,
                    })
                    current_holding = best_stock
                    buy_time = current_time
            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                if days_held >= self.hold_min:
                    if self._should_sell(current_holding, current_time, data):
                        self.signals.append({
                            'timestamp': current_time, 'action': 'sell', 'symbol': current_holding,
                        })
                        current_holding = None
                        buy_time = None

        print(f"AdaptiveWeighted: 生成 {len(self.signals)} 个信号")
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
        min_len = max(self.macd_slow + self.macd_signal, self.ema_period, self.fisher_period) + 15
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ===== 市场状态 =====
        regime, confidence, details = self.regime_detector.detect(close, high, low)

        # ===== 所有指标 =====
        # Fisher Transform
        fisher, fisher_sig = self._calc_fisher(high, low, n)
        fisher_cross_up = n >= 2 and fisher[-2] < fisher_sig[-2] and fisher[-1] > fisher_sig[-1]
        fisher_cross_down = n >= 2 and fisher[-2] > fisher_sig[-2] and fisher[-1] < fisher_sig[-1]
        fisher_oversold = fisher[-1] < -1.5
        fisher_overbought = fisher[-1] > 1.5

        # MACD
        fast_ema = self._calc_ema_series(close, self.macd_fast)
        slow_ema = self._calc_ema_series(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._calc_ema_series(macd_line, self.macd_signal)
        hist_vals = macd_line - signal_line
        macd_cross_up = n >= 2 and hist_vals[-2] < 0 and hist_vals[-1] > 0
        macd_cross_down = n >= 2 and hist_vals[-2] > 0 and hist_vals[-1] < 0
        macd_positive = hist_vals[-1] > 0

        # Supertrend
        st_dir = self._calc_supertrend(high, low, close)

        # RSI
        rsi = self._calc_rsi(close)

        # Z-score
        zscore = self._calc_zscore(close)

        # ATR
        atr = self._calc_atr(high, low, close)

        # PA信号
        pa_signal = self._detect_pa_signal(close, high, low, n)

        # 成交量
        vol_confirm = self._check_volume(data)

        current_price = close[-1]

        if sell_mode:
            return self._sell_logic(regime, fisher_cross_down, fisher_overbought,
                                    macd_cross_down, st_dir, rsi, zscore)
        else:
            return self._buy_logic(
                regime, fisher_cross_up, fisher_oversold, macd_cross_up, macd_positive,
                st_dir, rsi, zscore, pa_signal, vol_confirm, current_price, atr
            )

    def _buy_logic(self, regime, fisher_cross_up, fisher_oversold, macd_cross_up,
                   macd_positive, st_dir, rsi, zscore, pa_signal, vol_confirm, price, atr):
        """
        加权评分系统 (TradingView Acrypto启发):
        Fisher: 权重3, MACD: 权重2, Supertrend: 权重2, RSI: 权重1, PA/Z: 权重2, Vol: 权重1
        """
        score = 0

        # Fisher Transform (权重3)
        if fisher_cross_up:
            score += 3  # 金叉 = 3分
        if fisher_oversold:
            score += 2  # 超卖区回升潜力

        # MACD (权重2)
        if macd_cross_up:
            score += 2
        elif macd_positive:
            score += 1

        # Supertrend (权重2)
        if st_dir == 1:
            score += 2

        # RSI (权重1)
        if rsi < 35:
            score += 1  # 超卖
        elif rsi < 50:
            score += 0.5
        elif rsi > 70:
            score -= 1  # 超买扣分

        # PA/Z-score (权重2)
        if pa_signal in ('high2', 'pin_bar_bull', 'failed_breakout_bull'):
            score += 2
        if zscore < -1.5:
            score += 1  # Z-score偏低

        # 成交量 (权重1)
        if vol_confirm:
            score += 1

        # 根据市场状态设不同阈值
        if regime == MarketRegimeDetector.TREND:
            threshold = self.trend_threshold  # 趋势市: 需要趋势确认
            # 趋势市必须Supertrend看涨
            if st_dir != 1:
                score -= 3
        elif regime == MarketRegimeDetector.RANGE:
            threshold = self.range_threshold  # 震荡市: 重视均值回归
            # 震荡市Z-score和Fisher超卖更重要
            if fisher_oversold or zscore < -1.5:
                score += 2
        else:
            threshold = max(self.trend_threshold, self.range_threshold) + 1  # 过渡市更高阈值

        should_buy = score >= threshold
        stop = price - self.stop_atr_mult * atr if atr > 0 else price * 0.95

        return score, should_buy, stop

    def _sell_logic(self, regime, fisher_cross_down, fisher_overbought,
                    macd_cross_down, st_dir, rsi, zscore):
        """卖出逻辑: 任何强卖出信号即退出"""
        sell_score = 0

        if fisher_cross_down:
            sell_score += 3
        if fisher_overbought:
            sell_score += 2
        if macd_cross_down:
            sell_score += 2
        if st_dir == -1:
            sell_score += 2
        if rsi > 70:
            sell_score += 1
        if zscore > 2.0:
            sell_score += 1

        # 趋势市: Supertrend翻空即卖; 震荡市: Fisher超买即卖
        if regime == MarketRegimeDetector.TREND:
            should_sell = sell_score >= 3
        elif regime == MarketRegimeDetector.RANGE:
            should_sell = sell_score >= 3
        else:
            should_sell = sell_score >= 4

        return 0, should_sell, 0

    # ===== PA信号检测 =====

    def _detect_pa_signal(self, close, high, low, n):
        if n < 8:
            return None

        # High 2回调
        if n >= 10:
            recent_low = low[-7:-1]
            lows = []
            for i in range(1, len(recent_low) - 1):
                if recent_low[i] < recent_low[i - 1] and recent_low[i] < recent_low[i + 1]:
                    lows.append((i, recent_low[i]))
            if len(lows) >= 2 and lows[-1][1] >= lows[-2][1] * 0.998:
                if close[-1] > close[-7]:
                    return 'high2'

        # Pin Bar
        if n >= 2:
            bar_range = high[-1] - low[-1]
            if bar_range > 0:
                open_approx = close[-2]
                lower_wick = min(close[-1], open_approx) - low[-1]
                upper_wick = high[-1] - max(close[-1], open_approx)
                if lower_wick >= bar_range * 0.6 and upper_wick <= bar_range * 0.15:
                    return 'pin_bar_bull'

        # 外包线反转
        if n >= 2:
            if (high[-1] > high[-2] and low[-1] < low[-2] and
                    close[-1] > (high[-1] + low[-1]) / 2):
                return 'outside_bar_bull'

        # 失败突破 (空头陷阱)
        if n >= 5:
            swing_low = min(low[-5:-1])
            if low[-2] < swing_low and close[-1] > swing_low and close[-1] > close[-2]:
                return 'failed_breakout_bull'

        return None

    def _check_volume(self, data):
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col is None:
            return True
        vol = data[vol_col].values
        if len(vol) < 20:
            return True
        vol_ma = np.mean(vol[-20:])
        return vol[-1] >= vol_ma * 0.8

    # ===== 指标计算 =====

    def _calc_fisher(self, high, low, n):
        period = self.fisher_period
        median = (high + low) / 2.0
        fisher = np.zeros(n)
        smoothed = np.zeros(n)
        signal = np.zeros(n)
        for i in range(period, n):
            lowest = np.min(median[i - period + 1:i + 1])
            highest = np.max(median[i - period + 1:i + 1])
            pr = highest - lowest
            raw = 2.0 * ((median[i] - lowest) / pr) - 1.0 if pr > 0 else 0.0
            raw = max(min(raw, 0.999), -0.999)
            smoothed[i] = 0.33 * raw + 0.67 * smoothed[i - 1]
            val = max(min(smoothed[i], 0.999), -0.999)
            if (1 + val) > 0 and (1 - val) > 0:
                fisher[i] = 0.5 * np.log((1 + val) / (1 - val)) + 0.5 * fisher[i - 1]
            else:
                fisher[i] = fisher[i - 1]
            signal[i] = fisher[i - 1]
        return fisher, signal

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
        period = self.zscore_period
        if len(close) < period:
            return 0.0
        recent = close[-period:]
        std = np.std(recent, ddof=1)
        if std == 0:
            return 0.0
        return (close[-1] - np.mean(recent)) / std

    def _calc_atr(self, high, low, close):
        period = self.atr_period
        if len(close) < period + 1:
            return 0
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-period:])

    def _calc_supertrend(self, high, low, close):
        period = self.atr_period
        mult = self.atr_mult
        if len(close) < period + 2:
            return 0
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
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
