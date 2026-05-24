"""
多因子共振策略 (Confluence Matrix Strategy)
==========================================
多维度独立评分 + 共振确认的交易策略。

来源: TradingView "Confluence Matrix [JOAT]" + "Convergence Protocol [JOAT]"

核心逻辑:
  5个独立维度各评分(-1, 0, +1):
  1. 趋势: EMA方向 + 斜率
  2. 动量: RSI + MACD
  3. 波动率: ATR状态 + BB位置
  4. 成交量: 量价配合
  5. 结构: 摆动高低点突破

  总分范围: -5 到 +5
  高信念(≥3): 入场, 中信念(1-2): 观望, 低信念(0): 不交易

技术指标: EMA, RSI, MACD, ATR, BB, Volume, Swing Points
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class ConfluenceMatrixStrategy(BaseStrategy):
    """多因子共振策略 — 5维度独立评分 + 共振确认"""

    strategy_description = "多因子共振: 趋势+动量+波动率+成交量+结构 5维评分"
    strategy_category = "multi_factor"
    strategy_params_schema = {
        "ema_fast": {"type": "int", "default": 10, "label": "快EMA"},
        "ema_slow": {"type": "int", "default": 30, "label": "慢EMA"},
        "ema_trend": {"type": "int", "default": 50, "label": "趋势EMA"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "bb_period": {"type": "int", "default": 20, "label": "BB周期"},
        "bb_mult": {"type": "float", "default": 2.0, "label": "BB倍数"},
        "swing_period": {"type": "int", "default": 10, "label": "摆动点周期"},
        "entry_threshold": {"type": "int", "default": 3, "label": "入场共振阈值"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ema_fast = params.get('ema_fast', 10)
        self.ema_slow = params.get('ema_slow', 30)
        self.ema_trend = params.get('ema_trend', 50)
        self.rsi_period = params.get('rsi_period', 14)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.atr_period = params.get('atr_period', 14)
        self.bb_period = params.get('bb_period', 20)
        self.bb_mult = params.get('bb_mult', 2.0)
        self.swing_period = params.get('swing_period', 10)
        self.entry_threshold = params.get('entry_threshold', 3)
        self.hold_min = params.get('hold_min', 2)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'ema_fast': 10, 'ema_slow': 30, 'ema_trend': 50,
            'rsi_period': 14,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'atr_period': 14, 'bb_period': 20, 'bb_mult': 2.0,
            'swing_period': 10, 'entry_threshold': 3,
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

                if best_sym and abs(best_score) >= self.entry_threshold:
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

                    # Signal exit (lower threshold)
                    result = self._evaluate(hist)
                    if result is not None and not should_exit:
                        score, direction, _ = result
                        if position_dir == 1 and direction == -1 and score <= -2:
                            should_exit = True
                        elif position_dir == -1 and direction == 1 and score >= 2:
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

        print(f"ConfluenceMatrix: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """5维度独立评分"""
        min_len = max(self.ema_trend, self.macd_slow + self.macd_signal,
                      self.bb_period, self.swing_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        scores = []

        # ===== Dimension 1: Trend =====
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)
        ema_t = self._calc_ema_series(close, self.ema_trend)

        trend_score = 0
        if close[-1] > ema_t[-1]:
            trend_score += 1
        elif close[-1] < ema_t[-1]:
            trend_score -= 1

        if ema_f[-1] > ema_s[-1]:
            trend_score += 0.5
        else:
            trend_score -= 0.5

        # EMA slope
        if n >= 5:
            slope = (ema_t[-1] - ema_t[-5]) / ema_t[-5] if ema_t[-5] > 0 else 0
            if slope > 0.005:
                trend_score += 0.5
            elif slope < -0.005:
                trend_score -= 0.5

        scores.append(1 if trend_score >= 1 else (-1 if trend_score <= -1 else 0))

        # ===== Dimension 2: Momentum =====
        rsi = self._calc_rsi(close)
        macd_hist = self._calc_macd_hist(close)

        momentum_score = 0
        if rsi > 55:
            momentum_score += 1
        elif rsi < 45:
            momentum_score -= 1

        if macd_hist > 0:
            momentum_score += 1
        elif macd_hist < 0:
            momentum_score -= 1

        scores.append(1 if momentum_score >= 1 else (-1 if momentum_score <= -1 else 0))

        # ===== Dimension 3: Volatility =====
        atr = self._calc_atr_arrays(high, low, close, n)
        bb_mid = np.mean(close[-self.bb_period:])
        bb_std = np.std(close[-self.bb_period:], ddof=1)
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std
        bb_pos = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5

        vol_score = 0
        # ATR expanding = energy for move
        if n >= self.atr_period * 2:
            recent_tr = np.maximum(high[-self.atr_period:] - low[-self.atr_period:],
                                   np.maximum(np.abs(high[-self.atr_period:] - close[-self.atr_period - 1:-1]),
                                              np.abs(low[-self.atr_period:] - close[-self.atr_period - 1:-1])))
            older_tr = np.maximum(high[-self.atr_period * 2:-self.atr_period] - low[-self.atr_period * 2:-self.atr_period],
                                  np.maximum(np.abs(high[-self.atr_period * 2:-self.atr_period] - close[-self.atr_period * 2 - 1:-self.atr_period - 1]),
                                             np.abs(low[-self.atr_period * 2:-self.atr_period] - close[-self.atr_period * 2 - 1:-self.atr_period - 1])))
            if len(recent_tr) > 0 and len(older_tr) > 0:
                if np.mean(recent_tr) > np.mean(older_tr) * 1.2:
                    vol_score += 1 if close[-1] > close[-self.atr_period] else -1

        # BB position
        if bb_pos > 0.7:
            vol_score += 1
        elif bb_pos < 0.3:
            vol_score -= 1

        scores.append(1 if vol_score >= 1 else (-1 if vol_score <= -1 else 0))

        # ===== Dimension 4: Volume =====
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        vol_score = 0
        if vol_col and len(data) >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * 1.3:
                # High volume: confirms direction
                vol_score = 1 if close[-1] > data['open'].values[-1] else -1
            elif vol[-1] < vol_ma * 0.7:
                vol_score = 0  # Low volume: no confirmation
            else:
                vol_score = 0
        scores.append(vol_score)

        # ===== Dimension 5: Structure =====
        struct_score = 0
        if n >= self.swing_period + 1:
            recent_high = np.max(high[-self.swing_period:])
            recent_low = np.min(low[-self.swing_period:])
            if close[-1] >= recent_high * 0.99:
                struct_score = 1  # Near swing high = bullish breakout
            elif close[-1] <= recent_low * 1.01:
                struct_score = -1  # Near swing low = bearish breakout

            # Higher highs / lower lows pattern
            if n >= self.swing_period * 3:
                prev_high = np.max(high[-self.swing_period * 2:-self.swing_period])
                prev_prev_high = np.max(high[-self.swing_period * 3:-self.swing_period * 2])
                if recent_high > prev_high > prev_prev_high:
                    struct_score = 1  # Ascending peaks
                elif recent_high < prev_high < prev_prev_high:
                    struct_score = -1  # Descending peaks
        scores.append(struct_score)

        # ===== Aggregate =====
        total_score = sum(scores)
        direction = 1 if total_score > 0 else -1
        return total_score, direction, atr

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

    def _calc_macd_hist(self, close):
        fast_ema = self._calc_ema_series(close, self.macd_fast)
        slow_ema = self._calc_ema_series(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal = self._calc_ema_series(macd_line, self.macd_signal)
        return macd_line[-1] - signal[-1]

    def _calc_atr_arrays(self, high, low, close, n):
        if n < self.atr_period + 1:
            return 0
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

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
        if abs(score) >= self.entry_threshold:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"confluence={score}/5",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'confluence={score}/5', 'price': price}
