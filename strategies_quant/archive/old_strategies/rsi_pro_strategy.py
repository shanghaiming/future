"""
RSI Pro策略 (RSI Pro HA+BB+Divergence Strategy)
================================================
增强版RSI策略：Heikin Ashi平滑 + Bollinger Bands + 背离检测。

来源: TradingView "RSI Pro (HA + BB + Divergence + Signals)"

核心逻辑:
  1. Heikin Ashi平滑RSI，减少噪音
  2. RSI的Bollinger Bands判断超买超卖
  3. 价格与RSI背离检测反转信号
  4. ATR追踪止损

技术指标: RSI, Heikin Ashi, Bollinger Bands, Divergence
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class RSIProStrategy(BaseStrategy):
    """RSI Pro策略 — HA平滑RSI + BB包络 + 背离检测"""

    strategy_description = "RSI Pro: Heikin Ashi RSI + BB + 背离检测"
    strategy_category = "momentum"
    strategy_params_schema = {
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "bb_period": {"type": "int", "default": 20, "label": "BB周期"},
        "bb_mult": {"type": "float", "default": 2.0, "label": "BB倍数"},
        "ob_thresh": {"type": "int", "default": 70, "label": "超买阈值"},
        "os_thresh": {"type": "int", "default": 30, "label": "超卖阈值"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.rsi_period = params.get('rsi_period', 14)
        self.bb_period = params.get('bb_period', 20)
        self.bb_mult = params.get('bb_mult', 2.0)
        self.ob_thresh = params.get('ob_thresh', 70)
        self.os_thresh = params.get('os_thresh', 30)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'rsi_period': 14, 'bb_period': 20, 'bb_mult': 2.0,
            'ob_thresh': 70, 'os_thresh': 30,
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

        print(f"RSIPro: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """综合评估: HA-RSI + BB + 背离"""
        min_len = max(self.rsi_period + 1, self.bb_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        open_ = data['open'].values
        n = len(close)

        # 1. Heikin Ashi candlesticks
        ha_close = np.zeros(n)
        ha_open = np.zeros(n)
        ha_close[0] = (open_[0] + high[0] + low[0] + close[0]) / 4.0
        ha_open[0] = open_[0]
        for i in range(1, n):
            ha_close[i] = (open_[i] + high[i] + low[i] + close[i]) / 4.0
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0

        # 2. RSI on HA close
        rsi = self._calc_rsi(ha_close)
        score = 0

        # 3. RSI Bollinger Bands
        rsi_series = self._calc_rsi_series(ha_close)
        if rsi_series is not None and len(rsi_series) >= self.bb_period:
            rsi_slice = rsi_series[-self.bb_period:]
            rsi_ma = np.mean(rsi_slice)
            rsi_std = np.std(rsi_slice, ddof=1)
            rsi_upper = rsi_ma + self.bb_mult * rsi_std
            rsi_lower = rsi_ma - self.bb_mult * rsi_std

            # RSI above upper BB = overbought
            if rsi > rsi_upper:
                score -= 4
            # RSI below lower BB = oversold
            elif rsi < rsi_lower:
                score += 4

        # Simple RSI zones
        if rsi < self.os_thresh:
            score += 3
        elif rsi > self.ob_thresh:
            score -= 3

        # 4. Divergence detection
        div = self._detect_divergence(close, rsi_series if rsi_series is not None else np.full(n, 50.0))
        if div == 'bullish':
            score += 3
        elif div == 'bearish':
            score -= 3

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_rsi(self, close):
        """Calculate current RSI"""
        if len(close) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1 + rs)

    def _calc_rsi_series(self, close):
        """Calculate RSI for all bars using EMA method"""
        n = len(close)
        if n < self.rsi_period + 1:
            return None
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi_arr = np.full(n, 50.0)
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])
        if avg_loss == 0:
            rsi_arr[self.rsi_period] = 100.0
        else:
            rsi_arr[self.rsi_period] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)

        for i in range(self.rsi_period, len(gains)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i]) / self.rsi_period
            if avg_loss == 0:
                rsi_arr[i + 1] = 100.0
            else:
                rsi_arr[i + 1] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)
        return rsi_arr

    def _detect_divergence(self, price, rsi):
        """Detect price-RSI divergence over last ~20 bars"""
        n = len(price)
        lookback = min(20, n - 1)
        if lookback < 5:
            return None

        recent_price = price[-lookback:]
        recent_rsi = rsi[-lookback:]

        # Find local peaks and troughs (simple)
        price_highs = []
        rsi_at_highs = []
        price_lows = []
        rsi_at_lows = []

        for i in range(1, len(recent_price) - 1):
            if recent_price[i] > recent_price[i-1] and recent_price[i] > recent_price[i+1]:
                price_highs.append(recent_price[i])
                rsi_at_highs.append(recent_rsi[i])
            elif recent_price[i] < recent_price[i-1] and recent_price[i] < recent_price[i+1]:
                price_lows.append(recent_price[i])
                rsi_at_lows.append(recent_rsi[i])

        # Bearish divergence: price higher high, RSI lower high
        if len(price_highs) >= 2:
            if price_highs[-1] > price_highs[-2] and rsi_at_highs[-1] < rsi_at_highs[-2]:
                return 'bearish'

        # Bullish divergence: price lower low, RSI higher low
        if len(price_lows) >= 2:
            if price_lows[-1] < price_lows[-2] and rsi_at_lows[-1] > rsi_at_lows[-2]:
                return 'bullish'

        return None

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
                'reason': f"score={score} (rsi_pro)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
