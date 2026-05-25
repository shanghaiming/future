"""
Fibonacci波动率云策略 (Fibonacci Volatility Cloud Strategy)
============================================================
Fibonacci回撤位 + ATR波动率带的概率云交易系统。

来源: TradingView "Fibonacci Volatility Cloud [JOAT]"

核心逻辑:
  1. 计算近期高低点的Fibonacci回撤位(0.236, 0.382, 0.5, 0.618, 0.786)
  2. ATR倍数构建波动率云
  3. 价格在0.382-0.618区间 = 黄金回撤区做多
  4. 突破0.236 = 强趋势延续

技术指标: Fibonacci回撤, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class FibVolatilityCloudStrategy(BaseStrategy):
    """Fibonacci波动率云策略 — Fib回撤 + ATR云"""

    strategy_description = "Fib云: Fibonacci回撤 + ATR波动率概率云"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "lookback": {"type": "int", "default": 60, "label": "高低点回望"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_cloud_mult": {"type": "float", "default": 1.0, "label": "云ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    def __init__(self, data, params):
        super().__init__(data, params)
        self.lookback = params.get('lookback', 60)
        self.atr_period = params.get('atr_period', 14)
        self.atr_cloud_mult = params.get('atr_cloud_mult', 1.0)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'lookback': 60, 'atr_period': 14, 'atr_cloud_mult': 1.0,
            'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"FibVolCloud: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        if len(data) < self.lookback + 10:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        score = 0

        # Find swing high and low
        recent_high = np.max(high[-self.lookback:])
        recent_low = np.min(low[-self.lookback:])
        price_range = recent_high - recent_low

        if price_range <= 0:
            return None

        # Fibonacci levels
        fib_236 = recent_high - 0.236 * price_range
        fib_382 = recent_high - 0.382 * price_range
        fib_500 = recent_high - 0.500 * price_range
        fib_618 = recent_high - 0.618 * price_range
        fib_786 = recent_high - 0.786 * price_range

        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        current = close[-1]
        fib_position = (recent_high - current) / price_range

        # Golden zone (0.382 - 0.618)
        if 0.382 <= fib_position <= 0.618:
            score += 4  # In golden retracement zone

        # Near support levels from below (bounce)
        if abs(current - fib_618) < atr * self.atr_cloud_mult:
            score += 3  # Near 61.8% support
        elif abs(current - fib_382) < atr * self.atr_cloud_mult:
            score += 3  # Near 38.2% support

        # Breakout above 23.6%
        if current > fib_236:
            score += 2

        # Below 78.6% = breakdown
        if current < fib_786:
            score -= 3

        # Trend direction from fib position
        if fib_position < 0.236:
            score += 3  # Strong uptrend
        elif fib_position > 0.786:
            score -= 3  # Strong downtrend

        # Volume at fib level
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * 1.3 and score > 0:
                score += 1

        direction = 1 if score > 0 else -1
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
        if len(data) < 70:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (fib_cloud)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
