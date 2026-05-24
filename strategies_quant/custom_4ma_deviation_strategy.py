"""
Custom 4 MA偏差策略 (Custom 4 MA Deviation Strategy)
=====================================================
四层SMA配合历史偏差分析，在统计极端时做均值回归。

来源: TradingView "Custom 4 MA & Probability (KenshinC)"

核心逻辑:
  1. SMA(16/100/365)三层均线
  2. 追踪每条MA的历史最大正负偏离
  3. 价格接近历史极端偏离=均值回归机会
  4. 多层MA偏离共振增强信号

技术指标: SMA(16/100/365), Deviation Analysis
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class Custom4MaDeviationStrategy(BaseStrategy):
    """Custom 4 MA偏差策略 — 多层MA历史偏差分析"""

    strategy_description = "4MA偏差: SMA(16/100/365)历史偏差极端均值回归"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "sma_fast": {"type": "int", "default": 16, "label": "快SMA"},
        "sma_mid": {"type": "int", "default": 100, "label": "中SMA"},
        "sma_slow": {"type": "int", "default": 200, "label": "慢SMA"},
        "dev_threshold": {"type": "float", "default": 0.8, "label": "偏差阈值(%)"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.sma_fast = params.get('sma_fast', 16)
        self.sma_mid = params.get('sma_mid', 100)
        self.sma_slow = params.get('sma_slow', 200)
        self.dev_threshold = params.get('dev_threshold', 0.8)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'sma_fast': 16, 'sma_mid': 100, 'sma_slow': 200,
            'dev_threshold': 0.8, 'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

        print(f"Custom4MaDev: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        if len(data) < self.sma_slow + 50:
            return None

        close = data['close'].values
        n = len(close)
        score = 0

        # Calculate SMAs
        sma16 = np.mean(close[-self.sma_fast:])
        sma100 = np.mean(close[-self.sma_mid:])
        sma200 = np.mean(close[-self.sma_slow:])

        # Calculate deviation from each MA (percentage)
        dev_fast = (close[-1] - sma16) / sma16 * 100 if sma16 > 0 else 0
        dev_mid = (close[-1] - sma100) / sma100 * 100 if sma100 > 0 else 0
        dev_slow = (close[-1] - sma200) / sma200 * 100 if sma200 > 0 else 0

        # Historical max deviations for each MA
        lookback = min(n - self.sma_slow, 200)
        if lookback > 50:
            devs_fast = []
            devs_mid = []
            for i in range(n - lookback, n):
                if i >= self.sma_fast:
                    s16 = np.mean(close[i - self.sma_fast + 1:i + 1])
                    devs_fast.append((close[i] - s16) / s16 * 100 if s16 > 0 else 0)
                if i >= self.sma_mid:
                    s100 = np.mean(close[i - self.sma_mid + 1:i + 1])
                    devs_mid.append((close[i] - s100) / s100 * 100 if s100 > 0 else 0)

            if devs_fast:
                max_pos_fast = np.percentile(devs_fast, 95)
                max_neg_fast = np.percentile(devs_fast, 5)
                if dev_fast > max_pos_fast * self.dev_threshold:
                    score -= 3  # Overextended above → sell
                elif dev_fast < max_neg_fast * self.dev_threshold:
                    score += 3  # Overextended below → buy

            if devs_mid:
                max_pos_mid = np.percentile(devs_mid, 95)
                max_neg_mid = np.percentile(devs_mid, 5)
                if dev_mid > max_pos_mid * self.dev_threshold:
                    score -= 2
                elif dev_mid < max_neg_mid * self.dev_threshold:
                    score += 2

        # Slow MA deviation (statistical extreme)
        if abs(dev_slow) > 15:
            if dev_slow < -15:
                score += 2  # Deep below slow MA
            elif dev_slow > 15:
                score -= 2

        # MA alignment (trend context)
        if sma16 > sma100 > sma200:
            if score > 0:
                score += 1  # Buy dip in uptrend
        elif sma16 < sma100 < sma200:
            if score < 0:
                score += 1  # Sell rally in downtrend

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
        if len(data) < self.sma_slow + 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (4ma_dev)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
