"""
Algionics Ribbon Pressure Field
=================================
28条EMA组成"力场"的状态机策略, 核心思路来自TradingView学习:
1. 28条EMA从fast到slow均匀分布, 形成"力场"
2. 距离加权压力比: 不是简单计数, 而是按价格到EMA的距离加权
3. 压力得分 ∈ [-1, +1], 三个数学边界: 0/50/100 → latch state
4. Latch-based状态引擎: 需要持续N bar确认才切换状态
5. 买入/卖出: bearish_latch → bullish_latch (或反向)
6. ATR trailing stop

知识来源:
- TradingView Algionics Ribbon概念
- 多EMA ribbon / Guppy MMA 思想
- 距离加权场论
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class AlgionicsRibbonStrategy(BaseStrategy):
    """Algionics Ribbon Pressure Field — 28条EMA力场 + 距离加权 + Latch状态引擎"""

    strategy_description = (
        "EMA力场策略: 28条EMA距离加权压力 + "
        "Latch三态状态引擎(bear/neutral/bull) + N bar确认 + ATR止损"
    )
    strategy_category = "ma"
    strategy_params_schema = {
        "fast_period": {"type": "int", "default": 5, "label": "最快EMA周期"},
        "slow_period": {"type": "int", "default": 200, "label": "最慢EMA周期"},
        "num_lines": {"type": "int", "default": 28, "label": "EMA线条数"},
        "confirmation_bars": {"type": "int", "default": 2, "label": "状态确认bar数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "止损ATR倍数"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "max_hold": {"type": "int", "default": 90, "label": "最大持仓天数"},
    }

    # Latch states
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1

    def __init__(self, data, params):
        super().__init__(data, params)
        self.fast_period = params.get('fast_period', 5)
        self.slow_period = params.get('slow_period', 200)
        self.num_lines = params.get('num_lines', 28)
        self.confirmation_bars = params.get('confirmation_bars', 2)
        self.atr_period = params.get('atr_period', 14)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.0)
        self.hold_min = params.get('hold_min', 2)
        self.max_hold = params.get('max_hold', 90)

    def get_default_params(self):
        return {
            'fast_period': 5, 'slow_period': 200, 'num_lines': 28,
            'confirmation_bars': 2, 'atr_period': 14,
            'trail_atr_mult': 2.0, 'hold_min': 2, 'max_hold': 90,
        }

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _calc_ema(self, values, period):
        """EMA calculation"""
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

    def _calc_atr(self, high, low, close, n):
        """Average True Range"""
        if n < self.atr_period + 1:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
        )
        return float(np.mean(tr[-self.atr_period:]))

    def _generate_ema_periods(self):
        """从fast到slow均匀分布生成EMA周期列表"""
        periods = np.linspace(
            self.fast_period, self.slow_period, self.num_lines
        ).astype(int)
        # Ensure unique and sorted
        periods = np.unique(np.maximum(periods, 2))
        return periods

    def _calc_pressure_score(self, close, ema_matrix, n):
        """计算距离加权的压力得分
        ema_matrix: shape (num_lines, n)
        返回: pressure array of shape (n,), range [-1, +1]
        """
        pressure = np.zeros(n)
        for i in range(n):
            price = close[i]
            weighted_sum = 0.0
            weight_total = 0.0

            for j in range(ema_matrix.shape[0]):
                ema_val = ema_matrix[j, i]
                if ema_val == 0:
                    continue

                # Sign: price above EMA = +1, below = -1
                sign = 1.0 if price > ema_val else -1.0

                # Distance-based weight: closer EMA gets higher weight
                # Use inverse distance weighting
                dist = abs(price - ema_val)
                # Normalize by ATR-like measure (use price as scale)
                norm_dist = dist / (abs(price) + 1e-10)
                # Weight = 1 / (1 + dist), so closer = heavier
                weight = 1.0 / (1.0 + norm_dist * 100)

                weighted_sum += sign * weight
                weight_total += weight

            if weight_total > 0:
                pressure[i] = weighted_sum / weight_total
            else:
                pressure[i] = 0.0

        # Clamp to [-1, +1]
        pressure = np.clip(pressure, -1.0, 1.0)
        return pressure

    def _determine_target_state(self, pressure_val):
        """根据压力得分确定目标状态 (数学边界)
        pressure > 0.5  → BULLISH
        pressure < -0.5 → BEARISH
        else            → NEUTRAL
        """
        if pressure_val > 0.5:
            return self.BULLISH
        elif pressure_val < -0.5:
            return self.BEARISH
        else:
            return self.NEUTRAL

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate_full(self, data):
        """完整评估: 计算所有bar的latch状态序列
        返回: (latch_states, pressure_scores, atr_val) or None
        """
        min_len = self.slow_period + self.confirmation_bars + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ATR
        atr_val = self._calc_atr(high, low, close, n)
        if atr_val <= 0:
            return None

        # Compute all EMA ribbons
        periods = self._generate_ema_periods()
        num_ema = len(periods)
        ema_matrix = np.zeros((num_ema, n))
        for j, p in enumerate(periods):
            ema_matrix[j] = self._calc_ema(close, p)

        # Compute pressure scores
        pressure = self._calc_pressure_score(close, ema_matrix, n)

        # Latch state engine with confirmation
        latch_states = np.zeros(n, dtype=int)
        current_latch = self.NEUTRAL
        confirm_count = 0

        for i in range(n):
            target = self._determine_target_state(pressure[i])

            if target == current_latch:
                confirm_count = 0
                latch_states[i] = current_latch
            elif target != current_latch:
                # Need confirmation: target must persist for N bars
                if i > 0 and self._determine_target_state(pressure[i - 1]) == target:
                    confirm_count += 1
                else:
                    confirm_count = 1

                if confirm_count >= self.confirmation_bars:
                    current_latch = target
                    confirm_count = 0

                latch_states[i] = current_latch
            else:
                latch_states[i] = current_latch

        return latch_states, pressure, atr_val

    def _evaluate_last(self, data):
        """快速评估最后一根bar的状态变化
        返回: (state_changed, direction, atr_val) or None
        direction: 1=bullish transition, -1=bearish transition
        """
        result = self._evaluate_full(data)
        if result is None:
            return None

        latch_states, pressure, atr_val = result
        n = len(latch_states)

        if n < self.confirmation_bars + 1:
            return None

        current_state = latch_states[-1]
        prev_state = latch_states[-(self.confirmation_bars + 1)]

        # Detect state transition
        if current_state == self.BULLISH and prev_state == self.BEARISH:
            return True, 1, atr_val
        elif current_state == self.BULLISH and prev_state == self.NEUTRAL:
            return True, 1, atr_val
        elif current_state == self.BEARISH and prev_state == self.BULLISH:
            return True, -1, atr_val
        elif current_state == self.BEARISH and prev_state == self.NEUTRAL:
            return True, -1, atr_val

        return False, 0, atr_val

    # ------------------------------------------------------------------
    # Signal generation (backtest)
    # ------------------------------------------------------------------

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

        # Pre-compute per-symbol latch states for efficiency
        symbol_latch = {}
        symbol_atr = {}
        for sym in symbols:
            sym_data = data[data['symbol'] == sym].sort_index()
            if len(sym_data) >= self.slow_period + self.confirmation_bars + 10:
                result = self._evaluate_full(sym_data)
                if result is not None:
                    latch_states, pressure, atr_val = result
                    sym_times = sym_data.index
                    symbol_latch[sym] = dict(zip(sym_times, latch_states))
                    symbol_atr[sym] = atr_val

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_sym = None
                best_dir = 0
                best_pressure = -2.0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    if sym not in symbol_latch:
                        continue

                    latch_dict = symbol_latch[sym]
                    # Get current and recent latch states
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    hist_times = sorted(hist.index)
                    if len(hist_times) < self.confirmation_bars + 1:
                        continue

                    current_state = latch_dict.get(current_time, self.NEUTRAL)
                    prev_idx = hist_times[-(self.confirmation_bars + 1)]
                    prev_state = latch_dict.get(prev_idx, self.NEUTRAL)

                    # Detect transition
                    transition_dir = 0
                    if current_state == self.BULLISH and prev_state != self.BULLISH:
                        transition_dir = 1
                    elif current_state == self.BEARISH and prev_state != self.BEARISH:
                        transition_dir = -1

                    if transition_dir != 0:
                        price = float(bar['close'])
                        # Prefer stronger transitions
                        if abs(transition_dir) > abs(best_dir) or (
                            abs(transition_dir) == abs(best_dir) and best_pressure < 0
                        ):
                            best_sym = sym
                            best_dir = transition_dir
                            best_pressure = transition_dir

                if best_sym and best_dir != 0:
                    price = float(current_bars[current_bars['symbol'] == best_sym].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym, price)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym, price)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0

            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                if days_held >= self.hold_min:
                    bar_data = current_bars[current_bars['symbol'] == current_holding]
                    if len(bar_data) == 0:
                        continue
                    current_price = float(bar_data.iloc[0]['close'])

                    # Track high/low water mark
                    if position_dir == 1:
                        high_water = max(high_water, current_price)
                    else:
                        high_water = min(high_water, current_price) if high_water > 0 else current_price

                    # ATR trailing stop
                    hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                    atr_val = self._calc_atr(
                        hist['high'].values, hist['low'].values,
                        hist['close'].values, len(hist)
                    )
                    stop_hit = False

                    if atr_val > 0 and high_water > 0:
                        if position_dir == 1 and current_price < high_water - self.trail_atr_mult * atr_val:
                            stop_hit = True
                        elif position_dir == -1 and current_price > high_water + self.trail_atr_mult * atr_val:
                            stop_hit = True

                    # Max hold
                    if days_held >= self.max_hold:
                        stop_hit = True

                    # Latch-based exit: check if state reversed
                    latch_exit = False
                    if current_holding in symbol_latch:
                        latch_dict = symbol_latch[current_holding]
                        current_state = latch_dict.get(current_time, self.NEUTRAL)
                        if position_dir == 1 and current_state == self.BEARISH:
                            latch_exit = True
                        elif position_dir == -1 and current_state == self.BULLISH:
                            latch_exit = True

                    if stop_hit or latch_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding, current_price)
                        else:
                            self._record_signal(current_time, 'buy', current_holding, current_price)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0

        print(f"AlgionicsRibbon: 生成 {len(self.signals)} 个信号")
        return self.signals

    # ------------------------------------------------------------------
    # Real-time screening
    # ------------------------------------------------------------------

    def screen(self):
        data = self.data.copy()
        min_len = self.slow_period + self.confirmation_bars + 10
        if len(data) < min_len:
            return {'action': 'hold', 'reason': '数据不足(需要200+bars)', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate_full(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        latch_states, pressure, atr_val = result
        n = len(latch_states)
        current_state = latch_states[-1]
        prev_state = latch_states[-(self.confirmation_bars + 1)] if n > self.confirmation_bars else self.NEUTRAL
        current_pressure = pressure[-1]

        # State transition detection
        if current_state == self.BULLISH and prev_state != self.BULLISH:
            return {
                'action': 'buy',
                'reason': f"bear→bull latch, pressure={current_pressure:.3f}, atr={atr_val:.2f}",
                'price': price,
            }
        elif current_state == self.BEARISH and prev_state != self.BEARISH:
            return {
                'action': 'sell',
                'reason': f"bull→bear latch, pressure={current_pressure:.3f}, atr={atr_val:.2f}",
                'price': price,
            }

        state_names = {self.BULLISH: 'bull', self.NEUTRAL: 'neutral', self.BEARISH: 'bear'}
        return {
            'action': 'hold',
            'reason': f"latch={state_names.get(current_state, '?')}, pressure={current_pressure:.3f}",
            'price': price,
        }
