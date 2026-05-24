"""
Kelly仓位集中策略 (Kelly Criterion Position Sizing Strategy)
============================================================
基于Kelly准则的自适应仓位管理 + 结构张力/VDP确认入场。

哲学: "中庸其至矣乎" — 黄金中庸之道;
     既不过于激进，也不过于保守，用数学确定最优仓位。

核心逻辑:
  1. 入场: 结构张力 > 0 且 delta > 0 = 看多;
     张力 < 0 且 delta < 0 = 看空
  2. Kelly仓位: 从最近50根K线信号条件收益计算胜率,
     f* = (p*b - q) / b, 使用半Kelly: f_actual = f* * 0.5,
     钳位到 [0.1, 0.5]
  3. 自适应Kelly: 根据Kaufman效率比(KER)调整仓位:
     KER > 0.3 → 趋势市 f*1.2 (最高60%);
     KER < 0.15 → 震荡市 f*0.5;
     否则 → f 保持不变
  4. 出场: ATR追踪止损(2.0x) 或张力反转

技术指标: Structural Tension, Volume Delta, Kelly Criterion, KER, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class KellyConcentrationStrategy(BaseStrategy):
    """Kelly仓位集中策略 — Kelly准则仓位管理 + 结构张力/VDP入场 (中庸)"""

    strategy_description = (
        "Kelly仓位: Kelly准则自适应仓位 + 结构张力/VDP确认入场 "
        "— 中庸其至矣乎"
    )
    strategy_category = "risk"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "lookback": {"type": "int", "default": 50, "label": "Kelly回溯"},
        "ker_period": {"type": "int", "default": 20, "label": "KER周期"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "追踪止损ATR倍数"},
        "max_hold": {"type": "int", "default": 60, "label": "最大持仓天数"},
        "kelly_fraction": {"type": "float", "default": 0.5, "label": "Kelly分数(半Kelly)"},
        "min_position": {"type": "float", "default": 0.1, "label": "最小仓位(10%)"},
        "max_position": {"type": "float", "default": 0.5, "label": "最大仓位(50%)"},
        "delta_period": {"type": "int", "default": 10, "label": "Delta EMA周期"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.pivot_len = self.params.get('pivot_len', 5)
        self.atr_period = self.params.get('atr_period', 14)
        self.lookback = self.params.get('lookback', 50)
        self.ker_period = self.params.get('ker_period', 20)
        self.trail_atr_mult = self.params.get('trail_atr_mult', 2.0)
        self.max_hold = self.params.get('max_hold', 60)
        self.kelly_fraction = self.params.get('kelly_fraction', 0.5)
        self.min_position = self.params.get('min_position', 0.1)
        self.max_position = self.params.get('max_position', 0.5)
        self.delta_period = self.params.get('delta_period', 10)

    def get_default_params(self):
        return {
            'pivot_len': 5,
            'atr_period': 14,
            'lookback': 50,
            'ker_period': 20,
            'trail_atr_mult': 2.0,
            'max_hold': 60,
            'kelly_fraction': 0.5,
            'min_position': 0.1,
            'max_position': 0.5,
            'delta_period': 10,
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

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
                best_price = 0.0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[
                        (data['symbol'] == sym) & (data.index < current_time)
                    ]
                    result = self._evaluate_entry(hist)
                    if result is None:
                        continue
                    score, direction, price = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction
                        best_price = price

                if best_sym and abs(best_score) >= 3:
                    # Kelly position sizing
                    hist = data[
                        (data['symbol'] == best_sym)
                        & (data.index < current_time)
                    ]
                    position_pct = self._compute_kelly_position(hist)

                    if best_dir == 1:
                        self._record_signal(
                            current_time, 'buy', best_sym, best_price,
                            score=best_score,
                            position_pct=round(position_pct, 4),
                            kelly_method='half_kelly_regime_adjusted',
                        )
                        position_dir = 1
                    else:
                        self._record_signal(
                            current_time, 'sell', best_sym, best_price,
                            score=best_score,
                            position_pct=round(position_pct, 4),
                            kelly_method='half_kelly_regime_adjusted',
                        )
                        position_dir = -1

                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0
                    low_water = float('inf')

            else:
                days_held = len(
                    [t for t in unique_times if buy_time < t <= current_time]
                )
                bar_data = current_bars[
                    current_bars['symbol'] == current_holding
                ]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # Track water marks
                if position_dir == 1:
                    high_water = (
                        max(high_water, current_price)
                        if high_water > 0
                        else current_price
                    )
                else:
                    low_water = (
                        min(low_water, current_price)
                        if low_water < float('inf')
                        else current_price
                    )

                should_exit = False

                # --- ATR trailing stop ---
                hist = data[
                    (data['symbol'] == current_holding)
                    & (data.index < current_time)
                ]
                atr_val = self._calc_atr(hist)

                if atr_val > 0:
                    if (
                        position_dir == 1
                        and high_water > 0
                        and current_price < high_water - self.trail_atr_mult * atr_val
                    ):
                        should_exit = True
                    elif (
                        position_dir == -1
                        and low_water < float('inf')
                        and current_price > low_water + self.trail_atr_mult * atr_val
                    ):
                        should_exit = True

                # --- Max hold ---
                if days_held >= self.max_hold:
                    should_exit = True

                # --- Tension reversal exit ---
                if not should_exit and days_held >= 3:
                    result = self._evaluate_entry(hist)
                    if result is not None:
                        score, direction, _ = result
                        if position_dir == 1 and direction == -1 and score < -3:
                            should_exit = True
                        elif position_dir == -1 and direction == 1 and score > 3:
                            should_exit = True

                if should_exit:
                    if position_dir == 1:
                        self._record_signal(
                            current_time, 'sell', current_holding, current_price,
                            reason='kelly_trail_stop',
                        )
                    else:
                        self._record_signal(
                            current_time, 'buy', current_holding, current_price,
                            reason='kelly_trail_stop',
                        )
                    current_holding = None
                    buy_time = None
                    position_dir = 0
                    high_water = 0.0
                    low_water = float('inf')

        return self.signals

    # ------------------------------------------------------------------
    # Entry evaluation — structural tension + VDP confirmation
    # ------------------------------------------------------------------

    def _evaluate_entry(self, data):
        """
        Evaluate entry based on structural tension direction + delta confirmation.
        Returns (score, direction, current_price) or None.
        """
        min_bars = self.pivot_len * 2 + 20
        if len(data) < min_bars:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        swing_highs, swing_lows = self._find_pivots(high, low, n)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # 7 structural reference points
        prev_sh = swing_highs[-2][1]
        curr_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1]
        curr_sl = swing_lows[-1][1]
        inter_high = max(curr_sh, prev_sh)
        inter_low = min(curr_sl, prev_sl)
        inter_mid = (inter_high + inter_low) / 2.0

        current_price = close[-1]
        ref_points = [prev_sh, curr_sh, prev_sl, curr_sl, inter_high, inter_low, inter_mid]

        displacements = [(current_price - rp) / atr for rp in ref_points]
        tension = float(np.mean(displacements))

        # Compute tension delta (change from previous bar)
        tension_delta = 0.0
        if n >= 2:
            prev_price = close[-2]
            prev_disp = [(prev_price - rp) / atr for rp in ref_points]
            prev_tension = float(np.mean(prev_disp))
            tension_delta = tension - prev_tension

        # Compute volume delta pressure (VDP)
        vdp = self._calc_vdp(data)

        score = 0

        # --- Tension direction + delta confirmation ---
        # Bullish: tension > 0 AND delta > 0
        if tension > 0 and tension_delta > 0:
            score += 4
        elif tension > 0 and tension_delta <= 0:
            score += 1  # Positive tension but losing momentum
        # Bearish: tension < 0 AND delta < 0
        elif tension < 0 and tension_delta < 0:
            score -= 4
        elif tension < 0 and tension_delta >= 0:
            score -= 1  # Negative tension but recovering

        # --- VDP confirmation ---
        if abs(score) >= 4:
            if score > 0 and vdp > 0:
                score += 2  # VDP confirms bullish
            elif score < 0 and vdp < 0:
                score -= 2  # VDP confirms bearish
            elif score > 0 and vdp < 0:
                score -= 1  # VDP diverges, reduce confidence
            elif score < 0 and vdp > 0:
                score += 1  # VDP diverges, reduce confidence

        direction = 1 if score > 0 else -1
        return score, direction, current_price

    # ------------------------------------------------------------------
    # Kelly criterion position sizing
    # ------------------------------------------------------------------

    def _compute_kelly_position(self, data):
        """
        Compute Kelly criterion position size with regime adjustment.

        1. Rolling win rate from last `lookback` bars
        2. Kelly fraction: f* = (p*b - q) / b
        3. Half Kelly: f_actual = f* * 0.5
        4. Clamp to [min_position, max_position]
        5. Adjust by KER regime
        """
        close = data['close'].values
        n = len(close)

        if n < 30:
            return self.min_position

        # Use recent returns for Kelly estimation
        returns = np.diff(close) / close[:-1]
        lookback_returns = returns[-self.lookback:] if len(returns) >= self.lookback else returns

        if len(lookback_returns) < 10:
            return self.min_position

        # Win rate and average win/loss
        wins = lookback_returns[lookback_returns > 0]
        losses = lookback_returns[lookback_returns < 0]

        total = len(wins) + len(losses)
        if total < 5:
            return self.min_position

        p = len(wins) / total  # win rate
        q = 1.0 - p           # loss rate

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0

        # Edge case: no wins or no losses
        if avg_loss < 1e-10:
            return self.max_position  # All wins, max confidence
        if avg_win < 1e-10:
            return self.min_position  # No wins, minimum position

        b = avg_win / avg_loss  # win/loss ratio

        # Kelly fraction: f* = (p*b - q) / b
        kelly_f = (p * b - q) / b

        # Guard: negative Kelly means no edge
        if kelly_f <= 0:
            return self.min_position

        # Half Kelly
        f_actual = kelly_f * self.kelly_fraction

        # Clamp to [min, max]
        f_actual = max(self.min_position, min(self.max_position, f_actual))

        # --- Regime adjustment via KER ---
        ker = self._calc_ker(data)

        if ker > 0.3:
            # Trending market: increase position up to 60%
            f_actual *= 1.2
            f_actual = min(f_actual, 0.6)
        elif ker < 0.15:
            # Ranging market: decrease position
            f_actual *= 0.5
            f_actual = max(f_actual, self.min_position)
        # else: keep as is

        return f_actual

    # ------------------------------------------------------------------
    # Kaufman Efficiency Ratio (KER)
    # ------------------------------------------------------------------

    def _calc_ker(self, data):
        """
        Kaufman Efficiency Ratio = |Net Change| / Sum of Absolute Changes
        over ker_period bars.
        """
        close = data['close'].values
        n = len(close)
        period = self.ker_period

        if n < period + 1:
            return 0.0

        recent = close[-(period + 1):]
        net_change = abs(recent[-1] - recent[0])
        sum_abs_changes = np.sum(np.abs(np.diff(recent)))

        if sum_abs_changes < 1e-12:
            return 0.0

        return net_change / sum_abs_changes

    # ------------------------------------------------------------------
    # Volume Delta Pressure (VDP)
    # ------------------------------------------------------------------

    def _calc_vdp(self, data):
        """
        Compute normalized volume delta pressure over delta_period bars.
        Returns a value in roughly [-1, 1].
        """
        close = data['close'].values
        open_ = data['open'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        vol_col = self._volume_col(data)
        if vol_col is None or n < self.delta_period + 1:
            # No volume data: fall back to price direction only
            if n >= 2:
                return 1.0 if close[-1] > close[-2] else (-1.0 if close[-1] < close[-2] else 0.0)
            return 0.0

        vol = data[vol_col].values
        period = min(self.delta_period, n)
        deltas = np.zeros(period)

        for i in range(period):
            idx = n - period + i
            total_range = high[idx] - low[idx]
            if total_range < 1e-12:
                deltas[i] = 0.0
                continue
            body_ratio = abs(close[idx] - open_[idx]) / total_range
            direction = 1.0 if close[idx] > open_[idx] else (-1.0 if close[idx] < open_[idx] else 0.0)
            deltas[i] = direction * body_ratio * vol[idx]

        total_vol = np.sum(vol[n - period:])
        if total_vol < 1e-12:
            return 0.0

        return float(np.sum(deltas) / total_vol)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _find_pivots(self, high, low, n):
        """Identify swing highs and swing lows."""
        piv_len = self.pivot_len
        if n < piv_len * 2 + 1:
            return [], []
        swing_highs = []
        swing_lows = []
        for i in range(piv_len, n - piv_len):
            is_high = all(
                high[i] >= high[i + j]
                for j in range(-piv_len, piv_len + 1) if j != 0
            )
            is_low = all(
                low[i] <= low[i + j]
                for j in range(-piv_len, piv_len + 1) if j != 0
            )
            if is_high:
                swing_highs.append((i, high[i]))
            if is_low:
                swing_lows.append((i, low[i]))
        return swing_highs, swing_lows

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        return float(np.mean(tr[-self.atr_period:]))

    @staticmethod
    def _volume_col(data):
        """Return the volume column name, or None if absent."""
        if 'volume' in data.columns:
            return 'volume'
        if 'vol' in data.columns:
            return 'vol'
        return None

    # ------------------------------------------------------------------
    # Real-time screening
    # ------------------------------------------------------------------

    def screen(self):
        data = self.data.copy()
        if len(data) < 40:
            return {
                'action': 'hold',
                'reason': '数据不足',
                'price': float(data['close'].iloc[-1]),
            }

        result = self._evaluate_entry(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            position_pct = self._compute_kelly_position(data)
            ker = self._calc_ker(data)
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f'score={score} kelly={position_pct:.1%} ker={ker:.2f}',
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
