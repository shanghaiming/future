"""
Wasserstein Guard Strategy (穷则变 - when exhausted, change)
============================================================
Wasserstein-1 distance regime change detection.

Core logic:
  1. Rolling W1 distance between current (20-day) and reference (120-day) return distributions
  2. W1 = integral |CDF_current(x) - CDF_reference(x)| dx  (sorted arrays)
  3. When W1 > 95th percentile of historical W1 -> regime change detected
  4. During regime change: exit all positions, wait for stabilization
  5. After stabilization: re-enter with structural tension direction
  6. Structural tension: 7-point displacement from swing anchors (pivot_len=5)
  7. ATR trailing stop
  8. Score: tension_direction * (1 + W1_surge_bonus)

Philosophy: 穷则变，变则通 — when the old regime exhausts itself, change to find flow.
Technical: Wasserstein-1 Distance, Structural Tension, ATR Trailing Stop
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class WassersteinGuardStrategy(BaseStrategy):
    """Wasserstein distance regime change detection — change to find flow"""

    strategy_description = (
        "Wasserstein守卫: W1分布距离检测市场regime变化, "
        "regime稳定后以结构张力方向入场"
    )
    strategy_category = "regime"
    strategy_params_schema = {
        "current_window": {"type": "int", "default": 20, "label": "当前分布窗口"},
        "reference_window": {"type": "int", "default": 120, "label": "参考分布窗口"},
        "w1_percentile": {"type": "float", "default": 95.0, "label": "regime阈值百分位"},
        "stabilise_bars": {"type": "int", "default": 5, "label": "稳定等待K线数"},
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "warmup": {"type": "int", "default": 150, "label": "预热期"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)

    def get_default_params(self):
        return {
            "current_window": 20,
            "reference_window": 120,
            "w1_percentile": 95.0,
            "stabilise_bars": 5,
            "pivot_len": 5,
            "atr_period": 14,
            "trail_atr_mult": 2.5,
            "hold_min": 3,
            "warmup": 150,
        }

    # ------------------------------------------------------------------
    # Core indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_atr(high, low, close, period):
        """Compute ATR from numpy arrays."""
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = np.full(n, np.nan)
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
        return atr

    @staticmethod
    def _wasserstein_1(x, y):
        """
        Wasserstein-1 distance (Earth Mover's Distance) between two 1D distributions.
        For 1D: W1 = integral |CDF_x(t) - CDF_y(t)| dt
        Computed via sorted arrays: mean of absolute differences of sorted samples.
        """
        x = np.sort(x[~np.isnan(x)])
        y = np.sort(y[~np.isnan(y)])
        if len(x) == 0 or len(y) == 0:
            return np.nan

        # Use quantile-based integration for unequal sample sizes
        n_quantiles = max(len(x), len(y))
        quantiles = np.linspace(0, 1, n_quantiles)
        q_x = np.quantile(x, quantiles)
        q_y = np.quantile(y, quantiles)
        w1 = np.mean(np.abs(q_x - q_y))
        return w1

    def _rolling_w1(self, returns, current_window, reference_window):
        """
        Compute rolling W1 distance between current and reference return distributions.
        Returns array of same length as returns.
        """
        n = len(returns)
        w1_arr = np.full(n, np.nan)
        min_start = reference_window + current_window
        if n < min_start:
            return w1_arr

        for i in range(min_start - 1, n):
            current_returns = returns[i - current_window + 1 : i + 1]
            reference_returns = returns[i - reference_window - current_window + 1 : i - current_window + 1]

            # ensure we have enough data
            cur_valid = current_returns[~np.isnan(current_returns)]
            ref_valid = reference_returns[~np.isnan(reference_returns)]
            if len(cur_valid) < 5 or len(ref_valid) < 10:
                continue

            w1_arr[i] = self._wasserstein_1(cur_valid, ref_valid)

        return w1_arr

    @staticmethod
    def _find_pivots(high, low, n, pivot_len):
        """Find swing highs and lows."""
        if n < pivot_len * 2 + 1:
            return [], []
        swing_highs = []
        swing_lows = []
        for i in range(pivot_len, n - pivot_len):
            is_high = all(
                high[i] >= high[i + j]
                for j in range(-pivot_len, pivot_len + 1)
                if j != 0
            )
            is_low = all(
                low[i] <= low[i + j]
                for j in range(-pivot_len, pivot_len + 1)
                if j != 0
            )
            if is_high:
                swing_highs.append((i, high[i]))
            if is_low:
                swing_lows.append((i, low[i]))
        return swing_highs, swing_lows

    def _structural_tension(self, close, high, low, idx, atr_val):
        """
        Compute 7-point structural tension score at index idx.
        Returns (score, direction) or None if insufficient data.
        """
        pivot_len = self.params["pivot_len"]
        n = idx + 1
        if n < pivot_len * 2 + 5 or atr_val <= 0:
            return None

        h = high[:n]
        lo = low[:n]
        swing_highs, swing_lows = self._find_pivots(h, lo, n, pivot_len)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        prev_sh = swing_highs[-2][1]
        curr_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1]
        curr_sl = swing_lows[-1][1]

        inter_high = max(curr_sh, prev_sh)
        inter_low = min(curr_sl, prev_sl)
        inter_mid = (inter_high + inter_low) / 2.0

        current_price = close[idx]

        ref_points = [prev_sh, curr_sh, prev_sl, curr_sl, inter_high, inter_low, inter_mid]
        displacements = [(current_price - rp) / atr_val for rp in ref_points]
        tension = np.mean(displacements)

        # score the tension
        score = 0.0
        if tension > 1.0:
            score = 5.0
        elif tension > 0.5:
            score = 3.0
        elif tension > 0.2:
            score = 1.0
        elif tension < -1.0:
            score = -5.0
        elif tension < -0.5:
            score = -3.0
        elif tension < -0.2:
            score = -1.0

        direction = 1 if score > 0 else -1
        return score, direction

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self):
        data = self.data.copy()
        if "symbol" not in data.columns:
            data["symbol"] = "DEFAULT"

        warmup = self.params["warmup"]
        current_window = self.params["current_window"]
        reference_window = self.params["reference_window"]
        w1_percentile = self.params["w1_percentile"]
        stabilise_bars = self.params["stabilise_bars"]
        trail_atr_mult = self.params["trail_atr_mult"]
        hold_min = self.params["hold_min"]
        atr_period = self.params["atr_period"]

        close = data["close"].values.astype(float)
        high = data["high"].values.astype(float)
        low = data["low"].values.astype(float)
        n = len(close)

        min_required = max(warmup, reference_window + current_window + 20)
        if n < min_required:
            return self.signals

        # --- compute indicators ---
        atr = self._calc_atr(high, low, close, atr_period)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(close) / close[:-1]

        w1_dist = self._rolling_w1(returns, current_window, reference_window)

        # regime threshold: rolling percentile of historical W1
        w1_series = pd.Series(w1_dist)
        w1_rolling_p95 = w1_series.rolling(
            reference_window, min_periods=30
        ).quantile(w1_percentile / 100.0).values

        # --- trading state ---
        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        position_dir = 0
        buy_time = None
        high_water = 0.0
        low_water = float("inf")

        # regime state machine
        in_regime_change = False
        regime_change_bar = 0  # bar index when regime change was detected
        # number of bars since a regime change was flagged
        bars_since_regime = 0

        for current_time in unique_times:
            idx_positions = np.where(data.index == current_time)[0]
            if len(idx_positions) == 0:
                continue
            idx = idx_positions[0]

            if idx < warmup:
                continue

            w1_now = w1_dist[idx]
            w1_thresh = w1_rolling_p95[idx]

            # skip if indicators not ready
            if np.isnan(w1_now) or np.isnan(w1_thresh):
                continue

            atr_val = atr[idx]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # --- Regime change detection ---
            if w1_now > w1_thresh:
                if not in_regime_change:
                    in_regime_change = True
                    regime_change_bar = idx
                bars_since_regime = idx - regime_change_bar

                # If in a position during regime change -> EXIT
                if current_holding is not None:
                    action = "sell" if position_dir == 1 else "buy"
                    price = float(close[idx])
                    self._record_signal(
                        current_time, action, current_holding,
                        price=price,
                        reason="regime_change_exit",
                        w1=round(float(w1_now), 6),
                    )
                    current_holding = None
                    position_dir = 0
                    buy_time = None
                    high_water = 0.0
                    low_water = float("inf")
                continue  # wait for stabilization

            # --- Stabilization check ---
            if in_regime_change:
                bars_since_regime = idx - regime_change_bar
                if bars_since_regime < stabilise_bars:
                    continue  # still waiting
                # regime has stabilized — fall through to normal entry logic
                in_regime_change = False

            # --- No position: evaluate entry ---
            if current_holding is None:
                current_bars = data.loc[current_time]
                if isinstance(current_bars, pd.Series):
                    current_bars = pd.DataFrame([current_bars])

                best_score = 0.0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar["symbol"]

                    tension_result = self._structural_tension(
                        close, high, low, idx, atr_val
                    )
                    if tension_result is None:
                        continue

                    t_score, direction = tension_result
                    if abs(t_score) < 3:
                        continue

                    # W1 surge bonus: elevated but stabilizing W1 gives bonus
                    w1_median = np.nanmedian(w1_dist[max(0, idx - reference_window):idx])
                    if not np.isnan(w1_median) and w1_median > 0:
                        w1_surge = w1_now / w1_median
                    else:
                        w1_surge = 1.0

                    bonus = max(1.0, 1.0 + (w1_surge - 1.0) * 0.5) if w1_surge > 1.0 else 1.0
                    final_score = t_score * bonus

                    if abs(final_score) > abs(best_score):
                        best_score = final_score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= 2.0:
                    action = "buy" if best_dir == 1 else "sell"
                    self._record_signal(
                        current_time, action, best_sym,
                        price=float(close[idx]),
                        score=round(best_score, 4),
                        w1=round(float(w1_now), 6),
                        reason="tension_entry",
                    )
                    current_holding = best_sym
                    position_dir = best_dir
                    buy_time = current_time
                    high_water = float(close[idx])
                    low_water = float(close[idx])

            # --- In position: manage exit ---
            else:
                bar_data = data.loc[current_time]
                if isinstance(bar_data, pd.Series):
                    bar_data = pd.DataFrame([bar_data])
                bar_data = bar_data[bar_data["symbol"] == current_holding] if "symbol" in bar_data.columns else bar_data
                if len(bar_data) == 0:
                    continue

                current_price = float(close[idx])
                days_held = len(
                    [t for t in unique_times if buy_time < t <= current_time]
                )

                if position_dir == 1:
                    high_water = max(high_water, current_price)
                else:
                    low_water = min(low_water, current_price)

                should_exit = False

                # ATR trailing stop
                if days_held >= hold_min and atr_val > 0:
                    if position_dir == 1:
                        if current_price < high_water - trail_atr_mult * atr_val:
                            should_exit = True
                    else:
                        if current_price > low_water + trail_atr_mult * atr_val:
                            should_exit = True

                # structural tension reversal exit
                if days_held >= hold_min:
                    tension_result = self._structural_tension(
                        close, high, low, idx, atr_val
                    )
                    if tension_result is not None:
                        t_score, direction = tension_result
                        if position_dir == 1 and direction == -1 and t_score < -3:
                            should_exit = True
                        elif position_dir == -1 and direction == 1 and t_score > 3:
                            should_exit = True

                # max holding period
                if days_held >= 120:
                    should_exit = True

                if should_exit:
                    action = "sell" if position_dir == 1 else "buy"
                    self._record_signal(
                        current_time, action, current_holding,
                        price=current_price,
                        reason="trailing_stop_or_reversal",
                    )
                    current_holding = None
                    position_dir = 0
                    buy_time = None
                    high_water = 0.0
                    low_water = float("inf")

        return self.signals

    # ------------------------------------------------------------------
    # Fast screen for live filtering
    # ------------------------------------------------------------------

    def screen(self):
        """Real-time screening based on the latest bar."""
        data = self.data
        n = len(data)
        warmup = self.params["warmup"]

        if n < warmup:
            return {
                "action": "hold",
                "reason": "数据不足",
                "price": float(data["close"].iloc[-1]),
            }

        close = data["close"].values.astype(float)
        high = data["high"].values.astype(float)
        low = data["low"].values.astype(float)

        atr = self._calc_atr(high, low, close, self.params["atr_period"])
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(close) / close[:n - 1]  # Use n instead of len(close)

        cw = self.params["current_window"]
        rw = self.params["reference_window"]
        w1_dist = self._rolling_w1(returns, cw, rw)

        idx = n - 1
        price = float(close[idx])

        w1_now = w1_dist[idx]
        if np.isnan(w1_now):
            return {"action": "hold", "reason": "W1未就绪", "price": price}

        # check regime
        w1_series = pd.Series(w1_dist)
        w1_p95 = w1_series.rolling(rw, min_periods=30).quantile(
            self.params["w1_percentile"] / 100.0
        ).values
        w1_thresh = w1_p95[idx]

        if not np.isnan(w1_thresh) and w1_now > w1_thresh:
            return {"action": "hold", "reason": f"regime_change W1={w1_now:.6f}", "price": price}

        atr_val = atr[idx]
        if np.isnan(atr_val) or atr_val <= 0:
            return {"action": "hold", "reason": "ATR无效", "price": price}

        tension_result = self._structural_tension(close, high, low, idx, atr_val)
        if tension_result is None:
            return {"action": "hold", "reason": "张力不足", "price": price}

        t_score, direction = tension_result
        if abs(t_score) >= 3:
            action = "buy" if direction == 1 else "sell"
            return {
                "action": action,
                "reason": f"tension={t_score:.1f} W1={w1_now:.6f}",
                "price": price,
            }

        return {"action": "hold", "reason": f"tension={t_score:.1f}", "price": price}
