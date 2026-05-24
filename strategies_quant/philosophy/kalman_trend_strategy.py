"""
Kalman Trend Following Strategy (道法自然 - follow the way)
============================================================
2D Kalman filter with state [price, velocity] tracks price dynamics.
Kalman gain adapts to uncertainty — 上善若水, adapt like water.

Core logic:
  1. State vector: x = [price, velocity], transition F = [[1, dt],[0, 1]]
  2. Observation: H = [1, 0] — we only observe price
  3. Process noise Q scaled by ATR (volatility-adaptive)
  4. Measurement noise R = variance of recent returns
  5. Signal: velocity crosses zero with magnitude confirmation
  6. Kaufman ER gate: only trade when ER > 0.2 (trending market)
  7. ATR adaptive trailing stop using Kalman-estimated volatility
  8. Score: |velocity| / velocity_std * ER

Technical: Kalman Filter, Kaufman ER, ATR Trailing Stop
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class KalmanTrendStrategy(BaseStrategy):
    """Kalman filter trend following — adapt like water"""

    strategy_description = (
        "Kalman趋势: 2D卡尔曼滤波器跟踪价格速度, "
        "Kaufman ER门控 + ATR自适应追踪止损"
    )
    strategy_category = "trend_following"
    strategy_params_schema = {
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "er_period": {"type": "int", "default": 20, "label": "Kaufman ER周期"},
        "er_threshold": {"type": "float", "default": 0.2, "label": "ER门控阈值"},
        "q_scale": {"type": "float", "default": 0.01, "label": "过程噪声缩放"},
        "vel_threshold": {"type": "float", "default": 0.8, "label": "速度确认阈值(×std)"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "warmup": {"type": "int", "default": 30, "label": "预热期"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)

    def get_default_params(self):
        return {
            "atr_period": 14,
            "er_period": 20,
            "er_threshold": 0.2,
            "q_scale": 0.01,
            "vel_threshold": 0.8,
            "trail_atr_mult": 2.5,
            "hold_min": 3,
            "warmup": 30,
        }

    # ------------------------------------------------------------------
    # Core indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_atr(high, low, close, period):
        """Compute ATR from numpy arrays."""
        n = len(close)
        if n < period + 1:
            return np.nan
        tr = np.empty(n - 1)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = np.empty(n)
        atr[:period] = np.nan
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
        return atr

    @staticmethod
    def _kaufman_er(close, period):
        """Kaufman Efficiency Ratio: |direction| / sum_of_abs_changes."""
        n = len(close)
        er = np.full(n, np.nan)
        if n < period + 1:
            return er
        for i in range(period, n):
            direction = abs(close[i] - close[i - period])
            volatility = np.sum(np.abs(np.diff(close[i - period : i + 1])))
            er[i] = direction / volatility if volatility > 0 else 0.0
        return er

    @staticmethod
    def _run_kalman(close, atr, q_scale, warmup):
        """
        Run 2D Kalman filter on close prices.

        State:  x = [price, velocity]
        Transition:  F = [[1, dt], [0, 1]]   dt = 1 (daily)
        Observation: H = [1, 0]

        Process noise Q scaled by ATR, measurement noise R = var(recent returns).
        Returns arrays: est_price, est_velocity, kalman_gain.
        """
        n = len(close)
        est_price = np.full(n, np.nan)
        est_vel = np.full(n, np.nan)
        kg_arr = np.full(n, np.nan)

        # --- initialise from first two valid prices ---
        valid_mask = ~np.isnan(close)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < 2:
            return est_price, est_vel, kg_arr

        start = valid_idx[0]
        x = np.array([close[start], 0.0])  # [price, velocity]
        P = np.array([[1.0, 0.0], [0.0, 1.0]])  # covariance

        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([1.0, 0.0])

        # measurement noise from returns variance (will be updated online)
        ret_var = max(np.nanvar(np.diff(close[: min(20, n)])), 1e-10)

        for i in range(start, n):
            if np.isnan(close[i]) or np.isnan(atr[i]):
                continue

            # --- adaptive noise ---
            atr_val = atr[i] if not np.isnan(atr[i]) else 1.0
            q_val = q_scale * atr_val
            Q = np.array([[q_val, 0.0], [0.0, q_val * 0.1]])

            # measurement noise: use recent return variance, clamped
            if i >= 20:
                window = close[i - 20 : i]
                window = window[~np.isnan(window)]
                if len(window) > 2:
                    ret_var = max(np.var(np.diff(window)), 1e-10)
            R = ret_var

            # --- predict ---
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # --- update ---
            y = close[i] - H @ x_pred  # innovation
            S = H @ P_pred @ H + R  # innovation covariance (scalar)
            K = P_pred @ H / S  # Kalman gain (2x1)

            x = x_pred + K * y
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            # symmetrise P to avoid numerical drift
            P = (P + P.T) / 2.0

            est_price[i] = x[0]
            est_vel[i] = x[1]
            kg_arr[i] = K[0]  # price-component gain

        return est_price, est_vel, kg_arr

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self):
        data = self.data.copy()
        if "symbol" not in data.columns:
            data["symbol"] = "DEFAULT"

        atr_period = self.params["atr_period"]
        er_period = self.params["er_period"]
        er_threshold = self.params["er_threshold"]
        q_scale = self.params["q_scale"]
        vel_threshold = self.params["vel_threshold"]
        trail_atr_mult = self.params["trail_atr_mult"]
        hold_min = self.params["hold_min"]
        warmup = self.params["warmup"]

        close = data["close"].values.astype(float)
        high = data["high"].values.astype(float)
        low = data["low"].values.astype(float)
        n = len(close)

        if n < max(warmup, atr_period + 1, er_period + 1) + 5:
            return self.signals

        # --- indicators ---
        atr = self._calc_atr(high, low, close, atr_period)
        er = self._kaufman_er(close, er_period)
        _, velocity, _ = self._run_kalman(close, atr, q_scale, warmup)

        # velocity statistics (rolling)
        vel_std = pd.Series(velocity).rolling(warmup, min_periods=10).std().values

        # --- trading loop ---
        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        position_dir = 0
        buy_time = None
        high_water = 0.0
        low_water = float("inf")

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            # find row index for this timestamp in the numpy arrays
            idx_positions = np.where(data.index == current_time)[0]
            if len(idx_positions) == 0:
                continue
            idx = idx_positions[0]

            if idx < warmup or np.isnan(velocity[idx]) or np.isnan(er[idx]):
                continue

            # --- no position: look for entry ---
            if current_holding is None:
                best_score = 0.0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar["symbol"]

                    vel_now = velocity[idx]
                    vel_prev = velocity[idx - 1] if idx > 0 and not np.isnan(velocity[idx - 1]) else 0.0
                    v_std = vel_std[idx]
                    er_now = er[idx]

                    if v_std <= 0 or np.isnan(v_std):
                        continue

                    # Kaufman ER gate
                    if er_now < er_threshold:
                        continue

                    vel_norm = vel_now / v_std  # normalised velocity

                    # zero-crossing detection
                    crossed_up = vel_prev <= 0 < vel_now
                    crossed_dn = vel_prev >= 0 > vel_now

                    # magnitude confirmation
                    mag_ok = abs(vel_norm) > vel_threshold

                    score = 0.0
                    direction = 0

                    if crossed_up and mag_ok:
                        direction = 1
                        score = abs(vel_norm) * er_now
                    elif crossed_dn and mag_ok:
                        direction = -1
                        score = abs(vel_norm) * er_now
                    # also allow re-entry on strong sustained velocity
                    elif vel_now > 0 and mag_ok and vel_norm > 1.5:
                        direction = 1
                        score = vel_norm * er_now * 0.5
                    elif vel_now < 0 and mag_ok and vel_norm < -1.5:
                        direction = -1
                        score = abs(vel_norm) * er_now * 0.5

                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= 0.3:
                    action = "buy" if best_dir == 1 else "sell"
                    self._record_signal(
                        current_time, action, best_sym,
                        price=float(close[idx]),
                        score=round(best_score, 4),
                        velocity=round(float(velocity[idx]), 4),
                        er=round(float(er[idx]), 4),
                    )
                    current_holding = best_sym
                    position_dir = best_dir
                    buy_time = current_time
                    high_water = float(close[idx])
                    low_water = float(close[idx])

            # --- in position: manage exit ---
            else:
                bar_data = current_bars[current_bars["symbol"] == current_holding]
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
                atr_val = atr[idx]
                if not np.isnan(atr_val) and atr_val > 0 and days_held >= hold_min:
                    if position_dir == 1:
                        if current_price < high_water - trail_atr_mult * atr_val:
                            should_exit = True
                    else:
                        if current_price > low_water + trail_atr_mult * atr_val:
                            should_exit = True

                # velocity reversal exit
                if idx >= warmup and not np.isnan(velocity[idx]):
                    v_std = vel_std[idx]
                    if v_std > 0 and days_held >= hold_min:
                        vel_norm = velocity[idx] / v_std
                        if position_dir == 1 and vel_norm < -vel_threshold:
                            should_exit = True
                        elif position_dir == -1 and vel_norm > vel_threshold:
                            should_exit = True

                # max holding period
                if days_held >= 120:
                    should_exit = True

                if should_exit:
                    action = "sell" if position_dir == 1 else "buy"
                    self._record_signal(
                        current_time, action, current_holding,
                        price=current_price,
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
        if n < warmup + 5:
            return {
                "action": "hold",
                "reason": "数据不足",
                "price": float(data["close"].iloc[-1]),
            }

        close = data["close"].values.astype(float)
        high = data["high"].values.astype(float)
        low = data["low"].values.astype(float)

        atr = self._calc_atr(high, low, close, self.params["atr_period"])
        er = self._kaufman_er(close, self.params["er_period"])
        _, velocity, _ = self._run_kalman(close, atr, self.params["q_scale"], warmup)
        vel_std = pd.Series(velocity).rolling(warmup, min_periods=10).std().values

        idx = n - 1
        price = float(close[idx])

        if np.isnan(velocity[idx]) or np.isnan(er[idx]) or np.isnan(vel_std[idx]):
            return {"action": "hold", "reason": "指标未就绪", "price": price}

        vel_now = velocity[idx]
        vel_prev = velocity[idx - 1] if idx > 0 and not np.isnan(velocity[idx - 1]) else 0.0
        v_std = vel_std[idx]
        er_now = er[idx]

        if v_std <= 0 or er_now < self.params["er_threshold"]:
            return {"action": "hold", "reason": f"ER={er_now:.2f} 不足或v_std<=0", "price": price}

        vel_norm = vel_now / v_std
        crossed_up = vel_prev <= 0 < vel_now
        crossed_dn = vel_prev >= 0 > vel_now
        mag_ok = abs(vel_norm) > self.params["vel_threshold"]
        score = abs(vel_norm) * er_now

        if crossed_up and mag_ok and score >= 0.3:
            return {"action": "buy", "reason": f"Kalman vel={vel_norm:.2f} ER={er_now:.2f}", "price": price}
        if crossed_dn and mag_ok and score >= 0.3:
            return {"action": "sell", "reason": f"Kalman vel={vel_norm:.2f} ER={er_now:.2f}", "price": price}

        return {"action": "hold", "reason": f"vel={vel_norm:.2f} ER={er_now:.2f}", "price": price}
