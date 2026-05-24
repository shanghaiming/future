"""
Kalman VDP Fusion Strategy (卡尔曼+量价融合策略)
==================================================
Fuses Kalman filter state estimation with VDP volume confirmation.
Kalman adapts like water — velocity crossover = trend change.
VDP confirms with volume pressure.

Math: Kalman[state=price,velocity], KER gate, Shannon entropy filter
Philosophy: 上善若水 — adapts like water to market conditions
Category: fusion
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from core.base_strategy import BaseStrategy


class KalmanVDPFusionStrategy(BaseStrategy):
    strategy_description = "KalmanVDP: Kalman velocity + VDP delta + KER gate + entropy filter"
    strategy_category = "fusion"
    strategy_params_schema = {
        "ker_period": {"type": "int", "default": 20, "label": "KER period"},
        "vdp_ema_period": {"type": "int", "default": 10, "label": "VDP EMA period"},
        "entropy_bins": {"type": "int", "default": 10, "label": "Entropy bins"},
        "entropy_window": {"type": "int", "default": 50, "label": "Entropy window"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "Trail stop ATR mult"},
        "max_hold": {"type": "int", "default": 40, "label": "Max holding days"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.ker_period = p.get('ker_period', 20)
        self.vdp_ema_period = p.get('vdp_ema_period', 10)
        self.entropy_bins = p.get('entropy_bins', 10)
        self.entropy_window = p.get('entropy_window', 50)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.0)
        self.max_hold = p.get('max_hold', 40)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ker_period': 20, 'vdp_ema_period': 10, 'entropy_bins': 10,
            'entropy_window': 50, 'atr_period': 14, 'hold_min': 3,
            'trail_atr_mult': 2.0, 'max_hold': 40,
        }

    def _kalman_filter(self, prices):
        """Simple 2-state Kalman filter: [price, velocity]."""
        n = len(prices)
        if n < 5:
            return None, None, None
        # State: x = [price, velocity]
        x = np.array([prices[0], 0.0])
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        Q = np.array([[0.001, 0.0], [0.0, 0.0001]])  # process noise
        R = np.array([[0.01]])  # measurement noise
        H = np.array([[1.0, 0.0]])  # observe price
        F = np.array([[1.0, 1.0], [0.0, 1.0]])  # constant velocity model

        velocities = np.full(n, np.nan)
        innovations = np.full(n, np.nan)

        for i in range(1, n):
            if np.isnan(prices[i]):
                continue
            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            # Update
            z = np.array([prices[i]])
            y = z - H @ x_pred  # innovation
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ y
            P = (np.eye(2) - K @ H) @ P_pred
            velocities[i] = x[1]
            innovations[i] = y[0]

        return velocities, innovations, x[1]  # final velocity

    def _signal_kalman(self, close):
        """Kalman velocity signal."""
        velocities, innovations, vel_final = self._kalman_filter(close)
        if velocities is None:
            return 0, 0.0
        # Check velocity crossover in last 3 bars
        if np.isnan(velocities[-1]) or np.isnan(velocities[-2]):
            return 0, 0.0
        # Velocity positive and accelerating
        if velocities[-1] > 0 and velocities[-2] <= 0:
            return 1, velocities[-1]  # bullish crossover
        elif velocities[-1] < 0 and velocities[-2] >= 0:
            return -1, velocities[-1]  # bearish crossover
        elif velocities[-1] > 0:
            return 1, velocities[-1]
        elif velocities[-1] < 0:
            return -1, velocities[-1]
        return 0, 0.0

    def _signal_vdp(self, data):
        """VDP EMA delta signal."""
        vol_col = None
        for vc in ['vol', 'volume', 'Volume']:
            if vc in data.columns:
                vol_col = vc
                break
        if vol_col is None:
            return 0
        n = len(data)
        if n < self.vdp_ema_period + 5:
            return 0
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        vol = data[vol_col].values.astype(float)
        vol = np.nan_to_num(vol, nan=0.0)
        deltas = np.zeros(n)
        for i in range(n):
            hl = high[i] - low[i]
            if hl > 1e-10:
                deltas[i] = vol[i] * (2 * close[i] - high[i] - low[i]) / hl
        k = 2.0 / (self.vdp_ema_period + 1)
        ema = np.zeros(n)
        ema[0] = deltas[0]
        for i in range(1, n):
            ema[i] = deltas[i] * k + ema[i-1] * (1 - k)
        if n < 5:
            return 0
        slope = ema[-1] - ema[-5]
        if slope > 0 and ema[-1] > 0:
            return 1
        elif slope < 0 and ema[-1] < 0:
            return -1
        return 0

    def _calc_ker(self, close):
        period = self.ker_period
        if len(close) < period + 1:
            return 0.0
        recent = close[-(period + 1):]
        net = abs(recent[-1] - recent[0])
        total = np.sum(np.abs(np.diff(recent)))
        if total < 1e-12:
            return 0.0
        return net / total

    def _calc_entropy(self, close):
        if len(close) < self.entropy_window + 1:
            return 1.0
        recent = close[-(self.entropy_window + 1):]
        returns = np.diff(np.log(recent))
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            return 1.0
        counts, _ = np.histogram(returns, bins=self.entropy_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = -np.sum(probs * np.log2(probs))
        h_max = np.log2(self.entropy_bins)
        return h / h_max

    def _evaluate(self, symbol, data):
        min_len = max(self.entropy_window + 10, 60)
        if len(data) < min_len:
            return None
        close = data['close'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None
        # Entropy gate
        entropy = self._calc_entropy(close)
        if entropy > 0.80:
            return None
        # KER regime
        ker = self._calc_ker(close)
        # Kalman signal
        k_signal, velocity = self._signal_kalman(close)
        # VDP confirmation
        vdp_signal = self._signal_vdp(data)

        score = 0
        # Kalman + VDP agreement
        if k_signal != 0 and k_signal == vdp_signal:
            score = 3
        elif k_signal != 0 and vdp_signal == 0:
            score = 1
        else:
            return None  # disagreement or no signal

        # KER regime boost
        if ker > 0.25:
            score += 1

        direction = 1 if k_signal > 0 else -1
        return score, direction, atr

    def generate_signals(self) -> List[Dict]:
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
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    result = self._evaluate(sym, hist)
                    if result is None:
                        continue
                    score, direction, _ = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= 2:
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
                    hist = data[(data['symbol'] == current_holding) & (data.index <= current_time)]
                    atr_val = self._calc_atr(hist)
                    should_exit = False
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True
                    if days_held >= self.max_hold:
                        should_exit = True
                    # Kalman velocity reversal exit
                    if not should_exit and days_held >= self.hold_min:
                        close = hist['close'].values.astype(float)
                        _, _, vel = self._kalman_filter(close)
                        if vel is not None:
                            if position_dir == 1 and vel < 0:
                                should_exit = True
                            elif position_dir == -1 and vel > 0:
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

        print(f"KalmanVDPFusion: generated {len(self.signals)} signals")
        return self.signals

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        n = len(high)
        tr_list = []
        for i in range(max(1, n - self.atr_period), n):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            tr_list.append(tr)
        return float(np.mean(tr_list)) if tr_list else 0.0

    def screen(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'
        sym = data['symbol'].iloc[0]
        if len(data) < 60:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}
        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])
        if result is None:
            return {'action': 'hold', 'reason': 'no signal', 'price': price}
        score, direction, _ = result
        if abs(score) >= 2:
            return {'action': 'buy' if direction == 1 else 'sell',
                    'reason': f'kalman+vdp score={score}', 'price': price}
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
