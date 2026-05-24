"""
Wasserstein Regime Switch Strategy (最优传输体制切换策略)
=========================================================
Uses Wasserstein-1 distance to detect regime changes in return distributions.
Large W1 = distribution changed = regime switch.

Math: W1(P,Q) = E[|X-Y|] via sorted samples
Philosophy: 变化之变 — the only constant is change
Category: regime
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class WassersteinRegimeSwitchStrategy(BaseStrategy):
    strategy_description = "Wasserstein: W1 distribution distance + VDP + tension for regime switches"
    strategy_category = "regime"
    strategy_params_schema = {
        "w1_window": {"type": "int", "default": 20, "label": "W1 return window"},
        "w1_percentile": {"type": "float", "default": 95, "label": "W1 spike percentile"},
        "w1_history": {"type": "int", "default": 250, "label": "W1 history for threshold"},
        "vdp_ema_period": {"type": "int", "default": 10, "label": "VDP EMA period"},
        "tension_window": {"type": "int", "default": 20, "label": "Tension window"},
        "ker_period": {"type": "int", "default": 20, "label": "KER period"},
        "entropy_window": {"type": "int", "default": 50, "label": "Entropy window"},
        "entropy_bins": {"type": "int", "default": 10, "label": "Entropy bins"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min hold days"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "Trail stop ATR mult"},
        "max_hold": {"type": "int", "default": 30, "label": "Max hold days"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.w1_window = p.get('w1_window', 20)
        self.w1_percentile = p.get('w1_percentile', 95)
        self.w1_history = p.get('w1_history', 250)
        self.vdp_ema_period = p.get('vdp_ema_period', 10)
        self.tension_window = p.get('tension_window', 20)
        self.ker_period = p.get('ker_period', 20)
        self.entropy_window = p.get('entropy_window', 50)
        self.entropy_bins = p.get('entropy_bins', 10)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 30)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'w1_window': 20, 'w1_percentile': 95, 'w1_history': 250,
            'vdp_ema_period': 10, 'tension_window': 20, 'ker_period': 20,
            'entropy_window': 50, 'entropy_bins': 10, 'atr_period': 14,
            'hold_min': 3, 'trail_atr_mult': 2.5, 'max_hold': 30,
        }

    def _wasserstein1(self, dist_a, dist_b):
        """W1 = mean(|sort(A) - sort(B)|)."""
        a = np.sort(dist_a)
        b = np.sort(dist_b)
        if len(a) == 0 or len(b) == 0:
            return 0.0
        # Match lengths by interpolation
        min_len = min(len(a), len(b))
        if len(a) > min_len:
            idx = np.linspace(0, len(a)-1, min_len).astype(int)
            a = a[idx]
        elif len(b) > min_len:
            idx = np.linspace(0, len(b)-1, min_len).astype(int)
            b = b[idx]
        return float(np.mean(np.abs(a - b)))

    def _evaluate(self, symbol, data):
        min_len = max(self.w1_window * 2 + 20, 100)
        if len(data) < min_len:
            return None
        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Compute returns
        returns = np.diff(close) / close[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < self.w1_window * 2:
            return None

        # Compute rolling W1
        n_ret = len(returns)
        w1_values = []
        for i in range(self.w1_window, n_ret - self.w1_window):
            dist_curr = returns[i:i+self.w1_window]
            dist_prev = returns[i-self.w1_window:i]
            if len(dist_curr) < 10 or len(dist_prev) < 10:
                w1_values.append(0.0)
                continue
            w1_values.append(self._wasserstein1(dist_curr, dist_prev))

        if len(w1_values) < 20:
            return None

        # Current W1
        current_w1 = w1_values[-1]
        # W1 history for threshold
        hist_w1 = w1_values[-min(self.w1_history, len(w1_values)):]
        threshold = np.percentile(hist_w1, self.w1_percentile) if len(hist_w1) > 20 else 0

        # Entropy gate
        entropy = self._calc_entropy(close)
        if entropy > 0.80:
            return None

        # VDP direction
        vdp_signal = self._calc_vdp_signal(data)

        # Tension
        tension = self._calc_tension(close, high, low)

        score = 0
        direction = 0

        if current_w1 > threshold and threshold > 0:
            # REGIME CHANGE detected — determine direction
            curr_returns = returns[-self.w1_window:]
            prev_returns = returns[-2*self.w1_window:-self.w1_window]
            if len(curr_returns) > 0 and len(prev_returns) > 0:
                curr_mean = np.mean(curr_returns)
                prev_mean = np.mean(prev_returns)
                if curr_mean > prev_mean and vdp_signal >= 0:
                    direction = 1  # bullish regime change
                    score = 4
                elif curr_mean < prev_mean and vdp_signal <= 0:
                    direction = -1  # bearish regime change
                    score = 4
                elif curr_mean > prev_mean:
                    direction = 1
                    score = 2
                else:
                    direction = -1
                    score = 2
        else:
            # Stable regime — momentum continuation
            ker = self._calc_ker(close)
            if ker > 0.2:
                recent_ret = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
                if recent_ret > 0.02 and vdp_signal >= 0:
                    direction = 1
                    score = 2
                elif recent_ret < -0.02 and vdp_signal <= 0:
                    direction = -1
                    score = 2

        # Tension confirmation
        if score > 0 and abs(tension) > 0.2:
            if (direction == 1 and tension > 0) or (direction == -1 and tension < 0):
                score += 1

        if score < 2 or direction == 0:
            return None
        return score, direction, atr

    def _calc_vdp_signal(self, data):
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
                deltas[i] = vol[i] * (2*close[i] - high[i] - low[i]) / hl
        k = 2.0 / (self.vdp_ema_period + 1)
        ema = np.zeros(n)
        ema[0] = deltas[0]
        for i in range(1, n):
            ema[i] = deltas[i] * k + ema[i-1] * (1 - k)
        slope = ema[-1] - ema[-5] if n >= 5 else 0
        if slope > 0 and ema[-1] > 0:
            return 1
        elif slope < 0 and ema[-1] < 0:
            return -1
        return 0

    def _calc_tension(self, close, high, low):
        n = len(close)
        if n < self.tension_window:
            return 0.0
        c_win = close[-self.tension_window:]
        h_win = high[-self.tension_window:]
        l_win = low[-self.tension_window:]
        hh = np.nanmax(h_win)
        ll = np.nanmin(l_win)
        mid = (hh + ll) / 2.0
        rng = hh - ll
        if rng <= 0:
            return 0.0
        return float(((close[-1] - hh) + (close[-1] - ll) + (close[-1] - mid)) / (3 * rng))

    def _calc_ker(self, close):
        if len(close) < self.ker_period + 1:
            return 0.0
        recent = close[-(self.ker_period + 1):]
        net = abs(recent[-1] - recent[0])
        total = np.sum(np.abs(np.diff(recent)))
        return net / total if total > 1e-12 else 0.0

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
        return h / np.log2(self.entropy_bins)

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        n = len(high)
        tr_list = []
        for i in range(max(1, n - self.atr_period), n):
            tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            tr_list.append(tr)
        return float(np.mean(tr_list)) if tr_list else 0.0

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
                best_score, best_sym, best_dir = 0, None, 0
                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    result = self._evaluate(sym, hist)
                    if result is None:
                        continue
                    score, direction, _ = result
                    if abs(score) > abs(best_score):
                        best_score, best_sym, best_dir = score, sym, direction
                if best_sym and abs(best_score) >= 2:
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water, low_water = 0.0, float('inf')
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
                    if should_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding)
                        else:
                            self._record_signal(current_time, 'buy', current_holding)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water, low_water = 0.0, float('inf')

        print(f"WassersteinRegimeSwitch: generated {len(self.signals)} signals")
        return self.signals

    def screen(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'
        sym = data['symbol'].iloc[0]
        if len(data) < 100:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}
        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])
        if result is None:
            return {'action': 'hold', 'reason': 'no signal', 'price': price}
        score, direction, _ = result
        if abs(score) >= 2:
            return {'action': 'buy' if direction == 1 else 'sell',
                    'reason': f'W1 regime switch score={score}', 'price': price}
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
