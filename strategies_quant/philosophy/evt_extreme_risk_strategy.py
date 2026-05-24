"""
EVT Extreme Risk Strategy (极值理论极端风险策略)
=================================================
Uses Generalized Pareto Distribution to model tail risk.
A-share returns have heavy tails (ξ≈0.35), making normal-based risk estimates dangerous.

Math: GPD: P(X-x|X>u) = 1-(1+ξ(x-u)/σ)^(-1/ξ), Peaks Over Threshold
Philosophy: 无常 — extreme events are more common than we think
Category: risk
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class EVTExtremeRiskStrategy(BaseStrategy):
    strategy_description = "EVT: GPD tail fitting for extreme risk detection + momentum + VDP"
    strategy_category = "risk"
    strategy_params_schema = {
        "gpd_window": {"type": "int", "default": 60, "label": "GPD estimation window"},
        "var_threshold": {"type": "float", "default": 0.05, "label": "POT threshold quantile"},
        "var_confidence": {"type": "float", "default": 0.95, "label": "VaR confidence level"},
        "tail_scale": {"type": "float", "default": 1.5, "label": "Tail stop scale factor"},
        "vdp_ema_period": {"type": "int", "default": 10, "label": "VDP EMA period"},
        "ker_period": {"type": "int", "default": 20, "label": "KER period"},
        "entropy_window": {"type": "int", "default": 50, "label": "Entropy window"},
        "entropy_bins": {"type": "int", "default": 10, "label": "Entropy bins"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min hold days"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "Trail stop ATR mult"},
        "max_hold": {"type": "int", "default": 30, "label": "Max hold days"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.gpd_window = p.get('gpd_window', 60)
        self.var_threshold = p.get('var_threshold', 0.05)
        self.var_confidence = p.get('var_confidence', 0.95)
        self.tail_scale = p.get('tail_scale', 1.5)
        self.vdp_ema_period = p.get('vdp_ema_period', 10)
        self.ker_period = p.get('ker_period', 20)
        self.entropy_window = p.get('entropy_window', 50)
        self.entropy_bins = p.get('entropy_bins', 10)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.0)
        self.max_hold = p.get('max_hold', 30)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'gpd_window': 60, 'var_threshold': 0.05, 'var_confidence': 0.95,
            'tail_scale': 1.5, 'vdp_ema_period': 10, 'ker_period': 20,
            'entropy_window': 50, 'entropy_bins': 10, 'atr_period': 14,
            'hold_min': 3, 'trail_atr_mult': 2.0, 'max_hold': 30,
        }

    def _fit_gpd(self, losses):
        """Fit GPD to losses exceeding threshold. Returns (xi, sigma)."""
        if len(losses) < 10:
            return 0.0, 0.0
        threshold = np.percentile(losses, (1 - self.var_threshold) * 100)
        exceedances = losses[losses > threshold] - threshold
        if len(exceedances) < 5:
            return 0.0, 0.0
        # Method of Moments estimator for GPD
        xi = 0.5 * (1 - (np.mean(exceedances)**2 / np.var(exceedances)))
        sigma = np.mean(exceedances) * (1 - xi)
        if sigma <= 0:
            sigma = np.mean(exceedances)
        return float(xi), float(sigma)

    def _compute_var(self, losses, xi, sigma):
        """Compute VaR using GPD parameters."""
        if len(losses) < 10 or sigma <= 0:
            return 0.0
        threshold = np.percentile(losses, (1 - self.var_threshold) * 100)
        n = len(losses)
        n_exceed = np.sum(losses > threshold)
        if n_exceed < 3:
            return np.percentile(losses, self.var_confidence * 100)
        alpha = 1 - self.var_confidence
        var = threshold + (sigma / max(xi, 0.01)) * ((n / n_exceed * alpha)**(-xi) - 1)
        return float(var)

    def _evaluate(self, symbol, data):
        min_len = max(self.gpd_window + 20, 80)
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

        # Compute returns and losses (negative returns)
        returns = np.diff(close) / close[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < self.gpd_window:
            return None

        recent_returns = returns[-self.gpd_window:]
        losses = -recent_returns  # losses = negative returns

        # Fit GPD
        xi, sigma = self._fit_gpd(losses)
        gpd_var = self._compute_var(losses, xi, sigma)
        normal_var = -np.percentile(recent_returns, (1 - self.var_confidence) * 100)

        # Tail risk ratio: GPD VaR / Normal VaR
        tail_ratio = gpd_var / max(normal_var, 1e-6)
        # xi > 0 = heavy tail, higher = more dangerous

        # VDP direction
        vdp_signal = self._calc_vdp_signal(data)
        # KER
        ker = self._calc_ker(close)

        score = 0
        direction = 0

        # HIGH TAIL RISK: xi > 0.3 or tail_ratio > 1.5 → be cautious
        if xi > 0.4 or tail_ratio > 2.0:
            # Extreme tail risk — only trade with strong signals
            if vdp_signal < 0 and ker < 0.15:
                direction = -1
                score = 3  # Bearish in extreme risk
            else:
                return None  # Too risky
        elif xi > 0.2 or tail_ratio > 1.3:
            # Moderate tail risk
            if vdp_signal >= 0 and ker > 0.2:
                direction = 1
                score = 2
            elif vdp_signal <= 0:
                direction = -1
                score = 2
        else:
            # Low tail risk — normal trading
            recent_ret = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
            if recent_ret > 0.02 and vdp_signal >= 0:
                direction = 1
                score = 2
            elif recent_ret < -0.02 and vdp_signal <= 0:
                direction = -1
                score = 2

        if score < 2 or direction == 0:
            return None

        # Adjust trail stop by tail risk
        adjusted_trail = self.trail_atr_mult * (1 + xi * self.tail_scale)
        return score, direction, atr, adjusted_trail

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
        return float(-np.sum(probs * np.log2(probs)) / np.log2(self.entropy_bins))

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
        current_trail = self.trail_atr_mult

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
                    score, direction, atr_val = result[0], result[1], result[2]
                    if len(result) > 3:
                        current_trail = result[3]
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
                            if current_price < high_water - current_trail * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + current_trail * atr_val:
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

        print(f"EVTExtremeRisk: generated {len(self.signals)} signals")
        return self.signals

    def screen(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'
        sym = data['symbol'].iloc[0]
        if len(data) < 80:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}
        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])
        if result is None:
            return {'action': 'hold', 'reason': 'no signal', 'price': price}
        score, direction = result[0], result[1]
        if abs(score) >= 2:
            return {'action': 'buy' if direction == 1 else 'sell',
                    'reason': f'EVT score={score}', 'price': price}
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
