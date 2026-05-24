"""
Copula Tail Risk Strategy (连接函数尾部风险策略)
===============================================
Uses empirical copula to detect tail dependency between stock and market.
High lower tail dependency = systematic crash risk.

Math: Sklar's theorem, empirical copula via rank transform
Philosophy: 关系 — relationships reveal hidden risks
Category: risk
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class CopulaTailRiskStrategy(BaseStrategy):
    strategy_description = "Copula: empirical tail dependency + VDP + tension for risk-aware trading"
    strategy_category = "risk"
    strategy_params_schema = {
        "window": {"type": "int", "default": 50, "label": "Copula estimation window"},
        "tail_alpha": {"type": "float", "default": 0.1, "label": "Tail threshold alpha"},
        "dep_threshold": {"type": "float", "default": 0.4, "label": "Dependency threshold"},
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
        self.window = p.get('window', 50)
        self.tail_alpha = p.get('tail_alpha', 0.1)
        self.dep_threshold = p.get('dep_threshold', 0.4)
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
            'window': 50, 'tail_alpha': 0.1, 'dep_threshold': 0.4,
            'vdp_ema_period': 10, 'ker_period': 20, 'entropy_window': 50,
            'entropy_bins': 10, 'atr_period': 14, 'hold_min': 3,
            'trail_atr_mult': 2.0, 'max_hold': 30,
        }

    def _empirical_tail_dep(self, x, y):
        """Compute lower tail dependency lambda_L."""
        n = len(x)
        if n < 20:
            return 0.0, 0.0
        # Rank transform to uniform [0,1]
        u = np.argsort(np.argsort(x)).astype(float) / max(n - 1, 1)
        v = np.argsort(np.argsort(y)).astype(float) / max(n - 1, 1)
        # Lower tail: P(V < alpha | U < alpha)
        threshold = self.tail_alpha
        u_below = u < threshold
        if np.sum(u_below) < 3:
            return 0.0, 0.0
        lambda_L = np.sum(v[u_below] < threshold) / np.sum(u_below)
        # Upper tail: P(V > 1-alpha | U > 1-alpha)
        u_above = u > (1 - threshold)
        if np.sum(u_above) < 3:
            return lambda_L, 0.0
        lambda_U = np.sum(v[u_above] > (1 - threshold)) / np.sum(u_above)
        return float(lambda_L), float(lambda_U)

    def _evaluate(self, symbol, data):
        min_len = max(self.window + 20, 80)
        if len(data) < min_len:
            return None
        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Entropy gate
        entropy = self._calc_entropy(close)
        if entropy > 0.80:
            return None

        # Stock returns
        stock_returns = np.diff(close) / close[:-1]
        stock_returns = stock_returns[np.isfinite(stock_returns)]
        if len(stock_returns) < self.window:
            return None

        # Use stock returns vs its own lagged returns as proxy for "market" dependency
        # (since we don't have market index data in single-stock context)
        x = stock_returns[-self.window:]
        y = stock_returns[-self.window-1:-1]  # lagged by 1

        lambda_L, lambda_U = self._empirical_tail_dep(x, y)

        # VDP direction
        vdp_signal = self._calc_vdp_signal(data)
        # KER regime
        ker = self._calc_ker(close)
        # Tension
        tension = self._calc_tension(close, high, low)

        score = 0
        direction = 0

        # High upper tail dep = stock tends to rally with market
        if lambda_U > self.dep_threshold:
            if ker > 0.2 and vdp_signal >= 0 and tension > 0:
                direction = 1
                score = 3
            elif ker > 0.15 and vdp_signal >= 0:
                direction = 1
                score = 2

        # High lower tail dep = crash-prone, avoid or short
        if lambda_L > self.dep_threshold:
            if vdp_signal <= 0 and tension < 0:
                direction = -1
                score = 3
            elif vdp_signal <= 0:
                direction = -1
                score = max(score, 2)

        # Low tail dependency = independent mover, use momentum
        if lambda_L < 0.2 and lambda_U < 0.2:
            recent_ret = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
            if recent_ret > 0.03 and vdp_signal >= 0:
                direction = 1
                score = 2
            elif recent_ret < -0.03 and vdp_signal <= 0:
                direction = -1
                score = 2

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
        w = min(20, len(close))
        if w < 10:
            return 0.0
        hh = np.nanmax(high[-w:])
        ll = np.nanmin(low[-w:])
        rng = hh - ll
        if rng <= 0:
            return 0.0
        mid = (hh + ll) / 2.0
        return float(((close[-1]-hh) + (close[-1]-ll) + (close[-1]-mid)) / (3*rng))

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

        print(f"CopulaTailRisk: generated {len(self.signals)} signals")
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
        score, direction, _ = result
        if abs(score) >= 2:
            return {'action': 'buy' if direction == 1 else 'sell',
                    'reason': f'copula tail score={score}', 'price': price}
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
