"""
Independent Signal Ensemble Strategy (独立信号集成策略)
======================================================
Key insight from v109: I(ST,VDP)/H(ST)≈0, so combining independent signals
gives exponential performance boost. This strategy explicitly maximizes
signal independence.

Core Logic:
  1. Compute 5 INDEPENDENT signals (low mutual information):
     - Momentum rank (price-based)
     - VDP delta (volume-based)
     - Structural tension (geometry-based)
     - KER regime (efficiency-based)
     - Entropy gate (information-based)
  2. Each signal is orthogonal to others (different data source)
  3. Score = count of agreeing signals (consensus scoring)
  4. Require ≥3/5 agreement for entry
  5. ATR trailing stop for exits

Mathematical Foundation:
  Mutual Information: I(X;Y) = H(X) - H(X|Y)
  When I(X;Y) ≈ 0: P(X∩Y) ≈ P(X)·P(Y) → combined signal is exponentially stronger
  Consensus score: S = Σ s_i where s_i ∈ {-1, 0, +1}
  Entry threshold: |S| ≥ 3 (at least 3 independent signals agree)

Philosophy: 和而不同 — harmony through diversity, not uniformity.
Each signal sees the market differently; when they agree, it's significant.

Technical Indicators: Rank Momentum, VDP, Tension, KER, Entropy, ATR
Category: ensemble
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from core.base_strategy import BaseStrategy


class IndependentSignalEnsembleStrategy(BaseStrategy):
    """Independent signal ensemble — maximize signal independence"""

    strategy_description = (
        "IndependentEnsemble: 5 independent signals (momentum, VDP, tension, "
        "KER, entropy) with consensus scoring + ATR trailing stop"
    )
    strategy_category = "ensemble"
    strategy_params_schema = {
        "mom_period": {"type": "int", "default": 10, "label": "Momentum period"},
        "vdp_ema_period": {"type": "int", "default": 10, "label": "VDP EMA period"},
        "ker_period": {"type": "int", "default": 20, "label": "KER period"},
        "entropy_bins": {"type": "int", "default": 10, "label": "Entropy bins"},
        "entropy_window": {"type": "int", "default": 50, "label": "Entropy window"},
        "pivot_len": {"type": "int", "default": 5, "label": "Pivot lookback"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "Trail stop ATR mult"},
        "max_hold": {"type": "int", "default": 60, "label": "Max holding days"},
        "consensus_threshold": {"type": "int", "default": 3, "label": "Min consensus count"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.mom_period = p.get('mom_period', 10)
        self.vdp_ema_period = p.get('vdp_ema_period', 10)
        self.ker_period = p.get('ker_period', 20)
        self.entropy_bins = p.get('entropy_bins', 10)
        self.entropy_window = p.get('entropy_window', 50)
        self.pivot_len = p.get('pivot_len', 5)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 60)
        self.consensus_threshold = p.get('consensus_threshold', 3)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'mom_period': 10, 'vdp_ema_period': 10, 'ker_period': 20,
            'entropy_bins': 10, 'entropy_window': 50, 'pivot_len': 5,
            'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
            'max_hold': 60, 'consensus_threshold': 3,
        }

    # ================================================================
    # The 5 Independent Signals
    # ================================================================

    def _signal_momentum(self, close: np.ndarray) -> int:
        """Signal 1: Momentum (price-based). Returns -1, 0, +1."""
        if len(close) < self.mom_period + 1:
            return 0
        if close[-self.mom_period - 1] <= 0:
            return 0
        mom = (close[-1] - close[-self.mom_period - 1]) / close[-self.mom_period - 1]
        if mom > 0.02:
            return 1
        elif mom < -0.02:
            return -1
        return 0

    def _signal_vdp(self, data: pd.DataFrame) -> int:
        """Signal 2: VDP delta (volume-based). Returns -1, 0, +1."""
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
        # Check if EMA is trending up or down over last 5 bars
        if n < 5:
            return 0
        ema_recent = ema[-5:]
        if np.std(ema_recent) < 1e-10:
            return 0
        slope = (ema_recent[-1] - ema_recent[0]) / 5
        if slope > 0 and ema[-1] > 0:
            return 1
        elif slope < 0 and ema[-1] < 0:
            return -1
        return 0

    def _signal_tension(self, close: np.ndarray, high: np.ndarray,
                        low: np.ndarray, atr: float) -> int:
        """Signal 3: Structural tension (geometry-based). Returns -1, 0, +1."""
        n = len(close)
        window = 20
        if n < window or atr <= 0:
            return 0
        c_win = close[-window:]
        h_win = high[-window:]
        l_win = low[-window:]
        cv = c_win[~np.isnan(c_win)]
        hv = h_win[~np.isnan(h_win)]
        lv = l_win[~np.isnan(l_win)]
        if len(cv) < 10:
            return 0
        hh = np.max(hv) if len(hv) > 0 else np.max(cv)
        ll = np.min(lv) if len(lv) > 0 else np.min(cv)
        mid = (hh + ll) / 2.0
        rng = hh - ll
        if rng <= 0:
            return 0
        tension = ((close[-1] - hh) + (close[-1] - ll) + (close[-1] - mid)) / (3 * rng)
        if tension > 0.2:
            return 1
        elif tension < -0.2:
            return -1
        return 0

    def _signal_ker(self, close: np.ndarray) -> Tuple[int, float]:
        """Signal 4: KER regime (efficiency-based). Returns (signal, ker_value)."""
        period = self.ker_period
        if len(close) < period + 1:
            return 0, 0.0
        recent = close[-(period + 1):]
        net = abs(recent[-1] - recent[0])
        total = np.sum(np.abs(np.diff(recent)))
        if total < 1e-12:
            return 0, 0.0
        ker = net / total
        if ker < 0.15:
            return 0, ker  # Ranging → no signal
        # Direction from net change
        direction = recent[-1] - recent[0]
        if direction > 0 and ker > 0.2:
            return 1, ker
        elif direction < 0 and ker > 0.2:
            return -1, ker
        return 0, ker

    def _signal_entropy(self, close: np.ndarray) -> Tuple[int, float]:
        """Signal 5: Entropy gate (information-based). Returns (signal, entropy)."""
        if len(close) < self.entropy_window + 1:
            return 0, 0.0
        recent = close[-(self.entropy_window + 1):]
        returns = np.diff(np.log(recent))
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            return 0, 0.0
        counts, _ = np.histogram(returns, bins=self.entropy_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = float(-np.sum(probs * np.log2(probs)))
        h_max = np.log2(self.entropy_bins)
        h_norm = h / h_max  # normalized [0,1]

        # Low entropy = ordered = trade with trend direction
        if h_norm < 0.7:
            # Use recent return direction
            ret = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
            if ret > 0.01:
                return 1, h_norm
            elif ret < -0.01:
                return -1, h_norm
            return 0, h_norm
        # High entropy = chaotic = no trade
        return 0, h_norm

    # ================================================================
    # Consensus Scoring
    # ================================================================

    def _evaluate(self, symbol: str, data: pd.DataFrame):
        """Evaluate all 5 independent signals and compute consensus."""
        min_len = max(self.entropy_window + 10, self.mom_period + 20, 60)
        if len(data) < min_len:
            return None

        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Collect all 5 signals
        s1 = self._signal_momentum(close)
        s2 = self._signal_vdp(data)
        s3 = self._signal_tension(close, high, low, atr)
        s4, ker = self._signal_ker(close)
        s5, entropy = self._signal_entropy(close)

        signals = [s1, s2, s3, s4, s5]
        consensus = sum(signals)

        # Require minimum consensus
        agreeing = sum(1 for s in signals if s == np.sign(consensus) if s != 0)
        if agreeing < self.consensus_threshold:
            return None

        if abs(consensus) < self.consensus_threshold:
            return None

        # Score based on consensus strength
        score = abs(consensus)
        direction = 1 if consensus > 0 else -1

        return score, direction, atr

    # ================================================================
    # Signal Generation
    # ================================================================

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

                if best_sym and abs(best_score) >= self.consensus_threshold:
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

        print(f"IndependentEnsemble: generated {len(self.signals)} signals")
        return self.signals

    def _calc_atr(self, data):
        """ATR calculation."""
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
        """Quick screen based on latest data."""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'
        sym = data['symbol'].iloc[0]
        min_len = max(self.entropy_window + 10, self.mom_period + 20, 60)
        if len(data) < min_len:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}
        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])
        if result is None:
            return {'action': 'hold', 'reason': 'no consensus', 'price': price}
        score, direction, _ = result
        if abs(score) >= self.consensus_threshold:
            return {'action': 'buy' if direction == 1 else 'sell',
                    'reason': f'consensus={score}/5', 'price': price}
        return {'action': 'hold', 'reason': f'consensus={score}/5', 'price': price}
