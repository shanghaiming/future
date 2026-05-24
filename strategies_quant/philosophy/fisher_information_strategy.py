"""
Fisher Information Strategy (Fisher信息策略)
=============================================
Uses Fisher information to detect regime stability and regime shifts
before they become obvious in price action.

Mathematical Foundation:
  Fisher Information: I(θ) = E[(∂log f(X;θ)/∂θ)²]
  For Gaussian returns N(μ, σ²) with parameter θ=μ: I(μ) = n/σ²

  High Fisher info → σ² is low → regime is well-defined → trade with confidence
  Fisher info derivative → regime shift detection → reversal signal
  Low Fisher info → regime uncertain → no trade

Trading Logic:
  1. HIGH + INCREASING Fisher info: stable regime → trend-following entry
  2. SHARP DROP in Fisher info: regime change → reversal signal
  3. LOW Fisher info: uncertain regime → no trade (entropy gate)
  4. Confirmed by VDP delta and KER regime

Philosophy: 见微知著 — See the subtle, know the manifest.
Fisher information detects tiny distributional shifts that precede
major market moves. By measuring how much information the price series
carries about the regime parameter, we can identify when regimes are
stable (high information, tradeable) vs shifting (information dropping,
reversal opportunity) vs uncertain (low information, avoid).

Technical Indicators: Fisher Information, VDP, KER, Entropy, ATR
Category: information
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from core.base_strategy import BaseStrategy


class FisherInformationStrategy(BaseStrategy):
    """Fisher Information regime detection — trade regime stability, detect regime shifts"""

    strategy_description = (
        "FisherInfo: detects regime stability via Fisher information I(μ)=n/σ². "
        "High+rising Fisher → trend with confidence. Sharp drop → regime shift. "
        "Low Fisher → no trade. Confirmed by VDP+KER."
    )
    strategy_category = "information"
    strategy_params_schema = {
        "fisher_window": {"type": "int", "default": 20, "label": "Fisher estimation window"},
        "fisher_drop_threshold": {"type": "float", "default": -0.3,
                                  "label": "Fisher drop threshold for regime shift"},
        "vdp_ema": {"type": "int", "default": 10, "label": "VDP EMA period"},
        "ker_period": {"type": "int", "default": 20, "label": "KER period"},
        "entropy_window": {"type": "int", "default": 50, "label": "Entropy window"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "max_hold": {"type": "int", "default": 25, "label": "Max holding days"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "Trail stop ATR mult"},
        "score_threshold": {"type": "int", "default": 3, "label": "Min signal score to enter"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.fisher_window = p.get('fisher_window', 20)
        self.fisher_drop_threshold = p.get('fisher_drop_threshold', -0.3)
        self.vdp_ema = p.get('vdp_ema', 10)
        self.ker_period = p.get('ker_period', 20)
        self.entropy_window = p.get('entropy_window', 50)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.max_hold = p.get('max_hold', 25)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.0)
        self.score_threshold = p.get('score_threshold', 3)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fisher_window': 20,
            'fisher_drop_threshold': -0.3,
            'vdp_ema': 10,
            'ker_period': 20,
            'entropy_window': 50,
            'atr_period': 14,
            'hold_min': 3,
            'max_hold': 25,
            'trail_atr_mult': 2.0,
            'score_threshold': 3,
        }

    # ================================================================
    # Core Indicator: Fisher Information
    # ================================================================

    def _compute_fisher_information(self, close: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute rolling Fisher information about the mean parameter.

        For Gaussian returns ~ N(μ, σ²):
            I(μ) = n / σ²

        Returns:
            (fisher_info, fisher_delta, fisher_norm_level)
            - fisher_info: current Fisher information value
            - fisher_delta: rate of change of Fisher info (normalized)
            - fisher_norm_level: normalized level (0-1) vs historical
        """
        n = len(close)
        w = self.fisher_window

        # Need enough data for at least 2 windows to compute delta
        if n < w + 2:
            return 0.0, 0.0, 0.0

        # Compute rolling Fisher information series
        fisher_series = []
        for start in range(max(0, n - w * 3), n - w + 1):
            window_close = close[start:start + w + 1]
            returns = np.diff(window_close) / window_close[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) < 5:
                fisher_series.append(0.0)
                continue
            var = np.var(returns, ddof=1)
            if var < 1e-15:
                # Extremely low variance — very high Fisher info, but cap it
                fisher_series.append(float(len(returns)) / 1e-15)
                continue
            fisher_series.append(float(len(returns)) / var)

        if len(fisher_series) < 3:
            return 0.0, 0.0, 0.0

        current_fi = fisher_series[-1]
        prev_fi = fisher_series[-2]

        # Delta: normalized rate of change
        if abs(prev_fi) > 1e-15:
            delta = (current_fi - prev_fi) / abs(prev_fi)
        else:
            delta = 0.0

        # Normalized level: where current Fisher info sits vs recent range
        fi_min = np.min(fisher_series)
        fi_max = np.max(fisher_series)
        fi_range = fi_max - fi_min
        if fi_range > 1e-15:
            norm_level = (current_fi - fi_min) / fi_range
        else:
            norm_level = 0.5

        return current_fi, delta, norm_level

    # ================================================================
    # Confirmation Indicators
    # ================================================================

    def _calc_vdp_delta(self, data: pd.DataFrame) -> int:
        """
        VDP (Volume Delta Position) EMA slope signal.
        Returns -1, 0, +1.
        """
        vol_col = None
        for vc in ['vol', 'volume', 'Volume']:
            if vc in data.columns:
                vol_col = vc
                break
        if vol_col is None:
            return 0

        n = len(data)
        if n < self.vdp_ema + 5:
            return 0

        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        vol = data[vol_col].values.astype(float)
        vol = np.nan_to_num(vol, nan=0.0)

        # Compute VDP deltas
        deltas = np.zeros(n)
        for i in range(n):
            hl = high[i] - low[i]
            if hl > 1e-10:
                deltas[i] = vol[i] * (2 * close[i] - high[i] - low[i]) / hl

        # EMA smoothing
        k = 2.0 / (self.vdp_ema + 1)
        ema = np.zeros(n)
        ema[0] = deltas[0]
        for i in range(1, n):
            ema[i] = deltas[i] * k + ema[i - 1] * (1 - k)

        # Slope over last 5 bars
        recent = ema[-5:]
        if np.std(recent) < 1e-10:
            return 0
        slope = (recent[-1] - recent[0]) / 5
        if slope > 0 and ema[-1] > 0:
            return 1
        elif slope < 0 and ema[-1] < 0:
            return -1
        return 0

    def _calc_ker(self, close: np.ndarray) -> Tuple[int, float]:
        """
        Kaufman Efficiency Ratio (KER).
        Returns (direction_signal, ker_value).
        """
        period = self.ker_period
        if len(close) < period + 1:
            return 0, 0.0

        recent = close[-(period + 1):]
        net = abs(recent[-1] - recent[0])
        total = np.sum(np.abs(np.diff(recent)))
        if total < 1e-12:
            return 0, 0.0

        ker = net / total

        # Low efficiency = ranging, no signal
        if ker < 0.15:
            return 0, ker

        direction = recent[-1] - recent[0]
        if direction > 0 and ker > 0.2:
            return 1, ker
        elif direction < 0 and ker > 0.2:
            return -1, ker
        return 0, ker

    def _calc_entropy(self, close: np.ndarray) -> Tuple[float, float]:
        """
        Shannon entropy of log returns.
        Returns (normalized_entropy, entropy_level).
        normalized_entropy: 0 = perfectly ordered, 1 = maximum chaos
        """
        n_bins = 10
        if len(close) < self.entropy_window + 1:
            return 1.0, 1.0

        recent = close[-(self.entropy_window + 1):]
        returns = np.diff(np.log(recent))
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            return 1.0, 1.0

        counts, _ = np.histogram(returns, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = float(-np.sum(probs * np.log2(probs)))
        h_max = np.log2(n_bins)
        h_norm = h / h_max

        return h_norm, h_norm

    def _calc_atr(self, data) -> float:
        """ATR calculation."""
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        n = len(high)
        tr_list = []
        for i in range(max(1, n - self.atr_period), n):
            tr = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
            tr_list.append(tr)
        return float(np.mean(tr_list)) if tr_list else 0.0

    # ================================================================
    # Fisher-Based Signal Scoring
    # ================================================================

    def _score_fisher_regime(self, close: np.ndarray) -> Tuple[int, int]:
        """
        Core Fisher information regime analysis.
        Returns (score_contribution, direction).
        score_contribution: 0, 1, or 2
        direction: -1, 0, +1
        """
        fi, delta, norm_level = self._compute_fisher_information(close)

        if fi <= 0:
            return 0, 0

        # Determine recent price direction for momentum
        mom_period = min(10, len(close) - 1)
        if mom_period < 3:
            return 0, 0
        recent_ret = (close[-1] - close[-mom_period]) / close[-mom_period]
        if abs(recent_ret) < 1e-10:
            direction = 0
        else:
            direction = 1 if recent_ret > 0 else -1

        # ---- Regime 1: HIGH Fisher info + INCREASING ----
        # Regime is stable and well-defined → strong trend signal
        if norm_level > 0.6 and delta > 0:
            # High information, increasing → high confidence trend
            return 2, direction

        # ---- Regime 2: HIGH Fisher info + STABLE ----
        # Still a decent trend environment
        if norm_level > 0.5 and abs(delta) < 0.1:
            return 1, direction

        # ---- Regime 3: SHARP DROP in Fisher info ----
        # Regime change in progress → potential reversal
        if delta < self.fisher_drop_threshold:
            if recent_ret > 0:
                # Trend was up, Fisher dropping → trend ending → bearish reversal
                return 2, -1
            else:
                # Trend was down, Fisher dropping → bottom forming → bullish reversal
                return 2, 1

        # ---- Regime 4: LOW Fisher info ----
        # Regime uncertain → no trade signal
        if norm_level < 0.3:
            return 0, 0

        # Moderate Fisher info, moderate delta → weak signal
        if norm_level > 0.35 and abs(direction) > 0:
            return 1, direction

        return 0, 0

    def _evaluate(self, symbol: str, data: pd.DataFrame):
        """
        Evaluate all signals and compute entry score.
        Returns (score, direction, atr) or None.
        """
        min_len = max(self.fisher_window * 3 + 10,
                      self.entropy_window + 10,
                      self.ker_period + 20,
                      80)
        if len(data) < min_len:
            return None

        close = data['close'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        score = 0
        direction_votes = []

        # --- Signal 1: Fisher Information Regime (weight 2) ---
        fi_score, fi_dir = self._score_fisher_regime(close)
        score += fi_score
        if fi_dir != 0:
            direction_votes.extend([fi_dir] * fi_score)

        # --- Signal 2: VDP Delta confirmation ---
        vdp = self._calc_vdp_delta(data)
        if vdp != 0 and vdp == fi_dir:
            score += 1
            direction_votes.append(vdp)

        # --- Signal 3: KER regime confirmation ---
        ker_sig, ker_val = self._calc_ker(close)
        if ker_sig != 0 and ker_sig == fi_dir:
            score += 1
            direction_votes.append(ker_sig)

        # --- Signal 4: Entropy gate ---
        entropy_norm, _ = self._calc_entropy(close)
        if entropy_norm > 0.85:
            # Very high entropy = chaotic market → penalize
            score = max(0, score - 2)
        elif entropy_norm < 0.6:
            # Low entropy = ordered market → reward
            score += 1

        # Determine final direction
        if not direction_votes:
            return None
        direction = 1 if sum(direction_votes) > 0 else -1

        # Apply score threshold
        if score < self.score_threshold:
            return None

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
                    if score > best_score:
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and best_score >= self.score_threshold:
                    price = float(current_bars[
                        current_bars['symbol'] == best_sym
                    ].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym, price)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym, price)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0
                    low_water = float('inf')

            else:
                days_held = len([t for t in unique_times
                                 if buy_time < t <= current_time])
                bar_data = current_bars[
                    current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                if position_dir == 1:
                    high_water = max(high_water, current_price) \
                        if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) \
                        if low_water < float('inf') else current_price

                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding)
                                & (data.index <= current_time)]
                    atr_val = self._calc_atr(hist)
                    should_exit = False

                    # ATR trailing stop exit
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # Fisher reversal exit: check if Fisher info is signaling reversal
                    if not should_exit and days_held >= self.hold_min + 2:
                        close_arr = hist['close'].values.astype(float)
                        if len(close_arr) > self.fisher_window * 3:
                            _, delta, norm_level = self._compute_fisher_information(
                                close_arr)
                            # Exit if Fisher drops sharply against our position
                            if delta < self.fisher_drop_threshold:
                                should_exit = True

                    # Max hold exit
                    if days_held >= self.max_hold:
                        should_exit = True

                    if should_exit:
                        if position_dir == 1:
                            self._record_signal(
                                current_time, 'sell', current_holding, current_price)
                        else:
                            self._record_signal(
                                current_time, 'buy', current_holding, current_price)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0
                        low_water = float('inf')

        print(f"FisherInformation: generated {len(self.signals)} signals")
        return self.signals

    def screen(self) -> Dict:
        """Quick screen based on latest data."""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'
        sym = data['symbol'].iloc[0]
        min_len = max(self.fisher_window * 3 + 10,
                      self.entropy_window + 10,
                      80)
        if len(data) < min_len:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}
        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])
        if result is None:
            return {'action': 'hold', 'reason': 'no Fisher signal', 'price': price}
        score, direction, _ = result
        if score >= self.score_threshold:
            action = 'buy' if direction == 1 else 'sell'
            return {'action': action,
                    'reason': f'Fisher score={score}, dir={direction}',
                    'price': price}
        return {'action': 'hold', 'reason': f'Fisher score={score} < {self.score_threshold}',
                'price': price}
