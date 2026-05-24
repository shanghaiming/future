"""
DMD Regime Routing Strategy (动态模态分解状态路由策略)
=======================================================
Uses Dynamic Mode Decomposition (DMD) to decompose price dynamics into
growing, oscillating, and decaying modes, then routes to regime-appropriate
sub-strategies.

Core Logic:
  1. Build Hankel matrix from recent close prices
  2. SVD-based DMD: approximate Koopman operator A s.t. x(t+1) ~ A * x(t)
  3. Eigendecompose A_tilde to classify dominant mode:
     - Growing real eigenvalue  -> TREND regime
     - Complex eigenvalue       -> CYCLE regime
     - Decaying eigenvalue      -> RANGE regime
  4. Route to regime-specific entry logic:
     TREND: momentum + VDP + wide ATR trail
     CYCLE: oscillator (phase-based) + tight ATR trail
     RANGE: Bollinger mean-reversion + VDP delta + medium ATR trail
  5. Exit on mode change, trailing stop, or max hold

Mathematical Foundation:
  Koopman operator: x(t+1) = A * x(t)
  DMD via SVD:  X1 = U S V^T,  A_tilde = U^T X2 V S^{-1}
  Eigendecompose A_tilde = W Lambda W^{-1}
  Eigenvalue classification:
    |lambda| > 1, Im(lambda) ~ 0  -> growing trend
    |lambda| ~ 1, Im(lambda) != 0 -> oscillation
    |lambda| < 1, Im(lambda) ~ 0  -> decaying

  Entropy gate: H_norm > 0.8 blocks all entries
  KER filter: must agree with DMD regime direction

Philosophy: 道生一，一生二 — Complex dynamics decompose into simple modes.

Technical Indicators: DMD modes, KER, VDP, ATR, Bollinger Bands, Entropy
Category: regime
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from core.base_strategy import BaseStrategy


class DMDRegimeRoutingStrategy(BaseStrategy):
    """DMD regime detection + regime-specific routing for entry/exit."""

    strategy_description = (
        "DMDRegimeRouting: Dynamic Mode Decomposition for regime detection "
        "(trend/cycle/range) with regime-specific entry logic, entropy gate, "
        "KER confirmation, and regime-adaptive ATR trailing stops"
    )
    strategy_category = "regime"
    strategy_params_schema = {
        "dmd_window": {"type": "int", "default": 20, "label": "DMD Hankel window"},
        "dmd_rank": {"type": "int", "default": 5, "label": "SVD truncation rank"},
        "entropy_window": {"type": "int", "default": 50, "label": "Entropy window"},
        "entropy_bins": {"type": "int", "default": 10, "label": "Entropy bins"},
        "ker_period": {"type": "int", "default": 20, "label": "KER period"},
        "vdp_ema": {"type": "int", "default": 10, "label": "VDP EMA period"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "bb_period": {"type": "int", "default": 20, "label": "Bollinger period"},
        "bb_std": {"type": "float", "default": 2.0, "label": "Bollinger std dev"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "max_hold_trend": {"type": "int", "default": 50, "label": "Max hold (trend)"},
        "max_hold_cycle": {"type": "int", "default": 20, "label": "Max hold (cycle)"},
        "max_hold_range": {"type": "int", "default": 10, "label": "Max hold (range)"},
        "trail_mult_trend": {"type": "float", "default": 2.5, "label": "Trail ATR mult (trend)"},
        "trail_mult_cycle": {"type": "float", "default": 1.5, "label": "Trail ATR mult (cycle)"},
        "trail_mult_range": {"type": "float", "default": 1.8, "label": "Trail ATR mult (range)"},
        "score_threshold": {"type": "int", "default": 3, "label": "Min entry score"},
        "mom_threshold": {"type": "float", "default": 0.02, "label": "Momentum threshold"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI period for cycle"},
        "rsi_oversold": {"type": "float", "default": 30.0, "label": "RSI oversold"},
        "rsi_overbought": {"type": "float", "default": 70.0, "label": "RSI overbought"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.dmd_window = p.get('dmd_window', 20)
        self.dmd_rank = p.get('dmd_rank', 5)
        self.entropy_window = p.get('entropy_window', 50)
        self.entropy_bins = p.get('entropy_bins', 10)
        self.ker_period = p.get('ker_period', 20)
        self.vdp_ema = p.get('vdp_ema', 10)
        self.atr_period = p.get('atr_period', 14)
        self.bb_period = p.get('bb_period', 20)
        self.bb_std = p.get('bb_std', 2.0)
        self.hold_min = p.get('hold_min', 3)
        self.max_hold_trend = p.get('max_hold_trend', 50)
        self.max_hold_cycle = p.get('max_hold_cycle', 20)
        self.max_hold_range = p.get('max_hold_range', 10)
        self.trail_mult_trend = p.get('trail_mult_trend', 2.5)
        self.trail_mult_cycle = p.get('trail_mult_cycle', 1.5)
        self.trail_mult_range = p.get('trail_mult_range', 1.8)
        self.score_threshold = p.get('score_threshold', 3)
        self.mom_threshold = p.get('mom_threshold', 0.02)
        self.rsi_period = p.get('rsi_period', 14)
        self.rsi_oversold = p.get('rsi_oversold', 30.0)
        self.rsi_overbought = p.get('rsi_overbought', 70.0)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'dmd_window': 20, 'dmd_rank': 5,
            'entropy_window': 50, 'entropy_bins': 10,
            'ker_period': 20, 'vdp_ema': 10,
            'atr_period': 14, 'bb_period': 20, 'bb_std': 2.0,
            'hold_min': 3,
            'max_hold_trend': 50, 'max_hold_cycle': 20, 'max_hold_range': 10,
            'trail_mult_trend': 2.5, 'trail_mult_cycle': 1.5,
            'trail_mult_range': 1.8,
            'score_threshold': 3, 'mom_threshold': 0.02,
            'rsi_period': 14, 'rsi_oversold': 30.0, 'rsi_overbought': 70.0,
        }

    # ================================================================
    # DMD Core — Koopman Approximation via SVD
    # ================================================================

    def _run_dmd(self, close: np.ndarray) -> Optional[Dict]:
        """Run Dynamic Mode Decomposition on close prices.

        Methodology:
          1. Build Hankel matrix from normalized log-prices
          2. SVD-based DMD: X1 = U S V^T, A_tilde = U^T X2 V S^{-1}
          3. Eigendecompose A_tilde
          4. Classify regime using DMD spectral features + price metrics:
             - Trend efficiency (KER-like): high directional efficiency -> TREND
             - Oscillation score: dominant complex eigenvalue pairs -> CYCLE
             - Neither: -> RANGE

        Returns dict with keys:
          eigenvalues: complex array
          mode_amplitudes: float array (magnitude of each mode)
          dominant_idx: int index of highest-energy mode
          regime: str 'TREND' | 'CYCLE' | 'RANGE'
          growth_rate: float (|dominant eigenvalue|)
          phase: float (arg of dominant eigenvalue)
          direction: int +1 (bullish) or -1 (bearish)
        """
        n = len(close)
        if n < self.dmd_window + 2:
            return None

        prices = close[-(self.dmd_window + 1):]

        # --- Price-based regime metrics (supplements DMD) ---
        recent_window = min(20, len(prices) - 1)
        pr = prices[-(recent_window + 1):]
        net_change = abs(pr[-1] - pr[0])
        total_path = float(np.sum(np.abs(np.diff(pr))))
        efficiency = net_change / total_path if total_path > 1e-12 else 0.0

        # Trend strength: linear regression R^2 of recent prices
        t_idx = np.arange(recent_window + 1, dtype=float)
        t_mean = np.mean(t_idx)
        p_mean = np.mean(pr)
        ss_tot = float(np.sum((pr - p_mean) ** 2))
        slope_num = float(np.sum((t_idx - t_mean) * (pr - p_mean)))
        slope_den = float(np.sum((t_idx - t_mean) ** 2))
        if slope_den < 1e-12 or ss_tot < 1e-12:
            r_squared = 0.0
            lr_slope = 0.0
        else:
            lr_slope = slope_num / slope_den
            ss_res = float(np.sum((pr - (p_mean + lr_slope * (t_idx - t_mean))) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # --- DMD on normalized prices ---
        # Normalize to zero-mean, unit-variance for numerical stability
        log_p = np.log(prices)
        mu = np.mean(log_p)
        sigma = np.std(log_p)
        if sigma < 1e-12:
            sigma = 1.0
        norm_p = (log_p - mu) / sigma

        # Simple DMD: x(t+1) = A x(t) with Hankel embedding
        # Use raw time series as state vector (delay embedding)
        x1 = norm_p[:-1]
        x2 = norm_p[1:]

        # Build snapshot matrices with delay embedding
        delay = min(self.dmd_rank + 1, len(x1) // 3)
        if delay < 2 or len(x1) < delay + 2:
            # Fallback: direct DMD without delay embedding
            X1 = x1.reshape(1, -1)
            X2 = x2.reshape(1, -1)
        else:
            n_cols = len(x1) - delay
            X1 = np.zeros((delay, n_cols))
            X2 = np.zeros((delay, n_cols))
            for j in range(n_cols):
                X1[:, j] = x1[j:j + delay]
                X2[:, j] = x1[j + 1:j + delay + 1]

        # SVD of X1
        try:
            U, S, Vt = np.linalg.svd(X1, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        # Truncate to rank r
        r = min(self.dmd_rank, len(S))
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = Vt[:r, :].T

        S_inv = np.zeros(r)
        for i in range(r):
            S_inv[i] = 1.0 / S_r[i] if S_r[i] > 1e-12 else 0.0
        S_inv_diag = np.diag(S_inv)

        # Reduced Koopman operator
        A_tilde = U_r.T @ X2 @ V_r @ S_inv_diag

        # Eigendecompose
        try:
            eigenvalues, W = np.linalg.eig(A_tilde)
        except np.linalg.LinAlgError:
            return None

        # Mode amplitudes
        Phi = X2 @ V_r @ S_inv_diag @ W
        try:
            b = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        amplitudes = np.abs(b)
        total_energy = float(np.sum(amplitudes))
        if total_energy < 1e-12:
            return None

        # --- DMD spectral analysis ---
        # Count complex conjugate pairs (oscillatory modes)
        n_complex_pairs = 0
        complex_energy = 0.0
        classified = [False] * len(eigenvalues)
        for i in range(len(eigenvalues)):
            if classified[i]:
                continue
            eig_i = eigenvalues[i]
            if abs(np.imag(eig_i)) < 1e-8:
                continue
            # Look for conjugate pair
            for j in range(i + 1, len(eigenvalues)):
                if classified[j]:
                    continue
                eig_j = eigenvalues[j]
                # Check if conjugate: same real part, opposite imaginary
                if (abs(np.real(eig_i) - np.real(eig_j)) < 0.1 and
                        abs(np.imag(eig_i) + np.imag(eig_j)) < 0.1):
                    n_complex_pairs += 1
                    pair_energy = amplitudes[i] + amplitudes[j]
                    complex_energy += pair_energy
                    classified[i] = True
                    classified[j] = True
                    break

        complex_energy_frac = complex_energy / total_energy

        # --- Regime classification ---
        # Combine DMD spectral features with price metrics
        #
        # TREND: high directional efficiency (KER) + high R^2 + real eigenvalues dominate
        # CYCLE: complex conjugate pairs have high energy fraction
        # RANGE: low efficiency + low R^2 + low complex energy

        trend_score = 0.0
        cycle_score = 0.0
        range_score = 0.0

        # Efficiency-based scoring
        if efficiency > 0.4:
            trend_score += 2.0
        elif efficiency > 0.25:
            trend_score += 1.0
        elif efficiency < 0.15:
            range_score += 1.5

        # R-squared scoring
        if r_squared > 0.7:
            trend_score += 2.0
        elif r_squared > 0.4:
            trend_score += 1.0
        elif r_squared < 0.15:
            range_score += 1.5

        # DMD spectral scoring
        if complex_energy_frac > 0.5:
            cycle_score += 2.0
        elif complex_energy_frac > 0.3:
            cycle_score += 1.0

        # Real dominant mode -> trend confirmation
        dom_idx = int(np.argmax(amplitudes))
        dom_eig = eigenvalues[dom_idx]
        if abs(np.imag(dom_eig)) < 0.1 * max(abs(dom_eig), 1e-10):
            # Real dominant mode
            if abs(dom_eig) > 0.95:
                trend_score += 0.5

        # Final regime
        scores = {'TREND': trend_score, 'CYCLE': cycle_score, 'RANGE': range_score}
        regime = max(scores, key=scores.get)

        # If scores are tied or very close, use efficiency as tiebreaker
        max_score = max(scores.values())
        tied = [k for k, v in scores.items() if v == max_score]
        if len(tied) > 1:
            if efficiency > 0.3:
                regime = 'TREND'
            elif complex_energy_frac > 0.3:
                regime = 'CYCLE'
            else:
                regime = 'RANGE'

        dom_amp = amplitudes[dom_idx]
        growth = float(np.abs(dom_eig))
        phase = float(np.angle(dom_eig))

        # Direction from linear regression slope + recent price momentum
        if abs(lr_slope) > 1e-10:
            direction = 1 if lr_slope > 0 else -1
        else:
            recent_ret = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
            direction = 1 if recent_ret > 0 else -1

        return {
            'eigenvalues': eigenvalues,
            'mode_amplitudes': amplitudes,
            'dominant_idx': dom_idx,
            'regime': regime,
            'growth_rate': growth,
            'phase': phase,
            'direction': direction,
            'dominant_amplitude': float(dom_amp),
            'efficiency': float(efficiency),
            'r_squared': float(r_squared),
            'complex_energy_frac': float(complex_energy_frac),
            'trend_score': float(trend_score),
            'cycle_score': float(cycle_score),
            'range_score': float(range_score),
        }

    # ================================================================
    # Supporting Indicators
    # ================================================================

    def _calc_atr(self, data: pd.DataFrame) -> float:
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

    def _calc_entropy(self, close: np.ndarray) -> float:
        """Normalized Shannon entropy of log returns. Returns H_norm in [0, 1]."""
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
        h = float(-np.sum(probs * np.log2(probs)))
        h_max = np.log2(self.entropy_bins)
        return h / h_max if h_max > 0 else 1.0

    def _calc_ker(self, close: np.ndarray) -> Tuple[float, int]:
        """Kaufman Efficiency Ratio. Returns (ker_value, direction)."""
        period = self.ker_period
        if len(close) < period + 1:
            return 0.0, 0
        recent = close[-(period + 1):]
        net = abs(recent[-1] - recent[0])
        total = np.sum(np.abs(np.diff(recent)))
        if total < 1e-12:
            return 0.0, 0
        ker = net / total
        direction = 1 if recent[-1] > recent[0] else -1
        return float(ker), direction

    def _calc_vdp_delta(self, data: pd.DataFrame) -> Tuple[float, int]:
        """VDP EMA delta (volume-derived). Returns (ema_slope, direction)."""
        vol_col = None
        for vc in ['vol', 'volume', 'Volume']:
            if vc in data.columns:
                vol_col = vc
                break
        if vol_col is None:
            return 0.0, 0
        n = len(data)
        if n < self.vdp_ema + 5:
            return 0.0, 0
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

        k = 2.0 / (self.vdp_ema + 1)
        ema = np.zeros(n)
        ema[0] = deltas[0]
        for i in range(1, n):
            ema[i] = deltas[i] * k + ema[i - 1] * (1 - k)

        tail = ema[-5:]
        if np.std(tail) < 1e-10:
            return 0.0, 0
        slope = float(tail[-1] - tail[0]) / 5.0
        direction = 1 if slope > 0 else -1
        return slope, direction

    def _calc_rsi(self, close: np.ndarray) -> float:
        """RSI calculation."""
        if len(close) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(close[-(self.rsi_period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-12:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    def _calc_bollinger(self, close: np.ndarray) -> Tuple[float, float, float]:
        """Bollinger Bands. Returns (upper, middle, lower)."""
        if len(close) < self.bb_period:
            return 0.0, 0.0, 0.0
        tail = close[-self.bb_period:]
        mid = float(np.mean(tail))
        std = float(np.std(tail))
        return mid + self.bb_std * std, mid, mid - self.bb_std * std

    # ================================================================
    # Regime-Specific Entry Logic
    # ================================================================

    def _entry_trend(self, close: np.ndarray, data: pd.DataFrame,
                     atr: float, direction: int) -> int:
        """TREND regime: momentum + VDP confirmation. Returns score contribution."""
        if len(close) < 11 or atr <= 0:
            return 0
        mom_ret = (close[-1] - close[-11]) / close[-11] if close[-11] > 0 else 0

        score = 0
        # Momentum agrees with direction
        if direction == 1 and mom_ret > self.mom_threshold:
            score += 1
        elif direction == -1 and mom_ret < -self.mom_threshold:
            score += 1

        # VDP confirms
        _, vdp_dir = self._calc_vdp_delta(data)
        if vdp_dir == direction:
            score += 1

        # ATR-based volatility is adequate (not too flat)
        if atr / close[-1] > 0.005:
            score += 1

        return score

    def _entry_cycle(self, close: np.ndarray, data: pd.DataFrame,
                     atr: float, direction: int) -> int:
        """CYCLE regime: RSI oscillator at cycle extremes. Returns score contribution."""
        if len(close) < self.rsi_period + 1:
            return 0

        rsi = self._calc_rsi(close)
        score = 0

        # Buy at cycle bottom (oversold), sell at cycle top (overbought)
        if direction == 1 and rsi < self.rsi_oversold + 10:
            score += 1
        elif direction == -1 and rsi > self.rsi_overbought - 10:
            score += 1

        # VDP delta reversal confirmation
        _, vdp_dir = self._calc_vdp_delta(data)
        if vdp_dir == direction:
            score += 1

        # Phase coherence (growth rate near 1.0 = pure oscillation)
        # score already implied by CYCLE classification
        score += 1

        return score

    def _entry_range(self, close: np.ndarray, data: pd.DataFrame,
                     atr: float, direction: int) -> int:
        """RANGE regime: Bollinger mean reversion + VDP delta reversal."""
        if len(close) < self.bb_period:
            return 0

        bb_upper, bb_mid, bb_lower = self._calc_bollinger(close)
        if bb_mid <= 0:
            return 0

        score = 0
        price = close[-1]

        # Buy near lower BB, sell near upper BB
        if direction == 1 and price <= bb_lower + 0.1 * (bb_mid - bb_lower):
            score += 1
        elif direction == -1 and price >= bb_upper - 0.1 * (bb_upper - bb_mid):
            score += 1

        # VDP delta reversal
        vdp_slope, vdp_dir = self._calc_vdp_delta(data)
        if vdp_dir == direction:
            score += 1

        # Price is within bands (not a breakout)
        if bb_lower < price < bb_upper:
            score += 1

        return score

    # ================================================================
    # DMD Mode Change Detection
    # ================================================================

    def _detect_mode_change(self, close: np.ndarray, prev_regime: str) -> bool:
        """Check if DMD regime has shifted from the previous regime."""
        dmd_result = self._run_dmd(close)
        if dmd_result is None:
            return True  # Can't compute — conservative exit
        return dmd_result['regime'] != prev_regime

    # ================================================================
    # Core Evaluation
    # ================================================================

    def _evaluate(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Run full DMD regime detection + regime-specific entry evaluation.

        Returns dict with keys: regime, direction, score, atr, dmd_info
        or None if no entry signal.
        """
        close = data['close'].values.astype(float)
        min_len = max(self.dmd_window + 10, self.entropy_window + 10,
                      self.ker_period + 10, self.bb_period + 5, 60)
        if len(close) < min_len:
            return None

        # --- Entropy gate ---
        h_norm = self._calc_entropy(close)
        if h_norm > 0.8:
            return None

        # --- DMD regime detection ---
        dmd_result = self._run_dmd(close)
        if dmd_result is None:
            return None

        regime = dmd_result['regime']
        direction = dmd_result['direction']
        growth = dmd_result['growth_rate']

        # --- KER confirmation ---
        ker_val, ker_dir = self._calc_ker(close)
        if ker_val < 0.1:
            # Very low efficiency — no clear signal
            return None
        if ker_dir != direction:
            # KER disagrees with DMD direction
            return None

        # --- ATR ---
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # --- Regime-specific entry scoring ---
        score = 0

        # DMD mode growth contribution
        if regime == 'TREND' and growth > 1.0:
            score += 1
        elif regime == 'CYCLE':
            score += 1
        elif regime == 'RANGE' and growth < 1.0:
            score += 1

        # KER agreement
        if ker_dir == direction and ker_val > 0.15:
            score += 1

        # Regime-specific sub-strategy
        if regime == 'TREND':
            score += self._entry_trend(close, data, atr, direction)
        elif regime == 'CYCLE':
            score += self._entry_cycle(close, data, atr, direction)
        elif regime == 'RANGE':
            score += self._entry_range(close, data, atr, direction)

        if score < self.score_threshold:
            return None

        return {
            'regime': regime,
            'direction': direction,
            'score': score,
            'atr': atr,
            'dmd_info': dmd_result,
        }

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
        position_regime = None  # track regime at entry for mode-change exit
        high_water = 0.0
        low_water = float('inf')

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                # --- Entry scan ---
                best_score = 0
                best_sym = None
                best_dir = 0
                best_regime = None
                best_atr = 0.0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    result = self._evaluate(sym, hist)
                    if result is None:
                        continue
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_sym = sym
                        best_dir = result['direction']
                        best_regime = result['regime']
                        best_atr = result['atr']

                if best_sym is not None and best_score >= self.score_threshold:
                    entry_price = float(current_bars[
                        current_bars['symbol'] == best_sym].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym,
                                            price=entry_price,
                                            regime=best_regime, score=best_score)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym,
                                            price=entry_price,
                                            regime=best_regime, score=best_score)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    position_regime = best_regime
                    high_water = 0.0
                    low_water = float('inf')

            else:
                # --- Exit management ---
                days_held = len([t for t in unique_times
                                 if buy_time < t <= current_time])
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # Track high/low water mark
                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 \
                        else current_price
                else:
                    low_water = min(low_water, current_price) \
                        if low_water < float('inf') else current_price

                if days_held < self.hold_min:
                    continue

                hist = data[(data['symbol'] == current_holding) &
                            (data.index <= current_time)]
                close_arr = hist['close'].values.astype(float)
                atr_val = self._calc_atr(hist)
                should_exit = False

                # 1. DMD mode change exit
                if len(close_arr) >= self.dmd_window + 10:
                    if self._detect_mode_change(close_arr, position_regime):
                        should_exit = True

                # 2. ATR trailing stop (regime-dependent)
                if atr_val > 0 and not should_exit:
                    if position_regime == 'TREND':
                        trail_mult = self.trail_mult_trend
                        max_hold = self.max_hold_trend
                    elif position_regime == 'CYCLE':
                        trail_mult = self.trail_mult_cycle
                        max_hold = self.max_hold_cycle
                    else:
                        trail_mult = self.trail_mult_range
                        max_hold = self.max_hold_range

                    if position_dir == 1 and high_water > 0:
                        if current_price < high_water - trail_mult * atr_val:
                            should_exit = True
                    elif position_dir == -1 and low_water < float('inf'):
                        if current_price > low_water + trail_mult * atr_val:
                            should_exit = True
                else:
                    max_hold = self.max_hold_trend  # fallback

                # 3. Max hold
                if days_held >= max_hold:
                    should_exit = True

                if should_exit:
                    if position_dir == 1:
                        self._record_signal(current_time, 'sell', current_holding,
                                            price=current_price)
                    else:
                        self._record_signal(current_time, 'buy', current_holding,
                                            price=current_price)
                    current_holding = None
                    buy_time = None
                    position_dir = 0
                    position_regime = None
                    high_water = 0.0
                    low_water = float('inf')

        print(f"DMDRegimeRouting: generated {len(self.signals)} signals")
        return self.signals

    # ================================================================
    # Screen — Quick Real-Time Selection
    # ================================================================

    def screen(self) -> Dict:
        """Quick screen based on latest data slice."""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'
        sym = data['symbol'].iloc[0]
        min_len = max(self.dmd_window + 10, self.entropy_window + 10,
                      self.ker_period + 10, self.bb_period + 5, 60)
        if len(data) < min_len:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}

        price = float(data['close'].iloc[-1])
        result = self._evaluate(sym, data)
        if result is None:
            return {'action': 'hold', 'reason': 'no DMD signal', 'price': price}

        regime = result['regime']
        direction = result['direction']
        score = result['score']

        if score >= self.score_threshold:
            action = 'buy' if direction == 1 else 'sell'
            return {
                'action': action,
                'reason': f'DMD {regime} dir={direction} score={score}',
                'price': price,
            }
        return {'action': 'hold',
                'reason': f'DMD {regime} score={score}<{self.score_threshold}',
                'price': price}
