"""
DMDCycleStrategy - Dynamic Mode Decomposition Cycle Detection (易经周期)

Philosophy: "反复其道，七日来复" - The way repeats, returning after seven days.
Uses DMD eigenvalue decomposition to classify market regimes into I Ching modes,
then applies trend-following or mean-reversion accordingly.

Uses rank-transformed features (percentile) instead of raw values for robustness.
Falls back to Kaufman Efficiency Ratio regime detection if DMD fails.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from core.base_strategy import BaseStrategy


class DMDCycleStrategy(BaseStrategy):
    """Dynamic Mode Decomposition cycle detection with I Ching regime classification."""

    strategy_description = (
        "反复其道七日来复 - DMD eigenvalue decomposition classifies market into "
        "乾卦 (growing/trend), 否卦 (decaying/exit), neutral, or oscillation modes. "
        "Uses rank-transformed features. Bull energy ratio determines regime."
    )
    strategy_category = "adaptive"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'dmd_window': 30,          # snapshot matrix window length
            'dmd_rank': 5,             # SVD truncation rank
            'fast_ema': 9,             # fast EMA for trend mode
            'slow_ema': 21,            # slow EMA for trend mode
            'rsi_period': 14,          # RSI period for range mode
            'rsi_ob': 70,             # RSI overbought
            'rsi_os': 30,             # RSI oversold
            'atr_period': 14,          # ATR period
            'atr_stop_mult': 2.5,      # ATR trailing stop multiplier
            'bull_threshold': 0.6,      # rho_bull above this = trend mode
            'bear_threshold': 0.3,      # rho_bull below this = range mode
            'min_score': 1.5,          # minimum score to emit signal
            'ker_period': 10,          # Kaufman Efficiency Ratio period (fallback)
        }

    def generate_signals(self) -> List[Dict]:
        df = self.data.copy()
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'DEFAULT'
        p = self.params

        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)

        # --- Rank-transformed returns (percentile) ---
        returns = np.diff(close, prepend=close[0])
        returns[0] = 0.0
        rank_returns = self._rank_transform(returns)

        # --- Compute ATR ---
        atr = self._compute_atr(df, p['atr_period'])

        # --- Compute EMAs for trend mode ---
        ema_fast = self._ema(close, p['fast_ema'])
        ema_slow = self._ema(close, p['slow_ema'])

        # --- Compute RSI for range mode ---
        rsi = self._compute_rsi(close, p['rsi_period'])

        # --- DMD computation over rolling window ---
        warmup = max(p['dmd_window'] + p['dmd_rank'], p['slow_ema'], p['atr_period'], 40)
        dmd_window = p['dmd_window']
        dmd_rank = p['dmd_rank']

        # State tracking
        position = 0       # 0=flat, 1=long, -1=short
        stop_price = 0.0
        use_fallback = False

        for i in range(warmup, len(df)):
            ts = df.index[i]
            price = close[i]
            atr_now = atr[i]

            if np.isnan(atr_now) or atr_now < 1e-10:
                continue

            # --- DMD eigenvalue analysis on rank-transformed returns ---
            rho_bull = 0.5  # default neutral
            has_oscillation = False
            dmd_ok = False

            if i >= dmd_window + 1:
                try:
                    rho_bull, has_oscillation = self._compute_dmd_modes(
                        rank_returns[i - dmd_window: i], dmd_rank
                    )
                    dmd_ok = True
                except Exception:
                    dmd_ok = False

            # --- Fallback: Kaufman Efficiency Ratio regime detection ---
            if not dmd_ok:
                use_fallback = True
                ker = self._compute_ker(close, p['ker_period'], i)
                if np.isnan(ker):
                    continue
                # Map KER to bull energy: KER near 1 = strong trend, near 0 = noise
                if ker > 0.4:
                    rho_bull = 0.7  # trend mode
                elif ker < 0.15:
                    rho_bull = 0.2  # range mode
                else:
                    rho_bull = 0.45  # neutral

            # --- Trailing stop for existing position ---
            if position == 1:
                new_stop = price - p['atr_stop_mult'] * atr_now
                stop_price = max(stop_price, new_stop)
                if price <= stop_price:
                    self._record_signal(ts, 'sell', symbol, price,
                                       score=0, reason='atr_trailing_stop')
                    position = 0
                    continue

            elif position == -1:
                new_stop = price + p['atr_stop_mult'] * atr_now
                stop_price = min(stop_price, new_stop)
                if price >= stop_price:
                    self._record_signal(ts, 'buy', symbol, price,
                                       score=0, reason='atr_trailing_stop')
                    position = 0
                    continue

            # --- Regime-based signal generation ---
            action = None
            direction_strength = 0.0
            reason = ''

            if rho_bull > p['bull_threshold']:
                # Trend mode (乾卦 - creative/heaven): EMA crossover
                # direction_strength: ATR-normalized EMA gap (typically 0.2-1.0)
                if i >= 2:
                    prev_fast = ema_fast[i - 1]
                    prev_slow = ema_slow[i - 1]
                    cur_fast = ema_fast[i]
                    cur_slow = ema_slow[i]

                    if not (np.isnan(prev_fast) or np.isnan(prev_slow) or
                            np.isnan(cur_fast) or np.isnan(cur_slow)):
                        ema_gap = cur_fast - cur_slow
                        if cur_fast > cur_slow and prev_fast <= prev_slow:
                            action = 'buy'
                            direction_strength = abs(ema_gap) / max(atr_now, 1e-10)
                            reason = 'trend_ema_golden_cross_乾卦'
                        elif cur_fast < cur_slow and prev_fast >= prev_slow:
                            action = 'sell'
                            direction_strength = abs(ema_gap) / max(atr_now, 1e-10)
                            reason = 'trend_ema_death_cross_乾卦'

            elif rho_bull < p['bear_threshold']:
                # Range mode (否卦 - stagnation): RSI mean reversion
                # direction_strength: RSI excursion normalized to [0, 1]
                rsi_now = rsi[i]
                if not np.isnan(rsi_now):
                    if rsi_now < p['rsi_os']:
                        action = 'buy'
                        direction_strength = (p['rsi_os'] - rsi_now) / p['rsi_os']
                        reason = f'range_rsi_oversold_否卦_rsi={rsi_now:.1f}'
                    elif rsi_now > p['rsi_ob']:
                        action = 'sell'
                        direction_strength = (rsi_now - p['rsi_ob']) / (100.0 - p['rsi_ob'])
                        reason = f'range_rsi_overbought_否卦_rsi={rsi_now:.1f}'

            else:
                # Neutral zone: oscillation cycle detection
                if has_oscillation and i >= 3:
                    # Simple cycle: buy after 3 consecutive down bars, sell after 3 up
                    recent = close[i - 3:i + 1]
                    if all(recent[j] < recent[j - 1] for j in range(1, 4)):
                        action = 'buy'
                        direction_strength = 0.3
                        reason = 'oscillation_cycle_buy'
                    elif all(recent[j] > recent[j - 1] for j in range(1, 4)):
                        action = 'sell'
                        direction_strength = 0.3
                        reason = 'oscillation_cycle_sell'

            # --- Score calculation ---
            if action is not None:
                score = rho_bull * direction_strength * 10.0
                if has_oscillation and rho_bull > p['bear_threshold']:
                    score += 1.0  # oscillation bonus

                if score >= p['min_score']:
                    if action == 'buy' and position <= 0:
                        self._record_signal(ts, 'buy', symbol, price,
                                           score=round(score, 2), reason=reason)
                        position = 1
                        stop_price = price - p['atr_stop_mult'] * atr_now
                    elif action == 'sell' and position >= 0:
                        self._record_signal(ts, 'sell', symbol, price,
                                           score=round(score, 2), reason=reason)
                        position = -1
                        stop_price = price + p['atr_stop_mult'] * atr_now

        # Close open position at end of data
        if position != 0 and len(df) > 0:
            last_ts = df.index[-1]
            last_price = close[-1]
            self._record_signal(last_ts, 'sell' if position == 1 else 'buy',
                                symbol, last_price, score=0, reason='end_of_data_close')

        return self.signals

    # -----------------------------------------------------------------
    # DMD core
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_dmd_modes(data_window: np.ndarray, rank: int):
        """
        Compute DMD on a 1D data window (rank-transformed returns).
        Returns (rho_bull, has_oscillation).

        Construct snapshot matrix from the scalar series, apply SVD,
        extract eigenvalues of the reduced Koopman matrix, classify modes.
        """
        n = len(data_window)
        if n < rank + 2:
            return 0.5, False

        # Build Hankel-like snapshot matrix from the scalar series
        # X = columns of consecutive overlapping windows of length `rank`
        num_cols = n - rank
        if num_cols < 2:
            return 0.5, False

        X = np.empty((rank, num_cols - 1), dtype=float)
        X_next = np.empty((rank, num_cols - 1), dtype=float)
        for j in range(num_cols - 1):
            X[:, j] = data_window[j: j + rank]
            X_next[:, j] = data_window[j + 1: j + rank + 1]

        # SVD of X
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.5, False

        # Truncate to `rank`
        r = min(rank, len(S))
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]

        # Pseudoinverse weights
        S_inv = np.where(S_r > 1e-12, 1.0 / S_r, 0.0)

        # Reduced Koopman approximation: A_tilde = U_r^T @ X_next @ V_r @ S_inv
        A_tilde = U_r.T @ X_next @ Vt_r.T @ np.diag(S_inv)

        # Eigenvalues
        try:
            eigenvalues = np.linalg.eigvals(A_tilde)
        except np.linalg.LinAlgError:
            return 0.5, False

        # Mode amplitudes (projection onto DMD modes)
        # b_k = U_r^T @ x1 (first column of X_next)
        x1 = X_next[:, 0]
        b = U_r.T @ x1
        b_sq = np.abs(b) ** 2
        total_bsq = np.sum(b_sq)

        if total_bsq < 1e-15:
            return 0.5, False

        # Bull energy ratio: sum of |b_k|^2 for growing modes (|lambda| > 1)
        mag = np.abs(eigenvalues)
        bull_mask = mag > 1.0
        rho_bull = np.sum(b_sq[bull_mask]) / total_bsq

        # Detect oscillation: complex eigenvalues with significant imaginary part
        imag_part = np.abs(np.imag(eigenvalues))
        has_oscillation = bool(np.any(imag_part > 0.1))

        # Clamp
        rho_bull = float(np.clip(rho_bull, 0.0, 1.0))
        return rho_bull, has_oscillation

    # -----------------------------------------------------------------
    # Feature helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _rank_transform(arr: np.ndarray) -> np.ndarray:
        """Convert array values to percentile ranks [0, 1]."""
        n = len(arr)
        result = np.empty(n, dtype=float)
        # Rolling rank over a window for stationarity
        window = min(n, 60)
        for i in range(n):
            start = max(0, i - window)
            segment = arr[start: i + 1]
            if len(segment) == 0:
                result[i] = 0.5
            else:
                # Percentile rank
                result[i] = np.sum(segment < arr[i]) / max(len(segment) - 1, 1)
        return result

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA over a numpy array."""
        result = np.empty_like(arr, dtype=float)
        result[:] = np.nan
        if len(arr) == 0:
            return result
        alpha = 2.0 / (period + 1)
        start = 0
        for j in range(len(arr)):
            if not np.isnan(arr[j]):
                start = j
                break
        result[start] = arr[start]
        for j in range(start + 1, len(arr)):
            if np.isnan(arr[j]):
                result[j] = result[j - 1]
            else:
                if np.isnan(result[j - 1]):
                    result[j] = arr[j]
                else:
                    result[j] = alpha * arr[j] + (1.0 - alpha) * result[j - 1]
        return result

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
        """Compute Average True Range."""
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        tr = np.empty(len(df), dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
        return atr

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI."""
        n = len(close)
        rsi = np.full(n, np.nan, dtype=float)
        if n < period + 1:
            return rsi

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss < 1e-12:
            rsi[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss < 1e-12:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

        return rsi

    @staticmethod
    def _compute_ker(close: np.ndarray, period: int, idx: int) -> float:
        """
        Kaufman Efficiency Ratio at index idx.
        KER = |close[idx] - close[idx-period]| / sum(|close[j] - close[j-1]|)
        """
        if idx < period:
            return np.nan
        net_change = abs(close[idx] - close[idx - period])
        total_volatility = np.sum(np.abs(np.diff(close[idx - period: idx + 1])))
        if total_volatility < 1e-12:
            return 0.0
        return net_change / total_volatility


__all__ = ['DMDCycleStrategy']
