"""
DMDSpectralStrategy - Dynamic Mode Decomposition Spectral Trading

Philosophy: "观其复，察其微" - Observe the modes, discern the subtle patterns.
Uses compressed sensing DMD to decompose price dynamics into trend, oscillation, and noise modes.
Trades when dominant mode has positive growth rate (expanding mode = trending).

Key innovations:
- Compressed sensing DMD (more stable than standard DMD)
- Mode growth rate ranking (trade strongest growing mode direction)
- Hankel matrix embedding for better temporal structure
- Combine with KER for regime confirmation
- Better exit logic using mode decay detection

References:
- Section 27: Dynamic Mode Decomposition in probability_theory.md
- Section 30: Spectral Analysis (FFT) comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy import sparse
from scipy.sparse.linalg import svds

from core.base_strategy import BaseStrategy


class DMDSpectralStrategy(BaseStrategy):
    """Dynamic Mode Decomposition spectral trading with compressed sensing."""

    strategy_description = (
        "观其复察其微 - Compressed sensing DMD decomposes price dynamics into "
        "trend, oscillation, and noise modes. Trades when dominant mode has "
        "positive growth rate. Uses Hankel embedding and KER confirmation."
    )
    strategy_category = "ml"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            # DMD parameters
            'dmd_window': 40,           # snapshot matrix window length
            'dmd_rank': 8,              # SVD truncation rank
            'cs_alpha': 0.1,            # compressed sensing regularization
            'hankel_delay': 5,         # Hankel matrix delay parameter

            # Mode analysis
            'growth_threshold': 1.02,   # minimum growth rate to consider expanding
            'decay_threshold': 0.98,   # maximum decay rate to consider contracting
            'oscillation_threshold': 0.1,  # minimum imaginary part for oscillation
            'min_mode_energy': 0.05,    # minimum energy for a mode to be considered

            # Trading parameters
            'fast_ema': 12,            # fast EMA for confirmation
            'slow_ema': 26,            # slow EMA for confirmation
            'atr_period': 14,          # ATR period for stop loss
            'atr_stop_mult': 2.0,      # ATR trailing stop multiplier
            'ker_period': 20,          # KER period for regime confirmation
            'ker_trend_threshold': 0.4, # KER threshold for trend confirmation

            # Signal parameters
            'min_score': 2.0,          # minimum score to emit signal
            'score_mult': 10.0,        # score multiplier for strong signals
            'mode_rank_bonus': 0.5,    # bonus for top-ranked growing mode

            # Exit parameters
            'decay_exit_threshold': 0.95,  # exit if dominant mode decays below this
            'oscillation_exit': True,  # exit if strong oscillation detected
            'max_hold_days': 30,       # maximum hold period
        }

    def generate_signals(self) -> List[Dict]:
        df = self.data.copy()
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'DEFAULT'
        p = self.params

        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)

        # Compute indicators
        ema_fast = self._ema(close, p['fast_ema'])
        ema_slow = self._ema(close, p['slow_ema'])
        atr = self._compute_atr(df, p['atr_period'])
        ker = self._compute_ker(close, p['ker_period'])

        # DMD parameters
        dmd_window = p['dmd_window']
        dmd_rank = p['dmd_rank']
        hankel_delay = p['hankel_delay']

        # State tracking
        position = 0       # 0=flat, 1=long, -1=short
        stop_price = 0.0
        entry_price = 0.0
        entry_time = None
        dominant_mode_energy = 0.0
        dominant_mode_growth = 1.0

        warmup = max(dmd_window + dmd_rank, p['slow_ema'], p['atr_period'], 50)

        for i in range(warmup, len(df)):
            ts = df.index[i]
            price = close[i]
            atr_now = atr[i]
            ker_now = ker[i] if i < len(ker) else 0.5

            if np.isnan(atr_now) or atr_now < 1e-10:
                continue

            # --- DMD computation ---
            dmd_result = self._compute_compressed_sensing_dmd(
                close, i, dmd_window, dmd_rank, hankel_delay, p['cs_alpha']
            )

            if dmd_result is None:
                continue

            # Extract mode information
            modes = dmd_result['modes']
            eigenvalues = dmd_result['eigenvalues']
            amplitudes = dmd_result['amplitudes']
            energies = dmd_result['energies']

            # Rank modes by energy
            mode_rank = np.argsort(energies)[::-1]  # descending order

            # Find dominant mode
            dominant_idx = mode_rank[0]
            dominant_mode_energy = energies[dominant_idx]
            dominant_mode_growth = np.abs(eigenvalues[dominant_idx])

            # Check for oscillation
            has_oscillation = np.abs(np.imag(eigenvalues[dominant_idx])) > p['oscillation_threshold']

            # --- Regime confirmation with KER ---
            regime_confirmed = False
            if ker_now >= p['ker_trend_threshold']:
                regime_confirmed = True

            # --- Signal generation ---
            action = None
            direction_strength = 0.0
            reason = ''

            # Check if dominant mode is growing and has enough energy
            if (dominant_mode_growth > p['growth_threshold'] and
                dominant_mode_energy > p['min_mode_energy'] and
                regime_confirmed):

                # Strong growing mode - potential long entry
                if i >= 2:
                    # Check EMA confirmation
                    if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                        action = 'buy'
                        # Score based on growth rate, energy, and rank
                        rank_bonus = p['mode_rank_bonus'] if dominant_idx == 0 else 0
                        direction_strength = (dominant_mode_growth - 1.0) * p['score_mult'] + rank_bonus
                        reason = f'growing_mode_{dominant_idx+1}_growth={dominant_mode_growth:.3f}'

            # Check for short position on strong decay
            elif (dominant_mode_growth < p['decay_threshold] and
                  dominant_mode_energy > p['min_mode_energy']):

                # Strong decaying mode - potential short entry
                if i >= 2:
                    if ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                        action = 'sell'
                        direction_strength = (1.0 - dominant_mode_growth) * p['score_mult']
                        reason = f'decaying_mode_{dominant_idx+1}_growth={dominant_mode_growth:.3f}'

            # --- Exit conditions ---
            if position != 0:
                # Check for mode decay exit
                if dominant_mode_growth < p['decay_exit_threshold']:
                    if position == 1:
                        self._record_signal(ts, 'sell', symbol, price,
                                           score=0, reason='mode_decay_exit')
                        position = 0
                        continue
                    elif position == -1:
                        self._record_signal(ts, 'buy', symbol, price,
                                           score=0, reason='mode_decay_exit')
                        position = 0
                        continue

                # Check for oscillation exit
                if p['oscillation_exit'] and has_oscillation:
                    if position == 1:
                        self._record_signal(ts, 'sell', symbol, price,
                                           score=0, reason='oscillation_exit')
                        position = 0
                        continue
                    elif position == -1:
                        self._record_signal(ts, 'buy', symbol, price,
                                           score=0, reason='oscillation_exit')
                        position = 0
                        continue

                # Check max hold days
                if entry_time is not None:
                    hold_days = (ts - entry_time).days
                    if hold_days > p['max_hold_days']:
                        exit_action = 'sell' if position == 1 else 'buy'
                        self._record_signal(ts, exit_action, symbol, price,
                                           score=0, reason=f'max_hold_{hold_days}d')
                        position = 0
                        continue

                # Trailing stop
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

            # --- Record entry signal ---
            if action is not None:
                score = direction_strength
                if score >= p['min_score']:
                    if action == 'buy' and position <= 0:
                        self._record_signal(ts, 'buy', symbol, price,
                                           score=round(score, 2), reason=reason)
                        position = 1
                        stop_price = price - p['atr_stop_mult'] * atr_now
                        entry_price = price
                        entry_time = ts
                    elif action == 'sell' and position >= 0:
                        self._record_signal(ts, 'sell', symbol, price,
                                           score=round(score, 2), reason=reason)
                        position = -1
                        stop_price = price + p['atr_stop_mult'] * atr_now
                        entry_price = price
                        entry_time = ts

        # Close open position at end of data
        if position != 0 and len(df) > 0:
            last_ts = df.index[-1]
            last_price = close[-1]
            exit_action = 'sell' if position == 1 else 'buy'
            self._record_signal(last_ts, exit_action, symbol, last_price,
                               score=0, reason='end_of_data_close')

        return self.signals

    # -----------------------------------------------------------------
    # Compressed Sensing DMD Core
    # -----------------------------------------------------------------

    def _compute_compressed_sensing_dmd(self, prices, idx, window, rank, delay, alpha):
        """
        Compute compressed sensing DMD on a price window.

        Args:
            prices: Full price array
            idx: Current index
            window: Window size for DMD
            rank: SVD truncation rank
            delay: Hankel matrix delay parameter
            alpha: Compressed sensing regularization

        Returns:
            Dict with modes, eigenvalues, amplitudes, energies
        """
        # Extract window
        start_idx = idx - window
        if start_idx < 0:
            return None

        window_prices = prices[start_idx:idx]
        n = len(window_prices)

        if n < rank + delay:
            return None

        # Build Hankel matrix
        hankel_rows = n - delay
        hankel_cols = delay
        X = np.zeros((hankel_rows, hankel_cols))

        for i in range(hankel_cols):
            X[:, i] = window_prices[i:i + hankel_rows]

        X_next = np.roll(X, -1, axis=1)
        X_next = X_next[:, :-1]  # Remove last column

        # Compressed sensing DMD using sparse SVD
        try:
            # Add regularization matrix for compressed sensing
            reg_matrix = alpha * sparse.eye(min(X.shape))
            X_reg = X + reg_matrix

            # Compute sparse SVD
            U, S, Vt = svds(X_reg, k=min(rank, min(X.shape)-1))

            # Truncate to desired rank
            r = min(rank, len(S))
            U_r = U[:, :r]
            S_r = S[:r]
            Vt_r = Vt[:r, :]

            # Pseudoinverse
            S_inv = np.where(S_r > 1e-12, 1.0 / S_r, 0.0)

            # Reduced Koopman operator
            A_tilde = U_r.T @ X_next @ Vt_r.T @ np.diag(S_inv)

            # Eigenvalues and eigenvectors
            eigenvalues = np.linalg.eigvals(A_tilde)

            # DMD modes
            Phi = X_next @ Vt_r.T @ np.diag(S_inv) @ np.linalg.inv(np.diag(np.linalg.eigvals(A_tilde)))

            # Amplitudes
            x1 = X_next[:, 0]
            b = U_r.T @ x1

            # Energy of each mode
            amplitudes = np.abs(b)
            energies = (amplitudes ** 2) / np.sum(amplitudes ** 2)

            return {
                'modes': Phi,
                'eigenvalues': eigenvalues,
                'amplitudes': amplitudes,
                'energies': energies
            }

        except Exception as e:
            # Fallback to standard SVD if compressed sensing fails
            try:
                U, S, Vt = np.linalg.svd(X, full_matrices=False)

                r = min(rank, len(S))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]

                S_inv = np.where(S_r > 1e-12, 1.0 / S_r, 0.0)

                A_tilde = U_r.T @ X_next @ Vt_r.T @ np.diag(S_inv)

                eigenvalues = np.linalg.eigvals(A_tilde)

                Phi = X_next @ Vt_r.T @ np.diag(S_inv)

                x1 = X_next[:, 0]
                b = U_r.T @ x1

                amplitudes = np.abs(b)
                energies = (amplitudes ** 2) / np.sum(amplitudes ** 2)

                return {
                    'modes': Phi,
                    'eigenvalues': eigenvalues,
                    'amplitudes': amplitudes,
                    'energies': energies
                }
            except:
                return None

    # -----------------------------------------------------------------
    # Technical Indicators
    # -----------------------------------------------------------------

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
    def _compute_ker(close: np.ndarray, period: int) -> np.ndarray:
        """Kaufman Efficiency Ratio over array."""
        n = len(close)
        ker = np.full(n, np.nan, dtype=float)
        if n < period:
            return ker

        for i in range(period, n):
            net_change = abs(close[i] - close[i - period])
            total_volatility = np.sum(np.abs(np.diff(close[i - period: i + 1])))
            if total_volatility < 1e-12:
                ker[i] = 0.0
            else:
                ker[i] = net_change / total_volatility

        return ker


__all__ = ['DMDSpectralStrategy']