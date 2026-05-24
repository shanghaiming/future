"""
KernelDensityStrategy - Non-parametric Distribution Estimation for Trading

Philosophy: "Price outliers are opportunities" - Use kernel density estimation
to identify when price deviates significantly from recent distribution.

Core idea:
- Compute KDE of rolling returns (60 bars, Gaussian kernel)
- Low density = price is an outlier = mean reversion opportunity
- High density + momentum = trend continuation
- Silverman's rule for optimal bandwidth selection

Mathematical foundation:
Kernel density estimation: f(x) = (1/nh) * Σ K((x-xi)/h)
Gaussian kernel: K(u) = (1/√(2π)) * exp(-u²/2)
Silverman's bandwidth: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from typing import Dict, List, Any

from core.base_strategy import BaseStrategy


class KernelDensityStrategy(BaseStrategy):
    """Kernel Density Estimation for non-parametric trading signals."""

    strategy_description = (
        "Kernel Density Trading - Identify price outliers using non-parametric KDE. "
        "Low density = mean reversion, High density + momentum = trend continuation."
    )
    strategy_category = "ml"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'kde_period': 60,           # Lookback period for KDE estimation
            'silverman_factor': 0.9,    # Silverman's rule factor (usually 0.9)
            'density_threshold': 0.15,  # Threshold for low density (0-1)
            'momentum_period': 20,      # Period for momentum calculation
            'atr_period': 14,          # ATR calculation period
            'atr_stop_mult': 2.5,      # ATR trailing stop multiplier
            'vdp_period': 30,           # Period for VDP calculation
            'min_signal_score': 0.1,   # Minimum signal score threshold
            'position_size': 1.0,      # Position sizing factor
        }

    def validate_params(self):
        """Validate strategy parameters."""
        p = self.params

        if p['kde_period'] < 10:
            raise ValueError("KDE period must be at least 10")
        if p['density_threshold'] <= 0 or p['density_threshold'] >= 1:
            raise ValueError("Density threshold must be between 0 and 1")
        if p['momentum_period'] < 5:
            raise ValueError("Momentum period must be at least 5")
        if p['atr_period'] < 5:
            raise ValueError("ATR period must be at least 5")

    def generate_signals(self) -> List[Dict]:
        df = self.data.copy()
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'DEFAULT'
        p = self.params

        # Calculate returns for KDE
        df['returns'] = df['close'].pct_change()

        # Calculate technical indicators
        df['momentum'] = self._compute_momentum(df, p['momentum_period'])
        atr = self._compute_atr(df, p['atr_period'])
        df['atr'] = atr

        # Calculate VDP for direction confirmation
        if 'volume' in df.columns and df['volume'].notna().any():
            df['vdp'] = self._compute_vdp(df, p['vdp_period'])
            has_volume = True
        else:
            df['vdp'] = 0.0
            has_volume = False

        # Compute KDE density values
        df['density'], df['density_ratio'] = self._compute_kde_density(df, p['kde_period'], p['silverman_factor'])

        # Initialize position tracking
        position = 0       # 0=flat, 1=long, -1=short
        entry_price = 0.0
        stop_price = 0.0
        warmup = max(p['kde_period'], p['momentum_period'], p['atr_period'], p['vdp_period'], 100)

        for i in range(warmup, len(df)):
            ts = df.index[i]
            price = df['close'].iloc[i]
            density = df['density'].iloc[i]
            density_ratio = df['density_ratio'].iloc[i]
            momentum = df['momentum'].iloc[i]
            atr_now = df['atr'].iloc[i]
            vdp = df['vdp'].iloc[i]

            if np.isnan(density) or np.isnan(momentum) or np.isnan(atr_now):
                continue

            # --- ATR trailing stop for existing position ---
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

            # --- Generate new trading signals ---
            action = 'hold'
            reason = 'no_signal'
            score = 0.0

            # Condition 1: Low density = mean reversion opportunity
            if density < p['density_threshold']:
                # Mean reversion: low price → buy, high price → sell
                if momentum < -0.1:  # Price is decreasing
                    action = 'buy'
                    reason = 'low_density_mean_reversion'
                    score = self._compute_signal_score(
                        density_ratio=p['density_threshold'] / density,
                        momentum=abs(momentum),
                        atr_ratio=atr_now / np.mean(atr[:i]),
                        vdp=vdp if has_volume else 0.0,
                        signal_type='mean_reversion'
                    )

                elif momentum > 0.1:  # Price is increasing
                    action = 'sell'
                    reason = 'low_density_mean_reversion'
                    score = self._compute_signal_score(
                        density_ratio=p['density_threshold'] / density,
                        momentum=momentum,
                        atr_ratio=atr_now / np.mean(atr[:i]),
                        vdp=-vdp if has_volume else 0.0,
                        signal_type='mean_reversion'
                    )

            # Condition 2: High density + momentum = trend continuation
            elif density > p['density_threshold'] * 2:  # High density region
                if momentum > 0.2 and vdp > 0.1:  # Strong positive momentum
                    action = 'buy'
                    reason = 'high_density_trend_continuation'
                    score = self._compute_signal_score(
                        density_ratio=density / p['density_threshold'],
                        momentum=momentum,
                        atr_ratio=atr_now / np.mean(atr[:i]),
                        vdp=vdp if has_volume else 0.0,
                        signal_type='trend_continuation'
                    )

                elif momentum < -0.2 and (not has_volume or vdp < -0.1):  # Strong negative momentum
                    action = 'sell'
                    reason = 'high_density_trend_continuation'
                    score = self._compute_signal_score(
                        density_ratio=density / p['density_threshold'],
                        momentum=abs(momentum),
                        atr_ratio=atr_now / np.mean(atr[:i]),
                        vdp=-vdp if has_volume else 0.0,
                        signal_type='trend_continuation'
                    )

            # Record signal if score is above threshold
            if score >= p['min_signal_score'] and action != 'hold':
                self._record_signal(ts, action, symbol, price,
                                   score=score, reason=reason)

                # Update position tracking
                if action == 'buy' and position <= 0:
                    position = 1
                    entry_price = price
                    stop_price = price - p['atr_stop_mult'] * atr_now
                elif action == 'sell' and position >= 0:
                    position = -1
                    entry_price = price
                    stop_price = price + p['atr_stop_mult'] * atr_now

        # Close open position at end of data
        if position != 0 and len(df) > 0:
            last_ts = df.index[-1]
            last_price = df['close'].iloc[-1]
            if position == 1:
                self._record_signal(last_ts, 'sell', symbol, last_price,
                                   score=0, reason='end_of_data')
            else:
                self._record_signal(last_ts, 'buy', symbol, last_price,
                                   score=0, reason='end_of_data')

        return self.signals

    @staticmethod
    def _compute_kde_density(df: pd.DataFrame, period: int, silverman_factor: float) -> tuple:
        """Compute kernel density estimation using Gaussian kernel and Silverman's rule."""
        density_values = np.full(len(df), np.nan)
        density_ratios = np.full(len(df), np.nan)

        for i in range(period, len(df)):
            # Get rolling returns data
            returns = df['returns'].iloc[i-period:i].dropna()

            if len(returns) < 5:  # Minimum data points for KDE
                continue

            # Apply Silverman's rule for bandwidth selection
            std = np.std(returns)
            q75, q25 = np.percentile(returns, [75, 25])
            iqr = q75 - q25

            # Handle cases where std or IQR is zero
            if std == 0:
                std = 1e-10
            if iqr == 0:
                iqr = 1e-10

            # Silverman's bandwidth: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
            bandwidth = silverman_factor * min(std, iqr / 1.34) * (len(returns) ** (-0.2))

            # Create KDE object
            try:
                kde = gaussian_kde(returns, bw_method=bandwidth)

                # Evaluate density at current return
                current_return = df['returns'].iloc[i]
                density = kde(current_return)[0]

                # Normalize density by peak density for this window
                x_grid = np.linspace(returns.min() * 2, returns.max() * 2, 1000)
                peak_density = kde(x_grid).max()

                if peak_density > 0:
                    density_ratio = density / peak_density
                else:
                    density_ratio = 0

                density_values[i] = density
                density_ratios[i] = density_ratio

            except (np.linalg.LinAlgError, ValueError):
                # Fallback to simple density estimation if KDE fails
                density_values[i] = 1e-10
                density_ratios[i] = 0

        return density_values, density_ratios

    @staticmethod
    def _compute_momentum(df: pd.DataFrame, period: int) -> np.ndarray:
        """Compute momentum as normalized change over period."""
        momentum = np.full(len(df), np.nan)

        for i in range(period, len(df)):
            price_change = df['close'].iloc[i] - df['close'].iloc[i-period]
            avg_price = (df['close'].iloc[i] + df['close'].iloc[i-period]) / 2
            momentum[i] = price_change / avg_price if avg_price > 0 else 0

        return momentum

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
        """Compute Average True Range."""
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)

        tr = np.zeros(len(df))
        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1]),
            )

        # Simple moving average of TR
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
        return atr

    @staticmethod
    def _compute_vdp(df: pd.DataFrame, period: int) -> np.ndarray:
        """Compute Volume Delta Pressure for direction confirmation."""
        if 'volume' not in df.columns:
            return np.zeros(len(df))

        vdp = np.zeros(len(df))

        for i in range(period, len(df)):
            # Volume delta: buy - sell pressure
            # Using approximation: close is near high → buy pressure, near low → sell pressure
            recent_high = df['high'].iloc[i-period:i].max()
            recent_low = df['low'].iloc[i-period:i].min()
            recent_close = df['close'].iloc[i]

            # Normalize position between high and low
            if recent_high > recent_low:
                position = (recent_close - recent_low) / (recent_high - recent_low)
            else:
                position = 0.5

            # VDP = volume * (2*position - 1) - buy pressure when > 0, sell when < 0
            avg_volume = df['volume'].iloc[i-period:i].mean()
            vdp[i] = avg_volume * (2 * position - 1) / 1e6  # Normalize by million

        return vdp

    def _compute_signal_score(self, density_ratio: float, momentum: float,
                            atr_ratio: float, vdp: float, signal_type: str) -> float:
        """Calculate comprehensive signal score."""

        # Base score components
        density_score = np.log(density_ratio + 1) * 0.3
        momentum_score = np.tanh(momentum) * 0.4
        atr_score = np.log(atr_ratio + 1) * 0.1

        # Combine scores
        total_score = density_score + momentum_score + atr_score

        # Add VDP confirmation if available
        if vdp != 0:
            vdp_score = np.tanh(abs(vdp)) * 0.2
            if (signal_type == 'trend_continuation' and vdp > 0) or \
               (signal_type == 'mean_reversion' and vdp < 0):
                total_score += vdp_score
            else:
                total_score -= vdp_score

        # Clip score between 0 and 1
        return np.clip(total_score, 0, 1)