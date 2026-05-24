"""
Z-Score Cumulative Impulse Strategy - Regime Detection with Momentum Clustering

Philosophy: "Momentum clusters like gravitation" - Z-score impulses reveal market regimes
by quantifying momentum clusters and their directional bias.

Core idea:
- Rolling z-scores of returns measure momentum strength and direction
- Cumulative sum creates impulse function showing regime persistence
- Multiple timeframes (10, 20, 40) for robust regime confirmation
- VDP timing for entries, ATR trailing for exits
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class ZScoreRegimeStrategy(BaseStrategy):
    """Z-Score Cumulative Impulse Strategy for regime detection and timing."""

    strategy_description = (
        "Z-Score Cumulative Impulse detects market regimes using momentum clustering. "
        "Multiple z-score windows (10, 20, 40) with cumulative impulse tracking. "
        "VDP timing for entries, ATR trailing for exits."
    )
    strategy_category = "momentum"

    strategy_params_schema = {
        "z_windows": {
            "type": "list",
            "default": [10, 20, 40],
            "description": "Z-score calculation windows"
        },
        "z_threshold": {
            "type": "float",
            "default": 0.5,
            "description": "Z-score threshold for regime change"
        },
        "vdp_weight": {
            "type": "float",
            "default": 0.3,
            "description": "VDP signal weight in final decision"
        },
        "atr_period": {
            "type": "int",
            "default": 14,
            "description": "ATR calculation period"
        },
        "atr_stop_mult": {
            "type": "float",
            "default": 2.0,
            "description": "ATR trailing stop multiplier"
        },
        "atr_offset": {
            "type": "float",
            "default": 1.0,
            "description": "ATR offset from entry price"
        }
    }

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'z_windows': [10, 20, 40],           # Z-score calculation windows
            'z_threshold': 0.5,                  # Threshold for regime change
            'vdp_weight': 0.3,                   # VDP signal weight
            'atr_period': 14,                    # ATR calculation period
            'atr_stop_mult': 2.0,                # ATR trailing stop multiplier
            'atr_offset': 1.0,                   # ATR offset from entry price
        }

    def validate_params(self):
        """Validate strategy parameters."""
        if not isinstance(self.params['z_windows'], list):
            raise ValueError("z_windows must be a list")
        if self.params['z_threshold'] <= 0:
            raise ValueError("z_threshold must be positive")
        if not (0 <= self.params['vdp_weight'] <= 1):
            raise ValueError("vdp_weight must be between 0 and 1")
        if self.params['atr_period'] < 1:
            raise ValueError("atr_period must be positive")
        if self.params['atr_stop_mult'] <= 0:
            raise ValueError("atr_stop_mult must be positive")

    def generate_signals(self) -> List[Dict]:
        """Generate trading signals using Z-Score Cumulative Impulse."""
        df = self.data.copy()
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'DEFAULT'
        p = self.params

        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Calculate Z-scores for multiple windows
        z_scores = {}
        cumulative_impulses = {}

        for window in p['z_windows']:
            # Rolling z-score of returns
            df[f'zscore_{window}'] = (df['returns'] -
                                     df['returns'].rolling(window=window, min_periods=3).mean()) / \
                                     df['returns'].rolling(window=window, min_periods=3).std()

            # Cumulative impulse function
            df[f'impulse_{window}'] = df[f'zscore_{window}'].cumsum()

            # Detect regime crossings
            df[f'bull_regime_{window}'] = df[f'impulse_{window}'] > p['z_threshold']
            df[f'bear_regime_{window}'] = df[f'impulse_{window}'] < -p['z_threshold']

            z_scores[window] = df[f'zscore_{window}'].values
            cumulative_impulses[window] = df[f'impulse_{window}'].values

        # Calculate ATR
        df['atr'] = self._compute_atr(df, p['atr_period'])

        # Calculate VDP timing signal (simplified version)
        df['vdp_signal'] = self._compute_vdp_signal(df)

        # Combine signals with majority voting across windows
        df['regime_signal'] = self._combine_regime_signals(df, p['z_windows'], p['z_threshold'])

        # Final signal with VDP timing
        df['final_signal'] = self._apply_vdp_timing(df, p['vdp_weight'])

        # Entry/exit tracking
        position = 0
        entry_price = 0
        stop_price = 0
        trailing_stop = np.full(len(df), np.nan)

        for i in range(len(df)):
            if i < max(p['z_windows']) + 10:  # Skip early data
                continue

            timestamp = df.index[i]
            price = df['close'].iloc[i]
            vdp = df['vdp_signal'].iloc[i]
            regime = df['final_signal'].iloc[i]
            atr = df['atr'].iloc[i]

            # Update trailing stop
            if position == 1 and not np.isnan(atr):  # Long position
                new_stop = price - (p['atr_stop_mult'] * atr * p['atr_offset'])
                if np.isnan(trailing_stop[i-1]) or new_stop > trailing_stop[i-1]:
                    trailing_stop[i] = new_stop
                else:
                    trailing_stop[i] = trailing_stop[i-1]
            elif position == -1 and not np.isnan(atr):  # Short position
                new_stop = price + (p['atr_stop_mult'] * atr * p['atr_offset'])
                if np.isnan(trailing_stop[i-1]) or new_stop < trailing_stop[i-1]:
                    trailing_stop[i] = new_stop
                else:
                    trailing_stop[i] = trailing_stop[i-1]
            else:
                trailing_stop[i] = np.nan

            # Check for exit conditions
            if position == 1:  # Long position
                if (regime == 'sell' or
                    (not np.isnan(stop_price) and price < stop_price) or
                    (not np.isnan(trailing_stop[i]) and price < trailing_stop[i])):
                    self._record_signal(
                        timestamp=timestamp,
                        action='sell',
                        symbol=symbol,
                        price=price,
                        regime=regime,
                        vdp=vdp,
                        z_signals=[df[f'zscore_{w}'].iloc[i] for w in p['z_windows']],
                        impulse_signals=[df[f'impulse_{w}'].iloc[i] for w in p['z_windows']]
                    )
                    position = 0
                    entry_price = 0
                    stop_price = 0
                    continue

            elif position == -1:  # Short position
                if (regime == 'buy' or
                    (not np.isnan(stop_price) and price > stop_price) or
                    (not np.isnan(trailing_stop[i]) and price > trailing_stop[i])):
                    self._record_signal(
                        timestamp=timestamp,
                        action='buy',
                        symbol=symbol,
                        price=price,
                        regime=regime,
                        vdp=vdp,
                        z_signals=[df[f'zscore_{w}'].iloc[i] for w in p['z_windows']],
                        impulse_signals=[df[f'impulse_{w}'].iloc[i] for w in p['z_windows']]
                    )
                    position = 0
                    entry_price = 0
                    stop_price = 0
                    continue

            # Entry conditions
            if position == 0:
                if regime == 'buy' and vdp > 0.1:  # Bullish regime with VDP timing
                    self._record_signal(
                        timestamp=timestamp,
                        action='buy',
                        symbol=symbol,
                        price=price,
                        regime=regime,
                        vdp=vdp,
                        z_signals=[df[f'zscore_{w}'].iloc[i] for w in p['z_windows']],
                        impulse_signals=[df[f'impulse_{w}'].iloc[i] for w in p['z_windows']]
                    )
                    position = 1
                    entry_price = price
                    stop_price = price - (p['atr_stop_mult'] * atr * p['atr_offset']) if not np.isnan(atr) else np.nan

                elif regime == 'sell' and vdp < -0.1:  # Bearish regime with VDP timing
                    self._record_signal(
                        timestamp=timestamp,
                        action='sell',
                        symbol=symbol,
                        price=price,
                        regime=regime,
                        vdp=vdp,
                        z_signals=[df[f'zscore_{w}'].iloc[i] for w in p['z_windows']],
                        impulse_signals=[df[f'impulse_{w}'].iloc[i] for w in p['z_windows']]
                    )
                    position = -1
                    entry_price = price
                    stop_price = price + (p['atr_stop_mult'] * atr * p['atr_offset']) if not np.isnan(atr) else np.nan

        # Hold signal for last position
        if position != 0 and len(self.signals) > 0:
            last_signal = self.signals[-1]
            if last_signal['action'] in ['buy', 'sell']:
                timestamp = df.index[-1]
                price = df['close'].iloc[-1]
                regime = df['final_signal'].iloc[-1]
                vdp = df['vdp_signal'].iloc[-1]

                self._record_signal(
                    timestamp=timestamp,
                    action='hold',
                    symbol=symbol,
                    price=price,
                    regime=regime,
                    vdp=vdp,
                    z_signals=[df[f'zscore_{w}'].iloc[-1] for w in p['z_windows']],
                    impulse_signals=[df[f'impulse_{w}'].iloc[-1] for w in p['z_windows']]
                )

        return self.signals

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value

        # Smoothed ATR
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return pd.Series(atr, index=df.index)

    def _compute_vdp_signal(self, df: pd.DataFrame) -> pd.Series:
        """Simplified Volume Delta Profile signal."""
        if 'volume' in df.columns and df['volume'].notna().any():
            vol = df['volume'].values
        else:
            # Use position as volume proxy
            vol = np.ones(len(df))

        # Volume delta pressure calculation
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Normalize position (0 to 1)
        hl_range = np.where(high - low > 1e-10, high - low, 1e-10)
        position = (close - low) / hl_range

        # VDP = volume * (2 * position - 1)
        vdp = vol * (2 * position - 1)

        # Normalize VDP
        vdp_std = np.std(vdp)
        if vdp_std > 0:
            vdp = (vdp - np.mean(vdp)) / vdp_std

        return pd.Series(vdp, index=df.index)

    def _combine_regime_signals(self, df: pd.DataFrame, z_windows: List[int], threshold: float) -> pd.Series:
        """Combine signals from multiple Z-score windows using majority voting."""
        regime_signals = []

        for window in z_windows:
            bull_regime = df[f'bull_regime_{window}']
            bear_regime = df[f'bear_regime_{window}']

            # Convert to signals
            signals = np.where(bull_regime, 'buy', np.where(bear_regime, 'sell', 'hold'))
            regime_signals.append(signals)

        # Majority vote
        regime_signals = np.array(regime_signals)
        counts = np.zeros((len(regime_signals[0]), 3), dtype=int)

        for i, signal in enumerate(regime_signals.T):
            for j, s in enumerate(['buy', 'sell', 'hold']):
                counts[i, j] = np.sum(signal == s)

        # Get majority signal
        majority_signals = []
        for count in counts:
            max_count = np.max(count)
            if max_count > len(z_windows) // 2:  # Majority
                idx = np.argmax(count)
                majority_signals.append(['buy', 'sell', 'hold'][idx])
            else:
                majority_signals.append('hold')

        return pd.Series(majority_signals, index=df.index)

    def _apply_vdp_timing(self, df: pd.DataFrame, vdp_weight: float) -> pd.Series:
        """Apply VDP timing to regime signals."""
        regime_signals = df['regime_signal']
        vdp_signals = df['vdp_signal']

        final_signals = []

        for i, (regime, vdp) in enumerate(zip(regime_signals, vdp_signals)):
            if np.isnan(vdp):
                final_signals.append(regime)
                continue

            # Weighted combination
            if regime == 'buy':
                # Boost buy signal if VDP is positive
                if vdp > 0:
                    final_signals.append('buy')
                else:
                    # Only buy if VDP slightly positive
                    if vdp > -0.3:
                        final_signals.append('buy')
                    else:
                        final_signals.append('hold')

            elif regime == 'sell':
                # Boost sell signal if VDP is negative
                if vdp < 0:
                    final_signals.append('sell')
                else:
                    # Only sell if VDP slightly negative
                    if vdp < 0.3:
                        final_signals.append('sell')
                    else:
                        final_signals.append('hold')

            else:
                final_signals.append('hold')

        return pd.Series(final_signals, index=df.index)