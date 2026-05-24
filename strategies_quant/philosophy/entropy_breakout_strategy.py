"""
Entropy Breakout Strategy
========================

Core Philosophy: Market transitioning from chaos to order precedes breakouts.

Key Concepts:
  - Shannon entropy measures market randomness/chaos
  - Sharp entropy drop = market becoming ordered = setup for breakout
  - Direction from volume delta pressure + structural tension
  - Entropy-based position sizing: lower entropy = bigger position

Entry Logic:
  1. Compute 50-day Shannon entropy (10 bins) on returns
  2. Wait for entropy drop: ΔH < -0.1 in 5 days
  3. Enter on FIRST bar after entropy < 0.7 * H_max
  4. Direction: volume delta pressure + structural tension

Exit Logic:
  1. Entropy rises back above 0.8 * H_max
  2. ATR trailing stop
  3. Price targets from expanding ATR bands

Position Sizing:
  - Size = 1 / (1 + entropy)
  - Lower entropy = bigger position (more confident)

Mathematical Foundation:
  Shannon Entropy:
    H(X) = -sum(p_i * log2(p_i))
    Maximum entropy for 10 bins: log2(10) ~ 3.322 bits

  Volume Delta Pressure:
    delta = V * (2*C - H - L) / (H - L)
    cum_delta = EMA(delta, period)
    Positive cum_delta = buying pressure, negative = selling pressure

  Structural Tension:
    Measures displacement from fair value using 7-point regression
    Positive tension = bearish, negative tension = bullish
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class EntropyBreakoutStrategy(BaseStrategy):
    """Entropy Breakout Strategy - trade breakouts from chaos to order"""

    strategy_description = (
        "EntropyBreakout: Trade breakouts when entropy drops sharply "
        "(chaos to transition to order) with VDP + structural direction"
    )
    strategy_category = "momentum"
    strategy_params_schema = {
        "entropy_bins": {"type": "int", "default": 10, "label": "Entropy bin count"},
        "entropy_window": {"type": "int", "default": 50, "label": "Entropy rolling window"},
        "entropy_drop_threshold": {"type": "float", "default": -0.1, "label": "Entropy drop threshold"},
        "entropy_drop_period": {"type": "int", "default": 5, "label": "Entropy drop lookback period"},
        "entropy_entry_ratio": {"type": "float", "default": 0.7, "label": "Entropy entry threshold (ratio of max)"},
        "entropy_exit_ratio": {"type": "float", "default": 0.8, "label": "Entropy exit threshold (ratio of max)"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "atr_multiplier": {"type": "float", "default": 2.0, "label": "ATR multiplier for bands"},
        "vdp_period": {"type": "int", "default": 10, "label": "VDP EMA period"},
        "tension_period": {"type": "int", "default": 7, "label": "Structural tension period"},
        "min_data_points": {"type": "int", "default": 100, "label": "Minimum data points before signals"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        self.entropy_max = 0.0
        self._calculate_indicators()

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "entropy_bins": 10,
            "entropy_window": 50,
            "entropy_drop_threshold": -0.1,
            "entropy_drop_period": 5,
            "entropy_entry_ratio": 0.7,
            "entropy_exit_ratio": 0.8,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "vdp_period": 10,
            "tension_period": 7,
            "min_data_points": 100,
        }

    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        # Returns for entropy calculation
        self.data['returns'] = self.data['close'].pct_change()

        # Calculate Shannon entropy
        self.data['entropy'] = self._calculate_shannon_entropy(
            self.data['returns'],
            self.params['entropy_window'],
            self.params['entropy_bins']
        )

        # Track entropy maximum for threshold calculations
        if len(self.data) >= self.params['entropy_window']:
            self.entropy_max = self.data['entropy'].rolling(self.params['entropy_window']).max().ffill().bfill()

        # Volume Delta Pressure
        self.data['vdp_delta'] = self._calculate_volume_delta_pressure()
        self.data['vdp_ema'] = self.data['vdp_delta'].ewm(span=self.params['vdp_period']).mean()

        # Structural Tension
        self.data['structural_tension'] = self._calculate_structural_tension()

        # ATR and bands
        self.data['atr'] = self._calculate_atr()
        self.data['atr_upper'] = self.data['atr'] * self.params['atr_multiplier']
        self.data['atr_lower'] = -self.data['atr'] * self.params['atr_multiplier']

        # Expanding ATR bands for breakout targets
        self.data['expanding_atr_upper'] = self.data['atr_upper'].expanding().max()
        self.data['expanding_atr_lower'] = self.data['atr_lower'].expanding().min()

    def _calculate_shannon_entropy(self, series: pd.Series, window: int, bins: int) -> pd.Series:
        """Calculate rolling Shannon entropy"""
        def entropy_calc(x):
            if len(x) < 10 or np.std(x) == 0:  # Not enough data or no variation
                return 0.0

            # Create bins
            hist, _ = np.histogram(x, bins=bins, density=True)

            # Remove zero probabilities
            hist = hist[hist > 0]

            # Calculate entropy
            if len(hist) > 0:
                return -np.sum(hist * np.log2(hist))
            return 0.0

        return series.rolling(window=window, min_periods=max(10, window//2)).apply(entropy_calc, raw=False)

    def _calculate_volume_delta_pressure(self) -> pd.Series:
        """Calculate Volume Delta Pressure"""
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        position = (2 * typical_price - self.data['high'] - self.data['low']) / (self.data['high'] - self.data['low'])
        position = position.fillna(0)
        return self.data['volume'] * position

    def _calculate_structural_tension(self) -> pd.Series:
        """Calculate structural tension using regression displacement"""
        def tension_calc(prices):
            if len(prices) < 7:
                return 0.0

            # Create time index
            x = np.arange(len(prices))

            # Linear regression
            slope, intercept = np.polyfit(x, prices, 1)
            fitted_line = slope * x + intercept

            # Calculate displacement from fair value
            displacement = prices - fitted_line

            # Return the last displacement value
            return displacement[-1] if len(displacement) > 0 else 0.0

        return self.data['close'].rolling(
            window=self.params['tension_period'],
            min_periods=7
        ).apply(tension_calc, raw=False)

    def _calculate_atr(self) -> pd.Series:
        """Calculate Average True Range"""
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(window=self.params['atr_period']).mean()

    def _check_entropy_drop(self, idx: int) -> bool:
        """Check if entropy has dropped sharply"""
        if idx < self.params['entropy_drop_period']:
            return False

        current_entropy = self.data['entropy'].iloc[idx]
        past_entropy = self.data['entropy'].iloc[idx - self.params['entropy_drop_period']]

        # Check if past_entropy is valid and current_entropy is valid
        if pd.isna(past_entropy) or pd.isna(current_entropy):
            return False

        # Check if entropy has dropped by the threshold
        entropy_change = current_entropy - past_entropy
        return entropy_change < self.params['entropy_drop_threshold']

    def _check_entry_signal(self, idx: int) -> bool:
        """Check if we should enter a position"""
        if idx < self.params['min_data_points']:
            return False

        current_entropy = self.data['entropy'].iloc[idx]
        current_entropy_max = self.entropy_max.iloc[idx]

        # Check if we have valid entropy values
        if pd.isna(current_entropy) or pd.isna(current_entropy_max) or current_entropy_max <= 0:
            return False

        # Check entropy threshold
        entropy_threshold = current_entropy_max * self.params['entropy_entry_ratio']

        # Check if entropy is below threshold and we're not already in a position
        return current_entropy < entropy_threshold

    def _check_exit_signal(self, idx: int) -> bool:
        """Check if we should exit a position"""
        if idx < self.params['min_data_points']:
            return False

        current_entropy = self.data['entropy'].iloc[idx]
        current_entropy_max = self.entropy_max.iloc[idx]

        # Check if we have valid entropy values
        if pd.isna(current_entropy) or pd.isna(current_entropy_max) or current_entropy_max <= 0:
            return False

        # Check if entropy has risen back above exit threshold
        entropy_threshold = current_entropy_max * self.params['entropy_exit_ratio']
        return current_entropy > entropy_threshold

    def _get_direction(self, idx: int) -> str:
        """Determine trade direction based on VDP and structural tension"""
        if idx < self.params['min_data_points']:
            return 'hold'

        vdp = self.data['vdp_ema'].iloc[idx]
        tension = self.data['structural_tension'].iloc[idx]

        # VDP direction (positive = bullish, negative = bearish)
        vdp_direction = 1 if vdp > 0 else -1

        # Tension direction (positive = bearish, negative = bullish)
        tension_direction = -1 if tension > 0 else 1

        # Combined signal
        combined_signal = vdp_direction + tension_direction

        if combined_signal > 0:
            return 'buy'
        elif combined_signal < 0:
            return 'sell'
        else:
            # Neutral signals default to VDP
            return 'buy' if vdp > 0 else 'sell'

    def _get_position_size(self, idx: int) -> float:
        """Calculate position size based on entropy"""
        if idx < self.params['min_data_points']:
            return 0.1

        current_entropy = self.data['entropy'].iloc[idx]

        if pd.isna(current_entropy):
            return 0.1

        # Position size = 1 / (1 + entropy)
        # Lower entropy = bigger position
        position_size = 1.0 / (1.0 + current_entropy)

        # Clamp between reasonable bounds
        return np.clip(position_size, 0.05, 0.5)

    def generate_signals(self) -> List[Dict]:
        """Generate trading signals"""
        self.signals = []

        # Track position state
        in_position = False
        position_direction = None
        entry_price = None
        entry_idx = None

        for idx in range(len(self.data)):
            if idx < self.params['min_data_points']:
                continue

            timestamp = self.data.index[idx]
            price = self.data['close'].iloc[idx]

            # Check for exit signals
            if in_position:
                # Check entropy exit
                if self._check_exit_signal(idx):
                    self._record_signal(
                        timestamp=timestamp,
                        action='sell' if position_direction == 'buy' else 'buy',
                        symbol=self.data['symbol'].iloc[idx],
                        price=price,
                        reason=f'Entropy exit: {self.data["entropy"].iloc[idx]:.3f}',
                        position_size=self._get_position_size(idx),
                        entropy=self.data['entropy'].iloc[idx],
                        vdp=self.data['vdp_ema'].iloc[idx],
                        tension=self.data['structural_tension'].iloc[idx]
                    )
                    in_position = False
                    position_direction = None
                    entry_price = None
                    entry_idx = None

                # Check ATR trailing stop
                elif position_direction == 'buy':
                    # Long position: stop at lower ATR band
                    stop_price = price - self.data['expanding_atr_lower'].iloc[idx]
                    if price <= stop_price:
                        self._record_signal(
                            timestamp=timestamp,
                            action='sell',
                            symbol=self.data['symbol'].iloc[idx],
                            price=price,
                            reason=f'ATR trailing stop hit',
                            position_size=self._get_position_size(idx),
                            entropy=self.data['entropy'].iloc[idx],
                            vdp=self.data['vdp_ema'].iloc[idx],
                            tension=self.data['structural_tension'].iloc[idx]
                        )
                        in_position = False
                        position_direction = None
                        entry_price = None
                        entry_idx = None

                elif position_direction == 'sell':
                    # Short position: stop at upper ATR band
                    stop_price = price + self.data['expanding_atr_lower'].iloc[idx]
                    if price >= stop_price:
                        self._record_signal(
                            timestamp=timestamp,
                            action='buy',
                            symbol=self.data['symbol'].iloc[idx],
                            price=price,
                            reason=f'ATR trailing stop hit',
                            position_size=self._get_position_size(idx),
                            entropy=self.data['entropy'].iloc[idx],
                            vdp=self.data['vdp_ema'].iloc[idx],
                            tension=self.data['structural_tension'].iloc[idx]
                        )
                        in_position = False
                        position_direction = None
                        entry_price = None
                        entry_idx = None

            # Check for entry signals (only when not in position)
            if not in_position:
                # Check for entropy drop and entry threshold
                if self._check_entropy_drop(idx) and self._check_entry_signal(idx):
                    direction = self._get_direction(idx)

                    # Record entry signal
                    self._record_signal(
                        timestamp=timestamp,
                        action=direction,
                        symbol=self.data['symbol'].iloc[idx],
                        price=price,
                        reason=f'Entropy breakout: ΔH < {self.params["entropy_drop_threshold"]}, H = {self.data["entropy"].iloc[idx]:.3f}',
                        position_size=self._get_position_size(idx),
                        entropy=self.data['entropy'].iloc[idx],
                        vdp=self.data['vdp_ema'].iloc[idx],
                        tension=self.data['structural_tension'].iloc[idx],
                        atr=self.data['atr'].iloc[idx],
                        breakout_threshold=self.entropy_max.iloc[idx] * self.params['entropy_entry_ratio']
                    )

                    in_position = True
                    position_direction = direction
                    entry_price = price
                    entry_idx = idx

            # Always record hold signal for state tracking
            elif not in_position:
                self._record_signal(
                    timestamp=timestamp,
                    action='hold',
                    symbol=self.data['symbol'].iloc[idx],
                    price=price,
                    reason='Waiting for entropy setup',
                    entropy=self.data['entropy'].iloc[idx]
                )

        return self.signals

    def validate_params(self):
        """Validate strategy parameters"""
        if self.params['entropy_window'] <= 0:
            raise ValueError("Entropy window must be positive")
        if self.params['entropy_bins'] <= 0:
            raise ValueError("Entropy bins must be positive")
        if self.params['entropy_drop_threshold'] >= 0:
            raise ValueError("Entropy drop threshold must be negative")
        if self.params['entropy_drop_period'] <= 0:
            raise ValueError("Entropy drop period must be positive")
        if self.params['entropy_entry_ratio'] <= 0 or self.params['entropy_entry_ratio'] >= 1:
            raise ValueError("Entropy entry ratio must be between 0 and 1")
        if self.params['entropy_exit_ratio'] <= 0 or self.params['entropy_exit_ratio'] >= 1:
            raise ValueError("Entropy exit ratio must be between 0 and 1")
        if self.params['atr_period'] <= 0:
            raise ValueError("ATR period must be positive")
        if self.params['atr_multiplier'] <= 0:
            raise ValueError("ATR multiplier must be positive")
        if self.params['vdp_period'] <= 0:
            raise ValueError("VDP period must be positive")
        if self.params['tension_period'] <= 0:
            raise ValueError("Tension period must be positive")
        if self.params['min_data_points'] <= 0:
            raise ValueError("Minimum data points must be positive")

        # Ensure entry ratio < exit ratio
        if self.params['entropy_entry_ratio'] >= self.params['entropy_exit_ratio']:
            raise ValueError("Entry ratio must be less than exit ratio")