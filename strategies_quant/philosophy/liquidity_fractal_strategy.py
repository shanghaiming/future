"""
Smart Liquidity & Trend Engine V9.0
Weighted Multi-Timeframe Fractal Clustering with Two-Stage Signal Execution

Core Innovation:
- Weighted Multi-Timeframe Fractal Detection (HTF fractals 3x weight)
- ATR-based zone merging (not fixed-pip)
- Two-stage signal: TAP (preparation) -> Diamond (execution with momentum)
- Ghost zone invalidation when price closes beyond zone
- ATR trailing stop for exits

Author: Adapted from TradingView strategy pJqDBkVx (batch_6)
Implementation: Claude Code
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from core.base_strategy import BaseStrategy


class LiquidityFractalStrategy(BaseStrategy):
    """
    Smart Liquidity & Trend Engine V9.0

    Strategy Features:
    - Multi-timeframe fractal detection with weighted clustering
    - ATR-based dynamic zone detection and merging
    - Two-stage signal system (TAP preparation -> Diamond execution)
    - Ghost zone invalidation for adaptive liquidity mapping
    - ATR trailing stop for risk management
    """

    strategy_description = "Smart Liquidity & Trend Engine V9.0 - Weighted MTF Fractal Clustering"
    strategy_category = "general"

    def get_default_params(self) -> Dict:
        """Default parameters for the strategy"""
        return {
            # Fractal detection parameters
            'htf_weight': 3.0,  # Higher timeframe fractal weight multiplier
            'fractal_period': 5,  # Fractal lookback period
            'min_fractal_points': 3,  # Minimum points to form a fractal cluster

            # ATR parameters for zone detection
            'atr_period': 14,  # ATR calculation period
            'zone_atr_multiplier': 2.0,  # Zone size in ATR units
            'merge_threshold': 1.5,  # Distance threshold to merge adjacent zones (ATR)

            # Signal parameters
            'tap_confirmation': 2,  # Number of bars for TAP confirmation
            'momentum_threshold': 0.02,  # Minimum momentum for Diamond signal (2%)
            'ghost_invalidation': True,  # Enable ghost zone invalidation
            'ghost_threshold': 1.0,  # Price close beyond zone by this ATR multiple to invalidate

            # Risk management
            'atr_stop_multiplier': 2.0,  # ATR trailing stop multiplier
            'max_hold_bars': 50,  # Maximum bars to hold a position

            # Filter parameters
            'min_volume_multiplier': 1.0,  # Minimum volume relative to average
            'volatility_filter': True,  # Enable volatility filter
            'volatility_percentile': 75  # Volatility percentile threshold
        }

    def validate_params(self):
        """Validate strategy parameters"""
        if self.params['htf_weight'] <= 0:
            raise ValueError("HTF weight must be positive")
        if self.params['fractal_period'] < 3:
            raise ValueError("Fractal period must be at least 3")
        if self.params['atr_period'] < 1:
            raise ValueError("ATR period must be positive")
        if self.params['zone_atr_multiplier'] <= 0:
            raise ValueError("Zone ATR multiplier must be positive")
        if self.params['merge_threshold'] <= 0:
            raise ValueError("Merge threshold must be positive")
        if self.params['momentum_threshold'] <= 0:
            raise ValueError("Momentum threshold must be positive")

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = high.ffill()
        low = low.ffill()
        close = close.ffill()

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _detect_fractals(self, data: pd.DataFrame, period: int, weight: float) -> pd.DataFrame:
        """
        Detect fractal patterns with weighted multi-timeframe approach

        Returns DataFrame with fractal detection signals
        """
        # Calculate fractal patterns on different timeframes
        fractals = pd.DataFrame(index=data.index)

        # Primary timeframe fractals (weight = 1)
        fractals['primary_fractal'] = self._calculate_fractal_points(data, period, 1.0)

        # Higher timeframe fractals (weight = htf_weight)
        # Simulate HTF by using larger period
        htf_period = period * 2  # Simple HTF simulation
        fractals['htf_fractal'] = self._calculate_fractal_points(data, htf_period, weight)

        # Combined fractal score
        fractals['fractal_score'] = fractals['primary_fractal'] + fractals['htf_fractal']

        return fractals

    def _calculate_fractal_points(self, data: pd.DataFrame, period: int, weight: float) -> pd.Series:
        """Calculate fractal points for a given period"""
        high = data['high']
        low = data['low']

        fractals = pd.Series(0.0, index=data.index)

        # Detect fractal highs
        for i in range(period, len(data) - period):
            # Check if current high is highest in the window
            window_high = high.iloc[i-period:i+period+1]
            if high.iloc[i] == window_high.max():
                fractals.iloc[i] = weight

        # Detect fractal lows
        for i in range(period, len(data) - period):
            # Check if current low is lowest in the window
            window_low = low.iloc[i-period:i+period+1]
            if low.iloc[i] == window_low.min():
                fractals.iloc[i] = -weight

        return fractals

    def _detect_liquidity_zones(self, data: pd.DataFrame, fractals: pd.DataFrame) -> List[Dict]:
        """
        Detect liquidity zones based on fractal clusters and ATR

        Returns list of zone dictionaries
        """
        zones = []
        atr = self._calculate_atr(data['high'], data['low'], data['close'], self.params['atr_period'])

        # Find fractal clusters
        cluster_threshold = self.params['zone_atr_multiplier']
        current_zone = None

        for i in range(1, len(data)):
            # Check for fractal cluster
            if abs(fractals['fractal_score'].iloc[i]) >= self.params['min_fractal_points']:
                zone_high = data['high'].iloc[i]
                zone_low = data['low'].iloc[i]
                zone_center = (zone_high + zone_low) / 2
                zone_size = atr.iloc[i] * cluster_threshold

                if current_zone is None:
                    # Start new zone
                    current_zone = {
                        'start_idx': i,
                        'center': zone_center,
                        'size': zone_size,
                        'high': zone_center + zone_size/2,
                        'low': zone_center - zone_size/2,
                        'strength': abs(fractals['fractal_score'].iloc[i]),
                        'ghost': False
                    }
                else:
                    # Check if close enough to merge
                    distance = abs(zone_center - current_zone['center'])
                    if distance <= current_zone['size'] * self.params['merge_threshold']:
                        # Merge zones
                        current_zone['center'] = (current_zone['center'] + zone_center) / 2
                        current_zone['size'] = max(current_zone['size'], zone_size)
                        current_zone['high'] = current_zone['center'] + current_zone['size']/2
                        current_zone['low'] = current_zone['center'] - current_zone['size']/2
                        current_zone['strength'] += abs(fractals['fractal_score'].iloc[i])
                        current_zone['end_idx'] = i
                    else:
                        # Save current zone and start new one
                        current_zone['end_idx'] = i-1
                        zones.append(current_zone.copy())
                        current_zone = {
                            'start_idx': i,
                            'center': zone_center,
                            'size': zone_size,
                            'high': zone_center + zone_size/2,
                            'low': zone_center - zone_size/2,
                            'strength': abs(fractals['fractal_score'].iloc[i]),
                            'ghost': False
                        }

        # Add final zone
        if current_zone is not None:
            current_zone['end_idx'] = len(data) - 1
            zones.append(current_zone.copy())

        # Ghost zone invalidation
        if self.params['ghost_invalidation']:
            zones = self._apply_ghost_invalidation(data, zones, atr)

        return zones

    def _apply_ghost_invalidation(self, data: pd.DataFrame, zones: List[Dict], atr: pd.Series) -> List[Dict]:
        """Apply ghost zone invalidation when price closes beyond zone"""
        threshold = atr * self.params['ghost_threshold']

        for zone in zones:
            zone_start = max(0, zone['start_idx'])
            zone_end = min(len(data), zone['end_idx'])

            # Check if price closed beyond zone
            if zone_end < len(data):
                # Check if price closed above zone high
                high_breaches = data['close'].iloc[zone_end] > zone['high']
                # Check if price closed below zone low
                low_breaches = data['close'].iloc[zone_end] < zone['low']

                if high_breaches or low_breaches:
                    zone['ghost'] = True

        # Remove ghost zones that are too old
        non_ghost_zones = []
        for zone in zones:
            if not zone['ghost'] or (zone['end_idx'] - zone['start_idx']) < 10:
                non_ghost_zones.append(zone)

        return non_ghost_zones

    def _calculate_momentum(self, data: pd.DataFrame, period: int = 5) -> pd.Series:
        """Calculate price momentum"""
        momentum = data['close'].pct_change(period)
        return momentum

    def generate_signals(self) -> List[Dict]:
        """Generate trading signals"""
        signals = []

        # Calculate indicators
        fractals = self._detect_fractals(self.data, self.params['fractal_period'], self.params['htf_weight'])
        zones = self._detect_liquidity_zones(self.data, fractals)
        atr = self._calculate_atr(self.data['high'], self.data['low'], self.data['close'], self.params['atr_period'])
        momentum = self._calculate_momentum(self.data, 5)

        # Convert zones to a series for easy lookup
        zone_map = pd.Series(None, index=self.data.index)
        for zone in zones:
            start_idx = zone['start_idx']
            end_idx = zone['end_idx']
            for i in range(start_idx, end_idx + 1):
                if i < len(self.data):
                    zone_map.iloc[i] = zone

        # Generate signals
        position = None  # 'long' or 'short'
        position_entry_price = 0
        position_entry_idx = 0

        for i in range(5, len(self.data)):  # Start from 5 to ensure we have enough history
            current_zone = zone_map.iloc[i]
            current_momentum = momentum.iloc[i]

            # Risk management checks
            volatility_filter_pass = True
            if self.params['volatility_filter']:
                volatility_percentile = self._calculate_volatility_percentile(self.data, i)
                volatility_filter_pass = volatility_percentile >= self.params['volatility_percentile']

            # Exit conditions
            if position is not None:
                # ATR trailing stop
                if position == 'long':
                    stop_price = position_entry_price - atr.iloc[i] * self.params['atr_stop_multiplier']
                    if self.data['low'].iloc[i] <= stop_price:
                        signals.append({
                            'timestamp': self.data.index[i],
                            'action': 'sell',
                            'symbol': self.data['symbol'].iloc[i],
                            'price': self.data['close'].iloc[i],
                            'reason': f'ATR trailing stop hit at {stop_price:.2f}'
                        })
                        position = None
                        continue
                else:  # short
                    stop_price = position_entry_price + atr.iloc[i] * self.params['atr_stop_multiplier']
                    if self.data['high'].iloc[i] >= stop_price:
                        signals.append({
                            'timestamp': self.data.index[i],
                            'action': 'buy',
                            'symbol': self.data['symbol'].iloc[i],
                            'price': self.data['close'].iloc[i],
                            'reason': f'ATR trailing stop hit at {stop_price:.2f}'
                        })
                        position = None
                        continue

                # Maximum hold time
                if i - position_entry_idx >= self.params['max_hold_bars']:
                    if position == 'long':
                        signals.append({
                            'timestamp': self.data.index[i],
                            'action': 'sell',
                            'symbol': self.data['symbol'].iloc[i],
                            'price': self.data['close'].iloc[i],
                            'reason': 'Maximum hold time reached'
                        })
                    else:
                        signals.append({
                            'timestamp': self.data.index[i],
                            'action': 'buy',
                            'symbol': self.data['symbol'].iloc[i],
                            'price': self.data['close'].iloc[i],
                            'reason': 'Maximum hold time reached'
                        })
                    position = None
                    continue

                # Exit if price moves significantly against position
                if position == 'long':
                    if self.data['close'].iloc[i] < position_entry_price * 0.95:
                        signals.append({
                            'timestamp': self.data.index[i],
                            'action': 'sell',
                            'symbol': self.data['symbol'].iloc[i],
                            'price': self.data['close'].iloc[i],
                            'reason': '5% loss threshold'
                        })
                        position = None
                        continue
                else:  # short
                    if self.data['close'].iloc[i] > position_entry_price * 1.05:
                        signals.append({
                            'timestamp': self.data.index[i],
                            'action': 'buy',
                            'symbol': self.data['symbol'].iloc[i],
                            'price': self.data['close'].iloc[i],
                            'reason': '5% loss threshold'
                        })
                        position = None
                        continue

            # Entry conditions (only if no position)
            if position is None and current_zone is not None and not current_zone.get('ghost', False):
                if volatility_filter_pass:
                    # TAP (preparation) phase - check for consolidation near zone
                    consolidation_count = 0
                    for j in range(max(0, i - self.params['tap_confirmation']), i):
                        range_size = (self.data['high'].iloc[j] - self.data['low'].iloc[j]) / self.data['close'].iloc[j]
                        if range_size < 0.02:  # 2% range indicates consolidation
                            consolidation_count += 1

                    # Diamond (execution) phase - momentum confirmation
                    if consolidation_count >= self.params['tap_confirmation'] - 1:
                        # Buy signal: approaching support zone with momentum
                        if (current_momentum > self.params['momentum_threshold'] and
                            self.data['close'].iloc[i] <= current_zone['high'] * 1.01):
                            signals.append({
                                'timestamp': self.data.index[i],
                                'action': 'buy',
                                'symbol': self.data['symbol'].iloc[i],
                                'price': self.data['close'].iloc[i],
                                'reason': f'Diamond entry at {current_zone["center"]:.2f}'
                            })
                            position = 'long'
                            position_entry_price = self.data['close'].iloc[i]
                            position_entry_idx = i

                        # Sell signal: approaching resistance zone with negative momentum
                        elif (current_momentum < -self.params['momentum_threshold'] and
                              self.data['close'].iloc[i] >= current_zone['low'] * 0.99):
                            signals.append({
                                'timestamp': self.data.index[i],
                                'action': 'sell',
                                'symbol': self.data['symbol'].iloc[i],
                                'price': self.data['close'].iloc[i],
                                'reason': f'Diamond entry at {current_zone["center"]:.2f}'
                            })
                            position = 'short'
                            position_entry_price = self.data['close'].iloc[i]
                            position_entry_idx = i

        # Convert signals to format expected by base class
        formatted_signals = []
        for signal in signals:
            formatted_signals.append({
                'timestamp': signal['timestamp'],
                'action': signal['action'],
                'symbol': signal['symbol'],
                'price': float(signal['price'])
            })

        self.signals = formatted_signals
        return formatted_signals

    def _calculate_volatility_percentile(self, data: pd.DataFrame, current_idx: int) -> float:
        """Calculate volatility percentile for filtering"""
        if current_idx < 50:  # Need at least 50 bars for percentile calculation
            return 50

        lookback = min(current_idx, 100)
        recent_range = data['high'].iloc[current_idx-lookback:current_idx] - data['low'].iloc[current_idx-lookback:current_idx]
        recent_volatility = recent_range.mean()

        historical_volatility = []
        for i in range(50, current_idx):
            hist_range = data['high'].iloc[i-50:i] - data['low'].iloc[i-50:i]
            historical_volatility.append(hist_range.mean())

        if historical_volatility:
            percentile = (sum(1 for v in historical_volatility if v <= recent_volatility) / len(historical_volatility)) * 100
            return percentile

        return 50  # Default to median if calculation fails

    def screen(self) -> Dict:
        """Screen method for real-time selection"""
        if len(self.data) < 20:
            return {'action': 'hold', 'reason': 'Data insufficient', 'price': float(self.data['close'].iloc[-1])}

        try:
            # Get signals for last few bars
            signals = self.generate_signals()

            if signals:
                last_signal = signals[-1]
                if last_signal['action'] in ['buy', 'sell']:
                    # Check if the signal is recent (within last 5 bars)
                    last_signal_idx = self.data.index.get_loc(last_signal['timestamp'])
                    if len(self.data) - last_signal_idx <= 5:
                        return {
                            'action': last_signal['action'],
                            'reason': f'Liquidity fractal signal detected',
                            'price': last_signal['price']
                        }
        except Exception:
            pass

        return {'action': 'hold', 'reason': 'No signal', 'price': float(self.data['close'].iloc[-1])}