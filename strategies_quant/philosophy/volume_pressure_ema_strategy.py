"""
VolumePressureEMAStrategy - Advanced Volume Pressure Analysis

Philosophy: "Institutional Footprint Detection" - Multi-timeframe EMA of
volume-weighted price displacement with volume acceleration as leading indicator.

Core idea: VDP (Volume Delta Pressure) = V * (2C-H-L)/(H-L) measures the
quality of price movement. Volume acceleration (2nd derivative) detects
institutional activity before it becomes obvious in price.

Signal Logic:
1. VDP (Volume Delta Pressure) per bar: delta = volume * (2*close - high - low) / (high - low)
2. Multi-timeframe EMA smoothing: fast(5), medium(15), slow(30)
3. Volume acceleration = d(VDP_fast)/dt
4. Buy: All 3 EMAs aligned positive AND acceleration > threshold
5. Sell: All 3 EMAs aligned negative AND acceleration < -threshold
6. Volume profile determines position size (high volume = high conviction)
7. ATR trailing stop for exits

Technical indicators: VDP, 3-EMA, Volume Acceleration, ATR, Volume Profile
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from core.base_strategy import BaseStrategy


class VolumePressureEMAStrategy(BaseStrategy):
    """Advanced Volume Pressure Analysis with Multi-timeframe EMA and Acceleration."""

    strategy_description = (
        "机构足迹检测 - 多时间框架VDP EMA + 成交量加速度信号。\n"
        "VDP = 成交量 × (2×收盘价-最高价-最低价)/(最高价-最低价)\n"
        "快速EMA(5)、中速EMA(15)、慢速EMA(30)同向+成交量加速度突破阈值 = 强信号"
    )
    strategy_category = "volume"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            # Multi-timeframe EMA periods
            'fast_ema': 5,           # Fast EMA period
            'medium_ema': 15,        # Medium EMA period
            'slow_ema': 30,          # Slow EMA period

            # Volume acceleration parameters
            'accel_window': 3,       # Window for acceleration calculation
            'accel_threshold': 0.1,  # Acceleration threshold for signal

            # Volume profile parameters
            'vol_profile_window': 20,    # Window for volume profile calculation
            'vol_profile_mult': 1.5,     # Volume multiplier for high conviction

            # Exit parameters
            'atr_period': 14,        # ATR calculation period
            'atr_stop_mult': 2.0,    # ATR trailing stop multiplier
            'min_hold': 5,          # Minimum hold bars

            # Signal alignment thresholds
            'ema_alignment_threshold': 0.05,  # Minimum alignment between EMAs
            'volume_min': 1000,     # Minimum volume requirement
        }

    def validate_params(self):
        p = self.params
        # Validate EMA periods
        if p.get('fast_ema', 5) < 3:
            raise ValueError("fast_ema must be >= 3")
        if p.get('medium_ema', 15) <= p.get('fast_ema', 5):
            raise ValueError("medium_ema must be > fast_ema")
        if p.get('slow_ema', 30) <= p.get('medium_ema', 15):
            raise ValueError("slow_ema must be > medium_ema")

        # Validate acceleration parameters
        if p.get('accel_window', 3) < 1:
            raise ValueError("accel_window must be >= 1")
        if p.get('accel_threshold', 0.1) <= 0:
            raise ValueError("accel_threshold must be > 0")

        # Validate volume parameters
        if p.get('vol_profile_mult', 1.5) <= 1.0:
            raise ValueError("vol_profile_mult must be > 1.0")
        if p.get('volume_min', 1000) <= 0:
            raise ValueError("volume_min must be > 0")

    def generate_signals(self) -> List[Dict]:
        df = self.data.copy()
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'DEFAULT'
        p = self.params

        has_volume = 'volume' in df.columns and df['volume'].notna().any()

        if not has_volume:
            print("VolumePressureEMA: 缺少成交量数据，无法生成信号")
            return []

        vol = df['volume'].values.astype(float)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)

        # Compute VDP (Volume Delta Pressure)
        hl_range = high - low
        safe_range = np.where(hl_range > 1e-10, hl_range, 1e-10)
        vdp = vol * (2.0 * close - high - low) / safe_range

        # Compute multi-timeframe EMAs
        ema_fast = self._ema(vdp, p['fast_ema'])
        ema_medium = self._ema(vdp, p['medium_ema'])
        ema_slow = self._ema(vdp, p['slow_ema'])

        # Compute volume acceleration (2nd derivative of VDP)
        vdp_fast_smooth = self._ema(ema_fast, 3)  # Smooth VDP for derivative
        acceleration = self._compute_derivative(vdp_fast_smooth, p['accel_window'])

        # Compute ATR for trailing stop
        atr = self._compute_atr(high, low, close, p['atr_period'])

        # Compute volume profile for position sizing
        volume_profile = self._compute_volume_profile(vol, close, p['vol_profile_window'])

        # Initialize trading state
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        stop_price = 0.0
        high_water = 0.0
        low_water = float('inf')
        entry_bar = 0

        # Warm-up period
        warmup = max(p['fast_ema'], p['medium_ema'], p['slow_ema'],
                    p['atr_period'], p['vol_profile_window']) + 10

        for i in range(warmup, len(df)):
            ts = df.index[i]
            price = close[i]

            # Skip if any indicator is invalid
            if (np.isnan(ema_fast[i]) or np.isnan(ema_medium[i]) or
                np.isnan(ema_slow[i]) or np.isnan(acceleration[i]) or
                np.isnan(atr[i]) or np.isnan(volume_profile[i])):
                continue

            # Check for existing position exits
            if position != 0:
                bars_held = i - entry_bar

                # Update trailing stop
                if position == 1:  # Long position
                    new_stop = price - p['atr_stop_mult'] * atr[i]
                    stop_price = max(stop_price, new_stop)
                    if price <= stop_price:
                        self._record_signal(ts, 'sell', symbol, price,
                                           score=0, reason='atr_trailing_stop')
                        position = 0
                        continue

                elif position == -1:  # Short position
                    new_stop = price + p['atr_stop_mult'] * atr[i]
                    stop_price = min(stop_price, new_stop)
                    if price >= stop_price:
                        self._record_signal(ts, 'buy', symbol, price,
                                           score=0, reason='atr_trailing_stop')
                        position = 0
                        continue

                # Time-based exit
                if bars_held >= 60:
                    self._record_signal(ts, 'sell' if position == 1 else 'buy',
                                       symbol, price, score=0, reason='time_exit')
                    position = 0
                    continue

            # Entry logic - check for alignment
            ema_fast_val = ema_fast[i]
            ema_medium_val = ema_medium[i]
            ema_slow_val = ema_slow[i]
            accel_val = acceleration[i]

            # Check if all EMAs have the same sign
            all_positive = (ema_fast_val > 0 and ema_medium_val > 0 and ema_slow_val > 0)
            all_negative = (ema_fast_val < 0 and ema_medium_val < 0 and ema_slow_val < 0)

            # Check volume and acceleration conditions
            vol_enough = vol[i] >= p['volume_min']

            # Strong buy signal
            if (all_positive and
                accel_val > p['accel_threshold'] and
                vol_enough):

                # Check EMA alignment (not too far apart)
                ema_alignment = (abs(ema_fast_val - ema_medium_val) < p['ema_alignment_threshold'] and
                                abs(ema_medium_val - ema_slow_val) < p['ema_alignment_threshold'])

                if ema_alignment:
                    # Position size based on volume profile
                    position_size = self._get_position_size(volume_profile[i], p['vol_profile_mult'])

                    self._record_signal(ts, 'buy', symbol, price,
                                       score=position_size, reason='strong_buy')
                    position = 1
                    entry_price = price
                    stop_price = price - p['atr_stop_mult'] * atr[i]
                    high_water = price
                    entry_bar = i
                    continue

            # Strong sell signal
            elif (all_negative and
                  accel_val < -p['accel_threshold'] and
                  vol_enough):

                # Check EMA alignment
                ema_alignment = (abs(ema_fast_val - ema_medium_val) < p['ema_alignment_threshold'] and
                                abs(ema_medium_val - ema_slow_val) < p['ema_alignment_threshold'])

                if ema_alignment:
                    # Position size based on volume profile
                    position_size = self._get_position_size(volume_profile[i], p['vol_profile_mult'])

                    self._record_signal(ts, 'sell', symbol, price,
                                       score=position_size, reason='strong_sell')
                    position = -1
                    entry_price = price
                    stop_price = price + p['atr_stop_mult'] * atr[i]
                    low_water = price
                    entry_bar = i
                    continue

        # Close any open position at the end
        if position != 0 and len(df) > 0:
            last_ts = df.index[-1]
            last_price = close[-1]
            self._record_signal(last_ts, 'sell' if position == 1 else 'buy',
                                symbol, last_price, score=0, reason='end_of_data')

        print(f"VolumePressureEMA: 生成 {len(self.signals)} 个信号")
        return self.signals

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA over a numpy array."""
        result = np.empty_like(arr, dtype=float)
        result[:] = np.nan
        if len(arr) == 0 or period <= 0:
            return result

        # Find first valid value
        start_idx = 0
        while start_idx < len(arr) and np.isnan(arr[start_idx]):
            start_idx += 1

        if start_idx >= len(arr):
            return result

        alpha = 2.0 / (period + 1)
        result[start_idx] = arr[start_idx]

        for i in range(start_idx + 1, len(arr)):
            if np.isnan(arr[i]):
                result[i] = result[i - 1]
            else:
                result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]

        return result

    def _compute_derivative(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Compute derivative (rate of change) using central difference."""
        result = np.empty_like(arr, dtype=float)
        result[:] = np.nan

        if window < 1 or len(arr) < window * 2 + 1:
            return result

        for i in range(window, len(arr) - window):
            # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
            numerator = arr[i + window] - arr[i - window]
            denominator = 2 * window
            if denominator > 0:
                result[i] = numerator / denominator

        return result

    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Compute Average True Range."""
        n = len(high)
        if n < 2:
            return np.full(n, np.nan)

        tr = np.empty(n, dtype=float)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # Simple moving average of TR
        atr = np.full(n, np.nan)
        if period > 0:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = atr[i - 1] + (tr[i] - tr[i - period]) / period

        return atr

    def _compute_volume_profile(self, volume: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
        """Compute volume profile indicator - measures volume intensity."""
        n = len(volume)
        result = np.full(n, np.nan)

        if window <= 0:
            return result

        for i in range(window - 1, n):
            # Calculate volume-weighted close position
            window_volume = volume[i - window + 1:i + 1]
            window_high = np.max(close[i - window + 1:i + 1])
            window_low = np.min(close[i - window + 1:i + 1])

            if window_high > window_low:
                # Normalized position within the range
                normalized_pos = (close[i] - window_low) / (window_high - window_low)
                # Volume-weighted intensity
                volume_intensity = np.mean(window_volume) * normalized_pos
                result[i] = volume_intensity

        return result

    def _get_position_size(self, volume_profile: float, multiplier: float) -> float:
        """Calculate position size based on volume profile."""
        # Higher volume profile = higher conviction = larger position
        if np.isnan(volume_profile):
            return 1.0  # Default position size

        # Normalize volume profile to position size
        base_size = 1.0
        if volume_profile > multiplier:
            base_size = 1.5
        elif volume_profile > multiplier * 0.8:
            base_size = 1.2

        return base_size


__all__ = ['VolumePressureEMAStrategy']