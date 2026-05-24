"""
Wavelet Denoising Trading Strategy

Strategy Overview:
- Apply Haar wavelet decomposition to close prices (3 levels)
- Reconstruct using only detail coefficients at levels 2-3 (trend)
- Trade when denoised trend slope changes sign
- Use wavelet energy ratio (detail vs approximation) as volatility regime indicator
- High energy ratio = choppy → reduce position
- Low energy ratio = smooth → full position
- ATR trailing stop

Implementation Details:
- Manual Haar wavelet transform (no pywt dependency)
- Uses numpy for all computations
- Implements ATR for trailing stop loss
- Signal generation based on trend slope changes and volatility regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class WaveletDenoiseStrategy(BaseStrategy):
    """Wavelet Denoising Trading Strategy"""

    strategy_description = "Haar wavelet denoising with trend following and volatility regime adaptation"
    strategy_category = "wave"
    strategy_params_schema = {
        "wavelet_levels": {
            "type": "integer",
            "default": 3,
            "description": "Number of wavelet decomposition levels",
            "min": 2,
            "max": 5
        },
        "trend_window": {
            "type": "integer",
            "default": 5,
            "description": "Window for trend slope calculation",
            "min": 3,
            "max": 10
        },
        "energy_ratio_threshold": {
            "type": "float",
            "default": 0.7,
            "description": "Energy ratio threshold for volatility regime",
            "min": 0.1,
            "max": 1.0,
            "step": 0.1
        },
        "atr_period": {
            "type": "integer",
            "default": 14,
            "description": "ATR calculation period",
            "min": 7,
            "max": 30
        },
        "atr_multiplier": {
            "type": "float",
            "default": 2.0,
            "description": "ATR multiplier for trailing stop",
            "min": 1.0,
            "max": 4.0,
            "step": 0.5
        },
        "min_samples": {
            "type": "integer",
            "default": 50,
            "description": "Minimum samples required for wavelet decomposition",
            "min": 20,
            "max": 100
        }
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        # Initialize wavelet parameters first
        wavelet_levels = params.get('wavelet_levels', 3) if params else 3
        trend_window = params.get('trend_window', 5) if params else 5
        energy_ratio_threshold = params.get('energy_ratio_threshold', 0.7) if params else 0.7
        atr_period = params.get('atr_period', 14) if params else 14
        atr_multiplier = params.get('atr_multiplier', 2.0) if params else 2.0
        min_samples = params.get('min_samples', 50) if params else 50

        # Set as class attributes before super().__init__ calls validate_params
        self.wavelet_levels = wavelet_levels
        self.trend_window = trend_window
        self.energy_ratio_threshold = energy_ratio_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_samples = min_samples

        super().__init__(data, params)

        # Initialize state variables
        self.denoised_trend = None
        self.wavelet_energy_ratio = None
        self.atr_values = None
        self.trailing_stop = None
        self.position = None

    def get_default_params(self) -> Dict[str, Any]:
        """Default parameters for the strategy"""
        return {
            'wavelet_levels': 3,
            'trend_window': 5,
            'energy_ratio_threshold': 0.7,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'min_samples': 50
        }

    def validate_params(self):
        """Validate strategy parameters"""
        if self.wavelet_levels < 2 or self.wavelet_levels > 5:
            raise ValueError("wavelet_levels must be between 2 and 5")
        if self.trend_window < 3 or self.trend_window > 10:
            raise ValueError("trend_window must be between 3 and 10")
        if self.energy_ratio_threshold < 0.1 or self.energy_ratio_threshold > 1.0:
            raise ValueError("energy_ratio_threshold must be between 0.1 and 1.0")
        if self.atr_period < 7 or self.atr_period > 30:
            raise ValueError("atr_period must be between 7 and 30")
        if self.atr_multiplier < 1.0 or self.atr_multiplier > 4.0:
            raise ValueError("atr_multiplier must be between 1.0 and 4.0")
        if self.min_samples < 20 or self.min_samples > 100:
            raise ValueError("min_samples must be between 20 and 100")

    def haar_wavelet_decomposition(self, data: np.ndarray, levels: int) -> tuple:
        """
        Perform Haar wavelet decomposition manually

        Args:
            data: Input data array
            levels: Number of decomposition levels

        Returns:
            tuple: (approximation, details_list)
        """
        if len(data) < (2 ** levels):
            raise ValueError(f"Data length {len(data)} too small for {levels} levels of decomposition")

        # Initialize
        approx = data.copy().astype(float)
        details = []

        # Perform decomposition
        for _ in range(levels):
            n = len(approx)
            if n < 2:
                break

            # Pad to even length
            if n % 2 != 0:
                approx = np.append(approx, approx[-1])
                n = len(approx)

            # Haar wavelet transform
            approx_new = np.zeros(n // 2)
            detail = np.zeros(n // 2)

            for i in range(0, n, 2):
                approx_new[i // 2] = (approx[i] + approx[i + 1]) / 2
                detail[i // 2] = (approx[i] - approx[i + 1]) / 2

            approx = approx_new
            details.append(detail)

        return approx, details

    def reconstruct_trend(self, details: list, level_start: int = 1) -> np.ndarray:
        """
        Reconstruct signal using detail coefficients from specified levels

        Args:
            details: List of detail coefficients from decomposition
            level_start: Starting level for reconstruction (0-based)

        Returns:
            np.ndarray: Reconstructed trend
        """
        if not details or level_start >= len(details):
            return np.zeros(len(self.data))

        # Start with approximation
        recon = np.zeros(len(self.data))

        # Reconstruct using detail coefficients
        for level in range(level_start, len(details)):
            detail = details[level]

            # Upsample detail coefficients
            upsampled = np.zeros(len(self.data))
            step = 2 ** (level + 1)
            for i in range(len(detail)):
                upsampled[i * step : (i + 1) * step] = detail[i]

            recon += upsampled

        return recon

    def calculate_wavelet_energy_ratio(self, details: list) -> float:
        """
        Calculate energy ratio between detail and approximation coefficients

        Args:
            details: List of detail coefficients

        Returns:
            float: Energy ratio
        """
        if not details:
            return 0.0

        # Calculate energy in detail coefficients
        detail_energy = sum(np.sum(d ** 2) for d in details)

        # Approximation energy (from last level)
        if len(details) > 0:
            approx_energy = np.sum(details[-1] ** 2)
            # Avoid division by zero
            total_energy = detail_energy + approx_energy
            if total_energy > 0:
                return detail_energy / total_energy

        return 0.0

    def calculate_slope(self, data: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate slope of data using linear regression

        Args:
            data: Input data
            window: Window size for slope calculation

        Returns:
            np.ndarray: Slope values
        """
        slope = np.zeros(len(data))

        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1 : i + 1]
            x = np.arange(window)

            # Linear regression
            if len(window_data) > 1:
                slope[i] = np.polyfit(x, window_data, 1)[0]

        return slope

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period: int) -> np.ndarray:
        """
        Calculate Average True Range (ATR)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            np.ndarray: ATR values
        """
        tr = np.zeros(len(close))

        # Calculate True Range
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate ATR using smoothed moving average
        atr = np.zeros(len(close))
        atr[period - 1] = np.mean(tr[:period])

        for i in range(period, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    def generate_signals(self) -> List[Dict]:
        """Generate trading signals based on wavelet denoising"""
        self.signals = []

        if len(self.data) < self.min_samples:
            # Not enough data for wavelet decomposition
            return []

        # Extract price data
        close_prices = self.data['close'].values
        high_prices = self.data['high'].values
        low_prices = self.data['low'].values

        # Perform wavelet decomposition
        try:
            approx, details = self.haar_wavelet_decomposition(
                close_prices, self.wavelet_levels
            )

            # Reconstruct trend using detail coefficients
            self.denoised_trend = self.reconstruct_trend(details, level_start=1)

            # Calculate wavelet energy ratio
            self.wavelet_energy_ratio = self.calculate_wavelet_energy_ratio(details)

            # Calculate trend slope
            trend_slope = self.calculate_slope(self.denoised_trend, self.trend_window)

            # Calculate ATR
            self.atr_values = self.calculate_atr(high_prices, low_prices, close_prices,
                                               self.atr_period)

            # Initialize trailing stop
            self.trailing_stop = np.zeros(len(close_prices))

            # Generate signals
            for i in range(self.min_samples, len(close_prices)):
                # Determine volatility regime
                is_choppy = self.wavelet_energy_ratio > self.energy_ratio_threshold

                # Check for trend reversal
                if i >= self.trend_window:
                    # Trend change detection
                    if trend_slope[i] > 0 and trend_slope[i - self.trend_window] <= 0:
                        # Bullish trend reversal
                        position_size = 1.0 if not is_choppy else 0.5
                        price = close_prices[i]
                        self._record_signal(
                            timestamp=self.data.index[i],
                            action='buy',
                            symbol=self.data['symbol'].iloc[i],
                            price=price,
                            trend_slope=trend_slope[i],
                            energy_ratio=self.wavelet_energy_ratio,
                            position_size=position_size,
                            volatility_regime='choppy' if is_choppy else 'smooth'
                        )

                        # Set initial trailing stop
                        if i > 0:
                            self.trailing_stop[i] = price - self.atr_values[i] * self.atr_multiplier

                    elif trend_slope[i] < 0 and trend_slope[i - self.trend_window] >= 0:
                        # Bearish trend reversal
                        self._record_signal(
                            timestamp=self.data.index[i],
                            action='sell',
                            symbol=self.data['symbol'].iloc[i],
                            price=close_prices[i],
                            trend_slope=trend_slope[i],
                            energy_ratio=self.wavelet_energy_ratio,
                            volatility_regime='choppy' if is_choppy else 'smooth'
                        )

                    # Update trailing stop for long positions
                    if self.position == 'long' and i > 0:
                        new_stop = close_prices[i] - self.atr_values[i] * self.atr_multiplier
                        if new_stop > self.trailing_stop[i - 1]:
                            self.trailing_stop[i] = new_stop

                        # Stop loss hit
                        if close_prices[i] < self.trailing_stop[i]:
                            self._record_signal(
                                timestamp=self.data.index[i],
                                action='sell',
                                symbol=self.data['symbol'].iloc[i],
                                price=close_prices[i],
                                reason='stop_loss',
                                atr_stop=self.trailing_stop[i]
                            )
                            self.position = None
                    else:
                        self.trailing_stop[i] = self.trailing_stop[i - 1]

                # Update position state
                if self.position is None:
                    last_signal = self.signals[-1] if self.signals else None
                    if last_signal and last_signal['action'] == 'buy':
                        self.position = 'long'

        except Exception as e:
            # Handle wavelet decomposition errors
            print(f"Wavelet decomposition error: {e}")
            # Generate hold signals as fallback
            for i in range(len(self.data)):
                self._record_signal(
                    timestamp=self.data.index[i],
                    action='hold',
                    symbol=self.data['symbol'].iloc[i],
                    price=close_prices[i]
                )

        return self.signals

    def screen(self) -> Dict:
        """Real-time screening for wavelet strategy"""
        if len(self.data) < self.min_samples:
            return {
                'action': 'hold',
                'reason': 'Insufficient data for wavelet decomposition',
                'price': float(self.data['close'].iloc[-1])
            }

        # Try to generate signals for screening
        try:
            signals = self.generate_signals()
            if signals:
                last_signal = signals[-1]
                return {
                    'action': last_signal.get('action', 'hold'),
                    'reason': f'Wavelet trend: {last_signal.get("trend_slope", 0):.4f}, '
                             f'Energy ratio: {last_signal.get("energy_ratio", 0):.3f}',
                    'price': float(last_signal.get('price', self.data['close'].iloc[-1]))
                }
        except Exception:
            pass

        return {
            'action': 'hold',
            'reason': f'{self.strategy_name}: No signal',
            'price': float(self.data['close'].iloc[-1])
        }