"""
Nadaraya-Watson Kernel Regression Trading Strategy
Uses kernel regression to smooth price data and identify trend changes
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy import stats
from scipy.signal import savgol_filter
from ..core.base_strategy import BaseStrategy


class NadarayaWatsonStrategy(BaseStrategy):
    """
    Nadaraya-Watson Kernel Regression Strategy

    Strategy Overview:
    - Uses Nadaraya-Watson kernel regression to smooth price data
    - Buy when NW slope turns positive (first derivative > 0 for 3 consecutive bars)
    - Sell when NW slope turns negative (first derivative < 0 for 3 consecutive bars)
    - Confidence from NW residual bands (price vs NW prediction)
    - Uses VDP confirmation for entry timing
    - ATR trailing stop for exit management
    """

    strategy_description = "Nadaraya-Watson Kernel Regression Strategy - Uses kernel smoothing to identify trends"
    strategy_category = "ml"

    def __init__(self, data: pd.DataFrame, params: dict = None):
        # Define default parameters
        self.default_params = {
            'window_size': 60,           # Rolling window for NW regression
            'bandwidth_multiplier': 0.5,   # Multiplier for adaptive bandwidth
            'min_slope_periods': 3,       # Consecutive bars for slope confirmation
            'atr_period': 14,             # ATR calculation period
            'atr_multiplier': 2.0,        # ATR trailing stop multiplier
            'vdp_threshold': 0.6,         # Volume Delta Confirmation threshold
            'residual_threshold': 2.0,    # Residual band threshold for confidence
            'kernel_type': 'gaussian',    # Kernel function type
            'smoothing_alpha': 0.1        # Smoothing factor for NW values
        }

        super().__init__(data, params)

        # Compute indicators during initialization
        self._compute_indicators()

    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters"""
        return self.default_params

    def validate_params(self):
        """Validate parameters"""
        if self.params['window_size'] < 10:
            raise ValueError("window_size must be >= 10")
        if self.params['bandwidth_multiplier'] <= 0:
            raise ValueError("bandwidth_multiplier must be > 0")
        if self.params['min_slope_periods'] < 1:
            raise ValueError("min_slope_periods must be >= 1")
        if self.params['atr_period'] < 1:
            raise ValueError("atr_period must be >= 1")
        if self.params['atr_multiplier'] <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if self.params['vdp_threshold'] <= 0 or self.params['vdp_threshold'] > 1:
            raise ValueError("vdp_threshold must be between 0 and 1")
        if self.params['residual_threshold'] <= 0:
            raise ValueError("residual_threshold must be > 0")
        if self.params['kernel_type'] not in ['gaussian', 'epanechnikov', 'uniform']:
            raise ValueError("kernel_type must be one of: gaussian, epanechnikov, uniform")
        if self.params['smoothing_alpha'] < 0 or self.params['smoothing_alpha'] > 1:
            raise ValueError("smoothing_alpha must be between 0 and 1")

    def _gaussian_kernel(self, x, h):
        """Gaussian kernel function"""
        return np.exp(-0.5 * (x / h) ** 2) / (h * np.sqrt(2 * np.pi))

    def _epanechnikov_kernel(self, x, h):
        """Epanechnikov kernel function"""
        u = x / h
        mask = np.abs(u) <= 1
        result = np.zeros_like(x)
        result[mask] = (3/4) * (1 - u[mask]**2) / h
        return result

    def _uniform_kernel(self, x, h):
        """Uniform kernel function"""
        u = x / h
        mask = np.abs(u) <= 1
        result = np.zeros_like(x)
        result[mask] = 1 / (2 * h)
        return result

    def _compute_nadaraya_watson(self, series: pd.Series, window_size: int) -> pd.Series:
        """
        Compute Nadaraya-Watson kernel regression for a series
        """
        nw_values = np.zeros(len(series))

        for i in range(window_size, len(series)):
            # Get window data
            window_data = series.iloc[i-window_size+1:i+1]
            timestamps = np.arange(len(window_data))

            # Compute adaptive bandwidth based on ATR
            atr_window = window_data.diff().abs()
            atr = atr_window.mean() if len(atr_window) > 0 else 1.0
            bandwidth = self.params['bandwidth_multiplier'] * atr

            # Compute kernel weights
            if self.params['kernel_type'] == 'gaussian':
                weights = self._gaussian_kernel(timestamps - (len(window_data)-1), bandwidth)
            elif self.params['kernel_type'] == 'epanechnikov':
                weights = self._epanechnikov_kernel(timestamps - (len(window_data)-1), bandwidth)
            else:  # uniform
                weights = self._uniform_kernel(timestamps - (len(window_data)-1), bandwidth)

            # Compute NW estimate
            nw_values[i] = np.sum(weights * window_data) / np.sum(weights)

        return pd.Series(nw_values, index=series.index)

    def _compute_slope(self, series: pd.Series, periods: int = 1) -> pd.Series:
        """Compute slope of series using finite differences"""
        diff = series.diff(periods)
        return diff

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _compute_vdp(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """Compute Volume Delta Confirmation"""
        # Positive volume delta
        pos_delta = volume * (close > close.shift(1))
        neg_delta = volume * (close < close.shift(1))

        # Smooth the delta values
        pos_smooth = pos_delta.rolling(window=period).mean()
        neg_smooth = neg_delta.rolling(window=period).mean()

        # VDP ratio
        vdp = pos_smooth / (pos_smooth + neg_smooth + 1e-8)
        return vdp

    def _compute_indicators(self):
        """Compute all required indicators"""
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']

        # Compute Nadaraya-Watson regression
        self.data['nw_regression'] = self._compute_nadaraya_watson(close, self.params['window_size'])

        # Apply smoothing to NW values
        if self.params['smoothing_alpha'] > 0:
            self.data['nw_smooth'] = self.data['nw_regression'] * (1 - self.params['smoothing_alpha']) + \
                                   self.data['nw_regression'].shift(1) * self.params['smoothing_alpha']
        else:
            self.data['nw_smooth'] = self.data['nw_regression']

        # Compute slope (first derivative)
        self.data['nw_slope'] = self._compute_slope(self.data['nw_smooth'], 1)

        # Compute cumulative slope for consecutive analysis
        self.data['slope_cumulative'] = (self.data['nw_slope'] > 0).astype(int)
        self.data['slope_cumulative'] = self.data['slope_cumulative'].groupby(
            (self.data['slope_cumulative'] != self.data['slope_cumulative'].shift()).cumsum()
        ).cumsum()

        # Compute ATR
        self.data['atr'] = self._compute_atr(high, low, close, self.params['atr_period'])

        # Compute VDP if volume data is available
        if 'volume' in self.data.columns:
            self.data['vdp'] = self._compute_vdp(self.data['volume'])
        else:
            self.data['vdp'] = pd.Series(0.5, index=self.data.index)  # Neutral if no volume

        # Compute NW residuals
        self.data['nw_residual'] = close - self.data['nw_smooth']
        self.data['residual_std'] = self.data['nw_residual'].rolling(window=self.params['window_size']).std()

        # Compute confidence metric based on residual bands
        self.data['confidence'] = 1 / (1 + np.abs(self.data['nw_residual'] / (self.data['residual_std'] + 1e-8)))

        # Fill NaN values
        self.data = self.data.fillna(method='bfill')

    def generate_signals(self) -> List[Dict]:
        """Generate trading signals"""
        signals = []

        # Start after window_size + min_slope_periods
        start_idx = self.params['window_size'] + self.params['min_slope_periods']

        for i in range(start_idx, len(self.data)):
            timestamp = self.data.index[i]
            price = self.data['close'].iloc[i]

            # Check for buy signal
            if self._check_buy_signal(i):
                signal = {
                    'timestamp': timestamp,
                    'action': 'buy',
                    'symbol': self.data['symbol'].iloc[i],
                    'price': price,
                    'confidence': self.data['confidence'].iloc[i],
                    'atr': self.data['atr'].iloc[i],
                    'nw_value': self.data['nw_smooth'].iloc[i],
                    'slope': self.data['nw_slope'].iloc[i],
                    'residual': self.data['nw_residual'].iloc[i]
                }
                signals.append(signal)

            # Check for sell signal
            elif self._check_sell_signal(i):
                signal = {
                    'timestamp': timestamp,
                    'action': 'sell',
                    'symbol': self.data['symbol'].iloc[i],
                    'price': price,
                    'confidence': self.data['confidence'].iloc[i],
                    'atr': self.data['atr'].iloc[i],
                    'nw_value': self.data['nw_smooth'].iloc[i],
                    'slope': self.data['nw_slope'].iloc[i],
                    'residual': self.data['nw_residual'].iloc[i]
                }
                signals.append(signal)

            # Hold signal for every bar (useful for backtesting)
            else:
                signal = {
                    'timestamp': timestamp,
                    'action': 'hold',
                    'symbol': self.data['symbol'].iloc[i],
                    'price': price
                }
                signals.append(signal)

        return signals

    def _check_buy_signal(self, idx: int) -> bool:
        """Check if buy signal condition is met"""
        # Check slope conditions
        nw_slope = self.data['nw_slope'].iloc[idx]

        # Check for positive slope for min_slope_periods consecutive bars
        if nw_slope <= 0:
            return False

        # Check slope cumulative count
        slope_cum = self.data['slope_cumulative'].iloc[idx]
        if slope_cum < self.params['min_slope_periods']:
            return False

        # VDP confirmation
        vdp = self.data['vdp'].iloc[idx]
        if vdp < self.params['vdp_threshold']:
            return False

        # Confidence check
        confidence = self.data['confidence'].iloc[idx]
        if confidence < 0.5:  # Minimum confidence threshold
            return False

        # Price must be above NW regression for confirmation
        price_above_nw = self.data['close'].iloc[idx] > self.data['nw_smooth'].iloc[idx]

        return price_above_nw

    def _check_sell_signal(self, idx: int) -> bool:
        """Check if sell signal condition is met"""
        # Check slope conditions
        nw_slope = self.data['nw_slope'].iloc[idx]

        # Check for negative slope for min_slope_periods consecutive bars
        if nw_slope >= 0:
            return False

        # Check slope cumulative count
        slope_cum = self.data['slope_cumulative'].iloc[idx]
        if slope_cum < self.params['min_slope_periods']:
            return False

        # Confidence check
        confidence = self.data['confidence'].iloc[idx]
        if confidence < 0.5:  # Minimum confidence threshold
            return False

        # Price must be below NW regression for confirmation
        price_below_nw = self.data['close'].iloc[idx] < self.data['nw_smooth'].iloc[idx]

        return price_below_nw

    def screen(self) -> Dict:
        """
        Real-time screening method for quick signal generation
        """
        if len(self.data) < self.params['window_size'] + self.params['min_slope_periods']:
            return {'action': 'hold', 'reason': '数据不足',
                    'price': float(self.data['close'].iloc[-1])}

        idx = len(self.data) - 1
        timestamp = self.data.index[idx]
        price = float(self.data['close'].iloc[idx])

        # Check conditions
        is_buy = self._check_buy_signal(idx)
        is_sell = self._check_sell_signal(idx)

        if is_buy:
            return {
                'action': 'buy',
                'reason': 'NW slope positive with VDP confirmation',
                'price': price,
                'confidence': self.data['confidence'].iloc[idx],
                'atr': self.data['atr'].iloc[idx]
            }
        elif is_sell:
            return {
                'action': 'sell',
                'reason': 'NW slope negative',
                'price': price,
                'confidence': self.data['confidence'].iloc[idx],
                'atr': self.data['atr'].iloc[idx]
            }
        else:
            return {
                'action': 'hold',
                'reason': f'{self.strategy_name}: 无信号',
                'price': price
            }

    def get_trailing_stop(self, entry_price: float, position_type: str = 'long') -> float:
        """
        Calculate ATR trailing stop loss
        """
        idx = len(self.data) - 1
        atr = self.data['atr'].iloc[idx]

        if position_type == 'long':
            stop = entry_price - (self.params['atr_multiplier'] * atr)
        else:  # short
            stop = entry_price + (self.params['atr_multiplier'] * atr)

        return stop