"""
RSI Divergence Strategy
Detects bullish/bearish divergence between price and RSI.
Bullish divergence: price makes lower low, RSI makes higher low -> buy
Bearish divergence: price makes higher high, RSI makes lower high -> sell
"""
from core.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class RSIDivergenceStrategy(BaseStrategy):
    strategy_description = "RSI divergence detection - bullish/bearish divergences"
    strategy_category = "momentum"
    strategy_params_schema = {
        'rsi_period': {'type': 'int', 'default': 14, 'label': 'RSI Period'},
        'lookback': {'type': 'int', 'default': 30, 'label': 'Lookback Window'},
        'overbought': {'type': 'float', 'default': 70, 'label': 'Overbought Threshold'},
        'oversold': {'type': 'float', 'default': 30, 'label': 'Oversold Threshold'},
    }

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'rsi_period': 14,
            'lookback': 30,
            'overbought': 70,
            'oversold': 30,
        }

    def generate_signals(self) -> List[Dict]:
        data = self.data
        params = self.params
        period = params['rsi_period']
        lookback = params['lookback']

        close = data['close']
        high = data['high']
        low = data['low']

        # Calculate RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Find local peaks and troughs in price and RSI
        signals = []
        holding = False
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'DEFAULT'

        for i in range(lookback, len(data)):
            window_close = close.iloc[i - lookback:i + 1].values
            window_rsi = rsi.iloc[i - lookback:i + 1].values

            # Skip if NaN values
            if np.any(np.isnan(window_rsi)):
                continue

            current_rsi = window_rsi[-1]

            # Find two recent troughs (for bullish divergence)
            trough_indices = self._find_troughs(window_close)
            trough_rsi = self._find_troughs(window_rsi)

            # Find two recent peaks (for bearish divergence)
            peak_indices = self._find_peaks(window_close)
            peak_rsi = self._find_peaks(window_rsi)

            # Bullish divergence: price lower low + RSI higher low
            if (len(trough_indices) >= 2 and len(trough_rsi) >= 2 and
                window_close[trough_indices[-1]] < window_close[trough_indices[-2]] and
                    window_rsi[trough_rsi[-1]] > window_rsi[trough_rsi[-2]]):
                if not holding and current_rsi < 50:
                    self._record_signal(data.index[i], 'buy', symbol, close.iloc[i])
                    holding = True

            # Bearish divergence: price higher high + RSI lower high
            elif (len(peak_indices) >= 2 and len(peak_rsi) >= 2 and
                  window_close[peak_indices[-1]] > window_close[peak_indices[-2]] and
                      window_rsi[peak_rsi[-1]] < window_rsi[peak_rsi[-2]]):
                if holding:
                    self._record_signal(data.index[i], 'sell', symbol, close.iloc[i])
                    holding = False

            # Also sell on overbought
            if holding and current_rsi > params['overbought']:
                self._record_signal(data.index[i], 'sell', symbol, close.iloc[i])
                holding = False

        # Close position at end
        if holding and len(data) > 0:
            self._record_signal(data.index[-1], 'sell', symbol, close.iloc[-1])

        return self.signals

    @staticmethod
    def _find_peaks(arr):
        peaks = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peaks.append(i)
        return peaks

    @staticmethod
    def _find_troughs(arr):
        troughs = []
        for i in range(1, len(arr) - 1):
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                troughs.append(i)
        return troughs
