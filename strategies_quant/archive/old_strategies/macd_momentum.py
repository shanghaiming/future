"""
MACD Momentum Strategy
Uses MACD histogram momentum shifts + volume confirmation.
Buys when MACD histogram turns positive with rising volume.
Sells when histogram momentum weakens.
"""
from core.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class MACDMomentumStrategy(BaseStrategy):
    strategy_description = "MACD histogram momentum with volume confirmation"
    strategy_category = "momentum"
    strategy_params_schema = {
        'fast': {'type': 'int', 'default': 12, 'label': 'Fast EMA'},
        'slow': {'type': 'int', 'default': 26, 'label': 'Slow EMA'},
        'signal': {'type': 'int', 'default': 9, 'label': 'Signal Period'},
        'vol_ma': {'type': 'int', 'default': 20, 'label': 'Volume MA Period'},
        'vol_ratio': {'type': 'float', 'default': 1.2, 'label': 'Volume Ratio Threshold'},
    }

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fast': 5,
            'slow': 35,
            'signal': 5,
            'vol_ma': 15,
            'vol_ratio': 1.0,
        }

    def generate_signals(self) -> List[Dict]:
        data = self.data
        p = self.params
        close = data['close']

        # MACD
        ema_fast = close.ewm(span=p['fast'], adjust=False).mean()
        ema_slow = close.ewm(span=p['slow'], adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=p['signal'], adjust=False).mean()
        histogram = macd_line - signal_line

        # Volume ratio
        vol = data['volume'] if 'volume' in data.columns else pd.Series(0, index=data.index)
        vol_ma = vol.rolling(p['vol_ma']).mean()
        vol_ratio = vol / vol_ma.replace(0, np.nan)

        holding = False
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'DEFAULT'

        warmup = p['slow'] + p['signal'] + 5

        for i in range(warmup, len(data)):
            if np.isnan(histogram.iloc[i]) or np.isnan(histogram.iloc[i - 1]):
                continue

            prev_hist = histogram.iloc[i - 1]
            curr_hist = histogram.iloc[i]
            curr_vol_r = vol_ratio.iloc[i] if not np.isnan(vol_ratio.iloc[i]) else 1.0

            # Buy signal: histogram crosses above zero (momentum shift)
            # with volume confirmation
            if (not holding and
                prev_hist <= 0 and curr_hist > 0 and
                    curr_vol_r > p['vol_ratio']):
                self._record_signal(data.index[i], 'buy', symbol, close.iloc[i])
                holding = True

            # Sell signal: histogram starts declining (momentum weakening)
            elif holding:
                # Momentum fading: histogram decreasing for 3 consecutive bars
                if i >= 2 and (histogram.iloc[i] < histogram.iloc[i - 1] <
                               histogram.iloc[i - 2] < histogram.iloc[i - 3]):
                    self._record_signal(data.index[i], 'sell', symbol, close.iloc[i])
                    holding = False

                # Also sell on MACD death cross
                elif (macd_line.iloc[i] < signal_line.iloc[i] and
                      macd_line.iloc[i - 1] >= signal_line.iloc[i - 1]):
                    self._record_signal(data.index[i], 'sell', symbol, close.iloc[i])
                    holding = False

        if holding and len(data) > 0:
            self._record_signal(data.index[-1], 'sell', symbol, close.iloc[-1])

        return self.signals
