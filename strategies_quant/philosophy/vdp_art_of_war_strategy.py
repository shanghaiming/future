"""
VDPArtOfWarStrategy - Volume Delta Pressure with Art of War Deception Detection

Philosophy: "兵者，诡道也" - War is deception; use volume to see through price deception.
Detects 虚实 (real vs fake breakouts) using volume delta pressure.

Core idea: When price breaks a swing level, volume delta reveals whether
institutional flow supports the move (实 real) or traps retail (虚 fake).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from core.base_strategy import BaseStrategy


class VDPArtOfWarStrategy(BaseStrategy):
    """Volume Delta Pressure with Art of War deception detection."""

    strategy_description = (
        "兵者诡道也 - Volume Delta Pressure detects real vs fake breakouts "
        "using VDP delta = volume * (2*close - high - low) / (high - low). "
        "Fake breakouts reversed, real breakouts confirmed."
    )
    strategy_category = "volume"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ema_period': 14,           # EMA smoothing for delta
            'pivot_len': 5,             # swing high/low lookback
            'vol_surge_mult': 1.3,      # volume surge threshold
            'atr_period': 14,           # ATR calculation period
            'atr_stop_mult': 2.0,       # ATR trailing stop multiplier
            'min_score': 4,             # minimum signal score
            'score_real_breakout': 5,   # score for real breakout confirmation
            'score_fake_breakout': 4,   # score for fake breakout reversal
            'score_vol_surge': 2,       # bonus for volume surge
        }

    def generate_signals(self) -> List[Dict]:
        df = self.data.copy()
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'DEFAULT'
        p = self.params

        has_volume = 'volume' in df.columns and df['volume'].notna().any()

        # --- Compute ATR ---
        atr = self._compute_atr(df, p['atr_period'])

        # --- Compute Volume Delta Pressure ---
        if has_volume:
            vol = df['volume'].values.astype(float)
        else:
            # Fallback: use close position within range as volume proxy (0.5 = midpoint)
            vol = np.ones(len(df), dtype=float)

        hl_range = df['high'].values.astype(float) - df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)

        # VDP delta = volume * (2*close - high - low) / (high - low)
        # When close is near high: delta > 0 (buy pressure)
        # When close is near low: delta < 0 (sell pressure)
        safe_range = np.where(hl_range > 1e-10, hl_range, 1e-10)
        raw_delta = vol * (2.0 * close - high - low) / safe_range

        # If no volume, use position ratio directly as delta proxy
        if not has_volume:
            raw_delta = (2.0 * close - high - low) / safe_range

        # EMA smooth delta
        ema_period = p['ema_period']
        smooth_delta = self._ema(raw_delta, ema_period)

        # --- Volume average for surge detection ---
        if has_volume:
            vol_avg = pd.Series(vol).rolling(window=20, min_periods=1).mean().values
        else:
            vol_avg = np.ones(len(df))

        # --- Swing high / low detection ---
        pivot_len = p['pivot_len']
        swing_highs = self._find_swing_highs(high, pivot_len)
        swing_lows = self._find_swing_lows(low, pivot_len)

        # Running last known swing levels
        last_swing_high = np.full(len(df), np.nan)
        last_swing_low = np.full(len(df), np.nan)
        cur_high = np.nan
        cur_low = np.nan
        for i in range(len(df)):
            if swing_highs[i]:
                cur_high = high[i]
            if swing_lows[i]:
                cur_low = low[i]
            last_swing_high[i] = cur_high
            last_swing_low[i] = cur_low

        # --- ATR trailing stop state ---
        position = 0       # 0=flat, 1=long, -1=short
        entry_price = 0.0
        stop_price = 0.0
        warmup = max(p['ema_period'], p['atr_period'], p['pivot_len'] * 2, 30)

        for i in range(warmup, len(df)):
            ts = df.index[i]
            price = close[i]
            delta_now = smooth_delta[i]
            atr_now = atr[i]

            if np.isnan(delta_now) or np.isnan(atr_now):
                continue

            sh = last_swing_high[i - 1] if i > 0 else np.nan
            sl = last_swing_low[i - 1] if i > 0 else np.nan

            if np.isnan(sh) or np.isnan(sl):
                continue

            # Volume surge flag
            vol_surge = (has_volume and vol[i] > p['vol_surge_mult'] * vol_avg[i])

            # --- Trailing stop check for existing position ---
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

            # --- Breakout detection with 虚实 analysis ---
            prev_close = close[i - 1] if i > 0 else close[i]
            score = 0
            action = None
            reason = ''

            # Upward breakout: price closes above swing high
            if price > sh and prev_close <= sh:
                if delta_now > 0:
                    # 实 (real breakout): volume delta confirms buying pressure
                    score = p['score_real_breakout']
                    action = 'buy'
                    reason = 'real_breakout_above_swing_high'
                else:
                    # 虚 (fake breakout): price breaks but delta negative = trap
                    score = p['score_fake_breakout']
                    action = 'sell'
                    reason = 'fake_breakout_above_swing_high_reversal'

            # Downward breakout: price closes below swing low
            elif price < sl and prev_close >= sl:
                if delta_now < 0:
                    # 实 (real breakdown): volume delta confirms selling pressure
                    score = p['score_real_breakout']
                    action = 'sell'
                    reason = 'real_breakdown_below_swing_low'
                else:
                    # 虚 (fake breakdown): liquidity sweep, buy signal
                    score = p['score_fake_breakout']
                    action = 'buy'
                    reason = 'fake_breakdown_liquidity_sweep'

            # Volume surge bonus
            if vol_surge and action is not None:
                score += p['score_vol_surge']

            # --- Emit signal if score threshold met and position allows ---
            if action is not None and score >= p['min_score']:
                if action == 'buy' and position <= 0:
                    self._record_signal(ts, 'buy', symbol, price,
                                       score=score, reason=reason)
                    position = 1
                    entry_price = price
                    stop_price = price - p['atr_stop_mult'] * atr_now
                elif action == 'sell' and position >= 0:
                    self._record_signal(ts, 'sell', symbol, price,
                                       score=score, reason=reason)
                    position = -1
                    entry_price = price
                    stop_price = price + p['atr_stop_mult'] * atr_now

        # Close open position at end of data
        if position != 0 and len(df) > 0:
            last_ts = df.index[-1]
            last_price = close[-1]
            self._record_signal(last_ts, 'sell' if position == 1 else 'buy',
                                symbol, last_price, score=0, reason='end_of_data_close')

        return self.signals

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA over a numpy array."""
        result = np.empty_like(arr, dtype=float)
        result[:] = np.nan
        if len(arr) == 0:
            return result
        alpha = 2.0 / (period + 1)
        # Find first valid index
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
        # Simple moving average of TR
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
        return atr

    @staticmethod
    def _find_swing_highs(high: np.ndarray, pivot_len: int) -> np.ndarray:
        """Boolean array marking pivot highs."""
        n = len(high)
        result = np.zeros(n, dtype=bool)
        for i in range(pivot_len, n - pivot_len):
            window = high[i - pivot_len: i + pivot_len + 1]
            if high[i] == np.max(window) and np.sum(window == high[i]) == 1:
                result[i] = True
        return result

    @staticmethod
    def _find_swing_lows(low: np.ndarray, pivot_len: int) -> np.ndarray:
        """Boolean array marking pivot lows."""
        n = len(low)
        result = np.zeros(n, dtype=bool)
        for i in range(pivot_len, n - pivot_len):
            window = low[i - pivot_len: i + pivot_len + 1]
            if low[i] == np.min(window) and np.sum(window == low[i]) == 1:
                result[i] = True
        return result


__all__ = ['VDPArtOfWarStrategy']
