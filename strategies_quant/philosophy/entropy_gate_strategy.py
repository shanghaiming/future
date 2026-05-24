"""
Shannon Entropy Gate Strategy (EntropyGateStrategy)
===================================================
Only trade when market has low entropy (ordered/structured).

Core Logic (无为 / Wu Wei):
  1. Compute rolling Shannon entropy on binned returns (10 bins, 50-day window)
  2. H < 0.7 * log2(10) = low entropy = ordered -> allow trading
  3. H > 0.8 * log2(10) = high entropy = chaotic -> block all signals
  4. When entropy is low, use structural tension (7-point displacement) for direction
  5. When entropy is medium, require VDP delta confirmation
  6. ATR trailing stop for exits
  7. Score: base from structural tension + VDP bonus, gated by entropy level

Mathematical Foundation:
  Shannon Entropy:
    H(X) = -sum(p_i * log2(p_i))
    Maximum entropy for 10 bins: log2(10) ~ 3.322 bits
    Low threshold: 0.7 * 3.322 = 2.325 bits (ordered market)
    High threshold: 0.8 * 3.322 = 2.658 bits (chaotic market)

  VDP (Volume Delta Pressure):
    delta = V * (2*C - H - L) / (H - L)
    cum_delta = EMA(delta, period)
    Positive cum_delta = buying pressure, negative = selling pressure

Technical Indicators: Shannon Entropy, Structural Tension, VDP, ATR
Category: adaptive
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class EntropyGateStrategy(BaseStrategy):
    """Shannon Entropy Gate Strategy -- trade only in ordered markets"""

    strategy_description = (
        "EntropyGate: Shannon entropy gate + structural tension direction + "
        "VDP delta confirmation + ATR trailing stop"
    )
    strategy_category = "adaptive"
    strategy_params_schema = {
        "entropy_bins": {"type": "int", "default": 10, "label": "Entropy bin count"},
        "entropy_window": {"type": "int", "default": 50, "label": "Entropy rolling window"},
        "entropy_low_ratio": {"type": "float", "default": 0.7, "label": "Low entropy threshold ratio"},
        "entropy_high_ratio": {"type": "float", "default": 0.8, "label": "High entropy threshold ratio"},
        "pivot_len": {"type": "int", "default": 5, "label": "Pivot lookback for structural tension"},
        "vdp_ema_period": {"type": "int", "default": 10, "label": "VDP delta EMA period"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "Trail stop ATR multiplier"},
        "max_hold": {"type": "int", "default": 60, "label": "Max holding days"},
        "score_threshold": {"type": "int", "default": 4, "label": "Signal score threshold"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.entropy_bins = p.get('entropy_bins', 10)
        self.entropy_window = p.get('entropy_window', 50)
        self.entropy_low_ratio = p.get('entropy_low_ratio', 0.7)
        self.entropy_high_ratio = p.get('entropy_high_ratio', 0.8)
        self.pivot_len = p.get('pivot_len', 5)
        self.vdp_ema_period = p.get('vdp_ema_period', 10)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 60)
        self.score_threshold = p.get('score_threshold', 4)

        # Precompute thresholds
        self._max_entropy = np.log2(self.entropy_bins)
        self._low_thresh = self.entropy_low_ratio * self._max_entropy
        self._high_thresh = self.entropy_high_ratio * self._max_entropy

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'entropy_bins': 10, 'entropy_window': 50,
            'entropy_low_ratio': 0.7, 'entropy_high_ratio': 0.8,
            'pivot_len': 5, 'vdp_ema_period': 10,
            'atr_period': 14, 'hold_min': 3,
            'trail_atr_mult': 2.5, 'max_hold': 60,
            'score_threshold': 4,
        }

    # ================================================================
    # Signal Generation
    # ================================================================

    def generate_signals(self) -> List[Dict]:
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []

        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0
        low_water = float('inf')

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            # --- No position: seek entry ---
            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    result = self._evaluate(sym, hist)
                    if result is None:
                        continue
                    score, direction, _ = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= self.score_threshold:
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0
                    low_water = float('inf')

            # --- In position: check exits ---
            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) if low_water < float('inf') else current_price

                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding) & (data.index <= current_time)]
                    atr_val = self._calc_atr(hist)
                    should_exit = False

                    # ATR trailing stop
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # Max hold
                    if days_held >= self.max_hold:
                        should_exit = True

                    # Signal-based exit: reversed structural tension + entropy check
                    if not should_exit:
                        result = self._evaluate(current_holding, hist)
                        if result is not None:
                            score, direction, _ = result
                            if position_dir == 1 and direction == -1 and score < -self.score_threshold:
                                should_exit = True
                            elif position_dir == -1 and direction == 1 and score > self.score_threshold:
                                should_exit = True

                    if should_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding)
                        else:
                            self._record_signal(current_time, 'buy', current_holding)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0
                        low_water = float('inf')

        print(f"EntropyGate: generated {len(self.signals)} signals")
        return self.signals

    # ================================================================
    # Core Evaluation
    # ================================================================

    def _evaluate(self, symbol: str, data: pd.DataFrame):
        """
        Evaluate entropy gate + structural tension + VDP.

        Returns:
            (score, direction, atr) or None if insufficient data or entropy gate blocks.
        """
        min_len = max(self.entropy_window, self.pivot_len * 2 + 20, self.vdp_ema_period + 10) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Step 1: Compute rolling Shannon entropy
        entropy = self._calc_entropy(close)
        if np.isnan(entropy):
            return None

        # Step 2: Entropy gate
        # High entropy -> block all signals (Wu Wei: do nothing in chaos)
        if entropy > self._high_thresh:
            return None

        # Step 3: Determine entropy level
        is_low_entropy = entropy < self._low_thresh
        is_medium_entropy = self._low_thresh <= entropy <= self._high_thresh

        # Step 4: Structural tension for direction (base score)
        tension = self._calc_structural_tension(data, close, atr)
        base_score = self._tension_to_score(tension)

        # Step 5: VDP delta confirmation (required for medium entropy)
        cum_delta = self._calc_vdp_cum_delta(data)
        vdp_bonus = 0

        if is_medium_entropy:
            # In medium entropy, VDP delta confirmation is REQUIRED
            if np.isnan(cum_delta):
                return None  # No VDP data, cannot confirm -> block
            if base_score > 0 and cum_delta <= 0:
                return None  # Bullish tension but no buying pressure -> block
            if base_score < 0 and cum_delta >= 0:
                return None  # Bearish tension but no selling pressure -> block
            # VDP alignment bonus
            if (base_score > 0 and cum_delta > 0) or (base_score < 0 and cum_delta < 0):
                vdp_bonus = 2
        elif is_low_entropy:
            # Low entropy: VDP bonus but not required
            if not np.isnan(cum_delta):
                if (base_score > 0 and cum_delta > 0) or (base_score < 0 and cum_delta < 0):
                    vdp_bonus = 2
                elif (base_score > 0 and cum_delta < 0) or (base_score < 0 and cum_delta > 0):
                    vdp_bonus = -1  # Conflicting signal, reduce conviction

        # Step 6: Entropy-level gating multiplier
        # Lower entropy = stronger gate multiplier
        if is_low_entropy:
            entropy_mult = 1.0
        else:
            # Medium entropy: scale down as entropy approaches high threshold
            ratio = (self._high_thresh - entropy) / (self._high_thresh - self._low_thresh)
            entropy_mult = max(0.3, ratio)

        # Step 7: Final score
        final_score = int((base_score + vdp_bonus) * entropy_mult)

        direction = 1 if final_score > 0 else -1

        return final_score, direction, atr

    # ================================================================
    # Shannon Entropy
    # ================================================================

    def _calc_entropy(self, close):
        """
        Rolling Shannon entropy on binned log returns.

        H(X) = -sum(p_i * log2(p_i))

        Returns:
            float: Shannon entropy in bits, or np.nan if insufficient data
        """
        if len(close) < self.entropy_window + 1:
            return np.nan

        # Log returns over the rolling window
        recent = close[-(self.entropy_window + 1):]
        returns = np.diff(np.log(recent))
        returns = returns[np.isfinite(returns)]

        if len(returns) < 10:
            return np.nan

        # Bin returns into histogram
        counts, _ = np.histogram(returns, bins=self.entropy_bins, density=False)
        total = np.sum(counts)
        if total == 0:
            return np.nan

        probs = counts[counts > 0] / total
        entropy = float(-np.sum(probs * np.log2(probs)))
        return entropy

    # ================================================================
    # Structural Tension (7-Point Displacement)
    # ================================================================

    def _calc_structural_tension(self, data, close, atr):
        """
        7-point structural tension oscillator.
        Positive tension = bullish displacement, negative = bearish.
        """
        n = len(close)
        if n < self.pivot_len * 2 + 5:
            return 0.0

        high_arr = data['high'].values.astype(float)
        low_arr = data['low'].values.astype(float)

        # Find swing points
        swing_highs = []
        swing_lows = []
        for i in range(self.pivot_len, n - self.pivot_len):
            is_high = all(high_arr[i] >= high_arr[i + j]
                         for j in range(-self.pivot_len, self.pivot_len + 1) if j != 0)
            is_low = all(low_arr[i] <= low_arr[i + j]
                        for j in range(-self.pivot_len, self.pivot_len + 1) if j != 0)
            if is_high:
                swing_highs.append((i, high_arr[i]))
            if is_low:
                swing_lows.append((i, low_arr[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.0

        # 7 structural reference points
        prev_sh = swing_highs[-2][1]
        curr_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1]
        curr_sl = swing_lows[-1][1]
        inter_high = max(curr_sh, prev_sh)
        inter_low = min(curr_sl, prev_sl)
        inter_mid = (inter_high + inter_low) / 2.0

        ref_points = [prev_sh, curr_sh, prev_sl, curr_sl, inter_high, inter_low, inter_mid]
        current_price = close[-1]

        if atr <= 0:
            return 0.0

        displacements = [(current_price - rp) / atr for rp in ref_points]
        tension = float(np.mean(displacements))
        return tension

    def _tension_to_score(self, tension):
        """Convert structural tension to integer score."""
        if tension > 1.0:
            return 5
        elif tension > 0.5:
            return 3
        elif tension > 0.2:
            return 1
        elif tension < -1.0:
            return -5
        elif tension < -0.5:
            return -3
        elif tension < -0.2:
            return -1
        return 0

    # ================================================================
    # VDP (Volume Delta Pressure)
    # ================================================================

    def _calc_vdp_cum_delta(self, data):
        """
        Compute VDP cumulative delta.

        delta = V * (2*C - H - L) / (H - L)
        cum_delta = EMA(delta, period)

        Returns:
            float: latest cumulative delta, or np.nan if no volume data
        """
        vol_col = None
        for vc in ['vol', 'volume', 'Volume']:
            if vc in data.columns:
                vol_col = vc
                break

        if vol_col is None:
            return np.nan

        n = len(data)
        if n < 2:
            return np.nan

        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        vol = data[vol_col].values.astype(float)
        vol = np.nan_to_num(vol, nan=0.0)

        # Compute delta per bar
        deltas = np.full(n, 0.0)
        for i in range(n):
            hl_range = high[i] - low[i]
            if hl_range < 1e-10:
                deltas[i] = 0.0
            else:
                deltas[i] = vol[i] * (2 * close[i] - high[i] - low[i]) / hl_range

        # EMA of delta
        period = self.vdp_ema_period
        k = 2.0 / (period + 1)
        ema = np.full(n, 0.0)
        ema[0] = deltas[0]
        for i in range(1, n):
            ema[i] = deltas[i] * k + ema[i - 1] * (1 - k)

        return float(ema[-1])

    # ================================================================
    # ATR
    # ================================================================

    def _calc_atr(self, data):
        """ATR calculation."""
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        n = len(high)
        tr_list = []
        for i in range(max(1, n - self.atr_period), n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            tr_list.append(tr)
        return float(np.mean(tr_list)) if tr_list else 0.0

    def screen(self):
        """Quick screen based on latest data."""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        sym = data['symbol'].iloc[0]
        min_len = max(self.entropy_window, self.pivot_len * 2 + 20, self.vdp_ema_period + 10) + 10
        if len(data) < min_len:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}

        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])

        close = data['close'].values.astype(float)
        entropy = self._calc_entropy(close)

        if result is None:
            if not np.isnan(entropy):
                gate = 'BLOCKED' if entropy > self._high_thresh else 'no signal'
                return {
                    'action': 'hold',
                    'reason': f'entropy={entropy:.2f} ({gate})',
                    'price': price,
                }
            return {'action': 'hold', 'reason': 'evaluation failed', 'price': price}

        score, direction, _ = result
        gate_status = 'low' if entropy < self._low_thresh else 'medium'
        if abs(score) >= self.score_threshold:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f'score={score} entropy={entropy:.2f}({gate_status})',
                'price': price,
            }
        return {
            'action': 'hold',
            'reason': f'score={score} entropy={entropy:.2f}({gate_status})',
            'price': price,
        }
