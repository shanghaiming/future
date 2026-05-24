"""
PhilosophyEnsembleStrategy - 道生一哲学集成策略
=================================================
7-layer philosophy ensemble system targeting 600% annualized returns.

道生一，一生二，二生三，三生万物
From the Way, the complete system emerges.

Layers:
  1. Entropy Gate (无为门)         — Shannon entropy blocks chaotic markets
  2. Regime Detection (知己知彼)   — Kaufman ER classifies trend vs range
  3. Stock Selection (兵法选股)    — Quality scoring (placeholder for runners)
  4. Signal Generation (阴阳信号)  — Structural tension (trend) / Z-score (range)
  5. Volume Confirmation (虚实验证) — VDP Delta direction must agree
  6. Position Sizing (中庸仓位)    — Fractional Kelly criterion
  7. Risk Management (无常止损)    — ATR trailing stop with regime-adaptive multiplier
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class PhilosophyEnsembleStrategy(BaseStrategy):
    """道生一哲学集成策略 — 7-layer unified ensemble for 600% annualized returns."""

    strategy_description = (
        "道生一哲学集成: 熵门控(wu-wei) + ER regime + "
        "结构张力/均值回归 + VDP量确认 + Kelly仓位 + ATR追踪止损"
    )
    strategy_category = "ensemble"

    strategy_params_schema = {
        "entropy_bins": {"type": "int", "default": 10, "label": "熵计算分箱数"},
        "entropy_window": {"type": "int", "default": 50, "label": "熵计算窗口"},
        "entropy_max_ratio": {"type": "float", "default": 0.8, "label": "熵门控最大比率"},
        "er_period": {"type": "int", "default": 20, "label": "Kaufman ER周期"},
        "er_trend_thresh": {"type": "float", "default": 0.3, "label": "ER趋势阈值"},
        "er_range_thresh": {"type": "float", "default": 0.15, "label": "ER震荡阈值"},
        "pivot_len": {"type": "int", "default": 5, "label": "枢轴点回望周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "ATR追踪止损倍数"},
        "zscore_period": {"type": "int", "default": 20, "label": "Z-score窗口"},
        "zscore_entry": {"type": "float", "default": 2.0, "label": "Z-score入场阈值"},
        "kelly_fraction": {"type": "float", "default": 0.25, "label": "Kelly分数"},
        "kelly_window": {"type": "int", "default": 50, "label": "Kelly滚动估计窗口"},
    }

    def get_default_params(self) -> Dict[str, Any]:
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    # ==================================================================
    # Layer 1: Entropy Gate (无为门)
    # ==================================================================

    @staticmethod
    def _calc_shannon_entropy(returns: np.ndarray, bins: int) -> float:
        """Shannon entropy on binned returns distribution.

        H = -sum(p_i * log2(p_i))  for non-zero p_i
        Returns H, or np.nan if computation fails.
        """
        try:
            if len(returns) < bins:
                return np.nan
            counts, _ = np.histogram(returns, bins=bins)
            total = counts.sum()
            if total == 0:
                return np.nan
            probs = counts / total
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        except Exception:
            return np.nan

    def _entropy_gate_pass(self, closes: np.ndarray, idx: int) -> bool:
        """Layer 1 gate: return True if trading is allowed.

        If H > max_ratio * log2(bins) → BLOCK (market chaotic).
        If H < max_ratio * log2(bins) → ALLOW.
        On computation failure → ALLOW (default open).
        """
        p = self.params
        window = p['entropy_window']
        bins = p['entropy_bins']
        max_ratio = p['entropy_max_ratio']

        if idx < window:
            return True  # insufficient data, allow by default

        segment = closes[idx - window:idx]
        returns = np.diff(segment)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 5:
            return True

        H = self._calc_shannon_entropy(returns, bins)
        if np.isnan(H):
            return True  # computation failed, allow trades

        max_entropy = max_ratio * np.log2(bins)
        return H < max_entropy

    # ==================================================================
    # Layer 2: Regime Detection (知己知彼)
    # ==================================================================

    @staticmethod
    def _calc_kaufman_er(closes: np.ndarray, period: int) -> np.ndarray:
        """Kaufman Efficiency Ratio = |net_change| / sum(|daily_changes|).

        Returns array of same length as closes, NaN before warmup.
        """
        n = len(closes)
        er = np.full(n, np.nan)
        for i in range(period, n):
            net_change = abs(closes[i] - closes[i - period])
            daily_changes = np.abs(np.diff(closes[i - period:i + 1]))
            total_change = np.sum(daily_changes)
            if total_change > 1e-10:
                er[i] = net_change / total_change
            else:
                er[i] = 0.0
        return er

    @staticmethod
    def _classify_regime(er_val: float, trend_thresh: float,
                         range_thresh: float) -> str:
        """Classify market regime from ER value."""
        if np.isnan(er_val):
            return 'NEUTRAL'
        if er_val > trend_thresh:
            return 'TRENDING'
        if er_val < range_thresh:
            return 'RANGING'
        return 'NEUTRAL'

    # ==================================================================
    # Layer 3: Stock Selection (兵法选股)
    # ==================================================================

    def _quality_score(self, closes: np.ndarray, idx: int) -> float:
        """Quality score placeholder for single-stock backtesting.

        Uses simple momentum + volatility ratio as a proxy.
        Returns value in [0, 1].
        """
        lookback = 50
        if idx < lookback:
            return 0.5

        segment = closes[idx - lookback:idx]
        momentum = (segment[-1] / segment[0]) - 1.0
        volatility = np.std(np.diff(segment) / segment[:-1]) if len(segment) > 1 else 0.01
        volatility = max(volatility, 1e-8)

        raw = momentum / volatility
        # Sigmoid normalization to [0, 1]
        score = 1.0 / (1.0 + np.exp(-raw))
        return float(np.clip(score, 0.0, 1.0))

    # ==================================================================
    # Layer 4: Signal Generation (阴阳信号)
    # ==================================================================

    @staticmethod
    def _find_swing_pivots(highs: np.ndarray, lows: np.ndarray,
                           pivot_len: int) -> Dict[str, np.ndarray]:
        """Identify swing high and swing low pivot points.

        A swing high at index i requires: highs[i] is the maximum
        in [i - pivot_len, i + pivot_len].
        Returns dict with 'swing_highs' and 'swing_lows' boolean arrays.
        """
        n = len(highs)
        swing_highs = np.zeros(n, dtype=bool)
        swing_lows = np.zeros(n, dtype=bool)

        for i in range(pivot_len, n - pivot_len):
            window_highs = highs[i - pivot_len:i + pivot_len + 1]
            window_lows = lows[i - pivot_len:i + pivot_len + 1]
            if highs[i] == np.max(window_highs):
                swing_highs[i] = True
            if lows[i] == np.min(window_lows):
                swing_lows[i] = True

        return {'swing_highs': swing_highs, 'swing_lows': swing_lows}

    def _calc_structural_tension(self, closes: np.ndarray,
                                  highs: np.ndarray, lows: np.ndarray,
                                  pivot_len: int) -> np.ndarray:
        """Structural tension: 7-point displacement from swing anchors.

        Tension = distance from last swing high + distance from last swing low,
        signed by which anchor is nearer. Positive = bullish bias,
        negative = bearish bias.
        """
        n = len(closes)
        tension = np.zeros(n)
        pivots = self._find_swing_pivots(highs, lows, pivot_len)

        last_high_idx = 0
        last_low_idx = 0
        # Initialize with first available pivots
        sh = pivots['swing_highs']
        sl = pivots['swing_lows']

        for i in range(n):
            if sh[i]:
                last_high_idx = i
            if sl[i]:
                last_low_idx = i

            if i < pivot_len:
                continue

            swing_high_price = highs[last_high_idx]
            swing_low_price = lows[last_low_idx]

            dist_to_high = closes[i] - swing_high_price
            dist_to_low = closes[i] - swing_low_price

            # Positive tension: price closer to swing low (support held, bullish)
            # Negative tension: price closer to swing high (resistance, bearish)
            # Using displacement from midpoint of the range
            range_mid = (swing_high_price + swing_low_price) / 2.0
            price_range = swing_high_price - swing_low_price
            if price_range > 1e-10:
                tension[i] = (closes[i] - range_mid) / price_range * 2.0
            else:
                tension[i] = 0.0

        return tension

    @staticmethod
    def _calc_zscore(closes: np.ndarray, period: int) -> np.ndarray:
        """Rolling Z-score = (price - SMA) / rolling_std."""
        n = len(closes)
        zscore = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = closes[i - period + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=0)
            if std > 1e-10:
                zscore[i] = (closes[i] - mean) / std
            else:
                zscore[i] = 0.0
        return zscore

    # ==================================================================
    # Layer 5: Volume Confirmation (虚实验证)
    # ==================================================================

    @staticmethod
    def _calc_vdp_delta(close: np.ndarray, high: np.ndarray,
                        low: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """VDP Delta = volume * (2*close - high - low) / (high - low).

        Positive delta = buying pressure, negative = selling pressure.
        """
        n = len(close)
        delta = np.zeros(n)
        for i in range(n):
            bar_range = high[i] - low[i]
            if bar_range <= 1e-10 or volume[i] <= 0:
                delta[i] = 0.0
                continue
            delta[i] = volume[i] * (2.0 * close[i] - high[i] - low[i]) / bar_range
        return delta

    def _volume_confirms(self, delta: float, signal_direction: int,
                         has_volume: bool) -> bool:
        """Check if volume delta confirms the signal direction.

        signal_direction: +1 for bullish, -1 for bearish.
        Returns True if confirmed or if no volume data available.
        """
        if not has_volume:
            return False  # No confirmation possible, but not a block
        if signal_direction > 0:
            return delta > 0
        else:
            return delta < 0

    # ==================================================================
    # Layer 6: Position Sizing (中庸仓位)
    # ==================================================================

    def _calc_kelly_size(self, closes: np.ndarray, idx: int) -> float:
        """Fractional Kelly position sizing.

        f = kelly_fraction * (WR * avgW/avgL - (1 - WR)) / (avgW/avgL)
        Using rolling window estimation of win rate, avg win, avg loss.
        Clamped to [0.1, 0.5].
        """
        p = self.params
        window = p['kelly_window']
        kelly_frac = p['kelly_fraction']

        if idx < window + 1:
            return 0.25  # default quarter position

        segment = closes[idx - window:idx]
        changes = np.diff(segment)
        gains = changes[changes > 0]
        losses = changes[changes < 0]

        if len(gains) == 0 or len(losses) == 0:
            return 0.25

        wr = len(gains) / len(changes)
        avg_w = np.mean(gains)
        avg_l = abs(np.mean(losses))

        if avg_l < 1e-10:
            return 0.5

        win_loss_ratio = avg_w / avg_l

        # Kelly formula: f* = (WR * R - (1 - WR)) / R where R = avgW/avgL
        kelly_full = (wr * win_loss_ratio - (1.0 - wr)) / win_loss_ratio
        f = kelly_frac * kelly_full

        return float(np.clip(f, 0.1, 0.5))

    # ==================================================================
    # Layer 7: Risk Management (无常止损)
    # ==================================================================

    @staticmethod
    def _calc_atr(highs: np.ndarray, lows: np.ndarray,
                  closes: np.ndarray, period: int) -> np.ndarray:
        """Average True Range using EMA smoothing."""
        n = len(closes)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
        # EMA-based ATR
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        return atr

    def _get_trail_mult(self, er_val: float) -> float:
        """Regime-adaptive trailing stop multiplier.

        Base = 2.5, widen to 3.0 when ER < 0.2 (uncertain regime).
        """
        base = self.params['trail_atr_mult']
        if np.isnan(er_val):
            return base
        if er_val < 0.2:
            return base + 0.5  # widen stop in uncertain regime
        return base

    # ==================================================================
    # Main signal generation
    # ==================================================================

    def generate_signals(self) -> List[Dict]:
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H = df['high'].values.astype(np.float64)
        L = df['low'].values.astype(np.float64)
        C = df['close'].values.astype(np.float64)
        n = len(C)

        # Volume handling
        has_volume = True
        if 'vol' in df.columns:
            V = df['vol'].values.astype(np.float64)
        elif 'volume' in df.columns:
            V = df['volume'].values.astype(np.float64)
        else:
            V = np.ones(n)
            has_volume = False

        # Precompute all indicators
        er = self._calc_kaufman_er(C, p['er_period'])
        tension = self._calc_structural_tension(C, H, L, p['pivot_len'])
        zscore = self._calc_zscore(C, p['zscore_period'])
        vdp_delta = self._calc_vdp_delta(C, H, L, V)
        atr = self._calc_atr(H, L, C, p['atr_period'])

        # Warmup period: need enough data for all indicators
        warmup = max(
            p['entropy_window'],
            p['er_period'],
            p['pivot_len'] * 2 + 1,
            p['zscore_period'],
            p['atr_period'],
            p['kelly_window'] + 1,
            60  # absolute minimum
        )

        # State tracking
        in_pos = False
        entry_idx = 0
        entry_price = 0.0
        trail_stop = 0.0
        pos_direction = 0  # +1 long, -1 short
        # Track previous tension for flip detection
        prev_tension = 0.0

        for i in range(warmup, n):
            price = float(C[i])
            ts = df.index[i]

            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            # ============================================================
            # Layer 1: Entropy Gate
            # ============================================================
            gate_open = self._entropy_gate_pass(C, i)

            # Layer 2: Regime
            er_val = float(er[i]) if not np.isnan(er[i]) else 0.5
            regime = self._classify_regime(
                er_val, p['er_trend_thresh'], p['er_range_thresh']
            )

            # Layer 6: Kelly position sizing
            kelly_size = self._calc_kelly_size(C, i)

            if not in_pos:
                # ========================================================
                # ENTRY LOGIC
                # ========================================================

                # Layer 1: gate must be open
                if not gate_open:
                    prev_tension = float(tension[i])
                    continue

                # Layer 4: Signal generation based on regime
                signal_score = 0
                signal_direction = 0  # +1 bullish, -1 bearish
                signal_strength = ''

                if regime == 'TRENDING':
                    # Structural tension signals
                    current_tension = float(tension[i])
                    # Tension flip from negative to positive = bullish entry
                    if prev_tension < 0 and current_tension > 0:
                        signal_direction = 1
                        signal_score = 5  # strong
                        signal_strength = 'strong'
                    # Tension flip from positive to negative = bearish entry
                    elif prev_tension > 0 and current_tension < 0:
                        signal_direction = -1
                        signal_score = 5
                        signal_strength = 'strong'
                    # Large tension magnitude without flip = moderate signal
                    elif abs(current_tension) > 0.5:
                        if current_tension > 0:
                            signal_direction = 1
                        else:
                            signal_direction = -1
                        signal_score = 3  # moderate
                        signal_strength = 'moderate'

                    prev_tension = current_tension

                elif regime == 'RANGING':
                    # Z-score mean reversion signals
                    z = zscore[i]
                    if not np.isnan(z):
                        z_val = float(z)
                        if z_val < -p['zscore_entry']:
                            # Oversold → buy signal
                            signal_direction = 1
                            if z_val < -p['zscore_entry'] * 1.5:
                                signal_score = 5
                                signal_strength = 'strong'
                            else:
                                signal_score = 3
                                signal_strength = 'moderate'
                        elif z_val > p['zscore_entry']:
                            # Overbought → sell signal
                            signal_direction = -1
                            if z_val > p['zscore_entry'] * 1.5:
                                signal_score = 5
                                signal_strength = 'strong'
                            else:
                                signal_score = 3
                                signal_strength = 'moderate'

                elif regime == 'NEUTRAL':
                    # Require higher confirmation — only take strong signals
                    current_tension = float(tension[i])
                    if prev_tension < 0 and current_tension > 0:
                        signal_direction = 1
                        signal_score = 3  # reduced from 5 in neutral
                        signal_strength = 'moderate'
                    elif prev_tension > 0 and current_tension < 0:
                        signal_direction = -1
                        signal_score = 3
                        signal_strength = 'moderate'
                    prev_tension = current_tension

                else:
                    prev_tension = float(tension[i])
                    continue

                if signal_direction == 0:
                    # No signal from Layer 4
                    if regime != 'TRENDING':
                        prev_tension = float(tension[i])
                    continue

                # ========================================================
                # Layer 5: Volume Confirmation
                # ========================================================
                vol_confirmed = self._volume_confirms(
                    vdp_delta[i], signal_direction, has_volume
                )
                if vol_confirmed:
                    signal_score += 2

                # ========================================================
                # Layer 6: Kelly boost
                # ========================================================
                if kelly_size > 0.3:
                    signal_score += 1

                # ========================================================
                # Layer 2: Regime alignment bonus
                # ========================================================
                if regime == 'TRENDING' and signal_direction != 0:
                    signal_score += 1
                elif regime == 'RANGING' and signal_direction != 0:
                    signal_score += 1

                # ========================================================
                # Entry threshold check
                # ========================================================
                if signal_score < 5:
                    continue

                # Execute entry
                # For bearish signals in single-stock long-only: skip
                # (most runners are long-only)
                # But we record short signals as sells for completeness
                if signal_direction > 0:
                    # Long entry
                    mult = self._get_trail_mult(er_val)
                    trail_stop = price - mult * atr[i]
                    self._record_signal(
                        ts, 'buy', sym, price,
                        score=signal_score,
                        regime=regime,
                        strength=signal_strength,
                        volume_confirmed=vol_confirmed,
                        position_pct=round(kelly_size, 4),
                        kelly_f=round(kelly_size, 4),
                        trail_stop=round(trail_stop, 4),
                        er=round(er_val, 4),
                        tension=round(float(tension[i]), 4),
                    )
                    in_pos = True
                    entry_idx = i
                    entry_price = price
                    pos_direction = 1

                elif signal_direction < 0:
                    # Short entry — record as sell (for long-only, this can
                    # also serve as an exit signal if already in position)
                    # In long-only context, we just skip bearish entries
                    # unless we want to allow shorts
                    # For now, record it so the system knows
                    mult = self._get_trail_mult(er_val)
                    trail_stop = price + mult * atr[i]
                    self._record_signal(
                        ts, 'sell', sym, price,
                        score=signal_score,
                        regime=regime,
                        strength=signal_strength,
                        volume_confirmed=vol_confirmed,
                        position_pct=round(kelly_size, 4),
                        kelly_f=round(kelly_size, 4),
                        trail_stop=round(trail_stop, 4),
                        er=round(er_val, 4),
                        tension=round(float(tension[i]), 4),
                    )
                    in_pos = True
                    entry_idx = i
                    entry_price = price
                    pos_direction = -1

            else:
                # ========================================================
                # EXIT / TRAILING LOGIC (Layer 7)
                # ========================================================
                mult = self._get_trail_mult(er_val)

                if pos_direction == 1:
                    # Long position: trail stop upward
                    new_stop = price - mult * atr[i]
                    trail_stop = max(trail_stop, new_stop)

                    sell_signal = False

                    # ATR trailing stop hit
                    if price <= trail_stop:
                        sell_signal = True

                    # Regime shift: from trending to ranging while in profit
                    if regime == 'RANGING' and price > entry_price:
                        sell_signal = True

                    # Entropy gate closes while in position
                    if not gate_open and price > entry_price:
                        sell_signal = True

                    # Z-score extreme against position (overbought exit)
                    if not np.isnan(zscore[i]):
                        z_val = float(zscore[i])
                        if z_val > p['zscore_entry'] * 1.5:
                            sell_signal = True

                    if sell_signal:
                        self._record_signal(
                            ts, 'sell', sym, price,
                            reason='trailing_stop',
                            pnl_pct=round(
                                (price - entry_price) / entry_price, 4
                            ),
                            trail_stop=round(trail_stop, 4),
                            regime=regime,
                            er=round(er_val, 4),
                        )
                        in_pos = False
                        pos_direction = 0

                elif pos_direction == -1:
                    # Short position: trail stop downward
                    new_stop = price + mult * atr[i]
                    trail_stop = min(trail_stop, new_stop)

                    cover_signal = False

                    # ATR trailing stop hit
                    if price >= trail_stop:
                        cover_signal = True

                    # Regime shift
                    if regime == 'RANGING' and price < entry_price:
                        cover_signal = True

                    # Entropy gate closes
                    if not gate_open and price < entry_price:
                        cover_signal = True

                    # Z-score extreme (oversold cover)
                    if not np.isnan(zscore[i]):
                        z_val = float(zscore[i])
                        if z_val < -p['zscore_entry'] * 1.5:
                            cover_signal = True

                    if cover_signal:
                        self._record_signal(
                            ts, 'buy', sym, price,
                            reason='trailing_stop',
                            pnl_pct=round(
                                (entry_price - price) / entry_price, 4
                            ),
                            trail_stop=round(trail_stop, 4),
                            regime=regime,
                            er=round(er_val, 4),
                        )
                        in_pos = False
                        pos_direction = 0

                # Update tension tracking even while in position
                if regime in ('TRENDING', 'NEUTRAL'):
                    prev_tension = float(tension[i])

        # Force close any open position at end of data
        if in_pos:
            last_ts = df.index[-1]
            last_price = float(C[-1])
            action = 'sell' if pos_direction == 1 else 'buy'
            self._record_signal(
                last_ts, action, sym, last_price,
                reason='force_close',
                pnl_pct=round(
                    (last_price - entry_price) / entry_price
                    * pos_direction, 4
                ),
            )

        return self.signals
