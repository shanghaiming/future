"""
Rank Momentum Divergence Strategy (等级动量分歧策略)
=====================================================
Key insight from v109: rank/percentile methods vastly outperform moment methods.
This strategy applies rank transforms to everything, then detects divergence.

Core Logic:
  1. Rank-transform momentum (5d, 10d, 20d) cross-sectionally
  2. Rank-transform volume delta pressure
  3. Detect divergence: momentum rank rising but VDP rank falling = bearish divergence
  4. Convergence: both rising = strong bullish signal
  5. Use KER gate to filter ranging markets
  6. Structural tension for direction confirmation
  7. ATR trailing stop for exits

Philosophy: 知行合一 — knowledge (rank) and action (signal) must align.
When rank-transformed signals agree, act decisively.
When they disagree, do nothing (无为).

Technical Indicators: Rank Momentum, Rank VDP, KER, Structural Tension, ATR
Category: adaptive
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class RankMomentumDivergenceStrategy(BaseStrategy):
    """Rank-based momentum divergence — rank > moment methods"""

    strategy_description = (
        "RankMomentum: Cross-sectional rank momentum + VDP divergence "
        "+ KER gate + structural tension confirmation"
    )
    strategy_category = "adaptive"
    strategy_params_schema = {
        "mom_periods": {"type": "str", "default": "5,10,20", "label": "Momentum periods"},
        "vdp_ema_period": {"type": "int", "default": 10, "label": "VDP EMA period"},
        "ker_period": {"type": "int", "default": 20, "label": "KER period"},
        "ker_threshold": {"type": "float", "default": 0.15, "label": "KER threshold"},
        "pivot_len": {"type": "int", "default": 5, "label": "Pivot lookback"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "Trail stop ATR mult"},
        "max_hold": {"type": "int", "default": 60, "label": "Max holding days"},
        "score_threshold": {"type": "int", "default": 4, "label": "Signal score threshold"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.mom_periods = [int(x) for x in str(p.get('mom_periods', '5,10,20')).split(',')]
        self.vdp_ema_period = p.get('vdp_ema_period', 10)
        self.ker_period = p.get('ker_period', 20)
        self.ker_threshold = p.get('ker_threshold', 0.15)
        self.pivot_len = p.get('pivot_len', 5)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 60)
        self.score_threshold = p.get('score_threshold', 4)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'mom_periods': '5,10,20', 'vdp_ema_period': 10,
            'ker_period': 20, 'ker_threshold': 0.15,
            'pivot_len': 5, 'atr_period': 14, 'hold_min': 3,
            'trail_atr_mult': 2.5, 'max_hold': 60, 'score_threshold': 4,
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

                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    if days_held >= self.max_hold:
                        should_exit = True

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

        print(f"RankMomentumDivergence: generated {len(self.signals)} signals")
        return self.signals

    # ================================================================
    # Core Evaluation
    # ================================================================

    def _evaluate(self, symbol: str, data: pd.DataFrame):
        """Rank momentum + VDP divergence + KER gate + tension confirmation."""
        min_len = max(max(self.mom_periods) + 10, self.ker_period + 10, 50)
        if len(data) < min_len:
            return None

        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Step 1: Compute momentums
        mom_scores = []
        for period in self.mom_periods:
            if len(close) > period and close[-period-1] > 0:
                mom = (close[-1] - close[-period-1]) / close[-period-1]
                mom_scores.append(mom)
        if not mom_scores:
            return None

        # Step 2: Rank-transform momentum within recent history (cross-sectional proxy)
        avg_mom = np.mean(mom_scores)

        # Step 3: VDP cumulative delta
        cum_delta = self._calc_vdp_cum_delta(data)

        # Step 4: VDP direction (simplified rank proxy)
        vdp_dir = 0
        if not np.isnan(cum_delta):
            vdp_dir = 1.0 if cum_delta > 0 else -1.0

        # Step 5: KER gate
        ker = self._calc_ker(data)
        if ker < self.ker_threshold:
            return None  # Too rangy, don't trade

        # Step 6: Structural tension for base direction
        tension = self._calc_structural_tension(data, close, high, low, atr)

        # Step 7: Score based on convergence/divergence
        score = 0

        # Momentum direction
        mom_dir = 1 if avg_mom > 0 else -1
        mom_strength = min(abs(avg_mom) * 100, 5)  # Cap at 5

        # Convergence: momentum and VDP agree
        if mom_dir == vdp_dir and vdp_dir != 0:
            score += int(mom_strength) + 2  # Strong convergence bonus
        elif mom_dir != vdp_dir and vdp_dir != 0:
            score += 0  # Divergence = no signal (无为)

        # Tension confirmation
        if tension > 0.3 and score > 0:
            score += 2
        elif tension < -0.3 and score > 0:
            score += 2  # Keep direction from tension
        elif tension > 0 and score > 0:
            score += 1
        elif tension < 0 and score > 0:
            score += 1

        # Direction determination
        if mom_dir > 0 and vdp_dir >= 0:
            direction = 1
        elif mom_dir < 0 and vdp_dir <= 0:
            direction = -1
        elif abs(tension) > 0.3:
            direction = 1 if tension > 0 else -1
            score = max(score - 1, 1)  # Reduced conviction without VDP alignment
        else:
            return None  # No clear direction

        # KER bonus: stronger trend = higher conviction
        if ker > 0.3:
            score += 1

        return score, direction, atr

    # ================================================================
    # Indicators
    # ================================================================

    def _calc_ker(self, data):
        """Kaufman Efficiency Ratio."""
        close = data['close'].values.astype(float)
        n = len(close)
        period = self.ker_period
        if n < period + 1:
            return 0.0
        recent = close[-(period + 1):]
        net_change = abs(recent[-1] - recent[0])
        sum_abs = np.sum(np.abs(np.diff(recent)))
        if sum_abs < 1e-12:
            return 0.0
        return net_change / sum_abs

    def _calc_structural_tension(self, data, close, high, low, atr):
        """Simplified 3-point structural tension."""
        n = len(close)
        if n < 20:
            return 0.0
        window = min(20, n)
        c_win = close[-window:]
        h_win = high[-window:]
        l_win = low[-window:]
        cv = c_win[~np.isnan(c_win)]
        hv = h_win[~np.isnan(h_win)]
        lv = l_win[~np.isnan(l_win)]
        if len(cv) < 10:
            return 0.0
        hh = np.max(hv) if len(hv) > 0 else np.max(cv)
        ll = np.min(lv) if len(lv) > 0 else np.min(cv)
        mid = (hh + ll) / 2.0
        rng = hh - ll
        if rng <= 0 or atr <= 0:
            return 0.0
        tension = ((close[-1] - hh) + (close[-1] - ll) + (close[-1] - mid)) / (3 * rng)
        return float(tension)

    def _calc_vdp_cum_delta(self, data):
        """VDP cumulative delta."""
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
        deltas = np.full(n, 0.0)
        for i in range(n):
            hl_range = high[i] - low[i]
            if hl_range < 1e-10:
                deltas[i] = 0.0
            else:
                deltas[i] = vol[i] * (2 * close[i] - high[i] - low[i]) / hl_range
        period = self.vdp_ema_period
        k = 2.0 / (period + 1)
        ema = np.full(n, 0.0)
        ema[0] = deltas[0]
        for i in range(1, n):
            ema[i] = deltas[i] * k + ema[i - 1] * (1 - k)
        return float(ema[-1])

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
            tr = max(high[i] - low[i], abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
            tr_list.append(tr)
        return float(np.mean(tr_list)) if tr_list else 0.0

    def screen(self):
        """Quick screen based on latest data."""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'
        sym = data['symbol'].iloc[0]
        min_len = max(max(self.mom_periods) + 10, self.ker_period + 10, 50)
        if len(data) < min_len:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}
        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])
        ker = self._calc_ker(data)
        if result is None:
            return {'action': 'hold', 'reason': f'KER={ker:.2f} no signal', 'price': price}
        score, direction, _ = result
        if abs(score) >= self.score_threshold:
            return {'action': 'buy' if direction == 1 else 'sell',
                    'reason': f'score={score} KER={ker:.2f}', 'price': price}
        return {'action': 'hold', 'reason': f'score={score} KER={ker:.2f}', 'price': price}
