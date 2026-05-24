"""
Multi-Timeframe Momentum Alignment Strategy (多时间框架动量对齐策略)
==================================================================
Strategy Philosophy: Align momentum across multiple timeframes for high-conviction signals.
When all timeframes agree, act decisively; when they disagree, do nothing (无为).

Core Logic:
  1. Compute rank-transformed momentum at 3 timeframes: 5d, 10d, 20d
  2. Cross-sectionally rank each momentum within all stocks
  3. Score = +3 if ALL 3 timeframes have positive rank (>60)
  4. Score = -3 if ALL 3 timeframes have negative rank (<40)
  5. Score = +2 if 2 of 3 are bullish, etc.
  6. Add VDP delta confirmation (+1 if aligned)
  7. Add KER filter: skip if KER < 0.15
  8. ATR trailing stop for exits
  9. Position size proportional to score

Philosophy: 知行合一 — knowledge (rank alignment) and action (signal) must align.
When timeframes agree strongly, act with conviction; otherwise, wait.
Technical Indicators: Rank Momentum, VDP Delta, KER, ATR
Category: momentum
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class MultiTFMomentumStrategy(BaseStrategy):
    """Multi-Timeframe Momentum Alignment Strategy"""

    strategy_description = (
        "MultiTFMomentum: Cross-sectional rank momentum alignment across "
        "5d/10d/20d timeframes with VDP confirmation and KER filter"
    )
    strategy_category = "momentum"
    strategy_params_schema = {
        "mom_periods": {"type": "str", "default": "5,10,20", "label": "Momentum periods"},
        "rank_window": {"type": "int", "default": 60, "label": "Rank lookback window"},
        "ker_period": {"type": "int", "default": 20, "label": "KER period"},
        "ker_threshold": {"type": "float", "default": 0.15, "label": "KER threshold"},
        "vdp_ema_period": {"type": "int", "default": 14, "label": "VDP EMA period"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "Trail stop ATR mult"},
        "max_hold": {"type": "int", "default": 60, "label": "Max holding days"},
        "score_threshold": {"type": "int", "default": 2, "label": "Signal score threshold"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.mom_periods = [int(x) for x in str(p.get('mom_periods', '5,10,20')).split(',')]
        self.rank_window = p.get('rank_window', 60)
        self.ker_period = p.get('ker_period', 20)
        self.ker_threshold = p.get('ker_threshold', 0.15)
        self.vdp_ema_period = p.get('vdp_ema_period', 14)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 60)
        self.score_threshold = p.get('score_threshold', 2)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'mom_periods': '5,10,20', 'rank_window': 60,
            'ker_period': 20, 'ker_threshold': 0.15,
            'vdp_ema_period': 14, 'atr_period': 14,
            'hold_min': 3, 'trail_atr_mult': 2.5,
            'max_hold': 60, 'score_threshold': 2,
        }

    def generate_signals(self) -> List[Dict]:
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []

        # Track holdings
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
                # Find best symbol to trade
                best_score = 0
                best_sym = None
                best_dir = 0

                # Cross-sectional ranking for all symbols
                symbol_scores = {}
                for symbol in symbols:
                    hist = data[(data['symbol'] == symbol) & (data.index <= current_time)]
                    result = self._evaluate(symbol, hist)
                    if result is not None:
                        score, direction, _ = result
                        symbol_scores[symbol] = (score, direction)

                # Select symbol with highest absolute score
                for symbol, (score, direction) in symbol_scores.items():
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = symbol
                        best_dir = direction

                # Enter position if score meets threshold
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
                # Exit logic
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # Update high/low water marks for trailing stop
                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) if low_water < float('inf') else current_price

                # Check exit conditions
                should_exit = False

                # 1. Maximum holding period
                if days_held >= self.max_hold:
                    should_exit = True

                # 2. ATR trailing stop (after minimum holding period)
                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding) & (data.index <= current_time)]
                    atr_val = self._calc_atr(hist)

                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # 3. Signal reversal check
                    if not should_exit:
                        result = self._evaluate(current_holding, hist)
                        if result is not None:
                            score, direction, _ = result
                            # Exit if signal reverses strongly
                            if position_dir == 1 and direction == -1 and score < -self.score_threshold:
                                should_exit = True
                            elif position_dir == -1 and direction == 1 and score > self.score_threshold:
                                should_exit = True

                # Execute exit
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

        print(f"MultiTFMomentum: generated {len(self.signals)} signals")
        return self.signals

    def _evaluate(self, symbol: str, data: pd.DataFrame):
        """Evaluate momentum alignment score for a symbol."""
        min_len = max(max(self.mom_periods) + self.rank_window + 10,
                     self.ker_period + 10,
                     self.atr_period + 10)
        if len(data) < min_len:
            return None

        close = data['close'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Step 1: Compute momentum for each timeframe
        mom_ranks = []
        momentum_values = []

        for period in self.mom_periods:
            if len(close) > period + self.rank_window:
                # Calculate momentum
                mom = (close[-1] - close[-period-1]) / close[-period-1]
                momentum_values.append(mom)

                # Calculate rank within lookback window
                window_moms = []
                for i in range(len(close) - period - self.rank_window, len(close) - period):
                    if i >= 0 and i < len(close) - period:
                        win_mom = (close[i] - close[i-period]) / close[i-period]
                        window_moms.append(win_mom)

                if window_moms:
                    # Cross-sectional rank (proxy using historical momentum values)
                    rank = self._calc_percentile(mom, window_moms)
                    mom_ranks.append(rank)

        if not mom_ranks:
            return None

        # Step 2: Score based on alignment
        score = 0
        bullish_count = sum(1 for rank in mom_ranks if rank > 60)
        bearish_count = sum(1 for rank in mom_ranks if rank < 40)

        # Score: +3 for all bullish, -3 for all bearish
        if bullish_count == len(mom_ranks):
            score = 3
        elif bearish_count == len(mom_ranks):
            score = -3
        # Score: +2 for 2 bullish, -2 for 2 bearish
        elif bullish_count == 2:
            score = 2
        elif bearish_count == 2:
            score = -2
        # Score: +1 for 1 bullish with others neutral, etc.
        elif bullish_count == 1:
            score = 1
        elif bearish_count == 1:
            score = -1

        # Step 3: KER filter
        ker = self._calc_ker(data)
        if ker < self.ker_threshold:
            # Too ranging, reduce score
            score = int(score * 0.5)
            if abs(score) < 1:
                return None

        # Step 4: VDP delta confirmation
        vdp_delta = self._calc_vdp_delta(data)
        if not np.isnan(vdp_delta):
            if score > 0 and vdp_delta > 0:
                score += 1  # Bullish momentum with positive volume delta
            elif score < 0 and vdp_delta < 0:
                score += 1  # Bearish momentum with negative volume delta
            else:
                score -= 1  # Mismatch between momentum and volume

        # Direction determination
        direction = 1 if score > 0 else -1 if score < 0 else 0
        if direction == 0:
            return None

        return score, direction, atr

    def _calc_percentile(self, value: float, data: List[float]) -> float:
        """Calculate percentile rank of a value in a dataset."""
        if not data:
            return 50.0
        data = [x for x in data if not np.isnan(x)]
        if not data:
            return 50.0
        rank = (sum(1 for x in data if x <= value) / len(data)) * 100
        return rank

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

    def _calc_vdp_delta(self, data):
        """Volume Delta Pressure (VDP) cumulative delta."""
        vol_col = None
        for vc in ['vol', 'volume', 'Volume']:
            if vc in data.columns:
                vol_col = vc
                break
        if vol_col is None:
            return np.nan

        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        vol = data[vol_col].values.astype(float)
        vol = np.nan_to_num(vol, nan=0.0)

        # Calculate delta: volume * (2*close - high - low) / (high - low)
        deltas = np.zeros_like(close)
        for i in range(len(close)):
            if i < len(high) and i < len(low):
                hl_range = high[i] - low[i]
                if hl_range > 1e-10:
                    deltas[i] = vol[i] * (2 * close[i] - high[i] - low[i]) / hl_range
                else:
                    deltas[i] = 0.0

        # Apply EMA smoothing
        period = self.vdp_ema_period
        k = 2.0 / (period + 1)
        ema = np.zeros_like(deltas)
        ema[0] = deltas[0]
        for i in range(1, len(deltas)):
            ema[i] = deltas[i] * k + ema[i - 1] * (1 - k)

        return float(ema[-1])

    def _calc_atr(self, data):
        """Average True Range calculation."""
        if len(data) < self.atr_period + 1:
            return 0.0

        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        n = len(high)

        tr_list = []
        for i in range(max(1, n - self.atr_period), n):
            tr = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
            tr_list.append(tr)

        return float(np.mean(tr_list)) if tr_list else 0.0

    def screen(self):
        """Quick screen based on latest data."""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'
        sym = data['symbol'].iloc[0]

        min_len = max(max(self.mom_periods) + self.rank_window + 10,
                     self.ker_period + 10,
                     self.atr_period + 10)
        if len(data) < min_len:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}

        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': 'no signal', 'price': price}

        score, direction, _ = result

        if abs(score) >= self.score_threshold:
            return {'action': 'buy' if direction == 1 else 'sell',
                    'reason': f'score={score}', 'price': price}

        return {'action': 'hold', 'reason': f'score={score}', 'price': price}