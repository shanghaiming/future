"""
EVT止损策略 (Extreme Value Theory Stop Loss Strategy)
======================================================
基于极值理论(EVT)的统计尾部风险止损 + 结构张力入场。

哲学: "一切有为法，如梦幻泡影" — 所有趋势终将结束;
     用统计尾部风险设定止损，而非主观固定百分比。

核心逻辑:
  1. 入场: 结构张力(7点位移) 从摆动锚点计算，
     张力上穿0线=看多, 下穿0线=看空
  2. EVT止损: 对负收益拟合广义帕累托分布(GPD)，
     用95%分位数计算止损距离，在波动期产生比2sigma宽20-30%的止损
  3. 追踪止损: 价格上涨时止损上移(从不下移)
  4. 最大持仓: 60天

技术指标: Swing Points, Structural Tension, GPD Tail Fit, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EVTStopStrategy(BaseStrategy):
    """EVT止损策略 — 极值理论止损 + 结构张力入场 (无常)"""

    strategy_description = (
        "EVT止损: 极值理论GPD尾部止损 + 7点结构张力入场 "
        "— 一切有为法，如梦幻泡影"
    )
    strategy_category = "risk"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "lookback": {"type": "int", "default": 50, "label": "结构回溯"},
        "max_hold": {"type": "int", "default": 60, "label": "最大持仓天数"},
        "evt_percentile": {"type": "float", "default": 5.0, "label": "EVT尾部百分位"},
        "evt_confidence": {"type": "float", "default": 0.95, "label": "EVT置信度"},
        "score_threshold": {"type": "int", "default": 4, "label": "入场评分阈值"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.pivot_len = self.params.get('pivot_len', 5)
        self.atr_period = self.params.get('atr_period', 14)
        self.lookback = self.params.get('lookback', 50)
        self.max_hold = self.params.get('max_hold', 60)
        self.evt_percentile = self.params.get('evt_percentile', 5.0)
        self.evt_confidence = self.params.get('evt_confidence', 0.95)
        self.score_threshold = self.params.get('score_threshold', 4)

    def get_default_params(self):
        return {
            'pivot_len': 5,
            'atr_period': 14,
            'lookback': 50,
            'max_hold': 60,
            'evt_percentile': 5.0,
            'evt_confidence': 0.95,
            'score_threshold': 4,
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0
        low_water = float('inf')
        stop_price = 0.0

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0
                best_price = 0.0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate_entry(hist)
                    if result is None:
                        continue
                    score, direction, price = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction
                        best_price = price

                if best_sym and abs(best_score) >= self.score_threshold:
                    # Compute EVT stop at entry
                    hist = data[
                        (data['symbol'] == best_sym) & (data.index < current_time)
                    ]
                    stop_distance = self._compute_evt_stop_distance(hist)

                    if best_dir == 1:
                        stop_price = best_price - stop_distance
                        self._record_signal(
                            current_time, 'buy', best_sym, best_price,
                            score=best_score,
                            stop_price=round(stop_price, 4),
                            stop_distance=round(stop_distance, 4),
                        )
                        position_dir = 1
                    else:
                        stop_price = best_price + stop_distance
                        self._record_signal(
                            current_time, 'sell', best_sym, best_price,
                            score=best_score,
                            stop_price=round(stop_price, 4),
                            stop_distance=round(stop_distance, 4),
                        )
                        position_dir = -1

                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0
                    low_water = float('inf')

            else:
                days_held = len(
                    [t for t in unique_times if buy_time < t <= current_time]
                )
                bar_data = current_bars[
                    current_bars['symbol'] == current_holding
                ]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # Track high / low water marks
                if position_dir == 1:
                    high_water = (
                        max(high_water, current_price)
                        if high_water > 0
                        else current_price
                    )
                else:
                    low_water = (
                        min(low_water, current_price)
                        if low_water < float('inf')
                        else current_price
                    )

                should_exit = False

                # --- Trailing stop: ratchet up for longs, down for shorts ---
                hist = data[
                    (data['symbol'] == current_holding)
                    & (data.index < current_time)
                ]
                new_stop_dist = self._compute_evt_stop_distance(hist)

                if position_dir == 1 and high_water > 0:
                    candidate_stop = high_water - new_stop_dist
                    if candidate_stop > stop_price:
                        stop_price = candidate_stop  # trail up, never down
                    if current_price <= stop_price:
                        should_exit = True
                elif position_dir == -1 and low_water < float('inf'):
                    candidate_stop = low_water + new_stop_dist
                    if candidate_stop < stop_price or stop_price == 0.0:
                        stop_price = candidate_stop  # trail down, never up
                    if current_price >= stop_price:
                        should_exit = True

                # --- Max hold ---
                if days_held >= self.max_hold:
                    should_exit = True

                # --- Tension reversal exit ---
                if not should_exit and days_held >= 3:
                    result = self._evaluate_entry(hist)
                    if result is not None:
                        score, direction, _ = result
                        if position_dir == 1 and direction == -1 and score < -self.score_threshold:
                            should_exit = True
                        elif position_dir == -1 and direction == 1 and score > self.score_threshold:
                            should_exit = True

                if should_exit:
                    if position_dir == 1:
                        self._record_signal(
                            current_time, 'sell', current_holding, current_price,
                            reason='evt_stop',
                        )
                    else:
                        self._record_signal(
                            current_time, 'buy', current_holding, current_price,
                            reason='evt_stop',
                        )
                    current_holding = None
                    buy_time = None
                    position_dir = 0
                    high_water = 0.0
                    low_water = float('inf')
                    stop_price = 0.0

        return self.signals

    # ------------------------------------------------------------------
    # Entry evaluation — structural tension scoring
    # ------------------------------------------------------------------

    def _evaluate_entry(self, data):
        """Return (score, direction, current_price) or None."""
        min_bars = self.pivot_len * 2 + 20
        if len(data) < min_bars:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        swing_highs, swing_lows = self._find_pivots(high, low, n)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # 7 structural reference points
        prev_sh = swing_highs[-2][1]
        curr_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1]
        curr_sl = swing_lows[-1][1]
        inter_high = max(curr_sh, prev_sh)
        inter_low = min(curr_sl, prev_sl)
        inter_mid = (inter_high + inter_low) / 2.0

        current_price = close[-1]
        ref_points = [prev_sh, curr_sh, prev_sl, curr_sl, inter_high, inter_low, inter_mid]

        displacements = [(current_price - rp) / atr for rp in ref_points]
        tension = float(np.mean(displacements))

        score = 0

        # --- Tension flip (zero-line cross) ---
        if n >= 2:
            prev_price = close[-2]
            prev_disp = [(prev_price - rp) / atr for rp in ref_points]
            prev_tension = float(np.mean(prev_disp))

            if prev_tension <= 0 and tension > 0:
                score += 4  # Bullish flip
            elif prev_tension >= 0 and tension < 0:
                score -= 4  # Bearish flip

        # --- Tension sustained ---
        if tension > 0.5:
            score += 2
        elif tension > 0.2:
            score += 1
        elif tension < -0.5:
            score -= 2
        elif tension < -0.2:
            score -= 1

        # --- Volume surge ---
        vol_col = self._volume_col(data)
        if vol_col is not None and n >= 2:
            vol = data[vol_col].values
            if n >= 20:
                avg_vol = np.mean(vol[-20:])
                if avg_vol > 0 and vol[-1] > 1.5 * avg_vol:
                    if tension > 0:
                        score += 2
                    elif tension < 0:
                        score -= 2

        direction = 1 if score > 0 else -1
        return score, direction, current_price

    # ------------------------------------------------------------------
    # EVT / GPD stop distance
    # ------------------------------------------------------------------

    def _compute_evt_stop_distance(self, data):
        """
        Fit a Generalized Pareto Distribution to the tail of negative returns
        and return the stop distance for the configured confidence level.

        GPD: P(X > x | X > u) = (1 + xi*(x-u)/sigma)^(-1/xi)
        95% quantile stop: sigma/xi * ((1/(1-confidence))^xi - 1)

        Uses method-of-moments for xi and sigma estimation.
        Falls back to a 2*ATR stop if insufficient data.
        """
        close = data['close'].values
        n = len(close)
        atr = self._calc_atr(data)

        if n < 30 or atr <= 0:
            return 2.0 * atr if atr > 0 else 0.0

        # Compute log returns
        returns = np.diff(np.log(close))

        # Collect losses (negative returns)
        losses = -returns[returns < 0]

        if len(losses) < 10:
            return 2.0 * atr

        # Threshold at the configured percentile (e.g., 5th percentile of losses)
        # We take the largest losses (those above the threshold)
        threshold = np.percentile(losses, 100 - self.evt_percentile)
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 5:
            return 2.0 * atr

        mean_exc = float(np.mean(exceedances))
        var_exc = float(np.var(exceedances))

        # Method of moments: xi = 0.5*(1 - mean_exc^2 / var_exc)
        #                     sigma = 0.5*mean_exc*(mean_exc^2 / var_exc)
        if var_exc < 1e-12:
            return 2.0 * atr

        ratio = mean_exc ** 2 / var_exc
        xi = 0.5 * (1.0 - ratio)
        sigma = 0.5 * mean_exc * ratio

        # Guard: if xi estimation produces nonsensical results, use empirical default
        if xi <= 0.01 or sigma <= 0:
            xi = 0.35  # A-share empirical default
            sigma = mean_exc * 0.7  # Conservative estimate

        # GPD quantile at confidence level:
        # VaR_alpha = sigma/xi * ((1/(1-alpha))^xi - 1)
        tail_prob = 1.0 - self.evt_confidence  # 0.05
        if tail_prob <= 0 or tail_prob >= 1:
            tail_prob = 0.05

        try:
            quantile = (sigma / xi) * ((1.0 / tail_prob) ** xi - 1.0)
        except (OverflowError, ZeroDivisionError, ValueError):
            quantile = 2.0 * atr

        # Clamp to reasonable range: [0.5*ATR, 5.0*ATR]
        stop_distance = max(0.5 * atr, min(5.0 * atr, quantile))

        return stop_distance

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _find_pivots(self, high, low, n):
        """Identify swing highs and swing lows."""
        piv_len = self.pivot_len
        if n < piv_len * 2 + 1:
            return [], []
        swing_highs = []
        swing_lows = []
        for i in range(piv_len, n - piv_len):
            is_high = all(
                high[i] >= high[i + j]
                for j in range(-piv_len, piv_len + 1) if j != 0
            )
            is_low = all(
                low[i] <= low[i + j]
                for j in range(-piv_len, piv_len + 1) if j != 0
            )
            if is_high:
                swing_highs.append((i, high[i]))
            if is_low:
                swing_lows.append((i, low[i]))
        return swing_highs, swing_lows

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        return float(np.mean(tr[-self.atr_period:]))

    @staticmethod
    def _volume_col(data):
        """Return the volume column name, or None if absent."""
        if 'volume' in data.columns:
            return 'volume'
        if 'vol' in data.columns:
            return 'vol'
        return None

    # ------------------------------------------------------------------
    # Real-time screening
    # ------------------------------------------------------------------

    def screen(self):
        data = self.data.copy()
        if len(data) < 40:
            return {
                'action': 'hold',
                'reason': '数据不足',
                'price': float(data['close'].iloc[-1]),
            }

        result = self._evaluate_entry(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= self.score_threshold:
            stop_dist = self._compute_evt_stop_distance(data)
            stop = price - stop_dist if direction == 1 else price + stop_dist
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f'score={score} evt_stop={stop:.2f}',
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
