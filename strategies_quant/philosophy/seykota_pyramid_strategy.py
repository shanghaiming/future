"""
Seykota金字塔趋势策略 (Seykota Pyramiding Trend Strategy)
=========================================================
Ed Seykota风格趋势跟踪 + 金字塔加仓 — 集中优势兵力，在最佳机会上集中火力。

哲学: "集中优势兵力" — concentrate superior force on the best opportunities

核心逻辑:
  1. 入场: Donchian通道突破(20日) + KER趋势质量过滤
  2. 金字塔加仓: 每上涨1×ATR加仓，递减仓位 (25%→20%→18%→17%→20%)
  3. 追踪止损: 最高收盘价 - 2.5×ATR (弱趋势时放宽至3.0×ATR)
  4. 通道出场: 价格穿越10日Donchian反向边界
  5. 最大持仓: 120天
  6. KER门控: Kaufman Efficiency Ratio过滤震荡市
  7. VDP确认: 成交量方向确认(有成交量时)
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from core.base_strategy import BaseStrategy


class SeykotaPyramidStrategy(BaseStrategy):
    """Seykota金字塔 — 通道突破 + KER过滤 + 金字塔加仓 + 自适应追踪止损"""

    strategy_description = (
        "Seykota Pyramid: Donchian突破 + KER质量过滤 + "
        "金字塔递减加仓 + 自适应ATR追踪止损"
    )
    strategy_category = "trend_following"
    strategy_params_schema = {
        "entry_period": {
            "type": "int", "default": 20, "label": "入场Donchian周期",
        },
        "exit_period": {
            "type": "int", "default": 10, "label": "出场Donchian周期",
        },
        "atr_period": {
            "type": "int", "default": 20, "label": "ATR周期",
        },
        "atr_mult": {
            "type": "float", "default": 2.5, "label": "止损ATR倍数",
        },
        "max_pyramid": {
            "type": "int", "default": 4, "label": "最大金字塔加仓次数",
        },
        "ker_threshold": {
            "type": "float", "default": 0.25, "label": "KER入场阈值",
        },
        "max_hold": {
            "type": "int", "default": 120, "label": "最大持仓天数",
        },
    }

    # Pyramid position sizing: level 0=initial, 1-4=pyramid adds
    PYRAMID_SIZES = [0.25, 0.20, 0.18, 0.17, 0.20]

    def get_default_params(self) -> Dict:
        return {
            "entry_period": 20,
            "exit_period": 10,
            "atr_period": 20,
            "atr_mult": 2.5,
            "max_pyramid": 4,
            "ker_threshold": 0.25,
            "max_hold": 120,
        }

    def validate_params(self):
        p = self.params
        if p["entry_period"] < 5:
            raise ValueError("entry_period must be >= 5")
        if p["exit_period"] < 3:
            raise ValueError("exit_period must be >= 3")
        if p["atr_period"] < 5:
            raise ValueError("atr_period must be >= 5")
        if p["atr_mult"] <= 0:
            raise ValueError("atr_mult must be > 0")
        if p["max_pyramid"] < 0 or p["max_pyramid"] > 10:
            raise ValueError("max_pyramid must be between 0 and 10")
        if not (0 < p["ker_threshold"] < 1):
            raise ValueError("ker_threshold must be between 0 and 1")
        if p["max_hold"] < 10:
            raise ValueError("max_hold must be >= 10")

    # ------------------------------------------------------------------ #
    #  Indicator helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calc_true_range(high: np.ndarray, low: np.ndarray,
                         close: np.ndarray) -> np.ndarray:
        """Compute True Range array (length = len(high) - 1)."""
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        return tr

    def _calc_atr(self, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray) -> float:
        """Average True Range over atr_period bars."""
        period = self.params["atr_period"]
        if len(close) < period + 1:
            return 0.0
        tr = self._calc_true_range(high, low, close)
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _calc_ker(close: np.ndarray, period: int = 20) -> float:
        """Kaufman Efficiency Ratio: |net_change| / sum(|daily_changes|).

        Returns 0.0 if insufficient data.
        """
        if len(close) < period + 1:
            return 0.0
        window = close[-(period + 1):]
        net_change = abs(float(window[-1] - window[0]))
        daily_changes = np.abs(np.diff(window))
        total = float(np.sum(daily_changes))
        if total < 1e-12:
            return 0.0
        return net_change / total

    def _calc_vdp_delta(self, bar: pd.Series) -> float:
        """Volume-Directional Price delta.

        Delta = V * (2*C - H - L) / (H - L)
        Returns 0.0 if volume not available or high == low.
        """
        if "volume" not in self.data.columns:
            return 0.0
        vol = float(bar.get("volume", 0))
        if vol <= 0:
            return 0.0
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        denom = h - l
        if denom < 1e-12:
            return 0.0
        return vol * (2 * c - h - l) / denom

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _entry_score(self, direction: int, ker: float,
                     vdp_delta: float) -> float:
        """Compute entry score.

        - Breakout + KER filter = base score 5
        - VDP confirmation = +2
        - Threshold = 5
        """
        score = 5.0  # breakout + KER filter satisfied (caller ensures this)
        # Bonus for strong KER
        if ker > 0.4:
            score += 0.5
        # VDP confirmation
        if vdp_delta > 0 and direction == 1:
            score += 2.0
        elif vdp_delta < 0 and direction == -1:
            score += 2.0
        return score

    # ------------------------------------------------------------------ #
    #  Max pyramid count adjusted by KER                                   #
    # ------------------------------------------------------------------ #

    def _effective_max_pyramid(self, ker: float) -> int:
        """Reduce pyramid adds when trend quality is moderate."""
        base_max = self.params["max_pyramid"]
        if ker > 0.3:
            return base_max  # strong trend: full pyramid
        elif ker > 0.2:
            return min(base_max, 2)  # moderate: cap at 2
        else:
            return 0  # ranging: no trade at all

    # ------------------------------------------------------------------ #
    #  Adaptive ATR multiplier                                             #
    # ------------------------------------------------------------------ #

    def _adaptive_atr_mult(self, ker: float) -> float:
        """Widen stop when trend is weaker."""
        base = self.params["atr_mult"]
        if ker < 0.3:
            return base + 0.5  # widen from 2.5 to 3.0
        return base

    # ------------------------------------------------------------------ #
    #  Main signal generation                                              #
    # ------------------------------------------------------------------ #

    def generate_signals(self) -> List[Dict]:
        data = self.data.copy()
        if "symbol" not in data.columns:
            data["symbol"] = "DEFAULT"

        entry_period = self.params["entry_period"]
        exit_period = self.params["exit_period"]
        ker_threshold = self.params["ker_threshold"]
        max_hold = self.params["max_hold"]

        # Need enough bars for all indicators
        warmup = max(entry_period, self.params["atr_period"], exit_period) + 30
        if len(data) < warmup:
            return self.signals

        # Pre-extract arrays per symbol for performance
        symbols = data["symbol"].unique()
        self.signals = []

        # Position state per symbol
        pos_state: Dict[str, Dict] = {}
        # pos_state[sym] = {
        #   'direction': 1/-1/0,
        #   'entry_idx': int,          # index position in the symbol's array
        #   'high_water': float,
        #   'low_water': float,
        #   'pyramid_level': int,       # 0 = initial, 1..max = pyramid adds
        #   'last_pyramid_price': float,
        #   'bars_held': int,
        # }

        # Get unique timestamps sorted
        unique_times = sorted(data.index.unique())

        for t_idx, current_time in enumerate(unique_times):
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            for _, bar in current_bars.iterrows():
                sym = bar["symbol"]
                close_now = float(bar["close"])
                high_now = float(bar["high"])
                low_now = float(bar["low"])

                # Get historical data for this symbol up to (but not including) current bar
                sym_mask = (data["symbol"] == sym) & (data.index < current_time)
                sym_hist = data.loc[sym_mask]

                if len(sym_hist) < warmup - 1:
                    continue

                close_arr = sym_hist["close"].values.astype(float)
                high_arr = sym_hist["high"].values.astype(float)
                low_arr = sym_hist["low"].values.astype(float)

                # Current indicators
                atr = self._calc_atr(
                    np.append(high_arr, high_now),
                    np.append(low_arr, low_now),
                    np.append(close_arr, close_now),
                )
                ker = self._calc_ker(
                    np.append(close_arr, close_now), period=entry_period,
                )

                if atr <= 0:
                    continue

                # Donchian channels
                donchian_high = float(np.max(close_arr[-entry_period:]))
                donchian_low = float(np.min(close_arr[-entry_period:]))

                # Exit channel (shorter period)
                if len(close_arr) >= exit_period:
                    exit_high = float(np.max(close_arr[-exit_period:]))
                    exit_low = float(np.min(close_arr[-exit_period:]))
                else:
                    exit_high = donchian_high
                    exit_low = donchian_low

                # ---- Check exits for existing position ----
                if sym in pos_state and pos_state[sym]["direction"] != 0:
                    st = pos_state[sym]
                    st["bars_held"] += 1

                    # Update water marks
                    if st["direction"] == 1:
                        st["high_water"] = max(st["high_water"], close_now)
                    else:
                        st["low_water"] = min(st["low_water"], close_now)

                    should_exit = False
                    exit_reason = ""

                    # 1. Trailing stop
                    atr_mult = self._adaptive_atr_mult(ker)
                    if st["direction"] == 1:
                        trail_stop = st["high_water"] - atr_mult * atr
                        if close_now < trail_stop:
                            should_exit = True
                            exit_reason = "trailing_stop"
                    else:
                        trail_stop = st["low_water"] + atr_mult * atr
                        if close_now > trail_stop:
                            should_exit = True
                            exit_reason = "trailing_stop"

                    # 2. Channel exit (opposite Donchian boundary)
                    if not should_exit:
                        if st["direction"] == 1 and close_now < exit_low:
                            should_exit = True
                            exit_reason = "channel_exit"
                        elif st["direction"] == -1 and close_now > exit_high:
                            should_exit = True
                            exit_reason = "channel_exit"

                    # 3. Max hold
                    if not should_exit and st["bars_held"] >= max_hold:
                        should_exit = True
                        exit_reason = "max_hold"

                    if should_exit:
                        action = "sell" if st["direction"] == 1 else "buy"
                        self._record_signal(
                            current_time, action, sym, close_now,
                            reason=exit_reason,
                            pyramid_level=st["pyramid_level"],
                            position_pct=self._total_position_pct(
                                st["pyramid_level"]
                            ),
                            high_water=round(st["high_water"], 4),
                            low_water=round(st["low_water"], 4),
                            bars_held=st["bars_held"],
                            ker=round(ker, 4),
                            atr=round(atr, 4),
                        )
                        del pos_state[sym]
                        continue

                    # ---- Check pyramid add ----
                    eff_max = self._effective_max_pyramid(ker)
                    if (
                        st["pyramid_level"] < eff_max
                        and st["last_pyramid_price"] > 0
                    ):
                        price_move = (
                            (close_now - st["last_pyramid_price"])
                            if st["direction"] == 1
                            else (st["last_pyramid_price"] - close_now)
                        )
                        if price_move >= atr:
                            # KER must remain healthy for pyramid
                            if ker > 0.2:
                                st["pyramid_level"] += 1
                                new_pct = self.PYRAMID_SIZES[
                                    min(
                                        st["pyramid_level"],
                                        len(self.PYRAMID_SIZES) - 1,
                                    )
                                ]
                                action = (
                                    "buy" if st["direction"] == 1 else "sell"
                                )
                                self._record_signal(
                                    current_time, action, sym, close_now,
                                    reason="pyramid_add",
                                    pyramid_level=st["pyramid_level"],
                                    position_pct=new_pct,
                                    cumulative_pct=self._total_position_pct(
                                        st["pyramid_level"]
                                    ),
                                    ker=round(ker, 4),
                                    atr=round(atr, 4),
                                )
                                st["last_pyramid_price"] = close_now

                    continue  # position exists, skip new entry check

                # ---- Check new entry ----
                if sym in pos_state:
                    continue  # already in position

                # KER gate: no trade in ranging market
                if ker < ker_threshold:
                    continue

                # Donchian breakout check
                direction = 0
                if close_now > donchian_high:
                    direction = 1
                elif close_now < donchian_low:
                    direction = -1

                if direction == 0:
                    continue

                # Check effective max pyramid (which also gates on KER)
                if self._effective_max_pyramid(ker) == 0 and ker < ker_threshold:
                    continue

                # VDP confirmation
                vdp_delta = self._calc_vdp_delta(bar)

                # Entry score
                score = self._entry_score(direction, ker, vdp_delta)
                if score < 5.0:
                    continue

                action = "buy" if direction == 1 else "sell"
                init_pct = self.PYRAMID_SIZES[0]
                self._record_signal(
                    current_time, action, sym, close_now,
                    reason="breakout_entry",
                    pyramid_level=0,
                    position_pct=init_pct,
                    cumulative_pct=init_pct,
                    score=round(score, 2),
                    ker=round(ker, 4),
                    atr=round(atr, 4),
                    vdp_delta=round(vdp_delta, 4) if vdp_delta != 0 else 0,
                    donchian_high=round(donchian_high, 4),
                    donchian_low=round(donchian_low, 4),
                )

                pos_state[sym] = {
                    "direction": direction,
                    "entry_idx": t_idx,
                    "high_water": close_now if direction == 1 else 0.0,
                    "low_water": close_now if direction == -1 else float("inf"),
                    "pyramid_level": 0,
                    "last_pyramid_price": close_now,
                    "bars_held": 0,
                }

        return self.signals

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _total_position_pct(self, pyramid_level: int) -> float:
        """Cumulative position percentage up to given pyramid level."""
        total = 0.0
        for i in range(pyramid_level + 1):
            idx = min(i, len(self.PYRAMID_SIZES) - 1)
            total += self.PYRAMID_SIZES[idx]
        return round(total, 4)

    def screen(self) -> Dict:
        """Quick screen using latest data bar."""
        if len(self.data) < 50:
            return {
                "action": "hold",
                "reason": "insufficient data (< 50 bars)",
                "price": float(self.data["close"].iloc[-1]),
            }

        close = self.data["close"].values.astype(float)
        high = self.data["high"].values.astype(float)
        low = self.data["low"].values.astype(float)

        entry_period = self.params["entry_period"]
        ker_threshold = self.params["ker_threshold"]

        atr = self._calc_atr(high, low, close)
        ker = self._calc_ker(close, period=entry_period)
        donchian_high = float(np.max(close[-entry_period:]))
        donchian_low = float(np.min(close[-entry_period:]))
        last_close = float(close[-1])

        if ker < ker_threshold:
            return {
                "action": "hold",
                "reason": f"KER {ker:.3f} < threshold {ker_threshold}",
                "price": last_close,
                "ker": round(ker, 4),
            }

        direction = 0
        if last_close > donchian_high:
            direction = 1
        elif last_close < donchian_low:
            direction = -1

        if direction == 0:
            return {
                "action": "hold",
                "reason": "no Donchian breakout",
                "price": last_close,
                "ker": round(ker, 4),
                "donchian_high": round(donchian_high, 4),
                "donchian_low": round(donchian_low, 4),
            }

        bar = self.data.iloc[-1]
        vdp_delta = self._calc_vdp_delta(bar)
        score = self._entry_score(direction, ker, vdp_delta)

        action = "buy" if direction == 1 else "sell"
        return {
            "action": action,
            "reason": (
                f"breakout score={score:.1f} KER={ker:.3f} "
                f"ATR={atr:.2f}"
            ),
            "price": last_close,
            "score": round(score, 2),
            "ker": round(ker, 4),
            "atr": round(atr, 4),
            "vdp_delta": round(vdp_delta, 4),
        }
