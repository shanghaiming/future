"""
结构-量确认集成策略 (Structure-Volume Confirmation Ensemble)
===========================================================
双信号集成: 结构张力(方向) + VDP成交量Delta(确认).

哲学: "结构告诉你价格想去哪里，成交量告诉你谁在推动。"

核心设计:
  1. Signal A — 结构张力 (StructuralTension的7点位移)
     - 正张力 = 多头主导(价格远离结构锚点向上)
     - 张力翻正 = 结构性突破

  2. Signal B — VDP累积Delta (量价拆分)
     - cum_delta > 0 = 机构净买入
     - delta翻正 = 资金开始流入

  3. 入场: 张力翻正 + VDP delta > 0
  4. 退出: ATR追踪止损 + 张力翻负 + delta翻负

独立性:
  - 张力: 基于价格结构几何(Swing点距离)
  - VDP: 基于成交量微观结构(资金流)
  - 几何×资金流 = 真正独立 → 贝叶斯后验概率提升

数学核心:
  tension = Σ price - anchor_j (7个结构锚点)
  delta = V × (2C-H-L) / (H-L)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EnsembleStructureVolumeStrategy(BaseStrategy):
    """结构-量确认集成策略 — 结构张力方向 + VDP量确认 + ATR止损"""

    strategy_description = "结构量集成: 7点结构张力 + VDP量Delta确认 + ATR止损"
    strategy_category = "ensemble"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "lookback": {"type": "int", "default": 50, "label": "结构回溯"},
        "delta_ema_period": {"type": "int", "default": 10, "label": "VDP Delta EMA周期"},
        "score_threshold": {"type": "int", "default": 4, "label": "入场评分阈值"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 1.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    # ------------------------------------------------------------------
    # Swing points
    # ------------------------------------------------------------------

    @staticmethod
    def _find_pivots(high, low, length):
        """Find swing highs and lows"""
        n = len(high)
        swing_high = np.full(n, np.nan)
        swing_low = np.full(n, np.nan)
        for i in range(length, n - length):
            is_high = all(high[i] >= high[i + j] for j in range(1, length + 1)) and \
                      all(high[i] >= high[i - j] for j in range(1, length + 1))
            if is_high:
                swing_high[i] = high[i]
            is_low = all(low[i] <= low[i + j] for j in range(1, length + 1)) and \
                     all(low[i] <= low[i - j] for j in range(1, length + 1))
            if is_low:
                swing_low[i] = low[i]
        # Forward fill
        for i in range(1, n):
            if np.isnan(swing_high[i]):
                swing_high[i] = swing_high[i - 1]
            if np.isnan(swing_low[i]):
                swing_low[i] = swing_low[i - 1]
        return swing_high, swing_low

    # ------------------------------------------------------------------
    # Structural Tension (7-point)
    # ------------------------------------------------------------------

    def _calc_tension(self, close, swing_high, swing_low):
        """7点结构张力: 价格相对7个结构锚点的净位移"""
        n = len(close)
        tension = np.zeros(n)

        for i in range(1, n):
            if np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
                continue
            c = close[i]

            # 7 structural anchors
            prev_high = swing_high[i - 1] if i > 0 and not np.isnan(swing_high[i - 1]) else swing_high[i]
            prev_low = swing_low[i - 1] if i > 0 and not np.isnan(swing_low[i - 1]) else swing_low[i]
            curr_high = swing_high[i]
            curr_low = swing_low[i]
            mid_point = (curr_high + curr_low) / 2
            highest = max(prev_high, curr_high) if not np.isnan(prev_high) else curr_high
            lowest = min(prev_low, curr_low) if not np.isnan(prev_low) else curr_low

            anchors = [prev_high, prev_low, curr_high, curr_low, mid_point, highest, lowest]
            # Net displacement: positive = price above anchors (bullish)
            displacement = sum(c - a for a in anchors if not np.isnan(a))
            tension[i] = displacement / max(len([a for a in anchors if not np.isnan(a)]), 1)

        return tension

    # ------------------------------------------------------------------
    # Volume Delta
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_ema(values, period):
        n = len(values)
        ema = np.full(n, np.nan)
        if n < period:
            return ema
        seed = np.mean(values[:period])
        ema[period - 1] = seed
        k = 2.0 / (period + 1)
        for i in range(period, n):
            ema[i] = values[i] * k + ema[i - 1] * (1 - k)
        return ema

    @staticmethod
    def _bar_delta(close, high, low, volume):
        n = len(close)
        delta = np.zeros(n)
        for i in range(n):
            bar_range = high[i] - low[i]
            if bar_range <= 0 or volume[i] <= 0:
                continue
            delta[i] = volume[i] * (2 * close[i] - high[i] - low[i]) / bar_range
        return delta

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        n = len(C)

        if 'vol' in df.columns:
            V = df['vol'].values
        elif 'volume' in df.columns:
            V = df['volume'].values
        else:
            V = np.ones(n)

        # --- Structural Tension ---
        swing_high, swing_low = self._find_pivots(H, L, p['pivot_len'])
        tension = self._calc_tension(C, swing_high, swing_low)

        # --- VDP ---
        bar_delta = self._bar_delta(C, H, L, V)
        cum_delta = self._calc_ema(bar_delta, p['delta_ema_period'])

        # --- ATR ---
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # --- Walk through bars ---
        signals = []
        in_pos = False
        trail_stop = 0.0
        entry_idx = 0
        entry_price = 0.0

        warmup = max(p['pivot_len'] * 3, p['lookback'], p['delta_ema_period'] + 5, p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            # Tension direction
            t = tension[i]
            t_prev = tension[i - 1]

            # Delta
            d = cum_delta[i] if not np.isnan(cum_delta[i]) else 0
            d_prev = cum_delta[i - 1] if not np.isnan(cum_delta[i - 1]) else 0

            # --- Entry scoring ---
            if not in_pos:
                score = 0

                # Signal A: Tension direction (3 points for flip, 1 for sustained)
                if t > 0 and t_prev <= 0:
                    score += 3  # Structural breakout
                elif t > 0:
                    score += 1

                # Price above recent swing low (structure confirmation, 1 point)
                if not np.isnan(swing_low[i]) and price > swing_low[i]:
                    score += 1

                # Signal B: VDP delta positive (2 points)
                if d > 0:
                    score += 2

                # Delta flip bonus: just turned positive (1 point)
                if d > 0 and d_prev <= 0:
                    score += 1

                if score >= p['score_threshold']:
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    trail_stop = price - p['trail_atr_mult'] * atr[i]
                    entry_idx = i
                    entry_price = price

            else:
                # --- Exit logic ---
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * atr[i]
                trail_stop = max(trail_stop, new_stop)

                sell_signal = False
                sell_reason = ''

                # Exit 1: ATR trailing stop
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                    sell_reason = 'trailing_stop'

                # Exit 2: Tension reversal
                elif t < 0 and t_prev >= 0 and hold_days >= p['hold_min']:
                    sell_signal = True
                    sell_reason = 'tension_reversal'

                # Exit 3: VDP delta reversal
                elif (hold_days >= p['hold_min'] and d < 0 and d_prev > 0):
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct < -0.03 or pnl_pct > 0.02:
                        sell_signal = True
                        sell_reason = 'delta_reversal'

                if sell_signal:
                    signals.append({
                        'timestamp': ts, 'action': 'sell', 'symbol': sym, 'price': price
                    })
                    in_pos = False

        # Force close
        if in_pos:
            signals.append({
                'timestamp': df.index[-1], 'action': 'sell', 'symbol': sym,
                'price': C[-1]
            })

        self.signals = signals
        return signals
