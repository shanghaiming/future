"""
恐慌-结构-量 三因子集成策略 (Panic-Structure-Volume Ensemble)
============================================================
三正交信号集成: 恐慌超卖(情绪) + 结构张力(几何) + VDP(资金流).

哲学: "恐慌是情绪的极端，结构是空间的真理，成交量是金钱的投票。
       三者同时指向一个方向时，市场在告诉你真相。"

核心设计:
  1. Signal A — 恐慌超卖检测 (PanicOversold的情绪维度)
     - RSI(14)<30 或 RSI(6)<20 → 超卖区
     - Stoch K<15 → 极端超卖确认
     - 放量(>1.5x) → 恐慌抛售

  2. Signal B — 结构张力方向 (StructuralTension的几何维度)
     - 7点结构锚点净位移 > 0 → 多头结构仍完整
     - 张力翻正 → 结构性突破

  3. Signal C — VDP累积Delta (资金流维度)
     - cum_delta > 0 → 机构净买入
     - delta翻正 → 资金开始流入

独立性论证:
  - 恐慌: 情绪维度(RSI/Stoch极端) — "市场感觉如何"
  - 结构: 空间维度(Swing点几何) — "价格去了哪里"
  - VDP: 资金流维度(量微观结构) — "谁在推动"
  - 情绪×空间×资金 = 三正交 → 贝叶斯后验概率三次方提升

入场逻辑 (两种模式):
  Mode A — 恐慌反转: 恐慌检测(score≥4) + 张力>0 + delta>0
    "恐慌中的机构买入 = 最佳反转入场"
  Mode B — 标准STV: 张力翻正+delta>0+swing确认 (原始EnsembleSTV)
  Mode C — 恐慌增强: 恐慌(score≥2)时降低STV阈值1分

退出逻辑:
  Exit 1: ATR追踪止损 (防御性)
  Exit 2: 张力翻负 (结构破坏)
  Exit 3: Delta翻负 (资金撤退)
  Exit 4: RSI>70 (恐慌结束, 情绪恢复)

数学核心:
  panic_score = f(RSI14, RSI6, Stoch_K, Volume_ratio)
  tension = Σ(C - anchor_j) / 7
  delta = V × (2C-H-L) / (H-L), cum_delta = EMA(delta, 10)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EnsemblePSTVStrategy(BaseStrategy):
    """恐慌-结构-量三因子集成: 恐慌检测 + 结构张力 + VDP量确认"""

    strategy_description = "三因子集成: 恐慌超卖(情绪) + 结构张力(几何) + VDP量(资金流)"
    strategy_category = "ensemble"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "lookback": {"type": "int", "default": 50, "label": "结构回溯"},
        "delta_ema_period": {"type": "int", "default": 10, "label": "VDP Delta EMA周期"},
        "score_threshold": {"type": "int", "default": 4, "label": "STV入场评分阈值"},
        "panic_threshold": {"type": "int", "default": 4, "label": "恐慌反转入场阈值"},
        "panic_boost": {"type": "int", "default": 2, "label": "恐慌增强加分"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_fast": {"type": "int", "default": 6, "label": "快速RSI"},
        "stoch_period": {"type": "int", "default": 14, "label": "Stoch周期"},
        "vol_ma_period": {"type": "int", "default": 20, "label": "量均线周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 1.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "rsi_exit": {"type": "int", "default": 75, "label": "RSI退出阈值(恐慌恢复)"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    # ------------------------------------------------------------------
    # Swing points & Structural Tension (from EnsembleSTV)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_pivots(high, low, length):
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
        for i in range(1, n):
            if np.isnan(swing_high[i]):
                swing_high[i] = swing_high[i - 1]
            if np.isnan(swing_low[i]):
                swing_low[i] = swing_low[i - 1]
        return swing_high, swing_low

    def _calc_tension(self, close, swing_high, swing_low):
        n = len(close)
        tension = np.zeros(n)
        for i in range(1, n):
            if np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
                continue
            c = close[i]
            prev_high = swing_high[i - 1] if i > 0 and not np.isnan(swing_high[i - 1]) else swing_high[i]
            prev_low = swing_low[i - 1] if i > 0 and not np.isnan(swing_low[i - 1]) else swing_low[i]
            curr_high = swing_high[i]
            curr_low = swing_low[i]
            mid_point = (curr_high + curr_low) / 2
            highest = max(prev_high, curr_high) if not np.isnan(prev_high) else curr_high
            lowest = min(prev_low, curr_low) if not np.isnan(prev_low) else curr_low
            anchors = [prev_high, prev_low, curr_high, curr_low, mid_point, highest, lowest]
            displacement = sum(c - a for a in anchors if not np.isnan(a))
            tension[i] = displacement / max(len([a for a in anchors if not np.isnan(a)]), 1)
        return tension

    # ------------------------------------------------------------------
    # Volume Delta (from EnsembleSTV)
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
    # Panic Detection (from PanicOversold, vectorized)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_rsi_series(close, period):
        """Vectorized RSI calculation for entire series"""
        n = len(close)
        rsi = np.full(n, 50.0)
        if n < period + 1:
            return rsi
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        # Use rolling mean for RSI
        for i in range(period, n):
            avg_gain = np.mean(gains[i - period:i])
            avg_loss = np.mean(losses[i - period:i])
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rsi[i] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)
        return rsi

    @staticmethod
    def _calc_stoch_series(close, high, low, period):
        """Vectorized Stochastic K for entire series"""
        n = len(close)
        stoch = np.full(n, 50.0)
        for i in range(period - 1, n):
            lowest = np.min(low[i - period + 1:i + 1])
            highest = np.max(high[i - period + 1:i + 1])
            if highest == lowest:
                stoch[i] = 50.0
            else:
                stoch[i] = (close[i] - lowest) / (highest - lowest) * 100.0
        return stoch

    def _calc_panic_score(self, close, high, low, volume, p):
        """
        恐慌评分: 多维超卖共振
        RSI14<25→+5, <30→+3 | RSI6<15→+3, <20→+2
        Stoch<10→+3, <15→+2 | Volume>2x→+3, >1.5x→+2
        """
        n = len(close)
        panic = np.zeros(n)

        rsi14 = self._calc_rsi_series(close, p['rsi_period'])
        rsi6 = self._calc_rsi_series(close, p['rsi_fast'])
        stoch_k = self._calc_stoch_series(close, high, low, p['stoch_period'])

        # Volume ratio
        vol_ma = np.full(n, np.nan)
        vmp = p['vol_ma_period']
        for i in range(vmp - 1, n):
            vol_ma[i] = np.mean(volume[i - vmp + 1:i + 1])

        for i in range(max(p['rsi_period'], p['stoch_period'], vmp) + 5, n):
            score = 0

            # RSI(14) panic
            if rsi14[i] < 15:
                score += 5
            elif rsi14[i] < 25:
                score += 4
            elif rsi14[i] < 30:
                score += 3
            elif rsi14[i] > 85:
                score -= 5
            elif rsi14[i] > 75:
                score -= 3

            # RSI(6) fast confirmation
            if rsi6[i] < 10:
                score += 4
            elif rsi6[i] < 15:
                score += 3
            elif rsi6[i] < 20:
                score += 2
            elif rsi6[i] > 90:
                score -= 4
            elif rsi6[i] > 80:
                score -= 3

            # Stochastic K extreme
            if stoch_k[i] < 8:
                score += 3
            elif stoch_k[i] < 15:
                score += 2
            elif stoch_k[i] > 92:
                score -= 3
            elif stoch_k[i] > 85:
                score -= 2

            # Volume spike (panic selling)
            if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
                vol_ratio = volume[i] / vol_ma[i]
                if vol_ratio > 2.5:
                    if score > 0:
                        score += 4  # Panic capitulation
                    else:
                        score -= 4
                elif vol_ratio > 2.0:
                    if score > 0:
                        score += 3
                    else:
                        score -= 3
                elif vol_ratio > 1.5:
                    if score > 0:
                        score += 2
                    else:
                        score -= 2

            panic[i] = score

        return panic, rsi14, rsi6

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

        # --- Signal A: Panic Detection ---
        panic, rsi14, rsi6 = self._calc_panic_score(C, H, L, V, p)

        # --- Signal B: Structural Tension ---
        swing_high, swing_low = self._find_pivots(H, L, p['pivot_len'])
        tension = self._calc_tension(C, swing_high, swing_low)

        # --- Signal C: VDP ---
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
        entry_mode = ''

        warmup = max(p['pivot_len'] * 3, p['lookback'], p['delta_ema_period'] + 5,
                     p['atr_period'], p['rsi_period'] + 5, p['stoch_period'] + 5,
                     p['vol_ma_period'] + 5)

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            # Current signal values
            ps = panic[i]
            t = tension[i]
            t_prev = tension[i - 1]
            d = cum_delta[i] if not np.isnan(cum_delta[i]) else 0
            d_prev = cum_delta[i - 1] if not np.isnan(cum_delta[i - 1]) else 0
            r14 = rsi14[i]

            # --- Entry logic ---
            if not in_pos:

                # === Mode A: Panic Reversal (三因子全部确认) ===
                # 恐慌检测 + 结构仍向上 + 机构在买入 = "恐慌中的聪明钱"
                if (ps >= p['panic_threshold'] and t > 0 and d > 0):
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    trail_stop = price - p['trail_atr_mult'] * atr[i]
                    entry_idx = i
                    entry_price = price
                    entry_mode = 'panic_reversal'
                    continue

                # === Mode B: Standard STV (原始EnsembleSTV逻辑) ===
                stv_score = 0

                # Signal B: Tension direction
                if t > 0 and t_prev <= 0:
                    stv_score += 3
                elif t > 0:
                    stv_score += 1

                # Structure confirmation
                if not np.isnan(swing_low[i]) and price > swing_low[i]:
                    stv_score += 1

                # Signal C: VDP delta
                if d > 0:
                    stv_score += 2
                if d > 0 and d_prev <= 0:
                    stv_score += 1

                # Note: Removed panic boost (Mode C) — testing showed it degraded STV quality
                # Panic only contributes via Mode A (pure triple confirmation)

                if stv_score >= p['score_threshold']:
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    trail_stop = price - p['trail_atr_mult'] * atr[i]
                    entry_idx = i
                    entry_price = price
                    entry_mode = 'stv_standard' if ps < 2 else 'stv_panic_boosted'

            else:
                # --- Exit logic ---
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * atr[i]
                trail_stop = max(trail_stop, new_stop)

                sell_signal = False

                # Exit 1: ATR trailing stop
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True

                # Exit 2: Tension reversal
                elif t < 0 and t_prev >= 0 and hold_days >= p['hold_min']:
                    sell_signal = True

                # Exit 3: VDP delta reversal (with PnL filter)
                elif hold_days >= p['hold_min'] and d < 0 and d_prev > 0:
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct < -0.03 or pnl_pct > 0.02:
                        sell_signal = True

                # Exit 4: RSI recovery (panic over → take profit)
                elif entry_mode == 'panic_reversal' and hold_days >= p['hold_min']:
                    if r14 > p['rsi_exit']:
                        sell_signal = True

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
