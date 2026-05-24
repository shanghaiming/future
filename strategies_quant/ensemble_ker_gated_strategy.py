"""
KER门控集成策略 (KER-Gated Ensemble Strategy)
==============================================
三因子集成: KER效率门控(regime) + IRLS回归(方向) + VDP成交量(确认).

哲学: "效率是趋势的灵魂。只在市场有效率地运动时才跟随方向和信念。"

核心设计:
  1. Regime Gate — Kaufman效率比率 (KER)
     KER = |净变化| / Σ|日变化|
     - KER > trend_thresh (0.3) = 趋势市 → 允许交易
     - KER < range_thresh (0.15) = 震荡市 → 禁止交易
     - 中间区域 → 允许但降权(评分惩罚)
     数学: KER ≈ |H-0.5| × 标准化 — Hurst指数的线性近似，计算量仅1/100

  2. Signal A — IRLS斜率方向 (from EnsembleRegVol)
     - 买入: IRLS斜率翻正 (3分) 或持续正 (1分)

  3. Signal B — VDP累积Delta (from EnsembleRegVol)
     - 确认: cum_delta > 0 (2分)

  4. 入场条件:
     - KER >= range_thresh (至少非纯震荡)
     - 原始EnsembleRegVol评分 >= score_threshold
     - KER中间区域时 threshold 自动提升1分 (保守)

  5. 退出条件: 同EnsembleRegVol (ATR追踪止损 + 斜率反转 + delta反转)
     + KER降至极低时的防御性退出

WHY this improvement:
  - EnsembleRegVol(+81.4%)在横盘市的假信号是主要亏损来源
  - KER门控过滤掉横盘市交易 → 减少False Alarm → 提升整体收益
  - 数学: P(signal_correct)不变, P(signal_triggered)下降 → 精度↑
  - KER与IRLS/VDP独立: KER基于路径效率(几何), IRLS基于回归(统计), VDP基于量(资金流)

数学核心:
  KER = |C(t) - C(t-n)| / Σ|C(i) - C(i-1)| for i in [t-n+1, t]
  IRLS: w(j)=1/√(r²+ε²), β=(X'WX)⁻¹X'Wy
  VDP: delta = V × (2C-H-L) / (H-L), cum_delta = EMA(delta, 10)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EnsembleKERGatedStrategy(BaseStrategy):
    """KER门控集成策略 — KER regime门控 + IRLS方向 + VDP确认 + ATR止损"""

    strategy_description = "KER门控集成: 效率regime门控 + IRLS回归方向 + VDP量确认"
    strategy_category = "ensemble"
    strategy_params_schema = {
        "irls_window": {"type": "int", "default": 15, "label": "IRLS窗口"},
        "irls_iterations": {"type": "int", "default": 3, "label": "IRLS迭代次数"},
        "irls_sparsity": {"type": "float", "default": 0.3, "label": "IRLS稀疏比"},
        "delta_ema_period": {"type": "int", "default": 10, "label": "VDP Delta EMA周期"},
        "ema200_period": {"type": "int", "default": 200, "label": "趋势EMA周期"},
        "score_threshold": {"type": "int", "default": 3, "label": "入场评分阈值"},
        "ker_period": {"type": "int", "default": 20, "label": "KER计算周期"},
        "ker_trend_thresh": {"type": "float", "default": 0.3, "label": "KER趋势阈值"},
        "ker_range_thresh": {"type": "float", "default": 0.15, "label": "KER震荡阈值"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 1.0, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    # ------------------------------------------------------------------
    # KER: Kaufman Efficiency Ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_ker(closes, period):
        """Kaufman效率比率: KER = |净变化| / Σ|日变化|"""
        n = len(closes)
        ker = np.full(n, np.nan)
        for i in range(period, n):
            net_change = abs(closes[i] - closes[i - period])
            total_change = np.sum(np.abs(np.diff(closes[i - period:i + 1])))
            if total_change > 1e-10:
                ker[i] = net_change / total_change
            else:
                ker[i] = 0.0
        return ker

    # ------------------------------------------------------------------
    # IRLS smoother
    # ------------------------------------------------------------------

    def _irls_smooth(self, data, window, iters, s_ratio):
        """IRLS平滑: Hardy权重函数+稀疏重加权"""
        n = len(data)
        result = np.copy(data)

        if 'high' in self.data.columns and 'low' in self.data.columns:
            hl_range = (self.data['high'].values - self.data['low'].values)
            epsilon = np.nanmean(hl_range) * 0.5
        else:
            epsilon = np.nanstd(data) * 0.1
        epsilon = max(epsilon, 1e-8)

        for i in range(window, n):
            y = data[i - window:i + 1]
            n_w = len(y)
            weights = np.ones(n_w)

            for _ in range(iters):
                w_sum = weights.sum()
                if w_sum < 1e-12:
                    break
                y_hat = np.sum(weights * y) / w_sum
                residuals = y - y_hat
                new_weights = 1.0 / np.sqrt(residuals ** 2 + epsilon ** 2)

                sorted_idx = np.argsort(new_weights)[::-1]
                k = max(1, int(s_ratio * n_w))
                sparse_mask = np.zeros(n_w, dtype=bool)
                sparse_mask[sorted_idx[:k]] = True
                weights = np.where(sparse_mask, new_weights, 0.0)

            w_sum = weights.sum()
            if w_sum > 1e-12:
                result[i] = np.sum(weights * y) / w_sum

        return result

    # ------------------------------------------------------------------
    # Volume Delta
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_ema(values, period):
        """EMA计算"""
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
        """K线成交量方向拆分: delta = V × (2C-H-L) / (H-L)"""
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

        # --- KER: Kaufman Efficiency Ratio ---
        ker = self._calc_ker(C, p['ker_period'])

        # --- IRLS slope ---
        irls = self._irls_smooth(C, p['irls_window'], p['irls_iterations'], p['irls_sparsity'])
        slope = np.zeros(n)
        for i in range(1, n):
            slope[i] = irls[i] - irls[i - 1]

        # --- Volume Delta Pressure ---
        bar_delta = self._bar_delta(C, H, L, V)
        cum_delta = self._calc_ema(bar_delta, p['delta_ema_period'])

        # --- EMA200 ---
        ema200 = self._calc_ema(C, p['ema200_period'])

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

        warmup = max(p['irls_window'] + 5, p['ema200_period'], p['ker_period'],
                     p['delta_ema_period'] + 5, p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            # --- KER regime classification ---
            ker_val = ker[i] if not np.isnan(ker[i]) else 0.5  # default to mid if no data
            ker_is_trend = ker_val >= p['ker_trend_thresh']
            ker_is_range = ker_val < p['ker_range_thresh']
            ker_is_mid = not ker_is_trend and not ker_is_range

            # --- Scoring for entry ---
            if not in_pos:
                # Block entries in pure range market
                if ker_is_range:
                    continue

                score = 0

                # Signal A: IRLS slope direction
                if slope[i] > 0 and slope[i - 1] <= 0:
                    score += 3
                elif slope[i] > 0:
                    score += 1

                # Price above IRLS line
                if price > irls[i]:
                    score += 1

                # Signal B: VDP cumulative delta
                if not np.isnan(cum_delta[i]) and cum_delta[i] > 0:
                    score += 2

                # Zone bonus: accumulation zone
                if (not np.isnan(ema200[i]) and price < ema200[i] and
                        not np.isnan(cum_delta[i]) and cum_delta[i] > 0):
                    score += 1

                # KER mid-zone penalty: require higher score
                effective_threshold = p['score_threshold']
                if ker_is_mid:
                    effective_threshold += 1

                # KER trend bonus: bonus point for strong trend efficiency
                if ker_is_trend and ker_val > 0.5:
                    score += 1

                if score >= effective_threshold:
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

                # Exit 2: IRLS slope reversal
                elif (slope[i] < 0 and slope[i - 1] >= 0 and
                      hold_days >= p['hold_min']):
                    sell_signal = True
                    sell_reason = 'slope_reversal'

                # Exit 3: VDP delta reversal
                elif (hold_days >= p['hold_min'] and
                      not np.isnan(cum_delta[i]) and cum_delta[i] < 0 and
                      not np.isnan(cum_delta[i - 1]) and cum_delta[i - 1] > 0):
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct < -0.03 or pnl_pct > 0.02:
                        sell_signal = True
                        sell_reason = 'delta_reversal'

                # Exit 4 (NEW): KER drops to pure range — defensive exit
                elif ker_is_range and hold_days >= p['hold_min']:
                    pnl_pct = (price - entry_price) / entry_price
                    # Only exit if not in significant profit (let winners run)
                    if pnl_pct < 0.05:
                        sell_signal = True
                        sell_reason = 'ker_range_exit'

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
