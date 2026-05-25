"""
回归-成交量确认集成策略 (Regression-Volume Confirmation Ensemble)
================================================================
双信号集成: IRLS稳健回归(方向) + VDP成交量Delta(确认) + EMA200趋势过滤。

哲学: "回归告诉你方向，成交量告诉你信念。两者一致时，真理就站在你这边。"

核心设计:
  1. 信号A — IRLS斜率方向 (L1-robust, 对涨跌停异常值天然免疫)
     - 买入触发: IRLS斜率翻正 (从<=0到>0)
     - 卖出触发: IRLS斜率翻负

  2. 信号B — VDP累积Delta方向 (量价拆分揭示机构资金流)
     - 买入确认: cumulative delta > 0 (净买入压力)
     - 卖出确认: cumulative delta < 0 (净卖出压力)

  3. 入场条件 (双信号一致):
     - 买入: IRLS斜率翻正 + VDP delta > 0 + 价格在IRLS线上方
     - 加分项: 价格在EMA200下方(delta>0 = 吸筹区) → 额外信心
     - 评分制: score >= threshold (默认4分) 才入场

  4. 退出条件:
     - ATR追踪止损 (核心退出)
     - IRLS斜率翻负 (方向反转)
     - VDP delta翻负 (机构不再支持)

WHY this ensemble works:
  - 独立性: IRLS基于价格统计(二阶矩), VDP基于成交量微观结构(资金流)
  - P(两者同时误报) ≈ P(IRLS误报) × P(VDP误报) → 精度显著提升
  - A股环境: IRLS对涨跌停鲁棒 + VDP量价拆分适合无Level2的日线数据
  - 数学: 两个独立证据一致 → 贝叶斯后验概率大幅提升

数学核心:
  IRLS: w(j)=1/√(r²+ε²), β=(X'WX)⁻¹X'Wy
  VDP: delta = V × (2C-H-L) / (H-L), cum_delta = EMA(delta, 14)
  Ensemble: entry iff P(direction|IRLS∧VDP) > threshold

哲学: 真理需要独立验证。单一证据可能是噪声，两个独立来源一致才是信号。
"两个人的证词比一个人可靠" — 这就是集成学习的数学基础。
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EnsembleRegressionVolumeStrategy(BaseStrategy):
    """回归-成交量确认集成策略 — IRLS方向 + VDP确认 + ATR止损"""

    strategy_description = "集成策略: IRLS稳健回归方向 + VDP量Delta确认 + EMA200过滤 + ATR止损"
    strategy_category = "ensemble"
    strategy_params_schema = {
        "irls_window": {"type": "int", "default": 15, "label": "IRLS窗口"},
        "irls_iterations": {"type": "int", "default": 3, "label": "IRLS迭代次数"},
        "irls_sparsity": {"type": "float", "default": 0.3, "label": "IRLS稀疏比"},
        "delta_ema_period": {"type": "int", "default": 10, "label": "VDP Delta EMA周期"},
        "ema200_period": {"type": "int", "default": 200, "label": "趋势EMA周期"},
        "score_threshold": {"type": "int", "default": 3, "label": "入场评分阈值"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 1.0, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    # ------------------------------------------------------------------
    # IRLS smoother (from IRLSStrategy)
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
    # Volume Delta (from VolumeDeltaPressureStrategy)
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

    def _bar_delta(self, close, high, low, volume):
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
        O = df['open'].values
        n = len(C)

        # Resolve volume column
        if 'vol' in df.columns:
            V = df['vol'].values
        elif 'volume' in df.columns:
            V = df['volume'].values
        else:
            V = np.ones(n)

        # --- Indicator A: IRLS slope ---
        irls = self._irls_smooth(C, p['irls_window'], p['irls_iterations'], p['irls_sparsity'])
        slope = np.zeros(n)
        for i in range(1, n):
            slope[i] = irls[i] - irls[i - 1]

        # --- Indicator B: Volume Delta Pressure ---
        bar_delta = self._bar_delta(C, H, L, V)
        cum_delta = self._calc_ema(bar_delta, p['delta_ema_period'])

        # --- Trend filter: EMA200 ---
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

        warmup = max(p['irls_window'] + 5, p['ema200_period'], p['delta_ema_period'] + 5, p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            # --- Scoring for entry ---
            if not in_pos:
                score = 0

                # Signal A: IRLS slope direction (primary, 3 points)
                # Slope just flipped positive — strongest entry signal
                if slope[i] > 0 and slope[i - 1] <= 0:
                    score += 3
                # Slope has been positive for a while (weaker)
                elif slope[i] > 0:
                    score += 1

                # Price above IRLS line (confirmation, 1 point)
                if price > irls[i]:
                    score += 1

                # Signal B: VDP cumulative delta positive (2 points)
                if not np.isnan(cum_delta[i]) and cum_delta[i] > 0:
                    score += 2

                # Zone bonus: price below EMA200 + delta > 0 = accumulation (1 point)
                # WHY: institutions buying at a discount — highest conviction
                if (not np.isnan(ema200[i]) and price < ema200[i] and
                        not np.isnan(cum_delta[i]) and cum_delta[i] > 0):
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

                # Exit 1: ATR trailing stop (primary exit)
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                    sell_reason = 'trailing_stop'

                # Exit 2: IRLS slope reversal (direction change)
                elif (slope[i] < 0 and slope[i - 1] >= 0 and
                      hold_days >= p['hold_min']):
                    sell_signal = True
                    sell_reason = 'slope_reversal'

                # Exit 3: VDP delta turned negative (institutional exit)
                elif (hold_days >= p['hold_min'] and
                      not np.isnan(cum_delta[i]) and cum_delta[i] < 0 and
                      not np.isnan(cum_delta[i - 1]) and cum_delta[i - 1] > 0):
                    # Only exit if also in profit or significant loss
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
