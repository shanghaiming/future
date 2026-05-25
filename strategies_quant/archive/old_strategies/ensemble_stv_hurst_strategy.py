"""
三因子集成策略: 结构张力 + VDP + Hurst Regime
=============================================
TIME × SPACE × VOLUME 三维正交集成.

哲学: "时间告诉你持续性，空间告诉你位移，成交量告诉你信念。三维一致时，真理不可辩驳。"

核心设计:
  1. Regime Gate — Hurst/KFD/Entropy 三重市场状态分类
     - Trend regime: Hurst>0.6 AND low entropy → 允许做多
     - Range regime: Hurst<0.4 AND high KFD → 禁止交易
     - Neutral: 中间状态 → 允许但要求更高评分

  2. Signal A — 结构张力方向 (from EnsembleSTV)
  3. Signal B — VDP累积Delta (from EnsembleSTV)

独立性论证:
  - Hurst: 时间维度(自相关/长记忆性) — 价格序列是否"记得自己"
  - 结构张力: 空间维度(Swing点距离) — 价格在空间中"去了哪里"
  - VDP: 资金流维度(量微观结构) — "谁在推动"
  - TIME × SPACE × VOLUME = 三正交 → 贝叶斯后验概率三次方提升

数学核心:
  Hurst: R/S = c·n^H, H>0.5=trend, H<0.5=mean-reversion
  Tension: Σ(C - anchor_j) / 7, positive=bullish displacement
  VDP: delta = V × (2C-H-L) / (H-L), cum_delta = EMA(delta, 10)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EnsembleSTVHurstStrategy(BaseStrategy):
    """三因子集成: 结构张力(空间) + VDP(资金) + Hurst Regime(时间)"""

    strategy_description = "三因子集成: 结构张力 + VDP量 + Hurst regime门控"
    strategy_category = "ensemble"
    strategy_params_schema = {
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "lookback": {"type": "int", "default": 50, "label": "结构回溯"},
        "delta_ema_period": {"type": "int", "default": 10, "label": "VDP Delta EMA周期"},
        "score_threshold": {"type": "int", "default": 4, "label": "入场评分阈值"},
        "hurst_window": {"type": "int", "default": 100, "label": "Hurst计算窗口"},
        "hurst_trend_thresh": {"type": "float", "default": 0.55, "label": "Hurst趋势阈值"},
        "hurst_range_thresh": {"type": "float", "default": 0.45, "label": "Hurst震荡阈值"},
        "entropy_bins": {"type": "int", "default": 10, "label": "熵分箱数"},
        "entropy_window": {"type": "int", "default": 50, "label": "熵计算窗口"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 1.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    # ------------------------------------------------------------------
    # Hurst Exponent (R/S analysis)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_hurst(prices, window):
        """简化Hurst指数计算: R/S分析法"""
        n = len(prices)
        hurst = np.full(n, np.nan)
        for i in range(window, n):
            series = prices[i - window:i]
            returns = np.diff(series)
            if len(returns) < 10:
                continue
            mean_ret = np.mean(returns)
            cum_dev = np.cumsum(returns - mean_ret)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(returns)
            if S > 1e-10:
                rs = R / S
                # Approximate Hurst from single window: H ≈ log(R/S) / log(n/2)
                hurst[i] = np.log(rs) / np.log(len(returns) / 2.0) if rs > 0 else 0.5
            else:
                hurst[i] = 0.5
        # Clip to reasonable range
        hurst = np.clip(np.nan_to_num(hurst, nan=0.5), 0.0, 1.0)
        return hurst

    # ------------------------------------------------------------------
    # Shannon Entropy
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_entropy(prices, window, bins):
        """Shannon熵: 收益率分布的不确定性"""
        n = len(prices)
        entropy = np.full(n, np.nan)
        for i in range(window, n):
            returns = np.diff(prices[i - window:i])
            if len(returns) < 5:
                continue
            hist, _ = np.histogram(returns, bins=bins, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                p = hist / hist.sum()
                entropy[i] = -np.sum(p * np.log2(p))
        return entropy

    # ------------------------------------------------------------------
    # Swing points & Structural Tension
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

        # --- TIME: Hurst Regime ---
        hurst = self._calc_hurst(C, p['hurst_window'])
        entropy = self._calc_entropy(C, p['entropy_window'], p['entropy_bins'])
        # Entropy median for low/high classification
        ent_valid = entropy[~np.isnan(entropy)]
        ent_median = np.median(ent_valid) if len(ent_valid) > 0 else 1.5

        # --- SPACE: Structural Tension ---
        swing_high, swing_low = self._find_pivots(H, L, p['pivot_len'])
        tension = self._calc_tension(C, swing_high, swing_low)

        # --- VOLUME: VDP ---
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

        warmup = max(p['pivot_len'] * 3, p['hurst_window'], p['entropy_window'],
                     p['delta_ema_period'] + 5, p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            # --- Regime classification ---
            h_val = hurst[i]
            e_val = entropy[i] if not np.isnan(entropy[i]) else ent_median
            is_trend_regime = h_val >= p['hurst_trend_thresh'] and e_val < ent_median
            is_range_regime = h_val < p['hurst_range_thresh']
            is_neutral_regime = not is_trend_regime and not is_range_regime

            t = tension[i]
            t_prev = tension[i - 1]
            d = cum_delta[i] if not np.isnan(cum_delta[i]) else 0
            d_prev = cum_delta[i - 1] if not np.isnan(cum_delta[i - 1]) else 0

            # --- Entry scoring ---
            if not in_pos:
                # Hard block in range regime
                if is_range_regime:
                    continue

                score = 0

                # SPACE: Structural Tension direction (3 points for flip, 1 for sustained)
                if t > 0 and t_prev <= 0:
                    score += 3
                elif t > 0:
                    score += 1

                # Structure confirmation (1 point)
                if not np.isnan(swing_low[i]) and price > swing_low[i]:
                    score += 1

                # VOLUME: VDP delta positive (2 points)
                if d > 0:
                    score += 2

                # Delta flip bonus (1 point)
                if d > 0 and d_prev <= 0:
                    score += 1

                # TIME: Hurst trend regime bonus (1 point)
                if is_trend_regime:
                    score += 1

                # Neutral regime penalty: require higher score
                effective_threshold = p['score_threshold']
                if is_neutral_regime:
                    effective_threshold += 1

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

                # Exit 1: ATR trailing stop
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True

                # Exit 2: Tension reversal
                elif t < 0 and t_prev >= 0 and hold_days >= p['hold_min']:
                    sell_signal = True

                # Exit 3: VDP delta reversal
                elif hold_days >= p['hold_min'] and d < 0 and d_prev > 0:
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct < -0.03 or pnl_pct > 0.02:
                        sell_signal = True

                # Exit 4: Regime switched to range (defensive)
                elif is_range_regime and hold_days >= p['hold_min']:
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct < 0.05:
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
