"""
NW-波动率地形集成策略 (NW + Volatility Terrain Ensemble)
======================================================
双正交信号: NW核回归(方向) + VolTerrain波动率环境(过滤器).

哲学: "方向告诉你价格想去哪里，波动率环境告诉你时机是否成熟。
       在正确的环境中顺势而为，比在任何环境中都交易更安全。"

核心设计:
  1. Signal A — NW核回归方向 (回归维度)
     - NW斜率从负转正 = 多头方向确认
     - EMA过滤确保中长期趋势一致

  2. Signal B — VolTerrain波动率环境 (环境维度)
     - Squeeze→Expansion = 波动率释放(最佳入场时机)
     - Normal+ratio上升 = 波动率健康(允许入场)
     - Squeeze = 压缩期(禁止入场, 等待方向选择)
     - Fading = 能量耗尽(禁止入场)

独立性论证:
  - NW: 价格方向维度(核回归斜率) — "价格趋势在哪里"
  - VolTerrain: 波动率环境维度(ATR比率) — "市场能量状态"
  - 方向×环境 = 正交 → NW假突破在Squeeze/Fading中被过滤

入场逻辑:
  Best:  Squeeze→Expansion + NW斜率翻多 = "方向确认+能量释放"双触发
  Good:  Normal状态 + NW斜率翻多 + ratio>1 = "方向确认+健康环境"
  Block: Squeeze(方向未定) 或 Fading(能量耗尽)

退出逻辑:
  Exit 1: ATR追踪止损 (防御性)
  Exit 2: NW斜率翻空 (方向反转)
  Exit 3: Expansion→Fading (能量衰减)
  Exit 4: 残差>2sigma (极端偏离)

数学核心:
  NW: f̂(x)=ΣK(x,xi)yi/ΣK(x,xi), K=exp(-d²/2h²)
  VolTerrain: ratio=ATR_fast/ATR_slow, state=f(ratio,ratio',percentile)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EnsembleNWVTStrategy(BaseStrategy):
    """NW-波动率地形集成: NW方向 + VolTerrain环境过滤 + ATR止损"""

    strategy_description = "NW-波动率集成: NW核回归方向 + VolTerrain波动率环境过滤"
    strategy_category = "ensemble"
    strategy_params_schema = {
        "kernel_bandwidth": {"type": "int", "default": 10, "label": "NW核回归窗口"},
        "bandwidth_atr_mult": {"type": "float", "default": 1.0, "label": "ATR带宽倍数"},
        "ema_filter_period": {"type": "int", "default": 20, "label": "EMA过滤周期"},
        "use_ema_filter": {"type": "bool", "default": True, "label": "启用EMA过滤"},
        "fast_atr_period": {"type": "int", "default": 10, "label": "快ATR周期"},
        "slow_atr_period": {"type": "int", "default": 50, "label": "慢ATR周期"},
        "squeeze_ratio": {"type": "float", "default": 0.90, "label": "Squeeze比率阈值"},
        "expansion_ratio": {"type": "float", "default": 1.05, "label": "Expansion比率阈值"},
        "trend_ema": {"type": "int", "default": 50, "label": "趋势EMA"},
        "atr_period": {"type": "int", "default": 14, "label": "止损ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "residual_sigma_mult": {"type": "float", "default": 2.0, "label": "残差sigma倍数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    # ------------------------------------------------------------------
    # Nadaraya-Watson Kernel Regression
    # ------------------------------------------------------------------

    def _nadaraya_watson(self, y, h):
        n = len(y)
        nw = np.empty(n)
        bw = self.kernel_bandwidth
        x = np.arange(n, dtype=float)
        for i in range(n):
            start = max(0, i - bw)
            end = min(n, i + bw + 1)
            x_local = x[start:end]
            y_local = y[start:end]
            u = (x[i] - x_local) / (h if h > 0 else 1.0)
            kernel_weights = np.exp(-0.5 * u * u)
            weight_sum = np.sum(kernel_weights)
            if weight_sum > 0:
                nw[i] = np.sum(kernel_weights * y_local) / weight_sum
            else:
                nw[i] = y[i]
        return nw

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        self.kernel_bandwidth = p['kernel_bandwidth']

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        n = len(C)

        if 'vol' in df.columns:
            V = df['vol'].values
        elif 'volume' in df.columns:
            V = df['volume'].values
        else:
            V = np.ones(n)

        # --- Signal A: NW Regression ---
        # ATR for bandwidth
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr_full = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # Pre-compute NW for all bars (batch, much faster than per-bar)
        nw_line = np.full(n, np.nan)
        nw_slope = np.full(n, np.nan)
        warmup_nw = p['kernel_bandwidth'] * 3

        for i in range(warmup_nw, n):
            h = atr_full[i] * p['bandwidth_atr_mult'] if atr_full[i] > 0 else 1.0
            # Use local window
            start = max(0, i - p['kernel_bandwidth'])
            end = i + 1
            y_local = C[start:end]
            x_local = np.arange(len(y_local), dtype=float)
            center = len(y_local) - 1
            u = (center - x_local) / (h if h > 0 else 1.0)
            weights = np.exp(-0.5 * u * u)
            w_sum = np.sum(weights)
            if w_sum > 0:
                nw_line[i] = np.sum(weights * y_local) / w_sum
            else:
                nw_line[i] = C[i]

            if i >= warmup_nw + 1:
                # Need previous NW value for slope
                prev_start = max(0, i - 1 - p['kernel_bandwidth'])
                prev_end = i
                y_prev = C[prev_start:prev_end]
                x_prev = np.arange(len(y_prev), dtype=float)
                center_prev = len(y_prev) - 1
                u_prev = (center_prev - x_prev) / (h if h > 0 else 1.0)
                w_prev = np.exp(-0.5 * u_prev * u_prev)
                w_prev_sum = np.sum(w_prev)
                if w_prev_sum > 0:
                    nw_prev = np.sum(w_prev * y_prev) / w_prev_sum
                    slope_raw = nw_line[i] - nw_prev
                    nw_slope[i] = slope_raw / C[i] * 100 if C[i] > 0 else 0
                else:
                    nw_slope[i] = 0

        # EMA filter
        if p['use_ema_filter']:
            ema = pd.Series(C).ewm(span=p['ema_filter_period'], adjust=False).mean().values
        else:
            ema = np.full(n, C[0])

        # --- Signal B: Volatility Terrain ---
        fast_atr = pd.Series(tr).ewm(span=p['fast_atr_period'], adjust=False).mean().values
        slow_atr = pd.Series(tr).ewm(span=p['slow_atr_period'], adjust=False).mean().values
        slow_atr = np.where(slow_atr < 1e-8, 1e-8, slow_atr)
        ratio = fast_atr / slow_atr
        ratio_ema = pd.Series(ratio).ewm(span=5, adjust=False).mean().values
        fast_atr_ema = pd.Series(fast_atr).ewm(span=10, adjust=False).mean().values
        trend = pd.Series(C).ewm(span=p['trend_ema'], adjust=False).mean().values

        # State classification: 0=normal, 1=squeeze, 2=expansion, 3=fading
        vol_state = np.zeros(n, dtype=int)
        prev_ratio_rising = False
        for i in range(1, n):
            if np.isnan(ratio[i]):
                continue
            if ratio[i] < p['squeeze_ratio'] and fast_atr[i] < fast_atr_ema[i]:
                vol_state[i] = 1  # squeeze
            elif ratio[i] > p['expansion_ratio'] and ratio[i] > ratio[i - 1]:
                vol_state[i] = 2  # expansion
                prev_ratio_rising = True
            elif ratio[i] > 1.0 and prev_ratio_rising and ratio[i] <= ratio[i - 1]:
                vol_state[i] = 3  # fading
                prev_ratio_rising = False
            else:
                prev_ratio_rising = False

        # Residual sigma
        residuals = C - nw_line
        nw_sigma = np.full(n, np.nan)
        for i in range(p['kernel_bandwidth'], n):
            if not np.isnan(residuals[i]):
                r_window = residuals[i - p['kernel_bandwidth']:i + 1]
                r_valid = r_window[~np.isnan(r_window)]
                if len(r_valid) > 2:
                    nw_sigma[i] = np.std(r_valid)

        # --- Walk through bars ---
        signals = []
        in_pos = False
        trail_stop = 0.0
        entry_idx = 0
        entry_price = 0.0

        warmup = max(warmup_nw + 2, p['slow_atr_period'], p['trend_ema'],
                     p['ema_filter_period'], p['atr_period'])

        prev_slope = 0.0
        prev_vol_state = 0

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if np.isnan(atr_full[i]) or atr_full[i] <= 0:
                continue
            if np.isnan(nw_slope[i]):
                prev_slope = nw_slope[i] if not np.isnan(nw_slope[i]) else prev_slope
                prev_vol_state = vol_state[i]
                continue

            slope = nw_slope[i]
            vs = vol_state[i]

            # NW slope flip detection
            slope_bullish_flip = prev_slope < 0 and slope > 0
            slope_bearish_flip = prev_slope > 0 and slope < 0

            # EMA direction
            ema_ok = not p['use_ema_filter'] or price > ema[i]

            # Residual
            sigma = nw_sigma[i] if not np.isnan(nw_sigma[i]) else atr_full[i]
            residual = price - nw_line[i] if not np.isnan(nw_line[i]) else 0

            if not in_pos:
                # === Entry Logic ===
                entry_ok = False
                entry_reason = ''

                # Mode A: Squeeze→Expansion + NW bullish flip = BEST
                if (prev_vol_state == 1 and vs == 2 and slope_bullish_flip and ema_ok):
                    entry_ok = True
                    entry_reason = 'squeeze_release'

                # Mode B: Normal/Expansion + NW bullish + ratio healthy
                elif (vs in [0, 2] and slope_bullish_flip and ema_ok
                      and ratio[i] >= 1.0):
                    entry_ok = True
                    entry_reason = 'vol_healthy'

                # Block: Squeeze (direction uncertain) or Fading (energy exhausted)
                # vs==1 or vs==3 → no entry

                if entry_ok:
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    trail_stop = price - p['trail_atr_mult'] * atr_full[i]
                    entry_idx = i
                    entry_price = price

            else:
                # --- Exit logic ---
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * atr_full[i]
                trail_stop = max(trail_stop, new_stop)

                sell_signal = False

                # Exit 1: ATR trailing stop
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True

                # Exit 2: NW slope reversal
                elif slope_bearish_flip and hold_days >= p['hold_min']:
                    sell_signal = True

                # Exit 3: Volatility fading (energy exhausted)
                elif vs == 3 and hold_days >= p['hold_min']:
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct > 0 or pnl_pct < -0.03:
                        sell_signal = True

                # Exit 4: Residual band breach
                elif (sigma > 0 and abs(residual) > p['residual_sigma_mult'] * sigma
                      and hold_days >= p['hold_min']):
                    sell_signal = True

                if sell_signal:
                    signals.append({
                        'timestamp': ts, 'action': 'sell', 'symbol': sym, 'price': price
                    })
                    in_pos = False

            prev_slope = slope
            prev_vol_state = vs

        # Force close
        if in_pos:
            signals.append({
                'timestamp': df.index[-1], 'action': 'sell', 'symbol': sym,
                'price': C[-1]
            })

        self.signals = signals
        return signals
