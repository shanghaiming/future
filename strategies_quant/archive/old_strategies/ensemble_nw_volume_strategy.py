"""
Nadaraya-Watson + 成交量确认集成策略 (NW Volume Ensemble)
=========================================================
NW核回归(顶级方向信号) + PropVolSplit量拆分(确认) + ATR止损。

设计哲学: NW是目前最强单一策略(+75%), 但只有25/27盈利。
加入成交量确认来: (1)提升精度到27/27, (2)可能进一步提升收益。

数学:
  NW: f̂(x) = Σ K(x,xi)yi / Σ K(x,xi), K = Gaussian kernel
  VolSplit: delta = V × (2C-H-L) / (H-L), surge = delta > μ + σ
  Entry: NW slope flip + delta positive + volume surge confirmation
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EnsembleNWVolumeStrategy(BaseStrategy):
    """NW核回归方向 + 量拆分确认 + ATR止损"""

    strategy_description = "NW集成: NW核回归方向 + 比例量拆分确认 + ATR止损"
    strategy_category = "ensemble"
    strategy_params_schema = {
        "kernel_bandwidth": {"type": "int", "default": 10, "label": "NW核带宽"},
        "bandwidth_atr_mult": {"type": "float", "default": 1.0, "label": "ATR带宽倍数"},
        "vol_ema_period": {"type": "int", "default": 10, "label": "量Delta EMA周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 1.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "use_vol_filter": {"type": "bool", "default": True, "label": "启用成交量过滤"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def _nadaraya_watson(self, src, bandwidth, h):
        """Nadaraya-Watson Gaussian kernel regression"""
        n = len(src)
        nw = np.copy(src)
        for i in range(bandwidth, n):
            window = src[i - bandwidth:i + 1]
            indices = np.arange(len(window))
            mid = len(window) // 2
            dist = (indices - mid) ** 2
            weights = np.exp(-dist / (2 * h ** 2))
            weights /= weights.sum()
            nw[i] = np.sum(weights * window)
        return nw

    def _calc_ema(self, values, period):
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
        n = len(close)
        delta = np.zeros(n)
        for i in range(n):
            bar_range = high[i] - low[i]
            if bar_range <= 0 or volume[i] <= 0:
                continue
            delta[i] = volume[i] * (2 * close[i] - high[i] - low[i]) / bar_range
        return delta

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        n = len(C)

        # Resolve volume
        if 'vol' in df.columns:
            V = df['vol'].values
        elif 'volume' in df.columns:
            V = df['volume'].values
        else:
            V = np.ones(n)

        # ATR for adaptive bandwidth
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values
        avg_atr = np.nanmean(atr[p['atr_period']:]) if n > p['atr_period'] else 1.0

        # 1. NW regression with adaptive bandwidth
        bw = p['kernel_bandwidth']
        h = max(avg_atr * p['bandwidth_atr_mult'], 0.01)
        nw = self._nadaraya_watson(C, bw, h)

        # NW slope
        slope = np.zeros(n)
        for i in range(1, n):
            slope[i] = nw[i] - nw[i - 1]

        # 2. Volume delta
        bar_delta = self._bar_delta(C, H, L, V)
        cum_delta = self._calc_ema(bar_delta, p['vol_ema_period'])

        # 3. Signal generation
        signals = []
        in_pos = False
        trail_stop = 0.0
        entry_idx = 0
        entry_price = 0.0

        warmup = max(bw + 5, p['atr_period'], p['vol_ema_period'] + 5)

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            if not in_pos:
                # Buy: NW slope flips positive
                if slope[i] > 0 and slope[i - 1] <= 0:
                    # Volume confirmation filter
                    vol_ok = True
                    if p['use_vol_filter'] and not np.isnan(cum_delta[i]):
                        vol_ok = cum_delta[i] > 0

                    if vol_ok:
                        signals.append({
                            'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                        })
                        in_pos = True
                        trail_stop = price - p['trail_atr_mult'] * atr[i]
                        entry_idx = i
                        entry_price = price

            else:
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * atr[i]
                trail_stop = max(trail_stop, new_stop)

                sell_signal = False
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif slope[i] < 0 and slope[i - 1] >= 0 and hold_days >= p['hold_min']:
                    sell_signal = True

                if sell_signal:
                    signals.append({
                        'timestamp': ts, 'action': 'sell', 'symbol': sym, 'price': price
                    })
                    in_pos = False

        if in_pos:
            signals.append({
                'timestamp': df.index[-1], 'action': 'sell', 'symbol': sym,
                'price': C[-1]
            })

        self.signals = signals
        return signals
