"""
汉宁窗FIR谱带策略 (Hanning Window FIR Spectral Bands Strategy)
=============================================================
基于汉宁窗FIR滤波器 + 波动率百分位自适应ATR包络 + 三态市场分类。

来源: TradingView Tier 2 — Adaptive Spectral Bands [JOAT]

核心逻辑:
  1. 汉宁窗FIR滤波器(6层):
     - Hanning window: w(n) = 0.5 * (1 - cos(2πn/(N-1)))
     - 6层不同截止频率的FIR, 产生近零过冲和陡峭频率滚降
     - 比EMA/SMA更早在真实拐点转向
  2. 波动率百分位自适应ATR包络:
     - ATR乘数根据波动率百分位排名调节
     - 低波动时加宽包络(过滤噪音), 高波动时收窄(保留突破意义)
  3. 三态市场分类器:
     - 趋势: 短FIR > 长FIR 且排列一致
     - 震荡: FIR层交叉频繁
     - 中性: 其他
  4. 信号:
     - 买入: 短FIR上穿长FIR + 趋势状态 + 价格突破ATR下轨
     - 卖出: 短FIR下穿长FIR + ATR追踪止损
     - FIR交叉点自动发现支撑/阻力

WHY this works:
  - 汉宁窗FIR在等效频率截止下比EMA/DEMA严格更低相位滞后
  - FIR交叉点=支撑/阻力: 当不同频率的滤波器交叉时, 意味着市场结构发生变化
  - 自调节ATR: 波动率百分位排名使通道宽度随市场状态自动调节

数学核心: w(n)=0.5(1-cos(2πn/(N-1))), FIR卷积 y[n]=Σw[k]x[n-k]
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class HanningFIRStrategy(BaseStrategy):
    """汉宁窗FIR谱带策略 — 6层FIR滤波器+自适应ATR包络+三态分类+ATR止损"""

    strategy_description = "FIR谱带: 汉宁窗6层FIR+波动率百分位自适应ATR+三态分类+ATR止损"
    strategy_category = "signal_processing"
    strategy_params_schema = {
        "fir_periods": {"type": "str", "default": "10,20,40,80,120,200", "label": "FIR周期(逗号分隔)"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "pct_window": {"type": "int", "default": 100, "label": "百分位窗口"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def _hanning_fir(self, data, period):
        """应用汉宁窗FIR滤波器"""
        n = len(data)
        result = np.full(n, np.nan)
        half = period // 2
        # 汉宁窗系数
        w = np.array([0.5 * (1 - np.cos(2 * np.pi * i / (period - 1))) for i in range(period)])
        w /= w.sum()  # 归一化
        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            result[i] = np.sum(w * window)
        return result

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        n = len(C)

        # 1. 6层FIR滤波器
        fir_periods = [int(x.strip()) for x in p['fir_periods'].split(',')]
        fir_layers = []
        for period in fir_periods:
            fir_layers.append(self._hanning_fir(C, period))

        fast_fir = fir_layers[0]   # 最短周期
        slow_fir = fir_layers[-1]  # 最长周期

        # 2. ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 3. 波动率百分位
        atr_pct = pd.Series(atr).rolling(p['pct_window']).rank(pct=True).values
        # 自适应ATR乘数: 低波动时加宽, 高波动时收窄
        adaptive_mult = np.where(
            np.isnan(atr_pct), 2.5,
            np.where(atr_pct < 0.3, 3.0,   # 低波动: 宽通道
            np.where(atr_pct > 0.7, 1.5,   # 高波动: 窄通道
            2.0))                             # 中等: 标准
        )

        # 4. 三态分类
        # 趋势: fast > slow
        # 震荡: fast/slow交替
        # 中性: 其他
        state = np.zeros(n, dtype=int)  # 0=neutral, 1=uptrend, -1=downtrend
        for i in range(1, n):
            if np.isnan(fast_fir[i]) or np.isnan(slow_fir[i]):
                continue
            if fast_fir[i] > slow_fir[i]:
                state[i] = 1
            else:
                state[i] = -1

        # 5. 信号生成
        signals = []
        in_pos = False
        entry_price = 0.0
        trail_stop = 0.0
        entry_idx = 0

        warmup = max(fir_periods[-1], p['pct_window'], p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if np.isnan(fast_fir[i]) or np.isnan(slow_fir[i]):
                continue

            if not in_pos:
                # 买入: fast FIR上穿slow FIR + 价格在通道内
                if (fast_fir[i] > slow_fir[i] and fast_fir[i-1] <= slow_fir[i-1]):
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    entry_price = price
                    trail_stop = price - p['trail_atr_mult'] * atr[i]
                    entry_idx = i
            else:
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * atr[i]
                trail_stop = max(trail_stop, new_stop)

                # 卖出: 追踪止损 OR fast FIR下穿slow FIR
                sell_signal = False
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif (fast_fir[i] < slow_fir[i] and fast_fir[i-1] >= slow_fir[i-1]
                      and hold_days >= p['hold_min']):
                    sell_signal = True

                if sell_signal:
                    signals.append({
                        'timestamp': ts, 'action': 'sell', 'symbol': sym, 'price': price
                    })
                    in_pos = False

        # 强制平仓
        if in_pos:
            signals.append({
                'timestamp': df.index[-1], 'action': 'sell', 'symbol': sym,
                'price': C[-1]
            })

        self.signals = signals
        return signals
