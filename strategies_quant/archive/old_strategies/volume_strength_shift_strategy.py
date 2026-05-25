"""
量强度状态转换策略 (Volume Strength Shift Strategy)
====================================================
基于成交量强度归一化 + 状态转换检测 + ATR追踪止损。

来源: TradingView Tier 2 — Volume Strength Shift [MMT]

核心逻辑:
  1. 成交量强度:
     - vol_strength = volume / SMA(volume, N) × direction
     - direction = +1 (阳线) / -1 (阴线), 按body比例加权
     - 归一化到[-1, +1]范围
  2. 状态分类:
     - 强买入: vol_strength > +0.5 持续M天
     - 弱买入: vol_strength ∈ (0, +0.5)
     - 强卖出: vol_strength < -0.5 持续M天
     - 弱卖出: vol_strength ∈ (-0.5, 0)
  3. 状态转换信号:
     - 买入: 弱→强买入转换 + 价格在EMA上方
     - 卖出: 弱→强卖出转换 + ATR追踪止损
  4. 量价背离检测:
     - 价格新低但量强度不新低 = 看涨背离
     - 价格新高但量强度不新高 = 看跌背离

WHY this works:
  - "量在价先": 成交量的regime变化领先于价格变化
  - 状态转换而非绝对值: 关注变化的方向, 而非当前水平
  - 归一化消除规模差异: 每只股票的成交量水平不同, 归一化使阈值通用

数学核心: strength = V/SMA(V,N) × sign(C-O), 转换: state_{t-1}≠state_t
哲学: 信念变化比信念本身更重要 — 量从弱转强是真正的信号
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class VolumeStrengthShiftStrategy(BaseStrategy):
    """量强度状态转换策略 — 成交量归一化强度+状态转换检测+量价背离"""

    strategy_description = "量强度转换: 成交量归一化强度+4态分类+状态转换信号+量价背离+ATR止损"
    strategy_category = "volume"
    strategy_params_schema = {
        "vol_ma_period": {"type": "int", "default": 20, "label": "量均周期"},
        "strong_threshold": {"type": "float", "default": 0.5, "label": "强信号阈值"},
        "confirm_bars": {"type": "int", "default": 2, "label": "确认K线数"},
        "trend_ema": {"type": "int", "default": 50, "label": "趋势EMA"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        O = df['open'].values
        V = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        n = len(C)

        # 1. 成交量强度: volume / SMA(volume) × direction
        vol_ma = pd.Series(V).rolling(p['vol_ma_period']).mean().values
        vol_ma = np.where(np.isnan(vol_ma) | (vol_ma < 1e-8), 1e-8, vol_ma)
        raw_strength = V / vol_ma - 1.0  # 超额量比, >0=放量

        # 方向加权: body比例
        body = C - O
        rng = H - L
        rng = np.where(rng < 1e-8, 1e-8, rng)
        direction = body / rng  # [-1, +1]

        vol_strength = raw_strength * direction
        # 归一化到合理范围 (clip)
        vol_strength = np.clip(vol_strength, -3.0, 3.0) / 3.0  # [-1, +1]

        # 2. 状态分类: 0=弱卖, 1=强卖, 2=弱买, 3=强买
        state = np.zeros(n, dtype=int)
        for i in range(n):
            if vol_strength[i] > p['strong_threshold']:
                state[i] = 3  # 强买
            elif vol_strength[i] > 0:
                state[i] = 2  # 弱买
            elif vol_strength[i] > -p['strong_threshold']:
                state[i] = 0  # 弱卖
            else:
                state[i] = 1  # 强卖

        # 确认: 连续confirm_bars处于同一状态才确认转换
        confirmed_state = np.zeros(n, dtype=int)
        for i in range(p['confirm_bars'], n):
            if np.all(state[i-p['confirm_bars']+1:i+1] == state[i]):
                confirmed_state[i] = state[i]
            else:
                confirmed_state[i] = confirmed_state[i-1] if i > 0 else 0

        # 3. 趋势EMA
        trend = pd.Series(C).ewm(span=p['trend_ema'], adjust=False).mean().values

        # 4. ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 5. 量价背离
        lookback = 20
        bullish_div = np.zeros(n, dtype=bool)
        for i in range(lookback, n):
            price_low = np.min(C[i-lookback:i+1])
            if C[i] == price_low:
                vol_min = np.min(vol_strength[i-lookback:i+1])
                if vol_strength[i] > vol_min + 0.3:  # 价格新低但量强度不新低
                    bullish_div[i] = True

        # 6. 信号生成
        signals = []
        in_pos = False
        entry_price = 0.0
        trail_stop = 0.0
        entry_idx = 0

        warmup = max(p['vol_ma_period'], p['trend_ema'], p['atr_period'], p['confirm_bars'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            prev_st = confirmed_state[i-1] if i > 0 else 0
            curr_st = confirmed_state[i]

            if not in_pos:
                # 买入: 状态转换到强买(3) OR 看涨量价背离
                buy_cond = False
                if curr_st == 3 and prev_st != 3 and price > trend[i]:
                    buy_cond = True  # 弱→强买转换
                elif bullish_div[i] and price > trend[i]:
                    buy_cond = True  # 量价背离
                if buy_cond:
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

                # 卖出: 追踪止损 OR 转换到强卖
                sell_signal = False
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif curr_st == 1 and prev_st != 1 and hold_days >= p['hold_min']:
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
