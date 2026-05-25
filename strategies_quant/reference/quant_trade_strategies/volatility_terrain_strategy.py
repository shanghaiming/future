"""
波动率地形引擎策略 (Volatility Terrain Engine Strategy)
=====================================================
基于双ATR比率波动率状态分类 + 百分位排名 + 挤压/扩张检测。

来源: TradingView Tier 2 — Volatility Terrain Engine [JOAT]

核心逻辑:
  1. 双ATR比率:
     - fast_atr = ATR(fast_period, 默认10)
     - slow_atr = ATR(slow_period, 默认50)
     - ratio = fast_atr / slow_atr
     - ratio > 1 = 波动率扩张, ratio < 1 = 波动率收缩
  2. ATR百分位排名:
     - percentile_rank = rank(fast_atr) over lookback_window
     - 高百分位 = 历史性高波动, 低百分位 = 历史性低波动
  3. 状态分类:
     - Squeeze: ratio < 0.82 且 fast_atr < 自身EMA
     - Expansion: ratio > 1.25 且 ratio连续上升
     - Normal: 其他
  4. 交易信号:
     - 买入: Squeeze释放(expansion开始) + price > EMA
     - 卖出: 从Expansion转入衰减 + ATR追踪止损
  5. ATR追踪止损

WHY this works:
  - 快/慢ATR比率比单一ATR更有上下文意义: 同一ATR值在不同历史背景下含义不同
  - Squeeze使用双条件过滤单bar噪声, 避免误报
  - 四态分类(扩张正/衰减正/扩张负/衰减负)比简单正负更细粒度
  - 核心理念: 先判断波动率环境, 再选择策略方向

数学核心: ratio = ATR_fast/ATR_slow, percentile_rank = rank(ATR, window)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class VolatilityTerrainStrategy(BaseStrategy):
    """波动率地形引擎策略 — 双ATR比率+百分位排名+Squeeze/Expansion状态+ATR止损"""

    strategy_description = "波动率地形: 双ATR比率状态分类+百分位排名+挤压/扩张检测+ATR止损"
    strategy_category = "volatility"
    strategy_params_schema = {
        "fast_atr_period": {"type": "int", "default": 10, "label": "快ATR周期"},
        "slow_atr_period": {"type": "int", "default": 50, "label": "慢ATR周期"},
        "percentile_window": {"type": "int", "default": 100, "label": "百分位排名窗口"},
        "squeeze_ratio": {"type": "float", "default": 0.90, "label": "Squeeze比率阈值"},
        "expansion_ratio": {"type": "float", "default": 1.05, "label": "Expansion比率阈值"},
        "trend_ema": {"type": "int", "default": 50, "label": "趋势EMA"},
        "atr_period": {"type": "int", "default": 14, "label": "止损ATR周期"},
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

        # 1. True Range + ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]

        fast_atr = pd.Series(tr).ewm(span=p['fast_atr_period'], adjust=False).mean().values
        slow_atr = pd.Series(tr).ewm(span=p['slow_atr_period'], adjust=False).mean().values
        risk_atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 2. 双ATR比率
        slow_atr = np.where(slow_atr < 1e-8, 1e-8, slow_atr)
        ratio = fast_atr / slow_atr

        # 3. ATR百分位排名
        fast_atr_s = pd.Series(fast_atr)
        pct_rank = fast_atr_s.rolling(p['percentile_window']).rank(pct=True).values

        # 4. Ratio EMA (for smoothing)
        ratio_ema = pd.Series(ratio).ewm(span=5, adjust=False).mean().values
        fast_atr_ema = pd.Series(fast_atr).ewm(span=10, adjust=False).mean().values

        # 5. 趋势EMA
        trend = pd.Series(C).ewm(span=p['trend_ema'], adjust=False).mean().values

        # 6. 状态分类
        # 0=normal, 1=squeeze, 2=expansion, 3=expansion_fading
        state = np.zeros(len(df), dtype=int)
        prev_ratio_rising = False
        for i in range(1, len(df)):
            if np.isnan(ratio[i]) or np.isnan(pct_rank[i]):
                continue
            # Squeeze: ratio < threshold AND fast_atr < own EMA
            if ratio[i] < p['squeeze_ratio'] and fast_atr[i] < fast_atr_ema[i]:
                state[i] = 1  # squeeze
            # Expansion: ratio > threshold AND rising
            elif ratio[i] > p['expansion_ratio'] and ratio[i] > ratio[i-1]:
                state[i] = 2  # expansion
                prev_ratio_rising = True
            # Fading expansion: ratio > 1 but stopped rising
            elif ratio[i] > 1.0 and prev_ratio_rising and ratio[i] <= ratio[i-1]:
                state[i] = 3  # fading
                prev_ratio_rising = False
            else:
                prev_ratio_rising = False

        # 7. 信号生成
        signals = []
        in_pos = False
        entry_price = 0.0
        trail_stop = 0.0
        entry_idx = 0
        prev_state = 0

        warmup = max(p['slow_atr_period'], p['percentile_window'], p['trend_ema'], p['atr_period'])

        for i in range(warmup, len(df)):
            price = C[i]
            ts = df.index[i]

            if not in_pos:
                # 买入条件(宽松版): Squeeze→Expansion OR ratio上穿1.0, 且价格在趋势上方
                buy_cond = False
                if prev_state == 1 and state[i] == 2 and price > trend[i]:
                    buy_cond = True
                elif (ratio[i-1] < 1.0 and ratio[i] >= 1.0 and price > trend[i]
                      and fast_atr[i] > fast_atr[i-1]):
                    buy_cond = True
                if buy_cond:
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    entry_price = price
                    trail_stop = price - p['trail_atr_mult'] * risk_atr[i]
                    entry_idx = i
            else:
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * risk_atr[i]
                trail_stop = max(trail_stop, new_stop)

                # 卖出: 追踪止损 OR Expansion衰减
                sell_signal = False
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif state[i] == 3 and hold_days >= p['hold_min']:
                    sell_signal = True

                if sell_signal:
                    signals.append({
                        'timestamp': ts, 'action': 'sell', 'symbol': sym, 'price': price
                    })
                    in_pos = False

            prev_state = state[i]

        # 强制平仓
        if in_pos:
            signals.append({
                'timestamp': df.index[-1], 'action': 'sell', 'symbol': sym,
                'price': C[-1]
            })

        self.signals = signals
        return signals
