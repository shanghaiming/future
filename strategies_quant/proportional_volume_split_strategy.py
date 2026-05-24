"""
比例成交量拆分策略 (Proportional Volume Split Strategy)
======================================================
基于比例成交量拆分 + 统计异常放量检测 + 方向一致性过滤。

来源: TradingView Tier 2 — Proportional Volume Split

核心逻辑:
  1. 比例成交量拆分 (非二元):
     - buy_vol = volume * (close - low) / (high - low)  [连续比例,保留粒度]
     - sell_vol = volume * (high - close) / (high - low)
     - delta = buy_vol - sell_vol
  2. EMA平滑Delta: 短期EMA(14)捕捉动量, 长期EMA(50)作baseline
  3. 统计放量检测:
     - vol_ma = SMA(volume, 20)
     - vol_std = STD(volume, 20)
     - surge = volume > vol_ma + N * vol_std
  4. 信号 (方向+异常双过滤):
     - 买入: delta > 0 + surge + price > EMA50 (上升趋势中的放量买入)
     - 卖出: delta < 0 + surge + price < EMA50 (下降趋势中的放量卖出)
  5. ATR追踪止损

WHY this works:
  - 比例分配 vs 二元涨跌: 收盘在K线60%位置→60/40拆分, 保留信息粒度
  - 统计放量阈值: 按品种自身成交量分布定义surge, 非固定倍数
  - 方向+异常双过滤: 仅输出有意义的机构信号, 减少非方向性放量假信号

数学核心: buy_vol = V×(C-L)/(H-L), 统计N(μ,σ)异常检测
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class ProportionalVolumeSplitStrategy(BaseStrategy):
    """比例成交量拆分策略 — 比例拆分+统计放量+方向过滤+ATR止损"""

    strategy_description = "比例量拆分: 比例成交量拆分+统计异常放量+方向一致性+ATR止损"
    strategy_category = "volume"
    strategy_params_schema = {
        "delta_ema_fast": {"type": "int", "default": 14, "label": "Delta快线EMA"},
        "delta_ema_slow": {"type": "int", "default": 50, "label": "Delta慢线EMA"},
        "trend_ema": {"type": "int", "default": 50, "label": "趋势EMA"},
        "vol_ma_period": {"type": "int", "default": 20, "label": "均量周期"},
        "surge_std_mult": {"type": "float", "default": 1.5, "label": "放量标准差倍数"},
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
        V = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

        # 1. 比例成交量拆分
        rng = H - L
        rng = np.where(rng < 1e-8, 1e-8, rng)
        buy_vol = V * (C - L) / rng
        sell_vol = V * (H - C) / rng
        delta = buy_vol - sell_vol

        # 2. EMA平滑Delta
        fast_ema = pd.Series(delta).ewm(span=p['delta_ema_fast'], adjust=False).mean().values
        slow_ema = pd.Series(delta).ewm(span=p['delta_ema_slow'], adjust=False).mean().values

        # 3. 趋势EMA
        trend = pd.Series(C).ewm(span=p['trend_ema'], adjust=False).mean().values

        # 4. 统计放量检测
        vol_s = pd.Series(V)
        vol_ma = vol_s.rolling(p['vol_ma_period']).mean().values
        vol_std = vol_s.rolling(p['vol_ma_period']).std().values
        vol_std = np.where(np.isnan(vol_std) | (vol_std < 1e-8), 1e-8, vol_std)
        surge = V > (vol_ma + p['surge_std_mult'] * vol_std)

        # 5. ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 6. 信号生成
        signals = []
        in_pos = False
        entry_price = 0.0
        trail_stop = 0.0
        entry_idx = 0

        warmup = max(p['delta_ema_slow'], p['trend_ema'], p['vol_ma_period'], p['atr_period'])

        for i in range(warmup, len(df)):
            price = C[i]
            ts = df.index[i]

            if not in_pos:
                # 买入: delta正 + 放量surge + 价格在趋势上方
                if fast_ema[i] > 0 and surge[i] and price > trend[i]:
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    entry_price = price
                    trail_stop = price - p['trail_atr_mult'] * atr[i]
                    entry_idx = i
            else:
                hold_days = i - entry_idx
                # 追踪止损
                new_stop = price - p['trail_atr_mult'] * atr[i]
                trail_stop = max(trail_stop, new_stop)

                # 卖出条件: 追踪止损 OR 方向翻转surge
                sell_signal = False
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif (fast_ema[i] < 0 and surge[i] and price < trend[i]
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
