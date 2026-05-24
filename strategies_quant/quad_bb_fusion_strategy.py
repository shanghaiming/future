"""
四重布林带融合策略 (Quad BB Fusion Strategy)
============================================
基于4条布林带(EMA+WMA × High+Low) + 挤压检测 + 突破信号。

来源: TradingView Tier 2 — Quad BB Multi-TF Fusion

核心逻辑:
  1. 四重布林带:
     - BB1: EMA(close) ± K×STD(close)  — 标准中轨
     - BB2: WMA(close) ± K×STD(close)  — 加权中轨(近期权重大)
     - BB3: EMA(high) ± K×STD(high)    — 上轨EMA(捕捉上方压力)
     - BB4: WMA(low) ± K×STD(low)      — 下轨WMA(捕捉下方支撑)
  2. 挤压检测:
     - squeeze = BB1带宽 < 阈值 AND BB2带宽 < 阈值
     - 意味着EMA和WMA两条独立均线都在收缩
  3. 突破信号:
     - 买入: 挤压释放 + 价格突破BB1上轨 + 成交量确认
     - 卖出: ATR追踪止损 + 下穿BB4下轨
  4. 多带宽一致性:
     - 4条带方向一致(全部向上) = 高确信度趋势

WHY this works:
  - 4条带提供4个独立视角: EMA慢视角+WMA快视角 × 上方压力+下方支撑
  - 挤压=市场在憋气: 波动率压缩到极限后必然释放(Taleb反脆弱)
  - WMA对近期数据更敏感: 能更早检测到带宽变化

数学核心: BB_i = MA(source) ± K×STD(source), squeeze = (width_1 < θ) ∧ (width_2 < θ)
哲学: 波动率是均值回归的 — 极度收缩后必然扩张. 市场的"呼吸"节律不可违抗
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class QuadBBFusionStrategy(BaseStrategy):
    """四重布林带融合策略 — EMA+WMA×4挤压检测+突破+ATR止损"""

    strategy_description = "QuadBB融合: 4条BB(EMA+WMA×High+Low)挤压检测+突破信号+ATR止损"
    strategy_category = "volatility"
    strategy_params_schema = {
        "bb_period": {"type": "int", "default": 20, "label": "BB周期"},
        "bb_mult": {"type": "float", "default": 2.0, "label": "BB标准差倍数"},
        "squeeze_pct": {"type": "float", "default": 0.15, "label": "挤压百分位阈值"},
        "trend_ema": {"type": "int", "default": 50, "label": "趋势EMA"},
        "vol_mult": {"type": "float", "default": 1.2, "label": "放量确认倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def _bb(self, source, period, mult):
        """计算布林带: mid, upper, lower, width"""
        s = pd.Series(source)
        mid = s.ewm(span=period, adjust=False).mean().values
        std = s.rolling(period).std().values
        std = np.where(np.isnan(std), 0, std)
        upper = mid + mult * std
        lower = mid - mult * std
        width = (upper - lower) / np.where(mid < 1e-8, 1e-8, np.abs(mid))
        return mid, upper, lower, width

    def _bb_wma(self, source, period, mult):
        """WMA版布林带"""
        s = pd.Series(source)
        weights = np.arange(1, period + 1, dtype=float)
        weights /= weights.sum()
        wma = s.rolling(period).apply(lambda x: np.sum(weights * x), raw=True).values
        std = s.rolling(period).std().values
        std = np.where(np.isnan(std), 0, std)
        upper = wma + mult * std
        lower = wma - mult * std
        width = (upper - lower) / np.where(np.abs(wma) < 1e-8, 1e-8, np.abs(wma))
        return wma, upper, lower, width

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        V = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        n = len(C)

        # 1. 四重布林带
        mid_ema, upper_ema, lower_ema, width_ema = self._bb(C, p['bb_period'], p['bb_mult'])
        mid_wma, upper_wma, lower_wma, width_wma = self._bb_wma(C, p['bb_period'], p['bb_mult'])
        _, upper_h, _, width_h = self._bb(H, p['bb_period'], p['bb_mult'])
        _, _, lower_l, width_l = self._bb_wma(L, p['bb_period'], p['bb_mult'])

        # 2. 挤压检测: 带宽百分位排名 < squeeze_pct
        width_ema_s = pd.Series(width_ema)
        squeeze_pct_ema = width_ema_s.rolling(100).rank(pct=True).values
        width_wma_s = pd.Series(width_wma)
        squeeze_pct_wma = width_wma_s.rolling(100).rank(pct=True).values

        in_squeeze = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.isnan(squeeze_pct_ema[i]) or np.isnan(squeeze_pct_wma[i]):
                continue
            in_squeeze[i] = (squeeze_pct_ema[i] < p['squeeze_pct'] and
                            squeeze_pct_wma[i] < p['squeeze_pct'])

        # 挤压释放: 前一根在squeeze, 当前不在
        squeeze_release = np.zeros(n, dtype=bool)
        for i in range(1, n):
            squeeze_release[i] = in_squeeze[i-1] and not in_squeeze[i]

        # 3. 趋势EMA + 成交量
        trend = pd.Series(C).ewm(span=p['trend_ema'], adjust=False).mean().values
        vol_ma = pd.Series(V).rolling(p['bb_period']).mean().values
        vol_confirm = V > vol_ma * p['vol_mult']

        # 4. ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 5. 信号生成
        signals = []
        in_pos = False
        trail_stop = 0.0
        entry_idx = 0

        warmup = max(p['bb_period'] + 10, p['trend_ema'], 100, p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if not in_pos:
                # 买入: 挤压释放 + 价格突破EMA上轨 + 趋势向上 + 量确认
                if (squeeze_release[i] and price > upper_ema[i] and
                    price > trend[i] and vol_confirm[i]):
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    trail_stop = price - p['trail_atr_mult'] * atr[i]
                    entry_idx = i
            else:
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * atr[i]
                trail_stop = max(trail_stop, new_stop)

                sell_signal = False
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif price < lower_l[i] and hold_days >= p['hold_min']:
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
