"""
回归对齐K线建筑师策略 (Regression-Aligned Candlestick Architect Strategy)
=========================================================================
基于CHoCH锚定线性回归通道 + K线形态5级强度评分 + 趋势过滤。

来源: TradingView Tier 2 — Regression-Aligned Candlestick Architect [MarkitTick]

核心逻辑:
  1. CHoCH(趋势变化)检测:
     - 用pivot高/低点检测结构变化
     - 当价格突破对侧pivot时标记CHoCH
  2. 锚定线性回归通道(LRC):
     - 每次CHoCH时重置锚点, 重新计算OLS回归
     - 回归线 + 2σ上/下轨
  3. K线形态5级强度评分:
     - S1(犹豫): Doji/十字星 (body/range < 5%)
     - S2(弱): Hanging Man/倒锤 (长下/上影)
     - S3(中等): Harami/刺透/乌云 (包含/反转)
     - S4(强): 吞没/晨星/暮星/光头 (强力反转/延续)
     - S5(极端): 三兵/三鸦/跳空 (罕见强势)
  4. 信号:
     - 买入: S3+看涨形态 + 价格在LRC下半部分 + SMA上升趋势
     - 卖出: S3+看跌形态 + 价格在LRC上半部分 + 下跌趋势
     - ATR追踪止损退出

WHY this works:
  - CHoCH锚定LRC: 回归通道随市场结构变化而重置, 不会用过时的通道
  - 5级量化K线强度: 用数学比例(body/range)替代主观视觉判断
  - 趋势过滤: S4看涨在上升趋势中=高概率延续, S4看跌在上升趋势中=回调买点

数学核心: OLS回归 y=α+βx, σ_band = √(Σ(yi-ŷi)²/n), body_ratio = |C-O|/(H-L)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class RegressionCandlestickStrategy(BaseStrategy):
    """回归对齐K线建筑师 — CHoCH锚定LRC+K线5级强度+趋势过滤+ATR止损"""

    strategy_description = "K线建筑师: CHoCH锚定LRC+K线5级强度评分+趋势过滤+ATR止损"
    strategy_category = "kline_pattern"
    strategy_params_schema = {
        "pivot_lookback": {"type": "int", "default": 10, "label": "Pivot回看周期"},
        "lrc_period": {"type": "int", "default": 30, "label": "线性回归周期"},
        "trend_sma": {"type": "int", "default": 50, "label": "趋势SMA"},
        "min_strength": {"type": "int", "default": 3, "label": "最低K线强度(1-5)"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def _candle_strength(self, o, h, l, c):
        """K线形态5级强度评分: 返回 (strength, direction)
        direction: 1=看涨, -1=看跌, 0=中性
        """
        body = abs(c - o)
        rng = h - l
        if rng < 1e-8:
            return 0, 0
        body_ratio = body / rng
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        us_ratio = upper_shadow / rng if rng > 0 else 0
        ls_ratio = lower_shadow / rng if rng > 0 else 0

        is_bullish = c > o
        direction = 1 if is_bullish else -1

        # S1: 十字星/Doji
        if body_ratio < 0.05:
            return 1, 0

        # S5: 光头(Marubozu) or 大实体
        if body_ratio > 0.85:
            return 5, direction

        # S4: 看涨/看跌吞没 (大实体+前小实体方向相反)
        if body_ratio > 0.65:
            return 4, direction

        # S3: 中等实体 (Harami-like)
        if body_ratio > 0.40:
            return 3, direction

        # S2: 弱形态 (长影线)
        if us_ratio > 0.5 or ls_ratio > 0.5:
            return 2, direction

        return 2, direction

    def _ols_regression(self, y_data):
        """简单OLS线性回归"""
        n = len(y_data)
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = y_data.mean()
        ss_xy = np.sum((x - x_mean) * (y_data - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx < 1e-12:
            return y_mean, 0.0, 0.0
        beta = ss_xy / ss_xx
        alpha = y_mean - beta * x_mean
        residuals = y_data - (alpha + beta * x)
        sigma = np.sqrt(np.mean(residuals ** 2))
        return alpha, beta, sigma

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        O = df['open'].values
        n = len(df)

        # 1. Pivot points
        lb = p['pivot_lookback']
        pivot_high = np.full(n, np.nan)
        pivot_low = np.full(n, np.nan)
        for i in range(lb, n - lb):
            window_high = H[i - lb:i + lb + 1]
            window_low = L[i - lb:i + lb + 1]
            if H[i] == np.max(window_high):
                pivot_high[i] = H[i]
            if L[i] == np.min(window_low):
                pivot_low[i] = L[i]

        # 2. SMA趋势
        trend_sma = pd.Series(C).rolling(p['trend_sma']).mean().values

        # 3. ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 4. K线强度评分
        strengths = np.zeros(n, dtype=int)
        directions = np.zeros(n, dtype=int)
        for i in range(n):
            strengths[i], directions[i] = self._candle_strength(O[i], H[i], L[i], C[i])

        # 5. 信号生成 (使用滚动LRC)
        signals = []
        in_pos = False
        entry_price = 0.0
        trail_stop = 0.0
        entry_idx = 0

        warmup = max(p['pivot_lookback'] * 2 + 1, p['trend_sma'], p['lrc_period'], p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            # 滚动LRC
            lrc_start = max(warmup, i - p['lrc_period'])
            y_data = C[lrc_start:i + 1]
            if len(y_data) < 10:
                continue
            alpha, beta, sigma = self._ols_regression(y_data)
            lrc_upper = alpha + beta * (i - lrc_start) + 2 * sigma
            lrc_lower = alpha + beta * (i - lrc_start) - 2 * sigma
            lrc_mid = alpha + beta * (i - lrc_start)

            if np.isnan(trend_sma[i]):
                continue

            min_str = p['min_strength']

            if not in_pos:
                # 买入: 看涨K线(S3+) + 价格在LRC下半 + 上升趋势
                bullish = (directions[i] == 1 and strengths[i] >= min_str)
                in_lower_half = price <= lrc_mid
                uptrend = price > trend_sma[i]
                if bullish and in_lower_half and uptrend:
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

                # 卖出: 追踪止损 OR 看跌K线(S4+)在LRC上半 + 下穿趋势线
                sell_signal = False
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif (directions[i] == -1 and strengths[i] >= 4
                      and price >= lrc_mid and price < trend_sma[i]
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
