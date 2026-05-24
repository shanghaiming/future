"""
市场状态识别模块 (Market Regime Detector)
=========================================
基于统计学方法判断当前市场处于趋势状态还是震荡状态。

核心方法:
1. ADX (Average Directional Index): 衡量趋势强度, >25为趋势市
2. 线性回归R²: 价格线性拟合度, 高R²=趋势, 低R²=震荡
3. 价格波动效率 (Efficiency Ratio): 净变化/总波动, 高=趋势
4. EMA斜率 + 价格位置: 简单趋势方向判断

统计学基础:
- R² (决定系数): 回归平方和/总平方和, 衡量线性趋势的显著程度
- 效率率: |净价格变化| / Σ|每日变化|, 趋势市中接近1, 震荡市接近0
- ADX: 基于方向性运动的趋势强度, 不分涨跌
"""
import numpy as np


class MarketRegimeDetector:
    """市场状态识别器 — 判断趋势/震荡/过渡"""

    TREND = 'trend'
    RANGE = 'range'
    TRANSITION = 'transition'

    def __init__(self, lookback=20, adx_threshold=25, r2_threshold=0.4, er_threshold=0.3):
        self.lookback = lookback
        self.adx_threshold = adx_threshold
        self.r2_threshold = r2_threshold
        self.er_threshold = er_threshold

    def detect(self, close, high=None, low=None):
        """
        综合判断市场状态, 返回 (state, confidence, details)
        state: 'trend' | 'range' | 'transition'
        confidence: 0.0 ~ 1.0
        details: dict with individual scores
        """
        n = len(close)
        if n < self.lookback + 5:
            return self.TRANSITION, 0.0, {}

        scores = {}

        # 1. 效率率 (Kaufman ER)
        er = self._efficiency_ratio(close)
        scores['efficiency_ratio'] = er

        # 2. 线性回归R²
        r2 = self._linear_r2(close)
        scores['r_squared'] = r2

        # 3. ADX (需要HLC数据)
        if high is not None and low is not None:
            adx = self._calc_adx(high, low, close)
            scores['adx'] = adx
        else:
            scores['adx'] = 0

        # 4. EMA斜率
        ema_slope = self._ema_slope(close)
        scores['ema_slope'] = ema_slope

        # 5. 价格穿越EMA次数 (震荡市频繁穿越)
        ema_crosses = self._ema_cross_count(close)
        scores['ema_crosses'] = ema_crosses

        # === 综合评分 ===
        trend_score = 0.0
        total_weight = 0.0

        # 效率率: 权重0.25
        if er > self.er_threshold:
            trend_score += 0.25 * min(er / 0.6, 1.0)
        total_weight += 0.25

        # R²: 权重0.25
        if r2 > self.r2_threshold:
            trend_score += 0.25 * min(r2 / 0.7, 1.0)
        total_weight += 0.25

        # ADX: 权重0.25
        if scores['adx'] > self.adx_threshold:
            trend_score += 0.25 * min(scores['adx'] / 50.0, 1.0)
        total_weight += 0.25

        # EMA穿越次数: 权重0.25 (穿越多=震荡)
        max_crosses = self.lookback / 3  # 频繁穿越=震荡
        cross_ratio = 1.0 - min(ema_crosses / max_crosses, 1.0)
        trend_score += 0.25 * cross_ratio
        total_weight += 0.25

        confidence = trend_score / total_weight if total_weight > 0 else 0.5

        if confidence > 0.6:
            state = self.TREND
        elif confidence < 0.35:
            state = self.RANGE
        else:
            state = self.TRANSITION

        # 趋势方向
        direction = 'up' if ema_slope > 0 else 'down'

        return state, confidence, {**scores, 'direction': direction}

    def _efficiency_ratio(self, close):
        """Kaufman效率率: |净变化| / Σ|每日变化|"""
        recent = close[-self.lookback:]
        net_change = abs(recent[-1] - recent[0])
        total_volatility = np.sum(np.abs(np.diff(recent)))
        return net_change / total_volatility if total_volatility > 0 else 0

    def _linear_r2(self, close):
        """线性回归R² — 趋势强度"""
        recent = close[-self.lookback:]
        x = np.arange(self.lookback, dtype=float)
        y = recent

        # 最小二乘
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_yy = np.sum((y - y_mean) ** 2)

        if ss_xx == 0 or ss_yy == 0:
            return 0.0

        r = ss_xy / np.sqrt(ss_xx * ss_yy)
        return r ** 2

    def _calc_adx(self, high, low, close, period=14):
        """简化的ADX计算"""
        n = len(close)
        if n < period * 2:
            return 0

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )

        # +DM, -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Wilder平滑
        def wilder_smooth(arr, period):
            result = np.zeros_like(arr)
            result[period] = np.mean(arr[:period])
            for i in range(period + 1, len(arr)):
                result[i] = result[i - 1] - result[i - 1] / period + arr[i]
            return result

        atr = wilder_smooth(tr, period)
        atr_safe = np.where(atr > 1e-10, atr, 1e-10)
        plus_di = 100 * wilder_smooth(plus_dm, period) / atr_safe
        minus_di = 100 * wilder_smooth(minus_dm, period) / atr_safe

        # DX -> ADX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        di_sum_safe = np.where(di_sum > 1e-10, di_sum, 1e-10)
        dx = 100 * di_diff / di_sum_safe

        # ADX = DX的移动平均
        adx = np.mean(dx[-period:])
        return adx

    def _ema_slope(self, close, period=20):
        """EMA斜率 — 归一化"""
        k = 2.0 / (period + 1)
        ema = close[0]
        for i in range(1, len(close)):
            ema = close[i] * k + ema * (1 - k)
        ema_current = ema

        # 前5个bar的EMA
        ema_prev = close[0]
        for i in range(1, len(close) - 5):
            ema_prev = close[i] * k + ema_prev * (1 - k)

        if ema_prev == 0:
            return 0
        return (ema_current - ema_prev) / ema_prev

    def _ema_cross_count(self, close, period=20):
        """价格穿越EMA次数 — 穿越多=震荡"""
        k = 2.0 / (period + 1)
        ema = np.zeros(len(close))
        ema[0] = close[0]
        for i in range(1, len(close)):
            ema[i] = close[i] * k + ema[i - 1] * (1 - k)

        recent_close = close[-self.lookback:]
        recent_ema = ema[-self.lookback:]

        crosses = 0
        for i in range(1, len(recent_close)):
            prev_above = recent_close[i - 1] > recent_ema[i - 1]
            curr_above = recent_close[i] > recent_ema[i]
            if prev_above != curr_above:
                crosses += 1
        return crosses
