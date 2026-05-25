"""
双核动量策略 (Dual Momentum Strategy)
========================================
核心思想: 结合时间动量(收益率动量)和截面动量(RSI动量), 双重确认后再入场。
严格避免未来数据泄漏: 所有指标只使用当前bar之前的数据。

统计学基础:
1. 动量效应 (Jegadeesh & Titman, 1993): 过去表现好的股票短期内继续表现好
2. RSI均值回归: 极端超卖后的反弹概率高于随机
3. ATR波动率过滤: 高波动时信号噪音大, 需要更大确认阈值

TradingView指标对应:
- MACD → 双均线动量 (本策略用简单收益率动量代替, 更直观)
- RSI → 相对强弱指数 (直接实现)
- Supertrend → ATR通道趋势判断 (直接实现)
- Bollinger Band宽度 → 波动率过滤 (用ATR替代)

信号逻辑:
- 买入: 收益率动量 > 0 且 RSI从超卖区回升 且 Supertrend看涨
- 卖出: Supertrend翻空 或 RSI超买 或 动量翻负

作者: quant_trade system
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class DualMomentumStrategy(BaseStrategy):
    """双核动量策略 — 时间动量 + RSI动量 + ATR趋势过滤"""

    strategy_description = "结合收益率动量和RSI动量的双重确认策略"
    strategy_category = "momentum"
    strategy_params_schema = {
        "momentum_period": {"type": "int", "default": 20, "label": "动量周期(天)"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_oversold": {"type": "float", "default": 35, "label": "RSI超卖阈值"},
        "rsi_overbought": {"type": "float", "default": 70, "label": "RSI超买阈值"},
        "atr_period": {"type": "int", "default": 10, "label": "ATR周期"},
        "atr_multiplier": {"type": "float", "default": 3.0, "label": "Supertrend倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.momentum_period = params.get('momentum_period', 20)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 35)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.atr_period = params.get('atr_period', 10)
        self.atr_multiplier = params.get('atr_multiplier', 3.0)

    def get_default_params(self):
        return {
            'momentum_period': 20,
            'rsi_period': 14,
            'rsi_oversold': 35,
            'rsi_overbought': 70,
            'atr_period': 10,
            'atr_multiplier': 3.0,
        }

    def generate_signals(self):
        """逐bar生成信号, 严格避免未来数据"""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        self.signals = []
        unique_times = data.index.unique()
        current_holding = None

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                # 选股: 寻找最佳买入标的
                best_stock = self._select_best(current_bars, current_time, data)
                if best_stock:
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'symbol': best_stock,
                    })
                    current_holding = best_stock
            else:
                # 检查卖出
                if self._should_sell(current_holding, current_time, data):
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'sell',
                        'symbol': current_holding,
                    })
                    current_holding = None

        print(f"DualMomentum: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _select_best(self, current_bars, current_time, full_data):
        """选出评分最高的股票"""
        best_score = -float('inf')
        best_stock = None

        for _, bar in current_bars.iterrows():
            symbol = bar['symbol']
            hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
            score, should_buy = self._evaluate(hist)
            if should_buy and score > best_score:
                best_score = score
                best_stock = symbol
        return best_stock

    def _should_sell(self, symbol, current_time, full_data):
        """卖出判断"""
        hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
        _, should_sell = self._evaluate(hist, sell_mode=True)
        return should_sell

    def _evaluate(self, data, sell_mode=False):
        """
        核心: 计算动量评分和交易信号
        所有指标只使用data中已有的数据 (data.index <= current_time)
        """
        min_len = max(self.momentum_period, self.rsi_period, self.atr_period) + 5
        if len(data) < min_len:
            return 0, False

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # === 指标计算 (全部基于历史数据, 无未来泄漏) ===

        # 1. 收益率动量: N日简单收益率
        momentum = (close[-1] - close[-self.momentum_period]) / close[-self.momentum_period]

        # 2. RSI (Wilder平滑法)
        rsi = self._calc_rsi(close)

        # 3. Supertrend方向
        st_dir = self._calc_supertrend_dir(high, low, close)

        # === 评分 ===
        score = 0.0

        # 动量评分: 正动量加分
        if momentum > 0:
            score += min(momentum * 100, 20)  # 上限20分
        else:
            score += max(momentum * 100, -20)

        # RSI评分
        if rsi < self.rsi_oversold:
            score += 15  # 超卖, 潜在反弹
        elif rsi < 50:
            score += 5   # 中性偏低
        elif rsi > self.rsi_overbought:
            score -= 15  # 超买
        else:
            score += 0   # 中性偏高

        # Supertrend趋势评分
        if st_dir == 1:
            score += 10
        else:
            score -= 10

        # === 信号判断 ===
        if sell_mode:
            # 卖出: Supertrend翻空 或 RSI超买 或 动量翻负
            should_sell = (st_dir == -1) or (rsi > self.rsi_overbought) or (momentum < -0.02)
            return score, should_sell
        else:
            # 买入三重确认: 动量正 + RSI从低位回升 + Supertrend看涨
            should_buy = (momentum > 0.01) and (rsi < 60) and (st_dir == 1)
            return score, should_buy

    def _calc_rsi(self, close):
        """Wilder平滑RSI — 只返回最后一个值"""
        period = self.rsi_period
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # 初始均值
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Wilder平滑
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calc_supertrend_dir(self, high, low, close):
        """Supertrend方向 — 只返回最后一个方向 (1=看涨, -1=看跌)"""
        period = self.atr_period
        mult = self.atr_multiplier

        if len(close) < period + 2:
            return 0

        # ATR计算
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.zeros(len(close))
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period

        hl2 = (high + low) / 2.0
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        # 递推Supertrend
        direction = 0
        supertrend = 0.0
        for i in range(period + 1, len(close)):
            if direction == 1:
                lower[i] = max(lower[i], lower[i-1]) if i > period + 1 else lower[i]
            elif direction == -1:
                upper[i] = min(upper[i], upper[i-1]) if i > period + 1 else upper[i]

            if close[i] > upper[i-1]:
                direction = 1
            elif close[i] < lower[i-1]:
                direction = -1
            # else: direction不变

            if direction == 1:
                supertrend = lower[i]
            else:
                supertrend = upper[i]

        return direction
