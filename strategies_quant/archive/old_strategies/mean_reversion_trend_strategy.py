"""
均值回归+趋势过滤策略 (Mean Reversion with Trend Filter)
=========================================================
A股特性适配版:
1. A股以均值回归为主 (散户占比高, 情绪波动大)
2. T+1限制 → 买入后次日才能卖出, 信号需更谨慎
3. 涨跌停限制 → 极端波动时流动性风险

统计学基础:
1. Bollinger Band: 价格偏离均值N倍标准差后回归概率高 (正态分布假设)
2. RSI均值回归: RSI < 30时超卖, 统计上未来收益显著为正
3. 成交量确认: 放量下跌后的反弹信号更可靠 (量价关系)
4. 趋势过滤: 只在上升趋势中做均值回归 (不在下跌趋势中接飞刀)

TradingView Pine Script对照:
- ta.sma(close, 20) → SMA均线
- ta.rsi(close, 14) → RSI
- ta.stdev(close, 20) → 标准差 (Bollinger Band)
- ta.supertrend(3, 10) → Supertrend趋势方向
- ta.sma(volume, 20) → 成交量均线

防未来数据泄漏:
- 所有指标只用 data[data.index <= current_time]
- 无 shift(-n) 操作
- 无全量预计算
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class MeanReversionTrendStrategy(BaseStrategy):
    """均值回归+趋势过滤 — 在上升趋势中买入超卖反弹"""

    strategy_description = "Bollinger超卖+RSI确认+Supertrend趋势过滤, 适合A股均值回归特性"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "bb_period": {"type": "int", "default": 20, "label": "Bollinger周期"},
        "bb_std": {"type": "float", "default": 2.0, "label": "Bollinger标准差倍数"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_oversold": {"type": "float", "default": 30, "label": "RSI超卖线"},  # 实际买入阈值为 oversold+10=40
        "rsi_target": {"type": "float", "default": 50, "label": "RSI目标线(止盈)"},
        "atr_period": {"type": "int", "default": 10, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 3.0, "label": "Supertrend倍数"},
        "volume_ratio": {"type": "float", "default": 0.8, "label": "量比阈值(相对均量)"},
        "hold_days_min": {"type": "int", "default": 3, "label": "最少持仓天数(T+1)"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2.0)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_target = params.get('rsi_target', 50)
        self.atr_period = params.get('atr_period', 10)
        self.atr_mult = params.get('atr_mult', 3.0)
        self.volume_ratio = params.get('volume_ratio', 0.8)
        self.hold_days_min = params.get('hold_days_min', 3)

    def get_default_params(self):
        return {
            'bb_period': 20, 'bb_std': 2.0,
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_target': 50,
            'atr_period': 10, 'atr_mult': 3.0,
            'volume_ratio': 0.8, 'hold_days_min': 3,
        }

    def generate_signals(self):
        """逐bar生成信号, 严格避免未来数据"""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        self.signals = []
        unique_times = sorted(data.index.unique())
        current_holding = None
        buy_time = None

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_stock = self._select_best(current_bars, current_time, data)
                if best_stock:
                    bar = current_bars[current_bars['symbol'] == best_stock].iloc[0]
                    self._record_signal(
                        current_time, 'buy', best_stock, float(bar['close'])
                    )
                    current_holding = best_stock
                    buy_time = current_time
            else:
                # T+1: 至少持仓 hold_days_min 天
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                if days_held >= self.hold_days_min:
                    if self._should_sell(current_holding, current_time, data):
                        bar = current_bars[current_bars['symbol'] == current_holding]
                        sell_price = float(bar.iloc[0]['close']) if len(bar) > 0 else 0
                        self._record_signal(
                            current_time, 'sell', current_holding, sell_price
                        )
                        current_holding = None
                        buy_time = None

        print(f"MeanReversion: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _select_best(self, current_bars, current_time, full_data):
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
        hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
        _, should_sell = self._evaluate(hist, sell_mode=True)
        return should_sell

    def _evaluate(self, data, sell_mode=False):
        """核心评估: Bollinger + RSI + Supertrend + 成交量"""
        min_len = max(self.bb_period, self.rsi_period, self.atr_period) + 5
        if len(data) < min_len:
            return 0, False

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # 只用到最后一个完整bar之前的数据 (不含当bar, 防止用close当open)
        # 但因为是日线, close[-1]在当日收盘时已知, 下一日开盘执行, 所以可以用
        n = len(close)

        # === 1. Bollinger Band ===
        bb_ma = np.mean(close[-self.bb_period:])
        bb_std_val = np.std(close[-self.bb_period:], ddof=1)
        bb_lower = bb_ma - self.bb_std * bb_std_val
        bb_upper = bb_ma + self.bb_std * bb_std_val
        current_close = close[-1]
        bb_position = (current_close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        # === 2. RSI ===
        rsi = self._calc_rsi(close)

        # === 3. Supertrend方向 ===
        st_dir = self._calc_supertrend_dir(high, low, close)

        # === 4. 成交量确认 (可选) ===
        vol_signal = True
        if 'vol' in data.columns or 'volume' in data.columns:
            vol_col = 'vol' if 'vol' in data.columns else 'volume'
            vol = data[vol_col].values
            if len(vol) >= self.bb_period:
                vol_ma = np.mean(vol[-self.bb_period:])
                current_vol = vol[-1]
                vol_signal = current_vol >= vol_ma * self.volume_ratio

        # === 评分 ===
        score = 0.0

        # Bollinger位置评分: 越接近下轨越好(买入), 接近上轨不好
        if bb_position < 0.1:
            score += 20   # 严重超卖
        elif bb_position < 0.3:
            score += 10   # 偏低
        elif bb_position > 0.9:
            score -= 20   # 严重超买
        elif bb_position > 0.7:
            score -= 10   # 偏高

        # RSI评分
        if rsi < self.rsi_oversold:
            score += 15
        elif rsi < 40:
            score += 5
        elif rsi > 70:
            score -= 15

        # 趋势过滤评分
        if st_dir == 1:
            score += 10  # 上升趋势中做均值回归
        elif st_dir == -1:
            score -= 5   # 下降趋势中谨慎

        # === 信号判断 ===
        if sell_mode:
            # 止盈: RSI达到目标线 或 价格触及Bollinger上轨
            # 止损: Supertrend翻空
            should_sell = (rsi > self.rsi_target) or (bb_position > 0.8) or (st_dir == -1)
            return score, should_sell
        else:
            # 买入: Bollinger下轨附近 + RSI偏低 + 成交量确认
            # 趋势过滤: st_dir==1加分但不是硬性要求 (A股超卖反弹常在趋势转折点)
            should_buy = (
                (bb_position < 0.3) and            # 价格在Bollinger下30%区间
                (rsi < self.rsi_oversold + 10) and # RSI偏低(放宽到40)
                vol_signal                         # 成交量确认
            )
            # 趋势下跌时需要更严格的超卖条件
            if should_buy and st_dir == -1:
                should_buy = (bb_position < 0.15) and (rsi < 25)
            return score, should_buy

    def _calc_rsi(self, close):
        """Wilder平滑RSI"""
        period = self.rsi_period
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calc_supertrend_dir(self, high, low, close):
        """Supertrend方向 (1=看涨, -1=看跌)"""
        period = self.atr_period
        mult = self.atr_mult

        if len(close) < period + 2:
            return 0

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

        direction = 0
        for i in range(period + 1, len(close)):
            if direction == 1 and i > period + 1:
                lower[i] = max(lower[i], lower[i-1])
            elif direction == -1 and i > period + 1:
                upper[i] = min(upper[i], upper[i-1])

            if close[i] > upper[i-1]:
                direction = 1
            elif close[i] < lower[i-1]:
                direction = -1

        return direction
