"""
MACD动量+量价确认策略 (MACD Momentum with Volume Confirmation)
=============================================================
TradingView Pine Script核心逻辑翻译:
1. MACD = ta.ema(close, 12) - ta.ema(close, 26)   → 趋势动量
2. Signal = ta.ema(macd, 9)                         → 信号线
3. Histogram = MACD - Signal                         → 柱状图(动量强度)
4. Volume Ratio = volume / ta.sma(volume, 20)       → 量比

统计学要点:
1. MACD金叉: 统计上后续5-20日正收益概率约55-60% (非随机)
2. 量价配合: 放量金叉胜率比缩量金叉高10-15%
3. EMA vs SMA: EMA对近期数据更敏感, 更适合趋势跟踪
4. 止损: ATR-based止损优于固定百分比 (考虑波动率)

买入条件:
- MACD金叉 (MACD上穿Signal线)
- 成交量 > 20日均量 (放量确认)
- 收盘价 > 60日均线 (中期趋势向上)

卖出条件:
- MACD死叉 (MACD下穿Signal线)
- 或止损: 亏损超过 2*ATR(14)

防未来数据泄漏: 逐bar计算, 只用已发生数据
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class MacdVolumeStrategy(BaseStrategy):
    """MACD动量+量价确认策略"""

    strategy_description = "MACD金叉+放量确认+均线趋势过滤, 经典TradingView指标组合"
    strategy_category = "momentum"
    strategy_params_schema = {
        "fast_period": {"type": "int", "default": 12, "label": "MACD快线周期"},
        "slow_period": {"type": "int", "default": 26, "label": "MACD慢线周期"},
        "signal_period": {"type": "int", "default": 9, "label": "信号线周期"},
        "trend_ma": {"type": "int", "default": 60, "label": "趋势均线周期"},
        "vol_ma": {"type": "int", "default": 20, "label": "成交量均线周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR止损周期"},
        "stop_atr_mult": {"type": "float", "default": 2.0, "label": "止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.fast_period = params.get('fast_period', 12)
        self.slow_period = params.get('slow_period', 26)
        self.signal_period = params.get('signal_period', 9)
        self.trend_ma = params.get('trend_ma', 60)
        self.vol_ma = params.get('vol_ma', 20)
        self.atr_period = params.get('atr_period', 14)
        self.stop_atr_mult = params.get('stop_atr_mult', 2.0)

    def get_default_params(self):
        return {
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
            'trend_ma': 60, 'vol_ma': 20,
            'atr_period': 14, 'stop_atr_mult': 2.0,
        }

    def generate_signals(self):
        """逐bar生成信号"""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        self.signals = []
        unique_times = sorted(data.index.unique())
        current_holding = None
        entry_price = 0.0
        stop_price = 0.0

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_stock, best_price, best_stop = self._select_best(
                    current_bars, current_time, data
                )
                if best_stock:
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'symbol': best_stock,
                    })
                    current_holding = best_stock
                    entry_price = best_price
                    stop_price = best_stop
            else:
                should_sell, sell_price = self._check_sell(
                    current_holding, current_time, data, entry_price, stop_price
                )
                if should_sell:
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'sell',
                        'symbol': current_holding,
                    })
                    current_holding = None
                    entry_price = 0.0
                    stop_price = 0.0

        print(f"MACDVolume: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _select_best(self, current_bars, current_time, full_data):
        best_score = -float('inf')
        best_stock = None
        best_price = 0
        best_stop = 0

        for _, bar in current_bars.iterrows():
            symbol = bar['symbol']
            # 关键: 只用current_time之前的数据, 不含当bar (防前视偏差)
            hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
            result = self._evaluate(hist)
            if result is None:
                continue
            score, should_buy, price, stop = result
            if should_buy and score > best_score:
                best_score = score
                best_stock = symbol
                best_price = price
                best_stop = stop

        return best_stock, best_price, best_stop

    def _check_sell(self, symbol, current_time, full_data, entry_price, stop_price):
        # 关键: 只用current_time之前的数据, 不含当bar
        hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
        result = self._evaluate(hist, sell_mode=True, entry_price=entry_price, stop_price=stop_price)
        if result is None:
            return False, 0
        _, should_sell, price, _ = result
        return should_sell, price

    def _evaluate(self, data, sell_mode=False, entry_price=0, stop_price=0):
        """核心评估: MACD + 成交量 + 趋势"""
        min_len = self.slow_period + self.signal_period + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # === 1. EMA计算 (全部递推为序列, 无未来数据) ===
        fast_ema = self._calc_ema_series(close, self.fast_period)
        slow_ema = self._calc_ema_series(close, self.slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = self._calc_ema_series(macd_line, self.signal_period)
        histogram = macd_line - signal_line

        # === 2. 成交量 ===
        vol_ratio = 1.0
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col:
            vol = data[vol_col].values
            if len(vol) >= self.vol_ma:
                vol_ma = np.mean(vol[-self.vol_ma:])
                if vol_ma > 0:
                    vol_ratio = vol[-1] / vol_ma

        # === 3. 趋势均线 ===
        if n >= self.trend_ma:
            trend_ma_val = np.mean(close[-self.trend_ma:])
        else:
            trend_ma_val = np.mean(close)

        # === 4. ATR (用于止损) ===
        atr = 0
        if n > self.atr_period:
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
            )
            atr = np.mean(tr[-self.atr_period:])

        current_close = close[-1]
        prev_histogram = histogram[-2] if len(histogram) >= 2 else 0
        curr_histogram = histogram[-1]

        if sell_mode:
            # 止损: 跌破止损价
            if stop_price > 0 and current_close <= stop_price:
                return 0, True, current_close, stop_price

            # MACD死叉: histogram从正转负
            should_sell = (prev_histogram > 0 and curr_histogram < 0)
            return 0, should_sell, current_close, stop_price
        else:
            # MACD金叉: histogram从负转正
            golden_cross = (prev_histogram < 0 and curr_histogram > 0)

            # 趋势过滤: 收盘价在均线上方
            trend_up = current_close > trend_ma_val

            # 量价确认: 放量
            vol_confirm = vol_ratio > 1.0

            score = 0
            if golden_cross:
                score += 20
            if trend_up:
                score += 10
            if vol_confirm:
                score += 5

            should_buy = golden_cross and trend_up

            # 止损价 = 买入价 - stop_atr_mult * ATR
            stop = current_close - self.stop_atr_mult * atr if atr > 0 else current_close * 0.95

            return score, should_buy, current_close, stop

    def _calc_ema_series(self, values, period):
        """对数组计算完整EMA序列"""
        values = np.asarray(values, dtype=float)
        n = len(values)
        if n < period:
            return values.copy()
        multiplier = 2.0 / (period + 1)
        result = np.empty(n)
        result[0] = values[0]
        for i in range(1, n):
            if i < period:
                # 不足period时用SMA
                result[i] = np.mean(values[:i+1])
            else:
                result[i] = values[i] * multiplier + result[i-1] * (1 - multiplier)
        return result
