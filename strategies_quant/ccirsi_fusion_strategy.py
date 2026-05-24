"""
CCI-RSI融合背离策略 (CCI-RSI Merged Divergence Strategy)
========================================================
基于CCI+RSI双振荡器背离 + Heikin-Ashi降噪 + 趋势过滤。

来源: TradingView Tier 2 — SHK CCI RSI Merged HA Signal MA Dual Divergence

核心逻辑:
  1. 双振荡器:
     - CCI(20): 价格偏离统计均值的程度
     - RSI(14): 相对强弱, 0-100范围
     - 融合: 两者都确认背离才触发信号
  2. 背离检测:
     - 看涨背离: 价格新低, CCI不新低 AND RSI不新低
     - 看跌背离: 价格新高, CCI不新高 AND RSI不新高
  3. Heikin-Ashi降噪:
     - HA-C = (O+H+L+C)/4 — 平滑价格
     - HA-O = 前一根HA的(O+C)/2 — 惯性
     - HA趋势判断: 连续HA阳线=上升趋势
  4. 信号:
     - 买入: 看涨背离 + HA阳线 + 价格在MA上方
     - 卖出: 看跌背离 + ATR追踪止损

WHY this works:
  - 双振荡器独立确认: CCI衡量偏离度, RSI衡量速度, 两者不等价
  - 背离是唯一"领先"信号: 大多数指标是滞后的, 但背离预测反转
  - Heikin-Ashi去除噪音: 保留趋势信息, 过滤单bar假信号
  - "市场在说谎时就是交易机会": 背离=价格和动量不一致=即将回归一致

数学核心:
  CCI = (TP - SMA(TP)) / (0.015 × MAD), MAD=mean absolute deviation
  RSI = 100 - 100/(1 + avg_gain/avg_loss)
  背离: price_new_low ∧ (CCI > prev_CCI_low) ∧ (RSI > prev_RSI_low)
哲学: 真理是自洽的 — 价格和动量应该一致. 不一致=谎言=机会
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class CCIRSIFusionStrategy(BaseStrategy):
    """CCI-RSI融合背离策略 — 双振荡器背离+HA降噪+趋势过滤+ATR止损"""

    strategy_description = "CCI-RSI融合: 双振荡器背离确认+HA降噪+MA趋势过滤+ATR止损"
    strategy_category = "oscillator"
    strategy_params_schema = {
        "cci_period": {"type": "int", "default": 20, "label": "CCI周期"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "pivot_lookback": {"type": "int", "default": 20, "label": "背离检测回看"},
        "trend_ma": {"type": "int", "default": 50, "label": "趋势MA"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def _cci(self, H, L, C, period):
        """CCI计算"""
        tp = (H + L + C) / 3.0
        tp_s = pd.Series(tp)
        tp_ma = tp_s.rolling(period).mean().values
        mad = tp_s.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
        mad = np.where(np.isnan(mad) | (mad < 1e-8), 1e-8, mad)
        cci = (tp - tp_ma) / (0.015 * mad)
        return cci

    def _rsi(self, C, period):
        """RSI计算"""
        delta = np.diff(C, prepend=C[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
        avg_loss = np.where(avg_loss < 1e-8, 1e-8, avg_loss)
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)
        return rsi

    def _heikin_ashi(self, O, H, L, C):
        """Heikin-Ashi计算"""
        n = len(C)
        ha_c = np.zeros(n)
        ha_o = np.zeros(n)
        ha_c[0] = (O[0] + H[0] + L[0] + C[0]) / 4
        ha_o[0] = O[0]
        for i in range(1, n):
            ha_c[i] = (O[i] + H[i] + L[i] + C[i]) / 4
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
        return ha_o, ha_c

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        O = df['open'].values
        n = len(C)

        # 1. 振荡器
        cci = self._cci(H, L, C, p['cci_period'])
        rsi = self._rsi(C, p['rsi_period'])

        # 2. Heikin-Ashi
        ha_o, ha_c = self._heikin_ashi(O, H, L, C)
        ha_bullish = ha_c > ha_o  # HA阳线

        # 3. 趋势MA
        trend = pd.Series(C).rolling(p['trend_ma']).mean().values

        # 4. ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 5. 背离检测
        lb = p['pivot_lookback']
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)

        for i in range(lb, n):
            window = slice(i - lb, i + 1)
            price_min_idx = i - lb + np.argmin(C[window])
            price_max_idx = i - lb + np.argmax(C[window])

            # 看涨背离: 价格在近期新低附近, 但CCI/RSI不在新低
            if price_min_idx >= i - 3:  # 近3根内出现价格新低
                prev_low = np.min(C[i-lb:i-3]) if i > lb + 3 else C[0]
                curr_low = C[price_min_idx]
                if curr_low <= prev_low:  # 价格创新低
                    # CCI和RSI都没有创新低
                    cci_at_low = cci[price_min_idx]
                    cci_prev_min = np.min(cci[i-lb:i-3]) if i > lb + 3 else cci[0]
                    rsi_at_low = rsi[price_min_idx]
                    rsi_prev_min = np.min(rsi[i-lb:i-3]) if i > lb + 3 else rsi[0]
                    if cci_at_low > cci_prev_min and rsi_at_low > rsi_prev_min:
                        bullish_div[i] = True

            # 看跌背离: 价格在近期新高附近, 但CCI/RSI不在新高
            if price_max_idx >= i - 3:
                prev_high = np.max(C[i-lb:i-3]) if i > lb + 3 else C[0]
                curr_high = C[price_max_idx]
                if curr_high >= prev_high:
                    cci_at_high = cci[price_max_idx]
                    cci_prev_max = np.max(cci[i-lb:i-3]) if i > lb + 3 else cci[0]
                    rsi_at_high = rsi[price_max_idx]
                    rsi_prev_max = np.max(rsi[i-lb:i-3]) if i > lb + 3 else rsi[0]
                    if cci_at_high < cci_prev_max and rsi_at_high < rsi_prev_max:
                        bearish_div[i] = True

        # 6. 信号生成
        signals = []
        in_pos = False
        trail_stop = 0.0
        entry_idx = 0

        warmup = max(p['pivot_lookback'], p['trend_ma'], p['cci_period'], p['rsi_period'], p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if not in_pos:
                # 买入: 看涨背离 + HA阳线 + 价格在趋势上方
                if (bullish_div[i] and ha_bullish[i] and
                    not np.isnan(trend[i]) and price > trend[i]):
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
                elif bearish_div[i] and hold_days >= p['hold_min']:
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
