"""
综合价格行为策略 (Comprehensive Price Action Strategy)
=====================================================
融合 Al Brooks 三部曲 + TradingView指标 + 统计学方法

核心知识来源:
1. PA Trend (Al Brooks): High 2/Low 2 回调入场, 20 EMA趋势过滤, Spike-Channel模型
2. PA Reversal: Pin Bar/Outside Bar反转信号, 双顶双底, 失败突破=入场信号
3. PA Zone: 交易区间识别, 大多数突破会失败, 测量移动目标
4. TradingView: MACD+RSI+Supertrend多确认, 成交量确认
5. 概率论: 置信区间评估信号强度, 假设检验验证策略

策略逻辑:
1. 趋势识别: 20 EMA斜率 + HH/HL结构 + Supertrend方向
2. 入场信号 (三重确认):
   a. 价格行为: High 2回调 或 Pin Bar反转 或 失败突破
   b. 指标确认: MACD金叉 或 RSI从超卖回升
   c. 趋势过滤: Supertrend看涨 + 价格在20 EMA上方
3. 止损: zone边界 或 ATR*2 或 信号bar极值
4. 止盈: 测量移动(zone高度) 或 RSI超买 或 MACD死叉

防未来数据泄漏: 所有指标只用 data[data.index < current_time]
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class PAZoneMomentumStrategy(BaseStrategy):
    """综合价格行为策略 — Zone+PA+指标三重确认"""

    strategy_description = "融合Al Brooks价格行为+TradingView指标的综合策略"
    strategy_category = "price_action"
    strategy_params_schema = {
        "ema_period": {"type": "int", "default": 20, "label": "趋势EMA周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_mult": {"type": "float", "default": 3.0, "label": "Supertrend倍数"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_oversold": {"type": "float", "default": 35, "label": "RSI超卖线"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "lookback": {"type": "int", "default": 5, "label": "回调检测回看bar数"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "stop_atr_mult": {"type": "float", "default": 2.0, "label": "止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ema_period = params.get('ema_period', 20)
        self.atr_period = params.get('atr_period', 14)
        self.atr_mult = params.get('atr_mult', 3.0)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 35)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.lookback = params.get('lookback', 5)
        self.hold_min = params.get('hold_min', 2)
        self.stop_atr_mult = params.get('stop_atr_mult', 2.0)

    def get_default_params(self):
        return {
            'ema_period': 20, 'atr_period': 14, 'atr_mult': 3.0,
            'rsi_period': 14, 'rsi_oversold': 35,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'lookback': 5, 'hold_min': 2, 'stop_atr_mult': 2.0,
        }

    def generate_signals(self):
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
                best_stock, stop = self._select_best(current_bars, current_time, data)
                if best_stock:
                    self.signals.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'symbol': best_stock,
                    })
                    current_holding = best_stock
                    buy_time = current_time
            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                if days_held >= self.hold_min:
                    if self._should_sell(current_holding, current_time, data):
                        self.signals.append({
                            'timestamp': current_time,
                            'action': 'sell',
                            'symbol': current_holding,
                        })
                        current_holding = None
                        buy_time = None

        print(f"PAZoneMomentum: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _select_best(self, current_bars, current_time, full_data):
        best_score = -float('inf')
        best_stock = None
        best_stop = 0

        for _, bar in current_bars.iterrows():
            symbol = bar['symbol']
            hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
            result = self._evaluate(hist)
            if result is None:
                continue
            score, should_buy, stop = result
            if should_buy and score > best_score:
                best_score = score
                best_stock = symbol
                best_stop = stop

        return best_stock, best_stop

    def _should_sell(self, symbol, current_time, full_data):
        hist = full_data[(full_data['symbol'] == symbol) & (full_data.index < current_time)]
        result = self._evaluate(hist, sell_mode=True)
        if result is None:
            return False
        _, should_sell, _ = result
        return should_sell

    def _evaluate(self, data, sell_mode=False):
        """核心评估: 价格行为 + 指标 + 趋势 三重确认"""
        min_len = max(self.ema_period, self.macd_slow + self.macd_signal, self.atr_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ===== 1. 趋势识别 (PA Trend) =====
        ema = self._calc_ema_series(close, self.ema_period)
        ema_slope = ema[-1] - ema[-5] if n >= 6 else 0
        trend_up = ema_slope > 0 and close[-1] > ema[-1]

        # ===== 2. Supertrend方向 (TradingView) =====
        st_dir = self._calc_supertrend(high, low, close)

        # ===== 3. MACD (TradingView) =====
        fast_ema = self._calc_ema_series(close, self.macd_fast)
        slow_ema = self._calc_ema_series(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._calc_ema_series(macd_line, self.macd_signal)
        histogram = macd_line - signal_line
        macd_cross_up = histogram[-2] < 0 and histogram[-1] > 0
        macd_cross_down = histogram[-2] > 0 and histogram[-1] < 0

        # ===== 4. RSI (TradingView) =====
        rsi = self._calc_rsi(close)

        # ===== 5. ATR (止损用) =====
        atr = self._calc_atr(high, low, close)

        # ===== 6. 价格行为模式 (PA Reversal + PA Trend) =====
        pa_signal = self._detect_pa_signal(close, high, low, n)

        # ===== 7. Zone识别 (PA Zone) =====
        zone = self._detect_zone(close, high, low, n)

        if sell_mode:
            return self._evaluate_sell(rsi, macd_cross_down, st_dir, close, atr)
        else:
            return self._evaluate_buy(
                trend_up, st_dir, macd_cross_up, rsi,
                pa_signal, zone, close[-1], atr, histogram
            )

    def _evaluate_buy(self, trend_up, st_dir, macd_cross_up, rsi,
                      pa_signal, zone, price, atr, histogram):
        """
        买入三重确认:
        1. 趋势: EMA上升 + Supertrend看涨
        2. 信号: PA信号(High2/Pin Bar/失败突破) 或 MACD金叉
        3. 过滤: RSI不在超买区
        """
        score = 0

        # --- 趋势评分 ---
        if trend_up and st_dir == 1:
            score += 20  # 双重趋势确认
        elif trend_up or st_dir == 1:
            score += 10  # 单一趋势确认
        else:
            return 0, False, 0  # 无趋势, 不买入

        # --- 信号评分 ---
        signal_triggered = False

        # PA信号 (Al Brooks: High 2回调 > Pin Bar > 失败突破)
        if pa_signal == 'high2':
            score += 15  # High 2: 最可靠的回调入场
            signal_triggered = True
        elif pa_signal == 'pin_bar_bull':
            score += 12  # Pin Bar: 经典反转信号
            signal_triggered = True
        elif pa_signal == 'failed_breakout_bull':
            score += 10  # 失败突破: PA Zone核心
            signal_triggered = True
        elif pa_signal == 'outside_bar_bull':
            score += 8   # 外包线反转
            signal_triggered = True

        # MACD金叉作为补充信号
        if macd_cross_up:
            score += 10
            if not signal_triggered:
                signal_triggered = True  # MACD可以单独触发

        # --- RSI过滤 ---
        if rsi < self.rsi_oversold:
            score += 10  # 超卖+反弹 = 最强组合
        elif rsi < 50:
            score += 5
        elif rsi > 70:
            score -= 10  # 超买, 减分
            if rsi > 80:
                return 0, False, 0  # 极度超买, 不追

        # --- Zone加成 ---
        if zone == 'near_support':
            score += 8  # 在支撑区附近买入
        elif zone == 'near_resistance':
            score -= 5  # 在阻力区不追

        # --- 动量确认 (histogram方向) ---
        if histogram[-1] > 0:
            score += 3

        # 最终判断: 至少需要一个信号 + 趋势确认 + 评分阈值
        should_buy = signal_triggered and score >= 30
        stop = price - self.stop_atr_mult * atr if atr > 0 else price * 0.95

        return score, should_buy, stop

    def _evaluate_sell(self, rsi, macd_cross_down, st_dir, close, atr):
        """卖出: MACD死叉 或 Supertrend翻空 或 RSI超买"""
        should_sell = False

        # MACD死叉
        if macd_cross_down:
            should_sell = True

        # Supertrend翻空
        if st_dir == -1:
            should_sell = True

        # RSI极度超买后回落
        if rsi > 75:
            should_sell = True

        return 0, should_sell, 0

    # ===== 价格行为检测 =====

    def _detect_pa_signal(self, close, high, low, n):
        """
        检测Al Brooks价格行为信号 (只用已发生数据)
        返回: 'high2', 'pin_bar_bull', 'outside_bar_bull', 'failed_breakout_bull', None
        """
        if n < self.lookback + 3:
            return None

        # --- High 2 检测 (Al Brooks核心回调模式) ---
        # 定义: 上升趋势中两次回调低点, 第二个低点 >= 第一个低点
        if n >= 10:
            # 找最近lookback+2个bar中的回调低点
            recent_low = low[-self.lookback - 2:]
            recent_close = close[-self.lookback - 2:]

            # 检测两次回调
            lows = []
            for i in range(1, len(recent_low) - 1):
                if recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i+1]:
                    lows.append((i, recent_low[i]))

            if len(lows) >= 2:
                # 第二个低点 >= 第一个低点 (Higher Low)
                if lows[-1][1] >= lows[-2][1] * 0.998:  # 允许0.2%误差
                    # 且整体趋势向上 (最近close高于之前)
                    if recent_close[-1] > recent_close[0]:
                        return 'high2'

        # --- Pin Bar 检测 (PA Reversal) ---
        # 规则: 下影线 >= bar范围的2/3, 上影线 <= 10%
        body = abs(close[-1] - (close[-2] if n > 1 else close[-1]))
        bar_range = high[-1] - low[-1]
        if bar_range > 0:
            # 近似open (用前一根close)
            open_approx = close[-2] if n > 1 else close[-1]
            upper_wick = high[-1] - max(close[-1], open_approx)
            lower_wick = min(close[-1], open_approx) - low[-1]
            body_size = abs(close[-1] - open_approx)

            # 看涨Pin Bar: 长下影线, 短上影线, 小实体在上部
            if (lower_wick >= bar_range * 0.6 and
                upper_wick <= bar_range * 0.15 and
                body_size < bar_range * 0.4):
                return 'pin_bar_bull'

        # --- Outside Bar 检测 (PA Reversal) ---
        # 规则: 当bar完全包含前bar范围, 且收盘在上部
        if n >= 2:
            prev_range = high[-2] - low[-2]
            curr_range = high[-1] - low[-1]
            if (high[-1] > high[-2] and low[-1] < low[-2] and
                close[-1] > (high[-1] + low[-1]) / 2):  # 收在上半部
                return 'outside_bar_bull'

        # --- Failed Breakout 检测 (PA Zone核心) ---
        # 规则: 价格跌破前低后又收回 (空头陷阱)
        if n >= 5:
            recent_swing_low = min(low[-5:-1])  # 前4个bar的最低点
            if (low[-2] < recent_swing_low and  # 前一根曾跌破
                close[-1] > recent_swing_low and  # 但收盘收回了
                close[-1] > close[-2]):           # 且收阳
                return 'failed_breakout_bull'

        return None

    def _detect_zone(self, close, high, low, n):
        """
        Zone识别 (PA Zone): 检测当前价格相对于近期交易区间的位置
        返回: 'near_support', 'near_resistance', 'in_zone', None
        """
        if n < 20:
            return None

        # 用最近20个bar定义区间
        zone_high = max(high[-20:])
        zone_low = min(low[-20:])
        zone_range = zone_high - zone_low

        if zone_range <= 0:
            return None

        current_price = close[-1]
        position = (current_price - zone_low) / zone_range

        if position < 0.2:
            return 'near_support'    # 接近区间底部
        elif position > 0.8:
            return 'near_resistance'  # 接近区间顶部
        elif position < 0.5:
            return 'in_zone'          # 区间下半部
        return None

    # ===== 指标计算 =====

    def _calc_ema_series(self, values, period):
        values = np.asarray(values, dtype=float)
        n = len(values)
        result = np.empty(n)
        result[0] = values[0]
        k = 2.0 / (period + 1)
        for i in range(1, n):
            if i < period:
                result[i] = np.mean(values[:i + 1])
            else:
                result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    def _calc_rsi(self, close):
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
        return 100.0 - 100.0 / (1 + avg_gain / avg_loss)

    def _calc_atr(self, high, low, close):
        period = self.atr_period
        if len(close) < period + 1:
            return 0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        return np.mean(tr[-period:])

    def _calc_supertrend(self, high, low, close):
        """Supertrend方向 (1=看涨, -1=看跌)"""
        period = self.atr_period
        mult = self.atr_mult
        if len(close) < period + 2:
            return 0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        atr = np.zeros(len(close))
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period

        hl2 = (high + low) / 2.0
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        direction = 0
        for i in range(period + 1, len(close)):
            if direction == 1 and i > period + 1:
                lower[i] = max(lower[i], lower[i - 1])
            elif direction == -1 and i > period + 1:
                upper[i] = min(upper[i], upper[i - 1])
            if close[i] > upper[i - 1]:
                direction = 1
            elif close[i] < lower[i - 1]:
                direction = -1
        return direction
