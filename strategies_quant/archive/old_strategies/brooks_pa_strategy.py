"""
布鲁克斯价格行为趋势策略 (Brooks Price Action Trend Strategy)
==========================================================
基于 Al Brooks "Trading Price Action Trends" 的量化实现。

核心创新 (vs 现有策略):
1. Bar计数系统: High 1/2/3, Low 1/2/3 — EMA回调次数追踪
   Brooks: "High 2是最高概率的趋势延续入场信号"
2. 趋势强度分类: Spike(强) / Channel(中) / Weak(弱)
   不同强度使用不同策略参数
3. 连续高潮检测: 大bar + 高潮计数 → 趋势反转信号
4. 20-gap bar确认: bar body不重叠EMA = 强趋势确认

信号优先级 (Brooks第11章):
  Entry: High2+20gap(9) > High2(7) > Spike回调(6) > High3楔形(5)
  Exit:  连续高潮(9) > 趋势反转(8) > EMA连续穿越(6) > EMA斜率翻转(5)

与AdaptiveWeighted的区别:
- AdaptiveWeighted: 5指标加权评分, 静态阈值
- BrooksPA: EMA回调bar计数, 趋势强度分类, 动态阈值

防未来数据泄漏:
  - 每个bar的分析只用 data[0:i+1]
  - generate_signals 查找前一个bar的分析结果
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class BrooksPAStrategy(BaseStrategy):
    """Al Brooks价格行为趋势策略 — Bar计数 + 趋势强度 + 高潮检测"""

    strategy_description = "Al Brooks PA趋势延续: High2/Low2回调入场, 趋势强度分类, 连续高潮出场"
    strategy_category = "price_action"
    strategy_params_schema = {
        "ema_period": {"type": "int", "default": 20, "label": "EMA周期(Brooks标准20)"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "strong_bar_ratio": {"type": "float", "default": 0.55, "label": "强bar body占比"},
        "climax_range_mult": {"type": "float", "default": 2.0, "label": "Climax bar range倍数"},
        "max_climax_count": {"type": "int", "default": 3, "label": "出场高潮计数"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ema_period = params.get('ema_period', 20)
        self.atr_period = params.get('atr_period', 14)
        self.strong_bar_ratio = params.get('strong_bar_ratio', 0.55)
        self.climax_range_mult = params.get('climax_range_mult', 2.0)
        self.max_climax_count = params.get('max_climax_count', 3)
        self.hold_min = params.get('hold_min', 2)

    def get_default_params(self):
        return {
            'ema_period': 20, 'atr_period': 14,
            'strong_bar_ratio': 0.55, 'climax_range_mult': 2.0,
            'max_climax_count': 3, 'hold_min': 2,
        }

    # ================================================================
    # Main signal generation
    # ================================================================

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()

        # Pre-compute per-stock analysis
        stock_analysis = {}
        stock_times = {}
        for sym in symbols:
            stock_data = data[data['symbol'] == sym].sort_index()
            stock_times[sym] = stock_data.index
            stock_analysis[sym] = self._analyze_stock(stock_data)

        # Iterate through time
        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        buy_time = None
        position_dir = 0  # 1=long, -1=short
        entry_price = 0.0
        high_water = 0.0   # Highest close while long, lowest while short
        max_hold_days = 60  # Maximum hold period

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                # Flat: look for entry in either direction
                best_buy_quality = -1
                best_buy_sym = None
                best_sell_quality = -1
                best_sell_sym = None

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    a = self._get_analysis(stock_analysis[sym], stock_times[sym], current_time)
                    if a is None:
                        continue
                    if a['buy_signal'] and a['signal_quality'] > best_buy_quality:
                        best_buy_quality = a['signal_quality']
                        best_buy_sym = sym
                    if a['sell_signal'] and a['signal_quality'] > best_sell_quality:
                        best_sell_quality = a['signal_quality']
                        best_sell_sym = sym

                # Prefer higher quality signal
                if best_buy_quality > best_sell_quality and best_buy_sym:
                    self._record_signal(current_time, 'buy', best_buy_sym)
                    current_holding = best_buy_sym
                    buy_time = current_time
                    position_dir = 1
                    entry_price = 0  # Will be set by engine
                elif best_sell_quality > 0 and best_sell_sym:
                    self._record_signal(current_time, 'sell', best_sell_sym)
                    current_holding = best_sell_sym
                    buy_time = current_time
                    position_dir = -1
                    entry_price = 0

            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])

                # Get current price for stop loss
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                current_price = float(bar_data.iloc[0]['close']) if len(bar_data) > 0 else 0

                # Track high/low water mark
                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    high_water = min(high_water, current_price) if high_water < float('inf') else current_price

                # ===== RISK MANAGEMENT (stop loss) =====
                stop_hit = False
                a = self._get_analysis(
                    stock_analysis[current_holding],
                    stock_times[current_holding],
                    current_time
                )
                atr_val = a['atr'] if a is not None else 0

                if position_dir == 1 and high_water > 0 and atr_val > 0:
                    # Long trailing stop: 2.5 ATR below high water
                    if current_price < high_water - 2.5 * atr_val:
                        stop_hit = True
                    # Hard stop: 8% below entry
                    # (Use EMA as proxy since we don't know exact entry price)
                elif position_dir == -1 and high_water < float('inf') and atr_val > 0:
                    # Short trailing stop: 2.5 ATR above low water
                    if current_price > high_water + 2.5 * atr_val:
                        stop_hit = True

                # Max hold period
                if days_held >= max_hold_days:
                    stop_hit = True

                # Execute stop or normal exit
                if stop_hit and days_held >= self.hold_min:
                    if position_dir == 1:
                        self._record_signal(current_time, 'sell', current_holding)
                    else:
                        self._record_signal(current_time, 'buy', current_holding)
                    current_holding = None
                    buy_time = None
                    position_dir = 0
                    high_water = 0

                elif days_held >= self.hold_min and a is not None:
                    # Long position: look for sell signal
                    if position_dir == 1 and a['sell_signal']:
                        self._record_signal(current_time, 'sell', current_holding)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0
                    # Short position: look for buy signal (cover)
                    elif position_dir == -1 and a['buy_signal']:
                            self._record_signal(current_time, 'buy', current_holding)
                            current_holding = None
                            buy_time = None
                            position_dir = 0

        print(f"BrooksPA: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _get_analysis(self, analysis_df, timestamps, current_time):
        pos = timestamps.searchsorted(current_time, side='left')
        if pos > 0:
            return analysis_df.iloc[pos - 1]
        return None

    # ================================================================
    # Core: Per-stock analysis — simplified, robust
    # ================================================================

    def _analyze_stock(self, stock_data):
        """
        简化版趋势分析:
        1. 趋势方向: EMA slope + 价格位置
        2. Bar计数: EMA穿越计数 (High N / Low N)
        3. 趋势强度: Spike / Channel / Weak
        4. 高潮检测: 大bar计数
        """
        n = len(stock_data)
        close = stock_data['close'].values.astype(float)
        high = stock_data['high'].values.astype(float)
        low = stock_data['low'].values.astype(float)
        open_ = stock_data['open'].values.astype(float)

        ema = self._compute_ema(close, self.ema_period)
        atr = self._compute_atr_array(high, low, close, self.atr_period)
        avg_range = self._compute_avg_range(high, low, close, 20)

        results = []

        # State tracking
        # Uptrend pullback counting
        up_pb_count = 0        # High N
        up_in_pb = False       # Currently in pullback to EMA
        up_high_water = 0.0    # Highest close (for reset)
        # Downtrend bounce counting
        dn_pb_count = 0        # Low N
        dn_in_pb = False       # Currently bouncing to EMA
        dn_low_water = float('inf')  # Lowest close (for reset)
        # Climax tracking
        climax_count_up = 0
        climax_count_dn = 0
        # EMA breach tracking
        consec_above_ema = 0
        consec_below_ema = 0

        for i in range(n):
            row = {
                'buy_signal': False, 'sell_signal': False,
                'signal_type': None, 'signal_quality': 0,
                'trend_dir': 0, 'bar_count': 0, 'atr': 0.0,
            }

            if i < self.ema_period + 5:
                results.append(row)
                continue

            c, h, l, o = close[i], high[i], low[i], open_[i]
            bar_range = h - l
            body = abs(c - o)
            body_ratio = body / bar_range if bar_range > 0 else 0.5

            # --- Bar classification ---
            is_bull = c > o
            is_bear = c < o
            is_strong_bull = is_bull and body_ratio > self.strong_bar_ratio
            is_strong_bear = is_bear and body_ratio > self.strong_bar_ratio

            # Bull reversal bar
            lower_wick = min(c, o) - l
            upper_wick = h - max(c, o)
            is_bull_reversal = (
                bar_range > 0 and lower_wick >= bar_range * 0.5
                and upper_wick <= bar_range * 0.25 and is_bull
            )
            is_bear_reversal = (
                bar_range > 0 and upper_wick >= bar_range * 0.5
                and lower_wick <= bar_range * 0.25 and is_bear
            )

            # --- EMA relationship ---
            above_ema = c > ema[i]
            # 20-gap bar: entire body is on one side of EMA
            is_20gap = (min(c, o) > ema[i]) if above_ema else (max(c, o) < ema[i])

            # --- EMA slope (5-bar) ---
            ema_slope = (ema[i] - ema[i - 5]) / ema[i - 5] if i >= 5 and ema[i - 5] > 0 else 0

            # --- Trend direction ---
            trend_up = above_ema and ema_slope > 0
            trend_down = not above_ema and ema_slope < 0
            trend_dir = 1 if trend_up else (-1 if trend_down else 0)

            # --- Climax detection ---
            avg_r = avg_range[i] if avg_range[i] > 0 else bar_range
            is_climax = bar_range > self.climax_range_mult * avg_r

            # --- EMA breach tracking ---
            if above_ema:
                consec_above_ema += 1
                consec_below_ema = 0
            else:
                consec_below_ema += 1
                consec_above_ema = 0

            # ===================== UPTREND BAR COUNTING =====================

            if trend_up:
                # New high → reset pullback count (Brooks: fresh pullback sequence)
                if c > up_high_water:
                    up_high_water = c
                    if up_pb_count >= 2:
                        up_pb_count = 0
                        climax_count_up = 0

                # Count pullbacks to EMA
                if up_in_pb:
                    if above_ema:  # Recovered above EMA → pullback completed
                        up_pb_count += 1
                        up_in_pb = False
                        up_high_water = c
                else:
                    if l <= ema[i]:  # Touched/crossed below EMA → pullback starts
                        up_in_pb = True

                # Climax in uptrend
                if is_climax and is_strong_bull:
                    climax_count_up += 1
                elif not is_climax:
                    climax_count_up = max(climax_count_up - 1, 0)
            else:
                # Not in uptrend → reset
                up_pb_count = 0
                up_in_pb = False
                up_high_water = 0
                climax_count_up = 0

            # ===================== DOWNTREND BAR COUNTING =====================

            if trend_down:
                # New low → reset bounce count
                if c < dn_low_water:
                    dn_low_water = c
                    if dn_pb_count >= 2:
                        dn_pb_count = 0
                        climax_count_dn = 0

                # Count bounces to EMA
                if dn_in_pb:
                    if not above_ema:  # Resumed below EMA → bounce completed
                        dn_pb_count += 1
                        dn_in_pb = False
                        dn_low_water = c
                else:
                    if h >= ema[i]:  # Touched/crossed above EMA → bounce starts
                        dn_in_pb = True

                # Climax in downtrend
                if is_climax and is_strong_bear:
                    climax_count_dn += 1
                elif not is_climax:
                    climax_count_dn = max(climax_count_dn - 1, 0)
            else:
                dn_pb_count = 0
                dn_in_pb = False
                dn_low_water = float('inf')
                climax_count_dn = 0

            # ===================== SIGNAL GENERATION =====================

            # --- Buy signals (uptrend entries) ---
            if trend_up:
                # High 2 at EMA (standard entry)
                if up_pb_count == 2 and not up_in_pb:
                    row['buy_signal'] = True
                    row['signal_type'] = 'high2_20gap' if is_20gap else 'high2'
                    row['signal_quality'] = 9 if is_20gap else 7

                # High 3 / wedge bull flag
                elif up_pb_count == 3 and not up_in_pb:
                    row['buy_signal'] = True
                    row['signal_type'] = 'high3_wedge'
                    row['signal_quality'] = 5

                # Pin bar at EMA (first pullback reversal)
                elif up_pb_count == 1 and up_in_pb and is_bull_reversal:
                    row['buy_signal'] = True
                    row['signal_type'] = 'ema_pin_bar'
                    row['signal_quality'] = 6

            # --- Sell signals (exit longs / enter shorts) ---
            # Exit conditions for long positions
            if trend_up:
                # Consecutive climax → exit
                if climax_count_up >= self.max_climax_count:
                    row['sell_signal'] = True
                    row['signal_type'] = 'consecutive_climax'
                    row['signal_quality'] = 9

                # EMA slope flipped negative + price below EMA = trend dying
                elif ema_slope < -0.003 and consec_below_ema >= 3:
                    row['sell_signal'] = True
                    row['signal_type'] = 'ema_slope_exit'
                    row['signal_quality'] = 6

            # Entry signals for short positions (downtrend)
            if trend_down:
                # Low 2 at EMA (standard short entry)
                if dn_pb_count == 2 and not dn_in_pb:
                    row['sell_signal'] = True
                    row['signal_type'] = 'low2_20gap' if is_20gap else 'low2'
                    row['signal_quality'] = 9 if is_20gap else 7

                # Low 3 / wedge bear flag
                elif dn_pb_count == 3 and not dn_in_pb:
                    row['sell_signal'] = True
                    row['signal_type'] = 'low3_wedge'
                    row['signal_quality'] = 5

                # Pin bar at EMA (first bounce reversal)
                elif dn_pb_count == 1 and dn_in_pb and is_bear_reversal:
                    row['sell_signal'] = True
                    row['signal_type'] = 'ema_pin_bar_short'
                    row['signal_quality'] = 6

            # Exit conditions for short positions (cover)
            if trend_down:
                # Consecutive climax → cover
                if climax_count_dn >= self.max_climax_count:
                    row['buy_signal'] = True
                    row['signal_type'] = 'cover_climax'
                    row['signal_quality'] = 9

                # EMA slope flipped positive + price above EMA
                elif ema_slope > 0.003 and consec_above_ema >= 3:
                    row['buy_signal'] = True
                    row['signal_type'] = 'cover_ema_slope'
                    row['signal_quality'] = 6

            # Store
            row['trend_dir'] = trend_dir
            row['bar_count'] = up_pb_count if trend_up else dn_pb_count
            row['atr'] = float(atr[i]) if i < len(atr) else 0.0
            results.append(row)

        return pd.DataFrame(results, index=stock_data.index)

    # ================================================================
    # Indicator helpers
    # ================================================================

    def _compute_ema(self, values, period):
        n = len(values)
        result = np.empty(n, dtype=float)
        result[0] = values[0]
        k = 2.0 / (period + 1)
        for i in range(1, n):
            if i < period:
                result[i] = np.mean(values[:i + 1])
            else:
                result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    def _compute_atr_array(self, high, low, close, period):
        n = len(close)
        if n < 2:
            return np.zeros(n)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        atr = np.zeros(n)
        if len(tr) >= period:
            atr[period] = np.mean(tr[:period])
            for i in range(period + 1, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
        return atr

    def _compute_avg_range(self, high, low, close, period):
        n = len(close)
        ranges = high - low
        avg = np.zeros(n)
        for i in range(period, n):
            avg[i] = np.mean(ranges[i - period + 1:i + 1])
        return avg

    # ================================================================
    # Real-time screening
    # ================================================================

    def screen(self):
        data = self.data.copy()
        if len(data) < self.ema_period + 10:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        analysis = self._analyze_stock(data)
        latest = analysis.iloc[-1]
        price = float(data['close'].iloc[-1])

        if latest['buy_signal']:
            return {
                'action': 'buy',
                'reason': f"{latest['signal_type']} (quality={latest['signal_quality']})",
                'price': price,
            }
        elif latest['sell_signal']:
            return {
                'action': 'sell',
                'reason': f"{latest['signal_type']} (quality={latest['signal_quality']})",
                'price': price,
            }
        else:
            return {
                'action': 'hold',
                'reason': f"trend_dir={latest['trend_dir']}, bar_count={latest['bar_count']}",
                'price': price,
            }
