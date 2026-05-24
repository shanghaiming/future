"""
Smart Liquidity Trend Engine
=============================
基于流动性区域检测的趋势跟踪策略, 核心思路来自TradingView学习:
1. Swing High/Low检测流动性区域 + ATR zone merging
2. 两阶段信号: TAP(准备) → DIAMOND(执行)
3. Ghost zone invalidation: 收盘价穿过区域则失效
4. 评分系统: 流动性强度 + 反转确认 + 成交量
5. ATR trailing stop出场

知识来源:
- TradingView Smart Liquidity概念
- ICT流动性池/扫荡模型
- K线形态反转确认
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class SmartLiquidityTrendStrategy(BaseStrategy):
    """Smart Liquidity Trend Engine — 流动性区域检测 + 两阶段确认 + ATR止损"""

    strategy_description = (
        "流动性趋势引擎: Swing H/L区域合并 + TAP/DIAMOND两阶段信号 "
        "+ Ghost zone失效 + 评分系统 + ATR trailing stop"
    )
    strategy_category = "price_action"
    strategy_params_schema = {
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "swing_length": {"type": "int", "default": 10, "label": "Swing检测回望"},
        "zone_merge_atr_mult": {"type": "float", "default": 0.5, "label": "区域合并ATR倍数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "止损ATR倍数"},
        "score_threshold": {"type": "int", "default": 5, "label": "信号评分阈值"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "max_hold": {"type": "int", "default": 60, "label": "最大持仓天数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.atr_period = params.get('atr_period', 14)
        self.swing_length = params.get('swing_length', 10)
        self.zone_merge_atr_mult = params.get('zone_merge_atr_mult', 0.5)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)
        self.score_threshold = params.get('score_threshold', 5)
        self.hold_min = params.get('hold_min', 2)
        self.max_hold = params.get('max_hold', 60)

    def get_default_params(self):
        return {
            'atr_period': 14, 'swing_length': 10,
            'zone_merge_atr_mult': 0.5, 'trail_atr_mult': 2.5,
            'score_threshold': 5, 'hold_min': 2, 'max_hold': 60,
        }

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _calc_atr(self, high, low, close, n):
        """Average True Range"""
        if n < self.atr_period + 1:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
        )
        return float(np.mean(tr[-self.atr_period:]))

    def _find_swing_points(self, high, low, n):
        """检测Swing High/Low, 返回 (swing_highs, swing_lows)
        每个元素: (index, price)
        """
        swing_highs = []
        swing_lows = []
        length = self.swing_length
        for i in range(length, n - length):
            # Swing High: 中心bar的high > 左右各length根bar的high
            is_high = True
            for j in range(1, length + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append((i, high[i]))

            # Swing Low: 中心bar的low < 左右各length根bar的low
            is_low = True
            for j in range(1, length + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append((i, low[i]))
        return swing_highs, swing_lows

    def _merge_zones(self, zones, atr_val):
        """ATR归一化的区域合并: 两个相近的pivot点如果距离 < 0.5*ATR则合并"""
        if not zones or atr_val <= 0:
            return zones
        threshold = self.zone_merge_atr_mult * atr_val
        merged = []
        used = [False] * len(zones)
        for i in range(len(zones)):
            if used[i]:
                continue
            idx_i, price_i = zones[i]
            total_price = price_i
            count = 1
            total_idx = idx_i
            used[i] = True
            for j in range(i + 1, len(zones)):
                if used[j]:
                    continue
                idx_j, price_j = zones[j]
                if abs(price_j - price_i) < threshold:
                    used[j] = True
                    total_price += price_j
                    total_idx += idx_j
                    count += 1
            avg_price = total_price / count
            avg_idx = total_idx // count
            merged.append((avg_idx, avg_price, count))  # count = 被测试次数
        return merged

    def _is_ghost_zone(self, zone_price, is_supply, close_vals, lookback=5):
        """Ghost zone invalidation: 收盘价穿过区域则失效
        supply zone: 价格收盘突破上方则失效
        demand zone: 价格收盘跌破下方则失效
        """
        if len(close_vals) < lookback:
            return False
        recent = close_vals[-lookback:]
        if is_supply:
            # 价格收盘在supply zone之上 → 失效
            return any(c > zone_price for c in recent)
        else:
            # 价格收盘在demand zone之下 → 失效
            return any(c < zone_price for c in recent)

    def _reversal_score(self, open_vals, high, low, close, n):
        """K线形态反转评分 (看最后一根K线)"""
        if n < 2:
            return 0
        score = 0
        body = abs(close[-1] - open_vals[-1])
        bar_range = high[-1] - low[-1]
        if bar_range == 0:
            return 0

        # Bullish reversal patterns
        lower_wick = min(close[-1], open_vals[-1]) - low[-1]
        upper_wick = high[-1] - max(close[-1], open_vals[-1])

        # Bullish pin bar
        if lower_wick >= bar_range * 0.6 and close[-1] > open_vals[-1]:
            score += 2
        # Bearish pin bar
        elif upper_wick >= bar_range * 0.6 and close[-1] < open_vals[-1]:
            score -= 2

        # Engulfing
        if n >= 2:
            prev_body = close[-2] - open_vals[-2]
            curr_body = close[-1] - open_vals[-1]
            if prev_body < 0 and curr_body > 0 and abs(curr_body) > abs(prev_body):
                score += 2  # Bullish engulfing
            elif prev_body > 0 and curr_body < 0 and abs(curr_body) > abs(prev_body):
                score -= 2  # Bearish engulfing

        return score

    def _volume_score(self, vol_vals, n):
        """成交量评分: 当前成交量 vs 20日均量"""
        if n < 20 or vol_vals is None or len(vol_vals) < 20:
            return 0
        vol_ma = np.mean(vol_vals[-20:])
        if vol_ma == 0:
            return 0
        ratio = vol_vals[-1] / vol_ma
        if ratio > 1.5:
            return 2
        elif ratio > 1.0:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, data):
        """评估当前bar是否产生买入/卖出信号
        返回: (score, direction, atr_val) or None
        direction: 1=看多, -1=看空
        """
        min_len = self.swing_length * 2 + self.atr_period + 10
        if len(data) < min_len:
            return None

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        open_vals = data['open'].values
        n = len(close)

        # Volume
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        vol_vals = data[vol_col].values if vol_col else None

        # ATR
        atr_val = self._calc_atr(high, low, close, n)
        if atr_val <= 0:
            return None

        # Find swing points
        swing_highs, swing_lows = self._find_swing_points(high, low, n)

        # Merge zones with ATR normalization
        supply_zones = self._merge_zones(swing_highs, atr_val)   # resistance
        demand_zones = self._merge_zones(swing_lows, atr_val)    # support

        # Filter ghost zones
        valid_supply = [z for z in supply_zones if not self._is_ghost_zone(z[1], True, close)]
        valid_demand = [z for z in demand_zones if not self._is_ghost_zone(z[1], False, close)]

        current_price = close[-1]
        score = 0

        # --- TAP Phase: 价格接近流动性区域 ---
        tap_threshold = 1.0 * atr_val  # 接近 = 距离 < 1 ATR

        # Check proximity to demand zones (potential buy)
        near_demand = False
        best_demand_strength = 0
        for idx, price, tests in valid_demand:
            dist = current_price - price
            if 0 < dist < tap_threshold:
                near_demand = True
                best_demand_strength = max(best_demand_strength, tests)

        # Check proximity to supply zones (potential sell)
        near_supply = False
        best_supply_strength = 0
        for idx, price, tests in valid_supply:
            dist = price - current_price
            if 0 < dist < tap_threshold:
                near_supply = True
                best_supply_strength = max(best_supply_strength, tests)

        # --- DIAMOND Phase: 扫过区域后反转确认 ---
        # Sweep detection: price dipped below demand zone then closed above
        sweep_demand = False
        if n >= 2:
            for idx, zone_price, tests in valid_demand:
                if low[-1] < zone_price < close[-1]:
                    sweep_demand = True
                    best_demand_strength = max(best_demand_strength, tests)

        # Sweep supply: price spiked above supply zone then closed below
        sweep_supply = False
        if n >= 2:
            for idx, zone_price, tests in valid_supply:
                if high[-1] > zone_price > close[-1]:
                    sweep_supply = True
                    best_supply_strength = max(best_supply_strength, tests)

        # --- Scoring ---
        # Liquidity strength (区域被测试次数)
        if sweep_demand:
            score += min(best_demand_strength, 3)
        elif near_demand:
            score += min(best_demand_strength, 2)

        if sweep_supply:
            score -= min(best_supply_strength, 3)
        elif near_supply:
            score -= min(best_supply_strength, 2)

        # Reversal confirmation (K线形态)
        reversal = self._reversal_score(open_vals, high, low, close, n)
        score += reversal

        # Volume confirmation
        vol_s = self._volume_score(vol_vals, n)
        if (sweep_demand or near_demand) and vol_s > 0:
            score += vol_s
        elif (sweep_supply or near_supply) and vol_s > 0:
            score -= vol_s

        # DIAMOND requires sweep + reversal
        diamond_buy = sweep_demand and reversal > 0
        diamond_sell = sweep_supply and reversal < 0

        if diamond_buy:
            score += 2
        if diamond_sell:
            score -= 2

        direction = 1 if score > 0 else -1
        return score, direction, atr_val

    # ------------------------------------------------------------------
    # Signal generation (backtest)
    # ------------------------------------------------------------------

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_score = -1
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    score, direction, atr_val = result
                    if abs(score) > best_score:
                        best_score = abs(score)
                        best_sym = sym
                        best_dir = direction

                if best_sym and best_score >= self.score_threshold:
                    price = float(current_bars[current_bars['symbol'] == best_sym].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym, price)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym, price)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0

            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                if days_held >= self.hold_min:
                    bar_data = current_bars[current_bars['symbol'] == current_holding]
                    if len(bar_data) == 0:
                        continue
                    current_price = float(bar_data.iloc[0]['close'])

                    # Track high/low water mark
                    if position_dir == 1:
                        high_water = max(high_water, current_price)
                    else:
                        high_water = min(high_water, current_price) if high_water > 0 else current_price

                    # ATR trailing stop
                    hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                    atr_val = self._calc_atr(
                        hist['high'].values, hist['low'].values,
                        hist['close'].values, len(hist)
                    )
                    stop_hit = False

                    if atr_val > 0 and high_water > 0:
                        if position_dir == 1 and current_price < high_water - self.trail_atr_mult * atr_val:
                            stop_hit = True
                        elif position_dir == -1 and current_price > high_water + self.trail_atr_mult * atr_val:
                            stop_hit = True

                    # Max hold
                    if days_held >= self.max_hold:
                        stop_hit = True

                    # Signal-based exit
                    result = self._evaluate(hist)
                    signal_exit = False
                    if result is not None:
                        score, direction, _ = result
                        if position_dir == 1 and direction == -1 and score < -3:
                            signal_exit = True
                        elif position_dir == -1 and direction == 1 and score > 3:
                            signal_exit = True

                    if stop_hit or signal_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding, current_price)
                        else:
                            self._record_signal(current_time, 'buy', current_holding, current_price)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0

        print(f"SmartLiquidityTrend: 生成 {len(self.signals)} 个信号")
        return self.signals

    # ------------------------------------------------------------------
    # Real-time screening
    # ------------------------------------------------------------------

    def screen(self):
        data = self.data.copy()
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败(数据不足)', 'price': price}

        score, direction, atr_val = result
        if abs(score) >= self.score_threshold:
            action = 'buy' if direction == 1 else 'sell'
            return {
                'action': action,
                'reason': f"score={score}, atr={atr_val:.2f} (liquidity trend)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score} (低于阈值{self.score_threshold})', 'price': price}
