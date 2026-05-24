"""
KER Adaptive Strategy (道法自然 - Follow the Way)
===============================================
三层自适应系统：根据市场效率切换核心逻辑

哲学: "道法自然 - 效率决定战术。市场在不同状态下需要不同的应对方式。"

核心设计:
  1. KER (Kaufman Efficiency Ratio) 作为市场状态判别器
     KER = |净变化| / Σ|日变化|
     - KER > 0.3: TRENDING → 使用EMA交叉(快5慢20)跟趋势
     - KER 0.15-0.3: TRANSITION → 使用结构张力确定方向
     - KER < 0.15: RANGING → 使用均值回归(超买超卖)

  2. VDP (Volume Delta Pressure) 作为方向确认
     delta = volume * (2*close - high - low) / (high - low)
     必须有成交量支持才能入场

  3. ATR追踪止损自适应
     TRENDING: 2.5倍ATR (宽松)
     TRANSITION: 2.0倍ATR (中等)
     RANGING: 1.5倍ATR (收紧)

  4. 仓位大小自适应
     TRENDING: 0.5 (满仓)
     TRANSITION: 0.3 (中等)
     RANGING: 0.15 (轻仓)

技术指标: KER, EMA, RSI, 结构张力, VDP, ATR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class KERAdaptiveStrategy(BaseStrategy):
    """KER自适应策略 - 根据市场效率切换核心战术"""

    strategy_description = "道法自然 - KER自适应策略：趋势市EMA+横盘市RSI+过渡区结构张力"
    strategy_category = "regime"
    strategy_params_schema = {
        "ker_period": {"type": "int", "default": 20, "label": "KER计算周期"},
        "ker_trend_thresh": {"type": "float", "default": 0.3, "label": "趋势市KER阈值"},
        "ker_transition_thresh": {"type": "float", "default": 0.15, "label": "横盘市KER阈值"},
        "ema_fast": {"type": "int", "default": 5, "label": "快速EMA周期"},
        "ema_slow": {"type": "int", "default": 20, "label": "慢速EMA周期"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_ob": {"type": "int", "default": 70, "label": "RSI超买阈值"},
        "rsi_os": {"type": "int", "default": 30, "label": "RSI超卖阈值"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult_trend": {"type": "float", "default": 2.5, "label": "趋势市ATR止损倍数"},
        "trail_atr_mult_trans": {"type": "float", "default": 2.0, "label": "过渡区ATR止损倍数"},
        "trail_atr_mult_range": {"type": "float", "default": 1.5, "label": "横盘市ATR止损倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "score_threshold": {"type": "int", "default": 3, "label": "信号评分阈值"},
        "vdp_period": {"type": "int", "default": 10, "label": "VDP平滑周期"},
        "pivot_len": {"type": "int", "default": 5, "label": "枢轴点回望周期"},
    }

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ker_period': 20,
            'ker_trend_thresh': 0.3,
            'ker_transition_thresh': 0.15,
            'ema_fast': 5,
            'ema_slow': 20,
            'rsi_period': 14,
            'rsi_ob': 70,
            'rsi_os': 30,
            'atr_period': 14,
            'trail_atr_mult_trend': 2.5,
            'trail_atr_mult_trans': 2.0,
            'trail_atr_mult_range': 1.5,
            'hold_min': 3,
            'score_threshold': 3,
            'vdp_period': 10,
            'pivot_len': 5,
        }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        self.ker_period = params.get('ker_period', 20)
        self.ker_trend_thresh = params.get('ker_trend_thresh', 0.3)
        self.ker_transition_thresh = params.get('ker_transition_thresh', 0.15)
        self.ema_fast = params.get('ema_fast', 5)
        self.ema_slow = params.get('ema_slow', 20)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_ob = params.get('rsi_ob', 70)
        self.rsi_os = params.get('rsi_os', 30)
        self.atr_period = params.get('atr_period', 14)
        self.trail_atr_mult_trend = params.get('trail_atr_mult_trend', 2.5)
        self.trail_atr_mult_trans = params.get('trail_atr_mult_trans', 2.0)
        self.trail_atr_mult_range = params.get('trail_atr_mult_range', 1.5)
        self.hold_min = params.get('hold_min', 3)
        self.score_threshold = params.get('score_threshold', 3)
        self.vdp_period = params.get('vdp_period', 10)
        self.pivot_len = params.get('pivot_len', 5)

    def generate_signals(self) -> List[Dict]:
        """生成交易信号"""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []

        # 当前持仓状态
        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0
        low_water = float('inf')

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            # --- 计算技术指标 ---
            for _, bar in current_bars.iterrows():
                sym = bar['symbol']
                hist = data[(data['symbol'] == sym) & (data.index <= current_time)]

                # 计算所有指标
                ker = self._calc_ker(hist)
                atr = self._calc_atr(hist)
                vdp = self._calc_vdp(hist)
                ema_fast = self._calc_ema_series(hist['close'].values, self.ema_fast)
                ema_slow = self._calc_ema_series(hist['close'].values, self.ema_slow)
                rsi = self._calc_rsi(hist['close'].values)
                struct_tension = self._calc_structural_tension(hist)

                # 判断市场状态
                latest_ker = ker[-1]
                if latest_ker > self.ker_trend_thresh:
                    regime = 'TRENDING'
                    trail_mult = self.trail_atr_mult_trend
                    position_size = 0.5
                elif latest_ker > self.ker_transition_thresh:
                    regime = 'TRANSITION'
                    trail_mult = self.trail_atr_mult_trans
                    position_size = 0.3
                else:
                    regime = 'RANGING'
                    trail_mult = self.trail_atr_mult_range
                    position_size = 0.15

                # 根据市场状态生成信号
                signal_strength = 0
                direction = 0

                if regime == 'TRENDING':
                    # 趋势市：EMA交叉
                    if ema_fast[-1] > ema_slow[-1]:
                        signal_strength += 3
                    if ema_fast[-2] <= ema_slow[-2] and ema_fast[-1] > ema_slow[-1]:  # 金叉
                        signal_strength += 2

                elif regime == 'TRANSITION':
                    # 过渡区：结构张力
                    if struct_tension > 0.3:
                        signal_strength += 3
                    elif struct_tension > 0.1:
                        signal_strength += 1
                    elif struct_tension < -0.3:
                        signal_strength -= 3
                    elif struct_tension < -0.1:
                        signal_strength -= 1

                else:  # RANGING
                    # 横盘市：均值回归
                    if rsi < self.rsi_os:
                        signal_strength += 3
                        if rsi < 20:
                            signal_strength += 2
                    elif rsi < 40:
                        signal_strength += 1
                    elif rsi > self.rsi_ob:
                        signal_strength -= 3
                        if rsi > 80:
                            signal_strength -= 2
                    elif rsi > 60:
                        signal_strength -= 1

                # VDP确认
                vdp_direction = 1 if vdp[-1] > 0 else -1
                if direction == 0:
                    direction = vdp_direction

                # 如果信号强度有方向，覆盖VDP方向
                if signal_strength > 0:
                    direction = 1
                elif signal_strength < 0:
                    direction = -1

                # 只有VDP方向与信号方向一致时才入场
                if (direction == 1 and vdp[-1] > 0) or (direction == -1 and vdp[-1] < 0):
                    final_strength = abs(signal_strength)
                else:
                    final_strength = abs(signal_strength) * 0.5  # 打折

                # --- 交易逻辑 ---
                if current_holding is None:
                    # 寻找入场机会
                    if abs(final_strength) >= self.score_threshold:
                        if direction == 1 and final_strength > 0:
                            self._record_signal(
                                current_time, 'buy', sym,
                                float(bar['close']),
                                regime=regime,
                                ker=float(ker[-1]),
                                strength=final_strength,
                                atr=float(atr[-1]),
                                position_size=position_size
                            )
                            current_holding = sym
                            buy_time = current_time
                            position_dir = 1
                            high_water = float(bar['high'])
                            low_water = float(bar['low'])
                        elif direction == -1 and final_strength < 0:
                            self._record_signal(
                                current_time, 'sell', sym,
                                float(bar['close']),
                                regime=regime,
                                ker=float(ker[-1]),
                                strength=-final_strength,
                                atr=float(atr[-1]),
                                position_size=position_size
                            )
                            current_holding = sym
                            buy_time = current_time
                            position_dir = -1
                            high_water = float(bar['low'])
                            low_water = float(bar['high'])

                else:
                    # 持仓中：检查止损/止盈
                    should_exit = False

                    # 持仓天数检查
                    if buy_time:
                        days_held = (current_time - buy_time).days if hasattr(current_time, 'days') else 1
                        if days_held >= self.hold_min:
                            # ATR追踪止损
                            if position_dir == 1:  # 多头
                                trail_stop = high_water - trail_mult * float(atr[-1])
                                if float(bar['low']) <= trail_stop:
                                    should_exit = True
                            else:  # 空头
                                trail_stop = low_water + trail_mult * float(atr[-1])
                                if float(bar['high']) >= trail_stop:
                                    should_exit = True

                            # 更新高水位/低水位
                            if position_dir == 1:
                                if float(bar['high']) > high_water:
                                    high_water = float(bar['high'])
                            else:
                                if float(bar['low']) < low_water:
                                    low_water = float(bar['low'])

                            # 反向信号退出
                            if regime == 'TRENDING':
                                if (position_dir == 1 and ema_fast[-1] < ema_slow[-1]) or \
                                   (position_dir == -1 and ema_fast[-1] > ema_slow[-1]):
                                    should_exit = True
                            elif regime == 'RANGING':
                                if (position_dir == 1 and rsi > self.rsi_ob) or \
                                   (position_dir == -1 and rsi < self.rsi_os):
                                    should_exit = True

                    if should_exit:
                        self._record_signal(
                            current_time, 'sell' if position_dir == 1 else 'buy',
                            current_holding, float(bar['close']),
                            exit=True, regime=regime,
                            ker=float(ker[-1]),
                            atr=float(atr[-1])
                        )
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0
                        low_water = float('inf')

        print(f"KER Adaptive: generated {len(self.signals)} signals")
        return self.signals

    # ==================================================================
    # 市场状态计算
    # ==================================================================

    def _calc_ker(self, data):
        """计算Kaufman效率比率"""
        closes = data['close'].values.astype(float)
        n = len(closes)

        if n < self.ker_period:
            return np.full(n, 0.0)

        ker = np.full(n, 0.0)

        for i in range(self.ker_period - 1, n):
            # 净变化
            net_change = abs(closes[i] - closes[i - self.ker_period + 1])

            # 总变化
            total_change = 0.0
            for j in range(i - self.ker_period + 1, i + 1):
                if j > 0:
                    total_change += abs(closes[j] - closes[j - 1])

            if total_change > 1e-10:
                ker[i] = net_change / total_change
            else:
                ker[i] = 0.0

        return ker

    def _calc_vdp(self, data):
        """计算Volume Delta Pressure"""
        has_volume = 'volume' in data.columns and data['volume'].notna().any()

        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)

        # 计算原始delta
        hl_range = high - low
        safe_range = np.where(hl_range > 1e-10, hl_range, 1e-10)
        raw_delta = (2.0 * close - high - low) / safe_range

        # 如果有成交量，加权
        if has_volume:
            vol = data['volume'].values.astype(float)
            vol = np.nan_to_num(vol, nan=1.0)
            raw_delta = vol * raw_delta

        # EMA平滑
        vdp = self._calc_ema_series(raw_delta, self.vdp_period)

        return vdp

    # ==================================================================
    # 技术指标计算
    # ==================================================================

    def _calc_atr(self, data):
        """计算ATR"""
        if len(data) < self.atr_period + 1:
            return 0.0

        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        n = len(high)

        tr_list = []
        for i in range(max(1, n - self.atr_period), n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            tr_list.append(tr)

        return float(np.mean(tr_list)) if tr_list else 0.0

    def _calc_ema_series(self, values, period):
        """EMA序列计算"""
        values = np.asarray(values, dtype=float)
        n = len(values)
        result = np.empty(n)
        result[0] = values[0]

        if n > 1:
            k = 2.0 / (period + 1)
            for i in range(1, n):
                if i < period:
                    result[i] = np.mean(values[:i + 1])
                else:
                    result[i] = values[i] * k + result[i - 1] * (1 - k)

        return result

    def _calc_rsi(self, close):
        """计算RSI"""
        period = self.rsi_period
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))

        if avg_loss < 1e-10:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _calc_structural_tension(self, data):
        """计算结构张力"""
        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        n = len(close)

        if n < self.pivot_len * 2 + 5:
            return 0.0

        # 寻找枢轴点
        swing_highs = []
        swing_lows = []

        for i in range(self.pivot_len, n - self.pivot_len):
            is_high = all(high[i] >= high[i + j]
                         for j in range(-self.pivot_len, self.pivot_len + 1) if j != 0)
            is_low = all(low[i] <= low[i + j]
                        for j in range(-self.pivot_len, self.pivot_len + 1) if j != 0)

            if is_high:
                swing_highs.append((i, high[i]))
            if is_low:
                swing_lows.append((i, low[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.0

        # 构建参考点
        prev_sh = swing_highs[-2][1]
        curr_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1]
        curr_sl = swing_lows[-1][1]

        inter_high = max(curr_sh, prev_sh)
        inter_low = min(curr_sl, prev_sl)
        inter_mid = (inter_high + inter_low) / 2.0

        ref_points = [prev_sh, curr_sh, prev_sl, curr_sl, inter_high, inter_low, inter_mid]
        current_price = close[-1]

        # 计算相对于ATR的位移
        atr = self._calc_atr(data)
        if atr <= 0:
            return 0.0

        displacements = [(current_price - rp) / atr for rp in ref_points]
        tension = float(np.mean(displacements))

        return tension

    def screen(self):
        """实时选股"""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        if len(data) < 50:  # 需要足够数据计算KER
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}

        sym = data['symbol'].iloc[0]
        price = float(data['close'].iloc[-1])

        # 计算最新状态
        ker = self._calc_ker(data)
        atr = self._calc_atr(data)
        vdp = self._calc_vdp(data)
        ema_fast = self._calc_ema_series(data['close'].values, self.ema_fast)
        ema_slow = self._calc_ema_series(data['close'].values, self.ema_slow)
        rsi = self._calc_rsi(data['close'].values)
        struct_tension = self._calc_structural_tension(data)

        # 判断市场状态
        latest_ker = ker[-1]
        if latest_ker > self.ker_trend_thresh:
            regime = 'TRENDING'
        elif latest_ker > self.ker_transition_thresh:
            regime = 'TRANSITION'
        else:
            regime = 'RANGING'

        # 生成信号
        signal_strength = 0
        direction = 0

        if regime == 'TRENDING':
            if ema_fast[-1] > ema_slow[-1]:
                signal_strength += 3
                direction = 1
        elif regime == 'TRANSITION':
            if struct_tension > 0.2:
                signal_strength += 3
                direction = 1
            elif struct_tension < -0.2:
                signal_strength -= 3
                direction = -1
        else:  # RANGING
            if rsi < self.rsi_os:
                signal_strength += 3
                direction = 1
            elif rsi > self.rsi_ob:
                signal_strength -= 3
                direction = -1

        # VDP确认
        if vdp[-1] > 0:
            vdp_dir = 1
        else:
            vdp_dir = -1

        # 最终方向判断
        if signal_strength != 0:
            if (signal_strength > 0 and vdp_dir > 0) or (signal_strength < 0 and vdp_dir < 0):
                final_signal = abs(signal_strength)
            else:
                final_signal = abs(signal_strength) * 0.5
        else:
            final_signal = abs(vdp_dir) * 2

        if final_signal >= self.score_threshold:
            action = 'buy' if direction == 1 else 'sell'
            return {
                'action': action,
                'reason': f'{regime} regime (KER={latest_ker:.3f})',
                'price': price,
            }

        return {'action': 'hold', 'reason': f'score={final_signal:.1f}', 'price': price}