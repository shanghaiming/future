"""
红绿灯策略 (Traffic Light Strategy)
====================================
3根同色K线 + 成交量确认 + ATR追踪止损

来源: TradingView "Traffic Light" 红绿灯K线形态策略

核心逻辑:
  1. 连续3根同色K线(全红或全绿)形成短期情绪极化
  2. 成交量放大(>1.5倍20日均量)确认资金参与度
  3. 3绿+放量 → 动量延续做多; 3红+放量 → 恐慌抛售做空
  4. ATR追踪止损保护利润

数学原理:
  - 连续同色K线反映市场情绪单边倾斜, 类似行为金融学中的
    "锚定效应" — 投资者在连续上涨/下跌中形成方向性预期
  - 成交量放大排除低流动性的虚假形态, 只捕捉有资金推动的行情
  - ATR追踪止损自适应波动率, 避免被正常回调震出

技术指标: K线颜色判别, 成交量MA, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class TrafficLightStrategy(BaseStrategy):
    """红绿灯策略 — 3同色K线 + 量能确认 + ATR止损"""

    strategy_description = "红绿灯: 3同色K线+量能确认+ATR追踪止损"
    strategy_category = "kline_pattern"
    strategy_params_schema = {
        "vol_mult": {"type": "float", "default": 1.5, "label": "成交量倍数阈值"},
        "vol_ma_period": {"type": "int", "default": 20, "label": "成交量均线周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "max_hold": {"type": "int", "default": 60, "label": "最大持仓天数"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        p = self.params
        self.vol_mult = p.get('vol_mult', 1.5)
        self.vol_ma_period = p.get('vol_ma_period', 20)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 2)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 60)

    def get_default_params(self):
        return {
            'vol_mult': 1.5, 'vol_ma_period': 20,
            'atr_period': 14, 'hold_min': 2,
            'trail_atr_mult': 2.5, 'max_hold': 60,
        }

    def generate_signals(self) -> list:
        """
        生成交易信号。

        遍历时间线, 在无持仓时寻找3同色K线+放量入场;
        在有持仓时用ATR追踪止损或最大持仓天数退出。
        支持做多(3绿+放量)和做空(3红+放量)。
        """
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0
        low_water = float('inf')

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            # --- 无持仓: 寻找入场信号 ---
            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    score, direction = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= 3:
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0
                    low_water = float('inf')

            # --- 有持仓: 检查退出 ---
            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # 更新最高/最低水位
                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) if low_water < float('inf') else current_price

                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding) & (data.index <= current_time)]
                    atr_val = self._calc_atr(hist)
                    should_exit = False

                    # ATR trailing stop
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # Max hold days
                    if days_held >= self.max_hold:
                        should_exit = True

                    # Signal-based exit: 反向信号
                    if not should_exit:
                        result = self._evaluate(hist)
                        if result is not None:
                            score, direction = result
                            if position_dir == 1 and direction == -1 and score < -3:
                                should_exit = True
                            elif position_dir == -1 and direction == 1 and score > 3:
                                should_exit = True

                    if should_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding)
                        else:
                            self._record_signal(current_time, 'buy', current_holding)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0
                        low_water = float('inf')

        print(f"TrafficLight: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """
        评估3同色K线形态 + 成交量确认。

        Returns:
            (score, direction): score绝对值>=3触发信号, direction: 1=做多, -1=做空
            None if 数据不足
        """
        # 需要足够数据计算20日均量 + 3根K线
        min_len = self.vol_ma_period + 5
        if len(data) < min_len:
            return None

        close = data['close'].values
        open_ = data['open'].values

        # 需要 volume 列
        if 'volume' not in data.columns:
            return None
        volume = data['volume'].values

        n = len(close)
        score = 0

        # ===== 1. 判断最近3根K线颜色 =====
        # 绿色 = close > open (阳线), 红色 = close < open (阴线)
        last_3_colors = []
        for i in range(n - 3, n):
            if close[i] > open_[i]:
                last_3_colors.append('green')
            elif close[i] < open_[i]:
                last_3_colors.append('red')
            else:
                last_3_colors.append('neutral')

        all_green = all(c == 'green' for c in last_3_colors)
        all_red = all(c == 'red' for c in last_3_colors)

        if not all_green and not all_red:
            return (0, 0)

        # ===== 2. 成交量确认 =====
        vol_ma = np.mean(volume[-(self.vol_ma_period + 3):-3])
        current_vol = volume[-1]

        # 当前成交量 > vol_mult * 20日均量
        vol_ratio = current_vol / (vol_ma + 1e-10)
        vol_confirmed = vol_ratio >= self.vol_mult

        if not vol_confirmed:
            return (0, 0)

        # ===== 3. 评分 =====
        if all_green:
            # 3绿+放量 → 动量延续做多
            score = 3
            # 额外加分: 量能越强信号越强
            if vol_ratio >= 2.0:
                score += 1
            # 3根K线的涨幅累积越大越好
            cum_gain = (close[-1] - open_[-3]) / (open_[-3] + 1e-10)
            if cum_gain > 0.03:
                score += 1
            return (score, 1)

        elif all_red:
            # 3红+放量 → 恐慌抛售做空
            score = 3
            if vol_ratio >= 2.0:
                score += 1
            cum_loss = (open_[-3] - close[-1]) / (open_[-3] + 1e-10)
            if cum_loss > 0.03:
                score += 1
            return (score, -1)

        return (0, 0)

    def _calc_atr(self, data):
        """计算最新ATR值"""
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
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

    def screen(self):
        """
        实时选股: 基于最新数据判断当前是否有3同色K线+放量信号。

        Returns:
            Dict with keys: action, reason, price
        """
        data = self.data.copy()
        if len(data) < self.vol_ma_period + 5:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败(缺volume?)', 'price': price}

        score, direction = result
        if abs(score) >= 3:
            action = 'buy' if direction == 1 else 'sell'
            color_desc = "3绿" if direction == 1 else "3红"
            return {
                'action': action,
                'reason': f'{color_desc}+放量 score={score}',
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
