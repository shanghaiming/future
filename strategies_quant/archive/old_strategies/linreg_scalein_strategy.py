"""
线性回归趋势策略 (Linear Regression Scale-In Strategy)
======================================================
使用Pearson相关系数衡量趋势强度, 随趋势增强分步建仓。

来源: TradingView "LinReg Scale-In Strategy"

核心逻辑:
  Pearson's r 衡量价格与时间的线性相关:
  - r > 0.5: 强上升趋势 → 开始建仓
  - r > 0.75: 趋势加速 → 加仓
  - r < 0.25: 趋势消失 → 全部平仓

  特点:
  - 统计量化趋势强度 (比ADX更直观)
  - 分步建仓: 趋势越强仓位越大
  - 简单出场: 只需r < 0.25即全部平仓

技术指标: Pearson's r (线性回归相关系数), ATR止损
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class LinRegScaleInStrategy(BaseStrategy):
    """线性回归趋势策略 — Pearson's r量化趋势 + 分步建仓"""

    strategy_description = "LinReg: Pearson's r趋势强度 + 分步建仓 + ATR止损"
    strategy_category = "trend_following"
    strategy_params_schema = {
        "reg_period": {"type": "int", "default": 30, "label": "回归周期"},
        "entry_r": {"type": "float", "default": 0.4, "label": "入场r阈值"},
        "add_r": {"type": "float", "default": 0.6, "label": "加仓r阈值"},
        "strong_r": {"type": "float", "default": 0.75, "label": "强趋势r阈值"},
        "exit_r": {"type": "float", "default": 0.2, "label": "出场r阈值"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.reg_period = params.get('reg_period', 30)
        self.entry_r = params.get('entry_r', 0.4)
        self.add_r = params.get('add_r', 0.6)
        self.strong_r = params.get('strong_r', 0.75)
        self.exit_r = params.get('exit_r', 0.2)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'reg_period': 30, 'entry_r': 0.4, 'add_r': 0.6,
            'strong_r': 0.75, 'exit_r': 0.2,
            'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
        }

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
        low_water = float('inf')
        last_add_price = 0.0

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_sym = None
                best_dir = 0
                best_r = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    r_val, direction = result
                    if abs(r_val) > abs(best_r):
                        best_r = r_val
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_r) >= self.entry_r:
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
                    last_add_price = 0.0

            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) if low_water < float('inf') else current_price

                if last_add_price == 0.0:
                    last_add_price = current_price

                hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                result = self._evaluate(hist)
                should_exit = False

                if result is not None:
                    r_val, direction = result

                    if days_held >= self.hold_min:
                        # Exit: r dropped below threshold
                        if abs(r_val) < self.exit_r:
                            should_exit = True
                        # Exit: direction reversed
                        if position_dir == 1 and r_val < -self.entry_r:
                            should_exit = True
                        elif position_dir == -1 and r_val > self.entry_r:
                            should_exit = True

                # ATR trailing stop
                atr_val = self._calc_atr(hist)
                if atr_val > 0:
                    if position_dir == 1 and high_water > 0:
                        if current_price < high_water - self.trail_atr_mult * atr_val:
                            should_exit = True
                    elif position_dir == -1 and low_water < float('inf'):
                        if current_price > low_water + self.trail_atr_mult * atr_val:
                            should_exit = True

                # Max hold
                if days_held >= 90:
                    should_exit = True

                if should_exit and days_held >= self.hold_min:
                    if position_dir == 1:
                        self._record_signal(current_time, 'sell', current_holding)
                    else:
                        self._record_signal(current_time, 'buy', current_holding)
                    current_holding = None
                    buy_time = None
                    position_dir = 0
                    high_water = 0.0
                    low_water = float('inf')
                    last_add_price = 0.0

        print(f"LinRegScaleIn: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """计算Pearson's r"""
        if len(data) < self.reg_period + 5:
            return None

        close = data['close'].values
        n = len(close)

        # Pearson correlation: price vs time
        recent = close[-self.reg_period:]
        x = np.arange(self.reg_period, dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(recent)

        num = np.sum((x - x_mean) * (recent - y_mean))
        den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((recent - y_mean) ** 2))

        r = num / den if den > 0 else 0
        direction = 1 if r > 0 else -1

        return r, direction

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

    def screen(self):
        data = self.data.copy()
        if len(data) < 40:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        r_val, direction = result
        if abs(r_val) >= self.entry_r:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"r={r_val:.3f} (LinReg)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'r={r_val:.3f}', 'price': price}
