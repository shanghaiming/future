"""
Triple Confirmation Strategy (三确认策略)
========================================

Philosophy: 三思而后行 (Think thrice before acting)
Three independent confirmations required for action:

1. PRICE confirmation:
   - Close above 20-day MA
   - 5-day MA > 20-day MA (trend strength)

2. VOLUME confirmation:
   - VDP (Volume Delta Pressure) cumulative delta > 0
   - Shows institutional buying pressure

3. MOMENTUM confirmation:
   - 10-day momentum rank > 60 (cross-sectional)
   - Relative strength within universe

Entry: All 3 must agree for entry
Exit: Only 2 of 3 need to reverse
Score = count of confirming signals (max 3)
Risk: ATR trailing stop, max hold 30 days

Category: philosophy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class TripleConfirmationStrategy(BaseStrategy):
    """Triple Confirmation Strategy — Think thrice before acting."""

    strategy_description = (
        "三思而后行 - 三确认策略：价格趋势、成交量压力、动量排名三者确认后入场，"
        "任意两者反转即出场。ATR追踪止损，最大持仓30天。"
    )
    strategy_category = "philosophy"
    strategy_params_schema = {
        "ma_short": {"type": "int", "default": 5, "label": "短期MA周期"},
        "ma_long": {"type": "int", "default": 20, "label": "长期MA周期"},
        "vdp_ema_period": {"type": "int", "default": 10, "label": "VDP EMA周期"},
        "momentum_period": {"type": "int", "default": 10, "label": "动量计算周期"},
        "momentum_rank_threshold": {"type": "int", "default": 60, "label": "动量排名阈值"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_multiplier": {"type": "float", "default": 2.0, "label": "ATR追踪止损倍数"},
        "max_hold_days": {"type": "int", "default": 30, "label": "最大持仓天数"},
        "min_data_points": {"type": "int", "default": 60, "label": "最少数据点"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        # 设置默认参数
        default_params = self.get_default_params()
        self.params = {**default_params, **(params or {})}

        # 在调用super().__init__之前设置参数
        self.ma_short = self.params.get('ma_short', 5)
        self.ma_long = self.params.get('ma_long', 20)
        self.vdp_ema_period = self.params.get('vdp_ema_period', 10)
        self.momentum_period = self.params.get('momentum_period', 10)
        self.momentum_rank_threshold = self.params.get('momentum_rank_threshold', 60)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier = self.params.get('atr_multiplier', 2.0)
        self.max_hold_days = self.params.get('max_hold_days', 30)
        self.min_data_points = self.params.get('min_data_points', 60)

        # 调用父类初始化
        super().__init__(data, self.params)

        # 状态跟踪
        self.position = 0  # 0=空仓, 1=多头
        self.entry_price = 0.0
        self.entry_time = None
        self.high_water = 0.0  # 多头持仓期间的最高价
        self.low_water = float('inf')  # 空头持仓期间的最低价
        self.hold_days = 0

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ma_short': 5, 'ma_long': 20,
            'vdp_ema_period': 10, 'momentum_period': 10,
            'momentum_rank_threshold': 60,
            'atr_period': 14, 'atr_multiplier': 2.0,
            'max_hold_days': 30, 'min_data_points': 60,
        }

    def validate_params(self):
        """验证参数有效性"""
        if self.ma_short >= self.ma_long:
            raise ValueError("短期MA周期必须小于长期MA周期")
        if self.momentum_rank_threshold < 0 or self.momentum_rank_threshold > 100:
            raise ValueError("动量排名阈值必须在0-100之间")
        if self.atr_multiplier <= 0:
            raise ValueError("ATR倍数必须大于0")
        if self.max_hold_days <= 0:
            raise ValueError("最大持仓天数必须大于0")

    def _compute_ma(self, series: pd.Series, period: int) -> pd.Series:
        """计算移动平均"""
        return series.rolling(window=period, min_periods=period).mean()

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """计算ATR (Average True Range)"""
        # True Range = max(high-low, abs(high-前收), abs(低-前收))
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def _compute_vdp_delta(self, close: pd.Series, high: pd.Series, low: pd.Series,
                          volume: pd.Series) -> pd.Series:
        """计算VDP (Volume Delta Pressure)"""
        # VDP delta = volume * (2*close - high - low) / (high - low)
        # 当close接近high时，delta > 0 (买入压力)
        # 当close接近low时，delta < 0 (卖出压力)
        hl_range = high - low
        safe_range = hl_range.replace(0, 1e-10)  # 避免除零

        # 如果没有volume数据，使用position ratio作为代理
        if volume is None or volume.empty:
            raw_delta = (2.0 * close - high - low) / safe_range
        else:
            raw_delta = volume * (2.0 * close - high - low) / safe_range

        # EMA平滑
        alpha = 2.0 / (self.vdp_ema_period + 1)
        delta_ema = raw_delta.ewm(alpha=alpha, adjust=False).mean()

        return delta_ema

    def _compute_momentum(self, close: pd.Series, period: int) -> pd.Series:
        """计算动量"""
        momentum = (close - close.shift(period)) / close.shift(period) * 100
        return momentum

    def _compute_momentum_rank(self, momentum_series: pd.Series) -> pd.Series:
        """计算动量排名 (简化版 - 使用自身历史分位数)"""
        # 使用滚动窗口计算自身历史分位数作为排名代理
        rank_series = pd.Series(index=momentum_series.index, dtype=float)
        window_size = 60  # 60天滚动窗口

        for i in range(window_size, len(momentum_series)):
            window_data = momentum_series.iloc[i-window_size:i]
            valid_data = window_data.dropna()
            if len(valid_data) > 0:
                # 计算当前值在历史窗口中的百分位排名
                current_val = momentum_series.iloc[i]
                if not np.isnan(current_val):
                    rank = (valid_data < current_val).sum() / len(valid_data) * 100
                    rank_series.iloc[i] = min(rank, 100)  # 确保不超过100
                else:
                    rank_series.iloc[i] = np.nan
            else:
                rank_series.iloc[i] = np.nan

        return rank_series

    def _calculate_score(self, price_conf: bool, volume_conf: bool, momentum_conf: bool) -> int:
        """计算确认信号分数"""
        score = 0
        if price_conf:
            score += 1
        if volume_conf:
            score += 1
        if momentum_conf:
            score += 1
        return score

    def generate_signals(self) -> List[Dict]:
        """生成交易信号"""
        df = self.data.copy()
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'DEFAULT'

        # 检查数据是否足够
        if len(df) < self.min_data_points:
            print(f"⚠️ 数据不足: 需要{self.min_data_points}个数据点，只有{len(df)}个")
            return []

        # 计算技术指标
        df['ma_short'] = self._compute_ma(df['close'], self.ma_short)
        df['ma_long'] = self._compute_ma(df['close'], self.ma_long)
        df['atr'] = self._compute_atr(df['high'], df['low'], df['close'], self.atr_period)
        df['vdp_delta'] = self._compute_vdp_delta(df['close'], df['high'], df['low'],
                                                df.get('volume', None))
        df['momentum'] = self._compute_momentum(df['close'], self.momentum_period)

        # 计算动量排名 (简化版，实际应该使用universe数据)
        # 这里使用相对排名作为示例
        df['momentum_rank'] = self._compute_momentum_rank(df['momentum'])

        # 初始化信号列表
        self.signals = []

        # VDP累计 delta
        df['vdp_cumsum'] = df['vdp_delta'].cumsum()

        warmup = max(self.ma_long, self.vdp_ema_period, self.atr_period, 30)

        for i in range(warmup, len(df)):
            ts = df.index[i]
            price = float(df['close'].iloc[i])

            # 检查技术指标有效性
            if (df['ma_short'].iloc[i] is None or
                df['ma_long'].iloc[i] is None or
                df['vdp_cumsum'].iloc[i] is None or
                df['momentum_rank'].iloc[i] is None):
                continue

            # 1. 价格确认: 收盘价 > 20日MA 且 5日MA > 20日MA
            price_conf = (
                df['close'].iloc[i] > df['ma_long'].iloc[i] and
                df['ma_short'].iloc[i] > df['ma_long'].iloc[i]
            )

            # 2. 成交量确认: VDP累计delta > 0
            volume_conf = df['vdp_cumsum'].iloc[i] > 0

            # 3. 动量确认: 10日动量排名 > 60
            momentum_conf = (
                df['momentum_rank'].iloc[i] is not None and
                df['momentum_rank'].iloc[i] > self.momentum_rank_threshold
            )

            # 计算当前分数
            current_score = self._calculate_score(price_conf, volume_conf, momentum_conf)

            # 处理持仓状态
            if self.position == 0:  # 空仓
                # 三者都确认才买入
                if price_conf and volume_conf and momentum_conf:
                    self._record_signal(ts, 'buy', symbol, price,
                                     score=current_score,
                                     reason='triple_confirmation',
                                     price_conf=int(price_conf),
                                     volume_conf=int(volume_conf),
                                     momentum_conf=int(momentum_conf))
                    self.position = 1
                    self.entry_price = price
                    self.entry_time = ts
                    self.high_water = price
                    self.low_water = float('inf')
                    self.hold_days = 0

            else:  # 有持仓
                # 更新持仓天数
                if self.entry_time:
                    self.hold_days = (ts - self.entry_time).days

                # 更新水位
                if self.position == 1:  # 多头持仓
                    self.high_water = max(self.high_water, price)
                else:  # 空头持仓
                    self.low_water = min(self.low_water, price)

                # 检查止损条件
                atr_value = df['atr'].iloc[i]
                if atr_value is not None and atr_value > 0:
                    if self.position == 1:  # 多头
                        stop_loss = self.entry_price - self.atr_multiplier * atr_value
                        if price <= stop_loss:
                            self._record_signal(ts, 'sell', symbol, price,
                                             score=0,
                                             reason='atr_stop_loss',
                                             current_score=current_score)
                            self.position = 0
                            self.entry_price = 0.0
                            self.entry_time = None
                            self.high_water = 0.0
                            self.low_water = float('inf')
                            self.hold_days = 0
                            continue

                    elif self.position == -1:  # 空头
                        stop_loss = self.entry_price + self.atr_multiplier * atr_value
                        if price >= stop_loss:
                            self._record_signal(ts, 'buy', symbol, price,
                                             score=0,
                                             reason='atr_stop_loss',
                                             current_score=current_score)
                            self.position = 0
                            self.entry_price = 0.0
                            self.entry_time = None
                            self.high_water = 0.0
                            self.low_water = float('inf')
                            self.hold_days = 0
                            continue

                # 检查最大持仓天数
                if self.hold_days >= self.max_hold_days:
                    signal_type = 'sell' if self.position == 1 else 'buy'
                    self._record_signal(ts, signal_type, symbol, price,
                                     score=current_score,
                                     reason='max_hold_days',
                                     current_score=current_score)
                    self.position = 0
                    self.entry_price = 0.0
                    self.entry_time = None
                    self.high_water = 0.0
                    self.low_water = float('inf')
                    self.hold_days = 0
                    continue

                # 检查退出条件: 任意两者确认信号消失
                # 注意: 这里简化处理，实际应该检查原始确认条件的变化
                # 这里我们检查当前分数是否下降
                prev_score = getattr(self, 'prev_score', 3)
                if current_score < prev_score and current_score <= 1:
                    signal_type = 'sell' if self.position == 1 else 'buy'
                    self._record_signal(ts, signal_type, symbol, price,
                                     score=current_score,
                                     reason='confirmation_reverse',
                                     current_score=current_score)
                    self.position = 0
                    self.entry_price = 0.0
                    self.entry_time = None
                    self.high_water = 0.0
                    self.low_water = float('inf')
                    self.hold_days = 0
                    continue

                self.prev_score = current_score

        return self.signals

    def screen(self) -> Dict:
        """基于最新数据做实时选股判断"""
        if len(self.data) < self.min_data_points:
            return {'action': 'hold', 'reason': '数据不足',
                    'price': float(self.data['close'].iloc[-1])}

        last_close = float(self.data['close'].iloc[-1])

        try:
            # 获取最新的三个确认状态
            df = self.data.copy()
            df['ma_short'] = self._compute_ma(df['close'], self.ma_short)
            df['ma_long'] = self._compute_ma(df['close'], self.ma_long)
            df['vdp_delta'] = self._compute_vdp_delta(df['close'], df['high'], df['low'],
                                                    df.get('volume', None))
            df['momentum'] = self._compute_momentum(df['close'], self.momentum_period)

            last_idx = -1
            price_conf = (
                df['close'].iloc[last_idx] > df['ma_long'].iloc[last_idx] and
                df['ma_short'].iloc[last_idx] > df['ma_long'].iloc[last_idx]
            )

            # 简化的VDP检查 (使用最近10天的累计)
            recent_vdp = df['vdp_delta'].iloc[-10:].sum()
            volume_conf = recent_vdp > 0

            # 简化的动量排名检查
            momentum_pct = (df['momentum'].iloc[last_idx] - df['momentum'].min()) / \
                          (df['momentum'].max() - df['momentum'].min()) * 100
            momentum_conf = momentum_pct > self.momentum_rank_threshold

            score = self._calculate_score(price_conf, volume_conf, momentum_conf)

            if score == 3:
                return {
                    'action': 'buy',
                    'reason': 'triple_confirmation',
                    'score': score,
                    'price': last_close
                }
            elif score >= 2:
                return {
                    'action': 'hold',
                    'reason': 'partial_confirmation',
                    'score': score,
                    'price': last_close
                }
            else:
                return {
                    'action': 'sell' if score == 0 else 'hold',
                    'reason': 'weak_confirmation' if score > 0 else 'no_confirmation',
                    'score': score,
                    'price': last_close
                }

        except Exception as e:
            return {'action': 'hold', 'reason': f'计算错误: {str(e)}',
                    'price': last_close}