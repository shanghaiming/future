"""
Monte Carlo VaR策略 (Monte Carlo Value at Risk Strategy)
==========================================================
基于Bootstrap Monte Carlo的收益分布估计 + 重要性采样尾部风险量化

哲学: "以统计为镜，照见未来可能" — 用历史概率预测未来分布;
     95%分位数显著正向=买入机会，动态止损控制尾部风险

核心逻辑:
  1. 历史收益Bootstrap: 60天滚动窗口采样，生成1000条路径
  2. Monte Carlo模拟: 10天前向收益分布预测
  3. 风险指标: VaR(5%分位数) + 预期上行(95%分位数)
  4. 交易信号: 上行/下行比率 > 阈值时交易
  5. 结构张力确认: 7点结构位移方向确认
  6. 动态止损: 基于VaR估计的尾部风险止损
  7. Kelly仓位: 根据风险调整后的收益率分配仓位

技术指标: Bootstrap MC, VaR, Tail Risk, Importance Sampling, Structural Tension, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy
from typing import Dict, List, Any


class MonteCarloVARStrategy(BaseStrategy):
    """Monte Carlo VaR策略 — Bootstrap模拟收益分布 + 重要性采样尾部风险"""

    strategy_description = (
        "Monte Carlo VaR: Bootstrap收益模拟 + 重要性采样尾部风险 "
        "— 以统计为镜，照见未来可能"
    )
    strategy_category = "risk"
    strategy_params_schema = {
        "window": {"type": "int", "default": 60, "label": "滚动回溯窗口"},
        "n_simulations": {"type": "int", "default": 1000, "label": "模拟路径数"},
        "n_days": {"type": "int", "default": 10, "label": "前向预测天数"},
        "var_percentile": {"type": "float", "default": 5.0, "label": "VaR百分位"},
        "upside_percentile": {"type": "float", "default": 95.0, "label": "上行百分位"},
        "ratio_threshold": {"type": "float", "default": 2.0, "label": "上下行比率阈值"},
        "pivot_len": {"type": "int", "default": 5, "label": "摆动点回溯"},
        "lookback": {"type": "int", "default": 50, "label": "结构回溯"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "追踪止损ATR倍数"},
        "max_hold": {"type": "int", "default": 30, "label": "最大持仓天数"},
        "importance_samples": {"type": "int", "default": 200, "label": "重要性采样数量"},
        "importance_weight": {"type": "float", "default": 1.5, "label": "重要性权重"},
        "kelly_fraction": {"type": "float", "default": 0.25, "label": "Kelly分数"},
        "min_position": {"type": "float", "default": 0.05, "label": "最小仓位(5%)"},
        "max_position": {"type": "float", "default": 0.3, "label": "最大仓位(30%)"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 将params存储为实例属性，方便使用
        for key, value in self.params.items():
            setattr(self, key, value)

    def get_default_params(self):
        return {
            'window': 60, 'n_simulations': 1000, 'n_days': 10,
            'var_percentile': 5.0, 'upside_percentile': 95.0,
            'ratio_threshold': 2.0, 'pivot_len': 5, 'lookback': 50,
            'atr_period': 14, 'trail_atr_mult': 2.0,
            'max_hold': 30, 'importance_samples': 200,
            'importance_weight': 1.5, 'kelly_fraction': 0.25,
            'min_position': 0.05, 'max_position': 0.3,
        }

    def validate_params(self):
        if self.params['window'] < 30:
            raise ValueError("回溯窗口至少30天")
        if self.params['n_simulations'] < 100:
            raise ValueError("模拟路径数至少100")
        if self.params['n_days'] < 5 or self.params['n_days'] > 30:
            raise ValueError("前向预测天数5-30天")
        if not (0 < self.params['var_percentile'] < 100):
            raise ValueError("VaR百分位必须在0-100之间")
        if self.params['ratio_threshold'] <= 1.0:
            raise ValueError("上下行比率阈值必须大于1")
        if self.params['min_position'] < 0 or self.params['max_position'] > 1:
            raise ValueError("仓位范围必须在0-1之间")

    def generate_signals(self):
        data = self.data.copy()
        signals = []

        # 计算每日收益率
        data['returns'] = data['close'].pct_change()
        data['returns'] = data['returns'].fillna(0)

        # 添加技术指标
        self._add_technical_indicators(data)

        # 遍历数据进行回测
        for i in range(self.window, len(data)):
            # 截取回溯窗口数据
            window_data = data.iloc[i-self.window:i]

            # 生成模拟收益分布
            simulated_returns = self._bootstrap_simulation(window_data)

            # 计算风险指标
            var_value = np.percentile(simulated_returns, self.var_percentile)
            upside_value = np.percentile(simulated_returns, self.upside_percentile)

            # 计算上下行比率
            downside = abs(var_value) if var_value < 0 else 0
            upside = upside_value if upside_value > 0 else 0
            ratio = upside / (downside + 1e-7)

            # 获取当前价格和信号方向
            current_price = data['close'].iloc[i]
            current_date = data.index[i]

            # 结构张力确认
            tension_score, tension_direction = self._calculate_structural_tension(data.iloc[i-self.lookback:i])

            # 判断交易信号
            if ratio > self.ratio_threshold:
                # 多头信号
                if tension_direction > 0:
                    # Kelly仓位计算
                    position_size = self._calculate_kelly_position(simulated_returns)

                    # 动态止损
                    stop_loss = current_price + var_value  # VaR止损

                    signals.append({
                        'timestamp': current_date,
                        'action': 'buy',
                        'symbol': data['symbol'].iloc[i],
                        'price': current_price,
                        'var_value': var_value,
                        'upside_value': upside_value,
                        'ratio': ratio,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'structural_tension': tension_score,
                        'max_hold': self.max_hold
                    })

            elif ratio < 1.0 / self.ratio_threshold:
                # 空头信号
                if tension_direction < 0:
                    # Kelly仓位计算
                    position_size = self._calculate_kelly_position(simulated_returns)

                    # 动态止损
                    stop_loss = current_price + var_value  # VaR止损

                    signals.append({
                        'timestamp': current_date,
                        'action': 'sell',
                        'symbol': data['symbol'].iloc[i],
                        'price': current_price,
                        'var_value': var_value,
                        'upside_value': upside_value,
                        'ratio': ratio,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'structural_tension': tension_score,
                        'max_hold': self.max_hold
                    })

        return signals

    def _add_technical_indicators(self, data):
        """添加技术指标"""
        # 确保数据类型正确
        data = data.copy()
        data['pivot_high'] = np.nan
        data['pivot_low'] = np.nan
        data['atr'] = 0.0
        data['structural_tension'] = 0.0

        # 计算摆动点
        data['pivot_high'] = self._find_pivots(data['high'], self.pivot_len, is_high=True)
        data['pivot_low'] = self._find_pivots(data['low'], self.pivot_len, is_high=False)

        # 计算ATR
        data['atr'] = self._calc_atr(data)

        # 计算结构张力指标
        for i in range(self.lookback, len(data)):
            tension, _ = self._calculate_structural_tension(data.iloc[i-self.lookback:i])
            data.loc[data.index[i], 'structural_tension'] = float(tension)

    def _find_pivots(self, series, window, is_high=True):
        """查找摆动点"""
        pivots = np.zeros(len(series))
        for i in range(window, len(series) - window):
            if is_high:
                if series.iloc[i] == series.iloc[i-window:i+window].max():
                    pivots[i] = series.iloc[i]
            else:
                if series.iloc[i] == series.iloc[i-window:i+window].min():
                    pivots[i] = series.iloc[i]
        return pd.Series(pivots)

    def _bootstrap_simulation(self, window_data):
        """Bootstrap蒙特卡洛模拟"""
        simulated_returns = []

        # 基础bootstrap
        for _ in range(self.n_simulations):
            # 从历史收益中重采样
            bootstrapped_returns = np.random.choice(
                window_data['returns'].values,
                size=self.n_days,
                replace=True
            )
            # 计算累积收益
            cumulative_return = np.prod(1 + bootstrapped_returns) - 1
            simulated_returns.append(cumulative_return)

        # 重要性采样 - 重点评估尾部风险
        tail_returns = []
        for _ in range(self.importance_samples):
            # 向负收益方向采样
            if np.random.random() < 0.3:  # 30%概率采样尾部
                # 使用重要性权重偏向负收益
                tail_weight = self.importance_weight
                tail_return = np.random.choice(
                    window_data[window_data['returns'] < -0.02]['returns'].values,
                    size=min(self.n_days, 10),
                    replace=True
                )
                if len(tail_return) > 0:
                    cumulative_return = np.prod(1 + tail_return * tail_weight) - 1
                    tail_returns.append(cumulative_return)

        # 合并基础模拟和重要性采样
        simulated_returns.extend(tail_returns)

        return np.array(simulated_returns)

    def _calculate_structural_tension(self, data):
        """计算结构张力"""
        if len(data) < 7:
            return 0, 0

        # 提取7个关键结构点
        points = self._get_7_points(data)

        # 计算净位移
        net_displacement = 0
        base_price = data['close'].iloc[-1]  # 使用当前收盘价作为基准
        weights = [0.15, 0.1, 0.2, 0.1, 0.15, 0.1, 0.2]  # 权重

        for i, point in enumerate(points):
            if point is not None:
                net_displacement += (point - base_price) * weights[i]

        # 标准化
        atr = self._calc_atr(data)
        if atr > 0:
            tension_score = net_displacement / atr
            direction = 1 if tension_score > 0 else -1
        else:
            tension_score = 0
            direction = 0

        return tension_score, direction

    def _get_7_points(self, data):
        """获取7个关键结构点"""
        points = []

        if len(data) < 7:
            return [None] * 7

        # 点0: 前50天的最高价
        high_50 = data['high'].iloc[:-7].max() if len(data) > 57 else data['high'].iloc[0]
        points.append(high_50)

        # 点1: 前50天的最低价
        low_50 = data['low'].iloc[:-7].min() if len(data) > 57 else data['low'].iloc[0]
        points.append(low_50)

        # 点2: 前7天的最高价
        high_7 = data['high'].iloc[-7:].max()
        points.append(high_7)

        # 点3: 当前收盘价
        points.append(data['close'].iloc[-1])

        # 点4: 前7天的最低价
        low_7 = data['low'].iloc[-7:].min()
        points.append(low_7)

        # 点5: 当前到前50天的最高价
        high_50_current = data['high'].iloc[-50:].max()
        points.append(high_50_current)

        # 点6: 当前到前50天的最低价
        low_50_current = data['low'].iloc[-50:].min()
        points.append(low_50_current)

        return points

    def _calc_atr(self, data):
        """计算ATR"""
        if len(data) < self.atr_period + 1:
            return 0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

    def _calculate_kelly_position(self, simulated_returns):
        """基于Kelly准则的仓位计算"""
        # 计算收益分布
        positive_returns = simulated_returns[simulated_returns > 0]
        negative_returns = simulated_returns[simulated_returns < 0]

        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0

        # 计算胜率和平均盈亏
        p = len(positive_returns) / len(simulated_returns)  # 胜率
        avg_win = np.mean(positive_returns)  # 平均盈利
        avg_loss = abs(np.mean(negative_returns))  # 平均亏损

        # Kelly公式: f* = (p * b - q) / b
        # 其中 b = 平均盈利/平均亏损, q = 1-p
        b = avg_win / avg_loss
        kelly_fraction = (p * b - (1-p)) / b

        # 使用半Kelly原则，限制范围
        position = kelly_fraction * self.kelly_fraction
        position = np.clip(position, self.min_position, self.max_position)

        return position

    def screen(self):
        """实时选股"""
        data = self.data.copy()
        if len(data) < self.window + 10:
            return {'action': 'hold', 'reason': '数据不足',
                   'price': float(data['close'].iloc[-1])}

        current_price = data['close'].iloc[-1]
        current_date = data.index[-1]

        # 计算最新的技术指标
        data['returns'] = data['close'].pct_change()
        data['returns'] = data['returns'].fillna(0)
        self._add_technical_indicators(data)

        # 获取窗口数据
        window_data = data.iloc[-self.window:]

        # 生成模拟
        simulated_returns = self._bootstrap_simulation(window_data)

        # 计算指标
        var_value = np.percentile(simulated_returns, self.var_percentile)
        upside_value = np.percentile(simulated_returns, self.upside_percentile)
        downside = abs(var_value) if var_value < 0 else 0
        upside = upside_value if upside_value > 0 else 0
        ratio = upside / (downside + 1e-7)

        # 结构张力
        tension_score, tension_direction = self._calculate_structural_tension(
            data.iloc[-self.lookback:]
        )

        # 判断信号
        if ratio > self.ratio_threshold and tension_direction > 0:
            position = self._calculate_kelly_position(simulated_returns)
            return {
                'action': 'buy',
                'reason': f'Monte Carlo比率{ratio:.2f}>阈值{self.ratio_threshold}',
                'price': current_price,
                'position_size': position,
                'var_value': var_value,
                'upside_value': upside_value,
                'structural_tension': tension_score
            }
        elif ratio < 1.0 / self.ratio_threshold and tension_direction < 0:
            position = self._calculate_kelly_position(simulated_returns)
            return {
                'action': 'sell',
                'reason': f'Monte Carlo比率{ratio:.2f}<阈值{1/self.ratio_threshold:.2f}',
                'price': current_price,
                'position_size': position,
                'var_value': var_value,
                'upside_value': upside_value,
                'structural_tension': tension_score
            }
        else:
            return {'action': 'hold', 'reason': '无信号', 'price': current_price}