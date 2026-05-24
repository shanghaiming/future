"""
Martingale Stopping Strategy
基于鞅论和最优停止定理的入场时机选择
- 使用鞅停止理论优化入场时机
- 跟踪累计收益作为随机过程
- 应用可选停止定理：当过程到达边界且有正期望时入场
- 使用Doob分解将趋势从噪声中分离
- 方向来自动量+VDP确认
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy import stats
from sklearn.linear_model import LinearRegression

from core.base_strategy import BaseStrategy


class MartingaleStoppingStrategy(BaseStrategy):
    """鞅停止策略 - 使用鞅论理论优化入场时机"""

    strategy_description = "基于鞅论和最优停止定理的入场时机选择"
    strategy_category = "martingale"

    def __init__(self, data: pd.DataFrame, params: dict = None):
        # 默认参数
        default_params = {
            'lookback': 50,           # 鞅过程计算窗口
            'stop_loss_threshold': -0.05,  # 止损阈值（绝对收益）
            'take_profit_threshold': 0.15,  # 止盈阈值（绝对收益）
            'momentum_window': 20,    # 动量计算窗口
            'vdp_window': 10,        # VDP计算窗口
            'confidence_level': 0.05, # 统计显著性水平
            'drift_smoothing': 0.3,  # Doob分解中漂移项的平滑系数
        }

        super().__init__(data, params)

        # 计算必要的指标
        self._compute_indicators()

    def get_default_params(self) -> Dict[str, Any]:
        """返回默认参数"""
        return {
            'lookback': 50,
            'stop_loss_threshold': -0.05,
            'take_profit_threshold': 0.15,
            'momentum_window': 20,
            'vdp_window': 10,
            'confidence_level': 0.05,
            'drift_smoothing': 0.3,
        }

    def validate_params(self):
        """验证参数"""
        for param in ['lookback', 'momentum_window', 'vdp_window']:
            if self.params[param] <= 0:
                raise ValueError(f"{param} 必须大于0")

        for param in ['stop_loss_threshold', 'take_profit_threshold']:
            value = self.params[param]
            if not (-1 <= value <= 1):
                raise ValueError(f"{param} 必须在[-1, 1]范围内")

    def _compute_indicators(self):
        """计算所有必要的技术指标"""
        df = self.data.copy()

        # 1. 计算收益率
        df['returns'] = df['close'].pct_change()

        # 2. 计算累计收益（鞅过程）
        df['cumulative_returns'] = (1 + df['returns']).cumprod()

        # 3. 计算动量
        df['momentum'] = df['close'].pct_change(self.params['momentum_window'])

        # 4. 计算VDP (Volume Delta Pressure)
        df['vdp'] = self._compute_vdp(df)

        # 5. Doob分解：将过程分解为漂移项和鞅项
        df['drift'], df['martingale'] = self._doob_decomposition(df)

        # 6. 鞅假设检验
        df['martingale_test_pvalue'] = self._test_martingale_hypothesis(df)

        # 7. 边界距离（用于可选停止定理）
        df['upper_boundary_dist'] = self.params['take_profit_threshold'] - df['cumulative_returns'] + 1
        df['lower_boundary_dist'] = df['cumulative_returns'] + 1 - (1 + self.params['stop_loss_threshold'])

        self.indicators = df

    def _compute_vdp(self, df: pd.DataFrame) -> pd.Series:
        """计算VDP (Volume Delta Pressure)"""
        n = len(df)
        vdp = np.zeros(n)

        for i in range(1, n):
            if i >= self.params['vdp_window']:
                # 使用滚动窗口计算VDP
                window_returns = df['returns'].iloc[i-self.params['vdp_window']:i+1]
                window_volume = df['volume'].iloc[i-self.params['vdp_window']:i+1]

                # VDP delta公式: V × (2C-H-L)/(H-L)
                # 简化版本：使用收益率作为方向代理
                delta = np.mean(window_returns * window_volume)
                vdp[i] = delta / (np.std(window_returns) + 1e-8)  # 标准化

        return pd.Series(vdp, index=df.index)

    def _doob_decomposition(self, df: pd.DataFrame) -> tuple:
        """Doob分解：将过程分解为漂移项和鞅项"""
        n = len(df)
        drift = np.zeros(n)
        martingale = np.zeros(n)

        # 从第lookback个点开始
        for i in range(self.params['lookback'], n):
            # 使用滚动窗口估计条件期望
            window_returns = df['returns'].iloc[i-self.params['lookback']:i]

            # 移除NaN值
            clean_returns = window_returns.dropna()
            if len(clean_returns) < 5:  # 至少需要5个数据点
                continue

            # 简单的线性回归估计条件期望
            X = np.arange(len(clean_returns)).reshape(-1, 1)
            y = clean_returns.values

            try:
                model = LinearRegression().fit(X, y)
                # 漂移项是条件期望的累积
                expected_return = model.predict([[0]])[0]
                drift[i] = drift[i-1] + expected_return * self.params['drift_smoothing']

                # 鞅项是累计收益减去漂移项
                cum_return = (1 + df['returns'].iloc[:i+1]).prod()
                martingale[i] = cum_return - drift[i]
            except:
                # 如果回归失败，保持之前的值
                continue

        return pd.Series(drift, index=df.index), pd.Series(martingale, index=df.index)

    def _test_martingale_hypothesis(self, df: pd.DataFrame) -> pd.Series:
        """检验鞅假设"""
        pvalues = np.ones(len(df))  # 默认为1.0（不能拒绝零假设）

        for i in range(self.params['lookback'], len(df)):
            # 滚动窗口检验
            window_returns = df['returns'].iloc[i-self.params['lookback']:i+1]
            window_returns = window_returns.dropna()

            # 检验：E[r_t | r_{t-1},...,r_{t-k}] = 0
            if len(window_returns) > 10:
                try:
                    # 使用滞后变量
                    X = []
                    for lag in range(1, min(3, self.params['lookback']//2)):
                        if len(window_returns) > lag:
                            X.append(window_returns.iloc[:-lag].values)

                    if X and len(X[0]) > 10:
                        X = np.column_stack(X)
                        y = window_returns.iloc[len(X[0]):]

                        model = LinearRegression().fit(X, y)
                        r_squared = model.score(X, y)
                        # 简化的p值计算：R²越大，越可能拒绝鞅假设
                        pvalues[i] = max(0.01, 1.0 - r_squared * 5)
                except:
                    continue

        return pd.Series(pvalues, index=df.index)

    def generate_signals(self) -> List[Dict]:
        """生成交易信号"""
        signals = []
        df = self.indicators

        for i in range(self.params['lookback'], len(df)):
            current_row = df.iloc[i]

            # 应用可选停止定理的条件
            # 1. 鞅假设被拒绝（过程不是公平的）
            is_martingale_rejected = current_row['martingale_test_pvalue'] < self.params['confidence_level']

            # 2. 累计收益接近边界
            near_upper_boundary = current_row['upper_boundary_dist'] < 0.02
            near_lower_boundary = current_row['lower_boundary_dist'] < 0.02

            # 3. 漂移项显著（有可预测的趋势）
            drift_positive = current_row['drift'] > 0.001
            drift_negative = current_row['drift'] < -0.001

            # 4. 方向确认：动量+VDP
            momentum_positive = current_row['momentum'] > 0
            momentum_negative = current_row['momentum'] < 0
            vdp_positive = current_row['vdp'] > 0
            vdp_negative = current_row['vdp'] < 0

            # 决策逻辑
            if is_martingale_rejected:
                if near_upper_boundary and drift_positive and momentum_positive and vdp_positive:
                    # 多头信号
                    self._record_signal(
                        timestamp=current_row.name,
                        action='buy',
                        price=current_row['close'],
                        reason='鞅停止理论：接近上边界，有正漂移和确认信号',
                        drift_value=current_row['drift'],
                        martingale_test_pvalue=current_row['martingale_test_pvalue']
                    )

                elif near_lower_boundary and drift_negative and momentum_negative and vdp_negative:
                    # 空头信号
                    self._record_signal(
                        timestamp=current_row.name,
                        action='sell',
                        price=current_row['close'],
                        reason='鞅停止理论：接近下边界，有负漂移和确认信号',
                        drift_value=current_row['drift'],
                        martingale_test_pvalue=current_row['martingale_test_pvalue']
                    )

            # 中性信号（避免过度交易）
            if (abs(current_row['drift']) < 0.0005 or
                abs(current_row['momentum']) < 0.001 or
                abs(current_row['vdp']) < 0.001):
                self._record_signal(
                    timestamp=current_row.name,
                    action='hold',
                    price=current_row['close'],
                    reason='鞅停止理论：漂移/动量/VDP均较弱'
                )

        return self.signals

    def get_process_analysis(self) -> Dict:
        """返回鞅过程的分析结果"""
        if not hasattr(self, 'indicators') or self.indicators.empty:
            return {}

        df = self.indicators

        # 计算漂移和鞅项的统计特性
        drift_std = df['drift'].std()
        martingale_std = df['martingale'].std()

        # 计算边界触碰概率
        upper_touch = (df['cumulative_returns'] > 1 + self.params['take_profit_threshold']).sum()
        lower_touch = (df['cumulative_returns'] < 1 + self.params['stop_loss_threshold']).sum()

        # 鞅假设拒绝率
        rejection_rate = (df['martingale_test_pvalue'] < self.params['confidence_level']).mean()

        return {
            'drift_standard_deviation': float(drift_std),
            'martingale_standard_deviation': float(martingale_std),
            'upper_boundary_touches': int(upper_touch),
            'lower_boundary_touches': int(lower_touch),
            'martingale_rejection_rate': float(rejection_rate),
            'current_drift': float(df['drift'].iloc[-1]),
            'current_martingale': float(df['martingale'].iloc[-1]),
        }