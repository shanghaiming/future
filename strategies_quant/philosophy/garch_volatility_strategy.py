"""
GARCH Volatility Strategy
========================

基于GARCH(1,1)的波动率均值回归策略
核心思想：当条件波动率高时卖出（预期收缩），条件波动率低时买入（预期扩张，向上偏移）

数学模型：
σ²_t = ω + α * r_{t-1}^2 + β * σ_{t-1}^2

交易逻辑：
- 条件波动率 / 非条件波动率 > 1.5（高波动状态）→ 卖出
- 条件波动率 / 非条件波动率 < 0.7（低波动状态）→ 买入
- 仓位大小与条件波动率成反比
- GARCH预测波动率用于ATR-like跟踪止损
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.base_strategy import BaseStrategy


class GARCHVolatilityStrategy(BaseStrategy):
    """基于GARCH(1,1)的波动率均值回归策略"""

    strategy_description: str = "使用GARCH(1,1)建模条件波动率，波动率均值回归交易策略"
    strategy_category: str = "general"

    def get_default_params(self) -> Dict[str, Any]:
        """返回默认参数"""
        return {
            # GARCH参数
            'garch_window': 126,          # 滚动窗口大小（约6个月）
            'omega_init': 0.0001,         # GARCH omega初始值
            'alpha_init': 0.1,            # GARCH alpha初始值
            'beta_init': 0.85,            # GARCH beta初始值

            # 交易阈值
            'high_vol_threshold': 1.5,    # 高波动率阈值（条件/非条件）
            'low_vol_threshold': 0.7,     # 低波动率阈值（条件/非条件）

            # 仓位管理
            'max_position': 1.0,          # 最大仓位比例
            'position_scale': 0.5,       # 仓位缩放因子

            # 止损参数
            'atr_multiplier': 2.0,       # ATR倍数
            'atr_period': 14,             # ATR周期

            # 结构张力过滤
            'tension_threshold': 0.3,    # 结构张力阈值
        }

    def validate_params(self):
        """验证参数"""
        if self.params['garch_window'] < 20:
            raise ValueError("garch_window must be >= 20")
        if self.params['high_vol_threshold'] <= self.params['low_vol_threshold']:
            raise ValueError("high_vol_threshold must be > low_vol_threshold")
        if not (0 < self.params['position_scale'] <= 1):
            raise ValueError("position_scale must be in (0, 1]")

    def _calculate_garch_params(self, returns: np.ndarray) -> tuple:
        """使用最大似然估计计算GARCH(1,1)参数"""
        n = len(returns)
        if n < 50:  # 数据不足
            return self.params['omega_init'], self.params['alpha_init'], self.params['beta_init']

        # 定义负对数似然函数
        def neg_log_likelihood(params, returns):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return np.inf

            # 计算条件方差
            sigma2 = np.zeros(n)
            sigma2[0] = omega / (1 - alpha - beta)  # 初始值为无条件方差

            for t in range(1, n):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

            # 确保方差为正
            sigma2 = np.maximum(sigma2, 1e-8)

            # 负对数似然
            log_likelihood = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
            return -np.sum(log_likelihood)

        # 初始参数
        initial_params = [self.params['omega_init'], self.params['alpha_init'], self.params['beta_init']]

        # 参数边界
        bounds = [(0.00001, 0.01), (0, 0.5), (0, 0.99)]

        # 优化
        result = minimize(neg_log_likelihood, initial_params, args=(returns,),
                         bounds=bounds, method='L-BFGS-B')

        if result.success:
            omega, alpha, beta = result.x
            # 确保alpha + beta < 1
            if alpha + beta >= 0.99:
                beta = 0.99 - alpha
            return omega, alpha, beta
        else:
            # 优化失败，返回初始值
            return self.params['omega_init'], self.params['alpha_init'], self.params['beta_init']

    def _calculate_conditional_volatility(self, returns: np.ndarray, omega: float,
                                         alpha: float, beta: float) -> np.ndarray:
        """计算条件波动率"""
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)  # 无条件方差

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

        # 转换为波动率（标准差）
        sigma = np.sqrt(sigma2)
        return sigma

    def _calculate_structural_tension(self) -> pd.Series:
        """计算结构张力指标"""
        # 使用价格范围和波动率的比值
        price_range = self.data['high'] - self.data['low']
        atr = self._calculate_atr()
        tension = price_range / atr

        # 归一化
        tension_mean = tension.rolling(20).mean()
        # Avoid division by zero
        tension_mean = tension_mean.fillna(1.0)
        tension = tension / tension_mean
        return tension.fillna(1.0)

    def _calculate_atr(self, period: int = None) -> pd.Series:
        """计算ATR指标"""
        if period is None:
            period = self.params['atr_period']

        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high.iloc[0] - low.iloc[0]

        # EMA of True Range
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        return atr

    def _calculate_garch_predicted_vol(self, omega: float, alpha: float, beta: float,
                                     current_vol: float, last_return: float) -> float:
        """预测下一期的波动率"""
        sigma2_pred = omega + alpha * last_return**2 + beta * current_vol**2
        return np.sqrt(sigma2_pred)

    def generate_signals(self) -> List[Dict]:
        """生成交易信号"""
        signals = []
        data = self.data.copy()

        # 计算收益率
        returns = data['close'].pct_change().dropna().values

        # 计算结构张力
        tension = self._calculate_structural_tension()

        # 计算ATR用于止损
        atr = self._calculate_atr()

        # GARCH参数跟踪
        omega_series = []
        alpha_series = []
        beta_series = []
        cond_vol_series = []
        uncond_vol_series = []

        for i in range(self.params['garch_window'], len(data)):
            # 滚动窗口GARCH估计
            window_returns = returns[i-self.params['garch_window']:i]

            # 估计GARCH参数
            omega, alpha, beta = self._calculate_garch_params(window_returns)
            omega_series.append(omega)
            alpha_series.append(alpha)
            beta_series.append(beta)

            # 计算条件波动率
            sigma_cond = self._calculate_conditional_volatility(window_returns, omega, alpha, beta)
            cond_vol_series.append(sigma_cond[-1])

            # 计算无条件波动率
            uncond_vol = np.sqrt(window_returns.var())
            uncond_vol_series.append(uncond_vol)

        # 添加到数据
        for i in range(len(cond_vol_series)):
            idx = i + self.params['garch_window']
            data.loc[data.index[idx], 'cond_vol'] = cond_vol_series[i]
            data.loc[data.index[idx], 'uncond_vol'] = uncond_vol_series[i]
            data.loc[data.index[idx], 'vol_ratio'] = cond_vol_series[i] / uncond_vol_series[i]
            data.loc[data.index[idx], 'tension'] = tension.iloc[i + self.params['garch_window']]

        # 生成信号
        for i in range(self.params['garch_window'], len(data)):
            current_idx = data.index[i]
            current_row = data.iloc[i]

            # 跳过无效数据
            if pd.isna(current_row['vol_ratio']) or pd.isna(current_row['tension']):
                continue

            vol_ratio = current_row['vol_ratio']
            tension = current_row['tension']
            atr_current = atr.iloc[i]

            # 高波动率状态 - 卖出
            if vol_ratio > self.params['high_vol_threshold'] and tension > self.params['tension_threshold']:
                # 计算仓位大小（与波动率成反比）
                position_size = self.params['max_position'] * (1 / vol_ratio) * self.params['position_scale']
                position_size = min(position_size, self.params['max_position'])

                # 使用GARCH预测波动率作为止损
                if i > 0 and i + 1 < len(data):
                    last_return = returns[i-1]
                    garch_omega = omega_series[i-self.params['garch_window']]
                    garch_alpha = alpha_series[i-self.params['garch_window']]
                    garch_beta = beta_series[i-self.params['garch_window']]
                    current_cond_vol = cond_vol_series[i-self.params['garch_window']]

                    predicted_vol = self._calculate_garch_predicted_vol(
                        garch_omega, garch_alpha, garch_beta, current_cond_vol, last_return
                    )

                    # 跟踪止损价
                    stop_loss = current_row['close'] - self.params['atr_multiplier'] * predicted_vol * current_row['close']
                else:
                    stop_loss = 0

                self._record_signal(
                    timestamp=current_idx,
                    action='sell',
                    symbol=current_row.get('symbol', 'DEFAULT'),
                    price=current_row['close'],
                    position_size=position_size,
                    vol_ratio=vol_ratio,
                    tension=tension,
                    stop_loss=stop_loss,
                    regime='high_vol'
                )

            # 低波动率状态 - 买入（考虑向上偏移）
            elif vol_ratio < self.params['low_vol_threshold'] and tension > self.params['tension_threshold']:
                # 计算仓位大小（与波动率成反比）
                position_size = self.params['max_position'] * (1 / vol_ratio) * self.params['position_scale']
                position_size = min(position_size, self.params['max_position'])

                # 使用GARCH预测波动率作为止损
                if i > 0 and i + 1 < len(data):
                    last_return = returns[i-1]
                    garch_omega = omega_series[i-self.params['garch_window']]
                    garch_alpha = alpha_series[i-self.params['garch_window']]
                    garch_beta = beta_series[i-self.params['garch_window']]
                    current_cond_vol = cond_vol_series[i-self.params['garch_window']]

                    predicted_vol = self._calculate_garch_predicted_vol(
                        garch_omega, garch_alpha, garch_beta, current_cond_vol, last_return
                    )

                    # 跟踪止损价
                    stop_loss = current_row['close'] - self.params['atr_multiplier'] * predicted_vol * current_row['close']
                else:
                    stop_loss = 0

                self._record_signal(
                    timestamp=current_idx,
                    action='buy',
                    symbol=current_row.get('symbol', 'DEFAULT'),
                    price=current_row['close'],
                    position_size=position_size,
                    vol_ratio=vol_ratio,
                    tension=tension,
                    stop_loss=stop_loss,
                    regime='low_vol'
                )

            # 中间状态 - 持有
            else:
                self._record_signal(
                    timestamp=current_idx,
                    action='hold',
                    symbol=current_row.get('symbol', 'DEFAULT'),
                    price=current_row['close'],
                    vol_ratio=vol_ratio,
                    tension=tension,
                    regime='normal'
                )

        return self.signals

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """获取策略关键指标"""
        metrics = {}

        if len(self.signals) > 0:
            # 计算波动率统计
            vol_ratios = [s.get('vol_ratio', 0) for s in self.signals if 'vol_ratio' in s]
            if vol_ratios:
                metrics['avg_vol_ratio'] = np.mean(vol_ratios)
                metrics['vol_ratio_std'] = np.std(vol_ratios)

            # 计算各状态分布
            regimes = [s.get('regime', 'unknown') for s in self.signals]
            regime_counts = pd.Series(regimes).value_counts()
            metrics['regime_distribution'] = regime_counts.to_dict()

            # 计算仓位分布
            positions = [s.get('position_size', 0) for s in self.signals if 'position_size' in s]
            if positions:
                metrics['avg_position_size'] = np.mean(positions)
                metrics['max_position_size'] = np.max(positions)

        return metrics