"""
Random Matrix Filter Strategy
基于随机矩阵理论的市场噪声过滤
- 使用Marchenko-Pastur定律过滤相关矩阵中的噪声
- 计算N只股票的滚动相关矩阵
- 保留超过MP上界的特征值（信号）
- 重构过滤后的相关矩阵以识别真实市场因子
- 在过滤后的因子结构显示regime shift时交易
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

from core.base_strategy import BaseStrategy


class RandomMatrixFilterStrategy(BaseStrategy):
    """随机矩阵过滤策略 - 使用RMT识别市场真实结构"""

    strategy_description = "基于随机矩阵理论的市场噪声过滤"
    strategy_category = "random_matrix"

    def __init__(self, data: pd.DataFrame, params: dict = None):
        # 默认参数
        default_params = {
            'n_stocks': 10,           # 使用的股票数量
            'window_size': 60,        # 滚动窗口大小（天数）
            'min_observations': 30,   # 最小观测值要求
            'eigenvalue_threshold': 1.5,  # 特征值倍数阈值（MP上界的倍数）
            'regime_lookback': 20,    # Regime检测窗口
            'regime_threshold': 0.3,  # Regime切换阈值
            'weight_scheme': 'equal',  # 权重方案：equal, inverse_variance, market_cap
        }

        super().__init__(data, params)

        # 计算必要的指标
        self._compute_indicators()

    def get_default_params(self) -> Dict[str, Any]:
        """返回默认参数"""
        return {
            'n_stocks': 10,
            'window_size': 60,
            'min_observations': 30,
            'eigenvalue_threshold': 1.5,
            'regime_lookback': 20,
            'regime_threshold': 0.3,
            'weight_scheme': 'equal',
        }

    def validate_params(self):
        """验证参数"""
        if self.params['n_stocks'] <= 0:
            raise ValueError("n_stocks 必须大于0")

        if self.params['window_size'] < self.params['min_observations']:
            raise ValueError("window_size 必须大于min_observations")

        if self.params['window_size'] > len(self.data):
            raise ValueError("数据长度不足以支持设定的window_size")

    def _compute_indicators(self):
        """计算所有必要的指标"""
        df = self.data.copy()

        # 1. 计算收益率
        df['returns'] = df['close'].pct_change()

        # 2. 应用股票选择（如果有多个股票）
        if len(df['symbol'].unique()) > 1:
            # 选择表现最好的N只股票
            df['total_return'] = (1 + df['returns']).prod(axis=0) - 1
            top_stocks = df.groupby('symbol')['total_return'].sum().nlargest(self.params['n_stocks']).index
            df = df[df['symbol'].isin(top_stocks)]

        # 3. 计算滚动相关矩阵
        df['correlation_matrix'] = df.groupby(df.index.to_period('M')).apply(
            self._compute_windowed_correlation
        )

        # 4. 计算Marchenko-Pastur边界
        df['mp_bounds'] = df.groupby(df.index.to_period('M')).apply(
            self._compute_mp_bounds
        )

        # 5. 应用RMT过滤
        df['filtered_matrix'] = df.apply(
            lambda x: self._apply_rmt_filter(x['correlation_matrix'], x['mp_bounds'])
            if isinstance(x['correlation_matrix'], np.ndarray) and isinstance(x['mp_bounds'], dict)
            else None, axis=1
        )

        # 6. 计算因子结构
        df['factor_regime'] = df.apply(
            lambda x: self._detect_regime(x['filtered_matrix'], x['correlation_matrix'])
            if isinstance(x['filtered_matrix'], np.ndarray)
            else None, axis=1
        )

        # 7. 计算权重
        df['weights'] = df.apply(self._compute_weights, axis=1)

        self.indicators = df

    def _compute_windowed_correlation(self, window: pd.DataFrame) -> np.ndarray:
        """计算滚动窗口内的相关矩阵"""
        if len(window) < self.params['min_observations']:
            return np.eye(1)  # 返回单位矩阵

        # 获取唯一的股票
        unique_symbols = window['symbol'].unique()
        n_stocks = len(unique_symbols)

        if n_stocks == 1:
            # 单股票情况，返回单位矩阵
            return np.eye(1)

        # 多股票情况，计算相关矩阵
        # 创建一个共同的索引
        min_length = min(len(window[window['symbol'] == s]) for s in unique_symbols)
        if min_length < 2:
            return np.eye(n_stocks)

        # 获取每个股票的最后min_length个收益值
        returns_matrix = []
        for symbol in unique_symbols:
            symbol_data = window[window['symbol'] == symbol]['returns'].dropna()
            if len(symbol_data) >= min_length:
                returns_matrix.append(symbol_data.tail(min_length).values)

        if not returns_matrix or len(returns_matrix[0]) < 2:
            return np.eye(n_stocks)

        # 转置矩阵使股票为列
        returns_matrix = np.column_stack(returns_matrix)

        # 计算相关矩阵
        corr_matrix = np.corrcoef(returns_matrix)

        # 处理NaN值
        np.fill_diagonal(corr_matrix, 1)  # 对角线设为1

        return corr_matrix

    def _compute_mp_bounds(self, window: pd.DataFrame) -> Dict[str, float]:
        """计算Marchenko-Pastur边界"""
        n_assets = 1  # 单股票情况
        n_observations = len(window)

        if n_observations < self.params['min_observations']:
            return {'lambda_plus': 1.0, 'lambda_minus': 0.0}

        # Marchenko-Pastur参数
        gamma = n_assets / n_observations
        lambda_plus = (1 + np.sqrt(gamma))**2
        lambda_minus = (1 - np.sqrt(gamma))**2

        return {'lambda_plus': lambda_plus, 'lambda_minus': lambda_minus}

    def _apply_rmt_filter(self, corr_matrix: np.ndarray, mp_bounds: Dict[str, float]) -> np.ndarray:
        """应用RMT过滤相关矩阵"""
        eigenvalues, eigenvectors = eigh(corr_matrix)

        # 计算MP上界的倍数
        threshold = mp_bounds['lambda_plus'] * self.params['eigenvalue_threshold']

        # 识别信号特征值（超过阈值的）
        signal_mask = eigenvalues > threshold

        # 创建过滤后的特征值矩阵
        filtered_eigenvalues = eigenvalues.copy()
        filtered_eigenvalues[~signal_mask] = mp_bounds['lambda_plus']  # 将噪声特征值替换为MP上界

        # 重构相关矩阵
        filtered_matrix = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T

        # 确保矩阵是正定的
        np.fill_diagonal(filtered_matrix, 1)

        return filtered_matrix

    def _detect_regime(self, filtered_matrix: np.ndarray, original_matrix: np.ndarray) -> Dict[str, Any]:
        """检测因子regime"""
        # 计算原始和过滤矩阵的差异
        difference = np.abs(filtered_matrix - original_matrix)

        # 计算信号强度（过滤后保留的方差比例）
        original_variance = np.sum(np.diag(original_matrix))
        filtered_variance = np.sum(np.diag(filtered_matrix))
        signal_strength = filtered_variance / (original_variance + 1e-8)

        # 计算因子变化率（最近N天的变化）
        regime_score = np.mean(np.diag(filtered_matrix)) - 1.0  # 与单位矩阵的距离

        # Regime分类
        if regime_score > self.params['regime_threshold']:
            regime = 'bull'
        elif regime_score < -self.params['regime_threshold']:
            regime = 'bear'
        else:
            regime = 'neutral'

        return {
            'regime': regime,
            'signal_strength': float(signal_strength),
            'regime_score': float(regime_score),
            'matrix_difference': float(np.mean(difference)),
            'n_signal_factors': int(np.sum(np.diag(filtered_matrix) > 1.01)),  # 识别真正的主因子
        }

    def _compute_weights(self, row: pd.Series) -> Dict[str, float]:
        """根据不同方案计算权重"""
        n_stocks = self.params['n_stocks']
        weights = {}

        if row['factor_regime'] is None:
            return {'weight_1': 0.0, 'weight_2': 0.0}

        regime = row['factor_regime']['regime']
        signal_strength = row['factor_regime']['signal_strength']

        if self.params['weight_scheme'] == 'equal':
            # 等权重分配
            weights['weight_1'] = 0.5 * signal_strength
            weights['weight_2'] = 0.5 * signal_strength

        elif self.params['weight_scheme'] == 'inverse_variance':
            # 反向方差权重（简化实现）
            base_weight = signal_strength / n_stocks
            weights['weight_1'] = base_weight * 1.2  # 多头略高
            weights['weight_2'] = base_weight * 0.8

        elif self.params['weight_scheme'] == 'market_cap':
            # 市值权重（简化实现）
            if regime == 'bull':
                weights['weight_1'] = 0.7 * signal_strength  # 多头高权重
                weights['weight_2'] = 0.3 * signal_strength
            elif regime == 'bear':
                weights['weight_1'] = 0.3 * signal_strength  # 空头高权重
                weights['weight_2'] = 0.7 * signal_strength
            else:
                weights['weight_1'] = 0.5 * signal_strength
                weights['weight_2'] = 0.5 * signal_strength

        return weights

    def generate_signals(self) -> List[Dict]:
        """生成交易信号"""
        signals = []
        df = self.indicators

        for i in range(self.params['window_size'], len(df)):
            current_row = df.iloc[i]

            # 检查是否有有效的regime信息
            if current_row['factor_regime'] is None:
                continue

            regime = current_row['factor_regime']['regime']
            signal_strength = current_row['factor_regime']['signal_strength']
            regime_score = current_row['factor_regime']['regime_score']

            # 检查是否有有效的权重
            weights = current_row['weights']
            if not isinstance(weights, dict):
                weights = {'weight_1': 0.0, 'weight_2': 0.0}

            # 交易逻辑：当因子结构显示regime shift时交易
            # 1. 信号强度足够强
            if signal_strength < 0.1:
                self._record_signal(
                    timestamp=current_row.name,
                    action='hold',
                    price=current_row['close'],
                    reason='RMT：信号强度不足'
                )
                continue

            # 2. 检测regime shift（比较前regime变化）
            if i > self.params['window_size']:
                prev_regime = df.iloc[i-1]['factor_regime']['regime'] if df.iloc[i-1]['factor_regime'] is not None else 'neutral'

                if regime != prev_regime:
                    # 发生了regime shift
                    if regime == 'bull':
                        # 牛市regime - 做多
                        self._record_signal(
                            timestamp=current_row.name,
                            action='buy',
                            price=current_row['close'],
                            reason=f'RMT：regime shift至牛市，信号强度{signal_strength:.2f}',
                            signal_strength=signal_strength,
                            regime_score=regime_score,
                            factor_structure=current_row['filtered_matrix'] if isinstance(current_row['filtered_matrix'], np.ndarray) else None
                        )
                    elif regime == 'bear':
                        # 熊市regime - 做空
                        self._record_signal(
                            timestamp=current_row.name,
                            action='sell',
                            price=current_row['close'],
                            reason=f'RMT：regime shift至熊市，信号强度{signal_strength:.2f}',
                            signal_strength=signal_strength,
                            regime_score=regime_score,
                            factor_structure=current_row['filtered_matrix'] if isinstance(current_row['filtered_matrix'], np.ndarray) else None
                        )
                else:
                    # 相同regime - 根据信号强度调整仓位
                    if regime == 'bull' and regime_score > 0:
                        self._record_signal(
                            timestamp=current_row.name,
                            action='buy',
                            price=current_row['close'],
                            reason=f'RMT：牛市regime持续，增强多头',
                            signal_strength=signal_strength,
                            regime_score=regime_score
                        )
                    elif regime == 'bear' and regime_score < 0:
                        self._record_signal(
                            timestamp=current_row.name,
                            action='sell',
                            price=current_row['close'],
                            reason=f'RMT：熊市regime持续，增强空头',
                            signal_strength=signal_strength,
                            regime_score=regime_score
                        )
                    else:
                        self._record_signal(
                            timestamp=current_row.name,
                            action='hold',
                            price=current_row['close'],
                            reason=f'RMT：{regime} regime，等待信号'
                        )
            else:
                # 初始状态
                self._record_signal(
                    timestamp=current_row.name,
                    action='hold',
                    price=current_row['close'],
                    reason='RMT：初始化阶段'
                )

        return self.signals

    def get_rmt_analysis(self) -> Dict:
        """返回RMT分析结果"""
        if not hasattr(self, 'indicators') or self.indicators.empty:
            return {}

        df = self.indicators

        # 计算信号特征数量
        signal_counts = []
        for _, row in df.iterrows():
            if row['factor_regime'] is not None:
                signal_counts.append(row['factor_regime']['n_signal_factors'])

        # 计算regime分布
        regimes = []
        for _, row in df.iterrows():
            if row['factor_regime'] is not None:
                regimes.append(row['factor_regime']['regime'])

        regime_counts = pd.Series(regimes).value_counts() if regimes else {}

        # 计算平均信号强度
        signal_strengths = []
        for _, row in df.iterrows():
            if row['factor_regime'] is not None:
                signal_strengths.append(row['factor_regime']['signal_strength'])

        return {
            'average_signal_factors': float(np.mean(signal_counts)) if signal_counts else 0,
            'regime_distribution': dict(regime_counts) if regime_counts else {},
            'average_signal_strength': float(np.mean(signal_strengths)) if signal_strengths else 0,
            'current_regime': df['factor_regime'].iloc[-1]['regime'] if df['factor_regime'].iloc[-1] is not None else 'unknown',
            'current_signal_strength': float(df['factor_regime'].iloc[-1]['signal_strength']) if df['factor_regime'].iloc[-1] is not None else 0,
        }