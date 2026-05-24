"""
Copula Dependency Strategy - 基于 Copula 理论的尾部依赖检测策略

策略核心思想：
使用经验 Copula 检测个股收益与市场指数之间的尾部依赖关系变化
- 上尾部依赖增强 → 牛市 regime
- 下尾部依赖增强 → 熊市 regime

使用 Clayton copula 检测下尾部（暴跌相关）
使用 Gumbel copula 检测上尾部（暴涨相关）
结合 ATR trailing stop 实现风险控制
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入 BaseStrategy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.base_strategy import BaseStrategy

try:
    from copulae import Clayton, Gumbel, Gaussian
    from copulae.core import BaseCopula
    COPULA_AVAILABLE = True
except ImportError:
    # Fallback implementations when copulae is not available
    COPULA_AVAILABLE = False
    print("Warning: copulae package not available. Using simplified implementations.")


class CopulaDependencyStrategyCore:
    """Copula 依赖检测策略核心实现"""

    def __init__(self, data: pd.DataFrame, params: dict = None):
        """
        初始化 Copula 策略

        Args:
            data: 包含 open, high, low, close 数据的 DataFrame
            params: 参数字典
        """
        self.data = data.copy()
        self.params = {**self.get_default_params(), **(params or {})}

        # 策略属性
        self.strategy_name = "CopulaDependency"
        self.strategy_description = "基于 Copula 理论的尾部依赖检测策略"
        self.strategy_category = "ml"
        self.signals = []

        # 策略参数
        self.copula_window = self.params.get('copula_window', 252)  # Copula 计算窗口（约1年）
        self.tau_window = self.params.get('tau_window', 63)       # Kendall tau 计算窗口（约3个月）
        self.tail_threshold = self.params.get('tail_threshold', 0.1)  # 尾部依赖阈值
        self.position_size_factor = self.params.get('position_size_factor', 0.3)  # 仓位因子
        self.atr_multiplier = self.params.get('atr_multiplier', 2.0)  # ATR 倍数
        self.atr_period = self.params.get('atr_period', 14)  # ATR 周期

        # 初始化
        self._validate_data()
        self._precompute_indicators()

    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'copula_window': 252,    # Copula 计算窗口
            'tau_window': 63,        # Kendall tau 计算窗口
            'tail_threshold': 0.1,   # 尾部依赖阈值
            'position_size_factor': 0.3,  # 仓位因子
            'atr_multiplier': 2.0,  # ATR 倍数
            'atr_period': 14,       # ATR 周期
        }

    def validate_params(self):
        """验证参数"""
        if self.params['copula_window'] < 63:
            raise ValueError("Copula window must be at least 63")
        if self.params['tau_window'] < 21:
            raise ValueError("Tau window must be at least 21")
        if not 0.01 <= self.params['tail_threshold'] <= 0.5:
            raise ValueError("Tail threshold must be between 0.01 and 0.5")

    def _validate_data(self):
        """验证数据"""
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"缺少必需列: {missing}")

        if self.data.empty:
            raise ValueError("数据为空")

        if 'symbol' not in self.data.columns:
            self.data['symbol'] = 'DEFAULT'

    def _precompute_indicators(self):
        """预计算指标"""
        # 计算收益率
        self.data['returns'] = np.log(self.data['close'] / self.data['close'].shift(1))

        # 计算 ATR
        self.data['tr'] = self._calculate_true_range()
        self.data['atr'] = self.data['tr'].rolling(window=self.atr_period).mean()

        # 计算滚动波动率
        self.data['volatility'] = self.data['returns'].rolling(window=20).std()

        # 创建市场代理指数（简单等权重组合）
        self.data['market_proxy'] = self.data['close'].rolling(window=5).mean()
        self.data['market_returns'] = np.log(self.data['market_proxy'] / self.data['market_proxy'].shift(1))

    def _calculate_true_range(self) -> pd.Series:
        """计算真实波动范围"""
        high_low = self.data['high'] - self.data['low']
        high_close = abs(self.data['high'] - self.data['close'].shift(1))
        low_close = abs(self.data['low'] - self.data['close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        tr.iloc[0] = self.data['high'].iloc[0] - self.data['low'].iloc[0]

        return tr

    def _calculate_kendall_tau(self, x: np.ndarray, y: np.ndarray, window: int) -> float:
        """计算滚动 Kendall's tau"""
        if len(x) < window:
            return 0.0

        x_window = x[-window:]
        y_window = y[-window:]

        # 处理缺失值
        mask = ~(np.isnan(x_window) | np.isnan(y_window))
        if np.sum(mask) < window * 0.8:  # 如果缺失值太多，返回0
            return 0.0

        tau, p_value = kendalltau(x_window[mask], y_window[mask])
        return tau

    def _empirical_cdf(self, data: np.ndarray) -> np.ndarray:
        """经验累积分布函数"""
        sorted_idx = np.argsort(data)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(len(data))
        return ranks / (len(data) + 1)

    def _estimate_tail_dependency(self, u: np.ndarray, v: np.ndarray, tail: str = 'lower') -> float:
        """
        估计尾部依赖

        Args:
            u, v: 标准化的收益率序列
            tail: 'lower' 或 'upper' 尾部

        Returns:
            尾部依赖强度
        """
        n = len(u)
        if n < 50:
            return 0.0

        if tail == 'lower':
            # 下尾部依赖：P(V <= q | U <= q), q→0
            q_values = np.linspace(0.05, 0.2, 5)  # 尝试不同的分位数
            dependencies = []

            for q in q_values:
                mask_u = u <= q
                if np.sum(mask_u) > 0:
                    mask_v = v <= q
                    # 条件概率
                    dep = np.sum(mask_u & mask_v) / np.sum(mask_u)
                    dependencies.append(dep)

            return np.mean(dependencies) if dependencies else 0.0

        else:  # upper
            # 上尾部依赖：P(V > q | U > q), q→1
            q_values = np.linspace(0.8, 0.95, 5)
            dependencies = []

            for q in q_values:
                mask_u = u > q
                if np.sum(mask_u) > 0:
                    mask_v = v > q
                    dep = np.sum(mask_u & mask_v) / np.sum(mask_u)
                    dependencies.append(dep)

            return np.mean(dependencies) if dependencies else 0.0

    def _fit_copula(self, u: np.ndarray, v: np.ndarray, copula_type: str) -> Optional[BaseCopula]:
        """拟合 Copula"""
        if not COPULA_AVAILABLE:
            return None

        try:
            if copula_type == 'clayton':
                copula = Clayton(2)
            elif copula_type == 'gumbel':
                copula = Gumbel(2)
            else:
                return None

            copula.fit(np.column_stack([u, v]))
            return copula

        except Exception as e:
            print(f"Error fitting {copula_type} copula: {e}")
            return None

    def _calculate_copula_parameter(self, u: np.ndarray, v: np.ndarray,
                                   copula_type: str, window: int) -> float:
        """
        计算 Copula 参数（简化版本，当 copulae 不可用时）
        """
        if len(u) < window:
            return 0.0

        u_window = u[-window:]
        v_window = v[-window:]

        if copula_type == 'clayton':
            # Clayton copula 参数估计
            # 通过 Kendall's tau 估计 θ = 2τ/(1-τ)
            tau = kendalltau(u_window, v_window)[0]
            if tau > 0:
                return 2 * tau / (1 - tau)
            else:
                return 0.0

        elif copula_type == 'gumbel':
            # Gumbel copula 参数估计
            # τ = (1 - 1/θ), so θ = 1/(1 - τ)
            tau = kendalltau(u_window, v_window)[0]
            if tau > 0:
                return 1 / (1 - tau)
            else:
                return 1.0

        return 0.0

    def _detect_regime(self, lower_tail_dep: float, upper_tail_dep: float,
                      kendall_tau: float) -> str:
        """
        检测市场 regime

        Args:
            lower_tail_dep: 下尾部依赖
            upper_tail_dep: 上尾部依赖
            kendall_tau: Kendall's tau

        Returns:
            regime: 'bull', 'bear', 'neutral'
        """
        # 计算尾部依赖变化率
        lower_strength = lower_tail_dep - self.tail_threshold
        upper_strength = upper_tail_dep - self.tail_threshold

        if abs(kendall_tau) < 0.1:  # 弱相关性
            return 'neutral'

        if upper_strength > 0 and upper_strength > abs(lower_strength):
            return 'bull'
        elif lower_strength > 0 and lower_strength > abs(upper_strength):
            return 'bear'
        else:
            return 'neutral'

    def _calculate_position_size(self, regime: str, tail_dep_strength: float) -> float:
        """
        根据尾部依赖强度计算仓位
        """
        base_size = self.position_size_factor

        if regime == 'bull':
            # 牛市：上尾部依赖增强
            return base_size * (1 + tail_dep_strength * 2)
        elif regime == 'bear':
            # 熊市：下尾部依赖增强，减少仓位
            return base_size * (1 - tail_dep_strength)
        else:
            # 中性市场：标准仓位
            return base_size

    def _calculate_atr_stop(self, regime: str) -> float:
        """计算 ATR 止损"""
        atr = self.data['atr'].iloc[-1]

        if regime == 'bull':
            # 牛市：宽松止损，允许更大波动
            return self.atr_multiplier * atr
        elif regime == 'bear':
            # 熊市：紧止损，及时止损
            return self.atr_multiplier * 0.7 * atr
        else:
            # 中性：标准止损
            return self.atr_multiplier * atr

    def generate_signals(self) -> List[Dict]:
        """生成交易信号"""
        signals = []

        # 确保有足够的数据
        if len(self.data) < self.copula_window + 50:
            return []

        # 计算滚动指标
        stock_returns = self.data['returns'].values
        market_returns = self.data['market_returns'].values

        # 计算经验 CDF
        u_stock = self._empirical_cdf(stock_returns)
        u_market = self._empirical_cdf(market_returns)

        # 计算滚动 Kendall's tau
        kendall_tau = self._calculate_kendall_tau(
            stock_returns, market_returns, self.tau_window
        )

        # 计算尾部依赖
        lower_tail_dep = self._estimate_tail_dependency(u_stock, u_market, 'lower')
        upper_tail_dep = self._estimate_tail_dependency(u_stock, u_market, 'upper')

        #  Copula 参数（简化版）
        clayton_param = self._calculate_copula_parameter(
            u_stock, u_market, 'clayton', self.copula_window
        )
        gumbel_param = self._calculate_copula_parameter(
            u_stock, u_market, 'gumbel', self.copula_window
        )

        # 检测 regime
        regime = self._detect_regime(lower_tail_dep, upper_tail_dep, kendall_tau)

        # 计算仓位和止损
        if regime != 'neutral':
            tail_dep_strength = max(lower_tail_dep, upper_tail_dep)
            position_size = self._calculate_position_size(regime, tail_dep_strength)
            atr_stop = self._calculate_atr_stop(regime)

            current_price = self.data['close'].iloc[-1]
            stop_price = current_price - atr_stop if regime == 'bear' else current_price * (1 - atr_stop / current_price)
        else:
            position_size = 0
            stop_price = 0

        # 生成信号
        last_signal = signals[-1] if signals else {'action': 'hold'}

        # 判断是否需要切换 regime
        if len(self.data) > 100:  # 确保有足够历史数据
            # 计算 regime 变化
            regime_strength = {
                'bull': upper_tail_dep - self.tail_threshold,
                'bear': lower_tail_dep - self.tail_threshold,
                'neutral': 0
            }

            # 如果 regime 明显且与前一个信号不同
            max_strength = max(regime_strength.values())
            if max_strength > 0.05:  # 阈值
                current_regime = max(regime_strength, key=regime_strength.get)

                if last_signal.get('regime') != current_regime:
                    # Regime 切换信号
                    signal_action = 'buy' if current_regime == 'bull' else 'sell'

                    signals.append({
                        'timestamp': self.data.index[-1],
                        'action': signal_action,
                        'symbol': self.data['symbol'].iloc[-1],
                        'price': current_price,
                        'regime': current_regime,
                        'tail_dependency': {
                            'lower': lower_tail_dep,
                            'upper': upper_tail_dep,
                            'kendall_tau': kendall_tau
                        },
                        'copula_params': {
                            'clayton': clayton_param,
                            'gumbel': gumbel_param
                        },
                        'position_size': position_size,
                        'stop_price': stop_price,
                        'atr': self.data['atr'].iloc[-1]
                    })

        # 如果没有 regime 切换，但有持续信号
        elif regime != 'neutral' and last_signal.get('action') != 'hold':
            # 保持现有仓位，但更新止损
            signals.append({
                'timestamp': self.data.index[-1],
                'action': 'hold',
                'symbol': self.data['symbol'].iloc[-1],
                'price': current_price,
                'regime': regime,
                'tail_dependency': {
                    'lower': lower_tail_dep,
                    'upper': upper_tail_dep,
                    'kendall_tau': kendall_tau
                },
                'copula_params': {
                    'clayton': clayton_param,
                    'gumbel': gumbel_param
                },
                'position_size': position_size,
                'stop_price': stop_price,
                'atr': self.data['atr'].iloc[-1],
                'update_stop': True
            })

        # 记录当前状态供下一次使用
        if signals:
            last_signal = signals[-1]
            last_signal['regime'] = regime
            last_signal['tail_dependency'] = {
                'lower': lower_tail_dep,
                'upper': upper_tail_dep,
                'kendall_tau': kendall_tau
            }

        self.signals = signals
        return signals

    def screen(self) -> Dict:
        """实时选股判断"""
        signals = self.generate_signals()
        if not signals:
            return {
                'action': 'hold',
                'reason': '数据不足',
                'price': float(self.data['close'].iloc[-1])
            }

        last_signal = signals[-1]
        current_price = float(self.data['close'].iloc[-1])

        # 检查止损
        if last_signal.get('action') == 'buy' and 'stop_price' in last_signal:
            if current_price <= last_signal['stop_price']:
                return {
                    'action': 'sell',
                    'reason': f'触及止损价 {last_signal["stop_price"]:.2f}',
                    'price': current_price
                }

        # 返回当前信号
        return {
            'action': last_signal.get('action', 'hold'),
            'reason': f'{self.strategy_name}: {last_signal.get("regime", "neutral")} regime',
            'price': current_price
        }

    def get_regime_info(self) -> Dict:
        """获取当前 regime 信息"""
        if not self.signals:
            return {}

        last_signal = self.signals[-1]
        return {
            'regime': last_signal.get('regime', 'neutral'),
            'tail_dependency': last_signal.get('tail_dependency', {}),
            'copula_params': last_signal.get('copula_params', {}),
            'position_size': last_signal.get('position_size', 0),
            'stop_price': last_signal.get('stop_price', 0)
        }


class CopulaDependencyStrategy(BaseStrategy):
    """Copula 依赖策略 - 继承自 BaseStrategy"""

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        self._core_strategy = CopulaDependencyStrategyCore(data, params)
        self.signals = []

    def generate_signals(self) -> List[Dict]:
        """生成交易信号"""
        signals = self._core_strategy.generate_signals()

        # 转换为标准信号格式
        formatted_signals = []
        for signal in signals:
            formatted_signal = {
                'timestamp': signal['timestamp'],
                'action': signal['action'],
                'symbol': signal['symbol'],
                'price': signal['price']
            }

            # 添加额外信息
            if 'regime' in signal:
                formatted_signal['regime'] = signal['regime']
            if 'tail_dependency' in signal:
                formatted_signal['tail_dependency'] = signal['tail_dependency']
            if 'copula_params' in signal:
                formatted_signal['copula_params'] = signal['copula_params']
            if 'position_size' in signal:
                formatted_signal['position_size'] = signal['position_size']
            if 'stop_price' in signal:
                formatted_signal['stop_price'] = signal['stop_price']
            if 'atr' in signal:
                formatted_signal['atr'] = signal['atr']

            formatted_signals.append(formatted_signal)

        self.signals = formatted_signals
        return formatted_signals

    def screen(self) -> Dict:
        """实时选股判断"""
        return self._core_strategy.screen()

    def generate_signals(self) -> List[Dict]:
        """生成交易信号"""
        signals = self._core_strategy.generate_signals()

        # 转换为标准信号格式
        formatted_signals = []
        for signal in signals:
            formatted_signal = {
                'timestamp': signal['timestamp'],
                'action': signal['action'],
                'symbol': signal['symbol'],
                'price': signal['price']
            }

            # 添加额外信息
            if 'regime' in signal:
                formatted_signal['regime'] = signal['regime']
            if 'tail_dependency' in signal:
                formatted_signal['tail_dependency'] = signal['tail_dependency']
            if 'copula_params' in signal:
                formatted_signal['copula_params'] = signal['copula_params']
            if 'position_size' in signal:
                formatted_signal['position_size'] = signal['position_size']
            if 'stop_price' in signal:
                formatted_signal['stop_price'] = signal['stop_price']
            if 'atr' in signal:
                formatted_signal['atr'] = signal['atr']

            formatted_signals.append(formatted_signal)

        self.signals = formatted_signals
        return formatted_signals

    def screen(self) -> Dict:
        """实时选股判断"""
        return self._core_strategy.screen()


# 示例使用
if __name__ == "__main__":
    # 示例数据
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    })

    # 创建策略
    strategy = CopulaDependencyStrategy(data)

    # 生成信号
    signals = strategy.generate_signals()
    print("Generated signals:", signals)

    # 获取 regime 信息
    regime_info = strategy.get_regime_info()
    print("Regime info:", regime_info)