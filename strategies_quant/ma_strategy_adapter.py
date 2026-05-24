#!/usr/bin/env python3
"""
移动平均策略适配器 - 将quant_trade-main中的ma_strategy.py适配到统一策略框架

# 整合适配 - 自动添加

功能:
1. 包装原始ma_strategy.MovingAverageStrategy
2. 提供统一的接口和配置
3. 信号格式标准化
4. 性能指标增强
"""

# BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加原始项目路径
original_project_path = "/Users/chengming/downloads/quant_trade-main"
sys.path.append(os.path.join(original_project_path, "backtest", "src"))

# 导入统一策略基类
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入统一策略基类，如果失败则使用虚拟基类
try:
    from core.unified_strategy_base import UnifiedStrategyBase, BackwardCompatibleStrategy
    UNIFIED_BASE_AVAILABLE = True
    print("✅ 成功导入统一策略基类")
except ImportError as e:
    print(f"⚠️ 无法导入统一策略基类: {e}")
    print("将使用虚拟基类实现")
    UNIFIED_BASE_AVAILABLE = False
    
    # 定义虚拟基类
    class UnifiedStrategyBase:
        """虚拟统一策略基类"""
        def __init__(self, data, params, strategy_name=None):
            self.data = data.copy()
            self.params = params.copy()
            self.strategy_name = strategy_name or "VirtualStrategy"
            self.standard_signals = []
            self.performance_metrics = {}
            
        def get_performance_metrics(self):
            return self.performance_metrics
            
        def print_summary(self):
            print(f"虚拟策略摘要: {self.strategy_name}")
            
    class BackwardCompatibleStrategy:
        """虚拟向后兼容策略类"""
        def __init__(self, original_strategy, strategy_name=None):
            self.original_strategy = original_strategy
            self.strategy_name = strategy_name or "BackwardCompatibleVirtual"
            self.params = {}

# 尝试导入原始移动平均策略
try:
    from strategies.ma_strategy import MovingAverageStrategy as OriginalMovingAverageStrategy
    ORIGINAL_MA_STRATEGY_AVAILABLE = True
    print("✅ 成功导入原始MovingAverageStrategy")
except ImportError as e:
    print(f"⚠️ 无法导入原始MovingAverageStrategy: {e}")
    print("将使用兼容实现")
    ORIGINAL_MA_STRATEGY_AVAILABLE = False


class MovingAverageStrategyAdapter(UnifiedStrategyBase, BaseStrategy):
    """
    移动平均策略适配器 - 统一接口的移动平均策略
    
    提供两种模式:
    1. 包装模式: 使用原始MovingAverageStrategy (如果可用)
    2. 原生模式: 重新实现移动平均策略逻辑
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 params: Dict[str, Any],
                 strategy_name: Optional[str] = None,
                 use_original: bool = True):
        """
        初始化移动平均策略适配器
        
        参数:
            data: 交易数据
            params: 策略参数
            strategy_name: 策略名称
            use_original: 是否使用原始实现 (如果可用)
        """
        self.use_original = use_original and ORIGINAL_MA_STRATEGY_AVAILABLE
        
        # 存储原始策略实例 (如果使用包装模式)
        self.original_strategy = None
        
        # 调用父类初始化
        super().__init__(data, params, strategy_name)
        
        # 如果使用原始实现，创建原始策略实例
        if self.use_original:
            self._create_original_strategy()
    
    def _create_original_strategy(self):
        """创建原始策略实例"""
        try:
            self.original_strategy = OriginalMovingAverageStrategy(
                data=self.data.copy(),
                params=self.params
            )
            print(f"使用原始MovingAverageStrategy实现")
        except Exception as e:
            print(f"创建原始策略实例失败，将使用原生实现: {e}")
            self.use_original = False
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'short_window': 5,
            'long_window': 20,
            'threshold': 0.01,
            'selection_method': 'score',  # 'score' 或 'signal'
            'enable_multi_stock': True,
            'min_data_length': 50,
            'position_state_enabled': False,
            'enabled': True,
            'version': '1.1.0',
            'description': '双均线金叉死叉策略适配器'
        }
    
    def _validate_params(self):
        """验证策略参数"""
        # 验证窗口参数
        if self.params['short_window'] >= self.params['long_window']:
            raise ValueError(f"短期窗口({self.params['short_window']})必须小于长期窗口({self.params['long_window']})")
        
        if self.params['short_window'] < 1 or self.params['long_window'] < 1:
            raise ValueError("窗口大小必须大于0")
        
        # 验证阈值
        if not 0 <= self.params['threshold'] <= 0.1:
            print(f"警告: 阈值 {self.params['threshold']} 可能不合理，建议范围 0-0.1")
    
    def generate_signals(self) -> List[Dict]:
        """
        生成移动平均策略信号
        
        如果use_original=True且原始策略可用，则使用原始策略
        否则使用原生实现
        """
        if self.use_original and self.original_strategy:
            return self._generate_signals_original()
        else:
            return self._generate_signals_native()
    
    def _generate_signals_original(self) -> List[Dict]:
        """使用原始策略生成信号"""
        print(f"使用原始MovingAverageStrategy生成信号...")
        
        try:
            # 调用原始策略
            raw_signals = self.original_strategy.generate_signals()
            
            if not raw_signals:
                print("原始策略未生成任何信号")
                return []
            
            print(f"原始策略生成 {len(raw_signals)} 个信号")
            
            # 转换信号格式
            converted_signals = []
            for signal in raw_signals:
                converted = self._convert_original_signal(signal)
                if converted:
                    converted_signals.append(converted)
            
            print(f"转换后得到 {len(converted_signals)} 个标准信号")
            return converted_signals
            
        except Exception as e:
            print(f"原始策略执行失败: {e}")
            print("回退到原生实现")
            return self._generate_signals_native()
    
    def _convert_original_signal(self, original_signal: Dict) -> Optional[Dict]:
        """转换原始信号格式为标准格式"""
        try:
            # 提取基本信息
            timestamp = original_signal.get('timestamp')
            action = original_signal.get('action', 'hold')
            symbol = original_signal.get('symbol', 'UNKNOWN')
            
            # 确定价格
            price = original_signal.get('price', 0.0)
            if price == 0.0 and timestamp is not None:
                # 尝试从数据中获取价格
                try:
                    price = self.data.loc[timestamp, 'close']
                except:
                    # 如果无法获取，使用最近的价格
                    if not self.data.empty:
                        price = self.data['close'].iloc[-1]
            
            # 确定信号类型
            signal_type = 'ma_cross'
            if action == 'buy':
                signal_type = 'ma_golden_cross'
            elif action == 'sell':
                signal_type = 'ma_death_cross'
            
            # 计算置信度
            confidence = self._calculate_ma_signal_confidence(original_signal)
            
            # 构建标准信号
            standard_signal = {
                'timestamp': timestamp,
                'action': action,
                'price': price,
                'symbol': symbol,
                'type': signal_type,
                'confidence': confidence,
                'features': {
                    'strategy': 'MovingAverage',
                    'short_window': self.params['short_window'],
                    'long_window': self.params['long_window'],
                    'threshold': self.params['threshold'],
                    'original_signal': original_signal
                }
            }
            
            return standard_signal
            
        except Exception as e:
            print(f"转换原始信号失败: {e}")
            return None
    
    def _calculate_ma_signal_confidence(self, signal: Dict) -> float:
        """计算移动平均信号的置信度"""
        # 基础置信度
        confidence = 0.7
        
        # 可以根据信号特征调整置信度
        # 例如：窗口大小差异、价格位置等
        
        return min(max(confidence, 0.1), 0.95)  # 限制在0.1-0.95范围内
    
    def _generate_signals_native(self) -> List[Dict]:
        """使用原生实现生成信号"""
        print(f"使用原生移动平均策略生成信号...")
        
        data = self.data.copy()
        signals = []
        
        # 确保数据足够
        if len(data) < max(self.params['long_window'], self.params['min_data_length']):
            print(f"数据不足: {len(data)} < {max(self.params['long_window'], self.params['min_data_length'])}")
            return []
        
        # 计算移动平均线
        short_ma = data['close'].rolling(window=self.params['short_window'], min_periods=1).mean()
        long_ma = data['close'].rolling(window=self.params['long_window'], min_periods=1).mean()
        
        # 计算差值百分比
        ma_diff_pct = (short_ma - long_ma) / long_ma * 100
        
        # 生成信号
        in_position = False
        last_buy_price = 0
        last_buy_time = None
        
        for i in range(1, len(data)):
            prev_diff = ma_diff_pct.iloc[i-1]
            curr_diff = ma_diff_pct.iloc[i]
            prev_short = short_ma.iloc[i-1]
            curr_short = short_ma.iloc[i]
            prev_long = long_ma.iloc[i-1]
            curr_long = long_ma.iloc[i]
            
            timestamp = data.index[i]
            price = data['close'].iloc[i]
            
            # 金叉买入信号
            golden_cross = (prev_short <= prev_long and curr_short > curr_long and 
                           abs(curr_diff) >= self.params['threshold'] * 100)
            
            # 死叉卖出信号
            death_cross = (prev_short >= prev_long and curr_short < curr_long and 
                          abs(curr_diff) >= self.params['threshold'] * 100)
            
            # 生成买入信号
            if golden_cross and not in_position:
                signal = {
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'type': 'ma_golden_cross',
                    'confidence': 0.7 + min(abs(curr_diff) / 10.0, 0.25),  # 基于差异的置信度
                    'features': {
                        'short_ma': curr_short,
                        'long_ma': curr_long,
                        'ma_diff': curr_diff,
                        'short_window': self.params['short_window'],
                        'long_window': self.params['long_window'],
                        'position': 'entry'
                    }
                }
                
                # 添加symbol信息
                if 'symbol' in data.columns:
                    signal['symbol'] = data['symbol'].iloc[i]
                
                signals.append(signal)
                in_position = True
                last_buy_price = price
                last_buy_time = timestamp
            
            # 生成卖出信号
            elif death_cross and in_position:
                # 计算持仓收益
                holding_return = (price - last_buy_price) / last_buy_price if last_buy_price > 0 else 0
                
                signal = {
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'type': 'ma_death_cross',
                    'confidence': 0.7 + min(abs(curr_diff) / 10.0, 0.25),
                    'features': {
                        'short_ma': curr_short,
                        'long_ma': curr_long,
                        'ma_diff': curr_diff,
                        'short_window': self.params['short_window'],
                        'long_window': self.params['long_window'],
                        'position': 'exit',
                        'holding_return': holding_return,
                        'holding_bars': i - (data.index.get_loc(last_buy_time) if last_buy_time else i)
                    }
                }
                
                # 添加symbol信息
                if 'symbol' in data.columns:
                    signal['symbol'] = data['symbol'].iloc[i]
                
                signals.append(signal)
                in_position = False
                last_buy_price = 0
                last_buy_time = None
        
        print(f"原生策略生成 {len(signals)} 个信号")
        return signals
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析策略性能（专为移动平均策略设计）"""
        signals = self.standard_signals
        
        if not signals:
            return {'error': '没有信号可分析'}
        
        # 分离买入和卖出信号
        buy_signals = [s for s in signals if s['action'] == 'buy']
        sell_signals = [s for s in signals if s['action'] == 'sell']
        
        # 计算配对交易
        trades = []
        for i in range(min(len(buy_signals), len(sell_signals))):
            buy = buy_signals[i]
            sell = sell_signals[i] if i < len(sell_signals) else None
            
            if sell and sell['timestamp'] > buy['timestamp']:
                return_pct = (sell['price'] - buy['price']) / buy['price'] * 100
                holding_days = (sell['timestamp'] - buy['timestamp']).days
                
                trades.append({
                    'buy_time': buy['timestamp'],
                    'sell_time': sell['timestamp'],
                    'buy_price': buy['price'],
                    'sell_price': sell['price'],
                    'return_pct': return_pct,
                    'holding_days': holding_days,
                    'type': buy.get('type', 'unknown')
                })
        
        # 计算性能指标
        if trades:
            returns = [t['return_pct'] for t in trades]
            winning_trades = [r for r in returns if r > 0]
            
            performance = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(trades) - len(winning_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'avg_return': np.mean(returns) if returns else 0,
                'max_return': max(returns) if returns else 0,
                'min_return': min(returns) if returns else 0,
                'total_return': sum(returns),
                'avg_holding_days': np.mean([t['holding_days'] for t in trades]) if trades else 0,
                'trades': trades[:10]  # 只保存前10个交易详情
            }
        else:
            performance = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'message': '没有完整的交易配对'
            }
        
        # 添加到性能指标
        self.performance_metrics.update({
            'ma_strategy_analysis': performance,
            'buy_signals_count': len(buy_signals),
            'sell_signals_count': len(sell_signals),
            'parameters': self.params
        })
        
        return performance
    
    def print_ma_analysis(self):
        """打印移动平均策略分析"""
        print(f"\n{'='*60}")
        print(f"移动平均策略分析: {self.strategy_name}")
        print(f"{'='*60}")
        
        performance = self.analyze_performance()
        
        print(f"\n📊 交易统计:")
        print(f"  总交易: {performance.get('total_trades', 0)}")
        print(f"  盈利交易: {performance.get('winning_trades', 0)}")
        print(f"  亏损交易: {performance.get('losing_trades', 0)}")
        print(f"  胜率: {performance.get('win_rate', 0):.2%}")
        
        if performance.get('total_trades', 0) > 0:
            print(f"\n💰 收益统计:")
            print(f"  平均收益: {performance.get('avg_return', 0):.2f}%")
            print(f"  最大收益: {performance.get('max_return', 0):.2f}%")
            print(f"  最小收益: {performance.get('min_return', 0):.2f}%")
            print(f"  总收益: {performance.get('total_return', 0):.2f}%")
            print(f"  平均持仓天数: {performance.get('avg_holding_days', 0):.1f}天")
        
        print(f"\n⚙️ 策略参数:")
        print(f"  短期窗口: {self.params['short_window']}")
        print(f"  长期窗口: {self.params['long_window']}")
        print(f"  阈值: {self.params['threshold']}")
        print(f"  实现方式: {'原始策略' if self.use_original and self.original_strategy else '原生实现'}")
        
        print(f"\n📈 信号统计:")
        metrics = self.get_performance_metrics()
        signal_summary = metrics.get('signal_summary', {})
        print(f"  总信号: {signal_summary.get('total_signals', 0)}")
        print(f"  买入信号: {signal_summary.get('buy_signals', 0)}")
        print(f"  卖出信号: {signal_summary.get('sell_signals', 0)}")
        
        print(f"\n{'='*60}")


# ========== 向后兼容包装器 ==========

class OriginalMAStrategyWrapper(BackwardCompatibleStrategy):
    """
    原始移动平均策略包装器 - 专为MovingAverageStrategy设计的包装器
    
    使用示例:
        # 直接使用原始策略
        original_strategy = OriginalMovingAverageStrategy(data, params)
        wrapper = OriginalMAStrategyWrapper(original_strategy)
        
        # 或者通过管理器创建
        manager.create_strategy_instance('OriginalMAWrapper', data, params)
    """
    
    def __init__(self, original_strategy, strategy_name=None):
        """
        初始化包装器
        
        参数:
            original_strategy: 原始MovingAverageStrategy实例
            strategy_name: 策略名称
        """
        super().__init__(original_strategy, strategy_name)
        
        # 设置默认参数
        self.params.update({
            'strategy_type': 'OriginalMovingAverage',
            'wrapped': True,
            'original_class': original_strategy.__class__.__name__
        })
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析原始策略性能"""
        return MovingAverageStrategyAdapter.analyze_performance(self)


# ========== 注册函数 ==========

def register_ma_strategies(manager):
    """
    注册移动平均策略到策略管理器
    
    参数:
        manager: StrategyManager实例
    """
    # 注册适配器版本
    manager.register_strategy(
        name="MovingAverageAdapter",
        strategy_class=MovingAverageStrategyAdapter,
        default_config={
            'short_window': 5,
            'long_window': 20,
            'threshold': 0.01,
            'description': '移动平均策略适配器（支持原始/原生两种实现）'
        },
        description="移动平均策略适配器，支持原始实现和原生实现"
    )
    
    # 注册包装器版本（如果原始策略可用）
    if ORIGINAL_MA_STRATEGY_AVAILABLE:
        manager.register_strategy(
            name="OriginalMAWrapper",
            strategy_class=OriginalMAStrategyWrapper,
            default_config={
                'description': '原始移动平均策略包装器'
            },
            description="原始MovingAverageStrategy的直接包装器"
        )
    
    print(f"注册了 {2 if ORIGINAL_MA_STRATEGY_AVAILABLE else 1} 个移动平均策略版本")


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("测试移动平均策略适配器...")
    
    # 生成示例数据
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 105,
        'low': np.random.randn(200).cumsum() + 95,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200),
        'symbol': 'TEST'
    }, index=dates)
    
    print(f"\n1. 测试适配器版本 (使用原始实现: {ORIGINAL_MA_STRATEGY_AVAILABLE}):")
    
    adapter = MovingAverageStrategyAdapter(
        data=data,
        params={'short_window': 5, 'long_window': 20, 'threshold': 0.02},
        strategy_name="MA_Adapter_Test"
    )
    
    signals = adapter.generate_standard_signals()
    print(f"生成 {len(signals)} 个标准化信号")
    
    adapter.print_summary()
    adapter.print_ma_analysis()
    
    # 测试原生实现
    print(f"\n2. 测试原生实现版本:")
    
    adapter_native = MovingAverageStrategyAdapter(
        data=data,
        params={'short_window': 10, 'long_window': 30, 'threshold': 0.015},
        strategy_name="MA_Native_Test",
        use_original=False
    )
    
    signals_native = adapter_native.generate_standard_signals()
    print(f"生成 {len(signals_native)} 个标准化信号")
    
    adapter_native.print_ma_analysis()
    
    # 测试策略管理器集成
    print(f"\n3. 测试策略管理器集成:")
    
    from managers.strategy_manager import StrategyManager
    
    manager = StrategyManager(
        name="MATestManager",
        config_dir="./test_ma_configs",
        results_dir="./test_ma_results"
    )
    
    register_ma_strategies(manager)
    
    # 运行策略
    result = manager.run_strategy(
        strategy_name="MovingAverageAdapter",
        data=data,
        config={'short_window': 8, 'long_window': 21},
        save_results=True
    )
    
    print(f"\n✅ 移动平均策略适配器测试完成")

class MaStrategyAdapterStrategy(BaseStrategy):
    """基于ma_strategy_adapter的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.name = "MaStrategyAdapterStrategy"
        self.description = "基于ma_strategy_adapter的策略"
        
    def calculate_signals(self):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self):
        """MA策略适配器生成交易信号"""
        import numpy as np
        df = self.data
        short_w = self.params.get('short_window', 5)
        long_w = self.params.get('long_window', 20)
        short_ma = df['close'].rolling(short_w).mean()
        long_ma = df['close'].rolling(long_w).mean()
        for i in range(long_w, len(df)):
            sym = df['symbol'].iloc[i] if 'symbol' in df.columns else 'DEFAULT'
            price = float(df['close'].iloc[i])
            if short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i-1] <= long_ma.iloc[i-1]:
                self._record_signal(df.index[i], 'buy', sym, price)
            elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i-1] >= long_ma.iloc[i-1]:
                self._record_signal(df.index[i], 'sell', sym, price)
        return self.signals
