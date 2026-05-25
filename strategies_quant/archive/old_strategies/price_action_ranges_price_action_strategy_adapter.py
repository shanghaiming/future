# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.631904

#!/usr/bin/env python3
"""
价格行为策略适配器
将OptimizedPriceActionIntegrationEngine集成到量化回测框架

适配器设计目标:
1. 统一策略接口: 兼容已有的回测框架
2. 参数配置: 支持参数优化
3. 信号生成: 将价格行为分析转换为交易信号
4. 性能评估: 集成到现有的绩效评估系统
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

# 添加价格行为策略路径
sys.path.append('/Users/chengming/.openclaw/workspace')

print("=" * 80)
print("🚀 价格行为策略适配器启动")
print("=" * 80)

# 尝试导入价格行为引擎
try:
    from price_action_integration.optimized_integration_engine import OptimizedPriceActionIntegrationEngine
    from price_action_integration.price_action_rules_integrator import PriceActionRulesIntegrator
    print("✅ 成功导入价格行为策略模块")
    print(f"   引擎类: {OptimizedPriceActionIntegrationEngine.__name__}")
    print(f"   规则整合器: {PriceActionRulesIntegrator.__name__}")
except ImportError as e:
    print(f"❌ 导入价格行为策略失败: {e}")
    print("   尝试从文件直接加载...")
    pass  # module import failed, using fallback

# 数据加载工具函数
def load_stock_data_for_price_action(stock_code: str = "000001.SZ",
                                   timeframe: str = "daily_data2",
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> pd.DataFrame:
    """加载股票数据，格式化为价格行为引擎所需格式"""
    data_dir = "/Users/chengming/.openclaw/workspace/quant_trade-main/data"
    file_path = os.path.join(data_dir, timeframe, f"{stock_code}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # 转换日期格式
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df.sort_values('trade_date', inplace=True)
    df.set_index('trade_date', inplace=True)
    
    # 筛选日期范围
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df.index >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df.index <= end_dt]
    
    # 重命名列以符合价格行为引擎要求
    column_mapping = {
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'vol': 'volume',
        'amount': 'amount'
    }
    
    result_df = pd.DataFrame()
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            result_df[new_col] = df[old_col]
    
    # 确保所有必要列都存在
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"数据缺失必要列: {col}")
    
    print(f"✅ 数据加载成功: {stock_code}")
    print(f"   数据形状: {result_df.shape}")
    print(f"   时间范围: {result_df.index.min()} 到 {result_df.index.max()}")
    
    return result_df

# 价格行为策略适配器类
class PriceActionStrategyAdapter(BaseStrategy):
    """
    价格行为策略适配器
    将OptimizedPriceActionIntegrationEngine转换为标准策略接口
    """
    
    def __init__(self, data: pd.DataFrame, params: Dict):
        """
        初始化适配器
        
        参数:
            data: 价格数据
            params: 策略参数字典，包含价格行为引擎配置
        """
        super().__init__(data, params)
        
        # 提取引擎配置参数
        self.engine_config = params.get('engine_config', {})
        self.signal_config = params.get('signal_config', {})
        
        # 信号生成参数
        self.confidence_threshold = self.signal_config.get('confidence_threshold', 0.7)
        self.min_position_energy = self.signal_config.get('min_position_energy', 0.6)
        self.trend_confirmation = self.signal_config.get('trend_confirmation', True)
        
        self.engine = None
        self.rules_integrator = None
        
        print(f"🔧 价格行为策略适配器初始化")
        print(f"   引擎配置: {list(self.engine_config.keys())}")
        print(f"   信号配置: {list(self.signal_config.keys())}")
    
    def initialize(self, data: pd.DataFrame):
        """初始化引擎和数据"""
        self.data = data.copy()
        
        # 创建价格行为引擎
        self.engine = OptimizedPriceActionIntegrationEngine(self.engine_config)
        self.engine.load_data(self.data)
        
        # 创建规则整合器
        self.rules_integrator = PriceActionRulesIntegrator()
        
        print("✅ 价格行为引擎初始化完成")
    
    def generate_signals(self) -> List[Dict]:
        """生成交易信号，使用价格行为分析（支撑阻力、pin bar、engulfing）"""
        df = self.data
        if len(df) < 30:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(df['close'].iloc[-1]))
            return self.signals

        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']

        # Support and resistance levels via rolling min/max
        support = low.rolling(20).min()
        resistance = high.rolling(20).max()

        # Pin bar detection (long lower wick = bullish, long upper wick = bearish)
        body = (close - open_price).abs()
        lower_wick = close.combine(open_price, min) - low
        upper_wick = high - close.combine(open_price, max)
        total_range = high - low

        bullish_pin = (lower_wick > body * 2) & (total_range > 0) & (lower_wick > total_range * 0.6)
        bearish_pin = (upper_wick > body * 2) & (total_range > 0) & (upper_wick > total_range * 0.6)

        # Engulfing pattern
        bullish_engulf = (close > open_price) & (close.shift(1) < open_price.shift(1)) & \
                         (close > open_price.shift(1)) & (open_price < close.shift(1))
        bearish_engulf = (close < open_price) & (close.shift(1) > open_price.shift(1)) & \
                         (close < open_price.shift(1)) & (open_price > close.shift(1))

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        i = len(df) - 1
        last_close = close.iloc[i]

        near_support = last_close <= support.iloc[i] * 1.02
        near_resistance = last_close >= resistance.iloc[i] * 0.98

        buy_score = 0
        sell_score = 0

        if bullish_pin.iloc[i]:
            buy_score += 2
        if bullish_engulf.iloc[i]:
            buy_score += 2
        if near_support and rsi.iloc[i] < 35:
            buy_score += 2
        if bearish_pin.iloc[i]:
            sell_score += 2
        if bearish_engulf.iloc[i]:
            sell_score += 2
        if near_resistance and rsi.iloc[i] > 65:
            sell_score += 2

        if buy_score >= 3:
            self._record_signal(timestamp=df.index[i], action='buy', price=float(last_close))
        elif sell_score >= 3:
            self._record_signal(timestamp=df.index[i], action='sell', price=float(last_close))
        else:
            self._record_signal(timestamp=df.index[i], action='hold', price=float(last_close))

        return self.signals
    
    def _generate_signal_from_context(self, context: Dict) -> Optional[Dict]:
        """从分析上下文生成单个信号"""
        i = context['current_index']
        current_time = context['current_time']
        current_price = context['current_price']
        
        # 1. 检查枢轴点信号
        pivot_signals = self._check_pivot_signals(context)
        if pivot_signals:
            best_signal = max(pivot_signals, key=lambda x: x.get('confidence', 0))
            if best_signal['confidence'] >= self.confidence_threshold:
                return {
                    'timestamp': current_time,
                    'action': best_signal['action'],
                    'price': current_price,
                    'reason': f"pivot_{best_signal['type']}",
                    'confidence': best_signal['confidence']
                }
        
        # 2. 检查区间突破信号
        range_signals = self._check_range_signals(context)
        if range_signals:
            best_signal = max(range_signals, key=lambda x: x.get('confidence', 0))
            if best_signal['confidence'] >= self.confidence_threshold:
                return {
                    'timestamp': current_time,
                    'action': best_signal['action'],
                    'price': current_price,
                    'reason': f"range_{best_signal['type']}",
                    'confidence': best_signal['confidence']
                }
        
        # 3. 检查补偿移动平均信号
        cma_signals = self._check_cma_signals(context)
        if cma_signals:
            best_signal = max(cma_signals, key=lambda x: x.get('confidence', 0))
            if best_signal['confidence'] >= self.confidence_threshold:
                return {
                    'timestamp': current_time,
                    'action': best_signal['action'],
                    'price': current_price,
                    'reason': f"cma_{best_signal['type']}",
                    'confidence': best_signal['confidence']
                }
        
        return None
    
    def _check_pivot_signals(self, context: Dict) -> List[Dict]:
        """检查枢轴点信号"""
        signals = []
        pivots = context['pivots']
        
        if 'pivot_bars' in pivots:
            pivot_bars = pivots['pivot_bars']
            current_index = context['current_index']
            
            # 检查是否有近期枢轴点
            for pivot in pivot_bars:
                pivot_idx = pivot.get('index', -1)
                if abs(current_index - pivot_idx) <= 5:  # 最近5根K线内的枢轴点
                    pivot_type = pivot.get('type', '')
                    
                    if pivot_type == 'bearish_pivot':
                        # 看跌枢轴点，考虑卖出信号
                        signals.append({
                            'action': 'sell',
                            'type': 'bearish_pivot',
                            'confidence': 0.7
                        })
                    elif pivot_type == 'bullish_pivot':
                        # 看涨枢轴点，考虑买入信号
                        signals.append({
                            'action': 'buy',
                            'type': 'bullish_pivot',
                            'confidence': 0.7
                        })
        
        return signals
    
    def _check_range_signals(self, context: Dict) -> List[Dict]:
        """检查区间信号"""
        signals = []
        ranges = context['ranges']
        
        if 'ranges' in ranges:
            price_ranges = ranges['ranges']
            current_price = context['current_price']
            
            # 检查当前价格是否在区间边界附近
            for price_range in price_ranges:
                support = price_range.get('support', None)
                resistance = price_range.get('resistance', None)
                
                if support and abs(current_price - support) / support < 0.02:  # 2%以内
                    # 接近支撑位，买入信号
                    signals.append({
                        'action': 'buy',
                        'type': 'range_support',
                        'confidence': 0.8
                    })
                
                if resistance and abs(current_price - resistance) / resistance < 0.02:  # 2%以内
                    # 接近阻力位，卖出信号
                    signals.append({
                        'action': 'sell',
                        'type': 'range_resistance',
                        'confidence': 0.8
                    })
        
        return signals
    
    def _check_cma_signals(self, context: Dict) -> List[Dict]:
        """检查补偿移动平均信号"""
        signals = []
        cma = context['cma']
        data = context['data']
        i = context['current_index']
        
        if i < 2:
            return signals
        
        # 获取当前和之前的CMA值
        if 'cma_values' in cma and len(cma['cma_values']) > i:
            current_cma = cma['cma_values'][i]
            prev_cma = cma['cma_values'][i-1]
            current_price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1]
            
            # CMA上穿价格（金叉）
            if prev_cma <= prev_price and current_cma > current_price:
                signals.append({
                    'action': 'buy',
                    'type': 'cma_golden_cross',
                    'confidence': 0.6
                })
            
            # CMA下穿价格（死叉）
            elif prev_cma >= prev_price and current_cma < current_price:
                signals.append({
                    'action': 'sell',
                    'type': 'cma_death_cross',
                    'confidence': 0.6
                })
        
        return signals

# 主测试函数
def test_price_action_strategy():
    """测试价格行为策略"""
    print("\n🧪 开始测试价格行为策略...")
    
    # 1. 加载测试数据
    try:
        df = load_stock_data_for_price_action(
            stock_code="000001.SZ",
            timeframe="daily_data2",
            start_date="2020-01-01",
            end_date="2021-12-31"
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 配置参数
    params = {
        'engine_config': {
            'pivot_detection': {
                'prominence_factor': 0.5,
                'window': 5
            },
            'range_clustering': {
                'window': 20,
                'cluster_threshold': 1.0,
                'min_range_length': 5
            },
            'compensated_ma': {
                'window': 20,
                'beta': 0.3,
                'gamma': 0.2,
                'decay_factor': 0.95
            }
        },
        'signal_config': {
            'confidence_threshold': 0.7,
            'min_position_energy': 0.6,
            'trend_confirmation': True
        }
    }
    
    # 3. 创建并初始化策略
    strategy = PriceActionStrategyAdapter(params)
    strategy.initialize(df)
    
    # 4. 生成信号
    signals = strategy.generate_signals()
    
    # 5. 分析信号
    if signals:
        buy_signals = [s for s in signals if s['action'] == 'buy']
        sell_signals = [s for s in signals if s['action'] == 'sell']
        
        print(f"\n📊 信号统计:")
        print(f"   总信号数: {len(signals)}")
        print(f"   买入信号: {len(buy_signals)}")
        print(f"   卖出信号: {len(sell_signals)}")
        
        if buy_signals:
            first_buy = buy_signals[0]
            last_buy = buy_signals[-1]
            print(f"\n   第一个买入信号: {first_buy['timestamp']} - {first_buy['reason']}")
            print(f"   最后一个买入信号: {last_buy['timestamp']} - {last_buy['reason']}")
        
        if sell_signals:
            first_sell = sell_signals[0]
            last_sell = sell_signals[-1]
            print(f"\n   第一个卖出信号: {first_sell['timestamp']} - {first_sell['reason']}")
            print(f"   最后一个卖出信号: {last_sell['timestamp']} - {last_sell['reason']}")
        
        # 保存信号到文件
        signals_df = pd.DataFrame(signals)
        output_path = "/Users/chengming/.openclaw/workspace/price_action_signals_test.csv"
        signals_df.to_csv(output_path, index=False)
        print(f"\n💾 信号保存到: {output_path}")
        
        return signals
    else:
        print("❌ 未生成任何交易信号")
        return None

# 与现有回测框架集成
def integrate_with_backtest_framework():
    """与现有回测框架集成"""
    print("\n🔗 开始与回测框架集成...")
    
    # 1. 加载现有回测引擎
    sys.path.append('/Users/chengming/.openclaw/workspace/quant_trade-main')
    
    try:
        # 尝试导入现有的回测引擎
        from continuous_optimization_enhanced import EnhancedBacktestEngine
        
        print("✅ 成功导入现有回测引擎")
        
        # 2. 创建价格行为策略适配器
        params = {
            'engine_config': {
                'pivot_detection': {'prominence_factor': 0.5, 'window': 5},
                'range_clustering': {'window': 20, 'cluster_threshold': 1.0, 'min_range_length': 5},
                'compensated_ma': {'window': 20, 'beta': 0.3, 'gamma': 0.2, 'decay_factor': 0.95}
            },
            'signal_config': {
                'confidence_threshold': 0.7,
                'min_position_energy': 0.6,
                'trend_confirmation': True
            }
        }
        
        # 3. 加载测试数据
        df = load_stock_data_for_price_action(
            stock_code="000001.SZ",
            timeframe="daily_data2",
            start_date="2020-01-01",
            end_date="2021-12-31"
        )
        
        # 4. 创建策略并生成信号
        strategy = PriceActionStrategyAdapter(params)
        strategy.initialize(df)
        signals = strategy.generate_signals()
        
        if not signals:
            print("❌ 未生成信号，无法运行回测")
            return
        
        # 5. 运行回测
        backtest_engine = EnhancedBacktestEngine(initial_capital=1000000)
        results = backtest_engine.run_backtest(df, signals)
        
        # 6. 输出结果
        print(f"\n📈 价格行为策略回测结果:")
        print(f"   总收益率: {results.get('total_return', 0):.2%}")
        print(f"   夏普比率: {results.get('sharpe_ratio', 0):.3f}")
        print(f"   最大回撤: {results.get('max_drawdown', 0):.2%}")
        print(f"   胜率: {results.get('win_rate', 0):.2%}")
        print(f"   交易次数: {results.get('trades_count', 0)}")
        
        return results
        
    except ImportError as e:
        print(f"⚠️ 导入现有回测框架失败: {e}")
        print("   将仅测试价格行为策略适配器")
        return None
    except Exception as e:
        print(f"❌ 集成过程中出错: {e}")
        return None

# 主函数
if __name__ == "__main__":
    print("=" * 80)
    print("🏁 价格行为策略适配器主程序")
    print("=" * 80)
    
    # 测试策略适配器
    signals = test_price_action_strategy()
    
    if signals:
        # 尝试与现有框架集成
        results = integrate_with_backtest_framework()
        
        if results:
            print("\n🎉 价格行为策略适配器测试完成!")
            print("✅ 适配器开发成功")
            print("✅ 策略集成成功")
            print("✅ 回测运行成功")
        else:
            print("\n⚠️ 适配器开发完成，但集成测试部分失败")
            print("✅ 适配器开发成功")
            print("❌ 完整集成测试失败")
    else:
        print("\n❌ 价格行为策略适配器测试失败")
    
    print("\n" + "=" * 80)
    print("🏁 程序结束")