#!/usr/bin/env python3
"""
测试单个策略的信号生成
"""

try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

import sys
import os
import pandas as pd
import numpy as np

def test_strategy_signals(file_name):
    """测试策略信号生成"""
    print(f"\n🧪 测试策略: {file_name}")
    
    try:
        # 动态导入
        module_name = file_name.replace('.py', '')
        
        # 添加当前目录到路径
        sys.path.append('.')
        
        # 导入模块
        module = __import__(module_name)
        
        print(f"✅ 导入成功")
        
        # 查找策略类
        strategy_classes = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith('Strategy') and attr_name != 'BaseStrategy':
                strategy_classes.append(attr_name)
        
        if not strategy_classes:
            print("❌ 未找到策略类")
            return 0, False
        
        print(f"   找到策略类: {strategy_classes}")
        
        # 使用第一个策略类
        strategy_class_name = strategy_classes[0]
        StrategyClass = getattr(module, strategy_class_name)
        
        # 加载测试数据
        data_dir = "/Users/chengming/.openclaw/workspace/quant_trade-main/data/daily_data2"
        data_file = os.path.join(data_dir, "000001.SZ.csv")
        
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            return 0, False
        
        # 加载数据
        data = pd.read_csv(data_file)
        if 'trade_date' in data.columns:
            data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
            data.set_index('trade_date', inplace=True)
        
        # 使用2021-2024年数据
        data = data.loc['2021-01-01':'2024-12-31']
        
        if len(data) == 0:
            print("❌ 数据为空")
            return 0, False
        
        print(f"   数据加载: {len(data)} 行, {data.index.min()} 到 {data.index.max()}")
        
        # 策略参数
        params = {
            'symbol': '000001.SZ',
            'ma_window': 20,
            'rsi_period': 14,
            'risk_per_trade': 0.02
        }
        
        # 实例化策略
        strategy = StrategyClass(data, params)
        print(f"✅ 策略实例化成功: {strategy_class_name}")
        
        # 生成信号
        signals = strategy.generate_signals()
        
        if signals is None:
            print("❌ generate_signals 返回 None")
            return 0, False
        
        # 统计信号数量
        if isinstance(signals, list):
            signal_count = len(signals)
        elif isinstance(signals, pd.DataFrame):
            signal_count = len(signals)
        else:
            # 尝试获取长度
            try:
                signal_count = len(signals)
            except:
                print(f"❌ 无法确定信号数量，类型: {type(signals)}")
                return 0, False
        
        print(f"✅ 生成 {signal_count} 个信号")
        
        # 显示前3个信号示例
        if signal_count > 0:
            if isinstance(signals, list) and signals:
                for i, signal in enumerate(signals[:3]):
                    print(f"   信号 {i+1}: {signal}")
            elif isinstance(signals, pd.DataFrame) and not signals.empty:
                print(f"   前3行:\n{signals.head(3)}")
        
        return signal_count, True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, False

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 策略信号测试")
    print("=" * 60)
    
    # 测试几个关键策略
    test_files = [
        'ma_strategy.py',
        'tradingview_strategy.py',
        'market_structure_identifier.py',
        'exit_strategy_optimizer.py'
    ]
    
    total_signals = 0
    successful_tests = 0
    
    for file_name in test_files:
        if not os.path.exists(file_name):
            print(f"❌ 文件不存在: {file_name}")
            continue
        
        signals, success = test_strategy_signals(file_name)
        
        if success:
            successful_tests += 1
            total_signals += signals
    
    print("\n" + "=" * 60)
    print("📊 测试摘要")
    print("=" * 60)
    print(f"测试策略数: {len(test_files)}")
    print(f"成功测试: {successful_tests}")
    print(f"失败测试: {len(test_files) - successful_tests}")
    print(f"总生成信号: {total_signals}")
    print(f"平均信号/策略: {total_signals/max(successful_tests, 1):.1f}")

if __name__ == "__main__":
    main()


class TestSingleStrategySignalsStrategy(BaseStrategy):
    """基于test_single_strategy_signals的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "TestSingleStrategySignalsStrategy"
        self.description = "基于test_single_strategy_signals的策略"
        
    def calculate_signals(self):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self):
        """生成交易信号"""
        # 信号生成逻辑
        return self.signals
