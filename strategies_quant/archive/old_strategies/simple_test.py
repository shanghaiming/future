#!/usr/bin/env python3
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy
print("简单测试开始")
import sys
print("Python路径:", sys.path)

try:
    import strategies.ma_strategy as ma_strategy
    print("✅ ma_strategy导入成功")
    
    # 检查类
    print("模块内容:", [x for x in dir(ma_strategy) if not x.startswith('_')])
    
    if hasattr(ma_strategy, 'MovingAverageStrategy'):
        print("✅ 找到MovingAverageStrategy类")
    else:
        print("❌ 未找到MovingAverageStrategy类")
        
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("测试结束")


class SimpleTestStrategy(BaseStrategy):
    """基于simple_test的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "SimpleTestStrategy"
        self.description = "基于simple_test的策略"
        
    def calculate_signals(self):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self):
        """简单测试策略 - 基于MA交叉"""
        import numpy as np
        df = self.data
        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()
        for i in range(20, len(df)):
            sym = df['symbol'].iloc[i] if 'symbol' in df.columns else 'DEFAULT'
            price = float(df['close'].iloc[i])
            if ma5.iloc[i] > ma20.iloc[i] and ma5.iloc[i-1] <= ma20.iloc[i-1]:
                self._record_signal(df.index[i], 'buy', sym, price)
            elif ma5.iloc[i] < ma20.iloc[i] and ma5.iloc[i-1] >= ma20.iloc[i-1]:
                self._record_signal(df.index[i], 'sell', sym, price)
        return self.signals
