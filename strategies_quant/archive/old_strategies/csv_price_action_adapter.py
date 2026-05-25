"""
价格行为分析策略适配器
"""
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy
import pandas as pd
from typing import Dict, List
import numpy as np

class CSVPriceActionAdapter(BaseStrategy):
    """CSV价格行为策略适配器"""
    
    def generate_signals(self) -> List[Dict]:
        """生成价格行为信号"""
        signals = []
        
        try:
            # 计算简单移动平均
            window = self.params.get('ma_window', 20)
            self.data['ma'] = self.data['close'].rolling(window=window).mean()
            
            for i in range(window, len(self.data)):
                current_close = self.data['close'].iloc[i]
                current_ma = self.data['ma'].iloc[i]
                prev_close = self.data['close'].iloc[i-1]
                prev_ma = self.data['ma'].iloc[i-1]
                
                # 价格上穿均线 - 买入信号
                if prev_close <= prev_ma and current_close > current_ma:
                    signals.append({
                        'timestamp': self.data.index[i],
                        'action': 'buy',
                        'price': current_close,
                        'reason': '价格上穿移动平均线'
                    })
                
                # 价格下穿均线 - 卖出信号
                elif prev_close >= prev_ma and current_close < current_ma:
                    signals.append({
                        'timestamp': self.data.index[i],
                        'action': 'sell',
                        'price': current_close,
                        'reason': '价格下穿移动平均线'
                    })
            
            self.signals = signals
            return signals
            
        except Exception as e:
            print(f"价格行为策略执行错误: {e}")
            return []
