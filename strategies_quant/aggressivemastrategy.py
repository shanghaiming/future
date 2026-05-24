#!/usr/bin/env python3
"""
优化版MA策略变体: 激进型MA策略
基于ma_strategy优化
"""

import pandas as pd
import numpy as np
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class AggressiveMAStrategy(BaseStrategy):
    """激进型MA策略 - 移动平均策略变体"""
    
    def __init__(self, data, params=None):
        # 默认参数
        default_params = {
            'short_window': 3,
            'long_window': 10,
            'signal_threshold': 0.005,
            'min_confidence': 0.3,
            'max_daily_signals': 10,
            'stop_loss_pct': 0.1,
            'take_profit_pct': 0.15,
            'risk_adjusted': True
        }
        
        # 合并参数
        if params:
            for key, value in params.items():
                if key in default_params:
                    default_params[key] = value
        
        super().__init__(data, default_params)
        
        # 存储参数
        self.short_window = default_params['short_window']
        self.long_window = default_params['long_window']
        self.signal_threshold = default_params['signal_threshold']
        self.min_confidence = default_params['min_confidence']
        self.max_daily_signals = default_params['max_daily_signals']
        self.stop_loss_pct = default_params['stop_loss_pct']
        self.take_profit_pct = default_params['take_profit_pct']
        self.risk_adjusted = default_params['risk_adjusted']
        
    def get_default_params(self):
        """返回默认参数"""
        return {
            'short_window': 3,
            'long_window': 10,
            'signal_threshold': 0.005,
            'min_confidence': 0.3,
            'max_daily_signals': 10,
            'stop_loss_pct': 0.1,
            'take_profit_pct': 0.15,
            'risk_adjusted': True
        }
    
    def generate_signals(self):
        """生成交易信号 - 基于MA交叉"""
        df = self.data
        short_w = self.params.get('short_window', 3)
        long_w = self.params.get('long_window', 10)
        
        short_ma = df['close'].rolling(short_w).mean()
        long_ma = df['close'].rolling(long_w).mean()
        
        for i in range(long_w, len(df)):
            sym = df['symbol'].iloc[i] if 'symbol' in df.columns else 'DEFAULT'
            if short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i-1] <= long_ma.iloc[i-1]:
                self._record_signal(df.index[i], 'buy', sym, float(df['close'].iloc[i]))
            elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i-1] >= long_ma.iloc[i-1]:
                self._record_signal(df.index[i], 'sell', sym, float(df['close'].iloc[i]))
        
        return self.signals
# 策略测试代码
if __name__ == "__main__":
    print("🧪 AggressiveMAStrategy 策略测试")
    print("📋 参数配置:")
    print("  短窗口: 3")
    print("  长窗口: 10")
    print("  信号阈值: 0.005")
    print("  最小置信度: 0.3")
    print("  每日最大信号: 10")
    print("  止损比例: 0.1")
    print("  止盈比例: 0.15")
