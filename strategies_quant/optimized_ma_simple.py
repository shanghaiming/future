#!/usr/bin/env python3
"""
优化版MA策略 - 简单版本
继承自ma_strategy.MovingAverageStrategy，但使用优化参数
"""

from core.base_strategy import BaseStrategy
from strategies.ma_strategy import MovingAverageStrategy

class OptimizedMASimpleStrategy(MovingAverageStrategy, BaseStrategy):
    """优化版MA策略 - 简单版本"""

    def get_default_params(self):
        """返回优化参数"""
        return {
            'short_window': 5,
            'long_window': 30,
            'signal_threshold': 0.01,
            'min_confidence': 0.5,
            'max_daily_signals': 5,
            'stop_loss_pct': 0.08,
            'take_profit_pct': 0.08,
            'risk_adjusted': True,
            'symbol': '000001.SZ'
        }

    def __init__(self, data, params=None):
        """初始化策略"""
        if params is None:
            params = self.get_default_params()
        else:
            default_params = self.get_default_params()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value

        super().__init__(data, params)

        self.short_window = params['short_window']
        self.long_window = params['long_window']
        self.signal_threshold = params.get('signal_threshold', 0.01)
        self.min_confidence = params.get('min_confidence', 0.5)
        self.max_daily_signals = params.get('max_daily_signals', 5)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.08)
        self.take_profit_pct = params.get('take_profit_pct', 0.08)
        self.risk_adjusted = params.get('risk_adjusted', True)

        print(f"Optimized MA strategy init: short={self.short_window}, long={self.long_window}, threshold={self.signal_threshold}")

# 测试代码
if __name__ == "__main__":
    print("Optimized MA Strategy Simple Test")

    import pandas as pd
    import numpy as np

    dates = pd.date_range('2021-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    strategy = OptimizedMASimpleStrategy(data)
    signals = strategy.generate_signals()

    print(f"Signals generated: {len(signals)}")
    if signals:
        print("First 5 signals:")
        for i, sig in enumerate(signals[:5]):
            print(f"  {i+1}. {sig.get('timestamp', 'N/A')} - {sig.get('action', 'N/A')}")
