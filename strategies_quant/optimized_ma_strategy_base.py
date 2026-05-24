#!/usr/bin/env python3
"""
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy
优化版MA策略 - BaseStrategy子类
基于improved_ma_strategy_best_params.py的最佳参数，但继承自BaseStrategy
提供完整的移动平均交叉信号生成，集成了信号过滤和风险控制
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# 尝试导入BaseStrategy
try:
    from base_strategy import BaseStrategy
    BASE_STRATEGY_AVAILABLE = True
except ImportError:
    # 回退方案
    BASE_STRATEGY_AVAILABLE = False
    print("⚠️  base_strategy模块不可用，使用回退方案")

class OptimizedMAStrategy:
    """
    优化版MA策略（回退版本）
    如果BaseStrategy不可用，使用此版本
    """
    
    # 最佳参数配置（来自参数优化搜索）
    BEST_PARAMS = {
        'short_window': 5,           # 短期移动平均窗口
        'long_window': 30,           # 长期移动平均窗口
        'signal_threshold': 0.01,    # 信号阈值
        'min_confidence': 0.5,       # 最小置信度
        'max_daily_signals': 5,      # 每日最大信号数
        'stop_loss_pct': 0.08,       # 止损百分比
        'take_profit_pct': 0.08,     # 止盈百分比
        'risk_adjusted': True,       # 是否风险调整
        'use_volume_filter': True,   # 是否使用成交量过滤
        'symbol': '000001.SZ'        # 默认股票代码
    }
    
    def __init__(self, data=None, **kwargs):
        """
        初始化优化版策略
        Args:
            data: 价格数据 (DataFrame)
            **kwargs: 策略参数，可覆盖默认最佳参数
        """
        self.data = data
        self.params = self.BEST_PARAMS.copy()
        
        # 使用传入参数覆盖默认参数
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
        
        # 初始化状态
        self.signals = []
        self.performance_metrics = {}
        self.raw_signals_count = 0
    
    # ... 其他方法保持不变 ...

if BASE_STRATEGY_AVAILABLE:
    # 创建BaseStrategy子类
    class OptimizedMAStrategy(BaseStrategy):
        """优化版MA策略 - BaseStrategy子类"""
        
        def get_default_params(self) -> Dict[str, Any]:
            """返回默认参数"""
            return {
                'short_window': 5,           # 短期移动平均窗口
                'long_window': 30,           # 长期移动平均窗口
                'signal_threshold': 0.01,    # 信号阈值
                'min_confidence': 0.5,       # 最小置信度
                'max_daily_signals': 5,      # 每日最大信号数
                'stop_loss_pct': 0.08,       # 止损百分比
                'take_profit_pct': 0.08,     # 止盈百分比
                'risk_adjusted': True,       # 是否风险调整
                'use_volume_filter': True,   # 是否使用成交量过滤
                'symbol': '000001.SZ'        # 默认股票代码
            }
        
        def __init__(self, data, params=None):
            """初始化策略"""
            # 调用父类初始化
            super().__init__(data, params)
            
            # 确保params存在
            if self.params is None:
                self.params = self.get_default_params()
            
            # 从params获取参数
            self.short_window = self.params.get('short_window', 5)
            self.long_window = self.params.get('long_window', 30)
            self.signal_threshold = self.params.get('signal_threshold', 0.01)
            self.min_confidence = self.params.get('min_confidence', 0.5)
            self.max_daily_signals = self.params.get('max_daily_signals', 5)
            self.stop_loss_pct = self.params.get('stop_loss_pct', 0.08)
            self.take_profit_pct = self.params.get('take_profit_pct', 0.08)
            self.risk_adjusted = self.params.get('risk_adjusted', True)
            self.use_volume_filter = self.params.get('use_volume_filter', True)
            self.symbol = self.params.get('symbol', '000001.SZ')
            
            # 信号过滤状态
            self.daily_signal_counts = {}
        
        def calculate_moving_averages(self) -> Dict[str, np.ndarray]:
            """计算移动平均线"""
            if self.data is None or len(self.data) == 0:
                return {}
            
            close_prices = self.data['close'].values
            
            if len(close_prices) < max(self.short_window, self.long_window):
                return {
                    'short_ma': np.array([]),
                    'long_ma': np.array([]),
                    'ma_diff': np.array([])
                }
            
            # 使用pandas计算移动平均
            close_series = pd.Series(close_prices)
            short_ma = close_series.rolling(window=self.short_window, min_periods=1).mean().values
            long_ma = close_series.rolling(window=self.long_window, min_periods=1).mean().values
            
            # 计算差值
            ma_diff = short_ma - long_ma
            
            return {
                'short_ma': short_ma,
                'long_ma': long_ma,
                'ma_diff': ma_diff
            }
        
        def generate_signals(self) -> List[Dict]:
            """生成交易信号"""
            # 清空信号列表
            self.signals = []
            
            if self.data is None or len(self.data) == 0:
                return self.signals
            
            # 计算移动平均
            ma_results = self.calculate_moving_averages()
            if ma_results['short_ma'].size == 0:
                return self.signals
            
            short_ma = ma_results['short_ma']
            long_ma = ma_results['long_ma']
            ma_diff = ma_results['ma_diff']
            
            # 获取时间索引
            time_index = self.data.index
            
            # 重置每日信号计数
            self.daily_signal_counts = {}
            
            # 生成原始信号
            raw_signals = []
            for i in range(1, len(ma_diff)):
                prev_diff = ma_diff[i-1]
                curr_diff = ma_diff[i]
                
                # 金叉信号（短期MA上穿长期MA）
                if prev_diff <= 0 and curr_diff > 0 and abs(curr_diff) > self.signal_threshold:
                    signal_type = 'buy'
                    confidence = min(0.5 + abs(curr_diff) * 10, 1.0)
                    raw_signals.append({
                        'index': i,
                        'timestamp': time_index[i],
                        'action': signal_type,
                        'price': self.data.iloc[i]['close'] if 'close' in self.data.columns else self.data.iloc[i, 0],
                        'confidence': confidence,
                        'reason': f'MA金叉: 短期MA({self.short_window})上穿长期MA({self.long_window}), 差值: {curr_diff:.4f}'
                    })
                
                # 死叉信号（短期MA下穿长期MA）
                elif prev_diff >= 0 and curr_diff < 0 and abs(curr_diff) > self.signal_threshold:
                    signal_type = 'sell'
                    confidence = min(0.5 + abs(curr_diff) * 10, 1.0)
                    raw_signals.append({
                        'index': i,
                        'timestamp': time_index[i],
                        'action': signal_type,
                        'price': self.data.iloc[i]['close'] if 'close' in self.data.columns else self.data.iloc[i, 0],
                        'confidence': confidence,
                        'reason': f'MA死叉: 短期MA({self.short_window})下穿长期MA({self.long_window}), 差值: {curr_diff:.4f}'
                    })
            
            # 应用信号过滤
            filtered_signals = self.filter_signals(raw_signals)
            
            # 转换为BaseStrategy格式并记录
            for signal in filtered_signals:
                self._record_signal(
                    timestamp=signal['timestamp'],
                    action=signal['action'],
                    symbol=self.symbol,
                    price=signal['price'],
                    confidence=signal.get('confidence', 0.5),
                    reason=signal.get('reason', '')
                )
            
            print(f"📊 信号生成完成: 原始{len(raw_signals)}个 → 过滤后{len(self.signals)}个")
            return self.signals
        
        def filter_signals(self, raw_signals: List[Dict]) -> List[Dict]:
            """过滤信号"""
            if not raw_signals:
                return []
            
            # 1. 按置信度过滤
            conf_filtered = [s for s in raw_signals if s.get('confidence', 0) >= self.min_confidence]
            
            # 2. 按每日频率过滤
            freq_filtered = []
            for signal in conf_filtered:
                date_key = signal['timestamp'].date() if hasattr(signal['timestamp'], 'date') else signal['timestamp']
                
                if date_key not in self.daily_signal_counts:
                    self.daily_signal_counts[date_key] = 0
                
                if self.daily_signal_counts[date_key] < self.max_daily_signals:
                    freq_filtered.append(signal)
                    self.daily_signal_counts[date_key] += 1
            
            # 3. 添加风险控制信息
            for signal in freq_filtered:
                current_price = signal['price']
                signal['stop_loss'] = current_price * (1 - self.stop_loss_pct)
                signal['take_profit'] = current_price * (1 + self.take_profit_pct)
            
            return freq_filtered
        
        def get_performance_summary(self) -> Dict[str, Any]:
            """获取性能摘要"""
            if not self.signals:
                return {'message': '没有信号生成'}
            
            num_buy = sum(1 for s in self.signals if s.get('action') == 'buy')
            num_sell = sum(1 for s in self.signals if s.get('action') == 'sell')
            num_hold = sum(1 for s in self.signals if s.get('action') == 'hold')
            
            avg_confidence = np.mean([s.get('confidence', 0.5) for s in self.signals]) if self.signals else 0
            
            return {
                'total_signals': len(self.signals),
                'buy_signals': num_buy,
                'sell_signals': num_sell,
                'hold_signals': num_hold,
                'buy_sell_ratio': num_buy / max(num_sell, 1),
                'average_confidence': avg_confidence,
                'parameters': {
                    'short_window': self.short_window,
                    'long_window': self.long_window,
                    'signal_threshold': self.signal_threshold,
                    'min_confidence': self.min_confidence,
                    'max_daily_signals': self.max_daily_signals,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct
                }
            }

# 测试代码
if __name__ == "__main__":
    print("🧪 优化版MA策略测试")
    
    # 创建示例数据
    dates = pd.date_range('2021-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    if BASE_STRATEGY_AVAILABLE:
        print("✅ BaseStrategy可用，使用子类版本")
        strategy = OptimizedMAStrategy(data)
        signals = strategy.generate_signals()
        
        print(f"📊 生成信号数量: {len(signals)}")
        if signals:
            print("📋 前5个信号:")
            for i, sig in enumerate(signals[:5]):
                print(f"  {i+1}. {sig['timestamp']} - {sig['action']} - 价格: {sig['price']:.2f}")
        
        # 获取性能摘要
        perf = strategy.get_performance_summary()
        print(f"\n📈 性能摘要:")
        for key, value in perf.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    else:
        print("⚠️  BaseStrategy不可用，使用回退版本")
        strategy = OptimizedMAStrategy(data)
        print("✅ 回退版本创建成功")