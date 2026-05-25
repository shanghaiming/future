# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.640150

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势通道分析量化系统
第16章：趋势通道分析
AL Brooks《价格行为交易之区间篇》

核心概念：
1. 趋势通道识别：上升通道、下降通道、水平通道
2. 通道线绘制：基于极值点的线性回归
3. 突破信号检测：价格突破通道边界
4. 通道内交易策略：下轨买入，上轨卖出
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class TrendChannelAnalyzer:
    """趋势通道分析器"""
    
    def __init__(self, 
                 lookback_period: int = 50,
                 channel_deviation: float = 2.0):
        """
        初始化趋势通道分析器
        
        参数:
            lookback_period: 回溯周期
            channel_deviation: 通道偏差倍数（用于平行通道）
        """
        self.lookback_period = lookback_period
        self.channel_deviation = channel_deviation
        self.channels_history = []
        
    def identify_trend_channels(self, 
                               price_data: pd.DataFrame) -> Dict:
        """
        识别价格数据中的趋势通道
        
        参数:
            price_data: 价格数据，需包含'high', 'low', 'close'列
            
        返回:
            通道分析结果
        """
        if len(price_data) < self.lookback_period:
            return {'error': f'数据不足，至少需要{self.lookback_period}个数据点'}
        
        # 使用最近的数据
        recent_data = price_data.tail(self.lookback_period).copy()
        
        # 识别通道类型
        channel_type = self._determine_channel_type(recent_data)
        
        # 计算通道边界
        if channel_type == 'uptrend':
            channels = self._calculate_uptrend_channel(recent_data)
        elif channel_type == 'downtrend':
            channels = self._calculate_downtrend_channel(recent_data)
        else:
            channels = self._calculate_range_channel(recent_data)
        
        # 检测突破信号
        breakouts = self._detect_breakouts(recent_data, channels)
        
        # 生成交易信号
        signals = self._generate_trading_signals(recent_data, channels, breakouts)
        
        result = {
            'channel_type': channel_type,
            'channels': channels,
            'breakouts': breakouts,
            'signals': signals,
            'current_price': recent_data['close'].iloc[-1],
            'analysis_timestamp': pd.Timestamp.now(),
            'data_points_analyzed': len(recent_data)
        }
        
        # 保存到历史
        self.channels_history.append(result)
        
        return result
    
    def _determine_channel_type(self, data: pd.DataFrame) -> str:
        """
        确定通道类型
        
        返回:
            'uptrend' (上升趋势), 'downtrend' (下降趋势), 'range' (区间)
        """
        # 计算简单移动平均线斜率
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        
        # 移除NaN值
        sma_20_valid = sma_20.dropna()
        sma_50_valid = sma_50.dropna()
        
        if len(sma_20_valid) < 2 or len(sma_50_valid) < 2:
            return 'range'
        
        # 计算斜率（最近5个值的变化）
        sma_20_slope = self._calculate_slope(sma_20_valid.tail(5).values)
        sma_50_slope = self._calculate_slope(sma_50_valid.tail(5).values)
        
        # 判断趋势
        if sma_20_slope > 0.001 and sma_50_slope > 0.0005:
            return 'uptrend'
        elif sma_20_slope < -0.001 and sma_50_slope < -0.0005:
            return 'downtrend'
        else:
            return 'range'
    
    def _calculate_slope(self, values: np.ndarray) -> float:
        """计算线性回归斜率"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def _calculate_uptrend_channel(self, data: pd.DataFrame) -> Dict:
        """
        计算上升通道
        
        上升通道：下轨连接低点，上轨平行于下轨
        """
        # 识别低点（支撑线）
        lows = data['low'].values
        low_indices = self._find_extreme_points(lows, 'low')
        
        # 使用线性回归拟合支撑线
        if len(low_indices) >= 2:
            x_low = low_indices
            y_low = lows[low_indices]
            support_slope, support_intercept = np.polyfit(x_low, y_low, 1)
        else:
            # 如果没有足够低点，使用简单线性回归
            x_all = np.arange(len(lows))
            support_slope, support_intercept = np.polyfit(x_all, lows, 1)
        
        # 计算阻力线（平行于支撑线，通过高点）
        highs = data['high'].values
        high_indices = self._find_extreme_points(highs, 'high')
        
        if len(high_indices) >= 1:
            # 找到距离支撑线最远的高点
            distances = []
            for idx in high_indices:
                support_value = support_slope * idx + support_intercept
                distance = highs[idx] - support_value
                distances.append(distance)
            
            if distances:
                max_distance_idx = high_indices[np.argmax(distances)]
                resistance_intercept = highs[max_distance_idx] - support_slope * max_distance_idx
            else:
                # 默认距离为ATR的2倍
                atr = self._calculate_atr(data)
                resistance_intercept = support_intercept + atr * self.channel_deviation
        else:
            # 默认距离为ATR的2倍
            atr = self._calculate_atr(data)
            resistance_intercept = support_intercept + atr * self.channel_deviation
        
        # 计算通道边界
        x_points = np.arange(len(data))
        support_line = support_slope * x_points + support_intercept
        resistance_line = support_slope * x_points + resistance_intercept
        
        # 计算通道宽度
        channel_width = resistance_intercept - support_intercept
        
        return {
            'type': 'uptrend',
            'support_slope': support_slope,
            'support_intercept': support_intercept,
            'resistance_slope': support_slope,  # 相同斜率
            'resistance_intercept': resistance_intercept,
            'support_line': support_line,
            'resistance_line': resistance_line,
            'channel_width': channel_width,
            'channel_angle': np.degrees(np.arctan(support_slope))
        }
    
    def _calculate_downtrend_channel(self, data: pd.DataFrame) -> Dict:
        """
        计算下降通道
        
        下降通道：上轨连接高点，下轨平行于上轨
        """
        # 识别高点（阻力线）
        highs = data['high'].values
        high_indices = self._find_extreme_points(highs, 'high')
        
        # 使用线性回归拟合阻力线
        if len(high_indices) >= 2:
            x_high = high_indices
            y_high = highs[high_indices]
            resistance_slope, resistance_intercept = np.polyfit(x_high, y_high, 1)
        else:
            # 如果没有足够高点，使用简单线性回归
            x_all = np.arange(len(highs))
            resistance_slope, resistance_intercept = np.polyfit(x_all, highs, 1)
        
        # 计算支撑线（平行于阻力线，通过低点）
        lows = data['low'].values
        low_indices = self._find_extreme_points(lows, 'low')
        
        if len(low_indices) >= 1:
            # 找到距离阻力线最远的低点
            distances = []
            for idx in low_indices:
                resistance_value = resistance_slope * idx + resistance_intercept
                distance = resistance_value - lows[idx]
                distances.append(distance)
            
            if distances:
                max_distance_idx = low_indices[np.argmax(distances)]
                support_intercept = lows[max_distance_idx] - resistance_slope * max_distance_idx
            else:
                # 默认距离为ATR的2倍
                atr = self._calculate_atr(data)
                support_intercept = resistance_intercept - atr * self.channel_deviation
        else:
            # 默认距离为ATR的2倍
            atr = self._calculate_atr(data)
            support_intercept = resistance_intercept - atr * self.channel_deviation
        
        # 计算通道边界
        x_points = np.arange(len(data))
        resistance_line = resistance_slope * x_points + resistance_intercept
        support_line = resistance_slope * x_points + support_intercept
        
        # 计算通道宽度
        channel_width = resistance_intercept - support_intercept
        
        return {
            'type': 'downtrend',
            'resistance_slope': resistance_slope,
            'resistance_intercept': resistance_intercept,
            'support_slope': resistance_slope,  # 相同斜率
            'support_intercept': support_intercept,
            'resistance_line': resistance_line,
            'support_line': support_line,
            'channel_width': channel_width,
            'channel_angle': np.degrees(np.arctan(resistance_slope))
        }
    
    def _calculate_range_channel(self, data: pd.DataFrame) -> Dict:
        """
        计算区间通道（水平通道）
        """
        # 识别支撑和阻力水平
        highs = data['high'].values
        lows = data['low'].values
        
        # 使用近期高点和低点的分位数
        resistance_level = np.percentile(highs[-20:], 70)  # 70%分位数作为阻力
        support_level = np.percentile(lows[-20:], 30)      # 30%分位数作为支撑
        
        # 如果水平太接近，使用ATR调整
        atr = self._calculate_atr(data)
        if resistance_level - support_level < atr * 0.5:
            resistance_level = np.mean(highs[-20:]) + atr * 0.5
            support_level = np.mean(lows[-20:]) - atr * 0.5
        
        # 创建水平线
        x_points = np.arange(len(data))
        resistance_line = np.full_like(x_points, resistance_level, dtype=float)
        support_line = np.full_like(x_points, support_level, dtype=float)
        
        channel_width = resistance_level - support_level
        
        return {
            'type': 'range',
            'resistance_level': resistance_level,
            'support_level': support_level,
            'resistance_line': resistance_line,
            'support_line': support_line,
            'channel_width': channel_width,
            'channel_angle': 0  # 水平通道角度为0
        }
    
    def _find_extreme_points(self, values: np.ndarray, extreme_type: str) -> np.ndarray:
        """
        寻找极值点（高点或低点）
        
        参数:
            values: 价格序列
            extreme_type: 'high' 或 'low'
            
        返回:
            极值点索引数组
        """
        n = len(values)
        if n < 5:
            return np.array([])
        
        indices = []
        
        for i in range(2, n - 2):
            if extreme_type == 'high':
                # 局部高点：中间值比前后2个值都高
                if (values[i] > values[i-2] and values[i] > values[i-1] and
                    values[i] > values[i+1] and values[i] > values[i+2]):
                    indices.append(i)
            else:  # 'low'
                # 局部低点：中间值比前后2个值都低
                if (values[i] < values[i-2] and values[i] < values[i-1] and
                    values[i] < values[i+1] and values[i] < values[i+2]):
                    indices.append(i)
        
        return np.array(indices)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """计算平均真实波幅 (ATR)"""
        if len(data) < period + 1:
            return 0.01  # 默认值
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # 计算真实波幅
        tr = np.zeros(len(data))
        for i in range(1, len(data)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        # 计算ATR
        atr = np.mean(tr[-period:])
        return atr
    
    def _detect_breakouts(self, 
                         data: pd.DataFrame, 
                         channels: Dict) -> Dict:
        """
        检测通道突破信号
        
        返回:
            突破分析结果
        """
        current_price = data['close'].iloc[-1]
        support_line = channels['support_line'][-1]
        resistance_line = channels['resistance_line'][-1]
        
        # 计算与边界的距离
        distance_to_support = current_price - support_line
        distance_to_resistance = resistance_line - current_price
        
        # 突破阈值（ATR的百分比）
        atr = self._calculate_atr(data)
        breakout_threshold = atr * 0.3  # 30%的ATR作为突破阈值
        
        breakouts = {
            'bullish_breakout': False,
            'bearish_breakout': False,
            'false_breakout': False,
            'breakout_strength': 0,
            'distance_to_support': distance_to_support,
            'distance_to_resistance': distance_to_resistance,
            'breakout_threshold': breakout_threshold
        }
        
        # 检测向上突破
        if current_price > resistance_line + breakout_threshold:
            breakouts['bullish_breakout'] = True
            breakouts['breakout_strength'] = (current_price - resistance_line) / atr
            
            # 检查是否为假突破（快速返回通道内）
            if len(data) >= 3:
                prev_price = data['close'].iloc[-2]
                if prev_price > resistance_line and current_price < resistance_line:
                    breakouts['false_breakout'] = True
        
        # 检测向下突破
        elif current_price < support_line - breakout_threshold:
            breakouts['bearish_breakout'] = True
            breakouts['breakout_strength'] = (support_line - current_price) / atr
            
            # 检查是否为假突破
            if len(data) >= 3:
                prev_price = data['close'].iloc[-2]
                if prev_price < support_line and current_price > support_line:
                    breakouts['false_breakout'] = True
        
        return breakouts
    
    def _generate_trading_signals(self,
                                 data: pd.DataFrame,
                                 channels: Dict,
                                 breakouts: Dict) -> List[Dict]:
        """
        生成交易信号
        
        返回:
            交易信号列表
        """
        signals = []
        current_price = data['close'].iloc[-1]
        channel_type = channels['type']
        
        # 通道内交易信号
        if not breakouts['bullish_breakout'] and not breakouts['bearish_breakout']:
            # 价格在通道内
            
            # 计算价格在通道中的位置（0=下轨，1=上轨）
            support = channels['support_line'][-1]
            resistance = channels['resistance_line'][-1]
            
            if resistance != support:
                channel_position = (current_price - support) / (resistance - support)
            else:
                channel_position = 0.5
            
            # 生成信号
            if channel_position < 0.3:
                # 靠近下轨，买入信号
                signals.append({
                    'type': 'buy',
                    'reason': f'价格接近{channel_type}通道下轨',
                    'entry_price': current_price,
                    'stop_loss': support * 0.995,
                    'take_profit': resistance * 0.985,
                    'confidence': max(0.7, 1 - channel_position),
                    'position_size': 'normal'
                })
            elif channel_position > 0.7:
                # 靠近上轨，卖出信号
                signals.append({
                    'type': 'sell',
                    'reason': f'价格接近{channel_type}通道上轨',
                    'entry_price': current_price,
                    'stop_loss': resistance * 1.005,
                    'take_profit': support * 1.015,
                    'confidence': max(0.7, channel_position),
                    'position_size': 'normal'
                })
        
        # 突破交易信号
        else:
            if breakouts['bullish_breakout'] and not breakouts['false_breakout']:
                # 向上突破，买入信号
                signals.append({
                    'type': 'buy',
                    'reason': f'{channel_type}通道向上突破，强度{breakouts["breakout_strength"]:.1f}ATR',
                    'entry_price': current_price,
                    'stop_loss': channels['resistance_line'][-1] * 0.995,
                    'take_profit': current_price * (1 + breakouts['breakout_strength'] * 0.005),
                    'confidence': min(0.9, 0.6 + breakouts['breakout_strength'] * 0.1),
                    'position_size': 'aggressive' if breakouts['breakout_strength'] > 1 else 'normal'
                })
            
            elif breakouts['bearish_breakout'] and not breakouts['false_breakout']:
                # 向下突破，卖出信号
                signals.append({
                    'type': 'sell',
                    'reason': f'{channel_type}通道向下突破，强度{breakouts["breakout_strength"]:.1f}ATR',
                    'entry_price': current_price,
                    'stop_loss': channels['support_line'][-1] * 1.005,
                    'take_profit': current_price * (1 - breakouts['breakout_strength'] * 0.005),
                    'confidence': min(0.9, 0.6 + breakouts['breakout_strength'] * 0.1),
                    'position_size': 'aggressive' if breakouts['breakout_strength'] > 1 else 'normal'
                })
        
        return signals
    
    def analyze_multiple_timeframes(self,
                                   price_data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        多时间框架通道分析
        
        参数:
            price_data_dict: {timeframe: price_data}
            
        返回:
            多时间框架分析结果
        """
        results = {}
        
        for timeframe, data in price_data_dict.items():
            if len(data) >= self.lookback_period:
                channel_result = self.identify_trend_channels(data)
                results[timeframe] = channel_result
        
        # 分析时间框架一致性
        consistent_trend = self._analyze_timeframe_consistency(results)
        
        return {
            'timeframe_results': results,
            'consistent_trend': consistent_trend,
            'recommended_action': self._generate_multi_tf_recommendation(results)
        }
    
    def _analyze_timeframe_consistency(self, results: Dict) -> str:
        """分析多时间框架趋势一致性"""
        if not results:
            return 'unknown'
        
        trends = []
        for tf, result in results.items():
            if 'channel_type' in result:
                trends.append(result['channel_type'])
        
        # 统计趋势类型
        from collections import Counter
        trend_counts = Counter(trends)
        
        if len(trend_counts) == 1:
            return list(trend_counts.keys())[0]  # 完全一致
        elif 'uptrend' in trend_counts and trend_counts['uptrend'] >= len(trends) * 0.7:
            return 'mostly_uptrend'
        elif 'downtrend' in trend_counts and trend_counts['downtrend'] >= len(trends) * 0.7:
            return 'mostly_downtrend'
        else:
            return 'mixed'
    
    def _generate_multi_tf_recommendation(self, results: Dict) -> str:
        """生成多时间框架交易建议"""
        consistent_trend = self._analyze_timeframe_consistency(results)
        
        if consistent_trend == 'uptrend':
            return '各时间框架均显示上升趋势，建议寻找回调买入机会'
        elif consistent_trend == 'downtrend':
            return '各时间框架均显示下降趋势，建议寻找反弹卖出机会'
        elif consistent_trend == 'mostly_uptrend':
            return '多数时间框架显示上升趋势，可考虑轻仓买入'
        elif consistent_trend == 'mostly_downtrend':
            return '多数时间框架显示下降趋势，可考虑轻仓卖出'
        elif consistent_trend == 'range':
            return '各时间框架显示区间震荡，建议高抛低吸'
        else:
            return '趋势不一致，建议观望或降低仓位'

def generate_sample_price_data(n_bars: int = 200) -> pd.DataFrame:
    """生成样本价格数据"""
    np.random.seed(42)
    
    # 生成基础趋势
    time = np.arange(n_bars)
    
    # 创建上升趋势通道
    base_trend = 100 + time * 0.1  # 基础上升趋势
    
    # 添加通道波动
    channel_height = 5
    oscillation = np.sin(time * 0.1) * channel_height
    
    # 生成价格
    base_price = base_trend + oscillation
    noise = np.random.normal(0, 0.5, n_bars)
    
    prices = base_price + noise
    
    # 生成OHLC数据
    df = pd.DataFrame({
        'open': prices * 0.998,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='D'))
    
    return df


# ============================================================================
# 策略改造: 添加TrendChannelAnalyzerStrategy类
# 将趋势通道分析器转换为交易策略
# ============================================================================

class TrendChannelAnalyzerStrategy(BaseStrategy):
    """趋势通道分析策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        lookback_period = params.get('lookback_period', 50)
        channel_deviation = params.get('channel_deviation', 2.0)
        
        # 创建趋势通道分析器实例
        self.analyzer = TrendChannelAnalyzer(
            lookback_period=lookback_period,
            channel_deviation=channel_deviation
        )
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于趋势通道分析生成交易信号
        """
        # 调用趋势通道分析
        analysis_result = self.analyzer.analyze_trend_channels(self.data)
        
        # 获取通道信号
        channel_signals = analysis_result.get('channel_signals', [])
        
        # 将通道信号转换为交易信号
        for signal in channel_signals:
            action = signal.get('type', 'hold').lower()
            if action in ['buy', 'sell']:
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action=action,
                    price=signal.get('price', self.data['close'].iloc[-1])
                )
        
        # 如果没有信号，添加hold信号
        if not self.signals:
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        
        return self.signals


# ============================================================================
# 策略改造完成
# ============================================================================

def main():
    """主函数：演示趋势通道分析系统"""
    print("=== 趋势通道分析量化系统 ===")
    print("第16章：趋势通道分析\n")
    
    # 生成样本数据
    print("1. 生成样本价格数据...")
    price_data = generate_sample_price_data(200)
    print(f"   数据形状: {price_data.shape}")
    print(f"   时间范围: {price_data.index[0].date()} 到 {price_data.index[-1].date()}")
    print(f"   价格范围: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
    
    # 创建分析器
    print("\n2. 创建趋势通道分析器...")
    analyzer = TrendChannelAnalyzer(lookback_period=100, channel_deviation=2.0)
    
    # 分析趋势通道
    print("\n3. 分析趋势通道...")
    analysis_result = analyzer.identify_trend_channels(price_data)
    
    print(f"   通道类型: {analysis_result['channel_type']}")
    print(f"   通道宽度: {analysis_result['channels']['channel_width']:.2f}")
    print(f"   通道角度: {analysis_result['channels']['channel_angle']:.1f}°")
    
    current_price = analysis_result['current_price']
    support = analysis_result['channels']['support_line'][-1]
    resistance = analysis_result['channels']['resistance_line'][-1]
    
    print(f"   当前价格: ${current_price:.2f}")
    print(f"   通道下轨: ${support:.2f}")
    print(f"   通道上轨: ${resistance:.2f}")
    
    # 突破分析
    print("\n4. 突破信号分析...")
    breakouts = analysis_result['breakouts']
    
    if breakouts['bullish_breakout']:
        print(f"   🔼 检测到向上突破!")
        print(f"   突破强度: {breakouts['breakout_strength']:.1f} ATR")
        if breakouts['false_breakout']:
            print(f"   ⚠️  疑似假突破")
    elif breakouts['bearish_breakout']:
        print(f"   🔽 检测到向下突破!")
        print(f"   突破强度: {breakouts['breakout_strength']:.1f} ATR")
        if breakouts['false_breakout']:
            print(f"   ⚠️  疑似假突破")
    else:
        print(f"   ➖ 价格在通道内")
        print(f"   距下轨: {breakouts['distance_to_support']:.2f}")
        print(f"   距上轨: {breakouts['distance_to_resistance']:.2f}")
    
    # 交易信号
    print("\n5. 交易信号生成...")
    signals = analysis_result['signals']
    
    if signals:
        for i, signal in enumerate(signals, 1):
            print(f"   信号{i}: {signal['type'].upper()} - {signal['reason']}")
            print(f"     入场: ${signal['entry_price']:.2f}")
            print(f"     止损: ${signal['stop_loss']:.2f}")
            print(f"     止盈: ${signal['take_profit']:.2f}")
            print(f"     信心度: {signal['confidence']:.0%}")
            print(f"     仓位: {signal['position_size']}")
    else:
        print("   无交易信号")
    
    # 多时间框架分析演示
    print("\n6. 多时间框架分析演示...")
    
    # 生成不同时间框架数据
    timeframes = {
        'D1': price_data,
        'H4': price_data.iloc[::6],  # 简化：每6个日线数据作为H4
        'H1': price_data.iloc[::24], # 简化：每24个日线数据作为H1
    }
    
    multi_tf_result = analyzer.analyze_multiple_timeframes(timeframes)
    
    print(f"   趋势一致性: {multi_tf_result['consistent_trend']}")
    print(f"   推荐操作: {multi_tf_result['recommended_action']}")
    
    print("\n=== 系统演示完成 ===")
    print("第16章趋势通道分析系统已实现并测试。")

if __name__ == "__main__":
    main()