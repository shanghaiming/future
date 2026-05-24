# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.640817

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多重时间框架协调量化系统
第17章：多重时间框架协调
AL Brooks《价格行为交易之区间篇》

核心概念：
1. 时间框架层级：月线、周线、日线、4小时、1小时、15分钟
2. 信号一致性分析：各时间框架趋势方向一致性
3. 冲突解决：处理不同时间框架的冲突信号
4. 权重分配：根据时间框架重要性分配权重
5. 综合决策：生成基于多时间框架的最终交易信号
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

# 导入第16章的趋势通道分析器
try:
    from trend_channel_analyzer import TrendChannelAnalyzer
except ImportError:
    # 如果导入失败，定义简化版本
    class TrendChannelAnalyzer:
        def identify_trend_channels(self, price_data):
            return {'channel_type': 'unknown'}

class MultiTimeframeCoordinator:
    """多重时间框架协调器"""
    
    def __init__(self,
                 timeframe_weights: Optional[Dict[str, float]] = None,
                 consistency_threshold: float = 0.7):
        """
        初始化多时间框架协调器
        
        参数:
            timeframe_weights: 时间框架权重 {timeframe: weight}
            consistency_threshold: 一致性阈值（0-1）
        """
        # 默认时间框架权重（越高层权重越大）
        self.default_weights = {
            'MN': 0.25,   # 月线
            'W':  0.20,   # 周线  
            'D':  0.18,   # 日线
            'H4': 0.15,   # 4小时
            'H1': 0.12,   # 1小时
            'M15': 0.10,  # 15分钟
        }
        
        self.timeframe_weights = timeframe_weights or self.default_weights
        self.consistency_threshold = consistency_threshold
        self.channel_analyzer = TrendChannelAnalyzer()
        
        # 趋势类型映射到数值
        self.trend_value_map = {
            'uptrend': 1,
            'mostly_uptrend': 0.7,
            'range': 0,
            'mostly_downtrend': -0.7,
            'downtrend': -1,
            'unknown': 0
        }
    
    def analyze_timeframe_alignment(self,
                                   price_data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        分析多时间框架对齐情况
        
        参数:
            price_data_dict: {timeframe: price_data}
            
        返回:
            时间框架对齐分析结果
        """
        if not price_data_dict:
            return {'error': '无价格数据'}
        
        results = {}
        
        # 分析每个时间框架
        for timeframe, data in price_data_dict.items():
            if len(data) < 20:  # 最小数据要求
                continue
                
            # 使用趋势通道分析器
            channel_result = self.channel_analyzer.identify_trend_channels(data)
            
            # 提取关键信息
            trend_type = channel_result.get('channel_type', 'unknown')
            trend_value = self.trend_value_map.get(trend_type, 0)
            
            # 如果有通道信息，提取更多指标
            channels = channel_result.get('channels', {})
            breakouts = channel_result.get('breakouts', {})
            signals = channel_result.get('signals', [])
            
            results[timeframe] = {
                'trend_type': trend_type,
                'trend_value': trend_value,
                'channel_width': channels.get('channel_width', 0),
                'channel_angle': channels.get('channel_angle', 0),
                'bullish_breakout': breakouts.get('bullish_breakout', False),
                'bearish_breakout': breakouts.get('bearish_breakout', False),
                'breakout_strength': breakouts.get('breakout_strength', 0),
                'signals': signals,
                'current_price': channel_result.get('current_price', 0),
                'data_points': len(data),
                'weight': self.timeframe_weights.get(timeframe, 0.1)
            }
        
        # 计算整体一致性
        consistency_analysis = self._calculate_consistency(results)
        
        # 识别冲突信号
        conflict_analysis = self._identify_conflicts(results)
        
        # 生成综合信号
        consensus_signal = self._generate_consensus_signal(results, consistency_analysis)
        
        return {
            'timeframe_results': results,
            'consistency_analysis': consistency_analysis,
            'conflict_analysis': conflict_analysis,
            'consensus_signal': consensus_signal,
            'recommended_action': self._generate_recommendation(consistency_analysis, consensus_signal)
        }
    
    def _calculate_consistency(self, results: Dict) -> Dict:
        """
        计算时间框架一致性
        
        返回:
            一致性分析结果
        """
        if not results:
            return {'overall_consistency': 0, 'trend_consistency': 0, 'details': {}}
        
        # 提取趋势值和权重
        trend_values = []
        weights = []
        
        for tf, result in results.items():
            trend_values.append(result['trend_value'])
            weights.append(result['weight'])
        
        trend_values = np.array(trend_values)
        weights = np.array(weights)
        
        # 加权平均趋势值
        weighted_trend = np.average(trend_values, weights=weights)
        
        # 计算一致性（趋势值符号相同的比例）
        positive_count = np.sum(trend_values > 0)
        negative_count = np.sum(trend_values < 0)
        total_count = len(trend_values)
        
        if total_count > 0:
            max_aligned = max(positive_count, negative_count)
            trend_consistency = max_aligned / total_count
        else:
            trend_consistency = 0
        
        # 计算加权一致性
        positive_weight = np.sum(weights[trend_values > 0])
        negative_weight = np.sum(weights[trend_values < 0])
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            weighted_consistency = max(positive_weight, negative_weight) / total_weight
        else:
            weighted_consistency = 0
        
        # 判断整体趋势方向
        if weighted_trend > 0.3:
            overall_trend = 'strong_uptrend'
        elif weighted_trend > 0.1:
            overall_trend = 'uptrend'
        elif weighted_trend < -0.3:
            overall_trend = 'strong_downtrend'
        elif weighted_trend < -0.1:
            overall_trend = 'downtrend'
        else:
            overall_trend = 'range'
        
        return {
            'overall_trend': overall_trend,
            'weighted_trend_value': weighted_trend,
            'trend_consistency': trend_consistency,
            'weighted_consistency': weighted_consistency,
            'positive_count': int(positive_count),
            'negative_count': int(negative_count),
            'neutral_count': int(total_count - positive_count - negative_count),
            'total_timeframes': total_count,
            'is_consistent': weighted_consistency >= self.consistency_threshold
        }
    
    def _identify_conflicts(self, results: Dict) -> Dict:
        """
        识别时间框架冲突
        
        返回:
            冲突分析结果
        """
        if len(results) < 2:
            return {'has_conflicts': False, 'conflicting_pairs': []}
        
        timeframes = list(results.keys())
        conflicting_pairs = []
        
        for i in range(len(timeframes)):
            for j in range(i + 1, len(timeframes)):
                tf1 = timeframes[i]
                tf2 = timeframes[j]
                
                trend1 = results[tf1]['trend_value']
                trend2 = results[tf2]['trend_value']
                
                # 冲突定义：趋势方向相反且绝对值都大于0.2
                if (trend1 > 0.2 and trend2 < -0.2) or (trend1 < -0.2 and trend2 > 0.2):
                    conflicting_pairs.append({
                        'pair': (tf1, tf2),
                        'trend1': trend1,
                        'trend2': trend2,
                        'weight1': results[tf1]['weight'],
                        'weight2': results[tf2]['weight'],
                        'severity': abs(trend1 - trend2)  # 差异越大，冲突越严重
                    })
        
        # 计算冲突严重性分数
        if conflicting_pairs:
            severities = [cp['severity'] for cp in conflicting_pairs]
            avg_severity = np.mean(severities)
            max_severity = np.max(severities)
            
            # 根据冲突数量和严重性评估整体冲突
            conflict_score = (len(conflicting_pairs) / len(timeframes)) * avg_severity
        else:
            avg_severity = 0
            max_severity = 0
            conflict_score = 0
        
        return {
            'has_conflicts': len(conflicting_pairs) > 0,
            'conflicting_pairs': conflicting_pairs,
            'conflict_count': len(conflicting_pairs),
            'avg_severity': avg_severity,
            'max_severity': max_severity,
            'conflict_score': conflict_score,
            'recommendation': self._generate_conflict_recommendation(conflicting_pairs, results)
        }
    
    def _generate_conflict_recommendation(self, 
                                         conflicting_pairs: List,
                                         results: Dict) -> str:
        """生成冲突解决建议"""
        if not conflicting_pairs:
            return "无时间框架冲突"
        
        # 找出最严重的冲突
        most_severe = max(conflicting_pairs, key=lambda x: x['severity'])
        tf1, tf2 = most_severe['pair']
        
        # 根据权重给出建议
        weight1 = results[tf1]['weight']
        weight2 = results[tf2]['weight']
        
        if weight1 > weight2 * 1.5:
            return f"主要冲突：{tf1}(权重{weight1:.2f})与{tf2}(权重{weight2:.2f})趋势相反，建议优先遵循{tf1}时间框架"
        elif weight2 > weight1 * 1.5:
            return f"主要冲突：{tf1}(权重{weight1:.2f})与{tf2}(权重{weight2:.2f})趋势相反，建议优先遵循{tf2}时间框架"
        else:
            return f"主要冲突：{tf1}与{tf2}趋势相反且权重相近，建议降低仓位或等待更明确信号"
    
    def _generate_consensus_signal(self,
                                  results: Dict,
                                  consistency: Dict) -> Dict:
        """
        生成综合共识信号
        
        返回:
            共识交易信号
        """
        if not results:
            return {'signal': 'neutral', 'confidence': 0, 'reason': '无数据'}
        
        # 收集各时间框架信号
        all_signals = []
        signal_weights = []
        
        for tf, result in results.items():
            signals = result.get('signals', [])
            weight = result['weight']
            
            for signal in signals:
                # 转换信号为数值
                if signal['type'] == 'buy':
                    signal_value = 1
                elif signal['type'] == 'sell':
                    signal_value = -1
                else:
                    signal_value = 0
                
                # 考虑信号信心度
                confidence = signal.get('confidence', 0.5)
                weighted_signal = signal_value * confidence * weight
                
                all_signals.append({
                    'timeframe': tf,
                    'type': signal['type'],
                    'value': signal_value,
                    'weighted_value': weighted_signal,
                    'confidence': confidence,
                    'reason': signal.get('reason', ''),
                    'original_signal': signal
                })
                signal_weights.append(weight * confidence)
        
        if not all_signals:
            # 如果没有具体信号，基于趋势生成信号
            weighted_trend = consistency.get('weighted_trend_value', 0)
            
            if weighted_trend > 0.2:
                return {
                    'signal': 'buy',
                    'confidence': min(0.7, weighted_trend),
                    'reason': f'加权趋势值{weighted_trend:.2f}显示上升趋势',
                    'weighted_trend': weighted_trend,
                    'signal_source': 'trend_based'
                }
            elif weighted_trend < -0.2:
                return {
                    'signal': 'sell',
                    'confidence': min(0.7, -weighted_trend),
                    'reason': f'加权趋势值{weighted_trend:.2f}显示下降趋势',
                    'weighted_trend': weighted_trend,
                    'signal_source': 'trend_based'
                }
            else:
                return {
                    'signal': 'neutral',
                    'confidence': 0.3,
                    'reason': '趋势不明确，建议观望',
                    'weighted_trend': weighted_trend,
                    'signal_source': 'trend_based'
                }
        
        # 计算加权平均信号
        total_weight = np.sum(signal_weights)
        if total_weight > 0:
            weighted_signal_sum = np.sum([s['weighted_value'] for s in all_signals])
            avg_weighted_signal = weighted_signal_sum / total_weight
        else:
            avg_weighted_signal = 0
        
        # 确定最终信号
        if avg_weighted_signal > 0.1:
            final_signal = 'buy'
            confidence = min(0.9, avg_weighted_signal * 2)
        elif avg_weighted_signal < -0.1:
            final_signal = 'sell'
            confidence = min(0.9, -avg_weighted_signal * 2)
        else:
            final_signal = 'neutral'
            confidence = 0.3
        
        # 找出最重要的信号源
        if all_signals:
            primary_signal = max(all_signals, key=lambda x: abs(x['weighted_value']))
            reason = f"主要信号来自{primary_signal['timeframe']}: {primary_signal['reason']}"
        else:
            reason = "基于趋势分析"
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'reason': reason,
            'avg_weighted_signal': avg_weighted_signal,
            'signal_count': len(all_signals),
            'signal_source': 'multi_timeframe_integration',
            'detailed_signals': all_signals[:3]  # 只保留前3个详细信号
        }
    
    def _generate_recommendation(self,
                                consistency: Dict,
                                consensus_signal: Dict) -> str:
        """
        生成最终交易建议
        
        返回:
            人类可读的交易建议
        """
        signal = consensus_signal['signal']
        confidence = consensus_signal['confidence']
        is_consistent = consistency['is_consistent']
        
        if signal == 'buy':
            if is_consistent and confidence > 0.7:
                return "🟢 强烈买入：多时间框架高度一致显示上升趋势，建议积极买入"
            elif is_consistent and confidence > 0.5:
                return "🟡 买入：多时间框架显示上升趋势，建议适度买入"
            else:
                return "🟡 谨慎买入：趋势方向一致但信心度一般，建议轻仓买入"
        
        elif signal == 'sell':
            if is_consistent and confidence > 0.7:
                return "🔴 强烈卖出：多时间框架高度一致显示下降趋势，建议积极卖出"
            elif is_consistent and confidence > 0.5:
                return "🟠 卖出：多时间框架显示下降趋势，建议适度卖出"
            else:
                return "🟠 谨慎卖出：趋势方向一致但信心度一般，建议轻仓卖出"
        
        else:  # neutral
            if not is_consistent:
                return "⚪ 观望：时间框架存在冲突，建议等待更明确信号"
            elif confidence < 0.3:
                return "⚪ 观望：趋势不明确，建议等待方向确认"
            else:
                return "⚪ 保持中性：市场处于震荡区间，可考虑区间交易策略"
    
    def calculate_optimal_timeframe_weights(self,
                                          historical_performance: Dict[str, float]) -> Dict[str, float]:
        """
        基于历史表现计算最优时间框架权重
        
        参数:
            historical_performance: {timeframe: performance_score}
            
        返回:
            优化后的权重
        """
        if not historical_performance:
            return self.default_weights
        
        # 归一化历史表现分数
        scores = np.array(list(historical_performance.values()))
        timeframes = list(historical_performance.keys())
        
        if np.sum(scores) > 0:
            normalized_scores = scores / np.sum(scores)
        else:
            normalized_scores = np.ones(len(scores)) / len(scores)
        
        # 结合默认权重和历史表现
        optimized_weights = {}
        for i, tf in enumerate(timeframes):
            default_weight = self.default_weights.get(tf, 0.1)
            historical_weight = normalized_scores[i]
            
            # 加权平均：70%历史表现 + 30%默认权重
            optimized_weight = historical_weight * 0.7 + default_weight * 0.3
            optimized_weights[tf] = optimized_weight
        
        # 确保权重和为1
        total = sum(optimized_weights.values())
        if total > 0:
            optimized_weights = {tf: w/total for tf, w in optimized_weights.items()}
        
        return optimized_weights
    
    def generate_timeframe_analysis_report(self,
                                          analysis_result: Dict) -> str:
        """
        生成多时间框架分析报告
        
        返回:
            格式化报告字符串
        """
        results = analysis_result['timeframe_results']
        consistency = analysis_result['consistency_analysis']
        conflict = analysis_result['conflict_analysis']
        consensus = analysis_result['consensus_signal']
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("多重时间框架协调分析报告")
        report_lines.append("=" * 60)
        
        # 时间框架详情
        report_lines.append("\n📊 各时间框架分析:")
        report_lines.append("-" * 40)
        
        for tf, result in results.items():
            trend_type = result['trend_type']
            weight = result['weight']
            signals = result['signals']
            
            signal_text = "无信号"
            if signals:
                signal_text = f"{signals[0]['type']} ({signals[0]['confidence']:.0%})"
            
            report_lines.append(f"{tf:>4} | 趋势: {trend_type:12} | 权重: {weight:.2f} | 信号: {signal_text}")
        
        # 一致性分析
        report_lines.append("\n🎯 一致性分析:")
        report_lines.append("-" * 40)
        report_lines.append(f"整体趋势: {consistency['overall_trend']}")
        report_lines.append(f"趋势一致性: {consistency['trend_consistency']:.1%}")
        report_lines.append(f"加权一致性: {consistency['weighted_consistency']:.1%}")
        report_lines.append(f"是否一致: {'是' if consistency['is_consistent'] else '否'}")
        
        # 冲突分析
        report_lines.append("\n⚡ 冲突分析:")
        report_lines.append("-" * 40)
        if conflict['has_conflicts']:
            report_lines.append(f"冲突数量: {conflict['conflict_count']}")
            report_lines.append(f"平均冲突严重性: {conflict['avg_severity']:.2f}")
            report_lines.append(f"冲突建议: {conflict['recommendation']}")
        else:
            report_lines.append("无显著时间框架冲突")
        
        # 共识信号
        report_lines.append("\n🚦 共识交易信号:")
        report_lines.append("-" * 40)
        report_lines.append(f"信号: {consensus['signal'].upper()}")
        report_lines.append(f"信心度: {consensus['confidence']:.1%}")
        report_lines.append(f"理由: {consensus['reason']}")
        
        # 最终建议
        report_lines.append("\n💡 最终交易建议:")
        report_lines.append("-" * 40)
        report_lines.append(analysis_result['recommended_action'])
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)

def generate_multi_timeframe_sample_data() -> Dict[str, pd.DataFrame]:
    """生成多时间框架样本数据"""
    np.random.seed(42)
    
    # 生成日线数据（基础）
    n_daily = 200
    time_daily = np.arange(n_daily)
    
    # 上升趋势加波动
    base_trend = 100 + time_daily * 0.15
    oscillation = np.sin(time_daily * 0.08) * 8
    noise = np.random.normal(0, 1.5, n_daily)
    
    daily_prices = base_trend + oscillation + noise
    
    # 创建日线DataFrame
    daily_df = pd.DataFrame({
        'open': daily_prices * 0.998,
        'high': daily_prices * 1.008,
        'low': daily_prices * 0.992,
        'close': daily_prices,
        'volume': np.random.randint(5000, 20000, n_daily)
    }, index=pd.date_range('2024-01-01', periods=n_daily, freq='D'))
    
    # 从日线生成其他时间框架（简化）
    h4_df = daily_df.iloc[::6].copy()   # 每6个日线数据作为H4
    h1_df = daily_df.iloc[::24].copy()  # 每24个日线数据作为H1
    w_df = daily_df.iloc[::7].copy()    # 每周数据
    
    # 添加一些差异以模拟真实情况
    h4_df['close'] = h4_df['close'] * (1 + np.random.normal(0, 0.01, len(h4_df)))
    h1_df['close'] = h1_df['close'] * (1 + np.random.normal(0, 0.02, len(h1_df)))
    
    return {
        'D': daily_df,   # 日线
        'W': w_df,       # 周线
        'H4': h4_df,     # 4小时
        'H1': h1_df,     # 1小时
    }

def main():
    """主函数：演示多时间框架协调系统"""
    print("=== 多重时间框架协调量化系统 ===")
    print("第17章：多重时间框架协调\n")
    
    # 生成多时间框架样本数据
    print("1. 生成多时间框架样本数据...")
    price_data_dict = generate_multi_timeframe_sample_data()
    
    print("   时间框架数据:")
    for tf, data in price_data_dict.items():
        print(f"   {tf}: {len(data)}个数据点，{data.index[0].date()} 到 {data.index[-1].date()}")
    
    # 创建多时间框架协调器
    print("\n2. 创建多时间框架协调器...")
    coordinator = MultiTimeframeCoordinator(
        timeframe_weights={'D': 0.25, 'W': 0.25, 'H4': 0.25, 'H1': 0.25},
        consistency_threshold=0.65
    )
    
    # 分析时间框架对齐
    print("\n3. 分析时间框架对齐...")
    analysis_result = coordinator.analyze_timeframe_alignment(price_data_dict)
    
    # 生成分析报告
    print("\n4. 生成分析报告...")
    report = coordinator.generate_timeframe_analysis_report(analysis_result)
    print(report)
    
    # 显示详细分析结果
    print("\n5. 详细分析结果:")
    print("-" * 40)
    
    consistency = analysis_result['consistency_analysis']
    print(f"加权趋势值: {consistency['weighted_trend_value']:.3f}")
    print(f"趋势一致性: {consistency['trend_consistency']:.1%}")
    print(f"是否一致: {consistency['is_consistent']}")
    
    conflict = analysis_result['conflict_analysis']
    if conflict['has_conflicts']:
        print(f"检测到{conflict['conflict_count']}个时间框架冲突")
        print(f"最严重冲突: {conflict['conflicting_pairs'][0]['pair'] if conflict['conflicting_pairs'] else '无'}")
    
    consensus = analysis_result['consensus_signal']
    print(f"\n共识信号: {consensus['signal']}")
    print(f"信号信心度: {consensus['confidence']:.1%}")
    
    # 优化权重演示
    print("\n6. 权重优化演示...")
    historical_performance = {'D': 0.8, 'W': 0.7, 'H4': 0.6, 'H1': 0.5}
    optimized_weights = coordinator.calculate_optimal_timeframe_weights(historical_performance)
    
    print("   历史表现:", historical_performance)
    print("   优化后权重:", {k: f"{v:.3f}" for k, v in optimized_weights.items()})
    
    print("\n=== 系统演示完成 ===")
    print("第17章多重时间框架协调系统已实现并测试。")


# ============================================================================
# 策略改造: 添加PriceActionRangesMultiTimeframeCoordinatorStrategy类
# 将价格行为区间多重时间框架协调系统转换为交易策略
# ============================================================================

class PriceActionRangesMultiTimeframeCoordinatorStrategy(BaseStrategy):
    """价格行为区间多重时间框架协调策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        timeframe_weights = params.get('timeframe_weights', {'D': 0.4, 'W': 0.3, 'H4': 0.2, 'H1': 0.1})
        
        # 创建多重时间框架协调器实例
        self.coordinator = MultiTimeframeCoordinator(timeframe_weights=timeframe_weights)
    
    def generate_signals(self):
        """
        生成交易信号

        基于多重时间框架协调生成交易信号，使用短/中/长MA对齐度
        """
        df = self.data
        if len(df) < 30:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(df['close'].iloc[-1]))
            return self.signals

        close = df['close']

        # Multi-timeframe proxy: different MA periods
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()

        # Alignment: all MAs stacked properly
        bullish_stack = (ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1])
        bearish_stack = (ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1] < ma50.iloc[-1])

        # MACD confirmation
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_d = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss_d + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        last_close = close.iloc[-1]

        if bullish_stack and macd_line.iloc[-1] > signal_line.iloc[-1]:
            self._record_signal(timestamp=df.index[-1], action='buy', price=float(last_close))
        elif bearish_stack and macd_line.iloc[-1] < signal_line.iloc[-1]:
            self._record_signal(timestamp=df.index[-1], action='sell', price=float(last_close))
        else:
            # Check partial alignment
            short_align = ma5.iloc[-1] > ma10.iloc[-1]
            long_align = ma20.iloc[-1] > ma50.iloc[-1]
            if short_align and long_align:
                self._record_signal(timestamp=df.index[-1], action='buy', price=float(last_close))
            elif not short_align and not long_align:
                self._record_signal(timestamp=df.index[-1], action='sell', price=float(last_close))
            else:
                self._record_signal(timestamp=df.index[-1], action='hold', price=float(last_close))

        return self.signals


# ============================================================================
# 策略改造完成
# ============================================================================

if __name__ == "__main__":
    main()