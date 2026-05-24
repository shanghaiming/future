# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.629772

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场结构识别量化系统
第18章：市场结构识别
AL Brooks《价格行为交易之区间篇》

核心概念：
1. 市场结构类型识别：趋势市场、区间市场、转折市场
2. 摆动点检测：识别价格的高点和低点序列
3. 结构完整性分析：支撑阻力层级、结构突破信号
4. 结构转换检测：从一种结构转换到另一种结构
5. 基于结构的交易信号：根据市场结构调整交易策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class MarketStructureIdentifier:
    """市场结构识别器"""
    
    def __init__(self,
                 lookback_period: int = 100,
                 swing_sensitivity: float = 2.0,
                 structure_confirmation_bars: int = 3):
        """
        初始化市场结构识别器
        
        参数:
            lookback_period: 回溯周期
            swing_sensitivity: 摆动点敏感度（ATR倍数）
            structure_confirmation_bars: 结构确认所需K线数
        """
        self.lookback_period = lookback_period
        self.swing_sensitivity = swing_sensitivity
        self.structure_confirmation_bars = structure_confirmation_bars
        self.structure_history = []
        
    def identify_market_structure(self,
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        识别市场结构
        
        参数:
            price_data: 价格数据，需包含'high', 'low', 'close', 'volume'列
            
        返回:
            市场结构分析结果
        """
        if len(price_data) < 20:
            return {'error': '数据不足，至少需要20个数据点'}
        
        # 使用最近的数据
        recent_data = price_data.tail(self.lookback_period).copy()
        
        # 检测摆动点
        swing_points = self._detect_swing_points(recent_data)
        
        # 识别市场结构类型
        structure_type = self._identify_structure_type(swing_points, recent_data)
        
        # 分析结构完整性
        structure_integrity = self._analyze_structure_integrity(swing_points, recent_data)
        
        # 检测结构突破
        structure_breakdown = self._detect_structure_breakdown(swing_points, recent_data, structure_type)
        
        # 分析结构转换
        structure_transitions = self._analyze_structure_transitions(swing_points, recent_data)
        
        # 生成基于结构的交易信号
        structure_signals = self._generate_structure_based_signals(
            structure_type, structure_integrity, structure_breakdown, recent_data
        )
        
        result = {
            'structure_type': structure_type,
            'structure_integrity': structure_integrity,
            'structure_breakdown': structure_breakdown,
            'structure_transitions': structure_transitions,
            'structure_signals': structure_signals,
            'swing_points': swing_points,
            'current_price': recent_data['close'].iloc[-1],
            'current_strength': self._calculate_structure_strength(structure_type, structure_integrity),
            'analysis_timestamp': pd.Timestamp.now(),
            'data_points_analyzed': len(recent_data)
        }
        
        # 保存到历史
        self.structure_history.append(result)
        
        return result
    
    def _detect_swing_points(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """
        检测摆动点（高点和低点）- 优化版本
        
        返回:
            {'highs': [(index, price), ...], 'lows': [(index, price), ...]}
        """
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # 计算ATR用于摆动点阈值
        atr = self._calculate_atr(data)
        swing_threshold = atr * self.swing_sensitivity
        
        high_points = []
        low_points = []
        
        # 使用更灵敏的摆动点检测算法
        # 方法1：使用价格变化百分比和极值检测
        for i in range(5, len(closes) - 5):
            # 检查前5根和后5根K线
            left_window = closes[i-5:i]
            right_window = closes[i+1:i+6]
            
            # 高点检测：当前close是窗口内的最高点
            if closes[i] == max(list(left_window) + [closes[i]] + list(right_window)):
                # 进一步验证：high也应该是高点
                if highs[i] >= np.max(highs[i-3:i+4]):
                    # 计算与周围平均价格的差异
                    surrounding_avg = np.mean(closes[i-3:i+4])
                    price_diff_pct = (closes[i] - surrounding_avg) / surrounding_avg * 100
                    
                    # 更宽松的阈值：至少0.3%的差异
                    if price_diff_pct >= 0.3:
                        high_points.append((i, highs[i]))
            
            # 低点检测：当前close是窗口内的最低点
            if closes[i] == min(list(left_window) + [closes[i]] + list(right_window)):
                # 进一步验证：low也应该是低点
                if lows[i] <= np.min(lows[i-3:i+4]):
                    # 计算与周围平均价格的差异
                    surrounding_avg = np.mean(closes[i-3:i+4])
                    price_diff_pct = (surrounding_avg - closes[i]) / surrounding_avg * 100
                    
                    # 更宽松的阈值：至少0.3%的差异
                    if price_diff_pct >= 0.3:
                        low_points.append((i, lows[i]))
        
        # 方法2：补充检测明显的转折点
        if len(high_points) < 3 or len(low_points) < 3:
            # 使用简单的极值点检测作为补充
            for i in range(2, len(closes) - 2):
                # 简单的高点检测
                if (closes[i] > closes[i-1] and closes[i] > closes[i-2] and
                    closes[i] > closes[i+1] and closes[i] > closes[i+2]):
                    # 避免重复添加
                    if not any(abs(idx - i) < 5 for idx, _ in high_points):
                        high_points.append((i, highs[i]))
                
                # 简单的低点检测
                if (closes[i] < closes[i-1] and closes[i] < closes[i-2] and
                    closes[i] < closes[i+1] and closes[i] < closes[i+2]):
                    # 避免重复添加
                    if not any(abs(idx - i) < 5 for idx, _ in low_points):
                        low_points.append((i, lows[i]))
        
        # 按价格排序，保留最重要的摆动点
        if len(high_points) > 10:
            high_points = sorted(high_points, key=lambda x: x[1], reverse=True)[:10]
            high_points = sorted(high_points, key=lambda x: x[0])  # 按索引重新排序
        
        if len(low_points) > 10:
            low_points = sorted(low_points, key=lambda x: x[1])[:10]
            low_points = sorted(low_points, key=lambda x: x[0])  # 按索引重新排序
        
        return {'highs': high_points, 'lows': low_points}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """计算平均真实波幅 (ATR)"""
        if len(data) < period + 1:
            return 0.01
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr = np.zeros(len(data))
        for i in range(1, len(data)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        atr = np.mean(tr[-period:])
        return atr
    
    def _identify_structure_type(self,
                                swing_points: Dict[str, List[Tuple[int, float]]],
                                data: pd.DataFrame) -> Dict[str, Any]:
        """
        识别市场结构类型
        
        返回:
            结构类型分析结果
        """
        highs = swing_points['highs']
        lows = swing_points['lows']
        
        if len(highs) < 3 or len(lows) < 3:
            return {'type': 'unknown', 'confidence': 0, 'reason': '摆动点不足'}
        
        # 提取价格序列
        high_prices = [price for _, price in highs]
        low_prices = [price for _, price in lows]
        
        # 分析高点序列趋势
        if len(high_prices) >= 2:
            high_trend = self._analyze_price_sequence(high_prices)
        else:
            high_trend = {'trend': 'unknown', 'slope': 0}
        
        # 分析低点序列趋势
        if len(low_prices) >= 2:
            low_trend = self._analyze_price_sequence(low_prices)
        else:
            low_trend = {'trend': 'unknown', 'slope': 0}
        
        # 确定市场结构类型
        if high_trend['trend'] == 'uptrend' and low_trend['trend'] == 'uptrend':
            structure_type = 'uptrend'
            confidence = min(high_trend['confidence'], low_trend['confidence'])
            reason = '高点和高点均呈上升趋势'
        
        elif high_trend['trend'] == 'downtrend' and low_trend['trend'] == 'downtrend':
            structure_type = 'downtrend'
            confidence = min(high_trend['confidence'], low_trend['confidence'])
            reason = '高点和高点均呈下降趋势'
        
        elif (high_trend['trend'] == 'range' or abs(high_trend['slope']) < 0.001) and \
             (low_trend['trend'] == 'range' or abs(low_trend['slope']) < 0.001):
            structure_type = 'range'
            confidence = 0.8
            reason = '高点和高点均在区间内震荡'
        
        elif (high_trend['trend'] == 'downtrend' and low_trend['trend'] == 'uptrend') or \
             (high_trend['trend'] == 'uptrend' and low_trend['trend'] == 'downtrend'):
            structure_type = 'transition'
            confidence = 0.7
            reason = '高点和高点趋势相反，市场处于转换期'
        
        else:
            structure_type = 'complex'
            confidence = 0.5
            reason = '复杂的混合结构'
        
        # 计算结构强度
        current_price = data['close'].iloc[-1]
        volatility = data['close'].pct_change().std() * np.sqrt(252)
        
        return {
            'type': structure_type,
            'confidence': confidence,
            'reason': reason,
            'high_trend': high_trend,
            'low_trend': low_trend,
            'current_price': current_price,
            'volatility': volatility,
            'strength_score': self._calculate_structure_score(structure_type, high_trend, low_trend)
        }
    
    def _analyze_price_sequence(self, prices: List[float]) -> Dict[str, Any]:
        """分析价格序列趋势"""
        if len(prices) < 2:
            return {'trend': 'unknown', 'slope': 0, 'confidence': 0}
        
        # 线性回归分析趋势
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # 计算R²
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 确定趋势类型
        if abs(slope) < 0.0001:
            trend = 'range'
            confidence = max(0.3, r_squared)
        elif slope > 0:
            trend = 'uptrend'
            confidence = max(0.3, r_squared)
        else:
            trend = 'downtrend'
            confidence = max(0.3, r_squared)
        
        # 计算角度（度）
        angle = np.degrees(np.arctan(slope))
        
        return {
            'trend': trend,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'confidence': confidence,
            'angle': angle,
            'price_count': len(prices)
        }
    
    def _calculate_structure_score(self,
                                  structure_type: str,
                                  high_trend: Dict,
                                  low_trend: Dict) -> float:
        """计算结构强度分数（0-1）"""
        if structure_type == 'unknown':
            return 0.3
        
        # 基础分数
        base_scores = {
            'uptrend': 0.8,
            'downtrend': 0.8,
            'range': 0.7,
            'transition': 0.5,
            'complex': 0.4
        }
        
        base_score = base_scores.get(structure_type, 0.5)
        
        # 根据趋势一致性调整
        if high_trend['trend'] == low_trend['trend']:
            consistency_bonus = 0.15
        else:
            consistency_bonus = -0.1
        
        # 根据R²调整
        avg_r2 = (high_trend['r_squared'] + low_trend['r_squared']) / 2
        r2_bonus = avg_r2 * 0.2
        
        final_score = base_score + consistency_bonus + r2_bonus
        return max(0.1, min(0.95, final_score))
    
    def _analyze_structure_integrity(self,
                                    swing_points: Dict[str, List[Tuple[int, float]]],
                                    data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析结构完整性
        
        返回:
            结构完整性分析结果
        """
        highs = swing_points['highs']
        lows = swing_points['lows']
        
        if len(highs) < 2 or len(lows) < 2:
            return {'integrity': 'weak', 'score': 0.3, 'issues': ['摆动点不足']}
        
        # 提取最近的摆动点
        recent_highs = sorted(highs, key=lambda x: x[0])[-3:]
        recent_lows = sorted(lows, key=lambda x: x[0])[-3:]
        
        issues = []
        score_components = []
        
        # 检查高点序列
        if len(recent_highs) >= 2:
            high_prices = [price for _, price in recent_highs]
            high_trend = self._analyze_price_sequence(high_prices)
            
            if high_trend['r_squared'] < 0.3:
                issues.append(f"高点序列R²较低 ({high_trend['r_squared']:.2f})")
                score_components.append(0.4)
            else:
                score_components.append(0.8)
        
        # 检查低点序列
        if len(recent_lows) >= 2:
            low_prices = [price for _, price in recent_lows]
            low_trend = self._analyze_price_sequence(low_prices)
            
            if low_trend['r_squared'] < 0.3:
                issues.append(f"低点序列R²较低 ({low_trend['r_squared']:.2f})")
                score_components.append(0.4)
            else:
                score_components.append(0.8)
        
        # 检查摆动点分布
        total_swings = len(highs) + len(lows)
        expected_swings = len(data) // 20  # 每20根K线预期一个摆动点
        
        if total_swings < expected_swings * 0.5:
            issues.append(f"摆动点过少 ({total_swings}个，预期{expected_swings}个)")
            score_components.append(0.3)
        elif total_swings > expected_swings * 2:
            issues.append(f"摆动点过多 ({total_swings}个，预期{expected_swings}个)")
            score_components.append(0.6)
        else:
            score_components.append(0.9)
        
        # 计算完整性分数
        if score_components:
            integrity_score = np.mean(score_components)
        else:
            integrity_score = 0.5
        
        # 确定完整性等级
        if integrity_score >= 0.8:
            integrity_level = 'strong'
        elif integrity_score >= 0.6:
            integrity_level = 'moderate'
        else:
            integrity_level = 'weak'
        
        return {
            'integrity': integrity_level,
            'score': integrity_score,
            'issues': issues,
            'recent_highs_count': len(recent_highs),
            'recent_lows_count': len(recent_lows),
            'total_swings': total_swings
        }
    
    def _detect_structure_breakdown(self,
                                   swing_points: Dict[str, List[Tuple[int, float]]],
                                   data: pd.DataFrame,
                                   structure_type: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测结构突破
        
        返回:
            结构突破分析结果
        """
        if len(data) < 10:
            return {'breakdown': False, 'confidence': 0, 'reason': '数据不足'}
        
        current_price = data['close'].iloc[-1]
        structure = structure_type['type']
        
        # 获取最近的摆动点
        recent_highs = sorted(swing_points['highs'], key=lambda x: x[0])
        recent_lows = sorted(swing_points['lows'], key=lambda x: x[0])
        
        if not recent_highs or not recent_lows:
            return {'breakdown': False, 'confidence': 0, 'reason': '无摆动点'}
        
        latest_high_price = recent_highs[-1][1] if recent_highs else 0
        latest_low_price = recent_lows[-1][1] if recent_lows else 0
        
        # 计算ATR用于突破阈值
        atr = self._calculate_atr(data)
        breakdown_threshold = atr * 1.5
        
        breakdown_detected = False
        breakdown_type = None
        confidence = 0
        reason = ""
        
        if structure == 'uptrend':
            # 上升趋势突破：价格跌破最近的低点
            if current_price < latest_low_price - breakdown_threshold:
                breakdown_detected = True
                breakdown_type = 'uptrend_breakdown'
                confidence = min(0.9, (latest_low_price - current_price) / atr * 0.3)
                reason = f"价格跌破上升趋势低点{latest_low_price:.2f}"
        
        elif structure == 'downtrend':
            # 下降趋势突破：价格突破最近的高点
            if current_price > latest_high_price + breakdown_threshold:
                breakdown_detected = True
                breakdown_type = 'downtrend_breakdown'
                confidence = min(0.9, (current_price - latest_high_price) / atr * 0.3)
                reason = f"价格突破下降趋势高点{latest_high_price:.2f}"
        
        elif structure == 'range':
            # 区间突破：价格突破区间边界
            if recent_highs and recent_lows:
                range_high = max([price for _, price in recent_highs[-3:]])
                range_low = min([price for _, price in recent_lows[-3:]])
                
                if current_price > range_high + breakdown_threshold:
                    breakdown_detected = True
                    breakdown_type = 'range_breakout_up'
                    confidence = min(0.9, (current_price - range_high) / atr * 0.3)
                    reason = f"价格向上突破区间上轨{range_high:.2f}"
                elif current_price < range_low - breakdown_threshold:
                    breakdown_detected = True
                    breakdown_type = 'range_breakout_down'
                    confidence = min(0.9, (range_low - current_price) / atr * 0.3)
                    reason = f"价格向下突破区间下轨{range_low:.2f}"
        
        # 检查是否只是假突破
        false_breakout = False
        if breakdown_detected and len(data) >= 5:
            # 检查价格是否快速返回
            prev_prices = data['close'].iloc[-5:-1].values
            if breakdown_type in ['uptrend_breakdown', 'range_breakout_down']:
                # 向下突破后是否快速反弹
                if np.any(prev_prices < current_price):
                    false_breakout = True
                    confidence *= 0.5
                    reason += "（疑似假突破）"
            elif breakdown_type in ['downtrend_breakdown', 'range_breakout_up']:
                # 向上突破后是否快速回落
                if np.any(prev_prices > current_price):
                    false_breakout = True
                    confidence *= 0.5
                    reason += "（疑似假突破）"
        
        return {
            'breakdown': breakdown_detected,
            'breakdown_type': breakdown_type,
            'confidence': confidence,
            'reason': reason,
            'false_breakout': false_breakout,
            'threshold': breakdown_threshold,
            'current_vs_high': current_price - latest_high_price if latest_high_price else 0,
            'current_vs_low': current_price - latest_low_price if latest_low_price else 0
        }
    
    def _analyze_structure_transitions(self,
                                      swing_points: Dict[str, List[Tuple[int, float]]],
                                      data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析结构转换
        
        返回:
            结构转换分析结果
        """
        if len(self.structure_history) < 3:
            return {'transition': False, 'confidence': 0, 'reason': '历史数据不足'}
        
        # 获取最近的结构历史
        recent_structures = [h['structure_type'] for h in self.structure_history[-3:]]
        
        # 检查结构是否发生变化
        structure_changed = len(set([s['type'] for s in recent_structures])) > 1
        
        if not structure_changed:
            return {
                'transition': False,
                'confidence': 0.8,
                'reason': '市场结构稳定',
                'current_structure': recent_structures[-1]['type'] if recent_structures else 'unknown',
                'stability_period': len(recent_structures)
            }
        
        # 分析转换方向
        old_structure = recent_structures[0]['type']
        new_structure = recent_structures[-1]['type']
        
        transition_types = {
            ('range', 'uptrend'): 'range_to_uptrend',
            ('range', 'downtrend'): 'range_to_downtrend',
            ('uptrend', 'range'): 'uptrend_to_range',
            ('downtrend', 'range'): 'downtrend_to_range',
            ('uptrend', 'downtrend'): 'trend_reversal',
            ('downtrend', 'uptrend'): 'trend_reversal'
        }
        
        transition_key = (old_structure, new_structure)
        transition_type = transition_types.get(transition_key, 'complex_transition')
        
        # 计算转换强度
        old_strength = recent_structures[0].get('strength_score', 0.5)
        new_strength = recent_structures[-1].get('strength_score', 0.5)
        transition_strength = abs(new_strength - old_strength)
        
        return {
            'transition': True,
            'confidence': min(0.9, transition_strength * 2),
            'reason': f'从{old_structure}转换到{new_structure}',
            'transition_type': transition_type,
            'old_structure': old_structure,
            'new_structure': new_structure,
            'transition_strength': transition_strength,
            'recommendation': self._generate_transition_recommendation(transition_type)
        }
    
    def _generate_transition_recommendation(self, transition_type: str) -> str:
        """生成结构转换建议"""
        recommendations = {
            'range_to_uptrend': '区间突破转为上升趋势，建议寻找回调买入机会',
            'range_to_downtrend': '区间突破转为下降趋势，建议寻找反弹卖出机会',
            'uptrend_to_range': '上升趋势转为区间，建议减仓或采用区间交易策略',
            'downtrend_to_range': '下降趋势转为区间，建议减仓或采用区间交易策略',
            'trend_reversal': '趋势反转，建议等待确认后反向操作',
            'complex_transition': '复杂结构转换，建议观望或降低仓位'
        }
        return recommendations.get(transition_type, '结构变化，建议谨慎操作')
    
    def _generate_structure_based_signals(self,
                                         structure_type: Dict[str, Any],
                                         structure_integrity: Dict[str, Any],
                                         structure_breakdown: Dict[str, Any],
                                         data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        生成基于市场结构的交易信号
        
        返回:
            交易信号列表
        """
        signals = []
        current_price = data['close'].iloc[-1]
        structure = structure_type['type']
        integrity_score = structure_integrity['score']
        
        # 基本结构交易规则
        if structure == 'uptrend' and integrity_score > 0.7:
            # 上升趋势：回调买入
            signals.append({
                'type': 'buy',
                'reason': f'上升趋势中，结构完整性{integrity_score:.0%}',
                'entry_price': current_price,
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.04,
                'confidence': min(0.8, structure_type['confidence'] * integrity_score),
                'position_size': 'normal',
                'structure_based': True
            })
        
        elif structure == 'downtrend' and integrity_score > 0.7:
            # 下降趋势：反弹卖出
            signals.append({
                'type': 'sell',
                'reason': f'下降趋势中，结构完整性{integrity_score:.0%}',
                'entry_price': current_price,
                'stop_loss': current_price * 1.02,
                'take_profit': current_price * 0.96,
                'confidence': min(0.8, structure_type['confidence'] * integrity_score),
                'position_size': 'normal',
                'structure_based': True
            })
        
        elif structure == 'range' and integrity_score > 0.6:
            # 区间市场：高抛低吸
            recent_high = data['high'].tail(20).max()
            recent_low = data['low'].tail(20).min()
            range_mid = (recent_high + recent_low) / 2
            
            if current_price < range_mid:
                # 价格在区间下半部，买入
                signals.append({
                    'type': 'buy',
                    'reason': f'区间市场中，价格在区间下半部',
                    'entry_price': current_price,
                    'stop_loss': recent_low * 0.995,
                    'take_profit': range_mid * 1.01,
                    'confidence': min(0.7, integrity_score * 0.8),
                    'position_size': 'reduced',
                    'structure_based': True
                })
            else:
                # 价格在区间上半部，卖出
                signals.append({
                    'type': 'sell',
                    'reason': f'区间市场中，价格在区间上半部',
                    'entry_price': current_price,
                    'stop_loss': recent_high * 1.005,
                    'take_profit': range_mid * 0.99,
                    'confidence': min(0.7, integrity_score * 0.8),
                    'position_size': 'reduced',
                    'structure_based': True
                })
        
        # 结构突破信号
        if structure_breakdown['breakdown'] and not structure_breakdown['false_breakout']:
            breakdown_type = structure_breakdown['breakdown_type']
            confidence = structure_breakdown['confidence']
            
            if breakdown_type in ['range_breakout_up', 'downtrend_breakdown']:
                # 向上突破，买入
                signals.append({
                    'type': 'buy',
                    'reason': f'结构突破: {structure_breakdown["reason"]}',
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.99,
                    'take_profit': current_price * (1 + confidence * 0.05),
                    'confidence': confidence,
                    'position_size': 'aggressive' if confidence > 0.7 else 'normal',
                    'structure_based': True,
                    'breakout_signal': True
                })
            
            elif breakdown_type in ['range_breakout_down', 'uptrend_breakdown']:
                # 向下突破，卖出
                signals.append({
                    'type': 'sell',
                    'reason': f'结构突破: {structure_breakdown["reason"]}',
                    'entry_price': current_price,
                    'stop_loss': current_price * 1.01,
                    'take_profit': current_price * (1 - confidence * 0.05),
                    'confidence': confidence,
                    'position_size': 'aggressive' if confidence > 0.7 else 'normal',
                    'structure_based': True,
                    'breakout_signal': True
                })
        
        return signals
    
    def _calculate_structure_strength(self,
                                     structure_type: Dict[str, Any],
                                     structure_integrity: Dict[str, Any]) -> float:
        """计算结构强度"""
        type_score = structure_type.get('strength_score', 0.5)
        integrity_score = structure_integrity.get('score', 0.5)
        
        # 加权平均
        strength = type_score * 0.6 + integrity_score * 0.4
        return max(0.1, min(0.95, strength))
    
    def generate_structure_report(self, analysis_result: Dict[str, Any]) -> str:
        """生成市场结构分析报告"""
        structure = analysis_result['structure_type']
        integrity = analysis_result['structure_integrity']
        breakdown = analysis_result['structure_breakdown']
        transitions = analysis_result['structure_transitions']
        signals = analysis_result['structure_signals']
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("市场结构识别分析报告")
        report_lines.append("=" * 60)
        
        # 结构类型
        report_lines.append(f"\n🏛️ 市场结构类型: {structure['type'].upper()}")
        report_lines.append(f"   信心度: {structure['confidence']:.1%}")
        report_lines.append(f"   理由: {structure['reason']}")
        report_lines.append(f"   结构强度: {analysis_result['current_strength']:.1%}")
        
        # 结构完整性
        report_lines.append(f"\n🔧 结构完整性: {integrity['integrity'].upper()}")
        report_lines.append(f"   完整性分数: {integrity['score']:.1%}")
        if integrity['issues']:
            report_lines.append(f"   问题: {', '.join(integrity['issues'])}")
        
        # 结构突破
        report_lines.append(f"\n⚡ 结构突破检测:")
        if breakdown['breakdown']:
            report_lines.append(f"   🔍 检测到突破: {breakdown['reason']}")
            report_lines.append(f"   突破类型: {breakdown['breakdown_type']}")
            report_lines.append(f"   信心度: {breakdown['confidence']:.1%}")
            if breakdown['false_breakout']:
                report_lines.append(f"   ⚠️  疑似假突破")
        else:
            report_lines.append(f"   ✅ 结构完整，无突破信号")
        
        # 结构转换
        report_lines.append(f"\n🔄 结构转换分析:")
        if transitions['transition']:
            report_lines.append(f"   🔄 检测到结构转换: {transitions['reason']}")
            report_lines.append(f"   转换类型: {transitions['transition_type']}")
            report_lines.append(f"   转换强度: {transitions['transition_strength']:.2f}")
            report_lines.append(f"   建议: {transitions['recommendation']}")
        else:
            report_lines.append(f"   ✅ 结构稳定: {transitions['reason']}")
        
        # 交易信号
        report_lines.append(f"\n🚦 结构交易信号:")
        if signals:
            for i, signal in enumerate(signals, 1):
                signal_type = "🟢 买入" if signal['type'] == 'buy' else "🔴 卖出"
                report_lines.append(f"   信号{i}: {signal_type} - {signal['reason']}")
                report_lines.append(f"     入场: ${signal['entry_price']:.2f}")
                report_lines.append(f"     止损: ${signal['stop_loss']:.2f}")
                report_lines.append(f"     止盈: ${signal['take_profit']:.2f}")
                report_lines.append(f"     信心度: {signal['confidence']:.0%}")
                report_lines.append(f"     仓位: {signal['position_size']}")
        else:
            report_lines.append(f"   无推荐交易信号")
        
        report_lines.append(f"\n📊 摆动点统计:")
        swings = analysis_result['swing_points']
        report_lines.append(f"   摆动高点: {len(swings['highs'])}个")
        report_lines.append(f"   摆动低点: {len(swings['lows'])}个")
        
        report_lines.append(f"\n💡 综合建议:")
        if signals:
            primary_signal = max(signals, key=lambda x: x['confidence'])
            action = "买入" if primary_signal['type'] == 'buy' else "卖出"
            report_lines.append(f"   主要信号: {action} (信心度{primary_signal['confidence']:.0%})")
        elif breakdown['breakdown']:
            report_lines.append(f"   关注结构突破: {breakdown['reason']}")
        elif integrity['score'] < 0.6:
            report_lines.append(f"   结构完整性较弱，建议观望")
        else:
            report_lines.append(f"   当前结构稳定，可按结构类型交易")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)

def generate_sample_price_data(n_bars: int = 200) -> pd.DataFrame:
    """生成样本价格数据"""
    np.random.seed(42)
    
    # 生成上升趋势数据
    time = np.arange(n_bars)
    base_trend = 100 + time * 0.1
    oscillation = np.sin(time * 0.1) * 5
    noise = np.random.normal(0, 0.8, n_bars)
    
    prices = base_trend + oscillation + noise
    
    # 生成OHLC数据
    df = pd.DataFrame({
        'open': prices * 0.998,
        'high': prices * 1.008 + np.random.normal(0, 0.2, n_bars),
        'low': prices * 0.992 - np.random.normal(0, 0.2, n_bars),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='D'))
    
    return df

def main():
    """主函数：演示市场结构识别系统"""
    print("=== 市场结构识别量化系统 ===")
    print("第18章：市场结构识别\n")
    
    # 生成样本数据
    print("1. 生成样本价格数据...")
    price_data = generate_sample_price_data(200)
    print(f"   数据形状: {price_data.shape}")
    print(f"   时间范围: {price_data.index[0].date()} 到 {price_data.index[-1].date()}")
    print(f"   价格范围: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
    
    # 创建市场结构识别器
    print("\n2. 创建市场结构识别器...")
    identifier = MarketStructureIdentifier(
        lookback_period=100,
        swing_sensitivity=2.0,
        structure_confirmation_bars=3
    )
    
    # 分析市场结构
    print("\n3. 分析市场结构...")
    analysis_result = identifier.identify_market_structure(price_data)
    
    # 生成分析报告
    print("\n4. 生成分析报告...")
    report = identifier.generate_structure_report(analysis_result)
    print(report)
    
    # 显示详细分析结果
    print("\n5. 详细分析结果:")
    print("-" * 40)
    
    structure = analysis_result['structure_type']
    print(f"市场结构: {structure['type']}")
    print(f"结构信心度: {structure['confidence']:.1%}")
    print(f"结构强度: {analysis_result['current_strength']:.1%}")
    
    integrity = analysis_result['structure_integrity']
    print(f"\n结构完整性: {integrity['integrity']}")
    print(f"完整性分数: {integrity['score']:.1%}")
    
    breakdown = analysis_result['structure_breakdown']
    if breakdown['breakdown']:
        print(f"\n检测到结构突破: {breakdown['breakdown_type']}")
        print(f"突破信心度: {breakdown['confidence']:.1%}")
    
    signals = analysis_result['structure_signals']
    print(f"\n生成信号数: {len(signals)}")
    if signals:
        print(f"主要信号: {signals[0]['type']} (信心度{signals[0]['confidence']:.0%})")
    
    # 演示结构变化检测
    print("\n6. 结构变化检测演示...")
    
    # 模拟结构变化（添加另一组数据）
    print("   模拟市场结构变化...")
    
    # 创建区间市场数据
    n_range_bars = 100
    range_prices = 120 + np.sin(np.arange(n_range_bars) * 0.2) * 3
    range_noise = np.random.normal(0, 0.5, n_range_bars)
    range_data = pd.DataFrame({
        'open': range_prices * 0.998,
        'high': range_prices * 1.005,
        'low': range_prices * 0.995,
        'close': range_prices + range_noise,
        'volume': np.random.randint(800, 12000, n_range_bars)
    })
    
    range_result = identifier.identify_market_structure(range_data)
    print(f"   区间市场结构: {range_result['structure_type']['type']}")
    print(f"   结构强度: {range_result['current_strength']:.1%}")
    
    print("\n=== 系统演示完成 ===")
    print("第18章市场结构识别系统已实现并测试。")


# ============================================================================
# 策略改造: 添加MarketStructureIdentifierStrategy类
# 将市场结构分析器转换为交易策略
# ============================================================================

class MarketStructureIdentifierStrategy(BaseStrategy):
    """市场结构识别策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        lookback_period = params.get('lookback_period', 100)
        swing_sensitivity = params.get('swing_sensitivity', 2.0)
        structure_confirmation_bars = params.get('structure_confirmation_bars', 3)
        
        # 创建市场结构识别器实例
        self.identifier = MarketStructureIdentifier(
            lookback_period=lookback_period,
            swing_sensitivity=swing_sensitivity,
            structure_confirmation_bars=structure_confirmation_bars
        )
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于市场结构分析生成交易信号
        """
        # 调用市场结构识别
        analysis_result = self.identifier.identify_market_structure(self.data)
        
        # 获取结构信号
        structure_signals = analysis_result.get('structure_signals', [])
        
        # 将结构信号转换为BaseStrategy格式
        for signal in structure_signals:
            action = signal.get('type', 'hold').lower()
            if action in ['buy', 'sell']:
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action=action,
                    price=signal.get('entry_price', self.data['close'].iloc[-1])
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

if __name__ == "__main__":
    main()