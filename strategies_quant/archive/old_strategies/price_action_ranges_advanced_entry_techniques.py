#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级入场技术量化系统
第19章：高级入场技术
AL Brooks《价格行为交易之区间篇》

核心概念（极简实现）：
1. 回调入场：趋势中的回调买入/卖出点
2. 突破入场：价格突破关键水平的入场
3. 二次入场：错过第一次机会后的第二次入场
4. 入场条件验证：多重条件过滤
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class AdvancedEntryTechniques:
    """高级入场技术系统（最小可行版本）"""
    
    def __init__(self):
        """初始化入场技术分析器"""
        self.entry_history = []
    
    def analyze_entry_setups(self, 
                           price_data: pd.DataFrame,
                           structure_type: str = 'unknown') -> Dict:
        """
        分析入场设置
        
        参数:
            price_data: 价格数据
            structure_type: 市场结构类型
            
        返回:
            入场设置分析结果
        """
        if len(price_data) < 20:
            return {'error': '数据不足'}
        
        # 提取最近数据
        recent_data = price_data.tail(50).copy()
        current_price = recent_data['close'].iloc[-1]
        
        # 分析关键价格水平
        key_levels = self._identify_key_levels(recent_data)
        
        # 根据市场结构分析入场设置
        if structure_type == 'uptrend':
            setups = self._analyze_uptrend_setups(recent_data, key_levels)
        elif structure_type == 'downtrend':
            setups = self._analyze_downtrend_setups(recent_data, key_levels)
        elif structure_type == 'range':
            setups = self._analyze_range_setups(recent_data, key_levels)
        else:
            setups = self._analyze_general_setups(recent_data, key_levels)
        
        # 计算最优入场点
        optimal_entry = self._calculate_optimal_entry(setups, current_price)
        
        # 验证入场条件
        validation = self._validate_entry_conditions(optimal_entry, recent_data)
        
        # 生成入场信号
        signals = self._generate_entry_signals(optimal_entry, validation)
        
        result = {
            'current_price': current_price,
            'key_levels': key_levels,
            'entry_setups': setups,
            'optimal_entry': optimal_entry,
            'validation': validation,
            'signals': signals,
            'structure_type': structure_type,
            'analysis_time': pd.Timestamp.now()
        }
        
        self.entry_history.append(result)
        return result
    
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict:
        """识别关键价格水平"""
        # 简单实现：近期高、低、中点
        recent_high = data['high'].tail(20).max()
        recent_low = data['low'].tail(20).min()
        recent_close = data['close'].iloc[-1]
        
        # 支撑阻力水平
        pivot = (recent_high + recent_low + recent_close) / 3
        r1 = 2 * pivot - recent_low
        s1 = 2 * pivot - recent_high
        
        return {
            'recent_high': recent_high,
            'recent_low': recent_low,
            'pivot': pivot,
            'resistance_1': r1,
            'support_1': s1,
            'range_mid': (recent_high + recent_low) / 2,
            'current_vs_high': (recent_close - recent_high) / (recent_high - recent_low) if recent_high != recent_low else 0
        }
    
    def _analyze_uptrend_setups(self, data: pd.DataFrame, levels: Dict) -> List[Dict]:
        """分析上升趋势入场设置"""
        current_price = data['close'].iloc[-1]
        setups = []
        
        # 1. 回调买入：价格回调到支撑位
        if current_price <= levels['support_1'] * 1.01:
            setups.append({
                'type': 'pullback_buy',
                'entry_price': levels['support_1'],
                'reason': '价格回调至支撑位',
                'confidence': 0.7,
                'stop_loss': levels['support_1'] * 0.99,
                'take_profit': levels['recent_high'] * 0.995
            })
        
        # 2. 突破回调：突破后回踩确认
        if len(data) >= 10:
            prev_high = data['high'].iloc[-10:-1].max()
            if current_price > prev_high and current_price <= prev_high * 1.005:
                setups.append({
                    'type': 'breakout_retest',
                    'entry_price': prev_high,
                    'reason': '突破后回踩前高',
                    'confidence': 0.75,
                    'stop_loss': prev_high * 0.995,
                    'take_profit': prev_high * 1.03
                })
        
        return setups
    
    def _analyze_downtrend_setups(self, data: pd.DataFrame, levels: Dict) -> List[Dict]:
        """分析下降趋势入场设置"""
        current_price = data['close'].iloc[-1]
        setups = []
        
        # 1. 反弹卖出：价格反弹到阻力位
        if current_price >= levels['resistance_1'] * 0.99:
            setups.append({
                'type': 'rally_sell',
                'entry_price': levels['resistance_1'],
                'reason': '价格反弹至阻力位',
                'confidence': 0.7,
                'stop_loss': levels['resistance_1'] * 1.01,
                'take_profit': levels['recent_low'] * 1.005
            })
        
        # 2. 破位反弹：跌破后反弹确认
        if len(data) >= 10:
            prev_low = data['low'].iloc[-10:-1].min()
            if current_price < prev_low and current_price >= prev_low * 0.995:
                setups.append({
                    'type': 'breakdown_rally',
                    'entry_price': prev_low,
                    'reason': '跌破后反弹至前低',
                    'confidence': 0.75,
                    'stop_loss': prev_low * 1.005,
                    'take_profit': prev_low * 0.97
                })
        
        return setups
    
    def _analyze_range_setups(self, data: pd.DataFrame, levels: Dict) -> List[Dict]:
        """分析区间市场入场设置"""
        current_price = data['close'].iloc[-1]
        range_high = levels['recent_high']
        range_low = levels['recent_low']
        range_mid = levels['range_mid']
        
        setups = []
        
        # 1. 区间下轨买入
        if current_price <= range_low * 1.01:
            setups.append({
                'type': 'range_buy',
                'entry_price': range_low,
                'reason': '价格触及区间下轨',
                'confidence': 0.65,
                'stop_loss': range_low * 0.995,
                'take_profit': range_mid * 1.01
            })
        
        # 2. 区间上轨卖出
        elif current_price >= range_high * 0.99:
            setups.append({
                'type': 'range_sell',
                'entry_price': range_high,
                'reason': '价格触及区间上轨',
                'confidence': 0.65,
                'stop_loss': range_high * 1.005,
                'take_profit': range_mid * 0.99
            })
        
        # 3. 假突破后反向入场
        if len(data) >= 5:
            # 检查是否假突破上轨
            if data['high'].iloc[-5] > range_high and current_price < range_high:
                setups.append({
                    'type': 'false_breakout_sell',
                    'entry_price': range_high,
                    'reason': '假突破上轨后回落',
                    'confidence': 0.7,
                    'stop_loss': data['high'].iloc[-5] * 1.005,
                    'take_profit': range_mid * 0.99
                })
            
            # 检查是否假突破下轨
            if data['low'].iloc[-5] < range_low and current_price > range_low:
                setups.append({
                    'type': 'false_breakout_buy',
                    'entry_price': range_low,
                    'reason': '假突破下轨后反弹',
                    'confidence': 0.7,
                    'stop_loss': data['low'].iloc[-5] * 0.995,
                    'take_profit': range_mid * 1.01
                })
        
        return setups
    
    def _analyze_general_setups(self, data: pd.DataFrame, levels: Dict) -> List[Dict]:
        """通用入场设置分析"""
        # 简化的通用入场逻辑
        current_price = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2] if len(data) >= 2 else current_price
        
        setups = []
        
        # 基于价格位置的基本入场
        position_in_range = levels['current_vs_high']
        
        if position_in_range < 0.3:  # 在区间下半部
            setups.append({
                'type': 'general_buy',
                'entry_price': current_price,
                'reason': '价格在区间下半部',
                'confidence': 0.6,
                'stop_loss': current_price * 0.99,
                'take_profit': current_price * 1.02
            })
        elif position_in_range > 0.7:  # 在区间上半部
            setups.append({
                'type': 'general_sell',
                'entry_price': current_price,
                'reason': '价格在区间上半部',
                'confidence': 0.6,
                'stop_loss': current_price * 1.01,
                'take_profit': current_price * 0.98
            })
        
        return setups
    
    def _calculate_optimal_entry(self, setups: List[Dict], current_price: float) -> Dict:
        """计算最优入场点"""
        if not setups:
            return {
                'optimal_entry': None,
                'reason': '无合适入场设置',
                'confidence': 0,
                'recommended_action': '观望'
            }
        
        # 选择信心度最高的设置
        best_setup = max(setups, key=lambda x: x['confidence'])
        
        # 计算入场偏差（当前价格与建议入场价的差异）
        entry_deviation = abs(current_price - best_setup['entry_price']) / current_price
        
        # 调整信心度基于偏差
        adjusted_confidence = best_setup['confidence'] * max(0.5, 1 - entry_deviation * 10)
        
        return {
            'optimal_entry': best_setup,
            'entry_type': best_setup['type'],
            'suggested_price': best_setup['entry_price'],
            'current_price': current_price,
            'deviation_pct': entry_deviation * 100,
            'confidence': adjusted_confidence,
            'stop_loss': best_setup['stop_loss'],
            'take_profit': best_setup['take_profit'],
            'risk_reward_ratio': abs(best_setup['take_profit'] - best_setup['entry_price']) / 
                                 abs(best_setup['entry_price'] - best_setup['stop_loss'])
        }
    
    def _validate_entry_conditions(self, optimal_entry: Dict, data: pd.DataFrame) -> Dict:
        """验证入场条件"""
        if not optimal_entry.get('optimal_entry'):
            return {'valid': False, 'issues': ['无入场设置'], 'overall_score': 0}
        
        validation_checks = []
        scores = []
        
        # 检查1：风险回报比
        rr_ratio = optimal_entry.get('risk_reward_ratio', 0)
        if rr_ratio >= 2:
            validation_checks.append('✅ 风险回报比良好 (≥2:1)')
            scores.append(0.9)
        elif rr_ratio >= 1:
            validation_checks.append('⚠️ 风险回报比一般 (≥1:1)')
            scores.append(0.6)
        else:
            validation_checks.append('❌ 风险回报比不足 (<1:1)')
            scores.append(0.3)
        
        # 检查2：入场偏差
        deviation = optimal_entry.get('deviation_pct', 100)
        if deviation <= 0.5:
            validation_checks.append('✅ 入场偏差小 (≤0.5%)')
            scores.append(0.9)
        elif deviation <= 1.0:
            validation_checks.append('⚠️ 入场偏差适中 (≤1.0%)')
            scores.append(0.7)
        else:
            validation_checks.append('❌ 入场偏差大 (>1.0%)')
            scores.append(0.4)
        
        # 检查3：市场波动性
        if len(data) >= 10:
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            if volatility <= 0.2:
                validation_checks.append('✅ 市场波动性适中')
                scores.append(0.8)
            elif volatility <= 0.3:
                validation_checks.append('⚠️ 市场波动性较高')
                scores.append(0.6)
            else:
                validation_checks.append('❌ 市场波动性过高')
                scores.append(0.4)
        
        # 计算总体验证分数
        overall_score = np.mean(scores) if scores else 0.5
        
        return {
            'valid': overall_score >= 0.6,
            'validation_checks': validation_checks,
            'overall_score': overall_score,
            'recommendation': '入场条件满足' if overall_score >= 0.6 else '入场条件不足'
        }
    
    def _generate_entry_signals(self, optimal_entry: Dict, validation: Dict) -> List[Dict]:
        """生成入场信号"""
        if not optimal_entry.get('optimal_entry') or not validation['valid']:
            return []
        
        entry = optimal_entry['optimal_entry']
        confidence = optimal_entry['confidence'] * validation['overall_score']
        
        signal_type = 'buy' if 'buy' in entry['type'] else 'sell'
        
        return [{
            'type': signal_type,
            'entry_price': entry['entry_price'],
            'current_price': optimal_entry['current_price'],
            'stop_loss': entry['stop_loss'],
            'take_profit': entry['take_profit'],
            'confidence': confidence,
            'reason': entry['reason'],
            'entry_type': entry['type'],
            'risk_reward_ratio': optimal_entry.get('risk_reward_ratio', 0),
            'validation_score': validation['overall_score'],
            'recommended_position': 'normal' if confidence >= 0.6 else 'reduced'
        }]
    
    def generate_entry_report(self, analysis_result: Dict) -> str:
        """生成入场分析报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("高级入场技术分析报告")
        report_lines.append("=" * 60)
        
        report_lines.append(f"\n📊 市场结构: {analysis_result['structure_type']}")
        report_lines.append(f"当前价格: ${analysis_result['current_price']:.2f}")
        
        key_levels = analysis_result['key_levels']
        report_lines.append(f"\n🎯 关键价格水平:")
        report_lines.append(f"   近期高点: ${key_levels['recent_high']:.2f}")
        report_lines.append(f"   近期低点: ${key_levels['recent_low']:.2f}")
        report_lines.append(f"   枢轴点: ${key_levels['pivot']:.2f}")
        report_lines.append(f"   阻力1: ${key_levels['resistance_1']:.2f}")
        report_lines.append(f"   支撑1: ${key_levels['support_1']:.2f}")
        
        setups = analysis_result['entry_setups']
        report_lines.append(f"\n🔍 入场设置 ({len(setups)}个):")
        for i, setup in enumerate(setups[:3], 1):  # 只显示前3个
            report_lines.append(f"   {i}. {setup['type']}: {setup['reason']}")
            report_lines.append(f"      入场: ${setup['entry_price']:.2f} | 信心度: {setup['confidence']:.0%}")
        
        optimal = analysis_result['optimal_entry']
        report_lines.append(f"\n⭐ 最优入场点:")
        if optimal.get('optimal_entry'):
            report_lines.append(f"   类型: {optimal['entry_type']}")
            report_lines.append(f"   建议入场: ${optimal['suggested_price']:.2f}")
            report_lines.append(f"   当前价格: ${optimal['current_price']:.2f}")
            report_lines.append(f"   偏差: {optimal['deviation_pct']:.2f}%")
            report_lines.append(f"   信心度: {optimal['confidence']:.1%}")
            report_lines.append(f"   止损: ${optimal['stop_loss']:.2f}")
            report_lines.append(f"   止盈: ${optimal['take_profit']:.2f}")
            report_lines.append(f"   风险回报比: {optimal['risk_reward_ratio']:.2f}:1")
        else:
            report_lines.append(f"   无推荐入场点")
        
        validation = analysis_result['validation']
        report_lines.append(f"\n✅ 入场条件验证:")
        report_lines.append(f"   总体分数: {validation['overall_score']:.1%}")
        report_lines.append(f"   是否有效: {'是' if validation['valid'] else '否'}")
        report_lines.append(f"   建议: {validation['recommendation']}")
        for check in validation['validation_checks']:
            report_lines.append(f"   {check}")
        
        signals = analysis_result['signals']
        report_lines.append(f"\n🚦 入场信号:")
        if signals:
            signal = signals[0]
            action = "买入" if signal['type'] == 'buy' else "卖出"
            report_lines.append(f"   {action} @ ${signal['entry_price']:.2f}")
            report_lines.append(f"   信心度: {signal['confidence']:.1%}")
            report_lines.append(f"   理由: {signal['reason']}")
            report_lines.append(f"   验证分数: {signal['validation_score']:.1%}")
            report_lines.append(f"   推荐仓位: {signal['recommended_position']}")
        else:
            report_lines.append(f"   无推荐入场信号")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)

def generate_sample_price_data(n_bars: int = 100) -> pd.DataFrame:
    """生成样本价格数据"""
    np.random.seed(42)
    
    # 生成区间震荡数据
    time = np.arange(n_bars)
    base = 100 + np.sin(time * 0.1) * 5
    noise = np.random.normal(0, 0.5, n_bars)
    
    prices = base + noise
    
    df = pd.DataFrame({
        'open': prices * 0.998,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='D'))
    
    return df

def main():
    """主函数：演示高级入场技术系统"""
    print("=== 高级入场技术量化系统 ===")
    print("第19章：高级入场技术（最小可行版本）\n")
    
    # 生成样本数据
    print("1. 生成样本价格数据...")
    price_data = generate_sample_price_data(100)
    print(f"   数据形状: {price_data.shape}")
    print(f"   价格范围: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
    
    # 创建入场技术分析器
    print("\n2. 创建高级入场技术分析器...")
    entry_analyzer = AdvancedEntryTechniques()
    
    # 分析入场设置（假设为区间市场）
    print("\n3. 分析入场设置（区间市场）...")
    analysis_result = entry_analyzer.analyze_entry_setups(price_data, structure_type='range')
    
    # 生成分析报告
    print("\n4. 生成分析报告...")
    report = entry_analyzer.generate_entry_report(analysis_result)
    print(report)
    
    # 显示关键结果
    print("\n5. 关键结果摘要:")
    print("-" * 40)
    
    optimal = analysis_result['optimal_entry']
    if optimal.get('optimal_entry'):
        print(f"推荐入场: {optimal['entry_type']}")
        print(f"建议价格: ${optimal['suggested_price']:.2f}")
        print(f"当前价格: ${optimal['current_price']:.2f}")
        print(f"信心度: {optimal['confidence']:.1%}")
        print(f"风险回报比: {optimal['risk_reward_ratio']:.2f}:1")
    else:
        print("无推荐入场点")
    
    validation = analysis_result['validation']
    print(f"\n入场验证: {'通过' if validation['valid'] else '未通过'}")
    print(f"验证分数: {validation['overall_score']:.1%}")
    
    signals = analysis_result['signals']
    if signals:
        print(f"\n生成信号: {signals[0]['type'].upper()}")
        print(f"信号信心度: {signals[0]['confidence']:.1%}")
    else:
        print("\n无入场信号")
    
    print("\n=== 系统演示完成 ===")
    print("第19章高级入场技术系统（最小可行版本）已实现并测试。")


# ============================================================================
# 策略改造: 添加PriceActionRangesAdvancedEntryTechniquesStrategy类
# 将价格行为区间高级入场技术系统转换为交易策略
# ============================================================================

class PriceActionRangesAdvancedEntryTechniquesStrategy(BaseStrategy):
    """价格行为区间高级入场技术策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 创建高级入场技术分析器实例
        self.entry_analyzer = AdvancedEntryTechniques()
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于高级入场技术分析生成交易信号
        """
        # 分析入场设置
        analysis_result = self.entry_analyzer.analyze_entry_setups(self.data, structure_type='range')
        
        # 获取最优入场
        optimal_entry = analysis_result.get('optimal_entry', {})
        has_optimal = optimal_entry.get('optimal_entry', False)
        
        # 获取验证结果
        validation = analysis_result.get('validation', {})
        is_valid = validation.get('valid', False)
        
        # 获取信号
        signals = analysis_result.get('signals', [])
        
        # 根据分析生成信号
        if has_optimal and is_valid and signals:
            # 有最优入场且验证通过
            first_signal = signals[0]
            signal_type = first_signal.get('type', '').lower()
            
            if signal_type == 'buy':
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='buy',
                    price=self.data['close'].iloc[-1]
                )
            elif signal_type == 'sell':
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='sell',
                    price=self.data['close'].iloc[-1]
                )
            else:
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='hold',
                    price=self.data['close'].iloc[-1]
                )
        else:
            # 无有效入场信号，hold
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