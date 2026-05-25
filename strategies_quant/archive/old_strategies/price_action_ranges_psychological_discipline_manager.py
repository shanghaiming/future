# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.630437

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心理纪律管理量化系统
第22章：心理纪律管理
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 情绪检测：贪婪、恐惧、希望、后悔等交易情绪
2. 纪律规则：入场纪律、出场纪律、仓位纪律、风险纪律
3. 执行监控：计划vs执行、偏差检测、绩效评估
4. 偏差纠正：心理干预、规则强化、行为修正
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class PsychologicalDisciplineManager:
    """心理纪律管理器（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 trader_profile: Dict = None):
        """
        初始化心理纪律管理器
        
        参数:
            trader_profile: 交易者个人资料（经验水平、性格特征等）
        """
        self.trader_profile = trader_profile or {
            'experience_level': 'intermediate',  # beginner, intermediate, advanced
            'risk_tolerance': 'moderate',  # conservative, moderate, aggressive
            'trading_style': 'swing',  # day, swing, position
            'personality_traits': {}  # 可添加具体性格特征
        }
        
        self.discipline_history = []
        self.emotion_logs = []
        self.rule_violations = []
        self.performance_records = []
        
        # 初始化纪律规则
        self.discipline_rules = self._initialize_discipline_rules()
    
    def analyze_trading_psychology(self,
                                 trade_data: pd.DataFrame,
                                 market_conditions: Dict,
                                 trader_state: Dict) -> Dict:
        """
        分析交易心理状态
        
        参数:
            trade_data: 交易数据（近期交易记录）
            market_conditions: 市场条件
            trader_state: 交易者当前状态（压力水平、疲劳度等）
            
        返回:
            心理分析结果
        """
        if trade_data.empty:
            return {'error': '交易数据为空'}
        
        # 检测情绪状态
        emotional_state = self._detect_emotional_state(trade_data, market_conditions, trader_state)
        
        # 评估纪律遵守情况
        discipline_assessment = self._assess_discipline_compliance(trade_data, emotional_state)
        
        # 识别心理偏差
        psychological_biases = self._identify_psychological_biases(trade_data, emotional_state)
        
        # 评估执行质量
        execution_quality = self._evaluate_execution_quality(trade_data, discipline_assessment)
        
        # 生成心理干预建议
        intervention_suggestions = self._generate_intervention_suggestions(
            emotional_state, discipline_assessment, psychological_biases, execution_quality
        )
        
        result = {
            'timestamp': datetime.now(),
            'trader_profile': self.trader_profile,
            'trader_state': trader_state,
            'market_conditions': market_conditions,
            'trade_data_summary': self._summarize_trade_data(trade_data),
            'emotional_state': emotional_state,
            'discipline_assessment': discipline_assessment,
            'psychological_biases': psychological_biases,
            'execution_quality': execution_quality,
            'intervention_suggestions': intervention_suggestions,
            'overall_psychology_score': self._calculate_overall_psychology_score(
                emotional_state, discipline_assessment, execution_quality
            ),
            'risk_warnings': self._generate_risk_warnings(
                emotional_state, discipline_assessment, psychological_biases
            )
        }
        
        self.discipline_history.append(result)
        return result
    
    def _initialize_discipline_rules(self) -> Dict:
        """初始化纪律规则"""
        rules = {
            'entry_discipline': {
                'wait_for_confirmation': True,
                'max_entries_per_day': 5,
                'min_confidence_threshold': 0.6,
                'required_risk_reward_ratio': 2.0
            },
            'exit_discipline': {
                'follow_stop_loss': True,
                'follow_take_profit': True,
                'no_early_exit': True,
                'no_revenge_trading': True
            },
            'position_discipline': {
                'max_position_size': 0.1,  # 最大仓位比例
                'max_consecutive_losses': 3,
                'daily_loss_limit': 0.05,  # 单日最大亏损比例
                'weekly_loss_limit': 0.15
            },
            'risk_discipline': {
                'max_risk_per_trade': 0.02,
                'max_drawdown_limit': 0.2,
                'risk_adjustment_enabled': True
            },
            'psychological_discipline': {
                'emotional_threshold': 0.7,  # 情绪阈值
                'break_after_consecutive_losses': 2,
                'mandatory_break_after_big_win': True,
                'journal_required': True
            }
        }
        
        # 根据交易者资料调整规则
        if self.trader_profile['experience_level'] == 'beginner':
            rules['entry_discipline']['max_entries_per_day'] = 3
            rules['position_discipline']['max_position_size'] = 0.05
            rules['risk_discipline']['max_risk_per_trade'] = 0.01
        
        elif self.trader_profile['experience_level'] == 'advanced':
            rules['entry_discipline']['max_entries_per_day'] = 10
            rules['position_discipline']['max_position_size'] = 0.15
        
        if self.trader_profile['risk_tolerance'] == 'conservative':
            rules['risk_discipline']['max_risk_per_trade'] = 0.01
            rules['position_discipline']['max_position_size'] = 0.05
        
        elif self.trader_profile['risk_tolerance'] == 'aggressive':
            rules['risk_discipline']['max_risk_per_trade'] = 0.03
            rules['position_discipline']['max_position_size'] = 0.2
        
        return rules
    
    def _detect_emotional_state(self,
                               trade_data: pd.DataFrame,
                               market_conditions: Dict,
                               trader_state: Dict) -> Dict:
        """检测情绪状态"""
        emotions = {
            'greed': 0.0,
            'fear': 0.0,
            'hope': 0.0,
            'regret': 0.0,
            'overconfidence': 0.0,
            'frustration': 0.0,
            'calm': 0.0
        }
        
        # 分析近期交易表现
        if not trade_data.empty:
            recent_trades = trade_data.tail(10)
            
            # 贪婪检测：连续盈利后增加仓位或减少止损
            if len(recent_trades) >= 3:
                winning_streak = sum(1 for _, trade in recent_trades.iterrows() 
                                   if trade.get('pnl', 0) > 0)
                if winning_streak >= 3:
                    emotions['greed'] = min(1.0, winning_streak * 0.2)
                    emotions['overconfidence'] = min(1.0, winning_streak * 0.15)
            
            # 恐惧检测：连续亏损后减小仓位或过早出场
            losing_streak = sum(1 for _, trade in recent_trades.iterrows() 
                              if trade.get('pnl', 0) < 0)
            if losing_streak >= 2:
                emotions['fear'] = min(1.0, losing_streak * 0.25)
                emotions['frustration'] = min(1.0, losing_streak * 0.2)
            
            # 后悔检测：错过机会或错误出场
            missed_opportunities = sum(1 for _, trade in recent_trades.iterrows() 
                                     if trade.get('missed_opportunity', False))
            if missed_opportunities > 0:
                emotions['regret'] = min(1.0, missed_opportunities * 0.3)
            
            # 希望检测：持仓亏损但期待反转
            open_losses = sum(1 for _, trade in recent_trades.iterrows() 
                            if trade.get('status', '') == 'open' and 
                               trade.get('current_pnl', 0) < 0)
            if open_losses > 0:
                emotions['hope'] = min(1.0, open_losses * 0.2)
        
        # 考虑市场条件
        market_volatility = market_conditions.get('volatility', 0.01)
        if market_volatility > 0.02:
            emotions['fear'] = min(1.0, emotions['fear'] + 0.2)
        
        # 考虑交易者状态
        stress_level = trader_state.get('stress_level', 0.5)
        fatigue_level = trader_state.get('fatigue_level', 0.5)
        
        emotions['fear'] = min(1.0, emotions['fear'] + stress_level * 0.3)
        emotions['frustration'] = min(1.0, emotions['frustration'] + fatigue_level * 0.2)
        
        # 计算冷静程度（与其他情绪相反）
        negative_emotions = emotions['greed'] + emotions['fear'] + emotions['frustration']
        emotions['calm'] = max(0.0, 1.0 - negative_emotions)
        
        # 确定主要情绪
        max_emotion = max(emotions.items(), key=lambda x: x[1])
        dominant_emotion = max_emotion[0] if max_emotion[1] > 0.3 else 'calm'
        
        return {
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'dominant_intensity': max_emotion[1],
            'emotional_stability': emotions['calm'],
            'risk_level': self._calculate_emotional_risk_level(emotions, dominant_emotion)
        }
    
    def _calculate_emotional_risk_level(self, emotions: Dict, dominant_emotion: str) -> str:
        """计算情绪风险等级"""
        high_risk_emotions = ['greed', 'fear', 'overconfidence', 'frustration']
        
        if dominant_emotion in high_risk_emotions:
            intensity = emotions[dominant_emotion]
            if intensity > 0.7:
                return 'extreme_risk'
            elif intensity > 0.5:
                return 'high_risk'
            elif intensity > 0.3:
                return 'moderate_risk'
        
        return 'low_risk'
    
    def _assess_discipline_compliance(self,
                                     trade_data: pd.DataFrame,
                                     emotional_state: Dict) -> Dict:
        """评估纪律遵守情况"""
        compliance = {
            'entry_discipline': {'score': 0.8, 'violations': []},
            'exit_discipline': {'score': 0.7, 'violations': []},
            'position_discipline': {'score': 0.9, 'violations': []},
            'risk_discipline': {'score': 0.85, 'violations': []},
            'psychological_discipline': {'score': 0.6, 'violations': []}
        }
        
        if trade_data.empty:
            return compliance
        
        recent_trades = trade_data.tail(20)
        
        # 检查入场纪律
        entry_rules = self.discipline_rules['entry_discipline']
        entry_violations = []
        
        # 检查每日交易次数
        if 'entry_time' in trade_data.columns:
            trade_dates = pd.to_datetime(trade_data['entry_time']).dt.date
            trades_per_day = trade_dates.value_counts()
            if trades_per_day.max() > entry_rules['max_entries_per_day']:
                entry_violations.append(f"单日交易次数超过限制 ({trades_per_day.max()} > {entry_rules['max_entries_per_day']})")
                compliance['entry_discipline']['score'] *= 0.8
        
        # 检查出场纪律
        exit_rules = self.discipline_rules['exit_discipline']
        exit_violations = []
        
        # 检查是否遵循止损
        stopped_out_trades = recent_trades[recent_trades['exit_reason'] == 'stop_loss']
        if len(stopped_out_trades) < len(recent_trades[recent_trades['pnl'] < 0]):
            exit_violations.append("部分亏损交易未触发止损")
            compliance['exit_discipline']['score'] *= 0.7
        
        # 检查报复性交易
        if len(recent_trades) >= 2:
            for i in range(1, len(recent_trades)):
                prev_trade = recent_trades.iloc[i-1]
                current_trade = recent_trades.iloc[i]
                
                time_diff = (current_trade['entry_time'] - prev_trade['exit_time']).total_seconds() / 60
                if prev_trade['pnl'] < 0 and time_diff < 30:
                    exit_violations.append("疑似报复性交易（亏损后30分钟内再次入场）")
                    compliance['exit_discipline']['score'] *= 0.6
        
        compliance['entry_discipline']['violations'] = entry_violations
        compliance['exit_discipline']['violations'] = exit_violations
        
        # 计算总体纪律分数
        total_score = sum(c['score'] for c in compliance.values()) / len(compliance)
        compliance['overall_score'] = total_score
        compliance['overall_rating'] = self._get_discipline_rating(total_score)
        
        return compliance
    
    def _get_discipline_rating(self, score: float) -> str:
        """获取纪律评级"""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.7:
            return 'fair'
        elif score >= 0.6:
            return 'needs_improvement'
        else:
            return 'poor'
    
    def _identify_psychological_biases(self,
                                      trade_data: pd.DataFrame,
                                      emotional_state: Dict) -> List[Dict]:
        """识别心理偏差"""
        biases = []
        
        if trade_data.empty:
            return biases
        
        recent_trades = trade_data.tail(15)
        
        # 确认偏差：只关注支持自己观点的信息
        if len(recent_trades) >= 5:
            confirmation_trades = recent_trades[recent_trades['bias_confirmation'] == True]
            if len(confirmation_trades) > len(recent_trades) * 0.7:
                biases.append({
                    'type': 'confirmation_bias',
                    'severity': 'high',
                    'description': '过度关注支持自己观点的信息，忽视反面证据',
                    'suggested_action': '主动寻找并考虑反面观点'
                })
        
        # 近期效应：过度重视近期事件
        if len(recent_trades) >= 10:
            recent_performance = recent_trades.tail(5)['pnl'].mean()
            overall_performance = recent_trades['pnl'].mean()
            
            if abs(recent_performance - overall_performance) > overall_performance * 0.5:
                biases.append({
                    'type': 'recency_bias',
                    'severity': 'medium',
                    'description': '过度重视近期交易表现，忽视长期趋势',
                    'suggested_action': '回顾更长时期的交易记录'
                })
        
        # 损失厌恶：对损失的厌恶强于对盈利的喜好
        winning_trades = recent_trades[recent_trades['pnl'] > 0]
        losing_trades = recent_trades[recent_trades['pnl'] < 0]
        
        if len(losing_trades) > 0 and len(winning_trades) > 0:
            avg_win = winning_trades['pnl'].mean()
            avg_loss = abs(losing_trades['pnl'].mean())
            
            if avg_loss > avg_win * 1.5:
                biases.append({
                    'type': 'loss_aversion',
                    'severity': 'medium',
                    'description': '对损失的厌恶导致过早止盈或过晚止损',
                    'suggested_action': '重新评估风险回报目标'
                })
        
        # 过度自信：高估自己的交易能力
        if emotional_state['dominant_emotion'] == 'overconfidence':
            biases.append({
                'type': 'overconfidence_bias',
                'severity': 'high',
                'description': '过度自信可能导致过度交易或忽视风险',
                'suggested_action': '回顾过去的错误交易，保持谦虚'
            })
        
        # 锚定效应：过度依赖某个价格水平
        if 'anchor_price' in recent_trades.columns:
            anchor_trades = recent_trades[recent_trades['anchor_influence'] == True]
            if len(anchor_trades) > len(recent_trades) * 0.4:
                biases.append({
                    'type': 'anchoring_bias',
                    'severity': 'medium',
                    'description': '过度依赖某个价格水平（如成本价、前期高点）',
                    'suggested_action': '基于当前市场条件而非历史价格做决策'
                })
        
        return biases
    
    def _evaluate_execution_quality(self,
                                   trade_data: pd.DataFrame,
                                   discipline_assessment: Dict) -> Dict:
        """评估执行质量"""
        quality = {
            'plan_vs_execution': 0.8,
            'entry_accuracy': 0.7,
            'exit_timing': 0.75,
            'risk_management': 0.85,
            'overall_quality': 0.78
        }
        
        if trade_data.empty:
            return quality
        
        recent_trades = trade_data.tail(10)
        
        # 计划vs执行
        if 'planned_vs_actual' in recent_trades.columns:
            match_rate = recent_trades['planned_vs_actual'].mean()
            quality['plan_vs_execution'] = match_rate
        
        # 入场准确性
        if 'entry_deviation' in recent_trades.columns:
            avg_deviation = recent_trades['entry_deviation'].abs().mean()
            quality['entry_accuracy'] = max(0.0, 1.0 - avg_deviation * 10)
        
        # 出场时机
        if 'exit_timing_score' in recent_trades.columns:
            quality['exit_timing'] = recent_trades['exit_timing_score'].mean()
        
        # 风险管理
        risk_scores = []
        for _, trade in recent_trades.iterrows():
            risk_score = 0.8
            if trade.get('risk_reward_ratio', 0) >= 2.0:
                risk_score += 0.1
            if trade.get('position_size_appropriate', True):
                risk_score += 0.05
            if trade.get('stop_loss_hit', False) == False:
                risk_score += 0.05
            risk_scores.append(min(1.0, risk_score))
        
        if risk_scores:
            quality['risk_management'] = np.mean(risk_scores)
        
        # 计算总体质量
        quality_values = [quality[k] for k in ['plan_vs_execution', 'entry_accuracy', 'exit_timing', 'risk_management']]
        quality['overall_quality'] = np.mean(quality_values)
        
        # 考虑纪律分数
        discipline_score = discipline_assessment.get('overall_score', 0.7)
        quality['overall_quality'] = quality['overall_quality'] * 0.7 + discipline_score * 0.3
        
        quality['quality_rating'] = self._get_quality_rating(quality['overall_quality'])
        
        return quality
    
    def _get_quality_rating(self, score: float) -> str:
        """获取质量评级"""
        if score >= 0.85:
            return 'excellent'
        elif score >= 0.75:
            return 'good'
        elif score >= 0.65:
            return 'fair'
        elif score >= 0.55:
            return 'needs_improvement'
        else:
            return 'poor'
    
    def _generate_intervention_suggestions(self,
                                          emotional_state: Dict,
                                          discipline_assessment: Dict,
                                          psychological_biases: List[Dict],
                                          execution_quality: Dict) -> List[Dict]:
        """生成心理干预建议"""
        suggestions = []
        
        # 基于情绪状态的建议
        dominant_emotion = emotional_state['dominant_emotion']
        emotional_risk = emotional_state['risk_level']
        
        if emotional_risk in ['high_risk', 'extreme_risk']:
            suggestions.append({
                'type': 'emotional_regulation',
                'priority': 'high',
                'suggestion': f"检测到{dominant_emotion}情绪较高（风险等级: {emotional_risk}），建议暂停交易，进行深呼吸或短暂休息",
                'action': 'take_break',
                'duration_minutes': 15
            })
        
        # 基于纪律遵守情况的建议
        if discipline_assessment['overall_score'] < 0.7:
            worst_discipline = min(discipline_assessment.items(), 
                                 key=lambda x: x[1]['score'] if isinstance(x[1], dict) and 'score' in x[1] else 1.0)
            
            suggestions.append({
                'type': 'discipline_improvement',
                'priority': 'medium',
                'suggestion': f"{worst_discipline[0]}纪律需要改进（分数: {worst_discipline[1]['score']:.2f}），建议回顾相关规则并制定改进计划",
                'action': 'review_rules',
                'focus_area': worst_discipline[0]
            })
        
        # 基于心理偏差的建议
        for bias in psychological_biases:
            if bias['severity'] in ['high', 'medium']:
                suggestions.append({
                    'type': 'bias_correction',
                    'priority': 'medium',
                    'suggestion': f"检测到{bias['type']}：{bias['description']}。建议：{bias['suggested_action']}",
                    'action': 'bias_awareness',
                    'bias_type': bias['type']
                })
        
        # 基于执行质量的建议
        if execution_quality['overall_quality'] < 0.7:
            worst_quality = min([k for k in execution_quality.keys() 
                               if k not in ['overall_quality', 'quality_rating']],
                              key=lambda x: execution_quality[x])
            
            suggestions.append({
                'type': 'execution_improvement',
                'priority': 'medium',
                'suggestion': f"{worst_quality}需要改进（分数: {execution_quality[worst_quality]:.2f}），建议进行模拟交易练习",
                'action': 'practice',
                'practice_type': worst_quality
            })
        
        # 通用建议
        if not suggestions:
            suggestions.append({
                'type': 'maintenance',
                'priority': 'low',
                'suggestion': "心理状态和纪律执行良好，继续保持。建议定期进行心理自我评估",
                'action': 'continue',
                'frequency': 'weekly'
            })
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order[x['priority']])
        
        return suggestions
    
    def _summarize_trade_data(self, trade_data: pd.DataFrame) -> Dict:
        """汇总交易数据"""
        if trade_data.empty:
            return {'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0}
        
        summary = {
            'total_trades': len(trade_data),
            'winning_trades': len(trade_data[trade_data['pnl'] > 0]),
            'losing_trades': len(trade_data[trade_data['pnl'] < 0]),
            'breakeven_trades': len(trade_data[trade_data['pnl'] == 0]),
            'total_pnl': trade_data['pnl'].sum(),
            'avg_pnl': trade_data['pnl'].mean(),
            'max_win': trade_data['pnl'].max(),
            'max_loss': trade_data['pnl'].min(),
            'win_rate': len(trade_data[trade_data['pnl'] > 0]) / len(trade_data) if len(trade_data) > 0 else 0,
            'profit_factor': abs(trade_data[trade_data['pnl'] > 0]['pnl'].sum() / 
                               trade_data[trade_data['pnl'] < 0]['pnl'].sum()) if trade_data[trade_data['pnl'] < 0]['pnl'].sum() != 0 else float('inf'),
            'recent_trend': self._calculate_recent_trend(trade_data)
        }
        
        return summary
    
    def _calculate_recent_trend(self, trade_data: pd.DataFrame) -> str:
        """计算近期趋势"""
        if len(trade_data) < 5:
            return 'insufficient_data'
        
        recent_trades = trade_data.tail(5)
        winning_trades = len(recent_trades[recent_trades['pnl'] > 0])
        
        if winning_trades >= 4:
            return 'strong_winning'
        elif winning_trades >= 3:
            return 'winning'
        elif winning_trades >= 2:
            return 'neutral'
        elif winning_trades >= 1:
            return 'losing'
        else:
            return 'strong_losing'
    
    def _calculate_overall_psychology_score(self,
                                           emotional_state: Dict,
                                           discipline_assessment: Dict,
                                           execution_quality: Dict) -> float:
        """计算总体心理分数"""
        emotional_score = emotional_state['emotional_stability'] * 0.4
        discipline_score = discipline_assessment['overall_score'] * 0.3
        execution_score = execution_quality['overall_quality'] * 0.3
        
        overall_score = emotional_score + discipline_score + execution_score
        
        return overall_score
    
    def _generate_risk_warnings(self,
                               emotional_state: Dict,
                               discipline_assessment: Dict,
                               psychological_biases: List[Dict]) -> List[str]:
        """生成风险警告"""
        warnings = []
        
        # 情绪风险警告
        if emotional_state['risk_level'] == 'extreme_risk':
            warnings.append(f"⚠️ 情绪风险极高：{emotional_state['dominant_emotion']}情绪强烈，建议立即停止交易")
        elif emotional_state['risk_level'] == 'high_risk':
            warnings.append(f"⚠️ 情绪风险高：{emotional_state['dominant_emotion']}情绪较强，建议谨慎交易")
        
        # 纪律风险警告
        if discipline_assessment['overall_score'] < 0.6:
            warnings.append(f"⚠️ 纪律执行差：总体纪律分数{discipline_assessment['overall_score']:.2f}，建议加强纪律遵守")
        
        # 心理偏差警告
        high_severity_biases = [b for b in psychological_biases if b['severity'] == 'high']
        if high_severity_biases:
            bias_names = ', '.join([b['type'] for b in high_severity_biases])
            warnings.append(f"⚠️ 严重心理偏差：检测到{len(high_severity_biases)}个高风险偏差（{bias_names}）")
        
        return warnings
    
    def log_emotion(self, emotion: str, intensity: float, context: str = ""):
        """记录情绪"""
        log_entry = {
            'timestamp': datetime.now(),
            'emotion': emotion,
            'intensity': intensity,
            'context': context,
            'trader_state': self.trader_profile
        }
        self.emotion_logs.append(log_entry)
    
    def log_rule_violation(self, rule_type: str, violation: str, severity: str = 'medium'):
        """记录规则违反"""
        violation_entry = {
            'timestamp': datetime.now(),
            'rule_type': rule_type,
            'violation': violation,
            'severity': severity,
            'corrective_action': self._suggest_corrective_action(rule_type, violation)
        }
        self.rule_violations.append(violation_entry)
    
    def _suggest_corrective_action(self, rule_type: str, violation: str) -> str:
        """建议纠正措施"""
        corrective_actions = {
            'entry_discipline': "重新评估交易计划，等待确认信号",
            'exit_discipline': "严格执行止损止盈，避免情绪化出场",
            'position_discipline': "调整仓位大小，遵守风险限制",
            'risk_discipline': "重新计算风险参数，降低风险暴露",
            'psychological_discipline': "进行心理调节，恢复冷静状态"
        }
        return corrective_actions.get(rule_type, "回顾相关纪律规则并制定改进计划")
    
    def get_psychology_report(self, days: int = 7) -> Dict:
        """获取心理报告"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_history = [h for h in self.discipline_history 
                         if h['timestamp'] >= cutoff_date]
        recent_emotions = [e for e in self.emotion_logs 
                          if e['timestamp'] >= cutoff_date]
        recent_violations = [v for v in self.rule_violations 
                            if v['timestamp'] >= cutoff_date]
        
        if not recent_history:
            return {'error': '指定时间段内无数据'}
        
        # 计算平均分数
        avg_psychology_score = np.mean([h['overall_psychology_score'] for h in recent_history])
        avg_discipline_score = np.mean([h['discipline_assessment']['overall_score'] for h in recent_history])
        avg_emotional_stability = np.mean([h['emotional_state']['emotional_stability'] for h in recent_history])
        
        # 分析情绪趋势
        emotion_counts = {}
        for emotion_log in recent_emotions:
            emotion = emotion_log['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 分析常见违反
        violation_counts = {}
        for violation in recent_violations:
            rule_type = violation['rule_type']
            violation_counts[rule_type] = violation_counts.get(rule_type, 0) + 1
        
        report = {
            'period_days': days,
            'analysis_count': len(recent_history),
            'emotion_log_count': len(recent_emotions),
            'violation_count': len(recent_violations),
            'avg_psychology_score': avg_psychology_score,
            'avg_discipline_score': avg_discipline_score,
            'avg_emotional_stability': avg_emotional_stability,
            'psychology_trend': self._assess_trend([h['overall_psychology_score'] for h in recent_history]),
            'dominant_emotions': emotion_counts,
            'common_violations': violation_counts,
            'improvement_areas': self._identify_improvement_areas(recent_history),
            'strengths': self._identify_strengths(recent_history),
            'recommendations': self._generate_long_term_recommendations(recent_history)
        }
        
        return report
    
    def _assess_trend(self, scores: List[float]) -> str:
        """评估趋势"""
        if len(scores) < 3:
            return 'insufficient_data'
        
        # 简单趋势分析
        recent_avg = np.mean(scores[-3:])
        earlier_avg = np.mean(scores[:-3]) if len(scores) > 3 else scores[0]
        
        if recent_avg > earlier_avg + 0.05:
            return 'improving'
        elif recent_avg < earlier_avg - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _identify_improvement_areas(self, history: List[Dict]) -> List[str]:
        """识别改进领域"""
        improvement_areas = []
        
        if not history:
            return improvement_areas
        
        # 分析最近几次评估
        recent_assessments = history[-5:] if len(history) >= 5 else history
        
        # 检查纪律遵守
        discipline_scores = [a['discipline_assessment']['overall_score'] for a in recent_assessments]
        avg_discipline_score = np.mean(discipline_scores)
        
        if avg_discipline_score < 0.7:
            improvement_areas.append('discipline_compliance')
        
        # 检查情绪稳定性
        emotional_stabilities = [a['emotional_state']['emotional_stability'] for a in recent_assessments]
        avg_emotional_stability = np.mean(emotional_stabilities)
        
        if avg_emotional_stability < 0.6:
            improvement_areas.append('emotional_control')
        
        # 检查执行质量
        execution_qualities = [a['execution_quality']['overall_quality'] for a in recent_assessments]
        avg_execution_quality = np.mean(execution_qualities)
        
        if avg_execution_quality < 0.7:
            improvement_areas.append('execution_quality')
        
        return improvement_areas
    
    def _identify_strengths(self, history: List[Dict]) -> List[str]:
        """识别优势"""
        strengths = []
        
        if not history:
            return strengths
        
        recent_assessments = history[-5:] if len(history) >= 5 else history
        
        # 检查优势领域
        discipline_scores = [a['discipline_assessment']['overall_score'] for a in recent_assessments]
        emotional_stabilities = [a['emotional_state']['emotional_stability'] for a in recent_assessments]
        execution_qualities = [a['execution_quality']['overall_quality'] for a in recent_assessments]
        
        if np.mean(discipline_scores) >= 0.8:
            strengths.append('strong_discipline')
        
        if np.mean(emotional_stabilities) >= 0.7:
            strengths.append('emotional_stability')
        
        if np.mean(execution_qualities) >= 0.8:
            strengths.append('execution_quality')
        
        # 检查是否有持续改进
        if len(history) >= 3:
            psychology_scores = [a['overall_psychology_score'] for a in history]
            if self._assess_trend(psychology_scores) == 'improving':
                strengths.append('continuous_improvement')
        
        return strengths
    
    def _generate_long_term_recommendations(self, history: List[Dict]) -> List[Dict]:
        """生成长期建议"""
        recommendations = []
        
        if not history or len(history) < 5:
            recommendations.append({
                'type': 'data_collection',
                'priority': 'medium',
                'suggestion': '收集更多交易心理数据以获得更准确的分析',
                'action': 'continue_monitoring',
                'timeline': '2_weeks'
            })
            return recommendations
        
        # 分析趋势
        psychology_trend = self._assess_trend([h['overall_psychology_score'] for h in history])
        improvement_areas = self._identify_improvement_areas(history)
        strengths = self._identify_strengths(history)
        
        # 基于趋势的建议
        if psychology_trend == 'declining':
            recommendations.append({
                'type': 'trend_correction',
                'priority': 'high',
                'suggestion': '心理状态呈下降趋势，建议进行全面心理评估并可能寻求专业帮助',
                'action': 'comprehensive_assessment',
                'timeline': 'immediate'
            })
        elif psychology_trend == 'improving':
            recommendations.append({
                'type': 'trend_maintenance',
                'priority': 'low',
                'suggestion': '心理状态呈改善趋势，继续保持当前做法',
                'action': 'maintain_current_practices',
                'timeline': 'ongoing'
            })
        
        # 基于改进领域的建议
        if 'discipline_compliance' in improvement_areas:
            recommendations.append({
                'type': 'discipline_training',
                'priority': 'medium',
                'suggestion': '纪律遵守需要改进，建议进行专门的纪律训练',
                'action': 'discipline_practice',
                'timeline': '1_month'
            })
        
        if 'emotional_control' in improvement_areas:
            recommendations.append({
                'type': 'emotional_training',
                'priority': 'medium',
                'suggestion': '情绪控制需要改进，建议学习情绪调节技巧',
                'action': 'emotional_regulation_training',
                'timeline': '2_weeks'
            })
        
        # 基于优势的建议
        if 'continuous_improvement' in strengths:
            recommendations.append({
                'type': 'optimization',
                'priority': 'low',
                'suggestion': '显示持续改进，可以开始优化高级心理技巧',
                'action': 'advanced_psychology_training',
                'timeline': '1_month'
            })
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def reset_trader_profile(self, new_profile: Dict):
        """重置交易者资料"""
        self.trader_profile = new_profile
        # 重新初始化纪律规则
        self.discipline_rules = self._initialize_discipline_rules()


# ============================================================================
# 策略改造: 添加PriceActionRangesPsychologicalDisciplineManagerStrategy类
# 将价格行为区间心理纪律管理系统转换为交易策略
# ============================================================================

class PriceActionRangesPsychologicalDisciplineManagerStrategy(BaseStrategy):
    """价格行为区间心理纪律管理策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 创建心理纪律管理器实例
        self.psych_manager = PsychologicalDisciplineManager()
    
    def generate_signals(self):
        """
        生成交易信号

        基于心理纪律分析生成交易信号，使用波动率和趋势稳定性评估市场心理
        """
        df = self.data
        if len(df) < 20:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(df['close'].iloc[-1]))
            return self.signals

        close = df['close']

        # Volatility regime (proxy for market emotion)
        returns = close.pct_change()
        vol = returns.rolling(20).std() * np.sqrt(252)
        vol_mean = vol.rolling(50).mean()

        # Trend consistency (proxy for discipline)
        ma_short = close.rolling(10).mean()
        ma_long = close.rolling(30).mean()
        trend_consistent = ((close > ma_short) & (ma_short > ma_long)).rolling(10).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        last_vol = vol.iloc[-1]
        last_vol_mean = vol_mean.iloc[-1]
        last_consistency = trend_consistent.iloc[-1] if not np.isnan(trend_consistent.iloc[-1]) else 0.5

        # Low volatility + high trend consistency = disciplined market = buy
        # High volatility = emotional market = cautious
        if last_vol < last_vol_mean * 0.8 and last_consistency > 0.7:
            self._record_signal(timestamp=df.index[-1], action='buy', price=float(close.iloc[-1]))
        elif last_vol > last_vol_mean * 1.5:
            self._record_signal(timestamp=df.index[-1], action='sell', price=float(close.iloc[-1]))
        elif last_consistency < 0.3:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(close.iloc[-1]))
        elif ma_short.iloc[-1] > ma_long.iloc[-1] and rsi.iloc[-1] < 70:
            self._record_signal(timestamp=df.index[-1], action='buy', price=float(close.iloc[-1]))
        else:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(close.iloc[-1]))

        return self.signals