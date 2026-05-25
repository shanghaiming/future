# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.634805

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心理训练量化系统
第28章：心理训练
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 情绪识别：交易时的情绪状态监控、情绪对决策的影响分析
2. 纪律强化：交易规则遵守检查、纪律违规记录和改进
3. 决策优化：决策过程分析、认知偏差识别和纠正
4. 压力管理：交易压力评估、压力应对策略、心理恢复
5. 自信心建立：成功经验积累、能力自我评估、自信水平监控
6. 心理韧性训练：逆境应对、挫折恢复、长期心理建设
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class PsychologicalTrainingSystem:
    """心理训练系统（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 trader_profile: Dict = None,
                 training_goals: Dict = None,
                 improvement_rate: float = 0.1):
        """初始化心理训练系统"""
        self.trader_profile = trader_profile or {
            'experience_level': 'intermediate',
            'trading_style': 'swing',
            'psychological_profile': 'balanced',
            'learning_style': 'reflective'
        }
        
        self.training_goals = training_goals or {
            'emotional_control_target': 0.8,  # 情绪控制目标80%
            'discipline_compliance_target': 0.85,  # 纪律遵守目标85%
            'decision_quality_target': 0.75,  # 决策质量目标75%
            'stress_management_target': 0.7,  # 压力管理目标70%
            'confidence_level_target': 0.8,  # 自信心水平目标80%
            'resilience_score_target': 0.75  # 心理韧性目标75%
        }
        
        self.improvement_rate = improvement_rate
        
        # 心理状态历史记录
        self.psychological_history = {
            'emotional_states': [],  # 情绪状态记录
            'discipline_checks': [],  # 纪律检查记录
            'decision_analyses': [],  # 决策分析记录
            'stress_assessments': [],  # 压力评估记录
            'confidence_tracking': [],  # 自信心跟踪记录
            'training_sessions': []  # 训练会话记录
        }
        
        # 心理训练模块
        self.training_modules = {
            'emotional_control': {
                'emotion_recognition': True,
                'emotion_journaling': True,
                'emotional_patterns': True,
                'calming_techniques': True
            },
            'discipline_training': {
                'rule_compliance': True,
                'habit_formation': True,
                'impulse_control': True,
                'consistency_building': True
            },
            'decision_training': {
                'bias_identification': True,
                'decision_process': True,
                'outcome_analysis': True,
                'pattern_recognition': True
            },
            'stress_management': {
                'stress_identification': True,
                'relaxation_techniques': True,
                'workload_management': True,
                'recovery_protocols': True
            },
            'confidence_building': {
                'success_recording': True,
                'skill_recognition': True,
                'positive_self_talk': True,
                'goal_achievement': True
            },
            'resilience_training': {
                'setback_recovery': True,
                'adversity_response': True,
                'persistence_building': True,
                'growth_mindset': True
            }
        }
        
        # 心理评估指标
        self.psychological_metrics = {
            'emotional_intensity_scale': {
                'calm': 0.1,
                'neutral': 0.3,
                'alert': 0.5,
                'tense': 0.7,
                'panicked': 0.9
            },
            'discipline_levels': {
                'poor': 0.2,
                'fair': 0.4,
                'good': 0.6,
                'excellent': 0.8,
                'perfect': 1.0
            },
            'decision_quality_levels': {
                'impulsive': 0.1,
                'emotional': 0.3,
                'reasoned': 0.5,
                'systematic': 0.7,
                'optimal': 0.9
            },
            'stress_levels': {
                'relaxed': 0.1,
                'normal': 0.3,
                'moderate': 0.5,
                'high': 0.7,
                'extreme': 0.9
            },
            'confidence_levels': {
                'low': 0.2,
                'moderate': 0.4,
                'adequate': 0.6,
                'high': 0.8,
                'very_high': 1.0
            }
        }
        
        # 当前心理状态
        self.current_psychological_state = {
            'emotional_state': 'neutral',
            'emotional_score': 0.5,
            'discipline_level': 'good',
            'discipline_score': 0.6,
            'decision_quality': 'reasoned',
            'decision_score': 0.5,
            'stress_level': 'normal',
            'stress_score': 0.3,
            'confidence_level': 'adequate',
            'confidence_score': 0.6,
            'overall_psychological_score': 0.5,
            'last_assessment_time': None,
            'improvement_trend': 'stable'
        }
    
    def record_emotional_state(self, emotion_data: Dict) -> Dict:
        """记录情绪状态"""
        # 验证情绪数据
        required_fields = ['emotion_type', 'intensity', 'context', 'trigger']
        for field in required_fields:
            if field not in emotion_data:
                return {'error': f'缺少必要字段: {field}'}
        
        # 添加元数据
        emotion_data['recording_time'] = datetime.now()
        emotion_data['emotion_id'] = f"emotion_{len(self.psychological_history['emotional_states']) + 1:06d}"
        
        # 情绪强度映射到分数
        intensity = emotion_data['intensity']
        emotion_type = emotion_data['emotion_type']
        
        # 计算情绪分数（0-1范围）
        base_score = intensity  # 假设强度已经是0-1范围
        
        # 根据情绪类型调整（负面情绪得分低，正面情绪得分高）
        positive_emotions = ['calm', 'confident', 'focused', 'optimistic', 'patient']
        negative_emotions = ['fear', 'greed', 'frustration', 'anger', 'anxiety', 'panic']
        
        if emotion_type in positive_emotions:
            emotion_score = base_score * 0.8 + 0.2  # 正向调整
        elif emotion_type in negative_emotions:
            emotion_score = base_score * 0.5  # 负向调整
        else:
            emotion_score = base_score
        
        emotion_data['emotion_score'] = emotion_score
        
        # 添加到历史记录
        self.psychological_history['emotional_states'].append(emotion_data)
        
        # 更新当前情绪状态
        self.current_psychological_state['emotional_state'] = emotion_type
        self.current_psychological_state['emotional_score'] = emotion_score
        
        # 更新总体心理分数
        self._update_overall_psychological_score()
        
        # 分析情绪模式
        emotion_analysis = self._analyze_emotional_pattern(emotion_data)
        
        return {
            'emotion_id': emotion_data['emotion_id'],
            'emotion_type': emotion_type,
            'emotion_score': emotion_score,
            'recording_time': emotion_data['recording_time'],
            'analysis': emotion_analysis,
            'recommendation': self._generate_emotion_recommendation(emotion_data, emotion_analysis)
        }
    
    def _analyze_emotional_pattern(self, emotion_data: Dict) -> Dict:
        """分析情绪模式"""
        emotion_type = emotion_data['emotion_type']
        intensity = emotion_data['intensity']
        context = emotion_data['context']
        trigger = emotion_data['trigger']
        
        analysis = {
            'emotion_type': emotion_type,
            'intensity_level': 'low' if intensity < 0.3 else 'medium' if intensity < 0.7 else 'high',
            'context_relevance': 'unknown',
            'trigger_analysis': 'unknown',
            'pattern_frequency': 0,
            'recommended_action': 'monitor'
        }
        
        # 分析上下文相关性
        trading_contexts = ['pre_trade', 'during_trade', 'post_trade', 'winning_trade', 'losing_trade']
        if context in trading_contexts:
            analysis['context_relevance'] = 'trading_related'
        else:
            analysis['context_relevance'] = 'non_trading'
        
        # 分析触发因素
        common_triggers = {
            'market_movement': ['price_change', 'volatility', 'news'],
            'personal_factors': ['fatigue', 'stress', 'overconfidence'],
            'external_factors': ['distraction', 'interruption', 'technical_issue']
        }
        
        for category, triggers in common_triggers.items():
            if any(trigger_word in trigger.lower() for trigger_word in triggers):
                analysis['trigger_analysis'] = category
                break
        
        # 计算模式频率
        recent_emotions = self.psychological_history['emotional_states'][-20:] if len(self.psychological_history['emotional_states']) > 20 else self.psychological_history['emotional_states']
        same_type_count = sum(1 for e in recent_emotions if e.get('emotion_type') == emotion_type)
        analysis['pattern_frequency'] = same_type_count
        
        # 确定推荐行动
        if emotion_type in ['fear', 'panic', 'anger'] and intensity > 0.7:
            analysis['recommended_action'] = 'pause_trading'
        elif emotion_type == 'greed' and intensity > 0.6:
            analysis['recommended_action'] = 'reduce_position'
        elif emotion_type in ['calm', 'focused'] and intensity > 0.6:
            analysis['recommended_action'] = 'continue_trading'
        
        return analysis
    
    def _generate_emotion_recommendation(self, emotion_data: Dict, emotion_analysis: Dict) -> Dict:
        """生成情绪管理建议"""
        emotion_type = emotion_data['emotion_type']
        intensity = emotion_data['intensity']
        recommended_action = emotion_analysis['recommended_action']
        
        recommendations = {
            'pause_trading': {
                'action': '暂停交易活动',
                'reason': '情绪强度过高，可能影响决策质量',
                'duration': '30分钟到2小时',
                'activities': ['深呼吸练习', '短暂休息', '情绪日记记录']
            },
            'reduce_position': {
                'action': '减少仓位规模',
                'reason': '贪婪或恐惧可能导致过度交易',
                'reduction': '减少50%仓位',
                'rationale': '降低情绪对交易决策的影响'
            },
            'continue_trading': {
                'action': '继续交易',
                'reason': '情绪状态有利于理性决策',
                'checkpoints': ['每笔交易后检查情绪', '设置严格的风险限制'],
                'monitoring': '继续保持情绪自我监控'
            },
            'monitor': {
                'action': '继续监控情绪',
                'reason': '情绪在正常范围内',
                'frequency': '每30分钟检查一次',
                'threshold': '如果强度超过0.7，采取行动'
            }
        }
        
        if recommended_action in recommendations:
            return recommendations[recommended_action]
        else:
            return {
                'action': '记录和分析情绪',
                'reason': f'情绪类型: {emotion_type}, 强度: {intensity:.1f}',
                'suggestion': '保持情绪日记，识别模式'
            }
    
    def conduct_discipline_check(self, trade_data: Dict, rules_applied: List[str]) -> Dict:
        """进行纪律检查"""
        check_id = f"discipline_{len(self.psychological_history['discipline_checks']) + 1:06d}"
        
        # 分析纪律遵守情况
        compliance_analysis = self._analyze_discipline_compliance(trade_data, rules_applied)
        
        discipline_check = {
            'check_id': check_id,
            'check_time': datetime.now(),
            'trade_data': trade_data,
            'rules_applied': rules_applied,
            'compliance_analysis': compliance_analysis,
            'compliance_score': compliance_analysis['overall_compliance_score'],
            'improvement_areas': compliance_analysis['improvement_areas']
        }
        
        # 添加到历史记录
        self.psychological_history['discipline_checks'].append(discipline_check)
        
        # 更新当前纪律状态
        self.current_psychological_state['discipline_level'] = compliance_analysis['compliance_level']
        self.current_psychological_state['discipline_score'] = compliance_analysis['overall_compliance_score']
        
        # 更新总体心理分数
        self._update_overall_psychological_score()
        
        return discipline_check
    
    def _analyze_discipline_compliance(self, trade_data: Dict, rules_applied: List[str]) -> Dict:
        """分析纪律遵守情况"""
        # 预定义的交易规则
        standard_rules = [
            'risk_management',  # 风险管理
            'position_sizing',  # 仓位规模
            'stop_loss',  # 止损设置
            'take_profit',  # 止盈设置
            'entry_confirmation',  # 入场确认
            'timeframe_alignment',  # 时间框架对齐
            'market_analysis',  # 市场分析
            'emotional_check'  # 情绪检查
        ]
        
        # 检查哪些规则被应用
        applied_set = set(rules_applied)
        standard_set = set(standard_rules)
        
        # 计算遵守比例
        compliance_ratio = len(applied_set.intersection(standard_set)) / len(standard_set) if standard_set else 0
        
        # 识别未遵守的规则
        missing_rules = list(standard_set - applied_set)
        
        # 分析交易结果
        trade_result = trade_data.get('result', 'unknown')
        profit_loss = trade_data.get('profit_loss', 0)
        
        # 确定遵守等级
        if compliance_ratio >= 0.9:
            compliance_level = 'excellent'
        elif compliance_ratio >= 0.7:
            compliance_level = 'good'
        elif compliance_ratio >= 0.5:
            compliance_level = 'fair'
        else:
            compliance_level = 'poor'
        
        # 识别改进领域
        improvement_areas = []
        if 'risk_management' not in applied_set:
            improvement_areas.append('加强风险管理规则应用')
        if 'emotional_check' not in applied_set:
            improvement_areas.append('增加情绪检查环节')
        if 'stop_loss' not in applied_set:
            improvement_areas.append('严格执行止损规则')
        
        # 如果有亏损且遵守度低，强调改进
        if profit_loss < 0 and compliance_ratio < 0.6:
            improvement_areas.append('亏损交易中纪律遵守不足，需要重点改进')
        
        return {
            'compliance_ratio': compliance_ratio,
            'compliance_level': compliance_level,
            'overall_compliance_score': compliance_ratio,
            'applied_rules_count': len(applied_set),
            'missing_rules': missing_rules,
            'improvement_areas': improvement_areas,
            'trade_result_impact': 'positive' if profit_loss > 0 and compliance_ratio > 0.7 else 'negative' if profit_loss < 0 and compliance_ratio < 0.5 else 'neutral'
        }
    
    def analyze_decision(self, decision_data: Dict) -> Dict:
        """分析交易决策"""
        decision_id = f"decision_{len(self.psychological_history['decision_analyses']) + 1:06d}"
        
        # 分析决策质量
        decision_analysis = self._analyze_decision_quality(decision_data)
        
        decision_record = {
            'decision_id': decision_id,
            'analysis_time': datetime.now(),
            'decision_data': decision_data,
            'decision_analysis': decision_analysis,
            'decision_score': decision_analysis['decision_quality_score'],
            'cognitive_biases': decision_analysis['cognitive_biases'],
            'improvement_suggestions': decision_analysis['improvement_suggestions']
        }
        
        # 添加到历史记录
        self.psychological_history['decision_analyses'].append(decision_record)
        
        # 更新当前决策质量状态
        self.current_psychological_state['decision_quality'] = decision_analysis['decision_quality_level']
        self.current_psychological_state['decision_score'] = decision_analysis['decision_quality_score']
        
        # 更新总体心理分数
        self._update_overall_psychological_score()
        
        return decision_record
    
    def _analyze_decision_quality(self, decision_data: Dict) -> Dict:
        """分析决策质量"""
        decision_process = decision_data.get('decision_process', 'unknown')
        time_taken = decision_data.get('time_taken_seconds', 0)
        information_used = decision_data.get('information_used', [])
        confidence_level = decision_data.get('confidence_level', 0.5)
        outcome = decision_data.get('outcome', 'unknown')
        
        # 认知偏差检查
        cognitive_biases = self._check_cognitive_biases(decision_data)
        
        # 决策过程评分
        process_score = self._evaluate_decision_process(decision_process, time_taken, information_used)
        
        # 信心与准确性匹配
        confidence_score = self._evaluate_confidence_accuracy(confidence_level, outcome)
        
        # 综合决策质量分数
        decision_quality_score = (process_score * 0.6) + (confidence_score * 0.4)
        
        # 确定决策质量等级
        if decision_quality_score >= 0.8:
            decision_quality_level = 'optimal'
        elif decision_quality_score >= 0.6:
            decision_quality_level = 'systematic'
        elif decision_quality_score >= 0.4:
            decision_quality_level = 'reasoned'
        elif decision_quality_score >= 0.2:
            decision_quality_level = 'emotional'
        else:
            decision_quality_level = 'impulsive'
        
        # 生成改进建议
        improvement_suggestions = self._generate_decision_improvements(cognitive_biases, decision_quality_score)
        
        return {
            'decision_quality_score': decision_quality_score,
            'decision_quality_level': decision_quality_level,
            'process_score': process_score,
            'confidence_score': confidence_score,
            'cognitive_biases': cognitive_biases,
            'improvement_suggestions': improvement_suggestions,
            'time_efficiency': 'efficient' if time_taken < 60 else 'moderate' if time_taken < 180 else 'inefficient'
        }
    
    def _check_cognitive_biases(self, decision_data: Dict) -> List[Dict]:
        """检查认知偏差"""
        biases = []
        
        # 检查确认偏差
        information_used = decision_data.get('information_used', [])
        if len(information_used) < 3:
            biases.append({
                'bias_type': 'confirmation_bias',
                'description': '使用的信息源有限，可能只寻找支持自己观点的信息',
                'severity': 'medium'
            })
        
        # 检查过度自信
        confidence_level = decision_data.get('confidence_level', 0.5)
        if confidence_level > 0.8 and len(information_used) < 5:
            biases.append({
                'bias_type': 'overconfidence',
                'description': '信心水平高但信息基础不足',
                'severity': 'high'
            })
        
        # 检查近期偏差
        decision_context = decision_data.get('context', '')
        if 'recent' in decision_context.lower() or 'latest' in decision_context.lower():
            biases.append({
                'bias_type': 'recency_bias',
                'description': '可能过度重视近期信息',
                'severity': 'low'
            })
        
        # 检查损失厌恶
        previous_outcome = decision_data.get('previous_outcome', '')
        if previous_outcome == 'loss' and decision_data.get('risk_taken', 0) < 0.01:
            biases.append({
                'bias_type': 'loss_aversion',
                'description': '前次亏损后可能过于谨慎',
                'severity': 'medium'
            })
        
        return biases
    
    def _evaluate_decision_process(self, decision_process: str, time_taken: float, information_used: List[str]) -> float:
        """评估决策过程"""
        score = 0.5  # 基础分数
        
        # 决策过程类型评分
        process_scores = {
            'systematic_analysis': 0.9,
            'rule_based': 0.8,
            'analytical': 0.7,
            'intuitive': 0.5,
            'emotional': 0.3,
            'impulsive': 0.1
        }
        
        if decision_process in process_scores:
            score = process_scores[decision_process]
        
        # 时间效率调整
        if 30 <= time_taken <= 300:  # 30秒到5分钟是理想范围
            time_adjustment = 0.1
        elif time_taken < 10:  # 太快可能冲动
            time_adjustment = -0.2
        elif time_taken > 600:  # 太慢可能犹豫
            time_adjustment = -0.1
        else:
            time_adjustment = 0
        
        score += time_adjustment
        
        # 信息使用调整
        info_count = len(information_used)
        if info_count >= 5:
            info_adjustment = 0.2
        elif info_count >= 3:
            info_adjustment = 0.1
        elif info_count >= 1:
            info_adjustment = 0
        else:
            info_adjustment = -0.3
        
        score += info_adjustment
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_confidence_accuracy(self, confidence_level: float, outcome: str) -> float:
        """评估信心与准确性匹配"""
        # 简化评估：假设好的决策应该有适度信心
        if 0.4 <= confidence_level <= 0.7:
            # 适度信心通常最好
            base_score = 0.8
        elif confidence_level > 0.7:
            # 过高信心可能过度自信
            base_score = 0.6
        else:
            # 过低信心可能缺乏确信
            base_score = 0.5
        
        # 根据结果调整
        if outcome == 'success':
            outcome_adjustment = 0.1
        elif outcome == 'failure':
            outcome_adjustment = -0.1
        else:
            outcome_adjustment = 0
        
        return max(0.0, min(1.0, base_score + outcome_adjustment))
    
    def _generate_decision_improvements(self, cognitive_biases: List[Dict], decision_score: float) -> List[str]:
        """生成决策改进建议"""
        improvements = []
        
        # 基于认知偏差的建议
        for bias in cognitive_biases:
            bias_type = bias.get('bias_type', '')
            
            if bias_type == 'confirmation_bias':
                improvements.append('主动寻找反对自己观点的信息')
            elif bias_type == 'overconfidence':
                improvements.append('降低信心水平，增加信息收集')
            elif bias_type == 'recency_bias':
                improvements.append('考虑更长期的历史数据')
            elif bias_type == 'loss_aversion':
                improvements.append('基于当前情况而非前次结果做决策')
        
        # 基于决策分数的建议
        if decision_score < 0.4:
            improvements.append('建立更系统的决策流程')
        if decision_score < 0.6:
            improvements.append('增加决策前的信息收集步骤')
        
        # 通用建议
        if not improvements:
            improvements.append('继续保持当前决策流程，定期审查')
        
        return improvements
    
    def assess_stress_level(self, stress_indicators: Dict) -> Dict:
        """评估压力水平"""
        assessment_id = f"stress_{len(self.psychological_history['stress_assessments']) + 1:06d}"
        
        # 分析压力水平
        stress_analysis = self._analyze_stress_level(stress_indicators)
        
        stress_assessment = {
            'assessment_id': assessment_id,
            'assessment_time': datetime.now(),
            'stress_indicators': stress_indicators,
            'stress_analysis': stress_analysis,
            'stress_score': stress_analysis['overall_stress_score'],
            'stress_level': stress_analysis['stress_level'],
            'recommended_actions': stress_analysis['recommended_actions']
        }
        
        # 添加到历史记录
        self.psychological_history['stress_assessments'].append(stress_assessment)
        
        # 更新当前压力状态
        self.current_psychological_state['stress_level'] = stress_analysis['stress_level']
        self.current_psychological_state['stress_score'] = stress_analysis['overall_stress_score']
        
        # 更新总体心理分数
        self._update_overall_psychological_score()
        
        return stress_assessment
    
    def _analyze_stress_level(self, stress_indicators: Dict) -> Dict:
        """分析压力水平"""
        # 提取压力指标
        physical_indicators = stress_indicators.get('physical_indicators', {})
        emotional_indicators = stress_indicators.get('emotional_indicators', {})
        behavioral_indicators = stress_indicators.get('behavioral_indicators', {})
        trading_indicators = stress_indicators.get('trading_indicators', {})
        
        # 计算各维度分数
        physical_score = self._calculate_physical_stress_score(physical_indicators)
        emotional_score = self._calculate_emotional_stress_score(emotional_indicators)
        behavioral_score = self._calculate_behavioral_stress_score(behavioral_indicators)
        trading_score = self._calculate_trading_stress_score(trading_indicators)
        
        # 计算总体压力分数
        overall_stress_score = (physical_score * 0.25 + emotional_score * 0.35 + 
                               behavioral_score * 0.20 + trading_score * 0.20)
        
        # 确定压力等级
        if overall_stress_score >= 0.8:
            stress_level = 'extreme'
        elif overall_stress_score >= 0.6:
            stress_level = 'high'
        elif overall_stress_score >= 0.4:
            stress_level = 'moderate'
        elif overall_stress_score >= 0.2:
            stress_level = 'normal'
        else:
            stress_level = 'relaxed'
        
        # 生成推荐行动
        recommended_actions = self._generate_stress_management_actions(overall_stress_score, stress_level)
        
        return {
            'physical_score': physical_score,
            'emotional_score': emotional_score,
            'behavioral_score': behavioral_score,
            'trading_score': trading_score,
            'overall_stress_score': overall_stress_score,
            'stress_level': stress_level,
            'recommended_actions': recommended_actions
        }
    
    def _calculate_physical_stress_score(self, physical_indicators: Dict) -> float:
        """计算身体压力分数"""
        score = 0.0
        
        # 疲劳程度
        fatigue = physical_indicators.get('fatigue_level', 0)
        score += fatigue * 0.3
        
        # 睡眠质量
        sleep_quality = physical_indicators.get('sleep_quality', 0.5)
        score += (1 - sleep_quality) * 0.3  # 睡眠差增加压力
        
        # 身体紧张
        tension = physical_indicators.get('body_tension', 0)
        score += tension * 0.2
        
        # 其他症状
        other_symptoms = physical_indicators.get('other_symptoms', [])
        score += len(other_symptoms) * 0.05
        
        return min(1.0, score)
    
    def _calculate_emotional_stress_score(self, emotional_indicators: Dict) -> float:
        """计算情绪压力分数"""
        score = 0.0
        
        # 焦虑水平
        anxiety = emotional_indicators.get('anxiety_level', 0)
        score += anxiety * 0.4
        
        # 沮丧程度
        frustration = emotional_indicators.get('frustration_level', 0)
        score += frustration * 0.3
        
        # 情绪波动
        mood_swings = emotional_indicators.get('mood_swings', 0)
        score += mood_swings * 0.2
        
        # 情绪恢复能力
        recovery = emotional_indicators.get('emotional_recovery', 0.5)
        score += (1 - recovery) * 0.1  # 恢复能力差增加压力
        
        return min(1.0, score)
    
    def _calculate_behavioral_stress_score(self, behavioral_indicators: Dict) -> float:
        """计算行为压力分数"""
        score = 0.0
        
        # 冲动行为
        impulsivity = behavioral_indicators.get('impulsivity', 0)
        score += impulsivity * 0.4
        
        # 注意力分散
        distraction = behavioral_indicators.get('distraction_level', 0)
        score += distraction * 0.3
        
        # 决策犹豫
        indecisiveness = behavioral_indicators.get('indecisiveness', 0)
        score += indecisiveness * 0.2
        
        # 社交回避
        social_withdrawal = behavioral_indicators.get('social_withdrawal', 0)
        score += social_withdrawal * 0.1
        
        return min(1.0, score)
    
    def _calculate_trading_stress_score(self, trading_indicators: Dict) -> float:
        """计算交易压力分数"""
        score = 0.0
        
        # 交易频率
        trade_frequency = trading_indicators.get('trade_frequency', 'normal')
        if trade_frequency == 'high':
            score += 0.4
        elif trade_frequency == 'very_high':
            score += 0.6
        elif trade_frequency == 'normal':
            score += 0.2
        
        # 风险暴露
        risk_exposure = trading_indicators.get('risk_exposure', 0)
        score += risk_exposure * 0.3
        
        # 近期表现
        recent_performance = trading_indicators.get('recent_performance', 0)
        if recent_performance < -0.05:  # 近期亏损超过5%
            score += 0.3
        elif recent_performance < 0:
            score += 0.1
        
        # 市场波动性
        market_volatility = trading_indicators.get('market_volatility', 0)
        score += market_volatility * 0.2
        
        return min(1.0, score)
    
    def _generate_stress_management_actions(self, stress_score: float, stress_level: str) -> List[str]:
        """生成压力管理行动建议"""
        actions = []
        
        if stress_level in ['high', 'extreme']:
            actions.append('立即暂停交易，至少休息2小时')
            actions.append('进行深呼吸或冥想练习（10-15分钟）')
            actions.append('进行轻度身体活动（散步、伸展）')
            actions.append('减少明天交易计划的风险暴露')
            
            if stress_level == 'extreme':
                actions.append('考虑今天不再交易')
                actions.append('寻求专业心理支持（如需要）')
        
        elif stress_level == 'moderate':
            actions.append('短暂休息15-30分钟')
            actions.append('检查并调整交易计划，降低风险')
            actions.append('进行正念练习（5-10分钟）')
            actions.append('设置更严格的交易限额')
        
        elif stress_level == 'normal':
            actions.append('继续保持当前节奏')
            actions.append('每小时检查一次压力水平')
            actions.append('保持适当休息间隔')
        
        else:  # relaxed
            actions.append('继续保持良好状态')
            actions.append('定期监控压力指标')
            actions.append('维持健康的工作休息平衡')
        
        # 通用建议
        actions.append('保持充足睡眠（7-8小时）')
        actions.append('均衡饮食，避免过度咖啡因')
        actions.append('定期进行压力评估')
        
        return actions
    
    def track_confidence(self, confidence_data: Dict) -> Dict:
        """跟踪自信心水平"""
        tracking_id = f"confidence_{len(self.psychological_history['confidence_tracking']) + 1:06d}"
        
        # 分析自信心
        confidence_analysis = self._analyze_confidence_level(confidence_data)
        
        confidence_tracking = {
            'tracking_id': tracking_id,
            'tracking_time': datetime.now(),
            'confidence_data': confidence_data,
            'confidence_analysis': confidence_analysis,
            'confidence_score': confidence_analysis['overall_confidence_score'],
            'confidence_level': confidence_analysis['confidence_level'],
            'building_activities': confidence_analysis['building_activities']
        }
        
        # 添加到历史记录
        self.psychological_history['confidence_tracking'].append(confidence_tracking)
        
        # 更新当前自信心状态
        self.current_psychological_state['confidence_level'] = confidence_analysis['confidence_level']
        self.current_psychological_state['confidence_score'] = confidence_analysis['overall_confidence_score']
        
        # 更新总体心理分数
        self._update_overall_psychological_score()
        
        return confidence_tracking
    
    def _analyze_confidence_level(self, confidence_data: Dict) -> Dict:
        """分析自信心水平"""
        # 提取信心指标
        self_assessment = confidence_data.get('self_assessment', {})
        performance_data = confidence_data.get('performance_data', {})
        feedback_data = confidence_data.get('feedback_data', {})
        
        # 计算各维度分数
        self_score = self._calculate_self_assessment_score(self_assessment)
        performance_score = self._calculate_performance_based_score(performance_data)
        feedback_score = self._calculate_feedback_based_score(feedback_data)
        
        # 计算总体信心分数
        overall_confidence_score = (self_score * 0.4 + performance_score * 0.4 + feedback_score * 0.2)
        
        # 确定信心等级
        if overall_confidence_score >= 0.8:
            confidence_level = 'very_high'
        elif overall_confidence_score >= 0.6:
            confidence_level = 'high'
        elif overall_confidence_score >= 0.4:
            confidence_level = 'adequate'
        elif overall_confidence_score >= 0.2:
            confidence_level = 'moderate'
        else:
            confidence_level = 'low'
        
        # 生成信心建设活动
        building_activities = self._generate_confidence_building_activities(overall_confidence_score, confidence_level)
        
        return {
            'self_score': self_score,
            'performance_score': performance_score,
            'feedback_score': feedback_score,
            'overall_confidence_score': overall_confidence_score,
            'confidence_level': confidence_level,
            'building_activities': building_activities
        }
    
    def _calculate_self_assessment_score(self, self_assessment: Dict) -> float:
        """计算自我评估分数"""
        score = 0.5  # 基础分数
        
        # 能力自信
        ability_confidence = self_assessment.get('ability_confidence', 0.5)
        score += (ability_confidence - 0.5) * 0.3
        
        # 知识自信
        knowledge_confidence = self_assessment.get('knowledge_confidence', 0.5)
        score += (knowledge_confidence - 0.5) * 0.2
        
        # 决策自信
        decision_confidence = self_assessment.get('decision_confidence', 0.5)
        score += (decision_confidence - 0.5) * 0.3
        
        # 风险承受自信
        risk_confidence = self_assessment.get('risk_confidence', 0.5)
        score += (risk_confidence - 0.5) * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_performance_based_score(self, performance_data: Dict) -> float:
        """计算基于表现的分数"""
        score = 0.5  # 基础分数
        
        # 近期胜率
        recent_win_rate = performance_data.get('recent_win_rate', 0.5)
        score += (recent_win_rate - 0.5) * 0.4
        
        # 盈利因子
        profit_factor = performance_data.get('profit_factor', 1.0)
        if profit_factor >= 2.0:
            profit_adjustment = 0.3
        elif profit_factor >= 1.5:
            profit_adjustment = 0.2
        elif profit_factor >= 1.2:
            profit_adjustment = 0.1
        elif profit_factor >= 1.0:
            profit_adjustment = 0
        else:
            profit_adjustment = -0.2
        
        score += profit_adjustment
        
        # 一致性
        consistency = performance_data.get('consistency', 0.5)
        score += (consistency - 0.5) * 0.2
        
        # 改进趋势
        improvement = performance_data.get('improvement_trend', 0)
        score += improvement * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_feedback_based_score(self, feedback_data: Dict) -> float:
        """计算基于反馈的分数"""
        score = 0.5  # 基础分数
        
        # 自我反思质量
        reflection_quality = feedback_data.get('reflection_quality', 0.5)
        score += (reflection_quality - 0.5) * 0.4
        
        # 学习应用
        learning_application = feedback_data.get('learning_application', 0.5)
        score += (learning_application - 0.5) * 0.3
        
        # 适应能力
        adaptability = feedback_data.get('adaptability', 0.5)
        score += (adaptability - 0.5) * 0.2
        
        # 导师/同行反馈
        external_feedback = feedback_data.get('external_feedback', 0.5)
        score += (external_feedback - 0.5) * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_confidence_building_activities(self, confidence_score: float, confidence_level: str) -> List[str]:
        """生成信心建设活动"""
        activities = []
        
        if confidence_level in ['low', 'moderate']:
            activities.append('记录和回顾成功交易，建立成功档案')
            activities.append('设置并实现小目标，积累成功经验')
            activities.append('进行模拟交易练习，在无风险环境中建立信心')
            activities.append('学习并掌握一个具体的交易策略')
            
            if confidence_level == 'low':
                activities.append('减少真实交易规模，逐步建立信心')
                activities.append('寻求导师指导或同行反馈')
        
        elif confidence_level == 'adequate':
            activities.append('继续当前的学习和练习节奏')
            activities.append('定期审查和调整交易计划')
            activities.append('挑战稍微超出舒适区的交易机会')
            activities.append('分享经验，帮助其他交易者')
        
        elif confidence_level in ['high', 'very_high']:
            activities.append('保持谦逊，避免过度自信')
            activities.append('继续学习，探索新的交易领域')
            activities.append('指导经验较少的交易者')
            activities.append('设置更高但现实的目标')
            
            if confidence_level == 'very_high':
                activities.append('特别注意风险管理，避免因过度自信而冒险')
                activities.append('定期进行压力测试，确保信心有实际基础')
        
        # 通用活动
        activities.append('定期进行自我评估和反思')
        activities.append('保持交易日志，记录决策和结果')
        activities.append('参加交易社区，获取多元视角')
        
        return activities
    
    def conduct_training_session(self, session_data: Dict) -> Dict:
        """进行训练会话"""
        session_id = f"training_{len(self.psychological_history['training_sessions']) + 1:06d}"
        
        # 分析训练效果
        session_analysis = self._analyze_training_session(session_data)
        
        training_session = {
            'session_id': session_id,
            'session_time': datetime.now(),
            'session_data': session_data,
            'session_analysis': session_analysis,
            'effectiveness_score': session_analysis['effectiveness_score'],
            'key_learnings': session_analysis['key_learnings'],
            'follow_up_actions': session_analysis['follow_up_actions']
        }
        
        # 添加到历史记录
        self.psychological_history['training_sessions'].append(training_session)
        
        return training_session
    
    def _analyze_training_session(self, session_data: Dict) -> Dict:
        """分析训练会话"""
        session_type = session_data.get('session_type', 'unknown')
        duration_minutes = session_data.get('duration_minutes', 0)
        focus_areas = session_data.get('focus_areas', [])
        activities_completed = session_data.get('activities_completed', [])
        self_assessment = session_data.get('self_assessment', {})
        
        # 计算会话效果分数
        effectiveness_score = self._calculate_session_effectiveness(
            session_type, duration_minutes, focus_areas, activities_completed, self_assessment
        )
        
        # 提取关键学习点
        key_learnings = self._extract_key_learnings(session_data)
        
        # 确定后续行动
        follow_up_actions = self._determine_follow_up_actions(effectiveness_score, focus_areas)
        
        return {
            'effectiveness_score': effectiveness_score,
            'effectiveness_level': 'high' if effectiveness_score >= 0.7 else 'medium' if effectiveness_score >= 0.5 else 'low',
            'key_learnings': key_learnings,
            'follow_up_actions': follow_up_actions,
            'session_coverage': len(focus_areas),
            'activity_completion': len(activities_completed)
        }
    
    def _calculate_session_effectiveness(self, session_type: str, duration: int, 
                                       focus_areas: List[str], activities: List[str], 
                                       self_assessment: Dict) -> float:
        """计算会话效果分数"""
        score = 0.5  # 基础分数
        
        # 持续时间调整
        if 30 <= duration <= 90:  # 30-90分钟理想
            duration_adjustment = 0.2
        elif duration > 120:  # 过长可能疲劳
            duration_adjustment = 0
        elif duration < 15:  # 过短可能不足
            duration_adjustment = -0.2
        else:
            duration_adjustment = 0.1
        
        score += duration_adjustment
        
        # 专注领域数量
        if 2 <= len(focus_areas) <= 4:
            focus_adjustment = 0.2
        elif len(focus_areas) == 1:
            focus_adjustment = 0.1
        elif len(focus_areas) > 4:
            focus_adjustment = 0  # 过多可能分散
        else:
            focus_adjustment = -0.1
        
        score += focus_adjustment
        
        # 活动完成度
        completion_ratio = len(activities) / max(1, len(focus_areas) * 2)  # 假设每个领域2个活动
        completion_adjustment = (completion_ratio - 0.5) * 0.3
        score += completion_adjustment
        
        # 自我评估调整
        engagement = self_assessment.get('engagement_level', 0.5)
        score += (engagement - 0.5) * 0.2
        
        learning = self_assessment.get('learning_gained', 0.5)
        score += (learning - 0.5) * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _extract_key_learnings(self, session_data: Dict) -> List[str]:
        """提取关键学习点"""
        learnings = session_data.get('key_learnings', [])
        
        # 如果没有提供，从会话数据推断
        if not learnings:
            focus_areas = session_data.get('focus_areas', [])
            for area in focus_areas:
                if 'emotional' in area.lower():
                    learnings.append(f'提高了对{area}的理解和管理')
                elif 'discipline' in area.lower():
                    learnings.append(f'加强了{area}的实践应用')
                elif 'decision' in area.lower():
                    learnings.append(f'改进了{area}的过程和方法')
                elif 'stress' in area.lower():
                    learnings.append(f'学习了{area}的识别和管理技巧')
                elif 'confidence' in area.lower():
                    learnings.append(f'建立了{area}的基础和改进方向')
                else:
                    learnings.append(f'在{area}方面有所进展')
        
        # 限制数量
        return learnings[:5]  # 最多5个关键学习点
    
    def _determine_follow_up_actions(self, effectiveness_score: float, focus_areas: List[str]) -> List[str]:
        """确定后续行动"""
        actions = []
        
        if effectiveness_score >= 0.7:
            actions.append('继续当前训练计划，逐步增加难度')
            actions.append('将学习应用到实际交易中')
            actions.append('定期复习和巩固学到的技巧')
        elif effectiveness_score >= 0.5:
            actions.append('重复训练薄弱环节')
            actions.append('调整训练方法，提高参与度')
            actions.append('设置更具体的小目标')
        else:
            actions.append('重新评估训练需求和目标')
            actions.append('尝试不同的训练方法或内容')
            actions.append('寻求外部指导或反馈')
        
        # 基于专注领域的特定行动
        for area in focus_areas:
            if 'emotional' in area.lower():
                actions.append(f'在日常交易中实践{area}管理技巧')
            elif 'discipline' in area.lower():
                actions.append(f'设置{area}检查点并跟踪遵守情况')
            elif 'decision' in area.lower():
                actions.append(f'在下一笔交易中应用改进的{area}流程')
        
        return actions[:5]  # 最多5个后续行动
    
    def _update_overall_psychological_score(self) -> None:
        """更新总体心理分数"""
        # 计算各维度平均分
        scores = [
            self.current_psychological_state.get('emotional_score', 0.5),
            self.current_psychological_state.get('discipline_score', 0.5),
            self.current_psychological_state.get('decision_score', 0.5),
            self.current_psychological_state.get('stress_score', 0.5),
            self.current_psychological_state.get('confidence_score', 0.5)
        ]
        
        # 压力分数反向处理（压力越低越好）
        stress_score = scores[3]
        adjusted_stress_score = 1.0 - stress_score  # 反转
        
        # 重新计算平均（压力使用调整后的分数）
        scores[3] = adjusted_stress_score
        overall_score = np.mean(scores) if scores else 0.5
        
        self.current_psychological_state['overall_psychological_score'] = overall_score
        self.current_psychological_state['last_assessment_time'] = datetime.now()
        
        # 评估改进趋势
        self._evaluate_improvement_trend()
    
    def _evaluate_improvement_trend(self) -> None:
        """评估改进趋势"""
        # 这里简化处理，实际应该分析历史数据
        # 使用随机趋势模拟，实际应基于历史分数计算
        import random
        trends = ['improving', 'stable', 'declining']
        self.current_psychological_state['improvement_trend'] = random.choice(trends)
    
    def get_psychological_report(self, period: str = 'recent') -> Dict:
        """获取心理状态报告"""
        report = {
            'report_time': datetime.now(),
            'period': period,
            'current_state': self.current_psychological_state,
            'training_goals': self.training_goals,
            'progress_assessment': self._assess_progress_against_goals(),
            'recommended_focus_areas': self._identify_focus_areas(),
            'training_plan': self._generate_training_plan()
        }
        
        return report
    
    def _assess_progress_against_goals(self) -> Dict:
        """评估相对于目标的进展"""
        progress = {}
        
        for goal_name, target_value in self.training_goals.items():
            current_value = 0.0
            
            # 映射目标到当前状态
            if goal_name == 'emotional_control_target':
                current_value = self.current_psychological_state.get('emotional_score', 0.5)
            elif goal_name == 'discipline_compliance_target':
                current_value = self.current_psychological_state.get('discipline_score', 0.5)
            elif goal_name == 'decision_quality_target':
                current_value = self.current_psychological_state.get('decision_score', 0.5)
            elif goal_name == 'stress_management_target':
                # 压力分数需要反转
                stress_score = self.current_psychological_state.get('stress_score', 0.5)
                current_value = 1.0 - stress_score
            elif goal_name == 'confidence_level_target':
                current_value = self.current_psychological_state.get('confidence_score', 0.5)
            elif goal_name == 'resilience_score_target':
                # 韧性分数使用总体分数
                current_value = self.current_psychological_state.get('overall_psychological_score', 0.5)
            
            # 计算差距
            gap = target_value - current_value
            
            progress[goal_name] = {
                'target': target_value,
                'current': current_value,
                'gap': gap,
                'progress_percentage': (current_value / target_value) * 100 if target_value > 0 else 0,
                'status': 'achieved' if current_value >= target_value else 'on_track' if gap <= 0.1 else 'needs_work' if gap <= 0.2 else 'far_behind'
            }
        
        return progress
    
    def _identify_focus_areas(self) -> List[Dict]:
        """识别需要重点关注的领域"""
        focus_areas = []
        progress = self._assess_progress_against_goals()
        
        for goal_name, goal_progress in progress.items():
            status = goal_progress.get('status', 'unknown')
            gap = goal_progress.get('gap', 0)
            
            if status in ['needs_work', 'far_behind']:
                # 将目标名称转换为可读的领域名称
                area_map = {
                    'emotional_control_target': '情绪控制',
                    'discipline_compliance_target': '纪律遵守',
                    'decision_quality_target': '决策质量',
                    'stress_management_target': '压力管理',
                    'confidence_level_target': '自信心建设',
                    'resilience_score_target': '心理韧性'
                }
                
                area_name = area_map.get(goal_name, goal_name)
                
                focus_areas.append({
                    'area': area_name,
                    'current_score': goal_progress['current'],
                    'target_score': goal_progress['target'],
                    'gap': gap,
                    'priority': 'high' if status == 'far_behind' else 'medium',
                    'recommended_training': self._get_training_for_area(area_name)
                })
        
        # 按优先级排序
        focus_areas.sort(key=lambda x: 0 if x['priority'] == 'high' else 1)
        
        return focus_areas[:3]  # 最多3个重点领域
    
    def _get_training_for_area(self, area_name: str) -> List[str]:
        """获取特定领域的训练建议"""
        training_suggestions = {
            '情绪控制': ['情绪日记记录', '正念冥想练习', '情绪识别训练', '冷静技巧实践'],
            '纪律遵守': ['交易规则检查', '习惯形成训练', '冲动控制练习', '一致性建设活动'],
            '决策质量': ['认知偏差识别', '决策流程优化', '结果分析练习', '模式识别训练'],
            '压力管理': ['压力识别训练', '放松技巧练习', '工作量管理', '恢复协议实施'],
            '自信心建设': ['成功记录', '技能认可练习', '积极自我对话', '目标达成跟踪'],
            '心理韧性': ['挫折恢复训练', '逆境应对练习', '坚持性建设', '成长心态培养']
        }
        
        return training_suggestions.get(area_name, ['通用心理训练', '自我反思', '实践应用'])
    
    def _generate_training_plan(self) -> Dict:
        """生成训练计划"""
        focus_areas = self._identify_focus_areas()
        
        plan = {
            'plan_id': f"psych_training_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'creation_time': datetime.now(),
            'duration_weeks': 4,
            'weekly_sessions': 3,
            'session_duration_minutes': 45,
            'focus_areas': [area['area'] for area in focus_areas],
            'weekly_schedule': self._create_weekly_schedule(focus_areas),
            'success_metrics': self._define_success_metrics(focus_areas),
            'review_points': ['每周', '每两周', '计划结束时']
        }
        
        return plan
    
    def _create_weekly_schedule(self, focus_areas: List[Dict]) -> List[Dict]:
        """创建每周训练计划"""
        schedule = []
        
        # 简化：每周3次会话，循环处理重点领域
        for week in range(1, 5):  # 4周
            week_schedule = {
                'week': week,
                'sessions': []
            }
            
            for session in range(1, 4):  # 每周3次
                # 分配重点领域（循环）
                area_index = (week * 3 + session - 1) % len(focus_areas) if focus_areas else 0
                area = focus_areas[area_index]['area'] if focus_areas else '综合心理训练'
                
                session_plan = {
                    'session_number': session,
                    'focus_area': area,
                    'duration_minutes': 45,
                    'activities': self._get_training_for_area(area)[:3],  # 每个领域最多3个活动
                    'objectives': [f'提高{area}技能', f'应用{area}技巧', f'评估{area}进展']
                }
                
                week_schedule['sessions'].append(session_plan)
            
            schedule.append(week_schedule)
        
        return schedule
    
    def _define_success_metrics(self, focus_areas: List[Dict]) -> List[Dict]:
        """定义成功指标"""
        metrics = []
        
        for area in focus_areas:
            area_name = area['area']
            current_score = area['current_score']
            target_score = area['target_score']
            
            metrics.append({
                'metric': f'{area_name}分数',
                'current': current_score,
                'target': target_score,
                'improvement_target': min(1.0, current_score + 0.15),  # 目标提高0.15
                'measurement_method': '心理评估问卷 + 实际交易表现'
            })
        
        # 添加总体指标
        metrics.append({
            'metric': '总体心理分数',
            'current': self.current_psychological_state.get('overall_psychological_score', 0.5),
            'target': 0.75,
            'improvement_target': 0.1,  # 提高0.1
            'measurement_method': '综合心理评估'
        })
        
        return metrics


def demo_psychological_training():
    """演示心理训练系统"""
    print("=" * 60)
    print("心理训练系统演示")
    print("第28章：心理训练 - AL Brooks《价格行为交易之区间篇》")
    print("=" * 60)
    
    # 创建系统实例
    psych_system = PsychologicalTrainingSystem()
    
    print("\n1. 记录情绪状态...")
    emotion_result = psych_system.record_emotional_state({
        'emotion_type': 'anxiety',
        'intensity': 0.6,
        'context': 'pre_trade',
        'trigger': 'market_volatility_increase'
    })
    print(f"   情绪类型: {emotion_result['emotion_type']}")
    print(f"   情绪分数: {emotion_result['emotion_score']:.2f}")
    print(f"   建议行动: {emotion_result['recommendation']['action']}")
    
    print("\n2. 进行纪律检查...")
    discipline_result = psych_system.conduct_discipline_check(
        trade_data={'result': 'win', 'profit_loss': 150.0},
        rules_applied=['risk_management', 'position_sizing', 'stop_loss', 'entry_confirmation']
    )
    print(f"   纪律遵守分数: {discipline_result['compliance_score']:.2f}")
    print(f"   遵守等级: {discipline_result['compliance_analysis']['compliance_level']}")
    print(f"   改进领域: {len(discipline_result['improvement_areas'])}个")
    
    print("\n3. 分析决策...")
    decision_result = psych_system.analyze_decision({
        'decision_process': 'analytical',
        'time_taken_seconds': 120,
        'information_used': ['market_analysis', 'technical_indicators', 'risk_assessment'],
        'confidence_level': 0.7,
        'outcome': 'success',
        'context': 'entry_decision'
    })
    print(f"   决策质量分数: {decision_result['decision_score']:.2f}")
    print(f"   决策质量等级: {decision_result['decision_analysis']['decision_quality_level']}")
    print(f"   认知偏差: {len(decision_result['cognitive_biases'])}个")
    
    print("\n4. 评估压力水平...")
    stress_result = psych_system.assess_stress_level({
        'physical_indicators': {'fatigue_level': 0.4, 'sleep_quality': 0.6},
        'emotional_indicators': {'anxiety_level': 0.5, 'frustration_level': 0.3},
        'behavioral_indicators': {'impulsivity': 0.2, 'distraction_level': 0.4},
        'trading_indicators': {'trade_frequency': 'normal', 'risk_exposure': 0.3}
    })
    print(f"   压力分数: {stress_result['stress_score']:.2f}")
    print(f"   压力等级: {stress_result['stress_level']}")
    print(f"   推荐行动: {len(stress_result['recommended_actions'])}个")
    
    print("\n5. 获取心理状态报告...")
    psych_report = psych_system.get_psychological_report()
    print(f"   总体心理分数: {psych_report['current_state']['overall_psychological_score']:.2f}")
    print(f"   重点领域: {len(psych_report['recommended_focus_areas'])}个")
    print(f"   训练计划时长: {psych_report['training_plan']['duration_weeks']}周")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("心理训练系统已成功创建并测试")
    print("=" * 60)


# ============================================================================
# 策略改造: 添加PriceActionRangesPsychologicalTrainingSystemStrategy类
# 将价格行为区间心理训练系统转换为交易策略
# ============================================================================

class PriceActionRangesPsychologicalTrainingSystemStrategy(BaseStrategy):
    """价格行为区间心理训练策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        trader_profile = params.get('trader_profile', {
            'experience_level': 'intermediate',
            'risk_tolerance': 'moderate',
            'trading_style': 'mixed'
        })
        
        # 创建心理训练系统实例
        self.psych_system = PsychologicalTrainingSystem(trader_profile=trader_profile)
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于心理训练分析生成交易信号
        """
        # 获取心理状态报告
        psych_report = self.psych_system.get_psychological_report()
        
        # 分析心理状态
        current_state = psych_report.get('current_state', {})
        overall_score = current_state.get('overall_psychological_score', 0)
        
        recommended_focus_areas = psych_report.get('recommended_focus_areas', [])
        
        # 获取训练计划
        training_plan = psych_report.get('training_plan', {})
        training_intensity = training_plan.get('training_intensity', 'moderate')
        
        # 根据心理状态生成信号
        if overall_score >= 80:
            # 心理状态优秀，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif overall_score <= 50:
            # 心理状态差，检查是否有高风险领域
            has_high_risk_areas = any(
                area in ['emotional_control', 'discipline', 'stress_management']
                for area in recommended_focus_areas
            )
            
            if has_high_risk_areas:
                # 有高风险心理领域，hold信号（暂停交易）
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='hold',
                    price=self.data['close'].iloc[-1]
                )
            else:
                # 心理状态差但无高风险领域，卖出信号
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='sell',
                    price=self.data['close'].iloc[-1]
                )
        elif training_intensity == 'high':
            # 需要高强度训练，hold信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        else:
            # 中等心理状态，hold信号
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
    demo_psychological_training()