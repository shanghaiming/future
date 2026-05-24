# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.639317

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易日志分析量化系统
第24章：交易日志分析
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 交易记录：详细记录每笔交易的入场、出场、仓位、结果
2. 绩效分析：胜率、盈亏比、夏普比率、最大回撤等
3. 模式识别：识别交易行为模式、情绪模式、时间模式
4. 错误分析：识别常见错误、重复错误、代价高昂的错误
5. 改进建议：基于数据分析提供具体改进建议
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class TradingLogAnalyzer:
    """交易日志分析器（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 trader_profile: Dict = None,
                 log_retention_days: int = 365):
        """
        初始化交易日志分析器
        
        参数:
            trader_profile: 交易者个人资料
            log_retention_days: 日志保留天数
        """
        self.trader_profile = trader_profile or {
            'experience_level': 'intermediate',
            'trading_style': 'swing',
            'risk_tolerance': 'moderate'
        }
        
        self.log_retention_days = log_retention_days
        self.trades_log = []
        self.performance_history = []
        self.analysis_reports = []
        
        # 初始化错误分类
        self.error_categories = self._initialize_error_categories()
    
    def log_trade(self,
                  trade_data: Dict) -> Dict:
        """
        记录交易
        
        参数:
            trade_data: 交易数据，包含入场、出场、仓位等信息
            
        返回:
            记录结果
        """
        # 验证必要字段
        required_fields = ['entry_time', 'exit_time', 'entry_price', 'exit_price',
                          'position_size', 'direction', 'instrument']
        
        missing_fields = [field for field in required_fields if field not in trade_data]
        if missing_fields:
            return {'error': f'缺少必要字段: {missing_fields}'}
        
        # 生成交易ID
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.trades_log) + 1}"
        
        # 计算交易结果
        trade_result = self._calculate_trade_result(trade_data)
        
        # 计算交易质量评分
        trade_quality = self._evaluate_trade_quality(trade_data, trade_result)
        
        # 识别可能的错误
        trade_errors = self._identify_trade_errors(trade_data, trade_result)
        
        # 创建完整交易记录
        trade_record = {
            'trade_id': trade_id,
            'log_time': datetime.now(),
            **trade_data,
            'trade_result': trade_result,
            'trade_quality': trade_quality,
            'trade_errors': trade_errors,
            'tags': self._generate_trade_tags(trade_data, trade_result, trade_quality, trade_errors),
            'notes': trade_data.get('notes', ''),
            'attachments': trade_data.get('attachments', [])
        }
        
        self.trades_log.append(trade_record)
        
        # 更新绩效历史
        self._update_performance_history(trade_record)
        
        return {
            'status': 'success',
            'trade_id': trade_id,
            'trade_record': trade_record,
            'summary': {
                'pnl': trade_result['pnl'],
                'pnl_percentage': trade_result['pnl_percentage'],
                'quality_score': trade_quality['overall_score'],
                'error_count': len(trade_errors)
            }
        }
    
    def _initialize_error_categories(self) -> Dict:
        """初始化错误分类"""
        error_categories = {
            'entry_errors': {
                'description': '入场相关错误',
                'subcategories': {
                    'poor_entry_timing': '入场时机不佳',
                    'ignored_entry_conditions': '忽略入场条件',
                    'emotional_entry': '情绪化入场',
                    'overtrading': '过度交易',
                    'fear_of_missing_out': '害怕错过'
                }
            },
            'exit_errors': {
                'description': '出场相关错误',
                'subcategories': {
                    'early_exit': '过早出场',
                    'late_exit': '过晚出场',
                    'no_stop_loss': '未设置止损',
                    'ignored_stop_loss': '忽略止损',
                    'moved_stop_loss': '移动止损不当'
                }
            },
            'risk_management_errors': {
                'description': '风险管理错误',
                'subcategories': {
                    'oversized_position': '仓位过大',
                    'no_position_sizing': '未计算仓位',
                    'ignored_risk_limits': '忽略风险限制',
                    'revenge_trading': '报复性交易',
                    'martingale': '马丁格尔策略'
                }
            },
            'psychological_errors': {
                'description': '心理错误',
                'subcategories': {
                    'greed': '贪婪',
                    'fear': '恐惧',
                    'overconfidence': '过度自信',
                    'hope': '希望（持仓亏损）',
                    'regret': '后悔'
                }
            },
            'plan_execution_errors': {
                'description': '计划执行错误',
                'subcategories': {
                    'no_trading_plan': '无交易计划',
                    'ignored_plan': '忽略交易计划',
                    'poor_preparation': '准备不足',
                    'distracted_trading': '分心交易',
                    'fatigue_trading': '疲劳交易'
                }
            }
        }
        
        return error_categories
    
    def _calculate_trade_result(self, trade_data: Dict) -> Dict:
        """计算交易结果"""
        entry_price = trade_data['entry_price']
        exit_price = trade_data['exit_price']
        position_size = trade_data['position_size']
        direction = trade_data['direction']
        
        # 计算盈亏
        if direction == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:  # short
            pnl = (entry_price - exit_price) * position_size
        
        # 计算盈亏百分比
        investment = entry_price * position_size
        pnl_percentage = pnl / investment if investment > 0 else 0
        
        # 计算持仓时间
        entry_time = trade_data['entry_time']
        exit_time = trade_data['exit_time']
        
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
        
        holding_period = (exit_time - entry_time).total_seconds() / 3600  # 小时
        
        # 计算风险回报指标
        stop_loss = trade_data.get('stop_loss')
        take_profit = trade_data.get('take_profit')
        
        if stop_loss and take_profit:
            if direction == 'long':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
        else:
            risk_reward_ratio = 0
        
        # 计算实际风险回报
        if direction == 'long':
            actual_risk = entry_price - min(exit_price, entry_price)
            actual_reward = max(exit_price, entry_price) - entry_price
        else:
            actual_risk = max(exit_price, entry_price) - entry_price
            actual_reward = entry_price - min(exit_price, entry_price)
        
        actual_risk_reward_ratio = actual_reward / actual_risk if actual_risk > 0 else 0
        
        return {
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'holding_period_hours': holding_period,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'direction': direction,
            'instrument': trade_data['instrument'],
            'planned_risk_reward_ratio': risk_reward_ratio,
            'actual_risk_reward_ratio': actual_risk_reward_ratio,
            'win': pnl > 0,
            'breakeven': abs(pnl) < investment * 0.001,  # 盈亏平衡（0.1%以内）
            'significant_win': pnl_percentage > 0.02,  # 显著盈利（>2%）
            'significant_loss': pnl_percentage < -0.01  # 显著亏损（<-1%）
        }
    
    def _evaluate_trade_quality(self,
                               trade_data: Dict,
                               trade_result: Dict) -> Dict:
        """评估交易质量"""
        quality_score_components = []
        quality_notes = []
        
        # 1. 入场质量（30%）
        entry_quality = self._evaluate_entry_quality(trade_data, trade_result)
        quality_score_components.append(entry_quality['score'] * 0.3)
        quality_notes.extend(entry_quality['notes'])
        
        # 2. 出场质量（30%）
        exit_quality = self._evaluate_exit_quality(trade_data, trade_result)
        quality_score_components.append(exit_quality['score'] * 0.3)
        quality_notes.extend(exit_quality['notes'])
        
        # 3. 风险管理质量（20%）
        risk_quality = self._evaluate_risk_management_quality(trade_data, trade_result)
        quality_score_components.append(risk_quality['score'] * 0.2)
        quality_notes.extend(risk_quality['notes'])
        
        # 4. 计划执行质量（20%）
        execution_quality = self._evaluate_execution_quality(trade_data, trade_result)
        quality_score_components.append(execution_quality['score'] * 0.2)
        quality_notes.extend(execution_quality['notes'])
        
        # 计算总体质量分数
        overall_score = sum(quality_score_components)
        
        # 确定质量等级
        if overall_score >= 0.8:
            quality_grade = 'excellent'
            grade_description = '优秀交易：严格执行计划，风险管理良好'
        elif overall_score >= 0.7:
            quality_grade = 'good'
            grade_description = '良好交易：基本按计划执行，有小改进空间'
        elif overall_score >= 0.6:
            quality_grade = 'fair'
            grade_description = '一般交易：执行有偏差，需要改进'
        elif overall_score >= 0.5:
            quality_grade = 'poor'
            grade_description = '较差交易：执行问题较多，需要重大改进'
        else:
            quality_grade = 'very_poor'
            grade_description = '很差交易：严重问题，需要彻底反思'
        
        return {
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'grade_description': grade_description,
            'entry_quality': entry_quality,
            'exit_quality': exit_quality,
            'risk_quality': risk_quality,
            'execution_quality': execution_quality,
            'quality_notes': quality_notes,
            'improvement_areas': self._identify_quality_improvement_areas(
                entry_quality, exit_quality, risk_quality, execution_quality
            )
        }
    
    def _evaluate_entry_quality(self,
                               trade_data: Dict,
                               trade_result: Dict) -> Dict:
        """评估入场质量"""
        score = 0.7  # 基础分数
        notes = []
        
        # 检查是否有交易计划
        if trade_data.get('trading_plan_id'):
            score += 0.1
            notes.append('✅ 有交易计划参考')
        else:
            score -= 0.1
            notes.append('❌ 无交易计划参考')
        
        # 检查入场条件
        entry_conditions = trade_data.get('entry_conditions', [])
        if entry_conditions and len(entry_conditions) >= 2:
            score += 0.1
            notes.append('✅ 入场条件明确')
        else:
            score -= 0.05
            notes.append('⚠️ 入场条件不足')
        
        # 检查入场时机
        entry_time = trade_data.get('entry_time')
        if entry_time:
            # 简单检查：是否在主要交易时段（这里简化处理）
            if isinstance(entry_time, datetime):
                hour = entry_time.hour
                if 9 <= hour <= 16:  # 假设主要交易时段
                    score += 0.05
                    notes.append('✅ 在主要交易时段入场')
        
        # 检查情绪状态
        emotional_state = trade_data.get('emotional_state', 'calm')
        if emotional_state == 'calm':
            score += 0.05
            notes.append('✅ 情绪状态平稳')
        elif emotional_state in ['greed', 'fear', 'anxious']:
            score -= 0.1
            notes.append('❌ 情绪状态不佳')
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        
        return {
            'score': score,
            'notes': notes,
            'grade': self._score_to_grade(score)
        }
    
    def _evaluate_exit_quality(self,
                              trade_data: Dict,
                              trade_result: Dict) -> Dict:
        """评估出场质量"""
        score = 0.7  # 基础分数
        notes = []
        
        # 检查止损执行
        stop_loss = trade_data.get('stop_loss')
        exit_reason = trade_data.get('exit_reason', '')
        
        if stop_loss:
            if 'stop_loss' in exit_reason.lower():
                score += 0.1
                notes.append('✅ 按计划执行止损')
            else:
                # 检查是否应该触发止损但未触发
                direction = trade_result['direction']
                entry_price = trade_result['entry_price']
                exit_price = trade_result['exit_price']
                
                if direction == 'long' and exit_price < stop_loss:
                    score -= 0.2
                    notes.append('❌ 未按计划执行止损（价格跌破止损位）')
                elif direction == 'short' and exit_price > stop_loss:
                    score -= 0.2
                    notes.append('❌ 未按计划执行止损（价格突破止损位）')
                else:
                    score += 0.05
                    notes.append('✅ 止损设置合理')
        else:
            score -= 0.15
            notes.append('❌ 未设置止损')
        
        # 检查止盈执行
        take_profit = trade_data.get('take_profit')
        
        if take_profit:
            if 'take_profit' in exit_reason.lower():
                score += 0.1
                notes.append('✅ 按计划执行止盈')
            else:
                # 检查是否错过止盈机会
                direction = trade_result['direction']
                entry_price = trade_result['entry_price']
                exit_price = trade_result['exit_price']
                
                if direction == 'long' and exit_price < take_profit:
                    score -= 0.05
                    notes.append('⚠️ 可能过早出场，错过部分利润')
                elif direction == 'short' and exit_price > take_profit:
                    score -= 0.05
                    notes.append('⚠️ 可能过早出场，错过部分利润')
        else:
            score -= 0.05
            notes.append('⚠️ 未设置止盈目标')
        
        # 检查出场理由
        if exit_reason and len(exit_reason) > 5:
            score += 0.05
            notes.append('✅ 出场理由记录详细')
        else:
            score -= 0.05
            notes.append('⚠️ 出场理由记录不足')
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        
        return {
            'score': score,
            'notes': notes,
            'grade': self._score_to_grade(score)
        }
    
    def _evaluate_risk_management_quality(self,
                                         trade_data: Dict,
                                         trade_result: Dict) -> Dict:
        """评估风险管理质量"""
        score = 0.7  # 基础分数
        notes = []
        
        # 检查仓位大小
        position_size = trade_result['position_size']
        account_size = trade_data.get('account_size', 10000)
        
        if account_size > 0:
            position_percentage = (position_size * trade_result['entry_price']) / account_size
            
            if position_percentage <= 0.1:  # 10%以内
                score += 0.1
                notes.append('✅ 仓位大小合理')
            elif position_percentage <= 0.2:  # 20%以内
                score += 0.05
                notes.append('⚠️ 仓位偏大，需注意风险')
            else:
                score -= 0.15
                notes.append('❌ 仓位过大，风险过高')
        
        # 检查风险回报比
        planned_rr = trade_result.get('planned_risk_reward_ratio', 0)
        actual_rr = trade_result.get('actual_risk_reward_ratio', 0)
        
        if planned_rr >= 1.5:
            score += 0.1
            notes.append('✅ 计划风险回报比优秀')
        elif planned_rr >= 1.0:
            score += 0.05
            notes.append('✅ 计划风险回报比合格')
        else:
            score -= 0.1
            notes.append('❌ 计划风险回报比不足')
        
        # 检查实际风险控制
        pnl_percentage = trade_result['pnl_percentage']
        max_risk_per_trade = trade_data.get('max_risk_per_trade', 0.02)
        
        if abs(pnl_percentage) <= max_risk_per_trade:
            score += 0.05
            notes.append('✅ 实际风险控制在计划内')
        elif pnl_percentage > max_risk_per_trade:
            score += 0.1  # 盈利超过风险限制是好事
            notes.append('✅ 盈利超过预期')
        else:
            score -= 0.15
            notes.append('❌ 实际亏损超过风险限制')
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        
        return {
            'score': score,
            'notes': notes,
            'grade': self._score_to_grade(score)
        }
    
    def _evaluate_execution_quality(self,
                                   trade_data: Dict,
                                   trade_result: Dict) -> Dict:
        """评估执行质量"""
        score = 0.7  # 基础分数
        notes = []
        
        # 检查计划vs执行
        planned_vs_actual = trade_data.get('planned_vs_actual_match', 0.7)
        
        if planned_vs_actual >= 0.9:
            score += 0.15
            notes.append('✅ 计划执行匹配度很高')
        elif planned_vs_actual >= 0.7:
            score += 0.05
            notes.append('✅ 计划执行匹配度良好')
        elif planned_vs_actual >= 0.5:
            score -= 0.05
            notes.append('⚠️ 计划执行匹配度一般')
        else:
            score -= 0.15
            notes.append('❌ 计划执行匹配度差')
        
        # 检查交易记录完整性
        required_fields = ['entry_time', 'exit_time', 'entry_price', 'exit_price',
                          'position_size', 'direction', 'instrument', 'exit_reason']
        
        missing_fields = [field for field in required_fields if field not in trade_data]
        
        if not missing_fields:
            score += 0.1
            notes.append('✅ 交易记录完整')
        elif len(missing_fields) <= 2:
            score -= 0.05
            notes.append('⚠️ 交易记录缺少部分信息')
        else:
            score -= 0.15
            notes.append('❌ 交易记录不完整')
        
        # 检查交易后分析
        post_trade_analysis = trade_data.get('post_trade_analysis', '')
        
        if post_trade_analysis and len(post_trade_analysis) > 20:
            score += 0.05
            notes.append('✅ 有交易后分析')
        else:
            score -= 0.05
            notes.append('⚠️ 缺乏交易后分析')
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        
        return {
            'score': score,
            'notes': notes,
            'grade': self._score_to_grade(score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """分数转等级"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.85:
            return 'A'
        elif score >= 0.8:
            return 'A-'
        elif score >= 0.75:
            return 'B+'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.65:
            return 'B-'
        elif score >= 0.6:
            return 'C+'
        elif score >= 0.55:
            return 'C'
        elif score >= 0.5:
            return 'C-'
        elif score >= 0.4:
            return 'D'
        else:
            return 'F'
    
    def _identify_quality_improvement_areas(self,
                                           entry_quality: Dict,
                                           exit_quality: Dict,
                                           risk_quality: Dict,
                                           execution_quality: Dict) -> List[Dict]:
        """识别质量改进领域"""
        improvement_areas = []
        
        # 入场质量改进
        if entry_quality['score'] < 0.7:
            improvement_areas.append({
                'area': 'entry_quality',
                'priority': 'high' if entry_quality['score'] < 0.6 else 'medium',
                'description': '入场质量需要改进',
                'suggestions': [
                    '制定更明确的入场条件',
                    '等待更好的入场时机',
                    '控制入场时的情绪状态'
                ],
                'current_score': entry_quality['score'],
                'target_score': 0.8
            })
        
        # 出场质量改进
        if exit_quality['score'] < 0.7:
            improvement_areas.append({
                'area': 'exit_quality',
                'priority': 'high' if exit_quality['score'] < 0.6 else 'medium',
                'description': '出场质量需要改进',
                'suggestions': [
                    '严格执行止损止盈计划',
                    '记录详细的出场理由',
                    '避免情绪化出场'
                ],
                'current_score': exit_quality['score'],
                'target_score': 0.8
            })
        
        # 风险管理质量改进
        if risk_quality['score'] < 0.7:
            improvement_areas.append({
                'area': 'risk_management_quality',
                'priority': 'high' if risk_quality['score'] < 0.6 else 'medium',
                'description': '风险管理需要改进',
                'suggestions': [
                    '控制仓位大小',
                    '确保风险回报比至少1.5:1',
                    '严格遵守风险限制'
                ],
                'current_score': risk_quality['score'],
                'target_score': 0.8
            })
        
        # 执行质量改进
        if execution_quality['score'] < 0.7:
            improvement_areas.append({
                'area': 'execution_quality',
                'priority': 'high' if execution_quality['score'] < 0.6 else 'medium',
                'description': '执行质量需要改进',
                'suggestions': [
                    '提高计划执行匹配度',
                    '完善交易记录',
                    '进行交易后分析'
                ],
                'current_score': execution_quality['score'],
                'target_score': 0.8
            })
        
        return improvement_areas
    
    def _identify_trade_errors(self,
                              trade_data: Dict,
                              trade_result: Dict) -> List[Dict]:
        """识别交易错误"""
        errors = []
        
        # 检查入场错误
        entry_errors = self._identify_entry_errors(trade_data, trade_result)
        errors.extend(entry_errors)
        
        # 检查出场错误
        exit_errors = self._identify_exit_errors(trade_data, trade_result)
        errors.extend(exit_errors)
        
        # 检查风险管理错误
        risk_errors = self._identify_risk_management_errors(trade_data, trade_result)
        errors.extend(risk_errors)
        
        # 检查心理错误
        psychological_errors = self._identify_psychological_errors(trade_data, trade_result)
        errors.extend(psychological_errors)
        
        # 检查计划执行错误
        plan_execution_errors = self._identify_plan_execution_errors(trade_data, trade_result)
        errors.extend(plan_execution_errors)
        
        return errors
    
    def _identify_entry_errors(self,
                              trade_data: Dict,
                              trade_result: Dict) -> List[Dict]:
        """识别入场错误"""
        errors = []
        
        # 检查是否有交易计划
        if not trade_data.get('trading_plan_id'):
            errors.append({
                'category': 'entry_errors',
                'subcategory': 'poor_entry_timing',
                'description': '无交易计划，入场时机可能不佳',
                'severity': 'medium',
                'evidence': '缺少trading_plan_id字段'
            })
        
        # 检查入场条件
        entry_conditions = trade_data.get('entry_conditions', [])
        if len(entry_conditions) < 2:
            errors.append({
                'category': 'entry_errors',
                'subcategory': 'ignored_entry_conditions',
                'description': '入场条件不足，可能缺乏充分确认',
                'severity': 'low',
                'evidence': f'只有{len(entry_conditions)}个入场条件'
            })
        
        # 检查情绪状态
        emotional_state = trade_data.get('emotional_state', 'calm')
        if emotional_state in ['greed', 'fear', 'anxious', 'excited']:
            errors.append({
                'category': 'entry_errors',
                'subcategory': 'emotional_entry',
                'description': f'情绪化入场（{emotional_state}）',
                'severity': 'high',
                'evidence': f'情绪状态: {emotional_state}'
            })
        
        # 检查是否过度交易（通过交易频率，需要更多上下文，这里简化）
        recent_trades_count = trade_data.get('recent_trades_count', 0)
        if recent_trades_count > 5:
            errors.append({
                'category': 'entry_errors',
                'subcategory': 'overtrading',
                'description': '可能过度交易',
                'severity': 'medium',
                'evidence': f'近期交易次数: {recent_trades_count}'
            })
        
        return errors
    
    def _identify_exit_errors(self,
                             trade_data: Dict,
                             trade_result: Dict) -> List[Dict]:
        """识别出场错误"""
        errors = []
        
        # 检查止损执行
        stop_loss = trade_data.get('stop_loss')
        exit_reason = trade_data.get('exit_reason', '')
        direction = trade_result['direction']
        entry_price = trade_result['entry_price']
        exit_price = trade_result['exit_price']
        
        if stop_loss:
            # 检查是否应该触发止损但未触发
            if direction == 'long' and exit_price < stop_loss and 'stop_loss' not in exit_reason.lower():
                errors.append({
                    'category': 'exit_errors',
                    'subcategory': 'ignored_stop_loss',
                    'description': '价格跌破止损位但未执行止损',
                    'severity': 'high',
                    'evidence': f'止损位: {stop_loss}, 出场价: {exit_price}, 出场理由: {exit_reason}'
                })
            elif direction == 'short' and exit_price > stop_loss and 'stop_loss' not in exit_reason.lower():
                errors.append({
                    'category': 'exit_errors',
                    'subcategory': 'ignored_stop_loss',
                    'description': '价格突破止损位但未执行止损',
                    'severity': 'high',
                    'evidence': f'止损位: {stop_loss}, 出场价: {exit_price}, 出场理由: {exit_reason}'
                })
        else:
            errors.append({
                'category': 'exit_errors',
                'subcategory': 'no_stop_loss',
                'description': '未设置止损',
                'severity': 'critical',
                'evidence': '缺少stop_loss字段'
            })
        
        # 检查出场时机
        holding_period = trade_result['holding_period_hours']
        
        # 根据交易类型判断持仓时间是否合理
        trade_type = trade_data.get('trade_type', 'swing')
        
        if trade_type == 'scalping' and holding_period > 1:
            errors.append({
                'category': 'exit_errors',
                'subcategory': 'late_exit',
                'description': '剥头皮交易持仓时间过长',
                'severity': 'medium',
                'evidence': f'持仓时间: {holding_period:.1f}小时'
            })
        elif trade_type == 'swing' and holding_period < 4:
            errors.append({
                'category': 'exit_errors',
                'subcategory': 'early_exit',
                'description': '摆动交易持仓时间过短',
                'severity': 'medium',
                'evidence': f'持仓时间: {holding_period:.1f}小时'
            })
        
        return errors
    
    def _identify_risk_management_errors(self,
                                        trade_data: Dict,
                                        trade_result: Dict) -> List[Dict]:
        """识别风险管理错误"""
        errors = []
        
        # 检查仓位大小
        position_size = trade_result['position_size']
        account_size = trade_data.get('account_size', 10000)
        
        if account_size > 0:
            position_value = position_size * trade_result['entry_price']
            position_percentage = position_value / account_size
            
            if position_percentage > 0.2:  # 超过20%
                errors.append({
                    'category': 'risk_management_errors',
                    'subcategory': 'oversized_position',
                    'description': '仓位过大，风险过高',
                    'severity': 'high',
                    'evidence': f'仓位比例: {position_percentage:.1%}'
                })
            elif position_percentage > 0.1:  # 超过10%
                errors.append({
                    'category': 'risk_management_errors',
                    'subcategory': 'oversized_position',
                    'description': '仓位偏大',
                    'severity': 'medium',
                    'evidence': f'仓位比例: {position_percentage:.1%}'
                })
        
        # 检查风险回报比
        planned_rr = trade_result.get('planned_risk_reward_ratio', 0)
        
        if planned_rr < 1.0:
            errors.append({
                'category': 'risk_management_errors',
                'subcategory': 'no_position_sizing',
                'description': '计划风险回报比不足1:1',
                'severity': 'medium',
                'evidence': f'计划风险回报比: {planned_rr:.1f}:1'
            })
        
        # 检查实际风险控制
        pnl_percentage = trade_result['pnl_percentage']
        max_risk_per_trade = trade_data.get('max_risk_per_trade', 0.02)
        
        if pnl_percentage < -max_risk_per_trade:
            errors.append({
                'category': 'risk_management_errors',
                'subcategory': 'ignored_risk_limits',
                'description': '实际亏损超过风险限制',
                'severity': 'high',
                'evidence': f'亏损: {pnl_percentage:.1%}, 风险限制: {max_risk_per_trade:.1%}'
            })
        
        return errors
    
    def _identify_psychological_errors(self,
                                      trade_data: Dict,
                                      trade_result: Dict) -> List[Dict]:
        """识别心理错误"""
        errors = []
        
        # 检查情绪状态
        emotional_state = trade_data.get('emotional_state', 'calm')
        
        if emotional_state == 'greed':
            errors.append({
                'category': 'psychological_errors',
                'subcategory': 'greed',
                'description': '贪婪导致交易决策偏差',
                'severity': 'medium',
                'evidence': f'情绪状态: {emotional_state}'
            })
        elif emotional_state == 'fear':
            errors.append({
                'category': 'psychological_errors',
                'subcategory': 'fear',
                'description': '恐惧影响交易执行',
                'severity': 'medium',
                'evidence': f'情绪状态: {emotional_state}'
            })
        elif emotional_state == 'overconfidence':
            errors.append({
                'category': 'psychological_errors',
                'subcategory': 'overconfidence',
                'description': '过度自信可能导致风险忽视',
                'severity': 'medium',
                'evidence': f'情绪状态: {emotional_state}'
            })
        
        # 检查是否持仓亏损但期待反转（希望）
        if trade_result['pnl'] < 0 and trade_data.get('hoping_for_reversal', False):
            errors.append({
                'category': 'psychological_errors',
                'subcategory': 'hope',
                'description': '持仓亏损但期待反转（希望偏差）',
                'severity': 'high',
                'evidence': 'hoping_for_reversal字段为True'
            })
        
        # 检查是否有后悔情绪
        if trade_data.get('regret_previous_trade', False):
            errors.append({
                'category': 'psychological_errors',
                'subcategory': 'regret',
                'description': '后悔情绪影响当前交易',
                'severity': 'medium',
                'evidence': 'regret_previous_trade字段为True'
            })
        
        return errors
    
    def _identify_plan_execution_errors(self,
                                       trade_data: Dict,
                                       trade_result: Dict) -> List[Dict]:
        """识别计划执行错误"""
        errors = []
        
        # 检查是否有交易计划
        if not trade_data.get('trading_plan_id'):
            errors.append({
                'category': 'plan_execution_errors',
                'subcategory': 'no_trading_plan',
                'description': '无交易计划',
                'severity': 'high',
                'evidence': '缺少trading_plan_id字段'
            })
        
        # 检查计划vs执行匹配度
        planned_vs_actual = trade_data.get('planned_vs_actual_match', 0.7)
        
        if planned_vs_actual < 0.5:
            errors.append({
                'category': 'plan_execution_errors',
                'subcategory': 'ignored_plan',
                'description': '计划执行匹配度差',
                'severity': 'high',
                'evidence': f'匹配度: {planned_vs_actual:.0%}'
            })
        elif planned_vs_actual < 0.7:
            errors.append({
                'category': 'plan_execution_errors',
                'subcategory': 'ignored_plan',
                'description': '计划执行匹配度一般',
                'severity': 'medium',
                'evidence': f'匹配度: {planned_vs_actual:.0%}'
            })
        
        # 检查准备情况
        preparation_score = trade_data.get('preparation_score', 0.7)
        
        if preparation_score < 0.6:
            errors.append({
                'category': 'plan_execution_errors',
                'subcategory': 'poor_preparation',
                'description': '交易准备不足',
                'severity': 'medium',
                'evidence': f'准备分数: {preparation_score:.0%}'
            })
        
        # 检查是否分心
        if trade_data.get('distracted_during_trade', False):
            errors.append({
                'category': 'plan_execution_errors',
                'subcategory': 'distracted_trading',
                'description': '交易时分心',
                'severity': 'low',
                'evidence': 'distracted_during_trade字段为True'
            })
        
        # 检查是否疲劳交易
        if trade_data.get('fatigued_during_trade', False):
            errors.append({
                'category': 'plan_execution_errors',
                'subcategory': 'fatigue_trading',
                'description': '疲劳状态下交易',
                'severity': 'medium',
                'evidence': 'fatigued_during_trade字段为True'
            })
        
        return errors
    
    def _generate_trade_tags(self,
                            trade_data: Dict,
                            trade_result: Dict,
                            trade_quality: Dict,
                            trade_errors: List[Dict]) -> List[str]:
        """生成交易标签"""
        tags = []
        
        # 结果标签
        if trade_result['win']:
            tags.append('盈利')
            if trade_result['significant_win']:
                tags.append('大赚')
        else:
            tags.append('亏损')
            if trade_result['significant_loss']:
                tags.append('大亏')
        
        if trade_result['breakeven']:
            tags.append('盈亏平衡')
        
        # 质量标签
        quality_grade = trade_quality['quality_grade']
        if quality_grade in ['excellent', 'good']:
            tags.append('高质量')
        elif quality_grade in ['poor', 'very_poor']:
            tags.append('低质量')
        
        # 错误标签
        if trade_errors:
            error_count = len(trade_errors)
            if error_count >= 3:
                tags.append('多错误')
            elif error_count >= 1:
                tags.append('有错误')
            
            # 根据错误类型添加标签
            error_categories = set(error['category'] for error in trade_errors)
            for category in error_categories:
                if 'psychological' in category:
                    tags.append('心理问题')
                elif 'risk' in category:
                    tags.append('风控问题')
                elif 'entry' in category:
                    tags.append('入场问题')
                elif 'exit' in category:
                    tags.append('出场问题')
        
        # 交易类型标签
        trade_type = trade_data.get('trade_type', 'swing')
        tags.append(trade_type)
        
        # 时间标签
        entry_time = trade_data['entry_time']
        if isinstance(entry_time, datetime):
            hour = entry_time.hour
            if 9 <= hour <= 11:
                tags.append('上午交易')
            elif 13 <= hour <= 15:
                tags.append('下午交易')
            elif hour >= 21 or hour <= 3:
                tags.append('夜间交易')
        
        # 情绪标签
        emotional_state = trade_data.get('emotional_state', 'calm')
        if emotional_state != 'calm':
            tags.append(f'情绪化({emotional_state})')
        
        return tags
    
    def _update_performance_history(self, trade_record: Dict):
        """更新绩效历史"""
        performance_update = {
            'timestamp': datetime.now(),
            'trade_id': trade_record['trade_id'],
            'pnl': trade_record['trade_result']['pnl'],
            'pnl_percentage': trade_record['trade_result']['pnl_percentage'],
            'quality_score': trade_record['trade_quality']['overall_score'],
            'error_count': len(trade_record['trade_errors']),
            'holding_period': trade_record['trade_result']['holding_period_hours'],
            'tags': trade_record['tags']
        }
        
        self.performance_history.append(performance_update)
    
    def analyze_trading_performance(self,
                                   time_period: str = 'all',
                                   include_details: bool = True) -> Dict:
        """
        分析交易绩效
        
        参数:
            time_period: 时间周期（'all', 'month', 'week', 'today'）
            include_details: 是否包含详细分析
            
        返回:
            绩效分析结果
        """
        if not self.trades_log:
            return {'error': '无交易记录'}
        
        # 筛选指定时间段的交易
        filtered_trades = self._filter_trades_by_time_period(time_period)
        
        if not filtered_trades:
            return {'error': f'指定时间段内无交易: {time_period}'}
        
        # 计算基本绩效指标
        basic_metrics = self._calculate_basic_performance_metrics(filtered_trades)
        
        # 计算高级绩效指标
        advanced_metrics = self._calculate_advanced_performance_metrics(filtered_trades)
        
        # 识别交易模式
        trading_patterns = self._identify_trading_patterns(filtered_trades) if include_details else {}
        
        # 分析错误模式
        error_patterns = self._analyze_error_patterns(filtered_trades) if include_details else {}
        
        # 生成改进建议
        improvement_suggestions = self._generate_improvement_suggestions(
            basic_metrics, advanced_metrics, trading_patterns, error_patterns
        ) if include_details else []
        
        result = {
            'time_period': time_period,
            'analysis_time': datetime.now(),
            'trade_count': len(filtered_trades),
            'time_range': {
                'start': filtered_trades[0]['log_time'] if filtered_trades else None,
                'end': filtered_trades[-1]['log_time'] if filtered_trades else None
            },
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'performance_summary': self._generate_performance_summary(basic_metrics, advanced_metrics)
        }
        
        if include_details:
            result.update({
                'trading_patterns': trading_patterns,
                'error_patterns': error_patterns,
                'improvement_suggestions': improvement_suggestions,
                'detailed_analysis': self._generate_detailed_analysis(filtered_trades)
            })
        
        self.analysis_reports.append(result)
        return result
    
    def _filter_trades_by_time_period(self, time_period: str) -> List[Dict]:
        """按时间段筛选交易"""
        if time_period == 'all':
            return self.trades_log
        
        now = datetime.now()
        
        if time_period == 'month':
            cutoff_date = now - timedelta(days=30)
        elif time_period == 'week':
            cutoff_date = now - timedelta(days=7)
        elif time_period == 'today':
            cutoff_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return self.trades_log
        
        filtered_trades = [
            trade for trade in self.trades_log
            if trade['log_time'] >= cutoff_date
        ]
        
        return filtered_trades
    
    def _calculate_basic_performance_metrics(self, trades: List[Dict]) -> Dict:
        """计算基本绩效指标"""
        if not trades:
            return {}
        
        # 提取交易结果
        trade_results = [trade['trade_result'] for trade in trades]
        pnls = [result['pnl'] for result in trade_results]
        pnl_percentages = [result['pnl_percentage'] for result in trade_results]
        
        # 计算基本指标
        winning_trades = [result for result in trade_results if result['win']]
        losing_trades = [result for result in trade_results if not result['win'] and not result['breakeven']]
        breakeven_trades = [result for result in trade_results if result['breakeven']]
        
        total_trades = len(trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        breakeven_count = len(breakeven_trades)
        
        # 胜率
        win_rate = winning_count / total_trades if total_trades > 0 else 0
        
        # 平均盈亏
        avg_win = np.mean([result['pnl'] for result in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([result['pnl'] for result in losing_trades]) if losing_trades else 0
        
        # 盈亏比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0
        
        # 总盈亏
        total_pnl = sum(pnls)
        total_pnl_percentage = sum(pnl_percentages)
        
        # 最大单笔盈利/亏损
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0
        
        # 平均持仓时间
        holding_periods = [result['holding_period_hours'] for result in trade_results]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'breakeven_trades': breakeven_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'total_pnl_percentage': total_pnl_percentage,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_holding_period_hours': avg_holding_period,
            'performance_score': self._calculate_performance_score(
                win_rate, profit_factor, total_pnl_percentage
            )
        }
    
    def _calculate_advanced_performance_metrics(self, trades: List[Dict]) -> Dict:
        """计算高级绩效指标"""
        if len(trades) < 5:
            return {'insufficient_data': True}
        
        trade_results = [trade['trade_result'] for trade in trades]
        pnls = [result['pnl'] for result in trade_results]
        pnl_percentages = [result['pnl_percentage'] for result in trade_results]
        
        # 计算夏普比率（简化版）
        if len(pnl_percentages) >= 2:
            returns_mean = np.mean(pnl_percentages)
            returns_std = np.std(pnl_percentages)
            sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 计算最大回撤
        cumulative_returns = np.cumsum(pnl_percentages)
        
        if len(cumulative_returns) > 0:
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            max_drawdown_percentage = max_drawdown
        else:
            max_drawdown = 0
            max_drawdown_percentage = 0
        
        # 计算连胜/连败
        winning_streaks = []
        losing_streaks = []
        current_streak = 0
        current_streak_type = None
        
        for result in trade_results:
            if result['win']:
                if current_streak_type == 'win':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        if current_streak_type == 'win':
                            winning_streaks.append(current_streak)
                        else:
                            losing_streaks.append(current_streak)
                    current_streak = 1
                    current_streak_type = 'win'
            elif not result['breakeven']:
                if current_streak_type == 'loss':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        if current_streak_type == 'win':
                            winning_streaks.append(current_streak)
                        else:
                            losing_streaks.append(current_streak)
                    current_streak = 1
                    current_streak_type = 'loss'
        
        # 添加最后一个连胜/连败
        if current_streak > 0:
            if current_streak_type == 'win':
                winning_streaks.append(current_streak)
            else:
                losing_streaks.append(current_streak)
        
        max_winning_streak = max(winning_streaks) if winning_streaks else 0
        max_losing_streak = max(losing_streaks) if losing_streaks else 0
        
        # 计算交易一致性
        if len(pnls) >= 3:
            consistency_score = 1 - (np.std(pnl_percentages) / abs(np.mean(pnl_percentages))) if np.mean(pnl_percentages) != 0 else 0
            consistency_score = max(0, min(1, consistency_score))
        else:
            consistency_score = 0
        
        # 计算风险调整后收益
        risk_adjusted_return = total_return = sum(pnl_percentages)
        if max_drawdown_percentage > 0:
            risk_adjusted_return = total_return / max_drawdown_percentage
        else:
            risk_adjusted_return = total_return
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_percentage': max_drawdown_percentage,
            'max_winning_streak': max_winning_streak,
            'max_losing_streak': max_losing_streak,
            'consistency_score': consistency_score,
            'risk_adjusted_return': risk_adjusted_return,
            'return_per_trade': np.mean(pnl_percentages) if pnl_percentages else 0,
            'std_dev_per_trade': np.std(pnl_percentages) if len(pnl_percentages) >= 2 else 0,
            'skewness': self._calculate_skewness(pnl_percentages) if len(pnl_percentages) >= 3 else 0,
            'kurtosis': self._calculate_kurtosis(pnl_percentages) if len(pnl_percentages) >= 4 else 0
        }
    
    def _calculate_performance_score(self,
                                   win_rate: float,
                                   profit_factor: float,
                                   total_return: float) -> float:
        """计算绩效分数"""
        # 标准化指标
        win_rate_score = min(1.0, win_rate / 0.6)  # 60%胜率为满分
        profit_factor_score = min(1.0, profit_factor / 2.0)  # 2.0盈亏比为满分
        return_score = min(1.0, total_return / 0.1)  # 10%总回报为满分
        
        # 加权平均
        performance_score = (
            win_rate_score * 0.4 +
            profit_factor_score * 0.4 +
            return_score * 0.2
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skewness = np.mean([((x - mean) / std) ** 3 for x in data])
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """计算峰度"""
        if len(data) < 4:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        kurtosis = np.mean([((x - mean) / std) ** 4 for x in data]) - 3
        return kurtosis
    
    def _identify_trading_patterns(self, trades: List[Dict]) -> Dict:
        """识别交易模式"""
        if len(trades) < 10:
            return {'insufficient_data': True}
        
        patterns = {
            'time_patterns': self._analyze_time_patterns(trades),
            'instrument_patterns': self._analyze_instrument_patterns(trades),
            'direction_patterns': self._analyze_direction_patterns(trades),
            'quality_patterns': self._analyze_quality_patterns(trades),
            'emotional_patterns': self._analyze_emotional_patterns(trades)
        }
        
        return patterns
    
    def _analyze_time_patterns(self, trades: List[Dict]) -> Dict:
        """分析时间模式"""
        time_patterns = {
            'hourly_distribution': {},
            'weekday_distribution': {},
            'best_performing_hours': [],
            'worst_performing_hours': []
        }
        
        # 按小时分析
        hourly_pnl = {}
        hourly_count = {}
        
        for trade in trades:
            entry_time = trade['entry_time']
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            
            hour = entry_time.hour
            pnl = trade['trade_result']['pnl']
            
            if hour not in hourly_pnl:
                hourly_pnl[hour] = 0
                hourly_count[hour] = 0
            
            hourly_pnl[hour] += pnl
            hourly_count[hour] += 1
        
        # 计算每小时平均盈亏
        hourly_avg_pnl = {}
        for hour in hourly_pnl:
            if hourly_count[hour] > 0:
                hourly_avg_pnl[hour] = hourly_pnl[hour] / hourly_count[hour]
        
        # 找出最佳和最差时段
        if hourly_avg_pnl:
            sorted_hours = sorted(hourly_avg_pnl.items(), key=lambda x: x[1], reverse=True)
            
            best_hours = sorted_hours[:3]
            worst_hours = sorted_hours[-3:] if len(sorted_hours) >= 3 else sorted_hours
            
            time_patterns['best_performing_hours'] = [
                {'hour': hour, 'avg_pnl': avg_pnl, 'trade_count': hourly_count[hour]}
                for hour, avg_pnl in best_hours
            ]
            
            time_patterns['worst_performing_hours'] = [
                {'hour': hour, 'avg_pnl': avg_pnl, 'trade_count': hourly_count[hour]}
                for hour, avg_pnl in worst_hours
            ]
        
        # 按星期分析
        weekday_pnl = {}
        weekday_count = {}
        
        for trade in trades:
            entry_time = trade['entry_time']
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            
            weekday = entry_time.weekday()  # 0=Monday, 6=Sunday
            pnl = trade['trade_result']['pnl']
            
            if weekday not in weekday_pnl:
                weekday_pnl[weekday] = 0
                weekday_count[weekday] = 0
            
            weekday_pnl[weekday] += pnl
            weekday_count[weekday] += 1
        
        # 转换为可读格式
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i in range(7):
            if i in weekday_pnl:
                time_patterns['weekday_distribution'][weekday_names[i]] = {
                    'total_pnl': weekday_pnl[i],
                    'trade_count': weekday_count[i],
                    'avg_pnl': weekday_pnl[i] / weekday_count[i] if weekday_count[i] > 0 else 0
                }
        
        return time_patterns
    
    def _analyze_instrument_patterns(self, trades: List[Dict]) -> Dict:
        """分析交易品种模式"""
        instrument_patterns = {
            'instrument_performance': {},
            'best_performing_instruments': [],
            'worst_performing_instruments': []
        }
        
        instrument_pnl = {}
        instrument_count = {}
        
        for trade in trades:
            instrument = trade['instrument']
            pnl = trade['trade_result']['pnl']
            
            if instrument not in instrument_pnl:
                instrument_pnl[instrument] = 0
                instrument_count[instrument] = 0
            
            instrument_pnl[instrument] += pnl
            instrument_count[instrument] += 1
        
        # 计算每个品种的平均盈亏
        instrument_avg_pnl = {}
        for instrument in instrument_pnl:
            if instrument_count[instrument] > 0:
                instrument_avg_pnl[instrument] = instrument_pnl[instrument] / instrument_count[instrument]
        
        # 找出最佳和最差品种
        if instrument_avg_pnl:
            sorted_instruments = sorted(instrument_avg_pnl.items(), key=lambda x: x[1], reverse=True)
            
            best_instruments = sorted_instruments[:3]
            worst_instruments = sorted_instruments[-3:] if len(sorted_instruments) >= 3 else sorted_instruments
            
            instrument_patterns['best_performing_instruments'] = [
                {'instrument': instrument, 'avg_pnl': avg_pnl, 'trade_count': instrument_count[instrument]}
                for instrument, avg_pnl in best_instruments
            ]
            
            instrument_patterns['worst_performing_instruments'] = [
                {'instrument': instrument, 'avg_pnl': avg_pnl, 'trade_count': instrument_count[instrument]}
                for instrument, avg_pnl in worst_instruments
            ]
        
        # 记录所有品种表现
        for instrument in instrument_pnl:
            instrument_patterns['instrument_performance'][instrument] = {
                'total_pnl': instrument_pnl[instrument],
                'trade_count': instrument_count[instrument],
                'avg_pnl': instrument_avg_pnl.get(instrument, 0),
                'win_rate': self._calculate_instrument_win_rate(trades, instrument)
            }
        
        return instrument_patterns
    
    def _calculate_instrument_win_rate(self, trades: List[Dict], instrument: str) -> float:
        """计算特定品种的胜率"""
        instrument_trades = [trade for trade in trades if trade['instrument'] == instrument]
        
        if not instrument_trades:
            return 0
        
        winning_trades = [trade for trade in instrument_trades if trade['trade_result']['win']]
        
        return len(winning_trades) / len(instrument_trades)
    
    def _analyze_direction_patterns(self, trades: List[Dict]) -> Dict:
        """分析交易方向模式"""
        direction_patterns = {
            'long_performance': {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0},
            'short_performance': {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0},
            'preferred_direction': 'unknown'
        }
        
        long_trades = [trade for trade in trades if trade['direction'] == 'long']
        short_trades = [trade for trade in trades if trade['direction'] == 'short']
        
        # 分析多头交易
        if long_trades:
            long_pnls = [trade['trade_result']['pnl'] for trade in long_trades]
            long_wins = [trade for trade in long_trades if trade['trade_result']['win']]
            
            direction_patterns['long_performance'] = {
                'total_pnl': sum(long_pnls),
                'trade_count': len(long_trades),
                'win_rate': len(long_wins) / len(long_trades),
                'avg_pnl': sum(long_pnls) / len(long_trades) if long_trades else 0,
                'avg_holding_hours': np.mean([trade['trade_result']['holding_period_hours'] for trade in long_trades]) if long_trades else 0
            }
        
        # 分析空头交易
        if short_trades:
            short_pnls = [trade['trade_result']['pnl'] for trade in short_trades]
            short_wins = [trade for trade in short_trades if trade['trade_result']['win']]
            
            direction_patterns['short_performance'] = {
                'total_pnl': sum(short_pnls),
                'trade_count': len(short_trades),
                'win_rate': len(short_wins) / len(short_trades),
                'avg_pnl': sum(short_pnls) / len(short_trades) if short_trades else 0,
                'avg_holding_hours': np.mean([trade['trade_result']['holding_period_hours'] for trade in short_trades]) if short_trades else 0
            }
        
        # 确定偏好方向
        if long_trades and short_trades:
            long_avg = direction_patterns['long_performance']['avg_pnl']
            short_avg = direction_patterns['short_performance']['avg_pnl']
            
            if long_avg > short_avg:
                direction_patterns['preferred_direction'] = 'long'
            elif short_avg > long_avg:
                direction_patterns['preferred_direction'] = 'short'
            else:
                direction_patterns['preferred_direction'] = 'equal'
        elif long_trades:
            direction_patterns['preferred_direction'] = 'long'
        elif short_trades:
            direction_patterns['preferred_direction'] = 'short'
        
        return direction_patterns
    
    def _analyze_quality_patterns(self, trades: List[Dict]) -> Dict:
        """分析质量模式"""
        quality_patterns = {
            'quality_distribution': {},
            'quality_vs_performance': {},
            'common_quality_issues': []
        }
        
        # 质量等级分布
        quality_grades = {}
        for trade in trades:
            grade = trade['trade_quality']['quality_grade']
            quality_grades[grade] = quality_grades.get(grade, 0) + 1
        
        for grade, count in quality_grades.items():
            percentage = count / len(trades) if trades else 0
            quality_patterns['quality_distribution'][grade] = {
                'count': count,
                'percentage': percentage
            }
        
        # 质量vs绩效关系
        quality_performance = {}
        for trade in trades:
            grade = trade['trade_quality']['quality_grade']
            pnl = trade['trade_result']['pnl']
            
            if grade not in quality_performance:
                quality_performance[grade] = {'total_pnl': 0, 'count': 0}
            
            quality_performance[grade]['total_pnl'] += pnl
            quality_performance[grade]['count'] += 1
        
        for grade, data in quality_performance.items():
            if data['count'] > 0:
                quality_patterns['quality_vs_performance'][grade] = {
                    'avg_pnl': data['total_pnl'] / data['count'],
                    'trade_count': data['count']
                }
        
        # 常见质量问题
        all_errors = []
        for trade in trades:
            all_errors.extend(trade['trade_errors'])
        
        if all_errors:
            error_counts = {}
            for error in all_errors:
                error_key = f"{error['category']}.{error['subcategory']}"
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            # 取前5个最常见错误
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for error_key, count in sorted_errors:
                category, subcategory = error_key.split('.')
                
                # 从错误分类中获取描述
                category_info = self.error_categories.get(category, {})
                subcategory_info = category_info.get('subcategories', {}).get(subcategory, subcategory)
                
                quality_patterns['common_quality_issues'].append({
                    'category': category,
                    'subcategory': subcategory,
                    'description': subcategory_info,
                    'count': count,
                    'percentage': count / len(all_errors) if all_errors else 0
                })
        
        return quality_patterns
    
    def _analyze_emotional_patterns(self, trades: List[Dict]) -> Dict:
        """分析情绪模式"""
        emotional_patterns = {
            'emotional_state_distribution': {},
            'emotion_vs_performance': {},
            'common_emotional_issues': []
        }
        
        # 情绪状态分布
        emotional_states = {}
        for trade in trades:
            emotion = trade.get('emotional_state', 'calm')
            emotional_states[emotion] = emotional_states.get(emotion, 0) + 1
        
        for emotion, count in emotional_states.items():
            percentage = count / len(trades) if trades else 0
            emotional_patterns['emotional_state_distribution'][emotion] = {
                'count': count,
                'percentage': percentage
            }
        
        # 情绪vs绩效关系
        emotion_performance = {}
        for trade in trades:
            emotion = trade.get('emotional_state', 'calm')
            pnl = trade['trade_result']['pnl']
            
            if emotion not in emotion_performance:
                emotion_performance[emotion] = {'total_pnl': 0, 'count': 0}
            
            emotion_performance[emotion]['total_pnl'] += pnl
            emotion_performance[emotion]['count'] += 1
        
        for emotion, data in emotion_performance.items():
            if data['count'] > 0:
                emotional_patterns['emotion_vs_performance'][emotion] = {
                    'avg_pnl': data['total_pnl'] / data['count'],
                    'trade_count': data['count']
                }
        
        # 常见情绪问题
        negative_emotions = ['greed', 'fear', 'anxious', 'overconfidence', 'hope', 'regret']
        
        for emotion in negative_emotions:
            if emotion in emotional_states:
                count = emotional_states[emotion]
                emotional_patterns['common_emotional_issues'].append({
                    'emotion': emotion,
                    'count': count,
                    'percentage': count / len(trades) if trades else 0,
                    'description': self._get_emotion_description(emotion)
                })
        
        return emotional_patterns
    
    def _get_emotion_description(self, emotion: str) -> str:
        """获取情绪描述"""
        descriptions = {
            'greed': '贪婪可能导致过度交易或忽视风险',
            'fear': '恐惧可能导致过早出场或错过机会',
            'anxious': '焦虑可能影响决策质量',
            'overconfidence': '过度自信可能导致风险忽视',
            'hope': '希望（持仓亏损）可能导致不止损',
            'regret': '后悔可能影响后续交易决策'
        }
        return descriptions.get(emotion, '未知情绪')
    
    def _analyze_error_patterns(self, trades: List[Dict]) -> Dict:
        """分析错误模式"""
        error_patterns = {
            'error_frequency': 0,
            'error_distribution': {},
            'costly_errors': [],
            'recurring_errors': [],
            'error_trend': 'unknown'
        }
        
        # 计算错误频率
        total_errors = 0
        for trade in trades:
            total_errors += len(trade['trade_errors'])
        
        error_patterns['error_frequency'] = total_errors / len(trades) if trades else 0
        
        # 错误分布
        error_categories = {}
        for trade in trades:
            for error in trade['trade_errors']:
                category = error['category']
                error_categories[category] = error_categories.get(category, 0) + 1
        
        for category, count in error_categories.items():
            error_patterns['error_distribution'][category] = {
                'count': count,
                'percentage': count / total_errors if total_errors > 0 else 0
            }
        
        # 代价高昂的错误（导致大亏的错误）
        costly_errors = []
        for trade in trades:
            if trade['trade_result']['significant_loss'] and trade['trade_errors']:
                # 找出导致大亏的可能错误
                for error in trade['trade_errors']:
                    if error['severity'] in ['high', 'critical']:
                        costly_errors.append({
                            'trade_id': trade['trade_id'],
                            'error': error,
                            'pnl': trade['trade_result']['pnl'],
                            'pnl_percentage': trade['trade_result']['pnl_percentage']
                        })
        
        # 取前5个代价最高的错误
        if costly_errors:
            sorted_costly = sorted(costly_errors, key=lambda x: x['pnl'])[:5]
            error_patterns['costly_errors'] = sorted_costly
        
        # 重复性错误（同一子类别错误出现3次以上）
        subcategory_counts = {}
        for trade in trades:
            for error in trade['trade_errors']:
                subcategory = error['subcategory']
                subcategory_counts[subcategory] = subcategory_counts.get(subcategory, 0) + 1
        
        recurring_errors = [(sc, count) for sc, count in subcategory_counts.items() if count >= 3]
        
        for subcategory, count in recurring_errors:
            # 获取错误描述
            description = ''
            for category in self.error_categories:
                if subcategory in self.error_categories[category]['subcategories']:
                    description = self.error_categories[category]['subcategories'][subcategory]
                    break
            
            error_patterns['recurring_errors'].append({
                'subcategory': subcategory,
                'count': count,
                'description': description,
                'percentage': count / total_errors if total_errors > 0 else 0
            })
        
        # 错误趋势（如果数据足够）
        if len(trades) >= 10:
            # 将交易分成两部分（早期和近期）
            midpoint = len(trades) // 2
            early_trades = trades[:midpoint]
            recent_trades = trades[midpoint:]
            
            early_errors = sum(len(trade['trade_errors']) for trade in early_trades)
            recent_errors = sum(len(trade['trade_errors']) for trade in recent_trades)
            
            early_avg = early_errors / len(early_trades) if early_trades else 0
            recent_avg = recent_errors / len(recent_trades) if recent_trades else 0
            
            if recent_avg < early_avg * 0.8:
                error_patterns['error_trend'] = 'improving'
            elif recent_avg > early_avg * 1.2:
                error_patterns['error_trend'] = 'worsening'
            else:
                error_patterns['error_trend'] = 'stable'
        
        return error_patterns
    
    def _generate_improvement_suggestions(self,
                                         basic_metrics: Dict,
                                         advanced_metrics: Dict,
                                         trading_patterns: Dict,
                                         error_patterns: Dict) -> List[Dict]:
        """生成改进建议"""
        suggestions = []
        
        # 基于基本绩效指标的建议
        if basic_metrics['win_rate'] < 0.4:
            suggestions.append({
                'type': 'performance',
                'priority': 'high',
                'suggestion': f'胜率较低 ({basic_metrics["win_rate"]:.0%})，建议：',
                'actions': [
                    '加强入场条件筛选',
                    '等待更高概率的交易机会',
                    '减少低概率交易'
                ],
                'metric': 'win_rate',
                'current_value': basic_metrics['win_rate'],
                'target_value': 0.5
            })
        
        if basic_metrics['profit_factor'] < 1.2:
            suggestions.append({
                'type': 'performance',
                'priority': 'high',
                'suggestion': f'盈亏比较低 ({basic_metrics["profit_factor"]:.1f})，建议：',
                'actions': [
                    '提高盈利交易的持仓时间',
                    '减小亏损交易的损失',
                    '优化止盈止损比例'
                ],
                'metric': 'profit_factor',
                'current_value': basic_metrics['profit_factor'],
                'target_value': 1.5
            })
        
        # 基于高级绩效指标的建议
        if not advanced_metrics.get('insufficient_data', False):
            if advanced_metrics['max_drawdown_percentage'] > 0.1:
                suggestions.append({
                    'type': 'risk',
                    'priority': 'high',
                    'suggestion': f'最大回撤较大 ({advanced_metrics["max_drawdown_percentage"]:.1%})，建议：',
                    'actions': [
                        '降低仓位大小',
                        '加强风险管理',
                        '避免连续亏损交易'
                    ],
                    'metric': 'max_drawdown',
                    'current_value': advanced_metrics['max_drawdown_percentage'],
                    'target_value': 0.05
                })
            
            if advanced_metrics['consistency_score'] < 0.6:
                suggestions.append({
                    'type': 'consistency',
                    'priority': 'medium',
                    'suggestion': '交易一致性较低，建议：',
                    'actions': [
                        '标准化交易流程',
                        '减少情绪化交易',
                        '严格执行交易计划'
                    ],
                    'metric': 'consistency_score',
                    'current_value': advanced_metrics['consistency_score'],
                    'target_value': 0.7
                })
        
        # 基于交易模式的建议
        if 'time_patterns' in trading_patterns:
            worst_hours = trading_patterns['time_patterns'].get('worst_performing_hours', [])
            if worst_hours:
                hour_str = ', '.join([str(h['hour']) for h in worst_hours[:2]])
                suggestions.append({
                    'type': 'timing',
                    'priority': 'medium',
                    'suggestion': f'在{hour_str}时段的交易表现较差，建议：',
                    'actions': [
                        f'避免在{hour_str}时段交易',
                        '分析该时段市场特性',
                        '调整交易时间安排'
                    ],
                    'data_source': 'time_patterns'
                })
        
        if 'instrument_patterns' in trading_patterns:
            worst_instruments = trading_patterns['instrument_patterns'].get('worst_performing_instruments', [])
            if worst_instruments:
                instr_str = ', '.join([i['instrument'] for i in worst_instruments[:2]])
                suggestions.append({
                    'type': 'instrument',
                    'priority': 'medium',
                    'suggestion': f'在{instr_str}品种上表现较差，建议：',
                    'actions': [
                        f'减少{instr_str}交易',
                        '深入研究这些品种特性',
                        '考虑暂时避开这些品种'
                    ],
                    'data_source': 'instrument_patterns'
                })
        
        # 基于错误模式的建议
        if error_patterns.get('error_frequency', 0) > 2:
            suggestions.append({
                'type': 'error_reduction',
                'priority': 'high',
                'suggestion': f'交易错误频率较高 ({error_patterns["error_frequency"]:.1f}/交易)，建议：',
                'actions': [
                    '加强交易前准备',
                    '严格执行交易计划',
                    '进行交易后反思'
                ],
                'metric': 'error_frequency',
                'current_value': error_patterns['error_frequency'],
                'target_value': 1.0
            })
        
        recurring_errors = error_patterns.get('recurring_errors', [])
        if recurring_errors:
            top_error = recurring_errors[0]
            suggestions.append({
                'type': 'error_focus',
                'priority': 'high',
                'suggestion': f'重复性错误：{top_error["description"]} (出现{top_error["count"]}次)，建议：',
                'actions': [
                    '针对此错误进行专门训练',
                    '设置错误提醒',
                    '制定纠正措施'
                ],
                'error_type': top_error['subcategory'],
                'count': top_error['count']
            })
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order[x['priority']])
        
        return suggestions
    
    def _generate_performance_summary(self,
                                     basic_metrics: Dict,
                                     advanced_metrics: Dict) -> Dict:
        """生成绩效摘要"""
        performance_score = basic_metrics.get('performance_score', 0)
        
        # 确定绩效等级
        if performance_score >= 0.8:
            performance_grade = 'A'
            grade_description = '优秀：持续盈利，风险控制良好'
        elif performance_score >= 0.7:
            performance_grade = 'B'
            grade_description = '良好：总体盈利，有改进空间'
        elif performance_score >= 0.6:
            performance_grade = 'C'
            grade_description = '一般：勉强盈利，需要改进'
        elif performance_score >= 0.5:
            performance_grade = 'D'
            grade_description = '较差：接近盈亏平衡，急需改进'
        else:
            performance_grade = 'F'
            grade_description = '失败：持续亏损，需要重大调整'
        
        # 识别优势
        strengths = []
        if basic_metrics['win_rate'] >= 0.6:
            strengths.append(f'高胜率 ({basic_metrics["win_rate"]:.0%})')
        if basic_metrics['profit_factor'] >= 1.5:
            strengths.append(f'良好的盈亏比 ({basic_metrics["profit_factor"]:.1f})')
        if basic_metrics['total_pnl'] > 0:
            strengths.append(f'总体盈利 ({basic_metrics["total_pnl_percentage"]:.1%})')
        
        if not advanced_metrics.get('insufficient_data', False):
            if advanced_metrics['max_drawdown_percentage'] < 0.05:
                strengths.append(f'低回撤 ({advanced_metrics["max_drawdown_percentage"]:.1%})')
            if advanced_metrics['consistency_score'] >= 0.7:
                strengths.append('交易一致性高')
        
        # 识别弱点
        weaknesses = []
        if basic_metrics['win_rate'] < 0.4:
            weaknesses.append(f'低胜率 ({basic_metrics["win_rate"]:.0%})')
        if basic_metrics['profit_factor'] < 1.0:
            weaknesses.append(f'盈亏比不足 ({basic_metrics["profit_factor"]:.1f})')
        if basic_metrics['total_pnl'] < 0:
            weaknesses.append(f'总体亏损 ({basic_metrics["total_pnl_percentage"]:.1%})')
        
        if not advanced_metrics.get('insufficient_data', False):
            if advanced_metrics['max_drawdown_percentage'] > 0.1:
                weaknesses.append(f'高回撤 ({advanced_metrics["max_drawdown_percentage"]:.1%})')
            if advanced_metrics['consistency_score'] < 0.5:
                weaknesses.append('交易一致性低')
        
        summary = {
            'performance_grade': performance_grade,
            'performance_score': performance_score,
            'grade_description': grade_description,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'key_metrics': {
                'win_rate': basic_metrics['win_rate'],
                'profit_factor': basic_metrics['profit_factor'],
                'total_return': basic_metrics['total_pnl_percentage'],
                'avg_win': basic_metrics['avg_win'],
                'avg_loss': basic_metrics['avg_loss']
            },
            'recommendation': self._get_performance_recommendation(performance_score, strengths, weaknesses)
        }
        
        if not advanced_metrics.get('insufficient_data', False):
            summary['advanced_metrics'] = {
                'sharpe_ratio': advanced_metrics['sharpe_ratio'],
                'max_drawdown': advanced_metrics['max_drawdown_percentage'],
                'consistency': advanced_metrics['consistency_score']
            }
        
        return summary
    
    def _get_performance_recommendation(self,
                                       performance_score: float,
                                       strengths: List[str],
                                       weaknesses: List[str]) -> str:
        """获取绩效建议"""
        if performance_score >= 0.8:
            return "继续保持当前策略，可考虑小幅优化或适当增加风险暴露"
        elif performance_score >= 0.7:
            return "整体表现良好，专注于改进少数弱点即可获得更好结果"
        elif performance_score >= 0.6:
            return "需要系统性地改进交易策略，重点关注风险管理"
        elif performance_score >= 0.5:
            return "急需改进，建议暂停交易，重新评估策略和风险管理"
        else:
            return "立即停止交易，进行全面反思和策略重建"
    
    def _generate_detailed_analysis(self, trades: List[Dict]) -> Dict:
        """生成详细分析"""
        detailed_analysis = {
            'trade_by_trade_analysis': [],
            'correlation_analysis': {},
            'progression_analysis': {}
        }
        
        # 逐笔交易分析
        for trade in trades[-10:]:  # 最近10笔交易
            detailed_analysis['trade_by_trade_analysis'].append({
                'trade_id': trade['trade_id'],
                'date': trade['entry_time'].strftime('%Y-%m-%d') if isinstance(trade['entry_time'], datetime) else trade['entry_time'],
                'instrument': trade['instrument'],
                'direction': trade['direction'],
                'pnl': trade['trade_result']['pnl'],
                'pnl_percentage': trade['trade_result']['pnl_percentage'],
                'quality_grade': trade['trade_quality']['quality_grade'],
                'error_count': len(trade['trade_errors']),
                'tags': trade['tags'][:3]  # 前3个标签
            })
        
        # 相关性分析（如果数据足够）
        if len(trades) >= 10:
            # 准备数据
            pnls = [trade['trade_result']['pnl_percentage'] for trade in trades]
            quality_scores = [trade['trade_quality']['overall_score'] for trade in trades]
            holding_periods = [trade['trade_result']['holding_period_hours'] for trade in trades]
            error_counts = [len(trade['trade_errors']) for trade in trades]
            
            # 计算相关性（简化版）
            def simple_correlation(x, y):
                if len(x) != len(y) or len(x) < 2:
                    return 0
                
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
                denominator_x = sum((xi - x_mean) ** 2 for xi in x)
                denominator_y = sum((yi - y_mean) ** 2 for yi in y)
                
                if denominator_x * denominator_y == 0:
                    return 0
                
                return numerator / (denominator_x * denominator_y) ** 0.5
            
            detailed_analysis['correlation_analysis'] = {
                'quality_vs_pnl': simple_correlation(quality_scores, pnls),
                'holding_period_vs_pnl': simple_correlation(holding_periods, pnls),
                'errors_vs_pnl': simple_correlation(error_counts, pnls),
                'quality_vs_errors': simple_correlation(quality_scores, error_counts)
            }
        
        # 进展分析
        if len(trades) >= 20:
            # 将交易分成4个阶段
            chunk_size = len(trades) // 4
            stages = []
            
            for i in range(4):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < 3 else len(trades)
                stage_trades = trades[start_idx:end_idx]
                
                if stage_trades:
                    stage_pnls = [trade['trade_result']['pnl_percentage'] for trade in stage_trades]
                    stage_quality = [trade['trade_quality']['overall_score'] for trade in stage_trades]
                    stage_errors = [len(trade['trade_errors']) for trade in stage_trades]
                    
                    stages.append({
                        'stage': i + 1,
                        'trades': len(stage_trades),
                        'avg_pnl': np.mean(stage_pnls) if stage_pnls else 0,
                        'avg_quality': np.mean(stage_quality) if stage_quality else 0,
                        'avg_errors': np.mean(stage_errors) if stage_errors else 0,
                        'win_rate': sum(1 for trade in stage_trades if trade['trade_result']['win']) / len(stage_trades) if stage_trades else 0
                    })
            
            detailed_analysis['progression_analysis'] = {
                'stages': stages,
                'trend': self._analyze_progression_trend(stages)
            }
        
        return detailed_analysis
    
    def _analyze_progression_trend(self, stages: List[Dict]) -> str:
        """分析进展趋势"""
        if len(stages) < 2:
            return 'insufficient_data'
        
        # 提取指标
        avg_pnls = [stage['avg_pnl'] for stage in stages]
        avg_quality = [stage['avg_quality'] for stage in stages]
        avg_errors = [stage['avg_errors'] for stage in stages]
        win_rates = [stage['win_rate'] for stage in stages]
        
        # 计算趋势
        def calculate_trend(values):
            if len(values) < 2:
                return 'stable'
            
            # 简单线性趋势
            x = list(range(len(values)))
            y = values
            
            # 计算斜率（简化）
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                return 'stable'
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'
            else:
                return 'stable'
        
        pnl_trend = calculate_trend(avg_pnls)
        quality_trend = calculate_trend(avg_quality)
        error_trend = calculate_trend(avg_errors)
        win_rate_trend = calculate_trend(win_rates)
        
        # 综合趋势
        improving_count = sum(1 for trend in [pnl_trend, quality_trend, error_trend, win_rate_trend] if trend == 'improving')
        declining_count = sum(1 for trend in [pnl_trend, quality_trend, error_trend, win_rate_trend] if trend == 'declining')
        
        if improving_count >= 3:
            return 'strongly_improving'
        elif improving_count >= 2:
            return 'improving'
        elif declining_count >= 3:
            return 'strongly_declining'
        elif declining_count >= 2:
            return 'declining'
        else:
            return 'stable'
    
    def export_logs_to_dataframe(self) -> pd.DataFrame:
        """将日志导出为DataFrame"""
        if not self.trades_log:
            return pd.DataFrame()
        
        # 准备数据
        data = []
        for trade in self.trades_log:
            row = {
                'trade_id': trade['trade_id'],
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'instrument': trade['instrument'],
                'direction': trade['direction'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'position_size': trade['position_size'],
                'pnl': trade['trade_result']['pnl'],
                'pnl_percentage': trade['trade_result']['pnl_percentage'],
                'holding_hours': trade['trade_result']['holding_period_hours'],
                'quality_score': trade['trade_quality']['overall_score'],
                'quality_grade': trade['trade_quality']['quality_grade'],
                'error_count': len(trade['trade_errors']),
                'tags': ', '.join(trade['tags'][:3]),
                'exit_reason': trade.get('exit_reason', ''),
                'notes': trade.get('notes', '')
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 设置时间索引
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df = df.sort_values('entry_time')
        
        return df
    
    def generate_comprehensive_report(self,
                                     time_period: str = 'all',
                                     format: str = 'dict') -> Any:
        """
        生成综合报告
        
        参数:
            time_period: 时间周期
            format: 输出格式（'dict', 'markdown', 'html'）
            
        返回:
            综合报告
        """
        # 获取绩效分析
        performance_analysis = self.analyze_trading_performance(time_period, include_details=True)
        
        if 'error' in performance_analysis:
            return performance_analysis
        
        # 获取日志DataFrame
        logs_df = self.export_logs_to_dataframe()
        
        # 准备报告数据
        report_data = {
            'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generation_time': datetime.now(),
            'time_period': time_period,
            'trader_profile': self.trader_profile,
            'performance_analysis': performance_analysis,
            'log_summary': {
                'total_trades': len(self.trades_log),
                'time_period_trades': performance_analysis['trade_count'],
                'dataframe_shape': logs_df.shape if not logs_df.empty else (0, 0)
            }
        }
        
        if format == 'dict':
            return report_data
        
        elif format == 'markdown':
            return self._export_report_to_markdown(report_data)
        
        elif format == 'html':
            return self._export_report_to_html(report_data)
        
        else:
            return {'error': f'不支持的格式: {format}'}
    
    def _export_report_to_markdown(self, report_data: Dict) -> str:
        """将报告导出为Markdown格式"""
        perf = report_data['performance_analysis']
        summary = perf['performance_summary']
        
        markdown = f"""# 交易日志分析报告

## 报告信息
- **报告ID**: {report_data['report_id']}
- **生成时间**: {report_data['generation_time']}
- **分析周期**: {report_data['time_period']}
- **交易者**: {report_data['trader_profile'].get('experience_level', 'unknown')}级别

## 绩效概览
- **绩效等级**: {summary['performance_grade']} ({summary['performance_score']:.0%})
- **总体评价**: {summary['grade_description']}
- **交易次数**: {perf['trade_count']}
- **胜率**: {perf['basic_metrics']['win_rate']:.0%}
- **盈亏比**: {perf['basic_metrics']['profit_factor']:.1f}
- **总回报**: {perf['basic_metrics']['total_pnl_percentage']:.1%}

## 优势
"""
        
        for strength in summary['strengths']:
            markdown += f"- {strength}\n"
        
        markdown += "\n## 弱点\n"
        
        for weakness in summary['weaknesses']:
            markdown += f"- {weakness}\n"
        
        markdown += "\n## 关键指标\n"
        markdown += f"- 平均盈利: ${perf['basic_metrics']['avg_win']:.2f}\n"
        markdown += f"- 平均亏损: ${perf['basic_metrics']['avg_loss']:.2f}\n"
        markdown += f"- 最大盈利: ${perf['basic_metrics']['max_win']:.2f}\n"
        markdown += f"- 最大亏损: ${perf['basic_metrics']['max_loss']:.2f}\n"
        markdown += f"- 平均持仓时间: {perf['basic_metrics']['avg_holding_period_hours']:.1f}小时\n"
        
        if 'advanced_metrics' in perf and not perf['advanced_metrics'].get('insufficient_data', False):
            markdown += "\n## 高级指标\n"
            markdown += f"- 夏普比率: {perf['advanced_metrics']['sharpe_ratio']:.2f}\n"
            markdown += f"- 最大回撤: {perf['advanced_metrics']['max_drawdown_percentage']:.1%}\n"
            markdown += f"- 一致性分数: {perf['advanced_metrics']['consistency_score']:.0%}\n"
        
        if 'improvement_suggestions' in perf:
            markdown += "\n## 改进建议\n"
            
            for i, suggestion in enumerate(perf['improvement_suggestions'][:5], 1):
                markdown += f"{i}. **{suggestion['priority'].upper()}**: {suggestion['suggestion']}\n"
                for action in suggestion.get('actions', [])[:3]:
                    markdown += f"   - {action}\n"
        
        markdown += f"\n---\n*报告结束*"
        
        return markdown
    
    def _export_report_to_html(self, report_data: Dict) -> str:
        """将报告导出为HTML格式"""
        # 简化的HTML输出
        markdown = self._export_report_to_markdown(report_data)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交易日志分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; }}
        h2 {{ color: #4CAF50; }}
        h3 {{ color: #666; }}
        .strength {{ color: #4CAF50; }}
        .weakness {{ color: #f44336; }}
        .recommendation {{ background-color: #f9f9f9; padding: 10px; border-left: 4px solid #2196F3; margin: 10px 0; }}
        .metric {{ display: inline-block; margin: 5px 10px; padding: 5px 10px; background-color: #e8f5e8; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>交易日志分析报告</h1>
    <p><strong>报告ID</strong>: {report_data['report_id']}</p>
    <p><strong>生成时间</strong>: {report_data['generation_time']}</p>
    <p><strong>分析周期</strong>: {report_data['time_period']}</p>
</body>
</html>"""
        
        return html
    
    def get_log_statistics(self) -> Dict:
        """获取日志统计信息"""
        if not self.trades_log:
            return {'total_trades': 0}
        
        # 计算时间范围
        entry_times = [trade['entry_time'] for trade in self.trades_log]
        min_time = min(entry_times)
        max_time = max(entry_times)
        
        # 计算各种统计
        total_pnl = sum(trade['trade_result']['pnl'] for trade in self.trades_log)
        total_errors = sum(len(trade['trade_errors']) for trade in self.trades_log)
        
        # 质量分布
        quality_dist = {}
        for trade in self.trades_log:
            grade = trade['trade_quality']['quality_grade']
            quality_dist[grade] = quality_dist.get(grade, 0) + 1
        
        # 错误分布
        error_dist = {}
        for trade in self.trades_log:
            for error in trade['trade_errors']:
                category = error['category']
                error_dist[category] = error_dist.get(category, 0) + 1
        
        return {
            'total_trades': len(self.trades_log),
            'time_range': {
                'start': min_time,
                'end': max_time,
                'days': (max_time - min_time).days if isinstance(max_time, datetime) and isinstance(min_time, datetime) else 0
            },
            'total_pnl': total_pnl,
            'total_errors': total_errors,
            'avg_errors_per_trade': total_errors / len(self.trades_log) if self.trades_log else 0,
            'quality_distribution': quality_dist,
            'error_distribution': error_dist,
            'instrument_count': len(set(trade['instrument'] for trade in self.trades_log)),
            'recent_trades': [
                {
                    'trade_id': trade['trade_id'],
                    'date': trade['entry_time'].strftime('%Y-%m-%d') if isinstance(trade['entry_time'], datetime) else trade['entry_time'],
                    'instrument': trade['instrument'],
                    'pnl': trade['trade_result']['pnl']
                }
                for trade in self.trades_log[-5:]  # 最近5笔交易
            ]
        }
    
    def reset_logs(self):
        """重置日志"""
        self.trades_log = []
        self.performance_history = []
        self.analysis_reports = []


# ============================================================================
# 策略改造: 添加PriceActionRangesTradingLogAnalyzerStrategy类
# 将价格行为区间交易日志分析系统转换为交易策略
# ============================================================================

class PriceActionRangesTradingLogAnalyzerStrategy(BaseStrategy):
    """价格行为区间交易日志分析策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 创建交易日志分析器实例
        self.log_analyzer = TradingLogAnalyzer()
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于交易日志分析生成交易信号
        """
        # 获取绩效分析
        performance_analysis = self.log_analyzer.analyze_trading_performance('all')
        
        # 分析绩效数据
        if 'error' in performance_analysis:
            # 分析错误，hold信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
            return self.signals
        
        # 获取绩效摘要
        performance_summary = performance_analysis.get('performance_summary', {})
        performance_grade = performance_summary.get('performance_grade', '').upper()
        performance_score = performance_summary.get('performance_score', 0)
        
        # 获取基本指标
        basic_metrics = performance_analysis.get('basic_metrics', {})
        win_rate = basic_metrics.get('win_rate', 0)
        profit_factor = basic_metrics.get('profit_factor', 0)
        
        # 根据日志分析生成信号
        if performance_grade in ['A', 'B'] and win_rate >= 0.6:
            # 高绩效，高胜率，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif performance_grade in ['D', 'F'] or win_rate <= 0.4:
            # 低绩效，低胜率，卖出信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='sell',
                price=self.data['close'].iloc[-1]
            )
        elif profit_factor >= 1.5:
            # 高盈亏比，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif profit_factor <= 0.8:
            # 低盈亏比，卖出信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='sell',
                price=self.data['close'].iloc[-1]
            )
        else:
            # 中等绩效，hold信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        
        return self.signals