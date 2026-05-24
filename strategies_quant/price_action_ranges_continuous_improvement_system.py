# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.631219

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持续改进量化系统
第26章：持续改进
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 性能监控：交易结果跟踪、指标计算、趋势分析
2. 反馈分析：成功模式识别、失败原因分析、改进机会发现
3. 策略优化：参数调整、规则优化、算法改进
4. 自适应调整：市场条件适应、动态参数调整、策略切换
5. 学习循环：经验积累、知识更新、系统进化
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

class ContinuousImprovementSystem:
    """持续改进系统（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 trader_profile: Dict = None,
                 improvement_goals: Dict = None,
                 learning_rate: float = 0.1):
        """初始化持续改进系统"""
        self.trader_profile = trader_profile or {
            'experience_level': 'intermediate',
            'trading_style': 'swing',
            'improvement_focus': 'risk_management',
            'learning_preference': 'data_driven'
        }
        
        self.improvement_goals = improvement_goals or {
            'win_rate_target': 0.60,  # 胜率目标60%
            'profit_factor_target': 2.0,  # 盈利因子目标2.0
            'max_drawdown_target': -0.15,  # 最大回撤目标-15%
            'risk_reward_target': 2.5,  # 风险回报比目标2.5:1
            'consistency_target': 0.70,  # 一致性目标70%
            'adaptability_target': 0.80  # 适应性目标80%
        }
        
        self.learning_rate = learning_rate  # 学习率，控制调整幅度
        
        # 性能历史记录
        self.performance_history = {
            'trades': [],  # 交易记录
            'metrics': [],  # 绩效指标
            'improvements': [],  # 改进记录
            'feedback': []  # 反馈记录
        }
        
        # 改进策略库
        self.improvement_strategies = {
            'risk_management': {
                'position_sizing_adjustment': True,
                'stop_loss_optimization': True,
                'risk_exposure_control': True,
                'correlation_management': True
            },
            'entry_timing': {
                'timing_optimization': True,
                'confirmation_signals': True,
                'multiple_timeframe_alignment': True,
                'market_structure_consideration': True
            },
            'exit_strategy': {
                'profit_target_optimization': True,
                'trailing_stop_adjustment': True,
                'partial_exit_strategy': True,
                'risk_reward_optimization': True
            },
            'psychology': {
                'emotional_control': True,
                'discipline_enforcement': True,
                'confidence_building': True,
                'error_reduction': True
            }
        }
        
        # 自适应参数
        self.adaptive_parameters = {
            'market_regime_weights': {
                'trending': 1.0,
                'ranging': 0.8,
                'transition': 0.6,
                'volatile': 0.7,
                'calm': 1.2
            },
            'timeframe_weights': {
                '1h': 0.9,
                '4h': 1.0,
                '1d': 1.1,
                '1w': 1.2
            },
            'instrument_weights': {
                'forex': 1.0,
                'stocks': 0.9,
                'crypto': 0.8,
                'commodities': 0.95
            }
        }
        
        # 学习状态
        self.learning_state = {
            'total_trades_analyzed': 0,
            'improvements_applied': 0,
            'success_rate': 0.0,
            'adaptation_score': 0.0,
            'last_improvement_date': None,
            'improvement_trend': 'neutral'
        }
    
    def add_trade_result(self, trade_data: Dict) -> Dict:
        """添加交易结果进行分析"""
        # 验证交易数据
        required_fields = ['result', 'profit_loss', 'risk_reward', 'setup_type', 'market_conditions']
        for field in required_fields:
            if field not in trade_data:
                return {'error': f'缺少必要字段: {field}'}
        
        # 添加交易记录
        trade_data['analysis_time'] = datetime.now()
        trade_data['trade_id'] = f"trade_{len(self.performance_history['trades']) + 1:06d}"
        
        self.performance_history['trades'].append(trade_data)
        self.learning_state['total_trades_analyzed'] += 1
        
        # 分析交易结果
        analysis_result = self._analyze_trade_result(trade_data)
        
        # 生成改进建议
        if analysis_result['needs_improvement']:
            improvement_suggestions = self._generate_improvement_suggestions(trade_data, analysis_result)
            analysis_result['improvement_suggestions'] = improvement_suggestions
        
        # 更新性能指标
        self._update_performance_metrics()
        
        return {
            'trade_id': trade_data['trade_id'],
            'analysis_result': analysis_result,
            'total_trades': self.learning_state['total_trades_analyzed'],
            'improvement_status': self.learning_state['improvement_trend']
        }
    
    def _analyze_trade_result(self, trade_data: Dict) -> Dict:
        """分析单笔交易结果"""
        result = trade_data['result']  # 'win', 'loss', 'break_even'
        profit_loss = trade_data['profit_loss']
        risk_reward = trade_data.get('risk_reward', 1.0)
        setup_type = trade_data['setup_type']
        market_conditions = trade_data['market_conditions']
        
        analysis = {
            'result': result,
            'profit_loss': profit_loss,
            'risk_reward_ratio': risk_reward,
            'setup_effectiveness': 'unknown',
            'market_adaptation': 'unknown',
            'error_analysis': [],
            'success_factors': [],
            'needs_improvement': False,
            'improvement_priority': 'low'
        }
        
        # 分析设置有效性
        if result == 'win':
            if profit_loss > 0:
                analysis['setup_effectiveness'] = 'effective'
                analysis['success_factors'].append('正确的设置执行')
            else:
                analysis['setup_effectiveness'] = 'ineffective'
                analysis['error_analysis'].append('盈利但设置无效')
                analysis['needs_improvement'] = True
        elif result == 'loss':
            analysis['setup_effectiveness'] = 'ineffective'
            analysis['error_analysis'].append('亏损交易')
            analysis['needs_improvement'] = True
            analysis['improvement_priority'] = 'medium'
        
        # 分析市场适应性
        market_regime = market_conditions.get('regime', 'unknown')
        if market_regime in ['trending', 'ranging']:
            if result == 'win':
                analysis['market_adaptation'] = 'well_adapted'
            else:
                analysis['market_adaptation'] = 'poorly_adapted'
                analysis['needs_improvement'] = True
        elif market_regime == 'transition':
            analysis['market_adaptation'] = 'challenging'
            if result == 'loss':
                analysis['improvement_priority'] = 'low'  # 转换期亏损正常
        
        # 分析风险回报比
        if risk_reward < 1.5:
            analysis['error_analysis'].append('风险回报比不足')
            if result == 'loss':
                analysis['needs_improvement'] = True
                analysis['improvement_priority'] = 'high'
        
        # 分析盈利规模
        if result == 'win' and profit_loss < trade_data.get('expected_profit', 0):
            analysis['error_analysis'].append('盈利未达预期')
            analysis['needs_improvement'] = True
            analysis['improvement_priority'] = 'medium'
        
        return analysis
    
    def _generate_improvement_suggestions(self, trade_data: Dict, analysis_result: Dict) -> List[Dict]:
        """生成改进建议"""
        suggestions = []
        setup_type = trade_data['setup_type']
        market_conditions = trade_data['market_conditions']
        error_analysis = analysis_result['error_analysis']
        
        # 基于错误分析生成建议
        for error in error_analysis:
            if '风险回报比' in error:
                suggestions.append({
                    'area': 'risk_management',
                    'suggestion': '提高风险回报比至少到1.5:1',
                    'action': '调整止盈目标或收紧止损',
                    'priority': 'high',
                    'expected_impact': '提高长期盈利性'
                })
            
            if '盈利未达预期' in error:
                suggestions.append({
                    'area': 'exit_strategy',
                    'suggestion': '优化止盈策略，确保达到预期盈利',
                    'action': '调整止盈水平或使用移动止损',
                    'priority': 'medium',
                    'expected_impact': '提高单笔交易盈利'
                })
            
            if '亏损交易' in error and analysis_result['market_adaptation'] == 'poorly_adapted':
                suggestions.append({
                    'area': 'market_analysis',
                    'suggestion': '改进市场条件分析，避免在不适合的市场交易',
                    'action': '加强市场体制识别和适应性调整',
                    'priority': 'high',
                    'expected_impact': '减少不必要亏损'
                })
        
        # 基于设置类型生成建议
        if setup_type == 'breakout':
            suggestions.append({
                'area': 'entry_timing',
                'suggestion': '优化突破交易的确认信号',
                'action': '等待价格收盘突破关键水平后再入场',
                'priority': 'medium',
                'expected_impact': '提高突破交易成功率'
            })
        
        elif setup_type == 'reversal':
            suggestions.append({
                'area': 'entry_timing',
                'suggestion': '加强反转形态的确认',
                'action': '使用多个时间框架确认反转信号',
                'priority': 'medium',
                'expected_impact': '提高反转交易准确性'
            })
        
        # 基于市场条件生成建议
        market_regime = market_conditions.get('regime', 'unknown')
        if market_regime == 'transition':
            suggestions.append({
                'area': 'risk_management',
                'suggestion': '转换期减少仓位规模',
                'action': '将仓位减少到正常水平的50-70%',
                'priority': 'low',
                'expected_impact': '降低转换期风险'
            })
        
        # 限制建议数量，优先处理高优先级
        suggestions.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])
        return suggestions[:5]  # 返回最多5个建议
    
    def _update_performance_metrics(self) -> None:
        """更新性能指标"""
        if not self.performance_history['trades']:
            return
        
        trades = self.performance_history['trades']
        
        # 计算基本指标
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('result') == 'win']
        losing_trades = [t for t in trades if t.get('result') == 'loss']
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 计算风险回报比平均值
        risk_rewards = [t.get('risk_reward', 1.0) for t in trades if t.get('risk_reward') is not None]
        avg_risk_reward = np.mean(risk_rewards) if risk_rewards else 1.0
        
        # 计算一致性（连续盈利/亏损模式）
        consistency = self._calculate_consistency(trades)
        
        # 计算适应性得分
        adaptation_score = self._calculate_adaptation_score(trades)
        
        # 更新学习状态
        self.learning_state['success_rate'] = win_rate
        self.learning_state['adaptation_score'] = adaptation_score
        
        # 评估改进趋势
        self._evaluate_improvement_trend()
        
        # 保存指标
        metrics = {
            'timestamp': datetime.now(),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_risk_reward': avg_risk_reward,
            'consistency': consistency,
            'adaptation_score': adaptation_score,
            'total_profit': total_profit,
            'total_loss': total_loss
        }
        
        self.performance_history['metrics'].append(metrics)
    
    def _calculate_consistency(self, trades: List[Dict]) -> float:
        """计算交易一致性"""
        if len(trades) < 2:
            return 0.5  # 默认值
        
        results = [1 if t.get('result') == 'win' else 0 for t in trades]
        
        # 计算连胜和连败模式
        streaks = []
        current_streak = 1
        
        for i in range(1, len(results)):
            if results[i] == results[i-1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        
        streaks.append(current_streak)
        
        # 一致性计算：1 - (平均波动/最大可能波动)
        avg_streak = np.mean(streaks)
        max_possible_streak = len(results)
        
        consistency = 1.0 - (avg_streak / max_possible_streak)
        return max(0.0, min(1.0, consistency))
    
    def _calculate_adaptation_score(self, trades: List[Dict]) -> float:
        """计算市场适应性得分"""
        if len(trades) < 5:
            return 0.5  # 默认值
        
        # 分析不同市场条件下的表现
        regime_performance = {}
        for trade in trades:
            regime = trade.get('market_conditions', {}).get('regime', 'unknown')
            result = 1 if trade.get('result') == 'win' else 0
            
            if regime not in regime_performance:
                regime_performance[regime] = {'wins': 0, 'total': 0}
            
            regime_performance[regime]['total'] += 1
            if result == 1:
                regime_performance[regime]['wins'] += 1
        
        # 计算各体制胜率
        regime_win_rates = {}
        for regime, stats in regime_performance.items():
            if stats['total'] >= 3:  # 至少3笔交易才计算
                win_rate = stats['wins'] / stats['total']
                regime_win_rates[regime] = win_rate
        
        if not regime_win_rates:
            return 0.5
        
        # 适应性得分：各体制胜率的加权平均
        total_weight = 0
        weighted_score = 0
        
        for regime, win_rate in regime_win_rates.items():
            weight = self.adaptive_parameters['market_regime_weights'].get(regime, 0.5)
            total_weight += weight
            weighted_score += win_rate * weight
        
        adaptation_score = weighted_score / total_weight if total_weight > 0 else 0.5
        return adaptation_score
    
    def _evaluate_improvement_trend(self) -> None:
        """评估改进趋势"""
        if len(self.performance_history['metrics']) < 10:
            self.learning_state['improvement_trend'] = 'insufficient_data'
            return
        
        # 获取最近20个指标点
        recent_metrics = self.performance_history['metrics'][-20:]
        
        # 分析胜率趋势
        win_rates = [m['win_rate'] for m in recent_metrics]
        
        if len(win_rates) >= 5:
            # 计算简单移动平均趋势
            early_avg = np.mean(win_rates[:5])
            recent_avg = np.mean(win_rates[-5:])
            
            if recent_avg > early_avg + 0.05:  # 提高5%以上
                self.learning_state['improvement_trend'] = 'improving'
            elif recent_avg < early_avg - 0.05:  # 下降5%以上
                self.learning_state['improvement_trend'] = 'declining'
            else:
                self.learning_state['improvement_trend'] = 'stable'
        else:
            self.learning_state['improvement_trend'] = 'neutral'
    
    def generate_improvement_plan(self, focus_areas: List[str] = None) -> Dict:
        """生成改进计划"""
        if focus_areas is None:
            focus_areas = ['risk_management', 'entry_timing', 'exit_strategy', 'psychology']
        
        plan = {
            'plan_id': f"improvement_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'creation_time': datetime.now(),
            'trader_profile': self.trader_profile,
            'current_performance': self._get_current_performance_summary(),
            'improvement_goals': self.improvement_goals,
            'focus_areas': {},
            'action_items': [],
            'timeline': {},
            'success_metrics': []
        }
        
        # 为每个重点领域生成具体计划
        for area in focus_areas:
            if area in self.improvement_strategies:
                area_plan = self._generate_area_improvement_plan(area)
                plan['focus_areas'][area] = area_plan
                plan['action_items'].extend(area_plan.get('actions', []))
        
        # 设置时间线
        plan['timeline'] = {
            'short_term': '1-2周',
            'medium_term': '1-3个月',
            'long_term': '3-6个月'
        }
        
        # 设置成功指标
        plan['success_metrics'] = [
            {'metric': 'win_rate', 'target': self.improvement_goals['win_rate_target'], 'current': self.learning_state['success_rate']},
            {'metric': 'profit_factor', 'target': self.improvement_goals['profit_factor_target'], 'current': self._get_current_profit_factor()},
            {'metric': 'adaptation_score', 'target': self.improvement_goals['adaptability_target'], 'current': self.learning_state['adaptation_score']}
        ]
        
        # 保存改进计划
        self.performance_history['improvements'].append({
            'plan_id': plan['plan_id'],
            'plan_data': plan,
            'implementation_status': 'pending'
        })
        
        self.learning_state['last_improvement_date'] = datetime.now()
        
        return plan
    
    def _get_current_performance_summary(self) -> Dict:
        """获取当前性能摘要"""
        if not self.performance_history['metrics']:
            return {'status': 'no_data', 'message': '尚无交易数据'}
        
        latest_metrics = self.performance_history['metrics'][-1]
        
        return {
            'total_trades': latest_metrics['total_trades'],
            'win_rate': latest_metrics['win_rate'],
            'profit_factor': latest_metrics['profit_factor'],
            'avg_risk_reward': latest_metrics['avg_risk_reward'],
            'consistency': latest_metrics['consistency'],
            'adaptation_score': latest_metrics['adaptation_score'],
            'improvement_trend': self.learning_state['improvement_trend']
        }
    
    def _get_current_profit_factor(self) -> float:
        """获取当前盈利因子"""
        if not self.performance_history['metrics']:
            return 1.0
        
        latest_metrics = self.performance_history['metrics'][-1]
        return latest_metrics['profit_factor']
    
    def _generate_area_improvement_plan(self, area: str) -> Dict:
        """为特定领域生成改进计划"""
        strategies = self.improvement_strategies.get(area, {})
        
        if area == 'risk_management':
            return {
                'area_name': '风险管理',
                'current_status': self._assess_risk_management_status(),
                'strategies': strategies,
                'actions': [
                    '记录每笔交易的风险回报比',
                    '定期检查仓位规模是否合适',
                    '分析亏损交易的风险控制问题',
                    '调整止损策略基于市场波动性'
                ],
                'success_criteria': [
                    '风险回报比平均值达到2.0:1',
                    '最大回撤控制在15%以内',
                    '单笔交易风险不超过账户2%'
                ]
            }
        
        elif area == 'entry_timing':
            return {
                'area_name': '入场时机',
                'current_status': self._assess_entry_timing_status(),
                'strategies': strategies,
                'actions': [
                    '分析成功交易的入场信号',
                    '优化确认信号的使用',
                    '加强多时间框架对齐',
                    '记录入场时的市场结构'
                ],
                'success_criteria': [
                    '入场后立即盈利的交易比例提高',
                    '错误入场减少50%',
                    '入场时机与市场节奏更匹配'
                ]
            }
        
        elif area == 'exit_strategy':
            return {
                'area_name': '出场策略',
                'current_status': self._assess_exit_strategy_status(),
                'strategies': strategies,
                'actions': [
                    '分析止盈过早或过晚的交易',
                    '优化移动止损设置',
                    '实施部分出场策略',
                    '根据市场条件调整止盈目标'
                ],
                'success_criteria': [
                    '平均盈利规模增加20%',
                    '盈利交易持有时间更合理',
                    '止损执行更严格'
                ]
            }
        
        elif area == 'psychology':
            return {
                'area_name': '交易心理',
                'current_status': self._assess_psychology_status(),
                'strategies': strategies,
                'actions': [
                    '记录交易时的情绪状态',
                    '分析情绪对决策的影响',
                    '制定纪律执行检查清单',
                    '建立错误后的恢复流程'
                ],
                'success_criteria': [
                    '情绪化交易减少70%',
                    '纪律执行率提高',
                    '交易自信心增强'
                ]
            }
        
        else:
            return {
                'area_name': area,
                'current_status': 'unknown',
                'strategies': strategies,
                'actions': ['分析该领域的具体问题'],
                'success_criteria': ['有待确定']
            }
    
    def _assess_risk_management_status(self) -> str:
        """评估风险管理状态"""
        metrics = self._get_current_performance_summary()
        
        if metrics.get('status') == 'no_data':
            return '数据不足'
        
        avg_rr = metrics.get('avg_risk_reward', 1.0)
        
        if avg_rr >= 2.0:
            return '优秀'
        elif avg_rr >= 1.5:
            return '良好'
        elif avg_rr >= 1.0:
            return '需要改进'
        else:
            return '急需改进'
    
    def _assess_entry_timing_status(self) -> str:
        """评估入场时机状态"""
        win_rate = self.learning_state.get('success_rate', 0.0)
        
        if win_rate >= 0.65:
            return '优秀'
        elif win_rate >= 0.55:
            return '良好'
        elif win_rate >= 0.45:
            return '需要改进'
        else:
            return '急需改进'
    
    def _assess_exit_strategy_status(self) -> str:
        """评估出场策略状态"""
        profit_factor = self._get_current_profit_factor()
        
        if profit_factor >= 2.5:
            return '优秀'
        elif profit_factor >= 1.8:
            return '良好'
        elif profit_factor >= 1.2:
            return '需要改进'
        else:
            return '急需改进'
    
    def _assess_psychology_status(self) -> str:
        """评估交易心理状态"""
        consistency = self._get_current_performance_summary().get('consistency', 0.5)
        
        if consistency >= 0.8:
            return '优秀'
        elif consistency >= 0.6:
            return '良好'
        elif consistency >= 0.4:
            return '需要改进'
        else:
            return '急需改进'
    
    def apply_improvement(self, improvement_suggestion: Dict, trade_context: Dict = None) -> Dict:
        """应用改进建议"""
        improvement_id = f"imp_{len(self.performance_history['improvements']) + 1:06d}"
        
        improvement_record = {
            'improvement_id': improvement_id,
            'suggestion': improvement_suggestion,
            'application_time': datetime.now(),
            'trade_context': trade_context,
            'expected_impact': improvement_suggestion.get('expected_impact', 'unknown'),
            'implementation_status': 'applied',
            'effectiveness': 'pending_evaluation'
        }
        
        self.performance_history['improvements'].append(improvement_record)
        self.learning_state['improvements_applied'] += 1
        
        # 根据改进类型调整系统参数
        self._adjust_system_parameters(improvement_suggestion)
        
        return {
            'improvement_id': improvement_id,
            'status': 'applied',
            'applied_at': datetime.now(),
            'total_improvements': self.learning_state['improvements_applied'],
            'next_evaluation': 'after_5_trades'
        }
    
    def _adjust_system_parameters(self, improvement_suggestion: Dict) -> None:
        """根据改进建议调整系统参数"""
        area = improvement_suggestion.get('area', '')
        action = improvement_suggestion.get('action', '')
        
        if 'risk_management' in area:
            # 调整风险参数
            if '风险回报比' in action:
                # 提高风险回报比要求
                self.improvement_goals['risk_reward_target'] = min(
                    3.0, self.improvement_goals['risk_reward_target'] + 0.1
                )
            
            elif '仓位' in action:
                # 调整仓位管理
                self.learning_rate = max(0.05, self.learning_rate * 0.9)  # 降低学习率，更谨慎
        
        elif 'entry_timing' in area:
            # 调整入场参数
            if '确认信号' in action:
                # 加强确认要求
                pass  # 在实际系统中会有具体参数调整
        
        elif 'exit_strategy' in area:
            # 调整出场参数
            if '止盈' in action:
                # 优化止盈策略
                pass  # 在实际系统中会有具体参数调整
    
    def evaluate_improvement_effectiveness(self, improvement_id: str, evaluation_trades: List[Dict]) -> Dict:
        """评估改进效果"""
        # 查找改进记录
        improvement_record = None
        for imp in self.performance_history['improvements']:
            if imp.get('improvement_id') == improvement_id:
                improvement_record = imp
                break
        
        if not improvement_record:
            return {'error': f'未找到改进记录: {improvement_id}'}
        
        # 分析评估交易
        evaluation_results = []
        for trade in evaluation_trades:
            result = self._analyze_trade_result(trade)
            evaluation_results.append(result)
        
        # 计算改进效果
        win_rate_before = improvement_record.get('context_win_rate', 0.5)
        win_rate_after = sum(1 for r in evaluation_results if r['result'] == 'win') / len(evaluation_results) if evaluation_results else 0
        
        improvement_effect = win_rate_after - win_rate_before
        
        # 更新改进记录
        improvement_record['evaluation_time'] = datetime.now()
        improvement_record['evaluation_trades'] = len(evaluation_trades)
        improvement_record['win_rate_before'] = win_rate_before
        improvement_record['win_rate_after'] = win_rate_after
        improvement_record['improvement_effect'] = improvement_effect
        
        if improvement_effect > 0.1:
            improvement_record['effectiveness'] = 'highly_effective'
        elif improvement_effect > 0:
            improvement_record['effectiveness'] = 'effective'
        elif improvement_effect > -0.1:
            improvement_record['effectiveness'] = 'neutral'
        else:
            improvement_record['effectiveness'] = 'negative'
        
        return {
            'improvement_id': improvement_id,
            'effectiveness': improvement_record['effectiveness'],
            'improvement_effect': improvement_effect,
            'win_rate_before': win_rate_before,
            'win_rate_after': win_rate_after,
            'evaluation_trades': len(evaluation_trades),
            'recommendation': 'continue' if improvement_effect > 0 else 'reconsider'
        }
    
    def get_performance_report(self, period: str = 'all') -> Dict:
        """获取性能报告"""
        if not self.performance_history['metrics']:
            return {'status': 'no_data', 'message': '尚无性能数据'}
        
        # 根据时间段筛选数据
        if period == 'recent':
            metrics = self.performance_history['metrics'][-10:]  # 最近10个记录点
        elif period == 'month':
            # 这里简化处理，实际应该按时间筛选
            metrics = self.performance_history['metrics'][-30:] if len(self.performance_history['metrics']) >= 30 else self.performance_history['metrics']
        else:
            metrics = self.performance_history['metrics']
        
        if not metrics:
            return {'status': 'no_data_for_period', 'message': f'该时间段无数据: {period}'}
        
        # 计算统计信息
        win_rates = [m['win_rate'] for m in metrics]
        profit_factors = [m['profit_factor'] for m in metrics if m['profit_factor'] != float('inf')]
        risk_rewards = [m['avg_risk_reward'] for m in metrics]
        
        report = {
            'period': period,
            'report_time': datetime.now(),
            'summary': {
                'total_trades_analyzed': self.learning_state['total_trades_analyzed'],
                'improvements_applied': self.learning_state['improvements_applied'],
                'current_win_rate': self.learning_state['success_rate'],
                'current_adaptation_score': self.learning_state['adaptation_score'],
                'improvement_trend': self.learning_state['improvement_trend']
            },
            'statistics': {
                'win_rate': {
                    'mean': np.mean(win_rates) if win_rates else 0,
                    'std': np.std(win_rates) if win_rates else 0,
                    'min': min(win_rates) if win_rates else 0,
                    'max': max(win_rates) if win_rates else 0,
                    'trend': 'increasing' if len(win_rates) >= 2 and win_rates[-1] > win_rates[0] else 'stable'
                },
                'profit_factor': {
                    'mean': np.mean(profit_factors) if profit_factors else 1.0,
                    'std': np.std(profit_factors) if profit_factors else 0,
                    'min': min(profit_factors) if profit_factors else 1.0,
                    'max': max(profit_factors) if profit_factors else 1.0
                },
                'risk_reward': {
                    'mean': np.mean(risk_rewards) if risk_rewards else 1.0,
                    'std': np.std(risk_rewards) if risk_rewards else 0,
                    'min': min(risk_rewards) if risk_rewards else 1.0,
                    'max': max(risk_rewards) if risk_rewards else 1.0
                }
            },
            'improvement_analysis': {
                'total_suggestions_generated': sum(len(t.get('improvement_suggestions', [])) for t in self.performance_history['trades'] if 'improvement_suggestions' in t),
                'suggestions_applied': self.learning_state['improvements_applied'],
                'application_rate': self.learning_state['improvements_applied'] / max(1, self.learning_state['total_trades_analyzed']),
                'last_improvement_date': self.learning_state['last_improvement_date']
            },
            'recommendations': self._generate_report_recommendations(metrics)
        }
        
        return report
    
    def _generate_report_recommendations(self, metrics: List[Dict]) -> List[Dict]:
        """生成报告建议"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        latest = metrics[-1]
        
        # 基于胜率的建议
        win_rate = latest['win_rate']
        if win_rate < 0.45:
            recommendations.append({
                'priority': 'high',
                'area': 'entry_timing',
                'recommendation': '胜率偏低，建议重点改进入场时机选择',
                'action': '分析最近亏损交易的入场信号，优化确认条件'
            })
        elif win_rate < 0.55:
            recommendations.append({
                'priority': 'medium',
                'area': 'entry_timing',
                'recommendation': '胜率有改进空间',
                'action': '加强对成功交易模式的学习和复制'
            })
        
        # 基于盈利因子的建议
        profit_factor = latest['profit_factor']
        if profit_factor < 1.5 and profit_factor != float('inf'):
            recommendations.append({
                'priority': 'high',
                'area': 'exit_strategy',
                'recommendation': '盈利因子偏低，出场策略需要优化',
                'action': '分析盈利交易的持有时间和止盈策略'
            })
        
        # 基于一致性的建议
        consistency = latest['consistency']
        if consistency < 0.4:
            recommendations.append({
                'priority': 'medium',
                'area': 'psychology',
                'recommendation': '交易一致性较低，可能存在心理或纪律问题',
                'action': '记录交易时的决策过程和情绪状态'
            })
        
        # 基于适应性的建议
        adaptation_score = latest['adaptation_score']
        if adaptation_score < 0.5:
            recommendations.append({
                'priority': 'medium',
                'area': 'market_analysis',
                'recommendation': '市场适应性有待提高',
                'action': '分析不同市场条件下的表现差异，调整策略适应性'
            })
        
        return recommendations
    
    def export_learning_data(self, format: str = 'json') -> Union[Dict, str]:
        """导出学习数据"""
        export_data = {
            'export_time': datetime.now(),
            'system_version': '1.0.0',
            'trader_profile': self.trader_profile,
            'learning_state': self.learning_state,
            'performance_summary': self._get_current_performance_summary(),
            'improvement_goals': self.improvement_goals,
            'recent_trades': self.performance_history['trades'][-50:] if self.performance_history['trades'] else [],
            'recent_improvements': self.performance_history['improvements'][-20:] if self.performance_history['improvements'] else [],
            'adaptive_parameters': self.adaptive_parameters
        }
        
        if format == 'json':
            # 转换日期时间为字符串
            def datetime_converter(o):
                if isinstance(o, datetime):
                    return o.isoformat()
                raise TypeError(f"Object of type {type(o)} is not JSON serializable")
            
            return json.dumps(export_data, default=datetime_converter, indent=2, ensure_ascii=False)
        
        elif format == 'dict':
            return export_data
        
        else:
            return {'error': f'不支持的格式: {format}'}
    
    def reset_learning(self, keep_history: bool = False) -> Dict:
        """重置学习状态"""
        if keep_history:
            # 只重置学习状态，保留历史数据
            old_history = self.performance_history.copy()
            
            self.learning_state = {
                'total_trades_analyzed': 0,
                'improvements_applied': 0,
                'success_rate': 0.0,
                'adaptation_score': 0.0,
                'last_improvement_date': None,
                'improvement_trend': 'neutral'
            }
            
            return {
                'status': 'reset_learning_only',
                'history_preserved': True,
                'history_size': {
                    'trades': len(old_history['trades']),
                    'metrics': len(old_history['metrics']),
                    'improvements': len(old_history['improvements'])
                }
            }
        else:
            # 完全重置
            self.performance_history = {
                'trades': [],
                'metrics': [],
                'improvements': [],
                'feedback': []
            }
            
            self.learning_state = {
                'total_trades_analyzed': 0,
                'improvements_applied': 0,
                'success_rate': 0.0,
                'adaptation_score': 0.0,
                'last_improvement_date': None,
                'improvement_trend': 'neutral'
            }
            
            return {
                'status': 'complete_reset',
                'history_preserved': False
            }


# 辅助函数
def create_sample_trade_data(result: str = 'win', profit_loss: float = 100.0) -> Dict:
    """创建示例交易数据"""
    market_regimes = ['trending', 'ranging', 'transition', 'volatile', 'calm']
    setup_types = ['breakout', 'reversal', 'pullback', 'range_trade']
    
    return {
        'result': result,
        'profit_loss': profit_loss if result == 'win' else -profit_loss,
        'risk_reward': np.random.uniform(1.2, 3.0),
        'setup_type': np.random.choice(setup_types),
        'market_conditions': {
            'regime': np.random.choice(market_regimes),
            'trend_strength': np.random.uniform(0.3, 0.9),
            'volatility': np.random.uniform(0.01, 0.03)
        },
        'instrument': 'EURUSD',
        'timeframe': np.random.choice(['1h', '4h', '1d']),
        'entry_reason': '价格突破关键阻力位',
        'exit_reason': '达到止盈目标'
    }


def demo_continuous_improvement_system():
    """演示持续改进系统"""
    print("=" * 60)
    print("持续改进系统演示")
    print("第26章：持续改进 - AL Brooks《价格行为交易之区间篇》")
    print("=" * 60)
    
    # 创建系统实例
    system = ContinuousImprovementSystem()
    
    print("\n1. 添加交易结果进行分析...")
    
    # 添加一些示例交易
    sample_trades = [
        create_sample_trade_data('win', 150.0),
        create_sample_trade_data('loss', 100.0),
        create_sample_trade_data('win', 200.0),
        create_sample_trade_data('win', 120.0),
        create_sample_trade_data('loss', 80.0)
    ]
    
    for i, trade in enumerate(sample_trades, 1):
        result = system.add_trade_result(trade)
        print(f"   交易{i}: {trade['result']} ${trade['profit_loss']:.0f}, 分析完成")
    
    print("\n2. 生成改进计划...")
    improvement_plan = system.generate_improvement_plan()
    print(f"   计划ID: {improvement_plan['plan_id']}")
    print(f"   重点领域: {list(improvement_plan['focus_areas'].keys())}")
    print(f"   行动项: {len(improvement_plan['action_items'])}个")
    
    print("\n3. 获取性能报告...")
    performance_report = system.get_performance_report('all')
    print(f"   总交易分析: {performance_report['summary']['total_trades_analyzed']}")
    print(f"   当前胜率: {performance_report['summary']['current_win_rate']:.1%}")
    print(f"   改进趋势: {performance_report['summary']['improvement_trend']}")
    
    print("\n4. 导出学习数据...")
    export_data = system.export_learning_data('dict')
    print(f"   导出时间: {export_data['export_time']}")
    print(f"   交易记录: {len(export_data['recent_trades'])}笔")
    print(f"   改进记录: {len(export_data['recent_improvements'])}个")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("持续改进系统已成功创建并测试")
    print("=" * 60)


# ============================================================================
# 策略改造: 添加PriceActionRangesContinuousImprovementSystemStrategy类
# 将价格行为区间持续改进系统转换为交易策略
# ============================================================================

class PriceActionRangesContinuousImprovementSystemStrategy(BaseStrategy):
    """价格行为区间持续改进策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 创建持续改进系统实例
        self.improvement_system = ContinuousImprovementSystem()
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于持续改进分析生成交易信号
        """
        # 获取性能报告
        performance_report = self.improvement_system.get_performance_report()
        
        # 分析性能数据
        summary = performance_report.get('summary', {})
        current_win_rate = summary.get('current_win_rate', 0)
        improvement_trend = summary.get('improvement_trend', '').lower()
        
        # 根据改进趋势生成信号
        if improvement_trend == 'improving' and current_win_rate >= 0.6:
            # 改进趋势且高胜率，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif improvement_trend == 'declining' or current_win_rate <= 0.4:
            # 下降趋势或低胜率，卖出信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='sell',
                price=self.data['close'].iloc[-1]
            )
        else:
            # 稳定趋势，hold信号
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
    demo_continuous_improvement_system()