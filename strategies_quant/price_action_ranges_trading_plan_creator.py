# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.644841

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易计划制定量化系统
第23章：交易计划制定
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 市场分析：趋势识别、支撑阻力、波动性评估
2. 交易目标：盈利目标、风险目标、时间框架
3. 入场规则：触发条件、入场价格、入场时机
4. 出场规则：止损设置、止盈设置、移动止损
5. 风险管理：仓位大小、风险比例、风险控制
6. 执行计划：执行条件、监控要点、应急方案
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

class TradingPlanCreator:
    """交易计划制定器（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 trader_profile: Dict = None,
                 default_risk_percentage: float = 0.02):
        """
        初始化交易计划制定器
        
        参数:
            trader_profile: 交易者个人资料
            default_risk_percentage: 默认风险比例
        """
        self.trader_profile = trader_profile or {
            'experience_level': 'intermediate',
            'trading_style': 'swing',
            'risk_tolerance': 'moderate',
            'preferred_timeframe': '4h'
        }
        
        self.default_risk_percentage = default_risk_percentage
        self.plans_history = []
        self.plan_templates = self._initialize_plan_templates()
    
    def create_comprehensive_plan(self,
                                 market_analysis: Dict,
                                 trade_idea: Dict,
                                 risk_parameters: Dict) -> Dict:
        """
        创建综合交易计划
        
        参数:
            market_analysis: 市场分析结果
            trade_idea: 交易想法
            risk_parameters: 风险参数
            
        返回:
            完整的交易计划
        """
        # 输入验证
        if not market_analysis or not trade_idea:
            return {'error': '市场分析或交易想法为空'}
        
        # 生成唯一计划ID
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 分析市场条件
        market_conditions = self._analyze_market_conditions(market_analysis)
        
        # 制定交易目标
        trading_objectives = self._define_trading_objectives(trade_idea, risk_parameters)
        
        # 定义入场规则
        entry_rules = self._define_entry_rules(trade_idea, market_conditions)
        
        # 定义出场规则
        exit_rules = self._define_exit_rules(trade_idea, risk_parameters, market_conditions)
        
        # 制定风险管理方案
        risk_management = self._define_risk_management(risk_parameters, trade_idea, market_conditions)
        
        # 制定执行计划
        execution_plan = self._define_execution_plan(entry_rules, exit_rules, risk_management)
        
        # 验证计划可行性
        plan_validation = self._validate_trading_plan(
            market_conditions, entry_rules, exit_rules, risk_management
        )
        
        # 生成执行清单
        execution_checklist = self._generate_execution_checklist(
            entry_rules, exit_rules, risk_management, plan_validation
        )
        
        # 组装完整计划
        comprehensive_plan = {
            'plan_id': plan_id,
            'creation_time': datetime.now(),
            'trader_profile': self.trader_profile,
            'market_analysis': market_analysis,
            'market_conditions': market_conditions,
            'trade_idea': trade_idea,
            'trading_objectives': trading_objectives,
            'entry_rules': entry_rules,
            'exit_rules': exit_rules,
            'risk_management': risk_management,
            'execution_plan': execution_plan,
            'plan_validation': plan_validation,
            'execution_checklist': execution_checklist,
            'plan_summary': self._generate_plan_summary(
                trading_objectives, entry_rules, exit_rules, risk_management, plan_validation
            ),
            'recommendations': self._generate_plan_recommendations(plan_validation)
        }
        
        self.plans_history.append(comprehensive_plan)
        return comprehensive_plan
    
    def _initialize_plan_templates(self) -> Dict:
        """初始化计划模板"""
        templates = {
            'trend_following': {
                'description': '趋势跟随计划模板',
                'entry_condition': '趋势确认后的回调入场',
                'exit_condition': '趋势反转或达到盈利目标',
                'risk_management': '宽止损，跟踪止损',
                'position_size': '正常仓位'
            },
            'range_trading': {
                'description': '区间交易计划模板',
                'entry_condition': '区间边界附近入场',
                'exit_condition': '区间另一端或突破出场',
                'risk_management': '窄止损，严格风险管理',
                'position_size': '小到中等仓位'
            },
            'breakout_trading': {
                'description': '突破交易计划模板',
                'entry_condition': '突破关键水平后入场',
                'exit_condition': '突破延伸或失败出场',
                'risk_management': '窄止损，快速止损',
                'position_size': '小仓位'
            },
            'reversal_trading': {
                'description': '反转交易计划模板',
                'entry_condition': '趋势衰竭信号入场',
                'exit_condition': '反转确认或失败出场',
                'risk_management': '宽止损，高风险',
                'position_size': '小仓位'
            }
        }
        return templates
    
    def _analyze_market_conditions(self, market_analysis: Dict) -> Dict:
        """分析市场条件"""
        market_conditions = {
            'trend_strength': market_analysis.get('trend_strength', 0.5),
            'market_structure': market_analysis.get('market_structure', 'unknown'),
            'volatility': market_analysis.get('volatility', 0.01),
            'support_levels': market_analysis.get('support_levels', []),
            'resistance_levels': market_analysis.get('resistance_levels', []),
            'market_regime': 'neutral',
            'trading_opportunity': 'medium'
        }
        
        # 确定市场体制
        trend_strength = market_conditions['trend_strength']
        market_structure = market_conditions['market_structure']
        
        if trend_strength > 0.7:
            if market_structure in ['uptrend', 'downtrend']:
                market_conditions['market_regime'] = 'trending'
                market_conditions['trading_opportunity'] = 'high' if trend_strength > 0.8 else 'medium'
        elif trend_strength < 0.3:
            market_conditions['market_regime'] = 'ranging'
            market_conditions['trading_opportunity'] = 'medium'
        else:
            market_conditions['market_regime'] = 'transition'
            market_conditions['trading_opportunity'] = 'low'
        
        # 评估交易机会
        volatility = market_conditions['volatility']
        if volatility > 0.02:
            market_conditions['trading_opportunity'] = min(
                market_conditions['trading_opportunity'],
                'medium'  # 高波动性降低机会评级
            )
        elif volatility < 0.005:
            market_conditions['trading_opportunity'] = min(
                market_conditions['trading_opportunity'],
                'low'  # 低波动性降低机会评级
            )
        
        return market_conditions
    
    def _define_trading_objectives(self,
                                  trade_idea: Dict,
                                  risk_parameters: Dict) -> Dict:
        """定义交易目标"""
        objectives = {
            'primary_objective': 'capital_preservation',  # 资本保值
            'profit_target_percentage': risk_parameters.get('profit_target', 0.03),
            'risk_target_percentage': risk_parameters.get('max_risk', 0.02),
            'time_horizon': trade_idea.get('time_horizon', 'short_term'),
            'success_metrics': []
        }
        
        # 根据交易类型调整目标
        trade_type = trade_idea.get('type', 'speculative')
        
        if trade_type == 'investment':
            objectives['primary_objective'] = 'long_term_growth'
            objectives['profit_target_percentage'] = 0.10
            objectives['time_horizon'] = 'long_term'
            objectives['success_metrics'] = ['annual_return', 'sharpe_ratio', 'max_drawdown']
        
        elif trade_type == 'speculative':
            objectives['primary_objective'] = 'short_term_profit'
            objectives['profit_target_percentage'] = 0.05
            objectives['time_horizon'] = 'short_term'
            objectives['success_metrics'] = ['win_rate', 'profit_factor', 'risk_reward_ratio']
        
        elif trade_type == 'hedge':
            objectives['primary_objective'] = 'risk_reduction'
            objectives['profit_target_percentage'] = 0.01
            objectives['time_horizon'] = 'medium_term'
            objectives['success_metrics'] = ['correlation', 'volatility_reduction', 'downside_protection']
        
        # 计算风险回报比
        risk_reward_ratio = objectives['profit_target_percentage'] / objectives['risk_target_percentage']
        objectives['risk_reward_ratio'] = max(1.0, risk_reward_ratio)
        
        # 设置具体目标
        objectives['specific_goals'] = [
            f"实现{objectives['profit_target_percentage']:.1%}的盈利目标",
            f"将风险控制在{objectives['risk_target_percentage']:.1%}以内",
            f"风险回报比不低于{objectives['risk_reward_ratio']:.1f}:1",
            f"持仓时间：{objectives['time_horizon']}"
        ]
        
        return objectives
    
    def _define_entry_rules(self,
                           trade_idea: Dict,
                           market_conditions: Dict) -> Dict:
        """定义入场规则"""
        entry_rules = {
            'entry_type': trade_idea.get('entry_type', 'limit'),
            'entry_conditions': [],
            'entry_price': trade_idea.get('entry_price', 0),
            'entry_timeframe': trade_idea.get('timeframe', '4h'),
            'entry_confirmation': trade_idea.get('confirmation_required', True),
            'max_attempts': 3,
            'expiration_time': None
        }
        
        # 根据市场条件设置入场条件
        market_regime = market_conditions['market_regime']
        trade_direction = trade_idea.get('direction', 'long')
        
        if market_regime == 'trending':
            if trade_direction == 'long':
                entry_rules['entry_conditions'].extend([
                    '价格回调至趋势线或移动平均线支撑',
                    '出现看涨反转形态',
                    '成交量配合增加',
                    '关键支撑位未跌破'
                ])
            else:  # short
                entry_rules['entry_conditions'].extend([
                    '价格反弹至趋势线或移动平均线阻力',
                    '出现看跌反转形态',
                    '成交量配合增加',
                    '关键阻力位未突破'
                ])
        
        elif market_regime == 'ranging':
            support_levels = market_conditions.get('support_levels', [])
            resistance_levels = market_conditions.get('resistance_levels', [])
            
            if trade_direction == 'long' and support_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x - trade_idea.get('current_price', 0)))
                entry_rules['entry_conditions'].extend([
                    f'价格接近支撑位{nearest_support:.2f}',
                    '出现看涨反转信号',
                    '区间下轨附近成交量放大',
                    '支撑位多次测试有效'
                ])
                entry_rules['entry_price'] = nearest_support * 0.995  # 稍低于支撑位
            
            elif trade_direction == 'short' and resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - trade_idea.get('current_price', 0)))
                entry_rules['entry_conditions'].extend([
                    f'价格接近阻力位{nearest_resistance:.2f}',
                    '出现看跌反转信号',
                    '区间上轨附近成交量放大',
                    '阻力位多次测试有效'
                ])
                entry_rules['entry_price'] = nearest_resistance * 1.005  # 稍高于阻力位
        
        elif market_regime == 'transition':
            entry_rules['entry_conditions'].extend([
                '等待市场方向确认',
                '出现明确突破信号',
                '成交量显著放大',
                '关键水平被有效突破'
            ])
            entry_rules['entry_confirmation'] = True
            entry_rules['max_attempts'] = 1  # 转换期只尝试一次
        
        # 设置入场有效期
        if entry_rules['entry_timeframe'] == '1h':
            entry_rules['expiration_time'] = datetime.now() + timedelta(hours=4)
        elif entry_rules['entry_timeframe'] == '4h':
            entry_rules['expiration_time'] = datetime.now() + timedelta(hours=12)
        elif entry_rules['entry_timeframe'] == '1d':
            entry_rules['expiration_time'] = datetime.now() + timedelta(days=3)
        
        return entry_rules
    
    def _define_exit_rules(self,
                          trade_idea: Dict,
                          risk_parameters: Dict,
                          market_conditions: Dict) -> Dict:
        """定义出场规则"""
        entry_price = trade_idea.get('entry_price', 0)
        trade_direction = trade_idea.get('direction', 'long')
        
        # 计算止损和止盈
        if trade_direction == 'long':
            stop_loss = entry_price * (1 - risk_parameters.get('stop_loss_percentage', 0.02))
            take_profit = entry_price * (1 + risk_parameters.get('take_profit_percentage', 0.04))
        else:  # short
            stop_loss = entry_price * (1 + risk_parameters.get('stop_loss_percentage', 0.02))
            take_profit = entry_price * (1 - risk_parameters.get('take_profit_percentage', 0.04))
        
        exit_rules = {
            'stop_loss': {
                'price': stop_loss,
                'type': 'fixed',
                'reason': '初始风险控制',
                'adjustable': True
            },
            'take_profit': {
                'price': take_profit,
                'type': 'fixed',
                'levels': self._calculate_take_profit_levels(entry_price, take_profit, trade_direction),
                'partial_exit_enabled': True
            },
            'trailing_stop': {
                'enabled': market_conditions['market_regime'] == 'trending',
                'activation_price': entry_price * (1.01 if trade_direction == 'long' else 0.99),
                'distance_percentage': 0.01,
                'type': 'atr_based' if market_conditions.get('volatility', 0.01) > 0.015 else 'percentage'
            },
            'exit_conditions': [],
            'emergency_exit': {
                'price': stop_loss * (0.95 if trade_direction == 'long' else 1.05),
                'reason': '极端市场情况',
                'trigger_conditions': ['重大新闻事件', '市场崩盘', '流动性枯竭']
            }
        }
        
        # 根据市场体制添加出场条件
        market_regime = market_conditions['market_regime']
        
        if market_regime == 'trending':
            exit_rules['exit_conditions'].extend([
                '趋势线或移动平均线被突破',
                '出现趋势反转信号',
                '动量指标背离',
                '达到盈利目标'
            ])
            exit_rules['trailing_stop']['enabled'] = True
        
        elif market_regime == 'ranging':
            exit_rules['exit_conditions'].extend([
                '价格到达区间另一端',
                '区间突破确认',
                '价格在区间内震荡超过3次',
                '达到风险回报目标'
            ])
            exit_rules['take_profit']['partial_exit_enabled'] = True
        
        elif market_regime == 'transition':
            exit_rules['exit_conditions'].extend([
                '新趋势确认成立',
                '转换失败，价格返回原区间',
                '波动性急剧增加',
                '达到时间止损'
            ])
            exit_rules['stop_loss']['adjustable'] = False  # 转换期固定止损
        
        # 计算风险回报比
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)
        exit_rules['risk_reward_ratio'] = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return exit_rules
    
    def _calculate_take_profit_levels(self,
                                     entry_price: float,
                                     final_take_profit: float,
                                     direction: str) -> List[Dict]:
        """计算分批止盈水平"""
        levels = []
        
        if direction == 'long':
            price_range = final_take_profit - entry_price
            levels.append({
                'level': 1,
                'price': entry_price + price_range * 0.5,
                'percentage': 0.5,
                'reason': '第一目标位，锁定部分利润'
            })
            levels.append({
                'level': 2,
                'price': entry_price + price_range * 0.8,
                'percentage': 0.3,
                'reason': '第二目标位，继续持仓'
            })
            levels.append({
                'level': 3,
                'price': final_take_profit,
                'percentage': 0.2,
                'reason': '最终目标位，完全出场'
            })
        else:  # short
            price_range = entry_price - final_take_profit
            levels.append({
                'level': 1,
                'price': entry_price - price_range * 0.5,
                'percentage': 0.5,
                'reason': '第一目标位，锁定部分利润'
            })
            levels.append({
                'level': 2,
                'price': entry_price - price_range * 0.8,
                'percentage': 0.3,
                'reason': '第二目标位，继续持仓'
            })
            levels.append({
                'level': 3,
                'price': final_take_profit,
                'percentage': 0.2,
                'reason': '最终目标位，完全出场'
            })
        
        return levels
    
    def _define_risk_management(self,
                               risk_parameters: Dict,
                               trade_idea: Dict,
                               market_conditions: Dict) -> Dict:
        """定义风险管理方案"""
        account_size = risk_parameters.get('account_size', 10000.0)
        max_risk_per_trade = risk_parameters.get('max_risk_per_trade', self.default_risk_percentage)
        
        # 计算仓位大小
        entry_price = trade_idea.get('entry_price', 0)
        stop_loss = risk_parameters.get('stop_loss_percentage', 0.02)
        
        if trade_idea.get('direction', 'long') == 'long':
            risk_per_unit = entry_price - (entry_price * (1 - stop_loss))
        else:
            risk_per_unit = (entry_price * (1 + stop_loss)) - entry_price
        
        max_risk_amount = account_size * max_risk_per_trade
        
        if risk_per_unit > 0:
            position_size = max_risk_amount / risk_per_unit
        else:
            position_size = 0
        
        # 考虑市场波动性调整
        volatility = market_conditions.get('volatility', 0.01)
        if volatility > 0.02:
            position_size *= 0.7  # 高波动性减小仓位
        elif volatility < 0.005:
            position_size *= 1.3  # 低波动性增加仓位
        
        # 应用仓位限制
        max_position_size = account_size * 0.1  # 最大仓位10%
        position_size = min(position_size, max_position_size)
        
        risk_management = {
            'account_size': account_size,
            'max_risk_per_trade': max_risk_per_trade,
            'max_risk_amount': max_risk_amount,
            'position_size': position_size,
            'position_value': position_size * entry_price,
            'risk_per_unit': risk_per_unit,
            'max_drawdown_limit': risk_parameters.get('max_drawdown_limit', 0.2),
            'daily_loss_limit': risk_parameters.get('daily_loss_limit', 0.05),
            'weekly_loss_limit': risk_parameters.get('weekly_loss_limit', 0.15),
            'risk_adjustments': [],
            'hedging_strategy': None
        }
        
        # 根据市场条件添加风险调整
        market_regime = market_conditions['market_regime']
        
        if market_regime == 'transition':
            risk_management['risk_adjustments'].append({
                'type': 'position_reduction',
                'adjustment': -0.5,
                'reason': '市场转换期，降低风险暴露'
            })
            risk_management['position_size'] *= 0.5
        
        elif market_conditions.get('volatility', 0.01) > 0.02:
            risk_management['risk_adjustments'].append({
                'type': 'wider_stop_loss',
                'adjustment': 1.5,
                'reason': '高波动性市场，放宽止损'
            })
        
        # 设置对冲策略
        if risk_parameters.get('hedging_enabled', False):
            risk_management['hedging_strategy'] = {
                'type': 'correlation_hedge',
                'target_correlation': -0.7,
                'hedge_ratio': 0.3,
                'instrument': risk_parameters.get('hedge_instrument', '相关资产')
            }
        
        return risk_management
    
    def _define_execution_plan(self,
                              entry_rules: Dict,
                              exit_rules: Dict,
                              risk_management: Dict) -> Dict:
        """定义执行计划"""
        execution_plan = {
            'pre_trade_checklist': [
                '确认网络连接稳定',
                '检查账户资金充足',
                '确认交易平台正常',
                '设置价格提醒',
                '准备交易日志'
            ],
            'entry_execution': {
                'order_type': entry_rules['entry_type'],
                'price': entry_rules['entry_price'],
                'quantity': risk_management['position_size'],
                'validity': 'GTC' if entry_rules['expiration_time'] else 'DAY',
                'contingency_orders': [
                    {'type': 'stop', 'price': exit_rules['stop_loss']['price'], 'reason': '初始止损'},
                    {'type': 'limit', 'price': exit_rules['take_profit']['levels'][0]['price'], 'reason': '部分止盈'}
                ]
            },
            'monitoring_plan': {
                'check_frequency': '每小时' if entry_rules['entry_timeframe'] in ['1h', '4h'] else '每天',
                'key_levels_to_watch': [],
                'signals_to_monitor': ['价格行为', '成交量', '技术指标'],
                'news_events': ['经济数据发布', '公司财报', '央行决议']
            },
            'exit_execution': {
                'primary_exit': {
                    'type': 'OCO',  # 一对单
                    'orders': [
                        {'type': 'stop', 'price': exit_rules['stop_loss']['price'], 'reason': '止损出场'},
                        {'type': 'limit', 'price': exit_rules['take_profit']['price'], 'reason': '止盈出场'}
                    ]
                },
                'partial_exits': exit_rules['take_profit']['levels'],
                'trailing_stop': exit_rules['trailing_stop']
            },
            'contingency_plan': {
                'technical_failure': '使用备用设备或移动应用',
                'market_gap': '等待价格稳定后重新评估',
                'news_event': '暂停交易，等待市场消化',
                'emotions_high': '暂停交易，执行冷静程序'
            }
        }
        
        # 添加关键价位监控
        if exit_rules['stop_loss']['price'] > 0:
            execution_plan['monitoring_plan']['key_levels_to_watch'].append(
                f"止损位: {exit_rules['stop_loss']['price']:.2f}"
            )
        
        for level in exit_rules['take_profit']['levels']:
            execution_plan['monitoring_plan']['key_levels_to_watch'].append(
                f"止盈{level['level']}: {level['price']:.2f}"
            )
        
        return execution_plan
    
    def _validate_trading_plan(self,
                              market_conditions: Dict,
                              entry_rules: Dict,
                              exit_rules: Dict,
                              risk_management: Dict) -> Dict:
        """验证交易计划可行性"""
        validation_results = {
            'overall_viability': 'pending',
            'score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'risks': [],
            'recommendations': []
        }
        
        score_components = []
        
        # 1. 市场条件验证
        market_regime = market_conditions['market_regime']
        trading_opportunity = market_conditions['trading_opportunity']
        
        if trading_opportunity == 'high':
            score_components.append(0.9)
            validation_results['strengths'].append('市场机会评级高')
        elif trading_opportunity == 'medium':
            score_components.append(0.7)
            validation_results['strengths'].append('市场机会评级中等')
        else:
            score_components.append(0.4)
            validation_results['weaknesses'].append('市场机会评级低')
        
        # 2. 入场规则验证
        entry_conditions_count = len(entry_rules.get('entry_conditions', []))
        if entry_conditions_count >= 3:
            score_components.append(0.8)
            validation_results['strengths'].append('入场条件明确具体')
        elif entry_conditions_count >= 1:
            score_components.append(0.6)
            validation_results['weaknesses'].append('入场条件较少，需要更多确认')
        else:
            score_components.append(0.3)
            validation_results['weaknesses'].append('缺乏明确的入场条件')
        
        # 3. 风险回报验证
        risk_reward_ratio = exit_rules.get('risk_reward_ratio', 0)
        if risk_reward_ratio >= 2.0:
            score_components.append(0.9)
            validation_results['strengths'].append(f'风险回报比优秀 ({risk_reward_ratio:.1f}:1)')
        elif risk_reward_ratio >= 1.5:
            score_components.append(0.7)
            validation_results['strengths'].append(f'风险回报比良好 ({risk_reward_ratio:.1f}:1)')
        elif risk_reward_ratio >= 1.0:
            score_components.append(0.5)
            validation_results['weaknesses'].append(f'风险回报比一般 ({risk_reward_ratio:.1f}:1)')
        else:
            score_components.append(0.2)
            validation_results['weaknesses'].append(f'风险回报比不足 ({risk_reward_ratio:.1f}:1)')
        
        # 4. 仓位管理验证
        position_value = risk_management.get('position_value', 0)
        account_size = risk_management.get('account_size', 10000)
        position_percentage = position_value / account_size if account_size > 0 else 0
        
        if position_percentage <= 0.1:  # 10%以内
            score_components.append(0.8)
            validation_results['strengths'].append(f'仓位大小合理 ({position_percentage:.1%})')
        elif position_percentage <= 0.2:  # 20%以内
            score_components.append(0.6)
            validation_results['weaknesses'].append(f'仓位偏大 ({position_percentage:.1%})')
        else:
            score_components.append(0.3)
            validation_results['weaknesses'].append(f'仓位过大 ({position_percentage:.1%})')
        
        # 5. 退出策略验证
        trailing_stop_enabled = exit_rules.get('trailing_stop', {}).get('enabled', False)
        partial_exit_enabled = exit_rules.get('take_profit', {}).get('partial_exit_enabled', False)
        
        exit_strategy_score = 0.5
        if trailing_stop_enabled and partial_exit_enabled:
            exit_strategy_score = 0.9
            validation_results['strengths'].append('退出策略完善（移动止损+分批止盈）')
        elif trailing_stop_enabled or partial_exit_enabled:
            exit_strategy_score = 0.7
            validation_results['strengths'].append('退出策略良好')
        else:
            exit_strategy_score = 0.4
            validation_results['weaknesses'].append('退出策略单一')
        
        score_components.append(exit_strategy_score)
        
        # 计算总体分数
        if score_components:
            overall_score = np.mean(score_components)
        else:
            overall_score = 0.5
        
        validation_results['score'] = overall_score
        
        # 确定可行性
        if overall_score >= 0.8:
            validation_results['overall_viability'] = 'high'
            validation_results['recommendations'].append('计划可行性高，建议执行')
        elif overall_score >= 0.6:
            validation_results['overall_viability'] = 'medium'
            validation_results['recommendations'].append('计划可行性中等，建议微调后执行')
        else:
            validation_results['overall_viability'] = 'low'
            validation_results['recommendations'].append('计划可行性低，建议重新制定')
        
        # 识别风险
        if market_regime == 'transition':
            validation_results['risks'].append('市场处于转换期，方向不明确')
        
        if risk_management.get('volatility', 0.01) > 0.02:
            validation_results['risks'].append('市场波动性高，价格可能大幅波动')
        
        if entry_rules.get('max_attempts', 1) < 2:
            validation_results['risks'].append('入场尝试次数有限，可能错过机会')
        
        return validation_results
    
    def _generate_execution_checklist(self,
                                     entry_rules: Dict,
                                     exit_rules: Dict,
                                     risk_management: Dict,
                                     plan_validation: Dict) -> List[Dict]:
        """生成执行检查清单"""
        checklist = []
        
        # 入场前检查
        checklist.append({
            'phase': 'pre_entry',
            'items': [
                {
                    'item': '确认所有入场条件满足',
                    'checked': False,
                    'importance': 'critical'
                },
                {
                    'item': '检查市场无重大新闻事件',
                    'checked': False,
                    'importance': 'high'
                },
                {
                    'item': '确认账户资金充足',
                    'checked': False,
                    'importance': 'critical'
                },
                {
                    'item': '设置止损和止盈订单',
                    'checked': False,
                    'importance': 'critical'
                },
                {
                    'item': '记录交易计划到日志',
                    'checked': False,
                    'importance': 'medium'
                }
            ]
        })
        
        # 入场执行检查
        checklist.append({
            'phase': 'entry_execution',
            'items': [
                {
                    'item': f"按计划价格入场: {entry_rules['entry_price']:.2f}",
                    'checked': False,
                    'importance': 'critical'
                },
                {
                    'item': f"设置止损: {exit_rules['stop_loss']['price']:.2f}",
                    'checked': False,
                    'importance': 'critical'
                },
                {
                    'item': f"设置第一目标止盈: {exit_rules['take_profit']['levels'][0]['price']:.2f}",
                    'checked': False,
                    'importance': 'high'
                },
                {
                    'item': '确认订单已成交',
                    'checked': False,
                    'importance': 'critical'
                }
            ]
        })
        
        # 持仓监控检查
        checklist.append({
            'phase': 'position_monitoring',
            'items': [
                {
                    'item': '每日检查持仓状态',
                    'checked': False,
                    'importance': 'high'
                },
                {
                    'item': '监控关键技术水平',
                    'checked': False,
                    'importance': 'medium'
                },
                {
                    'item': '关注相关新闻事件',
                    'checked': False,
                    'importance': 'medium'
                },
                {
                    'item': '更新交易日志',
                    'checked': False,
                    'importance': 'medium'
                },
                {
                    'item': '评估是否需要调整止损',
                    'checked': False,
                    'importance': 'high'
                }
            ]
        })
        
        # 出场执行检查
        checklist.append({
            'phase': 'exit_execution',
            'items': [
                {
                    'item': '触发止损时立即出场',
                    'checked': False,
                    'importance': 'critical'
                },
                {
                    'item': '按计划执行分批止盈',
                    'checked': False,
                    'importance': 'high'
                },
                {
                    'item': '移动止损触发时出场',
                    'checked': False,
                    'importance': 'high'
                },
                {
                    'item': '记录出场原因和结果',
                    'checked': False,
                    'importance': 'medium'
                },
                {
                    'item': '进行交易后分析',
                    'checked': False,
                    'importance': 'medium'
                }
            ]
        })
        
        # 添加基于验证结果的额外检查项
        if plan_validation['overall_viability'] == 'medium':
            checklist[0]['items'].append({
                'item': '额外确认市场条件（可行性中等）',
                'checked': False,
                'importance': 'high'
            })
        
        if any('高波动性' in risk for risk in plan_validation.get('risks', [])):
            checklist[2]['items'].append({
                'item': '特别注意价格波动（高波动性风险）',
                'checked': False,
                'importance': 'high'
            })
        
        return checklist
    
    def _generate_plan_summary(self,
                              trading_objectives: Dict,
                              entry_rules: Dict,
                              exit_rules: Dict,
                              risk_management: Dict,
                              plan_validation: Dict) -> Dict:
        """生成计划摘要"""
        summary = {
            'key_elements': {
                'market_regime': 'trending/ranging/transition',
                'trade_direction': entry_rules.get('direction', 'long'),
                'entry_price': entry_rules['entry_price'],
                'stop_loss': exit_rules['stop_loss']['price'],
                'take_profit': exit_rules['take_profit']['price'],
                'risk_reward_ratio': exit_rules.get('risk_reward_ratio', 0),
                'position_size': risk_management['position_size'],
                'max_risk_amount': risk_management['max_risk_amount']
            },
            'objectives_summary': [
                f"盈利目标: {trading_objectives['profit_target_percentage']:.1%}",
                f"风险目标: {trading_objectives['risk_target_percentage']:.1%}",
                f"时间框架: {trading_objectives['time_horizon']}",
                f"风险回报比: {trading_objectives.get('risk_reward_ratio', 0):.1f}:1"
            ],
            'execution_summary': [
                f"入场类型: {entry_rules['entry_type']}",
                f"入场条件: {len(entry_rules['entry_conditions'])}个",
                f"止损类型: {exit_rules['stop_loss']['type']}",
                f"止盈类型: {exit_rules['take_profit']['type']}",
                f"移动止损: {'启用' if exit_rules['trailing_stop']['enabled'] else '禁用'}"
            ],
            'risk_summary': [
                f"仓位比例: {risk_management['position_value'] / risk_management['account_size']:.1%}",
                f"单笔最大风险: {risk_management['max_risk_per_trade']:.1%}",
                f"最大亏损金额: ${risk_management['max_risk_amount']:.2f}",
                f"日亏损限制: {risk_management.get('daily_loss_limit', 0.05):.1%}",
                f"周亏损限制: {risk_management.get('weekly_loss_limit', 0.15):.1%}"
            ],
            'validation_summary': {
                'viability': plan_validation['overall_viability'],
                'score': plan_validation['score'],
                'strength_count': len(plan_validation['strengths']),
                'weakness_count': len(plan_validation['weaknesses']),
                'risk_count': len(plan_validation['risks'])
            }
        }
        
        return summary
    
    def _generate_plan_recommendations(self, plan_validation: Dict) -> List[Dict]:
        """生成计划建议"""
        recommendations = []
        
        viability = plan_validation['overall_viability']
        score = plan_validation['score']
        
        if viability == 'high':
            recommendations.append({
                'type': 'execution',
                'priority': 'high',
                'recommendation': '计划可行性高，建议按计划执行',
                'action': 'proceed_with_plan'
            })
        
        elif viability == 'medium':
            recommendations.append({
                'type': 'optimization',
                'priority': 'medium',
                'recommendation': '计划可行性中等，建议优化以下方面：',
                'actions': [
                    '加强入场条件确认',
                    '考虑减小仓位以降低风险',
                    '设置更严格的风险控制'
                ]
            })
            
            if score < 0.7:
                recommendations.append({
                    'type': 'monitoring',
                    'priority': 'high',
                    'recommendation': '计划需要密切监控，准备随时调整',
                    'action': 'close_monitoring_required'
                })
        
        else:  # low viability
            recommendations.append({
                'type': 'revision',
                'priority': 'critical',
                'recommendation': '计划可行性低，建议重新制定或放弃',
                'actions': [
                    '重新分析市场条件',
                    '调整交易参数',
                    '考虑其他交易机会'
                ]
            })
        
        # 基于弱点和风险添加具体建议
        for weakness in plan_validation.get('weaknesses', []):
            if '入场条件' in weakness:
                recommendations.append({
                    'type': 'entry_improvement',
                    'priority': 'medium',
                    'recommendation': f'改善入场条件: {weakness}',
                    'action': 'add_entry_confirmations'
                })
            
            if '风险回报比' in weakness:
                recommendations.append({
                    'type': 'risk_reward_improvement',
                    'priority': 'high',
                    'recommendation': f'改善风险回报比: {weakness}',
                    'action': 'adjust_profit_target_or_stop_loss'
                })
        
        for risk in plan_validation.get('risks', []):
            if '波动性' in risk:
                recommendations.append({
                    'type': 'volatility_management',
                    'priority': 'medium',
                    'recommendation': f'管理波动性风险: {risk}',
                    'action': 'reduce_position_or_widen_stop'
                })
            
            if '转换期' in risk:
                recommendations.append({
                    'type': 'market_timing',
                    'priority': 'high',
                    'recommendation': f'注意市场时机: {risk}',
                    'action': 'wait_for_confirmation'
                })
        
        # 按优先级排序
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def get_plan_history_summary(self, limit: int = 10) -> Dict:
        """获取计划历史摘要"""
        if not self.plans_history:
            return {'total_plans': 0, 'recent_plans': []}
        
        recent_plans = self.plans_history[-limit:] if len(self.plans_history) > limit else self.plans_history
        
        summary = {
            'total_plans': len(self.plans_history),
            'recent_plans_count': len(recent_plans),
            'viability_distribution': {},
            'average_validation_score': 0.0,
            'recent_plan_ids': [],
            'performance_trend': 'insufficient_data'
        }
        
        # 计算可行性分布
        viability_counts = {}
        validation_scores = []
        
        for plan in recent_plans:
            viability = plan.get('plan_validation', {}).get('overall_viability', 'unknown')
            viability_counts[viability] = viability_counts.get(viability, 0) + 1
            
            score = plan.get('plan_validation', {}).get('score', 0)
            validation_scores.append(score)
            
            summary['recent_plan_ids'].append(plan.get('plan_id', 'unknown'))
        
        summary['viability_distribution'] = viability_counts
        
        if validation_scores:
            summary['average_validation_score'] = np.mean(validation_scores)
        
        # 分析趋势（如果数据足够）
        if len(recent_plans) >= 3:
            recent_scores = [plan.get('plan_validation', {}).get('score', 0) for plan in recent_plans[-3:]]
            earlier_scores = [plan.get('plan_validation', {}).get('score', 0) for plan in recent_plans[:-3]] if len(recent_plans) > 3 else recent_scores
            
            avg_recent = np.mean(recent_scores)
            avg_earlier = np.mean(earlier_scores) if earlier_scores else avg_recent
            
            if avg_recent > avg_earlier + 0.05:
                summary['performance_trend'] = 'improving'
            elif avg_recent < avg_earlier - 0.05:
                summary['performance_trend'] = 'declining'
            else:
                summary['performance_trend'] = 'stable'
        
        return summary
    
    def export_plan_to_markdown(self, plan_id: str) -> str:
        """将计划导出为Markdown格式"""
        plan = None
        for p in self.plans_history:
            if p.get('plan_id') == plan_id:
                plan = p
                break
        
        if not plan:
            return f"# 计划未找到: {plan_id}"
        
        # 生成Markdown内容
        markdown = f"""# 交易计划: {plan_id}

## 基本信息
- **创建时间**: {plan['creation_time']}
- **交易者**: {plan['trader_profile'].get('experience_level', 'unknown')}级别
- **交易风格**: {plan['trader_profile'].get('trading_style', 'unknown')}

## 市场分析
- **市场体制**: {plan['market_conditions'].get('market_regime', 'unknown')}
- **趋势强度**: {plan['market_conditions'].get('trend_strength', 0):.0%}
- **波动性**: {plan['market_conditions'].get('volatility', 0):.1%}
- **交易机会评级**: {plan['market_conditions'].get('trading_opportunity', 'unknown')}

## 交易目标
"""
        
        for goal in plan['trading_objectives'].get('specific_goals', []):
            markdown += f"- {goal}\n"
        
        markdown += f"""
## 入场规则
- **入场价格**: ${plan['entry_rules'].get('entry_price', 0):.2f}
- **入场类型**: {plan['entry_rules'].get('entry_type', 'unknown')}
- **时间框架**: {plan['entry_rules'].get('entry_timeframe', 'unknown')}
- **入场条件**:
"""
        
        for condition in plan['entry_rules'].get('entry_conditions', []):
            markdown += f"  - {condition}\n"
        
        markdown += f"""
## 出场规则
- **止损价格**: ${plan['exit_rules'].get('stop_loss', {}).get('price', 0):.2f}
- **止盈价格**: ${plan['exit_rules'].get('take_profit', {}).get('price', 0):.2f}
- **风险回报比**: {plan['exit_rules'].get('risk_reward_ratio', 0):.1f}:1
- **移动止损**: {'启用' if plan['exit_rules'].get('trailing_stop', {}).get('enabled', False) else '禁用'}

## 风险管理
- **账户规模**: ${plan['risk_management'].get('account_size', 0):.2f}
- **仓位大小**: {plan['risk_management'].get('position_size', 0):.2f}单位
- **仓位价值**: ${plan['risk_management'].get('position_value', 0):.2f}
- **单笔最大风险**: {plan['risk_management'].get('max_risk_per_trade', 0):.1%}
- **最大亏损金额**: ${plan['risk_management'].get('max_risk_amount', 0):.2f}

## 计划验证
- **可行性**: {plan['plan_validation'].get('overall_viability', 'unknown')}
- **验证分数**: {plan['plan_validation'].get('score', 0):.1%}
- **优势数量**: {len(plan['plan_validation'].get('strengths', []))}
- **弱点数量**: {len(plan['plan_validation'].get('weaknesses', []))}
- **风险数量**: {len(plan['plan_validation'].get('risks', []))}

## 执行检查清单
请按阶段完成以下检查:
"""
        
        for phase in plan['execution_checklist']:
            markdown += f"\n### {phase['phase'].replace('_', ' ').title()}\n"
            for item in phase['items']:
                markdown += f"- [ ] **{item['importance'].upper()}**: {item['item']}\n"
        
        markdown += f"""
## 建议
"""
        
        for rec in plan.get('recommendations', []):
            markdown += f"- **{rec['priority'].upper()}**: {rec['recommendation']}\n"
        
        markdown += f"\n---\n*计划生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return markdown
    
    def reset_trader_profile(self, new_profile: Dict):
        """重置交易者资料"""
        self.trader_profile = new_profile


# ============================================================================
# 策略改造: 添加PriceActionRangesTradingPlanCreatorStrategy类
# 将价格行为区间交易计划制定系统转换为交易策略
# ============================================================================

class PriceActionRangesTradingPlanCreatorStrategy(BaseStrategy):
    """价格行为区间交易计划制定策略"""
    
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
            'trading_style': 'swing'
        })
        
        # 创建交易计划制定器实例
        self.plan_creator = TradingPlanCreator(trader_profile=trader_profile)
    
    def generate_signals(self):
        """
        生成交易信号

        基于交易计划制定生成交易信号，使用趋势强度和支撑阻力制定计划
        """
        df = self.data
        if len(df) < 30:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(df['close'].iloc[-1]))
            return self.signals

        close = df['close']
        high = df['high']
        low = df['low']

        # Trend strength via ADX-like measure
        ma_short = close.rolling(10).mean()
        ma_long = close.rolling(30).mean()
        trend_strength = ((ma_short - ma_long) / ma_long).abs().rolling(10).mean()

        # Support/resistance
        support = low.rolling(20).min()
        resistance = high.rolling(20).min()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        last_close = close.iloc[-1]
        last_trend = trend_strength.iloc[-1]
        last_rsi = rsi.iloc[-1]

        # Plan: strong trend + RSI favorable + MACD confirm = execute
        trend_up = ma_short.iloc[-1] > ma_long.iloc[-1]
        macd_confirm = macd_line.iloc[-1] > signal_line.iloc[-1]

        if trend_up and macd_confirm and last_rsi < 70 and last_trend > 0.01:
            self._record_signal(timestamp=df.index[-1], action='buy', price=float(last_close))
        elif not trend_up and not macd_confirm and last_rsi > 30 and last_trend > 0.01:
            self._record_signal(timestamp=df.index[-1], action='sell', price=float(last_close))
        elif last_rsi < 30 and trend_up:
            self._record_signal(timestamp=df.index[-1], action='buy', price=float(last_close))
        elif last_rsi > 70 and not trend_up:
            self._record_signal(timestamp=df.index[-1], action='sell', price=float(last_close))
        else:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(last_close))

        return self.signals