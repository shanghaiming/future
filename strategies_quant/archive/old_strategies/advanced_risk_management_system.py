# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.635757

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级风险管理量化系统
第27章：风险管理高级主题
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 高级风险模型：VaR（风险价值）、CVaR（条件风险价值）、极端风险度量
2. 压力测试：历史情景分析、假设情景分析、极端市场条件模拟
3. 相关性风险管理：资产相关性分析、分散化优化、集中度控制
4. 流动性风险管理：流动性指标监控、市场深度分析、退出策略
5. 杠杆管理：动态杠杆调整、保证金监控、强制平仓预防
6. 极端情况处理：黑天鹅事件预案、市场崩溃保护、应急方案
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
import math
import random
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class AdvancedRiskManagementSystem:
    """高级风险管理系统（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 portfolio_config: Dict = None,
                 risk_tolerance: Dict = None,
                 regulatory_limits: Dict = None):
        """初始化高级风险管理系统"""
        self.portfolio_config = portfolio_config or {
            'total_capital': 100000.0,
            'max_drawdown_limit': -0.20,  # 最大回撤限制-20%
            'var_confidence_level': 0.95,  # VaR置信水平95%
            'stress_test_scenarios': ['2008_crisis', '2020_covid', 'flash_crash'],
            'correlation_threshold': 0.7,  # 相关性阈值0.7
            'liquidity_buffer': 0.10  # 流动性缓冲10%
        }
        
        self.risk_tolerance = risk_tolerance or {
            'daily_var_limit': -0.02,  # 日VaR限制-2%
            'weekly_var_limit': -0.05,  # 周VaR限制-5%
            'max_position_concentration': 0.15,  # 最大仓位集中度15%
            'max_sector_concentration': 0.30,  # 最大行业集中度30%
            'leverage_limit': 3.0,  # 杠杆限制3倍
            'liquidity_requirement': 0.05  # 流动性要求5%
        }
        
        self.regulatory_limits = regulatory_limits or {
            'margin_requirement': 0.50,  # 保证金要求50%
            'position_limit': 0.25,  # 单品种仓位限制25%
            'short_selling_limit': 0.20,  # 卖空限制20%
            'derivatives_limit': 0.30  # 衍生品限制30%
        }
        
        # 风险指标历史
        self.risk_metrics_history = {
            'var_metrics': [],
            'stress_test_results': [],
            'correlation_analyses': [],
            'liquidity_assessments': [],
            'leverage_monitoring': [],
            'extreme_event_alerts': []
        }
        
        # 风险模型参数
        self.risk_models = {
            'var_model': {
                'method': 'historical_simulation',
                'lookback_period': 252,  # 1年历史数据
                'confidence_level': 0.95,
                'holding_period': 1  # 1天持有期
            },
            'cvar_model': {
                'method': 'expected_shortfall',
                'confidence_level': 0.95,
                'tail_percentile': 0.05  # 尾部5%
            },
            'stress_test_model': {
                'historical_scenarios': True,
                'hypothetical_scenarios': True,
                'reverse_stress_test': True,
                'monte_carlo_scenarios': 10000
            },
            'correlation_model': {
                'rolling_window': 60,  # 60天滚动窗口
                'dynamic_correlation': True,
                'clustering_method': 'hierarchical',
                'threshold_alerts': True
            },
            'liquidity_model': {
                'bid_ask_spread_monitoring': True,
                'market_depth_analysis': True,
                'volume_profile_tracking': True,
                'exit_time_estimation': True
            }
        }
        
        # 风险控制规则
        self.risk_control_rules = {
            'position_sizing_rules': {
                'kelly_criterion': True,
                'risk_parity': True,
                'volatility_targeting': True,
                'drawdown_control': True
            },
            'stop_loss_rules': {
                'time_based_stops': True,
                'volatility_based_stops': True,
                'correlation_based_stops': True,
                'portfolio_level_stops': True
            },
            'diversification_rules': {
                'min_correlation': -0.2,
                'max_correlation': 0.7,
                'sector_diversification': True,
                'geographic_diversification': True
            },
            'liquidity_rules': {
                'min_liquidity_score': 0.7,
                'max_position_size_to_volume': 0.05,
                'exit_time_limit_days': 3,
                'liquidity_buffer_maintenance': True
            }
        }
        
        # 当前风险状态
        self.current_risk_state = {
            'portfolio_var': 0.0,
            'portfolio_cvar': 0.0,
            'stress_test_score': 0.0,
            'correlation_risk_score': 0.0,
            'liquidity_risk_score': 0.0,
            'leverage_ratio': 1.0,
            'concentration_risk_score': 0.0,
            'overall_risk_level': 'low',
            'last_calculation_time': None
        }
    
    def calculate_var(self, portfolio_returns: List[float], confidence_level: float = None) -> Dict:
        """计算风险价值（VaR）"""
        if not portfolio_returns:
            return {'error': '投资组合收益率数据为空'}
        
        confidence = confidence_level or self.risk_models['var_model']['confidence_level']
        
        # 历史模拟法计算VaR
        returns_array = np.array(portfolio_returns)
        
        # 按升序排序
        sorted_returns = np.sort(returns_array)
        
        # 计算百分位数
        var_index = int(len(sorted_returns) * (1 - confidence))
        
        if var_index >= len(sorted_returns):
            var_index = len(sorted_returns) - 1
        
        historical_var = sorted_returns[var_index]
        
        # 参数法计算（假设正态分布）
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Z分数（标准正态分布）
        z_score = self._get_z_score(confidence)
        parametric_var = mean_return + z_score * std_return
        
        # 计算条件风险价值（CVaR/Expected Shortfall）
        tail_returns = sorted_returns[:var_index]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else historical_var
        
        var_result = {
            'calculation_method': 'historical_simulation',
            'confidence_level': confidence,
            'historical_var': float(historical_var),
            'parametric_var': float(parametric_var),
            'conditional_var': float(cvar),
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'num_observations': len(portfolio_returns),
            'calculation_time': datetime.now()
        }
        
        # 更新当前风险状态
        self.current_risk_state['portfolio_var'] = float(historical_var)
        self.current_risk_state['portfolio_cvar'] = float(cvar)
        self.current_risk_state['last_calculation_time'] = datetime.now()
        
        # 保存到历史
        self.risk_metrics_history['var_metrics'].append(var_result)
        
        return var_result
    
    def _get_z_score(self, confidence_level: float) -> float:
        """获取标准正态分布Z分数"""
        # 常用置信水平对应的Z分数
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.975: 1.960,
            0.99: 2.326,
            0.995: 2.576,
            0.999: 3.090
        }
        
        # 如果正好是常用值，返回对应Z分数
        if confidence_level in z_scores:
            return z_scores[confidence_level]
        
        # 否则使用近似公式
        # 这是一个简化的近似，实际应用中应该使用scipy.stats.norm.ppf
        from math import sqrt, log, pi
        
        c = confidence_level
        t = sqrt(-2.0 * log(1 - c))
        
        # Cornish-Fisher展开近似
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / \
            (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
        
        return z
    
    def run_stress_test(self, portfolio_positions: List[Dict], scenarios: List[str] = None) -> Dict:
        """运行压力测试"""
        if not portfolio_positions:
            return {'error': '投资组合仓位数据为空'}
        
        scenarios_to_run = scenarios or self.portfolio_config['stress_test_scenarios']
        
        stress_results = {
            'test_time': datetime.now(),
            'portfolio_positions': len(portfolio_positions),
            'total_portfolio_value': sum(pos.get('value', 0) for pos in portfolio_positions),
            'scenario_results': [],
            'overall_impact': 0.0,
            'worst_case_scenario': None,
            'breach_indicators': []
        }
        
        # 定义压力测试场景
        scenario_definitions = {
            '2008_crisis': {
                'name': '2008年金融危机',
                'description': '类似2008年金融危机的市场崩溃',
                'equity_decline': -0.50,  # 股票下跌50%
                'credit_spread_widening': 0.05,  # 信用利差扩大5%
                'volatility_increase': 0.40,  # 波动率增加40%
                'liquidity_dry_up': 0.70,  # 流动性枯竭70%
                'correlation_increase': 0.30  # 相关性增加30%
            },
            '2020_covid': {
                'name': '2020年新冠疫情',
                'description': '类似2020年新冠疫情的市场冲击',
                'equity_decline': -0.35,
                'volatility_increase': 0.60,
                'liquidity_dry_up': 0.40,
                'rate_cuts': -0.015,  # 利率下降1.5%
                'flight_to_quality': 0.25  # 避险情绪上升25%
            },
            'flash_crash': {
                'name': '闪电崩盘',
                'description': '类似2010年闪电崩盘的极端事件',
                'instant_decline': -0.20,  # 瞬时下跌20%
                'recovery_time_hours': 2,  # 恢复时间2小时
                'liquidity_disappearance': 0.90,  # 流动性消失90%
                'market_maker_withdrawal': 0.80,  # 做市商退出80%
                'circuit_breaker_triggered': True  # 触发熔断机制
            },
            'inflation_surge': {
                'name': '通胀飙升',
                'description': '通胀意外大幅上升',
                'inflation_increase': 0.05,  # 通胀上升5%
                'rate_hikes': 0.03,  # 利率上升3%
                'bond_decline': -0.15,  # 债券下跌15%
                'currency_depreciation': -0.10,  # 货币贬值10%
                'commodity_inflation': 0.25  # 商品通胀25%
            },
            'geopolitical_crisis': {
                'name': '地缘政治危机',
                'description': '重大地缘政治事件冲击',
                'risk_off_sentiment': 0.40,  # 风险规避情绪40%
                'oil_price_spike': 0.30,  # 油价飙升30%
                'safe_haven_rally': 0.15,  # 避险资产上涨15%
                'trade_disruption': 0.25,  # 贸易中断25%
                'currency_volatility': 0.35  # 货币波动35%
            }
        }
        
        total_impact = 0.0
        worst_scenario = None
        worst_impact = 0.0
        
        # 运行每个场景
        for scenario_key in scenarios_to_run:
            if scenario_key not in scenario_definitions:
                continue
            
            scenario = scenario_definitions[scenario_key]
            scenario_impact = self._calculate_scenario_impact(portfolio_positions, scenario)
            
            scenario_result = {
                'scenario_name': scenario['name'],
                'scenario_key': scenario_key,
                'description': scenario['description'],
                'portfolio_impact_percent': scenario_impact['portfolio_impact_percent'],
                'position_impacts': scenario_impact['position_impacts'],
                'risk_factors': scenario_impact['risk_factors'],
                'breach_limits': scenario_impact['breach_limits']
            }
            
            stress_results['scenario_results'].append(scenario_result)
            
            # 更新总体影响
            total_impact += scenario_impact['portfolio_impact_percent']
            
            # 跟踪最坏情况
            if scenario_impact['portfolio_impact_percent'] < worst_impact:
                worst_impact = scenario_impact['portfolio_impact_percent']
                worst_scenario = scenario_key
        
        # 计算平均影响
        if stress_results['scenario_results']:
            stress_results['overall_impact'] = total_impact / len(stress_results['scenario_results'])
            stress_results['worst_case_scenario'] = worst_scenario
            stress_results['worst_case_impact'] = worst_impact
        
        # 检查是否违反风险限制
        breach_indicators = self._check_stress_test_breaches(stress_results)
        stress_results['breach_indicators'] = breach_indicators
        
        # 计算压力测试分数（0-100，越高越好）
        stress_score = self._calculate_stress_test_score(stress_results)
        stress_results['stress_test_score'] = stress_score
        
        # 更新当前风险状态
        self.current_risk_state['stress_test_score'] = stress_score
        
        # 保存到历史
        self.risk_metrics_history['stress_test_results'].append(stress_results)
        
        return stress_results
    
    def _calculate_scenario_impact(self, portfolio_positions: List[Dict], scenario: Dict) -> Dict:
        """计算单个场景对投资组合的影响"""
        position_impacts = []
        total_portfolio_value = sum(pos.get('value', 0) for pos in portfolio_positions)
        total_impact_value = 0.0
        
        for position in portfolio_positions:
            position_value = position.get('value', 0)
            asset_type = position.get('asset_type', 'unknown')
            sector = position.get('sector', 'unknown')
            region = position.get('region', 'unknown')
            
            # 根据资产类型和场景参数计算影响
            impact_factor = self._get_impact_factor(asset_type, sector, region, scenario)
            
            # 计算具体影响
            position_impact = position_value * impact_factor
            total_impact_value += position_impact
            
            position_impacts.append({
                'position_id': position.get('id', 'unknown'),
                'asset_type': asset_type,
                'sector': sector,
                'original_value': position_value,
                'impact_factor': impact_factor,
                'impact_value': position_impact,
                'new_value': position_value + position_impact
            })
        
        # 计算投资组合整体影响百分比
        portfolio_impact_percent = (total_impact_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        
        # 识别主要风险因素
        risk_factors = self._identify_risk_factors(scenario, portfolio_positions)
        
        # 检查是否违反限制
        breach_limits = self._check_position_breaches(position_impacts)
        
        return {
            'portfolio_impact_percent': portfolio_impact_percent,
            'total_impact_value': total_impact_value,
            'position_impacts': position_impacts,
            'risk_factors': risk_factors,
            'breach_limits': breach_limits
        }
    
    def _get_impact_factor(self, asset_type: str, sector: str, region: str, scenario: Dict) -> float:
        """获取特定资产在给定场景下的影响因子"""
        # 基础影响因子
        impact_factor = 0.0
        
        # 根据资产类型调整
        asset_multipliers = {
            'equity': 1.0,
            'bond': 0.6,
            'commodity': 0.8,
            'currency': 0.4,
            'derivative': 1.2,
            'real_estate': 0.7,
            'cash': 0.0
        }
        
        base_multiplier = asset_multipliers.get(asset_type, 0.5)
        
        # 根据场景参数调整
        if 'equity_decline' in scenario:
            if asset_type == 'equity':
                impact_factor = scenario['equity_decline'] * base_multiplier
        
        elif 'bond_decline' in scenario:
            if asset_type == 'bond':
                impact_factor = scenario['bond_decline'] * base_multiplier
        
        elif 'volatility_increase' in scenario:
            # 高波动性对衍生品影响更大
            volatility_impact = scenario['volatility_increase'] * 0.01
            if asset_type == 'derivative':
                impact_factor = -volatility_impact * 1.5
            else:
                impact_factor = -volatility_impact * base_multiplier
        
        elif 'liquidity_dry_up' in scenario:
            # 流动性枯竭对所有资产都有影响
            liquidity_impact = scenario['liquidity_dry_up'] * 0.01
            liquidity_multipliers = {
                'equity': 0.8,
                'bond': 0.6,
                'commodity': 0.7,
                'currency': 0.3,
                'derivative': 1.0,
                'real_estate': 0.9
            }
            multiplier = liquidity_multipliers.get(asset_type, 0.5)
            impact_factor = -liquidity_impact * multiplier
        
        # 随机成分模拟不确定性
        random_component = random.uniform(-0.05, 0.05)
        impact_factor += random_component
        
        return impact_factor
    
    def _identify_risk_factors(self, scenario: Dict, portfolio_positions: List[Dict]) -> List[Dict]:
        """识别主要风险因素"""
        risk_factors = []
        
        # 分析投资组合对各类风险的暴露
        equity_exposure = sum(pos.get('value', 0) for pos in portfolio_positions if pos.get('asset_type') == 'equity')
        bond_exposure = sum(pos.get('value', 0) for pos in portfolio_positions if pos.get('asset_type') == 'bond')
        derivative_exposure = sum(pos.get('value', 0) for pos in portfolio_positions if pos.get('asset_type') == 'derivative')
        
        total_value = sum(pos.get('value', 0) for pos in portfolio_positions)
        
        if total_value > 0:
            if equity_exposure / total_value > 0.5:
                risk_factors.append({
                    'factor': 'equity_concentration',
                    'exposure_percent': (equity_exposure / total_value) * 100,
                    'risk_level': 'high' if equity_exposure / total_value > 0.7 else 'medium'
                })
            
            if derivative_exposure / total_value > 0.3:
                risk_factors.append({
                    'factor': 'derivative_exposure',
                    'exposure_percent': (derivative_exposure / total_value) * 100,
                    'risk_level': 'high' if derivative_exposure / total_value > 0.5 else 'medium'
                })
        
        # 检查场景特定风险
        if 'volatility_increase' in scenario and scenario['volatility_increase'] > 0.3:
            risk_factors.append({
                'factor': 'volatility_risk',
                'scenario_volatility': scenario['volatility_increase'],
                'risk_level': 'high'
            })
        
        if 'liquidity_dry_up' in scenario and scenario['liquidity_dry_up'] > 0.5:
            risk_factors.append({
                'factor': 'liquidity_risk',
                'scenario_liquidity_loss': scenario['liquidity_dry_up'],
                'risk_level': 'high'
            })
        
        return risk_factors
    
    def _check_position_breaches(self, position_impacts: List[Dict]) -> List[Dict]:
        """检查仓位是否违反限制"""
        breaches = []
        
        for impact in position_impacts:
            position_id = impact['position_id']
            new_value = impact['new_value']
            original_value = impact['original_value']
            
            # 检查亏损是否超过单笔限制
            if new_value < original_value * 0.7:  # 亏损超过30%
                breaches.append({
                    'position_id': position_id,
                    'breach_type': 'excessive_loss',
                    'loss_percent': ((original_value - new_value) / original_value) * 100,
                    'limit': 'max_loss_30%',
                    'breach_severity': 'high'
                })
            
            # 检查新价值是否低于最小仓位要求
            if new_value < 1000:  # 假设最小仓位1000
                breaches.append({
                    'position_id': position_id,
                    'breach_type': 'below_minimum_position',
                    'new_value': new_value,
                    'minimum_required': 1000,
                    'breach_severity': 'medium'
                })
        
        return breaches
    
    def _check_stress_test_breaches(self, stress_results: Dict) -> List[Dict]:
        """检查压力测试结果是否违反风险限制"""
        breaches = []
        
        # 检查总体影响
        overall_impact = stress_results.get('overall_impact', 0)
        if overall_impact < -20:  # 总体影响超过-20%
            breaches.append({
                'breach_type': 'overall_impact_exceeded',
                'impact_percent': overall_impact,
                'limit': '-20%',
                'breach_severity': 'high'
            })
        
        # 检查最坏情况影响
        worst_impact = stress_results.get('worst_case_impact', 0)
        if worst_impact < -30:  # 最坏情况影响超过-30%
            breaches.append({
                'breach_type': 'worst_case_impact_exceeded',
                'impact_percent': worst_impact,
                'limit': '-30%',
                'breach_severity': 'critical'
            })
        
        # 检查场景结果
        for scenario_result in stress_results.get('scenario_results', []):
            impact = scenario_result.get('portfolio_impact_percent', 0)
            scenario_name = scenario_result.get('scenario_name', 'unknown')
            
            if impact < -25:  # 单个场景影响超过-25%
                breaches.append({
                    'breach_type': 'scenario_impact_exceeded',
                    'scenario': scenario_name,
                    'impact_percent': impact,
                    'limit': '-25%',
                    'breach_severity': 'high'
                })
        
        return breaches
    
    def _calculate_stress_test_score(self, stress_results: Dict) -> float:
        """计算压力测试分数（0-100）"""
        base_score = 100.0
        
        # 总体影响扣分
        overall_impact = abs(stress_results.get('overall_impact', 0))
        if overall_impact > 10:
            deduction = min(30, (overall_impact - 10) * 2)
            base_score -= deduction
        
        # 最坏情况扣分
        worst_impact = abs(stress_results.get('worst_case_impact', 0))
        if worst_impact > 20:
            deduction = min(40, (worst_impact - 20) * 2)
            base_score -= deduction
        
        # 违反限制扣分
        breach_count = len(stress_results.get('breach_indicators', []))
        base_score -= breach_count * 5
        
        # 确保分数在0-100之间
        return max(0.0, min(100.0, base_score))
    
    def analyze_correlation_risk(self, asset_returns: Dict[str, List[float]]) -> Dict:
        """分析相关性风险"""
        if not asset_returns or len(asset_returns) < 2:
            return {'error': '需要至少两个资产的收益率数据'}
        
        assets = list(asset_returns.keys())
        
        # 计算相关系数矩阵
        correlation_matrix = self._calculate_correlation_matrix(asset_returns)
        
        # 识别高相关性集群
        correlation_clusters = self._identify_correlation_clusters(correlation_matrix, assets)
        
        # 计算集中度风险
        concentration_risk = self._calculate_concentration_risk(correlation_matrix, assets)
        
        # 计算动态相关性指标
        dynamic_correlation = self._analyze_dynamic_correlation(asset_returns)
        
        # 检查相关性阈值违规
        threshold_breaches = self._check_correlation_thresholds(correlation_matrix, assets)
        
        correlation_analysis = {
            'analysis_time': datetime.now(),
            'num_assets': len(assets),
            'correlation_matrix': correlation_matrix,
            'correlation_clusters': correlation_clusters,
            'concentration_risk': concentration_risk,
            'dynamic_correlation': dynamic_correlation,
            'threshold_breaches': threshold_breaches,
            'recommendations': []
        }
        
        # 生成建议
        recommendations = self._generate_correlation_recommendations(correlation_analysis)
        correlation_analysis['recommendations'] = recommendations
        
        # 计算相关性风险分数
        correlation_risk_score = self._calculate_correlation_risk_score(correlation_analysis)
        correlation_analysis['correlation_risk_score'] = correlation_risk_score
        
        # 更新当前风险状态
        self.current_risk_state['correlation_risk_score'] = correlation_risk_score
        
        # 保存到历史
        self.risk_metrics_history['correlation_analyses'].append(correlation_analysis)
        
        return correlation_analysis
    
    def _calculate_correlation_matrix(self, asset_returns: Dict[str, List[float]]) -> Dict:
        """计算相关系数矩阵"""
        assets = list(asset_returns.keys())
        n_assets = len(assets)
        
        # 创建矩阵
        matrix = {}
        
        for i, asset1 in enumerate(assets):
            matrix[asset1] = {}
            returns1 = asset_returns[asset1]
            
            for j, asset2 in enumerate(assets):
                returns2 = asset_returns[asset2]
                
                # 确保长度一致
                min_length = min(len(returns1), len(returns2))
                if min_length < 2:
                    correlation = 0.0
                else:
                    # 计算相关系数
                    arr1 = np.array(returns1[:min_length])
                    arr2 = np.array(returns2[:min_length])
                    
                    # 使用numpy计算相关系数
                    correlation_matrix = np.corrcoef(arr1, arr2)
                    correlation = float(correlation_matrix[0, 1])
                
                matrix[asset1][asset2] = correlation
        
        return matrix
    
    def _identify_correlation_clusters(self, correlation_matrix: Dict, assets: List[str]) -> List[Dict]:
        """识别高相关性集群"""
        clusters = []
        threshold = self.portfolio_config.get('correlation_threshold', 0.7)
        
        visited = set()
        
        for i, asset1 in enumerate(assets):
            if asset1 in visited:
                continue
            
            cluster = [asset1]
            visited.add(asset1)
            
            for j, asset2 in enumerate(assets):
                if asset2 in visited or asset1 == asset2:
                    continue
                
                correlation = correlation_matrix[asset1][asset2]
                if correlation >= threshold:
                    cluster.append(asset2)
                    visited.add(asset2)
            
            if len(cluster) > 1:
                # 计算集群平均相关性
                cluster_correlations = []
                for a1 in cluster:
                    for a2 in cluster:
                        if a1 != a2:
                            cluster_correlations.append(correlation_matrix[a1][a2])
                
                clusters.append({
                    'assets': cluster,
                    'size': len(cluster),
                    'avg_correlation': np.mean(cluster_correlations) if cluster_correlations else 0,
                    'max_correlation': max(cluster_correlations) if cluster_correlations else 0,
                    'risk_level': 'high' if len(cluster) >= 3 else 'medium'
                })
        
        return clusters
    
    def _calculate_concentration_risk(self, correlation_matrix: Dict, assets: List[str]) -> Dict:
        """计算集中度风险"""
        # 计算平均绝对相关性
        correlations = []
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:  # 避免重复和自相关
                    correlations.append(abs(correlation_matrix[asset1][asset2]))
        
        avg_abs_correlation = np.mean(correlations) if correlations else 0
        
        # 计算特征值集中度（简化版）
        # 实际应用中应该计算相关系数矩阵的特征值
        n_assets = len(assets)
        
        # 创建numpy数组
        matrix_array = np.zeros((n_assets, n_assets))
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                matrix_array[i, j] = correlation_matrix[asset1][asset2]
        
        # 计算特征值
        try:
            eigenvalues = np.linalg.eigvals(matrix_array)
            eigenvalue_concentration = np.std(eigenvalues) / np.mean(eigenvalues) if np.mean(eigenvalues) != 0 else 0
        except:
            eigenvalue_concentration = 0
        
        # 评估风险等级
        if avg_abs_correlation > 0.6:
            concentration_level = 'high'
        elif avg_abs_correlation > 0.4:
            concentration_level = 'medium'
        else:
            concentration_level = 'low'
        
        return {
            'avg_absolute_correlation': avg_abs_correlation,
            'eigenvalue_concentration': eigenvalue_concentration,
            'concentration_level': concentration_level,
            'num_assets': n_assets
        }
    
    def _analyze_dynamic_correlation(self, asset_returns: Dict[str, List[float]]) -> Dict:
        """分析动态相关性"""
        if not asset_returns:
            return {'error': '收益率数据为空'}
        
        assets = list(asset_returns.keys())
        if len(assets) < 2:
            return {'error': '需要至少两个资产'}
        
        # 使用滚动窗口分析相关性变化
        window_size = 20  # 20期滚动窗口
        asset1, asset2 = assets[0], assets[1]
        
        returns1 = asset_returns[asset1]
        returns2 = asset_returns[asset2]
        
        min_length = min(len(returns1), len(returns2))
        if min_length < window_size:
            return {'error': '数据长度不足'}
        
        rolling_correlations = []
        
        for i in range(min_length - window_size + 1):
            window1 = returns1[i:i+window_size]
            window2 = returns2[i:i+window_size]
            
            # 计算窗口内相关系数
            corr_matrix = np.corrcoef(window1, window2)
            correlation = float(corr_matrix[0, 1])
            rolling_correlations.append(correlation)
        
        # 分析相关性稳定性
        if rolling_correlations:
            correlation_std = np.std(rolling_correlations)
            correlation_range = max(rolling_correlations) - min(rolling_correlations)
            
            if correlation_std > 0.2:
                stability = 'unstable'
            elif correlation_std > 0.1:
                stability = 'moderate'
            else:
                stability = 'stable'
        else:
            correlation_std = 0
            correlation_range = 0
            stability = 'unknown'
        
        return {
            'asset_pair': f'{asset1}-{asset2}',
            'window_size': window_size,
            'rolling_correlations': rolling_correlations,
            'correlation_std': correlation_std,
            'correlation_range': correlation_range,
            'stability': stability,
            'last_correlation': rolling_correlations[-1] if rolling_correlations else 0
        }
    
    def _check_correlation_thresholds(self, correlation_matrix: Dict, assets: List[str]) -> List[Dict]:
        """检查相关性阈值违规"""
        breaches = []
        threshold = self.portfolio_config.get('correlation_threshold', 0.7)
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i >= j:  # 避免重复和自相关
                    continue
                
                correlation = correlation_matrix[asset1][asset2]
                if abs(correlation) >= threshold:
                    breaches.append({
                        'asset_pair': f'{asset1}-{asset2}',
                        'correlation': correlation,
                        'threshold': threshold,
                        'breach_type': 'high_correlation' if correlation > 0 else 'high_negative_correlation',
                        'severity': 'high' if abs(correlation) > 0.8 else 'medium'
                    })
        
        return breaches
    
    def _generate_correlation_recommendations(self, correlation_analysis: Dict) -> List[Dict]:
        """生成相关性风险建议"""
        recommendations = []
        
        # 基于高相关性集群的建议
        clusters = correlation_analysis.get('correlation_clusters', [])
        for cluster in clusters:
            if cluster['size'] >= 3:
                recommendations.append({
                    'type': 'diversification',
                    'priority': 'high',
                    'message': f"发现高相关性集群包含{cluster['size']}个资产，平均相关性{cluster['avg_correlation']:.2f}",
                    'action': '考虑减少集群内资产的头寸或增加非相关资产'
                })
        
        # 基于阈值违规的建议
        breaches = correlation_analysis.get('threshold_breaches', [])
        if breaches:
            high_correlation_pairs = [b for b in breaches if b['severity'] == 'high']
            if high_correlation_pairs:
                recommendations.append({
                    'type': 'correlation_reduction',
                    'priority': 'medium',
                    'message': f"发现{len(high_correlation_pairs)}对资产相关性超过0.8",
                    'action': '调整这些资产的头寸以降低相关性风险'
                })
        
        # 基于集中度风险的建议
        concentration = correlation_analysis.get('concentration_risk', {})
        if concentration.get('concentration_level') == 'high':
            recommendations.append({
                'type': 'concentration_risk',
                'priority': 'high',
                'message': f"投资组合集中度风险高，平均绝对相关性{concentration['avg_absolute_correlation']:.2f}",
                'action': '增加资产分散化，降低相关性'
            })
        
        return recommendations
    
    def _calculate_correlation_risk_score(self, correlation_analysis: Dict) -> float:
        """计算相关性风险分数（0-100，越高越好）"""
        base_score = 100.0
        
        # 高相关性集群扣分
        clusters = correlation_analysis.get('correlation_clusters', [])
        for cluster in clusters:
            if cluster['size'] >= 3:
                base_score -= 10 * (cluster['size'] - 2)
        
        # 阈值违规扣分
        breaches = correlation_analysis.get('threshold_breaches', [])
        base_score -= len(breaches) * 5
        
        # 集中度风险扣分
        concentration = correlation_analysis.get('concentration_risk', {})
        if concentration.get('concentration_level') == 'high':
            base_score -= 20
        elif concentration.get('concentration_level') == 'medium':
            base_score -= 10
        
        # 确保分数在0-100之间
        return max(0.0, min(100.0, base_score))
    
    def assess_liquidity_risk(self, portfolio_positions: List[Dict], market_data: Dict) -> Dict:
        """评估流动性风险"""
        if not portfolio_positions:
            return {'error': '投资组合仓位数据为空'}
        
        liquidity_assessment = {
            'assessment_time': datetime.now(),
            'portfolio_positions': len(portfolio_positions),
            'position_liquidity_scores': [],
            'overall_liquidity_score': 0.0,
            'liquidity_risk_factors': [],
            'exit_time_estimates': [],
            'recommendations': []
        }
        
        # 评估每个仓位的流动性
        position_scores = []
        exit_times = []
        
        for position in portfolio_positions:
            position_id = position.get('id', 'unknown')
            asset_type = position.get('asset_type', 'unknown')
            position_size = position.get('value', 0)
            instrument = position.get('instrument', 'unknown')
            
            # 获取市场数据
            market_info = market_data.get(instrument, {})
            
            # 计算流动性分数
            liquidity_score = self._calculate_position_liquidity_score(position, market_info)
            position_scores.append(liquidity_score)
            
            # 估计退出时间
            exit_time = self._estimate_position_exit_time(position, market_info)
            exit_times.append(exit_time)
            
            liquidity_assessment['position_liquidity_scores'].append({
                'position_id': position_id,
                'asset_type': asset_type,
                'position_size': position_size,
                'liquidity_score': liquidity_score,
                'exit_time_days': exit_time,
                'liquidity_rating': self._get_liquidity_rating(liquidity_score)
            })
        
        # 计算整体流动性分数
        if position_scores:
            overall_score = np.mean(position_scores)
            liquidity_assessment['overall_liquidity_score'] = overall_score
        
        # 估计整体退出时间
        if exit_times:
            max_exit_time = max(exit_times)
            liquidity_assessment['max_exit_time_days'] = max_exit_time
        
        # 识别流动性风险因素
        risk_factors = self._identify_liquidity_risk_factors(liquidity_assessment)
        liquidity_assessment['liquidity_risk_factors'] = risk_factors
        
        # 生成建议
        recommendations = self._generate_liquidity_recommendations(liquidity_assessment)
        liquidity_assessment['recommendations'] = recommendations
        
        # 计算流动性风险分数
        liquidity_risk_score = self._calculate_liquidity_risk_score(liquidity_assessment)
        liquidity_assessment['liquidity_risk_score'] = liquidity_risk_score
        
        # 更新当前风险状态
        self.current_risk_state['liquidity_risk_score'] = liquidity_risk_score
        
        # 保存到历史
        self.risk_metrics_history['liquidity_assessments'].append(liquidity_assessment)
        
        return liquidity_assessment
    
    def _calculate_position_liquidity_score(self, position: Dict, market_info: Dict) -> float:
        """计算单个仓位的流动性分数（0-100）"""
        base_score = 100.0
        
        asset_type = position.get('asset_type', 'unknown')
        position_size = position.get('value', 0)
        
        # 基于资产类型的调整
        asset_type_multipliers = {
            'cash': 1.0,
            'major_currency': 0.9,
            'large_cap_equity': 0.8,
            'government_bond': 0.7,
            'corporate_bond': 0.6,
            'small_cap_equity': 0.5,
            'commodity': 0.4,
            'derivative': 0.3,
            'real_estate': 0.2,
            'private_equity': 0.1
        }
        
        asset_multiplier = asset_type_multipliers.get(asset_type, 0.5)
        base_score *= asset_multiplier
        
        # 基于仓位的调整（越大越不流动）
        if position_size > 1000000:  # 超过100万
            size_penalty = min(30, (position_size - 1000000) / 1000000 * 10)
            base_score -= size_penalty
        
        # 基于市场数据的调整
        if market_info:
            # 买卖价差影响
            bid_ask_spread = market_info.get('bid_ask_spread', 0)
            if bid_ask_spread > 0.001:  # 价差超过0.1%
                spread_penalty = min(20, bid_ask_spread * 10000)
                base_score -= spread_penalty
            
            # 市场深度影响
            market_depth = market_info.get('market_depth', 0)
            if market_depth > 0:
                depth_score = min(100, market_depth / position_size * 100) if position_size > 0 else 100
                base_score = (base_score + depth_score) / 2
        
        return max(0.0, min(100.0, base_score))
    
    def _estimate_position_exit_time(self, position: Dict, market_info: Dict) -> float:
        """估计仓位退出时间（天数）"""
        asset_type = position.get('asset_type', 'unknown')
        position_size = position.get('value', 0)
        
        # 基于资产类型的基准退出时间
        base_exit_times = {
            'cash': 0.1,
            'major_currency': 0.2,
            'large_cap_equity': 0.5,
            'government_bond': 1.0,
            'corporate_bond': 2.0,
            'small_cap_equity': 3.0,
            'commodity': 2.5,
            'derivative': 1.5,
            'real_estate': 30.0,
            'private_equity': 90.0
        }
        
        exit_time = base_exit_times.get(asset_type, 5.0)
        
        # 基于仓位大小的调整
        if position_size > 1000000:
            size_multiplier = 1.0 + (position_size - 1000000) / 1000000 * 0.5
            exit_time *= size_multiplier
        
        # 基于市场流动性的调整
        if market_info:
            daily_volume = market_info.get('daily_volume', 0)
            if daily_volume > 0 and position_size > 0:
                volume_ratio = position_size / daily_volume
                volume_multiplier = 1.0 + volume_ratio * 2.0
                exit_time *= volume_multiplier
        
        return max(0.1, min(365.0, exit_time))  # 限制在0.1-365天之间
    
    def _get_liquidity_rating(self, liquidity_score: float) -> str:
        """获取流动性评级"""
        if liquidity_score >= 80:
            return 'excellent'
        elif liquidity_score >= 60:
            return 'good'
        elif liquidity_score >= 40:
            return 'fair'
        elif liquidity_score >= 20:
            return 'poor'
        else:
            return 'very_poor'
    
    def _identify_liquidity_risk_factors(self, liquidity_assessment: Dict) -> List[Dict]:
        """识别流动性风险因素"""
        risk_factors = []
        
        # 检查低流动性仓位
        for position_score in liquidity_assessment.get('position_liquidity_scores', []):
            liquidity_score = position_score.get('liquidity_score', 0)
            position_id = position_score.get('position_id', 'unknown')
            
            if liquidity_score < 40:
                risk_factors.append({
                    'factor': 'low_liquidity_position',
                    'position_id': position_id,
                    'liquidity_score': liquidity_score,
                    'rating': position_score.get('liquidity_rating', 'unknown'),
                    'risk_level': 'high' if liquidity_score < 20 else 'medium'
                })
        
        # 检查长退出时间
        max_exit_time = liquidity_assessment.get('max_exit_time_days', 0)
        if max_exit_time > 10:
            risk_factors.append({
                'factor': 'long_exit_time',
                'max_exit_time_days': max_exit_time,
                'risk_level': 'high' if max_exit_time > 30 else 'medium'
            })
        
        # 检查整体流动性
        overall_score = liquidity_assessment.get('overall_liquidity_score', 0)
        if overall_score < 50:
            risk_factors.append({
                'factor': 'poor_overall_liquidity',
                'overall_score': overall_score,
                'risk_level': 'high' if overall_score < 30 else 'medium'
            })
        
        return risk_factors
    
    def _generate_liquidity_recommendations(self, liquidity_assessment: Dict) -> List[Dict]:
        """生成流动性风险建议"""
        recommendations = []
        
        # 低流动性仓位建议
        low_liquidity_positions = []
        for position in liquidity_assessment.get('position_liquidity_scores', []):
            if position.get('liquidity_score', 0) < 40:
                low_liquidity_positions.append(position)
        
        if low_liquidity_positions:
            recommendations.append({
                'type': 'low_liquidity_positions',
                'priority': 'high' if any(p.get('liquidity_score', 0) < 20 for p in low_liquidity_positions) else 'medium',
                'message': f"发现{len(low_liquidity_positions)}个低流动性仓位",
                'action': '考虑减少这些仓位的大小或增加对冲'
            })
        
        # 长退出时间建议
        max_exit_time = liquidity_assessment.get('max_exit_time_days', 0)
        if max_exit_time > 30:
            recommendations.append({
                'type': 'excessive_exit_time',
                'priority': 'high',
                'message': f"最长退出时间{max_exit_time:.1f}天",
                'action': '建立应急流动性计划，准备快速退出策略'
            })
        
        # 整体流动性建议
        overall_score = liquidity_assessment.get('overall_liquidity_score', 0)
        if overall_score < 50:
            recommendations.append({
                'type': 'overall_liquidity_improvement',
                'priority': 'medium',
                'message': f"整体流动性分数{overall_score:.1f}",
                'action': '增加高流动性资产比例，减少低流动性资产暴露'
            })
        
        return recommendations
    
    def _calculate_liquidity_risk_score(self, liquidity_assessment: Dict) -> float:
        """计算流动性风险分数（0-100，越高越好）"""
        base_score = liquidity_assessment.get('overall_liquidity_score', 50.0)
        
        # 低流动性仓位扣分
        low_liquidity_count = 0
        critical_liquidity_count = 0
        
        for position in liquidity_assessment.get('position_liquidity_scores', []):
            score = position.get('liquidity_score', 0)
            if score < 40:
                low_liquidity_count += 1
            if score < 20:
                critical_liquidity_count += 1
        
        base_score -= low_liquidity_count * 5
        base_score -= critical_liquidity_count * 10
        
        # 长退出时间扣分
        max_exit_time = liquidity_assessment.get('max_exit_time_days', 0)
        if max_exit_time > 30:
            base_score -= 20
        elif max_exit_time > 10:
            base_score -= 10
        
        # 确保分数在0-100之间
        return max(0.0, min(100.0, base_score))
    
    def monitor_leverage_risk(self, portfolio_data: Dict, margin_data: Dict) -> Dict:
        """监控杠杆风险"""
        if not portfolio_data:
            return {'error': '投资组合数据为空'}
        
        total_value = portfolio_data.get('total_value', 0)
        total_margin = portfolio_data.get('total_margin', 0)
        positions = portfolio_data.get('positions', [])
        
        if total_value <= 0:
            return {'error': '投资组合价值必须大于0'}
        
        # 计算杠杆比率
        leverage_ratio = total_margin / total_value if total_value > 0 else 0
        
        # 计算保证金覆盖率
        margin_coverage = self._calculate_margin_coverage(positions, margin_data)
        
        # 分析强制平仓风险
        liquidation_risk = self._analyze_liquidation_risk(positions, margin_data)
        
        # 计算动态杠杆限制
        dynamic_leverage_limit = self._calculate_dynamic_leverage_limit(portfolio_data)
        
        # 检查杠杆限制违规
        leverage_breaches = self._check_leverage_breaches(leverage_ratio, dynamic_leverage_limit, margin_coverage)
        
        leverage_monitoring = {
            'monitoring_time': datetime.now(),
            'total_portfolio_value': total_value,
            'total_margin_used': total_margin,
            'leverage_ratio': leverage_ratio,
            'margin_coverage': margin_coverage,
            'liquidation_risk': liquidation_risk,
            'dynamic_leverage_limit': dynamic_leverage_limit,
            'leverage_breaches': leverage_breaches,
            'recommendations': []
        }
        
        # 生成建议
        recommendations = self._generate_leverage_recommendations(leverage_monitoring)
        leverage_monitoring['recommendations'] = recommendations
        
        # 计算杠杆风险分数
        leverage_risk_score = self._calculate_leverage_risk_score(leverage_monitoring)
        leverage_monitoring['leverage_risk_score'] = leverage_risk_score
        
        # 更新当前风险状态
        self.current_risk_state['leverage_ratio'] = leverage_ratio
        
        # 保存到历史
        self.risk_metrics_history['leverage_monitoring'].append(leverage_monitoring)
        
        return leverage_monitoring
    
    def _calculate_margin_coverage(self, positions: List[Dict], margin_data: Dict) -> Dict:
        """计算保证金覆盖率"""
        total_margin_required = 0
        total_margin_available = 0
        
        for position in positions:
            position_id = position.get('id', 'unknown')
            position_value = position.get('value', 0)
            instrument = position.get('instrument', 'unknown')
            
            # 获取保证金要求
            margin_requirement = margin_data.get(instrument, {}).get('margin_requirement', 0.5)  # 默认50%
            
            # 计算所需保证金
            margin_required = position_value * margin_requirement
            total_margin_required += margin_required
            
            # 假设可用保证金为仓位价值
            total_margin_available += position_value
        
        coverage_ratio = total_margin_available / total_margin_required if total_margin_required > 0 else float('inf')
        
        return {
            'total_margin_required': total_margin_required,
            'total_margin_available': total_margin_available,
            'coverage_ratio': coverage_ratio,
            'coverage_status': 'adequate' if coverage_ratio >= 1.5 else 'inadequate'
        }
    
    def _analyze_liquidation_risk(self, positions: List[Dict], margin_data: Dict) -> Dict:
        """分析强制平仓风险"""
        liquidation_positions = []
        
        for position in positions:
            position_id = position.get('id', 'unknown')
            instrument = position.get('instrument', 'unknown')
            position_value = position.get('value', 0)
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', entry_price)
            
            # 获取保证金和强平价信息
            instrument_margin_data = margin_data.get(instrument, {})
            margin_requirement = instrument_margin_data.get('margin_requirement', 0.5)
            liquidation_price = instrument_margin_data.get('liquidation_price', 0)
            
            # 计算强平风险
            if liquidation_price > 0 and current_price > 0:
                price_distance = abs(current_price - liquidation_price) / current_price
                
                if price_distance < 0.10:  # 距离强平价10%以内
                    liquidation_risk = 'high'
                elif price_distance < 0.20:
                    liquidation_risk = 'medium'
                else:
                    liquidation_risk = 'low'
            else:
                price_distance = 0
                liquidation_risk = 'unknown'
            
            if liquidation_risk in ['high', 'medium']:
                liquidation_positions.append({
                    'position_id': position_id,
                    'instrument': instrument,
                    'current_price': current_price,
                    'liquidation_price': liquidation_price,
                    'price_distance_percent': price_distance * 100,
                    'liquidation_risk': liquidation_risk
                })
        
        # 计算整体强平风险
        if liquidation_positions:
            high_risk_count = sum(1 for p in liquidation_positions if p['liquidation_risk'] == 'high')
            medium_risk_count = sum(1 for p in liquidation_positions if p['liquidation_risk'] == 'medium')
            
            if high_risk_count > 0:
                overall_risk = 'high'
            elif medium_risk_count > 0:
                overall_risk = 'medium'
            else:
                overall_risk = 'low'
        else:
            overall_risk = 'low'
        
        return {
            'liquidation_positions': liquidation_positions,
            'high_risk_count': high_risk_count if 'high_risk_count' in locals() else 0,
            'medium_risk_count': medium_risk_count if 'medium_risk_count' in locals() else 0,
            'overall_liquidation_risk': overall_risk
        }
    
    def _calculate_dynamic_leverage_limit(self, portfolio_data: Dict) -> float:
        """计算动态杠杆限制"""
        base_limit = self.risk_tolerance.get('leverage_limit', 3.0)
        
        # 基于波动性调整
        portfolio_volatility = portfolio_data.get('portfolio_volatility', 0.15)  # 默认15%
        volatility_adjustment = max(0.5, 1.0 - portfolio_volatility / 0.3)  # 波动性30%时减半
        
        # 基于相关性调整
        portfolio_correlation = portfolio_data.get('avg_correlation', 0.3)  # 默认相关性0.3
        correlation_adjustment = max(0.6, 1.0 - portfolio_correlation / 0.8)  # 相关性0.8时减少40%
        
        # 基于流动性调整
        portfolio_liquidity = portfolio_data.get('liquidity_score', 70.0)  # 默认70分
        liquidity_adjustment = portfolio_liquidity / 100.0
        
        # 计算动态限制
        dynamic_limit = base_limit * volatility_adjustment * correlation_adjustment * liquidity_adjustment
        
        # 确保不低于1.0
        return max(1.0, min(base_limit, dynamic_limit))
    
    def _check_leverage_breaches(self, leverage_ratio: float, dynamic_limit: float, margin_coverage: Dict) -> List[Dict]:
        """检查杠杆限制违规"""
        breaches = []
        
        # 检查静态杠杆限制
        static_limit = self.risk_tolerance.get('leverage_limit', 3.0)
        if leverage_ratio > static_limit:
            breaches.append({
                'breach_type': 'static_leverage_limit',
                'current_leverage': leverage_ratio,
                'limit': static_limit,
                'excess_percent': (leverage_ratio - static_limit) / static_limit * 100,
                'severity': 'high' if leverage_ratio > static_limit * 1.2 else 'medium'
            })
        
        # 检查动态杠杆限制
        if leverage_ratio > dynamic_limit:
            breaches.append({
                'breach_type': 'dynamic_leverage_limit',
                'current_leverage': leverage_ratio,
                'limit': dynamic_limit,
                'excess_percent': (leverage_ratio - dynamic_limit) / dynamic_limit * 100,
                'severity': 'high' if leverage_ratio > dynamic_limit * 1.1 else 'medium'
            })
        
        # 检查保证金覆盖率
        coverage_ratio = margin_coverage.get('coverage_ratio', 0)
        if coverage_ratio < 1.2:
            breaches.append({
                'breach_type': 'margin_coverage',
                'current_coverage': coverage_ratio,
                'minimum_required': 1.2,
                'deficit_percent': (1.2 - coverage_ratio) / 1.2 * 100 if coverage_ratio < 1.2 else 0,
                'severity': 'high' if coverage_ratio < 1.0 else 'medium'
            })
        
        return breaches
    
    def _generate_leverage_recommendations(self, leverage_monitoring: Dict) -> List[Dict]:
        """生成杠杆风险建议"""
        recommendations = []
        
        # 杠杆超限建议
        breaches = leverage_monitoring.get('leverage_breaches', [])
        leverage_breaches = [b for b in breaches if 'leverage' in b['breach_type']]
        
        if leverage_breaches:
            worst_breach = max(leverage_breaches, key=lambda x: x['excess_percent'])
            recommendations.append({
                'type': 'leverage_reduction',
                'priority': 'high' if worst_breach['severity'] == 'high' else 'medium',
                'message': f"杠杆比率{leverage_monitoring['leverage_ratio']:.2f}超过限制",
                'action': '减少仓位或增加资本以降低杠杆'
            })
        
        # 保证金不足建议
        margin_breaches = [b for b in breaches if 'margin' in b['breach_type']]
        if margin_breaches:
            recommendations.append({
                'type': 'margin_increase',
                'priority': 'high',
                'message': f"保证金覆盖率{leverage_monitoring['margin_coverage']['coverage_ratio']:.2f}不足",
                'action': '增加保证金或减少风险暴露'
            })
        
        # 强平风险建议
        liquidation_risk = leverage_monitoring.get('liquidation_risk', {})
        if liquidation_risk.get('overall_liquidation_risk') in ['high', 'medium']:
            recommendations.append({
                'type': 'liquidation_risk_management',
                'priority': 'high' if liquidation_risk['overall_liquidation_risk'] == 'high' else 'medium',
                'message': f"发现{liquidation_risk.get('high_risk_count', 0)}个高强平风险仓位",
                'action': '调整高风险仓位，设置更严格的止损'
            })
        
        return recommendations
    
    def _calculate_leverage_risk_score(self, leverage_monitoring: Dict) -> float:
        """计算杠杆风险分数（0-100，越高越好）"""
        base_score = 100.0
        
        # 杠杆比率扣分
        leverage_ratio = leverage_monitoring.get('leverage_ratio', 0)
        limit = self.risk_tolerance.get('leverage_limit', 3.0)
        
        if leverage_ratio > limit:
            excess_ratio = (leverage_ratio - limit) / limit
            deduction = min(50, excess_ratio * 100)
            base_score -= deduction
        
        # 保证金覆盖率扣分
        coverage_ratio = leverage_monitoring.get('margin_coverage', {}).get('coverage_ratio', 0)
        if coverage_ratio < 1.2:
            deficit = 1.2 - coverage_ratio
            deduction = min(30, deficit * 50)
            base_score -= deduction
        
        # 强平风险扣分
        liquidation_risk = leverage_monitoring.get('liquidation_risk', {})
        high_risk_count = liquidation_risk.get('high_risk_count', 0)
        medium_risk_count = liquidation_risk.get('medium_risk_count', 0)
        
        base_score -= high_risk_count * 15
        base_score -= medium_risk_count * 5
        
        # 违规扣分
        breach_count = len(leverage_monitoring.get('leverage_breaches', []))
        base_score -= breach_count * 10
        
        # 确保分数在0-100之间
        return max(0.0, min(100.0, base_score))
    
    def detect_extreme_events(self, market_indicators: Dict, portfolio_state: Dict) -> Dict:
        """检测极端事件"""
        extreme_alerts = {
            'detection_time': datetime.now(),
            'market_indicators': market_indicators,
            'portfolio_state': portfolio_state,
            'detected_events': [],
            'risk_assessments': [],
            'action_recommendations': []
        }
        
        # 检查市场波动性异常
        volatility_events = self._check_volatility_extremes(market_indicators)
        extreme_alerts['detected_events'].extend(volatility_events)
        
        # 检查流动性异常
        liquidity_events = self._check_liquidity_extremes(market_indicators)
        extreme_alerts['detected_events'].extend(liquidity_events)
        
        # 检查相关性异常
        correlation_events = self._check_correlation_extremes(market_indicators)
        extreme_alerts['detected_events'].extend(correlation_events)
        
        # 检查投资组合压力
        portfolio_events = self._check_portfolio_stress(portfolio_state)
        extreme_alerts['detected_events'].extend(portfolio_events)
        
        # 风险评估
        if extreme_alerts['detected_events']:
            risk_assessments = self._assess_extreme_event_risks(extreme_alerts['detected_events'])
            extreme_alerts['risk_assessments'] = risk_assessments
        
        # 生成行动建议
        if extreme_alerts['detected_events']:
            recommendations = self._generate_extreme_event_recommendations(extreme_alerts)
            extreme_alerts['action_recommendations'] = recommendations
        
        # 更新当前风险状态
        if extreme_alerts['detected_events']:
            self.current_risk_state['overall_risk_level'] = 'high'
        
        # 保存到历史
        self.risk_metrics_history['extreme_event_alerts'].append(extreme_alerts)
        
        return extreme_alerts
    
    def _check_volatility_extremes(self, market_indicators: Dict) -> List[Dict]:
        """检查波动性异常"""
        events = []
        
        volatility = market_indicators.get('volatility', {})
        vix = volatility.get('vix', 0)
        historical_vol = volatility.get('historical_volatility', 0)
        
        # 检查VIX异常
        if vix > 40:
            events.append({
                'event_type': 'extreme_volatility',
                'indicator': 'vix',
                'value': vix,
                'threshold': 40,
                'severity': 'critical' if vix > 60 else 'high',
                'description': f'VIX指数达到{vix}，显示极端市场恐慌'
            })
        
        # 检查历史波动率异常
        if historical_vol > 0.4:  # 40%年化波动率
            events.append({
                'event_type': 'high_historical_volatility',
                'indicator': 'historical_volatility',
                'value': historical_vol,
                'threshold': 0.4,
                'severity': 'high' if historical_vol > 0.6 else 'medium',
                'description': f'历史波动率达到{historical_vol:.1%}，市场不稳定'
            })
        
        # 检查波动率跳跃
        volatility_change = volatility.get('volatility_change', 0)
        if abs(volatility_change) > 0.2:  # 波动率变化超过20%
            events.append({
                'event_type': 'volatility_jump',
                'indicator': 'volatility_change',
                'value': volatility_change,
                'threshold': 0.2,
                'severity': 'high',
                'description': f'波动率变化{volatility_change:.1%}，市场条件快速变化'
            })
        
        return events
    
    def _check_liquidity_extremes(self, market_indicators: Dict) -> List[Dict]:
        """检查流动性异常"""
        events = []
        
        liquidity = market_indicators.get('liquidity', {})
        bid_ask_spread = liquidity.get('avg_bid_ask_spread', 0)
        market_depth = liquidity.get('market_depth', 0)
        volume_ratio = liquidity.get('volume_ratio', 1.0)
        
        # 检查买卖价差扩大
        if bid_ask_spread > 0.005:  # 价差超过0.5%
            events.append({
                'event_type': 'wide_bid_ask_spread',
                'indicator': 'bid_ask_spread',
                'value': bid_ask_spread,
                'threshold': 0.005,
                'severity': 'high' if bid_ask_spread > 0.01 else 'medium',
                'description': f'平均买卖价差达到{bid_ask_spread:.3%}，流动性下降'
            })
        
        # 检查市场深度下降
        if market_depth < 0.5:  # 市场深度低于正常50%
            events.append({
                'event_type': 'low_market_depth',
                'indicator': 'market_depth',
                'value': market_depth,
                'threshold': 0.5,
                'severity': 'high' if market_depth < 0.3 else 'medium',
                'description': f'市场深度仅为正常的{market_depth:.0%}，大额交易困难'
            })
        
        # 检查成交量异常
        if volume_ratio < 0.5 or volume_ratio > 2.0:
            events.append({
                'event_type': 'abnormal_volume',
                'indicator': 'volume_ratio',
                'value': volume_ratio,
                'normal_range': '0.5-2.0',
                'severity': 'medium',
                'description': f'成交量比率为{volume_ratio:.2f}，偏离正常范围'
            })
        
        return events
    
    def _check_correlation_extremes(self, market_indicators: Dict) -> List[Dict]:
        """检查相关性异常"""
        events = []
        
        correlation = market_indicators.get('correlation', {})
        avg_correlation = correlation.get('avg_correlation', 0)
        correlation_change = correlation.get('correlation_change', 0)
        
        # 检查相关性急剧上升（所有资产同向运动）
        if avg_correlation > 0.8:
            events.append({
                'event_type': 'high_correlation_regime',
                'indicator': 'avg_correlation',
                'value': avg_correlation,
                'threshold': 0.8,
                'severity': 'high',
                'description': f'平均相关性达到{avg_correlation:.2f}，分散化失效'
            })
        
        # 检查相关性快速变化
        if abs(correlation_change) > 0.3:
            events.append({
                'event_type': 'correlation_breakdown',
                'indicator': 'correlation_change',
                'value': correlation_change,
                'threshold': 0.3,
                'severity': 'high',
                'description': f'相关性变化{correlation_change:.2f}，市场结构改变'
            })
        
        return events
    
    def _check_portfolio_stress(self, portfolio_state: Dict) -> List[Dict]:
        """检查投资组合压力"""
        events = []
        
        drawdown = portfolio_state.get('current_drawdown', 0)
        var_breach = portfolio_state.get('var_breach', False)
        margin_call = portfolio_state.get('margin_call_risk', False)
        
        # 检查大幅回撤
        if drawdown < -0.15:  # 回撤超过15%
            events.append({
                'event_type': 'large_drawdown',
                'indicator': 'current_drawdown',
                'value': drawdown,
                'threshold': -0.15,
                'severity': 'critical' if drawdown < -0.25 else 'high',
                'description': f'当前回撤{drawdown:.1%}，超过风险限制'
            })
        
        # 检查VaR违规
        if var_breach:
            events.append({
                'event_type': 'var_breach',
                'indicator': 'var_breach',
                'value': True,
                'severity': 'high',
                'description': '投资组合VaR超过限制，风险暴露过高'
            })
        
        # 检查保证金追缴风险
        if margin_call:
            events.append({
                'event_type': 'margin_call_risk',
                'indicator': 'margin_call_risk',
                'value': True,
                'severity': 'critical',
                'description': '面临保证金追缴风险，需要立即行动'
            })
        
        return events
    
    def _assess_extreme_event_risks(self, detected_events: List[Dict]) -> List[Dict]:
        """评估极端事件风险"""
        risk_assessments = []
        
        critical_events = [e for e in detected_events if e.get('severity') == 'critical']
        high_events = [e for e in detected_events if e.get('severity') == 'high']
        medium_events = [e for e in detected_events if e.get('severity') == 'medium']
        
        # 总体风险评估
        if critical_events:
            overall_risk = 'critical'
        elif high_events:
            overall_risk = 'high'
        elif medium_events:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        risk_assessments.append({
            'assessment_type': 'overall_risk',
            'risk_level': overall_risk,
            'critical_events': len(critical_events),
            'high_events': len(high_events),
            'medium_events': len(medium_events),
            'recommendation': '立即采取行动' if overall_risk in ['critical', 'high'] else '监控并准备行动'
        })
        
        # 按事件类型评估
        event_types = {}
        for event in detected_events:
            event_type = event.get('event_type', 'unknown')
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        for event_type, events in event_types.items():
            max_severity = max(e.get('severity', 'low') for e in events)
            
            risk_assessments.append({
                'assessment_type': 'event_type_risk',
                'event_type': event_type,
                'count': len(events),
                'max_severity': max_severity,
                'description': f'发现{len(events)}个{event_type}事件，最高严重程度为{max_severity}'
            })
        
        return risk_assessments
    
    def _generate_extreme_event_recommendations(self, extreme_alerts: Dict) -> List[Dict]:
        """生成极端事件行动建议"""
        recommendations = []
        detected_events = extreme_alerts.get('detected_events', [])
        
        # 检查是否有临界事件
        critical_events = [e for e in detected_events if e.get('severity') == 'critical']
        if critical_events:
            recommendations.append({
                'action_type': 'immediate_reduction',
                'priority': 'highest',
                'action': '立即减少风险暴露，降低杠杆，增加现金比例',
                'timeframe': '立即执行',
                'rationale': f'发现{len(critical_events)}个临界严重程度事件'
            })
        
        # 检查高严重程度事件
        high_events = [e for e in detected_events if e.get('severity') == 'high']
        if high_events:
            recommendations.append({
                'action_type': 'risk_reduction',
                'priority': 'high',
                'action': '显著减少风险暴露，收紧止损，增加对冲',
                'timeframe': '今日内执行',
                'rationale': f'发现{len(high_events)}个高严重程度事件'
            })
        
        # 波动性相关建议
        volatility_events = [e for e in detected_events if 'volatility' in e.get('event_type', '')]
        if volatility_events:
            recommendations.append({
                'action_type': 'volatility_protection',
                'priority': 'medium',
                'action': '增加波动性保护，如期权策略或动态对冲',
                'timeframe': '本周内执行',
                'rationale': '检测到波动性异常事件'
            })
        
        # 流动性相关建议
        liquidity_events = [e for e in detected_events if 'liquidity' in e.get('event_type', '')]
        if liquidity_events:
            recommendations.append({
                'action_type': 'liquidity_management',
                'priority': 'medium',
                'action': '减少低流动性仓位，增加现金缓冲，准备应急退出计划',
                'timeframe': '本周内执行',
                'rationale': '检测到流动性异常事件'
            })
        
        return recommendations
    
    def get_comprehensive_risk_report(self) -> Dict:
        """获取综合风险报告"""
        # 计算总体风险等级
        risk_scores = [
            self.current_risk_state.get('stress_test_score', 50),
            self.current_risk_state.get('correlation_risk_score', 50),
            self.current_risk_state.get('liquidity_risk_score', 50),
            (100 - abs(self.current_risk_state.get('portfolio_var', 0)) * 100) if self.current_risk_state.get('portfolio_var', 0) != 0 else 50
        ]
        
        avg_risk_score = np.mean(risk_scores) if risk_scores else 50
        
        # 确定总体风险等级
        if avg_risk_score >= 80:
            overall_risk = 'low'
        elif avg_risk_score >= 60:
            overall_risk = 'moderate'
        elif avg_risk_score >= 40:
            overall_risk = 'high'
        else:
            overall_risk = 'critical'
        
        # 更新当前风险状态
        self.current_risk_state['overall_risk_level'] = overall_risk
        
        comprehensive_report = {
            'report_time': datetime.now(),
            'overall_risk_assessment': {
                'risk_level': overall_risk,
                'risk_score': avg_risk_score,
                'component_scores': {
                    'stress_test': self.current_risk_state.get('stress_test_score', 0),
                    'correlation_risk': self.current_risk_state.get('correlation_risk_score', 0),
                    'liquidity_risk': self.current_risk_state.get('liquidity_risk_score', 0),
                    'var_risk': (100 - abs(self.current_risk_state.get('portfolio_var', 0)) * 100) if self.current_risk_state.get('portfolio_var', 0) != 0 else 50
                }
            },
            'current_risk_state': self.current_risk_state,
            'risk_metrics_history_summary': {
                'var_calculations': len(self.risk_metrics_history['var_metrics']),
                'stress_tests': len(self.risk_metrics_history['stress_test_results']),
                'correlation_analyses': len(self.risk_metrics_history['correlation_analyses']),
                'liquidity_assessments': len(self.risk_metrics_history['liquidity_assessments']),
                'leverage_monitoring': len(self.risk_metrics_history['leverage_monitoring']),
                'extreme_event_alerts': len(self.risk_metrics_history['extreme_event_alerts'])
            },
            'key_risk_indicators': self._get_key_risk_indicators(),
            'recommended_actions': self._generate_comprehensive_risk_actions(overall_risk)
        }
        
        return comprehensive_report
    
    def _get_key_risk_indicators(self) -> Dict:
        """获取关键风险指标"""
        return {
            'portfolio_var': self.current_risk_state.get('portfolio_var', 0),
            'portfolio_cvar': self.current_risk_state.get('portfolio_cvar', 0),
            'leverage_ratio': self.current_risk_state.get('leverage_ratio', 1.0),
            'stress_test_score': self.current_risk_state.get('stress_test_score', 0),
            'correlation_risk_score': self.current_risk_state.get('correlation_risk_score', 0),
            'liquidity_risk_score': self.current_risk_state.get('liquidity_risk_score', 0),
            'last_calculation_time': self.current_risk_state.get('last_calculation_time')
        }
    
    def _generate_comprehensive_risk_actions(self, overall_risk: str) -> List[Dict]:
        """生成综合风险行动建议"""
        actions = []
        
        if overall_risk == 'critical':
            actions.append({
                'action': '立即大幅减少风险暴露',
                'priority': 'highest',
                'timeframe': '立即',
                'details': '减少所有高风险仓位，增加现金比例至50%以上'
            })
            actions.append({
                'action': '暂停所有新交易',
                'priority': 'highest',
                'timeframe': '立即',
                'details': '直到风险水平降低到可接受范围'
            })
        
        elif overall_risk == 'high':
            actions.append({
                'action': '显著降低风险',
                'priority': 'high',
                'timeframe': '今日内',
                'details': '减少30-50%的风险暴露，收紧止损条件'
            })
            actions.append({
                'action': '增加对冲保护',
                'priority': 'high',
                'timeframe': '本周内',
                'details': '增加期权保护或相关对冲策略'
            })
        
        elif overall_risk == 'moderate':
            actions.append({
                'action': '适度调整风险',
                'priority': 'medium',
                'timeframe': '本周内',
                'details': '减少10-20%的风险暴露，优化仓位结构'
            })
            actions.append({
                'action': '加强监控',
                'priority': 'medium',
                'timeframe': '持续',
                'details': '增加风险指标监控频率'
            })
        
        else:  # low risk
            actions.append({
                'action': '维持当前策略',
                'priority': 'low',
                'timeframe': '持续',
                'details': '风险水平可接受，继续当前交易策略'
            })
            actions.append({
                'action': '定期审查',
                'priority': 'low',
                'timeframe': '每周',
                'details': '继续定期风险审查和压力测试'
            })
        
        return actions


# ============================================================================
# 策略改造: 添加AdvancedRiskManagementSystemStrategy类
# 将高级风险管理系统转换为交易策略
# ============================================================================

class AdvancedRiskManagementSystemStrategy(BaseStrategy):
    """高级风险管理策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict = None):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        max_drawdown_limit = self.params.get('max_drawdown_limit', 0.10)
        var_conf_level = self.params.get('var_conf_level', 0.95)
        
        # 创建高级风险管理系统实例
        portfolio_config = self.params.get('portfolio_config', None)
        risk_tolerance = self.params.get('risk_tolerance', None)
        self.risk_system = AdvancedRiskManagementSystem(
            portfolio_config=portfolio_config,
            risk_tolerance=risk_tolerance,
        )
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于高级风险管理生成交易信号
        """
        # 获取综合风险报告
        risk_report = self.risk_system.get_comprehensive_risk_report()
        
        # 分析风险等级和建议行动
        risk_assessment = risk_report.get('overall_risk_assessment', {})
        risk_level = risk_assessment.get('risk_level', 'unknown').lower()
        risk_score = risk_assessment.get('risk_score', 0)
        
        recommended_actions = risk_report.get('recommended_actions', [])
        
        # 根据风险等级生成信号
        if risk_level in ['low', 'very_low'] and risk_score >= 70:
            # 风险低且分数高，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif risk_level in ['high', 'very_high', 'extreme'] or risk_score <= 30:
            # 风险高或分数低，卖出或hold信号
            # 检查是否有强制减仓建议
            has_force_reduction = any(
                '减仓' in action.get('action', '') or 
                'reduce' in action.get('action', '').lower()
                for action in recommended_actions
            )
            
            if has_force_reduction:
                # 有强制减仓建议，卖出信号
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='sell',
                    price=self.data['close'].iloc[-1]
                )
            else:
                # 风险高但无强制减仓，hold信号
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='hold',
                    price=self.data['close'].iloc[-1]
                )
        else:
            # 中等风险，hold信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        
        return self.signals


# ============================================================================
# 策略改造完成
# ============================================================================

def demo_advanced_risk_management():
    """演示高级风险管理系统"""
    print("=" * 60)
    print("高级风险管理系统演示")
    print("第27章：风险管理高级主题 - AL Brooks《价格行为交易之区间篇》")
    print("=" * 60)
    
    # 创建系统实例
    risk_system = AdvancedRiskManagementSystem()
    
    print("\n1. 计算风险价值（VaR）...")
    
    # 生成示例投资组合收益率
    np.random.seed(42)
    portfolio_returns = np.random.normal(0.0005, 0.02, 1000)  # 1000个交易日
    
    var_result = risk_system.calculate_var(portfolio_returns.tolist(), 0.95)
    print(f"   95%置信水平VaR: {var_result['historical_var']:.3%}")
    print(f"   条件风险价值（CVaR）: {var_result['conditional_var']:.3%}")
    print(f"   平均收益率: {var_result['mean_return']:.3%}")
    print(f"   波动率: {var_result['std_return']:.3%}")
    
    print("\n2. 运行压力测试...")
    
    # 创建示例投资组合
    sample_positions = [
        {'id': 'pos1', 'value': 50000, 'asset_type': 'equity', 'sector': 'technology'},
        {'id': 'pos2', 'value': 30000, 'asset_type': 'bond', 'sector': 'government'},
        {'id': 'pos3', 'value': 20000, 'asset_type': 'commodity', 'sector': 'energy'}
    ]
    
    stress_test_result = risk_system.run_stress_test(sample_positions, ['2008_crisis', '2020_covid'])
    print(f"   测试场景: {len(stress_test_result['scenario_results'])}个")
    print(f"   总体影响: {stress_test_result['overall_impact']:.1f}%")
    print(f"   最坏情况: {stress_test_result['worst_case_scenario']}")
    print(f"   压力测试分数: {stress_test_result['stress_test_score']:.1f}/100")
    
    print("\n3. 分析相关性风险...")
    
    # 生成示例资产收益率
    asset_returns = {
        'AAPL': np.random.normal(0.0006, 0.025, 252).tolist(),
        'GOOGL': np.random.normal(0.0005, 0.023, 252).tolist(),
        'MSFT': np.random.normal(0.0004, 0.022, 252).tolist(),
        'TLT': np.random.normal(0.0002, 0.015, 252).tolist(),
        'GLD': np.random.normal(0.0003, 0.018, 252).tolist()
    }
    
    correlation_result = risk_system.analyze_correlation_risk(asset_returns)
    print(f"   分析资产: {correlation_result['num_assets']}个")
    print(f"   相关性集群: {len(correlation_result['correlation_clusters'])}个")
    print(f"   相关性风险分数: {correlation_result['correlation_risk_score']:.1f}/100")
    
    print("\n4. 评估流动性风险...")
    
    # 创建市场数据
    market_data = {
        'AAPL': {'bid_ask_spread': 0.0002, 'market_depth': 0.8, 'daily_volume': 10000000},
        'TLT': {'bid_ask_spread': 0.0005, 'market_depth': 0.6, 'daily_volume': 5000000}
    }
    
    liquidity_result = risk_system.assess_liquidity_risk(sample_positions, market_data)
    print(f"   评估仓位: {liquidity_result['portfolio_positions']}个")
    print(f"   整体流动性分数: {liquidity_result['overall_liquidity_score']:.1f}/100")
    print(f"   最长退出时间: {liquidity_result.get('max_exit_time_days', 0):.1f}天")
    
    print("\n5. 获取综合风险报告...")
    comprehensive_report = risk_system.get_comprehensive_risk_report()
    print(f"   总体风险等级: {comprehensive_report['overall_risk_assessment']['risk_level']}")
    print(f"   综合风险分数: {comprehensive_report['overall_risk_assessment']['risk_score']:.1f}/100")
    print(f"   推荐行动: {len(comprehensive_report['recommended_actions'])}个")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("高级风险管理系统已成功创建并测试")
    print("=" * 60)


if __name__ == "__main__":
    demo_advanced_risk_management()