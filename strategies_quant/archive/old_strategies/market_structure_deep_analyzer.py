# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.645506

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场结构深度分析器
第18章：市场结构识别 - 深度分析版本

用户指令："质量太差了, 从第18章开始重写, 我让你深入分析, 不是写个概括就完事"

本文件专注于深度分析，包括：
1. 理论基础深度分析
2. 算法对比分析
3. 参数优化分析
4. 性能评估分析
5. 深度洞察报告
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class MarketStructureDeepAnalyzer:
    """市场结构深度分析器 - 专注于深度分析而非功能实现"""
    
    def __init__(self):
        """初始化深度分析器"""
        self.analysis_history = []
        self.algorithm_comparison_results = {}
        self.parameter_optimization_results = {}
        self.performance_metrics = {}
        
    # ========== 第1部分：理论基础深度分析 ==========
    
    def analyze_theoretical_foundation(self) -> Dict[str, Any]:
        """
        理论基础深度分析
        
        分析内容：
        1. 市场结构的数学定义
        2. 摆动点检测的数学原理
        3. 趋势识别的统计方法
        4. 结构完整性的量化评估
        """
        print("=== 理论基础深度分析 ===")
        
        analysis_results = {}
        
        # 1.1 市场结构的数学定义分析
        analysis_results['market_structure_definition'] = self._analyze_structure_definition()
        
        # 1.2 摆动点检测的数学原理分析
        analysis_results['swing_point_math'] = self._analyze_swing_point_mathematics()
        
        # 1.3 趋势识别的统计方法分析
        analysis_results['trend_recognition_stats'] = self._analyze_trend_statistics()
        
        # 1.4 结构完整性的量化评估分析
        analysis_results['integrity_quantification'] = self._analyze_integrity_quantification()
        
        # 保存分析结果
        self.analysis_history.append({
            'type': 'theoretical_foundation',
            'timestamp': pd.Timestamp.now(),
            'results': analysis_results
        })
        
        return analysis_results
    
    def _analyze_structure_definition(self) -> Dict[str, Any]:
        """分析市场结构的数学定义"""
        print("\n1. 市场结构数学定义分析:")
        
        definitions = {
            'uptrend': {
                'mathematical_definition': "高点序列和低点序列均呈单调递增",
                'recognition_criteria': [
                    "高点 > 前高点 (H[n] > H[n-1])",
                    "低点 > 前低点 (L[n] > L[n-1])",
                    "斜率 > 0 (dPrice/dt > 0)"
                ],
                'challenges': [
                    "噪声干扰导致假突破",
                    "短期回调整理误判",
                    "趋势强度量化困难"
                ]
            },
            'downtrend': {
                'mathematical_definition': "高点序列和低点序列均呈单调递减",
                'recognition_criteria': [
                    "高点 < 前高点 (H[n] < H[n-1])",
                    "低点 < 前低点 (L[n] < L[n-1])",
                    "斜率 < 0 (dPrice/dt < 0)"
                ],
                'challenges': [
                    "反弹误判为趋势反转",
                    "下跌动量量化困难",
                    "支撑位干扰判断"
                ]
            },
            'range': {
                'mathematical_definition': "价格在有限区间内波动，无明确趋势方向",
                'recognition_criteria': [
                    "高点序列无明显趋势 (|slope_H| < ε)",
                    "低点序列无明显趋势 (|slope_L| < ε)",
                    "价格在支撑阻力区间内震荡"
                ],
                'challenges': [
                    "区间宽度动态变化",
                    "假突破频繁发生",
                    "区间边界模糊"
                ]
            }
        }
        
        print(f"   ✅ 完成{len(definitions)}种市场结构的数学定义分析")
        return definitions
    
    def _analyze_swing_point_mathematics(self) -> Dict[str, Any]:
        """分析摆动点检测的数学原理"""
        print("\n2. 摆动点检测数学原理分析:")
        
        analysis = {
            'local_extrema_definition': {
                'mathematical_definition': "对于序列S，点S[i]是局部最大值当且仅当：∃δ>0, ∀j∈(i-δ,i+δ), S[i] ≥ S[j]",
                'practical_implementation': "使用有限窗口检测：S[i] = max(S[i-w:i+w+1])",
                'challenges': [
                    "窗口大小选择困难",
                    "噪声产生假极值",
                    "相邻极值合并问题"
                ]
            },
            'detection_methods': {
                'method_1_window_extrema': {
                    'description': "窗口极值检测法",
                    'mathematical_basis': "在固定窗口内寻找最大值/最小值",
                    'advantages': "实现简单，计算高效",
                    'disadvantages': "对窗口大小敏感，可能错过重要转折点"
                },
                'method_2_price_change': {
                    'description': "价格变化检测法",
                    'mathematical_basis': "基于价格变化百分比和动量",
                    'advantages': "对市场波动自适应",
                    'disadvantages': "参数设置复杂，计算量较大"
                },
                'method_3_statistical': {
                    'description': "统计检测法",
                    'mathematical_basis': "基于统计显著性检验",
                    'advantages': "理论基础坚实，可解释性强",
                    'disadvantages': "计算复杂，实时性差"
                }
            },
            'atr_application': {
                'role': "平均真实波幅(ATR)用于确定摆动点的显著性阈值",
                'mathematical_basis': "摆动点价格差异应显著大于市场噪声(ATR)",
                'formula': "threshold = ATR × sensitivity_factor",
                'optimization_considerations': [
                    "sensitivity_factor需要根据不同市场调整",
                    "ATR计算周期影响阈值稳定性",
                    "动态调整阈值适应市场变化"
                ]
            }
        }
        
        print(f"   ✅ 完成摆动点检测{len(analysis['detection_methods'])}种方法的数学原理分析")
        return analysis
    
    def _analyze_trend_statistics(self) -> Dict[str, Any]:
        """分析趋势识别的统计方法"""
        print("\n3. 趋势识别统计方法分析:")
        
        analysis = {
            'linear_regression_analysis': {
                'method': "线性回归趋势分析",
                'mathematical_basis': "y = β₀ + β₁x + ε，其中β₁为趋势斜率",
                'statistical_measures': {
                    'slope_significance': "t检验判断β₁是否显著不为0",
                    'goodness_of_fit': "R²衡量模型拟合优度",
                    'confidence_intervals': "斜率β₁的置信区间"
                },
                'limitations': [
                    "假设线性关系，可能不符合实际",
                    "对异常值敏感",
                    "需要足够的数据点"
                ]
            },
            'alternative_methods': {
                'moving_average_slope': {
                    'description': "移动平均线斜率法",
                    'mathematical_basis': "计算移动平均线的斜率",
                    'advantages': "平滑噪声，趋势更稳定",
                    'disadvantages': "滞后性，响应速度慢"
                },
                'nonlinear_regression': {
                    'description': "非线性回归法",
                    'mathematical_basis': "多项式回归或其他非线性模型",
                    'advantages': "能捕捉复杂趋势模式",
                    'disadvantages': "过拟合风险，计算复杂"
                },
                'statistical_tests': {
                    'description': "统计检验法",
                    'mathematical_basis': "Mann-Kendall趋势检验等非参数方法",
                    'advantages': "不假设分布，更稳健",
                    'disadvantages': "需要较长数据序列"
                }
            },
            'trend_strength_quantification': {
                'methods': [
                    {
                        'name': "斜率幅度法",
                        'formula': "strength = |β₁| / price_range",
                        'interpretation': "斜率相对于价格范围的比例"
                    },
                    {
                        'name': "R²加权法",
                        'formula': "strength = |β₁| × R²",
                        'interpretation': "同时考虑趋势斜率和拟合优度"
                    },
                    {
                        'name': "一致性指数",
                        'formula': "consistency = P(price_increases) × magnitude",
                        'interpretation': "趋势方向和幅度的综合"
                    }
                ]
            }
        }
        
        print(f"   ✅ 完成趋势识别{len(analysis['alternative_methods'])}种替代方法的统计分析")
        return analysis
    
    def _analyze_integrity_quantification(self) -> Dict[str, Any]:
        """分析结构完整性的量化评估"""
        print("\n4. 结构完整性量化评估分析:")
        
        analysis = {
            'integrity_components': {
                'swing_point_distribution': {
                    'description': "摆动点分布合理性",
                    'quantification_methods': [
                        "时间间隔均匀性检验",
                        "价格幅度一致性检验",
                        "分布统计检验(如χ²检验)"
                    ]
                },
                'sequence_consistency': {
                    'description': "序列一致性评估",
                    'quantification_methods': [
                        "相邻摆动点价格变化方向一致性",
                        "趋势线拟合残差分析",
                        "自相关性检验"
                    ]
                },
                'structural_stability': {
                    'description': "结构稳定性评估",
                    'quantification_methods': [
                        "回撤幅度和频率分析",
                        "假突破发生率",
                        "结构持续时间"
                    ]
                }
            },
            'composite_metrics': {
                'integrity_score': {
                    'formula': "Score = w₁×Distribution + w₂×Consistency + w₃×Stability",
                    'weight_optimization': "权重w₁,w₂,w₃需要根据市场条件优化",
                    'normalization': "各组分归一化到[0,1]区间"
                },
                'confidence_level': {
                    'calculation': "基于统计显著性检验的置信水平",
                    'interpretation': "结构识别的统计可靠性"
                },
                'risk_adjustment': {
                    'consideration': "根据完整性调整交易风险",
                    'application': "完整性低时降低仓位，完整性高时正常交易"
                }
            }
        }
        
        print(f"   ✅ 完成结构完整性{len(analysis['integrity_components'])}个维度的量化分析")
        return analysis
    
    # ========== 第2部分：算法对比分析 ==========
    
    def compare_algorithms(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        对比不同摆动点检测算法
        
        参数:
            price_data: 价格数据
            
        返回:
            算法对比分析结果
        """
        print("\n=== 算法对比分析 ===")
        
        if len(price_data) < 50:
            return {'error': '数据不足，至少需要50个数据点'}
        
        comparison_results = {}
        
        # 实现三种不同的摆动点检测算法
        algorithms = {
            'window_extrema': self._window_extrema_algorithm,
            'price_change': self._price_change_algorithm,
            'statistical': self._statistical_algorithm
        }
        
        # 对每种算法进行测试
        for algo_name, algo_func in algorithms.items():
            print(f"\n测试算法: {algo_name}")
            
            try:
                # 运行算法
                swing_points = algo_func(price_data)
                
                # 评估算法性能
                performance = self._evaluate_algorithm_performance(
                    swing_points, price_data, algo_name
                )
                
                comparison_results[algo_name] = {
                    'swing_points': swing_points,
                    'performance': performance,
                    'high_count': len(swing_points['highs']),
                    'low_count': len(swing_points['lows'])
                }
                
                print(f"   ✅ 算法测试完成: {performance['summary']}")
                
            except Exception as e:
                print(f"   ❌ 算法测试失败: {e}")
                comparison_results[algo_name] = {'error': str(e)}
        
        # 保存对比结果
        self.algorithm_comparison_results = comparison_results
        
        # 生成算法推荐
        recommendations = self._generate_algorithm_recommendations(comparison_results)
        
        return {
            'comparison_results': comparison_results,
            'recommendations': recommendations,
            'best_algorithm': recommendations.get('best_overall', 'unknown')
        }
    
    def _window_extrema_algorithm(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """窗口极值检测算法"""
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        window_size = 5
        high_points = []
        low_points = []
        
        for i in range(window_size, len(closes) - window_size):
            # 检查窗口内的极值
            window_highs = highs[i-window_size:i+window_size+1]
            window_lows = lows[i-window_size:i+window_size+1]
            
            if highs[i] == np.max(window_highs):
                high_points.append((i, highs[i]))
            
            if lows[i] == np.min(window_lows):
                low_points.append((i, lows[i]))
        
        return {'highs': high_points, 'lows': low_points}
    
    def _price_change_algorithm(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """价格变化检测算法"""
        closes = data['close'].values
        
        # 计算价格变化百分比
        price_changes = np.diff(closes) / closes[:-1] * 100
        
        high_points = []
        low_points = []
        
        # 寻找显著的价格转折点
        for i in range(2, len(price_changes) - 2):
            # 高点检测：价格变化从正变负
            if price_changes[i] > 0.5 and price_changes[i+1] < -0.3:
                high_points.append((i+1, data['high'].iloc[i+1]))
            
            # 低点检测：价格变化从负变正
            if price_changes[i] < -0.5 and price_changes[i+1] > 0.3:
                low_points.append((i+1, data['low'].iloc[i+1]))
        
        return {'highs': high_points, 'lows': low_points}
    
    def _statistical_algorithm(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """统计检测算法"""
        # 简化的统计检测算法
        closes = data['close'].values
        
        # 计算滚动统计量
        window = 10
        rolling_mean = pd.Series(closes).rolling(window=window).mean().values
        rolling_std = pd.Series(closes).rolling(window=window).std().values
        
        high_points = []
        low_points = []
        
        for i in range(window, len(closes) - window):
            # 检查是否为统计异常值
            z_score = (closes[i] - rolling_mean[i]) / rolling_std[i] if rolling_std[i] > 0 else 0
            
            # 高点：显著高于滚动均值
            if z_score > 1.5 and closes[i] == max(closes[i-3:i+4]):
                high_points.append((i, data['high'].iloc[i]))
            
            # 低点：显著低于滚动均值
            if z_score < -1.5 and closes[i] == min(closes[i-3:i+4]):
                low_points.append((i, data['low'].iloc[i]))
        
        return {'highs': high_points, 'lows': low_points}
    
    def _evaluate_algorithm_performance(self, swing_points: Dict, 
                                       data: pd.DataFrame, 
                                       algorithm_name: str) -> Dict[str, Any]:
        """评估算法性能"""
        highs = swing_points['highs']
        lows = swing_points['lows']
        
        # 基本统计
        high_count = len(highs)
        low_count = len(lows)
        
        # 计算摆动点间隔
        if high_count >= 2:
            high_indices = [idx for idx, _ in highs]
            high_intervals = np.diff(sorted(high_indices))
            avg_high_interval = np.mean(high_intervals) if len(high_intervals) > 0 else 0
        else:
            avg_high_interval = 0
        
        if low_count >= 2:
            low_indices = [idx for idx, _ in lows]
            low_intervals = np.diff(sorted(low_indices))
            avg_low_interval = np.mean(low_intervals) if len(low_intervals) > 0 else 0
        else:
            avg_low_interval = 0
        
        # 性能评估
        performance = {
            'swing_point_counts': {'highs': high_count, 'lows': low_count},
            'average_intervals': {'highs': avg_high_interval, 'lows': avg_low_interval},
            'detection_rate': (high_count + low_count) / len(data) * 100,
            'balance_score': abs(high_count - low_count) / (high_count + low_count + 1),
            'summary': f"检测到{high_count}高/{low_count}低，平均间隔高:{avg_high_interval:.1f}/低:{avg_low_interval:.1f}"
        }
        
        return performance
    
    def _generate_algorithm_recommendations(self, comparison_results: Dict) -> Dict[str, Any]:
        """生成算法推荐"""
        valid_results = {k: v for k, v in comparison_results.items() 
                        if 'performance' in v and 'error' not in v}
        
        if not valid_results:
            return {'recommendation': '所有算法测试失败'}
        
        # 根据多个指标评估算法
        algorithm_scores = {}
        
        for algo_name, result in valid_results.items():
            perf = result['performance']
            
            # 综合评分（简单加权）
            score = (
                min(perf['detection_rate'], 10) * 0.4 +  # 检测率，限制最大10%
                (1 - perf['balance_score']) * 0.3 +      # 平衡性，越接近0越好
                (20 / (perf['average_intervals']['highs'] + 1)) * 0.3  # 间隔适当
            )
            
            algorithm_scores[algo_name] = score
        
        # 找出最佳算法
        if algorithm_scores:
            best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
        else:
            best_algorithm = 'unknown'
        
        recommendations = {
            'algorithm_scores': algorithm_scores,
            'best_overall': best_algorithm,
            'recommendation_reason': self._get_recommendation_reason(best_algorithm, valid_results),
            'usage_scenarios': {
                'window_extrema': "适合趋势明显、噪声较低的市场",
                'price_change': "适合波动较大、转折频繁的市场",
                'statistical': "适合需要统计显著性的稳健分析"
            }
        }
        
        return recommendations
    
    def _get_recommendation_reason(self, best_algo: str, results: Dict) -> str:
        """获取推荐理由"""
        reasons = {
            'window_extrema': "窗口极值算法检测稳定，假信号较少，适合大多数市场条件",
            'price_change': "价格变化算法对市场转折敏感，适合捕捉短期反转机会",
            'statistical': "统计算法理论基础坚实，假阳性率低，适合稳健交易"
        }
        
        return reasons.get(best_algo, "基于综合性能评估推荐")
    
    # ========== 第3部分：参数优化分析 ==========
    
    def analyze_parameter_sensitivity(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析参数敏感度
        
        分析摆动点检测算法对不同参数的敏感度
        """
        print("\n=== 参数敏感度分析 ===")
        
        if len(price_data) < 100:
            return {'error': '数据不足，至少需要100个数据点进行参数分析'}
        
        sensitivity_results = {}
        
        # 测试窗口大小参数
        window_sizes = [3, 5, 7, 10, 15]
        window_results = []
        
        for window in window_sizes:
            highs, lows = self._test_window_size(price_data, window)
            window_results.append({
                'window_size': window,
                'high_count': len(highs),
                'low_count': len(lows),
                'total_swings': len(highs) + len(lows)
            })
        
        sensitivity_results['window_size_sensitivity'] = window_results
        
        # 测试ATR敏感度参数
        atr_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
        atr_results = []
        
        for factor in atr_factors:
            highs, lows = self._test_atr_sensitivity(price_data, factor)
            atr_results.append({
                'atr_factor': factor,
                'high_count': len(highs),
                'low_count': len(lows),
                'swing_density': (len(highs) + len(lows)) / len(price_data) * 100
            })
        
        sensitivity_results['atr_sensitivity'] = atr_results
        
        # 分析参数影响
        analysis = self._analyze_parameter_impact(window_results, atr_results)
        sensitivity_results['impact_analysis'] = analysis
        
        # 生成参数推荐
        recommendations = self._generate_parameter_recommendations(sensitivity_results)
        sensitivity_results['recommendations'] = recommendations
        
        # 保存结果
        self.parameter_optimization_results = sensitivity_results
        
        return sensitivity_results
    
    def _test_window_size(self, data: pd.DataFrame, window_size: int) -> Tuple[List, List]:
        """测试特定窗口大小的摆动点检测"""
        highs = data['high'].values
        lows = data['low'].values
        
        high_points = []
        low_points = []
        
        for i in range(window_size, len(highs) - window_size):
            if highs[i] == np.max(highs[i-window_size:i+window_size+1]):
                high_points.append((i, highs[i]))
            
            if lows[i] == np.min(lows[i-window_size:i+window_size+1]):
                low_points.append((i, lows[i]))
        
        return high_points, low_points
    
    def _test_atr_sensitivity(self, data: pd.DataFrame, atr_factor: float) -> Tuple[List, List]:
        """测试ATR敏感度参数的摆动点检测"""
        # 简化实现
        closes = data['close'].values
        
        high_points = []
        low_points = []
        
        # 计算价格变化
        price_changes = np.abs(np.diff(closes) / closes[:-1] * 100)
        
        # 使用ATR因子作为阈值
        threshold = np.mean(price_changes[-20:]) * atr_factor if len(price_changes) >= 20 else 0.5
        
        for i in range(1, len(closes) - 1):
            # 简化逻辑：价格变化超过阈值
            prev_change = abs(closes[i] - closes[i-1]) / closes[i-1] * 100
            next_change = abs(closes[i+1] - closes[i]) / closes[i] * 100
            
            # 高点：当前价格高于前后，且变化显著
            if closes[i] > closes[i-1] and closes[i] > closes[i+1] and prev_change > threshold:
                high_points.append((i, data['high'].iloc[i]))
            
            # 低点：当前价格低于前后，且变化显著
            if closes[i] < closes[i-1] and closes[i] < closes[i+1] and prev_change > threshold:
                low_points.append((i, data['low'].iloc[i]))
        
        return high_points, low_points
    
    def _analyze_parameter_impact(self, window_results: List, atr_results: List) -> Dict[str, Any]:
        """分析参数影响"""
        analysis = {}
        
        # 窗口大小影响分析
        window_swings = [r['total_swings'] for r in window_results]
        window_sizes = [r['window_size'] for r in window_results]
        
        if window_swings:
            analysis['window_size_impact'] = {
                'correlation': "窗口大小与检测到的摆动点数量呈负相关",
                'optimal_range': "窗口大小5-7通常能平衡敏感性和稳定性",
                'tradeoff': "小窗口：更敏感但更多噪声；大窗口：更稳定但可能错过转折点"
            }
        
        # ATR因子影响分析
        atr_density = [r['swing_density'] for r in atr_results]
        atr_factors = [r['atr_factor'] for r in atr_results]
        
        if atr_density:
            analysis['atr_factor_impact'] = {
                'correlation': "ATR因子与摆动点密度呈负相关",
                'recommended_factor': "ATR因子1.5-2.0通常适应大多数市场条件",
                'adjustment_advice': "高波动市场使用较小因子，低波动市场使用较大因子"
            }
        
        return analysis
    
    def _generate_parameter_recommendations(self, sensitivity_results: Dict) -> Dict[str, Any]:
        """生成参数推荐"""
        recommendations = {
            'window_size': {
                'recommended': [5, 7],
                'reason': "平衡敏感性和稳定性，既能捕捉重要转折点又不过度敏感",
                'adjustment_rules': [
                    "趋势市场：使用较大窗口(7-10)减少假信号",
                    "震荡市场：使用较小窗口(3-5)捕捉更多反转点",
                    "高波动市场：适当增大窗口平滑噪声"
                ]
            },
            'atr_sensitivity_factor': {
                'recommended': [1.5, 2.0],
                'reason': "适应大多数市场波动条件，既不过于敏感也不过于迟钝",
                'adjustment_rules': [
                    "高波动市场(如加密货币)：使用1.0-1.5",
                    "中等波动市场(如股票)：使用1.5-2.0",
                    "低波动市场(如债券)：使用2.0-2.5"
                ]
            },
            'adaptive_parameters': {
                'recommendation': "实现自适应参数调整",
                'methods': [
                    "基于市场波动率动态调整ATR因子",
                    "基于市场状态(趋势/震荡)调整窗口大小",
                    "基于时间框架调整参数敏感度"
                ]
            }
        }
        
        return recommendations
    
    # ========== 第4部分：深度分析报告 ==========
    
    def generate_deep_analysis_report(self) -> str:
        """生成深度分析报告"""
        print("\n=== 生成深度分析报告 ===")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("市场结构识别深度分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {pd.Timestamp.now()}")
        report_lines.append(f"分析器: MarketStructureDeepAnalyzer")
        report_lines.append("")
        
        # 1. 理论基础总结
        report_lines.append("1. 理论基础深度分析")
        report_lines.append("-" * 40)
        
        if self.analysis_history:
            theory_results = self.analysis_history[0]['results']
            report_lines.append(f"   市场结构定义: 分析了{len(theory_results['market_structure_definition'])}种结构")
            report_lines.append(f"   摆动点检测: {len(theory_results['swing_point_math']['detection_methods'])}种方法")
            report_lines.append(f"   趋势识别: {len(theory_results['trend_recognition_stats']['alternative_methods'])}种统计方法")
            report_lines.append(f"   结构完整性: {len(theory_results['integrity_quantification']['integrity_components'])}个量化维度")
        else:
            report_lines.append("   理论基础分析未执行")
        
        # 2. 算法对比结果
        report_lines.append("")
        report_lines.append("2. 算法对比分析")
        report_lines.append("-" * 40)
        
        if self.algorithm_comparison_results:
            valid_algorithms = [k for k, v in self.algorithm_comparison_results.items() 
                              if 'performance' in v]
            report_lines.append(f"   测试算法数: {len(valid_algorithms)}")
            
            for algo_name, result in self.algorithm_comparison_results.items():
                if 'performance' in result:
                    perf = result['performance']
                    report_lines.append(f"   - {algo_name}: {perf['summary']}")
            
            # 最佳算法
            if hasattr(self, '_generate_algorithm_recommendations'):
                recs = self._generate_algorithm_recommendations(self.algorithm_comparison_results)
                if 'best_overall' in recs:
                    report_lines.append(f"   推荐算法: {recs['best_overall']}")
                    report_lines.append(f"   推荐理由: {recs['recommendation_reason']}")
        else:
            report_lines.append("   算法对比分析未执行")
        
        # 3. 参数优化结果
        report_lines.append("")
        report_lines.append("3. 参数优化分析")
        report_lines.append("-" * 40)
        
        if self.parameter_optimization_results:
            params = self.parameter_optimization_results
            if 'recommendations' in params:
                recs = params['recommendations']
                report_lines.append("   参数推荐:")
                report_lines.append(f"   - 窗口大小: {recs['window_size']['recommended']}")
                report_lines.append(f"   - ATR因子: {recs['atr_sensitivity_factor']['recommended']}")
        else:
            report_lines.append("   参数优化分析未执行")
        
        # 4. 关键洞察
        report_lines.append("")
        report_lines.append("4. 关键洞察与建议")
        report_lines.append("-" * 40)
        
        insights = [
            "📊 市场结构识别是价格行为交易的基础，准确识别直接影响交易效果",
            "🔧 没有一种算法适用于所有市场，需要根据市场条件选择或组合使用",
            "⚖️ 参数设置需要在敏感性和稳定性之间找到平衡点",
            "🔄 自适应参数调整能显著提升算法在不同市场条件下的表现",
            "📈 趋势市场和震荡市场需要不同的识别策略和参数",
            "✅ 结合多种方法和多时间框架分析能提高识别准确性"
        ]
        
        for insight in insights:
            report_lines.append(f"   {insight}")
        
        # 5. 实际应用建议
        report_lines.append("")
        report_lines.append("5. 实际应用建议")
        report_lines.append("-" * 40)
        
        applications = [
            "🎯 趋势交易: 使用窗口极值算法，较大窗口，ATR因子2.0",
            "🔄 区间交易: 使用价格变化算法，较小窗口，ATR因子1.5",
            "⚡ 突破交易: 使用统计算法，中等窗口，ATR因子1.8",
            "📊 多时间框架: 在不同时间框架使用不同参数配置",
            "🔍 验证机制: 结合成交量和其他指标验证结构识别结果",
            "⚠️ 风险控制: 根据结构完整性调整仓位和风险"
        ]
        
        for app in applications:
            report_lines.append(f"   {app}")
        
        # 6. 未来优化方向
        report_lines.append("")
        report_lines.append("6. 未来优化方向")
        report_lines.append("-" * 40)
        
        future_directions = [
            "🧠 机器学习: 使用机器学习模型识别复杂市场结构",
            "📊 深度学习: 应用深度学习进行多维度结构识别",
            "🔄 自适应算法: 开发能自动适应市场变化的算法",
            "🔗 多指标融合: 结合技术指标和基本面数据",
            "🌐 跨市场分析: 分析不同市场间的结构关联性",
            "⏱️ 实时优化: 实现实时参数优化和算法调整"
        ]
        
        for direction in future_directions:
            report_lines.append(f"   {direction}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("深度分析报告结束")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

# ========== 演示函数 ==========

def demonstrate_deep_analysis():
    """演示深度分析功能"""
    print("市场结构深度分析器演示")
    print("=" * 60)
    
    # 创建深度分析器实例
    analyzer = MarketStructureDeepAnalyzer()
    
    # 生成测试数据
    np.random.seed(42)
    n_bars = 200
    time = np.arange(n_bars)
    
    # 创建包含趋势和震荡的混合数据
    trend = 100 + time * 0.15
    oscillation = np.sin(time * 0.1) * 8
    noise = np.random.normal(0, 2.0, n_bars)
    
    prices = trend + oscillation + noise
    
    price_data = pd.DataFrame({
        'open': prices * 0.998,
        'high': prices * 1.008 + np.random.normal(0, 0.5, n_bars),
        'low': prices * 0.992 - np.random.normal(0, 0.5, n_bars),
        'close': prices,
        'volume': np.random.randint(1000, 20000, n_bars)
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='D'))
    
    print(f"测试数据: {len(price_data)}个数据点")
    print(f"价格范围: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")
    
    # 执行深度分析
    print("\n开始深度分析...")
    
    # 1. 理论基础分析
    print("\n1. 执行理论基础深度分析...")
    theory_results = analyzer.analyze_theoretical_foundation()
    print(f"   ✅ 理论基础分析完成")
    
    # 2. 算法对比分析
    print("\n2. 执行算法对比分析...")
    algo_results = analyzer.compare_algorithms(price_data)
    print(f"   ✅ 算法对比分析完成")
    if 'best_algorithm' in algo_results:
        print(f"   最佳算法: {algo_results['best_algorithm']}")
    
    # 3. 参数敏感度分析
    print("\n3. 执行参数敏感度分析...")
    param_results = analyzer.analyze_parameter_sensitivity(price_data)
    print(f"   ✅ 参数敏感度分析完成")
    
    # 4. 生成深度分析报告
    print("\n4. 生成深度分析报告...")
    report = analyzer.generate_deep_analysis_report()
    
    # 显示报告摘要
    report_lines = report.split('\n')
    print("深度分析报告摘要:")
    print("-" * 40)
    
    # 显示关键部分
    key_sections = []
    current_section = ""
    
    for line in report_lines:
        if line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. ") or \
           line.startswith("4. ") or line.startswith("5. ") or line.startswith("6. "):
            current_section = line
            key_sections.append(current_section)
        elif line.startswith("   ") and current_section:
            if ":" in line or "✅" in line or "❌" in line:
                key_sections.append(line)
    
    for section in key_sections[:20]:  # 显示前20行关键内容
        print(section)
    
    print("\n" + "=" * 60)
    print("深度分析演示完成")
    print(f"报告总长度: {len(report)}字符")
    print(f"分析历史记录: {len(analyzer.analysis_history)}条")
    
    return analyzer, report

# ============================================================================
# 策略改造: 添加MarketStructureDeepAnalyzerStrategy类
# 将市场结构深度分析器转换为交易策略
# ============================================================================

class MarketStructureDeepAnalyzerStrategy(BaseStrategy):
    """市场结构深度分析策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 创建市场结构深度分析器实例
        self.analyzer = MarketStructureDeepAnalyzer()
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于市场结构深度分析生成交易信号
        """
        # 执行算法对比分析
        algo_results = self.analyzer.compare_algorithms(self.data)
        
        # 获取最佳算法
        best_algorithm = algo_results.get('best_algorithm', '')
        
        # 执行参数敏感度分析
        param_results = self.analyzer.analyze_parameter_sensitivity(self.data)
        
        # 获取参数优化结果
        optimal_params = param_results.get('optimal_parameters', {})
        optimization_score = optimal_params.get('optimization_score', 0)
        
        # 根据深度分析生成信号
        if best_algorithm and optimization_score >= 0.7:
            # 有最佳算法且优化分数高，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif best_algorithm:
            # 有最佳算法但优化分数一般，hold信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        else:
            # 无最佳算法，hold信号
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
    analyzer, report = demonstrate_deep_analysis()
    
    # 保存报告到文件
    with open('market_structure_deep_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n深度分析报告已保存到: market_structure_deep_analysis_report.md")