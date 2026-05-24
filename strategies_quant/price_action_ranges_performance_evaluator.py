# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.633964

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绩效评估量化系统
第25章：绩效评估
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 综合绩效指标：夏普比率、索提诺比率、卡尔马比率等
2. 基准比较：与市场基准、同行基准比较
3. 风险评估：下行风险、风险价值（VaR）、条件风险价值（CVaR）
4. 绩效归因：收益来源分析、风险来源分析
5. 绩效报告：综合绩效报告、风险评估报告
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

class PerformanceEvaluator:
    """绩效评估器（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 benchmark_data: pd.DataFrame = None,
                 risk_free_rate: float = 0.02):
        """
        初始化绩效评估器
        
        参数:
            benchmark_data: 基准数据（包含日期和收益率）
            risk_free_rate: 无风险利率（年化）
        """
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate
        self.evaluation_history = []
        
        # 初始化绩效指标计算器
        self.metric_calculators = self._initialize_metric_calculators()
    
    def _initialize_metric_calculators(self) -> Dict:
        """初始化指标计算器"""
        calculators = {
            'basic': self._calculate_basic_metrics,
            'risk_adjusted': self._calculate_risk_adjusted_metrics,
            'drawdown': self._calculate_drawdown_metrics,
            'advanced': self._calculate_advanced_metrics,
            'benchmark': self._calculate_benchmark_metrics
        }
        return calculators
    
    def evaluate_performance(self,
                            returns: pd.Series,
                            initial_capital: float = 10000.0,
                            include_benchmark: bool = True,
                            confidence_level: float = 0.95) -> Dict:
        """
        评估绩效
        
        参数:
            returns: 收益率序列（日度或其它频率）
            initial_capital: 初始资本
            include_benchmark: 是否包含基准比较
            confidence_level: VaR/CVaR的置信水平
            
        返回:
            绩效评估结果
        """
        if returns.empty:
            return {'error': '收益率序列为空'}
        
        # 生成评估ID
        eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 计算基础指标
        basic_metrics = self._calculate_basic_metrics(returns, initial_capital)
        
        # 计算风险调整后指标
        risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(returns)
        
        # 计算回撤指标
        drawdown_metrics = self._calculate_drawdown_metrics(returns, initial_capital)
        
        # 计算高级指标
        advanced_metrics = self._calculate_advanced_metrics(returns, confidence_level)
        
        # 基准比较（如果可用）
        benchmark_metrics = {}
        if include_benchmark and self.benchmark_data is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, self.benchmark_data)
        
        # 计算综合绩效分数
        composite_score = self._calculate_composite_score(
            basic_metrics, risk_adjusted_metrics, drawdown_metrics, advanced_metrics
        )
        
        # 生成绩效报告
        performance_report = self._generate_performance_report(
            basic_metrics, risk_adjusted_metrics, drawdown_metrics,
            advanced_metrics, benchmark_metrics, composite_score
        )
        
        # 生成改进建议
        improvement_suggestions = self._generate_improvement_suggestions(
            basic_metrics, risk_adjusted_metrics, drawdown_metrics, advanced_metrics
        )
        
        # 组装结果
        evaluation_result = {
            'evaluation_id': eval_id,
            'evaluation_time': datetime.now(),
            'returns_summary': {
                'periods': len(returns),
                'start_date': returns.index[0] if hasattr(returns.index[0], 'strftime') else returns.index[0],
                'end_date': returns.index[-1] if hasattr(returns.index[-1], 'strftime') else returns.index[-1],
                'total_return': basic_metrics['total_return'],
                'annualized_return': basic_metrics['annualized_return']
            },
            'basic_metrics': basic_metrics,
            'risk_adjusted_metrics': risk_adjusted_metrics,
            'drawdown_metrics': drawdown_metrics,
            'advanced_metrics': advanced_metrics,
            'benchmark_metrics': benchmark_metrics,
            'composite_score': composite_score,
            'performance_report': performance_report,
            'improvement_suggestions': improvement_suggestions,
            'performance_grade': self._assign_performance_grade(composite_score)
        }
        
        self.evaluation_history.append(evaluation_result)
        return evaluation_result
    
    def _calculate_basic_metrics(self,
                                returns: pd.Series,
                                initial_capital: float) -> Dict:
        """计算基础绩效指标"""
        if returns.empty:
            return {}
        
        # 总收益率
        total_return = (1 + returns).prod() - 1
        
        # 年化收益率
        if len(returns) > 1:
            # 估算年化周期数
            if isinstance(returns.index, pd.DatetimeIndex):
                days_diff = (returns.index[-1] - returns.index[0]).days
                if days_diff > 0:
                    years = days_diff / 365.25
                    annualized_return = (1 + total_return) ** (1 / years) - 1
                else:
                    annualized_return = total_return
            else:
                # 如果没有日期，假设每日数据
                annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        else:
            annualized_return = total_return
        
        # 平均收益率
        mean_return = returns.mean()
        
        # 收益率标准差
        std_return = returns.std()
        
        # 偏度和峰度
        skewness = returns.skew() if len(returns) >= 3 else 0
        kurtosis = returns.kurtosis() if len(returns) >= 4 else 0
        
        # 正收益天数比例
        positive_days = (returns > 0).sum()
        positive_ratio = positive_days / len(returns) if len(returns) > 0 else 0
        
        # 最大单日收益/亏损
        max_gain = returns.max()
        max_loss = returns.min()
        
        # 计算累计净值
        cumulative_returns = (1 + returns).cumprod()
        final_capital = initial_capital * (1 + total_return)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'mean_return': mean_return,
            'std_return': std_return,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'positive_ratio': positive_ratio,
            'positive_days': positive_days,
            'total_days': len(returns),
            'max_gain': max_gain,
            'max_loss': max_loss,
            'final_capital': final_capital,
            'initial_capital': initial_capital,
            'profit': final_capital - initial_capital
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """计算风险调整后指标"""
        if len(returns) < 2:
            return {}
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # 夏普比率
        if std_return > 0:
            # 将无风险利率转换为相同频率
            if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 1:
                # 估算年化周期数
                days_diff = (returns.index[-1] - returns.index[0]).days
                if days_diff > 0:
                    years = days_diff / 365.25
                    periods_per_year = len(returns) / years
                else:
                    periods_per_year = 252  # 默认年化252个交易日
            else:
                periods_per_year = 252
            
            risk_free_per_period = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
            excess_return = mean_return - risk_free_per_period
            sharpe_ratio = excess_return / std_return * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
        
        # 索提诺比率（只考虑下行风险）
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino_ratio = (mean_return - self.risk_free_rate / 252) / downside_std * np.sqrt(252)
            else:
                sortino_ratio = float('inf') if mean_return > self.risk_free_rate / 252 else 0
        else:
            sortino_ratio = 0
        
        # 特雷诺比率（需要贝塔，这里简化）
        treynor_ratio = 0  # 将在基准比较中计算
        
        # 信息比率（需要基准，将在基准比较中计算）
        information_ratio = 0
        
        # 卡玛比率（Calmar Ratio）
        # 需要最大回撤，将在回撤指标中计算
        
        # M2测度（Modigliani-Modigliani）
        if std_return > 0 and sharpe_ratio > 0:
            # 简化计算
            m2_measure = sharpe_ratio * std_return
        else:
            m2_measure = 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'treynor_ratio': treynor_ratio,
            'information_ratio': information_ratio,
            'm2_measure': m2_measure,
            'downside_std': downside_returns.std() if len(downside_returns) > 1 else 0,
            'upside_std': returns[returns > 0].std() if len(returns[returns > 0]) > 1 else 0
        }
    
    def _calculate_drawdown_metrics(self,
                                   returns: pd.Series,
                                   initial_capital: float) -> Dict:
        """计算回撤指标"""
        if returns.empty:
            return {}
        
        # 计算累计净值
        cumulative_returns = (1 + returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        # 计算回撤
        running_max = portfolio_value.cummax()
        drawdown = (portfolio_value - running_max) / running_max
        
        # 最大回撤
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin() if hasattr(drawdown, 'idxmin') else None
        
        # 当前回撤
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        # 回撤持续时间
        drawdown_durations = []
        in_drawdown = False
        start_date = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.001:  # 超过0.1%的回撤
                if not in_drawdown:
                    in_drawdown = True
                    start_date = drawdown.index[i] if hasattr(drawdown.index, '__getitem__') else i
            else:
                if in_drawdown:
                    in_drawdown = False
                    end_date = drawdown.index[i] if hasattr(drawdown.index, '__getitem__') else i
                    
                    if start_date is not None:
                        if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime'):
                            duration_days = (end_date - start_date).days
                        else:
                            duration_days = end_date - start_date
                        
                        drawdown_durations.append(duration_days)
        
        # 如果结束时仍在回撤中
        if in_drawdown and start_date is not None:
            end_date = drawdown.index[-1] if hasattr(drawdown.index, '__getitem__') else len(drawdown) - 1
            
            if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime'):
                duration_days = (end_date - start_date).days
            else:
                duration_days = end_date - start_date
            
            drawdown_durations.append(duration_days)
        
        # 平均回撤持续时间
        avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0
        
        # 最长回撤持续时间
        max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
        
        # 恢复时间（从最大回撤恢复到前高）
        recovery_time = 0
        if max_drawdown_date is not None:
            # 找到回撤开始的日期
            max_dd_value = portfolio_value.loc[max_drawdown_date] if hasattr(portfolio_value, 'loc') else portfolio_value[max_drawdown_date]
            
            # 找到前高的日期
            pre_peak_idx = portfolio_value[:max_drawdown_date].idxmax() if hasattr(portfolio_value[:max_drawdown_date], 'idxmax') else 0
            
            # 找到恢复到前高的日期
            post_drawdown = portfolio_value[max_drawdown_date:]
            recovery_idx = None
            
            for i, val in enumerate(post_drawdown):
                if val >= portfolio_value[pre_peak_idx]:
                    recovery_idx = i
                    break
            
            if recovery_idx is not None:
                recovery_date = post_drawdown.index[recovery_idx] if hasattr(post_drawdown.index, '__getitem__') else max_drawdown_date + recovery_idx
                
                if hasattr(max_drawdown_date, 'strftime') and hasattr(recovery_date, 'strftime'):
                    recovery_time = (recovery_date - max_drawdown_date).days
                else:
                    recovery_time = recovery_idx
        
        # 卡玛比率（Calmar Ratio）
        if max_drawdown < 0:
            calmar_ratio = self._calculate_basic_metrics(returns, initial_capital)['annualized_return'] / abs(max_drawdown)
        else:
            calmar_ratio = float('inf')
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'current_drawdown': current_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'recovery_time': recovery_time,
            'calmar_ratio': calmar_ratio,
            'drawdown_count': len(drawdown_durations),
            'drawdown_durations': drawdown_durations
        }
    
    def _calculate_advanced_metrics(self,
                                   returns: pd.Series,
                                   confidence_level: float) -> Dict:
        """计算高级指标"""
        if len(returns) < 10:
            return {'insufficient_data': True}
        
        # 风险价值（VaR） - 历史模拟法
        var_historical = returns.quantile(1 - confidence_level)
        
        # 条件风险价值（CVaR） - 历史模拟法
        cvar_threshold = returns.quantile(1 - confidence_level)
        cvar_returns = returns[returns <= cvar_threshold]
        cvar_historical = cvar_returns.mean() if len(cvar_returns) > 0 else 0
        
        # Omega比率
        threshold = 0  # 通常使用0或风险免费利率
        positive_returns = returns[returns > threshold].sum()
        negative_returns = abs(returns[returns < threshold].sum())
        
        if negative_returns > 0:
            omega_ratio = positive_returns / negative_returns
        else:
            omega_ratio = float('inf') if positive_returns > 0 else 0
        
        # 上行捕获比率（需要基准，将在基准比较中计算）
        upside_capture_ratio = 0
        
        # 下行捕获比率（需要基准，将在基准比较中计算）
        downside_capture_ratio = 0
        
        # 赢率（Win Rate）
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # 平均赢利/平均亏损比率
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        if avg_loss != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
        else:
            win_loss_ratio = float('inf') if avg_win > 0 else 0
        
        # 盈亏比（Profit Factor）
        gross_profit = winning_returns.sum()
        gross_loss = abs(losing_returns.sum())
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0
        
        # 期望值（Expected Value）
        expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        return {
            'var_historical': var_historical,
            'cvar_historical': cvar_historical,
            'omega_ratio': omega_ratio,
            'upside_capture_ratio': upside_capture_ratio,
            'downside_capture_ratio': downside_capture_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expected_value': expected_value,
            'confidence_level': confidence_level
        }
    
    def _calculate_benchmark_metrics(self,
                                    returns: pd.Series,
                                    benchmark_data: pd.DataFrame) -> Dict:
        """计算基准比较指标"""
        # 确保benchmark_data有收益率列
        if 'return' not in benchmark_data.columns and len(benchmark_data.columns) > 0:
            # 假设第一列是收益率
            benchmark_returns = benchmark_data.iloc[:, 0]
        elif 'return' in benchmark_data.columns:
            benchmark_returns = benchmark_data['return']
        else:
            return {'error': '基准数据格式不正确'}
        
        # 对齐日期（如果有日期索引）
        if isinstance(returns.index, pd.DatetimeIndex) and isinstance(benchmark_returns.index, pd.DatetimeIndex):
            # 找到共同日期
            common_dates = returns.index.intersection(benchmark_returns.index)
            
            if len(common_dates) < 10:
                return {'insufficient_common_data': True}
            
            aligned_returns = returns.loc[common_dates]
            aligned_benchmark = benchmark_returns.loc[common_dates]
        else:
            # 如果没有日期或无法对齐，直接使用相同长度
            min_len = min(len(returns), len(benchmark_returns))
            if min_len < 10:
                return {'insufficient_common_data': True}
            
            aligned_returns = returns.iloc[:min_len]
            aligned_benchmark = benchmark_returns.iloc[:min_len]
        
        # 计算阿尔法（Alpha）和贝塔（Beta）
        if len(aligned_returns) >= 2 and len(aligned_benchmark) >= 2:
            # 使用线性回归计算贝塔
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
            else:
                beta = 0
            
            # 计算阿尔法
            portfolio_return = (1 + aligned_returns).prod() - 1
            benchmark_return = (1 + aligned_benchmark).prod() - 1
            
            # 估算年化周期数
            if isinstance(aligned_returns.index, pd.DatetimeIndex) and len(aligned_returns) > 1:
                days_diff = (aligned_returns.index[-1] - aligned_returns.index[0]).days
                if days_diff > 0:
                    years = days_diff / 365.25
                else:
                    years = 1
            else:
                years = len(aligned_returns) / 252  # 假设每日数据
            
            annualized_portfolio_return = (1 + portfolio_return) ** (1 / years) - 1
            annualized_benchmark_return = (1 + benchmark_return) ** (1 / years) - 1
            
            alpha = annualized_portfolio_return - (self.risk_free_rate + beta * (annualized_benchmark_return - self.risk_free_rate))
            
            # 信息比率
            excess_returns = aligned_returns - aligned_benchmark
            tracking_error = excess_returns.std()
            
            if tracking_error > 0:
                information_ratio = excess_returns.mean() / tracking_error * np.sqrt(252)  # 年化
            else:
                information_ratio = float('inf') if excess_returns.mean() > 0 else 0
            
            # 特雷诺比率
            if beta > 0:
                treynor_ratio = (annualized_portfolio_return - self.risk_free_rate) / beta
            else:
                treynor_ratio = 0
            
            # R平方（R²）
            correlation = np.corrcoef(aligned_returns, aligned_benchmark)[0, 1]
            r_squared = correlation ** 2
            
            # 上行/下行捕获比率
            up_market = aligned_benchmark > 0
            down_market = aligned_benchmark < 0
            
            if up_market.sum() > 0:
                portfolio_up_return = (1 + aligned_returns[up_market]).prod() - 1
                benchmark_up_return = (1 + aligned_benchmark[up_market]).prod() - 1
                
                if benchmark_up_return > 0:
                    upside_capture_ratio = portfolio_up_return / benchmark_up_return
                else:
                    upside_capture_ratio = 0
            else:
                upside_capture_ratio = 0
            
            if down_market.sum() > 0:
                portfolio_down_return = (1 + aligned_returns[down_market]).prod() - 1
                benchmark_down_return = (1 + aligned_benchmark[down_market]).prod() - 1
                
                if benchmark_down_return < 0:
                    downside_capture_ratio = portfolio_down_return / benchmark_down_return
                else:
                    downside_capture_ratio = 0
            else:
                downside_capture_ratio = 0
            
            # 相对收益
            relative_return = portfolio_return - benchmark_return
            
            # 超额收益
            excess_return = relative_return
            
            benchmark_metrics = {
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'treynor_ratio': treynor_ratio,
                'r_squared': r_squared,
                'upside_capture_ratio': upside_capture_ratio,
                'downside_capture_ratio': downside_capture_ratio,
                'relative_return': relative_return,
                'excess_return': excess_return,
                'benchmark_return': benchmark_return,
                'portfolio_return': portfolio_return,
                'tracking_error': tracking_error,
                'correlation': correlation,
                'common_periods': len(aligned_returns)
            }
            
            # 更新风险调整后指标中的值
            if hasattr(self, '_calculate_risk_adjusted_metrics'):
                # 这里我们将在主函数中更新
                pass
            
            # 更新高级指标中的值
            if hasattr(self, '_calculate_advanced_metrics'):
                # 这里我们将在主函数中更新
                pass
            
            return benchmark_metrics
        else:
            return {'insufficient_data': True}
    
    def _calculate_composite_score(self,
                                  basic_metrics: Dict,
                                  risk_adjusted_metrics: Dict,
                                  drawdown_metrics: Dict,
                                  advanced_metrics: Dict) -> Dict:
        """计算综合绩效分数"""
        score_components = {}
        
        # 1. 收益分数（30%）
        annualized_return = basic_metrics.get('annualized_return', 0)
        
        # 收益评分：10%为满分
        if annualized_return >= 0.10:
            return_score = 1.0
        elif annualized_return <= -0.10:
            return_score = 0.0
        else:
            return_score = (annualized_return + 0.10) / 0.20
        
        score_components['return_score'] = max(0.0, min(1.0, return_score))
        
        # 2. 风险调整后收益分数（25%）
        sharpe_ratio = risk_adjusted_metrics.get('sharpe_ratio', 0)
        
        # 夏普比率评分：1.0为及格，2.0为优秀
        if sharpe_ratio >= 2.0:
            sharpe_score = 1.0
        elif sharpe_ratio <= 0:
            sharpe_score = 0.0
        else:
            sharpe_score = sharpe_ratio / 2.0
        
        score_components['risk_adjusted_score'] = max(0.0, min(1.0, sharpe_score))
        
        # 3. 回撤控制分数（20%）
        max_drawdown = abs(drawdown_metrics.get('max_drawdown', 0))
        
        # 最大回撤评分：5%以内为优秀，20%以上为差
        if max_drawdown <= 0.05:
            drawdown_score = 1.0
        elif max_drawdown >= 0.20:
            drawdown_score = 0.0
        else:
            drawdown_score = 1.0 - (max_drawdown - 0.05) / 0.15
        
        score_components['drawdown_score'] = max(0.0, min(1.0, drawdown_score))
        
        # 4. 一致性分数（15%）
        positive_ratio = basic_metrics.get('positive_ratio', 0)
        
        # 正收益比例评分：60%为及格，80%为优秀
        if positive_ratio >= 0.80:
            consistency_score = 1.0
        elif positive_ratio <= 0.40:
            consistency_score = 0.0
        else:
            consistency_score = (positive_ratio - 0.40) / 0.40
        
        score_components['consistency_score'] = max(0.0, min(1.0, consistency_score))
        
        # 5. 高级指标分数（10%）
        profit_factor = advanced_metrics.get('profit_factor', 0)
        
        # 盈亏比评分：2.0为优秀，1.0为及格
        if profit_factor >= 2.0:
            advanced_score = 1.0
        elif profit_factor <= 1.0:
            advanced_score = 0.0
        else:
            advanced_score = (profit_factor - 1.0) / 1.0
        
        score_components['advanced_score'] = max(0.0, min(1.0, advanced_score))
        
        # 计算加权总分
        weights = {
            'return_score': 0.30,
            'risk_adjusted_score': 0.25,
            'drawdown_score': 0.20,
            'consistency_score': 0.15,
            'advanced_score': 0.10
        }
        
        composite_score = sum(score * weights[name] for name, score in score_components.items())
        composite_score = max(0.0, min(1.0, composite_score))
        
        return {
            'composite_score': composite_score,
            'score_components': score_components,
            'weights': weights,
            'interpretation': self._interpret_composite_score(composite_score)
        }
    
    def _interpret_composite_score(self, score: float) -> str:
        """解释综合分数"""
        if score >= 0.9:
            return '卓越：在所有维度表现优秀'
        elif score >= 0.8:
            return '优秀：整体表现优秀，有少量改进空间'
        elif score >= 0.7:
            return '良好：表现良好，有几个方面可以改进'
        elif score >= 0.6:
            return '一般：勉强合格，需要多方面改进'
        elif score >= 0.5:
            return '较差：需要重大改进'
        else:
            return '很差：绩效不合格，需要彻底重新评估'
    
    def _assign_performance_grade(self, composite_score: float) -> str:
        """分配绩效等级"""
        if composite_score >= 0.9:
            return 'A+'
        elif composite_score >= 0.85:
            return 'A'
        elif composite_score >= 0.8:
            return 'A-'
        elif composite_score >= 0.75:
            return 'B+'
        elif composite_score >= 0.7:
            return 'B'
        elif composite_score >= 0.65:
            return 'B-'
        elif composite_score >= 0.6:
            return 'C+'
        elif composite_score >= 0.55:
            return 'C'
        elif composite_score >= 0.5:
            return 'C-'
        elif composite_score >= 0.4:
            return 'D'
        else:
            return 'F'
    
    def _generate_performance_report(self,
                                    basic_metrics: Dict,
                                    risk_adjusted_metrics: Dict,
                                    drawdown_metrics: Dict,
                                    advanced_metrics: Dict,
                                    benchmark_metrics: Dict,
                                    composite_score: Dict) -> Dict:
        """生成绩效报告"""
        report = {
            'executive_summary': {
                'composite_score': composite_score['composite_score'],
                'performance_grade': self._assign_performance_grade(composite_score['composite_score']),
                'interpretation': composite_score['interpretation'],
                'key_strengths': [],
                'key_weaknesses': []
            },
            'detailed_analysis': {
                'return_analysis': {
                    'annualized_return': basic_metrics.get('annualized_return', 0),
                    'total_return': basic_metrics.get('total_return', 0),
                    'mean_return': basic_metrics.get('mean_return', 0),
                    'positive_ratio': basic_metrics.get('positive_ratio', 0)
                },
                'risk_analysis': {
                    'volatility': basic_metrics.get('std_return', 0),
                    'sharpe_ratio': risk_adjusted_metrics.get('sharpe_ratio', 0),
                    'sortino_ratio': risk_adjusted_metrics.get('sortino_ratio', 0),
                    'max_drawdown': drawdown_metrics.get('max_drawdown', 0),
                    'var_historical': advanced_metrics.get('var_historical', 0)
                },
                'consistency_analysis': {
                    'skewness': basic_metrics.get('skewness', 0),
                    'kurtosis': basic_metrics.get('kurtosis', 0),
                    'win_rate': advanced_metrics.get('win_rate', 0),
                    'profit_factor': advanced_metrics.get('profit_factor', 0),
                    'win_loss_ratio': advanced_metrics.get('win_loss_ratio', 0)
                }
            },
            'benchmark_comparison': benchmark_metrics if benchmark_metrics and 'error' not in benchmark_metrics else {},
            'score_breakdown': composite_score['score_components'],
            'recommendations': []
        }
        
        # 识别关键优势
        if basic_metrics.get('annualized_return', 0) > 0.10:
            report['executive_summary']['key_strengths'].append('高年化收益率')
        
        if risk_adjusted_metrics.get('sharpe_ratio', 0) > 1.5:
            report['executive_summary']['key_strengths'].append('优秀的风险调整后收益')
        
        if abs(drawdown_metrics.get('max_drawdown', 0)) < 0.10:
            report['executive_summary']['key_strengths'].append('良好的回撤控制')
        
        if basic_metrics.get('positive_ratio', 0) > 0.60:
            report['executive_summary']['key_strengths'].append('高交易胜率')
        
        # 识别关键弱点
        if basic_metrics.get('annualized_return', 0) < 0:
            report['executive_summary']['key_weaknesses'].append('负收益')
        
        if risk_adjusted_metrics.get('sharpe_ratio', 0) < 0:
            report['executive_summary']['key_weaknesses'].append('负的夏普比率')
        
        if abs(drawdown_metrics.get('max_drawdown', 0)) > 0.20:
            report['executive_summary']['key_weaknesses'].append('过大的最大回撤')
        
        if advanced_metrics.get('profit_factor', 0) < 1.2:
            report['executive_summary']['key_weaknesses'].append('盈亏比较低')
        
        # 生成建议
        if basic_metrics.get('annualized_return', 0) < 0.05:
            report['recommendations'].append({
                'area': '收益提升',
                'priority': 'high',
                'suggestion': '提高收益率，考虑优化交易策略或增加风险暴露',
                'action': 'review_strategy_and_increase_risk_exposure'
            })
        
        if risk_adjusted_metrics.get('sharpe_ratio', 0) < 0.5:
            report['recommendations'].append({
                'area': '风险调整',
                'priority': 'high',
                'suggestion': '提高风险调整后收益，降低波动性或提高收益',
                'action': 'reduce_volatility_or_increase_returns'
            })
        
        if abs(drawdown_metrics.get('max_drawdown', 0)) > 0.15:
            report['recommendations'].append({
                'area': '风险控制',
                'priority': 'high',
                'suggestion': '加强风险控制，降低最大回撤',
                'action': 'improve_risk_management_and_stop_losses'
            })
        
        if advanced_metrics.get('profit_factor', 0) < 1.5:
            report['recommendations'].append({
                'area': '交易效率',
                'priority': 'medium',
                'suggestion': '提高盈亏比，优化止盈止损比例',
                'action': 'optimize_take_profit_and_stop_loss_ratios'
            })
        
        return report
    
    def _generate_improvement_suggestions(self,
                                         basic_metrics: Dict,
                                         risk_adjusted_metrics: Dict,
                                         drawdown_metrics: Dict,
                                         advanced_metrics: Dict) -> List[Dict]:
        """生成改进建议"""
        suggestions = []
        
        # 基于收益的建议
        annualized_return = basic_metrics.get('annualized_return', 0)
        
        if annualized_return < 0.05:
            suggestions.append({
                'type': 'return_improvement',
                'priority': 'high',
                'suggestion': f'年化收益率较低 ({annualized_return:.1%})，建议：',
                'actions': [
                    '优化交易策略入场时机',
                    '提高盈利交易的持仓时间',
                    '减少低概率交易'
                ],
                'metric': 'annualized_return',
                'current_value': annualized_return,
                'target_value': 0.10
            })
        
        # 基于风险调整后收益的建议
        sharpe_ratio = risk_adjusted_metrics.get('sharpe_ratio', 0)
        
        if sharpe_ratio < 0.5:
            suggestions.append({
                'type': 'risk_adjustment',
                'priority': 'high',
                'suggestion': f'夏普比率较低 ({sharpe_ratio:.2f})，建议：',
                'actions': [
                    '降低投资组合波动性',
                    '提高收益的稳定性',
                    '优化资产配置'
                ],
                'metric': 'sharpe_ratio',
                'current_value': sharpe_ratio,
                'target_value': 1.0
            })
        
        # 基于回撤的建议
        max_drawdown = abs(drawdown_metrics.get('max_drawdown', 0))
        
        if max_drawdown > 0.15:
            suggestions.append({
                'type': 'drawdown_control',
                'priority': 'high',
                'suggestion': f'最大回撤较大 ({max_drawdown:.1%})，建议：',
                'actions': [
                    '加强止损纪律',
                    '降低仓位集中度',
                    '增加对冲策略'
                ],
                'metric': 'max_drawdown',
                'current_value': max_drawdown,
                'target_value': 0.10
            })
        
        # 基于一致性的建议
        positive_ratio = basic_metrics.get('positive_ratio', 0)
        
        if positive_ratio < 0.50:
            suggestions.append({
                'type': 'consistency_improvement',
                'priority': 'medium',
                'suggestion': f'正收益比例较低 ({positive_ratio:.0%})，建议：',
                'actions': [
                    '提高交易胜率',
                    '等待更高概率的交易机会',
                    '减少情绪化交易'
                ],
                'metric': 'positive_ratio',
                'current_value': positive_ratio,
                'target_value': 0.60
            })
        
        # 基于盈亏比的建议
        profit_factor = advanced_metrics.get('profit_factor', 0)
        
        if profit_factor < 1.5:
            suggestions.append({
                'type': 'profit_factor_improvement',
                'priority': 'medium',
                'suggestion': f'盈亏比较低 ({profit_factor:.2f})，建议：',
                'actions': [
                    '让盈利交易充分发展',
                    '及时止损亏损交易',
                    '优化风险回报比例'
                ],
                'metric': 'profit_factor',
                'current_value': profit_factor,
                'target_value': 2.0
            })
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order[x['priority']])
        
        return suggestions
    
    def get_evaluation_history_summary(self) -> Dict:
        """获取评估历史摘要"""
        if not self.evaluation_history:
            return {'total_evaluations': 0}
        
        recent_evaluations = self.evaluation_history[-5:] if len(self.evaluation_history) > 5 else self.evaluation_history
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'recent_evaluations': len(recent_evaluations),
            'average_composite_score': np.mean([e['composite_score']['composite_score'] for e in recent_evaluations]) if recent_evaluations else 0,
            'grade_distribution': {},
            'performance_trend': 'insufficient_data'
        }
        
        # 等级分布
        for eval_result in recent_evaluations:
            grade = eval_result['performance_grade']
            summary['grade_distribution'][grade] = summary['grade_distribution'].get(grade, 0) + 1
        
        # 绩效趋势
        if len(recent_evaluations) >= 3:
            recent_scores = [e['composite_score']['composite_score'] for e in recent_evaluations]
            earlier_scores = [e['composite_score']['composite_score'] for e in self.evaluation_history[:-3]] if len(self.evaluation_history) > 3 else recent_scores
            
            avg_recent = np.mean(recent_scores)
            avg_earlier = np.mean(earlier_scores) if earlier_scores else avg_recent
            
            if avg_recent > avg_earlier + 0.05:
                summary['performance_trend'] = 'improving'
            elif avg_recent < avg_earlier - 0.05:
                summary['performance_trend'] = 'declining'
            else:
                summary['performance_trend'] = 'stable'
        
        return summary
    
    def export_evaluation_to_markdown(self, evaluation_id: str) -> str:
        """将评估导出为Markdown格式"""
        evaluation = None
        for eval_result in self.evaluation_history:
            if eval_result.get('evaluation_id') == evaluation_id:
                evaluation = eval_result
                break
        
        if not evaluation:
            return f"# 评估未找到: {evaluation_id}"
        
        report = evaluation['performance_report']
        composite_score = evaluation['composite_score']
        
        markdown = f"""# 绩效评估报告

## 评估信息
- **评估ID**: {evaluation['evaluation_id']}
- **评估时间**: {evaluation['evaluation_time']}
- **绩效等级**: {evaluation['performance_grade']}
- **综合分数**: {composite_score['composite_score']:.0%}
- **绩效评价**: {composite_score['interpretation']}

## 执行摘要
- **综合分数**: {composite_score['composite_score']:.0%}
- **绩效等级**: {evaluation['performance_grade']}
- **关键优势**:
"""
        
        for strength in report['executive_summary']['key_strengths']:
            markdown += f"  - {strength}\n"
        
        markdown += "\n- **关键弱点**:\n"
        
        for weakness in report['executive_summary']['key_weaknesses']:
            markdown += f"  - {weakness}\n"
        
        markdown += f"""
## 详细分析

### 收益分析
- **年化收益率**: {report['detailed_analysis']['return_analysis']['annualized_return']:.1%}
- **总收益率**: {report['detailed_analysis']['return_analysis']['total_return']:.1%}
- **平均收益率**: {report['detailed_analysis']['return_analysis']['mean_return']:.1%}
- **正收益比例**: {report['detailed_analysis']['return_analysis']['positive_ratio']:.0%}

### 风险分析
- **波动率**: {report['detailed_analysis']['risk_analysis']['volatility']:.1%}
- **夏普比率**: {report['detailed_analysis']['risk_analysis']['sharpe_ratio']:.2f}
- **索提诺比率**: {report['detailed_analysis']['risk_analysis']['sortino_ratio']:.2f}
- **最大回撤**: {report['detailed_analysis']['risk_analysis']['max_drawdown']:.1%}
- **风险价值(VaR)**: {report['detailed_analysis']['risk_analysis']['var_historical']:.1%}

### 一致性分析
- **偏度**: {report['detailed_analysis']['consistency_analysis']['skewness']:.2f}
- **峰度**: {report['detailed_analysis']['consistency_analysis']['kurtosis']:.2f}
- **胜率**: {report['detailed_analysis']['consistency_analysis']['win_rate']:.0%}
- **盈亏比**: {report['detailed_analysis']['consistency_analysis']['profit_factor']:.2f}
- **赢亏比**: {report['detailed_analysis']['consistency_analysis']['win_loss_ratio']:.2f}

## 分数分解
"""
        
        for name, score in composite_score['score_components'].items():
            weight = composite_score['weights'].get(name, 0)
            markdown += f"- **{name}**: {score:.0%} (权重: {weight:.0%})\n"
        
        markdown += f"\n## 改进建议\n"
        
        for i, suggestion in enumerate(evaluation['improvement_suggestions'][:5], 1):
            markdown += f"{i}. **{suggestion['priority'].upper()}**: {suggestion['suggestion']}\n"
            for action in suggestion.get('actions', [])[:3]:
                markdown += f"   - {action}\n"
        
        markdown += f"\n---\n*报告生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return markdown
    
    def set_benchmark_data(self, benchmark_data: pd.DataFrame):
        """设置基准数据"""
        self.benchmark_data = benchmark_data
    
    def set_risk_free_rate(self, risk_free_rate: float):
        """设置无风险利率"""
        self.risk_free_rate = risk_free_rate


# ============================================================================
# 策略改造: 添加PriceActionRangesPerformanceEvaluatorStrategy类
# 将价格行为区间绩效评估系统转换为交易策略
# ============================================================================

class PriceActionRangesPerformanceEvaluatorStrategy(BaseStrategy):
    """价格行为区间绩效评估策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 创建绩效评估器实例
        self.evaluator = PerformanceEvaluator()
    
    def generate_signals(self):
        """
        生成交易信号

        基于绩效评估生成交易信号，使用动量和趋势质量评估
        """
        df = self.data
        if len(df) < 20:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(df['close'].iloc[-1]))
            return self.signals

        close = df['close']
        returns = close.pct_change().dropna()

        # Performance metrics
        total_return = (close.iloc[-1] / close.iloc[0]) - 1

        if len(returns) > 1:
            mean_ret = returns.mean()
            std_ret = returns.std()
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Trend
        ma_short = close.rolling(5).mean()
        ma_long = close.rolling(20).mean()
        trend_up = ma_short.iloc[-1] > ma_long.iloc[-1]

        # Composite score
        score = 0
        if total_return > 0:
            score += 1
        if sharpe > 0.5:
            score += 1
        if max_drawdown > -0.1:
            score += 1
        if win_rate > 0.5:
            score += 1
        if trend_up:
            score += 1

        if score >= 4:
            self._record_signal(timestamp=df.index[-1], action='buy', price=float(close.iloc[-1]))
        elif score >= 2:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(close.iloc[-1]))
        else:
            self._record_signal(timestamp=df.index[-1], action='sell', price=float(close.iloc[-1]))

        return self.signals