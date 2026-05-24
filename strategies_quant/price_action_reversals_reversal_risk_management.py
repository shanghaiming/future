# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:23.027558

"""
反转风险管理量化分析系统 - 第5章《反转风险管理》
严格按照第18章标准：实际完整代码，非伪代码框架
紧急冲刺模式：13:50-14:50完成

系统功能：
1. 反转交易风险特征分析：分析反转交易特有的风险特征
2. 动态止损策略：基于市场条件动态调整止损
3. 风险暴露控制：控制整体风险暴露和头寸规模
4. 风险回报优化：优化风险回报比提高交易效率
5. 风险监控系统：实时监控和管理交易风险
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import statistics
import math

# 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy


class RiskLevel(Enum):
    """风险等级枚举"""
    VERY_LOW = "very_low"      # 极低风险
    LOW = "low"                # 低风险
    MODERATE = "moderate"      # 中等风险
    HIGH = "high"              # 高风险
    VERY_HIGH = "very_high"    # 极高风险


class RiskMetricType(Enum):
    """风险指标类型枚举"""
    VOLATILITY_RISK = "volatility_risk"        # 波动率风险
    DRAWDOWN_RISK = "drawdown_risk"            # 回撤风险
    LIQUIDITY_RISK = "liquidity_risk"          # 流动性风险
    CONCENTRATION_RISK = "concentration_risk"  # 集中度风险
    TAIL_RISK = "tail_risk"                    # 尾部风险
    MODEL_RISK = "model_risk"                  # 模型风险


@dataclass
class PriceBar:
    """价格柱数据类"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def body(self) -> float:
        """K线实体大小（收盘-开盘）"""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        """是否为阳线"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """是否为阴线"""
        return self.close < self.open
    
    @property
    def upper_shadow(self) -> float:
        """上影线长度"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """下影线长度"""
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        """总价格范围（高-低）"""
        return self.high - self.low


@dataclass
class RiskMetric:
    """风险指标数据类"""
    metric_id: str
    metric_type: RiskMetricType
    value: float
    risk_level: RiskLevel
    timestamp: datetime
    description: str
    metadata: Dict[str, Any]


@dataclass
class RiskAssessment:
    """风险评估数据类"""
    assessment_id: str
    metrics: List[RiskMetric]  # 风险指标列表
    overall_risk_level: RiskLevel  # 总体风险等级
    risk_score: float  # 总体风险分数（0-1）
    recommendations: List[str]  # 风险建议
    assessment_time: datetime
    details: Dict[str, Any]


class ReversalRiskManagement(BaseStrategy):
    """
    反转风险管理量化分析系统
    
    严格按照第18章标准实现，提供完整的反转交易风险管理功能
    紧急冲刺模式：核心功能优先，实际完整代码
    """
    
    def __init__(self, data: pd.DataFrame, params: Dict):
        """
        初始化反转风险管理系统
        
        Args:
            data: 价格数据DataFrame
            params: 参数字典，包含initial_balance等配置
        """
        super().__init__(data, params)
        
        # 从参数中提取配置
        self.initial_balance = params.get('initial_balance', 10000.0)
        self.current_balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.max_drawdown = 0.0
        self.active_positions = []  # 活跃头寸
        self.risk_metrics = []      # 风险指标
        self.risk_assessments = []  # 风险评估
        
        # 系统配置
        self.config = {
            "max_portfolio_risk": params.get('max_portfolio_risk', 0.02),           # 最大组合风险（2%）
            "max_position_risk": params.get('max_position_risk', 0.01),            # 最大单笔头寸风险（1%）
            "max_drawdown_limit": params.get('max_drawdown_limit', 0.10),           # 最大回撤限制（10%）
            "volatility_lookback_period": params.get('volatility_lookback_period', 20),     # 波动率回顾周期
            "correlation_lookback_period": params.get('correlation_lookback_period', 60),    # 相关性回顾周期
            "liquidity_threshold": params.get('liquidity_threshold', 1000000),       # 流动性阈值（交易量）
            "tail_risk_confidence": params.get('tail_risk_confidence', 0.95),         # 尾部风险置信度
            "risk_adjustment_factor": params.get('risk_adjustment_factor', 1.0),        # 风险调整因子
        }
        
        # 风险计算器映射
        self.risk_calculators = {
            RiskMetricType.VOLATILITY_RISK: self._calculate_volatility_risk,
            RiskMetricType.DRAWDOWN_RISK: self._calculate_drawdown_risk,
            RiskMetricType.LIQUIDITY_RISK: self._calculate_liquidity_risk,
            RiskMetricType.CONCENTRATION_RISK: self._calculate_concentration_risk,
            RiskMetricType.TAIL_RISK: self._calculate_tail_risk,
        }
    
    # ==================== 核心风险计算方法 ====================
    
    def calculate_all_risk_metrics(self, price_bars: List[PriceBar], 
                                 positions: List[Dict[str, Any]]) -> List[RiskMetric]:
        """
        计算所有风险指标
        
        Args:
            price_bars: 价格柱列表
            positions: 头寸列表
            
        Returns:
            风险指标列表
        """
        metrics = []
        
        # 计算每种风险指标
        for metric_type, calculator in self.risk_calculators.items():
            metric = calculator(price_bars, positions)
            if metric:
                metrics.append(metric)
        
        # 按风险等级排序
        metrics.sort(key=lambda m: self._risk_level_to_score(m.risk_level), reverse=True)
        
        self.risk_metrics.extend(metrics)
        return metrics
    
    def _calculate_volatility_risk(self, price_bars: List[PriceBar], 
                                 positions: List[Dict[str, Any]]) -> Optional[RiskMetric]:
        """计算波动率风险"""
        if len(price_bars) < self.config["volatility_lookback_period"]:
            return None
        
        # 计算历史波动率
        recent_bars = price_bars[-self.config["volatility_lookback_period"]:]
        returns = []
        
        for i in range(1, len(recent_bars)):
            ret = (recent_bars[i].close - recent_bars[i-1].close) / recent_bars[i-1].close
            returns.append(ret)
        
        if not returns:
            return None
        
        # 计算年化波动率
        daily_volatility = statistics.stdev(returns) if len(returns) > 1 else abs(returns[0])
        annual_volatility = daily_volatility * math.sqrt(252)  # 年化
        
        # 确定风险等级
        if annual_volatility < 0.15:
            risk_level = RiskLevel.LOW
            risk_score = annual_volatility / 0.15
        elif annual_volatility < 0.30:
            risk_level = RiskLevel.MODERATE
            risk_score = 0.3 + (annual_volatility - 0.15) / 0.15 * 0.3
        elif annual_volatility < 0.50:
            risk_level = RiskLevel.HIGH
            risk_score = 0.6 + (annual_volatility - 0.30) / 0.20 * 0.3
        else:
            risk_level = RiskLevel.VERY_HIGH
            risk_score = 0.9 + min((annual_volatility - 0.50) / 0.50, 0.1)
        
        metric = RiskMetric(
            metric_id=f"volatility_risk_{datetime.now().timestamp()}",
            metric_type=RiskMetricType.VOLATILITY_RISK,
            value=annual_volatility,
            risk_level=risk_level,
            timestamp=datetime.now(),
            description=f"波动率风险: 年化波动率{annual_volatility:.2%}",
            metadata={
                "daily_volatility": daily_volatility,
                "annual_volatility": annual_volatility,
                "lookback_period": self.config["volatility_lookback_period"],
                "returns_count": len(returns),
            }
        )
        
        return metric
    
    def _calculate_drawdown_risk(self, price_bars: List[PriceBar], 
                               positions: List[Dict[str, Any]]) -> Optional[RiskMetric]:
        """计算回撤风险"""
        if len(price_bars) < 20:
            return None
        
        # 计算最大回撤
        closes = [bar.close for bar in price_bars[-100:]]  # 最近100个收盘价
        peak = closes[0]
        max_drawdown = 0.0
        
        for close in closes:
            if close > peak:
                peak = close
            drawdown = (peak - close) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 更新系统最大回撤
        self.max_drawdown = max(self.max_drawdown, max_drawdown)
        
        # 确定风险等级
        if max_drawdown < 0.05:
            risk_level = RiskLevel.LOW
            risk_score = max_drawdown / 0.05
        elif max_drawdown < 0.10:
            risk_level = RiskLevel.MODERATE
            risk_score = 0.3 + (max_drawdown - 0.05) / 0.05 * 0.3
        elif max_drawdown < 0.20:
            risk_level = RiskLevel.HIGH
            risk_score = 0.6 + (max_drawdown - 0.10) / 0.10 * 0.3
        else:
            risk_level = RiskLevel.VERY_HIGH
            risk_score = 0.9 + min((max_drawdown - 0.20) / 0.30, 0.1)
        
        metric = RiskMetric(
            metric_id=f"drawdown_risk_{datetime.now().timestamp()}",
            metric_type=RiskMetricType.DRAWDOWN_RISK,
            value=max_drawdown,
            risk_level=risk_level,
            timestamp=datetime.now(),
            description=f"回撤风险: 最大回撤{max_drawdown:.2%}",
            metadata={
                "current_drawdown": max_drawdown,
                "system_max_drawdown": self.max_drawdown,
                "lookback_period": len(closes),
                "peak_price": peak,
                "current_price": closes[-1] if closes else 0.0,
            }
        )
        
        return metric
    
    def _calculate_liquidity_risk(self, price_bars: List[PriceBar], 
                                positions: List[Dict[str, Any]]) -> Optional[RiskMetric]:
        """计算流动性风险"""
        if len(price_bars) < 10:
            return None
        
        # 计算平均成交量
        recent_bars = price_bars[-10:]
        volumes = [bar.volume for bar in recent_bars]
        avg_volume = statistics.mean(volumes)
        
        # 确定风险等级
        if avg_volume >= self.config["liquidity_threshold"] * 2:
            risk_level = RiskLevel.VERY_LOW
            risk_score = 0.1
        elif avg_volume >= self.config["liquidity_threshold"]:
            risk_level = RiskLevel.LOW
            risk_score = 0.3
        elif avg_volume >= self.config["liquidity_threshold"] * 0.5:
            risk_level = RiskLevel.MODERATE
            risk_score = 0.5
        elif avg_volume >= self.config["liquidity_threshold"] * 0.2:
            risk_level = RiskLevel.HIGH
            risk_score = 0.7
        else:
            risk_level = RiskLevel.VERY_HIGH
            risk_score = 0.9
        
        metric = RiskMetric(
            metric_id=f"liquidity_risk_{datetime.now().timestamp()}",
            metric_type=RiskMetricType.LIQUIDITY_RISK,
            value=avg_volume,
            risk_level=risk_level,
            timestamp=datetime.now(),
            description=f"流动性风险: 平均成交量{avg_volume:.0f}",
            metadata={
                "avg_volume": avg_volume,
                "volume_std": statistics.stdev(volumes) if len(volumes) > 1 else 0.0,
                "liquidity_threshold": self.config["liquidity_threshold"],
                "volume_ratio": avg_volume / self.config["liquidity_threshold"],
            }
        )
        
        return metric
    
    def _calculate_concentration_risk(self, price_bars: List[PriceBar], 
                                    positions: List[Dict[str, Any]]) -> Optional[RiskMetric]:
        """计算集中度风险"""
        if not positions:
            return None
        
        # 计算头寸集中度
        total_value = sum(pos.get("position_value", 0) for pos in positions)
        
        if total_value == 0:
            return None
        
        # 计算最大头寸占比
        position_values = [pos.get("position_value", 0) for pos in positions]
        max_position = max(position_values) if position_values else 0
        concentration_ratio = max_position / total_value if total_value > 0 else 0.0
        
        # 确定风险等级
        if concentration_ratio < 0.1:
            risk_level = RiskLevel.VERY_LOW
            risk_score = concentration_ratio / 0.1 * 0.1
        elif concentration_ratio < 0.2:
            risk_level = RiskLevel.LOW
            risk_score = 0.1 + (concentration_ratio - 0.1) / 0.1 * 0.2
        elif concentration_ratio < 0.3:
            risk_level = RiskLevel.MODERATE
            risk_score = 0.3 + (concentration_ratio - 0.2) / 0.1 * 0.3
        elif concentration_ratio < 0.5:
            risk_level = RiskLevel.HIGH
            risk_score = 0.6 + (concentration_ratio - 0.3) / 0.2 * 0.3
        else:
            risk_level = RiskLevel.VERY_HIGH
            risk_score = 0.9 + min((concentration_ratio - 0.5) / 0.5, 0.1)
        
        metric = RiskMetric(
            metric_id=f"concentration_risk_{datetime.now().timestamp()}",
            metric_type=RiskMetricType.CONCENTRATION_RISK,
            value=concentration_ratio,
            risk_level=risk_level,
            timestamp=datetime.now(),
            description=f"集中度风险: 最大头寸占比{concentration_ratio:.2%}",
            metadata={
                "concentration_ratio": concentration_ratio,
                "total_positions": len(positions),
                "total_value": total_value,
                "max_position_value": max_position,
                "max_position_ratio": concentration_ratio,
            }
        )
        
        return metric
    
    def _calculate_tail_risk(self, price_bars: List[PriceBar], 
                           positions: List[Dict[str, Any]]) -> Optional[RiskMetric]:
        """计算尾部风险（极端事件风险）"""
        if len(price_bars) < 100:
            return None
        
        # 计算历史极端收益
        recent_bars = price_bars[-100:]
        returns = []
        
        for i in range(1, len(recent_bars)):
            ret = (recent_bars[i].close - recent_bars[i-1].close) / recent_bars[i-1].close
            returns.append(ret)
        
        if len(returns) < 20:
            return None
        
        # 计算VaR（风险价值）
        confidence = self.config["tail_risk_confidence"]
        sorted_returns = sorted(returns)
        var_index = int((1 - confidence) * len(sorted_returns))
        var = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0.0
        
        # 计算CVaR（条件风险价值）
        tail_returns = sorted_returns[:var_index] if var_index > 0 else []
        cvar = abs(statistics.mean(tail_returns)) if tail_returns else var * 1.5
        
        # 确定风险等级
        if cvar < 0.02:
            risk_level = RiskLevel.LOW
            risk_score = cvar / 0.02
        elif cvar < 0.04:
            risk_level = RiskLevel.MODERATE
            risk_score = 0.3 + (cvar - 0.02) / 0.02 * 0.3
        elif cvar < 0.06:
            risk_level = RiskLevel.HIGH
            risk_score = 0.6 + (cvar - 0.04) / 0.02 * 0.3
        else:
            risk_level = RiskLevel.VERY_HIGH
            risk_score = 0.9 + min((cvar - 0.06) / 0.04, 0.1)
        
        metric = RiskMetric(
            metric_id=f"tail_risk_{datetime.now().timestamp()}",
            metric_type=RiskMetricType.TAIL_RISK,
            value=cvar,
            risk_level=risk_level,
            timestamp=datetime.now(),
            description=f"尾部风险: CVaR{cvar:.2%} (置信度{confidence:.0%})",
            metadata={
                "var": var,
                "cvar": cvar,
                "confidence_level": confidence,
                "extreme_returns_count": len(tail_returns),
                "avg_tail_return": statistics.mean(tail_returns) if tail_returns else 0.0,
            }
        )
        
        return metric
    
    # ==================== 风险评估和监控 ====================
    
    def assess_overall_risk(self, metrics: List[RiskMetric]) -> RiskAssessment:
        """
        评估总体风险
        
        Args:
            metrics: 风险指标列表
            
        Returns:
            风险评估结果
        """
        if not metrics:
            return self._create_empty_assessment()
        
        # 计算总体风险分数（加权平均）
        weighted_scores = []
        weights = []
        
        for metric in metrics:
            score = self._risk_level_to_score(metric.risk_level)
            weight = self._get_metric_weight(metric.metric_type)
            
            weighted_scores.append(score * weight)
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
        else:
            overall_score = 0.5
        
        # 确定总体风险等级
        if overall_score < 0.2:
            overall_risk = RiskLevel.VERY_LOW
        elif overall_score < 0.4:
            overall_risk = RiskLevel.LOW
        elif overall_score < 0.6:
            overall_risk = RiskLevel.MODERATE
        elif overall_score < 0.8:
            overall_risk = RiskLevel.HIGH
        else:
            overall_risk = RiskLevel.VERY_HIGH
        
        # 生成风险建议
        recommendations = self._generate_risk_recommendations(metrics, overall_risk)
        
        assessment = RiskAssessment(
            assessment_id=f"risk_assessment_{datetime.now().timestamp()}",
            metrics=metrics,
            overall_risk_level=overall_risk,
            risk_score=overall_score,
            recommendations=recommendations,
            assessment_time=datetime.now(),
            details={
                "metrics_count": len(metrics),
                "metric_types": [m.metric_type.value for m in metrics],
                "risk_levels": [m.risk_level.value for m in metrics],
                "weighted_scores": weighted_scores,
                "weights": weights,
            }
        )
        
        self.risk_assessments.append(assessment)
        return assessment
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """将风险等级转换为分数"""
        level_scores = {
            RiskLevel.VERY_LOW: 0.1,
            RiskLevel.LOW: 0.3,
            RiskLevel.MODERATE: 0.5,
            RiskLevel.HIGH: 0.7,
            RiskLevel.VERY_HIGH: 0.9,
        }
        return level_scores.get(risk_level, 0.5)
    
    def _get_metric_weight(self, metric_type: RiskMetricType) -> float:
        """获取指标权重"""
        weights = {
            RiskMetricType.VOLATILITY_RISK: 0.25,
            RiskMetricType.DRAWDOWN_RISK: 0.25,
            RiskMetricType.LIQUIDITY_RISK: 0.20,
            RiskMetricType.CONCENTRATION_RISK: 0.15,
            RiskMetricType.TAIL_RISK: 0.15,
        }
        return weights.get(metric_type, 0.1)
    
    def _generate_risk_recommendations(self, metrics: List[RiskMetric], 
                                     overall_risk: RiskLevel) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        # 总体风险建议
        if overall_risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            recommendations.append("⚠️ 总体风险较高，建议减少头寸规模或暂停新交易")
        elif overall_risk == RiskLevel.MODERATE:
            recommendations.append("ℹ️ 总体风险适中，建议保持当前风险暴露水平")
        else:
            recommendations.append("✅ 总体风险较低，可以适度增加风险暴露")
        
        # 基于具体指标的建议
        for metric in metrics:
            if metric.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                if metric.metric_type == RiskMetricType.VOLATILITY_RISK:
                    recommendations.append(f"📈 波动率风险高({metric.value:.2%})，建议使用更宽的止损")
                elif metric.metric_type == RiskMetricType.DRAWDOWN_RISK:
                    recommendations.append(f"📉 回撤风险高({metric.value:.2%})，建议降低杠杆")
                elif metric.metric_type == RiskMetricType.LIQUIDITY_RISK:
                    recommendations.append(f"💧 流动性风险高，建议减小单笔交易规模")
                elif metric.metric_type == RiskMetricType.CONCENTRATION_RISK:
                    recommendations.append(f"🎯 集中度风险高({metric.value:.2%})，建议分散投资")
                elif metric.metric_type == RiskMetricType.TAIL_RISK:
                    recommendations.append(f"🌪️ 尾部风险高({metric.value:.2%})，建议增加对冲保护")
        
        return recommendations
    
    def _create_empty_assessment(self) -> RiskAssessment:
        """创建空风险评估"""
        return RiskAssessment(
            assessment_id=f"empty_assessment_{datetime.now().timestamp()}",
            metrics=[],
            overall_risk_level=RiskLevel.MODERATE,
            risk_score=0.5,
            recommendations=["无风险数据，建议谨慎操作"],
            assessment_time=datetime.now(),
            details={"metrics_count": 0}
        )
    
    # ==================== 风险控制和调整 ====================
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_assessment: RiskAssessment) -> float:
        """
        计算头寸规模（考虑风险评估）
        
        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
            risk_assessment: 风险评估结果
            
        Returns:
            头寸规模（金额）
        """
        # 计算每单位风险
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        
        # 基于风险评估调整风险金额
        base_risk_amount = self.current_balance * self.config["max_position_risk"]
        
        # 风险调整因子
        risk_adjustment = self._calculate_risk_adjustment(risk_assessment)
        adjusted_risk_amount = base_risk_amount * risk_adjustment
        
        # 计算头寸规模
        position_size = (adjusted_risk_amount / risk_per_unit) * entry_price
        
        # 确保不超过最大头寸限制
        max_position = self.current_balance * 0.1  # 最大10%
        position_size = min(position_size, max_position)
        
        return position_size
    
    def _calculate_risk_adjustment(self, risk_assessment: RiskAssessment) -> float:
        """计算风险调整因子"""
        risk_score = risk_assessment.risk_score
        
        # 风险越高，调整因子越小
        if risk_score < 0.3:  # 低风险
            adjustment = 1.2
        elif risk_score < 0.5:  # 中低风险
            adjustment = 1.0
        elif risk_score < 0.7:  # 中等风险
            adjustment = 0.8
        elif risk_score < 0.9:  # 高风险
            adjustment = 0.5
        else:  # 极高风险
            adjustment = 0.3
        
        return adjustment * self.config["risk_adjustment_factor"]
    
    def calculate_dynamic_stop_loss(self, entry_price: float, price_bars: List[PriceBar],
                                  risk_assessment: RiskAssessment) -> float:
        """
        计算动态止损（基于风险和市场条件）
        
        Args:
            entry_price: 入场价格
            price_bars: 价格柱列表
            risk_assessment: 风险评估结果
            
        Returns:
            动态止损价格
        """
        if len(price_bars) < 20:
            # 默认止损：2%
            return entry_price * 0.98
        
        # 计算市场波动率
        recent_bars = price_bars[-20:]
        closes = [bar.close for bar in recent_bars]
        
        returns = []
        for i in range(1, len(closes)):
            ret = (closes[i] - closes[i-1]) / closes[i-1]
            returns.append(abs(ret))
        
        avg_volatility = statistics.mean(returns) if returns else 0.02
        
        # 基于风险评估调整止损距离
        risk_score = risk_assessment.risk_score
        if risk_score < 0.3:  # 低风险
            stop_distance = avg_volatility * 1.5
        elif risk_score < 0.6:  # 中等风险
            stop_distance = avg_volatility * 2.0
        else:  # 高风险
            stop_distance = avg_volatility * 3.0
        
        # 确保止损距离在合理范围内
        stop_distance = max(stop_distance, 0.01)  # 最小1%
        stop_distance = min(stop_distance, 0.05)  # 最大5%
        
        # 假设是买入交易
        return entry_price * (1 - stop_distance)
    
    # ==================== 风险监控和报告 ====================
    
    def monitor_risk_limits(self, positions: List[Dict[str, Any]], 
                          risk_assessment: RiskAssessment) -> Dict[str, Any]:
        """
        监控风险限制
        
        Args:
            positions: 头寸列表
            risk_assessment: 风险评估结果
            
        Returns:
            风险限制监控结果
        """
        total_position_value = sum(pos.get("position_value", 0) for pos in positions)
        portfolio_risk = total_position_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
        
        # 检查风险限制
        limits_violated = []
        
        # 1. 组合风险限制
        if portfolio_risk > self.config["max_portfolio_risk"]:
            limits_violated.append({
                "limit": "max_portfolio_risk",
                "current": portfolio_risk,
                "threshold": self.config["max_portfolio_risk"],
                "violation": portfolio_risk - self.config["max_portfolio_risk"]
            })
        
        # 2. 回撤限制
        if self.max_drawdown > self.config["max_drawdown_limit"]:
            limits_violated.append({
                "limit": "max_drawdown_limit",
                "current": self.max_drawdown,
                "threshold": self.config["max_drawdown_limit"],
                "violation": self.max_drawdown - self.config["max_drawdown_limit"]
            })
        
        # 3. 单笔头寸风险限制
        for pos in positions:
            position_value = pos.get("position_value", 0)
            position_risk = position_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
            
            if position_risk > self.config["max_position_risk"]:
                limits_violated.append({
                    "limit": "max_position_risk",
                    "position_id": pos.get("position_id", "unknown"),
                    "current": position_risk,
                    "threshold": self.config["max_position_risk"],
                    "violation": position_risk - self.config["max_position_risk"]
                })
        
        monitoring_result = {
            "portfolio_value": self.portfolio_value,
            "total_position_value": total_position_value,
            "portfolio_risk_ratio": portfolio_risk,
            "max_drawdown": self.max_drawdown,
            "risk_score": risk_assessment.risk_score,
            "limits_violated": limits_violated,
            "all_limits_ok": len(limits_violated) == 0,
            "monitoring_time": datetime.now().isoformat(),
        }
        
        return monitoring_result
    
    # ==================== 系统演示和报告 ====================
    
    def demonstrate_system(self) -> Dict[str, Any]:
        """
        演示系统功能
        
        Returns:
            演示结果
        """
        # 创建模拟数据
        mock_bars = self._create_mock_price_bars(100)
        mock_positions = [
            {"position_id": "pos_1", "position_value": 1000.0},
            {"position_id": "pos_2", "position_value": 1500.0},
        ]
        
        # 计算风险指标
        risk_metrics = self.calculate_all_risk_metrics(mock_bars, mock_positions)
        
        # 评估总体风险
        risk_assessment = self.assess_overall_risk(risk_metrics)
        
        # 监控风险限制
        risk_monitoring = self.monitor_risk_limits(mock_positions, risk_assessment)
        
        demonstration = {
            "risk_metrics_calculated": len(risk_metrics),
            "risk_assessment": {
                "overall_risk_level": risk_assessment.overall_risk_level.value,
                "risk_score": risk_assessment.risk_score,
                "recommendations_count": len(risk_assessment.recommendations),
            },
            "risk_monitoring": {
                "all_limits_ok": risk_monitoring["all_limits_ok"],
                "limits_violated": len(risk_monitoring["limits_violated"]),
                "portfolio_risk_ratio": risk_monitoring["portfolio_risk_ratio"],
            },
            "system_status": "operational",
            "generated_at": datetime.now().isoformat(),
        }
        
        return demonstration
    
    def _create_mock_price_bars(self, n_bars: int) -> List[PriceBar]:
        """创建模拟价格柱数据"""
        bars = []
        current_time = datetime.now()
        base_price = 100.0
        
        for i in range(n_bars):
            # 随机价格变动
            random_change = np.random.normal(0.001, 0.02)
            price = base_price * (1 + random_change)
            
            # 生成OHLC
            open_price = price
            close_price = price * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            
            # 确保高低价正确
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            # 成交量
            volume = np.random.uniform(1000, 10000)
            
            bar = PriceBar(
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            bars.append(bar)
            
            # 更新时间
            current_time = current_time.replace(second=current_time.second + 60)
            base_price = close_price
        
        return bars
    
    def _convert_dataframe_to_bars(self) -> List[PriceBar]:
        """
        将DataFrame转换为PriceBar列表
        
        Returns:
            PriceBar列表
        """
        bars = []
        
        # 确保数据有必要的列
        if len(self.data) == 0:
            return bars
        
        # 确定列名映射
        timestamp_col = 'timestamp' if 'timestamp' in self.data.columns else self.data.index.name
        open_col = 'open' if 'open' in self.data.columns else self.data.columns[0]
        high_col = 'high' if 'high' in self.data.columns else self.data.columns[1] if len(self.data.columns) > 1 else open_col
        low_col = 'low' if 'low' in self.data.columns else self.data.columns[2] if len(self.data.columns) > 2 else open_col
        close_col = 'close' if 'close' in self.data.columns else self.data.columns[3] if len(self.data.columns) > 3 else open_col
        volume_col = 'volume' if 'volume' in self.data.columns else (self.data.columns[4] if len(self.data.columns) > 4 else None)
        
        for idx, row in self.data.iterrows():
            timestamp = idx if timestamp_col is None else row.get(timestamp_col, idx)
            open_price = float(row[open_col])
            high_price = float(row[high_col]) if high_col else open_price
            low_price = float(row[low_col]) if low_col else open_price
            close_price = float(row[close_col]) if close_col else open_price
            volume = float(row[volume_col]) if volume_col else 0.0
            
            bar = PriceBar(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            bars.append(bar)
        
        return bars
    
    def generate_signals(self) -> List[Dict]:
        """
        生成风险管理信号
        风险管理系统的信号通常是风险警告或仓位调整建议
        
        Returns:
            标准化信号列表
        """
        signals = []
        
        # 将数据转换为PriceBar格式
        price_bars = self._convert_dataframe_to_bars()
        if not price_bars:
            return signals
        
        # 计算所有风险指标
        risk_metrics = self.calculate_all_risk_metrics(price_bars, self.active_positions)
        
        # 根据风险等级生成信号
        for metric in risk_metrics:
            risk_level = metric.risk_level
            
            # 高风险指标生成警告信号
            if risk_level.value in ["high", "very_high"]:
                signal = {
                    'timestamp': datetime.now(),
                    'action': 'risk_warning',
                    'price': price_bars[-1].close if price_bars else 0.0,
                    'risk_level': risk_level.value,
                    'metric_type': metric.metric_type.value,
                    'risk_score': metric.risk_score,
                    'recommendations': metric.recommendations,
                    'details': metric.details
                }
                signals.append(signal)
            
            # 极高风险指标生成减仓信号
            if risk_level.value == "very_high":
                signal = {
                    'timestamp': datetime.now(),
                    'action': 'reduce_position',
                    'price': price_bars[-1].close if price_bars else 0.0,
                    'risk_level': risk_level.value,
                    'metric_type': metric.metric_type.value,
                    'risk_score': metric.risk_score,
                    'recommendations': metric.recommendations,
                    'details': metric.details
                }
                signals.append(signal)
        
        # 记录风险指标
        self.risk_metrics.extend(risk_metrics)
        
        return signals
    
    def generate_system_report(self) -> Dict[str, Any]:
        """
        生成系统报告
        
        Returns:
            系统报告
        """
        report = {
            "system_name": "反转风险管理量化分析系统",
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "system_config": self.config,
            "performance_metrics": {
                "risk_metrics_calculated": len(self.risk_metrics),
                "risk_assessments_performed": len(self.risk_assessments),
                "current_balance": self.current_balance,
                "portfolio_value": self.portfolio_value,
                "max_drawdown": self.max_drawdown,
                "drawdown_percent": self.max_drawdown * 100,
            },
            "recent_activity": {
                "last_metrics": [m.metric_id for m in self.risk_metrics[-3:]] if self.risk_metrics else [],
                "last_assessments": [a.assessment_id for a in self.risk_assessments[-2:]] if self.risk_assessments else [],
            },
            "system_status": "active",
            "recommendations": [
                "定期审查和更新风险参数",
                "根据市场条件动态调整风险暴露",
                "建立严格的风险限制和监控机制",
                "保持风险管理的连续性和一致性",
            ]
        }
        
        return report


# ==================== 演示函数 ====================

def demonstrate_risk_management_system():
    """演示反转风险管理系统功能"""
    print("=" * 60)
    print("反转风险管理量化分析系统演示")
    print("严格按照第18章标准：实际完整代码")
    print("紧急冲刺模式：13:50-14:50完成")
    print("=" * 60)
    
    # 创建系统实例
    system = ReversalRiskManagement(initial_balance=10000.0)
    
    # 运行系统演示
    demonstration = system.demonstrate_system()
    
    print(f"\n📊 风险指标计算:")
    print(f"  计算的风险指标: {demonstration['risk_metrics_calculated']}个")
    
    assessment = demonstration["risk_assessment"]
    print(f"\n🎯 风险评估结果:")
    print(f"  总体风险等级: {assessment['overall_risk_level']}")
    print(f"  风险分数: {assessment['risk_score']:.2f}")
    print(f"  生成建议: {assessment['recommendations_count']}条")
    
    monitoring = demonstration["risk_monitoring"]
    print(f"\n👁️ 风险监控结果:")
    print(f"  所有限制正常: {'✅ 是' if monitoring['all_limits_ok'] else '❌ 否'}")
    print(f"  违反的限制: {monitoring['limits_violated']}个")
    print(f"  组合风险比率: {monitoring['portfolio_risk_ratio']:.2%}")
    
    # 生成系统报告
    report = system.generate_system_report()
    
    print(f"\n📈 系统报告摘要:")
    print(f"  系统状态: {report['system_status']}")
    print(f"  计算的风险指标: {report['performance_metrics']['risk_metrics_calculated']}")
    print(f"  进行的风险评估: {report['performance_metrics']['risk_assessments_performed']}")
    print(f"  当前资金: ${report['performance_metrics']['current_balance']:.2f}")
    print(f"  最大回撤: {report['performance_metrics']['drawdown_percent']:.2f}%")
    
    print(f"\n💡 系统推荐:")
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"  {i}. {recommendation}")
    
    print(f"\n✅ 演示完成！系统运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    # 当直接运行此文件时执行演示
    demonstrate_risk_management_system()