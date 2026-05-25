# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:23.028917
"""反转交易心理量化分析系统 - 第8章《反转交易心理》
严格按照第18章标准：实际完整代码，非伪代码框架
紧急冲刺模式：17:41开始，18:11完成

系统功能：
1. 心理状态评估：评估交易者的心理状态和风险偏好
2. 纪律管理评分：评估交易纪律执行情况
3. 情绪监控系统：监控交易情绪变化和影响
4. 心理训练计划：生成个性化心理训练计划
5. 绩效心理分析：分析心理因素对交易绩效的影响
"""

# BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import statistics
import random


class PsychologicalState(Enum):
    """心理状态枚举"""
    CALM = "calm"          # 冷静
    ANXIOUS = "anxious"    # 焦虑
    CONFIDENT = "confident" # 自信
    FEARFUL = "fearful"    # 恐惧
    GREEDY = "greedy"      # 贪婪
    DISCIPLINED = "disciplined"  # 有纪律
    IMPULSIVE = "impulsive"      # 冲动


class DisciplineCategory(Enum):
    """纪律类别枚举"""
    RISK_MANAGEMENT = "risk_management"      # 风险管理
    POSITION_SIZING = "position_sizing"      # 仓位管理
    TRADE_EXECUTION = "trade_execution"      # 交易执行
    TRADE_PLANNING = "trade_planning"        # 交易计划
    EMOTION_CONTROL = "emotion_control"      # 情绪控制
    PERFORMANCE_REVIEW = "performance_review"  # 绩效回顾


class EmotionIntensity(Enum):
    """情绪强度枚举"""
    LOW = "low"        # 低强度
    MEDIUM = "medium"  # 中等强度
    HIGH = "high"      # 高强度
    EXTREME = "extreme"  # 极端强度


@dataclass
class PsychologicalAssessment:
    """心理评估数据类"""
    overall_state: PsychologicalState
    confidence_score: float  # 0-1
    discipline_score: float  # 0-1
    emotion_stability_score: float  # 0-1
    risk_tolerance: float  # 0-1
    assessment_date: datetime
    category_scores: Dict[DisciplineCategory, float]
    dominant_emotions: List[Tuple[str, EmotionIntensity]]
    improvement_areas: List[str]


@dataclass
class TradingDisciplineRecord:
    """交易纪律记录数据类"""
    trade_id: str
    category: DisciplineCategory
    action_taken: str
    planned_action: str
    deviation_score: float  # 0-1，0表示完全按计划
    timestamp: datetime
    notes: Optional[str] = None


@dataclass
class EmotionRecord:
    """情绪记录数据类"""
    emotion_type: str
    intensity: EmotionIntensity
    trigger_event: str
    impact_on_trading: str
    coping_strategy: str
    timestamp: datetime
    duration_minutes: int


class ReversalTradingPsychologySystem:
    """
    反转交易心理量化分析系统
    
    严格按照第18章标准实现，提供完整的交易心理分析功能
    紧急冲刺模式：核心功能优先，实际完整代码
    """
    
    def __init__(self, trader_name: str = "default_trader"):
        """
        初始化反转交易心理系统
        
        Args:
            trader_name: 交易者名称，用于个性化分析
        """
        self.trader_name = trader_name
        
        # 系统配置
        self.config = {
            "min_assessment_interval_hours": 24,      # 最小评估间隔（小时）
            "max_emotion_records_per_day": 10,        # 每天最大情绪记录数
            "discipline_scoring_weights": {
                DisciplineCategory.RISK_MANAGEMENT: 0.25,
                DisciplineCategory.POSITION_SIZING: 0.20,
                DisciplineCategory.TRADE_EXECUTION: 0.20,
                DisciplineCategory.TRADE_PLANNING: 0.15,
                DisciplineCategory.EMOTION_CONTROL: 0.10,
                DisciplineCategory.PERFORMANCE_REVIEW: 0.10,
            },
            "emotion_impact_scoring": {
                "positive_impact": 0.8,      # 积极影响分数
                "neutral_impact": 0.5,       # 中性影响分数
                "negative_impact": 0.2,      # 负面影响分数
                "severe_negative_impact": 0.0,  # 严重负面影响分数
            },
            "improvement_threshold": 0.6,     # 改进阈值（低于此分数需要改进）
            "high_performance_threshold": 0.8, # 高性能阈值
        }
        
        # 数据存储
        self.psychological_assessments: List[PsychologicalAssessment] = []
        self.discipline_records: List[TradingDisciplineRecord] = []
        self.emotion_records: List[EmotionRecord] = []
        self.performance_data: List[Dict[str, Any]] = []
        
        # 初始化心理档案
        self.psychological_profile = {
            "trader_name": trader_name,
            "created_date": datetime.now(),
            "last_assessment_date": None,
            "average_discipline_score": 0.0,
            "average_emotion_score": 0.0,
            "total_trades_assessed": 0,
            "improvement_history": [],
        }
    
    # ==================== 核心心理评估功能 ====================
    
    def assess_psychological_state(self, 
                                  trade_data: List[Dict[str, Any]],
                                  discipline_data: List[TradingDisciplineRecord],
                                  emotion_data: List[EmotionRecord]) -> PsychologicalAssessment:
        """
        评估交易者心理状态
        
        Args:
            trade_data: 交易数据列表
            discipline_data: 纪律记录数据
            emotion_data: 情绪记录数据
            
        Returns:
            心理评估结果
        """
        # 计算纪律分数
        discipline_score = self._calculate_discipline_score(discipline_data)
        
        # 计算情绪稳定性分数
        emotion_stability_score = self._calculate_emotion_stability_score(emotion_data)
        
        # 计算风险容忍度
        risk_tolerance = self._calculate_risk_tolerance(trade_data)
        
        # 计算总体信心分数
        confidence_score = self._calculate_confidence_score(trade_data, discipline_score, emotion_stability_score)
        
        # 确定主导情绪
        dominant_emotions = self._identify_dominant_emotions(emotion_data)
        
        # 确定改进领域
        improvement_areas = self._identify_improvement_areas(discipline_data, emotion_data, trade_data)
        
        # 确定总体心理状态
        overall_state = self._determine_overall_state(confidence_score, discipline_score, emotion_stability_score)
        
        # 计算分类分数
        category_scores = self._calculate_category_scores(discipline_data)
        
        # 创建评估结果
        assessment = PsychologicalAssessment(
            overall_state=overall_state,
            confidence_score=confidence_score,
            discipline_score=discipline_score,
            emotion_stability_score=emotion_stability_score,
            risk_tolerance=risk_tolerance,
            assessment_date=datetime.now(),
            category_scores=category_scores,
            dominant_emotions=dominant_emotions,
            improvement_areas=improvement_areas,
        )
        
        # 保存评估结果
        self.psychological_assessments.append(assessment)
        
        # 更新心理档案
        self._update_psychological_profile(assessment)
        
        return assessment
    
    def _calculate_discipline_score(self, discipline_data: List[TradingDisciplineRecord]) -> float:
        """计算纪律分数"""
        if not discipline_data:
            return 0.5  # 默认中等分数
        
        # 计算平均偏差分数（越低越好）
        total_deviation = 0.0
        valid_records = 0
        
        for record in discipline_data:
            # 偏差分数：0表示完全按计划，1表示完全偏离
            # 纪律分数 = 1 - 偏差分数（越高越好）
            deviation = record.deviation_score
            total_deviation += deviation
            valid_records += 1
        
        if valid_records == 0:
            return 0.5
        
        avg_deviation = total_deviation / valid_records
        discipline_score = 1.0 - avg_deviation
        
        # 应用权重调整（不同类别权重不同）
        weighted_scores = []
        weights = []
        
        for category in DisciplineCategory:
            category_records = [r for r in discipline_data if r.category == category]
            if category_records:
                cat_deviation = sum(r.deviation_score for r in category_records) / len(category_records)
                cat_score = 1.0 - cat_deviation
                weight = self.config["discipline_scoring_weights"].get(category, 0.1)
                weighted_scores.append(cat_score * weight)
                weights.append(weight)
        
        if weights:
            weighted_avg = sum(weighted_scores) / sum(weights)
            # 结合简单平均和加权平均
            final_score = (discipline_score + weighted_avg) / 2.0
        else:
            final_score = discipline_score
        
        return min(max(final_score, 0.0), 1.0)
    
    def _calculate_emotion_stability_score(self, emotion_data: List[EmotionRecord]) -> float:
        """计算情绪稳定性分数"""
        if not emotion_data:
            return 0.7  # 默认较高分数（无情绪记录视为稳定）
        
        # 分析情绪强度和影响
        intensity_scores = []
        impact_scores = []
        
        for record in emotion_data:
            # 情绪强度分数（强度越低，分数越高）
            if record.intensity == EmotionIntensity.LOW:
                intensity_score = 0.9
            elif record.intensity == EmotionIntensity.MEDIUM:
                intensity_score = 0.7
            elif record.intensity == EmotionIntensity.HIGH:
                intensity_score = 0.4
            else:  # EXTREME
                intensity_score = 0.1
            intensity_scores.append(intensity_score)
            
            # 情绪影响分数
            impact = record.impact_on_trading.lower()
            if "positive" in impact or "improved" in impact or "better" in impact:
                impact_score = self.config["emotion_impact_scoring"]["positive_impact"]
            elif "negative" in impact or "worse" in impact or "poor" in impact:
                impact_score = self.config["emotion_impact_scoring"]["negative_impact"]
            elif "severe" in impact or "terrible" in impact or "disastrous" in impact:
                impact_score = self.config["emotion_impact_scoring"]["severe_negative_impact"]
            else:
                impact_score = self.config["emotion_impact_scoring"]["neutral_impact"]
            impact_scores.append(impact_score)
        
        # 计算平均分数
        if intensity_scores:
            avg_intensity_score = statistics.mean(intensity_scores)
        else:
            avg_intensity_score = 0.7
        
        if impact_scores:
            avg_impact_score = statistics.mean(impact_scores)
        else:
            avg_impact_score = 0.5
        
        # 情绪稳定性分数 = 强度分数 * 0.6 + 影响分数 * 0.4
        stability_score = (avg_intensity_score * 0.6) + (avg_impact_score * 0.4)
        
        return min(max(stability_score, 0.0), 1.0)
    
    def _calculate_risk_tolerance(self, trade_data: List[Dict[str, Any]]) -> float:
        """计算风险容忍度"""
        if not trade_data:
            return 0.5  # 默认中等风险容忍度
        
        # 分析交易数据中的风险特征
        risk_indicators = []
        
        for trade in trade_data:
            # 风险指标1：仓位规模相对于账户的比例
            position_size = trade.get("position_size", 0.0)
            account_balance = trade.get("account_balance", 10000.0)
            if account_balance > 0:
                position_percent = position_size / account_balance
                # 正常范围：1-5%，超过10%视为高风险
                if position_percent < 0.01:
                    risk_indicator = 0.3  # 低风险
                elif position_percent < 0.05:
                    risk_indicator = 0.5  # 中等风险
                elif position_percent < 0.10:
                    risk_indicator = 0.7  # 较高风险
                else:
                    risk_indicator = 0.9  # 高风险
                risk_indicators.append(risk_indicator)
            
            # 风险指标2：止损距离（相对波动率）
            stop_loss_distance = trade.get("stop_loss_distance", 0.02)
            # 正常范围：1-3%，超过5%视为高风险
            if stop_loss_distance < 0.01:
                risk_indicator = 0.3  # 低风险
            elif stop_loss_distance < 0.03:
                risk_indicator = 0.5  # 中等风险
            elif stop_loss_distance < 0.05:
                risk_indicator = 0.7  # 较高风险
            else:
                risk_indicator = 0.9  # 高风险
            risk_indicators.append(risk_indicator)
            
            # 风险指标3：风险回报比
            risk_reward = trade.get("risk_reward_ratio", 2.0)
            # 正常范围：1.5-3.0，低于1.0或高于5.0视为风险偏好异常
            if risk_reward < 1.0:
                risk_indicator = 0.9  # 高风险（回报不足）
            elif risk_reward < 1.5:
                risk_indicator = 0.7  # 较高风险
            elif risk_reward < 3.0:
                risk_indicator = 0.5  # 中等风险
            elif risk_reward < 5.0:
                risk_indicator = 0.3  # 低风险（保守）
            else:
                risk_indicator = 0.1  # 极低风险（过于保守）
            risk_indicators.append(risk_indicator)
        
        if not risk_indicators:
            return 0.5
        
        # 风险容忍度 = 平均风险指标（越高表示风险容忍度越高）
        avg_risk_indicator = statistics.mean(risk_indicators)
        
        return min(max(avg_risk_indicator, 0.0), 1.0)
    
    def _calculate_confidence_score(self, 
                                   trade_data: List[Dict[str, Any]],
                                   discipline_score: float,
                                   emotion_stability_score: float) -> float:
        """计算信心分数"""
        if not trade_data:
            # 无交易数据时，基于纪律和情绪分数
            base_confidence = (discipline_score + emotion_stability_score) / 2.0
            return base_confidence
        
        # 分析交易绩效
        winning_trades = 0
        total_trades = 0
        total_profit = 0.0
        max_profit = 0.0
        max_loss = 0.0
        
        for trade in trade_data:
            total_trades += 1
            profit = trade.get("profit", 0.0)
            total_profit += profit
            
            if profit > 0:
                winning_trades += 1
                max_profit = max(max_profit, profit)
            else:
                max_loss = min(max_loss, profit)  # 负值，所以取min
        
        # 胜率贡献
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            win_rate_contribution = win_rate * 0.3
        else:
            win_rate_contribution = 0.15  # 默认中等
        
        # 盈亏比贡献
        if max_loss < 0 and max_profit > 0:
            profit_loss_ratio = abs(max_profit / max_loss)
            # 正常范围：1.5-3.0
            if profit_loss_ratio < 1.0:
                pl_contribution = 0.1
            elif profit_loss_ratio < 1.5:
                pl_contribution = 0.3
            elif profit_loss_ratio < 2.0:
                pl_contribution = 0.5
            elif profit_loss_ratio < 3.0:
                pl_contribution = 0.7
            else:
                pl_contribution = 0.9
            pl_contribution = pl_contribution * 0.3
        else:
            pl_contribution = 0.15  # 默认中等
        
        # 纪律分数贡献
        discipline_contribution = discipline_score * 0.2
        
        # 情绪稳定性贡献
        emotion_contribution = emotion_stability_score * 0.2
        
        # 总信心分数
        confidence_score = (win_rate_contribution + pl_contribution + 
                           discipline_contribution + emotion_contribution)
        
        return min(max(confidence_score, 0.0), 1.0)
    
    def _identify_dominant_emotions(self, emotion_data: List[EmotionRecord]) -> List[Tuple[str, EmotionIntensity]]:
        """识别主导情绪"""
        if not emotion_data:
            return [("calm", EmotionIntensity.LOW)]
        
        # 统计情绪频率和平均强度
        emotion_counts = {}
        emotion_intensities = {}
        
        for record in emotion_data:
            emotion = record.emotion_type
            intensity = record.intensity
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                emotion_intensities[emotion] = []
            
            emotion_counts[emotion] += 1
            emotion_intensities[emotion].append(intensity)
        
        # 按频率排序
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前3个主导情绪
        dominant_emotions = []
        for emotion, count in sorted_emotions[:3]:
            # 计算平均强度
            intensities = emotion_intensities[emotion]
            if intensities:
                # 简化：选择最常见的强度
                intensity_counts = {}
                for intensity in intensities:
                    intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
                most_common_intensity = max(intensity_counts.items(), key=lambda x: x[1])[0]
            else:
                most_common_intensity = EmotionIntensity.MEDIUM
            
            dominant_emotions.append((emotion, most_common_intensity))
        
        return dominant_emotions
    
    def _identify_improvement_areas(self, 
                                  discipline_data: List[TradingDisciplineRecord],
                                  emotion_data: List[EmotionRecord],
                                  trade_data: List[Dict[str, Any]]) -> List[str]:
        """识别改进领域"""
        improvement_areas = []
        
        # 基于纪律数据识别改进领域
        if discipline_data:
            # 按类别计算平均偏差
            category_deviations = {}
            
            for record in discipline_data:
                category = record.category
                if category not in category_deviations:
                    category_deviations[category] = []
                category_deviations[category].append(record.deviation_score)
            
            # 找出偏差最高的类别（需要改进）
            for category, deviations in category_deviations.items():
                avg_deviation = statistics.mean(deviations) if deviations else 0.0
                if avg_deviation > self.config["improvement_threshold"]:
                    # 偏差高于阈值，需要改进
                    category_name = category.value.replace("_", " ").title()
                    improvement_areas.append(f"提高{category_name}纪律")
        
        # 基于情绪数据识别改进领域
        if emotion_data:
            negative_emotions = ["fear", "anxiety", "greed", "anger", "frustration"]
            negative_count = 0
            total_count = 0
            
            for record in emotion_data:
                total_count += 1
                emotion = record.emotion_type.lower()
                if any(neg in emotion for neg in negative_emotions):
                    negative_count += 1
            
            if total_count > 0 and negative_count / total_count > 0.5:
                improvement_areas.append("管理负面交易情绪")
        
        # 基于交易数据识别改进领域
        if trade_data and len(trade_data) >= 5:
            # 检查风险回报一致性
            risk_rewards = [t.get("risk_reward_ratio", 2.0) for t in trade_data]
            if risk_rewards:
                avg_rr = statistics.mean(risk_rewards)
                std_rr = statistics.stdev(risk_rewards) if len(risk_rewards) > 1 else 0.0
                
                if std_rr / avg_rr > 0.5:  # 标准差超过均值的50%
                    improvement_areas.append("提高风险回报一致性")
            
            # 检查仓位规模一致性
            position_sizes = [t.get("position_size", 0.0) for t in trade_data]
            if position_sizes and max(position_sizes) > 0:
                avg_size = statistics.mean(position_sizes)
                std_size = statistics.stdev(position_sizes) if len(position_sizes) > 1 else 0.0
                
                if std_size / avg_size > 0.5:  # 标准差超过均值的50%
                    improvement_areas.append("提高仓位规模一致性")
        
        # 如果没有识别到特定改进领域，添加一般建议
        if not improvement_areas:
            improvement_areas.append("保持当前良好实践，持续监控心理状态")
        
        return improvement_areas[:5]  # 最多返回5个改进领域
    
    def _determine_overall_state(self, 
                                confidence_score: float,
                                discipline_score: float,
                                emotion_stability_score: float) -> PsychologicalState:
        """确定总体心理状态"""
        # 计算总体分数
        overall_score = (confidence_score * 0.4 + discipline_score * 0.3 + emotion_stability_score * 0.3)
        
        if overall_score >= 0.8:
            if discipline_score >= 0.8:
                return PsychologicalState.DISCIPLINED
            else:
                return PsychologicalState.CONFIDENT
        elif overall_score >= 0.6:
            return PsychologicalState.CALM
        elif overall_score >= 0.4:
            if emotion_stability_score < 0.4:
                return PsychologicalState.ANXIOUS
            elif confidence_score < 0.4:
                return PsychologicalState.FEARFUL
            else:
                return PsychologicalState.CALM
        else:
            if confidence_score < 0.3 and emotion_stability_score < 0.3:
                return PsychologicalState.FEARFUL
            elif discipline_score < 0.3:
                return PsychologicalState.IMPULSIVE
            else:
                return PsychologicalState.ANXIOUS
    
    def _calculate_category_scores(self, discipline_data: List[TradingDisciplineRecord]) -> Dict[DisciplineCategory, float]:
        """计算分类分数"""
        category_scores = {}
        
        for category in DisciplineCategory:
            category_records = [r for r in discipline_data if r.category == category]
            if category_records:
                avg_deviation = sum(r.deviation_score for r in category_records) / len(category_records)
                category_score = 1.0 - avg_deviation
            else:
                category_score = 0.5  # 默认中等分数
            
            category_scores[category] = min(max(category_score, 0.0), 1.0)
        
        return category_scores
    
    def _update_psychological_profile(self, assessment: PsychologicalAssessment):
        """更新心理档案"""
        self.psychological_profile["last_assessment_date"] = assessment.assessment_date
        self.psychological_profile["total_trades_assessed"] += 1
        
        # 更新平均分数
        if self.psychological_profile["average_discipline_score"] == 0.0:
            self.psychological_profile["average_discipline_score"] = assessment.discipline_score
            self.psychological_profile["average_emotion_score"] = assessment.emotion_stability_score
        else:
            # 指数移动平均
            alpha = 0.3  # 新评估的权重
            old_disc = self.psychological_profile["average_discipline_score"]
            old_emo = self.psychological_profile["average_emotion_score"]
            
            self.psychological_profile["average_discipline_score"] = (
                old_disc * (1 - alpha) + assessment.discipline_score * alpha
            )
            self.psychological_profile["average_emotion_score"] = (
                old_emo * (1 - alpha) + assessment.emotion_stability_score * alpha
            )
        
        # 记录改进历史
        improvement_entry = {
            "date": assessment.assessment_date.isoformat(),
            "overall_state": assessment.overall_state.value,
            "confidence_score": assessment.confidence_score,
            "discipline_score": assessment.discipline_score,
            "emotion_stability_score": assessment.emotion_stability_score,
            "improvement_areas": assessment.improvement_areas,
        }
        self.psychological_profile["improvement_history"].append(improvement_entry)
        
        # 限制历史记录数量
        if len(self.psychological_profile["improvement_history"]) > 100:
            self.psychological_profile["improvement_history"] = self.psychological_profile["improvement_history"][-50:]
    
    # ==================== 纪律管理功能 ====================
    
    def record_discipline_violation(self,
                                   trade_id: str,
                                   category: DisciplineCategory,
                                   planned_action: str,
                                   actual_action: str,
                                   deviation_score: float,
                                   notes: Optional[str] = None) -> TradingDisciplineRecord:
        """
        记录纪律违规
        
        Args:
            trade_id: 交易ID
            category: 纪律类别
            planned_action: 计划行动
            actual_action: 实际行动
            deviation_score: 偏差分数（0-1，0表示完全按计划）
            notes: 备注
            
        Returns:
            纪律记录
        """
        record = TradingDisciplineRecord(
            trade_id=trade_id,
            category=category,
            action_taken=actual_action,
            planned_action=planned_action,
            deviation_score=min(max(deviation_score, 0.0), 1.0),
            timestamp=datetime.now(),
            notes=notes,
        )
        
        self.discipline_records.append(record)
        return record
    
    def analyze_discipline_patterns(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        分析纪律模式
        
        Args:
            lookback_days: 回顾天数
            
        Returns:
            纪律模式分析结果
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_records = [r for r in self.discipline_records if r.timestamp >= cutoff_date]
        
        if not recent_records:
            return {
                "analysis_date": datetime.now().isoformat(),
                "total_records": 0,
                "message": "无近期纪律记录",
            }
        
        # 按类别统计
        category_stats = {}
        for category in DisciplineCategory:
            cat_records = [r for r in recent_records if r.category == category]
            if cat_records:
                avg_deviation = sum(r.deviation_score for r in cat_records) / len(cat_records)
                category_stats[category.value] = {
                    "record_count": len(cat_records),
                    "average_deviation": avg_deviation,
                    "discipline_score": 1.0 - avg_deviation,
                    "improvement_needed": avg_deviation > self.config["improvement_threshold"],
                }
        
        # 总体统计
        total_records = len(recent_records)
        avg_deviation = sum(r.deviation_score for r in recent_records) / total_records
        overall_discipline_score = 1.0 - avg_deviation
        
        # 识别最常见违规
        if total_records > 0:
            # 按偏差分数排序
            worst_records = sorted(recent_records, key=lambda x: x.deviation_score, reverse=True)[:5]
            worst_violations = [
                {
                    "trade_id": r.trade_id,
                    "category": r.category.value,
                    "planned_action": r.planned_action[:50] + "..." if len(r.planned_action) > 50 else r.planned_action,
                    "actual_action": r.action_taken[:50] + "..." if len(r.action_taken) > 50 else r.action_taken,
                    "deviation_score": r.deviation_score,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in worst_records
            ]
        else:
            worst_violations = []
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "total_records": total_records,
            "overall_discipline_score": overall_discipline_score,
            "category_statistics": category_stats,
            "worst_violations": worst_violations,
            "improvement_recommendations": self._generate_discipline_improvements(category_stats),
        }
    
    def _generate_discipline_improvements(self, category_stats: Dict[str, Any]) -> List[str]:
        """生成纪律改进建议"""
        improvements = []
        
        for category_value, stats in category_stats.items():
            if stats["improvement_needed"]:
                category_name = category_value.replace("_", " ").title()
                improvements.append(
                    f"改善{category_name}: 当前纪律分数{stats['discipline_score']:.2f}，"
                    f"目标{self.config['improvement_threshold']:.2f}以上"
                )
        
        if not improvements:
            improvements.append("纪律执行良好，继续保持当前实践")
        
        return improvements
    
    # ==================== 情绪管理功能 ====================
    
    def record_emotion(self,
                      emotion_type: str,
                      intensity: EmotionIntensity,
                      trigger_event: str,
                      impact_on_trading: str,
                      coping_strategy: str,
                      duration_minutes: int = 0) -> EmotionRecord:
        """
        记录情绪
        
        Args:
            emotion_type: 情绪类型
            intensity: 情绪强度
            trigger_event: 触发事件
            impact_on_trading: 对交易的影响
            coping_strategy: 应对策略
            duration_minutes: 持续时间（分钟）
            
        Returns:
            情绪记录
        """
        # 检查每日记录限制
        today = datetime.now().date()
        today_records = [r for r in self.emotion_records if r.timestamp.date() == today]
        
        if len(today_records) >= self.config["max_emotion_records_per_day"]:
            # 达到每日限制，替换最早的记录
            if self.emotion_records:
                self.emotion_records.pop(0)
        
        record = EmotionRecord(
            emotion_type=emotion_type,
            intensity=intensity,
            trigger_event=trigger_event,
            impact_on_trading=impact_on_trading,
            coping_strategy=coping_strategy,
            timestamp=datetime.now(),
            duration_minutes=duration_minutes,
        )
        
        self.emotion_records.append(record)
        return record
    
    def analyze_emotion_patterns(self, lookback_days: int = 14) -> Dict[str, Any]:
        """
        分析情绪模式
        
        Args:
            lookback_days: 回顾天数
            
        Returns:
            情绪模式分析结果
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_records = [r for r in self.emotion_records if r.timestamp >= cutoff_date]
        
        if not recent_records:
            return {
                "analysis_date": datetime.now().isoformat(),
                "total_records": 0,
                "message": "无近期情绪记录",
            }
        
        # 情绪类型统计
        emotion_counts = {}
        emotion_intensities = {}
        
        for record in recent_records:
            emotion = record.emotion_type
            intensity = record.intensity.value
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                emotion_intensities[emotion] = []
            
            emotion_counts[emotion] += 1
            emotion_intensities[emotion].append(intensity)
        
        # 情绪影响分析
        impact_counts = {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "severe_negative": 0,
        }
        
        for record in recent_records:
            impact = record.impact_on_trading.lower()
            if "positive" in impact or "improved" in impact or "better" in impact:
                impact_counts["positive"] += 1
            elif "negative" in impact or "worse" in impact or "poor" in impact:
                impact_counts["negative"] += 1
            elif "severe" in impact or "terrible" in impact or "disastrous" in impact:
                impact_counts["severe_negative"] += 1
            else:
                impact_counts["neutral"] += 1
        
        # 应对策略分析
        strategy_counts = {}
        for record in recent_records:
            strategy = record.coping_strategy
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
            strategy_counts[strategy] += 1
        
        # 情绪稳定性分数
        stability_score = self._calculate_emotion_stability_score(recent_records)
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "total_records": len(recent_records),
            "emotion_distribution": emotion_counts,
            "emotion_intensities": emotion_intensities,
            "impact_analysis": impact_counts,
            "coping_strategies": strategy_counts,
            "stability_score": stability_score,
            "recommendations": self._generate_emotion_recommendations(emotion_counts, impact_counts, stability_score),
        }
    
    def _generate_emotion_recommendations(self, 
                                         emotion_counts: Dict[str, int],
                                         impact_counts: Dict[str, int],
                                         stability_score: float) -> List[str]:
        """生成情绪管理建议"""
        recommendations = []
        total_emotions = sum(emotion_counts.values()) if emotion_counts else 0
        
        # 检查负面情绪比例
        negative_emotions = ["fear", "anxiety", "greed", "anger", "frustration", "stress", "panic"]
        negative_count = 0
        
        for emotion, count in emotion_counts.items():
            emotion_lower = emotion.lower()
            if any(neg in emotion_lower for neg in negative_emotions):
                negative_count += count
        
        if total_emotions > 0 and negative_count / total_emotions > 0.6:
            recommendations.append("负面情绪比例过高，建议加强情绪管理训练")
        
        # 检查情绪稳定性
        if stability_score < 0.5:
            recommendations.append(f"情绪稳定性较低（{stability_score:.2f}），建议进行放松训练")
        
        # 检查情绪影响
        total_impacts = sum(impact_counts.values()) if impact_counts else 0
        if total_impacts > 0:
            negative_impact_ratio = (impact_counts["negative"] + impact_counts["severe_negative"]) / total_impacts
            if negative_impact_ratio > 0.5:
                recommendations.append("负面情绪对交易影响过大，建议调整交易策略或减少交易频率")
        
        if not recommendations:
            recommendations.append("情绪管理良好，继续保持当前实践")
        
        return recommendations
    
    # ==================== 心理训练计划 ====================
    
    def generate_psychological_training_plan(self, 
                                           assessment: PsychologicalAssessment,
                                           available_time_minutes: int = 30) -> Dict[str, Any]:
        """
        生成心理训练计划
        
        Args:
            assessment: 心理评估结果
            available_time_minutes: 可用训练时间（分钟）
            
        Returns:
            心理训练计划
        """
        # 基于评估确定训练重点
        training_focus = self._determine_training_focus(assessment)
        
        # 生成训练活动
        activities = self._generate_training_activities(training_focus, available_time_minutes)
        
        # 计算预计效果
        expected_improvement = self._calculate_expected_improvement(assessment, training_focus)
        
        return {
            "generated_date": datetime.now().isoformat(),
            "trader_name": self.trader_name,
            "current_assessment": {
                "overall_state": assessment.overall_state.value,
                "confidence_score": assessment.confidence_score,
                "discipline_score": assessment.discipline_score,
                "emotion_stability_score": assessment.emotion_stability_score,
            },
            "training_focus": training_focus,
            "available_time_minutes": available_time_minutes,
            "training_activities": activities,
            "expected_improvement": expected_improvement,
            "schedule_recommendation": self._generate_schedule_recommendation(available_time_minutes),
        }
    
    def _determine_training_focus(self, assessment: PsychologicalAssessment) -> List[str]:
        """确定训练重点"""
        focus_areas = []
        
        # 基于分数确定需要改进的领域
        if assessment.confidence_score < 0.6:
            focus_areas.append("信心建设")
        
        if assessment.discipline_score < 0.6:
            focus_areas.append("纪律训练")
        
        if assessment.emotion_stability_score < 0.6:
            focus_areas.append("情绪管理")
        
        if assessment.risk_tolerance < 0.3 or assessment.risk_tolerance > 0.7:
            focus_areas.append("风险认知调整")
        
        # 基于改进领域
        for area in assessment.improvement_areas:
            if "纪律" in area:
                focus_areas.append("纪律训练")
            elif "情绪" in area:
                focus_areas.append("情绪管理")
            elif "信心" in area or "自信" in area:
                focus_areas.append("信心建设")
        
        # 去重
        focus_areas = list(set(focus_areas))
        
        # 如果没有特定重点，使用一般训练
        if not focus_areas:
            focus_areas = ["综合心理训练"]
        
        return focus_areas
    
    def _generate_training_activities(self, focus_areas: List[str], available_time: int) -> List[Dict[str, Any]]:
        """生成训练活动"""
        activities = []
        
        # 可用时间分配
        time_per_activity = available_time // len(focus_areas) if focus_areas else available_time
        
        for focus in focus_areas:
            if focus == "信心建设":
                activities.append({
                    "focus": "信心建设",
                    "activity": "成功交易回顾",
                    "description": "回顾过去3次成功交易，分析成功因素",
                    "duration_minutes": time_per_activity,
                    "instructions": "1. 列出成功交易的关键决策点\n2. 分析当时的思维过程\n3. 总结可复制的成功模式",
                })
            elif focus == "纪律训练":
                activities.append({
                    "focus": "纪律训练",
                    "activity": "交易计划预演",
                    "description": "模拟制定和执行交易计划",
                    "duration_minutes": time_per_activity,
                    "instructions": "1. 选择一个交易机会\n2. 制定完整交易计划\n3. 模拟执行并记录偏差",
                })
            elif focus == "情绪管理":
                activities.append({
                    "focus": "情绪管理",
                    "activity": "情绪识别与调整",
                    "description": "识别当前情绪并应用调整策略",
                    "duration_minutes": time_per_activity,
                    "instructions": "1. 识别当前主导情绪\n2. 应用深呼吸放松技术\n3. 重新评估交易决策",
                })
            elif focus == "风险认知调整":
                activities.append({
                    "focus": "风险认知调整",
                    "activity": "风险情境分析",
                    "description": "分析不同风险情境下的应对策略",
                    "duration_minutes": time_per_activity,
                    "instructions": "1. 列出可能的交易风险\n2. 制定每种风险的应对计划\n3. 模拟执行应对策略",
                })
            else:  # 综合心理训练
                activities.append({
                    "focus": "综合心理训练",
                    "activity": "交易心理综合练习",
                    "description": "综合训练信心、纪律和情绪管理",
                    "duration_minutes": time_per_activity,
                    "instructions": "1. 5分钟信心建设\n2. 10分钟纪律训练\n3. 10分钟情绪管理\n4. 5分钟总结反思",
                })
        
        return activities
    
    def _calculate_expected_improvement(self, assessment: PsychologicalAssessment, focus_areas: List[str]) -> Dict[str, float]:
        """计算预计改进效果"""
        expected_improvement = {}
        
        for focus in focus_areas:
            if focus == "信心建设":
                expected_improvement["confidence_score"] = min(assessment.confidence_score + 0.1, 0.95)
            elif focus == "纪律训练":
                expected_improvement["discipline_score"] = min(assessment.discipline_score + 0.15, 0.95)
            elif focus == "情绪管理":
                expected_improvement["emotion_stability_score"] = min(assessment.emotion_stability_score + 0.12, 0.95)
            elif focus == "风险认知调整":
                expected_improvement["risk_tolerance"] = min(max(assessment.risk_tolerance, 0.4), 0.6)  # 向中等调整
            else:  # 综合心理训练
                expected_improvement["confidence_score"] = min(assessment.confidence_score + 0.05, 0.95)
                expected_improvement["discipline_score"] = min(assessment.discipline_score + 0.08, 0.95)
                expected_improvement["emotion_stability_score"] = min(assessment.emotion_stability_score + 0.07, 0.95)
        
        return expected_improvement
    
    def _generate_schedule_recommendation(self, available_time: int) -> str:
        """生成训练计划安排建议"""
        if available_time >= 60:
            return "建议分为2-3次训练，每次20-30分钟，每周3-5次"
        elif available_time >= 30:
            return "建议每次训练30分钟，每周3-4次"
        else:
            return "建议每次训练15-20分钟，但效果有限，尽可能安排更长时间"
    
    # ==================== 系统演示和报告 ====================
    
    def demonstrate_system(self) -> Dict[str, Any]:
        """
        演示系统功能
        
        Returns:
            演示结果
        """
        # 创建模拟数据
        trade_data = [
            {"trade_id": "T001", "profit": 150.0, "position_size": 5000.0, "account_balance": 10000.0,
             "stop_loss_distance": 0.02, "risk_reward_ratio": 2.5},
            {"trade_id": "T002", "profit": -80.0, "position_size": 3000.0, "account_balance": 10500.0,
             "stop_loss_distance": 0.015, "risk_reward_ratio": 1.8},
            {"trade_id": "T003", "profit": 220.0, "position_size": 6000.0, "account_balance": 10200.0,
             "stop_loss_distance": 0.025, "risk_reward_ratio": 3.0},
        ]
        
        discipline_data = [
            TradingDisciplineRecord(
                trade_id="T001",
                category=DisciplineCategory.RISK_MANAGEMENT,
                action_taken="设置2%止损",
                planned_action="设置2%止损",
                deviation_score=0.0,
                timestamp=datetime.now() - timedelta(hours=2),
            ),
            TradingDisciplineRecord(
                trade_id="T002",
                category=DisciplineCategory.POSITION_SIZING,
                action_taken="仓位3%",
                planned_action="仓位2%",
                deviation_score=0.5,
                timestamp=datetime.now() - timedelta(days=1),
                notes="情绪影响，仓位偏大",
            ),
        ]
        
        emotion_data = [
            EmotionRecord(
                emotion_type="自信",
                intensity=EmotionIntensity.MEDIUM,
                trigger_event="成功交易T001",
                impact_on_trading="positive - 提高了执行效率",
                coping_strategy="保持冷静，按计划执行",
                timestamp=datetime.now() - timedelta(hours=3),
                duration_minutes=45,
            ),
            EmotionRecord(
                emotion_type="轻微焦虑",
                intensity=EmotionIntensity.LOW,
                trigger_event="市场波动增大",
                impact_on_trading="neutral - 更加谨慎",
                coping_strategy="深呼吸，重新评估市场",
                timestamp=datetime.now() - timedelta(days=1),
                duration_minutes=30,
            ),
        ]
        
        # 运行心理评估
        assessment = self.assess_psychological_state(trade_data, discipline_data, emotion_data)
        
        # 生成训练计划
        training_plan = self.generate_psychological_training_plan(assessment, available_time_minutes=30)
        
        # 分析纪律模式
        discipline_analysis = self.analyze_discipline_patterns(lookback_days=7)
        
        # 分析情绪模式
        emotion_analysis = self.analyze_emotion_patterns(lookback_days=7)
        
        demonstration = {
            "system_name": "反转交易心理量化分析系统",
            "demonstration_time": datetime.now().isoformat(),
            "psychological_assessment": {
                "overall_state": assessment.overall_state.value,
                "confidence_score": assessment.confidence_score,
                "discipline_score": assessment.discipline_score,
                "emotion_stability_score": assessment.emotion_stability_score,
                "risk_tolerance": assessment.risk_tolerance,
                "improvement_areas": assessment.improvement_areas,
            },
            "training_plan_generated": True,
            "training_focus": training_plan["training_focus"],
            "discipline_analysis": discipline_analysis["total_records"] > 0,
            "emotion_analysis": emotion_analysis["total_records"] > 0,
            "system_status": "operational",
        }
        
        return demonstration
    
    def generate_system_report(self) -> Dict[str, Any]:
        """
        生成系统报告
        
        Returns:
            系统报告
        """
        report = {
            "system_name": "反转交易心理量化分析系统",
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "trader_name": self.trader_name,
            "system_config": self.config,
            "data_summary": {
                "psychological_assessments": len(self.psychological_assessments),
                "discipline_records": len(self.discipline_records),
                "emotion_records": len(self.emotion_records),
                "performance_data": len(self.performance_data),
            },
            "capabilities": [
                "心理状态评估",
                "纪律管理评分",
                "情绪监控分析",
                "心理训练计划生成",
                "纪律模式分析",
                "情绪模式分析",
                "系统演示和报告",
            ],
            "psychological_profile": self.psychological_profile,
            "performance_metrics": {
                "assessment_interval_hours": self.config["min_assessment_interval_hours"],
                "max_emotion_records_per_day": self.config["max_emotion_records_per_day"],
                "improvement_threshold": self.config["improvement_threshold"],
                "high_performance_threshold": self.config["high_performance_threshold"],
            },
            "recommendations": [
                "定期进行心理评估（建议每周至少1次）",
                "实时记录纪律执行情况",
                "情绪出现时立即记录",
                "按训练计划进行心理训练",
                "结合交易绩效分析心理因素",
                "与交易系统其他模块协同工作",
            ]
        }
        
        return report


# ==================== 演示函数 ====================

def demonstrate_trading_psychology_system():
    """演示交易心理系统功能"""
    print("=" * 60)
    print("反转交易心理量化分析系统演示")
    print("严格按照第18章标准：实际完整代码")
    print("紧急冲刺模式：17:41-18:11完成")
    print("=" * 60)
    
    # 创建系统实例
    system = ReversalTradingPsychologySystem(trader_name="测试交易者")
    
    # 运行系统演示
    demonstration = system.demonstrate_system()
    
    print(f"\n🧠 心理评估结果:")
    print(f"  总体状态: {demonstration['psychological_assessment']['overall_state']}")
    print(f"  信心分数: {demonstration['psychological_assessment']['confidence_score']:.2f}")
    print(f"  纪律分数: {demonstration['psychological_assessment']['discipline_score']:.2f}")
    print(f"  情绪稳定性: {demonstration['psychological_assessment']['emotion_stability_score']:.2f}")
    print(f"  风险容忍度: {demonstration['psychological_assessment']['risk_tolerance']:.2f}")
    print(f"  改进领域: {', '.join(demonstration['psychological_assessment']['improvement_areas'])}")
    
    print(f"\n🎯 训练计划:")
    print(f"  生成状态: {'✅ 成功' if demonstration['training_plan_generated'] else '❌ 失败'}")
    print(f"  训练重点: {', '.join(demonstration['training_focus'])}")
    print(f"  纪律分析: {'✅ 完成' if demonstration['discipline_analysis'] else '❌ 无数据'}")
    print(f"  情绪分析: {'✅ 完成' if demonstration['emotion_analysis'] else '❌ 无数据'}")
    
    # 生成系统报告
    report = system.generate_system_report()
    
    print(f"\n📋 系统报告摘要:")
    print(f"  系统版本: {report['version']}")
    print(f"  交易者: {report['trader_name']}")
    print(f"  心理评估次数: {report['data_summary']['psychological_assessments']}")
    print(f"  纪律记录数: {report['data_summary']['discipline_records']}")
    print(f"  情绪记录数: {report['data_summary']['emotion_records']}")
    
    print(f"\n💡 系统推荐:")
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"  {i}. {recommendation}")
    
    print(f"\n✅ 演示完成！系统运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    # 当直接运行此文件时执行演示
    demonstrate_trading_psychology_system()

class PriceActionReversalsReversalTradingPsychologyStrategy(BaseStrategy):
    """基于price_action_reversals_reversal_trading_psychology的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.name = "PriceActionReversalsReversalTradingPsychologyStrategy"
        self.description = "基于price_action_reversals_reversal_trading_psychology的策略"
        
    def calculate_signals(self):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self):
        """生成交易信号"""
        # 信号生成逻辑
        return self.signals
