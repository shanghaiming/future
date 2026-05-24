# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.641683

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
常见错误与避免量化系统
第31章：常见错误与避免
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 错误分类系统：系统化分类交易中的常见错误类型
2. 错误检测引擎：实时检测交易决策和操作中的潜在错误
3. 预防策略库：针对不同错误的预防措施和策略
4. 纠正机制：错误发生后的纠正和补救措施
5. 学习反馈循环：从错误中学习并改进交易系统
6. 风险评估：错误可能导致的后果和风险评估
7. 个性化错误分析：基于交易者特点的错误倾向分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime, timedelta
import json
import hashlib
import warnings
from dataclasses import dataclass, asdict, field
from enum import Enum
import re
import random
from collections import defaultdict
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy


class ErrorCategory(Enum):
    """错误类别枚举"""
    PSYCHOLOGICAL = "psychological"      # 心理错误
    DISCIPLINE = "discipline"            # 纪律错误
    RISK_MANAGEMENT = "risk_management"  # 风险管理错误
    TECHNICAL_ANALYSIS = "technical_analysis"  # 技术分析错误
    EXECUTION = "execution"              # 执行错误
    PLANNING = "planning"                # 计划错误
    POSITION_SIZING = "position_sizing"  # 仓位规模错误
    MARKET_ANALYSIS = "market_analysis"  # 市场分析错误
    EMOTIONAL = "emotional"              # 情绪错误
    COGNITIVE = "cognitive"              # 认知错误


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    LOW = "low"           # 低风险，轻微影响
    MEDIUM = "medium"     # 中等风险，可修复影响
    HIGH = "high"         # 高风险，重大影响
    CRITICAL = "critical" # 关键风险，可能导致账户毁灭


class ErrorFrequency(Enum):
    """错误频率枚举"""
    RARE = "rare"          # 罕见（<5%的交易）
    OCCASIONAL = "occasional"  # 偶尔（5-15%的交易）
    FREQUENT = "frequent"  # 频繁（15-30%的交易）
    CHRONIC = "chronic"    # 慢性（>30%的交易）


@dataclass
class TradingError:
    """交易错误数据结构"""
    error_id: str
    error_name: str
    category: ErrorCategory
    severity: ErrorSeverity
    frequency: ErrorFrequency
    description: str
    root_cause: str
    typical_symptoms: List[str]
    common_triggers: List[str]
    prevention_strategies: List[str]
    correction_actions: List[str]
    learning_questions: List[str]
    detection_rules: List[Dict]  # 检测规则
    risk_score: float            # 风险评分 (0-100)
    impact_score: float          # 影响评分 (0-100)
    difficulty_to_fix: float     # 修复难度 (0-1)
    related_errors: List[str]    # 相关错误ID
    tags: List[str]
    examples: List[Dict]         # 实际案例示例
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典（可序列化）"""
        data = asdict(self)
        # 处理枚举类型
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        data['frequency'] = self.frequency.value
        # 处理日期时间
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingError':
        """从字典创建实例"""
        # 转换枚举类型
        data['category'] = ErrorCategory(data['category'])
        data['severity'] = ErrorSeverity(data['severity'])
        data['frequency'] = ErrorFrequency(data['frequency'])
        # 转换日期时间
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ErrorOccurrence:
    """错误发生记录"""
    occurrence_id: str
    error_id: str
    timestamp: datetime
    context: Dict[str, Any]  # 发生上下文
    detected_by: List[str]   # 检测规则
    severity_at_occurrence: ErrorSeverity  # 发生时的严重程度
    impact_assessment: Dict[str, Any]      # 影响评估
    correction_applied: List[str]          # 应用的纠正措施
    learning_extracted: List[str]          # 提取的学习点
    prevented_future: bool = False         # 是否预防了未来发生
    recurrence_count: int = 0              # 复发次数
    resolved_at: Optional[datetime] = None
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """转换为字典（可序列化）"""
        data = asdict(self)
        data['severity_at_occurrence'] = self.severity_at_occurrence.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ErrorOccurrence':
        """从字典创建实例"""
        data['severity_at_occurrence'] = ErrorSeverity(data['severity_at_occurrence'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('resolved_at'):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        return cls(**data)


@dataclass
class ErrorPreventionPlan:
    """错误预防计划"""
    plan_id: str
    trader_id: str
    focus_errors: List[str]  # 重点关注错误
    prevention_strategies: List[str]
    implementation_steps: List[Dict]
    monitoring_metrics: List[Dict]
    review_schedule: Dict[str, Any]
    success_criteria: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    
    def to_dict(self) -> Dict:
        """转换为字典（可序列化）"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ErrorPreventionPlan':
        """从字典创建实例"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class CommonErrorsAvoidanceSystem:
    """常见错误与避免系统（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 storage_path: str = None,
                 enable_realtime_detection: bool = True,
                 enable_learning_feedback: bool = True):
        """初始化错误避免系统"""
        self.storage_path = storage_path
        self.enable_realtime_detection = enable_realtime_detection
        self.enable_learning_feedback = enable_learning_feedback
        
        # 错误知识库
        self.error_knowledge_base = {}  # error_id -> TradingError
        self.error_occurrences = {}     # occurrence_id -> ErrorOccurrence
        self.error_statistics = {}      # error_id -> 统计信息
        self.prevention_plans = {}      # plan_id -> ErrorPreventionPlan
        
        # 统计信息（必须在 _initialize_default_errors 之前）
        self.system_statistics = {
            'total_errors_defined': 0,
            'total_occurrences': 0,
            'prevented_errors': 0,
            'corrected_errors': 0,
            'average_detection_time_minutes': 0,
            'error_recurrence_rate': 0.0,
            'improvement_rate': 0.0
        }

        # 检测引擎
        self.detection_rules = self._initialize_detection_rules()
        self.detection_patterns = self._initialize_detection_patterns()

        # 学习系统
        self.learning_history = []
        self.improvement_tracking = {}

        # 实时监控
        self.realtime_monitors = {
            'psychological': self._monitor_psychological_errors,
            'risk_management': self._monitor_risk_management_errors,
            'execution': self._monitor_execution_errors,
            'discipline': self._monitor_discipline_errors
        }

        # 初始化默认错误库
        self._initialize_default_errors()
        
        # 如果提供了存储路径，尝试加载现有数据
        if storage_path:
            self._load_data_from_storage()
    
    def _initialize_default_errors(self) -> None:
        """初始化默认错误库"""
        # 心理错误
        psychological_errors = [
            TradingError(
                error_id="psych_001",
                error_name="过度自信",
                category=ErrorCategory.PSYCHOLOGICAL,
                severity=ErrorSeverity.HIGH,
                frequency=ErrorFrequency.FREQUENT,
                description="过高估计自己的交易能力，忽视风险信号",
                root_cause="成功交易后的自我膨胀，缺乏自我监控",
                typical_symptoms=[
                    "增加仓位规模超过计划",
                    "忽视止损信号",
                    "过度交易",
                    "轻视市场风险"
                ],
                common_triggers=[
                    "连续盈利后",
                    "大额盈利交易后",
                    "自我感觉良好时",
                    "市场看似简单时"
                ],
                prevention_strategies=[
                    "坚持交易计划，不随意更改",
                    "设置最大仓位限制",
                    "定期进行自我评估",
                    "记录和分析所有交易决策"
                ],
                correction_actions=[
                    "立即减少仓位规模",
                    "重新评估当前交易",
                    "暂停交易冷静思考",
                    "与交易伙伴讨论决策"
                ],
                learning_questions=[
                    "什么让我感到过度自信？",
                    "我忽略了哪些风险信号？",
                    "如何避免重复这种错误？"
                ],
                detection_rules=[
                    {"condition": "position_size > planned_size * 1.5", "severity": "high"},
                    {"condition": "ignore_stop_loss_signals > 2", "severity": "medium"},
                    {"condition": "trades_per_day > average * 2", "severity": "medium"}
                ],
                risk_score=85.0,
                impact_score=75.0,
                difficulty_to_fix=0.7,
                related_errors=["psych_002", "risk_003"],
                tags=["psychology", "overconfidence", "risk"],
                examples=[
                    {"description": "盈利后仓位加倍，导致重大亏损", "lesson": "坚持原定仓位规模"},
                    {"description": "忽视止损信号，相信市场会回转", "lesson": "尊重止损规则"}
                ]
            ),
            TradingError(
                error_id="psych_002",
                error_name="恐惧交易",
                category=ErrorCategory.PSYCHOLOGICAL,
                severity=ErrorSeverity.MEDIUM,
                frequency=ErrorFrequency.OCCASIONAL,
                description="因恐惧而避免交易或过早平仓",
                root_cause="前次亏损后的心理创伤，风险厌恶过度",
                typical_symptoms=[
                    "错过明显交易机会",
                    "过早平仓锁定微小利润",
                    "设置过紧止损",
                    "过度规避风险"
                ],
                common_triggers=[
                    "重大亏损后",
                    "市场波动性增加时",
                    "账户资金减少时",
                    "负面新闻影响"
                ],
                prevention_strategies=[
                    "建立风险承受能力评估",
                    "逐步重建信心的小额交易",
                    "设置最小盈利目标",
                    "心理训练和放松技巧"
                ],
                correction_actions=[
                    "分析恐惧来源",
                    "从小额交易开始重建信心",
                    "重新评估风险承受能力",
                    "寻求心理支持"
                ],
                learning_questions=[
                    "我真正害怕的是什么？",
                    "这种恐惧是否有事实依据？",
                    "如何逐步克服这种恐惧？"
                ],
                detection_rules=[
                    {"condition": "missed_opportunities > 3", "severity": "medium"},
                    {"condition": "early_exits > 2", "severity": "medium"},
                    {"condition": "tight_stops_ratio > 0.8", "severity": "low"}
                ],
                risk_score=60.0,
                impact_score=50.0,
                difficulty_to_fix=0.6,
                related_errors=["psych_001", "exec_003"],
                tags=["psychology", "fear", "risk_aversion"],
                examples=[
                    {"description": "因恐惧错过明显突破机会", "lesson": "相信交易系统信号"},
                    {"description": "过早平仓错过大趋势", "lesson": "让利润奔跑"}
                ]
            )
        ]
        
        # 纪律错误
        discipline_errors = [
            TradingError(
                error_id="disc_001",
                error_name="不设止损",
                category=ErrorCategory.DISCIPLINE,
                severity=ErrorSeverity.CRITICAL,
                frequency=ErrorFrequency.OCCASIONAL,
                description="交易时不设置止损订单",
                root_cause="侥幸心理，相信总能手动平仓",
                typical_symptoms=[
                    "手动管理止损",
                    "相信市场会回转",
                    "忽视风险控制",
                    "过度自信"
                ],
                common_triggers=[
                    "市场快速波动时",
                    "相信自己能更好判断",
                    "前次止损被触发后",
                    "情绪化交易时"
                ],
                prevention_strategies=[
                    "强制每笔交易必须设止损",
                    "设置系统自动止损",
                    "定期检查止损设置",
                    "将止损作为交易必要条件"
                ],
                correction_actions=[
                    "立即设置止损",
                    "重新评估当前风险",
                    "暂停新交易直到修复",
                    "分析不设止损的原因"
                ],
                learning_questions=[
                    "为什么不设止损？",
                    "不设止损的最大风险是什么？",
                    "如何确保每次交易都设止损？"
                ],
                detection_rules=[
                    {"condition": "stop_loss_not_set == True", "severity": "critical"},
                    {"condition": "manual_stop_management > 0", "severity": "high"}
                ],
                risk_score=95.0,
                impact_score=90.0,
                difficulty_to_fix=0.3,
                related_errors=["risk_001", "psych_001"],
                tags=["discipline", "stop_loss", "critical"],
                examples=[
                    {"description": "不设止损导致账户毁灭性亏损", "lesson": "止损是生存第一法则"},
                    {"description": "手动止损导致情绪化决策", "lesson": "自动止损避免情绪干扰"}
                ]
            ),
            TradingError(
                error_id="disc_002",
                error_name="过度交易",
                category=ErrorCategory.DISCIPLINE,
                severity=ErrorSeverity.HIGH,
                frequency=ErrorFrequency.FREQUENT,
                description="交易频率超过合理范围，追逐每个机会",
                root_cause="无聊、贪婪或试图弥补亏损",
                typical_symptoms=[
                    "日交易次数超标",
                    "交易质量下降",
                    "佣金成本增加",
                    "决策疲劳"
                ],
                common_triggers=[
                    "试图快速回本",
                    "市场波动增加时",
                    "无聊或寻找刺激",
                    "跟随他人交易"
                ],
                prevention_strategies=[
                    "设置每日交易上限",
                    "只交易高质量机会",
                    "等待多重确认信号",
                    "交易前强制等待时间"
                ],
                correction_actions=[
                    "暂停交易冷静",
                    "分析每笔交易的合理性",
                    "减少仓位规模",
                    "专注于质量而非数量"
                ],
                learning_questions=[
                    "为什么需要交易这么多？",
                    "哪些交易是不必要的？",
                    "如何提高交易选择标准？"
                ],
                detection_rules=[
                    {"condition": "trades_per_day > max_allowed", "severity": "high"},
                    {"condition": "win_rate < 40 and high_frequency", "severity": "high"},
                    {"condition": "commission_cost > profit", "severity": "medium"}
                ],
                risk_score=75.0,
                impact_score=65.0,
                difficulty_to_fix=0.5,
                related_errors=["psych_001", "exec_002"],
                tags=["discipline", "overtrading", "frequency"],
                examples=[
                    {"description": "日交易20次导致决策质量下降", "lesson": "质量优于数量"},
                    {"description": "过度交易耗尽账户资金", "lesson": "耐心等待最佳机会"}
                ]
            )
        ]
        
        # 风险管理错误
        risk_management_errors = [
            TradingError(
                error_id="risk_001",
                error_name="仓位过大",
                category=ErrorCategory.RISK_MANAGEMENT,
                severity=ErrorSeverity.CRITICAL,
                frequency=ErrorFrequency.OCCASIONAL,
                description="单个交易仓位超过风险承受能力",
                root_cause="贪婪、过度自信或试图快速盈利",
                typical_symptoms=[
                    "单笔风险超过账户2%",
                    "心理压力明显增加",
                    "影响其他交易决策",
                    "情绪化平仓"
                ],
                common_triggers=[
                    "试图快速回本",
                    "过度自信时",
                    "相信'确定性'机会",
                    "前次亏损后"
                ],
                prevention_strategies=[
                    "严格仓位规模规则",
                    "最大风险不超过账户1-2%",
                    "定期风险承受评估",
                    "使用仓位计算器"
                ],
                correction_actions=[
                    "立即减仓至安全水平",
                    "重新评估风险承受",
                    "暂停新交易",
                    "分析仓位过大原因"
                ],
                learning_questions=[
                    "为什么需要这么大仓位？",
                    "这种仓位的最大风险是什么？",
                    "如何避免重复这种错误？"
                ],
                detection_rules=[
                    {"condition": "position_size > max_allowed", "severity": "critical"},
                    {"condition": "risk_per_trade > account_2_percent", "severity": "critical"},
                    {"condition": "stress_level > high_threshold", "severity": "medium"}
                ],
                risk_score=90.0,
                impact_score=85.0,
                difficulty_to_fix=0.4,
                related_errors=["disc_001", "psych_001"],
                tags=["risk", "position_sizing", "critical"],
                examples=[
                    {"description": "单笔交易风险5%导致重大亏损", "lesson": "严格控制单笔风险"},
                    {"description": "大仓位导致心理压力决策错误", "lesson": "舒适仓位才能理性决策"}
                ]
            ),
            TradingError(
                error_id="risk_002",
                error_name="风险回报比不合理",
                category=ErrorCategory.RISK_MANAGEMENT,
                severity=ErrorSeverity.MEDIUM,
                frequency=ErrorFrequency.FREQUENT,
                description="风险回报比低于合理水平（通常<1:1.5）",
                root_cause="急于入场、缺乏耐心或过度乐观",
                typical_symptoms=[
                    "风险回报比低于1:1",
                    "止盈目标不现实",
                    "止损设置过宽",
                    "长期胜率要求过高"
                ],
                common_triggers=[
                    "害怕错过机会",
                    "缺乏耐心等待",
                    "过度乐观估计",
                    "市场快速波动时"
                ],
                prevention_strategies=[
                    "最小风险回报比要求（如1:1.5）",
                    "等待更好的入场点",
                    "现实评估盈利目标",
                    "使用风险回报计算器"
                ],
                correction_actions=[
                    "调整止盈止损比例",
                    "重新评估交易机会",
                    "等待更好风险回报",
                    "分析风险回报合理性"
                ],
                learning_questions=[
                    "这个风险回报比合理吗？",
                    "需要多高胜率才能盈利？",
                    "如何找到更好风险回报？"
                ],
                detection_rules=[
                    {"condition": "risk_reward_ratio < 1.0", "severity": "high"},
                    {"condition": "risk_reward_ratio < 1.5", "severity": "medium"},
                    {"condition": "required_win_rate > 70", "severity": "medium"}
                ],
                risk_score=65.0,
                impact_score=55.0,
                difficulty_to_fix=0.4,
                related_errors=["tech_002", "exec_001"],
                tags=["risk", "reward_ratio", "planning"],
                examples=[
                    {"description": "风险1%追求0.5%利润不合理", "lesson": "风险回报比至少1:1.5"},
                    {"description": "过宽止损导致实际风险过大", "lesson": "合理止损设置是关键"}
                ]
            )
        ]
        
        # 技术分析错误
        technical_errors = [
            TradingError(
                error_id="tech_001",
                error_name="确认信号不足",
                category=ErrorCategory.TECHNICAL_ANALYSIS,
                severity=ErrorSeverity.MEDIUM,
                frequency=ErrorFrequency.FREQUENT,
                description="基于单一信号或指标入场，缺乏多重确认",
                root_cause="急于入场、缺乏耐心或过度相信单一指标",
                typical_symptoms=[
                    "单一指标决策",
                    "忽略矛盾信号",
                    "入场过早",
                    "缺乏时间框架确认"
                ],
                common_triggers=[
                    "害怕错过机会",
                    "市场看似明确时",
                    "过度相信特定指标",
                    "缺乏系统交易方法"
                ],
                prevention_strategies=[
                    "要求多重确认信号",
                    "多时间框架分析",
                    "等待价格行为确认",
                    "建立系统入场规则"
                ],
                correction_actions=[
                    "等待额外确认",
                    "分析矛盾信号",
                    "调整入场时机",
                    "完善交易系统"
                ],
                learning_questions=[
                    "还需要哪些确认信号？",
                    "是否有矛盾信号？",
                    "如何改进确认流程？"
                ],
                detection_rules=[
                    {"condition": "confirmation_signals < 2", "severity": "medium"},
                    {"condition": "contradictory_signals_ignored > 0", "severity": "low"},
                    {"condition": "single_timeframe_decision", "severity": "low"}
                ],
                risk_score=60.0,
                impact_score=50.0,
                difficulty_to_fix=0.5,
                related_errors=["plan_001", "exec_001"],
                tags=["technical", "confirmation", "analysis"],
                examples=[
                    {"description": "仅凭RSI超卖入场导致继续下跌", "lesson": "需要多重确认信号"},
                    {"description": "忽略高时间框架阻力导致失败", "lesson": "多时间框架分析关键"}
                ]
            ),
            TradingError(
                error_id="tech_002",
                error_name="错误解读价格行为",
                category=ErrorCategory.TECHNICAL_ANALYSIS,
                severity=ErrorSeverity.HIGH,
                frequency=ErrorFrequency.OCCASIONAL,
                description="错误解读价格模式、结构或信号",
                root_cause="知识不足、经验缺乏或确认偏误",
                typical_symptoms=[
                    "模式识别错误",
                    "结构分析错误",
                    "信号误解",
                    "趋势判断错误"
                ],
                common_triggers=[
                    "复杂价格行为时",
                    "缺乏经验模式",
                    "确认偏误影响",
                    "市场状态变化时"
                ],
                prevention_strategies=[
                    "持续价格行为学习",
                    "使用检查清单",
                    "获取第二意见",
                    "记录和分析错误解读"
                ],
                correction_actions=[
                    "重新分析价格行为",
                    "学习正确解读",
                    "暂停相关模式交易",
                    "分析错误原因"
                ],
                learning_questions=[
                    "正确解读应该是什么？",
                    "为什么我会错误解读？",
                    "如何避免类似错误？"
                ],
                detection_rules=[
                    {"condition": "pattern_misidentification", "severity": "high"},
                    {"condition": "structure_misinterpretation", "severity": "medium"},
                    {"condition": "trend_misjudgment", "severity": "medium"}
                ],
                risk_score=70.0,
                impact_score=65.0,
                difficulty_to_fix=0.6,
                related_errors=["tech_001", "market_001"],
                tags=["technical", "price_action", "interpretation"],
                examples=[
                    {"description": "将整理形态误解为反转形态", "lesson": "深入学习价格模式"},
                    {"description": "错误判断趋势方向导致逆势", "lesson": "趋势分析需要多维度确认"}
                ]
            )
        ]
        
        # 执行错误
        execution_errors = [
            TradingError(
                error_id="exec_001",
                error_name="追涨杀跌",
                category=ErrorCategory.EXECUTION,
                severity=ErrorSeverity.MEDIUM,
                frequency=ErrorFrequency.FREQUENT,
                description="在价格快速波动时追单，通常买在高点卖在低点",
                root_cause="害怕错过、情绪化决策或缺乏耐心",
                typical_symptoms=[
                    "突破后追单",
                    "价格快速波动时入场",
                    "缺乏回调等待",
                    "情绪化执行"
                ],
                common_triggers=[
                    "害怕错过突破",
                    "价格快速上涨/下跌",
                    "情绪激动时",
                    "缺乏入场计划"
                ],
                prevention_strategies=[
                    "等待回调入场",
                    "设置合理入场点",
                    "避免追单规则",
                    "情绪冷静检查"
                ],
                correction_actions=[
                    "停止追单行为",
                    "等待更好入场",
                    "分析追单原因",
                    "完善入场策略"
                ],
                learning_questions=[
                    "为什么需要追单？",
                    "等待回调的风险是什么？",
                    "如何避免情绪化执行？"
                ],
                detection_rules=[
                    {"condition": "chasing_entry == True", "severity": "medium"},
                    {"condition": "entry_during_spike", "severity": "medium"},
                    {"condition": "emotional_entry", "severity": "low"}
                ],
                risk_score=65.0,
                impact_score=60.0,
                difficulty_to_fix=0.5,
                related_errors=["psych_001", "disc_002"],
                tags=["execution", "chasing", "emotional"],
                examples=[
                    {"description": "追涨买在高点立即回调", "lesson": "等待回调更好风险回报"},
                    {"description": "杀跌卖在低点错失反弹", "lesson": "避免情绪化执行"}
                ]
            ),
            TradingError(
                error_id="exec_002",
                error_name="过早平仓",
                category=ErrorCategory.EXECUTION,
                severity=ErrorSeverity.MEDIUM,
                frequency=ErrorFrequency.FREQUENT,
                description="在利润目标达到前过早平仓，害怕利润回吐",
                root_cause="恐惧、缺乏信心或过度监控",
                typical_symptoms=[
                    "微小盈利即平仓",
                    "害怕利润回吐",
                    "过度监控仓位",
                    "缺乏让利润奔跑"
                ],
                common_triggers=[
                    "前次利润回吐后",
                    "市场波动增加时",
                    "缺乏信心时",
                    "过度监控时"
                ],
                prevention_strategies=[
                    "设置明确止盈目标",
                    "分批平仓策略",
                    "减少仓位监控频率",
                    "信任交易系统"
                ],
                correction_actions=[
                    "分析过早平仓原因",
                    "重新评估止盈策略",
                    "实施分批平仓",
                    "建立信心训练"
                ],
                learning_questions=[
                    "为什么害怕利润回吐？",
                    "合理止盈目标是什么？",
                    "如何建立让利润奔跑的信心？"
                ],
                detection_rules=[
                    {"condition": "early_exit_before_target", "severity": "medium"},
                    {"condition": "profit_taken < 50_percent_target", "severity": "medium"},
                    {"condition": "fear_of_giving_back", "severity": "low"}
                ],
                risk_score=55.0,
                impact_score=50.0,
                difficulty_to_fix=0.6,
                related_errors=["psych_002", "risk_002"],
                tags=["execution", "early_exit", "profit"],
                examples=[
                    {"description": "微小盈利平仓错过大趋势", "lesson": "让利润奔跑是关键"},
                    {"description": "害怕回吐错过合理止盈", "lesson": "信任止盈目标"}
                ]
            )
        ]
        
        # 将所有错误添加到知识库
        all_errors = psychological_errors + discipline_errors + risk_management_errors + technical_errors + execution_errors
        
        for error in all_errors:
            self.error_knowledge_base[error.error_id] = error
            self.error_statistics[error.error_id] = {
                'occurrence_count': 0,
                'prevention_count': 0,
                'correction_count': 0,
                'average_severity': 0.0,
                'recurrence_rate': 0.0,
                'last_occurrence': None,
                'improvement_trend': 'stable'
            }
        
        self.system_statistics['total_errors_defined'] = len(all_errors)
    
    def _initialize_detection_rules(self) -> Dict:
        """初始化检测规则"""
        return {
            'psychological': [
                {
                    'rule_id': 'psych_001_detection',
                    'error_id': 'psych_001',
                    'conditions': ['confidence_level > 0.8', 'risk_taking_increased'],
                    'severity': 'high',
                    'confidence_threshold': 0.7
                },
                {
                    'rule_id': 'psych_002_detection',
                    'error_id': 'psych_002',
                    'conditions': ['fear_level > 0.6', 'missed_opportunities > 2'],
                    'severity': 'medium',
                    'confidence_threshold': 0.6
                }
            ],
            'discipline': [
                {
                    'rule_id': 'disc_001_detection',
                    'error_id': 'disc_001',
                    'conditions': ['stop_loss_not_set == True'],
                    'severity': 'critical',
                    'confidence_threshold': 0.9
                },
                {
                    'rule_id': 'disc_002_detection',
                    'error_id': 'disc_002',
                    'conditions': ['trades_today > daily_limit', 'trade_quality_declining'],
                    'severity': 'high',
                    'confidence_threshold': 0.8
                }
            ],
            'risk_management': [
                {
                    'rule_id': 'risk_001_detection',
                    'error_id': 'risk_001',
                    'conditions': ['position_size > max_allowed', 'risk_per_trade > 2_percent'],
                    'severity': 'critical',
                    'confidence_threshold': 0.85
                },
                {
                    'rule_id': 'risk_002_detection',
                    'error_id': 'risk_002',
                    'conditions': ['risk_reward_ratio < 1.0', 'required_win_rate > 70'],
                    'severity': 'medium',
                    'confidence_threshold': 0.7
                }
            ],
            'technical_analysis': [
                {
                    'rule_id': 'tech_001_detection',
                    'error_id': 'tech_001',
                    'conditions': ['confirmation_signals < 2', 'single_indicator_decision'],
                    'severity': 'medium',
                    'confidence_threshold': 0.6
                }
            ],
            'execution': [
                {
                    'rule_id': 'exec_001_detection',
                    'error_id': 'exec_001',
                    'conditions': ['chasing_entry == True', 'entry_during_price_spike'],
                    'severity': 'medium',
                    'confidence_threshold': 0.7
                }
            ]
        }
    
    def _initialize_detection_patterns(self) -> Dict:
        """初始化检测模式"""
        return {
            'overtrading_pattern': {
                'description': '过度交易模式检测',
                'indicators': ['trades_per_day', 'trade_frequency', 'decision_fatigue'],
                'thresholds': {'trades_per_day': 10, 'trade_frequency_minutes': 30},
                'error_ids': ['disc_002', 'psych_001']
            },
            'risk_escalation_pattern': {
                'description': '风险升级模式检测',
                'indicators': ['position_size_trend', 'risk_per_trade_trend', 'confidence_trend'],
                'thresholds': {'position_size_increase_percent': 50, 'consecutive_increases': 3},
                'error_ids': ['risk_001', 'psych_001']
            },
            'emotional_trading_pattern': {
                'description': '情绪化交易模式检测',
                'indicators': ['emotional_state', 'decision_speed', 'deviation_from_plan'],
                'thresholds': {'emotional_intensity': 0.7, 'plan_deviation_percent': 30},
                'error_ids': ['psych_001', 'psych_002', 'exec_001']
            },
            'discipline_breakdown_pattern': {
                'description': '纪律崩溃模式检测',
                'indicators': ['rule_violations', 'plan_adherence', 'consistency_score'],
                'thresholds': {'rule_violations_per_day': 3, 'adherence_below_percent': 70},
                'error_ids': ['disc_001', 'disc_002', 'risk_001']
            }
        }
    
    def _load_data_from_storage(self) -> None:
        """从存储加载数据"""
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 加载错误知识库
                    if 'error_knowledge_base' in data:
                        for error_data in data['error_knowledge_base'].values():
                            error = TradingError.from_dict(error_data)
                            self.error_knowledge_base[error.error_id] = error
                    
                    # 加载错误发生记录
                    if 'error_occurrences' in data:
                        for occ_data in data['error_occurrences'].values():
                            occurrence = ErrorOccurrence.from_dict(occ_data)
                            self.error_occurrences[occurrence.occurrence_id] = occurrence
                    
                    # 加载预防计划
                    if 'prevention_plans' in data:
                        for plan_data in data['prevention_plans'].values():
                            plan = ErrorPreventionPlan.from_dict(plan_data)
                            self.prevention_plans[plan.plan_id] = plan
                    
                    # 加载统计信息
                    if 'error_statistics' in data:
                        self.error_statistics = data['error_statistics']
                    
                    # 加载系统统计
                    if 'system_statistics' in data:
                        self.system_statistics.update(data['system_statistics'])
                    
                    print(f"已从 {self.storage_path} 加载错误避免系统数据")
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            # 保持默认初始化状态
    
    def _save_data_to_storage(self) -> None:
        """保存数据到存储"""
        try:
            import os
            
            # 准备保存数据
            save_data = {
                'error_knowledge_base': {
                    error_id: error.to_dict() 
                    for error_id, error in self.error_knowledge_base.items()
                },
                'error_occurrences': {
                    occ_id: occ.to_dict()
                    for occ_id, occ in self.error_occurrences.items()
                },
                'prevention_plans': {
                    plan_id: plan.to_dict()
                    for plan_id, plan in self.prevention_plans.items()
                },
                'error_statistics': self.error_statistics,
                'system_statistics': self.system_statistics,
                'saved_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存数据失败: {str(e)}")
    
    def detect_errors(self, trading_context: Dict) -> List[Dict]:
        """检测交易上下文中的潜在错误"""
        if not self.enable_realtime_detection:
            return []
        
        detected_errors = []
        
        # 应用检测规则
        for category, rules in self.detection_rules.items():
            for rule in rules:
                error_id = rule['error_id']
                
                if error_id not in self.error_knowledge_base:
                    continue
                
                # 检查条件（简化实现）
                conditions_met = self._check_rule_conditions(rule['conditions'], trading_context)
                
                if conditions_met:
                    error = self.error_knowledge_base[error_id]
                    
                    detection_result = {
                        'error_id': error_id,
                        'error_name': error.error_name,
                        'category': error.category.value,
                        'severity': error.severity.value,
                        'detected_by': rule['rule_id'],
                        'confidence': rule.get('confidence_threshold', 0.7),
                        'context': trading_context,
                        'timestamp': datetime.now(),
                        'recommended_actions': error.correction_actions[:3],
                        'prevention_strategies': error.prevention_strategies[:3]
                    }
                    
                    detected_errors.append(detection_result)
        
        # 检测模式
        for pattern_name, pattern_info in self.detection_patterns.items():
            pattern_detected = self._detect_pattern(pattern_name, trading_context)
            
            if pattern_detected:
                for error_id in pattern_info['error_ids']:
                    if error_id in self.error_knowledge_base:
                        error = self.error_knowledge_base[error_id]
                        
                        detection_result = {
                            'error_id': error_id,
                            'error_name': error.error_name,
                            'category': error.category.value,
                            'severity': error.severity.value,
                            'detected_by': f'pattern_{pattern_name}',
                            'confidence': 0.6,
                            'context': trading_context,
                            'timestamp': datetime.now(),
                            'pattern': pattern_name,
                            'pattern_description': pattern_info['description'],
                            'recommended_actions': error.correction_actions[:2],
                            'prevention_strategies': error.prevention_strategies[:2]
                        }
                        
                        detected_errors.append(detection_result)
        
        # 实时监控
        for monitor_name, monitor_func in self.realtime_monitors.items():
            monitor_results = monitor_func(trading_context)
            detected_errors.extend(monitor_results)
        
        # 去重和排序（按严重程度）
        if detected_errors:
            # 简单去重
            unique_errors = {}
            for error in detected_errors:
                key = f"{error['error_id']}_{error.get('detected_by', '')}"
                if key not in unique_errors:
                    unique_errors[key] = error
                else:
                    # 保留更高置信度
                    if error.get('confidence', 0) > unique_errors[key].get('confidence', 0):
                        unique_errors[key] = error
            
            detected_errors = list(unique_errors.values())
            
            # 按严重程度排序
            severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            detected_errors.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 0), reverse=True)
        
        return detected_errors
    
    def _check_rule_conditions(self, conditions: List[str], context: Dict) -> bool:
        """检查规则条件是否满足（简化实现）"""
        # 在实际系统中，这里应该有一个完整的条件引擎
        # 这里使用简化实现，检查上下文中的关键指标
        
        for condition in conditions:
            if '>' in condition:
                parts = condition.split('>')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    
                    if key in context and isinstance(context[key], (int, float)):
                        if not context[key] > value:
                            return False
                    else:
                        # 如果上下文没有这个键，假设条件不满足
                        return False
            
            elif '==' in condition:
                parts = condition.split('==')
                if len(parts) == 2:
                    key = parts[0].strip()
                    expected_value = parts[1].strip().lower()
                    
                    if key in context:
                        actual_value = str(context[key]).lower()
                        if actual_value != expected_value:
                            return False
                    else:
                        return False
        
        return True
    
    def _detect_pattern(self, pattern_name: str, context: Dict) -> bool:
        """检测特定模式"""
        pattern_info = self.detection_patterns.get(pattern_name)
        if not pattern_info:
            return False
        
        indicators = pattern_info.get('indicators', [])
        thresholds = pattern_info.get('thresholds', {})
        
        # 简化实现：检查关键指标是否超过阈值
        for indicator, threshold in thresholds.items():
            if indicator in context:
                if isinstance(context[indicator], (int, float)) and isinstance(threshold, (int, float)):
                    if context[indicator] > threshold:
                        return True
        
        return False
    
    def _monitor_psychological_errors(self, context: Dict) -> List[Dict]:
        """监控心理错误"""
        detected = []
        
        # 检查过度自信
        if context.get('confidence_level', 0) > 0.8:
            detected.append({
                'error_id': 'psych_001',
                'detected_by': 'psych_monitor',
                'confidence': 0.7,
                'metric': 'confidence_level',
                'value': context['confidence_level'],
                'threshold': 0.8
            })
        
        # 检查恐惧交易
        if context.get('fear_level', 0) > 0.6 and context.get('missed_opportunities', 0) > 2:
            detected.append({
                'error_id': 'psych_002',
                'detected_by': 'psych_monitor',
                'confidence': 0.65,
                'metrics': ['fear_level', 'missed_opportunities'],
                'values': [context.get('fear_level', 0), context.get('missed_opportunities', 0)],
                'thresholds': [0.6, 2]
            })
        
        return detected
    
    def _monitor_risk_management_errors(self, context: Dict) -> List[Dict]:
        """监控风险管理错误"""
        detected = []
        
        # 检查仓位过大
        position_size = context.get('position_size', 0)
        max_allowed = context.get('max_position_size', 1.0)
        account_risk = context.get('risk_per_trade_percent', 0)
        
        if position_size > max_allowed * 1.2:
            detected.append({
                'error_id': 'risk_001',
                'detected_by': 'risk_monitor',
                'confidence': 0.8,
                'metric': 'position_size_ratio',
                'value': position_size / max_allowed,
                'threshold': 1.2
            })
        
        if account_risk > 2.0:  # 超过2%账户风险
            detected.append({
                'error_id': 'risk_001',
                'detected_by': 'risk_monitor',
                'confidence': 0.85,
                'metric': 'risk_per_trade_percent',
                'value': account_risk,
                'threshold': 2.0
            })
        
        return detected
    
    def _monitor_execution_errors(self, context: Dict) -> List[Dict]:
        """监控执行错误"""
        detected = []
        
        # 检查追涨杀跌
        if context.get('chasing_entry', False):
            detected.append({
                'error_id': 'exec_001',
                'detected_by': 'execution_monitor',
                'confidence': 0.7,
                'condition': 'chasing_entry',
                'value': True
            })
        
        # 检查过早平仓
        if context.get('early_exit_ratio', 0) > 0.5:  # 50%以上交易过早平仓
            detected.append({
                'error_id': 'exec_002',
                'detected_by': 'execution_monitor',
                'confidence': 0.6,
                'metric': 'early_exit_ratio',
                'value': context['early_exit_ratio'],
                'threshold': 0.5
            })
        
        return detected
    
    def _monitor_discipline_errors(self, context: Dict) -> List[Dict]:
        """监控纪律错误"""
        detected = []
        
        # 检查不设止损
        if context.get('stop_loss_not_set', False):
            detected.append({
                'error_id': 'disc_001',
                'detected_by': 'discipline_monitor',
                'confidence': 0.9,
                'condition': 'stop_loss_not_set',
                'value': True
            })
        
        # 检查过度交易
        trades_today = context.get('trades_today', 0)
        daily_limit = context.get('daily_trade_limit', 5)
        
        if trades_today > daily_limit:
            detected.append({
                'error_id': 'disc_002',
                'detected_by': 'discipline_monitor',
                'confidence': 0.75,
                'metric': 'trades_today',
                'value': trades_today,
                'threshold': daily_limit
            })
        
        return detected
    
    def record_error_occurrence(self, 
                               error_id: str, 
                               context: Dict, 
                               severity: ErrorSeverity = None,
                               notes: str = "") -> Dict:
        """记录错误发生"""
        if error_id not in self.error_knowledge_base:
            return {
                'success': False,
                'error': f'未知错误ID: {error_id}',
                'valid_error_ids': list(self.error_knowledge_base.keys())
            }
        
        try:
            error = self.error_knowledge_base[error_id]
            
            # 生成发生ID
            occurrence_id = f"occ_{len(self.error_occurrences) + 1:06d}"
            
            # 确定严重程度
            if severity is None:
                severity = error.severity
            
            # 检测规则
            detected_by = []
            detection_results = self.detect_errors(context)
            for detection in detection_results:
                if detection['error_id'] == error_id:
                    detected_by.append(detection['detected_by'])
            
            # 创建发生记录
            occurrence = ErrorOccurrence(
                occurrence_id=occurrence_id,
                error_id=error_id,
                timestamp=datetime.now(),
                context=context,
                detected_by=detected_by,
                severity_at_occurrence=severity,
                impact_assessment=self._assess_impact(error, context),
                correction_applied=[],
                learning_extracted=[],
                notes=notes
            )
            
            # 添加到记录
            self.error_occurrences[occurrence_id] = occurrence
            
            # 更新统计
            self._update_error_statistics(error_id, occurrence)
            
            # 保存数据
            if self.storage_path:
                self._save_data_to_storage()
            
            return {
                'success': True,
                'occurrence_id': occurrence_id,
                'error_name': error.error_name,
                'severity': severity.value,
                'detected_by': detected_by,
                'prevention_suggestions': error.prevention_strategies[:3],
                'correction_suggestions': error.correction_actions[:3]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'记录错误发生失败: {str(e)}',
                'error_id': error_id
            }
    
    def _assess_impact(self, error: TradingError, context: Dict) -> Dict:
        """评估错误影响"""
        impact = {
            'financial_impact': 0.0,
            'psychological_impact': 0.0,
            'time_impact': 0.0,
            'learning_opportunity': 0.0,
            'overall_score': 0.0
        }
        
        # 简化实现：基于错误严重程度和上下文评估
        severity_scores = {
            ErrorSeverity.LOW: 0.2,
            ErrorSeverity.MEDIUM: 0.5,
            ErrorSeverity.HIGH: 0.8,
            ErrorSeverity.CRITICAL: 1.0
        }
        
        base_score = severity_scores.get(error.severity, 0.5)
        
        # 考虑上下文因素
        financial_factor = context.get('financial_impact_factor', 0.5)
        psychological_factor = context.get('psychological_impact_factor', 0.5)
        time_factor = context.get('time_impact_factor', 0.5)
        
        impact['financial_impact'] = base_score * financial_factor
        impact['psychological_impact'] = base_score * psychological_factor
        impact['time_impact'] = base_score * time_factor
        impact['learning_opportunity'] = min(1.0, base_score * 1.2)  # 错误越大学习机会越大
        
        # 总体评分
        weights = {'financial': 0.4, 'psychological': 0.3, 'time': 0.2, 'learning': 0.1}
        impact['overall_score'] = (
            impact['financial_impact'] * weights['financial'] +
            impact['psychological_impact'] * weights['psychological'] +
            impact['time_impact'] * weights['time'] +
            impact['learning_opportunity'] * weights['learning']
        )
        
        return impact
    
    def _update_error_statistics(self, error_id: str, occurrence: ErrorOccurrence) -> None:
        """更新错误统计"""
        if error_id not in self.error_statistics:
            self.error_statistics[error_id] = {
                'occurrence_count': 0,
                'prevention_count': 0,
                'correction_count': 0,
                'average_severity': 0.0,
                'recurrence_rate': 0.0,
                'last_occurrence': None,
                'improvement_trend': 'stable'
            }
        
        stats = self.error_statistics[error_id]
        
        # 更新发生次数
        stats['occurrence_count'] += 1
        
        # 更新平均严重程度
        severity_score = {
            ErrorSeverity.LOW: 1.0,
            ErrorSeverity.MEDIUM: 2.0,
            ErrorSeverity.HIGH: 3.0,
            ErrorSeverity.CRITICAL: 4.0
        }.get(occurrence.severity_at_occurrence, 2.0)
        
        old_avg = stats['average_severity']
        old_count = stats['occurrence_count'] - 1
        stats['average_severity'] = (old_avg * old_count + severity_score) / stats['occurrence_count']
        
        # 更新最后发生时间
        stats['last_occurrence'] = occurrence.timestamp.isoformat()
        
        # 更新系统统计
        self.system_statistics['total_occurrences'] += 1
        
        # 计算复发率（简化）
        if stats['occurrence_count'] > 1:
            # 假设复发率基于发生频率
            stats['recurrence_rate'] = min(1.0, stats['occurrence_count'] / 10)
        
        # 更新系统统计中的改进率（简化）
        prevented_rate = self.system_statistics.get('prevented_errors', 0) / max(1, self.system_statistics['total_occurrences'])
        self.system_statistics['improvement_rate'] = prevented_rate * 100
    
    def apply_correction(self, occurrence_id: str, correction_actions: List[str]) -> Dict:
        """应用纠正措施"""
        if occurrence_id not in self.error_occurrences:
            return {
                'success': False,
                'error': f'未知发生记录ID: {occurrence_id}'
            }
        
        try:
            occurrence = self.error_occurrences[occurrence_id]
            error_id = occurrence.error_id
            
            # 更新发生记录
            occurrence.correction_applied = correction_actions
            occurrence.resolved_at = datetime.now()
            
            # 更新错误统计
            if error_id in self.error_statistics:
                self.error_statistics[error_id]['correction_count'] += 1
            
            # 更新系统统计
            self.system_statistics['corrected_errors'] += 1
            
            # 提取学习点
            learning_points = self._extract_learning_points(occurrence)
            occurrence.learning_extracted = learning_points
            
            # 添加到学习历史
            self.learning_history.append({
                'occurrence_id': occurrence_id,
                'error_id': error_id,
                'timestamp': datetime.now(),
                'correction_actions': correction_actions,
                'learning_points': learning_points,
                'impact': occurrence.impact_assessment
            })
            
            # 保存数据
            if self.storage_path:
                self._save_data_to_storage()
            
            return {
                'success': True,
                'occurrence_id': occurrence_id,
                'correction_applied': correction_actions,
                'learning_points': learning_points,
                'resolved_at': occurrence.resolved_at.isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'应用纠正措施失败: {str(e)}',
                'occurrence_id': occurrence_id
            }
    
    def _extract_learning_points(self, occurrence: ErrorOccurrence) -> List[str]:
        """从错误发生中提取学习点"""
        error_id = occurrence.error_id
        
        if error_id not in self.error_knowledge_base:
            return ["未知错误类型，需要进一步分析"]
        
        error = self.error_knowledge_base[error_id]
        learning_points = []
        
        # 使用错误定义的学习问题
        for question in error.learning_questions:
            # 将问题转化为学习点（简化）
            if "什么让我" in question:
                learning_points.append(f"分析{error.error_name}的原因：{occurrence.notes[:50]}...")
            elif "如何避免" in question:
                learning_points.append(f"制定{error.error_name}预防策略：{error.prevention_strategies[0]}")
            elif "忽略了哪些" in question:
                learning_points.append(f"识别被忽略的信号：检查交易上下文")
            else:
                learning_points.append(f"从{error.error_name}中学习：{question}")
        
        # 添加上下文特定学习点
        context = occurrence.context
        if 'market_condition' in context:
            learning_points.append(f"在{context['market_condition']}市场条件下特别警惕{error.error_name}")
        
        if 'emotional_state' in context:
            learning_points.append(f"情绪状态{context['emotional_state']}时容易犯{error.error_name}")
        
        # 限制数量
        return learning_points[:5]
    
    def create_prevention_plan(self, 
                              trader_profile: Dict,
                              focus_error_ids: List[str] = None) -> Dict:
        """创建错误预防计划"""
        try:
            # 确定重点关注错误
            if not focus_error_ids:
                # 基于统计确定最常见或最严重错误
                focus_error_ids = self._identify_focus_errors(trader_profile)
            
            # 验证错误ID
            valid_error_ids = []
            for error_id in focus_error_ids:
                if error_id in self.error_knowledge_base:
                    valid_error_ids.append(error_id)
                else:
                    print(f"警告: 忽略未知错误ID: {error_id}")
            
            if not valid_error_ids:
                return {
                    'success': False,
                    'error': '没有有效的错误ID用于创建预防计划',
                    'available_error_ids': list(self.error_knowledge_base.keys())
                }
            
            # 生成计划ID
            plan_id = f"plan_{len(self.prevention_plans) + 1:06d}"
            
            # 收集预防策略
            prevention_strategies = []
            for error_id in valid_error_ids:
                error = self.error_knowledge_base[error_id]
                prevention_strategies.extend(error.prevention_strategies[:2])  # 每个错误取前2个策略
            
            # 去重
            prevention_strategies = list(set(prevention_strategies))
            
            # 创建实施步骤
            implementation_steps = [
                {
                    'step': 1,
                    'action': '错误意识培训',
                    'description': '学习识别重点关注错误',
                    'duration_days': 7,
                    'resources': [f"错误文档: {error_id}" for error_id in valid_error_ids]
                },
                {
                    'step': 2,
                    'action': '预防策略实施',
                    'description': '实施预防策略到交易流程',
                    'duration_days': 14,
                    'strategies': prevention_strategies[:3]
                },
                {
                    'step': 3,
                    'action': '监控和反馈',
                    'description': '监控错误发生，调整预防措施',
                    'duration_days': 30,
                    'metrics': ['occurrence_count', 'prevention_rate', 'improvement_trend']
                }
            ]
            
            # 监控指标
            monitoring_metrics = [
                {
                    'metric': '错误发生频率',
                    'target': '减少50%',
                    'measurement': 'weekly',
                    'threshold': 0.5
                },
                {
                    'metric': '预防成功率',
                    'target': '达到80%',
                    'measurement': 'weekly',
                    'threshold': 0.8
                },
                {
                    'metric': '纠正措施效果',
                    'target': '提高30%',
                    'measurement': 'monthly',
                    'threshold': 0.3
                }
            ]
            
            # 审核计划
            review_schedule = {
                'daily': ['错误检测记录', '预防策略执行'],
                'weekly': ['发生频率分析', '策略效果评估'],
                'monthly': ['计划调整', '学习总结']
            }
            
            # 成功标准
            success_criteria = [
                f'减少{len(valid_error_ids)}个重点错误的总体发生率30%',
                '提高预防策略执行率至90%',
                '建立持续学习和改进机制',
                trader_profile.get('success_criteria', '完成所有实施步骤')
            ]
            
            # 创建计划
            plan = ErrorPreventionPlan(
                plan_id=plan_id,
                trader_id=trader_profile.get('trader_id', 'default'),
                focus_errors=valid_error_ids,
                prevention_strategies=prevention_strategies,
                implementation_steps=implementation_steps,
                monitoring_metrics=monitoring_metrics,
                review_schedule=review_schedule,
                success_criteria=success_criteria
            )
            
            # 保存计划
            self.prevention_plans[plan_id] = plan
            
            # 保存数据
            if self.storage_path:
                self._save_data_to_storage()
            
            return {
                'success': True,
                'plan_id': plan_id,
                'focus_errors': valid_error_ids,
                'error_names': [self.error_knowledge_base[eid].error_name for eid in valid_error_ids],
                'prevention_strategies_count': len(prevention_strategies),
                'implementation_steps': len(implementation_steps),
                'estimated_duration_days': sum(step['duration_days'] for step in implementation_steps),
                'message': f'预防计划 {plan_id} 已创建'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'创建预防计划失败: {str(e)}',
                'trader_profile': trader_profile
            }
    
    def _identify_focus_errors(self, trader_profile: Dict) -> List[str]:
        """识别需要重点关注的错误"""
        # 基于多种因素确定重点关注错误
        
        focus_errors = []
        
        # 1. 基于统计：最常见错误
        if self.error_statistics:
            # 按发生次数排序
            sorted_by_occurrence = sorted(
                self.error_statistics.items(),
                key=lambda x: x[1].get('occurrence_count', 0),
                reverse=True
            )
            
            top_errors = [error_id for error_id, _ in sorted_by_occurrence[:3]]
            focus_errors.extend(top_errors)
        
        # 2. 基于严重程度：最严重错误
        critical_errors = []
        for error_id, error in self.error_knowledge_base.items():
            if error.severity == ErrorSeverity.CRITICAL:
                critical_errors.append(error_id)
        
        focus_errors.extend(critical_errors[:2])
        
        # 3. 基于交易者特点
        trader_weaknesses = trader_profile.get('weak_areas', [])
        trader_strengths = trader_profile.get('strong_areas', [])
        
        for error_id, error in self.error_knowledge_base.items():
            # 基于弱点匹配
            for weakness in trader_weaknesses:
                if weakness.lower() in error.description.lower() or \
                   weakness.lower() in ' '.join(error.tags).lower():
                    if error_id not in focus_errors:
                        focus_errors.append(error_id)
            
            # 避免在强项上过度关注
            for strength in trader_strengths:
                if strength.lower() in error.description.lower() and \
                   error_id in focus_errors and \
                   error.severity != ErrorSeverity.CRITICAL:
                    focus_errors.remove(error_id)
        
        # 去重和限制数量
        focus_errors = list(set(focus_errors))
        
        # 如果还是太多，基于严重程度和频率选择
        if len(focus_errors) > 5:
            scored_errors = []
            for error_id in focus_errors:
                error = self.error_knowledge_base[error_id]
                stats = self.error_statistics.get(error_id, {})
                
                # 评分：严重程度 * 频率 * 影响
                severity_score = {
                    ErrorSeverity.LOW: 1.0,
                    ErrorSeverity.MEDIUM: 2.0,
                    ErrorSeverity.HIGH: 3.0,
                    ErrorSeverity.CRITICAL: 4.0
                }.get(error.severity, 2.0)
                
                frequency_score = min(5.0, stats.get('occurrence_count', 1))
                impact_score = error.impact_score / 100.0
                
                total_score = severity_score * frequency_score * (1 + impact_score)
                
                scored_errors.append((error_id, total_score))
            
            # 按分数排序
            scored_errors.sort(key=lambda x: x[1], reverse=True)
            focus_errors = [error_id for error_id, _ in scored_errors[:5]]
        
        return focus_errors
    
    def get_error_analysis(self, error_id: str = None) -> Dict:
        """获取错误分析报告"""
        if error_id and error_id not in self.error_knowledge_base:
            return {
                'success': False,
                'error': f'未知错误ID: {error_id}',
                'available_error_ids': list(self.error_knowledge_base.keys())
            }
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'system_statistics': self.system_statistics,
            'total_errors_defined': len(self.error_knowledge_base),
            'total_occurrences': len(self.error_occurrences),
            'error_categories': defaultdict(int),
            'severity_distribution': defaultdict(int),
            'top_errors_by_frequency': [],
            'top_errors_by_severity': [],
            'improvement_opportunities': [],
            'recommendations': []
        }
        
        # 按类别统计
        for error in self.error_knowledge_base.values():
            category = error.category.value
            analysis['error_categories'][category] += 1
            
            severity = error.severity.value
            analysis['severity_distribution'][severity] += 1
        
        # 按频率排序错误
        if self.error_statistics:
            sorted_by_frequency = sorted(
                [(eid, stats.get('occurrence_count', 0)) 
                 for eid, stats in self.error_statistics.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            for error_id, count in sorted_by_frequency[:5]:
                if error_id in self.error_knowledge_base:
                    error = self.error_knowledge_base[error_id]
                    analysis['top_errors_by_frequency'].append({
                        'error_id': error_id,
                        'error_name': error.error_name,
                        'occurrence_count': count,
                        'severity': error.severity.value,
                        'category': error.category.value
                    })
        
        # 按严重程度排序错误
        sorted_by_severity = sorted(
            self.error_knowledge_base.values(),
            key=lambda x: {
                ErrorSeverity.CRITICAL: 4,
                ErrorSeverity.HIGH: 3,
                ErrorSeverity.MEDIUM: 2,
                ErrorSeverity.LOW: 1
            }.get(x.severity, 2),
            reverse=True
        )
        
        for error in sorted_by_severity[:5]:
            stats = self.error_statistics.get(error.error_id, {})
            analysis['top_errors_by_severity'].append({
                'error_id': error.error_id,
                'error_name': error.error_name,
                'severity': error.severity.value,
                'risk_score': error.risk_score,
                'occurrence_count': stats.get('occurrence_count', 0),
                'prevention_rate': stats.get('prevention_count', 0) / max(1, stats.get('occurrence_count', 1))
            })
        
        # 改进机会
        for error_id, stats in self.error_statistics.items():
            if error_id in self.error_knowledge_base:
                error = self.error_knowledge_base[error_id]
                
                occurrence_count = stats.get('occurrence_count', 0)
                prevention_count = stats.get('prevention_count', 0)
                
                if occurrence_count > 0:
                    prevention_rate = prevention_count / occurrence_count
                    
                    if prevention_rate < 0.5 and occurrence_count >= 3:
                        analysis['improvement_opportunities'].append({
                            'error_id': error_id,
                            'error_name': error.error_name,
                            'occurrence_count': occurrence_count,
                            'prevention_rate': prevention_rate,
                            'suggestion': f'加强{error.error_name}的预防措施',
                            'priority': 'high' if error.severity == ErrorSeverity.CRITICAL else 'medium'
                        })
        
        # 总体建议
        total_occurrences = len(self.error_occurrences)
        prevented_errors = self.system_statistics.get('prevented_errors', 0)
        
        if total_occurrences > 0:
            prevention_rate = prevented_errors / total_occurrences
            
            if prevention_rate < 0.3:
                analysis['recommendations'].append({
                    'area': '总体预防',
                    'issue': '错误预防率偏低',
                    'rate': prevention_rate,
                    'suggestion': '加强实时检测和预防措施',
                    'priority': 'high'
                })
            elif prevention_rate > 0.7:
                analysis['recommendations'].append({
                    'area': '总体预防',
                    'issue': '预防效果良好',
                    'rate': prevention_rate,
                    'suggestion': '保持当前预防策略，关注新错误类型',
                    'priority': 'low'
                })
        
        # 特定错误分析
        if error_id:
            error = self.error_knowledge_base[error_id]
            stats = self.error_statistics.get(error_id, {})
            
            specific_analysis = {
                'error_details': {
                    'error_id': error.error_id,
                    'error_name': error.error_name,
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'description': error.description,
                    'root_cause': error.root_cause,
                    'risk_score': error.risk_score,
                    'impact_score': error.impact_score,
                    'difficulty_to_fix': error.difficulty_to_fix
                },
                'statistics': stats,
                'prevention_strategies': error.prevention_strategies,
                'correction_actions': error.correction_actions,
                'learning_questions': error.learning_questions,
                'recent_occurrences': []
            }
            
            # 最近发生记录
            recent_occurrences = []
            for occ in self.error_occurrences.values():
                if occ.error_id == error_id:
                    recent_occurrences.append({
                        'occurrence_id': occ.occurrence_id,
                        'timestamp': occ.timestamp.isoformat(),
                        'severity': occ.severity_at_occurrence.value,
                        'context_summary': {k: v for k, v in occ.context.items() if isinstance(v, (int, float, str))},
                        'corrected': len(occ.correction_applied) > 0
                    })
            
            # 按时间排序
            recent_occurrences.sort(key=lambda x: x['timestamp'], reverse=True)
            specific_analysis['recent_occurrences'] = recent_occurrences[:10]
            
            analysis['specific_error_analysis'] = specific_analysis
        
        return analysis
    
    def generate_learning_report(self, 
                                days_back: int = 30,
                                min_occurrences: int = 1) -> Dict:
        """生成学习报告"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # 收集相关发生记录
        recent_occurrences = []
        for occ in self.error_occurrences.values():
            if occ.timestamp >= cutoff_date:
                recent_occurrences.append(occ)
        
        if not recent_occurrences:
            return {
                'success': False,
                'error': f'过去{days_back}天内没有错误发生记录',
                'days_back': days_back
            }
        
        # 分析学习点
        learning_points = defaultdict(list)
        corrected_errors = 0
        prevented_errors = 0
        
        for occ in recent_occurrences:
            error_id = occ.error_id
            
            if error_id in self.error_knowledge_base:
                error = self.error_knowledge_base[error_id]
                
                # 收集学习点
                for question in error.learning_questions:
                    key = f"{error.error_name}: {question}"
                    learning_points[key].append({
                        'occurrence_id': occ.occurrence_id,
                        'timestamp': occ.timestamp,
                        'context': occ.context.get('summary', '无摘要')
                    })
                
                # 统计纠正和预防
                if occ.correction_applied:
                    corrected_errors += 1
                
                if occ.prevented_future:
                    prevented_errors += 1
        
        # 识别模式
        patterns = self._identify_learning_patterns(recent_occurrences)
        
        # 生成报告
        report = {
            'report_id': f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'period': {
                'start': cutoff_date.isoformat(),
                'end': datetime.now().isoformat(),
                'days': days_back
            },
            'summary': {
                'total_occurrences': len(recent_occurrences),
                'unique_errors': len(set(occ.error_id for occ in recent_occurrences)),
                'corrected_errors': corrected_errors,
                'prevented_errors': prevented_errors,
                'correction_rate': corrected_errors / max(1, len(recent_occurrences)),
                'prevention_rate': prevented_errors / max(1, len(recent_occurrences))
            },
            'key_learnings': [],
            'patterns_identified': patterns,
            'action_items': [],
            'next_steps': []
        }
        
        # 提取关键学习点（按频率）
        for learning_point, occurrences in sorted(learning_points.items(), 
                                                 key=lambda x: len(x[1]), 
                                                 reverse=True):
            if len(occurrences) >= min_occurrences:
                error_name = learning_point.split(':')[0]
                
                report['key_learnings'].append({
                    'learning_point': learning_point,
                    'frequency': len(occurrences),
                    'last_occurrence': max(occ['timestamp'] for occ in occurrences),
                    'related_error': error_name,
                    'suggested_action': f'重点解决{error_name}问题'
                })
        
        # 生成行动项
        if report['summary']['correction_rate'] < 0.5:
            report['action_items'].append({
                'action': '提高错误纠正率',
                'priority': 'high',
                'target': '达到70%纠正率',
                'steps': ['记录所有错误发生', '立即应用纠正措施', '跟踪纠正效果']
            })
        
        if patterns.get('time_patterns'):
            peak_time = patterns['time_patterns'][0]['time']
            report['action_items'].append({
                'action': f'加强{peak_time}时段的错误预防',
                'priority': 'medium',
                'target': f'减少{peak_time}时段错误发生率30%',
                'steps': [f'{peak_time}前进行预防性检查', f'{peak_time}时段降低交易频率']
            })
        
        # 下一步
        report['next_steps'] = [
            '实施行动项并监控效果',
            '定期更新学习报告',
            '将学习点整合到交易系统',
            '分享学习经验'
        ]
        
        return report
    
    def _identify_learning_patterns(self, occurrences: List[ErrorOccurrence]) -> Dict:
        """识别学习模式"""
        patterns = {
            'time_patterns': [],
            'error_clusters': [],
            'context_patterns': [],
            'recurrence_patterns': []
        }
        
        if not occurrences:
            return patterns
        
        # 分析时间模式
        hour_counts = defaultdict(int)
        weekday_counts = defaultdict(int)
        
        for occ in occurrences:
            hour = occ.timestamp.hour
            weekday = occ.timestamp.strftime('%A')
            
            hour_counts[hour] += 1
            weekday_counts[weekday] += 1
        
        # 识别高峰时段
        if hour_counts:
            max_hour = max(hour_counts.items(), key=lambda x: x[1])
            patterns['time_patterns'].append({
                'type': 'hourly_peak',
                'time': f'{max_hour[0]}:00',
                'occurrence_count': max_hour[1],
                'percentage': max_hour[1] / len(occurrences) * 100,
                'suggestion': f'加强{max_hour[0]}:00时段的错误预防'
            })
        
        # 识别周模式
        if weekday_counts:
            max_weekday = max(weekday_counts.items(), key=lambda x: x[1])
            patterns['time_patterns'].append({
                'type': 'weekly_peak',
                'time': max_weekday[0],
                'occurrence_count': max_weekday[1],
                'percentage': max_weekday[1] / len(occurrences) * 100,
                'suggestion': f'加强{max_weekday[0]}的交易纪律'
            })
        
        # 分析错误集群
        error_counts = defaultdict(int)
        for occ in occurrences:
            error_counts[occ.error_id] += 1
        
        # 识别常见错误集群
        common_errors = [eid for eid, count in error_counts.items() if count >= 3]
        
        for error_id in common_errors:
            if error_id in self.error_knowledge_base:
                error = self.error_knowledge_base[error_id]
                patterns['error_clusters'].append({
                    'error_id': error_id,
                    'error_name': error.error_name,
                    'occurrence_count': error_counts[error_id],
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'suggestion': f'重点解决{error.error_name}集群问题'
                })
        
        # 分析上下文模式（简化）
        context_keywords = defaultdict(int)
        for occ in occurrences:
            # 从备注中提取关键词
            if occ.notes:
                words = occ.notes.split()
                for word in words[:10]:  # 只取前10个词
                    if len(word) > 2:  # 忽略短词
                        context_keywords[word.lower()] += 1
            
            # 从上下文中提取
            for key, value in occ.context.items():
                if isinstance(value, str) and len(value) < 50:
                    context_keywords[f"{key}:{value}"] += 1
        
        # 常见上下文
        common_contexts = [item for item, count in context_keywords.items() if count >= 2]
        for context in common_contexts[:5]:  # 最多5个
            patterns['context_patterns'].append({
                'context': context,
                'occurrence_count': context_keywords[context],
                'suggestion': f'关注{context}相关的错误模式'
            })
        
        # 分析复发模式
        error_recurrence = defaultdict(list)
        for occ in occurrences:
            error_recurrence[occ.error_id].append(occ.timestamp)
        
        for error_id, timestamps in error_recurrence.items():
            if len(timestamps) > 1:
                # 计算平均复发间隔
                timestamps.sort()
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # 小时
                    intervals.append(interval)
                
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    
                    if error_id in self.error_knowledge_base:
                        error = self.error_knowledge_base[error_id]
                        patterns['recurrence_patterns'].append({
                            'error_id': error_id,
                            'error_name': error.error_name,
                            'recurrence_count': len(timestamps),
                            'avg_interval_hours': avg_interval,
                            'suggestion': f'{error.error_name}平均{avg_interval:.1f}小时复发一次，需要加强预防'
                        })
        
        return patterns
    
    def demo_system(self) -> Dict:
        """系统演示函数"""
        print("=" * 70)
        print("常见错误与避免系统演示")
        print("第31章：常见错误与避免 - AL Brooks《价格行为交易之区间篇》")
        print("=" * 70)
        
        results = {
            'system_initialization': {
                'total_errors_defined': self.system_statistics['total_errors_defined'],
                'error_categories': defaultdict(int),
                'severity_distribution': defaultdict(int)
            },
            'error_detection_demo': [],
            'prevention_plan_demo': None,
            'learning_report_demo': None
        }
        
        # 统计错误分类和严重程度
        for error in self.error_knowledge_base.values():
            category = error.category.value
            severity = error.severity.value
            
            results['system_initialization']['error_categories'][category] += 1
            results['system_initialization']['severity_distribution'][severity] += 1
        
        print(f"\n1. 系统初始化完成")
        print(f"   - 定义错误总数: {self.system_statistics['total_errors_defined']}")
        print(f"   - 错误类别分布: {dict(results['system_initialization']['error_categories'])}")
        print(f"   - 严重程度分布: {dict(results['system_initialization']['severity_distribution'])}")
        
        # 错误检测演示
        print(f"\n2. 错误检测演示")
        
        test_contexts = [
            {
                'name': '过度自信场景',
                'context': {
                    'confidence_level': 0.9,
                    'risk_taking_increased': True,
                    'position_size': 2.5,
                    'max_position_size': 1.0
                }
            },
            {
                'name': '纪律崩溃场景',
                'context': {
                    'stop_loss_not_set': True,
                    'trades_today': 12,
                    'daily_trade_limit': 5,
                    'rule_violations_per_day': 4
                }
            },
            {
                'name': '风险管理场景',
                'context': {
                    'risk_per_trade_percent': 3.0,
                    'risk_reward_ratio': 0.8,
                    'required_win_rate': 75,
                    'emotional_state': 'greedy'
                }
            }
        ]
        
        for test in test_contexts:
            print(f"\n   [{test['name']}]")
            detected_errors = self.detect_errors(test['context'])
            
            if detected_errors:
                print(f"   检测到 {len(detected_errors)} 个潜在错误:")
                for i, error in enumerate(detected_errors[:3], 1):  # 最多显示3个
                    print(f"   {i}. {error['error_name']} ({error['severity']}): {error.get('detected_by', '未知规则')}")
                
                results['error_detection_demo'].append({
                    'scenario': test['name'],
                    'detected_errors': len(detected_errors),
                    'errors': [{'name': e['error_name'], 'severity': e['severity']} for e in detected_errors[:3]]
                })
            else:
                print(f"   未检测到明显错误")
                results['error_detection_demo'].append({
                    'scenario': test['name'],
                    'detected_errors': 0,
                    'errors': []
                })
        
        # 预防计划演示
        print(f"\n3. 预防计划创建演示")
        
        trader_profile = {
            'trader_id': 'demo_trader',
            'experience_years': 1,
            'weak_areas': ['discipline', 'risk_management'],
            'strong_areas': ['technical_analysis'],
            'success_criteria': '减少关键错误发生率40%'
        }
        
        plan_result = self.create_prevention_plan(trader_profile)
        
        if plan_result['success']:
            print(f"   预防计划创建成功:")
            print(f"   - 计划ID: {plan_result['plan_id']}")
            print(f"   - 关注错误: {len(plan_result['focus_errors'])} 个")
            print(f"   - 错误名称: {', '.join(plan_result['error_names'][:3])}" + 
                  ("..." if len(plan_result['error_names']) > 3 else ""))
            print(f"   - 预防策略: {plan_result['prevention_strategies_count']} 条")
            print(f"   - 实施步骤: {plan_result['implementation_steps']} 步")
            print(f"   - 预计时长: {plan_result['estimated_duration_days']} 天")
            
            results['prevention_plan_demo'] = plan_result
        else:
            print(f"   预防计划创建失败: {plan_result.get('error', '未知错误')}")
        
        # 记录一些错误发生用于学习报告
        print(f"\n4. 记录测试错误发生")
        
        test_errors = [
            ('psych_001', {'confidence_level': 0.85, 'market': 'bullish'}, "测试过度自信"),
            ('disc_002', {'trades_today': 10, 'daily_limit': 5}, "测试过度交易"),
            ('risk_001', {'position_size': 2.0, 'max_allowed': 1.0}, "测试仓位过大"),
            ('psych_002', {'fear_level': 0.7, 'missed_opportunities': 3}, "测试恐惧交易"),
            ('disc_001', {'stop_loss_not_set': True}, "测试不设止损")
        ]
        
        recorded_errors = []
        for error_id, context, notes in test_errors:
            result = self.record_error_occurrence(error_id, context, notes=notes)
            if result['success']:
                recorded_errors.append(result['occurrence_id'])
                print(f"   记录成功: {result['error_name']} (ID: {result['occurrence_id']})")
        
        # 应用一些纠正措施
        if recorded_errors:
            print(f"\n5. 应用纠正措施演示")
            
            for occ_id in recorded_errors[:2]:  # 前2个
                correction_result = self.apply_correction(
                    occ_id, 
                    ["立即执行纠正措施", "分析错误原因", "调整交易策略"]
                )
                
                if correction_result['success']:
                    print(f"   纠正成功: {occ_id}")
                else:
                    print(f"   纠正失败: {correction_result.get('error', '未知错误')}")
        
        # 生成学习报告
        print(f"\n6. 学习报告生成演示")
        
        report_result = self.generate_learning_report(days_back=7, min_occurrences=1)
        
        if 'report_id' in report_result:
            print(f"   学习报告生成成功:")
            print(f"   - 报告ID: {report_result['report_id']}")
            print(f"   - 分析周期: {report_result['period']['days']} 天")
            print(f"   - 总发生数: {report_result['summary']['total_occurrences']}")
            print(f"   - 纠正率: {report_result['summary']['correction_rate']:.1%}")
            print(f"   - 关键学习点: {len(report_result['key_learnings'])} 个")
            print(f"   - 行动项: {len(report_result['action_items'])} 条")
            
            results['learning_report_demo'] = {
                'report_id': report_result['report_id'],
                'summary': report_result['summary'],
                'key_learnings_count': len(report_result['key_learnings']),
                'action_items_count': len(report_result['action_items'])
            }
        else:
            print(f"   学习报告生成失败: {report_result.get('error', '未知错误')}")
        
        print(f"\n" + "=" * 70)
        print("演示完成")
        print("=" * 70)
        
        return results


# ============================================================================
# 策略改造: 添加PriceActionRangesCommonErrorsAvoidanceSystemStrategy类
# 将价格行为区间常见错误与避免系统转换为交易策略
# ============================================================================

class PriceActionRangesCommonErrorsAvoidanceSystemStrategy(BaseStrategy):
    """价格行为区间常见错误与避免策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict = None):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 创建常见错误与避免系统实例
        self.errors_system = CommonErrorsAvoidanceSystem()
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于常见错误分析生成交易信号
        """
        # 获取学习报告
        learning_report = self.errors_system.generate_learning_report(days_back=7, min_occurrences=1)
        
        # 分析错误数据
        summary = learning_report.get('summary', {})
        total_occurrences = summary.get('total_occurrences', 0)
        correction_rate = summary.get('correction_rate', 0)
        
        # 获取关键学习点
        key_learnings = learning_report.get('key_learnings', [])
        
        # 根据错误分析生成信号
        if total_occurrences == 0:
            # 无错误记录，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif correction_rate >= 0.8:
            # 高纠正率，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif correction_rate <= 0.3 and total_occurrences >= 5:
            # 低纠正率且错误多，卖出信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='sell',
                price=self.data['close'].iloc[-1]
            )
        else:
            # 中等情况，hold信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        
        return self.signals