# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:13.632614

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实战案例分析量化系统
第30章：实战案例分析
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 案例数据管理：实际交易案例的收集、存储、索引和检索
2. 模式识别分析：从案例中识别关键价格行为模式
3. 经验提取系统：从成功/失败案例中提取可复用的交易经验
4. 教训总结机制：分析错误案例，提炼避免重复错误的方法
5. 案例分类系统：按市场条件、交易风格、结果等维度分类案例
6. 模拟学习引擎：基于历史案例的模拟交易和学习
7. 知识库构建：将案例分析结果构建成可查询的知识库
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import hashlib
import warnings
from dataclasses import dataclass, asdict, field
from enum import Enum
import re
import random
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy


class CaseType(Enum):
    """案例类型枚举"""
    SUCCESS = "success"           # 成功案例
    FAILURE = "failure"           # 失败案例
    BREAK_EVEN = "break_even"     # 盈亏平衡
    PARTIAL_SUCCESS = "partial_success"  # 部分成功
    LEARNING = "learning"         # 学习案例（非实际交易）


class MarketCondition(Enum):
    """市场条件枚举"""
    TRENDING_UP = "trending_up"           # 上升趋势
    TRENDING_DOWN = "trending_down"       # 下降趋势
    RANGING = "ranging"                   # 区间震荡
    BREAKOUT = "breakout"                 # 突破行情
    REVERSAL = "reversal"                 # 反转行情
    HIGH_VOLATILITY = "high_volatility"   # 高波动性
    LOW_VOLATILITY = "low_volatility"     # 低波动性


class PatternType(Enum):
    """价格模式类型枚举"""
    SUPPORT_RESISTANCE = "support_resistance"      # 支撑阻力
    TREND_LINE = "trend_line"                      # 趋势线
    CHANNEL = "channel"                            # 通道
    DOUBLE_TOP_BOTTOM = "double_top_bottom"        # 双顶双底
    HEAD_SHOULDERS = "head_shoulders"              # 头肩形态
    TRIANGLE = "triangle"                          # 三角形
    FLAG_PENNANT = "flag_pennant"                  # 旗形三角旗
    BREAKOUT_RETEST = "breakout_retest"            # 突破回测
    FAKE_OUT = "fake_out"                          # 假突破
    INSIDE_BAR = "inside_bar"                      # 内包线


@dataclass
class TradingCase:
    """交易案例数据结构"""
    case_id: str
    title: str
    case_type: CaseType
    market: str                    # 市场（forex, stocks, futures等）
    symbol: str                    # 交易品种
    timeframe: str                 # 时间框架
    entry_date: datetime           # 入场时间
    exit_date: datetime            # 出场时间
    entry_price: float             # 入场价格
    exit_price: float              # 出场价格
    position_size: float           # 仓位规模
    profit_loss: float             # 盈亏金额
    profit_loss_pct: float         # 盈亏百分比
    market_condition: MarketCondition  # 市场条件
    patterns_observed: List[PatternType]  # 观察到的模式
    entry_reason: str              # 入场理由
    exit_reason: str               # 出场理由
    key_decisions: List[str]       # 关键决策点
    mistakes_made: List[str]       # 犯的错误
    lessons_learned: List[str]     # 学到的教训
    success_factors: List[str]     # 成功因素
    technical_indicators: Dict[str, Any]  # 技术指标
    risk_management: Dict[str, Any]  # 风险管理信息
    psychological_state: Dict[str, Any]  # 心理状态
    tags: List[str]                # 标签
    difficulty_level: int          # 难度等级 (1-5)
    confidence_level: float        # 信心水平 (0-1)
    data_snapshot: Dict[str, Any]  # 数据快照
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典（可序列化）"""
        data = asdict(self)
        # 处理枚举类型
        data['case_type'] = self.case_type.value
        data['market_condition'] = self.market_condition.value
        data['patterns_observed'] = [p.value for p in self.patterns_observed]
        # 处理日期时间
        data['entry_date'] = self.entry_date.isoformat()
        data['exit_date'] = self.exit_date.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingCase':
        """从字典创建实例"""
        # 转换枚举类型
        data['case_type'] = CaseType(data['case_type'])
        data['market_condition'] = MarketCondition(data['market_condition'])
        data['patterns_observed'] = [PatternType(p) for p in data['patterns_observed']]
        # 转换日期时间
        data['entry_date'] = datetime.fromisoformat(data['entry_date'])
        data['exit_date'] = datetime.fromisoformat(data['exit_date'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class CaseStudyAnalyzer:
    """实战案例分析器（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 storage_path: str = None,
                 max_cases: int = 1000,
                 enable_indexing: bool = True):
        """初始化案例分析器"""
        self.storage_path = storage_path
        self.max_cases = max_cases
        self.enable_indexing = enable_indexing
        
        # 案例存储
        self.cases = {}  # case_id -> TradingCase
        self.case_counter = 0
        
        # 索引系统
        self.indices = {
            'by_type': {},          # 按案例类型索引
            'by_market': {},        # 按市场索引
            'by_symbol': {},        # 按品种索引
            'by_pattern': {},       # 按模式索引
            'by_market_condition': {},  # 按市场条件索引
            'by_difficulty': {},    # 按难度索引
            'by_outcome': {},       # 按结果索引
            'by_tag': {}            # 按标签索引
        }
        
        # 分析统计
        self.statistics = {
            'total_cases': 0,
            'success_cases': 0,
            'failure_cases': 0,
            'break_even_cases': 0,
            'total_profit_loss': 0.0,
            'average_profit_loss': 0.0,
            'win_rate': 0.0,
            'average_hold_time_hours': 0.0,
            'pattern_frequency': {},
            'market_condition_frequency': {},
            'common_mistakes': {},
            'key_success_factors': {}
        }
        
        # 模式识别器
        self.pattern_recognizers = {
            PatternType.SUPPORT_RESISTANCE: self._recognize_support_resistance,
            PatternType.TREND_LINE: self._recognize_trend_line,
            PatternType.CHANNEL: self._recognize_channel,
            PatternType.DOUBLE_TOP_BOTTOM: self._recognize_double_top_bottom,
            PatternType.HEAD_SHOULDERS: self._recognize_head_shoulders,
            PatternType.TRIANGLE: self._recognize_triangle,
            PatternType.FLAG_PENNANT: self._recognize_flag_pennant,
            PatternType.BREAKOUT_RETEST: self._recognize_breakout_retest,
            PatternType.FAKE_OUT: self._recognize_fake_out,
            PatternType.INSIDE_BAR: self._recognize_inside_bar
        }
        
        # 经验提取规则
        self.experience_extraction_rules = {
            'success_patterns': [
                self._extract_success_pattern_experience,
                self._extract_market_condition_experience,
                self._extract_entry_timing_experience
            ],
            'failure_patterns': [
                self._extract_failure_pattern_experience,
                self._extract_mistake_analysis_experience,
                self._extract_risk_management_experience
            ],
            'general_lessons': [
                self._extract_psychological_lessons,
                self._extract_discipline_lessons,
                self._extract_adaptation_lessons
            ]
        }
        
        # 如果提供了存储路径，尝试加载现有案例
        if storage_path:
            self._load_cases_from_storage()
    
    def _load_cases_from_storage(self) -> None:
        """从存储加载案例"""
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    cases_data = json.load(f)
                    for case_data in cases_data:
                        case = TradingCase.from_dict(case_data)
                        self._add_case_to_storage(case, update_indices=False)
                
                # 批量更新索引
                self._rebuild_indices()
                self._update_statistics()
                print(f"已从 {self.storage_path} 加载 {len(self.cases)} 个案例")
        except Exception as e:
            print(f"加载案例失败: {str(e)}")
            # 初始化空存储
            self.cases = {}
    
    def _save_cases_to_storage(self) -> None:
        """保存案例到存储"""
        try:
            import os
            cases_data = [case.to_dict() for case in self.cases.values()]
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(cases_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存案例失败: {str(e)}")
    
    def _add_case_to_storage(self, case: TradingCase, update_indices: bool = True) -> None:
        """添加案例到存储"""
        # 检查是否达到最大案例数
        if len(self.cases) >= self.max_cases:
            # 删除最旧的案例
            oldest_case_id = min(self.cases.keys(), 
                                key=lambda k: self.cases[k].created_at)
            del self.cases[oldest_case_id]
        
        # 添加新案例
        self.cases[case.case_id] = case
        
        # 更新索引
        if update_indices and self.enable_indexing:
            self._update_case_indices(case)
        
        # 更新统计
        self._update_statistics()
        
        # 保存到文件
        if self.storage_path:
            self._save_cases_to_storage()
    
    def _update_case_indices(self, case: TradingCase) -> None:
        """更新案例索引"""
        # 按案例类型索引
        case_type = case.case_type.value
        if case_type not in self.indices['by_type']:
            self.indices['by_type'][case_type] = []
        self.indices['by_type'][case_type].append(case.case_id)
        
        # 按市场索引
        market = case.market
        if market not in self.indices['by_market']:
            self.indices['by_market'][market] = []
        self.indices['by_market'][market].append(case.case_id)
        
        # 按品种索引
        symbol = case.symbol
        if symbol not in self.indices['by_symbol']:
            self.indices['by_symbol'][symbol] = []
        self.indices['by_symbol'][symbol].append(case.case_id)
        
        # 按模式索引
        for pattern in case.patterns_observed:
            pattern_str = pattern.value
            if pattern_str not in self.indices['by_pattern']:
                self.indices['by_pattern'][pattern_str] = []
            self.indices['by_pattern'][pattern_str].append(case.case_id)
        
        # 按市场条件索引
        market_condition = case.market_condition.value
        if market_condition not in self.indices['by_market_condition']:
            self.indices['by_market_condition'][market_condition] = []
        self.indices['by_market_condition'][market_condition].append(case.case_id)
        
        # 按难度索引
        difficulty = case.difficulty_level
        if difficulty not in self.indices['by_difficulty']:
            self.indices['by_difficulty'][difficulty] = []
        self.indices['by_difficulty'][difficulty].append(case.case_id)
        
        # 按结果索引（盈利/亏损）
        outcome = 'profit' if case.profit_loss > 0 else 'loss' if case.profit_loss < 0 else 'break_even'
        if outcome not in self.indices['by_outcome']:
            self.indices['by_outcome'][outcome] = []
        self.indices['by_outcome'][outcome].append(case.case_id)
        
        # 按标签索引
        for tag in case.tags:
            if tag not in self.indices['by_tag']:
                self.indices['by_tag'][tag] = []
            self.indices['by_tag'][tag].append(case.case_id)
    
    def _rebuild_indices(self) -> None:
        """重建所有索引"""
        # 清空索引
        for index_name in self.indices:
            self.indices[index_name] = {}
        
        # 重新构建索引
        for case in self.cases.values():
            self._update_case_indices(case)
    
    def _update_statistics(self) -> None:
        """更新统计信息"""
        if not self.cases:
            return
        
        total_cases = len(self.cases)
        success_cases = len([c for c in self.cases.values() if c.profit_loss > 0])
        failure_cases = len([c for c in self.cases.values() if c.profit_loss < 0])
        break_even_cases = len([c for c in self.cases.values() if c.profit_loss == 0])
        
        total_profit_loss = sum(c.profit_loss for c in self.cases.values())
        average_profit_loss = total_profit_loss / total_cases if total_cases > 0 else 0
        
        win_rate = success_cases / total_cases if total_cases > 0 else 0
        
        # 计算平均持仓时间
        total_hold_hours = 0
        for case in self.cases.values():
            hold_time = (case.exit_date - case.entry_date).total_seconds() / 3600
            total_hold_hours += hold_time
        average_hold_time_hours = total_hold_hours / total_cases if total_cases > 0 else 0
        
        # 计算模式频率
        pattern_frequency = {}
        for case in self.cases.values():
            for pattern in case.patterns_observed:
                pattern_str = pattern.value
                pattern_frequency[pattern_str] = pattern_frequency.get(pattern_str, 0) + 1
        
        # 计算市场条件频率
        market_condition_frequency = {}
        for case in self.cases.values():
            condition = case.market_condition.value
            market_condition_frequency[condition] = market_condition_frequency.get(condition, 0) + 1
        
        # 计算常见错误
        common_mistakes = {}
        for case in self.cases.values():
            for mistake in case.mistakes_made:
                common_mistakes[mistake] = common_mistakes.get(mistake, 0) + 1
        
        # 计算关键成功因素
        key_success_factors = {}
        for case in self.cases.values():
            for factor in case.success_factors:
                key_success_factors[factor] = key_success_factors.get(factor, 0) + 1
        
        self.statistics = {
            'total_cases': total_cases,
            'success_cases': success_cases,
            'failure_cases': failure_cases,
            'break_even_cases': break_even_cases,
            'total_profit_loss': total_profit_loss,
            'average_profit_loss': average_profit_loss,
            'win_rate': win_rate,
            'average_hold_time_hours': average_hold_time_hours,
            'pattern_frequency': pattern_frequency,
            'market_condition_frequency': market_condition_frequency,
            'common_mistakes': common_mistakes,
            'key_success_factors': key_success_factors
        }
    
    def add_case(self, case_data: Dict) -> Dict:
        """添加新案例"""
        try:
            # 生成案例ID
            self.case_counter += 1
            case_id = f"case_{self.case_counter:06d}"
            
            # 创建案例对象
            case = TradingCase(
                case_id=case_id,
                title=case_data.get('title', f"案例 {case_id}"),
                case_type=CaseType(case_data.get('case_type', 'success')),
                market=case_data.get('market', 'forex'),
                symbol=case_data.get('symbol', 'EUR/USD'),
                timeframe=case_data.get('timeframe', '1h'),
                entry_date=case_data.get('entry_date', datetime.now() - timedelta(hours=24)),
                exit_date=case_data.get('exit_date', datetime.now()),
                entry_price=case_data.get('entry_price', 0.0),
                exit_price=case_data.get('exit_price', 0.0),
                position_size=case_data.get('position_size', 1.0),
                profit_loss=case_data.get('profit_loss', 0.0),
                profit_loss_pct=case_data.get('profit_loss_pct', 0.0),
                market_condition=MarketCondition(case_data.get('market_condition', 'ranging')),
                patterns_observed=[PatternType(p) for p in case_data.get('patterns_observed', [])],
                entry_reason=case_data.get('entry_reason', ''),
                exit_reason=case_data.get('exit_reason', ''),
                key_decisions=case_data.get('key_decisions', []),
                mistakes_made=case_data.get('mistakes_made', []),
                lessons_learned=case_data.get('lessons_learned', []),
                success_factors=case_data.get('success_factors', []),
                technical_indicators=case_data.get('technical_indicators', {}),
                risk_management=case_data.get('risk_management', {}),
                psychological_state=case_data.get('psychological_state', {}),
                tags=case_data.get('tags', []),
                difficulty_level=case_data.get('difficulty_level', 3),
                confidence_level=case_data.get('confidence_level', 0.5),
                data_snapshot=case_data.get('data_snapshot', {})
            )
            
            # 计算盈亏百分比（如果未提供）
            if case.profit_loss_pct == 0 and case.entry_price > 0:
                case.profit_loss_pct = (case.exit_price - case.entry_price) / case.entry_price * 100
            
            # 计算盈亏金额（如果未提供）
            if case.profit_loss == 0 and case.position_size > 0:
                price_change = case.exit_price - case.entry_price
                case.profit_loss = price_change * case.position_size
            
            # 添加案例
            self._add_case_to_storage(case)
            
            return {
                'success': True,
                'case_id': case_id,
                'message': f'案例 {case_id} 已成功添加',
                'case_summary': {
                    'title': case.title,
                    'type': case.case_type.value,
                    'profit_loss': case.profit_loss,
                    'market_condition': case.market_condition.value
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'添加案例失败: {str(e)}',
                'case_data': case_data
            }
    
    def get_case(self, case_id: str) -> Optional[TradingCase]:
        """获取指定案例"""
        return self.cases.get(case_id)
    
    def search_cases(self, 
                    filters: Dict = None,
                    sort_by: str = 'created_at',
                    sort_desc: bool = True,
                    limit: int = 50) -> List[TradingCase]:
        """搜索案例"""
        if not self.cases:
            return []
        
        filters = filters or {}
        
        # 初始结果集
        result_ids = set(self.cases.keys())
        
        # 应用过滤器
        for filter_key, filter_value in filters.items():
            if filter_key == 'case_type':
                filtered_ids = self.indices['by_type'].get(filter_value, [])
                result_ids = result_ids.intersection(set(filtered_ids))
            
            elif filter_key == 'market':
                filtered_ids = self.indices['by_market'].get(filter_value, [])
                result_ids = result_ids.intersection(set(filtered_ids))
            
            elif filter_key == 'symbol':
                filtered_ids = self.indices['by_symbol'].get(filter_value, [])
                result_ids = result_ids.intersection(set(filtered_ids))
            
            elif filter_key == 'pattern':
                filtered_ids = self.indices['by_pattern'].get(filter_value, [])
                result_ids = result_ids.intersection(set(filtered_ids))
            
            elif filter_key == 'market_condition':
                filtered_ids = self.indices['by_market_condition'].get(filter_value, [])
                result_ids = result_ids.intersection(set(filtered_ids))
            
            elif filter_key == 'difficulty':
                filtered_ids = self.indices['by_difficulty'].get(filter_value, [])
                result_ids = result_ids.intersection(set(filtered_ids))
            
            elif filter_key == 'outcome':
                filtered_ids = self.indices['by_outcome'].get(filter_value, [])
                result_ids = result_ids.intersection(set(filtered_ids))
            
            elif filter_key == 'tag':
                filtered_ids = self.indices['by_tag'].get(filter_value, [])
                result_ids = result_ids.intersection(set(filtered_ids))
            
            elif filter_key == 'profit_loss_min':
                result_ids = {cid for cid in result_ids 
                             if self.cases[cid].profit_loss >= filter_value}
            
            elif filter_key == 'profit_loss_max':
                result_ids = {cid for cid in result_ids 
                             if self.cases[cid].profit_loss <= filter_value}
            
            elif filter_key == 'date_from':
                date_from = datetime.fromisoformat(filter_value) if isinstance(filter_value, str) else filter_value
                result_ids = {cid for cid in result_ids 
                             if self.cases[cid].entry_date >= date_from}
            
            elif filter_key == 'date_to':
                date_to = datetime.fromisoformat(filter_value) if isinstance(filter_value, str) else filter_value
                result_ids = {cid for cid in result_ids 
                             if self.cases[cid].entry_date <= date_to}
        
        # 获取案例对象
        results = [self.cases[cid] for cid in result_ids]
        
        # 排序
        if sort_by == 'created_at':
            results.sort(key=lambda x: x.created_at, reverse=sort_desc)
        elif sort_by == 'entry_date':
            results.sort(key=lambda x: x.entry_date, reverse=sort_desc)
        elif sort_by == 'profit_loss':
            results.sort(key=lambda x: x.profit_loss, reverse=sort_desc)
        elif sort_by == 'profit_loss_pct':
            results.sort(key=lambda x: x.profit_loss_pct, reverse=sort_desc)
        elif sort_by == 'difficulty_level':
            results.sort(key=lambda x: x.difficulty_level, reverse=sort_desc)
        
        # 限制结果数量
        if limit > 0:
            results = results[:limit]
        
        return results
    
    def analyze_patterns(self, case_ids: List[str] = None) -> Dict:
        """分析价格模式"""
        if case_ids is None:
            cases_to_analyze = list(self.cases.values())
        else:
            cases_to_analyze = [self.cases[cid] for cid in case_ids if cid in self.cases]
        
        if not cases_to_analyze:
            return {'success': False, 'error': '没有可分析的案例'}
        
        pattern_analysis = {
            'total_cases_analyzed': len(cases_to_analyze),
            'pattern_distribution': {},
            'pattern_success_rates': {},
            'pattern_profitability': {},
            'pattern_market_conditions': {},
            'pattern_hold_times': {},
            'recommended_patterns': []
        }
        
        # 统计模式分布
        for case in cases_to_analyze:
            for pattern in case.patterns_observed:
                pattern_str = pattern.value
                pattern_analysis['pattern_distribution'][pattern_str] = \
                    pattern_analysis['pattern_distribution'].get(pattern_str, 0) + 1
        
        # 计算模式成功率
        for pattern_str in pattern_analysis['pattern_distribution']:
            pattern_cases = [c for c in cases_to_analyze 
                           if any(p.value == pattern_str for p in c.patterns_observed)]
            
            if pattern_cases:
                success_cases = [c for c in pattern_cases if c.profit_loss > 0]
                success_rate = len(success_cases) / len(pattern_cases)
                avg_profit = np.mean([c.profit_loss_pct for c in pattern_cases]) if pattern_cases else 0
                
                pattern_analysis['pattern_success_rates'][pattern_str] = success_rate
                pattern_analysis['pattern_profitability'][pattern_str] = avg_profit
                
                # 收集市场条件
                market_conditions = {}
                for case in pattern_cases:
                    condition = case.market_condition.value
                    market_conditions[condition] = market_conditions.get(condition, 0) + 1
                pattern_analysis['pattern_market_conditions'][pattern_str] = market_conditions
                
                # 计算平均持仓时间
                avg_hold_time = np.mean([(c.exit_date - c.entry_date).total_seconds() / 3600 
                                       for c in pattern_cases])
                pattern_analysis['pattern_hold_times'][pattern_str] = avg_hold_time
        
        # 推荐模式（基于成功率和盈利能力）
        for pattern_str, success_rate in pattern_analysis['pattern_success_rates'].items():
            profitability = pattern_analysis['pattern_profitability'][pattern_str]
            frequency = pattern_analysis['pattern_distribution'][pattern_str]
            
            # 评分公式：成功率 × 盈利能力 × 频率权重
            score = success_rate * (1 + abs(profitability)/100) * (1 + np.log1p(frequency))
            
            pattern_analysis['recommended_patterns'].append({
                'pattern': pattern_str,
                'success_rate': success_rate,
                'profitability_pct': profitability,
                'frequency': frequency,
                'score': score,
                'best_market_conditions': max(pattern_analysis['pattern_market_conditions'][pattern_str].items(), 
                                            key=lambda x: x[1])[0] if pattern_analysis['pattern_market_conditions'][pattern_str] else 'unknown'
            })
        
        # 按分数排序推荐模式
        pattern_analysis['recommended_patterns'].sort(key=lambda x: x['score'], reverse=True)
        
        return pattern_analysis
    
    def extract_experiences(self, case_ids: List[str] = None) -> Dict:
        """提取交易经验"""
        if case_ids is None:
            cases_to_analyze = list(self.cases.values())
        else:
            cases_to_analyze = [self.cases[cid] for cid in case_ids if cid in self.cases]
        
        if not cases_to_analyze:
            return {'success': False, 'error': '没有可分析的案例'}
        
        experiences = {
            'total_cases_analyzed': len(cases_to_analyze),
            'success_cases': len([c for c in cases_to_analyze if c.profit_loss > 0]),
            'failure_cases': len([c for c in cases_to_analyze if c.profit_loss < 0]),
            'success_experiences': [],
            'failure_lessons': [],
            'common_mistakes': [],
            'key_decisions': [],
            'market_insights': [],
            'psychological_insights': []
        }
        
        # 提取成功经验
        success_cases = [c for c in cases_to_analyze if c.profit_loss > 0]
        for case in success_cases:
            for factor in case.success_factors:
                experiences['success_experiences'].append({
                    'case_id': case.case_id,
                    'title': case.title,
                    'success_factor': factor,
                    'profit_loss_pct': case.profit_loss_pct,
                    'market_condition': case.market_condition.value,
                    'patterns': [p.value for p in case.patterns_observed]
                })
        
        # 提取失败教训
        failure_cases = [c for c in cases_to_analyze if c.profit_loss < 0]
        for case in failure_cases:
            for mistake in case.mistakes_made:
                experiences['failure_lessons'].append({
                    'case_id': case.case_id,
                    'title': case.title,
                    'mistake': mistake,
                    'profit_loss_pct': case.profit_loss_pct,
                    'market_condition': case.market_condition.value,
                    'patterns': [p.value for p in case.patterns_observed]
                })
        
        # 提取常见错误
        mistake_counter = {}
        for case in cases_to_analyze:
            for mistake in case.mistakes_made:
                mistake_counter[mistake] = mistake_counter.get(mistake, 0) + 1
        
        for mistake, count in sorted(mistake_counter.items(), key=lambda x: x[1], reverse=True):
            experiences['common_mistakes'].append({
                'mistake': mistake,
                'frequency': count,
                'percentage': count / len(cases_to_analyze) * 100
            })
        
        # 提取关键决策
        decision_counter = {}
        for case in cases_to_analyze:
            for decision in case.key_decisions:
                decision_counter[decision] = decision_counter.get(decision, 0) + 1
        
        for decision, count in sorted(decision_counter.items(), key=lambda x: x[1], reverse=True):
            experiences['key_decisions'].append({
                'decision': decision,
                'frequency': count,
                'percentage': count / len(cases_to_analyze) * 100
            })
        
        # 提取市场洞察
        market_insights = {}
        for case in cases_to_analyze:
            condition = case.market_condition.value
            if condition not in market_insights:
                market_insights[condition] = {
                    'total_cases': 0,
                    'success_cases': 0,
                    'total_profit_loss_pct': 0,
                    'common_patterns': {}
                }
            
            market_insights[condition]['total_cases'] += 1
            if case.profit_loss > 0:
                market_insights[condition]['success_cases'] += 1
            market_insights[condition]['total_profit_loss_pct'] += case.profit_loss_pct
            
            for pattern in case.patterns_observed:
                pattern_str = pattern.value
                market_insights[condition]['common_patterns'][pattern_str] = \
                    market_insights[condition]['common_patterns'].get(pattern_str, 0) + 1
        
        for condition, data in market_insights.items():
            success_rate = data['success_cases'] / data['total_cases'] * 100 if data['total_cases'] > 0 else 0
            avg_profit = data['total_profit_loss_pct'] / data['total_cases'] if data['total_cases'] > 0 else 0
            
            # 找出最常见的模式
            common_patterns = sorted(data['common_patterns'].items(), key=lambda x: x[1], reverse=True)[:3]
            
            experiences['market_insights'].append({
                'market_condition': condition,
                'total_cases': data['total_cases'],
                'success_rate_pct': success_rate,
                'average_profit_pct': avg_profit,
                'common_patterns': [p[0] for p in common_patterns]
            })
        
        # 提取心理洞察
        psychological_insights = {}
        for case in cases_to_analyze:
            for state, value in case.psychological_state.items():
                if state not in psychological_insights:
                    psychological_insights[state] = {
                        'total_cases': 0,
                        'success_cases': 0,
                        'total_profit_loss_pct': 0,
                        'values': []
                    }
                
                psychological_insights[state]['total_cases'] += 1
                if case.profit_loss > 0:
                    psychological_insights[state]['success_cases'] += 1
                psychological_insights[state]['total_profit_loss_pct'] += case.profit_loss_pct
                psychological_insights[state]['values'].append(value)
        
        for state, data in psychological_insights.items():
            success_rate = data['success_cases'] / data['total_cases'] * 100 if data['total_cases'] > 0 else 0
            avg_profit = data['total_profit_loss_pct'] / data['total_cases'] if data['total_cases'] > 0 else 0
            
            # 只计算数值类型的平均值
            numeric_values = []
            for value in data['values']:
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
            
            avg_value = np.mean(numeric_values) if numeric_values else 0
            
            experiences['psychological_insights'].append({
                'psychological_state': state,
                'total_cases': data['total_cases'],
                'success_rate_pct': success_rate,
                'average_profit_pct': avg_profit,
                'average_state_value': avg_value,
                'correlation_with_success': 'positive' if success_rate > 50 else 'negative' if success_rate < 50 else 'neutral'
            })
        
        return experiences
    
    def generate_learning_recommendations(self, trader_profile: Dict = None) -> Dict:
        """生成学习推荐"""
        profile = trader_profile or {
            'experience_level': 'intermediate',
            'trading_style': 'swing',
            'weak_areas': ['risk_management', 'patience'],
            'strong_areas': ['technical_analysis', 'pattern_recognition'],
            'recent_performance': 'average'
        }
        
        recommendations = {
            'based_on_profile': profile,
            'recommended_cases': [],
            'skill_development_path': [],
            'risk_management_focus': [],
            'psychological_training': []
        }
        
        # 基于薄弱领域推荐案例
        weak_areas = profile.get('weak_areas', [])
        for area in weak_areas:
            # 搜索相关案例
            related_cases = self.search_cases(
                filters={'tag': area} if area in ['risk_management', 'patience'] else {},
                limit=5
            )
            
            for case in related_cases:
                recommendations['recommended_cases'].append({
                    'case_id': case.case_id,
                    'title': case.title,
                    'reason': f'针对薄弱领域: {area}',
                    'key_learning': case.lessons_learned[0] if case.lessons_learned else '无特定教训',
                    'difficulty': case.difficulty_level
                })
        
        # 技能发展路径
        experience_level = profile.get('experience_level', 'intermediate')
        if experience_level == 'beginner':
            recommendations['skill_development_path'] = [
                {'skill': '基础价格模式识别', 'priority': 'high', 'estimated_hours': 10},
                {'skill': '简单风险管理', 'priority': 'high', 'estimated_hours': 8},
                {'skill': '基础交易心理', 'priority': 'medium', 'estimated_hours': 6}
            ]
        elif experience_level == 'intermediate':
            recommendations['skill_development_path'] = [
                {'skill': '高级价格模式组合', 'priority': 'high', 'estimated_hours': 15},
                {'skill': '动态仓位管理', 'priority': 'high', 'estimated_hours': 12},
                {'skill': '市场条件适应性', 'priority': 'medium', 'estimated_hours': 10}
            ]
        else:  # advanced
            recommendations['skill_development_path'] = [
                {'skill': '复杂市场结构分析', 'priority': 'high', 'estimated_hours': 20},
                {'skill': '系统性风险管理', 'priority': 'high', 'estimated_hours': 18},
                {'skill': '高级交易心理控制', 'priority': 'medium', 'estimated_hours': 15}
            ]
        
        # 风险管理重点
        risk_analysis = self.analyze_patterns()
        if risk_analysis.get('success'):
            for pattern in risk_analysis.get('recommended_patterns', [])[:3]:
                recommendations['risk_management_focus'].append({
                    'pattern': pattern['pattern'],
                    'success_rate': pattern['success_rate'],
                    'recommended_action': f'在{pattern["best_market_conditions"]}条件下重点关注此模式',
                    'risk_adjustment': '降低仓位规模20%' if pattern['success_rate'] < 0.6 else '正常风险'
                })
        
        # 心理训练建议
        psychological_analysis = self.extract_experiences()
        psychological_insights = psychological_analysis.get('psychological_insights', [])
        
        for insight in psychological_insights[:3]:
            if insight['correlation_with_success'] == 'negative':
                recommendations['psychological_training'].append({
                    'area': insight['psychological_state'],
                    'issue': f'此心理状态与较低成功率相关 ({insight["success_rate_pct"]:.1f}%)',
                    'recommendation': '进行专门的心理训练改善此状态',
                    'training_method': '正念冥想、情绪日记、认知重构'
                })
        
        return recommendations
    
    def create_simulation_scenario(self, 
                                  scenario_type: str = 'learning',
                                  difficulty: int = 3) -> Dict:
        """创建模拟学习场景"""
        # 基于案例库创建模拟场景
        relevant_cases = self.search_cases(
            filters={'difficulty': difficulty},
            limit=10
        )
        
        if not relevant_cases:
            # 如果没有找到相关案例，创建默认场景
            return self._create_default_simulation_scenario(difficulty)
        
        # 随机选择一个案例作为基础
        base_case = random.choice(relevant_cases)
        
        # 创建模拟场景
        scenario = {
            'scenario_id': f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'scenario_type': scenario_type,
            'difficulty': difficulty,
            'base_case': base_case.case_id,
            'market': base_case.market,
            'symbol': base_case.symbol,
            'timeframe': base_case.timeframe,
            'initial_conditions': {
                'market_condition': base_case.market_condition.value,
                'observed_patterns': [p.value for p in base_case.patterns_observed],
                'technical_indicators': base_case.technical_indicators,
                'initial_price': base_case.entry_price
            },
            'learning_objectives': [
                f'识别{base_case.market_condition.value}市场中的价格模式',
                f'练习在{", ".join([p.value for p in base_case.patterns_observed])}模式下的交易决策',
                f'管理{base_case.difficulty_level}级难度的交易情境'
            ],
            'key_decisions_required': base_case.key_decisions.copy(),
            'common_mistakes_to_avoid': base_case.mistakes_made.copy(),
            'success_criteria': [
                f'利润率超过{abs(base_case.profit_loss_pct):.1f}%',
                f'正确识别所有价格模式',
                f'避免重复{base_case.mistakes_made[0] if base_case.mistakes_made else "常见错误"}'
            ],
            'available_data': {
                'price_history': base_case.data_snapshot.get('price_history', {}),
                'indicator_data': base_case.data_snapshot.get('indicator_data', {}),
                'market_context': base_case.data_snapshot.get('market_context', {})
            },
            'hints_available': [
                '查看相似案例的成功因素',
                '分析失败案例的常见错误',
                '参考市场条件的最佳实践'
            ],
            'estimated_duration_minutes': 30 + difficulty * 10
        }
        
        return scenario
    
    def _create_default_simulation_scenario(self, difficulty: int) -> Dict:
        """创建默认模拟场景"""
        market_conditions = ['trending_up', 'trending_down', 'ranging', 'breakout']
        patterns = ['support_resistance', 'trend_line', 'channel', 'double_top_bottom']
        
        scenario = {
            'scenario_id': f"sim_default_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'scenario_type': 'learning',
            'difficulty': difficulty,
            'market': 'forex',
            'symbol': 'EUR/USD',
            'timeframe': '1h',
            'initial_conditions': {
                'market_condition': random.choice(market_conditions),
                'observed_patterns': random.sample(patterns, min(2, len(patterns))),
                'technical_indicators': {
                    'rsi': 45 + random.randint(-15, 15),
                    'macd': {'histogram': random.uniform(-0.001, 0.001)},
                    'bollinger_bands': {'width': random.uniform(0.5, 2.0)}
                },
                'initial_price': 1.0850 + random.uniform(-0.005, 0.005)
            },
            'learning_objectives': [
                '识别关键价格模式',
                '练习风险管理决策',
                '管理交易心理'
            ],
            'key_decisions_required': [
                '入场时机选择',
                '止损设置',
                '止盈目标确定',
                '仓位规模计算'
            ],
            'common_mistakes_to_avoid': [
                '过度交易',
                '不设止损',
                '情绪化决策',
                '仓位过大'
            ],
            'success_criteria': [
                f'实现正利润率',
                '正确执行风险管理',
                '保持交易纪律'
            ],
            'available_data': {
                'price_history': {'length_bars': 100},
                'indicator_data': {'available_indicators': ['rsi', 'macd', 'bollinger_bands']},
                'market_context': {'news_events': [], 'economic_data': []}
            },
            'hints_available': [
                '分析市场结构',
                '考虑风险回报比',
                '评估市场情绪'
            ],
            'estimated_duration_minutes': 30 + difficulty * 10
        }
        
        return scenario
    
    # ========== 模式识别方法 ==========
    
    def _recognize_support_resistance(self, price_data: Dict) -> Dict:
        """识别支撑阻力"""
        # 简化实现：检测价格水平聚集区域
        prices = price_data.get('prices', [])
        if len(prices) < 20:
            return {'detected': False, 'confidence': 0.0}
        
        # 计算价格水平（简化）
        price_levels = {}
        for price in prices[-20:]:
            level = round(price, 4)  # 4位小数精度
            price_levels[level] = price_levels.get(level, 0) + 1
        
        # 找出出现次数最多的价格水平
        significant_levels = [(level, count) for level, count in price_levels.items() if count >= 3]
        
        if significant_levels:
            main_level = max(significant_levels, key=lambda x: x[1])
            current_price = prices[-1]
            distance_pct = abs(current_price - main_level[0]) / current_price * 100
            
            return {
                'detected': True,
                'confidence': min(0.9, main_level[1] / 10),  # 基于出现次数
                'level': main_level[0],
                'strength': main_level[1],
                'distance_pct': distance_pct,
                'type': 'support' if current_price > main_level[0] else 'resistance'
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_trend_line(self, price_data: Dict) -> Dict:
        """识别趋势线"""
        # 简化实现：检测价格高点或低点的线性趋势
        highs = price_data.get('highs', [])
        lows = price_data.get('lows', [])
        
        if len(highs) < 10 or len(lows) < 10:
            return {'detected': False, 'confidence': 0.0}
        
        # 简单趋势检测
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        high_trend = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
        low_trend = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
        
        # 确定趋势方向
        if abs(high_trend) > 0.0005 or abs(low_trend) > 0.0005:
            direction = 'up' if high_trend > 0 and low_trend > 0 else \
                       'down' if high_trend < 0 and low_trend < 0 else 'sideways'
            
            confidence = min(0.8, (abs(high_trend) + abs(low_trend)) * 1000)
            
            return {
                'detected': True,
                'confidence': confidence,
                'direction': direction,
                'slope': (high_trend + low_trend) / 2,
                'strength': 'strong' if confidence > 0.6 else 'moderate' if confidence > 0.3 else 'weak'
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_channel(self, price_data: Dict) -> Dict:
        """识别通道"""
        # 简化实现：检测平行的支撑阻力线
        trend_result = self._recognize_trend_line(price_data)
        
        if not trend_result['detected'] or trend_result['direction'] == 'sideways':
            return {'detected': False, 'confidence': 0.0}
        
        highs = price_data.get('highs', [])
        lows = price_data.get('lows', [])
        
        if len(highs) < 15 or len(lows) < 15:
            return {'detected': False, 'confidence': 0.0}
        
        # 检查高点和高点、低点和低点之间的平行性
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        
        high_variance = np.var([h - l for h, l in zip(recent_highs, recent_lows[-15:])])
        
        if high_variance < 0.0001:  # 方差较小表示平行性好
            return {
                'detected': True,
                'confidence': trend_result['confidence'] * 0.8,
                'direction': trend_result['direction'],
                'width': np.mean([h - l for h, l in zip(recent_highs, recent_lows)]),
                'parallel_quality': 'good' if high_variance < 0.00005 else 'fair'
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_double_top_bottom(self, price_data: Dict) -> Dict:
        """识别双顶双底"""
        highs = price_data.get('highs', [])
        lows = price_data.get('lows', [])
        
        if len(highs) < 20 or len(lows) < 20:
            return {'detected': False, 'confidence': 0.0}
        
        # 简化实现：检测相似的高点或低点
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        # 寻找双顶
        if len(recent_highs) >= 6:
            # 检查最近的高点
            last_two_highs = recent_highs[-2:]
            second_last_two_highs = recent_highs[-6:-4]
            
            if len(last_two_highs) == 2 and len(second_last_two_highs) == 2:
                avg_last = np.mean(last_two_highs)
                avg_second_last = np.mean(second_last_two_highs)
                
                if abs(avg_last - avg_second_last) / avg_last < 0.01:  # 1%以内
                    # 检查中间的低点
                    middle_lows = recent_highs[-4:-2]
                    if middle_lows and min(middle_lows) < avg_last * 0.99:
                        return {
                            'detected': True,
                            'confidence': 0.7,
                            'pattern': 'double_top',
                            'formation_period_bars': 6,
                            'resistance_level': avg_last
                        }
        
        # 寻找双底
        if len(recent_lows) >= 6:
            last_two_lows = recent_lows[-2:]
            second_last_two_lows = recent_lows[-6:-4]
            
            if len(last_two_lows) == 2 and len(second_last_two_lows) == 2:
                avg_last = np.mean(last_two_lows)
                avg_second_last = np.mean(second_last_two_lows)
                
                if abs(avg_last - avg_second_last) / avg_last < 0.01:
                    middle_highs = recent_lows[-4:-2]
                    if middle_highs and max(middle_highs) > avg_last * 1.01:
                        return {
                            'detected': True,
                            'confidence': 0.7,
                            'pattern': 'double_bottom',
                            'formation_period_bars': 6,
                            'support_level': avg_last
                        }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_head_shoulders(self, price_data: Dict) -> Dict:
        """识别头肩形态"""
        # 简化实现：检测头肩形态模式
        highs = price_data.get('highs', [])
        
        if len(highs) < 15:
            return {'detected': False, 'confidence': 0.0}
        
        recent_highs = highs[-15:]
        
        # 简单模式检测：高中低高中高
        if len(recent_highs) >= 7:
            # 检查头肩形态特征
            pattern_highs = recent_highs[-7:]  # 7个高点构成潜在头肩形态
            
            # 头应该是最高的，肩膀高度相近
            head_idx = np.argmax(pattern_highs)
            if head_idx == 3:  # 头在中间位置
                left_shoulder = pattern_highs[1]
                right_shoulder = pattern_highs[5]
                head_height = pattern_highs[3]
                
                # 肩膀高度应该相近，头应该明显高于肩膀
                shoulder_diff = abs(left_shoulder - right_shoulder) / head_height
                head_shoulder_diff = min(head_height - left_shoulder, head_height - right_shoulder) / head_height
                
                if shoulder_diff < 0.02 and head_shoulder_diff > 0.01:
                    return {
                        'detected': True,
                        'confidence': 0.65,
                        'pattern': 'head_shoulders',
                        'neckline_estimate': (pattern_highs[0] + pattern_highs[6]) / 2,
                        'target_price': head_height - (head_height - (pattern_highs[0] + pattern_highs[6]) / 2)
                    }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_triangle(self, price_data: Dict) -> Dict:
        """识别三角形"""
        highs = price_data.get('highs', [])
        lows = price_data.get('lows', [])
        
        if len(highs) < 15 or len(lows) < 15:
            return {'detected': False, 'confidence': 0.0}
        
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        
        # 计算高点和低点的收敛性
        high_range = max(recent_highs) - min(recent_highs)
        low_range = max(recent_lows) - min(recent_lows)
        
        early_highs = recent_highs[:5]
        late_highs = recent_highs[-5:]
        early_lows = recent_lows[:5]
        late_lows = recent_lows[-5:]
        
        early_high_vol = np.var(early_highs)
        late_high_vol = np.var(late_highs)
        early_low_vol = np.var(early_lows)
        late_low_vol = np.var(late_lows)
        
        # 波动性收敛表示三角形形成
        if late_high_vol < early_high_vol * 0.7 and late_low_vol < early_low_vol * 0.7:
            # 确定三角形类型
            high_trend = (late_highs[-1] - early_highs[0]) / len(early_highs)
            low_trend = (late_lows[-1] - early_lows[0]) / len(early_lows)
            
            if high_trend < 0 and low_trend > 0:
                triangle_type = 'symmetrical'
            elif high_trend < 0 and abs(low_trend) < 0.0001:
                triangle_type = 'descending'
            elif abs(high_trend) < 0.0001 and low_trend > 0:
                triangle_type = 'ascending'
            else:
                triangle_type = 'symmetrical'
            
            return {
                'detected': True,
                'confidence': 0.6,
                'pattern': triangle_type,
                'convergence_ratio': (late_high_vol + late_low_vol) / (early_high_vol + early_low_vol),
                'breakout_direction': 'unknown'
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_flag_pennant(self, price_data: Dict) -> Dict:
        """识别旗形三角旗"""
        # 旗形和三角旗是趋势中的整理形态
        trend_result = self._recognize_trend_line(price_data)
        
        if not trend_result['detected'] or trend_result['direction'] == 'sideways':
            return {'detected': False, 'confidence': 0.0}
        
        highs = price_data.get('highs', [])
        lows = price_data.get('lows', [])
        
        if len(highs) < 10 or len(lows) < 10:
            return {'detected': False, 'confidence': 0.0}
        
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # 检查整理形态（价格在窄幅区间内）
        high_range = max(recent_highs) - min(recent_highs)
        low_range = max(recent_lows) - min(recent_lows)
        avg_price = np.mean(recent_highs + recent_lows)
        
        range_ratio = (high_range + low_range) / (2 * avg_price)
        
        if range_ratio < 0.005:  # 价格波动很小
            # 确定是旗形还是三角旗
            high_trend = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
            low_trend = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
            
            if abs(high_trend) > 0.0001 and abs(low_trend) > 0.0001:
                pattern_type = 'pennant'
            else:
                pattern_type = 'flag'
            
            return {
                'detected': True,
                'confidence': 0.7,
                'pattern': pattern_type,
                'trend_direction': trend_result['direction'],
                'consolidation_ratio': range_ratio
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_breakout_retest(self, price_data: Dict) -> Dict:
        """识别突破回测"""
        # 需要更复杂的实现，这里简化
        support_resistance = self._recognize_support_resistance(price_data)
        
        if not support_resistance['detected']:
            return {'detected': False, 'confidence': 0.0}
        
        prices = price_data.get('prices', [])
        if len(prices) < 10:
            return {'detected': False, 'confidence': 0.0}
        
        recent_prices = prices[-10:]
        level = support_resistance['level']
        
        # 检查突破和回测
        early_prices = recent_prices[:5]
        late_prices = recent_prices[-5:]
        
        # 早期价格在水平一侧，晚期价格在另一侧
        early_side = 'above' if np.mean(early_prices) > level else 'below'
        late_side = 'above' if np.mean(late_prices) > level else 'below'
        
        if early_side != late_side:
            # 突破发生
            # 检查是否回测水平
            min_distance = min(abs(p - level) for p in late_prices)
            if min_distance / level < 0.001:  # 回测到水平附近
                return {
                    'detected': True,
                    'confidence': 0.75,
                    'pattern': 'breakout_retest',
                    'level': level,
                    'breakout_direction': 'up' if late_side == 'above' else 'down',
                    'retest_quality': 'good' if min_distance / level < 0.0005 else 'fair'
                }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_fake_out(self, price_data: Dict) -> Dict:
        """识别假突破"""
        # 假突破：突破后迅速反转
        breakout_result = self._recognize_breakout_retest(price_data)
        
        if not breakout_result['detected']:
            return {'detected': False, 'confidence': 0.0}
        
        prices = price_data.get('prices', [])
        if len(prices) < 15:
            return {'detected': False, 'confidence': 0.0}
        
        # 检查突破后的价格行为
        level = breakout_result['level']
        direction = breakout_result['breakout_direction']
        
        # 突破后的价格
        post_breakout_prices = prices[-5:]
        
        # 计算突破后的价格趋势
        if len(post_breakout_prices) >= 3:
            price_trend = (post_breakout_prices[-1] - post_breakout_prices[0]) / len(post_breakout_prices)
            
            # 如果突破后价格迅速向反方向移动，可能是假突破
            expected_trend = 0.0005 if direction == 'up' else -0.0005
            if price_trend * expected_trend < 0:  # 方向相反
                return {
                    'detected': True,
                    'confidence': 0.65,
                    'pattern': 'fake_out',
                    'original_breakout_direction': direction,
                    'reversal_strength': abs(price_trend / expected_trend),
                    'trap_indication': 'bull_trap' if direction == 'up' else 'bear_trap'
                }
        
        return {'detected': False, 'confidence': 0.0}
    
    def _recognize_inside_bar(self, price_data: Dict) -> Dict:
        """识别内包线"""
        highs = price_data.get('highs', [])
        lows = price_data.get('lows', [])
        
        if len(highs) < 2 or len(lows) < 2:
            return {'detected': False, 'confidence': 0.0}
        
        # 内包线：当前K线的高低点完全包含在前一根K线内
        prev_high = highs[-2]
        prev_low = lows[-2]
        current_high = highs[-1]
        current_low = lows[-1]
        
        if current_high <= prev_high and current_low >= prev_low:
            # 完全内包
            return {
                'detected': True,
                'confidence': 0.9,
                'pattern': 'inside_bar',
                'size_ratio': (current_high - current_low) / (prev_high - prev_low),
                'position': 'middle' if (current_high + current_low) / 2 > (prev_high + prev_low) / 2 else 'lower'
            }
        
        return {'detected': False, 'confidence': 0.0}
    
    # ========== 经验提取方法 ==========
    
    def _extract_success_pattern_experience(self, case: TradingCase) -> List[str]:
        """提取成功模式经验"""
        experiences = []
        
        if case.profit_loss > 0 and case.patterns_observed:
            patterns = [p.value for p in case.patterns_observed]
            market_condition = case.market_condition.value
            
            experiences.append(f"在{market_condition}市场中，{', '.join(patterns)}模式组合表现良好")
            
            if case.profit_loss_pct > 5:
                experiences.append(f"高利润案例显示{patterns[0]}模式在{market_condition}条件下有强盈利潜力")
        
        return experiences
    
    def _extract_market_condition_experience(self, case: TradingCase) -> List[str]:
        """提取市场条件经验"""
        experiences = []
        market_condition = case.market_condition.value
        
        if case.profit_loss > 0:
            experiences.append(f"{market_condition}市场中的成功交易通常需要耐心等待确认信号")
        else:
            experiences.append(f"{market_condition}市场中的交易需要特别注意风险管理")
        
        return experiences
    
    def _extract_entry_timing_experience(self, case: TradingCase) -> List[str]:
        """提取入场时机经验"""
        experiences = []
        
        if case.profit_loss > 0 and 'entry_reason' in case.__dict__:
            entry_reason = case.entry_reason.lower()
            
            if 'confirmation' in entry_reason:
                experiences.append("等待多重确认信号可以提高入场成功率")
            elif 'retest' in entry_reason:
                experiences.append("突破后回测入场提供更好的风险回报比")
            elif 'pullback' in entry_reason:
                experiences.append("趋势中的回调入场是有效的策略")
        
        return experiences
    
    def _extract_failure_pattern_experience(self, case: TradingCase) -> List[str]:
        """提取失败模式经验"""
        experiences = []
        
        if case.profit_loss < 0 and case.patterns_observed:
            patterns = [p.value for p in case.patterns_observed]
            market_condition = case.market_condition.value
            
            experiences.append(f"在{market_condition}市场中，{', '.join(patterns)}模式组合可能需要额外谨慎")
            
            if 'fake_out' in patterns:
                experiences.append("假突破模式需要额外的确认和更严格的风险管理")
        
        return experiences
    
    def _extract_mistake_analysis_experience(self, case: TradingCase) -> List[str]:
        """提取错误分析经验"""
        experiences = []
        
        for mistake in case.mistakes_made:
            if 'overtrading' in mistake.lower():
                experiences.append("过度交易通常源于情绪化决策而非市场机会")
            elif 'stop loss' in mistake.lower():
                experiences.append("不设止损是导致重大亏损的主要原因")
            elif 'patience' in mistake.lower():
                experiences.append("缺乏耐心常常导致过早入场或出场")
            elif 'size' in mistake.lower():
                experiences.append("仓位过大增加心理压力和决策错误")
        
        return experiences
    
    def _extract_risk_management_experience(self, case: TradingCase) -> List[str]:
        """提取风险管理经验"""
        experiences = []
        
        risk_data = case.risk_management
        if risk_data:
            if 'stop_loss' in risk_data and 'take_profit' in risk_data:
                sl = risk_data['stop_loss']
                tp = risk_data['take_profit']
                rr_ratio = abs((tp - case.entry_price) / (case.entry_price - sl)) if case.entry_price != sl else 0
                
                if rr_ratio < 1:
                    experiences.append(f"风险回报比{rr_ratio:.1f}:1偏低，应考虑提高止盈目标或降低止损")
                elif rr_ratio > 2:
                    experiences.append(f"风险回报比{rr_ratio:.1f}:1良好，支持该交易决策")
        
        return experiences
    
    def _extract_psychological_lessons(self, case: TradingCase) -> List[str]:
        """提取心理教训"""
        experiences = []
        
        psych_data = case.psychological_state
        if psych_data:
            if 'confidence' in psych_data:
                confidence = psych_data['confidence']
                if confidence > 0.8 and case.profit_loss < 0:
                    experiences.append("过高自信可能导致忽略风险信号")
                elif confidence < 0.4 and case.profit_loss > 0:
                    experiences.append("信心不足可能导致过早平仓，错过更大利润")
            
            if 'emotional_state' in psych_data:
                emotion = psych_data['emotional_state']
                if emotion in ['fearful', 'greedy', 'frustrated']:
                    experiences.append(f"{emotion}情绪状态下应暂停交易决策")
        
        return experiences
    
    def _extract_discipline_lessons(self, case: TradingCase) -> List[str]:
        """提取纪律教训"""
        experiences = []
        
        if case.key_decisions:
            decisions_str = ', '.join(case.key_decisions[:3])
            experiences.append(f"遵循决策流程包括: {decisions_str}")
        
        if case.mistakes_made:
            experiences.append("纪律违反常导致: " + ', '.join(case.mistakes_made[:2]))
        
        return experiences
    
    def _extract_adaptation_lessons(self, case: TradingCase) -> List[str]:
        """提取适应性教训"""
        experiences = []
        market_condition = case.market_condition.value
        
        if case.profit_loss > 0:
            experiences.append(f"成功适应{market_condition}市场条件的关键因素: {', '.join(case.success_factors[:2])}")
        else:
            experiences.append(f"{market_condition}市场中需要调整的策略: 更严格的入场条件或更灵活的风险管理")
        
        return experiences
    
    def get_system_statistics(self) -> Dict:
        """获取系统统计信息"""
        return {
            'storage': {
                'total_cases': len(self.cases),
                'storage_path': self.storage_path,
                'max_cases': self.max_cases,
                'indexing_enabled': self.enable_indexing
            },
            'statistics': self.statistics,
            'indices_summary': {
                'by_type': {k: len(v) for k, v in self.indices['by_type'].items()},
                'by_market': {k: len(v) for k, v in self.indices['by_market'].items()},
                'by_pattern': {k: len(v) for k, v in self.indices['by_pattern'].items()},
                'by_market_condition': {k: len(v) for k, v in self.indices['by_market_condition'].items()},
                'by_difficulty': {k: len(v) for k, v in self.indices['by_difficulty'].items()}
            },
            'pattern_recognizers': list(self.pattern_recognizers.keys())
        }
    
    def demo_case_study_analyzer(self):
        """演示案例分析器"""
        print("=" * 60)
        print("实战案例分析器演示")
        print("第30章：实战案例分析 - AL Brooks《价格行为交易之区间篇》")
        print("=" * 60)
        
        # 创建案例分析器
        analyzer = CaseStudyAnalyzer(max_cases=50)
        
        print("\n1. 添加示例交易案例...")
        
        # 添加成功案例
        success_case = {
            'title': '欧元兑美元趋势通道成功交易',
            'case_type': 'success',
            'market': 'forex',
            'symbol': 'EUR/USD',
            'timeframe': '4h',
            'entry_date': datetime.now() - timedelta(days=3),
            'exit_date': datetime.now() - timedelta(days=1),
            'entry_price': 1.0820,
            'exit_price': 1.0885,
            'position_size': 1.0,
            'profit_loss': 65.0,
            'profit_loss_pct': 0.6,
            'market_condition': 'trending_up',
            'patterns_observed': ['trend_line', 'channel', 'breakout_retest'],
            'entry_reason': '趋势通道下轨反弹，多重确认信号',
            'exit_reason': '达到风险回报比目标，通道上轨阻力',
            'key_decisions': ['等待通道下轨确认', '设置1:2风险回报比', '分批减仓'],
            'mistakes_made': ['入场稍早，承受额外波动'],
            'lessons_learned': ['趋势通道交易需要耐心等待确认', '分批减仓锁定利润减少压力'],
            'success_factors': ['清晰的市场结构', '严格执行风险管理', '耐心等待最佳时机'],
            'technical_indicators': {'rsi': 55, 'macd': 'bullish_cross'},
            'risk_management': {'stop_loss': 1.0790, 'take_profit': 1.0890, 'position_size_pct': 2},
            'psychological_state': {'confidence': 0.7, 'emotional_state': 'calm'},
            'tags': ['trend_following', 'channel_trading', 'success'],
            'difficulty_level': 3,
            'confidence_level': 0.7,
            'data_snapshot': {'price_history': '100_bars', 'volume_profile': 'normal'}
        }
        
        success_result = analyzer.add_case(success_case)
        print(f"   成功案例添加: {'成功' if success_result['success'] else '失败'}")
        
        # 添加失败案例
        failure_case = {
            'title': '英镑兑美元假突破亏损交易',
            'case_type': 'failure',
            'market': 'forex',
            'symbol': 'GBP/USD',
            'timeframe': '1h',
            'entry_date': datetime.now() - timedelta(days=2),
            'exit_date': datetime.now() - timedelta(days=1, hours=12),
            'entry_price': 1.2650,
            'exit_price': 1.2610,
            'position_size': 1.0,
            'profit_loss': -40.0,
            'profit_loss_pct': -0.32,
            'market_condition': 'ranging',
            'patterns_observed': ['fake_out', 'support_resistance'],
            'entry_reason': '假突破追单，缺乏足够确认',
            'exit_reason': '止损触发，价格反转',
            'key_decisions': ['突破追单', '设置紧止损'],
            'mistakes_made': ['未等待突破确认', '在震荡市场中追突破', '止损设置过紧'],
            'lessons_learned': ['震荡市场突破需要额外确认', '避免在关键新闻前交易', '止损应考虑市场波动性'],
            'success_factors': [],
            'technical_indicators': {'rsi': 65, 'stochastic': 'overbought'},
            'risk_management': {'stop_loss': 1.2635, 'take_profit': 1.2700},
            'psychological_state': {'confidence': 0.8, 'emotional_state': 'greedy'},
            'tags': ['breakout_trading', 'fakeout', 'lesson_learned'],
            'difficulty_level': 4,
            'confidence_level': 0.8,
            'data_snapshot': {'price_history': '50_bars', 'news_events': ['NFP_release']}
        }
        
        failure_result = analyzer.add_case(failure_case)
        print(f"   失败案例添加: {'成功' if failure_result['success'] else '失败'}")
        
        print("\n2. 获取系统统计...")
        stats = analyzer.get_system_statistics()
        print(f"   总案例数: {stats['storage']['total_cases']}")
        print(f"   成功率: {stats['statistics']['win_rate']*100:.1f}%")
        print(f"   平均利润率: {stats['statistics']['average_profit_loss']:.2f}")
        
        print("\n3. 搜索案例...")
        success_cases = analyzer.search_cases(
            filters={'case_type': 'success'},
            limit=3
        )
        print(f"   找到成功案例: {len(success_cases)}个")
        
        print("\n4. 分析价格模式...")
        pattern_analysis = analyzer.analyze_patterns()
        if pattern_analysis.get('success', True):
            print(f"   分析案例数: {pattern_analysis['total_cases_analyzed']}")
            if pattern_analysis['recommended_patterns']:
                best_pattern = pattern_analysis['recommended_patterns'][0]
                print(f"   最佳模式: {best_pattern['pattern']} (成功率: {best_pattern['success_rate']*100:.1f}%)")
        
        print("\n5. 提取交易经验...")
        experiences = analyzer.extract_experiences()
        print(f"   成功经验数: {len(experiences['success_experiences'])}")
        print(f"   失败教训数: {len(experiences['failure_lessons'])}")
        print(f"   常见错误: {len(experiences['common_mistakes'])}个")
        
        print("\n6. 生成学习推荐...")
        recommendations = analyzer.generate_learning_recommendations()
        print(f"   推荐案例数: {len(recommendations['recommended_cases'])}")
        print(f"   技能发展路径: {len(recommendations['skill_development_path'])}个阶段")
        
        print("\n7. 创建模拟学习场景...")
        scenario = analyzer.create_simulation_scenario(difficulty=3)
        print(f"   场景ID: {scenario['scenario_id']}")
        print(f"   学习目标: {len(scenario['learning_objectives'])}个")
        print(f"   预计时长: {scenario['estimated_duration_minutes']}分钟")
        
        print("\n8. 模式识别测试...")
        test_data = {
            'prices': [1.0800, 1.0810, 1.0820, 1.0815, 1.0825, 1.0830, 1.0828, 1.0835],
            'highs': [1.0810, 1.0825, 1.0830, 1.0835],
            'lows': [1.0800, 1.0810, 1.0815, 1.0820]
        }
        
        inside_bar_result = analyzer._recognize_inside_bar(test_data)
        print(f"   内包线识别: {'成功' if inside_bar_result['detected'] else '失败'}")
        
        trend_result = analyzer._recognize_trend_line(test_data)
        print(f"   趋势线识别: {'成功' if trend_result['detected'] else '失败'}")
        
        print("\n" + "=" * 60)
        print("演示完成")
        print("实战案例分析器已成功创建并测试")
        print("=" * 60)


# ============================================================================
# 策略改造: 添加CaseStudyAnalyzerStrategy类
# 将实战案例分析器转换为交易策略
# ============================================================================

class CaseStudyAnalyzerStrategy(BaseStrategy):
    """实战案例分析策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict = None):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        case_count = self.params.get('case_count', 10)
        include_lessons = self.params.get('include_lessons', True)
        
        # 创建实战案例分析器实例
        self.analyzer = CaseStudyAnalyzer(
            case_count=case_count,
            include_lessons=include_lessons
        )
        
        # 初始化样本案例（实际使用中应提供真实案例）
        self._initialize_sample_cases()
    
    def _initialize_sample_cases(self):
        """初始化样本案例"""
        # 创建样本案例数据
        sample_cases = []
        for i in range(10):  # 10个样本案例
            case = {
                'case_id': f'case_{i}',
                'title': f'样本案例 {i+1}',
                'description': f'这是一个样本交易案例，用于演示目的',
                'market_condition': 'trending' if i % 2 == 0 else 'ranging',
                'outcome': 'success' if i < 7 else 'failure',  # 70%成功，30%失败
                'key_lessons': [
                    f'样本教训 {j+1}: 重要交易原则'
                    for j in range(3)
                ],
                'trading_patterns': ['breakout', 'pullback', 'reversal'][:2]
            }
            sample_cases.append(case)
        
        # 添加到分析器
        for case in sample_cases:
            self.analyzer.add_case(case)
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于实战案例分析生成交易信号
        """
        # 分析案例模式
        analysis_result = self.analyzer.analyze_patterns()
        
        # 获取关键教训和推荐
        key_lessons = analysis_result.get('key_lessons', [])
        pattern_recommendations = analysis_result.get('pattern_recommendations', [])
        
        # 基于案例学习生成信号
        success_rate = analysis_result.get('success_rate', 0)
        failure_patterns = analysis_result.get('failure_patterns', [])
        
        if success_rate >= 0.7 and not failure_patterns:
            # 高成功率且无失败模式，买入信号
            self._record_signal(
                timestamp=self.data.index[-1],
                action='buy',
                price=self.data['close'].iloc[-1]
            )
        elif failure_patterns:
            # 有失败模式识别，hold信号（避免重复错误）
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        else:
            # 默认基于模式推荐
            for recommendation in pattern_recommendations:
                if recommendation.get('recommended_action', '').lower() == 'buy':
                    self._record_signal(
                        timestamp=self.data.index[-1],
                        action='buy',
                        price=self.data['close'].iloc[-1]
                    )
                    return self.signals
                elif recommendation.get('recommended_action', '').lower() == 'sell':
                    self._record_signal(
                        timestamp=self.data.index[-1],
                        action='sell',
                        price=self.data['close'].iloc[-1]
                    )
                    return self.signals
            
            # 默认hold信号
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
    demo_case_study_analyzer()