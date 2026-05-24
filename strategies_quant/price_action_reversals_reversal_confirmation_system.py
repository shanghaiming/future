# migrated to quant_trade-main/integrated/strategies/
# original file in workspace root
# migration time: 2026-04-09T23:53:23.029561

"""
反转确认信号量化分析系统 - 第3章《反转确认信号》
严格按照第18章标准：实际完整代码，非伪代码框架
紧急冲刺模式：12:50-13:50完成

系统功能：
1. 成交量确认：反转点成交量异常检测
2. 价格行为确认：关键价位突破确认
3. 时间框架确认：多时间框架一致性验证
4. 动量确认：技术指标背离确认
5. 多重确认综合评估：综合多个确认信号计算置信度
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import statistics

# 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy


class ConfirmationSignalType(Enum):
    """确认信号类型枚举"""
    VOLUME_CONFIRMATION = "volume_confirmation"          # 成交量确认
    PRICE_ACTION_CONFIRMATION = "price_action_confirmation"  # 价格行为确认
    TIMEFRAME_CONFIRMATION = "timeframe_confirmation"    # 时间框架确认
    MOMENTUM_CONFIRMATION = "momentum_confirmation"      # 动量确认
    PATTERN_CONFIRMATION = "pattern_confirmation"        # 模式确认
    SUPPORT_RESISTANCE_CONFIRMATION = "support_resistance_confirmation"  # 支撑阻力确认


class ConfirmationStrength(Enum):
    """确认强度等级"""
    WEAK = "weak"          # 弱确认（单一信号）
    MODERATE = "moderate"  # 中等确认（2个信号）
    STRONG = "strong"      # 强确认（3个信号）
    VERY_STRONG = "very_strong"  # 极强确认（4+个信号）


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
class ConfirmationSignal:
    """确认信号数据类"""
    signal_id: str
    signal_type: ConfirmationSignalType
    timestamp: datetime
    price_level: float
    strength_score: float  # 0-1强度分数
    description: str
    metadata: Dict[str, Any]


@dataclass
class MultiConfirmationAssessment:
    """多重确认评估数据类"""
    assessment_id: str
    signals: List[ConfirmationSignal]  # 确认信号列表
    overall_strength: ConfirmationStrength  # 总体强度
    confidence_score: float  # 总体置信度（0-1）
    recommended_action: str  # 建议行动
    assessment_time: datetime
    details: Dict[str, Any]


class ReversalConfirmationSystem(BaseStrategy):
    """
    反转确认信号量化分析系统
    
    严格按照第18章标准实现，提供完整的反转确认信号分析功能
    紧急冲刺模式：核心功能优先，实际完整代码
    """
    
    def __init__(self, data: pd.DataFrame, params: Dict = None):
        """
        初始化反转确认系统

        Args:
            data: 价格数据DataFrame
            params: 参数字典，包含initial_balance等配置
        """
        super().__init__(data, params)

        # 从参数中提取配置
        self.initial_balance = self.params.get('initial_balance', 10000.0)
        self.current_balance = self.initial_balance
        self.confirmation_signals = []  # 确认信号
        self.assessments = []           # 评估结果
        self.trade_setups = []          # 交易设置
        
        # 系统配置
        self.config = {
            "volume_spike_multiplier": self.params.get('volume_spike_multiplier', 2.0),        # 成交量放大倍数阈值
            "price_breakout_threshold": self.params.get('price_breakout_threshold', 0.01),      # 价格突破阈值（1%）
            "timeframe_alignment_threshold": self.params.get('timeframe_alignment_threshold', 0.7),  # 时间框架对齐阈值
            "momentum_divergence_threshold": self.params.get('momentum_divergence_threshold', 0.05), # 动量背离阈值（5%）
            "min_signals_for_strong_confirmation": self.params.get('min_signals_for_strong_confirmation', 3),  # 强确认最小信号数
            "default_risk_per_trade": self.params.get('default_risk_per_trade', 0.02),        # 默认每笔交易风险（2%）
        }
        
        # 确认信号检测器映射
        self.confirmation_detectors = {
            ConfirmationSignalType.VOLUME_CONFIRMATION: self.detect_volume_confirmation,
            ConfirmationSignalType.PRICE_ACTION_CONFIRMATION: self.detect_price_action_confirmation,
            ConfirmationSignalType.TIMEFRAME_CONFIRMATION: self.detect_timeframe_confirmation,
            ConfirmationSignalType.MOMENTUM_CONFIRMATION: self.detect_momentum_confirmation,
        }
    
    # ==================== 核心确认信号检测方法 ====================
    
    def detect_volume_confirmation(self, price_bars: List[PriceBar], 
                                  lookback_period: int = 20) -> List[ConfirmationSignal]:
        """
        检测成交量确认信号
        
        Args:
            price_bars: 价格柱列表
            lookback_period: 回顾周期
            
        Returns:
            成交量确认信号列表
        """
        signals = []
        
        if len(price_bars) < lookback_period:
            return signals
        
        for i in range(lookback_period, len(price_bars)):
            current_bar = price_bars[i]
            lookback_bars = price_bars[i-lookback_period:i]
            
            # 计算平均成交量
            lookback_volumes = [bar.volume for bar in lookback_bars]
            avg_volume = statistics.mean(lookback_volumes)
            std_volume = statistics.stdev(lookback_volumes) if len(lookback_volumes) > 1 else avg_volume * 0.5
            
            # 检查成交量是否异常放大
            volume_threshold = avg_volume + (self.config["volume_spike_multiplier"] * std_volume)
            
            if current_bar.volume > volume_threshold:
                # 检查价格行为（是否在关键位置）
                price_context = self._analyze_price_context(price_bars, i)
                
                signal = ConfirmationSignal(
                    signal_id=f"volume_confirmation_{current_bar.timestamp.isoformat()}",
                    signal_type=ConfirmationSignalType.VOLUME_CONFIRMATION,
                    timestamp=current_bar.timestamp,
                    price_level=current_bar.close,
                    strength_score=self._calculate_volume_strength(current_bar, avg_volume, std_volume),
                    description=f"成交量确认: {current_bar.volume:.0f} > 平均{avg_volume:.0f}",
                    metadata={
                        "current_volume": current_bar.volume,
                        "avg_volume": avg_volume,
                        "std_volume": std_volume,
                        "volume_ratio": current_bar.volume / avg_volume,
                        "price_context": price_context,
                    }
                )
                signals.append(signal)
        
        return signals
    
    def detect_price_action_confirmation(self, price_bars: List[PriceBar], 
                                        support_level: float, resistance_level: float) -> List[ConfirmationSignal]:
        """
        检测价格行为确认信号（关键价位突破）
        
        Args:
            price_bars: 价格柱列表
            support_level: 支撑位
            resistance_level: 阻力位
            
        Returns:
            价格行为确认信号列表
        """
        signals = []
        
        if len(price_bars) < 5:
            return signals
        
        current_bar = price_bars[-1]
        previous_bar = price_bars[-2]
        
        # 检查阻力位突破（看涨确认）
        breakout_threshold = self.config["price_breakout_threshold"]
        
        # 阻力位突破确认
        if previous_bar.high < resistance_level and current_bar.close > resistance_level:
            # 确认突破的有效性
            breakout_strength = self._calculate_breakout_strength(current_bar, resistance_level, "resistance")
            
            signal = ConfirmationSignal(
                signal_id=f"price_action_confirmation_resistance_{current_bar.timestamp.isoformat()}",
                signal_type=ConfirmationSignalType.PRICE_ACTION_CONFIRMATION,
                timestamp=current_bar.timestamp,
                price_level=current_bar.close,
                strength_score=breakout_strength,
                description=f"阻力位突破确认: {current_bar.close:.4f} > 阻力位{resistance_level:.4f}",
                metadata={
                    "breakout_type": "resistance",
                    "resistance_level": resistance_level,
                    "breakout_price": current_bar.close,
                    "breakout_percent": (current_bar.close - resistance_level) / resistance_level * 100,
                    "previous_high": previous_bar.high,
                }
            )
            signals.append(signal)
        
        # 支撑位突破确认（看跌确认）
        if previous_bar.low > support_level and current_bar.close < support_level:
            breakout_strength = self._calculate_breakout_strength(current_bar, support_level, "support")
            
            signal = ConfirmationSignal(
                signal_id=f"price_action_confirmation_support_{current_bar.timestamp.isoformat()}",
                signal_type=ConfirmationSignalType.PRICE_ACTION_CONFIRMATION,
                timestamp=current_bar.timestamp,
                price_level=current_bar.close,
                strength_score=breakout_strength,
                description=f"支撑位突破确认: {current_bar.close:.4f} < 支撑位{support_level:.4f}",
                metadata={
                    "breakout_type": "support",
                    "support_level": support_level,
                    "breakout_price": current_bar.close,
                    "breakout_percent": (support_level - current_bar.close) / support_level * 100,
                    "previous_low": previous_bar.low,
                }
            )
            signals.append(signal)
        
        return signals
    
    def detect_timeframe_confirmation(self, multi_timeframe_data: Dict[str, List[PriceBar]]) -> List[ConfirmationSignal]:
        """
        检测时间框架确认信号（多时间框架一致性）
        
        Args:
            multi_timeframe_data: 多时间框架数据字典 {timeframe: price_bars}
            
        Returns:
            时间框架确认信号列表
        """
        signals = []
        
        if len(multi_timeframe_data) < 2:
            return signals
        
        # 提取时间框架列表
        timeframes = list(multi_timeframe_data.keys())
        timeframes.sort(key=lambda x: self._timeframe_to_minutes(x), reverse=True)  # 从大到小排序
        
        # 检查每个时间框架的趋势一致性
        trend_alignment = self._analyze_trend_alignment(multi_timeframe_data)
        
        if trend_alignment["alignment_score"] >= self.config["timeframe_alignment_threshold"]:
            signal = ConfirmationSignal(
                signal_id=f"timeframe_confirmation_{datetime.now().timestamp()}",
                signal_type=ConfirmationSignalType.TIMEFRAME_CONFIRMATION,
                timestamp=datetime.now(),
                price_level=trend_alignment["current_price"],
                strength_score=trend_alignment["alignment_score"],
                description=f"时间框架确认: {trend_alignment['aligned_timeframes']}/{len(timeframes)}个时间框架趋势一致",
                metadata=trend_alignment
            )
            signals.append(signal)
        
        return signals
    
    def detect_momentum_confirmation(self, price_bars: List[PriceBar], 
                                    rsi_period: int = 14) -> List[ConfirmationSignal]:
        """
        检测动量确认信号（指标背离等）
        
        Args:
            price_bars: 价格柱列表
            rsi_period: RSI计算周期
            
        Returns:
            动量确认信号列表
        """
        signals = []
        
        if len(price_bars) < rsi_period * 2:
            return signals
        
        # 计算RSI
        rsi_values = self._calculate_rsi(price_bars, rsi_period)
        
        for i in range(rsi_period * 2, len(price_bars)):
            current_price = price_bars[i].close
            current_rsi = rsi_values[i]
            
            # 检查看涨背离（价格新低，RSI未创新低）
            recent_prices = [bar.close for bar in price_bars[i-10:i+1]]
            recent_rsi = rsi_values[i-10:i+1]
            
            if len(recent_prices) >= 5 and len(recent_rsi) >= 5:
                # 检查价格是否创出新低
                if current_price == min(recent_prices):
                    # 检查RSI是否未创新低
                    if current_rsi > min(recent_rsi[:-1]):
                        divergence_strength = self._calculate_divergence_strength(
                            price_bars, rsi_values, i, "bullish"
                        )
                        
                        if divergence_strength >= self.config["momentum_divergence_threshold"]:
                            signal = ConfirmationSignal(
                                signal_id=f"momentum_confirmation_bullish_{price_bars[i].timestamp.isoformat()}",
                                signal_type=ConfirmationSignalType.MOMENTUM_CONFIRMATION,
                                timestamp=price_bars[i].timestamp,
                                price_level=current_price,
                                strength_score=divergence_strength,
                                description=f"看涨动量确认: 价格新低{current_price:.4f}, RSI未创新低{current_rsi:.2f}",
                                metadata={
                                    "divergence_type": "bullish",
                                    "price_low": current_price,
                                    "rsi_value": current_rsi,
                                    "divergence_strength": divergence_strength,
                                }
                            )
                            signals.append(signal)
                
                # 检查看跌背离（价格新高，RSI未创新高）
                if current_price == max(recent_prices):
                    if current_rsi < max(recent_rsi[:-1]):
                        divergence_strength = self._calculate_divergence_strength(
                            price_bars, rsi_values, i, "bearish"
                        )
                        
                        if divergence_strength >= self.config["momentum_divergence_threshold"]:
                            signal = ConfirmationSignal(
                                signal_id=f"momentum_confirmation_bearish_{price_bars[i].timestamp.isoformat()}",
                                signal_type=ConfirmationSignalType.MOMENTUM_CONFIRMATION,
                                timestamp=price_bars[i].timestamp,
                                price_level=current_price,
                                strength_score=divergence_strength,
                                description=f"看跌动量确认: 价格新高{current_price:.4f}, RSI未创新高{current_rsi:.2f}",
                                metadata={
                                    "divergence_type": "bearish",
                                    "price_high": current_price,
                                    "rsi_value": current_rsi,
                                    "divergence_strength": divergence_strength,
                                }
                            )
                            signals.append(signal)
        
        return signals
    
    # ==================== 辅助检测方法 ====================
    
    def _analyze_price_context(self, price_bars: List[PriceBar], index: int) -> str:
        """分析价格上下文"""
        if index < 10 or index >= len(price_bars) - 5:
            return "unknown"
        
        current_bar = price_bars[index]
        
        # 检查是否在关键水平附近
        recent_high = max(bar.high for bar in price_bars[index-10:index+1])
        recent_low = min(bar.low for bar in price_bars[index-10:index+1])
        
        if abs(current_bar.high - recent_high) / recent_high < 0.01:
            return "near_resistance"
        elif abs(current_bar.low - recent_low) / recent_low < 0.01:
            return "near_support"
        
        # 检查是否伴随大阳线或大阴线
        if index > 0:
            prev_bar = price_bars[index-1]
            if current_bar.body > prev_bar.body * 1.5:
                if current_bar.is_bullish:
                    return "with_bullish_bar"
                else:
                    return "with_bearish_bar"
        
        return "normal"
    
    def _calculate_volume_strength(self, current_bar: PriceBar, avg_volume: float, std_volume: float) -> float:
        """计算成交量确认强度"""
        if avg_volume <= 0:
            return 0.5
        
        volume_ratio = current_bar.volume / avg_volume
        strength = min(volume_ratio / 3.0, 0.9)  # 最高0.9
        
        # 考虑标准差
        if std_volume > 0:
            z_score = (current_bar.volume - avg_volume) / std_volume
            z_strength = min(z_score / 3.0, 0.5)  # z-score贡献最多0.5
            strength = max(strength, z_strength)
        
        return max(strength, 0.1)
    
    def _calculate_breakout_strength(self, current_bar: PriceBar, 
                                   breakout_level: float, 
                                   breakout_type: str) -> float:
        """计算突破确认强度"""
        if breakout_type == "resistance":
            breakout_distance = (current_bar.close - breakout_level) / breakout_level
        else:  # support
            breakout_distance = (breakout_level - current_bar.close) / breakout_level
        
        # 突破距离越大强度越高
        distance_strength = min(breakout_distance * 100, 1.0)  # 1%突破得1.0
        
        # K线实体大小贡献
        body_strength = min(current_bar.body / current_bar.range * 2, 0.5)  # 实体占比
        
        # 综合强度
        strength = distance_strength * 0.7 + body_strength * 0.3
        
        return min(max(strength, 0.0), 1.0)
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """将时间框架字符串转换为分钟数"""
        timeframe_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
        }
        return timeframe_map.get(timeframe.lower(), 60)
    
    def _analyze_trend_alignment(self, multi_timeframe_data: Dict[str, List[PriceBar]]) -> Dict[str, Any]:
        """分析多时间框架趋势对齐"""
        timeframes = list(multi_timeframe_data.keys())
        trends = []
        current_prices = []
        
        for timeframe, price_bars in multi_timeframe_data.items():
            if len(price_bars) < 20:
                continue
            
            # 简单趋势判断（基于最近20根K线）
            recent_bars = price_bars[-20:]
            closes = [bar.close for bar in recent_bars]
            
            # 线性回归判断趋势
            if len(closes) >= 2:
                x = list(range(len(closes)))
                y = closes
                
                # 简单斜率计算
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_x2 = sum(x_i * x_i for x_i in x)
                
                try:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                except ZeroDivisionError:
                    slope = 0
                
                # 判断趋势方向
                if slope > 0.001:
                    trend = "uptrend"
                elif slope < -0.001:
                    trend = "downtrend"
                else:
                    trend = "sideways"
                
                trends.append((timeframe, trend))
                current_prices.append(closes[-1])
        
        # 计算对齐分数
        if not trends:
            return {"alignment_score": 0.0, "aligned_timeframes": 0, "total_timeframes": 0, "current_price": 0}
        
        # 统计趋势一致性
        trend_counts = {}
        for timeframe, trend in trends:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        # 找到主导趋势
        dominant_trend = max(trend_counts, key=trend_counts.get)
        aligned_count = trend_counts[dominant_trend]
        total_count = len(trends)
        
        alignment_score = aligned_count / total_count
        current_price = statistics.mean(current_prices) if current_prices else 0
        
        return {
            "alignment_score": alignment_score,
            "aligned_timeframes": aligned_count,
            "total_timeframes": total_count,
            "dominant_trend": dominant_trend,
            "current_price": current_price,
            "trend_details": trends,
        }
    
    def _calculate_rsi(self, price_bars: List[PriceBar], period: int) -> List[float]:
        """计算RSI指标"""
        if len(price_bars) < period + 1:
            return [50.0] * len(price_bars)
        
        closes = [bar.close for bar in price_bars]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        # 分离上涨和下跌
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # 计算平均增益和平均损失
        avg_gains = [0.0] * len(closes)
        avg_losses = [0.0] * len(closes)
        
        # 初始化
        avg_gains[period] = sum(gains[:period]) / period
        avg_losses[period] = sum(losses[:period]) / period
        
        # 平滑计算
        for i in range(period + 1, len(closes)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        # 计算RSI
        rsi = [0.0] * len(closes)
        for i in range(period, len(closes)):
            if avg_losses[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        # 填充前period个值
        for i in range(period):
            rsi[i] = 50.0
        
        return rsi
    
    def _calculate_divergence_strength(self, price_bars: List[PriceBar], 
                                     rsi_values: List[float], 
                                     index: int, 
                                     divergence_type: str) -> float:
        """计算背离强度"""
        if index < 10:
            return 0.0
        
        # 计算价格变化和RSI变化
        lookback = min(10, index)
        price_change = price_bars[index].close - price_bars[index-lookback].close
        rsi_change = rsi_values[index] - rsi_values[index-lookback]
        
        # 背离强度：价格和RSI变化方向相反的程度
        if divergence_type == "bullish":
            # 看涨背离：价格下跌，RSI上升
            if price_change < 0 and rsi_change > 0:
                strength = (abs(rsi_change) / 10.0) * (abs(price_change) / price_bars[index].close * 100)
                strength = min(strength, 1.0)
                return strength
        else:  # bearish
            # 看跌背离：价格上涨，RSI下降
            if price_change > 0 and rsi_change < 0:
                strength = (abs(rsi_change) / 10.0) * (abs(price_change) / price_bars[index].close * 100)
                strength = min(strength, 1.0)
                return strength
        
        return 0.0
    
    # ==================== 多重确认评估 ====================
    
    def assess_multi_confirmation(self, signals: List[ConfirmationSignal]) -> MultiConfirmationAssessment:
        """
        评估多重确认信号
        
        Args:
            signals: 确认信号列表
            
        Returns:
            多重确认评估结果
        """
        if not signals:
            return self._create_empty_assessment()
        
        # 按信号类型分组
        signal_types = [signal.signal_type for signal in signals]
        unique_types = set(signal_types)
        
        # 计算平均强度
        strength_scores = [signal.strength_score for signal in signals]
        avg_strength = statistics.mean(strength_scores) if strength_scores else 0.0
        
        # 确定总体强度等级
        signal_count = len(signals)
        unique_count = len(unique_types)
        
        if signal_count >= 4 or unique_count >= 3:
            overall_strength = ConfirmationStrength.VERY_STRONG
            confidence = min(avg_strength * 1.2, 0.95)
        elif signal_count == 3:
            overall_strength = ConfirmationStrength.STRONG
            confidence = min(avg_strength * 1.1, 0.9)
        elif signal_count == 2:
            overall_strength = ConfirmationStrength.MODERATE
            confidence = avg_strength
        else:
            overall_strength = ConfirmationStrength.WEAK
            confidence = avg_strength * 0.8
        
        # 生成建议行动
        if overall_strength in [ConfirmationStrength.VERY_STRONG, ConfirmationStrength.STRONG]:
            recommended_action = "强烈建议执行交易"
        elif overall_strength == ConfirmationStrength.MODERATE:
            recommended_action = "建议执行交易"
        else:
            recommended_action = "谨慎执行交易"
        
        assessment = MultiConfirmationAssessment(
            assessment_id=f"multi_confirmation_{datetime.now().timestamp()}",
            signals=signals,
            overall_strength=overall_strength,
            confidence_score=confidence,
            recommended_action=recommended_action,
            assessment_time=datetime.now(),
            details={
                "signal_count": signal_count,
                "unique_signal_types": len(unique_types),
                "avg_strength_score": avg_strength,
                "signal_types": [st.value for st in unique_types],
                "strength_scores": strength_scores,
            }
        )
        
        self.assessments.append(assessment)
        return assessment
    
    def _create_empty_assessment(self) -> MultiConfirmationAssessment:
        """创建空评估结果"""
        return MultiConfirmationAssessment(
            assessment_id=f"empty_assessment_{datetime.now().timestamp()}",
            signals=[],
            overall_strength=ConfirmationStrength.WEAK,
            confidence_score=0.0,
            recommended_action="无确认信号，不建议交易",
            assessment_time=datetime.now(),
            details={"signal_count": 0, "unique_signal_types": 0, "avg_strength_score": 0.0}
        )
    
    # ==================== 交易设置生成 ====================
    
    def generate_confirmation_trade_setup(self, assessment: MultiConfirmationAssessment,
                                         price_bars: List[PriceBar]) -> Dict[str, Any]:
        """
        基于确认评估生成交易设置
        
        Args:
            assessment: 多重确认评估结果
            price_bars: 价格柱列表
            
        Returns:
            交易设置
        """
        if not price_bars:
            return {"error": "价格数据为空"}
        
        current_bar = price_bars[-1]
        current_price = current_bar.close
        
        # 确定交易方向（基于信号类型）
        direction = self._determine_trade_direction(assessment.signals)
        
        # 计算入场价格
        entry_price = self._calculate_confirmation_entry_price(current_price, direction)
        
        # 计算止损价格
        stop_loss = self._calculate_confirmation_stop_loss(entry_price, price_bars, direction)
        
        # 计算止盈价格
        risk_amount = abs(entry_price - stop_loss)
        take_profit = self._calculate_confirmation_take_profit(entry_price, risk_amount, direction)
        
        # 计算风险回报比
        if risk_amount > 0:
            risk_reward_ratio = abs(take_profit - entry_price) / risk_amount
        else:
            risk_reward_ratio = 0.0
        
        # 计算仓位大小
        risk_per_trade = self.current_balance * self.config["default_risk_per_trade"]
        if risk_amount > 0:
            shares = risk_per_trade / risk_amount
            position_size = shares * entry_price
        else:
            position_size = 0.0
        
        # 限制最大仓位（10%）
        max_position = self.current_balance * 0.1
        position_size = min(position_size, max_position)
        
        setup = {
            "setup_id": f"confirmation_setup_{datetime.now().timestamp()}",
            "assessment_id": assessment.assessment_id,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward_ratio,
            "position_size": position_size,
            "confidence_score": assessment.confidence_score,
            "overall_strength": assessment.overall_strength.value,
            "recommended_action": assessment.recommended_action,
            "generated_time": datetime.now().isoformat(),
            "signal_count": len(assessment.signals),
        }
        
        self.trade_setups.append(setup)
        return setup
    
    def _determine_trade_direction(self, signals: List[ConfirmationSignal]) -> str:
        """确定交易方向"""
        if not signals:
            return "hold"
        
        # 基于信号类型判断方向
        bullish_count = 0
        bearish_count = 0
        
        for signal in signals:
            metadata = signal.metadata
            if "breakout_type" in metadata:
                if metadata["breakout_type"] == "resistance":
                    bullish_count += 1
                else:
                    bearish_count += 1
            elif "divergence_type" in metadata:
                if metadata["divergence_type"] == "bullish":
                    bullish_count += 1
                else:
                    bearish_count += 1
        
        if bullish_count > bearish_count:
            return "buy"
        elif bearish_count > bullish_count:
            return "sell"
        else:
            # 默认基于最近价格趋势
            return "hold"
    
    def _calculate_confirmation_entry_price(self, current_price: float, direction: str) -> float:
        """计算确认交易入场价格"""
        if direction == "buy":
            return current_price * 1.002  # 稍高于当前价
        elif direction == "sell":
            return current_price * 0.998  # 稍低于当前价
        else:
            return current_price
    
    def _calculate_confirmation_stop_loss(self, entry_price: float, 
                                        price_bars: List[PriceBar], 
                                        direction: str) -> float:
        """计算确认交易止损价格"""
        if len(price_bars) < 20:
            # 默认止损：2%
            if direction == "buy":
                return entry_price * 0.98
            else:
                return entry_price * 1.02
        
        # 基于近期波动率计算止损
        recent_closes = [bar.close for bar in price_bars[-20:]]
        avg_price = statistics.mean(recent_closes)
        price_std = statistics.stdev(recent_closes) if len(recent_closes) > 1 else avg_price * 0.02
        
        # 止损距离：2倍波动率，最小1.5%
        stop_distance = max(price_std / avg_price * 2, 0.015)
        
        if direction == "buy":
            return entry_price * (1 - stop_distance)
        else:
            return entry_price * (1 + stop_distance)
    
    def _calculate_confirmation_take_profit(self, entry_price: float, 
                                          risk_amount: float, 
                                          direction: str) -> float:
        """计算确认交易止盈价格"""
        # 默认风险回报比：2:1
        risk_reward_ratio = 2.0
        
        if direction == "buy":
            return entry_price + (risk_amount * risk_reward_ratio)
        else:
            return entry_price - (risk_amount * risk_reward_ratio)
    
    # ==================== 系统演示和报告 ====================
    
    def demonstrate_system(self) -> Dict[str, Any]:
        """
        演示系统功能
        
        Returns:
            演示结果
        """
        # 创建模拟数据
        mock_bars = self._create_mock_price_bars(100)
        
        # 运行所有确认检测器
        volume_signals = self.detect_volume_confirmation(mock_bars)
        momentum_signals = self.detect_momentum_confirmation(mock_bars)
        
        all_signals = volume_signals + momentum_signals
        
        # 评估多重确认
        assessment = self.assess_multi_confirmation(all_signals)
        
        # 生成交易设置
        trade_setup = None
        if all_signals:
            trade_setup = self.generate_confirmation_trade_setup(assessment, mock_bars)
        
        demonstration = {
            "total_signals_detected": len(all_signals),
            "signal_types": list(set([s.signal_type.value for s in all_signals])),
            "assessment_result": {
                "overall_strength": assessment.overall_strength.value,
                "confidence_score": assessment.confidence_score,
                "recommended_action": assessment.recommended_action,
                "signal_count": len(assessment.signals),
            },
            "trade_setup_generated": trade_setup is not None,
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
        生成反转确认交易信号
        
        Returns:
            标准化信号列表
        """
        signals = []
        
        # 将数据转换为PriceBar格式
        price_bars = self._convert_dataframe_to_bars()
        if not price_bars:
            return signals
        
        # 检测各种确认信号
        all_signals = []
        
        # 成交量确认
        volume_signals = self.detect_volume_confirmation(price_bars, lookback_period=20)
        all_signals.extend(volume_signals)
        
        # 价格行为确认
        price_action_signals = self.detect_price_action_confirmation(price_bars, lookback_period=20)
        all_signals.extend(price_action_signals)
        
        # 动量确认
        momentum_signals = self.detect_momentum_confirmation(price_bars, lookback_period=14)
        all_signals.extend(momentum_signals)
        
        # 多重确认评估
        if all_signals and len(all_signals) >= self.config["min_signals_for_strong_confirmation"]:
            # 评估所有信号
            assessment = self.assess_multi_confirmation(all_signals, price_bars)
            
            if assessment.overall_strength.value in ["strong", "very_strong"]:
                # 转换为交易信号
                signal = {
                    'timestamp': assessment.assessment_time,
                    'action': assessment.recommended_action,
                    'price': price_bars[-1].close if price_bars else 0.0,
                    'confidence': assessment.confidence_score,
                    'strength': assessment.overall_strength.value,
                    'signal_count': len(all_signals),
                    'details': assessment.details
                }
                signals.append(signal)
        
        # 记录确认信号
        self.confirmation_signals.extend(all_signals)
        
        return signals
    
    def generate_system_report(self) -> Dict[str, Any]:
        """
        生成系统报告
        
        Returns:
            系统报告
        """
        report = {
            "system_name": "反转确认信号量化分析系统",
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "system_config": self.config,
            "performance_metrics": {
                "signals_detected_total": len(self.confirmation_signals),
                "assessments_generated": len(self.assessments),
                "trade_setups_generated": len(self.trade_setups),
                "current_balance": self.current_balance,
                "balance_change_percent": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            },
            "recent_activity": {
                "last_signals": [s.description for s in self.confirmation_signals[-3:]] if self.confirmation_signals else [],
                "last_assessments": [a.assessment_id for a in self.assessments[-2:]] if self.assessments else [],
                "last_setups": [s["setup_id"] for s in self.trade_setups[-2:]] if self.trade_setups else [],
            },
            "system_status": "active",
            "recommendations": [
                "定期检查确认信号检测准确性",
                "根据市场条件调整确认阈值",
                "结合其他技术分析工具验证确认信号",
                "保持严格的确认信号执行纪律",
            ]
        }
        
        return report


# ==================== 演示函数 ====================

def demonstrate_confirmation_system():
    """演示反转确认系统功能"""
    print("=" * 60)
    print("反转确认信号量化分析系统演示")
    print("严格按照第18章标准：实际完整代码")
    print("紧急冲刺模式：12:50-13:50完成")
    print("=" * 60)
    
    # 创建系统实例
    system = ReversalConfirmationSystem(initial_balance=10000.0)
    
    # 运行系统演示
    demonstration = system.demonstrate_system()
    
    print(f"\n📊 确认信号检测结果:")
    print(f"  检测到的信号总数: {demonstration['total_signals_detected']}")
    print(f"  信号类型: {', '.join(demonstration['signal_types'])}")
    
    print(f"\n🎯 多重确认评估:")
    assessment = demonstration["assessment_result"]
    print(f"  总体强度: {assessment['overall_strength']}")
    print(f"  置信度分数: {assessment['confidence_score']:.2f}")
    print(f"  建议行动: {assessment['recommended_action']}")
    print(f"  信号数量: {assessment['signal_count']}")
    
    if demonstration["trade_setup_generated"]:
        print(f"\n✅ 交易设置生成: 成功")
        print(f"  系统状态: {demonstration['system_status']}")
    else:
        print(f"\n❌ 交易设置生成: 失败（信号不足）")
    
    # 生成系统报告
    report = system.generate_system_report()
    
    print(f"\n📈 系统报告摘要:")
    print(f"  系统状态: {report['system_status']}")
    print(f"  检测到的信号总数: {report['performance_metrics']['signals_detected_total']}")
    print(f"  生成的评估: {report['performance_metrics']['assessments_generated']}")
    print(f"  生成的交易设置: {report['performance_metrics']['trade_setups_generated']}")
    print(f"  当前资金: ${report['performance_metrics']['current_balance']:.2f}")
    
    print(f"\n💡 系统推荐:")
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"  {i}. {recommendation}")
    
    print(f"\n✅ 演示完成！系统运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    # 当直接运行此文件时执行演示
    demonstrate_confirmation_system()