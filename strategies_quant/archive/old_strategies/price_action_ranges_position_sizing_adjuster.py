#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓位规模调整量化系统
第21章：仓位规模调整
AL Brooks《价格行为交易之区间篇》

核心概念（按照第18章标准：完整实际代码）：
1. 基于风险的仓位计算：固定风险比例、固定金额风险
2. 基于波动性的仓位调整：ATR调整、波动率调整
3. 基于账户规模的仓位管理：凯利公式、固定分数
4. 动态仓位调整：市场条件、交易信心度、账户状态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 策略改造: 添加BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

class PositionSizingAdjuster:
    """仓位规模调整器（按照第18章标准：完整实际代码）"""
    
    def __init__(self,
                 account_size: float = 10000.0,
                 max_risk_per_trade: float = 0.02,
                 max_position_size: float = 0.1):
        """
        初始化仓位规模调整器
        
        参数:
            account_size: 账户规模（美元）
            max_risk_per_trade: 单笔交易最大风险比例
            max_position_size: 最大仓位比例
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.sizing_history = []
    
    def calculate_position_size(self,
                              entry_info: Dict,
                              market_conditions: Dict,
                              risk_parameters: Dict) -> Dict:
        """
        计算仓位规模
        
        参数:
            entry_info: 入场信息（价格、类型、信心度等）
            market_conditions: 市场条件（波动率、趋势强度等）
            risk_parameters: 风险参数（止损距离、风险金额等）
            
        返回:
            仓位规模计算结果
        """
        # 输入验证
        if not entry_info or not market_conditions or not risk_parameters:
            return {'error': '输入参数不足'}
        
        entry_price = entry_info.get('entry_price', 0)
        entry_type = entry_info.get('type', 'buy')
        stop_loss = risk_parameters.get('stop_loss', 0)
        
        if entry_price <= 0 or stop_loss <= 0:
            return {'error': '无效的价格或止损'}
        
        # 计算风险金额
        risk_amount = self._calculate_risk_amount(risk_parameters)
        
        # 计算基础仓位规模
        base_size = self._calculate_base_position_size(
            entry_price, stop_loss, entry_type, risk_amount
        )
        
        # 基于波动性调整
        volatility_adjustment = self._adjust_for_volatility(
            base_size, market_conditions
        )
        
        # 基于市场条件调整
        market_adjustment = self._adjust_for_market_conditions(
            volatility_adjustment, market_conditions, entry_info
        )
        
        # 基于账户状态调整
        account_adjustment = self._adjust_for_account_status(
            market_adjustment, self.account_size
        )
        
        # 应用限制和约束
        final_size = self._apply_position_limits(account_adjustment)
        
        # 计算详细指标
        detailed_metrics = self._calculate_detailed_metrics(
            final_size, entry_price, stop_loss, entry_type, risk_amount
        )
        
        result = {
            'account_size': self.account_size,
            'max_risk_per_trade': self.max_risk_per_trade,
            'entry_info': entry_info,
            'risk_parameters': risk_parameters,
            'market_conditions': market_conditions,
            'risk_amount': risk_amount,
            'base_position_size': base_size,
            'volatility_adjustment': volatility_adjustment,
            'market_adjustment': market_adjustment,
            'account_adjustment': account_adjustment,
            'final_position_size': final_size,
            'detailed_metrics': detailed_metrics,
            'recommendation': self._generate_position_recommendation(final_size, detailed_metrics),
            'analysis_time': pd.Timestamp.now()
        }
        
        self.sizing_history.append(result)
        return result
    
    def _calculate_risk_amount(self, risk_parameters: Dict) -> float:
        """计算风险金额"""
        risk_percentage = risk_parameters.get('risk_percentage', self.max_risk_per_trade)
        fixed_risk = risk_parameters.get('fixed_risk', 0)
        
        if fixed_risk > 0:
            # 使用固定风险金额
            risk_amount = min(fixed_risk, self.account_size * self.max_risk_per_trade)
        else:
            # 使用风险比例
            risk_amount = self.account_size * min(risk_percentage, self.max_risk_per_trade)
        
        return risk_amount
    
    def _calculate_base_position_size(self,
                                     entry_price: float,
                                     stop_loss: float,
                                     entry_type: str,
                                     risk_amount: float) -> float:
        """计算基础仓位规模"""
        # 计算每单位风险
        if entry_type == 'buy':
            risk_per_unit = entry_price - stop_loss
        else:  # sell
            risk_per_unit = stop_loss - entry_price
        
        if risk_per_unit <= 0:
            return 0
        
        # 基础仓位 = 风险金额 / 每单位风险
        base_size = risk_amount / risk_per_unit
        
        return base_size
    
    def _adjust_for_volatility(self,
                              base_size: float,
                              market_conditions: Dict) -> float:
        """基于波动性调整仓位"""
        volatility = market_conditions.get('volatility', 0.01)
        atr = market_conditions.get('atr', 0)
        avg_volatility = market_conditions.get('avg_volatility', 0.01)
        
        adjusted_size = base_size
        
        # 基于ATR调整
        if atr > 0 and avg_volatility > 0:
            volatility_ratio = atr / avg_volatility
            # 高波动性时减小仓位，低波动性时增大仓位
            if volatility_ratio > 1.5:
                adjusted_size *= 0.7
            elif volatility_ratio > 1.2:
                adjusted_size *= 0.85
            elif volatility_ratio < 0.8:
                adjusted_size *= 1.2
            elif volatility_ratio < 0.9:
                adjusted_size *= 1.1
        
        # 基于历史波动率调整
        if volatility > 0.02:  # 高波动性
            adjusted_size *= 0.8
        elif volatility < 0.005:  # 低波动性
            adjusted_size *= 1.3
        
        return adjusted_size
    
    def _adjust_for_market_conditions(self,
                                     current_size: float,
                                     market_conditions: Dict,
                                     entry_info: Dict) -> float:
        """基于市场条件调整仓位"""
        adjusted_size = current_size
        
        # 获取市场条件
        trend_strength = market_conditions.get('trend_strength', 0.5)
        market_structure = market_conditions.get('market_structure', 'unknown')
        confidence = entry_info.get('confidence', 0.5)
        
        # 基于趋势强度调整
        if trend_strength > 0.7:
            # 强趋势，可适当增加仓位
            adjusted_size *= min(1.2, 1.0 + (trend_strength - 0.7) * 0.5)
        elif trend_strength < 0.3:
            # 弱趋势或无趋势，减小仓位
            adjusted_size *= max(0.7, trend_strength * 2)
        
        # 基于市场结构调整
        if market_structure == 'uptrend' or market_structure == 'downtrend':
            # 趋势市场，正常仓位
            pass
        elif market_structure == 'range':
            # 区间市场，减小仓位
            adjusted_size *= 0.8
        elif market_structure == 'transition':
            # 转换期，大幅减小仓位
            adjusted_size *= 0.6
        
        # 基于交易信心度调整
        if confidence > 0.8:
            adjusted_size *= min(1.3, 1.0 + (confidence - 0.8) * 1.5)
        elif confidence < 0.5:
            adjusted_size *= max(0.5, confidence * 1.5)
        
        return adjusted_size
    
    def _adjust_for_account_status(self,
                                  current_size: float,
                                  account_size: float) -> float:
        """基于账户状态调整仓位"""
        adjusted_size = current_size
        
        # 基于账户规模调整（凯利公式简化版）
        optimal_fraction = self._calculate_kelly_fraction()
        max_by_account = account_size * optimal_fraction
        
        # 确保不超过账户限制
        if current_size > max_by_account:
            adjusted_size = max_by_account
        
        # 检查连续亏损后的调整
        if len(self.sizing_history) >= 3:
            recent_results = self.sizing_history[-3:]
            losing_trades = sum(1 for r in recent_results 
                              if r.get('detailed_metrics', {}).get('expected_pnl', 0) < 0)
            
            if losing_trades >= 2:
                # 连续亏损，减小仓位
                adjusted_size *= 0.7
        
        return adjusted_size
    
    def _calculate_kelly_fraction(self) -> float:
        """计算凯利分数（简化版）"""
        # 假设胜率55%，盈亏比1.5:1
        win_rate = 0.55
        win_loss_ratio = 1.5
        
        # 凯利公式：f = (bp - q) / b
        # 其中：b = 盈亏比，p = 胜率，q = 败率
        b = win_loss_ratio
        p = win_rate
        q = 1 - p
        
        if b <= 0:
            return self.max_position_size * 0.5
        
        kelly_fraction = (b * p - q) / b
        
        # 限制在合理范围内
        kelly_fraction = max(0.01, min(kelly_fraction, self.max_position_size))
        
        return kelly_fraction
    
    def _apply_position_limits(self, position_size: float) -> float:
        """应用仓位限制"""
        # 最大仓位限制
        max_size = self.account_size * self.max_position_size
        position_size = min(position_size, max_size)
        
        # 最小仓位限制（至少0.01手或等值）
        min_size = self.account_size * 0.001
        position_size = max(position_size, min_size)
        
        # 取整处理（如适用）
        if position_size > 10:
            position_size = round(position_size, 1)
        
        return position_size
    
    def _calculate_detailed_metrics(self,
                                   position_size: float,
                                   entry_price: float,
                                   stop_loss: float,
                                   entry_type: str,
                                   risk_amount: float) -> Dict:
        """计算详细指标"""
        # 计算风险比例
        risk_percentage = risk_amount / self.account_size
        
        # 计算头寸价值
        position_value = position_size * entry_price
        
        # 计算风险回报指标
        if entry_type == 'buy':
            risk_per_unit = entry_price - stop_loss
            potential_loss = position_size * risk_per_unit
        else:
            risk_per_unit = stop_loss - entry_price
            potential_loss = position_size * risk_per_unit
        
        # 计算预期盈亏（简化）
        win_rate = 0.55
        win_loss_ratio = 1.5
        expected_pnl = (win_rate * win_loss_ratio - (1 - win_rate)) * risk_amount
        
        return {
            'position_value': position_value,
            'risk_percentage': risk_percentage,
            'risk_per_unit': risk_per_unit,
            'potential_loss': potential_loss,
            'potential_loss_percentage': potential_loss / self.account_size,
            'position_to_account_ratio': position_value / self.account_size,
            'leverage_ratio': position_value / self.account_size,
            'expected_pnl': expected_pnl,
            'expected_pnl_percentage': expected_pnl / self.account_size,
            'risk_reward_ratio': win_loss_ratio,
            'win_rate': win_rate
        }
    
    def _generate_position_recommendation(self,
                                        position_size: float,
                                        metrics: Dict) -> Dict:
        """生成仓位建议"""
        position_value = metrics.get('position_value', 0)
        risk_percentage = metrics.get('risk_percentage', 0)
        leverage_ratio = metrics.get('leverage_ratio', 0)
        
        recommendation = 'hold'
        confidence = 0.7
        reason = ""
        
        if position_size <= 0:
            recommendation = 'avoid'
            confidence = 0.9
            reason = "仓位规模为零或负值，避免交易"
        elif risk_percentage > 0.03:
            recommendation = 'reduce'
            confidence = 0.8
            reason = f"风险比例过高 ({risk_percentage:.1%})，建议减小仓位"
        elif leverage_ratio > 0.2:
            recommendation = 'reduce'
            confidence = 0.75
            reason = f"杠杆比率过高 ({leverage_ratio:.1%})，建议减小仓位"
        elif position_value > self.account_size * 0.15:
            recommendation = 'reduce'
            confidence = 0.7
            reason = f"头寸价值超过账户15%，建议减小仓位"
        else:
            recommendation = 'proceed'
            confidence = 0.85
            reason = f"仓位规模合理，风险可控 ({risk_percentage:.1%}风险)"
        
        return {
            'action': recommendation,
            'confidence': confidence,
            'reason': reason,
            'suggested_adjustment': self._calculate_suggested_adjustment(
                recommendation, position_size, metrics
            )
        }
    
    def _calculate_suggested_adjustment(self,
                                       recommendation: str,
                                       current_size: float,
                                       metrics: Dict) -> Dict:
        """计算建议的调整"""
        if recommendation == 'proceed':
            adjustment = 0
            new_size = current_size
        elif recommendation == 'reduce':
            # 建议减小30%
            adjustment = -0.3
            new_size = current_size * (1 + adjustment)
        elif recommendation == 'avoid':
            adjustment = -1.0
            new_size = 0
        else:  # hold
            adjustment = 0
            new_size = current_size
        
        return {
            'adjustment_percentage': adjustment,
            'new_position_size': new_size,
            'adjustment_reason': '基于风险管理和市场条件的建议调整'
        }
    
    def reset_account_size(self, new_account_size: float):
        """重置账户规模"""
        if new_account_size > 0:
            self.account_size = new_account_size
    
    def update_risk_parameters(self, max_risk_per_trade: float = None,
                              max_position_size: float = None):
        """更新风险参数"""
        if max_risk_per_trade is not None and 0 < max_risk_per_trade <= 0.5:
            self.max_risk_per_trade = max_risk_per_trade
        
        if max_position_size is not None and 0 < max_position_size <= 0.5:
            self.max_position_size = max_position_size
    
    def get_sizing_history_summary(self) -> Dict:
        """获取仓位调整历史摘要"""
        if not self.sizing_history:
            return {'total_trades': 0, 'avg_position_size': 0}
        
        total_trades = len(self.sizing_history)
        avg_position_size = np.mean([r['final_position_size'] 
                                   for r in self.sizing_history])
        avg_risk = np.mean([r['risk_amount'] 
                          for r in self.sizing_history])
        
        return {
            'total_trades': total_trades,
            'avg_position_size': avg_position_size,
            'avg_risk_amount': avg_risk,
            'avg_risk_percentage': avg_risk / self.account_size,
            'recent_recommendations': [r['recommendation']['action'] 
                                     for r in self.sizing_history[-5:]]
        }


# ============================================================================
# 策略改造: 添加PriceActionRangesPositionSizingAdjusterStrategy类
# 将价格行为区间仓位规模调整系统转换为交易策略
# ============================================================================

class PriceActionRangesPositionSizingAdjusterStrategy(BaseStrategy):
    """价格行为区间仓位规模调整策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        account_size = params.get('account_size', 100000.0)
        max_risk_per_trade = params.get('max_risk_per_trade', 0.02)
        
        # 创建仓位规模调整器实例
        self.sizing_adjuster = PositionSizingAdjuster(
            account_size=account_size,
            max_risk_per_trade=max_risk_per_trade
        )
    
    def generate_signals(self):
        """
        生成交易信号

        基于仓位规模调整生成交易信号，使用ATR和波动率确定仓位方向
        """
        df = self.data
        if len(df) < 20:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(df['close'].iloc[-1]))
            return self.signals

        close = df['close']
        high = df['high']
        low = df['low']

        # ATR
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low - close.shift(1)).abs()
        }).max(axis=1)
        atr = tr.rolling(14).mean()

        # Volatility
        vol = close.pct_change().rolling(20).std() * np.sqrt(252)

        # Trend
        ma_short = close.rolling(10).mean()
        ma_long = close.rolling(30).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        last_vol = vol.iloc[-1]
        last_rsi = rsi.iloc[-1]

        if last_vol < 0.3 and ma_short.iloc[-1] > ma_long.iloc[-1] and last_rsi < 70:
            self._record_signal(timestamp=df.index[-1], action='buy', price=float(close.iloc[-1]))
        elif ma_short.iloc[-1] < ma_long.iloc[-1]:
            self._record_signal(timestamp=df.index[-1], action='sell', price=float(close.iloc[-1]))
        else:
            self._record_signal(timestamp=df.index[-1], action='hold', price=float(close.iloc[-1]))

        return self.signals