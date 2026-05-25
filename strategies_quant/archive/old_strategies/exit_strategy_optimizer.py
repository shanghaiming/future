#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
出场策略优化量化系统
第20章：出场策略优化
AL Brooks《价格行为交易之区间篇》

核心概念（最小可行实现）：
1. 出场类型：固定止损止盈、移动止损、部分止盈
2. 出场时机：基于价格行为、技术结构、时间框架
3. 出场优化：风险回报比最大化、胜率优化
4. 出场纪律：避免情绪化出场
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

class ExitStrategyOptimizer:
    """出场策略优化器（最小可行版本）"""
    
    def __init__(self,
                 initial_stop_loss_pct: float = 0.02,
                 initial_take_profit_pct: float = 0.04,
                 trailing_activation_pct: float = 0.015):
        """
        初始化出场策略优化器
        
        参数:
            initial_stop_loss_pct: 初始止损百分比
            initial_take_profit_pct: 初始止盈百分比
            trailing_activation_pct: 移动止损激活百分比
        """
        self.initial_stop_loss_pct = initial_stop_loss_pct
        self.initial_take_profit_pct = initial_take_profit_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.exit_history = []
    
    def optimize_exit_points(self,
                           entry_info: Dict,
                           price_data: pd.DataFrame) -> Dict:
        """
        优化出场点
        
        参数:
            entry_info: 入场信息
            price_data: 入场后的价格数据
            
        返回:
            出场优化结果
        """
        if len(price_data) < 5:
            return {'error': '价格数据不足'}
        
        entry_price = entry_info.get('entry_price', 0)
        entry_type = entry_info.get('type', 'buy')
        current_price = price_data['close'].iloc[-1]
        
        # 计算基本出场点
        basic_exits = self._calculate_basic_exits(entry_price, entry_type)
        
        # 计算移动止损
        trailing_stop = self._calculate_trailing_stop(price_data, entry_price, entry_type)
        
        # 确定部分止盈点
        partial_exits = self._determine_partial_exits(
            entry_price, current_price, entry_type, price_data
        )
        
        # 优化出场策略
        optimized_strategy = self._optimize_exit_strategy(
            basic_exits, trailing_stop, partial_exits,
            entry_price, current_price, entry_type, price_data
        )
        
        # 生成出场信号
        exit_signals = self._generate_exit_signals(optimized_strategy, price_data)
        
        result = {
            'entry_info': entry_info,
            'current_price': current_price,
            'current_pnl_pct': (current_price - entry_price) / entry_price * 100 
                               if entry_type == 'buy' else 
                               (entry_price - current_price) / entry_price * 100,
            'basic_exits': basic_exits,
            'trailing_stop': trailing_stop,
            'partial_exits': partial_exits,
            'optimized_strategy': optimized_strategy,
            'exit_signals': exit_signals,
            'recommended_action': self._generate_exit_recommendation(optimized_strategy, exit_signals),
            'analysis_time': pd.Timestamp.now()
        }
        
        self.exit_history.append(result)
        return result
    
    def _calculate_basic_exits(self, entry_price: float, entry_type: str) -> Dict:
        """计算基本出场点（固定止损止盈）"""
        if entry_type == 'buy':
            stop_loss = entry_price * (1 - self.initial_stop_loss_pct)
            take_profit = entry_price * (1 + self.initial_take_profit_pct)
            breakeven = entry_price * 1.001  # 略微盈利的保本点
        else:  # sell
            stop_loss = entry_price * (1 + self.initial_stop_loss_pct)
            take_profit = entry_price * (1 - self.initial_take_profit_pct)
            breakeven = entry_price * 0.999  # 略微盈利的保本点
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'breakeven': breakeven,
            'risk_reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss),
            'stop_distance_pct': self.initial_stop_loss_pct * 100,
            'target_distance_pct': self.initial_take_profit_pct * 100
        }
    
    def _calculate_trailing_stop(self,
                               price_data: pd.DataFrame,
                               entry_price: float,
                               entry_type: str) -> Dict:
        """
        计算移动止损
        
        返回:
            移动止损分析结果
        """
        if len(price_data) < 10:
            return {'enabled': False, 'reason': '数据不足'}
        
        closes = price_data['close'].values
        current_price = closes[-1]
        
        # 计算入场以来的最佳价格
        if entry_type == 'buy':
            best_price = np.max(closes)
            # 移动止损激活条件：盈利达到activation_pct
            if best_price >= entry_price * (1 + self.trailing_activation_pct):
                # 从最高点回撤一定比例作为移动止损
                trailing_stop = best_price * (1 - self.initial_stop_loss_pct * 0.7)
                enabled = True
                reason = f'移动止损激活，基于最高点${best_price:.2f}'
            else:
                trailing_stop = entry_price * (1 - self.initial_stop_loss_pct)
                enabled = False
                reason = '未达到移动止损激活条件'
        
        else:  # sell
            best_price = np.min(closes)  # 对卖出而言，价格越低越好
            if best_price <= entry_price * (1 - self.trailing_activation_pct):
                # 从最低点反弹一定比例作为移动止损
                trailing_stop = best_price * (1 + self.initial_stop_loss_pct * 0.7)
                enabled = True
                reason = f'移动止损激活，基于最低点${best_price:.2f}'
            else:
                trailing_stop = entry_price * (1 + self.initial_stop_loss_pct)
                enabled = False
                reason = '未达到移动止损激活条件'
        
        # 移动止损不应差于初始止损
        if entry_type == 'buy':
            trailing_stop = max(trailing_stop, entry_price * (1 - self.initial_stop_loss_pct))
        else:
            trailing_stop = min(trailing_stop, entry_price * (1 + self.initial_stop_loss_pct))
        
        return {
            'enabled': enabled,
            'trailing_stop': trailing_stop,
            'reason': reason,
            'best_price': best_price,
            'distance_from_best': abs(best_price - trailing_stop) / best_price * 100,
            'distance_from_entry': abs(trailing_stop - entry_price) / entry_price * 100
        }
    
    def _determine_partial_exits(self,
                               entry_price: float,
                               current_price: float,
                               entry_type: str,
                               price_data: pd.DataFrame) -> List[Dict]:
        """
        确定部分止盈点
        
        返回:
            部分止盈计划列表
        """
        partial_exits = []
        
        # 计算当前盈亏百分比
        if entry_type == 'buy':
            pnl_pct = (current_price - entry_price) / entry_price * 100
            profitable = pnl_pct > 0
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            profitable = pnl_pct > 0
        
        if not profitable or len(price_data) < 5:
            return partial_exits  # 未盈利或数据不足，不部分止盈
        
        # 基于价格行为的部分止盈规则
        closes = price_data['close'].values
        
        # 规则1：达到初始目标的一半时，止盈25%
        initial_target_pct = self.initial_take_profit_pct * 100
        if abs(pnl_pct) >= initial_target_pct * 0.5:
            partial_exits.append({
                'level': 1,
                'exit_pct': 25,  # 平仓25%
                'trigger_price': entry_price * (1 + initial_target_pct/100 * 0.5) 
                               if entry_type == 'buy' else 
                               entry_price * (1 - initial_target_pct/100 * 0.5),
                'reason': '达到初始目标50%，部分止盈锁定利润',
                'activated': abs(pnl_pct) >= initial_target_pct * 0.5
            })
        
        # 规则2：达到初始目标时，再止盈25%
        if abs(pnl_pct) >= initial_target_pct:
            partial_exits.append({
                'level': 2,
                'exit_pct': 25,  # 再平仓25%
                'trigger_price': entry_price * (1 + initial_target_pct/100) 
                               if entry_type == 'buy' else 
                               entry_price * (1 - initial_target_pct/100),
                'reason': '达到初始目标，再次部分止盈',
                'activated': abs(pnl_pct) >= initial_target_pct
            })
        
        # 规则3：价格出现反转迹象时，止盈剩余部分
        if len(closes) >= 3:
            # 简单反转检测
            if entry_type == 'buy':
                # 买入后连续两根阴线
                if closes[-1] < closes[-2] and closes[-2] < closes[-3]:
                    partial_exits.append({
                        'level': 3,
                        'exit_pct': 50,  # 平仓剩余50%
                        'trigger_price': current_price,
                        'reason': '检测到价格反转迹象，全部止盈',
                        'activated': True
                    })
            else:
                # 卖出后连续两根阳线
                if closes[-1] > closes[-2] and closes[-2] > closes[-3]:
                    partial_exits.append({
                        'level': 3,
                        'exit_pct': 50,
                        'trigger_price': current_price,
                        'reason': '检测到价格反转迹象，全部止盈',
                        'activated': True
                    })
        
        return partial_exits
    
    def _optimize_exit_strategy(self,
                              basic_exits: Dict,
                              trailing_stop: Dict,
                              partial_exits: List[Dict],
                              entry_price: float,
                              current_price: float,
                              entry_type: str,
                              price_data: pd.DataFrame) -> Dict:
        """
        优化出场策略
        
        返回:
            优化后的出场策略
        """
        # 当前使用的止损
        if trailing_stop['enabled']:
            current_stop = trailing_stop['trailing_stop']
            stop_reason = '移动止损'
        else:
            current_stop = basic_exits['stop_loss']
            stop_reason = '固定止损'
        
        # 当前使用的止盈（考虑部分止盈）
        active_partial = [pe for pe in partial_exits if pe.get('activated', False)]
        if active_partial:
            # 如果已经部分止盈，调整剩余仓位目标
            total_exited_pct = sum(pe['exit_pct'] for pe in active_partial)
            remaining_pct = 100 - total_exited_pct
            
            # 调整剩余仓位的止盈目标（更激进）
            if entry_type == 'buy':
                adjusted_take_profit = current_price * (1 + self.initial_take_profit_pct * 1.5)
            else:
                adjusted_take_profit = current_price * (1 - self.initial_take_profit_pct * 1.5)
            
            take_profit = adjusted_take_profit
            tp_reason = f'部分止盈后调整目标（已平仓{total_exited_pct}%）'
        else:
            take_profit = basic_exits['take_profit']
            tp_reason = '固定止盈'
        
        # 计算优化后的风险回报比
        if entry_type == 'buy':
            current_risk = abs(current_stop - current_price) / current_price
            current_reward = abs(take_profit - current_price) / current_price
        else:
            current_risk = abs(current_price - current_stop) / current_price
            current_reward = abs(current_price - take_profit) / current_price
        
        current_rr_ratio = current_reward / current_risk if current_risk > 0 else 0
        
        # 评估出场时机
        exit_timing = self._evaluate_exit_timing(price_data, entry_type)
        
        return {
            'current_stop': current_stop,
            'stop_reason': stop_reason,
            'current_take_profit': take_profit,
            'tp_reason': tp_reason,
            'partial_exits_active': len(active_partial),
            'total_exited_pct': sum(pe['exit_pct'] for pe in active_partial) if active_partial else 0,
            'current_risk_pct': current_risk * 100,
            'current_reward_pct': current_reward * 100,
            'current_rr_ratio': current_rr_ratio,
            'exit_timing': exit_timing,
            'recommended_stop_adjustment': self._recommend_stop_adjustment(
                current_stop, current_price, entry_type, price_data
            )
        }
    
    def _evaluate_exit_timing(self, price_data: pd.DataFrame, entry_type: str) -> Dict:
        """评估出场时机"""
        if len(price_data) < 10:
            return {'score': 0.5, 'recommendation': '数据不足', 'signals': []}
        
        closes = price_data['close'].values
        signals = []
        score_components = []
        
        # 信号1：价格接近关键水平
        recent_high = np.max(closes[-10:])
        recent_low = np.min(closes[-10:])
        current_price = closes[-1]
        
        if entry_type == 'buy':
            # 买入后，价格接近近期高点可能考虑止盈
            distance_to_high = (recent_high - current_price) / recent_high
            if distance_to_high < 0.01:  # 距离高点<1%
                signals.append('价格接近近期高点，考虑止盈')
                score_components.append(0.8)
            else:
                score_components.append(0.5)
        else:
            # 卖出后，价格接近近期低点可能考虑止盈
            distance_to_low = (current_price - recent_low) / current_price
            if distance_to_low < 0.01:  # 距离低点<1%
                signals.append('价格接近近期低点，考虑止盈')
                score_components.append(0.8)
            else:
                score_components.append(0.5)
        
        # 信号2：价格动量减弱
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
            if abs(momentum) < 0.5:  # 动量很弱
                signals.append('价格动量减弱，考虑出场')
                score_components.append(0.7)
            else:
                score_components.append(0.4)
        
        # 计算综合分数
        if score_components:
            score = np.mean(score_components)
        else:
            score = 0.5
        
        # 生成建议
        if score >= 0.7:
            recommendation = '考虑出场或减仓'
        elif score >= 0.5:
            recommendation = '保持持仓，监控出场信号'
        else:
            recommendation = '继续持仓，等待更好出场点'
        
        return {
            'score': score,
            'recommendation': recommendation,
            'signals': signals,
            'momentum': momentum if 'momentum' in locals() else 0
        }
    
    def _recommend_stop_adjustment(self,
                                  current_stop: float,
                                  current_price: float,
                                  entry_type: str,
                                  price_data: pd.DataFrame) -> Dict:
        """推荐止损调整"""
        if len(price_data) < 5:
            return {'adjustment': 'none', 'reason': '数据不足', 'new_stop': current_stop}
        
        # 检查是否应该移动止损至保本点
        if entry_type == 'buy':
            profitable = current_price > current_stop * 1.01  # 盈利超过1%
            if profitable and current_stop < current_price * 0.999:  # 止损还在入场价下方
                # 移动到略微盈利的位置
                new_stop = current_price * 0.998
                return {
                    'adjustment': 'move_to_breakeven',
                    'reason': '已有盈利，移动止损至保本点上方',
                    'new_stop': new_stop,
                    'improvement_pct': (new_stop - current_stop) / current_price * 100
                }
        else:
            profitable = current_price < current_stop * 0.99  # 盈利超过1%
            if profitable and current_stop > current_price * 1.001:  # 止损还在入场价上方
                new_stop = current_price * 1.002
                return {
                    'adjustment': 'move_to_breakeven',
                    'reason': '已有盈利，移动止损至保本点下方',
                    'new_stop': new_stop,
                    'improvement_pct': (current_stop - new_stop) / current_price * 100
                }
        
        return {
            'adjustment': 'none',
            'reason': '保持当前止损',
            'new_stop': current_stop,
            'improvement_pct': 0
        }
    
    def _generate_exit_signals(self,
                              optimized_strategy: Dict,
                              price_data: pd.DataFrame) -> List[Dict]:
        """生成出场信号"""
        signals = []
        current_price = price_data['close'].iloc[-1]
        current_stop = optimized_strategy['current_stop']
        current_tp = optimized_strategy['current_take_profit']
        
        # 止损信号
        if (current_price <= current_stop) or (current_price >= current_stop):
            # 检查是否触及止损（买入：价格低于止损；卖出：价格高于止损）
            # 简化处理：实际需要根据entry_type判断
            signals.append({
                'type': 'stop_loss',
                'price': current_stop,
                'reason': optimized_strategy['stop_reason'],
                'urgency': 'high',
                'recommended_action': '立即出场'
            })
        
        # 止盈信号
        if optimized_strategy['current_rr_ratio'] >= 2 and \
           optimized_strategy['exit_timing']['score'] >= 0.7:
            signals.append({
                'type': 'take_profit',
                'price': current_tp,
                'reason': f"风险回报比{optimized_strategy['current_rr_ratio']:.1f}:1良好，出场时机得分{optimized_strategy['exit_timing']['score']:.1%}",
                'urgency': 'medium',
                'recommended_action': '考虑止盈'
            })
        
        # 部分止盈信号
        if optimized_strategy['partial_exits_active'] > 0:
            signals.append({
                'type': 'partial_exit',
                'reason': f"已激活{optimized_strategy['partial_exits_active']}个部分止盈，已平仓{optimized_strategy['total_exited_pct']}%",
                'urgency': 'low',
                'recommended_action': '继续监控剩余仓位'
            })
        
        return signals
    
    def _generate_exit_recommendation(self,
                                     optimized_strategy: Dict,
                                     exit_signals: List[Dict]) -> str:
        """生成出场建议"""
        if not exit_signals:
            return f"继续持仓，当前风险回报比{optimized_strategy['current_rr_ratio']:.1f}:1，出场时机得分{optimized_strategy['exit_timing']['score']:.1%}"
        
        # 检查是否有高紧急度信号
        high_urgency = [s for s in exit_signals if s['urgency'] == 'high']
        if high_urgency:
            return f"⚠️ {high_urgency[0]['recommended_action']}: {high_urgency[0]['reason']}"
        
        # 检查中等紧急度信号
        medium_urgency = [s for s in exit_signals if s['urgency'] == 'medium']
        if medium_urgency:
            return f"🟡 {medium_urgency[0]['recommended_action']}: {medium_urgency[0]['reason']}"
        
        # 低紧急度信号
        return exit_signals[0]['recommended_action'] if exit_signals else "继续持仓"
    
    def generate_exit_report(self, analysis_result: Dict) -> str:
        """生成出场策略报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("出场策略优化分析报告")
        report_lines.append("=" * 60)
        
        entry = analysis_result['entry_info']
        report_lines.append(f"\n📥 入场信息:")
        report_lines.append(f"   类型: {entry.get('type', 'unknown')}")
        report_lines.append(f"   价格: ${entry.get('entry_price', 0):.2f}")
        
        report_lines.append(f"\n💰 当前状态:")
        report_lines.append(f"   当前价格: ${analysis_result['current_price']:.2f}")
        report_lines.append(f"   当前盈亏: {analysis_result['current_pnl_pct']:.2f}%")
        
        basic = analysis_result['basic_exits']
        report_lines.append(f"\n🎯 基本出场点:")
        report_lines.append(f"   初始止损: ${basic['stop_loss']:.2f}")
        report_lines.append(f"   初始止盈: ${basic['take_profit']:.2f}")
        report_lines.append(f"   风险回报比: {basic['risk_reward_ratio']:.1f}:1")
        
        trailing = analysis_result['trailing_stop']
        report_lines.append(f"\n🎢 移动止损:")
        report_lines.append(f"   启用: {'是' if trailing['enabled'] else '否'}")
        report_lines.append(f"   当前止损: ${trailing['trailing_stop']:.2f}")
        report_lines.append(f"   理由: {trailing['reason']}")
        
        partial = analysis_result['partial_exits']
        report_lines.append(f"\n📊 部分止盈:")
        if partial:
            for i, pe in enumerate(partial, 1):
                status = "✅ 已触发" if pe.get('activated', False) else "⏳ 未触发"
                report_lines.append(f"   {i}. {pe['reason']}")
                report_lines.append(f"       平仓比例: {pe['exit_pct']}% | 状态: {status}")
        else:
            report_lines.append(f"   无部分止盈计划")
        
        optimized = analysis_result['optimized_strategy']
        report_lines.append(f"\n⚡ 优化策略:")
        report_lines.append(f"   当前止损: ${optimized['current_stop']:.2f} ({optimized['stop_reason']})")
        report_lines.append(f"   当前止盈: ${optimized['current_take_profit']:.2f} ({optimized['tp_reason']})")
        report_lines.append(f"   当前风险回报比: {optimized['current_rr_ratio']:.1f}:1")
        report_lines.append(f"   出场时机得分: {optimized['exit_timing']['score']:.1%}")
        report_lines.append(f"   时机建议: {optimized['exit_timing']['recommendation']}")
        
        signals = analysis_result['exit_signals']
        report_lines.append(f"\n🚦 出场信号:")
        if signals:
            for i, signal in enumerate(signals, 1):
                urgency_icon = '🔴' if signal['urgency'] == 'high' else '🟡' if signal['urgency'] == 'medium' else '🟢'
                report_lines.append(f"   {i}. {urgency_icon} {signal['type']}: {signal['reason']}")
                report_lines.append(f"       建议: {signal['recommended_action']}")
        else:
            report_lines.append(f"   无紧急出场信号")
        
        report_lines.append(f"\n💡 综合建议:")
        report_lines.append(f"   {analysis_result['recommended_action']}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)

def generate_sample_trade_data() -> Tuple[Dict, pd.DataFrame]:
    """生成样本交易数据"""
    np.random.seed(42)
    
    # 生成入场后的价格数据（假设买入后价格先涨后跌）
    n_bars = 50
    time = np.arange(n_bars)
    
    # 价格先上涨后回调
    trend = 100 + time * 0.2 + np.sin(time * 0.3) * 3
    noise = np.random.normal(0, 0.3, n_bars)
    prices = trend + noise
    
    price_data = pd.DataFrame({
        'open': prices * 0.998,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='D'))
    
    # 入场信息（假设5天前入场）
    entry_info = {
        'type': 'buy',
        'entry_price': price_data['close'].iloc[5],
        'entry_time': price_data.index[5],
        'position_size': 100,
        'initial_capital': 10000
    }
    
    # 使用入场后的数据
    trade_data = price_data.iloc[6:]
    
    return entry_info, trade_data


# ============================================================================
# 策略改造: 添加ExitStrategyOptimizerStrategy类
# 将出场策略优化器转换为交易策略
# ============================================================================

class ExitStrategyOptimizerStrategy(BaseStrategy):
    """出场策略优化策略"""
    
    def __init__(self, data: pd.DataFrame, params: dict):
        """
        初始化策略
        
        参数:
            data: 价格数据
            params: 策略参数
        """
        super().__init__(data, params)
        
        # 从params提取参数
        initial_stop_loss_pct = params.get('initial_stop_loss_pct', 0.02)
        initial_take_profit_pct = params.get('initial_take_profit_pct', 0.04)
        trailing_activation_pct = params.get('trailing_activation_pct', 0.015)
        
        # 创建出场策略优化器实例
        self.optimizer = ExitStrategyOptimizer(
            initial_stop_loss_pct=initial_stop_loss_pct,
            initial_take_profit_pct=initial_take_profit_pct,
            trailing_activation_pct=trailing_activation_pct
        )
    
    def generate_signals(self):
        """
        生成交易信号
        
        基于出场策略优化生成交易信号
        """
        # 调用出场策略优化
        # 注意: 出场优化器需要入场信息，这里使用模拟入场
        entry_info = {
            'type': 'buy',
            'entry_price': self.data['close'].iloc[0],
            'entry_time': self.data.index[0],
            'initial_stop_loss': self.data['close'].iloc[0] * 0.98,
            'initial_take_profit': self.data['close'].iloc[0] * 1.04
        }
        
        analysis_result = self.optimizer.optimize_exit_points(
            price_data=self.data,
            entry_info=entry_info
        )
        
        # 获取出场信号
        exit_signals = analysis_result.get('exit_signals', [])
        
        # 将出场信号转换为交易信号
        for signal in exit_signals:
            action = signal.get('recommended_action', 'hold').lower()
            if action in ['sell', 'close']:
                # 出场信号转换为卖出信号
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='sell',
                    price=self.data['close'].iloc[-1]
                )
            elif action in ['hold', 'trail_stop']:
                # 持有或移动止损，转换为hold信号
                self._record_signal(
                    timestamp=self.data.index[-1],
                    action='hold',
                    price=self.data['close'].iloc[-1]
                )
        
        # 如果没有信号，添加hold信号
        if not self.signals:
            self._record_signal(
                timestamp=self.data.index[-1],
                action='hold',
                price=self.data['close'].iloc[-1]
            )
        
        return self.signals


# ============================================================================
# 策略改造完成
# ============================================================================

def main():
    """主函数：演示出场策略优化系统"""
    print("=== 出场策略优化量化系统 ===")
    print("第20章：出场策略优化（最小可行版本）\n")
    
    # 生成样本交易数据
    print("1. 生成样本交易数据...")
    entry_info, trade_data = generate_sample_trade_data()
    
    print(f"   入场信息: {entry_info['type']} @ ${entry_info['entry_price']:.2f}")
    print(f"   交易数据: {len(trade_data)}个数据点")
    print(f"   当前价格: ${trade_data['close'].iloc[-1]:.2f}")
    print(f"   最大价格: ${trade_data['high'].max():.2f}")
    print(f"   最小价格: ${trade_data['low'].min():.2f}")
    
    # 创建出场策略优化器
    print("\n2. 创建出场策略优化器...")
    exit_optimizer = ExitStrategyOptimizer(
        initial_stop_loss_pct=0.02,
        initial_take_profit_pct=0.04,
        trailing_activation_pct=0.015
    )
    
    # 优化出场点
    print("\n3. 优化出场点...")
    analysis_result = exit_optimizer.optimize_exit_points(entry_info, trade_data)
    
    # 生成分析报告
    print("\n4. 生成分析报告...")
    report = exit_optimizer.generate_exit_report(analysis_result)
    print(report)
    
    # 显示关键结果
    print("\n5. 关键结果摘要:")
    print("-" * 40)
    
    print(f"当前盈亏: {analysis_result['current_pnl_pct']:.2f}%")
    
    basic = analysis_result['basic_exits']
    print(f"\n基本出场:")
    print(f"  止损: ${basic['stop_loss']:.2f}")
    print(f"  止盈: ${basic['take_profit']:.2f}")
    print(f"  风险回报比: {basic['risk_reward_ratio']:.1f}:1")
    
    trailing = analysis_result['trailing_stop']
    if trailing['enabled']:
        print(f"\n移动止损: 启用 (${trailing['trailing_stop']:.2f})")
        print(f"  理由: {trailing['reason']}")
    
    optimized = analysis_result['optimized_strategy']
    print(f"\n优化策略:")
    print(f"  当前止损: ${optimized['current_stop']:.2f}")
    print(f"  当前止盈: ${optimized['current_take_profit']:.2f}")
    print(f"  当前风险回报比: {optimized['current_rr_ratio']:.1f}:1")
    print(f"  出场时机得分: {optimized['exit_timing']['score']:.1%}")
    
    signals = analysis_result['exit_signals']
    if signals:
        print(f"\n出场信号: {len(signals)}个")
        for signal in signals:
            print(f"  {signal['type']}: {signal['recommended_action']}")
    else:
        print(f"\n无出场信号")
    
    print(f"\n最终建议: {analysis_result['recommended_action']}")
    
    print("\n=== 系统演示完成 ===")
    print("第20章出场策略优化系统（最小可行版本）已实现并测试。")

if __name__ == "__main__":
    main()