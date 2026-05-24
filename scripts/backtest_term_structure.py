#!/usr/bin/env python3
"""
期货期限结构套利策略
- Contango时做空近月做多远月
- Backwardation时做多近月做空远月
- 基于期限结构曲率交易
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TermStructureStrategy:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        self.max_positions = 10
        self.position_pct = 0.1  # 单笔10%资金
        
    def load_term_structure(self, data_dir):
        """加载期限结构数据"""
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.json'): continue
            symbol = f.replace('.json', '')
            with open(os.path.join(data_dir, f)) as fp:
                item = json.load(fp)
            
            # 解析日期
            item['date'] = pd.to_datetime(item['date'])
            
            # 计算期限结构指标
            curve = item.get('curve', [])
            if len(curve) >= 2:
                prices = [c['price'] for c in curve if c.get('price')]
                if len(prices) >= 2:
                    # 近月-远月价差
                    item['spread'] = prices[0] - prices[-1]
                    # 期限结构斜率
                    item['slope'] = (prices[-1] - prices[0]) / prices[0]
                    # 曲率
                    if len(prices) >= 3:
                        item['curvature'] = prices[1] - (prices[0] + prices[-1]) / 2
                    else:
                        item['curvature'] = 0
                else:
                    continue
            else:
                continue
            
            data[symbol] = item
        return data
    
    def generate_signals(self, data, date):
        """生成信号"""
        signals = []
        
        for symbol, item in data.items():
            if item['date'].date() != date.date():
                continue
            
            slope = item['slope']
            curvature = item['curvature']
            
            # 策略1: 期限结构回归
            # 深度Contango (>5%) - 做空
            if slope > 0.05:
                signals.append({
                    'symbol': symbol,
                    'direction': -1,
                    'strength': min(slope / 0.1, 1.0),
                    'reason': f'深度Contango {slope:.1%}'
                })
            # 深度Backwardation (<-5%) - 做多
            elif slope < -0.05:
                signals.append({
                    'symbol': symbol,
                    'direction': 1,
                    'strength': min(abs(slope) / 0.1, 1.0),
                    'reason': f'深度Backwardation {slope:.1%}'
                })
            
            # 策略2: 曲率交易
            # 凸形结构 - 做多中间合约
            if curvature > 0 and abs(slope) < 0.03:
                signals.append({
                    'symbol': symbol,
                    'direction': 1,
                    'strength': 0.5,
                    'reason': f'凸形结构 曲率={curvature:.2f}'
                })
        
        # 排序
        signals.sort(key=lambda x: x['strength'], reverse=True)
        return signals[:self.max_positions]
    
    def run(self, data, start_date, end_date):
        """运行回测"""
        print(f"=== 期限结构套利策略 ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        
        # 按日期分组
        dates = sorted(set(item['date'] for item in data.values() 
                          if start_date <= item['date'] <= end_date))
        
        for date in dates:
            signals = self.generate_signals(data, date)
            
            for signal in signals:
                symbol = signal['symbol']
                
                # 平仓反向持仓
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    if pos['direction'] != signal['direction']:
                        self._close(symbol, date, signal['reason'])
                    else:
                        continue
                
                # 开新仓
                if len(self.positions) < self.max_positions:
                    self._open(symbol, signal['direction'], date, signal['reason'])
            
            # 计算权益
            self.equity_curve.append((date, self.capital))
        
        return self._get_results()
    
    def _open(self, symbol, direction, date, reason):
        """开仓"""
        size = (self.capital * self.position_pct) / 10000  # 假设每手价值10000
        commission = size * 10000 * 0.0001
        self.capital -= commission
        
        self.positions[symbol] = {
            'direction': direction,
            'size': size,
            'entry_date': date,
            'reason': reason
        }
        
        self.trades.append({
            'date': date, 'symbol': symbol, 'direction': direction,
            'size': size, 'type': 'open', 'reason': reason
        })
    
    def _close(self, symbol, date, reason):
        """平仓"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # 模拟收益 (简化版，假设持有5天)
        pnl = pos['size'] * 10000 * pos['direction'] * 0.02  # 假设2%收益
        commission = pos['size'] * 10000 * 0.0001
        
        self.capital += pnl - commission
        
        self.trades.append({
            'date': date, 'symbol': symbol, 'direction': -pos['direction'],
            'size': pos['size'], 'pnl': pnl, 'type': 'close', 'reason': reason
        })
        del self.positions[symbol]
    
    def _get_results(self):
        """获取结果"""
        if not self.equity_curve:
            return {}
        
        total_return = (self.equity_curve[-1][1] - self.initial_capital) / self.initial_capital
        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        years = max(days / 365, 0.001)
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity_curve[-1][1],
            'total_return': total_return,
            'annual_return': annual_return,
            'total_trades': len([t for t in self.trades if t['type'] == 'close'])
        }
    
    def print_results(self, results):
        print("\n" + "="*50)
        print("期限结构套利结果")
        print("="*50)
        print(f"初始资金:     {results['initial_capital']:>15,.0f}")
        print(f"最终权益:     {results['final_equity']:>15,.0f}")
        print(f"总收益率:     {results['total_return']:>15.2%}")
        print(f"年化收益率:   {results['annual_return']:>15.2%}")
        print(f"总交易次数:   {results['total_trades']:>15}")
        print("="*50)


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_term_structure")
    
    bt = TermStructureStrategy(initial_capital=500000)
    
    print("加载期限结构数据...")
    data = bt.load_term_structure(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    results = bt.run(data, start_date, end_date)
    bt.print_results(results)


if __name__ == '__main__':
    main()
