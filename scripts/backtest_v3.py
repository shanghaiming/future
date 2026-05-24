#!/usr/bin/env python3
"""
期货回测 v3 - 简单均线策略
- 20日均线上穿60日均线做多
- 20日均线下穿60日均线做空
- ATR止损
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class BacktestV3:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        self.max_positions = 10
        self.risk_per_trade = 0.05  # 5%风险
        
    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'): continue
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 计算指标
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma30'] = df['close'].rolling(30).mean()
            df['return'] = df['close'].pct_change()
            
            # ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr14'] = df['tr'].rolling(14).mean()
            
            # 信号
            df['signal'] = 0
            df['prev_ma20'] = df['ma20'].shift(1)
            df['prev_ma30'] = df['ma30'].shift(1)
            
            # 金叉
            df.loc[(df['ma20'] > df['ma30']) & (df['prev_ma20'] <= df['prev_ma30']), 'signal'] = 1
            # 死叉
            df.loc[(df['ma20'] < df['ma30']) & (df['prev_ma20'] >= df['prev_ma30']), 'signal'] = -1
            
            data[symbol] = df
        return data
    
    def run(self, data, start_date, end_date):
        print(f"=== 回测 v3 (均线策略) ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        
        # 生成交易日
        all_dates = set()
        for df in data.values():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            all_dates.update(df[mask]['trade_date'])
        dates = sorted(all_dates)
        
        for date in dates:
            # 处理每个品种
            for symbol, df in data.items():
                day_data = df[df['trade_date'] == date]
                if len(day_data) == 0:
                    continue
                
                row = day_data.iloc[0]
                price = row['close']
                signal = row['signal']
                atr = row['atr14']
                
                # 有信号
                if signal != 0:
                    # 平仓反向持仓
                    if symbol in self.positions:
                        pos = self.positions[symbol]
                        if pos['direction'] != signal:
                            self._close(symbol, date, price, '反向信号')
                        else:
                            continue
                    
                    # 开新仓
                    if len(self.positions) < self.max_positions and atr > 0:
                        size = self._calc_size(price, atr)
                        if size > 0:
                            self._open(symbol, signal, size, price, date, atr)
                
                # 检查止损
                if symbol in self.positions:
                    self._check_stop(symbol, date, price)
            
            # 计算权益
            unrealized = self._calc_unrealized(date, data)
            self.equity_curve.append((date, self.capital + unrealized))
        
        return self._get_results()
    
    def _calc_size(self, price, atr):
        risk = self.capital * self.risk_per_trade
        stop_dist = atr * 2
        size = risk / (stop_dist * 10)  # 每手10吨
        max_size = (self.capital * 0.15) / (price * 10)
        return min(size, max_size)
    
    def _open(self, symbol, direction, size, price, date, atr):
        commission = price * size * 10 * 0.0001
        self.capital -= commission
        self.positions[symbol] = {
            'direction': direction,
            'size': size,
            'entry_price': price,
            'atr': atr,
            'stop': price - atr * 2 * direction
        }
        self.trades.append({
            'date': date, 'symbol': symbol, 'direction': direction,
            'size': size, 'price': price, 'pnl': 0, 'commission': commission,
            'type': 'open'
        })
    
    def _close(self, symbol, date, price, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
        commission = price * pos['size'] * 10 * 0.0001
        self.capital += pnl - commission
        
        self.trades.append({
            'date': date, 'symbol': symbol, 'direction': -pos['direction'],
            'size': pos['size'], 'price': price, 'pnl': pnl,
            'commission': commission, 'type': 'close', 'reason': reason
        })
        del self.positions[symbol]
    
    def _check_stop(self, symbol, date, price):
        pos = self.positions[symbol]
        if pos['direction'] == 1 and price < pos['stop']:
            self._close(symbol, date, price, '止损')
        elif pos['direction'] == -1 and price > pos['stop']:
            self._close(symbol, date, price, '止损')
    
    def _calc_unrealized(self, date, data):
        total = 0
        for symbol, pos in self.positions.items():
            df = data.get(symbol)
            if df is None: continue
            day_data = df[df['trade_date'] == date]
            if len(day_data) == 0: continue
            price = day_data.iloc[0]['close']
            total += (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
        return total
    
    def _get_results(self):
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        equity_df['return'] = equity_df['equity'].pct_change()
        
        total_return = (self.equity_curve[-1][1] - self.initial_capital) / self.initial_capital
        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        years = max(days / 365, 0.001)
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        daily_ret = equity_df['return'].dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        
        trades_df = pd.DataFrame(self.trades)
        close_trades = trades_df[trades_df['type'] == 'close']
        
        wins = close_trades[close_trades['pnl'] > 0]
        losses = close_trades[close_trades['pnl'] <= 0]
        win_rate = len(wins) / len(close_trades) if len(close_trades) > 0 else 0
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity_curve[-1][1],
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(close_trades),
            'win_rate': win_rate,
            'profit_factor': avg_win / avg_loss if avg_loss > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'equity_curve': self.equity_curve
        }
    
    def print_results(self, results):
        print("\n" + "="*50)
        print("回测结果 v3 (均线策略)")
        print("="*50)
        print(f"初始资金:     {results['initial_capital']:>15,.0f}")
        print(f"最终权益:     {results['final_equity']:>15,.0f}")
        print(f"总收益率:     {results['total_return']:>15.2%}")
        print(f"年化收益率:   {results['annual_return']:>15.2%}")
        print(f"最大回撤:     {results['max_drawdown']:>15.2%}")
        print(f"夏普比率:     {results['sharpe_ratio']:>15.2f}")
        print(f"总交易次数:   {results['total_trades']:>15}")
        print(f"胜率:         {results['win_rate']:>15.2%}")
        print(f"盈亏比:       {results['profit_factor']:>15.2f}")
        print("="*50)


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    
    bt = BacktestV3(initial_capital=500000)
    
    print("加载数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    results = bt.run(data, start_date, end_date)
    bt.print_results(results)
    
    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'backtest_v3.json'), 'w') as f:
        json.dump({
            'initial_capital': results['initial_capital'],
            'final_equity': results['final_equity'],
            'total_return': results['total_return'],
            'annual_return': results['annual_return'],
            'max_drawdown': results['max_drawdown'],
            'sharpe_ratio': results['sharpe_ratio'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
        }, f, indent=2)


if __name__ == '__main__':
    main()
