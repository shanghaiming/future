#!/usr/bin/env python3
"""
期货趋势策略 v5
- 只交易趋势性最强的品种 (ag, au, sn, bc, cu)
- MA20/MA30趋势跟踪
- 盈利加仓
- 移动止损
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TrendBacktest:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # 只交易这些品种
        self.target_symbols = ['agfi', 'aufi', 'snfi', 'bcfi', 'cufi']
        
        self.max_positions = 3
        self.position_pct = 0.3  # 每品种30%资金
        
    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'): continue
            symbol = f.replace('.csv', '')
            if symbol not in self.target_symbols:
                continue
            
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
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
            
            df.loc[(df['ma20'] > df['ma30']) & (df['prev_ma20'] <= df['prev_ma30']), 'signal'] = 1
            df.loc[(df['ma20'] < df['ma30']) & (df['prev_ma20'] >= df['prev_ma30']), 'signal'] = -1
            
            data[symbol] = df
        return data
    
    def run(self, data, start_date, end_date):
        print(f"=== 趋势策略 v5 (集中交易) ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"交易品种: {', '.join(self.target_symbols)}")
        
        all_dates = set()
        for df in data.values():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            all_dates.update(df[mask]['trade_date'])
        dates = sorted(all_dates)
        
        for date in dates:
            for symbol, df in data.items():
                day_data = df[df['trade_date'] == date]
                if len(day_data) == 0:
                    continue
                
                row = day_data.iloc[0]
                price = row['close']
                signal = row['signal']
                atr = row['atr14']
                
                if signal != 0:
                    if symbol in self.positions:
                        pos = self.positions[symbol]
                        if pos['direction'] != signal:
                            self._close(symbol, date, price, '反向信号')
                        else:
                            # 盈利加仓
                            unrealized = (price - pos['entry_price']) * pos['direction'] / pos['entry_price']
                            if unrealized > 0.08 and pos['add_ons'] < 1:
                                self._add(symbol, date, price, atr)
                    else:
                        if len(self.positions) < self.max_positions and atr > 0:
                            self._open(symbol, signal, price, date, atr)
                
                if symbol in self.positions:
                    self._check_stop(symbol, date, price)
            
            unrealized = self._calc_unrealized(date, data)
            self.equity_curve.append((date, self.capital + unrealized))
        
        return self._get_results()
    
    def _open(self, symbol, direction, price, date, atr):
        notional = self.capital * self.position_pct
        size = notional / (price * 10)
        
        commission = price * size * 10 * 0.0001
        self.capital -= commission
        
        self.positions[symbol] = {
            'direction': direction,
            'size': size,
            'entry_price': price,
            'atr': atr,
            'stop': price - atr * 2.5 * direction,
            'add_ons': 0,
            'highest': price,
            'lowest': price
        }
        
        self.trades.append({'date': date, 'symbol': symbol, 'direction': direction,
                           'size': size, 'price': price, 'type': 'open'})
    
    def _add(self, symbol, date, price, atr):
        pos = self.positions[symbol]
        add_size = pos['size'] * 0.5
        
        commission = price * add_size * 10 * 0.0001
        self.capital -= commission
        
        pos['size'] += add_size
        pos['add_ons'] += 1
        pos['stop'] = price - atr * 2 * pos['direction']
        
        self.trades.append({'date': date, 'symbol': symbol, 'direction': pos['direction'],
                           'size': add_size, 'price': price, 'type': 'add'})
    
    def _close(self, symbol, date, price, reason):
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
        commission = price * pos['size'] * 10 * 0.0001
        
        self.capital += pnl - commission
        
        self.trades.append({'date': date, 'symbol': symbol, 'direction': -pos['direction'],
                           'size': pos['size'], 'price': price, 'pnl': pnl,
                           'type': 'close', 'reason': reason})
        del self.positions[symbol]
    
    def _check_stop(self, symbol, date, price):
        pos = self.positions[symbol]
        
        # 更新最高/最低价
        if pos['direction'] == 1:
            pos['highest'] = max(pos['highest'], price)
            # 移动止损: 从最高点回撤5%
            trailing_stop = pos['highest'] * 0.95
            if price < trailing_stop:
                self._close(symbol, date, price, '移动止损')
            elif price < pos['stop']:
                self._close(symbol, date, price, 'ATR止损')
        else:
            pos['lowest'] = min(pos['lowest'], price)
            trailing_stop = pos['lowest'] * 1.05
            if price > trailing_stop:
                self._close(symbol, date, price, '移动止损')
            elif price > pos['stop']:
                self._close(symbol, date, price, 'ATR止损')
    
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
        }
    
    def print_results(self, results):
        print("\n" + "="*50)
        print("趋势策略 v5 结果")
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
    
    bt = TrendBacktest(initial_capital=500000)
    
    print("加载数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    results = bt.run(data, start_date, end_date)
    bt.print_results(results)
    
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'backtest_trend.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
