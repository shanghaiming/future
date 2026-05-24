#!/usr/bin/env python3
"""
期货高胜率策略 v17
目标: 年化500%+ 胜率>50% 持仓<=3
策略: RSI突破 + 成交量确认 + 固定持有15天 + 止损止盈
"""

import os, json, numpy as np, pandas as pd
from datetime import datetime, timedelta

class BacktestV17:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.peak_capital = initial_capital
        
        self.max_positions = 3
        self.base_position_pct = 0.33
        self.max_position_pct = 0.45
        self.stop_loss_pct = 0.01
        self.profit_target_pct = 0.03
        self.max_hold_days = 15
        
    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'): continue
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            df['return'] = df['close'].pct_change()
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))
            
            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']
            
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr_14'] / df['close']
            
            data[symbol] = df
        return data
    
    def generate_signals(self, data, date):
        signals = []
        
        for symbol, df in data.items():
            day_data = df[df['trade_date'] <= date]
            if len(day_data) < 60:
                continue
            
            latest = day_data.iloc[-1]
            prev = day_data.iloc[-2] if len(day_data) > 1 else latest
            
            # 做多: RSI突破58 + VOL>1.0
            if prev['rsi'] <= 58 and latest['rsi'] > 58 and latest['vol_ratio'] > 1.0:
                score = latest['rsi'] + latest['vol_ratio'] * 10
                signals.append({
                    'symbol': symbol,
                    'direction': 1,
                    'score': score,
                    'reason': f'RSI突破做多 RSI={latest["rsi"]:.0f} VOL={latest["vol_ratio"]:.1f}'
                })
            
            # 做空: RSI跌破42 + VOL>1.0
            elif prev['rsi'] >= 42 and latest['rsi'] < 42 and latest['vol_ratio'] > 1.0:
                score = (100 - latest['rsi']) + latest['vol_ratio'] * 10
                signals.append({
                    'symbol': symbol,
                    'direction': -1,
                    'score': score,
                    'reason': f'RSI跌破做空 RSI={latest["rsi"]:.0f} VOL={latest["vol_ratio"]:.1f}'
                })
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:self.max_positions]
    
    def calculate_position_size(self, price):
        size = self.capital * self.base_position_pct / (price * 10)
        max_size = self.capital * self.max_position_pct / (price * 10)
        return min(size, max_size)
    
    def run(self, data, start_date, end_date):
        print(f"=== 高胜率策略 v17 ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        
        all_dates = set()
        for df in data.values():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            all_dates.update(df[mask]['trade_date'])
        dates = sorted(all_dates)
        
        for date in dates:
            signals = self.generate_signals(data, date)
            
            for signal in signals:
                symbol = signal['symbol']
                if symbol not in data:
                    continue
                
                df = data[symbol]
                day_data = df[df['trade_date'] == date]
                if len(day_data) == 0:
                    continue
                
                price = day_data.iloc[0]['close']
                
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    if pos['direction'] != signal['direction']:
                        self._close(symbol, date, price, '反向信号')
                else:
                    if len(self.positions) < self.max_positions:
                        size = self.calculate_position_size(price)
                        if size > 0:
                            self._open(symbol, signal['direction'], size, price, date, signal['reason'])
            
            for symbol in list(self.positions.keys()):
                if symbol not in data:
                    continue
                df = data[symbol]
                day_data = df[df['trade_date'] == date]
                if len(day_data) == 0:
                    continue
                price = day_data.iloc[0]['close']
                self._check_stops(symbol, date, price)
            
            unrealized = self._calc_unrealized(date, data)
            total_equity = self.capital + unrealized
            self.equity_curve.append((date, total_equity))
            
            if total_equity > self.peak_capital:
                self.peak_capital = total_equity
        
        return self._get_results()
    
    def _open(self, symbol, direction, size, price, date, reason):
        commission = price * size * 10 * 0.0001
        self.capital -= commission
        self.positions[symbol] = {
            'direction': direction, 'size': size, 'entry_price': price,
            'highest_price': price, 'lowest_price': price,
            'reason': reason, 'entry_date': date
        }
        self.trades.append({
            'date': date, 'symbol': symbol, 'direction': direction,
            'size': size, 'price': price, 'type': 'open', 'reason': reason
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
            'type': 'close', 'reason': reason
        })
        del self.positions[symbol]
    
    def _check_stops(self, symbol, date, price):
        pos = self.positions[symbol]
        hold_days = (date - pos['entry_date']).days
        
        if pos['direction'] == 1:
            pos['highest_price'] = max(pos['highest_price'], price)
            
            # 硬止损
            hard_stop = pos['entry_price'] * (1 - self.stop_loss_pct)
            if price < hard_stop:
                self._close(symbol, date, price, '硬止损')
                return
            
            # 目标止盈
            if price > pos['entry_price'] * (1 + self.profit_target_pct):
                self._close(symbol, date, price, '目标止盈')
                return
            
            # 最大持有期
            if hold_days >= self.max_hold_days:
                self._close(symbol, date, price, '持有期满')
                return
        else:
            pos['lowest_price'] = min(pos['lowest_price'], price)
            
            hard_stop = pos['entry_price'] * (1 + self.stop_loss_pct)
            if price > hard_stop:
                self._close(symbol, date, price, '硬止损')
                return
            
            if price < pos['entry_price'] * (1 - self.profit_target_pct):
                self._close(symbol, date, price, '目标止盈')
                return
            
            if hold_days >= self.max_hold_days:
                self._close(symbol, date, price, '持有期满')
                return
    
    def _calc_unrealized(self, date, data):
        total = 0
        for symbol, pos in self.positions.items():
            if symbol not in data:
                continue
            df = data[symbol]
            day_data = df[df['trade_date'] == date]
            if len(day_data) == 0:
                continue
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
        if len(trades_df) == 0:
            close_trades = pd.DataFrame(columns=['pnl'])
        else:
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
        }
    
    def print_results(self, results):
        print("\n" + "="*60)
        print("高胜率策略 v17 结果")
        print("="*60)
        print(f"初始资金:     {results['initial_capital']:>15,.0f}")
        print(f"最终权益:     {results['final_equity']:>15,.0f}")
        print(f"总收益率:     {results['total_return']:>15.2%}")
        print(f"年化收益率:   {results['annual_return']:>15.2%}")
        print(f"最大回撤:     {results['max_drawdown']:>15.2%}")
        print(f"夏普比率:     {results['sharpe_ratio']:>15.2f}")
        print(f"总交易次数:   {results['total_trades']:>15}")
        print(f"胜率:         {results['win_rate']:>15.2%}")
        print(f"盈亏比:       {results['profit_factor']:>15.2f}")
        print(f"平均盈利:     {results['avg_win']:>15,.0f}")
        print(f"平均亏损:     {results['avg_loss']:>15,.0f}")
        print("="*60)
        
        ok = True
        if results['annual_return'] < 5.0:
            print("✗ 年化不足500%")
            ok = False
        else:
            print("✓ 年化>=500%")
        
        if results['win_rate'] < 0.5:
            print("✗ 胜率不足50%")
            ok = False
        else:
            print("✓ 胜率>=50%")
        
        if ok:
            print("\n★★★ 全部目标达成 ★★★")

def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    bt = BacktestV17(initial_capital=500000)
    print("加载期货数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    results = bt.run(data, start_date, end_date)
    bt.print_results(results)
    
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'backtest_v17.json'), 'w') as f:
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
    print(f"\n结果已保存")

if __name__ == '__main__':
    main()
