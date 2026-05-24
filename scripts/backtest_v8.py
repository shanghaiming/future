#!/usr/bin/env python3
"""
期货激进趋势策略 v8
- 基于v6框架，关键改进:
  1. 降低开仓门槛(trend_score>=2)增加交易频率
  2. 提高仓位(基础25%,最大60%)
  3. 降低加仓阈值(3%)
  4. 优化移动止损(盈利2%启动2.5%回撤)
  5. 加入品种动量排名，只选最强
- 目标: 年化600%
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class BacktestV8:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.peak_capital = initial_capital
        
        # 参数 - 更激进
        self.max_positions = 12
        self.base_position_pct = 0.15
        self.max_position_pct = 0.45
        self.stop_loss_pct = 0.015
        self.trailing_stop_pct = 0.025
        self.add_on_threshold = 0.03
        self.pyramid_factor = 0.5
        
    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'): continue
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            df['return'] = df['close'].pct_change()
            
            for period in [5, 10, 20, 30, 60]:
                df[f'ma_{period}'] = df['close'].rolling(period).mean()
            
            df['ma_score'] = 0
            for short, long in [(5,10), (10,20), (20,30), (30,60)]:
                df.loc[df[f'ma_{short}'] > df[f'ma_{long}'], 'ma_score'] += 1
                df.loc[df[f'ma_{short}'] < df[f'ma_{long}'], 'ma_score'] -= 1
            
            for p in [10, 20, 60]:
                df[f'mom_{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p)
            
            df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
            df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
            
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr_14'] / df['close']
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))
            
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            df['bb_mid'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_mid'] + 2 * bb_std
            df['bb_lower'] = df['bb_mid'] - 2 * bb_std
            
            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']
            
            # 综合趋势强度
            df['trend_score'] = 0.0
            df.loc[df['ma_score'] >= 3, 'trend_score'] += 2
            df.loc[df['ma_score'] == 2, 'trend_score'] += 1
            df.loc[df['ma_score'] <= -3, 'trend_score'] -= 2
            df.loc[df['ma_score'] == -2, 'trend_score'] -= 1
            df.loc[df['mom_20'] > 0.05, 'trend_score'] += 1
            df.loc[df['mom_20'] < -0.05, 'trend_score'] -= 1
            df.loc[df['macd'] > df['macd_signal'], 'trend_score'] += 1
            df.loc[df['macd'] < df['macd_signal'], 'trend_score'] -= 1
            df.loc[df['rsi'] > 60, 'trend_score'] += 0.5
            df.loc[df['rsi'] < 40, 'trend_score'] -= 0.5
            
            # 60日动量排名用
            df['mom_60_rank'] = df['mom_60']
            
            data[symbol] = df
        return data
    
    def generate_signals(self, data, date):
        signals = []
        
        for symbol, df in data.items():
            day_data = df[df['trade_date'] <= date]
            if len(day_data) < 80:
                continue
            
            latest = day_data.iloc[-1]
            prev = day_data.iloc[-2] if len(day_data) > 1 else latest
            
            trend_score = latest['trend_score']
            hv = latest['hv_20']
            atr_pct = latest['atr_pct']
            vol_ratio = latest['vol_ratio']
            mom_60 = latest['mom_60']
            
            if hv < 0.12 or hv > 0.8:
                continue
            if atr_pct < 0.005:
                continue
            if vol_ratio < 0.7:
                continue
            
            # === 做多 ===
            if trend_score >= 2:
                if (latest['close'] > latest['ma_20'] and 
                    latest['mom_10'] > 0 and
                    latest['macd_hist'] > prev['macd_hist']):
                    
                    strength = min(trend_score / 5, 1.0)
                    # 60日动量强的加分
                    if mom_60 > 0.1:
                        strength += 0.3
                    signals.append({
                        'symbol': symbol,
                        'direction': 1,
                        'strength': min(strength, 1.0),
                        'trend_score': trend_score,
                        'mom_60': mom_60,
                        'reason': f'趋势做多 分数={trend_score:.1f}'
                    })
            
            # === 做空 ===
            elif trend_score <= -2:
                if (latest['close'] < latest['ma_20'] and 
                    latest['mom_10'] < 0 and
                    latest['macd_hist'] < prev['macd_hist']):
                    
                    strength = min(abs(trend_score) / 5, 1.0)
                    if mom_60 < -0.1:
                        strength += 0.3
                    signals.append({
                        'symbol': symbol,
                        'direction': -1,
                        'strength': min(strength, 1.0),
                        'trend_score': trend_score,
                        'mom_60': mom_60,
                        'reason': f'趋势做空 分数={trend_score:.1f}'
                    })
        
        # 排序: 先按趋势分数，再按60日动量
        signals.sort(key=lambda x: (abs(x['trend_score']), abs(x['mom_60'])), reverse=True)
        return signals[:self.max_positions]
    
    def calculate_position_size(self, signal, price, atr_pct):
        base_size = self.capital * self.base_position_pct / (price * 10)
        strength_multiplier = 1 + signal['strength']
        
        vol_multiplier = 1.0
        if atr_pct < 0.008:
            vol_multiplier = 1.5
        elif atr_pct > 0.03:
            vol_multiplier = 0.5
        
        size = base_size * strength_multiplier * vol_multiplier
        max_size = self.capital * self.max_position_pct / (price * 10)
        return min(size, max_size)
    
    def run(self, data, start_date, end_date):
        print(f"=== 激进趋势策略 v8 ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"目标: 年化600%")
        
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
                
                latest = day_data.iloc[0]
                price = latest['close']
                atr_pct = latest['atr_pct']
                
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    
                    if pos['direction'] != signal['direction']:
                        self._close(symbol, date, price, '反向信号')
                    elif pos['add_ons'] < 2:
                        unrealized = (price - pos['entry_price']) * pos['direction'] / pos['entry_price']
                        if unrealized > self.add_on_threshold:
                            self._add(symbol, date, price, atr_pct)
                else:
                    if len(self.positions) < self.max_positions:
                        size = self.calculate_position_size(signal, price, atr_pct)
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
            
            drawdown = (self.peak_capital - total_equity) / self.peak_capital
            if drawdown > 0.5:
                print(f"\n!!! 触发最大回撤50%，停止回测")
                print(f"日期: {date.date()}, 权益: {total_equity:,.0f}, 回撤: {drawdown:.2%}")
                break
        
        return self._get_results()
    
    def _open(self, symbol, direction, size, price, date, reason):
        commission = price * size * 10 * 0.0001
        self.capital -= commission
        
        self.positions[symbol] = {
            'direction': direction,
            'size': size,
            'entry_price': price,
            'highest_price': price,
            'lowest_price': price,
            'add_ons': 0,
            'total_size': size,
            'reason': reason
        }
        
        self.trades.append({
            'date': date, 'symbol': symbol, 'direction': direction,
            'size': size, 'price': price, 'type': 'open', 'reason': reason
        })
    
    def _add(self, symbol, date, price, atr_pct):
        pos = self.positions[symbol]
        add_size = pos['size'] * self.pyramid_factor ** pos['add_ons']
        
        total_notional = (pos['total_size'] + add_size) * price * 10
        if total_notional > self.capital * self.max_position_pct * 2:
            return
        
        commission = price * add_size * 10 * 0.0001
        self.capital -= commission
        
        pos['size'] += add_size
        pos['total_size'] += add_size
        pos['add_ons'] += 1
        
        self.trades.append({
            'date': date, 'symbol': symbol, 'direction': pos['direction'],
            'size': add_size, 'price': price, 'type': 'add'
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
        
        if pos['direction'] == 1:
            pos['highest_price'] = max(pos['highest_price'], price)
            
            hard_stop = pos['entry_price'] * (1 - self.stop_loss_pct)
            if price < hard_stop:
                self._close(symbol, date, price, '硬止损')
                return
            
            trailing_stop = pos['highest_price'] * (1 - self.trailing_stop_pct)
            if price < trailing_stop and price > pos['entry_price'] * 1.02:
                self._close(symbol, date, price, '移动止损')
                return
        
        else:
            pos['lowest_price'] = min(pos['lowest_price'], price)
            
            hard_stop = pos['entry_price'] * (1 + self.stop_loss_pct)
            if price > hard_stop:
                self._close(symbol, date, price, '硬止损')
                return
            
            trailing_stop = pos['lowest_price'] * (1 + self.trailing_stop_pct)
            if price > trailing_stop and price < pos['entry_price'] * 0.98:
                self._close(symbol, date, price, '移动止损')
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
        close_trades = trades_df[trades_df['type'] == 'close']
        open_trades = trades_df[trades_df['type'] == 'open']
        add_trades = trades_df[trades_df['type'] == 'add']
        
        wins = close_trades[close_trades['pnl'] > 0]
        losses = close_trades[close_trades['pnl'] <= 0]
        win_rate = len(wins) / len(close_trades) if len(close_trades) > 0 else 0
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1
        
        # 持仓时间统计
        hold_times = []
        entry_times = {}
        for _, t in trades_df.iterrows():
            if t['type'] == 'open':
                entry_times[t['symbol']] = t['date']
            elif t['type'] == 'close' and t['symbol'] in entry_times:
                hold_times.append((t['date'] - entry_times[t['symbol']]).days)
                del entry_times[t['symbol']]
        avg_hold = np.mean(hold_times) if hold_times else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity_curve[-1][1],
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(close_trades),
            'open_count': len(open_trades),
            'add_count': len(add_trades),
            'win_rate': win_rate,
            'profit_factor': avg_win / avg_loss if avg_loss > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_days': avg_hold,
            'max_hold_days': max(hold_times) if hold_times else 0,
            'min_hold_days': min(hold_times) if hold_times else 0,
        }
    
    def print_results(self, results):
        print("\n" + "="*60)
        print("激进趋势策略 v8 结果")
        print("="*60)
        print(f"初始资金:     {results['initial_capital']:>15,.0f}")
        print(f"最终权益:     {results['final_equity']:>15,.0f}")
        print(f"总收益率:     {results['total_return']:>15.2%}")
        print(f"年化收益率:   {results['annual_return']:>15.2%}")
        print(f"最大回撤:     {results['max_drawdown']:>15.2%}")
        print(f"夏普比率:     {results['sharpe_ratio']:>15.2f}")
        print(f"总交易次数:   {results['total_trades']:>15}")
        print(f"开仓次数:     {results.get('open_count', 0):>15}")
        print(f"加仓次数:     {results.get('add_count', 0):>15}")
        print(f"平均持仓:     {results.get('avg_hold_days', 0):>15.1f}天")
        print(f"最长持仓:     {results.get('max_hold_days', 0):>15}天")
        print(f"最短持仓:     {results.get('min_hold_days', 0):>15}天")
        print(f"胜率:         {results['win_rate']:>15.2%}")
        print(f"盈亏比:       {results['profit_factor']:>15.2f}")
        print(f"平均盈利:     {results['avg_win']:>15,.0f}")
        print(f"平均亏损:     {results['avg_loss']:>15,.0f}")
        print("="*60)
        
        if results['annual_return'] >= 6.0:
            print("✓ 达到目标: 年化600%+")
        elif results['annual_return'] >= 3.0:
            print("△ 接近目标: 年化300%+")
        elif results['annual_return'] >= 1.0:
            print("○ 有潜力: 年化100%+")
        else:
            print("✗ 需优化: 年化<100%")


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    
    bt = BacktestV8(initial_capital=500000)
    
    print("加载期货数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    results = bt.run(data, start_date, end_date)
    bt.print_results(results)
    
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'backtest_v8.json'), 'w') as f:
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
