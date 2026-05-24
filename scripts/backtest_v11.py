#!/usr/bin/env python3
"""
期货高胜率策略 v11
目标: 年化500%+ 胜率>50% 持仓<=3
策略: 极端偏离反转 + 期限结构过滤 + 波动率收敛
"""

import os, sys, json, numpy as np, pandas as pd
from datetime import datetime, timedelta

class BacktestV11:
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
        self.stop_loss_pct = 0.015
        self.trailing_stop_pct = 0.02
        
    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'): continue
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            df['return'] = df['close'].pct_change()
            
            for period in [5, 10, 20, 60]:
                df[f'ma_{period}'] = df['close'].rolling(period).mean()
            
            df['ma_dev'] = (df['close'] - df['ma_20']) / df['ma_20']
            df['ma_dev_z'] = (df['ma_dev'] - df['ma_dev'].rolling(60).mean()) / df['ma_dev'].rolling(60).std()
            
            for p in [5, 10, 20, 60]:
                df[f'mom_{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p)
            
            df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
            df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
            df['hv_ratio'] = df['hv_20'] / df['hv_60']
            
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
            
            df['bb_mid'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_mid'] + 2 * bb_std
            df['bb_lower'] = df['bb_mid'] - 2 * bb_std
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
            
            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']
            
            # 连续涨跌天数
            df['up_days'] = 0
            df['down_days'] = 0
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    df.loc[i, 'up_days'] = df['up_days'].iloc[i-1] + 1
                    df.loc[i, 'down_days'] = 0
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    df.loc[i, 'down_days'] = df['down_days'].iloc[i-1] + 1
                    df.loc[i, 'up_days'] = 0
            
            # 20日高低点位置
            df['hh_20'] = df['high'].rolling(20).max()
            df['ll_20'] = df['low'].rolling(20).min()
            df['range_pos'] = (df['close'] - df['ll_20']) / (df['hh_20'] - df['ll_20'])
            
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
            
            rsi = latest['rsi']
            bb_pct = latest['bb_pct']
            ma_dev = latest['ma_dev']
            ma_dev_z = latest['ma_dev_z']
            hv = latest['hv_20']
            atr_pct = latest['atr_pct']
            vol_ratio = latest['vol_ratio']
            range_pos = latest['range_pos']
            up_days = latest['up_days']
            down_days = latest['down_days']
            mom_5 = latest['mom_5']
            mom_20 = latest['mom_20']
            bb_width = latest['bb_width']
            
            # 基础过滤
            if hv < 0.1 or hv > 0.8:
                continue
            if atr_pct < 0.003:
                continue
            if bb_width < 0.02:
                continue
            
            # === 极端超卖做多 ===
            # 条件: RSI<25 + 价格跌破布林带下轨 + 偏离MA超2个标准差 + 连续下跌>=3天 + 波动率足够
            if (rsi < 25 and 
                bb_pct < 0.05 and 
                ma_dev_z < -2.0 and
                down_days >= 3 and
                range_pos < 0.1 and
                vol_ratio > 1.0 and
                mom_5 < -0.03):  # 近期急跌
                
                strength = min((30 - rsi) / 30, 1.0) * 0.5 + min(abs(ma_dev_z) / 3, 1.0) * 0.5
                strength = min(strength + 0.2, 1.0)
                
                signals.append({
                    'symbol': symbol,
                    'direction': 1,
                    'strength': strength,
                    'score': abs(ma_dev_z) + (30-rsi)/5,
                    'reason': f'极端超卖 RSI={rsi:.0f} 偏离={ma_dev_z:.1f}σ'
                })
            
            # === 极端超买做空 ===
            elif (rsi > 75 and 
                  bb_pct > 0.95 and 
                  ma_dev_z > 2.0 and
                  up_days >= 3 and
                  range_pos > 0.9 and
                  vol_ratio > 1.0 and
                  mom_5 > 0.03):
                
                strength = min((rsi - 70) / 30, 1.0) * 0.5 + min(ma_dev_z / 3, 1.0) * 0.5
                strength = min(strength + 0.2, 1.0)
                
                signals.append({
                    'symbol': symbol,
                    'direction': -1,
                    'strength': strength,
                    'score': ma_dev_z + (rsi-70)/5,
                    'reason': f'极端超买 RSI={rsi:.0f} 偏离={ma_dev_z:.1f}σ'
                })
        
        # 排序: 极端程度高的优先
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:self.max_positions]
    
    def calculate_position_size(self, signal, price, atr_pct):
        base_size = self.capital * self.base_position_pct / (price * 10)
        strength_multiplier = 1 + signal['strength']
        
        vol_multiplier = 1.0
        if atr_pct < 0.008:
            vol_multiplier = 1.3
        elif atr_pct > 0.04:
            vol_multiplier = 0.6
        
        size = base_size * strength_multiplier * vol_multiplier
        max_size = self.capital * self.max_position_pct / (price * 10)
        return min(size, max_size)
    
    def run(self, data, start_date, end_date):
        print(f"=== 高胜率策略 v11 ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"目标: 年化500% 胜率>50% 持仓<=3")
        
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
                    else:
                        # 同向不重复开仓
                        pass
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
                self._check_stops(symbol, date, price, data)
            
            unrealized = self._calc_unrealized(date, data)
            total_equity = self.capital + unrealized
            self.equity_curve.append((date, total_equity))
            
            if total_equity > self.peak_capital:
                self.peak_capital = total_equity
            
            drawdown = (self.peak_capital - total_equity) / self.peak_capital
            if drawdown > 0.5:
                print(f"\n!!! 触发最大回撤50%")
                break
        
        return self._get_results()
    
    def _open(self, symbol, direction, size, price, date, reason):
        commission = price * size * 10 * 0.0001
        self.capital -= commission
        self.positions[symbol] = {
            'direction': direction, 'size': size, 'entry_price': price,
            'highest_price': price, 'lowest_price': price,
            'reason': reason
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
    
    def _check_stops(self, symbol, date, price, data):
        pos = self.positions[symbol]
        
        if pos['direction'] == 1:
            pos['highest_price'] = max(pos['highest_price'], price)
            hard_stop = pos['entry_price'] * (1 - self.stop_loss_pct)
            if price < hard_stop:
                self._close(symbol, date, price, '硬止损')
                return
            trailing = pos['highest_price'] * (1 - self.trailing_stop_pct)
            if price < trailing and price > pos['entry_price'] * 1.015:
                self._close(symbol, date, price, '移动止损')
                return
            # 止盈: RSI回到中性或价格回到MA20
            df = data[symbol]
            day_data = df[df['trade_date'] == date]
            if len(day_data) > 0:
                rsi = day_data.iloc[0]['rsi']
                ma20 = day_data.iloc[0]['ma_20']
                if rsi > 55 and price > ma20:
                    self._close(symbol, date, price, 'RSI回归')
                    return
        else:
            pos['lowest_price'] = min(pos['lowest_price'], price)
            hard_stop = pos['entry_price'] * (1 + self.stop_loss_pct)
            if price > hard_stop:
                self._close(symbol, date, price, '硬止损')
                return
            trailing = pos['lowest_price'] * (1 + self.trailing_stop_pct)
            if price > trailing and price < pos['entry_price'] * 0.985:
                self._close(symbol, date, price, '移动止损')
                return
            df = data[symbol]
            day_data = df[df['trade_date'] == date]
            if len(day_data) > 0:
                rsi = day_data.iloc[0]['rsi']
                ma20 = day_data.iloc[0]['ma_20']
                if rsi < 45 and price < ma20:
                    self._close(symbol, date, price, 'RSI回归')
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
        print("高胜率策略 v11 结果")
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
    bt = BacktestV11(initial_capital=500000)
    print("加载期货数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    results = bt.run(data, start_date, end_date)
    bt.print_results(results)
    
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'backtest_v11.json'), 'w') as f:
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
