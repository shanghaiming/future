#!/usr/bin/env python3
"""
期货高胜率策略 v10
- 最大持仓3个品种
- 回测8年（2018-2026）
- 策略: 市场状态过滤 + 只做高概率趋势
- 目标: 胜率>50%，年化>100%
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class BacktestV10:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.peak_capital = initial_capital
        
        self.max_positions = 3
        self.base_position_pct = 0.3
        self.max_position_pct = 0.7
        self.stop_loss_pct = 0.012
        self.trailing_stop_pct = 0.018
        self.add_on_threshold = 0.03
        self.pyramid_factor = 0.4
        
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
            
            for p in [5, 10, 20, 60]:
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
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']
            
            # 市场状态指标
            df['adx'] = self._calc_adx(df, 14)
            df['trend_strength'] = abs(df['mom_20']) / (df['atr_pct'] * np.sqrt(20))
            
            data[symbol] = df
        return data
    
    def _calc_adx(self, df, period=14):
        """计算ADX趋势强度"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        return adx
    
    def get_market_state(self, data, date):
        """判断整体市场状态"""
        adx_values = []
        mom_values = []
        
        for symbol, df in data.items():
            day_data = df[df['trade_date'] <= date]
            if len(day_data) < 80:
                continue
            latest = day_data.iloc[-1]
            if not pd.isna(latest['adx']):
                adx_values.append(latest['adx'])
            if not pd.isna(latest['mom_20']):
                mom_values.append(latest['mom_20'])
        
        if not adx_values:
            return 'unknown'
        
        avg_adx = np.mean(adx_values)
        avg_mom = np.mean(mom_values) if mom_values else 0
        
        if avg_adx > 25:
            return 'trending'
        elif avg_adx < 15:
            return 'ranging'
        else:
            return 'mixed'
    
    def generate_signals(self, data, date):
        signals = []
        market_state = self.get_market_state(data, date)
        
        for symbol, df in data.items():
            day_data = df[df['trade_date'] <= date]
            if len(day_data) < 80:
                continue
            
            latest = day_data.iloc[-1]
            prev = day_data.iloc[-2] if len(day_data) > 1 else latest
            prev3 = day_data.iloc[-3] if len(day_data) > 2 else prev
            
            trend_score = latest['ma_score']
            hv = latest['hv_20']
            atr_pct = latest['atr_pct']
            vol_ratio = latest['vol_ratio']
            rsi = latest['rsi']
            bb_pct = latest['bb_pct']
            mom_5 = latest['mom_5']
            mom_20 = latest['mom_20']
            adx = latest['adx']
            
            # === 基础过滤 ===
            if hv < 0.12 or hv > 0.7:
                continue
            if atr_pct < 0.005:
                continue
            if vol_ratio < 0.7:
                continue
            
            # === 趋势市策略: 只做最强趋势 ===
            if market_state == 'trending':
                if trend_score >= 3 and mom_20 > 0.02 and adx > 25:
                    # 趋势确认后，等回调到均线买入
                    pullback = (latest['close'] < latest['ma_5'] and 
                               latest['close'] > latest['ma_20'] and
                               prev['close'] < prev3['close'])
                    
                    if pullback and 40 <= rsi <= 60:
                        signals.append({
                            'symbol': symbol,
                            'direction': 1,
                            'strength': 0.9,
                            'reason': f'趋势市回调 RSI={rsi:.0f} ADX={adx:.0f}'
                        })
            
            # === 震荡市策略: 只做极端超卖 ===
            elif market_state == 'ranging':
                if (bb_pct < 0.1 and rsi < 30 and 
                    latest['close'] >= prev['close'] * 0.98 and
                    vol_ratio > 1.0):
                    signals.append({
                        'symbol': symbol,
                        'direction': 1,
                        'strength': 0.7,
                        'reason': f'震荡市超卖 RSI={rsi:.0f}'
                    })
            
            # === 混合市: 不做 ===
        
        signals.sort(key=lambda x: x['strength'], reverse=True)
        return signals[:self.max_positions]
    
    def calculate_position_size(self, signal, price, atr_pct):
        base_size = self.capital * self.base_position_pct / (price * 10)
        strength_multiplier = 1 + signal['strength']
        
        vol_multiplier = 1.0
        if atr_pct < 0.01:
            vol_multiplier = 1.3
        elif atr_pct > 0.025:
            vol_multiplier = 0.7
        
        size = base_size * strength_multiplier * vol_multiplier
        max_size = self.capital * self.max_position_pct / (price * 10)
        return min(size, max_size)
    
    def run(self, data, start_date, end_date):
        print(f"=== 高胜率策略 v10 市场过滤 (3持仓/8年) ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"最大持仓: {self.max_positions}个品种")
        print(f"回测周期: {start_date.date()} ~ {end_date.date()}")
        
        all_dates = set()
        for df in data.values():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            all_dates.update(df[mask]['trade_date'])
        dates = sorted(all_dates)
        
        print(f"总交易日: {len(dates)}")
        
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
                    elif pos['add_ons'] < 1:
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
            if price < trailing_stop and price > pos['entry_price'] * 1.015:
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
        
        wins = close_trades[close_trades['pnl'] > 0]
        losses = close_trades[close_trades['pnl'] <= 0]
        win_rate = len(wins) / len(close_trades) if len(close_trades) > 0 else 0
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1
        
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
            'win_rate': win_rate,
            'profit_factor': avg_win / avg_loss if avg_loss > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_days': avg_hold,
        }
    
    def print_results(self, results):
        print("\n" + "="*60)
        print("高胜率策略 v10 结果")
        print("="*60)
        print(f"初始资金:     {results['initial_capital']:>15,.0f}")
        print(f"最终权益:     {results['final_equity']:>15,.0f}")
        print(f"总收益率:     {results['total_return']:>15.2%}")
        print(f"年化收益率:   {results['annual_return']:>15.2%}")
        print(f"最大回撤:     {results['max_drawdown']:>15.2%}")
        print(f"夏普比率:     {results['sharpe_ratio']:>15.2f}")
        print(f"总平仓次数:   {results['total_trades']:>15}")
        print(f"开仓次数:     {results.get('open_count', 0):>15}")
        print(f"平均持仓:     {results.get('avg_hold_days', 0):>15.1f}天")
        print(f"胜率:         {results['win_rate']:>15.2%}")
        print(f"盈亏比:       {results['profit_factor']:>15.2f}")
        print(f"平均盈利:     {results['avg_win']:>15,.0f}")
        print(f"平均亏损:     {results['avg_loss']:>15,.0f}")
        print("="*60)
        
        if results['win_rate'] >= 0.5 and results['annual_return'] >= 1.0:
            print("✓ 达标: 胜率>50%且年化>100%")
        elif results['win_rate'] >= 0.5:
            print("△ 胜率达标但年化不足")
        elif results['annual_return'] >= 1.0:
            print("○ 年化达标但胜率不足")
        else:
            print("✗ 需优化")


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    
    bt = BacktestV10(initial_capital=500000)
    
    print("加载期货数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*8)
    
    results = bt.run(data, start_date, end_date)
    bt.print_results(results)
    
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'backtest_v10.json'), 'w') as f:
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
