#!/usr/bin/env python3
"""
期货高胜率趋势策略 v12
目标: 年化500%+ 胜率>50% 持仓<=3
策略: 趋势确认 + 回调入场 + 多因子共振 + 高盈亏比
"""

import os, sys, json, numpy as np, pandas as pd
from datetime import datetime, timedelta

class BacktestV12:
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
        
    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'): continue
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            df['return'] = df['close'].pct_change()
            
            # 多周期均线
            for p in [5, 10, 20, 30, 60, 120]:
                df[f'ma_{p}'] = df['close'].rolling(p).mean()
            
            # 趋势强度: 价格在多条均线上方/下方的比例
            df['trend_score'] = 0
            for p in [5, 10, 20, 30, 60]:
                df['trend_score'] += (df['close'] > df[f'ma_{p}']).astype(int)
            
            # 均线多头排列/空头排列
            df['ma_bull'] = (df['ma_5'] > df['ma_10']) & (df['ma_10'] > df['ma_20']) & (df['ma_20'] > df['ma_60'])
            df['ma_bear'] = (df['ma_5'] < df['ma_10']) & (df['ma_10'] < df['ma_20']) & (df['ma_20'] < df['ma_60'])
            
            # 动量
            for p in [5, 10, 20, 60]:
                df[f'mom_{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p)
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))
            
            # 布林带
            df['bb_mid'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_mid'] + 2 * bb_std
            df['bb_lower'] = df['bb_mid'] - 2 * bb_std
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
            
            # ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr_14'] / df['close']
            
            # 波动率
            df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
            df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
            
            # 成交量
            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']
            
            # 20日高低点
            df['hh_20'] = df['high'].rolling(20).max()
            df['ll_20'] = df['low'].rolling(20).min()
            df['range_pos'] = (df['close'] - df['ll_20']) / (df['hh_20'] - df['ll_20'])
            
            # 连续涨跌
            df['up_days'] = 0
            df['down_days'] = 0
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    df.loc[i, 'up_days'] = df['up_days'].iloc[i-1] + 1
                    df.loc[i, 'down_days'] = 0
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    df.loc[i, 'down_days'] = df['down_days'].iloc[i-1] + 1
                    df.loc[i, 'up_days'] = 0
            
            # ADX (简化版)
            df['dm_plus'] = df['high'].diff()
            df['dm_minus'] = -df['low'].diff()
            df['dm_plus'] = df['dm_plus'].where(df['dm_plus'] > df['dm_minus'], 0)
            df['dm_minus'] = df['dm_minus'].where(df['dm_minus'] > df['dm_plus'], 0)
            df['dm_plus'] = df['dm_plus'].where(df['dm_plus'] > 0, 0)
            df['dm_minus'] = df['dm_minus'].where(df['dm_minus'] > 0, 0)
            df['atr_14'] = df['tr'].rolling(14).mean()
            df['di_plus'] = 100 * df['dm_plus'].rolling(14).mean() / df['atr_14']
            df['di_minus'] = 100 * df['dm_minus'].rolling(14).mean() / df['atr_14']
            df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
            df['adx'] = df['dx'].rolling(14).mean()
            
            data[symbol] = df
        return data
    
    def generate_signals(self, data, date):
        signals = []
        
        for symbol, df in data.items():
            day_data = df[df['trade_date'] <= date]
            if len(day_data) < 120:
                continue
            
            latest = day_data.iloc[-1]
            prev = day_data.iloc[-2] if len(day_data) > 1 else latest
            prev2 = day_data.iloc[-3] if len(day_data) > 2 else prev
            
            # === 做多信号: 趋势确认后的回调 ===
            # 条件:
            # 1. 中长期趋势向上 (ma_bull or trend_score>=4)
            # 2. 短期回调 (连续下跌2-4天 or 价格触及布林带下轨)
            # 3. RSI在40-55之间 (不是极端超买也不是超卖)
            # 4. 成交量萎缩后放大 (洗盘结束)
            # 5. 波动率适中
            # 6. ADX > 25 (趋势明确)
            
            trend_up = latest['trend_score'] >= 4 or latest['ma_bull']
            trend_down = latest['trend_score'] <= 1 or latest['ma_bear']
            
            # 回调特征
            pullback_long = (
                latest['down_days'] >= 2 and 
                latest['down_days'] <= 5 and
                latest['close'] < latest['ma_5'] and
                latest['bb_pct'] < 0.4
            )
            
            pullback_short = (
                latest['up_days'] >= 2 and
                latest['up_days'] <= 5 and
                latest['close'] > latest['ma_5'] and
                latest['bb_pct'] > 0.6
            )
            
            # 成交量特征: 回调时缩量，当前开始放量
            vol_confirm_long = (
                prev['vol_ratio'] < 0.8 and  # 昨天缩量
                latest['vol_ratio'] > 1.0     # 今天放量
            )
            
            vol_confirm_short = (
                prev['vol_ratio'] < 0.8 and
                latest['vol_ratio'] > 1.0
            )
            
            # 波动率过滤
            hv_ok = 0.1 < latest['hv_20'] < 0.6
            atr_ok = 0.003 < latest['atr_pct'] < 0.05
            
            # ADX过滤
            adx_ok = latest['adx'] > 20 if pd.notna(latest['adx']) else False
            
            # 动量过滤: 中长期动量方向一致
            mom_ok_long = latest['mom_20'] > 0 and latest['mom_60'] > -0.05
            mom_ok_short = latest['mom_20'] < 0 and latest['mom_60'] < 0.05
            
            # RSI范围
            rsi_long = 35 < latest['rsi'] < 55
            rsi_short = 45 < latest['rsi'] < 75
            
            score = 0
            
            if trend_up and pullback_long and vol_confirm_long and hv_ok and atr_ok and adx_ok and mom_ok_long and rsi_long:
                score = (
                    latest['trend_score'] * 2 +
                    (5 - latest['down_days']) +
                    latest['vol_ratio'] * 2 +
                    (latest['adx'] - 20) / 10
                )
                signals.append({
                    'symbol': symbol,
                    'direction': 1,
                    'score': score,
                    'reason': f'趋势回调做多 趋势分={latest["trend_score"]} 回调{latest["down_days"]}天'
                })
            
            elif trend_down and pullback_short and vol_confirm_short and hv_ok and atr_ok and adx_ok and mom_ok_short and rsi_short:
                score = (
                    (5 - latest['trend_score']) * 2 +
                    (5 - latest['up_days']) +
                    latest['vol_ratio'] * 2 +
                    (latest['adx'] - 20) / 10
                )
                signals.append({
                    'symbol': symbol,
                    'direction': -1,
                    'score': score,
                    'reason': f'趋势回调做空 趋势分={latest["trend_score"]} 反弹{latest["up_days"]}天'
                })
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:self.max_positions]
    
    def calculate_position_size(self, signal, price, atr_pct):
        base_size = self.capital * self.base_position_pct / (price * 10)
        
        # 根据ATR调整仓位: ATR小则仓位大
        vol_multiplier = 1.0
        if atr_pct < 0.008:
            vol_multiplier = 1.5
        elif atr_pct < 0.015:
            vol_multiplier = 1.2
        elif atr_pct > 0.04:
            vol_multiplier = 0.6
        
        size = base_size * vol_multiplier
        max_size = self.capital * self.max_position_pct / (price * 10)
        return min(size, max_size)
    
    def run(self, data, start_date, end_date):
        print(f"=== 高胜率趋势策略 v12 ===")
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
                
                latest = day_data.iloc[0]
                price = latest['close']
                atr_pct = latest['atr_pct']
                
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    if pos['direction'] != signal['direction']:
                        self._close(symbol, date, price, '反向信号')
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
        atr_pct = data[symbol][data[symbol]['trade_date'] == date]['atr_pct'].iloc[0] if len(data[symbol][data[symbol]['trade_date'] == date]) > 0 else 0.01
        
        # ATR-based止损
        stop_mult = 2.0
        if pos['direction'] == 1:
            pos['highest_price'] = max(pos['highest_price'], price)
            hard_stop = pos['entry_price'] * (1 - atr_pct * stop_mult)
            if price < hard_stop:
                self._close(symbol, date, price, 'ATR止损')
                return
            # 移动止盈: 盈利3%后启动
            if price > pos['entry_price'] * 1.03:
                trailing = pos['highest_price'] * (1 - atr_pct * 1.5)
                if price < trailing:
                    self._close(symbol, date, price, '移动止盈')
                    return
            # 时间止盈: 持有10天且盈利
            # 目标止盈: 盈利5%
            if price > pos['entry_price'] * 1.05:
                self._close(symbol, date, price, '目标止盈5%')
                return
        else:
            pos['lowest_price'] = min(pos['lowest_price'], price)
            hard_stop = pos['entry_price'] * (1 + atr_pct * stop_mult)
            if price > hard_stop:
                self._close(symbol, date, price, 'ATR止损')
                return
            if price < pos['entry_price'] * 0.97:
                trailing = pos['lowest_price'] * (1 + atr_pct * 1.5)
                if price > trailing:
                    self._close(symbol, date, price, '移动止盈')
                    return
            if price < pos['entry_price'] * 0.95:
                self._close(symbol, date, price, '目标止盈5%')
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
        print("高胜率趋势策略 v12 结果")
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
    bt = BacktestV12(initial_capital=500000)
    print("加载期货数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    results = bt.run(data, start_date, end_date)
    bt.print_results(results)
    
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'backtest_v12.json'), 'w') as f:
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
