#!/usr/bin/env python3
"""
参数网格搜索 - 找满足年化500%+胜率>50%+持仓<=3的参数
"""

import os, json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from itertools import product

class GridSearch:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital
        
    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'): continue
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            df['return'] = df['close'].pct_change()
            
            for p in [5, 10, 20, 60]:
                df[f'ma_{p}'] = df['close'].rolling(p).mean()
            df['trend_score'] = sum((df['close'] > df[f'ma_{p}']).astype(int) for p in [5, 10, 20, 60])
            
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
    
    def run_backtest(self, data, params, start_date, end_date):
        capital = self.initial_capital
        positions = {}
        trades = []
        equity_curve = []
        peak_capital = capital
        
        max_positions = params['max_positions']
        base_position_pct = params['base_position_pct']
        max_position_pct = params['max_position_pct']
        stop_loss_pct = params['stop_loss_pct']
        profit_target_pct = params['profit_target_pct']
        max_hold_days = params['max_hold_days']
        rsi_long_entry = params['rsi_long_entry']
        rsi_short_entry = params['rsi_short_entry']
        vol_threshold = params['vol_threshold']
        use_trend_filter = params['use_trend_filter']
        
        all_dates = set()
        for df in data.values():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            all_dates.update(df[mask]['trade_date'])
        dates = sorted(all_dates)
        
        for date in dates:
            signals = []
            
            for symbol, df in data.items():
                day_data = df[df['trade_date'] <= date]
                if len(day_data) < 60:
                    continue
                
                latest = day_data.iloc[-1]
                prev = day_data.iloc[-2] if len(day_data) > 1 else latest
                
                if use_trend_filter and latest['trend_score'] < 2:
                    continue
                
                if prev['rsi'] <= rsi_long_entry and latest['rsi'] > rsi_long_entry and latest['vol_ratio'] > vol_threshold:
                    score = latest['rsi'] + latest['vol_ratio'] * 10
                    signals.append({'symbol': symbol, 'direction': 1, 'score': score})
                elif prev['rsi'] >= rsi_short_entry and latest['rsi'] < rsi_short_entry and latest['vol_ratio'] > vol_threshold:
                    score = (100 - latest['rsi']) + latest['vol_ratio'] * 10
                    signals.append({'symbol': symbol, 'direction': -1, 'score': score})
            
            signals.sort(key=lambda x: x['score'], reverse=True)
            signals = signals[:max_positions]
            
            for signal in signals:
                symbol = signal['symbol']
                if symbol not in data:
                    continue
                
                df = data[symbol]
                day_data = df[df['trade_date'] == date]
                if len(day_data) == 0:
                    continue
                
                price = day_data.iloc[0]['close']
                
                if symbol in positions:
                    pos = positions[symbol]
                    if pos['direction'] != signal['direction']:
                        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                        commission = price * pos['size'] * 10 * 0.0001
                        capital += pnl - commission
                        trades.append({'pnl': pnl})
                        del positions[symbol]
                else:
                    if len(positions) < max_positions:
                        size = capital * base_position_pct / (price * 10)
                        max_size = capital * max_position_pct / (price * 10)
                        size = min(size, max_size)
                        if size > 0:
                            commission = price * size * 10 * 0.0001
                            capital -= commission
                            positions[symbol] = {
                                'direction': signal['direction'], 'size': size, 
                                'entry_price': price, 'entry_date': date
                            }
            
            for symbol in list(positions.keys()):
                if symbol not in data:
                    continue
                df = data[symbol]
                day_data = df[df['trade_date'] == date]
                if len(day_data) == 0:
                    continue
                price = day_data.iloc[0]['close']
                pos = positions[symbol]
                hold_days = (date - pos['entry_date']).days
                
                if pos['direction'] == 1:
                    hard_stop = pos['entry_price'] * (1 - stop_loss_pct)
                    if price < hard_stop:
                        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                        commission = price * pos['size'] * 10 * 0.0001
                        capital += pnl - commission
                        trades.append({'pnl': pnl})
                        del positions[symbol]
                        continue
                    
                    if price > pos['entry_price'] * (1 + profit_target_pct):
                        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                        commission = price * pos['size'] * 10 * 0.0001
                        capital += pnl - commission
                        trades.append({'pnl': pnl})
                        del positions[symbol]
                        continue
                    
                    if hold_days >= max_hold_days:
                        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                        commission = price * pos['size'] * 10 * 0.0001
                        capital += pnl - commission
                        trades.append({'pnl': pnl})
                        del positions[symbol]
                        continue
                else:
                    hard_stop = pos['entry_price'] * (1 + stop_loss_pct)
                    if price > hard_stop:
                        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                        commission = price * pos['size'] * 10 * 0.0001
                        capital += pnl - commission
                        trades.append({'pnl': pnl})
                        del positions[symbol]
                        continue
                    
                    if price < pos['entry_price'] * (1 - profit_target_pct):
                        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                        commission = price * pos['size'] * 10 * 0.0001
                        capital += pnl - commission
                        trades.append({'pnl': pnl})
                        del positions[symbol]
                        continue
                    
                    if hold_days >= max_hold_days:
                        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                        commission = price * pos['size'] * 10 * 0.0001
                        capital += pnl - commission
                        trades.append({'pnl': pnl})
                        del positions[symbol]
                        continue
            
            unrealized = 0
            for symbol, pos in positions.items():
                if symbol not in data:
                    continue
                df = data[symbol]
                day_data = df[df['trade_date'] == date]
                if len(day_data) == 0:
                    continue
                price = day_data.iloc[0]['close']
                unrealized += (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
            
            total_equity = capital + unrealized
            equity_curve.append((date, total_equity))
            
            if total_equity > peak_capital:
                peak_capital = total_equity
        
        if not equity_curve:
            return None
        
        total_return = (equity_curve[-1][1] - self.initial_capital) / self.initial_capital
        days = (equity_curve[-1][0] - equity_curve[0][0]).days
        years = max(days / 365, 0.001)
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        equity_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        trades_df = pd.DataFrame(trades)
        if len(trades_df) == 0:
            return None
        
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1
        
        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': avg_win / avg_loss if avg_loss > 0 else 0,
            'final_equity': equity_curve[-1][1],
        }

def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    gs = GridSearch(initial_capital=500000)
    print("加载期货数据...")
    data = gs.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    # 参数网格
    param_grid = {
        'max_positions': [3],
        'base_position_pct': [0.30, 0.33],
        'max_position_pct': [0.40, 0.45],
        'stop_loss_pct': [0.008, 0.010, 0.012, 0.015],
        'profit_target_pct': [0.02, 0.03, 0.05, 0.08],
        'max_hold_days': [10, 15, 20],
        'rsi_long_entry': [55, 58, 60, 62],
        'rsi_short_entry': [38, 40, 42, 45],
        'vol_threshold': [1.0, 1.2, 1.5],
        'use_trend_filter': [True, False],
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    best = None
    best_score = -999
    total = 1
    for v in values:
        total *= len(v)
    
    print(f"总参数组合: {total}")
    count = 0
    
    for combo in product(*values):
        params = dict(zip(keys, combo))
        count += 1
        
        result = gs.run_backtest(data, params, start_date, end_date)
        if result is None:
            continue
        
        if result['win_rate'] >= 0.50 and result['annual_return'] >= 5.0:
            score = result['annual_return'] * result['win_rate'] / (1 + abs(result['max_drawdown']))
            if score > best_score:
                best_score = score
                best = (params, result)
                print(f"\n★ 找到满足条件的组合 #{count}:")
                print(f"  年化: {result['annual_return']:.1%}, 胜率: {result['win_rate']:.1%}, 回撤: {result['max_drawdown']:.1%}")
                print(f"  参数: {params}")
        
        if count % 500 == 0:
            print(f"已测试 {count}/{total}...")
    
    if best:
        print(f"\n{'='*60}")
        print("最佳参数组合:")
        print(f"{'='*60}")
        print(f"参数: {best[0]}")
        print(f"年化: {best[1]['annual_return']:.1%}")
        print(f"胜率: {best[1]['win_rate']:.1%}")
        print(f"回撤: {best[1]['max_drawdown']:.1%}")
        print(f"交易: {best[1]['total_trades']}")
        print(f"盈亏比: {best[1]['profit_factor']:.2f}")
        print(f"最终权益: {best[1]['final_equity']:,.0f}")
    else:
        print("\n未找到满足条件的参数组合")
        print("在测试的参数范围内，无法同时满足年化>=500%且胜率>=50%")

if __name__ == '__main__':
    main()
