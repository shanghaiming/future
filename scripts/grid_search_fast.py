#!/usr/bin/env python3
"""
快速参数搜索 - 减少参数组合数
"""

import os, json, numpy as np, pandas as pd
from datetime import datetime, timedelta

def load_data(data_dir):
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
        
        data[symbol] = df
    return data

def run_backtest(data, params, start_date, end_date):
    capital = 500000
    positions = {}
    trades = []
    equity_curve = []
    peak_capital = capital
    
    max_positions = 3
    base_position_pct = params['base_pct']
    stop_loss_pct = params['stop']
    profit_target_pct = params['profit']
    max_hold_days = params['hold']
    rsi_long = params['rsi_long']
    rsi_short = params['rsi_short']
    vol_th = params['vol']
    
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
            
            if prev['rsi'] <= rsi_long and latest['rsi'] > rsi_long and latest['vol_ratio'] > vol_th:
                if latest['trend_score'] >= 2:
                    signals.append({'symbol': symbol, 'direction': 1, 'score': latest['rsi']})
            elif prev['rsi'] >= rsi_short and latest['rsi'] < rsi_short and latest['vol_ratio'] > vol_th:
                if latest['trend_score'] <= 2:
                    signals.append({'symbol': symbol, 'direction': -1, 'score': 100 - latest['rsi']})
        
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
                    capital += pnl - price * pos['size'] * 10 * 0.0001
                    trades.append({'pnl': pnl})
                    del positions[symbol]
            else:
                if len(positions) < max_positions:
                    size = capital * base_position_pct / (price * 10)
                    if size > 0:
                        capital -= price * size * 10 * 0.0001
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
            
            exit_pnl = None
            if pos['direction'] == 1:
                if price < pos['entry_price'] * (1 - stop_loss_pct):
                    exit_pnl = (price - pos['entry_price']) * pos['size'] * 10
                elif price > pos['entry_price'] * (1 + profit_target_pct):
                    exit_pnl = (price - pos['entry_price']) * pos['size'] * 10
                elif hold_days >= max_hold_days:
                    exit_pnl = (price - pos['entry_price']) * pos['size'] * 10
            else:
                if price > pos['entry_price'] * (1 + stop_loss_pct):
                    exit_pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                elif price < pos['entry_price'] * (1 - profit_target_pct):
                    exit_pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
                elif hold_days >= max_hold_days:
                    exit_pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
            
            if exit_pnl is not None:
                capital += exit_pnl - price * pos['size'] * 10 * 0.0001
                trades.append({'pnl': exit_pnl})
                del positions[symbol]
        
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
    
    if not equity_curve or len(equity_curve) < 10:
        return None
    
    total_return = (equity_curve[-1][1] - 500000) / 500000
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
    win_rate = len(wins) / len(trades_df)
    
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
    print("加载期货数据...")
    data = load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    # 减少参数组合
    param_list = []
    for base_pct in [0.30, 0.33]:
        for stop in [0.008, 0.010, 0.012]:
            for profit in [0.03, 0.05, 0.08]:
                for hold in [10, 15, 20]:
                    for rsi_long in [55, 58, 60]:
                        for rsi_short in [40, 42]:
                            for vol in [1.0, 1.2, 1.5]:
                                param_list.append({
                                    'base_pct': base_pct, 'stop': stop, 'profit': profit,
                                    'hold': hold, 'rsi_long': rsi_long, 'rsi_short': rsi_short, 'vol': vol
                                })
    
    print(f"总参数组合: {len(param_list)}")
    
    best = None
    best_score = -999
    
    for i, params in enumerate(param_list):
        result = run_backtest(data, params, start_date, end_date)
        if result is None:
            continue
        
        if result['win_rate'] >= 0.50 and result['annual_return'] >= 5.0:
            score = result['annual_return'] * result['win_rate'] / (1 + abs(result['max_drawdown']))
            if score > best_score:
                best_score = score
                best = (params, result)
                print(f"\n★ 找到满足条件的组合 #{i+1}:")
                print(f"  年化: {result['annual_return']:.1%}, 胜率: {result['win_rate']:.1%}, 回撤: {result['max_drawdown']:.1%}")
                print(f"  参数: {params}")
        
        if (i + 1) % 100 == 0:
            print(f"已测试 {i+1}/{len(param_list)}...")
    
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
        
        # 找最接近的
        closest = None
        closest_dist = 999
        for params in param_list:
            result = run_backtest(data, params, start_date, end_date)
            if result is None:
                continue
            dist = (5.0 - result['annual_return']) ** 2 + (0.5 - result['win_rate']) ** 2
            if dist < closest_dist:
                closest_dist = dist
                closest = (params, result)
        
        if closest:
            print(f"\n最接近的组合:")
            print(f"年化: {closest[1]['annual_return']:.1%}, 胜率: {closest[1]['win_rate']:.1%}")
            print(f"参数: {closest[0]}")

if __name__ == '__main__':
    main()
