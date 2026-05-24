#!/usr/bin/env python3
"""
期货多策略回测引擎 v2
- 更严格的信号过滤
- 动态仓位管理
- 多时间框架确认
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Signal:
    symbol: str
    direction: int
    strength: float
    strategy: str
    reason: str

@dataclass 
class Trade:
    date: datetime
    symbol: str
    direction: int
    size: float
    price: float
    pnl: float
    commission: float
    strategy: str

class BacktestV2:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # 参数
        self.risk_per_trade = 0.02  # 单笔风险2%
        self.max_positions = 5  # 最多5个持仓
        self.stop_loss_atr = 2.0  # 2倍ATR止损
        self.take_profit_atr = 4.0  # 4倍ATR止盈
        
    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'): continue
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            df = self._calc_indicators(df)
            data[symbol] = df
        return data
    
    def _calc_indicators(self, df):
        close = df['close'].values
        
        # 收益率
        df['return'] = df['close'].pct_change()
        
        # 波动率
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
        
        # 均线
        for period in [5, 10, 20, 60]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
        
        # 均线斜率
        df['ma20_slope'] = df['ma_20'].diff(5) / df['ma_20'].shift(5) * 100
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
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
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # 动量
        df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        df['momentum_60'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
        
        # ADX (简化版)
        df['dm_plus'] = df['high'].diff()
        df['dm_minus'] = -df['low'].diff()
        df['dm_plus'] = df['dm_plus'].where(df['dm_plus'] > df['dm_minus'], 0)
        df['dm_minus'] = df['dm_minus'].where(df['dm_minus'] > df['dm_plus'], 0)
        
        return df
    
    def generate_signals(self, data, date):
        signals = []
        
        for symbol, df in data.items():
            day_data = df[df['trade_date'] <= date]
            if len(day_data) < 80:
                continue
            
            latest = day_data.iloc[-1]
            prev = day_data.iloc[-2]
            
            # === 严格过滤条件 ===
            
            # 1. 波动率过滤 - 要有足够波动
            if latest['hv_20'] < 0.2 or latest['hv_20'] > 0.8:
                continue
            
            # 2. 趋势强度过滤 - ADX简化
            trend_strength = abs(latest['momentum_20'])
            if trend_strength < 0.05:
                continue
            
            # 3. 成交量确认
            avg_vol = day_data['vol'].rolling(20).mean().iloc[-1]
            if latest['vol'] < avg_vol * 0.8:
                continue
            
            # === 策略1: 强趋势跟踪 ===
            # 条件: 多头排列 + MACD金叉 + 价格突破 + 动量向上
            if (latest['ma_5'] > latest['ma_10'] > latest['ma_20'] > latest['ma_60'] and
                latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal'] and
                latest['close'] > latest['bb_upper'] and
                latest['momentum_20'] > 0.05 and latest['momentum_60'] > 0.1 and
                latest['rsi'] > 50 and latest['rsi'] < 80):
                
                signals.append(Signal(symbol, 1, 0.95, '强趋势', '多头排列+放量突破'))
            
            # 空头同理
            elif (latest['ma_5'] < latest['ma_10'] < latest['ma_20'] < latest['ma_60'] and
                  latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal'] and
                  latest['close'] < latest['bb_lower'] and
                  latest['momentum_20'] < -0.05 and latest['momentum_60'] < -0.1 and
                  latest['rsi'] < 50 and latest['rsi'] > 20):
                
                signals.append(Signal(symbol, -1, 0.95, '强趋势', '空头排列+放量跌破'))
            
            # === 策略2: 波动率收缩突破 ===
            # 条件: HV低位 + 布林带收窄 + 放量突破
            elif latest['hv_20'] < latest['hv_60'] * 0.7:
                bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_mid']
                avg_bb_width = ((day_data['bb_upper'] - day_data['bb_lower']) / day_data['bb_mid']).rolling(20).mean().iloc[-1]
                
                if bb_width < avg_bb_width * 0.8 and latest['vol'] > avg_vol * 1.5:
                    if latest['close'] > latest['bb_upper'] and latest['momentum_20'] > 0:
                        signals.append(Signal(symbol, 1, 0.8, '波动率突破', '收缩后向上突破'))
                    elif latest['close'] < latest['bb_lower'] and latest['momentum_20'] < 0:
                        signals.append(Signal(symbol, -1, 0.8, '波动率突破', '收缩后向下突破'))
            
            # === 策略3: 均值回归 (极少使用) ===
            # 条件: 极端RSI + 背离 + 成交量萎缩
            elif latest['rsi'] < 10 and latest['vol'] < avg_vol * 0.6:
                signals.append(Signal(symbol, 1, 0.5, '均值回归', '极端超卖+缩量'))
            elif latest['rsi'] > 90 and latest['vol'] < avg_vol * 0.6:
                signals.append(Signal(symbol, -1, 0.5, '均值回归', '极端超买+缩量'))
        
        # 排序并限制数量
        signals.sort(key=lambda x: x.strength, reverse=True)
        return signals[:self.max_positions]
    
    def calculate_size(self, price, atr):
        risk_amount = self.capital * self.risk_per_trade
        stop_distance = atr * self.stop_loss_atr
        if stop_distance < price * 0.01:
            stop_distance = price * 0.01
        
        # 假设每手10吨
        size = risk_amount / (stop_distance * 10)
        
        # 限制单笔不超过20%资金
        max_size = (self.capital * 0.2) / (price * 10)
        size = min(size, max_size)
        
        return max(size, 0)
    
    def run(self, data, start_date, end_date):
        print(f"=== 回测 v2 ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"区间: {start_date.date()} ~ {end_date.date()}")
        
        all_dates = set()
        for df in data.values():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            all_dates.update(df[mask]['trade_date'])
        dates = sorted(all_dates)
        
        for date in dates:
            signals = self.generate_signals(data, date)
            
            for signal in signals:
                if signal.symbol not in data:
                    continue
                
                df = data[signal.symbol]
                day_data = df[df['trade_date'] <= date]
                if len(day_data) == 0:
                    continue
                
                latest = day_data.iloc[-1]
                price = latest['close']
                atr = latest['atr_14']
                
                # 执行信号
                self._execute(signal, date, price, atr)
            
            # 检查止损止盈
            self._check_exits(date, data)
            
            # 更新权益
            unrealized = self._calc_unrealized(date, data)
            self.equity_curve.append((date, self.capital + unrealized))
        
        return self._get_results()
    
    def _execute(self, signal, date, price, atr):
        symbol = signal.symbol
        
        # 平仓反向持仓
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos['direction'] != signal.direction:
                self._close(symbol, date, price, signal.strategy)
            else:
                return
        
        # 开新仓
        if len(self.positions) < self.max_positions:
            size = self.calculate_size(price, atr)
            if size > 0:
                commission = price * size * 10 * 0.0001
                self.capital -= commission
                
                self.positions[symbol] = {
                    'direction': signal.direction,
                    'size': size,
                    'entry_price': price,
                    'entry_date': date,
                    'atr': atr,
                    'strategy': signal.strategy
                }
                
                self.trades.append(Trade(date, symbol, signal.direction, size, price, 0, commission, signal.strategy))
    
    def _close(self, symbol, date, price, reason):
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pnl = (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
        commission = price * pos['size'] * 10 * 0.0001
        
        self.capital += pnl - commission
        
        self.trades.append(Trade(date, symbol, -pos['direction'], pos['size'], price, pnl, commission, reason))
        del self.positions[symbol]
    
    def _check_exits(self, date, data):
        for symbol, pos in list(self.positions.items()):
            if symbol not in data:
                continue
            
            df = data[symbol]
            day_data = df[df['trade_date'] <= date]
            if len(day_data) == 0:
                continue
            
            price = day_data.iloc[-1]['close']
            
            # ATR止损
            stop_dist = pos['atr'] * self.stop_loss_atr
            stop_price = pos['entry_price'] - stop_dist * pos['direction']
            
            if pos['direction'] == 1 and price < stop_price:
                self._close(symbol, date, price, 'ATR止损')
            elif pos['direction'] == -1 and price > stop_price:
                self._close(symbol, date, price, 'ATR止损')
            
            # ATR止盈
            profit_dist = pos['atr'] * self.take_profit_atr
            profit_price = pos['entry_price'] + profit_dist * pos['direction']
            
            if pos['direction'] == 1 and price > profit_price:
                self._close(symbol, date, price, 'ATR止盈')
            elif pos['direction'] == -1 and price < profit_price:
                self._close(symbol, date, price, 'ATR止盈')
    
    def _calc_unrealized(self, date, data):
        total = 0
        for symbol, pos in self.positions.items():
            if symbol not in data:
                continue
            df = data[symbol]
            day_data = df[df['trade_date'] <= date]
            if len(day_data) == 0:
                continue
            price = day_data.iloc[-1]['close']
            total += (price - pos['entry_price']) * pos['direction'] * pos['size'] * 10
        return total
    
    def _get_results(self):
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        equity_df['return'] = equity_df['equity'].pct_change()
        
        total_return = (self.equity_curve[-1][1] - self.initial_capital) / self.initial_capital
        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        years = days / 365
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        daily_returns = equity_df['return'].dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 1
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity_curve[-1][1],
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': avg_win / avg_loss if avg_loss > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'equity_curve': self.equity_curve
        }
    
    def print_results(self, results):
        print("\n" + "="*50)
        print("回测结果 v2")
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
    
    bt = BacktestV2(initial_capital=500000)
    
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
    with open(os.path.join(output_dir, 'backtest_v2.json'), 'w') as f:
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
