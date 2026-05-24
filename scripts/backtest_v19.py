#!/usr/bin/env python3
"""
期货趋势跟踪策略 v19
基于v18复盘: 去掉移动止损(杀利润), 改固定持有期+宽止损+高杠杆
目标: 年化600%+ | 胜率>50% | 最大持仓3 | 回测8年+
"""

import os, json, time, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict


class BacktestV19:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital

        # 杠杆与仓位
        self.max_positions = 3
        self.leverage = 10           # 提高杠杆
        self.equity_per_pos = 0.30
        self.risk_per_trade = 0.05   # 单笔风险5%

        # 持仓管理 (核心改动)
        self.hold_days = 10          # 固定持有10天
        self.initial_stop_atr = 5.0  # 宽止损5x ATR (防止极端行情)
        self.no_trailing = True      # 不用移动止损

        # 入场参数
        self.breakout_period = 15
        self.min_volume_ratio = 1.2  # 降低量能要求
        self.min_score = 65          # 降低分数门槛

        # 费用
        self.commission_rate = 0.0002

        # 状态
        self.equity = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_curve = []

    def load_data(self, data_dir):
        data = {}
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.csv'):
                continue
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(data_dir, f))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            if len(df) < 100:
                continue

            df['return'] = df['close'].pct_change()
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma60'] = df['close'].rolling(60).mean()
            df['ma5'] = df['close'].rolling(5).mean()

            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']

            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))

            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']

            df['high_n'] = df['close'].rolling(self.breakout_period).max().shift(1)
            df['low_n'] = df['close'].rolling(self.breakout_period).min().shift(1)
            df['roc10'] = df['close'].pct_change(10) * 100
            df['roc5'] = df['close'].pct_change(5) * 100

            # 新增: ADX (趋势强度)
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            atr_smooth = df['atr']
            plus_di = 100 * plus_dm.rolling(14).mean() / atr_smooth
            minus_di = 100 * minus_dm.rolling(14).mean() / atr_smooth
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(14).mean()

            df = df.dropna(subset=['ma20', 'ma60', 'atr', 'high_n', 'low_n', 'adx'])
            if len(df) > 100:
                data[symbol] = df
        return data

    def score_signal(self, row):
        """多因子评分 v2"""
        long_score = 0
        short_score = 0
        close = row['close']

        # 趋势对齐 (25分)
        if row['ma20'] > row['ma60'] and close > row['ma20']:
            long_score += 25
        if row['ma20'] < row['ma60'] and close < row['ma20']:
            short_score += 25

        # 突破 (20分)
        if row['high_n'] > 0 and close > row['high_n']:
            long_score += 20
        if row['low_n'] > 0 and close < row['low_n']:
            short_score += 20

        # 成交量 (15分)
        if row['vol_ratio'] > self.min_volume_ratio:
            long_score += 15
            short_score += 15

        # 动量 (15分)
        if row['roc5'] > 2:
            long_score += 15
        if row['roc5'] < -2:
            short_score += 15

        # ADX趋势强度 (15分)
        if row['adx'] > 25:
            long_score += 15
            short_score += 15

        # 波动率适中 (10分)
        if row['atr_pct'] < 0.045:
            long_score += 10
            short_score += 10

        return long_score, short_score

    def build_date_index(self, data, start_date, end_date):
        date_map = defaultdict(dict)
        for symbol, df in data.items():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            for _, row in df[mask].iterrows():
                date_map[row['trade_date']][symbol] = row
        return date_map

    def _open_position(self, symbol, direction, price, atr, atr_pct, date, score):
        stop_distance_pct = self.initial_stop_atr * atr_pct
        if stop_distance_pct <= 0.003:
            stop_distance_pct = 0.003

        risk_amount = self.equity * self.risk_per_trade
        equity_alloc = risk_amount / (stop_distance_pct * self.leverage)
        equity_alloc = min(equity_alloc, self.equity * 0.35)
        equity_alloc = max(equity_alloc, self.equity * 0.15)

        if equity_alloc > self.cash:
            return

        commission = equity_alloc * self.commission_rate
        self.cash -= equity_alloc + commission

        self.positions[symbol] = {
            'direction': direction,
            'entry_price': price,
            'entry_date': date,
            'entry_atr': atr,
            'entry_atr_pct': atr_pct,
            'equity_alloc': equity_alloc,
            'stop_price': price - direction * self.initial_stop_atr * atr,
            'commission': commission,
            'score': score,
        }

    def _close_position(self, symbol, date, price, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]

        price_change = (price - pos['entry_price']) / pos['entry_price']
        directional_change = price_change * pos['direction']
        leveraged_return = directional_change * self.leverage
        pnl = pos['equity_alloc'] * leveraged_return
        commission = pos['equity_alloc'] * self.commission_rate
        net_pnl = pnl - commission

        self.cash += pos['equity_alloc'] + net_pnl
        hold_days = (date - pos['entry_date']).days

        self.closed_trades.append({
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'equity_alloc': pos['equity_alloc'],
            'price_change': price_change,
            'leveraged_return': leveraged_return,
            'pnl': net_pnl,
            'commission': pos['commission'] + commission,
            'hold_days': hold_days,
            'reason': reason,
            'score': pos['score'],
        })
        del self.positions[symbol]

    def run(self, data, start_date, end_date):
        t0 = time.time()
        print(f"=== 期货趋势策略 v19 (固定持有期) ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"杠杆: {self.leverage}x | 持有: {self.hold_days}天 | 止损: {self.initial_stop_atr}x ATR")
        print(f"回测: {start_date.date()} ~ {end_date.date()}")
        print(f"品种: {len(data)}")
        print()

        date_map = self.build_date_index(data, start_date, end_date)
        dates = sorted(date_map.keys())
        print(f"交易日: {len(dates)}")

        for date in dates:
            day_data = date_map[date]

            # 1. 检查退出
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if symbol not in day_data:
                    continue

                price = day_data[symbol]['close']
                hold_days = (date - pos['entry_date']).days

                # 固定持有期退出
                if hold_days >= self.hold_days:
                    self._close_position(symbol, date, price, 'time_exit')
                    continue

                # 宽止损 (只在极端行情触发)
                if pos['direction'] == 1 and price < pos['stop_price']:
                    self._close_position(symbol, date, price, 'hard_stop')
                elif pos['direction'] == -1 and price > pos['stop_price']:
                    self._close_position(symbol, date, price, 'hard_stop')

            # 2. 新信号
            if len(self.positions) < self.max_positions:
                signals = []
                for symbol, row in day_data.items():
                    if symbol in self.positions:
                        continue

                    long_score, short_score = self.score_signal(row)
                    direction = 0
                    score = 0
                    if long_score >= self.min_score and long_score > short_score:
                        direction = 1
                        score = long_score
                    elif short_score >= self.min_score:
                        direction = -1
                        score = short_score

                    if direction != 0:
                        signals.append({
                            'symbol': symbol,
                            'direction': direction,
                            'score': score,
                            'atr': row['atr'],
                            'atr_pct': row['atr_pct'],
                            'price': row['close'],
                        })

                signals.sort(key=lambda x: x['score'], reverse=True)

                for sig in signals:
                    if len(self.positions) >= self.max_positions:
                        break
                    self._open_position(
                        sig['symbol'], sig['direction'],
                        sig['price'], sig['atr'], sig['atr_pct'],
                        date, sig['score']
                    )

            # 3. 权益
            unrealized = 0
            for symbol, pos in self.positions.items():
                if symbol in day_data:
                    price = day_data[symbol]['close']
                    price_change = (price - pos['entry_price']) / pos['entry_price']
                    unrealized += pos['equity_alloc'] * price_change * pos['direction'] * self.leverage

            self.equity = self.cash + unrealized
            self.equity_curve.append((date, self.equity))

        elapsed = time.time() - t0
        print(f"\n回测完成, 耗时 {elapsed:.1f}秒")
        return self._get_results()

    def _get_results(self):
        if not self.equity_curve:
            return {}

        final_equity = self.equity_curve[-1][1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        years = max(days / 365, 0.001)
        annual_return = (1 + total_return) ** (1 / years) - 1

        eq = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        eq['cummax'] = eq['equity'].cummax()
        eq['drawdown'] = (eq['equity'] - eq['cummax']) / eq['cummax']
        max_drawdown = eq['drawdown'].min()

        eq['return'] = eq['equity'].pct_change()
        daily_ret = eq['return'].dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

        trades_df = pd.DataFrame(self.closed_trades)
        if len(trades_df) == 0:
            return {'initial_capital': self.initial_capital, 'final_equity': final_equity,
                    'total_return': total_return, 'annual_return': annual_return,
                    'max_drawdown': max_drawdown, 'sharpe_ratio': sharpe,
                    'total_trades': 0, 'win_rate': 0, 'profit_factor': 0}

        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(wins) / len(trades_df)

        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1

        total_win = wins['pnl'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        reason_stats = trades_df.groupby('reason').agg(
            count=('pnl', 'count'),
            win_rate=('pnl', lambda x: (x > 0).mean()),
            avg_pnl=('pnl', 'mean'),
        ).to_dict('index')

        trades_df['year'] = trades_df['exit_date'].apply(
            lambda x: x.year if hasattr(x, 'year') else pd.Timestamp(x).year)
        yearly_stats = {}
        for year, group in trades_df.groupby('year'):
            yr_wins = group[group['pnl'] > 0]
            yearly_stats[int(year)] = {
                'trades': len(group),
                'win_rate': len(yr_wins) / len(group),
                'total_pnl': group['pnl'].sum(),
            }

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_days': trades_df['hold_days'].mean(),
            'reason_stats': reason_stats,
            'yearly_stats': yearly_stats,
        }

    def print_results(self, results):
        print("\n" + "=" * 60)
        print("期货趋势策略 v19 结果")
        print("=" * 60)
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
        if results.get('avg_hold_days'):
            print(f"平均持仓:     {results['avg_hold_days']:>15.1f}天")
        print("=" * 60)

        if results.get('reason_stats'):
            print("\n退出原因:")
            for reason, stats in results['reason_stats'].items():
                print(f"  {reason:15s}: {stats['count']:3d}笔  胜率={stats['win_rate']:.1%}  avg={stats['avg_pnl']:>10,.0f}")

        if results.get('yearly_stats'):
            print("\n年度统计:")
            for year, stats in sorted(results['yearly_stats'].items()):
                print(f"  {year}: {stats['trades']:3d}笔  WR={stats['win_rate']:.1%}  PnL={stats['total_pnl']:>12,.0f}")

        print("\n--- 目标检查 ---")
        ok = True
        if results['annual_return'] >= 6.0:
            print(f"✓ 年化 >= 600%: {results['annual_return']:.1%}")
        else:
            print(f"✗ 年化 < 600%: {results['annual_return']:.1%}")
            ok = False
        if results['win_rate'] >= 0.50:
            print(f"✓ 胜率 >= 50%: {results['win_rate']:.1%}")
        else:
            print(f"✗ 胜率 < 50%: {results['win_rate']:.1%}")
            ok = False
        print(f"  持仓上限: {self.max_positions} ✓")
        if ok:
            print("\n★★★ 全部目标达成 ★★★")


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")

    bt = BacktestV19(initial_capital=500000)
    print("加载期货数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")

    end_date = pd.Timestamp('2026-05-08')
    start_date = pd.Timestamp('2018-01-01')

    results = bt.run(data, start_date, end_date)
    bt.print_results(results)

    os.makedirs(output_dir, exist_ok=True)
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            save_results[k] = v

    with open(os.path.join(output_dir, 'backtest_v19.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()
