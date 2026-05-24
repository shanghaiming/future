#!/usr/bin/env python3
"""
期货策略 v22 — 满仓杠杆 + 动量突破
核心思路: 期货的优势就是杠杆, 用保证金满仓操作
- 每仓用25%权益作保证金 → 实际杠杆约10x
- 入场只做最强信号 (趋势+突破+量能)
- 止损用2x ATR (快速截断亏损)
- 盈利仓用移动止损让利润奔跑
- 最大持仓3, 空余仓位等待最佳机会
"""

import os, sys, json, time, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec


class BacktestV22:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital

        # 仓位: 每仓用25%权益的保证金
        self.max_positions = 3
        self.margin_per_pos = 0.25   # 每仓25%保证金
        self.max_total_margin = 0.80  # 总保证金不超80%

        # 止损止盈
        self.stop_atr = 2.5          # 止损2.5x ATR (紧)
        self.trail_atr = 3.0         # 移动止损3x ATR (盈利后)
        self.breakeven_atr = 1.5     # 盈利1.5x ATR后移至保本
        self.max_hold_days = 20
        self.min_hold_days = 3

        # 入场
        self.min_score = 70

        # 费用
        self.commission_rate = 0.00015
        self.slippage_pct = 0.0001

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
            loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss_s))

            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']

            df['roc5'] = df['close'].pct_change(5) * 100
            df['roc10'] = df['close'].pct_change(10) * 100

            df['high_15'] = df['close'].rolling(15).max().shift(1)
            df['low_15'] = df['close'].rolling(15).min().shift(1)

            # ADX
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            atr_s = df['atr']
            plus_di = 100 * plus_dm.rolling(14).mean() / (atr_s + 0.001)
            minus_di = 100 * minus_dm.rolling(14).mean() / (atr_s + 0.001)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
            df['adx'] = dx.rolling(14).mean()

            # OI
            df['oi_ma5'] = df['oi'].rolling(5).mean()
            df['oi_change'] = df['oi'].pct_change(3)

            df = df.dropna(subset=['ma20', 'ma60', 'atr', 'adx', 'high_15', 'low_15'])
            if len(df) > 100:
                data[symbol] = df
        return data

    def score_signal(self, row):
        long_score = 0
        short_score = 0
        close = row['close']

        # 趋势 (25)
        if row['ma20'] > row['ma60'] and close > row['ma20']:
            long_score += 25
        if row['ma20'] < row['ma60'] and close < row['ma20']:
            short_score += 25

        # 突破 (20)
        if close > row['high_15']:
            long_score += 20
        if close < row['low_15']:
            short_score += 20

        # 动量 (15)
        if row['roc10'] > 2:
            long_score += 15
        if row['roc10'] < -2:
            short_score += 15

        # 成交量 (10)
        if row['vol_ratio'] > 1.2:
            long_score += 10
            short_score += 10

        # ADX (15)
        if row['adx'] > 25:
            long_score += 15
            short_score += 15

        # OI (10)
        if row.get('oi_change', 0) > 0.01:
            long_score += 10
            short_score += 10

        # 波动率 (5)
        if row['atr_pct'] < 0.04:
            long_score += 5
            short_score += 5

        return long_score, short_score

    def build_date_index(self, data, start_date, end_date):
        date_map = defaultdict(dict)
        for symbol, df in data.items():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            for _, row in df[mask].iterrows():
                date_map[row['trade_date']][symbol] = row
        return date_map

    def _open_position(self, symbol, direction, price, atr, date, score):
        """开仓 — 保证金满仓模式"""
        multiplier, margin_rate, _, _ = get_spec(symbol)
        margin_per_lot = price * multiplier * margin_rate
        if margin_per_lot <= 0:
            return

        # 用 margin_per_pos% 的权益作为保证金
        target_margin = self.equity * self.margin_per_pos

        # 计算手数
        lots = max(int(target_margin / margin_per_lot), 1)

        # 检查总保证金约束
        total_margin_after = sum(p['margin'] for p in self.positions.values()) + margin_per_lot * lots
        if total_margin_after > self.equity * self.max_total_margin:
            lots = max(int((self.equity * self.max_total_margin - sum(p['margin'] for p in self.positions.values())) / margin_per_lot), 0)
            if lots <= 0:
                return

        actual_margin = margin_per_lot * lots

        # 检查现金够不够
        commission = price * multiplier * lots * self.commission_rate
        total_cost = actual_margin + commission

        if total_cost > self.cash:
            lots = max(int((self.cash - commission) / margin_per_lot), 0)
            if lots <= 0:
                return
            actual_margin = margin_per_lot * lots
            commission = price * multiplier * lots * self.commission_rate
            total_cost = actual_margin + commission

        # 滑点
        fill_price = price * (1 + self.slippage_pct * direction)

        self.cash -= total_cost

        self.positions[symbol] = {
            'direction': direction,
            'entry_price': fill_price,
            'entry_date': date,
            'entry_atr': atr,
            'lots': lots,
            'multiplier': multiplier,
            'margin': actual_margin,
            'commission_paid': commission,
            'stop_price': fill_price - direction * self.stop_atr * atr,
            'highest': fill_price if direction == 1 else fill_price,
            'lowest': fill_price if direction == -1 else fill_price,
            'breakeven_set': False,
            'score': score,
        }

    def _close_position(self, symbol, date, price, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]

        fill_price = price * (1 - self.slippage_pct * pos['direction'])
        pnl = (fill_price - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']
        commission = fill_price * pos['multiplier'] * pos['lots'] * self.commission_rate
        net_pnl = pnl - commission

        self.cash += pos['margin'] + net_pnl

        leveraged_return = net_pnl / pos['margin'] if pos['margin'] > 0 else 0
        price_change = (fill_price - pos['entry_price']) / pos['entry_price'] * pos['direction']

        self.closed_trades.append({
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'lots': pos['lots'],
            'margin': pos['margin'],
            'pnl': net_pnl,
            'gross_pnl': pnl,
            'commission': pos['commission_paid'] + commission,
            'hold_days': (date - pos['entry_date']).days,
            'leveraged_return': leveraged_return,
            'price_change': price_change,
            'reason': reason,
            'score': pos['score'],
        })
        del self.positions[symbol]

    def run(self, data, start_date, end_date):
        t0 = time.time()
        print(f"=== 期货策略 v22 (满仓杠杆) ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"每仓保证金: {self.margin_per_pos:.0%} | 总上限: {self.max_total_margin:.0%}")
        print(f"止损: {self.stop_atr}x ATR | 移动: {self.trail_atr}x ATR")
        print(f"回测: {start_date.date()} ~ {end_date.date()}")
        print(f"品种: {len(data)}")

        date_map = self.build_date_index(data, start_date, end_date)
        dates = sorted(date_map.keys())
        print(f"交易日: {len(dates)}")

        for date in dates:
            day_data = date_map[date]

            # === 1. 更新持仓 ===
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if symbol not in day_data:
                    continue

                row = day_data[symbol]
                price = row['close']
                atr = row['atr']
                hold_days = (date - pos['entry_date']).days

                # 更新极值
                if pos['direction'] == 1:
                    pos['highest'] = max(pos['highest'], price)
                else:
                    pos['lowest'] = min(pos['lowest'], price)

                # 浮盈检查
                unrealized = (price - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']

                # 保本止损: 盈利超过1.5x ATR后, 移至保本
                if not pos['breakeven_set']:
                    if pos['direction'] == 1 and price > pos['entry_price'] + self.breakeven_atr * pos['entry_atr']:
                        pos['stop_price'] = pos['entry_price'] + pos['multiplier'] * 0.001  # 略高于成本
                        pos['breakeven_set'] = True
                    elif pos['direction'] == -1 and price < pos['entry_price'] - self.breakeven_atr * pos['entry_atr']:
                        pos['stop_price'] = pos['entry_price'] - pos['multiplier'] * 0.001
                        pos['breakeven_set'] = True

                # 移动止损 (盈利后)
                if pos['breakeven_set']:
                    if pos['direction'] == 1:
                        trail_stop = pos['highest'] - self.trail_atr * atr
                        pos['stop_price'] = max(pos['stop_price'], trail_stop)
                    else:
                        trail_stop = pos['lowest'] + self.trail_atr * atr
                        pos['stop_price'] = min(pos['stop_price'], trail_stop)

                # 止损触发
                if pos['direction'] == 1 and price < pos['stop_price']:
                    self._close_position(symbol, date, price, 'stop_loss')
                    continue
                if pos['direction'] == -1 and price > pos['stop_price']:
                    self._close_position(symbol, date, price, 'stop_loss')
                    continue

                # 最大持有期
                if hold_days >= self.max_hold_days:
                    self._close_position(symbol, date, price, 'time_exit')
                    continue

                # 趋势反转 (盈利仓)
                if hold_days >= self.min_hold_days and unrealized > 0:
                    if pos['direction'] == 1 and row['ma5'] < row['ma20'] and row['rsi'] < 35:
                        self._close_position(symbol, date, price, 'trend_reverse')
                        continue
                    if pos['direction'] == -1 and row['ma5'] > row['ma20'] and row['rsi'] > 65:
                        self._close_position(symbol, date, price, 'trend_reverse')
                        continue

            # === 2. 新信号 ===
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
                            'price': row['close'],
                        })

                signals.sort(key=lambda x: x['score'], reverse=True)

                for sig in signals:
                    if len(self.positions) >= self.max_positions:
                        break
                    self._open_position(
                        sig['symbol'], sig['direction'],
                        sig['price'], sig['atr'],
                        date, sig['score']
                    )

            # === 3. 权益 ===
            unrealized = 0
            for symbol, pos in self.positions.items():
                if symbol in day_data:
                    price = day_data[symbol]['close']
                    unrealized += (price - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']

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
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': abs(losses['pnl'].mean()) if len(losses) > 0 else 0,
            'avg_hold_days': trades_df['hold_days'].mean(),
            'avg_leveraged_return': trades_df['leveraged_return'].mean(),
            'avg_price_change': trades_df['price_change'].mean(),
            'reason_stats': reason_stats,
            'yearly_stats': yearly_stats,
        }

    def print_results(self, results):
        print("\n" + "=" * 60)
        print("期货策略 v22 结果 (满仓杠杆)")
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
        if results.get('avg_leveraged_return'):
            print(f"平均杠杆收益: {results['avg_leveraged_return']:>15.2%}")
        if results.get('avg_price_change'):
            print(f"平均价格变动: {results['avg_price_change']:>15.2%}")
        print("=" * 60)

        if results.get('reason_stats'):
            print("\n退出原因:")
            for reason, stats in results['reason_stats'].items():
                print(f"  {reason:18s}: {stats['count']:3d}笔  WR={stats['win_rate']:.1%}  avg={stats['avg_pnl']:>10,.0f}")

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

    bt = BacktestV22(initial_capital=500000)
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

    with open(os.path.join(output_dir, 'backtest_v22.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()
