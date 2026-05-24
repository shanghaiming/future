#!/usr/bin/env python3
"""
期货策略 v24 — 纯动量排名 + 名义价值归一化
v23复盘:
  - 真实规格下不同品种杠杆差10x (铁矿vs螺纹), 导致爆仓
  - 多因子评分过拟合 (低分反而最好)
  - 需要按名义价值统一仓位
思路转变:
  - 不做复杂评分, 纯动量排名 (最简单最有效)
  - 按名义价值开仓 (所有品种等杠杆)
  - 固定10天持有, 不移动止损
"""

import os, sys, json, time, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec


class BacktestV24:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital

        self.max_positions = 3
        self.target_leverage = 10  # 总杠杆10x (每仓~3.3x)
        self.hold_days = 10
        self.hard_stop_pct = 0.05  # 5%价格止损

        self.commission_rate = 0.00015
        self.slippage_pct = 0.0001

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

            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']

            # 动量指标 (纯排名用)
            df['mom_5'] = df['close'].pct_change(5)
            df['mom_10'] = df['close'].pct_change(10)
            df['mom_20'] = df['close'].pct_change(20)

            # 趋势方向: MA20 vs MA60
            df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

            df = df.dropna(subset=['ma20', 'ma60', 'atr', 'mom_5', 'mom_10'])
            if len(df) > 100:
                data[symbol] = df
        return data

    def build_date_index(self, data, start_date, end_date):
        date_map = defaultdict(dict)
        for symbol, df in data.items():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            for _, row in df[mask].iterrows():
                date_map[row['trade_date']][symbol] = row
        return date_map

    def rank_momentum(self, day_data):
        """纯动量排名 — 返回 [(symbol, direction, momentum_score), ...]"""
        ranked = []
        for symbol, row in day_data.items():
            if symbol in self.positions:
                continue
            if pd.isna(row.get('mom_10', np.nan)):
                continue

            # 综合动量分 = 加权近期动量
            mom = (
                row.get('mom_5', 0) * 0.4 +
                row.get('mom_10', 0) * 0.4 +
                row.get('mom_20', 0) * 0.2
            )

            # 方向: 趋势方向 + 动量方向一致
            trend = row.get('trend', 0)
            direction = trend  # 1=多, -1=空

            # 只做动量和趋势一致的方向
            if direction == 1 and mom < 0:
                continue  # 趋势多但动量弱, 不做
            if direction == -1 and mom > 0:
                continue  # 趋势空但动量弱, 不做

            # 过滤: ATR太大不做 (避免高波动品种)
            if row.get('atr_pct', 0.1) > 0.045:
                continue

            # 分数 = |动量| * 趋势强度
            score = abs(mom) * 100

            ranked.append({
                'symbol': symbol,
                'direction': direction,
                'momentum': mom,
                'score': score,
                'atr': row['atr'],
                'price': row['close'],
            })

        # 按动量绝对值排序 (最强动量优先)
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked

    def _open_position(self, symbol, direction, price, atr, date, score, momentum):
        """开仓 — 按名义价值统一杠杆"""
        multiplier, margin_rate, _, _ = get_spec(symbol)
        margin_per_lot = price * multiplier * margin_rate
        if margin_per_lot <= 0:
            return

        # 目标: 每仓名义价值 = equity * (target_leverage / max_positions)
        target_notional = self.equity * (self.target_leverage / self.max_positions)
        lots = max(int(target_notional / (price * multiplier)), 1)

        # 检查总保证金
        total_margin = sum(p['margin'] for p in self.positions.values()) + margin_per_lot * lots
        max_margin = self.equity * 0.85
        if total_margin > max_margin:
            lots = max(int((max_margin - sum(p['margin'] for p in self.positions.values())) / margin_per_lot), 0)
            if lots <= 0:
                return

        actual_margin = margin_per_lot * lots
        commission = price * multiplier * lots * self.commission_rate
        total_cost = actual_margin + commission

        if total_cost > self.cash:
            lots = max(int((self.cash - commission) / margin_per_lot), 0)
            if lots <= 0:
                return
            actual_margin = margin_per_lot * lots
            commission = price * multiplier * lots * self.commission_rate
            total_cost = actual_margin + commission

        fill_price = price * (1 + self.slippage_pct * direction)
        self.cash -= total_cost

        self.positions[symbol] = {
            'direction': direction,
            'entry_price': fill_price,
            'entry_date': date,
            'lots': lots,
            'multiplier': multiplier,
            'margin': actual_margin,
            'notional': price * multiplier * lots,
            'commission_paid': commission,
            'stop_price': fill_price * (1 - direction * self.hard_stop_pct),
            'score': score,
            'momentum': momentum,
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

        price_change = (fill_price - pos['entry_price']) / pos['entry_price'] * pos['direction']
        notional_return = net_pnl / pos['notional'] if pos['notional'] > 0 else 0
        equity_return = net_pnl / self.equity if self.equity > 0 else 0

        self.closed_trades.append({
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'lots': pos['lots'],
            'notional': pos['notional'],
            'margin': pos['margin'],
            'pnl': net_pnl,
            'gross_pnl': pnl,
            'commission': pos['commission_paid'] + commission,
            'hold_days': (date - pos['entry_date']).days,
            'price_change': price_change,
            'notional_return': notional_return,
            'equity_return': equity_return,
            'reason': reason,
            'momentum': pos['momentum'],
        })
        del self.positions[symbol]

    def run(self, data, start_date, end_date):
        t0 = time.time()
        print(f"=== 期货策略 v24 (纯动量排名) ===")
        print(f"初始: {self.initial_capital:,.0f} | 杠杆: {self.target_leverage}x")
        print(f"持有: {self.hold_days}天 | 硬止损: {self.hard_stop_pct:.0%}")
        print(f"回测: {start_date.date()} ~ {end_date.date()} | 品种: {len(data)}")

        date_map = self.build_date_index(data, start_date, end_date)
        dates = sorted(date_map.keys())
        print(f"交易日: {len(dates)}")

        for date in dates:
            day_data = date_map[date]

            # 1. 退出
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if symbol not in day_data:
                    continue
                price = day_data[symbol]['close']
                hold_days = (date - pos['entry_date']).days

                if hold_days >= self.hold_days:
                    self._close_position(symbol, date, price, 'time_exit')
                    continue

                # 硬止损
                if pos['direction'] == 1 and price < pos['stop_price']:
                    self._close_position(symbol, date, price, 'hard_stop')
                    continue
                if pos['direction'] == -1 and price > pos['stop_price']:
                    self._close_position(symbol, date, price, 'hard_stop')
                    continue

            # 2. 动量排名入场
            if len(self.positions) < self.max_positions:
                ranked = self.rank_momentum(day_data)
                for candidate in ranked:
                    if len(self.positions) >= self.max_positions:
                        break
                    self._open_position(
                        candidate['symbol'], candidate['direction'],
                        candidate['price'], candidate['atr'],
                        date, candidate['score'], candidate['momentum'])

            # 3. 权益
            unrealized = 0
            for symbol, pos in self.positions.items():
                if symbol in day_data:
                    price = day_data[symbol]['close']
                    unrealized += (price - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']
            self.equity = self.cash + unrealized
            self.equity_curve.append((date, self.equity))

        print(f"\n回测完成, 耗时 {time.time()-t0:.1f}秒")
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

        # 方向统计
        long_trades = trades_df[trades_df['direction'] == 1]
        short_trades = trades_df[trades_df['direction'] == -1]
        direction_stats = {
            'long': {'count': len(long_trades), 'win_rate': (long_trades['pnl'] > 0).mean() if len(long_trades) > 0 else 0,
                     'avg_pnl': long_trades['pnl'].mean() if len(long_trades) > 0 else 0},
            'short': {'count': len(short_trades), 'win_rate': (short_trades['pnl'] > 0).mean() if len(short_trades) > 0 else 0,
                      'avg_pnl': short_trades['pnl'].mean() if len(short_trades) > 0 else 0},
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
            'avg_equity_return': trades_df['equity_return'].mean(),
            'reason_stats': reason_stats,
            'yearly_stats': yearly_stats,
            'direction_stats': direction_stats,
        }

    def print_results(self, results):
        print("\n" + "=" * 60)
        print("期货策略 v24 结果 (纯动量排名)")
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
        if results.get('avg_equity_return'):
            print(f"平均权益收益: {results['avg_equity_return']:>15.2%}")
        print("=" * 60)

        if results.get('direction_stats'):
            ds = results['direction_stats']
            print(f"\n方向统计:")
            for d, s in ds.items():
                print(f"  {d:6s}: {s['count']:3d}笔  WR={s['win_rate']:.1%}  avg={s['avg_pnl']:>10,.0f}")

        if results.get('reason_stats'):
            print(f"\n退出原因:")
            for reason, stats in results['reason_stats'].items():
                print(f"  {reason:15s}: {stats['count']:3d}笔  WR={stats['win_rate']:.1%}  avg={stats['avg_pnl']:>10,.0f}")

        if results.get('yearly_stats'):
            print(f"\n年度统计:")
            for year, stats in sorted(results['yearly_stats'].items()):
                print(f"  {year}: {stats['trades']:3d}笔  WR={stats['win_rate']:.1%}  PnL={stats['total_pnl']:>12,.0f}")

        print(f"\n--- 目标检查 ---")
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
            print(f"\n★★★ 全部目标达成 ★★★")


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")

    bt = BacktestV24(initial_capital=500000)
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

    with open(os.path.join(output_dir, 'backtest_v24.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()
