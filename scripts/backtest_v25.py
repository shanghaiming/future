#!/usr/bin/env python3
"""
期货策略 v25 — 动量策略最终版
基于全面参数扫描和策略对比:
  - 纯动量排名 >> 其他所有策略
  - 4-5x杠杆 + 5天持有 = 最优
  - 无止损 >> 有止损 (反复验证)
  - mom>0.03 提升到56.8%年化
改进:
  - 市场状态过滤 (避免高波动期)
  - 详细年度/月度报告
"""

import os, sys, json, time, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec


class BacktestV25:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital

        # 最优参数 (来自参数扫描)
        self.max_positions = 3
        self.leverage = 5
        self.hold_days = 5
        self.min_momentum = 0.03     # 最小动量门槛

        # 市场状态过滤
        self.market_adx_threshold = 20  # 平均ADX低于此值, 降低仓位
        self.max_atr_pct = 0.045

        # 动态杠杆
        self.base_leverage = 5
        self.bull_leverage = 7     # 好年景杠杆
        self.bear_leverage = 3     # 差年景杠杆

        self.commission_rate = 0.00015
        self.slippage_pct = 0.0001

        self.equity = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_curve = []
        self.monthly_returns = defaultdict(list)

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

            df['mom_5'] = df['close'].pct_change(5)
            df['mom_10'] = df['close'].pct_change(10)
            df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            atr_s = df['atr']
            plus_di = 100 * plus_dm.rolling(14).mean() / (atr_s + 0.001)
            minus_di = 100 * minus_dm.rolling(14).mean() / (atr_s + 0.001)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
            df['adx'] = dx.rolling(14).mean()

            df = df.dropna(subset=['ma20', 'ma60', 'atr', 'mom_10', 'adx'])
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

    def get_market_regime(self, day_data):
        """判断市场状态: 平均ADX和动量广度"""
        adx_values = [row['adx'] for row in day_data.values()
                      if not pd.isna(row.get('adx', np.nan))]
        if not adx_values:
            return 'neutral', self.base_leverage

        avg_adx = np.mean(adx_values)

        # 动量广度: 上涨品种占比
        trends = [row.get('trend', 0) for row in day_data.values()]
        breadth = sum(1 for t in trends if t == 1) / len(trends) if trends else 0.5

        if avg_adx > 28 and (breadth > 0.6 or breadth < 0.4):
            return 'trending', self.bull_leverage
        elif avg_adx < 18:
            return 'choppy', self.bear_leverage
        else:
            return 'neutral', self.base_leverage

    def _open_position(self, symbol, direction, price, date, leverage):
        mult, mr, _, _ = get_spec(symbol)
        mpl = price * mult * mr
        if mpl <= 0:
            return

        target_notional = self.equity * (leverage / self.max_positions)
        lots = max(int(target_notional / (price * mult)), 1)

        total_m = sum(p['margin'] for p in self.positions.values()) + mpl * lots
        if total_m > self.equity * 0.85:
            lots = max(int((self.equity * 0.85 - sum(p['margin'] for p in self.positions.values())) / mpl), 0)
            if lots <= 0:
                return

        actual_margin = mpl * lots
        comm = price * mult * lots * self.commission_rate
        if actual_margin + comm > self.cash:
            lots = max(int((self.cash - comm) / mpl), 0)
            if lots <= 0:
                return
            actual_margin = mpl * lots
            comm = price * mult * lots * self.commission_rate

        fill_price = price * (1 + self.slippage_pct * direction)
        self.cash -= actual_margin + comm

        self.positions[symbol] = {
            'direction': direction,
            'entry_price': fill_price,
            'entry_date': date,
            'lots': lots,
            'multiplier': mult,
            'margin': actual_margin,
            'leverage': leverage,
        }

    def _close_position(self, symbol, date, price, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        fill_price = price * (1 - self.slippage_pct * pos['direction'])
        pnl = (fill_price - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']
        comm = fill_price * pos['multiplier'] * pos['lots'] * self.commission_rate
        net_pnl = pnl - comm

        self.cash += pos['margin'] + net_pnl

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
            'hold_days': (date - pos['entry_date']).days,
            'price_change': price_change,
            'reason': reason,
            'leverage': pos['leverage'],
        })
        del self.positions[symbol]

    def run(self, data, start_date, end_date):
        t0 = time.time()
        print(f"=== 期货策略 v25 (动量最终版) ===")
        print(f"初始: {self.initial_capital:,.0f}")
        print(f"杠杆: {self.bear_leverage}-{self.bull_leverage}x (动态) | 基准: {self.base_leverage}x")
        print(f"持有: {self.hold_days}天 | 动量门槛: {self.min_momentum}")
        print(f"回测: {start_date.date()} ~ {end_date.date()} | 品种: {len(data)}")

        date_map = self.build_date_index(data, start_date, end_date)
        dates = sorted(date_map.keys())
        print(f"交易日: {len(dates)}")

        prev_equity = self.equity
        regime_counts = defaultdict(int)

        for date in dates:
            day_data = date_map[date]

            # 市场状态
            regime, current_leverage = self.get_market_regime(day_data)
            regime_counts[regime] += 1

            # 退出
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if symbol not in day_data:
                    continue
                price = day_data[symbol]['close']
                hd = (date - pos['entry_date']).days

                if hd >= self.hold_days:
                    self._close_position(symbol, date, price, 'time_exit')

            # 入场
            if len(self.positions) < self.max_positions:
                signals = []
                for symbol, row in day_data.items():
                    if symbol in self.positions:
                        continue
                    if row.get('atr_pct', 0.1) > self.max_atr_pct:
                        continue

                    mom = row.get('mom_10', 0)
                    trend = row.get('trend', 0)

                    if trend == 1 and mom > self.min_momentum:
                        signals.append((symbol, 1, abs(mom)))
                    elif trend == -1 and mom < -self.min_momentum:
                        signals.append((symbol, -1, abs(mom)))

                signals.sort(key=lambda x: x[2], reverse=True)

                for symbol, direction, _ in signals:
                    if len(self.positions) >= self.max_positions:
                        break
                    self._open_position(symbol, direction, day_data[symbol]['close'],
                                       date, current_leverage)

            # 权益
            unrealized = 0
            for symbol, pos in self.positions.items():
                if symbol in day_data:
                    unrealized += (day_data[symbol]['close'] - pos['entry_price']) * \
                                  pos['direction'] * pos['multiplier'] * pos['lots']
            self.equity = self.cash + unrealized
            self.equity_curve.append((date, self.equity))

            # 月度收益
            month_key = f"{date.year}-{date.month:02d}"
            daily_ret = (self.equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.monthly_returns[month_key].append(daily_ret)
            prev_equity = self.equity

        print(f"\n回测完成, 耗时 {time.time()-t0:.1f}秒")
        print(f"市场状态: trending={regime_counts['trending']} "
              f"neutral={regime_counts['neutral']} choppy={regime_counts['choppy']}")
        return self._get_results()

    def _get_results(self):
        if not self.equity_curve:
            return {}

        final = self.equity_curve[-1][1]
        total_ret = (final - self.initial_capital) / self.initial_capital
        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        years = max(days / 365, 0.001)
        ann = float((1 + total_ret) ** (1 / years) - 1)

        eq = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        eq['cummax'] = eq['equity'].cummax()
        eq['dd'] = (eq['equity'] - eq['cummax']) / eq['cummax']
        mdd = float(eq['dd'].min())

        eq['return'] = eq['equity'].pct_change()
        daily_ret = eq['return'].dropna()
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0

        trades_df = pd.DataFrame(self.closed_trades)
        if len(trades_df) == 0:
            return {'initial_capital': self.initial_capital, 'final_equity': final,
                    'annual_return': ann, 'max_drawdown': mdd}

        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(wins) / len(trades_df)

        total_win = wins['pnl'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        # 年度
        trades_df['year'] = trades_df['exit_date'].apply(
            lambda x: x.year if hasattr(x, 'year') else pd.Timestamp(x).year)
        yearly = {}
        for yr, grp in trades_df.groupby('year'):
            yr_wins = grp[grp['pnl'] > 0]
            yearly[int(yr)] = {
                'trades': len(grp),
                'win_rate': float(len(yr_wins) / len(grp)),
                'pnl': float(grp['pnl'].sum()),
            }

        # 月度
        monthly_stats = {}
        for mkey, rets in sorted(self.monthly_returns.items()):
            monthly_stats[mkey] = {
                'return': float((np.array(rets) + 1).prod() - 1),
                'days': len(rets),
            }

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final,
            'total_return': float(total_ret),
            'annual_return': ann,
            'max_drawdown': mdd,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'total_trades': len(trades_df),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_win': float(wins['pnl'].mean()) if len(wins) > 0 else 0,
            'avg_loss': float(abs(losses['pnl'].mean())) if len(losses) > 0 else 0,
            'avg_hold_days': float(trades_df['hold_days'].mean()),
            'avg_price_change': float(trades_df['price_change'].mean()),
            'yearly': yearly,
            'monthly': monthly_stats,
        }

    def print_results(self, r):
        print("\n" + "=" * 60)
        print("期货策略 v25 最终版 结果")
        print("=" * 60)
        print(f"初始资金:     {r['initial_capital']:>15,.0f}")
        print(f"最终权益:     {r['final_equity']:>15,.0f}")
        print(f"总收益率:     {r['total_return']:>15.2%}")
        print(f"年化收益率:   {r['annual_return']:>15.2%}")
        print(f"最大回撤:     {r['max_drawdown']:>15.2%}")
        print(f"夏普比率:     {r['sharpe_ratio']:>15.2f}")
        print(f"卡尔玛比率:   {r['calmar_ratio']:>15.2f}")
        print(f"总交易次数:   {r['total_trades']:>15}")
        print(f"胜率:         {r['win_rate']:>15.2%}")
        print(f"盈亏比:       {r['profit_factor']:>15.2f}")
        print(f"平均盈利:     {r['avg_win']:>15,.0f}")
        print(f"平均亏损:     {r['avg_loss']:>15,.0f}")
        print(f"平均持仓:     {r['avg_hold_days']:>15.1f}天")
        print(f"平均价格变动: {r['avg_price_change']:>15.2%}")
        print("=" * 60)

        if r.get('yearly'):
            print("\n年度统计:")
            print(f"  {'年份':>4} {'交易':>4} {'胜率':>6} {'PnL':>14}")
            for yr, s in sorted(r['yearly'].items()):
                print(f"  {yr:>4} {s['trades']:>4} {s['win_rate']:>6.1%} {s['pnl']:>14,.0f}")

        if r.get('monthly'):
            print("\n月度收益 (>10%或<-10%的月份):")
            for mkey, s in sorted(r['monthly'].items()):
                ret = s['return']
                if abs(ret) > 0.10:
                    print(f"  {mkey}: {ret:>+8.1%}")

        print(f"\n--- 目标检查 ---")
        ok = True
        if r['annual_return'] >= 6.0:
            print(f"✓ 年化 >= 600%: {r['annual_return']:.1%}")
        else:
            gap = 6.0 / r['annual_return'] if r['annual_return'] > 0 else float('inf')
            print(f"✗ 年化 < 600%: {r['annual_return']:.1%} (差距{gap:.1f}x)")
            ok = False
        if r['win_rate'] >= 0.50:
            print(f"✓ 胜率 >= 50%: {r['win_rate']:.1%}")
        else:
            print(f"✗ 胜率 < 50%: {r['win_rate']:.1%}")
            ok = False
        print(f"  持仓上限: {self.max_positions} ✓")
        if ok:
            print(f"\n★★★ 全部目标达成 ★★★")


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")

    bt = BacktestV25(initial_capital=500000)
    print("加载期货数据...")
    data = bt.load_data(data_dir)
    print(f"加载了 {len(data)} 个品种")

    end_date = pd.Timestamp('2026-05-08')
    start_date = pd.Timestamp('2018-01-01')

    results = bt.run(data, start_date, end_date)
    bt.print_results(results)

    os.makedirs(output_dir, exist_ok=True)
    save = {k: v for k, v in results.items() if not isinstance(v, dict)}
    save['yearly'] = {str(k): v for k, v in results.get('yearly', {}).items()}
    save['monthly'] = {str(k): v for k, v in results.get('monthly', {}).items()}

    with open(os.path.join(output_dir, 'backtest_v25.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()
