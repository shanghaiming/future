#!/usr/bin/env python3
"""
期货策略 v21 — 真实合约规格 + 动量轮动
核心改进:
1. 真实保证金/乘数/手数 — 不再用简化百分比模型
2. v19验证有效的逻辑: 固定持有期 + 宽止损 + 高杠杆
3. 持仓量(OI)确认 + ADX动态持有期
4. 单笔最大亏损保护
"""

import os, sys, json, time, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec, calc_margin, calc_pnl, calc_max_lots


class BacktestV21:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital

        # 仓位参数
        self.max_positions = 3
        self.margin_usage = 0.60   # 用60%资金作为保证金 (3仓各20%)
        self.risk_per_trade = 0.05 # 单笔风险5%权益

        # 持仓管理
        self.min_hold_days = 5
        self.max_hold_days_base = 15
        self.hard_stop_atr = 5.0
        self.max_loss_pct = 0.10   # 单笔最大亏损10%权益

        # 入场
        self.min_score = 65

        # 费用
        self.commission_rate = 0.00015  # 万1.5手续费
        self.slippage_pct = 0.0001      # 万1滑点

        # 状态
        self.equity = initial_capital
        self.cash = initial_capital
        self.margin_used = 0
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

            # 基础指标
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

            # OI分析
            df['oi_change'] = df['oi'].pct_change(5)

            df = df.dropna(subset=['ma20', 'ma60', 'atr', 'adx', 'high_15', 'low_15'])
            if len(df) > 100:
                data[symbol] = df
        return data

    def score_signal(self, row):
        """多因子评分"""
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

        # OI确认 (10)
        if row.get('oi_change', 0) > 0.02:
            long_score += 10
            short_score += 10

        # 波动率 (5)
        if row['atr_pct'] < 0.04:
            long_score += 5
            short_score += 5

        return long_score, short_score

    def get_hold_days(self, adx):
        if adx > 35: return 25
        elif adx > 28: return 20
        elif adx > 22: return 15
        else: return 10

    def build_date_index(self, data, start_date, end_date):
        date_map = defaultdict(dict)
        for symbol, df in data.items():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            for _, row in df[mask].iterrows():
                date_map[row['trade_date']][symbol] = row
        return date_map

    def _open_position(self, symbol, direction, price, atr, atr_pct, date, score, hold_days):
        """开仓 — 真实手数计算"""
        multiplier, margin_rate, tick_size, _ = get_spec(symbol)

        # 合约价值
        contract_value = price * multiplier
        margin_per_lot = contract_value * margin_rate

        if margin_per_lot <= 0:
            return

        # 基于风险确定手数: risk_amount = 止损距离 * multiplier * lots
        stop_distance = max(self.hard_stop_atr * atr, price * 0.02)  # 至少2%
        risk_per_lot = stop_distance * multiplier
        risk_amount = self.equity * self.risk_per_trade

        lots_by_risk = max(int(risk_amount / risk_per_lot), 1)

        # 保证金约束: 最多用 margin_usage/3 的资金
        max_margin = self.equity * self.margin_usage / self.max_positions
        lots_by_margin = max(int(max_margin / margin_per_lot), 0)

        lots = min(lots_by_risk, lots_by_margin)
        if lots <= 0:
            return

        # 检查资金够不够
        total_margin = margin_per_lot * lots
        commission = contract_value * lots * self.commission_rate
        total_cost = total_margin + commission

        if total_cost > self.cash:
            lots = max(int((self.cash - commission) / margin_per_lot), 0)
            if lots <= 0:
                return
            total_margin = margin_per_lot * lots
            commission = contract_value * lots * self.commission_rate
            total_cost = total_margin + commission

        # 滑点
        fill_price = price * (1 + self.slippage_pct * direction)

        self.cash -= total_cost  # 扣保证金+手续费
        self.margin_used += total_margin

        self.positions[symbol] = {
            'direction': direction,
            'entry_price': fill_price,
            'entry_date': date,
            'entry_atr': atr,
            'lots': lots,
            'multiplier': multiplier,
            'margin_rate': margin_rate,
            'margin': total_margin,
            'commission_paid': commission,
            'stop_price': fill_price - direction * self.hard_stop_atr * atr,
            'target_hold_days': hold_days,
            'max_loss_amount': self.equity * self.max_loss_pct,
            'score': score,
        }

    def _close_position(self, symbol, date, price, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]

        # 滑点
        fill_price = price * (1 - self.slippage_pct * pos['direction'])

        # 盈亏
        pnl = (fill_price - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']

        # 手续费
        commission = fill_price * pos['multiplier'] * pos['lots'] * self.commission_rate
        net_pnl = pnl - commission

        # 回收保证金 + 盈亏
        self.cash += pos['margin'] + net_pnl
        self.margin_used -= pos['margin']

        hold_days = (date - pos['entry_date']).days
        leveraged_return = net_pnl / (pos['margin']) if pos['margin'] > 0 else 0

        self.closed_trades.append({
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': fill_price,
            'lots': pos['lots'],
            'multiplier': pos['multiplier'],
            'margin': pos['margin'],
            'pnl': net_pnl,
            'gross_pnl': pnl,
            'commission': pos['commission_paid'] + commission,
            'hold_days': hold_days,
            'leveraged_return': leveraged_return,
            'reason': reason,
            'score': pos['score'],
        })
        del self.positions[symbol]

    def run(self, data, start_date, end_date):
        t0 = time.time()
        print(f"=== 期货策略 v21 (真实合约规格) ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"保证金利用率: {self.margin_usage:.0%}")
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
                hold_days = (date - pos['entry_date']).days

                # 当前浮动盈亏
                unrealized = (price - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']

                # 最大亏损保护
                if unrealized < -pos['max_loss_amount']:
                    self._close_position(symbol, date, price, 'max_loss')
                    continue

                # 硬止损
                if pos['direction'] == 1 and price < pos['stop_price']:
                    self._close_position(symbol, date, price, 'hard_stop')
                    continue
                if pos['direction'] == -1 and price > pos['stop_price']:
                    self._close_position(symbol, date, price, 'hard_stop')
                    continue

                # 动态持有期退出
                if hold_days >= pos['target_hold_days']:
                    self._close_position(symbol, date, price, 'time_exit')
                    continue

                # 趋势反转 (盈利仓, 持有>5天)
                if hold_days >= self.min_hold_days and unrealized > 0:
                    if pos['direction'] == 1 and row['ma5'] < row['ma20'] and row['rsi'] < 40:
                        self._close_position(symbol, date, price, 'trend_reverse')
                        continue
                    if pos['direction'] == -1 and row['ma5'] > row['ma20'] and row['rsi'] > 60:
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
                        hold_days = self.get_hold_days(row['adx'])
                        signals.append({
                            'symbol': symbol,
                            'direction': direction,
                            'score': score,
                            'atr': row['atr'],
                            'atr_pct': row['atr_pct'],
                            'price': row['close'],
                            'hold_days': hold_days,
                        })

                signals.sort(key=lambda x: x['score'], reverse=True)

                for sig in signals:
                    if len(self.positions) >= self.max_positions:
                        break
                    self._open_position(
                        sig['symbol'], sig['direction'],
                        sig['price'], sig['atr'], sig['atr_pct'],
                        date, sig['score'], sig['hold_days']
                    )

            # === 3. 权益计算 ===
            unrealized = 0
            for symbol, pos in self.positions.items():
                if symbol in day_data:
                    price = day_data[symbol]['close']
                    unrealized += (price - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']

            self.equity = self.cash + unrealized  # cash含保证金, unrealized是浮动盈亏
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

        # 平均杠杆收益
        avg_leveraged = trades_df['leveraged_return'].mean() if 'leveraged_return' in trades_df else 0

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
            'avg_leveraged_return': avg_leveraged,
            'reason_stats': reason_stats,
            'yearly_stats': yearly_stats,
        }

    def print_results(self, results):
        print("\n" + "=" * 60)
        print("期货策略 v21 结果 (真实合约规格)")
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

    bt = BacktestV21(initial_capital=500000)
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

    with open(os.path.join(output_dir, 'backtest_v21.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()
