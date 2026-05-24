#!/usr/bin/env python3
"""
期货策略 v20 — 动量轮动 + 持仓量确认
v19复盘: 固定10天太死, 2022大趋势被洗出, hard_stop致命
改进:
1. 动态持有期 (趋势强度决定持有8-25天)
2. 持仓量(OI)确认 — 跟随资金流向
3. 品种轮动 — 每日排名, 始终持有最强3个
4. 15x杠杆, 波动率目标仓位管理
5. 极端行情保护 (单日最大亏损限制)
"""

import os, json, time, numpy as np, pandas as pd
from collections import defaultdict


class BacktestV20:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital

        # 杠杆与仓位
        self.max_positions = 3
        self.leverage = 15            # 提高15x
        self.target_vol = 0.15        # 目标年化波动率15%/仓
        self.max_equity_per_pos = 0.35
        self.min_equity_per_pos = 0.10

        # 持仓管理
        self.min_hold_days = 5        # 最少持5天
        self.max_hold_days = 25       # 最长25天
        self.hard_stop_atr = 4.0      # 硬止损4x ATR
        self.max_loss_per_trade = 0.08  # 单笔最大亏损8%权益

        # 入场参数
        self.min_score = 65

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

            # 基础指标
            df['return'] = df['close'].pct_change()
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma60'] = df['close'].rolling(60).mean()
            df['ma5'] = df['close'].rolling(5).mean()

            # ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))

            # 成交量
            df['vol_ma20'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma20']

            # 动量指标
            df['roc5'] = df['close'].pct_change(5) * 100
            df['roc10'] = df['close'].pct_change(10) * 100
            df['roc20'] = df['close'].pct_change(20) * 100

            # 突破位
            df['high_15'] = df['close'].rolling(15).max().shift(1)
            df['low_15'] = df['close'].rolling(15).min().shift(1)

            # ADX
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            atr_smooth = df['atr']
            plus_di = 100 * plus_dm.rolling(14).mean() / atr_smooth
            minus_di = 100 * minus_dm.rolling(14).mean() / atr_smooth
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
            df['adx'] = dx.rolling(14).mean()

            # 持仓量分析 (OI)
            df['oi_ma10'] = df['oi'].rolling(10).mean()
            df['oi_change'] = df['oi'].pct_change(5)  # 5日OI变化率
            # OI与价格关系:
            # 价格涨+OI增 = 多头增仓 (强)
            # 价格跌+OI增 = 空头增仓 (强)
            df['oi_price_align'] = np.sign(df['return']) * np.sign(df['oi_change'])

            # 综合动量分 (用于品种排名)
            df['momentum_score'] = (
                df['roc5'].rank(pct=True) * 0.3 +
                df['roc10'].rank(pct=True) * 0.3 +
                df['roc20'].rank(pct=True) * 0.2 +
                df['vol_ratio'].rank(pct=True) * 0.2
            ) * 100

            df = df.dropna(subset=['ma20', 'ma60', 'atr', 'adx', 'high_15', 'low_15'])
            if len(df) > 100:
                data[symbol] = df
        return data

    def compute_signal_score(self, row):
        """多因子评分 (含OI确认)"""
        long_score = 0
        short_score = 0
        close = row['close']

        # 1. 趋势对齐 (20分)
        if row['ma20'] > row['ma60'] and close > row['ma20']:
            long_score += 20
        if row['ma20'] < row['ma60'] and close < row['ma20']:
            short_score += 20

        # 2. 突破 (15分)
        if close > row['high_15']:
            long_score += 15
        if close < row['low_15']:
            short_score += 15

        # 3. 动量 (15分)
        if row['roc10'] > 2:
            long_score += 15
        if row['roc10'] < -2:
            short_score += 15

        # 4. 成交量 (10分)
        if row['vol_ratio'] > 1.2:
            long_score += 10
            short_score += 10

        # 5. ADX趋势强度 (15分)
        if row['adx'] > 25:
            long_score += 15
            short_score += 15

        # 6. 持仓量确认 (15分) — 跟随资金
        if row['oi_change'] > 0.02:  # OI增加
            long_score += 15
            short_score += 15
        elif row['oi_change'] < -0.02:  # OI减少
            long_score -= 5   # 减分
            short_score -= 5

        # 7. 波动率适中 (10分)
        if row['atr_pct'] < 0.04:
            long_score += 10
            short_score += 10

        return long_score, short_score

    def get_hold_days(self, row):
        """根据趋势强度动态决定持有天数"""
        adx = row['adx']
        if adx > 35:
            return 25  # 强趋势, 长持
        elif adx > 28:
            return 20
        elif adx > 22:
            return 15
        else:
            return 10  # 弱趋势, 短持

    def build_date_index(self, data, start_date, end_date):
        date_map = defaultdict(dict)
        for symbol, df in data.items():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            for _, row in df[mask].iterrows():
                date_map[row['trade_date']][symbol] = row
        return date_map

    def _open_position(self, symbol, direction, price, atr, atr_pct, date, score, hold_days):
        """开仓 — 波动率目标仓位管理"""
        # 基于波动率的仓位: target annual vol / (daily vol * leverage * sqrt(252))
        daily_vol = atr_pct
        if daily_vol <= 0.005:
            daily_vol = 0.005

        # 目标: 该仓位的年化波动贡献 = target_vol
        # position_vol = equity_alloc * daily_vol * leverage * sqrt(252)
        # equity_alloc = target_vol / (daily_vol * leverage * sqrt(252))
        equity_alloc = self.target_vol / (daily_vol * self.leverage * np.sqrt(252))
        equity_alloc = min(equity_alloc, self.max_equity_per_pos)
        equity_alloc = max(equity_alloc, self.min_equity_per_pos)

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
            'stop_price': price - direction * self.hard_stop_atr * atr,
            'target_hold_days': hold_days,
            'commission': commission,
            'score': score,
            'max_loss_equity': self.equity * self.max_loss_per_trade,
        }

    def _close_position(self, symbol, date, price, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]

        price_change = (price - pos['entry_price']) / pos['entry_price']
        leveraged_return = price_change * pos['direction'] * self.leverage
        pnl = pos['equity_alloc'] * leveraged_return
        commission = pos['equity_alloc'] * self.commission_rate
        net_pnl = pnl - commission

        self.cash += pos['equity_alloc'] + net_pnl

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
            'hold_days': (date - pos['entry_date']).days,
            'reason': reason,
            'score': pos['score'],
            'target_hold': pos['target_hold_days'],
        })
        del self.positions[symbol]

    def run(self, data, start_date, end_date):
        t0 = time.time()
        print(f"=== 期货策略 v20 (动量轮动+OI确认) ===")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"杠杆: {self.leverage}x | 目标波动: {self.target_vol:.0%}")
        print(f"持有: {self.min_hold_days}-{self.max_hold_days}天(动态)")
        print(f"回测: {start_date.date()} ~ {end_date.date()}")
        print(f"品种: {len(data)}")

        date_map = self.build_date_index(data, start_date, end_date)
        dates = sorted(date_map.keys())
        print(f"交易日: {len(dates)}")

        for date in dates:
            day_data = date_map[date]

            # === 1. 更新持仓 & 退出检查 ===
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if symbol not in day_data:
                    continue

                row = day_data[symbol]
                price = row['close']
                hold_days = (date - pos['entry_date']).days

                # 浮亏检查 — 单笔最大亏损保护
                price_change = (price - pos['entry_price']) / pos['entry_price']
                unrealized_pnl = pos['equity_alloc'] * price_change * pos['direction'] * self.leverage
                if unrealized_pnl < -pos['max_loss_equity']:
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

                # 趋势反转退出 (仅盈利仓)
                if hold_days >= self.min_hold_days and unrealized_pnl > 0:
                    if pos['direction'] == 1 and row['ma5'] < row['ma20'] and row['rsi'] < 45:
                        self._close_position(symbol, date, price, 'trend_reverse')
                        continue
                    if pos['direction'] == -1 and row['ma5'] > row['ma20'] and row['rsi'] > 55:
                        self._close_position(symbol, date, price, 'trend_reverse')
                        continue

            # === 2. 品种轮动 — 每日排名, 替换弱势仓 ===
            if len(self.positions) > 0:
                # 给当前持仓重新评分
                for symbol in list(self.positions.keys()):
                    pos = self.positions[symbol]
                    if symbol not in day_data:
                        continue
                    row = day_data[symbol]
                    hold_days = (date - pos['entry_date']).days

                    if hold_days < self.min_hold_days:
                        continue

                    long_score, short_score = self.compute_signal_score(row)
                    current_score = long_score if pos['direction'] == 1 else short_score

                    # 如果信号大幅减弱, 准备被替换
                    if current_score < 40:
                        price = row['close']
                        unrealized_pnl = pos['equity_alloc'] * ((price - pos['entry_price']) / pos['entry_price']) * pos['direction'] * self.leverage
                        # 亏损且信号弱 → 退出
                        if unrealized_pnl < 0:
                            self._close_position(symbol, date, price, 'signal_decay')

            # === 3. 新信号入场 ===
            if len(self.positions) < self.max_positions:
                signals = []
                for symbol, row in day_data.items():
                    if symbol in self.positions:
                        continue

                    long_score, short_score = self.compute_signal_score(row)
                    direction = 0
                    score = 0
                    if long_score >= self.min_score and long_score > short_score:
                        direction = 1
                        score = long_score
                    elif short_score >= self.min_score:
                        direction = -1
                        score = short_score

                    if direction != 0:
                        hold_days = self.get_hold_days(row)
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

            # === 4. 权益计算 ===
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
            'reason_stats': reason_stats,
            'yearly_stats': yearly_stats,
        }

    def print_results(self, results):
        print("\n" + "=" * 60)
        print("期货策略 v20 结果")
        print("=" * 60)
        for k in ['initial_capital', 'final_equity']:
            print(f"{k:>14s}: {results[k]:>15,.0f}")
        for k in ['total_return', 'annual_return', 'max_drawdown', 'win_rate']:
            print(f"{k:>14s}: {results[k]:>15.2%}")
        for k in ['sharpe_ratio', 'profit_factor']:
            print(f"{k:>14s}: {results[k]:>15.2f}")
        print(f"{'total_trades':>14s}: {results['total_trades']:>15}")
        if results.get('avg_hold_days'):
            print(f"{'avg_hold_days':>14s}: {results['avg_hold_days']:>15.1f}")
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

    bt = BacktestV20(initial_capital=500000)
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

    with open(os.path.join(output_dir, 'backtest_v20.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()
