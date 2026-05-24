#!/usr/bin/env python3
"""
期货策略 v26 — 期权增强动量策略
核心: 用期权代替期货做方向性交易
  - 同样的动量信号 (已验证50.4% WR)
  - 买入ATM期权 (看涨/看跌) 代替开期货
  - 权利金 = 最大亏损, 盈利无上限
  - 每笔风险1-2%权益, 但收益可达5-20x权利金
  - 用BS模型估算期权价格 (用历史波动率)
"""

import os, sys, json, time, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec


def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes定价"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


class BacktestV26:
    def __init__(self, initial_capital=500000):
        self.initial_capital = initial_capital

        self.max_positions = 3
        self.risk_per_trade = 0.02  # 每笔风险2%权益
        self.hold_days = 5
        self.min_momentum = 0.03

        self.risk_free_rate = 0.02  # 无风险利率
        self.commission_rate = 0.0003  # 期权手续费更高

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

            # 历史波动率 (用于BS定价)
            df['hv_10'] = df['return'].rolling(10).std() * np.sqrt(252)
            df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
            df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)

            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']

            df['mom_10'] = df['close'].pct_change(10)
            df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

            df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10'])
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

    def _open_position(self, symbol, direction, price, hv, date):
        """买入ATM期权"""
        # ATM期权: K = 当前价格
        K = price
        T = self.hold_days / 365.0  # 持有期
        sigma = hv
        r = self.risk_free_rate

        option_type = 'call' if direction == 1 else 'put'
        premium = bs_price(price, K, T, r, sigma, option_type)

        if premium <= 0 or price <= 0:
            return

        # 期权成本 = 权利金 * 合约乘数 (每张期权对应1手期货)
        # 简化: 每张期权的权利金 = premium * multiplier
        mult, _, _, _ = get_spec(symbol)
        cost_per_contract = premium * mult  # 每张期权成本

        # 风险控制: 每笔最多亏 risk_per_trade 的权益
        risk_amount = self.equity * self.risk_per_trade
        contracts = max(int(risk_amount / cost_per_contract), 1)

        total_cost = cost_per_contract * contracts
        commission = total_cost * self.commission_rate

        if total_cost + commission > self.cash * 0.5:
            contracts = max(int((self.cash * 0.5 - commission) / cost_per_contract), 0)
            if contracts <= 0:
                return
            total_cost = cost_per_contract * contracts
            commission = total_cost * self.commission_rate

        self.cash -= total_cost + commission

        self.positions[symbol] = {
            'direction': direction,
            'option_type': option_type,
            'entry_price': price,
            'strike': K,
            'entry_date': date,
            'entry_hv': sigma,
            'premium': premium,
            'contracts': contracts,
            'cost_per': cost_per_contract,
            'total_cost': total_cost,
            'commission': commission,
            'mult': mult,
        }

    def _close_position(self, symbol, date, price, hv, reason):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]

        # 计算退出时期权价值
        remaining_T = max((self.hold_days - (date - pos['entry_date']).days) / 365.0, 0.001)
        exit_hv = hv if hv > 0.05 else 0.20  # 最低波动率

        exit_value_per = bs_price(price, pos['strike'], remaining_T,
                                   self.risk_free_rate, exit_hv, pos['option_type'])

        # 期权内在价值保底
        if pos['direction'] == 1:
            intrinsic = max(price - pos['strike'], 0)
        else:
            intrinsic = max(pos['strike'] - price, 0)
        exit_value_per = max(exit_value_per, intrinsic * 0.9)  # 考虑流动性折价

        total_exit_value = exit_value_per * pos['mult'] * pos['contracts']
        commission = total_exit_value * self.commission_rate
        net_pnl = total_exit_value - pos['total_cost'] - commission - pos['commission']

        self.cash += total_exit_value - commission  # 回收期权价值

        price_change = (price - pos['entry_price']) / pos['entry_price'] * pos['direction']
        option_return = net_pnl / pos['total_cost'] if pos['total_cost'] > 0 else 0
        equity_return = net_pnl / self.equity if self.equity > 0 else 0

        self.closed_trades.append({
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'strike': pos['strike'],
            'premium': pos['premium'],
            'contracts': pos['contracts'],
            'total_cost': pos['total_cost'],
            'exit_value': total_exit_value,
            'pnl': net_pnl,
            'price_change': price_change,
            'option_return': option_return,
            'equity_return': equity_return,
            'hold_days': (date - pos['entry_date']).days,
            'reason': reason,
        })
        del self.positions[symbol]

    def run(self, data, start_date, end_date):
        t0 = time.time()
        print(f"=== 期货策略 v26 (期权增强) ===")
        print(f"初始: {self.initial_capital:,.0f} | 每笔风险: {self.risk_per_trade:.0%}")
        print(f"持有: {self.hold_days}天 | 动量门槛: {self.min_momentum}")
        print(f"回测: {start_date.date()} ~ {end_date.date()} | 品种: {len(data)}")

        date_map = self.build_date_index(data, start_date, end_date)
        dates = sorted(date_map.keys())
        print(f"交易日: {len(dates)}")

        for date in dates:
            day_data = date_map[date]

            # 退出
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if symbol not in day_data:
                    continue
                row = day_data[symbol]
                price = row['close']
                hv = row.get('hv_20', 0.25)
                hd = (date - pos['entry_date']).days

                if hd >= self.hold_days:
                    self._close_position(symbol, date, price, hv, 'time_exit')

            # 入场
            if len(self.positions) < self.max_positions:
                signals = []
                for symbol, row in day_data.items():
                    if symbol in self.positions:
                        continue
                    if row.get('atr_pct', 0.1) > 0.045:
                        continue

                    mom = row.get('mom_10', 0)
                    trend = row.get('trend', 0)
                    hv = row.get('hv_20', 0)

                    if hv < 0.10 or hv > 0.60:  # 波动率异常
                        continue

                    if trend == 1 and mom > self.min_momentum:
                        signals.append((symbol, 1, abs(mom), hv))
                    elif trend == -1 and mom < -self.min_momentum:
                        signals.append((symbol, -1, abs(mom), hv))

                signals.sort(key=lambda x: x[2], reverse=True)

                for symbol, direction, _, hv in signals:
                    if len(self.positions) >= self.max_positions:
                        break
                    price = day_data[symbol]['close']
                    self._open_position(symbol, direction, price, hv, date)

            # 权益
            unrealized = 0
            for symbol, pos in self.positions.items():
                if symbol in day_data:
                    row = day_data[symbol]
                    price = row['close']
                    hv = row.get('hv_20', 0.25)
                    remaining_T = max((self.hold_days - (date - pos['entry_date']).days) / 365.0, 0.001)
                    exit_hv = hv if hv > 0.05 else 0.20
                    val = bs_price(price, pos['strike'], remaining_T,
                                   self.risk_free_rate, exit_hv, pos['option_type'])
                    if pos['direction'] == 1:
                        val = max(val, max(price - pos['strike'], 0) * 0.9)
                    else:
                        val = max(val, max(pos['strike'] - price, 0) * 0.9)
                    unrealized += val * pos['mult'] * pos['contracts']

            self.equity = self.cash + unrealized
            self.equity_curve.append((date, self.equity))

            if self.equity < 10000:
                break

        print(f"\n回测完成, 耗时 {time.time()-t0:.1f}秒")
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

        trades_df['year'] = trades_df['exit_date'].apply(
            lambda x: x.year if hasattr(x, 'year') else pd.Timestamp(x).year)
        yearly = {}
        for yr, grp in trades_df.groupby('year'):
            yr_wins = grp[grp['pnl'] > 0]
            yearly[int(yr)] = {
                'trades': len(grp),
                'win_rate': float(len(yr_wins) / len(grp)),
                'pnl': float(grp['pnl'].sum()),
                'avg_option_return': float(grp['option_return'].mean()),
            }

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final,
            'total_return': float(total_ret),
            'annual_return': ann,
            'max_drawdown': mdd,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades_df),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_option_return': float(trades_df['option_return'].mean()),
            'avg_price_change': float(trades_df['price_change'].mean()),
            'avg_hold_days': float(trades_df['hold_days'].mean()),
            'yearly': yearly,
        }

    def print_results(self, r):
        print("\n" + "=" * 60)
        print("期货策略 v26 (期权增强) 结果")
        print("=" * 60)
        print(f"初始资金:     {r['initial_capital']:>15,.0f}")
        print(f"最终权益:     {r['final_equity']:>15,.0f}")
        print(f"总收益率:     {r['total_return']:>15.2%}")
        print(f"年化收益率:   {r['annual_return']:>15.2%}")
        print(f"最大回撤:     {r['max_drawdown']:>15.2%}")
        print(f"夏普比率:     {r['sharpe_ratio']:>15.2f}")
        print(f"总交易次数:   {r['total_trades']:>15}")
        print(f"胜率:         {r['win_rate']:>15.2%}")
        print(f"盈亏比:       {r['profit_factor']:>15.2f}")
        if r.get('avg_option_return'):
            print(f"平均期权收益: {r['avg_option_return']:>15.2%}")
        if r.get('avg_price_change'):
            print(f"平均价格变动: {r['avg_price_change']:>15.2%}")
        print("=" * 60)

        if r.get('yearly'):
            print(f"\n年度统计:")
            print(f"  {'年份':>4} {'交易':>4} {'胜率':>6} {'PnL':>14} {'期权收益':>10}")
            for yr, s in sorted(r['yearly'].items()):
                print(f"  {yr:>4} {s['trades']:>4} {s['win_rate']:>6.1%} "
                      f"{s['pnl']:>14,.0f} {s['avg_option_return']:>10.1%}")

        print(f"\n--- 目标检查 ---")
        ok = True
        if r['annual_return'] >= 6.0:
            print(f"✓ 年化 >= 600%: {r['annual_return']:.1%}")
        else:
            print(f"✗ 年化 < 600%: {r['annual_return']:.1%}")
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

    bt = BacktestV26(initial_capital=500000)
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

    with open(os.path.join(output_dir, 'backtest_v26.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()
