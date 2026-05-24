#!/usr/bin/env python3
"""
动量策略精调 — 基于strategy_compare结果
已确认: momentum + 5x + 5天 = 55.8%年化
目标: 进一步优化到更高
"""

import os, sys, time, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

def load_data(data_dir):
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
        df['mom_20'] = df['close'].pct_change(20)
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

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

        # 波动率指标
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)

        df = df.dropna(subset=['ma20', 'ma60', 'atr', 'mom_5', 'mom_10', 'adx'])
        if len(df) > 100:
            data[symbol] = df
    return data


def run_backtest(data, start_date, end_date, leverage, hold_days,
                 min_adx=0, max_atr_pct=0.045, min_mom=0.01,
                 use_adx_filter=False, use_vol_filter=False):
    max_positions = 3
    commission_rate = 0.00015

    date_map = defaultdict(dict)
    for symbol, df in data.items():
        mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
        for _, row in df[mask].iterrows():
            date_map[row['trade_date']][symbol] = row
    dates = sorted(date_map.keys())

    equity = 500000.0
    cash = 500000.0
    positions = {}
    closed_pnls = []
    equity_curve = []
    yearly_pnl = defaultdict(float)

    for date in dates:
        day_data = date_map[date]

        # 退出
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data:
                continue
            price = day_data[symbol]['close']
            hd = (date - pos['entry_date']).days

            if hd >= hold_days:
                pnl = (price - pos['entry_price']) * pos['direction'] * pos['lots'] * pos['multiplier']
                comm = price * pos['multiplier'] * pos['lots'] * commission_rate * 2
                net = pnl - comm
                cash += pos['margin'] + net
                closed_pnls.append(net)
                yr = date.year
                yearly_pnl[yr] += net
                del positions[symbol]

        # 入场
        if len(positions) < max_positions:
            signals = []
            for symbol, row in day_data.items():
                if symbol in positions:
                    continue

                # 基础过滤
                if row.get('atr_pct', 0.1) > max_atr_pct:
                    continue

                # ADX过滤
                if use_adx_filter and row.get('adx', 0) < min_adx:
                    continue

                mom = row.get('mom_10', 0)
                trend = row.get('trend', 0)

                if trend == 1 and mom > min_mom:
                    signals.append((symbol, 1, abs(mom)))
                elif trend == -1 and mom < -min_mom:
                    signals.append((symbol, -1, abs(mom)))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score in signals:
                if len(positions) >= max_positions:
                    break

                row = day_data[symbol]
                price = row['close']
                mult, mr, _, _ = get_spec(symbol)
                mpl = price * mult * mr
                if mpl <= 0:
                    continue

                target_notional = equity * (leverage / max_positions)
                lots = max(int(target_notional / (price * mult)), 1)

                total_m = sum(p['margin'] for p in positions.values()) + mpl * lots
                if total_m > equity * 0.85:
                    lots = max(int((equity * 0.85 - sum(p['margin'] for p in positions.values())) / mpl), 0)
                    if lots <= 0:
                        continue

                actual_margin = mpl * lots
                comm = price * mult * lots * commission_rate
                if actual_margin + comm > cash:
                    lots = max(int((cash - comm) / mpl), 0)
                    if lots <= 0:
                        continue
                    actual_margin = mpl * lots
                    comm = price * mult * lots * commission_rate

                cash -= actual_margin + comm
                positions[symbol] = {
                    'direction': direction,
                    'entry_price': price * (1 + 0.0001 * direction),
                    'entry_date': date,
                    'lots': lots,
                    'multiplier': mult,
                    'margin': actual_margin,
                }

        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                unrealized += (day_data[symbol]['close'] - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']
        equity = cash + unrealized
        equity_curve.append((date, equity))

        if equity < 5000:
            break

    if not equity_curve or equity_curve[-1][1] <= 0:
        return None

    final = equity_curve[-1][1]
    total_ret = (final - 500000) / 500000
    days = (equity_curve[-1][0] - equity_curve[0][0]).days
    years = max(days / 365, 0.001)
    ann = float((1 + total_ret) ** (1 / years) - 1)

    eq = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq['cummax'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['cummax']) / eq['cummax']
    mdd = float(eq['dd'].min())

    pnls = np.array(closed_pnls)
    wr = float((pnls > 0).mean()) if len(pnls) > 0 else 0

    return {
        'leverage': leverage,
        'hold_days': hold_days,
        'min_adx': min_adx,
        'max_atr': max_atr_pct,
        'min_mom': min_mom,
        'annual': ann,
        'wr': wr,
        'mdd': mdd,
        'trades': len(pnls),
        'final': final,
        'total_ret': float(total_ret),
        'yearly_pnl': dict(yearly_pnl),
    }


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据...")
    data = load_data(data_dir)
    print(f"加载了 {len(data)} 个品种\n")

    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2026-05-08')

    results = []

    # 1. 精调杠杆+持有天数
    print("=== 扫描杠杆+持有天数 ===")
    for lev in [3, 4, 5, 6, 7, 8]:
        for hd in [2, 3, 4, 5, 6, 7, 8]:
            r = run_backtest(data, start_date, end_date, lev, hd)
            if r:
                results.append(r)

    # 2. ADX过滤
    print("=== 扫描ADX过滤 ===")
    for lev in [4, 5, 6]:
        for hd in [3, 4, 5]:
            for adx in [20, 25, 30]:
                r = run_backtest(data, start_date, end_date, lev, hd,
                                min_adx=adx, use_adx_filter=True)
                if r:
                    r['config'] = f'adx>{adx}'
                    results.append(r)

    # 3. 动量门槛
    print("=== 扫描动量门槛 ===")
    for lev in [4, 5, 6]:
        for hd in [3, 4, 5]:
            for mom in [0.005, 0.01, 0.02, 0.03]:
                r = run_backtest(data, start_date, end_date, lev, hd, min_mom=mom)
                if r:
                    r['config'] = f'mom>{mom}'
                    results.append(r)

    # 4. 波动率过滤
    print("=== 扫描波动率过滤 ===")
    for lev in [4, 5, 6]:
        for hd in [3, 4, 5]:
            for atr in [0.03, 0.035, 0.04, 0.045]:
                r = run_backtest(data, start_date, end_date, lev, hd, max_atr_pct=atr)
                if r:
                    r['config'] = f'atr<{atr}'
                    results.append(r)

    # 排序输出
    print(f"\n\n{'杠杆':>4} {'持有':>4} {'配置':>12} {'年化':>8} {'胜率':>6} {'回撤':>8} {'交易':>4} {'最终':>14}")
    print("-" * 75)

    results.sort(key=lambda x: x['annual'], reverse=True)
    for r in results[:50]:
        cfg = r.get('config', '')
        print(f"{r['leverage']:>4} {r['hold_days']:>4} {cfg:>12} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['mdd']:>8.1%} "
              f"{r['trades']:>4} {r['final']:>14,.0f}")

    # TOP 5 的年度明细
    print("\n\n=== TOP 5 年度明细 ===")
    for i, r in enumerate(results[:5]):
        print(f"\n--- #{i+1}: 杠杆={r['leverage']}x 持有={r['hold_days']}天 年化={r['annual']:.1%} ---")
        if r.get('yearly_pnl'):
            for yr in sorted(r['yearly_pnl'].keys()):
                pnl = r['yearly_pnl'][yr]
                print(f"  {yr}: {pnl:>12,.0f}")

    # 达标检查
    print("\n\n=== 达标统计 ===")
    above_100 = len([r for r in results if r['annual'] > 1.0])
    above_50 = len([r for r in results if r['annual'] > 0.5])
    wr_above_50 = len([r for r in results if r['wr'] >= 0.50])
    both = len([r for r in results if r['annual'] > 0.5 and r['wr'] >= 0.50])
    print(f"年化>100%: {above_100}个组合")
    print(f"年化>50%: {above_50}个组合")
    print(f"胜率>=50%: {wr_above_50}个组合")
    print(f"年化>50%且胜率>=50%: {both}个组合")


if __name__ == '__main__':
    main()
