#!/usr/bin/env python3
"""
参数扫描 — 找最优杠杆+持有天数+止损组合
基于v24框架, 扫描关键参数
"""

import os, sys, time, numpy as np, pandas as pd
from collections import defaultdict
from itertools import product

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

        df = df.dropna(subset=['ma20', 'ma60', 'atr', 'mom_5', 'mom_10'])
        if len(df) > 100:
            data[symbol] = df
    return data


def run_backtest(data, start_date, end_date, leverage, hold_days, stop_pct):
    """快速回测"""
    max_positions = 3
    commission_rate = 0.00015

    # 构建日期索引
    date_map = defaultdict(dict)
    for symbol, df in data.items():
        mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
        for _, row in df[mask].iterrows():
            date_map[row['trade_date']][symbol] = row
    dates = sorted(date_map.keys())

    equity = 500000
    cash = 500000
    positions = {}
    closed_pnls = []
    equity_curve = []

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
                del positions[symbol]
                continue

            if stop_pct > 0:
                if pos['direction'] == 1 and price < pos['entry_price'] * (1 - stop_pct):
                    pnl = (price - pos['entry_price']) * pos['lots'] * pos['multiplier']
                    comm = price * pos['multiplier'] * pos['lots'] * commission_rate * 2
                    net = pnl - comm
                    cash += pos['margin'] + net
                    closed_pnls.append(net)
                    del positions[symbol]
                    continue
                if pos['direction'] == -1 and price > pos['entry_price'] * (1 + stop_pct):
                    pnl = (price - pos['entry_price']) * (-1) * pos['lots'] * pos['multiplier']
                    comm = price * pos['multiplier'] * pos['lots'] * commission_rate * 2
                    net = pnl - comm
                    cash += pos['margin'] + net
                    closed_pnls.append(net)
                    del positions[symbol]
                    continue

        # 入场
        if len(positions) < max_positions:
            ranked = []
            for symbol, row in day_data.items():
                if symbol in positions:
                    continue
                mom = row.get('mom_5', 0) * 0.4 + row.get('mom_10', 0) * 0.4 + row.get('mom_20', 0) * 0.2
                trend = row.get('trend', 0)
                direction = trend
                if direction == 1 and mom < 0:
                    continue
                if direction == -1 and mom > 0:
                    continue
                if row.get('atr_pct', 0.1) > 0.045:
                    continue
                ranked.append((symbol, direction, abs(mom), row['close'], row['atr']))

            ranked.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, price, atr in ranked:
                if len(positions) >= max_positions:
                    break

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

        # 权益
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                unrealized += (day_data[symbol]['close'] - pos['entry_price']) * pos['direction'] * pos['multiplier'] * pos['lots']
        equity = cash + unrealized
        equity_curve.append((date, equity))

        # 爆仓检查
        if equity < 10000:
            break

    if not equity_curve:
        return None

    final = equity_curve[-1][1]
    if final <= 0:
        return None
    total_ret = (final - 500000) / 500000
    days = (equity_curve[-1][0] - equity_curve[0][0]).days
    years = max(days / 365, 0.001)
    if total_ret <= -1:
        ann = -1.0
    else:
        ann = (1 + total_ret) ** (1 / years) - 1
    ann = float(ann) if not isinstance(ann, complex) else -1.0

    eq = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq['cummax'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['cummax']) / eq['cummax']
    mdd = eq['dd'].min()

    pnls = np.array(closed_pnls)
    wr = (pnls > 0).mean() if len(pnls) > 0 else 0

    return {
        'annual': ann,
        'mdd': mdd,
        'wr': wr,
        'trades': len(pnls),
        'final': final,
        'total_ret': total_ret,
    }


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据...")
    data = load_data(data_dir)
    print(f"加载了 {len(data)} 个品种\n")

    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2026-05-08')

    # 参数扫描
    leverages = [4, 6, 8, 10, 12]
    hold_days_list = [5, 8, 10, 15, 20]
    stop_pcts = [0, 0.05, 0.08, 0.10]

    results = []
    for lev, hd, sp in product(leverages, hold_days_list, stop_pcts):
        r = run_backtest(data, start_date, end_date, lev, hd, sp)
        if r:
            results.append({
                'leverage': lev,
                'hold_days': hd,
                'stop_pct': sp,
                **r
            })

    # 排序: 年化收益
    print(f"\n{'杠杆':>4} {'持有':>4} {'止损':>5} {'年化':>8} {'胜率':>6} {'最大回撤':>8} {'交易':>4} {'最终权益':>12}")
    print("-" * 70)

    results.sort(key=lambda x: x['annual'], reverse=True)
    for r in results[:30]:
        print(f"{r['leverage']:>4} {r['hold_days']:>4} {r['stop_pct']:>5.0%} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['mdd']:>8.1%} "
              f"{r['trades']:>4} {r['final']:>12,.0f}")

    # 找满足条件的
    print("\n\n=== 年化>50% 且 胜率>50% ===")
    good = [r for r in results if r['annual'] > 0.5 and r['wr'] > 0.50]
    if good:
        for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:10]:
            print(f"杠杆={r['leverage']} 持有={r['hold_days']}天 止损={r['stop_pct']:.0%} "
                  f"年化={r['annual']:.1%} 胜率={r['wr']:.1%} 回撤={r['mdd']:.1%} 权益={r['final']:,.0f}")
    else:
        print("无满足条件的组合")

    print(f"\n=== 年化>100% ===")
    great = [r for r in results if r['annual'] > 1.0]
    if great:
        for r in sorted(great, key=lambda x: x['annual'], reverse=True)[:10]:
            print(f"杠杆={r['leverage']} 持有={r['hold_days']}天 止损={r['stop_pct']:.0%} "
                  f"年化={r['annual']:.1%} 胜率={r['wr']:.1%} 回撤={r['mdd']:.1%} 权益={r['final']:,.0f}")
    else:
        print("无满足条件的组合")


if __name__ == '__main__':
    main()
