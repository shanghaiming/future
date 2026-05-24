#!/usr/bin/env python3
"""
策略对比扫描 — 动量 vs 均值回归 vs 混合
目标: 找到最高年化+最高胜率的策略类型
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
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

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

        # 布林带
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        df['mom_20'] = df['close'].pct_change(20)
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']

        df = df.dropna(subset=['ma20', 'ma60', 'atr', 'rsi', 'bb_pct', 'mom_10'])
        if len(df) > 100:
            data[symbol] = df
    return data


def get_signals(strategy, day_data, positions):
    """根据策略类型生成信号列表"""
    signals = []

    for symbol, row in day_data.items():
        if symbol in positions:
            continue

        # 过滤高波动
        if row.get('atr_pct', 0.1) > 0.045:
            continue

        if strategy == 'momentum':
            # 纯动量: 趋势方向 + 强动量
            mom = row.get('mom_10', 0)
            trend = row.get('trend', 0)
            if trend == 1 and mom > 0.01:
                signals.append((symbol, 1, abs(mom)))
            elif trend == -1 and mom < -0.01:
                signals.append((symbol, -1, abs(mom)))

        elif strategy == 'mean_revert':
            # 均值回归: RSI极端 + 布林带外
            rsi = row.get('rsi', 50)
            bb = row.get('bb_pct', 0.5)
            if rsi < 25 and bb < 0.1:
                signals.append((symbol, 1, (30 - rsi) / 30))
            elif rsi > 75 and bb > 0.9:
                signals.append((symbol, -1, (rsi - 70) / 30))

        elif strategy == 'mr_trend':
            # 均值回归+趋势: RSI极端但方向和趋势一致
            rsi = row.get('rsi', 50)
            trend = row.get('trend', 0)
            bb = row.get('bb_pct', 0.5)
            if trend == 1 and rsi < 35 and bb < 0.2:
                signals.append((symbol, 1, (35 - rsi) / 35 * abs(trend)))
            elif trend == -1 and rsi > 65 and bb > 0.8:
                signals.append((symbol, -1, (rsi - 65) / 35 * abs(trend)))

        elif strategy == 'breakout_vol':
            # 突破+量能: 价格突破+成交量放大
            close = row['close']
            ma20 = row.get('ma20', close)
            ma60 = row.get('ma60', close)
            vr = row.get('vol_ratio', 1.0)
            if vr < 1.5:
                continue
            if close > ma20 > ma60:
                signals.append((symbol, 1, vr))
            elif close < ma20 < ma60:
                signals.append((symbol, -1, vr))

        elif strategy == 'bb_squeeze':
            # 布林带收缩后突破
            bb = row.get('bb_pct', 0.5)
            rsi = row.get('rsi', 50)
            vr = row.get('vol_ratio', 1.0)
            if vr < 1.3:
                continue
            if bb > 0.85 and rsi > 55:
                signals.append((symbol, 1, bb * vr))
            elif bb < 0.15 and rsi < 45:
                signals.append((symbol, -1, (1 - bb) * vr))

        elif strategy == 'multi_signal':
            # 多信号融合: 趋势+动量+RSI
            mom = row.get('mom_5', 0) * 0.5 + row.get('mom_10', 0) * 0.5
            trend = row.get('trend', 0)
            rsi = row.get('rsi', 50)
            vr = row.get('vol_ratio', 1.0)

            long_score = 0
            short_score = 0
            if trend == 1: long_score += 30
            if trend == -1: short_score += 30
            if mom > 0.01: long_score += 25
            if mom < -0.01: short_score += 25
            if rsi > 50 and rsi < 70: long_score += 20
            if rsi < 50 and rsi > 30: short_score += 20
            if vr > 1.2: long_score += 15; short_score += 15

            if long_score >= 60:
                signals.append((symbol, 1, long_score))
            elif short_score >= 60:
                signals.append((symbol, -1, short_score))

        elif strategy == 'long_only_momentum':
            # 只做多动量 (避免做空劣势)
            mom = row.get('mom_10', 0)
            trend = row.get('trend', 0)
            rsi = row.get('rsi', 50)
            if trend == 1 and mom > 0.01 and rsi < 70:
                signals.append((symbol, 1, abs(mom)))

    signals.sort(key=lambda x: x[2], reverse=True)
    return signals


def run_backtest(data, start_date, end_date, strategy, leverage=4, hold_days=10):
    max_positions = 3
    commission_rate = 0.00015

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

        # 入场
        if len(positions) < max_positions:
            signals = get_signals(strategy, day_data, positions)

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
    ann = (1 + total_ret) ** (1 / years) - 1

    eq = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq['cummax'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['cummax']) / eq['cummax']
    mdd = eq['dd'].min()

    pnls = np.array(closed_pnls)
    wr = (pnls > 0).mean() if len(pnls) > 0 else 0
    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
    avg_loss = abs(pnls[pnls <= 0].mean()) if (pnls <= 0).any() else 1
    pf = avg_win * (pnls > 0).sum() / (avg_loss * (pnls <= 0).sum()) if (pnls <= 0).sum() > 0 and avg_loss > 0 else 0

    return {
        'strategy': strategy,
        'leverage': leverage,
        'hold_days': hold_days,
        'annual': float(ann) if not isinstance(ann, complex) else -1.0,
        'wr': wr,
        'mdd': mdd,
        'pf': pf,
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

    strategies = [
        'momentum',
        'mean_revert',
        'mr_trend',
        'breakout_vol',
        'bb_squeeze',
        'multi_signal',
        'long_only_momentum',
    ]

    results = []
    for strat in strategies:
        for lev in [3, 4, 5, 6]:
            for hd in [5, 8, 10, 15]:
                r = run_backtest(data, start_date, end_date, strat, lev, hd)
                if r:
                    results.append(r)

    # 打印结果
    print(f"\n{'策略':>22} {'杠杆':>4} {'持有':>4} {'年化':>8} {'胜率':>6} {'盈亏比':>6} {'回撤':>8} {'交易':>4} {'最终':>12}")
    print("-" * 90)

    results.sort(key=lambda x: x['annual'], reverse=True)
    for r in results[:40]:
        print(f"{r['strategy']:>22} {r['leverage']:>4} {r['hold_days']:>4} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} {r['mdd']:>8.1%} "
              f"{r['trades']:>4} {r['final']:>12,.0f}")

    # 按策略分组最优
    print("\n\n=== 每种策略最优结果 ===")
    best_by_strat = {}
    for r in results:
        s = r['strategy']
        if s not in best_by_strat or r['annual'] > best_by_strat[s]['annual']:
            best_by_strat[s] = r

    for s, r in sorted(best_by_strat.items(), key=lambda x: x[1]['annual'], reverse=True):
        print(f"  {s:>22}: 年化={r['annual']:.1%}  胜率={r['wr']:.1%}  "
              f"杠杆={r['leverage']}x  持有={r['hold_days']}天  回撤={r['mdd']:.1%}  "
              f"权益={r['final']:,.0f}")


if __name__ == '__main__':
    main()
