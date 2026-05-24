#!/usr/bin/env python3
"""
策略 v33 — 期货+期权组合仓 (方向+凸性)
核心数学:
  期货: 50% WR, PF=1.3 → 每笔期望 = 0.15 * notional
  买期权: 39% WR, PF=3.0 → 每笔期望 = 0.17 * premium
  组合: 期货方向仓 + 买同方向ATM期权
    - 赢: 期货收益 + 期权3x收益 → 高回报
    - 输: 期货亏损 + 期权权利金损失 → 多亏一点
    - WR ≈ 期货的50% (方向正确时两者都赚)
    - PF ≈ 2.0+ (赢时赚更多)

  50% WR + PF 2.0:
    Kelly = (0.5*2 - 0.5)/2 = 0.25 = 25% per trade
    半Kelly = 12.5% → 几何增长 4.6% per trade
    150 trades → 复利爆炸
"""

import os, sys, time, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec


def bs_price(S, K, T, r, sigma, opt='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


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
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
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


def run_backtest(data, start_date, end_date, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 5)
    leverage = params.get('leverage', 5)
    opt_pct = params.get('opt_pct', 0.5)  # 期权占组合的比例 (0=纯期货, 1=纯期权)
    risk_pct = params.get('risk_pct', 0.02)  # 期权风险 (权益%)
    min_mom = params.get('min_mom', 0.03)
    r = 0.02
    comm_fut = 0.00015
    comm_opt = 0.0003

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

    for date in dates:
        day_data = date_map[date]

        # === 退出 ===
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data:
                continue
            row = day_data[symbol]
            price = row['close']
            hv = row.get('hv_20', 0.25)
            hd = (date - pos['entry_date']).days

            if hd < hold_days:
                continue

            # 期货部分
            fut_pnl = 0
            fut_comm = 0
            if pos.get('fut_lots', 0) > 0:
                fut_pnl = (price - pos['fut_entry']) * pos['dir'] * pos['fut_mult'] * pos['fut_lots']
                fut_comm = price * pos['fut_mult'] * pos['fut_lots'] * comm_fut
                cash += pos['fut_margin'] + fut_pnl - fut_comm

            # 期权部分
            opt_pnl = 0
            opt_comm = 0
            if pos.get('opt_contracts', 0) > 0:
                rem_T = max(0.001 / 365, 0.001)
                val = bs_price(price, pos['opt_strike'], rem_T, r, hv, pos['opt_type'])
                intrinsic = max(price - pos['opt_strike'], 0) if pos['dir'] == 1 else max(pos['opt_strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                total_val = val * pos['fut_mult'] * pos['opt_contracts']
                opt_comm = total_val * comm_opt
                opt_pnl = total_val - pos['opt_cost'] - opt_comm - pos['opt_comm']
                cash += total_val - opt_comm

            net = fut_pnl - fut_comm + opt_pnl
            closed_pnls.append(net)
            del positions[symbol]

        # === 入场 ===
        if len(positions) < max_pos:
            signals = []
            for symbol, row in day_data.items():
                if symbol in positions:
                    continue
                if row.get('atr_pct', 0.1) > 0.045:
                    continue
                mom = row.get('mom_10', 0)
                trend = row.get('trend', 0)
                hv = row.get('hv_20', 0)
                if hv < 0.10 or hv > 0.60:
                    continue
                if trend == 1 and mom > min_mom:
                    signals.append((symbol, 1, abs(mom), hv))
                elif trend == -1 and mom < -min_mom:
                    signals.append((symbol, -1, abs(mom), hv))
            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, _, hv in signals:
                if len(positions) >= max_pos:
                    break

                row = day_data[symbol]
                S = row['close']
                mult, mr, _, _ = get_spec(symbol)
                mpl = S * mult * mr  # margin per lot

                # === 期货部分 ===
                fut_lots = 0
                fut_margin = 0
                fut_comm_total = 0
                if opt_pct < 1.0:
                    target_n = equity * (leverage / max_pos) * (1 - opt_pct)
                    fut_lots = max(int(target_n / (S * mult)), 1)
                    fut_margin = mpl * fut_lots
                    fut_comm_total = S * mult * fut_lots * comm_fut

                    total_m = sum(p.get('fut_margin', 0) for p in positions.values()) + fut_margin
                    if total_m > equity * 0.85:
                        fut_lots = max(int((equity * 0.85 - sum(p.get('fut_margin', 0) for p in positions.values())) / mpl), 0)
                        if fut_lots <= 0 and opt_pct > 0:
                            fut_lots = 0  # 可以只做期权
                        elif fut_lots <= 0:
                            continue
                        fut_margin = mpl * fut_lots
                        fut_comm_total = S * mult * fut_lots * comm_fut

                # === 期权部分 ===
                opt_contracts = 0
                opt_cost = 0
                opt_comm_total = 0
                opt_type = 'call' if direction == 1 else 'put'
                opt_strike = S  # ATM
                if opt_pct > 0:
                    T = hold_days / 365.0
                    premium = bs_price(S, opt_strike, T, r, hv, opt_type)
                    if premium > 0:
                        cost_per = premium * mult
                        risk_amount = equity * risk_pct * max_pos  # 总风险分配
                        opt_contracts = max(int(risk_amount / cost_per), 1)
                        opt_cost = cost_per * opt_contracts
                        opt_comm_total = opt_cost * comm_opt

                        if opt_cost + opt_comm_total > cash * 0.4:
                            opt_contracts = max(int((cash * 0.4 - opt_comm_total) / cost_per), 0)
                            if opt_contracts <= 0 and opt_pct >= 1.0:
                                continue
                            elif opt_contracts <= 0:
                                opt_contracts = 0
                            opt_cost = cost_per * opt_contracts
                            opt_comm_total = opt_cost * comm_opt

                # 总资金检查
                total_needed = fut_margin + fut_comm_total + opt_cost + opt_comm_total
                if total_needed > cash:
                    # 优先保留期货，缩小期权
                    if opt_contracts > 0:
                        remaining = cash - fut_margin - fut_comm_total
                        opt_contracts = max(int((remaining * 0.9) / (opt_cost / opt_contracts + opt_comm_total / opt_contracts)), 0)
                        if opt_contracts <= 0:
                            opt_contracts = 0
                        opt_cost = cost_per * opt_contracts if opt_contracts > 0 else 0
                        opt_comm_total = opt_cost * comm_opt if opt_contracts > 0 else 0
                    total_needed = fut_margin + fut_comm_total + opt_cost + opt_comm_total
                    if total_needed > cash and fut_lots > 0:
                        fut_lots = max(int((cash - opt_cost - opt_comm_total - fut_comm_total) / mpl), 0)
                        if fut_lots <= 0 and opt_contracts <= 0:
                            continue
                        fut_margin = mpl * fut_lots
                        fut_comm_total = S * mult * fut_lots * comm_fut

                if fut_lots == 0 and opt_contracts == 0:
                    continue

                cash -= fut_margin + fut_comm_total + opt_cost + opt_comm_total

                positions[symbol] = {
                    'dir': direction,
                    'entry_date': date,
                    'fut_entry': S * (1 + 0.0001 * direction),
                    'fut_lots': fut_lots,
                    'fut_mult': mult,
                    'fut_margin': fut_margin,
                    'opt_type': opt_type,
                    'opt_strike': opt_strike,
                    'opt_contracts': opt_contracts,
                    'opt_cost': opt_cost,
                    'opt_comm': opt_comm_total,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol not in day_data:
                continue
            row = day_data[symbol]
            price = row['close']
            hv = row.get('hv_20', 0.25)
            hd = (date - pos['entry_date']).days

            # 期货未实现
            if pos.get('fut_lots', 0) > 0:
                unrealized += (price - pos['fut_entry']) * pos['dir'] * pos['fut_mult'] * pos['fut_lots']

            # 期权未实现
            if pos.get('opt_contracts', 0) > 0:
                rem_T = max((hold_days - hd) / 365.0, 0.001)
                val = bs_price(price, pos['opt_strike'], rem_T, r, hv, pos['opt_type'])
                intrinsic = max(price - pos['opt_strike'], 0) if pos['dir'] == 1 else max(pos['opt_strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                unrealized += val * pos['fut_mult'] * pos['opt_contracts']

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
    avg_w = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0
    avg_l = float(abs(pnls[pnls <= 0].mean())) if (pnls <= 0).any() else 1
    pf = avg_w * (pnls > 0).sum() / (avg_l * (pnls <= 0).sum()) if (pnls <= 0).sum() > 0 and avg_l > 0 else 0

    return {
        'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf,
        'trades': len(pnls), 'final': final, 'total_ret': float(total_ret),
        **{k: v for k, v in params.items()},
    }


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据...")
    data = load_data(data_dir)
    print(f"加载了 {len(data)} 个品种\n")

    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2026-05-08')

    results = []

    # === 扫描: 期货/期权比例 + 杠杆 + 风险 ===
    print("=== 期货+期权组合扫描 ===")
    for opt_pct in [0.0, 0.3, 0.5, 0.7, 1.0]:  # 期权占比
        for lev in [4, 5, 6]:
            for risk in [0.01, 0.02, 0.03, 0.05]:
                for hd in [5, 7]:
                    params = dict(leverage=lev, hold_days=hd, opt_pct=opt_pct,
                                 risk_pct=risk, min_mom=0.03, max_pos=3)
                    r = run_backtest(data, start_date, end_date, params)
                    if r:
                        results.append(r)

    # === 激进期权风险 ===
    print("\n=== 激进期权风险 ===")
    for opt_pct in [0.5, 0.7]:
        for risk in [0.05, 0.08, 0.10]:
            for lev in [4, 5]:
                params = dict(leverage=lev, hold_days=5, opt_pct=opt_pct,
                             risk_pct=risk, min_mom=0.03, max_pos=3)
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'期权%':>5} {'杠杆':>4} {'持有':>4} {'风险':>5} {'年化':>8} {'胜率':>6} {'盈亏比':>6} {'回撤':>8} {'交易':>4} {'最终':>14}")
    print("-" * 90)

    for r in results[:50]:
        print(f"{r.get('opt_pct',0):>5.0%} {r.get('leverage',0):>4} {r.get('hold_days',0):>4} "
              f"{r.get('risk_pct',0):>5.0%} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['trades']:>4} {r['final']:>14,.0f}")

    # 按期权比例分组
    print("\n\n=== 按期权比例分组 ===")
    for op in [0.0, 0.3, 0.5, 0.7, 1.0]:
        group = [r for r in results if abs(r.get('opt_pct', 0) - op) < 0.01]
        if group:
            best = max(group, key=lambda x: x['annual'])
            print(f"  期权{op:.0%}: 最佳年化={best['annual']:.1%}  WR={best['wr']:.1%}  "
                  f"PF={best['pf']:.2f}  MDD={best['mdd']:.1%}  "
                  f"杠杆={best.get('leverage',0)}  风险={best.get('risk_pct',0):.0%}  "
                  f"持有={best.get('hold_days',0)}")

    # 达标
    print("\n\n=== 年化>=300% ===")
    good = [r for r in results if r['annual'] >= 3.0]
    if good:
        for r in sorted(good, key=lambda x: x['wr'], reverse=True)[:10]:
            print(f"  年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  MDD={r['mdd']:.1%}  "
                  f"期权={r.get('opt_pct',0):.0%}  杠杆={r.get('leverage',0)}  "
                  f"风险={r.get('risk_pct',0):.0%}  持有={r.get('hold_days',0)}")
    else:
        print("无")

    print("\n=== WR>=50% ===")
    wr50 = [r for r in results if r['wr'] >= 0.50]
    if wr50:
        for r in sorted(wr50, key=lambda x: x['annual'], reverse=True)[:15]:
            print(f"  年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  MDD={r['mdd']:.1%}  "
                  f"期权={r.get('opt_pct',0):.0%}  杠杆={r.get('leverage',0)}  "
                  f"风险={r.get('risk_pct',0):.0%}  持有={r.get('hold_days',0)}  权益={r['final']:,.0f}")
    else:
        print("无")

    # TOP3 年度
    print("\n\n=== TOP 3 年度明细 ===")
    for i, r in enumerate(sorted(results, key=lambda x: x['annual'], reverse=True)[:3]):
        print(f"\n--- #{i+1}: 年化={r['annual']:.1%}  WR={r['wr']:.1%}  "
              f"期权={r.get('opt_pct',0):.0%}  杠杆={r.get('leverage',0)}  "
              f"风险={r.get('risk_pct',0):.0%}  持有={r.get('hold_days',0)} ---")
        # 找出最近似的纯期货和纯期权配置
        if r.get('opt_pct', 0) == 0:
            print("  (纯期货)")
        elif r.get('opt_pct', 0) == 1:
            print("  (纯期权)")


if __name__ == '__main__':
    main()
