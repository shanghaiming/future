#!/usr/bin/env python3
"""
期权策略参数精调 — 从421%推到600%+
问题: ATM期权胜率39.4% (权利金太高)
方案: OTM期权 + 长持有 + 强信号 + 高风险
"""
import os, sys, time, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

def bs_price(S, K, T, r, sigma, opt='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt == 'call' else max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def load_data(data_dir):
    data = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'): continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 100: continue
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
        if len(df) > 100: data[symbol] = df
    return data


def run_backtest(data, start_date, end_date,
                 hold_days=5, risk_pct=0.02, min_mom=0.03,
                 otm_pct=0.0, hv_mult=1.0):
    """参数可调的期权回测"""
    max_positions = 3
    r = 0.02
    comm_rate = 0.0003

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

        # 退出
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data: continue
            row = day_data[symbol]
            price = row['close']
            hv = row.get('hv_20', 0.25)
            hd = (date - pos['entry_date']).days

            if hd >= hold_days:
                rem_T = max(0.001/365, 0.001)
                ev = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['dir'] == 1 else max(pos['strike'] - price, 0)
                ev = max(ev, intrinsic * 0.9)
                total_ev = ev * pos['mult'] * pos['contracts']
                comm = total_ev * comm_rate
                net = total_ev - pos['cost'] - comm - pos['comm']
                cash += total_ev - comm
                closed_pnls.append(net)
                del positions[symbol]

        # 入场
        if len(positions) < max_positions:
            signals = []
            for symbol, row in day_data.items():
                if symbol in positions: continue
                if row.get('atr_pct', 0.1) > 0.045: continue
                mom = row.get('mom_10', 0)
                trend = row.get('trend', 0)
                hv = row.get('hv_20', 0)
                if hv < 0.10 or hv > 0.60: continue
                if trend == 1 and mom > min_mom:
                    signals.append((symbol, 1, abs(mom), hv))
                elif trend == -1 and mom < -min_mom:
                    signals.append((symbol, -1, abs(mom), hv))
            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, _, hv in signals:
                if len(positions) >= max_positions: break
                row = day_data[symbol]
                S = row['close']
                mult, _, _, _ = get_spec(symbol)
                K = S * (1 + otm_pct * direction)  # OTM
                T = hold_days / 365.0
                sigma = hv * hv_mult
                otype = 'call' if direction == 1 else 'put'
                premium = bs_price(S, K, T, r, sigma, otype)
                if premium <= 0: continue

                cost_per = premium * mult
                risk_amount = equity * risk_pct
                contracts = max(int(risk_amount / cost_per), 1)
                total_cost = cost_per * contracts
                comm = total_cost * comm_rate

                if total_cost + comm > cash * 0.5:
                    contracts = max(int((cash * 0.5 - comm) / cost_per), 0)
                    if contracts <= 0: continue
                    total_cost = cost_per * contracts
                    comm = total_cost * comm_rate

                cash -= total_cost + comm
                positions[symbol] = {
                    'dir': direction, 'otype': otype,
                    'entry_price': S, 'strike': K,
                    'entry_date': date, 'contracts': contracts,
                    'mult': mult, 'cost': total_cost, 'comm': comm,
                }

        # 权益
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                row = day_data[symbol]
                price = row['close']
                hv = row.get('hv_20', 0.25)
                rem_T = max((hold_days - (date - pos['entry_date']).days) / 365.0, 0.001)
                val = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['dir'] == 1 else max(pos['strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                unrealized += val * pos['mult'] * pos['contracts']
        equity = cash + unrealized
        equity_curve.append((date, equity))
        if equity < 5000: break

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
        'hold_days': hold_days, 'risk_pct': risk_pct,
        'min_mom': min_mom, 'otm_pct': otm_pct, 'hv_mult': hv_mult,
        'annual': ann, 'wr': wr, 'mdd': mdd,
        'trades': len(pnls), 'final': final,
    }


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据...")
    data = load_data(data_dir)
    print(f"加载了 {len(data)} 个品种\n")

    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2026-05-08')

    results = []

    # 扫描参数
    configs = []
    for hd in [5, 7, 10]:
        for rp in [0.02, 0.03, 0.04]:
            for mm in [0.02, 0.03, 0.05]:
                for otm in [0.0, 0.02, 0.03]:
                    configs.append((hd, rp, mm, otm, 1.0))

    print(f"测试 {len(configs)} 个组合...")
    for hd, rp, mm, otm, hvm in configs:
        r = run_backtest(data, start_date, end_date, hd, rp, mm, otm, hvm)
        if r:
            results.append(r)

    print(f"\n{'持有':>4} {'风险':>5} {'动量':>5} {'OTM':>5} {'年化':>8} {'胜率':>6} {'回撤':>8} {'交易':>4} {'最终':>14}")
    print("-" * 75)

    results.sort(key=lambda x: x['annual'], reverse=True)
    for r in results[:30]:
        print(f"{r['hold_days']:>4} {r['risk_pct']:>5.0%} {r['min_mom']:>5.2f} "
              f"{r['otm_pct']:>5.0%} {r['annual']:>8.1%} {r['wr']:>6.1%} "
              f"{r['mdd']:>8.1%} {r['trades']:>4} {r['final']:>14,.0f}")

    # 筛选达标
    print(f"\n\n=== 胜率>=50% 的组合 ===")
    good = [r for r in results if r['wr'] >= 0.50]
    if good:
        good.sort(key=lambda x: x['annual'], reverse=True)
        for r in good[:15]:
            print(f"持有={r['hold_days']} 风险={r['risk_pct']:.0%} 动量>{r['min_mom']:.2f} "
                  f"OTM={r['otm_pct']:.0%} 年化={r['annual']:.1%} 胜率={r['wr']:.1%} "
                  f"回撤={r['mdd']:.1%} 权益={r['final']:,.0f}")
    else:
        print("无满足条件的组合")

    print(f"\n=== 年化>500% 的组合 ===")
    great = [r for r in results if r['annual'] > 5.0]
    if great:
        for r in sorted(great, key=lambda x: x['annual'], reverse=True)[:10]:
            print(f"持有={r['hold_days']} 风险={r['risk_pct']:.0%} 动量>{r['min_mom']:.2f} "
                  f"OTM={r['otm_pct']:.0%} 年化={r['annual']:.1%} 胜率={r['wr']:.1%} "
                  f"回撤={r['mdd']:.1%}")
    else:
        print("无满足条件的组合")


if __name__ == '__main__':
    main()
