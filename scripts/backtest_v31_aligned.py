#!/usr/bin/env python3
"""
策略 v31 — 多时间框架对齐 + OTM期权
核心洞察:
  v26买ATM期权: 421%年化, 39%WR, PF=3.0
  问题: WR太低, 需要提升到50%

方法:
1. 多时间框架对齐: 5d/10d/20d动量必须同向 → 过滤掉假信号
2. OTM期权: 买2-5%OTM → 更便宜, 盈亏比更高
3. 长持有: 7-10天 → 更多时间兑现
4. 多因子确认: ADX+RSI+量能+OI → 高质量信号
5. 动态仓位: Kelly比例基于信号强度

数学: 如果WR能从39%提到50%, PF保持2.5+
  → Kelly = 0.5 - 0.5/2.5 = 0.3 = 30% per trade
  → 3 positions at 10% each = 30% equity as premium
  → 每笔赢: 2.5 * premium, 每笔输: premium
  → Expected per trade: 0.5*2.5*p - 0.5*p = 0.75p
  → With p=10% equity: 7.5% per trade
  → 100 trades/year: 750% per year (theoretical)
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
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)

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

        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        df['mom_20'] = df['close'].pct_change(20)
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

        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']

        if 'oi' in df.columns:
            df['oi_change'] = df['oi'].pct_change(5)
        else:
            df['oi_change'] = 0

        # HV percentile
        df['hv_rank'] = df['hv_20'].rolling(60).rank(pct=True)

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'adx', 'hv_rank'])
        if len(df) > 100:
            data[symbol] = df
    return data


def run_backtest(data, start_date, end_date, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 7)
    otm_pct = params.get('otm_pct', 0.0)  # 0=ATM
    risk_pct = params.get('risk_pct', 0.02)
    min_mom = params.get('min_mom', 0.02)
    require_align = params.get('require_align', True)  # 多时间框架对齐
    min_adx = params.get('min_adx', 0)
    require_vol = params.get('require_vol', False)
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
    signal_counts = defaultdict(int)

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

            if hd >= hold_days:
                rem_T = max(0.001 / 365, 0.001)
                val = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['dir'] == 1 else max(pos['strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                total_val = val * pos['mult'] * pos['contracts']
                comm = total_val * comm_rate
                net = total_val - pos['cost'] - comm - pos['comm']
                cash += total_val - comm
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

                trend = row.get('trend', 0)
                mom_5 = row.get('mom_5', 0)
                mom_10 = row.get('mom_10', 0)
                mom_20 = row.get('mom_20', 0)
                adx = row.get('adx', 0)
                rsi = row.get('rsi', 50)
                hv = row.get('hv_20', 0)
                vol_ratio = row.get('vol_ratio', 1.0)
                oi_change = row.get('oi_change', 0)

                if hv < 0.10 or hv > 0.60:
                    continue

                direction = 0

                if trend == 1 and mom_10 > min_mom:
                    direction = 1
                elif trend == -1 and mom_10 < -min_mom:
                    direction = -1

                if direction == 0:
                    continue

                # 多时间框架对齐
                if require_align:
                    if direction == 1 and not (mom_5 > 0 and mom_10 > 0 and mom_20 > 0):
                        continue
                    elif direction == -1 and not (mom_5 < 0 and mom_10 < 0 and mom_20 < 0):
                        continue

                # ADX过滤
                if min_adx > 0 and adx < min_adx:
                    continue

                # 量能过滤
                if require_vol and vol_ratio < 1.0:
                    continue

                # 综合评分
                score = 0
                score += min(abs(mom_10) * 200, 25)
                if adx > 30:
                    score += 15
                elif adx > 20:
                    score += 8
                if direction == 1 and 30 <= rsi <= 50:
                    score += 15
                elif direction == -1 and 50 <= rsi <= 70:
                    score += 15
                if vol_ratio > 1.5:
                    score += 10
                elif vol_ratio > 1.2:
                    score += 5
                if abs(oi_change) > 0.03:
                    score += 5

                signal_counts['total'] += 1
                signals.append((symbol, direction, score, hv))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv in signals:
                if len(positions) >= max_pos:
                    break

                row = day_data[symbol]
                S = row['close']
                mult, _, _, _ = get_spec(symbol)

                # OTM期权
                if direction == 1:
                    K = S * (1 + otm_pct)  # 买OTM call (strike更高)
                    otype = 'call'
                else:
                    K = S * (1 - otm_pct)  # 买OTM put (strike更低)
                    otype = 'put'

                T = hold_days / 365.0
                premium = bs_price(S, K, T, r, hv, otype)
                if premium <= 0:
                    continue

                cost_per = premium * mult
                risk_amount = equity * risk_pct
                contracts = max(int(risk_amount / cost_per), 1)
                total_cost = cost_per * contracts
                comm = total_cost * comm_rate

                if total_cost + comm > cash * 0.5:
                    contracts = max(int((cash * 0.5 - comm) / cost_per), 0)
                    if contracts <= 0:
                        continue
                    total_cost = cost_per * contracts
                    comm = total_cost * comm_rate

                cash -= total_cost + comm
                positions[symbol] = {
                    'dir': direction, 'otype': otype,
                    'entry_price': S, 'strike': K,
                    'entry_date': date, 'contracts': contracts,
                    'mult': mult, 'cost': total_cost, 'comm': comm,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                row = day_data[symbol]
                price = row['close']
                hv = row.get('hv_20', 0.25)
                hd = (date - pos['entry_date']).days
                rem_T = max((hold_days - hd) / 365.0, 0.001)
                val = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['dir'] == 1 else max(pos['strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                unrealized += val * pos['mult'] * pos['contracts']

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
        'signals': dict(signal_counts),
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

    # === 1. 多时间框架对齐 + ATM (baseline) ===
    print("=== 多时间框架对齐 + ATM ===")
    for hd in [5, 7, 10]:
        for risk in [0.02, 0.03, 0.05]:
            params = dict(hold_days=hd, risk_pct=risk, otm_pct=0.0,
                         require_align=True, min_mom=0.02, min_adx=0,
                         require_vol=False, max_pos=3)
            r = run_backtest(data, start_date, end_date, params)
            if r:
                results.append(r)
                print(f"  hd={hd} risk={risk:.0%} → ann={r['annual']:.1%} wr={r['wr']:.1%} pf={r['pf']:.2f} trades={r['trades']}")

    # === 2. 对齐 + OTM ===
    print("\n=== 多时间框架对齐 + OTM ===")
    for otm in [0.02, 0.03, 0.05]:
        for hd in [7, 10]:
            for risk in [0.02, 0.03, 0.05]:
                params = dict(hold_days=hd, risk_pct=risk, otm_pct=otm,
                             require_align=True, min_mom=0.02, min_adx=0,
                             require_vol=False, max_pos=3)
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)
                    print(f"  otm={otm:.0%} hd={hd} risk={risk:.0%} → ann={r['annual']:.1%} wr={r['wr']:.1%} pf={r['pf']:.2f} trades={r['trades']}")

    # === 3. 对齐 + ADX + Volume ===
    print("\n=== 多时间框架对齐 + ADX + Volume ===")
    for min_adx in [20, 25, 30]:
        for hd in [7, 10]:
            params = dict(hold_days=hd, risk_pct=0.03, otm_pct=0.0,
                         require_align=True, min_mom=0.02, min_adx=min_adx,
                         require_vol=True, max_pos=3)
            r = run_backtest(data, start_date, end_date, params)
            if r:
                results.append(r)
                print(f"  adx>{min_adx} hd={hd} → ann={r['annual']:.1%} wr={r['wr']:.1%} pf={r['pf']:.2f} trades={r['trades']}")

    # === 4. 不对齐 (对照组) ===
    print("\n=== 不对齐 (对照) ===")
    for hd in [5, 7]:
        for risk in [0.02, 0.03]:
            params = dict(hold_days=hd, risk_pct=risk, otm_pct=0.0,
                         require_align=False, min_mom=0.03, min_adx=0,
                         require_vol=False, max_pos=3)
            r = run_backtest(data, start_date, end_date, params)
            if r:
                results.append(r)
                print(f"  hd={hd} risk={risk:.0%} → ann={r['annual']:.1%} wr={r['wr']:.1%} pf={r['pf']:.2f} trades={r['trades']}")

    # === 5. 高动量 + 对齐 + OTM ===
    print("\n=== 高动量门槛 + 对齐 ===")
    for mom in [0.03, 0.05]:
        for otm in [0.0, 0.02, 0.03]:
            for hd in [7, 10]:
                params = dict(hold_days=hd, risk_pct=0.03, otm_pct=otm,
                             require_align=True, min_mom=mom, min_adx=0,
                             require_vol=False, max_pos=3)
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'OTM':>5} {'持有':>4} {'风险':>5} {'对齐':>4} {'ADX':>4} {'量能':>4} {'年化':>8} {'胜率':>6} {'盈亏比':>6} {'回撤':>8} {'交易':>4} {'最终':>14}")
    print("-" * 100)

    for r in results[:40]:
        align = 'Y' if r.get('require_align') else 'N'
        adx_str = f">{r.get('min_adx',0)}" if r.get('min_adx', 0) > 0 else '-'
        vol_str = 'Y' if r.get('require_vol') else '-'
        print(f"{r.get('otm_pct',0):>5.0%} {r.get('hold_days',0):>4} {r.get('risk_pct',0):>5.0%} "
              f"{align:>4} {adx_str:>4} {vol_str:>4} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['trades']:>4} {r['final']:>14,.0f}")

    # 对比分析
    print("\n\n=== 对齐 vs 不对齐 ===")
    aligned = [r for r in results if r.get('require_align')]
    not_aligned = [r for r in results if not r.get('require_align')]
    if aligned:
        best_a = max(aligned, key=lambda x: x['annual'])
        print(f"  对齐: 年化={best_a['annual']:.1%}  胜率={best_a['wr']:.1%}  盈亏比={best_a['pf']:.2f}  交易={best_a['trades']}")
    if not_aligned:
        best_na = max(not_aligned, key=lambda x: x['annual'])
        print(f"  不对齐: 年化={best_na['annual']:.1%}  胜率={best_na['wr']:.1%}  盈亏比={best_na['pf']:.2f}  交易={best_na['trades']}")

    print("\n=== WR>=45% 按年化排序 ===")
    wr45 = [r for r in results if r['wr'] >= 0.45]
    if wr45:
        for r in sorted(wr45, key=lambda x: x['annual'], reverse=True)[:10]:
            align = 'Y' if r.get('require_align') else 'N'
            print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  盈亏比={r['pf']:.2f}  回撤={r['mdd']:.1%}  "
                  f"交易={r['trades']}  OTM={r.get('otm_pct',0):.0%}  持有={r.get('hold_days',0)}  对齐={align}")

    print("\n=== WR>=50% 按年化排序 ===")
    wr50 = [r for r in results if r['wr'] >= 0.50]
    if wr50:
        for r in sorted(wr50, key=lambda x: x['annual'], reverse=True)[:10]:
            align = 'Y' if r.get('require_align') else 'N'
            print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  盈亏比={r['pf']:.2f}  回撤={r['mdd']:.1%}  "
                  f"交易={r['trades']}  OTM={r.get('otm_pct',0):.0%}  持有={r.get('hold_days',0)}  对齐={align}")
    else:
        print("无")


if __name__ == '__main__':
    main()
