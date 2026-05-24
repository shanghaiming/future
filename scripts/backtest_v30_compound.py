#!/usr/bin/env python3
"""
策略 v30 — 高频轮动卖期权 + 复利爆炸
核心发现:
  - v27: 卖OTM 2%期权 7天 = 77.5% WR, 81.6%年化
  - 关键: 卖期权是正期望策略, 但受限于仓位大小
  - 如果每笔只收1-2%权利金, 需要大量复利

突破点:
  1. 缩短持有到3-5天 → 更多交易次数 → 更多权利金收入
  2. 更激进保证金使用 → 更多合约
  3. OTM距离优化 → 平衡WR和权利金收入
  4. 复利加速: 权利金立刻投入下一笔交易

数学:
  - 每笔收权利金约1.5% notional
  - 每年150-200笔交易
  - 77%WR, 平均赢1.5%, 平均输5%
  - 期望收益 = 0.77*1.5% - 0.23*5% = 1.155% - 1.15% ≈ 0.005% per trade
  - 需要提高赢的收益或降低输的损失

  → 加入止损: 输的时候亏3%而不是5% → 期望 = 0.77*1.5% - 0.23*3% = 0.465% per trade
  → 每笔notional = equity * 0.3 → 每笔收益 = 0.465% * 0.3 = 0.14% of equity
  → 150笔 → 21% per year... 不够

  → 保证金杠杆: notional = equity * 1.0 → 0.465% per trade * 150 = 69.75% per year
  → 不够

  → 更激进: 3个仓位, 每个equity*0.5 notional → 1.5x leverage on premium
  → 每笔收益 = 0.465% * 0.5 = 0.23%, 450 trades → 103.5%

  → 要到600%: 需要 Kelly optimal sizing + 更多交易
  → 或者: 纯卖期权 + 高频 + 超大仓位

新方法: 双重杠杆卖期权
  - 卖出期权收取权利金
  - 权利金立即用于下一笔交易的保证金
  - 3个仓位同时运作, 每个用满保证金
  - 3-5天轮转 → 年化200+交易
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
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss_s))
        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10'])
        if len(df) > 100:
            data[symbol] = df
    return data


def run_backtest(data, start_date, end_date, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 5)
    otm_pct = params.get('otm_pct', 0.02)
    margin_usage = params.get('margin_usage', 0.7)  # 保证金使用率
    min_mom = params.get('min_mom', 0.02)
    stop_at_strike = params.get('stop_at_strike', False)
    early_take = params.get('early_take', 0)  # 0=不提前止盈, 否则=收足x%权利金就平
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

        # === 退出 ===
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data:
                continue
            row = day_data[symbol]
            price = row['close']
            hv = row.get('hv_20', 0.25)
            hd = (date - pos['entry_date']).days

            rem_T = max((hold_days - hd) / 365.0, 0.001)
            buyback = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
            intrinsic = max(price - pos['strike'], 0) if pos['otype'] == 'call' else max(pos['strike'] - price, 0)
            buyback = max(buyback, intrinsic * 0.95)
            total_buyback = buyback * pos['mult'] * pos['contracts']
            unrealized_net = pos['credit'] - total_buyback

            should_close = False
            # 时间退出
            if hd >= hold_days:
                should_close = True
            # 止损: 价格穿过strike
            elif stop_at_strike:
                if pos['dir'] == 1 and price < pos['strike']:
                    should_close = True
                elif pos['dir'] == -1 and price > pos['strike']:
                    should_close = True
            # 止损: 亏损>2倍权利金
            elif unrealized_net < -2.0 * pos['credit']:
                should_close = True
            # 提前止盈: 已收50%+权利金
            elif early_take > 0 and hd >= 2 and unrealized_net > early_take * pos['credit']:
                should_close = True

            if should_close:
                comm = total_buyback * comm_rate
                net = unrealized_net - comm - pos['comm']
                cash += pos['margin'] + net
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
                rsi = row.get('rsi', 50)

                if trend == 1 and mom > min_mom and rsi < 70:
                    signals.append((symbol, 1, abs(mom), hv))
                elif trend == -1 and mom < -min_mom and rsi > 30:
                    signals.append((symbol, -1, abs(mom), hv))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, _, hv in signals:
                if len(positions) >= max_pos:
                    break

                row = day_data[symbol]
                S = row['close']
                mult, mr, _, _ = get_spec(symbol)

                if direction == 1:
                    K = S * (1 - otm_pct)
                    otype = 'put'
                else:
                    K = S * (1 + otm_pct)
                    otype = 'call'

                T = hold_days / 365.0
                premium = bs_price(S, K, T, r, hv, otype)
                if premium <= 0:
                    continue

                credit_per = premium * mult
                margin_per = S * mult * mr

                # 激进仓位: 用满可用保证金的margin_usage比例
                total_existing_m = sum(p['margin'] for p in positions.values())
                available_margin = equity * margin_usage - total_existing_m
                if available_margin <= margin_per:
                    continue

                contracts = max(int(available_margin / margin_per), 1)
                total_margin = margin_per * contracts
                total_credit = credit_per * contracts
                comm = total_credit * comm_rate

                if total_margin - total_credit + comm > cash:
                    contracts = max(int((cash + total_credit - comm) / margin_per), 0)
                    if contracts <= 0:
                        continue
                    total_margin = margin_per * contracts
                    total_credit = credit_per * contracts
                    comm = total_credit * comm_rate

                cash -= total_margin - total_credit + comm
                positions[symbol] = {
                    'type': 'sell_option', 'dir': direction, 'otype': otype,
                    'entry_price': S, 'strike': K, 'entry_date': date,
                    'contracts': contracts, 'mult': mult, 'margin': total_margin,
                    'credit': total_credit, 'comm': comm, 'hold_days': hold_days,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                row = day_data[symbol]
                price = row['close']
                hv = row.get('hv_20', 0.25)
                hd = (date - pos['entry_date']).days
                rem_T = max((pos['hold_days'] - hd) / 365.0, 0.001)
                buyback = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['otype'] == 'call' else max(pos['strike'] - price, 0)
                buyback = max(buyback, intrinsic * 0.95)
                unrealized += pos['credit'] - buyback * pos['mult'] * pos['contracts']

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

    # === 扫描1: 高保证金使用率 + OTM ===
    print("=== 扫描: 保证金使用 + OTM + 持有 ===")
    for mu in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for otm in [0.015, 0.02, 0.025, 0.03]:
            for hd in [3, 5, 7]:
                params = {
                    'margin_usage': mu, 'otm_pct': otm, 'hold_days': hd,
                    'min_mom': 0.02, 'max_pos': 3,
                    'stop_at_strike': False, 'early_take': 0,
                }
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)

    # === 扫描2: 止损策略 ===
    print("\n=== 扫描: 止损策略 ===")
    for mu in [0.7, 0.8, 0.9]:
        for otm in [0.02, 0.025]:
            for hd in [3, 5]:
                # strike止损
                params = dict(margin_usage=mu, otm_pct=otm, hold_days=hd,
                             min_mom=0.02, max_pos=3, stop_at_strike=True, early_take=0)
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)
                # 提前止盈
                for et in [0.4, 0.6, 0.8]:
                    params = dict(margin_usage=mu, otm_pct=otm, hold_days=hd,
                                 min_mom=0.02, max_pos=3, stop_at_strike=False, early_take=et)
                    r = run_backtest(data, start_date, end_date, params)
                    if r:
                        results.append(r)

    # === 扫描3: 高动量门槛 + 高保证金 ===
    print("\n=== 扫描: 高门槛 + 高保证金 ===")
    for mom in [0.03, 0.05, 0.08]:
        for mu in [0.8, 0.9, 1.0]:
            for otm in [0.02, 0.03]:
                for hd in [3, 5]:
                    params = dict(margin_usage=mu, otm_pct=otm, hold_days=hd,
                                 min_mom=mom, max_pos=3, stop_at_strike=False, early_take=0)
                    r = run_backtest(data, start_date, end_date, params)
                    if r:
                        results.append(r)

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'保证金':>6} {'OTM':>5} {'持有':>4} {'止损':>6} {'止盈':>4} {'年化':>8} {'胜率':>6} {'盈亏比':>6} {'回撤':>8} {'交易':>4} {'最终':>14}")
    print("-" * 100)

    for r in results[:50]:
        stop_str = 'strike' if r.get('stop_at_strike') else '-'
        et_str = f"{r.get('early_take',0):.1f}" if r.get('early_take', 0) > 0 else '-'
        print(f"{r.get('margin_usage',0):>6.0%} {r.get('otm_pct',0):>5.1%} {r.get('hold_days',0):>4} "
              f"{stop_str:>6} {et_str:>4} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['trades']:>4} {r['final']:>14,.0f}")

    # 筛选
    for target_ann, label in [(6.0, "600%"), (3.0, "300%"), (1.0, "100%")]:
        print(f"\n\n=== 年化>={label} ===")
        good = [r for r in results if r['annual'] >= target_ann]
        if good:
            for r in sorted(good, key=lambda x: x['wr'], reverse=True)[:10]:
                print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  回撤={r['mdd']:.1%}  "
                      f"盈亏比={r['pf']:.2f}  交易={r['trades']}  "
                      f"保证金={r.get('margin_usage',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
                      f"持有={r.get('hold_days',0)}")
        else:
            print("无")

    # WR>=50%
    print("\n\n=== WR>=50% 按年化排序 ===")
    wr50 = [r for r in results if r['wr'] >= 0.50]
    if wr50:
        for r in sorted(wr50, key=lambda x: x['annual'], reverse=True)[:15]:
            print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  回撤={r['mdd']:.1%}  "
                  f"盈亏比={r['pf']:.2f}  交易={r['trades']}  "
                  f"保证金={r.get('margin_usage',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
                  f"持有={r.get('hold_days',0)}  权益={r['final']:,.0f}")
    else:
        print("无")


if __name__ == '__main__':
    main()
