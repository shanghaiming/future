#!/usr/bin/env python3
"""
策略 v35 — 期货 + 买期权 组合策略
约束: 不能卖期权, 只能买期权 + 做期货

之前买期权最佳结果:
  v32: 1027.7%年化, 52.8%WR, MDD=-98.4% (OTM=1%, hold=10d, risk=15%)
  v33: 824.2%年化, 39.4%WR, MDD=-47.7% (纯买期权, risk=1%)
  v26: 421.7%年化, 34.0%WR, PF=5.32, MDD=-20.8%

本版本:
  1. 期货+买期权双仓: 期货提供基础方向收益, 买期权提供凸性收益
  2. 多因子信号: 趋势+动量+ADX+OI+RSI
  3. 参数扫描: 寻找年化>=600%且WR>=50%的配置
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
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']
        if 'oi' in df.columns:
            df['oi_ma5'] = df['oi'].rolling(5).mean()
            df['oi_rising'] = df['oi'] > df['oi'].shift(5)
        else:
            df['oi_rising'] = True
        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'adx'])
        if len(df) > 100:
            data[symbol] = df
    return data


def run_backtest(data, start_date, end_date, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 5)
    otm_pct = params.get('otm_pct', 0.02)
    risk_pct = params.get('risk_pct', 0.03)
    margin_usage = params.get('margin_usage', 0.80)
    min_mom = params.get('min_mom', 0.02)
    take_profit = params.get('take_profit', 0)
    use_oi = params.get('use_oi', False)
    require_align = params.get('require_align', False)
    opt_leverage = params.get('opt_leverage', 1.0)
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
            mult = pos['mult']

            # 期货退出
            if pos.get('fut_lots', 0) > 0 and hd >= hold_days:
                fut_pnl = (price - pos['fut_entry']) * pos['dir'] * mult * pos['fut_lots']
                fut_comm = price * mult * pos['fut_lots'] * comm_fut
                cash += pos['fut_margin'] + fut_pnl - fut_comm
                closed_pnls.append(fut_pnl - fut_comm)
                pos['fut_lots'] = 0
                pos['fut_margin'] = 0

            # 期权退出
            if pos.get('opt_contracts', 0) > 0:
                rem_T = max((hold_days - hd) / 365.0, 0.001)
                val = bs_price(price, pos['opt_strike'], rem_T, r, hv, pos['opt_type'])
                intrinsic = max(price - pos['opt_strike'], 0) if pos['dir'] == 1 else max(pos['opt_strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                total_val = val * mult * pos['opt_contracts']

                should_close = False
                if hd >= hold_days:
                    should_close = True
                elif take_profit > 0 and total_val > pos['opt_cost'] * take_profit:
                    should_close = True

                if should_close:
                    opt_comm = total_val * comm_opt
                    net = total_val - pos['opt_cost'] - opt_comm - pos.get('opt_comm', 0)
                    cash += total_val - opt_comm
                    closed_pnls.append(net)
                    pos['opt_contracts'] = 0
                    pos['opt_cost'] = 0

            # 清理空仓位
            if pos.get('fut_lots', 0) == 0 and pos.get('opt_contracts', 0) == 0:
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
                mom_10 = row.get('mom_10', 0)
                adx = row.get('adx', 0)
                rsi = row.get('rsi', 50)
                hv = row.get('hv_20', 0)
                if hv < 0.10 or hv > 0.60:
                    continue

                direction = 0
                if trend == 1 and mom_10 > min_mom and rsi < 70:
                    direction = 1
                elif trend == -1 and mom_10 < -min_mom and rsi > 30:
                    direction = -1
                if direction == 0:
                    continue

                # 多时间框架对齐
                if require_align:
                    mom_5 = row.get('mom_5', 0)
                    mom_20 = row.get('mom_20', 0)
                    if direction == 1 and not (mom_5 > 0 and mom_10 > 0 and mom_20 > 0):
                        continue
                    elif direction == -1 and not (mom_5 < 0 and mom_10 < 0 and mom_20 < 0):
                        continue

                # OI过滤
                if use_oi and not row.get('oi_rising', True):
                    continue

                score = abs(mom_10) * 100 + (adx - 20) * 0.3
                signals.append((symbol, direction, score, hv))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, _, hv in signals:
                if len(positions) >= max_pos:
                    break

                row = day_data[symbol]
                S = row['close']
                mult, mr, _, _ = get_spec(symbol)
                mpl = S * mult * mr  # margin per lot

                # --- 期货部分 ---
                fut_lots = 0
                fut_margin = 0
                fut_comm_total = 0
                if margin_usage > 0:
                    target_n = equity * (margin_usage / max_pos)
                    fut_lots = max(int(target_n / (S * mult)), 1)
                    fut_margin = mpl * fut_lots
                    fut_comm_total = S * mult * fut_lots * comm_fut

                    total_m = sum(p.get('fut_margin', 0) for p in positions.values()) + fut_margin
                    if total_m > equity * margin_usage:
                        fut_lots = max(int((equity * margin_usage - sum(p.get('fut_margin', 0) for p in positions.values())) / mpl), 0)
                        fut_margin = mpl * fut_lots if fut_lots > 0 else 0
                        fut_comm_total = S * mult * fut_lots * comm_fut if fut_lots > 0 else 0

                # --- 买期权部分 ---
                opt_contracts = 0
                opt_cost = 0
                opt_comm_total = 0
                opt_type = 'call' if direction == 1 else 'put'
                opt_strike = S * (1 + otm_pct) if direction == 1 else S * (1 - otm_pct)
                T = hold_days / 365.0
                premium = bs_price(S, opt_strike, T, r, hv, opt_type)
                if premium > 0:
                    cost_per = premium * mult
                    risk_amount = equity * risk_pct * opt_leverage
                    opt_contracts = max(int(risk_amount / cost_per), 1)
                    opt_cost = cost_per * opt_contracts
                    opt_comm_total = opt_cost * comm_opt

                    if opt_cost + opt_comm_total > cash * 0.5:
                        opt_contracts = max(int((cash * 0.5) / (cost_per + cost_per * comm_opt)), 0)
                        opt_cost = cost_per * opt_contracts if opt_contracts > 0 else 0
                        opt_comm_total = opt_cost * comm_opt if opt_contracts > 0 else 0

                # 资金检查
                total_needed = fut_margin + fut_comm_total + opt_cost + opt_comm_total
                if total_needed > cash:
                    if opt_contracts > 0:
                        remaining = cash - fut_margin - fut_comm_total
                        opt_contracts = max(int(remaining * 0.9 / (cost_per + cost_per * comm_opt)), 0)
                        opt_cost = cost_per * opt_contracts if opt_contracts > 0 else 0
                        opt_comm_total = opt_cost * comm_opt if opt_contracts > 0 else 0
                    total_needed = fut_margin + fut_comm_total + opt_cost + opt_comm_total
                    if total_needed > cash and fut_lots > 0:
                        fut_lots = max(int((cash - opt_cost - opt_comm_total) / mpl), 0)
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
                    'mult': mult,
                    'fut_entry': S * (1 + 0.0001 * direction),
                    'fut_lots': fut_lots,
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

            if pos.get('fut_lots', 0) > 0:
                unrealized += (price - pos['fut_entry']) * pos['dir'] * pos['mult'] * pos['fut_lots']

            if pos.get('opt_contracts', 0) > 0:
                rem_T = max((hold_days - hd) / 365.0, 0.001)
                val = bs_price(price, pos['opt_strike'], rem_T, r, hv, pos['opt_type'])
                intrinsic = max(price - pos['opt_strike'], 0) if pos['dir'] == 1 else max(pos['opt_strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                unrealized += val * pos['mult'] * pos['opt_contracts'] - pos['opt_cost']

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

    # === 扫描1: 保证金 + OTM + 持有天数 ===
    print("=== 扫描1: 保证金 + OTM + 持有天数 ===")
    for mu in [0.70, 0.80, 0.90]:
        for otm in [0.01, 0.02, 0.03]:
            for hd in [5, 7, 10]:
                for risk in [0.02, 0.03, 0.05]:
                    params = dict(margin_usage=mu, otm_pct=otm, hold_days=hd,
                                 risk_pct=risk, min_mom=0.02, max_pos=3,
                                 take_profit=0, use_oi=False)
                    r = run_backtest(data, start_date, end_date, params)
                    if r:
                        results.append(r)

    # === 扫描2: 买期权为主(高risk + 对齐) ===
    print("\n=== 扫描2: 高风险期权 + 对齐 ===")
    for risk in [0.08, 0.10, 0.15]:
        for otm in [0.01, 0.02, 0.03]:
            for hd in [5, 7, 10]:
                for mu in [0.50, 0.60]:
                    for align in [True, False]:
                        params = dict(margin_usage=mu, otm_pct=otm, hold_days=hd,
                                     risk_pct=risk, min_mom=0.02, max_pos=3,
                                     take_profit=0, use_oi=False, require_align=align)
                        r = run_backtest(data, start_date, end_date, params)
                        if r:
                            results.append(r)

    # === 扫描3: OI过滤 + 提前止盈 ===
    print("\n=== 扫描3: OI + 止盈 ===")
    for mu in [0.70, 0.80]:
        for otm in [0.01, 0.02]:
            for hd in [5, 7, 10]:
                for tp in [2.0, 3.0]:
                    for oi in [True, False]:
                        params = dict(margin_usage=mu, otm_pct=otm, hold_days=hd,
                                     risk_pct=0.03, min_mom=0.02, max_pos=3,
                                     take_profit=tp, use_oi=oi)
                        r = run_backtest(data, start_date, end_date, params)
                        if r:
                            results.append(r)

    # === 扫描4: 纯买期权(无期货) — v32最佳参数附近密集搜索 ===
    print("\n=== 扫描4: 纯买期权(无期货) ===")
    for risk in [0.05, 0.10, 0.15]:
        for otm in [0.01, 0.02, 0.03]:
            for hd in [5, 10]:
                for tp in [0, 1.5, 2.0, 3.0]:
                    for align in [True, False]:
                        params = dict(margin_usage=0.0, otm_pct=otm, hold_days=hd,
                                     risk_pct=risk, min_mom=0.02, max_pos=3,
                                     take_profit=tp, use_oi=False, require_align=align,
                                     opt_leverage=1.5)
                        r = run_backtest(data, start_date, end_date, params)
                        if r:
                            results.append(r)

    # === 扫描5: 纯期货(无期权) ===
    print("\n=== 扫描5: 纯期货(无期权) ===")
    for mu in [0.80, 0.90, 0.95]:
        for hd in [3, 5, 7]:
            for mom in [0.02, 0.03, 0.05]:
                params = dict(margin_usage=mu, otm_pct=0.0, hold_days=hd,
                             risk_pct=0.0, min_mom=mom, max_pos=3,
                             take_profit=0, use_oi=False)
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'保证金':>6} {'OTM':>5} {'持有':>4} {'风险':>5} {'止盈':>4} {'OI':>3} {'对齐':>3} {'年化':>8} {'胜率':>6} {'PF':>6} {'MDD':>8} {'交易':>5} {'最终':>14}")
    print("-" * 110)

    for r in results[:60]:
        tp_str = f"{r.get('take_profit',0):.1f}" if r.get('take_profit', 0) > 0 else '-'
        oi_str = 'Y' if r.get('use_oi') else '-'
        al_str = 'Y' if r.get('require_align') else '-'
        print(f"{r.get('margin_usage',0):>6.0%} {r.get('otm_pct',0):>5.1%} {r.get('hold_days',0):>4} "
              f"{r.get('risk_pct',0):>5.0%} {tp_str:>4} {oi_str:>3} {al_str:>3} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['trades']:>5} {r['final']:>14,.0f}")

    # === 筛选 ===
    print(f"\n\n=== 年化>=600% ===")
    good = [r for r in results if r['annual'] >= 6.0]
    if good:
        for r in sorted(good, key=lambda x: x['wr'], reverse=True)[:15]:
            print(f"  年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  MDD={r['mdd']:.1%}  "
                  f"交易={r['trades']}  保证金={r.get('margin_usage',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
                  f"风险={r.get('risk_pct',0):.0%}  持有={r.get('hold_days',0)}  "
                  f"止盈={r.get('take_profit',0)}  OI={r.get('use_oi',False)}")
    else:
        print("无")

    print(f"\n\n=== WR>=50% ===")
    wr50 = [r for r in results if r['wr'] >= 0.50]
    if wr50:
        for r in sorted(wr50, key=lambda x: x['annual'], reverse=True)[:15]:
            print(f"  年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  MDD={r['mdd']:.1%}  "
                  f"交易={r['trades']}  保证金={r.get('margin_usage',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
                  f"风险={r.get('risk_pct',0):.0%}  持有={r.get('hold_days',0)}  "
                  f"止盈={r.get('take_profit',0)}  OI={r.get('use_oi',False)}")
    else:
        print("无")

    # 同时满足
    print(f"\n\n=== 同时满足: 年化>=600% AND WR>=50% ===")
    both = [r for r in results if r['annual'] >= 6.0 and r['wr'] >= 0.50]
    if both:
        for r in sorted(both, key=lambda x: x['annual'], reverse=True)[:10]:
            print(f"  ★ 年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  MDD={r['mdd']:.1%}  "
                  f"交易={r['trades']}  保证金={r.get('margin_usage',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
                  f"风险={r.get('risk_pct',0):.0%}  持有={r.get('hold_days',0)}  "
                  f"止盈={r.get('take_profit',0)}  OI={r.get('use_oi',False)}")
    else:
        print("无 — 需要进一步优化")

    # 按类型分组
    print(f"\n\n=== 按策略类型分组 ===")
    for label, cond in [("纯期货", lambda r: r.get('margin_usage',0) > 0 and r.get('otm_pct',0) == 0),
                         ("期货+期权", lambda r: r.get('margin_usage',0) > 0 and r.get('otm_pct',0) > 0),
                         ("纯买期权", lambda r: r.get('margin_usage',0) == 0)]:
        group = [r for r in results if cond(r)]
        if group:
            best = max(group, key=lambda x: x['annual'])
            best_wr = max(group, key=lambda x: x['wr'])
            print(f"\n  {label}:")
            print(f"    最高年化: {best['annual']:.1%}  WR={best['wr']:.1%}  MDD={best['mdd']:.1%}")
            print(f"    最高WR:   {best_wr['wr']:.1%}  年化={best_wr['annual']:.1%}  MDD={best_wr['mdd']:.1%}")


if __name__ == '__main__':
    main()
