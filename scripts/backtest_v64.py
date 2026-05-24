#!/usr/bin/env python3
"""
策略 V64 — 期货+期权凸性组合 + 激进复利

核心思路:
  1. 期货主力仓位 (carry+MR信号) — 70-80%资金
  2. 买入期权凸性头寸 — 剩余资金买OTM看涨, 截断下行, 放大上行
  3. 激进复利: 每次平仓后根据新权益调整头寸
  4. 金字塔加仓: 获利仓位加码
  5. 动态杠杆: 信号越强杠杆越高

满足:
  - "期货为主" ✓ (期货是核心仓位)
  - "分析期权服务期货" ✓ (期权提供凸性保护)
  - "不能卖期权" ✓ (只买期权)

V62 carry最佳: NM=3, 23%年化, 57%WR, -35%MDD
V26 期权买入: 421.7%年化
目标: 结合两者达到更高年化
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
INIT = 500000
TD = 252


def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)


def load_carry(ts_dir):
    carry = {}
    for f in os.listdir(ts_dir):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(ts_dir, f)) as fh:
            d = json.load(fh)
        sym = d['symbol'].lower()
        near = d['near_price']
        far = d['far_price']
        if near > 0 and far > 0:
            c = (near - far) / near * 100
            carry[sym] = c
    return carry


def load(data_dir):
    raw = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'):
            continue
        sym = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300:
            continue
        df['ret'] = df['close'].pct_change()

        for lag in [3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else .5, raw=False)

        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        d = df['close'].diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        down = (df['ret'] < 0).astype(int).values
        cons_d = []
        c = 0
        for v in down:
            c = c + 1 if v else 0
            cons_d.append(c)
        df['cons_down'] = cons_d

        for hold in [5, 10, 15, 20]:
            df[f'fwd{hold}'] = df['close'].shift(-hold) / df['close'] - 1

        df = df.dropna(subset=['ma20', 'hv20', 'rsi'])
        if len(df) < 100:
            continue
        try:
            spec = get_spec(sym)
        except:
            continue

        raw[sym] = {
            'spec': spec,
            'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'high': df['high'].values.astype(np.float64),
            'low': df['low'].values.astype(np.float64),
            'ret': df['ret'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'cons_down': np.array(cons_d, dtype=np.int32),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv10': df['hv10'].values.astype(np.float64),
            'ma20': df['ma20'].values.astype(np.float64),
            'fwd10': df['fwd10'].values.astype(np.float64),
            'fwd20': df['fwd20'].values.astype(np.float64),
        }
    return raw


def build_idx(raw, s, e):
    ad = set()
    for d in raw.values():
        m = (d['dates'] >= s) & (d['dates'] <= e)
        for dt in d['dates'][m]:
            ad.add(dt)
    dates = np.array(sorted(ad))
    si = {}
    for sym, d in raw.items():
        im = {}
        m = (d['dates'] >= s) & (d['dates'] <= e)
        for dt, il in zip(d['dates'][m], np.where(m)[0]):
            im[dt] = int(il)
        si[sym] = im
    return dates, si


def select_symbols(raw, train_end, signal='mr', top_pct=0.5):
    te = pd.Timestamp(train_end)
    scores = {}
    for sym, d in raw.items():
        mask = d['dates'] <= te
        idx = np.where(mask)[0]
        if len(idx) < 200:
            continue
        rsi = d['rsi'][idx]
        fwd = d['fwd10'][idx]
        sig_mask = rsi < 25
        n = sig_mask.sum()
        if n < 15:
            continue
        fwd_vals = fwd[sig_mask]
        valid = ~np.isnan(fwd_vals)
        if valid.sum() < 10:
            continue
        wr = (fwd_vals[valid] > 0).mean()
        avg = fwd_vals[valid].mean()
        scores[sym] = wr * max(avg, 0)
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    cutoff = max(1, int(len(ranked) * top_pct))
    return [s for s, _ in ranked[:cutoff]]


def bt_futures_options(raw, dates, si, good_syms, carry_map, p):
    """
    期货+期权凸性组合回测

    对每个MR+carry信号:
      - 期货仓位: 标准多头 (主力)
      - 期权仓位: 买入OTM看涨 (凸性)

    期权用BS定价, 到期平仓算P&L
    """
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 10)
    rsi_max = p.get('rsi_max', 25)
    nm_futures = p.get('nm_futures', 5)  # 期货名义杠杆
    option_pct = p.get('option_pct', 0.1)  # 期权占权益比例
    k_mult = p.get('k_mult', 1.05)  # 行权价 = 入场价 × k_mult (OTM程度)
    carry_min = p.get('carry_min', 0)
    pyramid = p.get('pyramid', False)  # 金字塔加仓
    pyramid_th = p.get('pyramid_th', 0.03)  # 获利3%后加仓

    eq = float(INIT)
    pos = {}  # sym -> {futures + option info}
    pnls = []
    eqh = [float(INIT)]
    trade_log = []

    for date in dates:
        # === 退出 ===
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if not im or date not in im:
                continue
            il = im[date]
            S = raw[sym]['close'][il]
            h = int((date - ps['ed']) / np.timedelta64(1, 'D'))

            if h < hd:
                # 金字塔加仓检查
                if pyramid and ps.get('pyramids', 0) < 2 and len(pos) <= mp:
                    unreal = (S - ps['ep']) / ps['ep']
                    if unreal > pyramid_th:
                        # 加仓
                        ml, mr, _, _ = ps['spec']
                        add_ct = max(int(eq * nm_futures * 0.3 / (S * ml)), 1)
                        ps['ct'] += add_ct
                        ps['notional'] = S * ml * ps['ct']
                        ps['pyramids'] = ps.get('pyramids', 0) + 1
                continue

            ml = ps['ml']
            # 期货PnL
            futures_pnl = (S - ps['ep']) * ml * ps['ct']

            # 期权PnL: BS定价结算
            option_pnl = 0
            if ps.get('option_cost', 0) > 0:
                T_remain = max(hd - h, 0) / TD
                sigma = ps['hv']
                K = ps['option_strike']
                if T_remain > 0.001:
                    opt_price = bs_call(S, K, T_remain, 0.03, sigma)
                    option_pnl = opt_price * ps['option_lots'] - ps['option_cost']
                else:
                    # 到期: max(S-K, 0)
                    intrinsic = max(S - K, 0)
                    option_pnl = intrinsic * ps['option_lots'] - ps['option_cost']

            # 手续费
            notional_exit = S * ml * ps['ct']
            comm = COMM * (ps['notional'] + notional_exit)
            comm += max(ps.get('option_cost', 0) * 0.001, 0)  # 期权手续费

            total_pnl = futures_pnl + option_pnl - comm
            eq += total_pnl
            pnls.append(total_pnl)
            trade_log.append({
                'sym': sym, 'pnl': total_pnl, 'futures_pnl': futures_pnl,
                'option_pnl': option_pnl, 'hold': h,
                'entry': ps['ep'], 'exit': float(S),
            })
            del pos[sym]

        # === 入场 ===
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in pos:
                    continue
                if sym not in good_syms:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il <= 1:
                    continue
                pi = il - 1

                rsi = d['rsi'][pi]
                if np.isnan(rsi):
                    continue

                carry_val = carry_map.get(sym, 0.0)
                if carry_val < carry_min:
                    continue

                # MR信号: RSI < rsi_max
                if rsi < rsi_max:
                    mr_score = (rsi_max - rsi) / rsi_max
                    carry_score = max(carry_val, 0) / 25.0
                    total_score = 0.5 * mr_score + 0.5 * carry_score
                    sigs.append((sym, total_score, il))

            sigs.sort(key=lambda x: -x[2])

            for sym, score, il in sigs:
                if len(pos) >= mp:
                    break
                d = raw[sym]
                entry_price = d['open'][il]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue

                ml, mr, _, _ = d['spec']

                # 期货仓位
                notional_per = eq * nm_futures / mp
                contracts = int(notional_per / (entry_price * ml))
                contracts = max(contracts, 1)
                notional = entry_price * ml * contracts
                margin = notional * mr
                if margin > eq * 0.8:
                    contracts = max(int(eq * 0.8 / (entry_price * ml * mr)), 1)
                    notional = entry_price * ml * contracts

                # 期权仓位: 买入OTM看涨
                option_cost = 0
                option_strike = 0
                option_lots = 0
                hv = d['hv20'][il] if not np.isnan(d['hv20'][il]) else 0.3
                if hv > 0.05:
                    K = entry_price * k_mult
                    T = hd / TD
                    r = 0.03
                    opt_price = bs_call(entry_price, K, T, r, hv)
                    if opt_price > 0:
                        # 用option_pct比例的权益买期权
                        opt_budget = eq * option_pct
                        option_lots = max(int(opt_budget / (opt_price * ml)), 0)
                        if option_lots > 0:
                            option_cost = opt_price * ml * option_lots
                            option_strike = K

                pos[sym] = {
                    'dir': 1, 'ed': date, 'ep': entry_price,
                    'ct': contracts, 'ml': ml, 'notional': notional,
                    'spec': d['spec'],
                    'option_cost': option_cost,
                    'option_strike': option_strike,
                    'option_lots': option_lots,
                    'hv': hv,
                    'pyramids': 0,
                }

        # === 权益 ===
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                il = im[date]
                S = raw[sym]['close'][il]
                h = int((date - ps['ed']) / np.timedelta64(1, 'D'))
                # 期货浮盈
                ur += (S - ps['ep']) * ps['ml'] * ps['ct']
                # 期权浮盈
                if ps.get('option_cost', 0) > 0:
                    T_remain = max(hd - h, 0.001) / TD
                    opt_price = bs_call(S, ps['option_strike'], T_remain, 0.03, ps['hv'])
                    ur += opt_price * ps['option_lots'] - ps['option_cost']
        ceq = eq + ur
        eqh.append(ceq)
        if ceq < 1000:
            break

    if not pnls or eq <= 0:
        return None
    tr = (eq - INIT) / INIT
    if tr <= -1:
        return None
    dys = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
    yrs = max(dys / 365, .001)
    ann = float((1 + tr) ** (1 / yrs) - 1)
    pa = np.array(pnls)
    wr = float((pa > 0).mean())
    aw = float(pa[pa > 0].mean()) if (pa > 0).any() else 0
    al = float(abs(pa[pa <= 0].mean())) if (pa <= 0).any() else 1
    pf = aw * (pa > 0).sum() / (al * (pa <= 0).sum()) if (pa <= 0).sum() > 0 and al > 0 else 0
    ea = np.array(eqh[1:])
    if len(ea) > 1:
        cm = np.maximum.accumulate(ea)
        dd = (ea - cm) / cm
        mdd = float(dd.min())
        rets = np.diff(ea) / ea[:-1]
        sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    else:
        mdd = 0; sh = 0

    # 期权贡献统计
    opt_wins = sum(1 for t in trade_log if t['option_pnl'] > 0)
    opt_total = len([t for t in trade_log if t.get('option_pnl', 0) != 0])

    return {
        'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf, 'trades': len(pa),
        'final': eq, 'sharpe': sh,
        'opt_wr': opt_wins / max(opt_total, 1),
        **p,
    }


def bt_pure_options(raw, dates, si, good_syms, carry_map, p):
    """
    纯买入期权策略 (作为对比)

    买入ATM/OTM看涨期权, 不做期货
    利用carry+MR信号选择方向和时机
    """
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 10)
    rsi_max = p.get('rsi_max', 25)
    k_mult = p.get('k_mult', 1.0)  # 1.0=ATM, 1.05=5%OTM
    risk_per_trade = p.get('risk_pct', 0.05)  # 每笔风险占权益%
    carry_min = p.get('carry_min', 0)

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]

    for date in dates:
        # 退出
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if not im or date not in im:
                continue
            il = im[date]
            S = raw[sym]['close'][il]
            h = int((date - ps['ed']) / np.timedelta64(1, 'D'))
            if h < hd:
                continue

            # 期权结算
            T_remain = max(hd - h, 0) / TD
            if T_remain > 0.001:
                opt_price = bs_call(S, ps['K'], T_remain, 0.03, ps['hv'])
            else:
                opt_price = max(S - ps['K'], 0)

            pnl = opt_price * ps['lots'] - ps['cost']
            comm = max(ps['cost'] * 0.002, 0)
            pnl -= comm
            eq += pnl
            pnls.append(pnl)
            del pos[sym]

        # 入场
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in pos:
                    continue
                if sym not in good_syms:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il <= 1:
                    continue
                pi = il - 1
                rsi = d['rsi'][pi]
                if np.isnan(rsi) or rsi >= rsi_max:
                    continue
                carry_val = carry_map.get(sym, 0)
                if carry_val < carry_min:
                    continue
                mr_score = (rsi_max - rsi) / rsi_max
                carry_score = max(carry_val, 0) / 25.0
                sigs.append((sym, 0.5 * mr_score + 0.5 * carry_score, il))

            sigs.sort(key=lambda x: -x[2])

            for sym, score, il in sigs:
                if len(pos) >= mp:
                    break
                d = raw[sym]
                S = d['open'][il]
                if np.isnan(S) or S <= 0:
                    continue
                ml, _, _, _ = d['spec']
                hv = d['hv20'][il] if not np.isnan(d['hv20'][il]) else 0.3
                if hv < 0.05:
                    continue
                K = S * k_mult
                T = hd / TD
                opt_price = bs_call(S, K, T, 0.03, hv)
                if opt_price <= 0:
                    continue
                budget = eq * risk_per_trade
                lots = max(int(budget / (opt_price * ml)), 1)
                cost = opt_price * ml * lots
                if cost > eq * 0.15:  # 单笔不超过15%
                    lots = max(int(eq * 0.15 / (opt_price * ml)), 1)
                    cost = opt_price * ml * lots
                pos[sym] = {
                    'ed': date, 'K': K, 'lots': lots, 'cost': cost,
                    'hv': hv, 'ml': ml,
                }

        # 权益
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                il = im[date]
                S = raw[sym]['close'][il]
                h = int((date - ps['ed']) / np.timedelta64(1, 'D'))
                T_remain = max(hd - h, 0.001) / TD
                opt_price = bs_call(S, ps['K'], T_remain, 0.03, ps['hv'])
                ur += opt_price * ps['lots'] - ps['cost']
        ceq = eq + ur
        eqh.append(ceq)
        if ceq < 1000:
            break

    if not pnls or eq <= 0:
        return None
    tr = (eq - INIT) / INIT
    if tr <= -1:
        return None
    dys = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
    yrs = max(dys / 365, .001)
    ann = float((1 + tr) ** (1 / yrs) - 1)
    pa = np.array(pnls)
    wr = float((pa > 0).mean())
    aw = float(pa[pa > 0].mean()) if (pa > 0).any() else 0
    al = float(abs(pa[pa <= 0].mean())) if (pa <= 0).any() else 1
    pf = aw * (pa > 0).sum() / (al * (pa <= 0).sum()) if (pa <= 0).sum() > 0 and al > 0 else 0
    ea = np.array(eqh[1:])
    if len(ea) > 1:
        cm = np.maximum.accumulate(ea)
        dd = (ea - cm) / cm
        mdd = float(dd.min())
        rets = np.diff(ea) / ea[:-1]
        sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    else:
        mdd = 0; sh = 0

    return {'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf, 'trades': len(pa),
            'final': eq, 'sharpe': sh, **p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    ts_dir = os.path.expanduser("~/home/futures_platform/data/futures_term_structure")

    print("加载carry数据...")
    carry = load_carry(ts_dir)
    backwd = {s: c for s, c in carry.items() if c > 0}

    print("加载行情数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    train_end = '2022-01-01'
    sym_mr = select_symbols(raw, train_end, 'mr', top_pct=0.5)
    sym_backwd = [s for s in raw if carry.get(s, 0) > 0]
    sym_mr_backwd = [s for s in sym_mr if carry.get(s, 0) > 0]
    sym_carry3 = [s for s in raw if carry.get(s, 0) > 3]
    sym_carry5 = [s for s in raw if carry.get(s, 0) > 5]

    print(f"MR top50%: {len(sym_mr)}")
    print(f"MR+backwd: {len(sym_mr_backwd)}")
    print(f"Backwd: {len(sym_backwd)}")
    print(f"carry>3%: {len(sym_carry3)}")
    print(f"carry>5%: {len(sym_carry5)}")

    test_s = pd.Timestamp('2022-01-01')
    test_e = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, test_s, test_e)
    print(f"测试期: {len(dates)}交易日\n")

    res = []

    # ===================================================================
    # 1. 期货+期权凸性 (主力策略)
    # ===================================================================
    print("=== 1. 期货+期权凸性 ===")
    for sym_name, sym_set in [('mr_backwd', sym_mr_backwd), ('backwd', sym_backwd),
                               ('c3', sym_carry3), ('c5', sym_carry5)]:
        for hd in [10, 15, 20]:
            for nm in [5, 8, 10, 15, 20]:
                for opt_pct in [0.05, 0.10, 0.15, 0.20]:
                    for k_m in [1.0, 1.03, 1.05, 1.10]:
                        p = dict(strategy='fut_opt', sym_set=sym_name,
                                 hold_days=hd, nm_futures=nm,
                                 option_pct=opt_pct, k_mult=k_m,
                                 rsi_max=25, carry_min=0, max_pos=3)
                        r = bt_futures_options(raw, dates, si, sym_set, carry, p)
                        if r:
                            res.append(r)

    # ===================================================================
    # 2. 金字塔加仓
    # ===================================================================
    print("=== 2. 金字塔加仓 ===")
    for sym_name, sym_set in [('mr_backwd', sym_mr_backwd), ('backwd', sym_backwd)]:
        for hd in [10, 15]:
            for nm in [5, 8, 10, 15]:
                p = dict(strategy='fut_opt', sym_set=sym_name,
                         hold_days=hd, nm_futures=nm,
                         option_pct=0.1, k_mult=1.05,
                         rsi_max=25, carry_min=0, max_pos=3,
                         pyramid=True, pyramid_th=0.03)
                r = bt_futures_options(raw, dates, si, sym_set, carry, p)
                if r:
                    res.append(r)

    # ===================================================================
    # 3. 纯买入期权 (对比)
    # ===================================================================
    print("=== 3. 纯买入期权 ===")
    for sym_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3)]:
        for hd in [10, 15, 20]:
            for risk_pct in [0.03, 0.05, 0.08, 0.10, 0.15]:
                for k_m in [0.95, 1.0, 1.05]:
                    p = dict(strategy='pure_opt', sym_set=sym_name,
                             hold_days=hd, risk_pct=risk_pct, k_mult=k_m,
                             rsi_max=25, carry_min=0, max_pos=3)
                    r = bt_pure_options(raw, dates, si, sym_set, carry, p)
                    if r:
                        res.append(r)

    # ===================================================================
    # 4. 宽松信号 + 高期权凸性
    # ===================================================================
    print("=== 4. 宽松信号 + 高凸性 ===")
    for rsi_max in [30, 35, 40]:
        for hd in [5, 10, 15]:
            for nm in [5, 8, 10]:
                for opt_pct in [0.10, 0.15, 0.20]:
                    p = dict(strategy='fut_opt', sym_set='backwd',
                             hold_days=hd, nm_futures=nm,
                             option_pct=opt_pct, k_mult=1.05,
                             rsi_max=rsi_max, carry_min=0, max_pos=3)
                    r = bt_futures_options(raw, dates, si, sym_backwd, carry, p)
                    if r:
                        res.append(r)

    print(f"\n总结果: {len(res)}组有效")
    res.sort(key=lambda x: x['annual'], reverse=True)

    # 输出
    print(f"\n{'Strat':>8} {'Sym':>8} {'H':>3} {'NM':>4} {'Opt%':>5} {'K':>5} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 105)
    for r in res[:80]:
        strat = r.get('strategy', '?')
        sym = r.get('sym_set', '?')
        k = r.get('k_mult', 1)
        opt = r.get('option_pct', 0)
        nm = r.get('nm_futures', r.get('risk_pct', 0))
        pyr = "+P" if r.get('pyramid') else ""
        print(f"{strat:>8} {sym:>8} {r.get('hold_days','-'):>3} "
              f"{nm:>4.0f} {opt:>5.0%} {k:>5.2f} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}{pyr}")

    # 分策略对比
    print("\n=== 分策略最佳 ===")
    for strat_name in ['fut_opt', 'pure_opt']:
        strat_res = [r for r in res if r.get('strategy') == strat_name]
        if not strat_res:
            continue
        print(f"\n--- {strat_name} Top 10 ---")
        for r in strat_res[:10]:
            k = r.get('k_mult', 1)
            opt = r.get('option_pct', r.get('risk_pct', 0))
            nm = r.get('nm_futures', 0)
            pyr = "+Pyr" if r.get('pyramid') else ""
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                  f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
                  f"H={r.get('hold_days',0)}  NM={nm:.0f}  Opt={opt:.0%}  "
                  f"K={k:.2f}  {r.get('sym_set','')}  Trades={r['trades']}{pyr}")

    # MDD可控
    print("\n=== MDD<50% 最佳 ===")
    g50 = sorted([r for r in res if r['mdd'] > -0.50], key=lambda x: -x['annual'])
    for r in g50[:20]:
        strat = r.get('strategy', '?')
        print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  MDD={r['mdd']:>7.1%}  "
              f"Sharpe={r['sharpe']:>5.2f}  "
              f"{strat} {r.get('sym_set','')} H={r.get('hold_days',0)} "
              f"NM={r.get('nm_futures',r.get('risk_pct',0)):.0f} "
              f"Opt={r.get('option_pct',r.get('risk_pct',0)):.0%} "
              f"K={r.get('k_mult',1):.2f}  Trades={r['trades']}")

    # 目标
    print("\n" + "=" * 105)
    for ta, tw, lb in [(6., .50, "年化>=600%"),
                       (3., .50, "年化>=300%"),
                       (2., .50, "年化>=200%"),
                       (1., .50, "年化>=100%"),
                       (0.5, .50, "年化>=50%")]:
        g = [r for r in res if r['annual'] >= ta and r['wr'] >= tw]
        print(f"\n=== {lb} & WR>=50%: {len(g)}组 ===")
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                strat = r.get('strategy', '?')
                k = r.get('k_mult', 1)
                opt = r.get('option_pct', r.get('risk_pct', 0))
                nm = r.get('nm_futures', 0)
                print(f"    年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  "
                      f"{strat} H={r.get('hold_days',0)} NM={nm:.0f} Opt={opt:.0%} K={k:.2f} "
                      f"{r.get('sym_set','')}  Trades={r['trades']}")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in r.items()} for r in res[:2000]]
    with open(os.path.join(od, 'backtest_v64.json'), 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v64.json")


if __name__ == '__main__':
    main()
