#!/usr/bin/env python3
"""
策略 V63 — Carry深度优化: 纯carry轮动 + Carry+MR组合 + 动态杠杆

V62发现: carry filter从6.6%提升到114%年化 (17倍!)
问题: MDD -100%+ 不实际, 需要风险可控的方案

V63方向:
  1. 纯carry轮动: 定期买入backwardated品种, 不等MR信号
  2. carry强度分级: carry>8%顶级 vs carry>3%中等
  3. MR+carry择时: 高carry品种等MR触发, 低carry品种不动
  4. 动态杠杆: carry越高杠杆越大
  5. 组合优化: 找到MDD<50%的最佳配置
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
INIT = 500000
TD = 252


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
        df['trade_date'] = pd.to_datetime(df['trade_date'])
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

        df['ma5'] = df['close'].rolling(5).mean()
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

        ch_hi = df['close'].rolling(20).max()
        ch_lo = df['close'].rolling(20).min()
        ch_range = (ch_hi - ch_lo).replace(0, np.nan)
        df['ch_pos'] = ((df['close'] - ch_lo) / ch_range).clip(0, 1)

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
            'ret': df['ret'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'cons_down': np.array(cons_d, dtype=np.int32),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'ch_pos': df['ch_pos'].values.astype(np.float64),
            'm5': df['m5'].values.astype(np.float64),
            'm10': df['m10'].values.astype(np.float64),
            'm20': df['m20'].values.astype(np.float64),
            'ma20': df['ma20'].values.astype(np.float64),
            'ma60': df['ma60'].values.astype(np.float64),
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
        if signal == 'mr':
            sig_mask = rsi < 25
        elif signal == 'mr_cons':
            sig_mask = (d['cons_down'][idx] >= 3) & (rsi < 35)
        else:
            sig_mask = rsi < 25
        n = sig_mask.sum()
        if n < 15:
            continue
        fwd_vals = fwd[sig_mask]
        valid = ~np.isnan(fwd_vals)
        n_valid = valid.sum()
        if n_valid < 10:
            continue
        wr = (fwd_vals[valid] > 0).mean()
        avg = fwd_vals[valid].mean()
        score = wr * max(avg, 0)
        scores[sym] = {'wr': wr, 'avg': avg, 'n': n_valid, 'score': score}
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda x: -x[1]['score'])
    cutoff = max(1, int(len(ranked) * top_pct))
    return [s for s, _ in ranked[:cutoff]]


def bt_carry_rotation(raw, dates, si, carry_syms, carry_map, p):
    """
    纯carry轮动策略:
    - 定期(每隔rebal_days)再平衡持仓
    - 持有top_k个最高carry品种
    - 不需要MR信号, 纯carry驱动
    """
    mp = p.get('max_pos', 3)
    rebal = p.get('rebal_days', 20)
    nm = p.get('notional_mult', 5)
    use_trend = p.get('use_trend', False)  # 趋势过滤
    use_mr = p.get('use_mr', False)  # MR择时
    rsi_max = p.get('rsi_max', 40)  # MR时RSI上限
    rebal_counter = 0
    last_rebal_date = None

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]

    for date in dates:
        # === 检查再平衡 ===
        if last_rebal_date is not None:
            days_since = int((date - last_rebal_date) / np.timedelta64(1, 'D'))
        else:
            days_since = rebal  # 首日触发再平衡

        if days_since >= rebal:
            # 先平所有仓
            for sym in list(pos):
                ps = pos[sym]
                im = si.get(sym)
                if im and date in im:
                    il = im[date]
                    S = raw[sym]['close'][il]
                    ml = ps['ml']
                    trade_pnl = (S - ps['ep']) * ml * ps['ct']
                    notional_exit = S * ml * ps['ct']
                    comm = COMM * (ps['notional'] + notional_exit)
                    pnl = trade_pnl - comm
                    eq += pnl
                    pnls.append(pnl)
                del pos[sym]

            # 选carry最高的品种建仓
            candidates = []
            for sym in carry_syms:
                if sym in pos:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il <= 1:
                    continue

                c_val = carry_map.get(sym, 0)
                if c_val <= 0:
                    continue

                d = raw[sym]

                # 趋势过滤: 价格在MA20上方
                if use_trend:
                    ma20 = d['ma20'][il]
                    if not np.isnan(ma20) and d['close'][il] < ma20:
                        continue

                # MR择时: RSI不能太高
                if use_mr:
                    rsi = d['rsi'][il]
                    if np.isnan(rsi) or rsi > rsi_max:
                        continue

                candidates.append((sym, c_val, il))

            # 按carry排序
            candidates.sort(key=lambda x: -x[1])

            for sym, c_val, il in candidates[:mp]:
                if len(pos) >= mp:
                    break
                d = raw[sym]
                entry_price = d['close'][il]  # 当日收盘建仓
                if np.isnan(entry_price) or entry_price <= 0:
                    continue
                ml, mr, _, _ = d['spec']
                notional_per = eq * nm / mp
                contracts = int(notional_per / (entry_price * ml))
                contracts = max(contracts, 1)
                notional = entry_price * ml * contracts
                margin = notional * mr
                if margin > eq * 0.9:
                    contracts = max(int(eq * 0.9 / (entry_price * ml * mr)), 1)
                    notional = entry_price * ml * contracts
                pos[sym] = {
                    'dir': 1, 'ed': date, 'ep': entry_price,
                    'ct': contracts, 'ml': ml, 'notional': notional,
                }
            last_rebal_date = date

        # === 权益 ===
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                ur += (S - ps['ep']) * ps['ml'] * ps['ct']
        ceq = eq + ur
        eqh.append(ceq)
        if ceq < 1000:
            break

    if not pnls and not pos:
        return None
    # 平掉剩余持仓
    if pos:
        last_date = dates[-1]
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if im and last_date in im:
                il = im[last_date]
                S = raw[sym]['close'][il]
                ml = ps['ml']
                trade_pnl = (S - ps['ep']) * ml * ps['ct']
                notional_exit = S * ml * ps['ct']
                comm = COMM * (ps['notional'] + notional_exit)
                pnl = trade_pnl - comm
                eq += pnl
                pnls.append(pnl)

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


def bt_mr_carry(raw, dates, si, good_syms, carry_map, p):
    """MR + carry组合 (V62逻辑优化版)"""
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 10)
    rsi_max = p.get('rsi_max', 25)
    nm = p.get('notional_mult', 10.0)
    carry_min = p.get('carry_min', 0.0)
    use_trend = p.get('use_trend', False)
    dynamic_nm = p.get('dynamic_nm', False)  # 动态杠杆: 高carry高杠杆

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]

    for date in dates:
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
            ml = ps['ml']
            trade_pnl = (S - ps['ep']) * ml * ps['ct']
            notional_exit = S * ml * ps['ct']
            comm = COMM * (ps['notional'] + notional_exit)
            pnl = trade_pnl - comm
            eq += pnl
            pnls.append(pnl)
            del pos[sym]

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

                if rsi < rsi_max:
                    # 趋势过滤
                    if use_trend:
                        ma20 = d['ma20'][il]
                        if not np.isnan(ma20) and d['close'][il] < ma20:
                            continue

                    mr_score = (rsi_max - rsi) / rsi_max
                    carry_score = max(carry_val, 0) / 25.0
                    total_score = 0.5 * mr_score + 0.5 * carry_score

                    # 动态杠杆
                    if dynamic_nm:
                        adj_nm = nm * (1 + carry_val / 20.0)  # carry 20% → 2x leverage
                    else:
                        adj_nm = nm

                    sigs.append((sym, 1, total_score, il, adj_nm))

            sigs.sort(key=lambda x: -x[2])

            for sym, direction, score, il, adj_nm in sigs:
                if len(pos) >= mp:
                    break
                d = raw[sym]
                entry_price = d['open'][il]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue
                ml, mr, _, _ = d['spec']
                notional_per = eq * adj_nm / mp
                contracts = int(notional_per / (entry_price * ml))
                contracts = max(contracts, 1)
                notional = entry_price * ml * contracts
                margin = notional * mr
                if margin > eq * 0.9:
                    contracts = max(int(eq * 0.9 / (entry_price * ml * mr)), 1)
                    notional = entry_price * ml * contracts
                pos[sym] = {
                    'dir': direction, 'ed': date, 'ep': entry_price,
                    'ct': contracts, 'ml': ml, 'notional': notional, 'hd': hd,
                }

        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                if ps['dir'] > 0:
                    ur += (S - ps['ep']) * ps['ml'] * ps['ct']
                else:
                    ur += (ps['ep'] - S) * ps['ml'] * ps['ct']
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
            'final': eq, 'sharpe': sh, 'n_sym': len(good_syms), **p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    ts_dir = os.path.expanduser("~/home/futures_platform/data/futures_term_structure")

    print("加载carry数据...")
    carry = load_carry(ts_dir)
    backwd = {s: c for s, c in carry.items() if c > 0}
    print(f"  Backwardation: {len(backwd)}品种")
    print(f"  Top: {sorted(backwd.items(), key=lambda x: -x[1])[:15]}")

    print("加载行情数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    train_end = '2022-01-01'
    sym_mr = select_symbols(raw, train_end, 'mr', top_pct=0.5)
    print(f"MR top50%: {len(sym_mr)}品种")

    # 不同carry层级的品种集
    sym_backwd = [s for s in raw if carry.get(s, 0) > 0]
    sym_carry3 = [s for s in raw if carry.get(s, 0) > 3]
    sym_carry5 = [s for s in raw if carry.get(s, 0) > 5]
    sym_carry8 = [s for s in raw if carry.get(s, 0) > 8]
    sym_mr_backwd = [s for s in sym_mr if carry.get(s, 0) > 0]
    sym_mr_carry3 = [s for s in sym_mr if carry.get(s, 0) > 3]

    print(f"\n品种集:")
    print(f"  backwardated (carry>0): {len(sym_backwd)} → {sym_backwd}")
    print(f"  carry>3%: {len(sym_carry3)} → {sym_carry3}")
    print(f"  carry>5%: {len(sym_carry5)} → {sym_carry5}")
    print(f"  carry>8%: {len(sym_carry8)} → {sym_carry8}")
    print(f"  MR+backwardated: {len(sym_mr_backwd)} → {sym_mr_backwd}")
    print(f"  MR+carry>3%: {len(sym_mr_carry3)} → {sym_mr_carry3}")

    test_s = pd.Timestamp('2022-01-01')
    test_e = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, test_s, test_e)
    print(f"\n测试期: {len(dates)}交易日")

    res = []

    # ===================================================================
    # 1. 纯carry轮动 (不依赖MR信号)
    # ===================================================================
    print("\n=== 1. 纯carry轮动 ===")
    for sym_set_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3),
                                   ('c5', sym_carry5), ('c8', sym_carry8)]:
        if not sym_set:
            continue
        for rebal in [10, 15, 20, 30]:
            for nm in [3, 5, 8, 10, 15]:
                for mp in [2, 3]:
                    p = dict(strategy='carry_rotation', max_pos=mp, rebal_days=rebal,
                             notional_mult=nm, sym_set=sym_set_name,
                             use_trend=False, use_mr=False)
                    r = bt_carry_rotation(raw, dates, si, sym_set, carry, p)
                    if r:
                        res.append(r)

    # ===================================================================
    # 2. Carry轮动 + 趋势过滤
    # ===================================================================
    print("=== 2. Carry轮动 + 趋势 ===")
    for sym_set_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3)]:
        for rebal in [10, 20, 30]:
            for nm in [3, 5, 8, 10]:
                p = dict(strategy='carry_rotation', max_pos=3, rebal_days=rebal,
                         notional_mult=nm, sym_set=sym_set_name,
                         use_trend=True, use_mr=False)
                r = bt_carry_rotation(raw, dates, si, sym_set, carry, p)
                if r:
                    res.append(r)

    # ===================================================================
    # 3. Carry轮动 + MR择时
    # ===================================================================
    print("=== 3. Carry轮动 + MR择时 ===")
    for sym_set_name, sym_set in [('backwd', sym_backwd), ('c3', sym_carry3)]:
        for rebal in [10, 20, 30]:
            for nm in [3, 5, 8, 10, 15]:
                for rsi_max in [35, 40, 50]:
                    p = dict(strategy='carry_rotation', max_pos=3, rebal_days=rebal,
                             notional_mult=nm, sym_set=sym_set_name,
                             use_trend=False, use_mr=True, rsi_max=rsi_max)
                    r = bt_carry_rotation(raw, dates, si, sym_set, carry, p)
                    if r:
                        res.append(r)

    # ===================================================================
    # 4. MR+carry组合 (V62优化)
    # ===================================================================
    print("=== 4. MR+carry组合 ===")
    for sym_set_name, sym_set in [('mr_backwd', sym_mr_backwd), ('mr_c3', sym_mr_carry3),
                                   ('backwd', sym_backwd), ('c3', sym_carry3)]:
        for hd in [5, 10, 15, 20]:
            for nm in [3, 5, 8, 10, 15, 20]:
                for carry_min in [0, 3]:
                    p = dict(strategy='mr_carry', hold_days=hd, rsi_max=25,
                             notional_mult=nm, carry_min=carry_min,
                             use_filter=True, sym_set=sym_set_name)
                    r = bt_mr_carry(raw, dates, si, sym_set, carry, p)
                    if r:
                        res.append(r)

    # ===================================================================
    # 5. MR+carry + 趋势过滤
    # ===================================================================
    print("=== 5. MR+carry + 趋势 ===")
    for sym_set_name, sym_set in [('mr_backwd', sym_mr_backwd), ('backwd', sym_backwd)]:
        for hd in [10, 15]:
            for nm in [5, 8, 10, 15]:
                p = dict(strategy='mr_carry', hold_days=hd, rsi_max=25,
                         notional_mult=nm, carry_min=0,
                         use_trend=True, sym_set=sym_set_name)
                r = bt_mr_carry(raw, dates, si, sym_set, carry, p)
                if r:
                    res.append(r)

    # ===================================================================
    # 6. 动态杠杆 (carry越高杠杆越大)
    # ===================================================================
    print("=== 6. 动态杠杆 ===")
    for sym_set_name, sym_set in [('mr_backwd', sym_mr_backwd), ('backwd', sym_backwd)]:
        for hd in [10, 15]:
            for nm in [5, 8, 10]:
                p = dict(strategy='mr_carry', hold_days=hd, rsi_max=25,
                         notional_mult=nm, carry_min=0,
                         dynamic_nm=True, sym_set=sym_set_name)
                r = bt_mr_carry(raw, dates, si, sym_set, carry, p)
                if r:
                    res.append(r)

    # ===================================================================
    # 7. 宽松信号 (RSI<35/40) + carry
    # ===================================================================
    print("=== 7. 宽松信号 + carry ===")
    for rsi_max in [30, 35, 40]:
        for hd in [5, 10, 15]:
            for nm in [5, 8, 10]:
                p = dict(strategy='mr_carry', hold_days=hd, rsi_max=rsi_max,
                         notional_mult=nm, carry_min=0,
                         sym_set='backwd')
                r = bt_mr_carry(raw, dates, si, sym_backwd, carry, p)
                if r:
                    res.append(r)

    print(f"\n总参数组合: {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    # 输出最佳结果
    print(f"\n{'Strat':>15} {'Sym':>8} {'Reb':>4} {'H':>3} {'NM':>4} {'MP':>3} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 110)
    for r in res[:80]:
        strat = r.get('strategy', 'mr_carry')
        sym = r.get('sym_set', '?')
        reb = r.get('rebal_days', '-')
        hd = r.get('hold_days', '-')
        trend = "+T" if r.get('use_trend') else ""
        mr = "+MR" if r.get('use_mr') else ""
        dyn = "+Dyn" if r.get('dynamic_nm') else ""
        print(f"{strat:>15} {sym:>8} {reb:>4} {hd:>3} "
              f"{r.get('notional_mult',0):>4.0f} {r.get('max_pos',3):>3} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}{trend}{mr}{dyn}")

    # MDD可控的结果
    print("\n" + "=" * 110)
    print("\n=== MDD<-50%的结果 ===")
    g50 = [r for r in res if r['mdd'] > -0.50]
    g50.sort(key=lambda x: x['annual'], reverse=True)
    for r in g50[:30]:
        strat = r.get('strategy', 'mr_carry')
        sym = r.get('sym_set', '?')
        trend = "+T" if r.get('use_trend') else ""
        mr = "+MR" if r.get('use_mr') else ""
        dyn = "+Dyn" if r.get('dynamic_nm') else ""
        rsi = r.get('rsi_max', 25)
        print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
              f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
              f"{strat} {sym} H={r.get('hold_days','-')} Reb={r.get('rebal_days','-')} "
              f"NM={r.get('notional_mult',0):.0f} RSI<{rsi} {trend}{mr}{dyn}  "
              f"Trades={r['trades']}")

    print("\n=== MDD<-30%的结果 ===")
    g30 = [r for r in res if r['mdd'] > -0.30]
    g30.sort(key=lambda x: x['annual'], reverse=True)
    for r in g30[:20]:
        strat = r.get('strategy', 'mr_carry')
        sym = r.get('sym_set', '?')
        print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
              f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
              f"{strat} {sym} NM={r.get('notional_mult',0):.0f}  Trades={r['trades']}")

    # 分策略对比
    print("\n=== 分策略对比 ===")
    for strat_name in ['carry_rotation', 'mr_carry']:
        strat_res = [r for r in res if r.get('strategy') == strat_name]
        if not strat_res:
            continue
        print(f"\n--- {strat_name} ---")
        print(f"  最佳年化: {strat_res[0]['annual']:.1%}")
        # MDD<50%最佳
        safe = [r for r in strat_res if r['mdd'] > -0.50]
        if safe:
            safe.sort(key=lambda x: x['annual'], reverse=True)
            best_safe = safe[0]
            print(f"  MDD<50%最佳: 年化={best_safe['annual']:.1%}  WR={best_safe['wr']:.1%}  "
                  f"MDD={best_safe['mdd']:.1%}  Sharpe={best_safe['sharpe']:.2f}")

    # 目标达成
    print("\n" + "=" * 110)
    for ta, tw, lb in [(6., .50, "年化>=600% & WR>=50%"),
                       (3., .50, "年化>=300% & WR>=50%"),
                       (2., .50, "年化>=200% & WR>=50%"),
                       (1., .50, "年化>=100% & WR>=50%"),
                       (0.5, .50, "年化>=50% & WR>=50%"),
                       (0., .50, "正收益 & WR>=50%")]:
        g = [r for r in res if r['annual'] >= ta and r['wr'] >= tw]
        print(f"\n=== {lb}: {len(g)}组 ===")
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                strat = r.get('strategy', 'mr_carry')
                sym = r.get('sym_set', '?')
                print(f"    年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  {strat} {sym} NM={r.get('notional_mult',0):.0f}  "
                      f"Trades={r['trades']}")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in r.items()} for r in res[:1000]]
    with open(os.path.join(od, 'backtest_v63.json'), 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v63.json")


if __name__ == '__main__':
    main()
