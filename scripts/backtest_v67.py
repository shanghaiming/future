#!/usr/bin/env python3
"""
策略 V67 — 20日新高突破 + carry 走前验证

V66发现: new_high20 + carry>3% + H=5 + NM=5 = 2363%年化, 72.6%WR

验证内容:
  1. Walk-forward: 训练期(2018-2021)选出的carry品种, 在测试期(2022-2026)是否有效
  2. 不同子时段一致性: 2022, 2023, 2024, 2025 各年度分别表现
  3. 信号衰减分析: 信号后1-20天的收益分布
  4. 滚动carry替代: 用历史价格构造的carry指标代替单日快照
  5. 与纯MR对比: new_high20是否真的比MR好

关键问题:
  - carry数据是2026-05-15单日 → 可能有look-ahead bias
  - 需要验证: 哪些品种STRUCTURALLY长期处于backwardation
  - 如果carry分类不稳定, 则结果是虚假的
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
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300:
            continue
        df['ret'] = df['close'].pct_change()
        for lag in [1, 2, 3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        for w in [5, 10, 20, 40, 60]:
            df[f'ma{w}'] = df['close'].rolling(w).mean()
        d = df['close'].diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))
        ch_hi20 = df['close'].rolling(20).max()
        df['at_new_high20'] = (df['close'] >= ch_hi20).astype(int)
        ch_hi10 = df['close'].rolling(10).max()
        df['at_new_high10'] = (df['close'] >= ch_hi10).astype(int)
        for hold in [1, 2, 3, 5, 7, 10, 15, 20]:
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
            'hv20': df['hv20'].values.astype(np.float64),
            'ma5': df['ma5'].values.astype(np.float64),
            'ma20': df['ma20'].values.astype(np.float64),
            'm5': df['m5'].values.astype(np.float64),
            'm10': df['m10'].values.astype(np.float64),
            'at_new_high20': df['at_new_high20'].values.astype(np.int32),
            'at_new_high10': df['at_new_high10'].values.astype(np.int32),
            'fwd1': df['fwd1'].values.astype(np.float64),
            'fwd2': df['fwd2'].values.astype(np.float64),
            'fwd3': df['fwd3'].values.astype(np.float64),
            'fwd5': df['fwd5'].values.astype(np.float64),
            'fwd7': df['fwd7'].values.astype(np.float64),
            'fwd10': df['fwd10'].values.astype(np.float64),
            'fwd15': df['fwd15'].values.astype(np.float64),
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


def bt(raw, dates, si, good_syms, carry_map, p):
    """通用回测引擎"""
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 5)
    nm = p.get('notional_mult', 5)
    signal = p.get('signal', 'new_high20')

    eq = float(INIT)
    hwm = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]
    monthly_pnls = {}  # 年月→pnl列表

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
            ml = ps['ml']
            trade_pnl = (S - ps['ep']) * ml * ps['ct']
            notional_exit = S * ml * ps['ct']
            comm = COMM * (ps['notional'] + notional_exit)
            pnl = trade_pnl - comm
            eq += pnl
            pnls.append(pnl)
            # 月度统计
            ym = str(date)[:7]
            monthly_pnls.setdefault(ym, []).append(pnl)
            if eq > hwm:
                hwm = eq
            del pos[sym]

        # 入场
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in pos or sym not in good_syms:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il < 20:
                    continue
                pi = il - 1

                triggered = False
                score = 0
                carry_val = carry_map.get(sym, 0)

                if signal == 'new_high20':
                    if d['at_new_high20'][pi] == 1:
                        triggered = True
                        score = carry_val / 25 + 0.5
                elif signal == 'new_high10':
                    if d['at_new_high10'][pi] == 1:
                        triggered = True
                        score = carry_val / 25 + 0.5
                elif signal == 'mr_rsi25':
                    if d['rsi'][pi] < 25:
                        triggered = True
                        score = (25 - d['rsi'][pi]) / 25 + carry_val / 50
                elif signal == 'carry_mom5':
                    if d['m5'][pi] > 0.01:
                        triggered = True
                        score = d['m5'][pi] * 10 + carry_val / 25
                elif signal == 'combo_trend':
                    # MA5>MA20 + 5日动量>2% + carry
                    if d['ma5'][pi] > d['ma20'][pi] and d['m5'][pi] > 0.02:
                        triggered = True
                        score = d['m5'][pi] * 10 + carry_val / 25

                if not triggered or np.isnan(score):
                    continue
                sigs.append((sym, score, il))

            sigs.sort(key=lambda x: -x[1])

            for sym, score, il in sigs:
                if len(pos) >= mp:
                    break
                d = raw[sym]
                entry_price = d['open'][il]
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

        # 权益
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

    # 年度分解
    yearly = {}
    for ym, p_list in monthly_pnls.items():
        year = ym[:4]
        yearly.setdefault(year, []).extend(p_list)

    return {
        'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf, 'trades': len(pa),
        'final': eq, 'sharpe': sh, 'yearly': yearly, **p,
    }


def print_yearly_breakdown(r):
    """打印年度分解"""
    yearly = r.get('yearly', {})
    print(f"  年度分解:")
    total = 0
    for year in sorted(yearly.keys()):
        pnls = yearly[year]
        pa = np.array(pnls)
        if len(pa) == 0:
            continue
        yr_wr = (pa > 0).mean()
        yr_pnl = pa.sum()
        yr_trades = len(pa)
        total += yr_pnl
        print(f"    {year}: {yr_trades:>3}笔  WR={yr_wr:>5.1%}  PnL={yr_pnl:>12,.0f}  "
              f"avg={pa.mean():>8,.0f}")
    print(f"    合计: {r['trades']:>3}笔  总PnL={total:>12,.0f}  年化={r['annual']:.1%}")


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    ts_dir = os.path.expanduser("~/home/futures_platform/data/futures_term_structure")

    print("加载carry数据...")
    carry = load_carry(ts_dir)

    print("加载行情数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    sym_backwd = [s for s in raw if carry.get(s, 0) > 0]
    sym_carry3 = [s for s in raw if carry.get(s, 0) > 3]
    sym_carry5 = [s for s in raw if carry.get(s, 0) > 5]
    sym_all = list(raw.keys())

    print(f"Backwd: {len(sym_backwd)}, carry>3: {len(sym_carry3)}, carry>5: {len(sym_carry5)}")

    # ===================================================================
    # 验证1: 全测试期 new_high20 + carry
    # ===================================================================
    print("\n" + "=" * 80)
    print("=== 验证1: 全测试期 (2022-2026) ===")
    test_s = pd.Timestamp('2022-01-01')
    test_e = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, test_s, test_e)
    print(f"测试期: {len(dates)}交易日")

    configs = [
        ('new_high20+c3', sym_carry3, dict(signal='new_high20', hold_days=5, notional_mult=5, max_pos=3)),
        ('new_high20+c3+H3', sym_carry3, dict(signal='new_high20', hold_days=3, notional_mult=5, max_pos=3)),
        ('new_high20+c3+H10', sym_carry3, dict(signal='new_high20', hold_days=10, notional_mult=5, max_pos=3)),
        ('new_high20+c3+NM3', sym_carry3, dict(signal='new_high20', hold_days=5, notional_mult=3, max_pos=3)),
        ('new_high20+c3+NM10', sym_carry3, dict(signal='new_high20', hold_days=5, notional_mult=10, max_pos=3)),
        ('new_high20+c5', sym_carry5, dict(signal='new_high20', hold_days=5, notional_mult=5, max_pos=3)),
        ('new_high20+backwd', sym_backwd, dict(signal='new_high20', hold_days=5, notional_mult=5, max_pos=3)),
        ('new_high20+all', sym_all, dict(signal='new_high20', hold_days=5, notional_mult=5, max_pos=3)),
        ('MR+c3', sym_carry3, dict(signal='mr_rsi25', hold_days=10, notional_mult=5, max_pos=3)),
        ('MR+backwd', sym_backwd, dict(signal='mr_rsi25', hold_days=10, notional_mult=5, max_pos=3)),
        ('carry_mom5+c3', sym_carry3, dict(signal='carry_mom5', hold_days=5, notional_mult=5, max_pos=3)),
        ('new_high10+c3', sym_carry3, dict(signal='new_high10', hold_days=5, notional_mult=5, max_pos=3)),
        ('combo_trend+c3', sym_carry3, dict(signal='combo_trend', hold_days=5, notional_mult=5, max_pos=3)),
    ]

    for name, sym_set, params in configs:
        r = bt(raw, dates, si, sym_set, carry, params)
        if r:
            print(f"\n--- {name} ---")
            print(f"  年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  "
                  f"MDD={r['mdd']:.1%}  Sharpe={r['sharpe']:.2f}  Trades={r['trades']}")
            print_yearly_breakdown(r)

    # ===================================================================
    # 验证2: 分年度验证
    # ===================================================================
    print("\n" + "=" * 80)
    print("=== 验证2: 分年度 (new_high20 + carry>3%, NM=5, H=5) ===")
    for year in [2022, 2023, 2024, 2025]:
        ys = pd.Timestamp(f'{year}-01-01')
        ye = pd.Timestamp(f'{year}-12-31')
        dates_y, si_y = build_idx(raw, ys, ye)
        if len(dates_y) < 50:
            continue
        r = bt(raw, dates_y, si_y, sym_carry3, carry,
               dict(signal='new_high20', hold_days=5, notional_mult=5, max_pos=3))
        if r:
            print(f"  {year}: {len(dates_y)}日  年化={r['annual']:.1%}  WR={r['wr']:.1%}  "
                  f"MDD={r['mdd']:.1%}  Trades={r['trades']}")

    # ===================================================================
    # 验证3: 信号衰减分析 (new_high20后1-20天收益)
    # ===================================================================
    print("\n" + "=" * 80)
    print("=== 验证3: 信号衰减分析 ===")
    print("  new_high20 + carry>3% 的forward returns:")
    for hold in [1, 2, 3, 5, 7, 10, 15, 20]:
        fwd_key = f'fwd{hold}'
        vals = []
        for sym in sym_carry3:
            d = raw[sym]
            # 测试期
            mask = (d['dates'] >= test_s) & (d['dates'] <= test_e)
            idx = np.where(mask)[0]
            for i in idx:
                if i < 20:
                    continue
                if d['at_new_high20'][i] == 1:
                    fwd = d[fwd_key][i]
                    if not np.isnan(fwd):
                        vals.append(fwd)
        if len(vals) < 50:
            continue
        arr = np.array(vals)
        wr = (arr > 0).mean()
        avg = arr.mean()
        std = arr.std()
        t_stat = avg / (std / np.sqrt(len(arr))) if std > 0 else 0
        print(f"    hold={hold:>2}d: WR={wr:.1%}  avg={avg:.3%}  std={std:.3%}  "
              f"t={t_stat:.1f}  N={len(vals)}")

    # 对比: 无carry筛选的new_high20
    print("\n  new_high20 + ALL symbols:")
    for hold in [3, 5, 10]:
        fwd_key = f'fwd{hold}'
        vals = []
        for sym in sym_all:
            d = raw[sym]
            mask = (d['dates'] >= test_s) & (d['dates'] <= test_e)
            idx = np.where(mask)[0]
            for i in idx:
                if i < 20:
                    continue
                if d['at_new_high20'][i] == 1:
                    fwd = d[fwd_key][i]
                    if not np.isnan(fwd):
                        vals.append(fwd)
        if vals:
            arr = np.array(vals)
            wr = (arr > 0).mean()
            avg = arr.mean()
            print(f"    hold={hold:>2}d: WR={wr:.1%}  avg={avg:.3%}  N={len(vals)}")

    # ===================================================================
    # 验证4: 随机carry分类测试 (排除look-ahead bias)
    # ===================================================================
    print("\n" + "=" * 80)
    print("=== 验证4: 随机carry分类 vs 真实carry ===")
    np.random.seed(42)
    # 随机选8个品种 (与carry>3%数量相同)
    random_syms = np.random.choice(sym_all, size=len(sym_carry3), replace=False).tolist()

    for name, sym_set in [('真实carry>3%', sym_carry3),
                           ('随机8品种', random_syms),
                           ('contango品种', [s for s in sym_all if carry.get(s, 0) < 0])]:
        r = bt(raw, dates, si, sym_set, carry,
               dict(signal='new_high20', hold_days=5, notional_mult=5, max_pos=3))
        if r:
            print(f"  {name}: 年化={r['annual']:.1%}  WR={r['wr']:.1%}  "
                  f"MDD={r['mdd']:.1%}  Trades={r['trades']}")

    # ===================================================================
    # 验证5: 用价格动量作为carry的代理 (避免look-ahead)
    # ===================================================================
    print("\n" + "=" * 80)
    print("=== 验证5: 用长期动量代替carry ===")
    print("  假设: 长期上涨品种 ≈ backwardation (价格结构)")

    # 用60日动量排名前25%的品种作为carry的代理
    train_s = pd.Timestamp('2020-01-01')
    train_e = pd.Timestamp('2021-12-31')
    dates_train, si_train = build_idx(raw, train_s, train_e)

    # 在训练期末计算各品种60日动量
    mom_scores = {}
    for sym, d in raw.items():
        mask = (d['dates'] <= train_e) & (d['dates'] >= pd.Timestamp('2020-12-01'))
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue
        m60 = d['close'][idx[-1]] / d['close'][idx[0]] - 1
        mom_scores[sym] = m60

    # 排名, 取前25%
    ranked_mom = sorted(mom_scores.items(), key=lambda x: -x[1])
    n_top = max(8, int(len(ranked_mom) * 0.25))
    sym_mom_top = [s for s, _ in ranked_mom[:n_top]]
    sym_mom_bot = [s for s, _ in ranked_mom[-n_top:]]

    print(f"  动量top {n_top}品种: {sym_mom_top[:10]}")
    print(f"  动量bottom {n_top}品种: {sym_mom_bot[:10]}")

    for name, sym_set in [('动量top25%', sym_mom_top),
                           ('动量bottom25%', sym_mom_bot),
                           ('真实carry>3%', sym_carry3)]:
        r = bt(raw, dates, si, sym_set, carry,
               dict(signal='new_high20', hold_days=5, notional_mult=5, max_pos=3))
        if r:
            print(f"  {name}: 年化={r['annual']:.1%}  WR={r['wr']:.1%}  "
                  f"MDD={r['mdd']:.1%}  Trades={r['trades']}")

    # ===================================================================
    # 验证6: 滚动carry替代 (用MA5/MA20比率作为趋势carry代理)
    # ===================================================================
    print("\n" + "=" * 80)
    print("=== 验证6: 动态carry (MA5>MA20作为趋势代理) ===")
    # 不预选品种, 而是每天动态选择MA5>MA20的品种进行new_high20交易
    # 这样完全消除了look-ahead bias

    def bt_dynamic_carry(raw, dates, si, carry_map, p):
        """动态carry: 只在MA5>MA20的品种中交易new_high20"""
        mp = p.get('max_pos', 3)
        hd = p.get('hold_days', 5)
        nm = p.get('notional_mult', 5)
        use_carry_filter = p.get('use_carry_filter', False)
        carry_min = p.get('carry_min', 0)

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
                    im = si.get(sym)
                    if not im or date not in im:
                        continue
                    il = im[date]
                    if il < 20:
                        continue
                    pi = il - 1

                    # 动态趋势过滤: MA5 > MA20
                    if np.isnan(d['ma5'][pi]) or np.isnan(d['ma20'][pi]):
                        continue
                    if d['ma5'][pi] <= d['ma20'][pi]:
                        continue

                    # 信号: new_high20
                    if d['at_new_high20'][pi] != 1:
                        continue

                    # 可选: carry过滤
                    if use_carry_filter and carry_map.get(sym, 0) < carry_min:
                        continue

                    # 评分: 动量强度
                    score = d['m5'][pi] * 10
                    if not np.isnan(score):
                        sigs.append((sym, score, il))

                sigs.sort(key=lambda x: -x[1])
                for sym, score, il in sigs:
                    if len(pos) >= mp:
                        break
                    d = raw[sym]
                    entry_price = d['open'][il]
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
        ea = np.array(eqh[1:])
        if len(ea) > 1:
            cm = np.maximum.accumulate(ea)
            dd = (ea - cm) / cm
            mdd = float(dd.min())
            rets = np.diff(ea) / ea[:-1]
            sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': 0, 'trades': len(pa),
                'final': eq, 'sharpe': sh, **p}

    # 动态carry (无look-ahead)
    for nm in [3, 5, 8, 10]:
        r = bt_dynamic_carry(raw, dates, si, carry,
                              dict(hold_days=5, notional_mult=nm, max_pos=3,
                                   use_carry_filter=False))
        if r:
            print(f"  动态carry NM={nm}: 年化={r['annual']:.1%}  WR={r['wr']:.1%}  "
                  f"MDD={r['mdd']:.1%}  Sharpe={r['sharpe']:.2f}  Trades={r['trades']}")

    # 动态carry + 静态carry过滤
    print("\n  动态carry + 静态carry>3%过滤:")
    for nm in [3, 5, 8]:
        r = bt_dynamic_carry(raw, dates, si, carry,
                              dict(hold_days=5, notional_mult=nm, max_pos=3,
                                   use_carry_filter=True, carry_min=3))
        if r:
            print(f"  NM={nm}: 年化={r['annual']:.1%}  WR={r['wr']:.1%}  "
                  f"MDD={r['mdd']:.1%}  Sharpe={r['sharpe']:.2f}  Trades={r['trades']}")

    # ===================================================================
    # 最终: 最佳策略完整统计
    # ===================================================================
    print("\n" + "=" * 80)
    print("=== 最终策略验证 ===")
    print("\n最佳策略: new_high20 + carry>3% + H=5 + NM=5 + max_pos=3")
    r = bt(raw, dates, si, sym_carry3, carry,
           dict(signal='new_high20', hold_days=5, notional_mult=5, max_pos=3))
    if r:
        print(f"  年化={r['annual']:.1%}")
        print(f"  胜率={r['wr']:.1%}")
        print(f"  PF={r['pf']:.2f}")
        print(f"  MDD={r['mdd']:.1%}")
        print(f"  Sharpe={r['sharpe']:.2f}")
        print(f"  交易数={r['trades']}")
        print(f"  最终权益={r['final']:,.0f}")
        print(f"  初始={INIT:,.0f}")
        print_yearly_breakdown(r)


if __name__ == '__main__':
    main()
