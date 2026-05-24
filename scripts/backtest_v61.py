#!/usr/bin/env python3
"""
策略 V61 — 优化MR参数 + 走前品种选择 + 低HV过滤

综合数据分析发现:
  1. 最优持有期: 10-20天 (t统计量在15-20天达峰)
  2. 最佳信号: RSI<25 hold10d → 53.1%WR +0.764%avg (t=18.1!)
  3. 品种选择: 排名前半品种 → 56.8%WR +1.08%avg
  4. 低HV过滤: HV<40th + MR → WR提升至54.2%
  5. 最优组合: 看涨品种 + 3cons_down + RSI<35 + HV<40th, hold20d → 54.2%WR +0.96%

核心改变 vs V60:
  - 持有期从10天→15-20天
  - 信号从3cons_down→RSI<25 (更高WR)
  - 品种选择: top-half (更稳健) 而非>55% (太少)
  - 新增HV<40th过滤
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
INIT = 500000
TD = 252


def load(data_dir):
    raw = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'): continue
        sym = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300: continue
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

        # RSI
        d = df['close'].diff(); g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        # 连续下跌
        down = (df['ret'] < 0).astype(int).values
        cons_d = []
        c = 0
        for v in down:
            c = c + 1 if v else 0
            cons_d.append(c)
        df['cons_down'] = cons_d

        # 通道位置
        ch_hi = df['close'].rolling(20).max()
        ch_lo = df['close'].rolling(20).min()
        ch_range = (ch_hi - ch_lo).replace(0, np.nan)
        df['ch_pos'] = ((df['close'] - ch_lo) / ch_range).clip(0, 1)

        # 前向收益
        for hold in [5, 10, 15, 20]:
            df[f'fwd{hold}'] = df['close'].shift(-hold) / df['close'] - 1

        df = df.dropna(subset=['ma20', 'hv20', 'rsi'])
        if len(df) < 100: continue
        try:
            spec = get_spec(sym)
        except:
            continue

        raw[sym] = {
            'spec': spec,
            'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'cons_down': np.array(cons_d, dtype=np.int32),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'ch_pos': df['ch_pos'].values.astype(np.float64),
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
    """
    走前品种选择: 用训练期数据选排名前top_pct的品种
    signal: 'mr' (RSI<25 hold 10d WR), 'mr_cons' (3down+RSI<35 hold 10d)
    """
    te = pd.Timestamp(train_end)
    scores = {}

    for sym, d in raw.items():
        mask = d['dates'] <= te
        idx = np.where(mask)[0]
        if len(idx) < 200:
            continue

        rsi = d['rsi'][idx]
        cons = d['cons_down'][idx]
        fwd = d['fwd10'][idx]

        if signal == 'mr':
            # RSI<25, 持有10天
            sig_mask = rsi < 25
        elif signal == 'mr_cons':
            # 3连跌 + RSI<35
            sig_mask = (cons >= 3) & (rsi < 35)
        elif signal == 'mr_tight':
            # RSI<20, 持有10天 (更严格)
            sig_mask = rsi < 20
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
        # 综合分数: WR × avg (两者都重要)
        score = wr * max(avg, 0)
        scores[sym] = {'wr': wr, 'avg': avg, 'n': n_valid, 'score': score}

    if not scores:
        return []

    # 排名, 取前top_pct
    ranked = sorted(scores.items(), key=lambda x: -x[1]['score'])
    cutoff = max(1, int(len(ranked) * top_pct))
    selected = [s for s, _ in ranked[:cutoff]]

    return selected


def bt(raw, dates, si, good_syms, p):
    """优化MR回测"""
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 15)
    rsi_max = p.get('rsi_max', 25)
    cons_min = p.get('cons_min', 0)  # 0 = 不要求连跌
    hv_max = p.get('hv_max', 1.0)   # 1.0 = 不过滤
    nm = p.get('notional_mult', 10.0)
    use_filter = p.get('use_filter', True)  # 是否使用品种过滤
    short_side = p.get('short_side', False)

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]

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
                continue

            ml = ps['ml']
            if ps['dir'] > 0:
                trade_pnl = (S - ps['ep']) * ml * ps['ct']
            else:
                trade_pnl = (ps['ep'] - S) * ml * ps['ct']

            notional_exit = S * ml * ps['ct']
            comm = COMM * (ps['notional'] + notional_exit)
            pnl = trade_pnl - comm
            eq += pnl
            pnls.append(pnl)
            del pos[sym]

        # === 入场 ===
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in pos:
                    continue
                if use_filter and sym not in good_syms:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il <= 1:
                    continue
                pi = il - 1

                rsi = d['rsi'][pi]
                cons = d['cons_down'][pi]
                hv_pct = d['hv_pct'][pi]

                if np.isnan(rsi) or np.isnan(hv_pct):
                    continue

                # 做多: 超卖
                if rsi < rsi_max:
                    if cons_min > 0 and cons < cons_min:
                        continue
                    if hv_pct > hv_max:
                        continue

                    # 信号强度
                    score = (rsi_max - rsi) / rsi_max
                    sigs.append((sym, 1, score, il))

                # 做空: 超买
                if short_side and rsi > (100 - rsi_max):
                    if hv_pct > hv_max:
                        continue
                    score = (rsi - (100 - rsi_max)) / (100 - rsi_max)
                    sigs.append((sym, -1, score, il))

            sigs.sort(key=lambda x: -x[2])

            for sym, direction, score, il in sigs:
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
                    'dir': direction, 'ed': date, 'ep': entry_price,
                    'ct': contracts, 'ml': ml, 'notional': notional, 'hd': hd,
                }

        # === 权益 ===
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
    print("加载..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    train_end = '2022-01-01'
    print(f"\n=== 走前品种选择 (训练截止 {train_end}) ===")

    # RSI<25, top 50%
    sym_mr = select_symbols(raw, train_end, 'mr', top_pct=0.5)
    print(f"  MR (RSI<25) top50%: {len(sym_mr)}品种 → {sym_mr[:10]}...")

    # RSI<20, top 50%
    sym_mr_tight = select_symbols(raw, train_end, 'mr_tight', top_pct=0.5)
    print(f"  MR (RSI<20) top50%: {len(sym_mr_tight)}品种 → {sym_mr_tight[:10]}...")

    # 3cons_down + RSI<35, top 50%
    sym_mr_cons = select_symbols(raw, train_end, 'mr_cons', top_pct=0.5)
    print(f"  MR (3down+RSI<35) top50%: {len(sym_mr_cons)}品种 → {sym_mr_cons[:10]}...")

    # 测试期
    test_s = pd.Timestamp('2022-01-01')
    test_e = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, test_s, test_e)
    print(f"\n测试期: {len(dates)}交易日")

    pl = []

    # === A: 品种过滤 + RSI<25 + 不同持有期 ===
    for hd in [10, 15, 20]:
        for nm in [3, 5, 8, 10, 15, 20]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=True, signal_type='mr'))

    # === B: 品种过滤 + RSI<20 (更严格) ===
    for hd in [15, 20]:
        for nm in [5, 10, 15, 20]:
            pl.append(dict(hold_days=hd, rsi_max=20, notional_mult=nm,
                           use_filter=True, signal_type='mr_tight'))

    # === C: 品种过滤 + 3cons_down + RSI<35 ===
    for hd in [10, 15, 20]:
        for nm in [5, 10, 15]:
            pl.append(dict(hold_days=hd, rsi_max=35, cons_min=3,
                           notional_mult=nm, use_filter=True, signal_type='mr_cons'))

    # === D: 加HV过滤 (HV<40th) ===
    for hd in [15, 20]:
        for nm in [5, 8, 10, 15]:
            pl.append(dict(hold_days=hd, rsi_max=25, hv_max=0.4,
                           notional_mult=nm, use_filter=True, signal_type='mr'))

    # === E: 无品种过滤 (对比基线) ===
    for hd in [10, 15, 20]:
        for nm in [3, 5, 8]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=False, signal_type='mr'))

    # === F: 双向 (做多超卖 + 做空超买) ===
    for hd in [10, 15]:
        for nm in [5, 8, 10]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=True, short_side=True, signal_type='mr'))

    # === G: 极端杠杆 ===
    for hd in [15, 20]:
        for nm in [20, 25, 30, 40]:
            pl.append(dict(hold_days=hd, rsi_max=25, notional_mult=nm,
                           use_filter=True, signal_type='mr'))

    print(f"\n参数组合: {len(pl)}组")
    bt0 = time.time()
    res = []

    # 根据signal_type选择品种集
    sym_map = {'mr': sym_mr, 'mr_tight': sym_mr_tight, 'mr_cons': sym_mr_cons}

    for i, p in enumerate(pl):
        if i % 50 == 0:
            print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        sig_type = p.pop('signal_type', 'mr')
        good_syms = sym_map.get(sig_type, sym_mr)
        r = bt(raw, dates, si, good_syms, p)
        if r:
            res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'Filter':>6} {'RSI':>4} {'H':>3} {'HV':>4} {'NM':>4} {'Nsym':>5} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 105)
    for r in res[:60]:
        filt = "Y" if r.get('use_filter', True) else "N"
        hv_s = f"<{r.get('hv_max',1):.1f}" if r.get('hv_max', 1) < 1 else "off"
        short = "+S" if r.get('short_side', False) else ""
        print(f"{filt:>6} {r.get('rsi_max',25):>4} {r['hold_days']:>3} {hv_s:>4} "
              f"{r.get('notional_mult',10):>4.0f} {r.get('n_sym',0):>5} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}{short}")

    print("\n" + "=" * 105)
    for ta, tw, lb in [(6., .50, "年化>=600% & WR>=50%"),
                       (3., .50, "年化>=300% & WR>=50%"),
                       (2., .50, "年化>=200% & WR>=50%"),
                       (1., .50, "年化>=100% & WR>=50%"),
                       (0.5, .50, "年化>=50% & WR>=50%"),
                       (0., .50, "正收益 & WR>=50%"),
                       (0., .48, "正收益 & WR>=48%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual'] >= ta and r['wr'] >= tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:15]:
                hv_s = f"HV<{r.get('hv_max',1):.1f}" if r.get('hv_max',1) < 1 else ""
                short = "+Short" if r.get('short_side') else ""
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                      f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
                      f"H={r['hold_days']}  RSI<{r.get('rsi_max',25)}  "
                      f"NM={r.get('notional_mult',10):.0f}  "
                      f"Nsym={r.get('n_sym',0)}  "
                      f"{hv_s} {short} Trades={r['trades']}")
        else:
            print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in r.items()} for r in res[:500]]
    with open(os.path.join(od, 'backtest_v61.json'), 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v61.json")


if __name__ == '__main__':
    main()
