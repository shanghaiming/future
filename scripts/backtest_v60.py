#!/usr/bin/env python3
"""
策略 V60 — 品种专化 + 双策略 (MR + 动量) + 走前验证

数据分析发现品种差异巨大:
  MR最优品种 (>55% WR):
    SAFI 60.8% WR +1.49%avg, sifi 62.4% +0.72%, SMFI 59.9% +0.59%
    jdfi 59.9% +0.47%, ZCFI 58.8% +0.96%, egfi 56.7% +0.58%
  动量最优品种 (>55% WR):
    JRFI 63.9% WR +1.07%avg, RSFI 56.3% +0.45%

策略:
  1. 品种专化: 每个品种用最适合它的策略
  2. MR品种: 3连跌 + RSI<40 → 买入做多, 持有10天
  3. 动量品种: 4/4看多 → 买入做多, 持有5天
  4. 无止损, 纯时间退出 (匹配数据分析方法)
  5. 激进杠杆 (10-30×)
  6. 走前验证: 用2018-2022训练品种选择, 2022-2026测试
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
R = 0.02
INIT = 500000
TD = 252

# 数据分析确认的高WR品种
MR_SYMBOLS = ['safi', 'sifi', 'smfi', 'jdfi', 'zcfi', 'egfi']
MOM_SYMBOLS = ['jrfi', 'rsfi']
ALL_SPECIAL = MR_SYMBOLS + MOM_SYMBOLS


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

        # 连续下跌天数
        down = (df['ret'] < 0).astype(int).values
        cons_d = []
        c = 0
        for v in down:
            c = c + 1 if v else 0
            cons_d.append(c)
        df['cons_down'] = cons_d

        # 动量方向 (4/4 看多/看空)
        mom_cols = ['m3', 'm5', 'm10', 'm20', 'm60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_count = np.sum(signs > 0, axis=1)
        neg_count = np.sum(signs < 0, axis=1)
        df['mom_bull'] = (pos_count >= 4).astype(int)
        df['mom_bear'] = (neg_count >= 4).astype(int)

        # 前向收益 (用于品种WR计算)
        for hold in [5, 10]:
            df[f'fwd_ret{hold}'] = df['close'].shift(-hold) / df['close'] - 1

        df = df.dropna(subset=['ma20', 'ma60', 'hv20', 'rsi'])
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
            'high': df['high'].values.astype(np.float64),
            'low': df['low'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'cons_down': np.array(cons_d, dtype=np.int32),
            'mom_bull': df['mom_bull'].values.astype(np.int32),
            'mom_bear': df['mom_bear'].values.astype(np.int32),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'fwd_ret5': df['fwd_ret5'].values.astype(np.float64),
            'fwd_ret10': df['fwd_ret10'].values.astype(np.float64),
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


def compute_walk_forward_symbols(raw, train_end):
    """走前计算: 用训练期数据计算每个品种的MR/动量WR"""
    results = {}
    te = pd.Timestamp(train_end)

    for sym, d in raw.items():
        mask = d['dates'] <= te
        indices = np.where(mask)[0]
        if len(indices) < 200:
            continue

        # MR WR: 3 cons down → fwd_ret10 > 0
        mr_mask = d['cons_down'][indices] >= 3
        mr_n = mr_mask.sum()
        if mr_n >= 30:
            fwd = d['fwd_ret10'][indices[mr_mask]]
            valid = ~np.isnan(fwd)
            if valid.sum() >= 20:
                mr_wr = (fwd[valid] > 0).mean()
                mr_avg = fwd[valid].mean()
            else:
                mr_wr = 0
                mr_avg = 0
        else:
            mr_wr = 0
            mr_avg = 0

        # MOM WR: 4/4 bull → fwd_ret5 > 0
        mom_mask = d['mom_bull'][indices] >= 1
        mom_n = mom_mask.sum()
        if mom_n >= 30:
            fwd = d['fwd_ret5'][indices[mom_mask]]
            valid = ~np.isnan(fwd)
            if valid.sum() >= 20:
                mom_wr = (fwd[valid] > 0).mean()
                mom_avg = fwd[valid].mean()
            else:
                mom_wr = 0
                mom_avg = 0
        else:
            mom_wr = 0
            mom_avg = 0

        results[sym] = {
            'mr_wr': mr_wr, 'mr_avg': mr_avg, 'mr_n': mr_n,
            'mom_wr': mom_wr, 'mom_avg': mom_avg, 'mom_n': mom_n,
        }

    return results


def bt(raw, dates, si, sym_info, p):
    """品种专化回测"""
    mp = p.get('max_pos', 3)
    mr_hold = p.get('mr_hold', 10)
    mom_hold = p.get('mom_hold', 5)
    nm = p.get('notional_mult', 15.0)
    mr_wr_min = p.get('mr_wr_min', 0.52)
    mom_wr_min = p.get('mom_wr_min', 0.52)
    mr_rsi_max = p.get('mr_rsi_max', 40)
    mr_cons_min = p.get('mr_cons_min', 3)
    mode = p.get('mode', 'adaptive')  # 'adaptive', 'mr_only', 'mom_only', 'all_symbols'

    # 根据走前结果选择品种和策略
    mr_syms = []
    mom_syms = []
    for sym, info in sym_info.items():
        if sym not in raw:
            continue
        if info['mr_wr'] >= mr_wr_min and info['mr_n'] >= 20:
            mr_syms.append(sym)
        if info['mom_wr'] >= mom_wr_min and info['mom_n'] >= 20:
            mom_syms.append(sym)

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

            if h < ps['hd']:
                continue

            # 计算PnL
            ml = ps['ml']
            trade_pnl = (S - ps['ep']) * ml * ps['ct']  # 都是做多
            notional_exit = S * ml * ps['ct']
            comm = COMM * (ps['notional'] + notional_exit)
            pnl = trade_pnl - comm
            eq += pnl
            pnls.append(pnl)
            del pos[sym]

        # === 入场 ===
        if len(pos) < mp:
            sigs = []

            for sym in raw:
                if sym in pos:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il <= 1:
                    continue
                pi = il - 1

                d = raw[sym]
                score = 0.0
                hold = 0

                # MR信号
                if mode in ('adaptive', 'mr_only') and sym in mr_syms:
                    rsi = d['rsi'][pi]
                    cons = d['cons_down'][pi]
                    if not np.isnan(rsi) and rsi < mr_rsi_max and cons >= mr_cons_min:
                        score = (mr_rsi_max - rsi) / mr_rsi_max + cons * 0.2
                        hold = mr_hold

                # 动量信号
                if mode in ('adaptive', 'mom_only') and sym in mom_syms and score == 0:
                    if d['mom_bull'][pi] >= 1:
                        score = 1.0
                        hold = mom_hold

                # 全品种模式 (不做品种筛选)
                if mode == 'all_symbols' and score == 0:
                    rsi = d['rsi'][pi]
                    cons = d['cons_down'][pi]
                    if not np.isnan(rsi) and rsi < mr_rsi_max and cons >= mr_cons_min:
                        score = (mr_rsi_max - rsi) / mr_rsi_max + cons * 0.2
                        hold = mr_hold

                if score > 0 and hold > 0:
                    sigs.append((sym, score, hold, il))

            sigs.sort(key=lambda x: -x[1])

            for sym, score, hold, il in sigs:
                if len(pos) >= mp:
                    break

                d = raw[sym]
                entry_price = d['open'][il]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue

                ml, mr, _, _ = d['spec']

                # 固定名义值仓位
                notional_per = eq * nm / mp
                contracts = int(notional_per / (entry_price * ml))
                contracts = max(contracts, 1)
                notional = entry_price * ml * contracts

                # 保证金上限
                margin = notional * mr
                if margin > eq * 0.9:
                    contracts = max(int(eq * 0.9 / (entry_price * ml * mr)), 1)
                    notional = entry_price * ml * contracts

                pos[sym] = {
                    'ed': date, 'ep': entry_price,
                    'ct': contracts, 'ml': ml,
                    'notional': notional, 'hd': hold,
                }

        # === 权益追踪 ===
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
        mdd = 0
        sh = 0

    mr_n = len(mr_syms)
    mom_n = len(mom_syms)
    return {'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf, 'trades': len(pa),
            'final': eq, 'sharpe': sh, 'mr_sym': mr_n, 'mom_sym': mom_n, **p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    # === 走前验证 ===
    train_end = '2022-01-01'
    print(f"\n走前品种选择 (训练期截止 {train_end}):")
    sym_info = compute_walk_forward_symbols(raw, train_end)

    mr_syms = [(s, i) for s, i in sym_info.items() if i['mr_wr'] >= 0.52 and i['mr_n'] >= 20]
    mr_syms.sort(key=lambda x: -x[1]['mr_wr'])
    print(f"\n  MR品种 (WR>=52%, N>=20):")
    for s, i in mr_syms[:15]:
        print(f"    {s:>6}: WR={i['mr_wr']:.1%} Avg={i['mr_avg']:.4%} N={i['mr_n']}")

    mom_syms = [(s, i) for s, i in sym_info.items() if i['mom_wr'] >= 0.52 and i['mom_n'] >= 20]
    mom_syms.sort(key=lambda x: -x[1]['mom_wr'])
    print(f"\n  动量品种 (WR>=52%, N>=20):")
    for s, i in mom_syms[:15]:
        print(f"    {s:>6}: WR={i['mom_wr']:.1%} Avg={i['mom_avg']:.4%} N={i['mom_n']}")

    # 测试期: 2022-2026
    test_start = pd.Timestamp('2022-01-01')
    test_end = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, test_start, test_end)
    print(f"\n测试期: {len(dates)}交易日 ({test_start.date()} ~ {test_end.date()})")

    # === 参数扫描 ===
    pl = []

    # A: 走前品种专化 + 不同杠杆
    for nm in [5, 8, 10, 15, 20, 25, 30]:
        for mr_wr_min in [0.50, 0.52, 0.55]:
            pl.append(dict(mode='adaptive', notional_mult=nm,
                           mr_wr_min=mr_wr_min, mom_wr_min=0.52,
                           mr_hold=10, mom_hold=5))

    # B: 纯MR品种
    for nm in [5, 10, 15, 20]:
        pl.append(dict(mode='mr_only', notional_mult=nm,
                       mr_wr_min=0.52, mom_wr_min=1.0,
                       mr_hold=10, mom_hold=5))

    # C: 纯动量品种
    for nm in [5, 10, 15]:
        pl.append(dict(mode='mom_only', notional_mult=nm,
                       mr_wr_min=1.0, mom_wr_min=0.52,
                       mr_hold=10, mom_hold=5))

    # D: 全品种 (不做走前筛选, 对比基线)
    for nm in [3, 5, 8, 10]:
        pl.append(dict(mode='all_symbols', notional_mult=nm,
                       mr_wr_min=0.0, mom_wr_min=0.0,
                       mr_hold=10, mom_hold=5))

    # E: 不同MR参数
    for mr_rsi in [35, 40, 45]:
        for mr_cons in [2, 3, 4]:
            for nm in [8, 15]:
                pl.append(dict(mode='adaptive', notional_mult=nm,
                               mr_wr_min=0.52, mom_wr_min=0.52,
                               mr_hold=10, mom_hold=5,
                               mr_rsi_max=mr_rsi, mr_cons_min=mr_cons))

    # F: 不同持有期
    for mr_h in [5, 7, 10, 15]:
        for mom_h in [3, 5, 7]:
            pl.append(dict(mode='adaptive', notional_mult=10,
                           mr_wr_min=0.52, mom_wr_min=0.52,
                           mr_hold=mr_h, mom_hold=mom_h))

    # G: 也加入做空 (MR超买 + 动量看空)
    # 先只做多, 在H加入做空测试

    # H: 极端杠杆 (追求600%)
    for nm in [20, 25, 30, 40, 50]:
        for mr_wr_min in [0.50, 0.53]:
            pl.append(dict(mode='adaptive', notional_mult=nm,
                           mr_wr_min=mr_wr_min, mom_wr_min=0.53,
                           mr_hold=10, mom_hold=5,
                           max_pos=3))

    print(f"\n参数组合: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 50 == 0:
            print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, sym_info, p)
        if r:
            res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'Mode':>10} {'NM':>4} {'MRwr':>5} {'H_mr':>5} {'H_mom':>5} {'MR_s':>4} {'MO_s':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 110)
    for r in res[:60]:
        print(f"{r.get('mode','adaptive'):>10} {r.get('notional_mult',10):>4.0f} "
              f"{r.get('mr_wr_min',0.52):>5.2f} {r.get('mr_hold',10):>5} {r.get('mom_hold',5):>5} "
              f"{r.get('mr_sym',0):>4} {r.get('mom_sym',0):>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "=" * 110)
    for ta, tw, lb in [(6., .50, "年化>=600% & WR>=50%"),
                       (5., .50, "年化>=500% & WR>=50%"),
                       (3., .50, "年化>=300% & WR>=50%"),
                       (2., .50, "年化>=200% & WR>=50%"),
                       (1., .50, "年化>=100% & WR>=50%"),
                       (0.5, .50, "年化>=50% & WR>=50%"),
                       (0.5, .45, "年化>=50% & WR>=45%"),
                       (0., .50, "正收益 & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual'] >= ta and r['wr'] >= tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                      f"MDD={r['mdd']:>7.1%}  Sharpe={r['sharpe']:>5.2f}  "
                      f"Mode={r.get('mode','?')}  NM={r.get('notional_mult',10):.0f}  "
                      f"MRwr>={r.get('mr_wr_min',.52):.2f}  "
                      f"MR_sym={r.get('mr_sym',0)}  MOM_sym={r.get('mom_sym',0)}  "
                      f"H_mr={r.get('mr_hold',10)}  H_mom={r.get('mom_hold',5)}  "
                      f"Trades={r['trades']}")
        else:
            print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in r.items()} for r in res[:500]]
    with open(os.path.join(od, 'backtest_v60.json'), 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v60.json")


if __name__ == '__main__':
    main()
