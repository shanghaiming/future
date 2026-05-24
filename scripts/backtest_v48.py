#!/usr/bin/env python3
"""
策略 V48 — 极致复利版

核心: 高频复利 × 期权杠杆 × 多维信号
  - 持有3-5天, 快速轮转
  - 高风险比例 (4-6%)
  - OTM 1-3% 获取杠杆
  - 共识信号过滤提高质量
  - 取消人为上限, 让复利发挥作用
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.0003
R = 0.02
INIT = 500000
TD = 252


def bs(S, K, T, r, sigma, flag='call'):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if flag == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + .5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) if flag == 'call' \
        else (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


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
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else .5, raw=False)
        for lag in [5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)
        df['ts'] = df['m5'] - df['m60']
        up = df['ret'].where(df['ret'] > 0, 0).rolling(20).std() * np.sqrt(TD)
        dn = (-df['ret'].where(df['ret'] < 0, 0)).rolling(20).std() * np.sqrt(TD)
        df['skew'] = (dn - up) / df['hv20'].replace(0, np.nan)
        for sp in [4, 8, 16]:
            ef = df['close'].ewm(span=sp).mean()
            es = df['close'].ewm(span=sp*4).mean()
            s = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ew{sp}'] = (ef - es) / s
        df['sdelta'] = df['ew4']*.2 + df['ew8']*.3 + df['ew16']*.3
        df['sgamma'] = (df['hv20'].diff(5) / df['hv20'].shift(5).replace(0, np.nan)).rolling(10).std()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1., -1.)
        for w in [10, 20, 40]:
            hh, ll = df['close'].rolling(w).max(), df['close'].rolling(w).min()
            df[f'bo{w}'] = (df['close'] - .5*(hh+ll)) / (hh-ll+.001) * 2
        s20 = df['ret'].rolling(20).std().replace(0, np.nan)
        df['ramom'] = df['m10'] / (s20 * np.sqrt(10))
        d = df['close'].diff(); g = d.where(d>0,0).rolling(14).mean()
        l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))
        tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / df['close']

        df = df.dropna(subset=['ma20','ma60','hv20','m5','rsi','sdelta'])
        if len(df) < 100:
            continue
        try: spec = get_spec(sym)
        except: continue

        # 预计算
        sd = df['sdelta'].values
        ts = df['ts'].values
        iv_ts = (df['hv20']/df['hv40']).replace(0,np.nan).values
        trend = df['trend'].values
        skew = df['skew'].values
        bo20 = df['bo20'].values
        ramom = df['ramom'].values
        gamma = df['sgamma'].values
        mom5 = df['m5'].values

        # 多维一致性
        dims = np.stack([
            np.where(~np.isnan(sd), np.sign(sd), 0),
            np.where(~np.isnan(mom5), np.sign(mom5), 0),
            trend,
            np.where(~np.isnan(bo20), np.sign(bo20), 0),
            np.where(~np.isnan(ts), np.sign(ts), 0),
            np.where(~np.isnan(iv_ts)&(iv_ts>0), np.where(iv_ts>1.2,-1.,np.where(iv_ts<.8,1.,0.)), 0),
            np.where(~np.isnan(skew), -np.sign(skew), 0),
        ], axis=1)
        csum = np.nansum(dims, axis=1)
        ccount = np.sum(np.abs(dims) > 0, axis=1)
        cpct = np.where(ccount > 0, np.abs(csum) / ccount, 0)

        # 加权分数
        def _cl(x, c): return np.sign(x) * np.minimum(np.abs(x), c)
        sc = np.zeros(len(df)); w = np.zeros(len(df))
        m = ~np.isnan(sd); sc[m] += _cl(sd[m],3)*3; w[m] += 3
        m = ~np.isnan(ts); sc[m] += _cl(ts[m]*20,2); w[m] += 2
        m = ~np.isnan(iv_ts)&(iv_ts>0); sc[m] += _cl(-(iv_ts[m]-1)*5,1.5); w[m] += 1
        m = ~np.isnan(skew); sc[m] += _cl(-skew[m],2); w[m] += 1.5
        m = ~np.isnan(bo20); sc[m] += bo20[m]; w[m] += 1
        m = ~np.isnan(ramom); sc[m] += _cl(ramom[m],3)*1.5; w[m] += 1.5
        m = (trend!=0)&~np.isnan(sc)&(np.sign(sc)==trend); sc[m] *= 1.4
        m = ~np.isnan(gamma)&(gamma>.25); sc[m] *= .7; w[m] *= .7
        sc = np.where(w>0, sc/np.maximum(w,1), 0)

        raw[sym] = {
            'spec': spec, 'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'sgamma': df['sgamma'].values.astype(np.float64),
            'dir': np.sign(csum).astype(np.float64),
            'cpct': cpct.astype(np.float64),
            'cabs': np.abs(csum).astype(np.float64),
            'score': sc.astype(np.float64),
        }
    return raw


def build_idx(raw, s, e):
    ad = set()
    for d in raw.values():
        m = (d['dates']>=s)&(d['dates']<=e)
        for dt in d['dates'][m]: ad.add(dt)
    dates = np.array(sorted(ad))
    si = {}
    for sym, d in raw.items():
        im = {}
        m = (d['dates']>=s)&(d['dates']<=e)
        for dt, il in zip(d['dates'][m], np.where(m)[0]): im[dt] = int(il)
        si[sym] = im
    return dates, si


def bt(raw, dates, si, p):
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 5)
    rp = p.get('risk_pct', .04)
    otm = p.get('otm_pct', .01)
    ms = p.get('min_score', .3)
    mc = p.get('min_consensus', 3)
    mstr = p.get('min_str', .5)
    ld = p.get('liq_disc', .9)
    hv_lo = p.get('hv_lo', .10)
    hv_hi = p.get('hv_hi', .60)

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]

    for date in dates:
        # 退出
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if not im or date not in im: continue
            il = im[date]
            S = raw[sym]['close'][il]
            h = int((date - ps['ed']) / np.timedelta64(1, 'D'))
            if h < hd: continue

            Tr = max(ps['T'] - h/TD, .001)
            ov = bs(S, ps['K'], Tr, R, ps['sig'], ps['fl'])
            intr = max(S - ps['K'], 0) if ps['fl'] == 'call' else max(ps['K'] - S, 0)
            ev = max(ov, intr) * ld
            pnl = (ev - ps['pr']) * ps['d'] * ps['ml'] * ps['ct']
            c = ev * ps['ml'] * ps['ct'] * COMM
            eq += pnl - c
            pnls.append(pnl - c)
            del pos[sym]

        # 入场
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in pos: continue
                im = si.get(sym)
                if not im or date not in im: continue
                il = im[date]
                if il <= 0: continue
                pi = il - 1

                hv = d['hv20'][pi]
                if np.isnan(hv) or hv < hv_lo or hv > hv_hi: continue
                hp = d['hv_pct'][pi]
                if np.isnan(hp) or hp > .90: continue
                rsi = d['rsi'][pi]
                if np.isnan(rsi) or rsi > 78 or rsi < 22: continue
                ap = d['atr_pct'][pi]
                if np.isnan(ap) or ap > .06: continue

                dir_v = d['dir'][pi]
                cpct = d['cpct'][pi]
                cabs = d['cabs'][pi]
                sc = d['score'][pi]

                if dir_v == 0: continue
                if cabs < mc: continue
                if cpct < mstr: continue
                if abs(sc) < ms: continue

                sigs.append((sym, dir_v, sc, cpct, il, hv))

            sigs.sort(key=lambda x: abs(x[2])*x[3], reverse=True)

            for sym, dv, sc, cp, il, hv in sigs:
                if len(pos) >= mp: break

                S = raw[sym]['open'][il]
                if np.isnan(S) or S <= 0: S = raw[sym]['close'][il]
                if S <= 0: continue

                fl = 'call' if dv > 0 else 'put'
                K = S * (1 + otm * dv)
                T = hd / TD
                sig = hv
                pr = bs(S, K, T, R, sig, fl)
                if pr <= 0: continue

                ml, mr, _, _ = raw[sym]['spec']
                risk = eq * rp
                ct = max(int(risk / (pr * ml)), 1)
                # 上限: 不超过权益的3倍名义
                ct = min(ct, max(int(eq * 3 / (S * ml)), 1))

                cost = pr * ml * ct
                c = cost * COMM
                tc = cost + c
                # 不超过权益30%
                if tc > eq * .3:
                    ct = max(int(eq * .3 / (pr * ml + 1)), 1)
                    cost = pr * ml * ct; c = cost * COMM; tc = cost + c
                if tc > eq:
                    continue

                pos[sym] = {
                    'd': dv, 'ed': date, 'S': S, 'K': K, 'T': T, 'sig': sig,
                    'pr': pr, 'fl': fl, 'ml': ml, 'ct': ct,
                }

        # 权益
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                Sn = raw[sym]['close'][im[date]]
                h = int((date - ps['ed']) / np.timedelta64(1, 'D'))
                Tr = max(ps['T'] - h/TD, .001)
                ov = bs(Sn, ps['K'], Tr, R, ps['sig'], ps['fl'])
                intr = max(Sn - ps['K'], 0) if ps['fl'] == 'call' else max(ps['K'] - Sn, 0)
                ur += (max(ov, intr)*ld - ps['pr']) * ps['d'] * ps['ml'] * ps['ct']
        ceq = eq + ur
        eqh.append(ceq)
        if ceq < 1000: break

    if not pnls or eq <= 0: return None
    tr = (eq - INIT) / INIT
    if tr <= -1: return None
    dys = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
    yrs = max(dys/365, .001)
    ann = float((1+tr)**(1/yrs)-1)
    pa = np.array(pnls)
    wr = float((pa>0).mean())
    aw = float(pa[pa>0].mean()) if (pa>0).any() else 0
    al = float(abs(pa[pa<=0].mean())) if (pa<=0).any() else 1
    pf = aw*(pa>0).sum()/(al*(pa<=0).sum()) if (pa<=0).sum()>0 and al>0 else 0
    ea = np.array(eqh[1:])
    if len(ea) > 1:
        cm = np.maximum.accumulate(ea); dd = (ea-cm)/cm; mdd = float(dd.min())
        rets = np.diff(ea)/ea[:-1]
        sh = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    else: mdd=0; sh=0
    return {'annual':ann,'wr':wr,'mdd':mdd,'pf':pf,'trades':len(pa),'final':eq,'sharpe':sh,**p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载..."); t0=time.time()
    raw = load(dd); print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")
    sd = pd.Timestamp('2018-01-01'); ed = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, sd, ed); print(f"  {len(dates)}交易日")

    pl = []
    # A: 短持有, 高风险
    for hd in [3, 5]:
        for rp in [.04, .05, .06]:
            for otm in [0, .01, .02, .03]:
                for mc in [2, 3, 4]:
                    pl.append(dict(hold_days=hd, risk_pct=rp, otm_pct=otm,
                                   min_score=.3, min_consensus=mc, min_str=.4))

    # B: 中持有, 中等风险
    for hd in [5, 7]:
        for rp in [.03, .04, .05]:
            for otm in [0, .01, .02]:
                for mc in [3, 4]:
                    pl.append(dict(hold_days=hd, risk_pct=rp, otm_pct=otm,
                                   min_score=.5, min_consensus=mc, min_str=.5))

    # C: 极短持有 (3天) + 高杠杆
    for rp in [.05, .06]:
        for otm in [.01, .02, .03]:
            for mc in [2, 3]:
                pl.append(dict(hold_days=3, risk_pct=rp, otm_pct=otm,
                               min_score=.3, min_consensus=mc, min_str=.4,
                               liq_disc=.85))

    # D: ATM + 强共识 + 长持有
    for hd in [7, 10]:
        for rp in [.04, .05]:
            for mc in [4, 5]:
                pl.append(dict(hold_days=hd, risk_pct=rp, otm_pct=0,
                               min_score=.5, min_consensus=mc, min_str=.6,
                               hv_hi=.50))

    # E: 高HV品种 + 远OTM (高杠杆)
    for hd in [3, 5]:
        for rp in [.04, .05]:
            for otm in [.02, .03, .04]:
                pl.append(dict(hold_days=hd, risk_pct=rp, otm_pct=otm,
                               min_score=.3, min_consensus=3, min_str=.4,
                               hv_lo=.20, hv_hi=.80))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 100 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'H':>3} {'R':>4} {'OTM':>4} {'Con':>3} {'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*80)
    for r in res[:60]:
        print(f"{r['hold_days']:>3} {r['risk_pct']:>4.0%} {r['otm_pct']*100:>4.0f} "
              f"{r['min_consensus']:>3} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*80)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),(3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),(6.,.45,"年化>=600% & WR>=45%"),
                       (3.,.45,"年化>=300% & WR>=45%"),(.5,.50,"年化>=50% & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  H={r['hold_days']}  R={r['risk_pct']:.0%}  "
                      f"OTM={r['otm_pct']*100:.0f}%  Con={r['min_consensus']}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:300]]
    with open(os.path.join(od,'backtest_v48.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v48.json")


if __name__ == '__main__':
    main()
