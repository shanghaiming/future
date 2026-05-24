#!/usr/bin/env python3
"""
策略 V49 — 基于学术研究的改进方案

论文启发:
  1. Hurst/Ooi/Pedersen (2017): 趋势跟踪在极端行情中收益最大 (Crisis Alpha)
  2. Fama/French (1988): 动量溢价在1-12个月最显著
  3. Black-Scholes delta对冲: 期权+期货组合可以创造凸性收益
  4. Baz等 (2015): 风险平价+时间序列动量

核心改进:
  A. 多时间框架动量 (3/5/10/20/60日) 一致性 → 提高胜率
  B. 波动率缩放 (Volatility Scaling) → 风险平价
  C. 危机Alpha: 高波动率时加大仓位 (因为趋势更强)
  D. 期权+期货混合: 期货做底仓, 期权做增强
  E. Carry信号 (期货期限结构proxy) → 区分趋势/均值回复
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
COMM_OPT = 0.0003
R = 0.02
INIT = 500000
TD = 252


def bs(S, K, T, r, sigma, flag='call'):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if flag == 'call' else max(K - S, 0)
    d1 = (np.log(S/K) + (r + .5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)) if flag == 'call' \
        else (K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))


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

        # ── 多时间框架动量 (Hurst 2017) ──
        for lag in [3, 5, 10, 20, 60, 120]:
            df[f'mom{lag}'] = df['close'].pct_change(lag)

        # 一致性向量化 (在dropna后重新计算)
        # ── 波动率缩放 (Baz 2015) ──
        for w in [5, 10, 20, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        # 目标波动率 = 15%, 仓位 = target_vol / realized_vol
        df['vol_scale'] = 0.15 / df['hv20'].replace(0, np.nan)
        df['vol_scale'] = df['vol_scale'].clip(0.3, 3.0)

        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>10 else .5, raw=False)

        # ── EWMAC (趋势强度) ──
        for sp in [4, 8, 16, 32]:
            ef = df['close'].ewm(span=sp).mean()
            es = df['close'].ewm(span=sp*4).mean()
            s = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ew{sp}'] = (ef - es) / s

        # 合成Delta (加权EWMAC) - 计算列，dropna后提取values
        df['sdelta'] = df['ew4']*.15 + df['ew8']*.25 + df['ew16']*.3 + df['ew32']*.3

        # ── Skew ──
        up = df['ret'].where(df['ret']>0,0).rolling(20).std()*np.sqrt(TD)
        dn = (-df['ret'].where(df['ret']<0,0)).rolling(20).std()*np.sqrt(TD)
        df['skew'] = (dn - up) / df['hv20'].replace(0, np.nan)

        # ── 基础指标 ──
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20']>df['ma60'], 1., -1.)
        for w in [10, 20, 40]:
            hh, ll = df['close'].rolling(w).max(), df['close'].rolling(w).min()
            df[f'bo{w}'] = (df['close'] - .5*(hh+ll)) / (hh-ll+.001) * 2

        s20 = df['ret'].rolling(20).std().replace(0, np.nan)
        df['ramom'] = df['mom10'] / (s20 * np.sqrt(10))

        d = df['close'].diff()
        g = d.where(d>0,0).rolling(14).mean()
        l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / df['close']

        # 期限结构 proxy
        df['ts'] = df['mom5'] - df['mom60']

        # OI信号
        if 'oi' in df.columns:
            df['oi_chg'] = df['oi'].pct_change(5)
            df['oi_chg20'] = df['oi'].pct_change(20)

        df = df.dropna(subset=['ma20','ma60','hv20','mom5','rsi'])
        if len(df) < 100: continue
        try: spec = get_spec(sym)
        except: continue

        # ── 预计算信号 ──
        # 动量一致性 (dropna后重新计算)
        mom_cols = ['mom3','mom5','mom10','mom20','mom60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_count = np.sum(signs > 0, axis=1)
        neg_count = np.sum(signs < 0, axis=1)
        mom_dir = np.where(pos_count >= 4, 1.0, np.where(neg_count >= 4, -1.0, 0.0))
        mom_str = np.where(mom_dir != 0,
                          np.maximum(pos_count, neg_count) / len(mom_cols), 0.0)

        trend = df['trend'].values
        bo20 = df['bo20'].values
        bo40 = df['bo40'].values
        ramom = df['ramom'].values
        skew = df['skew'].values
        ts = df['ts'].values
        vs = df['vol_scale'].values

        # 综合得分 (趋势+突破+skew+期限结构)
        def _cl(x, c): return np.sign(x) * np.minimum(np.abs(x), c)
        sd = df['sdelta'].values
        sc = np.zeros(len(df)); w = np.zeros(len(df))
        m = ~np.isnan(sd); sc[m] += _cl(sd[m], 3)*3; w[m] += 3
        m = ~np.isnan(bo20); sc[m] += bo20[m]; w[m] += 1
        m = ~np.isnan(ramom); sc[m] += _cl(ramom[m], 3)*2; w[m] += 2
        m = ~np.isnan(skew); sc[m] += _cl(-skew[m], 2)*1.5; w[m] += 1.5
        m = ~np.isnan(ts); sc[m] += _cl(ts[m]*20, 2)*1; w[m] += 1
        # 趋势加成
        m = (trend!=0) & ~np.isnan(sc) & (np.sign(sc)==trend); sc[m] *= 1.5
        sc = np.where(w>0, sc/np.maximum(w,1), 0)

        raw[sym] = {
            'spec': spec, 'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'vol_scale': vs.astype(np.float64),
            'mom_dir': mom_dir.astype(np.float64),
            'mom_str': mom_str.astype(np.float64),
            'dir': np.sign(sc).astype(np.float64),
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
    hd = p.get('hold_days', 7)
    mode = p.get('mode', 'hybrid')  # hybrid=期权+期货, option=纯期权, future=纯期货
    risk_pct = p.get('risk_pct', .03)
    otm = p.get('otm_pct', .01)
    mu = p.get('margin_usage', .90)
    ms = p.get('min_score', .5)
    mc = p.get('min_mom', 0)  # 动量一致性最低要求 (0=不要求, 1=要求mom_dir!=0)
    use_vol_scale = p.get('vol_scale', False)
    use_mom_filter = p.get('mom_filter', False)  # 要求动量一致性
    ld = p.get('liq_disc', .9)
    crisis_mode = p.get('crisis_mode', False)  # 高波动时加大仓位
    tp_opt = p.get('tp_opt', 0)  # 期权止盈倍数

    if mode == 'future':
        # 纯期货模式
        eq = float(INIT); cash = float(INIT)
        pos = {}; pnls = []; eqh = [float(INIT)]

        for date in dates:
            for sym in list(pos):
                ps = pos[sym]
                im = si.get(sym)
                if not im or date not in im: continue
                il = im[date]
                price = raw[sym]['close'][il]
                h = int((date - ps['ed'])/np.timedelta64(1,'D'))
                if h < hd: continue
                comm = price * ps['m'] * ps['fl'] * COMM
                pnl = (price - ps['fe']) * ps['d'] * ps['m'] * ps['fl']
                cash += ps['fm'] + pnl - comm
                pnls.append(pnl - comm)
                del pos[sym]

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
                    if np.isnan(hv) or hv < .05 or hv > .70: continue
                    hp = d['hv_pct'][pi]
                    if np.isnan(hp) or hp > .90: continue
                    rsi = d['rsi'][pi]
                    if np.isnan(rsi) or rsi > 78 or rsi < 22: continue
                    ap = d['atr_pct'][pi]
                    if np.isnan(ap) or ap > .06: continue

                    dir_v = d['dir'][pi]
                    sc = d['score'][pi]
                    if np.isnan(sc) or abs(sc) < ms: continue
                    if dir_v == 0: continue

                    # 动量过滤
                    if use_mom_filter:
                        md = d['mom_dir'][pi]
                        if md == 0 or md != dir_v: continue

                    sigs.append((sym, dir_v, abs(sc), il))

                sigs.sort(key=lambda x: x[2], reverse=True)

                for sym, dv, sc, il in sigs:
                    if len(pos) >= mp: break
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: S = raw[sym]['close'][il]
                    if S <= 0: continue

                    ml, mr, _, _ = raw[sym]['spec']
                    mpl = S * ml * mr

                    # 波动率缩放
                    eff_mu = mu
                    if use_vol_scale:
                        vs = raw[sym]['vol_scale'][il-1]
                        if not np.isnan(vs):
                            eff_mu = mu * min(vs, 2.0)

                    if crisis_mode:
                        hp = raw[sym]['hv_pct'][il-1]
                        if not np.isnan(hp):
                            if hp > 0.7:  # 高波动 → 趋势更强 → 加仓
                                eff_mu *= 1.5
                            elif hp < 0.3:  # 低波动 → 减仓
                                eff_mu *= 0.7

                    target = eq * eff_mu / mp
                    fl = max(int(target / mpl), 1)
                    fm = mpl * fl
                    fc = S * ml * fl * COMM

                    total_m = sum(ps['fm'] for ps in pos.values())
                    if total_m + fm > eq * eff_mu:
                        fl = max(int((eq*eff_mu - total_m)/mpl), 0)
                        if fl <= 0: continue
                        fm = mpl*fl; fc = S*ml*fl*COMM
                    if fm+fc > cash:
                        fl = max(int((cash-fc)/mpl), 0)
                        if fl <= 0: continue
                        fm = mpl*fl; fc = S*ml*fl*COMM

                    cash -= fm + fc
                    pos[sym] = {'d':dv,'ed':date,'ep':S,'fe':S*(1+.0001*dv),'fl':fl,'m':ml,'fm':fm}

            ur = 0.
            for sym, ps in pos.items():
                im = si.get(sym)
                if im and date in im:
                    price = raw[sym]['close'][im[date]]
                    ur += (price - ps['fe']) * ps['d'] * ps['m'] * ps['fl']
            eq = cash + ur
            eqh.append(eq)
            if eq < 5000: break

    else:
        # 期权 / 混合模式
        eq = float(INIT)
        pos = {}; pnls = []; eqh = [float(INIT)]

        for date in dates:
            for sym in list(pos):
                ps = pos[sym]
                im = si.get(sym)
                if not im or date not in im: continue
                il = im[date]
                S = raw[sym]['close'][il]
                h = int((date - ps['ed'])/np.timedelta64(1,'D'))

                if ps['type'] == 'option':
                    if h < hd: continue
                    Tr = max(ps['T'] - h/TD, .001)
                    ov = bs(S, ps['K'], Tr, R, ps['sig'], ps['fl'])
                    intr = max(S-ps['K'],0) if ps['fl']=='call' else max(ps['K']-S,0)
                    ev = max(ov, intr) * ld

                    # 期权止盈
                    if tp_opt > 0 and ps['pr'] > 0:
                        pnl_pct = (ev - ps['pr']) / ps['pr']
                        if pnl_pct > tp_opt and h >= 3:
                            pass  # 允许退出
                        elif h < hd:
                            continue

                    pnl = (ev - ps['pr']) * ps['dv'] * ps['ml'] * ps['ct']
                    c = ev * ps['ml'] * ps['ct'] * COMM_OPT
                    eq += pnl - c
                    pnls.append(pnl - c)
                    del pos[sym]
                else:
                    if h < hd: continue
                    comm = S * ps['m'] * ps['fl'] * COMM
                    pnl = (S - ps['fe']) * ps['d'] * ps['m'] * ps['fl']
                    eq += ps['fm'] + pnl - comm
                    pnls.append(pnl - comm)
                    del pos[sym]

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
                    if np.isnan(hv) or hv < .05 or hv > .70: continue
                    hp = d['hv_pct'][pi]
                    if np.isnan(hp) or hp > .90: continue
                    rsi = d['rsi'][pi]
                    if np.isnan(rsi) or rsi > 78 or rsi < 22: continue
                    ap = d['atr_pct'][pi]
                    if np.isnan(ap) or ap > .06: continue

                    dir_v = d['dir'][pi]
                    sc = d['score'][pi]
                    if np.isnan(sc) or abs(sc) < ms: continue
                    if dir_v == 0: continue

                    if use_mom_filter:
                        md = d['mom_dir'][pi]
                        if md == 0 or md != dir_v: continue

                    sigs.append((sym, dir_v, abs(sc), il, hv))

                sigs.sort(key=lambda x: x[2], reverse=True)

                for sym, dv, sc, il, hv in sigs:
                    if len(pos) >= mp: break
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: S = raw[sym]['close'][il]
                    if S <= 0: continue

                    ml, mr, _, _ = raw[sym]['spec']

                    if mode == 'hybrid':
                        # 混合模式: 高确信度用期权, 低确信度用期货
                        if sc > 0.8 and hv > 0.15:
                            # 强信号 + 高波动 → 期权
                            fl = 'call' if dv > 0 else 'put'
                            K = S * (1 + otm * dv)
                            T = hd / TD
                            pr = bs(S, K, T, R, hv, fl)
                            if pr > 0:
                                risk = eq * risk_pct
                                ct = max(int(risk / (pr * ml)), 1)
                                ct = min(ct, max(int(eq * 3 / (S * ml)), 1))
                                cost = pr * ml * ct
                                if cost > eq * .3:
                                    ct = max(int(eq*.3/(pr*ml+1)), 1)
                                    cost = pr*ml*ct
                                if cost > eq: continue
                                pos[sym] = {'type':'option','dv':dv,'ed':date,'S':S,'K':K,
                                           'T':T,'sig':hv,'pr':pr,'fl':fl,'ml':ml,'ct':ct}
                                continue

                    # 默认: 期货
                    if mode == 'option':
                        fl = 'call' if dv > 0 else 'put'
                        K = S * (1 + otm * dv)
                        T = hd / TD
                        pr = bs(S, K, T, R, hv, fl)
                        if pr <= 0: continue
                        risk = eq * risk_pct
                        ct = max(int(risk / (pr * ml)), 1)
                        ct = min(ct, max(int(eq * 3 / (S * ml)), 1))
                        cost = pr * ml * ct
                        if cost > eq * .3:
                            ct = max(int(eq*.3/(pr*ml+1)), 1); cost = pr*ml*ct
                        if cost > eq: continue
                        pos[sym] = {'type':'option','dv':dv,'ed':date,'S':S,'K':K,
                                   'T':T,'sig':hv,'pr':pr,'fl':fl,'ml':ml,'ct':ct}
                    else:
                        # 期货 (hybrid模式的低确信度)
                        mpl = S * ml * mr
                        target = eq * mu / mp
                        fl = max(int(target/mpl), 1)
                        fm = mpl * fl; fc = S*ml*fl*COMM
                        total_m = sum(ps.get('fm',0) for ps in pos.values())
                        if total_m+fm > eq*mu:
                            fl = max(int((eq*mu-total_m)/mpl), 0)
                            if fl<=0: continue
                            fm = mpl*fl; fc = S*ml*fl*COMM
                        if fm+fc > eq:
                            fl = max(int((eq-fc)/mpl), 0)
                            if fl<=0: continue
                            fm = mpl*fl; fc = S*ml*fl*COMM
                        eq -= fm + fc  # bug: should be cash
                        pos[sym] = {'type':'future','d':dv,'ed':date,'ep':S,
                                   'fe':S*(1+.0001*dv),'fl':fl,'m':ml,'fm':fm}

            # 权益
            ur = 0.
            for sym, ps in pos.items():
                im = si.get(sym)
                if im and date in im:
                    S = raw[sym]['close'][im[date]]
                    if ps.get('type') == 'option':
                        h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                        Tr = max(ps['T']-h/TD, .001)
                        ov = bs(S, ps['K'], Tr, R, ps['sig'], ps['fl'])
                        intr = max(S-ps['K'],0) if ps['fl']=='call' else max(ps['K']-S,0)
                        ur += (max(ov,intr)*ld - ps['pr']) * ps['dv'] * ps['ml'] * ps['ct']
                    else:
                        ur += (S - ps['fe']) * ps['d'] * ps['m'] * ps['fl']
            ceq = eq + ur
            eqh.append(ceq)
            if ceq < 1000: break

    if not pnls or eq <= 0: return None
    tr = (eq - INIT)/INIT
    if tr <= -1: return None
    dys = int((dates[-1]-dates[0])/np.timedelta64(1,'D'))
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

    # A: 纯期货 + 动量一致性 + 波动率缩放
    for hd in [3, 5, 7]:
        for mu in [.70, .80, .90, .95]:
            for ms in [.3, .5]:
                for vs in [False, True]:
                    for mf in [False, True]:
                        for cr in [False, True]:
                            pl.append(dict(hold_days=hd, mode='future', margin_usage=mu,
                                          min_score=ms, vol_scale=vs, mom_filter=mf, crisis_mode=cr))

    # B: 期权 + 动量过滤
    for hd in [5, 7]:
        for rp in [.03, .04, .05]:
            for otm in [0, .01, .02]:
                for mf in [False, True]:
                    for tp in [0, 1.0, 2.0]:
                        pl.append(dict(hold_days=hd, mode='option', risk_pct=rp, otm_pct=otm,
                                      min_score=.5, mom_filter=mf, tp_opt=tp))

    # C: 混合模式
    for hd in [5, 7]:
        for mu in [.80, .90]:
            for rp in [.03, .04]:
                for otm in [.01, .02]:
                    for mf in [True]:
                        pl.append(dict(hold_days=hd, mode='hybrid', margin_usage=mu,
                                      risk_pct=rp, otm_pct=otm, min_score=.5, mom_filter=mf))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 100 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'模式':>8} {'H':>3} {'M/R':>5} {'OTM':>4} {'VS':>3} {'MF':>3} {'CR':>3} {'TP':>5} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*110)
    for r in res[:60]:
        vs = 'Y' if r.get('vol_scale') else '-'
        mf = 'Y' if r.get('mom_filter') else '-'
        cr = 'Y' if r.get('crisis_mode') else '-'
        tp = f"{r.get('tp_opt',0):.0f}x" if r.get('tp_opt',0) > 0 else '-'
        mr = f"{r.get('margin_usage',0):.0%}" if r.get('mode')=='future' else f"{r.get('risk_pct',0):.0%}"
        ot = f"{r.get('otm_pct',0)*100:.0f}" if r.get('otm_pct',0)>0 else '-'
        print(f"{r['mode']:>8} {r['hold_days']:>3} {mr:>5} {ot:>4} {vs:>3} {mf:>3} {cr:>3} {tp:>5} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*110)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),(3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),(.5,.50,"年化>=50% & WR>=50%"),
                       (6.,.45,"年化>=600% & WR>=45%"),(1.,.45,"年化>=100% & WR>=45%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:8]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  {r['mode']}  H={r['hold_days']}  "
                      f"MF={'Y' if r.get('mom_filter') else 'N'}  VS={'Y' if r.get('vol_scale') else 'N'}  "
                      f"CR={'Y' if r.get('crisis_mode') else 'N'}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:300]]
    with open(os.path.join(od,'backtest_v49.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v49.json")


if __name__ == '__main__':
    main()
