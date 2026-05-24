#!/usr/bin/env python3
"""
策略 V50 — 多层策略叠加 (基于学术论文)

论文启发:
  1. Fuertes/Miffre/Rallis (2010): 动量+期限结构+特质波动率三因子
  2. University of Vaasa: 周度卖出跨式 ~70%胜率
  3. Baz等 (2015): 风险平价+波动率缩放
  4. Kelly Criterion: f* = (bp-q)/b 最优仓位
  5. SSRN (2024): 卖出波动率获取风险溢价

三层叠加:
  Layer 1: 方向性交易 — 动量+趋势+skew信号 (期货+期权混合)
  Layer 2: 波动率卖出 — IV>HV时卖出跨式/宽跨式 (高胜率基础收益)
  Layer 3: Kelly仓位 — 根据历史胜率和赔率动态调整

目标: 年化600%, 最大持仓3, 胜率>50%
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

        # 多时间框架动量
        for lag in [3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        # HV多窗口
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>10 else .5, raw=False)

        # EWMAC
        for sp in [4, 8, 16]:
            ef = df['close'].ewm(span=sp).mean()
            es = df['close'].ewm(span=sp*4).mean()
            s = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ew{sp}'] = (ef - es) / s
        df['sdelta'] = df['ew4']*.2 + df['ew8']*.3 + df['ew16']*.3

        # Skew
        up = df['ret'].where(df['ret']>0,0).rolling(20).std()*np.sqrt(TD)
        dn = (-df['ret'].where(df['ret']<0,0)).rolling(20).std()*np.sqrt(TD)
        df['skew'] = (dn - up) / df['hv20'].replace(0, np.nan)

        # 基础
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20']>df['ma60'], 1., -1.)
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
        df['ts'] = df['m5'] - df['m60']

        # 波动率缩放
        df['vol_scale'] = (0.15 / df['hv20'].replace(0, np.nan)).clip(0.3, 3.0)

        # IV vs HV proxy: 用HV5/HV20比率作为短期IV溢价指标
        # 当HV5 >> HV20时 → 近期波动率飙升 → 类似IV溢价
        df['iv_premium'] = (df['hv5'] / df['hv20'] - 1).clip(-0.5, 0.5)

        df = df.dropna(subset=['ma20','ma60','hv20','m5','rsi','sdelta'])
        if len(df) < 100: continue
        try: spec = get_spec(sym)
        except: continue

        # 预计算信号
        sd = df['sdelta'].values
        trend = df['trend'].values
        bo20 = df['bo20'].values
        ramom = df['ramom'].values
        skew = df['skew'].values
        ts = df['ts'].values

        # 动量一致性
        mom_cols = ['m3','m5','m10','m20','m60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_count = np.sum(signs > 0, axis=1)
        neg_count = np.sum(signs < 0, axis=1)
        mom_dir = np.where(pos_count >= 4, 1.0, np.where(neg_count >= 4, -1.0, 0.0))
        mom_str = np.where(mom_dir != 0, np.maximum(pos_count, neg_count) / len(mom_cols), 0.0)

        # 综合得分
        def _cl(x, c): return np.sign(x) * np.minimum(np.abs(x), c)
        sc = np.zeros(len(df)); w = np.zeros(len(df))
        m = ~np.isnan(sd); sc[m] += _cl(sd[m], 3)*3; w[m] += 3
        m = ~np.isnan(bo20); sc[m] += bo20[m]; w[m] += 1
        m = ~np.isnan(ramom); sc[m] += _cl(ramom[m], 3)*2; w[m] += 2
        m = ~np.isnan(skew); sc[m] += _cl(-skew[m], 2)*1.5; w[m] += 1.5
        m = ~np.isnan(ts); sc[m] += _cl(ts[m]*20, 2); w[m] += 1
        m = (trend!=0) & ~np.isnan(sc) & (np.sign(sc)==trend); sc[m] *= 1.5
        sc = np.where(w>0, sc/np.maximum(w,1), 0)

        # 波动率卖出信号: HV percentile > 0.7 → 卖出波动率的好时机
        # 用卖straddle/strangle的方式
        vol_sell = df['hv_pct'].values > 0.6  # 高波动率百分位

        raw[sym] = {
            'spec': spec, 'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'hv5': df['hv5'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'vol_scale': df['vol_scale'].values.astype(np.float64),
            'iv_premium': df['iv_premium'].values.astype(np.float64),
            'mom_dir': mom_dir.astype(np.float64),
            'mom_str': mom_str.astype(np.float64),
            'dir': np.sign(sc).astype(np.float64),
            'score': sc.astype(np.float64),
            'vol_sell': vol_sell.astype(np.bool_),
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


def kelly_fraction(wr, avg_win, avg_loss):
    """Kelly Criterion: f* = (bp - q) / b"""
    if avg_loss <= 0 or avg_win <= 0: return 0.02
    b = avg_win / avg_loss  # odds
    q = 1 - wr
    f = (wr * b - q) / b
    return max(min(f, 0.25), 0.01)  # 限制1-25%


def bt(raw, dates, si, p):
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 7)
    mode = p.get('mode', 'multi')  # multi/directional/vol_sell/kelly
    risk_pct = p.get('risk_pct', .03)
    otm = p.get('otm_pct', .01)
    mu = p.get('margin_usage', .90)
    ms = p.get('min_score', .5)
    ld = p.get('liq_disc', .9)
    use_mom = p.get('use_mom', True)
    use_vol_scale = p.get('vol_scale', False)
    straddle_width = p.get('straddle_width', 0.03)  # 跨式宽度 (3% OTM each side)

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]
    trade_history = []  # (pnl, date) for Kelly calculation

    for date in dates:
        # 退出
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if not im or date not in im: continue
            il = im[date]
            S = raw[sym]['close'][il]
            h = int((date - ps['ed'])/np.timedelta64(1,'D'))

            if ps.get('type') == 'straddle':
                # 卖出跨式退出: 收益 = 收到的权利金 - 当前跨式价值
                if h < hd: continue
                Tr = max(ps['T'] - h/TD, .001)
                call_val = bs(S, ps['K_call'], Tr, R, ps['sig'], 'call')
                put_val = bs(S, ps['K_put'], Tr, R, ps['sig'], 'put')
                exit_cost = (call_val + put_val) * ld
                # PnL = 收到的权利金 - 退出成本
                pnl = (ps['prem_received'] - exit_cost) * ps['ml'] * ps['ct']
                c = exit_cost * ps['ml'] * ps['ct'] * COMM_OPT
                eq += pnl - c
                pnls.append(pnl - c)
                trade_history.append((pnl - c, date))
                del pos[sym]
            elif ps.get('type') == 'option':
                if h < hd: continue
                Tr = max(ps['T'] - h/TD, .001)
                ov = bs(S, ps['K'], Tr, R, ps['sig'], ps['fl'])
                intr = max(S-ps['K'],0) if ps['fl']=='call' else max(ps['K']-S,0)
                ev = max(ov, intr) * ld
                pnl = (ev - ps['pr']) * ps['dv'] * ps['ml'] * ps['ct']
                c = ev * ps['ml'] * ps['ct'] * COMM_OPT
                eq += pnl - c
                pnls.append(pnl - c)
                trade_history.append((pnl - c, date))
                del pos[sym]
            else:  # future
                if h < hd: continue
                comm = S * ps['m'] * ps['fl'] * COMM
                pnl = (S - ps['fe']) * ps['d'] * ps['m'] * ps['fl']
                eq += ps['fm'] + pnl - comm
                pnls.append(pnl - comm)
                trade_history.append((pnl - comm, date))
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
                if np.isnan(hv) or hv < .05 or hv > .70: continue
                hp = d['hv_pct'][pi]
                if np.isnan(hp) or hp > .92: continue
                rsi = d['rsi'][pi]
                if np.isnan(rsi) or rsi > 78 or rsi < 22: continue
                ap = d['atr_pct'][pi]
                if np.isnan(ap) or ap > .06: continue

                dir_v = d['dir'][pi]
                sc = d['score'][pi]
                md = d['mom_dir'][pi]

                # 根据模式决定交易类型
                if mode == 'vol_sell':
                    # 纯波动率卖出: 高HV时卖出跨式
                    if not d['vol_sell'][pi]: continue
                    if hv < 0.20: continue  # 波动率太低不值得卖
                    sigs.append(('straddle', sym, 1.0, il, hv, hp))
                    continue

                if mode == 'directional':
                    if dir_v == 0: continue
                    if abs(sc) < ms: continue
                    if use_mom and (md == 0 or md != dir_v): continue
                    sigs.append(('option', sym, dir_v, abs(sc), il, hv, hp))
                    continue

                # multi模式: 综合判断
                if dir_v != 0 and abs(sc) >= ms:
                    if not use_mom or (md != 0 and md == dir_v):
                        # 有方向信号
                        if sc > 0.7 and hv > 0.15:
                            # 强信号+高波动 → 期权
                            sigs.append(('option', sym, dir_v, abs(sc), il, hv, hp))
                        else:
                            # 中等信号 → 期货
                            vs = d['vol_scale'][pi]
                            sigs.append(('future', sym, dir_v, abs(sc), il, hv, hp, vs))

                # 波动率卖出机会 (与方向性交易独立)
                if d['vol_sell'][pi] and hv > 0.25:
                    if len(sigs) < mp * 2:  # 不超过总候选数的2倍
                        sigs.append(('straddle', sym, 0.5, il, hv, hp))

            # 排序: option > straddle > future, 按score
            type_prio = {'option': 3, 'straddle': 2, 'future': 1}
            sigs.sort(key=lambda x: (type_prio.get(x[0],0), x[3] if len(x)>4 else 0), reverse=True)

            for sig in sigs:
                if len(pos) >= mp: break
                stype = sig[0]
                sym = sig[1]

                if stype == 'straddle':
                    _, sym, score, il, hv, hp = sig
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    T = hd / TD
                    K_call = S * (1 + straddle_width)
                    K_put = S * (1 - straddle_width)
                    call_prem = bs(S, K_call, T, R, hv, 'call')
                    put_prem = bs(S, K_put, T, R, hv, 'put')
                    total_prem = call_prem + put_prem
                    if total_prem <= 0: continue

                    # Kelly sizing for vol selling
                    k = 0.04  # default
                    if len(trade_history) > 50:
                        recent = [t[0] for t in trade_history[-200:]]
                        pa = np.array(recent)
                        wr = (pa > 0).mean()
                        if (pa > 0).any() and (pa <= 0).any():
                            aw = pa[pa>0].mean()
                            al = abs(pa[pa<=0].mean())
                            k = kelly_fraction(wr, aw, al) * 3  # 3x Kelly for vol selling

                    risk = eq * min(k, 0.08)
                    ct = max(int(risk / (total_prem * ml)), 1)
                    ct = min(ct, max(int(eq * 2 / (S * ml)), 1))
                    cost = total_prem * ml * ct
                    if cost > eq * .3:
                        ct = max(int(eq*.3/(total_prem*ml+1)), 1)
                        cost = total_prem*ml*ct
                    if cost > eq: continue

                    pos[sym] = {'type':'straddle','ed':date,'S':S,
                               'K_call':K_call,'K_put':K_put,
                               'T':T,'sig':hv,'prem_received':total_prem,
                               'ml':ml,'ct':ct}

                elif stype == 'option':
                    _, sym, dv, sc, il, hv, hp = sig
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    fl = 'call' if dv > 0 else 'put'
                    K = S * (1 + otm * dv)
                    T = hd / TD
                    pr = bs(S, K, T, R, hv, fl)
                    if pr <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    risk = eq * risk_pct
                    ct = max(int(risk / (pr * ml)), 1)
                    ct = min(ct, max(int(eq * 3 / (S * ml)), 1))
                    cost = pr * ml * ct
                    if cost > eq * .3:
                        ct = max(int(eq*.3/(pr*ml+1)), 1); cost = pr*ml*ct
                    if cost > eq: continue
                    pos[sym] = {'type':'option','dv':dv,'ed':date,'S':S,'K':K,
                               'T':T,'sig':hv,'pr':pr,'fl':fl,'ml':ml,'ct':ct}

                elif stype == 'future':
                    _, sym, dv, sc, il, hv, hp, vs = sig
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    mpl = S * ml * mr
                    eff_mu = mu
                    if use_vol_scale and not np.isnan(vs):
                        eff_mu = mu * min(vs, 2.0)
                    target = eq * eff_mu / mp
                    fl = max(int(target/mpl), 1)
                    fm = mpl * fl; fc = S*ml*fl*COMM
                    if fm+fc > eq * 0.95:
                        fl = max(int((eq*0.95-fc)/mpl), 0)
                        if fl <= 0: continue
                        fm = mpl*fl; fc = S*ml*fl*COMM
                    eq -= fm + fc
                    pos[sym] = {'type':'future','d':dv,'ed':date,'ep':S,
                               'fe':S*(1+.0001*dv),'fl':fl,'m':ml,'fm':fm}

        # 权益
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                if ps.get('type') == 'straddle':
                    Tr = max(ps['T']-h/TD, .001)
                    cv = bs(S, ps['K_call'], Tr, R, ps['sig'], 'call')
                    pv = bs(S, ps['K_put'], Tr, R, ps['sig'], 'put')
                    ur += (ps['prem_received'] - (cv+pv)*ld) * ps['ml'] * ps['ct']
                elif ps.get('type') == 'option':
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

    n_straddle = sum(1 for k,v in [('s',x) for x in []] if False)  # placeholder
    return {'annual':ann,'wr':wr,'mdd':mdd,'pf':pf,'trades':len(pa),'final':eq,'sharpe':sh,**p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载..."); t0=time.time()
    raw = load(dd); print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")
    sd = pd.Timestamp('2018-01-01'); ed = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, sd, ed); print(f"  {len(dates)}交易日")

    pl = []

    # A: 多层叠加
    for hd in [5, 7, 10]:
        for mu in [.80, .90]:
            for otm in [0, .01, .02]:
                for sw in [.02, .03, .04]:
                    for ms in [.3, .5]:
                        pl.append(dict(hold_days=hd, mode='multi', margin_usage=mu,
                                      otm_pct=otm, straddle_width=sw, min_score=ms,
                                      use_mom=True, vol_scale=True))

    # B: 纯方向性 (对照)
    for hd in [5, 7]:
        for rp in [.03, .04]:
            for otm in [0, .01, .02]:
                pl.append(dict(hold_days=hd, mode='directional', risk_pct=rp,
                              otm_pct=otm, min_score=.5, use_mom=True))

    # C: 纯波动率卖出
    for hd in [5, 7, 10]:
        for sw in [.02, .03, .04, .05]:
            pl.append(dict(hold_days=hd, mode='vol_sell', straddle_width=sw))

    # D: 多层 + 无动量过滤
    for hd in [5, 7]:
        for mu in [.80, .90]:
            for otm in [.01, .02]:
                pl.append(dict(hold_days=hd, mode='multi', margin_usage=mu,
                              otm_pct=otm, straddle_width=.03, min_score=.3,
                              use_mom=False, vol_scale=False))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 50 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'模式':>12} {'H':>3} {'M/R':>5} {'OTM':>4} {'SW':>4} {'VS':>3} {'MF':>3} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*100)
    for r in res[:60]:
        vs = 'Y' if r.get('vol_scale') else '-'
        mf = 'Y' if r.get('use_mom') else '-'
        mr = f"{r.get('margin_usage',0):.0%}" if r.get('mode')!='directional' else f"{r.get('risk_pct',0):.0%}"
        ot = f"{r.get('otm_pct',0)*100:.0f}"
        sw = f"{r.get('straddle_width',0)*100:.0f}" if r.get('straddle_width',0)>0 else '-'
        print(f"{r['mode']:>12} {r['hold_days']:>3} {mr:>5} {ot:>4} {sw:>4} {vs:>3} {mf:>3} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*100)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),(3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),(.5,.50,"年化>=50% & WR>=50%"),
                       (6.,.45,"年化>=600% & WR>=45%"),(1.,.45,"年化>=100% & WR>=45%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:8]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  {r['mode']}  H={r['hold_days']}  "
                      f"OTM={r.get('otm_pct',0)*100:.0f}%  SW={r.get('straddle_width',0)*100:.0f}%  "
                      f"MF={'Y' if r.get('use_mom') else 'N'}  VS={'Y' if r.get('vol_scale') else 'N'}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:300]]
    with open(os.path.join(od,'backtest_v50.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v50.json")


if __name__ == '__main__':
    main()
