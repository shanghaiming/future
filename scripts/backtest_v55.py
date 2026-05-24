#!/usr/bin/env python3
"""
策略 V55 — 期权期货融合 v2 (修正)

V54失败: slot分配导致交易太少(44笔), 全亏损
修正:
  1. 不分slot, 最优信号竞争3个仓位
  2. 期权信号(跨式卖出)优先占位
  3. 期货信号(方向交易)只填剩余仓位
  4. 期货只在极强动量一致性(5/5)时入场
  5. 保持V52的高杠杆参数

核心收益来源: 期权跨式卖出 (76%WR, 已验证)
期货增强: 强趋势时额外加仓
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

        for lag in [3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>10 else .5, raw=False)

        for sp in [4, 8, 16]:
            ef = df['close'].ewm(span=sp).mean()
            es = df['close'].ewm(span=sp*4).mean()
            s = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ew{sp}'] = (ef - es) / s

        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20']>df['ma60'], 1., -1.)

        mom_cols = ['m3','m5','m10','m20','m60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_c = np.sum(signs > 0, axis=1)
        neg_c = np.sum(signs < 0, axis=1)
        df['mom_dir'] = np.where(pos_c >= 4, 1.0, np.where(neg_c >= 4, -1.0, 0.0))
        # 极强信号: 5/5一致
        df['mom_strong'] = np.where((pos_c >= 5) | (neg_c >= 5), True, False)

        d = df['close'].diff(); g = d.where(d>0,0).rolling(14).mean()
        l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))
        tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / df['close']

        df = df.dropna(subset=['ma20','ma60','hv20','m5','rsi'])
        if len(df) < 100: continue
        try: spec = get_spec(sym)
        except: continue

        raw[sym] = {
            'spec': spec, 'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'mom_dir': df['mom_dir'].values.astype(np.float64),
            'mom_strong': df['mom_strong'].values.astype(np.bool_),
            'trend': df['trend'].values.astype(np.float64),
            'vol_sell': (df['hv_pct'].values > 0.6).astype(np.bool_),
            'm5': df['m5'].values.astype(np.float64),
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
    if avg_loss <= 0 or avg_win <= 0: return 0.03
    b = avg_win / avg_loss
    q = 1 - wr
    f = (wr * b - q) / b
    return max(min(f, 0.25), 0.01)


def bt(raw, dates, si, p):
    mp = p.get('max_pos', 3)
    hd_opt = p.get('hold_days_opt', 7)
    hd_fut = p.get('hold_days_fut', 5)
    straddle_width = p.get('straddle_width', 0.02)
    ld = p.get('liq_disc', .9)
    kelly_mult = p.get('kelly_mult', 3.0)
    kelly_cap = p.get('kelly_cap', 0.08)
    notional_mult = p.get('notional_mult', 2.0)
    mu_fut = p.get('margin_usage_fut', .90)
    bias_shift = p.get('bias_shift', 0.0)
    hv_min_opt = p.get('hv_min_opt', 0.20)
    hv_pct_min = p.get('hv_pct_min', 0.55)
    fut_strength = p.get('fut_strength', 'strong')  # strong/very_strong
    max_fut_pos = p.get('max_fut_pos', 1)  # 最多期货仓位

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]
    trade_history = []

    for date in dates:
        # === 退出 ===
        for key in list(pos):
            ps = pos[key]
            sym = ps['sym']
            im = si.get(sym)
            if not im or date not in im: continue
            il = im[date]
            S = raw[sym]['close'][il]
            h = int((date - ps['ed'])/np.timedelta64(1,'D'))

            if ps['ptype'] == 'option':
                if h < hd_opt: continue
                Tr = max(ps['T'] - h/TD, .001)
                cv = bs(S, ps['K_call'], Tr, R, ps['sig'], 'call')
                pv = bs(S, ps['K_put'], Tr, R, ps['sig'], 'put')
                exit_cost = (cv + pv) * ld
                pnl = (ps['prem_received'] - exit_cost) * ps['ml'] * ps['ct']
                c = max(exit_cost, 0.01) * ps['ml'] * ps['ct'] * COMM_OPT
                eq += pnl - c
                pnls.append(pnl - c)
                trade_history.append((pnl - c, date))
                del pos[key]

            elif ps['ptype'] == 'future':
                if h < hd_fut: continue
                comm = S * ps['m'] * ps['fl'] * COMM
                pnl = (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                eq += ps['fm'] + pnl - comm
                pnls.append(pnl - comm)
                trade_history.append((pnl - comm, date))
                del pos[key]

        # === 入场 ===
        if len(pos) < mp:
            sigs = []

            # --- 收集所有候选信号 ---
            for sym, d in raw.items():
                if sym in [ps['sym'] for ps in pos.values()]: continue
                im = si.get(sym)
                if not im or date not in im: continue
                il = im[date]
                if il <= 0: continue
                pi = il - 1

                hv = d['hv20'][pi]
                if np.isnan(hv) or hv < .05 or hv > .80: continue
                hp = d['hv_pct'][pi]
                if np.isnan(hp) or hp > .95: continue
                rsi = d['rsi'][pi]
                if np.isnan(rsi) or rsi > 82 or rsi < 18: continue
                ap = d['atr_pct'][pi]
                if np.isnan(ap) or ap > .06: continue

                # 期权信号: 跨式卖出
                if d['vol_sell'][pi] and hv >= hv_min_opt and hp >= hv_pct_min:
                    trend = d['trend'][pi]
                    sigs.append({
                        'type': 'option', 'sym': sym, 'hv': hv, 'hp': hp,
                        'trend': trend, 'il': il,
                        'priority': 100 + hv * 100  # HV越高优先级越高
                    })

                # 期货信号: 方向交易 (极强信号时)
                mom_dir = d['mom_dir'][pi]
                mom_strong = d['mom_strong'][pi]
                trend = d['trend'][pi]

                fut_ok = False
                if fut_strength == 'strong' and mom_dir != 0 and mom_dir == trend:
                    fut_ok = True
                elif fut_strength == 'very_strong' and mom_strong and mom_dir == trend:
                    fut_ok = True

                if fut_ok and abs(d['m5'][pi]) > 0.02:  # 5日动量>2%
                    sigs.append({
                        'type': 'future', 'sym': sym, 'dir': mom_dir,
                        'hv': hv, 'hp': hp, 'il': il,
                        'priority': 50 + abs(d['m5'][pi]) * 1000  # 动量越强优先级越高
                    })

            # 排序: 期权优先(priority>100), 期货其次
            sigs.sort(key=lambda x: -x['priority'])

            for sig in sigs:
                if len(pos) >= mp: break

                # 限制期货仓位数量
                n_fut = sum(1 for ps in pos.values() if ps['ptype'] == 'future')
                if sig['type'] == 'future' and n_fut >= max_fut_pos:
                    continue

                sym = sig['sym']
                il = sig['il']

                if sig['type'] == 'option':
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    T = hd_opt / TD

                    center = 1.0 + bias_shift * sig['trend']
                    K_call = S * center * (1 + straddle_width)
                    K_put = S * center * (1 - straddle_width)
                    if K_put <= 0 or K_call <= S: continue

                    call_prem = bs(S, K_call, T, R, sig['hv'], 'call')
                    put_prem = bs(S, K_put, T, R, sig['hv'], 'put')
                    total_prem = call_prem + put_prem
                    if total_prem <= 0: continue

                    k = 0.04
                    if len(trade_history) > 50:
                        recent = [t[0] for t in trade_history[-200:]]
                        pa = np.array(recent)
                        wr = (pa > 0).mean()
                        if (pa > 0).any() and (pa <= 0).any():
                            aw = pa[pa>0].mean()
                            al = abs(pa[pa<=0].mean())
                            k = kelly_fraction(wr, aw, al) * kelly_mult

                    risk = eq * min(k, kelly_cap)
                    ct = max(int(risk / (total_prem * ml)), 1)
                    ct = min(ct, max(int(eq * notional_mult / (S * ml)), 1))
                    cost = total_prem * ml * ct
                    if cost > eq * .3:
                        ct = max(int(eq*.3/(total_prem*ml+1)), 1)
                        cost = total_prem*ml*ct
                    if cost > eq: continue

                    pos[sym] = {
                        'ptype': 'option', 'sym': sym, 'ed': date, 'S': S,
                        'K_call': K_call, 'K_put': K_put,
                        'T': T, 'sig': sig['hv'], 'prem_received': total_prem,
                        'ml': ml, 'ct': ct,
                    }

                elif sig['type'] == 'future':
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    mpl = S * ml * mr
                    target = eq * mu_fut / mp
                    fl = max(int(target/mpl), 1)
                    fm = mpl * fl
                    fc = S * ml * fl * COMM
                    if fm + fc > eq * 0.6:
                        fl = max(int((eq*0.6 - fc)/mpl), 0)
                        if fl <= 0: continue
                        fm = mpl * fl; fc = S * ml * fl * COMM
                    eq -= fm + fc
                    pos[sym + '_F'] = {
                        'ptype': 'future', 'sym': sym, 'ed': date,
                        'ep': S, 'd': sig['dir'], 'fl': fl, 'm': ml, 'fm': fm,
                    }

        # === 权益 ===
        ur = 0.
        for key, ps in pos.items():
            sym = ps['sym']
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                if ps['ptype'] == 'option':
                    Tr = max(ps['T']-h/TD, .001)
                    cv = bs(S, ps['K_call'], Tr, R, ps['sig'], 'call')
                    pv = bs(S, ps['K_put'], Tr, R, ps['sig'], 'put')
                    ur += (ps['prem_received'] - (cv+pv)*ld) * ps['ml'] * ps['ct']
                elif ps['ptype'] == 'future':
                    ur += (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
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

    # 统计期货vs期权占比
    n_opt_trades = 0
    n_fut_trades = 0
    # (we can't easily track this post-hoc, skip)

    return {'annual':ann,'wr':wr,'mdd':mdd,'pf':pf,'trades':len(pa),
            'final':eq,'sharpe':sh,**p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载..."); t0=time.time()
    raw = load(dd); print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")
    sd = pd.Timestamp('2018-01-01'); ed = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, sd, ed); print(f"  {len(dates)}交易日")

    pl = []

    # === A: 基线 — V52纯期权(复现) ===
    for hd in [5, 7]:
        for sw in [.010, .015, .020]:
            pl.append(dict(hold_days_opt=hd, straddle_width=sw,
                          max_fut_pos=0,  # 无期货
                          notional_mult=3.0, kelly_mult=4.0, kelly_cap=0.12))

    # === B: 期权为主 + 1个期货增强 ===
    for hd_o in [5, 7]:
        for hd_f in [3, 5]:
            for sw in [.010, .015, .020]:
                for mu_f in [.80, .90]:
                    pl.append(dict(hold_days_opt=hd_o, hold_days_fut=hd_f,
                                  straddle_width=sw, margin_usage_fut=mu_f,
                                  max_fut_pos=1, fut_strength='strong',
                                  notional_mult=3.0, kelly_mult=4.0, kelly_cap=0.12))

    # === C: 期权为主 + 极强期货信号 ===
    for hd_o in [5, 7]:
        for hd_f in [3, 5]:
            for sw in [.010, .015, .020]:
                pl.append(dict(hold_days_opt=hd_o, hold_days_fut=hd_f,
                              straddle_width=sw, margin_usage_fut=.90,
                              max_fut_pos=1, fut_strength='very_strong',
                              notional_mult=3.0, kelly_mult=4.0, kelly_cap=0.12))

    # === D: 降低期权入场阈值(更多交易) + 期货增强 ===
    for hd_o in [5, 7]:
        for sw in [.010, .015, .020]:
            for hv_pct_min in [0.50, 0.55]:
                for hv_min in [0.15, 0.18]:
                    pl.append(dict(hold_days_opt=hd_o, hold_days_fut=5,
                                  straddle_width=sw, margin_usage_fut=.90,
                                  max_fut_pos=1, fut_strength='strong',
                                  hv_pct_min=hv_pct_min, hv_min_opt=hv_min,
                                  notional_mult=3.0, kelly_mult=4.0, kelly_cap=0.12))

    # === E: 趋势偏移 + 期货 ===
    for hd_o in [5, 7]:
        for sw in [.010, .015, .020]:
            for bs in [0.003, 0.005]:
                pl.append(dict(hold_days_opt=hd_o, hold_days_fut=5,
                              straddle_width=sw, bias_shift=bs,
                              margin_usage_fut=.90, max_fut_pos=1,
                              notional_mult=3.0, kelly_mult=4.0, kelly_cap=0.12))

    # === F: 高杠杆版 ===
    for hd_o in [5, 7]:
        for sw in [.010, .015]:
            for nm in [5.0]:
                pl.append(dict(hold_days_opt=hd_o, hold_days_fut=5,
                              straddle_width=sw, margin_usage_fut=.90,
                              max_fut_pos=1, fut_strength='strong',
                              notional_mult=nm, kelly_mult=5.0, kelly_cap=0.15))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 30 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'HO':>3} {'HF':>3} {'SW':>5} {'MFP':>4} {'FS':>5} {'NM':>4} {'KM':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*100)
    for r in res[:60]:
        sw = f"{r.get('straddle_width',0)*100:.1f}"
        mfp = f"{r.get('max_fut_pos',0)}"
        fs = r.get('fut_strength','strong')[:3]
        nm = f"{r.get('notional_mult',2):.0f}"
        km = f"{r.get('kelly_mult',3):.0f}"
        print(f"{r['hold_days_opt']:>3} {r.get('hold_days_fut',5):>3} {sw:>5} {mfp:>4} {fs:>5} {nm:>4} {km:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*100)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),
                       (3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),
                       (.5,.50,"年化>=50% & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  HO={r['hold_days_opt']}  HF={r.get('hold_days_fut',5)}  "
                      f"SW={r.get('straddle_width',0)*100:.1f}%  MFP={r.get('max_fut_pos',0)}  "
                      f"FS={r.get('fut_strength','strong')}  NM={r.get('notional_mult',2):.0f}  "
                      f"Trades={r['trades']}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:500]]
    with open(os.path.join(od,'backtest_v55.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v55.json")


if __name__ == '__main__':
    main()
