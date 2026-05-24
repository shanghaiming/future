#!/usr/bin/env python3
"""
策略 V51 — 极致波动率卖出优化 (基于V50突破)

V50突破: vol_sell straddle卖出 → 444.7%年化, 77.6%胜率
目标: 推向600%年化

优化路径:
  1. 更激进Kelly仓位 — 78%WR下可承受更大仓位
  2. 更短持有期 — 3-5天加速资本周转 + theta衰减加速
  3. 趋势偏移跨式 — 在趋势方向收取更多权利金
  4. 精细宽度网格 — 0.8%~2.5%寻找最优
  5. HV百分位甜点区 — 只在60-85%区间交易
  6. 资金利用率提升 — 从30%提升到40-50%
  7. 铁鹰保护翼 — 买入远端期权限制尾部风险
  8. 复利加速 — 每笔交易立即计入权益
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

        # EWMAC for trend direction
        for sp in [4, 8, 16]:
            ef = df['close'].ewm(span=sp).mean()
            es = df['close'].ewm(span=sp*4).mean()
            s = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ew{sp}'] = (ef - es) / s
        df['sdelta'] = df['ew4']*.2 + df['ew8']*.3 + df['ew16']*.3

        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20']>df['ma60'], 1., -1.)
        d = df['close'].diff(); g = d.where(d>0,0).rolling(14).mean()
        l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))
        tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / df['close']

        # 动量一致性
        mom_cols = ['m3','m5','m10','m20','m60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_count = np.sum(signs > 0, axis=1)
        neg_count = np.sum(signs < 0, axis=1)
        df['mom_dir'] = np.where(pos_count >= 4, 1.0, np.where(neg_count >= 4, -1.0, 0.0))

        df = df.dropna(subset=['ma20','ma60','hv20','m5','rsi','sdelta'])
        if len(df) < 100: continue
        try: spec = get_spec(sym)
        except: continue

        vol_sell = df['hv_pct'].values > 0.6

        raw[sym] = {
            'spec': spec, 'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'high': df['high'].values.astype(np.float64),
            'low': df['low'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'hv5': df['hv5'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'mom_dir': df['mom_dir'].values.astype(np.float64),
            'trend': df['trend'].values.astype(np.float64),
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
    if avg_loss <= 0 or avg_win <= 0: return 0.03
    b = avg_win / avg_loss
    q = 1 - wr
    f = (wr * b - q) / b
    return max(min(f, 0.30), 0.01)


def bt(raw, dates, si, p):
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 5)
    mode = p.get('mode', 'vol_sell')
    risk_pct = p.get('risk_pct', .04)
    straddle_width = p.get('straddle_width', 0.02)
    ld = p.get('liq_disc', .9)
    bias_shift = p.get('bias_shift', 0.0)  # 趋势偏移: 将跨式中心向趋势方向移动
    iron_wing = p.get('iron_wing', 0.0)  # 铁鹰保护翼宽度 (0=无保护)
    max_cap_use = p.get('max_cap_use', .40)  # 最大资金使用率
    kelly_mult = p.get('kelly_mult', 4.0)  # Kelly乘数
    kelly_cap = p.get('kelly_cap', 0.12)  # Kelly上限
    hv_min = p.get('hv_min', 0.18)  # 最低HV阈值
    hv_pct_lo = p.get('hv_pct_lo', 0.60)
    hv_pct_hi = p.get('hv_pct_hi', 0.92)
    direction = p.get('direction', 'both')  # both/bull/bear/neutral

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]
    trade_history = []

    for date in dates:
        # === 退出 ===
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if not im or date not in im: continue
            il = im[date]
            S = raw[sym]['close'][il]
            h = int((date - ps['ed'])/np.timedelta64(1,'D'))

            if ps['type'] == 'straddle':
                # 提前止盈: 如果未实现PnL达到premium的60%+，提前平仓
                Tr = max(ps['T'] - h/TD, .001)
                cv = bs(S, ps['K_call'], Tr, R, ps['sig'], 'call')
                pv = bs(S, ps['K_put'], Tr, R, ps['sig'], 'put')
                exit_cost = (cv + pv) * ld
                unrealized = ps['prem_received'] - exit_cost

                should_exit = h >= hd
                # 提前止盈: PnL达到premium的50%+
                if h >= 2 and unrealized > ps['prem_received'] * 0.5:
                    should_exit = True
                # 止损: PnL亏损超过premium的1.5倍
                if unrealized < -ps['prem_received'] * 1.5:
                    should_exit = True

                if not should_exit: continue

                pnl = unrealized * ps['ml'] * ps['ct']
                # 铁鹰保护翼的期权费
                wing_cost = 0
                if ps.get('wing_call_K'):
                    wing_cost += bs(S, ps['wing_call_K'], Tr, R, ps['sig'], 'call') * ld
                if ps.get('wing_put_K'):
                    wing_cost += bs(S, ps['wing_put_K'], Tr, R, ps['sig'], 'put') * ld
                pnl -= wing_cost * ps['ml'] * ps['ct']

                c = max(exit_cost, 0.01) * ps['ml'] * ps['ct'] * COMM_OPT
                eq += pnl - c
                pnls.append(pnl - c)
                trade_history.append((pnl - c, date))
                del pos[sym]

            elif ps['type'] == 'directional':
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

        # === 入场 ===
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
                if np.isnan(hv) or hv < hv_min or hv > .80: continue
                hp = d['hv_pct'][pi]
                if np.isnan(hp) or hp > hv_pct_hi or hp < hv_pct_lo: continue
                rsi = d['rsi'][pi]
                if np.isnan(rsi) or rsi > 80 or rsi < 20: continue
                ap = d['atr_pct'][pi]
                if np.isnan(ap) or ap > .06: continue

                if mode == 'vol_sell':
                    if not d['vol_sell'][pi]: continue
                    trend = d['trend'][pi]
                    md = d['mom_dir'][pi]
                    sigs.append(('straddle', sym, hv, hp, trend, md, il))
                    continue

                if mode == 'directional':
                    trend = d['trend'][pi]
                    md = d['mom_dir'][pi]
                    if md == 0 or md != trend: continue
                    sigs.append(('directional', sym, trend, hv, hp, il))
                    continue

                # multi模式
                trend = d['trend'][pi]
                md = d['mom_dir'][pi]
                # 波动率卖出优先
                if d['vol_sell'][pi] and hv > 0.20:
                    sigs.append(('straddle', sym, hv, hp, trend, md, il))
                # 方向性补充
                if md != 0 and md == trend:
                    sigs.append(('directional', sym, trend, hv, hp, il))

            # 排序: straddle优先, 按HV降序(高HV=更多权利金)
            sigs.sort(key=lambda x: (-x[2] if x[0]=='straddle' else 0, -x[2] if x[0]=='directional' else 0))

            for sig in sigs:
                if len(pos) >= mp: break
                stype = sig[0]
                sym = sig[1]

                if stype == 'straddle':
                    _, sym, hv, hp, trend, mom_dir, il = sig
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    T = hd / TD

                    # 趋势偏移: 将跨式中心向趋势方向移动
                    center = 1.0 + bias_shift * trend  # 上涨趋势时center>1, 下跌时<1

                    # 方向过滤
                    if direction == 'bull' and trend < 0: continue
                    if direction == 'bear' and trend > 0: continue

                    K_call = S * center * (1 + straddle_width)
                    K_put = S * center * (1 - straddle_width)
                    # 确保strike合理
                    if K_call <= 0 or K_put <= 0: continue
                    if K_put >= S or K_call <= S: continue

                    call_prem = bs(S, K_call, T, R, hv, 'call')
                    put_prem = bs(S, K_put, T, R, hv, 'put')
                    total_prem = call_prem + put_prem
                    if total_prem <= 0: continue

                    # 铁鹰保护翼
                    wing_call_cost = 0
                    wing_put_cost = 0
                    wing_call_K = None
                    wing_put_K = None
                    if iron_wing > 0:
                        wing_call_K = S * center * (1 + straddle_width + iron_wing)
                        wing_put_K = S * center * (1 - straddle_width - iron_wing)
                        wing_call_cost = bs(S, wing_call_K, T, R, hv, 'call') * 0.5
                        wing_put_cost = bs(S, wing_put_K, T, R, hv, 'put') * 0.5

                    net_prem = total_prem - wing_call_cost - wing_put_cost
                    if net_prem <= 0: continue

                    # Kelly sizing
                    k = 0.04
                    if len(trade_history) > 30:
                        recent = [t[0] for t in trade_history[-150:]]
                        pa = np.array(recent)
                        wr = (pa > 0).mean()
                        if (pa > 0).any() and (pa <= 0).any():
                            aw = pa[pa>0].mean()
                            al = abs(pa[pa<=0].mean())
                            k = kelly_fraction(wr, aw, al) * kelly_mult
                            k = min(k, kelly_cap)

                    risk = eq * k
                    ct = max(int(risk / (net_prem * ml)), 1)
                    ct = min(ct, max(int(eq * max_cap_use / (S * ml)), 1))
                    cost = net_prem * ml * ct
                    if cost > eq * max_cap_use:
                        ct = max(int(eq*max_cap_use/(net_prem*ml+1)), 1)
                        cost = net_prem * ml * ct
                    if cost > eq: continue

                    pos[sym] = {'type':'straddle','ed':date,'S':S,
                               'K_call':K_call,'K_put':K_put,
                               'T':T,'sig':hv,'prem_received':net_prem,
                               'ml':ml,'ct':ct,
                               'wing_call_K':wing_call_K,'wing_put_K':wing_put_K}

                elif stype == 'directional':
                    _, sym, dv, hv, hp, il = sig
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    fl = 'call' if dv > 0 else 'put'
                    K = S * (1 + 0.01 * dv)
                    T = hd / TD
                    pr = bs(S, K, T, R, hv, fl)
                    if pr <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    risk = eq * risk_pct
                    ct = max(int(risk / (pr * ml)), 1)
                    ct = min(ct, max(int(eq * max_cap_use / (S * ml)), 1))
                    cost = pr * ml * ct
                    if cost > eq * max_cap_use:
                        ct = max(int(eq*max_cap_use/(pr*ml+1)), 1); cost = pr*ml*ct
                    if cost > eq: continue
                    pos[sym] = {'type':'directional','dv':dv,'ed':date,'S':S,'K':K,
                               'T':T,'sig':hv,'pr':pr,'fl':fl,'ml':ml,'ct':ct}

        # === 权益 ===
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                if ps['type'] == 'straddle':
                    Tr = max(ps['T']-h/TD, .001)
                    cv = bs(S, ps['K_call'], Tr, R, ps['sig'], 'call')
                    pv = bs(S, ps['K_put'], Tr, R, ps['sig'], 'put')
                    ur += (ps['prem_received'] - (cv+pv)*ld) * ps['ml'] * ps['ct']
                elif ps['type'] == 'directional':
                    Tr = max(ps['T']-h/TD, .001)
                    ov = bs(S, ps['K'], Tr, R, ps['sig'], ps['fl'])
                    intr = max(S-ps['K'],0) if ps['fl']=='call' else max(ps['K']-S,0)
                    ur += (max(ov,intr)*ld - ps['pr']) * ps['dv'] * ps['ml'] * ps['ct']
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

    return {'annual':ann,'wr':wr,'mdd':mdd,'pf':pf,'trades':len(pa),
            'final':eq,'sharpe':sh,'avg_win':aw,'avg_loss':al,**p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载..."); t0=time.time()
    raw = load(dd); print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")
    sd = pd.Timestamp('2018-01-01'); ed = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, sd, ed); print(f"  {len(dates)}交易日")

    pl = []

    # === A: 精细vol_sell参数扫描 ===
    # V50最优: H=7, SW=2%. 现在精细扫描周边
    for hd in [3, 4, 5, 7]:
        for sw in [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025]:
            for bs in [0.0, 0.003, 0.005]:  # 趋势偏移
                for km in [3.0, 4.0, 5.0]:  # Kelly乘数
                    pl.append(dict(hold_days=hd, mode='vol_sell',
                                  straddle_width=sw, bias_shift=bs,
                                  kelly_mult=km, kelly_cap=0.15,
                                  max_cap_use=0.45, hv_min=0.18))

    # === B: 铁鹰保护翼 ===
    for hd in [5, 7]:
        for sw in [0.010, 0.015, 0.020]:
            for iw in [0.02, 0.03, 0.04]:
                pl.append(dict(hold_days=hd, mode='vol_sell',
                              straddle_width=sw, iron_wing=iw,
                              kelly_mult=4.0, kelly_cap=0.15,
                              max_cap_use=0.45, hv_min=0.18))

    # === C: 方向过滤 (只在趋势方向卖) ===
    for d in ['bull', 'bear']:
        for hd in [5, 7]:
            for sw in [0.010, 0.015, 0.020]:
                pl.append(dict(hold_days=hd, mode='vol_sell',
                              straddle_width=sw, direction=d,
                              bias_shift=0.005,
                              kelly_mult=4.0, kelly_cap=0.12,
                              max_cap_use=0.40, hv_min=0.18))

    # === D: 多层模式 ===
    for hd in [5, 7]:
        for sw in [0.012, 0.015, 0.020]:
            for bs in [0.0, 0.005]:
                pl.append(dict(hold_days=hd, mode='multi',
                              straddle_width=sw, bias_shift=bs,
                              risk_pct=0.04,
                              kelly_mult=4.0, kelly_cap=0.15,
                              max_cap_use=0.45, hv_min=0.18))

    # === E: 超短持有期 (1-2天快进快出) ===
    for hd in [2, 3]:
        for sw in [0.008, 0.010, 0.012]:
            for km in [3.0, 5.0]:
                pl.append(dict(hold_days=hd, mode='vol_sell',
                              straddle_width=sw,
                              kelly_mult=km, kelly_cap=0.12,
                              max_cap_use=0.35, hv_min=0.20))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 50 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    # 打印Top结果
    print(f"\n{'模式':>8} {'H':>3} {'SW':>5} {'Bias':>5} {'KM':>4} {'IW':>4} {'Dir':>6} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*110)
    for r in res[:80]:
        sw = f"{r.get('straddle_width',0)*100:.1f}"
        bs = f"{r.get('bias_shift',0)*100:.1f}"
        km = f"{r.get('kelly_mult',0):.0f}"
        iw = f"{r.get('iron_wing',0)*100:.0f}" if r.get('iron_wing',0)>0 else '-'
        dr = r.get('direction','both')[:3] if r.get('direction','both') != 'both' else 'both'
        print(f"{r['mode']:>8} {r['hold_days']:>3} {sw:>5} {bs:>5} {km:>4} {iw:>4} {dr:>6} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标达成统计
    print("\n" + "="*110)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),
                       (6.,.45,"年化>=600% & WR>=45%"),
                       (5.,.50,"年化>=500% & WR>=50%"),
                       (4.,.50,"年化>=400% & WR>=50%"),
                       (3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  {r['mode']}  H={r['hold_days']}  "
                      f"SW={r.get('straddle_width',0)*100:.1f}%  Bias={r.get('bias_shift',0)*100:.1f}%  "
                      f"KM={r.get('kelly_mult',0):.0f}  IW={r.get('iron_wing',0)*100:.0f}%  "
                      f"Dir={r.get('direction','both')}  Trades={r['trades']}")
        else: print("  无")

    # 保存
    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:500]]
    with open(os.path.join(od,'backtest_v51.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v51.json")


if __name__ == '__main__':
    main()
