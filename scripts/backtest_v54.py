#!/usr/bin/env python3
"""
策略 V54 — 期权期货融合策略 (真正结合)

核心架构:
  A. 期权收入核心: 卖跨式/宽跨式 (已验证: 76%WR, 高年化)
  B. 期货方向增强: 强趋势时在方向上开期货仓位
  C. 期货Delta对冲: 期权仓位浮亏时用期货对冲尾部风险

仓位分配 (最大3个):
  - 1-2个期权仓位 (跨式卖出)
  - 1-2个期货仓位 (方向交易 或 delta对冲)
  - 总仓位 <= 3

期权信号:
  - HV百分位 > 60% → 卖跨式
  - Skew → 选择偏移方向

期货信号:
  - 动量一致性 >= 4/5 + 趋势方向一致 → 方向期货
  - 期权浮亏 > 阈值 → delta对冲期货

目标: 年化600%, 胜率>50%
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


def bs_delta(S, K, T, r, sigma, flag='call'):
    if T <= 0 or sigma <= 0:
        return 1.0 if flag == 'call' else -1.0
    d1 = (np.log(S/K) + (r + .5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) if flag == 'call' else norm.cdf(d1) - 1


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

        vol_sell = df['hv_pct'].values > 0.6

        raw[sym] = {
            'spec': spec, 'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'high': df['high'].values.astype(np.float64),
            'low': df['low'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
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
    return max(min(f, 0.25), 0.01)


def bt(raw, dates, si, p):
    mp = p.get('max_pos', 3)
    hd_opt = p.get('hold_days_opt', 7)  # 期权持有天数
    hd_fut = p.get('hold_days_fut', 5)  # 期货持有天数
    straddle_width = p.get('straddle_width', 0.02)
    ld = p.get('liq_disc', .9)
    kelly_mult = p.get('kelly_mult', 3.0)
    kelly_cap = p.get('kelly_cap', 0.08)
    notional_mult = p.get('notional_mult', 2.0)
    mu_fut = p.get('margin_usage_fut', .90)
    bias_shift = p.get('bias_shift', 0.0)
    delta_hedge = p.get('delta_hedge', False)  # 是否用期货delta对冲
    delta_hedge_thresh = p.get('delta_hedge_thresh', 0.5)  # delta超过此值时对冲
    fut_signal_min = p.get('fut_signal_min', 4)  # 动量一致性最小值
    opt_slots = p.get('opt_slots', 2)  # 期权最大仓位
    fut_slots = p.get('fut_slots', 1)  # 期货最大仓位

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]
    trade_history = []

    for date in dates:
        # === 统计当前仓位类型 ===
        n_opt = sum(1 for ps in pos.values() if ps['ptype'] == 'option')
        n_fut = sum(1 for ps in pos.values() if ps['ptype'] == 'future')

        # === 退出 ===
        for sym in list(pos):
            ps = pos[sym]
            actual_sym = ps.get('sym', sym)
            im = si.get(actual_sym)
            if not im or date not in im: continue
            il = im[date]
            S = raw[actual_sym]['close'][il]
            h = int((date - ps['ed'])/np.timedelta64(1,'D'))

            if ps['ptype'] == 'option':
                # 期权: 跨式卖出
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
                del pos[sym]

            elif ps['ptype'] == 'future':
                # 期货: 方向交易
                if h < hd_fut: continue
                comm = S * ps['m'] * ps['fl'] * COMM
                pnl = (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                eq += ps['fm'] + pnl - comm
                pnls.append(pnl - comm)
                trade_history.append((pnl - comm, date))
                del pos[sym]

            elif ps['ptype'] == 'hedge':
                # 期货: delta对冲仓 → 跟随对应期权退出
                # 对冲仓在期权退出时一起退出
                if h < hd_opt: continue
                comm = S * ps['m'] * ps['fl'] * COMM
                pnl = (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                eq += ps['fm'] + pnl - comm
                # 对冲仓PnL合并到期权PnL中
                pnls.append(pnl - comm)
                trade_history.append((pnl - comm, date))
                del pos[sym]

        # === 重新统计 ===
        n_opt = sum(1 for ps in pos.values() if ps['ptype'] == 'option')
        n_fut = sum(1 for ps in pos.values() if ps['ptype'] in ('future', 'hedge'))

        # === 入场 ===
        if len(pos) < mp:
            # --- 期权入场: 跨式卖出 ---
            if n_opt < opt_slots:
                opt_sigs = []
                for sym, d in raw.items():
                    opt_key = f"{sym}_opt"
                    if opt_key in pos: continue
                    im = si.get(sym)
                    if not im or date not in im: continue
                    il = im[date]
                    if il <= 0: continue
                    pi = il - 1

                    hv = d['hv20'][pi]
                    if np.isnan(hv) or hv < 0.20 or hv > .80: continue
                    hp = d['hv_pct'][pi]
                    if np.isnan(hp) or hp > .95 or hp < 0.60: continue
                    rsi = d['rsi'][pi]
                    if np.isnan(rsi) or rsi > 82 or rsi < 18: continue

                    # 趋势偏移
                    trend = d['trend'][pi]
                    center = 1.0 + bias_shift * trend

                    opt_sigs.append((sym, hv, hp, trend, center, il))

                # 按HV降序排 (高HV=更多权利金)
                opt_sigs.sort(key=lambda x: -x[1])

                for sig in opt_sigs:
                    if n_opt >= opt_slots or len(pos) >= mp: break
                    sym, hv, hp, trend, center, il = sig
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    T = hd_opt / TD

                    K_call = S * center * (1 + straddle_width)
                    K_put = S * center * (1 - straddle_width)
                    if K_put <= 0 or K_call <= S: continue

                    call_prem = bs(S, K_call, T, R, hv, 'call')
                    put_prem = bs(S, K_put, T, R, hv, 'put')
                    total_prem = call_prem + put_prem
                    if total_prem <= 0: continue

                    # Kelly sizing
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

                    opt_key = f"{sym}_opt"
                    pos[opt_key] = {
                        'ptype': 'option', 'sym': sym, 'ed': date, 'S': S,
                        'K_call': K_call, 'K_put': K_put,
                        'T': T, 'sig': hv, 'prem_received': total_prem,
                        'ml': ml, 'ct': ct,
                    }
                    n_opt += 1

                    # === Delta对冲: 如果跨式delta偏大, 用期货对冲 ===
                    if delta_hedge and len(pos) < mp and n_fut < fut_slots:
                        # 计算当前组合delta
                        call_delta = bs_delta(S, K_call, T, R, hv, 'call')
                        put_delta = bs_delta(S, K_put, T, R, hv, 'put')
                        # 卖出跨式 → delta = -(call_delta + put_delta)
                        port_delta = -(call_delta + put_delta) * ct * ml
                        # delta exposure过大 → 用期货对冲
                        if abs(port_delta) > delta_hedge_thresh * S * ml * ct:
                            hedge_dir = -np.sign(port_delta)  # 反向对冲
                            mpl = S * ml * mr
                            fl = max(int(abs(port_delta) / (S * ml)), 1)
                            fl = min(fl, max(int(eq * .3 / mpl), 1))
                            fm = mpl * fl
                            fc = S * ml * fl * COMM
                            if fm + fc < eq * .3:
                                eq -= fm + fc
                                hedge_key = f"{sym}_hdg"
                                pos[hedge_key] = {
                                    'ptype': 'hedge', 'sym': sym, 'ed': date,
                                    'ep': S, 'd': hedge_dir, 'fl': fl, 'm': ml, 'fm': fm,
                                }
                                n_fut += 1

            # --- 期货入场: 方向交易 ---
            if n_fut < fut_slots and len(pos) < mp:
                fut_sigs = []
                for sym, d in raw.items():
                    fut_key = f"{sym}_fut"
                    if fut_key in pos or f"{sym}_opt" in pos: continue  # 不和同品种期权冲突
                    im = si.get(sym)
                    if not im or date not in im: continue
                    il = im[date]
                    if il <= 0: continue
                    pi = il - 1

                    hv = d['hv20'][pi]
                    if np.isnan(hv) or hv < .05 or hv > .70: continue
                    rsi = d['rsi'][pi]
                    if np.isnan(rsi) or rsi > 78 or rsi < 22: continue

                    mom_dir = d['mom_dir'][pi]
                    trend = d['trend'][pi]

                    # 强方向信号: 动量一致+趋势一致
                    if mom_dir == 0: continue
                    if mom_dir != trend: continue

                    # 方向强度评分
                    strength = abs(mom_dir)  # 基础: 动量一致

                    fut_sigs.append((sym, mom_dir, strength, hv, il))

                for sig in fut_sigs:
                    if n_fut >= fut_slots or len(pos) >= mp: break
                    sym, direction, strength, hv, il = sig
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']

                    mpl = S * ml * mr
                    target = eq * mu_fut / max(fut_slots, 1)
                    fl = max(int(target / mpl), 1)
                    fm = mpl * fl
                    fc = S * ml * fl * COMM
                    if fm + fc > eq * 0.6:
                        fl = max(int((eq*0.6 - fc)/mpl), 0)
                        if fl <= 0: continue
                        fm = mpl * fl; fc = S * ml * fl * COMM

                    eq -= fm + fc
                    fut_key = f"{sym}_fut"
                    pos[fut_key] = {
                        'ptype': 'future', 'sym': sym, 'ed': date,
                        'ep': S, 'd': direction, 'fl': fl, 'm': ml, 'fm': fm,
                    }
                    n_fut += 1

        # === 权益计算 ===
        ur = 0.
        for key, ps in pos.items():
            actual_sym = ps.get('sym', key.split('_')[0])
            im = si.get(actual_sym)
            if im and date in im:
                S = raw[actual_sym]['close'][im[date]]
                h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                if ps['ptype'] == 'option':
                    Tr = max(ps['T']-h/TD, .001)
                    cv = bs(S, ps['K_call'], Tr, R, ps['sig'], 'call')
                    pv = bs(S, ps['K_put'], Tr, R, ps['sig'], 'put')
                    ur += (ps['prem_received'] - (cv+pv)*ld) * ps['ml'] * ps['ct']
                elif ps['ptype'] in ('future', 'hedge'):
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

    return {'annual':ann,'wr':wr,'mdd':mdd,'pf':pf,'trades':len(pa),
            'final':eq,'sharpe':sh,**p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载..."); t0=time.time()
    raw = load(dd); print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")
    sd = pd.Timestamp('2018-01-01'); ed = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, sd, ed); print(f"  {len(dates)}交易日")

    pl = []

    # === A: 期权2+期货1 (核心配置) ===
    for hd_o in [5, 7, 10]:
        for hd_f in [3, 5, 7]:
            for sw in [.010, .015, .020, .025]:
                for mu_f in [.80, .90]:
                    pl.append(dict(hold_days_opt=hd_o, hold_days_fut=hd_f,
                                  straddle_width=sw, margin_usage_fut=mu_f,
                                  opt_slots=2, fut_slots=1))

    # === B: 期权1+期货2 (偏期货) ===
    for hd_o in [5, 7]:
        for hd_f in [5, 7]:
            for sw in [.015, .020]:
                for mu_f in [.80, .90]:
                    pl.append(dict(hold_days_opt=hd_o, hold_days_fut=hd_f,
                                  straddle_width=sw, margin_usage_fut=mu_f,
                                  opt_slots=1, fut_slots=2))

    # === C: 期权2+期货1 + Delta对冲 ===
    for hd_o in [5, 7]:
        for sw in [.010, .015, .020]:
            for bs in [0.0, 0.005]:
                pl.append(dict(hold_days_opt=hd_o, hold_days_fut=5,
                              straddle_width=sw, bias_shift=bs,
                              margin_usage_fut=.90, opt_slots=2, fut_slots=1,
                              delta_hedge=True, delta_hedge_thresh=0.5))

    # === D: 高杠杆期权+期货增强 ===
    for hd_o in [5, 7]:
        for sw in [.010, .015, .020]:
            for nm in [3.0, 5.0]:
                pl.append(dict(hold_days_opt=hd_o, hold_days_fut=5,
                              straddle_width=sw, margin_usage_fut=.90,
                              opt_slots=2, fut_slots=1,
                              notional_mult=nm, kelly_mult=4.0, kelly_cap=0.12))

    # === E: 期权2+期货1 + 趋势偏移 ===
    for hd_o in [5, 7]:
        for sw in [.010, .015, .020]:
            for bs in [0.003, 0.005, 0.008]:
                pl.append(dict(hold_days_opt=hd_o, hold_days_fut=5,
                              straddle_width=sw, bias_shift=bs,
                              margin_usage_fut=.90, opt_slots=2, fut_slots=1))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 30 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'HO':>3} {'HF':>3} {'SW':>5} {'MU':>4} {'OS':>3} {'FS':>3} {'BS':>5} {'DH':>3} {'NM':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*110)
    for r in res[:80]:
        sw = f"{r.get('straddle_width',0)*100:.1f}"
        mu = f"{r.get('margin_usage_fut',0):.0%}"
        bs = f"{r.get('bias_shift',0)*100:.1f}"
        dh = 'Y' if r.get('delta_hedge') else '-'
        nm = f"{r.get('notional_mult',2):.0f}" if r.get('notional_mult',2)!=2 else '-'
        print(f"{r['hold_days_opt']:>3} {r['hold_days_fut']:>3} {sw:>5} {mu:>4} "
              f"{r.get('opt_slots',2):>3} {r.get('fut_slots',1):>3} {bs:>5} {dh:>3} {nm:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*110)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),
                       (4.,.50,"年化>=400% & WR>=50%"),
                       (3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),
                       (.5,.50,"年化>=50% & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:8]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  HO={r['hold_days_opt']}  HF={r['hold_days_fut']}  "
                      f"SW={r.get('straddle_width',0)*100:.1f}%  OS={r.get('opt_slots',2)}  FS={r.get('fut_slots',1)}  "
                      f"BS={r.get('bias_shift',0)*100:.1f}%  DH={'Y' if r.get('delta_hedge') else 'N'}  "
                      f"NM={r.get('notional_mult',2):.0f}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:500]]
    with open(os.path.join(od,'backtest_v54.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v54.json")


if __name__ == '__main__':
    main()
