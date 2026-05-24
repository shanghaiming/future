#!/usr/bin/env python3
"""
策略 V58 — 保护性期权增强的期货交易 (只买期权, 不卖)

核心: 期货方向交易 + 买入保护性期权 → 允许激进仓位

原理:
  普通期货: 仓位受限于风险承受 → 保守
  保护期货: 买OTM保护期权封住尾部 → 仓位可以放大3-5倍

  例: 做多10手期货 + 买10手OTM看跌
  - 涨: 期货盈利 >> 看跌期权费 → 大赚
  - 跌: 看跌期权获利抵消大部分期货亏损 → 亏损有限
  - 最大亏损 = (行权价差 + 期权费) × 手数 → 已知可控

期权只买不卖:
  A. 买入保护性看跌 (多头时): 限制下行风险
  B. 买入保护性看涨 (空头时): 限制上行风险
  C. IV信号: IV低→保护便宜→加仓; IV高→保护贵→降仓

目标: 年化600%, 胜率>50%, 最大持仓3
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

        mom_cols = ['m3','m5','m10','m20','m60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_c = np.sum(signs > 0, axis=1)
        neg_c = np.sum(signs < 0, axis=1)
        df['mom_dir'] = np.where(pos_c >= 4, 1.0, np.where(neg_c >= 4, -1.0, 0.0))

        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>10 else .5, raw=False)
        df['iv_premium'] = (df['hv5'] / df['hv20'].replace(0,np.nan) - 1).clip(-0.5, 0.5)

        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma5'] > df['ma20'], 1., -1.)

        d = df['close'].diff(); g = d.where(d>0,0).rolling(14).mean()
        l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # Skew proxy
        up = df['ret'].where(df['ret']>0,0).rolling(20).std()*np.sqrt(TD)
        dn = (-df['ret'].where(df['ret']<0,0)).rolling(20).std()*np.sqrt(TD)
        df['skew'] = (dn - up) / df['hv20'].replace(0, np.nan)

        # 布林带
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_lower'] = bb_ma - 2 * bb_std
        df['bb_upper'] = bb_ma + 2 * bb_std

        if 'oi' in df.columns:
            df['oi_ma'] = df['oi'].rolling(20).mean()
            df['oi_signal'] = np.where(df['oi'] > df['oi_ma'], 1, -1)
        else:
            df['oi_signal'] = 0

        df = df.dropna(subset=['ma20','ma60','hv20','rsi','atr'])
        if len(df) < 100: continue
        try: spec = get_spec(sym)
        except: continue

        raw[sym] = {
            'spec': spec, 'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'high': df['high'].values.astype(np.float64),
            'low': df['low'].values.astype(np.float64),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'iv_premium': df['iv_premium'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'atr': df['atr'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'mom_dir': df['mom_dir'].values.astype(np.float64),
            'trend': df['trend'].values.astype(np.float64),
            'skew': df['skew'].values.astype(np.float64),
            'oi_signal': df['oi_signal'].values.astype(np.float64) if 'oi_signal' in df.columns else np.zeros(len(df)),
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
    hold_days = p.get('hold_days', 7)
    mu = p.get('margin_usage', .90)
    leverage_mult = p.get('leverage_mult', 3)  # 保护期权下的杠杆倍数
    protection_otm = p.get('prot_otm', 0.03)  # 保护期权OTM程度
    use_protection = p.get('use_protection', True)  # 是否买保护
    use_iv_sizing = p.get('use_iv_sizing', True)  # IV调整仓位
    use_iv_filter = p.get('use_iv_filter', True)  # IV过滤入场
    iv_pct_min = p.get('iv_pct_min', 0.40)
    use_skew = p.get('use_skew', True)
    max_loss_pct = p.get('max_loss_pct', 0.05)  # 最大单笔亏损占权益

    eq = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]
    trade_history = []
    ld = 0.9

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

            if h < hold_days: continue

            # 期货PnL
            fut_pnl = (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
            fut_comm = S * ps['m'] * ps['fl'] * COMM

            total_pnl = fut_pnl
            total_comm = fut_comm

            # 保护期权PnL
            if ps.get('prot_type'):
                Tr = max(ps['T'] - h/TD, .001)
                if ps['prot_type'] == 'put':
                    prot_val = bs(S, ps['K_prot'], Tr, R, ps['sig'], 'put')
                    intr = max(ps['K_prot'] - S, 0)
                    prot_pnl = (max(prot_val, intr) * ld - ps['prot_cost']) * ps['m'] * ps['fl']
                else:  # call
                    prot_val = bs(S, ps['K_prot'], Tr, R, ps['sig'], 'call')
                    intr = max(S - ps['K_prot'], 0)
                    prot_pnl = (max(prot_val, intr) * ld - ps['prot_cost']) * ps['m'] * ps['fl']
                total_pnl += prot_pnl
                total_comm += ps['prot_cost'] * ps['m'] * ps['fl'] * COMM_OPT

            eq += ps['fm'] + total_pnl - total_comm
            pnls.append(total_pnl - total_comm)
            trade_history.append((total_pnl - total_comm, date))
            del pos[key]

        # === 入场 ===
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in [ps['sym'] for ps in pos.values()]: continue
                im = si.get(sym)
                if not im or date not in im: continue
                il = im[date]
                if il <= 0: continue
                pi = il - 1

                hv = d['hv20'][pi]
                if np.isnan(hv) or hv < .03 or hv > .80: continue
                hp = d['hv_pct'][pi]
                if np.isnan(hp) or hp > .95: continue
                rsi = d['rsi'][pi]
                if np.isnan(rsi): continue
                atr = d['atr'][pi]
                if np.isnan(atr) or atr <= 0: continue
                atr_pct = d['atr_pct'][pi]
                if np.isnan(atr_pct) or atr_pct > .06: continue

                mom_dir = d['mom_dir'][pi]
                trend = d['trend'][pi]
                skew = d['skew'] if 'skew' in d else None
                iv_prem = d['iv_premium'][pi] if 'iv_premium' in d else 0
                oi_sig = d['oi_signal'][pi] if 'oi_signal' in d else 0

                # 方向信号: 动量+趋势一致
                if mom_dir == 0 or mom_dir != trend: continue

                direction = mom_dir
                score = 50

                # IV信号服务期货
                if use_iv_filter:
                    if hp < iv_pct_min: continue
                    score += min((hp - 0.4) * 40, 15)

                # IV溢价 (短期波动加速)
                if not np.isnan(iv_prem) and iv_prem > 0:
                    score += min(iv_prem * 20, 10)

                # Skew确认
                if use_skew and skew is not None and not np.isnan(skew[pi]):
                    sk = skew[pi]
                    if direction < 0 and sk > 0.1: score += 8
                    elif direction > 0 and sk < -0.1: score += 8
                    elif direction > 0 and sk > 0.2: score += 5
                    elif direction < 0 and sk < -0.2: score += 5

                # RSI过滤
                if direction > 0 and rsi > 75: continue
                if direction < 0 and rsi < 25: continue

                sigs.append({
                    'sym': sym, 'dir': direction, 'score': score,
                    'hv': hv, 'hp': hp, 'atr': atr, 'il': il,
                })

            sigs.sort(key=lambda x: -x['score'])

            for sig in sigs:
                if len(pos) >= mp: break

                sym = sig['sym']
                il = sig['il']
                S = raw[sym]['open'][il]
                if np.isnan(S) or S <= 0: continue
                ml, mr, _, _ = raw[sym]['spec']
                hv = sig['hv']
                hp = sig['hp']

                # === 仓位计算 ===
                mpl = S * ml * mr  # 一手保证金

                # IV调整杠杆
                eff_lev = leverage_mult
                if use_iv_sizing:
                    if hp > 0.75:
                        eff_lev *= 0.6  # 高IV保护成本高, 降杠杆
                    elif hp < 0.45:
                        eff_lev *= 1.3  # 低IV保护便宜, 加杠杆

                # 目标仓位
                target = eq * mu * eff_lev / mp
                fl = max(int(target / mpl), 1)

                # 有保护期权时, 可以用更大仓位
                # 计算保护成本来确定最大手数
                T_opt = hold_days / TD
                if use_protection:
                    if sig['dir'] > 0:
                        K_prot = S * (1 - protection_otm)
                        prot_cost_per_lot = bs(S, K_prot, T_opt, R, hv, 'put')
                    else:
                        K_prot = S * (1 + protection_otm)
                        prot_cost_per_lot = bs(S, K_prot, T_opt, R, hv, 'call')

                    # 最大亏损 = 保护期权行使价差 + 期权费
                    if sig['dir'] > 0:
                        max_loss_per_lot = (S - K_prot + prot_cost_per_lot) * ml
                    else:
                        max_loss_per_lot = (K_prot - S + prot_cost_per_lot) * ml

                    # 限制总最大亏损
                    max_total_loss = eq * max_loss_pct
                    fl = min(fl, max(int(max_total_loss / max(max_loss_per_lot, 1)), 1))

                # 资金限制
                fm = mpl * fl
                prot_total = prot_cost_per_lot * ml * fl if use_protection and prot_cost_per_lot > 0 else 0
                fc = S * ml * fl * COMM
                total_cost = fm + prot_total + fc
                if total_cost > eq * 0.9:
                    fl = max(int((eq * 0.9 - fc - prot_total) / mpl), 0)
                    if fl <= 0: continue
                    fm = mpl * fl
                    prot_total = prot_cost_per_lot * ml * fl if use_protection and prot_cost_per_lot > 0 else 0

                eq -= fm + prot_total + fc

                ps = {
                    'ptype': 'future', 'sym': sym, 'ed': date,
                    'ep': S, 'd': sig['dir'], 'fl': fl, 'm': ml, 'fm': fm,
                }

                # 保护期权
                if use_protection and prot_cost_per_lot > 0:
                    ps['prot_type'] = 'put' if sig['dir'] > 0 else 'call'
                    ps['K_prot'] = K_prot
                    ps['T'] = T_opt
                    ps['sig'] = hv
                    ps['prot_cost'] = prot_cost_per_lot

                pos[sym] = ps

        # === 权益 ===
        ur = 0.
        for key, ps in pos.items():
            sym = ps['sym']
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                ur += (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                if ps.get('prot_type'):
                    Tr = max(ps['T']-h/TD, .001)
                    if ps['prot_type'] == 'put':
                        pv = bs(S, ps['K_prot'], Tr, R, ps['sig'], 'put')
                        ur += (max(pv, max(ps['K_prot']-S,0))*ld - ps['prot_cost']) * ps['m'] * ps['fl']
                    else:
                        cv = bs(S, ps['K_prot'], Tr, R, ps['sig'], 'call')
                        ur += (max(cv, max(S-ps['K_prot'],0))*ld - ps['prot_cost']) * ps['m'] * ps['fl']
                if ps.get('covered_type'):
                    Tr = max(ps['T_cov']-h/TD, .001)
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
    if len(pa) < 10: return None
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

    # === A: 保护性期权 + 高杠杆 ===
    for hd in [5, 7, 10]:
        for lev in [2, 3, 5, 8, 12]:
            for prot_otm in [.02, .03, .05]:
                for ml_pct in [.03, .05, .08, .10]:
                    pl.append(dict(hold_days=hd, leverage_mult=lev,
                                  prot_otm=prot_otm, max_loss_pct=ml_pct,
                                  use_protection=True))

    # === B: 无保护 (对照) ===
    for hd in [5, 7, 10]:
        for lev in [1, 2, 3]:
            pl.append(dict(hold_days=hd, leverage_mult=lev,
                          use_protection=False))

    # === C: 不同IV阈值 ===
    for iv_min in [0.35, 0.40, 0.50]:
        for hd in [5, 7]:
            for lev in [3, 5, 8]:
                pl.append(dict(hold_days=hd, leverage_mult=lev,
                              prot_otm=.03, max_loss_pct=.05,
                              use_protection=True, iv_pct_min=iv_min))

    # === D: 极端杠杆 (测试上限) ===
    for hd in [5, 7]:
        for lev in [10, 15, 20]:
            for prot_otm in [.03, .05]:
                pl.append(dict(hold_days=hd, leverage_mult=lev,
                              prot_otm=prot_otm, max_loss_pct=.10,
                              use_protection=True))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 30 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'H':>3} {'LEV':>4} {'POTM':>5} {'ML':>4} {'IV':>4} {'PROT':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*100)
    for r in res[:80]:
        hd = r.get('hold_days',7)
        lev = f"{r.get('leverage_mult',1)}"
        potm = f"{r.get('prot_otm',0)*100:.0f}" if r.get('use_protection') else '-'
        ml = f"{r.get('max_loss_pct',0)*100:.0f}"
        iv = f"{r.get('iv_pct_min',.4)*100:.0f}" if r.get('use_iv_filter',True) else 'off'
        prot = 'Y' if r.get('use_protection') else '-'
        print(f"{hd:>3} {lev:>4} {potm:>5} {ml:>4} {iv:>4} {prot:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*110)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),
                       (3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),
                       (.5,.50,"年化>=50% & WR>=50%"),
                       (.2,.50,"年化>=20% & WR>=50%"),
                       (.1,.50,"年化>=10% & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  H={r.get('hold_days',7)}  "
                      f"LEV={r.get('leverage_mult',1)}  POTM={r.get('prot_otm',0)*100:.0f}%  "
                      f"ML={r.get('max_loss_pct',0)*100:.0f}%  "
                      f"PROT={'Y' if r.get('use_protection') else 'N'}  Trades={r['trades']}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:500]]
    with open(os.path.join(od,'backtest_v58.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v58.json")


if __name__ == '__main__':
    main()
