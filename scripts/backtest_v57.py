#!/usr/bin/env python3
"""
策略 V57 — 期权信号驱动的期货交易

核心理念: 分析期权是为了服务期货
  - IV percentile → 判断"何时交易": 高IV=大波动即将来临, 适合做方向
  - Skew proxy → 判断"市场情绪": 看跌偏重=做空信号
  - HV ratio → 判断"趋势加速": HV5>>HV20=近期波动飙升=趋势正在发生

信号架构:
  入场条件 (全部满足才交易):
    1. IV信号: HV percentile > 50% (市场预期有波动)
    2. 方向信号: 动量4/5一致 + 趋势方向一致
    3. 期权情绪: skew proxy确认方向
    4. 波动加速: HV5/HV20 > 1.0 (近期波动加剧)

  仓位管理:
    - 根据IV调整仓位: 高IV=大波动=更大的止损空间但更小的仓位
    - 固定风险比例: 每笔亏损不超过权益的2%
    - 止损: 2×ATR / 止盈: 4×ATR

  期权辅助 (非交易工具):
    - 持期货多头时卖OTM看涨增收 (covered call)
    - 持期货空头时卖OTM看跌增收 (covered put)

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

        # 动量
        for lag in [3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        mom_cols = ['m3','m5','m10','m20','m60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_c = np.sum(signs > 0, axis=1)
        neg_c = np.sum(signs < 0, axis=1)
        df['mom_dir'] = np.where(pos_c >= 4, 1.0, np.where(neg_c >= 4, -1.0, 0.0))

        # 波动率 (期权信号)
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>10 else .5, raw=False)
        # IV溢价: 短期vs中期波动率差 → 类似期权IV期限结构
        df['iv_premium'] = (df['hv5'] / df['hv20'].replace(0, np.nan) - 1).clip(-0.5, 0.5)

        # 趋势
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma5'] > df['ma20'], 1., -1.)

        # RSI
        d = df['close'].diff(); g = d.where(d>0,0).rolling(14).mean()
        l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        # ATR
        tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # Skew proxy (期权情绪): 下行波动率 - 上行波动率
        up = df['ret'].where(df['ret']>0,0).rolling(20).std()*np.sqrt(TD)
        dn = (-df['ret'].where(df['ret']<0,0)).rolling(20).std()*np.sqrt(TD)
        df['skew'] = (dn - up) / df['hv20'].replace(0, np.nan)

        # OI
        if 'oi' in df.columns:
            df['oi_ma'] = df['oi'].rolling(20).mean()
            df['oi_trend'] = np.where(df['oi'] > df['oi_ma'], 1, -1)
        else:
            df['oi_trend'] = 0

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
            'oi_trend': df['oi_trend'].values.astype(np.float64) if 'oi_trend' in df.columns else np.zeros(len(df)),
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
    risk_pct = p.get('risk_pct', 0.02)  # 每笔最大亏损占权益比例
    atr_sl = p.get('atr_sl', 2.0)  # 止损ATR倍数
    atr_tp = p.get('atr_tp', 4.0)  # 止盈ATR倍数
    max_hold = p.get('max_hold', 10)
    covered_otm = p.get('covered_otm', 0.0)
    use_iv_filter = p.get('use_iv_filter', True)
    iv_pct_min = p.get('iv_pct_min', 0.45)
    use_skew = p.get('use_skew', True)
    use_iv_premium = p.get('use_iv_premium', True)
    trail_after = p.get('trail_after', 0.5)  # 盈利超过50%ATR后开始跟踪止损

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
            H = raw[sym]['high'][il]
            L = raw[sym]['low'][il]
            h = int((date - ps['ed'])/np.timedelta64(1,'D'))

            should_exit = False
            exit_price = S

            # 最大持有期
            if h >= max_hold:
                should_exit = True

            # 止损
            if ps['d'] > 0:
                if L <= ps['sl']:
                    should_exit = True
                    exit_price = max(L, ps['sl'])
                # 跟踪止损: 盈利后收紧
                if S > ps['ep'] + ps['atr_val'] * trail_after:
                    trail = S - ps['atr_val'] * atr_sl * 0.75  # 收紧止损
                    if trail > ps.get('trail', 0):
                        ps['trail'] = trail
                    if L <= ps.get('trail', 0):
                        should_exit = True
                        exit_price = max(L, ps.get('trail', ps['sl']))
                # 止盈
                if H >= ps['tp']:
                    should_exit = True
                    exit_price = min(H, ps['tp'])
            else:
                if H >= ps['sl']:
                    should_exit = True
                    exit_price = min(H, ps['sl'])
                if S < ps['ep'] - ps['atr_val'] * trail_after:
                    trail = S + ps['atr_val'] * atr_sl * 0.75
                    if trail < ps.get('trail', 1e9):
                        ps['trail'] = trail
                    if H >= ps.get('trail', 1e9):
                        should_exit = True
                        exit_price = min(H, ps.get('trail', ps['sl']))
                if L <= ps['tp']:
                    should_exit = True
                    exit_price = max(L, ps['tp'])

            if not should_exit: continue

            # 期货PnL
            pnl = (exit_price - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
            comm = exit_price * ps['m'] * ps['fl'] * COMM

            # 备兑期权PnL
            opt_pnl = 0
            if ps.get('covered_type'):
                T_elapsed = max(ps['T_cov'] - h/TD, .001)
                if ps['covered_type'] == 'call':
                    ov = bs(exit_price, ps['K_cov'], T_elapsed, R, ps['sig'], 'call')
                else:
                    ov = bs(exit_price, ps['K_cov'], T_elapsed, R, ps['sig'], 'put')
                opt_pnl = (ps['cov_prem'] - max(ov,0)*ld) * ps['m'] * ps['fl']
                comm += max(ov, 0.01) * ps['m'] * ps['fl'] * COMM_OPT

            eq += ps['fm'] + pnl + opt_pnl - comm
            pnls.append(pnl + opt_pnl - comm)
            trade_history.append((pnl + opt_pnl - comm, date))
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

                mom_dir = d['mom_dir'][pi]
                trend = d['trend'][pi]
                skew = d['skew'][pi]
                iv_prem = d['iv_premium'][pi]

                # 方向: 动量+趋势一致
                if mom_dir == 0 or mom_dir != trend: continue

                direction = mom_dir

                # === 期权信号服务期货 ===
                score = 50  # 基础分

                # 1. IV百分位: 高IV=大波动预期=适合做方向
                if use_iv_filter:
                    if hp < iv_pct_min: continue  # IV太低, 市场没预期
                    score += min((hp - 0.5) * 60, 20)  # IV越高分越高

                # 2. IV溢价: 短期IV > 中期IV = 近期波动加剧 = 趋势加速
                if use_iv_premium and not np.isnan(iv_prem):
                    if iv_prem > 0:  # 近期波动率飙升
                        score += min(iv_prem * 30, 15)

                # 3. Skew信号: 看跌偏重(正skew) + 做空 → 确认
                #                看涨偏重(负skew) + 做多 → 确认
                if use_skew and not np.isnan(skew):
                    if direction < 0 and skew > 0.1:  # 看跌情绪 + 做空
                        score += 10
                    elif direction > 0 and skew < -0.1:  # 看涨情绪 + 做多
                        score += 10
                    elif direction < 0 and skew < -0.2:  # 过度看涨 → 做空更好
                        score += 5
                    elif direction > 0 and skew > 0.2:  # 过度看跌 → 做多更好
                        score += 5

                # 4. RSI过滤
                if direction > 0 and rsi > 75: continue  # 超买不做多
                if direction < 0 and rsi < 25: continue  # 超卖不做空

                sigs.append({
                    'sym': sym, 'dir': direction, 'score': score,
                    'hv': hv, 'hp': hp, 'atr': atr,
                })

            sigs.sort(key=lambda x: -x['score'])

            for sig in sigs:
                if len(pos) >= mp: break

                sym = sig['sym']
                im = si.get(sym)
                il = im[date]
                S = raw[sym]['open'][il]
                if np.isnan(S) or S <= 0: continue
                ml, mr, _, _ = raw[sym]['spec']
                atr_val = sig['atr']

                # 固定风险仓位: 每笔最多亏权益的risk_pct
                # 止损距离 = atr_sl * ATR
                stop_distance = atr_sl * atr_val
                if stop_distance <= 0: continue

                # 一手的止损金额
                loss_per_lot = stop_distance * ml
                if loss_per_lot <= 0: continue

                # 计算手数
                max_loss = eq * risk_pct
                fl = max(int(max_loss / loss_per_lot), 1)

                # 限制总仓位不超过可用资金的一定比例
                mpl = S * ml * mr
                fm = mpl * fl
                fc = S * ml * fl * COMM
                max_cap = eq * 0.7
                if fm + fc > max_cap:
                    fl = max(int((max_cap - fc) / mpl), 0)
                    if fl <= 0: continue
                    fm = mpl * fl; fc = S * ml * fl * COMM

                eq -= fm + fc

                ps = {
                    'ptype': 'future', 'sym': sym, 'ed': date,
                    'ep': S, 'd': sig['dir'], 'fl': fl, 'm': ml, 'fm': fm,
                    'atr_val': atr_val,
                }

                # 止损止盈
                if sig['dir'] > 0:
                    ps['sl'] = S - stop_distance
                    ps['tp'] = S + atr_tp * atr_val
                else:
                    ps['sl'] = S + stop_distance
                    ps['tp'] = S - atr_tp * atr_val

                # 备兑期权
                if covered_otm > 0 and sig['hp'] > 0.5 and sig['hv'] > 0.12:
                    T_cov = max_hold / TD
                    if sig['dir'] > 0:
                        K_cov = S * (1 + covered_otm)
                        cov_prem = bs(S, K_cov, T_cov, R, sig['hv'], 'call')
                        if cov_prem > 0:
                            ps['covered_type'] = 'call'
                            ps['K_cov'] = K_cov
                            ps['T_cov'] = T_cov
                            ps['sig'] = sig['hv']
                            ps['cov_prem'] = cov_prem
                    else:
                        K_cov = S * (1 - covered_otm)
                        cov_prem = bs(S, K_cov, T_cov, R, sig['hv'], 'put')
                        if cov_prem > 0:
                            ps['covered_type'] = 'put'
                            ps['K_cov'] = K_cov
                            ps['T_cov'] = T_cov
                            ps['sig'] = sig['hv']
                            ps['cov_prem'] = cov_prem

                pos[sym] = ps

        # === 权益 ===
        ur = 0.
        for key, ps in pos.items():
            sym = ps['sym']
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                ur += (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                if ps.get('covered_type'):
                    h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                    Tr = max(ps['T_cov']-h/TD, .001)
                    if ps['covered_type'] == 'call':
                        ov = bs(S, ps['K_cov'], Tr, R, ps['sig'], 'call')
                    else:
                        ov = bs(S, ps['K_cov'], Tr, R, ps['sig'], 'put')
                    ur += (ps['cov_prem'] - max(ov,0)*ld) * ps['m'] * ps['fl']
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

    # === A: 基础参数扫描 ===
    for rp in [0.01, 0.02, 0.03, 0.04, 0.05]:
        for asl in [1.5, 2.0, 2.5, 3.0]:
            for atp in [3.0, 4.0, 5.0]:
                for mh in [7, 10, 15]:
                    pl.append(dict(risk_pct=rp, atr_sl=asl, atr_tp=atp, max_hold=mh))

    # === B: +备兑期权 ===
    for rp in [0.02, 0.03, 0.04]:
        for asl in [2.0, 2.5]:
            for cov in [.015, .020, .030]:
                pl.append(dict(risk_pct=rp, atr_sl=asl, atr_tp=4.0, max_hold=10,
                              covered_otm=cov))

    # === C: 不同IV阈值 ===
    for iv_min in [0.35, 0.40, 0.45, 0.50, 0.55]:
        for rp in [0.02, 0.03]:
            pl.append(dict(risk_pct=rp, atr_sl=2.0, atr_tp=4.0, max_hold=10,
                          iv_pct_min=iv_min))

    # === D: 激进版 (高风险高回报) ===
    for rp in [0.05, 0.06, 0.08]:
        for asl in [1.5, 2.0]:
            for atp in [3.0, 4.0]:
                pl.append(dict(risk_pct=rp, atr_sl=asl, atr_tp=atp, max_hold=7))

    # === E: 无IV过滤 (全时段交易) ===
    for rp in [0.02, 0.03, 0.04]:
        for asl in [2.0, 2.5, 3.0]:
            pl.append(dict(risk_pct=rp, atr_sl=asl, atr_tp=4.0, max_hold=10,
                          use_iv_filter=False))

    # === F: 无skew/iv_premium过滤 (只看方向) ===
    for rp in [0.02, 0.03, 0.04]:
        for asl in [2.0, 2.5]:
            pl.append(dict(risk_pct=rp, atr_sl=asl, atr_tp=4.0, max_hold=10,
                          use_skew=False, use_iv_premium=False))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 50 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'RP':>4} {'AS':>4} {'AT':>4} {'MH':>3} {'COV':>5} {'IV':>4} {'SK':>3} {'IP':>3} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*100)
    for r in res[:80]:
        rp = f"{r.get('risk_pct',0)*100:.0f}"
        asl = f"{r.get('atr_sl',0):.1f}"
        atp = f"{r.get('atr_tp',0):.1f}"
        mh = f"{r.get('max_hold',10)}"
        cov = f"{r.get('covered_otm',0)*100:.1f}" if r.get('covered_otm',0)>0 else '-'
        iv = f"{r.get('iv_pct_min',.45)*100:.0f}" if r.get('use_iv_filter',True) else 'off'
        sk = 'Y' if r.get('use_skew',True) else '-'
        ip = 'Y' if r.get('use_iv_premium',True) else '-'
        print(f"{rp:>4} {asl:>4} {atp:>4} {mh:>3} {cov:>5} {iv:>4} {sk:>3} {ip:>3} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*100)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),
                       (3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),
                       (.5,.50,"年化>=50% & WR>=50%"),
                       (.3,.50,"年化>=30% & WR>=50%"),
                       (.1,.50,"年化>=10% & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  RP={r.get('risk_pct',0)*100:.0f}%  "
                      f"AS={r.get('atr_sl',0):.1f}  AT={r.get('atr_tp',0):.1f}  "
                      f"MH={r.get('max_hold',10)}  COV={r.get('covered_otm',0)*100:.0f}%  "
                      f"IV={r.get('iv_pct_min',.45)*100:.0f}%  Trades={r['trades']}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:500]]
    with open(os.path.join(od,'backtest_v57.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v57.json")


if __name__ == '__main__':
    main()
