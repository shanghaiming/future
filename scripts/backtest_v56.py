#!/usr/bin/env python3
"""
策略 V56 — 期货为主, 期权辅助

核心: 期货方向交易为主体, 期权作为信号和增收工具

架构:
  1. 波动率Regime自适应:
     - 低波 (HV pct < 35%): 趋势跟踪 → EWMAC + 突破
     - 中波 (35-65%): 均值回归 → RSI极端反转
     - 高波 (>65%): 方向交易 + 期权叠加(卖covered call/put增收)

  2. 动态出场 (不用固定天数):
     - ATR跟踪止损 (趋势模式)
     - 目标止盈 (均值回归模式)
     - 最大持有期保护

  3. 期权叠加:
     - 持多头 + IV高位 → 卖OTM看涨(备兑)
     - 持空头 + IV高位 → 卖OTM看跌(备兑看跌)
     - 无方向 + IV极高 → 独立卖跨式(1个仓位)

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

        # === 动量 ===
        for lag in [3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        # 动量一致性
        mom_cols = ['m3','m5','m10','m20','m60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_c = np.sum(signs > 0, axis=1)
        neg_c = np.sum(signs < 0, axis=1)
        df['mom_dir'] = np.where(pos_c >= 4, 1.0, np.where(neg_c >= 4, -1.0, 0.0))

        # === 波动率 ===
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>10 else .5, raw=False)

        # === 趋势 ===
        for sp in [4, 8, 16]:
            ef = df['close'].ewm(span=sp).mean()
            es = df['close'].ewm(span=sp*4).mean()
            s = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ew{sp}'] = (ef - es) / s
        df['trend_score'] = df['ew4']*.2 + df['ew8']*.3 + df['ew16']*.3

        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # === RSI ===
        d = df['close'].diff(); g = d.where(d>0,0).rolling(14).mean()
        l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        # === ATR ===
        tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # === 布林带 ===
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_ma + 2 * bb_std
        df['bb_lower'] = bb_ma - 2 * bb_std
        df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)

        # === OI信号 ===
        if 'oi' in df.columns:
            df['oi_ma'] = df['oi'].rolling(20).mean()
            df['oi_signal'] = np.where(df['oi'] > df['oi_ma'], 1, -1)
        else:
            df['oi_signal'] = 0

        # === Volume信号 ===
        if 'vol' in df.columns:
            df['vol_ma'] = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / df['vol_ma'].replace(0, np.nan)
        else:
            df['vol_ratio'] = 1.0

        # === 突破信号 ===
        hh20 = df['close'].rolling(20).max()
        ll20 = df['close'].rolling(20).min()
        df['breakout_up'] = (df['close'] >= hh20 * 0.998).astype(float)
        df['breakout_dn'] = (df['close'] <= ll20 * 1.002).astype(float)

        # === 综合趋势方向 ===
        df['trend_dir'] = np.where(df['ma5'] > df['ma20'], 1., -1.)

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
            'rsi': df['rsi'].values.astype(np.float64),
            'atr': df['atr'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'mom_dir': df['mom_dir'].values.astype(np.float64),
            'trend_score': df['trend_score'].values.astype(np.float64),
            'trend_dir': df['trend_dir'].values.astype(np.float64),
            'bb_pos': df['bb_pos'].values.astype(np.float64),
            'rsi_raw': df['rsi'].values.astype(np.float64),
            'breakout_up': df['breakout_up'].values,
            'breakout_dn': df['breakout_dn'].values,
            'oi_signal': df['oi_signal'].values.astype(np.float64),
            'vol_ratio': df['vol_ratio'].values.astype(np.float64) if 'vol_ratio' in df.columns else np.ones(len(df)),
            'vol_sell': (df['hv_pct'].values > 0.65).astype(np.bool_),
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
    if avg_loss <= 0 or avg_win <= 0: return 0.02
    b = avg_win / avg_loss
    q = 1 - wr
    f = (wr * b - q) / b
    return max(min(f, 0.30), 0.01)


def bt(raw, dates, si, p):
    mp = p.get('max_pos', 3)
    mu = p.get('margin_usage', .90)
    mode = p.get('mode', 'adaptive')  # adaptive/trend/mr/covered_only
    atr_stop_mult = p.get('atr_stop', 2.0)  # ATR止损倍数
    atr_tp_mult = p.get('atr_tp', 3.0)  # ATR止盈倍数
    max_hold = p.get('max_hold', 15)  # 最大持有天数
    covered_otm = p.get('covered_otm', 0.02)
    straddle_width = p.get('straddle_width', 0.02)
    opt_hold = p.get('opt_hold', 7)
    kelly_mult = p.get('kelly_mult', 3.0)
    kelly_cap = p.get('kelly_cap', 0.08)
    notional_mult = p.get('notional_mult', 2.0)
    max_opt_pos = p.get('max_opt_pos', 1)  # 最多期权仓位

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

            if ps['ptype'] == 'future':
                # 动态止损/止盈
                should_exit = False
                exit_reason = 'hold'

                if h >= max_hold:
                    should_exit = True
                    exit_reason = 'max_hold'

                # ATR跟踪止损 (趋势模式)
                if ps.get('stop_type') == 'trailing':
                    trail = ps.get('trail_stop', 0)
                    # 更新跟踪止损
                    if ps['d'] > 0:  # 多头
                        new_trail = S - ps['atr_val'] * atr_stop_mult
                        if new_trail > trail:
                            ps['trail_stop'] = new_trail
                        if L <= ps['trail_stop']:
                            should_exit = True
                            exit_reason = 'trail_stop'
                    else:  # 空头
                        new_trail = S + ps['atr_val'] * atr_stop_mult
                        if trail == 0 or new_trail < trail:
                            ps['trail_stop'] = new_trail
                        if H >= ps['trail_stop']:
                            should_exit = True
                            exit_reason = 'trail_stop'

                # 固定止损/止盈 (均值回归模式)
                elif ps.get('stop_type') == 'fixed':
                    if ps['d'] > 0:  # 多头
                        if L <= ps.get('stop_loss', 0):
                            should_exit = True
                            exit_reason = 'stop_loss'
                        if H >= ps.get('take_profit', 1e9):
                            should_exit = True
                            exit_reason = 'take_profit'
                    else:  # 空头
                        if H >= ps.get('stop_loss', 1e9):
                            should_exit = True
                            exit_reason = 'stop_loss'
                        if L <= ps.get('take_profit', 0):
                            should_exit = True
                            exit_reason = 'take_profit'

                if not should_exit: continue

                # 计算期货PnL
                # 用止损/止盈价或收盘价
                if exit_reason == 'trail_stop' and ps['d'] > 0:
                    exit_price = max(L, ps['trail_stop'])
                elif exit_reason == 'trail_stop' and ps['d'] < 0:
                    exit_price = min(H, ps['trail_stop'])
                elif exit_reason == 'stop_loss' and ps['d'] > 0:
                    exit_price = max(L, ps['stop_loss'])
                elif exit_reason == 'stop_loss' and ps['d'] < 0:
                    exit_price = min(H, ps['stop_loss'])
                elif exit_reason == 'take_profit' and ps['d'] > 0:
                    exit_price = min(H, ps['take_profit'])
                elif exit_reason == 'take_profit' and ps['d'] < 0:
                    exit_price = max(L, ps['take_profit'])
                else:
                    exit_price = S

                pnl = (exit_price - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                comm = exit_price * ps['m'] * ps['fl'] * COMM

                # 备兑期权PnL
                opt_pnl = 0
                if ps.get('covered_type'):
                    cv_type = ps['covered_type']
                    Tr = max(ps['T'] - h/TD, .001)
                    if cv_type == 'call':
                        ov = bs(exit_price, ps['K_cov'], Tr, R, ps['sig'], 'call')
                        opt_pnl = (ps['cov_prem'] - max(ov,0)*ld) * ps['m'] * ps['fl']
                    else:
                        ov = bs(exit_price, ps['K_cov'], Tr, R, ps['sig'], 'put')
                        opt_pnl = (ps['cov_prem'] - max(ov,0)*ld) * ps['m'] * ps['fl']
                    opt_comm = max(ov, 0.01) * ps['m'] * ps['fl'] * COMM_OPT
                    comm += opt_comm

                eq += ps['fm'] + pnl + opt_pnl - comm
                pnls.append(pnl + opt_pnl - comm)
                trade_history.append((pnl + opt_pnl - comm, date))
                del pos[key]

            elif ps['ptype'] == 'option':
                if h < opt_hold: continue
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

        # === 入场 ===
        if len(pos) < mp:
            sigs = []
            n_opt = sum(1 for ps in pos.values() if ps['ptype'] == 'option')

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
                atr_pct = d['atr_pct'][pi]
                if np.isnan(atr_pct) or atr_pct > .06: continue

                rsi = d['rsi'][pi]
                mom_dir = d['mom_dir'][pi]
                trend_score = d['trend_score'][pi]
                trend_dir = d['trend_dir'][pi]
                bb_pos = d['bb_pos'][pi]
                brk_up = d['breakout_up'][pi]
                brk_dn = d['breakout_dn'][pi]
                oi_sig = d['oi_signal'][pi]

                # === 期货信号 ===
                direction = 0
                signal_type = 'none'
                priority = 0

                if mode == 'adaptive':
                    # 低波regime: 趋势跟踪
                    if hp < 0.35:
                        if mom_dir != 0 and mom_dir == trend_dir:
                            direction = mom_dir
                            signal_type = 'trend'
                            priority = 80
                            # 突破加分
                            if (direction > 0 and brk_up) or (direction < 0 and brk_dn):
                                priority += 15
                            # OI确认加分
                            if oi_sig == direction:
                                priority += 5

                    # 中波regime: 均值回归
                    elif hp < 0.65:
                        if rsi < 25 and bb_pos < 0.15:  # 超卖
                            direction = 1
                            signal_type = 'mr'
                            priority = 70
                        elif rsi > 75 and bb_pos > 0.85:  # 超买
                            direction = -1
                            signal_type = 'mr'
                            priority = 70

                    # 高波regime: 动量交易 (only strong signals)
                    else:
                        if mom_dir != 0 and mom_dir == trend_dir and abs(trend_score) > 1.5:
                            direction = mom_dir
                            signal_type = 'momentum'
                            priority = 60

                elif mode == 'trend':
                    if mom_dir != 0 and mom_dir == trend_dir:
                        direction = mom_dir
                        signal_type = 'trend'
                        priority = 80

                elif mode == 'mr':
                    if rsi < 25 and bb_pos < 0.15:
                        direction = 1; signal_type = 'mr'; priority = 70
                    elif rsi > 75 and bb_pos > 0.85:
                        direction = -1; signal_type = 'mr'; priority = 70

                if direction != 0:
                    sigs.append({
                        'type': 'future', 'sym': sym, 'dir': direction,
                        'hv': hv, 'hp': hp, 'signal': signal_type,
                        'atr': d['atr'][il], 'priority': priority,
                    })

                # === 期权信号: 高IV时独立卖跨式 ===
                if d['vol_sell'][pi] and hv > 0.20 and n_opt < max_opt_pos:
                    sigs.append({
                        'type': 'option', 'sym': sym, 'dir': 0,
                        'hv': hv, 'hp': hp, 'signal': 'vol_sell',
                        'atr': d['atr'][il], 'priority': 40 + hv * 50,
                    })

            # 排序
            sigs.sort(key=lambda x: -x['priority'])

            for sig in sigs:
                if len(pos) >= mp: break

                sym = sig['sym']
                im = si.get(sym)
                il = im[date]

                if sig['type'] == 'future':
                    # 限制期货仓位(为期权留1个位)
                    n_fut = sum(1 for ps in pos.values() if ps['ptype'] == 'future')
                    if n_fut >= mp - max_opt_pos: continue

                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    atr_val = sig['atr'] if not np.isnan(sig['atr']) else S * 0.02

                    mpl = S * ml * mr
                    target = eq * mu / mp
                    fl = max(int(target / mpl), 1)
                    fm = mpl * fl
                    fc = S * ml * fl * COMM
                    if fm + fc > eq * 0.8:
                        fl = max(int((eq*0.8 - fc)/mpl), 0)
                        if fl <= 0: continue
                        fm = mpl * fl; fc = S * ml * fl * COMM

                    eq -= fm + fc

                    ps = {
                        'ptype': 'future', 'sym': sym, 'ed': date,
                        'ep': S, 'd': sig['dir'], 'fl': fl, 'm': ml, 'fm': fm,
                        'signal': sig['signal'],
                        'atr_val': atr_val,
                    }

                    # 设置止损止盈
                    if sig['signal'] == 'trend':
                        # 趋势: ATR跟踪止损
                        ps['stop_type'] = 'trailing'
                        if sig['dir'] > 0:
                            ps['trail_stop'] = S - atr_val * atr_stop_mult
                        else:
                            ps['trail_stop'] = S + atr_val * atr_stop_mult
                    else:
                        # 均值回归/动量: 固定止损止盈
                        ps['stop_type'] = 'fixed'
                        if sig['dir'] > 0:
                            ps['stop_loss'] = S - atr_val * atr_stop_mult
                            ps['take_profit'] = S + atr_val * atr_tp_mult
                        else:
                            ps['stop_loss'] = S + atr_val * atr_stop_mult
                            ps['take_profit'] = S - atr_val * atr_tp_mult

                    # === 备兑期权 (高IV时) ===
                    if sig['hp'] > 0.55 and sig['hv'] > 0.15:
                        T_cov = max_hold / TD
                        if sig['dir'] > 0:
                            K_cov = S * (1 + covered_otm)
                            cov_prem = bs(S, K_cov, T_cov, R, sig['hv'], 'call')
                            if cov_prem > 0:
                                ps['covered_type'] = 'call'
                                ps['K_cov'] = K_cov
                                ps['T'] = T_cov
                                ps['sig'] = sig['hv']
                                ps['cov_prem'] = cov_prem
                        else:
                            K_cov = S * (1 - covered_otm)
                            cov_prem = bs(S, K_cov, T_cov, R, sig['hv'], 'put')
                            if cov_prem > 0:
                                ps['covered_type'] = 'put'
                                ps['K_cov'] = K_cov
                                ps['T'] = T_cov
                                ps['sig'] = sig['hv']
                                ps['cov_prem'] = cov_prem

                    pos[sym] = ps

                elif sig['type'] == 'option':
                    n_opt_now = sum(1 for ps in pos.values() if ps['ptype'] == 'option')
                    if n_opt_now >= max_opt_pos: continue

                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']
                    T = opt_hold / TD
                    K_call = S * (1 + straddle_width)
                    K_put = S * (1 - straddle_width)
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

                    pos[sym + '_VS'] = {
                        'ptype': 'option', 'sym': sym, 'ed': date, 'S': S,
                        'K_call': K_call, 'K_put': K_put,
                        'T': T, 'sig': sig['hv'], 'prem_received': total_prem,
                        'ml': ml, 'ct': ct,
                    }

        # === 权益 ===
        ur = 0.
        for key, ps in pos.items():
            sym = ps['sym']
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                if ps['ptype'] == 'future':
                    ur += (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                    if ps.get('covered_type'):
                        Tr = max(ps['T']-h/TD, .001)
                        if ps['covered_type'] == 'call':
                            ov = bs(S, ps['K_cov'], Tr, R, ps['sig'], 'call')
                        else:
                            ov = bs(S, ps['K_cov'], Tr, R, ps['sig'], 'put')
                        ur += (ps['cov_prem'] - max(ov,0)*ld) * ps['m'] * ps['fl']
                elif ps['ptype'] == 'option':
                    Tr = max(ps['T']-h/TD, .001)
                    cv = bs(S, ps['K_call'], Tr, R, ps['sig'], 'call')
                    pv = bs(S, ps['K_put'], Tr, R, ps['sig'], 'put')
                    ur += (ps['prem_received'] - (cv+pv)*ld) * ps['ml'] * ps['ct']
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

    # === A: 自适应模式 (核心) ===
    for mu in [.70, .80, .90, .95]:
        for atr_s in [1.5, 2.0, 2.5, 3.0]:
            for atr_tp in [2.5, 3.0, 4.0]:
                for cov in [0, .015, .020, .030]:
                    pl.append(dict(mode='adaptive', margin_usage=mu,
                                  atr_stop=atr_s, atr_tp=atr_tp,
                                  covered_otm=cov, max_hold=15))

    # === B: 纯趋势跟踪 ===
    for mu in [.80, .90, .95]:
        for atr_s in [2.0, 3.0]:
            for mh in [10, 15, 20]:
                pl.append(dict(mode='trend', margin_usage=mu,
                              atr_stop=atr_s, max_hold=mh))

    # === C: 纯均值回归 ===
    for mu in [.70, .80, .90]:
        for atr_s in [1.5, 2.0]:
            for atr_tp in [2.0, 3.0]:
                pl.append(dict(mode='mr', margin_usage=mu,
                              atr_stop=atr_s, atr_tp=atr_tp, max_hold=7))

    # === D: 自适应 + 期权增收 ===
    for mu in [.80, .90, .95]:
        for atr_s in [2.0, 2.5]:
            for cov in [.020, .030]:
                for sw in [.015, .020]:
                    for mop in [1]:
                        pl.append(dict(mode='adaptive', margin_usage=mu,
                                      atr_stop=atr_s, atr_tp=3.0,
                                      covered_otm=cov, straddle_width=sw,
                                      max_opt_pos=mop, max_hold=15,
                                      notional_mult=3.0, kelly_mult=4.0, kelly_cap=0.12))

    # === E: 高杠杆自适应 ===
    for mu in [.95, 1.0]:
        for atr_s in [2.0, 2.5, 3.0]:
            pl.append(dict(mode='adaptive', margin_usage=mu,
                          atr_stop=atr_s, atr_tp=3.0,
                          covered_otm=.020, max_hold=15,
                          notional_mult=3.0, kelly_mult=4.0, kelly_cap=0.12,
                          max_opt_pos=1))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 30 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'模式':>10} {'MU':>4} {'AS':>4} {'AT':>4} {'MH':>3} {'COV':>5} {'MOP':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*110)
    for r in res[:80]:
        mu = f"{r.get('margin_usage',0):.0%}"
        as_ = f"{r.get('atr_stop',0):.1f}"
        at_ = f"{r.get('atr_tp',0):.1f}"
        mh = f"{r.get('max_hold',15)}"
        cov = f"{r.get('covered_otm',0)*100:.1f}" if r.get('covered_otm',0)>0 else '-'
        mop = f"{r.get('max_opt_pos',0)}" if r.get('max_opt_pos',0)>0 else '-'
        print(f"{r['mode']:>10} {mu:>4} {as_:>4} {at_:>4} {mh:>3} {cov:>5} {mop:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*110)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),
                       (3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),
                       (.5,.50,"年化>=50% & WR>=50%"),
                       (.3,.50,"年化>=30% & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  {r['mode']}  MU={r.get('margin_usage',0):.0%}  "
                      f"AS={r.get('atr_stop',0):.1f}  AT={r.get('atr_tp',0):.1f}  "
                      f"COV={r.get('covered_otm',0)*100:.1f}%  MOP={r.get('max_opt_pos',0)}  "
                      f"Trades={r['trades']}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:500]]
    with open(os.path.join(od,'backtest_v56.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v56.json")


if __name__ == '__main__':
    main()
