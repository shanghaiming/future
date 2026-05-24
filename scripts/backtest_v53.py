#!/usr/bin/env python3
"""
策略 V53 — 期权增强期货策略 (Options-Informed Futures Trading)

核心思路: 期货做方向交易, 期权提供信号/对冲/增收

三层架构:
  Layer 1 [信号层]: 期权波动率曲面 + 期限结构 → 生成交易信号
    - IV百分位 → 波动率regime判断
    - Skew (虚值看跌vs看涨IV差) → 方向情绪
    - 期限结构 (contango/backwardation) → 展期收益信号
    - Delta/Gamma → 趋势强度和波动预期

  Layer 2 [执行层]: 期货方向性交易
    - 动量+趋势信号 → 期货多空
    - 期权信号确认 → 只在高概率方向交易
    - IV regime → 调整仓位大小

  Layer 3 [增效层]: 期权叠加
    - 持有期货多头 + IV高位 → 卖出虚值看涨(备兑/covered call)
    - 持有期货空头 + IV高位 → 卖出虚值看跌(备兑看跌/covered put)
    - 无方向但IV极高 → 卖跨式独立增收

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


def load_options_surface(data_dir):
    """加载期权波动率曲面数据 (单日快照, 用于校准)"""
    surfaces = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.json'): continue
        sym = f.split('_')[0]
        with open(os.path.join(data_dir, f)) as fh:
            d = json.load(fh)
        if 'surface' not in d or not d['surface']: continue

        # 提取关键指标
        surface = d['surface']
        underlying = d.get('underlying_price', 0)
        hv20 = d.get('hv_20', 0)

        # ATM IV (moneyness=1.0, 30day)
        atm_iv_30 = [x['iv'] for x in surface
                     if abs(x['moneyness']-1.0)<0.01 and x['expiry_days']==30 and x['flag']=='call']
        atm_iv_60 = [x['iv'] for x in surface
                     if abs(x['moneyness']-1.0)<0.01 and x['expiry_days']==60 and x['flag']=='call']

        # Skew: OTM put IV vs OTM call IV (30day)
        otm_put_iv = [x['iv'] for x in surface
                      if x['moneyness']<=0.88 and x['expiry_days']==30 and x['flag']=='put']
        otm_call_iv = [x['iv'] for x in surface
                       if x['moneyness']>=1.12 and x['expiry_days']==30 and x['flag']=='call']
        skew = (np.mean(otm_put_iv) - np.mean(otm_call_iv)) if otm_put_iv and otm_call_iv else 0

        # 期限结构斜率: 60day IV vs 30day IV
        ts_slope = (np.mean(atm_iv_60) - np.mean(atm_iv_30)) if atm_iv_30 and atm_iv_60 else 0

        # ATM delta (30day call)
        atm_delta = [abs(x['delta']) for x in surface
                     if abs(x['moneyness']-1.0)<0.01 and x['expiry_days']==30 and x['flag']=='call']

        # ATM gamma (30day)
        atm_gamma = [x['gamma'] for x in surface
                     if abs(x['moneyness']-1.0)<0.01 and x['expiry_days']==30 and x['flag']=='call']

        # ATM theta (30day)
        atm_theta = [abs(x['theta']) for x in surface
                     if abs(x['moneyness']-1.0)<0.01 and x['expiry_days']==30 and x['flag']=='call']

        # ATM vega (30day)
        atm_vega = [x['vega'] for x in surface
                    if abs(x['moneyness']-1.0)<0.01 and x['expiry_days']==30 and x['flag']=='call']

        surfaces[sym] = {
            'underlying': underlying,
            'hv20': hv20,
            'atm_iv_30': np.mean(atm_iv_30) if atm_iv_30 else hv20,
            'atm_iv_60': np.mean(atm_iv_60) if atm_iv_60 else hv20,
            'skew': skew,
            'ts_slope': ts_slope,
            'atm_delta': np.mean(atm_delta) if atm_delta else 0.5,
            'atm_gamma': np.mean(atm_gamma) if atm_gamma else 0,
            'atm_theta': np.mean(atm_theta) if atm_theta else 0,
            'atm_vega': np.mean(atm_vega) if atm_vega else 0,
        }
    return surfaces


def load_term_structure(data_dir):
    """加载期货期限结构数据"""
    ts_data = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.json'): continue
        sym = f.split('_')[0]
        with open(os.path.join(data_dir, f)) as fh:
            d = json.load(fh)
        curve = d.get('curve', [])
        if len(curve) < 2: continue

        # 兼容两种格式: close/open或price/year/month
        def _price(c):
            return c.get('close') or c.get('price') or 0

        near_price = _price(curve[0])
        far_price = _price(curve[-1])
        if near_price <= 0 or far_price <= 0: continue

        roll_yield = (near_price - far_price) / far_price
        structure = d.get('structure', 'flat')
        prices = [_price(c) for c in curve if _price(c) > 0]
        slope = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0

        ts_data[sym] = {
            'structure': structure,
            'roll_yield': roll_yield,
            'slope': slope,
            'near_price': near_price,
            'far_price': far_price,
            'spread_pct': (near_price - far_price) / near_price,
        }
    return ts_data


def load_futures(data_dir):
    """加载期货价格数据并计算所有信号"""
    raw = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'): continue
        sym = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300: continue
        df['ret'] = df['close'].pct_change()

        # === 动量信号 ===
        for lag in [3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        # 多时间框架动量一致性
        mom_cols = ['m3','m5','m10','m20','m60']
        signs = np.stack([np.sign(df[c].values) for c in mom_cols], axis=1)
        signs = np.where(np.isnan(signs), 0, signs)
        pos_c = np.sum(signs > 0, axis=1)
        neg_c = np.sum(signs < 0, axis=1)
        df['mom_dir'] = np.where(pos_c >= 4, 1.0, np.where(neg_c >= 4, -1.0, 0.0))
        df['mom_str'] = np.where(df['mom_dir']!=0,
                                  np.maximum(pos_c, neg_c)/len(mom_cols), 0.0)

        # === 波动率信号 (期权替代) ===
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)

        # IV proxy: HV5/HV20比率 → 短期波动率溢价
        df['iv_premium'] = (df['hv5'] / df['hv20'].replace(0,np.nan) - 1).clip(-0.5, 0.5)

        # HV百分位 (252日滚动)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)>10 else .5, raw=False)

        # 波动率regime
        df['vol_regime'] = pd.cut(df['hv_pct'], bins=[0,0.3,0.7,1.0],
                                   labels=['low','mid','high'], include_lowest=True).astype(str)

        # === 趋势信号 ===
        for sp in [4, 8, 16]:
            ef = df['close'].ewm(span=sp).mean()
            es = df['close'].ewm(span=sp*4).mean()
            s = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ew{sp}'] = (ef - es) / s

        # 综合趋势方向
        df['trend_score'] = df['ew4']*.2 + df['ew8']*.3 + df['ew16']*.3

        # MA趋势
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20']>df['ma60'], 1., -1.)

        # === 期权相关信号 ===
        # Skew proxy: 下行波动率 vs 上行波动率
        up = df['ret'].where(df['ret']>0,0).rolling(20).std()*np.sqrt(TD)
        dn = (-df['ret'].where(df['ret']<0,0)).rolling(20).std()*np.sqrt(TD)
        df['skew_proxy'] = (dn - up) / df['hv20'].replace(0, np.nan)
        # 正skew → 下行风险大 → 看跌情绪

        # 期限结构信号: 短期动量 vs 长期动量差
        df['ts_signal'] = df['m5'] - df['m60']  # 正=近期强于远期 → 趋势加速

        # === 风险管理指标 ===
        d = df['close'].diff(); g = d.where(d>0,0).rolling(14).mean()
        l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / df['close']

        # 波动率缩放因子
        df['vol_scale'] = (0.15 / df['hv20'].replace(0, np.nan)).clip(0.3, 3.0)

        # === OI信号 (持仓量) ===
        if 'oi' in df.columns:
            df['oi_change'] = df['oi'].pct_change(5)
            df['oi_ma'] = df['oi'].rolling(20).mean()
            df['oi_signal'] = np.where(df['oi'] > df['oi_ma'], 1, -1)  # OI增加=趋势确认

        # === 综合方向评分 ===
        # 结合动量、趋势、期权信号
        df = df.dropna(subset=['ma20','ma60','hv20','m5','rsi','trend_score'])
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
            'hv5': df['hv5'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'vol_scale': df['vol_scale'].values.astype(np.float64),
            'iv_premium': df['iv_premium'].values.astype(np.float64),
            'mom_dir': df['mom_dir'].values.astype(np.float64),
            'mom_str': df['mom_str'].values.astype(np.float64),
            'trend': df['trend'].values.astype(np.float64),
            'trend_score': df['trend_score'].values.astype(np.float64),
            'skew_proxy': df['skew_proxy'].values.astype(np.float64),
            'ts_signal': df['ts_signal'].values.astype(np.float64),
            'vol_regime': df['vol_regime'].values,
            'm5': df['m5'].values.astype(np.float64),
            'm10': df['m10'].values.astype(np.float64),
            'm20': df['m20'].values.astype(np.float64),
        }
        # OI if available
        if 'oi_signal' in df.columns:
            raw[sym]['oi_signal'] = df['oi_signal'].values.astype(np.float64)
        else:
            raw[sym]['oi_signal'] = np.zeros(len(df))
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
    return max(min(f, 0.25), 0.01)


def bt(raw, dates, si, p):
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 7)
    mu = p.get('margin_usage', .90)
    mode = p.get('mode', 'hybrid')  # hybrid/futures_only/futures_hedged
    straddle_width = p.get('straddle_width', 0.02)
    ld = p.get('liq_disc', .9)
    kelly_mult = p.get('kelly_mult', 3.0)
    kelly_cap = p.get('kelly_cap', 0.08)
    notional_mult = p.get('notional_mult', 2.0)
    vol_sell_threshold = p.get('vol_sell_threshold', 0.70)
    covered_otm = p.get('covered_otm', 0.02)

    eq = float(INIT)
    pos = {}  # sym -> position info
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

            if ps['type'] == 'future':
                if h < hd: continue

                # 计算期货PnL
                pnl = (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                comm = S * ps['m'] * ps['fl'] * COMM
                net_pnl = pnl - comm

                # 如果有备兑期权, 计算期权PnL
                if ps.get('covered_type'):
                    cv_type = ps['covered_type']
                    Tr = max(ps['T'] - h/TD, .001)
                    if cv_type == 'call':
                        opt_val = bs(S, ps['K_covered'], Tr, R, ps['sig'], 'call')
                        # 卖出看涨: 收益 = 权利金 - 平仓成本
                        opt_pnl = (ps['covered_prem'] - max(opt_val, 0) * ld) * ps['m'] * ps['fl']
                    else:  # put
                        opt_val = bs(S, ps['K_covered'], Tr, R, ps['sig'], 'put')
                        opt_pnl = (ps['covered_prem'] - max(opt_val, 0) * ld) * ps['m'] * ps['fl']
                    opt_comm = max(opt_val, 0.01) * ps['m'] * ps['fl'] * COMM_OPT
                    net_pnl += opt_pnl - opt_comm

                eq += ps['fm'] + net_pnl
                pnls.append(net_pnl)
                trade_history.append((net_pnl, date))
                del pos[sym]

            elif ps['type'] == 'straddle':
                if h < hd: continue
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
                if np.isnan(hv) or hv < .05 or hv > .80: continue
                hp = d['hv_pct'][pi]
                if np.isnan(hp) or hp > .95: continue
                rsi = d['rsi'][pi]
                if np.isnan(rsi) or rsi > 82 or rsi < 18: continue
                ap = d['atr_pct'][pi]
                if np.isnan(ap) or ap > .06: continue

                mom_dir = d['mom_dir'][pi]
                mom_str = d['mom_str'][pi]
                trend = d['trend'][pi]
                trend_score = d['trend_score'][pi]
                skew = d['skew_proxy'][pi]
                ts_sig = d['ts_signal'][pi]
                oi_sig = d['oi_signal'][pi]

                # === 综合方向评分 ===
                score = 0.0
                weight = 0.0

                # 动量一致性 (权重最高)
                if mom_dir != 0 and not np.isnan(mom_str):
                    score += mom_dir * mom_str * 3.0
                    weight += 3.0

                # EWMAC趋势
                if not np.isnan(trend_score):
                    score += np.sign(trend_score) * min(abs(trend_score), 3) * 2.0
                    weight += 2.0

                # MA趋势方向
                if trend != 0:
                    score += trend * 1.5
                    weight += 1.5

                # 期限结构信号 (正ts=近强远弱=趋势加速)
                if not np.isnan(ts_sig):
                    score += np.sign(ts_sig) * min(abs(ts_sig)*5, 1.5) * 1.0
                    weight += 1.0

                # Skew信号 (正skew=看跌情绪 → 做空信号)
                if not np.isnan(skew):
                    score += -np.sign(skew) * min(abs(skew), 1.5) * 1.0
                    weight += 1.0

                # OI信号
                if not np.isnan(oi_sig):
                    score += oi_sig * 0.5
                    weight += 0.5

                if weight < 3: continue
                dir_score = score / weight  # 归一化到 [-1, 1]

                # === 信号过滤 ===
                # 1. 方向一致性: mom_dir, trend, dir_score三者必须一致
                direction = np.sign(dir_score)
                if direction == 0: continue
                if mom_dir != 0 and mom_dir != direction: continue
                if trend != direction: continue

                # 2. 信号强度
                abs_score = abs(dir_score)
                if abs_score < 0.3: continue

                # 3. IV regime过滤: 极高IV时降仓, 极低时加仓
                iv_filter = 1.0
                if hp > 0.8:
                    iv_filter = 0.5  # 极高IV, 降低仓位
                elif hp < 0.3:
                    iv_filter = 1.5  # 低IV, 可以更激进

                sigs.append(('future', sym, direction, abs_score, hv, hp, iv_filter, il))

            # 额外: IV极高时加入纯波动率卖出信号
            if mode in ('hybrid', 'futures_hedged'):
                for sym, d in raw.items():
                    if sym in pos: continue
                    if len(pos) + len([s for s in sigs]) >= mp * 2: break
                    im = si.get(sym)
                    if not im or date not in im: continue
                    il = im[date]
                    if il <= 0: continue
                    pi = il - 1

                    hv = d['hv20'][pi]
                    if np.isnan(hv) or hv < 0.20: continue
                    hp = d['hv_pct'][pi]
                    if np.isnan(hp) or hp < vol_sell_threshold: continue

                    sigs.append(('straddle', sym, 0, 0, hv, hp, 1.0, il))

            # 排序: future按score降序, straddle按HV降序
            sigs.sort(key=lambda x: (0 if x[0]=='future' else 1, -x[3]))

            for sig in sigs:
                if len(pos) >= mp: break
                stype = sig[0]
                sym = sig[1]

                if stype == 'future':
                    _, sym, direction, abs_score, hv, hp, iv_filter, il = sig
                    S = raw[sym]['open'][il]
                    if np.isnan(S) or S <= 0: continue
                    ml, mr, _, _ = raw[sym]['spec']

                    # 期货仓位计算
                    mpl = S * ml * mr  # 一手保证金
                    # IV regime调整后的仓位
                    eff_mu = mu * iv_filter
                    # 信号强度加权
                    eff_mu *= min(abs_score, 1.5)
                    target = eq * eff_mu / mp
                    fl = max(int(target/mpl), 1)
                    fm = mpl * fl
                    fc = S * ml * fl * COMM

                    if fm + fc > eq * 0.95:
                        fl = max(int((eq*0.95 - fc)/mpl), 0)
                        if fl <= 0: continue
                        fm = mpl * fl; fc = S * ml * fl * COMM

                    eq -= fm + fc

                    pos_info = {
                        'type': 'future',
                        'd': direction,  # 1=多, -1=空
                        'ed': date,
                        'ep': S,  # 入场价
                        'fl': fl,
                        'm': ml,
                        'fm': fm,
                    }

                    # === 期权叠加: 备兑策略 ===
                    # 持有期货 + IV高位 → 卖出虚值期权增收
                    if mode in ('hybrid', 'futures_hedged') and hp > 0.6 and hv > 0.15:
                        if direction > 0:
                            # 持多头 → 卖虚值看涨(covered call)
                            K_cov = S * (1 + covered_otm)
                            T_cov = hd / TD
                            cov_prem = bs(S, K_cov, T_cov, R, hv, 'call')
                            if cov_prem > 0:
                                pos_info['covered_type'] = 'call'
                                pos_info['K_covered'] = K_cov
                                pos_info['T'] = T_cov
                                pos_info['sig'] = hv
                                pos_info['covered_prem'] = cov_prem
                        else:
                            # 持空头 → 卖虚值看跌(covered put)
                            K_cov = S * (1 - covered_otm)
                            T_cov = hd / TD
                            cov_prem = bs(S, K_cov, T_cov, R, hv, 'put')
                            if cov_prem > 0:
                                pos_info['covered_type'] = 'put'
                                pos_info['K_covered'] = K_cov
                                pos_info['T'] = T_cov
                                pos_info['sig'] = hv
                                pos_info['covered_prem'] = cov_prem

                    pos[sym] = pos_info

                elif stype == 'straddle':
                    _, sym, _, _, hv, hp, _, il = sig
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

                    pos[sym + '_vs'] = {'type':'straddle','ed':date,'S':S,
                                        'K_call':K_call,'K_put':K_put,
                                        'T':T,'sig':hv,'prem_received':total_prem,
                                        'ml':ml,'ct':ct}

        # === 权益计算 ===
        ur = 0.
        for sym, ps in pos.items():
            actual_sym = sym.replace('_vs', '')
            im = si.get(actual_sym)
            if im and date in im:
                S = raw[actual_sym]['close'][im[date]]
                h = int((date-ps['ed'])/np.timedelta64(1,'D'))
                if ps['type'] == 'future':
                    ur += (S - ps['ep']) * ps['d'] * ps['m'] * ps['fl']
                    # 备兑期权未实现
                    if ps.get('covered_type'):
                        Tr = max(ps['T']-h/TD, .001)
                        if ps['covered_type'] == 'call':
                            ov = bs(S, ps['K_covered'], Tr, R, ps['sig'], 'call')
                        else:
                            ov = bs(S, ps['K_covered'], Tr, R, ps['sig'], 'put')
                        ur += (ps['covered_prem'] - max(ov,0)*ld) * ps['m'] * ps['fl']
                elif ps['type'] == 'straddle':
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

    # 统计期货vs期权占比
    n_fut = sum(1 for r in res if False) if False else 0  # placeholder

    return {'annual':ann,'wr':wr,'mdd':mdd,'pf':pf,'trades':len(pa),
            'final':eq,'sharpe':sh,**p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    opt_dir = os.path.expanduser("~/home/futures_platform/data/options")
    ts_dir = os.path.expanduser("~/home/futures_platform/data/futures_term_structure")

    print("加载数据..."); t0=time.time()

    # 加载期权和期限结构数据 (用于校准)
    opt_surfaces = load_options_surface(opt_dir)
    ts_data = load_term_structure(ts_dir)
    print(f"  期权曲面: {len(opt_surfaces)}品种")
    print(f"  期限结构: {len(ts_data)}品种")

    # 加载期货数据
    raw = load_futures(dd)
    print(f"  期货: {len(raw)}品种, {time.time()-t0:.1f}s")

    sd = pd.Timestamp('2018-01-01'); ed = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, sd, ed)
    print(f"  {len(dates)}交易日")

    pl = []

    # === A: 纯期货 (对照) ===
    for hd in [5, 7, 10]:
        for mu in [.70, .80, .90]:
            pl.append(dict(hold_days=hd, mode='futures_only', margin_usage=mu))

    # === B: 期货+备兑期权 ===
    for hd in [5, 7, 10]:
        for mu in [.70, .80, .90]:
            for cov in [.01, .02, .03]:
                pl.append(dict(hold_days=hd, mode='futures_hedged', margin_usage=mu,
                              covered_otm=cov))

    # === C: 混合 (期货+波动率卖出) ===
    for hd in [5, 7]:
        for mu in [.80, .90]:
            for sw in [.015, .020, .025]:
                for vs in [0.60, 0.70, 0.80]:
                    pl.append(dict(hold_days=hd, mode='hybrid', margin_usage=mu,
                                  straddle_width=sw, vol_sell_threshold=vs,
                                  covered_otm=.02))

    # === D: 高杠杆期货+备兑 ===
    for hd in [5, 7]:
        for mu in [.95, 1.0]:
            for cov in [.015, .02, .03]:
                pl.append(dict(hold_days=hd, mode='futures_hedged', margin_usage=mu,
                              covered_otm=cov, notional_mult=3.0,
                              kelly_mult=4.0, kelly_cap=0.12))

    # === E: 混合+高杠杆期权 ===
    for hd in [5, 7]:
        for mu in [.80, .90]:
            for sw in [.010, .015, .020]:
                pl.append(dict(hold_days=hd, mode='hybrid', margin_usage=mu,
                              straddle_width=sw, vol_sell_threshold=0.65,
                              covered_otm=.02, notional_mult=3.0,
                              kelly_mult=4.0, kelly_cap=0.12))

    print(f"\n参数: {len(pl)}组")
    bt0 = time.time()
    res = []
    for i, p in enumerate(pl):
        if i % 30 == 0: print(f"  [{i}/{len(pl)}] {time.time()-bt0:.0f}s...")
        r = bt(raw, dates, si, p)
        if r: res.append(r)

    print(f"\n耗时: {time.time()-t0:.0f}s, {len(res)}组有效结果")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'模式':>14} {'H':>3} {'MU':>5} {'SW':>5} {'COV':>5} {'VS':>4} {'NM':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-"*120)
    for r in res[:60]:
        mu = f"{r.get('margin_usage',0):.0%}"
        sw = f"{r.get('straddle_width',0)*100:.1f}" if r.get('straddle_width',0)>0 else '-'
        cov = f"{r.get('covered_otm',0)*100:.1f}" if r.get('covered_otm',0)>0 else '-'
        vs = f"{r.get('vol_sell_threshold',0):.0%}" if r.get('vol_sell_threshold',0)<1 else '-'
        nm = f"{r.get('notional_mult',2):.0f}" if r.get('notional_mult',2)!=2 else '-'
        print(f"{r['mode']:>14} {r['hold_days']:>3} {mu:>5} {sw:>5} {cov:>5} {vs:>4} {nm:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "="*120)
    for ta, tw, lb in [(6.,.50,"年化>=600% & WR>=50%"),
                       (6.,.45,"年化>=600% & WR>=45%"),
                       (4.,.50,"年化>=400% & WR>=50%"),
                       (3.,.50,"年化>=300% & WR>=50%"),
                       (1.,.50,"年化>=100% & WR>=50%"),
                       (.5,.50,"年化>=50% & WR>=50%")]:
        print(f"\n=== {lb} ===")
        g = [r for r in res if r['annual']>=ta and r['wr']>=tw]
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:8]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  {r['mode']}  H={r['hold_days']}  "
                      f"MU={r.get('margin_usage',0):.0%}  SW={r.get('straddle_width',0)*100:.1f}%  "
                      f"COV={r.get('covered_otm',0)*100:.1f}%  VS={r.get('vol_sell_threshold',0):.0%}  "
                      f"Trades={r['trades']}")
        else: print("  无")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in r.items()} for r in res[:500]]
    with open(os.path.join(od,'backtest_v53.json'),'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v53.json")


if __name__ == '__main__':
    main()
