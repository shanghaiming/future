#!/usr/bin/env python3
"""
策略 V47 — 精炼版: 多信号融合 + 期权增强 + 动态管理

关键改进:
  1. 信号融合: 只在多维度信号一致时交易 (提高胜率)
  2. Delta动态退出: 当delta衰减时提前退出 (保住利润)
  3. 自适应OTM: 根据HV自动调整行权价距离
  4. 仓位金字塔: 信号极强时加仓 (提高收益)
  5. 时间衰减管理: 接近到期时调整策略

目标: 年化600%, 最大持仓3, 胜率>50%
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.0003
R_RATE = 0.02
INIT = 500000
TD = 252


def bs_price(S, K, T, r, sigma, flag='call'):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if flag == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) if flag == 'call' \
        else (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def load_data(data_dir):
    raw = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'):
            continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300:
            continue

        df['return'] = df['close'].pct_change()
        for w in [5, 10, 20, 40, 60, 120]:
            df[f'hv_{w}'] = df['return'].rolling(w).std() * np.sqrt(TD)

        df['iv_ts'] = (df['hv_20'] / df['hv_60']).replace(0, np.nan)
        df['iv_ts2'] = (df['hv_5'] / df['hv_20']).replace(0, np.nan)
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False)

        for lag in [3, 5, 10, 20, 60]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)
        df['ts_signal'] = df['mom_5'] - df['mom_60']

        df['vol_ratio'] = df['vol'].rolling(5).mean() / df['vol'].rolling(20).mean().replace(0, np.nan)
        if 'oi' in df.columns:
            df['oi_chg5'] = df['oi'].pct_change(5)
            df['oi_chg20'] = df['oi'].pct_change(20)

        up = df['return'].where(df['return'] > 0, 0).rolling(20).std() * np.sqrt(TD)
        dn = (-df['return'].where(df['return'] < 0, 0)).rolling(20).std() * np.sqrt(TD)
        df['vol_skew'] = (dn - up) / df['hv_20'].replace(0, np.nan)

        for span in [4, 8, 16]:
            ef = df['close'].ewm(span=span).mean()
            es = df['close'].ewm(span=span * 4).mean()
            std = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ewmac_{span}'] = (ef - es) / std

        df['synth_delta'] = df['ewmac_4'] * 0.2 + df['ewmac_8'] * 0.3 + df['ewmac_16'] * 0.3
        df['synth_gamma'] = (df['hv_20'].diff(5) / df['hv_20'].shift(5).replace(0, np.nan)).rolling(10).std()

        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1.0, -1.0)
        for w in [10, 20, 40]:
            hh, ll = df['close'].rolling(w).max(), df['close'].rolling(w).min()
            df[f'bo_{w}'] = (df['close'] - 0.5*(hh+ll)) / (hh-ll+0.001) * 2

        s20 = df['return'].rolling(20).std().replace(0, np.nan)
        df['ramom'] = df['mom_10'] / (s20 * np.sqrt(10))

        d = df['close'].diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        tr = pd.concat([df['high']-df['low'],
                        abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / df['close']

        df = df.dropna(subset=['ma20','ma60','hv_20','mom_5','rsi','synth_delta'])
        if len(df) < 100:
            continue
        try:
            spec = get_spec(symbol)
        except:
            continue

        # ── 预计算多维一致性信号 ──
        # 每个维度给出方向和强度
        def _clip(x, c):
            return np.sign(x) * np.minimum(np.abs(x), c)

        sd = df['synth_delta'].values
        ts = df['ts_signal'].values
        iv_ts = df['iv_ts'].values
        trend = df['trend'].values
        skew = df['vol_skew'].values
        bo20 = df['bo_20'].values
        bo40 = df['bo_40'].values
        ramom = df['ramom'].values
        gamma = df['synth_gamma'].values
        hv_pct = df['hv_pct'].values
        mom5 = df['mom_5'].values
        vol_ratio = df['vol_ratio'].values

        # 维度1: Delta方向 (EWMAC)
        dim1 = np.where(~np.isnan(sd), np.sign(sd), 0)

        # 维度2: 动量方向
        dim2 = np.where(~np.isnan(mom5), np.sign(mom5), 0)

        # 维度3: 趋势方向
        dim3 = trend.copy()

        # 维度4: 突破方向
        dim4 = np.where(~np.isnan(bo20), np.sign(bo20), 0)

        # 维度5: 期限结构方向
        dim5 = np.where(~np.isnan(ts), np.sign(ts), 0)

        # 维度6: IV期限结构方向 (IV升→波动率升→反向)
        dim6 = np.where(~np.isnan(iv_ts) & (iv_ts > 0),
                        np.where(iv_ts > 1.2, -1.0, np.where(iv_ts < 0.8, 1.0, 0.0)), 0)

        # 维度7: Skew方向 (负skew=恐慌→看多)
        dim7 = np.where(~np.isnan(skew), -np.sign(skew), 0)

        # 一致性计数 (有多少维度方向一致)
        all_dims = np.stack([dim1, dim2, dim3, dim4, dim5, dim6, dim7], axis=1)

        # 计算一致方向和强度
        consensus_sum = np.nansum(all_dims, axis=1)  # 范围 -7 到 +7
        consensus_count = np.sum(np.abs(all_dims) > 0, axis=1)  # 有多少维度有信号
        consensus_pct = np.where(consensus_count > 0,
                                 np.abs(consensus_sum) / consensus_count, 0)

        # 综合得分: 方向 × 一致性强度 × 各维度加权得分
        # 方向
        direction = np.sign(consensus_sum)
        # 信号强度 = 一致性比例 × 最少需要一致维度数
        strength = consensus_pct

        # 加权得分 (用于排名)
        score = np.zeros(len(df))
        w = np.zeros(len(df))

        m = ~np.isnan(sd); score[m] += _clip(sd[m], 3)*3; w[m] += 3
        m = ~np.isnan(ts); score[m] += _clip(ts[m]*20, 2); w[m] += 2
        m = ~np.isnan(iv_ts)&(iv_ts>0); score[m] += _clip(-(iv_ts[m]-1)*5, 1.5); w[m] += 1
        m = ~np.isnan(skew); score[m] += _clip(-skew[m], 2); w[m] += 1.5
        m = ~np.isnan(bo20); score[m] += bo20[m]; w[m] += 1
        m = ~np.isnan(ramom); score[m] += _clip(ramom[m], 3)*1.5; w[m] += 1.5
        m = (trend!=0)&~np.isnan(score)&(np.sign(score)==trend); score[m] *= 1.4
        m = ~np.isnan(gamma)&(gamma>0.25); score[m] *= 0.7; w[m] *= 0.7
        score = np.where(w > 0, score/np.maximum(w, 1), 0)

        raw[symbol] = {
            'spec': spec,
            'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'hv_20': df['hv_20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'atr_pct': df['atr_pct'].values.astype(np.float64),
            'synth_gamma': df['synth_gamma'].values.astype(np.float64),
            'direction': direction.astype(np.float64),
            'strength': strength.astype(np.float64),
            'consensus': consensus_sum.astype(np.float64),
            'score': score.astype(np.float64),
        }
    return raw


def build_idx(raw, start_date, end_date):
    all_dates = set()
    for d in raw.values():
        m = (d['dates'] >= start_date) & (d['dates'] <= end_date)
        for dt in d['dates'][m]:
            all_dates.add(dt)
    dates = np.array(sorted(all_dates))
    sym_idx = {}
    for sym, d in raw.items():
        idx_map = {}
        m = (d['dates'] >= start_date) & (d['dates'] <= end_date)
        for dt, iloc in zip(d['dates'][m], np.where(m)[0]):
            idx_map[dt] = int(iloc)
        sym_idx[sym] = idx_map
    return dates, sym_idx


def run_backtest(raw, dates, sym_idx, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 7)
    risk_pct = params.get('risk_pct', 0.03)
    otm_pct = params.get('otm_pct', 0.01)
    min_score = params.get('min_score', 0.3)
    min_consensus = params.get('min_consensus', 3)  # 最少一致维度数
    min_strength = params.get('min_strength', 0.5)  # 一致性强度阈值
    delta_exit = params.get('delta_exit', 0.0)  # delta衰减退出阈值 (0=不用)
    liq_disc = params.get('liq_disc', 0.9)
    hv_lo = params.get('hv_lo', 0.10)
    hv_hi = params.get('hv_hi', 0.60)
    gamma_filter = params.get('gamma_filter', 0.0)
    # 自适应OTM: OTM = otm_pct * (hv_20 / 0.3)
    adaptive_otm = params.get('adaptive_otm', False)
    # 信号强度加权仓位
    strength_sizing = params.get('strength_sizing', False)

    equity = float(INIT)
    positions = {}
    closed_pnls = []
    eq_hist = [float(INIT)]

    for date in dates:
        # ── 退出 ──
        for sym in list(positions):
            pos = positions[sym]
            idx_map = sym_idx.get(sym)
            if not idx_map or date not in idx_map:
                continue
            iloc = idx_map[date]
            S_now = raw[sym]['close'][iloc]
            hd = int((date - pos['entry_date']) / np.timedelta64(1, 'D'))

            should_close = hd >= hold_days

            # Delta衰减退出: 如果方向反转或delta大幅衰减
            if not should_close and delta_exit > 0 and iloc > 0:
                pi = iloc - 1
                current_dir = raw[sym]['direction'][pi]
                if current_dir * pos['direction'] < 0:
                    should_close = True  # 信号反转

            # 盈利保护: 如果已经盈利超过阈值
            if not should_close and 'take_profit_pct' in params:
                T_rem = max(pos['T'] - hd / TD, 0.001)
                opt_val = bs_price(S_now, pos['K'], T_rem, R_RATE, pos['sigma'], pos['flag'])
                intrinsic = max(S_now - pos['K'], 0) if pos['flag'] == 'call' else max(pos['K'] - S_now, 0)
                exit_val = max(opt_val, intrinsic) * liq_disc
                pnl_pct = (exit_val - pos['premium']) / pos['premium'] if pos['premium'] > 0 else 0
                if pnl_pct > params['take_profit_pct']:
                    should_close = True

            if should_close:
                T_rem = max(pos['T'] - hd / TD, 0.001)
                opt_val = bs_price(S_now, pos['K'], T_rem, R_RATE, pos['sigma'], pos['flag'])
                intrinsic = max(S_now - pos['K'], 0) if pos['flag'] == 'call' else max(pos['K'] - S_now, 0)
                exit_val = max(opt_val, intrinsic) * liq_disc
                pnl = (exit_val - pos['premium']) * pos['direction'] * pos['mult'] * pos['contracts']
                comm_exit = exit_val * pos['mult'] * pos['contracts'] * COMM
                net = pnl - comm_exit
                equity += net
                closed_pnls.append(net)
                del positions[sym]

        # ── 入场 ──
        if len(positions) < max_pos:
            sigs = []
            for sym, d in raw.items():
                if sym in positions:
                    continue
                idx_map = sym_idx.get(sym)
                if not idx_map or date not in idx_map:
                    continue
                iloc = idx_map[date]
                if iloc <= 0:
                    continue
                pi = iloc - 1

                # 基本过滤
                hv = d['hv_20'][pi]
                if np.isnan(hv) or hv < hv_lo or hv > hv_hi:
                    continue
                hp = d['hv_pct'][pi]
                if np.isnan(hp) or hp > 0.90:
                    continue
                rsi = d['rsi'][pi]
                if np.isnan(rsi) or rsi > 78 or rsi < 22:
                    continue
                ap = d['atr_pct'][pi]
                if np.isnan(ap) or ap > 0.06:
                    continue
                if gamma_filter > 0:
                    sg = d['synth_gamma'][pi]
                    if not np.isnan(sg) and sg > gamma_filter:
                        continue

                # 一致性过滤
                dir_val = d['direction'][pi]
                strength_val = d['strength'][pi]
                consensus_val = abs(d['consensus'][pi])
                score_val = d['score'][pi]

                if dir_val == 0:
                    continue
                if consensus_val < min_consensus:
                    continue
                if strength_val < min_strength:
                    continue
                if abs(score_val) < min_score:
                    continue

                sigs.append((sym, dir_val, score_val, strength_val, consensus_val, iloc, hv))

            sigs.sort(key=lambda x: x[2]*x[3], reverse=True)  # score × strength

            for sym, dir_val, score_val, strength_val, consensus_val, iloc, hv in sigs:
                if len(positions) >= max_pos:
                    break

                S = raw[sym]['open'][iloc]
                if np.isnan(S) or S <= 0:
                    S = raw[sym]['close'][iloc]
                if S <= 0:
                    continue

                flag = 'call' if dir_val > 0 else 'put'

                # 自适应OTM
                if adaptive_otm:
                    actual_otm = otm_pct * (hv / 0.3)  # HV高→更远OTM
                    actual_otm = min(actual_otm, 0.05)   # 上限5%
                else:
                    actual_otm = otm_pct

                K = S * (1 + actual_otm * dir_val)
                T = hold_days / TD
                sigma = hv

                premium = bs_price(S, K, T, R_RATE, sigma, flag)
                if premium <= 0:
                    continue

                mult, mr, _, _ = raw[sym]['spec']

                # 仓位计算
                if strength_sizing:
                    # 信号越强仓位越大
                    adj_risk = risk_pct * (0.5 + strength_val)
                    adj_risk = min(adj_risk, 0.08)
                else:
                    adj_risk = risk_pct

                risk_amount = equity * adj_risk
                contracts = max(int(risk_amount / (premium * mult)), 1)
                max_contracts = max(int(equity * 5 / (S * mult)), 1)
                contracts = min(contracts, max_contracts)

                cost = premium * mult * contracts
                entry_comm = cost * COMM
                total_cost = cost + entry_comm

                if total_cost > equity * 0.10:
                    contracts = max(int(equity * 0.10 / (premium * mult + 1)), 1)
                    cost = premium * mult * contracts
                    entry_comm = cost * COMM
                    total_cost = cost + entry_comm

                if total_cost > equity * 0.3:
                    continue

                positions[sym] = {
                    'direction': dir_val, 'entry_date': date,
                    'entry_price': S, 'K': K, 'T': T, 'sigma': sigma,
                    'premium': premium, 'flag': flag,
                    'mult': mult, 'contracts': contracts, 'cost': total_cost,
                }

        # 权益
        unrealized = 0.0
        for sym, pos in positions.items():
            idx_map = sym_idx.get(sym)
            if idx_map and date in idx_map:
                S_now = raw[sym]['close'][idx_map[date]]
                hd = int((date - pos['entry_date']) / np.timedelta64(1, 'D'))
                T_rem = max(pos['T'] - hd / TD, 0.001)
                opt_val = bs_price(S_now, pos['K'], T_rem, R_RATE, pos['sigma'], pos['flag'])
                intrinsic = max(S_now - pos['K'], 0) if pos['flag'] == 'call' else max(pos['K'] - S_now, 0)
                unrealized += (max(opt_val, intrinsic) * liq_disc - pos['premium']) * pos['direction'] * pos['mult'] * pos['contracts']

        current_eq = equity + unrealized
        eq_hist.append(current_eq)
        if current_eq < 1000:
            break

    if not closed_pnls or equity <= 0:
        return None
    total_ret = (equity - INIT) / INIT
    if total_ret <= -1:
        return None

    days = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
    years = max(days / 365, 0.001)
    ann = float((1 + total_ret) ** (1 / years) - 1)

    pnls = np.array(closed_pnls)
    wr = float((pnls > 0).mean()) if len(pnls) > 0 else 0
    avg_w = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0
    avg_l = float(abs(pnls[pnls <= 0].mean())) if (pnls <= 0).any() else 1
    pf = avg_w * (pnls > 0).sum() / (avg_l * (pnls <= 0).sum()) if (pnls <= 0).sum() > 0 and avg_l > 0 else 0

    eq = np.array(eq_hist[1:])
    if len(eq) > 1:
        cummax = np.maximum.accumulate(eq)
        dd = (eq - cummax) / cummax
        mdd = float(dd.min())
        rets = np.diff(eq) / eq[:-1]
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    else:
        mdd = 0; sharpe = 0

    return {
        'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf,
        'trades': len(pnls), 'final': equity, 'sharpe': sharpe,
        **{k: v for k, v in params.items()},
    }


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载+预计算...")
    t0 = time.time()
    raw = load_data(data_dir)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2026-05-08')
    dates, sym_idx = build_idx(raw, start_date, end_date)
    print(f"  {len(dates)}交易日")

    params_list = []

    # A: 共识度过滤 + 期权参数
    for hd in [5, 7, 10]:
        for rp in [0.02, 0.03, 0.04]:
            for otm in [0.0, 0.01, 0.02, 0.03]:
                for mc in [2, 3, 4]:
                    for ms in [0.3, 0.5]:
                        for mstr in [0.4, 0.6]:
                            params_list.append(dict(
                                hold_days=hd, risk_pct=rp, otm_pct=otm,
                                min_score=ms, min_consensus=mc, min_strength=mstr,
                                max_pos=3))

    # B: 动态OTM + 强度加权
    for hd in [5, 7, 10]:
        for rp in [0.03, 0.04]:
            for mc in [3, 4]:
                params_list.append(dict(
                    hold_days=hd, risk_pct=rp, otm_pct=0.02,
                    min_score=0.3, min_consensus=mc, min_strength=0.5,
                    max_pos=3, adaptive_otm=True, strength_sizing=True))

    # C: Delta退出 + 盈利保护
    for hd in [7, 10]:
        for rp in [0.03, 0.04]:
            for tp in [1.0, 2.0, 3.0]:  # 期权盈利100%/200%/300%
                for mc in [3, 4]:
                    params_list.append(dict(
                        hold_days=hd, risk_pct=rp, otm_pct=0.01,
                        min_score=0.3, min_consensus=mc, min_strength=0.5,
                        max_pos=3, delta_exit=True, take_profit_pct=tp))

    # D: Gamma过滤 + 高OTM
    for hd in [5, 7]:
        for rp in [0.03, 0.04]:
            for otm in [0.02, 0.03]:
                for gf in [0.2, 0.3]:
                    params_list.append(dict(
                        hold_days=hd, risk_pct=rp, otm_pct=otm,
                        min_score=0.3, min_consensus=3, min_strength=0.5,
                        max_pos=3, gamma_filter=gf))

    # E: ATM (高胜率)
    for hd in [5, 7, 10]:
        for rp in [0.02, 0.03]:
            for mc in [3, 4]:
                params_list.append(dict(
                    hold_days=hd, risk_pct=rp, otm_pct=0.0,
                    min_score=0.3, min_consensus=mc, min_strength=0.5,
                    max_pos=3, hv_hi=0.50))

    print(f"\n参数组合: {len(params_list)}组")
    bt0 = time.time()
    results = []
    for i, p in enumerate(params_list):
        if i % 100 == 0:
            print(f"  [{i}/{len(params_list)}] {time.time()-bt0:.0f}s...")
        r = run_backtest(raw, dates, sym_idx, p)
        if r:
            results.append(r)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组有效结果")
    results.sort(key=lambda x: x['annual'], reverse=True)

    hdr = f"{'H':>3} {'R':>4} {'OTM':>4} {'Con':>3} {'Str':>4} {'MS':>4} {'GF':>3} {'Adpt':>4} {'TP':>5} " \
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}"
    print(f"\n{hdr}")
    print("-" * (len(hdr) + 50))

    for r in results[:60]:
        gf = f"{r.get('gamma_filter',0):.1f}" if r.get('gamma_filter',0) > 0 else '-'
        ad = 'Y' if r.get('adaptive_otm') else '-'
        tp = f"{r.get('take_profit_pct',0):.0f}x" if r.get('take_profit_pct',0) > 0 else '-'
        de = 'D' if r.get('delta_exit') else ' '
        print(f"{r['hold_days']:>3} {r['risk_pct']:>4.0%} {r.get('otm_pct',0)*100:>4.0f} "
              f"{r.get('min_consensus',0):>3} {r.get('min_strength',0):>4.1f} "
              f"{r['min_score']:>4.1f} {gf:>3} {ad:>4} {tp:>5} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5} {de}")

    print("\n" + "=" * 120)
    for ta, tw, label in [
        (6.0, 0.50, "年化>=600% & WR>=50%"),
        (3.0, 0.50, "年化>=300% & WR>=50%"),
        (1.0, 0.50, "年化>=100% & WR>=50%"),
        (6.0, 0.45, "年化>=600% & WR>=45%"),
        (3.0, 0.45, "年化>=300% & WR>=45%"),
        (0.5, 0.50, "年化>=50% & WR>=50%"),
    ]:
        print(f"\n=== {label} ===")
        good = [r for r in results if r['annual'] >= ta and r['wr'] >= tw]
        if good:
            for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  "
                      f"H={r['hold_days']}  R={r['risk_pct']:.0%}  OTM={r.get('otm_pct',0)*100:.0f}%  "
                      f"Con={r.get('min_consensus',0)}  Str={r.get('min_strength',0)}  "
                      f"Adpt={'Y' if r.get('adaptive_otm') else 'N'}  TP={r.get('take_profit_pct',0)}")
        else:
            print("  无")

    out_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    save = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in results[:300]]
    with open(os.path.join(out_dir, 'backtest_v47.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 300 → backtest_results/backtest_v47.json")


if __name__ == '__main__':
    main()
