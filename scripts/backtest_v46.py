#!/usr/bin/env python3
"""
策略 V46 — 期权增强 + 期限结构 + 波动率曲面 + 希腊字母

核心创新:
  1. 多维信号 (期限结构/曲面/Greeks) → 提高胜率
  2. 合成期权 (BS定价) → 非对称收益 (亏有限, 赢无限)
  3. Delta/Gamma仓位管理 → 风险控制
  4. IV期限结构择时 → 波动率择时

目标: 年化600%, 最大持仓3, 胜率>50%
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM_OPT = 0.0003  # 期权佣金
COMM_FUT = 0.00015
R_RATE = 0.02
INIT = 500000
TD = 252


def bs_price(S, K, T, r, sigma, flag='call'):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if flag == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, flag='call'):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if flag == 'call' else norm.cdf(d1) - 1


def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def load_and_precompute(data_dir):
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

        # HV多窗口
        for w in [5, 10, 20, 40, 60, 120]:
            df[f'hv_{w}'] = df['return'].rolling(w).std() * np.sqrt(TD)

        # IV期限结构
        df['iv_ts'] = (df['hv_20'] / df['hv_60']).replace(0, np.nan)
        df['iv_ts2'] = (df['hv_5'] / df['hv_20']).replace(0, np.nan)
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False)

        # 动量
        for lag in [3, 5, 10, 20, 60]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)
        df['ts_signal'] = df['mom_5'] - df['mom_60']

        # 成交量
        df['vol_ratio'] = df['vol'].rolling(5).mean() / df['vol'].rolling(20).mean().replace(0, np.nan)
        if 'oi' in df.columns:
            df['oi_chg'] = df['oi'].pct_change(5)
            df['oi_ratio'] = df['oi'].rolling(5).mean() / df['oi'].rolling(20).mean().replace(0, np.nan)

        # Skew
        up = df['return'].where(df['return'] > 0, 0).rolling(20).std() * np.sqrt(TD)
        dn = (-df['return'].where(df['return'] < 0, 0)).rolling(20).std() * np.sqrt(TD)
        df['vol_skew'] = (dn - up) / df['hv_20'].replace(0, np.nan)
        df['ret_skew'] = df['return'].rolling(60).skew()

        # EWMAC (合成Delta)
        for span in [4, 8, 16]:
            ef = df['close'].ewm(span=span).mean()
            es = df['close'].ewm(span=span * 4).mean()
            std = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ewmac_{span}'] = (ef - es) / std

        df['synth_delta'] = df['ewmac_4'] * 0.2 + df['ewmac_8'] * 0.3 + df['ewmac_16'] * 0.3

        # 合成Gamma
        df['hv_chg'] = df['hv_20'].diff(5) / df['hv_20'].shift(5).replace(0, np.nan)
        df['synth_gamma'] = df['hv_chg'].rolling(10).std()

        # 突破
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1.0, -1.0)
        for w in [10, 20, 40]:
            hh, ll = df['close'].rolling(w).max(), df['close'].rolling(w).min()
            df[f'bo_{w}'] = (df['close'] - 0.5 * (hh + ll)) / (hh - ll + 0.001) * 2

        # RAMOM
        s20 = df['return'].rolling(20).std().replace(0, np.nan)
        df['ramom'] = df['mom_10'] / (s20 * np.sqrt(10))

        # RSI
        d = df['close'].diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / df['close']

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_5', 'rsi', 'synth_delta'])
        if len(df) < 100:
            continue
        try:
            spec = get_spec(symbol)
        except:
            continue

        # ── 预计算各模式信号 ──
        sd = df['synth_delta'].values
        ts = df['ts_signal'].values
        iv_ts = df['iv_ts'].values
        trend = df['trend'].values
        skew = df['vol_skew'].values
        hv_pct = df['hv_pct'].values
        bo20 = df['bo_20'].values
        bo40 = df['bo_40'].values
        gamma = df['synth_gamma'].values
        ramom = df['ramom'].values
        mom5 = df['mom_5'].values
        oi_chg = df.get('oi_chg', pd.Series(np.nan, index=df.index)).values

        def _clip(x, cap):
            return np.sign(x) * np.minimum(np.abs(x), cap)

        # Mode: greek_ts
        sig_gt = np.zeros(len(df)); w_gt = np.zeros(len(df))
        m = ~np.isnan(sd); sig_gt[m] += _clip(sd[m], 3) * 2; w_gt[m] += 2
        m = ~np.isnan(ts); sig_gt[m] += _clip(ts[m]*20, 2); w_gt[m] += 1
        m = ~np.isnan(iv_ts) & (iv_ts > 0); sig_gt[m] += _clip(-(iv_ts[m]-1)*5, 1.5); w_gt[m] += 1
        m = trend != 0; sig_gt[m] += trend[m]*0.5; w_gt[m] += 0.5
        sig_gt = np.where(w_gt > 0, sig_gt/np.maximum(w_gt, 1), 0)

        # Mode: full
        sig_f = np.zeros(len(df)); w_f = np.zeros(len(df))
        m = ~np.isnan(sd); sig_f[m] += _clip(sd[m], 3)*3; w_f[m] += 3
        m = ~np.isnan(ts); sig_f[m] += _clip(ts[m]*20, 2); w_f[m] += 2
        m = ~np.isnan(iv_ts) & (iv_ts>0); sig_f[m] += _clip(-(iv_ts[m]-1)*5, 1.5); w_f[m] += 1
        m = ~np.isnan(skew); sig_f[m] += _clip(-skew[m], 2); w_f[m] += 1.5
        m = ~np.isnan(bo20); sig_f[m] += bo20[m]; w_f[m] += 1
        m = ~np.isnan(bo40); sig_f[m] += bo40[m]*0.5; w_f[m] += 0.5
        m = ~np.isnan(gamma) & (gamma>0.3); sig_f[m] *= 0.7; w_f[m] *= 0.7
        m = (trend!=0) & ~np.isnan(sig_f) & (np.sign(sig_f)==trend); sig_f[m] *= 1.4
        sig_f = np.where(w_f>0, sig_f/np.maximum(w_f,1), 0)

        # Mode: enhanced (更强的信号加权)
        sig_e = np.zeros(len(df)); w_e = np.zeros(len(df))
        m = ~np.isnan(sd); sig_e[m] += _clip(sd[m], 3)*4; w_e[m] += 4
        m = ~np.isnan(ts); sig_e[m] += _clip(ts[m]*25, 2.5); w_e[m] += 2
        m = ~np.isnan(iv_ts)&(iv_ts>0); sig_e[m] += _clip(-(iv_ts[m]-1)*8, 2); w_e[m] += 1.5
        m = ~np.isnan(skew); sig_e[m] += _clip(-skew[m], 2.5); w_e[m] += 2
        m = ~np.isnan(bo20); sig_e[m] += bo20[m]*1.5; w_e[m] += 1.5
        m = ~np.isnan(ramom); sig_e[m] += _clip(ramom[m], 3)*2; w_e[m] += 2
        m = ~np.isnan(gamma)&(gamma>0.25); sig_e[m] *= 0.6; w_e[m] *= 0.6
        m = (trend!=0)&~np.isnan(sig_e)&(np.sign(sig_e)==trend); sig_e[m] *= 1.5
        sig_e = np.where(w_e>0, sig_e/np.maximum(w_e,1), 0)

        # Mode: oi_momentum (持仓量+动量+期限结构)
        sig_oi = np.zeros(len(df)); w_oi = np.zeros(len(df))
        m = ~np.isnan(sd); sig_oi[m] += _clip(sd[m], 3)*3; w_oi[m] += 3
        m = ~np.isnan(ts); sig_oi[m] += _clip(ts[m]*20, 2); w_oi[m] += 2
        m = ~np.isnan(skew); sig_oi[m] += _clip(-skew[m], 2); w_oi[m] += 1.5
        # OI确认: OI增加+方向一致→加仓信号
        m = ~np.isnan(oi_chg)
        oi_sig = np.where(oi_chg[m] > 0.02, 1.0, np.where(oi_chg[m] < -0.02, -0.5, 0))
        sig_oi[m] += oi_sig; w_oi[m] += 1
        m = ~np.isnan(bo20); sig_oi[m] += bo20[m]; w_oi[m] += 1
        m = (trend!=0)&~np.isnan(sig_oi)&(np.sign(sig_oi)==trend); sig_oi[m] *= 1.4
        sig_oi = np.where(w_oi>0, sig_oi/np.maximum(w_oi,1), 0)

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
            'signals': {
                'greek_ts': sig_gt.astype(np.float64),
                'full': sig_f.astype(np.float64),
                'enhanced': sig_e.astype(np.float64),
                'oi_momentum': sig_oi.astype(np.float64),
            },
        }
    return raw


def build_date_index(raw, start_date, end_date):
    all_dates = set()
    for sym, d in raw.items():
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
    """期权增强回测 — 用BS定价创建合成期权"""
    max_pos = params.get('max_pos', 3)
    mu = params.get('margin_usage', 0.90)
    hold_days = params.get('hold_days', 5)
    min_score = params.get('min_score', 0.5)
    mode = params.get('mode', 'full')
    risk_pct = params.get('risk_pct', 0.02)  # 每笔风险(占权益)
    otm_pct = params.get('otm_pct', 0.0)     # OTM程度
    liq_discount = params.get('liq_discount', 0.9)  # 流动性折扣
    gamma_filter = params.get('gamma_filter', 0.0)
    hv_lo = params.get('hv_lo', 0.10)
    hv_hi = params.get('hv_hi', 0.60)

    equity = float(INIT)
    positions = {}  # sym -> {d, entry_date, entry_price, K, T, sigma, premium, direction}
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
            price = raw[sym]['close'][iloc]
            hd = int((date - pos['entry_date']) / np.timedelta64(1, 'D'))

            should_close = hd >= hold_days

            if should_close:
                # 用BS定价计算期权退出价值
                S_now = price
                K = pos['K']
                T_rem = max(pos['T'] - hd / TD, 0.001)  # 剩余时间
                sigma = pos['sigma']
                flag = pos['flag']

                # 期权理论价值
                opt_val = bs_price(S_now, K, T_rem, R_RATE, sigma, flag)

                # 内在价值底线
                if flag == 'call':
                    intrinsic = max(S_now - K, 0)
                else:
                    intrinsic = max(K - S_now, 0)

                exit_val = max(opt_val, intrinsic) * liq_discount

                # PnL = (退出价值 - 买入成本) * 方向 * 合约乘数
                mult = pos['mult']
                pnl = (exit_val - pos['premium']) * pos['direction'] * mult * pos['contracts']
                comm_exit = exit_val * mult * pos['contracts'] * COMM_OPT
                net_pnl = pnl - comm_exit

                equity += net_pnl
                closed_pnls.append(net_pnl)
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

                pi = iloc - 1  # 前一天信号

                # 过滤
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

                score = d['signals'][mode][pi]
                if np.isnan(score) or abs(score) < min_score:
                    continue

                sigs.append((sym, score, abs(score), iloc, hv))

            sigs.sort(key=lambda x: x[2], reverse=True)

            for sym, score, abs_score, iloc, hv in sigs:
                if len(positions) >= max_pos:
                    break

                # 入场价格 = 当日开盘价
                S = raw[sym]['open'][iloc]
                if np.isnan(S) or S <= 0:
                    S = raw[sym]['close'][iloc]
                if S <= 0:
                    continue

                # 方向: call(看多) / put(看空)
                direction = 1 if score > 0 else -1
                flag = 'call' if direction > 0 else 'put'

                # 行权价: OTM程度
                K = S * (1 + otm_pct * direction)

                # 期权参数
                T = hold_days / TD
                sigma = hv  # 用历史波动率作为IV代理

                # BS定价计算期权费
                premium = bs_price(S, K, T, R_RATE, sigma, flag)
                if premium <= 0:
                    continue

                # 合约乘数
                mult, mr, _, _ = raw[sym]['spec']

                # 仓位: 风险risk_pct的权益 / (期权费 × 乘数)
                risk_amount = equity * risk_pct
                contracts = max(int(risk_amount / (premium * mult)), 1)

                # 最大合约数限制 (不超过权益的5倍名义价值)
                max_contracts = max(int(equity * 5 / (S * mult)), 1)
                contracts = min(contracts, max_contracts)

                cost = premium * mult * contracts
                entry_comm = cost * COMM_OPT
                total_cost = cost + entry_comm

                if total_cost > equity * 0.10:  # 单笔不超过权益10%
                    contracts = max(int(equity * 0.10 / (premium * mult + 1)), 1)
                    cost = premium * mult * contracts
                    entry_comm = cost * COMM_OPT
                    total_cost = cost + entry_comm

                if total_cost > equity - sum(p.get('cost', 0) for p in positions.values()):
                    continue

                positions[sym] = {
                    'direction': direction,
                    'entry_date': date,
                    'entry_price': S,
                    'K': K,
                    'T': T,
                    'sigma': sigma,
                    'premium': premium,
                    'flag': flag,
                    'mult': mult,
                    'contracts': contracts,
                    'cost': total_cost,
                }

        # ── 权益 ──
        # 未平仓权益用BS定价
        unrealized = 0.0
        for sym, pos in positions.items():
            idx_map = sym_idx.get(sym)
            if idx_map and date in idx_map:
                S_now = raw[sym]['close'][idx_map[date]]
                hd = int((date - pos['entry_date']) / np.timedelta64(1, 'D'))
                T_rem = max(pos['T'] - hd / TD, 0.001)
                opt_val = bs_price(S_now, pos['K'], T_rem, R_RATE, pos['sigma'], pos['flag'])
                if pos['flag'] == 'call':
                    intrinsic = max(S_now - pos['K'], 0)
                else:
                    intrinsic = max(pos['K'] - S_now, 0)
                unrealized += (max(opt_val, intrinsic) * liq_discount - pos['premium']) * pos['direction'] * pos['mult'] * pos['contracts']

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
    avg_opt_ret = float(avg_w / (INIT * risk_pct)) if avg_w > 0 else 0

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
        'avg_opt_ret': avg_opt_ret,
        **{k: v for k, v in params.items()},
    }


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载+预计算...")
    t0 = time.time()
    raw = load_and_precompute(data_dir)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2026-05-08')

    print("构建日期索引...")
    dates, sym_idx = build_date_index(raw, start_date, end_date)
    print(f"  {len(dates)}交易日")

    # ── 参数扫描 ──
    param_list = []
    modes = ['greek_ts', 'full', 'enhanced', 'oi_momentum']
    risk_pcts = [0.02, 0.03, 0.04]
    otm_pcts = [0.0, 0.01, 0.02, 0.03]
    hold_dayss = [5, 7, 10]
    min_scores = [0.3, 0.5, 0.8]

    for mode in modes:
        for rp in risk_pcts:
            for otm in otm_pcts:
                for hd in hold_dayss:
                    for ms in min_scores:
                        param_list.append(dict(
                            mode=mode, risk_pct=rp, otm_pct=otm,
                            hold_days=hd, min_score=ms, max_pos=3))

    # Gamma过滤变体
    for mode in ['full', 'enhanced']:
        for rp in [0.03, 0.04]:
            for otm in [0.01, 0.02, 0.03]:
                for gf in [0.2, 0.3]:
                    param_list.append(dict(
                        mode=mode, risk_pct=rp, otm_pct=otm,
                        hold_days=5, min_score=0.5, max_pos=3,
                        gamma_filter=gf))

    # 短持有期
    for mode in ['greek_ts', 'enhanced']:
        for rp in [0.03, 0.04]:
            for otm in [0.02, 0.03]:
                for hd in [3, 5]:
                    param_list.append(dict(
                        mode=mode, risk_pct=rp, otm_pct=otm,
                        hold_days=hd, min_score=0.5, max_pos=3,
                        hv_hi=0.70))

    print(f"\n参数组合: {len(param_list)}组")
    bt0 = time.time()
    results = []
    for i, params in enumerate(param_list):
        if i % 100 == 0:
            print(f"  [{i}/{len(param_list)}] {time.time()-bt0:.0f}s...")
        r = run_backtest(raw, dates, sym_idx, params)
        if r:
            results.append(r)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组有效结果")

    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'模式':>14} {'Risk':>5} {'OTM':>4} {'Hold':>4} {'Score':>5} {'GF':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5} {'OptRet':>7}")
    print("-" * 120)
    for r in results[:60]:
        gf_s = f"{r.get('gamma_filter',0):.1f}" if r.get('gamma_filter',0) > 0 else '-'
        print(f"{r['mode'][:14]:>14} {r['risk_pct']:>5.0%} {r.get('otm_pct',0)*100:>4.0f} "
              f"{r['hold_days']:>4} {r['min_score']:>5.1f} {gf_s:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5} "
              f"{r.get('avg_opt_ret',0):>7.1f}x")

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
                      f"Sharpe={r['sharpe']:>5.2f}  OptRet={r.get('avg_opt_ret',0):.1f}x  "
                      f"{r['mode']}  R={r['risk_pct']:.0%}  OTM={r.get('otm_pct',0)*100:.0f}%  "
                      f"H={r['hold_days']}  MS={r['min_score']}  GF={r.get('gamma_filter',0)}")
        else:
            print("  无")

    out_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    save = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in results[:300]]
    with open(os.path.join(out_dir, 'backtest_v46.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 300 → backtest_results/backtest_v46.json")


if __name__ == '__main__':
    main()
