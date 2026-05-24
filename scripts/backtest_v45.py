#!/usr/bin/env python3
"""
策略 V45 — 期权期限结构 + 期货期限结构 + 波动率曲面 + 希腊字母

信号维度:
  1. 期权期限结构: IV短/长斜率 → 波动率预期
  2. 期货期限结构: 短/长动量差 → contango/backwardation
  3. 波动率曲面: skew → 恐慌/贪婪
  4. 希腊字母: delta加权, gamma风险预算, theta择时

优化: 所有信号预计算为numpy数组, 回测只做数组索引
目标: 年化600%, 最大持仓3, 胜率>50%
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
R_RATE = 0.02
INIT_CAPITAL = 500000
TD = 252


# ═══════════════════════════════════════════════════════════
# 数据加载 + 预计算所有信号
# ═══════════════════════════════════════════════════════════
def load_and_precompute(data_dir):
    """加载日线数据, 预计算所有信号为numpy数组"""
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

        # ── HV多窗口 (期权期限结构代理) ──
        for w in [5, 10, 20, 40, 60, 120]:
            df[f'hv_{w}'] = df['return'].rolling(w).std() * np.sqrt(TD)

        df['iv_ts'] = (df['hv_20'] / df['hv_60']).replace(0, np.nan)
        df['iv_ts2'] = (df['hv_5'] / df['hv_20']).replace(0, np.nan)
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False)

        # ── 动量 (期货期限结构代理) ──
        for lag in [5, 10, 20, 60]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)
        df['ts_signal'] = df['mom_5'] - df['mom_60']

        # ── 成交量/持仓量 ──
        df['vol_ratio'] = df['vol'].rolling(5).mean() / df['vol'].rolling(20).mean().replace(0, np.nan)
        if 'oi' in df.columns:
            df['oi_chg'] = df['oi'].pct_change(5)

        # ── 波动率曲面/skew ──
        up = df['return'].where(df['return'] > 0, 0).rolling(20).std() * np.sqrt(TD)
        dn = (-df['return'].where(df['return'] < 0, 0)).rolling(20).std() * np.sqrt(TD)
        df['vol_skew'] = (dn - up) / df['hv_20'].replace(0, np.nan)
        df['ret_skew'] = df['return'].rolling(60).skew()

        # ── EWMAC (合成Delta) ──
        for span in [4, 8, 16]:
            ef = df['close'].ewm(span=span).mean()
            es = df['close'].ewm(span=span * 4).mean()
            std = df['close'].rolling(20).std().replace(0, np.nan)
            df[f'ewmac_{span}'] = (ef - es) / std

        df['synth_delta'] = df['ewmac_4'] * 0.2 + df['ewmac_8'] * 0.3 + df['ewmac_16'] * 0.3

        # ── 合成Gamma (波动率的波动率) ──
        df['hv_chg'] = df['hv_20'].diff(5) / df['hv_20'].shift(5).replace(0, np.nan)
        df['synth_gamma'] = df['hv_chg'].rolling(10).std()

        # ── Theta (趋势衰减) ──
        df['theta'] = (df['mom_5'] - df['mom_5'].shift(5)) / 5

        # ── Vega (波动率×方向) ──
        df['vega'] = df['hv_20'].pct_change(5) * df['mom_5']

        # ── 基础趋势/突破 ──
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

        # ── 预计算各模式信号 (向量化) ──
        sd = df['synth_delta'].values
        ts = df['ts_signal'].values
        iv_ts = df['iv_ts'].values
        trend = df['trend'].values
        skew = df['vol_skew'].values
        hv_pct = df['hv_pct'].values
        bo20 = df['bo_20'].values
        bo40 = df['bo_40'].values
        gamma = df['synth_gamma'].values
        vega = df['vega'].values
        theta = df['theta'].values
        mom5 = df['mom_5'].values
        ramom = df['ramom'].values

        # Mode: greek_ts
        def _clip(x, cap):
            return np.sign(x) * np.minimum(np.abs(x), cap)

        sig_greek_ts = np.zeros(len(df))
        w_gt = np.zeros(len(df))
        # delta
        m = ~np.isnan(sd); sig_greek_ts[m] += _clip(sd[m], 3) * 2; w_gt[m] += 2
        # ts
        m = ~np.isnan(ts); sig_greek_ts[m] += _clip(ts[m] * 20, 2); w_gt[m] += 1
        # iv_ts
        m = ~np.isnan(iv_ts) & (iv_ts > 0); iv_sig = -(iv_ts[m] - 1) * 5; sig_greek_ts[m] += _clip(iv_sig, 1.5); w_gt[m] += 1
        # trend
        m = trend != 0; sig_greek_ts[m] += trend[m] * 0.5; w_gt[m] += 0.5
        sig_greek_ts = np.where(w_gt > 0, sig_greek_ts / np.maximum(w_gt, 1), 0)

        # Mode: vol_surface
        sig_vs = np.zeros(len(df))
        w_vs = np.zeros(len(df))
        m = ~np.isnan(skew); sig_vs[m] += _clip(-skew[m], 2); w_vs[m] += 1.5
        m = ~np.isnan(hv_pct)
        sig_vs[m] += np.where(hv_pct[m] > 0.85, -1.0, np.where(hv_pct[m] < 0.15, 1.0, 0.0))
        w_vs[m] += 1.0
        m = ~np.isnan(sd); sig_vs[m] += _clip(sd[m], 2); w_vs[m] += 1.0
        m = ~np.isnan(mom5); sig_vs[m] += _clip(mom5[m] * 15, 1.5); w_vs[m] += 1.0
        sig_vs = np.where(w_vs > 0, sig_vs / np.maximum(w_vs, 1), 0)

        # Mode: full
        sig_full = np.zeros(len(df))
        w_f = np.zeros(len(df))
        m = ~np.isnan(sd); sig_full[m] += _clip(sd[m], 3) * 3; w_f[m] += 3
        m = ~np.isnan(ts); sig_full[m] += _clip(ts[m] * 20, 2); w_f[m] += 2
        m = ~np.isnan(iv_ts) & (iv_ts > 0); iv_sig = -(iv_ts[m] - 1) * 5; sig_full[m] += _clip(iv_sig, 1.5); w_f[m] += 1
        m = ~np.isnan(skew); sig_full[m] += _clip(-skew[m], 2); w_f[m] += 1.5
        m = ~np.isnan(hv_pct)
        sig_full[m] += np.where(hv_pct[m] > 0.85, -0.8, np.where(hv_pct[m] < 0.15, 0.8, 0.0))
        w_f[m] += 0.5
        m = ~np.isnan(bo20); sig_full[m] += bo20[m]; w_f[m] += 1
        m = ~np.isnan(bo40); sig_full[m] += bo40[m] * 0.5; w_f[m] += 0.5
        # gamma折扣
        m = ~np.isnan(gamma) & (gamma > 0.3); sig_full[m] *= 0.7; w_f[m] *= 0.7
        # trend加成
        m = (trend != 0) & ~np.isnan(sig_full) & (np.sign(sig_full) == trend)
        sig_full[m] *= 1.4
        sig_full = np.where(w_f > 0, sig_full / np.maximum(w_f, 1), 0)

        # Mode: greeks_only
        sig_go = np.zeros(len(df))
        w_go = np.zeros(len(df))
        m = ~np.isnan(sd); sig_go[m] += _clip(sd[m], 3) * 3; w_go[m] += 3
        m = ~np.isnan(vega); sig_go[m] += _clip(vega[m] * 50, 1.5); w_go[m] += 1.5
        m = ~np.isnan(theta); sig_go[m] += _clip(theta[m] * 50, 1); w_go[m] += 1
        m = trend != 0; sig_go[m] += trend[m] * 0.5; w_go[m] += 0.5
        sig_go = np.where(w_go > 0, sig_go / np.maximum(w_go, 1), 0)

        # Mode: enhanced (stronger signals)
        sig_enh = np.zeros(len(df))
        w_e = np.zeros(len(df))
        # Heavy delta
        m = ~np.isnan(sd); sig_enh[m] += _clip(sd[m], 3) * 4; w_e[m] += 4
        # Term structure (futures + options)
        m = ~np.isnan(ts); sig_enh[m] += _clip(ts[m] * 25, 2.5); w_e[m] += 2
        m = ~np.isnan(iv_ts) & (iv_ts > 0); iv_sig = -(iv_ts[m] - 1) * 8; sig_enh[m] += _clip(iv_sig, 2); w_e[m] += 1.5
        # Skew
        m = ~np.isnan(skew); sig_enh[m] += _clip(-skew[m], 2.5); w_e[m] += 2
        # Breakout
        m = ~np.isnan(bo20); sig_enh[m] += bo20[m] * 1.5; w_e[m] += 1.5
        # RAMOM
        m = ~np.isnan(ramom); sig_enh[m] += _clip(ramom[m], 3) * 2; w_e[m] += 2
        # Trend
        m = trend != 0; sig_enh[m] += trend[m] * 0.3; w_e[m] += 0.3
        # Gamma折扣
        m = ~np.isnan(gamma) & (gamma > 0.25); sig_enh[m] *= 0.6; w_e[m] *= 0.6
        # Trend加成
        m = (trend != 0) & ~np.isnan(sig_enh) & (np.sign(sig_enh) == trend)
        sig_enh[m] *= 1.5
        sig_enh = np.where(w_e > 0, sig_enh / np.maximum(w_e, 1), 0)

        # 存储为numpy数组 (保留trade_date列)
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
            'iv_ts': df['iv_ts'].values.astype(np.float64),
            'vol_skew': df['vol_skew'].values.astype(np.float64),
            'signals': {
                'greek_ts': sig_greek_ts.astype(np.float64),
                'vol_surface': sig_vs.astype(np.float64),
                'full': sig_full.astype(np.float64),
                'greeks_only': sig_go.astype(np.float64),
                'enhanced': sig_enh.astype(np.float64),
            },
        }
    return raw


# ═══════════════════════════════════════════════════════════
# 构建全局日期索引
# ═══════════════════════════════════════════════════════════
def build_date_index(raw, start_date, end_date):
    """构建全局日期→{symbol→iloc}映射"""
    all_dates = set()
    for sym, d in raw.items():
        mask = (d['dates'] >= start_date) & (d['dates'] <= end_date)
        for dt in d['dates'][mask]:
            all_dates.add(dt)
    dates = np.array(sorted(all_dates))

    # symbol → date → iloc (用np.searchsorted加速)
    sym_idx = {}
    for sym, d in raw.items():
        # 只保留范围内日期的iloc
        idx_map = {}
        start_mask = d['dates'] >= start_date
        end_mask = d['dates'] <= end_date
        mask = start_mask & end_mask
        valid_dates = d['dates'][mask]
        valid_ilocs = np.where(mask)[0]
        for dt, iloc in zip(valid_dates, valid_ilocs):
            idx_map[dt] = int(iloc)
        sym_idx[sym] = idx_map

    return dates, sym_idx


# ═══════════════════════════════════════════════════════════
# 回测引擎 (优化版, 只做数组索引)
# ═══════════════════════════════════════════════════════════
def run_backtest(raw, dates, sym_idx, params):
    max_pos = params.get('max_pos', 3)
    mu = params.get('margin_usage', 0.90)
    hold_days = params.get('hold_days', 5)
    min_score = params.get('min_score', 0.5)
    mode = params.get('mode', 'full')
    trade_mode = params.get('trade_mode', 'multi_day')
    gamma_filter = params.get('gamma_filter', 0.0)
    tp_pct = params.get('tp_pct', 0.0)
    sl_pct = params.get('sl_pct', 0.0)

    ecm_on = params.get('ecm', False)
    ecm_f = params.get('ecm_fast', 10)
    ecm_s = params.get('ecm_slow', 30)
    ecm_up = params.get('ecm_up_mult', 1.5)
    ecm_dn = params.get('ecm_dn_mult', 0.5)

    equity = float(INIT_CAPITAL)
    cash = float(INIT_CAPITAL)
    positions = {}  # symbol -> {d, ed, ep, fe, fl, m, fm}
    closed_pnls = []
    eq_hist = [float(INIT_CAPITAL)]

    for date in dates:
        # ECM
        ecm_m = 1.0
        if ecm_on and len(eq_hist) >= ecm_s:
            eq_arr = np.array(eq_hist[-ecm_s:])
            mf = eq_arr[-ecm_f:].mean()
            ms = eq_arr.mean()
            ecm_m = ecm_up if mf > ms else ecm_dn

        eff_mu = mu * ecm_m

        # ── 退出 ──
        for sym in list(positions):
            pos = positions[sym]
            idx_map = sym_idx.get(sym)
            if not idx_map or date not in idx_map:
                continue
            iloc = idx_map[date]
            price = raw[sym]['close'][iloc]
            hd = int((date - pos['ed']) / np.timedelta64(1, 'D'))
            pnl_pct = (price / pos['ep'] - 1) * pos['d']

            close = False
            if trade_mode == 'daily':
                close = True
            else:
                if hd >= hold_days:
                    close = True
                if tp_pct > 0 and pnl_pct >= tp_pct:
                    close = True
                if sl_pct > 0 and pnl_pct <= -sl_pct:
                    close = True

            if close:
                comm = price * pos['m'] * pos['fl'] * COMM
                pnl = (price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']
                cash += pos['fm'] + pnl - comm
                closed_pnls.append(pnl - comm)
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

                # 前一天数据
                pi = iloc - 1

                # 过滤
                hv = d['hv_20'][pi]
                if np.isnan(hv) or hv < 0.05 or hv > 0.70:
                    continue
                hp = d['hv_pct'][pi]
                if np.isnan(hp) or hp > 0.85:
                    continue
                rsi = d['rsi'][pi]
                if np.isnan(rsi) or rsi > 75 or rsi < 25:
                    continue
                ap = d['atr_pct'][pi]
                if np.isnan(ap) or ap > 0.05:
                    continue
                if gamma_filter > 0:
                    sg = d['synth_gamma'][pi]
                    if not np.isnan(sg) and sg > gamma_filter:
                        continue

                score = d['signals'][mode][pi]
                if np.isnan(score) or abs(score) < min_score:
                    continue

                direction = 1.0 if score > 0 else -1.0
                sigs.append((sym, direction, abs(score), iloc))

            sigs.sort(key=lambda x: x[2], reverse=True)

            for sym, direction, score, iloc in sigs:
                if len(positions) >= max_pos:
                    break

                S = raw[sym]['open'][iloc]
                if np.isnan(S) or S <= 0:
                    S = raw[sym]['close'][iloc]
                if S <= 0:
                    continue

                mult, mr, _, _ = raw[sym]['spec']
                mpl = S * mult * mr
                target = equity * eff_mu / max_pos
                fl = max(int(target / mpl), 1)
                fm = mpl * fl
                fc = S * mult * fl * COMM

                total_m = sum(p['fm'] for p in positions.values())
                if total_m + fm > equity * eff_mu:
                    fl = max(int((equity * eff_mu - total_m) / mpl), 0)
                    if fl <= 0:
                        continue
                    fm = mpl * fl; fc = S * mult * fl * COMM

                if fm + fc > cash:
                    fl = max(int((cash - fc) / mpl), 0)
                    if fl <= 0:
                        continue
                    fm = mpl * fl; fc = S * mult * fl * COMM

                cash -= fm + fc
                positions[sym] = {
                    'd': direction, 'ed': date, 'ep': S,
                    'fe': S * (1 + 0.0001 * direction),
                    'fl': fl, 'm': mult, 'fm': fm,
                }

        # ── 权益 ──
        unrealized = 0.0
        for sym, pos in positions.items():
            idx_map = sym_idx.get(sym)
            if idx_map and date in idx_map:
                price = raw[sym]['close'][idx_map[date]]
                unrealized += (price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']

        equity = cash + unrealized
        eq_hist.append(equity)
        if equity < 5000:
            break

    if not closed_pnls or equity <= 0:
        return None
    total_ret = (equity - INIT_CAPITAL) / INIT_CAPITAL
    if total_ret <= -1:
        return None

    days = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
    years = max(days / 365, 0.001)
    ann = float((1 + total_ret) ** (1 / years) - 1)

    pnls = np.array(closed_pnls)
    wr = float((pnls > 0).mean())
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


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════
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

    # 生成参数组合 (精简)
    param_list = []
    modes = ['greek_ts', 'vol_surface', 'full', 'greeks_only', 'enhanced']
    mus = [0.60, 0.70, 0.80, 0.90]
    hds = [3, 5, 7]
    mss = [0.3, 0.5, 0.8]

    # A: 多日持有, 各模式
    for mode in modes:
        for mu in mus:
            for hd in hds:
                for ms in mss:
                    param_list.append(dict(
                        margin_usage=mu, hold_days=hd, min_score=ms,
                        max_pos=3, mode=mode, trade_mode='multi_day'))

    # B: 日频
    for mode in modes:
        for mu in [0.70, 0.80, 0.90]:
            for ms in [0.3, 0.5, 0.8]:
                param_list.append(dict(
                    margin_usage=mu, hold_days=1, min_score=ms,
                    max_pos=3, mode=mode, trade_mode='daily'))

    # C: ECM
    for mode in ['full', 'enhanced']:
        for mu in [0.80, 0.90]:
            for ecm_up in [1.5, 2.0, 3.0]:
                param_list.append(dict(
                    margin_usage=mu, hold_days=5, min_score=0.5,
                    max_pos=3, mode=mode, trade_mode='multi_day',
                    ecm=True, ecm_fast=5, ecm_slow=20,
                    ecm_up_mult=ecm_up, ecm_dn_mult=0.5))

    # D: TP/SL + Gamma过滤
    for mode in ['full', 'enhanced']:
        for tp in [0.01, 0.02]:
            for gf in [0.2, 0.3]:
                param_list.append(dict(
                    margin_usage=0.90, hold_days=5, min_score=0.5,
                    max_pos=3, mode=mode, trade_mode='multi_day',
                    tp_pct=tp, sl_pct=tp * 3, gamma_filter=gf))

    print(f"\n参数组合: {len(param_list)}组")
    bt0 = time.time()
    results = []
    for i, params in enumerate(param_list):
        if i % 50 == 0:
            print(f"  [{i}/{len(param_list)}] {time.time()-bt0:.0f}s...")
        r = run_backtest(raw, dates, sym_idx, params)
        if r:
            results.append(r)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组有效结果")

    results.sort(key=lambda x: x['annual'], reverse=True)

    # 输出
    print(f"\n{'模式':>14} {'M':>4} {'H':>3} {'TP':>5} {'ECM':>3} {'GF':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 105)
    for r in results[:60]:
        tp_s = f"{r.get('tp_pct',0)*100:.1f}" if r.get('tp_pct', 0) > 0 else '-'
        ecm_s = 'Y' if r.get('ecm') else '-'
        gf_s = f"{r.get('gamma_filter',0):.1f}" if r.get('gamma_filter', 0) > 0 else '-'
        print(f"{r['mode'][:14]:>14} {r['margin_usage']:>4.0%} {r['hold_days']:>3} "
              f"{tp_s:>5} {ecm_s:>3} {gf_s:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    print("\n" + "=" * 105)
    for ta, tw, label in [
        (6.0, 0.50, "年化>=600% & WR>=50%"),
        (3.0, 0.50, "年化>=300% & WR>=50%"),
        (1.0, 0.50, "年化>=100% & WR>=50%"),
        (0.5, 0.50, "年化>=50% & WR>=50%"),
        (6.0, 0.45, "年化>=600% & WR>=45%"),
    ]:
        print(f"\n=== {label} ===")
        good = [r for r in results if r['annual'] >= ta and r['wr'] >= tw]
        if good:
            for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  {r['mode']}  "
                      f"M={r['margin_usage']:.0%}  H={r['hold_days']}  "
                      f"ECM={'Y' if r.get('ecm') else 'N'}  "
                      f"TP={r.get('tp_pct',0)*100:.1f}%  GF={r.get('gamma_filter',0)}")
        else:
            print("  无")

    # 保存
    out_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    save = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in results[:300]]
    with open(os.path.join(out_dir, 'backtest_v45.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 300 → backtest_results/backtest_v45.json")


if __name__ == '__main__':
    main()
