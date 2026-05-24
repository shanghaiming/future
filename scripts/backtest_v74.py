#!/usr/bin/env python3
"""
V74: 进一步优化
目标确认: 年化1000%+ WR>50% MDD<30%
方向:
1. 持仓1天 (intraday gap fade最强)
2. 更多仓位 (5个持仓)
3. 波动率目标杠杆
4. 分级仓位: 高分多配, 低分少配
5. 部分复利: 有上限的复利
"""
import os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
CONTRACT_SPECS = 'scripts/contract_specs.py'


def load_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs", CONTRACT_SPECS)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    all_data = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        try: mult, margin, tick, tick_val = cs.get_spec(sym)
        except: continue
        df = pd.read_csv(f)
        if len(df) < 100: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if df['close'].isna().all() or (df['close'] == 0).any(): continue
        all_data[sym] = df
    print(f"  {len(all_data)}品种")
    return all_data


def compute_signals(all_data):
    signal_data = {}
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['vol'].values.astype(float)
        oi = df['oi'].values.astype(float)
        n = len(df)

        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        gap = np.full(n, np.nan); gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100

        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = pd.Series(tr).rolling(20).mean().values
        atr_pct = atr / c * 100

        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        ma5 = pd.Series(c).rolling(5).mean().values

        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100

        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v

        vol_ma5 = pd.Series(v).rolling(5).mean().values
        range_ = h - l
        clv = np.where(range_ > 0, (2*c - h - l) / range_, 0)

        gv = np.nan_to_num(gap)
        ga = np.nan_to_num(gap / np.where(atr_pct == 0, np.nan, atr_pct))

        # HV percentile (for vol targeting)
        ret = np.full(n, np.nan); ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 100
        hv20 = pd.Series(ret).rolling(20).std().values
        hv_pct = pd.Series(hv20).rolling(252).rank(pct=True).values * 100

        # 做多
        s_l = np.zeros(n)
        s_l += (gv < -0.5).astype(int) * 1
        s_l += (gv < -1.0).astype(int) * 2
        s_l += (gv < -1.5).astype(int) * 2
        s_l += (gv < -2.0).astype(int) * 3
        s_l += (ga < -1.0).astype(int) * 2
        s_l += (ga < -1.5).astype(int) * 3
        s_l += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_l += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        s_l += (mom5 < -3).astype(int) * 1
        s_l += (mom5 < -5).astype(int) * 1
        s_l += (c < ma5).astype(int) * 1
        s_l += ((v > vol_ma5 * 1.5) & (c < prev_c)).astype(int) * 1
        s_l += (clv > 0.5).astype(int) * 1
        s_l += (ma20 > ma60).astype(int) * 2

        # 做空
        s_s = np.zeros(n)
        s_s += (gv > 0.5).astype(int) * 1
        s_s += (gv > 1.0).astype(int) * 2
        s_s += (gv > 1.5).astype(int) * 2
        s_s += (gv > 2.0).astype(int) * 3
        s_s += (ga > 1.0).astype(int) * 2
        s_s += (ga > 1.5).astype(int) * 3
        s_s += ((oi_ch > 0) & (c > prev_c)).astype(int) * 3
        s_s += ((oi_ch < 0) & (c > prev_c)).astype(int) * 2
        s_s += (mom5 > 3).astype(int) * 1
        s_s += (mom5 > 5).astype(int) * 1
        s_s += (c > ma5).astype(int) * 1
        s_s += ((v > vol_ma5 * 1.5) & (c > prev_c)).astype(int) * 1
        s_s += (clv < -0.5).astype(int) * 1
        s_s += (ma20 < ma60).astype(int) * 2

        # Forward returns
        for hd in [1, 2, 3]:
            fwd = np.full(n, np.nan)
            if n > hd: fwd[:n-hd] = (c[hd:] - o[:n-hd]) / o[:n-hd] * 100
            df[f'fwd_{hd}d'] = fwd

        df['score_long'] = s_l
        df['score_short'] = s_s
        df['gap_pct'] = gap
        df['hv_pct'] = hv_pct
        signal_data[sym] = df
    return signal_data


def run_bt(signal_data, start, end, max_pos=3, lev=3, min_sc=7, hold=2,
           sl=None, tp=None, vol_target=None, score_sizing=False):
    """回测引擎"""
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq = []
    trades = []
    pos = []
    recent_vol = []

    for dt in dates:
        # 平仓
        pnl = 0
        keep = []
        for p in pos:
            df = signal_data.get(p['sym'])
            cur = None
            if df is not None:
                idx = df.index[df['trade_date'] == dt]
                if len(idx) > 0: cur = df.loc[idx[0], 'close']
            if cur is None or np.isnan(cur):
                keep.append(p); continue
            r = (cur - p['ep']) / p['ep'] * 100 if p['dir'] == 'long' else (p['ep'] - cur) / p['ep'] * 100
            d = (dt - p['ed']).days
            reason = None
            if sl and r <= sl: reason = 'SL'
            elif tp and r >= tp: reason = 'TP'
            elif d >= hold: reason = 'exp'
            if reason:
                pnl += p['not'] * r / 100
                trades.append({'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'],
                               'xd': dt, 'ep': p['ep'], 'xp': cur, 'r': r,
                               'pnl': p['not'] * r / 100, 'sc': p['sc'],
                               'hold': d, 'reason': reason})
                recent_vol.append(r)
            else:
                keep.append(p)
        pos = keep
        cap += pnl
        if cap <= 0:
            eq.append({'date': dt, 'capital': 0}); break

        # 波动率目标杠杆
        cur_lev = lev
        if vol_target and len(recent_vol) >= 30:
            rv = np.std(recent_vol[-30:])
            if rv > 0:
                target_vol = vol_target  # e.g. 2% per trade
                cur_lev = min(max(lev * target_vol / rv, lev * 0.5), lev * 2)

        # 开仓
        n_open = max_pos - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        cands = []
        for sym, df in signal_data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if row['score_long'] >= min_sc:
                cands.append({'sym': sym, 'dir': 'long', 'sc': row['score_long'], 'ep': row['open']})
            if row['score_short'] >= min_sc:
                cands.append({'sym': sym, 'dir': 'short', 'sc': row['score_short'], 'ep': row['open']})

        # 同品种取高分
        best = {}
        for c in cands:
            if c['sym'] not in best or c['sc'] > best[c['sym']]['sc']:
                best[c['sym']] = c

        ranked = sorted(best.values(), key=lambda x: -x['sc'])
        for c in ranked[:n_open]:
            if score_sizing:
                # 分级仓位: score越高, 仓位越大
                if c['sc'] >= 15: wt = 1.5
                elif c['sc'] >= 12: wt = 1.2
                elif c['sc'] >= 9: wt = 1.0
                else: wt = 0.7
            else:
                wt = 1.0

            notional = cap * cur_lev / max_pos * wt
            pos.append({'sym': c['sym'], 'dir': c['dir'], 'ed': dt,
                        'ep': c['ep'], 'not': notional, 'sc': c['sc']})

        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def pr(eq, trades, label):
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓"); return
    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
    ny = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    ann = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/ny) - 1) * 100
    mdd = eq_df['dd'].min()
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if len(dr) > 0 and dr.std() > 0 else 0
    if trades:
        td = pd.DataFrame(trades)
        wr = (td['r'] > 0).mean() * 100
        avg = td['r'].mean()
        aw = td[td['r'] > 0]['r'].mean() if (td['r'] > 0).any() else 0
        al = td[td['r'] <= 0]['r'].mean() if (td['r'] <= 0).any() else 0
        pf = abs(aw * (td['r'] > 0).sum() / (al * (td['r'] <= 0).sum())) if al != 0 and (td['r'] <= 0).sum() > 0 else 999
        td['year'] = pd.to_datetime(td['xd']).dt.year
    else:
        wr = avg = pf = 0; td = pd.DataFrame()

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  N:{len(trades)} WR:{wr:.1f}% PF:{pf:.2f} Sharpe:{sh:.2f} Avg:{avg:+.3f}%")
    print(f"  年化:{ann:.0f}% MDD:{mdd:.1f}%")
    if len(td) > 0 and 'year' in td.columns:
        for yr in sorted(td['year'].unique()):
            s = td[td['year'] == yr]
            print(f"    {yr}: N={len(s):4d} WR={(s['r']>0).mean()*100:.1f}% Avg={s['r'].mean():+.3f}%")

    return {'ann': ann, 'mdd': mdd, 'wr': wr, 'sh': sh}


def main():
    print("V74: 进一步优化 — 多仓位/波动率目标/分级仓位")
    print("目标: 年化1000%+ WR>50% MDD<30%")
    print("="*60)

    print("\n加载数据...")
    all_data = load_data()
    print("计算信号...")
    sd = compute_signals(all_data)

    # ═══ A. 基准: V73最佳 ═══
    print(f"\n{'='*60}")
    print("A. 基准: min=7 H=2 SL=-2 TP=3 lev=3 max_pos=3")
    print(f"{'='*60}")
    eq, tr = run_bt(sd, '2022-01-01', '2025-12-31', max_pos=3, lev=3, min_sc=7,
                     hold=2, sl=-2, tp=3)
    pr(eq, tr, "基准 lev=3x")

    # ═══ B. 更多仓位 ═══
    print(f"\n{'='*60}")
    print("B. 仓位数量测试")
    print(f"{'='*60}")
    for mp in [3, 5, 7]:
        for lev in [3, 5]:
            eq, tr = run_bt(sd, '2022-01-01', '2025-12-31', max_pos=mp, lev=lev,
                             min_sc=7, hold=2, sl=-2, tp=3)
            pr(eq, tr, f"max_pos={mp} lev={lev}x")

    # ═══ C. 持仓1天 ═══
    print(f"\n{'='*60}")
    print("C. 持仓1天 (intraday gap fade)")
    print(f"{'='*60}")
    for mp in [3, 5]:
        for lev in [3, 5, 7]:
            eq, tr = run_bt(sd, '2022-01-01', '2025-12-31', max_pos=mp, lev=lev,
                             min_sc=7, hold=1, sl=-2, tp=3)
            pr(eq, tr, f"H=1d mp={mp} lev={lev}x")

    # ═══ D. 波动率目标杠杆 ═══
    print(f"\n{'='*60}")
    print("D. 波动率目标杠杆")
    print(f"{'='*60}")
    for vt in [1.5, 2.0, 3.0]:
        for mp in [3, 5]:
            eq, tr = run_bt(sd, '2022-01-01', '2025-12-31', max_pos=mp, lev=5,
                             min_sc=7, hold=2, sl=-2, tp=3, vol_target=vt)
            pr(eq, tr, f"vol_target={vt}% mp={mp}")

    # ═══ E. 分级仓位 ═══
    print(f"\n{'='*60}")
    print("E. 分级仓位 (高分多配)")
    print(f"{'='*60}")
    for mp in [3, 5]:
        for lev in [3, 5]:
            eq, tr = run_bt(sd, '2022-01-01', '2025-12-31', max_pos=mp, lev=lev,
                             min_sc=7, hold=2, sl=-2, tp=3, score_sizing=True)
            pr(eq, tr, f"分级 mp={mp} lev={lev}x")

    # ═══ F. 组合: 分级+波动率目标+5仓 ═══
    print(f"\n{'='*60}")
    print("F. 组合优化")
    print(f"{'='*60}")
    combos = [
        (5, 5, 7, 2, -2, 3, 2.0, True),
        (5, 5, 7, 1, -2, 3, 2.0, True),
        (5, 7, 7, 1, -2, 3, 2.0, True),
        (5, 5, 8, 1, -2, 3, 2.0, True),
        (5, 5, 9, 1, -2, 3, 2.0, True),
        (3, 5, 7, 1, -2, 3, 2.0, True),
        (3, 7, 7, 1, -1.5, 2, 2.0, True),
        (5, 5, 7, 2, -2, 3, None, True),
        (7, 5, 7, 2, -2, 3, None, True),
        (7, 5, 7, 1, -2, 3, None, True),
        (5, 3, 7, 1, -2, 3, None, True),
    ]

    results = []
    for mp, lev, msc, hd, sl, tp, vt, ss in combos:
        eq, tr = run_bt(sd, '2022-01-01', '2025-12-31', max_pos=mp, lev=lev,
                         min_sc=msc, hold=hd, sl=sl, tp=tp, vol_target=vt, score_sizing=ss)
        eq_df = pd.DataFrame(eq)
        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
            continue
        td = pd.DataFrame(tr)
        wr = (td['r'] > 0).mean() * 100
        mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
        dr = eq_df['capital'].pct_change().dropna()
        sh = dr.mean() / dr.std() * (252**0.5) if len(dr) > 0 and dr.std() > 0 else 0
        ny = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
        ann = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/ny) - 1) * 100

        vt_s = f"VT={vt}" if vt else "-"
        results.append({
            'mp': mp, 'lev': lev, 'msc': msc, 'hd': hd, 'sl': sl, 'tp': tp,
            'vt': vt_s, 'ss': ss,
            'wr': wr, 'mdd': mdd, 'ann': ann, 'sh': sh, 'n': len(tr),
        })

    # 排序: WR>=50 且 MDD<=30% 中 Sharpe最高的
    good = [r for r in results if r['wr'] >= 50 and r['mdd'] >= -30]
    if good:
        good.sort(key=lambda x: -x['sh'])
        print(f"\n  满足条件 (WR>=50%, MDD<=30%) 的配置:")
        for r in good[:10]:
            print(f"    mp={r['mp']} lev={r['lev']} min={r['msc']} H={r['hd']}d "
                  f"SL={r['sl']} TP={r['tp']} {r['vt']}: "
                  f"N={r['n']} WR={r['wr']:.1f}% Ann={r['ann']:.0f}% MDD={r['mdd']:.1f}% Sh={r['sh']:.2f}")

        # 最佳配置详细
        best = good[0]
        eq, tr = run_bt(sd, '2022-01-01', '2025-12-31', max_pos=best['mp'], lev=best['lev'],
                         min_sc=best['msc'], hold=best['hd'], sl=best['sl'], tp=best['tp'],
                         vol_target=None if best['vt']=='-' else float(best['vt'].split('=')[1]),
                         score_sizing=best['ss'])
        pr(eq, tr, "★ 最佳配置 - 测试期")

        # Walk-forward
        eq_tr, tr_tr = run_bt(sd, '2015-01-01', '2021-12-31', max_pos=best['mp'], lev=best['lev'],
                               min_sc=best['msc'], hold=best['hd'], sl=best['sl'], tp=best['tp'],
                               vol_target=None if best['vt']=='-' else float(best['vt'].split('=')[1]),
                               score_sizing=best['ss'])
        pr(eq_tr, tr_tr, "★ 最佳配置 - 训练期")
    else:
        print(f"\n  未找到满足条件的配置, 显示最接近的:")
        results.sort(key=lambda x: -x['sh'])
        for r in results[:5]:
            print(f"    mp={r['mp']} lev={r['lev']} min={r['msc']} H={r['hd']}d "
                  f"SL={r['sl']} TP={r['tp']} {r['vt']}: "
                  f"N={r['n']} WR={r['wr']:.1f}% Ann={r['ann']:.0f}% MDD={r['mdd']:.1f}% Sh={r['sh']:.2f}")


if __name__ == '__main__':
    main()
