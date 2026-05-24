#!/usr/bin/env python3
"""
V83: 滚动Walk-Forward验证
最严格的参数稳定性测试:
- 每6个月重新优化参数
- 用过去1年数据选参数, 未来6个月测试
- 检查最优参数是否随时间变化
- 如果参数不稳定, 说明策略可能过拟合
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

        df['score_long'] = s_l
        df['score_short'] = s_s
        df['gap_pct'] = gap
        signal_data[sym] = df
    return signal_data


def run_bt(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
           sl_pct=-1.5, tp_pct=4.0, max_long=4, max_short=4):
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            df = signal_data.get(p['sym'])
            if df is None: keep.append(p); continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: keep.append(p); continue
            row = df.loc[idx[0]]
            cur_h, cur_l, cur_c = row['high'], row['low'], row['close']
            if np.isnan(cur_c): keep.append(p); continue
            d = (dt - p['ed']).days
            sp = 0.001
            triggered = False
            actual_ret = None
            reason = None
            if p['dir'] == 'long':
                if sl_pct:
                    stop = p['ep'] * (1 + sl_pct / 100)
                    if cur_l <= stop:
                        actual_ret = (stop * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and tp_pct:
                    tp_p = p['ep'] * (1 + tp_pct / 100)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if sl_pct:
                    stop = p['ep'] * (1 - sl_pct / 100)
                    if cur_h >= stop:
                        actual_ret = (p['ep'] - stop * (1 + sp)) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and tp_pct:
                    tp_p = p['ep'] * (1 - tp_pct / 100)
                    if cur_l <= tp_p:
                        actual_ret = (p['ep'] - tp_p * (1 + sp)) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100
            if d >= hold:
                if not triggered: reason = 'exp'
            else:
                if not triggered: keep.append(p); continue
            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'], 'xd': dt,
                    'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason,
                })
        pos = keep
        cap += pnl
        if cap <= 0: break

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
        best = {}
        for c_ in cands:
            if c_['sym'] not in best or c_['sc'] > best[c_['sym']]['sc']:
                best[c_['sym']] = c_
        ranked = sorted(best.values(), key=lambda x: -x['sc'])
        n_long = sum(1 for p in pos if p['dir'] == 'long')
        n_short = sum(1 for p in pos if p['dir'] == 'short')
        for c_ in ranked:
            if n_open <= 0: break
            if max_long and c_['dir'] == 'long' and n_long >= max_long: continue
            if max_short and c_['dir'] == 'short' and n_short >= max_short: continue
            notional = cap * lev / max_pos
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def optimize_params(signal_data, train_start, train_end):
    """在训练期搜索最优参数"""
    param_grid = [
        # (min_sc, sl, tp, hold, max_pos, max_long, max_short)
        (7, -1.5, 4.0, 1, 7, 4, 4),  # 默认最优
        (7, -1.5, 3.0, 1, 7, 4, 4),
        (7, -2.0, 4.0, 1, 7, 4, 4),
        (7, -2.0, 5.0, 1, 7, 4, 4),
        (8, -1.5, 4.0, 1, 7, 4, 4),
        (9, -1.5, 4.0, 1, 7, 4, 4),
        (7, -1.5, 4.0, 1, 5, 3, 3),
        (7, -2.0, 3.0, 1, 5, 3, 3),
        (7, -1.5, 4.0, 2, 7, 4, 4),
        (7, -2.0, 3.0, 2, 7, 4, 4),
    ]

    best_sharpe = -999
    best_params = param_grid[0]

    for min_sc, sl, tp, hd, mp, ml, ms in param_grid:
        eq, tr = run_bt(signal_data, train_start, train_end,
                         max_pos=mp, min_sc=min_sc, hold=hd,
                         sl_pct=sl, tp_pct=tp, max_long=ml, max_short=ms)
        eq_df = pd.DataFrame(eq)
        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
            continue
        dr = eq_df['capital'].pct_change().dropna()
        if len(dr) == 0 or dr.std() == 0:
            continue
        sh = dr.mean() / dr.std() * (252**0.5)

        if sh > best_sharpe:
            best_sharpe = sh
            best_params = (min_sc, sl, tp, hd, mp, ml, ms)

    return best_params, best_sharpe


def main():
    print("V83: 滚动Walk-Forward验证")
    print("="*60)

    print("\n加载数据...")
    all_data = load_data()
    print("计算信号...")
    sd = compute_signals(all_data)

    # ═══ 滚动Walk-Forward ═══
    print(f"\n{'='*60}")
    print("滚动Walk-Forward: 训练1年, 测试6个月, 滚动前进")
    print(f"{'='*60}")

    # 时间段: 2016-01 ~ 2025-12
    windows = []
    for test_year in range(2017, 2026):
        for test_half in [1, 2]:
            if test_half == 1:
                test_start = f'{test_year}-01-01'
                test_end = f'{test_year}-06-30'
            else:
                test_start = f'{test_year}-07-01'
                test_end = f'{test_year}-12-31'

            train_start = f'{test_year-1}-01-01'
            train_end = f'{test_year-1}-12-31'

            windows.append((train_start, train_end, test_start, test_end))

    print(f"\n  {'训练期':>24} {'测试期':>24} │ {'最优参数':>30} │ {'训练Sh':>6} {'测试WR':>6} {'测试Sh':>6} {'测试MDD':>7}")
    print("-" * 120)

    all_test_trades = []
    oos_results = []

    for train_s, train_e, test_s, test_e in windows:
        # 训练: 找最优参数
        best_params, train_sh = optimize_params(sd, train_s, train_e)
        min_sc, sl, tp, hd, mp, ml, ms = best_params

        # 测试: 用最优参数
        eq, tr = run_bt(sd, test_s, test_e, max_pos=mp, min_sc=min_sc, hold=hd,
                         sl_pct=sl, tp_pct=tp, max_long=ml, max_short=ms)
        eq_df = pd.DataFrame(eq)
        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
            print(f"  {train_s:>12}~{train_e:>10} {test_s:>12}~{test_e:>10} │ 测试期亏损")
            continue

        tdf = pd.DataFrame(tr) if tr else pd.DataFrame()
        wr = (tdf['r'] > 0).mean() * 100 if len(tdf) > 0 else 0
        mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
        dr = eq_df['capital'].pct_change().dropna()
        sh = dr.mean() / dr.std() * (252**0.5) if len(dr) > 0 and dr.std() > 0 else 0

        param_str = f"min={min_sc} SL={sl} TP={tp} H={hd} mp={mp}"
        train_label = f"{train_s}~{train_e}"
        test_label = f"{test_s}~{test_e}"

        print(f"  {train_label:>24} {test_label:>24} │ {param_str:>30} │ {train_sh:>6.2f} {wr:>5.1f}% {sh:>6.2f} {mdd:>+6.1f}%")

        if tr:
            all_test_trades.extend(tr)
        oos_results.append({'wr': wr, 'mdd': mdd, 'sh': sh, 'n': len(tr), 'params': best_params})

    # ═══ 汇总 ═══
    print(f"\n\n{'='*60}")
    print("Walk-Forward汇总")
    print(f"{'='*60}")

    if all_test_trades:
        tdf = pd.DataFrame(all_test_trades)
        wr = (tdf['r'] > 0).mean() * 100
        avg = tdf['r'].mean()
        tdf['year'] = pd.to_datetime(tdf['xd']).dt.year

        print(f"\n  全部OOS交易: N={len(all_test_trades)} WR={wr:.1f}% Avg={avg:+.3f}%")

        for yr in sorted(tdf['year'].unique()):
            s = tdf[tdf['year'] == yr]
            print(f"    {yr}: N={len(s):4d} WR={(s['r']>0).mean()*100:.1f}% Avg={s['r'].mean():+.3f}%")

    if oos_results:
        wrs = [r['wr'] for r in oos_results]
        mdds = [r['mdd'] for r in oos_results]
        shs = [r['sh'] for r in oos_results if r['sh'] != 0]
        print(f"\n  半年度统计:")
        print(f"    WR:  均值={np.mean(wrs):.1f}% 最低={np.min(wrs):.1f}% 最高={np.max(wrs):.1f}%")
        print(f"    MDD: 均值={np.mean(mdds):.1f}% 最差={np.min(mdds):.1f}%")
        print(f"    Sharpe: 均值={np.mean(shs):.2f} 最低={np.min(shs):.2f}")

        # 参数一致性
        print(f"\n  参数一致性分析:")
        param_counts = {}
        for r in oos_results:
            p = r['params']
            key = f"min={p[0]} SL={p[1]} TP={p[2]} H={p[3]} mp={p[4]}"
            param_counts[key] = param_counts.get(key, 0) + 1
        for k, v in sorted(param_counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}次/{len(oos_results)} ({v/len(oos_results)*100:.0f}%)")

    # ═══ 固定参数 vs 滚动优化 对比 ═══
    print(f"\n\n{'='*60}")
    print("固定参数 vs 滚动优化对比")
    print(f"{'='*60}")

    # 固定最优参数
    eq_fixed, tr_fixed = run_bt(sd, '2017-01-01', '2025-12-31')
    eq_f_df = pd.DataFrame(eq_fixed)
    if len(eq_f_df) > 0:
        tdf_f = pd.DataFrame(tr_fixed)
        wr_f = (tdf_f['r'] > 0).mean() * 100
        mdd_f = ((eq_f_df['capital'] - eq_f_df['capital'].cummax()) / eq_f_df['capital'].cummax() * 100).min()
        dr_f = eq_f_df['capital'].pct_change().dropna()
        sh_f = dr_f.mean() / dr_f.std() * (252**0.5) if dr_f.std() > 0 else 0

        print(f"\n  固定参数 (min=7, SL=-1.5, TP=4, H=1, mp=7, 4多4空):")
        print(f"    N={len(tr_fixed)} WR={wr_f:.1f}% Sharpe={sh_f:.2f} MDD={mdd_f:.1f}%")

    if all_test_trades:
        tdf_oos = pd.DataFrame(all_test_trades)
        wr_oos = (tdf_oos['r'] > 0).mean() * 100
        print(f"\n  滚动优化:")
        print(f"    N={len(all_test_trades)} WR={wr_oos:.1f}%")

        print(f"\n  WR差异: {wr_f:.1f}% vs {wr_oos:.1f}% = {wr_f - wr_oos:+.1f}%")


if __name__ == '__main__':
    main()
