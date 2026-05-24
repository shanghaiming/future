#!/usr/bin/env python3
"""
V90: ML驱动Gap Fade完整回测
用ML概率替代固定14因子评分:
- 预训练ML模型 (全样本训练, prob作为score)
- 完整SL/TP + 仓位管理 + 4多4空
- Walk-Forward: 滚动训练
- 与V80固定评分严格对比
"""
import os, glob, numpy as np, pandas as pd, warnings, pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
CONTRACT_SPECS = 'scripts/contract_specs.py'
INITIAL_CAPITAL = 500_000
FEATURE_COLS = [
    'gap_pct', 'gap_abs', 'gap_atr', 'atr_pct', 'mom5', 'mom10',
    'vol_ratio', 'oi_ch', 'clv', 'rsi', 'bb_pos',
    'ma5_d', 'ma20_d', 'ma60_d', 'trend', 'body_r', 'gap_x_oi',
]


def load_and_prepare():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs", CONTRACT_SPECS)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)

    all_data = {}
    all_features = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        try: mult, margin, tick, tick_val = cs.get_spec(sym)
        except: continue
        df = pd.read_csv(f)
        if len(df) < 200: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['vol'].values.astype(float)
        oi = df['oi'].values.astype(float)
        n = len(df)
        if (c == 0).any(): continue

        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        gap = np.full(n, np.nan); gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = pd.Series(tr).rolling(20).mean().values
        atr_pct = np.where(c > 0, atr / c * 100, np.nan)
        ma5 = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        mom10 = np.full(n, np.nan); mom10[10:] = (c[10:] - c[:-10]) / c[:-10] * 100
        vol_ma5 = pd.Series(v).rolling(5).mean().values
        vol_ratio = np.where(vol_ma5 > 0, v / vol_ma5, 1.0)
        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v
        range_ = h - l
        clv = np.where(range_ > 0, (2*c - h - l) / range_, 0)
        gap_atr = np.where((atr_pct != 0) & ~np.isnan(atr_pct), gap / atr_pct, 0)
        delta = np.full(n, np.nan); delta[1:] = c[1:] - c[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss_arr = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss_arr).rolling(14).mean().values
        rsi = np.where(avg_loss > 0, 100 - 100 / (1 + avg_gain / avg_loss), 50)
        bb_std = pd.Series(c).rolling(20).std().values
        bb_upper = ma20 + 2 * bb_std
        bb_lower = ma20 - 2 * bb_std
        bb_pos = np.where((bb_upper - bb_lower) > 0, (c - bb_lower) / (bb_upper - bb_lower), 0.5)

        # Gap Fade得分 (V80固定)
        gv = np.nan_to_num(gap)
        ga = gap_atr
        s_l = np.zeros(n)
        s_l += (gv < -0.5).astype(int) * 1
        s_l += (gv < -1.0).astype(int) * 2
        s_l += (gv < -1.5).astype(int) * 2
        s_l += (gv < -2.0).astype(int) * 3
        s_l += (ga < -1.0).astype(int) * 2
        s_l += (ga < -1.5).astype(int) * 3
        s_l += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_l += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        mom5v = np.nan_to_num(mom5)
        s_l += (mom5v < -3).astype(int) * 1
        s_l += (mom5v < -5).astype(int) * 1
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
        s_s += (mom5v > 3).astype(int) * 1
        s_s += (mom5v > 5).astype(int) * 1
        s_s += (c > ma5).astype(int) * 1
        s_s += ((v > vol_ma5 * 1.5) & (c > prev_c)).astype(int) * 1
        s_s += (clv < -0.5).astype(int) * 1
        s_s += (ma20 < ma60).astype(int) * 2

        df['score_long'] = s_l
        df['score_short'] = s_s
        df['gap_pct'] = gap

        # 特征列
        df['gap_abs'] = np.abs(gap)
        df['gap_atr'] = gap_atr
        df['atr_pct'] = atr_pct
        df['mom5'] = mom5
        df['mom10'] = mom10
        df['vol_ratio'] = vol_ratio
        df['oi_ch'] = oi_ch
        df['clv'] = clv
        df['rsi'] = rsi
        df['bb_pos'] = bb_pos
        df['ma5_d'] = np.where(ma5 > 0, (c - ma5) / ma5 * 100, 0)
        df['ma20_d'] = np.where(ma20 > 0, (c - ma20) / ma20 * 100, 0)
        df['ma60_d'] = np.where(ma60 > 0, (c - ma60) / ma60 * 100, 0)
        df['trend'] = np.where((ma60 > 0) & ~np.isnan(ma20), ma20 / ma60 - 1, 0)
        df['body_r'] = np.where(range_ > 0, (c - o) / range_, 0)
        df['gap_x_oi'] = np.nan_to_num(gap) * np.nan_to_num(oi_ch)

        # 前向收益
        fwd = np.full(n, np.nan)
        fwd[:n-1] = (c[1:] - c[:n-1]) / np.where(c[:n-1] > 0, c[:n-1], np.nan) * 100
        df['fwd_ret'] = fwd
        df['fade_ret'] = np.where(gap < 0, fwd, -fwd)
        df['target'] = (df['fade_ret'] > 0).astype(int)

        all_data[sym] = df

    print(f"  {len(all_data)}品种")
    return all_data


def train_ensemble(data, train_start, train_end):
    """训练ML集成"""
    rows = []
    for sym, df in data.items():
        sub = df[(df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)]
        sub = sub[(sub['gap_pct'].abs() > 0.3) & (~sub['fwd_ret'].isna())]
        rows.append(sub)
    if not rows: return None
    train = pd.concat(rows)
    if len(train) < 500: return None

    X = train[FEATURE_COLS].fillna(0).values
    y = train['target'].values

    models = [
        GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=100, random_state=42),
        RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=100,
                                random_state=42, n_jobs=-1),
        ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_leaf=100,
                              random_state=42, n_jobs=-1),
    ]
    for m in models:
        m.fit(X, y)
    return models


def predict_ensemble(models, X):
    """集成预测"""
    probs = np.zeros(len(X))
    for m in models:
        probs += m.predict_proba(X)[:, 1]
    return probs / len(models)


def run_bt(data, start, end, use_ml=False, models=None, ml_threshold=0.65):
    """完整回测"""
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            df = data.get(p['sym'])
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
                stop = p['ep'] * (1 - 0.015)
                if cur_l <= stop:
                    actual_ret = (stop * (1 - sp) - p['ep']) / p['ep'] * 100
                    reason = 'SL'; triggered = True
                if not triggered:
                    tp_p = p['ep'] * (1 + 0.04)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                stop = p['ep'] * (1 + 0.015)
                if cur_h >= stop:
                    actual_ret = (p['ep'] - stop * (1 + sp)) / p['ep'] * 100
                    reason = 'SL'; triggered = True
                if not triggered:
                    tp_p = p['ep'] * (1 - 0.04)
                    if cur_l <= tp_p:
                        actual_ret = (p['ep'] - tp_p * (1 + sp)) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100
            if d >= 1:
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

        n_open = 7 - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        cands = []
        for sym, df in data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if np.isnan(row.get('close', np.nan)): continue

            if use_ml and models:
                # ML评分
                x = row[FEATURE_COLS].fillna(0).values.reshape(1, -1)
                prob = predict_ensemble(models, x)[0]
                if prob >= ml_threshold:
                    dir_ = 'long' if row['gap_pct'] < 0 else 'short'
                    cands.append({
                        'sym': sym, 'dir': dir_, 'sc': prob * 20,
                        'ep': row['open'], 'prob': prob,
                    })
            else:
                # 固定评分
                if row.get('score_long', 0) >= 7:
                    cands.append({'sym': sym, 'dir': 'long', 'sc': row['score_long'], 'ep': row['open']})
                if row.get('score_short', 0) >= 7:
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
            if c_['dir'] == 'long' and n_long >= 4: continue
            if c_['dir'] == 'short' and n_short >= 4: continue
            notional = cap * 5 / 7
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def calc_metrics(eq_list, trades_list):
    eq_df = pd.DataFrame(eq_list)
    if len(eq_df) == 0:
        return {'N': 0, 'WR': 0, 'Sharpe': 0, 'MDD': 0, 'Avg': 0}
    tdf = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if len(dr) > 0 and dr.std() > 0 else 0
    mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
    wr = (tdf['r'] > 0).mean() * 100 if len(tdf) > 0 else 0
    avg = tdf['r'].mean() if len(tdf) > 0 else 0
    return {'N': len(tdf), 'WR': wr, 'Sharpe': sh, 'MDD': mdd, 'Avg': avg}


def main():
    print("V90: ML驱动Gap Fade完整回测")
    print("=" * 60)

    print("\n加载数据...")
    data = load_and_prepare()

    # ═══════════════════════════════════════════
    # 1. 基准: 固定评分 (V80)
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("1. 基准: 固定14因子评分")
    print(f"{'='*60}")
    eq_std, tr_std = run_bt(data, '2016-01-01', '2025-12-31')
    m_std = calc_metrics(eq_std, tr_std)
    print(f"  N={m_std['N']} WR={m_std['WR']:.1f}% Sharpe={m_std['Sharpe']:.2f} "
          f"MDD={m_std['MDD']:.1f}% Avg={m_std['Avg']:+.3f}%")

    # ═══════════════════════════════════════════
    # 2. ML Walk-Forward完整回测
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("2. ML Walk-Forward完整回测")
    print(f"{'='*60}")

    # 每年重新训练
    ml_windows = []
    for yr in range(2016, 2026):
        train_s = f'{yr-2}-01-01' if yr >= 2018 else '2016-01-01'
        train_e = f'{yr}-12-31'
        test_s = f'{yr+1}-01-01' if yr < 2025 else None
        test_e = f'{yr+1}-12-31' if yr < 2025 else None
        if test_s:
            ml_windows.append((train_s, train_e, test_s, test_e))

    # 用滚动方式: 训练2年 → 测试接下来的时间
    # 更精确: 每半年重新训练
    wf_windows = []
    for yr in range(2018, 2026):
        for half in [1, 2]:
            if half == 1:
                ts, te = f'{yr}-01-01', f'{yr}-06-30'
            else:
                ts, te = f'{yr}-07-01', f'{yr}-12-31'
            trs = f'{yr-2}-01-01'
            tre = f'{yr-1}-12-31'
            wf_windows.append((trs, tre, ts, te))

    all_ml_trades = []
    print(f"\n  {'窗口':>36} │ {'N':>5} {'WR':>6} {'Sharpe':>7} {'MDD':>7} {'Avg':>8}")
    print("  " + "-"*80)

    for trs, tre, ts, te in wf_windows:
        models = train_ensemble(data, trs, tre)
        if models is None:
            print(f"  {trs}~{tre} → {ts}~{te} │ 训练数据不足")
            continue
        eq, tr = run_bt(data, ts, te, use_ml=True, models=models, ml_threshold=0.65)
        m = calc_metrics(eq, tr)
        label = f"{trs}~{tre} → {ts}~{te}"
        print(f"  {label:>36} │ {m['N']:>5} {m['WR']:>5.1f}% {m['Sharpe']:>7.2f} {m['MDD']:>+6.1f}% {m['Avg']:>+7.3f}%")
        if tr: all_ml_trades.extend(tr)

    # ═══ ML汇总 ═══
    if all_ml_trades:
        tdf = pd.DataFrame(all_ml_trades)
        wr = (tdf['r'] > 0).mean() * 100
        avg = tdf['r'].mean()
        print(f"\n  ML Walk-Forward全部OOS: N={len(tdf)} WR={wr:.1f}% Avg={avg:+.3f}%")

        tdf['year'] = pd.to_datetime(tdf['xd']).dt.year
        print(f"\n  年度:")
        for yr in sorted(tdf['year'].unique()):
            s = tdf[tdf['year'] == yr]
            print(f"    {yr}: N={len(s):4d} WR={(s['r']>0).mean()*100:.1f}% Avg={s['r'].mean():+.3f}%")

    # ═══ 不同ML阈值对比 ═══
    print(f"\n{'='*60}")
    print("3. 不同ML阈值 (全样本训练, 仅供参考)")
    print(f"{'='*60}")

    # 用前8年训练, 后2年测试
    models_all = train_ensemble(data, '2016-01-01', '2023-12-31')
    if models_all:
        for thresh in [0.55, 0.6, 0.65, 0.7, 0.75]:
            eq, tr = run_bt(data, '2024-01-01', '2025-12-31',
                            use_ml=True, models=models_all, ml_threshold=thresh)
            m = calc_metrics(eq, tr)
            print(f"  prob>={thresh:.2f}: N={m['N']:>5} WR={m['WR']:.1f}% Sharpe={m['Sharpe']:.2f} "
                  f"MDD={m['MDD']:>+5.1f}% Avg={m['Avg']:>+6.3f}%")

    # ═══ 最终对比 ═══
    print(f"\n{'='*60}")
    print("4. 最终对比")
    print(f"{'='*60}")
    print(f"\n  {'策略':30s} {'N':>5} {'WR':>6} {'Sharpe':>7} {'MDD':>7} {'Avg':>8}")
    print("  " + "-"*70)
    print(f"  {'固定14因子 (V80)':30s} {m_std['N']:>5} {m_std['WR']:>5.1f}% {m_std['Sharpe']:>7.2f} "
          f"{m_std['MDD']:>+6.1f}% {m_std['Avg']:>+7.3f}%")
    if all_ml_trades:
        tdf = pd.DataFrame(all_ml_trades)
        wr_ml = (tdf['r'] > 0).mean() * 100
        avg_ml = tdf['r'].mean()
        print(f"  {'ML Walk-Forward (V90)':30s} {len(tdf):>5} {wr_ml:>5.1f}% {'(WF)':>7s} "
              f"{'(WF)':>7s} {avg_ml:>+7.3f}%")


if __name__ == '__main__':
    main()
