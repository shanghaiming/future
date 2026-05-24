#!/usr/bin/env python3
"""
V89: ML深度集成 (优化版)
只在有意义的gap信号上训练/预测, 大幅减少计算量
Walk-Forward: 2年训练, 6个月测试
集成: RF + GBM + ExtraTrees
"""
import os, glob, numpy as np, pandas as pd, warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
CONTRACT_SPECS = 'scripts/contract_specs.py'
INITIAL_CAPITAL = 500_000


def load_and_prepare():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs", CONTRACT_SPECS)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)

    all_rows = []
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

        # 前向收益 (gap fade方向)
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n-1] = (c[1:] - c[:n-1]) / c[:n-1] * 100

        # gap fade方向收益
        fade_ret = np.where(gap < 0, fwd_ret, -fwd_ret)

        row_df = pd.DataFrame({
            'sym': sym, 'date': df['trade_date'], 'open': o, 'close': c,
            'gap': gap, 'gap_abs': np.abs(gap), 'gap_atr': gap_atr,
            'atr_pct': atr_pct, 'mom5': mom5, 'mom10': mom10,
            'vol_ratio': vol_ratio, 'oi_ch': oi_ch, 'clv': clv,
            'rsi': rsi, 'bb_pos': bb_pos,
            'ma5_d': np.where(ma5 > 0, (c - ma5) / ma5 * 100, 0),
            'ma20_d': np.where(ma20 > 0, (c - ma20) / ma20 * 100, 0),
            'ma60_d': np.where(ma60 > 0, (c - ma60) / ma60 * 100, 0),
            'trend': np.where((ma60 > 0) & ~np.isnan(ma20), ma20 / ma60 - 1, 0),
            'body_r': np.where(range_ > 0, (c - o) / range_, 0),
            'gap_x_oi': np.nan_to_num(gap) * np.nan_to_num(oi_ch),
            'fwd_ret': fwd_ret, 'fade_ret': fade_ret,
        })
        all_rows.append(row_df)

    print(f"  {len(all_rows)}品种")
    return pd.concat(all_rows, ignore_index=True)


FEATURE_COLS = [
    'gap', 'gap_abs', 'gap_atr', 'atr_pct', 'mom5', 'mom10',
    'vol_ratio', 'oi_ch', 'clv', 'rsi', 'bb_pos',
    'ma5_d', 'ma20_d', 'ma60_d', 'trend', 'body_r', 'gap_x_oi',
]


def main():
    print("V89: ML深度集成 (优化版)")
    print("=" * 60)

    print("\n加载数据...")
    df = load_and_prepare()

    # 只保留有意义的gap
    df = df[(df['gap'].abs() > 0.3) & (~df['fwd_ret'].isna()) & (~df['gap'].isna())].copy()
    df['target'] = (df['fade_ret'] > 0).astype(int)
    print(f"  有效gap样本: {len(df)} (正: {df['target'].mean()*100:.1f}%)")

    # ═══ Walk-Forward ═══
    print(f"\n{'='*60}")
    print("1. Walk-Forward验证 (2年训练, 6个月测试)")
    print(f"{'='*60}")

    windows = []
    for yr in range(2018, 2026):
        for half in [1, 2]:
            ts = f'{yr}-01-01' if half == 1 else f'{yr}-07-01'
            te = f'{yr}-06-30' if half == 1 else f'{yr}-12-31'
            trs = f'{yr-2}-01-01'
            tre = f'{yr-1}-12-31'
            windows.append((trs, tre, ts, te))

    all_oos = []
    print(f"\n  {'窗口':>36} │ {'N_tr':>6} {'N_te':>5} │ {'WR_base':>7} {'WR_ml70':>7} {'Avg70':>7}")
    print("  " + "-"*85)

    for trs, tre, ts, te in windows:
        train = df[(df['date'] >= trs) & (df['date'] <= tre)]
        test = df[(df['date'] >= ts) & (df['date'] <= te)]
        if len(train) < 500 or len(test) < 50: continue

        X_tr = train[FEATURE_COLS].fillna(0).values
        y_tr = train['target'].values
        X_te = test[FEATURE_COLS].fillna(0).values

        # 3模型集成概率
        probs = np.zeros(len(test))
        for Model, name in [
            (lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, min_samples_leaf=100, random_state=42), 'GBM'),
            (lambda: RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=100, random_state=42, n_jobs=-1), 'RF'),
            (lambda: ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_leaf=100, random_state=42, n_jobs=-1), 'ET'),
        ]:
            m = Model()
            m.fit(X_tr, y_tr)
            probs += m.predict_proba(X_te)[:, 1]
        probs /= 3

        test = test.copy()
        test['ml_prob'] = probs

        wr_base = test['target'].mean() * 100
        ml70 = test[test['ml_prob'] >= 0.7]
        wr70 = ml70['target'].mean() * 100 if len(ml70) > 0 else 0
        avg70 = ml70['fade_ret'].mean() if len(ml70) > 0 else 0

        label = f"{trs}~{tre} → {ts}~{te}"
        print(f"  {label:>36} │ {len(train):>6} {len(test):>5} │ {wr_base:>6.1f}% {wr70:>6.1f}% {avg70:>+6.3f}%")
        all_oos.append(test)

    # ═══ 汇总 ═══
    if all_oos:
        oos = pd.concat(all_oos)
        print(f"\n{'='*60}")
        print("2. OOS汇总")
        print(f"{'='*60}")
        print(f"  总样本: {len(oos)}")
        print(f"  基准WR: {oos['target'].mean()*100:.1f}%")
        print(f"  基准Avg: {oos['fade_ret'].mean():+.3f}%")

        print(f"\n  按ML概率阈值:")
        for t in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            sub = oos[oos['ml_prob'] >= t]
            if len(sub) > 50:
                wr = sub['target'].mean() * 100
                avg = sub['fade_ret'].mean()
                print(f"    prob>={t:.2f}: N={len(sub):5d} WR={wr:.1f}% Avg={avg:+.3f}%")

        # 特征重要性 (用最后一个GBM)
        print(f"\n  特征重要性 (最近窗口):")
        try:
            # 快速训练一个GBM获取重要性
            last_train = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2025-12-31')]
            if len(last_train) > 500:
                gbm = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
                gbm.fit(last_train[FEATURE_COLS].fillna(0), last_train['target'])
                for fname, fval in sorted(zip(FEATURE_COLS, gbm.feature_importances_), key=lambda x: -x[1]):
                    print(f"    {fname:>12s}: {fval*100:.2f}%")
        except: pass

        # 按年份分解ML信号质量
        print(f"\n  按年份 (prob>=0.7):")
        ml70_all = oos[oos['ml_prob'] >= 0.7].copy()
        if len(ml70_all) > 0:
            ml70_all['year'] = ml70_all['date'].dt.year
            for yr in sorted(ml70_all['year'].unique()):
                s = ml70_all[ml70_all['year'] == yr]
                print(f"    {yr}: N={len(s):4d} WR={s['target'].mean()*100:.1f}% Avg={s['fade_ret'].mean():+.3f}%")

    # ═══ ML回测 ═══
    print(f"\n{'='*60}")
    print("3. ML信号回测 (prob>=0.7)")
    print(f"{'='*60}")

    if all_oos:
        ml_sig = oos[oos['ml_prob'] >= 0.7].copy()
        print(f"  ML信号数: {len(ml_sig)}")

        # 简单回测: 每个信号开仓, 持1天
        cap = INITIAL_CAPITAL
        trades = []
        for _, sig in ml_sig.iterrows():
            notional = cap * 5 / 7
            ret = sig['fade_ret']
            pnl = notional * ret / 100
            cap += pnl
            trades.append({'sym': sig['sym'], 'ret': ret, 'pnl': pnl, 'prob': sig['ml_prob'],
                           'date': sig['date']})

        tdf = pd.DataFrame(trades)
        if len(tdf) > 0:
            wr = (tdf['ret'] > 0).mean() * 100
            avg = tdf['ret'].mean()
            total_ret = (cap / INITIAL_CAPITAL - 1) * 100
            print(f"  N={len(tdf)} WR={wr:.1f}% Avg={avg:+.3f}%")
            print(f"  注意: 这是简化回测(无SL/TP, 无仓位限制), 仅用于评估ML信号质量")

    # ═══ 对比 ═══
    print(f"\n{'='*60}")
    print("4. 最终对比")
    print(f"{'='*60}")
    print(f"  固定14因子评分 (V80): Sharpe=24.54, WR=71.2%, MDD=-5.1%")
    print(f"  ML集成OOS (prob>=0.7): 见上方Walk-Forward各窗口")


if __name__ == '__main__':
    main()
