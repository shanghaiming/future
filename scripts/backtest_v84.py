#!/usr/bin/env python3
"""
V84: ML信号优化 — 用机器学习替代固定权重
1. XGBoost学习14因子最优组合权重
2. 特征重要性分析: 哪些因子真正有用?
3. ML信号 vs 固定权重信号对比
4. 非线性交互: 因子之间的组合效应
5. 时序交叉验证: 防止look-ahead bias
"""
import os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
CONTRACT_SPECS = 'scripts/contract_specs.py'
TEST_START = '2022-01-01'
TEST_END = '2025-12-31'


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


def compute_features(all_data):
    """计算所有特征(因子)和标签(前向收益)"""
    all_features = []
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['vol'].values.astype(float)
        oi = df['oi'].values.astype(float)
        n = len(df)
        if n < 61: continue

        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        gap = np.full(n, np.nan); gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100

        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = pd.Series(tr).rolling(20).mean().values
        atr_pct = atr / c * 100

        ma5 = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values

        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        mom10 = np.full(n, np.nan); mom10[10:] = (c[10:] - c[:-10]) / c[:-10] * 100
        mom20 = np.full(n, np.nan); mom20[20:] = (c[20:] - c[:-20]) / c[:-20] * 100

        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v

        vol_ma5 = pd.Series(v).rolling(5).mean().values
        vol_ma20 = pd.Series(v).rolling(20).mean().values
        range_ = h - l
        clv = np.where(range_ > 0, (2*c - h - l) / range_, 0)

        gv = np.nan_to_num(gap)
        ga = np.nan_to_num(gap / np.where(atr_pct == 0, np.nan, atr_pct))

        # 日内范围
        intraday_range = np.where(c > 0, (h - l) / c * 100, 0)
        # 日内位置 (close在range中的位置)
        day_pos = np.where(range_ > 0, (c - l) / range_, 0.5)
        # 开盘跳空方向 (用于区分多空信号)
        gap_dir = np.sign(gv)

        # ═══ 基础特征 (不做多空区分) ═══
        features = pd.DataFrame({
            'sym': sym,
            'date': df['trade_date'],
            # Gap特征
            'gap_abs': np.abs(gv),
            'gap_signed': gv,
            'gap_atr_abs': np.abs(ga),
            'gap_atr_signed': ga,
            'gap_dir': gap_dir,
            # 价格特征
            'atr_pct': atr_pct,
            'intraday_range': intraday_range,
            'day_pos': day_pos,
            'clv': clv,
            # 均线
            'close_vs_ma5': (c - ma5) / ma5 * 100,
            'close_vs_ma20': (c - ma20) / ma20 * 100,
            'ma20_vs_ma60': (ma20 - ma60) / ma60 * 100,
            # 动量
            'mom5': mom5,
            'mom10': mom10,
            'mom20': mom20,
            # OI
            'oi_chg': oi_ch,
            'oi_chg_abs': np.abs(oi_ch),
            # 量
            'vol_ratio': np.where(vol_ma5 > 0, v / vol_ma5, 1),
            'vol_vs_ma20': np.where(vol_ma20 > 0, v / vol_ma20, 1),
            # 历史波动率
            'hv20': pd.Series(np.full(n, np.nan)).rolling(20).apply(
                lambda x: np.std(np.diff(x)/x[:-1])*100 if len(x) > 1 and x[0] > 0 else np.nan
            ).values if n > 20 else np.full(n, np.nan),
        })

        # ═══ 多头特征 (gap < 0 时) ═══
        features['long_gap'] = np.where(gv < 0, np.abs(gv), 0)
        features['long_gap_atr'] = np.where(gv < 0, np.abs(ga), 0)
        features['long_oi_up_down'] = np.where((oi_ch > 0) & (c < prev_c), 1, 0).astype(float)
        features['long_oi_dn_down'] = np.where((oi_ch < 0) & (c < prev_c), 1, 0).astype(float)
        features['long_mom_weak'] = np.where(mom5 < 0, np.abs(mom5), 0)
        features['long_below_ma5'] = np.where(c < ma5, 1, 0).astype(float)
        features['long_vol_surge'] = np.where((v > vol_ma5 * 1.5) & (c < prev_c), 1, 0).astype(float)
        features['long_clv'] = np.where(clv > 0, clv, 0)
        features['long_trend_up'] = np.where(ma20 > ma60, 1, 0).astype(float)

        # ═══ 空头特征 (gap > 0 时) ═══
        features['short_gap'] = np.where(gv > 0, gv, 0)
        features['short_gap_atr'] = np.where(gv > 0, ga, 0)
        features['short_oi_up_up'] = np.where((oi_ch > 0) & (c > prev_c), 1, 0).astype(float)
        features['short_oi_dn_up'] = np.where((oi_ch < 0) & (c > prev_c), 1, 0).astype(float)
        features['short_mom_strong'] = np.where(mom5 > 0, mom5, 0)
        features['short_above_ma5'] = np.where(c > ma5, 1, 0).astype(float)
        features['short_vol_surge'] = np.where((v > vol_ma5 * 1.5) & (c > prev_c), 1, 0).astype(float)
        features['short_clv'] = np.where(clv < 0, np.abs(clv), 0)
        features['short_trend_down'] = np.where(ma20 < ma60, 1, 0).astype(float)

        # ═══ 标签: 前向1日收益 ═══
        fwd = np.full(n, np.nan)
        if n > 1: fwd[:n-1] = (c[1:] - o[:n-1]) / o[:n-1] * 100
        features['fwd_1d'] = fwd
        features['fwd_1d_binary'] = (fwd > 0).astype(float)

        # 固定权重得分
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

        features['score_long'] = s_l
        features['score_short'] = s_s
        features['best_score'] = np.maximum(s_l, s_s)
        features['best_dir'] = np.where(s_l >= s_s, 1, -1)  # 1=long, -1=short

        all_features.append(features)

    df_all = pd.concat(all_features, ignore_index=True)
    return df_all


def train_ml_model(df_all, train_start, train_end):
    """训练XGBoost模型"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # 特征列
    feature_cols = [
        'gap_abs', 'gap_signed', 'gap_atr_abs', 'gap_atr_signed', 'gap_dir',
        'atr_pct', 'intraday_range', 'day_pos', 'clv',
        'close_vs_ma5', 'close_vs_ma20', 'ma20_vs_ma60',
        'mom5', 'mom10', 'mom20',
        'oi_chg', 'oi_chg_abs',
        'vol_ratio', 'vol_vs_ma20',
        'long_gap', 'long_gap_atr', 'long_oi_up_down', 'long_oi_dn_down',
        'long_mom_weak', 'long_below_ma5', 'long_vol_surge', 'long_clv', 'long_trend_up',
        'short_gap', 'short_gap_atr', 'short_oi_up_up', 'short_oi_dn_up',
        'short_mom_strong', 'short_above_ma5', 'short_vol_surge', 'short_clv', 'short_trend_down',
    ]

    # 筛选有gap信号的样本 (做多或做空得分>=5)
    mask = (df_all['date'] >= train_start) & (df_all['date'] <= train_end)
    mask = mask & (df_all['best_score'] >= 5)  # 只用有一定信号的样本
    train = df_all[mask].dropna(subset=feature_cols + ['fwd_1d_binary'])

    if len(train) < 100:
        print(f"    训练样本不足: {len(train)}")
        return None, feature_cols

    X_train = train[feature_cols].values
    y_train = train['fwd_1d_binary'].values

    # Gradient Boosting
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 训练集准确率
    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    print(f"    训练样本: {len(train)}, 准确率: {acc:.3f}")

    return model, feature_cols


def apply_ml_signal(df_all, model, feature_cols, test_start, test_end, min_prob=0.6):
    """用ML模型生成信号"""
    mask = (df_all['date'] >= test_start) & (df_all['date'] <= test_end)
    mask = mask & (df_all['best_score'] >= 5)  # 只对有信号的样本应用ML
    test = df_all[mask].copy()

    if model is None or len(test) == 0:
        return test

    X_test = test[feature_cols].values
    # 处理NaN
    X_test = np.nan_to_num(X_test, nan=0)

    probs = model.predict_proba(X_test)
    if probs.shape[1] == 2:
        test['ml_prob'] = probs[:, 1]  # P(positive return)
    else:
        test['ml_prob'] = 0.5

    # ML信号: 概率>0.6认为是好信号
    test['ml_signal'] = (test['ml_prob'] >= min_prob).astype(int)
    # ML增强得分: 在原有得分基础上, ML概率高的加权
    test['ml_score'] = test['best_score'] * test['ml_prob']

    return test


def run_bt_with_score(signal_df, start, end, score_col='best_score', dir_col='best_dir',
                       min_score=7, max_pos=7, lev=5, hold=1, sl_pct=-1.5, tp_pct=4.0):
    """用任意得分列回测"""
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []

    # 构建快速查找
    signal_df = signal_df.dropna(subset=[score_col]).copy()

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            sub = signal_df[(signal_df['sym'] == p['sym']) & (signal_df['date'] == dt)]
            if len(sub) == 0:
                keep.append(p); continue
            row = sub.iloc[0]
            cur_h = row.get('intraday_range', 0)  # placeholder
            # 需要原始数据获取high/low/close
            # 简化: 从signal_df中用已有的信息
            keep.append(p); continue  # 暂时简化

        # 这里的回测需要原始OHLC数据, 用signal_df不够
        # 改用原始data
        break

    # 用不同的方法: 直接在signal_df上做回测
    # 重新设计: 把ML得分合并到原始信号中
    return None, None


def main():
    print("V84: ML信号优化 — 机器学习替代固定权重")
    print("="*60)

    print("\n加载数据...")
    all_data = load_data()

    print("计算特征...")
    df_all = compute_features(all_data)
    print(f"  总样本: {len(df_all):,}")
    print(f"  日期范围: {df_all['date'].min().strftime('%Y-%m-%d')} ~ {df_all['date'].max().strftime('%Y-%m-%d')}")

    # ═══ A. ML模型训练 ═══
    print(f"\n{'='*60}")
    print("A. ML模型训练 (Gradient Boosting)")
    print(f"{'='*60}")

    # 训练: 2016-2021
    print("\n  训练期: 2016-2021")
    model, feature_cols = train_ml_model(df_all, '2016-01-01', '2021-12-31')

    if model is not None:
        # 特征重要性
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n  特征重要性 Top 15:")
        for _, row in importances.head(15).iterrows():
            print(f"    {row['feature']:>20s}: {row['importance']:.4f}")

    # ═══ B. ML信号应用 ═══
    print(f"\n\n{'='*60}")
    print("B. ML信号评估")
    print(f"{'='*60}")

    if model is not None:
        # 测试期
        for min_prob in [0.55, 0.60, 0.65, 0.70]:
            test_mask = (df_all['date'] >= TEST_START) & (df_all['date'] <= TEST_END)
            test_mask = test_mask & (df_all['best_score'] >= 5)
            test = df_all[test_mask].dropna(subset=feature_cols).copy()

            X_test = np.nan_to_num(test[feature_cols].values)
            probs = model.predict_proba(X_test)
            test['ml_prob'] = probs[:, 1] if probs.shape[1] == 2 else 0.5

            # ML筛选后的信号
            ml_pass = test[test['ml_prob'] >= min_prob]
            if len(ml_pass) == 0: continue

            wr = (ml_pass['fwd_1d'] > 0).mean() * 100
            avg = ml_pass['fwd_1d'].mean()
            n = len(ml_pass)

            # 对比: 固定权重得分>=7
            fixed_pass = test[test['best_score'] >= 7]
            f_wr = (fixed_pass['fwd_1d'] > 0).mean() * 100
            f_avg = fixed_pass['fwd_1d'].mean()
            f_n = len(fixed_pass)

            print(f"\n  ML阈值={min_prob}:")
            print(f"    ML筛选: N={n:>5} WR={wr:.1f}% Avg={avg:+.3f}%")
            print(f"    固定≥7: N={f_n:>5} WR={f_wr:.1f}% Avg={f_avg:+.3f}%")
            print(f"    ML提升: WR={wr-f_wr:+.1f}% Avg={avg-f_avg:+.3f}%")

    # ═══ C. 按ML概率分组 ═══
    print(f"\n\n{'='*60}")
    print("C. ML概率分组分析")
    print(f"{'='*60}")

    if model is not None:
        test_mask = (df_all['date'] >= TEST_START) & (df_all['date'] <= TEST_END)
        test_mask = test_mask & (df_all['best_score'] >= 5)
        test = df_all[test_mask].dropna(subset=feature_cols).copy()
        X_test = np.nan_to_num(test[feature_cols].values)
        probs = model.predict_proba(X_test)
        test['ml_prob'] = probs[:, 1] if probs.shape[1] == 2 else 0.5

        print(f"\n  {'ML概率区间':>12} {'N':>6} {'WR':>6} {'Avg':>8} {'vs固定≥7':>10}")
        print("-" * 50)

        for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
            sub = test[(test['ml_prob'] >= lo) & (test['ml_prob'] < hi)]
            if len(sub) == 0: continue
            wr = (sub['fwd_1d'] > 0).mean() * 100
            avg = sub['fwd_1d'].mean()
            # 同区间内固定≥7的表现
            fixed_sub = sub[sub['best_score'] >= 7]
            f_wr = (fixed_sub['fwd_1d'] > 0).mean() * 100 if len(fixed_sub) > 0 else 0
            print(f"  [{lo:.1f}-{hi:.1f}): {len(sub):>6} {wr:>5.1f}% {avg:>+7.3f}% "
                  f"{'↑' if wr > f_wr else '↓'}{abs(wr-f_wr):.1f}%")

    # ═══ D. ML增强回测 ═══
    print(f"\n\n{'='*60}")
    print("D. ML增强得分回测 (需要原始数据)")
    print(f"{'='*60}")

    # 将ML概率合并到原始signal_data, 然后回测
    # 简化: 直接在df_all上做快速回测
    if model is not None:
        # 全部数据计算ML概率
        all_mask = df_all['best_score'] >= 5
        all_test = df_all[all_mask].dropna(subset=feature_cols).copy()
        X_all = np.nan_to_num(all_test[feature_cols].values)
        probs_all = model.predict_proba(X_all)
        all_test['ml_prob'] = probs_all[:, 1] if probs_all.shape[1] == 2 else 0.5

        # 生成ML增强得分
        # ML得分 = best_score * ml_prob * 2 (放大差异)
        all_test['ml_enhanced_score'] = all_test['best_score'] * all_test['ml_prob'] * 2

        # 不同得分方式对比
        print(f"\n  测试期 (2022-2025) 不同信号方式对比:")
        test_period = all_test[(all_test['date'] >= TEST_START) & (all_test['date'] <= TEST_END)]

        for method, col, threshold in [
            ("固定≥7", 'best_score', 7),
            ("ML prob≥0.6", 'ml_prob', 0.6),
            ("ML prob≥0.65", 'ml_prob', 0.65),
            ("ML prob≥0.7", 'ml_prob', 0.7),
            ("ML增强≥7", 'ml_enhanced_score', 7),
            ("ML增强≥8", 'ml_enhanced_score', 8),
            ("ML增强≥10", 'ml_enhanced_score', 10),
        ]:
            sub = test_period[test_period[col] >= threshold]
            if len(sub) < 100: continue
            wr = (sub['fwd_1d'] > 0).mean() * 100
            avg = sub['fwd_1d'].mean()
            print(f"    {method:>12}: N={len(sub):>5} WR={wr:.1f}% Avg={avg:+.3f}%")

    # ═══ E. 非线性交互分析 ═══
    print(f"\n\n{'='*60}")
    print("E. 因子交互分析")
    print(f"{'='*60}")

    test_mask = (df_all['date'] >= TEST_START) & (df_all['date'] <= TEST_END)
    test = df_all[test_mask & (df_all['best_score'] >= 7)].copy()

    # 交互组
    print(f"\n  固定≥7 + 条件组合:")
    conditions = [
        ("Gap>1%", test['gap_abs'] > 1.0),
        ("Gap>1.5%+ATR", (test['gap_abs'] > 1.5) | (test['gap_atr_abs'] > 1.5)),
        ("OI确认", (test['long_oi_up_down'] > 0) | (test['short_oi_up_up'] > 0)),
        ("趋势一致", (test['long_trend_up'] > 0) | (test['short_trend_down'] > 0)),
        ("放量", test['vol_ratio'] > 1.5),
        ("CLV极端", test['clv'].abs() > 0.5),
        ("Gap大+OI确认", (test['gap_abs'] > 1.0) & ((test['long_oi_up_down'] > 0) | (test['short_oi_up_up'] > 0))),
        ("Gap大+趋势", (test['gap_abs'] > 1.0) & ((test['long_trend_up'] > 0) | (test['short_trend_down'] > 0))),
        ("全部确认", (test['gap_abs'] > 1.0) & ((test['long_oi_up_down'] > 0) | (test['short_oi_up_up'] > 0)) & ((test['long_trend_up'] > 0) | (test['short_trend_down'] > 0))),
    ]

    print(f"    {'条件':>16} {'N':>6} {'WR':>6} {'Avg':>8}")
    print("    " + "-"*40)
    for name, cond in conditions:
        sub = test[cond]
        if len(sub) < 50: continue
        wr = (sub['fwd_1d'] > 0).mean() * 100
        avg = sub['fwd_1d'].mean()
        print(f"    {name:>16} {len(sub):>6} {wr:>5.1f}% {avg:>+7.3f}%")


if __name__ == '__main__':
    main()
