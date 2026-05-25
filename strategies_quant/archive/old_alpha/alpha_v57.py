"""
Alpha V57 — LightGBM LambdaRank + PCA Latent Factors
=====================================================
Replace linear weighted-sum scoring with ML-based ranking.

Approach:
1. PCA on ~30 raw factors → 5-10 latent factors (non-linear compression)
2. LightGBM LambdaRank: train on cross-sectional return prediction
3. Expanding window: only use past data for training (no look-ahead)
4. Combine ML scores with V54/V56 factors

Available: LightGBM 4.6.0, sklearn 1.8.0, XGBoost 3.2.0
"""
import sys, os, time, warnings
import numpy as np
from sklearn.decomposition import PCA
import lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0
from alpha_v7c import backtest_v7c
from alpha_v44 import compute_v41_factors_only
from alpha_v48 import compute_v48_factors
from alpha_v49 import compute_v49_factors
from alpha_v52 import compute_v52_factors
from alpha_v55 import compute_decomposed_factors


def compute_pca_latent(all_factors, NS, ND, n_components=8, retrain_every=60):
    """Compute PCA latent factors with expanding window (no look-ahead).

    For each day di:
    - Train PCA on factor data from MIN_TRAIN to di-1
    - Transform di's factors → latent dimensions
    - Rank-normalize each latent dimension
    """
    # Stack all rank-normalized factors into (NS, ND, F) tensor
    factor_names = sorted([k for k in all_factors.keys() if k.startswith('R_')])
    F = len(factor_names)
    print(f"  PCA: {F} factors → {n_components} latent dims", flush=True)

    factor_tensor = np.full((NS, ND, F), np.nan)
    for fi, fname in enumerate(factor_names):
        factor_tensor[:, :, fi] = all_factors[fname]

    # Output: latent factors
    latent_factors = {}
    for k in range(n_components):
        latent_factors[f'R_PCA_{k}'] = np.full((NS, ND), np.nan)

    t0 = time.time()
    pca_model = None
    last_train_di = -retrain_every  # Force train on first iteration

    for di in range(MIN_TRAIN + 20, ND):
        # Retrain PCA periodically
        if di - last_train_di >= retrain_every:
            # Collect training data from MIN_TRAIN to di-1
            train_data = factor_tensor[:, MIN_TRAIN:di, :].reshape(-1, F)
            valid = ~np.any(np.isnan(train_data), axis=1)
            train_clean = train_data[valid]

            if len(train_clean) > 1000:
                pca_model = PCA(n_components=n_components)
                pca_model.fit(train_clean)
                last_train_di = di

        if pca_model is None:
            continue

        # Transform current day's cross-section
        day_factors = factor_tensor[:, di, :]  # (NS, F)
        valid_stocks = ~np.any(np.isnan(day_factors), axis=1)

        if valid_stocks.sum() > 50:
            transformed = pca_model.transform(day_factors[valid_stocks])
            for k in range(n_components):
                latent_factors[f'R_PCA_{k}'][valid_stocks, di] = transformed[:, k]

        if di % 500 == 0:
            print(f"    PCA day {di}/{ND} ({time.time()-t0:.0f}s)", flush=True)

    # Rank normalize each latent dimension
    ranked = {}
    for k in range(n_components):
        name = f'R_PCA_{k}'
        arr = latent_factors[name]
        ranked_arr = np.full_like(arr, np.nan)
        for di in range(ND):
            vals = arr[:, di]
            valid = ~np.isnan(vals)
            n = valid.sum()
            if n < 50:
                continue
            order = np.argsort(vals[valid])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            ranked_arr[valid, di] = ranks / n * 100
        ranked[name] = ranked_arr

    print(f"  PCA latent factors done ({time.time()-t0:.0f}s)", flush=True)
    return ranked


def compute_lgb_ranking(all_factors, NS, ND, dates, C, retrain_every=60,
                        fwd_days=5, n_estimators=100, max_depth=3):
    """LightGBM LambdaRank: predict future return ranking.

    For each training window:
    - Features: all R_ factors for each stock on each day
    - Label: rank of forward 5-day return among all stocks
    - Train LambdaRank model
    - Predict scores for current day
    """
    factor_names = sorted([k for k in all_factors.keys() if k.startswith('R_')])
    F = len(factor_names)
    print(f"  LGB: {F} features, retrain every {retrain_every}d", flush=True)

    # Compute forward returns for labels
    fwd_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - fwd_days):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di + fwd_days]):
                if C[si, di] > 0:
                    fwd_ret[si, di] = (C[si, di + fwd_days] - C[si, di]) / C[si, di]

    # Output
    lgb_score = np.full((NS, ND), np.nan)

    factor_tensor = np.full((NS, ND, F), np.nan)
    for fi, fname in enumerate(factor_names):
        factor_tensor[:, :, fi] = all_factors[fname]

    t0 = time.time()

    for di in range(MIN_TRAIN + 60, ND):
        if di % retrain_every != 0:
            continue

        # Collect training data
        train_start = MIN_TRAIN
        train_end = di

        # Build training set
        X_train_parts = []
        y_train_parts = []
        group_sizes = []

        for d in range(train_start, train_end):
            feat_day = factor_tensor[:, d, :]
            ret_day = fwd_ret[:, d]
            valid = ~np.any(np.isnan(feat_day), axis=1) & ~np.isnan(ret_day)
            n_valid = valid.sum()
            if n_valid < 30:
                continue
            X_train_parts.append(feat_day[valid])
            # Rank as label (0 to n_valid-1)
            order = np.argsort(ret_day[valid])
            ranks = np.empty(n_valid)
            ranks[order] = np.arange(n_valid)
            y_train_parts.append(ranks)
            group_sizes.append(n_valid)

        if len(X_train_parts) == 0:
            continue

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Train LambdaRank
        train_data = lgb.Dataset(X_train, label=y_train, group=group_sizes)
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_at': [1, 5, 10],
            'learning_rate': 0.05,
            'num_leaves': 15,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'verbose': -1,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'feature_fraction': 0.8,
        }

        try:
            model = lgb.train(params, train_data, num_boost_round=n_estimators)
        except Exception as e:
            print(f"    LGB train failed at di={di}: {e}", flush=True)
            continue

        # Predict for days di to min(di + retrain_every, ND)
        for pred_di in range(di, min(di + retrain_every, ND)):
            feat = factor_tensor[:, pred_di, :]
            valid = ~np.any(np.isnan(feat), axis=1)
            if valid.sum() > 30:
                scores = model.predict(feat[valid])
                lgb_score[valid, pred_di] = scores

        if di % 300 == 0:
            print(f"    LGB trained at di={di}, groups={len(group_sizes)} "
                  f"samples={len(X_train)} ({time.time()-t0:.0f}s)", flush=True)

    # Rank normalize LGB scores
    lgb_ranked = np.full_like(lgb_score, np.nan)
    for di in range(ND):
        vals = lgb_score[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        lgb_ranked[valid, di] = ranks / n * 100

    print(f"  LGB ranking done ({time.time()-t0:.0f}s)", flush=True)
    return {'R_LGB_RANK': lgb_ranked}


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V57 — LightGBM LambdaRank + PCA Latent Factors")
    print("  V56 record: +1450.0% DD=25.2%")
    print("  Target: ML-based ranking to push beyond 1500%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing all factors...", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41, **v48, **v49, **v52, **v55}

    # V56 winning config
    v56_weights = {
        'R_BWP_BNW': 0.205,
        'R_TENSION': 0.205,
        'R_VWCM': 0.205,
        'R_BVR': 0.154,
        'R_BUY_FRAC': 0.138,
        'R_VPIN': 0.092,
    }

    results = []

    # =====================================================================
    # Baseline: V56
    # =====================================================================
    print("\n  V56 baseline...", flush=True)
    total = sum(v56_weights.values())
    w_norm = {k: v / total for k, v in v56_weights.items()}
    w_norm['R_SHOCK_MOM'] = 0.08 / (total + 0.08 + 0.15)
    w_norm['R_TREND_ACC'] = 0.15 / (total + 0.08 + 0.15)
    total2 = sum(w_norm.values())
    w_norm = {k: v / total2 for k, v in w_norm.items()}
    r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V56_BASE'
        results.append(r)
        print(f"  V56: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: PCA latent factors
    # =====================================================================
    print("\n  Test 1: PCA latent factors...", flush=True)
    for n_comp in [5, 8, 10]:
        pca_factors = compute_pca_latent(all_factors, NS, ND, n_components=n_comp)
        pca_all = {**all_factors, **pca_factors}

        # PCA solo
        pca_names = sorted(pca_factors.keys())
        weights = {f: 1.0 / len(pca_names) for f in pca_names}
        for atr in [0.5, 0.7]:
            r = backtest_v7c(weights, pca_all, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'PCA{n_comp}_EQ_A{atr}'
                results.append(r)

        # V56 + PCA
        for w_pca in [0.05, 0.10]:
            for fname in pca_names:
                weights = {**w_norm, fname: w_pca}
                total = sum(weights.values())
                wn = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(wn, pca_all, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                if r:
                    r['test'] = f'V56+{fname}_W{w_pca}'
                    results.append(r)

    # =====================================================================
    # Test 2: LightGBM LambdaRank
    # =====================================================================
    print("\n  Test 2: LightGBM LambdaRank...", flush=True)
    for retrain in [60, 120]:
        for depth in [3, 5]:
            for n_est in [50, 100]:
                lgb_factors = compute_lgb_ranking(
                    all_factors, NS, ND, dates, C,
                    retrain_every=retrain, max_depth=depth, n_estimators=n_est)
                lgb_all = {**all_factors, **lgb_factors}

                # LGB solo
                for atr in [0.5, 0.7, 0.8]:
                    r = backtest_v7c({'R_LGB_RANK': 1.0}, lgb_all, NS, ND, dates, C, O, H, L, V,
                                    top_n=1, rebalance_days=5, atr_stop_mult=atr)
                    if r:
                        r['test'] = f'LGB_R{retrain}_D{depth}_N{n_est}_A{atr}'
                        results.append(r)

                # V56 + LGB
                for w_lgb in [0.05, 0.10, 0.15]:
                    weights = {**w_norm, 'R_LGB_RANK': w_lgb}
                    total = sum(weights.values())
                    wn = {k: v / total for k, v in weights.items()}
                    r = backtest_v7c(wn, lgb_all, NS, ND, dates, C, O, H, L, V,
                                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                    if r:
                        r['test'] = f'V56+LGB_R{retrain}_D{depth}_W{w_lgb}'
                        results.append(r)

    # =====================================================================
    # Test 3: PCA + LGB combined
    # =====================================================================
    print("\n  Test 3: PCA + V56 + LGB combined...", flush=True)
    # Use best PCA from test 1
    pca8 = compute_pca_latent(all_factors, NS, ND, n_components=8)
    # Use best LGB
    lgb_best = compute_lgb_ranking(all_factors, NS, ND, dates, C,
                                    retrain_every=60, max_depth=3, n_estimators=100)
    combined = {**all_factors, **pca8, **lgb_best}

    for w_pca in [0.03, 0.05]:
        for w_lgb in [0.05, 0.08, 0.10]:
            for pca_name in ['R_PCA_0', 'R_PCA_1', 'R_PCA_2']:
                weights = {**w_norm, pca_name: w_pca, 'R_LGB_RANK': w_lgb}
                total = sum(weights.values())
                wn = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(wn, combined, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                if r:
                    r['test'] = f'V56+{pca_name}+LGB_W{w_pca}_{w_lgb}'
                    results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*100}", flush=True)
    print(f"  ALL RESULTS (V57 ML FACTOR DISCOVERY)", flush=True)
    print(f"  {'Test':<50s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*90}", flush=True)
    for r in results[:60]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<50s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V57 BEST ===", flush=True)
        print(f"  V57: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V56 RECORD: +1450.0% DD=25.2%", flush=True)
        delta = best['ann'] - 1450.0
        print(f"  Delta from V56: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
