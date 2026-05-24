"""
Alpha V57b — LightGBM Regression (Fixed) + PCA Latent Factors
==============================================================
Fix: 1) Use regression instead of LambdaRank (avoid label mapping issue)
     2) Fix V56 weight construction
     3) Fix PCA to use RAW (unranked) factors for dimensionality reduction
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


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V57b — LightGBM Regression + PCA (Fixed)")
    print("  V56 record: +1450.0% DD=25.2%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing all factors...", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41, **v48, **v49, **v52, **v55}

    # V56 winning weights (EXACT same as alpha_v55.py test)
    v54_base = {'R_BWP_BNW': 0.205, 'R_TENSION': 0.205, 'R_VWCM': 0.205,
                'R_BVR': 0.154, 'R_BUY_FRAC': 0.138, 'R_VPIN': 0.092}
    v56_weights = {**v54_base, 'R_SHOCK_MOM': 0.08, 'R_TREND_ACC': 0.15}
    total = sum(v56_weights.values())
    v56_norm = {k: v / total for k, v in v56_weights.items()}

    results = []

    # =====================================================================
    # Baseline: V56 (exact same construction as V55 test)
    # =====================================================================
    print("\n  V56 baseline...", flush=True)
    r = backtest_v7c(v56_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V56_BASE'
        results.append(r)
        print(f"  V56: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # LightGBM Regression: predict forward 5d return rank
    # =====================================================================
    # Use regression (not lambdarank) to avoid label mapping issues
    factor_names = sorted([k for k in all_factors.keys() if k.startswith('R_')])
    F = len(factor_names)
    print(f"\n  {F} factors available for ML", flush=True)

    # Build factor tensor
    factor_tensor = np.full((NS, ND, F), np.nan)
    for fi, fname in enumerate(factor_names):
        factor_tensor[:, :, fi] = all_factors[fname]

    # Forward 5d return
    fwd_days = 5
    fwd_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - fwd_days):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di + fwd_days]) and C[si, di] > 0:
                fwd_ret[si, di] = (C[si, di + fwd_days] - C[si, di]) / C[si, di]

    # Rank-normalize forward returns per day
    fwd_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = fwd_ret[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1) / n * 100
        fwd_rank[valid, di] = ranks

    # LGB score
    lgb_score = np.full((NS, ND), np.nan)
    retrain_every = 120

    t0 = time.time()
    for di in range(MIN_TRAIN + 120, ND):
        if di % retrain_every != 0:
            continue

        # Training data: all valid (stock, day) pairs from MIN_TRAIN to di-1
        # Sample to keep memory reasonable
        train_days = list(range(MIN_TRAIN, di))
        np.random.seed(42)
        if len(train_days) > 500:
            train_days = sorted(np.random.choice(train_days, 500, replace=False))

        X_parts = []
        y_parts = []
        for d in train_days:
            feat = factor_tensor[:, d, :]
            label = fwd_rank[:, d]
            valid = ~np.any(np.isnan(feat), axis=1) & ~np.isnan(label)
            if valid.sum() > 30:
                X_parts.append(feat[valid])
                y_parts.append(label[valid])

        if not X_parts:
            continue

        X_train = np.vstack(X_parts).astype(np.float32)
        y_train = np.concatenate(y_parts).astype(np.float32)

        # Train regression model
        try:
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                max_depth=3,
                num_leaves=15,
                learning_rate=0.05,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
            )
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"    LGB train failed at di={di}: {e}", flush=True)
            continue

        # Predict for next retrain_every days
        for pred_di in range(di, min(di + retrain_every, ND)):
            feat = factor_tensor[:, pred_di, :]
            valid = ~np.any(np.isnan(feat), axis=1)
            if valid.sum() > 30:
                scores = model.predict(feat[valid])
                lgb_score[valid, pred_di] = scores

        if di % 600 == 0:
            print(f"    LGB trained at di={di}, samples={len(X_train)} ({time.time()-t0:.0f}s)", flush=True)

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

    all_factors['R_LGB_SCORE'] = lgb_ranked
    print(f"  LGB done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Test LGB solo + combined
    # =====================================================================
    print("\n  Testing LGB...", flush=True)

    # LGB solo
    for atr in [0.5, 0.6, 0.7, 0.8, 1.0]:
        r = backtest_v7c({'R_LGB_SCORE': 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=atr)
        if r:
            r['test'] = f'LGB_SOLO_A{atr}'
            results.append(r)
            print(f"    LGB_SOLO_A{atr}: {r['ann']:+.1f}%", flush=True)

    # V56 + LGB
    for w_lgb in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        weights = {**v56_norm, 'R_LGB_SCORE': w_lgb}
        total = sum(weights.values())
        wn = {k: v / total for k, v in weights.items()}
        r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.5)
        if r:
            r['test'] = f'V56+LGB_W{w_lgb:.2f}'
            results.append(r)
            print(f"    V56+LGB_W{w_lgb:.2f}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # PCA on RANK-NORMALIZED factors
    # =====================================================================
    print("\n  Computing PCA latent factors...", flush=True)
    for n_comp in [5, 8]:
        # Use expanding window PCA
        pca_score = np.full((NS, ND, n_comp), np.nan)
        pca_model = None
        last_train = -999
        t0_pca = time.time()

        for di in range(MIN_TRAIN + 60, ND):
            if di - last_train < 60:
                # Just transform
                if pca_model is not None:
                    feat = factor_tensor[:, di, :]
                    valid = ~np.any(np.isnan(feat), axis=1)
                    if valid.sum() > 50:
                        transformed = pca_model.transform(feat[valid])
                        for k in range(n_comp):
                            pca_score[valid, di, k] = transformed[:, k]
                continue

            # Train PCA
            train_data = factor_tensor[:, MIN_TRAIN:di, :].reshape(-1, F)
            valid = ~np.any(np.isnan(train_data), axis=1)
            clean = train_data[valid]
            if len(clean) > 2000:
                pca_model = PCA(n_components=n_comp)
                pca_model.fit(clean)
                last_train = di

                # Also transform current day
                feat = factor_tensor[:, di, :]
                valid_now = ~np.any(np.isnan(feat), axis=1)
                if valid_now.sum() > 50:
                    transformed = pca_model.transform(feat[valid_now])
                    for k in range(n_comp):
                        pca_score[valid_now, di, k] = transformed[:, k]

        # Rank normalize and add to factors
        for k in range(n_comp):
            name = f'R_PCA_{n_comp}_{k}'
            arr = pca_score[:, :, k]
            ranked = np.full_like(arr, np.nan)
            for di in range(ND):
                vals = arr[:, di]
                valid = ~np.isnan(vals)
                n = valid.sum()
                if n < 50:
                    continue
                order = np.argsort(vals[valid])
                ranks = np.empty(n)
                ranks[order] = np.arange(1, n + 1)
                ranked[valid, di] = ranks / n * 100
            all_factors[name] = ranked

        print(f"  PCA{n_comp} done ({time.time()-t0_pca:.0f}s)", flush=True)

        # Test PCA solo and combined
        pca_names = [f'R_PCA_{n_comp}_{k}' for k in range(n_comp)]

        # Solo: equal weight all PCA components
        weights = {n: 1.0/n_comp for n in pca_names}
        for atr in [0.5, 0.7, 0.8]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'PCA{n_comp}_EQ_A{atr}'
                results.append(r)

        # V56 + top PCA components
        for k in range(min(3, n_comp)):
            name = f'R_PCA_{n_comp}_{k}'
            for w in [0.05, 0.08, 0.10]:
                weights = {**v56_norm, name: w}
                total = sum(weights.values())
                wn = {k2: v / total for k2, v in weights.items()}
                r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                if r:
                    r['test'] = f'V56+PCA{n_comp}_{k}_W{w:.2f}'
                    results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*100}", flush=True)
    print(f"  ALL RESULTS (V57b ML FACTOR DISCOVERY)", flush=True)
    print(f"  {'Test':<40s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<40s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V57b BEST ===", flush=True)
        print(f"  V57b: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V56 RECORD: +1450.0% DD=25.2%", flush=True)
        delta = best['ann'] - 1450.0
        print(f"  Delta from V56: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
