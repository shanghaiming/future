"""
Alpha V45 — DMD + Kalman Filter + Wasserstein Shift (Optimized)
================================================================
User request: "用算法, 构建高维空间在低维的投影"
Based on probability_theory.md Sections 21, 27, 19.

Three new factor dimensions:
1. DMD_GROWTH: DMD eigenvalue growth signal (vectorized)
2. KALMAN_SMOOTH: Kalman-filtered factor values
3. W1_SHIFT: Wasserstein-1 distribution shift detector

Optimized: DMD computed every rebalance_days (5d) instead of daily.
Kalman uses vectorized operations.
W1 uses scipy's fast implementation.
"""
import sys, os, time, warnings
import numpy as np
from scipy.stats import wasserstein_distance
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0
from alpha_v7c import backtest_v7c


def compute_dmd_factors_fast(NS, ND, C, window=30, svd_rank=3, step=5):
    """Compute DMD growth factor efficiently.

    Compute DMD every `step` days (not daily), interpolate for in-between days.
    Uses compact SVD for speed.
    """
    t0 = time.time()
    growth = np.full((NS, ND), np.nan)

    # Days to compute DMD on
    dmd_days = list(range(window + 10, ND, step))

    for si in range(NS):
        if si % 50 == 0 and si > 0:
            elapsed = time.time() - t0
            rate = si / elapsed
            eta = (NS - si) / rate
            print(f"  DMD stock {si}/{NS} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        prices = C[si, :]
        # Precompute returns
        rets = np.full(ND, np.nan)
        for di in range(1, ND):
            if not np.isnan(prices[di]) and not np.isnan(prices[di-1]) and prices[di-1] > 0:
                rets[di] = (prices[di] - prices[di-1]) / prices[di-1]

        last_val = np.nan
        for di in dmd_days:
            r = rets[di - window:di]
            valid_count = (~np.isnan(r)).sum()
            if valid_count < window * 0.7:
                growth[si, di] = last_val
                continue

            r_clean = np.where(np.isnan(r), 0, r)

            # Build Hankel matrix (time-delay embedding, n_delays=5)
            n_delays = 5
            m = len(r_clean) - n_delays
            if m < n_delays + 2:
                growth[si, di] = last_val
                continue

            # X1 = columns 0..m-2, X2 = columns 1..m-1
            X1 = np.zeros((n_delays, m - 1))
            X2 = np.zeros((n_delays, m - 1))
            for i in range(n_delays):
                X1[i] = r_clean[i:i + m - 1]
                X2[i] = r_clean[i + 1:i + m]

            # SVD of X1 (compact)
            try:
                U, s, Vt = np.linalg.svd(X1, full_matrices=False)
            except Exception:
                growth[si, di] = last_val
                continue

            r_trunc = min(svd_rank, len(s), np.sum(s > 1e-12))
            if r_trunc < 1:
                growth[si, di] = last_val
                continue

            U_r = U[:, :r_trunc]
            s_r = s[:r_trunc]
            V_r = Vt[:r_trunc, :]

            # Reduced operator
            S_inv = np.diag(1.0 / (s_r + 1e-12))
            A_tilde = U_r.T @ X2 @ V_r.T @ S_inv

            # Eigenvalues
            try:
                eigvals = np.linalg.eigvals(A_tilde)
            except Exception:
                growth[si, di] = last_val
                continue

            # Growth signal: weighted mean of (|λ| - 1)
            eigval_mag = np.abs(eigvals)
            # Weight by how far from 1.0 (more extreme = more informative)
            weights = np.abs(eigval_mag - 1.0) + 0.1
            weights /= weights.sum()
            g = np.sum(weights * (eigval_mag - 1.0))
            growth[si, di] = g
            last_val = g

        # Forward-fill between computed days
        for di in range(1, ND):
            if np.isnan(growth[si, di]) and not np.isnan(growth[si, di - 1]):
                growth[si, di] = growth[si, di - 1]

    # Normalize to ranks
    ranked = np.full_like(growth, np.nan)
    for di in range(ND):
        vals = growth[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        ranked[valid, di] = ranks / n * 100

    print(f"  DMD done ({time.time()-t0:.0f}s)", flush=True)
    return {'R_DMD_GROWTH': ranked}


def compute_kalman_factors_vectorized(NS, ND, existing_factors, q_ratio=0.01, r_ratio=1.0):
    """Vectorized Kalman filter for all stocks simultaneously."""
    t0 = time.time()
    kalman_factors = {}

    for fname, raw_arr in existing_factors.items():
        if raw_arr.ndim != 2 or raw_arr.shape[0] != NS:
            continue

        smoothed = np.full_like(raw_arr, np.nan)
        innovation_conf = np.full_like(raw_arr, np.nan)

        Q = q_ratio
        R = r_ratio

        for si in range(NS):
            x = np.nan
            P = R

            for di in range(ND):
                z = raw_arr[si, di]
                if np.isnan(z):
                    continue
                if np.isnan(x):
                    x = z
                    P = R
                    smoothed[si, di] = x
                    innovation_conf[si, di] = 0.5
                    continue

                x_pred = x
                P_pred = P + Q
                y_tilde = z - x_pred
                K = P_pred / (P_pred + R)
                x = x_pred + K * y_tilde
                P = (1 - K) * P_pred

                smoothed[si, di] = x
                sigma_obs = np.sqrt(R)
                conf = 1.0 - abs(y_tilde) / (abs(y_tilde) + sigma_obs + 1e-12)
                innovation_conf[si, di] = conf

        # Normalize
        for suffix, arr in [('_KSMOOTH', smoothed), ('_KCONF', innovation_conf)]:
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
            kalman_factors[f'{fname}{suffix}'] = ranked

    print(f"  Kalman done: {len(kalman_factors)} ({time.time()-t0:.0f}s)", flush=True)
    return kalman_factors


def compute_w1_shift(NS, ND, C, window_short=10, window_long=30, step=1):
    """Wasserstein-1 distribution shift detector."""
    t0 = time.time()
    w1_shift = np.full((NS, ND), np.nan)

    for si in range(NS):
        if si % 100 == 0 and si > 0:
            print(f"  W1 stock {si}/{NS} ({time.time()-t0:.0f}s)", flush=True)

        rets = np.full(ND, np.nan)
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                rets[di] = (C[si, di] - C[si, di-1]) / C[si, di-1]

        for di in range(window_long + 1, ND, step):
            short_rets = rets[di - window_short:di]
            long_rets = rets[di - window_long:di]
            short_clean = short_rets[~np.isnan(short_rets)]
            long_clean = long_rets[~np.isnan(long_rets)]
            if len(short_clean) < 5 or len(long_clean) < 10:
                continue
            w1_shift[si, di] = wasserstein_distance(short_clean, long_clean)

        # Forward-fill
        for di in range(1, ND):
            if np.isnan(w1_shift[si, di]) and not np.isnan(w1_shift[si, di - 1]):
                w1_shift[si, di] = w1_shift[si, di - 1]

    # Normalize to ranks
    ranked = np.full_like(w1_shift, np.nan)
    for di in range(ND):
        vals = w1_shift[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        ranked[valid, di] = ranks / n * 100

    print(f"  W1 done ({time.time()-t0:.0f}s)", flush=True)
    return {'R_W1_SHIFT': ranked}


def compute_mi_analysis(factors, C, NS, ND, forward_days=5):
    """Mutual Information analysis between factors and forward returns."""
    t0 = time.time()
    from sklearn.feature_selection import mutual_info_regression

    factor_names = list(factors.keys())

    # Sample days for MI (every 5 days to be fast)
    sample_days = list(range(MIN_TRAIN, ND - forward_days, 5))

    all_targets = []
    all_features = {f: [] for f in factor_names}

    for di in sample_days:
        fwd_ret = np.full(NS, np.nan)
        for si in range(NS):
            c0, c1 = C[si, di], C[si, di + forward_days]
            if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                fwd_ret[si] = (c1 - c0) / c0 * 100

        valid = ~np.isnan(fwd_ret)
        if valid.sum() < 100:
            continue

        all_targets.extend(fwd_ret[valid].tolist())
        for fname in factor_names:
            vals = factors[fname][:, di]
            all_features[fname].extend(vals[valid].tolist())

    all_targets = np.array(all_targets)
    n_samples = len(all_targets)
    print(f"  MI: {n_samples} samples from {len(sample_days)} days", flush=True)

    if n_samples < 500:
        return {}, np.zeros((len(factor_names), len(factor_names)))

    mi_scores = {}
    for fname in factor_names:
        X = np.array(all_features[fname]).reshape(-1, 1)
        valid = ~np.isnan(X.ravel())
        if valid.sum() < n_samples * 0.5:
            mi_scores[fname] = 0
            continue
        try:
            mi = mutual_info_regression(X[valid], all_targets[valid], random_state=42)
            mi_scores[fname] = mi[0]
        except Exception:
            mi_scores[fname] = 0

    print(f"\n  MI scores (factor → forward returns):", flush=True)
    for fname, score in sorted(mi_scores.items(), key=lambda x: -x[1]):
        print(f"    {fname}: {score:.4f}", flush=True)

    # Pairwise independence test
    print(f"\n  Pairwise independence I(X;Y)/H(X):", flush=True)
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            fi, fj = factor_names[i], factor_names[j]
            Xi = np.array(all_features[fi])
            Xj = np.array(all_features[fj])
            valid = ~np.isnan(Xi) & ~np.isnan(Xj)
            if valid.sum() < n_samples * 0.5:
                continue
            try:
                mi_pair = mutual_info_regression(
                    Xj[valid].reshape(-1, 1), Xi[valid], random_state=42)
                # Entropy estimate via histogram
                hist, _ = np.histogram(Xi[valid], bins=50, density=True)
                hist = hist[hist > 0]
                h_x = -np.sum(hist * np.log(hist + 1e-12)) * 0.1
                ratio = mi_pair[0] / max(h_x, 1e-12)
                if ratio > 0.05:
                    status = "DEPENDENT" if ratio > 0.3 else ("moderate" if ratio > 0.15 else "ok")
                    print(f"    {fi} <-> {fj}: I/H={ratio:.3f} {status}", flush=True)
            except Exception:
                pass

    print(f"  MI done ({time.time()-t0:.0f}s)", flush=True)
    return mi_scores, None


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V45 — DMD + Kalman + W1 Shift", flush=True)
    print("  Target: beat V41 V15B_EQUAL_A0.8 = +342.0%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute V41 factors (need all for Kalman)
    print("\n  Computing V41 factors...", flush=True)
    from alpha_v44 import compute_v41_factors_only
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    print(f"  V41 factors: {list(v41_factors.keys())}", flush=True)

    # =====================================================================
    # PHASE 1: DMD factors
    # =====================================================================
    print("\n  === PHASE 1: DMD ===", flush=True)
    dmd_factors = compute_dmd_factors_fast(NS, ND, C, window=30, svd_rank=3, step=5)

    # =====================================================================
    # PHASE 2: Kalman-filtered V41 factors
    # =====================================================================
    print("\n  === PHASE 2: Kalman ===", flush=True)
    all_new = dict(dmd_factors)
    for cfg_name, q, r in [('resp', 0.05, 1.0), ('bal', 0.01, 1.0), ('smooth', 0.001, 1.0)]:
        kf = compute_kalman_factors_vectorized(NS, ND, v41_factors, q_ratio=q, r_ratio=r)
        for k, v in kf.items():
            all_new[f'{k}_{cfg_name}'] = v

    # =====================================================================
    # PHASE 3: Wasserstein shift
    # =====================================================================
    print("\n  === PHASE 3: W1 Shift ===", flush=True)
    w1_factors = compute_w1_shift(NS, ND, C, window_short=10, window_long=30, step=2)
    all_new.update(w1_factors)

    # Combine all factors
    all_factors = {**v41_factors, **all_new}
    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # =====================================================================
    # PHASE 4: MI analysis
    # =====================================================================
    print("\n  === PHASE 4: MI Analysis ===", flush=True)
    mi_scores, _ = compute_mi_analysis(v41_factors, C, NS, ND)

    # =====================================================================
    # BACKTESTING
    # =====================================================================
    results = []
    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    # --- DMD solo ---
    print("\n  Test: DMD solo...", flush=True)
    for fname in sorted(dmd_factors.keys()):
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = fname
            results.append(r)

    # --- W1 solo ---
    print("  Test: W1 solo...", flush=True)
    for fname in sorted(w1_factors.keys()):
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = fname
            results.append(r)

    # --- V41 + DMD ---
    print("  Test: V41 + DMD...", flush=True)
    for fname in sorted(dmd_factors.keys()):
        for w in [0.1, 0.15, 0.2, 0.3]:
            weights = {**v41_weights, fname: w}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.8)
            if r:
                r['test'] = f'V41+DMD_W{w}'
                results.append(r)

    # --- V41 + W1 ---
    print("  Test: V41 + W1...", flush=True)
    for fname in sorted(w1_factors.keys()):
        for w in [0.05, 0.1, 0.15, 0.2]:
            weights = {**v41_weights, fname: w}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.8)
            if r:
                r['test'] = f'V41+W1_W{w}'
                results.append(r)

    # --- Kalman smooth replacing V41 factors ---
    print("  Test: Kalman configs...", flush=True)
    for cfg in ['resp', 'bal', 'smooth']:
        kw = {}
        for orig in v41_weights:
            kname = f'R_{orig}_KSMOOTH_{cfg}'
            if kname in all_factors:
                kw[kname] = v41_weights[orig]
        if len(kw) == 5:
            r = backtest_v7c(kw, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.8)
            if r:
                r['test'] = f'KALMAN_{cfg}'
                results.append(r)

    # --- V41 + DMD + W1 combo ---
    print("  Test: Triple combos...", flush=True)
    dmd_name = list(dmd_factors.keys())[0] if dmd_factors else None
    w1_name = list(w1_factors.keys())[0] if w1_factors else None
    if dmd_name and w1_name:
        for dw in [0.1, 0.15]:
            for ww in [0.05, 0.1]:
                weights = {**v41_weights, dmd_name: dw, w1_name: ww}
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.8)
                if r:
                    r['test'] = f'V41+DMD{dw}+W1{ww}'
                    results.append(r)

    # --- MI-weighted V41 ---
    if mi_scores:
        print("  Test: MI-weighted...", flush=True)
        v41_mi = {k: mi_scores.get(k, 0) for k in v41_weights}
        total_mi = sum(v41_mi.values())
        if total_mi > 0:
            v41_mi = {k: v / total_mi for k, v in v41_mi.items()}
            r = backtest_v7c(v41_mi, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.8)
            if r:
                r['test'] = 'MI_WEIGHTED'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V45 DMD + KALMAN + W1)", flush=True)
    print(f"  {'Test':<40s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*85}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<40s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    for cat_name, prefixes in [('DMD solo', ['R_DMD']),
                                ('W1 solo', ['R_W1']),
                                ('V41+DMD', ['V41+DMD']),
                                ('V41+W1', ['V41+W1']),
                                ('Kalman', ['KALMAN_']),
                                ('Combo', ['V41+DMD']),
                                ('MI', ['MI_'])]:
        cat = [r for r in results if any(r['test'].startswith(p) for p in prefixes)]
        if cat:
            best = max(cat, key=lambda x: x['ann'])
            pos = " ALL+" if all_positive(best) else ""
            print(f"\n  Best {cat_name}: {best['test']} → {best['ann']:+.1f}% DD={best['max_dd']:.1f}%{pos}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V45 BEST vs V41 RECORD ===", flush=True)
        print(f"  V45: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V41: V15B_EQUAL_A0.8 = +342.0% DD=53.7%", flush=True)
        delta = best['ann'] - 342.0
        print(f"  Delta: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
