"""
Alpha V46 — Independent Factor Selection + Regime-Adaptive Weights
==================================================================
V45 MI analysis revealed:
- R_TENSION <-> R_SMA_DEV DEPENDENT (I/H=1.424) — redundant!
- R_BWP_BNW <-> R_TENSION DEPENDENT (I/H=1.156) — redundant!
- R_BWP_BNW has MI=0 with forward returns
- R_HAR_RV_RATIO_INV has MI=0 with forward returns

Hypothesis: Removing dependent factors and focusing on truly independent
ones may improve returns by reducing noise from redundant signals.

Tests:
1. Remove one of each dependent pair, keep the higher-MI one
2. Only use truly independent subsets
3. Add DMD as independent factor (it's uncorrelated with V41 factors)
4. Kalman-filtered V41 factors with MI-optimal weights
5. Search for new factor combinations using all available factors
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0
from alpha_v7c import backtest_v7c
from alpha_v44 import compute_v41_factors_only
from alpha_v45 import compute_dmd_factors_fast, compute_kalman_factors_vectorized


def compute_independent_subsets(factors, C, NS, ND):
    """Find maximally independent factor subsets using MI."""
    from sklearn.feature_selection import mutual_info_regression

    factor_names = list(factors.keys())
    n_factors = len(factor_names)

    # Sample for MI
    sample_days = list(range(MIN_TRAIN, ND - 5, 5))
    all_features = {f: [] for f in factor_names}

    for di in sample_days:
        for fname in factor_names:
            vals = factors[fname][:, di]
            all_features[fname].extend(vals.tolist())

    # Pairwise MI
    mi_matrix = np.zeros((n_factors, n_factors))
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            fi, fj = factor_names[i], factor_names[j]
            Xi = np.array(all_features[fi])
            Xj = np.array(all_features[fj])
            valid = ~np.isnan(Xi) & ~np.isnan(Xj)
            if valid.sum() < 1000:
                continue
            try:
                mi = mutual_info_regression(
                    Xj[valid].reshape(-1, 1), Xi[valid], random_state=42)
                mi_matrix[i, j] = mi_matrix[j, i] = mi[0]
            except Exception:
                pass

    # MI with forward returns
    mi_forward = np.zeros(n_factors)
    for di in sample_days:
        fwd = np.full(NS, np.nan)
        for si in range(NS):
            c0, c1 = C[si, di], C[si, di + 5]
            if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                fwd[si] = (c1 - c0) / c0 * 100

    all_fwd = []
    all_feat = {f: [] for f in factor_names}
    for di in sample_days:
        fwd = np.full(NS, np.nan)
        for si in range(NS):
            c0, c1 = C[si, di], C[si, di + 5]
            if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                fwd[si] = (c1 - c0) / c0 * 100
        valid = ~np.isnan(fwd)
        if valid.sum() < 100:
            continue
        all_fwd.extend(fwd[valid].tolist())
        for fname in factor_names:
            vals = factors[fname][:, di]
            all_feat[fname].extend(vals[valid].tolist())

    all_fwd = np.array(all_fwd)
    for i, fname in enumerate(factor_names):
        X = np.array(all_feat[fname]).reshape(-1, 1)
        valid = ~np.isnan(X.ravel())
        if valid.sum() < len(all_fwd) * 0.5:
            continue
        try:
            mi = mutual_info_regression(X[valid], all_fwd[valid], random_state=42)
            mi_forward[i] = mi[0]
        except Exception:
            pass

    print("\n  Factor MI ranking (forward returns):", flush=True)
    for i in np.argsort(-mi_forward):
        print(f"    {factor_names[i]}: MI={mi_forward[i]:.4f}", flush=True)

    print("\n  Pairwise MI matrix:", flush=True)
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            if mi_matrix[i, j] > 0.001:
                print(f"    {factor_names[i][:20]} <-> {factor_names[j][:20]}: MI={mi_matrix[i, j]:.4f}", flush=True)

    return mi_forward, mi_matrix


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V46 — Independent Factor Selection", flush=True)
    print("  Target: beat V41 V15B_EQUAL_A0.8 = +342.0%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute V41 factors + DMD
    print("\n  Computing factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    dmd_factors = compute_dmd_factors_fast(NS, ND, C, window=30, svd_rank=3, step=5)
    all_factors = {**v41_factors, **dmd_factors}

    # V41 baseline weights
    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    results = []

    # =====================================================================
    # TEST 0: V41 baseline (should be +342%)
    # =====================================================================
    print("\n  Test 0: V41 baseline...", flush=True)
    r = backtest_v7c(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.8)
    if r:
        r['test'] = 'V41_BASELINE'
        results.append(r)
        print(f"  Baseline: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # TEST 1: Remove dependent factors
    # =====================================================================
    print("\n  Test 1: Remove dependent factors...", flush=True)
    # V45 showed: TENSION <-> SMA_DEV dependent (I/H=1.424)
    # BWP_BNW <-> TENSION dependent (I/H=1.156)
    # Strategy: Remove the lower-MI factor from each dependent pair

    # Option A: Remove TENSION (MI=0.0019), keep SMA_DEV (MI=0.0070)
    # Also remove BWP_BNW (MI=0, dependent on TENSION)
    configs = [
        # Remove nothing extra — just test subsets
        ('3F_NoTension', {'R_BWP_BNW': 0.25, 'R_R_SQUARED': 0.25,
                          'R_SMA_DEV': 0.25, 'R_HAR_RV_RATIO_INV': 0.25}),
        ('3F_NoSMADev', {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25,
                         'R_R_SQUARED': 0.25, 'R_HAR_RV_RATIO_INV': 0.25}),
        ('3F_NoBWP', {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                      'R_SMA_DEV': 0.25, 'R_HAR_RV_RATIO_INV': 0.25}),
        ('3F_NoHAR', {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25,
                      'R_R_SQUARED': 0.25, 'R_SMA_DEV': 0.25}),
        # Remove both BWP_BNW (MI=0) and HAR_RV (MI=0)
        ('2F_Core', {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                     'R_SMA_DEV': 0.25, 'R_HAR_RV_RATIO_INV': 0.25}),
        # Keep only top-2 MI factors: SMA_DEV + R_SQUARED
        ('2F_TopMI', {'R_SMA_DEV': 0.5, 'R_R_SQUARED': 0.5}),
        ('2F_TopMI+W', {'R_SMA_DEV': 0.4, 'R_R_SQUARED': 0.4, 'R_TENSION': 0.2}),
        # SMA_DEV + R_SQUARED + DMD (independent?)
        ('3F_SMA_R2_DMD', {'R_SMA_DEV': 0.35, 'R_R_SQUARED': 0.35, 'R_DMD_GROWTH': 0.3}),
        ('3F_SMA_R2_DMDv2', {'R_SMA_DEV': 0.3, 'R_R_SQUARED': 0.3, 'R_DMD_GROWTH': 0.2,
                              'R_TENSION': 0.1, 'R_HAR_RV_RATIO_INV': 0.1}),
        # All V41 + DMD various weights
        ('V41+DMD_0.05', {**v41_weights, 'R_DMD_GROWTH': 0.05}),
        ('V41+DMD_0.03', {**v41_weights, 'R_DMD_GROWTH': 0.03}),
        # Drop BWP_BNW and HAR_RV (both MI=0), replace with DMD
        ('4F_Ind+DMD', {'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                        'R_SMA_DEV': 0.3, 'R_DMD_GROWTH': 0.3}),
        ('4F_Ind+DMDv2', {'R_TENSION': 0.15, 'R_R_SQUARED': 0.2,
                          'R_SMA_DEV': 0.35, 'R_DMD_GROWTH': 0.15,
                          'R_HAR_RV_RATIO_INV': 0.15}),
    ]

    for name, weights in configs:
        total = sum(weights.values())
        w = {k: v / total for k, v in weights.items()}
        r = backtest_v7c(w, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = name
            results.append(r)

    print(f"  Configs done: {len(results)}", flush=True)

    # =====================================================================
    # TEST 2: ATR sweep on best configs
    # =====================================================================
    print("\n  Test 2: ATR sweep on promising configs...", flush=True)
    # Find top configs and test with different ATR
    promising = [r for r in results if r['ann'] > 100]
    for pr in promising[:5]:
        name = pr['test']
        # Find the weights for this config
        for cfg_name, weights in configs:
            if cfg_name == name:
                for atr in [0.6, 0.7, 0.9, 1.0]:
                    total = sum(weights.values())
                    w = {k: v / total for k, v in weights.items()}
                    r = backtest_v7c(w, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=1, rebalance_days=5, atr_stop_mult=atr)
                    if r:
                        r['test'] = f'{name}_A{atr}'
                        results.append(r)
                break

    # Also sweep ATR on V41 baseline
    for atr in [0.6, 0.7, 0.9, 1.0, 1.2]:
        r = backtest_v7c(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=atr)
        if r:
            r['test'] = f'V41_A{atr}'
            results.append(r)

    # =====================================================================
    # TEST 3: Different rebalance days
    # =====================================================================
    print("\n  Test 3: Rebalance sweep...", flush=True)
    for rebal in [3, 4, 6, 7]:
        r = backtest_v7c(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=rebal, atr_stop_mult=0.8)
        if r:
            r['test'] = f'V41_R{rebal}'
            results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V46 INDEPENDENT FACTOR SELECTION)", flush=True)
    print(f"  {'Test':<40s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*85}", flush=True)
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<40s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        baseline = next((r for r in results if r['test'] == 'V41_BASELINE'), None)
        print(f"\n  === V46 BEST vs V41 BASELINE ===", flush=True)
        print(f"  V46: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        if baseline:
            print(f"  V41: BASELINE = {baseline['ann']:+.1f}% DD={baseline['max_dd']:.1f}%", flush=True)
            delta = best['ann'] - baseline['ann']
            print(f"  Delta: {delta:+.1f}%", flush=True)
        print(f"  V41 RECORD: V15B_EQUAL_A0.8 = +342.0% DD=53.7%", flush=True)

    print(f"\n{'='*70}", flush=True)
