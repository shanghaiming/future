"""
Alpha V21 — Factor Independence Analysis via Kendall τ
=======================================================
From probability_theory.md Section 10 (Copula):

"策略组合的关键不是哪些好，而是哪些独立"
"独立信号一致时的贝叶斯后验概率大幅提升"

This script:
1. Loads all factors from V7-V14
2. Computes Kendall τ correlation matrix between all factors
3. Identifies the most INDEPENDENT factor pairs (τ closest to 0)
4. Tests only independent factor combinations
5. Records findings to learning notes

NO LOOK-AHEAD: Uses only factor values computed from di-1 data.
"""
import sys, os, time, warnings
import numpy as np
from scipy.stats import kendalltau, spearmanr
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors
from alpha_v7b import compute_interaction_factors
from alpha_v7d import compute_extra_factors
from alpha_v7e import compute_v7e_factors
from alpha_v7f import compute_advanced_interactions
from alpha_v8 import compute_v8_factors, compute_v8_interactions
from alpha_v9 import compute_v9_factors, compute_v9_interactions
from alpha_v10 import compute_v10_factors, compute_v10_interactions
from alpha_v11 import compute_v11_factors, compute_v11_interactions
from alpha_v14 import compute_v14_factors, compute_v14_interactions
from alpha_v7c import backtest_v7c


def compute_tau_matrix(factors, NS, ND, factor_names, sample_di=1500):
    """Compute Kendall τ matrix between factors at a specific day.

    Uses cross-sectional data (all stocks at one point in time).
    """
    n = len(factor_names)
    tau_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))

    # Get factor values at sample_di
    values = {}
    for fname in factor_names:
        if fname in factors:
            arr = factors[fname][:, sample_di]
            values[fname] = arr
        else:
            values[fname] = np.full(NS, np.nan)

    for i in range(n):
        for j in range(i + 1, n):
            vi = values[factor_names[i]]
            vj = values[factor_names[j]]
            mask = ~np.isnan(vi) & ~np.isnan(vj)
            if mask.sum() < 50:
                tau_matrix[i, j] = np.nan
                tau_matrix[j, i] = np.nan
                continue
            tau, p = kendalltau(vi[mask], vj[mask])
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    np.fill_diagonal(tau_matrix, 1.0)
    return tau_matrix, p_matrix


def find_most_independent_pairs(tau_matrix, factor_names, top_k=20):
    """Find factor pairs with lowest |τ| (most independent)."""
    n = len(factor_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(tau_matrix[i, j]):
                continue
            pairs.append((abs(tau_matrix[i, j]), tau_matrix[i, j],
                          factor_names[i], factor_names[j]))
    pairs.sort(key=lambda x: x[0])
    return pairs[:top_k]


def find_most_correlated_pairs(tau_matrix, factor_names, top_k=20):
    """Find factor pairs with highest |τ| (most correlated = redundant)."""
    n = len(factor_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(tau_matrix[i, j]):
                continue
            pairs.append((abs(tau_matrix[i, j]), tau_matrix[i, j],
                          factor_names[i], factor_names[j]))
    pairs.sort(key=lambda x: -x[0])
    return pairs[:top_k]


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V21 — Factor Independence Analysis (Kendall τ)", flush=True)
    print("  From probability_theory.md Section 10 (Copula)", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load all factors
    base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
    inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)
    extra_factors = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e_factors = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv_inter = compute_advanced_interactions(
        {**base_factors, **inter_factors, **extra_factors, **v7e_factors}, NS, ND)
    v8_factors = compute_v8_factors(NS, ND, C, O, H, L, V)
    v8_all = {**base_factors, **inter_factors, **extra_factors,
              **v7e_factors, **adv_inter, **v8_factors}
    v8_inter = compute_v8_interactions(v8_all, NS, ND)
    v8_all.update(v8_inter)
    v9_factors = compute_v9_factors(NS, ND, C, O, H, L, V)
    v9_all = {**v8_all, **v9_factors}
    v9_inter = compute_v9_interactions(v9_all, NS, ND)
    v9_all.update(v9_inter)
    v10_factors = compute_v10_factors(NS, ND, C, O, H, L, V)
    v10_all = {**v9_all, **v10_factors}
    v10_inter = compute_v10_interactions(v10_all, NS, ND)
    v10_all.update(v10_inter)
    v11_factors = compute_v11_factors(NS, ND, C, O, H, L, V)
    v11_all = {**v10_all, **v11_factors}
    v11_inter = compute_v11_interactions(v11_all, NS, ND)
    v11_all.update(v11_inter)
    v14_factors = compute_v14_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **v14_factors}
    v14_inter = compute_v14_interactions(all_factors, NS, ND)
    all_factors.update(v14_inter)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # =====================================================================
    # SELECT KEY FACTORS FOR ANALYSIS
    # =====================================================================
    # Use only R_ (ranked) factors — these are what we combine
    key_factors = [
        # V7 base
        'R_MOM5', 'R_MOM10', 'R_MOM20', 'R_KINETIC', 'R_BODY_RATIO',
        'R_SHADOW_PRESSURE', 'R_VDP', 'R_FISHER', 'R_DRAWDOWN_52W',
        'R_LINREG_SLOPE', 'R_R_SQUARED', 'R_BREAKOUT', 'R_VOL_ANOMALY',
        'R_VOLATILITY_PCT', 'R_TENSION', 'R_PRICE_PCT',
        # V7b interactions
        'R_DD_VOL', 'R_COND_MOM5', 'R_BODY_VOL', 'R_TENS_SHAD',
        # V7d/e
        'R_SMA_DEV', 'R_ATR_RATIO', 'R_NW_SLOPE', 'R_KINETIC_EMA', 'R_VOL_ACCEL',
        # V8
        'R_ENTROPY', 'R_KALMAN_SLOPE', 'R_OFI', 'R_VOL_DELTA',
        # V9
        'R_HURST', 'R_KFD', 'R_BB_SQUEEZE_INV', 'R_SUPPLY_DIST_INV',
        # V10
        'R_SQZ_DEPTH', 'R_SQZ_DURATION', 'R_RELEASE_MOM',
        'R_BB_WIDTH_PCT_INV',
        # V10 interactions (the winners)
        'R_BWP_BNW', 'R_SD_BNW', 'R_SDU_BNW',
        # V11
        'R_RSI', 'R_MACD_HIST', 'R_KER', 'R_REL_STR',
        # V14
        'R_HAR_RV_RATIO_INV', 'R_LOG_PRESSURE', 'R_ATR_TERRAIN',
    ]

    # Filter to only existing factors
    key_factors = [f for f in key_factors if f in all_factors]
    print(f"\n  Analyzing {len(key_factors)} factors", flush=True)

    # =====================================================================
    # COMPUTE KENDALL τ MATRIX
    # =====================================================================
    print(f"\n  Computing Kendall τ matrix...", flush=True)
    t0 = time.time()

    # Sample at multiple time points for robustness
    sample_points = [1000, 1500, 2000, ND - 100]
    all_tau = []

    for sp in sample_points:
        if sp >= ND:
            continue
        tau_mat, p_mat = compute_tau_matrix(all_factors, NS, ND, key_factors, sp)
        all_tau.append(tau_mat)
        print(f"  τ at di={sp} done ({time.time()-t0:.1f}s)", flush=True)

    # Average τ across time points
    avg_tau = np.nanmean(all_tau, axis=0)

    # =====================================================================
    # FIND MOST INDEPENDENT PAIRS
    # =====================================================================
    independent_pairs = find_most_independent_pairs(avg_tau, key_factors, top_k=30)

    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 30 MOST INDEPENDENT FACTOR PAIRS (lowest |τ|)", flush=True)
    print(f"  {'Factor A':<25s} {'Factor B':<25s} {'τ':>7s} {'|τ|':>7s}", flush=True)
    print(f"  {'-'*70}", flush=True)
    for abs_t, t, fa, fb in independent_pairs:
        print(f"  {fa:<25s} {fb:<25s} {t:+7.3f} {abs_t:7.3f}", flush=True)

    # =====================================================================
    # FIND MOST CORRELATED PAIRS (redundant)
    # =====================================================================
    correlated_pairs = find_most_correlated_pairs(avg_tau, key_factors, top_k=20)

    print(f"\n  TOP 20 MOST CORRELATED PAIRS (redundant)", flush=True)
    print(f"  {'Factor A':<25s} {'Factor B':<25s} {'τ':>7s} {'|τ|':>7s}", flush=True)
    print(f"  {'-'*70}", flush=True)
    for abs_t, t, fa, fb in correlated_pairs:
        print(f"  {fa:<25s} {fb:<25s} {t:+7.3f} {abs_t:7.3f}", flush=True)

    # =====================================================================
    # FIND INDEPENDENT FACTOR CLUSTERS
    # =====================================================================
    # Greedy: start with best factor, add next factor that's most independent
    # from all already selected
    print(f"\n  === INDEPENDENT FACTOR CLUSTERS ===", flush=True)

    # Cluster 1: Start from BWP_BNW (the best factor)
    selected = ['R_BWP_BNW']
    remaining = [f for f in key_factors if f != 'R_BWP_BNW']

    for _ in range(4):  # Add 4 more factors
        best_avg_tau = 1.0
        best_factor = None
        idx_selected = [key_factors.index(s) for s in selected]

        for f in remaining:
            idx_f = key_factors.index(f)
            taus = [abs(avg_tau[idx_f, idx_s]) for idx_s in idx_selected]
            avg = np.mean(taus)
            if avg < best_avg_tau:
                best_avg_tau = avg
                best_factor = f

        if best_factor:
            selected.append(best_factor)
            remaining.remove(best_factor)

    print(f"  Cluster from R_BWP_BNW:", flush=True)
    for f in selected:
        idx = key_factors.index(f)
        print(f"    {f:<25s} (avg |τ| to others: {np.mean([abs(avg_tau[idx, key_factors.index(s)]) for s in selected if s != f]):.3f})", flush=True)

    # =====================================================================
    # TEST INDEPENDENT COMBINATIONS
    # =====================================================================
    print(f"\n  === TESTING INDEPENDENT COMBINATIONS ===", flush=True)

    # Test the independent cluster
    weights = {f: 1.0 / len(selected) for f in selected}
    results = []
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'Indep_T{top_n}_A{atr}'
                results.append(r)

    # Also test all pairs of top independent factors
    top_indep = [p[2] for p in independent_pairs[:10]] + [p[3] for p in independent_pairs[:10]]
    top_indep = list(set(top_indep))[:15]  # Unique factors

    for i in range(min(10, len(top_indep))):
        for j in range(i + 1, min(10, len(top_indep))):
            fa, fb = top_indep[i], top_indep[j]
            # Add TENSION and R_SQUARED as base
            weights = {fa: 0.25, fb: 0.25, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.25}
            for top_n in [1]:
                for atr in [1.2]:
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                    if r:
                        r['test'] = f'τ_{fa}_{fb}_T{top_n}'
                        results.append(r)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  TOP 20 INDEPENDENT COMBO RESULTS:", flush=True)
    print(f"  {'Test':<50s} | {'Ann':>7s} {'WR':>5s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:20]:
        print(f"  {r['test']:<50s} | {r['ann']:+7.1f}% {r['wr']:5.1f}% {r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # FULL τ MATRIX HEATMAP (text-based)
    # =====================================================================
    # Print τ matrix for core factors
    core = ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV',
            'R_MOM5', 'R_KER', 'R_HURST', 'R_HAR_RV_RATIO_INV',
            'R_LOG_PRESSURE', 'R_ATR_TERRAIN', 'R_BB_WIDTH_PCT_INV',
            'R_BODY_NW']
    core = [f for f in core if f in key_factors]
    core_idx = [key_factors.index(f) for f in core]

    print(f"\n  === CORE FACTOR τ MATRIX ===", flush=True)
    header = f"  {'':>20s}"
    for f in core:
        header += f" {f[2:6]:>6s}"
    print(header, flush=True)
    for i, fi in enumerate(core):
        row = f"  {fi:>20s}"
        for j, fj in enumerate(core):
            t = avg_tau[core_idx[i], core_idx[j]]
            row += f" {t:+6.3f}"
        print(row, flush=True)

    print(f"\n{'='*70}", flush=True)
