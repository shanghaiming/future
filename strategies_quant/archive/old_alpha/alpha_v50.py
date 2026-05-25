"""
Alpha V50 — Optimize V48 Breakthrough (BVR + ISKEW)
=====================================================
V48 found two breakthrough factors:
1. R_BVR: V41+BVR_W0.1 = +381.3% DD=46.6%
2. R_ISKEW: V41+ISKEW_W0.1 = +352.4% DD=34.4%

V50 systematically optimizes:
1. Weight sweep for BVR and ISKEW (fine-grained)
2. ATR sweep with new factors
3. Rebalance frequency sweep
4. Both BVR + ISKEW together
5. Replacing V41 factors with BVR/ISKEW
6. Novel 4-factor combos
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0
from alpha_v7c import backtest_v7c
from alpha_v44 import compute_v41_factors_only
from alpha_v48 import compute_v48_factors


def _rank_normalize(factor_2d, min_stocks=50):
    NS, ND = factor_2d.shape
    ranked = np.full_like(factor_2d, np.nan)
    for di in range(ND):
        vals = factor_2d[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < min_stocks:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        ranked[valid, di] = ranks / n * 100
    return ranked


def _rolling_mean(arr, window, min_valid=None):
    if min_valid is None:
        min_valid = window // 2
    NS, ND = arr.shape
    out = np.full_like(arr, np.nan)
    cumsum = np.where(np.isnan(arr), 0, arr)
    cumcount = (~np.isnan(arr)).astype(float)
    cs = np.cumsum(cumsum, axis=1)
    cc = np.cumsum(cumcount, axis=1)
    for di in range(window, ND):
        s = cs[:, di] - (cs[:, di - window] if di > window else 0)
        c = cc[:, di] - (cc[:, di - window] if di > window else 0)
        valid = c >= min_valid
        out[valid, di] = s[valid] / c[valid]
    return out


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V50 — Optimize V48 BVR + ISKEW Breakthrough", flush=True)
    print("  Target: beat V48 V41+BVR_W0.1 = +381.3%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute V41 factors
    print("\n  Computing V41 factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)

    # Compute V48 factors (only need BVR and ISKEW)
    print("\n  Computing V48 novel factors...", flush=True)
    v48_factors = compute_v48_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41_factors, **v48_factors}

    # Also add V49 factors
    from alpha_v49 import compute_v49_factors
    print("\n  Computing V49 factors...", flush=True)
    v49_factors = compute_v49_factors(NS, ND, C, O, H, L, V)
    all_factors.update(v49_factors)

    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    results = []

    # =====================================================================
    # Baseline
    # =====================================================================
    print("\n  Baseline...", flush=True)
    r = backtest_v7c(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.6)
    if r:
        r['test'] = 'V41_A0.6_BASE'
        results.append(r)
        print(f"  Baseline: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: Fine-grained BVR weight sweep
    # =====================================================================
    print("\n  Test 1: BVR weight sweep...", flush=True)
    for w in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]:
        weights = {**v41_weights, 'R_BVR': w}
        total = sum(weights.values())
        w_norm = {k: v / total for k, v in weights.items()}
        for atr in [0.5, 0.6, 0.7, 0.8]:
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'BVR_W{w}_A{atr}'
                results.append(r)
    print(f"  BVR sweep: {len(results)}", flush=True)

    # =====================================================================
    # Test 2: ISKEW weight sweep
    # =====================================================================
    print("\n  Test 2: ISKEW weight sweep...", flush=True)
    for w in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]:
        weights = {**v41_weights, 'R_ISKEW': w}
        total = sum(weights.values())
        w_norm = {k: v / total for k, v in weights.items()}
        for atr in [0.5, 0.6, 0.7, 0.8]:
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'ISKEW_W{w}_A{atr}'
                results.append(r)

    # =====================================================================
    # Test 3: BVR + ISKEW together
    # =====================================================================
    print("\n  Test 3: BVR + ISKEW together...", flush=True)
    for w_bvr in [0.05, 0.1, 0.15]:
        for w_iskew in [0.05, 0.1, 0.15]:
            weights = {**v41_weights, 'R_BVR': w_bvr, 'R_ISKEW': w_iskew}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            for atr in [0.5, 0.6, 0.7]:
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'BVR{w_bvr}+ISK{w_iskew}_A{atr}'
                    results.append(r)

    # =====================================================================
    # Test 4: Rebalance sweep with best new config
    # =====================================================================
    print("\n  Test 4: Rebalance sweep...", flush=True)
    # Use V41+BVR_W0.1 as base
    base_w = {**v41_weights, 'R_BVR': 0.1}
    total = sum(base_w.values())
    base_w = {k: v / total for k, v in base_w.items()}
    for rebal in [3, 4, 5, 6, 7, 8, 10]:
        r = backtest_v7c(base_w, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=rebal, atr_stop_mult=0.6)
        if r:
            r['test'] = f'BVR_W0.1_R{rebal}'
            results.append(r)

    # =====================================================================
    # Test 5: Replace V41 factors with BVR/ISKEW
    # =====================================================================
    print("\n  Test 5: Factor replacement...", flush=True)
    v41_list = ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV', 'R_HAR_RV_RATIO_INV']
    for old_f in v41_list:
        for new_f in ['R_BVR', 'R_ISKEW']:
            new_w = {k: v for k, v in v41_weights.items() if k != old_f}
            new_w[new_f] = 0.2
            r = backtest_v7c(new_w, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.6)
            if r:
                r['test'] = f'REP_{old_f[-4:]}→{new_f[-4:]}'
                results.append(r)
    # Replace 2 V41 factors with BVR + ISKEW
    for i in range(len(v41_list)):
        for j in range(i + 1, len(v41_list)):
            new_w = {k: v for k, v in v41_weights.items() if k != v41_list[i] and k != v41_list[j]}
            new_w['R_BVR'] = 0.1
            new_w['R_ISKEW'] = 0.1
            r = backtest_v7c(new_w, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.6)
            if r:
                r['test'] = f'DROP_{v41_list[i][-3:]}+{v41_list[j][-3:]}+BVR+ISK'
                results.append(r)

    # =====================================================================
    # Test 6: BVR + best V49 factors
    # =====================================================================
    print("\n  Test 6: BVR + V49 factors...", flush=True)
    v49_names = sorted(v49_factors.keys())
    for fname in v49_names:
        weights = {**v41_weights, 'R_BVR': 0.1, fname: 0.1}
        total = sum(weights.values())
        w_norm = {k: v / total for k, v in weights.items()}
        r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.6)
        if r:
            r['test'] = f'BVR+{fname[-4:]}'
            results.append(r)

    # =====================================================================
    # Test 7: BVR with modified BVR parameters
    # =====================================================================
    print("\n  Test 7: BVR parameter variations...", flush=True)
    # Different BVR lookback windows
    hl_range = np.full((NS, ND), np.nan)
    mask = ~np.isnan(H) & ~np.isnan(L)
    hl_range[mask] = H[mask] - L[mask]
    cl_diff = np.full((NS, ND), np.nan)
    mask = ~np.isnan(C) & ~np.isnan(L) & (hl_range > 1e-6)
    cl_diff[mask] = C[mask] - L[mask]
    buyer_vol = np.where(mask, V * cl_diff / hl_range, np.nan)

    for bv_window in [3, 5, 7, 10]:
        for vol_window in [10, 15, 20, 30]:
            bv_mean = _rolling_mean(buyer_vol, bv_window)
            v_mean = _rolling_mean(V, vol_window)
            bvr_var = np.full((NS, ND), np.nan)
            m = ~np.isnan(bv_mean) & ~np.isnan(v_mean) & (v_mean > 0)
            bvr_var[m] = bv_mean[m] / v_mean[m]
            bvr_ranked = _rank_normalize(bvr_var)
            all_factors[f'R_BVR_{bv_window}_{vol_window}'] = bvr_ranked

            weights = {**v41_weights, f'R_BVR_{bv_window}_{vol_window}': 0.1}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.6)
            if r:
                r['test'] = f'BVR_{bv_window}_{vol_window}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V50 OPTIMIZE BVR + ISKEW)", flush=True)
    print(f"  {'Test':<45s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*90}", flush=True)
    for r in results[:80]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<45s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
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
        print(f"\n  === V50 BEST ===", flush=True)
        print(f"  V50: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V48 RECORD: V41+BVR_W0.1 = +381.3%", flush=True)
        print(f"  V46 RECORD: V41_A0.6 = +344.6%", flush=True)
        delta = best['ann'] - 381.3
        print(f"  Delta from V48: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
