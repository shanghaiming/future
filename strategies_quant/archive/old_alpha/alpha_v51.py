"""
Alpha V51 — Final Optimization of VWCM + BVR Breakthrough
===========================================================
V49 found: REP_HAR→VWCM = +544.6% DD=29.1% ALL+
V48 found: V41+BVR_W0.1 = +381.3% DD=46.6%
V50 found: BVR+VWCM (additive) = +493.7% DD=41.4%

V51 tests the UNTSTED optimal combination:
- Replace HAR_RV_RATIO_INV with VWCM (V49 winner)
- Add BVR on top (V48 winner)
- VWCM weight sweep (0.15, 0.2, 0.25, 0.3)
- BVR weight sweep (0.05, 0.1, 0.15)
- ATR sweep (0.5-0.8)
- Rebalance sweep (3-8)
- VWCM parameter variations (lookback windows)
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
from alpha_v49 import compute_v49_factors


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
    print("  Alpha V51 — Final VWCM + BVR Optimization")
    print("  Target: beat V49 REP_HAR→VWCM = +544.6%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute V41 factors
    print("\n  Computing V41 factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)

    # Compute V48 + V49 factors
    print("\n  Computing V48 + V49 factors...", flush=True)
    v48_factors = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49_factors = compute_v49_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41_factors, **v48_factors, **v49_factors}

    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    # Base config: V41 with HAR_RV replaced by VWCM
    v49_winner = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                  'R_SMA_DEV': 0.2, 'R_VWCM': 0.2}

    results = []

    # =====================================================================
    # Baselines
    # =====================================================================
    print("\n  Baselines...", flush=True)
    r = backtest_v7c(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.6)
    if r:
        r['test'] = 'V41_A0.6'
        results.append(r)
        print(f"  V41: {r['ann']:+.1f}%", flush=True)

    r = backtest_v7c(v49_winner, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.6)
    if r:
        r['test'] = 'V49_WINNER'
        results.append(r)
        print(f"  V49 winner: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: VWCM weight sweep (replace HAR_RV, different weights)
    # =====================================================================
    print("\n  Test 1: VWCM weight sweep...", flush=True)
    for w_vwcm in [0.15, 0.2, 0.25, 0.3, 0.35]:
        weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_VWCM': w_vwcm}
        total = sum(weights.values())
        w_norm = {k: v / total for k, v in weights.items()}
        for atr in [0.5, 0.6, 0.7, 0.8]:
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'VWCM_{w_vwcm}_A{atr}'
                results.append(r)
    print(f"  VWCM sweep: {len(results)}", flush=True)

    # =====================================================================
    # Test 2: VWCM + BVR (replace HAR_RV, add BVR)
    # =====================================================================
    print("\n  Test 2: VWCM + BVR...", flush=True)
    for w_vwcm in [0.15, 0.2, 0.25]:
        for w_bvr in [0.05, 0.1, 0.15]:
            weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                       'R_SMA_DEV': 0.2, 'R_VWCM': w_vwcm, 'R_BVR': w_bvr}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            for atr in [0.5, 0.6, 0.7]:
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'VWCM{w_vwcm}+BVR{w_bvr}_A{atr}'
                    results.append(r)
    print(f"  VWCM+BVR: {len(results)}", flush=True)

    # =====================================================================
    # Test 3: VWCM + ISKEW (replace HAR_RV, add ISKEW)
    # =====================================================================
    print("\n  Test 3: VWCM + ISKEW...", flush=True)
    for w_vwcm in [0.15, 0.2, 0.25]:
        for w_iskew in [0.05, 0.1, 0.15]:
            weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                       'R_SMA_DEV': 0.2, 'R_VWCM': w_vwcm, 'R_ISKEW': w_iskew}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            for atr in [0.5, 0.6, 0.7]:
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'VWCM{w_vwcm}+ISK{w_iskew}_A{atr}'
                    results.append(r)

    # =====================================================================
    # Test 4: VWCM + BVR + ISKEW
    # =====================================================================
    print("\n  Test 4: VWCM + BVR + ISKEW...", flush=True)
    for w_vwcm in [0.2, 0.25]:
        for w_bvr in [0.05, 0.1]:
            for w_iskew in [0.05, 0.1]:
                weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                           'R_SMA_DEV': 0.2, 'R_VWCM': w_vwcm, 'R_BVR': w_bvr, 'R_ISKEW': w_iskew}
                total = sum(weights.values())
                w_norm = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.6)
                if r:
                    r['test'] = f'VWCM{w_vwcm}+BVR{w_bvr}+ISK{w_iskew}'
                    results.append(r)

    # =====================================================================
    # Test 5: Rebalance sweep with V49 winner
    # =====================================================================
    print("\n  Test 5: Rebalance sweep...", flush=True)
    for rebal in [3, 4, 5, 6, 7, 8, 10]:
        r = backtest_v7c(v49_winner, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=rebal, atr_stop_mult=0.6)
        if r:
            r['test'] = f'V49WINNER_R{rebal}'
            results.append(r)

    # =====================================================================
    # Test 6: VWCM parameter variations
    # =====================================================================
    print("\n  Test 6: VWCM parameter sweep...", flush=True)
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[mask, di] = (C[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    v_signed = np.full((NS, ND), np.nan)
    mask = ~np.isnan(V) & ~np.isnan(C) & ~np.isnan(ret) & (V > 0)
    v_signed[mask] = V[mask] * np.sign(ret[mask]) * C[mask]
    v_signed_mean = _rolling_mean(v_signed, 10)
    v_mean = _rolling_mean(V * C, 10)

    for vsigned_window in [5, 7, 10, 15, 20]:
        vsigned_m = _rolling_mean(v_signed, vsigned_window)
        vmean_m = _rolling_mean(V * C, vsigned_window)
        vwcm_var = np.full((NS, ND), np.nan)
        m = ~np.isnan(vsigned_m) & ~np.isnan(vmean_m) & (vmean_m > 0)
        vwcm_var[m] = vsigned_m[m] / vmean_m[m]
        vwcm_ranked = _rank_normalize(vwcm_var)
        all_factors[f'R_VWCM_{vsigned_window}'] = vwcm_ranked

        weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, f'R_VWCM_{vsigned_window}': 0.2}
        r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.6)
        if r:
            r['test'] = f'VWCM_W{vsigned_window}'
            results.append(r)

    # =====================================================================
    # Test 7: Non-equal weights with VWCM
    # =====================================================================
    print("\n  Test 7: Non-equal weights...", flush=True)
    weight_configs = [
        ('VWCM_HIGH', {'R_BWP_BNW': 0.15, 'R_TENSION': 0.15, 'R_R_SQUARED': 0.2,
                       'R_SMA_DEV': 0.2, 'R_VWCM': 0.3}),
        ('VWCM_VHIGH', {'R_BWP_BNW': 0.1, 'R_TENSION': 0.1, 'R_R_SQUARED': 0.2,
                        'R_SMA_DEV': 0.2, 'R_VWCM': 0.4}),
        ('SMA_VWCM', {'R_BWP_BNW': 0.15, 'R_TENSION': 0.15, 'R_R_SQUARED': 0.15,
                      'R_SMA_DEV': 0.25, 'R_VWCM': 0.3}),
        ('TENS_VWCM', {'R_BWP_BNW': 0.1, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.15,
                       'R_SMA_DEV': 0.2, 'R_VWCM': 0.3}),
    ]
    for name, weights in weight_configs:
        for atr in [0.5, 0.6, 0.7]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{name}_A{atr}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V51 VWCM + BVR OPTIMIZATION)", flush=True)
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
        print(f"\n  === V51 BEST ===", flush=True)
        print(f"  V51: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V49 RECORD: REP_HAR→VWCM = +544.6% DD=29.1%", flush=True)
        print(f"  V48 RECORD: V41+BVR_W0.1 = +381.3%", flush=True)
        delta = best['ann'] - 544.6
        print(f"  Delta from V49: {delta:+.1f}%", flush=True)
        print(f"  Target: 600%", flush=True)

    print(f"\n{'='*70}", flush=True)
