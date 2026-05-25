"""
Alpha V53 — Optimize BUY_FRAC/OFI Breakthrough toward 1000%
=============================================================
V52 found: REP_ARED→R_BUY_FRAC = +932.7% DD=27.1% ALL+
V52 found: REP_ARED→R_OFI = +891.3% DD=20.3% ALL+

V53 systematically optimizes:
1. BUY_FRAC weight fine sweep (0.02-0.30)
2. OFI weight fine sweep
3. ATR fine sweep (0.3-0.6) with best configs
4. BUY_FRAC replacing each V51 factor (not just SMA_DEV)
5. BUY_FRAC + OFI together (both strong microstructure)
6. VWCM + BUY_FRAC weight interaction
7. Rebalance sweep (3-7 days)
8. Non-equal weight optimization
9. Top 3 factor combos (drop weakest V41 factors)
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
from alpha_v52 import compute_v52_factors


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


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V53 — Optimize BUY_FRAC/OFI toward 1000%")
    print("  V52 best: REP_ARED→R_BUY_FRAC = +932.7% DD=27.1%", flush=True)
    print("  Target: break 1000%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing V41 factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)

    print("\n  Computing V48 + V49 factors...", flush=True)
    v48_factors = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49_factors = compute_v49_factors(NS, ND, C, O, H, L, V)

    print("\n  Computing V52 factors...", flush=True)
    v52_factors = compute_v52_factors(NS, ND, C, O, H, L, V)

    all_factors = {**v41_factors, **v48_factors, **v49_factors, **v52_factors}

    # V51 winning base (without SMA_DEV)
    v51_base = {'R_BWP_BNW': 0.178, 'R_TENSION': 0.178, 'R_R_SQUARED': 0.178,
                'R_VWCM': 0.178, 'R_BVR': 0.134}

    v51_full = {'R_BWP_BNW': 0.178, 'R_TENSION': 0.178, 'R_R_SQUARED': 0.178,
                'R_SMA_DEV': 0.178, 'R_VWCM': 0.178, 'R_BVR': 0.134}

    results = []

    # =====================================================================
    # Baselines
    # =====================================================================
    print("\n  Baselines...", flush=True)
    r = backtest_v7c(v51_full, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V51_FULL_A0.5'
        results.append(r)
        print(f"  V51 full: {r['ann']:+.1f}%", flush=True)

    # V52 best: SMA_DEV→BUY_FRAC w=0.15
    w52_best = {**v51_base, 'R_BUY_FRAC': 0.15}
    total = sum(w52_best.values())
    w52_norm = {k: v / total for k, v in w52_best.items()}
    r = backtest_v7c(w52_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V52_BEST_CHECK'
        results.append(r)
        print(f"  V52 best check: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: BUY_FRAC weight fine sweep
    # =====================================================================
    print("\n  Test 1: BUY_FRAC weight sweep...", flush=True)
    for w_bf in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        for atr in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            weights = {**v51_base, 'R_BUY_FRAC': w_bf}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'BF{w_bf}_A{atr}'
                results.append(r)
    print(f"  BF sweep: {len(results)}", flush=True)

    # =====================================================================
    # Test 2: OFI weight fine sweep
    # =====================================================================
    print("\n  Test 2: OFI weight sweep...", flush=True)
    for w_ofi in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        for atr in [0.3, 0.35, 0.4, 0.45, 0.5]:
            weights = {**v51_base, 'R_OFI': w_ofi}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'OFI{w_ofi}_A{atr}'
                results.append(r)

    # =====================================================================
    # Test 3: BUY_FRAC + OFI together
    # =====================================================================
    print("\n  Test 3: BUY_FRAC + OFI combo...", flush=True)
    for w_bf in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for w_ofi in [0.05, 0.08, 0.10, 0.12]:
            for atr in [0.35, 0.4, 0.45, 0.5]:
                weights = {**v51_base, 'R_BUY_FRAC': w_bf, 'R_OFI': w_ofi}
                total = sum(weights.values())
                w_norm = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'BF{w_bf}+OFI{w_ofi}_A{atr}'
                    results.append(r)
    print(f"  BF+OFI: {len(results)}", flush=True)

    # =====================================================================
    # Test 4: Replace each V51 factor with BUY_FRAC
    # =====================================================================
    print("\n  Test 4: Replace each V51 factor...", flush=True)
    for old_f in ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_VWCM', 'R_BVR']:
        for w_bf in [0.10, 0.15, 0.20]:
            weights = {k: v for k, v in v51_full.items() if k != old_f}
            weights['R_BUY_FRAC'] = w_bf
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'REP_{old_f[-4:]}→BF{w_bf}'
                results.append(r)

    # =====================================================================
    # Test 5: Non-equal weights with BUY_FRAC
    # =====================================================================
    print("\n  Test 5: Non-equal weights...", flush=True)
    weight_configs = [
        ('VWCM_HIGH', {'R_BWP_BNW': 0.15, 'R_TENSION': 0.15, 'R_R_SQUARED': 0.15,
                       'R_VWCM': 0.25, 'R_BVR': 0.10, 'R_BUY_FRAC': 0.20}),
        ('BF_HIGH', {'R_BWP_BNW': 0.12, 'R_TENSION': 0.12, 'R_R_SQUARED': 0.15,
                     'R_VWCM': 0.18, 'R_BVR': 0.10, 'R_BUY_FRAC': 0.33}),
        ('BF_VWCM', {'R_BWP_BNW': 0.10, 'R_TENSION': 0.10, 'R_R_SQUARED': 0.15,
                     'R_VWCM': 0.25, 'R_BVR': 0.10, 'R_BUY_FRAC': 0.30}),
        ('BF_BVR', {'R_BWP_BNW': 0.12, 'R_TENSION': 0.12, 'R_R_SQUARED': 0.15,
                    'R_VWCM': 0.20, 'R_BVR': 0.15, 'R_BUY_FRAC': 0.26}),
        ('TENS_BF', {'R_BWP_BNW': 0.10, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.12,
                     'R_VWCM': 0.18, 'R_BVR': 0.10, 'R_BUY_FRAC': 0.25}),
        ('RSQ_BF', {'R_BWP_BNW': 0.10, 'R_TENSION': 0.12, 'R_R_SQUARED': 0.25,
                    'R_VWCM': 0.18, 'R_BVR': 0.10, 'R_BUY_FRAC': 0.25}),
        ('EQ_ALL', {'R_BWP_BNW': 0.167, 'R_TENSION': 0.167, 'R_R_SQUARED': 0.167,
                    'R_VWCM': 0.167, 'R_BVR': 0.167, 'R_BUY_FRAC': 0.167}),
    ]
    for name, weights in weight_configs:
        for atr in [0.35, 0.4, 0.45, 0.5]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{name}_A{atr}'
                results.append(r)

    # =====================================================================
    # Test 6: Rebalance sweep with best config
    # =====================================================================
    print("\n  Test 6: Rebalance sweep...", flush=True)
    for rebal in [3, 4, 5, 6, 7]:
        # BUY_FRAC = 0.15
        weights = {**v51_base, 'R_BUY_FRAC': 0.15}
        total = sum(weights.values())
        w_norm = {k: v / total for k, v in weights.items()}
        for atr in [0.35, 0.4, 0.45, 0.5]:
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=rebal, atr_stop_mult=atr)
            if r:
                r['test'] = f'BF0.15_R{rebal}_A{atr}'
                results.append(r)

    # =====================================================================
    # Test 7: VPIN + DEPTH_IMB variants (also strong in V52)
    # =====================================================================
    print("\n  Test 7: Other micro combos...", flush=True)
    # BUY_FRAC + VPIN
    for w_bf in [0.10, 0.12, 0.15]:
        for w_vp in [0.05, 0.08, 0.10]:
            weights = {**v51_base, 'R_BUY_FRAC': w_bf, 'R_VPIN': w_vp}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'BF{w_bf}+VP{w_vp}'
                results.append(r)

    # BUY_FRAC + DEPTH_IMB
    for w_bf in [0.10, 0.12, 0.15]:
        for w_di in [0.05, 0.08, 0.10]:
            weights = {**v51_base, 'R_BUY_FRAC': w_bf, 'R_DEPTH_IMB': w_di}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'BF{w_bf}+DI{w_di}'
                results.append(r)

    # =====================================================================
    # Test 8: Drop weakest V41 factor, add BUY_FRAC + OFI
    # =====================================================================
    print("\n  Test 8: 7-factor combos...", flush=True)
    # Full V51 + BUY_FRAC + OFI
    weights = {**v51_full, 'R_BUY_FRAC': 0.10, 'R_OFI': 0.08}
    total = sum(weights.values())
    w_norm = {k: v / total for k, v in weights.items()}
    for atr in [0.35, 0.4, 0.45, 0.5]:
        r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=atr)
        if r:
            r['test'] = f'V51+BF+OFI_A{atr}'
            results.append(r)

    # =====================================================================
    # Test 9: BUY_FRAC window parameter variations
    # =====================================================================
    print("\n  Test 9: BUY_FRAC parameter sweep...", flush=True)
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[mask, di] = (C[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    from alpha_v52 import _rolling_mean as _rm
    buy_tick = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        buy_tick[mask, di] = (C[mask, di] > C[mask, di - 1]).astype(float)

    for window in [5, 10, 15, 20, 30, 40, 60]:
        bf_var = _rm(buy_tick, window)
        bf_ranked = _rank_normalize(bf_var)
        all_factors[f'R_BUY_FRAC_{window}'] = bf_ranked

        for w in [0.10, 0.15, 0.20]:
            weights = {**v51_base, f'R_BUY_FRAC_{window}': w}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'BF_W{window}_{w}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V53 BUY_FRAC/OFI OPTIMIZATION)", flush=True)
    print(f"  {'Test':<45s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*90}", flush=True)
    for r in results[:120]:
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
        print(f"\n  === V53 BEST ===", flush=True)
        print(f"  V53: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V52 RECORD: REP_ARED→R_BUY_FRAC = +932.7% DD=27.1%", flush=True)
        print(f"  V51 RECORD: VWCM0.2+BVR0.15_A0.5 = +620.8% DD=25.1%", flush=True)
        delta = best['ann'] - 932.7
        print(f"  Delta from V52: {delta:+.1f}%", flush=True)
        print(f"  Target: 1000%", flush=True)

    print(f"\n{'='*70}", flush=True)
