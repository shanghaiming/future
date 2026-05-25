"""
Alpha V41 — Factor Weight Optimization + Top_n Sweep
=====================================================
V35 found rebalance=5d as the key engine breakthrough (+290.4%).
V41 asks: can we squeeze more from the FACTOR SIDE?

Tests:
1. Fine-grained weight sweep on V15-B factors (BWP_BNW, TENSION, R_SQUARED, SMA_DEV, HAR_RV)
2. top_n sweep: 1 vs 2 (with 500K capital, 2 stocks = 250K each)
3. ATR stop sweep: 0.6, 0.8, 1.0, 1.2, 1.5
4. Holding period sweep with new top_n values

All using V35's rebalance=5d engine.
"""
import sys, os, time, warnings
import numpy as np
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


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V41 — Weight + Engine Parameter Sweep", flush=True)
    print("  Target: beat V35 R5_A1.0_B = +290.4%", flush=True)
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

    print(f"  Factors loaded: {len(all_factors)}", flush=True)

    results = []

    # =====================================================================
    # TEST 1: V15-B weight variations
    # =====================================================================
    print("\n  Test 1: V15-B weight sweep...", flush=True)
    weight_configs = [
        # name, weights
        ('V15B_ORIG', {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}),
        # Emphasize momentum more
        ('V15B_BWP35', {'R_BWP_BNW': 0.35, 'R_TENSION': 0.20, 'R_R_SQUARED': 0.15, 'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}),
        ('V15B_TENS35', {'R_BWP_BNW': 0.20, 'R_TENSION': 0.35, 'R_R_SQUARED': 0.15, 'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}),
        ('V15B_HAR35', {'R_BWP_BNW': 0.20, 'R_TENSION': 0.20, 'R_R_SQUARED': 0.15, 'R_SMA_DEV': 0.10, 'R_HAR_RV_RATIO_INV': 0.35}),
        ('V15B_RS35', {'R_BWP_BNW': 0.20, 'R_TENSION': 0.20, 'R_R_SQUARED': 0.35, 'R_SMA_DEV': 0.10, 'R_HAR_RV_RATIO_INV': 0.15}),
        ('V15B_SMA35', {'R_BWP_BNW': 0.20, 'R_TENSION': 0.20, 'R_R_SQUARED': 0.10, 'R_SMA_DEV': 0.35, 'R_HAR_RV_RATIO_INV': 0.15}),
        # 3-factor only (drop weakest)
        ('V15B_3F_BWP_TENS_HAR', {'R_BWP_BNW': 0.35, 'R_TENSION': 0.35, 'R_HAR_RV_RATIO_INV': 0.30}),
        ('V15B_3F_BWP_TENS_RS', {'R_BWP_BNW': 0.35, 'R_TENSION': 0.35, 'R_R_SQUARED': 0.30}),
        ('V15B_3F_BWP_HAR_SMA', {'R_BWP_BNW': 0.35, 'R_HAR_RV_RATIO_INV': 0.35, 'R_SMA_DEV': 0.30}),
        # Equal weights
        ('V15B_EQUAL', {'R_BWP_BNW': 0.20, 'R_TENSION': 0.20, 'R_R_SQUARED': 0.20, 'R_SMA_DEV': 0.20, 'R_HAR_RV_RATIO_INV': 0.20}),
        # 2-factor pairs
        ('V15B_BWP_TENS', {'R_BWP_BNW': 0.50, 'R_TENSION': 0.50}),
        ('V15B_BWP_HAR', {'R_BWP_BNW': 0.50, 'R_HAR_RV_RATIO_INV': 0.50}),
        ('V15B_TENS_HAR', {'R_TENSION': 0.50, 'R_HAR_RV_RATIO_INV': 0.50}),
        ('V15B_BWP_RS', {'R_BWP_BNW': 0.50, 'R_R_SQUARED': 0.50}),
        # Single factors
        ('V15B_BWP_ONLY', {'R_BWP_BNW': 1.0}),
        ('V15B_TENS_ONLY', {'R_TENSION': 1.0}),
        ('V15B_HAR_ONLY', {'R_HAR_RV_RATIO_INV': 1.0}),
        ('V15B_RS_ONLY', {'R_R_SQUARED': 1.0}),
        ('V15B_SMA_ONLY', {'R_SMA_DEV': 1.0}),
    ]

    for name, weights in weight_configs:
        for atr in [0.8, 1.0, 1.2]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{name}_A{atr}'
                results.append(r)
    print(f"  Weight sweep done: {len(results)} results", flush=True)

    # =====================================================================
    # TEST 2: top_n sweep (top 1 vs top 2)
    # =====================================================================
    print("\n  Test 2: top_n sweep...", flush=True)
    for top_n in [1, 2]:
        for atr in [0.8, 1.0, 1.2]:
            # V15-B original weights
            weights = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.2,
                       'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'TOPN{top_n}_A{atr}'
                results.append(r)

    # top_n=2 with different rebalance
    for rebal in [3, 5, 7, 10]:
        weights = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}
        r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=2, rebalance_days=rebal, atr_stop_mult=1.0)
        if r:
            r['test'] = f'TOPN2_R{rebal}_A1.0'
            results.append(r)
    print(f"  top_n sweep done: {len(results)} results", flush=True)

    # =====================================================================
    # TEST 3: Rebalance sweep (already found 5d optimal, but try 3d and 4d)
    # =====================================================================
    print("\n  Test 3: Fine rebalance sweep...", flush=True)
    for rebal in [3, 4, 5, 6, 7]:
        weights = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}
        for atr in [0.8, 1.0]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=rebal, atr_stop_mult=atr)
            if r:
                r['test'] = f'R{rebal}_A{atr}'
                results.append(r)
    print(f"  Rebalance sweep done: {len(results)} results", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V41 WEIGHT + ENGINE SWEEP)", flush=True)
    print(f"  {'Test':<35s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<35s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best by category
    for cat_name, prefix in [('Weight sweep', 'V15B_'), ('top_n', 'TOPN'), ('Rebalance', 'R')]:
        cat = [r for r in results if r['test'].startswith(prefix)]
        if cat:
            best = max(cat, key=lambda x: x['ann'])
            pos = " ALL+" if all_positive(best) else ""
            print(f"\n  Best {cat_name}: {best['test']} → {best['ann']:+.1f}%DD={best['max_dd']:.1f}%{pos}", flush=True)

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
        print(f"\n  === V41 BEST vs V35 RECORD ===", flush=True)
        print(f"  V41: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V35: R5_A1.0_B = +290.4% DD=43.7%", flush=True)
        delta = best['ann'] - 290.4
        print(f"  Delta: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
