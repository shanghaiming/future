"""
Alpha V12 — Weight Optimization + Industry Diversification
==========================================================
V11 showed diminishing returns from new factors (+248% plateau).
V12 shifts strategy: optimize WEIGHTS and add INDUSTRY DIVERSIFICATION.

Key insight from philosophy doc: "集中兵力" + "知己知彼"
- Still pick the best stock, but ensure it's from the RIGHT industry
- Optimize weights with grid search rather than manual guessing

Approach:
  1. Fine-grained weight grid search on top factor combos
  2. Industry-based diversification (top-1 from different sectors)
  3. Dynamic rebalance frequency based on volatility
  4. Combine best V9-V11 factors with optimized weights

LOOK-AHEAD SELF-CHECK:
  [x] No new factors — reuses existing V10/V11 factors
  [x] Only changes portfolio construction logic
"""
import sys, os, time, warnings, itertools
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0
from alpha_v7b import compute_interaction_factors
from alpha_v7d import compute_extra_factors
from alpha_v7e import compute_v7e_factors
from alpha_v7f import compute_advanced_interactions
from alpha_v8 import compute_v8_factors, compute_v8_interactions
from alpha_v9 import compute_v9_factors, compute_v9_interactions
from alpha_v10 import compute_v10_factors, compute_v10_interactions
from alpha_v11 import compute_v11_factors, compute_v11_interactions
from alpha_v7c import backtest_v7c


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V12 — Weight Optimization", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
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
    all_factors = {**v10_all, **v11_factors}
    v11_inter = compute_v11_interactions(all_factors, NS, ND)
    all_factors.update(v11_inter)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # === WEIGHT GRID SEARCH ===
    # Key factors from V8-V10 that drive alpha
    core_factors = ['R_BWP_BNW', 'R_BODY_NW', 'R_TENSION', 'R_R_SQUARED',
                    'R_SMA_DEV', 'R_SD_BNW', 'R_BB_SQUEEZE_INV',
                    'R_BB_WIDTH_PCT_INV', 'R_KER']

    # Grid: each weight from [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # Must sum to 1.0
    results = []
    tested = 0

    # Strategy: fix top factors, vary their relative weights
    # Pattern: vary the weight of the MAIN factor (BWP_BNW, SD_BNW, BODY_NW)
    # against the supporting factors

    main_factors = ['R_BWP_BNW', 'R_SD_BNW', 'R_BODY_NW', 'R_BB_SQUEEZE_INV']
    support_combos = [
        ['R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV'],
        ['R_TENSION', 'R_R_SQUARED', 'R_KER'],
        ['R_TENSION', 'R_R_SQUARED', 'R_BB_WIDTH_PCT_INV'],
        ['R_TENSION', 'R_KER', 'R_SMA_DEV'],
        ['R_BODY_NW', 'R_TENSION', 'R_R_SQUARED'],
    ]

    print(f"\n  === WEIGHT GRID SEARCH ===", flush=True)
    for main in main_factors:
        for supports in support_combos:
            # Skip if main is also in supports
            if main in supports:
                continue

            # Grid: main weight from 0.2 to 0.5
            for main_w in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                # Distribute remaining weight among supports
                remaining = 1.0 - main_w
                n_sup = len(supports)

                # Equal weights for supports
                sup_w = remaining / n_sup
                weights = {main: main_w}
                for s in supports:
                    weights[s] = round(sup_w, 3)

                # Test with different params
                for atr in [1.0, 1.2]:
                    for rebal in [7, 10]:
                        r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                        top_n=1, rebalance_days=rebal, atr_stop_mult=atr)
                        if r:
                            w_str = f"{main}={main_w:.2f}"
                            r.update({'portfolio': w_str, 'top_n': 1,
                                      'rebal': rebal, 'atr': atr,
                                      'weights': str(weights)})
                            results.append(r)
                            tested += 1

            print(f"  {main} × {len(support_combos)} support combos done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n  Tested {tested} combinations", flush=True)
    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 30 (V12 WEIGHT OPTIMIZED)", flush=True)
    print(f"  {'Portfolio':<30s} {'Reb':>3s} {'ATR':>3s} | "
          f"{'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for r in results[:30]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['portfolio']:<30s} {r['rebal']:3d} {r['atr']:3.1f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best per main factor
    best_per_main = {}
    for r in results:
        # Extract main factor from portfolio name
        parts = r['portfolio'].split('=')
        main = parts[0] if parts else ''
        if main not in best_per_main or r['ann'] > best_per_main[main]['ann']:
            best_per_main[main] = r

    print(f"\n  Best per main factor:", flush=True)
    for main, r in sorted(best_per_main.items(), key=lambda x: -x[1]['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {main:<30s} Reb={r['rebal']} ATR={r['atr']:.1f} → "
              f"{r['ann']:+.1f}%DD={r['max_dd']:.1f}%{pos}", flush=True)
        print(f"      Weights: {r.get('weights', '')}", flush=True)

    # Top 3 year-by-year
    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['portfolio']} Reb={r['rebal']} "
              f"ATR={r['atr']:.1f} (Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
