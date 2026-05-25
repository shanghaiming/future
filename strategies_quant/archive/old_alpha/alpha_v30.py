"""
Alpha V30 — Multi-Dimensional Factor Fusion
=============================================
From factor independence analysis, we identified 6 independent signal dimensions.
Each dimension has one "most independent" factor:

  1. Volatility:    R_HAR_RV_RATIO_INV (Corsi 2009 realized variance prediction)
  2. Volume:        R_LOG_PRESSURE (log-normalized institutional flow)
  3. Price Structure: R_TENSION (multi-point displacement)
  4. Efficiency:    R_KER (Kaufman efficiency ratio)
  5. Mean Reversion: R_FISHER (Gaussian price transformation)
  6. Path Complexity: R_KFD (Katz fractal dimension)

PLUS the proven winners:
  - R_BWP_BNW (BB_WIDTH × BODY — V10 best)
  - R_R_SQUARED (linear regression fit)
  - R_SMA_DEV (SMA deviation)

V30 tests:
  Phase 1: Each independent factor alone
  Phase 2: Each independent factor + proven winners
  Phase 3: 2-dimensional fusion (proven + one new)
  Phase 4: 3-dimensional fusion (proven + two new)
  Phase 5: Best V15 combo + gate logic from V29

NO LOOK-AHEAD: All factors use d = di - 1.
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


def run_backtest(name, weights, all_factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, atr=1.0, results=None):
    """Run a single backtest and append result."""
    missing = [f for f in weights if f not in all_factors]
    if missing:
        print(f"  SKIP {name}: missing {missing}", flush=True)
        return results or []
    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
    if r:
        r['test'] = name
        if results is None:
            results = []
        results.append(r)
    return results or []


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V30 — Multi-Dimensional Factor Fusion", flush=True)
    print("  Independent signal dimensions → systematic combination", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load all factors up to V14
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
    v14_inter = compute_v14_interactions({**v11_all, **v14_factors}, NS, ND)
    all_factors = {**v11_all, **v14_factors, **v14_inter}

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    results = []

    # =====================================================================
    # BASELINE: BwpBNW (proven best)
    # =====================================================================
    print(f"\n  === BASELINE ===", flush=True)
    bwp = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
            'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    for tn in [1]:
        for a in [1.0, 1.2]:
            results = run_backtest(f'BwpBNW_T{tn}_A{a}', bwp, all_factors,
                                  NS, ND, dates, C, O, H, L, V, tn, a, results)
    print(f"  Baseline done", flush=True)

    # V15 winner: HAR-RV + BwpBNW
    har_bwp = {'R_HAR_RV_RATIO_INV': 0.3, 'R_BWP_BNW': 0.3,
               'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    results = run_backtest('HAR_BWP_T1_A1.0', har_bwp, all_factors,
                          NS, ND, dates, C, O, H, L, V, 1, 1.0, results)
    print(f"  V15 reproduction done", flush=True)

    # =====================================================================
    # PHASE 1: Independent factors alone (top=3 for single-factor tests)
    # =====================================================================
    print(f"\n  === PHASE 1: INDEPENDENT FACTOR SINGLE TESTS ===", flush=True)
    independent_factors = [
        'R_HAR_RV_RATIO_INV', 'R_LOG_PRESSURE', 'R_TENSION',
        'R_KER', 'R_FISHER', 'R_KFD',
        'R_HURST', 'R_KINETIC', 'R_REL_STRENGTH',
        'R_NW_SLOPE', 'R_BB_WIDTH_PCT_INV', 'R_SQUEEZE_DEPTH',
    ]
    for fname in independent_factors:
        results = run_backtest(f'{fname}_T3', {fname: 1.0}, all_factors,
                              NS, ND, dates, C, O, H, L, V, 3, 1.5, results)
        if results and results[-1]['test'] == f'{fname}_T3':
            r = results[-1]
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # PHASE 2: Independent factor + proven winners (2D fusion)
    # =====================================================================
    print(f"\n  === PHASE 2: 2D FUSION (independent + BwpBNW) ===", flush=True)
    proven_core = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    # Replace one weight with independent factor
    for fname in independent_factors:
        # Replace R_R_SQUARED (weakest in baseline) with new factor
        weights = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                   fname: 0.2, 'R_SMA_DEV': 0.2}
        for a in [1.0, 1.2]:
            results = run_backtest(f'2D_{fname.split("R_")[-1]}_A{a}',
                                  weights, all_factors,
                                  NS, ND, dates, C, O, H, L, V, 1, a, results)
        print(f"  2D {fname} done", flush=True)

    # =====================================================================
    # PHASE 3: HAR-RV (proven winner) + each independent factor (3D fusion)
    # =====================================================================
    print(f"\n  === PHASE 3: 3D FUSION (HAR-RV + independent + BwpBNW) ===", flush=True)
    for fname in ['R_LOG_PRESSURE', 'R_KER', 'R_FISHER', 'R_KFD',
                  'R_HURST', 'R_REL_STRENGTH', 'R_NW_SLOPE',
                  'R_BB_WIDTH_PCT_INV', 'R_SQUEEZE_DEPTH', 'R_KINETIC']:
        weights = {'R_HAR_RV_RATIO_INV': 0.25, 'R_BWP_BNW': 0.25,
                   fname: 0.25, 'R_TENSION': 0.25}
        for a in [1.0, 1.2]:
            results = run_backtest(f'3D_{fname.split("R_")[-1]}_A{a}',
                                  weights, all_factors,
                                  NS, ND, dates, C, O, H, L, V, 1, a, results)
        print(f"  3D {fname} done", flush=True)

    # =====================================================================
    # PHASE 4: 4D fusion (HAR-RV + two independent + BwpBNW)
    # =====================================================================
    print(f"\n  === PHASE 4: 4D FUSION ===", flush=True)
    four_d_combos = [
        ('HAR_KER_LP', {'R_HAR_RV_RATIO_INV': 0.2, 'R_KER': 0.2,
                        'R_LOG_PRESSURE': 0.2, 'R_BWP_BNW': 0.2, 'R_TENSION': 0.2}),
        ('HAR_KER_FIS', {'R_HAR_RV_RATIO_INV': 0.2, 'R_KER': 0.2,
                         'R_FISHER': 0.2, 'R_BWP_BNW': 0.2, 'R_TENSION': 0.2}),
        ('HAR_LP_FIS', {'R_HAR_RV_RATIO_INV': 0.2, 'R_LOG_PRESSURE': 0.2,
                        'R_FISHER': 0.2, 'R_BWP_BNW': 0.2, 'R_TENSION': 0.2}),
        ('HAR_KER_HUR', {'R_HAR_RV_RATIO_INV': 0.2, 'R_KER': 0.2,
                         'R_HURST': 0.2, 'R_BWP_BNW': 0.2, 'R_TENSION': 0.2}),
        ('HAR_LP_HUR', {'R_HAR_RV_RATIO_INV': 0.2, 'R_LOG_PRESSURE': 0.2,
                        'R_HURST': 0.2, 'R_BWP_BNW': 0.2, 'R_TENSION': 0.2}),
        ('HAR_KER_KFD', {'R_HAR_RV_RATIO_INV': 0.2, 'R_KER': 0.2,
                         'R_KFD': 0.2, 'R_BWP_BNW': 0.2, 'R_TENSION': 0.2}),
        ('HAR_BBP_SQZ', {'R_HAR_RV_RATIO_INV': 0.2, 'R_BB_WIDTH_PCT_INV': 0.2,
                         'R_SQUEEZE_DEPTH': 0.2, 'R_BWP_BNW': 0.2, 'R_TENSION': 0.2}),
        ('HAR_KER_REL', {'R_HAR_RV_RATIO_INV': 0.2, 'R_KER': 0.2,
                         'R_REL_STRENGTH': 0.2, 'R_BWP_BNW': 0.2, 'R_TENSION': 0.2}),
    ]
    for cname, weights in four_d_combos:
        for a in [1.0, 1.2]:
            results = run_backtest(f'4D_{cname}_A{a}', weights, all_factors,
                                  NS, ND, dates, C, O, H, L, V, 1, a, results)
        print(f"  4D {cname} done", flush=True)

    # =====================================================================
    # PHASE 5: Top=2 tests for best combos
    # =====================================================================
    print(f"\n  === PHASE 5: TOP=2 TESTS ===", flush=True)
    # Pick the best weights from Phase 3-4 based on early results
    top2_combos = {
        'T2_BwpBNW': bwp,
        'T2_HAR_BWP': har_bwp,
    }
    for cname, weights in top2_combos.items():
        results = run_backtest(f'{cname}_T2_A1.0', weights, all_factors,
                              NS, ND, dates, C, O, H, L, V, 2, 1.0, results)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 50 RESULTS (V30 MULTI-DIMENSIONAL FUSION)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # Best per phase
    groups = {}
    for r in results:
        if r['test'].startswith('BwpBNW') or r['test'].startswith('HAR_BWP'):
            prefix = 'Baseline'
        elif r['test'].startswith('2D_'):
            prefix = 'Phase2'
        elif r['test'].startswith('3D_'):
            prefix = 'Phase3'
        elif r['test'].startswith('4D_'):
            prefix = 'Phase4'
        else:
            prefix = 'Phase1'
        if prefix not in groups or r['ann'] > groups[prefix]['ann']:
            groups[prefix] = r
    print(f"\n  Best per phase:", flush=True)
    for phase in ['Baseline', 'Phase1', 'Phase2', 'Phase3', 'Phase4']:
        if phase in groups:
            r = groups[phase]
            pos = " ALL+" if all_positive(r) else ""
            print(f"    {phase:<12s}: {r['test']:<30s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    print(f"\n{'='*70}", flush=True)
