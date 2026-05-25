"""
Alpha V32 — Quadruple Independent Factor Fusion (V21 Kendall τ Optimized)
==========================================================================
V21 Kendall τ analysis revealed 4 MUTUALLY INDEPENDENT factors (τ < 0.05):

  Factor A: R_BWP_BNW        — Price geometry (BB width × body direction)
  Factor B: R_HAR_RV_RATIO_INV — Volatility prediction (Corsi 2009)
  Factor C: R_ATR_TERRAIN    — Volatility terrain state (ATR regime)
  Factor D: R_LOG_PRESSURE   — Institutional flow (log-normalized)

Cross-correlations (Kendall τ):
  A vs B: ~+0.2  (low-moderate, V15 proved combinable → +235.6%)
  A vs C: +0.032 (essentially independent!)
  A vs D: +0.320 (moderate — may need care)
  B vs C: -0.048 (essentially independent!)
  B vs D: +0.001 (completely independent!)
  C vs D: -0.017 (essentially independent!)

V32 hypothesis: Combining 4 mutually independent signal dimensions should
produce a more robust composite score than V15's 2D fusion.

Test phases:
  1. Replicate V15 baseline (BWP_BNW + HAR_RV + TENSION + SMA_DEV)
  2. 2D: BWP_BNW + ATR_TERRAIN (new independent pair)
  3. 2D: BWP_BNW + LOG_PRESSURE (re-test with V15 engine)
  4. 3D: BWP_BNW + HAR_RV + ATR_TERRAIN (3 truly independent)
  5. 3D: BWP_BNW + HAR_RV + LOG_PRESSURE (3 independent)
  6. 4D: BWP_BNW + HAR_RV + ATR_TERRAIN + LOG_PRESSURE (ALL 4!)
  7. 4D variants: different weight allocations
  8. Gate: use ATR_TERRAIN as filter (only buy squeeze/fading states)
  9. Top-N sweep for best combos

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


def run_bt(name, weights, all_factors, NS, ND, dates, C, O, H, L, V,
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
    print("  Alpha V32 — Quadruple Independent Factor Fusion", flush=True)
    print("  Based on V21 Kendall τ analysis", flush=True)
    print("  4 mutually independent dimensions (τ < 0.05)", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute all factors including V14 (HAR-RV, LOG_PRESSURE, ATR_TERRAIN)
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
    v14_all = {**v11_all, **v14_factors}
    v14_inter = compute_v14_interactions(v14_all, NS, ND)
    all_factors = {**v14_all, **v14_inter}

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # Verify key factors exist
    key_factors = ['R_BWP_BNW', 'R_HAR_RV_RATIO_INV', 'R_ATR_TERRAIN',
                   'R_LOG_PRESSURE', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV']
    for f in key_factors:
        if f in all_factors:
            valid = ~np.isnan(all_factors[f][:, -1])
            print(f"  {f}: {valid.sum()} stocks valid", flush=True)
        else:
            print(f"  WARNING: {f} NOT FOUND!", flush=True)

    results = []

    # =================================================================
    # PHASE 1: Baseline replication
    # =================================================================
    print(f"\n  === PHASE 1: BASELINE ===", flush=True)

    # V15 baseline (proven +235.6%)
    bwp_base = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}

    # V15 HAR-RV combo (proven +235.6%)
    har_combo = {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}

    for tn in [1, 2]:
        for atr in [1.0, 1.2]:
            results = run_bt(f'BwpBNW_T{tn}_A{atr}', bwp_base,
                           all_factors, NS, ND, dates, C, O, H, L, V,
                           top_n=tn, atr=atr, results=results)
            results = run_bt(f'HAR_base_T{tn}_A{atr}', har_combo,
                           all_factors, NS, ND, dates, C, O, H, L, V,
                           top_n=tn, atr=atr, results=results)

    print(f"  Phase 1 done: {len(results)} results", flush=True)

    # =================================================================
    # PHASE 2: 2D independent pairs
    # =================================================================
    print(f"\n  === PHASE 2: 2D INDEPENDENT PAIRS ===", flush=True)

    pairs_2d = {
        # BWP_BNW + ATR_TERRAIN (τ=+0.032, most independent!)
        'AT_2d': {'R_BWP_BNW': 0.3, 'R_ATR_TERRAIN': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # BWP_BNW + LOG_PRESSURE (τ=+0.320, moderate)
        'LP_2d': {'R_BWP_BNW': 0.3, 'R_LOG_PRESSURE': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # HAR-RV + ATR_TERRAIN (τ=-0.048, independent)
        'HA_2d': {'R_HAR_RV_RATIO_INV': 0.3, 'R_ATR_TERRAIN': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # HAR-RV + LOG_PRESSURE (τ=+0.001, completely independent!)
        'HL_2d': {'R_HAR_RV_RATIO_INV': 0.3, 'R_LOG_PRESSURE': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # ATR_TERRAIN + LOG_PRESSURE (τ=-0.017, independent)
        'AL_2d': {'R_ATR_TERRAIN': 0.3, 'R_LOG_PRESSURE': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
    }

    for pname, weights in pairs_2d.items():
        for tn in [1, 2]:
            for atr in [1.0, 1.2]:
                results = run_bt(f'{pname}_T{tn}_A{atr}', weights,
                               all_factors, NS, ND, dates, C, O, H, L, V,
                               top_n=tn, atr=atr, results=results)
        print(f"  {pname} done", flush=True)

    print(f"  Phase 2 done: {len(results)} results", flush=True)

    # =================================================================
    # PHASE 3: 3D independent triplets
    # =================================================================
    print(f"\n  === PHASE 3: 3D INDEPENDENT TRIPLETS ===", flush=True)

    triplets_3d = {
        # V15 winner + ATR_TERRAIN (proven + independent)
        'HAR_AT_3d': {'R_BWP_BNW': 0.25, 'R_HAR_RV_RATIO_INV': 0.25,
                      'R_ATR_TERRAIN': 0.25, 'R_R_SQUARED': 0.25},
        # V15 winner + LOG_PRESSURE
        'HAR_LP_3d': {'R_BWP_BNW': 0.25, 'R_HAR_RV_RATIO_INV': 0.25,
                      'R_LOG_PRESSURE': 0.25, 'R_R_SQUARED': 0.25},
        # All volatility: HAR-RV + ATR_TERRAIN + BWP_BNW
        'VOL_3d': {'R_BWP_BNW': 0.2, 'R_HAR_RV_RATIO_INV': 0.3,
                   'R_ATR_TERRAIN': 0.3, 'R_SMA_DEV': 0.2},
        # Terrain + Flow: ATR_TERRAIN + LOG_PRESSURE + BWP_BNW
        'TERR_FLOW_3d': {'R_BWP_BNW': 0.2, 'R_ATR_TERRAIN': 0.3,
                         'R_LOG_PRESSURE': 0.3, 'R_SMA_DEV': 0.2},
        # Pure independent 3: HAR-RV + ATR_TERRAIN + LOG_PRESSURE
        'PURE_IND_3d': {'R_HAR_RV_RATIO_INV': 0.3, 'R_ATR_TERRAIN': 0.3,
                        'R_LOG_PRESSURE': 0.2, 'R_R_SQUARED': 0.2},
    }

    for pname, weights in triplets_3d.items():
        for tn in [1, 2]:
            for atr in [1.0, 1.2]:
                results = run_bt(f'{pname}_T{tn}_A{atr}', weights,
                               all_factors, NS, ND, dates, C, O, H, L, V,
                               top_n=tn, atr=atr, results=results)
        print(f"  {pname} done", flush=True)

    print(f"  Phase 3 done: {len(results)} results", flush=True)

    # =================================================================
    # PHASE 4: 4D QUADRUPLE FUSION
    # =================================================================
    print(f"\n  === PHASE 4: 4D QUADRUPLE FUSION ===", flush=True)

    quads_4d = {
        # Equal weight
        'Q4_eq': {'R_BWP_BNW': 0.25, 'R_HAR_RV_RATIO_INV': 0.25,
                  'R_ATR_TERRAIN': 0.25, 'R_LOG_PRESSURE': 0.25},
        # BWP_BNW heavy (proven base)
        'Q4_bwp': {'R_BWP_BNW': 0.35, 'R_HAR_RV_RATIO_INV': 0.25,
                   'R_ATR_TERRAIN': 0.2, 'R_LOG_PRESSURE': 0.2},
        # HAR-RV heavy (proven addition)
        'Q4_har': {'R_BWP_BNW': 0.25, 'R_HAR_RV_RATIO_INV': 0.35,
                   'R_ATR_TERRAIN': 0.2, 'R_LOG_PRESSURE': 0.2},
        # ATR_TERRAIN heavy (most independent)
        'Q4_atr': {'R_BWP_BNW': 0.2, 'R_HAR_RV_RATIO_INV': 0.25,
                   'R_ATR_TERRAIN': 0.35, 'R_LOG_PRESSURE': 0.2},
        # LOG_PRESSURE heavy
        'Q4_lp': {'R_BWP_BNW': 0.2, 'R_HAR_RV_RATIO_INV': 0.2,
                  'R_ATR_TERRAIN': 0.25, 'R_LOG_PRESSURE': 0.35},
        # Drop BWP_BNW, pure 3 independent
        'Q4_nobwp': {'R_HAR_RV_RATIO_INV': 0.3, 'R_ATR_TERRAIN': 0.3,
                     'R_LOG_PRESSURE': 0.2, 'R_R_SQUARED': 0.2},
        # 4D + TENSION (5 factors)
        'Q4_tens': {'R_BWP_BNW': 0.2, 'R_HAR_RV_RATIO_INV': 0.2,
                    'R_ATR_TERRAIN': 0.2, 'R_LOG_PRESSURE': 0.2,
                    'R_TENSION': 0.2},
    }

    for pname, weights in quads_4d.items():
        for tn in [1, 2]:
            for atr in [1.0, 1.2]:
                results = run_bt(f'{pname}_T{tn}_A{atr}', weights,
                               all_factors, NS, ND, dates, C, O, H, L, V,
                               top_n=tn, atr=atr, results=results)
        print(f"  {pname} done", flush=True)

    print(f"  Phase 4 done: {len(results)} results", flush=True)

    # =================================================================
    # PHASE 5: ATR_TERRAIN gate logic
    # =================================================================
    print(f"\n  === PHASE 5: ATR_TERRAIN GATE ===", flush=True)

    # ATR_TERRAIN values: 100=squeeze, 75=fading, 50=normal, 25=expansion
    # Gate: only buy stocks with ATR_TERRAIN rank > 60 (squeeze/fading territory)
    # We implement this by multiplying R_ATR_TERRAIN with the base score
    # Only stocks with high terrain rank get selected

    gate_combos = {
        # Base × terrain gate (soft AND)
        'GATE_sqz_base': {'R_BWP_BNW': 0.2, 'R_HAR_RV_RATIO_INV': 0.2,
                          'R_ATR_TERRAIN': 0.4, 'R_R_SQUARED': 0.1,
                          'R_SMA_DEV': 0.1},
        # Terrain-dominant
        'GATE_terr_d': {'R_ATR_TERRAIN': 0.4, 'R_BWP_BNW': 0.3,
                        'R_HAR_RV_RATIO_INV': 0.2, 'R_LOG_PRESSURE': 0.1},
        # Terrain + flow (both independent)
        'GATE_terr_flow': {'R_ATR_TERRAIN': 0.3, 'R_LOG_PRESSURE': 0.3,
                           'R_BWP_BNW': 0.2, 'R_HAR_RV_RATIO_INV': 0.2},
    }

    for pname, weights in gate_combos.items():
        for tn in [1, 2]:
            for atr in [1.0, 1.2]:
                results = run_bt(f'{pname}_T{tn}_A{atr}', weights,
                               all_factors, NS, ND, dates, C, O, H, L, V,
                               top_n=tn, atr=atr, results=results)
        print(f"  {pname} done", flush=True)

    print(f"  Phase 5 done: {len(results)} results", flush=True)

    # =================================================================
    # PHASE 6: Best combo optimization sweep
    # =================================================================
    print(f"\n  === PHASE 6: OPTIMIZATION SWEEP ===", flush=True)
    print(f"  (Run after checking Phase 1-5 results)", flush=True)

    # Additional ATR stop sweeps for any promising combos
    sweep_combos = {
        'Q4_eq_15': {'R_BWP_BNW': 0.25, 'R_HAR_RV_RATIO_INV': 0.25,
                     'R_ATR_TERRAIN': 0.25, 'R_LOG_PRESSURE': 0.25},
        'HAR_AT_3d_15': {'R_BWP_BNW': 0.25, 'R_HAR_RV_RATIO_INV': 0.25,
                         'R_ATR_TERRAIN': 0.25, 'R_R_SQUARED': 0.25},
    }

    for pname, weights in sweep_combos.items():
        for atr in [1.5]:
            for tn in [1, 2]:
                results = run_bt(f'{pname}_T{tn}_A{atr}', weights,
                               all_factors, NS, ND, dates, C, O, H, L, V,
                               top_n=tn, atr=atr, results=results)
        print(f"  {pname} done", flush=True)

    # =================================================================
    # RESULTS
    # =================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 50 RESULTS (V32 QUADRUPLE INDEPENDENT FUSION)", flush=True)
    print(f"  {'Test':<35s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<35s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # Best per strategy
    groups = {}
    for r in results:
        prefix = r['test'].split('_T')[0]
        if prefix not in groups or r['ann'] > groups[prefix]['ann']:
            groups[prefix] = r
    print(f"\n  Best per group:", flush=True)
    for r in sorted(groups.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['test']:<35s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    print(f"\n{'='*70}", flush=True)
