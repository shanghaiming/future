"""
Alpha V34 — Epanechnikov Soft AND Gate (Single Variable Test)
==============================================================
V14 tested Epanechnikov scoring but changed 5 things at once → failed.
V34 isolates ONLY the scoring method: Epanechnikov kernel vs linear combination.

Key idea: K(u) = 0.75 × (1 - u²) where u = (100 - rank) / 50
- Stocks with rank > 50 get positive weight (K > 0)
- Stocks with rank = 100 get max weight (K = 0.75)
- Stocks with rank ≤ 50 get ZERO weight (K ≤ 0) → soft AND gate

This means ALL factors must agree (rank > 50) for a stock to be selected.
Compare to linear combination: a stock with rank 100 on factor A but rank 0 on
factor B gets score = 0.5×100 + 0.5×0 = 50 → might still be selected!

With Epanechnikov: K(100)×K(0) = 0.75×(-0.75) = negative → filtered out.

Test phases:
  1. Baseline: Linear combination (should match V15 ~235.6%)
  2. Epanechnikov: Same factors, Epanechnikov scoring
  3. Factor sweep: Epanechnikov with different factor combinations
  4. Best combos with ATR/top_n sweep

NO LOOK-AHEAD: All factors use d = di - 1. Epanechnikov is just a transform.
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


def epanechnikov_score(factor_names, weights, all_factors, NS, ND):
    """Compute Epanechnikov soft AND gate score.

    K(u) = 0.75 × (1 - u²) where u = (100 - rank) / 50
    Score = product of K(rank_i) for all factors
    """
    score = np.ones((NS, ND))
    for fname, w in factor_names.items():
        if fname not in all_factors:
            return None
        f = all_factors[fname]
        # Epanechnikov kernel
        u = (100.0 - f) / 50.0  # u ∈ [-1, 1] when rank ∈ [0, 100]
        k = 0.75 * (1.0 - u * u)  # K(u) = 0.75(1-u²), peaks at u=0 (rank=100)
        k = np.where(np.isnan(f), np.nan, k)
        k = np.maximum(k, 0.0)  # Clip negatives → soft AND gate

        # Apply weight as exponent (geometric weighting)
        k_weighted = np.power(np.maximum(k, 1e-10), w)
        score = score * k_weighted

    return score


def backtest_epan(factor_names, all_factors, NS, ND, dates, C, O, H, L, V,
                  top_n=1, rebalance_days=10, atr_stop_mult=1.0):
    """Backtest using Epanechnikov scoring instead of linear combination."""
    score = epanechnikov_score(factor_names, {}, all_factors, NS, ND)
    if score is None:
        return None

    # Use score as if it were a single "factor" — but we need backtest_v7c format
    # Create a temporary factor and call backtest_v7c with just that factor
    all_factors_temp = dict(all_factors)
    all_factors_temp['_EPAN_SCORE'] = score

    # Rank normalize the Epanechnikov score
    def rank_pct(arr, start=60):
        res = np.full_like(arr, np.nan)
        for di in range(start, arr.shape[1]):
            vals = arr[:, di]
            mask = ~np.isnan(vals) & (vals > 0)
            if mask.sum() < 50:
                continue
            ranked = np.argsort(np.argsort(vals[mask])).astype(float)
            n = len(ranked)
            pct = ranked / max(n - 1, 1) * 100
            for k, idx in enumerate(np.where(mask)[0]):
                res[idx, di] = pct[k]
        return res

    all_factors_temp['R_EPAN_SCORE'] = rank_pct(score)

    return backtest_v7c({'R_EPAN_SCORE': 1.0}, all_factors_temp, NS, ND, dates,
                       C, O, H, L, V, top_n=top_n,
                       rebalance_days=rebalance_days, atr_stop_mult=atr_stop_mult)


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V34 — Epanechnikov Soft AND Gate", flush=True)
    print("  Single variable test: Epanechnikov vs Linear scoring", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute all factors
    base = compute_all_factors(NS, ND, C, O, H, L, V)
    inter = compute_interaction_factors(base, NS, ND, C, O, H, L, V)
    extra = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv = compute_advanced_interactions({**base, **inter, **extra, **v7e}, NS, ND)
    v8f = compute_v8_factors(NS, ND, C, O, H, L, V)
    all8 = {**base, **inter, **extra, **v7e, **adv, **v8f}
    all8.update(compute_v8_interactions(all8, NS, ND))
    v9f = compute_v9_factors(NS, ND, C, O, H, L, V)
    all9 = {**all8, **v9f}
    all9.update(compute_v9_interactions(all9, NS, ND))
    v10f = compute_v10_factors(NS, ND, C, O, H, L, V)
    all10 = {**all9, **v10f}
    all10.update(compute_v10_interactions(all10, NS, ND))
    v11f = compute_v11_factors(NS, ND, C, O, H, L, V)
    all11 = {**all10, **v11f}
    all11.update(compute_v11_interactions(all11, NS, ND))
    v14f = compute_v14_factors(NS, ND, C, O, H, L, V)
    all_f = {**all11, **v14f}
    all_f.update(compute_v14_interactions(all_f, NS, ND))

    print(f"\n  Total factors: {len(all_f)}", flush=True)

    results = []

    # =================================================================
    # PHASE 1: Baseline (Linear) — should match V15
    # =================================================================
    print(f"\n  === PHASE 1: BASELINE (LINEAR) ===", flush=True)

    har_combo = {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    bwp_base = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}

    for tn in [1]:
        for atr in [1.0]:
            r = backtest_v7c(har_combo, all_f, NS, ND, dates, C, O, H, L, V,
                            top_n=tn, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'LINEAR_HAR_T{tn}_A{atr}'
                results.append(r)
                print(f"  {r['test']}: Ann={r['ann']:+.1f}% WR={r['wr']:.1f}% "
                      f"DD={r['max_dd']:.1f}%", flush=True)

    # =================================================================
    # PHASE 2: Epanechnikov — same factors, different scoring
    # =================================================================
    print(f"\n  === PHASE 2: EPANECHNIKOV SCORING ===", flush=True)

    epan_combos = {
        # V15 winner factors with Epanechnikov
        'EP_HAR': {'R_BWP_BNW': 1.0, 'R_HAR_RV_RATIO_INV': 1.0,
                   'R_R_SQUARED': 1.0, 'R_SMA_DEV': 1.0},
        # BWP_BNW + TENSION (core 2 factors)
        'EP_BWP_TENS': {'R_BWP_BNW': 1.0, 'R_TENSION': 1.0},
        # BWP_BNW + HAR_RV (2 independent dimensions)
        'EP_BWP_HAR': {'R_BWP_BNW': 1.0, 'R_HAR_RV_RATIO_INV': 1.0},
        # 4D independent (from V21 τ analysis)
        'EP_4D': {'R_BWP_BNW': 1.0, 'R_HAR_RV_RATIO_INV': 1.0,
                  'R_ATR_TERRAIN': 1.0, 'R_LOG_PRESSURE': 1.0},
        # BWP_BNW + ATR_TERRAIN (most independent pair)
        'EP_BWP_ATR': {'R_BWP_BNW': 1.0, 'R_ATR_TERRAIN': 1.0},
        # 3D: BWP + HAR + ATR
        'EP_3D_VOL': {'R_BWP_BNW': 1.0, 'R_HAR_RV_RATIO_INV': 1.0,
                      'R_ATR_TERRAIN': 1.0},
        # Momentum confirmation gate
        'EP_BWP_MOM': {'R_BWP_BNW': 1.0, 'R_MOM5': 1.0},
        # Squeeze confirmation gate
        'EP_BWP_SQZ': {'R_BWP_BNW': 1.0, 'R_BB_WIDTH_PCT_INV': 1.0},
        # Pure HAR_RV
        'EP_HAR_ONLY': {'R_HAR_RV_RATIO_INV': 1.0},
        # Pure BWP_BNW
        'EP_BWP_ONLY': {'R_BWP_BNW': 1.0},
    }

    for pname, factors in epan_combos.items():
        for tn in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_epan(factors, all_f, NS, ND, dates,
                                 C, O, H, L, V, top_n=tn,
                                 rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{tn}_A{atr}'
                    results.append(r)
                    print(f"  {r['test']:<25s}: Ann={r['ann']:+7.1f}% N={r['n']:4d} "
                          f"WR={r['wr']:5.1f}% Edge={r['edge']:+5.2f}% "
                          f"DD={r['max_dd']:5.1f}%", flush=True)
        print(f"  {pname} done", flush=True)

    # =================================================================
    # RESULTS
    # =================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V34 EPANECHNIKOV SOFT AND GATE)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:30]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Top 3 year-by-year
    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # Best per group
    groups = {}
    for r in results:
        prefix = r['test'].split('_T')[0]
        if prefix not in groups or r['ann'] > groups[prefix]['ann']:
            groups[prefix] = r
    print(f"\n  Best per group:", flush=True)
    for r in sorted(groups.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['test']:<30s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    print(f"\n{'='*70}", flush=True)
