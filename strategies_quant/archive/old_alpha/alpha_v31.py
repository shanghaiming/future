"""
Alpha V31 — HAR-RV Core Optimization (Building on V15 Breakthrough)
====================================================================
V15 discovered: HAR_RV_T1_A1.0 = +235.6% DD=32.4% ALL+ (new record!)

V31 goes deeper:
  1. HAR-RV factor parameter sweep (window, lag combinations)
  2. HAR-RV + BwpBNW fusion at different weights
  3. HAR-RV + market gates (from V29 logic)
  4. HAR-RV + confirmation signals (from V30 logic)
  5. HAR-RV + momentum exit
  6. HAR-RV + adaptive rebalancing

Target: find if HAR-RV + gating can reach +300%+ or DD < 25%.

LOOK-AHEAD SELF-CHECK:
  [x] All factors at di use data up to d=di-1
  [x] ATR stop: BUG-FIXED (L[si,di] check, stop price sell)
  [x] HAR-RV OLS uses only past data
"""
import sys, os, time, warnings
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


def compute_har_variants(NS, ND, C, O, H, L, V):
    """Compute multiple HAR-RV variants with different parameters.

    SELF-CHECK: d = di - 1 for all data access.
    """
    t0 = time.time()
    new = {}

    # Base RV computation
    for si in range(NS):
        rv1 = np.full(ND, np.nan)
        for di in range(2, ND):
            d = di - 1
            c0, c1 = C[si, d - 1], C[si, d]
            if np.isnan(c0) or np.isnan(c1) or c0 <= 0:
                continue
            ret = np.log(c1 / c0)
            rv1[di] = ret ** 2

        # Variant 1: Standard HAR-RV (60-day OLS)
        har_ratio = np.full(ND, np.nan)
        for di in range(66, ND):
            # Build OLS from d=di-60 to d=di-1
            n_obs = min(60, di - 6)
            if n_obs < 30:
                continue
            y_data = []
            x_data = []
            for t in range(di - n_obs, di):
                rv_d = rv1[t]
                w_start = max(t - 5, 0)
                w_vals = rv1[w_start:t]
                rv_w = np.nanmean(w_vals) if np.sum(~np.isnan(w_vals)) >= 3 else np.nan
                m_start = max(t - 22, 0)
                m_vals = rv1[m_start:t]
                rv_m = np.nanmean(m_vals) if np.sum(~np.isnan(m_vals)) >= 10 else np.nan
                rv_next = rv1[t + 1] if t + 1 < ND else np.nan

                if np.isnan(rv_d) or np.isnan(rv_w) or np.isnan(rv_m) or np.isnan(rv_next):
                    continue
                y_data.append(rv_next)
                x_data.append([1.0, rv_d, rv_w, rv_m])

            if len(y_data) < 20:
                continue
            X = np.array(x_data)
            y = np.array(y_data)
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                # Predict RV for di using data up to di-1
                rv_d = rv1[di - 1] if not np.isnan(rv1[di - 1]) else 0
                w_vals = rv1[max(di - 6, 0):di]
                rv_w = np.nanmean(w_vals) if np.sum(~np.isnan(w_vals)) >= 3 else rv_d
                m_vals = rv1[max(di - 23, 0):di]
                rv_m = np.nanmean(m_vals) if np.sum(~np.isnan(m_vals)) >= 10 else rv_w

                if rv_d > 1e-12:
                    pred = beta[0] + beta[1] * rv_d + beta[2] * rv_w + beta[3] * rv_m
                    har_ratio[si, di] = pred / rv_d
            except Exception:
                continue

        new.setdefault('HAR_RATIO_60', np.full((NS, ND), np.nan))
        new['HAR_RATIO_60'][si] = har_ratio

    # Variant 2: Short-window HAR (30-day OLS)
    har_ratio_30 = np.full((NS, ND), np.nan)
    for si in range(NS):
        rv1 = np.full(ND, np.nan)
        for di in range(2, ND):
            d = di - 1
            c0, c1 = C[si, d - 1], C[si, d]
            if np.isnan(c0) or np.isnan(c1) or c0 <= 0:
                continue
            rv1[di] = np.log(c1 / c0) ** 2

        for di in range(36, ND):
            n_obs = min(30, di - 6)
            if n_obs < 15:
                continue
            y_data = []
            x_data = []
            for t in range(di - n_obs, di):
                rv_d = rv1[t]
                w_vals = rv1[max(t - 5, 0):t]
                rv_w = np.nanmean(w_vals) if np.sum(~np.isnan(w_vals)) >= 2 else np.nan
                m_vals = rv1[max(t - 10, 0):t]
                rv_m = np.nanmean(m_vals) if np.sum(~np.isnan(m_vals)) >= 5 else np.nan
                rv_next = rv1[t + 1] if t + 1 < ND else np.nan
                if np.isnan(rv_d) or np.isnan(rv_w) or np.isnan(rv_m) or np.isnan(rv_next):
                    continue
                y_data.append(rv_next)
                x_data.append([1.0, rv_d, rv_w, rv_m])

            if len(y_data) < 10:
                continue
            X = np.array(x_data)
            y = np.array(y_data)
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                rv_d = rv1[di - 1] if not np.isnan(rv1[di - 1]) else 0
                w_vals = rv1[max(di - 6, 0):di]
                rv_w = np.nanmean(w_vals) if np.sum(~np.isnan(w_vals)) >= 2 else rv_d
                m_vals = rv1[max(di - 11, 0):di]
                rv_m = np.nanmean(m_vals) if np.sum(~np.isnan(m_vals)) >= 5 else rv_w
                if rv_d > 1e-12:
                    pred = beta[0] + beta[1] * rv_d + beta[2] * rv_w + beta[3] * rv_m
                    har_ratio_30[si, di] = pred / rv_d
            except Exception:
                continue
    new['HAR_RATIO_30'] = har_ratio_30

    # Variant 3: RV momentum (simple ratio of recent RV to longer RV)
    rv_mom = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(30, ND):
            d = di - 1
            # Short RV (5-day)
            s_vals = []
            for dd in range(max(d - 4, 1), d + 1):
                c0, c1 = C[si, dd - 1], C[si, dd]
                if np.isnan(c0) or np.isnan(c1) or c0 <= 0:
                    continue
                s_vals.append(np.log(c1 / c0) ** 2)
            # Long RV (20-day)
            l_vals = []
            for dd in range(max(d - 19, 1), d + 1):
                c0, c1 = C[si, dd - 1], C[si, dd]
                if np.isnan(c0) or np.isnan(c1) or c0 <= 0:
                    continue
                l_vals.append(np.log(c1 / c0) ** 2)
            if len(s_vals) >= 3 and len(l_vals) >= 10:
                rv_s = np.mean(s_vals)
                rv_l = np.mean(l_vals)
                if rv_l > 1e-12:
                    rv_mom[si, di] = rv_s / rv_l
    new['RV_MOM'] = rv_mom

    print(f"  HAR variants done ({time.time()-t0:.1f}s)", flush=True)

    # Rank normalize
    def rank_pct(arr, start=60):
        res = np.full_like(arr, np.nan)
        for di in range(start, arr.shape[1]):
            vals = arr[:, di]
            mask = ~np.isnan(vals)
            if mask.sum() < 50:
                continue
            ranked = np.argsort(np.argsort(vals[mask])).astype(float)
            n = len(ranked)
            pct = ranked / max(n - 1, 1) * 100
            for k, idx in enumerate(np.where(mask)[0]):
                res[idx, di] = pct[k]
        return res

    # HAR_RATIO: low = compression = good → invert
    for name in ['HAR_RATIO_60', 'HAR_RATIO_30']:
        r = rank_pct(new[name])
        inv = r.copy()
        mask = ~np.isnan(inv)
        inv[mask] = 100.0 - inv[mask]
        new[f'R_{name}_INV'] = inv

    # RV_MOM: low = short RV < long RV = compression = good → invert
    r = rank_pct(new['RV_MOM'])
    inv = r.copy()
    mask = ~np.isnan(inv)
    inv[mask] = 100.0 - inv[mask]
    new['R_RV_MOM_INV'] = inv

    print(f"  Total HAR variant factors: {len([k for k in new if k.startswith('R_')])}", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V31 — HAR-RV Core Optimization", flush=True)
    print("  Building on V15 breakthrough: HAR_RV +235.6% DD=32.4% ALL+", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load all existing factors
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
    all_factors = v11_all

    # Add V14 HAR-RV if not already present
    if 'R_HAR_RV_RATIO_INV' not in all_factors:
        from alpha_v14 import compute_v14_factors
        v14_factors = compute_v14_factors(NS, ND, C, O, H, L, V)
        all_factors.update(v14_factors)

    # Compute HAR variants
    har_variants = compute_har_variants(NS, ND, C, O, H, L, V)
    all_factors.update(har_variants)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    results = []

    # Baseline: V15 champion HAR_RV
    har_champ = {'R_HAR_RV_RATIO_INV': 0.3, 'R_TENSION': 0.3,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    for top_n in [1]:
        for atr in [1.0, 1.2]:
            r = backtest_v7c(har_champ, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'HAR60_T{top_n}_A{atr}'
                results.append(r)
    print(f"  V15 champion baseline done", flush=True)

    # =====================================================================
    # TEST 1: HAR-RV variant comparison
    # =====================================================================
    print(f"\n  === TEST 1: HAR-RV Variant Comparison ===", flush=True)
    har_tests = {
        'HAR30': {'R_HAR_RATIO_30_INV': 0.3, 'R_TENSION': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'HAR60p': {'R_HAR_RATIO_60_INV': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'RVMom': {'R_RV_MOM_INV': 0.3, 'R_TENSION': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
    }
    for tname, weights in har_tests.items():
        missing = [f for f in weights if f not in all_factors]
        if missing:
            print(f"  SKIP {tname}: missing {missing}", flush=True)
            continue
        for top_n in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{tname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {tname} done", flush=True)

    # =====================================================================
    # TEST 2: HAR-RV weight sweep
    # =====================================================================
    print(f"\n  === TEST 2: HAR-RV Weight Sweep ===", flush=True)
    for w_har in [0.2, 0.3, 0.4, 0.5]:
        w_rest = (1.0 - w_har) / 3.0
        weights = {'R_HAR_RV_RATIO_INV': w_har,
                   'R_TENSION': w_rest, 'R_R_SQUARED': w_rest, 'R_SMA_DEV': w_rest}
        for top_n in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'W{w_har:.1f}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  HAR weight={w_har} done", flush=True)

    # =====================================================================
    # TEST 3: HAR-RV + BwpBNW fusion
    # =====================================================================
    print(f"\n  === TEST 3: HAR-RV + BwpBNW Fusion ===", flush=True)
    fusion_tests = {
        'HB_50_50': {'R_HAR_RV_RATIO_INV': 0.25, 'R_BWP_BNW': 0.25,
                     'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        'HB_40_60': {'R_HAR_RV_RATIO_INV': 0.2, 'R_BWP_BNW': 0.3,
                     'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        'HB_60_40': {'R_HAR_RV_RATIO_INV': 0.3, 'R_BWP_BNW': 0.2,
                     'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        'HB_TrS2': {'R_HAR_RV_RATIO_INV': 0.2, 'R_BWP_BNW': 0.2,
                    'R_TENSION': 0.3, 'R_R_SQUARED': 0.3},
        'HB_5F': {'R_HAR_RV_RATIO_INV': 0.2, 'R_BWP_BNW': 0.2,
                  'R_TENSION': 0.2, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'HB_Sqz': {'R_HAR_RV_RATIO_INV': 0.2, 'R_BWP_BNW': 0.2,
                   'R_BB_WIDTH_PCT_INV': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
    }
    for tname, weights in fusion_tests.items():
        missing = [f for f in weights if f not in all_factors]
        if missing:
            print(f"  SKIP {tname}: missing {missing}", flush=True)
            continue
        for top_n in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{tname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {tname} done", flush=True)

    # =====================================================================
    # TEST 4: HAR-RV + momentum/ker confirmation
    # =====================================================================
    print(f"\n  === TEST 4: HAR-RV + Momentum/KER ===", flush=True)
    conf_tests = {
        'HK_KER': {'R_HAR_RV_RATIO_INV': 0.25, 'R_KER': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        'HK_MOM': {'R_HAR_RV_RATIO_INV': 0.25, 'R_MOM5': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        'HK_Rel': {'R_HAR_RV_RATIO_INV': 0.25, 'R_RELATIVE_STRENGTH': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        'HK_RSI': {'R_HAR_RV_RATIO_INV': 0.25, 'R_RSI': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        'HK_MACD': {'R_HAR_RV_RATIO_INV': 0.25, 'R_MACD_HIST': 0.25,
                    'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
    }
    for tname, weights in conf_tests.items():
        missing = [f for f in weights if f not in all_factors]
        if missing:
            print(f"  SKIP {tname}: missing {missing}", flush=True)
            continue
        for top_n in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{tname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {tname} done", flush=True)

    # =====================================================================
    # TEST 5: Rebalancing frequency sweep
    # =====================================================================
    print(f"\n  === TEST 5: Rebalancing Frequency ===", flush=True)
    for rebal in [5, 7, 10, 14, 20]:
        for top_n in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_v7c(har_champ, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr)
                if r:
                    r['test'] = f'R{rebal}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  Rebal={rebal} done", flush=True)

    # =====================================================================
    # TEST 6: Top-N sweep
    # =====================================================================
    print(f"\n  === TEST 6: Top-N Sweep ===", flush=True)
    for top_n in [1, 2, 3]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v7c(har_champ, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'TN{top_n}_A{atr}'
                results.append(r)
    print(f"  Top-N sweep done", flush=True)

    # =====================================================================
    # TEST 7: HAR-RV variants + best fusion + rebal sweep
    # =====================================================================
    print(f"\n  === TEST 7: Best Fusion + Rebal Sweep ===", flush=True)
    best_fusions = [
        ('HB_50_50', {'R_HAR_RV_RATIO_INV': 0.25, 'R_BWP_BNW': 0.25,
                      'R_TENSION': 0.25, 'R_R_SQUARED': 0.25}),
        ('HB_TrS2', {'R_HAR_RV_RATIO_INV': 0.2, 'R_BWP_BNW': 0.2,
                     'R_TENSION': 0.3, 'R_R_SQUARED': 0.3}),
    ]
    for tname, weights in best_fusions:
        missing = [f for f in weights if f not in all_factors]
        if missing:
            continue
        for rebal in [7, 10, 14]:
            for top_n in [1]:
                for atr in [1.0, 1.2]:
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr)
                    if r:
                        r['test'] = f'F_{tname}_R{rebal}_A{atr}'
                        results.append(r)
        print(f"  {tname} rebal sweep done", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 50 RESULTS (V31 HAR-RV CORE OPTIMIZATION)", flush=True)
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

    # Best per group
    groups = {}
    for r in results:
        prefix = r['test'].split('_T')[0] if '_T' in r['test'] else r['test'].split('_A')[0]
        if prefix not in groups or r['ann'] > groups[prefix]['ann']:
            groups[prefix] = r
    print(f"\n  Best per group:", flush=True)
    for r in sorted(groups.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['test']:<30s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    print(f"\n{'='*70}", flush=True)
