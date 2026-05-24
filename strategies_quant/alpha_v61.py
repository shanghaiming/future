"""
Alpha V61 — R_REL_STRENGTH Fine-Tuning
=======================================
R_REL_STRENGTH (market-relative momentum) is the strongest new factor:
  Solo: +2583.6% DD=33.9%
  V56 + W0.15: +1746.0% DD=25.2%

This script fine-tunes:
1. EMA span for relative return smoothing (3, 5, 10, 15, 20, 30)
2. ATR stop multiplier (0.3, 0.5, 0.6, 0.7, 0.8, 1.0)
3. Rebalance days (3, 5, 7, 10)
4. Top N stocks (1, 2, 3)
5. V56 + REL_STRENGTH weight grid (0.05-0.30)
6. Lookback for market-relative calculation (5, 10, 20, 40)
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7c import backtest_v7c
from alpha_v44 import compute_v41_factors_only
from alpha_v48 import compute_v48_factors
from alpha_v49 import compute_v49_factors
from alpha_v52 import compute_v52_factors
from alpha_v55 import compute_decomposed_factors


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


def _ema(arr, span):
    """EMA along axis=1. No look-ahead: uses arr[:, di-1]."""
    NS, ND = arr.shape
    alpha = 2.0 / (span + 1)
    out = np.full_like(arr, np.nan)
    for di in range(2, ND):
        mask_prev = ~np.isnan(out[:, di - 1])
        mask_curr = ~np.isnan(arr[:, di - 1])
        both = mask_prev & mask_curr
        out[both, di] = alpha * arr[both, di - 1] + (1 - alpha) * out[both, di - 1]
        new_only = mask_curr & ~mask_prev
        out[new_only, di] = arr[new_only, di - 1]
    return out


def _rolling_mean(arr, window, min_valid=None):
    if min_valid is None:
        min_valid = window // 2
    NS, ND = arr.shape
    out = np.full_like(arr, np.nan)
    cumsum = np.nancumsum(arr, axis=1)
    cumcount = np.cumsum(~np.isnan(arr), axis=1)
    for di in range(window, ND):
        s = cumsum[:, di - 1] - (cumsum[:, di - window - 1] if di > window else 0)
        c = cumcount[:, di - 1] - (cumcount[:, di - window - 1] if di > window else 0)
        valid = c >= min_valid
        out[valid, di] = s[valid] / c[valid]
    return out


def compute_rel_strength_factors(NS, ND, C, O, H, L, V):
    """Compute R_REL_STRENGTH variants with different lookbacks."""
    t0 = time.time()
    factors = {}

    # Daily returns
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    # Market return (equal-weighted)
    mkt_ret = np.full(ND, np.nan)
    for di in range(1, ND):
        valid = ~np.isnan(ret[:, di])
        if valid.sum() > 50:
            mkt_ret[di] = np.mean(ret[valid, di])

    # Relative return
    rel_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di]) & ~np.isnan(mkt_ret[di])
        rel_ret[m, di] = ret[m, di] - mkt_ret[di]

    # Test different EMA spans
    for span in [3, 5, 10, 15, 20, 30]:
        name = f'R_REL_STR_S{span}'
        smoothed = _ema(rel_ret, span)
        factors[name] = _rank_normalize(smoothed)

    # Test rolling mean variants
    for window in [5, 10, 20, 40]:
        name = f'R_REL_STR_M{window}'
        smoothed = _rolling_mean(rel_ret, window)
        factors[name] = _rank_normalize(smoothed)

    print(f"  REL_STRENGTH variants done ({time.time()-t0:.0f}s)", flush=True)
    return factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V61 — R_REL_STRENGTH Fine-Tuning")
    print("  V60 record: +2583.6% DD=33.9% (REL_STRENGTH solo)")
    print("  V56 record: +1630.7% DD=25.2%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing existing factors...", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)
    base_factors = {**v41, **v48, **v49, **v52, **v55}

    print("\n  Computing REL_STRENGTH variants...", flush=True)
    rel_factors = compute_rel_strength_factors(NS, ND, C, O, H, L, V)
    all_factors = {**base_factors, **rel_factors}

    # V56 weights
    v54_base = {'R_BWP_BNW': 0.205, 'R_TENSION': 0.205, 'R_VWCM': 0.205,
                'R_BVR': 0.154, 'R_BUY_FRAC': 0.138, 'R_VPIN': 0.092}
    v56_weights = {**v54_base, 'R_SHOCK_MOM': 0.08, 'R_TREND_ACC': 0.15}
    total = sum(v56_weights.values())
    v56_norm = {k: v / total for k, v in v56_weights.items()}

    results = []

    # =====================================================================
    # Baseline: V56
    # =====================================================================
    print("\n  V56 baseline...", flush=True)
    r = backtest_v7c(v56_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V56_BASE'
        results.append(r)
        print(f"  V56: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: REL_STRENGTH variants solo
    # =====================================================================
    print("\n  Test 1: REL_STRENGTH variants solo...", flush=True)
    for fname in sorted(rel_factors.keys()):
        for atr in [0.5, 0.7, 0.8]:
            r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{fname}_SOLO_A{atr}'
                results.append(r)
                print(f"    {fname}_A{atr}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 2: Best solo variants with parameter sweep
    # =====================================================================
    print("\n  Test 2: Parameter sweep on best variants...", flush=True)
    # Find best solo variants
    solos = [r for r in results if '_SOLO_' in r['test']]
    best_names = []
    if solos:
        solos_sorted = sorted(solos, key=lambda x: -x['ann'])
        for s in solos_sorted[:6]:
            name = s['test'].split('_SOLO_')[0]
            if name not in best_names:
                best_names.append(name)

        # ATR + rebalance + top_n sweep for best variants
        for fname in best_names[:3]:
            for atr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
                for rebal in [3, 5, 7]:
                    for top_n in [1]:
                        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                                        top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr)
                        if r:
                            r['test'] = f'{fname}_A{atr}_R{rebal}'
                            results.append(r)

            print(f"    {fname} sweep done", flush=True)

    # =====================================================================
    # Test 3: V56 + best REL_STRENGTH variants
    # =====================================================================
    print("\n  Test 3: V56 + REL_STRENGTH weight sweep...", flush=True)
    for fname in best_names[:4]:
        for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
            weights = {**v56_norm, fname: w}
            total = sum(weights.values())
            wn = {k: v / total for k, v in weights.items()}
            for atr in [0.5]:
                r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'V56+{fname}_W{w:.2f}'
                    results.append(r)
                    print(f"    V56+{fname}_W{w:.2f}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 4: Replace V56 factors with REL_STRENGTH
    # =====================================================================
    print("\n  Test 4: V54 base + REL_STRENGTH (no SHOCK/TREND)...", flush=True)
    # Test V54 base + REL_STRENGTH only
    for fname in best_names[:2]:
        for w in [0.10, 0.15, 0.20]:
            v54_test = {**v54_base, fname: w}
            total = sum(v54_test.values())
            wn = {k: v / total for k, v in v54_test.items()}
            r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'V54+{fname}_W{w:.2f}'
                results.append(r)
                print(f"    V54+{fname}_W{w:.2f}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 5: Pure REL_STRENGTH combos (no V56)
    # =====================================================================
    print("\n  Test 5: Multi-lookback REL_STRENGTH combos...", flush=True)
    # Combine multiple REL_STRENGTH spans
    ema_names = [n for n in best_names if '_S' in n]
    if len(ema_names) >= 2:
        for w1_name in ema_names[:3]:
            for w2_name in ema_names[:3]:
                if w1_name >= w2_name:
                    continue
                for w1 in [0.5, 0.6]:
                    for w2 in [0.4, 0.5]:
                        weights = {w1_name: w1, w2_name: w2}
                        total = sum(weights.values())
                        wn = {k: v / total for k, v in weights.items()}
                        r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                                        top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                        if r:
                            r['test'] = f'{w1_name}+{w2_name}'
                            results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'=' * 100}", flush=True)
    print(f"  ALL RESULTS (V61 REL_STRENGTH FINE-TUNING)", flush=True)
    print(f"  {'Test':<50s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-' * 90}", flush=True)
    for r in results[:80]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<50s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Solo summary
    print(f"\n  === SOLO REL_STRENGTH SUMMARY ===", flush=True)
    solos_sorted = sorted([r for r in results if '_SOLO_' in r['test']], key=lambda x: -x['ann'])
    for r in solos_sorted[:15]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<50s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    # V56+ summary
    print(f"\n  === V56 + REL_STRENGTH BEST ===", flush=True)
    v56_new = sorted([r for r in results if r['test'].startswith('V56+')], key=lambda x: -x['ann'])
    for r in v56_new[:20]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<50s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    # Parameter sweep summary
    print(f"\n  === PARAMETER SWEEP (ATR + REBALANCE) ===", flush=True)
    sweep = sorted([r for r in results if '_A' in r['test'] and '_R' in r['test']], key=lambda x: -x['ann'])
    for r in sweep[:15]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<50s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i + 1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V61 BEST ===", flush=True)
        print(f"  V61: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V60 RECORD: +2583.6% DD=33.9% (R_REL_STRENGTH solo)", flush=True)
        print(f"  V56 RECORD: +1630.7% DD=25.2%", flush=True)
        delta = best['ann'] - 2583.6
        print(f"  Delta from V60: {delta:+.1f}%", flush=True)

    print(f"\n{'=' * 70}", flush=True)
