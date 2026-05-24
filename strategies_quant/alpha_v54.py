"""
Alpha V54 — Final Push to 1000%
================================
V53 best: BF0.1 replacing R_R_SQUARED = +937.2% DD=20.6% ALL+

V54 explores:
1. Ultra-fine BUY_FRAC weight sweep (0.05-0.25 step 0.01)
2. Ultra-fine ATR sweep (0.30-0.55 step 0.02)
3. 7-factor: add OFI/VPIN/BUY_FRAC_5 as 7th factor
4. Non-equal weights: BUY_FRAC-heavy, VWCM-heavy
5. Drop weakest factor + add microstructure
6. Rebalance 3-5 with best config
7. BUY_FRAC window combos (short + long)
8. Top_n=1 vs top_n=2
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
    print("  Alpha V54 — Final Push to 1000%")
    print("  V53 best: BF0.1 rep R_R_SQUARED = +937.2% DD=20.6%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing all factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48_factors = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49_factors = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52_factors = compute_v52_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41_factors, **v48_factors, **v49_factors, **v52_factors}

    # BUY_FRAC window variants
    from alpha_v52 import _rolling_mean as _rm
    buy_tick = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        buy_tick[mask, di] = (C[mask, di] > C[mask, di - 1]).astype(float)

    for window in [3, 5, 7, 10, 15, 20, 30, 40]:
        bf_var = _rm(buy_tick, window)
        all_factors[f'R_BUY_FRAC_{window}'] = _rank_normalize(bf_var)

    # V53 best base: V51 with R_R_SQUARED replaced by BUY_FRAC
    v54_base = {'R_BWP_BNW': 0.178, 'R_TENSION': 0.178, 'R_SMA_DEV': 0.178,
                'R_VWCM': 0.178, 'R_BVR': 0.134}

    results = []

    # =====================================================================
    # Baseline
    # =====================================================================
    print("\n  Baseline...", flush=True)
    w = {**v54_base, 'R_BUY_FRAC': 0.1}
    total = sum(w.values())
    w_norm = {k: v / total for k, v in w.items()}
    r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V53_BEST_BASELINE'
        results.append(r)
        print(f"  V53 baseline: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: Ultra-fine weight × ATR grid
    # =====================================================================
    print("\n  Test 1: Ultra-fine grid...", flush=True)
    best_w_bf = None
    best_atr = None
    best_ann = 0
    for w_bf in np.arange(0.04, 0.26, 0.02):
        for atr in np.arange(0.30, 0.56, 0.02):
            atr = round(atr, 2)
            weights = {**v54_base, 'R_BUY_FRAC': round(w_bf, 2)}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'BF{w_bf:.2f}_A{atr:.2f}'
                results.append(r)
                if r['ann'] > best_ann:
                    best_ann = r['ann']
                    best_w_bf = w_bf
                    best_atr = atr
    if best_w_bf is not None:
        print(f"  Best: BF{best_w_bf:.2f}_A{best_atr:.2f} = {best_ann:+.1f}%", flush=True)
    else:
        print(f"  No profitable combination found", flush=True)

    # =====================================================================
    # Test 2: 7-factor: add OFI to V53 best
    # =====================================================================
    print("\n  Test 2: 7-factor (add OFI/VPIN)...", flush=True)
    for w_extra in [0.03, 0.05, 0.08, 0.10, 0.12]:
        for extra_f in ['R_OFI', 'R_VPIN', 'R_DEPTH_IMB', 'R_BUY_FRAC_5']:
            for atr in [0.40, 0.45, 0.50]:
                weights = {**v54_base, 'R_BUY_FRAC': 0.10, extra_f: w_extra}
                total = sum(weights.values())
                w_norm = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    short = extra_f[-4:]
                    r['test'] = f'7F_BF+{short}_{w_extra}_A{atr}'
                    results.append(r)

    # =====================================================================
    # Test 3: Drop weakest + add microstructure
    # =====================================================================
    print("\n  Test 3: Drop weakest...", flush=True)
    # Which is weakest? Try dropping each factor from V53 best
    for drop_f in ['R_BWP_BNW', 'R_TENSION', 'R_SMA_DEV', 'R_VWCM', 'R_BVR']:
        for add_f in ['R_OFI', 'R_VPIN', 'R_BUY_FRAC_5', 'R_BUY_FRAC_10', 'R_DEPTH_IMB']:
            weights = {k: v for k, v in v54_base.items() if k != drop_f}
            weights['R_BUY_FRAC'] = 0.12
            weights[add_f] = 0.08
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'DROP_{drop_f[-3:]}+{add_f[-3:]}'
                results.append(r)

    # =====================================================================
    # Test 4: Non-equal weights favoring BUY_FRAC
    # =====================================================================
    print("\n  Test 4: BUY_FRAC-heavy weights...", flush=True)
    weight_configs = [
        ('BF30', {'R_BWP_BNW': 0.10, 'R_TENSION': 0.12, 'R_SMA_DEV': 0.12,
                  'R_VWCM': 0.18, 'R_BVR': 0.10, 'R_BUY_FRAC': 0.30}),
        ('BF35', {'R_BWP_BNW': 0.08, 'R_TENSION': 0.10, 'R_SMA_DEV': 0.10,
                  'R_VWCM': 0.17, 'R_BVR': 0.10, 'R_BUY_FRAC': 0.35}),
        ('VWCM25_BF20', {'R_BWP_BNW': 0.10, 'R_TENSION': 0.12, 'R_SMA_DEV': 0.10,
                         'R_VWCM': 0.25, 'R_BVR': 0.13, 'R_BUY_FRAC': 0.20}),
        ('TENS25_BF20', {'R_BWP_BNW': 0.10, 'R_TENSION': 0.25, 'R_SMA_DEV': 0.10,
                         'R_VWCM': 0.15, 'R_BVR': 0.10, 'R_BUY_FRAC': 0.20}),
        ('BVR20_BF15', {'R_BWP_BNW': 0.12, 'R_TENSION': 0.12, 'R_SMA_DEV': 0.12,
                        'R_VWCM': 0.18, 'R_BVR': 0.20, 'R_BUY_FRAC': 0.15}),
        ('EQ6', {'R_BWP_BNW': 0.167, 'R_TENSION': 0.167, 'R_SMA_DEV': 0.167,
                 'R_VWCM': 0.167, 'R_BVR': 0.167, 'R_BUY_FRAC': 0.167}),
        ('EQ5noSMA', {'R_BWP_BNW': 0.20, 'R_TENSION': 0.20,
                      'R_VWCM': 0.20, 'R_BVR': 0.20, 'R_BUY_FRAC': 0.20}),
    ]
    for name, weights in weight_configs:
        for atr in [0.40, 0.45, 0.50, 0.55]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{name}_A{atr}'
                results.append(r)

    # =====================================================================
    # Test 5: Rebalance sweep with V53 best
    # =====================================================================
    print("\n  Test 5: Rebalance sweep...", flush=True)
    for rebal in [3, 4, 5, 6, 7]:
        for atr in [0.40, 0.45, 0.50]:
            weights = {**v54_base, 'R_BUY_FRAC': 0.10}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=rebal, atr_stop_mult=atr)
            if r:
                r['test'] = f'BF0.10_R{rebal}_A{atr}'
                results.append(r)

    # =====================================================================
    # Test 6: BUY_FRAC window combo (short + long together)
    # =====================================================================
    print("\n  Test 6: BF window combo...", flush=True)
    for w_short in [0.05, 0.08, 0.10]:
        for w_long in [0.05, 0.08, 0.10]:
            if w_short + w_long > 0.20:
                continue
            for short_w in [3, 5, 7]:
                for long_w in [20, 30]:
                    weights = {k: v for k, v in v54_base.items() if k not in ['R_VWCM', 'R_BVR']}
                    weights[f'R_BUY_FRAC_{short_w}'] = w_short
                    weights[f'R_BUY_FRAC_{long_w}'] = w_long
                    weights['R_VWCM'] = 0.18
                    weights['R_BVR'] = 0.10
                    total = sum(weights.values())
                    w_norm = {k: v / total for k, v in weights.items()}
                    r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                    if r:
                        r['test'] = f'BF{short_w}+BF{long_w}_{w_short}+{w_long}'
                        results.append(r)

    # =====================================================================
    # Test 7: top_n=2 with best configs
    # =====================================================================
    print("\n  Test 7: top_n=2...", flush=True)
    for w_bf in [0.10, 0.12, 0.15]:
        for atr in [0.4, 0.5, 0.6]:
            weights = {**v54_base, 'R_BUY_FRAC': w_bf}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=2, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'BF{w_bf}_N2_A{atr}'
                results.append(r)

    # =====================================================================
    # Test 8: BUY_FRAC replacing VWCM (alternative replacement)
    # =====================================================================
    print("\n  Test 8: BF replacing VWCM...", flush=True)
    for w_bf in [0.10, 0.15, 0.20, 0.25, 0.30]:
        weights = {'R_BWP_BNW': 0.178, 'R_TENSION': 0.178, 'R_R_SQUARED': 0.178,
                   'R_SMA_DEV': 0.178, 'R_BUY_FRAC': w_bf, 'R_BVR': 0.134}
        total = sum(weights.values())
        w_norm = {k: v / total for k, v in weights.items()}
        for atr in [0.40, 0.45, 0.50]:
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'BF_rep_VWCM_{w_bf}_A{atr}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V54 FINAL PUSH TO 1000%)", flush=True)
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
        print(f"\n  === V54 BEST ===", flush=True)
        print(f"  V54: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V53: BF0.1 rep R_R_SQUARED = +937.2% DD=20.6%", flush=True)
        print(f"  V52: REP_ARED→R_BUY_FRAC = +932.7% DD=27.1%", flush=True)
        print(f"  V51: VWCM0.2+BVR0.15_A0.5 = +620.8%DD=25.1%", flush=True)
        delta = best['ann'] - 937.2
        print(f"  Delta from V53: {delta:+.1f}%", flush=True)
        target = 1000
        gap = target - best['ann']
        print(f"  Gap to 1000%: {gap:.1f}%", flush=True)
        if best['ann'] >= target:
            print(f"  *** 1000% TARGET REACHED! ***", flush=True)

    print(f"\n{'='*70}", flush=True)
