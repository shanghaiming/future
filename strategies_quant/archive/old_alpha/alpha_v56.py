"""
Alpha V56 — V55 Best Fine-Tuning (SHOCK_MOM + TREND_ACC)
==========================================================
V55 discovered: R_SHOCK_MOM + R_TREND_ACC adds +169% to V54
V54 base: +1203.8% → V55 best: +1373.1%

Now: fine-tune weights, ATR, rebalance to maximize.
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
from alpha_v55 import compute_decomposed_factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V56 — Fine-Tune V55 Best (SHOCK_MOM + TREND_ACC)")
    print("  V55 best: +1373.1% DD=26.2%")
    print("  Target: push beyond 1400%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing all factors...", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41, **v48, **v49, **v52, **v55}

    # V54 base weights (unnormalized)
    v54_base = {
        'R_BWP_BNW': 0.205,
        'R_TENSION': 0.205,
        'R_VWCM': 0.205,
        'R_BVR': 0.154,
        'R_BUY_FRAC': 0.138,
        'R_VPIN': 0.092,
    }

    results = []

    # =====================================================================
    # Baseline
    # =====================================================================
    print("\n  V54 baseline...", flush=True)
    total = sum(v54_base.values())
    w_norm = {k: v / total for k, v in v54_base.items()}
    r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V54_BASE'
        results.append(r)
        print(f"  V54: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Fine weight grid: V54 + SHOCK_MOM + TREND_ACC
    # =====================================================================
    print("\n  Fine weight grid (SHOCK_MOM × TREND_ACC)...", flush=True)
    for w_shock in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
        for w_acc in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
            weights = {**v54_base, 'R_SHOCK_MOM': w_shock, 'R_TREND_ACC': w_acc}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'V54+S{w_shock:.2f}+A{w_acc:.2f}'
                results.append(r)

    # Print top-5 so far
    results_sorted = sorted(results, key=lambda x: -x['ann'])
    for r in results_sorted[:5]:
        print(f"  {r['test']}: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # ATR sweep with best weights
    # =====================================================================
    print("\n  ATR sweep with best weights...", flush=True)
    if results_sorted:
        best_test = results_sorted[0]['test']
        # Reconstruct weights from best test
        for w_shock in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
            for w_acc in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
                tname = f'V54+S{w_shock:.2f}+A{w_acc:.2f}'
                if tname == best_test:
                    best_ws = w_shock
                    best_wa = w_acc
                    break

        for atr in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            weights = {**v54_base, 'R_SHOCK_MOM': best_ws, 'R_TREND_ACC': best_wa}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'BEST_ATR{atr:.2f}'
                results.append(r)

    # =====================================================================
    # Also try: V54 + SHOCK_MOM only (simpler)
    # =====================================================================
    print("\n  V54 + SHOCK_MOM only...", flush=True)
    for w in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18]:
        for atr in [0.4, 0.5, 0.6]:
            weights = {**v54_base, 'R_SHOCK_MOM': w}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'V54+SHOCK_W{w:.2f}_A{atr:.1f}'
                results.append(r)

    # =====================================================================
    # V54 + TREND_ACC only
    # =====================================================================
    print("\n  V54 + TREND_ACC only...", flush=True)
    for w in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18]:
        for atr in [0.4, 0.5, 0.6]:
            weights = {**v54_base, 'R_TREND_ACC': w}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'V54+TACC_W{w:.2f}_A{atr:.1f}'
                results.append(r)

    # =====================================================================
    # Rebalance sweep with best
    # =====================================================================
    print("\n  Rebalance sweep...", flush=True)
    results_sorted = sorted(results, key=lambda x: -x['ann'])
    if results_sorted:
        # Try top config with different rebalance periods
        best_r = results_sorted[0]
        for rebal in [3, 4, 5, 7, 10]:
            weights = {**v54_base, 'R_SHOCK_MOM': best_ws, 'R_TREND_ACC': best_wa}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=rebal, atr_stop_mult=0.5)
            if r:
                r['test'] = f'BEST_REBAL{rebal}'
                results.append(r)

    # =====================================================================
    # Try dropping weakest V54 factor + adding both new
    # =====================================================================
    print("\n  Drop-weakest tests...", flush=True)
    for drop_f in ['R_VPIN', 'R_BUY_FRAC', 'R_BVR']:
        for w_shock in [0.08, 0.10, 0.12]:
            for w_acc in [0.08, 0.10, 0.12]:
                weights = {k: v for k, v in v54_base.items() if k != drop_f}
                weights['R_SHOCK_MOM'] = w_shock
                weights['R_TREND_ACC'] = w_acc
                total = sum(weights.values())
                w_norm = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                if r:
                    r['test'] = f'DROP_{drop_f[-3:]}+S{w_shock:.2f}+A{w_acc:.2f}'
                    results.append(r)

    # =====================================================================
    # Also try VWCM_FLUCT as replacement (solo was +562.6%!)
    # =====================================================================
    print("\n  VWCM_FLUCT replacement...", flush=True)
    for drop_f in ['R_BVR', 'R_BUY_FRAC', 'R_VPIN']:
        for w in [0.10, 0.15, 0.20]:
            weights = {k: v for k, v in v54_base.items() if k != drop_f}
            weights['R_VWCM_FLUCT'] = w
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'REP_{drop_f[-3:]}→FLUCT_W{w:.2f}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*100}", flush=True)
    print(f"  ALL RESULTS (V56 FINE-TUNING)", flush=True)
    print(f"  {'Test':<40s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:60]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<40s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
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
        print(f"\n  === V56 BEST ===", flush=True)
        print(f"  V56: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V54 RECORD: +1203.8% DD=31.5%", flush=True)
        print(f"  V55 RECORD: +1373.1% DD=26.2%", flush=True)
        delta = best['ann'] - 1373.1
        print(f"  Delta from V55: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
