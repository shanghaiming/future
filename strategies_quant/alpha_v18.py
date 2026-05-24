"""
Alpha V18 — Independent Strategy Exploration
=============================================
NOT built on BwpBNW. 5 completely independent strategies, each testing
a DIFFERENT signal dimension from the 260-strategy deep study.

Strategy 1: Pure HAR-RV Volatility Compression
  - Factor: R_HAR_RV_RATIO_INV (low predicted vol = good)
  - Thesis: Stocks with calm predicted volatility are ready for breakout
  - Combine with: structural tension for direction

Strategy 2: Pure Institutional Pressure
  - Factor: R_LOG_PRESSURE (accumulation signal)
  - Thesis: Institutional buying precedes price moves
  - Combine with: body quality for timing

Strategy 3: Pure ATR Terrain
  - Factor: R_ATR_TERRAIN (4-state volatility regime)
  - Thesis: Volatility compression→expansion is the dominant cycle
  - Combine with: momentum for direction

Strategy 4: Pure Squeeze+Release (no BB_WIDTH_PCT_INV)
  - Factor: R_SQZ_DEPTH × R_RELEASE_MOM
  - Thesis: Squeeze energy + release direction = timing
  - No interaction with BB_WIDTH_PCT or BODY_NW

Strategy 5: Dual Orthogonal (completely independent signals)
  - R_KER (Kaufman efficiency) × R_FISHER (Gaussian transform)
  - Thesis: Efficient markets + Fisher-confirmed signal = robust

Each strategy tested with backtest_v7c (proven engine).

LOOK-AHEAD SELF-CHECK:
  [x] All factors use ONLY data up to d=di-1
  [x] Results stored at index di
  [x] No same-day data used
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
from alpha_v14 import compute_v14_factors, compute_v14_interactions
from alpha_v7c import backtest_v7c


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V18 — Independent Strategy Exploration", flush=True)
    print("  5 Completely Different Signal Dimensions", flush=True)
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

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    results = []

    # =====================================================================
    # BASELINE: V10 BwpBNW for reference
    # =====================================================================
    bwp = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
            'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v7c(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'BwpBNW_T{top_n}_A{atr}'
                results.append(r)
    print(f"  Baseline done", flush=True)

    # =====================================================================
    # STRATEGY 1: Pure HAR-RV Volatility Compression
    # =====================================================================
    print(f"\n  === STRATEGY 1: Pure HAR-RV ===", flush=True)
    har_portfolios = {
        'HAR_pure': {'R_HAR_RV_RATIO_INV': 0.4, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.3},
        'HAR_tens': {'R_HAR_RV_RATIO_INV': 0.3, 'R_TENSION': 0.4, 'R_SMA_DEV': 0.3},
        'HAR_mom': {'R_HAR_RV_RATIO_INV': 0.3, 'R_MOM5': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'HAR_lp': {'R_HAR_RV_RATIO_INV': 0.3, 'R_LOG_PRESSURE': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'HAR_at': {'R_HAR_RV_RATIO_INV': 0.25, 'R_ATR_TERRAIN': 0.25, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
    }
    for pname, weights in har_portfolios.items():
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {pname} done", flush=True)

    # =====================================================================
    # STRATEGY 2: Pure Institutional Pressure
    # =====================================================================
    print(f"\n  === STRATEGY 2: Pure LOG_PRESSURE ===", flush=True)
    lp_portfolios = {
        'LP_pure': {'R_LOG_PRESSURE': 0.4, 'R_BODY_NW': 0.3, 'R_TENSION': 0.3},
        'LP_tens': {'R_LOG_PRESSURE': 0.3, 'R_TENSION': 0.4, 'R_R_SQUARED': 0.3},
        'LP_squeeze': {'R_LOG_PRESSURE': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3, 'R_BODY_NW': 0.2, 'R_TENSION': 0.2},
        'LP_har': {'R_LOG_PRESSURE': 0.25, 'R_HAR_RV_RATIO_INV': 0.25, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        'LP_mom': {'R_LOG_PRESSURE': 0.3, 'R_MOM5': 0.3, 'R_KER': 0.2, 'R_R_SQUARED': 0.2},
    }
    for pname, weights in lp_portfolios.items():
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {pname} done", flush=True)

    # =====================================================================
    # STRATEGY 3: Pure ATR Terrain
    # =====================================================================
    print(f"\n  === STRATEGY 3: Pure ATR_TERRAIN ===", flush=True)
    at_portfolios = {
        'AT_pure': {'R_ATR_TERRAIN': 0.4, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.3},
        'AT_body': {'R_ATR_TERRAIN': 0.3, 'R_BODY_NW': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'AT_squeeze': {'R_ATR_TERRAIN': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'AT_mom': {'R_ATR_TERRAIN': 0.25, 'R_MOM5': 0.25, 'R_KER': 0.25, 'R_TENSION': 0.25},
        'AT_har_lp': {'R_ATR_TERRAIN': 0.2, 'R_HAR_RV_RATIO_INV': 0.2, 'R_LOG_PRESSURE': 0.2,
                       'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
    }
    for pname, weights in at_portfolios.items():
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {pname} done", flush=True)

    # =====================================================================
    # STRATEGY 4: Pure Squeeze+Release (NO BWP, NO BODY_NW)
    # =====================================================================
    print(f"\n  === STRATEGY 4: Pure Squeeze+Release ===", flush=True)
    sqz_portfolios = {
        'Sqz_pure': {'R_SQZ_DEPTH': 0.3, 'R_RELEASE_MOM': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'Sqz_dur': {'R_SQZ_DURATION': 0.3, 'R_RELEASE_MOM': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'Sqz_bb': {'R_BB_SQUEEZE_INV': 0.3, 'R_RELEASE_MOM': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'Sqz_har': {'R_SQZ_DEPTH': 0.25, 'R_RELEASE_MOM': 0.25, 'R_HAR_RV_RATIO_INV': 0.25, 'R_TENSION': 0.25},
        'Sqz_vol': {'R_SQZ_DEPTH': 0.25, 'R_RELEASE_MOM': 0.25, 'R_VOL_ANOMALY': 0.25, 'R_TENSION': 0.25},
    }
    for pname, weights in sqz_portfolios.items():
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {pname} done", flush=True)

    # =====================================================================
    # STRATEGY 5: Dual Orthogonal — KER × FISHER
    # =====================================================================
    print(f"\n  === STRATEGY 5: Dual Orthogonal KER×FISHER ===", flush=True)
    orth_portfolios = {
        'KF_pure': {'R_KER': 0.35, 'R_FISHER': 0.35, 'R_TENSION': 0.3},
        'KF_tens': {'R_KER': 0.3, 'R_FISHER': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'KF_mom': {'R_KER': 0.25, 'R_MOM5': 0.25, 'R_FISHER': 0.25, 'R_TENSION': 0.25},
        'KF_r2': {'R_KER': 0.3, 'R_R_SQUARED': 0.3, 'R_FISHER': 0.2, 'R_TENSION': 0.2},
        'KF_rel': {'R_KER': 0.25, 'R_REL_STR': 0.25, 'R_FISHER': 0.25, 'R_TENSION': 0.25},
    }
    # Also try completely different pairings
    orth_portfolios['KalTens'] = {'R_KALMAN_SLOPE': 0.3, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    orth_portfolios['HursR2'] = {'R_HURST': 0.3, 'R_R_SQUARED': 0.3, 'R_TENSION': 0.2, 'R_SMA_DEV': 0.2}
    orth_portfolios['OfiBody'] = {'R_OFI': 0.3, 'R_BODY_NW': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2}
    orth_portfolios['MacdTens'] = {'R_MACD_HIST': 0.3, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    orth_portfolios['RsiBwp'] = {'R_RSI': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2}
    orth_portfolios['NWsqz'] = {'R_NW_SLOPE': 0.3, 'R_BB_SQUEEZE_INV': 0.3, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2}

    for pname, weights in orth_portfolios.items():
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {pname} done", flush=True)

    # =====================================================================
    # STRATEGY 6: V14 Interaction Factors (completely new)
    # =====================================================================
    print(f"\n  === STRATEGY 6: V14 Interactions ===", flush=True)
    v14_int_portfolios = {
        'LP_BNW': {'R_LP_BNW': 0.3, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'LP_TENS': {'R_LP_TENS': 0.3, 'R_R_SQUARED': 0.3, 'R_SMA_DEV': 0.2, 'R_BODY_NW': 0.2},
        'LP_BWP': {'R_LP_BWP': 0.3, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'AT_BNW': {'R_AT_BNW': 0.3, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'AT_TENS': {'R_AT_TENS': 0.3, 'R_R_SQUARED': 0.3, 'R_SMA_DEV': 0.2, 'R_BODY_NW': 0.2},
        'HAR_BWP': {'R_HAR_BWP': 0.3, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'LPA_BNW': {'R_LPA_BNW': 0.3, 'R_TENSION': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
    }
    for pname, weights in v14_int_portfolios.items():
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {pname} done", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 50 RESULTS (V18 INDEPENDENT STRATEGIES)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best per strategy
    strategies = {
        'Baseline': 'BwpBNW',
        'S1_HAR-RV': 'HAR_',
        'S2_Pressure': 'LP_',
        'S3_Terrain': 'AT_',
        'S4_Squeeze': 'Sqz_',
        'S5_Orthogonal': ['KF_', 'Kal', 'Hurs', 'Ofi', 'Macd', 'Rsi', 'NW'],
        'S6_V14Int': ['LP_BNW', 'LP_TENS', 'LP_BWP', 'AT_BNW', 'AT_TENS', 'HAR_BWP', 'LPA_BNW'],
    }
    print(f"\n  Best per strategy:", flush=True)
    for sname, prefixes in strategies.items():
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        best = None
        for r in results:
            for p in prefixes:
                if r['test'].startswith(p):
                    if best is None or r['ann'] > best['ann']:
                        best = r
        if best:
            pos = " ALL+" if all_positive(best) else ""
            print(f"    {sname:<15s}: {best['test']:<30s} → {best['ann']:+.1f}% DD={best['max_dd']:.1f}%{pos}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # Best Top=2
    top2 = [r for r in results if '_T2_' in r['test']]
    if top2:
        top2.sort(key=lambda x: -x['ann'])
        print(f"\n  Best Top=2:", flush=True)
        for r in top2[:5]:
            pos = " ALL+" if all_positive(r) else ""
            print(f"    {r['test']:<30s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    print(f"\n{'='*70}", flush=True)
