"""
Alpha V7g — Push BodyNW Further
=================================
V7f发现: BodyNW Top=2, Reb=10, ATR=1.2 → +276.9%, DD=48.8%

进一步优化:
  1. ATR 1.0-1.3 细粒度
  2. Top=1 测试 (最激进)
  3. 更细致的BodyNW权重调优
  4. Regime filter
  5. 更多交互因子
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
from alpha_v7c import backtest_v7c, compute_market_regime


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V7g — Push BodyNW Further", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
    inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)
    extra_factors = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e_factors = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv_inter = compute_advanced_interactions(
        {**base_factors, **inter_factors, **extra_factors, **v7e_factors}, NS, ND)

    all_factors = {**base_factors, **inter_factors, **extra_factors, **v7e_factors, **adv_inter}
    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # Market regime
    print("[Regime] Computing market regime...", flush=True)
    mkt_ret20, mkt_ret5, breadth = compute_market_regime(NS, ND, C, O, H, L, V, syms)
    print("  Market regime done", flush=True)

    # BodyNW weight variants
    portfolios = {
        # Original BodyNW winner
        'BodyNW': {'R_BODY_NW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Heavier on BODY_NW
        'BNW_H': {'R_BODY_NW': 0.4, 'R_TENSION': 0.2,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # No SMA_DEV
        'BNW_NS': {'R_BODY_NW': 0.35, 'R_TENSION': 0.35,
                   'R_R_SQUARED': 0.3},
        # With TENS_SHAD instead of direct TENSION
        'BNW_TS': {'R_BODY_NW': 0.3, 'R_TENS_SHAD': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # With Fisher instead of R²
        'BNW_F': {'R_BODY_NW': 0.3, 'R_TENSION': 0.3,
                  'R_FISHER': 0.2, 'R_SMA_DEV': 0.2},
        # Pure BODY_NW + BODY_VOL (double body conviction)
        'BNW_BV': {'R_BODY_NW': 0.3, 'R_BODY_VOL': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # With TENS_SQ (extreme tension focus)
        'BNW_SQ': {'R_BODY_NW': 0.3, 'R_TENS_SQ': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # 3-factor minimal
        'BNW_3': {'R_BODY_NW': 0.4, 'R_TENSION': 0.3,
                  'R_R_SQUARED': 0.3},
        # SQ_base (no BodyNW, reference)
        'SQ_base': {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                    'R_TENS_SHAD': 0.25, 'R_BODY_VOL': 0.25},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [1, 2]:
            for rebal in [7, 10]:
                for atr in [1.0, 1.1, 1.2, 1.3]:
                    # Without regime
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr,
                                    market_ret20=mkt_ret20, breadth=breadth,
                                    regime_filter=False)
                    if r:
                        r.update({'portfolio': pname, 'top_n': top_n,
                                  'rebal': rebal, 'atr': atr, 'regime': 'off'})
                        results.append(r)

                    # With regime filter for BodyNW variants, top_n=2
                    if 'BNW' in pname and top_n == 2 and rebal == 10:
                        r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                        top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr,
                                        market_ret20=mkt_ret20, breadth=breadth,
                                        regime_filter=True, min_breadth=40)
                        if r:
                            r.update({'portfolio': pname + '_RF', 'top_n': top_n,
                                      'rebal': rebal, 'atr': atr, 'regime': 'br>40'})
                            results.append(r)
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # Filter for DD < 60% and all positive years
    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    viable = [r for r in results if r['max_dd'] < 60]
    viable_pos = [r for r in viable if all_positive(r)]

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 (all)", flush=True)
    print(f"  {'Portfolio':<18s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} {'Reg':<6s} | "
          f"{'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for r in results[:40]:
        pos_mark = " *" if all_positive(r) else ""
        print(f"  {r['portfolio']:<18s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.1f} "
              f"{r['regime']:<6s} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best viable (DD < 60%)
    print(f"\n  TOP 20 (DD < 60%, all positive years):", flush=True)
    for r in viable_pos[:20]:
        print(f"  {r['portfolio']:<18s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.1f} "
              f"{r['regime']:<6s} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        pos = "ALL+" if all_positive(r) else ""
        print(f"    {r['portfolio']:<18s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}% "
              f"(Top={r['top_n']}, Reb={r['rebal']}, ATR={r['atr']:.1f}) {pos}", flush=True)

    # Top 3 year-by-year
    for i, r in enumerate(viable_pos[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['portfolio']} Top={r['top_n']} Reb={r['rebal']} "
              f"ATR={r['atr']:.1f} (Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # ATR sweep for BodyNW Top=2
    print(f"\n  === BodyNW ATR SWEEP (Top=2, Reb=10) ===", flush=True)
    for r in results:
        if r['portfolio'] == 'BodyNW' and r['top_n'] == 2 and r['rebal'] == 10 and r['regime'] == 'off':
            print(f"    ATR={r['atr']:.1f} → {r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    # Top=1 results
    print(f"\n  === Top=1 Results ===", flush=True)
    top1 = [r for r in results if r['top_n'] == 1]
    for r in top1[:10]:
        print(f"    {r['portfolio']:<18s} ATR={r['atr']:.1f} Reb={r['rebal']} → "
              f"{r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
