"""
Alpha V7h — Re-test with BUG-FIXED backtest
=============================================
Bug fix: ATR止损不再使用look-ahead
  - 旧: 用C[si,di]检查止损, 以O[si,di]卖出 (look-ahead!)
  - 新: 用L[si,di]检查止损, 以stop price卖出
  - hw用H[si,di]更新 (用最高价而非收盘价)

测试同样的BodyNW组合看真实表现
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
from alpha_v7c import backtest_v7c  # BUG-FIXED version


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V7h — BUG-FIXED Re-test", flush=True)
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

    portfolios = {
        'BodyNW': {'R_BODY_NW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'BNW_F': {'R_BODY_NW': 0.3, 'R_TENSION': 0.3,
                  'R_FISHER': 0.2, 'R_SMA_DEV': 0.2},
        'BNW_SQ': {'R_BODY_NW': 0.3, 'R_TENS_SQ': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'BNW_BV': {'R_BODY_NW': 0.3, 'R_BODY_VOL': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'BNW_H': {'R_BODY_NW': 0.4, 'R_TENSION': 0.2,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'StructQual': {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                       'R_TENS_SHAD': 0.25, 'R_BODY_VOL': 0.25},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [1, 2, 3]:
            for rebal in [5, 7, 10]:
                for atr in [0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr)
                    if r:
                        r.update({'portfolio': pname, 'top_n': top_n,
                                  'rebal': rebal, 'atr': atr})
                        results.append(r)
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 30 (BUG-FIXED)", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} | "
          f"{'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*90}", flush=True)
    for r in results[:30]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.1f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['portfolio']:<15s} Top={r['top_n']} Reb={r['rebal']} ATR={r['atr']:.1f} → "
              f"{r['ann']:+.1f}%DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Top 3 year-by-year
    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['portfolio']} Top={r['top_n']} Reb={r['rebal']} "
              f"ATR={r['atr']:.1f} (Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
