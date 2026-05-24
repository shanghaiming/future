"""
Alpha V7f — Fine-Tuning + Advanced Interactions
================================================
V7e发现: SQ_SMA10 DD=50.5%, StructQual DD=51.5%
新因子: NW_SLOPE(+74.6%), KINETIC_EMA(+44.6%)

方向:
  1. 细致权重调优 (5%-15% SMA_DEV)
  2. 更细的ATR止损 (1.0-2.0步进0.1)
  3. 条件因子: R² × TENSION — 只在趋势质量高时交易结构张力
  4. Regime filter结合
  5. Top=2 尝试 (更集中)
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
from alpha_v7c import backtest_v7c, compute_market_regime


def compute_advanced_interactions(factors, NS, ND):
    """Compute advanced interaction factors."""
    t0 = time.time()
    new = {}

    # R² × TENSION — only high-quality trends with structural displacement
    r2 = factors.get('R_R_SQUARED', np.full((NS, ND), np.nan))
    tens = factors.get('R_TENSION', np.full((NS, ND), np.nan))
    R2_TENS = np.full((NS, ND), np.nan)
    mask = ~np.isnan(r2) & ~np.isnan(tens)
    R2_TENS[mask] = r2[mask] * tens[mask] / 100  # normalized product
    new['R_R2_TENS'] = R2_TENS

    # TENSION² — reward extreme tension disproportionately
    new['R_TENS_SQ'] = np.full((NS, ND), np.nan)
    mask = ~np.isnan(tens)
    new['R_TENS_SQ'][mask] = (tens[mask] / 100) ** 2 * 100

    # Body × NW_SLOPE — directional conviction + smooth trend
    body = factors.get('R_BODY_RATIO', np.full((NS, ND), np.nan))
    nw = factors.get('R_NW_SLOPE', np.full((NS, ND), np.nan))
    BODY_NW = np.full((NS, ND), np.nan)
    mask = ~np.isnan(body) & ~np.isnan(nw)
    BODY_NW[mask] = body[mask] * nw[mask] / 100
    new['R_BODY_NW'] = BODY_NW

    # SMA_DEV × TENSION — mean reversion at structural extremes
    sma_dev = factors.get('R_SMA_DEV', np.full((NS, ND), np.nan))
    SMA_TENS = np.full((NS, ND), np.nan)
    mask = ~np.isnan(sma_dev) & ~np.isnan(tens)
    SMA_TENS[mask] = sma_dev[mask] * tens[mask] / 100
    new['R_SMA_TENS'] = SMA_TENS

    # Rank normalize new interactions
    def rank_pct(arr, start=60):
        res = np.full_like(arr, np.nan)
        for di in range(start, arr.shape[1]):
            vals = arr[:, di]
            m = ~np.isnan(vals)
            if m.sum() < 50:
                continue
            ranked = np.argsort(np.argsort(vals[m])).astype(float)
            n = len(ranked)
            pct = ranked / max(n - 1, 1) * 100
            for k, idx in enumerate(np.where(m)[0]):
                res[idx, di] = pct[k]
        return res

    for name in list(new.keys()):
        if not name.startswith('R_'):
            new[f'R_{name}'] = rank_pct(new[name])

    print(f"  Advanced interactions done ({time.time()-t0:.1f}s)", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V7f — Fine-Tuning + Advanced Interactions", flush=True)
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

    # Compute market regime for regime-filtered tests
    print("[Regime] Computing market regime...", flush=True)
    mkt_ret20, mkt_ret5, breadth = compute_market_regime(NS, ND, C, O, H, L, V, syms)
    print("  Market regime done", flush=True)

    # Fine-grained SMA_DEV weight tests
    portfolios = {
        # SMA_DEV weight sweep: 0%, 5%, 7%, 10%, 12%, 15%
        'SQ_base': {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                    'R_TENS_SHAD': 0.25, 'R_BODY_VOL': 0.25},
        'SQ_SMA05': {'R_TENSION': 0.2375, 'R_R_SQUARED': 0.2375,
                     'R_TENS_SHAD': 0.2375, 'R_BODY_VOL': 0.2375,
                     'R_SMA_DEV': 0.05},
        'SQ_SMA07': {'R_TENSION': 0.2325, 'R_R_SQUARED': 0.2325,
                     'R_TENS_SHAD': 0.2325, 'R_BODY_VOL': 0.2325,
                     'R_SMA_DEV': 0.07},
        'SQ_SMA10': {'R_TENSION': 0.225, 'R_R_SQUARED': 0.225,
                     'R_TENS_SHAD': 0.225, 'R_BODY_VOL': 0.225,
                     'R_SMA_DEV': 0.10},
        'SQ_SMA12': {'R_TENSION': 0.22, 'R_R_SQUARED': 0.22,
                     'R_TENS_SHAD': 0.22, 'R_BODY_VOL': 0.22,
                     'R_SMA_DEV': 0.12},
        'SQ_SMA15': {'R_TENSION': 0.2125, 'R_R_SQUARED': 0.2125,
                     'R_TENS_SHAD': 0.2125, 'R_BODY_VOL': 0.2125,
                     'R_SMA_DEV': 0.15},
        # Advanced interactions
        'R2_TENS': {'R_R2_TENS': 0.3, 'R_BODY_VOL': 0.3,
                    'R_TENS_SHAD': 0.2, 'R_SMA_DEV': 0.2},
        'TensSq': {'R_TENS_SQ': 0.3, 'R_R_SQUARED': 0.3,
                   'R_BODY_VOL': 0.2, 'R_SMA_DEV': 0.2},
        'BodyNW': {'R_BODY_NW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'SmaTens': {'R_SMA_TENS': 0.3, 'R_R_SQUARED': 0.3,
                    'R_BODY_VOL': 0.2, 'R_TENS_SHAD': 0.2},
        # Best with R2_TENS replacing individual R2+TENSION
        'Q3_R2T': {'R_R2_TENS': 0.35, 'R_TENS_SHAD': 0.25,
                   'R_BODY_VOL': 0.25, 'R_SMA_DEV': 0.15},
        # NW slope replacing TENSION (smooth trend vs structural)
        'NW_Struct': {'R_NW_SLOPE': 0.25, 'R_R_SQUARED': 0.25,
                      'R_TENS_SHAD': 0.25, 'R_BODY_VOL': 0.25},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [2, 3]:
            for rebal in [7, 10]:
                for atr in [1.2, 1.3, 1.5, 1.8]:
                    # Without regime
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr,
                                    market_ret20=mkt_ret20, breadth=breadth,
                                    regime_filter=False)
                    if r:
                        r.update({'portfolio': pname, 'top_n': top_n,
                                  'rebal': rebal, 'atr': atr, 'regime': 'off'})
                        results.append(r)

                    # With regime filter for top combinations
                    if pname in ['SQ_base', 'SQ_SMA10', 'Q3_R2T', 'SmaTens'] and top_n == 3:
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

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS", flush=True)
    print(f"  {'Portfolio':<18s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} {'Reg':<6s} | "
          f"{'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*110}", flush=True)
    for r in results[:40]:
        print(f"  {r['portfolio']:<18s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.1f} "
              f"{r['regime']:<6s} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['tpy']:4.0f} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['portfolio']:<18s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}% "
              f"(Top={r['top_n']}, Reb={r['rebal']}, ATR={r['atr']:.1f}, Reg={r['regime']})", flush=True)

    # Top 3 year-by-year
    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['portfolio']} (Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # SMA_DEV weight sweep comparison
    print(f"\n  === SMA_DEV WEIGHT SWEEP (Top=3, Reb=10, ATR=1.5) ===", flush=True)
    sweep_names = ['SQ_base', 'SQ_SMA05', 'SQ_SMA07', 'SQ_SMA10', 'SQ_SMA12', 'SQ_SMA15']
    for r in results:
        if r['portfolio'] in sweep_names and r['top_n'] == 3 and r['rebal'] == 10 and r['atr'] == 1.5 and r['regime'] == 'off':
            print(f"    {r['portfolio']:<18s} → {r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
