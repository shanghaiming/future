"""
Alpha V7d — Add Agent-Discovered Factors
=========================================
从15个agent学习中提取的新因子:
  1. SMA_DEV: 多层SMA偏差 (custom_4ma_deviation_strategy)
  2. ATR_RATIO: 波动率压缩/扩张 (volatility_terrain_strategy)

目标: 在保持+160%年化的基础上降低DD
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0


def _rolling_sma_std(arr, period):
    """Compute rolling SMA and std using cumsum — O(N) per array."""
    n = len(arr)
    valid = ~np.isnan(arr) & (arr > 0)
    filled = np.where(valid, arr, 0.0)
    cs = np.cumsum(filled)
    cs2 = np.cumsum(filled ** 2)
    cv = np.cumsum(valid.astype(float))

    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        cnt = cv[i] - (cv[i - period] if i >= period else 0)
        if cnt < period * 0.7:
            continue
        s = cs[i] - (cs[i - period] if i >= period else 0)
        s2 = cs2[i] - (cs2[i - period] if i >= period else 0)
        m = s / cnt
        var = s2 / cnt - m * m
        if var > 0:
            sma[i] = m
            std[i] = np.sqrt(var)
    return sma, std


def _rolling_mean_1d(arr, period):
    """Fast rolling mean on 1D array handling NaN."""
    n = len(arr)
    valid = ~np.isnan(arr)
    filled = np.where(valid, arr, 0.0)
    cs = np.cumsum(filled)
    cv = np.cumsum(valid.astype(float))
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        cnt = cv[i] - (cv[i - period] if i >= period else 0)
        if cnt < period * 0.5:
            continue
        s = cs[i] - (cs[i - period] if i >= period else 0)
        result[i] = s / cnt
    return result


def compute_extra_factors(NS, ND, C, O, H, L, V):
    """Extra factors discovered by study agents — VECTORIZED."""
    t0 = time.time()
    new = {}

    # === SMA_DEV: Multi-SMA deviation — vectorized ===
    # For each period, pre-compute rolling SMA/std, then z-score, then rolling percentile
    SMA_DEV = np.full((NS, ND), np.nan)
    for si in range(NS):
        prices = C[si, :].copy()
        z_scores = []  # collect z-scores per period
        for period in [16, 50, 100]:
            sma, std = _rolling_sma_std(prices, period)
            # z-score for each day
            z = np.full(ND, np.nan)
            mask = ~np.isnan(sma) & ~np.isnan(std) & (std > 0) & ~np.isnan(prices)
            z[mask] = (prices[mask] - sma[mask]) / std[mask]
            # Rolling percentile rank of z over last 60 days
            # NO LOOK-AHEAD: store z[di-1] at index di (backtest reads factor at di)
            pct = np.full(ND, np.nan)
            for di in range(period + 61, ND):
                # Use z up to di-1 (yesterday's z-score)
                window_z = z[max(period, di - 61):di - 1]
                valid_z = window_z[~np.isnan(window_z)]
                if len(valid_z) < 10:
                    continue
                z_yesterday = z[di - 1]
                if not np.isnan(z_yesterday):
                    pct[di] = np.sum(valid_z < z_yesterday) / max(len(valid_z) - 1, 1) * 100
            z_scores.append(pct)

        # Average percentile across periods
        valid_count = np.zeros(ND)
        pct_sum = np.zeros(ND)
        for zs in z_scores:
            mask = ~np.isnan(zs)
            pct_sum[mask] += zs[mask]
            valid_count[mask] += 1
        has_data = valid_count > 0
        SMA_DEV[si, has_data] = pct_sum[has_data] / valid_count[has_data]

        if si % 100 == 0 and si > 0:
            print(f"    SMA_DEV stock {si}/{NS} ({time.time()-t0:.1f}s)", flush=True)

    new['SMA_DEV'] = SMA_DEV
    print(f"  SMA deviation done ({time.time()-t0:.1f}s)", flush=True)

    # === ATR_RATIO: Fast ATR(7) / Slow ATR(28) — fully vectorized ===
    # Pre-compute True Range matrix
    TR = np.full((NS, ND), np.nan)
    # Basic TR = H - L
    hl = H[:, 1:] - L[:, 1:]
    TR[:, 1:] = np.where(~np.isnan(hl), hl, np.nan)
    # Adjust for gaps: TR = max(HL, |H-Cp|, |L-Cp|)
    for di in range(1, ND):
        gap1 = np.abs(H[:, di] - C[:, di - 1])
        gap2 = np.abs(L[:, di] - C[:, di - 1])
        tr_base = TR[:, di]
        g1 = np.where(~np.isnan(gap1), gap1, 0)
        g2 = np.where(~np.isnan(gap2), gap2, 0)
        mx = np.maximum(g1, g2)
        TR[:, di] = np.where(~np.isnan(tr_base), np.maximum(tr_base, mx), np.nan)

    ATR_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        fast_atr = _rolling_mean_1d(TR[si, :], 7)
        slow_atr = _rolling_mean_1d(TR[si, :], 28)
        mask = ~np.isnan(fast_atr) & ~np.isnan(slow_atr) & (slow_atr > 0)
        # NO LOOK-AHEAD: shift by 1 — store di-1's ratio at index di
        raw_ratio = np.full(ND, np.nan)
        raw_ratio[mask] = fast_atr[mask] / slow_atr[mask]
        ATR_RATIO[si, 1:] = raw_ratio[:-1]  # shift: di gets value from di-1

    new['ATR_RATIO'] = ATR_RATIO
    print(f"  ATR ratio done ({time.time()-t0:.1f}s)", flush=True)

    # === Rank normalize ===
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

    new['R_SMA_DEV'] = rank_pct(new['SMA_DEV'])
    new['R_ATR_RATIO'] = rank_pct(new['ATR_RATIO'])
    print(f"  Ranked done ({time.time()-t0:.1f}s)", flush=True)

    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V7d — Agent-Discovered Factors", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
    from alpha_v7b import compute_interaction_factors
    inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)
    extra_factors = compute_extra_factors(NS, ND, C, O, H, L, V)

    all_factors = {**base_factors, **inter_factors, **extra_factors}
    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # Test new factors individually and in combination
    from alpha_v7c import backtest_v7c

    # Single factor tests for new ones
    for fname in ['R_SMA_DEV', 'R_ATR_RATIO']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname}: Ann={r['ann']:+.1f}% WR={r['wr']:.0f}% DD={r['max_dd']:.1f}%", flush=True)

    # Combinations with new factors
    portfolios = {
        # Original best
        'StructQual': {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                       'R_TENS_SHAD': 0.25, 'R_BODY_VOL': 0.25},
        # Add SMA deviation
        'SQ_SMA': {'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_TENS_SHAD': 0.2, 'R_BODY_VOL': 0.2,
                   'R_SMA_DEV': 0.2},
        # Add ATR ratio
        'SQ_ATR': {'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_TENS_SHAD': 0.2, 'R_BODY_VOL': 0.2,
                   'R_ATR_RATIO': 0.2},
        # Both new factors
        'SQ_Both': {'R_TENSION': 0.15, 'R_R_SQUARED': 0.15,
                    'R_TENS_SHAD': 0.15, 'R_BODY_VOL': 0.15,
                    'R_SMA_DEV': 0.2, 'R_ATR_RATIO': 0.2},
        # SMA deviation heavy
        'SMA_Heavy': {'R_SMA_DEV': 0.4, 'R_TENSION': 0.2,
                      'R_R_SQUARED': 0.2, 'R_BODY_VOL': 0.2},
        # ATR ratio heavy (volatility breakout)
        'ATR_Heavy': {'R_ATR_RATIO': 0.4, 'R_TENSION': 0.2,
                      'R_R_SQUARED': 0.2, 'R_BODY_VOL': 0.2},
        # FisherStruct (V7c second best)
        'FisherStruct': {'R_FISHER': 0.3, 'R_TENSION': 0.3,
                         'R_R_SQUARED': 0.2, 'R_VOLATILITY_PCT': 0.2},
        # FisherStruct + new
        'FS_Both': {'R_FISHER': 0.2, 'R_TENSION': 0.2,
                    'R_R_SQUARED': 0.15, 'R_VOLATILITY_PCT': 0.15,
                    'R_SMA_DEV': 0.15, 'R_ATR_RATIO': 0.15},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [3, 5]:
            for rebal in [10, 15]:
                for atr in [1.5, 2.0]:
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr)
                    if r:
                        r.update({'portfolio': pname, 'top_n': top_n,
                                  'rebal': rebal, 'atr': atr})
                        results.append(r)
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 30 RESULTS", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} | "
          f"{'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for r in results[:30]:
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.1f} | "
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
        print(f"    {r['portfolio']:<15s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}% "
              f"(Top={r['top_n']}, Reb={r['rebal']}, ATR={r['atr']:.1f})", flush=True)

    # Year-by-year for best
    if results:
        best = results[0]
        print(f"\n  Year-by-year: {best['portfolio']} (Ann={best['ann']:+.1f}%, DD={best['max_dd']:.1f}%)", flush=True)
        for y in sorted(best.get('year_stats', {}).keys()):
            s = best['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
