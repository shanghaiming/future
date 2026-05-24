"""
Alpha V59 — V56 Optimization: More Decomposition Scales
========================================================
V56 verified: +1630.7% with SHOCK_MOM + TREND_ACC

Now: try more EMA windows, multi-scale shocks, non-linear interactions.
Target: push beyond +1700%.
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


def ema(arr, span):
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


def compute_multiscale_factors(NS, ND, C, O, H, L, V):
    """Compute multi-scale decomposition factors across many EMA windows."""
    factors = {}
    t0 = time.time()

    # Returns
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[mask, di] = (C[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    # ATR
    tr_arr = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(H[:, di]) & ~np.isnan(L[:, di])
        tr_arr[m, di] = H[m, di] - L[m, di]
        m2 = m & (~np.isnan(C[:, di - 1]))
        tr_arr[m2, di] = np.maximum(tr_arr[m2, di],
                                     np.maximum(np.abs(H[m2, di] - C[m2, di - 1]),
                                                np.abs(L[m2, di] - C[m2, di - 1])))
    atr = _rolling_mean(tr_arr, 14, min_valid=7)

    # =================================================================
    # Multi-scale shock: try EMA spans 5, 10, 20, 40, 60
    # =================================================================
    for span in [5, 10, 20, 40, 60]:
        ema_c = ema(C, span)
        # Shock = (C - EMA) / ATR
        shock = np.full((NS, ND), np.nan)
        mask = ~np.isnan(C) & ~np.isnan(ema_c) & ~np.isnan(atr) & (atr > 1e-6)
        shock[mask] = (C[mask] - ema_c[mask]) / atr[mask]

        # Shock momentum (5-day mean)
        shock_mom = _rolling_mean(shock, 5)
        factors[f'R_SHOCK_MOM_E{span}'] = _rank_normalize(shock_mom)

        # Shock reversal
        shock_rev = _rolling_mean(-shock, 5)
        factors[f'R_SHOCK_REV_E{span}'] = _rank_normalize(shock_rev)

        # Shock 10d
        shock_mom10 = _rolling_mean(shock, 10)
        factors[f'R_SHOCK_MOM10_E{span}'] = _rank_normalize(shock_mom10)

    # =================================================================
    # Multi-scale trend: EMA crossovers
    # =================================================================
    for fast, slow in [(3, 20), (5, 20), (5, 40), (5, 60), (10, 40), (10, 60), (20, 60)]:
        ema_f = ema(C, fast)
        ema_s = ema(C, slow)

        # Trend strength: normalized gap
        trend_str = np.full((NS, ND), np.nan)
        mask = ~np.isnan(ema_f) & ~np.isnan(ema_s) & (ema_s > 0)
        trend_str[mask] = (ema_f[mask] - ema_s[mask]) / ema_s[mask]
        factors[f'R_TREND_F{fast}_S{slow}'] = _rank_normalize(trend_str)

        # Trend acceleration
        trend_acc = np.full((NS, ND), np.nan)
        for di in range(6, ND):
            m = ~np.isnan(trend_str[:, di]) & ~np.isnan(trend_str[:, di - 5])
            trend_acc[m, di] = trend_str[m, di] - trend_str[m, di - 5]
        factors[f'R_TREND_ACC_F{fast}_S{slow}'] = _rank_normalize(trend_acc)

    # =================================================================
    # Volume-price divergence
    # =================================================================
    # Price up but volume down = weakening
    ret_5 = _rolling_mean(ret, 5)
    vol_chg = np.full((NS, ND), np.nan)
    for di in range(5, ND):
        mask = ~np.isnan(V[:, di]) & ~np.isnan(V[:, di - 5]) & (V[:, di - 5] > 0)
        vol_chg[mask, di] = (V[mask, di] - V[mask, di - 5]) / V[mask, di - 5]
    vol_chg_5 = _rolling_mean(vol_chg, 5)

    # Price-volume divergence: price up + volume down = bearish divergence
    pv_div = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ret_5) & ~np.isnan(vol_chg_5)
    pv_div[mask] = np.sign(ret_5[mask]) * np.sign(vol_chg_5[mask])
    factors['R_PV_CONCORD'] = _rank_normalize(pv_div)

    # Pure price-volume correlation (20d rolling)
    pv_corr = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        chunk_ret = ret[:, di - 20:di]
        chunk_vol = vol_chg[:, di - 20:di]
        for si in range(NS):
            r = chunk_ret[si]
            v = chunk_vol[si]
            valid = ~np.isnan(r) & ~np.isnan(v)
            if valid.sum() >= 10:
                pv_corr[si, di] = np.corrcoef(r[valid], v[valid])[0, 1]
    factors['R_PV_CORR'] = _rank_normalize(pv_corr)

    # =================================================================
    # Intraday range ratio (high-low relative to close)
    # =================================================================
    hl_pct = np.full((NS, ND), np.nan)
    mask = ~np.isnan(H) & ~np.isnan(L) & ~np.isnan(C) & (C > 0)
    hl_pct[mask] = (H[mask] - L[mask]) / C[mask]
    hl_pct_ma = _rolling_mean(hl_pct, 20)
    factors['R_HL_RANGE'] = _rank_normalize(-hl_pct_ma)  # Low range = tight = good

    # Range expansion: today's range vs 20d average
    range_exp = np.full((NS, ND), np.nan)
    mask = ~np.isnan(hl_pct) & ~np.isnan(hl_pct_ma) & (hl_pct_ma > 0)
    range_exp[mask] = hl_pct[mask] / hl_pct_ma[mask]
    factors['R_RANGE_EXP'] = _rank_normalize(-range_exp)  # Contraction = good

    # =================================================================
    # Close position in range (where does close sit within H-L?)
    # =================================================================
    close_pos = np.full((NS, ND), np.nan)
    mask = ~np.isnan(H) & ~np.isnan(L) & (H > L)
    close_pos[mask] = (C[mask] - L[mask]) / (H[mask] - L[mask])
    close_pos_ma = _rolling_mean(close_pos, 10)
    factors['R_CLOSE_POS'] = _rank_normalize(close_pos_ma)

    # =================================================================
    # Gap analysis (open vs previous close)
    # =================================================================
    gap = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(O[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        gap[mask, di] = (O[mask, di] - C[mask, di - 1]) / C[mask, di - 1]
    gap_5 = _rolling_mean(gap, 5)
    factors['R_GAP_MOM'] = _rank_normalize(gap_5)
    factors['R_GAP_REV'] = _rank_normalize(-gap_5)

    # =================================================================
    # Non-linear: factor squared (volatility of factor)
    # =================================================================
    # Volatility of return (20d std)
    ret_std = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        chunk = ret[:, di - 20:di]
        for si in range(NS):
            valid = chunk[si][~np.isnan(chunk[si])]
            if len(valid) >= 10:
                ret_std[si, di] = valid.std()
    factors['R_RET_VOL'] = _rank_normalize(-ret_std)  # Low vol = good

    # Skewness of return (20d)
    ret_skew = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        chunk = ret[:, di - 20:di]
        for si in range(NS):
            valid = chunk[si][~np.isnan(chunk[si])]
            if len(valid) >= 10:
                m = valid.mean()
                s = valid.std()
                if s > 1e-10:
                    ret_skew[si, di] = ((valid - m) ** 3).mean() / s ** 3
    factors['R_RET_SKEW'] = _rank_normalize(ret_skew)

    print(f"  Multi-scale factors done: {len(factors)} factors ({time.time()-t0:.0f}s)", flush=True)
    return factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V59 — Multi-Scale Optimization")
    print("  V56 verified: +1630.7% DD=25.2%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing all factors...", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)
    v59 = compute_multiscale_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41, **v48, **v49, **v52, **v55, **v59}

    # V56 winning weights
    v54_base = {'R_BWP_BNW': 0.205, 'R_TENSION': 0.205, 'R_VWCM': 0.205,
                'R_BVR': 0.154, 'R_BUY_FRAC': 0.138, 'R_VPIN': 0.092}
    v56_weights = {**v54_base, 'R_SHOCK_MOM': 0.08, 'R_TREND_ACC': 0.15}
    total = sum(v56_weights.values())
    v56_norm = {k: v / total for k, v in v56_weights.items()}

    results = []

    # =====================================================================
    # Baseline
    # =====================================================================
    print("\n  V56 baseline...", flush=True)
    r = backtest_v7c(v56_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V56_BASE'
        results.append(r)
        print(f"  V56: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Solo factors
    # =====================================================================
    v59_names = sorted(v59.keys())
    print(f"\n  Testing {len(v59_names)} new factors solo...", flush=True)
    for fname in v59_names:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = f'{fname}_SOLO'
            results.append(r)
            if r['ann'] > 50:
                print(f"    {fname}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # V56 + promising new factors
    # =====================================================================
    promising = sorted([r for r in results if '_SOLO' in r['test'] and r['ann'] > 50],
                       key=lambda x: -x['ann'])[:15]
    promising_names = []
    for r in promising:
        for fname in v59_names:
            if fname in r['test']:
                promising_names.append(fname)
                break

    print(f"\n  Testing V56 + {len(promising_names)} promising factors...", flush=True)
    for fname in promising_names:
        for w in [0.05, 0.08, 0.10]:
            weights = {**v56_norm, fname: w}
            total = sum(weights.values())
            wn = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'V56+{fname[-6:]}_W{w:.2f}'
                results.append(r)

    # =====================================================================
    # Replace SHOCK_MOM with multi-scale versions
    # =====================================================================
    print("\n  Testing multi-scale shock replacements...", flush=True)
    for span in [5, 10, 20, 40, 60]:
        fname = f'R_SHOCK_MOM_E{span}'
        for w_shock in [0.05, 0.08, 0.10]:
            weights = {k: v for k, v in v56_norm.items() if k != 'R_SHOCK_MOM'
                       and k != 'R_TREND_ACC'}
            weights[fname] = w_shock
            weights['R_TREND_ACC'] = v56_norm.get('R_TREND_ACC', 0.12)
            total = sum(weights.values())
            wn = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'REP_SHOCK→E{span}_W{w_shock:.2f}'
                results.append(r)

    # =====================================================================
    # Replace TREND_ACC with multi-scale versions
    # =====================================================================
    print("\n  Testing multi-scale trend replacements...", flush=True)
    for fast, slow in [(3, 20), (5, 40), (5, 60), (10, 40), (10, 60), (20, 60)]:
        fname = f'R_TREND_ACC_F{fast}_S{slow}'
        for w_acc in [0.08, 0.10, 0.15]:
            weights = {k: v for k, v in v56_norm.items() if k != 'R_TREND_ACC'
                       and k != 'R_SHOCK_MOM'}
            weights[fname] = w_acc
            weights['R_SHOCK_MOM'] = v56_norm.get('R_SHOCK_MOM', 0.06)
            total = sum(weights.values())
            wn = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'REP_ACC→F{fast}S{slow}_W{w_acc:.2f}'
                results.append(r)

    # =====================================================================
    # Best V56 + 2 new factors
    # =====================================================================
    print("\n  Testing V56 + 2 new factors...", flush=True)
    if len(promising_names) >= 2:
        top2 = promising_names[:2]
        for w1 in [0.05, 0.08]:
            for w2 in [0.05, 0.08]:
                weights = {**v56_norm, top2[0]: w1, top2[1]: w2}
                total = sum(weights.values())
                wn = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                if r:
                    r['test'] = f'V56+2NEW_{top2[0][-4:]}+{top2[1][-4:]}'
                    results.append(r)

    # =====================================================================
    # ATR sweep with best combo
    # =====================================================================
    print("\n  ATR sweep...", flush=True)
    results_sorted = sorted(results, key=lambda x: -x['ann'])
    # Use top result's weights for ATR sweep
    # Just test a few ATR values
    for fname in promising_names[:3]:
        weights = {**v56_norm, fname: 0.08}
        total = sum(weights.values())
        wn = {k: v / total for k, v in weights.items()}
        for atr in [0.3, 0.4, 0.5, 0.6]:
            r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'V56+{fname[-4:]}_A{atr}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*100}", flush=True)
    print(f"  ALL RESULTS (V59 MULTI-SCALE OPTIMIZATION)", flush=True)
    print(f"  {'Test':<45s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*85}", flush=True)
    for r in results[:60]:
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
        print(f"\n  === V59 BEST ===", flush=True)
        print(f"  V59: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V56 RECORD: +1630.7% DD=25.2%", flush=True)
        delta = best['ann'] - 1630.7
        print(f"  Delta from V56: {delta:+.1f}%", flush=True)

        print(f"\n  === SOLO FACTOR SUMMARY ===", flush=True)
        solo = sorted([r for r in results if '_SOLO' in r['test']], key=lambda x: -x['ann'])
        for r in solo[:20]:
            pos_mark = " ALL+" if all_positive(r) else ""
            print(f"  {r['test']:<45s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    print(f"\n{'='*70}", flush=True)
