"""
Alpha V55 — ACT Temporal Decomposition (arXiv 2604.20204)
==========================================================
ACT: Anti-Crosstalk Learning for Cross-Sectional Stock Ranking

Key idea: Decompose each stock's price/volume series into:
  1. TREND component: slow-moving EMA (captures regime direction)
  2. FLUCTUATION component: price minus EMA (captures cyclical swings)
  3. SHOCK component: high-frequency residual / ATR (captures sudden moves)

Then compute factors on EACH component separately → 3x factor space.

V54 base: +1203.8% with BWP_BNW + TENSION + VWCM + BVR + BUY_FRAC + VPIN
V55 adds: decomposed versions of each factor → richer signal space
V56 best: +1450.0% with V54 + R_SHOCK_MOM(0.08) + R_TREND_ACC(0.15)
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


def _rolling_std(arr, window, min_valid=None):
    if min_valid is None:
        min_valid = window // 2
    mean = _rolling_mean(arr, window, min_valid)
    NS, ND = arr.shape
    out = np.full_like(arr, np.nan)
    for di in range(window, ND):
        m = mean[:, di]
        if np.all(np.isnan(m)):
            continue
        chunk = arr[:, di - window:di]
        sq_diff = (chunk - m[:, np.newaxis]) ** 2
        sq_diff = np.where(np.isnan(sq_diff), 0, sq_diff)
        n_valid = np.sum(~np.isnan(chunk), axis=1)
        valid = n_valid >= min_valid
        ss = np.sum(sq_diff, axis=1)
        out[valid, di] = np.sqrt(ss[valid] / n_valid[valid])
    return out


def ema(arr, span):
    """Exponential moving average along axis=1. No look-ahead: uses arr[:, di-1]."""
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


def compute_decomposed_factors(NS, ND, C, O, H, L, V):
    """Compute ACT-style decomposed factors.

    Decomposition:
    - TREND: EMA(20) of close → slow component
    - FLUCTUATION: Close - EMA(20) → cyclical component
    - SHOCK: (Close - EMA(20)) / ATR(14) → normalized shock

    For each component, compute:
    - Momentum (5d, 10d)
    - Volume-weighted direction (VWCM-like)
    - Buy fraction (persistency)
    - Order flow imbalance (OFI-like)
    - Buyer volume ratio (BVR-like)
    """
    factors = {}
    t_total = time.time()

    # Returns
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[mask, di] = (C[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    # HL range
    hl_range = np.full((NS, ND), np.nan)
    mask = ~np.isnan(H) & ~np.isnan(L)
    hl_range[mask] = H[mask] - L[mask]
    safe_hl = np.where(hl_range > 1e-6, hl_range, np.nan)

    # ATR(14) — vectorized True Range + rolling mean
    tr_arr = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(H[:, di]) & ~np.isnan(L[:, di])
        tr_arr[m, di] = H[m, di] - L[m, di]
        m2 = m & (~np.isnan(C[:, di - 1]))
        tr_arr[m2, di] = np.maximum(tr_arr[m2, di],
                                     np.maximum(np.abs(H[m2, di] - C[m2, di - 1]),
                                                np.abs(L[m2, di] - C[m2, di - 1])))
    atr = _rolling_mean(tr_arr, 14, min_valid=7)
    print(f"  ATR done ({time.time()-t_total:.0f}s)", flush=True)

    # =================================================================
    # DECOMPOSITION
    # =================================================================
    # TREND: EMA(20) of Close
    trend_c = ema(C, 20)
    # FLUCTUATION: Close - Trend
    fluct_c = np.full((NS, ND), np.nan)
    mask = ~np.isnan(C) & ~np.isnan(trend_c)
    fluct_c[mask] = C[mask] - trend_c[mask]
    # SHOCK: Fluctuation / ATR (normalized)
    shock_c = np.full((NS, ND), np.nan)
    mask = ~np.isnan(fluct_c) & ~np.isnan(atr) & (atr > 1e-6)
    shock_c[mask] = fluct_c[mask] / atr[mask]

    # Also decompose volume
    # Volume trend
    trend_v = ema(V, 20)
    # Volume fluctuation: V - EMA(V)
    fluct_v = np.full((NS, ND), np.nan)
    mask = ~np.isnan(V) & ~np.isnan(trend_v)
    fluct_v[mask] = V[mask] - trend_v[mask]

    # Also decompose H and L for range-based factors
    trend_h = ema(H, 20)
    trend_l = ema(L, 20)

    print(f"  Decomposition done ({time.time()-t_total:.0f}s)", flush=True)

    # =================================================================
    # TREND FACTORS (slow-moving, regime-like signals)
    # =================================================================
    # Trend momentum: 20d return of the trend component
    trend_mom = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        mask = ~np.isnan(trend_c[:, di - 1]) & ~np.isnan(trend_c[:, di - 21]) & (trend_c[:, di - 21] > 0)
        trend_mom[mask, di] = (trend_c[mask, di - 1] - trend_c[mask, di - 21]) / trend_c[mask, di - 21]
    factors['R_TREND_MOM'] = _rank_normalize(trend_mom)

    # Trend slope: 10d linear regression slope of trend_c
    trend_slope = np.full((NS, ND), np.nan)
    x = np.arange(10, dtype=float)
    x_mean = x.mean()
    x_ss = ((x - x_mean) ** 2).sum()
    for di in range(10, ND):
        chunk = trend_c[:, di - 10:di]
        n_valid = (~np.isnan(chunk)).sum(axis=1)
        valid = n_valid >= 7
        for si in np.where(valid)[0]:
            vals = chunk[si]
            v = vals[~np.isnan(vals)]
            if len(v) < 7:
                continue
            xv = x[:len(v)]
            xv_m = xv - xv.mean()
            slope = (xv_m * (v - v.mean())).sum() / ((xv_m ** 2).sum() + 1e-10)
            trend_slope[si, di] = slope
    factors['R_TREND_SLOPE'] = _rank_normalize(trend_slope)
    print(f"  Trend factors done ({time.time()-t_total:.0f}s)", flush=True)

    # =================================================================
    # FLUCTUATION FACTORS (cyclical, mean-reverting signals)
    # =================================================================
    # Fluctuation momentum: sum of fluctuation over 5 days
    fluct_mom = _rolling_mean(fluct_c, 5)
    factors['R_FLUCT_MOM'] = _rank_normalize(fluct_mom)

    # Fluctuation reversal: negative of fluctuation (buy when oversold)
    fluct_rev = -fluct_mom
    factors['R_FLUCT_REV'] = _rank_normalize(fluct_rev)

    # Fluctuation volatility: std of fluctuation over 20 days
    fluct_vol = _rolling_std(fluct_c, 20)
    factors['R_FLUCT_VOL'] = _rank_normalize(-fluct_vol)  # Low vol = better

    # Fluctuation with volume confirmation
    fluct_vol_conf = np.full((NS, ND), np.nan)
    mask = ~np.isnan(fluct_c) & ~np.isnan(V) & ~np.isnan(trend_v) & (trend_v > 0)
    fluct_vol_conf[mask] = fluct_c[mask] * np.sign(V[mask] - trend_v[mask])
    fvc_10 = _rolling_mean(fluct_vol_conf, 10)
    factors['R_FLUCT_VOL_CONF'] = _rank_normalize(fvc_10)
    print(f"  Fluctuation factors done ({time.time()-t_total:.0f}s)", flush=True)

    # =================================================================
    # SHOCK FACTORS (sudden moves, event-driven)
    # =================================================================
    # Shock momentum: mean shock over 5 days
    shock_mom = _rolling_mean(shock_c, 5)
    factors['R_SHOCK_MOM'] = _rank_normalize(shock_mom)

    # Shock reversal: buy on negative shock (oversold)
    shock_rev = _rolling_mean(-shock_c, 5)
    factors['R_SHOCK_REV'] = _rank_normalize(shock_rev)

    # Shock with volume: large shock + high volume = institutional
    shock_vol = np.full((NS, ND), np.nan)
    mask = ~np.isnan(shock_c) & ~np.isnan(V) & ~np.isnan(trend_v) & (trend_v > 0)
    shock_vol[mask] = np.abs(shock_c[mask]) * (V[mask] / trend_v[mask])
    shock_vol_5 = _rolling_mean(shock_vol, 5)
    factors['R_SHOCK_VOL'] = _rank_normalize(shock_vol_5)
    print(f"  Shock factors done ({time.time()-t_total:.0f}s)", flush=True)

    # =================================================================
    # CROSS-COMPONENT INTERACTIONS
    # =================================================================
    # Trend-Fluctuation alignment: trend positive AND fluctuation positive
    tf_align = np.full((NS, ND), np.nan)
    mask = ~np.isnan(trend_slope) & ~np.isnan(fluct_mom)
    # Normalize both to similar scale
    tf_align[mask] = np.sign(trend_slope[mask]) * fluct_mom[mask]
    factors['R_TF_ALIGN'] = _rank_normalize(tf_align)

    # Trend-Shock divergence: trend up but shock negative (dip in uptrend = buy)
    ts_div = np.full((NS, ND), np.nan)
    mask = ~np.isnan(trend_slope) & ~np.isnan(shock_mom)
    ts_div[mask] = -np.sign(trend_slope[mask]) * shock_mom[mask]
    factors['R_TS_DIVERGE'] = _rank_normalize(ts_div)

    # Volume shock vs price shock: vol spike without price move = accumulation
    vol_shock = np.full((NS, ND), np.nan)
    mask = ~np.isnan(fluct_v) & ~np.isnan(atr) & (atr > 0) & ~np.isnan(fluct_c)
    vol_shock[mask] = np.abs(fluct_v[mask]) / (atr[mask] * 10000 + 1) - np.abs(shock_c[mask])
    vs_5 = _rolling_mean(vol_shock, 5)
    factors['R_VOL_SHOCK_PX'] = _rank_normalize(vs_5)
    print(f"  Cross-component done ({time.time()-t_total:.0f}s)", flush=True)

    # =================================================================
    # DECOMPOSED V54 FACTORS
    # =================================================================
    # VWCM on trend component
    v_signed_trend = np.full((NS, ND), np.nan)
    ret_trend = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(trend_c[:, di]) & ~np.isnan(trend_c[:, di - 1]) & (trend_c[:, di - 1] > 0)
        ret_trend[mask, di] = (trend_c[mask, di] - trend_c[mask, di - 1]) / trend_c[mask, di - 1]
    mask = ~np.isnan(V) & ~np.isnan(trend_c) & ~np.isnan(ret_trend) & (V > 0)
    v_signed_trend[mask] = V[mask] * np.sign(ret_trend[mask]) * trend_c[mask]
    vsigned_trend_mean = _rolling_mean(v_signed_trend, 10)
    vc_trend_mean = _rolling_mean(V * trend_c, 10)
    vwcm_trend = np.full((NS, ND), np.nan)
    mask = ~np.isnan(vsigned_trend_mean) & ~np.isnan(vc_trend_mean) & (vc_trend_mean > 0)
    vwcm_trend[mask] = vsigned_trend_mean[mask] / vc_trend_mean[mask]
    factors['R_VWCM_TREND'] = _rank_normalize(vwcm_trend)

    # VWCM on fluctuation component
    v_signed_fluct = np.full((NS, ND), np.nan)
    mask = ~np.isnan(V) & ~np.isnan(fluct_c) & ~np.isnan(ret) & (V > 0)
    v_signed_fluct[mask] = V[mask] * np.sign(ret[mask]) * np.abs(fluct_c[mask])
    vsigned_fluct_mean = _rolling_mean(v_signed_fluct, 10)
    v_abs_fluct = np.where(~np.isnan(fluct_c), V * np.abs(fluct_c), np.nan)
    vf_mean = _rolling_mean(v_abs_fluct, 10)
    vwcm_fluct = np.full((NS, ND), np.nan)
    mask = ~np.isnan(vsigned_fluct_mean) & ~np.isnan(vf_mean) & (vf_mean > 0)
    vwcm_fluct[mask] = vsigned_fluct_mean[mask] / vf_mean[mask]
    factors['R_VWCM_FLUCT'] = _rank_normalize(vwcm_fluct)

    # BUY_FRAC on trend: fraction of days trend_c rises
    buy_trend = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(trend_c[:, di]) & ~np.isnan(trend_c[:, di - 1]) & (trend_c[:, di - 1] > 0)
        buy_trend[mask, di] = (trend_c[mask, di] > trend_c[mask, di - 1]).astype(float)
    bf_trend_20 = _rolling_mean(buy_trend, 20)
    factors['R_BUY_FRAC_TREND'] = _rank_normalize(bf_trend_20)

    # BUY_FRAC on shock: fraction of days with positive shock
    buy_shock = np.full((NS, ND), np.nan)
    mask = ~np.isnan(shock_c)
    buy_shock[mask] = (shock_c[mask] > 0).astype(float)
    bf_shock_10 = _rolling_mean(buy_shock, 10)
    factors['R_BUY_FRAC_SHOCK'] = _rank_normalize(bf_shock_10)
    print(f"  Decomposed V54 factors done ({time.time()-t_total:.0f}s)", flush=True)

    # =================================================================
    # RANGE DECOMPOSITION
    # =================================================================
    # Range in trend (smoothed range)
    range_trend = np.full((NS, ND), np.nan)
    mask = ~np.isnan(trend_h) & ~np.isnan(trend_l)
    range_trend[mask] = trend_h[mask] - trend_l[mask]
    range_trend_10 = _rolling_mean(range_trend, 10)
    # Normalize by price
    range_trend_norm = np.full((NS, ND), np.nan)
    mask = ~np.isnan(range_trend_10) & ~np.isnan(trend_c) & (trend_c > 0)
    range_trend_norm[mask] = range_trend_10[mask] / trend_c[mask]
    factors['R_RANGE_TREND'] = _rank_normalize(-range_trend_norm)  # Low = tight = good

    # Fluctuation range: H-L minus trend range = excess volatility
    excess_range = np.full((NS, ND), np.nan)
    mask = ~np.isnan(hl_range) & ~np.isnan(range_trend)
    excess_range[mask] = hl_range[mask] - range_trend[mask]
    excess_range_10 = _rolling_mean(excess_range, 10)
    factors['R_EXCESS_RANGE'] = _rank_normalize(-excess_range_10)  # Low excess = good
    print(f"  Range decomposition done ({time.time()-t_total:.0f}s)", flush=True)

    # =================================================================
    # MULTI-SCALE TREND (short vs long EMA crossover)
    # =================================================================
    ema5 = ema(C, 5)
    ema60 = ema(C, 60)

    # Trend strength: EMA5 vs EMA60 normalized
    trend_strength = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ema5) & ~np.isnan(ema60) & (ema60 > 0)
    trend_strength[mask] = (ema5[mask] - ema60[mask]) / ema60[mask]
    factors['R_TREND_STR'] = _rank_normalize(trend_strength)

    # Trend acceleration: change in trend strength over 5 days
    trend_acc = np.full((NS, ND), np.nan)
    for di in range(6, ND):
        mask = ~np.isnan(trend_strength[:, di]) & ~np.isnan(trend_strength[:, di - 5])
        trend_acc[mask, di] = trend_strength[mask, di] - trend_strength[mask, di - 5]
    factors['R_TREND_ACC'] = _rank_normalize(trend_acc)

    print(f"  Multi-scale done ({time.time()-t_total:.0f}s)", flush=True)
    print(f"  Total V55 computation: {time.time()-t_total:.0f}s", flush=True)
    return factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V55 — ACT Temporal Decomposition")
    print("  V54 record: +1203.8% DD=31.5% ALL+", flush=True)
    print("  Target: improve with decomposed factor space", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute all existing factors
    print("\n  Computing V41 factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    print("\n  Computing V48+V49 factors...", flush=True)
    v48_factors = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49_factors = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52_factors = compute_v52_factors(NS, ND, C, O, H, L, V)

    # Compute decomposed factors
    print("\n  Computing V55 ACT decomposed factors...", flush=True)
    v55_factors = compute_decomposed_factors(NS, ND, C, O, H, L, V)

    all_factors = {**v41_factors, **v48_factors, **v49_factors, **v52_factors, **v55_factors}

    v55_names = sorted(v55_factors.keys())
    print(f"\n  New V55 factors: {len(v55_names)} — {v55_names}", flush=True)

    # V54 winning base (no SMA_DEV, no R_SQUARED)
    v54_base = {'R_BWP_BNW': 0.178, 'R_TENSION': 0.178, 'R_VWCM': 0.178, 'R_BVR': 0.134}
    v54_winner = {**v54_base, 'R_BUY_FRAC': 0.12, 'R_VPIN': 0.08}

    results = []

    # =====================================================================
    # Baseline: V54 winner
    # =====================================================================
    print("\n  Baseline (V54)...", flush=True)
    total = sum(v54_winner.values())
    w_norm = {k: v / total for k, v in v54_winner.items()}
    r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V54_BASELINE'
        results.append(r)
        print(f"  V54: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: Each decomposed factor solo
    # =====================================================================
    print("\n  Test 1: Decomposed factors solo...", flush=True)
    for fname in v55_names:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = f'{fname}_SOLO'
            results.append(r)
            print(f"    {fname}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 2: V54 + each decomposed factor
    # =====================================================================
    print("\n  Test 2: V54 + decomposed...", flush=True)
    for fname in v55_names:
        for w in [0.05, 0.08, 0.10, 0.12]:
            weights = {**v54_winner, fname: w}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'V54+{fname[-4:]}_W{w}'
                results.append(r)

    # =====================================================================
    # Test 3: Replace V54 factors with decomposed versions
    # =====================================================================
    print("\n  Test 3: Replace V54 factors...", flush=True)
    promising_solo = sorted([r for r in results if '_SOLO' in r['test'] and r['ann'] > 50],
                           key=lambda x: -x['ann'])[:8]
    promising_names = set()
    for r in promising_solo:
        for fname in v55_names:
            if fname in r['test']:
                promising_names.add(fname)
                break

    for fname in promising_names:
        for old_f in ['R_BWP_BNW', 'R_TENSION', 'R_VWCM', 'R_BVR', 'R_BUY_FRAC', 'R_VPIN']:
            weights = {k: v for k, v in v54_winner.items() if k != old_f}
            weights[fname] = 0.12
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'REP_{old_f[-3:]}→{fname[-4:]}'
                results.append(r)

    # =====================================================================
    # Test 4: Best V54 + best decomposed combo
    # =====================================================================
    print("\n  Test 4: V54 + 2 decomposed...", flush=True)
    if len(promising_names) >= 2:
        top_2 = sorted(promising_names)[:2]
        for w1 in [0.05, 0.08]:
            for w2 in [0.05, 0.08]:
                weights = {**v54_winner, top_2[0]: w1, top_2[1]: w2}
                total = sum(weights.values())
                w_norm = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                if r:
                    r['test'] = f'V54+{top_2[0][-3:]}+{top_2[1][-3:]}'
                    results.append(r)

    # =====================================================================
    # Test 5: Pure decomposed portfolio
    # =====================================================================
    print("\n  Test 5: Pure decomposed portfolio...", flush=True)
    if len(promising_names) >= 3:
        top_4 = sorted(promising_names)[:4]
        weights = {f: 0.25 for f in top_4}
        for atr in [0.5, 0.6, 0.7, 0.8]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'DECOMPOSED_EQ_A{atr}'
                results.append(r)

    # =====================================================================
    # Test 6: ATR sweep with best additions
    # =====================================================================
    print("\n  Test 6: ATR sweep...", flush=True)
    if promising_names:
        best_new = sorted(promising_names)[0]
        for atr in [0.3, 0.4, 0.5, 0.6]:
            weights = {**v54_winner, best_new: 0.08}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'V54+{best_new[-4:]}_A{atr}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V55 ACT TEMPORAL DECOMPOSITION)", flush=True)
    print(f"  {'Test':<50s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*95}", flush=True)
    for r in results[:100]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<50s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
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
        print(f"\n  === V55 BEST ===", flush=True)
        print(f"  V55: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V54 RECORD: +1203.8% DD=31.5%", flush=True)
        delta = best['ann'] - 1203.8
        print(f"  Delta from V54: {delta:+.1f}%", flush=True)

        print(f"\n  === SOLO DECOMPOSED FACTOR SUMMARY ===", flush=True)
        solo = sorted([r for r in results if '_SOLO' in r['test']], key=lambda x: -x['ann'])
        for r in solo:
            pos_mark = " ALL+" if all_positive(r) else ""
            print(f"  {r['test']:<45s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    print(f"\n{'='*70}", flush=True)
