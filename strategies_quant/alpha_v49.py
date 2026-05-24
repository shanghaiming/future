"""
Alpha V49 — WorldQuant-Style Alpha Expressions + Advanced Novel Factors
=======================================================================
V48 tests 14 novel factors. V49 extends with:
1. More WQ Alpha 101 expressions adapted for A-shares
2. Kernel-based factors (from probability_theory.md Section 26)
3. Cross-sectional interaction factors (novel combos)
4. Regime-conditional factors (factor values depend on market state)

All fully vectorized for speed. No look-ahead (use di-1 data).
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0
from alpha_v7c import backtest_v7c
from alpha_v44 import compute_v41_factors_only


def _rank_normalize(factor_2d, min_stocks=50):
    """Rank-normalize a (NS, ND) array cross-sectionally to [1, 100]."""
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
    cumsum = np.where(np.isnan(arr), 0, arr)
    cumcount = (~np.isnan(arr)).astype(float)
    cs = np.cumsum(cumsum, axis=1)
    cc = np.cumsum(cumcount, axis=1)
    for di in range(window, ND):
        s = cs[:, di - 1] - (cs[:, di - window - 1] if di > window else 0)
        c = cc[:, di - 1] - (cc[:, di - window - 1] if di > window else 0)
        valid = c >= min_valid
        out[valid, di] = s[valid] / c[valid]
    return out


def _ts_corr(x, y, window, min_valid=None):
    """Rolling correlation between x and y along axis=1."""
    if min_valid is None:
        min_valid = window * 0.6
    NS, ND = x.shape
    out = np.full((NS, ND), np.nan)
    for di in range(window, ND):
        xv = x[:, di - window:di]
        yv = y[:, di - window:di]
        valid_both = ~np.isnan(xv) & ~np.isnan(yv)
        n = valid_both.sum(axis=1)
        enough = n >= min_valid
        for si in np.where(enough)[0]:
            xm = xv[si][valid_both[si]]
            ym = yv[si][valid_both[si]]
            if len(xm) < 3:
                continue
            xm = xm - xm.mean()
            ym = ym - ym.mean()
            denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
            if denom > 1e-10:
                out[si, di] = (xm * ym).sum() / denom
    return out


def compute_v49_factors(NS, ND, C, O, H, L, V):
    """Compute V49 WQ-style alpha expressions + advanced factors."""

    factors = {}
    t_total = time.time()

    # Pre-compute common arrays
    # Returns
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[mask, di] = (C[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    # Log close
    log_C = np.full((NS, ND), np.nan)
    mask = C > 0
    log_C[mask] = np.log(C[mask])

    # Log volume
    log_V = np.full((NS, ND), np.nan)
    mask = V > 0
    log_V[mask] = np.log(V[mask] + 1)

    # HL range
    hl_range = np.full((NS, ND), np.nan)
    mask = ~np.isnan(H) & ~np.isnan(L)
    hl_range[mask] = H[mask] - L[mask]

    # CO ratio
    co_ratio = np.full((NS, ND), np.nan)
    mask = ~np.isnan(C) & ~np.isnan(O) & (O > 0)
    co_ratio[mask] = C[mask] / O[mask]

    # Safe HL for division
    safe_hl = np.where(hl_range > 1e-6, hl_range, np.nan)

    # =================================================================
    # 1. WQ Alpha#1: rank(delta(log(volume), 2)) * (-rank(delta(((C-O)/O), 2)))
    #    Rising volume + declining intraday return → reversal
    # =================================================================
    t0 = time.time()
    d_log_v2 = np.full((NS, ND), np.nan)
    d_intraday2 = np.full((NS, ND), np.nan)
    for di in range(3, ND):
        mask1 = ~np.isnan(log_V[:, di]) & ~np.isnan(log_V[:, di - 2])
        d_log_v2[mask1, di] = log_V[mask1, di] - log_V[mask1, di - 2]
        io_now = co_ratio[:, di]
        io_prev = co_ratio[:, di - 2]
        mask2 = ~np.isnan(io_now) & ~np.isnan(io_prev)
        d_intraday2[mask2, di] = io_now[mask2] - io_prev[mask2]

    r_dlv2 = _rank_normalize(d_log_v2)
    r_di2 = _rank_normalize(-d_intraday2)
    wq1 = r_dlv2 * r_di2 / 100  # Scaled to ~0-100
    factors['R_WQ1'] = _rank_normalize(wq1)
    print(f"  WQ#1 done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 2. WQ Alpha#6: -rank(ts_corr(open, volume, 10))
    #    Negative open-volume correlation signals abnormal flow
    # =================================================================
    t0 = time.time()
    corr_ov = _ts_corr(O, V, 10)
    factors['R_WQ6'] = _rank_normalize(-corr_ov)
    print(f"  WQ#6 done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 3. WQ Alpha#20: -rank(O - delay(H, 1)) * rank(O - delay(C, 1)) * rank(O - delay(L, 1))
    #    Opening below prior day's range → downtrend continuation
    # =================================================================
    t0 = time.time()
    gap_h = np.full((NS, ND), np.nan)
    gap_c = np.full((NS, ND), np.nan)
    gap_l = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(O[:, di]) & ~np.isnan(H[:, di - 1]) & ~np.isnan(C[:, di - 1]) & ~np.isnan(L[:, di - 1])
        gap_h[mask, di] = -(O[mask, di] - H[mask, di - 1])
        gap_c[mask, di] = O[mask, di] - C[mask, di - 1]
        gap_l[mask, di] = O[mask, di] - L[mask, di - 1]

    r_gh = _rank_normalize(gap_h)
    r_gc = _rank_normalize(gap_c)
    r_gl = _rank_normalize(gap_l)
    wq20 = r_gh * r_gc * r_gl / 10000
    factors['R_WQ20'] = _rank_normalize(wq20)
    print(f"  WQ#20 done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 4. WQ-style: rank(ts_corr(rank(C), rank(V), 10))
    #    Price-volume correlation over 10 days
    #    When high prices consistently accompany high volume = institutional
    # =================================================================
    t0 = time.time()
    r_C = _rank_normalize(C)
    r_V = _rank_normalize(V)
    corr_pv = _ts_corr(r_C, r_V, 10)
    factors['R_PV_CORR'] = _rank_normalize(corr_pv)
    print(f"  PV Corr done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 5. Kernel-weighted momentum (Gaussian kernel from Section 26)
    #    Weight recent returns by Gaussian kernel: w(t) = exp(-t^2/(2*sigma^2))
    #    More theoretically grounded than equal-weighted momentum
    # =================================================================
    t0 = time.time()
    sigma = 5.0
    window = 20
    kw_mom = np.full((NS, ND), np.nan)
    weights = np.array([np.exp(-k ** 2 / (2 * sigma ** 2)) for k in range(window)])
    weights /= weights.sum()

    for di in range(window + 1, ND):
        ret_chunk = ret[:, di - window:di]
        # Weighted sum (NaN → 0 for computation)
        ret_clean = np.where(np.isnan(ret_chunk), 0, ret_chunk)
        n_valid = (~np.isnan(ret_chunk)).sum(axis=1)
        valid = n_valid >= window // 2
        kw_mom[valid, di] = (ret_clean[valid] * weights[np.newaxis, :]).sum(axis=1)

    factors['R_KW_MOM'] = _rank_normalize(kw_mom)
    print(f"  Kernel Momentum done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 6. Volume-Weighted Close Momentum
    #    Sum(V * C * sign(ret)) / Sum(V) over 10 days
    #    Volume-weighted net directional conviction
    # =================================================================
    t0 = time.time()
    v_signed = np.full((NS, ND), np.nan)
    mask = ~np.isnan(V) & ~np.isnan(C) & ~np.isnan(ret) & (V > 0)
    v_signed[mask] = V[mask] * np.sign(ret[mask]) * C[mask]
    v_signed_mean = _rolling_mean(v_signed, 10)
    v_mean = _rolling_mean(V * C, 10)
    vwm = np.full((NS, ND), np.nan)
    mask = ~np.isnan(v_signed_mean) & ~np.isnan(v_mean) & (v_mean > 0)
    vwm[mask] = v_signed_mean[mask] / v_mean[mask]
    factors['R_VWCM'] = _rank_normalize(vwm)
    print(f"  VW Close Momentum done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 7. Return Asymmetry Factor
    #    ratio = mean(positive_ret, 20) / mean(|negative_ret|, 20)
    #    Stocks with stronger up-moves than down-moves are healthier
    # =================================================================
    t0 = time.time()
    pos_ret = np.where(ret > 0, ret, np.nan)
    neg_ret = np.where(ret < 0, np.abs(ret), np.nan)
    pos_mean = _rolling_mean(pos_ret, 20)
    neg_mean = _rolling_mean(neg_ret, 20)
    asym = np.full((NS, ND), np.nan)
    mask = ~np.isnan(pos_mean) & ~np.isnan(neg_mean) & (neg_mean > 1e-10)
    asym[mask] = pos_mean[mask] / neg_mean[mask]
    factors['R_RET_ASYM'] = _rank_normalize(asym)
    print(f"  Return Asymmetry done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 8. Range Position Factor
    #    mean((C - L) / (H - L), 10) — where close sits in the daily range
    #    High = close near top of range = buying pressure
    #    (Different from ISKEW which is (C-O)/(H-L))
    # =================================================================
    t0 = time.time()
    range_pos = np.full((NS, ND), np.nan)
    cl_diff = np.full((NS, ND), np.nan)
    mask = ~np.isnan(C) & ~np.isnan(L) & (safe_hl > 0)
    cl_diff[mask] = C[mask] - L[mask]
    range_pos = np.where(~np.isnan(safe_hl), cl_diff / safe_hl, np.nan)
    rp_10 = _rolling_mean(range_pos, 10)
    factors['R_RANGE_POS'] = _rank_normalize(rp_10)
    print(f"  Range Position done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 9. Volume Concentration Factor
    #    max(V, 20) / mean(V, 20) — how concentrated is volume in one day
    #    High = one big day dominates → event-driven, unreliable
    #    Low = even volume → steady institutional flow
    #    Inverted: even flow preferred
    # =================================================================
    t0 = time.time()
    v_max_20 = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        chunk = V[:, di - 20:di]
        n_valid = (~np.isnan(chunk)).sum(axis=1)
        valid = n_valid >= 10
        v_max_20[valid, di] = np.nanmax(chunk[valid], axis=1)
    v_mean_20 = _rolling_mean(V, 20)
    v_conc = np.full((NS, ND), np.nan)
    mask = ~np.isnan(v_max_20) & ~np.isnan(v_mean_20) & (v_mean_20 > 0)
    v_conc[mask] = -v_max_20[mask] / v_mean_20[mask]  # Inverted
    factors['R_V_CONC'] = _rank_normalize(v_conc)
    print(f"  Volume Concentration done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 10. Gap Momentum Factor
    #     mean(sign(O[di] - C[di-1]) * C[di], 10)
    #     Measures the consistency and direction of overnight gaps
    #     Positive gap momentum = stocks that gap up and keep rising
    # =================================================================
    t0 = time.time()
    gap_daily = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(O[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        gap_daily[mask, di] = (O[mask, di] - C[mask, di - 1]) / C[mask, di - 1]
    gap_mom = _rolling_mean(gap_daily, 10)
    factors['R_GAP_MOM'] = _rank_normalize(gap_mom)
    print(f"  Gap Momentum done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 11. Intraday Volatility Trend
    #     (H-L)/O averaged over 5 days vs 20 days
    #     Declining intraday volatility = compression before breakout
    # =================================================================
    t0 = time.time()
    intraday_vol = np.full((NS, ND), np.nan)
    mask = ~np.isnan(hl_range) & ~np.isnan(O) & (O > 0)
    intraday_vol[mask] = hl_range[mask] / O[mask]
    iv_5 = _rolling_mean(intraday_vol, 5)
    iv_20 = _rolling_mean(intraday_vol, 20)
    iv_trend = np.full((NS, ND), np.nan)
    mask = ~np.isnan(iv_5) & ~np.isnan(iv_20) & (iv_20 > 1e-10)
    iv_trend[mask] = -iv_5[mask] / iv_20[mask]  # Inverted: compression = positive
    factors['R_IV_TREND'] = _rank_normalize(iv_trend)
    print(f"  IV Trend done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 12. Price-Volume Trend (PVT) derivative
    #     PVT = cumsum(V * ret / prev_ret)
    #     We use rate of change of PVT over 10 days
    # =================================================================
    t0 = time.time()
    pvt = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(V[:, di]) & ~np.isnan(ret[:, di])
        pvt[mask, di] = V[mask, di] * ret[mask, di]
    pvt_cumsum = np.nancumsum(pvt, axis=1)
    pvt_change = np.full((NS, ND), np.nan)
    for di in range(11, ND):
        mask = ~np.isnan(pvt_cumsum[:, di]) & ~np.isnan(pvt_cumsum[:, di - 10])
        pvt_change[mask, di] = pvt_cumsum[mask, di] - pvt_cumsum[mask, di - 10]
    factors['R_PVT_ROC'] = _rank_normalize(pvt_change)
    print(f"  PVT Rate of Change done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  Total V49 computation: {time.time()-t_total:.0f}s", flush=True)
    return factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V49 — WQ-Style Alpha Expressions + Advanced Novel Factors")
    print("  Target: beat V46 V41_A0.6 = +344.6%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute V41 factors (baseline)
    print("\n  Computing V41 factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    # Compute V49 factors
    print("\n  Computing V49 factors...", flush=True)
    v49_factors = compute_v49_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41_factors, **v49_factors}

    new_factor_names = sorted(v49_factors.keys())
    print(f"\n  New factors: {len(new_factor_names)} — {new_factor_names}", flush=True)

    results = []

    # =====================================================================
    # Baseline
    # =====================================================================
    print("\n  Baseline...", flush=True)
    r = backtest_v7c(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.6)
    if r:
        r['test'] = 'V41_A0.6_BASELINE'
        results.append(r)
        print(f"  Baseline: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # Test 1: Each new factor SOLO
    # =====================================================================
    print("\n  Test 1: New factors solo...", flush=True)
    for fname in new_factor_names:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = f'{fname}_SOLO'
            results.append(r)
            print(f"    {fname}: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # Test 2: V41 + each new factor
    # =====================================================================
    print("\n  Test 2: V41 + new factors...", flush=True)
    for fname in new_factor_names:
        for w in [0.1, 0.15, 0.2]:
            weights = {**v41_weights, fname: w}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.6)
            if r:
                r['test'] = f'V41+{fname[-4:]}_W{w}'
                results.append(r)

    print(f"  V41+new: {len(results)} results", flush=True)

    # =====================================================================
    # Test 3: Replace V41 factors with promising ones
    # =====================================================================
    print("\n  Test 3: Replace V41...", flush=True)
    promising = [r for r in results
                 if 'V41+' in r['test'] and '_W0.15' in r['test']
                 and r['ann'] > 300]
    promising_names = set()
    for r in promising:
        for fname in new_factor_names:
            if fname[-4:] in r['test']:
                promising_names.add(fname)
                break

    v41_list = ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV', 'R_HAR_RV_RATIO_INV']
    for fname in promising_names:
        for old_f in v41_list:
            new_w = {k: v for k, v in v41_weights.items() if k != old_f}
            new_w[fname] = 0.2
            r = backtest_v7c(new_w, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.6)
            if r:
                r['test'] = f'REP_{old_f[-4:]}→{fname[-4:]}'
                results.append(r)

    # =====================================================================
    # Test 4: Novel standalone combos (best V49 factors only)
    # =====================================================================
    print("\n  Test 4: Novel combos...", flush=True)
    from itertools import combinations
    solo_good = sorted([r for r in results if '_SOLO' in r['test'] and r['ann'] > 100],
                       key=lambda x: -x['ann'])[:5]
    top_solo = []
    for r in solo_good:
        for fname in new_factor_names:
            if fname in r['test']:
                top_solo.append(fname)
                break

    # Pair with V41
    for combo in combinations(top_solo, 2):
        weights = {f: 0.1 for f in combo}
        weights.update(v41_weights)
        total = sum(weights.values())
        w_norm = {k: v / total for k, v in weights.items()}
        r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.6)
        if r:
            names_short = '+'.join(f[-3:] for f in combo)
            r['test'] = f'V41+2_{names_short}'
            results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V49 WQ-STYLE ALPHAS)", flush=True)
    print(f"  {'Test':<45s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*90}", flush=True)
    for r in results[:60]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<45s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:.1f}%{pos_mark}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        baseline = next((r for r in results if 'BASELINE' in r['test']), None)
        print(f"\n  === V49 BEST ===", flush=True)
        print(f"  V49: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        if baseline:
            print(f"  Baseline: {baseline['ann']:+.1f}%", flush=True)
        print(f"  RECORD: V41_A0.6 = +344.6%", flush=True)
        delta = best['ann'] - 344.6
        print(f"  Delta: {delta:+.1f}%", flush=True)

        print(f"\n  === SOLO SUMMARY ===", flush=True)
        solo = sorted([r for r in results if '_SOLO' in r['test']], key=lambda x: -x['ann'])
        for r in solo:
            print(f"  {r['test']:<30s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
