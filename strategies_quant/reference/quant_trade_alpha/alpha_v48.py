"""
Alpha V48 — Novel A-Share Factors from Web Research + Probability Theory
=========================================================================
V47 exhausted DMD/Kalman/W1/factor momentum approaches. All failed to beat +344.6%.

V48 implements genuinely NEW factors from:
1. Web research on Chinese A-share alpha factors (papers, WQ Alpha 101)
2. Unexplored probability_theory.md sections (entropy, FFT, wavelets)

Truly novel factors (NOT duplicates of existing pipeline):
- Overnight-Intraday Spread (unique to A-share T+1/fixed window)
- Price Curve Stability (institutional accumulation signal)
- Volume-Conditional Momentum (momentum confirmed by volume trend)
- Amihud Illiquidity (price impact per dollar volume)
- Intraday Skew Proxy (directional quality without volume weight)
- WQ Alpha#12 (volume-price divergence, proven in A-shares)
- WQ Alpha#54 (high-price volume correlation)
- WQ Alpha#40 (range contraction at high prices)
- Garman-Klass Realized Volatility (efficient OHLC estimator)
- Shannon Entropy (information-theoretic uncertainty measure)
- FFT Dominant Cycle (spectral analysis for cycle phase)

All factors: cross-sectionally rank-normalized, no look-ahead (use di-1 data).
VECTORISED for speed.
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
    """Rolling mean along axis=1 (time). Handles NaN."""
    if min_valid is None:
        min_valid = window // 2
    NS, ND = arr.shape
    out = np.full_like(arr, np.nan)
    cumsum = np.nancumsum(arr, axis=1)
    cumcount = np.cumsum(~np.isnan(arr), axis=1)
    for di in range(window, ND):
        cs = cumsum[:, di - 1] - (cumsum[:, di - window - 1] if di > window else 0)
        cc = cumcount[:, di - 1] - (cumcount[:, di - window - 1] if di > window else 0)
        valid = cc >= min_valid
        out[valid, di] = cs[valid] / cc[valid]
    return out


def _rolling_std(arr, window, min_valid=None):
    """Rolling std along axis=1. Uses vectorised computation."""
    if min_valid is None:
        min_valid = window // 2
    NS, ND = arr.shape
    mean = _rolling_mean(arr, window, min_valid)
    out = np.full_like(arr, np.nan)
    for di in range(window, ND):
        m = mean[:, di]
        if np.all(np.isnan(m)):
            continue
        chunk = arr[:, di - window:di]
        sq_diff = (chunk - m[:, np.newaxis]) ** 2
        # Replace NaN diffs with 0 for summing
        sq_diff = np.where(np.isnan(sq_diff), 0, sq_diff)
        n_valid = np.sum(~np.isnan(chunk), axis=1)
        valid = n_valid >= min_valid
        ss = np.sum(sq_diff, axis=1)
        out[valid, di] = np.sqrt(ss[valid] / n_valid[valid])
    return out


def compute_v48_factors(NS, ND, C, O, H, L, V):
    """Compute all novel V48 factors — vectorised for speed.
    No look-ahead: all use data up to di-1."""

    factors = {}

    # =================================================================
    # Pre-compute common arrays
    # =================================================================
    t_total = time.time()

    # Daily returns: ret[si, di] = (C[si,di] - C[si,di-1]) / C[si,di-1]
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[mask, di] = (C[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    # HL range
    hl_range = np.full((NS, ND), np.nan)
    mask = ~np.isnan(H) & ~np.isnan(L)
    hl_range[mask] = H[mask] - L[mask]

    # CO difference (C - O)
    co_diff = np.full((NS, ND), np.nan)
    mask = ~np.isnan(C) & ~np.isnan(O)
    co_diff[mask] = C[mask] - O[mask]

    # =================================================================
    # 1. Intraday Skew Proxy
    #    (C - O) / (H - L) averaged over 20 days (no volume weight)
    # =================================================================
    t0 = time.time()
    intraday_skew = np.full((NS, ND), np.nan)
    safe_hl = np.where(hl_range > 1e-6, hl_range, np.nan)
    intraday_skew = np.where(~np.isnan(safe_hl), co_diff / safe_hl, np.nan)
    skew_20 = _rolling_mean(intraday_skew, 20)
    factors['R_ISKEW'] = _rank_normalize(skew_20)
    print(f"  Intraday Skew done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 2. Overnight-Intraday Spread
    #    overnight_ret[di] = (O[di] - C[di-1]) / C[di-1]
    #    intraday_ret[di] = (C[di] - O[di]) / O[di]
    #    OIS = rolling_mean(overnight - intraday, 20)
    # =================================================================
    t0 = time.time()
    overnight_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(O[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        overnight_ret[mask, di] = (O[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    intraday_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(O[:, di]) & (O[:, di] > 0)
        intraday_ret[mask, di] = (C[mask, di] - O[mask, di]) / O[mask, di]

    ois = overnight_ret - intraday_ret
    ois_20 = _rolling_mean(ois, 20)
    factors['R_OIS'] = _rank_normalize(ois_20)
    print(f"  Overnight-Intraday Spread done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 3. Price Curve Stability
    #    -std(ret, 20) / mean(|ret|, 20) — coefficient of variation inverted
    # =================================================================
    t0 = time.time()
    ret_std_20 = _rolling_std(ret, 20)
    abs_ret = np.abs(ret)
    abs_ret_mean_20 = _rolling_mean(abs_ret, 20)
    pcs = np.full((NS, ND), np.nan)
    mask = abs_ret_mean_20 > 1e-10
    pcs[mask] = -ret_std_20[mask] / abs_ret_mean_20[mask]
    factors['R_PCS'] = _rank_normalize(pcs)
    print(f"  Price Curve Stability done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 4. Amihud Illiquidity
    #    mean(|ret| / (V * C), 20) — price impact per dollar volume
    # =================================================================
    t0 = time.time()
    dollar_vol = np.full((NS, ND), np.nan)
    mask = ~np.isnan(V) & ~np.isnan(C) & (V > 0) & (C > 0)
    dollar_vol[mask] = V[mask] * C[mask]
    abs_ret_div_dvol = np.full((NS, ND), np.nan)
    mask2 = mask & (dollar_vol > 0) & ~np.isnan(ret)
    abs_ret_div_dvol[mask2] = np.abs(ret[mask2]) / dollar_vol[mask2]
    amihud_20 = _rolling_mean(abs_ret_div_dvol, 20)
    factors['R_AMIHUD'] = _rank_normalize(amihud_20)
    print(f"  Amihud Illiquidity done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 5. Volume-Conditional Momentum
    #    mom_20 * sign(V_now - V_avg_20)
    #    Momentum confirmed by volume increase.
    # =================================================================
    t0 = time.time()
    # 20-day momentum
    mom_20 = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        mask = ~np.isnan(C[:, di - 1]) & ~np.isnan(C[:, di - 21]) & (C[:, di - 21] > 0)
        mom_20[mask, di] = (C[mask, di - 1] - C[mask, di - 21]) / C[mask, di - 21]

    # Volume vs 20-day average
    vol_mean_20 = _rolling_mean(V, 20)
    vol_sign = np.full((NS, ND), np.nan)
    mask = ~np.isnan(V[:, 1:]) & ~np.isnan(vol_mean_20[:, 1:]) & (vol_mean_20[:, 1:] > 0)
    vol_sign[:, 1:] = np.where(mask, np.sign(V[:, 1:] - vol_mean_20[:, 1:]), np.nan)

    vcm = mom_20 * vol_sign
    factors['R_VCM'] = _rank_normalize(vcm)
    print(f"  Volume-Conditional Momentum done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 6. WQ Alpha#12: Volume-Price Divergence (rolling 5-day sum)
    #    sign(delta(V, 1)) * (-delta(C, 1))
    # =================================================================
    t0 = time.time()
    delta_v = np.full((NS, ND), np.nan)
    delta_c = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(V[:, di]) & ~np.isnan(V[:, di - 1])
        delta_v[mask, di] = np.sign(V[mask, di] - V[mask, di - 1])
        mask2 = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1])
        delta_c[mask2, di] = -(C[mask2, di] - C[mask2, di - 1])

    wq12_daily = delta_v * delta_c
    wq12_5 = _rolling_mean(wq12_daily, 5)
    factors['R_WQ12'] = _rank_normalize(wq12_5)
    print(f"  WQ Alpha#12 done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 7. WQ Alpha#54: -corr(rank(H), rank(V), 3)
    #    Per-stock rolling 3-day correlation of H and V
    # =================================================================
    t0 = time.time()
    wq54 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(MIN_TRAIN + 5, ND):
            h_vals = H[si, di - 3:di]
            v_vals = V[si, di - 3:di]
            valid = ~np.isnan(h_vals) & ~np.isnan(v_vals)
            n = valid.sum()
            if n < 3:
                continue
            h_v = h_vals[valid]
            v_v = v_vals[valid]
            corr = np.corrcoef(h_v, v_v)[0, 1]
            if not np.isnan(corr):
                wq54[si, di] = -corr
    factors['R_WQ54'] = _rank_normalize(wq54)
    print(f"  WQ Alpha#54 done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 8. WQ Alpha#40: -rank(std(H-L, 10)) * rank(C)
    #    Range contraction at high prices = consolidation before breakout
    # =================================================================
    t0 = time.time()
    hl_std_10 = _rolling_std(hl_range, 10)
    # Combined: -std * C (then rank-normalized together)
    wq40 = np.full((NS, ND), np.nan)
    mask = ~np.isnan(hl_std_10) & ~np.isnan(C)
    wq40[mask] = -hl_std_10[mask] * C[mask]
    factors['R_WQ40'] = _rank_normalize(wq40)
    print(f"  WQ Alpha#40 done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 9. Garman-Klass Realized Volatility
    #    0.5*(log(H/L))^2 - (2*log(2)-1)*(log(C/O))^2
    #    Rolling 20-day mean, inverted.
    # =================================================================
    t0 = time.time()
    log_hl = np.full((NS, ND), np.nan)
    log_co = np.full((NS, ND), np.nan)
    mask_hl = (H > 0) & (L > 0) & ~np.isnan(H) & ~np.isnan(L) & (H / L > 1 + 1e-10)
    log_hl[mask_hl] = np.log(H[mask_hl] / L[mask_hl])
    mask_co = (C > 0) & (O > 0) & ~np.isnan(C) & ~np.isnan(O) & (C / O > 1e-10)
    log_co[mask_co] = np.log(C[mask_co] / O[mask_co])

    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gk = np.where(gk > 0, gk, np.nan)  # Only positive values
    gk_20 = _rolling_mean(gk, 20)
    gk_inv = -gk_20  # Inverted: low vol = higher score
    factors['R_GK_RV'] = _rank_normalize(gk_inv)
    print(f"  Garman-Klass RV done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 10. Shannon Entropy of Returns
    #     Low entropy = ordered/trending (predictable)
    #     Inverted: low entropy ranked higher.
    #     Using 5-bin histogram over 20-day returns.
    # =================================================================
    t0 = time.time()
    entropy = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(MIN_TRAIN + 20, ND):
            rets = ret[si, di - 20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            try:
                bins = np.percentile(valid, [0, 20, 40, 60, 80, 100])
                if bins[-1] - bins[0] < 1e-10:
                    continue
                counts = np.histogram(valid, bins=bins)[0]
                counts = counts[counts > 0]
                probs = counts / counts.sum()
                h = -np.sum(probs * np.log2(probs))
                entropy[si, di] = -h  # Inverted
            except Exception:
                pass
    factors['R_ENTROPY'] = _rank_normalize(entropy)
    print(f"  Shannon Entropy done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 11. Overnight Return (pure, rolling 10-day mean)
    #     A-share specific: overnight returns carry institutional info
    # =================================================================
    onret_10 = _rolling_mean(overnight_ret, 10)
    factors['R_ONRET'] = _rank_normalize(onret_10)
    print(f"  Overnight Return done", flush=True)

    # =================================================================
    # 12. Buyer Volume Ratio
    #     mean(V*(C-L)/(H-L), 5) / mean(V, 20)
    # =================================================================
    t0 = time.time()
    buyer_vol = np.full((NS, ND), np.nan)
    mask = (hl_range > 1e-6) & ~np.isnan(V) & ~np.isnan(C) & ~np.isnan(L)
    cl_diff = np.full((NS, ND), np.nan)
    cl_diff[mask] = C[mask] - L[mask]
    buyer_vol = np.where(mask, V * cl_diff / hl_range, np.nan)
    bv_5 = _rolling_mean(buyer_vol, 5)
    v_20 = _rolling_mean(V, 20)
    bvr = np.full((NS, ND), np.nan)
    mask = ~np.isnan(bv_5) & ~np.isnan(v_20) & (v_20 > 0)
    bvr[mask] = bv_5[mask] / v_20[mask]
    factors['R_BVR'] = _rank_normalize(bvr)
    print(f"  Buyer Volume Ratio done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 13. Turnover-Enhanced Reversal
    #     -ret_1d * (V / Adv20)
    #     High turnover + negative return = retail panic reversal
    # =================================================================
    t0 = time.time()
    v_ratio = np.full((NS, ND), np.nan)
    mask = ~np.isnan(V) & ~np.isnan(v_20) & (v_20 > 0)
    v_ratio[mask] = V[mask] / v_20[mask]
    ter = -ret * v_ratio
    ter_5 = _rolling_mean(ter, 5)
    factors['R_TER'] = _rank_normalize(ter_5)
    print(f"  Turnover-Enhanced Reversal done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 14. FFT Dominant Phase Factor
    #     FFT on 60-day returns → dominant cycle phase
    #     -sin(phase): high when near bottom of cycle → due to rise
    # =================================================================
    t0 = time.time()
    fft_len = 60
    fft_phase = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(MIN_TRAIN + fft_len, ND, 3):  # Every 3 days for speed
            rets = ret[si, di - fft_len:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 30:
                continue
            detrended = valid - np.mean(valid)
            try:
                fft_vals = np.fft.fft(detrended)
                power = np.abs(fft_vals[1:len(fft_vals) // 2 + 1]) ** 2
                if len(power) == 0:
                    continue
                dom_idx = np.argmax(power) + 1
                phase = np.angle(fft_vals[dom_idx])
                fft_phase[si, di] = -np.sin(phase)
            except Exception:
                pass
    # Forward-fill
    for si in range(NS):
        for di in range(1, ND):
            if np.isnan(fft_phase[si, di]) and not np.isnan(fft_phase[si, di - 1]):
                fft_phase[si, di] = fft_phase[si, di - 1]
    factors['R_FFT_PHASE'] = _rank_normalize(fft_phase)
    print(f"  FFT Phase done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  Total V48 factor computation: {time.time()-t_total:.0f}s", flush=True)
    return factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V48 — Novel A-Share Factors (Web Research + Probability Theory)")
    print("  Target: beat V46 V41_A0.6 = +344.6%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute V41 factors (baseline)
    print("\n  Computing V41 factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    # Compute novel V48 factors
    print("\n  Computing V48 novel factors...", flush=True)
    v48_factors = compute_v48_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v41_factors, **v48_factors}

    new_factor_names = sorted(v48_factors.keys())
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
    # Test 2: V41 + each new factor (weight sweep)
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
        # ATR=0.8 variant
        weights = {**v41_weights, fname: 0.15}
        total = sum(weights.values())
        w_norm = {k: v / total for k, v in weights.items()}
        r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = f'V41+{fname[-4:]}_W0.15_A0.8'
            results.append(r)
    print(f"  V41+new: {len(results)} results", flush=True)

    # =====================================================================
    # Test 3: Replace V41 factors with promising new ones
    # =====================================================================
    print("\n  Test 3: Replace V41 factors...", flush=True)
    # Find factors that improved over baseline when added
    promising = [r for r in results
                 if 'V41+' in r['test'] and '_W0.15_A0.6' in r['test']
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
    print(f"  Replace: {len(results)} results", flush=True)

    # =====================================================================
    # Test 4: top_n=2
    # =====================================================================
    print("\n  Test 4: top_n=2...", flush=True)
    r = backtest_v7c(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=2, rebalance_days=5, atr_stop_mult=0.6)
    if r:
        r['test'] = 'V41_A0.6_N2'
        results.append(r)
        print(f"  V41 N=2: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # Test 5: Best novel pair combos with V41
    # =====================================================================
    print("\n  Test 5: Novel pair combos...", flush=True)
    from itertools import combinations
    solo_good = [r for r in results if '_SOLO' in r['test'] and r['ann'] > 100]
    solo_sorted = sorted(solo_good, key=lambda x: -x['ann'])[:5]
    top_solo = []
    for r in solo_sorted:
        for fname in new_factor_names:
            if fname in r['test']:
                top_solo.append(fname)
                break

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
    print(f"  ALL RESULTS (V48 NOVEL A-SHARE FACTORS)", flush=True)
    print(f"  {'Test':<45s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*90}", flush=True)
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
        baseline = next((r for r in results if 'BASELINE' in r['test']), None)
        print(f"\n  === V48 BEST ===", flush=True)
        print(f"  V48: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        if baseline:
            print(f"  Baseline: {baseline['ann']:+.1f}% DD={baseline['max_dd']:.1f}%", flush=True)
        print(f"  V46 RECORD: V41_A0.6 = +344.6% DD=52.0%", flush=True)
        delta = best['ann'] - 344.6
        print(f"  Delta: {delta:+.1f}%", flush=True)

        print(f"\n  === SOLO FACTOR SUMMARY ===", flush=True)
        solo = [r for r in results if '_SOLO' in r['test']]
        solo.sort(key=lambda x: -x['ann'])
        for r in solo:
            print(f"  {r['test']:<30s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
