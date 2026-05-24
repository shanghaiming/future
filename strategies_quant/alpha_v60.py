"""
Alpha V60 — Novel Information-Theoretic & Statistical Factors
=============================================================
20 novel factors from comprehensive research:

Tier 1 (Highest Expected Alpha):
  R_MI_VOL_RET   - Mutual information between volume and returns
  R_KL_DIV       - KL divergence of recent vs reference return distribution
  R_AC1          - Lag-1 autocorrelation
  R_REL_STRENGTH - Relative strength vs market index
  (R_SAMP_EN removed - too slow O(N^2) template matching)

Tier 2 (High Confidence):
  R_VOL_REGIME   - Volatility regime z-score
  R_AMIHUD_ACCEL - Amihud illiquidity acceleration
  R_VOLUME_SKEW  - Volume-weighted return skewness
  R_CUSUM        - CUSUM change-point detection
  R_RANK_PERSIST - Rank persistence score

Tier 3:
  R_SPREAD_TIGHTEN - Corwin-Schultz spread tightening rate
  R_KATZ_FD        - Katz fractal dimension
  R_HURST          - Hurst exponent (variance ratio)
  R_OU_SPEED       - Ornstein-Uhlenbeck mean reversion speed
  R_WAVELET_ENERGY - Wavelet energy ratio (Haar 3-level)
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


def _ema(arr, span):
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


def compute_v60_factors(NS, ND, C, O, H, L, V):
    """Compute all novel V60 factors — VECTORIZED for speed."""
    t0 = time.time()
    factors = {}

    # Daily returns — vectorized
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    # =====================================================================
    # Tier 1: R_AC1 — Lag-1 Autocorrelation (VECTORIZED per day)
    # =====================================================================
    print("  Computing R_AC1...", flush=True)
    ac1 = np.full((NS, ND), np.nan)
    W = 20
    for di in range(MIN_TRAIN + W, ND):
        chunk = ret[:, di - W:di]  # (NS, W)
        # Count valid per stock
        n_valid = np.sum(~np.isnan(chunk), axis=1)
        valid_stocks = n_valid >= 10
        for si in np.where(valid_stocks)[0]:
            rv = chunk[si, ~np.isnan(chunk[si])]
            if len(rv) < 10:
                continue
            r1, r2 = rv[:-1], rv[1:]
            s1, s2 = np.std(r1), np.std(r2)
            if s1 > 1e-10 and s2 > 1e-10:
                ac1[si, di] = np.corrcoef(r1, r2)[0, 1]
    factors['R_AC1'] = _rank_normalize(ac1)
    print(f"    R_AC1 done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 1: R_REL_STRENGTH — Relative Strength vs Market (VECTORIZED)
    # =====================================================================
    print("  Computing R_REL_STRENGTH...", flush=True)
    mkt_ret = np.full(ND, np.nan)
    for di in range(1, ND):
        valid = ~np.isnan(ret[:, di])
        if valid.sum() > 50:
            mkt_ret[di] = np.mean(ret[valid, di])

    rel_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di]) & ~np.isnan(mkt_ret[di])
        rel_ret[m, di] = ret[m, di] - mkt_ret[di]

    rel_strength = _ema(rel_ret, 10)
    factors['R_REL_STRENGTH'] = _rank_normalize(rel_strength)
    print(f"    R_REL_STRENGTH done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 1: R_KL_DIV — KL Divergence (VECTORIZED per day)
    # =====================================================================
    print("  Computing R_KL_DIV...", flush=True)
    kl_div = np.full((NS, ND), np.nan)
    eps = 1e-10
    W_p, W_q = 20, 60
    for di in range(MIN_TRAIN + W_q, ND):
        p_chunk = ret[:, di - W_p:di]  # (NS, W_p)
        q_chunk = ret[:, di - W_q:di]  # (NS, W_q)

        # Count valid
        n_p = np.sum(~np.isnan(p_chunk), axis=1)
        n_q = np.sum(~np.isnan(q_chunk), axis=1)
        valid = (n_p >= 10) & (n_q >= 30)

        for si in np.where(valid)[0]:
            pv = p_chunk[si, ~np.isnan(p_chunk[si])]
            qv = q_chunk[si, ~np.isnan(q_chunk[si])]
            cmin = min(pv.min(), qv.min())
            cmax = max(pv.max(), qv.max())
            if cmax - cmin < 1e-10:
                continue
            bins = np.linspace(cmin, cmax, 11)
            p_c = np.histogram(pv, bins)[0].astype(float) + eps
            q_c = np.histogram(qv, bins)[0].astype(float) + eps
            p_p = p_c / p_c.sum()
            q_p = q_c / q_c.sum()
            kl_div[si, di] = np.sum(p_p * np.log(p_p / q_p))

    factors['R_KL_DIV'] = _rank_normalize(kl_div)
    print(f"    R_KL_DIV done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 1: R_MI_VOL_RET — Mutual Information (VECTORIZED per day)
    # =====================================================================
    print("  Computing R_MI_VOL_RET...", flush=True)
    mi_vr = np.full((NS, ND), np.nan)
    W_mi = 20
    for di in range(MIN_TRAIN + W_mi, ND):
        r_chunk = ret[:, di - W_mi:di]
        v_chunk = V[:, di - W_mi:di]

        # Count valid
        valid_mask = ~np.isnan(r_chunk) & ~np.isnan(v_chunk) & (v_chunk > 0)
        n_valid = valid_mask.sum(axis=1)
        stocks = np.where(n_valid >= 10)[0]

        for si in stocks:
            vm = valid_mask[si]
            r_v = r_chunk[si, vm]
            v_v = v_chunk[si, vm]
            r_bins = np.percentile(r_v, [0, 20, 40, 60, 80, 100])
            v_bins = np.percentile(v_v, [0, 20, 40, 60, 80, 100])
            if r_bins[-1] - r_bins[0] < 1e-15:
                continue
            joint, _, _ = np.histogram2d(r_v, v_v, bins=[r_bins, v_bins])
            joint = joint.astype(float) + 1e-10
            p_j = joint / joint.sum()
            p_r = p_j.sum(axis=1)
            p_v = p_j.sum(axis=0)
            mi_vr[si, di] = np.sum(p_j * np.log(p_j / (p_r[:, None] * p_v[None, :])))

    factors['R_MI_VOL_RET'] = _rank_normalize(mi_vr)
    print(f"    R_MI_VOL_RET done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 2: R_VOL_REGIME — Volatility Regime z-Score (FULLY VECTORIZED)
    # =====================================================================
    print("  Computing R_VOL_REGIME...", flush=True)
    rv5 = _rolling_std(ret, 5)
    rv_long = _rolling_mean(rv5, 60)
    rv_diff = rv5 - rv_long
    rv_vol = _rolling_std(rv_diff, 20)
    vol_z = np.full((NS, ND), np.nan)
    mask = ~np.isnan(rv5) & ~np.isnan(rv_long) & ~np.isnan(rv_vol) & (rv_vol > 1e-10)
    vol_z[mask] = (rv5[mask] - rv_long[mask]) / rv_vol[mask]
    factors['R_VOL_REGIME'] = _rank_normalize(-vol_z)
    print(f"    R_VOL_REGIME done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 2: R_AMIHUD_ACCEL — Amihud Acceleration (FULLY VECTORIZED)
    # =====================================================================
    print("  Computing R_AMIHUD_ACCEL...", flush=True)
    dollar_vol = V * C
    with np.errstate(divide='ignore', invalid='ignore'):
        ami_daily = np.where(
            (dollar_vol > 0) & ~np.isnan(ret) & ~np.isnan(dollar_vol),
            np.abs(ret) / dollar_vol, np.nan)
    ami_fast = _ema(ami_daily, 5)
    ami_slow = _ema(ami_daily, 20)
    ami_accel = ami_fast - ami_slow
    factors['R_AMIHUD_ACCEL'] = _rank_normalize(-ami_accel)
    print(f"    R_AMIHUD_ACCEL done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 2: R_VOLUME_SKEW — Volume-Weighted Skewness (VECTORIZED per day)
    # =====================================================================
    print("  Computing R_VOLUME_SKEW...", flush=True)
    vol_skew = np.full((NS, ND), np.nan)
    W_sk = 20
    for di in range(MIN_TRAIN + W_sk, ND):
        r_chunk = ret[:, di - W_sk:di]
        v_chunk = V[:, di - W_sk:di]
        valid_mask = ~np.isnan(r_chunk) & ~np.isnan(v_chunk) & (v_chunk > 0)
        n_valid = valid_mask.sum(axis=1)
        stocks = np.where(n_valid >= 10)[0]
        for si in stocks:
            vm = valid_mask[si]
            r_v = r_chunk[si, vm]
            v_v = v_chunk[si, vm]
            total_v = v_v.sum()
            vw_mean = (v_v * r_v).sum() / total_v
            vw_var = (v_v * (r_v - vw_mean) ** 2).sum() / total_v
            if vw_var < 1e-20:
                continue
            vw_std = np.sqrt(vw_var)
            vol_skew[si, di] = (v_v * (r_v - vw_mean) ** 3).sum() / (total_v * vw_std ** 3)
    factors['R_VOLUME_SKEW'] = _rank_normalize(vol_skew)
    print(f"    R_VOLUME_SKEW done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 2: R_CUSUM — CUSUM Change-Point (VECTORIZED per day)
    # =====================================================================
    print("  Computing R_CUSUM...", flush=True)
    cusum_factor = np.full((NS, ND), np.nan)
    W_c = 60
    for di in range(MIN_TRAIN + W_c, ND):
        chunk = ret[:, di - W_c:di]  # (NS, 60)
        n_valid = np.sum(~np.isnan(chunk), axis=1)
        stocks = np.where(n_valid >= 30)[0]
        for si in stocks:
            rv = chunk[si, ~np.isnan(chunk[si])]
            mu = np.mean(rv[:40])
            dev = rv[40:] - mu
            if len(dev) < 5:
                continue
            cs = np.cumsum(dev)
            cusum_factor[si, di] = np.maximum(0, cs - np.minimum.accumulate(cs))[-1]
    factors['R_CUSUM'] = _rank_normalize(cusum_factor)
    print(f"    R_CUSUM done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 2: R_RANK_PERSIST — Rank Persistence (FULLY VECTORIZED)
    # =====================================================================
    print("  Computing R_RANK_PERSIST...", flush=True)
    mom5 = np.full((NS, ND), np.nan)
    for di in range(6, ND):
        m = ~np.isnan(C[:, di - 1]) & ~np.isnan(C[:, di - 6]) & (C[:, di - 6] > 0)
        mom5[m, di] = (C[m, di - 1] - C[m, di - 6]) / C[m, di - 6]

    daily_rank = np.full((NS, ND), np.nan)
    for di in range(6, ND):
        valid = ~np.isnan(mom5[:, di])
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(mom5[valid, di])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1) / n
        daily_rank[valid, di] = ranks

    top_decile = np.where(~np.isnan(daily_rank), (daily_rank > 0.9).astype(float), np.nan)
    persist = _rolling_mean(top_decile, 10)
    factors['R_RANK_PERSIST'] = _rank_normalize(persist)
    print(f"    R_RANK_PERSIST done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 3: R_SPREAD_TIGHTEN — Spread Tightening (VECTORIZED)
    # =====================================================================
    print("  Computing R_SPREAD_TIGHTEN...", flush=True)
    alpha_cs = 3 - 2 * np.sqrt(2)  # Corwin-Schultz formula constant
    S = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(H[:, di]) & ~np.isnan(L[:, di]) & (H[:, di] > L[:, di])
        m2 = m & ~np.isnan(H[:, di - 1]) & ~np.isnan(L[:, di - 1])
        if m2.sum() == 0:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            g = np.log(H[m2, di] / L[m2, di])
            k = 2 * np.log(
                np.maximum(H[m2, di], H[m2, di - 1]) /
                np.minimum(L[m2, di], L[m2, di - 1])
            ) - g
            denom = np.maximum(g, 1e-10)
            gamma = k / denom
            with np.errstate(invalid='ignore'):
                inner = gamma ** 2 - alpha_cs * g
            inner = np.maximum(inner, 0)
            s_val = (2 * (np.exp(inner) - 1)) / (1 + np.exp(inner))
        S[m2, di] = np.abs(s_val)

    cs_20 = _rolling_mean(S, 20)
    cs_tighten = np.full((NS, ND), np.nan)
    for di in range(6, ND):
        m = (~np.isnan(cs_20[:, di]) & ~np.isnan(cs_20[:, di - 5]) &
             (cs_20[:, di - 5] > 1e-10))
        cs_tighten[m, di] = (cs_20[m, di] - cs_20[m, di - 5]) / cs_20[m, di - 5]
    factors['R_SPREAD_TIGHTEN'] = _rank_normalize(-cs_tighten)
    print(f"    R_SPREAD_TIGHTEN done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 3: R_KATZ_FD — Katz Fractal Dimension (VECTORIZED per day)
    # =====================================================================
    print("  Computing R_KATZ_FD...", flush=True)
    katz_fd = np.full((NS, ND), np.nan)
    W_k = 20
    for di in range(MIN_TRAIN + W_k, ND):
        chunk = ret[:, di - W_k:di]  # (NS, W_k)
        n_valid = np.sum(~np.isnan(chunk), axis=1)
        stocks = np.where(n_valid >= 10)[0]
        for si in stocks:
            rv = chunk[si, ~np.isnan(chunk[si])]
            n = len(rv)
            cumsum = np.cumsum(rv)
            L = np.sum(np.abs(rv))
            d = np.max(np.abs(cumsum - cumsum[0]))
            if L < 1e-10 or d < 1e-10:
                continue
            log_n = np.log(n)
            katz_fd[si, di] = -(log_n / (log_n + np.log(d / L)))
    factors['R_KATZ_FD'] = _rank_normalize(katz_fd)
    print(f"    R_KATZ_FD done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 3: R_HURST — Hurst Exponent (VECTORIZED per day)
    # =====================================================================
    print("  Computing R_HURST...", flush=True)
    hurst = np.full((NS, ND), np.nan)
    W_h = 100
    for di in range(MIN_TRAIN + W_h, ND):
        chunk = ret[:, di - W_h:di]
        n_valid = np.sum(~np.isnan(chunk), axis=1)
        stocks = np.where(n_valid >= 50)[0]
        for si in stocks:
            rv = chunk[si, ~np.isnan(chunk[si])]
            var1 = np.var(rv)
            n_chunks = len(rv) // 10
            if n_chunks < 3:
                continue
            ret10 = rv[:n_chunks * 10].reshape(n_chunks, 10).sum(axis=1)
            var10 = np.var(ret10)
            if var1 < 1e-20 or var10 < 1e-20:
                continue
            H = 0.5 * (np.log(var10 / (10 * var1)) / np.log(10)) + 0.5
            hurst[si, di] = H - 0.5
    factors['R_HURST'] = _rank_normalize(hurst)
    print(f"    R_HURST done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 3: R_OU_SPEED — OU Mean Reversion Speed (FULLY VECTORIZED)
    # =====================================================================
    print("  Computing R_OU_SPEED...", flush=True)
    ou_speed = np.full((NS, ND), np.nan)
    mask_ou = ~np.isnan(ac1) & (ac1 > 0.01)
    ou_speed[mask_ou] = -np.log(np.clip(ac1[mask_ou], 0.01, 0.99))
    factors['R_OU_SPEED'] = _rank_normalize(-ou_speed)
    print(f"    R_OU_SPEED done ({time.time()-t0:.0f}s)", flush=True)

    # =====================================================================
    # Tier 3: R_WAVELET_ENERGY — Wavelet Energy (VECTORIZED per day)
    # =====================================================================
    print("  Computing R_WAVELET_ENERGY...", flush=True)
    wavelet_e = np.full((NS, ND), np.nan)
    W_w = 60
    for di in range(MIN_TRAIN + W_w, ND):
        chunk = ret[:, di - W_w:di]
        n_valid = np.sum(~np.isnan(chunk), axis=1)
        stocks = np.where(n_valid >= 40)[0]
        for si in stocks:
            rv = chunk[si, ~np.isnan(chunk[si])]
            a = rv.copy()
            detail_energy = 0.0
            for level in range(3):
                n = len(a) // 2
                if n < 2:
                    break
                approx = (a[0::2][:n] + a[1::2][:n]) / np.sqrt(2)
                detail = (a[0::2][:n] - a[1::2][:n]) / np.sqrt(2)
                detail_energy += np.sum(detail ** 2)
                a = approx
            approx_energy = np.sum(a ** 2)
            total = detail_energy + approx_energy
            if total < 1e-20:
                continue
            wavelet_e[si, di] = 1.0 - detail_energy / total
    factors['R_WAVELET_ENERGY'] = _rank_normalize(wavelet_e)
    print(f"    R_WAVELET_ENERGY done ({time.time()-t0:.0f}s)", flush=True)

    print(f"  All V60 factors computed ({time.time()-t0:.0f}s)", flush=True)
    return factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V60 — Novel Information-Theoretic & Statistical Factors")
    print("  V56 record: +1630.7% DD=25.2%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    print("\n  Computing existing factors...", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)
    base_factors = {**v41, **v48, **v49, **v52, **v55}

    print("\n  Computing V60 novel factors...", flush=True)
    v60 = compute_v60_factors(NS, ND, C, O, H, L, V)
    all_factors = {**base_factors, **v60}

    # V56 winning weights
    v54_base = {'R_BWP_BNW': 0.205, 'R_TENSION': 0.205, 'R_VWCM': 0.205,
                'R_BVR': 0.154, 'R_BUY_FRAC': 0.138, 'R_VPIN': 0.092}
    v56_weights = {**v54_base, 'R_SHOCK_MOM': 0.08, 'R_TREND_ACC': 0.15}
    total = sum(v56_weights.values())
    v56_norm = {k: v / total for k, v in v56_weights.items()}

    results = []

    # =====================================================================
    # Baseline: V56
    # =====================================================================
    print("\n  V56 baseline...", flush=True)
    r = backtest_v7c(v56_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V56_BASE'
        results.append(r)
        print(f"  V56: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 1: Each new factor SOLO
    # =====================================================================
    print("\n  Test 1: Solo factor tests...", flush=True)
    v60_names = sorted(v60.keys())
    for fname in v60_names:
        for atr in [0.5, 0.8]:
            r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{fname}_SOLO_A{atr}'
                results.append(r)
                print(f"    {fname}_A{atr}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 2: V56 + each new factor
    # =====================================================================
    print("\n  Test 2: V56 + new factors...", flush=True)
    for fname in v60_names:
        for w in [0.05, 0.08, 0.10, 0.15]:
            weights = {**v56_norm, fname: w}
            total = sum(weights.values())
            wn = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'V56+{fname}_W{w:.2f}'
                results.append(r)
                print(f"    V56+{fname[-10:]}_W{w:.2f}: {r['ann']:+.1f}%", flush=True)

    # =====================================================================
    # Test 3: Top 3 new factors combined with V56
    # =====================================================================
    print("\n  Test 3: Top factors combined...", flush=True)
    # Find best solo factors
    solos = [r for r in results if '_SOLO_' in r['test']]
    if solos:
        solos_sorted = sorted(solos, key=lambda x: -x['ann'])
        top3_names = []
        for s in solos_sorted[:5]:
            name = s['test'].replace('_SOLO_A0.5', '').replace('_SOLO_A0.8', '')
            if name not in top3_names:
                top3_names.append(name)
            if len(top3_names) >= 3:
                break

        # Test all combos of top 3 new factors
        for i, n1 in enumerate(top3_names):
            for j, n2 in enumerate(top3_names):
                if j <= i:
                    continue
                for w1 in [0.05, 0.08]:
                    for w2 in [0.05, 0.08]:
                        weights = {**v56_norm, n1: w1, n2: w2}
                        total = sum(weights.values())
                        wn = {k: v / total for k, v in weights.items()}
                        r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                                        top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                        if r:
                            r['test'] = f'V56+{n1[:8]}+{n2[:8]}_W{w1:.2f}_{w2:.2f}'
                            results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'=' * 100}", flush=True)
    print(f"  ALL RESULTS (V60 NOVEL FACTORS)", flush=True)
    print(f"  {'Test':<45s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-' * 85}", flush=True)
    for r in results[:60]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<45s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Solo summary
    print(f"\n  === SOLO FACTOR SUMMARY ===", flush=True)
    solos_sorted = sorted([r for r in results if '_SOLO_' in r['test']], key=lambda x: -x['ann'])
    for r in solos_sorted[:20]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<45s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    # V56+ summary
    print(f"\n  === V56 + NEW FACTOR BEST ===", flush=True)
    v56_new = sorted([r for r in results if r['test'].startswith('V56+')], key=lambda x: -x['ann'])
    for r in v56_new[:20]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<45s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i + 1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V60 BEST ===", flush=True)
        print(f"  V60: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V56 RECORD: +1630.7% DD=25.2%", flush=True)
        delta = best['ann'] - 1630.7
        print(f"  Delta from V56: {delta:+.1f}%", flush=True)

    print(f"\n{'=' * 70}", flush=True)
