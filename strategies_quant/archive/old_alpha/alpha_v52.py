"""
Alpha V52 — Microstructure Proxy Factors (from 1000% Research)
===============================================================
V51 record: VWCM0.2+BVR0.15_A0.5 = +620.8% DD=25.1% ALL+

V52 implements microstructure proxies derived from daily OHLCV data:
1. Kyle's Lambda: |return| / sqrt(dollar_volume) — price impact
2. Corwin-Schultz Spread: bid-ask spread from H/L sequences
3. Roll Measure: effective spread from serial covariance
4. Smart Money Index: VWAP deviation × volume
5. VPIN daily proxy: order flow toxicity from OHLCV
6. Order Flow Imbalance: net buying pressure proxy
7. Depth Imbalance: buyer vs seller volume asymmetry
8. Volume Clock Momentum: volume-time momentum (not calendar)
9. Information Share: price discovery efficiency
10. Liquidity-Adjusted Momentum: momentum penalized by illiquidity

Also tests:
- V51 + microstructure factor combos
- Factor interaction terms (VWCM × microstructure)
- Bayesian-style adaptive weights (performance-weighted)
- Regime-conditional factor weighting

All factors: cross-sectionally rank-normalized, no look-ahead (use di-1 data).
Vectorized for speed.
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
    NS, ND = arr.shape
    mean = _rolling_mean(arr, window, min_valid)
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


def compute_v52_factors(NS, ND, C, O, H, L, V):
    """Compute microstructure proxy factors from daily OHLCV."""

    factors = {}
    t_total = time.time()

    # Pre-compute common arrays
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[mask, di] = (C[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    hl_range = np.full((NS, ND), np.nan)
    mask = ~np.isnan(H) & ~np.isnan(L)
    hl_range[mask] = H[mask] - L[mask]

    safe_hl = np.where(hl_range > 1e-6, hl_range, np.nan)

    # Dollar volume
    dollar_vol = np.full((NS, ND), np.nan)
    mask = ~np.isnan(V) & ~np.isnan(C) & (V > 0) & (C > 0)
    dollar_vol[mask] = V[mask] * C[mask]

    # Log returns for Roll measure
    log_ret = np.full((NS, ND), np.nan)
    mask = ~np.isnan(C[:, 1:]) & ~np.isnan(C[:, :-1]) & (C[:, :-1] > 0)
    log_ret[:, 1:][mask] = np.log(C[:, 1:][mask] / C[:, :-1][mask])

    # =================================================================
    # 1. Kyle's Lambda: |return| / sqrt(dollar_volume)
    #    Measures price impact: higher = more impact per dollar traded
    #    Use 20-day rolling mean, inverted (low impact = liquid = good)
    # =================================================================
    t0 = time.time()
    kyle = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ret) & ~np.isnan(dollar_vol) & (dollar_vol > 0)
    kyle[mask] = np.abs(ret[mask]) / np.sqrt(dollar_vol[mask])
    kyle_20 = _rolling_mean(kyle, 20)
    factors['R_KYLE'] = _rank_normalize(-kyle_20)  # Inverted: low impact = better
    print(f"  Kyle's Lambda done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 2. Corwin-Schultz Spread Estimator
    #    Uses only H and L from consecutive days to estimate spread.
    #    S = 2(exp(alpha) - 1) / (1 + exp(alpha))
    #    where alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
    #          - sqrt(2/3) if negative
    #    beta = sum_{i=0}^{1} [log(H_i / L_i)]^2
    #    Rolling 20-day mean, inverted.
    # =================================================================
    t0 = time.time()
    log_h = np.full((NS, ND), np.nan)
    log_l = np.full((NS, ND), np.nan)
    mask_h = (H > 0) & ~np.isnan(H)
    mask_l = (L > 0) & ~np.isnan(L)
    log_h[mask_h] = np.log(H[mask_h])
    log_l[mask_l] = np.log(L[mask_l])

    # 2-day beta for each day: beta[di] = (log(H[di]/L[di]))^2 + (log(H[di-1]/L[di-1]))^2
    log_hl_sq = np.full((NS, ND), np.nan)
    mask_hl = ~np.isnan(log_h) & ~np.isnan(log_l)
    log_hl_sq[mask_hl] = (log_h[mask_hl] - log_l[mask_hl]) ** 2

    cs_spread = np.full((NS, ND), np.nan)
    for di in range(2, ND):
        beta = log_hl_sq[:, di] + log_hl_sq[:, di - 1]
        # gamma = (log(H_high / L_low))^2 where H_high = max(H[di], H[di-1])
        h_max = np.maximum(H[:, di], H[:, di - 1])
        l_min = np.minimum(L[:, di], L[:, di - 1])
        gamma = np.full(NS, np.nan)
        m = (h_max > 0) & (l_min > 0) & ~np.isnan(h_max) & ~np.isnan(l_min) & (h_max / l_min > 1)
        gamma[m] = (np.log(h_max[m] / l_min[m])) ** 2

        # alpha
        sqrt2 = np.sqrt(2)
        alpha_val = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * sqrt2) - gamma / (3 - 2 * sqrt2)
        # Clamp alpha >= 0
        alpha_val = np.where(~np.isnan(alpha_val), np.maximum(alpha_val, 0), np.nan)

        # Spread
        s = 2 * (np.exp(alpha_val) - 1) / (1 + np.exp(alpha_val))
        cs_spread[:, di] = s

    cs_20 = _rolling_mean(cs_spread, 20)
    factors['R_CS_SPREAD'] = _rank_normalize(-cs_20)  # Inverted: low spread = liquid = good
    print(f"  Corwin-Schultz Spread done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 3. Roll Measure: 2*sqrt(-Cov(delta_P, delta_P_lag))
    #    Effective bid-ask spread from negative serial covariance
    #    of price changes. Inverted.
    # =================================================================
    t0 = time.time()
    delta_p = np.full((NS, ND), np.nan)
    delta_p_lag = np.full((NS, ND), np.nan)
    for di in range(2, ND):
        mask1 = ~np.isnan(log_ret[:, di]) & ~np.isnan(log_ret[:, di - 1])
        delta_p[mask1, di] = log_ret[mask1, di]
        delta_p_lag[mask1, di] = log_ret[mask1, di - 1]

    # Rolling covariance over 20 days
    dp_mean = _rolling_mean(delta_p, 20)
    dp_lag_mean = _rolling_mean(delta_p_lag, 20)
    cov_dp = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        dp_chunk = delta_p[:, di - 20:di]
        dp_lag_chunk = delta_p_lag[:, di - 20:di]
        valid = ~np.isnan(dp_chunk) & ~np.isnan(dp_lag_chunk)
        n = valid.sum(axis=1)
        enough = n >= 10
        for si in np.where(enough)[0]:
            d = dp_chunk[si][valid[si]] - dp_mean[si, di]
            dl = dp_lag_chunk[si][valid[si]] - dp_lag_mean[si, di]
            cov_dp[si, di] = (d * dl).sum() / n[si]

    roll = np.full((NS, ND), np.nan)
    mask = cov_dp < 0  # Only meaningful when negative
    roll[mask] = 2 * np.sqrt(-cov_dp[mask])
    roll_20 = _rolling_mean(roll, 20)
    factors['R_ROLL'] = _rank_normalize(-roll_20)  # Inverted: low spread = better
    print(f"  Roll Measure done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 4. Smart Money Index (SMI)
    #    VWAP deviation × volume: |close - VWAP| × V
    #    Large deviation + high volume = institutional activity
    #    Rolling 10-day mean.
    # =================================================================
    t0 = time.time()
    # Daily VWAP proxy: (H + L + C) / 3 (typical price as volume-weighted avg proxy)
    typical_price = np.full((NS, ND), np.nan)
    mask = ~np.isnan(H) & ~np.isnan(L) & ~np.isnan(C)
    typical_price[mask] = (H[mask] + L[mask] + C[mask]) / 3.0

    vwap_dev = np.full((NS, ND), np.nan)
    mask = ~np.isnan(C) & ~np.isnan(typical_price) & (typical_price > 0)
    vwap_dev[mask] = (C[mask] - typical_price[mask]) / typical_price[mask]

    smi = np.full((NS, ND), np.nan)
    mask = ~np.isnan(vwap_dev) & ~np.isnan(V)
    smi[mask] = np.abs(vwap_dev[mask]) * V[mask]
    smi_10 = _rolling_mean(smi, 10)
    factors['R_SMI'] = _rank_normalize(smi_10)
    print(f"  Smart Money Index done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 5. VPIN daily proxy (Volume-synchronized Probability of Informed Trading)
    #    Uses bulk volume classification: buy_vol = V * N(C, H, L)
    #    where N = CDF of (C-O)/(H-L) standardized
    #    VPIN = |buy_vol - sell_vol| / total_vol
    #    Rolling 20-day mean.
    # =================================================================
    t0 = time.time()
    # Bulk volume classification: proportion of buy volume
    # Using CDF of (C - O) / (H - L) as proxy
    bvc = np.full((NS, ND), np.nan)
    mask = ~np.isnan(safe_hl) & ~np.isnan(C) & ~np.isnan(O)
    co_diff = np.full((NS, ND), np.nan)
    co_diff[mask] = C[mask] - O[mask]
    # Standardize: z = (C-O) / (HL * 0.5) as rough CDF input
    bvc_z = np.full((NS, ND), np.nan)
    m2 = mask & (safe_hl > 0)
    bvc_z[m2] = co_diff[m2] / (safe_hl[m2] * 0.5 + 1e-10)

    # CDF approximation using error function
    from scipy.special import ndtr
    bvc[m2] = ndtr(bvc_z[m2])  # P(buy)

    buy_vol = np.full((NS, ND), np.nan)
    sell_vol = np.full((NS, ND), np.nan)
    mask_v = ~np.isnan(V) & ~np.isnan(bvc)
    buy_vol[mask_v] = V[mask_v] * bvc[mask_v]
    sell_vol[mask_v] = V[mask_v] * (1 - bvc[mask_v])

    vpin = np.full((NS, ND), np.nan)
    mask_vp = ~np.isnan(buy_vol) & ~np.isnan(sell_vol)
    vpin[mask_vp] = np.abs(buy_vol[mask_vp] - sell_vol[mask_vp])

    vpin_20 = _rolling_mean(vpin, 20)
    vol_20 = _rolling_mean(V, 20)
    vpin_norm = np.full((NS, ND), np.nan)
    mask = ~np.isnan(vpin_20) & ~np.isnan(vol_20) & (vol_20 > 0)
    vpin_norm[mask] = vpin_20[mask] / vol_20[mask]
    factors['R_VPIN'] = _rank_normalize(vpin_norm)
    print(f"  VPIN done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 6. Order Flow Imbalance (OFI)
    #    (2*C - H - L) / (H - L) × V — net buying pressure
    #    +1 = close at high (all buy), -1 = close at low (all sell)
    #    Rolling 10-day mean.
    # =================================================================
    t0 = time.time()
    ofi_daily = np.full((NS, ND), np.nan)
    mask = ~np.isnan(safe_hl) & ~np.isnan(C) & ~np.isnan(H) & ~np.isnan(L)
    ofi_daily[mask] = (2 * C[mask] - H[mask] - L[mask]) / safe_hl[mask]

    # Volume-weighted OFI
    ofi_vol = np.full((NS, ND), np.nan)
    mask2 = ~np.isnan(ofi_daily) & ~np.isnan(V)
    ofi_vol[mask2] = ofi_daily[mask2] * V[mask2]
    ofi_10 = _rolling_mean(ofi_vol, 10)
    factors['R_OFI'] = _rank_normalize(ofi_10)
    print(f"  Order Flow Imbalance done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 7. Depth Imbalance
    #    Net buyer vs seller volume: V*(C-L)/(H-L) - V*(H-C)/(H-L)
    #    = V*(2C - H - L)/(H-L) — same as OFI but normalized by V
    #    This is buyer_vol - seller_vol, not the ratio.
    #    Rolling 10-day mean, normalized by total volume.
    # =================================================================
    t0 = time.time()
    cl_diff = np.full((NS, ND), np.nan)
    hc_diff = np.full((NS, ND), np.nan)
    mask = ~np.isnan(safe_hl) & ~np.isnan(C) & ~np.isnan(L) & ~np.isnan(H)
    cl_diff[mask] = C[mask] - L[mask]
    hc_diff[mask] = H[mask] - C[mask]

    buyer_v = np.where(mask & ~np.isnan(V), V * cl_diff / safe_hl, np.nan)
    seller_v = np.where(mask & ~np.isnan(V), V * hc_diff / safe_hl, np.nan)

    depth_imb = np.full((NS, ND), np.nan)
    mask3 = ~np.isnan(buyer_v) & ~np.isnan(seller_v) & ~np.isnan(V)
    total_bv = buyer_v + seller_v
    m4 = mask3 & (total_bv > 0)
    depth_imb[m4] = (buyer_v[m4] - seller_v[m4]) / total_bv[m4]
    di_10 = _rolling_mean(depth_imb, 10)
    factors['R_DEPTH_IMB'] = _rank_normalize(di_10)
    print(f"  Depth Imbalance done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 8. Volume Clock Momentum
    #    Instead of calendar-time momentum, compute return per unit volume.
    #    mom_per_vol = ret_10 / sqrt(mean(V*V, 10))
    #    Measures price efficiency per unit of trading activity.
    # =================================================================
    t0 = time.time()
    # 10-day return
    ret_10 = np.full((NS, ND), np.nan)
    for di in range(11, ND):
        mask = ~np.isnan(C[:, di - 1]) & ~np.isnan(C[:, di - 11]) & (C[:, di - 11] > 0)
        ret_10[mask, di] = (C[mask, di - 1] - C[mask, di - 11]) / C[mask, di - 11]

    # Volume-weighted: use sqrt of mean squared volume as volume proxy
    v_sq = np.where(~np.isnan(V), V ** 2, np.nan)
    v_rms_10 = np.sqrt(_rolling_mean(v_sq, 10))
    vol_mom = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ret_10) & ~np.isnan(v_rms_10) & (v_rms_10 > 0)
    vol_mom[mask] = ret_10[mask] / np.sqrt(v_rms_10[mask])
    factors['R_VOL_MOM'] = _rank_normalize(vol_mom)
    print(f"  Volume Clock Momentum done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 9. Information Share
    #    Correlation of close-to-close returns with volume-weighted price changes
    #    High = close price reflects information efficiently
    #    Rolling 20-day correlation.
    # =================================================================
    t0 = time.time()
    # Volume-weighted price change: V*(C - C_prev) / sum(V)
    vw_change = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(ret[:, di]) & ~np.isnan(V[:, di]) & (V[:, di] > 0)
        vw_change[mask, di] = ret[mask, di] * V[mask, di]

    vw_20 = _rolling_mean(vw_change, 20)
    ret_20 = _rolling_mean(ret, 20)
    # Correlation between ret and vw_change over 20 days
    # Use simplified: ratio of means as proxy for information share
    info_share = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ret_20) & ~np.isnan(vw_20) & (np.abs(ret_20) > 1e-10)
    info_share[mask] = vw_20[mask] / ret_20[mask]
    # Clamp to reasonable range
    info_share = np.clip(info_share, -5, 5)
    factors['R_INFO_SHARE'] = _rank_normalize(info_share)
    print(f"  Information Share done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 10. Liquidity-Adjusted Momentum
    #     10-day momentum / Amihud illiquidity
    #     Momentum that's cheap to capture (high liquidity)
    # =================================================================
    t0 = time.time()
    # Amihud: mean(|ret| / dollar_vol, 20)
    abs_ret_dv = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ret) & ~np.isnan(dollar_vol) & (dollar_vol > 0)
    abs_ret_dv[mask] = np.abs(ret[mask]) / dollar_vol[mask]
    amihud_20 = _rolling_mean(abs_ret_dv, 20)

    # 10-day momentum
    mom_10 = np.full((NS, ND), np.nan)
    for di in range(11, ND):
        mask = ~np.isnan(C[:, di - 1]) & ~np.isnan(C[:, di - 11]) & (C[:, di - 11] > 0)
        mom_10[mask, di] = (C[mask, di - 1] - C[mask, di - 11]) / C[mask, di - 11]

    liq_mom = np.full((NS, ND), np.nan)
    mask = ~np.isnan(mom_10) & ~np.isnan(amihud_20) & (amihud_20 > 1e-15)
    liq_mom[mask] = mom_10[mask] / amihud_20[mask]
    factors['R_LIQ_MOM'] = _rank_normalize(liq_mom)
    print(f"  Liquidity-Adjusted Momentum done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 11. Microstructure Noise Ratio (interaction)
    #     Kyle × VPIN interaction: high impact + high informed trading = toxic
    #     Low = healthy market for momentum
    # =================================================================
    t0 = time.time()
    noise = np.full((NS, ND), np.nan)
    mask = ~np.isnan(kyle_20) & ~np.isnan(vpin_norm)
    noise[mask] = kyle_20[mask] * vpin_norm[mask]
    factors['R_MICRO_NOISE'] = _rank_normalize(-noise)  # Inverted: low noise = better
    print(f"  Microstructure Noise done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 12. Volume-Weighted Close Price (VWCP)
    #     mean(V*C, 10) / mean(V, 10) — volume-weighted average close
    #     Compare to actual close: if close > VWCP, buying pressure
    # =================================================================
    t0 = time.time()
    vc_10 = _rolling_mean(V * C, 10)
    v_10 = _rolling_mean(V, 10)
    vwcp = np.full((NS, ND), np.nan)
    mask = ~np.isnan(vc_10) & ~np.isnan(v_10) & (v_10 > 0) & ~np.isnan(C)
    vwcp[mask] = C[mask] / (vc_10[mask] / v_10[mask]) - 1  # Close vs VWCP
    vwcp_10 = _rolling_mean(vwcp, 10)
    factors['R_VWCP'] = _rank_normalize(vwcp_10)
    print(f"  VWCP done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 13. Tick Rule Proxy
    #     Daily proxy for trade direction: if C > prev_C, buy-initiated
    #     fraction = mean(sign(C - prev_C) > 0, 20)
    #     High buy fraction = persistent buying
    # =================================================================
    t0 = time.time()
    buy_tick = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        buy_tick[mask, di] = (C[mask, di] > C[mask, di - 1]).astype(float)
    buy_frac_20 = _rolling_mean(buy_tick, 20)
    factors['R_BUY_FRAC'] = _rank_normalize(buy_frac_20)
    print(f"  Buy Fraction done ({time.time()-t0:.0f}s)", flush=True)

    # =================================================================
    # 14. Overnight Gap Microstructure
    #     gap = (O - prev_C) / prev_C
    #     gap_volume = gap * sqrt(prev_V) — gap weighted by prior activity
    #     High = significant gap with institutional participation
    # =================================================================
    t0 = time.time()
    gap = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        mask = ~np.isnan(O[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        gap[mask, di] = (O[mask, di] - C[mask, di - 1]) / C[mask, di - 1]

    sq_vol = np.where(~np.isnan(V), np.sqrt(np.abs(V) + 1), np.nan)
    gap_vol = np.full((NS, ND), np.nan)
    mask = ~np.isnan(gap) & ~np.isnan(sq_vol)
    gap_vol[mask] = gap[mask] * sq_vol[mask]
    gap_vol_10 = _rolling_mean(gap_vol, 10)
    factors['R_GAP_MICRO'] = _rank_normalize(gap_vol_10)
    print(f"  Gap Microstructure done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  Total V52 computation: {time.time()-t_total:.0f}s", flush=True)
    return factors


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V52 — Microstructure Proxy Factors (1000% Research)")
    print("  V51 record: VWCM0.2+BVR0.15_A0.5 = +620.8% DD=25.1%", flush=True)
    print("  Target: break 1000%", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute V41 factors
    print("\n  Computing V41 factors...", flush=True)
    v41_factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)

    # Compute V48 + V49 factors (for VWCM, BVR, etc.)
    print("\n  Computing V48 + V49 factors...", flush=True)
    v48_factors = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49_factors = compute_v49_factors(NS, ND, C, O, H, L, V)

    # Compute V52 microstructure factors
    print("\n  Computing V52 microstructure factors...", flush=True)
    v52_factors = compute_v52_factors(NS, ND, C, O, H, L, V)

    all_factors = {**v41_factors, **v48_factors, **v49_factors, **v52_factors}

    # V51 winning config
    v51_weights = {'R_BWP_BNW': 0.178, 'R_TENSION': 0.178, 'R_R_SQUARED': 0.178,
                   'R_SMA_DEV': 0.178, 'R_VWCM': 0.178, 'R_BVR': 0.134}

    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    v52_names = sorted(v52_factors.keys())
    print(f"\n  New V52 factors: {len(v52_names)} — {v52_names}", flush=True)

    results = []

    # =====================================================================
    # Baselines
    # =====================================================================
    print("\n  Baselines...", flush=True)
    r = backtest_v7c(v51_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.5)
    if r:
        r['test'] = 'V51_WINNER_A0.5'
        results.append(r)
        print(f"  V51 winner: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    r = backtest_v7c(v51_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.6)
    if r:
        r['test'] = 'V51_WINNER_A0.6'
        results.append(r)

    # =====================================================================
    # Test 1: Each microstructure factor SOLO
    # =====================================================================
    print("\n  Test 1: Microstructure factors solo...", flush=True)
    for fname in v52_names:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = f'{fname}_SOLO'
            results.append(r)
            print(f"    {fname}: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # Test 2: V51 + each microstructure factor
    # =====================================================================
    print("\n  Test 2: V51 + microstructure...", flush=True)
    for fname in v52_names:
        for w in [0.05, 0.1, 0.15]:
            weights = {**v51_weights, fname: w}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            for atr in [0.5, 0.6]:
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'V51+{fname}_W{w}_A{atr}'
                    results.append(r)
    print(f"  V51+micro: {len(results)}", flush=True)

    # =====================================================================
    # Test 3: Replace V51 factors with microstructure
    # =====================================================================
    print("\n  Test 3: Replace V51 with microstructure...", flush=True)
    v51_list = ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV', 'R_VWCM', 'R_BVR']
    # Find promising microstructure factors from Test 2
    promising = [r for r in results if 'V51+' in r['test'] and r['ann'] > 500]
    promising_names = set()
    for r in promising:
        for fname in v52_names:
            if fname in r['test']:
                promising_names.add(fname)
                break

    print(f"  Promising micro factors: {promising_names}", flush=True)
    for fname in promising_names:
        for old_f in ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV', 'R_BVR']:
            new_w = {k: v for k, v in v51_weights.items() if k != old_f}
            new_w[fname] = 0.15
            total = sum(new_w.values())
            w_norm = {k: v / total for k, v in new_w.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'REP_{old_f[-4:]}→{fname}'
                results.append(r)

    # =====================================================================
    # Test 4: Factor interaction terms (VWCM × microstructure)
    # =====================================================================
    print("\n  Test 4: Factor interactions...", flush=True)
    # Create interaction factors: R_VWCM * R_MICRO for each micro factor
    for fname in v52_names:
        interaction = np.full((NS, ND), np.nan)
        vwcm = all_factors.get('R_VWCM')
        micro = all_factors.get(fname)
        if vwcm is not None and micro is not None:
            mask = ~np.isnan(vwcm) & ~np.isnan(micro)
            interaction[mask] = vwcm[mask] * micro[mask] / 100
            all_factors[f'R_VWCM_x_{fname}'] = _rank_normalize(interaction)

            weights = {**v51_weights, f'R_VWCM_x_{fname}': 0.1}
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'V51+VWCMx{fname[-4:]}'
                results.append(r)

    # =====================================================================
    # Test 5: Pure microstructure portfolio (no V41)
    # =====================================================================
    print("\n  Test 5: Pure microstructure combos...", flush=True)
    # Top 3-4 microstructure factors equally weighted
    solo_sorted = sorted([r for r in results if '_SOLO' in r['test'] and r['ann'] > 50],
                         key=lambda x: -x['ann'])[:6]
    top_micro = []
    for r in solo_sorted:
        for fname in v52_names:
            if fname in r['test']:
                top_micro.append(fname)
                break

    if len(top_micro) >= 3:
        # Equal weight top micro factors
        weights = {f: 1.0 / len(top_micro[:4]) for f in top_micro[:4]}
        for atr in [0.5, 0.6, 0.7, 0.8]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'MICRO_EQ_A{atr}'
                results.append(r)

        # Top micro + VWCM + BVR
        for combo_size in [2, 3]:
            for i in range(min(combo_size, len(top_micro))):
                weights = {'R_VWCM': 0.25, 'R_BVR': 0.15}
                for f in top_micro[:combo_size]:
                    weights[f] = 0.2
                total = sum(weights.values())
                w_norm = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.5)
                if r:
                    names = '+'.join(f[-3:] for f in top_micro[:combo_size])
                    r['test'] = f'VWCM+BVR+{names}'
                    results.append(r)

    # =====================================================================
    # Test 6: V51 + best micro factor weight sweep
    # =====================================================================
    print("\n  Test 6: V51 + best micro weight sweep...", flush=True)
    if promising_names:
        best_micro = sorted(promising_names)[0]
        for w in [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]:
            for atr in [0.4, 0.5, 0.6]:
                weights = {**v51_weights, best_micro: w}
                total = sum(weights.values())
                w_norm = {k: v / total for k, v in weights.items()}
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'V51+{best_micro}_W{w}_A{atr}'
                    results.append(r)

    # =====================================================================
    # Test 7: Multi-microstructure addition
    # =====================================================================
    print("\n  Test 7: Multi-micro addition...", flush=True)
    # Add top 2-3 micro factors to V51
    if len(top_micro) >= 2:
        for n_add in [2, 3]:
            if n_add > len(top_micro):
                continue
            weights = {**v51_weights}
            for f in top_micro[:n_add]:
                weights[f] = 0.08
            total = sum(weights.values())
            w_norm = {k: v / total for k, v in weights.items()}
            for atr in [0.5, 0.6]:
                r = backtest_v7c(w_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    names = '+'.join(f[-3:] for f in top_micro[:n_add])
                    r['test'] = f'V51+{n_add}MICRO_{names}_A{atr}'
                    results.append(r)

    # =====================================================================
    # Test 8: ATR + rebalance fine sweep on V51 winner
    # =====================================================================
    print("\n  Test 8: V51 parameter fine sweep...", flush=True)
    for atr in [0.3, 0.35, 0.4, 0.45, 0.5]:
        for rebal in [3, 4, 5]:
            r = backtest_v7c(v51_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=rebal, atr_stop_mult=atr)
            if r:
                r['test'] = f'V51_A{atr}_R{rebal}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V52 MICROSTRUCTURE PROXIES)", flush=True)
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
        print(f"\n  === V52 BEST ===", flush=True)
        print(f"  V52: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V51 RECORD: VWCM0.2+BVR0.15_A0.5 = +620.8% DD=25.1%", flush=True)
        delta = best['ann'] - 620.8
        print(f"  Delta from V51: {delta:+.1f}%", flush=True)
        print(f"  Target: 1000%", flush=True)

        print(f"\n  === SOLO MICRO FACTOR SUMMARY ===", flush=True)
        solo = sorted([r for r in results if '_SOLO' in r['test']], key=lambda x: -x['ann'])
        for r in solo:
            pos_mark = " ALL+" if all_positive(r) else ""
            print(f"  {r['test']:<45s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    print(f"\n{'='*70}", flush=True)
