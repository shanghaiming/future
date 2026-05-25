"""V118: Factor Interaction Terms — manual non-linear factor engineering.
================================================================
Key insight: ML discovers non-linear factor interactions. But we can
engineer these manually from V116's proven factor set, which is:
1. Faster than ML (no training needed)
2. More interpretable
3. No overfitting risk from model training

Interaction ideas from literature:
- VOL_ADJ_MOM × TENSION = "safe momentum with compression" (breakout pending)
- VOL_ADJ_MOM × REL_STR = "strong stock getting stronger safely"
- TENSION × OIS = "compressed price + order imbalance" (smart money)
- SMA_DEV × VOL_MOM = "trending with volume confirmation"

Also adds:
- Momentum acceleration (3d mom vs 10d mom)
- Volume price confirmation ratio
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import MIN_TRAIN


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


def compute_interaction_factors(NS, ND, C, O, H, L, V):
    """Compute factor interaction terms from V116's proven factors."""
    t0 = time.time()
    factors = {}

    # Basic building blocks
    # NO LOOK-AHEAD: all returns use C[di-1] (yesterday's close), not C[di]
    # 5-day return (as of yesterday's close)
    ret5 = np.full((NS, ND), np.nan)
    for di in range(7, ND):
        m = ~np.isnan(C[:, di-1]) & ~np.isnan(C[:, di-6]) & (C[:, di-6] > 0)
        ret5[m, di] = (C[m, di-1] - C[m, di-6]) / C[m, di-6]

    # 3-day return (as of yesterday's close)
    ret3 = np.full((NS, ND), np.nan)
    for di in range(5, ND):
        m = ~np.isnan(C[:, di-1]) & ~np.isnan(C[:, di-4]) & (C[:, di-4] > 0)
        ret3[m, di] = (C[m, di-1] - C[m, di-4]) / C[m, di-4]

    # 10-day return (as of yesterday's close)
    ret10 = np.full((NS, ND), np.nan)
    for di in range(12, ND):
        m = ~np.isnan(C[:, di-1]) & ~np.isnan(C[:, di-11]) & (C[:, di-11] > 0)
        ret10[m, di] = (C[m, di-1] - C[m, di-11]) / C[m, di-11]

    # 20-day volatility (uses daily_ret[di-20:di], all from di-1 and earlier)
    daily_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di-1]) & ~np.isnan(C[:, di-2]) & (C[:, di-2] > 0)
        daily_ret[m, di] = (C[m, di-1] - C[m, di-2]) / C[m, di-2]

    vol20 = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        window = daily_ret[:, di-20:di]
        valid_count = (~np.isnan(window)).sum(axis=1)
        enough = valid_count >= 10
        if enough.sum() > 0:
            vol20[enough, di] = np.nanstd(window[enough], axis=1)

    # Vol-adjusted momentum (same as V116)
    vol_adj_mom = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        m = ~np.isnan(ret5[:, di]) & ~np.isnan(vol20[:, di]) & (vol20[:, di] > 0)
        vol_adj_mom[m, di] = ret5[m, di] / vol20[m, di]
    factors['VOL_ADJ_MOM'] = vol_adj_mom

    # --- Factor 1: Momentum Acceleration ---
    # ret3 / ret10 → short-term momentum accelerating vs medium-term
    # High = momentum is accelerating (bullish continuation signal)
    mom_accel = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        m = (~np.isnan(ret3[:, di]) & ~np.isnan(ret10[:, di]) &
             (np.abs(ret10[:, di]) > 1e-6))
        # Sign-aware ratio: both positive and ret3 > ret10 = acceleration
        mom_accel[m, di] = ret3[m, di] - ret10[m, di]  # difference, not ratio
    factors['MOM_ACCEL'] = mom_accel

    # --- Factor 2: Safe Momentum = VOL_ADJ_MOM × positive return ---
    # Only reward momentum if it's consistently positive (not one-day spike)
    safe_mom = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        m = ~np.isnan(vol_adj_mom[:, di]) & ~np.isnan(ret3[:, di])
        safe_mom[m, di] = vol_adj_mom[m, di] * np.sign(ret3[m, di])
    factors['SAFE_MOM'] = safe_mom

    # --- Factor 3: Volume-Price Confirmation ---
    # Volume change ratio × return sign = "is volume confirming the move?"
    vol_ma5 = np.full((NS, ND), np.nan)
    vol_ma20 = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        v5 = V[:, di-5:di]; v20 = V[:, di-20:di]
        m5 = (~np.isnan(v5)).sum(axis=1) >= 3
        m20 = (~np.isnan(v20)).sum(axis=1) >= 10
        both = m5 & m20
        if both.sum() > 0:
            vol_ma5[both, di] = np.nanmean(v5[both], axis=1)
            vol_ma20[both, di] = np.nanmean(v20[both], axis=1)

    vol_price_conf = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        m = (~np.isnan(vol_ma5[:, di]) & ~np.isnan(vol_ma20[:, di]) &
             (vol_ma20[:, di] > 0) & ~np.isnan(ret5[:, di]))
        # Volume ratio × direction = confirmation
        vol_ratio = vol_ma5[m, di] / vol_ma20[m, di]
        vol_price_conf[m, di] = vol_ratio * np.sign(ret5[m, di]) * np.abs(ret5[m, di])
    factors['VOL_PRICE_CONF'] = vol_price_conf

    # --- Factor 4: Compression-Momentum = TENSION proxy × VOL_ADJ_MOM ---
    # TENSION = how compressed is the price range (low HL range / ATR)
    # We approximate with: -(H-L)/ATR over last 5 days
    compression = np.full((NS, ND), np.nan)
    for di in range(16, ND):
        # Average (H-L)/ATR over last 5 days
        atr_sum = np.zeros(NS)
        range_sum = np.zeros(NS)
        count = np.zeros(NS)
        for dd in range(di-5, di):
            h = H[:, dd]; l = L[:, dd]
            m = ~np.isnan(h) & ~np.isnan(l)
            range_sum[m] += (h[m] - l[m])
            if dd > 0:
                tr = np.maximum(h - l, np.abs(h - C[:, dd-1]))
                tr = np.maximum(tr, np.abs(l - C[:, dd-1]))
                m2 = ~np.isnan(tr)
                atr_sum[m2] += tr[m2]
            count[m] += 1
        valid = (count >= 3) & (atr_sum > 0)
        if valid.sum() > 50:
            compression[valid, di] = -(range_sum[valid] / count[valid]) / (atr_sum[valid] / count[valid])

    # Interaction: compression × vol_adj_mom
    comp_mom = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        m = ~np.isnan(compression[:, di]) & ~np.isnan(vol_adj_mom[:, di])
        comp_mom[m, di] = compression[m, di] * vol_adj_mom[m, di]
    factors['COMP_MOM'] = comp_mom

    # --- Factor 5: Relative Volume Surge ---
    # Stocks with sudden volume increase + positive return = institutional activity
    vol_surge = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        m = (~np.isnan(vol_ma5[:, di]) & ~np.isnan(vol_ma20[:, di]) &
             (vol_ma20[:, di] > 0))
        vol_surge[m, di] = vol_ma5[m, di] / vol_ma20[m, di]
    factors['VOL_SURGE'] = vol_surge

    # --- Factor 6: Smooth Momentum ---
    # Low intraday volatility + positive close-to-close momentum
    intraday_vol = np.full((NS, ND), np.nan)
    for di in range(6, ND):
        h5 = H[:, di-5:di]; l5 = L[:, di-5:di]
        m = (~np.isnan(h5) & ~np.isnan(l5)).sum(axis=1) >= 3
        if m.sum() > 50:
            intraday_vol[m, di] = np.nanmean((h5[m] - l5[m]) / np.maximum(l5[m], 1e-6), axis=1)

    smooth_mom = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        m = (~np.isnan(intraday_vol[:, di]) & ~np.isnan(ret5[:, di]) &
             (intraday_vol[:, di] > 0))
        smooth_mom[m, di] = ret5[m, di] / intraday_vol[m, di]
    factors['SMOOTH_MOM'] = smooth_mom

    # Rank normalize all
    result = {}
    for k, v in factors.items():
        result[f'R_{k}'] = _rank_normalize(v)
        result[k] = v  # also keep raw

    print(f"  Interaction factors done ({time.time()-t0:.0f}s)", flush=True)
    return result


if __name__ == '__main__':
    print("V118: Factor Interaction Terms")
    print("Use test_v118.py to run backtests.")
