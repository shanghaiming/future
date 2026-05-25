"""V116: Dynamic Lookback Window for REL_STRENGTH.
=============================================
Based on paper 2106.08420: dynamic lookback improves Sharpe by 66%.
High volatility → short lookback (3-5 days)
Low volatility → long lookback (20-30 days)

Also adds:
- Sortino-based ranking (per report: better for A-shares)
- Volatility regime detection
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import MIN_TRAIN


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


def _rank_normalize(factor_2d, min_stocks=50):
    NS, ND = factor_2d.shape
    ranked = np.full_like(factor_2d, np.nan)
    for di in range(ND):
        vals = factor_2d[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < min_stocks: continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        ranked[valid, di] = ranks / n * 100
    return ranked


def compute_dynamic_lookback_factors(NS, ND, C, O, H, L, V):
    """Compute REL_STRENGTH with volatility-adaptive dynamic lookback."""
    t0 = time.time()
    factors = {}

    # Daily returns
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    # Market return
    mkt_ret = np.full(ND, np.nan)
    for di in range(1, ND):
        valid = ~np.isnan(ret[:, di])
        if valid.sum() > 50:
            mkt_ret[di] = np.mean(ret[valid, di])

    # Relative return
    rel_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di]) & ~np.isnan(mkt_ret[di])
        rel_ret[m, di] = ret[m, di] - mkt_ret[di]

    # Individual stock volatility (20-day rolling std of returns)
    vol20 = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        window = ret[:, di-20:di]
        valid_count = (~np.isnan(window)).sum(axis=1)
        enough = valid_count >= 10
        if enough.sum() > 0:
            vol20[enough, di] = np.nanstd(window[enough], axis=1)

    # Market volatility regime
    mkt_vol = np.full(ND, np.nan)
    for di in range(21, ND):
        valid = ~np.isnan(ret[:, di-20:di])
        if valid.sum() > 500:
            mkt_vol[di] = np.nanstd(ret[:, di-20:di])

    # Classify regimes: LOW/MEDIUM/HIGH volatility
    mkt_vol_p25 = np.nanpercentile(mkt_vol[~np.isnan(mkt_vol)], 25)
    mkt_vol_p75 = np.nanpercentile(mkt_vol[~np.isnan(mkt_vol)], 75)

    # --- Factor 1: Dynamic lookback REL_STRENGTH ---
    # HIGH vol → span 3, MEDIUM → span 10, LOW → span 30
    dynamic_rel = np.full((NS, ND), np.nan)
    ema_s3 = _ema(rel_ret, 3)
    ema_s10 = _ema(rel_ret, 10)
    ema_s30 = _ema(rel_ret, 30)

    for di in range(MIN_TRAIN, ND):
        if np.isnan(mkt_vol[di]): continue
        if mkt_vol[di] > mkt_vol_p75:
            dynamic_rel[:, di] = ema_s3[:, di]
        elif mkt_vol[di] < mkt_vol_p25:
            dynamic_rel[:, di] = ema_s30[:, di]
        else:
            dynamic_rel[:, di] = ema_s10[:, di]

    factors['R_DYN_REL_STR'] = _rank_normalize(dynamic_rel)

    # --- Factor 2: Sortino-based ranking ---
    # Downside deviation as risk measure, then Sortino = mean(excess) / downside_dev
    sortino_20d = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        window = ret[:, di-20:di]
        for si in range(NS):
            r = window[si, :]
            valid = r[~np.isnan(r)]
            if len(valid) < 10: continue
            mean_r = np.mean(valid)
            neg_r = valid[valid < 0]
            if len(neg_r) > 3:
                downside_dev = np.sqrt(np.mean(neg_r**2))
                if downside_dev > 0:
                    sortino_20d[si, di] = mean_r / downside_dev

    factors['R_SORTINO_20D'] = _rank_normalize(sortino_20d)

    # --- Factor 3: Vol-adjusted momentum ---
    # momentum / volatility = risk-adjusted momentum
    # NO LOOK-AHEAD: uses C[di-1] (yesterday's close), not C[di]
    mom5 = np.full((NS, ND), np.nan)
    for di in range(7, ND):
        m = ~np.isnan(C[:, di-1]) & ~np.isnan(C[:, di-6]) & (C[:, di-6] > 0)
        mom5[m, di] = (C[m, di-1] - C[m, di-6]) / C[m, di-6]

    vol_adj_mom = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        m = ~np.isnan(mom5[:, di]) & ~np.isnan(vol20[:, di]) & (vol20[:, di] > 0)
        vol_adj_mom[m, di] = mom5[m, di] / vol20[m, di]

    factors['R_VOL_ADJ_MOM'] = _rank_normalize(vol_adj_mom)

    # --- Factor 4: Intraday range ratio ---
    # High intraday range relative to ATR = speculative activity
    range_ratio = np.full((NS, ND), np.nan)
    for di in range(15, ND):
        h = H[:, di-1]; l = L[:, di-1]; c = C[:, di-1]
        m = ~np.isnan(h) & ~np.isnan(l) & (c > 0)
        if m.sum() < 50: continue
        day_range = h[m] - l[m]
        # ATR(14)
        atr_sum = np.zeros(m.sum())
        for dd in range(max(di-15, 1), di):
            tr = H[m, dd] - L[m, dd]
            if dd > 0:
                tr = np.maximum(tr, np.abs(H[m, dd] - C[m, dd-1]))
                tr = np.maximum(tr, np.abs(L[m, dd] - C[m, dd-1]))
            atr_sum += tr
        atr_val = atr_sum / min(14, di - max(di-15, 1))
        valid_atr = atr_val > 0
        rr = np.zeros(m.sum())
        rr[valid_atr] = day_range[valid_atr] / atr_val[valid_atr]
        temp = np.full(NS, np.nan)
        temp[m] = rr
        range_ratio[:, di] = temp

    # Lower range ratio = calmer stock (better for momentum)
    factors['R_CALMNESS'] = _rank_normalize(-range_ratio)  # negative because calm = good

    print(f"  Dynamic lookback factors done ({time.time()-t0:.0f}s)", flush=True)
    return factors


if __name__ == '__main__':
    print("V116: Dynamic lookback window factors")
    print("Use test_v116.py to run backtests.")
