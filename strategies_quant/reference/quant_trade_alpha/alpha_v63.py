"""
Alpha V63 — Market Essence Factors
====================================
Factors that capture the ESSENCE of market behavior using only daily OHLCV data.

Five signal dimensions:
1. SMART MONEY DETECTION
   - Gap reversal (overnight gap + intraday reversal = institutional entry)
   - Institutional body (large body + high volume = smart money)
   - Close vs VWAP deviation (institutional accumulation/distribution)
   - Volume-weighted close position (informed trading proxy)

2. MOMENTUM QUALITY (not raw momentum)
   - Return per unit risk (5d return / 5d ATR)
   - Trend consistency (consecutive positive days in last 10)
   - Smooth momentum (momentum penalized by path volatility)

3. SUPPLY/DEMAND IMBALANCE
   - Close position in range (demand dominance over 10 days)
   - Accumulation/Distribution (volume on up days vs down days)
   - Range absorption (small range + high volume = absorption)

4. MARKET PHASE AWARENESS
   - Breakout proximity (price vs 20-day range)
   - Contraction-Expansion cycle (narrow then wide = breakout coming)
   - Relative volume (today vs 20-day average)

5. MEAN REVERSION TIMING
   - VWAP distance (deviation from volume-weighted price)
   - Oversold bounce (3+ down days + reversal with volume)
   - Cross-sectional momentum surprise (sudden rank improvement)

All factors: cross-sectionally rank-normalized, no look-ahead (use di-1 data).
Vectorised for speed (500 stocks x 2520 days).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ===========================================================================
# 1. SMART MONEY DETECTION
# ===========================================================================

def smart_money_factors(NS, ND, C, O, H, L, V):
    """
    Detect institutional / informed trading from OHLCV patterns.

    Factors:
      R_GAP_REVERSAL   : Overnight gap followed by intraday reversal = smart money
      R_INST_BODY       : Large body * high relative volume = institutional conviction
      R_VWAP_DEV        : Close deviation from VWAP-like proxy (10d vol-weighted price)
      R_INF_CLOSE_POS   : Volume-weighted close position in range (informed trading)
    """
    t0 = time.time()
    factors = {}

    # --- daily returns (using di-1 data) ---
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    # --- overnight gap ---
    gap = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(O[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        gap[m, di] = (O[m, di] - C[m, di - 1]) / C[m, di - 1]

    # --- intraday return (close-to-close relative to open) ---
    intraday_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(O[:, di]) & (O[:, di] > 0)
        intraday_ret[m, di] = (C[m, di] - O[m, di]) / O[m, di]

    # =================================================================
    # R_GAP_REVERSAL: gap direction opposite to intraday direction
    # Smart money enters overnight, then price reverses intraday.
    # gap_reversal = -sign(gap) * intraday_ret  (positive = reversal)
    # Smoothed with 5-day rolling mean.
    # =================================================================
    gap_rev = np.full((NS, ND), np.nan)
    m = ~np.isnan(gap) & ~np.isnan(intraday_ret)
    gap_rev[m] = -np.sign(gap[m]) * intraday_ret[m]
    # Also scale by gap magnitude — larger gaps are more meaningful
    gap_rev_scaled = np.full((NS, ND), np.nan)
    m2 = m & (~np.isnan(gap))
    gap_rev_scaled[m2] = gap_rev[m2] * np.abs(gap[m2])
    gap_rev_5 = _rolling_mean(gap_rev_scaled, 5)
    factors['R_GAP_REVERSAL'] = _rank_normalize(gap_rev_5)
    print(f"  GAP_REVERSAL done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_INST_BODY: Institutional body = |C-O|/(H-L) * rel_volume
    # Large body (high conviction) + high volume = institutional activity
    # =================================================================
    hl_range = np.full((NS, ND), np.nan)
    m_hl = ~np.isnan(H) & ~np.isnan(L)
    hl_range[m_hl] = H[m_hl] - L[m_hl]
    safe_hl = np.where(hl_range > 1e-6, hl_range, np.nan)

    body_ratio = np.full((NS, ND), np.nan)
    m_br = ~np.isnan(C) & ~np.isnan(O) & ~np.isnan(safe_hl)
    body_ratio[m_br] = np.abs(C[m_br] - O[m_br]) / safe_hl[m_br]

    # Relative volume
    vol_20 = _rolling_mean(V, 20)
    rel_vol = np.full((NS, ND), np.nan)
    m_rv = ~np.isnan(V) & ~np.isnan(vol_20) & (vol_20 > 0)
    rel_vol[m_rv] = V[m_rv] / vol_20[m_rv]

    inst_body = np.full((NS, ND), np.nan)
    m_ib = ~np.isnan(body_ratio) & ~np.isnan(rel_vol)
    # Direction matters: bullish body (C>O) * vol is positive signal
    direction = np.full((NS, ND), np.nan)
    m_dir = ~np.isnan(C) & ~np.isnan(O)
    direction[m_dir] = np.sign(C[m_dir] - O[m_dir])
    inst_body[m_ib] = body_ratio[m_ib] * rel_vol[m_ib] * direction[m_ib]
    inst_body_10 = _rolling_mean(inst_body, 10)
    factors['R_INST_BODY'] = _rank_normalize(inst_body_10)
    print(f"  INST_BODY done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_VWAP_DEV: Close deviation from 10-day volume-weighted price
    # proxy for VWAP using sum(V*C)/sum(V) over 10 days.
    # If close > VWAP, there is accumulation pressure.
    # =================================================================
    vc = np.where(~np.isnan(V) & ~np.isnan(C) & (V > 0) & (C > 0), V * C, np.nan)
    vc_10 = _rolling_mean(vc, 10)
    v_10 = _rolling_mean(V, 10)
    vwap_proxy = np.full((NS, ND), np.nan)
    m_vw = ~np.isnan(vc_10) & ~np.isnan(v_10) & (v_10 > 0)
    vwap_proxy[m_vw] = vc_10[m_vw] / v_10[m_vw]

    vwap_dev = np.full((NS, ND), np.nan)
    m_vd = ~np.isnan(C) & ~np.isnan(vwap_proxy) & (vwap_proxy > 0)
    # Use data up to di-1
    C_prev = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        C_prev[:, di] = C[:, di - 1]
    m_vd2 = ~np.isnan(C_prev) & ~np.isnan(vwap_proxy) & (vwap_proxy > 0)
    vwap_dev[m_vd2] = (C_prev[m_vd2] - vwap_proxy[m_vd2]) / vwap_proxy[m_vd2]
    vwap_dev_5 = _rolling_mean(vwap_dev, 5)
    factors['R_VWAP_DEV'] = _rank_normalize(vwap_dev_5)
    print(f"  VWAP_DEV done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_INF_CLOSE_POS: Volume-weighted close position in range
    # close_pos = (C - L) / (H - L), weighted by volume.
    # High = close near high with high volume = informed buying.
    # Rolling 10-day mean.
    # =================================================================
    close_pos = np.full((NS, ND), np.nan)
    m_cp = ~np.isnan(safe_hl) & ~np.isnan(C) & ~np.isnan(L)
    close_pos[m_cp] = (C[m_cp] - L[m_cp]) / safe_hl[m_cp]

    vol_weighted_cp = np.full((NS, ND), np.nan)
    m_vcp = ~np.isnan(close_pos) & ~np.isnan(V) & (V > 0)
    vol_weighted_cp[m_vcp] = close_pos[m_vcp] * V[m_vcp]
    vwcp_10 = _rolling_mean(vol_weighted_cp, 10)
    v_10b = _rolling_mean(V, 10)
    inf_cp = np.full((NS, ND), np.nan)
    m_icp = ~np.isnan(vwcp_10) & ~np.isnan(v_10b) & (v_10b > 0)
    inf_cp[m_icp] = vwcp_10[m_icp] / v_10b[m_icp]
    factors['R_INF_CLOSE_POS'] = _rank_normalize(inf_cp)
    print(f"  INF_CLOSE_POS done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  Smart Money factors done ({time.time()-t0:.1f}s)", flush=True)
    return factors


# ===========================================================================
# 2. MOMENTUM QUALITY
# ===========================================================================

def momentum_quality_factors(NS, ND, C, O, H, L, V):
    """
    Momentum that captures TREND QUALITY, not just raw direction.

    Factors:
      R_RETURN_PER_RISK : 5d return / 5d ATR = return per unit of risk
      R_TREND_CONSIST   : Count of consecutive positive days in last 10
      R_SMOOTH_MOM      : Momentum penalized by path volatility
      R_CALM_TREND      : 10d return * (1 / return_path_std)
    """
    t0 = time.time()
    factors = {}

    # --- daily returns ---
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    # --- ATR(14) ---
    hl_range = np.full((NS, ND), np.nan)
    m_hl = ~np.isnan(H) & ~np.isnan(L)
    hl_range[m_hl] = H[m_hl] - L[m_hl]

    tr_arr = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(H[:, di]) & ~np.isnan(L[:, di])
        tr_arr[m, di] = H[m, di] - L[m, di]
        m2 = m & (~np.isnan(C[:, di - 1]))
        tr_arr[m2, di] = np.maximum(
            tr_arr[m2, di],
            np.maximum(
                np.abs(H[m2, di] - C[m2, di - 1]),
                np.abs(L[m2, di] - C[m2, di - 1])
            )
        )
    atr = _rolling_mean(tr_arr, 14, min_valid=7)
    atr5 = _rolling_mean(tr_arr, 5, min_valid=3)

    # =================================================================
    # R_RETURN_PER_RISK: 5d return / 5d ATR
    # High = strong move relative to noise = quality trend
    # =================================================================
    ret_5 = np.full((NS, ND), np.nan)
    for di in range(6, ND):
        m = ~np.isnan(C[:, di - 1]) & ~np.isnan(C[:, di - 6]) & (C[:, di - 6] > 0)
        ret_5[m, di] = (C[m, di - 1] - C[m, di - 6]) / C[m, di - 6]

    rpr = np.full((NS, ND), np.nan)
    m_rpr = ~np.isnan(ret_5) & ~np.isnan(atr5) & (atr5 > 1e-6)
    rpr[m_rpr] = ret_5[m_rpr] / atr5[m_rpr]
    factors['R_RETURN_PER_RISK'] = _rank_normalize(rpr)
    print(f"  RETURN_PER_RISK done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_TREND_CONSIST: Consecutive positive days in last 10
    # Count of up-days out of last 10, weighted by recency
    # High = consistently trending up
    # =================================================================
    up_day = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di])
        up_day[m, di] = (ret[m, di] > 0).astype(float)

    # Simple: fraction of up days in last 10
    consist = _rolling_mean(up_day, 10)
    # Weighted version: more recent days count more
    decay_weights = np.array([0.5**i for i in range(10)][::-1])  # older to newer
    decay_weights /= decay_weights.sum()
    consist_weighted = np.full((NS, ND), np.nan)
    for di in range(11, ND):
        chunk = up_day[:, di - 10:di]  # 10 days up to di-1
        n_valid = (~np.isnan(chunk)).sum(axis=1)
        valid = n_valid >= 7
        # Weighted sum
        filled = np.where(np.isnan(chunk), 0, chunk)
        wsum = filled @ decay_weights
        consist_weighted[valid, di] = wsum[valid]

    factors['R_TREND_CONSIST'] = _rank_normalize(consist_weighted)
    print(f"  TREND_CONSIST done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_SMOOTH_MOM: 10d return * (1 - return path volatility)
    # Smooth trend = momentum * low path noise
    # =================================================================
    ret_10 = np.full((NS, ND), np.nan)
    for di in range(11, ND):
        m = ~np.isnan(C[:, di - 1]) & ~np.isnan(C[:, di - 11]) & (C[:, di - 11] > 0)
        ret_10[m, di] = (C[m, di - 1] - C[m, di - 11]) / C[m, di - 11]

    ret_std_10 = _rolling_std(ret, 10)
    smooth_mom = np.full((NS, ND), np.nan)
    m_sm = ~np.isnan(ret_10) & ~np.isnan(ret_std_10) & (ret_std_10 > 1e-10)
    # Sharpe-like: return / volatility of daily returns
    smooth_mom[m_sm] = ret_10[m_sm] / ret_std_10[m_sm]
    factors['R_SMOOTH_MOM'] = _rank_normalize(smooth_mom)
    print(f"  SMOOTH_MOM done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_CALM_TREND: 10d absolute return * inverse of path volatility
    # "Calm" trend = strong direction with low daily noise
    # =================================================================
    calm = np.full((NS, ND), np.nan)
    m_cl = ~np.isnan(ret_10) & ~np.isnan(ret_std_10) & (ret_std_10 > 1e-10)
    calm[m_cl] = ret_10[m_cl] / (ret_std_10[m_cl] + 1e-10)
    calm_5 = _rolling_mean(calm, 5)
    factors['R_CALM_TREND'] = _rank_normalize(calm_5)
    print(f"  CALM_TREND done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  Momentum Quality factors done ({time.time()-t0:.1f}s)", flush=True)
    return factors


# ===========================================================================
# 3. SUPPLY/DEMAND IMBALANCE
# ===========================================================================

def supply_demand_factors(NS, ND, C, O, H, L, V):
    """
    Capture genuine buying/selling pressure from price-volume patterns.

    Factors:
      R_DEMAND_DOMINANCE : Close in top 20% of range, 10-day average
      R_ACCUM_DIST       : Volume on up days vs down days ratio
      R_RANGE_ABSORB     : Small range + high volume = absorption
      R_CLOSE_POS_SMA    : Close position within range, smoothed 10d
    """
    t0 = time.time()
    factors = {}

    # --- daily returns ---
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    # --- range ---
    hl_range = np.full((NS, ND), np.nan)
    m_hl = ~np.isnan(H) & ~np.isnan(L)
    hl_range[m_hl] = H[m_hl] - L[m_hl]
    safe_hl = np.where(hl_range > 1e-6, hl_range, np.nan)

    # --- close position in range ---
    close_pos = np.full((NS, ND), np.nan)
    m_cp = ~np.isnan(safe_hl) & ~np.isnan(C) & ~np.isnan(L)
    close_pos[m_cp] = (C[m_cp] - L[m_cp]) / safe_hl[m_cp]

    # =================================================================
    # R_DEMAND_DOMINANCE: fraction of days close is in top 20% of range
    # over the last 10 days. High = persistent demand.
    # =================================================================
    demand_day = np.full((NS, ND), np.nan)
    m_dd = ~np.isnan(close_pos)
    demand_day[m_dd] = (close_pos[m_dd] > 0.8).astype(float)

    demand_10 = _rolling_mean(demand_day, 10)
    factors['R_DEMAND_DOMINANCE'] = _rank_normalize(demand_10)
    print(f"  DEMAND_DOMINANCE done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_ACCUM_DIST: Accumulation/Distribution ratio
    # sum(V * sign(ret>0), 10) / sum(V * sign(ret<0), 10)
    # High = more volume on up days than down days.
    # Use signed volume: V * sign(return)
    # =================================================================
    signed_vol = np.full((NS, ND), np.nan)
    m_sv = ~np.isnan(V) & ~np.isnan(ret) & (V > 0)
    signed_vol[m_sv] = V[m_sv] * np.sign(ret[m_sv])

    # Accumulation = EMA-smoothed signed volume, normalized by total volume
    sv_ema = _ema(signed_vol, 10)
    vol_ema = _ema(V, 10)
    accum_dist = np.full((NS, ND), np.nan)
    m_ad = ~np.isnan(sv_ema) & ~np.isnan(vol_ema) & (vol_ema > 0)
    accum_dist[m_ad] = sv_ema[m_ad] / vol_ema[m_ad]
    factors['R_ACCUM_DIST'] = _rank_normalize(accum_dist)
    print(f"  ACCUM_DIST done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_RANGE_ABSORB: Small range + high volume = absorption
    # Institutions absorb supply without price movement.
    # absorption = relative_volume / (range / ATR)
    # High absorption = high vol in tight range.
    # =================================================================
    tr_arr = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(H[:, di]) & ~np.isnan(L[:, di])
        tr_arr[m, di] = H[m, di] - L[m, di]
        m2 = m & (~np.isnan(C[:, di - 1]))
        tr_arr[m2, di] = np.maximum(
            tr_arr[m2, di],
            np.maximum(
                np.abs(H[m2, di] - C[m2, di - 1]),
                np.abs(L[m2, di] - C[m2, di - 1])
            )
        )
    atr = _rolling_mean(tr_arr, 14, min_valid=7)

    # Range normalized by ATR
    range_norm = np.full((NS, ND), np.nan)
    m_rn = ~np.isnan(hl_range) & ~np.isnan(atr) & (atr > 1e-6)
    range_norm[m_rn] = hl_range[m_rn] / atr[m_rn]

    # Relative volume
    vol_20 = _rolling_mean(V, 20)
    rel_vol = np.full((NS, ND), np.nan)
    m_rv = ~np.isnan(V) & ~np.isnan(vol_20) & (vol_20 > 0)
    rel_vol[m_rv] = V[m_rv] / vol_20[m_rv]

    # Absorption = high rel_vol / low range_norm
    absorb = np.full((NS, ND), np.nan)
    m_ab = ~np.isnan(rel_vol) & ~np.isnan(range_norm) & (range_norm > 0.01)
    absorb[m_ab] = rel_vol[m_ab] / range_norm[m_ab]
    absorb_10 = _rolling_mean(absorb, 10)
    factors['R_RANGE_ABSORB'] = _rank_normalize(absorb_10)
    print(f"  RANGE_ABSORB done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_CLOSE_POS_SMA: Close position in range, EMA smoothed 10d
    # Simple but effective: where does close sit in the day's range?
    # Average over 10 days. 0=always at low, 100=always at high.
    # =================================================================
    cp_ema10 = _ema(close_pos, 10)
    factors['R_CLOSE_POS_SMA'] = _rank_normalize(cp_ema10)
    print(f"  CLOSE_POS_SMA done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  Supply/Demand factors done ({time.time()-t0:.1f}s)", flush=True)
    return factors


# ===========================================================================
# 4. MARKET PHASE AWARENESS
# ===========================================================================

def market_phase_factors(NS, ND, C, O, H, L, V):
    """
    Detect WHERE the stock is in its price cycle.

    Factors:
      R_BREAKOUT_PROX   : How close price is to 20d high (1=at high, 0=at low)
      R_CONTRA_EXPAND    : Narrow ranges followed by wide = breakout imminent
      R_REL_VOLUME       : Today's volume / 20d average volume
      R_RANGE_PCT        : Where is current range in 20d range history
    """
    t0 = time.time()
    factors = {}

    # --- range ---
    hl_range = np.full((NS, ND), np.nan)
    m_hl = ~np.isnan(H) & ~np.isnan(L)
    hl_range[m_hl] = H[m_hl] - L[m_hl]
    safe_hl = np.where(hl_range > 1e-6, hl_range, np.nan)

    # --- ATR ---
    tr_arr = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(H[:, di]) & ~np.isnan(L[:, di])
        tr_arr[m, di] = H[m, di] - L[m, di]
        m2 = m & (~np.isnan(C[:, di - 1]))
        tr_arr[m2, di] = np.maximum(
            tr_arr[m2, di],
            np.maximum(
                np.abs(H[m2, di] - C[m2, di - 1]),
                np.abs(L[m2, di] - C[m2, di - 1])
            )
        )
    atr = _rolling_mean(tr_arr, 14, min_valid=7)

    # =================================================================
    # R_BREAKOUT_PROX: Where is current price in the 20-day range?
    # (C - low20) / (high20 - low20)
    # 1.0 = at 20d high (breakout), 0.0 = at 20d low (breakdown)
    # =================================================================
    breakout_prox = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        # Use data up to di-1
        h20 = H[:, di - 21:di - 1]  # 20 days
        l20 = L[:, di - 21:di - 1]
        c_now = C[:, di - 1]
        for si in range(NS):
            if np.isnan(c_now[si]):
                continue
            hv = h20[si, ~np.isnan(h20[si])]
            lv = l20[si, ~np.isnan(l20[si])]
            if len(hv) < 10 or len(lv) < 10:
                continue
            h_max = np.max(hv)
            l_min = np.min(lv)
            rng = h_max - l_min
            if rng <= 0:
                continue
            breakout_prox[si, di] = (c_now[si] - l_min) / rng
    factors['R_BREAKOUT_PROX'] = _rank_normalize(breakout_prox)
    print(f"  BREAKOUT_PROX done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_CONTRA_EXPAND: Contraction-Expansion cycle
    # Ratio of recent range to prior range.
    # Narrow then wide = breakout coming.
    # range_recent / range_prior where:
    #   range_recent = ATR(5) (last 5 days)
    #   range_prior = ATR(5) from 10 days ago
    # High ratio = expansion after contraction = breakout signal
    # =================================================================
    atr5 = _rolling_mean(tr_arr, 5, min_valid=3)
    atr5_lagged = np.full((NS, ND), np.nan)
    # Shift by 5 days: atr5_lagged[di] = atr5[di-5]
    for di in range(6, ND):
        atr5_lagged[:, di] = atr5[:, di - 5]

    contra_expand = np.full((NS, ND), np.nan)
    m_ce = ~np.isnan(atr5) & ~np.isnan(atr5_lagged) & (atr5_lagged > 1e-6)
    contra_expand[m_ce] = atr5[m_ce] / atr5_lagged[m_ce]

    # We want: low prior range + high recent range = expansion
    # Already captured by ratio > 1. But also check that prior was low:
    # Use interaction: contra_expand * (1 / atr5_lagged) normalized
    ce_5 = _rolling_mean(contra_expand, 5)
    factors['R_CONTRA_EXPAND'] = _rank_normalize(ce_5)
    print(f"  CONTRA_EXPAND done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_REL_VOLUME: Today's volume / 20-day average volume
    # High = unusual activity, possible institutional or event
    # =================================================================
    vol_20 = _rolling_mean(V, 20)
    rel_vol = np.full((NS, ND), np.nan)
    m_rv = ~np.isnan(V) & ~np.isnan(vol_20) & (vol_20 > 0)
    rel_vol[m_rv] = V[m_rv] / vol_20[m_rv]
    rel_vol_5 = _rolling_mean(rel_vol, 5)
    factors['R_REL_VOLUME'] = _rank_normalize(rel_vol_5)
    print(f"  REL_VOLUME done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_RANGE_PCT: Where is today's range in its 20-day range history?
    # Percentile of today's range within last 20 days.
    # High = expansion phase, Low = contraction phase.
    # Useful for combining with momentum.
    # =================================================================
    range_pct = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        r_now = hl_range[:, di - 1]  # yesterday's range
        r_20 = hl_range[:, di - 21:di - 1]  # 20 days of ranges
        for si in range(NS):
            if np.isnan(r_now[si]) or r_now[si] <= 0:
                continue
            rv = r_20[si, ~np.isnan(r_20[si]) & (r_20[si] > 0)]
            if len(rv) < 10:
                continue
            pct = np.sum(rv < r_now[si]) / len(rv) * 100
            range_pct[si, di] = pct

    factors['R_RANGE_PCT'] = _rank_normalize(range_pct)
    print(f"  RANGE_PCT done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  Market Phase factors done ({time.time()-t0:.1f}s)", flush=True)
    return factors


# ===========================================================================
# 5. MEAN REVERSION TIMING
# ===========================================================================

def mean_reversion_factors(NS, ND, C, O, H, L, V):
    """
    Time mean-reversion entries using price extremes and volume patterns.

    Factors:
      R_VWAP_DISTANCE   : Distance from 10d VWAP-like proxy
      R_OVERSOLD_BOUNCE  : 3+ down days followed by up day + volume
      R_MOM_SURPRISE     : Sudden cross-sectional rank improvement
      R_EXTREME_REVERT   : Oversold stocks with volume confirmation
    """
    t0 = time.time()
    factors = {}

    # --- daily returns ---
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    # --- range ---
    hl_range = np.full((NS, ND), np.nan)
    m_hl = ~np.isnan(H) & ~np.isnan(L)
    hl_range[m_hl] = H[m_hl] - L[m_hl]
    safe_hl = np.where(hl_range > 1e-6, hl_range, np.nan)

    # =================================================================
    # R_VWAP_DISTANCE: Distance from 10-day volume-weighted price
    # Oversold relative to VWAP = bounce candidate
    # NEGATIVE = oversold, so we use -distance for rank (higher = more oversold)
    # =================================================================
    vc = np.where(~np.isnan(V) & ~np.isnan(C) & (V > 0) & (C > 0), V * C, np.nan)
    vc_10 = _rolling_mean(vc, 10)
    v_10 = _rolling_mean(V, 10)
    vwap_proxy = np.full((NS, ND), np.nan)
    m_vw = ~np.isnan(vc_10) & ~np.isnan(v_10) & (v_10 > 0)
    vwap_proxy[m_vw] = vc_10[m_vw] / v_10[m_vw]

    vwap_dist = np.full((NS, ND), np.nan)
    # Use C[:, di-1] (yesterday's close)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di - 1]) & ~np.isnan(vwap_proxy[:, di]) & (vwap_proxy[:, di] > 0)
        vwap_dist[m, di] = (C[m, di - 1] - vwap_proxy[m, di]) / vwap_proxy[m, di]

    # Negative = oversold = buy signal
    vwap_dist_neg = -vwap_dist
    vwap_dist_5 = _rolling_mean(vwap_dist_neg, 5)
    factors['R_VWAP_DISTANCE'] = _rank_normalize(vwap_dist_5)
    print(f"  VWAP_DISTANCE done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_OVERSOLD_BOUNCE: 3+ consecutive down days, then up day with volume
    # This is a classic mean-reversion setup.
    # We compute a score: how many consecutive down days (up to 5),
    # multiplied by whether today is up, weighted by volume.
    # =================================================================
    # Consecutive down days (count)
    consec_down = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di])
        count = np.zeros(NS)
        # Look back up to 5 days
        for lookback in range(1, 6):
            idx = di - lookback
            if idx < 0:
                break
            m_lb = ~np.isnan(ret[:, idx])
            is_down = np.zeros(NS)
            is_down[m_lb] = (ret[m_lb, idx] < 0).astype(float)
            count += is_down
            # If any day was not down, stop counting
            broke = (is_down == 0) | np.isnan(ret[:, idx])
            # Reset count where streak broke
            count[broke] = 0

        consec_down[m, di] = count[m]

    # Up day with volume confirmation
    up_today = np.full((NS, ND), np.nan)
    vol_20 = _rolling_mean(V, 20)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di]) & ~np.isnan(V[:, di]) & (V[:, di] > 0)
        up_t = np.zeros(NS)
        up_t[m] = (ret[m, di] > 0).astype(float)
        # Volume factor
        vol_f = np.ones(NS)
        m_vf = m & ~np.isnan(vol_20[:, di]) & (vol_20[:, di] > 0)
        vol_f[m_vf] = np.minimum(V[m_vf, di] / vol_20[m_vf, di], 3.0)
        up_today[m, di] = up_t[m] * vol_f[m]

    # Bounce score = consecutive_downs * up_today * volume_factor
    bounce = np.full((NS, ND), np.nan)
    m_bn = ~np.isnan(consec_down) & ~np.isnan(up_today)
    bounce[m_bn] = consec_down[m_bn] * up_today[m_bn]
    bounce_5 = _rolling_mean(bounce, 5)
    factors['R_OVERSOLD_BOUNCE'] = _rank_normalize(bounce_5)
    print(f"  OVERSOLD_BOUNCE done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_MOM_SURPRISE: Sudden cross-sectional rank improvement
    # How much has this stock's return rank jumped in the last 3 days?
    # A stock suddenly outperforming = positive surprise.
    # =================================================================
    # Cross-sectional rank of daily return
    ret_rank = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        vals = ret[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        ret_rank[valid, di] = ranks / n * 100

    # Change in rank over 3 days
    rank_delta = np.full((NS, ND), np.nan)
    for di in range(4, ND):
        m = ~np.isnan(ret_rank[:, di]) & ~np.isnan(ret_rank[:, di - 3])
        rank_delta[m, di] = ret_rank[m, di] - ret_rank[m, di - 3]

    rank_delta_5 = _rolling_mean(rank_delta, 5)
    factors['R_MOM_SURPRISE'] = _rank_normalize(rank_delta_5)
    print(f"  MOM_SURPRISE done ({time.time()-t0:.1f}s)", flush=True)

    # =================================================================
    # R_EXTREME_REVERT: Stocks at extreme low close position + volume
    # Close in bottom 20% of range + high volume = capitulation
    # Buy the capitulation (mean reversion).
    # =================================================================
    close_pos = np.full((NS, ND), np.nan)
    m_cp = ~np.isnan(safe_hl) & ~np.isnan(C) & ~np.isnan(L)
    close_pos[m_cp] = (C[m_cp] - L[m_cp]) / safe_hl[m_cp]

    # Extreme oversold = close in bottom 20% of range
    extreme = np.full((NS, ND), np.nan)
    m_ex = ~np.isnan(close_pos)
    extreme[m_ex] = np.where(close_pos[m_ex] < 0.2, 1.0 - close_pos[m_ex], 0.0)
    # Weight by relative volume
    rel_vol = np.full((NS, ND), np.nan)
    m_rv = ~np.isnan(V) & ~np.isnan(vol_20) & (vol_20 > 0)
    rel_vol[m_rv] = V[m_rv] / vol_20[m_rv]

    extreme_vol = np.full((NS, ND), np.nan)
    m_ev = ~np.isnan(extreme) & ~np.isnan(rel_vol)
    extreme_vol[m_ev] = extreme[m_ev] * rel_vol[m_ev]
    extreme_5 = _rolling_mean(extreme_vol, 5)
    factors['R_EXTREME_REVERT'] = _rank_normalize(extreme_5)
    print(f"  EXTREME_REVERT done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  Mean Reversion factors done ({time.time()-t0:.1f}s)", flush=True)
    return factors


# ===========================================================================
# MAIN: Combine all V63 factors
# ===========================================================================

def compute_v63_factors(NS, ND, C, O, H, L, V):
    """Compute all V63 market-essence factors."""
    t_total = time.time()
    print("[V63] Computing market-essence factors...", flush=True)

    all_factors = {}

    # 1. Smart Money Detection
    sm = smart_money_factors(NS, ND, C, O, H, L, V)
    all_factors.update(sm)

    # 2. Momentum Quality
    mq = momentum_quality_factors(NS, ND, C, O, H, L, V)
    all_factors.update(mq)

    # 3. Supply/Demand Imbalance
    sd = supply_demand_factors(NS, ND, C, O, H, L, V)
    all_factors.update(sd)

    # 4. Market Phase Awareness
    mp = market_phase_factors(NS, ND, C, O, H, L, V)
    all_factors.update(mp)

    # 5. Mean Reversion Timing
    mr = mean_reversion_factors(NS, ND, C, O, H, L, V)
    all_factors.update(mr)

    print(f"\n  V63 total: {len(all_factors)} factors ({time.time()-t_total:.1f}s)", flush=True)
    print(f"  Factor names: {sorted(all_factors.keys())}", flush=True)
    return all_factors


# ===========================================================================
# MAIN BLOCK: Test all V63 factors
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V63 — Market Essence Factors")
    print("  Target: fundamentally better strategies from market essence", flush=True)
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    print(f"  Data loaded: NS={NS}, ND={ND}", flush=True)

    # --- Test 1: Compute V63 factors only (validate no errors) ---
    print("\n  === Computing V63 factors ===", flush=True)
    v63_factors = compute_v63_factors(NS, ND, C, O, H, L, V)

    # Validate: check shapes, NaN counts
    print("\n  === Factor Validation ===", flush=True)
    for name in sorted(v63_factors.keys()):
        arr = v63_factors[name]
        valid = ~np.isnan(arr)
        n_valid = valid.sum()
        # Count valid on day MIN_TRAIN (first possible trading day)
        if ND > 500:
            valid_cs = (~np.isnan(arr[:, 500])).sum()
        else:
            valid_cs = 0
        print(f"  {name:<25s} shape={arr.shape} "
              f"valid={n_valid:>8d}/{arr.size} "
              f"cross-section@500={valid_cs}", flush=True)

    # --- Test 2: Compute existing factors for combo tests ---
    print("\n  === Computing existing factors ===", flush=True)
    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)

    # Import V49 if available
    try:
        from alpha_v49 import compute_v49_factors
        v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    except ImportError:
        v49 = {}

    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)

    all_factors = {**v41, **v48, **v49, **v52, **v55, **v63_factors}

    v63_names = sorted(v63_factors.keys())
    print(f"\n  New V63 factors: {len(v63_names)}", flush=True)

    results = []

    # =====================================================================
    # Test A: Each V63 factor SOLO
    # =====================================================================
    print("\n  === Test A: V63 factors solo ===", flush=True)
    for fname in v63_names:
        for atr in [0.5, 0.8]:
            r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{fname}_SOLO_A{atr}'
                results.append(r)
                print(f"    {fname}_A{atr}: {r['ann']:+.1f}% WR={r['wr']:.0f}% DD={r['max_dd']:.1f}%",
                      flush=True)

    # =====================================================================
    # Test B: V56 base + each V63 factor
    # =====================================================================
    print("\n  === Test B: V56 + V63 factors ===", flush=True)
    v54_base = {'R_BWP_BNW': 0.205, 'R_TENSION': 0.205, 'R_VWCM': 0.205,
                'R_BVR': 0.154, 'R_BUY_FRAC': 0.138, 'R_VPIN': 0.092}
    v56_weights = {**v54_base, 'R_SHOCK_MOM': 0.08, 'R_TREND_ACC': 0.15}
    total = sum(v56_weights.values())
    v56_norm = {k: v / total for k, v in v56_weights.items()}

    for fname in v63_names:
        for w in [0.05, 0.10, 0.15]:
            weights = {**v56_norm, fname: w}
            tot = sum(weights.values())
            wn = {k: v / tot for k, v in weights.items()}
            r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.5)
            if r:
                r['test'] = f'V56+{fname}_W{w:.2f}'
                results.append(r)
                print(f"    V56+{fname}_W{w:.2f}: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%",
                      flush=True)

    # =====================================================================
    # Test C: Pure V63 combos (equal weight top solo factors)
    # =====================================================================
    print("\n  === Test C: Pure V63 combos ===", flush=True)
    solo_results = sorted([r for r in results if '_SOLO_' in r['test']],
                          key=lambda x: -x['ann'])
    top_solo_names = []
    for r in solo_results[:8]:
        name = r['test'].split('_SOLO_')[0]
        if name not in top_solo_names:
            top_solo_names.append(name)

    if len(top_solo_names) >= 3:
        # Equal weight top 3-6
        for n_top in [3, 4, 5, 6]:
            if n_top > len(top_solo_names):
                break
            weights = {f: 1.0 / n_top for f in top_solo_names[:n_top]}
            for atr in [0.5, 0.7, 0.8]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'V63_EQ{n_top}_A{atr}'
                    results.append(r)
                    print(f"    V63_EQ{n_top}_A{atr}: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%",
                          flush=True)

    # =====================================================================
    # Test D: Category combos (one from each category)
    # =====================================================================
    print("\n  === Test D: Category combos ===", flush=True)
    categories = {
        'smart_money': [n for n in v63_names if n.startswith('R_GAP') or n.startswith('R_INST')
                        or n.startswith('R_VWAP_DEV') or n.startswith('R_INF')],
        'mom_quality': [n for n in v63_names if n.startswith('R_RETURN') or n.startswith('R_TREND_C')
                        or n.startswith('R_SMOOTH') or n.startswith('R_CALM')],
        'supply_demand': [n for n in v63_names if n.startswith('R_DEMAND') or n.startswith('R_ACCUM')
                          or n.startswith('R_RANGE_ABS') or n.startswith('R_CLOSE_P')],
        'phase': [n for n in v63_names if n.startswith('R_BREAK') or n.startswith('R_CONTRA')
                  or n.startswith('R_REL_VOL') or n.startswith('R_RANGE_P')],
        'reversion': [n for n in v63_names if n.startswith('R_VWAP_D') or n.startswith('R_OVER')
                      or n.startswith('R_MOM_SUR') or n.startswith('R_EXTREME')],
    }

    # Pick best solo from each category and combine
    cat_best = {}
    for cat, names in categories.items():
        best_r = None
        for n in names:
            for r in solo_results:
                if r['test'].startswith(n + '_SOLO'):
                    if best_r is None or r['ann'] > best_r['ann']:
                        best_r = r
                        cat_best[cat] = n

    if len(cat_best) >= 3:
        cat_names = list(cat_best.values())
        weights = {f: 1.0 / len(cat_names) for f in cat_names}
        for atr in [0.5, 0.6, 0.7, 0.8]:
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'V63_CATCOMBO_A{atr}'
                results.append(r)
                print(f"    CATCOMBO_A{atr}: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # Test E: V63 best + V56 with weight sweep
    # =====================================================================
    print("\n  === Test E: V56 + best V63 weight sweep ===", flush=True)
    if top_solo_names:
        best_v63 = top_solo_names[0]
        for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
            weights = {**v56_norm, best_v63: w}
            tot = sum(weights.values())
            wn = {k: v / tot for k, v in weights.items()}
            for atr in [0.5, 0.7]:
                r = backtest_v7c(wn, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    r['test'] = f'V56+{best_v63}_W{w:.2f}_A{atr}'
                    results.append(r)
        print(f"    {best_v63} weight sweep done", flush=True)

    # =====================================================================
    # RESULTS SUMMARY
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'=' * 110}", flush=True)
    print(f"  ALL RESULTS (V63 MARKET ESSENCE FACTORS)", flush=True)
    print(f"  {'Test':<55s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-' * 100}", flush=True)
    for r in results[:80]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<55s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Solo summary
    print(f"\n  === SOLO V63 FACTOR SUMMARY ===", flush=True)
    solo_sorted = sorted([r for r in results if '_SOLO_' in r['test']], key=lambda x: -x['ann'])
    for r in solo_sorted:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<55s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    # V56+ summary
    print(f"\n  === V56 + V63 BEST ===", flush=True)
    v56_new = sorted([r for r in results if r['test'].startswith('V56+')], key=lambda x: -x['ann'])
    for r in v56_new[:20]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<55s} {r['ann']:+7.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i + 1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%",
                  flush=True)

    if results:
        best = results[0]
        print(f"\n  === V63 BEST ===", flush=True)
        print(f"  V63: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)

    print(f"\n{'=' * 70}", flush=True)
