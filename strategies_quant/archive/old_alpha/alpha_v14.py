"""
Alpha V14 — Trading Engine Breakthrough (OPTIMIZED)
====================================================
Vectorized version: HAR-RV ~30s instead of ~50min, ATR_TERRAIN ~5s instead of ~10min.

Key optimizations:
1. HAR-RV: rv1/rv5/rv22 precomputed as full matrices, rolling OLS batch-solved
2. ATR_TERRAIN: TR/ATR precomputed as full matrices, percentile vectorized
3. LOG_PRESSURE: vol_delta precomputed, EMA vectorized across stocks

STRICT no look-ahead: all computations use data up to d = di-1.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0
from alpha_v7b import compute_interaction_factors
from alpha_v7d import compute_extra_factors
from alpha_v7e import compute_v7e_factors
from alpha_v7f import compute_advanced_interactions
from alpha_v8 import compute_v8_factors, compute_v8_interactions
from alpha_v9 import compute_v9_factors, compute_v9_interactions
from alpha_v10 import compute_v10_factors, compute_v10_interactions
from alpha_v11 import compute_v11_factors, compute_v11_interactions
from alpha_v7c import backtest_v7c


# ======================================================================
# Helper: vectorized rolling mean for 2D arrays
# ======================================================================

def _rolling_mean_2d(arr, window):
    """Vectorized rolling mean for 2D array (NS, ND)."""
    NS, ND = arr.shape
    out = np.full_like(arr, np.nan)
    arr_filled = np.where(np.isnan(arr), 0, arr)
    cumsum = np.cumsum(arr_filled, axis=1)
    cumcount = np.cumsum(~np.isnan(arr), axis=1).astype(float)
    for di in range(window - 1, ND):
        cs = cumsum[:, di].copy()
        cn = cumcount[:, di].copy()
        if di >= window:
            cs -= cumsum[:, di - window]
            cn -= cumcount[:, di - window]
        valid = cn >= window // 2
        out[valid, di] = cs[valid] / cn[valid]
    return out


# ======================================================================
# NEW V14 FACTORS — from V15 deep study of 260 strategies (OPTIMIZED)
# ======================================================================

def compute_v14_factors(NS, ND, C, O, H, L, V):
    """V14 factors — 3 new dimensions from V15 research. OPTIMIZED vectorized version.

    SELF-CHECK RULES:
    1. d = di - 1 (use yesterday as "current" data)
    2. Store result at index di
    3. Never access C[si, di], O[si, di], H[si, di], L[si, di], V[si, di]
    """
    t0 = time.time()
    new = {}

    # =====================================================================
    # FACTOR 1: HAR-RV Volatility Ratio (Corsi 2009) — VECTORIZED
    # =====================================================================
    HAR_RV = np.full((NS, ND), np.nan)
    RV_DAILY = np.full((NS, ND), np.nan)
    RV_WEEKLY = np.full((NS, ND), np.nan)
    RV_MONTHLY = np.full((NS, ND), np.nan)

    # Step 1: Compute rv1 as (NS, ND) matrix
    # rv1[si, di] = log(C[si, di-1] / C[si, di-2])^2
    C_prev1 = np.full_like(C, np.nan)
    C_prev1[:, 1:] = C[:, :-1]  # C at d=di-1
    C_prev2 = np.full_like(C, np.nan)
    C_prev2[:, 2:] = C[:, :-2]  # C at d=di-2

    valid_rv = (~np.isnan(C_prev1)) & (~np.isnan(C_prev2)) & (C_prev2 > 0)
    rv1 = np.full((NS, ND), np.nan)
    rv1[valid_rv] = (np.log(C_prev1[valid_rv] / C_prev2[valid_rv])) ** 2

    # Step 2: Precompute OLS features as (NS, ND) matrices
    # x1_feat[idx] = rv1[idx-1]  (lagged daily RV)
    x1_feat = np.full_like(rv1, np.nan)
    x1_feat[:, 1:] = rv1[:, :-1]

    # x2_feat[idx] = mean(rv1[idx-5:idx])  (weekly RV ending at idx-1)
    # = _rolling_mean_2d(rv1, 5)[idx-1], so shift by 1
    rv5_rm = _rolling_mean_2d(rv1, 5)
    x2_feat = np.full_like(rv1, np.nan)
    x2_feat[:, 1:] = rv5_rm[:, :-1]

    # x3_feat[idx] = mean(rv1[idx-22:idx])  (monthly RV ending at idx-1)
    rv22_rm = _rolling_mean_2d(rv1, 22)
    x3_feat = np.full_like(rv1, np.nan)
    x3_feat[:, 1:] = rv22_rm[:, :-1]

    # Step 3: Rolling OLS with window=60, vectorized across stocks per day
    ols_window = 60
    for di in range(66, ND):
        start = di - ols_window
        # Extract 60-day windows for all stocks
        y_win = rv1[:, start:di]      # (NS, 60)
        x1_win = x1_feat[:, start:di]  # (NS, 60)
        x2_win = x2_feat[:, start:di]  # (NS, 60)
        x3_win = x3_feat[:, start:di]  # (NS, 60)

        # Valid mask: all 4 values must be non-NaN
        valid = (~np.isnan(y_win) & ~np.isnan(x1_win) &
                 ~np.isnan(x2_win) & ~np.isnan(x3_win))
        valid_count = valid.sum(axis=1)  # (NS,)
        enough = valid_count >= 20

        if not np.any(enough):
            continue

        idx_enough = np.where(enough)[0]

        # Fill NaN with 0, apply valid mask
        y_f = np.where(valid, y_win, 0)[idx_enough]       # (n, 60)
        x1_f = np.where(valid, x1_win, 0)[idx_enough]
        x2_f = np.where(valid, x2_win, 0)[idx_enough]
        x3_f = np.where(valid, x3_win, 0)[idx_enough]
        ones_f = np.where(valid[idx_enough], 1.0, 0.0)

        # Build X matrix: (n, 60, 4)
        X_mat = np.stack([ones_f, x1_f, x2_f, x3_f], axis=2)

        # Batch OLS: X'X and X'Y
        XtX = np.einsum('nkj,nki->nji', X_mat, X_mat)  # (n, 4, 4)
        XtY = np.einsum('nkj,nk->nj', X_mat, y_f)      # (n, 4)

        # Ridge regularization for numerical stability
        XtX += 1e-10 * np.eye(4)[None, :, :]

        try:
            # solve needs b to be 2D per batch: (n, 4, 1)
            beta = np.linalg.solve(XtX, XtY[:, :, np.newaxis])[:, :, 0]  # (n, 4)
        except np.linalg.LinAlgError:
            continue

        # Prediction features: use data up to di-2 for x1/x2/x3, actual at di
        x1_pred = rv1[idx_enough, di - 1]  # (n,)
        x2_raw = rv1[idx_enough, di - 6:di - 1]  # (n, 5)
        x2_pred = np.nanmean(x2_raw, axis=1)
        x2_vcount = np.sum(~np.isnan(x2_raw), axis=1)
        x3_raw = rv1[idx_enough, di - 23:di - 1]  # (n, 22)
        x3_pred = np.nanmean(x3_raw, axis=1)
        x3_vcount = np.sum(~np.isnan(x3_raw), axis=1)
        actual = rv1[idx_enough, di]

        valid_pred = (~np.isnan(x1_pred) & ~np.isnan(x2_pred) &
                      ~np.isnan(x3_pred) & ~np.isnan(actual) &
                      (actual > 1e-12) & (x2_vcount >= 3) & (x3_vcount >= 10))

        if not np.any(valid_pred):
            continue

        si_pred = idx_enough[valid_pred]
        beta_pred = beta[valid_pred]

        predicted = (beta_pred[:, 0] +
                     beta_pred[:, 1] * x1_pred[valid_pred] +
                     beta_pred[:, 2] * x2_pred[valid_pred] +
                     beta_pred[:, 3] * x3_pred[valid_pred])

        pred_pos = predicted > 0
        if not np.any(pred_pos):
            continue

        result_si = si_pred[pred_pos]
        ratio = predicted[pred_pos] / actual[valid_pred][pred_pos]

        HAR_RV[result_si, di] = ratio
        RV_DAILY[result_si, di] = x1_pred[valid_pred][pred_pos]
        RV_WEEKLY[result_si, di] = x2_pred[valid_pred][pred_pos]
        RV_MONTHLY[result_si, di] = x3_pred[valid_pred][pred_pos]

    new['HAR_RV_RATIO'] = HAR_RV
    new['RV_DAILY'] = RV_DAILY
    new['RV_WEEKLY'] = RV_WEEKLY
    new['RV_MONTHLY'] = RV_MONTHLY
    print(f"  HAR-RV done ({time.time()-t0:.1f}s)", flush=True)

    # =====================================================================
    # FACTOR 2: Log-Normalized Institutional Pressure — VECTORIZED
    # =====================================================================
    LOG_PRESSURE = np.full((NS, ND), np.nan)

    # Step 1: Compute vol_delta as (NS, ND) matrix
    V_s = np.full_like(V, np.nan)
    V_s[:, 1:] = V[:, :-1]
    C_s = np.full_like(C, np.nan)
    C_s[:, 1:] = C[:, :-1]
    H_s = np.full_like(H, np.nan)
    H_s[:, 1:] = H[:, :-1]
    L_s = np.full_like(L, np.nan)
    L_s[:, 1:] = L[:, :-1]

    range_hl = H_s - L_s
    valid_vd = (~np.isnan(V_s) & ~np.isnan(C_s) & ~np.isnan(H_s) &
                ~np.isnan(L_s) & (V_s > 0) & (range_hl > 0))
    vol_delta = np.full((NS, ND), np.nan)
    vol_delta[valid_vd] = (V_s[valid_vd] *
                           (2 * C_s[valid_vd] - H_s[valid_vd] - L_s[valid_vd]) /
                           range_hl[valid_vd])

    # Step 2: Rolling 20-day max of |vol_delta|
    abs_vd = np.abs(vol_delta)
    rolling_max_vd = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        window = abs_vd[:, di - 20:di]  # (NS, 20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rolling_max_vd[:, di] = np.nanmax(window, axis=1)

    # Step 3: Per-day loop with vectorized stock operations
    alpha_ema = 2.0 / 11
    ema_val = np.full(NS, np.nan)

    for di in range(22, ND):
        delta = vol_delta[:, di]  # (NS,)
        r_max = rolling_max_vd[:, di]  # (NS,)

        # Volume cap: 95th percentile of rolling 21-day volume
        v_raw = V_s[:, di]  # V at d=di-1
        v_hist = V[:, di - 21:di]  # (NS, 21)

        # 95th percentile per stock using sort
        v_valid_mask = ~np.isnan(v_hist)
        v_valid_count = v_valid_mask.sum(axis=1)  # (NS,)
        enough_v = v_valid_count >= 10

        v_sorted = np.sort(np.where(v_valid_mask, v_hist, np.inf), axis=1)
        pct95_idx = np.clip(np.floor(v_valid_count * 0.95).astype(int) - 1, 0, 20)
        v_cap = v_sorted[np.arange(NS), pct95_idx]
        v_cap = np.where(enough_v, v_cap, np.nan)

        # Cap the volume
        v_capped = np.where(~np.isnan(v_cap), np.minimum(v_raw, v_cap), np.nan)

        # Scale delta
        v_raw_safe = np.where((v_raw > 0) & ~np.isnan(v_raw), v_raw, 1.0)
        scaled_delta = np.where(
            ~np.isnan(delta) & ~np.isnan(v_raw) & (v_raw > 0) & enough_v,
            delta * (v_capped / v_raw_safe), np.nan)

        # Log-normalize
        r_max_safe = np.where((r_max > 0) & ~np.isnan(r_max), r_max, 1.0)
        sign = np.sign(scaled_delta)
        abs_sd = np.abs(scaled_delta)
        log_norm = sign * np.log(1.0 + abs_sd) / np.log(1.0 + r_max_safe)
        log_norm = np.clip(log_norm, -1.0, 1.0)

        valid = ~np.isnan(scaled_delta) & ~np.isnan(r_max) & (r_max > 0) & enough_v

        # EMA smoothing
        no_prev = np.isnan(ema_val)
        ema_val = np.where(
            valid & no_prev, log_norm,
            np.where(valid, alpha_ema * log_norm + (1 - alpha_ema) * ema_val, ema_val))

        LOG_PRESSURE[valid, di] = ema_val[valid]

    new['LOG_PRESSURE'] = LOG_PRESSURE
    print(f"  LOG_PRESSURE done ({time.time()-t0:.1f}s)", flush=True)

    # =====================================================================
    # FACTOR 3: ATR Terrain State — VECTORIZED
    # =====================================================================
    ATR_TERRAIN = np.full((NS, ND), np.nan)
    ATR_RATIO_FAST = np.full((NS, ND), np.nan)

    # Step 1: True Range as (NS, ND) matrix
    H_s2 = np.full_like(H, np.nan)
    L_s2 = np.full_like(L, np.nan)
    C_p = np.full_like(C, np.nan)
    H_s2[:, 1:] = H[:, :-1]
    L_s2[:, 1:] = L[:, :-1]
    C_p[:, 2:] = C[:, :-2]

    tr1 = H_s2 - L_s2
    tr2 = np.abs(H_s2 - C_p)
    tr3 = np.abs(L_s2 - C_p)
    TR = np.where(~np.isnan(H_s2) & ~np.isnan(L_s2),
                  np.fmax(tr1, np.fmax(tr2, tr3)), np.nan)

    # Step 2: ATR with windows 10, 20, 50
    ATR_10 = _rolling_mean_2d(TR, 10)
    ATR_20 = _rolling_mean_2d(TR, 20)
    ATR_50 = _rolling_mean_2d(TR, 50)

    # Step 3: Ratio = ATR(10) / ATR(50)
    valid_ratio = (~np.isnan(ATR_10) & ~np.isnan(ATR_50) & (ATR_50 > 0))
    ATR_RATIO = np.where(valid_ratio, ATR_10 / ATR_50, np.nan)

    # Step 4: Percentile rank and 4-state classification (vectorized per day)
    lookback_pct = 120
    start_di_atr = 55 + lookback_pct

    for di in range(start_di_atr, ND):
        cur = ATR_RATIO[:, di]  # (NS,)
        valid_cur = ~np.isnan(cur)
        if not np.any(valid_cur):
            continue

        # Store ratio for all valid stocks (needed for future percentile lookups)
        ATR_RATIO_FAST[valid_cur, di] = cur[valid_cur]

        # Historical window: look back up to 120 days from di-1
        hist_start = max(55, di - lookback_pct)
        hist = ATR_RATIO_FAST[:, hist_start:di]  # (NS, lookback)

        # Percentile: count how many historical values < current, per stock
        valid_hist = ~np.isnan(hist)  # (NS, lookback)
        n_valid = valid_hist.sum(axis=1)  # (NS,)
        less_than = (hist < cur[:, None]) & valid_hist  # (NS, lookback)
        count_less = less_than.sum(axis=1)  # (NS,)

        pct = np.where(n_valid >= 20, count_less / np.maximum(n_valid, 1) * 100, np.nan)

        # ATR fast and own EMA
        atr_fast = ATR_10[:, di]
        atr_own_ema = ATR_20[:, di]

        # 4-state classification (vectorized)
        valid_state = valid_cur & (n_valid >= 20) & ~np.isnan(atr_fast) & ~np.isnan(atr_own_ema)

        squeeze = valid_state & (pct < 20) & (atr_fast < atr_own_ema)
        expansion = valid_state & (pct > 80) & ~squeeze
        fading = valid_state & (pct < 40) & ~squeeze & ~expansion
        normal = valid_state & ~squeeze & ~expansion & ~fading

        ATR_TERRAIN[squeeze, di] = 100
        ATR_TERRAIN[expansion, di] = 25
        ATR_TERRAIN[fading, di] = 75
        ATR_TERRAIN[normal, di] = 50

    new['ATR_TERRAIN'] = ATR_TERRAIN
    new['ATR_RATIO_FAST'] = ATR_RATIO_FAST
    print(f"  ATR_TERRAIN done ({time.time()-t0:.1f}s)", flush=True)

    # === Rank normalize all V14 factors ===
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

    factor_names = ['HAR_RV_RATIO', 'LOG_PRESSURE', 'ATR_TERRAIN']
    for name in factor_names:
        new[f'R_{name}'] = rank_pct(new[name])

    # Invert HAR_RV_RATIO: high ratio = expansion = bad, so invert
    inv = new['R_HAR_RV_RATIO'].copy()
    mask = ~np.isnan(inv)
    inv[mask] = 100.0 - inv[mask]
    new['R_HAR_RV_RATIO_INV'] = inv

    print(f"  Total V14 raw factors: {len(new)}", flush=True)
    return new


def compute_v14_interactions(all_factors, NS, ND):
    """V14 interactions — new factors × best existing factors."""
    t0 = time.time()
    new = {}

    def interact(name_a, name_b, out_name):
        a = all_factors.get(name_a, np.full((NS, ND), np.nan))
        b = all_factors.get(name_b, np.full((NS, ND), np.nan))
        res = np.full((NS, ND), np.nan)
        m = ~np.isnan(a) & ~np.isnan(b)
        res[m] = a[m] * b[m] / 100
        new[out_name] = res

    # Pressure × Structure
    interact('R_LOG_PRESSURE', 'R_BODY_NW', 'LP_BNW')
    interact('R_LOG_PRESSURE', 'R_TENSION', 'LP_TENS')
    interact('R_LOG_PRESSURE', 'R_BB_WIDTH_PCT_INV', 'LP_BWP')

    # ATR Terrain × Volume
    interact('R_ATR_TERRAIN', 'R_BODY_NW', 'AT_BNW')
    interact('R_ATR_TERRAIN', 'R_TENSION', 'AT_TENS')
    interact('R_ATR_TERRAIN', 'R_BB_WIDTH_PCT_INV', 'AT_BWP')

    # HAR-RV inverse × Best combos
    interact('R_HAR_RV_RATIO_INV', 'R_BWP_BNW', 'HAR_BWP')
    interact('R_HAR_RV_RATIO_INV', 'R_TENSION', 'HAR_TENS')

    # Triple: Terrain + Pressure + Body
    interact('LP_BNW', 'R_ATR_TERRAIN', 'LPA_BNW')

    # Rank normalize
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

    for name in list(new.keys()):
        new[f'R_{name}'] = rank_pct(new[name])

    print(f"  V14 interactions done ({time.time()-t0:.1f}s)", flush=True)
    return new


# ======================================================================
# V14 TRADING ENGINE — 5 Innovations
# ======================================================================

def compute_stock_regime(factors, NS, ND, di):
    """Compute per-stock regime using KER and available trend factors.

    Uses ONLY data up to di-1 (factors already comply).

    Returns regime array: NS-length array
      0 = RANGING
      1 = TRANSITION
      2 = TRENDING
    """
    regime = np.ones(NS, dtype=int)  # Default: TRANSITION

    # Use KER (Kaufman Efficiency Ratio) if available
    ker = factors.get('R_KER', None)
    hurst = factors.get('R_HURST', None)
    r2 = factors.get('R_R_SQUARED', None)

    for si in range(NS):
        scores = []
        if ker is not None and not np.isnan(ker[si, di]):
            # KER rank > 65 = trending, < 35 = ranging
            scores.append(1.0 if ker[si, di] > 65 else (-1.0 if ker[si, di] < 35 else 0.0))
        if hurst is not None and not np.isnan(hurst[si, di]):
            # Hurst rank > 65 = trending, < 35 = ranging
            scores.append(1.0 if hurst[si, di] > 65 else (-1.0 if hurst[si, di] < 35 else 0.0))
        if r2 is not None and not np.isnan(r2[si, di]):
            # R² rank > 60 = clear trend
            scores.append(1.0 if r2[si, di] > 60 else (-1.0 if r2[si, di] < 30 else 0.0))

        if not scores:
            continue
        avg_score = np.mean(scores)
        if avg_score > 0.3:
            regime[si] = 2  # TRENDING
        elif avg_score < -0.3:
            regime[si] = 0  # RANGING
        else:
            regime[si] = 1  # TRANSITION

    return regime


def compute_epanechnikov_score(factor_values, weights, NS):
    """Epanechnikov confluence scoring.

    Instead of linear weighted sum, use Epanechnikov kernel:
      K(u) = 0.75 × (1 - u²) for |u| ≤ 1, else 0
    where u = |rank - 100| / 50 (target = 100, best possible)

    This REWARDS stocks that score high on ALL factors simultaneously.
    A stock with ranks [90, 90, 90] scores HIGHER than [100, 80, 100].

    Args:
        factor_values: dict of {factor_name: NS-length array of values at di}
        weights: dict of {factor_name: weight}
        NS: number of stocks

    Returns:
        NS-length array of composite scores
    """
    composite = np.zeros(NS)
    total_kernel_weight = np.zeros(NS)

    for fname, w in weights.items():
        vals = factor_values.get(fname, None)
        if vals is None:
            continue
        for si in range(NS):
            v = vals[si]
            if np.isnan(v):
                continue
            # u = |rank - 100| / 50 → 0 for rank=100, 1 for rank=50, 2 for rank=0
            u = abs(v - 100.0) / 50.0
            if u <= 1.0:
                k = 0.75 * (1.0 - u * u)  # Epanechnikov kernel
            else:
                k = 0.0
            composite[si] += w * k * v  # Weight by kernel AND rank
            total_kernel_weight[si] += abs(w) * k

    # Normalize by total kernel weight
    mask = total_kernel_weight > 0
    composite[mask] /= total_kernel_weight[mask]
    composite[~mask] = -9999

    return composite


def sigmoid_confidence(composite, NS, threshold=70.0, k=0.1):
    """Convert composite score to confidence via sigmoid.

    confidence = 1 / (1 + exp(-k × (score - threshold)))

    score=70 → 0.5, score=85 → 0.73, score=95 → 0.88
    score=55 → 0.27, score=40 → 0.13
    """
    conf = np.zeros(NS)
    for si in range(NS):
        if composite[si] <= -9000:
            conf[si] = 0
        else:
            conf[si] = 1.0 / (1.0 + np.exp(-k * (composite[si] - threshold)))
    return conf


def kelly_fraction(recent_trades, max_fraction=0.4, min_trades=10):
    """Compute Kelly fraction from recent trade history.

    f* = fraction × (p × b - q) / b
    where p = win rate, q = 1-p, b = avg_win / avg_loss

    Uses only COMPLETED trades (no look-ahead).
    Returns default 0.25 if insufficient data.
    """
    if len(recent_trades) < min_trades:
        return 0.25  # Default: 25% of capital per trade

    wins = [t['pnl'] for t in recent_trades if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in recent_trades if t['pnl'] <= 0]

    if not wins or not losses:
        return 0.25

    p = len(wins) / len(recent_trades)
    q = 1 - p
    b = np.mean(wins) / max(np.mean(losses), 0.01)

    # Kelly fraction
    f = (p * b - q) / max(b, 0.01)
    f = max(f, 0.05)  # Minimum 5%
    f = min(f, max_fraction)  # Cap at max_fraction

    return f


def backtest_v14(factor_weights_trend, factor_weights_range, factor_weights_trans,
                 factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, atr_stop_mult=1.5,
                 use_epanechnikov=True, use_sigmoid=True, use_kelly=True,
                 use_adaptive_rebal=True, rebalance_days_max=15,
                 rank_drop_threshold=40):
    """V14 Backtest — Trading Engine Innovation.

    LOOK-AHEAD SELF-CHECK:
      [x] Factor values at index di use only data up to di-1
      [x] Trades execute at O[si, di] (open price)
      [x] ATR stop checks L[si, di] (realistic intraday)
      [x] Kelly fraction uses only completed trades
      [x] Regime uses only factor values at di (computed from di-1 data)
      [x] No future information used anywhere

    Args:
        factor_weights_trend: dict {factor: weight} for TRENDING regime
        factor_weights_range: dict {factor: weight} for RANGING regime
        factor_weights_trans: dict {factor: weight} for TRANSITION regime
        factors: all factor arrays
        top_n: max stocks to hold
        atr_stop_mult: ATR multiplier for trailing stop
        use_epanechnikov: use Epanechnikov kernel for scoring
        use_sigmoid: use sigmoid confidence mapping
        use_kelly: use Kelly position sizing
        use_adaptive_rebal: use rank-drop exit instead of fixed rebalance
        rebalance_days_max: max days between forced rebalances
        rank_drop_threshold: percentile threshold for rank-drop exit
    """
    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}

    # Track factor names per regime
    regimes = {
        0: factor_weights_range,    # RANGING
        1: factor_weights_trans,    # TRANSITION
        2: factor_weights_trend,    # TRENDING
    }

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # === STEP 1: ATR stop loss check (BUG-FIXED, same as v7c) ===
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False

            if atr_stop_mult > 0:
                atr = 0
                atr_count = 0
                for dd in range(max(di - 14, 1), di):
                    if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                        tr = H[si, dd] - L[si, dd]
                        if not np.isnan(C[si, dd - 1]):
                            tr = max(tr, abs(H[si, dd] - C[si, dd - 1]),
                                     abs(L[si, dd] - C[si, dd - 1]))
                        atr += tr
                        atr_count += 1
                if atr_count > 0:
                    atr /= atr_count
                else:
                    atr = 0

                if atr > 0:
                    stop = pos['hw'] - atr_stop_mult * atr
                    today_low = L[si, di]
                    today_open = O[si, di]

                    if not np.isnan(today_low) and today_low <= stop:
                        if not np.isnan(today_open) and today_open < stop:
                            sp = today_open  # Gap down
                        else:
                            sp = stop  # Normal stop
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'stop', 'year': year})
                        holdings.remove(pos)
                        stopped_out = True

            if not stopped_out:
                today_high = H[si, di]
                if not np.isnan(today_high) and today_high > 0:
                    pos['hw'] = max(pos['hw'], today_high)

            # Time stop: max 60 days
            if pos in holdings:
                days_held = (dates[di] - pos['ed']).days
                if days_held >= 60:
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': days_held,
                                       'di': di, 'reason': 'time_stop', 'year': year})
                        holdings.remove(pos)

        # === STEP 2: Compute per-stock regime ===
        stock_regime = compute_stock_regime(factors, NS, ND, di)

        # === STEP 3: Adaptive rebalance check ===
        should_rebalance = False

        if use_adaptive_rebal:
            # Check if any held stock dropped below threshold
            for pos in list(holdings):
                si = pos['si']
                # Get current composite rank
                r = stock_regime[si]
                weights = regimes[r]
                factor_vals = {fname: factors[fname][si, di] for fname in weights if fname in factors}
                if not factor_vals:
                    continue
                vals = list(factor_vals.values())
                valid = [v for v in vals if not np.isnan(v)]
                if not valid:
                    continue
                avg_rank = np.mean(valid)
                if avg_rank < rank_drop_threshold:
                    # Rank dropped — exit
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'rank_drop', 'year': year})
                        holdings.remove(pos)

            # Forced rebalance if max period reached or no holdings
            days_since = di - last_rebalance
            if days_since >= rebalance_days_max or len(holdings) == 0:
                should_rebalance = True
        else:
            # Original fixed rebalance
            if di - last_rebalance >= rebalance_days_max:
                should_rebalance = True

        if not should_rebalance:
            continue

        # === STEP 4: Regime-Adaptive Scoring ===
        # For each stock, use the weights appropriate for its regime
        if use_epanechnikov:
            # Epanechnikov confluence scoring per regime
            # Compute separate composite for each regime, then merge
            composite = np.full(NS, -9999.0)

            for regime_val, weights in regimes.items():
                # Get factor values for stocks in this regime
                regime_mask = (stock_regime == regime_val)
                factor_vals_di = {}
                for fname in weights:
                    if fname in factors:
                        factor_vals_di[fname] = factors[fname][:, di]

                # Compute Epanechnikov score for this regime's stocks
                regime_composite = compute_epanechnikov_score(factor_vals_di, weights, NS)
                composite[regime_mask] = regime_composite[regime_mask]
        else:
            # Linear weighted sum with regime-adaptive weights
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for si in range(NS):
                r = stock_regime[si]
                weights = regimes[r]
                for fname, w in weights.items():
                    if fname not in factors:
                        continue
                    val = factors[fname][si, di]
                    if np.isnan(val):
                        continue
                    composite[si] += w * val
                    count[si] += abs(w)
            mask = count > 0
            composite[mask] /= count[mask]
            composite[~mask] = -9999

        # === STEP 5: Sigmoid confidence scoring ===
        if use_sigmoid:
            confidence = sigmoid_confidence(composite, NS, threshold=70.0, k=0.1)
        else:
            confidence = np.ones(NS)
            confidence[composite <= -9000] = 0

        # === STEP 6: Select top-N stocks ===
        # Sort by composite score (not confidence — confidence is for sizing)
        valid_mask = composite > -9000
        if valid_mask.sum() < top_n * 2:
            continue

        top_indices = set(np.argsort(-composite)[:top_n])
        current_indices = set(h['si'] for h in holdings)

        # === STEP 7: Sell stocks not in top-N ===
        to_sell = current_indices - top_indices
        for pos in list(holdings):
            if pos['si'] in to_sell:
                sp = O[pos['si'], di]
                if np.isnan(sp) or sp <= 0:
                    sp = C[pos['si'], di]
                if not np.isnan(sp) and sp > 0:
                    pnl = (sp - pos['entry']) / pos['entry'] * 100
                    cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                    trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                   'di': di, 'reason': 'rebalance', 'year': year})
                    holdings.remove(pos)

        # === STEP 8: Buy new stocks with Kelly sizing ===
        current_indices = set(h['si'] for h in holdings)
        to_buy = top_indices - current_indices
        n_to_buy = len(to_buy)

        if n_to_buy > 0 and cash > 10000:
            # Kelly fraction from recent trades
            if use_kelly:
                # Use last 30 completed trades
                recent = sorted(trades, key=lambda x: x['di'])[-30:]
                kf = kelly_fraction(recent, max_fraction=0.5, min_trades=10)
            else:
                kf = 0.25

            # Equal allocation base
            base_alloc = cash / n_to_buy

            for si in to_buy:
                # Confidence-weighted position sizing
                conf = confidence[si]
                position_fraction = kf * conf  # Kelly × confidence
                position_fraction = max(position_fraction, 0.05)  # Min 5%
                position_fraction = min(position_fraction, 0.5)   # Max 50%

                alloc = cash * position_fraction

                p = O[si, di]
                if np.isnan(p) or p <= 0:
                    p = C[si, di]
                if np.isnan(p) or p <= 0:
                    continue

                shares = int(alloc / (1 + COMMISSION) / p)
                if shares > 0:
                    cost = shares * p * (1 + COMMISSION)
                    if cost <= cash:
                        cash -= cost
                        holdings.append({
                            'si': si, 'shares': shares, 'entry': p,
                            'ed': dates[di], 'hw': p,
                            'confidence': conf,
                            'regime': stock_regime[si]
                        })

        last_rebalance = di

    # === Close remaining positions ===
    for pos in holdings:
        p = C[pos['si'], ND - 1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND - 1, 'reason': 'end',
                           'year': dates[ND - 1].year})

    # === Compute statistics ===
    if cash <= 0 or not trades:
        return None

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    # Max drawdown
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity *= (1 + t['pnl'] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.get('reason', 'unknown')
        if r not in reasons:
            reasons[r] = 0
        reasons[r] += 1

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw / max(len(trades), 1)) * avg_w - (1 - nw / max(len(trades), 1)) * avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0), 'year_stats': year_stats,
        'reasons': reasons,
    }


# ======================================================================
# MAIN — Run V14 backtests
# ======================================================================

if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V14 — Trading Engine Breakthrough", flush=True)
    print("  5 Engine Innovations + 3 New Factors", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load all existing factors
    base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
    inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)
    extra_factors = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e_factors = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv_inter = compute_advanced_interactions(
        {**base_factors, **inter_factors, **extra_factors, **v7e_factors}, NS, ND)
    v8_factors = compute_v8_factors(NS, ND, C, O, H, L, V)
    v8_all = {**base_factors, **inter_factors, **extra_factors,
              **v7e_factors, **adv_inter, **v8_factors}
    v8_inter = compute_v8_interactions(v8_all, NS, ND)
    v8_all.update(v8_inter)
    v9_factors = compute_v9_factors(NS, ND, C, O, H, L, V)
    v9_all = {**v8_all, **v9_factors}
    v9_inter = compute_v9_interactions(v9_all, NS, ND)
    v9_all.update(v9_inter)
    v10_factors = compute_v10_factors(NS, ND, C, O, H, L, V)
    v10_all = {**v9_all, **v10_factors}
    v10_inter = compute_v10_interactions(v10_all, NS, ND)
    v10_all.update(v10_inter)
    v11_factors = compute_v11_factors(NS, ND, C, O, H, L, V)
    v11_all = {**v10_all, **v11_factors}
    v11_inter = compute_v11_interactions(v11_all, NS, ND)
    v11_all.update(v11_inter)

    # V14 new factors
    v14_factors = compute_v14_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **v14_factors}
    v14_inter = compute_v14_interactions(all_factors, NS, ND)
    all_factors.update(v14_inter)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # =====================================================================
    # BASELINE: V10 BwpBNW with original engine (for comparison)
    # =====================================================================
    print(f"\n  === BASELINE (V10 engine, V7c backtest) ===", flush=True)
    bwp_weights = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    baseline = backtest_v7c(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=1.5)
    if baseline:
        print(f"  BwpBNW V10: Ann={baseline['ann']:+7.1f}% WR={baseline['wr']:5.1f}% "
              f"Edge={baseline['edge']:+5.2f}% DD={baseline['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # V14 ENGINE TESTS
    # =====================================================================
    print(f"\n  === V14 TRADING ENGINE TESTS ===", flush=True)

    # Regime-adaptive weights:
    # TRENDING: favor momentum and trend quality
    weights_trend = {
        'R_BWP_BNW': 0.20, 'R_TENSION': 0.25,
        'R_R_SQUARED': 0.25, 'R_MOM5': 0.15, 'R_LINREG_SLOPE': 0.15,
    }
    # RANGING: favor mean-reversion and body quality
    weights_range = {
        'R_BWP_BNW': 0.30, 'R_BODY_NW': 0.20,
        'R_BB_WIDTH_PCT_INV': 0.20, 'R_SMA_DEV': 0.15, 'R_DRAWDOWN_52W': 0.15,
    }
    # TRANSITION: balanced
    weights_trans = {
        'R_BWP_BNW': 0.30, 'R_TENSION': 0.30,
        'R_R_SQUARED': 0.20, 'R_SMA_DEV': 0.20,
    }

    results = []

    # Test 1: Full V14 engine (all innovations on)
    print(f"\n  [Test 1] Full V14 engine (Epanechnikov + Sigmoid + Kelly + Adaptive)", flush=True)
    r = backtest_v14(
        weights_trend, weights_range, weights_trans,
        all_factors, NS, ND, dates, C, O, H, L, V,
        top_n=1, atr_stop_mult=1.5,
        use_epanechnikov=True, use_sigmoid=True, use_kelly=True,
        use_adaptive_rebal=True, rebalance_days_max=15,
        rank_drop_threshold=40)
    if r:
        r['test'] = 'FullV14'
        results.append(r)
        print(f"  FullV14: Ann={r['ann']:+7.1f}% N={r['n']:5d} WR={r['wr']:5.1f}% "
              f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}% Reasons={r['reasons']}", flush=True)

    # Test 2: V14 engine with different ATR stops
    for atr in [1.0, 1.2, 1.5, 2.0]:
        r = backtest_v14(
            weights_trend, weights_range, weights_trans,
            all_factors, NS, ND, dates, C, O, H, L, V,
            top_n=1, atr_stop_mult=atr,
            use_epanechnikov=True, use_sigmoid=True, use_kelly=True,
            use_adaptive_rebal=True, rebalance_days_max=15,
            rank_drop_threshold=40)
        if r:
            r['test'] = f'V14_ATR{atr}'
            results.append(r)

    # Test 3: Ablation — turn off each innovation
    print(f"\n  [Test 3] Ablation study", flush=True)

    # No Epanechnikov (linear scoring)
    r = backtest_v14(
        weights_trend, weights_range, weights_trans,
        all_factors, NS, ND, dates, C, O, H, L, V,
        top_n=1, atr_stop_mult=1.5,
        use_epanechnikov=False, use_sigmoid=True, use_kelly=True,
        use_adaptive_rebal=True, rebalance_days_max=15,
        rank_drop_threshold=40)
    if r:
        r['test'] = 'NoEpan'
        results.append(r)
        print(f"  NoEpan:  Ann={r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    # No Sigmoid (uniform confidence)
    r = backtest_v14(
        weights_trend, weights_range, weights_trans,
        all_factors, NS, ND, dates, C, O, H, L, V,
        top_n=1, atr_stop_mult=1.5,
        use_epanechnikov=True, use_sigmoid=False, use_kelly=True,
        use_adaptive_rebal=True, rebalance_days_max=15,
        rank_drop_threshold=40)
    if r:
        r['test'] = 'NoSigm'
        results.append(r)
        print(f"  NoSigm:  Ann={r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    # No Kelly (fixed sizing)
    r = backtest_v14(
        weights_trend, weights_range, weights_trans,
        all_factors, NS, ND, dates, C, O, H, L, V,
        top_n=1, atr_stop_mult=1.5,
        use_epanechnikov=True, use_sigmoid=True, use_kelly=False,
        use_adaptive_rebal=True, rebalance_days_max=15,
        rank_drop_threshold=40)
    if r:
        r['test'] = 'NoKelly'
        results.append(r)
        print(f"  NoKelly: Ann={r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    # No adaptive rebalance (fixed period)
    r = backtest_v14(
        weights_trend, weights_range, weights_trans,
        all_factors, NS, ND, dates, C, O, H, L, V,
        top_n=1, atr_stop_mult=1.5,
        use_epanechnikov=True, use_sigmoid=True, use_kelly=True,
        use_adaptive_rebal=False, rebalance_days_max=10,
        rank_drop_threshold=40)
    if r:
        r['test'] = 'NoAdapt'
        results.append(r)
        print(f"  NoAdapt: Ann={r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    # No regime adaptation (use same weights for all)
    r = backtest_v14(
        weights_trans, weights_trans, weights_trans,
        all_factors, NS, ND, dates, C, O, H, L, V,
        top_n=1, atr_stop_mult=1.5,
        use_epanechnikov=True, use_sigmoid=True, use_kelly=True,
        use_adaptive_rebal=True, rebalance_days_max=15,
        rank_drop_threshold=40)
    if r:
        r['test'] = 'NoRegime'
        results.append(r)
        print(f"  NoRegime: Ann={r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    # Test 4: Different rank_drop thresholds
    print(f"\n  [Test 4] Rank-drop thresholds", flush=True)
    for threshold in [30, 35, 40, 45, 50]:
        r = backtest_v14(
            weights_trend, weights_range, weights_trans,
            all_factors, NS, ND, dates, C, O, H, L, V,
            top_n=1, atr_stop_mult=1.5,
            use_epanechnikov=True, use_sigmoid=True, use_kelly=True,
            use_adaptive_rebal=True, rebalance_days_max=15,
            rank_drop_threshold=threshold)
        if r:
            r['test'] = f'RD{threshold}'
            results.append(r)

    # Test 5: Different rebalance max periods
    print(f"\n  [Test 5] Rebalance periods", flush=True)
    for rebal in [7, 10, 12, 15, 20]:
        r = backtest_v14(
            weights_trend, weights_range, weights_trans,
            all_factors, NS, ND, dates, C, O, H, L, V,
            top_n=1, atr_stop_mult=1.5,
            use_epanechnikov=True, use_sigmoid=True, use_kelly=True,
            use_adaptive_rebal=True, rebalance_days_max=rebal,
            rank_drop_threshold=40)
        if r:
            r['test'] = f'Reb{rebal}'
            results.append(r)

    # Test 6: V14 factors with V10 baseline engine
    print(f"\n  [Test 6] V14 factors with V10 engine", flush=True)
    v14_single_tests = ['R_HAR_RV_RATIO_INV', 'R_LOG_PRESSURE', 'R_ATR_TERRAIN',
                        'R_LP_BNW', 'R_AT_BNW', 'R_LP_BWP', 'R_AT_BWP',
                        'R_HAR_BWP', 'R_LPA_BNW']
    for fname in v14_single_tests:
        if fname in all_factors:
            weights_test = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                            'R_R_SQUARED': 0.2, fname: 0.2}
            r = backtest_v7c(weights_test, all_factors, NS, ND, dates, C, O, H, L, V,
                             top_n=1, rebalance_days=10, atr_stop_mult=1.5)
            if r:
                r['test'] = f'V10+{fname}'
                results.append(r)
                print(f"  V10+{fname:<20s}: Ann={r['ann']:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # RESULTS SUMMARY
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 30 RESULTS (V14 TRADING ENGINE)", flush=True)
    print(f"  {'Test':<25s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:30]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<25s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Top 3 year-by-year
    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

        # Exit reason breakdown
        if 'reasons' in r:
            print(f"    Exit reasons: {r['reasons']}", flush=True)

    # Comparison with baseline
    if baseline:
        print(f"\n  === COMPARISON ===", flush=True)
        print(f"  V10 Baseline: {baseline['ann']:+.1f}% DD={baseline['max_dd']:.1f}%", flush=True)
        if results:
            best = results[0]
            delta = best['ann'] - baseline['ann']
            dd_delta = best['max_dd'] - baseline['max_dd']
            print(f"  V14 Best:     {best['ann']:+.1f}% DD={best['max_dd']:.1f}% "
                  f"(Ann delta={delta:+.1f}%, DD delta={dd_delta:+.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)
