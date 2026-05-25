"""
Alpha V67 — A-Share Specific: Pullback-Reversal Strategy
=========================================================
Deep insight: A-share short-term (1wk-1mo) shows REVERSAL, not momentum.
T+1 + 涨跌停板 amplifies overreaction → buying highest momentum = pullback
→ tight stop triggers → 58% stopped out in 1-3 days.

V67 approach: 50% mid-term momentum + 50% short-term reversal → Sharpe 0.8

Factors:
  1. UPTREND_PULLBACK — stocks in uptrend that pulled back (buy the dip)
  2. REVERSAL_SIGNAL — short-term mean reversion weighted by volume
  3. SORTINO_RANK — Sortino-ratio based ranking (penalizes downside only)
  4. VOL_REGIME — volatility regime for dynamic parameter adjustment
  5. QUALITY_MOMENTUM — momentum penalized by max drawdown

All factors: no look-ahead (data up to di-1), continuous, rankable.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0


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


def _rolling_mean_axis1(arr, window, min_valid=None):
    """Rolling mean along axis=1 (time). Handles NaN. Uses data up to di-1."""
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


def compute_v67_factors(NS, ND, C, O, H, L, V):
    """Compute V67 A-share specific factors. No look-ahead: all use data up to di-1.

    Returns dict of {name: ndarray(NS, ND)} with raw factor values.
    """
    print("[V67] Computing A-share pullback-reversal factors...", flush=True)
    t0 = time.time()
    factors = {}

    # =========================================================================
    # Precompute daily returns (for reuse across factors)
    # =========================================================================
    daily_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        daily_ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]
    print(f"  Daily returns precomputed ({time.time()-t0:.1f}s)", flush=True)

    # =========================================================================
    # 1. UPTREND_PULLBACK
    #    Identifies stocks in medium-term uptrends that pulled back short-term.
    #    Uptrend: C[di-1] > MA20 AND MA20 > MA60
    #    Pullback: 3-day return is negative
    #    Score: (MA20 - MA60) / MA60 * 100 * sign(pullback_depth)
    # =========================================================================
    UPTREND_PB = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(61, ND):  # need at least 60 days of history
            # Compute MA20 and MA60 using data up to di-1
            window20 = C[si, di - 21:di]  # indices di-21 to di-1 (20 values)
            window60 = C[si, di - 61:di]  # indices di-61 to di-1 (60 values)

            valid20 = window20[~np.isnan(window20)]
            valid60 = window60[~np.isnan(window60)]

            if len(valid20) < 15 or len(valid60) < 40:
                continue

            ma20 = np.mean(valid20)
            ma60 = np.mean(valid60)
            c_now = C[si, di - 1]

            if np.isnan(c_now) or ma60 <= 0:
                continue

            # Uptrend conditions
            in_uptrend = (c_now > ma20) and (ma20 > ma60)

            # 3-day return (pullback measure)
            c_3ago = C[si, di - 4]
            if np.isnan(c_3ago) or c_3ago <= 0:
                continue
            ret_3d = (c_now - c_3ago) / c_3ago

            # Trend strength: how far MA20 is above MA60
            trend_strength = (ma20 - ma60) / ma60 * 100

            if in_uptrend and ret_3d < 0:
                # Deeper pullback in stronger uptrend = better buy opportunity
                # score = trend_strength * abs(pullback_depth)
                # More negative ret_3d = deeper pullback = higher score
                UPTREND_PB[si, di] = trend_strength * abs(ret_3d)
            elif in_uptrend:
                # In uptrend but not pulled back — small positive score
                UPTREND_PB[si, di] = trend_strength * 0.1
            else:
                # Not in uptrend — zero score
                UPTREND_PB[si, di] = 0.0

    factors['UPTREND_PULLBACK'] = UPTREND_PB
    print(f"  UPTREND_PULLBACK done ({time.time()-t0:.1f}s)", flush=True)

    # =========================================================================
    # 2. REVERSAL_SIGNAL
    #    Short-term mean reversion for A-shares.
    #    score = -3day_return * vol_ratio
    #    Stocks that dropped on high volume = strong reversal candidates.
    # =========================================================================
    REVERSAL = np.full((NS, ND), np.nan)

    # Precompute rolling mean volume (20-day)
    vol_rolling_mean = _rolling_mean_axis1(V, window=20, min_valid=10)

    for si in range(NS):
        for di in range(22, ND):  # need 21 days for volume lookback + 3d return
            c_now = C[si, di - 1]
            c_3ago = C[si, di - 4]
            v_now = V[si, di - 1]
            v_mean = vol_rolling_mean[si, di]

            if np.isnan(c_now) or np.isnan(c_3ago) or c_3ago <= 0:
                continue
            if np.isnan(v_now) or np.isnan(v_mean) or v_mean <= 0:
                continue

            ret_3d = (c_now - c_3ago) / c_3ago
            vol_ratio = v_now / v_mean

            # Negative return * high volume = strong reversal signal
            # We negate so that more negative returns get higher scores
            REVERSAL[si, di] = -ret_3d * vol_ratio

    factors['REVERSAL_SIGNAL'] = REVERSAL
    print(f"  REVERSAL_SIGNAL done ({time.time()-t0:.1f}s)", flush=True)

    # =========================================================================
    # 3. SORTINO_RANK
    #    Sortino-ratio based ranking (only penalizes downside).
    #    Sortino = mean(return) / std(negative_returns)
    #    Rolling 60-day window.
    # =========================================================================
    SORTINO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(61, ND):  # need 60 days of returns
            # Get 60 daily returns up to di-1
            window_rets = daily_ret[si, di - 60:di]
            valid_rets = window_rets[~np.isnan(window_rets)]

            if len(valid_rets) < 30:
                continue

            mean_ret = np.mean(valid_rets)
            neg_rets = valid_rets[valid_rets < 0]

            if len(neg_rets) < 5:
                # Not enough downside observations — use a small default
                downside_std = np.std(valid_rets) if np.std(valid_rets) > 0 else 1e-8
            else:
                downside_std = np.std(neg_rets)

            if downside_std > 0:
                SORTINO[si, di] = mean_ret / downside_std
            else:
                SORTINO[si, di] = 0.0

    factors['SORTINO_RANK'] = SORTINO
    print(f"  SORTINO_RANK done ({time.time()-t0:.1f}s)", flush=True)

    # =========================================================================
    # 4. VOL_REGIME
    #    Volatility regime indicator — cross-sectional percentile rank of
    #    realized volatility. Low vol = good for momentum, high vol = good
    #    for reversal.
    #    Rolling 20-day realized volatility: std(daily_returns, 20)
    # =========================================================================
    VOL_REGIME = np.full((NS, ND), np.nan)

    # Rolling 20-day std of daily returns
    realized_vol = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            window_rets = daily_ret[si, di - 20:di]
            valid_rets = window_rets[~np.isnan(window_rets)]
            if len(valid_rets) >= 10:
                realized_vol[si, di] = np.std(valid_rets)

    # Cross-sectional percentile rank (inline, not using _rank_normalize
    # because we want to keep it as the raw VOL_REGIME factor)
    for di in range(21, ND):
        vals = realized_vol[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        VOL_REGIME[valid, di] = ranks / n * 100

    factors['VOL_REGIME'] = VOL_REGIME
    print(f"  VOL_REGIME done ({time.time()-t0:.1f}s)", flush=True)

    # =========================================================================
    # 5. QUALITY_MOMENTUM
    #    Combines momentum with quality filter (max drawdown).
    #    20-day momentum penalized by max drawdown over 60 days.
    #    score = momentum * (1 - max_dd/30)
    #    Only positive for stocks with positive momentum AND limited drawdown.
    # =========================================================================
    QUALITY_MOM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(61, ND):  # need 60 days for drawdown + 21 for momentum
            # 20-day momentum
            c_now = C[si, di - 1]
            c_20ago = C[si, di - 21]
            if np.isnan(c_now) or np.isnan(c_20ago) or c_20ago <= 0:
                continue
            mom_20 = (c_now - c_20ago) / c_20ago

            # Max drawdown over 60 days
            window_c = C[si, di - 60:di]
            valid_c = window_c[~np.isnan(window_c)]
            if len(valid_c) < 30:
                continue

            # Compute max drawdown
            peak = valid_c[0]
            max_dd = 0.0
            for c_val in valid_c:
                if c_val > peak:
                    peak = c_val
                if peak > 0:
                    dd = (peak - c_val) / peak
                    if dd > max_dd:
                        max_dd = dd

            # Quality filter: penalize stocks with deep drawdowns
            quality_mult = max(0.0, 1.0 - max_dd / 0.30)  # max_dd/30%

            # Only positive for stocks with positive momentum and limited drawdown
            QUALITY_MOM[si, di] = mom_20 * quality_mult

    factors['QUALITY_MOMENTUM'] = QUALITY_MOM
    print(f"  QUALITY_MOMENTUM done ({time.time()-t0:.1f}s)", flush=True)

    print(f"[V67] All 5 factors computed ({time.time()-t0:.1f}s)", flush=True)
    return factors


if __name__ == '__main__':
    # Quick standalone test
    from alpha_v2 import load_all_data

    print("=" * 70, flush=True)
    print("  Alpha V67 — A-Share Pullback-Reversal Factors (standalone)")
    print("=" * 70)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    v67 = compute_v67_factors(NS, ND, C, O, H, L, V)

    print("\nFactor summary:", flush=True)
    for name, arr in v67.items():
        valid = ~np.isnan(arr)
        pct = valid.sum() / arr.size * 100
        vals = arr[valid]
        if len(vals) > 0:
            print(f"  {name:<25s} fill={pct:.1f}% mean={np.mean(vals):.4f} "
                  f"std={np.std(vals):.4f} min={np.min(vals):.4f} max={np.max(vals):.4f}",
                  flush=True)
        else:
            print(f"  {name:<25s} fill={pct:.1f}% (no valid values)", flush=True)

    print(f"\n{'=' * 70}", flush=True)
