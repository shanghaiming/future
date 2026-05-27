"""
V98: Trend Quality Engine Strategy
====================================
Core insight: Distinguish "good momentum" (stable trends) from "bad momentum"
(noisy bounces) using ADX regime cross-filtering.

Innovation: ADX regime-based factor weighting
- ADX > trend_th (trending): Boost LINEARREG_SLOPE and SAR_deviation weights
- ADX < range_th (range-bound): Boost ret5d and TRIX weights (reversal)
- range_th <= ADX <= trend_th: Standard weights

Factor mix (7 factors, rank-based):
  ret5d_rank, TRIX(30)_rank, LINEARREG_SLOPE(20)_rank,
  ADX(14)_rank, oi5d_rank, SAR_deviation_rank, atrp5d_rank

Built on V96's NW kernel regression + volatility-adaptive framework.

Walk-forward 2019-2026. No leverage. CASH0=1,000,000, COMM=0.0005.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005
LEVERAGE = 1.0

# Sector definitions
SECTOR_MAP = {
    'i': 'BLACK', 'j': 'BLACK', 'jm': 'BLACK', 'hc': 'BLACK',
    'sf': 'BLACK', 'sm': 'BLACK', 'wr': 'BLACK', 'im': 'BLACK',
    'cu': 'METAL', 'al': 'METAL', 'zn': 'METAL', 'pb': 'METAL',
    'ni': 'METAL', 'sn': 'METAL', 'ss': 'METAL', 'ao': 'METAL',
    'au': 'METAL', 'ag': 'METAL', 'rb': 'METAL', 'si': 'METAL',
    'sc': 'ENERGY', 'fu': 'ENERGY', 'bu': 'ENERGY',
    'pg': 'ENERGY', 'eb': 'ENERGY', 'ta': 'ENERGY',
    'fg': 'ENERGY', 'oi': 'ENERGY',
    'v': 'CHEMICAL', 'pp': 'CHEMICAL', 'l': 'CHEMICAL',
    'eg': 'CHEMICAL', 'ma': 'CHEMICAL', 'sa': 'CHEMICAL',
    'ur': 'CHEMICAL', 'pf': 'CHEMICAL', 'sh': 'CHEMICAL',
    'lc': 'CHEMICAL',
    'm': 'AGRI', 'y': 'AGRI', 'a': 'AGRI', 'p': 'AGRI',
    'c': 'AGRI', 'cs': 'AGRI', 'jd': 'AGRI', 'rr': 'AGRI',
    'lrm': 'AGRI', 'rm': 'AGRI', 'ru': 'AGRI',
    'cf': 'SOFTS', 'sr': 'SOFTS', 'ap': 'SOFTS',
    'cj': 'SOFTS', 'pk': 'SOFTS', 'lh': 'SOFTS',
    'sp': 'SOFTS', 'b': 'SOFTS', 'br': 'SOFTS',
}

# V98 factors: 7 factors for trend quality
V98_FACTOR_NAMES = [
    "ret5d", "trix30", "linearreg_slope20",
    "adx14", "oi5d", "sar_deviation", "atrp5d",
]
N_FACTORS = len(V98_FACTOR_NAMES)


def _extract_base_symbol(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    if s.endswith('fi'):
        s = s[:-2]
    return s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    sector_lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        sector_lookup[si] = SECTOR_MAP.get(base, 'OTHER')
    return sector_lookup


# =====================================================================
# V98 Factor Computation (TA-Lib based)
# =====================================================================

def compute_v98_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute 7 trend-quality factors using TA-Lib."""
    t0 = time.time()
    print("[V98] Computing trend quality factors...", flush=True)

    # Factor 1: ret5d (5-day return)
    ret5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    # Factor 2: TRIX(30) - triple exponential derivative
    trix30 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), np.nan, C[si]).astype(np.float64)
            # TA-Lib needs NaN replaced for computation
            c_clean = np.where(np.isnan(c), 0, c)
            nan_mask = np.isnan(c)
            try:
                t = talib.TRIX(c_clean, timeperiod=30)
                trix30[si] = np.where(nan_mask, np.nan, t)
            except Exception:
                pass

    # Factor 3: LINEARREG_SLOPE(20) - linear regression slope
    linearreg_slope20 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), np.nan, C[si]).astype(np.float64)
            c_clean = np.where(np.isnan(c), 0, c)
            nan_mask = np.isnan(c)
            try:
                slope = talib.LINEARREG_SLOPE(c_clean, timeperiod=20)
                linearreg_slope20[si] = np.where(nan_mask, np.nan, slope)
            except Exception:
                pass

    # Factor 4: ADX(14) - Average Directional Index
    adx14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l_arr = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            h_nan = np.isnan(H[si])
            l_nan = np.isnan(L[si])
            c_nan = np.isnan(C[si])
            nan_mask = h_nan | l_nan | c_nan
            try:
                adx = talib.ADX(h, l_arr, c, timeperiod=14)
                adx14[si] = np.where(nan_mask, np.nan, adx)
            except Exception:
                pass

    # Factor 5: oi5d (5-day open interest change)
    oi5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
                oi5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # Factor 6: SAR_deviation - deviation of price from Parabolic SAR
    sar_deviation = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l_arr = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            h_nan = np.isnan(H[si])
            l_nan = np.isnan(L[si])
            c_nan = np.isnan(C[si])
            nan_mask = h_nan | l_nan | c_nan
            try:
                sar = talib.SAR(h, l_arr, acceleration=0.02, maximum=0.2)
                # Deviation: (close - SAR) / close
                with np.errstate(divide='ignore', invalid='ignore'):
                    dev = np.where(
                        (c > 0) & ~nan_mask,
                        (c - sar) / c,
                        np.nan)
                sar_deviation[si] = np.where(nan_mask, np.nan, dev)
            except Exception:
                pass

    # Factor 7: atrp5d (5-day ATR percentage)
    atrp5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            atr_vals = []
            for j in range(di - 5, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp5d[si, di] = np.mean(atr_vals) / C[si, di]

    # Target: next 5-day forward return
    fwd_ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 5):
            if (not np.isnan(C[si, di + 5])
                    and not np.isnan(C[si, di])
                    and C[si, di] > 0):
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0

    # ATR mean for adaptive bandwidth (NW kernel)
    atr_mean = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            atr_vals = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atr_mean[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret5d": ret5d, "trix30": trix30,
        "linearreg_slope20": linearreg_slope20,
        "adx14": adx14, "oi5d": oi5d,
        "sar_deviation": sar_deviation, "atrp5d": atrp5d,
        "fwd_ret_5d": fwd_ret_5d, "atr_mean": atr_mean,
    }


def cross_sectional_rank(
    factor: np.ndarray, NS: int, ND: int,
    min_count: int = 10,
) -> np.ndarray:
    """Cross-sectional percentile rank (0 to 1) for a factor."""
    ranked = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = factor[:, di]
        valid_mask = ~np.isnan(vals)
        valid = vals[valid_mask]
        if len(valid) < min_count:
            continue
        order = np.argsort(np.argsort(valid))
        ranks = (order + 1).astype(float) / len(valid)
        idx = 0
        for si in range(NS):
            if valid_mask[si]:
                ranked[si, di] = ranks[idx]
                idx += 1
    return ranked


def normalize_factor(
    factor: np.ndarray, NS: int, ND: int,
    min_count: int = 10,
) -> np.ndarray:
    """Cross-sectional z-score normalization for a factor."""
    normed = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = factor[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) < min_count:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid)
        if sigma < 1e-12:
            continue
        for si in range(NS):
            if not np.isnan(vals[si]):
                normed[si, di] = (vals[si] - mu) / sigma
    return normed


# =====================================================================
# INNOVATION: ADX Regime-Based Factor Weighting
# =====================================================================

def get_regime_weights(
    adx_val: float,
    adx_trend_th: float = 25.0,
    adx_range_th: float = 20.0,
) -> List[float]:
    """Get factor weights based on ADX regime.

    Returns 7 weights in order:
      [ret5d, trix30, linearreg_slope20, adx14, oi5d, sar_deviation, atrp5d]

    Regime logic:
      - Trending (ADX > trend_th): boost slope + SAR deviation
      - Range-bound (ADX < range_th): boost ret5d + TRIX (reversal)
      - Neutral: standard weights
    """
    # Base weights
    w_base = [0.20, 0.20, 0.20, 0.15, 0.10, 0.10, 0.05]
    # Trend weights (boost slope + SAR)
    w_trend = [0.10, 0.15, 0.30, 0.15, 0.10, 0.15, 0.05]
    # Range weights (boost ret5d + TRIX)
    w_range = [0.30, 0.25, 0.10, 0.15, 0.10, 0.05, 0.05]

    if np.isnan(adx_val):
        return w_base
    if adx_val > adx_trend_th:
        return w_trend
    if adx_val < adx_range_th:
        return w_range
    return w_base


def compute_regime_weighted_score(
    ranked_factors: Dict[str, np.ndarray],
    adx14_raw: np.ndarray,
    NS: int, ND: int,
    adx_trend_th: float = 25.0,
    adx_range_th: float = 20.0,
) -> np.ndarray:
    """Compute ADX-regime-weighted composite score.

    For each (si, di), determine ADX regime and apply corresponding weights
    to the ranked factors, producing a single composite alpha score.
    """
    t0 = time.time()
    print(
        f"[V98] Computing regime-weighted scores "
        f"(trend_th={adx_trend_th}, range_th={adx_range_th})...",
        flush=True)

    score = np.full((NS, ND), np.nan)

    for di in range(ND):
        for si in range(NS):
            # Get ADX value for regime determination
            adx_val = adx14_raw[si, di]

            # Get regime-specific weights
            weights = get_regime_weights(
                adx_val, adx_trend_th, adx_range_th)

            # Compute weighted sum of ranked factors
            val = 0.0
            valid_count = 0
            for fi, fname in enumerate(V98_FACTOR_NAMES):
                r = ranked_factors[fname][si, di]
                if not np.isnan(r):
                    val += weights[fi] * r
                    valid_count += 1

            if valid_count >= 5:  # Need at least 5 of 7 factors
                score[si, di] = val

    print(f"  Regime-weighted scores done: {time.time() - t0:.1f}s",
          flush=True)
    return score


# =====================================================================
# NW Kernel Regression (from V86/V96 framework, adapted for V98)
# =====================================================================

def compute_nw_predicted_returns(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    training_window: int = 40,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    """Compute NW kernel regression predicted returns using V98 factors."""
    t0 = time.time()
    print(
        f"[V98] Computing NW predicted returns "
        f"(window={training_window}, bw={kernel_bandwidth:.1f})...",
        flush=True)

    # Normalize factors cross-sectionally (z-score)
    normed = {}
    for fname in V98_FACTOR_NAMES:
        normed[fname] = normalize_factor(raw_factors[fname], NS, ND)

    fwd_ret = raw_factors["fwd_ret_5d"]
    atr_mean = raw_factors["atr_mean"]
    predicted = np.full((NS, ND), np.nan)

    MIN_TRAIN = 20

    for di in range(training_window + 10, ND):
        train_features: List[np.ndarray] = []
        train_targets: List[float] = []

        start_di = max(10, di - training_window)
        for tdi in range(start_di, di):
            for si in range(NS):
                feat = np.array([
                    normed[fname][si, tdi] for fname in V98_FACTOR_NAMES
                ])
                target = fwd_ret[si, tdi]
                if np.any(np.isnan(feat)) or np.isnan(target):
                    continue
                train_features.append(feat)
                train_targets.append(target)

        if len(train_features) < MIN_TRAIN:
            continue

        train_X = np.array(train_features)
        train_Y = np.array(train_targets)

        feat_std = np.std(train_X, axis=0)
        feat_std[feat_std < 1e-12] = 1.0

        for si in range(NS):
            query_feat = np.array([
                normed[fname][si, di] for fname in V98_FACTOR_NAMES
            ])
            if np.any(np.isnan(query_feat)):
                continue

            atr_val = atr_mean[si, di]
            if np.isnan(atr_val):
                h = kernel_bandwidth
            else:
                h = max(atr_val * kernel_bandwidth, 0.1)

            diff = train_X - query_feat[np.newaxis, :]
            dist = np.sqrt(
                np.sum((diff / feat_std[np.newaxis, :]) ** 2, axis=1))

            scaled_dist = dist / h
            weights = np.zeros(len(train_X))
            mask = scaled_dist <= 1.0
            if not np.any(mask):
                min_dist_idx = np.argmin(dist)
                if dist[min_dist_idx] < 1e12:
                    weights[min_dist_idx] = 1.0
                else:
                    continue
            else:
                weights[mask] = 0.75 * (1.0 - scaled_dist[mask] ** 2)

            weight_sum = np.sum(weights)
            if weight_sum < 1e-12:
                continue

            predicted[si, di] = np.sum(weights * train_Y) / weight_sum

        if di % 100 == 0:
            valid_count = np.sum(~np.isnan(predicted[:, di]))
            print(
                f"  di={di}/{ND} valid={valid_count}/{NS} "
                f"train_size={len(train_features)}",
                flush=True)

    print(f"  NW prediction done: {time.time() - t0:.1f}s", flush=True)
    return predicted


# =====================================================================
# Helper functions (from V96)
# =====================================================================

def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Kaufman efficiency ratio for regime detection."""
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker_10[si, di] = net_change / total_change

    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1
    return ker_regime


def get_dynamic_mode(
    recent_trades_win: List[int],
    win_threshold: float,
    win_rate_window: int,
) -> str:
    if len(recent_trades_win) < 5:
        return "normal"
    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)
    if win_rate > win_threshold:
        return "winning"
    elif win_rate < 0.50:
        return "losing"
    return "normal"


def get_mode_threshold(
    mode: str,
    win_threshold: float,
    normal_threshold: float,
    lose_threshold: float,
) -> float:
    if mode == "winning":
        return win_threshold
    elif mode == "losing":
        return lose_threshold
    return normal_threshold


def compute_atr_at(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                   si: int, di: int, start_di: int) -> Optional[float]:
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


# =====================================================================
# Volatility-Adaptive Position Sizing (from V96)
# =====================================================================

def compute_portfolio_volatility(
    C: np.ndarray, NS: int, ND: int,
    vol_lookback: int = 20,
) -> np.ndarray:
    """Compute rolling portfolio volatility proxy."""
    port_vol = np.full(ND, np.nan)
    for di in range(vol_lookback + 1, ND):
        daily_rets = []
        for dd in range(di - vol_lookback, di):
            rets = []
            for si in range(NS):
                if (not np.isnan(C[si, dd])
                        and not np.isnan(C[si, dd - 1])
                        and C[si, dd - 1] > 0):
                    rets.append(C[si, dd] / C[si, dd - 1] - 1.0)
            if rets:
                daily_rets.append(np.mean(rets))
        if len(daily_rets) >= vol_lookback // 2:
            port_vol[di] = np.std(daily_rets)
    return port_vol


def get_vol_multiplier(
    port_vol: float,
    vol_median: float,
    vol_high_mult: float,
    vol_low_mult: float,
    size_reduce: float,
    size_boost: float,
) -> float:
    """Get position size multiplier based on volatility regime."""
    if np.isnan(port_vol) or np.isnan(vol_median) or vol_median < 1e-12:
        return 1.0
    ratio = port_vol / vol_median
    if ratio > vol_high_mult:
        return size_reduce
    elif ratio < vol_low_mult:
        return size_boost
    return 1.0


# =====================================================================
# V98 Backtest: Combined signal (NW + regime-weighted) + vol sizing
# =====================================================================

def backtest_v98(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
    regime_scores: np.ndarray,
    ker_regime: np.ndarray,
    port_vol: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 5,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    vol_high_mult: float = 2.0,
    vol_low_mult: float = 0.5,
    size_reduce: float = 0.5,
    size_boost: float = 1.3,
    vol_lookback: int = 20,
    nw_weight: float = 0.5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V98: Trend Quality Engine.

    Combines NW kernel predicted returns with ADX-regime-weighted scores.
    Final alpha = nw_weight * NW_predicted + (1-nw_weight) * regime_score
    """
    if end_di is None:
        end_di = ND - 1

    vol_data = port_vol[max(start_di, vol_lookback + 1):end_di]
    vol_data_valid = vol_data[~np.isnan(vol_data)]
    vol_median = (
        np.median(vol_data_valid) if len(vol_data_valid) > 10 else 1e-6)

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float]] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float]] = []

        # Dynamic mode
        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)

        # Vol-adaptive sizing
        vol_mult = get_vol_multiplier(
            port_vol[di], vol_median,
            vol_high_mult, vol_low_mult,
            size_reduce, size_boost)

        # Exit logic
        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc in positions:
            pos_by_si[si].append((edi, ep, sp, alloc))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any(c < sp for _, _, sp, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di,
                        "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop", "mode": mode[:1].upper(),
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di,
                        "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold", "mode": mode[:1].upper(),
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc))

        positions = new_positions
        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        # --- ENTRY: combined signal ---
        held = {p[0] for p in positions}
        if len(held) >= top_n:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue

            # Combined alpha: NW + regime-weighted score
            nw_val = predicted[si, di]
            rs_val = regime_scores[si, di]

            if np.isnan(nw_val) and np.isnan(rs_val):
                continue

            # Use whichever is available, or blend
            if np.isnan(nw_val):
                alpha = rs_val
            elif np.isnan(rs_val):
                alpha = nw_val
            else:
                alpha = nw_weight * nw_val + (1.0 - nw_weight) * rs_val

            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            if ker_regime[si, di] < 0:
                continue

            candidates.append((alpha, si))

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[0])

        n_to_take = top_n
        if mode == "winning":
            n_to_take = min(top_n + 1, top_n * 2)
        elif mode == "losing":
            n_to_take = max(1, top_n - 1)

        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, 'OTHER')] += 1

        new_entries = []
        for alpha_val, si in candidates:
            if len(held) + len(new_entries) >= n_to_take:
                break
            if si in held:
                continue
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            if alpha_val <= 0:
                continue
            new_entries.append((alpha_val, si, sym_sector))
            sector_counts[sym_sector] += 1

        if not new_entries:
            continue

        num_total = len(positions) + len(new_entries)
        alloc_per_pos = LEVERAGE / num_total * vol_mult

        updated_positions = []
        for si, edi, ep, sp, old_alloc in positions:
            updated_positions.append((si, edi, ep, sp, alloc_per_pos))

        for alpha_val, si, sym_sector in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc_per_pos))

        positions = updated_positions

    # Close remaining positions
    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# =====================================================================
# Analysis and Walk-Forward
# =====================================================================

def analyze(
    trades: List[dict], equity: float, max_dd: float,
    label: str = "",
) -> Optional[dict]:
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)

    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    mode_counts = {"W": 0, "N": 0, "L": 0}
    for t in trades:
        m = t.get("mode", "N")
        if m in mode_counts:
            mode_counts[m] += 1

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    avg_win = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    avg_loss = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}")
    print(
        f"    avg_pnl={avg_pnl:+.3f}% avg_win={avg_win:+.3f}% "
        f"avg_loss={avg_loss:+.3f}% "
        f"modes=[W:{mode_counts['W']} N:{mode_counts['N']} "
        f"L:{mode_counts['L']}]")
    print(f"    sectors: {sector_str}")

    yr: Dict[int, dict] = {}
    for t in trades:
        y = t["year"]
        if y not in yr:
            yr[y] = {"n": 0, "w": 0, "pnl": []}
        yr[y]["n"] += 1
        if t["pnl_pct"] > 0:
            yr[y]["w"] += 1
        yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["pnl"]]) - 1
        print(
            f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% "
            f"cum={cum:+.1%}")

    return {
        "n": len(trades), "wr": wr, "dd": max_dd,
        "ann": ann, "sh": sh, "eq": equity,
    }


def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
    regime_scores: np.ndarray,
    ker_regime: np.ndarray,
    port_vol: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 5,
    vol_high_mult: float = 2.0,
    vol_low_mult: float = 0.5,
    size_reduce: float = 0.5,
    size_boost: float = 1.3,
    vol_lookback: int = 20,
    nw_weight: float = 0.5,
    label: str = "",
) -> List[dict]:
    cfg_str = (
        f"tn={top_n} mps={max_per_sector} hd={hold_days} "
        f"vhm={vol_high_mult:.1f} vlm={vol_low_mult:.1f} "
        f"sr={size_reduce:.1f} sb={size_boost:.1f} vlb={vol_lookback} "
        f"nw_w={nw_weight:.1f}")
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V98 {label}")
    print(f"  {cfg_str}")
    print(f"  NO LEVERAGE (leverage=1.0)")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades: List[dict] = []

    for test_year in range(2019, years[-1] + 1):
        test_start = None
        test_end_idx = None
        for i, d in enumerate(dates):
            if d.year == test_year and test_start is None:
                test_start = i
            if d.year == test_year:
                test_end_idx = i
        if test_start is None:
            continue

        trades, _, _ = backtest_v98(
            C, O, H, L, NS, ND, dates, syms,
            predicted, regime_scores, ker_regime, port_vol,
            sector_lookup=sector_lookup,
            top_n=top_n,
            max_per_sector=max_per_sector,
            hold_days=hold_days,
            vol_high_mult=vol_high_mult,
            vol_low_mult=vol_low_mult,
            size_reduce=size_reduce,
            size_boost=size_boost,
            vol_lookback=vol_lookback,
            nw_weight=nw_weight,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades
                       if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", 'OTHER')] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"sectors=[{sec_str}]",
                flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod(
            [1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", 'OTHER')] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


# =====================================================================
# Main: Parameter sweep + Walk-forward
# =====================================================================

def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V98: TREND QUALITY ENGINE STRATEGY")
    print("  Innovation: ADX regime-based factor weighting")
    print("  Trending -> boost LINEARREG_SLOPE + SAR_deviation")
    print("  Range-bound -> boost ret5d + TRIX (reversal)")
    print("  NW kernel + volatility-adaptive sizing (from V96)")
    print("  Walk-forward 2019-2026. No leverage.")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}")

    sector_lookup = build_sector_lookup(syms)
    sector_dist: Dict[str, int] = defaultdict(int)
    for sec in sector_lookup.values():
        sector_dist[sec] += 1
    print(f"  Sector distribution: {dict(sector_dist)}")

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Compute V98 factors ===
    raw_factors = compute_v98_factors(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # === 2. Compute ranked factors ===
    print("[V98] Computing cross-sectional ranks...", flush=True)
    ranked_factors = {}
    for fname in V98_FACTOR_NAMES:
        ranked_factors[fname] = cross_sectional_rank(
            raw_factors[fname], NS, ND)

    # === 3. Compute regime-weighted scores for each ADX threshold pair ===
    adx_threshold_configs = [
        (25, 20),  # default
        (30, 20),  # wider neutral zone
        (20, 15),  # tighter neutral zone
    ]

    regime_score_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for trend_th, range_th in adx_threshold_configs:
        key = (trend_th, range_th)
        regime_score_cache[key] = compute_regime_weighted_score(
            ranked_factors, raw_factors["adx14"],
            NS, ND, trend_th, range_th)

    # === 4. Compute NW predictions ===
    pred_nw = compute_nw_predicted_returns(
        raw_factors, NS, ND,
        training_window=40,
        kernel_bandwidth=1.0,
    )

    # === 5. Compute portfolio volatility for each lookback ===
    vol_cache: Dict[int, np.ndarray] = {}
    for vlb in [15, 20]:
        vol_cache[vlb] = compute_portfolio_volatility(C, NS, ND, vlb)

    # === 6. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  NO LEVERAGE. Trend Quality Engine + NW + vol-adaptive.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for (trend_th, range_th) in adx_threshold_configs:
        regime_scores = regime_score_cache[(trend_th, range_th)]
        for top_n in [2, 3]:
            for mps in [2, 3]:
                for vlb in [15, 20]:
                    for sr in [0.3, 0.5]:
                        for sb in [1.2, 1.5]:
                            for nw_w in [0.3, 0.5, 0.7]:
                                sweep_count += 1
                                trades, eq, dd = backtest_v98(
                                    C, O, H, L, NS, ND,
                                    dates, syms,
                                    pred_nw, regime_scores,
                                    ker_regime,
                                    vol_cache[vlb],
                                    sector_lookup=sector_lookup,
                                    top_n=top_n,
                                    max_per_sector=mps,
                                    hold_days=5,
                                    vol_high_mult=2.0,
                                    vol_low_mult=0.5,
                                    size_reduce=sr,
                                    size_boost=sb,
                                    vol_lookback=vlb,
                                    nw_weight=nw_w,
                                    start_di=bt_2019,
                                )

                                if len(trades) < 10:
                                    continue

                                nw_count = sum(
                                    1 for t in trades
                                    if t["pnl_pct"] > 0)
                                wr = nw_count / len(trades) * 100
                                n_days = max(
                                    1,
                                    trades[-1]["di"] - trades[0]["di"])
                                ann = ((eq / CASH0) ** (
                                    1 / max(
                                        1.0, n_days / 252)) - 1) * 100
                                ap = [t["pnl_abs"]
                                      for t in sorted(
                                          trades,
                                          key=lambda x: x["di"])]
                                rets_arr = np.array(ap) / CASH0
                                sh_val = (
                                    np.mean(rets_arr)
                                    / np.std(rets_arr) * np.sqrt(252)
                                    if np.std(rets_arr) > 0 else 0)

                                results.append({
                                    "trend_th": trend_th,
                                    "range_th": range_th,
                                    "top_n": top_n, "mps": mps,
                                    "vlb": vlb,
                                    "sr": sr, "sb": sb,
                                    "nw_w": nw_w,
                                    "n": len(trades), "wr": wr,
                                    "ann": ann, "dd": dd,
                                    "sharpe": sh_val, "eq": eq,
                                })

    results.sort(key=lambda x: -x["ann"])
    print(
        f"\n  Evaluated {sweep_count} configs, "
        f"{len(results)} with 10+ trades")

    # Report top 10
    print(
        f"\n{'ADXt':>4} {'ADXr':>4} {'TN':>3} {'MPS':>3} "
        f"{'Vlb':>4} {'SR':>4} {'SB':>4} {'NWw':>4} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} "
        f"{'Sh':>7}")
    print("-" * 100)
    for r in results[:10]:
        print(
            f"{r['trend_th']:>4} {r['range_th']:>4} "
            f"{r['top_n']:>3} {r['mps']:>3} "
            f"{r['vlb']:>4} {r['sr']:>4.1f} {r['sb']:>4.1f} "
            f"{r['nw_w']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
            f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 7. Walk-forward for top configs ===
    best_by_ann = results[0]
    best_by_sharpe = max(results, key=lambda x: x["sharpe"])
    best_risk_adj = max(
        results,
        key=lambda x: x["ann"] / max(x["dd"], 1.0))

    for label, best in [
        ("BEST-ANN", best_by_ann),
        ("BEST-SHARPE", best_by_sharpe),
        ("BEST-RISK-ADJ", best_risk_adj),
    ]:
        regime_scores = regime_score_cache[
            (best["trend_th"], best["range_th"])]
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            pred_nw, regime_scores, ker_regime,
            vol_cache[best["vlb"]],
            sector_lookup=sector_lookup,
            top_n=best["top_n"],
            max_per_sector=best["mps"],
            hold_days=5,
            vol_high_mult=2.0,
            vol_low_mult=0.5,
            size_reduce=best["sr"],
            size_boost=best["sb"],
            vol_lookback=best["vlb"],
            nw_weight=best["nw_w"],
            label=label,
        )

    # === 8. Full backtest for top 5 unique configs ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL BACKTEST")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["trend_th"], r["range_th"],
               r["top_n"], r["mps"], r["vlb"],
               r["sr"], r["sb"], r["nw_w"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 5:
            break

    for r in unique_top:
        regime_scores = regime_score_cache[
            (r["trend_th"], r["range_th"])]
        trades, eq, dd = backtest_v98(
            C, O, H, L, NS, ND, dates, syms,
            pred_nw, regime_scores, ker_regime,
            vol_cache[r["vlb"]],
            sector_lookup=sector_lookup,
            top_n=r["top_n"],
            max_per_sector=r["mps"],
            hold_days=5,
            vol_high_mult=2.0,
            vol_low_mult=0.5,
            size_reduce=r["sr"],
            size_boost=r["sb"],
            vol_lookback=r["vlb"],
            nw_weight=r["nw_w"],
            start_di=60,
        )
        label = (
            f"ADXt={r['trend_th']} ADXr={r['range_th']} "
            f"tn={r['top_n']} mps={r['mps']} vlb={r['vlb']} "
            f"sr={r['sr']} sb={r['sb']} nw_w={r['nw_w']}")
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    print(f"\n[V98] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
