"""
V99: Volume Directional Breakout Strategy
==========================================
Core Insight: Build a complete "smart money flow" dimension using
3-dimensional volume signals — MFI + ADOSC + OBV.

Innovation over V96 (which only has vol_5d = volume LEVEL):
- MFI(14): Volume-weighted RSI — volume-price combined overbought/oversold
- ADOSC(3,10): Chaikin A/D Oscillator — money flow direction
- OBV_chg5d: Cumulative volume momentum (5d rate of change)
- PPO(12,26): Moving average momentum acceleration (cross-instrument comparable)

Factor Mix (7 factors):
| Factor         | Weight | Information Dimension         |
|----------------|--------|-------------------------------|
| ret5d_rank     | 0.15   | Price momentum                |
| MFI(14)_rank   | 0.20   | Volume-price RSI              |
| ADOSC(3,10)_rank | 0.20 | Money flow direction          |
| PPO(12,26)_rank  | 0.15 | MA momentum acceleration      |
| OBV_chg5d_rank   | 0.10 | Cumulative volume momentum    |
| oi5d_rank      | 0.10   | Open interest change          |
| rsi14_rank     | 0.10   | Overbought/oversold           |

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

FACTOR_NAMES = [
    "ret_5d", "mfi14", "adosc_3_10", "ppo_12_26",
    "obv_chg5d", "oi_5d", "rsi14",
]
FACTOR_WEIGHTS = np.array([0.15, 0.20, 0.20, 0.15, 0.10, 0.10, 0.10])
N_FACTORS = len(FACTOR_NAMES)


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


def compute_rsi_manual(
    C: np.ndarray, NS: int, ND: int, period: int = 14,
) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        gains = np.full(ND, np.nan)
        losses = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di - 1]):
                continue
            delta = c[di] - c[di - 1]
            gains[di] = max(delta, 0.0)
            losses[di] = max(-delta, 0.0)

        avg_gain = np.nan
        avg_loss = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]):
                continue
            if np.isnan(avg_gain):
                valid_g = []
                valid_l = []
                for j in range(di, min(di + period, ND)):
                    if not np.isnan(gains[j]):
                        valid_g.append(gains[j])
                        valid_l.append(
                            losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(valid_g) >= period:
                    avg_gain = np.mean(valid_g)
                    avg_loss = np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[si, di + period - 1] = (
                            100.0 - 100.0 / (1.0 + rs))
                continue

            avg_gain = (avg_gain * (period - 1) + gains[di]) / period
            avg_loss = (avg_loss * (period - 1) + losses[di]) / period
            if avg_loss == 0:
                rsi[si, di] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[si, di] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_raw_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute 7 raw factors including volume directional signals."""
    t0 = time.time()
    print("[V99] Computing raw factors (volume directional breakout)...",
          flush=True)

    # --- Factor 1: ret_5d (price momentum) ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    # --- Factor 2: MFI(14) — Money Flow Index (volume-weighted RSI) ---
    mfi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            v = np.where(np.isnan(V[si]), 0, V[si]).astype(np.float64)
            nan_mask = np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(
                C[si]) | np.isnan(V[si])
            try:
                r = talib.MFI(h, l, c, v, timeperiod=14)
                mfi14[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    # --- Factor 3: ADOSC(3,10) — Chaikin A/D Oscillator ---
    adosc_3_10 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            v = np.where(np.isnan(V[si]), 0, V[si]).astype(np.float64)
            nan_mask = np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(
                C[si]) | np.isnan(V[si])
            try:
                r = talib.ADOSC(h, l, c, v, fastperiod=3, slowperiod=10)
                adosc_3_10[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    # --- Factor 4: PPO(12,26) — Price Oscillator percentage ---
    ppo_12_26 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.PPO(c, fastperiod=12, slowperiod=26,
                              matype=0)  # SMA
                ppo_12_26[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    # --- Factor 5: OBV_chg5d — OBV 5-day rate of change ---
    obv_raw = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            v = np.where(np.isnan(V[si]), 0, V[si]).astype(np.float64)
            nan_mask = np.isnan(C[si]) | np.isnan(V[si])
            try:
                r = talib.OBV(c, v)
                obv_raw[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    obv_chg5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(obv_raw[si, di])
                    and not np.isnan(obv_raw[si, di - 5])
                    and obv_raw[si, di - 5] != 0):
                obv_chg5d[si, di] = (
                    obv_raw[si, di] / obv_raw[si, di - 5] - 1.0)

    # --- Factor 6: oi_5d (open interest change) ---
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # --- Factor 7: rsi14 ---
    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    needs_fallback = np.all(np.isnan(rsi14), axis=1)
    if needs_fallback.any():
        rsi_manual = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fallback[si]:
                rsi14[si] = rsi_manual[si]

    # --- Target: next 5-day forward return ---
    fwd_ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 5):
            if (not np.isnan(C[si, di + 5])
                    and not np.isnan(C[si, di])
                    and C[si, di] > 0):
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0

    # --- ATR mean for adaptive bandwidth ---
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

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d,
        "mfi14": mfi14,
        "adosc_3_10": adosc_3_10,
        "ppo_12_26": ppo_12_26,
        "obv_chg5d": obv_chg5d,
        "oi_5d": oi_5d,
        "rsi14": rsi14,
        "fwd_ret_5d": fwd_ret_5d,
        "atr_mean": atr_mean,
    }


def normalize_factor(factor: np.ndarray, NS: int, ND: int,
                     min_count: int = 10) -> np.ndarray:
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
# Bayesian Model Averaging for factor weighting
# =====================================================================

def compute_rolling_ic(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    ic_window: int = 60,
    min_pairs: int = 15,
) -> np.ndarray:
    """Compute rolling IC for each factor."""
    t0 = time.time()
    print(f"[V99] Computing rolling IC (window={ic_window})...", flush=True)

    fwd_ret = raw_factors["fwd_ret_5d"]
    ic_array = np.full((N_FACTORS, ND), np.nan)

    for fi, fname in enumerate(FACTOR_NAMES):
        factor = raw_factors[fname]
        for di in range(ic_window + 5, ND):
            ic_vals = []
            for tdi in range(di - ic_window, di):
                f_day = factor[:, tdi]
                r_day = fwd_ret[:, tdi]
                valid_mask = (~np.isnan(f_day)) & (~np.isnan(r_day))
                f_valid = f_day[valid_mask]
                r_valid = r_day[valid_mask]
                if len(f_valid) >= min_pairs:
                    f_rank = pd.Series(f_valid).rank().values
                    r_rank = pd.Series(r_valid).rank().values
                    corr = np.corrcoef(f_rank, r_rank)[0, 1]
                    if not np.isnan(corr):
                        ic_vals.append(corr)

            if len(ic_vals) >= 5:
                ic_array[fi, di] = np.mean(ic_vals)

        if fi % 2 == 0:
            print(f"  IC for {fname}: {time.time() - t0:.1f}s", flush=True)

    print(f"  Rolling IC done: {time.time() - t0:.1f}s", flush=True)
    return ic_array


def compute_bma_weights(
    ic_array: np.ndarray,
    ND: int,
    prior_strength: float = 5.0,
    min_ic_history: int = 20,
) -> np.ndarray:
    """Compute BMA weights from IC history."""
    t0 = time.time()
    print(
        f"[V99] Computing BMA weights (prior={prior_strength:.1f})...",
        flush=True)

    weights = np.full((N_FACTORS, ND), np.nan)

    for fi in range(N_FACTORS):
        for di in range(min_ic_history, ND):
            ic_hist = ic_array[fi, max(0, di - 120):di]
            valid_ic = ic_hist[~np.isnan(ic_hist)]

            if len(valid_ic) < 5:
                continue

            n_positive = np.sum(valid_ic > 0)
            n_negative = len(valid_ic) - n_positive

            alpha_post = prior_strength / 2.0 + n_positive
            beta_post = prior_strength / 2.0 + n_negative

            weights[fi, di] = alpha_post / (alpha_post + beta_post)

    # Normalize weights to sum to 1 across factors for each day
    for di in range(ND):
        w = weights[:, di]
        valid = w[~np.isnan(w)]
        if len(valid) == N_FACTORS:
            w_sum = np.nansum(w)
            if w_sum > 0:
                weights[:, di] = w / w_sum

    print(f"  BMA weights done: {time.time() - t0:.1f}s", flush=True)
    return weights


def apply_bma_to_features(
    normed_factors: Dict[str, np.ndarray],
    bma_weights: np.ndarray,
    NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Apply BMA weighting to normalized features."""
    t0 = time.time()
    print("[V99] Applying BMA weights to features...", flush=True)

    weighted = {}
    for fi, fname in enumerate(FACTOR_NAMES):
        original = normed_factors[fname]
        result = np.full((NS, ND), np.nan)
        for di in range(ND):
            w = bma_weights[fi, di]
            if np.isnan(w):
                w = 1.0 / N_FACTORS
            for si in range(NS):
                if not np.isnan(original[si, di]):
                    result[si, di] = original[si, di] * (w * N_FACTORS)
        weighted[fname] = result

    print(f"  BMA feature weighting done: {time.time() - t0:.1f}s",
          flush=True)
    return weighted


def compute_nw_predicted_returns(
    raw_factors: Dict[str, np.ndarray],
    bma_weights: np.ndarray,
    NS: int, ND: int,
    training_window: int = 40,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    """NW kernel regression with BMA-weighted features."""
    t0 = time.time()
    print(
        f"[V99] Computing NW+BMA predicted returns "
        f"(window={training_window}, bw={kernel_bandwidth:.1f})...",
        flush=True)

    # Normalize factors cross-sectionally
    normed = {}
    for fname in FACTOR_NAMES:
        normed[fname] = normalize_factor(raw_factors[fname], NS, ND)

    # Apply BMA weighting
    weighted_normed = apply_bma_to_features(normed, bma_weights, NS, ND)

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
                    weighted_normed[fname][si, tdi]
                    for fname in FACTOR_NAMES
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
                weighted_normed[fname][si, di] for fname in FACTOR_NAMES
            ])
            if np.any(np.isnan(query_feat)):
                continue

            atr_val = atr_mean[si, di]
            if np.isnan(atr_val):
                h = kernel_bandwidth
            else:
                h = atr_val * kernel_bandwidth
                h = max(h, 0.1)

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
                    mask = np.array([False] * len(dist))
                    mask[min_dist_idx] = True
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

    print(f"  NW+BMA prediction done: {time.time() - t0:.1f}s", flush=True)
    return predicted


# =====================================================================
# Alternative: Fixed-weight composite score (no NW kernel)
# =====================================================================

def compute_fixed_weight_score(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
) -> np.ndarray:
    """Compute composite score using fixed factor weights + cross-sectional
    rank normalization. Fast alternative to NW kernel."""
    t0 = time.time()
    print("[V99] Computing fixed-weight composite score...", flush=True)

    score = np.zeros((NS, ND))
    valid_count = np.zeros((NS, ND))

    for fi, fname in enumerate(FACTOR_NAMES):
        factor = raw_factors[fname]
        w = FACTOR_WEIGHTS[fi]
        # Cross-sectional rank normalization (0-1 scale)
        for di in range(ND):
            vals = factor[:, di]
            valid_mask = ~np.isnan(vals)
            n_valid = np.sum(valid_mask)
            if n_valid < 10:
                continue
            ranks = np.full(NS, np.nan)
            valid_vals = vals[valid_mask]
            sorted_idx = np.argsort(valid_vals)
            for rank_pos, idx in enumerate(sorted_idx):
                orig_si = np.where(valid_mask)[0][idx]
                ranks[orig_si] = (rank_pos + 1) / n_valid
            for si in range(NS):
                if not np.isnan(ranks[si]):
                    score[si, di] += w * ranks[si]
                    valid_count[si, di] += 1

    # Only return scores where at least 5 of 7 factors are available
    for si in range(NS):
        for di in range(ND):
            if valid_count[si, di] < 5:
                score[si, di] = np.nan

    print(f"  Fixed-weight score done: {time.time() - t0:.1f}s", flush=True)
    return score


# =====================================================================
# Helper functions
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
# Volatility-Adaptive Position Sizing
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
    if np.isnan(port_vol) or np.isnan(vol_median) or vol_median < 1e-12:
        return 1.0
    ratio = port_vol / vol_median
    if ratio > vol_high_mult:
        return size_reduce
    elif ratio < vol_low_mult:
        return size_boost
    return 1.0


# =====================================================================
# Backtest
# =====================================================================

def backtest_v99(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    signal: np.ndarray,
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
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V99 with volume directional breakout signals."""
    if end_di is None:
        end_di = ND - 1

    vol_data = port_vol[max(start_di, vol_lookback + 1):end_di]
    vol_data_valid = vol_data[~np.isnan(vol_data)]
    vol_median = (np.median(vol_data_valid)
                  if len(vol_data_valid) > 10 else 1e-6)

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

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

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

        # --- ENTRY: select top_n by signal ---
        held = {p[0] for p in positions}
        if len(held) >= top_n:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            sig = signal[si, di]
            if np.isnan(sig):
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            if ker_regime[si, di] < 0:
                continue
            candidates.append((sig, si))

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
        for sig_val, si in candidates:
            if len(held) + len(new_entries) >= n_to_take:
                break
            if si in held:
                continue
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            if sig_val <= 0:
                continue
            new_entries.append((sig_val, si, sym_sector))
            sector_counts[sym_sector] += 1

        if not new_entries:
            continue

        num_total = len(positions) + len(new_entries)
        alloc_per_pos = LEVERAGE / num_total * vol_mult

        updated_positions = []
        for si, edi, ep, sp, old_alloc in positions:
            updated_positions.append((si, edi, ep, sp, alloc_per_pos))

        for sig_val, si, sym_sector in new_entries:
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


def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "") -> Optional[dict]:
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
        f"avg_loss={avg_loss:+.3f}%")
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
    signal: np.ndarray,
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
    label: str = "",
) -> List[dict]:
    cfg_str = (
        f"tn={top_n} mps={max_per_sector} hd={hold_days} "
        f"vhm={vol_high_mult:.1f} vlm={vol_low_mult:.1f} "
        f"sr={size_reduce:.1f} sb={size_boost:.1f} vlb={vol_lookback}")
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V99 {label}")
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

        trades, _, _ = backtest_v99(
            C, O, H, L, NS, ND, dates, syms,
            signal, ker_regime, port_vol,
            sector_lookup=sector_lookup,
            top_n=top_n,
            max_per_sector=max_per_sector,
            hold_days=hold_days,
            vol_high_mult=vol_high_mult,
            vol_low_mult=vol_low_mult,
            size_reduce=size_reduce,
            size_boost=size_boost,
            vol_lookback=vol_lookback,
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
                yr_sectors[t.get("sector", "OTHER")] += 1
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
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V99: VOLUME DIRECTIONAL BREAKOUT")
    print("  Innovation: MFI + ADOSC + OBV 3D volume signals")
    print("  Framework: NW kernel + BMA + vol-adaptive sizing")
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

    # === 1. Compute raw factors ===
    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # === 2. Compute BMA weights ===
    ic_window = 60
    prior_strength = 5.0
    ic_array = compute_rolling_ic(raw_factors, NS, ND, ic_window=ic_window)
    bma_w = compute_bma_weights(ic_array, ND, prior_strength=prior_strength)

    # === 3. Compute NW+BMA predicted returns ===
    print("\n--- Computing NW+BMA predictions (V99 volume factors) ---")
    pred_nw_bma = compute_nw_predicted_returns(
        raw_factors, bma_w, NS, ND,
        training_window=40,
        kernel_bandwidth=1.0,
    )

    # === 4. Compute fixed-weight composite score (fast path) ===
    score_fixed = compute_fixed_weight_score(raw_factors, NS, ND)

    # === 5. Compute portfolio volatility ===
    vol_cache: Dict[int, np.ndarray] = {}
    for vlb in [15, 20]:
        vol_cache[vlb] = compute_portfolio_volatility(C, NS, ND, vlb)

    # === 6. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  Signals: NW+BMA kernel + Fixed-weight composite")
    print("  NO LEVERAGE.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    # Signal sources to sweep
    signal_sources: List[Tuple[str, np.ndarray]] = [
        ("NW+BMA", pred_nw_bma),
        ("FIXED", score_fixed),
    ]

    for sig_label, signal in signal_sources:
        for top_n in [2, 3, 4]:
            for mps in [2, 3]:
                for vlb in [15, 20]:
                    for sr in [0.3, 0.5]:
                        for sb in [1.2, 1.5]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v99(
                                C, O, H, L, NS, ND,
                                dates, syms,
                                signal, ker_regime,
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
                                start_di=bt_2019,
                            )

                            if len(trades) < 10:
                                continue

                            nw = sum(
                                1 for t in trades
                                if t["pnl_pct"] > 0)
                            wr = nw / len(trades) * 100
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
                                "sig": sig_label,
                                "top_n": top_n, "mps": mps,
                                "vlb": vlb,
                                "sr": sr, "sb": sb,
                                "n": len(trades), "wr": wr,
                                "ann": ann, "dd": dd,
                                "sharpe": sh_val, "eq": eq,
                            })

    results.sort(key=lambda x: -x["ann"])
    print(
        f"\n  Evaluated {sweep_count} configs, "
        f"{len(results)} with 10+ trades")

    # Report top 10 by annualized return
    print(
        f"\n{'Sig':>6} {'TN':>3} {'MPS':>3} "
        f"{'Vlb':>4} {'SR':>4} {'SB':>4} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} "
        f"{'Sh':>7}")
    print("-" * 80)
    for r in results[:10]:
        print(
            f"{r['sig']:>6} {r['top_n']:>3} {r['mps']:>3} "
            f"{r['vlb']:>4} {r['sr']:>4.1f} {r['sb']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
            f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 7. Walk-forward for best configs ===
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
        sig_source = dict(signal_sources)[best["sig"]]
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            sig_source, ker_regime,
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
            label=label,
        )

    # === 8. Final comparison ===
    print("\n" + "=" * 70)
    print("  V99 FINAL RESULTS (2019-2026 OOS)")
    print(f"  Target: beat V96 ann +73.1%")
    print("=" * 70)

    sig_best = dict(signal_sources)[best_by_ann["sig"]]
    trades_best, eq_best, dd_best = backtest_v99(
        C, O, H, L, NS, ND, dates, syms,
        sig_best, ker_regime,
        vol_cache[best_by_ann["vlb"]],
        sector_lookup=sector_lookup,
        top_n=best_by_ann["top_n"],
        max_per_sector=best_by_ann["mps"],
        hold_days=5,
        vol_high_mult=2.0,
        vol_low_mult=0.5,
        size_reduce=best_by_ann["sr"],
        size_boost=best_by_ann["sb"],
        vol_lookback=best_by_ann["vlb"],
        start_di=bt_2019,
    )
    analyze(trades_best, eq_best, dd_best, "V99-BEST-ANN")

    print(f"\n[V99] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
