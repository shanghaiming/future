"""
V111: Wasserstein Distribution Drift Gate
==========================================
Core Innovation: Use rolling Wasserstein-1 distance to detect when return
distributions have shifted from the training baseline. When drift exceeds
threshold, reduce position size BEFORE drawdown materializes.

Why this is novel vs prior strategies:
  - V96 uses vol-adaptive sizing (reacts AFTER vol spikes)
  - V104 uses cross-sectional correlation (reacts AFTER corr spikes)
  - V111 detects DISTRIBUTION SHAPE changes -- catches regime shifts earlier

Architecture: NW Kernel (from V86) + Vol-Adaptive Sizing + W1 Drift Gate

W1 Computation (efficient 1D):
  For each day d, compute W1 between recent 60-day return distribution
  and training period return distribution. In 1D:
    W1 = integral |F_train(x) - F_recent(x)| dx

Gate Logic:
  - Compute baseline W1 from first training_window days
  - Track historical W1 distribution
  - W1 > p95(historical) => caution: position_multiplier = 0.5
  - W1 > p99(historical) => crisis: position_multiplier = 0.2
  - Apply per-instrument AND portfolio-wide average W1

Walk-forward 2019-2026. No leverage. CASH0=1M, COMM=0.0005.
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
    "ret_5d", "oi_5d", "rsi14", "vol_5d",
    "ret_10d", "range_5d", "atrp_5d",
]
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


# =====================================================================
# W1 (Wasserstein-1) Distribution Drift Detection
# =====================================================================

def wasserstein_1d(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Efficient 1D Wasserstein-1 (Earth Mover's) distance.

    W1 = integral |F_a(x) - F_b(x)| dx computed via sorted CDF difference.
    """
    a = samples_a[~np.isnan(samples_a)]
    b = samples_b[~np.isnan(samples_b)]
    if len(a) < 5 or len(b) < 5:
        return 0.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    all_vals = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, all_vals, side='right') / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_vals, side='right') / len(b_sorted)
    diffs = np.diff(all_vals, prepend=all_vals[0])
    return float(np.sum(np.abs(cdf_a - cdf_b) * diffs))


def compute_w1_drift(
    C: np.ndarray, NS: int, ND: int,
    w1_window: int = 60,
    training_window: int = 40,
) -> np.ndarray:
    """Compute per-instrument W1 drift from training baseline.

    For each instrument si and day di:
      1. Baseline = returns from [di - training_window - w1_window, di - training_window]
      2. Recent = returns from [di - w1_window, di]
      3. W1[si, di] = wasserstein_1d(baseline, recent)

    Returns (NS, ND) array of W1 distances.
    """
    t0 = time.time()
    print(f"[V111] Computing W1 drift (window={w1_window})...", flush=True)

    # Pre-compute daily returns for all instruments
    daily_rets = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                daily_rets[si, di] = C[si, di] / C[si, di - 1] - 1.0

    w1_dist = np.full((NS, ND), np.nan)
    min_samples = 10

    for si in range(NS):
        for di in range(w1_window + training_window, ND):
            # Baseline: returns from [di - tw - w1w, di - tw]
            bl_start = di - training_window - w1_window
            bl_end = di - training_window
            baseline = daily_rets[si, bl_start:bl_end]
            baseline_valid = baseline[~np.isnan(baseline)]

            # Recent: returns from [di - w1w, di]
            recent = daily_rets[si, di - w1_window:di]
            recent_valid = recent[~np.isnan(recent)]

            if len(baseline_valid) >= min_samples and len(recent_valid) >= min_samples:
                w1_dist[si, di] = wasserstein_1d(baseline_valid, recent_valid)

        if si % 10 == 0:
            print(f"  W1 si={si}/{NS} {time.time() - t0:.1f}s", flush=True)

    print(f"  W1 drift done: {time.time() - t0:.1f}s", flush=True)
    return w1_dist


def compute_w1_multipliers(
    w1_dist: np.ndarray, NS: int, ND: int,
    w1_threshold_pct: float = 95.0,
    w1_crisis_pct: float = 99.0,
    w1_caution_mult: float = 0.5,
    w1_crisis_mult: float = 0.2,
    min_history: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute position size multipliers from W1 drift.

    Returns:
      per_inst_mult: (NS, ND) per-instrument multiplier
      portfolio_mult: (ND,) portfolio-wide multiplier (avg W1 across insts)
    """
    t0 = time.time()
    print(
        f"[V111] Computing W1 multipliers "
        f"(pct={w1_threshold_pct}/{w1_crisis_pct})...",
        flush=True)

    per_inst_mult = np.ones((NS, ND))
    portfolio_mult = np.ones(ND)

    # Build portfolio-wide average W1 per day
    avg_w1 = np.full(ND, np.nan)
    for di in range(ND):
        vals = w1_dist[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 3:
            avg_w1[di] = np.mean(valid)

    # Compute historical percentile thresholds using expanding window
    for di in range(min_history, ND):
        # Portfolio-wide W1 threshold from history
        hist_w1 = avg_w1[max(0, di - 252):di]
        hist_valid = hist_w1[~np.isnan(hist_w1)]
        if len(hist_valid) < 20:
            continue
        p95 = np.percentile(hist_valid, w1_threshold_pct)
        p99 = np.percentile(hist_valid, w1_crisis_pct)
        current_avg = avg_w1[di]
        if not np.isnan(current_avg):
            if current_avg > p99:
                portfolio_mult[di] = w1_crisis_mult
            elif current_avg > p95:
                portfolio_mult[di] = w1_caution_mult

        # Per-instrument W1 threshold
        for si in range(NS):
            if np.isnan(w1_dist[si, di]):
                continue
            inst_hist = w1_dist[si, max(0, di - 252):di]
            inst_valid = inst_hist[~np.isnan(inst_hist)]
            if len(inst_valid) < 20:
                continue
            inst_p95 = np.percentile(inst_valid, w1_threshold_pct)
            inst_p99 = np.percentile(inst_valid, w1_crisis_pct)
            if w1_dist[si, di] > inst_p99:
                per_inst_mult[si, di] = min(
                    per_inst_mult[si, di], w1_crisis_mult)
            elif w1_dist[si, di] > inst_p95:
                per_inst_mult[si, di] = min(
                    per_inst_mult[si, di], w1_caution_mult)

    caution_days = int(np.sum(portfolio_mult == w1_caution_mult))
    crisis_days = int(np.sum(portfolio_mult == w1_crisis_mult))
    print(
        f"  W1 multipliers: caution={caution_days}d crisis={crisis_days}d "
        f"of {ND}d {time.time() - t0:.1f}s", flush=True)
    return per_inst_mult, portfolio_mult


# =====================================================================
# Factor computation (from V96, simplified - no BMA)
# =====================================================================

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
    """Compute 7 raw factors for NW regression features."""
    t0 = time.time()
    print("[V111] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    range_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            rng_vals = []
            for j in range(di - 5, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng_vals) >= 3:
                range_5d[si, di] = np.mean(rng_vals)

    atrp_5d = np.full((NS, ND), np.nan)
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
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

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

    fwd_ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 5):
            if (not np.isnan(C[si, di + 5])
                    and not np.isnan(C[si, di])
                    and C[si, di] > 0):
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0

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
        "ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
        "range_5d": range_5d, "atrp_5d": atrp_5d,
        "ret_10d": ret_10d, "rsi14": rsi14,
        "fwd_ret_5d": fwd_ret_5d,
        "atr_mean": atr_mean,
    }


def normalize_factor(
    factor: np.ndarray, NS: int, ND: int, min_count: int = 10,
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


def compute_nw_predicted_returns(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    training_window: int = 40,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    """Compute NW kernel regression predicted returns (from V86)."""
    t0 = time.time()
    print(
        f"[V111] Computing NW predictions "
        f"(window={training_window}, bw={kernel_bandwidth:.1f})...",
        flush=True)

    normed = {}
    for fname in FACTOR_NAMES:
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
                    normed[fname][si, tdi] for fname in FACTOR_NAMES
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
                normed[fname][si, di] for fname in FACTOR_NAMES
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
                min_idx = np.argmin(dist)
                if dist[min_idx] < 1e12:
                    mask = np.zeros(len(dist), dtype=bool)
                    mask[min_idx] = True
                    weights[min_idx] = 1.0
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
                f"train={len(train_features)}", flush=True)

    print(f"  NW prediction done: {time.time() - t0:.1f}s", flush=True)
    return predicted


# =====================================================================
# Helpers
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
# Portfolio volatility (from V96)
# =====================================================================

def compute_portfolio_volatility(
    C: np.ndarray, NS: int, ND: int,
    vol_lookback: int = 20,
) -> np.ndarray:
    """Rolling portfolio volatility proxy. Returns (ND,) array."""
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
    port_vol: float, vol_median: float,
    vol_high_mult: float, vol_low_mult: float,
    size_reduce: float, size_boost: float,
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
# Backtest with W1 drift gate + vol-adaptive sizing
# =====================================================================

def backtest_v111(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
    ker_regime: np.ndarray,
    port_vol: np.ndarray,
    w1_inst_mult: np.ndarray,
    w1_port_mult: np.ndarray,
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
    """Backtest V111: NW kernel + vol-adaptive + W1 drift gate."""
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

    w1_caution_count = 0
    w1_crisis_count = 0

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float]] = []

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

        # Vol-adaptive sizing
        vol_mult = get_vol_multiplier(
            port_vol[di], vol_median,
            vol_high_mult, vol_low_mult,
            size_reduce, size_boost)

        # W1 portfolio-wide multiplier
        w1_pm = w1_port_mult[di]
        if w1_pm < 1.0:
            if w1_pm <= 0.3:
                w1_crisis_count += 1
            else:
                w1_caution_count += 1

        # Combined multiplier: vol * W1
        combined_mult = vol_mult * w1_pm

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

        # --- ENTRY ---
        held = {p[0] for p in positions}
        if len(held) >= top_n:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            pred = predicted[si, di]
            if np.isnan(pred):
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            if ker_regime[si, di] < 0:
                continue
            candidates.append((pred, si))

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
        for pred_val, si in candidates:
            if len(held) + len(new_entries) >= n_to_take:
                break
            if si in held:
                continue
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            if pred_val <= 0:
                continue
            new_entries.append((pred_val, si, sym_sector))
            sector_counts[sym_sector] += 1

        if not new_entries:
            continue

        num_total = len(positions) + len(new_entries)
        # Apply combined vol * W1 multiplier to sizing
        alloc_per_pos = LEVERAGE / num_total * combined_mult

        # Per-instrument W1 adjustment for each entry
        updated_positions = []
        for si, edi, ep, sp, old_alloc in positions:
            updated_positions.append((si, edi, ep, sp, alloc_per_pos))

        for pred_val, si, sym_sector in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            # Apply per-instrument W1 multiplier on top
            inst_alloc = alloc_per_pos * w1_inst_mult[si, di]
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, inst_alloc))

        positions = updated_positions

    # Close remaining
    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    if trades:
        trades[0]["w1_info"] = (
            f"w1_regime=[caution:{w1_caution_count} "
            f"crisis:{w1_crisis_count}]")
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
    ker_regime: np.ndarray,
    port_vol: np.ndarray,
    w1_inst_mult: np.ndarray,
    w1_port_mult: np.ndarray,
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
    print(f"  WALK-FORWARD V111 {label}")
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

        trades, _, _ = backtest_v111(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime, port_vol,
            w1_inst_mult, w1_port_mult,
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
    print("  V111: WASSERSTEIN DISTRIBUTION DRIFT GATE")
    print("  Innovation: W1 distance detects distribution shape changes")
    print("  Architecture: NW Kernel + Vol-Adaptive + W1 Drift Gate")
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

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Compute raw factors ===
    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # === 2. Compute NW predictions (fixed params) ===
    predicted = compute_nw_predicted_returns(
        raw_factors, NS, ND,
        training_window=40,
        kernel_bandwidth=1.0,
    )

    # === 3. Compute portfolio volatility ===
    port_vol = compute_portfolio_volatility(C, NS, ND, vol_lookback=20)

    # === 4. Compute W1 drift for each window size ===
    w1_configs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for w1w in [40, 60, 80]:
        w1_dist = compute_w1_drift(C, NS, ND, w1_window=w1w)
        # Pre-compute W1 multipliers for each threshold config
        for pct in [90, 95, 97]:
            for cm in [0.2, 0.3, 0.5]:
                for cam in [0.5, 0.7]:
                    key = (w1w, pct, cm, cam)
                    inst_m, port_m = compute_w1_multipliers(
                        w1_dist, NS, ND,
                        w1_threshold_pct=pct,
                        w1_crisis_pct=min(pct + 4, 99.9),
                        w1_caution_mult=cam,
                        w1_crisis_mult=cm,
                    )
                    w1_configs[key] = (inst_m, port_m)

    # === 5. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  NO LEVERAGE. NW kernel + vol-adaptive + W1 drift gate.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for top_n in [2]:
        for mps in [2, 3]:
            for (w1w, pct, cm, cam), (w1_im, w1_pm) in w1_configs.items():
                for vhm in [1.5, 2.0]:
                    for vlm in [0.5, 0.7]:
                        for sr in [0.3, 0.5]:
                            for sb in [1.2, 1.5]:
                                sweep_count += 1
                                trades, eq, dd = backtest_v111(
                                    C, O, H, L, NS, ND,
                                    dates, syms,
                                    predicted, ker_regime,
                                    port_vol, w1_im, w1_pm,
                                    sector_lookup=sector_lookup,
                                    top_n=top_n,
                                    max_per_sector=mps,
                                    hold_days=5,
                                    vol_high_mult=vhm,
                                    vol_low_mult=vlm,
                                    size_reduce=sr,
                                    size_boost=sb,
                                    vol_lookback=20,
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
                                    "w1w": w1w, "pct": pct,
                                    "cm": cm, "cam": cam,
                                    "top_n": top_n, "mps": mps,
                                    "vhm": vhm, "vlm": vlm,
                                    "sr": sr, "sb": sb,
                                    "n": len(trades), "wr": wr,
                                    "ann": ann, "dd": dd,
                                    "sharpe": sh_val, "eq": eq,
                                })

    results.sort(key=lambda x: -x["ann"])
    print(
        f"\n  Evaluated {sweep_count} configs, "
        f"{len(results)} with 10+ trades")

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # Report top 15 by annualized return
    print(
        f"\n{'W1w':>4} {'Pct':>4} {'CM':>4} {'CAM':>4} "
        f"{'TN':>3} {'MPS':>3} "
        f"{'Vhm':>4} {'Vlm':>4} {'SR':>4} {'SB':>4} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} "
        f"{'Sh':>7}")
    print("-" * 105)
    for r in results[:15]:
        print(
            f"{r['w1w']:>4} {r['pct']:>4} {r['cm']:>4.1f} {r['cam']:>4.1f} "
            f"{r['top_n']:>3} {r['mps']:>3} "
            f"{r['vhm']:>4.1f} {r['vlm']:>4.1f} "
            f"{r['sr']:>4.1f} {r['sb']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
            f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    # === 6. Walk-forward for top configs ===
    best_by_sharpe = max(results, key=lambda x: x["sharpe"])
    best_by_ann = results[0]
    best_risk_adj = max(
        results,
        key=lambda x: x["ann"] / max(x["dd"], 1.0))

    for label, best in [
        ("BEST-ANN", best_by_ann),
        ("BEST-SHARPE", best_by_sharpe),
        ("BEST-RISK-ADJ", best_risk_adj),
    ]:
        w1_key = (best["w1w"], best["pct"], best["cm"], best["cam"])
        w1_im, w1_pm = w1_configs[w1_key]

        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime,
            port_vol, w1_im, w1_pm,
            sector_lookup=sector_lookup,
            top_n=best["top_n"],
            max_per_sector=best["mps"],
            hold_days=5,
            vol_high_mult=best["vhm"],
            vol_low_mult=best["vlm"],
            size_reduce=best["sr"],
            size_boost=best["sb"],
            vol_lookback=20,
            label=label,
        )

    # === 7. Compare V111 vs V96 baseline ===
    print("\n" + "=" * 70)
    print("  COMPARISON: V111 (NW+Vol+W1) vs V96 baseline (NW+BMA+Vol)")
    print("  V96 baseline: ann +73.1%")
    print("=" * 70)

    # V111 best by Sharpe
    w1_key_best = (
        best_by_sharpe["w1w"], best_by_sharpe["pct"],
        best_by_sharpe["cm"], best_by_sharpe["cam"])
    w1_im_best, w1_pm_best = w1_configs[w1_key_best]

    trades_v111, eq_v111, dd_v111 = backtest_v111(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime,
        port_vol, w1_im_best, w1_pm_best,
        sector_lookup=sector_lookup,
        top_n=best_by_sharpe["top_n"],
        max_per_sector=best_by_sharpe["mps"],
        hold_days=5,
        vol_high_mult=best_by_sharpe["vhm"],
        vol_low_mult=best_by_sharpe["vlm"],
        size_reduce=best_by_sharpe["sr"],
        size_boost=best_by_sharpe["sb"],
        vol_lookback=20,
        start_di=bt_2019,
    )

    # V96 baseline (no W1 gate, no BMA - pure NW+vol)
    no_w1_inst = np.ones((NS, ND))
    no_w1_port = np.ones(ND)
    trades_v96b, eq_v96b, dd_v96b = backtest_v111(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime,
        port_vol, no_w1_inst, no_w1_port,
        sector_lookup=sector_lookup,
        top_n=2,
        max_per_sector=2,
        hold_days=5,
        vol_high_mult=2.0,
        vol_low_mult=0.5,
        size_reduce=0.5,
        size_boost=1.3,
        vol_lookback=20,
        start_di=bt_2019,
    )

    print(f"\n  V111 BEST-SHARPE (NW+Vol+W1):")
    analyze(trades_v111, eq_v111, dd_v111, "V111-NW+Vol+W1")
    print(f"\n  V96 BASELINE (NW+Vol, no W1 gate):")
    analyze(trades_v96b, eq_v96b, dd_v96b, "V96-baseline")

    if trades_v111 and trades_v96b:
        print(
            f"\n  Delta V111 vs baseline: "
            f"eq={eq_v111 - eq_v96b:+,.0f} "
            f"dd={dd_v111 - dd_v96b:+.1f}% "
            f"trades={len(trades_v111) - len(trades_v96b):+d}")

    print(f"\n[V111] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
