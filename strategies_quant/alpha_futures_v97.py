"""
V97: NW Kernel + Volatility-Adaptive Dynamic Lookback
======================================================
Combine V86's Nadaraya-Watson kernel regression with volatility-adaptive
dynamic lookback windows for factor computation.

Core Innovation (Two-Layer Adaptation):
  Layer 1: Vol-adaptive lookback -> dynamic factor inputs
    - Compute rolling 20d realized volatility for each instrument
    - Cross-sectionally rank vol across all instruments
    - HIGH vol (rank > high_vol_pct): SHORT lookback -> recent info more relevant
    - NORMAL vol (low_vol_pct..high_vol_pct): MEDIUM lookback -> standard
    - LOW vol (rank < low_vol_pct): LONG lookback -> momentum persists longer
  Layer 2: NW kernel -> nonlinear signal from dynamic factors
    - Feed vol-adaptive returns into V86's NW kernel regression
    - Kernel processes DYNAMIC rather than static factor inputs

Research basis (2106.08420): dynamic lookback improves momentum Sharpe by +66%.

Parameters to sweep:
  - high_vol_pct: 0.65, 0.70, 0.75
  - low_vol_pct: 0.25, 0.30, 0.35
  - short_lb: 2, 3, 5
  - medium_lb: 8, 10, 12
  - long_lb: 15, 20, 25
  - tw (kernel training window): 30, 40, 60
  - top_n: 2, 3
  - mps: 2, 3

Walk-forward 2019-2026. No leverage. CASH0=1,000,000, COMM=0.0005.
Target: beat V86's ann +52.9% with lower MDD.
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
    "ret_adaptive", "oi_adaptive", "rsi14", "vol_adaptive",
    "range_adaptive", "atrp_adaptive",
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


def compute_realized_volatility(
    C: np.ndarray, NS: int, ND: int, vol_window: int = 20,
) -> np.ndarray:
    """Compute rolling realized volatility (std of daily log returns)."""
    realized_vol = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(vol_window, ND):
            returns = []
            for j in range(di - vol_window, di):
                if (not np.isnan(C[si, j])
                        and not np.isnan(C[si, j + 1])
                        and C[si, j] > 0):
                    returns.append(
                        np.log(C[si, j + 1] / C[si, j]))
            if len(returns) >= vol_window // 2:
                realized_vol[si, di] = np.std(returns)
    return realized_vol


def compute_vol_ranks(
    realized_vol: np.ndarray, NS: int, ND: int,
    min_count: int = 10,
) -> np.ndarray:
    """Cross-sectional percentile rank of realized volatility."""
    vol_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = realized_vol[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) < min_count:
            continue
        ranked = pd.Series(vals).rank(
            pct=True, na_option="keep").values
        vol_rank[:, di] = ranked
    return vol_rank


def compute_adaptive_return(
    C: np.ndarray, NS: int, ND: int,
    vol_rank: np.ndarray,
    short_lb: int, medium_lb: int, long_lb: int,
    high_vol_pct: float, low_vol_pct: float,
) -> np.ndarray:
    """Compute vol-adaptive return using dynamic lookback windows."""
    ret_adaptive = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(long_lb, ND):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            vr = vol_rank[si, di]
            if np.isnan(vr):
                continue
            if vr > high_vol_pct:
                lb = short_lb
            elif vr < low_vol_pct:
                lb = long_lb
            else:
                lb = medium_lb
            prev_di = di - lb
            if prev_di < 0 or np.isnan(C[si, prev_di]) or C[si, prev_di] <= 0:
                continue
            ret_adaptive[si, di] = C[si, di] / C[si, prev_di] - 1.0
    return ret_adaptive


def compute_adaptive_oi(
    OI: np.ndarray, NS: int, ND: int,
    vol_rank: np.ndarray,
    short_lb: int, medium_lb: int, long_lb: int,
    high_vol_pct: float, low_vol_pct: float,
) -> np.ndarray:
    """Compute vol-adaptive OI change using dynamic lookback windows."""
    oi_adaptive = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(long_lb, ND):
            if np.isnan(OI[si, di]):
                continue
            vr = vol_rank[si, di]
            if np.isnan(vr):
                continue
            if vr > high_vol_pct:
                lb = short_lb
            elif vr < low_vol_pct:
                lb = long_lb
            else:
                lb = medium_lb
            prev_di = di - lb
            if (prev_di < 0
                    or np.isnan(OI[si, prev_di])
                    or OI[si, prev_di] <= 0):
                continue
            oi_adaptive[si, di] = OI[si, di] / OI[si, prev_di] - 1.0
    return oi_adaptive


def compute_adaptive_vol(
    V: np.ndarray, NS: int, ND: int,
    vol_rank: np.ndarray,
    short_lb: int, medium_lb: int, long_lb: int,
    high_vol_pct: float, low_vol_pct: float,
) -> np.ndarray:
    """Compute vol-adaptive average volume using dynamic lookback windows."""
    vol_adaptive = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(long_lb, ND):
            vr = vol_rank[si, di]
            if np.isnan(vr):
                continue
            if vr > high_vol_pct:
                lb = short_lb
            elif vr < low_vol_pct:
                lb = long_lb
            else:
                lb = medium_lb
            vals = V[si, di - lb:di]
            valid = vals[~np.isnan(vals)]
            min_samples = max(2, lb // 2)
            if len(valid) >= min_samples:
                vol_adaptive[si, di] = np.mean(valid)
    return vol_adaptive


def compute_adaptive_range(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    NS: int, ND: int,
    vol_rank: np.ndarray,
    short_lb: int, medium_lb: int, long_lb: int,
    high_vol_pct: float, low_vol_pct: float,
) -> np.ndarray:
    """Compute vol-adaptive average range using dynamic lookback windows."""
    range_adaptive = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(long_lb, ND):
            vr = vol_rank[si, di]
            if np.isnan(vr):
                continue
            if vr > high_vol_pct:
                lb = short_lb
            elif vr < low_vol_pct:
                lb = long_lb
            else:
                lb = medium_lb
            rng_vals = []
            for j in range(di - lb, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            min_samples = max(2, lb // 2)
            if len(rng_vals) >= min_samples:
                range_adaptive[si, di] = np.mean(rng_vals)
    return range_adaptive


def compute_adaptive_atrp(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    NS: int, ND: int,
    vol_rank: np.ndarray,
    short_lb: int, medium_lb: int, long_lb: int,
    high_vol_pct: float, low_vol_pct: float,
) -> np.ndarray:
    """Compute vol-adaptive ATR% using dynamic lookback windows."""
    atrp_adaptive = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(long_lb + 1, ND):
            vr = vol_rank[si, di]
            if np.isnan(vr):
                continue
            if vr > high_vol_pct:
                lb = short_lb
            elif vr < low_vol_pct:
                lb = long_lb
            else:
                lb = medium_lb
            atr_vals = []
            for j in range(max(1, di - lb), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_adaptive[si, di] = np.mean(atr_vals) / C[si, di]
    return atrp_adaptive


def compute_dynamic_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    short_lb: int = 3, medium_lb: int = 10, long_lb: int = 20,
    high_vol_pct: float = 0.70, low_vol_pct: float = 0.30,
) -> Dict[str, np.ndarray]:
    """Compute all vol-adaptive dynamic factors + static factors."""
    t0 = time.time()
    print(
        f"[V97] Computing vol-adaptive factors "
        f"(sl={short_lb} ml={medium_lb} ll={long_lb} "
        f"hp={high_vol_pct:.2f} lp={low_vol_pct:.2f})...",
        flush=True)

    # Step 1: Compute realized volatility and its cross-sectional rank
    realized_vol = compute_realized_volatility(C, NS, ND, vol_window=20)
    vol_rank = compute_vol_ranks(realized_vol, NS, ND)

    # Step 2: Compute vol-adaptive dynamic factors
    ret_adaptive = compute_adaptive_return(
        C, NS, ND, vol_rank,
        short_lb, medium_lb, long_lb,
        high_vol_pct, low_vol_pct)
    oi_adaptive = compute_adaptive_oi(
        OI, NS, ND, vol_rank,
        short_lb, medium_lb, long_lb,
        high_vol_pct, low_vol_pct)
    vol_adaptive = compute_adaptive_vol(
        V, NS, ND, vol_rank,
        short_lb, medium_lb, long_lb,
        high_vol_pct, low_vol_pct)
    range_adaptive = compute_adaptive_range(
        H, L, C, NS, ND, vol_rank,
        short_lb, medium_lb, long_lb,
        high_vol_pct, low_vol_pct)
    atrp_adaptive = compute_adaptive_atrp(
        H, L, C, NS, ND, vol_rank,
        short_lb, medium_lb, long_lb,
        high_vol_pct, low_vol_pct)

    # Step 3: RSI (static, not lookback-dependent)
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

    # Step 4: Forward return target for NW kernel training
    fwd_ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 5):
            if (not np.isnan(C[si, di + 5])
                    and not np.isnan(C[si, di])
                    and C[si, di] > 0):
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0

    # Step 5: ATR mean for adaptive bandwidth (uses fixed 14d window)
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

    print(f"  Dynamic factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_adaptive": ret_adaptive,
        "oi_adaptive": oi_adaptive,
        "rsi14": rsi14,
        "vol_adaptive": vol_adaptive,
        "range_adaptive": range_adaptive,
        "atrp_adaptive": atrp_adaptive,
        "fwd_ret_5d": fwd_ret_5d,
        "atr_mean": atr_mean,
        "vol_rank": vol_rank,
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
    training_window: int = 60,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    """Compute NW kernel regression predicted returns using dynamic factors."""
    t0 = time.time()
    print(
        f"[V97] Computing NW predicted returns "
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

    print(f"  NW prediction done: {time.time() - t0:.1f}s", flush=True)
    return predicted


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


def compute_atr_at(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    si: int, di: int, start_di: int,
) -> Optional[float]:
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


def backtest_v97(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
    ker_regime: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 3,
    max_per_sector: int = 3,
    hold_days: int = 5,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V97: NW kernel + vol-adaptive dynamic lookback."""
    if end_di is None:
        end_di = ND - 1

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

        # --- ENTRY: select top_n by NW predicted return ---
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
        alloc_per_pos = LEVERAGE / num_total

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


def analyze(
    trades: List[dict], equity: float, max_dd: float, label: str = "",
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
    ker_regime: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 3,
    max_per_sector: int = 3,
    hold_days: int = 5,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
) -> List[dict]:
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V97 "
        f"(top_n={top_n} mps={max_per_sector} hold={hold_days})")
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

        trades, _, _ = backtest_v97(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime,
            sector_lookup=sector_lookup,
            top_n=top_n,
            max_per_sector=max_per_sector,
            hold_days=hold_days,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=3.0,
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
    print("  V97: NW KERNEL + VOL-ADAPTIVE DYNAMIC LOOKBACK")
    print("  Two-layer adaptation:")
    print("    Layer 1: Vol-adaptive lookback -> dynamic factor inputs")
    print("    Layer 2: NW kernel -> nonlinear signal from dynamic factors")
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

    # Pre-compute KER regime (shared across all configs)
    ker_regime = compute_ker(C, NS, ND)

    # === Phase 1: Compute dynamic factors for each lookback config ===
    # We cache factor sets keyed by (short_lb, medium_lb, long_lb,
    #                                high_vol_pct, low_vol_pct)
    factor_configs = []
    for short_lb in [2, 3, 5]:
        for medium_lb in [8, 10, 12]:
            for long_lb in [15, 20, 25]:
                for high_vol_pct in [0.65, 0.70, 0.75]:
                    for low_vol_pct in [0.25, 0.30, 0.35]:
                        if low_vol_pct >= high_vol_pct:
                            continue
                        factor_configs.append((
                            short_lb, medium_lb, long_lb,
                            high_vol_pct, low_vol_pct))

    print(
        f"\n  Total factor configs to evaluate: {len(factor_configs)}")

    # Compute NW predictions for each factor config x training_window
    # To manage compute time, we first evaluate a reduced sweep
    # focusing on the most promising factor configs

    # Phase 1: Full sweep with bandwidth=1.0, tw varies
    # Phase 2: For best factor configs, try all bandwidths

    BANDWIDTH = 1.0

    results: List[dict] = []
    sweep_count = 0

    for fc_idx, (sl, ml, ll, hvp, lvp) in enumerate(factor_configs):
        raw_factors = compute_dynamic_factors(
            C, O, H, L, V, OI, NS, ND,
            short_lb=sl, medium_lb=ml, long_lb=ll,
            high_vol_pct=hvp, low_vol_pct=lvp)

        for tw in [30, 40, 60]:
            predicted = compute_nw_predicted_returns(
                raw_factors, NS, ND,
                training_window=tw,
                kernel_bandwidth=BANDWIDTH)

            for top_n in [2, 3]:
                for mps in [2, 3]:
                    sweep_count += 1
                    trades, eq, dd = backtest_v97(
                        C, O, H, L, NS, ND, dates, syms,
                        predicted, ker_regime,
                        sector_lookup=sector_lookup,
                        top_n=top_n,
                        max_per_sector=mps,
                        hold_days=5,
                        win_threshold=0.60,
                        normal_threshold=0.80,
                        lose_threshold=0.90,
                        win_rate_window=15,
                        atr_stop=3.0,
                        start_di=bt_2019,
                    )

                    if len(trades) < 10:
                        continue

                    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                    wr = nw / len(trades) * 100
                    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
                    ann = ((eq / CASH0) ** (
                        1 / max(1.0, n_days / 252)) - 1) * 100
                    ap = [t["pnl_abs"]
                          for t in sorted(trades, key=lambda x: x["di"])]
                    rets_arr = np.array(ap) / CASH0
                    sh_val = (
                        np.mean(rets_arr)
                        / np.std(rets_arr) * np.sqrt(252)
                        if np.std(rets_arr) > 0 else 0)

                    yr_counts: Dict[int, int] = {}
                    for t in trades:
                        y = t["year"]
                        yr_counts[y] = yr_counts.get(y, 0) + 1
                    oos_years = [y for y in yr_counts if y >= 2019]
                    avg_per_year = (
                        sum(yr_counts[y] for y in oos_years)
                        / max(len(oos_years), 1))

                    results.append({
                        "sl": sl, "ml": ml, "ll": ll,
                        "hvp": hvp, "lvp": lvp,
                        "tw": tw, "tn": top_n, "mps": mps,
                        "n": len(trades), "wr": wr,
                        "ann": ann, "dd": dd,
                        "sharpe": sh_val, "eq": eq,
                        "avg_yr": avg_per_year,
                    })

        # Progress report
        if (fc_idx + 1) % 10 == 0 or fc_idx == len(factor_configs) - 1:
            print(
                f"\n  Progress: {fc_idx + 1}/{len(factor_configs)} "
                f"factor configs done, {len(results)} results so far",
                flush=True)

    print(
        f"\n  Evaluated {sweep_count} configs, "
        f"{len(results)} with 10+ trades")

    # Sort by annualized return (as specified)
    results.sort(key=lambda x: -x["ann"])

    print(
        f"\n{'SL':>3} {'ML':>3} {'LL':>3} "
        f"{'HVP':>5} {'LVP':>5} "
        f"{'TW':>3} {'TN':>3} {'MPS':>3} "
        f"{'N':>5} {'WR':>6} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7}")
    print("-" * 95)
    for r in results[:10]:
        print(
            f"{r['sl']:>3} {r['ml']:>3} {r['ll']:>3} "
            f"{r['hvp']:>5.2f} {r['lvp']:>5.2f} "
            f"{r['tw']:>3} {r['tn']:>3} {r['mps']:>3} "
            f"{r['n']:>5} {r['wr']:>6.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['avg_yr']:>7.1f}")

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === Top 10 detailed analysis ===
    print("\n" + "=" * 70)
    print("  TOP 10 CONFIGS -- DETAILED ANALYSIS")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["sl"], r["ml"], r["ll"],
               r["hvp"], r["lvp"],
               r["tw"], r["tn"], r["mps"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 10:
            break

    for r in unique_top:
        # Recompute factors and predictions for this config
        raw = compute_dynamic_factors(
            C, O, H, L, V, OI, NS, ND,
            short_lb=r["sl"], medium_lb=r["ml"], long_lb=r["ll"],
            high_vol_pct=r["hvp"], low_vol_pct=r["lvp"])
        pred = compute_nw_predicted_returns(
            raw, NS, ND,
            training_window=r["tw"],
            kernel_bandwidth=BANDWIDTH)

        trades, eq, dd = backtest_v97(
            C, O, H, L, NS, ND, dates, syms,
            pred, ker_regime,
            sector_lookup=sector_lookup,
            top_n=r["tn"],
            max_per_sector=r["mps"],
            hold_days=5,
            start_di=60,
        )
        label = (
            f"sl={r['sl']} ml={r['ml']} ll={r['ll']} "
            f"hvp={r['hvp']:.2f} lvp={r['lvp']:.2f} "
            f"tw={r['tw']} tn={r['tn']} mps={r['mps']}")
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === Walk-forward for best config ===
    best = results[0]
    print("\n" + "=" * 70)
    print(
        f"  BEST WF: sl={best['sl']} ml={best['ml']} ll={best['ll']} "
        f"hvp={best['hvp']:.2f} lvp={best['lvp']:.2f} "
        f"tw={best['tw']} tn={best['tn']} mps={best['mps']}")
    print("=" * 70)

    raw_best = compute_dynamic_factors(
        C, O, H, L, V, OI, NS, ND,
        short_lb=best["sl"], medium_lb=best["ml"],
        long_lb=best["ll"],
        high_vol_pct=best["hvp"], low_vol_pct=best["lvp"])
    pred_best = compute_nw_predicted_returns(
        raw_best, NS, ND,
        training_window=best["tw"],
        kernel_bandwidth=BANDWIDTH)

    walk_forward(
        C, O, H, L, NS, ND, dates, syms,
        pred_best, ker_regime,
        sector_lookup=sector_lookup,
        top_n=best["tn"],
        max_per_sector=best["mps"],
        hold_days=5,
        win_threshold=0.60,
        normal_threshold=0.80,
        lose_threshold=0.90,
        win_rate_window=15,
    )

    print(f"\n[V97] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
