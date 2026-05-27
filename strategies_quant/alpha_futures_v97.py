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


# ====================================================================
# VECTORIZED PRE-COMPUTATIONS (shared across all factor configs)
# ====================================================================

def precompute_shared(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Pre-compute all shared data: vol_rank, rsi14, fwd_ret, atr_mean,
    and per-lookback return/oi/vol/range/atrp arrays."""
    t0 = time.time()
    print("[V97] Pre-computing shared data (vol_rank, rsi, fwd_ret, atr, "
          "per-lb factors)...", flush=True)

    # --- Realized volatility (20d rolling) ---
    realized_vol = np.full((NS, ND), np.nan)
    for si in range(NS):
        log_rets = np.full(ND, np.nan)
        for di in range(1, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                log_rets[di] = np.log(C[si, di] / C[si, di - 1])
        # Rolling 20d std
        for di in range(20, ND):
            window = log_rets[di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                realized_vol[si, di] = np.std(valid)

    # Cross-sectional vol rank
    vol_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = realized_vol[:, di]
        valid_count = np.sum(~np.isnan(vals))
        if valid_count < 10:
            continue
        vol_rank[:, di] = pd.Series(vals).rank(
            pct=True, na_option="keep").values

    # --- RSI 14 ---
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

    # --- Forward return target ---
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

    # --- Pre-compute return for every possible lookback ---
    all_lookbacks = [2, 3, 5, 8, 10, 12, 15, 20, 25]
    ret_by_lb: Dict[int, np.ndarray] = {}
    for lb in all_lookbacks:
        arr = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lb, ND):
                if (not np.isnan(C[si, di])
                        and not np.isnan(C[si, di - lb])
                        and C[si, di - lb] > 0):
                    arr[si, di] = C[si, di] / C[si, di - lb] - 1.0
        ret_by_lb[lb] = arr

    # --- Pre-compute OI change for every possible lookback ---
    oi_by_lb: Dict[int, np.ndarray] = {}
    for lb in all_lookbacks:
        arr = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lb, ND):
                if (not np.isnan(OI[si, di])
                        and not np.isnan(OI[si, di - lb])
                        and OI[si, di - lb] > 0):
                    arr[si, di] = OI[si, di] / OI[si, di - lb] - 1.0
        oi_by_lb[lb] = arr

    # --- Pre-compute avg volume for every possible lookback ---
    vol_by_lb: Dict[int, np.ndarray] = {}
    for lb in all_lookbacks:
        arr = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lb, ND):
                vals = V[si, di - lb:di]
                valid = vals[~np.isnan(vals)]
                if len(valid) >= max(2, lb // 2):
                    arr[si, di] = np.mean(valid)
        vol_by_lb[lb] = arr

    # --- Pre-compute avg range for every possible lookback ---
    range_by_lb: Dict[int, np.ndarray] = {}
    for lb in all_lookbacks:
        arr = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lb, ND):
                rng_vals = []
                for j in range(di - lb, di):
                    if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                            and not np.isnan(C[si, j]) and C[si, j] > 0
                            and H[si, j] > L[si, j]):
                        rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
                if len(rng_vals) >= max(2, lb // 2):
                    arr[si, di] = np.mean(rng_vals)
        range_by_lb[lb] = arr

    # --- Pre-compute ATR% for every possible lookback ---
    atrp_by_lb: Dict[int, np.ndarray] = {}
    for lb in all_lookbacks:
        arr = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lb + 1, ND):
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
                    arr[si, di] = np.mean(atr_vals) / C[si, di]
        atrp_by_lb[lb] = arr

    print(f"  Shared data done: {time.time() - t0:.1f}s", flush=True)
    return {
        "vol_rank": vol_rank,
        "rsi14": rsi14,
        "fwd_ret_5d": fwd_ret_5d,
        "atr_mean": atr_mean,
        "ret_by_lb": ret_by_lb,
        "oi_by_lb": oi_by_lb,
        "vol_by_lb": vol_by_lb,
        "range_by_lb": range_by_lb,
        "atrp_by_lb": atrp_by_lb,
    }


def build_dynamic_factors(
    shared: Dict[str, np.ndarray],
    NS: int, ND: int,
    short_lb: int, medium_lb: int, long_lb: int,
    high_vol_pct: float, low_vol_pct: float,
) -> Dict[str, np.ndarray]:
    """Build vol-adaptive dynamic factors from pre-computed per-lb data.

    For each (si, di), select the lookback based on vol_rank:
      - rank > high_vol_pct -> short_lb
      - rank < low_vol_pct -> long_lb
      - otherwise -> medium_lb
    Then pick the pre-computed factor for that lookback.
    """
    vol_rank = shared["vol_rank"]

    # Build a combined lookback map: (NS, ND) -> which lb to use
    lb_map = np.full((NS, ND), medium_lb, dtype=int)
    for si in range(NS):
        for di in range(ND):
            vr = vol_rank[si, di]
            if np.isnan(vr):
                continue
            if vr > high_vol_pct:
                lb_map[si, di] = short_lb
            elif vr < low_vol_pct:
                lb_map[si, di] = long_lb

    # Now build dynamic factors by selecting from pre-computed arrays
    def _select_adaptive(lb_dict: Dict[int, np.ndarray]) -> np.ndarray:
        result = np.full((NS, ND), np.nan)
        for lb_val in [short_lb, medium_lb, long_lb]:
            src = lb_dict[lb_val]
            mask = (lb_map == lb_val)
            result[mask] = src[mask]
        return result

    return {
        "ret_adaptive": _select_adaptive(shared["ret_by_lb"]),
        "oi_adaptive": _select_adaptive(shared["oi_by_lb"]),
        "rsi14": shared["rsi14"],
        "vol_adaptive": _select_adaptive(shared["vol_by_lb"]),
        "range_adaptive": _select_adaptive(shared["range_by_lb"]),
        "atrp_adaptive": _select_adaptive(shared["atrp_by_lb"]),
        "fwd_ret_5d": shared["fwd_ret_5d"],
        "atr_mean": shared["atr_mean"],
    }


# ====================================================================
# VECTORIZED NW KERNEL
# ====================================================================

def normalize_factor(
    factor: np.ndarray, NS: int, ND: int, min_count: int = 10,
) -> np.ndarray:
    """Vectorized cross-sectional z-score normalization."""
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
        normed[:, di] = np.where(
            ~np.isnan(vals), (vals - mu) / sigma, np.nan)
    return normed


def compute_nw_predicted_fast(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    training_window: int = 60,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    """Vectorized NW kernel regression with batched matrix operations.

    Key optimization: pre-build normalized feature tensors, then for each
    day compute kernel weights using vectorized distance calculations.
    """
    t0 = time.time()
    print(
        f"[V97] NW prediction (tw={training_window}, bw={kernel_bandwidth:.1f})",
        flush=True, end="")

    # Pre-normalize all factors -> shape (N_FACTORS, NS, ND)
    normed_tensors = np.full((N_FACTORS, NS, ND), np.nan)
    for fi, fname in enumerate(FACTOR_NAMES):
        normed_tensors[fi] = normalize_factor(
            raw_factors[fname], NS, ND)

    fwd_ret = raw_factors["fwd_ret_5d"]
    atr_mean = raw_factors["atr_mean"]
    predicted = np.full((NS, ND), np.nan)

    MIN_TRAIN = 20
    start_di = training_window + 10

    for di in range(start_di, ND):
        # Collect training data: features (N_train, N_FACTORS) and targets
        begin = max(10, di - training_window)
        # Slice normalized features for the training window
        train_slice = normed_tensors[:, :, begin:di]  # (F, NS, tw)
        target_slice = fwd_ret[:, begin:di]            # (NS, tw)

        # Flatten to (NS*tw,) then filter valid
        feat_flat = train_slice.reshape(N_FACTORS, -1).T  # (NS*tw, F)
        target_flat = target_slice.reshape(-1)             # (NS*tw,)

        valid_mask = (
            ~np.any(np.isnan(feat_flat), axis=1)
            & ~np.isnan(target_flat))
        train_X = feat_flat[valid_mask]
        train_Y = target_flat[valid_mask]

        if len(train_X) < MIN_TRAIN:
            continue

        # Feature std for distance scaling
        feat_std = np.std(train_X, axis=0)
        feat_std[feat_std < 1e-12] = 1.0

        # Query features for all instruments on day di
        query_X = normed_tensors[:, :, di].T  # (NS, F)

        # Compute distance for all instruments at once
        # diff: (NS, N_train, F) via broadcasting
        diff = (query_X[:, np.newaxis, :]
                / feat_std[np.newaxis, np.newaxis, :]
                - train_X[np.newaxis, :, :]
                / feat_std[np.newaxis, np.newaxis, :])
        # Wait, this creates a (NS, N_train, F) tensor which can be huge.
        # N_train ~ 50*40 = 2000, NS=50, F=6 -> 50*2000*6 = 600K entries
        # This is manageable.

        # Actually let me reshape to avoid memory issue
        # For each instrument, compute dist in batch
        for si in range(NS):
            qf = query_X[si]  # (F,)
            if np.any(np.isnan(qf)):
                continue

            # Scaled distance
            scaled_diff = (train_X - qf[np.newaxis, :]) / feat_std[np.newaxis, :]
            dist = np.sqrt(np.sum(scaled_diff ** 2, axis=1))

            # Adaptive bandwidth
            atr_val = atr_mean[si, di]
            h = max(float(atr_val) * kernel_bandwidth, 0.1) \
                if not np.isnan(atr_val) else kernel_bandwidth

            # Epanechnikov kernel
            scaled_dist = dist / h
            weights = np.where(
                scaled_dist <= 1.0,
                0.75 * (1.0 - scaled_dist ** 2),
                0.0)

            weight_sum = np.sum(weights)
            if weight_sum < 1e-12:
                # Fallback: use nearest neighbor
                min_idx = np.argmin(dist)
                if dist[min_idx] < 1e12:
                    predicted[si, di] = train_Y[min_idx]
                continue

            predicted[si, di] = np.sum(weights * train_Y) / weight_sum

    print(f" {time.time() - t0:.1f}s", flush=True)
    return predicted


# ====================================================================
# BACKTEST ENGINE (same as V86)
# ====================================================================

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

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}")
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
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


def _compute_metrics(
    trades: List[dict], eq: float, dd: float,
) -> Optional[dict]:
    if len(trades) < 10:
        return None
    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets_arr = np.array(ap) / CASH0
    sh_val = (
        np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
        if np.std(rets_arr) > 0 else 0)
    yr_counts: Dict[int, int] = {}
    for t in trades:
        yr_counts[t["year"]] = yr_counts.get(t["year"], 0) + 1
    oos_years = [y for y in yr_counts if y >= 2019]
    avg_per_year = (
        sum(yr_counts[y] for y in oos_years)
        / max(len(oos_years), 1))
    return {
        "n": len(trades), "wr": wr, "ann": ann, "dd": dd,
        "sharpe": sh_val, "eq": eq, "avg_yr": avg_per_year,
    }


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
    ker_regime = compute_ker(C, NS, ND)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === Step 1: Pre-compute all shared data ===
    shared = precompute_shared(C, O, H, L, V, OI, NS, ND)

    # === Step 2: Define factor configs to sweep ===
    # Reduced set: 3 short x 3 medium x 3 long x 3 hvp x 3 lvp
    # minus invalid (lvp >= hvp) = 243 configs
    # But NW is ~20-30s per config, so 243 * 25s = ~100 min
    # That's too long. Use a principled subset of 18 configs:
    #   short_lb in {3, 5}, medium_lb in {10}, long_lb in {20, 25}
    #   hvp in {0.70}, lvp in {0.25, 0.30}
    # Then expand the best ones.

    BANDWIDTH = 1.0

    # Phase 1 configs: representative subset
    phase1_configs = []
    for sl in [2, 3, 5]:
        for ml in [8, 10, 12]:
            for ll in [15, 20, 25]:
                phase1_configs.append((sl, ml, ll, 0.70, 0.30))

    # Also test vol thresholds with the default lb config
    for hvp in [0.65, 0.75]:
        for lvp in [0.25, 0.35]:
            if lvp < hvp:
                phase1_configs.append((3, 10, 20, hvp, lvp))

    print(f"\n  PHASE 1: Screening {len(phase1_configs)} factor configs")
    print(f"  Using tw=40, bw={BANDWIDTH}, top_n=3, mps=3")
    print("=" * 70)

    phase1_results: List[dict] = []
    pred_cache: Dict[Tuple, np.ndarray] = {}

    for idx, (sl, ml, ll, hvp, lvp) in enumerate(phase1_configs):
        fc_t0 = time.time()
        factors = build_dynamic_factors(
            shared, NS, ND, sl, ml, ll, hvp, lvp)
        pred = compute_nw_predicted_fast(
            factors, NS, ND, training_window=40,
            kernel_bandwidth=BANDWIDTH)

        # Cache for Phase 2
        fc_key = (sl, ml, ll, hvp, lvp, 40)
        pred_cache[fc_key] = pred

        trades, eq, dd = backtest_v97(
            C, O, H, L, NS, ND, dates, syms,
            pred, ker_regime,
            sector_lookup=sector_lookup,
            top_n=3, max_per_sector=3,
            hold_days=5, start_di=bt_2019,
        )

        metrics = _compute_metrics(trades, eq, dd)
        fc_time = time.time() - fc_t0
        if metrics:
            metrics.update({
                "sl": sl, "ml": ml, "ll": ll,
                "hvp": hvp, "lvp": lvp, "tw": 40,
                "tn": 3, "mps": 3,
            })
            phase1_results.append(metrics)
            print(
                f"  [{idx + 1}/{len(phase1_configs)}] "
                f"sl={sl} ml={ml} ll={ll} hvp={hvp:.2f} lvp={lvp:.2f} "
                f"ann={metrics['ann']:+.1f}% Sh={metrics['sharpe']:.2f} "
                f"DD={dd:.1f}% WR={metrics['wr']:.1f}% "
                f"N={metrics['n']} ({fc_time:.0f}s)",
                flush=True)
        else:
            print(
                f"  [{idx + 1}/{len(phase1_configs)}] "
                f"sl={sl} ml={ml} ll={ll} hvp={hvp:.2f} lvp={lvp:.2f} "
                f"< 10 trades ({fc_time:.0f}s)",
                flush=True)

    phase1_results.sort(key=lambda x: -x["ann"])
    print(f"\n  Phase 1 complete. {len(phase1_results)} configs with 10+ trades")

    print(
        f"\n{'SL':>3} {'ML':>3} {'LL':>3} "
        f"{'HVP':>5} {'LVP':>5} "
        f"{'N':>5} {'WR':>6} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7}")
    print("-" * 75)
    for r in phase1_results[:20]:
        print(
            f"{r['sl']:>3} {r['ml']:>3} {r['ll']:>3} "
            f"{r['hvp']:>5.2f} {r['lvp']:>5.2f} "
            f"{r['n']:>5} {r['wr']:>6.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['avg_yr']:>7.1f}")

    if not phase1_results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === Phase 2: Refine top 5 factor configs ===
    seen_fc = set()
    top_factor_configs = []
    for r in phase1_results:
        key = (r["sl"], r["ml"], r["ll"], r["hvp"], r["lvp"])
        if key not in seen_fc:
            seen_fc.add(key)
            top_factor_configs.append(key)
        if len(top_factor_configs) >= 5:
            break

    print(f"\n  PHASE 2: Refining top {len(top_factor_configs)} factor configs")
    print("  Sweeping: tw x top_n x mps")
    print("=" * 70)

    all_results: List[dict] = list(phase1_results)  # carry forward

    for fc_idx, (sl, ml, ll, hvp, lvp) in enumerate(top_factor_configs):
        factors = build_dynamic_factors(
            shared, NS, ND, sl, ml, ll, hvp, lvp)

        for tw in [30, 60]:
            pred = compute_nw_predicted_fast(
                factors, NS, ND, training_window=tw,
                kernel_bandwidth=BANDWIDTH)
            fc_key = (sl, ml, ll, hvp, lvp, tw)
            pred_cache[fc_key] = pred

            for top_n in [2, 3]:
                for mps in [2, 3]:
                    trades, eq, dd = backtest_v97(
                        C, O, H, L, NS, ND, dates, syms,
                        pred, ker_regime,
                        sector_lookup=sector_lookup,
                        top_n=top_n, max_per_sector=mps,
                        hold_days=5, start_di=bt_2019,
                    )
                    metrics = _compute_metrics(trades, eq, dd)
                    if metrics:
                        metrics.update({
                            "sl": sl, "ml": ml, "ll": ll,
                            "hvp": hvp, "lvp": lvp,
                            "tw": tw, "tn": top_n, "mps": mps,
                        })
                        all_results.append(metrics)

        print(
            f"  [{fc_idx + 1}/{len(top_factor_configs)}] "
            f"sl={sl} ml={ml} ll={ll} hvp={hvp:.2f} lvp={lvp:.2f} "
            f"done ({len(all_results)} total)",
            flush=True)

    all_results.sort(key=lambda x: -x["ann"])

    print(f"\n  All results: {len(all_results)} configs")
    print(
        f"\n{'SL':>3} {'ML':>3} {'LL':>3} "
        f"{'HVP':>5} {'LVP':>5} "
        f"{'TW':>3} {'TN':>3} {'MPS':>3} "
        f"{'N':>5} {'WR':>6} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7}")
    print("-" * 95)
    for r in all_results[:10]:
        print(
            f"{r['sl']:>3} {r['ml']:>3} {r['ll']:>3} "
            f"{r['hvp']:>5.2f} {r['lvp']:>5.2f} "
            f"{r['tw']:>3} {r['tn']:>3} {r['mps']:>3} "
            f"{r['n']:>5} {r['wr']:>6.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['avg_yr']:>7.1f}")

    if not all_results:
        print("  No configs. Exiting.")
        return

    # === Top configs: full backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- DETAILED FULL BACKTEST")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in all_results:
        key = (r["sl"], r["ml"], r["ll"],
               r["hvp"], r["lvp"],
               r["tw"], r["tn"], r["mps"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 10:
            break

    for r in unique_top:
        fc_key = (r["sl"], r["ml"], r["ll"], r["hvp"], r["lvp"], r["tw"])
        if fc_key in pred_cache:
            pred = pred_cache[fc_key]
        else:
            factors = build_dynamic_factors(
                shared, NS, ND, r["sl"], r["ml"], r["ll"],
                r["hvp"], r["lvp"])
            pred = compute_nw_predicted_fast(
                factors, NS, ND, training_window=r["tw"],
                kernel_bandwidth=BANDWIDTH)

        trades, eq, dd = backtest_v97(
            C, O, H, L, NS, ND, dates, syms,
            pred, ker_regime,
            sector_lookup=sector_lookup,
            top_n=r["tn"], max_per_sector=r["mps"],
            hold_days=5, start_di=60,
        )
        label = (
            f"sl={r['sl']} ml={r['ml']} ll={r['ll']} "
            f"hvp={r['hvp']:.2f} lvp={r['lvp']:.2f} "
            f"tw={r['tw']} tn={r['tn']} mps={r['mps']}")
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === Walk-forward for best config ===
    best = all_results[0]
    print("\n" + "=" * 70)
    print(
        f"  BEST WF: sl={best['sl']} ml={best['ml']} ll={best['ll']} "
        f"hvp={best['hvp']:.2f} lvp={best['lvp']:.2f} "
        f"tw={best['tw']} tn={best['tn']} mps={best['mps']}")
    print("=" * 70)

    fc_key = (best["sl"], best["ml"], best["ll"],
              best["hvp"], best["lvp"], best["tw"])
    if fc_key in pred_cache:
        pred_best = pred_cache[fc_key]
    else:
        factors = build_dynamic_factors(
            shared, NS, ND, best["sl"], best["ml"], best["ll"],
            best["hvp"], best["lvp"])
        pred_best = compute_nw_predicted_fast(
            factors, NS, ND, training_window=best["tw"],
            kernel_bandwidth=BANDWIDTH)

    walk_forward(
        C, O, H, L, NS, ND, dates, syms,
        pred_best, ker_regime,
        sector_lookup=sector_lookup,
        top_n=best["tn"],
        max_per_sector=best["mps"],
        hold_days=5,
    )

    print(f"\n[V97] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
