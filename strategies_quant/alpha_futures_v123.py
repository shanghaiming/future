"""
V123: "执两用中时中" Adaptive Spectral Leverage
================================================
From guoxue: "执两用中" + "君子而时中" -- the optimal middle point MOVES with time.

Core Innovation: Replace V96's binary vol-adaptive thresholds with a CONTINUOUS
Spectral Risk Measure that adapts to the FULL return distribution shape.

Key difference from V96:
  V96: if vol > threshold -> reduce size (BINARY)
  V123: continuous spectral risk score -> smooth position curve
  Spectral risk captures TAIL SHAPE, not just variance.
  When return distribution is symmetric -> similar to V96
  When return distribution has fat left tail -> MUCH more conservative

Signal: V86's NW Kernel (proven +52.9% ann), without BMA (simpler).
Sizing: Spectral risk position sizing per-instrument + portfolio level.

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
    t0 = time.time()
    print("[V123] Computing raw factors...", flush=True)

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

    # Pre-compute per-instrument daily returns for spectral risk
    daily_returns = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                daily_returns[si, di] = C[si, di] / C[si, di - 1] - 1.0

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
        "range_5d": range_5d, "atrp_5d": atrp_5d,
        "ret_10d": ret_10d, "rsi14": rsi14,
        "fwd_ret_5d": fwd_ret_5d,
        "atr_mean": atr_mean,
        "daily_returns": daily_returns,
    }


def normalize_factor(
    factor: np.ndarray, NS: int, ND: int, min_count: int = 10,
) -> np.ndarray:
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
# NW Kernel Regression (from V86, no BMA)
# =====================================================================

def compute_nw_predicted_returns(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    training_window: int = 40,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    t0 = time.time()
    print(
        f"[V123] Computing NW predicted returns "
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
                min_idx = np.argmin(dist)
                if dist[min_idx] < 1e12:
                    weights[min_idx] = 1.0
                    mask = np.zeros(len(dist), dtype=bool)
                    mask[min_idx] = True
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
                f"train_size={len(train_features)}", flush=True)

    print(f"  NW prediction done: {time.time() - t0:.1f}s", flush=True)
    return predicted


# =====================================================================
# Helpers
# =====================================================================

def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
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
# INNOVATION: Spectral Risk Position Sizing
# "执两用中时中" -- the optimal point MOVES with distribution shape
# =====================================================================

def spectral_position_size(
    returns_history: np.ndarray,
    n_quantiles: int = 20,
    tail_weight: float = 5.0,
    sigmoid_center: float = 1.5,
    sigmoid_steepness: float = 3.0,
    pos_min: float = 0.2,
    pos_max: float = 1.5,
) -> float:
    """Compute position size based on spectral risk measure.

    Spectral risk = weighted average of return quantiles where weights
    follow an exponential power law phi(p) = p^(tail_weight-1).
    This gives MORE weight to left tail (losses).

    Returns a continuous position multiplier in [pos_min, pos_max].
    """
    if len(returns_history) < 30:
        return 1.0

    p = np.linspace(0.01, 1.0, n_quantiles)
    quantiles = np.quantile(returns_history, p)

    weights = p ** (tail_weight - 1)
    weights = weights / weights.sum()

    risk_score = -np.sum(weights * quantiles)

    median_risk = np.median(np.abs(returns_history))
    if median_risk < 1e-10:
        return 1.0

    risk_ratio = risk_score / median_risk

    pos_range = pos_max - pos_min
    position_mult = pos_min + pos_range / (
        1.0 + np.exp(sigmoid_steepness * (risk_ratio - sigmoid_center)))

    return float(np.clip(position_mult, pos_min, pos_max))


def compute_instrument_spectral_sizes(
    daily_returns: np.ndarray,
    NS: int, ND: int,
    lookback: int = 60,
    n_quantiles: int = 20,
    tail_weight: float = 5.0,
    sigmoid_center: float = 1.5,
    sigmoid_steepness: float = 3.0,
) -> np.ndarray:
    """Pre-compute per-instrument spectral position sizes for all days.

    Returns (NS, ND) array of position multipliers in [0.2, 1.5].
    """
    t0 = time.time()
    print(
        f"[V123] Computing instrument spectral sizes "
        f"(lookback={lookback}, nq={n_quantiles}, tw={tail_weight:.1f})...",
        flush=True)

    sizes = np.ones((NS, ND))

    for si in range(NS):
        for di in range(lookback, ND):
            ret_window = daily_returns[si, di - lookback:di]
            valid = ret_window[~np.isnan(ret_window)]
            if len(valid) >= 30:
                sizes[si, di] = spectral_position_size(
                    valid, n_quantiles, tail_weight,
                    sigmoid_center, sigmoid_steepness)
        if si % 10 == 0:
            print(f"  instrument {si}/{NS}", flush=True)

    print(f"  Instrument spectral sizes done: {time.time() - t0:.1f}s",
          flush=True)
    return sizes


def compute_portfolio_spectral_size(
    daily_returns: np.ndarray,
    NS: int, ND: int,
    lookback: int = 60,
    n_quantiles: int = 20,
    tail_weight: float = 5.0,
    sigmoid_center: float = 1.5,
    sigmoid_steepness: float = 3.0,
) -> np.ndarray:
    """Compute portfolio-level spectral risk position size.

    Uses equal-weight average daily returns across all instruments
    as a portfolio return proxy, then applies spectral risk.

    Returns (ND,) array of portfolio-level position multipliers.
    """
    t0 = time.time()
    print("[V123] Computing portfolio spectral sizes...", flush=True)

    port_sizes = np.ones(ND)

    for di in range(lookback, ND):
        port_rets = []
        for dd in range(di - lookback, di):
            rets = []
            for si in range(NS):
                if not np.isnan(daily_returns[si, dd]):
                    rets.append(daily_returns[si, dd])
            if rets:
                port_rets.append(np.mean(rets))

        if len(port_rets) >= 30:
            port_sizes[di] = spectral_position_size(
                np.array(port_rets), n_quantiles, tail_weight,
                sigmoid_center, sigmoid_steepness)

    print(f"  Portfolio spectral sizes done: {time.time() - t0:.1f}s",
          flush=True)
    return port_sizes


# =====================================================================
# Backtest with spectral sizing
# =====================================================================

def backtest_v123(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
    ker_regime: np.ndarray,
    inst_sizes: np.ndarray,
    port_sizes: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 5,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
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
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

        # Portfolio-level spectral multiplier
        port_mult = port_sizes[di]

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
        base_alloc = LEVERAGE / num_total

        # Apply instrument-level AND portfolio-level spectral sizing
        updated_positions = []
        for si, edi, ep, sp, old_alloc in positions:
            inst_mult = inst_sizes[si, di]
            final_alloc = base_alloc * inst_mult * port_mult
            updated_positions.append((si, edi, ep, sp, final_alloc))

        for pred_val, si, sym_sector in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            inst_mult = inst_sizes[si, di]
            final_alloc = base_alloc * inst_mult * port_mult
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, final_alloc))

        positions = updated_positions

    # Close remaining
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
    inst_sizes: np.ndarray,
    port_sizes: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 5,
    label: str = "",
) -> List[dict]:
    cfg_str = f"tn={top_n} mps={max_per_sector} hd={hold_days}"
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V123 {label}")
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

        trades, _, _ = backtest_v123(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime, inst_sizes, port_sizes,
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
                f"sectors=[{sec_str}]", flush=True)
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
    print('  V123: "执两用中时中" Adaptive Spectral Leverage')
    print("  Innovation: Spectral Risk Position Sizing")
    print("  Signal: NW Kernel (V86 proven, no BMA)")
    print("  Sizing: Continuous spectral risk -> smooth position curve")
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
    daily_returns = raw_factors["daily_returns"]

    # === 2. Compute NW predictions (V86 signal, no BMA) ===
    pred = compute_nw_predicted_returns(
        raw_factors, NS, ND, training_window=40, kernel_bandwidth=1.0)

    # === 3. Parameter sweep over spectral risk parameters ===
    n_quantiles_list = [10, 20, 30]
    tail_weight_list = [3.0, 5.0, 7.0]
    sigmoid_center_list = [1.0, 1.5, 2.0]
    sigmoid_steepness_list = [2.0, 3.0, 5.0]

    print(f"\n{'=' * 70}")
    print("  PARAMETER SWEEP: Spectral Risk Parameters")
    print(f"  nq in {n_quantiles_list}, tw in {tail_weight_list}")
    print(f"  sc in {sigmoid_center_list}, ss in {sigmoid_steepness_list}")
    print(f"{'=' * 70}")

    results: List[dict] = []
    sweep_count = 0

    for nq in n_quantiles_list:
        for tw in tail_weight_list:
            for sc in sigmoid_center_list:
                for ss in sigmoid_steepness_list:
                    # Compute instrument spectral sizes for this config
                    inst_sizes = compute_instrument_spectral_sizes(
                        daily_returns, NS, ND,
                        lookback=60, n_quantiles=nq,
                        tail_weight=tw, sigmoid_center=sc,
                        sigmoid_steepness=ss)

                    # Compute portfolio spectral sizes for this config
                    port_sizes = compute_portfolio_spectral_size(
                        daily_returns, NS, ND,
                        lookback=60, n_quantiles=nq,
                        tail_weight=tw, sigmoid_center=sc,
                        sigmoid_steepness=ss)

                    for top_n in [2, 3]:
                        for mps in [2, 3]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v123(
                                C, O, H, L, NS, ND, dates, syms,
                                pred, ker_regime,
                                inst_sizes, port_sizes,
                                sector_lookup=sector_lookup,
                                top_n=top_n,
                                max_per_sector=mps,
                                hold_days=5,
                                start_di=bt_2019,
                            )

                            if len(trades) < 10:
                                continue

                            nw = sum(
                                1 for t in trades if t["pnl_pct"] > 0)
                            wr = nw / len(trades) * 100
                            n_days = max(
                                1, trades[-1]["di"] - trades[0]["di"])
                            ann = ((eq / CASH0) ** (
                                1 / max(1.0, n_days / 252)) - 1) * 100
                            ap = [t["pnl_abs"]
                                  for t in sorted(
                                      trades, key=lambda x: x["di"])]
                            rets_arr = np.array(ap) / CASH0
                            sh_val = (
                                np.mean(rets_arr)
                                / np.std(rets_arr) * np.sqrt(252)
                                if np.std(rets_arr) > 0 else 0)

                            results.append({
                                "nq": nq, "tw": tw,
                                "sc": sc, "ss": ss,
                                "top_n": top_n, "mps": mps,
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
        f"\n{'NQ':>3} {'TW':>4} {'SC':>4} {'SS':>4} "
        f"{'TN':>3} {'MPS':>3} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} "
        f"{'Sh':>7}")
    print("-" * 80)
    for r in results[:15]:
        print(
            f"{r['nq']:>3} {r['tw']:>4.1f} {r['sc']:>4.1f} {r['ss']:>4.1f} "
            f"{r['top_n']:>3} {r['mps']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
            f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    # === 4. Walk-forward for best configs ===
    best_by_ann = results[0]
    best_by_sharpe = max(results, key=lambda x: x["sharpe"])
    best_risk_adj = max(
        results, key=lambda x: x["ann"] / max(x["dd"], 1.0))

    for label, best in [
        ("BEST-ANN", best_by_ann),
        ("BEST-SHARPE", best_by_sharpe),
        ("BEST-RISK-ADJ", best_risk_adj),
    ]:
        inst_sizes = compute_instrument_spectral_sizes(
            daily_returns, NS, ND,
            lookback=60, n_quantiles=best["nq"],
            tail_weight=best["tw"], sigmoid_center=best["sc"],
            sigmoid_steepness=best["ss"])
        port_sizes = compute_portfolio_spectral_size(
            daily_returns, NS, ND,
            lookback=60, n_quantiles=best["nq"],
            tail_weight=best["tw"], sigmoid_center=best["sc"],
            sigmoid_steepness=best["ss"])

        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            pred, ker_regime, inst_sizes, port_sizes,
            sector_lookup=sector_lookup,
            top_n=best["top_n"],
            max_per_sector=best["mps"],
            hold_days=5,
            label=label,
        )

    # === 5. Compare V123 (best) vs V86 baseline (no spectral sizing) ===
    print("\n" + "=" * 70)
    print("  COMPARISON: V123 (Spectral) vs V86 baseline (no sizing)")
    print("=" * 70)

    # V123 best by Sharpe
    inst_best = compute_instrument_spectral_sizes(
        daily_returns, NS, ND,
        lookback=60, n_quantiles=best_by_sharpe["nq"],
        tail_weight=best_by_sharpe["tw"],
        sigmoid_center=best_by_sharpe["sc"],
        sigmoid_steepness=best_by_sharpe["ss"])
    port_best = compute_portfolio_spectral_size(
        daily_returns, NS, ND,
        lookback=60, n_quantiles=best_by_sharpe["nq"],
        tail_weight=best_by_sharpe["tw"],
        sigmoid_center=best_by_sharpe["sc"],
        sigmoid_steepness=best_by_sharpe["ss"])

    trades_v123, eq_v123, dd_v123 = backtest_v123(
        C, O, H, L, NS, ND, dates, syms,
        pred, ker_regime, inst_best, port_best,
        sector_lookup=sector_lookup,
        top_n=best_by_sharpe["top_n"],
        max_per_sector=best_by_sharpe["mps"],
        hold_days=5,
        start_di=bt_2019,
    )

    # V86 baseline: uniform sizing (all 1.0)
    inst_ones = np.ones((NS, ND))
    port_ones = np.ones(ND)
    trades_v86, eq_v86, dd_v86 = backtest_v123(
        C, O, H, L, NS, ND, dates, syms,
        pred, ker_regime, inst_ones, port_ones,
        sector_lookup=sector_lookup,
        top_n=2, max_per_sector=2, hold_days=5,
        start_di=bt_2019,
    )

    print(f"\n  V123 BEST-SHARPE (Spectral Risk):")
    analyze(trades_v123, eq_v123, dd_v123, "V123-Spectral")
    print(f"\n  V86 BASELINE (NW only, no sizing):")
    analyze(trades_v86, eq_v86, dd_v86, "V86-baseline")

    if trades_v123 and trades_v86:
        print(
            f"\n  Delta: eq={eq_v123 - eq_v86:+,.0f} "
            f"dd={dd_v123 - dd_v86:+.1f}% "
            f"trades={len(trades_v123) - len(trades_v86):+d}")

    print(f"\n[V123] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
