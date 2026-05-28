"""
V120: "凡事豫则立" Multi-Period Readiness Index Strategy
=========================================================
Innovation: Trade Quality = min(Research, Regime, Risk, Execution)

From guoxue: "凡事豫则立，不豫则废" — preparation brings success,
lack of preparation brings failure.

For each potential trade, score readiness across 4 dimensions:
  P_r = Research readiness  — fraction of factors with |IC| > 0.02
  P_g = Regime readiness    — 1.0 if KER regime supports signal
  P_k = Risk readiness      — 1.0 if portfolio heat < threshold
  P_e = Execution readiness — min(1.0, volume_5d / median_volume)

Overall readiness = min(P_r, P_g, P_k, P_e)
Position size = base_alloc * readiness. Skip if readiness < threshold.

Walk-forward 2019-2026. LEVERAGE=1.0, CASH0=1M, COMM=0.0005.
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
    """Compute 7 raw factors plus volume/ATR for readiness."""
    t0 = time.time()
    print("[V120] Computing raw factors...", flush=True)

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

    # Target: next 5-day forward return
    fwd_ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 5):
            if (not np.isnan(C[si, di + 5])
                    and not np.isnan(C[si, di])
                    and C[si, di] > 0):
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0

    # ATR mean for adaptive bandwidth
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

    # Rolling median volume per instrument (for execution readiness)
    vol_median = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            vals = V[si, di - 60:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 20:
                vol_median[si, di] = np.median(valid)

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
        "range_5d": range_5d, "atrp_5d": atrp_5d,
        "ret_10d": ret_10d, "rsi14": rsi14,
        "fwd_ret_5d": fwd_ret_5d,
        "atr_mean": atr_mean,
        "vol_median": vol_median,
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
# READINESS DIMENSIONS
# =====================================================================

def compute_research_readiness(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    ic_window: int = 20,
    ic_threshold: float = 0.02,
    min_pairs: int = 15,
) -> np.ndarray:
    """P_r: fraction of factors with |IC_window| > ic_threshold.

    Rolling Spearman IC for each factor, then count how many of 7
    factors have abs(IC) > threshold in the rolling window.
    Returns (NS, ND) array of research readiness in [0, 1].
    """
    t0 = time.time()
    print(
        f"[V120] Computing research readiness "
        f"(ic_win={ic_window}, thresh={ic_threshold})...", flush=True)

    fwd_ret = raw_factors["fwd_ret_5d"]
    readiness = np.zeros((NS, ND), dtype=np.float64)

    # Compute per-factor IC for each day (cross-sectional)
    # Then count factors with strong IC
    factor_ic_strong = np.zeros((N_FACTORS, ND), dtype=bool)

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
                mean_ic = np.mean(ic_vals)
                if abs(mean_ic) > ic_threshold:
                    factor_ic_strong[fi, di] = True

        if fi % 2 == 0:
            print(f"  IC for {fname}: {time.time() - t0:.1f}s", flush=True)

    # Research readiness = fraction of strong factors
    for di in range(ND):
        n_strong = np.sum(factor_ic_strong[:, di])
        readiness[:, di] = n_strong / N_FACTORS

    print(f"  Research readiness done: {time.time() - t0:.1f}s", flush=True)
    return readiness


def compute_regime_readiness(
    ker_regime: np.ndarray, NS: int, ND: int,
) -> np.ndarray:
    """P_g: 1.0 if KER regime supports signal (consolidation/trend ok),
    0.5 if neutral, 0.0 if counter-regime.

    ker_regime: 1 = consolidation (good for MR), -1 = trend, 0 = neutral.
    We accept all non-counter-regime states as ready.
    """
    p_g = np.full((NS, ND), 0.5)
    p_g[ker_regime >= 0] = 1.0
    p_g[ker_regime < 0] = 0.5  # still allow trending, just cautious
    return p_g


def compute_risk_readiness(
    C: np.ndarray, NS: int, ND: int,
    heat_threshold: float = 0.08,
    vol_lookback: int = 20,
) -> np.ndarray:
    """P_k: 1.0 if portfolio heat < threshold, scale down if high.

    Portfolio heat = rolling portfolio vol as fraction of equity.
    Uses a simple proxy: rolling cross-sectional return dispersion.
    Returns (ND,) array.
    """
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

    # Scale readiness: 1.0 if vol < threshold, linear decay above
    risk_readiness = np.ones(ND, dtype=np.float64)
    for di in range(ND):
        v = port_vol[di]
        if np.isnan(v) or v <= 0:
            continue
        if v > heat_threshold:
            risk_readiness[di] = max(0.2, heat_threshold / v)
    return risk_readiness


def compute_execution_readiness(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    vol_ratio_min: float = 0.5,
) -> np.ndarray:
    """P_e: min(1.0, volume_5d / median_volume_60d).

    Ensure sufficient liquidity before trading.
    Returns (NS, ND) array.
    """
    vol_5d = raw_factors["vol_5d"]
    vol_median = raw_factors["vol_median"]
    p_e = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(60, ND):
            v5 = vol_5d[si, di]
            vm = vol_median[si, di]
            if np.isnan(v5) or np.isnan(vm) or vm <= 0:
                continue
            ratio = v5 / vm
            p_e[si, di] = min(1.0, max(0.0, ratio))
            # Floor at vol_ratio_min -> 0, linear interpolation
            if ratio < vol_ratio_min:
                p_e[si, di] = ratio / vol_ratio_min
    return p_e


# =====================================================================
# NW Kernel (from V86, no BMA)
# =====================================================================

def compute_nw_predicted_returns(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    training_window: int = 40,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    """V86-style NW kernel regression with equal-weight factors."""
    t0 = time.time()
    print(
        f"[V120] Computing NW predicted returns "
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
# Backtest with Readiness scoring
# =====================================================================

def backtest_v120(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
    ker_regime: np.ndarray,
    p_r: np.ndarray,       # (NS, ND) research readiness
    risk_readiness: np.ndarray,  # (ND,) risk readiness
    p_e: np.ndarray,       # (NS, ND) execution readiness
    sector_lookup: Dict[int, str],
    readiness_min: float = 0.3,
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
    """Backtest V120: NW kernel with readiness-based sizing.

    readiness = min(P_r, P_g, P_k, P_e)
    position_size = base_alloc * readiness
    Skip trade if readiness < readiness_min.
    """
    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float]] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []
    skipped_trades = 0
    total_candidates = 0

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float]] = []

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

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

        # --- ENTRY: select top_n by predicted return + readiness ---
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

            total_candidates += 1

            # Compute readiness for this (si, di)
            pr_val = p_r[si, di] if di < ND else 0.0
            pg_val = 1.0 if ker_regime[si, di] >= 0 else 0.5
            pk_val = risk_readiness[di] if di < ND else 0.5
            pe_val = p_e[si, di] if not np.isnan(p_e[si, di]) else 0.5

            readiness = min(pr_val, pg_val, pk_val, pe_val)

            if readiness < readiness_min:
                skipped_trades += 1
                continue

            candidates.append((pred, si, readiness))

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
        for pred_val, si, readiness in candidates:
            if len(held) + len(new_entries) >= n_to_take:
                break
            if si in held:
                continue
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            if pred_val <= 0:
                continue
            new_entries.append((pred_val, si, sym_sector, readiness))
            sector_counts[sym_sector] += 1

        if not new_entries:
            continue

        num_total = len(positions) + len(new_entries)
        base_alloc = LEVERAGE / num_total

        updated_positions = []
        for si, edi, ep, sp, old_alloc in positions:
            updated_positions.append((si, edi, ep, sp, base_alloc))

        for pred_val, si, sym_sector, readiness in new_entries:
            # KEY INNOVATION: scale position by readiness
            alloc = base_alloc * readiness
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc))

        positions = updated_positions

    # Close remaining positions
    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    # Attach readiness stats to first trade for reporting
    readiness_info = (
        f"readiness_min={readiness_min:.2f} "
        f"skipped={skipped_trades}/{total_candidates} "
        f"pass_rate={1 - skipped_trades/max(total_candidates,1):.1%}"
    )
    if trades:
        trades[0]["readiness_info"] = readiness_info

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

    # Readiness info
    ri = trades[0].get("readiness_info", "")

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}")
    print(
        f"    avg_pnl={avg_pnl:+.3f}% avg_win={avg_win:+.3f}% "
        f"avg_loss={avg_loss:+.3f}% "
        f"modes=[W:{mode_counts['W']} N:{mode_counts['N']} "
        f"L:{mode_counts['L']}]")
    if ri:
        print(f"    readiness: {ri}")
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
    p_r: np.ndarray,
    risk_readiness: np.ndarray,
    p_e: np.ndarray,
    sector_lookup: Dict[int, str],
    readiness_min: float = 0.3,
    top_n: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 5,
    label: str = "",
) -> List[dict]:
    cfg_str = (
        f"tn={top_n} mps={max_per_sector} hd={hold_days} "
        f"rdy_min={readiness_min:.2f}")
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V120 {label}")
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

        trades, _, _ = backtest_v120(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime,
            p_r, risk_readiness, p_e,
            sector_lookup=sector_lookup,
            readiness_min=readiness_min,
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
    print("  V120: 凡事豫则立 - Multi-Period Readiness Index Strategy")
    print("  Innovation: Trade Quality = min(Research,Regime,Risk,Execution)")
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

    # === 2. Compute NW predictions (once, shared) ===
    predicted = compute_nw_predicted_returns(
        raw_factors, NS, ND,
        training_window=40,
        kernel_bandwidth=1.0,
    )

    # === 3. Compute readiness components ===
    # Pre-compute for all ic_window / heat / vol_ratio combos
    ic_window_configs = [15, 20, 30]
    heat_configs = [0.05, 0.08]
    vol_ratio_configs = [0.5, 0.7]

    p_r_cache: Dict[int, np.ndarray] = {}
    for icw in ic_window_configs:
        p_r_cache[icw] = compute_research_readiness(
            raw_factors, NS, ND, ic_window=icw)

    risk_cache: Dict[float, np.ndarray] = {}
    for ht in heat_configs:
        risk_cache[ht] = compute_risk_readiness(
            C, NS, ND, heat_threshold=ht)

    pe_cache: Dict[float, np.ndarray] = {}
    for vrm in vol_ratio_configs:
        pe_cache[vrm] = compute_execution_readiness(
            raw_factors, NS, ND, vol_ratio_min=vrm)

    # === 4. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  凡事豫则立: readiness = min(P_r, P_g, P_k, P_e)")
    print("  NO LEVERAGE.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    readiness_min_configs = [0.2, 0.3, 0.4]

    for icw in ic_window_configs:
        p_r = p_r_cache[icw]
        for ht in heat_configs:
            rk = risk_cache[ht]
            for vrm in vol_ratio_configs:
                pe = pe_cache[vrm]
                for rdy_min in readiness_min_configs:
                    for top_n in [2, 3]:
                        for mps in [2, 3]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v120(
                                C, O, H, L, NS, ND,
                                dates, syms,
                                predicted, ker_regime,
                                p_r, rk, pe,
                                sector_lookup=sector_lookup,
                                readiness_min=rdy_min,
                                top_n=top_n,
                                max_per_sector=mps,
                                hold_days=5,
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
                                "icw": icw, "ht": ht,
                                "vrm": vrm, "rdy_min": rdy_min,
                                "top_n": top_n, "mps": mps,
                                "n": len(trades), "wr": wr,
                                "ann": ann, "dd": dd,
                                "sharpe": sh_val, "eq": eq,
                            })

    # Also run V96 baseline (no readiness) for comparison
    # Use readiness_min=0.0 effectively (everything passes)
    print("\n--- V96 baseline (no readiness filter) ---")
    for top_n in [2, 3]:
        for mps in [2, 3]:
            sweep_count += 1
            # Use the most permissive readiness (all 0.0)
            p_r_loose = np.ones((NS, ND))
            rk_loose = np.ones(ND)
            pe_loose = np.ones((NS, ND))

            trades, eq, dd = backtest_v120(
                C, O, H, L, NS, ND,
                dates, syms,
                predicted, ker_regime,
                p_r_loose, rk_loose, pe_loose,
                sector_lookup=sector_lookup,
                readiness_min=0.0,
                top_n=top_n,
                max_per_sector=mps,
                hold_days=5,
                start_di=bt_2019,
            )

            if len(trades) < 10:
                continue

            nw = sum(1 for t in trades if t["pnl_pct"] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]["di"] - trades[0]["di"])
            ann = ((eq / CASH0) ** (
                1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
            rets_arr = np.array(ap) / CASH0
            sh_val = (
                np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                if np.std(rets_arr) > 0 else 0)

            results.append({
                "icw": 0, "ht": 0, "vrm": 0, "rdy_min": 0.0,
                "top_n": top_n, "mps": mps,
                "n": len(trades), "wr": wr,
                "ann": ann, "dd": dd,
                "sharpe": sh_val, "eq": eq,
            })

    results.sort(key=lambda x: -x["ann"])
    print(
        f"\n  Evaluated {sweep_count} configs, "
        f"{len(results)} with 10+ trades")

    # Report top 15 by annualized return
    print(
        f"\n{'ICw':>4} {'HT':>5} {'VRM':>4} {'Rdy':>4} "
        f"{'TN':>3} {'MPS':>3} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} "
        f"{'Sh':>7}")
    print("-" * 80)
    for r in results[:15]:
        tag = f"{r['icw']}" if r['icw'] > 0 else "base"
        print(
            f"{tag:>4} {r['ht']:>5.2f} {r['vrm']:>4.1f} "
            f"{r['rdy_min']:>4.1f} "
            f"{r['top_n']:>3} {r['mps']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
            f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 5. Walk-forward for top configs ===
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
        if best["icw"] == 0:
            p_r_wf = np.ones((NS, ND))
            rk_wf = np.ones(ND)
            pe_wf = np.ones((NS, ND))
        else:
            p_r_wf = p_r_cache[best["icw"]]
            rk_wf = risk_cache[best["ht"]]
            pe_wf = pe_cache[best["vrm"]]

        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime,
            p_r_wf, rk_wf, pe_wf,
            sector_lookup=sector_lookup,
            readiness_min=best["rdy_min"],
            top_n=best["top_n"],
            max_per_sector=best["mps"],
            hold_days=5,
            label=label,
        )

    # === 6. Compare V120 (best) vs V96 baseline ===
    print("\n" + "=" * 70)
    print("  COMPARISON: V120 (Readiness) vs V96 baseline (no filter)")
    print("  (2019-2026 OOS)")
    print("=" * 70)

    # V120 best
    if best_by_ann["icw"] == 0:
        p_r_best = np.ones((NS, ND))
        rk_best = np.ones(ND)
        pe_best = np.ones((NS, ND))
    else:
        p_r_best = p_r_cache[best_by_ann["icw"]]
        rk_best = risk_cache[best_by_ann["ht"]]
        pe_best = pe_cache[best_by_ann["vrm"]]

    trades_v120, eq_v120, dd_v120 = backtest_v120(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime,
        p_r_best, rk_best, pe_best,
        sector_lookup=sector_lookup,
        readiness_min=best_by_ann["rdy_min"],
        top_n=best_by_ann["top_n"],
        max_per_sector=best_by_ann["mps"],
        hold_days=5,
        start_di=bt_2019,
    )

    # V96 baseline (no readiness)
    p_r_loose = np.ones((NS, ND))
    rk_loose = np.ones(ND)
    pe_loose = np.ones((NS, ND))

    trades_v96, eq_v96, dd_v96 = backtest_v120(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime,
        p_r_loose, rk_loose, pe_loose,
        sector_lookup=sector_lookup,
        readiness_min=0.0,
        top_n=2,
        max_per_sector=2,
        hold_days=5,
        start_di=bt_2019,
    )

    print(f"\n  V120 BEST-ANN (Readiness Filter):")
    analyze(trades_v120, eq_v120, dd_v120, "V120-Readiness")
    print(f"\n  V96 BASELINE (no readiness filter):")
    analyze(trades_v96, eq_v96, dd_v96, "V96-baseline")

    if trades_v120 and trades_v96:
        print(
            f"\n  Delta: eq={eq_v120 - eq_v96:+,.0f} "
            f"dd={dd_v120 - dd_v96:+.1f}% "
            f"trades={len(trades_v120) - len(trades_v96):+d}")

    print(f"\n[V120] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
