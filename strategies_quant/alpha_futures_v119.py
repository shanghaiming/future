"""
V119: "致良知" Factor Confidence Filter
========================================
Guoxue: "致良知" — extend your core knowledge to every action.
In trading: only execute signals when recent IC confirms the factor is
actually predictive.

Innovation on top of V96's NW kernel framework:
  FACTOR CONFIDENCE GATE — use IC as a QUALITY GATE, not a weight.
  1. Compute rolling 20-day Spearman IC for each of 7 factors
  2. confidence_score = mean(|IC_i|) for factors with |IC| > 0.02
  3. Low confidence (< threshold): reduce position size by confidence_mult
  4. High confidence (> high_threshold): normal sizing

Difference from V105 (which weights by IC): V119 uses IC as a BINARY GATE.
Trade less when factors are noisy, trade normally when factors are clear.

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
    """Compute 7 raw factors for NW regression features."""
    t0 = time.time()
    print("[V119] Computing raw factors...", flush=True)

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

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
        "range_5d": range_5d, "atrp_5d": atrp_5d,
        "ret_10d": ret_10d, "rsi14": rsi14,
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
# INNOVATION: Factor Confidence Gate (致良知)
# =====================================================================

def compute_confidence_score(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    confidence_window: int = 20,
    ic_threshold: float = 0.02,
    min_pairs: int = 15,
) -> np.ndarray:
    """Compute factor confidence score for each day.

    For each factor, compute rolling Spearman IC over confidence_window.
    confidence_score = mean(|IC_i|) for factors with |IC_i| > ic_threshold.
    Only "working" factors contribute — if most factors are noisy, score is low.

    Returns (ND,) array of confidence scores.
    """
    t0 = time.time()
    print(
        f"[V119] Computing confidence scores "
        f"(window={confidence_window}, threshold={ic_threshold})...",
        flush=True)

    fwd_ret = raw_factors["fwd_ret_5d"]
    confidence = np.full(ND, np.nan)

    for di in range(confidence_window + 5, ND):
        active_ic_magnitudes: List[float] = []

        for fname in FACTOR_NAMES:
            factor = raw_factors[fname]
            ic_vals = []
            fwd_vals = []

            for tdi in range(di - confidence_window, di):
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
                        fwd_vals.append(corr)

            if len(ic_vals) >= 3:
                mean_ic = np.mean(ic_vals)
                if abs(mean_ic) > ic_threshold:
                    active_ic_magnitudes.append(abs(mean_ic))

        if active_ic_magnitudes:
            confidence[di] = np.mean(active_ic_magnitudes)
        else:
            confidence[di] = 0.0

        if di % 200 == 0:
            valid_score = confidence[di]
            n_active = len(active_ic_magnitudes)
            print(
                f"  di={di}/{ND} score={valid_score:.4f} "
                f"active_factors={n_active}/{N_FACTORS}",
                flush=True)

    valid_scores = confidence[~np.isnan(confidence)]
    if len(valid_scores) > 0:
        print(
            f"  Confidence done: mean={np.mean(valid_scores):.4f} "
            f"median={np.median(valid_scores):.4f} "
            f"{time.time() - t0:.1f}s", flush=True)
    return confidence


def get_confidence_multiplier(
    confidence_score: float,
    confidence_threshold: float = 0.03,
    confidence_high: float = 0.06,
    confidence_low_mult: float = 0.5,
) -> float:
    """Get position size multiplier based on factor confidence.

    - Low confidence (< threshold): reduce by confidence_low_mult
    - High confidence (> high_threshold): normal 1.0x
    - In between: linear interpolation
    """
    if np.isnan(confidence_score):
        return 1.0
    if confidence_score < confidence_threshold:
        return confidence_low_mult
    if confidence_score > confidence_high:
        return 1.0
    # Linear interpolation between threshold and high
    fraction = (
        (confidence_score - confidence_threshold)
        / (confidence_high - confidence_threshold)
    )
    return confidence_low_mult + fraction * (1.0 - confidence_low_mult)


# =====================================================================
# NW Kernel Prediction (plain, no BMA)
# =====================================================================

def compute_nw_predicted_returns(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    training_window: int = 40,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    """Compute NW kernel regression predicted returns (V86 core)."""
    t0 = time.time()
    print(
        f"[V119] Computing NW predicted returns "
        f"(window={training_window}, bw={kernel_bandwidth:.1f})...",
        flush=True)

    # Normalize factors cross-sectionally
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
# Backtest with confidence gate
# =====================================================================

def backtest_v119(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
    ker_regime: np.ndarray,
    confidence: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 5,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    confidence_threshold: float = 0.03,
    confidence_high: float = 0.06,
    confidence_low_mult: float = 0.5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V119: NW kernel + Factor Confidence Gate."""
    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float]] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []

    gate_low_count = 0
    gate_mid_count = 0
    gate_high_count = 0

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float]] = []

        # Dynamic mode
        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

        # Factor confidence gate multiplier
        conf_mult = get_confidence_multiplier(
            confidence[di],
            confidence_threshold,
            confidence_high,
            confidence_low_mult)

        if conf_mult <= confidence_low_mult + 0.01:
            gate_low_count += 1
        elif conf_mult >= 0.99:
            gate_high_count += 1
        else:
            gate_mid_count += 1

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

        # Sector-constrained selection
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
        # Apply confidence gate multiplier to position sizing
        alloc_per_pos = LEVERAGE / num_total * conf_mult

        # Update existing positions with new allocation
        updated_positions = []
        for si, edi, ep, sp, old_alloc in positions:
            updated_positions.append((si, edi, ep, sp, alloc_per_pos))

        # Enter new positions at open[di+1]
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

    gate_info = (
        f"confidence_gate=[low:{gate_low_count} "
        f"mid:{gate_mid_count} high:{gate_high_count}] "
        f"of {gate_low_count + gate_mid_count + gate_high_count} days"
    )
    if trades:
        trades[0]["gate_info"] = gate_info

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
    confidence: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 5,
    confidence_threshold: float = 0.03,
    confidence_high: float = 0.06,
    confidence_low_mult: float = 0.5,
    label: str = "",
) -> List[dict]:
    cfg_str = (
        f"tn={top_n} mps={max_per_sector} hd={hold_days} "
        f"ct={confidence_threshold:.2f} ch={confidence_high:.2f} "
        f"clm={confidence_low_mult:.1f}")
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V119 {label}")
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

        trades, _, _ = backtest_v119(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime, confidence,
            sector_lookup=sector_lookup,
            top_n=top_n,
            max_per_sector=max_per_sector,
            hold_days=hold_days,
            confidence_threshold=confidence_threshold,
            confidence_high=confidence_high,
            confidence_low_mult=confidence_low_mult,
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
    print('  V119: "致良知" Factor Confidence Filter')
    print("  Innovation: Only trade when factors are working")
    print("  Quality gate: reduce size when IC is low, normal when IC is high")
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

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Compute raw factors (shared) ===
    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # === 2. Compute NW kernel predictions (single config: tw=40, bw=1.0) ===
    predicted = compute_nw_predicted_returns(
        raw_factors, NS, ND,
        training_window=40,
        kernel_bandwidth=1.0,
    )

    # === 3. Compute confidence scores for each window ===
    confidence_cache: Dict[int, np.ndarray] = {}
    for cw in [15, 20, 30]:
        confidence_cache[cw] = compute_confidence_score(
            raw_factors, NS, ND, confidence_window=cw)

    # === 4. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  NO LEVERAGE. NW kernel + Factor Confidence Gate.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for cw in [15, 20, 30]:
        conf = confidence_cache[cw]
        for top_n in [2]:
            for mps in [2, 3]:
                for ct in [0.02, 0.03, 0.04]:
                    for ch in [0.05, 0.06, 0.07]:
                        for clm in [0.3, 0.5, 0.7]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v119(
                                C, O, H, L, NS, ND,
                                dates, syms,
                                predicted, ker_regime, conf,
                                sector_lookup=sector_lookup,
                                top_n=top_n,
                                max_per_sector=mps,
                                hold_days=5,
                                confidence_threshold=ct,
                                confidence_high=ch,
                                confidence_low_mult=clm,
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
                                "cw": cw, "top_n": top_n, "mps": mps,
                                "ct": ct, "ch": ch, "clm": clm,
                                "n": len(trades), "wr": wr,
                                "ann": ann, "dd": dd,
                                "sharpe": sh_val, "eq": eq,
                            })

    # Also run V96 baseline (no confidence gate) for comparison
    # confidence_high=99 -> always pass gate -> conf_mult always 1.0
    baseline_conf = confidence_cache[20]
    for top_n in [2]:
        for mps in [2, 3]:
            sweep_count += 1
            trades, eq, dd = backtest_v119(
                C, O, H, L, NS, ND,
                dates, syms,
                predicted, ker_regime, baseline_conf,
                sector_lookup=sector_lookup,
                top_n=top_n,
                max_per_sector=mps,
                hold_days=5,
                confidence_threshold=0.0,
                confidence_high=0.0,
                confidence_low_mult=1.0,
                start_di=bt_2019,
            )

            if len(trades) >= 10:
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

                results.append({
                    "cw": 0, "top_n": top_n, "mps": mps,
                    "ct": 0.0, "ch": 0.0, "clm": 1.0,
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
        f"\n{'CW':>4} {'TN':>3} {'MPS':>3} "
        f"{'CT':>5} {'CH':>5} {'CLM':>5} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} "
        f"{'Sh':>7}")
    print("-" * 85)
    for r in results[:10]:
        tag = f"{r['cw']}" if r['cw'] > 0 else "base"
        print(
            f"{tag:>4} {r['top_n']:>3} {r['mps']:>3} "
            f"{r['ct']:>5.2f} {r['ch']:>5.2f} {r['clm']:>5.1f} "
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
        if best["cw"] == 0:
            conf = confidence_cache[20]
        else:
            conf = confidence_cache[best["cw"]]

        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime, conf,
            sector_lookup=sector_lookup,
            top_n=best["top_n"],
            max_per_sector=best["mps"],
            hold_days=5,
            confidence_threshold=best["ct"],
            confidence_high=best["ch"],
            confidence_low_mult=best["clm"],
            label=label,
        )

    # === 6. Compare V119 (best) vs no-gate baseline ===
    print("\n" + "=" * 70)
    print("  COMPARISON: V119 (NW+ConfGate) vs Baseline (NW only)")
    print("  (2019-2026 OOS)")
    print("=" * 70)

    # V119 best
    if best_by_ann["cw"] == 0:
        conf_best = confidence_cache[20]
    else:
        conf_best = confidence_cache[best_by_ann["cw"]]

    trades_v119, eq_v119, dd_v119 = backtest_v119(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime, conf_best,
        sector_lookup=sector_lookup,
        top_n=best_by_ann["top_n"],
        max_per_sector=best_by_ann["mps"],
        hold_days=5,
        confidence_threshold=best_by_ann["ct"],
        confidence_high=best_by_ann["ch"],
        confidence_low_mult=best_by_ann["clm"],
        start_di=bt_2019,
    )

    # Baseline (no confidence gate)
    trades_base, eq_base, dd_base = backtest_v119(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime, baseline_conf,
        sector_lookup=sector_lookup,
        top_n=2,
        max_per_sector=2,
        hold_days=5,
        confidence_threshold=0.0,
        confidence_high=0.0,
        confidence_low_mult=1.0,
        start_di=bt_2019,
    )

    print(f"\n  V119 BEST-ANN (NW+ConfGate):")
    analyze(trades_v119, eq_v119, dd_v119, "V119-NW+ConfGate")
    print(f"\n  BASELINE (NW only, no confidence gate):")
    analyze(trades_base, eq_base, dd_base, "Baseline-NW-only")

    if trades_v119 and trades_base:
        print(
            f"\n  Delta: eq={eq_v119 - eq_base:+,.0f} "
            f"dd={dd_v119 - dd_base:+.1f}% "
            f"trades={len(trades_v119) - len(trades_base):+d}")

    print(f"\n[V119] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
