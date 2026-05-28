"""
V118: Upgraded Factor Set — MFI + ADX Replace Redundant Factors
================================================================
Factor engineering analysis showed:
- vol_5d → MFI(14) captures MORE info (volume-weighted RSI)
- range_5d → ADX(14) measures trend existence, orthogonal to direction
- ret_10d → Keep (let NW kernel handle redundancy)

Hypothesis: BETTER factor quality (not more factors) improves performance.

9-Factor Set:
  ret_5d, oi_5d, rsi14, vol_5d, ret_10d, range_5d, atrp_5d,
  mfi_14 (NEW), adx_14 (NEW)

Architecture: V96 base. NW kernel (no BMA) + vol-adaptive sizing.
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
    "mfi_14", "adx_14",
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


def compute_mfi_manual(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    V: np.ndarray, NS: int, ND: int, period: int = 14,
) -> np.ndarray:
    """Manual MFI computation.

    MFI = 100 - 100 / (1 + MFR)
    MFR = sum(pos_money_flow) / sum(neg_money_flow) over period
    Money Flow = Volume * Typical Price
    Typical Price = (H + L + C) / 3
    """
    mfi = np.full((NS, ND), np.nan)
    for si in range(NS):
        tp = np.full(ND, np.nan)
        mf = np.full(ND, np.nan)
        for di in range(ND):
            if (not np.isnan(H[si, di]) and not np.isnan(L[si, di])
                    and not np.isnan(C[si, di])
                    and not np.isnan(V[si, di])):
                tp[di] = (H[si, di] + L[si, di] + C[si, di]) / 3.0
                mf[di] = tp[di] * V[si, di]

        for di in range(period, ND):
            pos_mf = 0.0
            neg_mf = 0.0
            valid_count = 0
            for j in range(di - period + 1, di + 1):
                if np.isnan(tp[j]) or np.isnan(tp[j - 1]):
                    continue
                if np.isnan(mf[j]):
                    continue
                valid_count += 1
                if tp[j] > tp[j - 1]:
                    pos_mf += mf[j]
                else:
                    neg_mf += mf[j]
            if valid_count < period // 2:
                continue
            if neg_mf < 1e-12:
                mfi[si, di] = 100.0
            else:
                mfr = pos_mf / neg_mf
                mfi[si, di] = 100.0 - 100.0 / (1.0 + mfr)
    return mfi


def compute_mfi_talib(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    V: np.ndarray, NS: int, ND: int, period: int = 14,
) -> np.ndarray:
    """MFI using talib."""
    mfi = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
        l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
        c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
        v = np.where(np.isnan(V[si]), 0, V[si]).astype(np.float64)
        nan_mask = np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(C[si]) | np.isnan(V[si])
        try:
            r = talib.MFI(h, l, c, v, timeperiod=period)
            mfi[si] = np.where(nan_mask, np.nan, r)
        except Exception:
            pass
    return mfi


def compute_adx_manual(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    NS: int, ND: int, period: int = 14,
) -> np.ndarray:
    """Manual ADX computation.

    ADX measures trend strength regardless of direction.
    """
    adx = np.full((NS, ND), np.nan)
    for si in range(NS):
        # True Range
        tr = np.full(ND, np.nan)
        plus_dm = np.full(ND, np.nan)
        minus_dm = np.full(ND, np.nan)

        for di in range(1, ND):
            hh, ll, cc = H[si, di], L[si, di], C[si, di]
            prev_h = H[si, di - 1]
            prev_l = L[si, di - 1]
            prev_c = C[si, di - 1]
            if any(np.isnan([hh, ll, cc, prev_h, prev_l, prev_c])):
                continue
            tr[di] = max(
                hh - ll, abs(hh - prev_c), abs(ll - prev_c))
            up_move = hh - prev_h
            down_move = prev_l - ll
            if up_move > down_move and up_move > 0:
                plus_dm[di] = up_move
            else:
                plus_dm[di] = 0.0
            if down_move > up_move and down_move > 0:
                minus_dm[di] = down_move
            else:
                minus_dm[di] = 0.0

        # Smooth with Wilder's method
        atr_s = np.full(ND, np.nan)
        plus_di_s = np.full(ND, np.nan)
        minus_di_s = np.full(ND, np.nan)

        # Initial sums
        init_tr = 0.0
        init_pdm = 0.0
        init_mdm = 0.0
        init_count = 0
        for di in range(1, ND):
            if np.isnan(tr[di]):
                continue
            init_count += 1
            init_tr += tr[di]
            init_pdm += plus_dm[di] if not np.isnan(plus_dm[di]) else 0.0
            init_mdm += minus_dm[di] if not np.isnan(minus_dm[di]) else 0.0
            if init_count == period:
                if init_tr > 0:
                    atr_s[di] = init_tr
                    plus_di_s[di] = 100.0 * init_pdm / init_tr
                    minus_di_s[di] = 100.0 * init_mdm / init_tr
                break

        # Wilder's smoothing
        for di in range(1, ND):
            if not np.isnan(atr_s[di]):
                # Already initialized
                if di + 1 < ND and not np.isnan(tr[di + 1]):
                    atr_s[di + 1] = atr_s[di] - atr_s[di] / period + tr[di + 1]
                    pdm_val = plus_dm[di + 1] if not np.isnan(plus_dm[di + 1]) else 0.0
                    mdm_val = minus_dm[di + 1] if not np.isnan(minus_dm[di + 1]) else 0.0
                    prev_pdi = plus_di_s[di] if not np.isnan(plus_di_s[di]) else 0.0
                    prev_mdi = minus_di_s[di] if not np.isnan(minus_di_s[di]) else 0.0
                    if atr_s[di + 1] > 0:
                        plus_di_s[di + 1] = (
                            prev_pdi * (period - 1) / period
                            + 100.0 * pdm_val / atr_s[di + 1])
                        minus_di_s[di + 1] = (
                            prev_mdi * (period - 1) / period
                            + 100.0 * mdm_val / atr_s[di + 1])

        # DX and ADX
        dx = np.full(ND, np.nan)
        for di in range(ND):
            pdi = plus_di_s[di]
            mdi = minus_di_s[di]
            if np.isnan(pdi) or np.isnan(mdi):
                continue
            di_sum = pdi + mdi
            if di_sum > 0:
                dx[di] = 100.0 * abs(pdi - mdi) / di_sum

        # ADX = smoothed DX
        first_dx_idx = None
        dx_sum = 0.0
        dx_count = 0
        for di in range(ND):
            if np.isnan(dx[di]):
                continue
            dx_count += 1
            dx_sum += dx[di]
            if dx_count == period:
                adx[si, di] = dx_sum / period
                first_dx_idx = di
                break

        if first_dx_idx is not None:
            for di in range(first_dx_idx + 1, ND):
                if np.isnan(dx[di]) or np.isnan(adx[si, di - 1]):
                    continue
                adx[si, di] = (adx[si, di - 1] * (period - 1) + dx[di]) / period

    return adx


def compute_adx_talib(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    NS: int, ND: int, period: int = 14,
) -> np.ndarray:
    """ADX using talib."""
    adx = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
        l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
        c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
        nan_mask = np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(C[si])
        try:
            r = talib.ADX(h, l, c, timeperiod=period)
            adx[si] = np.where(nan_mask, np.nan, r)
        except Exception:
            pass
    return adx


def compute_raw_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute 9 raw factors including MFI(14) and ADX(14)."""
    t0 = time.time()
    print("[V118] Computing raw factors (9 total)...", flush=True)

    # --- Original 7 factors ---
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

    # RSI14
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

    print(f"  Original 7 factors done: {time.time() - t0:.1f}s", flush=True)

    # --- NEW Factor 8: MFI(14) ---
    t_mfi = time.time()
    mfi_14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        mfi_14 = compute_mfi_talib(H, L, C, V, NS, ND, 14)

    needs_fallback_mfi = np.all(np.isnan(mfi_14), axis=1)
    if needs_fallback_mfi.any():
        mfi_manual = compute_mfi_manual(H, L, C, V, NS, ND, 14)
        for si in range(NS):
            if needs_fallback_mfi[si]:
                mfi_14[si] = mfi_manual[si]
    mfi_valid = np.sum(~np.isnan(mfi_14))
    print(
        f"  MFI(14) done: {time.time() - t_mfi:.1f}s "
        f"({mfi_valid} valid cells)", flush=True)

    # --- NEW Factor 9: ADX(14) ---
    t_adx = time.time()
    adx_14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        adx_14 = compute_adx_talib(H, L, C, NS, ND, 14)

    needs_fallback_adx = np.all(np.isnan(adx_14), axis=1)
    if needs_fallback_adx.any():
        adx_manual = compute_adx_manual(H, L, C, NS, ND, 14)
        for si in range(NS):
            if needs_fallback_adx[si]:
                adx_14[si] = adx_manual[si]
    adx_valid = np.sum(~np.isnan(adx_14))
    print(
        f"  ADX(14) done: {time.time() - t_adx:.1f}s "
        f"({adx_valid} valid cells)", flush=True)

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

    print(f"  All 9 factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
        "range_5d": range_5d, "atrp_5d": atrp_5d,
        "ret_10d": ret_10d, "rsi14": rsi14,
        "mfi_14": mfi_14, "adx_14": adx_14,
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
    """Compute NW kernel regression with 9 factors (no BMA)."""
    t0 = time.time()
    print(
        f"[V118] Computing NW predicted returns "
        f"(9 factors, window={training_window}, bw={kernel_bandwidth:.1f})...",
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
        # Collect training data
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


def compute_portfolio_volatility(
    C: np.ndarray, NS: int, ND: int, vol_lookback: int = 20,
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


def backtest_v118(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
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
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V118: NW kernel with 9 factors + vol-adaptive sizing."""
    if end_di is None:
        end_di = ND - 1

    vol_data = port_vol[max(start_di, 21):end_di]
    vol_data_valid = vol_data[~np.isnan(vol_data)]
    vol_median = np.median(vol_data_valid) if len(vol_data_valid) > 10 else 1e-6

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
        alloc_per_pos = LEVERAGE / num_total * vol_mult

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
    sector_lookup: Dict[int, str],
    top_n: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 5,
    vol_high_mult: float = 2.0,
    vol_low_mult: float = 0.5,
    size_reduce: float = 0.5,
    size_boost: float = 1.3,
    label: str = "",
) -> List[dict]:
    cfg_str = (
        f"tn={top_n} mps={max_per_sector} hd={hold_days} "
        f"vhm={vol_high_mult:.1f} vlm={vol_low_mult:.1f} "
        f"sr={size_reduce:.1f} sb={size_boost:.1f}")
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V118 {label}")
    print(f"  {cfg_str}")
    print(f"  9 FACTORS: ret_5d oi_5d rsi14 vol_5d ret_10d range_5d atrp_5d MFI ADX")
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

        trades, _, _ = backtest_v118(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime, port_vol,
            sector_lookup=sector_lookup,
            top_n=top_n,
            max_per_sector=max_per_sector,
            hold_days=hold_days,
            vol_high_mult=vol_high_mult,
            vol_low_mult=vol_low_mult,
            size_reduce=size_reduce,
            size_boost=size_boost,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
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
    print("  V118: UPGRADED FACTOR SET (MFI + ADX)")
    print("  Innovation: Better factors > More factors")
    print("  9 factors: 7 original + MFI(14) + ADX(14)")
    print("  NW kernel (no BMA) + vol-adaptive sizing")
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

    # === 1. Compute raw factors (9 total) ===
    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # === 2. Compute NW predictions for each (window, bandwidth) ===
    pred_cache: Dict[Tuple[int, float], np.ndarray] = {}
    for tw in [30, 40, 50]:
        for kb in [0.8, 1.0, 1.5]:
            key = (tw, kb)
            print(f"\n--- NW predictions (tw={tw}, kb={kb:.1f}) ---")
            pred_cache[key] = compute_nw_predicted_returns(
                raw_factors, NS, ND,
                training_window=tw,
                kernel_bandwidth=kb,
            )

    # === 3. Compute portfolio volatility ===
    port_vol = compute_portfolio_volatility(C, NS, ND, 20)

    # === 4. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  9 factors: 7 original + MFI(14) + ADX(14)")
    print("  NW kernel + vol-adaptive sizing. NO BMA.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for pred_key in pred_cache.keys():
        tw, kb = pred_key
        pred = pred_cache[pred_key]
        for top_n in [2, 3]:
            for mps in [2, 3]:
                for vhm in [1.5, 2.0]:
                    for vlm in [0.5, 0.7]:
                        for sr in [0.3, 0.5]:
                            for sb in [1.2, 1.5]:
                                sweep_count += 1
                                trades, eq, dd = backtest_v118(
                                    C, O, H, L, NS, ND,
                                    dates, syms,
                                    pred, ker_regime, port_vol,
                                    sector_lookup=sector_lookup,
                                    top_n=top_n,
                                    max_per_sector=mps,
                                    hold_days=5,
                                    vol_high_mult=vhm,
                                    vol_low_mult=vlm,
                                    size_reduce=sr,
                                    size_boost=sb,
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
                                    "tw": tw, "kb": kb,
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

    # Report top 15 by annualized return
    print(
        f"\n{'TW':>3} {'KB':>4} {'TN':>3} {'MPS':>3} "
        f"{'Vhm':>4} {'Vlm':>4} {'SR':>4} {'SB':>4} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} "
        f"{'Sh':>7}")
    print("-" * 90)
    for r in results[:15]:
        print(
            f"{r['tw']:>3} {r['kb']:>4.1f} {r['top_n']:>3} {r['mps']:>3} "
            f"{r['vhm']:>4.1f} {r['vlm']:>4.1f} "
            f"{r['sr']:>4.1f} {r['sb']:>4.1f} "
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
        pred = pred_cache[(best["tw"], best["kb"])]
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            pred, ker_regime, port_vol,
            sector_lookup=sector_lookup,
            top_n=best["top_n"],
            max_per_sector=best["mps"],
            hold_days=5,
            vol_high_mult=best["vhm"],
            vol_low_mult=best["vlm"],
            size_reduce=best["sr"],
            size_boost=best["sb"],
            label=label,
        )

    # === 6. Final summary ===
    print("\n" + "=" * 70)
    print("  V118 SUMMARY: Upgraded Factor Set (MFI + ADX)")
    print("  Hypothesis: BETTER factors > MORE factors")
    print(f"  V96 baseline: +73.1% ann (7 factors)")
    print("=" * 70)

    for label, best in [
        ("BEST-ANN", best_by_ann),
        ("BEST-SHARPE", best_by_sharpe),
    ]:
        pred = pred_cache[(best["tw"], best["kb"])]
        trades, eq, dd = backtest_v118(
            C, O, H, L, NS, ND, dates, syms,
            pred, ker_regime, port_vol,
            sector_lookup=sector_lookup,
            top_n=best["top_n"],
            max_per_sector=best["mps"],
            hold_days=5,
            vol_high_mult=best["vhm"],
            vol_low_mult=best["vlm"],
            size_reduce=best["sr"],
            size_boost=best["sb"],
            start_di=bt_2019,
        )
        analyze(trades, eq, dd, f"V118-{label}")

    print(f"\n[V118] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
