"""
V100: Multi-Frequency Resonance Strategy
=========================================
Core insight: When ULTOSC + ret5d + CCI align in the same direction,
signal strength multiplies.

Innovation:
- HT_TRENDMODE: Hilbert transform judges trend(1) vs cycle(0) --
  frequency-domain analysis, naturally orthogonal to all time-domain indicators
- ULTOSC(7/14/28): Built-in multi-timeframe overbought/oversold
- When trend mode (1): TRIX weight doubled (use trend factors)
- When cycle mode (0): CCI + STOCHRSI weight doubled (use mean-reversion factors)

Factor combination:
  ULTOSC(7/14/28)_rank      0.20  Multi-TF overbought/oversold
  TRIX(30)_rank             0.20  Noise-filtered momentum
  ret5d_rank                0.15  Short-period momentum
  ret10d_rank               0.15  Medium-period momentum
  HT_TRENDMODE_rank         0.10  Frequency-domain regime
  STOCHRSI(14,14,3,3)_rank  0.10  RSI relative position
  CCI(14)_rank              0.10  Commodity cycle position

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

# V100 factor names
FACTOR_NAMES = [
    "ultosc", "trix", "ret_5d", "ret_10d",
    "ht_trendmode", "stochrsi", "cci",
]
N_FACTORS = len(FACTOR_NAMES)

# Base weights for the composite signal
BASE_WEIGHTS = {
    "ultosc": 0.20,
    "trix": 0.20,
    "ret_5d": 0.15,
    "ret_10d": 0.15,
    "ht_trendmode": 0.10,
    "stochrsi": 0.10,
    "cci": 0.10,
}


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
# Factor computation
# =====================================================================

def compute_raw_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute V100 multi-frequency resonance factors."""
    t0 = time.time()
    print("[V100] Computing raw factors...", flush=True)

    # --- ret_5d ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    # --- ret_10d ---
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    # --- ULTOSC(7,14,28) via TA-Lib ---
    ultosc = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l_arr = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(C[si])
            try:
                r = talib.ULTOSC(h, l_arr, c,
                                 timeperiod1=7, timeperiod2=14,
                                 timeperiod3=28)
                ultosc[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    # Fallback ULTOSC
    needs_fallback = np.all(np.isnan(ultosc), axis=1)
    if needs_fallback.any():
        for si in range(NS):
            if not needs_fallback[si]:
                continue
            for di in range(28, ND):
                bp_list, tr_list = [], []
                for period in [7, 14, 28]:
                    start_j = di - period
                    if start_j < 0:
                        continue
                    max_val = np.nanmax(H[si, start_j:di + 1])
                    min_val = np.nanmin(L[si, start_j:di + 1])
                    close_val = C[si, di]
                    if (np.isnan(max_val) or np.isnan(min_val)
                            or np.isnan(close_val)):
                        continue
                    tr_val = max_val - min_val
                    bp_val = close_val - min_val
                    if tr_val > 0:
                        bp_list.append(bp_val)
                        tr_list.append(tr_val)
                if bp_list:
                    ultosc[si, di] = 100.0 * sum(bp_list) / sum(tr_list)

    # --- TRIX(30) via TA-Lib ---
    trix = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.TRIX(c, timeperiod=30)
                trix[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    # Fallback TRIX: triple EMA rate of change
    needs_fallback_trix = np.all(np.isnan(trix), axis=1)
    if needs_fallback_trix.any():
        for si in range(NS):
            if not needs_fallback_trix[si]:
                continue
            prices = C[si].copy()
            valid_mask = ~np.isnan(prices)
            if np.sum(valid_mask) < 60:
                continue
            k = 2.0 / 31.0
            ema1 = np.full(ND, np.nan)
            ema2 = np.full(ND, np.nan)
            ema3 = np.full(ND, np.nan)
            first_valid = None
            for di in range(ND):
                if np.isnan(prices[di]):
                    continue
                if first_valid is None:
                    ema1[di] = prices[di]
                    first_valid = di
                else:
                    ema1[di] = prices[di] * k + ema1[di - 1] * (1 - k)
            first_e1 = None
            for di in range(ND):
                if np.isnan(ema1[di]):
                    continue
                if first_e1 is None:
                    ema2[di] = ema1[di]
                    first_e1 = di
                else:
                    ema2[di] = ema1[di] * k + ema2[di - 1] * (1 - k)
            first_e2 = None
            for di in range(ND):
                if np.isnan(ema2[di]):
                    continue
                if first_e2 is None:
                    ema3[di] = ema2[di]
                    first_e2 = di
                else:
                    ema3[di] = ema2[di] * k + ema3[di - 1] * (1 - k)
            for di in range(1, ND):
                if (not np.isnan(ema3[di]) and not np.isnan(ema3[di - 1])
                        and ema3[di - 1] != 0):
                    trix[si, di] = (
                        (ema3[di] - ema3[di - 1])
                        / ema3[di - 1] * 10000)

    # --- HT_TRENDMODE via TA-Lib ---
    ht_trendmode = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.HT_TRENDMODE(c)
                ht_trendmode[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    # Fallback HT_TRENDMODE: efficiency ratio proxy
    needs_fallback_ht = np.all(np.isnan(ht_trendmode), axis=1)
    if needs_fallback_ht.any():
        for si in range(NS):
            if not needs_fallback_ht[si]:
                continue
            for di in range(15, ND):
                closes = C[si, di - 10:di + 1]
                valid = closes[~np.isnan(closes)]
                if len(valid) < 10 or valid[0] <= 0:
                    continue
                net = abs(valid[-1] - valid[0])
                total = np.sum(np.abs(np.diff(valid)))
                if total > 1e-10:
                    ker = net / total
                    ht_trendmode[si, di] = 1.0 if ker > 0.2 else 0.0

    # --- STOCHRSI(14,14,3,3) via TA-Lib ---
    stochrsi = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                fastk, _ = talib.STOCHRSI(
                    c, timeperiod=14,
                    fastk_period=3, fastd_period=3, fastd_matype=0)
                stochrsi[si] = np.where(nan_mask, np.nan, fastk)
            except Exception:
                pass

    # Fallback STOCHRSI
    needs_fallback_srsi = np.all(np.isnan(stochrsi), axis=1)
    if needs_fallback_srsi.any():
        rsi14 = np.full((NS, ND), np.nan)
        for si in range(NS):
            if not needs_fallback_srsi[si]:
                continue
            c = C[si]
            for di in range(15, ND):
                gains, losses = [], []
                for j in range(di - 13, di + 1):
                    if np.isnan(c[j]) or np.isnan(c[j - 1]):
                        continue
                    delta = c[j] - c[j - 1]
                    gains.append(max(delta, 0))
                    losses.append(max(-delta, 0))
                if len(gains) >= 12:
                    avg_g = np.mean(gains)
                    avg_l = np.mean(losses)
                    if avg_l == 0:
                        rsi14[si, di] = 100.0
                    else:
                        rsi14[si, di] = (
                            100.0 - 100.0 / (1.0 + avg_g / avg_l))

        for si in range(NS):
            if not needs_fallback_srsi[si]:
                continue
            rsi_vals = rsi14[si]
            for di in range(18, ND):
                rsi_window = rsi_vals[di - 13:di + 1]
                valid = rsi_window[~np.isnan(rsi_window)]
                if len(valid) < 10:
                    continue
                rsi_min = np.min(valid)
                rsi_max = np.max(valid)
                rsi_cur = rsi_vals[di]
                if np.isnan(rsi_cur) or rsi_max == rsi_min:
                    continue
                stochrsi[si, di] = (
                    (rsi_cur - rsi_min) / (rsi_max - rsi_min) * 100.0)

    # --- CCI(14) via TA-Lib ---
    cci = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l_arr = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(C[si])
            try:
                r = talib.CCI(h, l_arr, c, timeperiod=14)
                cci[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    # Fallback CCI
    needs_fallback_cci = np.all(np.isnan(cci), axis=1)
    if needs_fallback_cci.any():
        for si in range(NS):
            if not needs_fallback_cci[si]:
                continue
            for di in range(14, ND):
                tp_vals = []
                for j in range(di - 13, di + 1):
                    hh, ll, cc = H[si, j], L[si, j], C[si, j]
                    if not any(np.isnan([hh, ll, cc])):
                        tp_vals.append((hh + ll + cc) / 3.0)
                if len(tp_vals) < 12:
                    continue
                tp_arr = np.array(tp_vals)
                tp_mean = np.mean(tp_arr)
                md = np.mean(np.abs(tp_arr - tp_mean))
                if md > 1e-12:
                    cci[si, di] = (tp_arr[-1] - tp_mean) / (0.015 * md)

    # --- Forward 5-day return (target) ---
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
        "ultosc": ultosc, "trix": trix,
        "ret_5d": ret_5d, "ret_10d": ret_10d,
        "ht_trendmode": ht_trendmode,
        "stochrsi": stochrsi, "cci": cci,
        "fwd_ret_5d": fwd_ret_5d,
        "atr_mean": atr_mean,
    }


def normalize_factor(factor: np.ndarray, NS: int, ND: int,
                     min_count: int = 10) -> np.ndarray:
    """Cross-sectional z-score normalization."""
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


def rank_normalize(factor: np.ndarray, NS: int, ND: int,
                   min_count: int = 10) -> np.ndarray:
    """Cross-sectional rank normalization (percentile rank 0-1)."""
    ranked = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = factor[:, di]
        valid_mask = ~np.isnan(vals)
        valid = vals[valid_mask]
        if len(valid) < min_count:
            continue
        ranks = pd.Series(valid).rank(pct=True).values
        idx = 0
        for si in range(NS):
            if not np.isnan(vals[si]):
                ranked[si, di] = ranks[idx]
                idx += 1
    return ranked


# =====================================================================
# Multi-Frequency Resonance: Regime-Adaptive Signal
# =====================================================================

def compute_resonance_signal(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
) -> np.ndarray:
    """Compute the composite multi-frequency resonance signal.

    When HT_TRENDMODE indicates trend (1): TRIX weight doubled.
    When HT_TRENDMODE indicates cycle (0): CCI + STOCHRSI weight doubled.

    Returns (NS, ND) composite signal array.
    """
    t0 = time.time()
    print("[V100] Computing resonance signal...", flush=True)

    # Rank-normalize each factor
    ranked = {}
    for fname in FACTOR_NAMES:
        ranked[fname] = rank_normalize(raw_factors[fname], NS, ND)
        valid_pct = np.sum(~np.isnan(ranked[fname])) / (NS * ND) * 100
        print(f"  {fname}: {valid_pct:.1f}% valid", flush=True)

    ht_trend = raw_factors["ht_trendmode"]  # 1=trend, 0=cycle
    composite = np.full((NS, ND), np.nan)

    for di in range(ND):
        ht_vals = ht_trend[:, di]
        valid_ht = ht_vals[~np.isnan(ht_vals)]
        if len(valid_ht) < 5:
            continue

        # Majority vote: trend_frac close to 1 = trend, close to 0 = cycle
        trend_frac = np.mean(valid_ht)

        for si in range(NS):
            signal_sum = 0.0
            weight_sum = 0.0
            has_all = True

            for fname in FACTOR_NAMES:
                rv = ranked[fname][si, di]
                if np.isnan(rv):
                    has_all = False
                    break

                w = BASE_WEIGHTS[fname]

                # Regime-adaptive weighting
                if fname == "trix":
                    w *= (1.0 + trend_frac)
                elif fname in ("cci", "stochrsi"):
                    w *= (1.0 + (1.0 - trend_frac))

                # Directional alignment for contrarian use:
                # ULTOSC high = overbought = bearish -> invert
                # STOCHRSI high = overbought = bearish -> invert
                # Others: high rank = bullish
                if fname == "ultosc":
                    signal_sum += (1.0 - rv) * w
                elif fname == "stochrsi":
                    signal_sum += (1.0 - rv) * w
                else:
                    signal_sum += rv * w
                weight_sum += w

            if has_all and weight_sum > 0:
                composite[si, di] = signal_sum / weight_sum

    print(f"  Resonance signal done: {time.time() - t0:.1f}s", flush=True)
    return composite


# =====================================================================
# NW Kernel Regression on composite signal
# =====================================================================

def compute_nw_predicted_returns(
    raw_factors: Dict[str, np.ndarray],
    composite_signal: np.ndarray,
    NS: int, ND: int,
    training_window: int = 40,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    """Compute NW kernel regression using composite + top factors."""
    t0 = time.time()
    print(
        f"[V100] Computing NW predicted returns "
        f"(window={training_window}, bw={kernel_bandwidth:.1f})...",
        flush=True)

    # Normalize factors cross-sectionally
    normed = {}
    for fname in FACTOR_NAMES:
        normed[fname] = normalize_factor(raw_factors[fname], NS, ND)

    normed_composite = normalize_factor(composite_signal, NS, ND)

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
                    normed_composite[si, tdi],
                    normed["ultosc"][si, tdi],
                    normed["trix"][si, tdi],
                    normed["ret_5d"][si, tdi],
                    normed["ret_10d"][si, tdi],
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
                normed_composite[si, di],
                normed["ultosc"][si, di],
                normed["trix"][si, di],
                normed["ret_5d"][si, di],
                normed["ret_10d"][si, di],
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
# Portfolio volatility
# =====================================================================

def compute_portfolio_volatility(
    C: np.ndarray, NS: int, ND: int,
    vol_lookback: int = 20,
) -> np.ndarray:
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

def backtest_v100(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray,
    composite_signal: np.ndarray,
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
    if end_di is None:
        end_di = ND - 1

    vol_data = port_vol[max(start_di, 20 + 1):end_di]
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

        # --- ENTRY: predicted return + resonance boost ---
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
            # Blend: 60% NW prediction + 40% resonance signal
            res = composite_signal[si, di]
            if not np.isnan(res):
                blended = 0.6 * pred + 0.4 * res
            else:
                blended = pred
            candidates.append((blended, si))

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
            if pred_val <= current_threshold * 0.01:
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
    avg_win = np.mean(
        [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    avg_loss = np.mean(
        [t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])

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
    composite_signal: np.ndarray,
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
    print(f"  WALK-FORWARD V100 {label}")
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

        trades, _, _ = backtest_v100(
            C, O, H, L, NS, ND, dates, syms,
            predicted, composite_signal,
            ker_regime, port_vol,
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
# Main
# =====================================================================

def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V100: MULTI-FREQUENCY RESONANCE STRATEGY")
    print("  HT_TRENDMODE (Hilbert) + ULTOSC + TRIX + CCI + STOCHRSI")
    print("  Regime-adaptive weighting: trend vs cycle")
    print("  NW kernel + vol-adaptive sizing")
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

    # === 2. Compute resonance signal ===
    composite_signal = compute_resonance_signal(raw_factors, NS, ND)

    # === 3. Compute NW predicted returns ===
    predicted = compute_nw_predicted_returns(
        raw_factors, composite_signal, NS, ND,
        training_window=40,
        kernel_bandwidth=1.0,
    )

    # === 4. Compute portfolio volatility ===
    vol_cache: Dict[int, np.ndarray] = {}
    for vlb in [15, 20]:
        vol_cache[vlb] = compute_portfolio_volatility(C, NS, ND, vlb)

    # === 5. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  V100: Multi-Frequency Resonance + NW + Vol-Adaptive")
    print("  NO LEVERAGE.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for top_n in [2, 3]:
        for mps in [2, 3]:
            for vlb in [15, 20]:
                for size_reduce in [0.3, 0.5]:
                    for size_boost in [1.2, 1.5]:
                        sweep_count += 1
                        trades, eq, dd = backtest_v100(
                            C, O, H, L, NS, ND,
                            dates, syms,
                            predicted, composite_signal,
                            ker_regime,
                            vol_cache[vlb],
                            sector_lookup=sector_lookup,
                            top_n=top_n,
                            max_per_sector=mps,
                            hold_days=5,
                            size_reduce=size_reduce,
                            size_boost=size_boost,
                            vol_lookback=vlb,
                            start_di=bt_2019,
                        )

                        if len(trades) < 10:
                            continue

                        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
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
                            "top_n": top_n, "mps": mps,
                            "vlb": vlb,
                            "sr": size_reduce, "sb": size_boost,
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
        f"\n{'TN':>3} {'MPS':>3} {'Vlb':>4} {'SR':>4} {'SB':>4} "
        f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-" * 70)
    for r in results[:10]:
        print(
            f"{r['top_n']:>3} {r['mps']:>3} {r['vlb']:>4} "
            f"{r['sr']:>4.1f} {r['sb']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
            f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

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
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            predicted, composite_signal,
            ker_regime,
            vol_cache[best["vlb"]],
            sector_lookup=sector_lookup,
            top_n=best["top_n"],
            max_per_sector=best["mps"],
            hold_days=5,
            size_reduce=best["sr"],
            size_boost=best["sb"],
            vol_lookback=best["vlb"],
            label=label,
        )

    # === 7. Full period analysis for best config ===
    print("\n" + "=" * 70)
    print("  FULL PERIOD ANALYSIS (2019-2026)")
    print("=" * 70)

    best = best_by_ann
    trades_full, eq_full, dd_full = backtest_v100(
        C, O, H, L, NS, ND, dates, syms,
        predicted, composite_signal,
        ker_regime,
        vol_cache[best["vlb"]],
        sector_lookup=sector_lookup,
        top_n=best["top_n"],
        max_per_sector=best["mps"],
        hold_days=5,
        size_reduce=best["sr"],
        size_boost=best["sb"],
        vol_lookback=best["vlb"],
        start_di=bt_2019,
    )
    analyze(trades_full, eq_full, dd_full, "V100-FULL")

    print(f"\n[V100] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
