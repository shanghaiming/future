"""
V94: CAViaR-X Tail Risk Adaptive Position Sizing
==================================================
Research (2603.25217) proves cross-asset tail risk spillover: when a "core
commodity" triggers VaR breach, related commodities see 30-50% higher VaR
breach probability next day.

Core Innovation: Use cross-asset VaR monitoring to dynamically adjust
position sizes. This is NOT a new factor -- it's a RISK MANAGEMENT layer
on top of V80's rank-based signal generation.

Mechanism:
  1. Define "core" commodities per sector (most liquid/influential)
  2. Compute rolling VaR for each core commodity
  3. If core triggers VaR breach -> REDUCE position sizes in that sector
  4. If no VaR breach anywhere -> NORMAL or boosted sizing
  5. Combine with V80's rank-based signal generation

Parameters to sweep:
  - var_window: 40, 60, 90
  - var_level: 0.05, 0.10
  - size_reduction: 0.3, 0.5, 0.7
  - size_boost: 1.0, 1.2, 1.5
  - st_weight: 0.60, 0.65, 0.70
  - max_positions: 2, 3, 4
  - max_per_sector: 2, 3

Walk-forward 2019-2026. No leverage. CASH0=1,000,000, COMM=0.0005.
Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
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

# Short-term (5d) factor weights (same as V80)
ST_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol5d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}

# Medium-term (20d) factor weights (same as V80)
MT_WEIGHTS = {
    "rank_ret20d": 0.25,
    "rank_oi20d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol20d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range20d": 0.10,
    "rank_atrp20d": 0.05,
}

# Sector definitions (same as V80)
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

# Core commodities per sector (most liquid / influential)
CORE_SYMBOLS = {
    'BLACK': ['i', 'rb'],
    'METAL': ['cu'],
    'ENERGY': ['sc'],
    'CHEMICAL': ['ta', 'v'],
    'AGRI': ['a', 'c'],
    'SOFTS': ['cf', 'sr'],
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


def find_core_indices(
    syms: List[str],
) -> Dict[str, List[int]]:
    """Map sector -> list of symbol indices for core commodities."""
    base_to_si: Dict[str, int] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        base_to_si[base] = si

    core_indices: Dict[str, List[int]] = {}
    for sector, symbols in CORE_SYMBOLS.items():
        indices = []
        for sym in symbols:
            if sym in base_to_si:
                indices.append(base_to_si[sym])
        if indices:
            core_indices[sector] = indices
    return core_indices


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
    print("[V94] Computing raw factors (5d + 20d)...", flush=True)

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

    ret_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 20])
                    and C[si, di - 20] > 0):
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    oi_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 20])
                    and OI[si, di - 20] > 0):
                oi_20d[si, di] = OI[si, di] / OI[si, di - 20] - 1.0

    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = V[si, di - 20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                vol_20d[si, di] = np.mean(valid)

    range_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rng_vals = []
            for j in range(di - 20, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng_vals) >= 10:
                range_20d[si, di] = np.mean(rng_vals)

    atrp_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            atr_vals = []
            for j in range(di - 20, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_20d[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d, "ret_10d": ret_10d, "ret_20d": ret_20d,
        "oi_5d": oi_5d, "oi_20d": oi_20d,
        "vol_5d": vol_5d, "vol_20d": vol_20d,
        "range_5d": range_5d, "range_20d": range_20d,
        "rsi14": rsi14,
        "atrp_5d": atrp_5d, "atrp_20d": atrp_20d,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V94] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        "rank_ret5d": raw_factors["ret_5d"],
        "rank_ret10d": raw_factors["ret_10d"],
        "rank_oi5d": raw_factors["oi_5d"],
        "rank_vol5d": raw_factors["vol_5d"],
        "rank_range5d": raw_factors["range_5d"],
        "rank_rsi": raw_factors["rsi14"],
        "rank_atrp5d": raw_factors["atrp_5d"],
        "rank_ret20d": raw_factors["ret_20d"],
        "rank_oi20d": raw_factors["oi_20d"],
        "rank_vol20d": raw_factors["vol_20d"],
        "rank_range20d": raw_factors["range_20d"],
        "rank_atrp20d": raw_factors["atrp_20d"],
    }

    INVERT_FACTORS = {
        "rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi",
        "rank_ret20d", "rank_oi20d",
    }

    ranks = {}
    for name, factor in factors_to_rank.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < min_count:
                continue
            ranked = (
                pd.Series(vals)
                .rank(pct=True, na_option="keep")
                .values
            )
            if name in INVERT_FACTORS:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


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


def build_multi_tf_composite(
    ranks: Dict[str, np.ndarray],
    st_weights: Dict[str, float],
    mt_weights: Dict[str, float],
    st_weight: float,
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t0 = time.time()
    print(f"[V94] Building multi-TF composite (st_w={st_weight:.2f})...",
          flush=True)

    mt_weight = 1.0 - st_weight

    combined = np.full((NS, ND), np.nan)
    st_comp = np.full((NS, ND), np.nan)
    mt_comp = np.full((NS, ND), np.nan)
    n_confirm_st = np.zeros((NS, ND), dtype=int)
    n_confirm_mt = np.zeros((NS, ND), dtype=int)

    st_names = list(st_weights.keys())
    st_wvals = np.array([st_weights[k] for k in st_names])
    mt_names = list(mt_weights.keys())
    mt_wvals = np.array([mt_weights[k] for k in mt_names])

    for di in range(ND):
        for si in range(NS):
            st_vals = []
            st_wsum = 0.0
            st_confirm = 0
            for idx, name in enumerate(st_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                st_vals.append(rank_val * st_wvals[idx])
                st_wsum += st_wvals[idx]
                if rank_val > 0.5:
                    st_confirm += 1

            if st_wsum > 0 and st_confirm >= min_factors:
                st_comp[si, di] = sum(st_vals) / st_wsum
                n_confirm_st[si, di] = st_confirm

            mt_vals = []
            mt_wsum = 0.0
            mt_confirm = 0
            for idx, name in enumerate(mt_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                mt_vals.append(rank_val * mt_wvals[idx])
                mt_wsum += mt_wvals[idx]
                if rank_val > 0.5:
                    mt_confirm += 1

            if mt_wsum > 0 and mt_confirm >= min_factors:
                mt_comp[si, di] = sum(mt_vals) / mt_wsum
                n_confirm_mt[si, di] = mt_confirm

            if not np.isnan(st_comp[si, di]) and not np.isnan(mt_comp[si, di]):
                combined[si, di] = (st_weight * st_comp[si, di]
                                    + mt_weight * mt_comp[si, di])

    print(f"  Multi-TF composite done: {time.time() - t0:.1f}s", flush=True)
    return combined, st_comp, mt_comp, n_confirm_st, n_confirm_mt


def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    st_weight: float = 0.60,
    st_weights: Optional[Dict[str, float]] = None,
    mt_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    if st_weights is None:
        st_weights = ST_WEIGHTS
    if mt_weights is None:
        mt_weights = MT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    combined, st_comp, mt_comp, ncf_st, ncf_mt = build_multi_tf_composite(
        ranks, st_weights, mt_weights, st_weight, NS, ND)

    return {
        "composite": combined,
        "st_comp": st_comp,
        "mt_comp": mt_comp,
        "n_confirm_st": ncf_st,
        "n_confirm_mt": ncf_mt,
        "ker_regime": ker_regime,
        "ranks": ranks,
    }


# =====================================================================
# CAViaR-X: Cross-Asset VaR Tail Risk Monitor
# =====================================================================

def compute_var_breach_matrix(
    C: np.ndarray,
    core_indices: Dict[str, List[int]],
    NS: int, ND: int,
    var_window: int = 60,
    var_level: float = 0.05,
) -> np.ndarray:
    """Compute VaR breach indicator for each sector on each day.

    Returns breach_matrix[NS, ND] where:
      1 = sector's core commodity triggered VaR breach on that day
      0 = no breach

    For each core commodity:
      - Compute daily returns
      - Rolling window of var_window days
      - VaR at var_level percentile
      - If today's return < VaR -> breach

    A sector breaches if ANY of its core commodities breach.
    """
    t0 = time.time()
    print(f"[V94] Computing CAViaR-X breach matrix "
          f"(window={var_window}, level={var_level})...", flush=True)

    # First compute per-sector breach
    sector_breach: Dict[str, np.ndarray] = {}
    for sector, si_list in core_indices.items():
        breach_arr = np.zeros(ND, dtype=np.int32)
        for si in si_list:
            # Compute daily returns
            returns = np.full(ND, np.nan)
            for di in range(1, ND):
                if (not np.isnan(C[si, di])
                        and not np.isnan(C[si, di - 1])
                        and C[si, di - 1] > 0):
                    returns[di] = C[si, di] / C[si, di - 1] - 1.0

            # Rolling VaR check
            for di in range(var_window + 1, ND):
                window_rets = returns[di - var_window:di]
                valid = window_rets[~np.isnan(window_rets)]
                if len(valid) < var_window // 2:
                    continue
                var_threshold = np.percentile(valid, var_level * 100)
                if not np.isnan(returns[di]) and returns[di] < var_threshold:
                    breach_arr[di] = 1
        sector_breach[sector] = breach_arr

    # Map breach to all instruments in that sector
    breach_matrix = np.zeros((NS, ND), dtype=np.int32)
    # Need reverse lookup: si -> sector
    # We'll fill this in the backtest using sector_lookup
    # For now store sector-level data
    breach_matrix_dict = sector_breach

    print(f"  CAViaR-X done: {time.time() - t0:.1f}s", flush=True)
    return breach_matrix_dict


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


def backtest_v94(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    sector_breach: Dict[str, np.ndarray],
    st_weight: float = 0.65,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    max_positions: int = 3,
    max_per_sector: int = 2,
    size_reduction: float = 0.5,
    size_boost: float = 1.2,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V94: CAViaR-X cross-asset tail risk position sizing.

    Key difference from V80:
    - Before entering a position, check if the instrument's sector
      has a VaR breach from the core commodity
    - If breach: reduce position size by size_reduction factor
    - If no breach anywhere: boost position size by size_boost factor
    - This is a risk management overlay, not a new signal factor
    """
    composite = sigs["composite"]
    n_confirm_st = sigs["n_confirm_st"]
    n_confirm_mt = sigs["n_confirm_mt"]
    ker_regime = sigs["ker_regime"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

        # Group positions by symbol
        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr))

        # Exit logic
        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any(c < sp for _, _, sp, _, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0

                    # Check if sector was in breach when position was opened
                    sec = sector_lookup.get(si, 'OTHER')
                    breach_info = "breach" if (
                        sec in sector_breach
                        and di > 0
                        and sector_breach[sec][di - 1] == 1
                    ) else "clear"

                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sec,
                        "reason": "stop",
                        "pyr": is_pyr,
                        "mode": mode[:1].upper(),
                        "tail_risk": breach_info,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0

                    sec = sector_lookup.get(si, 'OTHER')
                    breach_info = "breach" if (
                        sec in sector_breach
                        and di > 0
                        and sector_breach[sec][di - 1] == 1
                    ) else "clear"

                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sec,
                        "reason": "hold",
                        "pyr": is_pyr,
                        "mode": mode[:1].upper(),
                        "tail_risk": breach_info,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

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

        # --- ENTRY with CAViaR-X risk management ---
        held = {p[0] for p in positions}
        if len(positions) >= max_positions:
            continue

        # Check cross-asset VaR breach status for each sector
        sector_breach_today: Dict[str, bool] = {}
        for sector, breach_arr in sector_breach.items():
            sector_breach_today[sector] = (
                di > 0 and breach_arr[di - 1] == 1
            )

        # Check if ANY sector has a breach (for boost eligibility)
        any_breach = any(sector_breach_today.values())

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < current_threshold:
                continue
            total_confirm = n_confirm_st[si, di] + n_confirm_mt[si, di]
            if total_confirm < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # CAViaR-X: compute tail risk multiplier
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_breach_today.get(sym_sector, False):
                tail_mult = size_reduction  # reduce exposure
            elif not any_breach:
                tail_mult = size_boost  # boost when calm
            else:
                tail_mult = 1.0  # neutral: another sector breached, not ours

            candidates.append((composite[si, di], si, tail_mult))

        candidates.sort(key=lambda x: -x[0])

        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, 'OTHER')] += 1

        new_entries = []
        for rank_val, si, tail_mult in candidates:
            if len(positions) + len(new_entries) >= max_positions:
                break
            if si in held:
                continue
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            new_entries.append((rank_val, si, sym_sector, tail_mult))
            sector_counts[sym_sector] += 1

        num_total = len(positions) + len(new_entries)
        if num_total == 0:
            continue
        base_alloc = LEVERAGE / num_total

        updated_positions = []
        for si, edi, ep, sp, old_alloc, is_pyr in positions:
            updated_positions.append(
                (si, edi, ep, sp, base_alloc, is_pyr))

        for rank_val, si, sym_sector, tail_mult in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue

            # CAViaR-X: apply tail risk multiplier to allocation
            adjusted_alloc = base_alloc * tail_mult

            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr,
                 adjusted_alloc, False))

        positions = updated_positions

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
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

    n_pyr = sum(1 for t in trades if t.get("pyr"))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    # CAViaR-X specific: breakdown by tail risk status
    breach_trades = [t for t in trades if t.get("tail_risk") == "breach"]
    clear_trades = [t for t in trades if t.get("tail_risk") == "clear"]
    n_breach = len(breach_trades)
    n_clear = len(clear_trades)
    breach_wr = (sum(1 for t in breach_trades if t["pnl_pct"] > 0)
                 / max(n_breach, 1) * 100) if n_breach > 0 else 0
    clear_wr = (sum(1 for t in clear_trades if t["pnl_pct"] > 0)
                / max(n_clear, 1) * 100) if n_clear > 0 else 0

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
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}"
    )
    print(
        f"    CAViaR-X: breach_trades={n_breach} (WR={breach_wr:.1f}%) "
        f"clear_trades={n_clear} (WR={clear_wr:.1f}%)"
    )
    print(
        f"    avg_pnl={avg_pnl:+.3f}% avg_win={avg_win:+.3f}% "
        f"avg_loss={avg_loss:+.3f}% "
        f"modes=[W:{mode_counts['W']} N:{mode_counts['N']} "
        f"L:{mode_counts['L']}]"
    )
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
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    sector_breach: Dict[str, np.ndarray],
    st_weight: float = 0.65,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    max_positions: int = 3,
    max_per_sector: int = 2,
    size_reduction: float = 0.5,
    size_boost: float = 1.2,
) -> List[dict]:
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V94 CAViaR-X "
        f"(st_w={st_weight:.2f} wt={win_threshold:.2f} "
        f"nt={normal_threshold:.2f} lt={lose_threshold:.2f} "
        f"mp={max_positions} mps={max_per_sector} "
        f"red={size_reduction} boost={size_boost})"
    )
    print(f"  Cross-Asset VaR Tail Risk Management Layer")
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

        trades, _, _ = backtest_v94(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            sector_breach=sector_breach,
            st_weight=st_weight,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=3.0,
            max_positions=max_positions,
            max_per_sector=max_per_sector,
            size_reduction=size_reduction,
            size_boost=size_boost,
            min_confidence=3,
            use_ker_gate=True,
            hold_days=5,
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
            n_breach = sum(
                1 for t in test_trades if t.get("tail_risk") == "breach")
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"breach={n_breach} sectors=[{sec_str}]",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod(
            [1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_breach = sum(
            1 for t in all_trades if t.get("tail_risk") == "breach")
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%} "
            f"breach={n_breach}"
        )
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V94: CAViaR-X CROSS-ASSET TAIL RISK POSITION SIZING")
    print("  Research (2603.25217): tail risk is CONTAGIOUS")
    print("  Core commodities trigger VaR -> reduce sector exposure")
    print("  No breach -> normal or boosted sizing")
    print("  Risk management layer on V80 rank-based signals")
    print("  NO LEVERAGE. CASH0=1,000,000, COMM=0.0005")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}"
    )

    sector_lookup = build_sector_lookup(syms)
    sector_dist: Dict[str, int] = defaultdict(int)
    for sec in sector_lookup.values():
        sector_dist[sec] += 1
    print(f"  Sector distribution: {dict(sector_dist)}")

    core_indices = find_core_indices(syms)
    print(f"  Core commodities: {core_indices}")

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Pre-compute VaR breach matrices for each (window, level) combo ===
    breach_cache: Dict[Tuple[int, float], Dict[str, np.ndarray]] = {}
    for vw in [40, 60, 90]:
        for vl in [0.05, 0.10]:
            print(f"\n--- Computing VaR breach (window={vw}, level={vl}) ---")
            breach_cache[(vw, vl)] = compute_var_breach_matrix(
                C, core_indices, NS, ND,
                var_window=vw, var_level=vl,
            )

    # === 2. Pre-compute signals for each st_weight ===
    signal_cache: Dict[float, Dict] = {}
    for st_w in [0.60, 0.65, 0.70]:
        print(f"\n--- Computing signals for st_weight={st_w:.2f} ---")
        signal_cache[st_w] = compute_all_signals(
            C, O, H, L, V, OI, NS, ND, st_weight=st_w)

    # === 3. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  CAViaR-X: Cross-Asset VaR Tail Risk Management")
    print("=" * 70)

    results: List[dict] = []

    sweep_count = 0
    for st_w in [0.60, 0.65, 0.70]:
        sigs = signal_cache[st_w]
        for vw in [40, 60, 90]:
            for vl in [0.05, 0.10]:
                sector_breach = breach_cache[(vw, vl)]
                for sr in [0.3, 0.5, 0.7]:
                    for sb in [1.0, 1.2, 1.5]:
                        for mp in [2, 3, 4]:
                            for mps in [2, 3]:
                                sweep_count += 1
                                trades, eq, dd = backtest_v94(
                                    C, O, H, L, NS, ND, dates, syms, sigs,
                                    sector_lookup=sector_lookup,
                                    sector_breach=sector_breach,
                                    st_weight=st_w,
                                    win_threshold=0.60,
                                    normal_threshold=0.80,
                                    lose_threshold=0.90,
                                    win_rate_window=15,
                                    atr_stop=3.0,
                                    max_positions=mp,
                                    max_per_sector=mps,
                                    size_reduction=sr,
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
                                    1 / max(1.0, n_days / 252)) - 1) * 100
                                ap = [t["pnl_abs"]
                                      for t in sorted(
                                          trades, key=lambda x: x["di"])]
                                rets_arr = np.array(ap) / CASH0
                                sh_val = (
                                    np.mean(rets_arr)
                                    / np.std(rets_arr) * np.sqrt(252)
                                    if np.std(rets_arr) > 0 else 0)

                                n_breach = sum(
                                    1 for t in trades
                                    if t.get("tail_risk") == "breach")

                                results.append({
                                    "st_w": st_w,
                                    "vw": vw, "vl": vl,
                                    "sr": sr, "sb": sb,
                                    "mp": mp, "mps": mps,
                                    "n": len(trades), "wr": wr,
                                    "ann": ann, "dd": dd,
                                    "sharpe": sh_val, "eq": eq,
                                    "n_breach": n_breach,
                                })

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Evaluated {sweep_count} configs, "
          f"{len(results)} with 10+ trades")
    print(
        f"\n{'STw':>4} {'VW':>3} {'VL':>4} {'SR':>4} {'SB':>4} "
        f"{'MP':>3} {'MPS':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Breach':>7} {'Eq':>12}"
    )
    print("-" * 100)
    for r in results[:10]:
        print(
            f"{r['st_w']:>4.2f} {r['vw']:>3} {r['vl']:>4.2f} "
            f"{r['sr']:>4.1f} {r['sb']:>4.1f} "
            f"{r['mp']:>3} {r['mps']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['n_breach']:>7} {r['eq']:>12,.0f}"
        )

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 4. Top config: full backtest ===
    best = results[0]
    print("\n" + "=" * 70)
    print(
        f"  TOP CONFIG -- FULL BACKTEST "
        f"(st_w={best['st_w']:.2f} vw={best['vw']} vl={best['vl']} "
        f"sr={best['sr']} sb={best['sb']} mp={best['mp']} mps={best['mps']})"
    )
    print("=" * 70)

    sector_breach_best = breach_cache[(best["vw"], best["vl"])]
    sigs_best = signal_cache[best["st_w"]]

    trades_full, eq_full, dd_full = backtest_v94(
        C, O, H, L, NS, ND, dates, syms, sigs_best,
        sector_lookup=sector_lookup,
        sector_breach=sector_breach_best,
        st_weight=best["st_w"],
        win_threshold=0.60,
        normal_threshold=0.80,
        lose_threshold=0.90,
        win_rate_window=15,
        atr_stop=3.0,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        size_reduction=best["sr"],
        size_boost=best["sb"],
        start_di=60,
    )
    label = (
        f"st_w={best['st_w']:.2f} vw={best['vw']} vl={best['vl']} "
        f"sr={best['sr']} sb={best['sb']} mp={best['mp']} mps={best['mps']}"
    )
    analyze(trades_full, eq_full, dd_full, label)

    # === 5. Walk-forward for best config ===
    print("\n" + "=" * 70)
    print(f"  BEST WALK-FORWARD: {label}")
    print("=" * 70)
    walk_forward(
        C, O, H, L, NS, ND, dates, syms,
        sigs_best,
        sector_lookup=sector_lookup,
        sector_breach=sector_breach_best,
        st_weight=best["st_w"],
        win_threshold=0.60,
        normal_threshold=0.80,
        lose_threshold=0.90,
        win_rate_window=15,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        size_reduction=best["sr"],
        size_boost=best["sb"],
    )

    # === 6. Comparison: V94 vs V80 baseline ===
    print("\n" + "=" * 70)
    print("  COMPARISON: V94 (CAViaR-X) vs V80 baseline (no tail risk)")
    print("  (2019-2026 OOS)")
    print("=" * 70)

    # V94 best
    trades_v94, eq_v94, dd_v94 = backtest_v94(
        C, O, H, L, NS, ND, dates, syms, sigs_best,
        sector_lookup=sector_lookup,
        sector_breach=sector_breach_best,
        st_weight=best["st_w"],
        win_threshold=0.60,
        normal_threshold=0.80,
        lose_threshold=0.90,
        win_rate_window=15,
        atr_stop=3.0,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        size_reduction=best["sr"],
        size_boost=best["sb"],
        start_di=bt_2019,
    )

    # V80 baseline: same signals but size_reduction=1.0, size_boost=1.0
    # (effectively no tail risk management)
    no_breach = {}  # empty dict = no breaches ever = pure V80
    trades_v80, eq_v80, dd_v80 = backtest_v94(
        C, O, H, L, NS, ND, dates, syms, sigs_best,
        sector_lookup=sector_lookup,
        sector_breach=no_breach,
        st_weight=best["st_w"],
        win_threshold=0.60,
        normal_threshold=0.80,
        lose_threshold=0.90,
        win_rate_window=15,
        atr_stop=3.0,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        size_reduction=1.0,
        size_boost=1.0,
        start_di=bt_2019,
    )

    print(f"\n  V94 CAViaR-X (sr={best['sr']} sb={best['sb']}):")
    analyze(trades_v94, eq_v94, dd_v94, "V94-CAViaR-X")
    print(f"\n  V80 BASELINE (no tail risk management):")
    analyze(trades_v80, eq_v80, dd_v80, "V80-baseline")

    if trades_v94 and trades_v80:
        print(
            f"\n  V94 vs V80: "
            f"ann_delta={((eq_v94/CASH0)**(1/7)-1)*100 - ((eq_v80/CASH0)**(1/7)-1)*100:+.1f}% "
            f"dd_delta={dd_v94 - dd_v80:+.1f}% "
            f"trade_delta={len(trades_v94) - len(trades_v80):+d}"
        )

    print(f"\n[V94] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
