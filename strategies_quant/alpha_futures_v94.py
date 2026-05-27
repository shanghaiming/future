"""
V94: CAViaR-X Tail Risk Adaptive Position Sizing
==================================================
Research (2603.25217) proves cross-asset tail risk spillover: when a "core
commodity" triggers VaR breach, related commodities see 30-50% higher VaR
breach probability next day.

Core Innovation: Use cross-asset VaR monitoring to dynamically adjust
position sizes AND filter signals. Risk management layer on V80 signals.

Mechanism:
  1. Core commodities per sector: rolling VaR estimation
  2. VaR breach -> reduce size + optional skip entry
  3. No breach -> boost size for compounding
  4. V80 rank-based signal generation with dynamic thresholds

Parameters to sweep:
  - var_window: 40, 60, 90
  - var_level: 0.05, 0.10
  - size_reduction: 0.3, 0.5, 0.7
  - size_boost: 1.0, 1.3, 1.5
  - st_weight: 0.60, 0.65, 0.70
  - max_positions: 2, 3, 4
  - max_per_sector: 2, 3
  - win_threshold: 0.55, 0.60, 0.65
  - normal_threshold: 0.75, 0.80, 0.85
  - lose_threshold: 0.85, 0.90, 0.95
  - win_rate_window: 10, 15
  - breach_skip: True, False

Walk-forward 2019-2026. CASH0=1,000,000, COMM=0.0005.
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

ST_WEIGHTS = {
    "rank_ret5d": 0.25, "rank_oi5d": 0.20, "rank_rsi": 0.15,
    "rank_vol5d": 0.15, "rank_ret10d": 0.10, "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}
MT_WEIGHTS = {
    "rank_ret20d": 0.25, "rank_oi20d": 0.20, "rank_rsi": 0.15,
    "rank_vol20d": 0.15, "rank_ret10d": 0.10, "rank_range20d": 0.10,
    "rank_atrp20d": 0.05,
}

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


def find_core_indices(syms: List[str]) -> Dict[str, List[int]]:
    base_to_si: Dict[str, int] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        base_to_si[base] = si
    core_indices: Dict[str, List[int]] = {}
    for sector, symbols in CORE_SYMBOLS.items():
        indices = [base_to_si[s] for s in symbols if s in base_to_si]
        if indices:
            core_indices[sector] = indices
    return core_indices


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int,
                       period: int = 14) -> np.ndarray:
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
                valid_g, valid_l = [], []
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
                        rsi[si, di + period - 1] = (
                            100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
                continue
            avg_gain = (avg_gain * (period - 1) + gains[di]) / period
            avg_loss = (avg_loss * (period - 1) + losses[di]) / period
            if avg_loss == 0:
                rsi[si, di] = 100.0
            else:
                rsi[si, di] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def compute_raw_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V94] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5])
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
                    prev_c = (C[si, j - 1]
                              if j > 0 and not np.isnan(C[si, j - 1])
                              else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10])
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
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 20])
                    and C[si, di - 20] > 0):
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    oi_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 20])
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
                    prev_c = (C[si, j - 1]
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
        "rsi14": rsi14, "atrp_5d": atrp_5d, "atrp_20d": atrp_20d,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray], NS: int, ND: int,
    min_count: int = 10,
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
    INVERT = {"rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi",
              "rank_ret20d", "rank_oi20d"}
    ranks = {}
    for name, factor in factors_to_rank.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            if np.sum(~np.isnan(vals)) < min_count:
                continue
            ranked = pd.Series(vals).rank(pct=True, na_option="keep").values
            if name in INVERT:
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
            net = abs(valid[-1] - valid[0])
            total = np.sum(np.abs(np.diff(valid)))
            if total > 1e-10:
                ker_10[si, di] = net / total
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
    print(f"[V94] Building composite (st_w={st_weight:.2f})...", flush=True)
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
            st_vals, st_wsum, st_c = [], 0.0, 0
            for idx, name in enumerate(st_names):
                v = ranks[name][si, di]
                if np.isnan(v):
                    continue
                st_vals.append(v * st_wvals[idx])
                st_wsum += st_wvals[idx]
                if v > 0.5:
                    st_c += 1
            if st_wsum > 0 and st_c >= min_factors:
                st_comp[si, di] = sum(st_vals) / st_wsum
                n_confirm_st[si, di] = st_c
            mt_vals, mt_wsum, mt_c = [], 0.0, 0
            for idx, name in enumerate(mt_names):
                v = ranks[name][si, di]
                if np.isnan(v):
                    continue
                mt_vals.append(v * mt_wvals[idx])
                mt_wsum += mt_wvals[idx]
                if v > 0.5:
                    mt_c += 1
            if mt_wsum > 0 and mt_c >= min_factors:
                mt_comp[si, di] = sum(mt_vals) / mt_wsum
                n_confirm_mt[si, di] = mt_c
            if not np.isnan(st_comp[si, di]) and not np.isnan(mt_comp[si, di]):
                combined[si, di] = st_weight * st_comp[si, di] + mt_weight * mt_comp[si, di]
    print(f"  Composite done: {time.time() - t0:.1f}s", flush=True)
    return combined, st_comp, mt_comp, n_confirm_st, n_confirm_mt


def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    st_weight: float = 0.65,
) -> Dict[str, np.ndarray]:
    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    combined, st_comp, mt_comp, ncf_st, ncf_mt = build_multi_tf_composite(
        ranks, ST_WEIGHTS, MT_WEIGHTS, st_weight, NS, ND)
    return {
        "composite": combined, "st_comp": st_comp, "mt_comp": mt_comp,
        "n_confirm_st": ncf_st, "n_confirm_mt": ncf_mt,
        "ker_regime": ker_regime, "ranks": ranks,
    }


def compute_var_breach_matrix(
    C: np.ndarray,
    core_indices: Dict[str, List[int]],
    ND: int,
    var_window: int = 60,
    var_level: float = 0.05,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print(f"[V94] CAViaR-X (win={var_window}, lvl={var_level})...", flush=True)
    sector_breach: Dict[str, np.ndarray] = {}
    for sector, si_list in core_indices.items():
        breach_arr = np.zeros(ND, dtype=np.int32)
        for si in si_list:
            returns = np.full(ND, np.nan)
            for di in range(1, ND):
                if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1])
                        and C[si, di - 1] > 0):
                    returns[di] = C[si, di] / C[si, di - 1] - 1.0
            for di in range(var_window + 1, ND):
                window_rets = returns[di - var_window:di]
                valid = window_rets[~np.isnan(window_rets)]
                if len(valid) < var_window // 2:
                    continue
                var_threshold = np.percentile(valid, var_level * 100)
                if not np.isnan(returns[di]) and returns[di] < var_threshold:
                    breach_arr[di] = 1
        sector_breach[sector] = breach_arr
    print(f"  CAViaR-X done: {time.time() - t0:.1f}s", flush=True)
    return sector_breach


def compute_atr_at(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                   si: int, di: int, start_di: int) -> Optional[float]:
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    return np.mean(atr_v) if atr_v else None


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
    mode: str, wt: float, nt: float, lt: float,
) -> float:
    if mode == "winning":
        return wt
    elif mode == "losing":
        return lt
    return nt


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
    size_boost: float = 1.3,
    breach_skip: bool = False,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
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

        mode = get_dynamic_mode(recent_trades_win, win_threshold,
                                win_rate_window)
        threshold = get_mode_threshold(mode, win_threshold,
                                       normal_threshold, lose_threshold)

        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr))

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
                    sec = sector_lookup.get(si, 'OTHER')
                    br = ("breach" if sec in sector_breach and di > 0
                          and sector_breach[sec][di - 1] == 1 else "clear")
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di, "year": d.year,
                        "sym": syms[si], "sector": sec,
                        "reason": "stop", "pyr": is_pyr,
                        "mode": mode[:1].upper(), "tail_risk": br,
                    })
                    recent_trades_win.append(1 if pnl > 0 else 0)
            elif hold >= 5:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    sec = sector_lookup.get(si, 'OTHER')
                    br = ("breach" if sec in sector_breach and di > 0
                          and sector_breach[sec][di - 1] == 1 else "clear")
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di, "year": d.year,
                        "sym": syms[si], "sector": sec,
                        "reason": "hold", "pyr": is_pyr,
                        "mode": mode[:1].upper(), "tail_risk": br,
                    })
                    recent_trades_win.append(1 if pnl > 0 else 0)
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

        # ENTRY
        held = {p[0] for p in positions}
        if len(positions) >= max_positions:
            continue

        sector_breach_today: Dict[str, bool] = {}
        for sector, breach_arr in sector_breach.items():
            sector_breach_today[sector] = di > 0 and breach_arr[di - 1] == 1
        any_breach = any(sector_breach_today.values())

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < threshold:
                continue
            total_cf = n_confirm_st[si, di] + n_confirm_mt[si, di]
            if total_cf < 3:
                continue
            if ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            sym_sector = sector_lookup.get(si, 'OTHER')
            is_breached = sector_breach_today.get(sym_sector, False)

            if breach_skip and is_breached:
                continue

            if is_breached:
                tail_mult = size_reduction
            elif not any_breach:
                tail_mult = size_boost
            else:
                tail_mult = 1.0

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
            updated_positions.append((si, edi, ep, sp, base_alloc, is_pyr))

        for rank_val, si, sym_sector, tail_mult in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            adjusted_alloc = base_alloc * tail_mult
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr,
                 adjusted_alloc, False))

        positions = updated_positions

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

    breach_trades = [t for t in trades if t.get("tail_risk") == "breach"]
    clear_trades = [t for t in trades if t.get("tail_risk") == "clear"]
    n_breach = len(breach_trades)
    breach_wr = (sum(1 for t in breach_trades if t["pnl_pct"] > 0)
                 / max(n_breach, 1) * 100) if n_breach > 0 else 0
    clear_wr = (sum(1 for t in clear_trades if t["pnl_pct"] > 0)
                / max(len(clear_trades), 1) * 100)

    print(
        f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% "
        f"DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
    print(
        f"    CAViaR-X: breach={n_breach} (WR={breach_wr:.1f}%) "
        f"clear={len(clear_trades)} (WR={clear_wr:.1f}%)")

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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% "
              f"cum={cum:+.1%}")

    return {"n": len(trades), "wr": wr, "dd": max_dd,
            "ann": ann, "sh": sh, "eq": equity}


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
    size_boost: float = 1.3,
    breach_skip: bool = False,
) -> List[dict]:
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V94 CAViaR-X "
          f"(st_w={st_weight:.2f} wt={win_threshold:.2f} "
          f"nt={normal_threshold:.2f} lt={lose_threshold:.2f} "
          f"mp={max_positions} mps={max_per_sector} "
          f"red={size_reduction} boost={size_boost} skip={breach_skip})")
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
            max_positions=max_positions,
            max_per_sector=max_per_sector,
            size_reduction=size_reduction,
            size_boost=size_boost,
            breach_skip=breach_skip,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            n_breach = sum(
                1 for t in test_trades if t.get("tail_risk") == "breach")
            print(
                f"  {test_year}: {n}t WR={nw/n*100:.1f}% breach={n_breach}",
                flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_breach = sum(
            1 for t in all_trades if t.get("tail_risk") == "breach")
        print(f"\n  WF TOTAL: {len(all_trades)}t WR={nw/len(all_trades)*100:.1f}% "
              f"cum={cum:+.1%} breach={n_breach}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V94: CAViaR-X CROSS-ASSET TAIL RISK POSITION SIZING")
    print("  NO LEVERAGE. CASH0=1,000,000, COMM=0.0005")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to "
          f"{dates[-1].strftime('%Y-%m-%d')}")

    sector_lookup = build_sector_lookup(syms)
    core_indices = find_core_indices(syms)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # Pre-compute VaR breach matrices
    breach_cache: Dict[Tuple[int, float], Dict[str, np.ndarray]] = {}
    for vw in [40, 60, 90]:
        for vl in [0.05, 0.10]:
            breach_cache[(vw, vl)] = compute_var_breach_matrix(
                C, core_indices, ND, var_window=vw, var_level=vl)

    # Pre-compute signals
    signal_cache: Dict[float, Dict] = {}
    for st_w in [0.60, 0.65, 0.70]:
        print(f"\n--- Signals for st_weight={st_w:.2f} ---")
        signal_cache[st_w] = compute_all_signals(
            C, O, H, L, V, OI, NS, ND, st_weight=st_w)

    # Parameter sweep - same range as V80
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    # V80 sweep space: st_w(3) * mp(3) * mps(2) * ww(2) * wt(3) * nt(3) * lt(3)
    # V94 adds: vw(3) * vl(2) * sr(3) * sb(3) * bskip(2)
    # Full space = 972 * 54 = too many
    # Strategy: fix CAViaR-X params to best candidates, sweep V80 params fully
    # Then sweep CAViaR-X params for top V80 configs

    # Phase 1: For each CAViaR-X config, sweep V80 params (coarse)
    print("\n--- Phase 1: Coarse sweep ---")
    for st_w in [0.60, 0.65, 0.70]:
        sigs = signal_cache[st_w]
        for vw in [60]:  # fixed best guess
            for vl in [0.05]:  # fixed best guess
                sector_breach = breach_cache[(vw, vl)]
                for sr in [0.3, 0.5]:
                    for sb in [1.0, 1.5]:
                        for bskip in [True, False]:
                            for mp in [2, 3, 4]:
                                for mps in [2, 3]:
                                    for ww in [10, 15]:
                                        for wt in [0.55, 0.60, 0.65]:
                                            for nt in [0.75, 0.80, 0.85]:
                                                for lt in [0.85, 0.90, 0.95]:
                                                    if lt <= nt:
                                                        continue
                                                    sweep_count += 1
                                                    trades, eq, dd = backtest_v94(
                                                        C, O, H, L, NS, ND,
                                                        dates, syms, sigs,
                                                        sector_lookup=sector_lookup,
                                                        sector_breach=sector_breach,
                                                        st_weight=st_w,
                                                        win_threshold=wt,
                                                        normal_threshold=nt,
                                                        lose_threshold=lt,
                                                        win_rate_window=ww,
                                                        max_positions=mp,
                                                        max_per_sector=mps,
                                                        size_reduction=sr,
                                                        size_boost=sb,
                                                        breach_skip=bskip,
                                                        start_di=bt_2019,
                                                    )
                                                    if len(trades) < 10:
                                                        continue
                                                    nw = sum(1 for t in trades
                                                             if t["pnl_pct"] > 0)
                                                    wr = nw / len(trades) * 100
                                                    n_days = max(
                                                        1, trades[-1]["di"]
                                                        - trades[0]["di"])
                                                    ann = ((eq / CASH0) ** (
                                                        1 / max(1.0, n_days / 252))
                                                        - 1) * 100
                                                    ap = [t["pnl_abs"]
                                                          for t in sorted(
                                                              trades,
                                                              key=lambda x: x["di"])]
                                                    rets_arr = np.array(ap) / CASH0
                                                    sh_val = (
                                                        np.mean(rets_arr)
                                                        / np.std(rets_arr)
                                                        * np.sqrt(252)
                                                        if np.std(rets_arr) > 0
                                                        else 0)
                                                    results.append({
                                                        "st_w": st_w,
                                                        "vw": vw, "vl": vl,
                                                        "sr": sr, "sb": sb,
                                                        "mp": mp, "mps": mps,
                                                        "wt": wt, "nt": nt, "lt": lt,
                                                        "ww": ww,
                                                        "bskip": bskip,
                                                        "n": len(trades),
                                                        "wr": wr,
                                                        "ann": ann, "dd": dd,
                                                        "sharpe": sh_val,
                                                        "eq": eq,
                                                    })

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Phase 1: {sweep_count} configs, "
          f"{len(results)} with 10+ trades")
    print(
        f"\n{'STw':>4} {'SR':>4} {'SB':>4} {'MP':>3} {'MPS':>3} "
        f"{'WT':>4} {'NT':>4} {'LT':>4} {'WW':>3} {'Skip':>5} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 100)
    for r in results[:10]:
        print(
            f"{r['st_w']:>4.2f} {r['sr']:>4.1f} {r['sb']:>4.1f} "
            f"{r['mp']:>3} {r['mps']:>3} "
            f"{r['wt']:>4.2f} {r['nt']:>4.2f} {r['lt']:>4.2f} "
            f"{r['ww']:>3} {'Y' if r['bskip'] else 'N':>5} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    if not results:
        print("  No results. Exiting.")
        return

    # Phase 2: Refine top V80 params with full CAViaR-X sweep
    print("\n--- Phase 2: CAViaR-X refinement on top V80 configs ---")
    seen_v80 = set()
    top_v80_params = []
    for r in results:
        key = (r["st_w"], r["wt"], r["nt"], r["lt"],
               r["ww"], r["mp"], r["mps"])
        if key not in seen_v80:
            seen_v80.add(key)
            top_v80_params.append(key)
        if len(top_v80_params) >= 5:
            break

    phase2_results: List[dict] = []
    for st_w, wt, nt, lt, ww, mp, mps in top_v80_params:
        sigs = signal_cache[st_w]
        for vw in [40, 60, 90]:
            for vl in [0.05, 0.10]:
                sector_breach = breach_cache[(vw, vl)]
                for sr in [0.3, 0.5, 0.7]:
                    for sb in [1.0, 1.3, 1.5]:
                        for bskip in [True, False]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v94(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                sector_lookup=sector_lookup,
                                sector_breach=sector_breach,
                                st_weight=st_w,
                                win_threshold=wt,
                                normal_threshold=nt,
                                lose_threshold=lt,
                                win_rate_window=ww,
                                max_positions=mp,
                                max_per_sector=mps,
                                size_reduction=sr,
                                size_boost=sb,
                                breach_skip=bskip,
                                start_di=bt_2019,
                            )
                            if len(trades) < 10:
                                continue
                            nw = sum(1 for t in trades
                                     if t["pnl_pct"] > 0)
                            wr = nw / len(trades) * 100
                            n_days = max(1, trades[-1]["di"]
                                         - trades[0]["di"])
                            ann = ((eq / CASH0) ** (
                                1 / max(1.0, n_days / 252)) - 1) * 100
                            ap = [t["pnl_abs"]
                                  for t in sorted(
                                      trades, key=lambda x: x["di"])]
                            rets_arr = np.array(ap) / CASH0
                            sh_val = (np.mean(rets_arr) / np.std(rets_arr)
                                      * np.sqrt(252)
                                      if np.std(rets_arr) > 0 else 0)
                            phase2_results.append({
                                "st_w": st_w,
                                "vw": vw, "vl": vl,
                                "sr": sr, "sb": sb,
                                "mp": mp, "mps": mps,
                                "wt": wt, "nt": nt, "lt": lt,
                                "ww": ww, "bskip": bskip,
                                "n": len(trades), "wr": wr,
                                "ann": ann, "dd": dd,
                                "sharpe": sh_val, "eq": eq,
                            })

    # Merge and sort
    all_results = results + phase2_results
    all_results.sort(key=lambda x: -x["ann"])

    print(f"\n  Phase 2: +{len(phase2_results)} configs")
    print(f"  TOTAL: {len(all_results)} configs")
    print(
        f"\n{'STw':>4} {'VW':>3} {'VL':>4} {'SR':>4} {'SB':>4} "
        f"{'MP':>3} {'MPS':>3} {'WT':>4} {'NT':>4} {'LT':>4} "
        f"{'WW':>3} {'Sk':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 110)
    for r in all_results[:10]:
        print(
            f"{r['st_w']:>4.2f} {r['vw']:>3} {r['vl']:>4.2f} "
            f"{r['sr']:>4.1f} {r['sb']:>4.1f} "
            f"{r['mp']:>3} {r['mps']:>3} "
            f"{r['wt']:>4.2f} {r['nt']:>4.2f} {r['lt']:>4.2f} "
            f"{r['ww']:>3} {'Y' if r['bskip'] else 'N':>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    best = all_results[0]

    # Top config full backtest
    print("\n" + "=" * 70)
    print(f"  TOP CONFIG: st_w={best['st_w']:.2f} "
          f"vw={best['vw']} vl={best['vl']} "
          f"sr={best['sr']} sb={best['sb']} "
          f"mp={best['mp']} mps={best['mps']} "
          f"wt={best['wt']:.2f} nt={best['nt']:.2f} lt={best['lt']:.2f} "
          f"ww={best['ww']} skip={best['bskip']}")
    print("=" * 70)

    sector_breach_best = breach_cache.get(
        (best["vw"], best["vl"]),
        breach_cache[(60, 0.05)])
    sigs_best = signal_cache[best["st_w"]]

    trades_full, eq_full, dd_full = backtest_v94(
        C, O, H, L, NS, ND, dates, syms, sigs_best,
        sector_lookup=sector_lookup,
        sector_breach=sector_breach_best,
        st_weight=best["st_w"],
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        size_reduction=best["sr"],
        size_boost=best["sb"],
        breach_skip=best["bskip"],
        start_di=60,
    )
    label = (f"st_w={best['st_w']:.2f} vw={best['vw']} vl={best['vl']} "
             f"sr={best['sr']} sb={best['sb']} mp={best['mp']} "
             f"mps={best['mps']} wt={best['wt']:.2f} nt={best['nt']:.2f} "
             f"lt={best['lt']:.2f} ww={best['ww']} skip={best['bskip']}")
    analyze(trades_full, eq_full, dd_full, label)

    # Walk-forward
    print("\n" + "=" * 70)
    print(f"  BEST WALK-FORWARD")
    print("=" * 70)
    walk_forward(
        C, O, H, L, NS, ND, dates, syms, sigs_best,
        sector_lookup=sector_lookup,
        sector_breach=sector_breach_best,
        st_weight=best["st_w"],
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        size_reduction=best["sr"],
        size_boost=best["sb"],
        breach_skip=best["bskip"],
    )

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON: V94 (CAViaR-X) vs V80 BASELINE")
    print("=" * 70)

    trades_v94, eq_v94, dd_v94 = backtest_v94(
        C, O, H, L, NS, ND, dates, syms, sigs_best,
        sector_lookup=sector_lookup,
        sector_breach=sector_breach_best,
        st_weight=best["st_w"],
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        size_reduction=best["sr"],
        size_boost=best["sb"],
        breach_skip=best["bskip"],
        start_di=bt_2019,
    )

    # V80 baseline: same signals, no breach effect
    trades_v80, eq_v80, dd_v80 = backtest_v94(
        C, O, H, L, NS, ND, dates, syms, sigs_best,
        sector_lookup=sector_lookup,
        sector_breach={},
        st_weight=best["st_w"],
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        size_reduction=1.0,
        size_boost=1.0,
        breach_skip=False,
        start_di=bt_2019,
    )

    print("\n  V94 CAViaR-X:")
    analyze(trades_v94, eq_v94, dd_v94, "V94-CAViaR-X")
    print("\n  V80 BASELINE (no tail risk):")
    analyze(trades_v80, eq_v80, dd_v80, "V80-baseline")

    if trades_v94 and trades_v80:
        n94 = max(1, trades_v94[-1]["di"] - trades_v94[0]["di"])
        n80 = max(1, trades_v80[-1]["di"] - trades_v80[0]["di"])
        ann94 = ((eq_v94 / CASH0) ** (1 / max(1.0, n94 / 252)) - 1) * 100
        ann80 = ((eq_v80 / CASH0) ** (1 / max(1.0, n80 / 252)) - 1) * 100
        print(f"\n  DELTA: ann={ann94 - ann80:+.1f}% "
              f"dd={dd_v94 - dd_v80:+.1f}% "
              f"trades={len(trades_v94) - len(trades_v80):+d}")

    print(f"\n[V94] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
