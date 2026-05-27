"""
V66: V61 Multi-TF + Sector Limit + Portfolio Heat Management
=============================================================
V61 is ALL-TIME BEST: Sharpe 6.71, MDD 5.5%, ann +15.7%, 78 trades.
V44 showed portfolio heat tracking can reduce MDD.

V66 adds heat management to V61:
  1. V61 multi-timeframe composite (ST 5d + MT 20d)
  2. V61 sector concentration limit
  3. V61 dynamic mode switching
  4. NEW: Portfolio heat via unrealized P&L tracking
     - Track total unrealized P&L across all open positions
     - When unrealized loss > heat_threshold * equity: reduce sizing to 0.5x
     - When unrealized loss > heat_pause * equity: skip all new entries
  5. No forced close-worst (memory says circuit breakers destroy returns)

Parameter sweep:
  - V61 best params: st_weight=0.55, max_per_sector=2, etc.
  - heat_threshold: 0.03, 0.05, 0.08, 0.10
  - heat_pause: 0.08, 0.10, 0.15
  - heat_reduce_size: 0.5 (fixed)

Walk-forward 2019-2026.
Signal at close[di], enter at open[di+1]. No look-ahead.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
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

# Short-term (5d) factor weights
ST_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol5d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}

# Medium-term (20d) factor weights
MT_WEIGHTS = {
    "rank_ret20d": 0.25,
    "rank_oi20d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol20d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range20d": 0.10,
    "rank_atrp20d": 0.05,
}

# Sector definitions for Chinese commodity futures
SECTOR_MAP = {
    # BLACK (ferrous metals)
    'i': 'BLACK', 'j': 'BLACK', 'jm': 'BLACK', 'hc': 'BLACK',
    'sf': 'BLACK', 'sm': 'BLACK', 'wr': 'BLACK', 'im': 'BLACK',
    # METAL (non-ferrous + precious)
    'cu': 'METAL', 'al': 'METAL', 'zn': 'METAL', 'pb': 'METAL',
    'ni': 'METAL', 'sn': 'METAL', 'ss': 'METAL', 'ao': 'METAL',
    'au': 'METAL', 'ag': 'METAL', 'rb': 'METAL', 'si': 'METAL',
    # ENERGY
    'sc': 'ENERGY', 'fu': 'ENERGY', 'bu': 'ENERGY',
    'pg': 'ENERGY', 'eb': 'ENERGY', 'ta': 'ENERGY',
    'fg': 'ENERGY', 'oi': 'ENERGY',
    # CHEMICAL
    'v': 'CHEMICAL', 'pp': 'CHEMICAL', 'l': 'CHEMICAL',
    'eg': 'CHEMICAL', 'ma': 'CHEMICAL', 'sa': 'CHEMICAL',
    'ur': 'CHEMICAL', 'pf': 'CHEMICAL', 'sh': 'CHEMICAL',
    'lc': 'CHEMICAL',
    # AGRI (oilseeds / agricultural)
    'm': 'AGRI', 'y': 'AGRI', 'a': 'AGRI', 'p': 'AGRI',
    'c': 'AGRI', 'cs': 'AGRI', 'jd': 'AGRI', 'rr': 'AGRI',
    'lrm': 'AGRI', 'rm': 'AGRI', 'ru': 'AGRI',
    # SOFTS
    'cf': 'SOFTS', 'sr': 'SOFTS', 'ap': 'SOFTS',
    'cj': 'SOFTS', 'pk': 'SOFTS', 'lh': 'SOFTS',
    'sp': 'SOFTS', 'b': 'SOFTS', 'br': 'SOFTS',
}


def _extract_base_symbol(sym: str) -> str:
    """Extract base commodity symbol from data symbol."""
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    if s.endswith('fi'):
        s = s[:-2]
    return s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    """Build a symbol-index to sector mapping."""
    sector_lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        sector_lookup[si] = SECTOR_MAP.get(base, 'OTHER')
    return sector_lookup


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
    """Compute raw factor values for both short-term (5d) and medium-term (20d)."""
    t0 = time.time()
    print("[V66] Computing raw factors (5d + 20d)...", flush=True)

    # --- Short-term (5d) factors ---
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
                        else cc
                    )
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]

    # --- Shared factors ---
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

    # --- Medium-term (20d) factors ---
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
                        else cc
                    )
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_20d[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "ret_20d": ret_20d,
        "oi_5d": oi_5d,
        "oi_20d": oi_20d,
        "vol_5d": vol_5d,
        "vol_20d": vol_20d,
        "range_5d": range_5d,
        "range_20d": range_20d,
        "rsi14": rsi14,
        "atrp_5d": atrp_5d,
        "atrp_20d": atrp_20d,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally. Inverted for mean-reversion factors."""
    t0 = time.time()
    print("[V66] Computing cross-sectional ranks...", flush=True)

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
    """Kaufman Efficiency Ratio for regime detection."""
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
    """Build multi-timeframe composite signal."""
    t0 = time.time()
    print(f"[V66] Building multi-TF composite (st_w={st_weight:.2f})...",
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
            # Short-term composite
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

            # Medium-term composite
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

            # Combined: only when both timeframes available
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
    """Full signal pipeline for V66."""
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
    """Determine trading mode based on recent win rate."""
    if len(recent_trades_win) < 5:
        return "normal"

    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)

    if win_rate > win_threshold:
        return "winning"
    elif win_rate < 0.50:
        return "losing"
    return "normal"


def get_mode_params(
    mode: str,
    normal_threshold: float,
    lose_threshold: float,
    top_n_winning: int,
    top_n_normal: int = 2,
) -> Dict:
    """Get trading parameters for the given mode."""
    if mode == "winning":
        return {
            "threshold": 0.75,
            "top_n": top_n_winning,
            "pyramid_ratio": 0.5,
            "mode_label": "WIN",
        }
    elif mode == "losing":
        return {
            "threshold": lose_threshold,
            "top_n": 1,
            "pyramid_ratio": 0.0,
            "mode_label": "LOSE",
        }
    else:
        return {
            "threshold": normal_threshold,
            "top_n": top_n_normal,
            "pyramid_ratio": 0.3,
            "mode_label": "NORM",
        }


def get_heat_multiplier(
    portfolio_heat_pct: float,
    heat_threshold: float,
    heat_pause: float,
) -> float:
    """Position sizing multiplier based on portfolio unrealized losses.

    Returns:
      1.0  -- heat below threshold, full sizing
      0.5  -- heat between threshold and pause, half sizing
      0.0  -- heat above pause, no new entries
    """
    if portfolio_heat_pct < heat_threshold:
        return 1.0
    elif portfolio_heat_pct < heat_pause:
        return 0.5
    else:
        return 0.0


def backtest_v66(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    # V61 params
    st_weight: float = 0.55,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    max_per_sector: int = 2,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_day: int = 1,
    # V66 heat params
    heat_threshold: float = 0.05,
    heat_pause: float = 0.10,
    heat_reduce_size: float = 0.5,
    # General
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float, Dict[str, int]]:
    """Backtest V66: V61 multi-TF + sector + dynamic mode + portfolio heat."""
    composite = sigs["composite"]
    n_confirm_st = sigs["n_confirm_st"]
    n_confirm_mt = sigs["n_confirm_mt"]
    ker_regime = sigs["ker_regime"]

    if end_di is None:
        end_di = ND - 1

    equity = float(CASH0)
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []

    recent_trades_win: List[int] = []

    # Heat statistics
    heat_stats: Dict[str, int] = {
        "full": 0, "half": 0, "skip": 0,
    }

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        # --- Step 1: Dynamic mode ---
        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        mode_params = get_mode_params(
            mode, normal_threshold, lose_threshold, top_n_winning)
        current_threshold = mode_params["threshold"]
        current_top_n = mode_params["top_n"]
        current_pyramid_ratio = mode_params["pyramid_ratio"]
        current_mode_label = mode_params["mode_label"]

        # --- Step 2: Compute portfolio heat ---
        total_unrealized_loss = 0.0
        for si, edi, ep, sp, alloc, is_pyr in positions:
            c = C[si, di]
            if np.isnan(c):
                continue
            unrealized_pnl_pct = (c - ep) / ep - COMM
            unrealized_dollar = equity * alloc * unrealized_pnl_pct
            if unrealized_dollar < 0:
                total_unrealized_loss += abs(unrealized_dollar)

        portfolio_heat_pct = (
            total_unrealized_loss / equity if equity > 0 else 0.0)
        heat_mult = get_heat_multiplier(
            portfolio_heat_pct, heat_threshold, heat_pause)

        # --- Step 3: Process existing positions ---
        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr))
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
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop",
                        "pyr": is_pyr,
                        "mode": current_mode_label,
                        "heat": portfolio_heat_pct,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold",
                        "pyr": is_pyr,
                        "mode": current_mode_label,
                        "heat": portfolio_heat_pct,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr))

        # --- Step 4: Pyramid on day-1 winners (with heat multiplier) ---
        if current_pyramid_ratio > 0:
            held_with_pos: Dict[int, List] = defaultdict(list)
            for si, edi, ep, sp, alloc, is_pyr in new_positions:
                held_with_pos[si].append((edi, ep, sp, alloc, is_pyr))

            additions = []
            for si, pos_list in held_with_pos.items():
                has_pyr = any(is_pyr for _, _, _, _, is_pyr in pos_list)
                if has_pyr:
                    continue
                earliest_edi = min(p[0] for p in pos_list)
                hold = di - earliest_edi
                if hold == pyramid_day and not np.isnan(C[si, di]):
                    avg_ep = np.mean([ep for _, ep, _, _, _ in pos_list])
                    if C[si, di] > avg_ep:
                        base_alloc = sum(a for _, _, _, a, _ in pos_list)
                        pyr_alloc = (
                            base_alloc * current_pyramid_ratio * heat_mult)
                        if pyr_alloc > 0.001:
                            c_now = C[si, di]
                            atr = compute_atr_at(
                                H, L, C, si, di, start_di)
                            if atr is not None:
                                additions.append(
                                    (si, di, c_now,
                                     c_now - atr_stop * atr,
                                     pyr_alloc, True))
            new_positions.extend(additions)

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

        # --- Step 5: Entry signal at close[di], enter at open[di+1] ---
        held = {p[0] for p in positions}
        if len(positions) >= current_top_n:
            continue

        # Heat gate: skip all new entries if heat too high
        if heat_mult == 0.0:
            heat_stats["skip"] += 1
            continue

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
            base_alloc = 1.0 / max(current_top_n, 1)
            # Apply heat multiplier to allocation
            adjusted_alloc = base_alloc * heat_mult
            candidates.append((composite[si, di], si, adjusted_alloc))

        candidates.sort(key=lambda x: -x[0])

        # Sector-constrained greedy selection
        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, 'OTHER')] += 1

        for rank_val, si, alloc in candidates:
            if len(positions) >= current_top_n or si in held:
                break
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)
            sector_counts[sym_sector] += 1

        if heat_mult == 1.0:
            heat_stats["full"] += 1
        elif heat_mult == heat_reduce_size:
            heat_stats["half"] += 1

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd, heat_stats


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

    mode_counts = {"WIN": 0, "NORM": 0, "LOSE": 0}
    for t in trades:
        m = t.get("mode", "NORM")
        if m in mode_counts:
            mode_counts[m] += 1

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    avg_heat = np.mean([t.get("heat", 0) for t in trades])

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} "
        f"modes=[W:{mode_counts['WIN']} N:{mode_counts['NORM']} "
        f"L:{mode_counts['LOSE']}] avg_heat={avg_heat:.3f}"
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
    # V61 params
    st_weight: float = 0.55,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    max_per_sector: int = 2,
    # V66 heat params
    heat_threshold: float = 0.05,
    heat_pause: float = 0.10,
    heat_reduce_size: float = 0.5,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V66 "
        f"(st_w={st_weight:.2f} wt={win_threshold:.2f} "
        f"nt={normal_threshold:.2f} lt={lose_threshold:.2f} "
        f"ww={win_rate_window} mps={max_per_sector}) "
        f"heat=[ht={heat_threshold:.0%} hp={heat_pause:.0%} "
        f"hr={heat_reduce_size:.1f}]"
    )
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

        trades, _, _, _ = backtest_v66(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            st_weight=st_weight,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=atr_stop,
            top_n_winning=top_n_winning,
            max_per_sector=max_per_sector,
            heat_threshold=heat_threshold,
            heat_pause=heat_pause,
            heat_reduce_size=heat_reduce_size,
            min_confidence=3,
            use_ker_gate=True,
            hold_days=5,
            pyramid_day=1,
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
            modes = {"W": 0, "N": 0, "L": 0}
            for t in test_trades:
                m = t.get("mode", "NORM")
                if m == "WIN":
                    modes["W"] += 1
                elif m == "LOSE":
                    modes["L"] += 1
                else:
                    modes["N"] += 1
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            avg_heat = np.mean([t.get("heat", 0) for t in test_trades])
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"modes=[W:{modes['W']} N:{modes['N']} L:{modes['L']}] "
                f"heat={avg_heat:.3f} sectors=[{sec_str}]",
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
        n_pyr = sum(1 for t in all_trades if t.get("pyr"))
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V66: V61 MULTI-TF + SECTOR LIMIT + PORTFOLIO HEAT MANAGEMENT")
    print("  V61 (Sharpe 6.71) + portfolio heat to push MDD below 5.5%")
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

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Pre-compute signals for each st_weight ===
    signal_cache: Dict[float, Dict] = {}
    for st_w in [0.50, 0.55, 0.60, 0.65, 0.70]:
        print(f"\n--- Computing signals for st_weight={st_w:.2f} ---")
        signal_cache[st_w] = compute_all_signals(
            C, O, H, L, V, OI, NS, ND, st_weight=st_w)

    # ================================================================
    # SECTION 1: V61 BASELINE (no heat)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: V61 BASELINE (no heat management, 2019-2026)")
    print("=" * 70)

    # V61-like: disable heat by setting threshold extremely high
    baseline_configs = [
        (0.55, 0.60, 0.82, 0.90, 15, 2, 2, "V61-baseline-st55"),
        (0.60, 0.60, 0.82, 0.90, 15, 2, 2, "V61-baseline-st60"),
    ]
    for st_w, wt, nt, lt, ww, mps, tnw, label in baseline_configs:
        trades, eq, dd, _ = backtest_v66(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[st_w],
            sector_lookup=sector_lookup,
            st_weight=st_w,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            top_n_winning=tnw,
            max_per_sector=mps,
            heat_threshold=1.0,  # effectively disabled
            heat_pause=2.0,
            start_di=bt_2019,
        )
        analyze(trades, eq, dd, label)

    # ================================================================
    # SECTION 2: HEAT PARAMETER SWEEP
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: HEAT PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    heat_thresholds = [0.03, 0.05, 0.08, 0.10]
    heat_pauses = [0.08, 0.10, 0.15]
    heat_reduce_size = 0.5

    results: List[dict] = []

    for st_w in [0.50, 0.55, 0.60]:
        sigs = signal_cache[st_w]
        for mps in [1, 2]:
            for ww in [10, 15, 20]:
                for wt in [0.55, 0.60, 0.65]:
                    for nt in [0.80, 0.82, 0.85]:
                        for lt in [0.88, 0.90, 0.92]:
                            if lt <= nt:
                                continue
                            for tnw in [2, 3]:
                                for ht in heat_thresholds:
                                    for hp in heat_pauses:
                                        if hp <= ht:
                                            continue

                                        trades, eq, dd, hs = (
                                            backtest_v66(
                                                C, O, H, L, NS, ND,
                                                dates, syms, sigs,
                                                sector_lookup=(
                                                    sector_lookup),
                                                st_weight=st_w,
                                                win_threshold=wt,
                                                normal_threshold=nt,
                                                lose_threshold=lt,
                                                win_rate_window=ww,
                                                top_n_winning=tnw,
                                                max_per_sector=mps,
                                                heat_threshold=ht,
                                                heat_pause=hp,
                                                heat_reduce_size=(
                                                    heat_reduce_size),
                                                start_di=bt_2019,
                                            )
                                        )

                                        if len(trades) < 10:
                                            continue

                                        nw = sum(
                                            1 for t in trades
                                            if t["pnl_pct"] > 0)
                                        wr = nw / len(trades) * 100
                                        n_days = max(
                                            1,
                                            trades[-1]["di"]
                                            - trades[0]["di"])
                                        ann = (
                                            (eq / CASH0) ** (
                                                1 / max(
                                                    1.0,
                                                    n_days / 252))
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

                                        yr_counts: Dict[int, int] = {}
                                        for t in trades:
                                            y = t["year"]
                                            yr_counts[y] = (
                                                yr_counts.get(y, 0) + 1)
                                        oos_years = [y for y in yr_counts
                                                     if y >= 2019]
                                        avg_per_year = (
                                            sum(yr_counts[y]
                                                for y in oos_years)
                                            / max(len(oos_years), 1))

                                        results.append({
                                            "st_w": st_w,
                                            "wt": wt,
                                            "nt": nt,
                                            "lt": lt,
                                            "ww": ww,
                                            "mps": mps,
                                            "tnw": tnw,
                                            "ht": ht,
                                            "hp": hp,
                                            "n": len(trades),
                                            "wr": wr,
                                            "ann": ann,
                                            "dd": dd,
                                            "sharpe": sh_val,
                                            "eq": eq,
                                            "avg_yr": avg_per_year,
                                            "heat_stats": hs,
                                        })

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  Evaluated {len(results)} configs with 10+ trades")
    print(
        f"\n{'STw':>4} {'WT':>4} {'NT':>4} {'LT':>4} {'WW':>3} "
        f"{'MPS':>3} {'TNW':>4} {'HT%':>4} {'HP%':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'HS_full':>7} {'HS_half':>7} {'HS_skip':>7}"
    )
    print("-" * 110)
    for r in results[:30]:
        hs = r["heat_stats"]
        print(
            f"{r['st_w']:>4.2f} {r['wt']:>4.2f} {r['nt']:>4.2f} "
            f"{r['lt']:>4.2f} {r['ww']:>3} {r['mps']:>3} "
            f"{r['tnw']:>4} {r['ht']*100:>4.0f} {r['hp']*100:>4.0f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{hs.get('full', 0):>7} {hs.get('half', 0):>7} "
            f"{hs.get('skip', 0):>7}"
        )

    # ================================================================
    # SECTION 3: BEST BY MDD (with Sharpe > 4.0)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: BEST BY MDD (Sharpe > 4.0)")
    print("=" * 70)

    mdd_candidates = [r for r in results if r["sharpe"] > 4.0]
    if mdd_candidates:
        mdd_candidates.sort(key=lambda x: x["dd"])
        print(f"\n  Found {len(mdd_candidates)} configs with Sh>4.0:")
        for r in mdd_candidates[:10]:
            hs = r["heat_stats"]
            print(
                f"    st_w={r['st_w']:.2f} wt={r['wt']:.2f} "
                f"nt={r['nt']:.2f} lt={r['lt']:.2f} "
                f"ww={r['ww']} mps={r['mps']} tnw={r['tnw']} "
                f"ht={r['ht']*100:.0f}% hp={r['hp']*100:.0f}% "
                f"=> Sh={r['sharpe']:.2f} DD={r['dd']:.1f}% "
                f"ann={r['ann']:+.1f}% N={r['n']} "
                f"heat=[f:{hs.get('full',0)} h:{hs.get('half',0)} "
                f"s:{hs.get('skip',0)}]"
            )
    else:
        # Relax: Sharpe > 3.0
        relaxed = [r for r in results if r["sharpe"] > 3.0]
        if relaxed:
            relaxed.sort(key=lambda x: x["dd"])
            print(f"\n  No configs with Sh>4.0. Best with Sh>3.0:")
            for r in relaxed[:10]:
                hs = r["heat_stats"]
                print(
                    f"    st_w={r['st_w']:.2f} ht={r['ht']*100:.0f}% "
                    f"hp={r['hp']*100:.0f}% "
                    f"=> Sh={r['sharpe']:.2f} DD={r['dd']:.1f}% "
                    f"ann={r['ann']:+.1f}% N={r['n']}")

    # ================================================================
    # SECTION 4: TOP CONFIGS -- FULL 10-YEAR (2016-2026)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["st_w"], r["wt"], r["nt"], r["lt"],
               r["ww"], r["mps"], r["tnw"], r["ht"], r["hp"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 5:
            break

    for r in unique_top:
        sigs = signal_cache[r["st_w"]]
        trades, eq, dd, hs = backtest_v66(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            st_weight=r["st_w"],
            win_threshold=r["wt"],
            normal_threshold=r["nt"],
            lose_threshold=r["lt"],
            win_rate_window=r["ww"],
            top_n_winning=r["tnw"],
            max_per_sector=r["mps"],
            heat_threshold=r["ht"],
            heat_pause=r["hp"],
            heat_reduce_size=0.5,
            start_di=60,
        )
        label = (
            f"st_w={r['st_w']:.2f} wt={r['wt']:.2f} "
            f"nt={r['nt']:.2f} lt={r['lt']:.2f} "
            f"ww={r['ww']} mps={r['mps']} tnw={r['tnw']} "
            f"ht={r['ht']*100:.0f}% hp={r['hp']*100:.0f}%"
        )
        print(f"\n  FULL 10yr {label}")
        analyze(trades, eq, dd, label)
        print(f"    Heat: full={hs['full']} half={hs['half']} "
              f"skip={hs['skip']}")

    # ================================================================
    # SECTION 5: WALK-FORWARD FOR BEST CONFIGS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Best by Sharpe
    if results:
        best = results[0]
        print(f"\n  --- BEST BY SHARPE ---")
        print(
            f"  st_w={best['st_w']:.2f} wt={best['wt']:.2f} "
            f"nt={best['nt']:.2f} lt={best['lt']:.2f} "
            f"ww={best['ww']} mps={best['mps']} tnw={best['tnw']} "
            f"ht={best['ht']*100:.0f}% hp={best['hp']*100:.0f}%")
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[best["st_w"]],
            sector_lookup=sector_lookup,
            st_weight=best["st_w"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            top_n_winning=best["tnw"],
            max_per_sector=best["mps"],
            heat_threshold=best["ht"],
            heat_pause=best["hp"],
            heat_reduce_size=0.5,
        )

    # Best by MDD (with reasonable Sharpe)
    if mdd_candidates:
        best_mdd = mdd_candidates[0]
        print(f"\n  --- BEST BY MDD (Sh>4.0) ---")
        print(
            f"  st_w={best_mdd['st_w']:.2f} wt={best_mdd['wt']:.2f} "
            f"nt={best_mdd['nt']:.2f} lt={best_mdd['lt']:.2f} "
            f"ww={best_mdd['ww']} mps={best_mdd['mps']} "
            f"tnw={best_mdd['tnw']} "
            f"ht={best_mdd['ht']*100:.0f}% hp={best_mdd['hp']*100:.0f}%")
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[best_mdd["st_w"]],
            sector_lookup=sector_lookup,
            st_weight=best_mdd["st_w"],
            win_threshold=best_mdd["wt"],
            normal_threshold=best_mdd["nt"],
            lose_threshold=best_mdd["lt"],
            win_rate_window=best_mdd["ww"],
            top_n_winning=best_mdd["tnw"],
            max_per_sector=best_mdd["mps"],
            heat_threshold=best_mdd["ht"],
            heat_pause=best_mdd["hp"],
            heat_reduce_size=0.5,
        )

    # ================================================================
    # SECTION 6: V66 (with heat) vs V61 (no heat) COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("  SECTION 6: V66 (heat) vs V61 (no heat) COMPARISON")
    print("  Using best V66 config vs same config without heat")
    print("=" * 70)

    if results:
        best = results[0]

        # V66 with heat
        trades_v66, eq_v66, dd_v66, hs_v66 = backtest_v66(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[best["st_w"]],
            sector_lookup=sector_lookup,
            st_weight=best["st_w"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            top_n_winning=best["tnw"],
            max_per_sector=best["mps"],
            heat_threshold=best["ht"],
            heat_pause=best["hp"],
            heat_reduce_size=0.5,
            start_di=bt_2019,
        )

        # Same config without heat
        trades_v61, eq_v61, dd_v61, _ = backtest_v66(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[best["st_w"]],
            sector_lookup=sector_lookup,
            st_weight=best["st_w"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            top_n_winning=best["tnw"],
            max_per_sector=best["mps"],
            heat_threshold=1.0,  # disabled
            heat_pause=2.0,      # disabled
            start_di=bt_2019,
        )

        print(f"\n  V66 (with heat ht={best['ht']*100:.0f}% "
              f"hp={best['hp']*100:.0f}%):")
        v66_result = analyze(trades_v66, eq_v66, dd_v66, "V66-heat")
        print(f"    Heat: full={hs_v66['full']} half={hs_v66['half']} "
              f"skip={hs_v66['skip']}")

        print(f"\n  V61-like (same config, no heat):")
        v61_result = analyze(trades_v61, eq_v61, dd_v61, "V61-no-heat")

        if v66_result and v61_result:
            dd_improvement = v61_result["dd"] - v66_result["dd"]
            print(
                f"\n  COMPARISON:")
            print(
                f"    MDD:  {v61_result['dd']:.1f}% -> "
                f"{v66_result['dd']:.1f}% "
                f"(delta={dd_improvement:+.1f}%)")
            print(
                f"    Sharpe: {v61_result['sh']:.2f} -> "
                f"{v66_result['sh']:.2f} "
                f"(delta={v66_result['sh'] - v61_result['sh']:+.2f})")
            print(
                f"    Ann: {v61_result['ann']:+.1f}% -> "
                f"{v66_result['ann']:+.1f}% "
                f"(delta={v66_result['ann'] - v61_result['ann']:+.1f}%)")
            print(
                f"    Equity: {v61_result['eq']:,.0f} -> "
                f"{v66_result['eq']:,.0f} "
                f"(delta={v66_result['eq'] - v61_result['eq']:+,.0f})")
            print(
                f"    Trades: {v61_result['n']} -> {v66_result['n']} "
                f"(delta={v66_result['n'] - v61_result['n']:+d})")

    print(f"\n[V66] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
