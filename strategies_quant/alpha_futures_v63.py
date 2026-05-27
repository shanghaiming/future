"""
V63: V61 (ALL-TIME BEST) with Leverage + Contract Specs
========================================================
V61 = V47 sector limit + V43 dynamic mode + 20d multi-TF confirmation
V61 results: Sharpe 6.71, MDD 5.5%, ann +15.7%, 78 trades

V63 adds deployable features on top of V61:
  1. `leverage` parameter (default 1.0) — scales P&L by leverage
  2. Contract count and notional exposure using contract_specs
  3. Leverage sweep: 1.0, 1.5, 2.0, 3.0x with WF validation
  4. Leverage impact reporting on Sharpe/MDD/ann

Signal logic unchanged from V61. Only P&L computation and reporting
are enhanced for deployable sizing.
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
from contract_specs import (
    get_contract_multiplier,
    get_notional_value,
    get_margin_rate,
    get_margin_required,
    get_contracts_for_target,
)

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005

# Short-term (5d) factor weights -- same as V61
ST_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol5d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}

# Medium-term (20d) factor weights -- same as V61
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


def build_multiplier_lookup(syms: List[str]) -> Dict[int, int]:
    """Build a symbol-index to contract multiplier mapping."""
    mult_lookup: Dict[int, int] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        mult_lookup[si] = get_contract_multiplier(base)
    return mult_lookup


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
    print("[V63] Computing raw factors (5d + 20d)...", flush=True)

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
    print("[V63] Computing cross-sectional ranks...", flush=True)

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
    print(f"[V63] Building multi-TF composite (st_w={st_weight:.2f})...",
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
    """Full signal pipeline (same as V61)."""
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


# ---------------------------------------------------------------------------
# V63 backtest: leverage + contract specs integrated
# ---------------------------------------------------------------------------

Position = Tuple[int, int, float, float, float, bool]


def backtest_v63(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    mult_lookup: Dict[int, int],
    st_weight: float = 0.60,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    max_per_sector: int = 1,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
    leverage: float = 1.0,
) -> Tuple[List[dict], float, float, Dict[str, float]]:
    """Backtest V63: V61 signal + leverage + contract specs.

    Returns (trades, equity, max_dd, exposure_stats).
    exposure_stats contains aggregate contract/exposure metrics.
    """
    composite = sigs["composite"]
    ker_regime = sigs["ker_regime"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Position] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []

    # Exposure tracking
    total_notional_exposure: float = 0.0
    total_contracts_traded: int = 0
    exposure_samples: int = 0
    max_notional_pct: float = 0.0

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Position] = []

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        mode_params = get_mode_params(
            mode, normal_threshold, lose_threshold, top_n_winning)
        current_threshold = mode_params["threshold"]
        current_top_n = mode_params["top_n"]
        current_pyramid_ratio = mode_params["pyramid_ratio"]
        current_mode_label = mode_params["mode_label"]

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
                    pnl_raw = (c - ep) / ep - COMM
                    pnl_levered = pnl_raw * leverage
                    profit = equity * alloc * pnl_levered
                    daily_pnl += profit

                    # Contract-level metrics
                    base_sym = _extract_base_symbol(syms[si])
                    multiplier = get_contract_multiplier(base_sym)
                    notional = c * multiplier
                    n_contracts = max(1, int(equity * alloc / notional))

                    is_win = pnl_raw > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl_levered * 100,
                        "pnl_pct_raw": pnl_raw * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop",
                        "pyr": is_pyr,
                        "mode": current_mode_label,
                        "contracts": n_contracts,
                        "notional": notional * n_contracts,
                        "leverage": leverage,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl_raw = (c - ep) / ep - COMM
                    pnl_levered = pnl_raw * leverage
                    profit = equity * alloc * pnl_levered
                    daily_pnl += profit

                    base_sym = _extract_base_symbol(syms[si])
                    multiplier = get_contract_multiplier(base_sym)
                    notional = c * multiplier
                    n_contracts = max(1, int(equity * alloc / notional))

                    is_win = pnl_raw > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl_levered * 100,
                        "pnl_pct_raw": pnl_raw * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold",
                        "pyr": is_pyr,
                        "mode": current_mode_label,
                        "contracts": n_contracts,
                        "notional": notional * n_contracts,
                        "leverage": leverage,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr))

        # Pyramid on day-1 winners
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
                        pyr_alloc = base_alloc * current_pyramid_ratio
                        c_now = C[si, di]
                        atr = compute_atr_at(H, L, C, si, di, start_di)
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

        # Track exposure at close
        if positions and equity > 0:
            day_notional = 0.0
            day_contracts = 0
            for si, edi, ep, sp, alloc, is_pyr in positions:
                c = C[si, di]
                if np.isnan(c):
                    continue
                base_sym = _extract_base_symbol(syms[si])
                multiplier = get_contract_multiplier(base_sym)
                notional_one = c * multiplier
                n_cont = max(1, int(equity * alloc * leverage / notional_one))
                day_notional += notional_one * n_cont
                day_contracts += n_cont
            total_notional_exposure += day_notional
            total_contracts_traded += day_contracts
            exposure_samples += 1
            notional_pct = day_notional / equity * 100
            if notional_pct > max_notional_pct:
                max_notional_pct = notional_pct

        held = {p[0] for p in positions}
        if len(positions) >= current_top_n:
            continue

        # Entry signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < current_threshold:
                continue
            ncf_st = sigs["n_confirm_st"][si, di]
            ncf_mt = sigs["n_confirm_mt"][si, di]
            total_confirm = ncf_st + ncf_mt
            if total_confirm < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(current_top_n, 1)
            candidates.append((composite[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])

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

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl_raw = (c - ep) / ep - COMM
            pnl_levered = pnl_raw * leverage
            equity += equity * alloc * pnl_levered

    # Build exposure stats
    avg_notional = total_notional_exposure / max(exposure_samples, 1)
    avg_contracts = total_contracts_traded / max(exposure_samples, 1)
    exposure_stats: Dict[str, float] = {
        "avg_notional": avg_notional,
        "max_notional_pct": max_notional_pct,
        "avg_contracts": avg_contracts,
        "total_trades": float(len(trades)),
    }

    return trades, equity, max_dd, exposure_stats


def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "", leverage: float = 1.0) -> Optional[dict]:
    """Analyze trades and print summary. Returns metrics dict."""
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t["pnl_pct_raw"] > 0)
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

    total_contracts = sum(t.get("contracts", 0) for t in trades)
    total_notional = sum(t.get("notional", 0) for t in trades)

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} "
        f"lev={leverage:.1f}x"
    )
    print(f"    contracts_total={total_contracts} "
          f"notional_total={total_notional:,.0f}")
    print(f"    modes=[W:{mode_counts['WIN']} N:{mode_counts['NORM']} "
          f"L:{mode_counts['LOSE']}] sectors: {sector_str}")

    yr: Dict[int, dict] = {}
    for t in trades:
        y = t["year"]
        if y not in yr:
            yr[y] = {"n": 0, "w": 0, "pnl": []}
        yr[y]["n"] += 1
        if t["pnl_pct_raw"] > 0:
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
        "leverage": leverage,
    }


def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    mult_lookup: Dict[int, int],
    st_weight: float = 0.60,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    max_per_sector: int = 1,
    leverage: float = 1.0,
) -> List[dict]:
    """Walk-forward validation with leverage."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V63 "
        f"(st_w={st_weight:.2f} wt={win_threshold:.2f} "
        f"nt={normal_threshold:.2f} lt={lose_threshold:.2f} "
        f"ww={win_rate_window} mps={max_per_sector} "
        f"lev={leverage:.1f}x)"
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

        trades, _, _, _ = backtest_v63(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            mult_lookup=mult_lookup,
            st_weight=st_weight,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=atr_stop,
            top_n_winning=top_n_winning,
            max_per_sector=max_per_sector,
            min_confidence=3,
            use_ker_gate=True,
            hold_days=5,
            pyramid_day=1,
            start_di=test_start,
            end_di=test_end_idx + 1,
            leverage=leverage,
        )

        test_trades = [t for t in trades
                       if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct_raw"] > 0)
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
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"modes=[W:{modes['W']} N:{modes['N']} L:{modes['L']}] "
                f"sectors=[{sec_str}]",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct_raw"] > 0)
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
            f"\n  WF TOTAL (lev={leverage:.1f}x): "
            f"{len(all_trades)}t (pyr:{n_pyr}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def leverage_sweep(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    mult_lookup: Dict[int, int],
    best_params: dict,
    bt_2019: int,
    leverages: Optional[List[float]] = None,
) -> List[dict]:
    """Run leverage sweep and print impact table."""
    if leverages is None:
        leverages = [1.0, 1.5, 2.0, 3.0]

    print(f"\n{'=' * 70}")
    print("  LEVERAGE SWEEP (WF 2019-2026)")
    print(
        f"  Params: st_w={best_params['st_w']:.2f} "
        f"wt={best_params['wt']:.2f} nt={best_params['nt']:.2f} "
        f"lt={best_params['lt']:.2f} ww={best_params['ww']} "
        f"mps={best_params['mps']} tnw={best_params['tnw']}"
    )
    print(f"{'=' * 70}")

    sweep_results: List[dict] = []

    for lev in leverages:
        # WF validation per leverage
        years = sorted(set(d.year for d in dates))
        wf_trades: List[dict] = []

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

            trades, _, _, _ = backtest_v63(
                C, O, H, L, NS, ND, dates, syms, sigs,
                sector_lookup=sector_lookup,
                mult_lookup=mult_lookup,
                st_weight=best_params["st_w"],
                win_threshold=best_params["wt"],
                normal_threshold=best_params["nt"],
                lose_threshold=best_params["lt"],
                win_rate_window=best_params["ww"],
                atr_stop=3.0,
                top_n_winning=best_params["tnw"],
                max_per_sector=best_params["mps"],
                min_confidence=3,
                use_ker_gate=True,
                hold_days=5,
                pyramid_day=1,
                start_di=test_start,
                end_di=test_end_idx + 1,
                leverage=lev,
            )
            yr_trades = [t for t in trades
                         if dates[t["di"]].year == test_year]
            wf_trades.extend(yr_trades)

        if not wf_trades:
            print(f"  lev={lev:.1f}x: no trades")
            continue

        # WF metrics
        nw = sum(1 for t in wf_trades if t["pnl_pct_raw"] > 0)
        wr = nw / len(wf_trades) * 100

        # Compute equity curve from WF trades
        eq = CASH0
        peak_wf = eq
        max_dd_wf = 0.0
        sorted_trades = sorted(wf_trades, key=lambda x: x["di"])
        for t in sorted_trades:
            eq += t["pnl_abs"]
            if eq > peak_wf:
                peak_wf = eq
            if peak_wf > 0:
                dd = (peak_wf - eq) / peak_wf * 100
                if dd > max_dd_wf:
                    max_dd_wf = dd

        n_days = max(1, sorted_trades[-1]["di"] - sorted_trades[0]["di"])
        ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100

        ap = [t["pnl_abs"] for t in sorted_trades]
        rets = np.array(ap) / CASH0
        sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
              if np.std(rets) > 0 else 0)

        total_contracts = sum(t.get("contracts", 0) for t in wf_trades)
        total_notional = sum(t.get("notional", 0) for t in wf_trades)
        avg_notional_per_trade = total_notional / max(len(wf_trades), 1)

        # Per-year consistency
        yr_wins = 0
        yr_total = 0
        yr_by_year: Dict[int, List[float]] = defaultdict(list)
        for t in wf_trades:
            yr_by_year[t["year"]].append(t["pnl_pct"])
        for y, pnls in yr_by_year.items():
            yr_total += 1
            if sum(1 for p in pnls if p > 0) / len(pnls) > 0.5:
                yr_wins += 1
        yr_consistency = yr_wins / max(yr_total, 1) * 100

        sweep_results.append({
            "leverage": lev,
            "n": len(wf_trades),
            "wr": wr,
            "ann": ann,
            "dd": max_dd_wf,
            "sharpe": sh,
            "eq": eq,
            "contracts": total_contracts,
            "notional_avg": avg_notional_per_trade,
            "yr_consistency": yr_consistency,
        })

        print(
            f"  lev={lev:.1f}x: {len(wf_trades)}t WR={wr:.1f}% "
            f"ann={ann:+.1f}% DD={max_dd_wf:.1f}% Sh={sh:.2f} "
            f"eq={eq:,.0f} contracts={total_contracts} "
            f"yr_consistency={yr_consistency:.0f}%"
        )

    # Summary table
    if sweep_results:
        print(f"\n  {'='*65}")
        print("  LEVERAGE IMPACT TABLE")
        print(f"  {'='*65}")
        print(
            f"  {'Lev':>5} {'N':>5} {'WR':>6} {'Ann':>8} "
            f"{'DD':>7} {'Sharpe':>8} {'Equity':>12} "
            f"{'YrWin':>6}"
        )
        print(f"  {'-'*65}")
        for r in sweep_results:
            print(
                f"  {r['leverage']:>4.1f}x "
                f"{r['n']:>5} "
                f"{r['wr']:>5.1f}% "
                f"{r['ann']:>+7.1f}% "
                f"{r['dd']:>6.1f}% "
                f"{r['sharpe']:>7.2f} "
                f"{r['eq']:>11,.0f} "
                f"{r['yr_consistency']:>5.0f}%"
            )

        # Impact ratios vs 1.0x baseline
        if len(sweep_results) > 1:
            baseline = sweep_results[0]
            print(f"\n  LEVERAGE IMPACT vs 1.0x BASELINE:")
            print(
                f"  {'Lev':>5} {'Ann_mult':>9} {'DD_mult':>8} "
                f"{'Sharpe_ch':>10} {'Eq_mult':>8}"
            )
            print(f"  {'-'*50}")
            for r in sweep_results:
                ann_mult = r["ann"] / baseline["ann"] if baseline["ann"] != 0 else 0
                dd_mult = r["dd"] / baseline["dd"] if baseline["dd"] != 0 else 0
                sh_ch = r["sharpe"] - baseline["sharpe"]
                eq_mult = r["eq"] / baseline["eq"] if baseline["eq"] != 0 else 0
                print(
                    f"  {r['leverage']:>4.1f}x "
                    f"{ann_mult:>8.2f}x "
                    f"{dd_mult:>7.2f}x "
                    f"{sh_ch:>+9.2f} "
                    f"{eq_mult:>7.2f}x"
                )

    return sweep_results


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V63: V61 (ALL-TIME BEST) + LEVERAGE + CONTRACT SPECS")
    print("  V61 = V47 sector limit + V43 dynamic mode + 20d multi-TF")
    print("  Adds: leverage param, contract multiplier, notional exposure")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}"
    )

    sector_lookup = build_sector_lookup(syms)
    mult_lookup = build_multiplier_lookup(syms)

    # Print contract specs summary
    mult_dist: Dict[int, int] = defaultdict(int)
    for m in mult_lookup.values():
        mult_dist[m] += 1
    print(f"  Multiplier distribution: {dict(sorted(mult_dist.items()))}")

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

    # === 2. Parameter sweep (leverage=1.0 baseline) ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026, lev=1.0x)")
    print("=" * 70)

    results: List[dict] = []

    for st_w in [0.50, 0.55, 0.60, 0.65, 0.70]:
        sigs = signal_cache[st_w]
        for mps in [1, 2]:
            for ww in [10, 15, 20]:
                for wt in [0.55, 0.60, 0.65]:
                    for nt in [0.80, 0.82, 0.85]:
                        for lt in [0.88, 0.90, 0.92]:
                            if lt <= nt:
                                continue
                            for tnw in [2, 3]:
                                trades, eq, dd, _ = backtest_v63(
                                    C, O, H, L, NS, ND, dates, syms, sigs,
                                    sector_lookup=sector_lookup,
                                    mult_lookup=mult_lookup,
                                    st_weight=st_w,
                                    win_threshold=wt,
                                    normal_threshold=nt,
                                    lose_threshold=lt,
                                    win_rate_window=ww,
                                    atr_stop=3.0,
                                    top_n_winning=tnw,
                                    max_per_sector=mps,
                                    start_di=bt_2019,
                                    leverage=1.0,
                                )

                                if len(trades) < 10:
                                    continue

                                nw = sum(
                                    1 for t in trades
                                    if t["pnl_pct_raw"] > 0)
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

                                yr_counts: Dict[int, int] = {}
                                for t in trades:
                                    y = t["year"]
                                    yr_counts[y] = (
                                        yr_counts.get(y, 0) + 1)
                                oos_years = [y for y in yr_counts
                                             if y >= 2019]
                                avg_per_year = (
                                    sum(yr_counts[y] for y in oos_years)
                                    / max(len(oos_years), 1))

                                sec_trades: Dict[str, int] = defaultdict(int)
                                for t in trades:
                                    sec_trades[t.get(
                                        "sector", "OTHER")] += 1
                                max_sec_pct = (
                                    max(sec_trades.values())
                                    / len(trades) * 100)

                                results.append({
                                    "st_w": st_w, "wt": wt,
                                    "nt": nt, "lt": lt,
                                    "ww": ww, "mps": mps,
                                    "tnw": tnw,
                                    "n": len(trades), "wr": wr,
                                    "ann": ann, "dd": dd,
                                    "sharpe": sh_val, "eq": eq,
                                    "avg_yr": avg_per_year,
                                    "max_sec": max_sec_pct,
                                })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n  Evaluated {len(results)} configs with 10+ trades"
    )
    print(
        f"\n{'STw':>4} {'WT':>4} {'NT':>4} {'LT':>4} {'WW':>3} "
        f"{'MPS':>3} {'TNW':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7} {'MaxSec':>7}"
    )
    print("-" * 95)
    for r in results[:30]:
        print(
            f"{r['st_w']:>4.2f} {r['wt']:>4.2f} {r['nt']:>4.2f} "
            f"{r['lt']:>4.2f} {r['ww']:>3} {r['mps']:>3} "
            f"{r['tnw']:>4} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['avg_yr']:>7.1f} {r['max_sec']:>6.1f}%"
        )

    if not results:
        print("  No valid configs found. Exiting.")
        return

    # === 3. Top config: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIG -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    best = results[0]
    best_sigs = signal_cache[best["st_w"]]

    trades_full, eq_full, dd_full, exp_stats = backtest_v63(
        C, O, H, L, NS, ND, dates, syms, best_sigs,
        sector_lookup=sector_lookup,
        mult_lookup=mult_lookup,
        st_weight=best["st_w"],
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        atr_stop=3.0,
        top_n_winning=best["tnw"],
        max_per_sector=best["mps"],
        start_di=60,
        leverage=1.0,
    )
    label = (
        f"st_w={best['st_w']:.2f} wt={best['wt']:.2f} "
        f"nt={best['nt']:.2f} lt={best['lt']:.2f} "
        f"ww={best['ww']} mps={best['mps']} tnw={best['tnw']}"
    )
    print(f"\n  FULL 10Y (lev=1.0x): {label}")
    analyze(trades_full, eq_full, dd_full, "V63-10Y", leverage=1.0)
    print(
        f"    avg_notional/exposure_day: "
        f"{exp_stats['avg_notional']:,.0f} "
        f"max_notional%: {exp_stats['max_notional_pct']:.1f}% "
        f"avg_contracts/day: {exp_stats['avg_contracts']:.1f}"
    )

    # === 4. Walk-forward for best config (1.0x) ===
    print("\n" + "=" * 70)
    print(
        f"  BEST WF (lev=1.0x): st_w={best['st_w']:.2f} "
        f"wt={best['wt']:.2f} nt={best['nt']:.2f} "
        f"lt={best['lt']:.2f} ww={best['ww']} "
        f"mps={best['mps']} tnw={best['tnw']}"
    )
    print("=" * 70)
    walk_forward(
        C, O, H, L, NS, ND, dates, syms,
        best_sigs,
        sector_lookup=sector_lookup,
        mult_lookup=mult_lookup,
        st_weight=best["st_w"],
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        atr_stop=3.0,
        top_n_winning=best["tnw"],
        max_per_sector=best["mps"],
        leverage=1.0,
    )

    # === 5. LEVERAGE SWEEP ===
    sweep_results = leverage_sweep(
        C, O, H, L, NS, ND, dates, syms,
        best_sigs, sector_lookup, mult_lookup,
        best, bt_2019,
        leverages=[1.0, 1.5, 2.0, 3.0],
    )

    # === 6. Detailed WF for recommended leverage ===
    # Pick the best leverage from sweep (highest Sharpe with DD < 15%)
    recommended_lev = 1.0
    viable = [r for r in sweep_results if r["dd"] < 15.0]
    if viable:
        viable.sort(key=lambda x: -x["sharpe"])
        recommended_lev = viable[0]["leverage"]

    print(f"\n{'=' * 70}")
    print(f"  RECOMMENDED LEVERAGE: {recommended_lev:.1f}x")
    print(f"  (highest Sharpe with DD < 15%)")
    print(f"{'=' * 70}")

    if recommended_lev != 1.0:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            best_sigs,
            sector_lookup=sector_lookup,
            mult_lookup=mult_lookup,
            st_weight=best["st_w"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=best["tnw"],
            max_per_sector=best["mps"],
            leverage=recommended_lev,
        )

    # === 7. Contract specs summary for top trades ===
    print(f"\n{'=' * 70}")
    print("  CONTRACT SPECS SUMMARY (lev=1.0x, top trades)")
    print(f"{'=' * 70}")

    trades_2019, _, _, _ = backtest_v63(
        C, O, H, L, NS, ND, dates, syms, best_sigs,
        sector_lookup=sector_lookup,
        mult_lookup=mult_lookup,
        st_weight=best["st_w"],
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        atr_stop=3.0,
        top_n_winning=best["tnw"],
        max_per_sector=best["mps"],
        start_di=bt_2019,
        leverage=1.0,
    )

    sym_stats: Dict[str, dict] = defaultdict(
        lambda: {"n": 0, "contracts": 0, "notional": 0.0})
    for t in trades_2019:
        s = t["sym"]
        sym_stats[s]["n"] += 1
        sym_stats[s]["contracts"] += t.get("contracts", 0)
        sym_stats[s]["notional"] += t.get("notional", 0)

    print(
        f"  {'Symbol':>8} {'Trades':>6} {'Contracts':>10} "
        f"{'Total Notional':>16} {'Multiplier':>10}"
    )
    print(f"  {'-'*60}")
    for sym in sorted(sym_stats.keys(),
                       key=lambda x: -sym_stats[x]["notional"]):
        st = sym_stats[sym]
        base = _extract_base_symbol(sym)
        mult = get_contract_multiplier(base)
        print(
            f"  {sym:>8} {st['n']:>6} {st['contracts']:>10} "
            f"{st['notional']:>15,.0f} {mult:>9}x"
        )

    print(f"\n[V63] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
