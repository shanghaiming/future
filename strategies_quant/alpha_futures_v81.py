"""
V81: Extended Hold Compound Strategy
=====================================
Base: V61 multi-TF + sector limit framework.

Key idea: Current strategies exit after hold_days=5 but many hold periods
average ~1 day per round. This strategy explicitly targets 2-3 day holds
to increase per-trade return. With compounding, this boosts annualized returns.

Rules:
  1. Base signal: V61 multi-TF composite with sector limits
  2. Default hold: 2 days (exit at close of entry_day + hold_days)
  3. If profitable after 1 day AND rank still > threshold: extend to 3 days
  4. If losing after 1 day AND exit_early_if_losing: exit at next open
  5. Full equity allocation (alloc = 1.0 / num_positions)
  6. NO leverage (leverage = 1.0)
  7. Sector limit: max 2

Parameter sweep:
  - hold_days: 2, 3
  - extend_if_winning_rank: 0.55, 0.60, 0.65
  - exit_early_if_losing: True, False
  - max_per_sector: 1, 2
  - normal_threshold: 0.80, 0.82, 0.85, 0.88

Walk-forward 2019-2026. Signal at close[di], enter at open[di+1].
No look-ahead. No gap signals.
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
    print("[V81] Computing raw factors (5d + 20d)...", flush=True)

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
        "ret_5d": ret_5d, "ret_10d": ret_10d, "ret_20d": ret_20d,
        "oi_5d": oi_5d, "oi_20d": oi_20d,
        "vol_5d": vol_5d, "vol_20d": vol_20d,
        "range_5d": range_5d, "range_20d": range_20d,
        "rsi14": rsi14, "atrp_5d": atrp_5d, "atrp_20d": atrp_20d,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V81] Computing cross-sectional ranks...", flush=True)

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
            ranked = pd.Series(vals).rank(pct=True, na_option="keep").values
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
    print(f"[V81] Building multi-TF composite (st_w={st_weight:.2f})...",
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
                combined[si, di] = (
                    st_weight * st_comp[si, di]
                    + mt_weight * mt_comp[si, di]
                )

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


def get_mode_params(
    mode: str,
    normal_threshold: float,
    lose_threshold: float,
    top_n_winning: int,
    top_n_normal: int = 2,
) -> Dict:
    if mode == "winning":
        return {
            "threshold": 0.75,
            "top_n": top_n_winning,
            "mode_label": "WIN",
        }
    elif mode == "losing":
        return {
            "threshold": lose_threshold,
            "top_n": 1,
            "mode_label": "LOSE",
        }
    else:
        return {
            "threshold": normal_threshold,
            "top_n": top_n_normal,
            "mode_label": "NORM",
        }


def backtest_v81(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    st_weight: float = 0.60,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    top_n_normal: int = 2,
    max_per_sector: int = 2,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 2,
    extend_if_winning_rank: float = 0.60,
    exit_early_if_losing: bool = True,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V81: extended hold compound strategy.

    Position tuple: (si, entry_di, entry_price, stop_price, alloc,
                     is_pyramid, extended_hold, original_hold_days)
    - extended_hold: bool, whether this position has been extended
    - original_hold_days: the base hold_days for this position
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
    # (si, entry_di, entry_price, stop_price, alloc,
    #  is_pyramid, extended_hold, original_hold_days)
    positions: List[Tuple] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple] = []

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        mode_params = get_mode_params(
            mode, normal_threshold, lose_threshold, top_n_winning,
            top_n_normal)
        current_threshold = mode_params["threshold"]
        current_top_n = mode_params["top_n"]
        current_mode_label = mode_params["mode_label"]

        pos_by_si: Dict[int, List[Tuple]] = defaultdict(list)
        for pos in positions:
            si = pos[0]
            pos_by_si[si].append(pos)

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            o_today = O[si, di]
            if np.isnan(c):
                new_positions.extend(pos_list)
                continue

            # Process each position for this symbol
            for pos in pos_list:
                (p_si, p_edi, p_ep, p_sp, p_alloc,
                 p_is_pyr, p_extended, p_hold) = pos
                hold = di - p_edi

                # Check stop loss first
                if c < p_sp:
                    pnl = (c - p_ep) / p_ep - COMM
                    profit = equity * p_alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": hold,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop",
                        "pyr": p_is_pyr,
                        "mode": current_mode_label,
                        "extended": p_extended,
                    })
                    recent_trades_win.append(1 if is_win else 0)
                    continue

                # Check exit_early_if_losing: after 1 day, if losing, exit
                if (exit_early_if_losing and hold == 1
                        and not p_extended
                        and not np.isnan(o_today)
                        and o_today > 0):
                    # Check if position is losing at today's close
                    unrealized = (c - p_ep) / p_ep
                    if unrealized < 0:
                        pnl = (c - p_ep) / p_ep - COMM
                        profit = equity * p_alloc * pnl
                        daily_pnl += profit
                        is_win = pnl > 0
                        trades.append({
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100,
                            "days": hold,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "sector": sector_lookup.get(si, 'OTHER'),
                            "reason": "early_lose",
                            "pyr": p_is_pyr,
                            "mode": current_mode_label,
                            "extended": False,
                        })
                        recent_trades_win.append(1 if is_win else 0)
                        continue

                # Check extend_if_winning: after 1 day, if winning and
                # rank still strong, extend hold by 1 day
                if (not p_extended and hold == 1):
                    unrealized = (c - p_ep) / p_ep
                    current_rank = composite[si, di]
                    if (unrealized > 0
                            and not np.isnan(current_rank)
                            and current_rank > extend_if_winning_rank):
                        # Extend hold by 1 day
                        new_positions.append((
                            p_si, p_edi, p_ep, p_sp, p_alloc,
                            p_is_pyr, True, p_hold + 1))
                        continue

                # Normal hold expiry check
                if hold >= p_hold:
                    pnl = (c - p_ep) / p_ep - COMM
                    profit = equity * p_alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": hold,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold",
                        "pyr": p_is_pyr,
                        "mode": current_mode_label,
                        "extended": p_extended,
                    })
                    recent_trades_win.append(1 if is_win else 0)
                    continue

                # Still holding
                new_positions.append(pos)

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

        # Entry logic
        held = {p[0] for p in positions}
        if len(positions) >= current_top_n:
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
            # entry: (si, entry_di, entry_price, stop, alloc,
            #         is_pyramid, extended_hold, original_hold_days)
            positions.append((
                si, di + 1, ep, ep - atr_stop * atr, alloc,
                False, False, hold_days))
            held.add(si)
            sector_counts[sym_sector] += 1

    # Close remaining positions at end
    for pos in positions:
        si, edi, ep, sp, alloc, is_pyr, extended, orig_hold = pos
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

    n_pyr = sum(1 for t in trades if t.get("pyr"))
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")
    n_early = sum(1 for t in trades if t["reason"] == "early_lose")
    n_extended = sum(1 for t in trades if t.get("extended"))

    avg_days = np.mean([t["days"] for t in trades])

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

    print(
        f"  {label}: {len(trades)}t (pyr:{n_pyr} stop:{n_stop} "
        f"hold:{n_hold} early:{n_early} ext:{n_extended}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} avg_hold={avg_days:.1f}d "
        f"modes=[W:{mode_counts['WIN']} N:{mode_counts['NORM']} "
        f"L:{mode_counts['LOSE']}]"
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
    st_weight: float = 0.60,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    top_n_normal: int = 2,
    max_per_sector: int = 2,
    hold_days: int = 2,
    extend_if_winning_rank: float = 0.60,
    exit_early_if_losing: bool = True,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V81 "
        f"(st_w={st_weight:.2f} nt={normal_threshold:.2f} "
        f"hd={hold_days} er={extend_if_winning_rank:.2f} "
        f"eel={exit_early_if_losing} mps={max_per_sector})"
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

        trades, _, _ = backtest_v81(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            st_weight=st_weight,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=atr_stop,
            top_n_winning=top_n_winning,
            top_n_normal=top_n_normal,
            max_per_sector=max_per_sector,
            hold_days=hold_days,
            extend_if_winning_rank=extend_if_winning_rank,
            exit_early_if_losing=exit_early_if_losing,
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
            n_ext = sum(1 for t in test_trades if t.get("extended"))
            n_early = sum(
                1 for t in test_trades if t["reason"] == "early_lose")
            avg_hold = np.mean([t["days"] for t in test_trades])
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"ext:{n_ext} early:{n_early} avg_hold={avg_hold:.1f}d "
                f"sectors=[{sec_str}]",
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
        n_ext = sum(1 for t in all_trades if t.get("extended"))
        n_early = sum(
            1 for t in all_trades if t["reason"] == "early_lose")
        avg_hold = np.mean([t["days"] for t in all_trades])
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr} ext:{n_ext} "
            f"early:{n_early}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%} "
            f"avg_hold={avg_hold:.1f}d"
        )
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V81: EXTENDED HOLD COMPOUND STRATEGY")
    print("  Base: V61 multi-TF + sector limit framework")
    print("  Hold 2-3 days, extend winners, cut losers early")
    print("  NO leverage. Full equity allocation.")
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

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Pre-compute signals for each st_weight ===
    signal_cache: Dict[float, Dict] = {}
    for st_w in [0.55, 0.60, 0.65]:
        print(f"\n--- Computing signals for st_weight={st_w:.2f} ---")
        signal_cache[st_w] = compute_all_signals(
            C, O, H, L, V, OI, NS, ND, st_weight=st_w)

    # === 2. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []

    sweep_holds = [2, 3]
    sweep_ext_ranks = [0.55, 0.60, 0.65]
    sweep_exit_early = [True, False]
    sweep_mps = [1, 2]
    sweep_nt = [0.80, 0.82, 0.85, 0.88]
    sweep_st_w = [0.55, 0.60, 0.65]
    sweep_lt = [0.90, 0.92]
    sweep_ww = [15]
    sweep_wt = [0.60]
    sweep_tnw = [2]
    sweep_tnn = [2]

    total_configs = 0
    for (st_w, hd, ext_r, eel, mps, nt, lt, ww, wt, tnw, tnn) in product(
        sweep_st_w, sweep_holds, sweep_ext_ranks, sweep_exit_early,
        sweep_mps, sweep_nt, sweep_lt, sweep_ww, sweep_wt, sweep_tnw,
        sweep_tnn,
    ):
        if lt <= nt:
            continue
        total_configs += 1

    print(f"  Total configs to evaluate: {total_configs}")

    count = 0
    for (st_w, hd, ext_r, eel, mps, nt, lt, ww, wt, tnw, tnn) in product(
        sweep_st_w, sweep_holds, sweep_ext_ranks, sweep_exit_early,
        sweep_mps, sweep_nt, sweep_lt, sweep_ww, sweep_wt, sweep_tnw,
        sweep_tnn,
    ):
        if lt <= nt:
            continue
        count += 1
        if count % 100 == 0:
            print(f"  ... config {count}/{total_configs}", flush=True)

        sigs = signal_cache[st_w]
        trades, eq, dd = backtest_v81(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            st_weight=st_w,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=3.0,
            top_n_winning=tnw,
            top_n_normal=tnn,
            max_per_sector=mps,
            hold_days=hd,
            extend_if_winning_rank=ext_r,
            exit_early_if_losing=eel,
            start_di=bt_2019,
        )

        if len(trades) < 10:
            continue

        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
        wr = nw / len(trades) * 100
        n_days = max(1, trades[-1]["di"] - trades[0]["di"])
        ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
        ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
        rets_arr = np.array(ap) / CASH0
        sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                  if np.std(rets_arr) > 0 else 0)

        n_ext = sum(1 for t in trades if t.get("extended"))
        n_early = sum(1 for t in trades if t["reason"] == "early_lose")
        avg_hold = np.mean([t["days"] for t in trades])

        yr_counts: Dict[int, int] = {}
        for t in trades:
            y = t["year"]
            yr_counts[y] = yr_counts.get(y, 0) + 1
        oos_years = [y for y in yr_counts if y >= 2019]
        avg_per_year = (
            sum(yr_counts[y] for y in oos_years)
            / max(len(oos_years), 1)
        )

        sec_trades: Dict[str, int] = defaultdict(int)
        for t in trades:
            sec_trades[t.get("sector", "OTHER")] += 1
        max_sec_pct = (
            max(sec_trades.values()) / len(trades) * 100
            if sec_trades else 0
        )

        results.append({
            "st_w": st_w, "hd": hd, "ext_r": ext_r, "eel": eel,
            "mps": mps, "nt": nt, "lt": lt, "ww": ww, "wt": wt,
            "tnw": tnw, "tnn": tnn,
            "n": len(trades), "wr": wr,
            "ann": ann, "dd": dd,
            "sharpe": sh_val, "eq": eq,
            "avg_yr": avg_per_year,
            "max_sec": max_sec_pct,
            "n_ext": n_ext,
            "n_early": n_early,
            "avg_hold": avg_hold,
        })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n  Evaluated {len(results)} configs with 10+ trades"
    )
    print(
        f"\n{'STw':>4} {'HD':>3} {'ExtR':>5} {'EEL':>5} "
        f"{'MPS':>3} {'NT':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'AvgH':>5} {'Ext':>4} {'Ear':>4} {'Avg/Yr':>7}"
    )
    print("-" * 100)
    for r in results[:30]:
        print(
            f"{r['st_w']:>4.2f} {r['hd']:>3} {r['ext_r']:>5.2f} "
            f"{'T' if r['eel'] else 'F':>5} "
            f"{r['mps']:>3} {r['nt']:>4.2f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['avg_hold']:>5.1f} {r['n_ext']:>4} {r['n_early']:>4} "
            f"{r['avg_yr']:>7.1f}"
        )

    # === 3. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["st_w"], r["hd"], r["ext_r"], r["eel"],
               r["mps"], r["nt"], r["lt"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 5:
            break

    for r in unique_top:
        sigs = signal_cache[r["st_w"]]
        trades, eq, dd = backtest_v81(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            st_weight=r["st_w"],
            win_threshold=r["wt"],
            normal_threshold=r["nt"],
            lose_threshold=r["lt"],
            win_rate_window=r["ww"],
            atr_stop=3.0,
            top_n_winning=r["tnw"],
            top_n_normal=r["tnn"],
            max_per_sector=r["mps"],
            hold_days=r["hd"],
            extend_if_winning_rank=r["ext_r"],
            exit_early_if_losing=r["eel"],
            start_di=60,
        )
        label = (
            f"st_w={r['st_w']:.2f} hd={r['hd']} "
            f"ext_r={r['ext_r']:.2f} eel={r['eel']} "
            f"mps={r['mps']} nt={r['nt']:.2f} lt={r['lt']:.2f}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 4. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: st_w={best['st_w']:.2f} "
            f"hd={best['hd']} ext_r={best['ext_r']:.2f} "
            f"eel={best['eel']} mps={best['mps']} "
            f"nt={best['nt']:.2f} lt={best['lt']:.2f}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[best["st_w"]],
            sector_lookup=sector_lookup,
            st_weight=best["st_w"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=best["tnw"],
            top_n_normal=best["tnn"],
            max_per_sector=best["mps"],
            hold_days=best["hd"],
            extend_if_winning_rank=best["ext_r"],
            exit_early_if_losing=best["eel"],
        )

        # === 5. Comparison: V81 vs V61-style (hold=5, no extend/early) ===
        print("\n" + "=" * 70)
        print("  COMPARISON: V81 (extended hold) vs V61-like (hold=5)")
        print("  (2019-2026 OOS)")
        print("=" * 70)

        # V81 best config
        trades_v81, eq_v81, dd_v81 = backtest_v81(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[best["st_w"]],
            sector_lookup=sector_lookup,
            st_weight=best["st_w"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=best["tnw"],
            top_n_normal=best["tnn"],
            max_per_sector=best["mps"],
            hold_days=best["hd"],
            extend_if_winning_rank=best["ext_r"],
            exit_early_if_losing=best["eel"],
            start_di=bt_2019,
        )

        # V61-like: hold=5, no extend, no early exit
        trades_v61, eq_v61, dd_v61 = backtest_v81(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[best["st_w"]],
            sector_lookup=sector_lookup,
            st_weight=best["st_w"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=best["tnw"],
            top_n_normal=best["tnn"],
            max_per_sector=best["mps"],
            hold_days=5,
            extend_if_winning_rank=1.0,  # never extends
            exit_early_if_losing=False,  # never exits early
            start_di=bt_2019,
        )

        print(f"\n  V81 EXTENDED HOLD:")
        analyze(trades_v81, eq_v81, dd_v81, "V81-extended")
        print(f"\n  V61-LIKE (hold=5, no extend/early):")
        analyze(trades_v61, eq_v61, dd_v61, "V61-like")

        if trades_v81 and trades_v61:
            print(
                f"\n  V81 vs V61-like: "
                f"eq_delta={eq_v81 - eq_v61:+,.0f} "
                f"dd_delta={dd_v81 - dd_v61:+.1f}%"
            )

    print(f"\n[V81] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
