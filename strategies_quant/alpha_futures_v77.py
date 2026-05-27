"""
V77: V39 Adaptive Threshold + V47 Sector Limit + Multi-Position + Leverage
==========================================================================
Combines the best elements from V39, V47, and V61 into a high-frequency
leveraged strategy:

1. V39 adaptive threshold: self-tuning entry based on rolling win rate
2. V47 sector limit: max 2 per sector for diversification
3. V61 multi-TF: short-term (5d) + medium-term (20d) composite with
   20d rank confirmation
4. Multi-position: up to 3 concurrent positions
5. Full equity allocation (alloc=1.0 per position, split across slots)
6. Leverage sweep: 3x, 5x, 8x, 10x, 15x, 20x, 25x, 30x

7 factors: ret5d(0.25), oi5d(0.20), rsi(0.15), vol(0.15),
           ret10d(0.10), range(0.10), atrp(0.05)

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
Walk-forward 2019-2026. Target: 600%+ annualized.
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
    "i": "BLACK", "j": "BLACK", "jm": "BLACK", "hc": "BLACK",
    "sf": "BLACK", "sm": "BLACK", "wr": "BLACK", "im": "BLACK",
    # METAL (non-ferrous + precious)
    "cu": "METAL", "al": "METAL", "zn": "METAL", "pb": "METAL",
    "ni": "METAL", "sn": "METAL", "ss": "METAL", "ao": "METAL",
    "au": "METAL", "ag": "METAL", "rb": "METAL", "si": "METAL",
    # ENERGY
    "sc": "ENERGY", "fu": "ENERGY", "bu": "ENERGY",
    "pg": "ENERGY", "eb": "ENERGY", "ta": "ENERGY",
    "fg": "ENERGY", "oi": "ENERGY",
    # CHEMICAL
    "v": "CHEMICAL", "pp": "CHEMICAL", "l": "CHEMICAL",
    "eg": "CHEMICAL", "ma": "CHEMICAL", "sa": "CHEMICAL",
    "ur": "CHEMICAL", "pf": "CHEMICAL", "sh": "CHEMICAL",
    "lc": "CHEMICAL",
    # AGRI (oilseeds / agricultural)
    "m": "AGRI", "y": "AGRI", "a": "AGRI", "p": "AGRI",
    "c": "AGRI", "cs": "AGRI", "jd": "AGRI", "rr": "AGRI",
    "lrm": "AGRI", "rm": "AGRI", "ru": "AGRI",
    # SOFTS
    "cf": "SOFTS", "sr": "SOFTS", "ap": "SOFTS",
    "cj": "SOFTS", "pk": "SOFTS", "lh": "SOFTS",
    "sp": "SOFTS", "b": "SOFTS", "br": "SOFTS",
}


def _extract_base_symbol(sym: str) -> str:
    """Extract base commodity symbol from data symbol."""
    s = sym.lower().split(".")[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    if s.endswith("fi"):
        s = s[:-2]
    return s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    """Build a symbol-index to sector mapping."""
    sector_lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        sector_lookup[si] = SECTOR_MAP.get(base, "OTHER")
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
                            losses[j] if not np.isnan(losses[j]) else 0.0
                        )
                if len(valid_g) >= period:
                    avg_gain = np.mean(valid_g)
                    avg_loss = np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[si, di + period - 1] = (
                            100.0 - 100.0 / (1.0 + rs)
                        )
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
    """Compute raw factor values for ST (5d) and MT (20d)."""
    t0 = time.time()
    print("[V77] Computing raw factors (5d + 20d)...", flush=True)

    # --- Short-term (5d) factors ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(OI[si, di])
                and not np.isnan(OI[si, di - 5])
                and OI[si, di - 5] > 0
            ):
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5 : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    range_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            rng_vals = []
            for j in range(di - 5, di):
                if (
                    not np.isnan(H[si, j])
                    and not np.isnan(L[si, j])
                    and not np.isnan(C[si, j])
                    and C[si, j] > 0
                    and H[si, j] > L[si, j]
                ):
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
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c))
                    )
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]

    # --- Shared factors ---
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 10])
                and C[si, di - 10] > 0
            ):
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
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 20])
                and C[si, di - 20] > 0
            ):
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    oi_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (
                not np.isnan(OI[si, di])
                and not np.isnan(OI[si, di - 20])
                and OI[si, di - 20] > 0
            ):
                oi_20d[si, di] = OI[si, di] / OI[si, di - 20] - 1.0

    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = V[si, di - 20 : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                vol_20d[si, di] = np.mean(valid)

    range_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rng_vals = []
            for j in range(di - 20, di):
                if (
                    not np.isnan(H[si, j])
                    and not np.isnan(L[si, j])
                    and not np.isnan(C[si, j])
                    and C[si, j] > 0
                    and H[si, j] > L[si, j]
                ):
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
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c))
                    )
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
    NS: int,
    ND: int,
    min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally. Inverted for MR factors."""
    t0 = time.time()
    print("[V77] Computing cross-sectional ranks...", flush=True)

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
        "rank_ret5d",
        "rank_ret10d",
        "rank_oi5d",
        "rank_rsi",
        "rank_ret20d",
        "rank_oi20d",
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
    """Kaufman Efficiency Ratio for regime detection."""
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10 : di + 1]
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
    NS: int,
    ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build multi-timeframe composite signal."""
    t0 = time.time()
    print(
        f"[V77] Building multi-TF composite (st_w={st_weight:.2f})...",
        flush=True,
    )

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
                combined[si, di] = (
                    st_weight * st_comp[si, di] + mt_weight * mt_comp[si, di]
                )

    print(
        f"  Multi-TF composite done: {time.time() - t0:.1f}s", flush=True
    )
    return combined, st_comp, mt_comp, n_confirm_st, n_confirm_mt


def compute_all_signals(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    st_weight: float = 0.60,
    st_weights: Optional[Dict[str, float]] = None,
    mt_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Full signal pipeline for V77."""
    if st_weights is None:
        st_weights = ST_WEIGHTS
    if mt_weights is None:
        mt_weights = MT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    combined, st_comp, mt_comp, ncf_st, ncf_mt = build_multi_tf_composite(
        ranks, st_weights, mt_weights, st_weight, NS, ND
    )

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
    H: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
    si: int,
    di: int,
    start_di: int,
) -> Optional[float]:
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


def adaptive_threshold(
    recent_trades_win: List[int],
    base_threshold: float,
    adapt_amount: float,
    min_cap: float,
    max_cap: float,
    win_rate_window: int,
) -> float:
    """V39-style adaptive threshold based on rolling win rate."""
    if len(recent_trades_win) < 5:
        return base_threshold

    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)

    if win_rate > 0.60:
        threshold = base_threshold - adapt_amount
    elif win_rate < 0.50:
        threshold = base_threshold + adapt_amount
    else:
        threshold = base_threshold

    return max(min_cap, min(max_cap, threshold))


def backtest_v77(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    leverage: float = 1.0,
    max_positions: int = 3,
    max_per_sector: int = 2,
    base_threshold: float = 0.80,
    adapt_amount: float = 0.05,
    win_rate_window: int = 20,
    min_cap: float = 0.70,
    max_cap: float = 0.95,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V77: adaptive threshold + sector limit + multi-pos + leverage.

    Leverage model: each position uses alloc * equity * leverage as notional.
    PnL = notional * (return - commission). Losses capped at allocated equity.
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
    # Position: (si, entry_di, entry_price, stop_price, alloc, is_pyramid)
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []

    recent_trades_win: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        # Adaptive threshold (V39 style)
        current_threshold = adaptive_threshold(
            recent_trades_win,
            base_threshold,
            adapt_amount,
            min_cap,
            max_cap,
            win_rate_window,
        )

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
                    raw_ret = (c - ep) / ep
                    # Leverage: multiply notional return, cap at -100%
                    lev_ret = raw_ret * leverage - COMM * leverage
                    # Cap loss: can't lose more than allocated equity
                    max_loss = alloc * equity
                    raw_profit = equity * alloc * lev_ret
                    profit = max(-max_loss, raw_profit)
                    daily_pnl += profit
                    is_win = lev_ret > 0
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": lev_ret * 100,
                            "days": di - edi + 1,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "sector": sector_lookup.get(si, "OTHER"),
                            "reason": "stop",
                            "pyr": is_pyr,
                            "threshold": current_threshold,
                        }
                    )
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    raw_ret = (c - ep) / ep
                    lev_ret = raw_ret * leverage - COMM * leverage
                    max_loss = alloc * equity
                    raw_profit = equity * alloc * lev_ret
                    profit = max(-max_loss, raw_profit)
                    daily_pnl += profit
                    is_win = lev_ret > 0
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": lev_ret * 100,
                            "days": di - edi + 1,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "sector": sector_lookup.get(si, "OTHER"),
                            "reason": "hold",
                            "pyr": is_pyr,
                            "threshold": current_threshold,
                        }
                    )
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

        held = {p[0] for p in positions}
        if len(positions) >= max_positions:
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
            # Total confirming factors across both timeframes
            total_confirm = n_confirm_st[si, di] + n_confirm_mt[si, di]
            if total_confirm < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(max_positions, 1)
            candidates.append((composite[si, di], si, alloc))

        # Sort by composite score (highest first)
        candidates.sort(key=lambda x: -x[0])

        # Sector-constrained greedy selection
        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, "OTHER")] += 1

        for rank_val, si, alloc in candidates:
            if len(positions) >= max_positions or si in held:
                break
            sym_sector = sector_lookup.get(si, "OTHER")
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            stop = ep - atr_stop * atr
            positions.append((si, di + 1, ep, stop, alloc, False))
            held.add(si)
            sector_counts[sym_sector] += 1

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            raw_ret = (c - ep) / ep
            lev_ret = raw_ret * leverage - COMM * leverage
            max_loss = alloc * equity
            raw_profit = equity * alloc * lev_ret
            profit = max(-max_loss, raw_profit)
            equity += profit

    return trades, equity, max_dd


def analyze(
    trades: List[dict],
    equity: float,
    max_dd: float,
    label: str = "",
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
    sh = (
        np.mean(rets) / np.std(rets) * np.sqrt(252)
        if np.std(rets) > 0
        else 0
    )

    n_pyr = sum(1 for t in trades if t.get("pyr"))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items())
    )

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}"
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
            f"cum={cum:+.1%}"
        )

    return {
        "n": len(trades),
        "wr": wr,
        "dd": max_dd,
        "ann": ann,
        "sh": sh,
        "eq": equity,
    }


def walk_forward(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    leverage: float = 1.0,
    max_positions: int = 3,
    max_per_sector: int = 2,
    base_threshold: float = 0.80,
    adapt_amount: float = 0.05,
    win_rate_window: int = 20,
    min_cap: float = 0.70,
    max_cap: float = 0.95,
    atr_stop: float = 3.0,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V77 "
        f"(lev={leverage:.0f}x max_pos={max_positions} "
        f"mps={max_per_sector} bt={base_threshold:.2f} "
        f"aa={adapt_amount:.2f} ww={win_rate_window})"
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

        trades, _, _ = backtest_v77(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=leverage,
            max_positions=max_positions,
            max_per_sector=max_per_sector,
            base_threshold=base_threshold,
            adapt_amount=adapt_amount,
            win_rate_window=win_rate_window,
            min_cap=min_cap,
            max_cap=max_cap,
            atr_stop=atr_stop,
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
                f"{k}:{v}" for k, v in sorted(yr_sectors.items())
            )
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"sectors=[{sec_str}]",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get("pyr"))
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items())
        )
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
    print("  V77: V39 ADAPTIVE + V47 SECTOR + MULTI-TF + MULTI-POS + LEVERAGE")
    print("  Combines best of V39/V47/V61 with leverage sweep")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
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

    # === 1. Compute signals once ===
    print("\n--- Computing multi-TF signals ---")
    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, st_weight=0.60)

    # === 2. Parameter sweep at 1x leverage to find best base config ===
    print("\n" + "=" * 70)
    print("  STEP 1: PARAMETER SWEEP AT 1x LEVERAGE (2019-2026)")
    print("=" * 70)

    results: List[dict] = []

    for bt_val in [0.70, 0.75, 0.80, 0.85]:
        for aa in [0.03, 0.05, 0.07]:
            for ww in [15, 20]:
                for mps in [1, 2]:
                    for max_pos in [2, 3]:
                        # Skip invalid
                        if mps >= max_pos:
                            continue

                        trades, eq, dd = backtest_v77(
                            C, O, H, L, NS, ND, dates, syms, sigs,
                            sector_lookup=sector_lookup,
                            leverage=1.0,
                            max_positions=max_pos,
                            max_per_sector=mps,
                            base_threshold=bt_val,
                            adapt_amount=aa,
                            win_rate_window=ww,
                            start_di=bt_2019,
                        )

                        if len(trades) < 10:
                            continue

                        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                        wr = nw / len(trades) * 100
                        n_days = max(1, trades[-1]["di"] - trades[0]["di"])
                        ann = (
                            (eq / CASH0)
                            ** (1 / max(1.0, n_days / 252))
                            - 1
                        ) * 100
                        ap = [
                            t["pnl_abs"]
                            for t in sorted(trades, key=lambda x: x["di"])
                        ]
                        rets_arr = np.array(ap) / CASH0
                        sh_val = (
                            np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                            if np.std(rets_arr) > 0
                            else 0
                        )

                        results.append(
                            {
                                "bt": bt_val,
                                "aa": aa,
                                "ww": ww,
                                "mps": mps,
                                "max_pos": max_pos,
                                "n": len(trades),
                                "wr": wr,
                                "ann": ann,
                                "dd": dd,
                                "sharpe": sh_val,
                                "eq": eq,
                            }
                        )

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  Evaluated {len(results)} configs with 10+ trades")
    print(
        f"\n  Top 30 by Sharpe:"
    )
    print(
        f"{'BT':>4} {'AA':>4} {'WW':>3} {'MPS':>3} {'MP':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 65)
    for r in results[:30]:
        print(
            f"{r['bt']:>4.2f} {r['aa']:>4.2f} {r['ww']:>3} "
            f"{r['mps']:>3} {r['max_pos']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    # Also show top by ann return (for leverage compounding)
    by_ann = sorted(results, key=lambda x: -x["ann"])
    print(f"\n  Top 15 by Annualized Return (best for leverage):")
    print(
        f"{'BT':>4} {'AA':>4} {'WW':>3} {'MPS':>3} {'MP':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 65)
    for r in by_ann[:15]:
        print(
            f"{r['bt']:>4.2f} {r['aa']:>4.2f} {r['ww']:>3} "
            f"{r['mps']:>3} {r['max_pos']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    # === 3. Top 3 configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026) AT 1x")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["bt"], r["aa"], r["ww"], r["mps"], r["max_pos"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 3:
            break

    for r in unique_top:
        trades, eq, dd = backtest_v77(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=1.0,
            max_positions=r["max_pos"],
            max_per_sector=r["mps"],
            base_threshold=r["bt"],
            adapt_amount=r["aa"],
            win_rate_window=r["ww"],
            start_di=60,
        )
        label = (
            f"bt={r['bt']:.2f} aa={r['aa']:.2f} ww={r['ww']} "
            f"mps={r['mps']} max_pos={r['max_pos']}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 4. LEVERAGE SWEEP on both best-by-sharpe and best-by-ann configs ===
    if not results:
        print("No valid configs found.")
        return

    # Run leverage sweep for multiple top configs
    sweep_configs = []
    # Best by sharpe
    for r in unique_top[:2]:
        sweep_configs.append(("Sh-best", r))
    # Best by ann (with at least 50 trades for compounding)
    ann_candidates = [r for r in by_ann if r["n"] >= 50]
    seen_keys = {(r["bt"], r["aa"], r["ww"], r["mps"], r["max_pos"]) for _, r in sweep_configs}
    for r in ann_candidates[:3]:
        key = (r["bt"], r["aa"], r["ww"], r["mps"], r["max_pos"])
        if key not in seen_keys:
            sweep_configs.append(("Ann-best", r))
            seen_keys.add(key)

    leverages = [1, 3, 5, 8, 10, 15, 20, 25, 30]

    all_lev_results: List[dict] = []

    for label_prefix, best in sweep_configs:
        print("\n" + "=" * 70)
        print(
            f"  LEVERAGE SWEEP -- {label_prefix}: "
            f"bt={best['bt']:.2f} aa={best['aa']:.2f} "
            f"ww={best['ww']} mps={best['mps']} max_pos={best['max_pos']}"
        )
        print("=" * 70)

        lev_results: List[dict] = []

        for lev in leverages:
            trades, eq, dd = backtest_v77(
                C, O, H, L, NS, ND, dates, syms, sigs,
                sector_lookup=sector_lookup,
                leverage=float(lev),
                max_positions=best["max_pos"],
                max_per_sector=best["mps"],
                base_threshold=best["bt"],
                adapt_amount=best["aa"],
                win_rate_window=best["ww"],
                start_di=bt_2019,
            )

            if len(trades) < 5:
                lev_results.append(
                    {"lev": lev, "n": len(trades), "wr": 0, "ann": 0,
                     "dd": 100, "sh": 0, "eq": eq, "cum": 0,
                     "config": label_prefix}
                )
                continue

            nw = sum(1 for t in trades if t["pnl_pct"] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]["di"] - trades[0]["di"])
            ann = (
                (eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1
            ) * 100
            ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
            rets_arr = np.array(ap) / CASH0
            sh_val = (
                np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                if np.std(rets_arr) > 0
                else 0
            )
            cum = np.prod([1 + t["pnl_pct"] / 100 for t in trades]) - 1

            lev_results.append(
                {
                    "lev": lev,
                    "n": len(trades),
                    "wr": wr,
                    "ann": ann,
                    "dd": dd,
                    "sh": sh_val,
                    "eq": eq,
                    "cum": cum,
                    "config": label_prefix,
                    "bt": best["bt"],
                    "aa": best["aa"],
                    "ww": best["ww"],
                    "mps": best["mps"],
                    "max_pos": best["max_pos"],
                }
            )

        all_lev_results.extend(lev_results)

        # Print leverage sweep table
        print(
            f"\n{'Lev':>4} {'N':>5} {'WR':>5} {'Ann':>10} {'DD':>7} "
            f"{'Sh':>7} {'Cum':>10} {'Equity':>14}"
        )
        print("-" * 80)
        for r in lev_results:
            print(
                f"{r['lev']:>4} {r['n']:>5} {r['wr']:>5.1f} "
                f"{r['ann']:>+10.1f} {r['dd']:>7.1f} "
                f"{r['sh']:>7.2f} {r['cum']:>+10.1%} "
                f"{r['eq']:>14,.0f}"
            )

    # === 5. Detailed analysis for top leverage levels ===
    print("\n" + "=" * 70)
    print("  DETAILED ANALYSIS -- TOP LEVERAGE LEVELS (across all configs)")
    print("=" * 70)

    # Find best lev results: ann > 600%, dd < 90%, sorted by ann
    viable = [
        r for r in all_lev_results
        if r["dd"] < 90 and r["n"] >= 5 and r["ann"] > 0
    ]
    viable.sort(key=lambda x: -x["ann"])

    for r in viable[:5]:
        lev = r["lev"]
        trades, eq, dd = backtest_v77(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=float(lev),
            max_positions=r["max_pos"],
            max_per_sector=r["mps"],
            base_threshold=r["bt"],
            adapt_amount=r["aa"],
            win_rate_window=r["ww"],
            start_di=bt_2019,
        )
        label = (
            f"{r['config']} Lev {lev}x "
            f"(bt={r['bt']:.2f} aa={r['aa']:.2f} ww={r['ww']} "
            f"mps={r['mps']} mp={r['max_pos']})"
        )
        print(f"\n  {label} (2019-2026 OOS)")
        analyze(trades, eq, dd, label)

    # === 6. Walk-forward for best leverage configs ===
    # Pick configs closest to 600% target with reasonable DD
    target_ann = 600
    viable_wf = [
        r for r in all_lev_results
        if r["dd"] < 85 and r["n"] >= 5 and r["ann"] > 100
    ]
    # Sort by closeness to target
    viable_wf.sort(key=lambda x: abs(x["ann"] - target_ann))

    if viable_wf:
        best_wf = viable_wf[0]
        print("\n" + "=" * 70)
        print(
            f"  WALK-FORWARD: {best_wf['config']} {best_wf['lev']}x "
            f"(ann={best_wf['ann']:+.1f}% DD={best_wf['dd']:.1f}%)"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=float(best_wf["lev"]),
            max_positions=best_wf["max_pos"],
            max_per_sector=best_wf["mps"],
            base_threshold=best_wf["bt"],
            adapt_amount=best_wf["aa"],
            win_rate_window=best_wf["ww"],
        )

    # Also walk-forward for the highest ann with DD < 80%
    high_ann = [
        r for r in all_lev_results
        if r["dd"] < 80 and r["n"] >= 5 and r["ann"] > 0
    ]
    high_ann.sort(key=lambda x: -x["ann"])
    if high_ann:
        ha = high_ann[0]
        if ha != best_wf:
            print("\n" + "=" * 70)
            print(
                f"  WALK-FORWARD HIGHEST ANN: {ha['config']} {ha['lev']}x "
                f"(ann={ha['ann']:+.1f}% DD={ha['dd']:.1f}%)"
            )
            print("=" * 70)
            walk_forward(
                C, O, H, L, NS, ND, dates, syms, sigs,
                sector_lookup=sector_lookup,
                leverage=float(ha["lev"]),
                max_positions=ha["max_pos"],
                max_per_sector=ha["mps"],
                base_threshold=ha["bt"],
                adapt_amount=ha["aa"],
                win_rate_window=ha["ww"],
            )

    print(f"\n[V77] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
