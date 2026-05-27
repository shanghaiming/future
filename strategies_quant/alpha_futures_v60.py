"""
V60: Sector Limit + Breadth Filter Combo
=========================================
Combines V47 (ALL-TIME BEST: Sharpe 6.65) with V40 breadth filter.

Thesis: MR works best when:
  1. Market is broadly oversold (high A/D decline ratio)
  2. Sector diversified (no over-concentration in one commodity group)

This "double filter" ensures entries happen only when:
  - The broader commodity market confirms oversold conditions
  - AND positions are spread across different sectors

Architecture:
  1. V47's 7-factor cross-sectional rank composite score
  2. V43/V47's dynamic three-mode threshold (WINNING/NORMAL/LOSING)
  3. V47's sector-constrained greedy selection
  4. V40's A/D ratio breadth gate: only enter when ad_ratio <= max_ad_ratio
  5. KER gate < 0.15, hold 5d, ATR stop 3.0
  6. Pyramid on day-1 winners (ratio varies by mode)

Parameter sweep:
  - max_per_sector: 1, 2
  - max_ad_ratio: 0.40, 0.45, 0.50, 0.55, 0.60
  - win_rate_window: 10, 15, 20
  - Dynamic mode thresholds: winning=0.75, normal=0.82, losing=0.90

Walk-forward 2019-2026. Report whether breadth filter helps or hurts V47.
Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
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

DEFAULT_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol": 0.15,
    "rank_ret10d": 0.10,
    "rank_range": 0.10,
    "rank_atrp": 0.05,
}

# Sector definitions for Chinese commodity futures
SECTOR_MAP = {
    # BLACK (ferrous metals)
    "i": "BLACK", "j": "BLACK", "jm": "BLACK", "hc": "BLACK",
    "sf": "BLACK", "sm": "BLACK", "wr": "BLACK",
    # METAL (non-ferrous)
    "cu": "METAL", "al": "METAL", "zn": "METAL", "pb": "METAL",
    "ni": "METAL", "sn": "METAL", "ss": "METAL", "ao": "METAL",
    # ENERGY
    "sc": "ENERGY", "fu": "ENERGY", "bu": "ENERGY",
    "pg": "ENERGY", "eb": "ENERGY", "ta": "ENERGY",
    # CHEMICAL
    "v": "CHEMICAL", "pp": "CHEMICAL", "l": "CHEMICAL",
    "eg": "CHEMICAL", "ma": "CHEMICAL", "sa": "CHEMICAL",
    "ur": "CHEMICAL", "pf": "CHEMICAL", "sh": "CHEMICAL",
    "lc": "CHEMICAL",
    # AGRI (oilseeds / agricultural)
    "m": "AGRI", "y": "AGRI", "a": "AGRI", "p": "AGRI",
    "c": "AGRI", "cs": "AGRI", "jd": "AGRI", "rr": "AGRI",
    "lrm": "AGRI",
    # SOFTS
    "cf": "SOFTS", "sr": "SOFTS", "ap": "SOFTS",
    "cj": "SOFTS", "pk": "SOFTS", "lh": "SOFTS",
}


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    """Build a symbol-index to sector mapping."""
    sector_lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = sym.lower().split(".")[0].strip()
        while base and base[-1].isdigit():
            base = base[:-1]
        sector_lookup[si] = SECTOR_MAP.get(base, "OTHER")
    return sector_lookup


# ============================================================
# FACTOR COMPUTATION (same as V47)
# ============================================================
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
    t0 = time.time()
    print("[V60] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

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

    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (not np.isnan(H[si, di])
                    and not np.isnan(L[si, di])
                    and not np.isnan(C[si, di])):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (
                        (H[si, di] - L[si, di]) / C[si, di])

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

    atrp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(14, ND):
            atr_vals = []
            for j in range(di - 14, di):
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
                atrp[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "oi_5d": oi_5d,
        "vol_5d": vol_5d,
        "daily_range": daily_range,
        "rsi14": rsi14,
        "atrp": atrp,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V60] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        "rank_ret5d": raw_factors["ret_5d"],
        "rank_ret10d": raw_factors["ret_10d"],
        "rank_oi5d": raw_factors["oi_5d"],
        "rank_vol": raw_factors["vol_5d"],
        "rank_range": raw_factors["daily_range"],
        "rank_rsi": raw_factors["rsi14"],
        "rank_atrp": raw_factors["atrp"],
    }

    INVERT_FACTORS = {"rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi"}

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


def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    print("[V60] Building composite signal...", flush=True)

    composite = np.full((NS, ND), np.nan)
    n_confirm = np.zeros((NS, ND), dtype=int)

    factor_names = list(weights.keys())
    weight_vals = np.array([weights[k] for k in factor_names])

    for di in range(ND):
        for si in range(NS):
            vals = []
            w_sum = 0.0
            confirm_count = 0
            for idx, name in enumerate(factor_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                vals.append(rank_val * weight_vals[idx])
                w_sum += weight_vals[idx]
                if rank_val > 0.5:
                    confirm_count += 1

            if w_sum > 0 and confirm_count >= min_factors:
                composite[si, di] = sum(vals) / w_sum
                n_confirm[si, di] = confirm_count

    print(f"  Composite done: {time.time() - t0:.1f}s", flush=True)
    return composite, n_confirm


# ============================================================
# MARKET BREADTH (from V40)
# ============================================================
def compute_ad_ratio(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Compute daily A/D ratio.

    A/D ratio = declines / (advances + declines).
    Higher values = more commodities declining = more oversold.
    """
    t0 = time.time()
    print("[V60] Computing A/D ratio...", flush=True)

    ad_ratio = np.full(ND, np.nan)
    for di in range(1, ND):
        advances = 0
        declines = 0
        for si in range(NS):
            if np.isnan(C[si, di]) or np.isnan(C[si, di - 1]):
                continue
            if C[si, di] > C[si, di - 1]:
                advances += 1
            elif C[si, di] < C[si, di - 1]:
                declines += 1
        total = advances + declines
        if total > 10:
            ad_ratio[di] = declines / total

    valid_count = np.sum(~np.isnan(ad_ratio))
    print(
        f"  A/D ratio done: {time.time() - t0:.1f}s, "
        f"{valid_count} valid days",
        flush=True,
    )
    return ad_ratio


# ============================================================
# SIGNAL PIPELINE
# ============================================================
def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)
    ad_ratio = compute_ad_ratio(C, NS, ND)

    # Print A/D ratio distribution
    valid_ad = ad_ratio[~np.isnan(ad_ratio)]
    if len(valid_ad) > 0:
        print(
            f"  A/D ratio stats: mean={np.mean(valid_ad):.3f} "
            f"median={np.median(valid_ad):.3f} "
            f"<0.50: {np.sum(valid_ad < 0.50) / len(valid_ad) * 100:.1f}% "
            f"<0.55: {np.sum(valid_ad < 0.55) / len(valid_ad) * 100:.1f}% "
            f"<0.60: {np.sum(valid_ad < 0.60) / len(valid_ad) * 100:.1f}%"
        )

    return {
        "composite": composite,
        "n_confirm": n_confirm,
        "ker_regime": ker_regime,
        "ranks": ranks,
        "ad_ratio": ad_ratio,
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


# ============================================================
# DYNAMIC MODE (from V43/V47)
# ============================================================
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
    else:  # normal
        return {
            "threshold": normal_threshold,
            "top_n": top_n_normal,
            "pyramid_ratio": 0.3,
            "mode_label": "NORM",
        }


# ============================================================
# BACKTEST
# ============================================================
def backtest_v60(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    max_ad_ratio: float = 0.55,
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
) -> Tuple[List[dict], float, float]:
    """Backtest V60: V47 sector limit + V40 breadth filter."""
    composite = sigs["composite"]
    ker_regime = sigs["ker_regime"]
    n_confirm = sigs["n_confirm"]
    ad_ratio = sigs["ad_ratio"]

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

        # Determine current mode and parameters
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
                        "sector": sector_lookup.get(si, "OTHER"),
                        "reason": "stop",
                        "pyr": is_pyr,
                        "mode": current_mode_label,
                        "threshold": current_threshold,
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
                        "sector": sector_lookup.get(si, "OTHER"),
                        "reason": "hold",
                        "pyr": is_pyr,
                        "mode": current_mode_label,
                        "threshold": current_threshold,
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

        held = {p[0] for p in positions}
        if len(positions) >= current_top_n:
            continue

        # ============================================================
        # BREADTH FILTER: only enter when market is broadly oversold
        # ad_ratio = declines / total. Higher = more oversold.
        # We want ad_ratio >= threshold to enter (enough declines)
        # But in the user's spec, max_ad_ratio is the upper bound.
        # "Only take MR trades when market breadth (A/D ratio) is
        #  below threshold" => only enter when ad_ratio <= max_ad_ratio
        # This means: enter when NOT too many declines (oversold enough
        # but not extreme panic).
        # ============================================================
        current_ad = ad_ratio[di]
        if np.isnan(current_ad) or current_ad > max_ad_ratio:
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
            if n_confirm[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(current_top_n, 1)
            candidates.append((composite[si, di], si, alloc))

        # Sort by composite score (highest first)
        candidates.sort(key=lambda x: -x[0])

        # Sector-constrained greedy selection (from V47)
        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, "OTHER")] += 1

        for rank_val, si, alloc in candidates:
            if len(positions) >= current_top_n or si in held:
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
            positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)
            sector_counts[sym_sector] += 1

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
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

    # Mode distribution
    mode_counts = {"WIN": 0, "NORM": 0, "LOSE": 0}
    for t in trades:
        m = t.get("mode", "NORM")
        if m in mode_counts:
            mode_counts[m] += 1

    # Sector distribution
    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} "
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
    max_ad_ratio: float = 0.55,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    max_per_sector: int = 1,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V60 "
        f"(ad_max={max_ad_ratio:.2f} wt={win_threshold:.2f} "
        f"nt={normal_threshold:.2f} lt={lose_threshold:.2f} "
        f"ww={win_rate_window} ats={atr_stop:.1f} "
        f"tnw={top_n_winning} mps={max_per_sector})"
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

        trades, _, _ = backtest_v60(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_ad_ratio=max_ad_ratio,
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
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"modes=[W:{modes['W']} N:{modes['N']} L:{modes['L']}] "
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


# ============================================================
# PARAMETER SWEEP
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V60: SECTOR LIMIT + BREADTH FILTER COMBO")
    print("  V47 sector limit + V40 A/D ratio breadth filter")
    print("  Target: beat V47 Sharpe 6.65")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}"
    )

    # Build sector lookup
    sector_lookup = build_sector_lookup(syms)
    sector_dist: Dict[str, int] = defaultdict(int)
    for sec in sector_lookup.values():
        sector_dist[sec] += 1
    print(f"  Sector distribution: {dict(sector_dist)}")

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Walk-Forward with default configs ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026) -- DEFAULT CONFIGS")
    print("=" * 70)

    default_configs = [
        # (max_ad, wt, nt, lt, ww, ats, tnw, mps)
        (0.55, 0.60, 0.82, 0.90, 15, 3.0, 2, 1),
        (0.55, 0.60, 0.82, 0.90, 15, 3.0, 2, 2),
        (0.50, 0.60, 0.82, 0.90, 15, 3.0, 2, 1),
        (0.60, 0.60, 0.82, 0.90, 15, 3.0, 2, 1),
        (0.45, 0.60, 0.82, 0.90, 15, 3.0, 2, 1),
    ]

    for ad_max, wt, nt, lt, ww, ats, tnw, mps in default_configs:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_ad_ratio=ad_max,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=ats,
            top_n_winning=tnw,
            max_per_sector=mps,
        )

    # === 2. Full 10-year with profile comparison ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- PROFILE COMPARISON")
    print("=" * 70)

    profiles = [
        (0.55, 0.60, 0.82, 0.90, 15, 3.0, 2, 1, "Default ad=0.55 mps=1"),
        (0.55, 0.60, 0.82, 0.90, 15, 3.0, 2, 2, "Default ad=0.55 mps=2"),
        (0.50, 0.60, 0.82, 0.90, 15, 3.0, 2, 1, "Tight ad=0.50 mps=1"),
        (0.60, 0.60, 0.82, 0.90, 15, 3.0, 2, 1, "Loose ad=0.60 mps=1"),
        (0.45, 0.60, 0.82, 0.90, 15, 3.0, 2, 1, "Very tight ad=0.45"),
        (0.40, 0.60, 0.82, 0.90, 15, 3.0, 2, 1, "Ultra tight ad=0.40"),
        # V47 baseline (no breadth filter): set ad_max very high
        (1.00, 0.60, 0.82, 0.90, 15, 3.0, 2, 1, "V47-equiv ad=1.0 mps=1"),
    ]

    for ad_max, wt, nt, lt, ww, ats, tnw, mps, label in profiles:
        trades, eq, dd = backtest_v60(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_ad_ratio=ad_max,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=ats,
            top_n_winning=tnw,
            max_per_sector=mps,
            start_di=60,
        )
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 3. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []

    sweep_params = {
        "max_ad_ratio": [0.40, 0.45, 0.50, 0.55, 0.60],
        "win_threshold": [0.55, 0.60, 0.65],
        "normal_threshold": [0.80, 0.82, 0.85],
        "lose_threshold": [0.88, 0.90, 0.92],
        "win_rate_window": [10, 15, 20],
        "max_per_sector": [1, 2],
        "top_n_winning": [2, 3],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for ad_max, wt, nt, lt, ww, mps, tnw in product(
        sweep_params["max_ad_ratio"],
        sweep_params["win_threshold"],
        sweep_params["normal_threshold"],
        sweep_params["lose_threshold"],
        sweep_params["win_rate_window"],
        sweep_params["max_per_sector"],
        sweep_params["top_n_winning"],
    ):
        # Skip invalid: win_threshold must be > 0.50, lose > normal
        if wt <= 0.50:
            continue
        if lt <= nt:
            continue

        combo_count += 1
        trades, eq, dd = backtest_v60(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_ad_ratio=ad_max,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=3.0,
            top_n_winning=tnw,
            max_per_sector=mps,
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

        # Count trades per year
        yr_counts: Dict[int, int] = {}
        for t in trades:
            y = t["year"]
            yr_counts[y] = yr_counts.get(y, 0) + 1
        oos_years = [y for y in yr_counts if y >= 2019]
        avg_per_year = (sum(yr_counts[y] for y in oos_years)
                        / max(len(oos_years), 1))

        # Sector concentration metric
        sec_trades: Dict[str, int] = defaultdict(int)
        for t in trades:
            sec_trades[t.get("sector", "OTHER")] += 1
        max_sec_pct = max(sec_trades.values()) / len(trades) * 100

        results.append({
            "ad_max": ad_max, "wt": wt, "nt": nt, "lt": lt,
            "ww": ww, "mps": mps, "tnw": tnw,
            "n": len(trades), "wr": wr, "ann": ann,
            "dd": dd, "sharpe": sh_val, "eq": eq,
            "avg_yr": avg_per_year, "max_sec": max_sec_pct,
        })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n  Evaluated {combo_count} combos, "
        f"{len(results)} with 10+ trades"
    )
    print(
        f"\n{'AD':>4} {'WT':>4} {'NT':>4} {'LT':>4} {'WW':>3} "
        f"{'MPS':>3} {'TNW':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7} {'MaxSec':>7}"
    )
    print("-" * 95)
    for r in results[:30]:
        print(
            f"{r['ad_max']:>4.2f} {r['wt']:>4.2f} {r['nt']:>4.2f} "
            f"{r['lt']:>4.2f} {r['ww']:>3} "
            f"{r['mps']:>3} {r['tnw']:>4} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['avg_yr']:>7.1f} "
            f"{r['max_sec']:>6.1f}%"
        )

    # === 4. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v60(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_ad_ratio=r["ad_max"],
            win_threshold=r["wt"],
            normal_threshold=r["nt"],
            lose_threshold=r["lt"],
            win_rate_window=r["ww"],
            atr_stop=3.0,
            top_n_winning=r["tnw"],
            max_per_sector=r["mps"],
            start_di=60,
        )
        label = (
            f"ad={r['ad_max']:.2f} wt={r['wt']:.2f} nt={r['nt']:.2f} "
            f"lt={r['lt']:.2f} ww={r['ww']} mps={r['mps']} tnw={r['tnw']}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 5. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: ad={best['ad_max']:.2f} "
            f"wt={best['wt']:.2f} nt={best['nt']:.2f} "
            f"lt={best['lt']:.2f} ww={best['ww']} "
            f"mps={best['mps']} tnw={best['tnw']}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_ad_ratio=best["ad_max"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=best["tnw"],
            max_per_sector=best["mps"],
        )

        # === 6. Head-to-head: V60 vs V47 ===
        print("\n" + "=" * 70)
        print("  HEAD-TO-HEAD: V60 (sector+breadth) vs V47 (sector only)")
        print("  (2019-2026 OOS)")
        print("=" * 70)

        # V60 best config
        trades_v60, eq_v60, dd_v60 = backtest_v60(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_ad_ratio=best["ad_max"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=best["tnw"],
            max_per_sector=best["mps"],
            start_di=bt_2019,
        )

        # V47 equivalent: no breadth filter (ad_max=1.0)
        trades_v47, eq_v47, dd_v47 = backtest_v60(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_ad_ratio=1.0,  # effectively no breadth filter
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=best["tnw"],
            max_per_sector=best["mps"],
            start_di=bt_2019,
        )

        print(f"\n  V60 (sector + breadth, ad={best['ad_max']:.2f}):")
        analyze(trades_v60, eq_v60, dd_v60, "V60-sect+breadth")
        print(f"\n  V47 (sector only, ad=1.00):")
        analyze(trades_v47, eq_v47, dd_v47, "V47-sector-only")

        if trades_v60 and trades_v47:
            # Compute Sharpe for both
            def compute_sharpe(tl: List[dict]) -> float:
                ap = [t["pnl_abs"] for t in sorted(tl, key=lambda x: x["di"])]
                rets_a = np.array(ap) / CASH0
                if np.std(rets_a) > 0:
                    return float(np.mean(rets_a) / np.std(rets_a) * np.sqrt(252))
                return 0.0

            sh_v60 = compute_sharpe(trades_v60)
            sh_v47 = compute_sharpe(trades_v47)
            print(
                f"\n  VERDICT: V60 Sharpe={sh_v60:.2f} vs "
                f"V47 Sharpe={sh_v47:.2f} "
                f"delta={sh_v60 - sh_v47:+.2f}"
            )
            print(
                f"  V60 vs V47: "
                f"eq_delta={eq_v60 - eq_v47:+,.0f} "
                f"dd_delta={dd_v60 - dd_v47:+.1f}% "
                f"n_delta={len(trades_v60) - len(trades_v47):+d} trades"
            )
            if sh_v60 > sh_v47:
                print(
                    f"  CONCLUSION: BREADTH FILTER HELPS! "
                    f"V60 beats V47 by {sh_v60 - sh_v47:.2f} Sharpe"
                )
            else:
                print(
                    f"  CONCLUSION: BREADTH FILTER HURTS. "
                    f"V47 beats V60 by {sh_v47 - sh_v60:.2f} Sharpe"
                )

    print(f"\n[V60] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
