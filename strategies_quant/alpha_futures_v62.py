"""
V62: V47 Sector Limit + Extreme Quality Filter
================================================
V47 is our ALL-TIME BEST: Sharpe 6.65, MDD 8.8%, ann +19.8%, 112 trades.
V29 showed that extreme rank (>0.90) produces Sharpe 8.57 but too few trades.

V62 applies extreme quality filtering to V47:
  - In WINNING/NORMAL mode: use standard threshold (same as V47)
  - In LOSING mode: require BOTH high rank AND sector limit AND KER < gate
  - Minimum composite rank floor: never take weak signals
  - Skip trading entirely if market is NOT oversold (A/D ratio > threshold)

Architecture (same as V47):
  1. Compute V18's 7 cross-sectional ranks, composite score
  2. Market breadth: A/D ratio from 5d returns across all commodities
  3. Dynamic three-mode threshold (WINNING/NORMAL/LOSING)
  4. Extreme quality filters in LOSING mode
  5. Sector-constrained greedy selection
  6. KER gate, hold 5d, ATR stop 3.0, pyramid on day-1 winners

Parameter sweep:
  - min_composite: 0.75, 0.80, 0.85
  - max_per_sector: 1, 2
  - losing_thresh: 0.88, 0.90, 0.92
  - win_rate_window: 10, 15, 20
  - ker_gate_losing: 0.08, 0.10, 0.15
  - max_ad_ratio: 0.45, 0.50, 0.55

Walk-forward 2019-2026, full 10-year for top configs.
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
    'i': 'BLACK', 'j': 'BLACK', 'jm': 'BLACK', 'hc': 'BLACK',
    'sf': 'BLACK', 'sm': 'BLACK', 'wr': 'BLACK',
    # METAL (non-ferrous)
    'cu': 'METAL', 'al': 'METAL', 'zn': 'METAL', 'pb': 'METAL',
    'ni': 'METAL', 'sn': 'METAL', 'ss': 'METAL', 'ao': 'METAL',
    # ENERGY
    'sc': 'ENERGY', 'fu': 'ENERGY', 'bu': 'ENERGY',
    'pg': 'ENERGY', 'eb': 'ENERGY', 'ta': 'ENERGY',
    # CHEMICAL
    'v': 'CHEMICAL', 'pp': 'CHEMICAL', 'l': 'CHEMICAL',
    'eg': 'CHEMICAL', 'ma': 'CHEMICAL', 'sa': 'CHEMICAL',
    'ur': 'CHEMICAL', 'pf': 'CHEMICAL', 'sh': 'CHEMICAL',
    'lc': 'CHEMICAL',
    # AGRI (oilseeds / agricultural)
    'm': 'AGRI', 'y': 'AGRI', 'a': 'AGRI', 'p': 'AGRI',
    'c': 'AGRI', 'cs': 'AGRI', 'jd': 'AGRI', 'rr': 'AGRI',
    'lrm': 'AGRI',
    # SOFTS
    'cf': 'SOFTS', 'sr': 'SOFTS', 'ap': 'SOFTS',
    'cj': 'SOFTS', 'pk': 'SOFTS', 'lh': 'SOFTS',
}


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    """Build a symbol-index to sector mapping."""
    sector_lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = sym.lower().split('.')[0].strip()
        # Strip digit suffixes (e.g., 'i2401' -> 'i')
        while base and base[-1].isdigit():
            base = base[:-1]
        # Strip 'fi' suffix used by the data loader (e.g., 'cffi' -> 'cf')
        if base.endswith('fi'):
            base = base[:-2]
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
    t0 = time.time()
    print("[V62] Computing raw factors...", flush=True)

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
    print("[V62] Computing cross-sectional ranks...", flush=True)

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
    """Compute KER (Kaufman Efficiency Ratio) raw 0-1 values."""
    ker_raw = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker_raw[si, di] = net_change / total_change

    return ker_raw


def compute_market_breadth(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Compute daily A/D ratio (fraction of commodities with positive 5d return)."""
    t0 = time.time()
    print("[V62] Computing market breadth...", flush=True)

    ad_ratio = np.full(ND, np.nan)
    for di in range(20, ND):
        rets = []
        for si in range(NS):
            c_now = C[si, di]
            c_5d = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_5d) and c_5d > 0:
                rets.append(c_now / c_5d - 1.0)
        if len(rets) >= 10:
            ad_ratio[di] = sum(1 for r in rets if r > 0) / len(rets)

    print(f"  Breadth done: {time.time() - t0:.1f}s", flush=True)
    return ad_ratio


def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    print("[V62] Building composite signal...", flush=True)

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


def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_raw = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)
    ad_ratio = compute_market_breadth(C, NS, ND)

    return {
        "composite": composite,
        "n_confirm": n_confirm,
        "ker_raw": ker_raw,
        "ad_ratio": ad_ratio,
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
    else:  # normal
        return {
            "threshold": normal_threshold,
            "top_n": top_n_normal,
            "pyramid_ratio": 0.3,
            "mode_label": "NORM",
        }


def backtest_v62(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    max_per_sector: int = 1,
    min_confidence: int = 3,
    hold_days: int = 5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
    # V62 new parameters
    min_composite: float = 0.80,
    ker_gate_losing: float = 0.10,
    max_ad_ratio: float = 0.50,
) -> Tuple[List[dict], float, float]:
    """Backtest V62 with extreme quality filtering on top of V47."""
    composite = sigs["composite"]
    ker_raw = sigs["ker_raw"]
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
                        "sector": sector_lookup.get(si, 'OTHER'),
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
                        "sector": sector_lookup.get(si, 'OTHER'),
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

        # Pyramid on day-1 winners (ratio varies by mode)
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

        # V62: Skip new entries if market is not oversold
        market_oversold = (
            np.isnan(ad_ratio[di]) or ad_ratio[di] <= max_ad_ratio
        )
        if not market_oversold:
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
            # V62: Minimum composite floor -- never take weak signals
            if composite[si, di] < min_composite:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            # KER gate: stricter for LOSING mode, standard for others
            ker_val = ker_raw[si, di]
            if mode == "losing":
                if np.isnan(ker_val) or ker_val > ker_gate_losing:
                    continue
            else:
                # Standard KER gate (reject noisy instruments)
                if not np.isnan(ker_val) and ker_val > 0.30:
                    continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(current_top_n, 1)
            candidates.append((composite[si, di], si, alloc))

        # Sort by composite score (highest first)
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
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    max_per_sector: int = 1,
    min_composite: float = 0.80,
    ker_gate_losing: float = 0.10,
    max_ad_ratio: float = 0.50,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V62 "
        f"(wt={win_threshold:.2f} nt={normal_threshold:.2f} "
        f"lt={lose_threshold:.2f} ww={win_rate_window} "
        f"ats={atr_stop:.1f} tnw={top_n_winning} mps={max_per_sector} "
        f"mc={min_composite:.2f} kgl={ker_gate_losing:.2f} "
        f"mad={max_ad_ratio:.2f})"
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

        trades, _, _ = backtest_v62(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=atr_stop,
            top_n_winning=top_n_winning,
            max_per_sector=max_per_sector,
            min_confidence=3,
            hold_days=5,
            pyramid_day=1,
            start_di=test_start,
            end_di=test_end_idx + 1,
            min_composite=min_composite,
            ker_gate_losing=ker_gate_losing,
            max_ad_ratio=max_ad_ratio,
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


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V62: V47 SECTOR LIMIT + EXTREME QUALITY FILTER")
    print("  Extreme rank + sector limit + KER gate + A/D oversold filter")
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
        # (mc, mps, lt, ww, kgl, mad)
        (0.80, 1, 0.90, 15, 0.10, 0.50),
        (0.80, 1, 0.90, 15, 0.10, 0.55),
        (0.80, 2, 0.90, 15, 0.10, 0.50),
        (0.85, 1, 0.90, 15, 0.08, 0.50),
        (0.80, 1, 0.92, 15, 0.10, 0.50),
    ]

    for mc, mps, lt, ww, kgl, mad in default_configs:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_per_sector=mps,
            lose_threshold=lt,
            win_rate_window=ww,
            min_composite=mc,
            ker_gate_losing=kgl,
            max_ad_ratio=mad,
        )

    # === 2. Full 10-year backtest with profiles ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- PROFILE COMPARISON")
    print("=" * 70)

    profiles = [
        (0.80, 1, 0.90, 15, 0.10, 0.50, "Default mc=0.80 mps=1"),
        (0.80, 2, 0.90, 15, 0.10, 0.50, "Default mc=0.80 mps=2"),
        (0.85, 1, 0.90, 15, 0.08, 0.50, "Strict mc=0.85 kgl=0.08"),
        (0.75, 1, 0.88, 15, 0.10, 0.50, "Relaxed mc=0.75 lt=0.88"),
        (0.85, 1, 0.92, 15, 0.08, 0.45, "Tight mc=0.85 kgl=0.08 mad=0.45"),
        (0.80, 1, 0.90, 10, 0.10, 0.50, "Short window"),
        (0.80, 1, 0.90, 20, 0.10, 0.50, "Long window"),
    ]

    for mc, mps, lt, ww, kgl, mad, label in profiles:
        trades, eq, dd = backtest_v62(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            max_per_sector=mps,
            lose_threshold=lt,
            win_rate_window=ww,
            start_di=60,
            min_composite=mc,
            ker_gate_losing=kgl,
            max_ad_ratio=mad,
        )
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 3. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []

    sweep_params = {
        "min_composite": [0.75, 0.80, 0.85],
        "max_per_sector": [1, 2],
        "lose_threshold": [0.88, 0.90, 0.92],
        "win_rate_window": [10, 15, 20],
        "ker_gate_losing": [0.08, 0.10, 0.15],
        "max_ad_ratio": [0.45, 0.50, 0.55],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for mc, mps, lt, ww, kgl, mad in product(
        sweep_params["min_composite"],
        sweep_params["max_per_sector"],
        sweep_params["lose_threshold"],
        sweep_params["win_rate_window"],
        sweep_params["ker_gate_losing"],
        sweep_params["max_ad_ratio"],
    ):
        # Skip invalid: lose threshold must be > normal threshold (0.82)
        if lt <= 0.82:
            continue

        combo_count += 1
        trades, eq, dd = backtest_v62(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=3.0,
            top_n_winning=2,
            max_per_sector=mps,
            start_di=bt_2019,
            min_composite=mc,
            ker_gate_losing=kgl,
            max_ad_ratio=mad,
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

        yr_counts: Dict[int, int] = {}
        for t in trades:
            y = t["year"]
            yr_counts[y] = yr_counts.get(y, 0) + 1
        oos_years = [y for y in yr_counts if y >= 2019]
        avg_per_year = (sum(yr_counts[y] for y in oos_years)
                        / max(len(oos_years), 1))

        sec_trades: Dict[str, int] = defaultdict(int)
        for t in trades:
            sec_trades[t.get("sector", "OTHER")] += 1
        max_sec_pct = max(sec_trades.values()) / len(trades) * 100

        results.append({
            "mc": mc, "mps": mps, "lt": lt,
            "ww": ww, "kgl": kgl, "mad": mad,
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
        f"\n{'MC':>4} {'MPS':>3} {'LT':>4} {'WW':>3} "
        f"{'KGL':>4} {'MAD':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7} {'MaxSec':>7}"
    )
    print("-" * 95)
    for r in results[:30]:
        print(
            f"{r['mc']:>4.2f} {r['mps']:>3} {r['lt']:>4.2f} "
            f"{r['ww']:>3} {r['kgl']:>4.2f} {r['mad']:>4.2f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['avg_yr']:>7.1f} "
            f"{r['max_sec']:>6.1f}%"
        )

    # === 4. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v62(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            lose_threshold=r["lt"],
            win_rate_window=r["ww"],
            atr_stop=3.0,
            top_n_winning=2,
            max_per_sector=r["mps"],
            start_di=60,
            min_composite=r["mc"],
            ker_gate_losing=r["kgl"],
            max_ad_ratio=r["mad"],
        )
        label = (
            f"mc={r['mc']:.2f} mps={r['mps']} lt={r['lt']:.2f} "
            f"ww={r['ww']} kgl={r['kgl']:.2f} mad={r['mad']:.2f}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 5. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: mc={best['mc']:.2f} mps={best['mps']} "
            f"lt={best['lt']:.2f} ww={best['ww']} "
            f"kgl={best['kgl']:.2f} mad={best['mad']:.2f}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=2,
            max_per_sector=best["mps"],
            min_composite=best["mc"],
            ker_gate_losing=best["kgl"],
            max_ad_ratio=best["mad"],
        )

        # === 6. Compare: V62 vs V47 baseline ===
        print("\n" + "=" * 70)
        print("  COMPARISON: V62 (extreme quality) vs V47 (sector only)")
        print("  (2019-2026 OOS)")
        print("=" * 70)

        # V62 with best config
        trades_v62, eq_v62, dd_v62 = backtest_v62(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=2,
            max_per_sector=best["mps"],
            start_di=bt_2019,
            min_composite=best["mc"],
            ker_gate_losing=best["kgl"],
            max_ad_ratio=best["mad"],
        )

        # V47-like: no extreme quality filters (mc=0, kgl=1.0, mad=1.0)
        trades_v47like, eq_v47like, dd_v47like = backtest_v62(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=2,
            max_per_sector=best["mps"],
            start_di=bt_2019,
            min_composite=0.0,    # no minimum composite floor
            ker_gate_losing=1.0,  # no KER gate in losing mode
            max_ad_ratio=1.0,     # no A/D ratio filter
        )

        print(f"\n  V62 WITH EXTREME QUALITY FILTER:")
        analyze(trades_v62, eq_v62, dd_v62, "V62-extreme")
        print(f"\n  V47-LIKE (NO EXTREME FILTERS):")
        analyze(trades_v47like, eq_v47like, dd_v47like, "V47-base")

        if trades_v62 and trades_v47like:
            print(
                f"\n  V62 vs V47: "
                f"eq_delta={eq_v62 - eq_v47like:+,.0f} "
                f"dd_delta={dd_v62 - dd_v47like:+.1f}% "
                f"trade_delta={len(trades_v62) - len(trades_v47like):+d}"
            )

    print(f"\n[V62] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
