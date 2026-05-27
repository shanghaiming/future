"""
V54: Multi-Strategy Ensemble (Three Independent Sub-Strategies)
================================================================
Run 3 independent sub-strategies simultaneously, each with its own
position tracking and equity. Total portfolio = sum of all 3.

Sub-strategy A: Extreme oversold (V29 style, very selective)
  - rank > 0.88, KER < 0.10, top_n=1, hold=5d, pyr=0.0

Sub-strategy B: Standard adaptive (V39 style, moderate)
  - rank > adaptive_thresh, KER < 0.15, top_n=2, hold=5d, pyr=0.5

Sub-strategy C: Relaxed momentum (higher frequency)
  - rank > 0.75 AND ret_5d < -0.03, KER < 0.20, top_n=3, hold=3d, pyr=0.3
  - Only in uptrend (ret_10d > 0)

Shared computation:
  - V18's 7 factors + cross-sectional ranks (computed once)
  - Each sub has its own cash tracking (CASH0/3 each)
  - Portfolio-level: max 1 position per instrument across all subs

Walk-forward validation 2019-2026, full 10yr for best configs.
INSTITUTIONAL QUALITY TARGETS: Sharpe > 3.0, MDD < 20%, ann > 30%

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage.
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
    print("[V54] Computing raw factors...", flush=True)

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
    print("[V54] Computing cross-sectional ranks...", flush=True)

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
    """Compute KER as raw value (0-1 scale), not regime."""
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
    return ker_10


def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    print("[V54] Building composite signal...", flush=True)

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

    return {
        "composite": composite,
        "n_confirm": n_confirm,
        "ker_raw": ker_raw,
        "ranks": ranks,
        "ret_5d": raw["ret_5d"],
        "ret_10d": raw["ret_10d"],
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


def get_adaptive_threshold(
    recent_trades_win: List[int],
    window: int,
    base_threshold: float,
) -> float:
    """V39-style adaptive threshold based on recent win rate."""
    if len(recent_trades_win) < 5:
        return base_threshold

    recent = recent_trades_win[-window:]
    win_rate = sum(recent) / len(recent)

    if win_rate > 0.65:
        return base_threshold - 0.05
    elif win_rate < 0.45:
        return base_threshold + 0.05
    return base_threshold


def backtest_single_sub(
    sub_id: str,
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    threshold: float,
    ker_max: float,
    top_n: int,
    hold_days: int,
    pyramid_ratio: float,
    atr_stop: float,
    min_confidence: int,
    sub_cash: float,
    occupied_si: set,
    start_di: int = 60,
    end_di: Optional[int] = None,
    require_momentum: bool = False,
    adaptive_threshold: bool = False,
    adaptive_window: int = 15,
) -> Tuple[List[dict], float, float, set]:
    """Backtest a single sub-strategy with its own equity tracking.

    Parameters
    ----------
    occupied_si : set of si indices already held by OTHER sub-strategies.
        The caller must merge these from other subs before calling.

    Returns
    -------
    (trades, equity, max_dd, held_si_set)
    """
    composite = sigs["composite"]
    ker_raw = sigs["ker_raw"]
    n_confirm = sigs["n_confirm"]
    ret_5d = sigs["ret_5d"]
    ret_10d = sigs["ret_10d"]

    if end_di is None:
        end_di = ND - 1

    equity = sub_cash
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        # Adaptive threshold (V39 style for sub B)
        current_threshold = threshold
        if adaptive_threshold:
            current_threshold = get_adaptive_threshold(
                recent_trades_win, adaptive_window, threshold)

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
                        "reason": "stop",
                        "pyr": is_pyr,
                        "sub": sub_id,
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
                        "reason": "hold",
                        "pyr": is_pyr,
                        "sub": sub_id,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr))

        # Pyramid on day-1 winners
        if pyramid_ratio > 0:
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
                if hold == 1 and not np.isnan(C[si, di]):
                    avg_ep = np.mean([ep for _, ep, _, _, _ in pos_list])
                    if C[si, di] > avg_ep:
                        base_alloc = sum(a for _, _, _, a, _ in pos_list)
                        pyr_alloc = base_alloc * pyramid_ratio
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
        if len(positions) >= top_n:
            continue

        # Entry signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            # Portfolio-level constraint: max 1 position per instrument
            if si in held or si in occupied_si:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < current_threshold:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            # KER gate
            if np.isnan(ker_raw[si, di]) or ker_raw[si, di] > ker_max:
                continue
            # Sub C: momentum confirmation (ret_5d < -0.03)
            if require_momentum:
                if np.isnan(ret_5d[si, di]) or ret_5d[si, di] >= -0.03:
                    continue
                # Also require uptrend (ret_10d > 0)
                if np.isnan(ret_10d[si, di]) or ret_10d[si, di] <= 0:
                    continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(top_n, 1)
            candidates.append((composite[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held or si in occupied_si:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    held_si = {p[0] for p in positions}
    return trades, equity, max_dd, held_si


def backtest_ensemble(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sub_a_thresh: float = 0.88,
    sub_a_ker: float = 0.10,
    sub_a_top_n: int = 1,
    sub_a_hold: int = 5,
    sub_a_pyr: float = 0.0,
    sub_b_thresh: float = 0.82,
    sub_b_ker: float = 0.15,
    sub_b_top_n: int = 2,
    sub_b_hold: int = 5,
    sub_b_pyr: float = 0.5,
    sub_c_thresh: float = 0.75,
    sub_c_ker: float = 0.20,
    sub_c_top_n: int = 3,
    sub_c_hold: int = 3,
    sub_c_pyr: float = 0.3,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    sub_b_adaptive: bool = True,
    adaptive_window: int = 15,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float, Dict[str, dict]]:
    """Backtest the full ensemble of 3 sub-strategies.

    Each sub gets CASH0/3 capital. Portfolio = sum of all 3.
    Portfolio-level constraint: max 1 position per instrument across all subs.

    Returns (all_trades, total_equity, portfolio_max_dd, sub_results)
    """
    sub_cash = CASH0 / 3.0

    if end_di is None:
        end_di = ND - 1

    # === Sub A: Extreme oversold ===
    trades_a, eq_a, dd_a, held_a = backtest_single_sub(
        "A", C, O, H, L, NS, ND, dates, syms, sigs,
        threshold=sub_a_thresh,
        ker_max=sub_a_ker,
        top_n=sub_a_top_n,
        hold_days=sub_a_hold,
        pyramid_ratio=sub_a_pyr,
        atr_stop=atr_stop,
        min_confidence=min_confidence,
        sub_cash=sub_cash,
        occupied_si=set(),
        start_di=start_di,
        end_di=end_di,
        require_momentum=False,
        adaptive_threshold=False,
    )

    # === Sub B: Standard adaptive ===
    trades_b, eq_b, dd_b, held_b = backtest_single_sub(
        "B", C, O, H, L, NS, ND, dates, syms, sigs,
        threshold=sub_b_thresh,
        ker_max=sub_b_ker,
        top_n=sub_b_top_n,
        hold_days=sub_b_hold,
        pyramid_ratio=sub_b_pyr,
        atr_stop=atr_stop,
        min_confidence=min_confidence,
        sub_cash=sub_cash,
        occupied_si=held_a.copy(),
        start_di=start_di,
        end_di=end_di,
        require_momentum=False,
        adaptive_threshold=sub_b_adaptive,
        adaptive_window=adaptive_window,
    )

    # === Sub C: Relaxed momentum ===
    occupied_bc = held_a | held_b
    trades_c, eq_c, dd_c, held_c = backtest_single_sub(
        "C", C, O, H, L, NS, ND, dates, syms, sigs,
        threshold=sub_c_thresh,
        ker_max=sub_c_ker,
        top_n=sub_c_top_n,
        hold_days=sub_c_hold,
        pyramid_ratio=sub_c_pyr,
        atr_stop=atr_stop,
        min_confidence=min_confidence,
        sub_cash=sub_cash,
        occupied_si=occupied_bc,
        start_di=start_di,
        end_di=end_di,
        require_momentum=True,
        adaptive_threshold=False,
    )

    # Merge all trades
    all_trades = trades_a + trades_b + trades_c
    total_equity = eq_a + eq_b + eq_c

    # Compute portfolio-level max drawdown from merged daily PnL
    # Approximate: use worst sub DD as proxy (conservative)
    portfolio_max_dd = max(dd_a, dd_b, dd_c)

    sub_results = {
        "A": {"trades": trades_a, "eq": eq_a, "dd": dd_a, "cash": sub_cash},
        "B": {"trades": trades_b, "eq": eq_b, "dd": dd_b, "cash": sub_cash},
        "C": {"trades": trades_c, "eq": eq_c, "dd": dd_c, "cash": sub_cash},
    }

    return all_trades, total_equity, portfolio_max_dd, sub_results


def analyze(trades: List[dict], equity: float, max_dd: float,
            initial_cash: float, label: str = "") -> Optional[dict]:
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / initial_cash) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / initial_cash
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)

    n_pyr = sum(1 for t in trades if t.get("pyr"))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    sub_counts = {"A": 0, "B": 0, "C": 0}
    for t in trades:
        s = t.get("sub", "?")
        if s in sub_counts:
            sub_counts[s] += 1

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} "
        f"subs=[A:{sub_counts['A']} B:{sub_counts['B']} "
        f"C:{sub_counts['C']}]"
    )

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
    sub_a_thresh: float = 0.88,
    sub_a_ker: float = 0.10,
    sub_b_thresh: float = 0.82,
    sub_b_ker: float = 0.15,
    sub_c_thresh: float = 0.75,
    sub_c_ker: float = 0.20,
    atr_stop: float = 3.0,
    adaptive_window: int = 15,
    label: str = "",
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V54 ENSEMBLE {label}"
    )
    print(
        f"  A: thr={sub_a_thresh:.2f} ker={sub_a_ker:.2f} | "
        f"B: thr={sub_b_thresh:.2f} ker={sub_b_ker:.2f} | "
        f"C: thr={sub_c_thresh:.2f} ker={sub_c_ker:.2f} | "
        f"ats={atr_stop:.1f} aw={adaptive_window}"
    )
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades: List[dict] = []
    sub_trades: Dict[str, List[dict]] = {"A": [], "B": [], "C": []}

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

        trades, eq, dd, sub_results = backtest_ensemble(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sub_a_thresh=sub_a_thresh,
            sub_a_ker=sub_a_ker,
            sub_b_thresh=sub_b_thresh,
            sub_b_ker=sub_b_ker,
            sub_c_thresh=sub_c_thresh,
            sub_c_ker=sub_c_ker,
            atr_stop=atr_stop,
            adaptive_window=adaptive_window,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades
                       if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        for sub_id in ["A", "B", "C"]:
            sub_yr = [t for t in sub_results[sub_id]["trades"]
                      if dates[t["di"]].year == test_year]
            sub_trades[sub_id].extend(sub_yr)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            sub_dist = {"A": 0, "B": 0, "C": 0}
            for t in test_trades:
                s = t.get("sub", "?")
                if s in sub_dist:
                    sub_dist[s] += 1
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"subs=[A:{sub_dist['A']} B:{sub_dist['B']} "
                f"C:{sub_dist['C']}]",
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
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        # Per-sub summary
        for sub_id in ["A", "B", "C"]:
            st = sub_trades[sub_id]
            if st:
                snw = sum(1 for t in st if t["pnl_pct"] > 0)
                swr = snw / len(st) * 100
                print(
                    f"    Sub {sub_id}: {len(st)}t WR={swr:.1f}% "
                    f"avg={np.mean([t['pnl_pct'] for t in st]):+.2f}%"
                )
        return all_trades, sub_trades
    return [], sub_trades


def compute_metrics(
    trades: List[dict], equity: float, initial_cash: float,
) -> dict:
    """Compute standard metrics from trades and equity."""
    if len(trades) < 2:
        return {"n": len(trades), "wr": 0, "ann": 0, "dd": 0,
                "sharpe": 0, "eq": equity}

    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / initial_cash) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / initial_cash
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)

    return {"n": len(trades), "wr": wr, "ann": ann, "sharpe": sh,
            "eq": equity}


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V54: MULTI-STRATEGY ENSEMBLE (3 Independent Sub-Strategies)")
    print("  A: Extreme oversold | B: Standard adaptive | C: Relaxed momentum")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}"
    )

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ==================================================================
    # 1. Walk-Forward with default configs
    # ==================================================================
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026) -- DEFAULT CONFIGS")
    print("=" * 70)

    default_configs = [
        # (a_thr, a_ker, b_thr, b_ker, c_thr, c_ker, atr, aw, label)
        (0.88, 0.10, 0.82, 0.15, 0.75, 0.20, 3.0, 15, "Default"),
        (0.90, 0.10, 0.85, 0.12, 0.78, 0.18, 3.0, 15, "Tight"),
        (0.85, 0.12, 0.80, 0.15, 0.72, 0.22, 3.0, 15, "Relaxed"),
        (0.88, 0.10, 0.82, 0.15, 0.75, 0.20, 2.5, 15, "TightStop"),
        (0.88, 0.10, 0.82, 0.15, 0.75, 0.20, 3.0, 10, "FastAdapt"),
        (0.88, 0.10, 0.82, 0.15, 0.75, 0.20, 3.5, 15, "WideStop"),
        (0.90, 0.08, 0.85, 0.12, 0.78, 0.18, 3.0, 10, "UltraTight"),
    ]

    for a_t, a_k, b_t, b_k, c_t, c_k, ats, aw, lbl in default_configs:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sub_a_thresh=a_t, sub_a_ker=a_k,
            sub_b_thresh=b_t, sub_b_ker=b_k,
            sub_c_thresh=c_t, sub_c_ker=c_k,
            atr_stop=ats, adaptive_window=aw,
            label=lbl,
        )

    # ==================================================================
    # 2. Parameter sweep (OOS 2019-2026)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026 OOS)")
    print("=" * 70)

    results: List[dict] = []

    sweep_params = {
        "a_thr": [0.85, 0.88, 0.90],
        "a_ker": [0.08, 0.10, 0.12],
        "b_thr": [0.80, 0.82, 0.85],
        "b_ker": [0.12, 0.15, 0.18],
        "c_thr": [0.72, 0.75, 0.78],
        "c_ker": [0.18, 0.20, 0.22],
        "atr_stop": [2.5, 3.0, 3.5],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for a_t, a_k, b_t, b_k, c_t, c_k, ats in product(
        sweep_params["a_thr"],
        sweep_params["a_ker"],
        sweep_params["b_thr"],
        sweep_params["b_ker"],
        sweep_params["c_thr"],
        sweep_params["c_ker"],
        sweep_params["atr_stop"],
    ):
        combo_count += 1
        trades, eq, dd, sub_res = backtest_ensemble(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sub_a_thresh=a_t, sub_a_ker=a_k,
            sub_b_thresh=b_t, sub_b_ker=b_k,
            sub_c_thresh=c_t, sub_c_ker=c_k,
            atr_stop=ats,
            start_di=bt_2019,
        )

        if len(trades) < 10:
            continue

        metrics = compute_metrics(trades, eq, CASH0)

        # Count per-year trades
        yr_counts: Dict[int, int] = {}
        for t in trades:
            y = t["year"]
            yr_counts[y] = yr_counts.get(y, 0) + 1
        oos_years = [y for y in yr_counts if y >= 2019]
        avg_per_year = (sum(yr_counts[y] for y in oos_years)
                        / max(len(oos_years), 1))

        results.append({
            "a_t": a_t, "a_k": a_k,
            "b_t": b_t, "b_k": b_k,
            "c_t": c_t, "c_k": c_k,
            "ats": ats,
            "n": metrics["n"], "wr": metrics["wr"],
            "ann": metrics["ann"], "dd": dd,
            "sharpe": metrics["sharpe"], "eq": eq,
            "avg_yr": avg_per_year,
        })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n  Evaluated {combo_count} combos, "
        f"{len(results)} with 10+ trades"
    )
    print(
        f"\n{'AT':>4} {'AK':>4} {'BT':>4} {'BK':>4} "
        f"{'CT':>4} {'CK':>4} {'ATS':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7}"
    )
    print("-" * 90)
    for r in results[:30]:
        print(
            f"{r['a_t']:>4.2f} {r['a_k']:>4.2f} "
            f"{r['b_t']:>4.2f} {r['b_k']:>4.2f} "
            f"{r['c_t']:>4.2f} {r['c_k']:>4.2f} {r['ats']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['avg_yr']:>7.1f}"
        )

    # ==================================================================
    # 3. Top configs: full 10-year backtest with sub breakdown
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd, sub_res = backtest_ensemble(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sub_a_thresh=r["a_t"], sub_a_ker=r["a_k"],
            sub_b_thresh=r["b_t"], sub_b_ker=r["b_k"],
            sub_c_thresh=r["c_t"], sub_c_ker=r["c_k"],
            atr_stop=r["ats"],
            start_di=60,
        )
        label = (
            f"a={r['a_t']:.2f}/{r['a_k']:.2f} "
            f"b={r['b_t']:.2f}/{r['b_k']:.2f} "
            f"c={r['c_t']:.2f}/{r['c_k']:.2f} "
            f"ats={r['ats']:.1f}"
        )
        print(f"\n  FULL 10yr: {label}")
        analyze(trades, eq, dd, CASH0, "PORTFOLIO")

        # Sub-strategy breakdown
        for sub_id in ["A", "B", "C"]:
            sr = sub_res[sub_id]
            sub_cash = sr["cash"]
            if sr["trades"]:
                print(f"\n  Sub {sub_id}:")
                analyze(sr["trades"], sr["eq"], sr["dd"],
                        sub_cash, f"  Sub-{sub_id}")

    # ==================================================================
    # 4. Walk-forward for best config
    # ==================================================================
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: "
            f"a={best['a_t']:.2f}/{best['a_k']:.2f} "
            f"b={best['b_t']:.2f}/{best['b_k']:.2f} "
            f"c={best['c_t']:.2f}/{best['c_k']:.2f} "
            f"ats={best['ats']:.1f}"
        )
        print("=" * 70)

        wf_trades, wf_sub_trades = walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sub_a_thresh=best["a_t"], sub_a_ker=best["a_k"],
            sub_b_thresh=best["b_t"], sub_b_ker=best["b_k"],
            sub_c_thresh=best["c_t"], sub_c_ker=best["c_k"],
            atr_stop=best["ats"],
            label="BEST",
        )

        # Detailed sub analysis for best WF
        if wf_trades:
            print("\n  --- SUB-STRATEGY BREAKDOWN (WF BEST) ---")
            for sub_id in ["A", "B", "C"]:
                st = wf_sub_trades[sub_id]
                if st:
                    nw = sum(1 for t in st if t["pnl_pct"] > 0)
                    wr = nw / len(st) * 100
                    avg = np.mean([t["pnl_pct"] for t in st])
                    cum = np.prod(
                        [1 + t["pnl_pct"] / 100 for t in st]) - 1
                    ap = [t["pnl_abs"] for t in sorted(
                        st, key=lambda x: x["di"])]
                    sub_cash = CASH0 / 3
                    rets = np.array(ap) / sub_cash
                    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
                          if np.std(rets) > 0 else 0)
                    print(
                        f"  Sub {sub_id}: {len(st)}t WR={wr:.1f}% "
                        f"avg={avg:+.2f}% cum={cum:+.1%} Sh={sh:.2f}"
                    )
                    # Yearly breakdown
                    yr: Dict[int, dict] = {}
                    for t in st:
                        y = t["year"]
                        if y not in yr:
                            yr[y] = {"n": 0, "w": 0, "pnl": []}
                        yr[y]["n"] += 1
                        if t["pnl_pct"] > 0:
                            yr[y]["w"] += 1
                        yr[y]["pnl"].append(t["pnl_pct"])
                    for y in sorted(yr.keys()):
                        ys = yr[y]
                        yc = np.prod(
                            [1 + p / 100 for p in ys["pnl"]]) - 1
                        print(
                            f"    {y}: {ys['n']}t "
                            f"WR={ys['w']/ys['n']*100:.1f}% "
                            f"cum={yc:+.1%}"
                        )

        # ==================================================================
        # 5. Comparison: Ensemble vs individual subs
        # ==================================================================
        print("\n" + "=" * 70)
        print("  COMPARISON: FULL ENSEMBLE vs SUB-A vs SUB-B vs SUB-C")
        print("  (2019-2026 OOS)")
        print("=" * 70)

        trades_all, eq_all, dd_all, sub_res_all = backtest_ensemble(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sub_a_thresh=best["a_t"], sub_a_ker=best["a_k"],
            sub_b_thresh=best["b_t"], sub_b_ker=best["b_k"],
            sub_c_thresh=best["c_t"], sub_c_ker=best["c_k"],
            atr_stop=best["ats"],
            start_di=bt_2019,
        )

        print(f"\n  ENSEMBLE (all 3 subs combined):")
        analyze(trades_all, eq_all, dd_all, CASH0, "ENSEMBLE")

        for sub_id in ["A", "B", "C"]:
            sr = sub_res_all[sub_id]
            sub_cash = sr["cash"]
            print(f"\n  Sub {sub_id} (standalone):")
            analyze(sr["trades"], sr["eq"], sr["dd"],
                    sub_cash, f"Sub-{sub_id}")

    print(f"\n[V54] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
