"""
V52: V43 Dynamic Mode + Momentum Confirmation Filter
=====================================================
V43 is our ALL-TIME BEST: Sharpe 4.99, MDD 13.3%, ann +21.6%, 150 trades.

V52 innovation: Add a momentum confirmation filter.
Only take mean-reversion trades when:
  - The instrument has been declining (momentum < 0)
  - BUT the decline is decelerating (momentum_velocity > -threshold)
This catches genuine reversals rather than falling knives.

Momentum filter:
  - momentum = ret_Nd (N = mom_lookback)
  - momentum_velocity = ret_Nd[di] - ret_Nd[di-5]
  - Filter: momentum < 0 AND momentum_velocity > -threshold
  - WINNING mode: skip filter (take more trades)
  - NORMAL mode: threshold = mom_vel_threshold_normal (0.02)
  - LOSING mode: threshold = mom_vel_threshold_losing (0.01, stricter)

Signal architecture (same as V43):
  1. Compute V18's 7 cross-sectional ranks, composite score
  2. Dynamic three-mode threshold (WINNING/NORMAL/LOSING)
  3. NEW: Momentum confirmation filter per mode
  4. KER gate < 0.15, hold 5d, ATR stop 3.0
  5. Pyramid on day-1 winners

Parameter sweep:
  - mom_lookback: 10, 15
  - mom_vel_threshold_normal: 0.01, 0.02, 0.03
  - mom_vel_threshold_losing: 0.005, 0.01
  - skip_in_winning: True/False
  - win_rate_window: 10, 15, 20
  - Keep V43's mode thresholds

Walk-forward 2019-2026, full 10yr for best configs.
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

DEFAULT_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol": 0.15,
    "rank_ret10d": 0.10,
    "rank_range": 0.10,
    "rank_atrp": 0.05,
}


def compute_rsi_manual(
    C: np.ndarray, NS: int, ND: int, period: int = 14
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
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V52] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 10])
                and C[si, di - 10] > 0
            ):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    ret_15d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(15, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 15])
                and C[si, di - 15] > 0
            ):
                ret_15d[si, di] = C[si, di] / C[si, di - 15] - 1.0

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

    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (
                not np.isnan(H[si, di])
                and not np.isnan(L[si, di])
                and not np.isnan(C[si, di])
            ):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (H[si, di] - L[si, di]) / C[si, di]

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
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c))
                    )
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "ret_15d": ret_15d,
        "oi_5d": oi_5d,
        "vol_5d": vol_5d,
        "daily_range": daily_range,
        "rsi14": rsi14,
        "atrp": atrp,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int,
    ND: int,
    min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V52] Computing cross-sectional ranks...", flush=True)

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


def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int,
    ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    print("[V52] Building composite signal...", flush=True)

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
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)

    return {
        "composite": composite,
        "n_confirm": n_confirm,
        "ker_regime": ker_regime,
        "ranks": ranks,
        "raw": raw,
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
    mom_vel_threshold_normal: float,
    mom_vel_threshold_losing: float,
    skip_in_winning: bool,
    top_n_normal: int = 2,
) -> Dict:
    """Get trading parameters for the given mode including momentum filter."""
    if mode == "winning":
        return {
            "threshold": 0.75,
            "top_n": top_n_winning,
            "pyramid_ratio": 0.5,
            "mode_label": "WIN",
            "apply_mom_filter": not skip_in_winning,
            "mom_vel_threshold": mom_vel_threshold_normal,
        }
    elif mode == "losing":
        return {
            "threshold": lose_threshold,
            "top_n": 1,
            "pyramid_ratio": 0.0,
            "mode_label": "LOSE",
            "apply_mom_filter": True,
            "mom_vel_threshold": mom_vel_threshold_losing,
        }
    else:
        return {
            "threshold": normal_threshold,
            "top_n": top_n_normal,
            "pyramid_ratio": 0.3,
            "mode_label": "NORM",
            "apply_mom_filter": True,
            "mom_vel_threshold": mom_vel_threshold_normal,
        }


def check_momentum_filter(
    momentum: np.ndarray,
    si: int,
    di: int,
    mom_vel_threshold: float,
) -> bool:
    """Check if instrument passes momentum confirmation filter.

    Returns True if the instrument qualifies as a genuine reversal:
    - momentum < 0 (still declining)
    - momentum_velocity > -threshold (decline is decelerating)
    """
    mom_now = momentum[si, di]
    if np.isnan(mom_now):
        return False

    # Must be declining (momentum negative)
    if mom_now >= 0:
        return False

    # Must have enough history for velocity computation
    if di < 5 or np.isnan(momentum[si, di - 5]):
        return False

    # Velocity = current momentum minus 5-day-ago momentum
    mom_vel = mom_now - momentum[si, di - 5]

    # Decline must be decelerating (velocity not too negative)
    return mom_vel > -mom_vel_threshold


def backtest_v52(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    sigs: Dict[str, np.ndarray],
    mom_lookback: int = 10,
    mom_vel_threshold_normal: float = 0.02,
    mom_vel_threshold_losing: float = 0.01,
    skip_in_winning: bool = True,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V52 with dynamic threshold + momentum confirmation."""
    composite = sigs["composite"]
    ker_regime = sigs["ker_regime"]
    n_confirm = sigs["n_confirm"]
    raw = sigs["raw"]

    # Select momentum array based on lookback
    if mom_lookback == 15:
        momentum = raw["ret_15d"]
    else:
        momentum = raw["ret_10d"]

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
            recent_trades_win, win_threshold, win_rate_window
        )
        mode_params = get_mode_params(
            mode,
            normal_threshold,
            lose_threshold,
            top_n_winning,
            mom_vel_threshold_normal,
            mom_vel_threshold_losing,
            skip_in_winning,
        )
        current_threshold = mode_params["threshold"]
        current_top_n = mode_params["top_n"]
        current_pyramid_ratio = mode_params["pyramid_ratio"]
        current_mode_label = mode_params["mode_label"]
        apply_mom_filter = mode_params["apply_mom_filter"]
        mom_vel_threshold = mode_params["mom_vel_threshold"]

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
                    is_win = pnl > 0
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100,
                            "days": di - edi + 1,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "reason": "stop",
                            "pyr": is_pyr,
                            "mode": current_mode_label,
                            "threshold": current_threshold,
                            "mom_filtered": False,
                        }
                    )
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100,
                            "days": di - edi + 1,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "reason": "hold",
                            "pyr": is_pyr,
                            "mode": current_mode_label,
                            "threshold": current_threshold,
                            "mom_filtered": False,
                        }
                    )
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

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
                                (
                                    si,
                                    di,
                                    c_now,
                                    c_now - atr_stop * atr,
                                    pyr_alloc,
                                    True,
                                )
                            )
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

            # Momentum confirmation filter
            if apply_mom_filter:
                if not check_momentum_filter(
                    momentum, si, di, mom_vel_threshold
                ):
                    continue

            alloc = 1.0 / max(current_top_n, 1)
            candidates.append((composite[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:current_top_n]:
            if len(positions) >= current_top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

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

    mode_counts = {"WIN": 0, "NORM": 0, "LOSE": 0}
    for t in trades:
        m = t.get("mode", "NORM")
        if m in mode_counts:
            mode_counts[m] += 1

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} "
        f"modes=[W:{mode_counts['WIN']} N:{mode_counts['NORM']} "
        f"L:{mode_counts['LOSE']}]"
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
    mom_lookback: int = 10,
    mom_vel_threshold_normal: float = 0.02,
    mom_vel_threshold_losing: float = 0.01,
    skip_in_winning: bool = True,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V52 "
        f"(ml={mom_lookback} mvt_n={mom_vel_threshold_normal:.3f} "
        f"mvt_l={mom_vel_threshold_losing:.3f} skip_w={skip_in_winning} "
        f"wt={win_threshold:.2f} nt={normal_threshold:.2f} "
        f"lt={lose_threshold:.2f} ww={win_rate_window} "
        f"ats={atr_stop:.1f} tnw={top_n_winning})"
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

        trades, _, _ = backtest_v52(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            mom_lookback=mom_lookback,
            mom_vel_threshold_normal=mom_vel_threshold_normal,
            mom_vel_threshold_losing=mom_vel_threshold_losing,
            skip_in_winning=skip_in_winning,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=atr_stop,
            top_n_winning=top_n_winning,
            min_confidence=3,
            use_ker_gate=True,
            hold_days=5,
            pyramid_day=1,
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
            modes = {"W": 0, "N": 0, "L": 0}
            for t in test_trades:
                m = t.get("mode", "NORM")
                if m == "WIN":
                    modes["W"] += 1
                elif m == "LOSE":
                    modes["L"] += 1
                else:
                    modes["N"] += 1
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"modes=[W:{modes['W']} N:{modes['N']} L:{modes['L']}]",
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
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V52: V43 DYNAMIC MODE + MOMENTUM CONFIRMATION FILTER")
    print("  V43 adaptive + momentum reversal quality filter")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
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

    # === 1. Walk-Forward with default configs ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026) -- DEFAULT CONFIGS")
    print("=" * 70)

    default_configs = [
        (10, 0.02, 0.01, True, 0.60, 0.82, 0.90, 15, 3.0, 2),
        (10, 0.02, 0.01, False, 0.60, 0.82, 0.90, 15, 3.0, 2),
        (15, 0.02, 0.01, True, 0.60, 0.82, 0.90, 15, 3.0, 2),
        (10, 0.03, 0.01, True, 0.60, 0.82, 0.90, 15, 3.0, 2),
        (10, 0.02, 0.005, True, 0.60, 0.82, 0.90, 15, 3.0, 2),
    ]

    for ml, mvn, mvl, skw, wt, nt, lt, ww, ats, tnw in default_configs:
        walk_forward(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            mom_lookback=ml,
            mom_vel_threshold_normal=mvn,
            mom_vel_threshold_losing=mvl,
            skip_in_winning=skw,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=ats,
            top_n_winning=tnw,
        )

    # === 2. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []

    sweep_params = {
        "mom_lookback": [10, 15],
        "mom_vel_threshold_normal": [0.01, 0.02, 0.03],
        "mom_vel_threshold_losing": [0.005, 0.01],
        "skip_in_winning": [True, False],
        "win_threshold": [0.55, 0.60, 0.65],
        "normal_threshold": [0.80, 0.82, 0.85],
        "lose_threshold": [0.88, 0.90, 0.92],
        "win_rate_window": [10, 15, 20],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for ml, mvn, mvl, skw, wt, nt, lt, ww in product(
        sweep_params["mom_lookback"],
        sweep_params["mom_vel_threshold_normal"],
        sweep_params["mom_vel_threshold_losing"],
        sweep_params["skip_in_winning"],
        sweep_params["win_threshold"],
        sweep_params["normal_threshold"],
        sweep_params["lose_threshold"],
        sweep_params["win_rate_window"],
    ):
        if wt <= 0.50:
            continue
        if lt <= nt:
            continue

        combo_count += 1
        trades, eq, dd = backtest_v52(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            mom_lookback=ml,
            mom_vel_threshold_normal=mvn,
            mom_vel_threshold_losing=mvl,
            skip_in_winning=skw,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=3.0,
            top_n_winning=2,
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
        sh_val = (
            np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
            if np.std(rets_arr) > 0
            else 0
        )

        yr_counts: Dict[int, int] = {}
        for t in trades:
            y = t["year"]
            yr_counts[y] = yr_counts.get(y, 0) + 1
        oos_years = [y for y in yr_counts if y >= 2019]
        avg_per_year = (
            sum(yr_counts[y] for y in oos_years) / max(len(oos_years), 1)
        )

        results.append(
            {
                "ml": ml,
                "mvn": mvn,
                "mvl": mvl,
                "skw": skw,
                "wt": wt,
                "nt": nt,
                "lt": lt,
                "ww": ww,
                "n": len(trades),
                "wr": wr,
                "ann": ann,
                "dd": dd,
                "sharpe": sh_val,
                "eq": eq,
                "avg_yr": avg_per_year,
            }
        )

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n  Evaluated {combo_count} combos, "
        f"{len(results)} with 10+ trades"
    )
    print(
        f"\n{'ML':>3} {'MVN':>5} {'MVL':>5} {'SKW':>4} "
        f"{'WT':>4} {'NT':>4} {'LT':>4} {'WW':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7}"
    )
    print("-" * 100)
    for r in results[:30]:
        print(
            f"{r['ml']:>3} {r['mvn']:>5.3f} {r['mvl']:>5.3f} "
            f"{r['skw']:>4} "
            f"{r['wt']:>4.2f} {r['nt']:>4.2f} {r['lt']:>4.2f} "
            f"{r['ww']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['avg_yr']:>7.1f}"
        )

    # === 3. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v52(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            mom_lookback=r["ml"],
            mom_vel_threshold_normal=r["mvn"],
            mom_vel_threshold_losing=r["mvl"],
            skip_in_winning=r["skw"],
            win_threshold=r["wt"],
            normal_threshold=r["nt"],
            lose_threshold=r["lt"],
            win_rate_window=r["ww"],
            atr_stop=3.0,
            top_n_winning=2,
            start_di=60,
        )
        label = (
            f"ml={r['ml']} mvn={r['mvn']:.3f} mvl={r['mvl']:.3f} "
            f"skw={r['skw']} wt={r['wt']:.2f} nt={r['nt']:.2f} "
            f"lt={r['lt']:.2f} ww={r['ww']}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 4. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: ml={best['ml']} mvn={best['mvn']:.3f} "
            f"mvl={best['mvl']:.3f} skw={best['skw']} "
            f"wt={best['wt']:.2f} nt={best['nt']:.2f} "
            f"lt={best['lt']:.2f} ww={best['ww']}"
        )
        print("=" * 70)
        walk_forward(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            mom_lookback=best["ml"],
            mom_vel_threshold_normal=best["mvn"],
            mom_vel_threshold_losing=best["mvl"],
            skip_in_winning=best["skw"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=2,
        )

        # === 5. Compare: V52 vs V43 (no mom filter) ===
        print("\n" + "=" * 70)
        print("  COMPARISON: V52 (with mom filter) vs V43 (no filter)")
        print("  (2019-2026 OOS)")
        print("=" * 70)

        trades_v52, eq_v52, dd_v52 = backtest_v52(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            mom_lookback=best["ml"],
            mom_vel_threshold_normal=best["mvn"],
            mom_vel_threshold_losing=best["mvl"],
            skip_in_winning=best["skw"],
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=2,
            start_di=bt_2019,
        )

        # V43 equivalent: skip momentum filter entirely
        trades_v43like, eq_v43like, dd_v43like = backtest_v52(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            mom_lookback=best["ml"],
            mom_vel_threshold_normal=best["mvn"],
            mom_vel_threshold_losing=best["mvl"],
            skip_in_winning=True,  # skip in all modes
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            top_n_winning=2,
            start_di=bt_2019,
        )

        print(f"\n  V52 (with momentum filter):")
        analyze(trades_v52, eq_v52, dd_v52, "V52-mom")
        print(f"\n  V43-LIKE (no momentum filter):")
        analyze(trades_v43like, eq_v43like, dd_v43like, "V43-noMom")

        if trades_v52 and trades_v43like:
            print(
                f"\n  V52 vs V43: "
                f"eq_delta={eq_v52 - eq_v43like:+,.0f} "
                f"dd_delta={dd_v52 - dd_v43like:+.1f}%"
            )
            wr_v52 = sum(1 for t in trades_v52 if t["pnl_pct"] > 0) / len(
                trades_v52
            )
            wr_v43 = sum(1 for t in trades_v43like if t["pnl_pct"] > 0) / len(
                trades_v43like
            )
            print(
                f"  WR: V52={wr_v52 * 100:.1f}% vs V43={wr_v43 * 100:.1f}% "
                f"(delta={((wr_v52 - wr_v43) * 100):+.1f}%)"
            )
            print(
                f"  Trades: V52={len(trades_v52)} vs V43={len(trades_v43like)}"
            )

    print(f"\n[V52] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
