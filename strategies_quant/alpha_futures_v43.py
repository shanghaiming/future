"""
V43: Adaptive Threshold with Dynamic Extreme Rank Switching
============================================================
Combines V39's adaptive mechanism with V29's extreme rank quality filter.

Core thesis:
  V39 (Sharpe 4.47) uses adaptive thresholds based on recent win rate.
  V29 (Sharpe 8.57) proved extreme oversold (>0.90) produces incredible
  per-trade quality. V43 combines both: dynamically switch between
  standard and extreme thresholds based on recent performance.

Dynamic threshold mode:
  - Track rolling win rate over last N trades (configurable window)
  - WINNING MODE (WR > win_threshold): threshold = relaxed (e.g. 0.75)
    -> More trades when strategy is working well
  - NORMAL MODE (WR between win_threshold and 0.50): threshold = standard
    -> Default quality level
  - LOSING MODE (WR < 0.50): threshold = extreme (e.g. 0.90)
    -> Only highest quality trades (V29-style)

  Each mode also adjusts:
    - top_n: winning=2/3, normal=2, losing=1
    - pyramid_ratio: winning=0.5, normal=0.3, losing=0.0

Signal architecture (same as V18/V29/V39):
  1. Compute V18's 7 cross-sectional ranks, composite score
  2. Dynamic threshold mode based on rolling win rate
  3. KER gate, hold 5d, ATR stop
  4. Pyramid on day-1 winners (ratio varies by mode)

Parameter sweep:
  - win_threshold: 0.55, 0.60, 0.65
  - normal_threshold: 0.80, 0.82, 0.85
  - lose_threshold: 0.88, 0.90, 0.92
  - win_rate_window: 10, 15, 20
  - atr_stop: 2.5, 3.0
  - top_n_winning: 2, 3

Walk-forward 2019-2026, full 10-year for top configs.

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
    print("[V43] Computing raw factors...", flush=True)

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
    print("[V43] Computing cross-sectional ranks...", flush=True)

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
    print("[V43] Building composite signal...", flush=True)

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
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)

    return {
        "composite": composite,
        "n_confirm": n_confirm,
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
    """Determine trading mode based on recent win rate.

    Returns: 'winning', 'normal', or 'losing'
    """
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
    """Get trading parameters for the given mode.

    Returns dict with threshold, top_n, pyramid_ratio, mode_label.
    """
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


def backtest_v43(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
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
    """Backtest V43 with dynamic threshold switching."""
    composite = sigs["composite"]
    ker_regime = sigs["ker_regime"]
    n_confirm = sigs["n_confirm"]

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
            positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

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

    # Mode distribution
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
            f"cum={cum:+.1%}")

    return {
        "n": len(trades), "wr": wr, "dd": max_dd,
        "ann": ann, "sh": sh, "eq": equity,
    }


def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
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
        f"  WALK-FORWARD V43 "
        f"(wt={win_threshold:.2f} nt={normal_threshold:.2f} "
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

        trades, _, _ = backtest_v43(
            C, O, H, L, NS, ND, dates, syms, sigs,
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
        cum = np.prod(
            [1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
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
    print("  V43: ADAPTIVE THRESHOLD + DYNAMIC EXTREME RANK SWITCHING")
    print("  V39 adaptive + V29 extreme quality filter")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}"
    )

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
        (0.60, 0.82, 0.90, 15, 3.0, 2),
        (0.60, 0.80, 0.88, 15, 3.0, 2),
        (0.60, 0.85, 0.92, 15, 3.0, 2),
        (0.55, 0.82, 0.90, 15, 3.0, 2),
        (0.65, 0.82, 0.90, 15, 3.0, 2),
    ]

    for wt, nt, lt, ww, ats, tnw in default_configs:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=ats,
            top_n_winning=tnw,
        )

    # === 2. Full 10-year backtest with profiles ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- PROFILE COMPARISON")
    print("=" * 70)

    profiles = [
        (0.60, 0.82, 0.90, 15, 3.0, 2, "Default (balanced)"),
        (0.55, 0.80, 0.88, 15, 3.0, 2, "Relaxed (more trades)"),
        (0.65, 0.85, 0.92, 15, 3.0, 2, "Tight (quality focus)"),
        (0.60, 0.82, 0.90, 10, 3.0, 2, "Short window (fast adapt)"),
        (0.60, 0.82, 0.90, 20, 3.0, 2, "Long window (slow adapt)"),
        (0.60, 0.82, 0.90, 15, 2.5, 2, "Tight stop (2.5 ATR)"),
        (0.60, 0.82, 0.90, 15, 3.0, 3, "Wide winning (tnw=3)"),
    ]

    for wt, nt, lt, ww, ats, tnw, label in profiles:
        trades, eq, dd = backtest_v43(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=ats,
            top_n_winning=tnw,
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
        "win_threshold": [0.55, 0.60, 0.65],
        "normal_threshold": [0.80, 0.82, 0.85],
        "lose_threshold": [0.88, 0.90, 0.92],
        "win_rate_window": [10, 15, 20],
        "atr_stop": [2.5, 3.0],
        "top_n_winning": [2, 3],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for wt, nt, lt, ww, ats, tnw in product(
        sweep_params["win_threshold"],
        sweep_params["normal_threshold"],
        sweep_params["lose_threshold"],
        sweep_params["win_rate_window"],
        sweep_params["atr_stop"],
        sweep_params["top_n_winning"],
    ):
        # Skip invalid: win_threshold must be > 0.50, lose > normal
        if wt <= 0.50:
            continue
        if lt <= nt:
            continue

        combo_count += 1
        trades, eq, dd = backtest_v43(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=wt,
            normal_threshold=nt,
            lose_threshold=lt,
            win_rate_window=ww,
            atr_stop=ats,
            top_n_winning=tnw,
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

        # Count trades per year for trade frequency check
        yr_counts: Dict[int, int] = {}
        for t in trades:
            y = t["year"]
            yr_counts[y] = yr_counts.get(y, 0) + 1
        oos_years = [y for y in yr_counts if y >= 2019]
        avg_per_year = (sum(yr_counts[y] for y in oos_years)
                        / max(len(oos_years), 1))

        results.append({
            "wt": wt, "nt": nt, "lt": lt,
            "ww": ww, "ats": ats, "tnw": tnw,
            "n": len(trades), "wr": wr, "ann": ann,
            "dd": dd, "sharpe": sh_val, "eq": eq,
            "avg_yr": avg_per_year,
        })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n  Evaluated {combo_count} combos, "
        f"{len(results)} with 10+ trades"
    )
    print(
        f"\n{'WT':>4} {'NT':>4} {'LT':>4} {'WW':>3} "
        f"{'ATS':>4} {'TNW':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7}"
    )
    print("-" * 85)
    for r in results[:30]:
        print(
            f"{r['wt']:>4.2f} {r['nt']:>4.2f} {r['lt']:>4.2f} "
            f"{r['ww']:>3} {r['ats']:>4.1f} {r['tnw']:>4} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['avg_yr']:>7.1f}"
        )

    # === 4. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v43(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=r["wt"],
            normal_threshold=r["nt"],
            lose_threshold=r["lt"],
            win_rate_window=r["ww"],
            atr_stop=r["ats"],
            top_n_winning=r["tnw"],
            start_di=60,
        )
        label = (
            f"wt={r['wt']:.2f} nt={r['nt']:.2f} lt={r['lt']:.2f} "
            f"ww={r['ww']} ats={r['ats']:.1f} tnw={r['tnw']}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 5. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: wt={best['wt']:.2f} nt={best['nt']:.2f} "
            f"lt={best['lt']:.2f} ww={best['ww']} "
            f"ats={best['ats']:.1f} tnw={best['tnw']}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=best["ats"],
            top_n_winning=best["tnw"],
        )

        # === 6. Compare: V43 dynamic vs V39 adaptive vs V29 extreme ===
        print("\n" + "=" * 70)
        print("  COMPARISON: V43 DYNAMIC vs V39 ADAPTIVE vs V29 EXTREME")
        print("  (2019-2026 OOS)")
        print("=" * 70)

        # V43 dynamic (best config)
        trades_v43, eq_v43, dd_v43 = backtest_v43(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=best["ats"],
            top_n_winning=best["tnw"],
            start_di=bt_2019,
        )

        # V39-like static: fixed normal threshold, no dynamic switching
        # (simulate by setting win_threshold very high so mode never changes)
        trades_v39like, eq_v39like, dd_v39like = backtest_v43(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=0.99,  # never enters "winning" mode
            normal_threshold=best["nt"],
            lose_threshold=0.99,  # never enters "losing" mode
            win_rate_window=best["ww"],
            atr_stop=best["ats"],
            top_n_winning=best["tnw"],
            start_di=bt_2019,
        )

        # V29-like extreme: fixed extreme threshold
        trades_v29like, eq_v29like, dd_v29like = backtest_v43(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=0.99,  # never enters "winning" mode
            normal_threshold=0.99,  # always "losing" -> extreme threshold
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=best["ats"],
            top_n_winning=1,
            start_di=bt_2019,
        )

        print(f"\n  V43 DYNAMIC:")
        analyze(trades_v43, eq_v43, dd_v43, "V43-dynamic")
        print(f"\n  V39-LIKE STATIC (fixed {best['nt']:.2f}):")
        analyze(trades_v39like, eq_v39like, dd_v39like, "V39-static")
        print(f"\n  V29-LIKE EXTREME (fixed {best['lt']:.2f}):")
        analyze(trades_v29like, eq_v29like, dd_v29like, "V29-extreme")

        if trades_v43:
            print(
                f"\n  V43 vs V39-static: "
                f"eq_delta={eq_v43 - eq_v39like:+,.0f} "
                f"dd_delta={dd_v43 - dd_v39like:+.1f}%"
            )
            print(
                f"  V43 vs V29-extreme: "
                f"eq_delta={eq_v43 - eq_v29like:+,.0f} "
                f"dd_delta={dd_v43 - dd_v29like:+.1f}%"
            )

    print(f"\n[V43] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
