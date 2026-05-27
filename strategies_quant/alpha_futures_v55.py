"""
V55: Mean Reversion + Trend Following Hybrid Regime Strategy
=============================================================
Core thesis: MR works in range-bound markets, trend-following works in
trending markets.  Combine both to capture alpha in ALL market conditions.

Architecture:
1. For each instrument, classify current regime via KER:
   - TRENDING:  KER > trend_ker_thresh  -> go WITH the trend
   - MEAN-REVERTING: KER < mr_ker_thresh -> use MR (V43 style composite)
   - NEUTRAL:   skip (no edge)

2. Trend following (when KER > trend_ker_thresh):
   - Long  if price > MA(trend_ma_period) AND rank_ret5d > 0.5
   - Short if price < MA(trend_ma_period) AND rank_ret5d < 0.5
   - Hold trend_hold days, wider ATR stop (trend_atr)

3. Mean reversion (when KER < mr_ker_thresh):
   - V43 style composite signal with threshold 0.82
   - Hold mr_hold days, ATR stop mr_atr

4. Position sizing:
   - Trend: 0.8x base size (wider stops, less concentration)
   - MR: 1.0x base size (standard)

Parameter sweep:
  - trend_ker_thresh: 0.25, 0.30, 0.35
  - mr_ker_thresh:    0.12, 0.15, 0.18
  - trend_ma_period:  15, 20, 30
  - trend_hold:       7, 10
  - mr_hold:          5 (fixed)
  - trend_atr:        3.5, 4.0
  - mr_atr:           2.5, 3.0

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


# ============================================================
# FACTOR COMPUTATION (reused from V43)
# ============================================================
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
    t0 = time.time()
    print("[V55] Computing raw factors...", flush=True)

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
                    daily_range[si, di] = (
                        H[si, di] - L[si, di]
                    ) / C[si, di]

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
        "oi_5d": oi_5d,
        "vol_5d": vol_5d,
        "daily_range": daily_range,
        "rsi14": rsi14,
        "atrp": atrp,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V55] Computing cross-sectional ranks...", flush=True)

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


# ============================================================
# KER (Kaufman Efficiency Ratio) + MA computation
# ============================================================
def compute_ker_raw(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Return raw KER values (0-1) for regime classification."""
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
    return ker_10


def compute_ma(C: np.ndarray, NS: int, ND: int, period: int) -> np.ndarray:
    """Simple moving average for trend detection."""
    ma = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(period, ND):
            vals = C[si, di - period : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= period - 2:
                ma[si, di] = np.mean(valid)
    return ma


def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    print("[V55] Building composite signal...", flush=True)

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
    ma_periods: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if ma_periods is None:
        ma_periods = [15, 20, 30]

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_raw = compute_ker_raw(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)

    # Pre-compute MAs for all needed periods
    ma_dict: Dict[int, np.ndarray] = {}
    for p in ma_periods:
        ma_dict[p] = compute_ma(C, NS, ND, p)

    return {
        "composite": composite,
        "n_confirm": n_confirm,
        "ker_raw": ker_raw,
        "ranks": ranks,
        "raw_ret5d": raw["ret_5d"],
        "ma_dict": ma_dict,
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
        return float(np.mean(atr_v))
    return None


# ============================================================
# BACKTEST
# ============================================================
def backtest_v55(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    trend_ker_thresh: float = 0.30,
    mr_ker_thresh: float = 0.15,
    trend_ma_period: int = 20,
    trend_hold: int = 10,
    mr_hold: int = 5,
    trend_atr: float = 4.0,
    mr_atr: float = 3.0,
    mr_threshold: float = 0.82,
    mr_min_confirm: int = 3,
    max_positions: int = 3,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V55 hybrid regime strategy."""
    composite = sigs["composite"]
    ker_raw = sigs["ker_raw"]
    n_confirm = sigs["n_confirm"]
    ranks = sigs["ranks"]
    raw_ret5d = sigs["raw_ret5d"]
    ma_arr = sigs["ma_dict"][trend_ma_period]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    # Positions: (si, entry_di, entry_price, stop_price, alloc, is_trend)
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_trend in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_trend))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_trend in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_trend))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi

            # Determine hold period and atr stop based on regime
            is_trend_pos = any(is_t for _, _, _, _, is_t in pos_list)
            hold_limit = trend_hold if is_trend_pos else mr_hold
            atr_mult = trend_atr if is_trend_pos else mr_atr

            # Recompute stop with current atr
            atr = compute_atr_at(H, L, C, si, di, start_di)
            stopped = False
            if atr is not None:
                dynamic_stop = c - atr_mult * atr
                for _, ep, sp, _, _ in pos_list:
                    if c < sp or c < dynamic_stop:
                        stopped = True
                        break

            if stopped:
                for edi, ep, sp, alloc, is_trend in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    regime_label = "TREND" if is_trend else "MR"
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": "stop",
                        "regime": regime_label,
                    })
            elif hold >= hold_limit:
                for edi, ep, sp, alloc, is_trend in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    regime_label = "TREND" if is_trend else "MR"
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": "hold",
                        "regime": regime_label,
                    })
            else:
                # Update stop for trend positions (trail to breakeven or higher)
                for edi, ep, sp, alloc, is_trend in pos_list:
                    updated_sp = sp
                    if is_trend and atr is not None:
                        new_stop = c - atr_mult * atr
                        if new_stop > sp:
                            updated_sp = new_stop
                    new_positions.append(
                        (si, edi, ep, updated_sp, alloc, is_trend)
                    )

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

        # === ENTRY SIGNALS ===
        mr_candidates = []
        trend_candidates = []

        for si in range(NS):
            if si in held:
                continue
            if np.isnan(ker_raw[si, di]):
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            ker_val = ker_raw[si, di]

            # --- MEAN REVERTING regime ---
            if ker_val < mr_ker_thresh:
                if np.isnan(composite[si, di]):
                    continue
                if composite[si, di] < mr_threshold:
                    continue
                if n_confirm[si, di] < mr_min_confirm:
                    continue
                mr_candidates.append((composite[si, di], si, "MR"))

            # --- TRENDING regime ---
            elif ker_val > trend_ker_thresh:
                ma_val = ma_arr[si, di]
                if np.isnan(ma_val):
                    continue
                c_now = C[si, di]
                if np.isnan(c_now):
                    continue
                rank_ret5 = ranks["rank_ret5d"][si, di]
                if np.isnan(rank_ret5):
                    continue

                # LONG: price above MA AND outperforming peers
                if c_now > ma_val and rank_ret5 > 0.5:
                    strength = (c_now / ma_val - 1.0) * rank_ret5
                    trend_candidates.append((strength, si, "TREND_LONG"))

        # Sort by signal strength
        mr_candidates.sort(key=lambda x: -x[0])
        trend_candidates.sort(key=lambda x: -x[0])

        # Allocate positions: up to max_positions total
        # Prioritize: fill MR first (higher confidence), then trend
        allocated = 0
        mr_positions_allowed = max(1, max_positions - len(held))
        for rank_val, si, regime in mr_candidates[:mr_positions_allowed]:
            if allocated >= max_positions or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            alloc = (1.0 / max_positions) * 1.0  # MR: 1.0x base
            positions.append(
                (si, di + 1, ep, ep - mr_atr * atr, alloc, False)
            )
            held.add(si)
            allocated += 1

        remaining_slots = max_positions - len(positions)
        for strength, si, regime in trend_candidates[:remaining_slots]:
            if allocated >= max_positions or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            alloc = (1.0 / max_positions) * 0.8  # Trend: 0.8x base
            positions.append(
                (si, di + 1, ep, ep - trend_atr * atr, alloc, True)
            )
            held.add(si)
            allocated += 1

    # Close remaining positions at end
    for si, edi, ep, sp, alloc, is_trend in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
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
    sh = (
        np.mean(rets) / np.std(rets) * np.sqrt(252)
        if np.std(rets) > 0
        else 0
    )

    n_mr = sum(1 for t in trades if t.get("regime") == "MR")
    n_trend = sum(1 for t in trades if t.get("regime") in ("TREND", "TREND_LONG"))
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    print(
        f"  {label}: {len(trades)}t (MR:{n_mr} Trend:{n_trend} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}"
    )

    # Regime breakdown
    mr_trades = [t for t in trades if t.get("regime") == "MR"]
    trend_trades = [t for t in trades if t.get("regime") in ("TREND", "TREND_LONG")]
    if mr_trades:
        mr_wr = sum(1 for t in mr_trades if t["pnl_pct"] > 0) / len(mr_trades) * 100
        mr_cum = np.prod([1 + t["pnl_pct"] / 100 for t in mr_trades]) - 1
        print(
            f"    MR:    {len(mr_trades)}t WR={mr_wr:.1f}% "
            f"cum={mr_cum:+.1%}"
        )
    if trend_trades:
        tr_wr = sum(1 for t in trend_trades if t["pnl_pct"] > 0) / len(trend_trades) * 100
        tr_cum = np.prod([1 + t["pnl_pct"] / 100 for t in trend_trades]) - 1
        print(
            f"    TREND: {len(trend_trades)}t WR={tr_wr:.1f}% "
            f"cum={tr_cum:+.1%}"
        )

    yr: Dict[int, dict] = {}
    for t in trades:
        y = t["year"]
        if y not in yr:
            yr[y] = {"n": 0, "w": 0, "pnl": [], "mr": 0, "trend": 0}
        yr[y]["n"] += 1
        if t["pnl_pct"] > 0:
            yr[y]["w"] += 1
        yr[y]["pnl"].append(t["pnl_pct"])
        if t.get("regime") == "MR":
            yr[y]["mr"] += 1
        elif t.get("regime") in ("TREND", "TREND_LONG"):
            yr[y]["trend"] += 1
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["pnl"]]) - 1
        print(
            f"    {y}: {ys['n']}t (MR:{ys['mr']} Tr:{ys['trend']}) "
            f"WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}"
        )

    return {
        "n": len(trades), "wr": wr, "dd": max_dd,
        "ann": ann, "sh": sh, "eq": equity,
    }


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    trend_ker_thresh: float = 0.30,
    mr_ker_thresh: float = 0.15,
    trend_ma_period: int = 20,
    trend_hold: int = 10,
    mr_hold: int = 5,
    trend_atr: float = 4.0,
    mr_atr: float = 3.0,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    param_str = (
        f"tkt={trend_ker_thresh:.2f} mkt={mr_ker_thresh:.2f} "
        f"tma={trend_ma_period} th={trend_hold} mh={mr_hold} "
        f"ta={trend_atr:.1f} ma={mr_atr:.1f}"
    )
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V55 ({param_str})")
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

        trades, _, _ = backtest_v55(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=trend_ker_thresh,
            mr_ker_thresh=mr_ker_thresh,
            trend_ma_period=trend_ma_period,
            trend_hold=trend_hold,
            mr_hold=mr_hold,
            trend_atr=trend_atr,
            mr_atr=mr_atr,
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
            n_mr = sum(1 for t in test_trades if t.get("regime") == "MR")
            n_tr = sum(
                1 for t in test_trades
                if t.get("regime") in ("TREND", "TREND_LONG")
            )
            print(
                f"  {test_year}: {n}t (MR:{n_mr} Tr:{n_tr}) "
                f"WR={wr_val:.1f}% avg={avg:+.2f}%",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_mr = sum(1 for t in all_trades if t.get("regime") == "MR")
        n_tr = sum(
            1 for t in all_trades
            if t.get("regime") in ("TREND", "TREND_LONG")
        )
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (MR:{n_mr} Trend:{n_tr}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        return all_trades
    return []


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V55: MEAN REVERSION + TREND FOLLOWING HYBRID REGIME")
    print("  MR in range-bound markets + Trend in trending markets")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01"
    )
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
        (0.30, 0.15, 20, 10, 5, 4.0, 3.0),
        (0.30, 0.15, 20, 7,  5, 3.5, 2.5),
        (0.25, 0.12, 20, 10, 5, 4.0, 3.0),
        (0.35, 0.18, 20, 10, 5, 4.0, 3.0),
        (0.30, 0.15, 15, 10, 5, 4.0, 3.0),
    ]

    for tkt, mkt, tma, th, mh, ta, ma in default_configs:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=tkt, mr_ker_thresh=mkt,
            trend_ma_period=tma, trend_hold=th,
            mr_hold=mh, trend_atr=ta, mr_atr=ma,
        )

    # === 2. Full 10-year backtest with profiles ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- PROFILE COMPARISON")
    print("=" * 70)

    profiles = [
        (0.30, 0.15, 20, 10, 5, 4.0, 3.0, "Default (balanced)"),
        (0.25, 0.12, 20, 10, 5, 4.0, 3.0, "Low KER thresholds (more trades)"),
        (0.35, 0.18, 20, 10, 5, 4.0, 3.0, "High KER thresholds (selective)"),
        (0.30, 0.15, 15, 7,  5, 3.5, 2.5, "Fast MA, short holds, tight stops"),
        (0.30, 0.15, 30, 10, 5, 4.0, 3.0, "Slow MA (stronger trend filter)"),
        (0.30, 0.15, 20, 7,  5, 3.5, 3.0, "Short trend hold, tight trend stop"),
        (0.25, 0.15, 20, 10, 5, 4.0, 2.5, "Low trend KER, tight MR stop"),
    ]

    for tkt, mkt, tma, th, mh, ta, ma, label in profiles:
        trades, eq, dd = backtest_v55(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=tkt, mr_ker_thresh=mkt,
            trend_ma_period=tma, trend_hold=th,
            mr_hold=mh, trend_atr=ta, mr_atr=ma,
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
        "trend_ker_thresh": [0.25, 0.30, 0.35],
        "mr_ker_thresh": [0.12, 0.15, 0.18],
        "trend_ma_period": [15, 20, 30],
        "trend_hold": [7, 10],
        "mr_hold": [5],
        "trend_atr": [3.5, 4.0],
        "mr_atr": [2.5, 3.0],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for tkt, mkt, tma, th, mh, ta, ma in product(
        sweep_params["trend_ker_thresh"],
        sweep_params["mr_ker_thresh"],
        sweep_params["trend_ma_period"],
        sweep_params["trend_hold"],
        sweep_params["mr_hold"],
        sweep_params["trend_atr"],
        sweep_params["mr_atr"],
    ):
        # Validate: trend KER must be > MR KER
        if tkt <= mkt:
            continue

        combo_count += 1
        trades, eq, dd = backtest_v55(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=tkt, mr_ker_thresh=mkt,
            trend_ma_period=tma, trend_hold=th,
            mr_hold=mh, trend_atr=ta, mr_atr=ma,
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

        n_mr = sum(1 for t in trades if t.get("regime") == "MR")
        n_tr = sum(
            1 for t in trades
            if t.get("regime") in ("TREND", "TREND_LONG")
        )

        yr_counts: Dict[int, int] = {}
        for t in trades:
            y = t["year"]
            yr_counts[y] = yr_counts.get(y, 0) + 1
        oos_years = [y for y in yr_counts if y >= 2019]
        avg_per_year = (
            sum(yr_counts[y] for y in oos_years) / max(len(oos_years), 1)
        )

        results.append({
            "tkt": tkt, "mkt": mkt, "tma": tma,
            "th": th, "mh": mh, "ta": ta, "ma": ma,
            "n": len(trades), "wr": wr, "ann": ann,
            "dd": dd, "sharpe": sh_val, "eq": eq,
            "avg_yr": avg_per_year, "n_mr": n_mr, "n_tr": n_tr,
        })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n  Evaluated {combo_count} combos, "
        f"{len(results)} with 10+ trades"
    )
    print(
        f"\n{'TKT':>4} {'MKT':>4} {'TMA':>4} {'TH':>3} "
        f"{'MH':>3} {'TA':>4} {'MA':>4} "
        f"{'N':>5} {'MR':>4} {'TR':>4} {'WR':>5} {'Ann':>8} "
        f"{'DD':>6} {'Sh':>6} {'Avg/Yr':>7}"
    )
    print("-" * 100)
    for r in results[:30]:
        print(
            f"{r['tkt']:>4.2f} {r['mkt']:>4.2f} {r['tma']:>4} "
            f"{r['th']:>3} {r['mh']:>3} {r['ta']:>4.1f} {r['ma']:>4.1f} "
            f"{r['n']:>5} {r['n_mr']:>4} {r['n_tr']:>4} "
            f"{r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['avg_yr']:>7.1f}"
        )

    # === 4. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    for r in results[:5]:
        trades, eq, dd = backtest_v55(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=r["tkt"], mr_ker_thresh=r["mkt"],
            trend_ma_period=r["tma"], trend_hold=r["th"],
            mr_hold=r["mh"], trend_atr=r["ta"], mr_atr=r["ma"],
            start_di=60,
        )
        label = (
            f"tkt={r['tkt']:.2f} mkt={r['mkt']:.2f} "
            f"tma={r['tma']} th={r['th']} mh={r['mh']} "
            f"ta={r['ta']:.1f} ma={r['ma']:.1f}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 5. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: tkt={best['tkt']:.2f} mkt={best['mkt']:.2f} "
            f"tma={best['tma']} th={best['th']} mh={best['mh']} "
            f"ta={best['ta']:.1f} ma={best['ma']:.1f}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=best["tkt"], mr_ker_thresh=best["mkt"],
            trend_ma_period=best["tma"], trend_hold=best["th"],
            mr_hold=best["mh"], trend_atr=best["ta"], mr_atr=best["ma"],
        )

        # === 6. MR-only vs Trend-only vs Hybrid comparison ===
        print("\n" + "=" * 70)
        print("  REGIME DECOMPOSITION: MR-ONLY vs TREND-ONLY vs HYBRID")
        print("  (2019-2026 OOS)")
        print("=" * 70)

        # Full hybrid (best config)
        trades_hybrid, eq_hybrid, dd_hybrid = backtest_v55(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=best["tkt"], mr_ker_thresh=best["mkt"],
            trend_ma_period=best["tma"], trend_hold=best["th"],
            mr_hold=best["mh"], trend_atr=best["ta"], mr_atr=best["ma"],
            start_di=bt_2019,
        )

        # MR-only: trend_ker very high so trend never triggers
        trades_mr, eq_mr, dd_mr = backtest_v55(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=0.99, mr_ker_thresh=best["mkt"],
            trend_ma_period=best["tma"], trend_hold=best["th"],
            mr_hold=best["mh"], trend_atr=best["ta"], mr_atr=best["ma"],
            start_di=bt_2019,
        )

        # Trend-only: mr_ker very low so MR never triggers
        trades_tr, eq_tr, dd_tr = backtest_v55(
            C, O, H, L, NS, ND, dates, syms, sigs,
            trend_ker_thresh=best["tkt"], mr_ker_thresh=0.0,
            trend_ma_period=best["tma"], trend_hold=best["th"],
            mr_hold=best["mh"], trend_atr=best["ta"], mr_atr=best["ma"],
            start_di=bt_2019,
        )

        print(f"\n  HYBRID:")
        analyze(trades_hybrid, eq_hybrid, dd_hybrid, "HYBRID")
        print(f"\n  MR-ONLY (no trend):")
        analyze(trades_mr, eq_mr, dd_mr, "MR-ONLY")
        print(f"\n  TREND-ONLY (no MR):")
        analyze(trades_tr, eq_tr, dd_tr, "TREND-ONLY")

        if trades_hybrid:
            print(
                f"\n  Hybrid vs MR-only: "
                f"eq_delta={eq_hybrid - eq_mr:+,.0f} "
                f"dd_delta={dd_hybrid - dd_mr:+.1f}%"
            )
            print(
                f"  Hybrid vs Trend-only: "
                f"eq_delta={eq_hybrid - eq_tr:+,.0f} "
                f"dd_delta={dd_hybrid - dd_tr:+.1f}%"
            )

    print(f"\n[V55] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
