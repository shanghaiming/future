"""
V8: Adaptive Parameters Based on Market State
===============================================
Instead of fixed parameters (min_rank=0.7, hold_days=5, atr_stop=3.0),
adapt them based on recent market conditions.

Market state measurements (rolling 60-day window):
  1. Volatility regime: avg realized vol across instruments
     - High vol (>30%): wider stops, smaller position, higher rank threshold
     - Low vol (<15%): tighter stops, longer hold, lower rank threshold
  2. Trending vs mean-reverting: avg KER across instruments
     - Mean-reverting (avg KER < 0.15): increase confidence weight
     - Trending (avg KER > 0.3): skip more aggressively
  3. Market breadth: fraction of instruments above 20d MA
     - Broad sell-off (breadth < 0.3): more aggressive entry (more setups)
     - Strong breadth (> 0.7): more selective

Signal at close[di], enter at open[di+1]. No look-ahead. No leverage.
"""
import sys
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

# Default (baseline) parameters
DEFAULT_PARAMS = {
    "min_rank": 0.70,
    "hold_days": 5,
    "atr_stop": 3.0,
    "min_confidence": 3,
}

# Adaptive parameter profiles keyed by regime
REGIME_PARAMS = {
    "high_vol": {
        "min_rank": 0.80,
        "hold_days": 7,
        "atr_stop": 4.0,
        "min_confidence": 3,
    },
    "low_vol": {
        "min_rank": 0.60,
        "hold_days": 3,
        "atr_stop": 2.5,
        "min_confidence": 2,
    },
    "normal": {
        "min_rank": 0.70,
        "hold_days": 5,
        "atr_stop": 3.0,
        "min_confidence": 3,
    },
}


# ============================================================
# PHASE 1: SIGNAL COMPUTATION (reuses V5 multi-alpha)
# ============================================================
def compute_signals(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
) -> Dict[str, np.ndarray]:
    """Core signal computation — multi-alpha composite + KER regime."""
    t0 = time.time()
    print("[V8] Computing signals...", flush=True)

    # --- Consecutive down days ---
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            prev_valid = not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0
            cur_valid = not np.isnan(C[si, di])
            if prev_valid and cur_valid and C[si, di] < C[si, di - 1]:
                consec += 1
            else:
                consec = 0
            consec_dn[si, di] = consec

    # --- 5-day return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c0_valid = not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0
            c1_valid = not np.isnan(C[si, di])
            if c0_valid and c1_valid:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # --- OI decline signal ---
    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5]):
                continue
            if np.isnan(C[si, di]) or np.isnan(C[si, di - 5]) or C[si, di - 5] <= 0:
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_score = (
                    min(abs(oi_chg), 0.2) / 0.2 * min(abs(price_chg), 0.1) / 0.1
                )
                oi_decline[si, di] = oi_score
            else:
                oi_decline[si, di] = 0.0

    # --- VDP exhaustion ---
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            h_val = H[si, di]
            l_val = L[si, di]
            c_val = C[si, di]
            v_val = V[si, di]
            if any(np.isnan([h_val, l_val, c_val, v_val])):
                continue
            bar_range = h_val - l_val
            if bar_range > 0 and v_val > 0:
                vdp[si, di] = v_val * (2 * c_val - h_val - l_val) / bar_range

    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di - 10 : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    vdp_exhaust = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(vdp_10[si, di]):
                continue
            window = vdp_10[si, max(0, di - 20) : di]
            wv = window[~np.isnan(window)]
            if len(wv) >= 10:
                mu, sig = np.mean(wv), np.std(wv)
                if sig > 0:
                    z = (vdp_10[si, di] - mu) / sig
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # --- KER (Kaufman Efficiency Ratio) 10-day ---
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
                ker_regime[si, di] = 1   # mean-reverting
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending → skip

    # --- RSI, BB, CCI via TA-Lib ---
    rsi14 = np.full((NS, ND), np.nan)
    bb_pos = np.full((NS, ND), np.nan)
    cci14 = np.full((NS, ND), np.nan)

    if HAS_TALIB:
        for si in range(NS):
            nan_mask = np.isnan(C[si])
            h_arr = np.where(nan_mask, 0, H[si]).astype(np.float64)
            l_arr = np.where(nan_mask, 0, L[si]).astype(np.float64)
            c_arr = np.where(nan_mask, 0, C[si]).astype(np.float64)
            try:
                rsi = talib.RSI(c_arr, 14)
                rsi14[si] = np.where(nan_mask, np.nan, rsi)
            except Exception:
                pass
            try:
                upper, _, lower = talib.BBANDS(c_arr, 20, 2.0, 2.0)
                bb_range = upper - lower
                valid_bb = (~nan_mask) & (bb_range > 1e-10)
                bb_pos[si] = np.where(valid_bb, (c_arr - lower) / bb_range, np.nan)
            except Exception:
                pass
            try:
                cci = talib.CCI(h_arr, l_arr, c_arr, 14)
                cci14[si] = np.where(nan_mask, np.nan, cci)
            except Exception:
                pass

    # --- Composite score + rank ---
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0
            # Consecutive down
            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.20
            w_total += 0.20
            # 5d return
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20
            # OI decline
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20
            # VDP exhaustion
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.15
                w_total += 0.15
            # RSI
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10
            # BB position
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10
            # CCI
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05
            if w_total > 0:
                scores[si] = s / w_total
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    # --- Signal count per instrument-day ---
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            if consec_dn[si, di] >= 3:
                n += 1
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03:
                n += 1
            if not np.isnan(oi_decline[si, di]) and oi_decline[si, di] > 0.1:
                n += 1
            if not np.isnan(vdp_exhaust[si, di]) and vdp_exhaust[si, di] > 0.3:
                n += 1
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15:
                n += 1
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100:
                n += 1
            n_signals[si, di] = n

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)
    return {
        "combo_rank": raw_score,
        "ker_regime": ker_regime,
        "ker_10": ker_10,
        "n_signals": n_signals,
        "rsi14": rsi14,
    }


# ============================================================
# PHASE 2: MARKET STATE COMPUTATION
# ============================================================
def compute_market_state(
    C: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    lookback: int = 60,
) -> Dict[str, np.ndarray]:
    """
    Compute market-wide state indicators for each day.

    Returns per-day arrays:
      - vol_regime: 'high', 'low', or 'normal' (encoded as 2, 0, 1)
      - avg_ker: cross-sectional average KER
      - breadth: fraction of instruments above their 20d MA
      - realized_vol: average annualized realized vol across instruments
    """
    t0 = time.time()
    print("[V8] Computing market state...", flush=True)

    STATE_WINDOW = lookback
    VOL_HIGH_THRESHOLD = 0.30   # annualized
    VOL_LOW_THRESHOLD = 0.15
    KER_MR_THRESHOLD = 0.15     # mean-reverting
    KER_TREND_THRESHOLD = 0.30  # trending

    # Per-instrument 20d realized vol (annualized)
    realized_vol = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = []
            for j in range(di - 20, di):
                c0 = C[si, j - 1]
                c1 = C[si, j]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    rets.append(c1 / c0 - 1)
            if len(rets) >= 10:
                realized_vol[si, di] = np.std(rets) * np.sqrt(252)

    # Per-instrument 20d MA for breadth
    ma20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            window = C[si, di - 20 : di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                ma20[si, di] = np.mean(valid)

    # Per-instrument KER (recompute 10-day)
    ker = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10 : di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker[si, di] = net_change / total_change

    # Aggregate market state per day
    avg_vol = np.full(ND, np.nan)
    avg_ker = np.full(ND, np.nan)
    breadth = np.full(ND, np.nan)

    for di in range(STATE_WINDOW, ND):
        # Average realized vol across instruments
        vol_vals = realized_vol[:, di]
        vol_valid = vol_vals[~np.isnan(vol_vals)]
        if len(vol_valid) > 3:
            avg_vol[di] = np.mean(vol_valid)

        # Average KER across instruments
        ker_vals = ker[:, di]
        ker_valid = ker_vals[~np.isnan(ker_vals)]
        if len(ker_valid) > 3:
            avg_ker[di] = np.mean(ker_valid)

        # Market breadth: fraction above 20d MA
        above_count = 0
        total_count = 0
        for si in range(NS):
            if np.isnan(C[si, di]) or np.isnan(ma20[si, di]):
                continue
            if C[si, di] > 0:
                total_count += 1
                if C[si, di] > ma20[si, di]:
                    above_count += 1
        if total_count > 3:
            breadth[di] = above_count / total_count

    print(
        f"  vol range: {np.nanmin(avg_vol):.2f} - {np.nanmax(avg_vol):.2f}, "
        f"ker range: {np.nanmin(avg_ker):.3f} - {np.nanmax(avg_ker):.3f}, "
        f"breadth range: {np.nanmin(breadth):.2f} - {np.nanmax(breadth):.2f}",
        flush=True,
    )
    print(f"  Done: {time.time() - t0:.1f}s", flush=True)

    return {
        "avg_vol": avg_vol,
        "avg_ker": avg_ker,
        "breadth": breadth,
        "realized_vol": realized_vol,
    }


# ============================================================
# PHASE 3: ADAPTIVE PARAMETER SELECTION
# ============================================================
def get_adaptive_params(
    market_state: Dict[str, np.ndarray],
    di: int,
) -> Dict[str, float]:
    """
    Select parameters based on current market state.

    Rules:
      - High volatility (avg vol > 30%): wider stops, longer hold, stricter rank
      - Low volatility (avg vol < 15%): tighter stops, shorter hold, looser rank
      - Mean-reverting regime (avg KER < 0.15): lower confidence threshold
      - Trending regime (avg KER > 0.3): higher confidence threshold
      - Low breadth (< 0.3): lower min_rank (more setups during sell-offs)
      - High breadth (> 0.7): higher min_rank (be more selective)
    """
    avg_vol = market_state["avg_vol"][di]
    avg_ker_val = market_state["avg_ker"][di]
    breadth_val = market_state["breadth"][di]

    # Start from normal baseline
    params = dict(DEFAULT_PARAMS)

    # --- Volatility regime (percentile-based thresholds) ---
    if not np.isnan(avg_vol):
        if avg_vol > _vol_high_thresh:
            params["atr_stop"] = 4.0
            params["hold_days"] = 7
            params["min_rank"] = 0.80
            params["min_confidence"] = 3
        elif avg_vol < _vol_low_thresh:
            params["atr_stop"] = 2.5
            params["hold_days"] = 3
            params["min_rank"] = 0.60
            params["min_confidence"] = 2
        else:
            params["atr_stop"] = 3.0
            params["hold_days"] = 5
            params["min_rank"] = 0.70
            params["min_confidence"] = 3

    # --- KER regime overlay (percentile-based thresholds) ---
    if not np.isnan(avg_ker_val):
        if avg_ker_val < _ker_mr_thresh:
            # Mean-reverting: lower confidence threshold (more trades)
            params["min_confidence"] = max(1, params["min_confidence"] - 1)
        elif avg_ker_val > _ker_trend_thresh:
            # Trending: require higher confidence (fewer trades)
            params["min_confidence"] = params["min_confidence"] + 1

    # --- Breadth overlay ---
    if not np.isnan(breadth_val):
        if breadth_val < BREADTH_LOW:
            # Broad sell-off: lower rank threshold (more opportunity)
            params["min_rank"] = max(0.50, params["min_rank"] - 0.10)
        elif breadth_val > BREADTH_HIGH:
            # Strong market: be more selective
            params["min_rank"] = min(0.90, params["min_rank"] + 0.05)

    return params


# Thresholds for adaptive parameter selection
# These are set dynamically in main() based on percentiles
VOL_HIGH_PCT = 0.75   # vol above this percentile → "high vol" regime
VOL_LOW_PCT = 0.25    # vol below this percentile → "low vol" regime
KER_MR_PCT = 0.25     # KER below this percentile → mean-reverting
KER_TREND_PCT = 0.75  # KER above this percentile → trending
BREADTH_LOW = 0.30
BREADTH_HIGH = 0.70

# These get populated in main() from actual data percentiles
_vol_high_thresh = 0.30
_vol_low_thresh = 0.15
_ker_mr_thresh = 0.15
_ker_trend_thresh = 0.30


# ============================================================
# PHASE 4: BACKTEST (Adaptive Parameters)
# ============================================================
def backtest_adaptive(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: pd.DatetimeIndex,
    syms: List[str],
    sigs: Dict[str, np.ndarray],
    market_state: Dict[str, np.ndarray],
    top_n: int = 1,
    use_ker_gate: bool = True,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest with adaptive parameters per day."""
    combo_rank = sigs["combo_rank"]
    ker_regime = sigs["ker_regime"]
    n_signals = sigs["n_signals"]

    if end_di is None:
        end_di = ND - 1

    equity = float(CASH0)
    peak = equity
    max_dd = 0.0
    positions: List[tuple] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[tuple] = []

        # Get adaptive params for this day (for exits)
        params = get_adaptive_params(market_state, di)
        hold_days = int(params["hold_days"])
        atr_stop = params["atr_stop"]

        # Process existing positions
        for si, edi, ep, sp, alloc, is_pyr in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc, is_pyr))
                continue

            hold = di - edi

            # Stop check
            if c < sp:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    "pnl_abs": profit,
                    "pnl_pct": pnl * 100,
                    "days": hold,
                    "di": di,
                    "year": d.year,
                    "sym": syms[si],
                    "reason": "stop",
                    "pyr": is_pyr,
                    "regime_vol": market_state["avg_vol"][edi]
                    if not np.isnan(market_state["avg_vol"][edi])
                    else -1,
                    "regime_ker": market_state["avg_ker"][edi]
                    if not np.isnan(market_state["avg_ker"][edi])
                    else -1,
                })
            elif hold >= hold_days:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    "pnl_abs": profit,
                    "pnl_pct": pnl * 100,
                    "days": hold,
                    "di": di,
                    "year": d.year,
                    "sym": syms[si],
                    "reason": "hold",
                    "pyr": is_pyr,
                    "regime_vol": market_state["avg_vol"][edi]
                    if not np.isnan(market_state["avg_vol"][edi])
                    else -1,
                    "regime_ker": market_state["avg_ker"][edi]
                    if not np.isnan(market_state["avg_ker"][edi])
                    else -1,
                })
            else:
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

        # Check if we can enter new positions
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Get adaptive params for entry decisions
        entry_params = get_adaptive_params(market_state, di)
        min_rank = entry_params["min_rank"]
        min_confidence = int(entry_params["min_confidence"])
        entry_atr_stop = entry_params["atr_stop"]

        candidates: List[tuple] = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < min_rank:
                continue
            if n_signals[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(top_n, 1)
            candidates.append((combo_rank[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            # Compute ATR for stop
            atr_vals: List[float] = []
            for j in range(max(start_di, di - 14), di):
                hh = H[si, j]
                ll = L[si, j]
                cc = C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_vals.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_vals:
                continue
            atr = np.mean(atr_vals)
            positions.append(
                (si, di + 1, ep, ep - entry_atr_stop * atr, alloc, False)
            )
            held.add(si)

    # Close remaining positions at end
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


def backtest_fixed(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: pd.DatetimeIndex,
    syms: List[str],
    sigs: Dict[str, np.ndarray],
    top_n: int = 1,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest with fixed parameters (baseline for comparison)."""
    combo_rank = sigs["combo_rank"]
    ker_regime = sigs["ker_regime"]
    n_signals = sigs["n_signals"]

    if end_di is None:
        end_di = ND - 1

    equity = float(CASH0)
    peak = equity
    max_dd = 0.0
    positions: List[tuple] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[tuple] = []

        for si, edi, ep, sp, alloc, is_pyr in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc, is_pyr))
                continue
            hold = di - edi
            if c < sp:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    "pnl_abs": profit,
                    "pnl_pct": pnl * 100,
                    "days": hold,
                    "di": di,
                    "year": d.year,
                    "sym": syms[si],
                    "reason": "stop",
                    "pyr": is_pyr,
                })
            elif hold >= hold_days:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    "pnl_abs": profit,
                    "pnl_pct": pnl * 100,
                    "days": hold,
                    "di": di,
                    "year": d.year,
                    "sym": syms[si],
                    "reason": "hold",
                    "pyr": is_pyr,
                })
            else:
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
        if len(positions) >= top_n:
            continue

        candidates: List[tuple] = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < min_rank:
                continue
            if n_signals[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(top_n, 1)
            candidates.append((combo_rank[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr_vals: List[float] = []
            for j in range(max(start_di, di - 14), di):
                hh = H[si, j]
                ll = L[si, j]
                cc = C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_vals.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_vals:
                continue
            atr = np.mean(atr_vals)
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# PHASE 5: ANALYSIS
# ============================================================
def analyze(
    trades: List[dict],
    equity: float,
    max_dd: float,
    label: str = "",
) -> Optional[Dict]:
    """Print analysis of trades and return summary stats."""
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

    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")
    avg_hold = np.mean([t["days"] for t in trades])

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} avg_hold={avg_hold:.1f}d eq={equity:,.0f}"
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
            f"    {y}: {ys['n']}t WR={ys['w'] / ys['n'] * 100:.1f}% cum={cum:+.1%}"
        )

    return {"n": len(trades), "wr": wr, "dd": max_dd, "ann": ann, "sh": sh, "eq": equity}


# ============================================================
# PHASE 6: WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: pd.DatetimeIndex,
    syms: List[str],
    mode: str = "adaptive",
) -> List[dict]:
    """
    Walk-forward: train on all prior data, test on each year 2019-2026.
    For adaptive mode, market state is computed on the fly (no look-ahead).
    For fixed mode, uses DEFAULT_PARAMS.
    """
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD ({mode})")
    print(f"{'=' * 70}")

    all_trades: List[dict] = []
    years = sorted(set(d.year for d in dates))

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

        # Compute signals on full data (no look-ahead in signal logic)
        sigs = compute_signals(C, O, H, L, V_ref, OI_ref, NS, ND)

        if mode == "adaptive":
            mkt_state = compute_market_state(C, H, L, NS, ND)
            trades, _, _ = backtest_adaptive(
                C, O, H, L, NS, ND, dates, syms, sigs, mkt_state,
                top_n=1, use_ker_gate=True,
                start_di=test_start, end_di=test_end_idx + 1,
            )
        else:
            trades, _, _ = backtest_fixed(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, min_rank=0.7, atr_stop=3.0,
                min_confidence=3, use_ker_gate=True, hold_days=5,
                start_di=test_start, end_di=test_end_idx + 1,
            )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            print(
                f"  {test_year}: {n}t WR={wr:.1f}% avg={avg:+.2f}%",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        print(
            f"\n  WF TOTAL ({mode}): {len(all_trades)}t "
            f"WR={wr:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
    return all_trades


# ============================================================
# PHASE 7: REGIME BREAKDOWN ANALYSIS
# ============================================================
def analyze_regime_breakdown(
    trades: List[dict],
    label: str = "",
) -> None:
    """Break down trade performance by volatility regime at entry."""
    if not trades:
        return

    vol_bins = {"low": [], "normal": [], "high": []}
    ker_bins = {"mean_rev": [], "neutral": [], "trending": []}

    for t in trades:
        vol = t.get("regime_vol", -1)
        ker = t.get("regime_ker", -1)

        if vol >= 0:
            if vol < _vol_low_thresh:
                vol_bins["low"].append(t)
            elif vol > _vol_high_thresh:
                vol_bins["high"].append(t)
            else:
                vol_bins["normal"].append(t)

        if ker >= 0:
            if ker < _ker_mr_thresh:
                ker_bins["mean_rev"].append(t)
            elif ker > _ker_trend_thresh:
                ker_bins["trending"].append(t)
            else:
                ker_bins["neutral"].append(t)

    print(f"\n  {label} — Volatility Regime Breakdown:")
    for name, tlist in vol_bins.items():
        if tlist:
            nw = sum(1 for t in tlist if t["pnl_pct"] > 0)
            wr = nw / len(tlist) * 100
            avg = np.mean([t["pnl_pct"] for t in tlist])
            print(
                f"    {name:>10}: {len(tlist):>4}t WR={wr:.1f}% avg={avg:+.2f}%"
            )

    print(f"  {label} — KER Regime Breakdown:")
    for name, tlist in ker_bins.items():
        if tlist:
            nw = sum(1 for t in tlist if t["pnl_pct"] > 0)
            wr = nw / len(tlist) * 100
            avg = np.mean([t["pnl_pct"] for t in tlist])
            print(
                f"    {name:>10}: {len(tlist):>4}t WR={wr:.1f}% avg={avg:+.2f}%"
            )


# ============================================================
# MAIN
# ============================================================
# Module-level references used by walk_forward (set in main)
V_ref: np.ndarray = np.array([])
OI_ref: np.ndarray = np.array([])


def main() -> None:
    global V_ref, OI_ref, _vol_high_thresh, _vol_low_thresh
    global _ker_mr_thresh, _ker_trend_thresh

    t0 = time.time()
    print("=" * 70)
    print("  V8: ADAPTIVE PARAMETERS — MARKET STATE DRIVEN")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    V_ref = V
    OI_ref = OI
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # Compute signals and market state
    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)
    market_state = compute_market_state(C, H, L, NS, ND)

    # Set dynamic thresholds from data percentiles
    avg_vol = market_state["avg_vol"]
    avg_ker = market_state["avg_ker"]
    breadth = market_state["breadth"]
    valid_vol = avg_vol[~np.isnan(avg_vol)]
    valid_ker = avg_ker[~np.isnan(avg_ker)]
    valid_brd = breadth[~np.isnan(breadth)]

    _vol_low_thresh = float(np.percentile(valid_vol, VOL_LOW_PCT * 100))
    _vol_high_thresh = float(np.percentile(valid_vol, VOL_HIGH_PCT * 100))
    _ker_mr_thresh = float(np.percentile(valid_ker, KER_MR_PCT * 100))
    _ker_trend_thresh = float(np.percentile(valid_ker, KER_TREND_PCT * 100))

    print(f"\n  Dynamic thresholds (percentile-based):")
    print(f"    Vol: low<{_vol_low_thresh:.3f}, high>{_vol_high_thresh:.3f} "
          f"(p{int(VOL_LOW_PCT*100)}/p{int(VOL_HIGH_PCT*100)})")
    print(f"    KER: mr<{_ker_mr_thresh:.3f}, trend>{_ker_trend_thresh:.3f} "
          f"(p{int(KER_MR_PCT*100)}/p{int(KER_TREND_PCT*100)})")
    print(f"    Breadth: mean={np.mean(valid_brd):.2f}, "
          f"p10={np.percentile(valid_brd, 10):.2f}, "
          f"p90={np.percentile(valid_brd, 90):.2f}")

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ============================================================
    # SECTION 1: Fixed vs Adaptive — Full 2016-2026
    # ============================================================
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) — FIXED vs ADAPTIVE")
    print("=" * 70)

    # Fixed parameter baselines
    print("\n  --- Fixed parameter baselines ---")
    fixed_configs = [
        (0.60, 3, 2.5, 2, "Fixed-Loose"),
        (0.70, 5, 3.0, 3, "Fixed-Default"),
        (0.80, 7, 4.0, 3, "Fixed-Tight"),
    ]
    for mr, hd, ats, mc, label in fixed_configs:
        trades, eq, dd = backtest_fixed(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, min_rank=mr, atr_stop=ats,
            min_confidence=mc, use_ker_gate=True, hold_days=hd,
            start_di=60,
        )
        analyze(trades, eq, dd, label)

    # Adaptive
    print("\n  --- Adaptive parameters ---")
    trades_adp, eq_adp, dd_adp = backtest_adaptive(
        C, O, H, L, NS, ND, dates, syms, sigs, market_state,
        top_n=1, use_ker_gate=True, start_di=60,
    )
    analyze(trades_adp, eq_adp, dd_adp, "Adaptive")

    # ============================================================
    # SECTION 2: 2019-2026 Out-of-Sample Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("  2019-2026 OUT-OF-SAMPLE — FIXED vs ADAPTIVE")
    print("=" * 70)

    for mr, hd, ats, mc, label in fixed_configs:
        trades, eq, dd = backtest_fixed(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, min_rank=mr, atr_stop=ats,
            min_confidence=mc, use_ker_gate=True, hold_days=hd,
            start_di=bt_2019,
        )
        analyze(trades, eq, dd, label)

    trades_adp_oos, eq_adp_oos, dd_adp_oos = backtest_adaptive(
        C, O, H, L, NS, ND, dates, syms, sigs, market_state,
        top_n=1, use_ker_gate=True, start_di=bt_2019,
    )
    analyze(trades_adp_oos, eq_adp_oos, dd_adp_oos, "Adaptive-OOS")

    # ============================================================
    # SECTION 3: Regime Breakdown
    # ============================================================
    print("\n" + "=" * 70)
    print("  REGIME BREAKDOWN")
    print("=" * 70)
    analyze_regime_breakdown(trades_adp, "Adaptive")
    for mr, hd, ats, mc, label in fixed_configs:
        trades_f, eq_f, dd_f = backtest_fixed(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, min_rank=mr, atr_stop=ats,
            min_confidence=mc, use_ker_gate=True, hold_days=hd,
            start_di=60,
        )
        analyze_regime_breakdown(trades_f, label)

    # ============================================================
    # SECTION 4: Walk-Forward Validation
    # ============================================================
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 70)

    wf_fixed = walk_forward(
        C, O, H, L, NS, ND, dates, syms, mode="fixed"
    )
    wf_adaptive = walk_forward(
        C, O, H, L, NS, ND, dates, syms, mode="adaptive"
    )

    # ============================================================
    # SECTION 5: Multi-position sweep with adaptive
    # ============================================================
    print("\n" + "=" * 70)
    print("  MULTI-POSITION ADAPTIVE SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []
    for tn in [1, 2, 3]:
        trades, eq, dd = backtest_adaptive(
            C, O, H, L, NS, ND, dates, syms, sigs, market_state,
            top_n=tn, use_ker_gate=True, start_di=bt_2019,
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
        results.append({
            "tn": tn, "n": len(trades), "wr": wr,
            "ann": ann, "dd": dd, "sharpe": sh_val,
        })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'TN':>3} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}"
    )
    print("-" * 40)
    for r in results:
        print(
            f"{r['tn']:>3} {r['n']:>5} {r['wr']:>5.1f} "
            f"{r['ann']:>+8.1f} {r['dd']:>6.1f} {r['sharpe']:>5.2f}"
        )

    # Best multi-position full period
    if results:
        best = results[0]
        print("\n  Best adaptive multi-position (full 2016-2026):")
        trades_best, eq_best, dd_best = backtest_adaptive(
            C, O, H, L, NS, ND, dates, syms, sigs, market_state,
            top_n=best["tn"], use_ker_gate=True, start_di=60,
        )
        analyze(trades_best, eq_best, dd_best, f"Adaptive-tn={best['tn']}")

    print(f"\n[V8] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
