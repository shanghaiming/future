"""
V15: Hurst Exponent Regime Detection Strategy
==============================================
Uses rolling Hurst exponent (R/S analysis) to detect market regime
and adapt trading behavior:

  H < 0.4 : Strong mean-reversion → maximum MR position size
  0.4-0.6 : Random walk → skip or minimal size
  H > 0.6 : Trending → trend-following OR skip

Three strategy modes:
  MR-only  : Only trade when H < 0.45 (strong MR confirmation)
  MR-enhanced : Trade all, but size proportional to (0.6 - H) when H < 0.6
  Dual     : MR when H<0.45, trend-following when H>0.6

Combines with V1 oversold composite signal, pyramid on day-1 winners,
and walk-forward validation.

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

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005

# Hurst thresholds
HURST_MR_STRONG = 0.40
HURST_MR_SOFT = 0.45
HURST_RANDOM_HI = 0.60
HURST_TREND = 0.60

HURST_WINDOW = 100
HURST_MIN_SUB = 10


# ============================================================
# HURST EXPONENT (R/S Analysis)
# ============================================================
def hurst_rs(prices: np.ndarray, window: int = HURST_WINDOW) -> float:
    """
    Compute Hurst exponent via R/S analysis on a single price series.

    For different sub-interval sizes n:
      1. Split log-returns into blocks of size n
      2. For each block: compute cumulative deviation range R and std S
      3. Average R/S across blocks
      4. Regress log(R/S) vs log(n) → slope = Hurst exponent

    Returns NaN if insufficient data.
    """
    if len(prices) < window:
        return np.nan

    series = prices[-window:]
    returns = np.diff(np.log(series))
    n_total = len(returns)

    min_n = HURST_MIN_SUB
    max_n = n_total // 2
    if min_n >= max_n:
        return np.nan

    num_splits = min(8, max_n - min_n + 1)
    split_sizes = np.unique(np.linspace(min_n, max_n, num_splits).astype(int))

    ns: List[float] = []
    rs_vals: List[float] = []

    for n in split_sizes:
        num_sub = n_total // n
        if num_sub < 1:
            continue
        rs_list: List[float] = []
        for i in range(num_sub):
            sub = returns[i * n:(i + 1) * n]
            mean_sub = np.mean(sub)
            cumdev = np.cumsum(sub - mean_sub)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(sub, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            ns.append(np.log(n))
            rs_vals.append(np.log(np.mean(rs_list)))

    if len(ns) < 2:
        return np.nan

    coeffs = np.polyfit(ns, rs_vals, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


def compute_hurst_matrix(C: np.ndarray, NS: int, ND: int,
                         window: int = HURST_WINDOW) -> np.ndarray:
    """Compute rolling Hurst exponent per instrument."""
    t0 = time.time()
    print("[V15] Computing Hurst exponents...", flush=True)

    hurst = np.full((NS, ND), np.nan)
    for si in range(NS):
        prices = C[si]
        for di in range(window, ND):
            window_prices = prices[di - window:di + 1]
            valid = window_prices[~np.isnan(window_prices)]
            if len(valid) >= window:
                hurst[si, di] = hurst_rs(valid, window)

    h_valid = ~np.isnan(hurst)
    if h_valid.any():
        vals = hurst[h_valid]
        below_mr = (vals < HURST_MR_SOFT).sum()
        above_trend = (vals > HURST_RANDOM_HI).sum()
        print(f"  Hurst: mean={np.mean(vals):.3f}, "
              f"<{HURST_MR_SOFT}={below_mr} ({below_mr/len(vals)*100:.1f}%), "
              f">{HURST_RANDOM_HI}={above_trend} ({above_trend/len(vals)*100:.1f}%), "
              f"{time.time()-t0:.1f}s", flush=True)

    return hurst


# ============================================================
# V1 OVERSOLD SIGNALS (from V5, simplified)
# ============================================================
def compute_oversold_signals(C: np.ndarray, O: np.ndarray,
                             H: np.ndarray, L: np.ndarray,
                             V: np.ndarray, OI: np.ndarray,
                             NS: int, ND: int) -> Dict[str, np.ndarray]:
    """Compute composite oversold score (V1-style)."""
    t0 = time.time()
    print("[V15] Computing oversold signals...", flush=True)

    # Consecutive down days
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                consec = consec + 1 if C[si, di] < C[si, di - 1] else 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # 5-day return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # OI decline signal
    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5])
                    or np.isnan(C[si, di]) or np.isnan(C[si, di - 5])
                    or C[si, di - 5] <= 0):
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_decline[si, di] = (min(abs(oi_chg), 0.2) / 0.2
                                      * min(abs(price_chg), 0.1) / 0.1)
            else:
                oi_decline[si, di] = 0.0

    # RSI 14
    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                rsi = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, rsi)
            except Exception:
                pass

    # Bollinger Band position
    bb_pos = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                upper, mid, lower = talib.BBANDS(c, 20, 2.0, 2.0)
                bb_range = upper - lower
                valid_bb = (~nan_mask) & (bb_range > 1e-10)
                bb_pos[si] = np.where(valid_bb, (c - lower) / bb_range, np.nan)
            except Exception:
                pass

    # CCI 14
    cci14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                cci = talib.CCI(h, l, c, 14)
                cci14[si] = np.where(nan_mask, np.nan, cci)
            except Exception:
                pass

    # Volume delta proxy (simplified VDP)
    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        vdp = np.full(ND, np.nan)
        for di in range(1, ND):
            if (np.isnan(H[si, di]) or np.isnan(L[si, di])
                    or np.isnan(C[si, di]) or np.isnan(V[si, di])):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[di] = (V[si, di]
                           * (2 * C[si, di] - H[si, di] - L[si, di])
                           / bar_range)
        for di in range(10, ND):
            vals = vdp[di - 10:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    # Composite score
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0

            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.20
            w_total += 0.20

            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20

            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20

            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10

            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10

            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05

            if not np.isnan(vdp_10[si, di]):
                if vdp_10[si, di] < -0.3:
                    s += min(-vdp_10[si, di] / 1.0, 1.0) * 0.05
                w_total += 0.05

            if w_total > 0:
                scores[si] = s / w_total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = (pd.Series(scores)
                                .rank(pct=True, na_option='keep').values)

    # Count confirming signals
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
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15:
                n += 1
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100:
                n += 1
            n_signals[si, di] = n

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)

    return {
        'combo_rank': raw_score,
        'n_signals': n_signals,
        'rsi14': rsi14,
    }


# ============================================================
# TREND-FOLLOWING SIGNALS (for dual mode when H > 0.6)
# ============================================================
def compute_trend_signals(C: np.ndarray, H: np.ndarray, L: np.ndarray,
                          NS: int, ND: int) -> Dict[str, np.ndarray]:
    """Compute trend-following signals for trending regime."""
    t0 = time.time()
    print("[V15] Computing trend signals...", flush=True)

    # ADX
    adx = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                adx_val = talib.ADX(h, l, c, 14)
                adx[si] = np.where(nan_mask, np.nan, adx_val)
            except Exception:
                pass

    # EMA 10 / EMA 30 crossover
    ema_fast = np.full((NS, ND), np.nan)
    ema_slow = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        for period, arr in [(10, ema_fast), (30, ema_slow)]:
            mult = 2.0 / (period + 1)
            for di in range(period, ND):
                if np.isnan(c[di]):
                    continue
                window = c[di - period:di + 1]
                valid = window[~np.isnan(window)]
                if len(valid) < period:
                    continue
                ema = float(valid[0])
                for val in valid[1:]:
                    ema = (val - ema) * mult + ema
                arr[si, di] = ema

    # Trend score: positive = uptrend, negative = downtrend
    trend_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(30, ND):
            ef = ema_fast[si, di]
            es = ema_slow[si, di]
            adx_val = adx[si, di]
            if np.isnan(ef) or np.isnan(es) or np.isnan(adx_val):
                continue

            score = 0.0
            if ef > es:
                score += 1.0
            else:
                score -= 1.0

            # ADX strength bonus
            if adx_val > 25:
                score += 0.5 * np.sign(score)
            elif adx_val < 20:
                score *= 0.5

            trend_score[si, di] = score

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)
    return {
        'trend_score': trend_score,
        'adx': adx,
    }


# ============================================================
# BACKTEST — HURST-AWARE
# ============================================================
def backtest_v15(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    sigs: Dict[str, np.ndarray],
    hurst: np.ndarray,
    mode: str = 'mr_only',
    top_n: int = 1,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    hurst_mr_thresh: float = HURST_MR_SOFT,
    start_di: int = 100,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """
    Hurst-aware backtest.

    mode:
      'mr_only'     — only trade when H < hurst_mr_thresh
      'mr_enhanced' — trade all, size proportional to (0.6 - H)
      'dual'        — MR when H<0.45, trend-following when H>0.6
    """
    combo_rank = sigs['combo_rank']
    n_signals = sigs['n_signals']

    has_trend = 'trend_score' in sigs
    if has_trend:
        trend_score = sigs['trend_score']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple] = []

        # --- Exit logic ---
        pos_by_si: Dict[int, List[Tuple]] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr, direction in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr, direction))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr, direction))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi

            # Stop check (direction-aware)
            stopped = False
            for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                if direction > 0 and c < sp:
                    stopped = True
                    break
                if direction < 0 and c > sp:
                    stopped = True
                    break

            if stopped:
                for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                    pnl = direction * (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit,
                        'pnl_pct': pnl * 100,
                        'days': di - edi + 1,
                        'di': di,
                        'year': d.year,
                        'sym': syms[si],
                        'reason': 'stop',
                        'pyr': is_pyr,
                        'dir': direction,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                    pnl = direction * (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit,
                        'pnl_pct': pnl * 100,
                        'days': di - edi + 1,
                        'di': di,
                        'year': d.year,
                        'sym': syms[si],
                        'reason': 'hold',
                        'pyr': is_pyr,
                        'dir': direction,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr, direction))

        # --- Pyramid check ---
        if pyramid_ratio > 0:
            held_with_pos: Dict[int, List[Tuple]] = defaultdict(list)
            for si, edi, ep, sp, alloc, is_pyr, direction in new_positions:
                held_with_pos[si].append((edi, ep, sp, alloc, is_pyr, direction))

            additions: List[Tuple] = []
            for si, pos_list in held_with_pos.items():
                has_pyr = any(p[4] for p in pos_list)
                if has_pyr:
                    continue
                earliest_edi = min(p[0] for p in pos_list)
                hold = di - earliest_edi
                if hold == pyramid_day and not np.isnan(C[si, di]):
                    direction = pos_list[0][5]
                    avg_ep = np.mean([p[1] for p in pos_list])
                    # Pyramid only if position is profitable
                    if direction > 0 and C[si, di] > avg_ep:
                        base_alloc = sum(p[3] for p in pos_list)
                        pyr_alloc = base_alloc * pyramid_ratio
                        c_now = C[si, di]
                        atr_v: List[float] = []
                        for j in range(max(start_di, di - 14), di):
                            hh, ll, cc = H[si, j], L[si, j], C[si, j]
                            if not any(np.isnan([hh, ll, cc])):
                                atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                        if atr_v:
                            atr = np.mean(atr_v)
                            stop = c_now - atr_stop * atr
                            additions.append((si, di, c_now, stop, pyr_alloc, True, 1))
                    elif direction < 0 and C[si, di] < avg_ep:
                        base_alloc = sum(p[3] for p in pos_list)
                        pyr_alloc = base_alloc * pyramid_ratio
                        c_now = C[si, di]
                        atr_v: List[float] = []
                        for j in range(max(start_di, di - 14), di):
                            hh, ll, cc = H[si, j], L[si, j], C[si, j]
                            if not any(np.isnan([hh, ll, cc])):
                                atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                        if atr_v:
                            atr = np.mean(atr_v)
                            stop = c_now + atr_stop * atr
                            additions.append((si, di, c_now, stop, pyr_alloc, True, -1))
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

        # --- Entry logic ---
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        candidates: List[Tuple] = []

        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < min_rank:
                continue
            if n_signals[si, di] < min_confidence:
                continue
            if np.isnan(hurst[si, di]):
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            h_val = hurst[si, di]

            if mode == 'mr_only':
                # Only trade in strong MR regime
                if h_val >= hurst_mr_thresh:
                    continue
                # Size bonus: lower H → bigger size
                size_mult = (hurst_mr_thresh - h_val) / hurst_mr_thresh
                candidates.append((combo_rank[si, di], si, 1, size_mult))

            elif mode == 'mr_enhanced':
                # Trade all, but size proportional to MR strength
                if h_val < HURST_RANDOM_HI:
                    size_mult = max((HURST_RANDOM_HI - h_val) / HURST_RANDOM_HI, 0.1)
                    candidates.append((combo_rank[si, di], si, 1, size_mult))
                # else skip: trending regime → no MR trades

            elif mode == 'dual':
                if h_val < hurst_mr_thresh:
                    # Strong MR regime → buy oversold (direction=+1)
                    size_mult = (hurst_mr_thresh - h_val) / hurst_mr_thresh
                    candidates.append((combo_rank[si, di], si, 1, size_mult))
                elif h_val > HURST_TREND and has_trend:
                    # Trending regime → short overbought
                    ts = trend_score[si, di]
                    if np.isnan(ts):
                        continue
                    if ts < -0.5:
                        # Downtrend confirmed → short (direction=-1)
                        size_mult = (h_val - HURST_TREND) / (1.0 - HURST_TREND)
                        candidates.append((combo_rank[si, di], si, -1, size_mult))
                    elif ts > 0.5:
                        # Uptrend confirmed → long trend-following
                        size_mult = (h_val - HURST_TREND) / (1.0 - HURST_TREND)
                        candidates.append((combo_rank[si, di], si, 1, size_mult))

        candidates.sort(key=lambda x: (-x[2] * x[3], -x[0]))

        for rank, si, direction, size_mult in candidates:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue

            atr_v: List[float] = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)

            base_alloc = 1.0 / max(top_n, 1)
            alloc = base_alloc * size_mult
            alloc = min(alloc, base_alloc * 1.5)  # cap at 1.5x base

            if direction > 0:
                stop = ep - atr_stop * atr
            else:
                stop = ep + atr_stop * atr

            positions.append((si, di + 1, ep, stop, alloc, False, direction))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, is_pyr, direction in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = direction * (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "") -> Optional[Dict]:
    """Print analysis of backtest results."""
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100

    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)

    n_pyr = sum(1 for t in trades if t.get('pyr'))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t['reason'] == 'stop')
    n_hold = sum(1 for t in trades if t['reason'] == 'hold')
    n_long = sum(1 for t in trades if t.get('dir', 1) > 0)
    n_short = len(trades) - n_long

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
          f"stop:{n_stop} hold:{n_hold} L:{n_long} S:{n_short}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")

    yr: Dict[int, dict] = {}
    for t in trades:
        y = t['year']
        if y not in yr:
            yr[y] = {'n': 0, 'w': 0, 'pnl': []}
        yr[y]['n'] += 1
        if t['pnl_pct'] > 0:
            yr[y]['w'] += 1
        yr[y]['pnl'].append(t['pnl_pct'])

    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys['pnl']]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% "
              f"cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd,
            'ann': ann, 'sh': sh, 'eq': equity}


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    sigs: Dict[str, np.ndarray],
    hurst: np.ndarray,
    mode: str = 'mr_only',
    top_n: int = 1,
    pyramid_ratio: float = 0.5,
    hurst_mr_thresh: float = HURST_MR_SOFT,
) -> List[dict]:
    """Walk-forward validation by year."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD mode={mode} pyr={pyramid_ratio} "
          f"h_thresh={hurst_mr_thresh}")
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

        trades, _, _ = backtest_v15(
            C, O, H, L, NS, ND, dates, syms, sigs, hurst,
            mode=mode, top_n=top_n,
            hold_days=5, atr_stop=3.0,
            min_confidence=3,
            pyramid_ratio=pyramid_ratio, pyramid_day=1,
            hurst_mr_thresh=hurst_mr_thresh,
            start_di=test_start, end_di=test_end_idx + 1)

        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr:.1f}% "
                  f"avg={avg:+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get('pyr'))
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr:.1f}% "
              f"avg={avg:+.2f}% cum={cum:+.1%}")

    return all_trades


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V15: HURST EXPONENT REGIME DETECTION — WALK-FORWARD VALIDATED")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to "
          f"{dates[-1].strftime('%Y-%m-%d')}")

    # --- Phase 1: Compute Hurst ---
    hurst = compute_hurst_matrix(C, NS, ND, window=HURST_WINDOW)

    # --- Phase 2: Compute oversold signals ---
    sigs = compute_oversold_signals(C, O, H, L, V, OI, NS, ND)

    # --- Phase 3: Compute trend signals (for dual mode) ---
    trend_sigs = compute_trend_signals(C, H, L, NS, ND)
    sigs.update(trend_sigs)

    # Find 2019 index for later tests
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # ========================================
    # SECTION 1: MODE COMPARISON (full period)
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 1: MODE COMPARISON (2016-2026)")
    print("=" * 70)

    for mode in ['mr_only', 'mr_enhanced', 'dual']:
        for pyr in [0.0, 0.5]:
            label = f"{mode} pyr={pyr:.1f}"
            trades, eq, dd = backtest_v15(
                C, O, H, L, NS, ND, dates, syms, sigs, hurst,
                mode=mode, top_n=1,
                hold_days=5, atr_stop=3.0,
                min_confidence=3,
                pyramid_ratio=pyr, pyramid_day=1,
                start_di=100)
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    # ========================================
    # SECTION 2: VARY HURST THRESHOLD (MR-only)
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 2: HURST THRESHOLD SWEEP (MR-only, 2019-2026)")
    print("=" * 70)

    results = []
    for h_thresh in [0.35, 0.40, 0.45, 0.50, 0.55]:
        for pyr in [0.0, 0.3, 0.5]:
            for mc in [2, 3]:
                trades, eq, dd = backtest_v15(
                    C, O, H, L, NS, ND, dates, syms, sigs, hurst,
                    mode='mr_only', top_n=1,
                    hold_days=5, atr_stop=3.0,
                    min_confidence=mc,
                    pyramid_ratio=pyr, pyramid_day=1,
                    hurst_mr_thresh=h_thresh,
                    start_di=bt_2019)
                if len(trades) < 10:
                    continue
                nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                wr = nw / len(trades) * 100
                n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                rets_arr = np.array(ap) / CASH0
                sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                          if np.std(rets_arr) > 0 else 0)
                results.append({
                    'h': h_thresh, 'pyr': pyr, 'mc': mc,
                    'n': len(trades), 'wr': wr,
                    'ann': ann, 'dd': dd, 'sh': sh_val,
                })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'H':>4} {'Pyr':>4} {'MC':>3} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 55)
    for r in results[:20]:
        print(f"{r['h']:>4.2f} {r['pyr']:>4.1f} {r['mc']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # ========================================
    # SECTION 3: BEST FULL-RUN
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 3: BEST CONFIG FULL 2016-2026")
    print("=" * 70)

    if results:
        best = results[0]
        label = (f"mr_only h={best['h']:.2f} pyr={best['pyr']:.1f} "
                 f"mc={best['mc']}")
        trades, eq, dd = backtest_v15(
            C, O, H, L, NS, ND, dates, syms, sigs, hurst,
            mode='mr_only', top_n=1,
            hold_days=5, atr_stop=3.0,
            min_confidence=best['mc'],
            pyramid_ratio=best['pyr'], pyramid_day=1,
            hurst_mr_thresh=best['h'],
            start_di=100)
        print(f"\n  BEST FULL: {label}")
        analyze(trades, eq, dd, label)

    # ========================================
    # SECTION 4: WALK-FORWARD (top configs)
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 4: WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Walk-forward for top 3 configs
    for r in results[:3]:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs, hurst,
            mode='mr_only', top_n=1,
            pyramid_ratio=r['pyr'],
            hurst_mr_thresh=r['h'])

    # Walk-forward for dual mode
    walk_forward(
        C, O, H, L, NS, ND, dates, syms, sigs, hurst,
        mode='dual', top_n=1,
        pyramid_ratio=0.5,
        hurst_mr_thresh=0.45)

    print(f"\n[V15] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
