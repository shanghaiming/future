"""
V34: RANK MOMENTUM ACCELERATION -- Mean Reversion
==================================================
Core thesis: V18 uses rank LEVEL (rank > 0.80 = oversold). V34 uses rank
ACCELERATION: the second derivative of the composite rank. A commodity whose
oversold rank is accelerating (getting oversold faster and faster) may be
near capitulation. Combine level + velocity + acceleration for timing.

Signal architecture:
  1. Same 7 cross-sectional ranks as V18 (ret5d, oi5d, vol, ret10d, range, rsi, atrp)
  2. Compute rank_velocity = composite_rank[di] - composite_rank[di-5]
     (5-day change in rank -- how fast rank is moving)
  3. Compute rank_acceleration = rank_velocity[di] - rank_velocity[di-5]
     (change in velocity -- is it accelerating?)
  4. Final score = rank_level (0.50) + rank_velocity_rank (0.30) + rank_acceleration_rank (0.20)
  5. This captures "how oversold" AND "how fast getting oversold" AND "is it accelerating"
  6. KER gate, confidence >= 2, hold 5d, ATR stop 3.0
  7. Pyramid on day-1 winners

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage. Walk-forward validation required.
"""
import sys
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict

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

# V18 default weights for the 7 cross-sectional rank factors
DEFAULT_RANK_WEIGHTS: Dict[str, float] = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_rsi':    0.15,
    'rank_vol':    0.15,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_atrp':   0.05,
}

# Default V34 scoring weights
DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    'level':      0.50,
    'velocity':   0.30,
    'acceleration': 0.20,
}


# ============================================================
# RAW FACTOR COMPUTATION (from V18)
# ============================================================
def compute_rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
    """Compute RSI without talib as fallback."""
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
                        valid_l.append(losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(valid_g) >= period:
                    avg_gain = np.mean(valid_g)
                    avg_loss = np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[si, di + period - 1] = 100.0 - 100.0 / (1.0 + rs)
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
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute raw factor values before cross-sectional ranking."""
    t0 = time.time()
    print("[V34] Computing raw factors...", flush=True)

    # --- 5d return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    # --- 10d return ---
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    # --- OI 5d change ---
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # --- Volume (5d average for stability) ---
    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    # --- Daily range (H-L) / C ---
    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(H[si, di]) and not np.isnan(L[si, di]) and not np.isnan(C[si, di]):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (H[si, di] - L[si, di]) / C[si, di]

    # --- RSI 14 ---
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

    # --- ATR% (14d) ---
    atrp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(14, ND):
            atr_vals = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                    atr_vals.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'oi_5d': oi_5d,
        'vol_5d': vol_5d,
        'daily_range': daily_range,
        'rsi14': rsi14,
        'atrp': atrp,
    }


# ============================================================
# CROSS-SECTIONAL RANKS (from V18)
# ============================================================
def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally (across commodities per day).
    Low rank = oversold / extreme for mean reversion."""
    t0 = time.time()
    print("[V34] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        'rank_ret5d': raw_factors['ret_5d'],
        'rank_ret10d': raw_factors['ret_10d'],
        'rank_oi5d': raw_factors['oi_5d'],
        'rank_vol': raw_factors['vol_5d'],
        'rank_range': raw_factors['daily_range'],
        'rank_rsi': raw_factors['rsi14'],
        'rank_atrp': raw_factors['atrp'],
    }

    INVERT_FACTORS = {'rank_ret5d', 'rank_ret10d', 'rank_oi5d', 'rank_rsi'}

    ranks = {}
    for name, factor in factors_to_rank.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < min_count:
                continue
            ranked = pd.Series(vals).rank(pct=True, na_option='keep').values
            if name in INVERT_FACTORS:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


# ============================================================
# KER (from V18)
# ============================================================
def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Kaufman Efficiency Ratio for regime detection."""
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
                ker_regime[si, di] = 1  # sideways -> good for mean reversion
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending -> avoid counter-trend
    return ker_regime


# ============================================================
# COMPOSITE RANK (from V18) + VELOCITY + ACCELERATION (V34 new)
# ============================================================
def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build weighted composite rank from individual factor ranks.
    Also count how many factors confirm (rank > 0.5 for each factor)."""
    t0 = time.time()
    print("[V34] Building composite signal...", flush=True)

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


def compute_rank_dynamics(
    composite: np.ndarray,
    NS: int, ND: int,
    velocity_lag: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rank velocity and acceleration from composite rank.

    velocity[di] = composite[di] - composite[di - velocity_lag]
      (positive = rank is rising = getting more oversold)
    acceleration[di] = velocity[di] - velocity[di - velocity_lag]
      (positive = velocity is increasing = oversold momentum is accelerating)
    """
    t0 = time.time()
    print("[V34] Computing rank velocity & acceleration...", flush=True)

    velocity = np.full((NS, ND), np.nan)
    acceleration = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(2 * velocity_lag, ND):
            rank_now = composite[si, di]
            rank_prev = composite[si, di - velocity_lag]
            if np.isnan(rank_now) or np.isnan(rank_prev):
                continue
            vel = rank_now - rank_prev
            velocity[si, di] = vel

            vel_prev = velocity[si, di - velocity_lag] if di >= 3 * velocity_lag else np.nan
            # Recompute vel_prev directly for accuracy at the boundary
            rank_prev2 = composite[si, di - velocity_lag]
            rank_prev3 = composite[si, di - 2 * velocity_lag]
            if np.isnan(rank_prev2) or np.isnan(rank_prev3):
                continue
            vel_prev2 = rank_prev2 - rank_prev3

            acc = vel - vel_prev2
            acceleration[si, di] = acc

    print(f"  Rank dynamics done: {time.time() - t0:.1f}s", flush=True)
    return velocity, acceleration


def cross_sectional_rank_array(
    arr: np.ndarray,
    NS: int, ND: int,
    min_count: int = 10,
    invert: bool = False,
) -> np.ndarray:
    """Cross-sectionally rank a 2D array (NS x ND) across symbols per day."""
    ranked = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = arr[:, di]
        valid_count = np.sum(~np.isnan(vals))
        if valid_count < min_count:
            continue
        pct = pd.Series(vals).rank(pct=True, na_option='keep').values
        if invert:
            pct = 1.0 - pct
        ranked[:, di] = pct
    return ranked


def build_v34_score(
    composite: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    n_confirm: np.ndarray,
    NS: int, ND: int,
    level_weight: float = 0.50,
    velocity_weight: float = 0.30,
    acceleration_weight: float = 0.20,
    require_positive_velocity: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the V34 final score combining level, velocity rank, and acceleration rank.

    For mean reversion, we want:
      - High level rank (oversold)
      - High velocity (rank is rising fast = getting oversold fast)
      - High acceleration (oversold momentum is accelerating)

    Velocity and acceleration are cross-sectionally ranked so they are
    comparable across commodities with different volatilities.
    """
    t0 = time.time()
    print("[V34] Building V34 final score...", flush=True)

    # Rank velocity cross-sectionally (high velocity = high rank = more oversold momentum)
    vel_rank = cross_sectional_rank_array(velocity, NS, ND, invert=False)

    # Rank acceleration cross-sectionally (high acceleration = high rank = accelerating oversold)
    acc_rank = cross_sectional_rank_array(acceleration, NS, ND, invert=False)

    # Combine: final_score = w_l * level + w_v * vel_rank + w_a * acc_rank
    score = np.full((NS, ND), np.nan)
    total_w = level_weight + velocity_weight + acceleration_weight

    for di in range(ND):
        for si in range(NS):
            lv = composite[si, di]
            vr = vel_rank[si, di]
            ar = acc_rank[si, di]
            if np.isnan(lv):
                continue
            # If velocity rank or acceleration rank is missing, use available
            parts = [lv * level_weight]
            actual_w = level_weight
            if not np.isnan(vr):
                parts.append(vr * velocity_weight)
                actual_w += velocity_weight
            if not np.isnan(ar):
                parts.append(ar * acceleration_weight)
                actual_w += acceleration_weight

            if actual_w > 0:
                score[si, di] = sum(parts) / actual_w

            # If require_positive_velocity, nan out scores where velocity is negative
            if require_positive_velocity:
                vel_raw = velocity[si, di]
                if np.isnan(vel_raw) or vel_raw <= 0:
                    score[si, di] = np.nan

    print(f"  V34 score done: {time.time() - t0:.1f}s", flush=True)
    return score, vel_rank


def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int,
    rank_weights: Optional[Dict[str, float]] = None,
    level_weight: float = 0.50,
    velocity_weight: float = 0.30,
    acceleration_weight: float = 0.20,
    require_positive_velocity: bool = False,
) -> Dict[str, np.ndarray]:
    """Full V34 signal pipeline."""
    if rank_weights is None:
        rank_weights = DEFAULT_RANK_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, rank_weights, NS, ND)

    velocity, acceleration = compute_rank_dynamics(composite, NS, ND, velocity_lag=5)

    v34_score, vel_rank = build_v34_score(
        composite, velocity, acceleration, n_confirm,
        NS, ND,
        level_weight=level_weight,
        velocity_weight=velocity_weight,
        acceleration_weight=acceleration_weight,
        require_positive_velocity=require_positive_velocity,
    )

    return {
        'composite': composite,
        'v34_score': v34_score,
        'velocity': velocity,
        'acceleration': acceleration,
        'vel_rank': vel_rank,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
    }


# ============================================================
# BACKTEST
# ============================================================
def backtest_v34(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    sigs: Dict[str, np.ndarray],
    top_n: int = 1,
    min_rank: float = 0.75,
    atr_stop: float = 3.0,
    min_confidence: int = 2,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[list, float, float]:
    """Backtest V34 with rank momentum acceleration signals + pyramid."""
    v34_score = sigs['v34_score']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']
    composite = sigs['composite']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: list = []
    trades: list = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: list = []

        pos_by_si: Dict[int, list] = defaultdict(list)
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
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'stop', 'pyr': is_pyr,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'hold', 'pyr': is_pyr,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # Pyramid check
        if pyramid_ratio > 0:
            held_with_pos: Dict[int, list] = defaultdict(list)
            for si, edi, ep, sp, alloc, is_pyr in new_positions:
                held_with_pos[si].append((edi, ep, sp, alloc, is_pyr))

            additions: list = []
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
                        pyr_alloc = base_alloc * pyramid_ratio
                        c_now = C[si, di]
                        atr_v = []
                        for j in range(max(start_di, di - 14), di):
                            hh, ll, cc = H[si, j], L[si, j], C[si, j]
                            if not any(np.isnan([hh, ll, cc])):
                                atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                        if atr_v:
                            atr = np.mean(atr_v)
                            additions.append((si, di, c_now, c_now - atr_stop * atr, pyr_alloc, True))
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
        candidates: list = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(v34_score[si, di]):
                continue
            if v34_score[si, di] < min_rank:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(top_n, 1)
            candidates.append((v34_score[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

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
def analyze(
    trades: list, equity: float, max_dd: float, label: str = "",
) -> Optional[Dict]:
    """Analyze backtest results."""
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

    n_pyr = sum(1 for t in trades if t.get('pyr'))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t['reason'] == 'stop')
    n_hold = sum(1 for t in trades if t['reason'] == 'hold')

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")

    yr: Dict = {}
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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    sigs: Dict[str, np.ndarray],
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    top_n: int = 1,
    min_confidence: int = 2,
    hold_days: int = 5,
    atr_stop: float = 3.0,
    min_rank: float = 0.75,
) -> list:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V34 (pyr={pyramid_ratio}, day={pyramid_day})")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades: list = []

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

        trades, _, _ = backtest_v34(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank,
            min_confidence=min_confidence, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            start_di=test_start, end_di=test_end_idx + 1)

        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get('pyr'))
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr_val:.1f}% "
              f"avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V34: RANK MOMENTUM ACCELERATION MEAN REVERSION")
    print("  Level + Velocity + Acceleration of composite rank")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Default signal computation
    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # =================================================================
    # 1. Walk-Forward Validation (default params)
    # =================================================================
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026)")
    print("=" * 70)

    for ratio in [0.0, 0.5]:
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                     pyramid_ratio=ratio, pyramid_day=1,
                     top_n=1, min_confidence=2)

    # =================================================================
    # 2. Full 10-year backtest with pyramid profiles
    # =================================================================
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- PYRAMID PROFILES")
    print("=" * 70)

    profiles = [
        (0.0, 1, "No pyramid (baseline)"),
        (0.5, 1, "Moderate pyramid (50%)"),
    ]

    for ratio, pday, label in profiles:
        trades, eq, dd = backtest_v34(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=2, use_ker_gate=True,
            pyramid_ratio=ratio, pyramid_day=pday,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # =================================================================
    # 3. PARAMETER SWEEP (2019-2026)
    # =================================================================
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: list = []

    for level_w in [0.40, 0.50, 0.60]:
        for vel_w in [0.20, 0.30]:
            acc_w = round(1.0 - level_w - vel_w, 2)
            if acc_w < 0.05:
                continue

            for req_pos_vel in [True, False]:
                sweep_sigs = compute_all_signals(
                    C, O, H, L, V, OI, NS, ND,
                    level_weight=level_w,
                    velocity_weight=vel_w,
                    acceleration_weight=acc_w,
                    require_positive_velocity=req_pos_vel,
                )

                for tn in [1, 2, 3]:
                    for mr in [0.70, 0.75, 0.80]:
                        for ratio in [0.0, 0.5]:
                            for a_stop in [2.5, 3.0]:
                                trades, eq, dd = backtest_v34(
                                    C, O, H, L, NS, ND, dates, syms, sweep_sigs,
                                    top_n=tn, hold_days=5, atr_stop=a_stop,
                                    min_rank=mr, min_confidence=2,
                                    use_ker_gate=True,
                                    pyramid_ratio=ratio, pyramid_day=1,
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
                                    'lw': level_w, 'vw': vel_w, 'aw': acc_w,
                                    'rpv': req_pos_vel,
                                    'tn': tn, 'mr': mr, 'ratio': ratio,
                                    'a_stop': a_stop,
                                    'n': len(trades), 'wr': wr, 'ann': ann,
                                    'dd': dd, 'sharpe': sh_val,
                                })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'LW':>4} {'VW':>4} {'AW':>4} {'RPV':>5} "
          f"{'TN':>3} {'MR':>4} {'Pyr':>4} {'ATR':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 85)
    for r in results[:30]:
        print(f"{r['lw']:>4.2f} {r['vw']:>4.2f} {r['aw']:>4.2f} "
              f"{'Y' if r['rpv'] else 'N':>5} "
              f"{r['tn']:>3} {r['mr']:>4.2f} {r['ratio']:>4.1f} {r['a_stop']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # =================================================================
    # 4. TOP CONFIGS -- FULL 10-YEAR + WALK-FORWARD
    # =================================================================
    if results:
        # Find best unique score weight combos from top results
        seen_weights: set = set()
        top_configs: list = []
        for r in results:
            key = (r['lw'], r['vw'], r['aw'], r['rpv'])
            if key not in seen_weights:
                seen_weights.add(key)
                top_configs.append(r)
            if len(top_configs) >= 3:
                break

        print("\n" + "=" * 70)
        print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
        print("=" * 70)

        for cfg in top_configs[:3]:
            lw, vw, aw, rpv = cfg['lw'], cfg['vw'], cfg['aw'], cfg['rpv']
            cfg_sigs = compute_all_signals(
                C, O, H, L, V, OI, NS, ND,
                level_weight=lw, velocity_weight=vw,
                acceleration_weight=aw,
                require_positive_velocity=rpv,
            )
            # Use the top structural params from sweep for this weight combo
            best_for_weights = [r for r in results
                                if r['lw'] == lw and r['vw'] == vw
                                and r['rpv'] == rpv][:1]
            if not best_for_weights:
                best_for_weights = [cfg]

            b = best_for_weights[0]
            label = (f"lw={lw:.2f} vw={vw:.2f} aw={aw:.2f} rpv={'Y' if rpv else 'N'} "
                     f"tn={b['tn']} mr={b['mr']:.2f} pyr={b['ratio']:.1f} "
                     f"atr={b['a_stop']:.1f}")

            trades, eq, dd = backtest_v34(
                C, O, H, L, NS, ND, dates, syms, cfg_sigs,
                top_n=b['tn'], hold_days=5, atr_stop=b['a_stop'],
                min_rank=b['mr'], min_confidence=2,
                use_ker_gate=True,
                pyramid_ratio=b['ratio'], pyramid_day=1,
                start_di=60)
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

            # Walk-forward for top config
            print(f"\n  WALK-FORWARD {label}")
            walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, cfg_sigs,
                         pyramid_ratio=b['ratio'], pyramid_day=1,
                         top_n=b['tn'], min_confidence=2,
                         hold_days=5, atr_stop=b['a_stop'],
                         min_rank=b['mr'])

    print(f"\n[V34] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
